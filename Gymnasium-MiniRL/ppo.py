import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from minimal_env import make_cartpole_env
from torch import nn
from torch.distributions import Categorical


class ActorCriticNetwork(nn.Module):
    def __init__(self, action_dim=2, obs_dim=4, value_dim=1, hidden_dim=64):
        super().__init__()
        self.shared_backbone = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh())
        self.actor_tower = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_tower = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.critic_head = nn.Linear(hidden_dim, value_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        # orthogonal initialization
        for module in (self.shared_backbone, self.actor_tower, self.critic_tower):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, np.sqrt(2))
                    nn.init.zeros_(layer.bias)

        nn.init.orthogonal_(self.actor_head.weight, 0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, 1)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, obs: torch.tensor):
        features = self.shared_backbone(obs)
        actor_features = self.actor_tower(features)
        logits = self.actor_head(actor_features)
        critic_features = self.critic_tower(features)
        value = self.critic_head(critic_features)  # tensor: [batch,value_dim]
        return logits, value.squeeze(-1)


@torch.no_grad()
def collect_rollout(
    net: ActorCriticNetwork,
    env: gym.Env,
    obs,
    episode_return: float,
    rollout_steps: int,
):
    """
    fixed length rollout
    output:
        rollout: complete rollout with fixed 1024 steps
        obs: last state of current rollout, it will be re-passed in function(collect_rollout) next rollout
        episode_returns: total rewards of one episode, it will be re-passed in function(collect_rollout) next rollout
        completed_episodes: used to statistics performance
    """
    obs_dim = env.observation_space.shape[0]
    # initialize and pre-allocate
    rollout = {
        "observations": torch.empty((rollout_steps, obs_dim), dtype=torch.float32),
        "actions": torch.empty(rollout_steps, dtype=torch.long),
        "rewards": torch.empty(rollout_steps),
        "old_log_probs": torch.empty(rollout_steps),
        "old_values": torch.empty(rollout_steps),
        "next_values": torch.empty(rollout_steps),
        "terminated": torch.empty(rollout_steps, dtype=torch.bool),
        "truncated": torch.empty(rollout_steps, dtype=torch.bool),
    }
    completed_episodes = []
    for t in range(rollout_steps):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, value = net(obs_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        rollout["observations"][t] = obs_tensor.squeeze(0)
        rollout["old_values"][t] = value.squeeze(0).item()
        rollout["old_log_probs"][t] = log_prob.squeeze(0).item()
        rollout["actions"][t] = action.squeeze(0).item()
        rollout["rewards"][t] = float(reward)
        rollout["terminated"][t] = terminated
        rollout["truncated"][t] = truncated

        # compute probability bootstrap
        if terminated:
            rollout["next_values"][t] = 0.0
        elif truncated or t == rollout_steps - 1:
            next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(
                0
            )
            _, next_value = net(next_obs_tensor)
            rollout["next_values"][t] = next_value.item()

        episode_return += float(reward)
        if terminated or truncated:
            completed_episodes.append(
                {
                    "return": episode_return,
                    "success": bool(truncated and not terminated),
                }
            )
            obs, _ = env.reset()
            episode_return = 0.0
        else:
            obs = next_obs

    for t in range(rollout_steps - 1):
        episode_ended = rollout["terminated"][t] or rollout["truncated"][t]
        if not episode_ended:
            rollout["next_values"][t] = rollout["old_values"][t + 1]

    return rollout, obs, episode_return, completed_episodes


@torch.no_grad()
def compute_rollout_gae(rollout, gamma, gae_lambda):
    rewards = rollout["rewards"]
    values = rollout["old_values"]
    next_values = rollout["next_values"]
    terminated = rollout["terminated"]
    truncated = rollout["truncated"]
    advantages = torch.zeros_like(rewards)
    returns = []
    next_advantage = 0
    for t in reversed(range(len(rewards))):
        delta = (
            rewards[t] + gamma * next_values[t] * (1 - int(terminated[t])) - values[t]
        )
        done = int(terminated[t] or truncated[t])
        advantage = delta + gamma * gae_lambda * (1 - done) * next_advantage
        advantages[t] = advantage
        next_advantage = advantage
    returns = advantages + values
    return advantages, returns


@torch.no_grad()
def prepare_batch(rollout, gamma, gae_lambda):
    advantages, returns = compute_rollout_gae(rollout, gamma, gae_lambda)
    return {
        "observations": rollout["observations"],
        "actions": rollout["actions"],
        "old_log_probs": rollout["old_log_probs"],
        "old_values": rollout["old_values"],
        "advantages": advantages,
        "returns": returns,
    }


def update_actor_critic(
    net: ActorCriticNetwork,
    batch: dict,
    optimizer: torch.optim,
    value_coef: float,
    entropy_coef: float,
    epsilon: float,
):
    # forward with new policy network
    logits, values = net(batch["observations"])
    dist = Categorical(logits=logits)
    log_probs = dist.log_prob(batch["actions"])
    entropies = dist.entropy()
    # compute ratio
    ratio = torch.exp(log_probs - batch["old_log_probs"])
    # compute loss
    advantages = batch["advantages"]
    raw_adv_std = advantages.std(unbiased=False).item()
    # mini-batch normalization
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
    actor_loss = -torch.min(
        torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages, (ratio * advantages)
    ).mean()
    critic_loss = F.smooth_l1_loss(values, batch["returns"], beta=0.5)
    entropy = entropies.mean()
    total_loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

    # update
    optimizer.zero_grad()
    total_loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
    optimizer.step()
    with torch.no_grad():
        clip_fraction = ((ratio - 1.0).abs() > epsilon).float().mean()
    return {
        "loss": total_loss.item(),
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "entropy": entropy.item(),
        "ratio_mean": ratio.mean().item(),
        "clip_fraction": clip_fraction.item(),
        "grad_norm": float(grad_norm),
        "raw_adv_std": float(raw_adv_std),
    }


def update_ppo(
    net,
    batch,
    optimizer,
    update_epochs,
    minibatch_size,
    value_coef,
    entropy_coef,
    epsilon,
):
    batch_size = batch["actions"].shape[0]
    assert batch_size % minibatch_size == 0
    all_stats = []
    for epoch in range(update_epochs):
        # shuffle
        indices = torch.randperm(batch_size)
        for start in range(0, batch_size, minibatch_size):
            minibatch_indices = indices[start : start + minibatch_size]
            minibatch = {key: value[minibatch_indices] for key, value in batch.items()}
            stats = update_actor_critic(
                net, minibatch, optimizer, value_coef, entropy_coef, epsilon
            )
            all_stats.append(stats)
    return {
        key: float(np.mean([stats[key] for stats in all_stats])) for key in all_stats[0]
    }


@torch.no_grad()
def evaluate(net: ActorCriticNetwork, env: gym.Env, num_episodes=50):
    episode_returns = []
    success_count = 0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_return = 0.0

        while True:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

            logits, _ = net(obs_tensor)
            action = logits.argmax(dim=-1).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)

            if terminated or truncated:
                success_count += int(truncated and not terminated)
                episode_returns.append(episode_return)
                break

    return {
        "success_rate": success_count / num_episodes,
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
    }


def main():
    # hyper-parameters
    learning_rate = 3e-4
    rollout_steps = 1024
    num_iterations = 300
    update_epochs = 4
    minibatch_size = 256

    gamma = 0.99
    gae_lambda = 0.95
    value_coef = 0.1
    entropy_coef = 0.001
    epsilon = 0.2
    seed = 42

    np.random.seed(seed)
    torch.manual_seed(seed)
    # env
    env = make_cartpole_env(max_steps=200)
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    obs, _ = env.reset(seed=seed)  # save observation in first reset
    env.action_space.seed(seed)

    eval_env = make_cartpole_env(max_steps=200)
    eval_env.reset(seed=seed + 10_000)
    # ac-network
    net = ActorCriticNetwork(action_dim, obs_dim)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    total_steps = 0
    episode_return = 0.0
    # collector
    for iteration in range(num_iterations):
        training_process = iteration / num_iterations
        current_lr = learning_rate * (1 - training_process)
        for param in optimizer.param_groups:
            param["lr"] = current_lr
        # compute values and old_log_probs and freeze
        rollout, obs, episode_return, completed = collect_rollout(
            net, env, obs, episode_return, rollout_steps
        )
        total_steps += rollout_steps
        # compute GAE and freeze
        batch = prepare_batch(rollout, gamma, gae_lambda)

        # update ppo
        stats = update_ppo(
            net,
            batch,
            optimizer,
            update_epochs,
            minibatch_size,
            value_coef,
            entropy_coef,
            epsilon,
        )
        if iteration == 0 or (iteration + 1) % 10 == 0:
            eval_stats = evaluate(net, eval_env, num_episodes=50)

            # 统计本 rollout 内结束的完整 episodes
            if completed:
                mean_return = np.mean([episode["return"] for episode in completed])
                success_rate = np.mean([episode["success"] for episode in completed])
            else:
                mean_return = float("nan")
                success_rate = float("nan")

            print(
                f"Iteration {iteration + 1:3d} | "
                f"steps={total_steps:6d} | "
                f"lr={current_lr:.2e} | "
                f"train_return={mean_return:.2f} | "
                f"train_success={success_rate:.1%} | "
                f"actor={stats['actor_loss']:+.4f} | "
                f"critic={stats['critic_loss']:.4f} | "
                f"entropy={stats['entropy']:.4f} | "
                f"clip_frac={stats['clip_fraction']:.3f} | "
                f"raw_adv_std={stats['raw_adv_std']:.3f} | "
                f"eval_success={eval_stats['success_rate']:.1%} | "
                f"eval_return={eval_stats['mean_return']:.2f}"
            )


if __name__ == "__main__":
    main()
