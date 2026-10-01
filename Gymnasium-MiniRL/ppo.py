import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from actor_critic_gae import gae
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


def select_action(obs, net: ActorCriticNetwork):
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    logits, _ = net(obs_tensor)
    dist = Categorical(logits=logits)
    action = dist.sample()
    return int(action.squeeze(0).item())


@torch.no_grad()
def collect_episode(net: ActorCriticNetwork, env: gym.Env):
    terminated_buffer = []
    rewards = []
    observations = []
    actions = []
    obs, _ = env.reset()
    while True:
        action = select_action(obs, net)
        actions.append(action)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(obs.copy())
        rewards.append(reward)
        terminated_buffer.append(terminated)
        obs = next_obs
        if terminated or truncated:
            break
    # warning: we need last next_obs to compute bootstrap
    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminated": terminated_buffer,
        "final_next_obs": next_obs.copy(),
        "final_terminated": bool(terminated),
        "final_truncated": bool(truncated),
    }


def compute_bootstrap(episode: dict, net: ActorCriticNetwork):
    if episode["final_terminated"]:
        return torch.tensor(0.0, dtype=torch.float32)
    else:
        final_obs = torch.as_tensor(
            episode["final_next_obs"], dtype=torch.float32
        ).unsqueeze(0)
        _, values = net(final_obs)
        return values.squeeze(0)


@torch.no_grad()
def prepare_batch(
    episodes: list[dict], net: ActorCriticNetwork, gamma: float, gae_lambda: float
):
    """
    compute gae, old_logprob before update
    notice avoiding old_logprob's gradient

    return:
        a combined episode with length of 1024
    """
    observations_buffer = []
    actions_buffer = []
    old_log_probs_buffer = []
    advantages_buffer = []
    returns_buffer = []

    for episode in episodes:
        observations = torch.as_tensor(
            np.asarray(episode["observations"]), dtype=torch.float32
        )
        actions = torch.as_tensor(np.asarray(episode["actions"]), dtype=torch.long)

        logits, values = net(observations)
        dist = Categorical(logits=logits)
        old_log_probs = dist.log_prob(actions)
        bootstrap_value = compute_bootstrap(episode, net)
        advantages, returns = gae(
            episode["rewards"],
            values,
            episode["terminated"],
            bootstrap_value,
            gamma,
            gae_lambda,
        )
        observations_buffer.append(observations)
        actions_buffer.append(actions)
        old_log_probs_buffer.append(old_log_probs)
        advantages_buffer.append(advantages)
        returns_buffer.append(returns)

    advantages = torch.cat(advantages_buffer)
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
    return {
        "observations": torch.cat(observations_buffer),  # [B, obs_dim]
        "actions": torch.cat(actions_buffer),  # [B]
        "old_log_probs": torch.cat(old_log_probs_buffer),  # [B]
        "advantages": advantages,  # [B]
        "returns": torch.cat(returns_buffer),  # [B]
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
    learning_rate = 1e-3
    rollout_step = 1024
    num_iterations = 300
    update_per_iter = 4

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
    env.reset(seed=seed)
    env.action_space.seed(seed)

    eval_env = make_cartpole_env(max_steps=200)
    eval_env.reset(seed=seed + 10_000)
    # ac-network
    net = ActorCriticNetwork(action_dim, obs_dim)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    total_steps = 0
    # collector
    for iteration in range(num_iterations):
        training_process = iteration / num_iterations
        current_lr = learning_rate * (1 - training_process)
        for param in optimizer.param_groups:
            param["lr"] = current_lr
        episodes = []
        collected_steps = 0
        while collected_steps < rollout_step:
            episode = collect_episode(net, env)
            episodes.append(episode)
            collected_steps += len(episode["rewards"])
        total_steps += collected_steps

        # compute and freeze training target of this batch
        batch = prepare_batch(episodes, net, gamma, gae_lambda)

        for epoch in range(update_per_iter):
            stats = update_actor_critic(
                net, batch, optimizer, value_coef, entropy_coef, epsilon
            )
        if iteration == 0 or (iteration + 1) % 10 == 0:
            eval_stats = evaluate(net, eval_env, num_episodes=50)
            mean_return = np.mean([sum(episode["rewards"]) for episode in episodes])
            train_success_rate = np.mean(
                [
                    episode["final_truncated"] and not episode["final_terminated"]
                    for episode in episodes
                ]
            )
            print(
                f"Iteration {iteration + 1:3d} | "
                f"steps={total_steps:6d} | "
                f"return={mean_return:7.2f} | "
                f"train_success={train_success_rate:.1%} | "
                f"actor={stats['actor_loss']:+.4f} | "
                f"critic={stats['critic_loss']:.4f} | "
                f"entropy={stats['entropy']:.4f} | "
                f"clip_frac={stats['clip_fraction']:.3f} | "
                f"eval_success={eval_stats['success_rate']:.1%} | "
                f"return={eval_stats['mean_return']:.2f} | "
                f"std={eval_stats['std_return']:.2f}"
            )


if __name__ == "__main__":
    main()
