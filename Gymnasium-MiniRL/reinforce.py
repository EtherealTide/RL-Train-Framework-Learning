import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from actor_critic_gae import gae
from minimal_env import make_cartpole_env
from torch import nn, optim
from torch.distributions import Categorical


# 1. actor-critic network
class ActorCriticNetwork(nn.Module):
    """
    an Actor-Critic network with shared backbone

    input:
        observation, shape=[batch_size,obs_dim]
    output:
        actor_logits, shape=[batch_size,action_dim]
        value, shape=[batch_size]
        notice that there is no softmax in the end
    """

    def __init__(self, obs_dim=1, action_dim=2, critic_dim=1, hidden_dim=64):
        super().__init__()
        # backbone is usually used to extract features
        self.backbone = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh())
        # actor tower
        self.actor_tower = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        # critic tower
        self.critic_tower = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())

        # actor head
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        # critic head
        self.critic_head = nn.Linear(hidden_dim, critic_dim)

        # initialize
        self._initialize_weights()

    def _initialize_weights(self):
        """
        正交初始化Orthogonal initialization是on-policy强化学习的常见初始化选择

        隐藏层采用sqrt(2)的增益
        actor网络的输出层使用小的增益确保接近均匀分布
        critic网络的输出层使用增益1
        """
        for module in (self.backbone, self.actor_tower, self.critic_tower):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.0))
                    nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, obs):
        features = self.backbone(obs)
        actor_features = self.actor_tower(features)
        critic_features = self.critic_tower(features)
        logits = self.actor_head(actor_features)
        value = self.critic_head(critic_features).squeeze(-1)  # [batch,1]->[batch]
        return logits, value


@torch.no_grad()
def select_action(model: ActorCriticNetwork, obs):
    """
    Sample an action during rollout

    Rollout does not construct an autograd graph. Log Prob, values, and
    entropy will be computed in the update phase.
    """
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    logits, _ = model(obs_tensor)
    distribution = Categorical(logits=logits)
    action = distribution.sample()
    return int(action.item())


def update_actor_critic(
    optimizer: torch.optim,
    model: ActorCriticNetwork,
    trajectories: list[dict],
    gamma=0.95,
    gae_lambda=0.95,
    value_coef=0.1,
    entropy_coef=0.001,
):
    """
    Total loss:

    actor_loss
    + value_coef * Huber(values, value_targets)
    - entropy_coef * entropy
    input:
        optimizer: Adam or SGD
        model: Actor-Critic network, used in gradient clipping
        trajectory: data collected in an episode
        bootstrap_value:
        gamma: decrease factor
        gae_lambda
        value_coef
        entropy_coef
    output:
        loss, advantages, grad
    """
    episode_lengths = [len(trajectory["rewards"]) for trajectory in trajectories]
    observations = torch.as_tensor(
        np.concatenate(
            [np.asarray(trajectory["observations"]) for trajectory in trajectories],
            axis=0,
        ),
        dtype=torch.float32,
    )
    actions = torch.as_tensor(
        np.concatenate(
            [np.asarray(trajectory["actions"]) for trajectory in trajectories],
            axis=0,
        ),
        dtype=torch.long,
    )
    # Run the complete multi-episode batch through the network once
    logits, values = model(observations)
    distribution = Categorical(logits=logits)
    log_probs_tensor = distribution.log_prob(actions)
    entropies = distribution.entropy()
    # GAE must be calculated independently inside every episode
    advantages_buffer = []
    returns_buffer = []
    start_index = 0
    for trajectory, episode_length in zip(trajectories, episode_lengths):
        end_index = start_index + episode_length
        episode_values = values[start_index:end_index]
        bootstrap_value = compute_bootstrap_value(model, trajectory)
        episode_advantages, episode_returns = gae(
            trajectory["rewards"],
            episode_values,
            trajectory["terminated"],
            bootstrap_value,
            gamma,
            gae_lambda,
        )
        advantages_buffer.append(episode_advantages)
        returns_buffer.append(episode_returns)
        start_index = end_index
    advantages = torch.cat(advantages_buffer)
    returns = torch.cat(returns_buffer)

    # advantage normalization
    if len(advantages) > 1:
        advantages_normalized = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
    else:
        advantages_normalized = advantages
    actor_loss = -(log_probs_tensor * advantages_normalized.detach()).mean()
    critic_loss = F.smooth_l1_loss(values, returns.detach(), beta=0.5)
    entropy = entropies.mean()
    total_loss = actor_loss + value_coef * critic_loss - entropy * entropy_coef

    # update
    optimizer.zero_grad()
    total_loss.backward()
    # gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()
    return {
        "loss": total_loss.item(),
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "entropy": entropy.item(),
        "adv_mean": (advantages.mean().item()),
        "adv_std": (advantages.std(unbiased=False).item()),
        "return_mean": (returns.mean().item()),
        "value_mean": (values.detach().mean().item()),
        "grad_norm": float(grad_norm),
    }


# collect one trajectory
def collect_episode(env: gym.Env, model: ActorCriticNetwork):
    obs, _ = env.reset()
    observations = []
    actions = []
    rewards = []
    terminated_buffer = []

    final_terminated = False
    final_truncated = False
    final_next_obs = None

    # first collect trajectory
    while True:
        action = select_action(model, obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(obs.copy())
        actions.append(action)
        rewards.append(float(reward))
        terminated_buffer.append(bool(terminated))
        obs = next_obs
        if terminated or truncated:
            final_terminated = terminated
            final_truncated = truncated
            final_next_obs = next_obs
            break

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminated": terminated_buffer,
        "final_next_obs": final_next_obs,
        "final_terminated": final_terminated,
        "final_truncated": final_truncated,
    }


def compute_bootstrap_value(model: ActorCriticNetwork, trajectory: dict):
    """
    compute V(T) after last state of trajectory
    If terminated, future return = 0, Else if truncated, future return != 0
    So if truncated, we shoud use critic network to calculate bookstrap
    """
    if trajectory["final_terminated"]:  # if is terminated, there is no bootstrap
        return torch.tensor(0.0, dtype=torch.float32)
    with torch.no_grad():
        obs_tensor = torch.as_tensor(
            trajectory["final_next_obs"], dtype=torch.float32
        ).unsqueeze(0)
        _, next_value = model(obs_tensor)
    return next_value.squeeze(0)


def evaluate(env: gym.Env, model: ActorCriticNetwork, num_episodes=100):
    """
    Evaluate the policy using deterministic greedy actions.

    For CartPole:
        terminated means failure;
        truncated without termination means surviving until the time limit.
    """
    success_count = 0
    total_reward = 0.0
    total_steps = 0
    with torch.no_grad():
        for _ in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_steps = 0
            while True:
                obs_tensor = torch.as_tensor(
                    obs,
                    dtype=torch.float32,
                ).unsqueeze(0)
                logits, _ = model(obs_tensor)
                action = torch.argmax(
                    logits,
                    dim=-1,
                ).item()
                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = env.step(action)
                episode_reward += reward
                episode_steps += 1
                if not terminated and truncated:
                    success_count += 1
                if terminated or truncated:
                    break

            total_reward += episode_reward
            total_steps += episode_steps

    return {
        "success_rate": (success_count / num_episodes),
        "avg_reward": (total_reward / num_episodes),
        "avg_steps": (total_steps / num_episodes),
    }


def main():
    # hyperparameters
    num_episodes = 2000
    gamma = 0.95
    gae_lambda = 0.95
    value_coef = 0.1
    entropy_coef = 0.001
    learning_rate = 1e-3
    episodes_per_update = 4
    report_every = 100
    eval_every = 200

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_cartpole_env(max_steps=200)
    eval_env = make_cartpole_env(max_steps=200)

    env.reset(seed=seed)
    env.action_space.seed(seed)

    eval_env.reset(seed=seed + 10_000)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = ActorCriticNetwork(obs_dim, action_dim, critic_dim=1, hidden_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    recent_returns = []
    successes = []
    recent_returns = []
    recent_lengths = []
    successes = []
    episodes_seen = 0
    update_index = 0
    while episodes_seen < num_episodes:
        previous_episode_count = episodes_seen
        training_progress = episodes_seen / num_episodes
        current_learning_rate = learning_rate * (1.0 - training_progress)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_learning_rate
        batch_size = min(episodes_per_update, num_episodes - episodes_seen)
        trajectories = [collect_episode(env, model) for _ in range(batch_size)]
        stats = update_actor_critic(
            optimizer,
            model,
            trajectories,
            gamma,
            gae_lambda,
            value_coef,
            entropy_coef,
        )
        update_index += 1
        episodes_seen += batch_size

        batch_returns = [sum(trajectory["rewards"]) for trajectory in trajectories]

        batch_lengths = [len(trajectory["rewards"]) for trajectory in trajectories]

        batch_successes = [
            int(trajectory["final_truncated"] and not trajectory["final_terminated"])
            for trajectory in trajectories
        ]

        recent_returns.extend(batch_returns)
        recent_lengths.extend(batch_lengths)
        successes.extend(batch_successes)

        crossed_report_boundary = (
            episodes_seen == num_episodes
            or episodes_seen // report_every != previous_episode_count // report_every
        )

        if crossed_report_boundary:
            print(
                f"Update {update_index:3d} | "
                f"episodes={episodes_seen:4d} | "
                f"lr={current_learning_rate:.6f} | "
                f"train_return={np.mean(recent_returns[-100:]):7.2f} | "
                f"train_length={np.mean(recent_lengths[-100:]):6.2f} | "
                f"train_success={np.mean(successes[-100:]):.3f} | "
                f"actor={stats['actor_loss']:+.4f} | "
                f"critic={stats['critic_loss']:.4f} | "
                f"entropy={stats['entropy']:.4f} | "
                f"adv_std={stats['adv_std']:.3f} | "
                f"grad={stats['grad_norm']:.3f}"
            )

        crossed_eval_boundary = (
            episodes_seen == num_episodes
            or episodes_seen // eval_every != previous_episode_count // eval_every
        )

        if crossed_eval_boundary:
            eval_stats = evaluate(
                eval_env,
                model,
                num_episodes=100,
            )

            print(
                f"[Eval] episodes={episodes_seen:4d} | "
                f"success_rate={eval_stats['success_rate']:.3f} | "
                f"avg_reward={eval_stats['avg_reward']:.2f} | "
                f"avg_steps={eval_stats['avg_steps']:.2f}"
            )

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
