import random

import numpy as np
import torch
from actor_critic_gae import gae
from minimal_env import LineWorldEnv
from torch import nn, optim
from torch.distributions import Categorical


# 1. actor-critic network
class ActorCriticNetwork(nn.Module):
    """
    an Actor-Critic network with shared backbone

    input:
        observation, shape=[batch_size,1]
    output:
        actor_logits, shape=[batch_size,2]
        value, shape=[batch_size]
        notice that there is no softmax in the end
    """

    def __init__(self, obs_dim=1, action_dim=2, critic_dim=1, hidden_dim=64):
        super().__init__()
        # backbone is usually used to extract features
        self.backbone = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh())
        # actor head
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        # critic head
        self.critic_head = nn.Linear(hidden_dim, critic_dim)

    def forward(self, obs):
        features = self.backbone(obs)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)  # [batch,1]->[batch]
        return logits, value


# choose action from output of actor network, estimate value from output of critic network
def select_action(model: ActorCriticNetwork, obs):
    """
    input:
        model: actor-critic network
        obs: numpy returned by Gymnasium  type:numpy array requested by gym

    output:
        action: sample an action under probability distribution of action space
        log_prob: the probability of chosen action
        value: the estimated value of current state output by critic network
        entropy
    """
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
    # increase dimension of obs_tensor using unsqueeze because neural network usually accept:
    # [batch_size, feature_dim]
    obs_tensor = obs_tensor.unsqueeze(0)

    logits, value = model(obs_tensor)
    dist = Categorical(logits=logits)
    action_tensor = dist.sample()
    log_prob = dist.log_prob(action_tensor)
    entropy = dist.entropy()  # 熵
    action = (
        action_tensor.item()
    )  # action is only sent to Gym without participating upgrade of network
    log_prob = log_prob.squeeze(0)
    return action, log_prob, value.squeeze(0), entropy.squeeze(0)


def update_actor_critic(
    optimizer: torch.optim,
    model: ActorCriticNetwork,
    trajectory: dict,
    bootstrap_value,
    gamma=0.95,
    gae_lambda=0.95,
    value_coef=0.5,
    entropy_coef=0.01,
):
    """
    Loss Equation: loss=actor_loss+value_coef*critic_loss-entropy_coef*entropy
                       =sum(A*log_prob)+value_coef*||returns-values||-entropy_coef*entropy(distribution)
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
    log_probs_tensor = torch.stack(trajectory["log_probs"])
    values = torch.stack(trajectory["values"])
    entropies = torch.stack(trajectory["entropies"])
    # gae
    advantages, returns = gae(
        trajectory["rewards"],
        values,
        trajectory["terminated"],
        bootstrap_value,
        gamma,
        gae_lambda,
    )
    # advantage normalization
    if len(advantages) > 1:
        advantages_normalized = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )
    else:
        advantages_normalized = advantages
    actor_loss = -(log_probs_tensor * advantages_normalized.detach()).mean()
    critic_loss = ((values - returns.detach()) ** 2).mean()
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
        "adv_std": (advantages.std().item() if len(advantages) > 1 else 0.0),
        "return_mean": (returns.mean().item()),
        "value_mean": (values.detach().mean().item()),
        "grad_norm": float(grad_norm),
    }


# collect one trajectory
def collect_episode(env: LineWorldEnv, model: ActorCriticNetwork):
    obs, _ = env.reset()
    rewards = []
    log_probs = []
    values = []
    entropies = []
    terminated_buffer = []
    obs_buffer = []
    actions = []
    final_terminated = False
    final_truncated = False
    final_next_obs = None

    # first collect trajectory
    while True:
        action, log_prob, value, entropy = select_action(model, obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)  # each log_prob is a tensor
        obs_buffer.append(obs)
        values.append(value)
        entropies.append(entropy)
        terminated_buffer.append(terminated)
        obs = next_obs
        if terminated or truncated:
            final_terminated = terminated
            final_truncated = truncated
            final_next_obs = next_obs
            break

    return {
        "rewards": rewards,
        "log_probs": log_probs,
        "values": values,
        "entropies": entropies,
        "terminated": terminated_buffer,
        "actions": actions,
        "obs": obs,
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


def evaluate(env: LineWorldEnv, model: ActorCriticNetwork, num_episodes=100):
    """
    using sample in training while using argmax in testing
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
                if terminated:
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
    value_coef = 0.5
    entropy_coef = 0.01
    learning_rate = 3e-4

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = LineWorldEnv(max_steps=50)
    env.reset(seed=seed)
    model = ActorCriticNetwork(obs_dim=1, action_dim=2, critic_dim=1, hidden_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    recent_returns = []
    successes = []
    for episode in range(1, num_episodes + 1):
        # phase 1: rollout/sample
        trajectory = collect_episode(env, model)
        bootstrap_value = compute_bootstrap_value(model, trajectory)
        # phase 2: model nework update
        stats = update_actor_critic(
            optimizer,
            model,
            trajectory,
            bootstrap_value,
            gamma,
            gae_lambda,
            value_coef,
            entropy_coef,
        )
        episode_return = sum(trajectory["rewards"])
        episode_length = len(trajectory["rewards"])
        recent_returns.append(episode_return)

        if trajectory["final_terminated"]:
            successes.append(1)
        else:
            successes.append(0)
        if episode == 10 or episode % 100 == 0:
            print(
                f"Episode {episode:3d} | "
                f"return={episode_return:7.2f} | "
                f"length={episode_length:2d} | "
                f"terminated={trajectory['final_terminated']} | "
                f"truncated={trajectory['final_truncated']} | "
                f"loss={stats['loss']:8.4f} | "
                f"actor={stats['actor_loss']:8.4f} | "
                f"critic={stats['critic_loss']:8.4f} | "
                f"entropy={stats['entropy']:6.4f} | "
                f"adv={stats['adv_mean']:7.3f} | "
                f"value={stats['value_mean']:7.3f} | "
                f"grad={stats['grad_norm']:7.3f}"
            )
        if episode % 200 == 0:
            eval_stats = evaluate(env, model, num_episodes=100)
            print(
                f"[Eval] episode={episode:3d} | "
                f"success_rate={eval_stats['success_rate']:.3f} | "
                f"avg_reward={eval_stats['avg_reward']:.3f} | "
                f"avg_steps={eval_stats['avg_steps']:.2f}"
            )
        if episode % 200 == 0:
            print(
                f"[Train] episode={episode} | "
                f"return_avg={np.mean(recent_returns[-20:]):.3f} | "
                f"success_avg={np.mean(successes[-20:]):.3f}"
            )


if __name__ == "__main__":
    main()
