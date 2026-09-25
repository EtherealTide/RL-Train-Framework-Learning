import random

import numpy as np
import torch
from minimal_env import LineWorldEnv
from torch import nn, optim
from torch.distributions import Categorical


# 1. policy network
class PolicyNetwork(nn.Module):
    """
    input:
        observation, shape=[batch_size,1]
    output:
        logits, shape=[batch_size,2]
        notice that there is no softmax in the end
    """

    def __init__(self, obs_dim=1, action_dim=2, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs):
        logits = self.net(obs)
        return logits


# choose action from output of policy network
def select_action(policy: PolicyNetwork, obs):
    """
    input:
        policy: policy network
        obs: numpy returned by Gymnasium  type:numpy array requested by gym

    output:
        action: sample an action under probability distribution of action space
        log_prob: the probability of chosen action
    """
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
    # increase dimension of obs_tensor using unsqueeze because neural network usually accept:
    # [batch_size, feature_dim]
    obs_tensor = obs_tensor.unsqueeze(0)

    logits = policy(obs_tensor)
    dist = Categorical(logits=logits)
    action_tensor = dist.sample()
    log_prob = dist.log_prob(action_tensor)
    action = (
        action_tensor.item()
    )  # action is only sent to Gym without participating upgrade of network
    log_prob = log_prob.squeeze(0)
    return action, log_prob


def compute_discounted_returns(rewards: list, gamma=0.99):
    """
    input:
        rewards: reward list
        gamma: discount factor
    output:
        returns: (type)torch.float32
    """
    returns = []
    running_return = 0.0
    for reward in reversed(rewards):
        running_return = running_return * gamma + reward
        returns.append(running_return)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def update_policy(optimizer: torch.optim, returns, log_probs):
    """
    Loss Equation: loss=sum(G*log_prob)
    input:
        returns:G
        log_probs:log_prob
        optimizer:'Adam' 'SGD' ...
    output:
        new policy network with new paramenters
    """
    log_probs_tensor = torch.stack(log_probs)  # turn into a 1 dimension tensor
    policy_loss = -(log_probs_tensor * returns).sum()

    # update
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    return policy_loss.item()


def collect_episode(env: LineWorldEnv, policy: PolicyNetwork):
    obs, _ = env.reset()
    rewards = []
    log_probs = []
    obs_buffer = []
    actions = []
    # first collect trajectory
    while True:
        action, log_prob = select_action(policy, obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)  # each log_prob is a tensor
        obs_buffer.append(obs)
        obs = next_obs
        done = terminated or truncated
        if done:
            break

    return rewards, log_probs, terminated


def main():
    # hyperparameters
    num_episodes = 300
    gamma = 0.95
    learning_rate = 1e-2

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = LineWorldEnv(max_steps=50)
    env.reset(seed=seed)
    policy = PolicyNetwork(obs_dim=1, action_dim=2, hidden_dim=32)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    recent_returns = []
    successes = []
    for episode in range(1, num_episodes + 1):
        # phase 1: rollout/sample
        rewards, log_probs, success = collect_episode(env, policy)

        # phase 2: policy nework update
        returns = compute_discounted_returns(rewards, gamma)
        policy_loss = update_policy(optimizer, returns, log_probs)

        # debug information
        episode_return = sum(rewards)
        episode_length = len(rewards)
        recent_returns.append(episode_return)
        successes.append(success)

        if len(recent_returns) > 20:
            recent_returns.pop(0)

        if episode % 10 == 0:
            average_return = np.mean(recent_returns)
            success_rate = np.mean(successes[-200:])
            print(
                f"episode={episode:4d} "
                f"return={episode_return:7.2f} "
                f"success_rate={success_rate:.2%} "
                f"avg_return={average_return:7.2f} "
                f"length={episode_length:3d} "
                f"loss={policy_loss:8.3f}"
            )


if __name__ == "__main__":
    main()
