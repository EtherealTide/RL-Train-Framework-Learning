"""
Day 05 - Soft Actor-Critic (SAC)

Main goals:

1. Move from on-policy PPO to off-policy SAC.
2. Implement a circular Replay Buffer.
3. Implement a squashed Gaussian Actor.
4. Implement Twin Q Networks.
5. Implement Target Q Networks.
6. Implement critic / actor / alpha updates.
7. Implement Polyak soft target updates.

Data flow:

Environment
    |
    v
(s, a, r, s', terminated)
    |
    v
Replay Buffer
    |
    | random sample
    v
Minibatch
    |
    +------> Critic Update
    |
    +------> Actor Update
    |
    +------> Alpha Update
    |
    +------> Target Network Soft Update
"""

import copy
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal


# replay buffer
class ReplayBuffer:
    """
    Circular replay buffer.
    Each transition stores: observation action next_observation reward terminated
    Notice that we only store 'terminated' but not 'terminated or truncated'
    because a time-limit truncation does not mean that the mdp truly ends

    For a truncated transition, Q-learning should still bootstrap from Q(S', a')
    """

    def __init__(self, capacity, obs_dim, action_dim):
        self.capacity = capacity
        # pre-allocate contiguous arrays
        self.observations = np.empty((capacity, obs_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.next_observations = np.empty((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.empty((capacity, 1), dtype=np.float32)

        # ptr: where the next transition should be written
        # size: how many valid transitions currently exist
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, terminated):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.terminated[self.ptr] = terminated
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "observations": torch.as_tensor(self.observations[indices], device=device),
            "actions": torch.as_tensor(self.actions[indices], device=device),
            "rewards": torch.as_tensor(self.rewards[indices], device=device),
            "next_observations": torch.as_tensor(
                self.next_observations[indices], device=device
            ),
            "terminated": torch.as_tensor(self.terminated[indices], device=device),
        }

    def __len__(self):
        return self.size


# squashed gaussian actor 压缩高斯actor
class GaussianActor(nn.Module):
    """
    contiunous-action SAC actor
    input:
        observation

    output:
        mean
        log_std

    The actual action is sample as:
        u ~ Normal(mean, std)
        y = tanh(u)

    action = y*action_scale+action_bias

    Why tanh?

    A Gaussian is unbounded: (-infinity,infinity)

    but enviromnets normally have bounded action spaces, for example Pendulum-v1:
        action in [-2,2]
    """

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(
        self, obs_dim, action_dim, action_low=-2, action_high=2, hidden_dim=256
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        # tanh output: [-1,1] -> action = tanh(x) * scale + bias
        action_low = torch.as_tensor(action_low, dtype=torch.float32)
        action_high = torch.as_tensor(action_high, dtype=torch.float32)
        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0

        # register_buffer: tensors that are not trainable parameters but should move automatically when model.to("cuda")
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(self, obs):
        """
        Return Gaussian parameters

        notice that we predict log_std instead of std, so we need to implement std=exp(log_std)
        which guarantees: std>0
        """
        features = self.net(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        # prevent extremely tiny or huge std
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        return mean, log_std

    def sample(self, obs):
        """
        rSample a differentiable SAC action

        Returns:
            action
            log_prob
            deterministic_action

        notice that rsample() uses the reparameterization trick(重参数化技巧):
            gradient flows: action x_t -> mean/std -> actor para
        """

        mean, log_std = self(obs)  # call self.forward
        std = log_std.exp()
        distribution = Normal(mean, std)
        x_t = distribution.rsample()  # [-infinity,infinity]
        y_t = torch.tanh(x_t)  # [-1,1]
        action = y_t * self.action_scale + self.action_bias
        log_prob = distribution.log_prob(x_t)
        # log probability correction
        # correct changes of log prob caused by tanh scale - 使用雅可比矩阵变换修正
        log_prob -= torch.log(self.action_scale * (1.0 - y_t.pow(2)) + 1e-6)
        # compute sum of every dimensions of action
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        # 评估所需的确定性动作：用mean来代替rsample采样的x_t
        deterministic_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, deterministic_action


# Q Network
class QNetwork(nn.Module):
    """
    State-action value network: Q(s,a), it receives both state and action
    They are concatenated before entering the MLP
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


# Twin Q Network
class TwinQNetwork(nn.Module):
    """
    SAC keeps two independent Q networks.
    Goal: reduce Q-Value overestimation bias
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q1 = QNetwork(obs_dim, action_dim, hidden_dim)
        self.q2 = QNetwork(obs_dim, action_dim, hidden_dim)

    def forward(self, obs, action):
        return self.q1(obs, action), self.q2(obs, action)


# polyak / soft target update
@torch.no_grad()
def soft_update(online_network, target_network, tau):
    """
    Target-network update:
        target=(1-tau)*target+tau*online
    generally tau=0.005
    This means update copies only 0.5% of the online-network movement into target network.
    """
    for online_param, target_param in zip(
        online_network.parameters(), target_network.parameters()
    ):
        # in-place tensor operations:
        # mul_(x): tensor*=x
        # add_(x, alpha): tensor+=alpha*x
        target_param.mul_(1.0 - tau)
        target_param.add_(online_param, alpha=tau)


# one SAC Gradient Update
def update_sac(
    replay_buffer: ReplayBuffer,
    actor: GaussianActor,
    critic: TwinQNetwork,
    target_critic: TwinQNetwork,
    actor_optimizer: torch.optim,
    critic_optimizer: torch.optim,
    log_alpha,
    alpha_optimizer: torch.optim,
    batch_size,
    gamma,
    tau,
    target_entropy,
    device,
):
    """
    One complete SAC learner update

    Order:
        1. sample replay buffer
        2. critic update
        3. actor update
        4. entropy temperature update
        5. target critic soft update
    """
    batch = replay_buffer.sample(batch_size, device)
    obs = batch["observations"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    next_obs = batch["next_observations"]
    terminated = batch["terminated"]

    # critic target
    with torch.no_grad():
        # sample 'a' from current actor:a~pi(.|s)
        next_actions, next_log_prob, _ = actor.sample(next_obs)
        target_q1, target_q2 = target_critic(next_obs, next_actions)
        # clipped double-Q
        min_target_q = torch.min(target_q1, target_q2)
        # current entropy temperature
        alpha = log_alpha.exp()
        # soft state-action value: min(q1,q2)-alpha*log pi(a|s)
        next_soft_value = min_target_q - alpha * next_log_prob
        # Bellman target
        # only true termination disables bootstrap
        target_q = rewards + gamma * (1.0 - terminated) * next_soft_value

    # critic update
    current_q1, current_q2 = critic(obs, actions)
    q1_loss = F.mse_loss(current_q1, target_q)
    q2_loss = F.mse_loss(current_q2, target_q)
    critic_loss = q1_loss + q2_loss
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # actor update
    new_actions, log_prob, _ = actor.sample(obs)
    # freeze critic parameters
    for param in critic.parameters():
        param.requires_grad_(False)
    q1_pi, q2_pi = critic(obs, new_actions)
    min_q_pi = torch.min(q1_pi, q2_pi)
    # detach alpha: actor loss should update actor but not entropy-temp para
    alpha = log_alpha.exp().detach()

    # SAC actor objective
    # maximize: q-alpha*log_pi -> minimize: alpha*log_pi-Q
    actor_loss = (alpha * log_prob - min_q_pi).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # re-enable critic gradients for next iteration
    for param in critic.parameters():
        param.requires_grad_(True)

    # automatic entropy temperature
    alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    # target network update
    soft_update(critic, target_critic, tau)

    return {
        "critic_loss": critic_loss.item(),
        "actor_loss": actor_loss.item(),
        "alpha_loss": alpha_loss.item(),
        "alpha": log_alpha.exp().item(),
        "q1_mean": current_q1.detach().mean().item(),
        "q2_mean": current_q2.detach().mean().item(),
        "target_q_mean": target_q.mean().item(),
        "log_prob_mean": log_prob.detach().mean().item(),
    }


# Action selection
@torch.no_grad()
def select_action(actor: GaussianActor, obs, device, deterministic=False):
    """
    Actor inference used during environment interaction

    Training: deterministic=False -> stochastic sample
    Evaluation: deterministic=True -> tanh(mean), without random sampling.
    """
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    if deterministic:
        mean, _ = actor(obs_tensor)
        selected_action = torch.tanh(mean) * actor.action_scale + actor.action_bias
    else:
        selected_action, _, _ = actor.sample(obs_tensor)
    return selected_action.squeeze(0).cpu().numpy()


# Pendulum evaluation
@torch.no_grad()
def evaluate(
    actor,
    env,
    device,
    num_episodes=50,
    eval_seed=10_000,
    angle_tolerance_deg=10.0,
    velocity_tolerance=0.5,
    success_window=50,
):
    """
    Evaluate Pendulum with deterministic actions and fixed episode seeds.

    Custom success criterion (not a built-in environment signal):
    every one of the final `success_window` post-action states must satisfy
    |theta| < angle_tolerance_deg and |angular velocity| < velocity_tolerance.
    Episodes shorter than success_window cannot count as successful.

    upright_ratio covers the whole episode, including swing-up.
    tail_mean_reward averages the rewards of the final success_window steps.
    """
    if num_episodes <= 0 or success_window <= 0:
        raise ValueError("num_episodes and success_window must be positive")
    if angle_tolerance_deg <= 0 or velocity_tolerance <= 0:
        raise ValueError("angle and velocity tolerances must be positive")

    angle_tolerance = np.deg2rad(angle_tolerance_deg)
    episode_returns = []
    episode_successes = []
    episode_upright_ratios = []
    episode_tail_rewards = []
    for episode_idx in range(num_episodes):
        # Reuse the same initial states at every evaluation checkpoint.
        obs, _ = env.reset(seed=eval_seed + episode_idx)
        episode_return = 0.0
        episode_rewards = []
        upright_steps = 0
        consecutive_upright_steps = 0
        while True:
            action = select_action(actor, obs, device, True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            episode_rewards.append(float(reward))

            # Pendulum observation: [cos(theta), sin(theta), angular velocity].
            theta = np.arctan2(obs[1], obs[0])
            is_upright = (
                abs(theta) < angle_tolerance and abs(obs[2]) < velocity_tolerance
            )
            upright_steps += int(is_upright)
            # Reset on failure: merely passing through upright is not success.
            consecutive_upright_steps = (
                consecutive_upright_steps + 1 if is_upright else 0
            )
            if terminated or truncated:
                break
        episode_returns.append(episode_return)
        episode_successes.append(consecutive_upright_steps >= success_window)
        episode_upright_ratios.append(upright_steps / len(episode_rewards))
        episode_tail_rewards.append(np.mean(episode_rewards[-success_window:]))
    return {
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "success_rate": float(np.mean(episode_successes)),
        "upright_ratio": float(np.mean(episode_upright_ratios)),
        "tail_mean_reward": float(np.mean(episode_tail_rewards)),
    }


def main():
    # hyper parameters
    seed = 42
    total_steps = 100000
    buffer_size = 100000
    batch_size = 256
    learning_starts = 1000
    gamma = 0.99
    tau = 0.005
    actor_lr = 3e-4
    critic_lr = 3e-4
    alpha_lr = 3e-4
    eval_interval = 5000
    eval_episodes = 50
    eval_seed = seed + 10_000
    # Custom Pendulum success criterion; adjust these for a different standard.
    angle_tolerance_deg = 10.0
    velocity_tolerance = 0.5
    success_window = 50

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda")

    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    actor = GaussianActor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=(env.action_space.low),
        action_high=(env.action_space.high),
    ).to(device)
    critic = TwinQNetwork(
        obs_dim,
        action_dim,
    ).to(device)
    target_critic = copy.deepcopy(critic)
    for param in target_critic.parameters():
        param.requires_grad_(False)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    log_alpha = torch.zeros(1, device=device, requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=alpha_lr)
    target_entropy = float(-action_dim)
    replay_buffer = ReplayBuffer(
        capacity=buffer_size, obs_dim=obs_dim, action_dim=action_dim
    )
    episode_return = 0.0
    episode_length = 0
    recent_returns = []

    for global_step in range(1, total_steps + 1):
        # environment interation
        # fills the initial first
        if global_step <= learning_starts:
            action = env.action_space.sample()
        else:
            action = select_action(actor, obs, device, False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        replay_buffer.add(obs, action, reward, next_obs, terminated)
        episode_return += float(reward)
        episode_length += 1

        # sac upsate
        if global_step > learning_starts and len(replay_buffer) >= batch_size:
            stats = update_sac(
                replay_buffer,
                actor,
                critic,
                target_critic,
                actor_optimizer,
                critic_optimizer,
                log_alpha,
                alpha_optimizer,
                batch_size,
                gamma,
                tau,
                target_entropy,
                device,
            )

        # episode boundary
        if terminated or truncated:
            recent_returns.append(episode_return)
            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0
        else:
            obs = next_obs

        # evalution
        if global_step % eval_interval == 0:
            eval_stats = evaluate(
                actor,
                eval_env,
                device,
                num_episodes=eval_episodes,
                eval_seed=eval_seed,
                angle_tolerance_deg=angle_tolerance_deg,
                velocity_tolerance=velocity_tolerance,
                success_window=success_window,
            )

            train_return = (
                np.mean(recent_returns[-20:]) if recent_returns else float("nan")
            )

            if global_step > learning_starts:
                print(
                    f"step="
                    f"{global_step:7d} | "
                    f"buffer="
                    f"{len(replay_buffer):6d} | "
                    f"train_return="
                    f"{train_return:8.2f} | "
                    f"eval_return="
                    f"{eval_stats['mean_return']:8.2f} "
                    f"+/- {eval_stats['std_return']:.2f} | "
                    f"success={eval_stats['success_rate']:.1%} | "
                    f"upright={eval_stats['upright_ratio']:.1%} | "
                    f"tail_reward={eval_stats['tail_mean_reward']:.4f} | "
                    f"critic="
                    f"{stats['critic_loss']:8.4f} | "
                    f"actor="
                    f"{stats['actor_loss']:8.4f} | "
                    f"alpha="
                    f"{stats['alpha']:.4f} | "
                    f"logp="
                    f"{stats['log_prob_mean']:.4f} | "
                    f"Q1="
                    f"{stats['q1_mean']:.3f} | "
                    f"targetQ="
                    f"{stats['target_q_mean']:.3f}"
                )


if __name__ == "__main__":
    main()
