import torch


def gae(rewards, values, terminated, bootstrap_value, gamma=0.95, gae_lambda=0.95):
    """
    input:
        rewards: output by env, general list
        values: output by critic nework, torch list
        terminated: output by env, general list
        bootstrap_value: torch
    output:
        advantages: torch list
    """
    values_detached = values.detach()
    rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32)
    terminated_tensor = torch.as_tensor(terminated, dtype=torch.float32)
    advantages = torch.zeros_like(rewards_tensor)
    next_advantage = torch.tensor(0.0)
    next_value = bootstrap_value.detach()  # next_value initialization
    for t in reversed(range(len(rewards))):
        non_terminal = 1 - terminated_tensor[t]
        delta_t = (
            rewards_tensor[t] + gamma * non_terminal * next_value - values_detached[t]
        )
        advantages[t] = delta_t + gamma * gae_lambda * non_terminal * next_advantage
        next_value = values_detached[t]
        next_advantage = advantages[t]

    returns = advantages + values_detached
    return advantages, returns
