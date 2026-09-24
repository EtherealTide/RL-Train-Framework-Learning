import numpy as np
from minimal_env import LineWorldEnv

def random_policy(obs): # 实际要通过根据当前obs通过policy网络获取概率，再进行一步采样
    return np.random.randint(0,2) # 随机选择0或者1

def compute_discounted_returns(rewards,gamma=0.99):
    '''
    G[t]=sum gamma^k r[t+k]  r is short for reward
    G[t]=r[t]+gamma*G[t+1]
    '''
    returns=np.zeros_like(rewards,dtype=np.float32)

    running_add=0.0
    for t in reversed(range(len(returns))):
        running_add=rewards[t]+gamma*running_add
        returns[t]=running_add

    return returns

def main():
    env=LineWorldEnv(max_steps=30)
    gamma=0.95

    obs_buffer=[]
    action_buffer=[]
    reward_buffer=[]
    done_buffer=[]

    obs,info=env.reset(seed=42)
    print(f"start position: {obs[0]}")

    while True:
        action=random_policy(obs)
        next_obs,reward,terminated,truncated,info=env.step(action)
        done=terminated or truncated
        obs_buffer.append(obs)
        done_buffer.append(done)
        reward_buffer.append(reward)
        action_buffer.append(action)

        obs=next_obs
        if done:
            break

    rewards=np.array(reward_buffer,dtype=np.float32)
    returns=compute_discounted_returns(rewards,gamma)
    print(returns)
    print(np.array(obs_buffer,dtype=np.float32))

if __name__ =="__main__":
    main()