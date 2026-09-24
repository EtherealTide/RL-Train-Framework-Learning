import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LineWorldEnv(gym.Env):
    '''
    state: pos[-10,10]
    action: [left,right]
    reward: [left:-1, right:+1, target:+10】
    target:pos10
    maxstep:30
    '''

    metadata={"render_modes":["human"]} # metadata是元数据的意思 键是rendermode 值是列表，里面有human

    def __init__(self,max_steps: int=30):
        super().__init__()
        self.max_steps=max_steps
        self.target_position=10.0

        self.action_space=spaces.Discrete(2) # action space:0 means left, 1 means right
        # observation dimention: 1d
        self.observation_space=spaces.Box(low=-10.0,high=10.0,shape=(1,),dtype=np.float32)
        self.position=0.0
        self.current_step=0

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)

        self.position=float(self.np_random.uniform(-5.0,5.0))
        self.current_step=0

        obs=np.array([self.position],dtype=np.float32)
        info={}
        return obs,info

    def step(self,action:int):
        '''
        1. 更新步骤编号
        2. 更新状态，注意限制范围
        3. 计算奖励
        4. 判断终止条件
        '''
        self.current_step+=1
        old_dist=abs(self.target_position-self.position)

        step_size=1.0
        if action==1:
            self.position+=step_size
        else:
            self.position-=step_size

        self.position=float(np.clip(self.position,-10.0,10.0))
        new_dist=abs(self.target_position-self.position)

        reward=1.0 if new_dist<old_dist else -1.0
        terminated=False
        if(self.position>=self.target_position):
            reward+=10.0
            terminated=True

        truncated=False
        if(self.current_step>self.max_steps):
            truncated=True
            reward-=5

        obs=np.array([self.position],dtype=np.float32)
        info={"position":self.position}

        return obs,reward,terminated,truncated,info

    