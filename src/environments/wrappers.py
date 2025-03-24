class MiniGridWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space['image']
        self.action_space = env.action_space
        
    def reset(self):
        obs, info = self.env.reset()
        return obs['image'], info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs['image'], reward, terminated, truncated, info 