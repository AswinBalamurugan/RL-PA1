from tqdm import tqdm
from .base import RLAgent

class SARSAAgent(RLAgent):
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        super().__init__(env, learning_rate, discount_factor, epsilon)
    
    def train(self, num_episodes, max_steps):
        episode_rewards = []
        
        for episode in tqdm(range(num_episodes), desc="SARSA Training"):
            state, _ = self.env.reset()
            action = self.epsilon_greedy_policy(state)
            
            episode_reward = 0
            done = False
            truncated = False
            step = 0
            
            while not (done or truncated) and step < max_steps:
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_action = self.epsilon_greedy_policy(next_state)
                
                # SARSA update rule
                current_q = self.get_q_value(state, action)
                next_q = self.get_q_value(next_state, next_action)
                
                new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_q - current_q)
                self.set_q_value(state, action, new_q)
                
                state = next_state
                action = next_action
                episode_reward += reward
                step += 1
            
            episode_rewards.append(episode_reward)
        
        return episode_rewards

class SARSAAgentWithShaping(SARSAAgent):
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=0.1, reward_shaping_fn=None):
        super().__init__(env, learning_rate, discount_factor, epsilon)
        self.reward_shaping_fn = reward_shaping_fn
    
    def train(self, num_episodes, max_steps):
        episode_rewards = []
        
        for episode in tqdm(range(num_episodes), desc="SARSA with Shaping Training"):
            state, _ = self.env.reset()
            action = self.epsilon_greedy_policy(state)
            
            episode_reward = 0
            done = False
            truncated = False
            step = 0
            
            while not (done or truncated) and step < max_steps:
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_action = self.epsilon_greedy_policy(next_state)
                
                # Apply reward shaping if specified
                if self.reward_shaping_fn:
                    shaped_reward = self.reward_shaping_fn(next_state, reward, done)
                else:
                    shaped_reward = reward
                
                # SARSA update rule
                current_q = self.get_q_value(state, action)
                next_q = self.get_q_value(next_state, next_action)
                
                new_q = current_q + self.learning_rate * (shaped_reward + self.discount_factor * next_q - current_q)
                self.set_q_value(state, action, new_q)
                
                state = next_state
                action = next_action
                episode_reward += reward  # Track original reward for reporting
                step += 1
            
            episode_rewards.append(episode_reward)
        
        return episode_rewards 