
#
# Introduction to Reinforcement Learning
# Workshop 2 Code
#

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

lr = 0.01
gamma = 0.99
betas = (0.9, 0.999)


class ActorCritic(nn.Module):

    def __init__(self):
        super(ActorCritic, self).__init__()
        self.affine = nn.Linear(8, 128)
        self.action_layer = nn.Linear(128, 4)
        self.value_layer = nn.Linear(128, 1)
        self.state_values = []
        self.logprobs = []
        self.rewards = []

    def forward(self, state):
        state = torch.from_numpy(state).float()
        state = F.relu(self.affine(state))
        
        state_value = self.value_layer(state)
        
        action_probs = F.softmax(self.action_layer(state))
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        
        return action.item()
    
    def calc_loss(self, gamma=0.99):
        
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)
                
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        
        loss = 0
        z = zip(self.logprobs, self.state_values, rewards)

        for logprob, value, reward in z:
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            loss += (action_loss + value_loss)   

        return loss
    
    def clear_memory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]


if __name__ == '__main__':

    env = gym.make('LunarLander-v2')

    running_reward = 0
    policy = ActorCritic()
    optimizer = optim.Adam(policy.parameters(), lr=lr, betas=betas)

    for i in range(0, 10000):

        state = env.reset()

        for t in range(10000):
            action = policy(state)
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            running_reward += reward
            if i % 100 == 0: env.render()
            if done: break
              
        optimizer.zero_grad()
        loss = policy.calc_loss(gamma)
        loss.backward()
        optimizer.step()
        policy.clear_memory()
        
        if running_reward > 4000:
            print('Congratulations, you\'ve solved this challenge!')
            break
        
        if i % 20 == 0:
            running_reward = running_reward / 20
            output = 'Episode {}\tlength={}\treward:{}'
            print(output.format(i, t, running_reward))
            running_reward = 0
