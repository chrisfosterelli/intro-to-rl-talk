
#
# Introduction to Reinforcement Learning
# Workshop 1 Code
#

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

percentile = 80
batch_size = 100
session_size = 50
learning_rate = 0.01
completion_score = 195


class CEM(nn.Module):

    def __init__(self, obs_size, n_actions):
        super(CEM, self).__init__()
        # =========== START OF FILL IN ===========
        # Define your neural network layers here!
        # Your network must accept obs_size inputs and have n_actions outputs
        # The size in between, and how many layers, is up to you.
        #
        # In PyTorch, we recommend you assign the network layers to this class,
        # and then you can call them later when you do your forward pass in
        # the function below. See the PyTorch documentation for an example.
        # =========== END OF FILL IN ===========
        
    def forward(self, x):
        # =========== START OF FILL IN ===========
        # Execute a foward pass of your neural network!
        # You'll want to call your neural network layers here with the input x,
        # and return the output from this function. This code should be paired
        # with the layers you attached to the class in the above function.
        #
        # Note: Do not apply an activation function to your second layer. 
        # Softmax is called by the loss function and by the batch generator.
        # =========== END OF FILL IN ===========


def generate_batch(env, batch_size, t_max=5000):
    
    activation = nn.Softmax(dim=1)
    batch_actions, batch_states, batch_rewards = [], [], []
    
    for b in range(batch_size):

        s = env.reset()
        total_reward = 0
        states, actions = [], []
        
        for t in range(t_max):
            
            if b == 0: env.render()
            s_v = torch.FloatTensor([ s ])
            act_probs_v = activation(net(s_v))
            act_probs = act_probs_v.data.numpy()[0]
            a = np.random.choice(len(act_probs), p=act_probs)

            new_s, r, done, info = env.step(a)

            states.append(s)
            actions.append(a)
            total_reward += r

            s = new_s

            if done:
                batch_actions.append(actions)
                batch_states.append(states)
                batch_rewards.append(total_reward)
                break
                
    return batch_states, batch_actions, batch_rewards


def filter_batch(states, actions, rewards, percentile=70):
    # =========== START OF FILL IN ===========
    # Implement some code to filter the batch! 
    # If you are familiar with evolutionary algorithms, this is just like that.
    #
    # Here is the shape of your input:
    #   states[i][j] - returns a state array at step j of episode i
    #   actions[i][j] - returns an action array at step j of episode i
    #   rewards[i] - the total reward in episode i
    #
    # You should output:
    #   elite_states[n] - returns the n-th elite state array 
    #   elite_actions[n] - returns the n-th elite actions array
    #
    # Where elite episodes are episodes with reward in the top percentile.
    # =========== END OF FILL IN ===========


if __name__ == '__main__':

    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]

    net = CEM(n_states, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)

    for i in range(session_size):

        batch_states, batch_actions, batch_rewards = generate_batch(
            env,
            batch_size,
            t_max=5000
        )

        elite_states, elite_actions = filter_batch(
            batch_states,
            batch_actions,
            batch_rewards,
            percentile
        )

        optimizer.zero_grad()
        tensor_states = torch.FloatTensor(elite_states)
        tensor_actions = torch.LongTensor(elite_actions)
        action_scores_v = net(tensor_states)
        loss_v = objective(action_scores_v, tensor_actions)
        loss_v.backward()
        optimizer.step()

        mean_reward = np.mean(batch_rewards)
        threshold = np.percentile(batch_rewards, percentile)

        output = '{}: loss={:.3f}, reward_mean={:.1f}, reward_threshold={:.1f}'
        print(output.format(i, loss_v.item(), mean_reward, threshold))
        
        if mean_reward > completion_score:
            print('Congratulations, you\'ve solved this challenge!')
