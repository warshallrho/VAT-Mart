# FROM https://github.com/quantumiracle/SOTA-RL-Algorithms/blob/master/td3.py
'''
Twin Delayed DDPG (TD3), if no twin no delayed then it's DDPG.
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net, 1 target policy net
original paper: https://arxiv.org/pdf/1802.09477.pdf
'''
import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

        
class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(QNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)
        
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1) # the dim 0 is number of samples
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
        
        
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2, device=0, pred_world_xyz=0, wp_rot=False):
        super(PolicyNetwork, self).__init__()

        self.wp_rot = wp_rot
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = action_range
        self.num_actions = num_actions
        self.pred_world_xyz = pred_world_xyz
        self.device=device

        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean = torch.tanh(self.mean_linear(x))
        # mean = F.leaky_relu(self.mean_linear(x))
        # mean = torch.clamp(mean, -30, 30)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max) # clip the log_std into reasonable range
        
        return mean, log_std
    
    def evaluate(self, state, deterministic, eval_noise_scale, epsilon=1e-6):
        '''
        generate action with state as input wrt the policy network, for calculating gradients
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z = normal.sample() 
        action_0 = torch.tanh(mean + std * z.to(self.device)) # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range * mean if deterministic else self.action_range * action_0
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(self.device)) - torch.log(1. - action_0.pow(2) + epsilon).to(self.device) - torch.log(self.action_range) #torch.from_numpy(np.log(self.action_range.detach().cpu().numpy())).to(self.device)
        # both dims of normal.log_prob and -log(1-a**2) are (N, dim_of_action); 
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        ''' add noise '''
        eval_noise_clip = 2 * eval_noise_scale
        noise = normal.sample(action.shape) * eval_noise_scale
        if not self.wp_rot:
            noise = torch.clamp(noise, -eval_noise_clip, eval_noise_clip)
        else:
            if type(eval_noise_clip) == type(0.0):
                eval_noise_clip = torch.tensor([eval_noise_clip, eval_noise_clip, eval_noise_clip, eval_noise_clip, eval_noise_clip, eval_noise_clip])
            for idx in range(6):
                noise[:, idx] = torch.clamp(noise[:, idx], -eval_noise_clip[idx].float(), eval_noise_clip[idx].float())
        action = action + noise.to(self.device)
        if self.pred_world_xyz:
            start_gripper_root_position = state[0, 3: 6]
            for idx in range(3):
                action[idx] += start_gripper_root_position[idx]

        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state, deterministic, explore_noise_scale):
        '''
        generate action for interaction with env
        '''
        # state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z = normal.sample().to(self.device)
        
        action = mean.detach().cpu().numpy()[0] if deterministic else torch.tanh(mean + std * z).detach().cpu().numpy()[0]
        action = torch.tensor(action).to(self.device)

        ''' add noise '''
        noise = (normal.sample(action.shape) * explore_noise_scale).to(self.device)
        action = self.action_range * action + noise
        if self.pred_world_xyz:
            start_gripper_root_position = state[0, 3: 6]
            for idx in range(3):
                action[idx] += start_gripper_root_position[idx]

        return action


    def sample_action(self,):
        a=torch.FloatTensor(self.num_actions).uniform_(-1, 1)
        return self.action_range*a.numpy()


class TD3(nn.Module):
    def __init__(self, replay_buffer, state_dim, action_dim, hidden_dim, action_range, policy_target_update_interval=1, q_lr=3e-4, policy_lr=3e-4, device=0, pred_world_xyz=0, wp_rot=False):
        super(TD3, self).__init__()
        self.replay_buffer = replay_buffer
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.wp_rot = wp_rot

        self.q_net1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q_net2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_q_net1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_q_net2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range, device=device, pred_world_xyz=pred_world_xyz, wp_rot=wp_rot).to(device)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range, device=device, pred_world_xyz=pred_world_xyz, wp_rot=wp_rot).to(device)

        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)

        q_lr = q_lr
        policy_lr = policy_lr
        self.update_cnt = 0
        self.policy_target_update_interval = policy_target_update_interval

        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.pred_world_xyz = pred_world_xyz

        self.device = device
    
    def target_ini(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(param.data)
        return target_net

    def target_soft_update(self, net, target_net, soft_tau):    # Soft update the target net
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return target_net
    
    def update(self, batch_size, deterministic, eval_noise_scale, reward_scale=10., gamma=0.9, soft_tau=1e-2):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        predicted_q_value1 = self.q_net1(state, action)
        predicted_q_value2 = self.q_net2(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state, deterministic, eval_noise_scale=0.0)  # no noise, deterministic policy gradients
        new_next_action, _, _, _, _ = self.target_policy_net.evaluate(next_state, deterministic, eval_noise_scale=eval_noise_scale) # clipped normal noise

        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        target_q_min = torch.min(self.target_q_net1(next_state, new_next_action), self.target_q_net2(next_state, new_next_action))

        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward

        q_value_loss1 = ((predicted_q_value1 - target_q_value.detach())**2).mean()  # detach: no gradients for the variable
        q_value_loss2 = ((predicted_q_value2 - target_q_value.detach())**2).mean()
        self.q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.q_optimizer1.step()
        self.q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.q_optimizer2.step()

        # Training Policy Function
        if self.update_cnt % self.policy_target_update_interval == 0:
            ''' implementation 1 '''
            # predicted_new_q_value = torch.min(self.q_net1(state, new_action),self.q_net2(state, new_action))
            ''' implementation 2 '''
            predicted_new_q_value = self.q_net1(state, new_action)

            policy_loss = - predicted_new_q_value.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
        
            # Soft update the target nets
            self.target_q_net1 = self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2 = self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net = self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

        self.update_cnt += 1

        if (self.update_cnt - 1) % self.policy_target_update_interval == 0:
            return predicted_q_value1.mean(), q_value_loss1.mean(), q_value_loss2.mean(), policy_loss.mean()
        else:
            return predicted_q_value1.mean(), q_value_loss1.mean(), q_value_loss2.mean()

    def save_model(self, path):
        torch.save(self.q_net1.state_dict(), path+'_q1')
        torch.save(self.q_net2.state_dict(), path+'_q2')
        torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):
        self.q_net1.load_state_dict(torch.load(path+'_q1'))
        self.q_net2.load_state_dict(torch.load(path+'_q2'))
        self.policy_net.load_state_dict(torch.load(path+'_policy'))
        self.q_net1.eval()
        self.q_net2.eval()
        self.policy_net.eval()
