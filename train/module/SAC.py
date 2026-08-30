import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import random
from train import config
# -------------------------- 经验回放缓冲区 --------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (torch.FloatTensor(state),
                torch.FloatTensor(action),
                torch.FloatTensor(reward).unsqueeze(1),
                torch.FloatTensor(next_state),
                torch.FloatTensor(done).unsqueeze(1))

    def __len__(self):
        return len(self.buffer)
# -------------------------- SAC 网络结构 --------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256,128,128], log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
            self.feature = nn.Sequential(*layers)
        self.mean = nn.Linear(prev_dim, action_dim)
        self.log_std = nn.Linear(prev_dim, action_dim)

    def forward(self, state):
        x = self.feature(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        if deterministic:
            z = mean
        else:
            z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, mean

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256,128,128]):
        super(QNetwork, self).__init__()
        layers = []
        prev_dim = state_dim + action_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q = self.net(x)
        return q
# -------------------------- SAC Agent --------------------------
class SACAgent:
    def __init__(self, state_dim, action_dim, action_scale, action_bias,
                 gamma=0.99, tau=0.005, alpha_lr=3e-4, q_lr=3e-4, policy_lr=3e-4,
                 buffer_capacity=1e6, batch_size=256, target_entropy=None, device='cpu'):
        self.device = device
        self.gamma = config.GAMMA
        self.tau = config.TAU
        self.alpha_lr = config.ALPHA_LR
        self.q_lr = config.QNETWORK_LR
        self.policy_lr = config.POLICY_LR
        self.batch_size = config.BATCH_SIZE
        self.action_scale = torch.FloatTensor(action_scale).to(device)
        self.action_bias = torch.FloatTensor(action_bias).to(device)

        self.actor = Actor(state_dim, action_dim).to(device)
        self.q1 = QNetwork(state_dim, action_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim).to(device)
        self.q1_target = QNetwork(state_dim, action_dim).to(device)
        self.q2_target = QNetwork(state_dim, action_dim).to(device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.policy_lr)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=self.q_lr)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=self.q_lr)

        if target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.alpha_lr)

        self.replay_buffer = ReplayBuffer(int(buffer_capacity))

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.actor.sample(state, deterministic)
        action = action.cpu().numpy()[0] * self.action_scale.cpu().numpy() + self.action_bias.cpu().numpy()
        return action

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            next_action_scaled = next_action * self.action_scale + self.action_bias
            target_q1 = self.q1_target(next_state, next_action_scaled)
            target_q2 = self.q2_target(next_state, next_action_scaled)
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        new_action, log_prob, _ = self.actor.sample(state)
        new_action_scaled = new_action * self.action_scale + self.action_bias
        q1_new = self.q1(state, new_action_scaled)
        q2_new = self.q2(state, new_action_scaled)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.log_alpha.exp().detach() * log_prob - q_new).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return q1_loss.item(), q2_loss.item(), actor_loss.item(), alpha_loss.item()