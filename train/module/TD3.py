# TD3.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from collections import deque
import random
import MDPenv
# ----------------------------- 经验回放缓冲区 -----------------------------
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

# ----------------------------- 网络结构 -----------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dims=[256, 128, 128]):
        super(Actor, self).__init__()
        self.max_action = max_action
        layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return torch.tanh(self.net(state)) * self.max_action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 128, 128]):
        super(Critic, self).__init__()
        # Q1
        layers1 = []
        prev_dim = state_dim + action_dim
        for h_dim in hidden_dims:
            layers1.append(nn.Linear(prev_dim, h_dim))
            layers1.append(nn.ReLU())
            prev_dim = h_dim
        layers1.append(nn.Linear(prev_dim, 1))
        self.q1 = nn.Sequential(*layers1)
        # Q2
        layers2 = []
        prev_dim = state_dim + action_dim
        for h_dim in hidden_dims:
            layers2.append(nn.Linear(prev_dim, h_dim))
            layers2.append(nn.ReLU())
            prev_dim = h_dim
        layers2.append(nn.Linear(prev_dim, 1))
        self.q2 = nn.Sequential(*layers2)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa)

# ----------------------------- TD3 智能体 -----------------------------
class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action,
                 lr=3e-4, gamma=0.99, tau=0.005, policy_noise=0.2,
                 noise_clip=0.5, policy_freq=2, buffer_capacity=1e6,
                 batch_size=256, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.batch_size = batch_size
        self.max_action = max_action

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(int(buffer_capacity))
        self.total_it = 0

    def select_action(self, state, deterministic=True, noise_scale=0.1):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        if not deterministic:
            action += np.random.normal(0, noise_scale, size=action.shape)
            action = np.clip(action, -self.max_action, self.max_action)
        return action

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        self.total_it += 1

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.q1_forward(state, self.actor(state)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item()

    def save_checkpoint(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim'])


# ----------------------------- 训练与评估函数 -----------------------------
def create_env():
    """创建环境并返回实例（需要确保 MDPenv 模块可导入）"""
    DT = 1.0
    MAX_STEPS = 8000
    CAPTURE_DIST = 100.0
    AMAX_P = 1.5
    AMAX_E = 1.2
    VMAX_ABS_P = 75.0
    VMAX_ABS_E = 60.0
    FUEL_INIT_P = 1000
    FUEL_INIT_E = 1000
    FUEL_CONSUMPTION_COEF = 1.0
    N = 7.292e-5

    env = MDPenv.PursuitEvasionEnv(dt=DT, max_steps=MAX_STEPS, capture_dist=CAPTURE_DIST,
                            amax_p=AMAX_P, amax_e=AMAX_E, vmax_abs_p=VMAX_ABS_P, vmax_abs_e=VMAX_ABS_E,
                            fuel_init_p=FUEL_INIT_P, fuel_init_e=FUEL_INIT_E,
                            fuel_consumption_coef=FUEL_CONSUMPTION_COEF, n=N)
    return env

def train_td3(num_episodes=1000, checkpoint_dir="checkpoints_td3"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    env = create_env()

    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action_pursuer = env.amax_p
    max_action_evader = env.amax_e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pursuer = TD3Agent(state_dim, action_dim, max_action_pursuer, device=device)
    evader = TD3Agent(state_dim, action_dim, max_action_evader, device=device)

    pursuer_rewards = []
    evader_rewards = []
    capture_rates = []
    exploration_noise = 0.2

    for episode in range(1, num_episodes + 1):
        obs_p, obs_e = env.reset()
        ep_p_reward = 0
        ep_e_reward = 0
        step = 0
        captured = False

        while True:
            action_p = pursuer.select_action(obs_p, deterministic=False, noise_scale=exploration_noise)
            action_e = evader.select_action(obs_e, deterministic=False, noise_scale=exploration_noise)

            (next_obs_p, next_obs_e), (r_p, r_e), done, info = env.step(action_p, action_e)

            pursuer.replay_buffer.push(obs_p, action_p, r_p, next_obs_p, done)
            evader.replay_buffer.push(obs_e, action_e, r_e, next_obs_e, done)

            obs_p, obs_e = next_obs_p, next_obs_e
            ep_p_reward += r_p
            ep_e_reward += r_e
            step += 1

            pursuer.update()
            evader.update()

            if done:
                captured = info['捕获']
                break

        exploration_noise = max(0.05, exploration_noise * 0.999)

        pursuer_rewards.append(ep_p_reward)
        evader_rewards.append(ep_e_reward)
        capture_rates.append(1 if captured else 0)

        if episode % 20 == 0:
            avg_cap = np.mean(capture_rates[-20:])
            print(f"Episode {episode}, Pursuer Reward: {ep_p_reward:.2f}, Evader Reward: {ep_e_reward:.2f}, "
                  f"Capture Rate (last20): {avg_cap:.2f}, Final Dist: {info['距离']:.2f}")

        if episode % 1000 == 0:
            pursuer.save_checkpoint(os.path.join(checkpoint_dir, f"pursuer_{episode}.pth"))
            evader.save_checkpoint(os.path.join(checkpoint_dir, f"evader_{episode}.pth"))

    return pursuer_rewards, evader_rewards, capture_rates, pursuer, evader

def evaluate_td3(pursuer_agent, evader_agent, num_episodes=100):
    env = create_env()
    success_count = 0
    capture_times = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs_p, obs_e = env.reset()
        step = 0
        captured = False
        while True:
            action_p = pursuer_agent.select_action(obs_p, deterministic=True)
            action_e = evader_agent.select_action(obs_e, deterministic=True)
            (obs_p, obs_e), _, done, info = env.step(action_p, action_e)
            step += 1
            if done:
                if info['捕获']:
                    success_count += 1
                    capture_times.append(info['时间'])
                    episode_lengths.append(step)
                break

    success_rate = success_count / num_episodes
    avg_time = np.mean(capture_times) if capture_times else np.inf
    avg_length = np.mean(episode_lengths) if episode_lengths else np.inf

    print(f"\nTD3 Evaluation Results ({num_episodes} episodes):")
    print(f"Success rate: {success_rate*100:.2f}%")
    print(f"Average capture time: {avg_time:.2f} s")
    print(f"Average episode length: {avg_length:.1f} steps")
    return success_rate, avg_time

# ----------------------------- 主程序 -----------------------------
if __name__ == "__main__":
    # 训练
    pursuer_rewards, evader_rewards, capture_rates, pursuer, evader = train_td3(num_episodes=5000)

    # 评估
    success_rate, avg_time = evaluate_td3(pursuer, evader, num_episodes=100)

    # 保存最终模型
    pursuer.save_checkpoint("pursuer_td3_final.pth")
    evader.save_checkpoint("evader_td3_final.pth")