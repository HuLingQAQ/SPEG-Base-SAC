import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False
# -------------------------- 评估函数 --------------------------
def evaluate(env, pursuer_agent, evader_agent, num_episodes=100, save_trajectory=False):
    """
    评估智能体性能，可保存其中一条成功捕获的轨迹数据（包括相对和全局）。
    """
    success_count = 0
    capture_times = []
    success_data = None

    for ep in range(num_episodes):
        obs_p, obs_e = env.reset()
        rel_traj = [env.rel_pos.copy()]
        rel_vel_traj = [env.rel_vel.copy()]
        pos_p_traj = [env.pos_p.copy()]
        pos_e_traj = [env.pos_e.copy()]
        while True:
            action_p = pursuer_agent.select_action(obs_p, deterministic=True)
            action_e = evader_agent.select_action(obs_e, deterministic=True)
            (obs_p, obs_e), _, done, info = env.step(action_p, action_e)
            rel_traj.append(env.rel_pos.copy())
            rel_vel_traj.append(env.rel_vel.copy())
            pos_p_traj.append(env.pos_p.copy())
            pos_e_traj.append(env.pos_e.copy())
            if done:
                if info['捕获']:
                    success_count += 1
                    capture_times.append(info['时间'])
                    if save_trajectory and success_data is None:
                        success_data = {
                            'rel_traj': np.array(rel_traj),
                            'rel_vel_traj': np.array(rel_vel_traj),
                            'pos_p_traj': np.array(pos_p_traj),
                            'pos_e_traj': np.array(pos_e_traj),
                            'time': info['时间']
                        }
                break

    success_rate = success_count / num_episodes
    avg_time = np.mean(capture_times) if capture_times else np.inf
    print(f"\n蒙特卡洛测试 {num_episodes} 次:")
    print(f"捕获成功率: {success_rate*100:.2f}%")
    print(f"平均捕获时间: {avg_time:.2f} s")
    return success_rate, avg_time, success_data

# -------------------------- 曲线绘图 --------------------------
def plot_learning_curves(pursuer_rewards, evader_rewards, capture_rate):
    """
    绘制学习曲线：
    - 追击星/逃逸星回合奖励：散点图（每个episode一个点）
    - 捕获率：原始0/1散点图 + 移动平均线
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 追击星奖励（散点）
    axes[0, 0].scatter(range(len(pursuer_rewards)), pursuer_rewards, s=1, alpha=0.5, color='blue')
    axes[0, 0].set_title('追击星奖励')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('最终奖励')

    # 逃逸星奖励（散点）
    axes[0, 1].scatter(range(len(evader_rewards)), evader_rewards, s=1, alpha=0.5, color='orange')
    axes[0, 1].set_title('逃逸星奖励')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('最终奖励')

    # 捕获率：原始散点（0/1）+ 移动平均线
    episodes = np.arange(len(capture_rate))
    axes[1, 0].scatter(episodes, capture_rate, s=1, alpha=0.3, color='black', label='原始数据 (0/1)')
    # 计算移动平均（窗口20）
    window = 20
    moving_avg = np.convolve(capture_rate, np.ones(window) / window, mode='valid')
    axes[1, 0].plot(episodes[window - 1:], moving_avg, 'r-', linewidth=2, label=f'移动平均 (窗口大小 {window})')
    axes[1, 0].set_title('捕获成功率')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('平均成功率')
    axes[1, 0].set_ylim([-0.05, 1.05])
    axes[1, 0].legend(loc='upper left')

    axes[1, 1].axis('off')  # 右下角留空
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=150)
    plt.show()


def plot_distance_velocity(rel_traj, rel_vel_traj, dt):
    """
    绘制相对距离和相对速度大小随时间的变化曲线。
    rel_traj: (T, 3) 相对位置数组
    rel_vel_traj: (T, 3) 相对速度数组
    dt: 时间步长 (s)
    """
    T = rel_traj.shape[0]
    time = np.arange(T) * dt
    distance = np.linalg.norm(rel_traj, axis=1)
    speed = np.linalg.norm(rel_vel_traj, axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(time, distance, 'b-', linewidth=2)
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('相对距离 (m)')
    ax1.set_title('相对距离 vs 时间')
    ax1.grid(True)

    ax2.plot(time, speed, 'r-', linewidth=2)
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('相对速度 (m/s)')
    ax2.set_title('相对速度 vs 时间')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('distance_velocity.png', dpi=150)
    plt.show()


def plot_losses(q1_losses, q2_losses, actor_losses, alpha_losses):
    """
    绘制训练过程中的损失曲线（散点+移动平均）
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    episodes = np.arange(len(q1_losses))

    # Q1 损失
    axes[0, 0].scatter(episodes, q1_losses, s=1, alpha=0.3, color='blue')
    axes[0, 0].set_title('Q1 Loss')
    axes[0, 0].set_xlabel('Update Step')
    axes[0, 0].set_ylabel('Loss')
    # 移动平均
    window = 100
    if len(q1_losses) > window:
        q1_avg = np.convolve(q1_losses, np.ones(window) / window, mode='valid')
        axes[0, 0].plot(episodes[window - 1:], q1_avg, 'r-', linewidth=2)

    # Q2 损失
    axes[0, 1].scatter(episodes, q2_losses, s=1, alpha=0.3, color='green')
    axes[0, 1].set_title('Q2 Loss')
    axes[0, 1].set_xlabel('Update Step')
    axes[0, 1].set_ylabel('Loss')
    if len(q2_losses) > window:
        q2_avg = np.convolve(q2_losses, np.ones(window) / window, mode='valid')
        axes[0, 1].plot(episodes[window - 1:], q2_avg, 'r-', linewidth=2)

    # Actor 损失
    axes[1, 0].scatter(episodes, actor_losses, s=1, alpha=0.3, color='orange')
    axes[1, 0].set_title('Actor Loss')
    axes[1, 0].set_xlabel('Update Step')
    axes[1, 0].set_ylabel('Loss')
    if len(actor_losses) > window:
        actor_avg = np.convolve(actor_losses, np.ones(window) / window, mode='valid')
        axes[1, 0].plot(episodes[window - 1:], actor_avg, 'r-', linewidth=2)

    # Alpha 损失
    axes[1, 1].scatter(episodes, alpha_losses, s=1, alpha=0.3, color='purple')
    axes[1, 1].set_title('Alpha Loss')
    axes[1, 1].set_xlabel('Update Step')
    axes[1, 1].set_ylabel('Loss')
    if len(alpha_losses) > window:
        alpha_avg = np.convolve(alpha_losses, np.ones(window) / window, mode='valid')
        axes[1, 1].plot(episodes[window - 1:], alpha_avg, 'r-', linewidth=2)

    plt.tight_layout()
    plt.savefig('loss_curves.png', dpi=150)
    plt.show()