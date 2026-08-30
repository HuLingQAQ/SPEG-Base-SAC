import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from train import config
from train.module import SAC,MDPenv,evaluate,trajectories

# -------------------------- 导入参数 --------------------------
DT = config.DT
MAX_STEPS = config.MAX_STEPS
CAPTURE_DIST = config.CAPTURE_DIST
AMAX_P = config.AMAX_P
AMAX_E = config.AMAX_E
VMAX_ABS_P = config.VMAX_ABS_P
VMAX_ABS_E = config.VMAX_ABS_E
FUEL_INIT_P = config.FUEL_INIT_P
FUEL_INIT_E = config.FUEL_INIT_E
FUEL_CONSUMPTION_COEF = config.FUEL_CONSUMPTION_COEF
N = config.N

# --------------------------- 相对距离 ---------------------------
def moving_average(data, window=5):
    """简单移动平均，window为窗口大小（奇数）"""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def smooth_curve(data, method='moving_average', window=5):
    """
    对一维序列进行平滑。
    method: 'moving_average' 或 'savgol'
    """
    if method == 'moving_average':
        return moving_average(data, window)
    elif method == 'savgol':
        return savgol_filter(data, window_length=window, polyorder=2)
    else:
        return data
# --------------------------- 蒙特卡洛测试与全局轨迹保存 ---------------------------
def monte_carlo_test(env, pursuer_agent, evader_agent, num_episodes=100, smooth_window=11, save_result=True, result_dir="test_result"):
    """
    执行蒙特卡洛测试，可选择为每个 episode 保存全局轨迹 GIF。
    返回: success_rate, avg_capture_time, capture_times (列表)
    """
    if save_result:
        os.makedirs(result_dir, exist_ok=True)

    success_count = 0
    capture_times = []
    all_distance_curves = []
    all_speed_curves = []
    all_smooth_distance_curves = []
    all_smooth_speed_curves = []
    max_length = 0

    for ep in range(1, num_episodes + 1):
        obs_p, obs_e = env.reset()
        # 记录全局轨迹
        pos_p_traj = [env.pos_p.copy()]
        pos_e_traj = [env.pos_e.copy()]
        distances = []
        speeds = []
        step = 0
        captured = False

        while True:
            action_p = pursuer_agent.select_action(obs_p, deterministic=True)
            action_e = evader_agent.select_action(obs_e, deterministic=True)
            (obs_p, obs_e), _, done, info = env.step(action_p, action_e)
            pos_p_traj.append(env.pos_p.copy())
            pos_e_traj.append(env.pos_e.copy())
            dist = info['距离']
            distances.append(dist)
            rel_vel_norm = np.linalg.norm(env.rel_vel) if hasattr(env, 'rel_vel') else 0.0
            speeds.append(rel_vel_norm)
            step += 1
            if done:
                captured = info['捕获']
                if captured:
                    success_count += 1
                    capture_times.append(info['时间'])
                break

        # 平滑原始距离序列
        smoothed_distance = smooth_curve(distances, method='moving_average', window=smooth_window)
        smoothed_speed = smooth_curve(speeds, method='moving_average', window=smooth_window)

        all_distance_curves.append(np.array(distances))
        all_speed_curves.append(np.array(speeds))

        all_smooth_distance_curves.append(smoothed_distance)
        all_smooth_speed_curves.append(smoothed_speed)

        max_length = max(max_length, len(distances))

        # 保存 GIF（无论成败）
        if save_result:
            status = "success" if captured else "fail"
            result_filename = os.path.join(result_dir, f"episode_{ep:03d}_{status}.gif")
            print(f"正在保存 {result_filename} ...")
            trajectories.generate_global_animation(np.array(pos_p_traj), np.array(pos_e_traj), env.dt, result_filename)
        if ep % 10 == 0:
            print(f"已测试 {ep}/{num_episodes} 次, 目前捕获率: {success_count/ep*100:.1f}%")

    # 绘制所有平滑距离曲线
    plt.figure(figsize=(12, 6))
    for smooth_curve_data in all_smooth_distance_curves:
        # 时间轴需相应截断
        time_axis = np.arange(len(smooth_curve_data)) * env.dt
        plt.plot(time_axis, smooth_curve_data, color='blue', alpha=0.2, linewidth=0.8)
    plt.xlim(0, 500)
    plt.ylim(0, 6000)
    # 添加捕获半径虚线
    plt.axhline(y=100, color='black', linestyle='--', linewidth=1, label='捕获半径 (100 m)')
    # 计算平均平滑曲线
    padded_smooth = np.full((len(all_smooth_distance_curves), max_length), np.nan)
    for i, curve in enumerate(all_smooth_distance_curves):
        padded_smooth[i, :len(curve)] = curve
    mean_curve = np.nanmean(padded_smooth, axis=0)
    std_curve = np.nanstd(padded_smooth, axis=0)
    time_axis = np.arange(max_length) * env.dt

    plt.plot(time_axis, mean_curve, color='red', linewidth=2, label='平均相对距离')
    plt.fill_between(time_axis, mean_curve - std_curve, mean_curve + std_curve,
                     color='red', alpha=0.2, label='±1 std (标准差)')

    plt.xlabel('时间 (s)')
    plt.ylabel('相对距离 (m)')
    plt.title(f'相对距离变化 (共 {num_episodes} 次测试)\n'
              f'捕获成功率: {success_count / num_episodes * 100:.1f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('distance_curves.png', dpi=150)
    plt.show()

    # 绘制所有平滑速度曲线
    plt.figure(figsize=(12, 6))
    for smooth_curve_data in all_smooth_speed_curves:
        time_axis = np.arange(len(smooth_curve_data)) * env.dt
        plt.plot(time_axis, smooth_curve_data, color='blue', alpha=0.2, linewidth=0.8)
    plt.xlim(0, 500)
    plt.ylim(0, 50)
    # 计算平均平滑曲线
    padded_smooth = np.full((len(all_smooth_speed_curves), max_length), np.nan)
    for i, curve in enumerate(all_smooth_speed_curves):
        padded_smooth[i, :len(curve)] = curve
    mean_curve = np.nanmean(padded_smooth, axis=0)
    std_curve = np.nanstd(padded_smooth, axis=0)
    time_axis = np.arange(max_length) * env.dt

    plt.plot(time_axis, mean_curve, color='red', linewidth=2, label='平均相对速度')
    plt.fill_between(time_axis, mean_curve - std_curve, mean_curve + std_curve,
                     color='red', alpha=0.2, label='±1 std (标准差)')

    plt.xlabel('时间 (s)')
    plt.ylabel('相对速度 (m/s)')
    plt.title(f'相对速度变化 (共 {num_episodes} 次测试)\n'
              f'捕获成功率: {success_count / num_episodes * 100:.1f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('speed_curves.png', dpi=150)
    plt.show()

    success_rate = success_count / num_episodes
    avg_time = np.mean(capture_times) if capture_times else np.inf
    print(f"\n蒙特卡洛测试 ({num_episodes} 次):")
    print(f"捕获成功率: {success_rate*100:.2f}%")
    print(f"平均捕获时间: {avg_time:.2f} s")
    return success_rate, avg_time, capture_times, all_distance_curves, all_speed_curves

# -------------------------- 测试程序 --------------------------
if __name__ == "__main__":
    # 创建环境
    env = MDPenv.PursuitEvasionEnv(dt=DT, max_steps=MAX_STEPS, capture_dist=CAPTURE_DIST,
                                   amax_p=AMAX_P, amax_e=AMAX_E, vmax_abs_p=VMAX_ABS_P, vmax_abs_e=VMAX_ABS_E,
                                   fuel_init_p=FUEL_INIT_P, fuel_init_e=FUEL_INIT_E,
                                   fuel_consumption_coef=FUEL_CONSUMPTION_COEF, n=N)

    # 智能体参数
    STATE_DIM = env.state_dim
    ACTION_DIM = env.action_dim
    ACTION_SCALE = np.array([AMAX_P, AMAX_P, AMAX_P])
    ACTION_BIAS = np.array([0.0, 0.0, 0.0])
    ACTION_SCALE_E = np.array([AMAX_E, AMAX_E, AMAX_E])
    ACTION_BIAS_E = np.array([0.0, 0.0, 0.0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建两个SAC智能体
    pursuer_test = SAC.SACAgent(STATE_DIM, ACTION_DIM, ACTION_SCALE, ACTION_BIAS, device=device)
    evader_test = SAC.SACAgent(STATE_DIM, ACTION_DIM, ACTION_SCALE_E, ACTION_BIAS_E, device=device)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    # 加载模型权重
    models_dir = os.path.join(project_root, 'models')
    pursuer_path = os.path.join(models_dir, 'pursuer_actor.pth')
    evader_path = os.path.join(models_dir, 'evader_actor.pth')

    pursuer_test.actor.load_state_dict(torch.load(pursuer_path, map_location=device))
    evader_test.actor.load_state_dict(torch.load(evader_path, map_location=device))

    print("开始测试...")
    # 测试次数，根据需求设置，注意生成大量 GIF 可能耗时
    num_test_episodes = 100
    success_rate, avg_time, capture_times, all_distance_curves, all_speed_curves = monte_carlo_test(
        env, pursuer_test, evader_test,
        num_episodes=num_test_episodes,
        smooth_window=11,
        save_result=False,
        result_dir="test_result"
    )

    # 取前 20 次相对距离/速度曲线
    N = 20
    dist_curves = all_distance_curves[:N]
    speed_curves = all_speed_curves[:N]

    for i in range(N):
        # 提取第 i 次测试的曲线数据
        dist_smooth = dist_curves[i]
        speed_smooth = speed_curves[i]
        time_dist = np.arange(len(dist_smooth)) * env.dt
        time_speed = np.arange(len(speed_smooth)) * env.dt

        # 绘制相对距离变化曲线
        plt.figure(figsize=(8, 5))
        plt.plot(time_dist, dist_smooth, 'b-', linewidth=1.5)
        plt.axhline(y=100, color='r', linestyle='--', linewidth=1, label='捕获半径 (100 m)')
        plt.xlabel('时间 (s)')
        plt.ylabel('相对距离 (m)')
        plt.title(f'相对距离变化')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'test_{i + 1}_distance.png', dpi=150)
        plt.close()

        # 绘制相对速度变化曲线
        plt.figure(figsize=(8, 5))
        plt.plot(time_speed, speed_smooth, 'r-', linewidth=1.5)
        plt.xlabel('时间 (s)')
        plt.ylabel('相对速度 (m/s)')
        plt.title(f'相对速度变化')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'test_{i + 1}_speed.png', dpi=150)
        plt.close()

    print(f"已生成前 {N} 次测试的 {2 * N} 张曲线图（distance + speed）。")