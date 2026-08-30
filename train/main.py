import numpy as np
import torch
import config
from train.module import SAC,MDPenv,SACtrain,evaluate,trajectories

# -------------------------- 参数导入 --------------------------
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

# -------------------------- 主程序 --------------------------
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
    pursuer = SAC.SACAgent(STATE_DIM, ACTION_DIM, ACTION_SCALE, ACTION_BIAS, device=device)
    evader = SAC.SACAgent(STATE_DIM, ACTION_DIM, ACTION_SCALE_E, ACTION_BIAS_E, device=device)

    # 训练
    NUM_EPISODES = 5000
    print("开始训练...")
    pursuer_rewards, evader_rewards, capture_rate, q1_losses, q2_losses, actor_losses, alpha_losses = SACtrain.train(env, pursuer, evader, NUM_EPISODES)

    # 保存模型
    torch.save(pursuer.actor.state_dict(), "pursuer_actor.pth")
    torch.save(evader.actor.state_dict(), "evader_actor.pth")

    print("正在生成训练结果曲线...")
    # 绘制学习曲线
    evaluate.plot_learning_curves(pursuer_rewards, evader_rewards, capture_rate)
    # 绘制损失曲线
    evaluate.plot_losses(q1_losses, q2_losses, actor_losses, alpha_losses)

    # 临时评估并保存成功轨迹数据
    success_rate, avg_time, success_data = evaluate.evaluate(env, pursuer, evader, num_episodes=100, save_trajectory=True)

    if success_data is not None:
        # 绘制相对距离和相对速度曲线
        evaluate.plot_distance_velocity(success_data['rel_traj'], success_data['rel_vel_traj'], DT)
        print("生成第一条捕获成功的轨迹...")
        trajectories.plot_single_trajectory(success_data['rel_traj'])
        print("全局追捕轨迹生成中...")
        trajectories.generate_global_animation(success_data['pos_p_traj'], success_data['pos_e_traj'], DT, filename='global_trajectory.gif')
    else:
        print("评估测试中未找到成功捕获的轨迹！")
