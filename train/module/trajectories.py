import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

# -------------------------- 单条轨迹绘图 --------------------------
def plot_single_trajectory(traj):
    """
    用于临时观察训练轨迹是否正常，不作为最终结果使用
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=2, color='blue', label='轨迹')
    ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color='green', s=80, label='起点')
    ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color='red', s=80, label='终点')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title('单次成功捕获轨迹示意图')
    ax.legend()
    plt.savefig('success_trajectory.png', dpi=150)
    plt.show()

# -------------------------- 动态轨迹示意图（测试） --------------------------
def generate_global_animation(pos_p_traj, pos_e_traj, dt, filename='global_trajectory.gif'):
    """
    生成追击星和逃逸星的全局轨迹动画。
    若需单独加载模型测试，则使用test.py进行
    pos_p_traj, pos_e_traj: (T, 3) 数组，时间步数T
    dt: 时间步长
    filename: 保存的文件名，支持.gif或.mp4（需额外编码器）
    """
    T = pos_p_traj.shape[0]
    time = np.arange(T) * dt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 初始化两个点（追击星红色，逃逸星蓝色）
    line_p, = ax.plot([], [], [], 'r-', linewidth=1, label='追击星')
    line_e, = ax.plot([], [], [], 'b-', linewidth=1, label='逃逸星')
    point_p, = ax.plot([], [], [], 'ro', markersize=4)
    point_e, = ax.plot([], [], [], 'bo', markersize=4)

    # 设置坐标轴范围（考虑全局轨迹的范围）
    all_pos = np.vstack([pos_p_traj, pos_e_traj])
    max_range = np.max(np.max(all_pos, axis=0) - np.min(all_pos, axis=0)) * 0.5
    mid = np.mean(all_pos, axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'全局轨迹示意图 (时间 {time[-1]:.1f} s)')
    ax.legend()

    # 标记起点和终点
    ax.scatter(pos_p_traj[0, 0], pos_p_traj[0, 1], pos_p_traj[0, 2], c='darkred', marker='^', s=50,
               label='追击星起点')
    ax.scatter(pos_e_traj[0, 0], pos_e_traj[0, 1], pos_e_traj[0, 2], c='darkblue', marker='^', s=50,
               label='逃逸星起点')
    ax.scatter(pos_p_traj[-1, 0], pos_p_traj[-1, 1], pos_p_traj[-1, 2], c='gold', marker='*', s=100,
               label='成功捕获点')

    def init():
        line_p.set_data([], [])
        line_p.set_3d_properties([])
        line_e.set_data([], [])
        line_e.set_3d_properties([])
        point_p.set_data([], [])
        point_p.set_3d_properties([])
        point_e.set_data([], [])
        point_e.set_3d_properties([])
        return line_p, line_e, point_p, point_e

    def update(frame):
        # 显示从开始到当前帧的轨迹
        line_p.set_data(pos_p_traj[:frame + 1, 0], pos_p_traj[:frame + 1, 1])
        line_p.set_3d_properties(pos_p_traj[:frame + 1, 2])
        line_e.set_data(pos_e_traj[:frame + 1, 0], pos_e_traj[:frame + 1, 1])
        line_e.set_3d_properties(pos_e_traj[:frame + 1, 2])
        # 当前点位置
        point_p.set_data([pos_p_traj[frame, 0]], [pos_p_traj[frame, 1]])
        point_p.set_3d_properties([pos_p_traj[frame, 2]])
        point_e.set_data([pos_e_traj[frame, 0]], [pos_e_traj[frame, 1]])
        point_e.set_3d_properties([pos_e_traj[frame, 2]])
        return line_p, line_e, point_p, point_e

    ani = FuncAnimation(fig, update, frames=T, init_func=init, blit=True, interval=50)
    ani.save(filename, writer=PillowWriter(fps=20))
    plt.close(fig)