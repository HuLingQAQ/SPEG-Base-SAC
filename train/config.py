# --------------------------- 基本参数列表 ---------------------------
# 环境参数
R_GEO = 42164000.0            # 地球同步轨道半径（m）
N = 7.292e-5                  # 地球自转角速度(rad/s)

# 航天器参数
DT = 1.0                      # 时间步长
MAX_STEPS = 8000              # 最大时间步数(s)
CAPTURE_DIST = 100.0          # 捕获距离(m)
AMAX_P = 1.5                  # 追击星最大控制加速度(m/s^2)
AMAX_E = 1.2                  # 逃逸星最大控制加速度(m/s^2)
VMAX_ABS_P = 75.0             # 追击星速度增量上限(m/s)
VMAX_ABS_E = 60.0             # 逃逸星速度增量上限(m/s)
FUEL_INIT_P = 1000            # 追击星初始燃料
FUEL_INIT_E = 1000            # 逃逸星初始燃料
FUEL_CONSUMPTION_COEF = 1.0   # 燃料消耗系数：每步消耗 = coef * ||a|| * dt

# SAC超参数
STATE_DIM = 13                # 状态维度： 13维 (绝对位置，绝对速度，相对位置，相对速度，燃料消耗)
ACTION_DIM = 3                # 动作维度：3维连续加速度 [ax, ay, az] (m/s²)
LEARNING_RATE = 3e-4          # 学习率
GAMMA = 0.99                  # 折扣因子 γ
TAU = 0.005                   # 软更新参数（用于更新目标网络的权重。 tau越小，目标网络更新得越快）
BATCH_SIZE = 256              # 批次大小
ALPHA_LR = 3e-4               # 温度系数 α 的学习率
QNETWORK_LR = 3e-4            # Q网络的学习率
POLICY_LR = 3e-4              # 策略网络的学习率
TARGET_ENTROPY = -ACTION_DIM  # 目标熵，直接取动作维度的负值