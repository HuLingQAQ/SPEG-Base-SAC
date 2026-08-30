import numpy as np
from train import config
# -------------------------- 追逃环境 --------------------------
class PursuitEvasionEnv:
    """
    基于CW方程的追逃环境，使用绝对状态积分并施加速度限制。
    状态: 内部使用12维绝对状态 [pos_p(3), vel_p(3), pos_e(3), vel_e(3)]
    观测: 13维 [自身绝对位置归一化, 自身绝对速度归一化, 相对位置归一化, 相对速度归一化, 燃料剩余归一化]
    动作: 3维连续加速度 [ax, ay, az] (m/s²)
    追击星和逃逸星具体参数由配置文件给出
    燃料消耗: 每步消耗 = fuel_consumption_coef * ||a|| * dt
    """
    def __init__(self, dt, max_steps, capture_dist,
                 amax_p, amax_e, vmax_abs_p, vmax_abs_e,
                 fuel_init_p, fuel_init_e, fuel_consumption_coef,
                 n):
        self.n = config.N
        self.dt = config.DT
        self.max_steps = config.MAX_STEPS
        self.capture_dist = config.CAPTURE_DIST
        self.amax_p = config.AMAX_P
        self.amax_e = config.AMAX_E
        self.vmax_abs_p = config.VMAX_ABS_P
        self.vmax_abs_e = config.VMAX_ABS_E
        self.fuel_init_p = config.FUEL_INIT_P
        self.fuel_init_e = config.FUEL_INIT_E
        self.fuel_consumption_coef = config.FUEL_CONSUMPTION_COEF
        self.R_geo = config.R_GEO
        self.V_geo = self.R_geo * self.n
        self.state_dim = config.STATE_DIM
        self.action_dim = config.ACTION_DIM
        self.reset()

    def reset(self):
        # 随机初始相对位置（距离 4000~5000 m），速度为零
        r = np.random.uniform(4000, 5000)
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(2 * np.random.random() - 1)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        vx, vy, vz = 0.0, 0.0, 0.0
        self.rel_pos = np.array([x, y, z], dtype=np.float32)
        self.rel_vel = np.array([vx, vy, vz], dtype=np.float32)
        self.theta0 = np.random.uniform(0, 2*np.pi)

        angle = self.theta0
        self.pos_e = self.R_geo * np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)
        self.vel_e = self.V_geo * np.array([-np.sin(angle), np.cos(angle), 0.0], dtype=np.float32)

        self.pos_p = self.pos_e + self.rel_pos
        self.vel_p = self.vel_e + self.rel_vel
        self.fuel_p = self.fuel_init_p
        self.fuel_e = self.fuel_init_e
        self.time = 0.0
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        # 计算当前相对状态
        rel_pos = self.pos_p - self.pos_e
        rel_vel = self.vel_p - self.vel_e

        # 归一化尺度
        pos_scale = self.R_geo
        vel_scale = self.V_geo
        rel_pos_scale = 5000.0
        rel_vel_scale = max(self.vmax_abs_p, self.vmax_abs_e)  # 用较大的速度上限作为相对速度归一化尺度
        fuel_scale = self.fuel_init_p

        # 追击星观测
        obs_p = np.concatenate([
            self.pos_p / pos_scale,
            self.vel_p / vel_scale,
            rel_pos / rel_pos_scale,
            rel_vel / rel_vel_scale,
            [self.fuel_p / fuel_scale]
        ]).astype(np.float32)

        # 逃逸星观测
        obs_e = np.concatenate([
            self.pos_e / pos_scale,
            self.vel_e / vel_scale,
            -rel_pos / rel_pos_scale,
            -rel_vel / rel_vel_scale,
            [self.fuel_e / fuel_scale]
        ]).astype(np.float32)

        return obs_p, obs_e

    def _apply_speed_limit(self, vel, vmax):
        speed = np.linalg.norm(vel)
        if speed > vmax and vmax > 0:
            vel = vel / speed * vmax
        return vel

    def _rk4(self, rel_pos, rel_vel, u_p, u_e, dt):
        def f(s, up, ue):
            x, y, z, vx, vy, vz = s
            n = self.n
            ax = 2 * n * vy + 3 * n * n * x + up[0] - ue[0]
            ay = -2 * n * vx + up[1] - ue[1]
            az = -n * n * z + up[2] - ue[2]
            return np.array([vx, vy, vz, ax, ay, az])

        state = np.concatenate([rel_pos, rel_vel])
        k1 = f(state, u_p, u_e)
        k2 = f(state + 0.5 * dt * k1, u_p, u_e)
        k3 = f(state + 0.5 * dt * k2, u_p, u_e)
        k4 = f(state + dt * k3, u_p, u_e)
        new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return new_state

    def step(self, action_p, action_e):
        # 动作裁剪
        action_p = np.clip(action_p, -self.amax_p, self.amax_p)
        action_e = np.clip(action_e, -self.amax_e, self.amax_e)

        # 用RK4积分相对状态（基于CW方程）
        new_rel_state = self._rk4(self.rel_pos, self.rel_vel, action_p, action_e, self.dt)
        new_rel_pos = new_rel_state[:3]
        new_rel_vel_rk4 = new_rel_state[3:]

        # 更新绝对速度
        new_vel_p = self._apply_speed_limit(self.vel_p + action_p * self.dt, self.vmax_abs_p)
        new_vel_e = self._apply_speed_limit(self.vel_e + action_e * self.dt, self.vmax_abs_e)

        # 计算修正后的相对速度
        new_rel_vel = new_vel_p - new_vel_e

        # 更新绝对位置（梯形法）
        new_pos_e = self.pos_e + (self.vel_e + new_vel_e) * 0.5 * self.dt
        new_pos_p = new_pos_e + new_rel_pos

        # 更新内部状态
        self.pos_p = new_pos_p
        self.vel_p = new_vel_p
        self.pos_e = new_pos_e
        self.vel_e = new_vel_e
        self.rel_pos = new_rel_pos
        self.rel_vel = new_rel_vel

        # 燃料消耗（线性）
        consumption_p = self.fuel_consumption_coef * np.linalg.norm(action_p) * self.dt
        consumption_e = self.fuel_consumption_coef * np.linalg.norm(action_e) * self.dt
        self.fuel_p -= consumption_p
        self.fuel_e -= consumption_e
        self.fuel_p = max(self.fuel_p, 0.0)
        self.fuel_e = max(self.fuel_e, 0.0)
        self.time += self.dt

        # 终止判断
        dist = np.linalg.norm(self.rel_pos)
        capture = dist < self.capture_dist
        time_out = self.time >= self.max_steps * self.dt
        fuel_exhaust_p = self.fuel_p <= 0
        fuel_exhaust_e = self.fuel_e <= 0
        done = capture or time_out or fuel_exhaust_p

        # 计算奖励
        r_p, r_e = self._compute_reward(action_p, action_e, dist, capture,
                                        fuel_exhaust_p, fuel_exhaust_e, time_out, done)

        self.done = done
        self.info = {'捕获': capture, '距离': dist, '时间': self.time,
                     '追击星燃料耗尽判断': fuel_exhaust_p, '逃逸星燃料耗尽判断': fuel_exhaust_e,
                     '超时判断': time_out}
        obs_p, obs_e = self._get_obs()
        return (obs_p, obs_e), (r_p, r_e), done, self.info

    # 奖励函数
    def _compute_reward(self, action_p, action_e, dist, capture,
                        fuel_exhaust_p, fuel_exhaust_e, time_out, done_flag):
        r_p = 0.0
        r_e = 0.0

        # 奖励系数 k2
        distance_coef = 0.01
        # 距离惩罚项：R_dist = k1 * ( k2 * dist - 1 / k2 * dist )
        # k1 = 0.1
        r_p -= 0.1 * ((distance_coef * dist) - (1 / (distance_coef * dist)))
        r_e += 0.1 * ((distance_coef * dist) - (1 / (distance_coef * dist)))
        # 终端奖励
        if done_flag:
            # 燃料剩余激励：R_fuel = μ * fuel
            # μ = 0.1
            r_p += 0.1 * self.fuel_p
            r_e += 0.1 * self.fuel_e
            # 捕获奖励：R_final = C
            # C = 2000.0
            if capture:
                r_p += 2000.0
                r_e -= 2000.0
            else:
                # 任何失败（燃料耗尽或超时）双方惩罚
                r_p -= 0.0
                r_e -= 0.0

        return r_p, r_e