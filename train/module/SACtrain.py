import numpy as np
# -------------------------- 训练函数 --------------------------
def train(env, pursuer_agent, evader_agent, num_episodes, update_after=200, update_every=1):
    pursuer_rewards = []
    evader_rewards = []
    capture_rate = []

    q1_losses = []
    q2_losses = []
    actor_losses = []
    alpha_losses = []

    for episode in range(1, num_episodes+1):
            obs_p, obs_e = env.reset()
            ep_p_reward = 0
            ep_e_reward = 0
            step = 0
            captured = False

            while True:
                action_p = pursuer_agent.select_action(obs_p)
                action_e = evader_agent.select_action(obs_e)

                (next_obs_p, next_obs_e), (r_p, r_e), done, info = env.step(action_p, action_e)

                pursuer_agent.replay_buffer.push(obs_p, action_p, r_p, next_obs_p, done)
                evader_agent.replay_buffer.push(obs_e, action_e, r_e, next_obs_e, done)

                obs_p, obs_e = next_obs_p, next_obs_e
                ep_p_reward += r_p
                ep_e_reward += r_e
                step += 1

                if len(pursuer_agent.replay_buffer) > update_after and step % update_every == 0:
                    for _ in range(update_every):
                        loss_p = pursuer_agent.update()
                        loss_e = evader_agent.update()
                        if loss_p is not None:
                            q1l_p, q2l_p ,al_p, alphal_p = loss_p
                            q1l_e, q2l_e ,al_e, alphal_e = loss_e
                            # 记录两个智能体的平均损失（或分别记录）
                            q1_losses.append((q1l_p + q1l_e) / 2)
                            q2_losses.append((q2l_p + q2l_e) / 2)
                            actor_losses.append((al_p + al_e) / 2)
                            alpha_losses.append((alphal_p + alphal_e) / 2)
                if done:
                    captured = info['捕获']
                    break

            pursuer_rewards.append(ep_p_reward)
            evader_rewards.append(ep_e_reward)
            capture_rate.append(1 if captured else 0)

            if episode % 20 == 0:
                avg_cap = np.mean(capture_rate[-20:])
                print(f"Episode {episode}, 追击星奖励: {ep_p_reward:.2f}, 逃逸星奖励: {ep_e_reward:.2f}, "
                      f"平均捕获率 (last20): {avg_cap:.2f}, 终端距离: {info['距离']:.2f}, 终止原因: {info}")

    return pursuer_rewards, evader_rewards, capture_rate, q1_losses, q2_losses, actor_losses, alpha_losses
