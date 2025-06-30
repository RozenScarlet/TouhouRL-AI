# improved_train.py
"""
改进的训练脚本 - 集成监控和诊断功能
"""

import numpy as np
import torch
import time
import os
import sys
from training_monitor import TrainingMonitor

from env import ImprovedTouhouEnv
from PPO import ImprovedPPOAgent

def train_with_monitoring(model_path=None, force_restart=False):
    """带监控的训练函数"""
    
    # 创建监控器
    monitor = TrainingMonitor()
    
    # 创建project_player/models目录
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 创建环境和智能体
    env = ImprovedTouhouEnv()
    env.debug_rewards = True  # 启用奖励调试
    agent = ImprovedPPOAgent(n_actions=9)

    n_episodes = 10000
    save_interval = 10
    monitor_interval = 5  # 每5轮生成一次监控报告
    best_reward = float('-inf')
    start_episode = 0

    # 模型加载逻辑
    if not force_restart and model_path and os.path.exists(model_path):
        try:
            agent.load(model_path)
            print(f"✅ 成功加载模型: {model_path}")
            # 从文件名提取episode数
            if "ppo_" in model_path:
                try:
                    start_episode = int(model_path.split("ppo_")[1].split(".")[0])
                    print(f"从Episode {start_episode}继续训练")
                except:
                    pass
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            print("将从头开始训练")

    print(f"\n开始训练，目标轮数: {n_episodes}")
    print(f"保存间隔: {save_interval} episodes")
    print(f"监控间隔: {monitor_interval} episodes")
    print("-" * 60)

    # 训练循环
    for episode in range(start_episode, n_episodes):
        episode_start_time = time.time()
        
        print(f"\n🎮 开始Episode {episode + 1}")
        
        state = env.reset()
        total_reward = 0
        steps = 0
        policy_loss = None
        value_loss = None
        
        # Episode数据收集
        episode_bullet_distances = []
        episode_rewards_breakdown = []
        
        done = False
        while not done:
            # PPO获取动作和价值
            action, log_prob, value = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            # 收集监控数据
            if 'min_bullet_dist' in info:
                episode_bullet_distances.append(info['min_bullet_dist'])
                monitor.log_step({'bullet_distance': info['min_bullet_dist']})
            
            if hasattr(env, 'last_reward_breakdown'):
                episode_rewards_breakdown.append(env.last_reward_breakdown.copy())

            # 存储转换到PPO缓冲区
            agent.store_transition(state, action, reward, log_prob, value, done)

            state = next_state
            total_reward += reward
            steps += 1

            # 实时显示状态（减少频率避免刷屏）
            if steps % 100 == 0:
                print(f"\r步数: {steps} | 奖励: {total_reward:.2f} | 缓冲区: {agent.buffer.size()}", end="")

            if done:
                episode_end_time = time.time()
                survival_time = episode_end_time - episode_start_time
                
                print(f"\n游戏结束，原因: {'生命耗尽' if info['lives'] == 0 else '其他原因'}")
                print(f"存活时间: {survival_time:.1f}秒")
                
                # 结束episode，计算优势和回报
                agent.finish_episode(next_state if not done else None)
                
                # 尝试更新网络
                policy_loss, value_loss = agent.update()
                
                # 添加episode奖励用于跟踪
                agent.add_episode_reward(total_reward)
                
                # 记录监控数据
                episode_data = {
                    'total_reward': total_reward,
                    'steps': steps,
                    'survival_time': survival_time,
                    'death_reason': 'bullet_collision' if info['lives'] == 0 else 'other'
                }
                
                # 添加奖励分解统计
                if episode_rewards_breakdown:
                    avg_breakdown = {}
                    for key in episode_rewards_breakdown[0].keys():
                        avg_breakdown[key] = np.mean([rb[key] for rb in episode_rewards_breakdown])
                    episode_data['reward_breakdown'] = avg_breakdown
                
                monitor.log_episode(episode_data)
                
                state = env.reset()
                break

        # 定期保存和监控
        if (episode + 1) % save_interval == 0:
            save_path = os.path.join(models_dir, f"ppo_{episode+1}.pth")
            agent.save(save_path)
            print(f"💾 已保存模型: {save_path}")

        # 定期生成监控报告
        if (episode + 1) % monitor_interval == 0:
            monitor.print_status(episode + 1)
            report = monitor.generate_report(episode + 1)
            monitor.plot_training_curves()
            
            # 检查是否有改进
            if report and report['performance']['avg_reward_last_100'] > best_reward:
                best_reward = report['performance']['avg_reward_last_100']
                best_save_path = os.path.join(models_dir, f"ppo_best_{episode+1}.pth")
                agent.save(best_save_path)
                print(f"🏆 新的最佳模型! 平均奖励: {best_reward:.2f}")

        # 打印episode信息
        print(f"\n📊 Episode {episode+1} 完成:")
        print(f"总步数: {steps}")
        print(f"总奖励: {total_reward:.2f}")
        if policy_loss is not None:
            print(f"策略损失: {policy_loss:.4f}")
            print(f"价值损失: {value_loss:.4f}")
        print(f"平均奖励: {agent.get_average_reward():.2f}")
        print(f"分数: {info['score']}")
        print(f"剩余生命: {info['lives']}")
        
        if episode_bullet_distances:
            avg_bullet_dist = np.mean(episode_bullet_distances)
            min_bullet_dist = min(episode_bullet_distances)
            print(f"平均子弹距离: {avg_bullet_dist:.1f}")
            print(f"最小子弹距离: {min_bullet_dist:.1f}")
            
        print("-" * 50)

    # 训练结束
    final_save_path = os.path.join(models_dir, "ppo_final.pth")
    agent.save(final_save_path)
    
    # 生成最终报告
    final_report = monitor.generate_report(n_episodes)
    monitor.plot_training_curves(os.path.join(models_dir, "final_training_curves.png"))
    
    env.close()
    print(f"\n🎉 训练完成!")
    print(f"💾 最终模型已保存: {final_save_path}")
    print(f"📈 训练曲线已保存到models目录")
    
    if final_report:
        print(f"\n📊 最终训练统计:")
        print(f"平均奖励: {final_report['performance']['avg_reward_last_100']:.2f}")
        print(f"平均存活时间: {final_report['performance']['avg_survival_time']:.1f}秒")
        print(f"最高奖励: {final_report['performance']['max_reward']:.2f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='改进的东方AI训练脚本')
    parser.add_argument('--model', type=str, help='要加载的模型路径')
    parser.add_argument('--restart', action='store_true', help='强制重新开始训练')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 改进的东方红魔乡AI训练系统")
    print("=" * 60)
    print("主要改进:")
    print("✅ 优化奖励函数 - 更平衡的奖励机制")
    print("✅ 改进死亡惩罚 - 基于存活时间的动态奖励")
    print("✅ 增强子弹检测 - 多阈值检测算法")
    print("✅ 实时监控 - 详细的训练状态监控")
    print("✅ 调试功能 - 奖励分解和性能分析")
    print("=" * 60)
    
    train_with_monitoring(model_path=args.model, force_restart=args.restart)
