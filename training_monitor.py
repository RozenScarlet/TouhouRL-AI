# training_monitor.py
"""
训练监控脚本 - 实时监控AI训练状态和性能
"""

import os
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import json

class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, log_dir="training_logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 性能指标
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.survival_times = deque(maxlen=1000)
        self.death_reasons = deque(maxlen=100)
        self.bullet_distances = deque(maxlen=10000)
        
        # 奖励分解统计
        self.reward_components = {
            'score': deque(maxlen=1000),
            'survival': deque(maxlen=1000),
            'danger': deque(maxlen=1000),
            'position': deque(maxlen=1000),
            'dodge': deque(maxlen=1000)
        }
        
    def log_episode(self, episode_data):
        """记录一个episode的数据"""
        self.episode_rewards.append(episode_data['total_reward'])
        self.episode_lengths.append(episode_data['steps'])
        self.survival_times.append(episode_data['survival_time'])
        
        if 'death_reason' in episode_data:
            self.death_reasons.append(episode_data['death_reason'])
            
        # 记录奖励分解
        if 'reward_breakdown' in episode_data:
            for component, value in episode_data['reward_breakdown'].items():
                if component in self.reward_components:
                    self.reward_components[component].append(value)
    
    def log_step(self, step_data):
        """记录单步数据"""
        if 'bullet_distance' in step_data:
            self.bullet_distances.append(step_data['bullet_distance'])
    
    def generate_report(self, episode_num):
        """生成训练报告"""
        if len(self.episode_rewards) == 0:
            return
            
        report = {
            'episode': episode_num,
            'timestamp': time.time(),
            'performance': {
                'avg_reward_last_100': np.mean(list(self.episode_rewards)[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
                'avg_survival_time': np.mean(self.survival_times) if self.survival_times else 0,
                'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
                'max_reward': max(self.episode_rewards) if self.episode_rewards else 0,
                'min_bullet_distance_avg': np.mean(self.bullet_distances) if self.bullet_distances else 0
            },
            'death_analysis': {
                'total_deaths': len(self.death_reasons),
                'death_rate': len(self.death_reasons) / max(len(self.episode_rewards), 1)
            }
        }
        
        # 保存报告
        report_path = os.path.join(self.log_dir, f"report_episode_{episode_num}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        return report
    
    def plot_training_curves(self, save_path=None):
        """绘制训练曲线"""
        if len(self.episode_rewards) < 10:
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('AI训练监控面板', fontsize=16)
        
        # 1. 奖励曲线
        axes[0, 0].plot(self.episode_rewards, alpha=0.7, label='Episode奖励')
        if len(self.episode_rewards) >= 10:
            # 计算移动平均
            window = min(50, len(self.episode_rewards) // 4)
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(self.episode_rewards)), moving_avg, 
                           color='red', linewidth=2, label=f'{window}期移动平均')
        axes[0, 0].set_title('训练奖励趋势')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('总奖励')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 存活时间
        axes[0, 1].plot(self.survival_times, color='green', alpha=0.7)
        axes[0, 1].set_title('存活时间趋势')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('存活时间(秒)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Episode长度
        axes[0, 2].plot(self.episode_lengths, color='orange', alpha=0.7)
        axes[0, 2].set_title('Episode长度趋势')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('步数')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 奖励分解
        for i, (component, values) in enumerate(self.reward_components.items()):
            if values and i < 3:  # 只显示前3个组件
                axes[1, i].plot(values, alpha=0.7, label=component)
                axes[1, i].set_title(f'{component}奖励')
                axes[1, i].set_xlabel('Episode')
                axes[1, i].set_ylabel('奖励值')
                axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"训练曲线已保存到: {save_path}")
        else:
            plt.savefig(os.path.join(self.log_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def print_status(self, episode_num):
        """打印当前训练状态"""
        if len(self.episode_rewards) == 0:
            return
            
        recent_rewards = list(self.episode_rewards)[-10:]
        recent_survival = list(self.survival_times)[-10:]
        
        print(f"\n{'='*60}")
        print(f"训练状态报告 - Episode {episode_num}")
        print(f"{'='*60}")
        print(f"最近10轮平均奖励: {np.mean(recent_rewards):.2f}")
        print(f"最近10轮平均存活时间: {np.mean(recent_survival):.1f}秒")
        print(f"历史最高奖励: {max(self.episode_rewards):.2f}")
        print(f"总训练轮数: {len(self.episode_rewards)}")
        
        if self.bullet_distances:
            recent_distances = list(self.bullet_distances)[-1000:]
            print(f"最近1000步平均子弹距离: {np.mean(recent_distances):.1f}像素")
            dangerous_steps = sum(1 for d in recent_distances if d < 30)
            print(f"危险步数比例: {dangerous_steps/len(recent_distances)*100:.1f}%")
        
        print(f"{'='*60}\n")

def main():
    """测试监控器"""
    monitor = TrainingMonitor()
    
    # 模拟一些训练数据
    for i in range(100):
        episode_data = {
            'total_reward': np.random.normal(50, 20),
            'steps': np.random.randint(100, 1000),
            'survival_time': np.random.uniform(10, 120),
            'reward_breakdown': {
                'score': np.random.normal(10, 5),
                'survival': np.random.normal(5, 2),
                'danger': np.random.normal(-5, 3),
                'position': np.random.normal(0, 1),
                'dodge': np.random.normal(2, 1)
            }
        }
        monitor.log_episode(episode_data)
        
        if i % 20 == 0:
            monitor.print_status(i)
            monitor.generate_report(i)
    
    monitor.plot_training_curves()
    print("监控器测试完成！")

if __name__ == "__main__":
    main()
