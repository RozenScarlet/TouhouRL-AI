# PPO.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from torch.distributions import Categorical
import torch.optim as optim
from typing import Dict, Tuple
import os

class MultiInputCNNExtractor(nn.Module):
    """多输入CNN特征提取器，融合图像、危险图和特征向量"""
    
    def __init__(self):
        super(MultiInputCNNExtractor, self).__init__()
        
        # 处理游戏帧的CNN
        self.frame_cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 处理危险图的CNN（单通道输入）
        self.danger_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 处理额外特征的MLP
        self.feature_mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # 计算各分支的输出维度
        self.frame_output_dim = self._get_frame_output_dim()
        self.danger_output_dim = self._get_danger_output_dim()
        self.feature_output_dim = 128
        
        # 融合层
        total_dim = self.frame_output_dim + self.danger_output_dim + self.feature_output_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        
    def _get_frame_output_dim(self):
        """计算帧CNN的输出维度"""
        dummy_input = torch.zeros(1, 4, 96, 128)
        output = self.frame_cnn(dummy_input)
        return output.shape[1]
    
    def _get_danger_output_dim(self):
        """计算危险图CNN的输出维度"""
        dummy_input = torch.zeros(1, 1, 96, 128)
        output = self.danger_cnn(dummy_input)
        return output.shape[1]
    
    def forward(self, frames, danger_map, features):
        # 处理各个输入
        frame_features = self.frame_cnn(frames)
        danger_features = self.danger_cnn(danger_map)
        extra_features = self.feature_mlp(features)
        
        # 融合特征
        combined = torch.cat([frame_features, danger_features, extra_features], dim=1)
        fused_features = self.fusion_layer(combined)
        
        return fused_features

class ImprovedPolicyNetwork(nn.Module):
    """改进的策略网络，使用多输入"""
    
    def __init__(self, n_actions):
        super(ImprovedPolicyNetwork, self).__init__()
        
        self.feature_extractor = MultiInputCNNExtractor()
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        
        # 添加一个紧急避险模块（基于危险特征的快速决策）
        self.emergency_head = nn.Sequential(
            nn.Linear(4, 32),  # 直接从特征向量
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )
        
    def forward(self, state_dict):
        frames = state_dict['frames']
        danger_map = state_dict['danger_map']
        features = state_dict['features']
        
        # 提取融合特征
        fused_features = self.feature_extractor(frames, danger_map, features)
        
        # 常规策略
        policy_logits = self.policy_head(fused_features)
        
        # 紧急避险策略（当子弹很近时权重更高）
        emergency_logits = self.emergency_head(features)
        
        # 改进的危险程度动态融合策略
        min_bullet_dist = features[:, 2:3]  # shape (batch, 1)

        # 使用更合理的危险权重计算
        # 当距离小于30像素时，紧急避险权重显著增加
        danger_threshold = 30.0
        normalized_dist = torch.clamp(min_bullet_dist / danger_threshold, 0, 1)

        # 使用sigmoid函数创建更平滑的权重过渡
        danger_weight = torch.sigmoid(-10 * (normalized_dist - 0.5))

        # 融合logits - 在危险时更多依赖紧急避险
        final_logits = (1 - danger_weight) * policy_logits + danger_weight * emergency_logits
        
        return F.softmax(final_logits, dim=-1)

class ImprovedValueNetwork(nn.Module):
    """改进的价值网络"""
    
    def __init__(self):
        super(ImprovedValueNetwork, self).__init__()
        
        self.feature_extractor = MultiInputCNNExtractor()
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, state_dict):
        frames = state_dict['frames']
        danger_map = state_dict['danger_map']
        features = state_dict['features']
        
        # 提取融合特征
        fused_features = self.feature_extractor(frames, danger_map, features)
        
        # 计算价值
        value = self.value_head(fused_features)
        
        return value.squeeze()

class ImprovedPPOBuffer:
    """改进的PPO缓冲区，存储字典状态"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.clear()
        
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []
        
    def store(self, state_dict, action, reward, value, log_prob, done):
        # 深拷贝状态字典
        state_copy = {
            'frames': state_dict['frames'].copy(),
            'danger_map': state_dict['danger_map'].copy(),
            'features': state_dict['features'].copy()
        }
        
        self.states.append(state_copy)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
    def finish_path(self, last_value=0):
        """计算优势和回报"""
        rewards = self.rewards + [last_value]
        values = self.values + [last_value]
        
        # 计算GAE优势
        gamma = 0.99
        gae_lambda = 0.95
        
        advantages = []
        gae = 0
        for i in reversed(range(len(self.rewards))):
            delta = rewards[i] + gamma * values[i+1] * (1 - self.dones[i]) - values[i]
            gae = delta + gamma * gae_lambda * (1 - self.dones[i]) * gae
            advantages.insert(0, gae)
            
        self.advantages = advantages
        
        # 计算回报
        self.returns = [adv + val for adv, val in zip(advantages, self.values)]
        
    def get(self):
        """获取所有数据"""
        # 标准化优势
        advantages = np.array(self.advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return (
            self.states,  # 保持字典格式
            np.array(self.actions),
            np.array(self.log_probs),
            advantages,
            np.array(self.returns)
        )
        
    def size(self):
        return len(self.states)

class ImprovedPPOAgent:
    """改进的PPO智能体，适配新的状态表示"""
    
    def __init__(self, n_actions, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.n_actions = n_actions
        
        # 创建改进的网络
        self.policy_net = ImprovedPolicyNetwork(n_actions).to(device)
        self.value_net = ImprovedValueNetwork().to(device)
        
        # 优化器 - 使用更高的学习率
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=5e-4)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=5e-4)

        # 学习率调度器 - 使用更温和的调度策略
        # 使用StepLR，更适合游戏AI训练
        self.policy_scheduler = optim.lr_scheduler.StepLR(
            self.policy_optimizer, step_size=3000, gamma=0.9
        )
        self.value_scheduler = optim.lr_scheduler.StepLR(
            self.value_optimizer, step_size=3000, gamma=0.9
        )

        # PPO参数 - 优化训练稳定性
        self.clip_ratio = 0.25  # 稍微增加裁剪比例，允许更大的策略更新
        self.target_kl = 0.02   # 增加KL散度阈值，允许更多探索
        self.train_policy_iters = 30  # 减少训练迭代次数，避免过拟合
        self.train_value_iters = 30

        # 使用更大的缓冲区，适合高频决策任务
        self.buffer = ImprovedPPOBuffer(capacity=4096)
        
        # 性能跟踪
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.survival_times = deque(maxlen=100)
        
    def get_action(self, state_dict):
        """获取动作（适配字典状态）"""
        # 将状态转换为张量
        state_tensors = {
            'frames': torch.FloatTensor(state_dict['frames']).unsqueeze(0).to(self.device),
            'danger_map': torch.FloatTensor(state_dict['danger_map']).unsqueeze(0).to(self.device),
            'features': torch.FloatTensor(state_dict['features']).unsqueeze(0).to(self.device)
        }
        
        with torch.no_grad():
            # 获取策略
            probs = self.policy_net(state_tensors)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # 获取价值
            value = self.value_net(state_tensors)
            
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        """存储转换"""
        self.buffer.store(state, action, reward, value, log_prob, done)
    
    def update(self):
        """更新网络（适配新的状态格式）"""
        if self.buffer.size() < 1024:
            return None, None
            
        # 获取数据
        state_dicts, actions, old_log_probs, advantages, returns = self.buffer.get()
        
        # 转换状态为张量
        frames = torch.FloatTensor(np.array([s['frames'] for s in state_dicts])).to(self.device)
        danger_maps = torch.FloatTensor(np.array([s['danger_map'] for s in state_dicts])).to(self.device)
        features = torch.FloatTensor(np.array([s['features'] for s in state_dicts])).to(self.device)
        
        state_tensors = {
            'frames': frames,
            'danger_map': danger_maps,
            'features': features
        }
        
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 更新策略网络
        policy_losses = []
        for _ in range(self.train_policy_iters):
            # 计算新的概率
            probs = self.policy_net(state_tensors)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            
            # PPO损失
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 熵正则化
            entropy_loss = -dist.entropy().mean()
            total_loss = policy_loss + 0.01 * entropy_loss
            
            # 早停检查
            kl = (old_log_probs - new_log_probs).mean()
            if kl > 1.5 * self.target_kl:
                break
            
            # 优化
            self.policy_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()
            
            policy_losses.append(policy_loss.item())
        
        # 获取旧的价值估计（用于价值函数裁剪）
        with torch.no_grad():
            old_values = self.value_net(state_tensors).detach()

        # 更新价值网络 - 添加价值函数裁剪
        value_losses = []
        for _ in range(self.train_value_iters):
            values = self.value_net(state_tensors)

            # PPO价值函数裁剪
            values_clipped = old_values + torch.clamp(
                values - old_values, -self.clip_ratio, self.clip_ratio
            )

            # 计算两种损失
            value_loss_1 = F.mse_loss(values, returns)
            value_loss_2 = F.mse_loss(values_clipped, returns)

            # 取最大值（更保守的更新）
            value_loss = torch.max(value_loss_1, value_loss_2)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()

            value_losses.append(value_loss.item())
        
        # 更新学习率
        self.policy_scheduler.step()
        self.value_scheduler.step()
        
        # 清空缓冲区
        self.buffer.clear()
        
        return np.mean(policy_losses), np.mean(value_losses)
    
    def finish_episode(self, last_state=None):
        """结束一个episode"""
        if last_state is not None:
            # 将状态转换为张量
            state_tensors = {
                'frames': torch.FloatTensor(last_state['frames']).unsqueeze(0).to(self.device),
                'danger_map': torch.FloatTensor(last_state['danger_map']).unsqueeze(0).to(self.device),
                'features': torch.FloatTensor(last_state['features']).unsqueeze(0).to(self.device)
            }
            with torch.no_grad():
                last_value = self.value_net(state_tensors).item()
        else:
            last_value = 0
            
        self.buffer.finish_path(last_value)
    
    def save(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'policy_scheduler_state_dict': self.policy_scheduler.state_dict(),
            'value_scheduler_state_dict': self.value_scheduler.state_dict(),
        }, path)
        
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        if 'policy_scheduler_state_dict' in checkpoint:
            self.policy_scheduler.load_state_dict(checkpoint['policy_scheduler_state_dict'])
            self.value_scheduler.load_state_dict(checkpoint['value_scheduler_state_dict'])
    
    def add_episode_reward(self, reward):
        """添加episode奖励用于跟踪"""
        self.episode_rewards.append(reward)
        
    def get_average_reward(self):
        """获取平均奖励"""
        if len(self.episode_rewards) == 0:
            return 0
        return np.mean(self.episode_rewards)
    
    def add_episode_length(self, length):
        """添加episode长度"""
        self.episode_lengths.append(length)
    
    def add_survival_time(self, time):
        """添加生存时间"""
        self.survival_times.append(time)
    
    def get_average_episode_length(self):
        """获取平均episode长度"""
        if len(self.episode_lengths) == 0:
            return 0
        return np.mean(self.episode_lengths)
    
    def get_average_survival_time(self):
        """获取平均生存时间"""
        if len(self.survival_times) == 0:
            return 0
        return np.mean(self.survival_times)