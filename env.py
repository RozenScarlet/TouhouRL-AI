# env.py
from touhou_rl import TouhouEnv
from yolo_cap import ImprovedGameCapture
import numpy as np
import cv2
import keyboard
import time
import os
from typing import Dict, Tuple
import subprocess
import win32gui

class ImprovedTouhouEnv:
    """改进的东方环境，使用组合而非继承"""
    
    def __init__(self, game_path="D:\\Games\\th06\\vpatch.exe"):
        # 基础属性
        self.game_path = game_path
        self.window_name = "搶曽峠杺嫿丂乣 the Embodiment of Scarlet Devil"
        
        # 使用改进的捕获类
        self.cap = ImprovedGameCapture(window_name=self.window_name)
        
        # 帧堆叠
        self.frame_stack = 4
        self.frames = []
        
        # 动作空间
        self.action_space = 9   # 8方向 + 不动
        
        # 扩展观察空间：4帧图像 + 1帧危险图 + 额外特征
        self.observation_space = {
            'frames': (4, 96, 128),
            'danger_map': (1, 96, 128),
            'features': 4
        }
        
        # 游戏状态
        self.last_score = 0
        self.start_time = None
        self.max_steps = 10000
        self.current_step = 0
        self.first_reset = True

        # 射击控制
        self.last_shoot_time = 0
        self.shoot_interval = 0.02  # 20ms间隔超快速射击（50次/秒）
        
        # 动作映射：基本8方向移动
        self.action_map = {
            0: [],                      # 不动
            1: ['up'],                  # 上
            2: ['down'],                # 下
            3: ['left'],                # 左
            4: ['right'],               # 右
            5: ['up','left'],           # 左上
            6: ['up','right'],          # 右上
            7: ['down','left'],         # 左下
            8: ['down','right']         # 右下
        }
        
        # 记录历史信息用于奖励计算
        self.prev_min_bullet_dist = 1000.0  # 使用像素距离
        self.prev_player_pos = (0.5, 0.8)  # 默认在底部中间
        self.episode_max_score = 0
        
        # 先启动游戏，再初始化截屏类
        self._open_game()
        self.cap.set_window_by_name()
        
        # 初始化 frame stack
        for _ in range(self.frame_stack):
            self.frames.append(np.zeros((96, 128), dtype=np.uint8))
    
    def _open_game(self):
        """启动游戏，并等待窗口出现"""
        print(f"正在启动游戏: {self.game_path}")
        subprocess.Popen(f'"{self.game_path}"', shell=True)
        time.sleep(3)  # 等待游戏窗口初始化

        hwnd = win32gui.FindWindow(None, self.window_name)
        if hwnd != 0:
            print("游戏窗口已找到!")
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(1)
        else:
            print("未找到游戏窗口，请检查游戏是否正常启动")

    def _start_game(self):
        """首次重置时自动按键进入游戏的简易示例"""
        print("等待游戏界面加载完成...4s")
        time.sleep(4)

        print("开始按键序列...")
        # 连续按Z键试图跳过菜单
        for i in range(5):
            keyboard.press('z')
            time.sleep(0.2)
            keyboard.release('z')
            print(f"按Z键 {i+1}/5")
            time.sleep(0.5)

        print("游戏开始按键序列已执行完成")
        # 开始快速射击模式
        self.last_shoot_time = time.time()

    def _restart_game(self):
        """游戏结束后重新开始的按键流程示例"""
        print("\n------ 游戏重新开始 ------")
        try:
            # 先按ESC，返回主菜单
            keyboard.press('esc')
            time.sleep(0.2)
            keyboard.release('esc')
            print("按ESC键返回主菜单")
            time.sleep(0.5)

            # 按下键选择游戏模式
            keyboard.press('down')
            time.sleep(0.2)
            keyboard.release('down')
            print("按下键")
            time.sleep(0.5)

            # 按Z键确认
            keyboard.press('z')
            time.sleep(0.2)
            keyboard.release('z')
            time.sleep(0.5)

            keyboard.press('up')
            time.sleep(0.2)
            keyboard.release('up')
            time.sleep(0.5)

            keyboard.press('z')
            time.sleep(0.2)
            keyboard.release('z')
            time.sleep(2)

            # 再按几次Z进入游戏
            for i in range(6):
                keyboard.press('z')
                time.sleep(0.2)
                keyboard.release('z')
                print(f"按Z键 {i+1}/5")
                time.sleep(0.5)
            print("游戏重新开始完成，开始快速射击")
            # 重新开始后也要开始快速射击
            self.last_shoot_time = time.time()

            return True
        except Exception as e:
            print(f"重启游戏时出现异常: {e}")
            return False
    
    def reset(self):
        """重置环境"""
        print("\n------ 重置游戏环境 ------")
        # 释放按键（不需要释放Z键，因为我们使用快速按键模式）
        for action_keys in self.action_map.values():
            for k in action_keys:
                keyboard.release(k)

        # 初始化状态
        self.frames = []
        for _ in range(self.frame_stack):
            self.frames.append(np.zeros((96, 128), dtype=np.uint8))
        self.last_score = 0
        self.start_time = time.time()
        self.current_step = 0
        
        # 重置历史信息
        self.prev_min_bullet_dist = 1000.0  # 使用像素距离
        self.prev_player_pos = (0.5, 0.8)
        self.episode_max_score = 0

        # 重置射击时间
        self.last_shoot_time = 0

        # 首次reset vs 后续reset
        if self.first_reset:
            print("首次重置，启动游戏...")
            self._start_game()
            self.first_reset = False
        else:
            print("再次重置，重启游戏...")
            self._restart_game()

        # 等待1秒让游戏状态稳定，否则分数和生命会是0
        print("等待游戏状态稳定...1s")
        time.sleep(1)

        # 获取若干帧(初始化帧堆叠)
        print("获取初始游戏状态...")
        for _ in range(self.frame_stack):
            info = self.cap.capture_frame_with_detection()
            print(f"游戏状态: 分数={info['score']}, 生命={info['lives']}")
            self.frames.append(info['frame'])
            self.frames.pop(0)

        print("环境重置完成，准备开始训练")
        return self._get_enhanced_state()
    
    def step(self, action):
        """执行动作并返回增强的状态和奖励"""
        self.current_step += 1

        # 快速射击逻辑
        current_time = time.time()
        if current_time - self.last_shoot_time >= self.shoot_interval:
            keyboard.press('z')
            time.sleep(0.01)  # 短暂按下
            keyboard.release('z')
            self.last_shoot_time = current_time

        # 执行动作
        for k in self.action_map[action]:
            keyboard.press(k)
        time.sleep(1/60)
        
        # 获取增强的游戏信息
        info = self.cap.capture_frame_with_detection()
        
        # 更新帧堆栈
        self.frames.append(info['frame'])
        self.frames.pop(0)
        
        # 释放按键
        for k in self.action_map[action]:
            keyboard.release(k)
        
        # 计算增强奖励
        reward = self._calculate_enhanced_reward(info, action)
        
        # 检查游戏是否结束
        done = False
        if info['lives'] == 0 and self.current_step > 100:
            done = True
            # 改进的死亡惩罚：更合理的惩罚机制
            base_death_penalty = -50  # 减少基础死亡惩罚

            # 基于存活时间的动态奖励/惩罚
            if self.current_step > 2000:  # 存活超过2000步给予奖励
                survival_bonus = min((self.current_step - 2000) / 1000 * 10, 20)  # 最多+20奖励
                reward += survival_bonus
                print(f"长时间存活奖励: +{survival_bonus:.1f}")

            # 基于最高分数的奖励
            if self.episode_max_score > 0:
                score_bonus = min(self.episode_max_score * 0.001, 10)  # 基于分数的奖励
                reward += score_bonus
                print(f"分数奖励: +{score_bonus:.1f}")

            reward += base_death_penalty
            print(f"死亡惩罚: {base_death_penalty}, 最终奖励: {reward:.1f}")
        
        # 更新历史信息
        self.prev_min_bullet_dist = info['min_bullet_dist']
        self.prev_player_pos = info['player_pos']
        self.last_score = info['score']
        self.episode_max_score = max(self.episode_max_score, info['score'])
        
        # 构建状态
        state = self._get_enhanced_state_from_info(info)
        
        # 打印调试信息
        if self.current_step % 50 == 0:
            print(f"步数:{self.current_step}, 分数:{info['score']}, 生命:{info['lives']}, 最近子弹距离:{info['min_bullet_dist']:.3f}, 子弹数量:{info['bullet_count']}, 自机位置:({info['player_pos'][0]:.2f},{info['player_pos'][1]:.2f}), 奖励:{reward:.3f}")

        # 增强调试信息输出
        if self.current_step % 100 == 0:
            print(f"\n=== 步数 {self.current_step} 调试信息 ===")
            print(f"玩家位置: ({info['player_pos'][0]:.2f}, {info['player_pos'][1]:.2f})")
            print(f"最近子弹距离: {info['min_bullet_dist']:.1f}")
            print(f"子弹数量: {info['bullet_count']}")
            print(f"当前分数: {info['score']}")
            print(f"当前奖励: {reward:.2f}")
            if hasattr(self, 'last_reward_breakdown'):
                print(f"奖励分解: {self.last_reward_breakdown}")
            print("=" * 40)

        # 每200步保存一次调试图像
        if self.current_step % 200 == 0:
            try:
                # 直接使用当前目录，因为我们已经在project_player目录中
                debug_dir = "debug"
                if not os.path.exists(debug_dir):
                    os.makedirs(debug_dir)
                    print(f"创建调试目录: {debug_dir}")

                debug_path = os.path.join(debug_dir, f"debug_step_{self.current_step}.png")
                self.render_debug(debug_path)
                print(f"✅ 已保存调试图像: {debug_path}")
            except Exception as e:
                print(f"❌ 保存调试图像失败: {e}")
        
        return state, reward, done, info
    
    def _calculate_enhanced_reward(self, info, action):
        """改进的奖励函数 - 更平衡和有效的奖励机制"""
        reward = 0
        reward_breakdown = {}  # 用于调试的奖励分解

        # 1. 分数奖励（主要正奖励来源）- 增强分数信号
        score_diff = info['score'] - self.last_score
        score_reward = 0
        if score_diff > 0:
            score_reward = score_diff * 0.01  # 提高10倍，使分数信号更强
            reward += score_reward
        reward_breakdown['score'] = score_reward

        # 2. 基础生存奖励（鼓励存活）
        survival_reward = 0.5  # 提高生存奖励
        reward += survival_reward
        reward_breakdown['survival'] = survival_reward

        # 3. 改进的危险距离奖励/惩罚系统
        bullet_dist = info['min_bullet_dist']
        danger_reward = 0

        if bullet_dist < 80:  # 扩大检测范围
            # 使用连续函数而非阶梯函数，提供更平滑的梯度
            if bullet_dist < 10:  # 极度危险
                danger_reward = -50  # 强烈惩罚
            elif bullet_dist < 20:  # 高度危险
                danger_reward = -20
            elif bullet_dist < 30:  # 中度危险
                danger_reward = -5
            elif bullet_dist < 50:  # 轻度危险
                danger_reward = -1
            else:  # 安全距离 (50-80)
                danger_reward = 1  # 给予小奖励鼓励保持安全距离

            reward += danger_reward
        else:
            # 距离很远时给予安全奖励
            danger_reward = 2
            reward += danger_reward

        reward_breakdown['danger'] = danger_reward

        # 4. 简化位置奖励（减少过度约束）
        px, py = info['player_pos']
        position_reward = 0
        if py > 0.8:  # 在最底部20%区域
            position_reward = 0.5  # 适度奖励
        elif py < 0.3:  # 在上部30%区域
            position_reward = -2  # 轻微惩罚，不要过重
        reward += position_reward
        reward_breakdown['position'] = position_reward

        # 5. 躲避奖励（成功远离危险的奖励）
        dodge_reward = 0
        if hasattr(self, 'prev_min_bullet_dist') and self.prev_min_bullet_dist < 50:
            if bullet_dist > self.prev_min_bullet_dist + 5:  # 成功拉开距离
                dodge_reward = 5  # 增强躲避奖励
                reward += dodge_reward
        reward_breakdown['dodge'] = dodge_reward

        # 6. 新增：动作一致性奖励（减少无意义的抖动）
        consistency_reward = 0
        if hasattr(self, 'prev_action'):
            if action == self.prev_action and action != 0:  # 连续执行相同的移动动作
                consistency_reward = 0.1
                reward += consistency_reward
        reward_breakdown['consistency'] = consistency_reward

        # 存储当前动作用于下次比较
        self.prev_action = action

        # 存储前一帧的子弹距离
        self.prev_min_bullet_dist = bullet_dist

        # 存储奖励分解用于调试
        if hasattr(self, 'debug_rewards') and self.debug_rewards:
            self.last_reward_breakdown = reward_breakdown

        return reward
    
    def _get_enhanced_state(self):
        """获取增强的状态表示"""
        info = self.cap.capture_frame_with_detection()
        return self._get_enhanced_state_from_info(info)
    
    def _get_enhanced_state_from_info(self, info):
        """从信息构建增强状态"""
        # 1. 图像帧堆栈
        frames = np.stack(self.frames, axis=0)
        
        # 2. 危险度图
        # 获取原始弹幕图像来创建危险图
        full_img = self.cap.capture_full_window()
        gameplay_img = full_img[
            self.cap.gameplay_y : self.cap.gameplay_y + self.cap.gameplay_height,
            self.cap.gameplay_x : self.cap.gameplay_x + self.cap.gameplay_width
        ]
        # 为子弹检测创建player_bbox
        px, py = info['raw_player_pos']
        bbox_size = 30
        player_bbox = (px - bbox_size//2, py - bbox_size//2, bbox_size, bbox_size)

        bullets = self.cap.detect_bullets(gameplay_img, player_bbox)
        danger_map = self.cap.create_danger_map(gameplay_img, info['raw_player_pos'], bullets)
        danger_map = np.expand_dims(danger_map, axis=0)
        
        # 3. 额外特征
        features = np.array([
            info['player_pos'][0],  # 自机x坐标（归一化）
            info['player_pos'][1],  # 自机y坐标（归一化）
            info['min_bullet_dist'] / 100.0, # 最近子弹距离（像素距离/100进行缩放）
            min(info['bullet_count'] / 100.0, 1.0)  # 子弹数量（归一化）
        ], dtype=np.float32)
        
        return {
            'frames': frames,
            'danger_map': danger_map,
            'features': features
        }
    
    def _get_state(self):
        """获取基础状态（兼容原始接口）"""
        return np.stack(self.frames, axis=0)
    
    def render_debug(self, save_path=None):
        """渲染调试视图（增强版）"""
        info = self.cap.capture_frame_with_detection()

        # 获取原始图像
        full_img = self.cap.capture_full_window()
        gameplay_img = full_img[
            self.cap.gameplay_y : self.cap.gameplay_y + self.cap.gameplay_height,
            self.cap.gameplay_x : self.cap.gameplay_x + self.cap.gameplay_width
        ].copy()

        # 转换为BGR用于OpenCV绘制
        debug_img = cv2.cvtColor(gameplay_img, cv2.COLOR_BGRA2BGR)

        # 绘制自机位置（绿色圆圈和十字）
        px, py = info['raw_player_pos']
        cv2.circle(debug_img, (int(px), int(py)), 15, (0, 255, 0), 2)  # 外圈
        cv2.circle(debug_img, (int(px), int(py)), 5, (0, 255, 0), 2)   # 内圈
        cv2.circle(debug_img, (int(px), int(py)), 2, (0, 255, 0), -1)  # 中心点

        # 绘制十字线
        cv2.line(debug_img, (int(px-10), int(py)), (int(px+10), int(py)), (0, 255, 0), 1)
        cv2.line(debug_img, (int(px), int(py-10)), (int(px), int(py+10)), (0, 255, 0), 1)

        # 为子弹检测创建player_bbox
        px, py = info['raw_player_pos']
        bbox_size = 30
        player_bbox = (px - bbox_size//2, py - bbox_size//2, bbox_size, bbox_size)

        # 绘制子弹位置（不同颜色表示不同距离）
        bullets = self.cap.detect_bullets(gameplay_img, player_bbox)
        min_dist = float('inf')
        closest_bullet = None

        for bx, by in bullets:
            bullet_pos = (int(bx), int(by))
            # 计算距离
            dist = np.sqrt((bx - px)**2 + (by - py)**2)

            # 根据距离选择颜色
            if dist < 30:  # 危险距离 - 红色
                color = (0, 0, 255)
                radius = 4
            elif dist < 60:  # 警戒距离 - 橙色
                color = (0, 165, 255)
                radius = 3
            else:  # 安全距离 - 蓝色
                color = (255, 0, 0)
                radius = 2

            cv2.circle(debug_img, bullet_pos, radius, color, -1)

            # 记录最近的子弹
            if dist < min_dist:
                min_dist = dist
                closest_bullet = bullet_pos

        # 绘制到最近子弹的连线
        if closest_bullet:
            cv2.line(debug_img, (int(px), int(py)), closest_bullet, (255, 255, 0), 1)
            # 在连线中点显示距离
            mid_x = (int(px) + closest_bullet[0]) // 2
            mid_y = (int(py) + closest_bullet[1]) // 2
            cv2.putText(debug_img, f"{min_dist:.1f}", (mid_x, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # 绘制安全区域指示（下方30%区域）
        safe_zone_y = int(gameplay_img.shape[0] * 0.7)
        cv2.line(debug_img, (0, safe_zone_y), (gameplay_img.shape[1], safe_zone_y), (0, 255, 255), 1)
        cv2.putText(debug_img, "Safe Zone", (10, safe_zone_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # 添加详细文本信息
        info_text = [
            f"Step: {self.current_step}",
            f"Score: {info['score']}",
            f"Lives: {info['lives']}",
            f"Min Bullet Dist: {info['min_bullet_dist']:.1f}px",
            f"Bullet Count: {info['bullet_count']}",
            f"Player Pos: ({int(px)}, {int(py)})",
            f"Norm Pos: ({info['player_pos'][0]:.2f}, {info['player_pos'][1]:.2f})",
            f"In Safe Zone: {'Yes' if info['player_pos'][1] > 0.7 else 'No'}"
        ]

        # 创建半透明背景
        overlay = debug_img.copy()
        cv2.rectangle(overlay, (5, 5), (300, 25 * len(info_text) + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, debug_img, 0.3, 0, debug_img)

        y_offset = 25
        for text in info_text:
            cv2.putText(debug_img, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25

        if save_path:
            cv2.imwrite(save_path, debug_img)
        else:
            cv2.imshow("Debug View", debug_img)
            cv2.waitKey(1)

        return debug_img
    
    def close(self):
        """关闭环境"""
        # 关闭截屏
        if self.cap:
            self.cap.close()
        # 释放键盘（不需要释放Z键，因为我们使用快速按键模式）
        for action_keys in self.action_map.values():
            for k in action_keys:
                keyboard.release(k)
        print("环境已关闭。")