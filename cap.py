# cap.py
import numpy as np
import cv2
import mss
from touhou_rl import OptimizedGameCapture
import keyboard
from typing import Dict, Tuple
from PIL import Image

class ImprovedGameCapture:
    """增强版游戏捕获，使用组合而非继承"""
    def __init__(self, window_name="搶曽峠杺嫿丂乣 the Embodiment of Scarlet Devil"):
        # 组合：包含一个OptimizedGameCapture实例
        self.base_capture = OptimizedGameCapture(window_name)
        
        # 暴露必要的属性
        self.window_name = self.base_capture.window_name
        self.gameplay_width = self.base_capture.gameplay_width
        self.gameplay_height = self.base_capture.gameplay_height
        self.gameplay_x = self.base_capture.gameplay_x
        self.gameplay_y = self.base_capture.gameplay_y
        self.status_x = self.base_capture.status_x
        self.status_y = self.base_capture.status_y
        self.status_width = self.base_capture.status_width
        self.status_height = self.base_capture.status_height

        # 初始化历史位置用于稳定性检查
        self.last_player_pos = None
        
    def set_window_by_name(self):
        """代理到基础捕获类"""
        self.base_capture.set_window_by_name()
        
    def capture_full_window(self):
        """代理到基础捕获类"""
        return self.base_capture.capture_full_window()
        
    def capture_frame(self):
        """代理到基础捕获类"""
        return self.base_capture.capture_frame()
        
    def extract_status_info(self, status_img):
        """代理到基础捕获类"""
        return self.base_capture.extract_status_info(status_img)
    def close(self):
        """代理到基础捕获类"""
        return self.base_capture.close()
    def find_hitbox_center(self, img, player_bbox):
        """在自机区域内查找白色判定点"""
        x, y, w, h = player_bbox

        # 判定点通常在自机的中心区域
        center_x = x + w // 2
        center_y = y + h // 2

        # 定义中心搜索区域（比自机小很多）
        search_radius = min(w, h) // 4  # 搜索半径为自机大小的1/4
        x1 = max(0, center_x - search_radius)
        y1 = max(0, center_y - search_radius)
        x2 = min(img.shape[1], center_x + search_radius)
        y2 = min(img.shape[0], center_y + search_radius)

        # 提取中心区域
        roi = img[y1:y2, x1:x2]

        # 转换为灰度图
        if len(roi.shape) == 3:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = roi

        # 查找白色区域（判定点通常是纯白色）
        _, white_mask = cv2.threshold(gray_roi, 250, 255, cv2.THRESH_BINARY)

        # 查找轮廓
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 查找最接近ROI中心的小白色圆形区域
        best_center = None
        min_distance = float('inf')
        roi_center_x = roi.shape[1] // 2
        roi_center_y = roi.shape[0] // 2

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 判定点通常很小，面积在5-50之间
            if 5 < area < 50:
                # 检查圆形度
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.8:  # 要求更高的圆形度
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])

                            # 计算到ROI中心的距离
                            distance = np.sqrt((cx - roi_center_x)**2 + (cy - roi_center_y)**2)

                            # 判定点应该非常接近中心
                            if distance < search_radius // 2:
                                if distance < min_distance:
                                    min_distance = distance
                                    best_center = (cx + x1, cy + y1)  # 转换回原图坐标

        # 如果没找到合适的白色圆点，使用自机中心
        if best_center is None:
            best_center = (center_x, center_y)
        else:
            # 验证找到的点是否真的在自机边界内
            if not (x <= best_center[0] <= x + w and y <= best_center[1] <= y + h):
                best_center = (center_x, center_y)

        return best_center



    def detect_player_position(self, gameplay_img):
        """检测自机位置（完全照搬位置测试.py的逻辑）"""
        # 转换为BGR格式（如果是BGRA）
        if len(gameplay_img.shape) == 4:  # BGRA
            img = cv2.cvtColor(gameplay_img, cv2.COLOR_BGRA2BGR)
        else:
            img = gameplay_img

        results = []

        # 使用颜色检测（完全照搬位置测试.py的逻辑）

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 红色范围（与位置测试.py完全一致）
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 找到最大的轮廓（与位置测试.py完全一致）
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if 100 < area < 5000:
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = w / h if h > 0 else 0

                if 0.5 < aspect_ratio < 2.0:
                    bbox = (x, y, w, h)
                    # 查找判定点
                    hitbox_center = self.find_hitbox_center(img, bbox)

                    results.append({
                        'method': 'color',
                        'confidence': 0.5,
                        'bbox': bbox,
                        'center': hitbox_center
                    })
                    # print(f"使用颜色检测找到自机: 判定点位置={hitbox_center}")  # 注释掉调试输出

                    # 保存位置用于下次稳定性检查
                    self.last_player_pos = hitbox_center

                    return hitbox_center

        if not results:
            pass  # print("警告：未能检测到自机！")  # 注释掉调试输出

        # 如果都检测不到，返回屏幕中心偏下（默认位置）
        return (gameplay_img.shape[1] // 2, gameplay_img.shape[0] - 50)
    
    def detect_bullets(self, gameplay_img, player_bbox=None):
        """检测子弹位置（完全照搬位置测试.py的代码）"""
        bullets = []

        # 转换为灰度图
        if len(gameplay_img.shape) == 4:  # BGRA
            bgr = cv2.cvtColor(gameplay_img, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        elif len(gameplay_img.shape) == 3:  # BGR
            gray = cv2.cvtColor(gameplay_img, cv2.COLOR_BGR2GRAY)
        else:  # 已经是灰度图
            gray = gameplay_img

        # 如果有自机位置，创建一个掩码来排除自机区域
        mask = np.ones(gray.shape, dtype=np.uint8) * 255
        if player_bbox is not None:
            x, y, w, h = player_bbox
            # 在自机中心区域创建一个小的排除区域
            center_x = x + w // 2
            center_y = y + h // 2
            exclusion_radius = min(w, h) // 3
            cv2.circle(mask, (center_x, center_y), exclusion_radius, 0, -1)

        # 改进的子弹检测：使用多阈值检测
        # 主要检测高亮度子弹
        high_brightness_mask = cv2.inRange(gray, 240, 255)

        # 检测中等亮度子弹（某些子弹可能不是纯白色）
        mid_brightness_mask = cv2.inRange(gray, 200, 239)

        # 对中等亮度掩码进行形态学操作，减少噪声
        kernel = np.ones((2, 2), np.uint8)
        mid_brightness_mask = cv2.morphologyEx(mid_brightness_mask, cv2.MORPH_CLOSE, kernel)

        # 合并两个掩码
        brightness_mask = cv2.bitwise_or(high_brightness_mask, mid_brightness_mask)

        # 应用自机排除掩码
        brightness_mask = cv2.bitwise_and(brightness_mask, mask)

        contours, _ = cv2.findContours(brightness_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 3 < area < 500:
                x, y, w, h = cv2.boundingRect(cnt)

                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # 减小中心正方形区域的大小
                    square_size = min(w, h) // 3  # 改为1/3大小
                    square_size = max(2, min(square_size, 4))  # 限制在2-4像素之间

                    # 提取中心正方形区域
                    half_size = square_size // 2
                    center_x1 = max(0, cx - half_size)
                    center_y1 = max(0, cy - half_size)
                    center_x2 = min(gray.shape[1], cx + half_size)
                    center_y2 = min(gray.shape[0], cy + half_size)

                    center_region = brightness_mask[center_y1:center_y2, center_x1:center_x2]

                    if center_region.size > 0:
                        white_pixel_ratio = np.sum(center_region > 0) / center_region.size

                        if white_pixel_ratio > 0.3:
                            # 避免重复检测
                            is_duplicate = False
                            for existing_bullet in bullets:
                                if abs(existing_bullet[0] - cx) < 5 and abs(existing_bullet[1] - cy) < 5:
                                    is_duplicate = True
                                    break

                            if not is_duplicate:
                                bullets.append((cx, cy))

        return bullets
    
    def capture_frame_with_detection(self):
        """增强版捕获，返回更多信息"""
        full_img = self.capture_full_window()
        
        # 提取弹幕区
        gameplay_img = full_img[
            self.gameplay_y : self.gameplay_y + self.gameplay_height,
            self.gameplay_x : self.gameplay_x + self.gameplay_width
        ]
        
        # 检测自机位置
        player_pos = self.detect_player_position(gameplay_img)

        # 为子弹检测创建player_bbox（从自机位置推算）
        player_bbox = None
        if player_pos:
            # 假设自机大小约为30x30像素
            px, py = player_pos
            bbox_size = 30
            player_bbox = (px - bbox_size//2, py - bbox_size//2, bbox_size, bbox_size)

        # 检测子弹
        bullets = self.detect_bullets(gameplay_img, player_bbox)
        
        # 转灰度并缩放
        gray_frame = cv2.cvtColor(gameplay_img, cv2.COLOR_BGRA2GRAY)
        scaled_frame = cv2.resize(gray_frame, (128, 96), interpolation=cv2.INTER_AREA)
        
        # 获取状态信息
        status_img = full_img[
            self.status_y : self.status_y + self.status_height,
            self.status_x : self.status_x + self.status_width
        ]
        score, lives, bombs = self.extract_status_info(status_img)
        
        # 将位置信息归一化到0-1范围
        player_x_norm = player_pos[0] / self.gameplay_width
        player_y_norm = player_pos[1] / self.gameplay_height
        
        # 计算最近子弹的距离（像素距离，不归一化）
        min_bullet_dist = 1000.0  # 设置一个较大的初始值
        if bullets:
            for bx, by in bullets:
                dist = np.sqrt((bx - player_pos[0])**2 + (by - player_pos[1])**2)
                min_bullet_dist = min(min_bullet_dist, dist)
        
        return {
            'frame': scaled_frame,
            'player_pos': (player_x_norm, player_y_norm),
            'min_bullet_dist': min_bullet_dist,
            'score': score,
            'lives': lives,
            'bombs': bombs,
            'raw_player_pos': player_pos,  # 原始坐标用于调试
            'bullet_count': len(bullets)
        }
    
    def create_danger_map(self, gameplay_img, player_pos, bullets):
        """创建危险度热力图"""
        danger_map = np.zeros((96, 128), dtype=np.float32)
        
        # 缩放因子
        scale_x = 128 / self.gameplay_width
        scale_y = 96 / self.gameplay_height
        
        # 自机位置
        px = int(player_pos[0] * scale_x)
        py = int(player_pos[1] * scale_y)
        
        # 为每个子弹添加危险度
        for bx, by in bullets:
            bx_scaled = int(bx * scale_x)
            by_scaled = int(by * scale_y)
            
            # 创建高斯分布的危险度
            for y in range(max(0, by_scaled-10), min(96, by_scaled+10)):
                for x in range(max(0, bx_scaled-10), min(128, bx_scaled+10)):
                    dist = np.sqrt((x - bx_scaled)**2 + (y - by_scaled)**2)
                    danger = np.exp(-dist**2 / 20)  # 高斯衰减
                    danger_map[y, x] = max(danger_map[y, x], danger)
        
        return danger_map