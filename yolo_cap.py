# yolo_cap.py
"""
YOLO集成游戏捕获类 - 替换颜色检测系统
基于cap.py，使用YOLO模型进行自机和子弹检测
"""

import numpy as np
import cv2
import mss
from touhou_rl import OptimizedGameCapture
import keyboard
from typing import Dict, Tuple, List, Optional
from PIL import Image
import torch
import time
import os
from pathlib import Path
import pytesseract  # 添加OCR支持

# 尝试导入YOLO，如果失败则提供降级方案
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ 警告: ultralytics未安装，YOLO功能不可用")
    YOLO_AVAILABLE = False


class YOLOGameCapture:
    """YOLO增强版游戏捕获类，集成YOLO目标检测"""

    def __init__(self,
                 window_name="搶曽峠杺嫿丂乣 the Embodiment of Scarlet Devil",
                 model_path="runs/detect/touhou_feature_enhanced/weights/best.pt",  # 使用训练好的YOLO模型
                 fallback_to_color=True,
                 conf_threshold=0.25,
                 iou_threshold=0.7):
        """
        初始化YOLO游戏捕获

        Args:
            window_name: 游戏窗口名称
            model_path: YOLO模型路径
            fallback_to_color: 是否在YOLO失败时回退到颜色检测
            conf_threshold: YOLO置信度阈值
            iou_threshold: YOLO IoU阈值
        """
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

        # YOLO相关配置
        self.model_path = model_path
        self.fallback_to_color = fallback_to_color
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # YOLO模型
        self.yolo_model = None
        self.yolo_available = False

        # 性能统计
        self.yolo_detection_times = []
        self.color_detection_times = []
        self.detection_method_stats = {'yolo': 0, 'color': 0, 'failed': 0}

        # 初始化历史位置用于稳定性检查
        self.last_player_pos = None

        # 设置Tesseract OCR路径
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        # 初始化YOLO模型
        self._initialize_yolo()

    def _initialize_yolo(self):
        """初始化YOLO模型"""
        if not YOLO_AVAILABLE:
            print("⚠️ YOLO不可用，将使用颜色检测")
            return

        try:
            if os.path.exists(self.model_path):
                print(f"🔄 加载YOLO模型: {self.model_path}")
                self.yolo_model = YOLO(self.model_path)
                self.yolo_available = True
                print("✅ YOLO模型加载成功")

                # 预热模型
                dummy_img = np.zeros((384, 384, 3), dtype=np.uint8)
                _ = self.yolo_model(dummy_img, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
                print("🔥 YOLO模型预热完成")
            else:
                print(f"❌ YOLO模型文件不存在: {self.model_path}")
                if not self.fallback_to_color:
                    raise FileNotFoundError(f"YOLO模型文件不存在: {self.model_path}")
        except Exception as e:
            print(f"❌ YOLO模型加载失败: {e}")
            if not self.fallback_to_color:
                raise e

    def _yolo_detect_objects(self, img) -> Tuple[Optional[Tuple[int, int]], List[Tuple[int, int]]]:
        """使用YOLO检测游戏对象"""
        if not self.yolo_available or self.yolo_model is None:
            return None, []

        start_time = time.time()

        try:
            # 确保输入图像是3通道BGR格式
            if len(img.shape) == 3 and img.shape[2] == 4:  # BGRA格式
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 3:  # 已经是BGR格式
                pass
            else:
                print(f"⚠️ 不支持的图像格式: {img.shape}")
                return None, []

            # YOLO推理
            results = self.yolo_model(img, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)

            player_pos = None
            bullets = []

            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    classes = result.boxes.cls.cpu().numpy()  # 类别ID
                    confidences = result.boxes.conf.cpu().numpy()  # 置信度

                    for box, cls, conf in zip(boxes, classes, confidences):
                        x1, y1, x2, y2 = box
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)

                        if int(cls) == 0:  # player_hitbox
                            if player_pos is None or conf > 0.5:  # 选择置信度最高的
                                player_pos = (center_x, center_y)
                        elif int(cls) == 1:  # bullet_center
                            bullets.append((center_x, center_y))

            # 记录性能
            detection_time = time.time() - start_time
            self.yolo_detection_times.append(detection_time)
            self.detection_method_stats['yolo'] += 1

            return player_pos, bullets

        except Exception as e:
            print(f"⚠️ YOLO检测出错: {e}")
            return None, []

    def _color_detect_player(self, img) -> Optional[Tuple[int, int]]:
        """颜色检测自机位置（备用方案）"""
        start_time = time.time()

        try:
            # 转换为BGR格式（如果是BGRA）
            if len(img.shape) == 4:  # BGRA
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # 红色范围
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
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)

                if 100 < area < 5000:
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    aspect_ratio = w / h if h > 0 else 0

                    if 0.5 < aspect_ratio < 2.0:
                        bbox = (x, y, w, h)
                        hitbox_center = self._find_hitbox_center(img, bbox)

                        # 记录性能
                        detection_time = time.time() - start_time
                        self.color_detection_times.append(detection_time)
                        self.detection_method_stats['color'] += 1

                        return hitbox_center

            return None

        except Exception as e:
            print(f"⚠️ 颜色检测出错: {e}")
            return None

    def _color_detect_bullets(self, img, player_bbox=None) -> List[Tuple[int, int]]:
        """颜色检测子弹位置（备用方案）"""
        bullets = []

        try:
            # 转换为灰度图
            if len(img.shape) == 4:  # BGRA
                bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            elif len(img.shape) == 3:  # BGR
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:  # 已经是灰度图
                gray = img

            # 如果有自机位置，创建一个掩码来排除自机区域
            mask = np.ones(gray.shape, dtype=np.uint8) * 255
            if player_bbox is not None:
                x, y, w, h = player_bbox
                center_x = x + w // 2
                center_y = y + h // 2
                exclusion_radius = min(w, h) // 3
                cv2.circle(mask, (center_x, center_y), exclusion_radius, 0, -1)

            # 多阈值检测
            high_brightness_mask = cv2.inRange(gray, 240, 255)
            mid_brightness_mask = cv2.inRange(gray, 200, 239)

            kernel = np.ones((2, 2), np.uint8)
            mid_brightness_mask = cv2.morphologyEx(mid_brightness_mask, cv2.MORPH_CLOSE, kernel)

            brightness_mask = cv2.bitwise_or(high_brightness_mask, mid_brightness_mask)
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

                        square_size = min(w, h) // 3
                        square_size = max(2, min(square_size, 4))

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

        except Exception as e:
            print(f"⚠️ 子弹颜色检测出错: {e}")
            return []

    def _find_hitbox_center(self, img, player_bbox):
        """在自机区域内查找白色判定点"""
        x, y, w, h = player_bbox

        center_x = x + w // 2
        center_y = y + h // 2

        search_radius = min(w, h) // 4
        x1 = max(0, center_x - search_radius)
        y1 = max(0, center_y - search_radius)
        x2 = min(img.shape[1], center_x + search_radius)
        y2 = min(img.shape[0], center_y + search_radius)

        roi = img[y1:y2, x1:x2]

        if len(roi.shape) == 3:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = roi

        _, white_mask = cv2.threshold(gray_roi, 250, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_center = None
        min_distance = float('inf')
        roi_center_x = roi.shape[1] // 2
        roi_center_y = roi.shape[0] // 2

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5 < area < 50:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.8:
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])

                            distance = np.sqrt((cx - roi_center_x)**2 + (cy - roi_center_y)**2)

                            if distance < search_radius // 2:
                                if distance < min_distance:
                                    min_distance = distance
                                    best_center = (cx + x1, cy + y1)

        if best_center is None:
            best_center = (center_x, center_y)
        else:
            if not (x <= best_center[0] <= x + w and y <= best_center[1] <= y + h):
                best_center = (center_x, center_y)

        return best_center

    def detect_player_position(self, gameplay_img):
        """检测自机位置 - 优先使用YOLO，失败时回退到颜色检测"""
        # 首先尝试YOLO检测
        if self.yolo_available:
            player_pos, _ = self._yolo_detect_objects(gameplay_img)
            if player_pos is not None:
                self.last_player_pos = player_pos
                return player_pos

        # YOLO失败或不可用时，使用颜色检测
        if self.fallback_to_color:
            player_pos = self._color_detect_player(gameplay_img)
            if player_pos is not None:
                self.last_player_pos = player_pos
                return player_pos

        # 都失败时，记录失败并返回默认位置
        self.detection_method_stats['failed'] += 1

        # 如果有历史位置，使用历史位置
        if self.last_player_pos is not None:
            return self.last_player_pos

        # 否则返回屏幕中心偏下（默认位置）
        return (gameplay_img.shape[1] // 2, gameplay_img.shape[0] - 50)

    def detect_bullets(self, gameplay_img, player_bbox=None):
        """检测子弹位置 - 优先使用YOLO，失败时回退到颜色检测"""
        # 首先尝试YOLO检测
        if self.yolo_available:
            _, bullets = self._yolo_detect_objects(gameplay_img)
            if bullets:  # 如果检测到子弹
                return bullets

        # YOLO失败或不可用时，使用颜色检测
        if self.fallback_to_color:
            return self._color_detect_bullets(gameplay_img, player_bbox)

        return []

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

    # 代理方法 - 保持与cap.py的接口兼容性
    def set_window_by_name(self):
        """代理到基础捕获类"""
        return self.base_capture.set_window_by_name()

    def capture_full_window(self):
        """代理到基础捕获类"""
        return self.base_capture.capture_full_window()

    def capture_frame(self):
        """代理到基础捕获类"""
        return self.base_capture.capture_frame()

    def extract_status_info(self, status_img):
        """
        给定状态栏图(高448,宽192)，在里面找 (score, lives)。
        注意：不再检测炸弹数，炸弹数由内部计数管理。
        """
        try:
            # ----------------
            # 在状态栏图上精确定位你的 score 区域:
            # 例如这里示例: score 大约在 (y1=30, y2=60, x1=20, x2=150)
            score_box  = (60, 90,  10, 200)

            # 生命星星(红色)
            # 例如示例: (y1=100, y2=120, x1=50, x2=180)
            player_box = (110, 130, 50, 180)
            # ----------------

            # 裁剪(注意status_img的 shape ~ (448,192,4))
            def crop(box):
                y1, y2, x1, x2 = box
                return status_img[y1:y2, x1:x2]

            score_roi  = crop(score_box)
            player_roi = crop(player_box)

            # Debug: 在一张拷贝图中画出矩形看看对不对
            # debug_show = status_img.copy()
            # cv2.rectangle(debug_show, (score_box[2],  score_box[0]),  (score_box[3],  score_box[1]),  (0,0,255),   2)
            # cv2.rectangle(debug_show, (player_box[2], player_box[0]), (player_box[3], player_box[1]), (0,255,0),  2)
            # # 你可在调试时打开:
            # cv2.imshow("debug_status", debug_show)
            # cv2.waitKey(1)

            # OCR 解析分数
            score_val = self._extract_score(score_roi)

            # HSV 解析红星=生命
            lives_val = self._count_stars(player_roi, color="red")

            # 炸弹数不再通过OCR检测，直接返回0
            bombs_val = 0

            return score_val, lives_val, bombs_val

        except Exception as e:
            print(f"状态信息提取失败: {e}")
            return 0, 0, 0

    def _extract_score(self, roi_bgra):
        """
        OCR数字。先转灰度并二值化，然后只允许数字。
        """
        gray = cv2.cvtColor(roi_bgra, cv2.COLOR_BGRA2GRAY)
        _, bin_ = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        txt = pytesseract.image_to_string(
            bin_, config="--psm 7 -c tessedit_char_whitelist=0123456789"
        )
        try:
            return int(txt.strip())
        except ValueError:
            return 0

    def _count_stars(self, roi_bgra, color="red"):
        """
        基于HSV统计轮廓面积>20的块数，认为是星星数量。
        red: (0~10) or (160~180)
        green: 多种绿色范围
        """
        bgr = cv2.cvtColor(roi_bgra, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        if color == "red":
            m1 = cv2.inRange(hsv, (0,   100, 100), (10,  255, 255))
            m2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
            mask = cv2.bitwise_or(m1, m2)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = sum(1 for c in cnts if cv2.contourArea(c) > 20)
        return min(count, 5)  # 最多5颗星

    def close(self):
        """代理到基础捕获类"""
        return self.base_capture.close()

    def capture_frame_with_detection(self):
        """增强版捕获，返回更多信息 - 使用YOLO检测"""
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

        # 确定使用的检测方法
        detection_method = 'yolo' if self.yolo_available and self.detection_method_stats['yolo'] > 0 else 'color'

        return {
            'frame': scaled_frame,
            'player_pos': (player_x_norm, player_y_norm),
            'min_bullet_dist': min_bullet_dist,
            'score': score,
            'lives': lives,
            'bombs': bombs,
            'raw_player_pos': player_pos,  # 原始坐标用于调试
            'bullet_count': len(bullets),
            'detection_method': detection_method  # 新增：检测方法标识
        }

    def print_performance_stats(self):
        """打印性能统计信息"""
        print("\n📊 YOLO游戏捕获性能统计:")
        print(f"   检测方法使用次数: {self.detection_method_stats}")

        if self.yolo_detection_times:
            avg_yolo_time = np.mean(self.yolo_detection_times) * 1000
            print(f"   YOLO平均检测时间: {avg_yolo_time:.2f}ms")

        if self.color_detection_times:
            avg_color_time = np.mean(self.color_detection_times) * 1000
            print(f"   颜色检测平均时间: {avg_color_time:.2f}ms")

        total_detections = sum(self.detection_method_stats.values())
        if total_detections > 0:
            yolo_ratio = self.detection_method_stats['yolo'] / total_detections * 100
            color_ratio = self.detection_method_stats['color'] / total_detections * 100
            failed_ratio = self.detection_method_stats['failed'] / total_detections * 100
            print(f"   YOLO使用率: {yolo_ratio:.1f}%")
            print(f"   颜色检测使用率: {color_ratio:.1f}%")
            print(f"   检测失败率: {failed_ratio:.1f}%")


# 兼容性别名，便于替换现有代码
ImprovedGameCapture = YOLOGameCapture


def test_yolo_capture():
    """测试YOLO捕获功能"""
    print("🧪 开始测试YOLO游戏捕获...")

    # 创建捕获实例
    cap = YOLOGameCapture()

    try:
        # 设置窗口
        if not cap.set_window_by_name():
            print("❌ 未找到游戏窗口")
            return

        print("✅ 游戏窗口已找到，开始测试...")

        # 测试几帧
        for i in range(5):
            print(f"\n🔍 测试第 {i+1} 帧:")

            # 捕获并检测
            result = cap.capture_frame_with_detection()

            print(f"   检测方法: {result['detection_method']}")
            print(f"   自机位置: {result['raw_player_pos']}")
            print(f"   子弹数量: {result['bullet_count']}")
            print(f"   最近子弹距离: {result['min_bullet_dist']:.1f}")

            time.sleep(0.1)  # 短暂延迟

        # 打印性能统计
        cap.print_performance_stats()

    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
    except Exception as e:
        print(f"❌ 测试出错: {e}")
    finally:
        cap.close()
        print("🔚 测试结束")


if __name__ == "__main__":
    test_yolo_capture()
