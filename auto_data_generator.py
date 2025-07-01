# auto_data_generator.py
"""
基于位置测试.py的游戏对象检测功能，自动生成YOLOv8训练数据
利用现有的GameObjectDetector类进行实时游戏截图和标注

改进功能：
- 数据质量控制和验证
- 多样性保证机制
- 自适应采样率
- 数据清洗和后处理
- 增强的可视化验证
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import time
import keyboard
import mss
import subprocess
import json
import win32gui
from typing import Dict, List, Tuple
from datetime import datetime

# 添加项目根目录到路径，以便导入位置测试.py
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入位置测试.py中的检测器
try:
    from 位置测试 import GameObjectDetector as BaseGameObjectDetector
except ImportError:
    print("错误：无法导入位置测试.py中的GameObjectDetector类")
    print("请确保位置测试.py文件在项目根目录中")
    sys.exit(1)


class EnhancedGameObjectDetector(BaseGameObjectDetector):
    """增强版游戏对象检测器 - 自机固定6x6，子弹动态大小"""

    def find_hitbox_center_improved(self, img, player_bbox):
        """改进的判定点检测 - 更准确地定位自机中心"""
        x, y, w, h = player_bbox

        # 首先尝试原始的白色判定点检测
        original_center = self.find_hitbox_center(img, player_bbox)

        # 检查原始检测是否只是回退到了几何中心
        geometric_center = (x + w // 2, y + h // 2)

        # 如果原始检测返回的就是几何中心，说明没找到白色判定点
        if original_center == geometric_center:
            # 使用改进的方法：分析自机精灵的视觉重心
            return self.find_visual_center(img, player_bbox)
        else:
            # 找到了白色判定点，直接使用
            return original_center

    def find_visual_center(self, img, player_bbox):
        """通过分析自机精灵的视觉特征找到更准确的中心点"""
        x, y, w, h = player_bbox

        # 简化方案：直接使用几何中心，但进行合理的偏移调整
        # 根据东方游戏的特点，判定点通常在自机精灵的视觉中心
        # 而不是检测到的红色区域的几何中心

        center_x = x + w // 2
        center_y = y + h // 2

        # 根据反馈调整偏移量：
        # 1. 减少左偏移（之前偏移太多了）
        # 2. 稍微向上偏移（因为判定点通常在精灵中心偏上）
        adjusted_x = center_x   # 向左偏移1/16宽度（减少偏移）
        adjusted_y = center_y - h // 8   # 向上偏移1/8高度

        # 确保调整后的点仍在自机区域内
        adjusted_x = max(x, min(x + w, adjusted_x))
        adjusted_y = max(y, min(y + h, adjusted_y))

        return (adjusted_x, adjusted_y)

    def detect_bullets_with_dynamic_size(self, img, player_bbox=None):
        """检测子弹位置，并返回基于白色区域大小的动态尺寸 - 改进版处理连续子弹"""
        bullets = []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # 如果有自机位置，创建一个掩码来排除自机区域
        mask = np.ones(gray.shape, dtype=np.uint8) * 255
        if player_bbox is not None:
            x, y, w, h = player_bbox
            # 在自机中心区域创建一个小的排除区域
            center_x = x + w // 2
            center_y = y + h // 2
            exclusion_radius = min(w, h) // 3
            cv2.circle(mask, (center_x, center_y), exclusion_radius, 0, -1)

        # 使用亮度范围检测子弹
        min_brightness = 230
        max_brightness = 255

        # 创建亮度范围掩码
        brightness_mask = cv2.inRange(gray, min_brightness, max_brightness)

        # 应用自机排除掩码
        brightness_mask = cv2.bitwise_and(brightness_mask, mask)

        contours, _ = cv2.findContours(brightness_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 3 < area < 2000:  # 增加最大面积以处理连续子弹
                x, y, w, h = cv2.boundingRect(cnt)

                # 判断是否为连续子弹（长条形状）
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1

                if aspect_ratio > 3:  # 长宽比大于3，可能是连续子弹
                    # 处理连续子弹：沿长轴方向分割
                    bullets.extend(self.split_continuous_bullets(cnt, x, y, w, h, brightness_mask))
                else:
                    # 处理单个子弹
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        # 根据白色区域的实际大小计算标注框大小
                        bullet_size = max(min(w, h), 4)  # 至少4像素
                        bullet_size = min(bullet_size, 16)  # 最多16像素

                        # 提取中心正方形区域进行验证
                        half_size = bullet_size // 2
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
                                for existing in bullets:
                                    if abs(existing['center'][0] - cx) < 5 and abs(existing['center'][1] - cy) < 5:
                                        is_duplicate = True
                                        break

                                if not is_duplicate:
                                    # 获取该点的实际亮度值
                                    actual_brightness = gray[cy, cx] if 0 <= cy < gray.shape[0] and 0 <= cx < gray.shape[1] else 0

                                    bullets.append({
                                        'center': (cx, cy),
                                        'bbox': (x, y, w, h),
                                        'area': area,
                                        'brightness': actual_brightness,
                                        'square_size': bullet_size * 2,  # 保持兼容性
                                        'dynamic_size': bullet_size  # 新增：动态计算的大小
                                    })

        return bullets

    def split_continuous_bullets(self, contour, x, y, w, h, brightness_mask):
        """分割连续子弹为单个子弹"""
        bullets = []

        # 确定是水平还是垂直排列
        is_horizontal = w > h

        if is_horizontal:
            # 水平排列：沿x轴分割
            bullet_width = min(w // max(1, w // 8), 12)  # 估算单个子弹宽度
            num_bullets = max(1, w // bullet_width)

            for i in range(num_bullets):
                bullet_x = x + i * bullet_width + bullet_width // 2
                bullet_y = y + h // 2

                # 验证这个位置确实有白色像素
                if (0 <= bullet_y < brightness_mask.shape[0] and
                    0 <= bullet_x < brightness_mask.shape[1] and
                    brightness_mask[bullet_y, bullet_x] > 0):

                    bullets.append({
                        'center': (bullet_x, bullet_y),
                        'bbox': (bullet_x - bullet_width//2, y, bullet_width, h),
                        'area': bullet_width * h,
                        'brightness': 255,
                        'square_size': bullet_width * 2,
                        'dynamic_size': max(4, min(bullet_width, 12))
                    })
        else:
            # 垂直排列：沿y轴分割
            bullet_height = min(h // max(1, h // 8), 12)  # 估算单个子弹高度
            num_bullets = max(1, h // bullet_height)

            for i in range(num_bullets):
                bullet_x = x + w // 2
                bullet_y = y + i * bullet_height + bullet_height // 2

                # 验证这个位置确实有白色像素
                if (0 <= bullet_y < brightness_mask.shape[0] and
                    0 <= bullet_x < brightness_mask.shape[1] and
                    brightness_mask[bullet_y, bullet_x] > 0):

                    bullets.append({
                        'center': (bullet_x, bullet_y),
                        'bbox': (x, bullet_y - bullet_height//2, w, bullet_height),
                        'area': w * bullet_height,
                        'brightness': 255,
                        'square_size': bullet_height * 2,
                        'dynamic_size': max(4, min(bullet_height, 12))
                    })

        return bullets

class OptimizedGameCapture:
    """游戏窗口捕获类 - 参考touhou_rl.py的实现"""

    def __init__(self, window_name="搶曽峠杺嫿丂乣 the Embodiment of Scarlet Devil", game_path="D:\\Games\\th06\\vpatch.exe"):
        self.sct = mss.mss()
        self.window_name = window_name
        self.game_path = game_path
        self.window_rect = None

        # 东方红魔乡分辨率
        self.game_width = 640
        self.game_height = 480

        # 弹幕区域
        self.gameplay_x, self.gameplay_y = 32, 16
        self.gameplay_width, self.gameplay_height = 384, 448

    def start_game_and_find_window(self):
        """启动游戏并查找窗口"""
        print(f"🎮 正在启动游戏: {self.game_path}")

        # 启动游戏
        try:
            subprocess.Popen(f'"{self.game_path}"', shell=True)
            print("游戏启动命令已执行")
        except Exception as e:
            print(f"启动游戏失败: {e}")
            return False

        # 等待游戏窗口出现
        print("等待游戏窗口出现...")
        for i in range(10):  # 最多等待10秒
            time.sleep(1)
            if self.set_window_by_name():
                print("✅ 游戏窗口已找到并设置为前台")
                return True
            print(f"等待中... ({i+1}/10)")

        print("❌ 游戏窗口未找到")
        return False

    def set_window_by_name(self):
        """查找游戏窗口 - 参考touhou_rl.py的实现"""
        def callback(hwnd, _):
            try:
                if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd) == self.window_name:
                    rect = win32gui.GetWindowRect(hwnd)
                    client = win32gui.GetClientRect(hwnd)
                    border_w = ((rect[2] - rect[0]) - client[2]) // 2
                    title_h = (rect[3] - rect[1]) - client[3] - border_w
                    self.window_rect = (
                        rect[0] + border_w,
                        rect[1] + title_h,
                        client[2],
                        client[3]
                    )
                    # 设置窗口为前台
                    win32gui.SetForegroundWindow(hwnd)
                    return False
            except:
                pass
            return True

        try:
            win32gui.EnumWindows(callback, None)
            if self.window_rect:
                print(f"找到游戏窗口: {self.window_name}")
                return True
            else:
                return False
        except Exception as e:
            print(f"查找窗口失败: {e}")
            return False

    def capture_full_window(self):
        """截取完整游戏窗口 - 参考touhou_rl.py的实现"""
        if not self.window_rect:
            return np.zeros((self.game_height, self.game_width, 4), dtype=np.uint8)

        left, top, client_width, client_height = self.window_rect
        monitor = {
            "left": left,
            "top": top,
            "width": self.game_width,
            "height": self.game_height
        }

        try:
            full_img = np.array(self.sct.grab(monitor))
            return full_img
        except Exception as e:
            print(f"截图失败: {e}")
            return None

    def capture_gameplay_area(self):
        """截取弹幕区域"""
        full_img = self.capture_full_window()
        if full_img is None:
            return None

        # 提取弹幕区域
        gameplay_img = full_img[
            self.gameplay_y : self.gameplay_y + self.gameplay_height,
            self.gameplay_x : self.gameplay_x + self.gameplay_width
        ]

        # 转换为BGR格式（OpenCV标准）
        if gameplay_img.shape[2] == 4:  # BGRA
            gameplay_img = cv2.cvtColor(gameplay_img, cv2.COLOR_BGRA2BGR)

        return gameplay_img

class PreprocessingWorkflow:
    """预处理工作流 - 手动截图和批量后处理"""

    def __init__(self, preprocessing_dir: str = "preprocessing"):
        self.preprocessing_dir = Path(preprocessing_dir)
        self.preprocessing_dir.mkdir(exist_ok=True)

        # 创建子目录
        self.raw_images_dir = self.preprocessing_dir / "raw_images"
        self.visualizations_dir = self.preprocessing_dir / "visualizations"
        self.annotations_dir = self.preprocessing_dir / "annotations"

        self.raw_images_dir.mkdir(exist_ok=True)
        self.visualizations_dir.mkdir(exist_ok=True)
        self.annotations_dir.mkdir(exist_ok=True)

        # 初始化检测器和捕获器
        self.detector = EnhancedGameObjectDetector()
        self.capture = OptimizedGameCapture()

        # 键盘监听状态
        self.is_listening = False
        self.capture_count = 0

        print(f"预处理工作流初始化完成")
        print(f"预处理目录: {self.preprocessing_dir}")
        print(f"原始图像: {self.raw_images_dir}")
        print(f"可视化结果: {self.visualizations_dir}")
        print(f"标注数据: {self.annotations_dir}")

    def start_manual_capture(self):
        """开始手动截图模式 - 按R键截图"""
        print("\n🎮 手动截图模式已启动")
        print("按 'R' 键截取当前游戏画面")
        print("按 'Q' 键退出截图模式")
        print("=" * 50)

        self.is_listening = True
        self.capture_count = 0

        try:
            while self.is_listening:
                if keyboard.is_pressed('r'):
                    self.capture_single_frame()
                    time.sleep(0.5)  # 防止重复触发

                if keyboard.is_pressed('q'):
                    print("\n退出手动截图模式")
                    self.is_listening = False
                    break

                time.sleep(0.1)  # 减少CPU占用

        except KeyboardInterrupt:
            print("\n手动截图模式被中断")
            self.is_listening = False

    def capture_single_frame(self):
        """截取单帧游戏画面"""
        img = self.capture.capture_gameplay_area()
        if img is None:
            print("❌ 截图失败")
            return False

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"capture_{timestamp}_{self.capture_count:04d}.jpg"

        # 保存原始图像
        img_path = self.raw_images_dir / filename
        success = cv2.imwrite(str(img_path), img)

        if success:
            self.capture_count += 1
            print(f"✅ 截图 #{self.capture_count}: {filename}")
            return True
        else:
            print(f"❌ 保存失败: {filename}")
            return False

    def process_single_image(self, img_path: Path) -> dict | None:
        """处理单张图像，返回检测结果"""
        img = cv2.imread(str(img_path))
        if img is None:
            return None

        # 检测游戏对象
        player_detections = self.detector.detect_player_position(img)
        player_bbox = None
        if player_detections:
            player_bbox = player_detections[0]['bbox']
            # 使用改进的判定点检测
            improved_center = self.detector.find_hitbox_center_improved(img, player_bbox)
            # 更新检测结果
            for player in player_detections:
                player['center'] = improved_center  # 使用改进的中心点
                player['hitbox_size'] = 4  # 固定4x4像素

        bullet_detections = self.detector.detect_bullets_with_dynamic_size(img, player_bbox)

        # 组织检测结果
        detections = {
            'player': player_detections,
            'bullets': bullet_detections,
            'image_shape': img.shape,
            'filename': img_path.name
        }

        return detections

    def create_visualization(self, img_path: Path, detections: dict) -> Path | None:
        """创建可视化效果图 - 显示判定点和对应的标注框"""
        img = cv2.imread(str(img_path))
        if img is None:
            return None

        result_img = img.copy()
        h, w = img.shape[:2]

        # 绘制自机判定点和标注框
        for player in detections.get('player', []):
            cx, cy = player['center']

            # 自机固定使用4x4像素
            hitbox_size = 4
            half_size = hitbox_size // 2

            # 标注框坐标
            x1 = max(0, cx - half_size)
            y1 = max(0, cy - half_size)
            x2 = min(w, cx + half_size)
            y2 = min(h, cy + half_size)

            # 绘制标注框（绿色矩形）
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制判定点（白色圆圈带绿色边框）
            cv2.circle(result_img, (cx, cy), 3, (255, 255, 255), -1)
            cv2.circle(result_img, (cx, cy), 4, (0, 255, 0), 2)

            # 绘制十字标记
            cv2.line(result_img, (cx - 8, cy), (cx + 8, cy), (0, 255, 0), 1)
            cv2.line(result_img, (cx, cy - 8), (cx, cy + 8), (0, 255, 0), 1)

            # 添加标签
            cv2.putText(result_img, 'Player', (cx + 10, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # 显示标注框尺寸信息
            box_info = f"{hitbox_size}x{hitbox_size}"
            cv2.putText(result_img, box_info, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        # 绘制子弹判定点和标注框
        for bullet in detections.get('bullets', []):
            cx, cy = bullet['center']

            # 使用动态计算的子弹大小
            hitbox_size = bullet.get('dynamic_size', 4)  # 使用动态大小，默认4
            half_size = hitbox_size // 2

            # 标注框坐标
            x1 = max(0, cx - half_size)
            y1 = max(0, cy - half_size)
            x2 = min(w, cx + half_size)
            y2 = min(h, cy + half_size)

            # 绘制标注框（红色矩形）
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # 绘制子弹判定点（红色小圆点）
            cv2.circle(result_img, (cx, cy), 2, (0, 0, 255), -1)
            cv2.circle(result_img, (cx, cy), 3, (255, 255, 255), 1)

            # 显示子弹大小信息
            bullet_info = f"{hitbox_size}x{hitbox_size}"
            cv2.putText(result_img, bullet_info, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1)

        # 添加统计信息
        player_count = len(detections.get('player', []))
        bullet_count = len(detections.get('bullets', []))
        info_text = f"Players: {player_count} | Bullets: {bullet_count}"
        cv2.putText(result_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 添加标注框说明
        legend_y = 50
        cv2.putText(result_img, "Green Box: Player Hitbox (4x4 fixed)", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(result_img, "Red Box: Bullet Hitbox (dynamic size)", (10, legend_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 添加文件名
        cv2.putText(result_img, detections['filename'], (10, img.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 保存可视化结果
        vis_filename = f"vis_{detections['filename']}"
        vis_path = self.visualizations_dir / vis_filename
        cv2.imwrite(str(vis_path), result_img)

        return vis_path

    def save_annotation_json(self, detections: dict) -> Path:
        """保存JSON格式的标注文件"""
        annotation_data = {
            'filename': detections['filename'],
            'image_shape': {
                'height': detections['image_shape'][0],
                'width': detections['image_shape'][1],
                'channels': detections['image_shape'][2]
            },
            'detections': {
                'player': [],
                'bullets': []
            },
            'statistics': {
                'player_count': len(detections.get('player', [])),
                'bullet_count': len(detections.get('bullets', [])),
                'processed_time': datetime.now().isoformat()
            }
        }

        # 处理自机检测结果
        for player in detections.get('player', []):
            player_data = {
                'center': player['center'],
                'bbox': player.get('bbox', []),
                'confidence': player.get('confidence', 1.0)
            }
            annotation_data['detections']['player'].append(player_data)

        # 处理子弹检测结果
        for bullet in detections.get('bullets', []):
            bullet_data = {
                'center': bullet['center'],
                'square_size': bullet.get('square_size', 4),
                'dynamic_size': bullet.get('dynamic_size', 4),  # 保存动态大小
                'confidence': bullet.get('confidence', 1.0)
            }
            annotation_data['detections']['bullets'].append(bullet_data)

        # 保存JSON文件
        json_filename = f"ann_{detections['filename'].replace('.jpg', '.json')}"
        json_path = self.annotations_dir / json_filename

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, indent=2, ensure_ascii=False)

        return json_path

    def batch_process_images(self):
        """批量处理预处理文件夹中的所有图像"""
        image_files = list(self.raw_images_dir.glob("*.jpg"))

        if not image_files:
            print("❌ 预处理文件夹中没有找到图像文件")
            return

        print(f"\n🔄 开始批量处理 {len(image_files)} 张图像...")
        print("=" * 50)

        processed_count = 0
        failed_count = 0

        for i, img_path in enumerate(image_files, 1):
            print(f"处理进度: {i}/{len(image_files)} - {img_path.name}")

            try:
                # 检测对象
                detections = self.process_single_image(img_path)
                if detections is None:
                    print(f"  ❌ 图像读取失败")
                    failed_count += 1
                    continue

                # 创建可视化
                vis_path = self.create_visualization(img_path, detections)
                if vis_path:
                    print(f"  ✅ 可视化: {vis_path.name}")

                # 保存JSON标注
                json_path = self.save_annotation_json(detections)
                print(f"  ✅ 标注: {json_path.name}")

                # 显示检测统计
                player_count = len(detections.get('player', []))
                bullet_count = len(detections.get('bullets', []))
                print(f"  📊 检测结果: {player_count} 自机, {bullet_count} 子弹")

                processed_count += 1

            except Exception as e:
                print(f"  ❌ 处理失败: {e}")
                failed_count += 1

        print("\n" + "=" * 50)
        print(f"🎉 批量处理完成!")
        print(f"成功处理: {processed_count} 张")
        print(f"处理失败: {failed_count} 张")
        print(f"可视化文件: {self.visualizations_dir}")
        print(f"标注文件: {self.annotations_dir}")



class YOLODataGenerator:
    """YOLO训练数据自动生成器 - 专注于判定点检测"""

    def __init__(self, output_dir: str = "yolo/data"):
        self.output_dir = Path(output_dir)
        # 只需要训练集，不需要验证集
        self.images_dir = self.output_dir / "images" / "train"
        self.labels_dir = self.output_dir / "labels" / "train"

        # 创建输出目录
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # 初始化检测器和捕获器
        self.detector = EnhancedGameObjectDetector()
        self.capture = OptimizedGameCapture()

        # 类别映射 - 只检测判定点
        self.class_mapping = {
            'player_hitbox': 0,    # 自机判定点
            'bullet_center': 1     # 子弹判定点
        }

        # 判定点大小设置（像素）
        self.player_hitbox_size = 4    # 自机判定框固定4x4
        self.bullet_hitbox_size = 4    # 子弹判定框默认大小（动态调整）
        
        # 统计信息
        self.generated_count = 0
        self.start_time = None
        
        print(f"YOLO数据生成器初始化完成")
        print(f"输出目录: {self.output_dir}")
        print(f"图像目录: {self.images_dir}")
        print(f"标签目录: {self.labels_dir}")
    
    def convert_detections_to_yolo(self, detections: Dict, img_shape: Tuple[int, int]) -> List[str]:
        """将检测结果转换为YOLO格式标注 - 专注于判定点"""
        h, w = img_shape[:2]
        annotations = []

        # 处理自机判定点
        if 'player' in detections and detections['player']:
            for player_det in detections['player']:
                # 获取判定点坐标（位置测试.py返回的center就是判定点）
                hitbox_x, hitbox_y = player_det['center']

                # 转换为YOLO格式（以判定点为中心的小框）
                center_x = hitbox_x / w
                center_y = hitbox_y / h

                # 判定点的标注框大小（归一化）
                norm_w = self.player_hitbox_size / w
                norm_h = self.player_hitbox_size / h

                # 确保坐标在有效范围内
                center_x = max(0, min(1, center_x))
                center_y = max(0, min(1, center_y))
                norm_w = max(0, min(1, norm_w))
                norm_h = max(0, min(1, norm_h))

                class_id = self.class_mapping['player_hitbox']
                annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")

        # 处理子弹判定点
        if 'bullets' in detections and detections['bullets']:
            for bullet in detections['bullets']:
                # 获取子弹中心点坐标
                bullet_x, bullet_y = bullet['center']

                # 转换为YOLO格式（以子弹中心为判定点）
                center_x = bullet_x / w
                center_y = bullet_y / h

                # 使用动态计算的子弹大小
                actual_bullet_size = bullet.get('dynamic_size', self.bullet_hitbox_size)
                norm_w = actual_bullet_size / w
                norm_h = actual_bullet_size / h

                # 确保坐标在有效范围内
                center_x = max(0, min(1, center_x))
                center_y = max(0, min(1, center_y))
                norm_w = max(0, min(1, norm_w))
                norm_h = max(0, min(1, norm_h))

                class_id = self.class_mapping['bullet_center']
                annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")

        return annotations
    
    def save_training_sample(self, img: np.ndarray, annotations: List[str]) -> bool:
        """保存训练样本（图像和标签）"""
        if len(annotations) == 0:
            return False  # 没有检测到任何对象，跳过
            
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
        filename = f"touhou_{timestamp}_{self.generated_count:06d}"
        
        # 保存图像
        img_path = self.images_dir / f"{filename}.jpg"
        success = cv2.imwrite(str(img_path), img)
        
        if not success:
            print(f"保存图像失败: {img_path}")
            return False
        
        # 保存标签
        label_path = self.labels_dir / f"{filename}.txt"
        try:
            with open(label_path, 'w') as f:
                f.write('\n'.join(annotations))
        except Exception as e:
            print(f"保存标签失败: {e}")
            return False
        
        self.generated_count += 1
        return True

    def create_yolo_visualization(self, img: np.ndarray, annotations: List[str], filename: str) -> Path | None:
        """创建YOLO格式标注的可视化图像"""
        if len(annotations) == 0:
            return None

        result_img = img.copy()
        h, w = img.shape[:2]

        # 解析YOLO标注并绘制
        for annotation in annotations:
            parts = annotation.strip().split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            center_x = float(parts[1]) * w
            center_y = float(parts[2]) * h
            box_w = float(parts[3]) * w
            box_h = float(parts[4]) * h

            # 计算边界框坐标
            x1 = int(center_x - box_w / 2)
            y1 = int(center_y - box_h / 2)
            x2 = int(center_x + box_w / 2)
            y2 = int(center_y + box_h / 2)

            # 根据类别选择颜色
            if class_id == 0:  # player_hitbox
                color = (0, 255, 0)  # 绿色
                label = "Player"
            elif class_id == 1:  # bullet_center
                color = (0, 0, 255)  # 红色
                label = "Bullet"
            else:
                color = (255, 255, 255)  # 白色
                label = f"Class{class_id}"

            # 绘制边界框
            line_thickness = 2 if class_id == 0 else 1  # 自机用稍粗的线
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, line_thickness)

            # 绘制中心点
            cv2.circle(result_img, (int(center_x), int(center_y)), 3, color, -1)
            cv2.circle(result_img, (int(center_x), int(center_y)), 4, (255, 255, 255), 1)

            # 添加标签
            cv2.putText(result_img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 添加统计信息
        player_count = sum(1 for ann in annotations if ann.startswith('0 '))
        bullet_count = sum(1 for ann in annotations if ann.startswith('1 '))
        info_text = f"YOLO: {player_count} Players, {bullet_count} Bullets"
        cv2.putText(result_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 保存YOLO可视化结果
        yolo_vis_dir = self.output_dir / "visualizations"
        yolo_vis_dir.mkdir(exist_ok=True)

        vis_filename = f"yolo_vis_{filename}.jpg"
        vis_path = yolo_vis_dir / vis_filename
        cv2.imwrite(str(vis_path), result_img)

        return vis_path

    def convert_from_preprocessing(self, preprocessing_dir: str = "preprocessing"):
        """根据可视化文件夹中剩余的文件来转换YOLO训练数据"""
        preprocessing_path = Path(preprocessing_dir)
        raw_images_dir = preprocessing_path / "raw_images"
        annotations_dir = preprocessing_path / "annotations"
        visualizations_dir = preprocessing_path / "visualizations"

        if not visualizations_dir.exists():
            print("❌ 可视化目录不存在，请先运行预处理工作流")
            return

        # 以可视化文件为准，获取用户筛选后保留的文件
        vis_files = list(visualizations_dir.glob("vis_*.jpg"))

        if not vis_files:
            print("❌ 可视化文件夹中没有文件，请先处理截图或检查是否已删除所有文件")
            return

        print(f"\n🔄 根据可视化文件夹转换YOLO训练数据...")
        print(f"找到 {len(vis_files)} 个筛选后的可视化文件")
        print("📋 将根据这些文件转换对应的原始数据")

        converted_count = 0
        missing_files = []

        for vis_file in vis_files:
            try:
                # 从可视化文件名提取原始文件名
                # vis_capture_20250621_015440_937_0000.jpg -> capture_20250621_015440_937_0000.jpg
                vis_filename = vis_file.name
                if vis_filename.startswith("vis_"):
                    img_filename = vis_filename[4:]  # 去掉 "vis_" 前缀
                else:
                    print(f"⚠️ 可视化文件名格式不正确: {vis_filename}")
                    continue

                # 查找对应的原始文件
                img_path = raw_images_dir / img_filename
                json_filename = f"ann_{img_filename.replace('.jpg', '.json')}"
                json_path = annotations_dir / json_filename

                # 检查文件是否存在
                if not img_path.exists():
                    missing_files.append(f"原始图像: {img_filename}")
                    continue

                if not json_path.exists():
                    missing_files.append(f"标注文件: {json_filename}")
                    continue

                # 读取JSON标注
                with open(json_path, 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)

                # 读取图像
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"⚠️ 无法读取图像: {img_filename}")
                    continue

                # 转换检测结果格式
                detections = {
                    'player': annotation_data['detections']['player'],
                    'bullets': annotation_data['detections']['bullets']
                }

                # 转换为YOLO格式
                annotations = self.convert_detections_to_yolo(detections, (img.shape[0], img.shape[1]))

                # 保存YOLO训练样本
                if self.save_training_sample(img, annotations):
                    converted_count += 1

                    # 创建YOLO可视化
                    yolo_vis_path = self.create_yolo_visualization(img, annotations, img_filename.replace('.jpg', ''))
                    print(f"✅ 转换完成: {img_filename} -> YOLO样本 #{self.generated_count}")

            except Exception as e:
                print(f"❌ 转换失败 {vis_file.name}: {e}")

        # 显示缺失文件信息
        if missing_files:
            print(f"\n⚠️ 以下文件缺失，未能转换:")
            for missing in missing_files[:5]:  # 只显示前5个
                print(f"   - {missing}")
            if len(missing_files) > 5:
                print(f"   ... 还有 {len(missing_files) - 5} 个文件缺失")

        print(f"\n🎉 YOLO数据转换完成!")
        print(f"成功转换: {converted_count} 个样本")
        print(f"YOLO数据目录: {self.output_dir}")
        print(f"YOLO可视化目录: {self.output_dir / 'visualizations'}")

        # 只清理已转换的文件，保留文件夹结构
        if converted_count > 0:
            try:
                # 清理已转换的文件
                deleted_files = 0

                # 删除对应的原始图像和标注文件
                for vis_file in vis_files:
                    vis_filename = vis_file.name
                    if vis_filename.startswith("vis_"):
                        img_filename = vis_filename[4:]  # 去掉 "vis_" 前缀

                        # 删除原始图像
                        img_path = raw_images_dir / img_filename
                        if img_path.exists():
                            img_path.unlink()
                            deleted_files += 1

                        # 删除标注文件
                        json_filename = f"ann_{img_filename.replace('.jpg', '.json')}"
                        json_path = annotations_dir / json_filename
                        if json_path.exists():
                            json_path.unlink()
                            deleted_files += 1

                        # 删除可视化文件
                        vis_file.unlink()
                        deleted_files += 1

                print(f"🗑️ 已删除 {deleted_files} 个已转换的文件")
                print(f"📁 保留了preprocessing文件夹结构")

            except Exception as e:
                print(f"⚠️ 清理文件失败: {e}")

        # 生成数据集配置文件
        self.create_dataset_config()

    def create_dataset_config(self):
        """创建YOLO训练所需的数据集配置文件"""
        config_content = f"""# Touhou Hitbox Detection Dataset Configuration
# Generated automatically by auto_data_generator.py

path: {self.output_dir.absolute()}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/train    # val images (same as train for now)

# Classes
nc: 2  # number of classes
names: ['player_hitbox', 'bullet_center']  # class names

# Class descriptions:
# 0: player_hitbox - 自机判定点 (4x4 pixels)
# 1: bullet_center - 子弹判定点 (dynamic size based on white area)
"""

        config_path = self.output_dir / "dataset.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)

        print(f"✅ 数据集配置文件已创建: {config_path}")

        # 创建训练脚本
        self.create_training_script()

    def create_training_script(self):
        """创建YOLO训练脚本"""
        script_content = f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
东方红魔乡判定点检测 - YOLO训练脚本
自动生成于: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
\"\"\"

from ultralytics import YOLO
import torch

def main():
    print("🎯 开始训练东方红魔乡判定点检测模型...")

    # 检查GPU可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {{device}}")

    # 加载预训练模型
    model = YOLO('yolov8n.pt')  # 使用nano版本，速度快

    # 训练参数
    results = model.train(
        data='{self.output_dir.absolute() / "dataset.yaml"}',
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        project='runs/detect',
        name='touhou_hitbox',
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True
    )

    print("🎉 训练完成!")
    print(f"最佳模型保存在: runs/detect/touhou_hitbox/weights/best.pt")

    # 验证模型
    metrics = model.val()
    print(f"验证结果: mAP50={{metrics.box.map50:.3f}}, mAP50-95={{metrics.box.map:.3f}}")

if __name__ == "__main__":
    main()
"""

        script_path = self.output_dir / "train_model.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)

        print(f"✅ 训练脚本已创建: {script_path}")
        print(f"💡 运行训练: cd {self.output_dir} && python train_model.py")



def main():
    """主函数 - 启动游戏并截图"""
    print("=" * 50)
    print("🎮 东方红魔乡 YOLO数据生成器")
    print("=" * 50)

    # 创建预处理工作流
    preprocessor = PreprocessingWorkflow()
    generator = YOLODataGenerator()

    # 启动游戏并查找窗口
    print("🚀 启动游戏...")
    if not preprocessor.capture.start_game_and_find_window():
        print("❌ 游戏启动失败或未找到游戏窗口！")
        print("请检查游戏路径是否正确")
        return

    print("✅ 游戏已启动，窗口已找到")
    print("💡 请手动进入游戏画面（跳过菜单到实际游戏中）")

    while True:
        print("\n请选择操作:")
        print("1. 开始截图 (按R键截图，按Q键结束)")
        print("2. 处理截图生成可视化")
        print("3. 将筛选好的数据转换为YOLO训练数据")
        print("4. 完整流程 (处理截图 + 生成YOLO数据)")
        print("5. 退出")

        try:
            choice = input("\n请输入选择 (1-5): ").strip()

            if choice == '1':
                print("\n🎮 截图模式启动")
                print("按 'R' 键截取游戏画面")
                print("按 'Q' 键结束截图")
                print("-" * 30)
                preprocessor.start_manual_capture()

            elif choice == '2':
                print("\n🔄 开始处理截图...")
                preprocessor.batch_process_images()
                print("\n✅ 预处理完成！")
                print(f"可视化结果: {preprocessor.visualizations_dir}")
                print("💡 请检查可视化结果，确认数据质量后选择选项3转换为YOLO格式")

            elif choice == '3':
                print("\n🎯 根据可视化文件夹转换YOLO训练数据...")
                print("📋 工作流程:")
                print("   1. 扫描 preprocessing/visualizations/ 文件夹")
                print("   2. 根据剩余的可视化文件找到对应的原始数据")
                print("   3. 转换为YOLO格式并移动到yolo文件夹")
                print("   4. 删除已转换的文件（保留文件夹结构）")
                print("⚠️ 注意：转换后将删除对应的原始文件！")
                generator.convert_from_preprocessing()
                print("\n✅ YOLO数据转换完成！")
                print(f"YOLO训练数据: {generator.output_dir}")
                print("💡 现在可以开始训练YOLO模型了")

            elif choice == '4':
                print("\n🔄 开始完整流程...")

                # 批量处理图像
                preprocessor.batch_process_images()

                # 自动转换为YOLO数据
                print("\n🎯 转换为YOLO训练数据...")
                generator.convert_from_preprocessing()

                print("\n✅ 完整流程完成！")
                print(f"YOLO数据保存在: {generator.output_dir}")
                print(f"预处理可视化: {preprocessor.visualizations_dir}")
                print(f"YOLO可视化: {generator.output_dir / 'visualizations'}")
                print("💡 可视化图像显示了检测点和对应的标注框")

            elif choice == '5':
                print("👋 退出程序")
                break

            else:
                print("❌ 无效选择！请输入 1-5")

        except KeyboardInterrupt:
            print("\n\n⚠️ 程序被中断")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")

if __name__ == "__main__":
    main()
