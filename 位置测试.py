#位置测试.py
import numpy as np
import cv2
import sys
from pathlib import Path

class GameObjectDetector:
    """游戏对象检测器（改进版本）"""
    
    def __init__(self):
        self.player_template = None
        self.load_player_template()
        self.enable_color_fallback = True
    
    def load_player_template(self):
        """加载自机模板图像"""
        try:
            self.player_template = cv2.imread('reimu.png', cv2.IMREAD_GRAYSCALE)
            if self.player_template is not None:
                h, w = self.player_template.shape
                print(f"自机模板加载成功，大小: {w}x{h}")
            else:
                print("警告：未找到自机模板reimu.png")
        except Exception as e:
            print(f"警告：加载自机模板失败: {e}")
    
    def find_hitbox_center(self, img, player_bbox):
        """在自机区域内查找白色判定点"""
        x, y, w, h = player_bbox
        
        # 判定点通常在自机的中心区域，不需要扩大搜索范围
        # 实际上，我们应该缩小搜索范围到自机中心
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
            # 判定点通常很小，面积在5-50之间（比之前的范围更小）
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
                            if distance < search_radius // 2:  # 只考虑非常接近中心的点
                                if distance < min_distance:
                                    min_distance = distance
                                    best_center = (cx + x1, cy + y1)  # 转换回原图坐标
        
        # 如果没找到合适的白色圆点，使用自机中心
        if best_center is None:
            best_center = (center_x, center_y)
            print("警告：未找到判定点，使用自机中心")
        else:
            # 验证找到的点是否真的在自机边界内
            if not (x <= best_center[0] <= x + w and y <= best_center[1] <= y + h):
                best_center = (center_x, center_y)
                print("警告：检测到的判定点在自机外部，使用自机中心")
        
        return best_center
    def detect_player_position(self, img):
        """检测自机位置"""
        results = []
        
        # 方法1：模板匹配（已注释）
        # if self.player_template is not None:
        #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        #     
        #     result = cv2.matchTemplate(gray, self.player_template, cv2.TM_CCOEFF_NORMED)
        #     threshold = 0.2
        #     locations = np.where(result >= threshold)
        #     
        #     matches = []
        #     for pt in zip(*locations[::-1]):
        #         matches.append({
        #             'x': pt[0],
        #             'y': pt[1],
        #             'confidence': result[pt[1], pt[0]]
        #         })
        #     
        #     if matches:
        #         matches.sort(key=lambda x: x['confidence'], reverse=True)
        #         
        #         for match in matches:
        #             player_x = match['x']
        #             player_y = match['y']
        #             w, h = self.player_template.shape[::-1]
        #             
        #             roi = img[player_y:player_y+h, player_x:player_x+w]
        #             if roi.size > 0:
        #                 b, g, r = cv2.split(roi)
        #                 red_dominance = np.mean(r) - np.mean(b)
        #                 
        #                 if red_dominance > 10:
        #                     bbox = (player_x, player_y, w, h)
        #                     # 查找判定点
        #                     hitbox_center = self.find_hitbox_center(img, bbox)
        #                     
        #                     results.append({
        #                         'method': 'template',
        #                         'confidence': match['confidence'],
        #                         'bbox': bbox,
        #                         'center': hitbox_center  # 使用判定点作为中心
        #                     })
        #                     return results
          # 方法2：颜色检测（主要方法）
        if self.enable_color_fallback:
            print("使用颜色检测...")
            
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
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
                        # 查找判定点
                        hitbox_center = self.find_hitbox_center(img, bbox)
                        
                        results.append({
                            'method': 'color',
                            'confidence': 0.5,
                            'bbox': bbox,
                            'center': hitbox_center
                        })
                        print(f"使用颜色检测找到自机: 判定点位置={hitbox_center}")
        
        if not results:
            print("警告：未能检测到自机！")
        
        return results
    
    def detect_bullets(self, img, player_bbox=None):
        """检测子弹位置"""
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
        
        # 使用亮度范围检测子弹 (从min_brightness到max_brightness)
        min_brightness = 230  # 最低亮度阈值
        max_brightness = 255  # 最高亮度阈值
        
        # 创建亮度范围掩码
        brightness_mask = cv2.inRange(gray, min_brightness, max_brightness)
        
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
                                    'brightness': actual_brightness,  # 记录实际亮度值
                                    'square_size': square_size * 2
                                })
        
        return bullets


def test_detection(image_path, output_path=None):
    """测试检测功能"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图片 {image_path}")
        return
    
    print(f"图片大小: {img.shape[1]}x{img.shape[0]}")
    
    detector = GameObjectDetector()
    
    print("\n检测自机...")
    player_detections = detector.detect_player_position(img)
    
    player_bbox = None
    if player_detections:
        player_bbox = player_detections[0]['bbox']
    
    print("\n检测子弹...")
    bullet_detections = detector.detect_bullets(img, player_bbox)
    
    result_img = img.copy()
    
    # 绘制自机检测结果
    for i, player in enumerate(player_detections):
        x, y, w, h = player['bbox']
        cx, cy = player['center']
        
        # 绘制红色边界框
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # 绘制判定点（白色圆圈带黑色边框）
        cv2.circle(result_img, (cx, cy), 5, (255, 255, 255), -1)
        cv2.circle(result_img, (cx, cy), 6, (0, 0, 0), 2)
        
        # 绘制一个小十字标记在判定点上
        cv2.line(result_img, (cx - 8, cy), (cx + 8, cy), (255, 0, 255), 1)
        cv2.line(result_img, (cx, cy - 8), (cx, cy + 8), (255, 0, 255), 1)
        
        # 根据匹配方法显示不同的信息
        if player['method'] == 'template':
            print(f"  自机 {i+1}: 【模板匹配成功】判定点位置=({cx}, {cy}), 置信度={player['confidence']:.2f}")
        elif player['method'] == 'color':
            print(f"  自机 {i+1}: 【颜色检测成功】判定点位置=({cx}, {cy}), 置信度={player['confidence']:.2f}")
        else:
            print(f"  自机 {i+1}: 【未知方法】判定点位置=({cx}, {cy}), 方法={player['method']}, 置信度={player['confidence']:.2f}")
    
    # 绘制子弹检测结果
    print(f"\n检测到 {len(bullet_detections)} 个子弹")
    for bullet in bullet_detections:
        cx, cy = bullet['center']
        square_size = bullet.get('square_size', 4)
        
        # 绘制更小的中心实心绿色矩形
        half_size = square_size // 2
        cv2.rectangle(result_img, 
                     (cx - half_size, cy - half_size), 
                     (cx + half_size, cy + half_size), 
                     (0, 255, 0), -1)
    
    # 添加统计信息
    info_text = f"Players: {len(player_detections)} | Bullets: {len(bullet_detections)}"
    cv2.putText(result_img, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    if output_path:
        cv2.imwrite(output_path, result_img)
        print(f"\n结果已保存到: {output_path}")
    
    cv2.imshow("Detection Result", result_img)
    
    # 调试信息
    if len(player_detections) == 0:
        print("\n警告：未检测到自机！")
        print("可能的原因：")
        print("1. 模板图像与游戏中的自机不匹配")
        print("2. 游戏画面分辨率或缩放比例不同")
        print("3. 自机被遮挡或处于特殊状态")
        
    if len(player_detections) == 0 or len(bullet_detections) < 5:
        print("\n显示调试信息...")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Grayscale", gray)
        
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        cv2.imshow("Threshold", thresh)
        
        if detector.player_template is not None:
            cv2.imshow("Player Template", detector.player_template)
            
        if len(player_detections) == 0 and detector.player_template is not None:
            result = cv2.matchTemplate(gray, detector.player_template, cv2.TM_CCOEFF_NORMED)
            result_normalized = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            result_colored = cv2.applyColorMap(result_normalized, cv2.COLORMAP_JET)
            cv2.imshow("Template Match Heatmap", result_colored)
    
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """主函数"""
    script_dir = Path(__file__).parent
    
    if len(sys.argv) < 2:
        image_path = script_dir / "test1.png"
        output_path = script_dir / "detection_result.png"
    else:
        image_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else script_dir / "detection_result.png"
    
    if not Path(image_path).exists():
        print(f"错误：文件不存在: {image_path}")
        print("使用方法:")
        print("  python test_detection.py <图片路径> [输出路径]")
        return
    
    test_detection(str(image_path), str(output_path))


if __name__ == "__main__":
    main()