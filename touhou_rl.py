# touhou_rl.py
import numpy as np
import mss
import cv2
import time
import subprocess
import win32gui
import keyboard
import pytesseract

#-----------------------------
#   1. 截屏 + OCR + HSV识别类
#-----------------------------
class OptimizedGameCapture:
    """
    一次性截取整个游戏窗口(640x480)，再通过在CPU内对 NumPy 数组进行切片，
    拿到弹幕区(384×448)和状态区(192×448)的图像，从而避免双重坐标混乱。
    """
    def __init__(self, window_name="搶曽峠杺嫿丂乣 the Embodiment of Scarlet Devil"):
        self.sct = mss.mss()
        self.window_name = window_name
        self.window_rect = None   # (left, top, width, height)

        # 东方红魔乡常见原生分辨率
        self.game_width  = 640
        self.game_height = 480

        # 弹幕区(在640x480下): 左上(32,16), 宽384, 高448
        self.gameplay_x, self.gameplay_y = 32, 16
        self.gameplay_width,  self.gameplay_height  = 384, 448

        # 状态区(在640x480下): 左上(448,16), 宽192, 高448
        self.status_x,   self.status_y   = 448, 16
        self.status_width, self.status_height = 192, 448

        # Tesseract OCR 路径
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        # 是否可用 GPU (仅用于演示，可选择CPU或GPU)
        self.use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.use_gpu:
            print("使用GPU加速截屏处理")
            self.gpu_cvt = cv2.cuda.createColorConversion(code=cv2.COLOR_BGRA2GRAY)
        else:
            print("使用CPU处理截屏")

    def set_window_by_name(self):
        """
        根据self.window_name查找游戏窗口，把它的在屏幕上的坐标(含边框)记录到self.window_rect。
        如果找不到，self.window_rect会是None。
        """
        def callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd) == self.window_name:
                rect = win32gui.GetWindowRect(hwnd)
                client = win32gui.GetClientRect(hwnd)
                border_w = ((rect[2] - rect[0]) - client[2]) // 2
                title_h  = (rect[3] - rect[1]) - client[3] - border_w
                self.window_rect = (
                    rect[0] + border_w,
                    rect[1] + title_h,
                    client[2],
                    client[3]
                )
                return False
            return True

        win32gui.EnumWindows(callback, None)
        if self.window_rect:
            print(f"找到游戏窗口: {self.window_name}")
        else:
            print(f"警告：未找到窗口 [{self.window_name}]，截屏可能是黑屏！")

    def capture_full_window(self):
        """
        一次性截取整个640×480游戏窗口，返回格式为 BGRA 的 np.array (480,640,4)。
        如果window_rect是None，则返回纯黑图。
        """
        if not self.window_rect:
            return np.zeros((self.game_height, self.game_width, 4), dtype=np.uint8)
        
        left, top, width, height = self.window_rect
        if width != self.game_width or height != self.game_height:

            pass

        monitor = {
            "left":   left,
            "top":    top,
            "width":  self.game_width,
            "height": self.game_height
        }

        # 截屏
        full_img = np.array(self.sct.grab(monitor))  # shape=(480,640,4)
        return full_img

    def capture_frame(self):
        """
        获取游戏当前帧，并返回:
          1) 处理过的弹幕区(96×128灰度，缩放后)
          2) (score, lives, bombs)
        """
        full_img = self.capture_full_window()  # shape ~ (480,640,4)

        # 1) 提取弹幕区 gameplay_img
        gameplay_img = full_img[
            self.gameplay_y : self.gameplay_y + self.gameplay_height,
            self.gameplay_x : self.gameplay_x + self.gameplay_width
        ]  # shape(448,384,4)

        # 转灰度
        if self.use_gpu:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(gameplay_img)
            gray_frame = self.gpu_cvt.apply(gpu_frame)
            gray_frame = gray_frame.download()
        else:
            gray_frame = cv2.cvtColor(gameplay_img, cv2.COLOR_BGRA2GRAY)
        
        # 缩放到96x128以减少内存使用
        scaled_frame = cv2.resize(gray_frame, (128, 96), interpolation=cv2.INTER_AREA)

        # 2) 提取状态区 status_img
        status_img = full_img[
            self.status_y : self.status_y + self.status_height,
            self.status_x : self.status_x + self.status_width
        ]  # shape(448,192,4)

        # 从status_img中OCR/HSV解析 (score, lives, bombs)
        score, lives, bombs = self.extract_status_info(status_img)

        return scaled_frame, (score, lives, bombs)

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
        self.sct.close()


#-------------------------------------
#   2. 强化学习环境 TouhouEnv
#-------------------------------------
class TouhouEnv:
    """
    使用一次性截全屏的方式获取图像, 并将弹幕区(448×384)作为state。
    """
    def __init__(self, game_path="D:\\Games\\th06\\vpatch.exe"):
        self.game_path = game_path
        self.window_name = "搶曽峠杺嫿丂乣 the Embodiment of Scarlet Devil"
        self.cap = None  # 我们稍后再初始化

        self.frame_stack = 4
        self.frames = []

        self.action_space = 9   # 8方向 + 不动 (移除shift动作组合)
        self.observation_space = (4, 96, 128)  # 更新为缩放后的分辨率

        self.last_score = 0
        self.start_time = None
        self.max_steps = 10000
        self.current_step = 0
        self.first_reset = True
        
        # 炸弹系统 (暂时注释掉)
        # self.bombs = 3  # 初始炸弹数为3
        # self.initial_bombs = 3
        # self.last_lives = 2  # 记录上一次的生命数，用于检测生命变化       
        # 炸弹冷却系统 (暂时注释掉)
        # self.bomb_cooldown = 5.0  # 5秒冷却时间
        # self.last_bomb_time = 0  # 上次使用炸弹的时间

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
            # 移除了所有包含shift的动作组合
            # 9: ['x']            # bomb (注释掉)
        }

        # 先启动游戏，再初始化截屏类
        self._open_game()
        self.cap = OptimizedGameCapture(window_name=self.window_name)
        self.cap.set_window_by_name()

        # 初始化 frame stack
        for _ in range(self.frame_stack):
            self.frames.append(np.zeros((96, 128), dtype=np.uint8))

    def _open_game(self):
        """
        启动游戏，并等待窗口出现
        """
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
        """
        首次重置时自动按键进入游戏的简易示例。
        具体按键流程需根据你自己的版本/菜单界面来调试。
        """
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
        # 默认一直按住Z键射击
        keyboard.press('z')

    def _restart_game(self):
        """
        游戏结束后重新开始的按键流程示例。可根据需要修改。
        """
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
            for i in range(5):
                keyboard.press('z')
                time.sleep(0.2)
                keyboard.release('z')
                print(f"按Z键 {i+1}/5")
                time.sleep(0.5)
            print("游戏重新开始完成，开始持续射击")
            # 重新开始后也要按住Z键进行射击
            keyboard.press('z')

            return True
        except Exception as e:
            print(f"重启游戏时出现异常: {e}")
            return False

    def reset(self):
        print("\n------ 重置游戏环境 ------")
        # 释放按键
        keyboard.release('z')
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
        # 重置炸弹数 (注释掉)
        # self.bombs = self.initial_bombs
        # self.last_lives = 2  # 重置生命记录（东方游戏通常初始3条命）
        # self.last_bomb_time = 0

        # 首次reset vs 后续reset
        if self.first_reset:
            print("首次重置，启动游戏...")
            self._start_game()
            self.first_reset = False
        else:
            print("再次重置，重启游戏...")
            self._restart_game()

        # 获取若干帧(初始化帧堆叠)
        print("获取初始游戏状态...")
        for _ in range(self.frame_stack):
            frame_96x128, (score, lives, _) = self.cap.capture_frame()  # 不再使用OCR的炸弹数
            #print(f"游戏状态: 分数={score}, 生命={lives}, 炸弹={self.bombs}")
            print(f"游戏状态: 分数={score}, 生命={lives}")
            self.frames.append(frame_96x128)
            self.frames.pop(0)

        print("环境重置完成，准备开始训练")
        return self._get_state()

    def step(self, action):
        self.current_step += 1
        current_time = time.time()
        
        # 检查炸弹动作是否可执行，并计算相应奖励 (注释掉炸弹逻辑)
        # bomb_penalty = 0
        # if action == 9:  # 炸弹动作
        #     time_since_last_bomb = current_time - self.last_bomb_time
        #     
        #     if time_since_last_bomb < self.bomb_cooldown:
        #         # 还在冷却中
        #         remaining_cooldown = self.bomb_cooldown - time_since_last_bomb
        #         print(f"炸弹冷却中！剩余冷却时间: {remaining_cooldown:.1f}秒")
        #         # 给予负奖励，让AI学会不要在冷却期间尝试使用炸弹
        #         bomb_penalty = -5
        #         # 不执行炸弹动作，改为不动
        #         action = 0
        #     elif self.bombs <= 0:
        #         print(f"炸弹已用完，选择炸弹动作给予负奖励！当前炸弹数: {self.bombs}")
        #         # 给予负奖励但不改变动作（让AI学会不要在没炸弹时选择炸弹）
        #         bomb_penalty = -10
        #         # 不执行炸弹动作，改为不动
        #         action = 0
        #     else:
        #         # 可以使用炸弹
        #         self.bombs -= 1
        #         self.last_bomb_time = current_time
        #         print(f"使用炸弹！剩余炸弹数: {self.bombs}，下次可用时间: {self.bomb_cooldown}秒后")
        
        # 按下对应方向键
        for k in self.action_map[action]:
            keyboard.press(k)

        # 小延时模拟帧
        time.sleep(1/60)

        # 截屏 + OCR (只获取分数和生命，不再需要炸弹数)
        frame_96x128, (score, lives, _) = self.cap.capture_frame()  # 忽略OCR的炸弹数
        self.frames.append(frame_96x128)
        self.frames.pop(0)

        # 检查生命变化，如果生命减少则重置炸弹数 (注释掉炸弹逻辑)
        # if lives < self.last_lives:
        #     print(f"检测到生命减少：{self.last_lives} -> {lives}，重置炸弹数为{self.initial_bombs}")
        #     self.bombs = self.initial_bombs
        #     # 生命减少时，也重置炸弹冷却时间
        #     self.last_bomb_time = 0
        # self.last_lives = lives

        # 计算奖励
        reward = 0
        done = False

        # 生存奖励
        reward += 1

        # 分数奖励
        score_diff = score - self.last_score
        reward += score_diff * 0.001
        self.last_score = score

        # 炸弹相关惩罚 (注释掉)
        # reward += bomb_penalty

        # 死亡惩罚
        if lives == 0 and self.current_step > 100:
            print(f"检测到角色死亡：生命值={lives}, 步数={self.current_step}")
            reward = -1000
            done = True

        # 释放方向键
        for k in self.action_map[action]:
            keyboard.release(k)

        # 计算剩余冷却时间（用于info）(注释掉)
        # bomb_cooldown_remaining = max(0, self.bomb_cooldown - (current_time - self.last_bomb_time))
        
        # 可以打印调试
        #print(f"score:{score},lives:{lives},bombs:{self.bombs},bomb_cd:{bomb_cooldown_remaining:.1f}s")
        print(f"score:{score},lives:{lives}")
        return self._get_state(), reward, done, {
            'score': score, 
            'lives': lives
            # 'bombs': self.bombs,  # 注释掉炸弹相关信息
            # 'bomb_cooldown_remaining': bomb_cooldown_remaining
        }

    def _get_state(self):
        # 将 4 帧堆叠成 shape=(4,96,128)
        return np.stack(self.frames, axis=0)

    def close(self):
        # 关闭截屏
        if self.cap:
            self.cap.close()
        # 释放键盘
        keyboard.release('z')
        for action_keys in self.action_map.values():
            for k in action_keys:
                keyboard.release(k)
        print("环境已关闭。")