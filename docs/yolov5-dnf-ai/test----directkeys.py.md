# `yolov5-DNF\test\directkeys.py`

```py
# 设置文件编码格式为utf-8
# 导入win32api模块
# 导入win32con模块
import win32api
import win32con

# 定义键盘按键与键码的映射关系
key_map = {
    "0": 49, "1": 50, "2": 51, "3": 52, "4": 53, "5": 54, "6": 55, "7": 56, "8": 57, "9": 58,
    "A": 65, "B": 66, "C": 67, "D": 68, "E": 69, "F": 70, "G": 71, "H": 72, "I": 73, "J": 74,
    "K": 75, "L": 76, "M": 77, "N": 78, "O": 79, "P": 80, "Q": 81, "R": 82, "S": 83, "T": 84,
    "U": 85, "V": 86, "W": 87, "X": 88, "Y": 89, "Z": 90, "LEFT": 37, "UP": 38, "RIGHT": 39,
    "DOWN": 40, "CTRL": 17, "ALT": 18, "F2": 113, "ESC": 27, "SPACE": 32, "NUM0": 96
}

# 定义按下按键的函数
def key_down(key):
    """
    函数功能：按下按键
    参    数：key:按键值
    """
    key = key.upper()
    vk_code = key_map[key]
    win32api.keybd_event(vk_code, win32api.MapVirtualKey(vk_code, 0), 0, 0)

# 定义抬起按键的函数
def key_up(key):
    """
    函数功能：抬起按键
    参    数：key:按键值
    """
    key = key.upper()
    vk_code = key_map[key]
    win32api.keybd_event(vk_code, win32api.MapVirtualKey(vk_code, 0), win32con.KEYEVENTF_KEYUP, 0)

# 定义点击按键（按下并抬起）的函数
def key_press(key):
    """
    函数功能：点击按键（按下并抬起）
    参    数：key:按键值
    """
    key_down(key)
    time.sleep(0.02)
    key_up(key)
    time.sleep(0.01)

# 导入ctypes模块
import ctypes
import time

# 定义键盘按键对应的键码
# ...

# 定义方向键的键码映射
direct_dic = {"UP": 0xC8, "DOWN": 0xD0, "LEFT": 0xCB, "RIGHT": 0xCD}

# 定义ESC键的键码
esc = 0x01

# 定义C结构的重新定义
PUL = ctypes.POINTER(ctypes.c_ulong)

# 定义键盘输入结构
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

# 定义硬件输入结构
class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]
# 定义鼠标输入的数据结构
class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),  # 鼠标水平移动距离
                ("dy", ctypes.c_long),  # 鼠标垂直移动距离
                ("mouseData", ctypes.c_ulong),  # 鼠标滚轮滚动距离
                ("dwFlags", ctypes.c_ulong),  # 鼠标操作标志
                ("time", ctypes.c_ulong),  # 操作发生的时间
                ("dwExtraInfo", PUL)]  # 额外信息

# 定义输入联合体
class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),  # 键盘输入
                ("mi", MouseInput),  # 鼠标输入
                ("hi", HardwareInput)]  # 硬件输入

# 定义输入数据结构
class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),  # 输入类型
                ("ii", Input_I)]  # 输入联合体

# 模拟按下键盘按键的函数
def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

# 模拟释放键盘按键的函数
def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

# 防御操作
def defense():
    PressKey(M)  # 模拟按下 M 键
    time.sleep(0.05)  # 等待0.05秒
    ReleaseKey(M)  # 模拟释放 M 键

# 攻击操作
def attack():
    PressKey(J)  # 模拟按下 J 键
    time.sleep(0.05)  # 等待0.05秒
    ReleaseKey(J)  # 模拟释放 J 键

# 向前移动
def go_forward():
    PressKey(W)  # 模拟按下 W 键
    time.sleep(0.4)  # 等待0.4秒
    ReleaseKey(W)  # 模拟释放 W 键

# 后退
def go_back():
    PressKey(S)  # 模拟按下 S 键
    time.sleep(0.4)  # 等待0.4秒
    ReleaseKey(S)  # 模拟释放 S 键

# 向左移动
def go_left():
    PressKey(A)  # 模拟按下 A 键
    time.sleep(0.4)  # 等待0.4秒
    ReleaseKey(A)  # 模拟释放 A 键

# 向右移动
def go_right():
    PressKey(D)  # 模拟按下 D 键
    time.sleep(0.4)  # 等待0.4秒
    ReleaseKey(D)  # 模拟释放 D 键

# 跳跃
def jump():
    PressKey(K)  # 模拟按下 K 键
    time.sleep(0.1)  # 等待0.1秒
    ReleaseKey(K)  # 模拟释放 K 键

# 闪避
def dodge():
    PressKey(R)  # 模拟按下 R 键
    time.sleep(0.1)  # 等待0.1秒
    ReleaseKey(R)  # 模拟释放 R 键

# 锁定视角
def lock_vision():
    PressKey(V)  # 模拟按下 V 键
    time.sleep(0.3)  # 等待0.3秒
    ReleaseKey(V)  # 模拟释放 V 键
    time.sleep(0.1)  # 再等待0.1秒

# 向前移动（自定义时间）
def go_forward_QL(t):
    PressKey(W)  # 模拟按下 W 键
    time.sleep(t)  # 等待指定时间
    ReleaseKey(W)  # 模拟释放 W 键

# 向左转向（自定义时间）
def turn_left(t):
    PressKey(left)  # 模拟按下左转向键
    time.sleep(t)  # 等待指定时间
    # 释放左键
    ReleaseKey(left)
def turn_up(t):
    # 按下向上键
    PressKey(up)
    # 等待 t 秒
    time.sleep(t)
    # 释放向上键
    ReleaseKey(up)


def turn_right(t):
    # 按下向右键
    PressKey(right)
    # 等待 t 秒
    time.sleep(t)
    # 释放向右键
    ReleaseKey(right)


def F_go():
    # 按下 F 键
    PressKey(F)
    # 等待 0.5 秒
    time.sleep(0.5)
    # 释放 F 键
    ReleaseKey(F)


def forward_jump(t):
    # 按下 W 键
    PressKey(W)
    # 等待 t 秒
    time.sleep(t)
    # 按下 K 键
    PressKey(K)
    # 释放 W 键
    ReleaseKey(W)
    # 释放 K 键
    ReleaseKey(K)


def press_esc():
    # 按下 esc 键
    PressKey(esc)
    # 等待 0.3 秒
    time.sleep(0.3)
    # 释放 esc 键
    ReleaseKey(esc)


def dead():
    # 按下 M 键
    PressKey(M)
    # 等待 0.5 秒
    time.sleep(0.5)
    # 释放 M 键
    ReleaseKey(M)

if __name__ == "__main__":
    # 记录当前时间
    time1 = time.time()
    # 初始化 k 和 s
    k = "LEFT"
    s = "D"
    while True:
        # 如果当前时间与记录的时间差超过 10 秒，则跳出循环
        if abs(time.time() - time1) > 10:
            break
        else:
            # 按下 k 对应的键
            PressKey(direct_dic[k])
            # 按下 s 键
            key_down(s)
            # 等待 0.02 秒
            time.sleep(0.02)
            # 释放 s 键
            key_up(s)
            # 释放 k 对应的键
            ReleaseKey(direct_dic[k])
            # 等待 0.02 秒
            time.sleep(0.02)
```