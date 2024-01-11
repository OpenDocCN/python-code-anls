# `yolov5-DNF\directkeys.py`

```
# 设置编码格式为utf-8
# 导入win32api模块
import win32api
# 导入win32con模块
import win32con

# 定义按键映射字典
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
# 导入time模块
import time

# 定义SendInput为ctypes.windll.user32.SendInput
SendInput = ctypes.windll.user32.SendInput

# 定义W、A、S、D等按键的键码
W = 0x11
A = 0x1E
S = 0x1F
D = 0x20
M = 0x32
J = 0x24
K = 0x25
LSHIFT = 0x2A
R = 0x13  # 用R代替识破
V = 0x2F
Q = 0x10
I = 0x17
O = 0x18
P = 0x19
C = 0x2E
F = 0x21
up = 0xC8
down = 0xD0
left = 0xCB
right = 0xCD

# 定义方向键的键码字典
direct_dic = {"UP": 0xC8, "DOWN": 0xD0, "LEFT": 0xCB, "RIGHT": 0xCD}

# 定义ESC键的键码
esc = 0x01

# 定义C结构的重新定义
PUL = ctypes.POINTER(ctypes.c_ulong)

# 定义KeyBdInput结构
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

# 定义HardwareInput结构
class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)
# 定义鼠标输入的数据结构
class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),  # 鼠标水平移动距离
                ("dy", ctypes.c_long),  # 鼠标垂直移动距离
                ("mouseData", ctypes.c_ulong),  # 鼠标滚轮滚动距离
                ("dwFlags", ctypes.c_ulong),  # 鼠标操作标志
                ("time", ctypes.c_ulong),  # 时间戳
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

# 模拟按下键盘按键
def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

# 模拟释放键盘按键
def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

# 定义防御操作
def defense():
    PressKey(M)  # 模拟按下 M 键
    time.sleep(0.05)  # 等待0.05秒
    ReleaseKey(M)  # 模拟释放 M 键

# 定义攻击操作
def attack():
    PressKey(J)  # 模拟按下 J 键
    time.sleep(0.05)  # 等待0.05秒
    ReleaseKey(J)  # 模拟释放 J 键

# 定义向前移动操作
def go_forward():
    PressKey(W)  # 模拟按下 W 键
    time.sleep(0.4)  # 等待0.4秒
    ReleaseKey(W)  # 模拟释放 W 键

# 定义向后移动操作
def go_back():
    PressKey(S)  # 模拟按下 S 键
    time.sleep(0.4)  # 等待0.4秒
    ReleaseKey(S)  # 模拟释放 S 键

# 定义向左移动操作
def go_left():
    PressKey(A)  # 模拟按下 A 键
    time.sleep(0.4)  # 等待0.4秒
    ReleaseKey(A)  # 模拟释放 A 键

# 定义向右移动操作
def go_right():
    PressKey(D)  # 模拟按下 D 键
    time.sleep(0.4)  # 等待0.4秒
    ReleaseKey(D)  # 模拟释放 D 键

# 定义跳跃操作
def jump():
    PressKey(K)  # 模拟按下 K 键
    time.sleep(0.1)  # 等待0.1秒
    ReleaseKey(K)  # 模拟释放 K 键

# 定义闪避操作
def dodge():
    PressKey(R)  # 模拟按下 R 键
    time.sleep(0.1)  # 等待0.1秒
    ReleaseKey(R)  # 模拟释放 R 键

# 定义锁定视角操作
def lock_vision():
    PressKey(V)  # 模拟按下 V 键
    time.sleep(0.3)  # 等待0.3秒
    ReleaseKey(V)  # 模拟释放 V 键
    time.sleep(0.1)  # 等待0.1秒

# 定义向前移动并施放技能操作
def go_forward_QL(t):
    PressKey(W)  # 模拟按下 W 键
    time.sleep(t)  # 等待指定时间
    ReleaseKey(W)  # 模拟释放 W 键

# 定义向左转向操作
def turn_left(t):
    PressKey(left)  # 模拟按下左转向键
    time.sleep(t)  # 等待指定时间
    # 释放左键
    ReleaseKey(left)
# 定义向上转向函数，按下向上键，等待一段时间，释放向上键
def turn_up(t):
    PressKey(up)
    time.sleep(t)
    ReleaseKey(up)

# 定义向右转向函数，按下向右键，等待一段时间，释放向右键
def turn_right(t):
    PressKey(right)
    time.sleep(t)
    ReleaseKey(right)

# 定义前进函数，按下前进键，等待0.5秒，释放前进键
def F_go():
    PressKey(F)
    time.sleep(0.5)
    ReleaseKey(F)

# 定义前跳函数，按下前进键，等待一段时间，按下跳跃键，释放前进键和跳跃键
def forward_jump(t):
    PressKey(W)
    time.sleep(t)
    PressKey(K)
    ReleaseKey(W)
    ReleaseKey(K)

# 定义按下ESC键函数，按下ESC键，等待0.3秒，释放ESC键
def press_esc():
    PressKey(esc)
    time.sleep(0.3)
    ReleaseKey(esc)

# 定义按下M键函数，按下M键，等待0.5秒，释放M键
def dead():
    PressKey(M)
    time.sleep(0.5)
    ReleaseKey(M)

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 记录当前时间
    time1 = time.time()
    # 初始化变量k和s
    k = "LEFT"
    s = "D"
    # 循环执行以下操作
    while True:
        # 如果当前时间与记录的时间差超过10秒，则跳出循环
        if abs(time.time() - time1) > 10:
            break
        else:
            # 按下指定方向键和动作键，等待一段时间，释放指定方向键和动作键
            PressKey(direct_dic[k])
            key_down(s)
            time.sleep(0.02)
            key_up(s)
            ReleaseKey(direct_dic[k])
            time.sleep(0.02)
```