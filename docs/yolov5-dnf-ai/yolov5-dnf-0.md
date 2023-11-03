# Yolov5DNF源码解析 0

# `datasets_utils.py`

这段代码的作用是将标注好的图像和标签从一个路径中转移到另一个路径中。

具体来说，它首先定义了两个变量：`root_path` 和 `yolo5_data_dir`。`root_path` 表示数据集的根目录，`yolo5_data_dir` 表示数据集 YOLOv5 的数据目录。

接着，它使用 `cv2.pyplot` 库中的 `cv2.wait直到` 函数(即 `cv2.wait直到` 函数的 `cv2.CAP_PROP_FRAME_NUMBER` 参数为 0，且 `cv2.CAP_PROP_FRAME_WIDTH` 参数为 0)来等待图像文件加载完成。使用 `os.path.join` 函数将每个图像文件的完整路径存储到 `imgs_list` 列表中。

接下来，它使用 `shutil.copy` 函数将每个图像文件和相应的标签文件从一个路径复制到另一个路径中。具体来说，它使用 `os.path.join` 函数将源路径和目标路径连接起来，并将文件名和路径参数传递给 `shutil.copy` 函数。

最后，它将 `imgs_list` 和 `json_list` 存储在一个名为 `data` 的字典中，并将该字典存储到 `yolo5_data_dir` 目录下。


```py
import cv2 as cv
import os
import shutil

# 将标注好的图像和标签转移到新路径下
root_path = "datasets/guiqi/patch1"
yolo5_data_dir = "datasets/guiqi/patch1_yolo5"

json_list = []
imgs_list = []
dir = os.listdir(root_path)
for d in dir:
    if d.endswith(".json"):
        imgs_list.append(d.strip().split(".")[0] + ".jpg")
        json_list.append(d)
```

这段代码的主要作用是上传一些图像数据和对应的标签到指定的文件夹中，然后从这些文件中随机选择一部分作为验证集。

具体来说，代码首先通过 `print` 函数输出两个列表 `imgs_list` 和 `json_list`，其中 `imgs_list` 包含了所有图像文件的名称，而 `json_list` 包含了所有文件的 JSON 数据。

接着，代码通过 `zip` 函数将 `imgs_list` 和 `json_list` 中的所有元素分别存储到一个名为 `img_name` 和 `json` 的元组中，这样我们就可以对每个元素进行操作了。

在下一行中，代码使用 `for` 循环遍历 `imgs_list` 和 `json_list` 中的元素，并将它们对应的照片文件名和 JSON 数据分别复制到 `yolo5_data_dir` 目录下的 `images` 文件夹和 `labels_json` 文件夹中。注意，这些文件夹可能不存在，如果它们不存在，代码会自动创建它们。

在代码的最后，代码还定义了一些变量，包括 `eval_ratio`、`dir` 和 `eval_nums`，用于确定从每个数据集中随机选择多少个样本作为验证集，以及计算评估指标。


```py
print(imgs_list)
print(json_list)

for img_name, json in zip(imgs_list, json_list):
    shutil.copy(os.path.join(root_path + "/" + img_name), os.path.join(yolo5_data_dir + '/imgs'))
    shutil.copy(os.path.join(root_path + "/" + json), os.path.join(yolo5_data_dir + '/labels_json'))

# # 选一部分数据作为验证集
# img_train_path = r"F:\Computer_vision\yolov5\YOLO5\DNF\train\images"
# img_valid_path = r"F:\Computer_vision\yolov5\YOLO5\DNF\valid\images"
# label_train_path = r"F:\Computer_vision\yolov5\YOLO5\DNF\train\labels"
# label_valid_path = r"F:\Computer_vision\yolov5\YOLO5\DNF\valid\labels"
# eval_ratio = 0.1
# dir = os.listdir(img_train_path)
# eval_nums = int(eval_ratio * len(dir))
```

这段代码的作用是实现数据增强。具体来说，它的主要思想是通过对训练数据和验证数据目录的随机移动，使得模型能够更好地学习到数据的分布和特征，从而提高模型的泛化能力和鲁棒性。

具体实现可以分为以下几个步骤：

1. 导入random模块，用于生成随机数。
2. 通过shutil库的move函数，将训练数据目录下的图片文件随机移动到验证数据目录下，并将对应的标签文件（.txt文本文件）一并移动。
3. 通过shutil库的copy函数，将验证数据目录下的图片文件复制到标签数据目录下，并对图片文件进行编号，以便训练模型时能够正确地识别图片和标签。
4. 定义了一个字典id2name，用于存储数据目录之间的映射关系，以便在训练模型时能够根据主题进行迁移。
5. 通过for循环，遍历目录树中的所有数据文件夹，实现了数据增强的功能。


```py
# import random
# random.shuffle(dir)
# for d in dir[:eval_nums]:
#     shutil.move(os.path.join(img_train_path + "\\" + d), os.path.join(img_valid_path + "\\" + d))
#     shutil.move(os.path.join(label_train_path + "\\" + d.strip().split(".")[0] + ".txt"),
#                 os.path.join(label_valid_path + "\\" + d.strip().split(".")[0] + ".txt"))

# undict生成
#
# name2id = {'hero': 0, 'small_map': 1, "monster": 2, 'money': 3, 'material': 4, 'door': 5, 'BOSS': 6, 'box': 7, 'options': 8}
# id2name = {}
# for key, val in name2id.items():
#     id2name[val] = key
# print(id2name)
```

# `direction_move.py`

It seems like there is a typo in the code. I assume you meant to write "right" instead of "up" in the if statement, so the code should be:
scss
if direct == "RIGHT":
   if action_cache != None:
       if action_cache != "RIGHT":
           if action_cache not in ["RIGHT", "LEFT", "UP", "DOWN"]:
               ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
               ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
           else:
               ReleaseKey(direct_dic[action_cache])
               if not material:
                   PressKey(direct_dic["RIGHT"])
                   time.sleep(press_delay)
                   ReleaseKey(direct_dic["RIGHT"])
                   time.sleep(release_delay)
                   PressKey(direct_dic["RIGHT"])
                   time.sleep(press_delay)
               if material:
                   PressKey(direct_dic["RIGHT"])
               PressKey(direct_dic["DOWN"])
               # time.sleep(press_delay)
               action_cache = "RIGHT_DOWN"
               print("右下移动")
           else:
               print("右下移动")
       else:
           print("右下移动")
   else:
       print("右下移动")

This code snippet checks if the user wants to press "RIGHT" or "LEFT" keyboard keys. If the user wants to press "RIGHT", the code will perform actions depending on the action that should be performed. If the user wants to press "LEFT", the code will perform an action and then check if the user has pressed "RIGHT" or "LEFT" again. If the user presses "RIGHT" and then "LEFT", the code will perform an action and then return the action that should be performed.


```py
import time
from directkeys import PressKey, ReleaseKey, key_down, key_up

direct_dic = {"UP": 0xC8, "DOWN": 0xD0, "LEFT": 0xCB, "RIGHT": 0xCD}

def move(direct, material=False, action_cache=None, press_delay=0.1, release_delay=0.1):
    if direct == "RIGHT":
        if action_cache != None:
            if action_cache != "RIGHT":
                if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                else:
                    ReleaseKey(direct_dic[action_cache])
                PressKey(direct_dic["RIGHT"])
                if not material:
                    time.sleep(press_delay)
                    ReleaseKey(direct_dic["RIGHT"])
                    time.sleep(release_delay)
                    PressKey(direct_dic["RIGHT"])
                action_cache = "RIGHT"
                print("向右移动")
            else:
                print("向右移动")
        else:
            PressKey(direct_dic["RIGHT"])
            if not material:
                time.sleep(press_delay)
                ReleaseKey(direct_dic["RIGHT"])
                time.sleep(release_delay)
                PressKey(direct_dic["RIGHT"])
            action_cache = "RIGHT"
            print("向右移动")
        return action_cache

    elif direct == "LEFT":
        if action_cache != None:
            if action_cache != "LEFT":
                if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                else:
                    ReleaseKey(direct_dic[action_cache])
                PressKey(direct_dic["LEFT"])
                if not material:
                    time.sleep(press_delay)
                    ReleaseKey(direct_dic["LEFT"])
                    time.sleep(release_delay)
                    PressKey(direct_dic["LEFT"])
                action_cache = "LEFT"
                print("向左移动")
            else:
                print("向左移动")
        else:
            PressKey(direct_dic["LEFT"])
            if not material:
                time.sleep(press_delay)
                ReleaseKey(direct_dic["LEFT"])
                time.sleep(release_delay)
                PressKey(direct_dic["LEFT"])
            action_cache = "LEFT"
            print("向左移动")
        return action_cache

    elif direct == "UP":
        if action_cache != None:
            if action_cache != "UP":
                if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                else:
                    ReleaseKey(direct_dic[action_cache])
                PressKey(direct_dic["UP"])
                # time.sleep(press_delay)
                # ReleaseKey(direct_dic["UP"])
                # time.sleep(release_delay)
                # PressKey(direct_dic["UP"])
                action_cache = "UP"
                print("向上移动")
            else:
                print("向上移动")
        else:
            PressKey(direct_dic["UP"])
            # time.sleep(press_delay)
            # ReleaseKey(direct_dic["UP"])
            # time.sleep(release_delay)
            # PressKey(direct_dic["UP"])
            action_cache = "UP"
            print("向上移动")
        return action_cache

    elif direct == "DOWN":
        if action_cache != None:
            if action_cache != "DOWN":
                if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                else:
                    ReleaseKey(direct_dic[action_cache])
                PressKey(direct_dic["DOWN"])
                # time.sleep(press_delay)
                # ReleaseKey(direct_dic["DOWN"])
                # time.sleep(release_delay)
                # PressKey(direct_dic["DOWN"])
                action_cache = "DOWN"
                print("向下移动")
            else:
                print("向下移动")
        else:
            PressKey(direct_dic["DOWN"])
            # time.sleep(press_delay)
            # ReleaseKey(direct_dic["DOWN"])
            # time.sleep(release_delay)
            # PressKey(direct_dic["DOWN"])
            action_cache = "DOWN"
            print("向下移动")
        return action_cache

    elif direct == "RIGHT_UP":
        if action_cache != None:
            if action_cache != "RIGHT_UP":
                if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                else:
                    ReleaseKey(direct_dic[action_cache])
                if not material:
                    PressKey(direct_dic["RIGHT"])
                    time.sleep(press_delay)
                    ReleaseKey(direct_dic["RIGHT"])
                    time.sleep(release_delay)
                    PressKey(direct_dic["RIGHT"])
                    time.sleep(press_delay)
                if material:
                    PressKey(direct_dic["RIGHT"])
                PressKey(direct_dic["UP"])
                # time.sleep(release_delay)
                action_cache = "RIGHT_UP"
                print("右上移动")
            else:
                print("右上移动")
        else:
            if not material:
                PressKey(direct_dic["RIGHT"])
                time.sleep(press_delay)
                ReleaseKey(direct_dic["RIGHT"])
                time.sleep(release_delay)
                PressKey(direct_dic["RIGHT"])
                time.sleep(press_delay)
            if material:
                PressKey(direct_dic["RIGHT"])
            PressKey(direct_dic["UP"])
            # time.sleep(press_delay)
            action_cache = "RIGHT_UP"
            print("右上移动")
        return action_cache

    elif direct == "RIGHT_DOWN":
        if action_cache != None:
            if action_cache != "RIGHT_DOWN":
                if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                else:
                    ReleaseKey(direct_dic[action_cache])
                if not material:
                    PressKey(direct_dic["RIGHT"])
                    time.sleep(press_delay)
                    ReleaseKey(direct_dic["RIGHT"])
                    time.sleep(release_delay)
                    PressKey(direct_dic["RIGHT"])
                    time.sleep(press_delay)
                if material:
                    PressKey(direct_dic["RIGHT"])
                PressKey(direct_dic["DOWN"])
                # time.sleep(press_delay)
                action_cache = "RIGHT_DOWN"
                print("右上移动")
            else:
                print("右上移动")
        else:
            if not material:
                PressKey(direct_dic["RIGHT"])
                time.sleep(press_delay)
                ReleaseKey(direct_dic["RIGHT"])
                time.sleep(release_delay)
                PressKey(direct_dic["RIGHT"])
                time.sleep(press_delay)
            if material:
                PressKey(direct_dic["RIGHT"])
            PressKey(direct_dic["DOWN"])
            # time.sleep(press_delay)
            action_cache = "RIGHT_DOWN"
            print("右上移动")
        return action_cache

    elif direct == "LEFT_UP":
        if action_cache != None:
            if action_cache != "LEFT_UP":
                if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                else:
                    ReleaseKey(direct_dic[action_cache])
                if not material:
                    PressKey(direct_dic["LEFT"])
                    time.sleep(press_delay)
                    ReleaseKey(direct_dic["LEFT"])
                    time.sleep(release_delay)
                    PressKey(direct_dic["LEFT"])
                    time.sleep(press_delay)
                if material:
                    PressKey(direct_dic["LEFT"])
                PressKey(direct_dic["UP"])
                # time.sleep(press_delay)
                action_cache = "LEFT_UP"
                print("左上移动")
            else:
                print("左上移动")
        else:
            if not material:
                PressKey(direct_dic["LEFT"])
                time.sleep(press_delay)
                ReleaseKey(direct_dic["LEFT"])
                time.sleep(release_delay)
                PressKey(direct_dic["LEFT"])
                time.sleep(press_delay)
            if material:
                PressKey(direct_dic["LEFT"])
            PressKey(direct_dic["UP"])
            # time.sleep(press_delay)
            action_cache = "LEFT_UP"
            print("左上移动")
        return action_cache

    elif direct == "LEFT_DOWN":
        if action_cache != None:
            if action_cache != "LEFT_DOWN":
                if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                else:
                    ReleaseKey(direct_dic[action_cache])
                if not material:
                    PressKey(direct_dic["LEFT"])
                    time.sleep(press_delay)
                    ReleaseKey(direct_dic["LEFT"])
                    time.sleep(release_delay)
                    PressKey(direct_dic["LEFT"])
                    time.sleep(press_delay)
                if material:
                    PressKey(direct_dic["LEFT"])
                PressKey(direct_dic["DOWN"])
                # time.sleep(press_delay)
                action_cache = "LEFT_DOWN"
                print("左下移动")
            else:
                print("左下移动")
        else:
            if not material:
                PressKey(direct_dic["LEFT"])
                time.sleep(press_delay)
                ReleaseKey(direct_dic["LEFT"])
                time.sleep(release_delay)
                PressKey(direct_dic["LEFT"])
                time.sleep(press_delay)
            if material:
                PressKey(direct_dic["LEFT"])
            PressKey(direct_dic["DOWN"])
            # time.sleep(press_delay)
            action_cache = "LEFT_DOWN"
            print("左下移动")
        return action_cache


```

这段代码是一个if语句，判断当前运行脚本是否为__main__。如果是，则执行以下代码：

1. 创建一个名为action_cache的变量，类型为None。
2. 创建一个名为t1的变量，类型为time.time()，并将其赋值为0。
3. 进入一个无限循环。
4. 在循环中，使用time.time()获取当前时间，并将其与0的差值除以2，如果是偶数，则执行以下操作：
   a. 将action_cache变量设置为"LEFT_DOWN"操作，并将material参数设置为False，将action_cache参数设置为action_cache，将press_delay参数设置为0.1，将release_delay参数设置为0.1。
   b. 将user_data变量设置为"example user data"。
5. 如果当前时间不是0，则执行以下操作：
   a. 将action_cache变量设置为"RIGHT_UP"操作，并将material参数设置为True，将action_cache参数设置为action_cache，将press_delay参数设置为0.1，将release_delay参数设置为0.1。
   b. 将user_data变量设置为"example user data"。
6. 循环结束后，action_cache变量仍然指向None。


```py
if __name__ == "__main__":
    action_cache = None
    t1 = time.time()
    # while True:
        # if  int(time.time() - t1) % 2 == 0:
        #     action_cache = move("LEFT_DOWN", material=False, action_cache=action_cache, press_delay=0.1, release_delay=0.1)
        # else:
    action_cache = move("RIGHT_UP", material=True, action_cache=action_cache, press_delay=0.1, release_delay=0.1)
```

# `directkeys.py`

这段代码的作用是记录了每个按键对应的虚拟键盘码(VK_CODE)，并实现了键盘输入的功能。

具体来说，代码首先定义了一个名为 key_map 的字典，用于存储每个按键对应的虚拟键盘码。接着，定义了一个名为 key_down 的函数，用于按下并释放按键时，将其对应的键值转换为虚拟键盘码并按下。

在 key_down 函数中，首先将按键的键值转换为大写，然后使用 win32api.MapVirtualKey 函数获取其对应的虚拟键盘码。接着，使用 win32api.keybd_event 函数将虚拟键盘码发送到对应虚拟键盘的 echo 队列中。

这个程序可以记录并模拟键盘输入，对于按键输入的处理，主要集中在游戏的实现中，例如在游戏中的角色移动、攻击等操作，可以通过调用 key_down 函数来模拟键盘输入。


```py
# coding=utf-8
import win32con
import win32api
import time

key_map = {
    "0": 49, "1": 50, "2": 51, "3": 52, "4": 53, "5": 54, "6": 55, "7": 56, "8": 57, "9": 58,
    "A": 65, "B": 66, "C": 67, "D": 68, "E": 69, "F": 70, "G": 71, "H": 72, "I": 73, "J": 74,
    "K": 75, "L": 76, "M": 77, "N": 78, "O": 79, "P": 80, "Q": 81, "R": 82, "S": 83, "T": 84,
    "U": 85, "V": 86, "W": 87, "X": 88, "Y": 89, "Z": 90, "LEFT": 37, "UP": 38, "RIGHT": 39,
    "DOWN": 40, "CTRL": 17, "ALT": 18, "F2": 113, "ESC": 27, "SPACE": 32, "NUM0": 96
}


def key_down(key):
    """
    函数功能：按下按键
    参    数：key:按键值
    """
    key = key.upper()
    vk_code = key_map[key]
    win32api.keybd_event(vk_code, win32api.MapVirtualKey(vk_code, 0), 0, 0)


```

这段代码定义了两个函数：`key_up()` 和 `key_press()`。这两个函数的功能是不同的，`key_up()` 是抬起按键，而 `key_press()` 是按下并抬起按键。它们的参数 `key` 是一个字符串类型的变量。

函数 `key_up()` 的作用是抬起指定的按键。函数内部首先将 `key` 变量中的字符串转换为大写，然后使用 `key_map` 字典存储键值对。接着，使用 `win32api.keybd_event()` 函数发送一个虚拟键 `vk_code`，并在参数 `win32api.MapVirtualKey()` 中指定键的虚拟键（即 `key_map` 中键的值），使得虚拟键对应的键被按下。最后，使用 `win32con.KEYEVENTF_KEYUP` 格式化数据 `win32api.keybd_event()` 的 `按键类型`，指定是按下并抬起按键。

函数 `key_press()` 的作用是按下并抬起指定的按键。函数内部首先调用 `key_up()` 函数抬起按键，然后等待 0.02 秒。接着，它再次调用 `key_up()` 函数抬起按键，并等待 0.01 秒。这样，按下和抬起按键的时间是相等的，实现了快速键抬起的操作。

总之，这两个函数根据用户按下和抬起按键的需求，通过调用 `win32api.keybd_event()` 函数实现了在不同场景下对按键的控制。


```py
def key_up(key):
    """
    函数功能：抬起按键
    参    数：key:按键值
    """
    key = key.upper()
    vk_code = key_map[key]
    win32api.keybd_event(vk_code, win32api.MapVirtualKey(vk_code, 0), win32con.KEYEVENTF_KEYUP, 0)


def key_press(key):
    """
    函数功能：点击按键（按下并抬起）
    参    数：key:按键值
    """
    key_down(key)
    time.sleep(0.02)
    key_up(key)
    time.sleep(0.01)

```

这段代码使用了Python的ctypes库来实现Windows系统的函数调用，函数名为“SendInput”。函数接收3个整数参数，分别为“W”（窗口键，对应CTypes中的“WIN_KEY”常量，值为0x11）；“A”（应用程序区键，对应CTypes中的“CONTROL_SUBSTITUTE”常量，值为0x1E）；“S”（启动窗口键，对应CTypes中的“WIN_KEY”常量，值为0x1F）。

函数的作用是发送一个数组给定的窗口键，该数组包含4个整数，分别对应启动窗口键、应用程序区键、窗口扩展键和LSHIFT键。


```py
####################################
import ctypes
import time

SendInput = ctypes.windll.user32.SendInput

W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

M = 0x32
J = 0x24
K = 0x25
LSHIFT = 0x2A
```

以上代码的作用是执行以下操作：

1. 将十六进制数 "0x13" 转换为整数并将其赋值给变量 R。
2. 将十六进制数 "0x2F" 转换为整数并将其赋值给变量 V。
3. 将十六进制数 "0x10" 转换为整数并将其赋值给变量 Q。
4. 将十六进制数 "0x17" 转换为整数并将其赋值给变量 I。
5. 将十六进制数 "0x18" 转换为整数并将其赋值给变量 O。
6. 将十六进制数 "0x19" 转换为整数并将其赋值给变量 P。
7. 将十六进制数 "0x2E" 转换为整数并将其赋值给变量 C。
8. 将十六进制数 "0x21" 转换为整数并将其赋值给变量 F。
9. 将变量 up 的值设置为 0xC8。
10. 将变量 down 的值设置为 0xD0。
11. 将变量 left 的值设置为 0xCB。
12. 将变量 right 的值设置为 0xCD。


```py
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

```

这段代码定义了一个名为 "direct_dic" 的字典，包含了四个键值对，键分别是 "UP"、"DOWN"、"LEFT" 和 "RIGHT"，它们分别对应的是一个名为 "keyboard_shortcut" 的结构体类型的变量，该结构体定义了在一个笛卡尔坐标系中按下对应键时，与按键对应的扫描码。

接着，定义了一个名为 "esc" 的整数类型的变量，并将其赋值为 0x01。

接下来的代码定义了一个名为 "KeyBdInput" 的ctypes.Structure类型的类，该类包含了一个笛卡尔坐标系的输入结构体，该结构体定义了输入数据的时间戳、扫描码、位 flag 和其他相关信息。

该 "KeyBdInput" 类的定义了一系列的 "time" 和 "dwFlags" 字段，这些字段似乎没有在代码中使用，但它们可以用来记录用户输入的上下文信息，比如时间戳可以用来判断用户输入是否在有效的输入时间范围内。

另外，还定义了一个名为 "PUL" 的ctypes.POINTER类型的变量，该变量指针了一个名为 "wVk" 和 "wScan" 的ctypes.c_ulong类型的变量，但这两变量在代码中没有定义任何成员变量。


```py
direct_dic = {"UP": 0xC8, "DOWN": 0xD0, "LEFT": 0xCB, "RIGHT": 0xCD}

esc = 0x01

# C struct redefinitions
PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


```

这两段代码定义了两个结构体类：HardwareInput 和 MouseInput。它们都继承自 ctypes.Structure 类。

在这两段代码中，定义了一个共用体（ctypes.Structure）类型的字段，它由两个整型字段和一个无类型的字段组成。这个共用体字段被称为_fields_，表示这个结构体中所有字段的名称。

对于 HardwareInput，它定义了一个名为 uMsg 的字段，它的类型是 ctypes.c_ulong，它的长度为 4。这个字段表示一个消息，可能是硬件制造商发送给程序的，告诉它硬件已经准备好接收数据。

对于 MouseInput，它定义了一个名为 dx 和 dy 的字段，它们的类型是 ctypes.c_long，它们的长度都为 4。这个字段表示鼠标移动的距离，这个距离可能是为了在运行时记录鼠标移动的历史而定义的。

此外，MouseInput 还有一个名为 mouseData 的字段，它的类型是 ctypes.c_ulong，它的长度为 4。这个字段表示鼠标数据，可能是用于在运行时记录鼠标按下或释放的事件而定义的。

最后，MouseInput 和 HardwareInput 都有一个名为 time 的字段，它的类型是 ctypes.c_ulong，它的长度为 4。这个字段表示事件发生的时间，可能是用于在运行时记录事件发生的时间而定义的。

此外，MouseInput 和 HardwareInput 都有一个名为 dwFlags 的字段，它的类型是 ctypes.c_ulong，它的长度为 4。这个字段表示一些与鼠标输入相关的设置。

最后，MouseInput 和 HardwareInput 都有一个名为 dwExtraInfo 的字段，它的类型是 PUL，它的长度为 4。这个字段表示一些与鼠标输入相关的设置。


```py
class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


```

这段代码定义了两个类，`Input_I` 和 `Input`。这两个类都继承自 `ctypes.Structure` 类，即 C++ 的 `struct` 类型。

`Input_I` 类包含了一个键盘输入和一个鼠标输入，它们都被继承自 `KeyBdInput` 和 `MouseInput` 类，这些类可能来源于一个标准 C++ 库，比如 Boost 库。

`Input` 类包含了一个输入字符串类型和一个输入 `Input_I` 类型的结构体成员。这个 `Input` 类的成员函数 `PressKey` 接受一个六位数字，表示要按下哪个键，然后将其作为参数传递给 `ctypes.windll.user32.SendInput` 函数，这个函数将发送一个无限字符串到用户显卡设备，指定字符串中的每一个字符代表一个按键，比如 `ctypes.WIN32.F6` 键对应的是 `"F6"`。


```py
class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]


# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


```

这段代码是一个名为 "ReleaseKey" 的函数，其作用是释放一个指定长度的十六进制密钥。该函数首先创建一个名为 "extra" 的 ctypes 变量，并将其初始化为 0。

接下来，函数使用 "Input_I" 函数从用户那里获取一个 16 进制字符串，并将其存储在名为 "ii_" 的变量中。然后，函数将获取到的字符串作为第一个参数传递给 "KeyBdInput" 函数，该函数将第一个参数的十六进制字符串转换为字节，并将第二个参数设置为 0x0008 和 0x0002，表示输入模式为 8 位字符和两位符号。函数还使用 "Input" 函数从用户那里获取一个 16 进制字符串，并将其存储在名为 "x" 的变量中。

最后，函数使用 "ctypes.windll.user32.SendInput" 函数从用户那里获取一个字节数组，并将其传递给 "x"，从而实现了释放十六进制密钥的过程。


```py
def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def defense():
    PressKey(M)
    time.sleep(0.05)
    ReleaseKey(M)
    # time.sleep(0.1)


```

这段代码是一个攻击行为和植物大战背景的模拟。在这个游戏中，玩家需要通过按键来控制角色移动和攻击。

`attack()`函数的作用是模拟攻击。具体来说，它会导致角色在原地停留0.05秒，然后释放一个键盘按键(J)，并暂停0.1秒。然后，它再次在原地停留0.05秒，并释放另一个键盘按键(J)。这样可以模拟玩家在攻击时需要考虑的重要停顿步骤。

`go_forward()`函数模拟玩家的推进行为。它会导致角色在原地停留0.4秒，然后释放一个键盘按键(W)，并暂停0.4秒。接下来，它释放另一个键盘按键(W)，以便角色继续前进。

`go_back()`函数模拟玩家返回的行为。它会导致角色在原地停留0.4秒，然后释放一个键盘按键(S)，并暂停0.4秒。接下来，它释放另一个键盘按键(S)，以便角色返回。


```py
def attack():
    PressKey(J)
    time.sleep(0.05)
    ReleaseKey(J)
    # time.sleep(0.1)


def go_forward():
    PressKey(W)
    time.sleep(0.4)
    ReleaseKey(W)


def go_back():
    PressKey(S)
    time.sleep(0.4)
    ReleaseKey(S)


```

这段代码定义了三个函数，go_left、go_right和jump，它们分别对A、D和K键进行了一次按下和一次释放。这三个函数都使用了time.sleep()函数来暂停程序的执行0.4秒钟，然后对相应键盘键进行操作。 

go_left函数的作用是对A键进行一次按下，然后暂停0.4秒钟，最后释放A键。 

go_right函数的作用是对D键进行一次按下，然后暂停0.4秒钟，最后释放D键。 

jump函数的作用是对K键进行一次按下，然后暂停0.1秒钟，接着释放K键，并再次暂停0.1秒钟。 

总的来说，这三个函数只是对 keyboard 上的三个键进行了一些基本的操作，然后又不管什么情况地暂停了0.4秒钟，紧接着又不管什么情况地释放了这些键。


```py
def go_left():
    PressKey(A)
    time.sleep(0.4)
    ReleaseKey(A)


def go_right():
    PressKey(D)
    time.sleep(0.4)
    ReleaseKey(D)


def jump():
    PressKey(K)
    time.sleep(0.1)
    ReleaseKey(K)
    # time.sleep(0.1)


```

这段代码定义了三个函数：dodge、lock_vision 和 go_forward_QL。

dodge函数是一个简单的闪避操作，它在 PressKey(R) 时模拟按键输入，然后time.sleep(0.1) 会让程序暂停0.1秒，然后 ReleaseKey(R) 会让程序继续暂停0.1秒，这样就模拟了一个闪避的动作。

lock_vision函数也是一个简单的锁视操作，它在 PressKey(V) 时模拟按键输入，然后time.sleep(0.3) 会让程序暂停0.3秒，然后 ReleaseKey(V) 会让程序继续暂停0.1秒，这样就模拟了一个锁视的动作。

go_forward_QL函数是一个可以控制机器人向前移动的函数。在 PressKey(W) 时模拟按键输入，然后根据传入的时间参数t，程序会暂停t秒，接着 ReleaseKey(W) 会让程序继续暂停0.1秒，从而让机器人移动t米。

总的来说，这段代码定义了三个函数，用于模拟按键输入的动作，以实现闪避、锁视和机器人向前移动的功能。


```py
def dodge():  # 闪避
    PressKey(R)
    time.sleep(0.1)
    ReleaseKey(R)
    # time.sleep(0.1)


def lock_vision():
    PressKey(V)
    time.sleep(0.3)
    ReleaseKey(V)
    time.sleep(0.1)


def go_forward_QL(t):
    PressKey(W)
    time.sleep(t)
    ReleaseKey(W)


```

这段代码定义了三个函数：turn_left、turn_up和turn_right。

turn_left函数的作用是在鼠标向左移动时，按下键盘上的“左”键，然后等待一段时间（由参数t指定）后释放键盘上的“左”键。

turn_up函数的作用是在鼠标向上移动时，按下键盘上的“上”键，然后等待一段时间（由参数t指定）后释放键盘上的“上”键。

turn_right函数的作用是在鼠标向右移动时，按下键盘上的“右”键，然后等待一段时间（由参数t指定）后释放键盘上的“右”键。


```py
def turn_left(t):
    PressKey(left)
    time.sleep(t)
    ReleaseKey(left)


def turn_up(t):
    PressKey(up)
    time.sleep(t)
    ReleaseKey(up)


def turn_right(t):
    PressKey(right)
    time.sleep(t)
    ReleaseKey(right)


```

这段代码是一个 Python 编写的函数，主要是用于模拟在虚拟键盘上按下 "F"、"W"、"↑" 和 "↓" 四个按键的功能。

具体来说，这段代码执行以下操作：

1. 定义了一个名为 "F_go" 的函数，该函数包含以下操作：按下 "F" 键，等待0.5秒，然后释放 "F" 键。

2. 定义了一个名为 "forward_jump" 的函数，该函数包含以下操作：按下 "W" 键，等待指定时间(假设为 t)，然后按下 "K" 键，再释放 "W" 和 "K" 键。

3. 定义了一个名为 "press_esc" 的函数，该函数包含以下操作：按下 "esc" 键，等待0.3秒，然后释放 "esc" 键。

4. 在主程序中，通过调用这些函数，可以模拟按下不同的按键。例如，可以调用 "F_go" 函数来模拟按下 "F" 键，可以调用 "press_esc" 函数来模拟按下 "ESC" 键，以此类推。


```py
def F_go():
    PressKey(F)
    time.sleep(0.5)
    ReleaseKey(F)


def forward_jump(t):
    PressKey(W)
    time.sleep(t)
    PressKey(K)
    ReleaseKey(W)
    ReleaseKey(K)


def press_esc():
    PressKey(esc)
    time.sleep(0.3)
    ReleaseKey(esc)


```

这段代码是一个 Python 函数，名为 `dead()`，它模拟了一个死机的过程。函数内部使用了 `time` 和 `keys` 模块，以及自定义的 `PressKey` 和 `ReleaseKey` 函数。

该函数的作用是在死机过程中，模拟用户按下键盘上的某些键，并在一定时间内释放这些键。在这个过程中，程序会不断地检查当前时间与开始时间之间的时间差，如果这个时间差大于 10，那么程序就会退出死机过程。

具体来说，该函数的实现过程如下：

1. 定义一个名为 `dead` 的函数，该函数内包含以下操作：
	* 调用 `PressKey(M)` 函数，其中 `M` 是用户当前按下的键，这个函数会模拟按下键的过程，并返回按下的键的 ASCII 码。
	* 调用 `time.sleep(0.5)` 函数，用于在死机过程中暂停 0.5 秒。
	* 调用 `ReleaseKey(M)` 函数，用于释放用户当前按下的键。
	* 调用自定义的 `PressKey(direct_dic[k])` 和 `ReleaseKey(direct_dic[k])` 函数，用于模拟用户按下和释放键的过程。其中，`direct_dic` 是一个字典，存储了用户当前按下的键的 ASCII 码，键的 ASCII 码对应一个字典中的一个键值对，这个键值对存储了键对应的 ASCII 码。
	* 调用 `time.sleep(0.02)` 函数，用于在死机过程中暂停 0.02 秒。
	* 调用 `key_down(s)` 和 `key_up(s)` 函数，用于模拟用户按下和释放键的过程。其中，`key_down(s)` 函数用于模拟用户按下键的过程，`key_up(s)` 函数用于模拟用户释放键的过程。其中，`s` 是用户当前按下的键的 ASCII 码。
	* 调用 `PressKey(direct_dic[k])` 函数，用于模拟用户按下键的过程。其中，`direct_dic[k]` 是字典中存储用户当前按下的键的 ASCII 码的键值对，键值对中的键是用户当前按下的键的 ASCII 码。
	* 调用 `ReleaseKey(M)` 函数，用于释放用户当前按下的键。其中，`M` 是用户当前按下的键的 ASCII 码。


```py
def dead():
    PressKey(M)
    time.sleep(0.5)
    ReleaseKey(M)

if __name__ == "__main__":
    time1 = time.time()
    k = "LEFT"
    s = "D"
    while True:
        if abs(time.time() - time1) > 10:
            break
        else:
            # if k not in ["LEFT", "RIGHT", "UP", "DOWN"]:
            #     key_press(k)
            # else:
            #     PressKey(direct_dic[k])
            #     time.sleep(0.1)
            #     ReleaseKey(direct_dic[k])
            #     time.sleep(0.2)
            PressKey(direct_dic[k])
            key_down(s)
            time.sleep(0.02)
            key_up(s)
            ReleaseKey(direct_dic[k])
            time.sleep(0.02)


```

# `getkeys.py`

It appears that the code is a JavaScript object with properties for different views or游戏 screens. The `left_T` through `right_T` properties appear to correspond to the different views in a 2D game, while the `left_Y` through `right_Y` properties appear to correspond to the different screens in the same game. The `up_A` through `down_A` properties appear to correspond to different game elements or states, while the `up_S` through `down_S` properties appear to correspond to different camera settings. The `left_D` through `right_D` properties appear to correspond to different parts of the game world, such as the player or enemies. The `up_F` through `down_F` properties appear to correspond to different special abilities or moves that the player can use. The `up_G` through `down_G` properties appear to correspond to different game endings or other significant events.



```py
import win32api as wapi
import time

dict = {"A": 0, "S": 1,"D": 2, "F": 3, "G": 4, "H": 5, "Q": 6, "W": 7, "E": 8, "R": 9, "T": 10, "Y": 11, "up": 12,
        "down": 13, "left": 14, "right":15, "ctrl": 16, "alt": 17, "Z":18, "X":19, "C": 20, "esc": 21, "f2": 22,
        "space": 23, "num0": 24, "left_up": 25, "left down": 26, "right_up": 27, "right_down": 28, "left_A": 29,
        "left_S": 30, "left_D": 31, "left_F": 32, "left_G": 33, "left_H": 34,"left_Q": 35, "left_W": 36, "left_E": 37,
        "left_R": 38, "left_T": 39, "left_Y": 40, "up_A": 41,"up_S": 42, "up_D": 43, "up_F": 44, "up_G": 45,
        "up_H": 46,"up_Q": 47, "up_W": 48, "up_E": 49, "up_R": 50, "up_T": 51, "up_Y": 52,"down_A": 53,
        "down_S": 54, "down_D": 55, "down_F": 56, "down_G": 57, "down_H": 58,"down_Q": 59, "down_W": 60, "down_E": 61,
        "down_R": 62, "down_T": 63, "down_Y": 64, "right_A": 65, "right_S": 66, "right_D": 67, "right_F": 68, "right_G": 69,
        "right_H": 70,"right_Q": 71, "right_W": 72, "right_E": 73, "right_R": 74, "right_T": 75, "right_Y": 76, "left_z": 77,
        "left_x": 78, "left_c": 79, "up_z": 80,"up_x": 81, "up_c": 82, "down_z": 83, "down_x": 84, "down_c": 85, "right_z": 86,
        "right_x": 87, "right_c": 88, "left_ctrl": 89, "up_ctrl": 90, "down_ctrl": 91, "right_ctrl": 92, "P": 100}

```

这段代码定义了一个名为 "keyList" 的列表变量，用于存储一个字符串中的所有字符。接下来，定义了一个名为 "key_check" 的函数，该函数获取了上面定义的列表 "keyList" 中所有按键的状态，并返回一个包含按键的列表。

具体来说，代码中使用了一个名为 "wapi" 的类，该类似乎用于在 Windows 操作系统上获取按键的状态信息。然后，代码使用一个 for 循环遍历字符串 "ASDFGHQWERTYZXCP"，将每个字符添加到 "keyList" 中。

接着，代码定义了一个名为 "key_check" 的函数。在这个函数中，使用一个 for 循环遍历上面获取的 "keyList" 中的所有字符，并使用 wapi.GetAsyncKeyState 方法获取每个按键的状态信息(如果按键在窗口中移动，则返回假)。然后，根据需要将状态信息添加到输出列表 "keys" 中。

最后，代码没有定义任何其他函数或类，也没有做其他事情，似乎只是一个简单的自定义函数。


```py
keyList = []
for char in "ASDFGHQWERTYZXCP":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    if wapi.GetAsyncKeyState(37):
        keys.append("left")
    if wapi.GetAsyncKeyState(39):
        keys.append("right")
    if wapi.GetAsyncKeyState(38):
        keys.append("up")
    if wapi.GetAsyncKeyState(40):
        keys.append("down")
    if wapi.GetAsyncKeyState(17):
        keys.append("ctrl")
    if wapi.GetAsyncKeyState(18):
        keys.append("alt")
    if wapi.GetAsyncKeyState(27):
        keys.append("esc")
    if wapi.GetAsyncKeyState(113):
        keys.append("f2")
    if wapi.GetAsyncKeyState(32):
        keys.append("space")
    if wapi.GetAsyncKeyState(96):
        keys.append("num0")
    return keys


```

这段代码定义了一个名为 `get_key` 的函数，它接受一个名为 `keys` 的列表参数。函数的作用是获取一个字典中的键，根据传入的键的个数，采取不同的操作，最终返回对应的键值。

如果传入的键只有一个，函数会尝试使用 Python 内置的字典 `dict` 来获取该键的值，并返回该键对应的值。

如果传入的键有两个 or more，函数会遍历这些键，尝试从字典中删除指定的键，并尝试在字典中查找指定的键。如果指定的键在字典中存在，函数将返回该键对应的值；否则，函数将返回一个默认值(在这种情况下，通常是 `93`)。

函数的实现基于以下几个假设：

- 字典中的键必须是唯一的。
- 指定的键只能有一个。
- 指定的键不可能在字典中存在。


```py
def get_key(keys):
    if len(keys) == 1:
        output = dict[keys[0]]
    elif len(keys) == 2:
        for k in keys:
            if k == "left" or k == "up" or k == "down" or k == "right":
                keys.pop(keys.index(k))
                key_name = k + "_" + keys[0]
                if key_name in dict.keys():
                    output = dict[key_name]
                else:
                    output = dict[keys[0]]
            else:
                output = dict[keys[0]]
    elif len(keys) > 2:
        output = dict[keys[0]]
    else:
        output = 93   # 不做任何动作
    return output

```

这段代码是一个Python脚本，它的作用是：

1. 如果当前脚本被作为主程序运行（即，通过调用`if __name__ == '__main__':`中的条件检查），那么执行以下内容：

2. 在一个无限循环中。

3. 首先，函数`get_key(key_check())`会获取一个键盘中的键（`key_check()`会返回一个字符类型的值，代表用户输入的键盘键），并检查它是否等于100。

4. 如果`get_key(key_check())`不等于100，那么它会打印键（即`key_check()`的值）并打印键的补码（即`get_key(key_check())`的值取反后得到的值）。

5. 如果`get_key(key_check())`等于100，那么它会打印"stop listen keyboard"，然后跳出无限循环。

6. 接着，函数`undict`被创建，它的作用是将从`dict`中读取的每个键值对（键和值）存储到`undict`中。

7. 最后，`undict`中的键值对被打印出来。


```py
if __name__ == '__main__':
    # while True:
    #     if get_key(key_check()) != 100:
    #         print(key_check())
    #         print(get_key(key_check()))
    #     else:
    #         print("stop listen keyboard")
    #         break
    undict = {}
    for key, val in dict.items():
        undict[val] = key
    print(undict)

```

# `grabscreen.py`

This appears to be a Python function named "grab\_screen" that takes a region of the screen as input and returns a pixel buffer (i.e., an image) of the same region. The region can be defined using OpenCV cv2.frame parameter.

The function uses the win32gui library to retrieve the desktop window of the current user, and the win32api library to retrieve system metrics such as the virtual screen size and location of the window. The function then retrieves the DC (Direct心肺) and咬合光标的DC from the window DC, and creates a bitmap of the same size and resolution as the screen. Finally, it copies the pixels of the bitmap to the image object, and returns the image.


```py
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 12:14:29 2020

@author: analoganddigital   ( GitHub )
"""
import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api

def grab_screen(region=None):

    hwin = win32gui.GetDesktopWindow()

    if region:
            left,top,x2,y2 = region
            width = x2 - left + 1
            height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height,width,4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return img
```

# `image_grab.py`

这段代码的作用是截取屏幕上的区域并进行数字化，然后将截取到的区域保存为图像文件。

具体的实现步骤如下：

1. 导入所需的库：numpy, PIL（用于获取图像）、cv2（用于图像处理）、directkeys（用于获取键值）、grabscreen（用于截取屏幕区域）、getkeys（用于获取按键）、os（用于操作文件）。
2. 定义一个常量，表示截取的区域大小，例如50x50像素。
3. 创建一个空的numpy数组，用于存储截取的屏幕区域。
4. 使用PIL的ImageGrab函数获取屏幕上指定大小的图像。
5. 使用cv2的imread函数将获取的图像转换为RGB格式。
6. 使用cv2的cvtColor函数将图像从RGB格式转换为灰度格式。
7. 使用cv2的resize函数将图像缩小到指定的大小。
8. 使用cv2的threshold函数将灰度图像转换为阈值图像，用于检测区域。
9. 使用cv2的findContours函数在阈值图像中查找轮廓。
10. 使用cv2的drawContours函数绘制轮廓。
11. 使用numpy的len函数检测轮廓是否正确，如果是，则说明截取成功。
12. 创建一个新的图像文件，并将截取的屏幕区域保存到该文件中。
13. 最后，使用直接keys库中的export功能将截取的屏幕区域保存为桌面上的壁纸。


```py
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:45:04 2020

@author: analoganddigital   ( GitHub )
"""

import numpy as np
from PIL import ImageGrab
import cv2
import time
import directkeys
import grabscreen
import getkeys
import os

```

这段代码的主要作用是创建一个名为“training_data_2_3.npy”的 numpy 数据文件，该文件将用于保存从名为“datasets/guiqi/material”的文件夹中读取的训练数据。

具体来说，代码首先定义了一个变量“wait_time”，其值为5，表示在读取数据之前需要等待一段时间(这对于训练过程中的超时控制非常重要)。

接着，代码定义了一个变量“L_t”，其值为3，用于记录训练数据中的稀疏标记数据。

然后，代码定义了一个变量“save_step”，其值为200，表示在训练过程中每个周期性地保存数据的时间步长。

接下来，代码定义了一个变量“file_name”，其值为“training_data_2_3.npy”，用于指定要保存数据的文件名。

接着，代码定义了一个变量“data_path”，其值为“datasets/guiqi/material”的路径，用于指定训练数据文件的存储位置。

然后，代码定义了一个变量“window_size”，其值为(0,0,1280,800)，表示窗口的大小(即图像的尺寸)，这是代码将在训练过程中读取的数据区域。

接下来，代码判断是否已经读取到了文件指定的位置，如果是，则代码将读取之前的数据并将其保存到“training_data”列表中，否则，则代码将会创建一个新的“training_data”列表，其中包含一个空列表。

最后，代码将“training_data”列表中的元素打印出来，以便在训练过程中进行超时控制。


```py
wait_time = 5
L_t = 3
save_step = 200
# file_name = 'training_data_2_3.npy'
data_path = 'datasets/guiqi/material'
window_size = (0,0,1280,800)#384,344  192,172 96,86

# if os.path.isfile(file_name):
#     print("file exists , loading previous data")
#     training_data = list(np.load(file_name,allow_pickle=True))
# else:
#     print("file don't exists , create new one")
#     training_data = []

training_data = []
```

运行该代码的注意事项：

1. 请确保您已安装`cv2`、`time`和`os`库，或者使用`pip`安装。

2. 请确保您的计算机上安装了`X看天下`或`神秘博士`等类似的分屏软件，以便在运行该代码时进行跨屏显示。

3. 运行该代码前，请先运行`getkeys.py`，以便收集按键。

4. 运行该代码时，可能会因为计算机性能和配置而出现运行时间不确定的情况，请耐心运行。


```py
save = True
for i in list(range(wait_time))[::-1]:
    print(i+1)
    time.sleep(1)

last_time = time.time()
counter = 0

org_num = len(os.listdir(data_path))
while(True):
    output_key = getkeys.get_key(getkeys.key_check())#按键收集
    if output_key == 100:
        if save:
            print(len(training_data) + counter*save_step)
            for i, d in enumerate(training_data):
                file_name = os.path.join(data_path, str(org_num + counter*save_step + i) + "_" + str(d[1]) + '.jpg')
                cv2.imwrite(file_name, d[0])
            print("save finish")
        break

    screen_gray = cv2.cvtColor(grabscreen.grab_screen(window_size),cv2.COLOR_BGRA2BGR)#灰度图像收集
    screen_reshape = cv2.resize(screen_gray,(1280,800)) # 1200, 750   600, 375

    training_data.append([screen_reshape,output_key])

    if len(training_data) % save_step == 0 and save:
        print(len(training_data))
        for i, d in enumerate(training_data):
            file_name = os.path.join(data_path, str(org_num + counter*save_step + i) + "_" + str(d[1]) + '.jpg')
            cv2.imwrite(file_name, d[0])
        training_data.clear()
        counter += 1
    cv2.imshow('window1',screen_reshape)

    #测试时间用
    print('每帧用时 {} 秒'.format(time.time()-last_time))
    print("瞬时fps：", 1/(time.time()-last_time))
    last_time = time.time()

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
```

这段代码是用Python中的OpenCV库实现的。具体来说，它起到了以下几个作用：

1. `cv2.waitKey()` 函数用于等待用户按下任意键，以便在视频播放结束后，按任意键可以退出。当用户按下任意键后，该函数会返回一个 `True` 值，表示用户已经按下了任意键，因此可以退出视频播放。

2. `cv2.destroyAllWindows()` 函数用于销毁打开的窗口，以便在程序关闭时，所有打开的窗口都关闭。这样，就可以确保程序不会留下任何未关闭的窗口，从而保持良好的用户体验。


```py
cv2.waitKey()# 视频结束后，按任意键退出
cv2.destroyAllWindows()

```

# `json2yolo.py`

这段代码的主要作用是定义了一个名为 `convert` 的函数，接受两个参数 `img_size` 和 `box`，对输入的图像大小和边界框进行转换，使得图像能够适应规则尺寸的显示设备。

具体来说，代码首先定义了一个名为 `name2id` 的字典，用于将英文名称与编号对应。接着，定义了一个 `convert` 函数，该函数接收两个参数 `img_size` 和 `box`，其中 `img_size` 是一个 tuple 类型的列表，包含了输入图像的大小，`box` 是一个 tuple 类型的列表，包含了输入图像的边界框。

在函数内部，代码先计算了每个边界框在原始图像中的坐标，然后根据图像尺寸对边界框进行缩放，将缩放因子 `dw` 和 `dh` 应用到边界框的坐标，将它们乘以 `img_size[0]` 和 `img_size[1]`，以得到每个边界框在缩放后的图像中的坐标。最后，函数返回经过缩放后的四个参数 `x`、`y`、`w` 和 `h`，它们分别对应于原始图像中的边界框在缩放后的四个坐标。

总之，这段代码定义了一个通用的图像转换函数，可以应用于将特定尺寸的图像转换为适应不同尺寸显示设备的尺寸，使得图像能够在不同尺寸的设备上显示时不会出现拉伸或压缩等问题。


```py

import json
import os

# name2id = {'person':0,'helmet':1,'Fire extinguisher':2,'Hook':3,'Gas cylinder':4}
name2id = {'hero': 0, 'small_map': 1, "monster": 2, 'money': 3, 'material': 4, 'door': 5, 'BOSS': 6, 'box': 7, 'options': 8}
               
def convert(img_size, box):
    dw = 1./(img_size[0])
    dh = 1./(img_size[1])
    x = (box[0] + box[2])/2.0 - 1
    y = (box[1] + box[3])/2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
 
 
```

这段代码的作用是读取一个JSON数据文件，其中的每个对象都包含一个矩形框(shapes)和对应的标签信息(labels)。它将读取数据文件中的内容，并将其存储为文本文件。

具体来说，代码首先定义了一个名为 decode_json 的函数，它接受两个参数 json_file_path 和 json_name，分别指定了 JSON 文件的路径和文件名。接着，函数定义了一个 txt_name 变量，用于指定文本文件的名称。

在函数内部，首先创建了一个名为 txt_file 的文件对象，并使用 write() 方法打开。接着，使用 os.path.join() 函数将 JSON 文件路径和文件名连接起来，以便从文件系统中读取文件。接着，代码通过 json.load() 函数从文件中读取 JSON 文件的内容，并将其存储为 data 变量。

接下来，代码遍历 data 变量中的每个形状(shapes)，并提取出其对应的信息，如矩形框的左上角坐标、宽度和高度，以及标签名称等。对于每个形状，代码都会计算其对应的矩形框，并将其转换为 pixel 数组。最后，代码将标签名称映射到从的诗库 id 映射表中，并输出到文本文件中。


```py
def decode_json(json_floder_path,json_name):
 
    txt_name = 'E:\\Computer_vision\\object_DNF\\datasets\\guiqi\\yolo5_datasets\\labels\\' + json_name[0:-5] + '.txt'
    txt_file = open(txt_name, 'w')
 
    json_path = os.path.join(json_floder_path, json_name)
    data = json.load(open(json_path, 'r', encoding='gb2312'))
 
    img_w = data['imageWidth']
    img_h = data['imageHeight']
 
    for i in data['shapes']:
        
        label_name = i['label']
        if (i['shape_type'] == 'rectangle'):
            print(txt_name)
            x1 = int(i['points'][0][0])
            y1 = int(i['points'][0][1])
            x2 = int(i['points'][1][0])
            y2 = int(i['points'][1][1])
 
            bb = (x1,y1,x2,y2)
            bbox = convert((img_w,img_h),bb)
            txt_file.write(str(name2id[label_name]) + " " + " ".join([str(a) for a in bbox]) + '\n')
    
```

这段代码的作用是读取一个名为 "E:\Computer_vision\object_DNF\datasets\guiqi\yolo5_datasets\labels_json" 的 JSON 文件，并将其解码为 Python 能够读取的 Python 字典。具体来说，代码首先通过 `os.listdir()` 函数获取 JSON 文件所在目录下的所有文件名，然后循环遍历这些文件名。在循环内部，使用 `os.path.isfile()` 函数判断文件是否合法，如果是，则使用 `json.loads()` 函数将 JSON 文件内容读取到一个 Python 字典中。注意，`json_floder_path` 变量是硬编码的，即在代码中直接写死了 `r'E:\Computer_vision\object_DNF\datasets\guiqi\yolo5_datasets\labels_json'`。


```py
if __name__ == "__main__":
    
    json_floder_path = r'E:\Computer_vision\object_DNF\datasets\guiqi\yolo5_datasets\labels_json'
    json_names = os.listdir(json_floder_path)
    for json_name in json_names:
        decode_json(json_floder_path,json_name)

```

# `main2.py`

这段代码的作用是实现了一个基于深度学习的物体检测模型。它包括以下组件：

1. 从numpy导入了一个用于数学计算的库，以及一个用于屏幕抓取的库，还有两个用于图像处理的库（OpenCV和PyTorch）。
2. 从grabscreen库中使用grab_screen方法获取了屏幕的当前图像。
3. 从cv2库中读取图像，并将其转换为numpy图像。
4. 从torch库中导入了一个Variable变量，用于表示模型的输入数据。
5. 从directkeys库中导入了一个PressKey、ReleaseKey和key_down、key_up函数，用于设置物体检测的方向。
6. 从getkeys库中导入了一个key_check函数，用于检查输入数据是否为数字。
7. 从torch库中导入了一个称为ModelType的类，用于定义模型的结构。
8. 从utils库中导入了一个设置图像大小的函数、一个非最大抑制的函数、一个将类别从图像中分离的函数、一个将坐标从xyxy2xywh转换为xywh2xyxy的函数、一个绘制一个物体框的函数以及一个用于屏幕抓取的函数。
9. 从models.experimental库中导入了一个称为AttachProductions的函数，用于设置模型的生产设备。
10. 从direction_move库中导入了一个名为move的函数，用于移动物体。
11. 从我的库中导入了一些 utility 函数，包括一个检查输入是否为数字的函数、一个加载分类器的函数、一个按需加载模型的函数、一个设置日志记录的函数以及一个绘制一个物体框的函数。
12. 从我的库中导入了 Model 和 Vision 类，用于定义模型的结构和进行物体检测。
13. 使用 attempt_load函数加载模型，并使用 load_classifier函数加载分类器。
14. 使用 time_synchronized 函数保证在训练过程中时间步长一致。
15. 使用 select_device 函数选择 GPU 或 CPU。
16. 使用 load_classifier 函数加载分类器模型。
17. 导入了一些直接使用的函数，包括一个检查图像大小的函数、一个非最大抑制的函数、一个将类别从图像中分离的函数、一个将坐标从xyxy2xywh转换为xywh2xyxy的函数、一个绘制一个物体框的函数以及一个用于屏幕抓取的函数。
18. 从models.experimental库中使用AttachProductions函数设置模型的生产设备。
19. 从direction_move库中使用move函数移动物体。


```py
import numpy as np
from grabscreen import grab_screen
import cv2
import time
import directkeys
import torch
from torch.autograd import Variable
from directkeys import PressKey, ReleaseKey, key_down, key_up
from getkeys import key_check
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, xywh2xyxy, plot_one_box, strip_optimizer, set_logging)
from models.experimental import attempt_load
from direction_move import move
```




```py
from small_recgonize import current_door, next_door
from skill_recgnize import skill_rec
import random

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

```

这段代码是一个Python脚本，它设置了一个YOLOv5模型的参数，用于进行物体检测。具体来说，它做了以下几件事情：

1. 设置了一个字符串变量weights，它指定了存储YOLOv5模型文件的路径。
2. 检查了PyTorch是否支持GPU，并把设备设置为使用GPU（如果可用）。如果没有GPU，则使用CPU设备。
3. 加载了一个FP32版本的模型，并将其存储在变量model中。
4. 设置了一个变量window_size，用于控制截屏的位置和尺寸。
5. 设置了一个变量img_size，用于指定输入到模型中的图像的大小。
6. 设置了一个变量paused，用于控制是否暂停模型的运行。
7. 设置了一个变量view_img，用于控制是否查看目标检测结果。
8. 设置了一个变量save_txt，用于控制是否保存检测结果为文本文件。
9. 设置了一个变量conf_thres和iou_thres，用于控制NMS的置信度阈值。
10. 初始化了一个变量classes，用于保存用于检测的类别名称。

总之，这段代码定义了一个用于检测物体目标的函数，它使用了YOLOv5模型，可以根据用户设置的参数来调整模型的检测效果。


```py
# 设置所有用到的参数
weights = r'E:\Computer_vision\yolov5\YOLO5\yolov5-master\DNF_runs\4s\weights\best.pt'    #yolo5 模型存放的位置
# weights = r'F:\Computer_vision\yolov5\YOLO5\yolov5-master\runs\exp0\weights\best.pt'
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
model = attempt_load(weights, map_location=device)  # load FP32 model
window_size = (0,0,1280,800)          # 截屏的位置
img_size = 800                      # 输入到yolo5中的模型尺寸
paused = False
half = device.type != 'cpu'
view_img = True                     # 是否观看目标检测结果
save_txt = False
conf_thres = 0.3                    # NMS的置信度过滤
iou_thres = 0.2                     # NMS的IOU阈值
classes = None
```

这段代码是一个用于游戏开发的动作捕捉工具，其目的是让玩家在游戏中使用不同的技能，并实现一些特性，如延迟和按键检测。以下是这段代码的详细解释：

1. `agnostic_nms = False`：设置了一个布尔值，表示是否在游戏进行时执行非阻塞IQN(Immediate Quality Notification)机制。IQN是一种能够在游戏进行时快速检测和响应技术，可以帮助实现更好的游戏性能。

2. `skill_char = "XYHGXFAXDSWXETX"`：定义了一个字符，用于表示游戏中不同的技能。这个字符串用于随机选择一个技能，以实现游戏中的技能系统。

3. `direct_dic = {"UP": 0xC8, "DOWN": 0xD0, "LEFT": 0xCB, "RIGHT": 0xCD}`：定义了一个字典，用于存储游戏中的方向键。这个字典 maps 方向键 to 键码，以便在游戏过程中实现键盘输入检测和对应的键位。

4. `names = ['hero', 'small_map', "monster', 'money', 'material', 'door', 'BOSS', 'box', 'options']`：定义了一个列表，用于存储游戏中所有玩家的目标。这个列表包含了游戏中的各种目标，如“hero”、“small_map”、“monster”、“money”、“material”、“door”、“BOSS”、“box”、“options”等。

5. `colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]`：用于在游戏中给每个目标分配一个颜色。颜色是随机生成的，用于在游戏过程中区分不同的目标。

6. `if half:`：这是一个条件语句，用于判断是否使用半浮点数进行模型转换。如果 half 为 True，则表示使用半浮点数进行模型转换。

7. `model.half()`：这是 PyTorch中的一个函数，用于将模型从浮点数表示转换为半浮点数表示。这个函数可以帮助游戏在FP16浮点数表示下更有效地利用内存，并提高游戏的性能。

8. `action_cache = None`：用于存储玩家在游戏中的动作，如技能、方向键等。

9. `press_delay = 0.1`：设置了一个变量，表示玩家按键的延迟时间。这个延迟时间可以根据游戏需要进行调整，以提供更好的用户体验。

10. `release_delay = 0.1`：设置了一个变量，表示玩家释放按键的时间延迟。这个延迟时间也可以根据游戏需要进行调整，以提供更好的用户体验。

11. `last_time = time.time()`：设置了一个变量，表示上一次游戏进行的时间。这个变量可以在游戏中用于记录和记录游戏时间，以帮助实现游戏中的进度和统计。

12. `frame = 0`：设置了一个变量，表示当前游戏的帧数。这个变量可以在游戏中用于跟踪游戏中的进度和状态，以便在游戏进行时进行更新和渲染。

13. `door1_time_start = -20`：设置了一个变量，表示玩家当前按键的时间戳。这个变量可以在游戏中用于检测玩家是否按下了某个技能或按键，以及技能或按键什么时候被执行的。

14. `next_door_time = -20`：设置了一个变量，表示玩家技能释放的时间戳。这个变量可以在游戏中用于检测玩家是否释放了一个技能，并准许玩家在技能释放后进入游戏状态。


```py
agnostic_nms = False                # 不同类别的NMS时也参数过滤
skill_char = "XYHGXFAXDSWXETX"          # 技能按键，使用均匀分布随机抽取
direct_dic = {"UP": 0xC8, "DOWN": 0xD0, "LEFT": 0xCB, "RIGHT": 0xCD}    # 上下左右的键码
names = ['hero', 'small_map', "monster", 'money', 'material', 'door', 'BOSS', 'box', 'options']   # 所有类别名
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
if half:
    model.half()  # to FP16
action_cache = None  # 动作标记
press_delay = 0.1  # 按压时间
release_delay = 0.1  # 释放时间
# last_time = time.time()
frame = 0  # 帧
door1_time_start = -20
next_door_time = -20
fs = 1 # 每四帧处理一次

```



This is a Python game program that uses the Pygame library to create a 2D game board with four slides. The program is控制在 via the `keys` dictionary, which is generated by the `key_check` function.

The `key_check` function checks if the keys pressed are valid. If the keys are valid, the program will perform the corresponding action (e.g. moving an item to the four slides). The program will always pause if the user presses the space bar to continue.

The program has a variable `view_img` that is a boolean indicating whether the user should display the game board in the main window or not.

There are also some comments in the program that indicate the purpose of some of the functions, such as the `ReleaseKey` function, which is used to release the key that was pressed.


```py
# 倒计时
for i in list(range(5))[::-1]:
    print(i + 1)
    time.sleep(1)

# 捕捉画面+目标检测+玩游戏
while True:
    if not paused:
        t_start = time.time()
        img0 = grab_screen(window_size)
        frame += 1
        if frame % fs == 0:
            # img0 = cv2.imread("datasets/guiqi/yolo5_datasets/imgs/1004_14.jpg")
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGRA2BGR)
            # Padded resize
            img = letterbox(img0, new_shape=img_size)[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device).unsqueeze(0)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            pred = model(img, augment=False)[0]

            # Apply NMS
            det = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            det = det[0]
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                img_object = []
                cls_object = []
                # Write results
                hero_conf = 0
                hero_index = 0
                for idx, (*xyxy, conf, cls) in enumerate(reversed(det)):
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    cls = int(cls)
                    img_object.append(xywh)
                    cls_object.append(names[cls])

                    if names[cls] == "hero" and conf > hero_conf:
                        hero_conf = conf
                        hero_index = idx


                    if view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=2)

                # 游戏
                thx = 30    # 捡东西时，x方向的阈值
                thy = 30    # 捡东西时，y方向的阈值
                attx = 150   # 攻击时，x方向的阈值
                atty = 50   # 攻击时，y方向的阈值

                if current_door(img0) == 1 and time.time() - door1_time_start > 10:
                    door1_time_start = time.time()
                    # move(direct="RIGHT", action_cache=action_cache, press_delay=press_delay,
                    #      release_delay=release_delay)
                    # ReleaseKey(direct_dic["RIGHT"])
                    # directkeys.key_press("SPACE")
                    directkeys.key_press("CTRL")
                    time.sleep(1)
                    directkeys.key_press("ALT")
                    time.sleep(0.5)
                    action_cache = None
                # 扫描英雄
                if "hero" in cls_object:
                    # hero_xywh = img_object[cls_object.index("hero")]
                    hero_xywh = img_object[hero_index]
                    cv2.circle(img0, (int(hero_xywh[0]), int(hero_xywh[1])), 1, (0,0,255), 10)
                    # print(hero_index)
                    # print(cls_object.index("hero"))
                else:
                    continue
                # 打怪
                if "monster" in cls_object or "BOSS" in cls_object:
                    min_distance = float("inf")
                    for idx, (c, box) in enumerate(zip(cls_object, img_object)):
                        if c == 'monster' or c == "BOSS":
                            dis = ((hero_xywh[0] - box[0])**2 + (hero_xywh[1] - box[1])**2)**0.5
                            if dis < min_distance:
                                monster_box = box
                                monster_index = idx
                                min_distance = dis
                    if abs(hero_xywh[0] - monster_box[0]) < attx and abs(hero_xywh[1] - monster_box[1]) < atty:
                        if "BOSS" in cls_object:
                            directkeys.key_press("R")
                            directkeys.key_press("Q")
                            # time.sleep(0.5)
                            skill_name = skill_char[int(np.random.randint(len(skill_char), size=1)[0])]
                            while True:
                                if skill_rec(skill_name, img0):
                                    directkeys.key_press(skill_name)
                                    directkeys.key_press(skill_name)
                                    directkeys.key_press(skill_name)
                                    break
                                else:
                                    skill_name = skill_char[int(np.random.randint(len(skill_char), size=1)[0])]

                        else:
                            skill_name = skill_char[int(np.random.randint(len(skill_char), size=1)[0])]
                            while True:
                                if skill_rec(skill_name, img0):
                                    directkeys.key_press(skill_name)
                                    directkeys.key_press(skill_name)
                                    directkeys.key_press(skill_name)
                                    break
                                else:
                                    skill_name = skill_char[int(np.random.randint(len(skill_char), size=1)[0])]
                        print("释放技能攻击")
                        if not action_cache:
                            pass
                        elif action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                            ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                            ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                            action_cache = None
                        elif action_cache:
                            ReleaseKey(direct_dic[action_cache])
                            action_cache = None
                        # break
                    elif monster_box[1] - hero_xywh[1] < 0 and monster_box[0] - hero_xywh[0] > 0:
                        if abs(monster_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="RIGHT", material=True, action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - monster_box[1] < monster_box[0] - hero_xywh[0]:
                            action_cache = move(direct="RIGHT_UP", material=True, action_cache=action_cache,
                                                press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - monster_box[1] >= monster_box[0] - hero_xywh[0]:
                            action_cache = move(direct="UP", material=True, action_cache=action_cache,
                                                    press_delay=press_delay,
                                                    release_delay=release_delay)
                        # break
                    elif monster_box[1] - hero_xywh[1] < 0 and monster_box[0] - hero_xywh[0] < 0:
                        if abs(monster_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="LEFT", material=True, action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - monster_box[1] < hero_xywh[0] - monster_box[0]:
                            action_cache = move(direct="LEFT_UP", material=True, action_cache=action_cache,
                                                    press_delay=press_delay,
                                                    release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - monster_box[1] >= hero_xywh[0] - monster_box[0]:
                            action_cache = move(direct="UP", material=True, action_cache=action_cache,
                                                    press_delay=press_delay,
                                                    release_delay=release_delay)
                            # break
                    elif monster_box[1] - hero_xywh[1] > 0 and monster_box[0] - hero_xywh[0] < 0:
                        if abs(monster_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="LEFT", material=True, action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif monster_box[1] - hero_xywh[1] < hero_xywh[0] - monster_box[0]:
                            action_cache = move(direct="LEFT_DOWN", material=True, action_cache=action_cache,
                                                    press_delay=press_delay,
                                                    release_delay=release_delay)
                            # break
                        elif monster_box[1] - hero_xywh[1] >= hero_xywh[0] - monster_box[0]:
                            action_cache = move(direct="DOWN", material=True, action_cache=action_cache,
                                                    press_delay=press_delay,
                                                    release_delay=release_delay)
                            # break
                    elif monster_box[1] - hero_xywh[1] > 0 and monster_box[0] - hero_xywh[0] > 0:
                        if abs(monster_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="RIGHT", material=True, action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif monster_box[1] - hero_xywh[1] < monster_box[0] - hero_xywh[0]:
                            action_cache = move(direct="RIGHT_DOWN", material=True, action_cache=action_cache,
                                                    press_delay=press_delay,
                                                    release_delay=release_delay)
                            # break
                        elif monster_box[1] - hero_xywh[1] >= monster_box[0] - hero_xywh[0]:
                            action_cache = move(direct="DOWN", material=True, action_cache=action_cache,
                                                    press_delay=press_delay,
                                                    release_delay=release_delay)
                            # break

                # 移动到下一个地图
                if "door" in cls_object and "monster" not in cls_object and "BOSS" not in cls_object and "material" not in cls_object and "money" not in cls_object:
                    for idx, (c, box) in enumerate(zip(cls_object, img_object)):
                        if c == 'door':
                            door_box = box
                            door_index = idx
                    if door_box[0] < img0.shape[0] // 2:
                        action_cache = move(direct="RIGHT", action_cache=action_cache, press_delay=press_delay,
                                            release_delay=release_delay)
                        # break
                    elif door_box[1] - hero_xywh[1] < 0 and door_box[0] - hero_xywh[0] > 0:
                        if abs(door_box[1] - hero_xywh[1]) < thy and abs(door_box[0] - hero_xywh[0]) < thx:
                            action_cache = None
                            print("进入下一地图")
                            # break
                        elif abs(door_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="RIGHT", action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - door_box[1] < door_box[0] - hero_xywh[0]:
                            action_cache = move(direct="RIGHT_UP", action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - door_box[1] >= door_box[0] - hero_xywh[0]:
                            action_cache = move(direct="UP", action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                    elif door_box[1] - hero_xywh[1] < 0 and door_box[0] - hero_xywh[0] < 0:
                        if abs(door_box[1] - hero_xywh[1]) < thy and abs(door_box[0] - hero_xywh[0]) < thx:
                            action_cache = None
                            print("进入下一地图")
                            # break
                        elif abs(door_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="LEFT", action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - door_box[1] < hero_xywh[0] - door_box[0]:
                            action_cache = move(direct="LEFT_UP", action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - door_box[1] >= hero_xywh[0] - door_box[0]:
                            action_cache = move(direct="UP", action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                    elif door_box[1] - hero_xywh[1] > 0 and door_box[0] - hero_xywh[0] < 0:
                        if abs(door_box[1] - hero_xywh[1]) < thy and abs(door_box[0] - hero_xywh[0]) < thx:
                            action_cache = None
                            print("进入下一地图")
                            # break
                        elif abs(door_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="LEFT", action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif door_box[1] - hero_xywh[1] < hero_xywh[0] - door_box[0]:
                            action_cache = move(direct="LEFT_DOWN", action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif door_box[1] - hero_xywh[1] >= hero_xywh[0] - door_box[0]:
                            action_cache = move(direct="DOWN", action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                    elif door_box[1] - hero_xywh[1] > 0 and door_box[0] - hero_xywh[0] > 0:
                        if abs(door_box[1] - hero_xywh[1]) < thy and abs(door_box[0] - hero_xywh[0]) < thx:
                            action_cache = None
                            print("进入下一地图")
                            # break
                        elif abs(door_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="RIGHT", action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif door_box[1] - hero_xywh[1] < door_box[0] - hero_xywh[0]:
                            action_cache = move(direct="RIGHT_DOWN", action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif door_box[1] - hero_xywh[1] >= door_box[0] - hero_xywh[0]:
                            action_cache = move(direct="DOWN", action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                if "money" not in cls_object and "material" not in cls_object and "monster" not in cls_object \
                        and "BOSS" not in cls_object and "door" not in cls_object and 'box' not in cls_object \
                        and 'options' not in cls_object:
                    # if next_door(img0) == 0 and abs(time.time()) - next_door_time > 10:
                    #     next_door_time = time.time()
                    #     action_cache = move(direct="LEFT", action_cache=action_cache, press_delay=press_delay,
                    #                         release_delay=release_delay)
                    #     # time.sleep(3)
                    # else:
                    #     action_cache = move(direct="RIGHT", action_cache=action_cache, press_delay=press_delay,
                    #                     release_delay=release_delay)

                    action_cache = move(direct="RIGHT", action_cache=action_cache, press_delay=press_delay,
                                        release_delay=release_delay)
                    # break

                # 捡材料
                if "monster" not in cls_object and "hero" in cls_object and ("material" in cls_object or "money" in cls_object):
                    min_distance = float("inf")
                    hero_xywh[1] = hero_xywh[1] + (hero_xywh[3] // 2) * 0.7
                    thx = thx / 2
                    thy = thy / 2
                    for idx, (c, box) in enumerate(zip(cls_object, img_object)):
                        if c == 'material' or c == "money":
                            dis = ((hero_xywh[0] - box[0]) ** 2 + (hero_xywh[1] - box[1]) ** 2) ** 0.5
                            if dis < min_distance:
                                material_box = box
                                material_index = idx
                                min_distance = dis
                    if abs(material_box[1] - hero_xywh[1]) < thy and abs(material_box[0] - hero_xywh[0]) < thx:
                        if not action_cache:
                            pass
                        elif action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                            ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                            ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                            action_cache = None
                        else:
                            ReleaseKey(direct_dic[action_cache])
                            action_cache = None
                        time.sleep(1)
                        directkeys.key_press("X")
                        print("捡东西")
                        # break

                    elif material_box[1] - hero_xywh[1] < 0 and material_box[0] - hero_xywh[0] > 0:

                        if abs(material_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="RIGHT", material=True, action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - material_box[1] < material_box[0] - hero_xywh[0]:
                            action_cache = move(direct="RIGHT_UP", material=True, action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - material_box[1] >= material_box[0] - hero_xywh[0]:
                            action_cache = move(direct="UP", material=True, action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                    elif material_box[1] - hero_xywh[1] < 0 and material_box[0] - hero_xywh[0] < 0:
                        if abs(material_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="LEFT", material=True, action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - material_box[1] < hero_xywh[0] - material_box[0]:
                            action_cache = move(direct="LEFT_UP", material=True, action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif hero_xywh[1] - material_box[1] >= hero_xywh[0] - material_box[0]:
                            action_cache = move(direct="UP", material=True, action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                    elif material_box[1] - hero_xywh[1] > 0 and material_box[0] - hero_xywh[0] < 0:
                        if abs(material_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="LEFT", material=True, action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif material_box[1] - hero_xywh[1] < hero_xywh[0] - material_box[0]:
                            action_cache = move(direct="LEFT_DOWN", material=True, action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif material_box[1] - hero_xywh[1] >= hero_xywh[0] - material_box[0]:
                            action_cache = move(direct="DOWN", material=True, action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                    elif material_box[1] - hero_xywh[1] > 0 and material_box[0] - hero_xywh[0] > 0:
                        if abs(material_box[1] - hero_xywh[1]) < thy:
                            action_cache = move(direct="RIGHT", material=True, action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif material_box[1] - hero_xywh[1] < material_box[0] - hero_xywh[0]:
                            action_cache = move(direct="RIGHT_DOWN", material=True, action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                        elif material_box[1] - hero_xywh[1] >= material_box[0] - hero_xywh[0]:
                            action_cache = move(direct="DOWN", material=True, action_cache=action_cache, press_delay=press_delay,
                                                release_delay=release_delay)
                            # break
                # 开箱子
                if "box" in cls_object:
                    box_num = 0
                    for b in cls_object:
                        if b == "box":
                            box_num += 1
                    if box_num >= 4:
                        directkeys.key_press("ESC")
                        print("打开箱子ESC")
                        # break62

                # 重新开始
                time_option = -20
                if "options" in cls_object:
                    if not action_cache:
                        pass
                    elif action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                        ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                        ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                        action_cache = None
                    else:
                        ReleaseKey(direct_dic[action_cache])
                        action_cache = None
                    if time.time() - time_option > 10:
                        directkeys.key_press("NUM0")
                        print("移动物品到脚下")
                        directkeys.key_press("X")
                        time_option = time.time()
                    directkeys.key_press("F2")
                    print("重新开始F2")
                    # break
            t_end = time.time()
            print("一帧游戏操作所用时间：", (t_end - t_start)/fs)

            img0 = cv2.resize(img0, (600, 375))
            # Stream results
            if view_img:
                cv2.imshow('window', img0)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    raise StopIteration


    # Setting pause and unpause
    keys = key_check()
    if 'P' in keys:
        if not action_cache:
            pass
        elif action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
            ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
            ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
            action_cache = None
        else:
            ReleaseKey(direct_dic[action_cache])
            action_cache = None
        if paused:
            paused = False
            time.sleep(1)
        else:
            paused = True
            time.sleep(1)


```