# `yolov5-DNF\test\directInput.py`

```py
# 导入需要的库
import pyautogui
import pydirectinput
import time
import win32con

# 从directkeys模块中导入PressKey, key_press, ReleaseKey函数
from directkeys import (PressKey, key_press, ReleaseKey)
import win32api

# 等待2秒
time.sleep(2)
# 模拟按下'i'键
pydirectinput.press('i')
# 等待1秒
time.sleep(1)
# 将鼠标移动到指定位置(776, 471)，持续时间为0.2秒
pyautogui.moveTo(776, 471, duration=0.2)
# 等待0.5秒
time.sleep(0.5)
# 在指定位置(776, 471)执行鼠标右键单击
pydirectinput.rightClick(776, 471)
# 等待1秒
time.sleep(1)
# 模拟按下'a'键
key_press('a')
# 打印信息
print('打完了')
```