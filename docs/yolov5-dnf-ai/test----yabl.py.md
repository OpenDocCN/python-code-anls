# `yolov5-DNF\test\yabl.py`

```py
# 导入随机模块
import random
# 导入opencv模块
import cv2
# 导入numpy模块
import numpy as np
# 导入torch模块
import torch
# 导入时间模块
import time
# 导入directkeys模块
import directkeys
# 从directkeys模块中导入key_press, ReleaseKey, PressKey函数
from directkeys import (key_press, ReleaseKey, PressKey)
# 从direction_move模块中导入move函数
from direction_move import move
# 从grabscreen模块中导入grab_screen函数
from grabscreen import grab_screen
# 从models.experimental模块中导入attempt_load函数
from models.experimental import attempt_load
# 从utils.general模块中导入non_max_suppression, xyxy2xywh函数
from utils.general import non_max_suppression, xyxy2xywh
# 从getkeys模块中导入key_check函数
from getkeys import key_check

# 定义上下左右的键码
direct_dic = {"UP": 0xC8, "DOWN": 0xD0, "LEFT": 0xCB, "RIGHT": 0xCD}
# 定义角色名称列表
names = ['monster', 'hero', 'boss', 'option']
# 为每个角色生成随机颜色
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
# yolo5 模型存放的位置
weights = 'best.pt'
# 如果有GPU，则使用GPU，否则使用CPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# 判断是否使用半精度浮点数
half = device.type != 'cpu'
# 加载yolo5模型
model = attempt_load(weights, device)
# 如果使用半精度浮点数，则转换模型为FP16
if half:
    model.half()
# NMS的置信度过滤
conf_thres = 0.3
# NMS的IOU阈值
iou_thres = 0.2
# 类别列表
classes = None
# 不同类别的NMS时也参数过滤
agnostic_nms = False
# 是否显示图像
view_img = True
# 帧数
frame = 0
# 帧速率
fs = 20
# 动作标记
action_cache = None
# 游戏是否暂停
paused = False

# 定义绘制单个边界框的函数
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # 计算线条/字体的厚度
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    # 生成随机颜色
    color = color or [random.randint(0, 255) for _ in range(3)]
    # 获取边界框的两个顶点坐标
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # 在图像上绘制矩形边界框
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    # 如果有标签，则在边界框上绘制标签
    if label:
        # 计算字体的厚度
        tf = max(tl - 1, 1)
        # 获取标签的大小
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        # 计算标签的位置
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # 在图像上绘制填充矩形
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        # 在图像上绘制标签
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# 幽暗密林的门的方向
yamlDoorDirection = ['RIGHT', 'RIGHT', 'UP', 'RIGHT']
# 王之摇篮的门的方向
wzylDoorDirection = ['UP', 'LEFT', 'UP', 'RIGHT', 'UP', 'UP']

# 进入循环
while True:
    # 检测按键
    keys = key_check()
    # 如果键盘按键 'P' 在键盘输入中
    if 'P' in keys:
        # 如果动作缓存为空
        if not action_cache:
            pass
        # 如果动作缓存不在 ["LEFT", "RIGHT", "UP", "DOWN"] 中
        elif action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
            # 释放动作缓存中第一个动作对应的键盘按键
            ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
            # 释放动作缓存中第二个动作对应的键盘按键
            ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
            # 将动作缓存置为 None
            action_cache = None
        # 如果动作缓存在 ["LEFT", "RIGHT", "UP", "DOWN"] 中
        else:
            # 释放动作缓存对应的键盘按键
            ReleaseKey(direct_dic[action_cache])
            # 将动作缓存置为 None
            action_cache = None
        # 如果游戏处于暂停状态
        if paused:
            # 将暂停状态置为 False
            paused = False
            # 等待 1 秒
            time.sleep(1)
        # 如果游戏不处于暂停状态
        else:
            # 将暂停状态置为 True
            paused = True
            # 等待 1 秒
            time.sleep(1)
```