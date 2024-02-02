# `yolov5-DNF\yabl.py`

```py
# 导入随机数模块
import random
# 导入 OpenCV 模块
import cv2
# 导入 NumPy 模块
import numpy as np
# 导入 PyTorch 模块
import torch
# 导入系统模块
import sys
# 导入时间模块
import time
# 将上级目录添加到系统路径中
sys.path.append('../')
# 从 directkeys 模块中导入 key_press 和 ReleaseKey 函数
from directkeys import (key_press, ReleaseKey)
# 从 direction_move 模块中导入 move 函数
from direction_move import move
# 从 grabscreen 模块中导入 grab_screen 函数
from grabscreen import grab_screen
# 从 models.experimental 模块中导入 attempt_load 函数
from models.experimental import attempt_load
# 从 utils.general 模块中导入 non_max_suppression 和 xyxy2xywh 函数
from utils.general import non_max_suppression, xyxy2xywh
# 从 getkeys 模块中导入 key_check 函数

# 定义一个字典，包含上下左右键的键码
direct_dic = {"UP": 0xC8, "DOWN": 0xD0, "LEFT": 0xCB, "RIGHT": 0xCD}
# 定义一个列表，包含游戏中的角色名称
names = ['monster', 'hero', 'boss', 'option']
# 为每个角色生成一个随机颜色
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
# 指定 yolo5 模型的存放位置
weights = 'best.pt'
# 如果有可用的 CUDA 设备，则使用 CUDA 设备，否则使用 CPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# 判断是否使用半精度浮点数运算
half = device.type != 'cpu'
# 尝试加载模型
model = attempt_load(weights, device)
# 如果使用半精度浮点数运算，则将模型转换为半精度
if half:
    model.half()  # to FP16
# NMS 的置信度过滤阈值
conf_thres = 0.3
# NMS 的 IOU 阈值
iou_thres = 0.2
# 类别列表
classes = None
# 是否对不同类别进行 NMS 参数过滤
agnostic_nms = False
# 是否显示图像
view_img = True
# 帧数
frame = 0
# 帧率
fs = 4
# 动作标记
action_cache = None
# 游戏是否暂停
paused = False

# 定义一个函数，用于在图像上绘制一个边界框
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # 计算线条/字体的厚度
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    # 生成随机颜色
    color = color or [random.randint(0, 255) for _ in range(3)]
    # 获取边界框的两个顶点坐标
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # 在图像上绘制矩形边界框
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    # 如果有标签，则在图像上绘制标签
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

# 进入无限循环
while True:
```