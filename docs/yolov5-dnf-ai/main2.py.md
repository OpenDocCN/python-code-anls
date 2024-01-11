# `yolov5-DNF\main2.py`

```
# 导入需要的库
import random
import time

import cv2
import numpy as np
import torch

import directkeys
from direction_move import move
from directkeys import ReleaseKey
from getkeys import key_check
from grabscreen import grab_screen
from models.experimental import attempt_load
# from skill_recgnize import skill_rec
from small_recgonize import current_door
from utils.general import (
    non_max_suppression, scale_coords,
    xyxy2xywh)

# 定义一个函数，用于将图像调整为指定大小的矩形
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    # 获取图像的当前形状（高度和宽度）
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 计算缩放比例（新的形状 / 旧的形状）
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 只缩小，不放大（用于更好的测试 mAP）
        r = min(r, 1.0)

    # 计算填充
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # 最小矩形
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # 拉伸
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # 将填充分成两侧
    dh /= 2

    if shape[::-1] != new_unpad:  # 调整大小
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 添加边框
    return img, ratio, (dw, dh)

# 设置所有用到的参数
weights = r'resource\best.pt'  # yolo5 模型存放的位置
# 根据是否有可用的 CUDA 设备选择运行设备
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# 加载 FP32 模型
model = attempt_load(weights, map_location=device)
# 截屏的位置
window_size = (0, 0, 1280, 800)
# 输入到 yolo5 中的模型尺寸
img_size = 640
# 是否暂停
paused = False
# 是否使用 GPU 的一半内存
half = device.type != 'cpu'
# 是否观看目标检测结果
view_img = True
# 是否保存检测结果到文本文件
save_txt = False
# NMS 的置信度过滤
conf_thres = 0.3
# NMS 的 IOU 阈值
iou_thres = 0.2
# 目标类别
classes = None
# 不同类别的 NMS 时也参数过滤
agnostic_nms = False
# 技能按键，使用均匀分布随机抽取
skill_char = "XYHGXFAXDSWXETX"
# 上下左右的键码
direct_dic = {"UP": 0xC8, "DOWN": 0xD0, "LEFT": 0xCB, "RIGHT": 0xCD}
# 所有类别名
names = ['hero', 'small_map', "monster", 'money', 'material', 'door', 'BOSS', 'box', 'options']
# 每个类别的颜色
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
# 如果使用 GPU 的一半内存，则将模型转换为 FP16
if half:
    model.half()
# 动作标记
action_cache = None
# 按压时间
press_delay = 0.1
# 释放时间
release_delay = 0.1
# 帧
frame = 0
# 第一个门的时间开始
door1_time_start = -20
# 下一个门的时间
next_door_time = -20
# 每四帧处理一次
fs = 1

# 绘制一个边界框
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # 在图像 img 上绘制一个边界框
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # 线条/字体的厚度
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # 字体的厚度
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # 填充
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# 捕捉画面+目标检测+玩游戏
while True:
    # 设置暂停和取消暂停
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
            # 将动作缓存置为空
            action_cache = None
        # 如果动作缓存在 ["LEFT", "RIGHT", "UP", "DOWN"] 中
        else:
            # 释放动作缓存对应的键盘按键
            ReleaseKey(direct_dic[action_cache])
            # 将动作缓存置为空
            action_cache = None
        # 如果程序处于暂停状态
        if paused:
            # 将暂停状态置为 False
            paused = False
            # 程序休眠 1 秒
            time.sleep(1)
        # 如果程序不处于暂停状态
        else:
            # 将暂停状态置为 True
            paused = True
            # 程序休眠 1 秒
            time.sleep(1)
```