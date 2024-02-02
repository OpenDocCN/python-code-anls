# `yolov5-DNF\test\main2.py`

```py
# 导入需要的库
import random
import time

import cv2
import numpy as np
import torch

# 导入自定义模块
import directkeys
from direction_move import move
from directkeys import ReleaseKey
from getkeys import key_check
from grabscreen import grab_screen
from models.experimental import attempt_load
from utils.general import (
    non_max_suppression, scale_coords,
    xyxy2xywh)

# 定义函数，用于将图像调整为指定大小的矩形
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    # 获取图像的当前形状 [高度, 宽度]
    shape = img.shape[:2]  
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 计算缩放比例 (新 / 旧)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 只缩小，不放大 (用于更好的测试 mAP)
        r = min(r, 1.0)

    # 计算填充
    ratio = r, r  # 宽度、高度比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 宽高填充
    if auto:  # 最小矩形
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # 宽高填充
    elif scaleFill:  # 拉伸
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 宽度、高度比例

    dw /= 2  # 将填充分成两侧
    dh /= 2

    if shape[::-1] != new_unpad:  # 调整大小
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 添加边框
    return img, ratio, (dw, dh)

# 设置所有用到的参数
weights = 'best.pt'  # yolo5 模型存放的位置
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# 加载FP32模型
model = attempt_load(weights, device)  # load FP32 model
# 设置截屏的位置
window_size = (0, 0, 1280, 800)  # 截屏的位置
# 输入到yolo5中的模型尺寸
img_size = 640  # 输入到yolo5中的模型尺寸
# 是否暂停
paused = False
# 是否使用GPU的一半内存
half = device.type != 'cpu'
# 是否观看目标检测结果
view_img = True  # 是否观看目标检测结果
# 是否保存文本文件
save_txt = False
# NMS的置信度过滤
conf_thres = 0.3  # NMS的置信度过滤
# NMS的IOU阈值
iou_thres = 0.2  # NMS的IOU阈值
# 类别
classes = None
# 不同类别的NMS时也参数过滤
agnostic_nms = False  # 不同类别的NMS时也参数过滤
# 技能按键，使用均匀分布随机抽取
skill_char = "XYHGXFAXDSWXETX"  # 技能按键，使用均匀分布随机抽取
# 上下左右的键码
direct_dic = {"UP": 0xC8, "DOWN": 0xD0, "LEFT": 0xCB, "RIGHT": 0xCD}  # 上下左右的键码
# 所有类别名
names = ['hero', 'small_map', "monster", 'money', 'material', 'door', 'BOSS', 'box', 'options']  # 所有类别名
# 随机颜色
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
# 如果使用GPU的一半内存，则将模型转换为FP16
if half:
    model.half()  # to FP16
# 动作标记
action_cache = None  # 动作标记
# 按压时间
press_delay = 0.1  # 按压时间
# 释放时间
release_delay = 0.1  # 释放时间
# 帧
frame = 0  # 帧
# 第一个门的时间开始
door1_time_start = -20
# 下一个门的时间
next_door_time = -20
# 每四帧处理一次
fs = 1  # 每四帧处理一次

# 绘制一个边界框
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    # 线条/字体的厚度
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    # 随机颜色
    color = color or [random.randint(0, 255) for _ in range(3)]
    # 边界框的两个点
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # 绘制矩形
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    # 如果有标签，则绘制标签
    if label:
        # 字体的厚度
        tf = max(tl - 1, 1)  # font thickness
        # 获取标签的大小
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # 绘制矩形
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # 绘制标签
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


# 捕捉画面+目标检测+玩游戏
while True:
    # 设置暂停和取消暂停
    keys = key_check()
    # 如果键盘输入中包含 'P'
    if 'P' in keys:
        # 如果 action_cache 为空
        if not action_cache:
            pass
        # 如果 action_cache 不在 ["LEFT", "RIGHT", "UP", "DOWN"] 中
        elif action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
            # 释放对应键盘按键
            ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
            ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
            # 将 action_cache 置为 None
            action_cache = None
        # 如果 action_cache 在 ["LEFT", "RIGHT", "UP", "DOWN"] 中
        else:
            # 释放对应键盘按键
            ReleaseKey(direct_dic[action_cache])
            # 将 action_cache 置为 None
            action_cache = None
        # 如果程序处于暂停状态
        if paused:
            # 将 paused 置为 False
            paused = False
            # 程序休眠 1 秒
            time.sleep(1)
        # 如果程序不处于暂停状态
        else:
            # 将 paused 置为 True
            paused = True
            # 程序休眠 1 秒
            time.sleep(1)
```