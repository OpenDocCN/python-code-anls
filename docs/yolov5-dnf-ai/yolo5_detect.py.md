# `yolov5-DNF\yolo5_detect.py`

```py
# 导入所需的库
import random
import time

import cv2
import numpy as np
import torch

# 导入自定义的函数和类
from getkeys import key_check
from grabscreen import grab_screen
from models.experimental import attempt_load
from utils.general import (
    non_max_suppression, scale_coords,
    plot_one_box)
from utils.torch_utils import time_synchronized

# 定义一个函数，用于将图像调整为指定大小的矩形
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    # 获取图像的当前形状（高度和宽度）
    shape = img.shape[:2]  # current shape [height, width]
    # 如果新形状是一个整数，则将其转换为元组
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 计算缩放比例（新的形状 / 旧的形状）
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # 如果不允许放大，则将缩放比例限制在1.0以内
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # 计算填充值
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # 如果需要自动调整大小，则进行最小矩形调整
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    # 如果需要填充整个矩形，则进行拉伸调整
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    # 将填充值分成两部分
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    # 如果图像形状与新的未填充形状不同，则进行调整大小
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    # 添加边框
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

# 模型权重文件的路径
weights = r'E:\Computer_vision\yolov5\YOLO5\yolov5-master\runs\exp0\weights\best.pt'
# 根据是否有GPU选择设备
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# 尝试加载权重到模型中，使用设备的映射位置
model = attempt_load(weights, map_location=device)  # load FP32 model
# 设置窗口大小
window_size = (0,0,1200,750)
# 记录当前时间
last_time = time.time()
# 倒计时5秒
for i in list(range(5))[::-1]:
    print(i + 1)
    time.sleep(1)
# 设置图像大小
img_size = 608
# 暂停标志
paused = False
# 判断设备是否为CPU
half = device.type != 'cpu'
# 是否查看图像
view_img = True
# 是否保存文本
save_txt = False
# 置信度阈值
conf_thres = 0.3
# IOU阈值
iou_thres = 0.2
# 类别
classes = None
# 是否使用不考虑类别的NMS
agnostic_nms = True
# 类别名称
names = ['hero', 'small_map', "monster", 'money', 'material', 'door', 'BOSS', 'box', 'options']
# 随机颜色
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
# 如果是半精度模型，则转换为FP16
if half:
    model.half()  # to FP16

# 进入循环
while (True):
```