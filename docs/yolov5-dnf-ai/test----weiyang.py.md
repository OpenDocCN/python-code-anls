# `yolov5-DNF\test\weiyang.py`

```py
# 导入需要的库
import cv2
import numpy as np
import torch
from directkeys import key_press
from grabscreen import grab_screen
# 从自定义模块中导入需要的函数和类
from models.experimental import attempt_load
from utils.general import non_max_suppression, xyxy2xywh

# 目标类别的名称列表
names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

# 1-8 自动爬楼

# 模型权重文件的路径
weights = 'best.pt'  # yolo5 模型存放的位置
# 根据是否有GPU选择设备
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# 是否使用半精度浮点数
half = device.type != 'cpu'
# NMS的置信度过滤阈值
conf_thres = 0.3  
# NMS的IOU阈值
iou_thres = 0.2  
# 目标类别
classes = None
# 不同类别的NMS时也参数过滤
agnostic_nms = False  
# 加载模型
model = attempt_load(weights, device)

# 从屏幕截取图像
img0 = grab_screen((0, 0, 1280, 800))

# 将图像从BGRA转换为BGR
img = cv2.cvtColor(img0, cv2.COLOR_BGRA2BGR)
# 调整图像通道顺序
img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
# 转换为连续数组
img = np.ascontiguousarray(img)
# 将图像转换为PyTorch张量，并移动到指定设备上
img = torch.from_numpy(img).to(device).unsqueeze(dim=0)
# 如果使用半精度浮点数，则转换为半精度
img = img.half() if half else img.float()  # uint8 to fp16/32~
# 将图像像素值归一化到0-1之间
img /= 255.0  # 0 - 255 to 0.0 - 1.0

# 模型推理，获取预测结果
pred = model(img, augment=False)[0]
# 应用NMS进行目标框筛选
det = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
det = det[0]

if det is not None and len(det):
    # 如果检测结果不为空
    img_object = []
    cls_object = []
    # 初始化空列表用于存储检测到的物体的位置和分类
    for idx, (*xyxy, conf, cls) in enumerate(reversed(det)):
        # 遍历检测结果，获取位置、置信度和分类信息
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
        # 将位置信息转换为xywh格式，并转换为列表
        cls = int(cls)
        img_object.append(xywh)  # 存储位置信息
        cls_object.append(names[cls])  # 存储分类信息
        if names[cls] == 'giraffe':
            hero_index = idx
        # 如果检测到的物体是长颈鹿，则记录其索引
    thx = 30  # 捡东西时，x方向的阈值
    thy = 30  # 捡东西时，y方向的阈值
    attx = 150  # 攻击时，x方向的阈值
    atty = 50  # 攻击时，y方向的阈值
    skillDis = 800
    skillDis = 400
    # 初始化一些阈值和参数
    if 'giraffe' in cls_object:
        hero_xywh = img_object[hero_index]
    # 如果检测到的物体中包含长颈鹿，则记录长颈鹿的位置信息
    if 'orange' in cls_object:
        # 如果检测到的物体中包含橙色物体
        for idx, (c, box) in enumerate(zip(cls_object, img_object)):
            # 遍历检测到的物体的分类和位置信息
            dis = ((hero_xywh[0] - box[0]) ** 2 + (hero_xywh[1] - box[1]) ** 2) ** 0.5
            # 计算长颈鹿与其他物体的距离
    elif 'material' in cls_object:
        if 'option' in cls_object:
            # 如果检测到的物体中包含材料和选项
            # 聚物捡东西  f2  xxx
        else:
            # 如果检测到的物体中包含材料但不包含选项
            # 普通捡东西
    elif 'boss' in cls_object:
        # 如果检测到的物体中包含boss
        key_press('g')
        # 按下键盘上的'g'键，进行相应操作
    # 如果在 cls_object 中出现了 'sinan'，则执行以下操作
    # 下一局出现思南选择界面，选择1-8阶随机一个思南，鼠标移动并点击
    # 中场清理背包
    # 如果1-8在cls_object中：
    #     进入下一局游戏
    # 否则：
    #     关闭游戏并通知爬楼完成，需要补货，清理背包，然后返回
```