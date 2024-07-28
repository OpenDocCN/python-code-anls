# `.\comic-translate\modules\ocr\pororo\pororo\models\brainOCR\detection.py`

```py
# 从 collections 模块导入 OrderedDict，用于保持字典添加顺序
from collections import OrderedDict

# 导入必要的库和模块
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

# 从当前包中导入相关模块和函数
from .craft import CRAFT
from .craft_utils import adjust_result_coordinates, get_det_boxes
from .imgproc import normalize_mean_variance, resize_aspect_ratio


# 函数：复制给定状态字典的内容
def copy_state_dict(state_dict):
    # 判断是否以 "module" 开头，确定起始索引
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    # 重新构造不带前缀的新状态字典
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


# 函数：对给定图像进行文本检测
def test_net(image: np.ndarray, net, opt2val: dict):
    # 从参数字典中获取必要的配置参数
    canvas_size = opt2val["canvas_size"]
    mag_ratio = opt2val["mag_ratio"]
    text_threshold = opt2val["text_threshold"]
    link_threshold = opt2val["link_threshold"]
    low_text = opt2val["low_text"]
    device = opt2val["device"]

    # 将图像按照指定的画布大小进行调整
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio  # 计算高宽比例

    # 图像预处理：归一化处理
    x = normalize_mean_variance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # 调整维度顺序 [h, w, c] 到 [c, h, w]
    x = Variable(x.unsqueeze(0))  # 扩展维度 [c, h, w] 到 [b, c, h, w]
    x = x.to(device)  # 将数据传输到指定设备

    # 执行模型的前向传播
    with torch.no_grad():
        y, feature = net(x)

    # 生成得分图和链接图
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # 后处理：获取文本检测框和多边形
    boxes, polys = get_det_boxes(
        score_text,
        score_link,
        text_threshold,
        link_threshold,
        low_text,
    )

    # 调整坐标信息
    boxes = adjust_result_coordinates(boxes, ratio_w, ratio_h)
    polys = adjust_result_coordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    return boxes, polys


# 函数：加载文本检测模型并返回该模型
def get_detector(det_model_ckpt_fp: str, device: str = "cpu"):
    net = CRAFT()  # 初始化 CRAFT 模型对象

    # 加载模型的状态字典，并根据设备情况选择加载位置
    net.load_state_dict(
        copy_state_dict(torch.load(det_model_ckpt_fp, map_location=device)))
    if device == "cuda":  # 如果设备为 CUDA
        net = torch.nn.DataParallel(net).to(device)  # 使用 DataParallel 将模型加载到 CUDA
        cudnn.benchmark = False  # 禁用 cudnn 自动寻找最适合网络配置的算法

    net.eval()  # 设置模型为评估模式
    return net  # 返回加载后的模型对象


# 函数：获取图像中的文本框信息
def get_textbox(detector, image: np.ndarray, opt2val: dict):
    # 使用文本检测器获取文本框和多边形
    bboxes, polys = test_net(image, detector, opt2val)
    result = []
    # 将多边形信息转换为整数类型，并添加到结果列表中
    for i, box in enumerate(polys):
        poly = np.array(box).astype(np.int32).reshape((-1))
        result.append(poly)

    return result  # 返回最终的文本框信息列表
```