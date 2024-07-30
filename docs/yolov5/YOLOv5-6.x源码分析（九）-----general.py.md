<!--yml
category: 游戏
date: 2023-09-17 14:43:57
-->

# YOLOv5-6.x源码分析（九）---- general.py

> 来源：[https://blog.csdn.net/weixin_51322383/article/details/130447757](https://blog.csdn.net/weixin_51322383/article/details/130447757)

### 文章目录

*   [前言](#_1)
*   [🚀YOLOv5-6.x源码分析（九）---- general.py](#YOLOv56x_generalpy_7)
*   *   [0\. 导包和基本配置](#0__8)
    *   [1\. set_logging](#1_set_logging_69)
    *   [2\. init_seeds](#2_init_seeds_92)
    *   [3\. get_latest_run](#3_get_latest_run_117)
    *   [4\. colorstr](#4_colorstr_133)
    *   [5\. make_divisible](#5_make_divisible_164)
    *   [6\. one_cycle](#6_one_cycle_176)
    *   [7\. labels_to_class_weights](#7__labels_to_class_weights_193)
    *   [8\. labels_to_image_weights](#8_labels_to_image_weights_233)
    *   [9\. clip_coords](#9_clip_coords_249)
    *   [10\. scale_coords](#10_font_colorred_scale_coords_font_269)
    *   [11\. non_max_suppression](#11_font_colorred_non_max_suppression_font_296)
    *   [12\. increment_path](#12_increment_path_442)
    *   [总结](#_474)

# 前言

**这个文件是YOLOv5的通用工具类，总共有1K多行的代码，非常庞大，我会列出大部分常用的，最主要的是要掌握NMS，这是后处理的核心代码。**

**导航**：[YOLOv5-6.x源码分析 全流程记录](https://blog.csdn.net/weixin_51322383/article/details/130353834?spm=1001.2014.3001.5502)

* * *

# 🚀YOLOv5-6.x源码分析（九）---- general.py

## 0\. 导包和基本配置

```py
import contextlib   # python上下文管理器   执行with…as…的时候调用contextlib
import glob         # 仅支持部分通配符的文件搜索模块
import inspect
import logging      # 日志模块
import math         # 数学公式
import os           # 操作系统交互
import platform     # 提供获取操作系统相关信息的模块
import random       # 随机数
import re           # 用来匹配字符串（动态、模糊）的模块
import shutil       # 文件操作模块
import signal       # 信号处理模块
import threading    # 多线程
import time         # 时间模块
import urllib       # 用于操作网页URL, 并对网页的内容进行抓取处理  如urllib.parse: 解析url
from datetime import datetime
from itertools import repeat     # 循环器模块  创建一个迭代器，重复生成object
from multiprocessing.pool import ThreadPool # 线程池
from pathlib import Path                    # Path将str转换为Path对象 使字符串路径易于操作的模块
from subprocess import check_output         # 创建一个子进程再命令行执行..., 最后返回执行结果(文件)
from typing import Optional
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
import pkg_resources as pkg # 查找
import torch        # pytorch框架
import torchvision  # pytorch辅助工具
import yaml     # yaml配置文件读写模块

from utils.downloads import gsutil_getsize
from utils.metrics import box_iou, fitness

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
RANK = int(os.getenv('RANK', -1))

# Settings
DATASETS_DIR = ROOT.parent / 'datasets'  # YOLOv5 datasets directory
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLOv5 multiprocessing threads
AUTOINSTALL = str(os.getenv('YOLOv5_AUTOINSTALL', True)).lower() == 'true'  # global auto-install mode
VERBOSE = str(os.getenv('YOLOv5_VERBOSE', True)).lower() == 'true'  # global verbose mode
FONT = 'Arial.ttf'  # https://ultralytics.com/assets/Arial.ttf

# 设置运行相关的一些基本的配置  Settings
# 控制print打印torch.tensor格式设置  tensor精度为5(小数点后5位)  每行字符数为320个  显示方法为long
torch.set_printoptions(linewidth=320, precision=5, profile='long')
# 控制print打印np.array格式设置  精度为5  每行字符数为320个  format short g, %precision=5
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
# pandas的最大显示行数是10
pd.options.display.max_columns = 10
# 阻止opencv参与多线程(与 Pytorch的 Dataloader不兼容)
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader) 使用一个线程
# 确定最大的线程数 这里被限制在了8
os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_THREADS)  # NumExpr max threads
os.environ['OMP_NUM_THREADS'] = '1' if platform.system() == 'darwin' else str(NUM_THREADS)  # OpenMP (PyTorch and SciPy) 
```

## 1\. set_logging

```py
# 设置日志保存 # 对日志的设置(format、level)等进行初始化
def set_logging(name=None, verbose=VERBOSE):
    # Sets level and returns logger
    # 先判断是否是kaggle环境
    if is_kaggle():
        for h in logging.root.handlers:
            logging.root.removeHandler(h)  # remove all handlers associated with the root logger object
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    # 设置日志级别  rank不为-1或0时设置输出级别level为WARN  为-1或0时设置级别为INFO
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    log = logging.getLogger(name)
    log.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(level)
    log.addHandler(handler) 
```

**这个函数一般用在`train.py`、`val.py`等文件的main函数第一步，进行日志等级、格式的初始化。**

## 2\. init_seeds

```py
# 初始化随机种子
def init_seeds(seed=0, deterministic=False):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    # cudnn.deterministic=True 可避免随机性
    if deterministic and check_version(torch.__version__, '1.12.0'):  # https://github.com/ultralytics/yolov5/pull/8213
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)           # 设置随机数 针对使用random.random()生成随机数的时候相同
    np.random.seed(seed)        # 设置随机数 针对使用np.random.rand()生成随机数的时候相同
    torch.manual_seed(seed)
    # cudnn.benchmark =True 随机模式
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe 
```

**这个函数是使用random.random()、np.random.rand()、init_torch_seeds（调用torch_utils.py中的函数）等生成一系列的随机数种子，以保证结果的可复现性。**

## 3\. get_latest_run

```py
# 获取最近训练的权重信息 last.pt  # 这个函数的作用是查找最近保存的权重文件 last*.pt，用以进行断点续训。
def get_latest_run(search_dir='.'):
    """用在train.py查找最近的pt文件进行断点续训
       用于返回该项目中最近的模型 'last.pt'对应的路径
       :params search_dir: 要搜索的文件的根目录 默认是 '.'  表示搜索该项目中的文件
    """
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)  # 为True会递归匹配路径
    return max(last_list, key=os.path.getctime) if last_list else '' 
```

这个函数的作用是获取最近训练的权重信息 last.pt ，以进行断点续训。用在`train.py`中。

## 4\. colorstr

```py
def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end'] 
```

**这个函数是将输出的开头和结尾加上颜色，使命令行输出显示会更加好看。**

## 5\. make_divisible

```py
def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor 
```

这个函数用来取大于等于x且是divisor的最小倍数，保证输入的x（一般是长宽）是算法的最大下采样率的倍数。

## 6\. one_cycle

```py
def one_cycle(y1=0.0, y2=1.0, steps=100):
    """用在train.py的学习率衰减策略模块
        one_cycle lr  lr先增加, 再减少, 再以更小的斜率减少
        论文: https://arxiv.org/pdf/1803.09820.pdf
        """
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1 
```

**这个函数是一种特殊的学习率衰减策略，在train.py的学习率衰减策略模块中使用。（余弦退火学习率）**

论文： [one_cycle](https://arxiv.org/pdf/1803.09820.pdf)

![在这里插入图片描述](img/1136b2ac7db9754e681744f8cf78abe0.png)

## 7\. labels_to_class_weights

```py
def labels_to_class_weights(labels, nc=80):
    """用在train.py中  得到每个类别的权重   标签频率高的类权重低
        从训练(gt)标签获得每个类的权重  标签频率高的类权重低
        Get class weights (inverse frequency) from training labels
        :params labels: gt框的所有真实标签labels
        :params nc: 数据集的类别数
        :return torch.from_numpy(weights): 每一个类别根据labels得到的占比(次数越多权重越小) tensor
        """
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    # classes: 所有标签对应的类别labels   labels[:, 0]: 类别   .astype(np.int): 取整
    classes = labels[:, 0].astype(int)  # labels = [class xywh]
    # weight: 返回每个类别出现的次数 [1, nc]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    # 将出现次数为0的类别权重全部取1
    weights[weights == 0] = 1  # replace empty bins with 1
    # 其他所有的类别的权重全部取次数的倒数  number of targets per class
    weights = 1 / weights  # number of targets per class
    # normalize 求出每一类别的占比
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights).float() 
```

**这个函数就是堆labels中的每一个类别求一个权重，标签频率越高的，权重越低。如果某个类别数量为0，则置为1**

比如说：

classes为[2 1 3 4 4 3]，weights为[0 1 1 2 2]，即将每个类别的数量作为初始化权重，接下来将所有的0置换为1，weights为[1 1 1 2 2]，再取倒数[1 1 1 0.5 0.5]，归一化权重并返回。

## 8\. labels_to_image_weights

```py
def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class_weights and image contents
    # Usage: index = random.choices(range(n), weights=image_weights, k=1)  # weighted image sample
    # [80] -> [1, 80]
    # 整个数据集的每个类别权重[1, 80] *  每张图片的每个类别出现的次数[num_labels, 80] = 得到每一张图片每个类对应的权重[128, 80]
    # 另外注意: 这里不是矩阵相乘, 是元素相乘 [1, 80] 和每一行图片的每个类别出现的次数 [1, 80] 分别按元素相乘
    # 再sum(1): 按行相加  得到最终image_weights: 得到每一张图片对应的采样权重[128]
    class_counts = np.array([np.bincount(x[:, 0].astype(int), minlength=nc) for x in labels])
    return (class_weights.reshape(1, nc) * class_counts).sum(1) 
```

这个函数是利用每张图片真实gt框的真实标签labels和上一步labels_to_class_weights得到的每个类别的权重得到数据集中每张图片对应的权重。

## 9\. clip_coords

```py
def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        # .clamp_(min, max): 将取整限定在(min, max)之间, 超出这个范围自动划到边界上
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)  超出这个范围自动划到边界上
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2 
```

**这个函数的作用是：将boxes的坐标(x1y1x2y2 左上角右下角)限定在图像的尺寸(img_shape hw)内，防止出界。**

**这个函数会用在下面的xyxy2xywhn、save_one_boxd等函数中，很重要，必须掌握。**

## 10\. scale_coords

```py
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    # ratio_pad为空就先算放缩比例gain和pad值 calculate from img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        # gain  = old / new  取高宽缩放比例中较小的,之后还可以再pad  如果直接取大的, 裁剪就可能减去目标
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        # wh padding  wh中有一个为0  主要是pad另一个
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    # 防止放缩后的坐标过界 边界处直接剪切
    clip_coords(coords, img0_shape)
    return coords 
```

**这个函数是将坐标coords(x1y1x2y2)从img1_shape尺寸缩回到img0_shape尺寸。x的正坐标是向右，y的正坐标是向下。这个函数也是很重要的。**

**先将横纵坐标减去偏移量，再统一除以gain获取新的值，最后再限定在img0的大小内。**

## 11\. non_max_suppression

NMS，非极大值抑制，是目标检测领域最基本的算法操作了。这篇博客，其他的函数可以都只是看看，但这个函数必须必须必须掌握！

**这里借另一位博主的一张整体流程图：**

![在这里插入图片描述](img/b4b38a085256b46dcc77dbc566e393e5.png)

在推理阶段，后处理我们会用到四个函数，分别是

*   `letterbox`：将图片缩放到指定大小
*   `NMS`：去除多余的框
*   `scale_coords`：将图片还原回原大小
*   `draw_box`：将预测框画出来

而NMS的操作流程是：

![在这里插入图片描述](img/f501eadfc1ef3798f3de50545c21a21c.png)

步骤一：将所有矩形框按照不同的类别标签分组，组内按照置信度高低得分进行排序；
步骤二：将步骤一中得分最高的矩形框拿出来，遍历剩余矩形框，计算与当前得分最高的矩形框的交并比，将剩余矩形框中大于设定的IOU阈值的框删除；
步骤三：将步骤二结果中，对剩余的矩形框重复步骤二操作，直到处理完所有矩形框；

```py
def non_max_suppression(prediction,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        classes=None,
                        agnostic=False, # 进行nms是否也去除不同类别之间的框 默认False
                        multi_label=False,
                        labels=(),
                        max_det=300):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes
    Params:
        prediction: [batch, num_anchors(3个yolo预测层), (x+y+w+h+1+num_classes)] = [1, 18900, 25]  3个anchor的预测结果总和
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates 选出大于阈值的框

    # Checks 检查传入的conf_thres和iou_thres两个阈值是否符合范围
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings 设置一些变量
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height  预测物体宽度和高度的大小范围
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms() # 每个图像最多检测物体的个数
    time_limit = 0.3 + 0.03 * bs  # seconds to quit after       nms执行时间阈值 超过这个时间就退出了
    redundant = True  # require redundant detections        是否需要冗余的detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS  use merge-NMS 多个bounding box给它们一个权重进行融合  默认False

    t = time.time()     # 当前时间
    # 存放最终筛选后的预测框结果
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference 遍历所有框
        # Apply constraints
        # 第一层过滤 虑除超小anchor标和超大anchor   x=[18900, 25]
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height

        # 第二层过滤 根据conf_thres虑除背景目标(obj_conf<conf_thres 0.1的目标 置信度极低的目标)  x=[59, 25]
        x = x[xc[xi]]  # confidence

        # {list: bs} 第一张图片的target[17, 5] 第二张[1, 5] 第三张[7, 5] 第四张[6, 5]
        # Cat apriori labels if autolabelling   自动标注label时调用  一般不用
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image  经过前两层过滤后如果该feature map没有目标框了，就结束这轮直接进行下一张图
        if not x.shape[0]:
            continue

        # Compute conf 计算conf_score
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)   左上角 右下角   [59, 4]
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            # 第三轮过滤:针对每个类别score(obj_conf * cls_conf) > conf_thres    [59, 6] -> [51, 6]
            # 这里一个框是有可能有多个物体的，所以要筛选
            # nonzero: 获得矩阵中的非0(True)数据的下标  a.t(): 将a矩阵拆开
            # i: 下标 [43]   j: 类别index [43] 过滤了两个score太低的
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T  # as_tuple = False： 输出的每一行为非零元素的索引
            # pred = [43, xyxy+score+class] [43, 6]
            # unsqueeze(1): [43] => [43, 1] add batch dimension
            # box[i]: [43,4] xyxy
            # pred[i, j + 5].unsqueeze(1): [43,1] score  对每个i,取第（j+5）个位置的值（第j个class的值cla_conf）
            # j.float().unsqueeze(1): [43,1] class
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True) # 一个类别直接取分数最大类的即可
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class   是否只保留特定的类别  默认None  不执行这里
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes 如果经过第三轮过滤该feature map没有目标框了，就结束这轮直接进行下一张图
            continue
        elif n > max_nms:  # excess boxes 如果经过第三轮过滤该feature map还有很多框(>max_nms)   就需要排序
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence 置信度排序

        # 第4轮过滤 Batched NMS [51, 6] -> [5, 6]
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # 做个切片 得到boxes和scores   不同类别的box位置信息加上一个很大的数但又不同的数c
        # 这样作非极大抑制的时候不同类别的框就不会掺和到一块了  这是一个作nms挺巧妙的技巧
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # 返回nms过滤后的bounding box(boxes)的索引（降序排列）
        # i=tensor([18, 19, 32, 25, 27])   nms后只剩下5个预测框了
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS    返回值：keep：NMS过滤后的bounding box索引（降序） 去除冗余的框
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            # bounding box合并  其实就是把权重和框相乘再除以权重之和
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]   # 最终输出   [5, 6]
        # 看下时间超没超时  超时没做完的就不做了
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output 
```

## 12\. increment_path

```py
def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic  # string/win路径 -> win路径

    # 如果该文件夹已经存在 则将路径run/train/exp修改为 runs/train/exp1
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')    # with_suffix:更改路径后缀 suffix文件后缀

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path 
```

**用于递增路径。比如我输入路径是run/train/exp，但是发现文件夹里面已经有这个文件了，那么就将文件路径扩展围为：runs/train/exp{sep}0, runs/exp{sep}1 etc。**

## 总结

**这个脚本文件总共有1K多行，我没有全部写完，我只挑了部分重点的来写（偷懒了），重点掌握[NMS](#11_font_colorred_non_max_suppression_font_296)和[scale_coords](#10_font_colorred_scale_coords_font_269)，这是后处理中极其重要的函数。**

**References**

> CSDN xjunjin： [2021SC@SDUSC山东大学软件学院软件工程应用与实践–YOLOV5代码分析（五）general.py-3](https://blog.csdn.net/xjunjin/article/details/120828464)
> 
> CSDN 猫猫与橙子：[YOLOV5目标检测-后处理NMS(非极大值抑制)](