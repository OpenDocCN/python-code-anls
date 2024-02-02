# `yolov5-DNF\utils\general.py`

```py
# 导入所需的库
import glob  # 用于查找文件路径模式匹配
import logging  # 用于记录日志信息
import math  # 用于数学运算
import os  # 用于操作系统相关功能
import platform  # 用于获取操作系统的信息
import random  # 用于生成随机数
import re  # 用于正则表达式匹配
import shutil  # 用于文件操作
import subprocess  # 用于执行外部命令
import time  # 用于时间相关功能
from contextlib import contextmanager  # 用于创建上下文管理器
from copy import copy  # 用于复制对象
from pathlib import Path  # 用于操作文件路径

import cv2  # OpenCV库，用于图像处理
import matplotlib  # 用于绘图
import matplotlib.pyplot as plt  # 用于绘图
import numpy as np  # 用于数值计算
import torch  # PyTorch深度学习库
import torch.nn as nn  # PyTorch中的神经网络模块
import yaml  # 用于读写YAML文件
from scipy.cluster.vq import kmeans  # 用于K均值聚类
from scipy.signal import butter, filtfilt  # 用于信号处理
from tqdm import tqdm  # 用于显示进度条
from utils.google_utils import gsutil_getsize  # 自定义的Google工具函数

from utils.torch_utils import is_parallel, init_torch_seeds  # 自定义的PyTorch工具函数

# 设置打印选项
torch.set_printoptions(linewidth=320, precision=5, profile='long')  # 设置PyTorch打印选项
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # 设置NumPy打印选项
matplotlib.rc('font', **{'size': 11})  # 设置matplotlib字体大小

# 防止OpenCV多线程（以使用PyTorch DataLoader）
cv2.setNumThreads(0)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()  # 如果不是主进程，则等待所有进程到达此处
    yield  # 返回上下文管理器的值
    if local_rank == 0:
        torch.distributed.barrier()  # 如果是主进程，则等待所有进程到达此处


def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)  # 配置日志记录格式和级别


def init_seeds(seed=0):
    random.seed(seed)  # 设置随机数种子
    np.random.seed(seed)  # 设置NumPy随机数种子
    init_torch_seeds(seed)  # 设置PyTorch随机数种子


def get_latest_run(search_dir='./runs'):
    # 返回/runs目录中最近的'last.pt'文件的路径（用于--resume）
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)  # 查找/runs目录下所有最近的'last.pt'文件
    return max(last_list, key=os.path.getctime) if last_list else ''  # 返回最近的'last.pt'文件路径，如果没有则返回空字符串


def check_git_status():
    # 如果仓库过时，建议执行'git pull'
    # 检查操作系统是否为 Linux 或 Darwin，并且当前环境不是在 Docker 容器中
    if platform.system() in ['Linux', 'Darwin'] and not os.path.isfile('/.dockerenv'):
        # 使用 subprocess 模块执行 shell 命令，检查 git 仓库状态并获取输出
        s = subprocess.check_output('if [ -d .git ]; then git fetch && git status -uno; fi', shell=True).decode('utf-8')
        # 如果输出中包含 'Your branch is behind'，则打印该信息
        if 'Your branch is behind' in s:
            print(s[s.find('Your branch is behind'):s.find('\n\n')] + '\n')
def check_img_size(img_size, s=32):
    # 验证 img_size 是否是步长 s 的倍数
    new_size = make_divisible(img_size, int(s))  # 向上取整，使其成为 s 的倍数
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


def check_anchors(dataset, model, thr=4.0, imgsz=640):
    # 检查锚点是否适合数据，必要时重新计算
    print('\nAnalyzing anchors... ', end='')
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect()
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # 增广比例
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # 宽高

    def metric(k):  # 计算度量
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # 比率度量
        best = x.max(1)[0]  # 最佳比率
        aat = (x > 1. / thr).float().sum(1).mean()  # 超过阈值的锚点
        bpr = (best > 1. / thr).float().mean()  # 最佳可能召回率
        return bpr, aat

    bpr, aat = metric(m.anchor_grid.clone().cpu().view(-1, 2))
    print('anchors/target = %.2f, Best Possible Recall (BPR) = %.4f' % (aat, bpr), end='')
    # 如果平均最佳预测召回率小于0.98，则重新计算
    if bpr < 0.98:  # threshold to recompute
        # 打印提示信息，尝试生成改进的锚点，请等待...
        print('. Attempting to generate improved anchors, please wait...' % bpr)
        # 计算锚点的数量
        na = m.anchor_grid.numel() // 2  # number of anchors
        # 通过 k 均值算法生成新的锚点
        new_anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        # 计算新锚点的平均最佳预测召回率
        new_bpr = metric(new_anchors.reshape(-1, 2))[0]
        # 如果新锚点的平均最佳预测召回率大于原来的，则替换锚点
        if new_bpr > bpr:  # replace anchors
            # 将新锚点转换为与模型锚点相同的数据类型和设备
            new_anchors = torch.tensor(new_anchors, device=m.anchors.device).type_as(m.anchors)
            # 更新模型的锚点网格用于推理
            m.anchor_grid[:] = new_anchors.clone().view_as(m.anchor_grid)  # for inference
            # 更新模型的锚点用于损失计算
            m.anchors[:] = new_anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # loss
            # 检查锚点的顺序
            check_anchor_order(m)
            # 打印提示信息，新锚点已保存到模型，更新模型 *.yaml 文件以在将来使用这些锚点
            print('New anchors saved to model. Update model *.yaml to use these anchors in the future.')
        else:
            # 打印提示信息，原始锚点比新锚点更好，继续使用原始锚点
            print('Original anchors better than new anchors. Proceeding with original anchors.')
    # 打印空行
    print('')  # newline
# 检查 YOLOv5 Detect() 模块 m 的锚点顺序是否与步长顺序一致，如果不一致则进行修正
def check_anchor_order(m):
    # 计算锚点面积
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    # 计算面积差
    da = a[-1] - a[0]  # delta a
    # 计算步长差
    ds = m.stride[-1] - m.stride[0]  # delta s
    # 如果面积差和步长差的符号不一致，则进行修正
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        # 反转锚点顺序
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)

# 检查文件是否存在，如果不存在则搜索文件
def check_file(file):
    # 如果文件存在或者文件名为空，则直接返回文件名
    if os.path.isfile(file) or file == '':
        return file
    else:
        # 在当前目录及子目录中搜索文件
        files = glob.glob('./**/' + file, recursive=True)  # find file
        # 断言找到了文件
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        # 断言找到的文件是唯一的
        assert len(files) == 1, "Multiple files match '%s', specify exact path: %s" % (file, files)  # assert unique
        # 返回找到的文件
        return files[0]  # return file

# 检查数据集是否存在，如果不存在则下载数据集
def check_dataset(dict):
    # 获取数据集的验证集和下载链接
    val, s = dict.get('val'), dict.get('download')
    # 如果验证集存在且不为空
    if val and len(val):
        # 将验证集路径转换为绝对路径
        val = [os.path.abspath(x) for x in (val if isinstance(val, list) else [val])]  # val path
        # 如果验证集中有不存在的路径
        if not all(os.path.exists(x) for x in val):
            # 打印警告信息
            print('\nWARNING: Dataset not found, nonexistent paths: %s' % [*val])
            # 如果存在下载链接
            if s and len(s):  # download script
                # 打印下载信息
                print('Downloading %s ...' % s)
                # 如果下载链接是以 http 开头且以 .zip 结尾
                if s.startswith('http') and s.endswith('.zip'):  # URL
                    # 获取文件名
                    f = Path(s).name  # filename
                    # 使用 torch.hub 下载文件
                    torch.hub.download_url_to_file(s, f)
                    # 解压文件
                    r = os.system('unzip -q %s -d ../ && rm %s' % (f, f))  # unzip
                else:  # bash script
                    # 执行下载脚本
                    r = os.system(s)
                # 打印数据集自动下载结果
                print('Dataset autodownload %s\n' % ('success' if r == 0 else 'failure'))  # analyze return value
            else:
                # 抛出异常，数据集未找到
                raise Exception('Dataset not found.')

# 返回可以被除数整除的最小整数
def make_divisible(x, divisor):
    # 返回 x 被 divisor 整除的最小整数
    return math.ceil(x / divisor) * divisor
def labels_to_class_weights(labels, nc=80):
    # 从训练标签中获取类别权重（倒数频率）
    if labels[0] is None:  # 如果没有加载标签
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # 提取类别信息，labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # 每个类别的出现次数

    # 在前面添加网格点数（用于 uCE 训练）
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # 每张图像的网格点数
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # 在开头添加网格点数

    weights[weights == 0] = 1  # 将空的类别替换为1
    weights = 1 / weights  # 每个类别的目标数量
    weights /= weights.sum()  # 归一化
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # 基于类别 mAPs 生成图像权重
    n = len(labels)
    class_counts = np.array([np.bincount(labels[i][:, 0].astype(np.int), minlength=nc) for i in range(n)])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # 加权图像采样
    return image_weights


def coco80_to_coco91_class():  # 将80索引（val2014）转换为91索引（paper）
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    # 创建一个包含数字1到90的列表
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    # 返回列表x
    return x
# 将包含 [x1, y1, x2, y2] 的 nx4 的坐标框转换为 [x, y, w, h] 的格式，其中 xy1 为左上角，xy2 为右下角
def xyxy2xywh(x):
    # 如果输入是 torch.Tensor 类型，则创建一个与 x 相同大小的全零张量，否则创建一个与 x 相同大小的全零数组
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    # 计算 x 中每个框的中心 x 坐标
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    # 计算 x 中每个框的中心 y 坐标
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    # 计算 x 中每个框的宽度
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    # 计算 x 中每个框的高度
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


# 将包含 [x, y, w, h] 的 nx4 的坐标框转换为 [x1, y1, x2, y2] 的格式，其中 xy1 为左上角，xy2 为右下角
def xywh2xyxy(x):
    # 如果输入是 torch.Tensor 类型，则创建一个与 x 相同大小的全零张量，否则创建一个与 x 相同大小的全零数组
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    # 计算 x 中每个框的左上角 x 坐标
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    # 计算 x 中每个框的左上角 y 坐标
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    # 计算 x 中每个框的右下角 x 坐标
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    # 计算 x 中每个框的右下角 y 坐标
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


# 将坐标框从 img1_shape 缩放到 img0_shape
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # 如果 ratio_pad 为 None，则从 img0_shape 计算
    if ratio_pad is None:  # calculate from img0_shape
        # 计算缩放比例
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        # 计算填充
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        # 从 ratio_pad 中获取缩放比例
        gain = ratio_pad[0][0]
        # 从 ratio_pad 中获取填充
        pad = ratio_pad[1]

    # 减去 x 填充
    coords[:, [0, 2]] -= pad[0]  # x padding
    # 减去 y 填充
    coords[:, [1, 3]] -= pad[1]  # y padding
    # 坐标框缩放
    coords[:, :4] /= gain
    # 裁剪坐标框
    clip_coords(coords, img0_shape)
    return coords


# 裁剪坐标框到图像形状 (height, width)
def clip_coords(boxes, img_shape):
    # 裁剪 x1
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    # 裁剪 y1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    # 裁剪 x2
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    # 裁剪 y2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


# 计算每个类别的平均精度
def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, fname='precision-recall_curve.png'):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).  # 真正例（nparray，nx1或nx10）
        conf:  Objectness value from 0-1 (nparray).  # 目标性值从0到1（nparray）
        pred_cls:  Predicted object classes (nparray).  # 预测的目标类别（nparray）
        target_cls:  True object classes (nparray).  # 真实的目标类别（nparray）
        plot:  Plot precision-recall curve at mAP@0.5  # 在mAP@0.5处绘制精度-召回曲线
        fname:  Plot filename  # 绘图文件名
    # Returns
        The average precision as computed in py-faster-rcnn.  # 在py-faster-rcnn中计算的平均精度
    """

    # Sort by objectness  # 按目标性值排序
    i = np.argsort(-conf)  # 对目标性值进行降序排序
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]  # 根据排序结果重新排列tp、conf和pred_cls

    # Find unique classes  # 查找唯一的类别
    unique_classes = np.unique(target_cls)  # 找到目标类别中的唯一值

    # Create Precision-Recall curve and compute AP for each class  # 创建精度-召回曲线并计算每个类别的AP
    px, py = np.linspace(0, 1, 1000), []  # for plotting  # 用于绘图
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898  # 评估P和R的分数
    s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)  # 类别数，iou阈值数
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)  # 初始化ap、p和r数组
    for ci, c in enumerate(unique_classes):  # 遍历唯一的类别
        i = pred_cls == c  # 找到预测类别等于当前类别的索引
        n_gt = (target_cls == c).sum()  # Number of ground truth objects  # 真实对象的数量
        n_p = i.sum()  # Number of predicted objects  # 预测对象的数量

        if n_p == 0 or n_gt == 0:  # 如果预测对象或真实对象的数量为0
            continue  # 继续下一次循环
        else:
            # Accumulate FPs and TPs  # 累积FP和TP
            fpc = (1 - tp[i]).cumsum(0)  # 累积FP
            tpc = tp[i].cumsum(0)  # 累积TP

            # Recall  # 召回率
            recall = tpc / (n_gt + 1e-16)  # recall curve  # 召回率曲线
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases  # 在pr_score处的r

            # Precision  # 精度
            precision = tpc / (tpc + fpc)  # precision curve  # 精度曲线
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score  # 在pr_score处的p

            # AP from recall-precision curve  # 从召回率-精度曲线计算AP
            py.append(np.interp(px, recall[:, 0], precision[:, 0]))  # precision at mAP@0.5  # mAP@0.5处的精度
            for j in range(tp.shape[1]):  # 遍历iou阈值
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])  # 计算AP
    # 计算 F1 分数（精确率和召回率的调和平均值）
    f1 = 2 * p * r / (p + r + 1e-16)

    # 如果需要绘图
    if plot:
        # 将 py 中的数组沿着新的轴堆叠
        py = np.stack(py, axis=1)
        # 创建一个新的图形和一个子图
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # 绘制曲线（横轴为 recall，纵轴为 precision）
        ax.plot(px, py, linewidth=0.5, color='grey')
        # 绘制曲线（横轴为 recall，纵轴为所有类别的平均 precision）
        ax.plot(px, py.mean(1), linewidth=2, color='blue', label='all classes')
        # 设置 x 轴标签
        ax.set_xlabel('Recall')
        # 设置 y 轴标签
        ax.set_ylabel('Precision')
        # 设置 x 轴范围
        ax.set_xlim(0, 1)
        # 设置 y 轴范围
        ax.set_ylim(0, 1)
        # 添加图例
        plt.legend()
        # 调整子图布局
        fig.tight_layout()
        # 保存图形到文件
        fig.savefig(fname, dpi=200)

    # 返回精确率、召回率、平均精确率、F1 分数、唯一类别的整数形式
    return p, r, ap, f1, unique_classes.astype('int32')
def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    # 计算第一个边界框的宽度和高度
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    # 计算第二个边界框的宽度和高度
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    # 计算两个边界框的交集面积
    union = w1 * h1 + w2 * h2 - inter + eps

    # 计算交并比
    iou = inter / union
    # 如果是 GIoU 或者 DIoU 或者 CIoU
    if GIoU or DIoU or CIoU:
        # 计算凸多边形的宽度
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        # 计算凸多边形的高度
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        # 如果是 CIoU 或者 DIoU
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # 计算凸多边形的对角线的平方
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            # 计算中心点之间的距离的平方
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            # 如果是 DIoU
            if DIoU:
                return iou - rho2 / c2  # DIoU
            # 如果是 CIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                # 计算参数 v
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                # 计算参数 alpha
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                # 返回 CIoU
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        # 如果是 GIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            # 计算凸多边形的面积
            c_area = cw * ch + eps  # convex area
            # 返回 GIoU
            return iou - (c_area - union) / c_area  # GIoU
    # 如果不是 GIoU 或者 DIoU 或者 CIoU
    else:
        # 返回 IoU
        return iou  # IoU
def box_iou(box1, box2):
    # 计算两个框的交并比（Jaccard指数）
    # 期望输入的框格式为 (x1, y1, x2, y2)
    # 参数：
    #   box1 (Tensor[N, 4])
    #   box2 (Tensor[M, 4])
    # 返回：
    #   iou (Tensor[N, M]): 包含每个框在boxes1和boxes2中每个元素的两两IoU值的NxM矩阵

    def box_area(box):
        # 计算框的面积
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # 计算交集
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # 计算IoU

def wh_iou(wh1, wh2):
    # 返回nxm的IoU矩阵，wh1是nx2，wh2是mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # 计算IoU

class FocalLoss(nn.Module):
    # 将focal loss包装在现有的loss_fcn()周围，例如 criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # 必须是nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # 必须将reduction设置为'none'，以便将FL应用于每个元素
    # 定义一个前向传播函数，计算损失值
    def forward(self, pred, true):
        # 使用损失函数计算预测值和真实值之间的损失
        loss = self.loss_fcn(pred, true)
        
        # 计算 p_t，用于提高梯度稳定性
        pred_prob = torch.sigmoid(pred)  # 将预测值转换为概率
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)  # 根据预测概率计算 p_t
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)  # 计算 alpha 调整因子
        modulating_factor = (1.0 - p_t) ** self.gamma  # 计算调制因子
        loss *= alpha_factor * modulating_factor  # 应用 alpha 调整因子和调制因子到损失值上

        # 根据指定的 reduction 方式返回损失值
        if self.reduction == 'mean':
            return loss.mean()  # 返回损失值的平均值
        elif self.reduction == 'sum':
            return loss.sum()  # 返回损失值的总和
        else:  # 'none'
            return loss  # 返回原始的损失值
def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # 返回正面和负面标签平滑的二元交叉熵损失函数的目标值
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # 使用减少缺失标签效应的BCEwithLogitLoss()
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # 必须是nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # 从logits计算概率
        dx = pred - true  # 仅减少缺失标签效应
        # dx = (pred - true).abs()  # 减少缺失标签和错误标签效应
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


def compute_loss(p, targets, model):  # 预测值，目标值，模型
    device = targets.device
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    tcls, tbox, indices, anchors = build_targets(p, targets, model)  # 目标
    h = model.hyp  # 超参数

    # 定义标准
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['cls_pw']])).to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['obj_pw']])).to(device)

    # 类别标签平滑 https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # Focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # 损失
    nt = 0  # 目标数量
    np = len(p)  # 输出数量
    balance = [4.0, 1.0, 0.4] if np == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    # 遍历列表 p 中的元素，同时获取索引值 i 和元素值 pi
    for i, pi in enumerate(p):  # layer index, layer predictions
        # 解包元组 indices[i]，分别赋值给变量 b, a, gj, gi
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        # 创建与 pi[..., 0] 相同形状的全零张量，设备为 device
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        # 获取目标数量 n
        n = b.shape[0]  # number of targets
        # 如果目标数量不为零
        if n:
            # 累加目标数量到 nt
            nt += n  # cumulative targets
            # 从 pi 中获取与目标对应的子集 ps
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # 回归
            # 计算预测框的中心坐标和宽高
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            # 将中心坐标和宽高拼接成预测框
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            # 计算预测框和目标框的 IoU
            iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            # 计算 IoU 损失并累加到 lbox
            lbox += (1.0 - iou).mean()  # iou loss

            # 目标性
            # 计算目标性值并赋值给 tobj
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

            # 分类
            # 如果类别数大于 1
            if model.nc > 1:  # cls loss (only if multiple classes)
                # 创建与 ps[:, 5:] 相同形状的张量，并填充为 cn
                t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                # 将目标类别对应的位置赋值为 cp
                t[range(n), tcls[i]] = cp
                # 计算分类损失并累加到 lcls
                lcls += BCEcls(ps[:, 5:], t)  # BCE

            # 将目标追加到文本文件
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        # 计算目标性损失并乘以 balance[i] 累加到 lobj
        lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

    # 计算输出数量缩放比例
    s = 3 / np  # output count scaling
    # 将 lbox 乘以 h['box'] 和 s 累加到 loss
    lbox *= h['box'] * s
    # 将 lobj 乘以 h['obj'] 和 s 以及 1.4（如果 np 等于 4 则乘以 1.4，否则乘以 1.） 累加到 loss
    lobj *= h['obj'] * s * (1.4 if np == 4 else 1.)
    # 将 lcls 乘以 h['cls'] 和 s 累加到 loss
    lcls *= h['cls'] * s
    # 获取 tobj 的批量大小
    bs = tobj.shape[0]  # batch size

    # 计算总损失并返回
    loss = lbox + lobj + lcls
    return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
def build_targets(p, targets, model):
    # 为 compute_loss() 构建目标，输入目标（图像，类别，x，y，w，h）
    det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # 获取 Detect() 模块
    na, nt = det.na, targets.shape[0]  # 锚点数，目标数
    tcls, tbox, indices, anch = [], [], [], []  # 初始化空列表
    gain = torch.ones(7, device=targets.device)  # 归一化到网格空间的增益
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # 与 .repeat_interleave(nt) 相同
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # 添加锚点索引

    g = 0.5  # 偏置
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # 偏移量
    # 遍历目标检测结果中的目标数量
    for i in range(det.nl):
        # 获取当前目标检测结果中的锚框
        anchors = det.anchors[i]
        # 计算xyxy增益并转换为张量
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

        # 将目标与锚框进行匹配
        t = targets * gain
        if nt:
            # 匹配
            r = t[:, :, 4:6] / anchors[:, None]  # 计算宽高比
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # 比较
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # 过滤

            # 计算偏移量
            gxy = t[:, 2:4]  # 网格xy
            gxi = gain[[2, 3]] - gxy  # 反向
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # 定义
        b, c = t[:, :2].long().T  # 图像, 类别
        gxy = t[:, 2:4]  # 网格xy
        gwh = t[:, 4:6]  # 网格宽高
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # 网格xy索引

        # 追加
        a = t[:, 6].long()  # 锚框索引
        indices.append((b, a, gj, gi))  # 图像, 锚框, 网格索引
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # 目标框
        anch.append(anchors[a])  # 锚框
        tcls.append(c)  # 类别

    return tcls, tbox, indices, anch
def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    # 获取类别数
    nc = prediction[0].shape[1] - 5  # number of classes
    # 获取置信度大于阈值的候选框
    xc = prediction[..., 4] > conf_thres  # candidates

    # 设置参数
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    return output


def strip_optimizer(f='weights/best.pt', s=''):  # from utils.general import *; strip_optimizer()
    # 从 'f' 中去除优化器以完成训练，可选择保存为 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    x['optimizer'] = None
    x['training_results'] = None
    x['epoch'] = -1
    x['model'].half()  # 转换为 FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # 文件大小
    print('Optimizer stripped from %s,%s %.1fMB' % (f, (' saved as %s,' % s) if s else '', mb))


def coco_class_count(path='../coco/labels/train2014/'):
    # 每个类别的出现次数直方图
    nc = 80  # number classes
    x = np.zeros(nc, dtype='int32')
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        x += np.bincount(labels[:, 0].astype('int32'), minlength=nc)
        print(i, len(files))


def coco_only_people(path='../coco/labels/train2017/'):  # from utils.general import *; coco_only_people()
    # 找到只包含人的图像
    files = sorted(glob.glob('%s/*.*' % path))
    # 遍历文件列表，同时获取索引和文件名
    for i, file in enumerate(files):
        # 从文件中加载数据，数据类型为 32 位浮点数，然后重新组织成 5 列的形状
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        # 如果所有标签的第一列都等于 0
        if all(labels[:, 0] == 0):
            # 打印标签的行数和文件名
            print(labels.shape[0], file)
# 将图片随机裁剪成正方形，裁剪比例为给定比例，默认路径为'../images/'
def crop_images_random(path='../images/', scale=0.50):  # from utils.general import *; crop_images_random()
    # 随机裁剪图片成正方形，裁剪比例为给定比例
    # 警告：会覆盖原始图片！
    for file in tqdm(sorted(glob.glob('%s/*.*' % path))):
        # 读取图片
        img = cv2.imread(file)  # BGR
        if img is not None:
            h, w = img.shape[:2]

            # 创建随机蒙版
            a = 30  # 最小尺寸（像素）
            mask_h = random.randint(a, int(max(a, h * scale)))  # 蒙版高度
            mask_w = mask_h  # 蒙版宽度

            # 裁剪框
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # 应用随机颜色蒙版
            cv2.imwrite(file, img[ymin:ymax, xmin:xmax])


# 创建单类别的 COCO 数据集，默认路径为'../coco/labels/train2014/'，默认类别为43
def coco_single_class_labels(path='../coco/labels/train2014/', label_class=43):
    # 创建单类别的 COCO 数据集
    # from utils.general import *; coco_single_class_labels()
    if os.path.exists('new/'):
        shutil.rmtree('new/')  # 删除输出文件夹
    os.makedirs('new/')  # 创建新的输出文件夹
    os.makedirs('new/labels/')
    os.makedirs('new/images/')
    # 遍历指定路径下的所有文件，并按文件名排序
    for file in tqdm(sorted(glob.glob('%s/*.*' % path))):
        # 打开文件，读取其中内容，并按行分割成数组，然后转换成浮点数类型的数组
        with open(file, 'r') as f:
            labels = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
        # 找到数组中第一列等于指定标签类别的索引
        i = labels[:, 0] == label_class
        # 如果存在符合条件的索引
        if any(i):
            # 替换文件路径中的关键词，得到对应的图像文件路径
            img_file = file.replace('labels', 'images').replace('txt', 'jpg')
            # 将数组中所有元素的第一列重置为0
            labels[:, 0] = 0  # reset class to 0
            # 打开文件，追加模式，将图像文件路径添加到数据集列表中
            with open('new/images.txt', 'a') as f:  # add image to dataset list
                f.write(img_file + '\n')
            # 打开文件，追加模式，将符合条件的标签写入文件
            with open('new/labels/' + Path(file).name, 'a') as f:  # write label
                for l in labels[i]:
                    f.write('%g %.6f %.6f %.6f %.6f\n' % tuple(l))
            # 复制图像文件到指定目录
            shutil.copyfile(src=img_file, dst='new/images/' + Path(file).name.replace('txt', 'jpg'))  # copy images
def kmean_anchors(path='./data/coco128.yaml', n=9, img_size=640, thr=1.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            path: path to dataset *.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.general import *; _ = kmean_anchors()
    """
    thr = 1. / thr  # 计算 thr 的倒数

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]  # 计算宽高比
        x = torch.min(r, 1. / r).min(2)[0]  # 计算比例指标
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # 返回比例指标和最佳比例指标

    def fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)  # 计算指标
        return (best * (best > thr).float()).mean()  # 返回适应度

    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # sort small to large，按照面积从小到大排序
        x, best = metric(k, wh0)  # 计算指标
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        print('thr=%.2f: %.4f best possible recall, %.2f anchors past thr' % (thr, bpr, aat))  # 打印最佳召回率和超过阈值的锚点数量
        print('n=%g, img_size=%s, metric_all=%.3f/%.3f-mean/best, past_thr=%.3f-mean: ' %
              (n, img_size, x.mean(), best.mean(), x[x > thr].mean()), end='')  # 打印参数和指标
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg，打印锚点的宽高
        return k  # 返回锚点
    # 如果路径是字符串，则读取 *.yaml 文件
    if isinstance(path, str):  # *.yaml file
        # 打开文件并加载数据到字典
        with open(path) as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        # 导入 LoadImagesAndLabels 函数
        from utils.datasets import LoadImagesAndLabels
        # 使用数据字典中的训练数据创建数据集对象
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)
    else:
        # 如果路径不是字符串，则直接将路径赋值给数据集对象
        dataset = path  # dataset

    # 计算标签的宽度和高度
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    # 计算宽度和高度，并将结果连接成一个数组
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

    # 过滤
    # 统计小于3像素的标签数量
    i = (wh0 < 3.0).any(1).sum()
    # 如果存在小于3像素的标签，则打印警告信息
    if i:
        print('WARNING: Extremely small objects found. '
              '%g of %g labels are < 3 pixels in width or height.' % (i, len(wh0)))
    # 过滤掉宽度和高度小于2像素的标签
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels

    # 进行 Kmeans 计算
    print('Running kmeans for %g anchors on %g points...' % (n, len(wh)))
    # 计算标准差用于白化
    s = wh.std(0)  # sigmas for whitening
    # 进行 Kmeans 计算，得到聚类中心和距离
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    # 将聚类中心乘以标准差
    k *= s
    # 将 wh 转换为 torch 的 float32 类型
    wh = torch.tensor(wh, dtype=torch.float32)  # filtered
    # 将 wh0 转换为 torch 的 float32 类型
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # unfiltered
    # 打印 Kmeans 计算结果
    k = print_results(k)

    # 绘图
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.tight_layout()
    # fig.savefig('wh.png', dpi=200)

    # 进化
    # 导入 numpy 的随机模块
    npr = np.random
    # 计算适应度、代数、变异概率和标准差
    f, sh, mp, s = fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    # 创建进化锚点的进度条
    pbar = tqdm(range(gen), desc='Evolving anchors with Genetic Algorithm')  # progress bar
    # 对于进度条中的每个步骤
    for _ in pbar:
        # 创建一个与 sh 相同形状的全为 1 的数组
        v = np.ones(sh)
        # 当数组 v 中的所有元素都为 1 时，进行变异（防止重复）
        while (v == 1).all():
            # 生成一个与 sh 相同形状的随机数组，小于 mp 的元素设为 1，其余为 0
            # 乘以随机数，再乘以标准正态分布的随机数，再乘以 s，最后加上 1，取值范围为 0.3 到 3.0
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        # 将 k 的副本乘以 v，并取值范围为 2.0 到正无穷
        kg = (k.copy() * v).clip(min=2.0)
        # 计算 kg 的适应度
        fg = fitness(kg)
        # 如果 kg 的适应度大于 f
        if fg > f:
            # 更新 f 和 k
            f, k = fg, kg.copy()
            # 更新进度条的描述
            pbar.desc = 'Evolving anchors with Genetic Algorithm: fitness = %.4f' % f
            # 如果 verbose 为真，打印结果
            if verbose:
                print_results(k)

    # 返回结果
    return print_results(k)
# 定义一个函数，用于打印变异结果到文件hyp_evolved.yaml，并可选择上传到指定的bucket
def print_mutation(hyp, results, yaml_file='hyp_evolved.yaml', bucket=''):
    # 将超参数的键格式化成10个字符的字符串，用空格填充
    a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    # 将超参数的值格式化成10个字符的浮点数，保留3位小数
    b = '%10.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    # 将结果格式化成10个字符的浮点数，保留4位小数
    c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
    # 打印超参数键、值和结果
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    # 如果指定了bucket，则进行以下操作
    if bucket:
        # 构建文件在bucket中的URL
        url = 'gs://%s/evolve.txt' % bucket
        # 如果远程文件大小大于本地文件大小，则下载远程文件
        if gsutil_getsize(url) > (os.path.getsize('evolve.txt') if os.path.exists('evolve.txt') else 0):
            os.system('gsutil cp %s .' % url)  # download evolve.txt if larger than local

    # 以追加模式打开文件evolve.txt
    with open('evolve.txt', 'a') as f:  # append result
        # 将结果和超参数值写入文件
        f.write(c + b + '\n')
    # 加载唯一的行
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)  # load unique rows
    # 根据适应度对行进行排序
    x = x[np.argsort(-fitness(x))]  # sort
    # 将排序后的结果保存到文件evolve.txt中
    np.savetxt('evolve.txt', x, '%10.3g')  # save sort by fitness

    # 保存yaml文件
    for i, k in enumerate(hyp.keys()):
        hyp[k] = float(x[0, i + 7])
    with open(yaml_file, 'w') as f:
        results = tuple(x[0, :7])
        c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
        # 写入超参数进化结果和指标到yaml文件
        f.write('# Hyperparameter Evolution Results\n# Generations: %g\n# Metrics: ' % len(x) + c + '\n\n')
        yaml.dump(hyp, f, sort_keys=False)

    # 如果指定了bucket，则进行以下操作
    if bucket:
        # 上传文件evolve.txt和yaml_file到指定的bucket
        os.system('gsutil cp evolve.txt %s gs://%s' % (yaml_file, bucket))  # upload


# 定义一个函数，用于将yolo输出应用到第二阶段分类器
def apply_classifier(x, model, img, im0):
    # 如果im0是numpy数组，则将其转换为列表
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # 遍历 x 中的每个元素，同时获取索引和元素值，表示每张图片
        if d is not None and len(d):  # 如果元素值不为空且长度大于0
            d = d.clone()  # 复制元素值

            # 重塑和填充切割图像
            b = xyxy2xywh(d[:, :4])  # 提取边界框坐标
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # 将矩形边界框转换为正方形
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # 填充边界框
            d[:, :4] = xywh2xyxy(b).long()  # 更新边界框坐标

            # 将边界框坐标从 img_size 缩放到 im0 大小
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # 类别
            pred_cls1 = d[:, 5].long()  # 提取类别信息
            ims = []
            for j, a in enumerate(d):  # 遍历元素值中的每个项目，同时获取索引和元素值
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]  # 根据边界框坐标裁剪图像
                im = cv2.resize(cutout, (224, 224))  # 调整图像大小
                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR 转换为 RGB，转换为 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # 转换数据类型为 float32
                im /= 255.0  # 将像素值范围从 0 - 255 转换为 0.0 - 1.0
                ims.append(im)  # 将处理后的图像添加到列表中

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # 使用模型进行分类预测
            x[i] = x[i][pred_cls1 == pred_cls2]  # 保留匹配类别的检测结果

    return x  # 返回处理后的结果
# 计算适应度函数，用于结果文件或进化文件
def fitness(x):
    w = [0.0, 0.0, 0.1, 0.9]  # 权重 [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)

# 将模型输出转换为目标格式 [batch_id, class_id, x, y, w, h, conf]
def output_to_target(output, width, height):
    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()

    targets = []
    for i, o in enumerate(output):
        if o is not None:
            for pred in o:
                box = pred[:4]
                w = (box[2] - box[0]) / width
                h = (box[3] - box[1]) / height
                x = box[0] / width + w / 2
                y = box[1] / height + h / 2
                conf = pred[4]
                cls = int(pred[5])

                targets.append([i, cls, x, y, w, h, conf])

    return np.array(targets)

# 增加目录 runs/exp1 --> runs/exp2_comment
def increment_dir(dir, comment=''):
    n = 0  # 数字
    dir = str(Path(dir))  # 适配不同操作系统
    dirs = sorted(glob.glob(dir + '*'))  # 目录
    if dirs:
        matches = [re.search(r"exp(\d+)", d) for d in dirs]
        idxs = [int(m.groups()[0]) for m in matches if m]
        if idxs:
            n = max(idxs) + 1  # 增加
    return dir + str(n) + ('_' + comment if comment else '')

# 绘图函数 ---------------------------------------------------------------------------------------------------
# 2D 直方图，用于 labels.png 和 evolve.png
def hist2d(x, y, n=100):
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])

# 低通滤波器
def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):
    # 定义一个 butter_lowpass 函数，用于创建一个低通滤波器
    def butter_lowpass(cutoff, fs, order):
        # 计算奈奎斯特频率
        nyq = 0.5 * fs
        # 计算归一化截止频率
        normal_cutoff = cutoff / nyq
        # 使用 butter 函数创建滤波器的分子和分母系数
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        # 返回滤波器的分子和分母系数
        return b, a
    
    # 调用 butter_lowpass 函数，获取滤波器的分子和分母系数
    b, a = butter_lowpass(cutoff, fs, order=order)
    # 对数据进行前向-后向滤波
    return filtfilt(b, a, data)
# 在图像上绘制一个边界框
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # 线条/字体的厚度
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  
    # 如果未指定颜色，则随机生成一个颜色
    color = color or [random.randint(0, 255) for _ in range(3)]
    # 获取边界框的两个顶点坐标
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # 在图像上绘制矩形边界框
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
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


def plot_wh_methods():  
    # 比较两种宽度-高度锚点乘法的方法
    x = np.arange(-4.0, 4.0, .1)
    ya = np.exp(x)
    yb = torch.sigmoid(torch.from_numpy(x)).numpy() * 2
    # 创建图表
    fig = plt.figure(figsize=(6, 3), dpi=150)
    # 绘制曲线
    plt.plot(x, ya, '.-', label='YOLOv3')
    plt.plot(x, yb ** 2, '.-', label='YOLOv5 ^2')
    plt.plot(x, yb ** 1.6, '.-', label='YOLOv5 ^1.6')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=0, top=6)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.grid()
    plt.legend()
    fig.tight_layout()
    fig.savefig('comparison.png', dpi=200)


def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=640, max_subplots=16):
    # 线条的厚度
    tl = 3  
    # 字体的厚度
    tf = max(tl - 1, 1)  

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()

    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # 反归一化
    if np.max(images[0]) <= 1:
        images *= 255

    # 获取图像的批量大小、通道数、高度和宽度
    bs, _, h, w = images.shape  
    # 限制绘图图像的数量
    bs = min(bs, max_subplots)  
    # 计算子图的数量（取平方根向上取整）
    ns = np.ceil(bs ** 0.5)  

    # 检查是否需要调整大小
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    # 创建输出的空数组
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)

    # 修正类别-颜色映射
    prop_cycle = plt.rcParams['axes.prop_cycle']
    # 将十六进制颜色转换为 RGB
    hex2rgb = lambda h: tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    # 从颜色循环中获取颜色并转换为 RGB 存储在列表中
    color_lut = [hex2rgb(h) for h in prop_cycle.by_key()['color']]
    # 遍历图像列表，并返回索引和图像
    for i, img in enumerate(images):
        # 如果是最后一批图像，并且图像数量少于预期
        if i == max_subplots:  
            # 跳出循环
            break

        # 计算图像在拼接图中的位置
        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        # 调整图像通道顺序
        img = img.transpose(1, 2, 0)
        # 如果缩放因子小于1，对图像进行缩放
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        # 将图像放入拼接图中对应位置
        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img

        # 如果存在目标信息
        if len(targets) > 0:
            # 获取当前图像的目标信息
            image_targets = targets[targets[:, 0] == i]
            # 转换目标框的格式为(x_min, y_min, x_max, y_max)
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            # 获取目标类别
            classes = image_targets[:, 1].astype('int')
            # 判断是否为真实标签
            gt = image_targets.shape[1] == 6  # ground truth if no conf column
            # 如果不是真实标签，获取置信度信息
            conf = None if gt else image_targets[:, 6]  # check for confidence presence (gt vs pred)

            # 将目标框坐标转换为拼接图中的坐标
            boxes[[0, 2]] *= w
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] *= h
            boxes[[1, 3]] += block_y
            # 遍历目标框
            for j, box in enumerate(boxes.T):
                # 获取类别
                cls = int(classes[j])
                # 根据类别获取颜色
                color = color_lut[cls % len(color_lut)]
                # 如果存在类别名称，则使用名称，否则使用类别索引
                cls = names[cls] if names else cls
                # 如果是真实标签或者置信度大于0.3
                if gt or conf[j] > 0.3:  # 0.3 conf thresh
                    # 设置标签信息
                    label = '%s' % cls if gt else '%s %.1f' % (cls, conf[j])
                    # 绘制目标框
                    plot_one_box(box, mosaic, label=label, color=color, line_thickness=tl)

        # 绘制图像文件名标签
        if paths is not None:
            # 获取文件名并进行截断
            label = os.path.basename(paths[i])[:40]  # trim to 40 char
            # 获取文本大小
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            # 绘制文件名标签
            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf,
                        lineType=cv2.LINE_AA)

        # 绘制图像边框
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)
    # 如果文件名不为空，则执行以下操作
    if fname is not None:
        # 将马赛克图像调整大小为原始图像的一半，并使用INTER_AREA插值方法
        mosaic = cv2.resize(mosaic, (int(ns * w * 0.5), int(ns * h * 0.5)), interpolation=cv2.INTER_AREA)
        # 将调整大小后的马赛克图像转换为RGB格式，并保存为指定文件名
        cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))

    # 返回处理后的马赛克图像
    return mosaic
# 绘制学习率调度器的变化图像
def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=''):
    # 复制优化器和调度器，以免修改原始对象
    optimizer, scheduler = copy(optimizer), copy(scheduler)
    # 存储每个epoch的学习率
    y = []
    for _ in range(epochs):
        # 更新学习率
        scheduler.step()
        # 存储学习率
        y.append(optimizer.param_groups[0]['lr'])
    # 绘制学习率变化图
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'LR.png', dpi=200)


def plot_test_txt():  # from utils.general import *; plot_test()
    # 绘制test.txt的直方图
    x = np.loadtxt('test.txt', dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    plt.savefig('hist2d.png', dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig('hist1d.png', dpi=200)


def plot_targets_txt():  # from utils.general import *; plot_targets_txt()
    # 绘制targets.txt的直方图
    x = np.loadtxt('targets.txt', dtype=np.float32).T
    s = ['x targets', 'y targets', 'width targets', 'height targets']
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label='%.3g +/- %.3g' % (x[i].mean(), x[i].std()))
        ax[i].legend()
        ax[i].set_title(s[i])
    plt.savefig('targets.jpg', dpi=200)


def plot_study_txt(f='study.txt', x=None):  # from utils.general import *; plot_study_txt()
    # 绘制由test.py生成的study.txt的图表
    fig, ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)
    ax = ax.ravel()

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    # 遍历包含不同尺寸后缀的文件名列表
    for f in ['study/study_coco_yolov5%s.txt' % x for x in ['s', 'm', 'l', 'x']]:
        # 从文件中加载数据到数组 y，指定数据类型为 np.float32，指定要使用的列，转置后赋值给 y
        y = np.loadtxt(f, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9], ndmin=2).T
        # 如果 x 为 None，则创建一个与 y.shape[1] 大小相同的数组，否则使用给定的 x 创建数组
        x = np.arange(y.shape[1]) if x is None else np.array(x)
        # 创建字符串列表 s
        s = ['P', 'R', 'mAP@.5', 'mAP@.5:.95', 't_inference (ms/img)', 't_NMS (ms/img)', 't_total (ms/img)']
        # 遍历字符串列表 s
        for i in range(7):
            # 在第 i 个子图上绘制 x 和 y[i] 的关系图，设置线型、线宽和标记大小
            ax[i].plot(x, y[i], '.-', linewidth=2, markersize=8)
            # 设置第 i 个子图的标题为 s[i]
            ax[i].set_title(s[i])

        # 找到 y[3] 中最大值的索引并加 1，赋值给 j
        j = y[3].argmax() + 1
        # 在第二个子图上绘制 y[6, :j] 和 y[3, :j]*1E2 的关系图，设置线型、线宽、标记大小和标签
        ax2.plot(y[6, :j], y[3, :j] * 1E2, '.-', linewidth=2, markersize=8, label=Path(f).stem.replace('study_coco_', '').replace('yolo', 'YOLO'))

    # 在第二个子图上绘制指定的数据，设置线型、线宽、标记大小、透明度和标签
    ax2.plot(1E3 / np.array([209, 140, 97, 58, 35, 18]), [34.6, 40.5, 43.0, 47.5, 49.7, 51.5], 'k.-', linewidth=2, markersize=8, alpha=.25, label='EfficientDet')
    # 设置第二个子图的网格
    ax2.grid()
    # 设置第二个子图的 x 轴范围
    ax2.set_xlim(0, 30)
    # 设置第二个子图的 y 轴范围
    ax2.set_ylim(28, 50)
    # 设置第二个子图的 y 轴刻度
    ax2.set_yticks(np.arange(30, 55, 5))
    # 设置第二个子图的 x 轴标签
    ax2.set_xlabel('GPU Speed (ms/img)')
    # 设置第二个子图的 y 轴标签
    ax2.set_ylabel('COCO AP val')
    # 在第二个子图上添加图例，位置为右下角
    ax2.legend(loc='lower right')
    # 保存图像为 study_mAP_latency.png，设置分辨率为 300
    plt.savefig('study_mAP_latency.png', dpi=300)
    # 保存图像为与输入文件名相同，只是后缀改为 .png，设置分辨率为 300
    plt.savefig(f.replace('.txt', '.png'), dpi=300)
def plot_labels(labels, save_dir=''):
    # 绘制数据集标签
    c, b = labels[:, 0], labels[:, 1:].transpose()  # 分离类别和框
    nc = int(c.max() + 1)  # 类别数量

    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)  # 创建子图
    ax = ax.ravel()
    ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)  # 绘制类别直方图
    ax[0].set_xlabel('classes')  # 设置 x 轴标签
    ax[1].scatter(b[0], b[1], c=hist2d(b[0], b[1], 90), cmap='jet')  # 绘制散点图
    ax[1].set_xlabel('x')  # 设置 x 轴标签
    ax[1].set_ylabel('y')  # 设置 y 轴标签
    ax[2].scatter(b[2], b[3], c=hist2d(b[2], b[3], 90), cmap='jet')  # 绘制散点图
    ax[2].set_xlabel('width')  # 设置 x 轴标签
    ax[2].set_ylabel('height')  # 设置 y 轴标签
    plt.savefig(Path(save_dir) / 'labels.png', dpi=200)  # 保存图像
    plt.close()  # 关闭图像

    # seaborn correlogram
    try:
        import seaborn as sns
        import pandas as pd
        x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height'])  # 转换数据格式
        sns.pairplot(x, corner=True, diag_kind='hist', kind='scatter', markers='o',
                     plot_kws=dict(s=3, edgecolor=None, linewidth=1, alpha=0.02),
                     diag_kws=dict(bins=50))  # 绘制 correlogram
        plt.savefig(Path(save_dir) / 'labels_correlogram.png', dpi=200)  # 保存图像
        plt.close()  # 关闭图像
    except Exception as e:
        pass


def plot_evolution(yaml_file='data/hyp.finetune.yaml'):  # 绘制超参数演化结果
    # 从 evolve.txt 文件中读取超参数演化结果
    with open(yaml_file) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    x = np.loadtxt('evolve.txt', ndmin=2)  # 读取演化结果数据
    f = fitness(x)  # 计算适应度
    # weights = (f - f.min()) ** 2  # 用于加权结果的权重
    plt.figure(figsize=(10, 12), tight_layout=True)  # 创建图像
    matplotlib.rc('font', **{'size': 8})  # 设置字体大小
    # 遍历字典 hyp 的键值对，同时获取索引 i
    for i, (k, v) in enumerate(hyp.items()):
        # 从数组 x 中获取第 i+7 列的数据，赋值给 y
        y = x[:, i + 7]
        # 从 y 中获取最大值的索引，赋值给 f.argmax()，然后获取对应的值，赋值给 mu
        mu = y[f.argmax()]  # best single result
        # 创建一个 6x5 的子图，并定位到第 i+1 个位置
        plt.subplot(6, 5, i + 1)
        # 绘制散点图，x 轴为 y，y 轴为 f，颜色根据二维直方图的密度着色，使用 viridis 颜色映射，设置透明度和边缘颜色
        plt.scatter(y, f, c=hist2d(y, f, 20), cmap='viridis', alpha=.8, edgecolors='none')
        # 在散点图上绘制一个黑色加号，位置为 (mu, f.max())，大小为 15
        plt.plot(mu, f.max(), 'k+', markersize=15)
        # 设置子图标题，格式为 '%s = %.3g' % (k, mu)，字体大小为 9
        plt.title('%s = %.3g' % (k, mu), fontdict={'size': 9})  # limit to 40 characters
        # 如果 i 除以 5 的余数不为 0，则不显示 y 轴刻度
        if i % 5 != 0:
            plt.yticks([])
        # 打印格式化字符串，格式为 '%15s: %.3g' % (k, mu)
        print('%15s: %.3g' % (k, mu))
    # 保存绘制的图形为 evolve.png，设置分辨率为 200
    plt.savefig('evolve.png', dpi=200)
    # 打印提示信息，表示图形已保存
    print('\nPlot saved as evolve.png')
def plot_results_overlay(start=0, stop=0):  # 定义函数，用于绘制训练结果的叠加图，可以指定起始和结束位置
    # legends列表，包含了要显示的图例名称
    s = ['train', 'train', 'train', 'Precision', 'mAP@0.5', 'val', 'val', 'val', 'Recall', 'mAP@0.5:0.95']
    # titles列表，包含了要显示的图表标题
    t = ['Box', 'Objectness', 'Classification', 'P-R', 'mAP-F1']
    # 遍历所有符合条件的文件，读取结果数据
    for f in sorted(glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')):
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]  # 获取结果数据的行数
        x = range(start, min(stop, n) if stop else n)  # 生成 x 轴坐标
        # 创建图表和子图
        fig, ax = plt.subplots(1, 5, figsize=(14, 3.5), tight_layout=True)
        ax = ax.ravel()
        # 遍历每个子图
        for i in range(5):
            # 绘制曲线
            for j in [i, i + 5]:
                y = results[j, x]
                ax[i].plot(x, y, marker='.', label=s[j])
            # 设置子图标题和图例
            ax[i].set_title(t[i])
            ax[i].legend()
            # 添加文件名作为 y 轴标签
            ax[i].set_ylabel(f) if i == 0 else None
        # 保存图表
        fig.savefig(f.replace('.txt', '.png'), dpi=200)


def plot_results(start=0, stop=0, bucket='', id=(), labels=(), save_dir=''):
    # 定义函数，用于绘制训练结果图表
    fig, ax = plt.subplots(2, 5, figsize=(12, 6))  # 创建图表和子图
    ax = ax.ravel()
    # legends列表，包含了要显示的图例名称
    s = ['Box', 'Objectness', 'Classification', 'Precision', 'Recall',
         'val Box', 'val Objectness', 'val Classification', 'mAP@0.5', 'mAP@0.5:0.95']
    # 如果 bucket 存在
    if bucket:
        # 使用 gsutil 命令从 Google Cloud Storage 下载文件到本地
        files = ['results%g.txt' % x for x in id]
        # 构建 gsutil 命令，将文件从 Google Cloud Storage 复制到当前目录
        c = ('gsutil cp ' + '%s ' * len(files) + '.') % tuple('gs://%s/results%g.txt' % (bucket, x) for x in id)
        # 执行 gsutil 命令
        os.system(c)
    else:
        # 如果 bucket 不存在，则从本地目录和 Downloads 目录中获取文件列表
        files = glob.glob(str(Path(save_dir) / 'results*.txt')) + glob.glob('../../Downloads/results*.txt')
    # 遍历文件列表
    for fi, f in enumerate(files):
        try:
            # 从文件中加载数据到 results 变量
            results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
            # 获取结果数据的行数
            n = results.shape[1]  # number of rows
            # 设置 x 轴的范围
            x = range(start, min(stop, n) if stop else n)
            # 遍历结果数据的前 10 行
            for i in range(10):
                # 获取当前行的数据
                y = results[i, x]
                # 如果是指定的行，则将值为 0 的元素替换为 NaN
                if i in [0, 1, 2, 5, 6, 7]:
                    y[y == 0] = np.nan  # don't show zero loss values
                    # 对数据进行归一化处理
                    # y /= y[0]  # normalize
                # 获取标签信息
                label = labels[fi] if len(labels) else Path(f).stem
                # 在对应的子图上绘制折线图
                ax[i].plot(x, y, marker='.', label=label, linewidth=1, markersize=6)
                # 设置子图的标题
                ax[i].set_title(s[i])
                # 如果是指定的行，则共享训练和验证损失的 y 轴
                # if i in [5, 6, 7]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            # 捕获异常并打印警告信息
            print('Warning: Plotting error for %s; %s' % (f, e))

    # 调整图形布局
    fig.tight_layout()
    # 在第二个子图上添加图例
    ax[1].legend()
    # 保存图形为图片文件
    fig.savefig(Path(save_dir) / 'results.png', dpi=200)
```