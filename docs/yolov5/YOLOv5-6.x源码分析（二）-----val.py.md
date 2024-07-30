<!--yml
category: 游戏
date: 2023-09-17 14:45:14
-->

# YOLOv5-6.x源码分析（二）---- val.py

> 来源：[https://blog.csdn.net/weixin_51322383/article/details/130317934](https://blog.csdn.net/weixin_51322383/article/details/130317934)

### 文章目录

*   [前言](#_1)
*   [🚀YOLOv5-6.x源码分析（二）---- val.py](#YOLOv56x_valpy_11)
*   *   [1\. 导入需要的包](#1__12)
    *   [2\. 保存信息](#2__55)
    *   [3\. 计算指标](#3__92)
    *   [4\. 设置opt参数](#4_opt_132)
    *   [5\. 执行main函数](#5_main_187)
    *   [6\. 执行run函数](#6_run_224)
    *   *   [6.1 设置参数](#61__227)
        *   [6.2 初始化/加载模型以及设置设备](#62__261)
        *   [6.3 加载配置](#63__308)
        *   [6.4 加载val数据集](#64_val_327)
        *   [6.5 初始化](#65__356)
        *   [6.6 开始验证](#66__383)
        *   *   [6.6.1 验证前的预处理](#661__384)
            *   [6.6.2 前向推理](#662__407)
            *   [6.6.3 计算损失](#663__418)
            *   [6.6.4 NMS](#664_NMS_432)
            *   [6.6.5 统计真实框、预测框信息](#665__451)
            *   [6.6.6 保存预测信息](#666__504)
            *   [6.6.7 画出前3个bs图片的gt和pred框](#667_3bsgtpred_520)
            *   [6.6.8 计算mAP](#668_mAP_538)
            *   [6.6.9 打印各种指标](#669__567)
            *   [6.6.10 Return Results](#6610_Return_Results_591)
*   [总结](#_617)

# 前言

今天又看到了一位博主的分类专栏，更加坚定了我要养成坚持写博客的习惯。

昨天把`detect.py`的源码解读了，今天来解读一下`val.py`。这个脚本文件主要是在每一轮训练结束后，验证当前模型的mAP、混淆矩阵等指标，并修改`train.py`的参数。
这个脚本主要运用在`train.py`中的run函数里面，直接调用，当然也可以在模型训练完毕后，运行该脚本，进行模型的评估。

预计我会自顶向下，来解读每一个脚本文件。

**导航：**[YOLOv5-6.x源码分析 全流程记录](https://blog.csdn.net/weixin_51322383/article/details/130353834)

* * *

# 🚀YOLOv5-6.x源码分析（二）---- val.py

## 1\. 导入需要的包

```py
import argparse                 # 解析命令行
import json                     # 实现字典和json之间的解析
import os                       # 操作系统交互模块，包括文件路径等函数
import sys                      # sys系统模块 包含python解释器和它的环境相关的函数
from pathlib import Path        # path将str转换为Path对象，使字符串路径易于操作

import numpy as np
import torch
from tqdm import tqdm			# 进度条 
```

一些常用的基本库。

```py
 FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative 
```

老样子，定义一些相对路径参数，方便后面函数中调用。

```py
 from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, emojis, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync 
```

这个part是自定义的库，具体内容为：

*   `models.common：`网络结构类模块
*   `utils.callbacks：`定义回调函数，为logger服务
*   `utils.callbacks：`dataset和dataloader，数据集和数据加载
*   `utils.general：`常用的工具函数，比如检查文件存在、检查图片大小、打印命令行参数等
*   `utils.metrics：`模型验证指标，包括ap、混淆矩阵等
*   `utils.plots：`定义了Annotator类，绘制图像的信息
*   `utils.torch_utils：`Pytorch有关的工具函数

## 2\. 保存信息

```py
'''======================1.保存预测信息到txt文件====================='''
def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    # gn = [w,h,w,h]对应图片宽高，用于后面归一化
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist(): # tolist：变为列表
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format 保存的格式
        with open(file, 'a') as f:
            # 写入对应的文件夹里，路径默认为“runs\detect\exp*\labels”
            f.write(('%g ' * len(line)).rstrip() % line + '\n') 
```

这一段是将预测的信息保存到**txt**文件中。

```py
 '''======================2.保存预测信息到coco格式的json字典====================='''
def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh   坐标转换
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    # 序列解包
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)}) 
```

这一段是将预测的信息保存到**json**文件中。
`zip()`生成(x, y)形式的迭代器。

> 注意：之前的的xyxy格式是左上角右下角坐标 ，xywh是中心的坐标和宽高，**而coco的json格式的框坐标是xywh(左上角坐标 + 宽高)**，所以 box[:, :2] -= box[:, 2:] / 2 这行代码是将中心点坐标 -> 左上角坐标。

## 3\. 计算指标

```py
'''========================三、计算指标==========================='''
def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device) 
```

![matches](img/ae7de49a3942a59176bd03f1f21dd4fa.png)
这段代码主要是计算correct，来获取匹配预测框的iou信息。
看得有点晕乎乎的，python的高纬度矩阵运算也太难了。

这段函数的主要作用：

*   对预测框与gt进行匹配
*   对匹配上的预测框进行iou数之判断，用True来填充，其余没有匹配上的预测框的所有行数全部设为False

对于每张图像的预测框，需要筛选出能与gt匹配的框来进行相关的iou计算，设置了iou从0.5-0.95的10个梯度，如果匹配的预测框iou大于相对于的阈值，则在对应位置设置为True，否则设置为False；而对于没有匹配上的预测框全部设置为False。

> **Q：为什么要筛选**
> 这是因为一个gt只可能是一个类别，不可能是多个类别，所以需要取置信度最高的类别进行匹配。但是此时还可能多个gt和一个预测框匹配，同样的，为这个预测框分配iou值最高的gt，依次来实现一一配对

## 4\. 设置opt参数

```py
def parse_opt():
    parser = argparse.ArgumentParser()
    # 数据集配置文件地址 包含数据集的路径、类别个数、类名、下载地址等信息
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    # 模型的权重文件地址yolov5s.pt
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    # 前向传播的批次大小 默认32
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    # 输入网络的图片分辨率 默认640
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    # object置信度阈值 默认0.001
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    # 进行NMS时IOU的阈值 默认0.6
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    # 设置测试的类型 有train, val, test, speed or study几种 默认val
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    # 测试的设备
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 数据集是否只用一个类别 默认False
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    # 测试是否使用TTA Test Time Augment 默认False
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # 是否打印出每个类别的mAP 默认False
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    # 是否以txt文件的形式保存模型预测的框坐标, 默认False
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # 保存label+prediction杂交结果到对应.txt，默认False
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    # 保存置信度
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # 是否按照coco的json格式保存预测框，并且使用cocoapi做评估（需要同样coco的json格式的标签） 默认False
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    # 测试保存的源文件 默认runs/val
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    # 测试保存的文件地址 默认exp  保存在runs/val/exp下
    parser.add_argument('--name', default='exp', help='save to project/name')
    # 是否存在当前文件 默认False 一般是 no exist-ok 连用  所以一般都要重新创建文件夹
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # 是否使用半精度推理 默认False
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    # 是否使用 OpenCV DNN对ONNX 模型推理
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')

    # 解析上述参数
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)
    # |或 左右两个变量有一个为True 左边变量就为True
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt 
```

## 5\. 执行main函数

```py
def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    # 如果task in ['train', 'val', 'test']就正常测试 训练集/验证集/测试集
    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(emojis(f'WARNING: confidence threshold {opt.conf_thres} > 0.001 produces invalid results ⚠️'))
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        # 如果opt.task == 'speed' 就测试yolov5系列和yolov3-spp各个模型的速度评估
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        # 如果opt.task = ['study']就评估yolov5系列和yolov3-spp各个模型在各个尺度下的指标并可视化
        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot 
```

基本`opt.task`都是`val`，废话不多说，直接进入`run`函数。

## 6\. 执行run函数

这个run函数实际上是train.py执行的，当然也可以手动去执行`val.py`。
![train.py中调用run](img/faf58630de2a94ac8d5e2157c2def3cc.png)

### 6.1 设置参数

```py
'''======================1.设置参数====================='''
@torch.no_grad()
def run(data, # 数据集配置文件地址 包含数据集的路径、类别个数、类名、下载地址等信息 train.py时传入data_dict
        weights=None,  # 模型的权重文件地址 运行train.py=None 运行test.py=默认weights/yolov5s
        batch_size=32,  # 前向传播的批次大小 运行test.py传入默认32 运行train.py则传入batch_size // WORLD_SIZE * 2
        imgsz=640,  # 输入网络的图片分辨率 运行test.py传入默认640 运行train.py则传入imgsz_test
        conf_thres=0.001,  # object置信度阈值 默认0.001
        iou_thres=0.6,  # 进行NMS时IOU的阈值 默认0.6
        task='val',  # 设置测试的类型 有train, val, test, speed or study几种 默认val
        device='',  # 执行 val.py 所在的设备 cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # 数据集是否只有一个类别 默认False
        augment=False,  # 测试时增强
        verbose=False,  # 是否打印出每个类别的mAP 运行test.py传入默认Fasle 运行train.py则传入nc < 50 and final_epoch
        save_txt=False,  # 是否以txt文件的形式保存模型预测框的坐标 默认True
        save_hybrid=False,  # 是否保存预测每个目标的置信度到预测txt文件中 默认True
        save_conf=False,  # 保存置信度
        save_json=False,  # 是否按照coco的json格式保存预测框，并且使用cocoapi做评估（需要同样coco的json格式的标签）,
                      #运行test.py传入默认Fasle 运行train.py则传入is_coco and final_epoch(一般也是False)
        project=ROOT / 'runs/val',  # 验证结果保存的根目录 默认是 runs/val
        name='exp',  # 验证结果保存的目录 默认是exp  最终: runs/val/exp
        exist_ok=False,  # 如果文件存在就increment name，不存在就新建  默认False(默认文件都是不存在的)
        half=True,  # 使用 FP16 的半精度推理
        dnn=False,  # 在 ONNX 推理时使用 OpenCV DNN 后段端
        model=None,  # 如果执行val.py就为None 如果执行train.py就会传入( model=attempt_load(f, device).half() )
        dataloader=None, # 数据加载器 如果执行val.py就为None 如果执行train.py就会传入testloader
        save_dir=Path(''), # 文件保存路径 如果执行val.py就为‘’ , 如果执行train.py就会传入save_dir(runs/train/expn)
        plots=True, # 是否可视化 运行val.py传入，默认True
        callbacks=Callbacks(),  # 回调函数
        compute_loss=None, # 损失函数 运行val.py传入默认None 运行train.py则传入compute_loss(train)
        ): 
```

### 6.2 初始化/加载模型以及设置设备

```py
# ============================================== 初始化配置 ==================================================
    # 初始化模型并选择相应的计算设备
    # 判断是否是训练时调用run函数(执行train.py脚本), 如果是就使用训练时的设备 一般都是train
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        # 如果设备类型不是cpu 则将模型由32位浮点数转换为16位浮点数
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        # 调用torch_utils中select_device来选择执行程序时的设备
        device = select_device(device, batch_size=batch_size)

        # Directories  # 生成save_dir文件路径  run\test\expn
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        data = check_dataset(data)  # check 
```

这段代码主要是**初始化配置，加载模型、设置设备**

首先会判断模型是否存在。

接着判断是否是训练时调用`run`函数，即执行`train.py`，如果不是，就调用`select_device`选择设备，并生成**save_dir + make dir + 加载模型model + check imgsz + 加载data配置信息**。

*   训练时（train.py）调用：初始化模型参数、训练设备
*   验证时（val.py）调用：初始化设备、save_dir文件路径、make dir、加载模型、check imgsz、 加载+check data配置信息

最后**判断设备类型并仅仅单GPU支持一半的精度**。Half model 只能在单GPU设备上才能使用， 一旦使用half，不但模型需要设为half，输入模型的图片也需要设为half。如果设备类型不是CPU 则将模型由32位浮点数转换为16位浮点数。

> **model.half**是将网络权重和输入数据转换为半精度浮点数进行存储和计算，而**model.float**是使用单精度浮点数进行存储和计算。使用半精度浮点数可以减少内存占用和加速计算，但可能会影响模型的精度。**即GPU用half更快**

### 6.3 加载配置

```py
# ============================================== 加载配置 ==================================================
    model.eval()    # 启动模型验证模式；不启用 Batch Normalization 和 Dropout。 在eval模式下不会进行反向传播。
    cuda = device.type != 'cpu'
    # 通过 COCO 数据集的文件夹组织结构判断当前数据集是否为 COCO 数据集
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset
    # 确定检测的类别数目
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    # 计算mAP相关参数
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    # numel为pytorch预置函数 用来获取张量中的元素个数
    niou = iouv.numel() 
```

这一段是加载数据的.yaml配置文件信息。

首先进入模型验证模式，然后**确定检测的类别个数nc** ，以及**计算mAP相关参数**，设置iou阈值从0.5-0.95取10个(0.05间隔) 所以iouv: [0.50000, 0.55000, 0.60000, 0.65000, 0.70000, 0.75000, 0.80000, 0.85000, 0.90000, 0.95000]

### 6.4 加载val数据集

```py
# ============================================== 加载val数据集 ==================================================
    # 如果不是训练(执行val.py脚本调用run函数)就调用create_dataloader生成dataloader
    # 如果是训练(执行train.py调用run函数)就不需要生成dataloader 可以直接从参数中传过来testloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                              f'classes). Pass correct combination of --weights and --data that are trained together.'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad = 0.0 if task in ('speed', 'benchmark') else 0.5
        rect = False if task == 'benchmark' else pt  # square inference for benchmarks
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        # 调用datasets.py文件中的create_dataloader函数创建dataloader
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0] 
```

这段是**加载val数据集。**

如果是训练的时候调用 ，则不需要这段代码。如果是val调用，则需要用create_dataloader创建数据集。

### 6.5 初始化

```py
# ============================================== 初始化配置 ==================================================
    # 初始化一些测试需要的参数
    seen = 0    # 初始化测试的图片的数量
    # 初始化混淆矩阵
    confusion_matrix = ConfusionMatrix(nc=nc)
    # 获取数据集所有类别的类名
    names = dict(enumerate(model.names if hasattr(model, 'names') else model.module.names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    # 设置tqdm进度条的显示信息
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # 初始化测试集的损失
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar 
```

这段代码主要是获取数据集的相应参数。

**（1）初始化已完成测试图片数量，设置seen=0
（2）初始化混淆矩阵
（3）获取数据集类名 和coco数据集的类别索引
（4）设置tqdm进度条的显示信息
（5）初始化p, r, f1, mp, mr, map50, map指标和初始化测试集的损失以及初始化json文件中的字典 统计信息、ap等**

### 6.6 开始验证

#### 6.6.1 验证前的预处理

```py
# ============================================== 开始验证 ==================================================
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        # 1\. 预处理图片和target
        t1 = time_sync()
        if cuda:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()    # 获取当前时间
        dt[0] += t2 - t1    # 累计处理数据时间 
```

预处理大致可这样几步：

*   将图片和targets数据放到device上
*   将图片转换为半精度
*   图片像素值归一化
*   再获取一些图片信息，比如形状、batch等

#### 6.6.2 前向推理

```py
# 2\. run model  前向推理
        out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs
        dt[1] += time_sync() - t2 
```

模型前向推理、记录前向推理时间

*   `out:` 推理结果。1个 ，[bs, anchor_num*grid_w*grid_h, xywh+c+20classes] = [1, 19200+4800+1200, 25]
*   `train_out:` 训练结果。3个， [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]。如: [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]

> 有的模型输出使用的out，有的使用的train_out，我当时在部署地平线板子上时因为这个研究了好久

#### 6.6.3 计算损失

```py
# 3\. 计算验证集损失
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls 
```

不为空说明在执行`train.py`
loss 包含bounding box 回归的GIoU、object和class 三者的损失

*   `分类损失(cls_loss)`：该损失用于判断模型是否能够准确地识别出图像中的对象，并将其分类到正确的类别中。
*   `置信度损失(obj_loss)`：该损失用于衡量模型预测的框（即包含对象的矩形）与真实框之间的差异。
*   `边界框损失(box_loss)`：该损失用于衡量模型预测的边界框与真实边界框之间的差异，这有助于确保模型能够准确地定位对象。

> 置信度损失指的是模型预测的物体是否存在的概率和实际存在的概率之间的差距。边界框损失指的是模型预测的物体边界框位置和实际位置之间的差距。两种损失函数的重点不同，**置信度损失**的重点在判断物体**是否存在**，而**边界框损失**的重点在于**精确地定位物体的位置**。

#### 6.6.4 NMS

```py
# 4\. Run NMS
        # # 将真实框target的xywh(因为target是在labelimg中做了归一化的)映射到img(test)尺寸
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        # targets: [num_target, img_index+class_index+xywh] = [31, 6]
        # lb: {list: bs} 第一张图片的target[17, 5] 第二张[1, 5] 第三张[7, 5] 第四张[6, 5]
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t3 = time_sync()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        dt[2] += time_sync() - t3 
```

用于删除冗余的 bbox。

首先将真实框`target`的xywh (因为 target 是在 labelimg 中做了归一化的)映射到真实的图像尺寸
然后，在NMS之前将数据集标签 targets 添加到模型预测中，这允许在数据集中自动标记(for autolabelling)其它对象(在pred中混入gt)并且mAP反映了新的混合标签。nb为bs，即一个batch一个batch地计算。

最后调用general.py中的函数，进行NMS操作，并计算NMS过程所需要的时间。

#### 6.6.5 统计真实框、预测框信息

```py
 # 5\. 统计每张图片的真实框、预测框信息  Statistics per image
        # si代表第si张图片，pred是对应图片预测的label信息
        for si, pred in enumerate(out):
            # 获取第si张图片的gt标签信息 包括class, x, y, w, h    target[:, 0]为标签属于哪张图片的编号
            labels = targets[targets[:, 0] == si, 1:]
            # nl为图片检测到的目标个数
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            # 第si张图片对应的文件路径
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            # 统计测试图片数量 +1
            seen += 1

            if npr == 0:
                if nl:  # 预测为空但同时有label信息
                    # stats初始化为一个空列表[] 此处添加一个空信息
                    # 添加的每一个元素均为tuple 其中第二第三个变量为一个空的tensor
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            # 预测框评估
            if nl:
                # 获得xyxy的框
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                # 将图片调整为原来的大小
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                # 处理完gt的尺寸信息，重新构建成 (cls, xyxy)的格式
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels 在维度1上拼接
                # 对当前的预测框与gt进行一一匹配，并且在预测框的对应位置上获取iou的评分信息，其余没有匹配上的预测框设置为False
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    # 计算混淆矩阵
                    confusion_matrix.process_batch(predn, labelsn)
            # 每张图片的结果统计到stats里
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls) 
```

**这段代码主要是统计每张图片真实框和预测框的相关信息，包括计算混淆矩阵、计算correct、生成stats，非常重要！**

首先统计每张图片的相关信息，如预测label信息、标签gt信息等。然后统计检测到的目标个数和类别以及相对应的文件路径。

接着利用得到的上述信息进行目标的预测，并将结果保存同时输出日志，分别保存预测信息到image_name.txt文件和coco格式的json字典。

#### 6.6.6 保存预测信息

```py
# Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si]) 
```

利用得到的上述信息进行目标的预测，并将结果保存同时输出日志，分别保存预测信息到image_name.txt文件和coco格式的json字典。

*   txt文件保存的预测信息：cls＋xywh＋conf
*   jdict字典保存的预测信息：image_id + category_id + bbox + score

#### 6.6.7 画出前3个bs图片的gt和pred框

```py
# 画出前三个batch的图片的ground truth和预测框predictions(两个图)一起保存
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
            plot_images(im, output_to_target(out), paths, save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

        callbacks.run('on_val_batch_end') 
```

+ `gt` : 真实框，`Ground truth box`, 是人工标注的位置，存放在标注文件中
+`pred`: 预测框，`Prediction box`， 是由目标检测模型计算输出的框

val_batch0_labels.jpg
![在这里插入图片描述](img/8b72d97783bb43f4e6755ef199d1f4d4.png)
val_batch0_pred.jpg
![在这里插入图片描述](img/b9398626bb3cb2ddccaedb5ef3b95fe1.png)
我这里的batchsize为2，所以只有两张，看着比较空

#### 6.6.8 计算mAP

```py
 # 计算mAP
    # 统计stats中所有图片的统计结果 将stats列表的信息拼接到一起
    # stats(concat后): list{4} correct, conf, pcls, tcls  统计出的整个数据集的GT
    # correct [img_sum, 10] 整个数据集所有图片中所有预测框在每一个iou条件下是否是TP  [1905, 10]
    # conf [img_sum] 整个数据集所有图片中所有预测框的conf  [1905]
    # pcls [img_sum] 整个数据集所有图片中所有预测框的类别   [1905]
    # tcls [gt_sum] 整个数据集所有图片所有gt框的class     [929]
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    # stats[0].any(): stats[0]是否全部为False, 是则返回 False, 如果有一个为 True, 则返回 True
    if len(stats) and stats[0].any():
        # 根据上面的统计预测结果计算p, r, ap, f1, ap_class（ap_per_class函数是计算每个类的mAP等指标的）等指标
        # p: [nc] 最大平均f1时每个类别的precision
        # r: [nc] 最大平均f1时每个类别的recall
        # ap: [71, 10] 数据集每个类别在10个iou阈值下的mAP
        # f1 [nc] 最大平均f1时每个类别的f1
        # ap_class: [nc] 返回数据集中所有的类别index
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        # mp: [1] 所有类别的平均precision(最大f1时)
        # mr: [1] 所有类别的平均recall(最大f1时)
        # map50: [1] 所有类别的平均mAP@0.5
        # map: [1] 所有类别的平均mAP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        # nt: [nc] 统计出整个数据集的gt框中数据集各个类别的个数
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class 
```

#### 6.6.9 打印各种指标

```py
# Print results
    # Print results  数据集图片数量 + 数据集gt框的数量 + 所有类别的平均precision + 
    #                所有类别的平均recall + 所有类别的平均mAP@0.5 + 所有类别的平均mAP@0.5:0.95
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(emojis(f'WARNING: no labels found in {task} set, can not compute metrics without labels ⚠️'))

    # Print results per class
    # 细节展示每个类别的各个指标  类别 + 数据集图片数量 + 这个类别的gt框数量 + 这个类别的precision +
    #                        这个类别的recall + 这个类别的mAP@0.5 + 这个类别的mAP@0.5:0.95
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds  打印前向传播耗费的总时间、nms耗费总时间、总时间
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t) 
```

#### 6.6.10 Return Results

```py
# Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]	# maps [80] 所有类别的mAP@0.5:0.95
    # (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()): {tuple:7}
    #      0: mp [1] 所有类别的平均precision(最大f1时)
    #      1: mr [1] 所有类别的平均recall(最大f1时)
    #      2: map50 [1] 所有类别的平均mAP@0.5
    #      3: map [1] 所有类别的平均mAP@0.5:0.95
    #      4: val_box_loss [1] 验证集回归损失
    #      5: val_obj_loss [1] 验证集置信度损失
    #      6: val_cls_loss [1] 验证集分类损失
    # maps: [80] 所有类别的mAP@0.5:0.95
    # t: {tuple: 3} 0: 打印前向传播耗费的总时间   1: nms耗费总时间   2: 总时间
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t 
```

一般返回的结果会在train中获取。
![在这里插入图片描述](img/d136cbdb719b97c19c280aa0af3aecc8.png)

* * *

# 总结

这部分代码主要是对`train.py`训练后的ing进行评估和验证。难点在于[6.6.5统计真实框、预测框信息](#665__449)和[6.6.8计算mAP](#668_mAP_536)，需结合`metrics.py`脚本一起看。