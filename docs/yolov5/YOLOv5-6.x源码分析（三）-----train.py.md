<!--yml
category: 游戏
date: 2023-09-17 14:45:04
-->

# YOLOv5-6.x源码分析（三）---- train.py

> 来源：[https://blog.csdn.net/weixin_51322383/article/details/130336035](https://blog.csdn.net/weixin_51322383/article/details/130336035)

### 文章目录

*   [前引](#_1)
*   [🚀YOLOv5-6.x源码分析（三）---- train.py](#YOLOv56x_trainpy_5)
*   *   [1\. 导包](#1__6)
    *   [2\. 设置opt参数](#2_opt_58)
    *   [3\. 执行main函数](#3_main_147)
    *   *   [3.1 检查分布式训练环境](#31__148)
        *   [3.2 判断是否断点续训](#32__158)
        *   [3.3 判断是否是分布式训练](#33__192)
        *   [3.4 判断是否进化训练](#34__210)
        *   [3.5 遗传进化算法](#35__220)
    *   [4\. 执行train()函数](#4_train_223)
    *   *   [4.1 载入参数和配置信息](#41__224)
        *   [4.2 model](#42_model_283)
        *   *   [4.2.1 加载y训练模型](#421_y_284)
            *   [4.2.2 优化器](#422__342)
            *   [4.2.3 学习率设置](#423__363)
            *   [4.2.4 训练前的最后准备](#424__382)
        *   [4.3 数据加载](#43__438)
        *   *   [4.3.1 创建数据集](#431__440)
            *   [4.3.2 计算anchor](#432_anchor_501)
        *   [4.4 模型训练](#44__521)
        *   *   [4.4.1 初始化一些训练要用的参数](#441__523)
            *   [4.4.2 训练热身部分](#442__549)
            *   [4.4.3 开始训练](#443__585)
            *   [4.4.4 打印信息和保存模型](#444__723)
        *   [4.5 打印信息，释放内存](#45__851)
    *   [总结](#_900)

# 前引

`train.py`是YOLOv5的训练部分，通过这个文件，用来读取数据集、加载模型并训练。

**导航：**[YOLOv5-6.x源码分析 全流程记录](https://blog.csdn.net/weixin_51322383/article/details/130353834)

# 🚀YOLOv5-6.x源码分析（三）---- train.py

## 1\. 导包

```py
import argparse                     # 解析命令行
import math
import os
import random
import sys
import time
from copy import deepcopy           # 深度拷贝模块
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist        # 分布式训练模块
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler    # 学习率模块
from tqdm import tqdm

FILE = Path(__file__).resolve() # 解析该py文件路径
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val  # for end-of-epoch mAP 测试集
from models.experimental import attempt_load        # 实验性质的代码，包括MixConv2d、跨层权重Sum等
from models.yolo import Model                       # yolo的特定模块，包括BaseModel，DetectionModel，ClassificationModel，parse_model等
from utils.autoanchor import check_anchors          # 定义了自动生成锚框的方法
from utils.autobatch import check_train_batch_size  # 定义了自动生成批量大小的方法
from utils.callbacks import Callbacks           # 定义了回调函数，主要为logger服务
from utils.dataloaders import create_dataloader # dateset和dateloader定义代码
from utils.downloads import attempt_download    # 谷歌云盘内容下载
from utils.general import (LOGGER, check_amp, check_dataset, check_file, check_git_status, check_img_size,          # 定义了一些常用的工具函数，比如检查文件是否存在、检查图像大小是否符合要求、打印命令行参数等等
                           check_requirements, check_suffix, check_yaml, colorstr, get_latest_run, increment_path,
                           init_seeds, intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods,
                           one_cycle, print_args, print_mutation, strip_optimizer)
from utils.loggers import Loggers   # 日志打印
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.loss import ComputeLoss                  # 各种损失函数
from utils.metrics import fitness                   # 模型验证指标，包括ap，混淆矩阵等
from utils.plots import plot_evolve, plot_labels    # 定义了Annotator类，可以在图像上绘制矩形框和标注信息
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer, # 定义了一些与PyTorch有关的工具函数，比如选择设备、同步时间等
                               torch_distributed_zero_first)

# 分布式训练初始化
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1)) 
```

## 2\. 设置opt参数

```py
def parse_opt(known=False):
    """
    weights: 权重文件
    cfg: 模型配置文件 包括nc、depth_multiple、width_multiple、anchors、backbone、head等
    data: 数据集配置文件 包括path、train、val、test、nc、names、download等
    hyp: 初始超参文件
    epochs: 训练轮次
    batch-size: 训练批次大小
    img-size: 输入网络的图片分辨率大小
    resume: 断点续训, 从上次打断的训练结果处接着训练  默认False
    nosave: 不保存模型  默认False(保存)      True: only test final epoch
    notest: 是否只测试最后一轮 默认False  True: 只测试最后一轮   False: 每轮训练完都测试mAP
    workers: dataloader中的最大work数（线程个数）
    device: 训练的设备
    single-cls: 数据集是否只有一个类别 默认False

    rect: 训练集是否采用矩形训练  默认False
    noautoanchor: 不自动调整anchor 默认False(自动调整anchor)
    evolve: 是否进行超参进化 默认False
    multi-scale: 是否使用多尺度训练 默认False
    label-smoothing: 标签平滑增强 默认0.0不增强  要增强一般就设为0.1
    adam: 是否使用adam优化器 默认False(使用SGD)
    sync-bn: 是否使用跨卡同步bn操作,再DDP中使用  默认False
    linear-lr: 是否使用linear lr  线性学习率  默认False 使用cosine lr
    cache-image: 是否提前缓存图片到内存cache,以加速训练  默认False
    image-weights: 是否使用图片采用策略(selection img to training by class weights) 默认False 不使用

    bucket: 谷歌云盘bucket 一般用不到
    project: 训练结果保存的根目录 默认是runs/train
    name: 训练结果保存的目录 默认是exp  最终: runs/train/exp
    exist-ok: 如果文件存在就ok不存在就新建或increment name  默认False(默认文件都是不存在的)
    quad: dataloader取数据时, 是否使用collate_fn4代替collate_fn  默认False
    save_period: Log model after every "save_period" epoch    默认-1 不需要log model 信息
    artifact_alias: which version of dataset artifact to be stripped  默认lastest  貌似没用到这个参数？
    local_rank: rank为进程编号  -1且gpu=1时不进行分布式  -1且多块gpu使用DataParallel模式

    entity: wandb entity 默认None
    upload_dataset: 是否上传dataset到wandb tabel(将数据集作为交互式 dsviz表 在浏览器中查看、查询、筛选和分析数据集) 默认False
    bbox_interval: 设置界框图像记录间隔 Set bounding-box image logging interval for W&B 默认-1   opt.epochs // 10
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'weights/yolov5s.pt', help='initial weights path')    # 初始权重
    parser.add_argument('--cfg', type=str, default= ROOT / 'models/yolov5s.yaml', help='model.yaml path')                              # 训练模型文件
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')   # 数据集参数文件
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')    # 超参数设置
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=2, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)') # 图片大小
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')   # 断续训练
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')           # 设备选择
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='W&B: Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt 
```

一般使用的时候重点关注前面几个参数：`weights`、`cfg`、`data`、`epochs`、`batch-size`

## 3\. 执行main函数

### 3.1 检查分布式训练环境

```py
# Checks
    if RANK in {-1, 0}: # 不执行分布式训练：-1
        print_args(vars(opt))       # 打印参数；vars() 函数返回对象object的属性和属性值的字典对象。
        check_git_status() # 是否有搭建github仓库
        check_requirements(exclude=['thop']) # 检查是否做好依赖 
```

这一段是**检查分布式训练环境。**

### 3.2 判断是否断点续训

```py
# Resume  判断是否使用断点续训resume, 读取参数
    # 使用断点续训 就从last.pt中读取相关参数；不使用断点续训 就从文件中读取相关参数
    if opt.resume and not check_wandb_resume(opt) and not opt.evolve:  # resume an interrupted run     resume：从中断中恢复
        # 如果resume是True，则通过get_lastest_run()函数找到runs为文件夹中最近的权重文件last.pt
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        # 判断是否为文件，若不是文件抛出异常
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        # opt.yaml是训练时的命令行参数文件
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            # 超参数替换，将训练时的命令行参数加载进opt参数对象中
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
        # 打印断点续训信息
        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        # 不使用断点续训，就从文件中读取相关参数
        # check_file （utils/general.py）的作用为查找/下载文件 并返回该文件的路径
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        # 保存路径，根据increment_path生成目录
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)) 
```

*   **若使用断点续训**，则在`last.pt`中读取参数
*   **若不使用断点续训**，则在`opt.weight`读取参数

### 3.3 判断是否是分布式训练

```py
# DDP mode设置
    # 判断是否采用分布式训练 支持多机多卡、分布式训练
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:    # 进行多GPU训练
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo") 
```

DDP（Distributed Data Parallel）用于单机或多机的多GPU分布式训练，但是DDP只能在Linux系统下使用。这部分它会选择你是使用cpu还是gpu，假如你采用的是分布式训练的话，它就会额外执行下面的一些操作，我们这里一般不会用到分布式，所以也就没有执行什么东西。

### 3.4 判断是否进化训练

```py
# Train 不进化算法，正常训练
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)
        if WORLD_SIZE > 1 and RANK == 0:
            LOGGER.info('Destroying process group... ')
            dist.destroy_process_group() 
```

### 3.5 遗传进化算法

这段用的少，就不写了（有点懒）
大致就是超参进化训练，迭代300epoch，基本用不到。

## 4\. 执行train()函数

### 4.1 载入参数和配置信息

```py
# 解析参数
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    callbacks.run('on_pretrain_routine_start')

    # Directories 定义路径
    w = save_dir / 'weights'  # weights dir 结果保存的目录
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir  判断是否存在
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters	读取hyp(超参数)配置文件
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict  加载yaml的标准函数接口
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    if not evolve:
        with open(save_dir / 'hyp.yaml', 'w') as f: # 超参数
            yaml.safe_dump(hyp, f, sort_keys=False) # yaml.safe_dump()是将yaml文件序列化
        with open(save_dir / 'opt.yaml', 'w') as f: # 脚本文件的参数
            yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Loggers
    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # Config
    plots = not evolve and not opt.noplots  # create plots
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True) # 初始化随机种子，目的是同意训练策略可复现 general.py
    with torch_distributed_zero_first(LOCAL_RANK):  # 分布式相关
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset 
```

这一段就是将opt的参数解析一下，方便后面使用。

*   每次训练后，会产生两个模型，一个是`last.pt`，一个是`best.pt`。
*   加载超参。
*   加载日志信息。
*   加载其他参数。

> 小结：这部分代码就是解析各种yaml的参数＋创建训练权重目录和保存路径+ 读取超参数配置文件 + 设置保存参数保存路径 + 加载数据配置信息 + 加载日志信息(logger + wandb) + 加载其他参数(plots、cuda、nc、names、is_coco)

### 4.2 model

#### 4.2.1 加载y训练模型

```py
# ============================================== 1、model =================================================
    # Model 模型加载
    check_suffix(weights, '.pt')  # check weights 预训练权重
    pretrained = weights.endswith('.pt')    # 载入模型
    if pretrained:  # 预训练
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally 官方下载
        # 加载模型和参数
        # 这里加载到cpu是为了避免在加载一个模型检查点时GPU内存激增
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        '''
        这里有两种加载方式：1\. cfg；2\. ckpt['model']yaml
        区别在于是否用resume断点续训，如果resume则不加载anchor
        因为resume时，保存的模型会保存anchor,所以不需要加载，
        所以如果用户自定义了anchor，再加载预训练权重进行训练，会覆盖掉用户自定义的anchor
        '''
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        # 以下三行是获得anchor
        # 若cfg 或 hyp.get('anchors')不为空且不使用中断训练 exclude=['anchor'] 否则 exclude=[]
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        # 将预训练模型中的所有参数保存下来，赋值给csd
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        # 判断预训练参数和新创建的模型参数有多少是相同的
        # 筛选字典中的键值对，把exclude删除
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        # 载入模型权重
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        # 直接加载模型，ch为通道数
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        # Freeze 冻结哪些层数
    # 这里只是给了冻结权重层的一个例子, 但是作者并不建议冻结权重层, 训练全部层参数, 可以得到更好的性能, 当然也会更慢
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False
    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size}) 
```

这一段是加载模型，分为是否使用预训练权重模型。

*   **若没有采用预训练**，则直接调用`model.load_state_dict`加载模型。
*   **若采用预训练**，就会先去官网尝试下载`yolo`权重文件，加载权重文件；根据.yaml文件加载模型；**将该文件的参数提取出来**，并载入到新的模型里面，即创建模型成功。

最后，获取的`train_path`和`test_path`分别表示在`data.yaml`中训练数据集和测试数据集的地址。

> 其实这里的预训练，就是一种迁移学习。这样做可以加快训练速度。

#### 4.2.2 优化器

```py
# ============================================== 2、优化器 =================================================
    # nbs 标称的batch_size,模拟的batch_size 比如默认的话上面设置的opt.batch_size=16 -> nbs=64
    # 也就是模型梯度累计 64/16=4(accumulate) 次之后就更新一次模型 等于变相的扩大了batch_size
    # Optimizer
    nbs = 64  # nominal batch size
    """
    nbs = 64
    batchsize = 16
    accumulate = 64 / 16 = 4
    模型梯度累计accumulate次之后就更新一次模型 相当于使用更大batch_size
    """
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    # 根据accumulate设置权重衰减参数，防止过拟合
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay']) 
```

#### 4.2.3 学习率设置

```py
# ============================================== 3、学习率 ================================================
    # Scheduler
    if opt.cos_lr:
        # 使用余弦退火学习率
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        # 使用线性学习率
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)   学习率衰减 
```

这一段是学习率衰减方法。

*   **使用linear线性学习率**：通过线性插值的方式调整学习率
*   **使用One Cycle余弦退火学习率**：即周期性学习率调整中，周期被设置为1。在一周期策略中，最大学习率被设置为 LR Range test 中可以找到的最高值，最小学习率比最大学习率小几个数量级。这里默认one_cycle。

#### 4.2.4 训练前的最后准备

```py
# ---------------------------------------------- 训练前最后准备 ------------------------------------------------------
    # EMA 指数移动平均方法
    # EMA 设置ema（指数移动平均），考虑历史值对参数的影响，目的是为了收敛的曲线更加平滑
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume 使用预训练，将上次训练的模型的参数加载出来
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer # 选择优化器 并设置pg0(bn参数)的优化方式
        if ckpt['optimizer'] is not None:
            # 将预训练模型中的参数加载进优化器
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Epochs 加载训练的迭代次数
        start_epoch = ckpt['epoch'] + 1
        if resume:
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
        # 如果训练的轮数小于开始的轮数
        if epochs < start_epoch:
            # 打印日志 恢复训练
            LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            # 计算新的与训练轮数
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, csd

    # DP mode 多显卡 一般不用
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm 分布式训练
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()') 
```

这一段是训练前的最后一段代码，包括EMA+resume+迭代次数加载+DP单机多卡+SyncBatchNorm分布式训练

*   **EMA为指数加权平均或滑动平均**：其将前面模型训练权重，偏差进行保存，在本次训练过程中，假设为第n次，将第一次到第n-1次以指数权重进行加和，再加上本次的结果，且越远离第n次，指数系数越大，其所占的比重越小。
*   **断点续训**：将上次训练的模型参数提取出来，包括模型参数、epoch等，继续训练。

> **4.2 小节总结：**
> 
> 1.  **载入模型**：载入模型(预训练/不预训练) + 检查数据集 + 设置数据集路径参数(train_path、test_path) + 设置冻结层
> 2.  **优化器**：参数设置(nbs、hyp[‘weight_decay’])
> 3.  **学习率**：线性学习率 + one cycle学习率 + 实例化 scheduler
> 4.  **训练前最后准备**：EMA ＋断点续训+ 迭代次数的加载 + DP ＋SyncBatchNorm

### 4.3 数据加载

#### 4.3.1 创建数据集

```py
# ============================================== 4、数据加载 ===============================================
    # Trainloader 加载训练集数据
    '''
    返回一个训练数据加载器，一个数据集对象：
    训练数据加载器是一个可迭代的对象，可以通过for循环加载1个batch_size的数据
    数据集对象包括数据集的一些参数，包括所有标签值、所有的训练数据路径、每张图片的尺寸等等
    '''
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True)
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    nb = len(train_loader)  # number of batches
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0 加载验证集数据
    if RANK in {-1, 0}: # 加载验证集数据加载器
        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]

        if not resume:
            # 统计dataset的label信息
            labels = np.concatenate(dataset.labels, 0)
            # c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1\.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:   # 画出标签信息
                plot_labels(labels, names, save_dir) 
```

这一段是创建数据集，通过`create_dataloader`获得两个对象，一个是`train_loader`，一个是`dataset`

*   **train_loader是训练数据加载器**，可以通过for循环加载1个batch的数据
*   **dataset是数据集对象**，包裹路径、图片大小、标签等

将所有样本的标签拼接到一起，统计后做可视化，同时获得所有样本的类别，根据上面的统计对所有样本的类别，中心点xy位置，长宽wh做可视化。

#### 4.3.2 计算anchor

```py
# Anchors
            # 计算默认锚框anchor与数据集标签框的高宽比
            # 标签的高h宽w与anchor的高h_a宽h_b的比值 即h/h_a, w/w_a都要在(1/hyp['anchor_t'], hyp['anchor_t'])是可以接受的
            # 如果bpr小于98%，则根据k-mean算法聚类新的锚框
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

        callbacks.run('on_pretrain_routine_end')

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model) 
```

这一段是检查anchors + 将model调整为半精度。

### 4.4 模型训练

#### 4.4.1 初始化一些训练要用的参数

```py
# ============================================== 5、训练 ===============================================
    # Model attributes   设置/初始化一些训练要用的参数
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    # box为预测框的损失
    hyp['box'] *= 3 / nl  # scale to layers 
    # cls为分类的损失
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    # obj为置信度的损失
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    # 标签平滑
    hyp['label_smoothing'] = opt.label_smoothing
    # 类别数，将检测的类别数保存到model里面
    model.nc = nc  # attach number of classes to model
    # 模型的超参数，将超参保存到model里面
    model.hyp = hyp  # attach hyperparameters to model
    # 从训练的样本标签得到类别权重，将类别权重保存到模型
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    # 获取类别的名字，将分类标签保存至模型
    model.names = names # 获取类别名 
```

这段代码主要是根据自己数据集，将一些参数保存到模型里面。在后面训练或者以后使用该模型的时候，就会需要这些参数。

#### 4.4.2 训练热身部分

```py
'''
    训练热身部分
    '''
    # Start training
    t0 = time.time()
    # 获取热身迭代的次数
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    # 初始化maps(每个类别的map)和results
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    # 设置学习率衰减所进行到的轮次，即使打断训练，使用resume接着训练也能正常衔接之前的训练进行学习率衰减
    scheduler.last_epoch = start_epoch - 1  # do not move
    # 设置amp混合精度训练    GradScaler + autocast
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    # 初始化损失函数
    compute_loss = ComputeLoss(model)  # init loss class 定义损失函数
    callbacks.run('on_train_start')
    # 打印日志信息
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...') 
```

训练前的热身，**做一些参数的初始化、损失函数的初始化等**

训练热身一个方法为warmup，该方法主要是在训练前期使用较小的学习率，经过几轮迭代后使用较大的学习率加速收敛，在快结束时，再降低学习率，使模型逼近最优。（**以较低学习率逐渐增大至较高学习率的方式**，学习率变化：上升，平稳，下降）

另一个方法是早停，若训练一定的epochs后，模型效果未提升，则提前停止训练。判断模型的效果为fitness，**fitness为0.1乘mAP@0.5加上0.9乘mAP@0.5:0.95**。

#### 4.4.3 开始训练

```py
# 开始训练
    # start training -----------------------------------------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        '''
        告诉模型现在是训练阶段 因为BN层、DropOut层、两阶段目标检测模型等
        训练阶段阶段和预测阶段进行的运算是不同的，所以要将二者分开
        model.eval()指的是预测推断阶段
        '''
        callbacks.run('on_train_epoch_start')
        model.train()

        # Update image weights (optional)  并不一定好  默认是False的
        # 如果为True 进行图片采样策略(按数据集各类别权重采样)
        if opt.image_weights:
            # 根据前面初始化的图片采样权重model.class_weights（每个类别的权重 频率高的权重小）以及maps配合每张图片包含的类别数,若哪一类的精确度不高，则会被分配一个较高的权重
            # 通过rando.choices生成图片索引indices从而进行采用 （作者自己写的采样策略，效果不一定ok）
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            # 将计算出的权重换算到图片的维度，将类别的权重换算为图片的权重
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            # 通过random.choices生成图片索引indices从而进行采样，这时图像会包含一些难识别的样本
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx 
```

这段代码首先告诉模型，**进入训练阶段**，即`model.train()`。

*   `model.train()`时，`BatchNormalization`的参数会根据输入更新，Dropout使输入以p的概率参与计算
*   `model.eval()`时，`BatchNormalization`的参数则会固定，与保存的值一致，Dropout不起作用，所有输入参与计算

然后是**更新图片的权重**。训练的时候一些类比准确率不高，那么在下一轮的时候，就**会为这个类产生一些权重高的图片，以这种方式来增加识别率低的类别的数据量。**

```py
mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            # DDP模式打乱数据，并且dpp.sampler的随机采样数据是基于epoch+seed作为随机种子，每次epoch不同，随机种子不同
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
        if RANK in {-1, 0}:
            # 进度条，方便展示信息
            pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        # 梯度清零
        optimizer.zero_grad() 
```

**分布式训练的设置 + 训练时终端的显示**；

**最后将优化器中的参数梯度清零**。

```py
 for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run('on_train_batch_start')
            # ni: 计算当前迭代次数 iteration
            ni = i + nb * epoch  # number integrated batches (since train start)
            # 将图片载入设备，并做归一化 0~1
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            # 热身训练（前nw次迭代）热身训练迭代的次数iteration范围[1:nw]  选取较小的accumulate，学习率以及momentum,慢慢的训练
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    # bias的学习率从0.1下降到基准学习率lr*lf(epoch) 其他的参数学习率增加到lr*lf(epoch)
                    # lf为上面设置的余弦退火的衰减函数
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale 多尺度训练   从[imgsz*0.5, imgsz*1.5+gs]间随机选取一个尺寸(32的倍数)作为当前batch的尺寸送入模型开始训练
            # imgsz: 默认训练尺寸   gs: 模型最大stride=32   [32 16 8]
            if opt.multi_scale:	# 随机改变图片的尺寸
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    # 下采样
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False) 
```

这部分代码主要做分批加载数据，因为前面讲了，train_loader是一个可迭代对象，一个对象包含了一个bs的数据。

在分批加载数据的时候，用ni计算当前迭代的次数，并对图片进行归一化。

然后是热身训练，用ni和nw做比较，一开始只采用较小的学习率，逐渐上升。对于bias参数组的学习率策略是从0.1逐渐降低至初始学习率，其余参数组则从0开始逐渐增长至初始学习率。

最后是多尺度训练。

```py
# Forward  混合精度训练 开启autocast的上下文
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward
                # 计算损失，包括分类损失，置信度损失和框的回归损失
                # loss为总损失值  loss_items为一个元组，包含分类损失、置信度损失、框的回归损失和总损失
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    # 采用DDP训练,平均不同gpu之间的梯度
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    # 如果采用collate_fn4取出mosaic4数据loss也要翻4倍
                    loss *= 4.

            # Backward  反向传播  scale为使用自动混合精度运算    将梯度放大防止梯度的underflow（amp混合精度训练）
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            # 模型反向传播accumulate次（iterations）后再根据累计的梯度更新一次参数
            # Optimize 模型会对多批数据进行累积，只有达到累计次数的时候才会更新参数，在还没有达到累积次数时 loss会不断的叠加 不会被新的反传替代
            if ni - last_opt_step >= accumulate:
                '''
                 scaler.step()首先把梯度的值unscale回来，
                 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
                 否则，忽略step调用，从而保证权重不更新（不被破坏）
                '''
                scaler.unscale_(optimizer)  # unscale gradients 擦
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                # 更新参数
                scaler.update()
                # 梯度清零
                optimizer.zero_grad()
                if ema:
                    # 更新ema
                    ema.update(model)
                last_opt_step = ni 
```

**这段代码是正向传播、反向传播和梯度更新。**

一开始将图片输入模型，进行正向传播，得到结果。将这个结果和label通过损失函数求出损失。

通过损失，**进行反向传播，求出每层梯度**。

最后利用`optimizer.step`**更新参数**。但是要注意，在更新参数时这里有一个不一样的地方，并不会在每次反向传播时更新参数，而是做一定的累积，**反向传播的结果并不会顶替上一次反向传播结果，而是做一个累积。完成一次积累后，再将梯度清零，方便下一次清零**。这样做是为了以更小的batch_size实现更高的batch_size效果。

#### 4.4.4 打印信息和保存模型

```py
 if RANK in {-1, 0}:
            # mAP
            # 将model中的属性赋值给ema
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            # 判断当前epoch是否是最后一轮
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            # 是否只测最后一轮
            if not noval or final_epoch:  # Calculate mAP
                """
                测试使用的是ema（指数移动平均 对模型的参数做平均）的模型
                       results: [1] Precision 所有类别的平均precision(最大f1时)
                                [1] Recall 所有类别的平均recall
                                [1] map@0.5 所有类别的平均mAP@0.5
                                [1] map@0.5:0.95 所有类别的平均mAP@0.5:0.95
                                [1] box_loss 验证集回归损失, obj_loss 验证集置信度损失, cls_loss 验证集分类损失
                       maps: [80] 所有类别的mAP@0.5:0.95
                """
                results, maps, _ = val.run(data_dict,   # 数据集地址
                                           batch_size=batch_size // WORLD_SIZE * 2,
                                           imgsz=imgsz,
                                           half=amp,
                                           model=ema.ema,
                                           single_cls=single_cls,
                                           dataloader=val_loader,   # 验证集loader
                                           save_dir=save_dir,
                                           plots=False, # 是否可视化
                                           callbacks=callbacks,
                                           compute_loss=compute_loss)   # 损失函数（train）

            # Update best mAP 更新best_fitness
            # fi: [P, R, mAP@.5, mAP@.5-.95]的一个加权值 = 0.1*mAP@.5 + 0.9*mAP@.5-.95
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            # 若当前的fitness大于最佳的fitness
            if fi > best_fitness:
                best_fitness = fi   # 将最佳fitness更新为当前fitness
            # 保存验证结果
            log_vals = list(mloss) + list(results) + lr
            # 记录验证数据
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi) 
```

这段代码是得到results，mAP等评价指标。详细可看[val.py](https://blog.csdn.net/weixin_51322383/article/details/130317934)。

首先判断是否训练结束，若选择每轮验证或者当前已经是最后一轮，才做验证。一般都是训练完毕后做验证。

然后计算出最好的模型。这里“最好”的评判标准即为fitness。**fi: [P, R, mAP@.5, mAP@.5-.95]的一个加权值 = 0.1*mAP@.5 + 0.9*mAP@.5-.95**，在评判标准中，更加强调**mAP@0.5:0.95**的作用。mAP@0.5:0.95大代表模型在多个IOU阈值的情况下，都可以较好的识别物体。

```py
# Save model
            """
            保存带checkpoint的模型用于inference或resuming training
            保存模型, 还保存了epoch, results, optimizer等信息
            optimizer将不会在最后一轮完成后保存
            model保存的是EMA的模型
            """
            if (not nosave) or (final_epoch and not evolve):  # if save
                # 将当前训练过程中的所有参数赋值给ckpt
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                # 记录保存模型时的日志
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping 停止单卡训练
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks 
```

**终于要结束了！**

最后保存模型，将所有参数给`ckpt`。

然后判断这个模型的`fitness`是否为最佳，如果是，就保存，保存后将变量从内存删除。

> 4.4 模型训练小结：
> 
> 1.  初始化训练需要的模型参数：设置一些超参、获取模型的一些参数、names等
> 2.  热身：热身迭代的次数iterationsnw、last|opt|step、初始化results、学习率衰减到进行的轮次、设置amp混合精度训练scaler、初始化损失函数
> 3.  开始训练：图片采样策略 + Warmup热身训练 + multi_scale多尺度训练 + amp混合精度训练 + accumulate 梯度更新策略+ 打印训练相关信息(包括当前epoch、显存、损失(box、obj、cls、total)＋当前batch的target的数量和图片的size等 + 调整学习率、scheduler.step() 、emp val.run()得到results, maps相关信息
> 4.  保存模型：将结果写入results.txt中、wandb_logger、Update best mAP 以加权mAP fitness为衡量标准＋保存模型

### 4.5 打印信息，释放内存

```py
# end training -----------------------------------------------------------------------------------------------------
    # 打印一些信息
    if RANK in {-1, 0}:
        # 训练停止 向控制台输出信息
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        # 可视化训练结果
        for f in last, best:
            if f.exists():  # 在验证集上再跑一次
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    # 把最好的模型在验证集上面跑一次 并绘图
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = val.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools results at 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss)  # val best model with plots
                    if is_coco: # coco数据集才用得到
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)
        # 记录训练终止时的日志
        callbacks.run('on_train_end', last, best, plots, epoch, results)

    # 释放现存
    torch.cuda.empty_cache()
    return results 
```

**打印信息，释放内存**

首先训练停止的时候**打印信息**，比如各种评价指标、训练时间、等等

然后把best.pt取出，**用这个模型跑`val.run()`**，再把结果保存下来。

**最后释放显存**。

happy end！！！

## 总结

总体代码比较简单，主要是 数据集创建+模型+学习率+优化器+训练 五个步骤。