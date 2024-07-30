<!--yml
category: 游戏
date: 2023-09-17 14:44:55
-->

# YOLOv5-6.x源码分析（四）---- 模型搭建之yolo.py

> 来源：[https://blog.csdn.net/weixin_51322383/article/details/130353750](https://blog.csdn.net/weixin_51322383/article/details/130353750)

### 文章目录

*   [前引](#_1)
*   [🚀YOLOv5-6.x源码分析（四）---- yolo.py](#YOLOv56x_yolopy_6)
*   *   [1\. 导入需要的包](#1__7)
    *   [2\. parse_model](#2_parse_model_54)
    *   *   [2.1 获取对应参数](#21__56)
        *   [2.2 搭建网络的准备](#22__81)
        *   [2.3 更新args，计算c2](#23_argsc2_99)
        *   [2.4 使用当前层的参数搭建当前层](#24__124)
        *   [2.5 打印并保存layer](#25_layer_162)
    *   [3\. Detect](#3_Detect_189)
    *   *   [3.1 参数初始化](#31__201)
        *   [3.2 前向传播](#32__234)
        *   [3.3 相对坐标转换到grid绝对坐标系](#33_grid_303)
    *   [4\. Model](#4_Model_331)
    *   *   [4.1 __init__](#41___init___335)
        *   [4.2 数据增强](#42__428)
        *   *   [4.2.1 前向传播](#421__429)
            *   [4.2.2 采用数据增强的forward](#422_forward_445)
            *   [4.2.3 不采用数据增强的forward](#423_forward_482)
            *   [4.2.4 _descale_pred():将推理结果恢复到原图尺寸](#424__descale_pred_526)
            *   [4.2.5 _clip_augmented()：TTA的时候对图片进行裁剪](#425__clip_augmentedTTA_562)
            *   [4.2.6 打印日志信息](#426__581)
            *   [4.2.7 bias信息](#427_bias_609)
            *   [4.2.8 fuse()：融合Conv2d+BN](#428_fuseConv2dBN_641)
    *   [5\. 主函数](#5__665)
*   [结束语](#_673)

# 前引

**这个脚本文件是YOLOv5模型搭建的部分，非常重要**。这部分代码主要有三个部分：`parse_model`、`Detect`、`Model`。

**导航：**[YOLOv5-6.x源码分析 全流程记录](https://blog.csdn.net/weixin_51322383/article/details/130353834)

* * *

# 🚀YOLOv5-6.x源码分析（四）---- yolo.py

## 1\. 导入需要的包

```py
import argparse             # 解析命令行参数模块
import os                   # # sys系统模块 包含了与Python解释器和它的环境有关的函数
import platform
import sys                  # 系统模块，包括与Python解释器和它环境相关的函数
from copy import deepcopy   # 数据拷贝模块 深拷贝
from pathlib import Path    # Path将str转换为Path对象 使字符串路径易于操作的模块 
```

**导入Python已安装好的库**。

```py
FILE = Path(__file__).resolve() #  # __file__指的是当前文件(即yolo.py),FILE最终保存着当前文件的绝对路径 E:\ComputerScience\DeepLearning\yolov5
ROOT = FILE.parents[1]  # YOLOv5 root directory 父目录 E:\ComputerScience\DeepLearning\yolov5
if str(ROOT) not in sys.path:   # sys.path即当前python环境可以运行的路径,假如当前项目不在该路径中,就无法运行其中的模块,所以就需要加载路径
    sys.path.append(str(ROOT))  # add ROOT to PATH 把ROOT添加到运行路径上
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative 绝对路径转化为相对路径 
```

这段代码会获取**当前文件的绝对路径**，并使用Path库将其转换为Path对象。

这一部分的主要作用有两个：

*   将当前项目添加到系统路径上，以使得项目中的模块可以调用。
*   将当前项目的相对路径保存在ROOT中，便于寻找项目中的文件。

```py
from models.common import * # 网络结构，基本模块
from models.experimental import *   # 导入在线下载模块
from utils.autoanchor import check_anchor_order # 导入检查anchors合法性的函数
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args # 定义了一些常用的工具函数
from utils.plots import feature_visualization   # 定义了Annotator类，可以在图像上绘制矩形框和标注信息
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)   # 定义了一些与PyTorch有关的工具函数

# 导入thop包 用于计算FLOPs
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None 
```

这部分是**自定义模块**。

## 2\. parse_model

### 2.1 获取对应参数

`parse_model`主要就是通过读取yaml文件的配置，到`common.py`中找到对应的模块，然后组成一个完整的模型解析文件（字典形式），并搭建网络结构。

在对YOLOv5进行tricks魔改的时候，也主要是要修改这个模块，将你自己写的模块放到这里来。

```py
def parse_model(d, ch):  # model_dict, input_channels(3)列表中只有3
    '''===================1\. 获取对应参数============================'''
    # 打印列标签
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")    # 打印信息
    # 获取yaml的一些参数，如anchor大小、nc数目、宽度、深度
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    # na: 每组先验框包含的先验框数
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors chahor数量
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5) 输出通道数（80+5）*3=255 
```

这一段是获取yaml文件的一些参数，其中`d`就是传入的yaml文件（以字典形式传入），读取的形式也就如代码所示，这样就将yaml的数据给读取进来了。

*   `ch`：记录模型每一层的输出channel，初始ch=[3]，后面会删除

*   `na`：anchor数量

*   `no`：根据anchor数量推断的输出维度

### 2.2 搭建网络的准备

```py
'''===================2\. 搭建网络前的准备 ============================'''
layers, save, c2 = [], [], ch[-1]  # layers网络单元列表, savelist哪些层需要保存, ch out输出通道数
# from(当前层输入来自哪些层), number(当前层次数 初定), module(当前层类别), args(当前层类参数 初定)
for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
    # eval(string) 得到当前层的真实类名 例如: m= Focus -> <class 'models.common.Focus'>
    m = eval(m) if isinstance(m, str) else m  # eval strings    eval推断是什么类型
    for j, a in enumerate(args):
        with contextlib.suppress(NameError):
            args[j] = eval(a) if isinstance(a, str) else a  # eval strings 推断成int 
```

这一段代码是遍历yaml文件中的`backbone`层和`head`层，其实就是遍历整个网络，以前官方的网络参数文件好像还有`neck`，现在只有`backbone`和`head`了。

这里用到了一个函数`eval()`，主要作用是将str转化为有效的表达式，并返回执行的结果。比如说， m= Focus -> <class ‘models.common.Focus’>。即将str变成对应common中的类模块。

### 2.3 更新args，计算c2

```py
# ------------------- 3\. 更新当前层的args（参数）,计算c2（当前层的输出channel） -------------------
# depth gain 控制深度  如v5s: n*0.33   n: 当前模块的次数(间接控制深度)
n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
if m in (Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
         BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x):
    c1, c2 = ch[f], args[0] # c1输入，c2输出  ch记录着所有层的输出channel数
    if c2 != no:  # if not output 判断c2是否是最终输出255
        c2 = make_divisible(c2 * gw, 8) # gw为宽度，调整为8的倍数（对gpu计算更友好） 
```

这段代码首先将网络深度改变，即乘以`gd`参数取整。例如，对于yolo5s来讲，gd为0.33，那么就是n*0.33，也就是把默认的深度缩放为原来的1/3。

最后，如果输出通道不等于255即Detect层的输出通道， 则将通道数乘上`gw`，并调整为8的倍数。通过函数`make_divisible`来实现。主要是这样对gpu的计算更加友好。

`make_divisible`的代码如下：

```py
 # 使得X能够被divisor整除
     def make_divisible(x, divisor):
         return math.ceil(x / divisor) * divisor 
```

### 2.4 使用当前层的参数搭建当前层

```py
'''=================== 4.使用当前层的参数搭建当前层 ============================'''
            # 在初始arg的基础上更新 加入当前层的输入channel并更新当前层
            args = [c1, c2, *args[1:]]  # c1 c2 args的后3个参数 拼接起来
            # 如果当前层是BottleneckCSP/C3/C3TR, 则需要在args中加入bottleneck的个数
            if m in [BottleneckCSP, C3, C3TR, C3Ghost, C3x]:    # 如果为C3层
                args.insert(2, n)  # number of repeats  额外将n加入
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]  # BN层只需要返回上一层的输出channel
        elif m is Concat:
            c2 = sum(ch[x] for x in f)   # Concat层则将f中所有的输出累加得到这层的输出channel
        elif m is Detect:
            args.append([ch[x] for x in f]) # 在args中加入三个Detect层的输出channel
            if isinstance(args[1], int):  # number of anchors    几乎不执行
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract: # 不怎么用
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:   # 不怎么用
            c2 = ch[f] // args[0] ** 2
        # elif m in [BiFPN_Add2, BiFPN_Add3]:
        #     c2 = max([ch[x] for x in f])
        # elif m is Concat_bifpn:
        #     c2 = max([ch[x] for x in f])
        else:
            c2 = ch[f]      # args不变 
```

**根据参数重新搭建当前层**。

到这之后，args里面包括的参数有`c1`输入channel、`c2`输出channel。只有`BottleneckCSP`和`C3`这两种`module`会根据深度参数n调整该模块的重复迭加次数，加入n到args后。

*   如果是BN层：只需要返回上一层的输出channel，通道数不变
*   如果是Concat：将f的所有层的通道数加起来，作为c2输出channel
*   如果是Detect：对应检测头部分，这部分下一小节讲。

### 2.5 打印并保存layer

```py
'''===================5.打印和保存layers信息============================'''
        # *args表示接收任意个数量的参数，调用时会将实际参数打包成一个元组传入实参
        # m_: 得到当前层module  如果n>1就创建多个m(当前层结构), 如果n=1就创建一个m
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        # 打印当前层结构的一些基本信息
        t = str(m)[8:-2].replace('__main__.', '')  # module type 用''替换掉__main__
        # 计算这一层的参数量
        np = sum(x.numel() for x in m_.parameters())  # number params 统计第0层参数量
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        # 把所有层结构中from不是-1的值记下  [6, 4, 14, 10, 17, 20, 23]
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist 统计哪些层需要保存（需要concat的层）
        # 将当前层结构module加入layers中
        layers.append(m_)
        if i == 0:
            ch = [] # 去除输入channel [3]
        ch.append(c2)   # [32] [32,64] [32,64,64] 按照-1取出上一层的输出通道
    return nn.Sequential(*layers), sorted(save) # [6,4,14,10,17,20,23] 
```

这段代码主要是**打印当前层结构的一些基本信息并保存。**

把需要保存的层保存到save里（比如Concat），把构建的模块保存到layers里，把该层的输出通道数写入ch列表里。待全部循环结束后再构建成模型。

## 3\. Detect

**Detect是YOLO最后一层，对应着yaml的最后一行**

`[[17, 20, 23], 1, Detect, [nc, anchors]]`like this.

其中，nc为分类数，anchors为先验框，在yaml文件的最上方定义过了。

在`parse_model`中，会根据 `from` 参数，找到对应网络层的输出通道数。检测每一层，如果是`Detect`，传参给`Detect`模块，`Detect`模块是用来构建`Detect`层的，将输入`feature map` 通过一个卷积操作和公式计算到我们想要的`shape`，为后面的计算损失或者`NMS`作准备。

* * *

### 3.1 参数初始化

```py
# Detect模块是用来构建Detect层的，将输入feature map 通过一个卷积操作和公式计算到我们想要的shape，为后面的计算损失或者NMS作准备。
class Detect(nn.Module):
    stride = None  # 特征图的缩放步长
    onnx_dynamic = False  # ONNX动态量化
    export = False  # export mode

    '''===================1.参数初始化============================'''
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor 一个anchor的输出数
        self.nl = len(anchors)  # number of detection layers = 3 预测层数
        self.na = len(anchors[0]) // 2  # number of anchors = 3
        self.grid = [torch.zeros(1)] * self.nl  # init grid 表示初始化anchor_grid列表大小，空列表
        # 注册常量anchor，并将预选框（尺寸）以数对形式存入，并命名为anchors
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        # 模型中需要保存的参数一般有两种：一种是反向传播需要被optimizer更新的，称为parameter; 另一种不要被更新称为buffer
        # buffer的参数更新是在forward中，而optim.step只能更新nn.parameter类型的参数
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)   反向传播不需要被optimizer更新，称之为buffer
        # output conv 3个输出层最后的1乘1卷积
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv 1*1卷积 得到预测后的值（坐标、分类概率。。）
        # inplace: 一般都是True，默认不使用AWS，Inferentia加速
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        # 包含了三个信息pred_box [x,y,w,h] pred_conf[confidence] pre_cls[cls0,cls1,cls2,...clsn] 
```

**这里是获取传入的一些参数**。

这里注意，`grid`参数，他的含义是格子坐标系，比如左上角为`(1,1)`,右下角为`(input.w/stride,input.h/stride)`

### 3.2 前向传播

```py
'''===================2.前向传播============================'''
def forward(self, x):
    """
        :return train: 一个tensor list 存放三个元素   [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                       分别是 [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
                inference: 0 [1, 19200+4800+1200, 25] = [bs, anchor_num*grid_w*grid_h, xywh+c+20classes]
                           1 一个tensor list 存放三个元素 [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                             [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
        """
    z = []  # inference output
    for i in range(self.nl):    # 预测层数遍历
        x[i] = self.m[i](x[i])  # conv 第i个预测层做第i个m中的1*1卷积
        bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        # 维度重排列: bs, 先验框组数, 检测框行数, 检测框列数, 属性数 + 分类数
        x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()  # 把no调到最后（25个值）

        # 前向传播时需要将坐标转换到grid绝对坐标系中
        if not self.training:  # inference 推理
            # 构造网格
            # 因为推理返回的不是归一化后的网格偏移量 需要再加上网格的位置 得到最终的推理坐标 再送入nms
            # 所以这里构建网格就是为了记录每个grid的网格坐标 方面后面使用
            '''
            生成坐标系
            grid[i].shape = [1,1,ny,nx,2]
                            [[[[1,1],[1,2],...[1,nx]],
                            [[2,1],[2,2],...[2,nx]],
                            ...,
                            [[ny,1],[ny,2],...[ny,nx]]]]
            '''
            # 换输入后重新设定锚框
            if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                # 加载网格点坐标 先验框尺寸
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)  # anchor_grid构造网格

            # 按损失函数的回归方式来转换坐标
            y = x[i].sigmoid()
            if self.inplace:    # 最后一个维度是85，其前四个元素[tx,ty,tw,th]经线性回归计算，得到[bx,by,bw,bh]
                # grid: 位置基准 或者理解为 cell的预测初始位置，而y[..., 0:2]是作为在grid坐标基础上的位置偏移
                y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy坐标信息
                # anchor_grid: 预测框基准 或者理解为 预测框的初始位置，而 y[..., 2:4]是作为预测框位置的调整
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh坐标信息
            else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                # stride: 是一个grid cell的实际尺寸
                # 经过sigmoid, 值范围变成了(0-1),下一行代码将值变成范围（-0.5，1.5）
                xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                # 范围变成(0-4)倍，设置为4倍的原因是下层的感受野是上层的2倍
                # 因下层注重检测大目标，相对比上层而言，计算量更小，4倍是一个折中的选择
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf), 4)
            # 存储每个特征图检测框的信息
            z.append(y.view(bs, -1, self.no))   # 预测框坐标信息
    # 训练阶段直接返回x
    # 预测阶段返回3个特征图拼接的结果
    return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)  # 预测框坐标 obj,cls 
```

这段代码挺难的，坐标转换那看了半天没看懂，主要是下面的`_make_grid`没看懂。

首先进行for循环，因为detect会有三层，所以for循环就会执行3次，每次i的循环，产生一个z。

然后是维度重排列：**(n, 255, _, _) -> (n, 3, nc+5, ny, nx) -> (n, 3, ny, nx, nc+5)**，三个detect分别预测了**80*80、40*40、20*20**次。（到这里都没问题）

接着**构造网络**，因为推理返回的不是归一化后的网格偏移量，需要再加上网格的位置，得到最终的推理坐标，再送入nms。所以这里构建网格就是为了记录每个grid的网格坐标 方面后面使用。

最后**按损失函数的回归方式来转换坐标**，利用sigmoid激活函数计算定位参数，cat(dim=-1)为直接拼接。注意： **训练阶段直接返回x ，而预测阶段返回3个特征图拼接的结果** 。

### 3.3 相对坐标转换到grid绝对坐标系

```py
'''===================3.相对坐标转换到grid绝对坐标系============================'''
def _make_grid(self, nx=20, ny=20, i=0):
    d = self.anchors[i].device
    t = self.anchors[i].dtype
    shape = 1, self.na, ny, nx, 2  # grid shape
    # 划分单元网格
    y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
    if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
        yv, xv = torch.meshgrid(y, x, indexing='ij')
    else:
        yv, xv = torch.meshgrid(y, x)
    # grid --> (20, 20, 2), 复制成3倍，因为是三个框 -> (3, 20, 20, 2)
    grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
    anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
    return grid, anchor_grid 
```

**首先构造网格标尺坐标**

*   `indexing='ij'`：表示的是i是同一行，j表示同一列
*   `indexing='xy'`：表示的是x是同一列，y表示同一行

grid复制3倍，因为有3个框。`torch.meshgrid()`功能是生成网格，可以用于生成坐标。可看这篇博客[torch.meshgrid（）函数解析](https://blog.csdn.net/weixin_39504171/article/details/106356977)

`anchor_grid`是每个anchor的宽高。`anchor_grid = (self.anchors[i] * self.stride[i])`。因为外面已经把anchors除以了下采样率，所以这里要再乘`self.stride[i]`。

## 4\. Model

**`Model`类是整个模型的搭建模块。**

### 4.1 **init**

```py
class Model(nn.Module):
    # YOLOv5 model
    '''===================1.__init__函数==========================='''
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        """
                :params cfg:模型配置文件
                :params ch: input img channels 一般是3 RGB文件
                :params nc: number of classes 数据集的类别个数
                :anchors: 一般是None
                """
        super().__init__()  # 父类构造方法
        if isinstance(cfg, dict):   # 判断cfg是不是字典
            self.yaml = cfg  # model dict
        else:  # is *.yaml 一般执行这里
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name # 获取文件名
            # 如果配置文件中有中文，打开时要加encoding参数
            with open(cfg, encoding='ascii', errors='ignore') as f:
                # 将yaml文件加载问字典
                self.yaml = yaml.safe_load(f)  # model dict 以字典形式存放 
```

* * *

**接着开始搭建模型**。

```py
# Define model 搭建模型
ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels 找文件中的ch，若找不到以init中的ch代替，即3
# 设置类别数 一般不执行, 因为nc=self.yaml['nc']恒成立
if nc and nc != self.yaml['nc']:
    # 打印出来，给出提示
    LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
    self.yaml['nc'] = nc  # override yaml value
# 重写anchor，一般不执行, 因为传进来的anchors一般都是None
if anchors:
    LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
    self.yaml['anchors'] = round(anchors)  # override yaml value
# 创建网络模型
# self.model: 初始化的整个网络模型(包括Detect层结构)
# self.save: 所有层结构中from不等于-1的序号，并排好序  [4, 6, 10, 14, 17, 20, 23]
self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist 搭建网络的每一层
self.names = [str(i) for i in range(self.yaml['nc'])]  # default names 初始化name参数，给每一个类附一个类名
# inplace指的是原地操作 如x+=1 有利于节约内存
# self.inplace=True  默认True  不使用加速推理
self.inplace = self.yaml.get('inplace', True)   # 加载inplace关键字，若没有则返回true 
```

**通过`parse_model`进行解析和建立模型。**

**到此为止，我们的整个yolo模型的网络架构就搭建完毕了**。

* * *

最后来计算图像从输入到输出的缩放倍数和anchor在head上的大小

```py
# Build strides, anchors
# 获取Detect模块的stride(相对输入图像的下采样率)和anchors在当前Detect输出的feature map的尺度
m = self.model[-1]  # Detect()
# 判断最后一层是不是Detect层
if isinstance(m, Detect):
    # 定义一个256 * 256大小的输入
    s = 256  # 2x min stride
    m.inplace = self.inplace
    # 计算三个feature map下采样的倍率  [8, 16, 32]
    # 保存特征层的stride,并且将anchor处理成相对于特征层的格式
    m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward [8,16,32]
    check_anchor_order(m)  # must be in pixel-space (not grid-space) 检查anchor的顺序对不对 低层用低层anchor
    # 原始定义的anchor是原始图片上的像素值，要将其缩放至特征图的大小  anchor大小计算，例如[10,13]->[1.25,1.625]
    m.anchors /= m.stride.view(-1, 1, 1)
    self.stride = m.stride  # 保存步长
    # 初始化bias
    self._initialize_biases()  # only run once 初始化偏置

# Init weights, biases
initialize_weights(self)    # 初始化权重
self.info()
LOGGER.info('') 
```

**主要步骤：**

1.  获取网络的最后一层`Detect`
2.  定义一个256*256的输入
3.  将[1, ch, 256, 256]大小的tensor进行一次向前传播，得到3层的输出，用输入大小256分别除以输出大小得到每一层的下采样倍数stride
4.  分别用最初的anchor大小除以stride将anchor线性缩放到对应层上

### 4.2 数据增强

#### 4.2.1 前向传播

```py
def forward(self, x, augment=False, profile=False, visualize=False):
    # 是否在测试时也用数据增强
    if augment:
        # 增强训练，对数据采取了一系列操作
        return self._forward_augment(x)  # augmented inference, None
    # 默认执行，正常前向传播
    return self._forward_once(x, profile, visualize)  # single-scale inference, train 
```

**这里有一个分支，关于是否在测试时采用数据增强**。

* * *

#### 4.2.2 采用数据增强的forward

```py
def _forward_augment(self, x):
    # 图像高宽
    img_size = x.shape[-2:]  # height, width
    s = [1, 0.83, 0.67]  # scales 规模
    # flip是翻转，这里的参数表示沿着哪个轴翻转
    f = [None, 3, None]  # flips (2-ud, 3-lr)
    y = []  # outputs
    for si, fi in zip(s, f):
        # scale_img函数的作用就是根据传入的参数缩放和翻转图像
        xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
        # 模型前向传播
        yi = self._forward_once(xi)[0]  # forward
        # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
        #  恢复数据增强前的模样
        yi = self._descale_pred(yi, fi, si, img_size)
        y.append(yi)
    # 对不同尺寸进行不同程度的筛选
    y = self._clip_augmented(y)  # clip augmented tails
    return torch.cat(y, 1), None  # augmented inference, train 
```

在推理的时候做数据增强TTA**（**Test Time Augmentation**）**

这个函数只在 val、detect 主函数中使用，用于提高推导的精度。

> **设分类数为 80 、检测框属性数为 5，则基本步骤是**：
> 
> 1.  对图像进行变换：总共 3 次，分别是 [ 原图 ]，[ 尺寸缩小到原来的 0.83，同时水平翻转 ]，[ 尺寸缩小到原来的 0.67 ]
> 2.  对图像使用 _forward_once 函数，得到在 eval 模式下网络模型的推导结果。对原图是 shape 为 [1, 22743, 85] 的图像检测框信息 (见 Detect 对象的 forward 函数)
> 3.  根据 尺寸缩小倍数、翻转维度 对检测框信息进行逆变换，添加进列表 y
> 4.  截取 y[0] 对大物体的检测结果，保留 y[1] 所有的检测结果，截取 y[2] 对小物体的检测结果，拼接得到新的检测框信息

* * *

#### 4.2.3 不采用数据增强的forward

```py
def _forward_once(self, x, profile=False, visualize=False):
    """
            :params x: 输入图像
            :params profile: True 可以做一些性能评估
            :params feature_vis: True 可以做一些特征可视化
            :return train: 一个tensor list 存放三个元素   [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                           分别是 [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
                    inference: 0 [1, 19200+4800+1200, 25] = [bs, anchor_num*grid_w*grid_h, xywh+c+20classes]
                               1 一个tensor list 存放三个元素 [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                                 [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
    """
    # 各网络层输出, 各网络层推导耗时
    # y: 存放着self.save=True的每一层的输出，因为后面的层结构concat等操作要用到
    # dt: 在profile中做性能评估时使用
    y, dt = [], []  # outputs
    # 前向推理每一层结构   m.i=index   m.f=from   m.type=类名   m.np=number of params
    # if not from previous layer   m.f=当前层的输入来自哪一层的输出  s的m.f都是-1
    # 遍历model每个模块
    for m in self.model:
        # 输入来源
        if m.f != -1:  # if not from previous layer
            # 这里需要做4个concat操作和1个Detect操作
            # concat操作如m.f=[-1, 6] x就有两个元素,一个是上一层的输出,另一个是index=6的层的输出 再送到x=m(x)做concat操作
            # Detect操作m.f=[17, 20, 23] x有三个元素,分别存放第17层第20层第23层的输出 再送到x=m(x)做Detect的forward
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

        if profile: # 性能评估
            self._profile_one_layer(m, x, dt)
            # 推理，得到输出
        x = m(x)  # run
        # 存放到y里面，concat这些层会用到的
        y.append(x if m.i in self.save else None)  # save output
        if visualize:
            feature_visualization(x, m.type, m.i, save_dir=visualize)
    return x 
```

这个函数是训练的forward，对模型的每一层进行推理迭代。

* * *

#### 4.2.4 _descale_pred():将推理结果恢复到原图尺寸

```py
def _descale_pred(self, p, flips, scale, img_size):
    """用在上面的__init__函数上
            将推理结果恢复到原图图片尺寸  Test Time Augmentation(TTA)中用到
            de-scale predictions following augmented inference (inverse operation)
            :params p: 推理结果
            :params flips:
            :params scale:
            :params img_size:
    """
    # de-scale predictions following augmented inference (inverse operation)
    if self.inplace:
        # 将xywh除以scale，恢复原来的大小
        p[..., :4] /= scale  # de-scale
        # bs c h w  当flips=2是对h进行变换，那就是上下进行翻转
        if flips == 2:
            p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
        # 同理flips=3是对水平进行翻转
        elif flips == 3:
            p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
    else:
        x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        p = torch.cat((x, y, wh, p[..., 4:]), -1)
    return p 
```

这个函数用在上面的 `__init__` 上面，将推理结果恢复到原图尺寸TTA中用到。

* * *

#### 4.2.5 _clip_augmented()：TTA的时候对图片进行裁剪

```py
def _clip_augmented(self, y):
    # Clip YOLOv5 augmented inference tails
    nl = self.model[-1].nl  # number of detection layers (P3-P5)
    g = sum(4 ** x for x in range(nl))  # grid points
    e = 1  # exclude layer count
    i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
    y[0] = y[0][:, :-i]  # large
    i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
    y[-1] = y[-1][:, i:]  # small
    return y 
```

**裁剪的数据增强方式。**

* * *

#### 4.2.6 打印日志信息

```py
def _profile_one_layer(self, m, x, dt):
    c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
    o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
    t = time_sync()
    for _ in range(10):
        m(x.copy() if c else x)
    dt.append((time_sync() - t) * 100)
    if m == self.model[0]:
        LOGGER.info(f"{'time (ms)':>10s}  {'GFLOPs':>10s}  {'params':>10s} module")
    LOGGER.info(f'{dt[-1]:10.2f}  {o:10.2f}  {m.np:10.0f}  {m.type}')
    if c:
        LOGGER.info(f"{sum(dt):10.2f}  {'-':>10s}  {'-':>10s} Total") 
```

**用于测试网络层的性能**

**使用 logging 模块输出有：**

*   time (ms)： 前向推导时间
*   GFLOPs：浮点运算量，需要安装 thop 模块
*   params： 网络层参数量
*   module： 网络层名称

* * *

#### 4.2.7 bias信息

```py
 def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency 初始化detect偏置
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1).detach()  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    # 打印模型中最后Detect层的偏置bias信息(也可以任选哪些层bias信息)
    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # 打印模型中Bottleneck层的权重参数weights信息(也可以任选哪些层weights信息)
    def _print_weights(self):
        for m in self.model.modules():
            if type(m) is Bottleneck:
                LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights 
```

初始化detect偏置 + 打印偏置pias信息

* * *

#### 4.2.8 fuse()：融合Conv2d+BN

```py
"""用在detect.py、val.py
    fuse model Conv2d() + BatchNorm2d() layers
    调用torch_utils.py中的fuse_conv_and_bn函数和common.py中Conv模块的fuseforward函数"""
    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers   Conv2d+BN 融合
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):  # hasattr判断对象是否包含对应的属性
                # 更新卷积层
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                # 移除BN
                delattr(m, 'bn')  # remove batchnorm    删除对象的属性
                # 更新前向传播
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self 
```

**主要是用在推理额和验证上面，起一个加速推理的作用**

* * *

## 5\. 主函数

主函数是可以执行的，这一点很多人都不知道。

在设置参数的时候，将`profile`和`line-profile`设置为`true`，就可以输出网络层结构、参数量、`GFLOPs`等参数量了，还可与输出`yaml`网络结构每一层的耗时和`GFLOPs`。

在进行网络改进的时候，就有这样一个指标可以看，可以增加实验的对比性。
![在这里插入图片描述](img/412fb175e9f09e103d7eac1714b90424.png)

# 结束语

**到这里整个yolo脚本文件就结束了，大部分难度还是不大，主要是用得比较多，所以这部分掌握的比较好。**

**今天带电脑去敲了2节课的博客，感觉从大三开始就一直这样了哈哈哈，基本没怎么听课。跟室友同学吃午饭的时候，看见了一个巴基斯坦的小姐姐，我和室友就谈到现在出国的事情。正好昨天才看到一个新闻说Meta裁员，已经裁了2.1万人了，哎，现在大环境真的很糟糕，全球经济大衰退。但我身边的同学却说了一句：但关我什么事呢？我感觉我有点鸡同鸭讲，怪无语的，我觉得但凡稍微懂一点常识的人都不会说出如此可笑的话。因为国外的大部分裁员势必也说明了国内的经济环境也不好，而大规模的海外裁员又会让海归们跟我们国内竞争同一个岗位，在国内hc本身就在极度缩减的情况下，就业形势难度呈指数上升。这方面我虽然没有专门去了解，但大致情况还是知道的。所有我当时意味深长地说了一句：你过一两年就知道了。**

**但这个时候我才意识到，原来身边的人包括我自己都还处于所谓的“校园生活“环境中。我们在这个温室里面待了太久了，以至于近乎快和社会脱节了，都说大学是离社会最近的地方，尤其是我这学期又搬到了仙桃这边，和社会打交道，但我仍感觉身边人的学生气息浓厚。或许在他们大部分眼中，现在只需要跟风准备考研，平时有事没事刷着重复着网上一些毫无意义的梗，对自己的未来想要什么完全不知。前几天一个写简历的职规课作业，说来可笑，卓越班里一大半的技术岗位简历加起来都凑不了一个拿得出手像样的项目，还真的就蛮怪诞的。**

**我是一个思想很容易受他人影响的人，好的坏的都是如此。前几天认识的一位friend，通过对他的了解和他和我的一些real talk，让我意识到一方面时代变了，环境真的很糟糕；另一方面让我意识到我前两年虽有在一直学习，但我的持续不断的内耗使我兜了太多的圈子，当我抬头的时候才发现我和他人的距离已经相差甚远。今天中午就想着这些bullshit，午觉也没睡着，就赶紧起床来实验室继续学习了。我以前对某种意义上的”卷王“有一种看不起的轻蔑，但我现在才明白，我并不是憎恨谁，我只是想成为最强的，我觉得这样的人才是一直清楚明白他们想要的是什么，respect。刘聪告诉我要保持愤怒，才能得到你想要的，只有自己清楚自己想要什么，一直向前走吧。各位下篇见~**