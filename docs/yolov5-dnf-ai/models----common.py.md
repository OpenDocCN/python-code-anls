# `yolov5-DNF\models\common.py`

```
# 导入数学模块
import math

# 导入 PyTorch 深度学习框架相关模块
import torch
import torch.nn as nn
from torch.nn import functional as F

# 导入自定义的非最大抑制函数
from utils.general import non_max_suppression

# 定义交叉卷积类
class CrossConv(nn.Module):
    # 交叉卷积下采样
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super(CrossConv, self).__init__()
        c_ = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, c_, (1, k), (1, s))  # 第一个卷积层
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)  # 第二个卷积层
        self.add = shortcut and c1 == c2  # 是否添加快捷连接

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))  # 前向传播函数

# 定义 C3 类
class C3(nn.Module):
    # 交叉卷积 CSP
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 第一个卷积层
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)  # 第二个卷积层
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)  # 第三个卷积层
        self.cv4 = Conv(2 * c_, c2, 1, 1)  # 第四个卷积层
        self.bn = nn.BatchNorm2d(2 * c_)  # 应用于 cat(cv2, cv3) 的批归一化层
        self.act = nn.LeakyReLU(0.1, inplace=True)  # 激活函数
        self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])  # 交叉卷积模块

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))  # 第一个分支
        y2 = self.cv2(x)  # 第二个分支
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))  # 返回结果

# 定义 SPPF 类
class SPPF(nn.Module):
    # YOLOv5 的空间金字塔池化 - 快速 (SPPF) 层
    def __init__(self, c1, c2, k=5):  # 相当于 SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 第一个卷积层
        self.cv2 = Conv(c_ * 4, c2, 1, 1)  # 第二个卷积层
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)  # 最大池化层
    # 定义一个前向传播函数，接受输入 x
    def forward(self, x):
        # 使用 self.cv1 对输入 x 进行处理
        x = self.cv1(x)
        # 使用 warnings 模块捕获警告
        with warnings.catch_warnings():
            # 忽略 torch 1.9.0 版本的 max_pool2d() 警告
            warnings.simplefilter('ignore')
            # 对处理后的 x 进行模型处理，得到 y1
            y1 = self.m(x)
            # 对 y1 进行模型处理，得到 y2
            y2 = self.m(y1)
            # 将 x、y1、y2、self.m(y2) 进行拼接，并使用 self.cv2 进行处理
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))
class CrossConv(nn.Module):
    # 交叉卷积下采样
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super(CrossConv, self).__init__()
        c_ = int(c2 * e)  # 隐藏通道
        self.cv1 = Conv(c1, c_, (1, k), (1, s))  # 第一个卷积层
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)  # 第二个卷积层
        self.add = shortcut and c1 == c2  # 是否添加快捷连接

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))  # 前向传播计算结果


class C3(nn.Module):
    # 交叉卷积 CSP
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # 隐藏通道
        self.cv1 = Conv(c1, c_, 1, 1)  # 第一个卷积层
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)  # 第二个卷积层
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)  # 第三个卷积层
        self.cv4 = Conv(2 * c_, c2, 1, 1)  # 第四个卷积层
        self.bn = nn.BatchNorm2d(2 * c_)  # 应用于 cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)  # 激活函数
        self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])  # 交叉卷积层序列

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))  # 计算第一个分支结果
        y2 = self.cv2(x)  # 计算第二个分支结果
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))  # 返回最终结果


def autopad(k, p=None):  # kernel, padding
    # 自动填充到 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # 自动填充
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # 深度卷积
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # 标准卷积
    # 初始化卷积层对象
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        # 调用父类的初始化方法
        super(Conv, self).__init__()
        # 创建卷积层对象，设置输入通道数、输出通道数、卷积核大小、步长、填充方式、分组数和是否包含偏置
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        # 创建批归一化层对象，设置通道数为输出通道数
        self.bn = nn.BatchNorm2d(c2)
        # 如果需要激活函数，则创建 Hardswish 激活函数对象，否则创建恒等映射对象
        self.act = nn.Hardswish() if act else nn.Identity()

    # 前向传播方法
    def forward(self, x):
        # 返回经过卷积、批归一化和激活函数处理后的结果
        return self.act(self.bn(self.conv(x)))

    # 融合前向传播方法
    def fuseforward(self, x):
        # 返回经过融合后的卷积和激活函数处理后的结果
        return self.act(self.conv(x))
class Bottleneck(nn.Module):
    # 标准的瓶颈结构
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # 输入通道数，输出通道数，是否使用快捷连接，分组数，扩张因子
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # 隐藏层通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 第一个卷积层
        self.cv2 = Conv(c_, c2, 3, 1, g=g)  # 第二个卷积层
        self.add = shortcut and c1 == c2  # 是否使用快捷连接

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))  # 前向传播


class BottleneckCSP(nn.Module):
    # CSP瓶颈结构 https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # 输入通道数，输出通道数，重复次数，是否使用快捷连接，分组数，扩张因子
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # 隐藏层通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 第一个卷积层
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)  # 第二个卷积层
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)  # 第三个卷积层
        self.cv4 = Conv(2 * c_, c2, 1, 1)  # 第四个卷积层
        self.bn = nn.BatchNorm2d(2 * c_)  # 应用于cat(cv2, cv3)的批归一化层
        self.act = nn.LeakyReLU(0.1, inplace=True)  # 激活函数
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])  # 重复的瓶颈结构

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))  # 第一个分支
        y2 = self.cv2(x)  # 第二个分支
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))  # 前向传播


class SPPF(nn.Module):
    # YOLOv5的空间金字塔池化快速（SPPF）层，由Glenn Jocher提供
    def __init__(self, c1, c2, k=5):  # 等效于SPP(k=(5, 9, 13))的输入通道数，输出通道数，池化核大小
        super().__init__()
        c_ = c1 // 2  # 隐藏层通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 第一个卷积层
        self.cv2 = Conv(c_ * 4, c2, 1, 1)  # 第二个卷积层
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)  # 最大池化层
    # 定义一个前向传播函数，接受输入 x
    def forward(self, x):
        # 使用 self.cv1 对输入 x 进行处理
        x = self.cv1(x)
        # 使用 warnings 模块捕获警告
        with warnings.catch_warnings():
            # 忽略 torch 1.9.0 版本的 max_pool2d() 警告
            warnings.simplefilter('ignore')
            # 对处理后的 x 进行模型处理，得到 y1
            y1 = self.m(x)
            # 对 y1 进行模型处理，得到 y2
            y2 = self.m(y1)
            # 将 x、y1、y2、self.m(y2) 进行拼接，然后使用 self.cv2 进行处理，返回结果
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
class SPP(nn.Module):
    # YOLOv3-SPP 中使用的空间金字塔池化层
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 第一个卷积层
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)  # 第二个卷积层
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])  # 最大池化层列表

    def forward(self, x):
        x = self.cv1(x)  # 使用第一个卷积层
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))  # 使用第二个卷积层和最大池化层拼接


class Focus(nn.Module):
    # 将 wh 信息聚焦到 c 空间
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # 输入通道数，输出通道数，卷积核大小，步长，填充，分组卷积
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)  # 卷积层

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))  # 使用卷积层进行拼接


class Concat(nn.Module):
    # 沿着指定维度连接张量列表
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class NMS(nn.Module):
    # 非极大值抑制（NMS）模块
    conf = 0.3  # 置信度阈值
    iou = 0.6  # IoU 阈值
    classes = None  # （可选列表）按类别过滤

    def __init__(self, dimension=1):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class Flatten(nn.Module):
    # 在 nn.AdaptiveAvgPool2d(1) 之后使用，去除最后两个维度
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class Classify(nn.Module):
    # 分类头，即将 x(b,c1,20,20) 转换为 x(b,c2)
    # 初始化函数，设置分类器的参数和默认值
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        # 调用父类的初始化函数
        super(Classify, self).__init__()
        # 创建自适应平均池化层，将输入数据池化成1x1的大小
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        # 创建卷积层，设置输入通道数、输出通道数、卷积核大小、步长、填充方式、分组数和是否使用偏置
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)  # to x(b,c2,1,1)
        # 创建展平层，用于展平卷积后的数据
        self.flat = Flatten()

    # 前向传播函数
    def forward(self, x):
        # 如果输入是列表，则对列表中的每个元素进行自适应平均池化，并拼接起来
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        # 将拼接后的数据经过卷积层和展平层处理，得到分类结果
        return self.flat(self.conv(z))  # flatten to x(b,c2)
# 定义 RGA 模块类
class RGA_Module(nn.Module):
```