# `yolov5-DNF\models\experimental.py`

```
# 导入 numpy 库，并使用别名 np
import numpy as np
# 导入 torch 库
import torch
# 导入 torch.nn 模块
import torch.nn as nn
# 从 utils.google_utils 模块中导入 attempt_download 函数
from utils.google_utils import attempt_download
# 从 models.common 模块中导入 Conv, DWConv 类
from models.common import Conv, DWConv

# 定义一个名为 CrossConv 的类，继承自 nn.Module
class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super(CrossConv, self).__init__()
        # 计算隐藏通道数
        c_ = int(c2 * e)  # hidden channels
        # 创建一个 1xk 的卷积层
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        # 创建一个 kx1 的卷积层
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        # 判断是否添加 shortcut
        self.add = shortcut and c1 == c2

    # 前向传播函数
    def forward(self, x):
        # 如果添加了 shortcut，则返回 x 加上两个卷积层的输出
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

# 定义一个名为 C3 的类，继承自 nn.Module
class C3(nn.Module):
    # Cross Convolution CSP
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

# 定义一个名为 Sum 的类，继承自 nn.Module
class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super(Sum, self).__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1., n) / 2, requires_grad=True)  # layer weights
    # 定义一个前向传播函数，接受输入 x
    def forward(self, x):
        # 取输入 x 的第一个元素作为初始值 y，没有权重
        y = x[0]  # no weight
        # 如果有权重，则对权重进行处理
        if self.weight:
            # 对权重进行 sigmoid 函数处理，并乘以 2
            w = torch.sigmoid(self.w) * 2
            # 遍历迭代器中的元素
            for i in self.iter:
                # 更新 y 的值，加上输入 x 中对应位置的值乘以权重
                y = y + x[i + 1] * w[i]
        # 如果没有权重
        else:
            # 遍历迭代器中的元素
            for i in self.iter:
                # 更新 y 的值，加上输入 x 中对应位置的值
                y = y + x[i + 1]
        # 返回最终的 y 值
        return y
class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super(GhostConv, self).__init__()
        c_ = c2 // 2  # 计算隐藏层的通道数
        self.cv1 = Conv(c1, c_, k, s, g, act)  # 创建第一个卷积层
        self.cv2 = Conv(c_, c_, 5, 1, c_, act)  # 创建第二个卷积层

    def forward(self, x):
        y = self.cv1(x)  # 使用第一个卷积层处理输入
        return torch.cat([y, self.cv2(y)], 1)  # 将第一个卷积层的输出和第二个卷积层的输出拼接在一起


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k, s):
        super(GhostBottleneck, self).__init__()
        c_ = c2 // 2  # 计算隐藏层的通道数
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)  # 返回卷积层处理后的结果加上快捷连接的结果


class MixConv2d(nn.Module):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    # 初始化函数，接受输入通道数 c1，输出通道数 c2，卷积核大小 k，默认步长 s 为 1，是否等通道数 equal_ch 为 True
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        # 调用父类的初始化函数
        super(MixConv2d, self).__init__()
        # 计算卷积组数
        groups = len(k)
        # 如果 equal_ch 为 True，表示每个组的通道数相等
        if equal_ch:  # equal c_ per group
            # 生成 c2 个均匀分布的索引
            i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2 indices
            # 计算每个组的中间通道数
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # 如果 equal_ch 为 False，表示每个组的权重数量相等
            # 初始化 b 数组
            b = [c2] + [0] * groups
            # 生成单位矩阵
            a = np.eye(groups + 1, groups, k=-1)
            # 对单位矩阵进行滚动操作
            a -= np.roll(a, 1, axis=1)
            # 对滚动后的矩阵进行乘法操作
            a *= np.array(k) ** 2
            # 设置第一行为 1
            a[0] = 1
            # 使用最小二乘法求解等权重索引
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        # 创建 nn.ModuleList，包含每个组的卷积层
        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        # 创建批归一化层
        self.bn = nn.BatchNorm2d(c2)
        # 创建激活函数层
        self.act = nn.LeakyReLU(0.1, inplace=True)

    # 前向传播函数
    def forward(self, x):
        # 返回输入 x 与卷积、批归一化、激活函数的组合结果
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))
class Ensemble(nn.ModuleList):
    # 模型集合
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # 最大值集合
        y = torch.cat(y, 1)  # 非极大值抑制集合
        # y = torch.stack(y).mean(0)  # 平均值集合
        return y, None  # 推断，训练输出


def attempt_load(weights, map_location=None):
    # 加载模型集合的权重 weights=[a,b,c] 或单个模型的权重 weights=[a] 或权重为单个模型 weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        model.append(torch.load(w, map_location=map_location)['model'].float().fuse().eval())  # 加载 FP32 模型

    if len(model) == 1:
        return model[-1]  # 返回单个模型
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # 返回模型集合
```