# `yolov5-DNF\utils\activations.py`

```py
# 导入 PyTorch 库
import torch
import torch.nn as nn
import torch.nn.functional as F

# Swish https://arxiv.org/pdf/1905.02244.pdf ---------------------------------------------------------------------------
# 定义 Swish 激活函数
class Swish(nn.Module):  #
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

# 定义 export-friendly 版本的 nn.Hardswish()，用于导出模型
class Hardswish(nn.Module):  
    @staticmethod
    def forward(x):
        # 使用 F.hardtanh() 实现 Hardswish 激活函数
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX

# 定义 MemoryEfficientSwish 类，实现内存高效的 Swish 激活函数
class MemoryEfficientSwish(nn.Module):
    # 定义 F 类，继承自 torch.autograd.Function
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * torch.sigmoid(x)

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            return grad_output * (sx * (1 + x * (1 - sx)))

    # 实现前向传播
    def forward(self, x):
        return self.F.apply(x)

# Mish https://github.com/digantamisra98/Mish --------------------------------------------------------------------------
# 定义 Mish 激活函数
class Mish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()

# 定义内存高效的 Mish 激活函数
class MemoryEfficientMish(nn.Module):
    # 定义 F 类，继承自 torch.autograd.Function
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    # 实现前向传播
    def forward(self, x):
        return self.F.apply(x)

# FReLU https://arxiv.org/abs/2007.11824 -------------------------------------------------------------------------------
# 定义 FReLU 激活函数
class FReLU(nn.Module):
    # 初始化函数，接受输入参数 c1 和 k，默认值为 3
    def __init__(self, c1, k=3):  # ch_in, kernel
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个二维卷积层，输入通道数为 c1，输出通道数为 c1，卷积核大小为 k
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1)
        # 创建一个二维批归一化层，输入通道数为 c1
        self.bn = nn.BatchNorm2d(c1)

    # 前向传播函数，接受输入参数 x
    def forward(self, x):
        # 对输入 x 进行卷积和批归一化操作，然后取最大值
        return torch.max(x, self.bn(self.conv(x)))
```