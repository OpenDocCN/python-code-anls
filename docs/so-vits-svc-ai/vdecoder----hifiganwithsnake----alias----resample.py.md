# `so-vits-svc\vdecoder\hifiganwithsnake\alias\resample.py`

```
# 从 https://github.com/junjun3518/alias-free-torch 中的 Apache License 2.0 下进行了修改
#   LICENSE 在 incl_licenses 目录中。

# 导入 torch.nn 和 functional 模块
import torch.nn as nn
from torch.nn import functional as F

# 从 filter 模块中导入 LowPassFilter1d 和 kaiser_sinc_filter1d
from .filter import LowPassFilter1d, kaiser_sinc_filter1d

# 定义 UpSample1d 类，继承自 nn.Module
class UpSample1d(nn.Module):
    # 初始化方法
    def __init__(self, ratio=2, kernel_size=None, C=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        # 使用 kaiser_sinc_filter1d 函数创建滤波器
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio,
                                      half_width=0.6 / ratio,
                                      kernel_size=self.kernel_size)
        # 将滤波器注册为模型的缓冲区
        self.register_buffer("filter", filter)
        self.conv_transpose1d_block = None
        # 如果 C 不为 None
        if C is not None:
            # 创建 nn.ConvTranspose1d 模块列表
            self.conv_transpose1d_block = [nn.ConvTranspose1d(C,
                                                            C,
                                                            kernel_size=self.kernel_size,
                                                            stride=self.stride, 
                                                            groups=C, 
                                                            bias=False
                                                            ),]
            # 将滤波器作为权重赋值给 nn.ConvTranspose1d 模块
            self.conv_transpose1d_block[0].weight = nn.Parameter(self.filter.expand(C, -1, -1).clone())
            # 设置权重不需要梯度
            self.conv_transpose1d_block[0].requires_grad_(False)
            
    # x: [B, C, T]
    # 定义一个前向传播函数，接受输入 x 和条件 C（可选）
    def forward(self, x, C=None):
        # 如果第一个转置卷积块的权重所在设备与输入 x 的设备不同，则将其移动到输入 x 所在的设备
        if self.conv_transpose1d_block[0].weight.device != x.device:
            self.conv_transpose1d_block[0] = self.conv_transpose1d_block[0].to(x.device)
        # 如果转置卷积块为空
        if self.conv_transpose1d_block is None:
            # 如果条件 C 为空，则获取输入 x 的通道数
            if C is None:
                _, C, _ = x.shape
            # 对输入 x 进行填充，使用“replicate”模式
            x = F.pad(x, (self.pad, self.pad), mode='replicate')
            # 进行转置卷积操作，使用扩展后的滤波器，步长为 self.stride，分组数为 C
            x = self.ratio * F.conv_transpose1d(
                x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
            # 对输出 x 进行裁剪，去除填充部分
            x = x[..., self.pad_left:-self.pad_right]
        else:
            # 对输入 x 进行填充，使用“replicate”模式
            x = F.pad(x, (self.pad, self.pad), mode='replicate')
            # 使用转置卷积块进行转置卷积操作
            x = self.ratio * self.conv_transpose1d_block[0](x)
            # 对输出 x 进行裁剪，去除填充部分
            x = x[..., self.pad_left:-self.pad_right]
        # 返回输出 x
        return x
# 定义一个名为 DownSample1d 的类，继承自 nn.Module
class DownSample1d(nn.Module):
    # 初始化方法，接受 ratio（下采样比例）、kernel_size（卷积核大小）、C（参数）三个参数
    def __init__(self, ratio=2, kernel_size=None, C=None):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的 ratio 参数赋给实例变量 ratio
        self.ratio = ratio
        # 如果没有传入 kernel_size 参数，则根据 ratio 计算出卷积核大小
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        # 创建一个名为 lowpass 的 LowPassFilter1d 实例，传入相应的参数
        self.lowpass = LowPassFilter1d(cutoff=0.5 / ratio,
                                       half_width=0.6 / ratio,
                                       stride=ratio,
                                       kernel_size=self.kernel_size,
                                       C=C)

    # 前向传播方法，接受输入 x
    def forward(self, x):
        # 对输入 x 进行低通滤波处理
        xx = self.lowpass(x)
        # 返回处理后的结果
        return xx
```