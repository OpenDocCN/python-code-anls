# `.\cogview3-finetune\sat\sgm\modules\autoencoding\lpips\model\model.py`

```
# 导入 functools 模块，用于函数工具
import functools

# 导入 nn 模块，用于构建神经网络
import torch.nn as nn

# 从上级目录的 util 模块导入 ActNorm
from ..util import ActNorm


# 定义权重初始化函数
def weights_init(m):
    # 获取模块的类名
    classname = m.__class__.__name__
    # 如果类名包含 "Conv"，则初始化卷积层的权重
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    # 如果类名包含 "BatchNorm"，则初始化批归一化层的权重和偏置
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# 定义 NLayerDiscriminator 类，继承自 nn.Module
class NLayerDiscriminator(nn.Module):
    """定义一个 PatchGAN 判别器，如 Pix2Pix 所示
    --> 参见 https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    # 构造函数，初始化参数
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """构造一个 PatchGAN 判别器
        参数:
            input_nc (int)  -- 输入图像的通道数
            ndf (int)       -- 最后一个卷积层中的滤波器数量
            n_layers (int)  -- 判别器中的卷积层数量
            norm_layer      -- 归一化层
        """
        # 调用父类的构造函数
        super(NLayerDiscriminator, self).__init__()
        # 根据是否使用 ActNorm 来选择归一化层
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        # 如果归一化层是 functools.partial 类型，判断是否使用偏置
        if (
            type(norm_layer) == functools.partial
        ):  # 不需要使用偏置，因为 BatchNorm2d 有仿射参数
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4  # 卷积核的大小
        padw = 1  # 卷积层的填充
        # 初始化序列，添加第一个卷积层和 LeakyReLU 激活函数
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1  # 当前滤波器倍数
        nf_mult_prev = 1  # 上一层的滤波器倍数
        # 逐渐增加滤波器数量，构建后续的卷积层
        for n in range(1, n_layers):  # 逐渐增加滤波器数量
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)  # 滤波器倍数最大为 8
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,  # 输入通道数
                    ndf * nf_mult,  # 输出通道数
                    kernel_size=kw,  # 卷积核大小
                    stride=2,  # 步幅
                    padding=padw,  # 填充
                    bias=use_bias,  # 是否使用偏置
                ),
                norm_layer(ndf * nf_mult),  # 添加归一化层
                nn.LeakyReLU(0.2, True),  # 添加 LeakyReLU 激活函数
            ]

        nf_mult_prev = nf_mult  # 更新上一个滤波器倍数
        nf_mult = min(2**n_layers, 8)  # 计算最终滤波器倍数
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,  # 输入通道数
                ndf * nf_mult,  # 输出通道数
                kernel_size=kw,  # 卷积核大小
                stride=1,  # 步幅
                padding=padw,  # 填充
                bias=use_bias,  # 是否使用偏置
            ),
            norm_layer(ndf * nf_mult),  # 添加归一化层
            nn.LeakyReLU(0.2, True),  # 添加 LeakyReLU 激活函数
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # 输出 1 通道的预测图
        # 将所有层组合成一个序列
        self.main = nn.Sequential(*sequence)

    # 定义前向传播函数
    def forward(self, input):
        """标准前向传播。"""
        return self.main(input)  # 将输入传入序列并返回输出
```