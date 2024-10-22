# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\lpips\model\model.py`

```py
# 导入 functools 模块以使用函数工具
import functools

# 导入 PyTorch 的神经网络模块
import torch.nn as nn

# 从上级目录导入 ActNorm 实用程序
from ..util import ActNorm


# 权重初始化函数
def weights_init(m):
    # 获取模块的类名
    classname = m.__class__.__name__
    # 如果类名中包含 "Conv"，进行卷积层的权重初始化
    if classname.find("Conv") != -1:
        try:
            # 初始化权重为均值 0，标准差 0.02 的正态分布
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        except:
            # 如果出错，尝试初始化卷积层的权重
            nn.init.normal_(m.conv.weight.data, 0.0, 0.02)
    # 如果类名中包含 "BatchNorm"，进行批量归一化层的初始化
    elif classname.find("BatchNorm") != -1:
        # 初始化权重为均值 1，标准差 0.02 的正态分布
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        # 将偏置初始化为 0
        nn.init.constant_(m.bias.data, 0)


# 定义一个 NLayerDiscriminator 类，继承自 nn.Module
class NLayerDiscriminator(nn.Module):
    """定义一个 PatchGAN 判别器，参照 Pix2Pix
    --> 参考链接：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    # 初始化函数
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """构造一个 PatchGAN 判别器
        参数：
            input_nc (int)  -- 输入图像的通道数
            ndf (int)       -- 最后一层卷积的过滤器数量
            n_layers (int)  -- 判别器中的卷积层数量
            norm_layer      -- 归一化层
        """
        # 调用父类初始化方法
        super(NLayerDiscriminator, self).__init__()
        # 根据是否使用 ActNorm 选择归一化层
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        # 检查归一化层是否需要偏置
        if type(norm_layer) == functools.partial:  # 如果使用偏函数，BatchNorm2d 具有仿射参数
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        # 定义卷积核大小和填充
        kw = 4
        padw = 1
        # 初始化网络序列
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),  # 输入层卷积
            nn.LeakyReLU(0.2, True),  # 激活函数
        ]
        # 初始化滤波器数量的倍数
        nf_mult = 1
        nf_mult_prev = 1
        # 循环添加卷积层和归一化层
        for n in range(1, n_layers):  # 逐渐增加过滤器的数量
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)  # 最大值限制为 8
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
                nn.LeakyReLU(0.2, True),  # 激活函数
            ]

        # 最后一层卷积的参数设置
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)  # 最大值限制为 8
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
            nn.LeakyReLU(0.2, True),  # 激活函数
        ]

        # 添加输出层
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)  # 输出 1 通道预测图
        ]  
        # 将序列转换为顺序容器
        self.main = nn.Sequential(*sequence)

    # 定义前向传播方法
    def forward(self, input):
        """标准前向传播。"""
        return self.main(input)  # 返回主网络的输出
```