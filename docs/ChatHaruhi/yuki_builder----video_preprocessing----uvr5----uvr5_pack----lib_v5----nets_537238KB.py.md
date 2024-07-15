# `.\Chat-Haruhi-Suzumiya\yuki_builder\video_preprocessing\uvr5\uvr5_pack\lib_v5\nets_537238KB.py`

```py
# 导入PyTorch库
import torch
# 导入NumPy库，并用np作为别名
import numpy as np
# 从torch中导入nn模块
from torch import nn
# 从torch.nn.functional中导入F，并用F作为别名
import torch.nn.functional as F

# 导入当前目录下的layers模块，并命名为layers
from . import layers_537238KB as layers

# 定义一个名为BaseASPPNet的类，继承自nn.Module类
class BaseASPPNet(nn.Module):
    # 初始化方法，接收输入通道数nin，初始通道数ch，以及扩张率元组dilations
    def __init__(self, nin, ch, dilations=(4, 8, 16)):
        super(BaseASPPNet, self).__init__()
        # 初始化编码器层enc1，使用layers模块中的Encoder类，参数依次为输入通道数nin、输出通道数ch、卷积核大小3、步幅2、填充1
        self.enc1 = layers.Encoder(nin, ch, 3, 2, 1)
        # 初始化编码器层enc2，使用layers模块中的Encoder类，参数依次为输入通道数ch、输出通道数ch*2、卷积核大小3、步幅2、填充1
        self.enc2 = layers.Encoder(ch, ch * 2, 3, 2, 1)
        # 初始化编码器层enc3，使用layers模块中的Encoder类，参数依次为输入通道数ch*2、输出通道数ch*4、卷积核大小3、步幅2、填充1
        self.enc3 = layers.Encoder(ch * 2, ch * 4, 3, 2, 1)
        # 初始化编码器层enc4，使用layers模块中的Encoder类，参数依次为输入通道数ch*4、输出通道数ch*8、卷积核大小3、步幅2、填充1
        self.enc4 = layers.Encoder(ch * 4, ch * 8, 3, 2, 1)

        # 初始化ASPP模块，使用layers模块中的ASPPModule类，参数依次为输入通道数ch*8、输出通道数ch*16、扩张率元组dilations
        self.aspp = layers.ASPPModule(ch * 8, ch * 16, dilations)

        # 初始化解码器层dec4，使用layers模块中的Decoder类，参数依次为输入通道数ch*(8+16)、输出通道数ch*8、卷积核大小3、步幅1、填充1
        self.dec4 = layers.Decoder(ch * (8 + 16), ch * 8, 3, 1, 1)
        # 初始化解码器层dec3，使用layers模块中的Decoder类，参数依次为输入通道数ch*(4+8)、输出通道数ch*4、卷积核大小3、步幅1、填充1
        self.dec3 = layers.Decoder(ch * (4 + 8), ch * 4, 3, 1, 1)
        # 初始化解码器层dec2，使用layers模块中的Decoder类，参数依次为输入通道数ch*(2+4)、输出通道数ch*2、卷积核大小3、步幅1、填充1
        self.dec2 = layers.Decoder(ch * (2 + 4), ch * 2, 3, 1, 1)
        # 初始化解码器层dec1，使用layers模块中的Decoder类，参数依次为输入通道数ch*(1+2)、输出通道数ch、卷积核大小3、步幅1、填充1
        self.dec1 = layers.Decoder(ch * (1 + 2), ch, 3, 1, 1)

    # 定义__call__方法，用于模型调用时的前向传播过程，接收输入张量x
    def __call__(self, x):
        # 对输入张量x进行编码过程，获取编码结果h和各阶段的编码特征e1, e2, e3, e4
        h, e1 = self.enc1(x)
        h, e2 = self.enc2(h)
        h, e3 = self.enc3(h)
        h, e4 = self.enc4(h)

        # 将编码后的特征张量h输入到ASPP模块中进行多尺度空洞卷积操作
        h = self.aspp(h)

        # 将ASPP模块输出的特征张量h输入到解码器层dec4中进行解码，得到解码结果
        h = self.dec4(h, e4)
        # 将解码结果h输入到解码器层dec3中进行解码，得到解码结果
        h = self.dec3(h, e3)
        # 将解码结果h输入到解码器层dec2中进行解码，得到解码结果
        h = self.dec2(h, e2)
        # 将解码结果h输入到解码器层dec1中进行解码，得到最终的输出结果
        h = self.dec1(h, e1)

        # 返回最终的输出结果h
        return h


# 定义一个名为CascadedASPPNet的类，继承自nn.Module类
class CascadedASPPNet(nn.Module):
    # 初始化方法，接收输入的FFT大小n_fft
    def __init__(self, n_fft):
        super(CascadedASPPNet, self).__init__()
        # 初始化第一阶段低频段网络，使用BaseASPPNet类，参数依次为输入通道数2，初始通道数64
        self.stg1_low_band_net = BaseASPPNet(2, 64)
        # 初始化第一阶段高频段网络，使用BaseASPPNet类，参数依次为输入通道数2，初始通道数64
        self.stg1_high_band_net = BaseASPPNet(2, 64)

        # 初始化第二阶段的桥接层，使用layers模块中的Conv2DBNActiv类，参数依次为输入通道数66，输出通道数32，卷积核大小1，步幅1，填充0
        self.stg2_bridge = layers.Conv2DBNActiv(66, 32, 1, 1, 0)
        # 初始化第二阶段全频段网络，使用BaseASPPNet类，参数依次为输入通道数32，初始通道数64
        self.stg2_full_band_net = BaseASPPNet(32, 64)

        # 初始化第三阶段的桥接层，使用layers模块中的Conv2DBNActiv类，参数依次为输入通道数130，输出通道数64，卷积核大小1，步幅1，填充0
        self.stg3_bridge = layers.Conv2DBNActiv(130, 64, 1, 1, 0)
        # 初始化第三阶段全频段网络，使用BaseASPPNet类，参数依次为输入通道数64，初始通道数128
        self.stg3_full_band_net = BaseASPPNet(64, 128)

        # 初始化输出层，使用nn.Conv2d类，参数依次为输入通道数128，输出通道数2，卷积核大小1，无偏置
        self.out = nn.Conv2d(128, 2, 1, bias=False)
        # 初始化辅助输出层1，使用nn.Conv2d类，参数依次为输入通道数64，输出通道数2，卷积核大小1，无偏置
        self.aux1_out = nn.Conv2d(64, 2, 1, bias=False)
        # 初始化辅助输出层2，使用nn.Conv2d类，参数依次为输入通道数64，输出通道数2，卷积核大小1，无偏置
        self.aux2_out = nn.Conv2d(64, 2, 1, bias=False)

        # 设置最大频率分量数，即FFT大小的一半
        self.max
    # 定义神经网络的前向传播函数，输入参数包括输入张量 x 和可选的攻击性参数 aggressiveness
    def forward(self, x, aggressiveness=None):
        # 将输入张量 x 的梯度分离出来，用于后续计算梯度的截断
        mix = x.detach()
        # 克隆输入张量 x，用于保留原始数据的副本
        x = x.clone()

        # 截取输入张量 x 的前 self.max_bin 个通道（或列）
        x = x[:, :, : self.max_bin]

        # 计算 bandw 作为 x 的通道数除以2的结果
        bandw = x.size()[2] // 2
        # 使用 stg1_low_band_net 和 stg1_high_band_net 对 x 的前半部分和后半部分进行处理，然后连接起来
        aux1 = torch.cat(
            [
                self.stg1_low_band_net(x[:, :, :bandw]),
                self.stg1_high_band_net(x[:, :, bandw:]),
            ],
            dim=2,
        )

        # 将 x 和 aux1 连接起来，形成新的输入 h
        h = torch.cat([x, aux1], dim=1)
        # 使用 stg2_bridge 将 h 进行转换后，再经过 stg2_full_band_net 处理得到 aux2
        aux2 = self.stg2_full_band_net(self.stg2_bridge(h))

        # 将 h、aux1 和 aux2 连接起来形成新的输入 h
        h = torch.cat([x, aux1, aux2], dim=1)
        # 使用 stg3_bridge 将 h 进行转换后，再经过 stg3_full_band_net 处理得到输出 h
        h = self.stg3_full_band_net(self.stg3_bridge(h))

        # 对 h 进行 sigmoid 操作，得到 mask，用于输出
        mask = torch.sigmoid(self.out(h))
        # 使用 F.pad 对 mask 进行填充，使其通道数变为 self.output_bin
        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode="replicate",
        )

        # 如果处于训练状态
        if self.training:
            # 对 aux1 进行 sigmoid 操作，得到 aux1_out，再对其进行填充
            aux1 = torch.sigmoid(self.aux1_out(aux1))
            aux1 = F.pad(
                input=aux1,
                pad=(0, 0, 0, self.output_bin - aux1.size()[2]),
                mode="replicate",
            )
            # 对 aux2 进行 sigmoid 操作，得到 aux2_out，再对其进行填充
            aux2 = torch.sigmoid(self.aux2_out(aux2))
            aux2 = F.pad(
                input=aux2,
                pad=(0, 0, 0, self.output_bin - aux2.size()[2]),
                mode="replicate",
            )
            # 返回 mask、aux1 和 aux2 与 mix 的乘积
            return mask * mix, aux1 * mix, aux2 * mix
        else:
            # 如果存在 aggressiveness 参数，则对 mask 进行分段幂操作
            if aggressiveness:
                mask[:, :, : aggressiveness["split_bin"]] = torch.pow(
                    mask[:, :, : aggressiveness["split_bin"]],
                    1 + aggressiveness["value"] / 3,
                )
                mask[:, :, aggressiveness["split_bin"] :] = torch.pow(
                    mask[:, :, aggressiveness["split_bin"] :],
                    1 + aggressiveness["value"],
                )

            # 返回 mask 与 mix 的乘积作为预测结果
            return mask * mix

    # 定义预测函数，输入参数为 x_mag 和可选的攻击性参数 aggressiveness
    def predict(self, x_mag, aggressiveness=None):
        # 调用前向传播函数计算 h
        h = self.forward(x_mag, aggressiveness)

        # 如果存在偏移量 offset 大于0，则对 h 进行截断操作
        if self.offset > 0:
            h = h[:, :, :, self.offset : -self.offset]
            # 断言截断后的 h 的最后一个维度大于0，确保有效性
            assert h.size()[3] > 0

        # 返回处理后的 h 作为预测结果
        return h
```