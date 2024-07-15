# `.\Chat-Haruhi-Suzumiya\yuki_builder\video_preprocessing\uvr5\uvr5_pack\lib_v5\nets_537227KB.py`

```py
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

# 导入自定义模块 layers_537238KB，用作网络中的各层定义
from . import layers as layers_537238KB

class BaseASPPNet(nn.Module):
    def __init__(self, nin, ch, dilations=(4, 8, 16)):
        super(BaseASPPNet, self).__init__()
        # 定义四个编码器层，每层都是 layers.Encoder 类的实例
        self.enc1 = layers_537238KB.Encoder(nin, ch, 3, 2, 1)
        self.enc2 = layers_537238KB.Encoder(ch, ch * 2, 3, 2, 1)
        self.enc3 = layers_537238KB.Encoder(ch * 2, ch * 4, 3, 2, 1)
        self.enc4 = layers_537238KB.Encoder(ch * 4, ch * 8, 3, 2, 1)

        # ASPP 模块的定义，使用 ch * 8 输入，输出 ch * 16，使用给定的扩张率 dilations
        self.aspp = layers_537238KB.ASPPModule(ch * 8, ch * 16, dilations)

        # 定义四个解码器层，每层都是 layers.Decoder 类的实例
        self.dec4 = layers_537238KB.Decoder(ch * (8 + 16), ch * 8, 3, 1, 1)
        self.dec3 = layers_537238KB.Decoder(ch * (4 + 8), ch * 4, 3, 1, 1)
        self.dec2 = layers_537238KB.Decoder(ch * (2 + 4), ch * 2, 3, 1, 1)
        self.dec1 = layers_537238KB.Decoder(ch * (1 + 2), ch, 3, 1, 1)

    def __call__(self, x):
        # 编码过程，包括四个编码器和 ASPP 模块的应用
        h, e1 = self.enc1(x)
        h, e2 = self.enc2(h)
        h, e3 = self.enc3(h)
        h, e4 = self.enc4(h)

        # ASPP 模块的应用，处理编码后的特征 h
        h = self.aspp(h)

        # 解码过程，包括四个解码器的应用
        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = self.dec1(h)

        return h


class CascadedASPPNet(nn.Module):
    def __init__(self, n_fft):
        super(CascadedASPPNet, self).__init__()
        # 定义阶段1低频带和高频带的 BaseASPPNet 网络
        self.stg1_low_band_net = BaseASPPNet(2, 64)
        self.stg1_high_band_net = BaseASPPNet(2, 64)

        # 阶段2的桥接层和全频带的 BaseASPPNet 网络
        self.stg2_bridge = layers_537238KB.Conv2DBNActiv(66, 32, 1, 1, 0)
        self.stg2_full_band_net = BaseASPPNet(32, 64)

        # 阶段3的桥接层和全频带的 BaseASPPNet 网络
        self.stg3_bridge = layers_537238KB.Conv2DBNActiv(130, 64, 1, 1, 0)
        self.stg3_full_band_net = BaseASPPNet(64, 128)

        # 最终输出层和两个辅助输出层的定义
        self.out = nn.Conv2d(128, 2, 1, bias=False)
        self.aux1_out = nn.Conv2d(64, 2, 1, bias=False)
        self.aux2_out = nn.Conv2d(64, 2, 1, bias=False)

        # 计算最大频率 bin 和输出频率 bin
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        # 偏移量设置为 128
        self.offset = 128
    # 前向传播函数，接受输入 x 和可选的攻击性参数 aggressiveness
    def forward(self, x, aggressiveness=None):
        # 创建 x 的一个分离副本，用于计算梯度
        mix = x.detach()
        # 克隆输入 x，保留原始数据
        x = x.clone()

        # 限制 x 的通道数到 self.max_bin
        x = x[:, :, : self.max_bin]

        # 计算 x 的通道数的一半
        bandw = x.size()[2] // 2
        # 将 x 分成两部分，分别通过两个不同的神经网络处理
        aux1 = torch.cat(
            [
                self.stg1_low_band_net(x[:, :, :bandw]),
                self.stg1_high_band_net(x[:, :, bandw:]),
            ],
            dim=2,
        )

        # 将 x 和第一阶段处理结果 aux1 连接起来
        h = torch.cat([x, aux1], dim=1)
        # 将连接后的数据通过第二阶段的神经网络处理
        aux2 = self.stg2_full_band_net(self.stg2_bridge(h))

        # 将 x、aux1 和 aux2 连接起来
        h = torch.cat([x, aux1, aux2], dim=1)
        # 将连接后的数据通过第三阶段的神经网络处理
        h = self.stg3_full_band_net(self.stg3_bridge(h))

        # 对输出 h 应用 sigmoid 函数得到掩码
        mask = torch.sigmoid(self.out(h))
        # 使用 F.pad 对掩码进行填充，以匹配预期的输出维度
        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode="replicate",
        )

        # 如果处于训练模式
        if self.training:
            # 对 aux1 应用 sigmoid 函数得到辅助输出
            aux1 = torch.sigmoid(self.aux1_out(aux1))
            # 使用 F.pad 对 aux1 进行填充，以匹配预期的输出维度
            aux1 = F.pad(
                input=aux1,
                pad=(0, 0, 0, self.output_bin - aux1.size()[2]),
                mode="replicate",
            )
            # 对 aux2 应用 sigmoid 函数得到辅助输出
            aux2 = torch.sigmoid(self.aux2_out(aux2))
            # 使用 F.pad 对 aux2 进行填充，以匹配预期的输出维度
            aux2 = F.pad(
                input=aux2,
                pad=(0, 0, 0, self.output_bin - aux2.size()[2]),
                mode="replicate",
            )
            # 返回 mask、aux1 和 aux2 与 mix 的乘积作为训练模式下的输出
            return mask * mix, aux1 * mix, aux2 * mix
        else:
            # 如果未处于训练模式但设置了 aggressiveness 参数
            if aggressiveness:
                # 根据 aggressiveness 参数调整 mask 的部分值
                mask[:, :, : aggressiveness["split_bin"]] = torch.pow(
                    mask[:, :, : aggressiveness["split_bin"]],
                    1 + aggressiveness["value"] / 3,
                )
                mask[:, :, aggressiveness["split_bin"] :] = torch.pow(
                    mask[:, :, aggressiveness["split_bin"] :],
                    1 + aggressiveness["value"],
                )

            # 返回 mask 与 mix 的乘积作为非训练模式下的输出
            return mask * mix

    # 预测函数，接受输入 x_mag 和可选的攻击性参数 aggressiveness
    def predict(self, x_mag, aggressiveness=None):
        # 调用 forward 方法得到预测结果 h
        h = self.forward(x_mag, aggressiveness)

        # 如果设置了偏移量 offset，则在第三维上进行裁剪以移除边缘数据
        if self.offset > 0:
            h = h[:, :, :, self.offset : -self.offset]
            # 断言裁剪后的数据维度仍大于 0
            assert h.size()[3] > 0

        # 返回处理后的预测结果 h
        return h
```