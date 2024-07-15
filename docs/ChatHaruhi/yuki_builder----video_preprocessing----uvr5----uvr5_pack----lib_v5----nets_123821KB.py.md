# `.\Chat-Haruhi-Suzumiya\yuki_builder\video_preprocessing\uvr5\uvr5_pack\lib_v5\nets_123821KB.py`

```py
import torch
from torch import nn
import torch.nn.functional as F

from . import layers_123821KB as layers


class BaseASPPNet(nn.Module):
    def __init__(self, nin, ch, dilations=(4, 8, 16)):
        super(BaseASPPNet, self).__init__()
        # 初始化四个编码器层，每层使用自定义的Encoder类
        self.enc1 = layers.Encoder(nin, ch, 3, 2, 1)
        self.enc2 = layers.Encoder(ch, ch * 2, 3, 2, 1)
        self.enc3 = layers.Encoder(ch * 2, ch * 4, 3, 2, 1)
        self.enc4 = layers.Encoder(ch * 4, ch * 8, 3, 2, 1)

        # ASPP模块的初始化，输入通道为ch * 8，输出通道为ch * 16，使用给定的dilations参数
        self.aspp = layers.ASPPModule(ch * 8, ch * 16, dilations)

        # 初始化四个解码器层，每层使用自定义的Decoder类
        self.dec4 = layers.Decoder(ch * (8 + 16), ch * 8, 3, 1, 1)
        self.dec3 = layers.Decoder(ch * (4 + 8), ch * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(ch * (2 + 4), ch * 2, 3, 1, 1)
        self.dec1 = layers.Decoder(ch * (1 + 2), ch, 3, 1, 1)

    def __call__(self, x):
        # 前向传播函数，依次通过四个编码器层和ASPP模块，然后通过四个解码器层
        h, e1 = self.enc1(x)
        h, e2 = self.enc2(h)
        h, e3 = self.enc3(h)
        h, e4 = self.enc4(h)

        h = self.aspp(h)

        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = self.dec1(h)

        return h


class CascadedASPPNet(nn.Module):
    def __init__(self, n_fft):
        super(CascadedASPPNet, self).__init__()
        # 初始化第一阶段低频带和高频带的BaseASPPNet网络
        self.stg1_low_band_net = BaseASPPNet(2, 32)
        self.stg1_high_band_net = BaseASPPNet(2, 32)

        # 第二阶段的桥接层和全频带的BaseASPPNet网络
        self.stg2_bridge = layers.Conv2DBNActiv(34, 16, 1, 1, 0)
        self.stg2_full_band_net = BaseASPPNet(16, 32)

        # 第三阶段的桥接层和全频带的BaseASPPNet网络
        self.stg3_bridge = layers.Conv2DBNActiv(66, 32, 1, 1, 0)
        self.stg3_full_band_net = BaseASPPNet(32, 64)

        # 输出层，输出通道为2，没有偏置
        self.out = nn.Conv2d(64, 2, 1, bias=False)
        self.aux1_out = nn.Conv2d(32, 2, 1, bias=False)  # 辅助输出1，输出通道为2，没有偏置
        self.aux2_out = nn.Conv2d(32, 2, 1, bias=False)  # 辅助输出2，输出通道为2，没有偏置

        # 最大频谱bin和输出频谱bin的设置
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        # 偏移量设置为128
        self.offset = 128
    # 前向传播函数，接受输入张量 x 和（可选的）攻击性参数 aggressiveness
    def forward(self, x, aggressiveness=None):
        # 将输入张量 x 的梯度分离出来，避免在反向传播时计算其梯度
        mix = x.detach()
        # 克隆输入张量 x，以防止修改原始数据
        x = x.clone()

        # 截取输入张量 x 的前 self.max_bin 列
        x = x[:, :, : self.max_bin]

        # 计算截取后的张量 x 的一半长度
        bandw = x.size()[2] // 2
        # 将张量 x 分成低频部分和高频部分，分别通过 stg1_low_band_net 和 stg1_high_band_net 处理
        aux1 = torch.cat(
            [
                self.stg1_low_band_net(x[:, :, :bandw]),
                self.stg1_high_band_net(x[:, :, bandw:]),
            ],
            dim=2,
        )

        # 将 x 和辅助计算的 aux1 拼接在一起
        h = torch.cat([x, aux1], dim=1)
        # 将拼接后的张量 h 经过 stg2_bridge 和 stg2_full_band_net 处理得到 aux2
        aux2 = self.stg2_full_band_net(self.stg2_bridge(h))

        # 将 x、aux1 和 aux2 拼接在一起
        h = torch.cat([x, aux1, aux2], dim=1)
        # 将拼接后的张量 h 经过 stg3_bridge 和 stg3_full_band_net 处理得到输出 h
        h = self.stg3_full_band_net(self.stg3_bridge(h))

        # 使用 sigmoid 函数处理输出 h，得到掩码 mask
        mask = torch.sigmoid(self.out(h))
        # 对 mask 进行填充，使其达到指定的输出长度 self.output_bin
        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode="replicate",
        )

        # 如果模型处于训练模式
        if self.training:
            # 使用 sigmoid 函数处理 aux1，并进行填充操作，得到处理后的 aux1
            aux1 = torch.sigmoid(self.aux1_out(aux1))
            aux1 = F.pad(
                input=aux1,
                pad=(0, 0, 0, self.output_bin - aux1.size()[2]),
                mode="replicate",
            )
            # 使用 sigmoid 函数处理 aux2，并进行填充操作，得到处理后的 aux2
            aux2 = torch.sigmoid(self.aux2_out(aux2))
            aux2 = F.pad(
                input=aux2,
                pad=(0, 0, 0, self.output_bin - aux2.size()[2]),
                mode="replicate",
            )
            # 返回带有混合数据的 mask、aux1 和 aux2
            return mask * mix, aux1 * mix, aux2 * mix
        else:
            # 如果未处于训练模式，根据给定的 aggressiveness 对 mask 进行进一步处理
            if aggressiveness:
                mask[:, :, : aggressiveness["split_bin"]] = torch.pow(
                    mask[:, :, : aggressiveness["split_bin"]],
                    1 + aggressiveness["value"] / 3,
                )
                mask[:, :, aggressiveness["split_bin"] :] = torch.pow(
                    mask[:, :, aggressiveness["split_bin"] :],
                    1 + aggressiveness["value"],
                )

            # 返回经过处理的 mask 与 mix 的乘积
            return mask * mix

    # 预测函数，接受输入 x_mag 和（可选的）攻击性参数 aggressiveness
    def predict(self, x_mag, aggressiveness=None):
        # 调用 forward 方法进行前向传播，得到预测结果 h
        h = self.forward(x_mag, aggressiveness)

        # 如果偏移量 offset 大于 0，则对 h 进行切片操作，去除偏移部分
        if self.offset > 0:
            h = h[:, :, :, self.offset : -self.offset]
            # 断言切片后的 h 的最后一维长度大于 0
            assert h.size()[3] > 0

        # 返回预测结果 h
        return h
```