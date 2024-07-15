# `.\Chat-Haruhi-Suzumiya\yuki_builder\video_preprocessing\uvr5\uvr5_pack\lib_v5\nets_33966KB.py`

```py
import torch
from torch import nn
import torch.nn.functional as F

from . import layers_33966KB as layers

class BaseASPPNet(nn.Module):
    def __init__(self, nin, ch, dilations=(4, 8, 16, 32)):
        super(BaseASPPNet, self).__init__()
        # 定义四个Encoder层，逐步增加通道数和空洞卷积的扩张率
        self.enc1 = layers.Encoder(nin, ch, 3, 2, 1)
        self.enc2 = layers.Encoder(ch, ch * 2, 3, 2, 1)
        self.enc3 = layers.Encoder(ch * 2, ch * 4, 3, 2, 1)
        self.enc4 = layers.Encoder(ch * 4, ch * 8, 3, 2, 1)

        # ASPP模块接受ch * 8通道的输入，使用指定的多个扩张率
        self.aspp = layers.ASPPModule(ch * 8, ch * 16, dilations)

        # 定义四个Decoder层，逐步减少通道数，恢复到原始输入通道数
        self.dec4 = layers.Decoder(ch * (8 + 16), ch * 8, 3, 1, 1)
        self.dec3 = layers.Decoder(ch * (4 + 8), ch * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(ch * (2 + 4), ch * 2, 3, 1, 1)
        self.dec1 = layers.Decoder(ch * (1 + 2), ch, 3, 1, 1)

    def __call__(self, x):
        # Encoder阶段，对输入x逐步进行编码，得到中间特征e1, e2, e3, e4
        h, e1 = self.enc1(x)
        h, e2 = self.enc2(h)
        h, e3 = self.enc3(h)
        h, e4 = self.enc4(h)

        # ASPP模块处理中间特征h，增强其对多尺度信息的感知能力
        h = self.aspp(h)

        # Decoder阶段，对处理后的特征h进行逐步解码，恢复到原始输入尺寸
        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = self.dec1(h, e1)

        # 返回最终的解码结果h
        return h


class CascadedASPPNet(nn.Module):
    def __init__(self, n_fft):
        super(CascadedASPPNet, self).__init__()
        # 定义两个BaseASPPNet实例，用于低频段和高频段的处理
        self.stg1_low_band_net = BaseASPPNet(2, 16)
        self.stg1_high_band_net = BaseASPPNet(2, 16)

        # 第二阶段的连接层，将两个BaseASPPNet的输出合并为一个输入
        self.stg2_bridge = layers.Conv2DBNActiv(18, 8, 1, 1, 0)
        self.stg2_full_band_net = BaseASPPNet(8, 16)

        # 第三阶段的连接层，将两个BaseASPPNet的输出合并为一个输入
        self.stg3_bridge = layers.Conv2DBNActiv(34, 16, 1, 1, 0)
        self.stg3_full_band_net = BaseASPPNet(16, 32)

        # 输出层，分别输出最终结果和辅助结果
        self.out = nn.Conv2d(32, 2, 1, bias=False)
        self.aux1_out = nn.Conv2d(16, 2, 1, bias=False)
        self.aux2_out = nn.Conv2d(16, 2, 1, bias=False)

        # 初始化参数
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.offset = 128
    # 定义前向传播函数，接受输入张量 x 和可选的激进度参数
    def forward(self, x, aggressiveness=None):
        # 创建 x 的一个分离版本，用于计算梯度
        mix = x.detach()
        # 克隆输入张量 x
        x = x.clone()

        # 限制 x 的第三维度的大小为 max_bin
        x = x[:, :, : self.max_bin]

        # 计算第三维度大小的一半作为分频带标记
        bandw = x.size()[2] // 2
        # 将输入 x 的前半部分和后半部分分别通过两个网络处理后连接起来
        aux1 = torch.cat(
            [
                self.stg1_low_band_net(x[:, :, :bandw]),  # 低频带网络处理
                self.stg1_high_band_net(x[:, :, bandw:]),  # 高频带网络处理
            ],
            dim=2,
        )

        # 将 x 和辅助变量 aux1 拼接在第二维度上
        h = torch.cat([x, aux1], dim=1)
        # 通过一个桥接层和一个全带网络处理 h，得到第二个辅助变量 aux2
        aux2 = self.stg2_full_band_net(self.stg2_bridge(h))

        # 将 x、aux1 和 aux2 拼接在第二维度上
        h = torch.cat([x, aux1, aux2], dim=1)
        # 通过第三个全带网络处理 h，得到最终的输出 h
        h = self.stg3_full_band_net(self.stg3_bridge(h))

        # 使用 sigmoid 函数处理最终输出 h，得到掩码 mask
        mask = torch.sigmoid(self.out(h))
        # 在第三维度上使用填充操作，使得 mask 的大小变为 output_bin
        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode="replicate",
        )

        # 如果处于训练模式
        if self.training:
            # 使用 sigmoid 函数处理 aux1 和 aux2，并进行与 mask 相同的填充操作
            aux1 = torch.sigmoid(self.aux1_out(aux1))
            aux1 = F.pad(
                input=aux1,
                pad=(0, 0, 0, self.output_bin - aux1.size()[2]),
                mode="replicate",
            )
            aux2 = torch.sigmoid(self.aux2_out(aux2))
            aux2 = F.pad(
                input=aux2,
                pad=(0, 0, 0, self.output_bin - aux2.size()[2]),
                mode="replicate",
            )
            # 返回经过 mask 处理的 mix、aux1 和 aux2
            return mask * mix, aux1 * mix, aux2 * mix
        else:
            # 如果未处于训练模式，并且传入了 aggressiveness 参数
            if aggressiveness:
                # 对 mask 进行分段指数变换，根据 aggressiveness 参数的不同部分
                mask[:, :, : aggressiveness["split_bin"]] = torch.pow(
                    mask[:, :, : aggressiveness["split_bin"]],
                    1 + aggressiveness["value"] / 3,
                )
                mask[:, :, aggressiveness["split_bin"] :] = torch.pow(
                    mask[:, :, aggressiveness["split_bin"] :],
                    1 + aggressiveness["value"],
                )

            # 返回经过 mask 处理的 mix
            return mask * mix

    # 定义预测函数，接受输入 x_mag 和可选的激进度参数
    def predict(self, x_mag, aggressiveness=None):
        # 调用前向传播函数计算 h
        h = self.forward(x_mag, aggressiveness)

        # 如果存在偏移量 offset，对 h 进行切片操作，确保切片后大小大于零
        if self.offset > 0:
            h = h[:, :, :, self.offset : -self.offset]
            assert h.size()[3] > 0

        # 返回处理后的 h
        return h
```