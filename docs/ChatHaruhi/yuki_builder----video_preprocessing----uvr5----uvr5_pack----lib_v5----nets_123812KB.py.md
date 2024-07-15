# `.\Chat-Haruhi-Suzumiya\yuki_builder\video_preprocessing\uvr5\uvr5_pack\lib_v5\nets_123812KB.py`

```py
import torch
from torch import nn
import torch.nn.functional as F

from . import layers_123821KB as layers  # 导入自定义模块layers，用于定义网络层


class BaseASPPNet(nn.Module):
    def __init__(self, nin, ch, dilations=(4, 8, 16)):
        super(BaseASPPNet, self).__init__()
        # 定义四个编码器层，逐步提取特征
        self.enc1 = layers.Encoder(nin, ch, 3, 2, 1)  # 输入通道nin，输出通道ch
        self.enc2 = layers.Encoder(ch, ch * 2, 3, 2, 1)  # 输入通道ch，输出通道ch*2
        self.enc3 = layers.Encoder(ch * 2, ch * 4, 3, 2, 1)  # 输入通道ch*2，输出通道ch*4
        self.enc4 = layers.Encoder(ch * 4, ch * 8, 3, 2, 1)  # 输入通道ch*4，输出通道ch*8

        # ASPP模块用于多尺度信息聚合
        self.aspp = layers.ASPPModule(ch * 8, ch * 16, dilations)  # 输入通道ch*8，输出通道ch*16，采用指定扩张率dilations

        # 定义四个解码器层，逐步恢复特征图尺寸
        self.dec4 = layers.Decoder(ch * (8 + 16), ch * 8, 3, 1, 1)  # 输入通道为ch*(8+16)，输出通道为ch*8
        self.dec3 = layers.Decoder(ch * (4 + 8), ch * 4, 3, 1, 1)  # 输入通道为ch*(4+8)，输出通道为ch*4
        self.dec2 = layers.Decoder(ch * (2 + 4), ch * 2, 3, 1, 1)  # 输入通道为ch*(2+4)，输出通道为ch*2
        self.dec1 = layers.Decoder(ch * (1 + 2), ch, 3, 1, 1)  # 输入通道为ch*(1+2)，输出通道为ch

    def __call__(self, x):
        # 编码阶段
        h, e1 = self.enc1(x)  # 第一编码器，得到特征h和编码结果e1
        h, e2 = self.enc2(h)  # 第二编码器，得到特征h和编码结果e2
        h, e3 = self.enc3(h)  # 第三编码器，得到特征h和编码结果e3
        h, e4 = self.enc4(h)  # 第四编码器，得到特征h和编码结果e4

        # ASPP模块，多尺度特征聚合
        h = self.aspp(h)

        # 解码阶段
        h = self.dec4(h, e4)  # 第四解码器，恢复特征h，使用e4辅助
        h = self.dec3(h, e3)  # 第三解码器，恢复特征h，使用e3辅助
        h = self.dec2(h, e2)  # 第二解码器，恢复特征h，使用e2辅助
        h = self.dec1(h, e1)  # 第一解码器，恢复特征h，使用e1辅助

        return h


class CascadedASPPNet(nn.Module):
    def __init__(self, n_fft):
        super(CascadedASPPNet, self).__init__()
        # 阶段1低频带网络和高频带网络
        self.stg1_low_band_net = BaseASPPNet(2, 32)  # 输入通道为2，输出通道为32
        self.stg1_high_band_net = BaseASPPNet(2, 32)  # 输入通道为2，输出通道为32

        # 阶段2桥接层和全频带网络
        self.stg2_bridge = layers.Conv2DBNActiv(34, 16, 1, 1, 0)  # 输入通道34，输出通道16
        self.stg2_full_band_net = BaseASPPNet(16, 32)  # 输入通道为16，输出通道为32

        # 阶段3桥接层和全频带网络
        self.stg3_bridge = layers.Conv2DBNActiv(66, 32, 1, 1, 0)  # 输入通道66，输出通道32
        self.stg3_full_band_net = BaseASPPNet(32, 64)  # 输入通道为32，输出通道为64

        # 输出层和辅助输出层
        self.out = nn.Conv2d(64, 2, 1, bias=False)  # 输出通道为2
        self.aux1_out = nn.Conv2d(32, 2, 1, bias=False)  # 输出通道为2
        self.aux2_out = nn.Conv2d(32, 2, 1, bias=False)  # 输出通道为2

        # 最大频率和输出频率
        self.max_bin = n_fft // 2  # 最大频率为n_fft的一半
        self.output_bin = n_fft // 2 + 1  # 输出频率为n_fft的一半加一

        # 偏移量
        self.offset = 128  # 偏移量为128
    # 定义前向传播方法，接收输入张量 x 和可选的攻击性参数
    def forward(self, x, aggressiveness=None):
        # 创建 x 的副本并且分离出来，以便后续计算梯度
        mix = x.detach()
        # 克隆输入张量 x
        x = x.clone()

        # 限制输入张量 x 的第三维度长度为 self.max_bin
        x = x[:, :, : self.max_bin]

        # 计算频带宽度
        bandw = x.size()[2] // 2
        # 分别通过低频和高频网络处理输入张量 x 的两部分
        aux1 = torch.cat(
            [
                self.stg1_low_band_net(x[:, :, :bandw]),
                self.stg1_high_band_net(x[:, :, bandw:]),
            ],
            dim=2,
        )

        # 将原始输入 x 和辅助特征 aux1 拼接起来
        h = torch.cat([x, aux1], dim=1)
        # 通过阶段2的全频段网络处理连接后的特征
        aux2 = self.stg2_full_band_net(self.stg2_bridge(h))

        # 再次将原始输入 x、辅助特征 aux1 和辅助特征 aux2 拼接起来
        h = torch.cat([x, aux1, aux2], dim=1)
        # 通过阶段3的全频段网络处理连接后的特征
        h = self.stg3_full_band_net(self.stg3_bridge(h))

        # 使用 sigmoid 函数生成输出的掩码 mask
        mask = torch.sigmoid(self.out(h))
        # 使用 replicate 模式填充 mask，使其达到指定的输出维度
        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode="replicate",
        )

        # 如果处于训练模式
        if self.training:
            # 使用 sigmoid 函数生成辅助输出 aux1 和 aux2
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
            # 返回带有掩码和辅助输出的加权结果
            return mask * mix, aux1 * mix, aux2 * mix
        else:
            # 如果未处于训练模式，根据给定的 aggressiveness 对掩码进行调整
            if aggressiveness:
                mask[:, :, : aggressiveness["split_bin"]] = torch.pow(
                    mask[:, :, : aggressiveness["split_bin"]],
                    1 + aggressiveness["value"] / 3,
                )
                mask[:, :, aggressiveness["split_bin"] :] = torch.pow(
                    mask[:, :, aggressiveness["split_bin"] :],
                    1 + aggressiveness["value"],
                )
            # 返回调整后的掩码结果
            return mask * mix

    # 定义预测方法，接收输入 x_mag 和可选的攻击性参数
    def predict(self, x_mag, aggressiveness=None):
        # 调用前向传播方法获取预测结果 h
        h = self.forward(x_mag, aggressiveness)

        # 如果偏移量大于 0，则对 h 进行切片操作以去除偏移后的结果
        if self.offset > 0:
            h = h[:, :, :, self.offset : -self.offset]
            assert h.size()[3] > 0  # 断言确保切片操作后 h 的长度仍大于 0

        # 返回最终预测结果 h
        return h
```