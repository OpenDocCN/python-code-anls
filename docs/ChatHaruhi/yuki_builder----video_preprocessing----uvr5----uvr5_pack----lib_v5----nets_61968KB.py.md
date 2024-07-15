# `.\Chat-Haruhi-Suzumiya\yuki_builder\video_preprocessing\uvr5\uvr5_pack\lib_v5\nets_61968KB.py`

```py
# 导入 PyTorch 库
import torch
# 导入神经网络模块
from torch import nn
# 导入神经网络中常用的函数
import torch.nn.functional as F

# 导入自定义的层模块
from . import layers_123821KB as layers

# 定义基础 ASPP 网络类
class BaseASPPNet(nn.Module):
    def __init__(self, nin, ch, dilations=(4, 8, 16)):
        super(BaseASPPNet, self).__init__()
        # 编码器阶段：四个编码器层
        self.enc1 = layers.Encoder(nin, ch, 3, 2, 1)
        self.enc2 = layers.Encoder(ch, ch * 2, 3, 2, 1)
        self.enc3 = layers.Encoder(ch * 2, ch * 4, 3, 2, 1)
        self.enc4 = layers.Encoder(ch * 4, ch * 8, 3, 2, 1)

        # ASPP 模块
        self.aspp = layers.ASPPModule(ch * 8, ch * 16, dilations)

        # 解码器阶段：四个解码器层
        self.dec4 = layers.Decoder(ch * (8 + 16), ch * 8, 3, 1, 1)
        self.dec3 = layers.Decoder(ch * (4 + 8), ch * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(ch * (2 + 4), ch * 2, 3, 1, 1)
        self.dec1 = layers.Decoder(ch * (1 + 2), ch, 3, 1, 1)

    # 前向传播方法
    def __call__(self, x):
        # 编码阶段
        h, e1 = self.enc1(x)
        h, e2 = self.enc2(h)
        h, e3 = self.enc3(h)
        h, e4 = self.enc4(h)

        # ASPP 模块
        h = self.aspp(h)

        # 解码阶段
        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = self.dec1(h, e1)

        return h


# 级联 ASPP 网络类
class CascadedASPPNet(nn.Module):
    def __init__(self, n_fft):
        super(CascadedASPPNet, self).__init__()
        # 阶段 1 低频带网络和高频带网络
        self.stg1_low_band_net = BaseASPPNet(2, 32)
        self.stg1_high_band_net = BaseASPPNet(2, 32)

        # 阶段 2 桥接层和全频带网络
        self.stg2_bridge = layers.Conv2DBNActiv(34, 16, 1, 1, 0)
        self.stg2_full_band_net = BaseASPPNet(16, 32)

        # 阶段 3 桥接层和全频带网络
        self.stg3_bridge = layers.Conv2DBNActiv(66, 32, 1, 1, 0)
        self.stg3_full_band_net = BaseASPPNet(32, 64)

        # 输出层定义
        self.out = nn.Conv2d(64, 2, 1, bias=False)
        self.aux1_out = nn.Conv2d(32, 2, 1, bias=False)
        self.aux2_out = nn.Conv2d(32, 2, 1, bias=False)

        # 最大频谱线和输出频谱线设置
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        # 偏移设置
        self.offset = 128
    # 前向传播函数，接受输入张量 x 和可选的攻击性参数 aggressiveness
    def forward(self, x, aggressiveness=None):
        # 将输入张量 x 的副本保存到 mix 中，并断开与计算图的连接
        mix = x.detach()
        # 克隆输入张量 x，以便后续修改
        x = x.clone()

        # 限制输入张量 x 的第三维度不超过 self.max_bin
        x = x[:, :, : self.max_bin]

        # 计算第三维度的一半作为 bandw
        bandw = x.size()[2] // 2
        # 将输入张量 x 分成两部分，分别经过低频和高频网络处理，并在第三维度上连接结果
        aux1 = torch.cat(
            [
                self.stg1_low_band_net(x[:, :, :bandw]),
                self.stg1_high_band_net(x[:, :, bandw:]),
            ],
            dim=2,
        )

        # 在第二维度上连接输入张量 x 和辅助变量 aux1，形成新的张量 h
        h = torch.cat([x, aux1], dim=1)
        # 经过第二阶段的全频段网络和桥接层处理辅助变量 aux2
        aux2 = self.stg2_full_band_net(self.stg2_bridge(h))

        # 再次在第二维度上连接输入张量 x、aux1 和 aux2，形成最终的张量 h
        h = torch.cat([x, aux1, aux2], dim=1)
        # 经过第三阶段的全频段网络处理最终的张量 h
        h = self.stg3_full_band_net(self.stg3_bridge(h))

        # 经过 sigmoid 函数处理输出层，得到 mask，并对 mask 进行维度扩展填充
        mask = torch.sigmoid(self.out(h))
        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode="replicate",
        )

        # 如果处于训练模式
        if self.training:
            # 分别对 aux1 和 aux2 经过 sigmoid 函数处理，并对其进行维度扩展填充
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
            # 返回经过 mask 处理后的 mix 、aux1 和 aux2
            return mask * mix, aux1 * mix, aux2 * mix
        else:
            # 如果提供了 aggressiveness 参数，则根据参数对 mask 进行进一步处理
            if aggressiveness:
                mask[:, :, : aggressiveness["split_bin"]] = torch.pow(
                    mask[:, :, : aggressiveness["split_bin"]],
                    1 + aggressiveness["value"] / 3,
                )
                mask[:, :, aggressiveness["split_bin"] :] = torch.pow(
                    mask[:, :, aggressiveness["split_bin"] :],
                    1 + aggressiveness["value"],
                )
            # 返回经过 mask 处理后的 mix
            return mask * mix

    # 预测函数，接受输入 x_mag 和可选的攻击性参数 aggressiveness
    def predict(self, x_mag, aggressiveness=None):
        # 调用前向传播函数，得到预测结果 h
        h = self.forward(x_mag, aggressiveness)

        # 如果存在偏移量 self.offset，则对 h 进行切片操作
        if self.offset > 0:
            h = h[:, :, :, self.offset : -self.offset]
            # 断言切片后的 h 第四维度长度大于零
            assert h.size()[3] > 0

        # 返回预测结果 h
        return h
```