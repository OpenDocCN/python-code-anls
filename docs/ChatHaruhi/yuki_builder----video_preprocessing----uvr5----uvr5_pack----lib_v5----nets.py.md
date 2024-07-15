# `.\Chat-Haruhi-Suzumiya\yuki_builder\video_preprocessing\uvr5\uvr5_pack\lib_v5\nets.py`

```py
# 导入PyTorch库
import torch
# 导入神经网络模块
from torch import nn
# 导入神经网络的函数操作模块
import torch.nn.functional as F

# 从当前目录下导入自定义的层模块
from . import layers

# 定义基础ASPP网络模型类
class BaseASPPNet(nn.Module):
    def __init__(self, nin, ch, dilations=(4, 8, 16)):
        super(BaseASPPNet, self).__init__()
        # 定义四个编码器层，每层具有不同的通道数和扩张率
        self.enc1 = layers.Encoder(nin, ch, 3, 2, 1)
        self.enc2 = layers.Encoder(ch, ch * 2, 3, 2, 1)
        self.enc3 = layers.Encoder(ch * 2, ch * 4, 3, 2, 1)
        self.enc4 = layers.Encoder(ch * 4, ch * 8, 3, 2, 1)

        # ASPP模块，接收最后一个编码器层的输出
        self.aspp = layers.ASPPModule(ch * 8, ch * 16, dilations)

        # 定义四个解码器层，与编码器对应，用于恢复特征图尺寸
        self.dec4 = layers.Decoder(ch * (8 + 16), ch * 8, 3, 1, 1)
        self.dec3 = layers.Decoder(ch * (4 + 8), ch * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(ch * (2 + 4), ch * 2, 3, 1, 1)
        self.dec1 = layers.Decoder(ch * (1 + 2), ch, 3, 1, 1)

    # 定义网络的前向传播逻辑
    def __call__(self, x):
        # 编码阶段，依次通过四个编码器层，并获取各层的编码输出
        h, e1 = self.enc1(x)
        h, e2 = self.enc2(h)
        h, e3 = self.enc3(h)
        h, e4 = self.enc4(h)

        # ASPP模块处理编码器的输出特征图
        h = self.aspp(h)

        # 解码阶段，依次通过四个解码器层，并传入对应编码器层的输出
        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = self.dec1(h, e1)

        # 返回最终的解码输出
        return h


# 级联ASPP网络模型类，继承自nn.Module
class CascadedASPPNet(nn.Module):
    # 初始化方法，接收一个n_fft参数
    def __init__(self, n_fft):
        super(CascadedASPPNet, self).__init__()
        # 第一阶段低频段网络和高频段网络，都使用BaseASPPNet模型
        self.stg1_low_band_net = BaseASPPNet(2, 16)
        self.stg1_high_band_net = BaseASPPNet(2, 16)

        # 第二阶段桥接层和全频段网络
        self.stg2_bridge = layers.Conv2DBNActiv(18, 8, 1, 1, 0)
        self.stg2_full_band_net = BaseASPPNet(8, 16)

        # 第三阶段桥接层和全频段网络
        self.stg3_bridge = layers.Conv2DBNActiv(34, 16, 1, 1, 0)
        self.stg3_full_band_net = BaseASPPNet(16, 32)

        # 输出层定义，分别用于整体输出和辅助输出
        self.out = nn.Conv2d(32, 2, 1, bias=False)
        self.aux1_out = nn.Conv2d(16, 2, 1, bias=False)
        self.aux2_out = nn.Conv2d(16, 2, 1, bias=False)

        # 设置最大频率和输出频率的参数
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        # 设置偏移量参数
        self.offset = 128
    # 定义神经网络的前向传播方法，接受输入 x 和可选的攻击性参数 aggressiveness
    def forward(self, x, aggressiveness=None):
        # 创建 x 的一个分离副本，用于避免梯度传播到原始张量
        mix = x.detach()
        # 克隆输入张量 x
        x = x.clone()

        # 限制输入张量 x 的宽度到 self.max_bin
        x = x[:, :, : self.max_bin]

        # 计算宽度一半的带宽
        bandw = x.size()[2] // 2
        # 分别对输入张量的低频和高频部分应用两个网络模块，并在第二维度上连接它们
        aux1 = torch.cat(
            [
                self.stg1_low_band_net(x[:, :, :bandw]),  # 低频部分网络输出
                self.stg1_high_band_net(x[:, :, bandw:]),  # 高频部分网络输出
            ],
            dim=2,
        )

        # 在第一维度上连接原始输入张量 x 和第一阶段辅助输出 aux1
        h = torch.cat([x, aux1], dim=1)
        # 对连接后的张量应用第二阶段全频带网络模块，并传递给桥接层处理
        aux2 = self.stg2_full_band_net(self.stg2_bridge(h))

        # 再次在第一维度上连接输入张量 x、第一阶段输出 aux1 和第二阶段输出 aux2
        h = torch.cat([x, aux1, aux2], dim=1)
        # 应用第三阶段全频带网络模块并传递给输出层
        h = self.stg3_full_band_net(self.stg3_bridge(h))

        # 经过输出层并应用 sigmoid 函数得到最终的 mask
        mask = torch.sigmoid(self.out(h))
        # 使用 F.pad 函数在最后一个维度上填充 mask，使其达到预期的输出宽度 self.output_bin
        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode="replicate",
        )

        # 如果处于训练模式
        if self.training:
            # 对 aux1 应用 sigmoid 函数并在最后一个维度上进行填充
            aux1 = torch.sigmoid(self.aux1_out(aux1))
            aux1 = F.pad(
                input=aux1,
                pad=(0, 0, 0, self.output_bin - aux1.size()[2]),
                mode="replicate",
            )
            # 对 aux2 应用 sigmoid 函数并在最后一个维度上进行填充
            aux2 = torch.sigmoid(self.aux2_out(aux2))
            aux2 = F.pad(
                input=aux2,
                pad=(0, 0, 0, self.output_bin - aux2.size()[2]),
                mode="replicate",
            )
            # 返回经过 mask 处理的 mix，aux1，aux2 的结果
            return mask * mix, aux1 * mix, aux2 * mix
        else:
            # 如果非训练模式，根据 aggressiveness 参数对 mask 进行进一步处理
            if aggressiveness:
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

    # 定义预测方法，接受输入 x_mag 和可选的攻击性参数 aggressiveness
    def predict(self, x_mag, aggressiveness=None):
        # 调用 forward 方法进行前向传播得到预测结果 h
        h = self.forward(x_mag, aggressiveness)

        # 如果存在偏移量大于 0，则对 h 在第四个维度上进行切片以去除偏移量部分，并确保结果长度大于 0
        if self.offset > 0:
            h = h[:, :, :, self.offset : -self.offset]
            assert h.size()[3] > 0

        # 返回预测结果 h
        return h
```