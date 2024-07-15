# `.\Chat-Haruhi-Suzumiya\yuki_builder\video_preprocessing\uvr5\uvr5_pack\lib_v5\nets_new.py`

```py
import torch
from torch import nn
import torch.nn.functional as F
from . import layers_new as layers

# 定义基础网络类
class BaseNet(nn.Module):
    def __init__(
        self, nin, nout, nin_lstm, nout_lstm, dilations=((4, 2), (8, 4), (12, 6))
    ):
        super(BaseNet, self).__init__()
        # 创建卷积层、BatchNorm和激活函数的组合，用于网络的第一层 
        self.enc1 = layers.Conv2DBNActiv(nin, nout, 3, 1, 1)
        # 创建编码器层，用于提取特征
        self.enc2 = layers.Encoder(nout, nout * 2, 3, 2, 1)
        self.enc3 = layers.Encoder(nout * 2, nout * 4, 3, 2, 1)
        self.enc4 = layers.Encoder(nout * 4, nout * 6, 3, 2, 1)
        self.enc5 = layers.Encoder(nout * 6, nout * 8, 3, 2, 1)

        # 创建ASPP模块，进行多尺度特征提取
        self.aspp = layers.ASPPModule(nout * 8, nout * 8, dilations, dropout=True)

        # 创建解码器层，恢复输出分辨率
        self.dec4 = layers.Decoder(nout * (6 + 8), nout * 6, 3, 1, 1)
        self.dec3 = layers.Decoder(nout * (4 + 6), nout * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(nout * (2 + 4), nout * 2, 3, 1, 1)
        # 创建LSTM模块，参与解码操作
        self.lstm_dec2 = layers.LSTMModule(nout * 2, nin_lstm, nout_lstm)
        self.dec1 = layers.Decoder(nout * (1 + 2) + 1, nout * 1, 3, 1, 1)

    # 网络前向传播函数
    def __call__(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        h = self.aspp(e5)

        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = torch.cat([h, self.lstm_dec2(h)], dim=1)
        h = self.dec1(h, e1)

        return h

# 定义级联网络类
class CascadedNet(nn.Module):
    def __init__(self, n_fft, nout=32, nout_lstm=128):
        super(CascadedNet, self).__init__()

        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.nin_lstm = self.max_bin // 2
        self.offset = 64

        # 第一阶段低频带网络
        self.stg1_low_band_net = nn.Sequential(
            BaseNet(2, nout // 2, self.nin_lstm // 2, nout_lstm),
            layers.Conv2DBNActiv(nout // 2, nout // 4, 1, 1, 0),
        )

        # 第一阶段高频带网络
        self.stg1_high_band_net = BaseNet(
            2, nout // 4, self.nin_lstm // 2, nout_lstm // 2
        )

        # 第二阶段低频带网络
        self.stg2_low_band_net = nn.Sequential(
            BaseNet(nout // 4 + 2, nout, self.nin_lstm // 2, nout_lstm),
            layers.Conv2DBNActiv(nout, nout // 2, 1, 1, 0),
        )

        # 第二阶段高频带网络
        self.stg2_high_band_net = BaseNet(
            nout // 4 + 2, nout // 2, self.nin_lstm // 2, nout_lstm // 2
        )

        # 第三阶段全频带网络
        self.stg3_full_band_net = BaseNet(
            3 * nout // 4 + 2, nout, self.nin_lstm, nout_lstm
        )

        # 输出层和辅助输出层
        self.out = nn.Conv2d(nout, 2, 1, bias=False)
        self.aux_out = nn.Conv2d(3 * nout // 4, 2, 1, bias=False)
    # 定义神经网络的前向传播函数，接受输入张量 x
    def forward(self, x):
        # 仅保留输入张量 x 的前 self.max_bin 列
        x = x[:, :, : self.max_bin]

        # 计算中间带宽的位置
        bandw = x.size()[2] // 2
        # 分别提取低频和高频部分的输入数据
        l1_in = x[:, :, :bandw]
        h1_in = x[:, :, bandw:]
        # 将低频和高频部分输入到第一个阶段的低频和高频网络中
        l1 = self.stg1_low_band_net(l1_in)
        h1 = self.stg1_high_band_net(h1_in)
        # 将低频和高频部分的输出拼接在一起形成辅助数据1
        aux1 = torch.cat([l1, h1], dim=2)

        # 将第一阶段的低频部分 l1_in 和 l1 以及高频部分 h1_in 和 h1 拼接起来作为第二阶段的输入
        l2_in = torch.cat([l1_in, l1], dim=1)
        h2_in = torch.cat([h1_in, h1], dim=1)
        # 将拼接后的低频和高频部分输入到第二个阶段的低频和高频网络中
        l2 = self.stg2_low_band_net(l2_in)
        h2 = self.stg2_high_band_net(h2_in)
        # 将第二阶段的低频和高频部分的输出拼接在一起形成辅助数据2
        aux2 = torch.cat([l2, h2], dim=2)

        # 将原始输入 x、辅助数据1 aux1 和 辅助数据2 aux2 拼接在一起作为第三阶段的输入
        f3_in = torch.cat([x, aux1, aux2], dim=1)
        # 将拼接后的数据输入到第三阶段的全频段网络中
        f3 = self.stg3_full_band_net(f3_in)

        # 使用 sigmoid 函数将输出转换为介于 0 和 1 之间的概率掩码
        mask = torch.sigmoid(self.out(f3))
        # 使用 F.pad 函数对掩码进行填充，保证其维度与 self.output_bin 一致
        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode="replicate",
        )

        # 如果处于训练模式
        if self.training:
            # 将辅助数据 aux1 和 aux2 拼接在一起
            aux = torch.cat([aux1, aux2], dim=1)
            # 使用 sigmoid 函数处理辅助数据
            aux = torch.sigmoid(self.aux_out(aux))
            # 对处理后的辅助数据进行填充，保证其维度与 self.output_bin 一致
            aux = F.pad(
                input=aux,
                pad=(0, 0, 0, self.output_bin - aux.size()[2]),
                mode="replicate",
            )
            # 返回掩码和辅助数据
            return mask, aux
        else:
            # 如果不处于训练模式，仅返回掩码
            return mask

    # 预测掩码的函数，接受输入张量 x
    def predict_mask(self, x):
        # 调用前向传播函数计算掩码
        mask = self.forward(x)

        # 如果存在偏移量 self.offset
        if self.offset > 0:
            # 截取掩码在第三个维度上去掉偏移量后的部分
            mask = mask[:, :, :, self.offset : -self.offset]
            # 断言截取后的掩码维度大于 0
            assert mask.size()[3] > 0

        # 返回预测的掩码
        return mask

    # 预测函数，接受输入张量 x 和可选参数 aggressiveness
    def predict(self, x, aggressiveness=None):
        # 调用前向传播函数计算掩码
        mask = self.forward(x)
        # 将原始输入张量 x 乘以掩码得到预测的幅度谱
        pred_mag = x * mask

        # 如果存在偏移量 self.offset
        if self.offset > 0:
            # 截取预测幅度谱在第三个维度上去掉偏移量后的部分
            pred_mag = pred_mag[:, :, :, self.offset : -self.offset]
            # 断言截取后的预测幅度谱维度大于 0
            assert pred_mag.size()[3] > 0

        # 返回预测的幅度谱
        return pred_mag
```