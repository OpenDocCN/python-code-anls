# `.\Chat-Haruhi-Suzumiya\yuki_builder\video_preprocessing\uvr5\uvr5_pack\lib_v5\layers_new.py`

```py
import torch  # 导入PyTorch库
from torch import nn  # 从PyTorch中导入神经网络模块
import torch.nn.functional as F  # 导入PyTorch中的函数式接口

from . import spec_utils  # 导入当前包中的spec_utils模块


class Conv2DBNActiv(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(Conv2DBNActiv, self).__init__()
        self.conv = nn.Sequential(  # 定义一个包含卷积、批归一化和激活函数的序列
            nn.Conv2d(  # 二维卷积层
                nin,  # 输入通道数
                nout,  # 输出通道数
                kernel_size=ksize,  # 卷积核大小
                stride=stride,  # 步长
                padding=pad,  # 填充
                dilation=dilation,  # 空洞卷积率
                bias=False,  # 不使用偏置项
            ),
            nn.BatchNorm2d(nout),  # 二维批归一化层
            activ(),  # 激活函数，通过参数传入
        )

    def __call__(self, x):
        return self.conv(x)  # 返回卷积操作的结果


class Encoder(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.LeakyReLU):
        super(Encoder, self).__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, stride, pad, activ=activ)  # 第一个卷积层
        self.conv2 = Conv2DBNActiv(nout, nout, ksize, 1, pad, activ=activ)  # 第二个卷积层

    def __call__(self, x):
        h = self.conv1(x)  # 第一层卷积的输出
        h = self.conv2(h)  # 第二层卷积的输出

        return h  # 返回最终输出


class Decoder(nn.Module):
    def __init__(
        self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.ReLU, dropout=False
    ):
        super(Decoder, self).__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)  # 单个卷积层
        # self.conv2 = Conv2DBNActiv(nout, nout, ksize, 1, pad, activ=activ)  # 可选的第二个卷积层
        self.dropout = nn.Dropout2d(0.1) if dropout else None  # Dropout层，根据参数决定是否添加

    def __call__(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)  # 双线性插值上采样

        if skip is not None:
            skip = spec_utils.crop_center(skip, x)  # 使用spec_utils中的函数裁剪中心部分
            x = torch.cat([x, skip], dim=1)  # 在通道维度上拼接张量

        h = self.conv1(x)  # 卷积操作

        if self.dropout is not None:
            h = self.dropout(h)  # 使用Dropout层

        return h  # 返回最终输出


class ASPPModule(nn.Module):
    def __init__(self, nin, nout, dilations=(4, 8, 12), activ=nn.ReLU, dropout=False):
        super(ASPPModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),  # 自适应平均池化，第一步
            Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ),  # 卷积操作，第二步
        )
        self.conv2 = Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ)  # 单个卷积层，第三步
        self.conv3 = Conv2DBNActiv(  # 带空洞卷积的卷积操作，第四步
            nin, nout, 3, 1, dilations[0], dilations[0], activ=activ
        )
        self.conv4 = Conv2DBNActiv(  # 带空洞卷积的卷积操作，第五步
            nin, nout, 3, 1, dilations[1], dilations[1], activ=activ
        )
        self.conv5 = Conv2DBNActiv(  # 带空洞卷积的卷积操作，第六步
            nin, nout, 3, 1, dilations[2], dilations[2], activ=activ
        )
        self.bottleneck = Conv2DBNActiv(nout * 5, nout, 1, 1, 0, activ=activ)  # 卷积操作，第七步
        self.dropout = nn.Dropout2d(0.1) if dropout else None  # Dropout层，根据参数决定是否添加
    # 定义前向传播函数，接受输入张量 x
    def forward(self, x):
        # 从输入张量 x 中获取其尺寸信息，并解包为宽度 w 和高度 h
        _, _, h, w = x.size()
        # 对输入张量 x 进行卷积操作，并使用双线性插值将结果调整到尺寸 (h, w)
        feat1 = F.interpolate(
            self.conv1(x), size=(h, w), mode="bilinear", align_corners=True
        )
        # 对输入张量 x 进行第二个卷积操作
        feat2 = self.conv2(x)
        # 对输入张量 x 进行第三个卷积操作
        feat3 = self.conv3(x)
        # 对输入张量 x 进行第四个卷积操作
        feat4 = self.conv4(x)
        # 对输入张量 x 进行第五个卷积操作
        feat5 = self.conv5(x)
        # 将 feat1 到 feat5 的特征张量沿着通道维度拼接起来
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        # 将拼接后的特征张量输入到瓶颈层（bottleneck）中进行进一步处理
        out = self.bottleneck(out)

        # 如果存在 dropout 层，则对 out 进行 dropout 操作
        if self.dropout is not None:
            out = self.dropout(out)

        # 返回处理后的输出张量 out
        return out
# 定义一个名为 LSTMModule 的神经网络模块类，继承自 nn.Module
class LSTMModule(nn.Module):
    # 初始化函数，接收卷积层输入维度 nin_conv、LSTM 输入维度 nin_lstm 和输出维度 nout_lstm
    def __init__(self, nin_conv, nin_lstm, nout_lstm):
        super(LSTMModule, self).__init__()
        # 使用 Conv2DBNActiv 类创建一个卷积层对象 self.conv
        self.conv = Conv2DBNActiv(nin_conv, 1, 1, 1, 0)
        # 使用 nn.LSTM 创建一个 LSTM 层对象 self.lstm，设置输入维度为 nin_lstm，隐藏状态维度为 nout_lstm // 2，双向
        self.lstm = nn.LSTM(
            input_size=nin_lstm, hidden_size=nout_lstm // 2, bidirectional=True
        )
        # 使用 nn.Sequential 创建一个包含线性层、批标准化层和ReLU激活函数的序列 self.dense
        self.dense = nn.Sequential(
            nn.Linear(nout_lstm, nin_lstm), nn.BatchNorm1d(nin_lstm), nn.ReLU()
        )

    # 前向传播函数，接收输入张量 x
    def forward(self, x):
        # 获取输入张量的维度信息：批大小 N、通道数、频率条数 nbins、帧数 nframes
        N, _, nbins, nframes = x.size()
        # 将输入张量 x 传入卷积层 self.conv，取出其输出的第一个通道，得到 h，维度为 N, nbins, nframes
        h = self.conv(x)[:, 0]  # N, nbins, nframes
        # 将 h 的维度进行置换，变为 nframes, N, nbins
        h = h.permute(2, 0, 1)  # nframes, N, nbins
        # 将置换后的 h 传入 LSTM 层 self.lstm，得到输出 h 和隐藏状态 _
        h, _ = self.lstm(h)
        # 将 LSTM 输出 h 的最后一个维度展平，并传入线性层和批标准化层 self.dense，得到 h
        h = self.dense(h.reshape(-1, h.size()[-1]))  # nframes * N, nbins
        # 将 h 重新变形为 nframes, N, 1, nbins
        h = h.reshape(nframes, N, 1, nbins)
        # 再次置换 h 的维度，变为 N, 1, nbins, nframes
        h = h.permute(1, 2, 3, 0)

        # 返回前向传播的结果张量 h
        return h
```