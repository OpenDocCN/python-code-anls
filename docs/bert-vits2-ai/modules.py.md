# `Bert-VITS2\modules.py`

```
import math  # 导入 math 模块
import torch  # 导入 torch 模块
from torch import nn  # 从 torch 模块中导入 nn 模块
from torch.nn import functional as F  # 从 torch.nn 模块中导入 functional 模块并重命名为 F

from torch.nn import Conv1d  # 从 torch.nn 模块中导入 Conv1d 类
from torch.nn.utils import weight_norm, remove_weight_norm  # 从 torch.nn.utils 模块中导入 weight_norm 和 remove_weight_norm 函数

import commons  # 导入自定义的 commons 模块
from commons import init_weights, get_padding  # 从 commons 模块中导入 init_weights 和 get_padding 函数
from transforms import piecewise_rational_quadratic_transform  # 从 transforms 模块中导入 piecewise_rational_quadratic_transform 函数
from attentions import Encoder  # 从 attentions 模块中导入 Encoder 类

LRELU_SLOPE = 0.1  # 定义 LRELU_SLOPE 常量为 0.1


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):  # 定义 LayerNorm 类的构造函数，channels 为输入通道数，eps 为小数值
        super().__init__()  # 调用父类的构造函数
        self.channels = channels  # 初始化实例变量 channels
        self.eps = eps  # 初始化实例变量 eps

        self.gamma = nn.Parameter(torch.ones(channels))  # 初始化可学习参数 gamma 为全 1 向量
        self.beta = nn.Parameter(torch.zeros(channels))  # 初始化可学习参数 beta 为全 0 向量

    def forward(self, x):  # 定义前向传播函数，输入参数 x 为输入数据
        x = x.transpose(1, -1)  # 调换输入数据的维度
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)  # 对输入数据进行 Layer Normalization
        return x.transpose(1, -1)  # 再次调换维度并返回结果


class ConvReluNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        kernel_size,
        n_layers,
        p_dropout,
    # 定义一个类，继承自 nn.Module
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
        super().__init__()
        # 初始化输入通道数、隐藏通道数、输出通道数、卷积核大小、层数、dropout 概率
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        # 断言层数大于1，否则抛出异常
        assert n_layers > 1, "Number of layers should be larger than 0."

        # 初始化卷积层和归一化层
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        # 添加第一层卷积层和归一化层
        self.conv_layers.append(
            nn.Conv1d(
                in_channels, hidden_channels, kernel_size, padding=kernel_size // 2
            )
        )
        self.norm_layers.append(LayerNorm(hidden_channels))
        # 定义 ReLU 和 Dropout 操作
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))
        # 循环添加剩余的卷积层和归一化层
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )
            self.norm_layers.append(LayerNorm(hidden_channels))
        # 定义投影层，将隐藏通道数映射到输出通道数
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        # 将投影层的权重和偏置初始化为零
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    # 定义前向传播方法
    def forward(self, x, x_mask):
        # 保存输入的原始数据
        x_org = x
        # 循环进行卷积、归一化、ReLU 和 Dropout 操作
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        # 将原始输入和经过投影层的输出相加，得到最终输出
        x = x_org + self.proj(x)
        # 返回最终输出并乘以掩码
        return x * x_mask
class DDSConv(nn.Module):
    """
    Dilated and Depth-Separable Convolution
    """

    def __init__(self, channels, kernel_size, n_layers, p_dropout=0.0):
        # 初始化函数，定义了DDSConv类的属性
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        # 初始化Dropout层
        self.drop = nn.Dropout(p_dropout)
        # 初始化深度可分离卷积和1x1卷积的ModuleList
        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        # 初始化LayerNorm层的ModuleList
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(n_layers):
            # 计算膨胀率和填充大小
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            # 添加深度可分离卷积层到ModuleList
            self.convs_sep.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    groups=channels,
                    dilation=dilation,
                    padding=padding,
                )
            )
            # 添加1x1卷积层到ModuleList
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            # 添加LayerNorm层到ModuleList
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(self, x, x_mask, g=None):
        # 前向传播函数
        if g is not None:
            x = x + g
        for i in range(self.n_layers):
            # 执行深度可分离卷积
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = F.gelu(y)
            # 执行1x1卷积
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            y = F.gelu(y)
            y = self.drop(y)
            x = x + y
        return x * x_mask


class WN(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
        p_dropout=0,
        # 调用父类的构造函数，初始化神经网络模型
        super(WN, self).__init__()
        # 断言卷积核大小为奇数
        assert kernel_size % 2 == 1
        # 初始化隐藏层通道数
        self.hidden_channels = hidden_channels
        # 初始化卷积核大小
        self.kernel_size = (kernel_size,)
        # 初始化膨胀率
        self.dilation_rate = dilation_rate
        # 初始化层数
        self.n_layers = n_layers
        # 初始化输入通道数
        self.gin_channels = gin_channels
        # 初始化丢弃概率
        self.p_dropout = p_dropout

        # 初始化输入层列表
        self.in_layers = torch.nn.ModuleList()
        # 初始化残差跳跃连接层列表
        self.res_skip_layers = torch.nn.ModuleList()
        # 初始化丢弃层
        self.drop = nn.Dropout(p_dropout)

        # 如果输入通道数不为0
        if gin_channels != 0:
            # 创建条件层
            cond_layer = torch.nn.Conv1d(
                gin_channels, 2 * hidden_channels * n_layers, 1
            )
            # 对条件层进行权重归一化
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name="weight")

        # 遍历层数
        for i in range(n_layers):
            # 计算膨胀率
            dilation = dilation_rate**i
            # 计算填充大小
            padding = int((kernel_size * dilation - dilation) / 2)
            # 创建输入层
            in_layer = torch.nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            # 对输入层进行权重归一化
            in_layer = torch.nn.utils.weight_norm(in_layer, name="weight")
            # 将输入层添加到输入层列表
            self.in_layers.append(in_layer)

            # 如果不是最后一层
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            # 创建残差跳跃连接层
            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            # 对残差跳跃连接层进行权重归一化
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name="weight")
            # 将残差跳跃连接层添加到残差跳跃连接层列表
            self.res_skip_layers.append(res_skip_layer)
    # 定义前向传播函数，接受输入 x、输入掩码 x_mask 和条件输入 g，返回输出结果
    def forward(self, x, x_mask, g=None, **kwargs):
        # 创建一个和输入 x 维度相同的全零张量作为输出
        output = torch.zeros_like(x)
        # 创建一个包含隐藏通道数的张量
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        # 如果条件输入 g 不为空，则通过条件层处理 g
        if g is not None:
            g = self.cond_layer(g)

        # 循环执行每一层的操作
        for i in range(self.n_layers):
            # 通过输入层 i 处理输入 x
            x_in = self.in_layers[i](x)
            # 如果条件输入 g 不为空，则根据当前层的索引和隐藏通道数获取对应的条件输入 g_l
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            # 通过融合操作处理输入 x_in 和条件输入 g_l，得到激活值 acts
            acts = commons.fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            # 对激活值 acts 进行丢弃操作
            acts = self.drop(acts)

            # 通过残差连接和跳跃连接层处理激活值 acts，得到残差跳跃激活值 res_skip_acts
            res_skip_acts = self.res_skip_layers[i](acts)
            # 如果当前层不是最后一层，则将残差激活值 res_acts 加到输入 x 上，并乘以输入掩码 x_mask
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            # 如果当前层是最后一层，则直接将残差跳跃激活值 res_skip_acts 加到输出上
            else:
                output = output + res_skip_acts
        # 返回最终输出结果乘以输入掩码 x_mask
        return output * x_mask

    # 定义移除权重归一化的函数
    def remove_weight_norm(self):
        # 如果输入通道数不为 0，则移除条件层的权重归一化
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        # 移除所有输入层的权重归一化
        for l in self.in_layers:
            torch.nn.utils.remove_weight_norm(l)
        # 移除所有残差跳跃连接层的权重归一化
        for l in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(l)
# 定义一个名为 ResBlock1 的神经网络模块
class ResBlock1(torch.nn.Module):
    # 前向传播函数，接受输入 x 和可选的输入 x_mask
    def forward(self, x, x_mask=None):
        # 遍历 self.convs1 和 self.convs2 中的卷积层对
        for c1, c2 in zip(self.convs1, self.convs2):
            # 对输入 x 应用 Leaky ReLU 激活函数
            xt = F.leaky_relu(x, LRELU_SLOPE)
            # 如果存在输入 x_mask，则将 xt 与 x_mask 相乘
            if x_mask is not None:
                xt = xt * x_mask
            # 将 xt 输入到 c1 卷积层中
            xt = c1(xt)
            # 对输出 xt 应用 Leaky ReLU 激活函数
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            # 如果存在输入 x_mask，则将 xt 与 x_mask 相乘
            if x_mask is not None:
                xt = xt * x_mask
            # 将 xt 输入到 c2 卷积层中
            xt = c2(xt)
            # 将 xt 与输入 x 相加，得到输出 x
            x = xt + x
        # 如果存在输入 x_mask，则将 x 与 x_mask 相乘
        if x_mask is not None:
            x = x * x_mask
        # 返回输出 x
        return x

    # 移除权重归一化
    def remove_weight_norm(self):
        # 遍历 self.convs1 中的卷积层，移除权重归一化
        for l in self.convs1:
            remove_weight_norm(l)
        # 遍历 self.convs2 中的卷积层，移除权重归一化
        for l in self.convs2:
            remove_weight_norm(l)


# 定义一个名为 ResBlock2 的神经网络模块，继承自 torch.nn.Module
class ResBlock2(torch.nn.Module):
    # 初始化函数，接受 channels、kernel_size 和 dilation 三个参数
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        # 调用父类的初始化函数
        super(ResBlock2, self).__init__()
        # 定义卷积层列表 self.convs，包含两个卷积层，使用权重归一化
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        # 对 self.convs 中的卷积层应用初始化权重函数
        self.convs.apply(init_weights)

    # 前向传播函数，接受输入 x 和可选的输入 x_mask
    def forward(self, x, x_mask=None):
        # 遍历 self.convs 中的卷积层
        for c in self.convs:
            # 对输入 x 应用 Leaky ReLU 激活函数
            xt = F.leaky_relu(x, LRELU_SLOPE)
            # 如果存在输入 x_mask，则将 xt 与 x_mask 相乘
            if x_mask is not None:
                xt = xt * x_mask
            # 将 xt 输入到卷积层 c 中
            xt = c(xt)
            # 将 xt 与输入 x 相加，得到输出 x
            x = xt + x
        # 如果存在输入 x_mask，则将 x 与 x_mask 相乘
        if x_mask is not None:
            x = x * x_mask
        # 返回输出 x
        return x
    # 定义一个方法，用于移除权重归一化
    def remove_weight_norm(self):
        # 遍历卷积层列表
        for l in self.convs:
            # 调用remove_weight_norm函数，移除权重归一化
            remove_weight_norm(l)
class Log(nn.Module):
    def forward(self, x, x_mask, reverse=False, **kwargs):
        # 如果不是反向操作
        if not reverse:
            # 对输入进行对数运算，并且将小于1e-5的值替换为1e-5，然后乘以掩码
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            # 计算对数行列式
            logdet = torch.sum(-y, [1, 2])
            # 返回对数运算后的结果和对数行列式
            return y, logdet
        else:
            # 如果是反向操作，对输入进行指数运算，并乘以掩码
            x = torch.exp(x) * x_mask
            # 返回指数运算后的结果
            return x


class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        # 对输入进行沿着第一个维度翻转
        x = torch.flip(x, [1])
        # 如果不是反向操作
        if not reverse:
            # 创建与输入相同大小的全零张量作为对数行列式
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            # 返回翻转后的结果和对数行列式
            return x, logdet
        else:
            # 如果是反向操作，直接返回翻转后的结果
            return x


class ElementwiseAffine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # 创建可学习参数 m 和 logs
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, x_mask, reverse=False, **kwargs):
        # 如果不是反向操作
        if not reverse:
            # 对输入进行仿射变换
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            # 计算对数行列式
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            # 返回仿射变换后的结果和对数行列式
            return y, logdet
        else:
            # 如果是反向操作，对输入进行逆仿射变换
            x = (x - self.m) * torch.exp(-self.logs) * x_mask
            # 返回逆仿射变换后的结果
            return x


class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
    # 断言通道数必须是偶数
    assert channels % 2 == 0, "channels should be divisible by 2"
    # 调用父类的初始化方法
    super().__init__()
    # 初始化实例变量
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.half_channels = channels // 2
    self.mean_only = mean_only

    # 创建一个 1 维卷积层，用于预处理输入数据
    self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
    # 创建 WaveNet 模型的编码器部分
    self.enc = WN(
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=p_dropout,
        gin_channels=gin_channels,
    )
    # 创建一个 1 维卷积层，用于后处理编码器输出
    self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
    # 将后处理卷积层的权重和偏置初始化为零
    self.post.weight.data.zero_()
    self.post.bias.data.zero_()

def forward(self, x, x_mask, g=None, reverse=False):
    # 将输入数据 x 按通道数的一半分割成两部分
    x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
    # 对第一部分数据进行预处理
    h = self.pre(x0) * x_mask
    # 将预处理后的数据输入编码器，得到编码结果
    h = self.enc(h, x_mask, g=g)
    # 对编码结果进行后处理，得到统计信息
    stats = self.post(h) * x_mask
    # 如果不是仅计算均值，则将统计信息分割成均值和标准差
    if not self.mean_only:
        m, logs = torch.split(stats, [self.half_channels] * 2, 1)
    else:
        m = stats
        logs = torch.zeros_like(m)

    # 如果不是反向操作
    if not reverse:
        # 根据均值和标准差对第二部分数据进行变换
        x1 = m + x1 * torch.exp(logs) * x_mask
        # 将处理后的数据拼接成完整的输出
        x = torch.cat([x0, x1], 1)
        # 计算对数行列式
        logdet = torch.sum(logs, [1, 2])
        return x, logdet
    else:
        # 如果是反向操作，则对第二部分数据进行逆变换
        x1 = (x1 - m) * torch.exp(-logs) * x_mask
        # 将处理后的数据拼接成完整的输出
        x = torch.cat([x0, x1], 1)
        return x
# 定义一个名为 ConvFlow 的类，继承自 nn.Module
class ConvFlow(nn.Module):
    # 初始化方法，接受输入通道数、滤波器通道数、卷积核大小、层数、分箱数和尾部边界等参数
    def __init__(
        self,
        in_channels,
        filter_channels,
        kernel_size,
        n_layers,
        num_bins=10,
        tail_bound=5.0,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化各个参数
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        # 创建一个 1x1 的卷积层，用于对输入进行预处理
        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        # 创建一个自定义的卷积层 DDSConv
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)
        # 创建一个 1x1 的卷积层，用于对卷积结果进行投影
        self.proj = nn.Conv1d(
            filter_channels, self.half_channels * (num_bins * 3 - 1), 1
        )
        # 将投影层的权重和偏置初始化为零
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    # 前向传播方法，接受输入 x、输入掩码 x_mask、条件 g 和是否反向操作的标志 reverse
    def forward(self, x, x_mask, g=None, reverse=False):
        # 将输入 x 按通道分割成两部分 x0 和 x1
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        # 对 x0 进行预处理
        h = self.pre(x0)
        # 通过自定义的卷积层对 h 进行卷积操作
        h = self.convs(h, x_mask, g=g)
        # 通过投影层对 h 进行投影，并乘以输入掩码
        h = self.proj(h) * x_mask

        # 获取 x0 的形状信息
        b, c, t = x0.shape
        # 重塑 h 的形状
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # [b, cx?, t] -> [b, c, t, ?]

        # 将 h 拆分成未归一化的宽度、高度和导数
        unnormalized_widths = h[..., : self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / math.sqrt(
            self.filter_channels
        )
        unnormalized_derivatives = h[..., 2 * self.num_bins :]

        # 通过分段有理二次变换对 x1 进行变换
        x1, logabsdet = piecewise_rational_quadratic_transform(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )

        # 将 x0 和变换后的 x1 连接起来，并乘以输入掩码
        x = torch.cat([x0, x1], 1) * x_mask
        # 计算对数行列式，并乘以输入掩码后在指定维度上求和
        logdet = torch.sum(logabsdet * x_mask, [1, 2])
        # 如果不是反向操作，则返回变换后的 x 和对数行列式
        if not reverse:
            return x, logdet
        # 如果是反向操作，则只返回变换后的 x
        else:
            return x


# 定义一个名为 TransformerCouplingLayer 的类，继承自 nn.Module
class TransformerCouplingLayer(nn.Module):
    # 初始化函数，设置模型的参数
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        n_layers,
        n_heads,
        p_dropout=0,
        filter_channels=0,
        mean_only=False,
        wn_sharing_parameter=None,
        gin_channels=0,
    ):
        # 断言 channels 应该是 2 的倍数
        assert channels % 2 == 0, "channels should be divisible by 2"
        # 调用父类的初始化函数
        super().__init__()
        # 设置模型的参数
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        # 创建一个 1 维卷积层，输入通道数为 half_channels，输出通道数为 hidden_channels，卷积核大小为 1
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        # 创建一个编码器对象，传入隐藏通道数、过滤通道数、头数、层数、卷积核大小、丢弃概率、是否是流模型、GIN 通道数
        self.enc = (
            Encoder(
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                isflow=True,
                gin_channels=gin_channels,
            )
            # 如果 wn_sharing_parameter 为 None，则创建新的编码器对象，否则使用传入的 wn_sharing_parameter
            if wn_sharing_parameter is None
            else wn_sharing_parameter
        )
        # 创建一个 1 维卷积层，输入通道数为 hidden_channels，输出通道数为 half_channels * (2 - mean_only)，卷积核大小为 1
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        # 将卷积层的权重和偏置初始化为 0
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()
    # 定义一个前向传播函数，接受输入 x、输入掩码 x_mask、条件 g 和是否反向传播的标志 reverse
    def forward(self, x, x_mask, g=None, reverse=False):
        # 将输入 x 按照通道数一分为二
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        # 对 x0 进行预处理，并乘以输入掩码
        h = self.pre(x0) * x_mask
        # 对 h 进行编码，传入输入掩码和条件 g
        h = self.enc(h, x_mask, g=g)
        # 对编码后的结果进行后处理，并乘以输入掩码
        stats = self.post(h) * x_mask
        # 如果不是仅计算均值
        if not self.mean_only:
            # 将 stats 按照通道数一分为二，分别表示均值 m 和对数标准差 logs
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            # 如果仅计算均值，则 m 为 stats，logs 为全零张量
            m = stats
            logs = torch.zeros_like(m)

        # 如果不是反向传播
        if not reverse:
            # 根据均值 m 和对数标准差 logs 对 x1 进行变换，并乘以输入掩码
            x1 = m + x1 * torch.exp(logs) * x_mask
            # 将处理后的 x0 和 x1 拼接在一起
            x = torch.cat([x0, x1], 1)
            # 计算对数行列式的和，作为返回值
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            # 如果是反向传播，则对 x1 进行反向变换，并乘以输入掩码
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            # 将处理后的 x0 和 x1 拼接在一起
            x = torch.cat([x0, x1], 1)
            return x

        # 对 x1 进行分段有理二次变换，得到变换后的结果和对数绝对值行列式
        x1, logabsdet = piecewise_rational_quadratic_transform(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )

        # 将处理后的 x0 和 x1 拼接在一起，并乘以输入掩码
        x = torch.cat([x0, x1], 1) * x_mask
        # 计算对数绝对值行列式的和，作为返回值
        logdet = torch.sum(logabsdet * x_mask, [1, 2])
        # 如果不是反向传播，则返回处理后的 x 和对数绝对值行列式的和
        if not reverse:
            return x, logdet
        else:
            # 如果是反向传播，则返回处理后的 x
            return x
```