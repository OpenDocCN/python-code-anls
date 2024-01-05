# `d:/src/tocomm/Bert-VITS2\onnx_modules\V220\models_onnx.py`

```
import math  # 导入 math 模块，用于数学运算
import torch  # 导入 torch 模块，用于构建神经网络
from torch import nn  # 从 torch 模块中导入 nn 模块，用于构建神经网络层
from torch.nn import functional as F  # 从 torch.nn 模块中导入 functional 模块，并重命名为 F，用于包含各种函数

import commons  # 导入自定义的 commons 模块
import modules  # 导入自定义的 modules 模块
from . import attentions_onnx  # 从当前目录下导入 attentions_onnx 模块
from vector_quantize_pytorch import VectorQuantize  # 从 vector_quantize_pytorch 模块中导入 VectorQuantize 类

from torch.nn import Conv1d, ConvTranspose1d, Conv2d  # 从 torch.nn 模块中导入 Conv1d、ConvTranspose1d、Conv2d 类
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm  # 从 torch.nn.utils 模块中导入 weight_norm、remove_weight_norm、spectral_norm 函数
from commons import init_weights, get_padding  # 从 commons 模块中导入 init_weights、get_padding 函数
from .text import symbols, num_tones, num_languages  # 从当前目录下的 text 模块中导入 symbols、num_tones、num_languages

class DurationDiscriminator(nn.Module):  # 定义 DurationDiscriminator 类，继承自 nn.Module 类，用于描述持续时间鉴别器
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):  # 定义初始化方法，接收输入通道数、滤波器通道数、卷积核大小、丢弃概率、全局信息通道数等参数
        super().__init__()  # 调用父类的构造函数进行初始化

        self.in_channels = in_channels  # 设置输入通道数
        self.filter_channels = filter_channels  # 设置滤波器通道数
        self.kernel_size = kernel_size  # 设置卷积核大小
        self.p_dropout = p_dropout  # 设置dropout概率
        self.gin_channels = gin_channels  # 设置输入通道数

        self.drop = nn.Dropout(p_dropout)  # 创建一个dropout层
        self.conv_1 = nn.Conv1d(  # 创建一个一维卷积层
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)  # 创建一个LayerNorm层
        self.conv_2 = nn.Conv1d(  # 创建一个一维卷积层
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)  # 创建一个LayerNorm层
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)  # 创建一个一维卷积层

        self.pre_out_conv_1 = nn.Conv1d(  # 创建一个一维卷积层
        2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
    )
    # 创建一个包含卷积层的神经网络模块，输入通道数为2 * filter_channels，输出通道数为filter_channels，卷积核大小为kernel_size，填充大小为kernel_size // 2
    self.pre_out_conv_1 = nn.Conv1d(
        filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
    )
    # 创建一个包含层归一化的神经网络模块，输入通道数为filter_channels
    self.pre_out_norm_1 = modules.LayerNorm(filter_channels)
    # 创建一个包含卷积层的神经网络模块，输入通道数为filter_channels，输出通道数为filter_channels，卷积核大小为kernel_size，填充大小为kernel_size // 2
    self.pre_out_conv_2 = nn.Conv1d(
        filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
    )
    # 创建一个包含层归一化的神经网络模块，输入通道数为filter_channels
    self.pre_out_norm_2 = modules.LayerNorm(filter_channels)

    # 如果gin_channels不为0，则创建一个包含卷积层的神经网络模块，输入通道数为gin_channels，输出通道数为in_channels，卷积核大小为1
    if gin_channels != 0:
        self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    # 创建一个包含线性层和Sigmoid激活函数的神经网络模块，输入维度为filter_channels，输出维度为1
    self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1), nn.Sigmoid())

def forward_probability(self, x, x_mask, dur, g=None):
    # 将dur通过dur_proj进行投影
    dur = self.dur_proj(dur)
    # 将x和dur在维度1上进行拼接
    x = torch.cat([x, dur], dim=1)
    # 将x乘以x_mask后通过pre_out_conv_1进行卷积操作
    x = self.pre_out_conv_1(x * x_mask)
    # 对x进行ReLU激活函数操作
    x = torch.relu(x)
    # 对x进行层归一化操作
    x = self.pre_out_norm_1(x)
    # 对x进行dropout操作
    x = self.drop(x)
        x = self.pre_out_conv_2(x * x_mask)  # 使用预定义的卷积层对输入进行卷积操作
        x = torch.relu(x)  # 对卷积结果进行激活函数处理
        x = self.pre_out_norm_2(x)  # 使用预定义的归一化层对数据进行归一化处理
        x = self.drop(x)  # 对数据进行dropout操作
        x = x * x_mask  # 将数据与掩码相乘
        x = x.transpose(1, 2)  # 对数据进行转置操作
        output_prob = self.output_layer(x)  # 使用预定义的输出层对数据进行处理，得到输出概率
        return output_prob  # 返回输出概率作为前向传播的结果

    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        x = torch.detach(x)  # 将输入张量进行分离，使其不再与计算图关联
        if g is not None:
            g = torch.detach(g)  # 将条件张量进行分离，使其不再与计算图关联
            x = x + self.cond(g)  # 将输入张量与条件张量经过条件模块处理后相加
        x = self.conv_1(x * x_mask)  # 使用预定义的卷积层对输入进行卷积操作
        x = torch.relu(x)  # 对卷积结果进行激活函数处理
        x = self.norm_1(x)  # 使用预定义的归一化层对数据进行归一化处理
        x = self.drop(x)  # 对数据进行dropout操作
        x = self.conv_2(x * x_mask)  # 使用预定义的卷积层对输入进行卷积操作
        x = torch.relu(x)  # 对卷积结果进行激活函数处理
        x = self.norm_2(x)  # 对输入数据进行第二层归一化处理
        x = self.drop(x)  # 对输入数据进行丢弃操作，以防止过拟合

        output_probs = []  # 初始化一个空列表，用于存储输出的概率值
        for dur in [dur_r, dur_hat]:  # 遍历持续时间列表
            output_prob = self.forward_probability(x, x_mask, dur, g)  # 调用 forward_probability 方法计算输出概率
            output_probs.append(output_prob)  # 将计算得到的输出概率添加到列表中

        return output_probs  # 返回输出概率列表


class TransformerCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,  # 初始化 TransformerCouplingBlock 类的参数
        p_dropout,  # 设置概率丢弃率
        n_flows=4,  # 设置默认值为4的流数
        gin_channels=0,  # 设置默认值为0的gin通道数
        share_parameter=False,  # 设置共享参数为False

        super().__init__()  # 调用父类的初始化方法

        self.channels = channels  # 设置通道数
        self.hidden_channels = hidden_channels  # 设置隐藏通道数
        self.kernel_size = kernel_size  # 设置卷积核大小
        self.n_layers = n_layers  # 设置层数
        self.n_flows = n_flows  # 设置流数
        self.gin_channels = gin_channels  # 设置gin通道数

        self.flows = nn.ModuleList()  # 创建一个空的模块列表

        self.wn = (  # 设置权重归一化
            attentions_onnx.FFT(  # 调用attentions_onnx模块中的FFT方法
                hidden_channels,  # 设置隐藏通道数
                filter_channels,  # 设置滤波器通道数
                n_heads,  # 设置头数
                n_layers,  # 定义变量n_layers，表示TransformerCouplingLayer的层数
                kernel_size,  # 定义变量kernel_size，表示TransformerCouplingLayer的卷积核大小
                p_dropout,  # 定义变量p_dropout，表示TransformerCouplingLayer的dropout概率
                isflow=True,  # 定义变量isflow，表示是否为流模型，默认为True
                gin_channels=self.gin_channels,  # 定义变量gin_channels，表示输入的通道数，默认为self.gin_channels
            )
            if share_parameter  # 如果share_parameter为真
            else None  # 否则为None
        )

        for i in range(n_flows):  # 循环n_flows次
            self.flows.append(  # 将新的TransformerCouplingLayer添加到self.flows列表中
                modules.TransformerCouplingLayer(  # 创建TransformerCouplingLayer模块
                    channels,  # 定义变量channels，表示TransformerCouplingLayer的输入通道数
                    hidden_channels,  # 定义变量hidden_channels，表示TransformerCouplingLayer的隐藏层通道数
                    kernel_size,  # 定义变量kernel_size，表示TransformerCouplingLayer的卷积核大小
                    n_layers,  # 定义变量n_layers，表示TransformerCouplingLayer的层数
                    n_heads,  # 定义变量n_heads，表示TransformerCouplingLayer的头数
                    p_dropout,  # 定义变量p_dropout，表示TransformerCouplingLayer的dropout概率
                    filter_channels,  # 定义变量filter_channels，表示TransformerCouplingLayer的滤波器通道数
                    mean_only=True,  # 设置参数 mean_only 为 True
                    wn_sharing_parameter=self.wn,  # 设置参数 wn_sharing_parameter 为 self.wn
                    gin_channels=self.gin_channels,  # 设置参数 gin_channels 为 self.gin_channels
                )
            )
            self.flows.append(modules.Flip())  # 将 modules.Flip() 添加到 self.flows 列表中

    def forward(self, x, x_mask, g=None, reverse=True):  # 定义 forward 方法，接受参数 x, x_mask, g, reverse
        if not reverse:  # 如果 reverse 不为 True
            for flow in self.flows:  # 遍历 self.flows 列表中的元素
                x, _ = flow(x, x_mask, g=g, reverse=reverse)  # 调用 flow 方法，传入参数 x, x_mask, g, reverse
        else:  # 如果 reverse 为 True
            for flow in reversed(self.flows):  # 遍历 self.flows 列表中的元素（倒序）
                x = flow(x, x_mask, g=g, reverse=reverse)  # 调用 flow 方法，传入参数 x, x_mask, g, reverse
        return x  # 返回 x

class StochasticDurationPredictor(nn.Module):  # 定义 StochasticDurationPredictor 类，继承自 nn.Module
    def __init__(self,  # 定义初始化方法，接受参数
        in_channels,  # 输入通道数
        filter_channels,  # 过滤器通道数
        kernel_size,  # 卷积核大小
        p_dropout,  # 丢弃概率
        n_flows=4,  # 流的数量，默认为4
        gin_channels=0,  # gin通道数，默认为0
    ):
        super().__init__()  # 调用父类的构造方法
        filter_channels = in_channels  # 这行代码需要从将来的版本中移除
        self.in_channels = in_channels  # 设置输入通道数
        self.filter_channels = filter_channels  # 设置过滤器通道数
        self.kernel_size = kernel_size  # 设置卷积核大小
        self.p_dropout = p_dropout  # 设置丢弃概率
        self.n_flows = n_flows  # 设置流的数量
        self.gin_channels = gin_channels  # 设置gin通道数

        self.log_flow = modules.Log()  # 创建Log模块
        self.flows = nn.ModuleList()  # 创建模块列表
        self.flows.append(modules.ElementwiseAffine(2))  # 向模块列表中添加ElementwiseAffine模块
        for i in range(n_flows):  # 循环n_flows次
        self.flows.append(
            modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
        )  # 向self.flows列表中添加一个卷积流模块，参数为输入通道数2，滤波器通道数filter_channels，卷积核大小kernel_size，层数n_layers为3

        self.flows.append(modules.Flip())  # 向self.flows列表中添加一个翻转模块

        self.post_pre = nn.Conv1d(1, filter_channels, 1)  # 创建一个1维卷积层，输入通道数为1，输出通道数为filter_channels，卷积核大小为1

        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)  # 创建一个1维卷积层，输入通道数为filter_channels，输出通道数为filter_channels，卷积核大小为1

        self.post_convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )  # 创建一个DDSConv模块，参数为滤波器通道数filter_channels，卷积核大小kernel_size，层数n_layers为3，丢弃概率p_dropout

        self.post_flows = nn.ModuleList()  # 创建一个空的ModuleList用于存储后处理流模块

        self.post_flows.append(modules.ElementwiseAffine(2))  # 向self.post_flows列表中添加一个元素级别的仿射模块，参数为输入通道数2

        for i in range(4):
            self.post_flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )  # 向self.post_flows列表中添加4个卷积流模块，参数为输入通道数2，滤波器通道数filter_channels，卷积核大小kernel_size，层数n_layers为3
            self.post_flows.append(modules.Flip())  # 向self.post_flows列表中添加4个翻转模块

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)  # 创建一个1维卷积层，输入通道数为in_channels，输出通道数为filter_channels，卷积核大小为1

        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)  # 创建一个1维卷积层，输入通道数为filter_channels，输出通道数为filter_channels，卷积核大小为1
        self.convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )  # 使用DDSConv模块创建卷积层，设置滤波器通道数、核大小、层数和丢失率

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)  # 如果输入通道数不为0，使用1维卷积层进行条件处理

    def forward(self, x, x_mask, z, g=None):
        x = torch.detach(x)  # 将输入张量x进行反向传播分离
        x = self.pre(x)  # 使用预处理层对输入x进行处理
        if g is not None:
            g = torch.detach(g)  # 如果条件g不为空，对条件g进行反向传播分离
            x = x + self.cond(g)  # 将条件g经过卷积层处理后的结果与输入x相加
        x = self.convs(x, x_mask)  # 使用DDSConv模块对输入x进行卷积操作
        x = self.proj(x) * x_mask  # 对卷积结果进行投影并乘以掩码

        flows = list(reversed(self.flows))  # 将self.flows列表进行反转
        flows = flows[:-2] + [flows[-1]]  # 移除一个无用的vflow
        for flow in flows:
            z = flow(z, x_mask, g=x, reverse=True)  # 对z进行流动操作，传入输入x、掩码和条件g
        z0, z1 = torch.split(z, [1, 1], 1)  # 将z按指定大小进行分割，得到z0和z1
        logw = z0  # 将变量z0的值赋给变量logw
        return logw  # 返回变量logw的值


class DurationPredictor(nn.Module):
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()  # 调用父类的构造函数

        self.in_channels = in_channels  # 初始化输入通道数
        self.filter_channels = filter_channels  # 初始化滤波器通道数
        self.kernel_size = kernel_size  # 初始化卷积核大小
        self.p_dropout = p_dropout  # 初始化dropout概率
        self.gin_channels = gin_channels  # 初始化全局输入通道数

        self.drop = nn.Dropout(p_dropout)  # 初始化dropout层
        self.conv_1 = nn.Conv1d(  # 初始化一维卷积层
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)  # 初始化第一个 LayerNorm 模块，用于对输入进行归一化处理
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )  # 初始化第二个一维卷积层，用于对输入进行卷积操作
        self.norm_2 = modules.LayerNorm(filter_channels)  # 初始化第二个 LayerNorm 模块，用于对卷积结果进行归一化处理
        self.proj = nn.Conv1d(filter_channels, 1, 1)  # 初始化一个一维卷积层，用于将卷积结果投影到一个标量值

        if gin_channels != 0:  # 如果条件输入的通道数不为0
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)  # 初始化一个一维卷积层，用于将条件输入进行卷积操作，以匹配输入的通道数

    def forward(self, x, x_mask, g=None):  # 定义前向传播函数，接受输入 x、输入的掩码 x_mask 和条件输入 g（可选）
        x = torch.detach(x)  # 将输入 x 转换为不需要梯度的张量
        if g is not None:  # 如果有条件输入 g
            g = torch.detach(g)  # 将条件输入 g 转换为不需要梯度的张量
            x = x + self.cond(g)  # 将输入 x 加上经过条件输入卷积后的结果
        x = self.conv_1(x * x_mask)  # 对输入进行一维卷积操作，并考虑输入的掩码
        x = torch.relu(x)  # 对卷积结果进行 ReLU 激活函数处理
        x = self.norm_1(x)  # 对激活后的结果进行归一化处理
        x = self.drop(x)  # 对归一化后的结果进行丢弃操作
        x = self.conv_2(x * x_mask)  # 对丢弃后的结果进行一维卷积操作，并考虑输入的掩码
        x = torch.relu(x)  # 使用 PyTorch 中的 relu 函数对输入 x 进行激活函数处理
        x = self.norm_2(x)  # 使用模型中的 norm_2 方法对输入 x 进行规范化处理
        x = self.drop(x)  # 使用模型中的 drop 方法对输入 x 进行 dropout 处理
        x = self.proj(x * x_mask)  # 使用模型中的 proj 方法对输入 x 乘以 x_mask 进行投影处理
        return x * x_mask  # 返回处理后的结果乘以 x_mask


class Bottleneck(nn.Sequential):
    def __init__(self, in_dim, hidden_dim):
        c_fc1 = nn.Linear(in_dim, hidden_dim, bias=False)  # 创建一个线性层，输入维度为 in_dim，输出维度为 hidden_dim，不使用偏置
        c_fc2 = nn.Linear(in_dim, hidden_dim, bias=False)  # 创建另一个线性层，输入维度为 in_dim，输出维度为 hidden_dim，不使用偏置
        super().__init__(*[c_fc1, c_fc2])  # 调用父类的初始化方法，将创建的线性层作为参数传入


class Block(nn.Module):
    def __init__(self, in_dim, hidden_dim) -> None:
        super().__init__()  # 调用父类的初始化方法
        self.norm = nn.LayerNorm(in_dim)  # 初始化一个 LayerNorm 层，输入维度为 in_dim
        self.mlp = MLP(in_dim, hidden_dim)  # 初始化一个 MLP 模型，输入维度为 in_dim，隐藏层维度为 hidden_dim
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 对输入张量进行归一化处理
        x = x + self.mlp(self.norm(x))
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        # 创建一个线性层，输入维度为in_dim，输出维度为hidden_dim，无偏置
        self.c_fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        # 创建一个线性层，输入维度为in_dim，输出维度为hidden_dim，无偏置
        self.c_fc2 = nn.Linear(in_dim, hidden_dim, bias=False)
        # 创建一个线性层，输入维度为hidden_dim，输出维度为in_dim，无偏置
        self.c_proj = nn.Linear(hidden_dim, in_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # 使用SILU激活函数对第一个线性层的输出进行处理，然后与第二个线性层的输出相乘
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        # 使用第三个线性层对结果进行处理
        x = self.c_proj(x)
        return x


class TextEncoder(nn.Module):
    def __init__(
        self,  # 定义一个类的初始化方法，self代表类的实例
        n_vocab,  # 词汇量的大小
        out_channels,  # 输出通道的数量
        hidden_channels,  # 隐藏层通道的数量
        filter_channels,  # 过滤器通道的数量
        n_heads,  # 多头注意力机制的头数
        n_layers,  # 网络层数
        kernel_size,  # 卷积核大小
        p_dropout,  # dropout概率
        n_speakers,  # 说话者数量
        gin_channels=0,  # GIN模型的通道数量，默认为0
    ):
        super().__init__()  # 调用父类的初始化方法
        self.n_vocab = n_vocab  # 初始化词汇量
        self.out_channels = out_channels  # 初始化输出通道数量
        self.hidden_channels = hidden_channels  # 初始化隐藏层通道数量
        self.filter_channels = filter_channels  # 初始化过滤器通道数量
        self.n_heads = n_heads  # 初始化多头注意力机制的头数
        self.n_layers = n_layers  # 初始化网络层数
        self.kernel_size = kernel_size  # 初始化卷积核大小
        self.p_dropout = p_dropout  # 设置对象的属性 p_dropout 为传入的参数值
        self.gin_channels = gin_channels  # 设置对象的属性 gin_channels 为传入的参数值
        self.emb = nn.Embedding(len(symbols), hidden_channels)  # 创建一个嵌入层对象，用于将符号映射到隐藏层的维度
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)  # 对嵌入层的权重进行正态分布初始化
        self.tone_emb = nn.Embedding(num_tones, hidden_channels)  # 创建一个嵌入层对象，用于将音调映射到隐藏层的维度
        nn.init.normal_(self.tone_emb.weight, 0.0, hidden_channels**-0.5)  # 对嵌入层的权重进行正态分布初始化
        self.language_emb = nn.Embedding(num_languages, hidden_channels)  # 创建一个嵌入层对象，用于将语言映射到隐藏层的维度
        nn.init.normal_(self.language_emb.weight, 0.0, hidden_channels**-0.5)  # 对嵌入层的权重进行正态分布初始化
        self.bert_proj = nn.Conv1d(1024, hidden_channels, 1)  # 创建一个 1 维卷积层对象，用于对输入进行卷积操作
        self.ja_bert_proj = nn.Conv1d(1024, hidden_channels, 1)  # 创建一个 1 维卷积层对象，用于对输入进行卷积操作
        self.en_bert_proj = nn.Conv1d(1024, hidden_channels, 1)  # 创建一个 1 维卷积层对象，用于对输入进行卷积操作
        # self.emo_proj = nn.Linear(1024, 1024)  # 创建一个线性层对象，用于进行线性变换
        # self.emo_quantizer = nn.ModuleList()  # 创建一个空的模块列表
        # for i in range(0, n_speakers):  # 遍历说话者数量
        #    self.emo_quantizer.append(  # 向模块列表中添加元素
        #        VectorQuantize(  # 创建一个向量量化层对象
        #            dim=1024,  # 设置向量量化的维度
        #            codebook_size=10,  # 设置码书的大小
        #            decay=0.8,  # 设置衰减率
        #            commitment_weight=1.0,  # 设置承诺损失的权重
        #            learnable_codebook=True,  # 设置学习可能的码书
        #            ema_update=False,  # 设置指数移动平均更新为假
        #        )
        #    )
        # self.emo_q_proj = nn.Linear(1024, hidden_channels)  # 创建一个线性层，输入维度为1024，输出维度为隐藏通道数
        self.n_speakers = n_speakers  # 初始化说话者数量
        self.in_feature_net = nn.Sequential(  # 创建一个包含多个模块的神经网络
            # input is assumed to an already normalized embedding
            nn.Linear(512, 1028, bias=False),  # 创建一个线性层，输入维度为512，输出维度为1028，不使用偏置
            nn.GELU(),  # 使用GELU激活函数
            nn.LayerNorm(1028),  # 对输入进行层归一化
            *[Block(1028, 512) for _ in range(1)],  # 创建一个包含多个Block模块的列表
            nn.Linear(1028, 512, bias=False),  # 创建一个线性层，输入维度为1028，输出维度为512，不使用偏置
            # normalize before passing to VQ?
            # nn.GELU(),
            # nn.LayerNorm(512),
        )
        self.emo_vq = VectorQuantize(  # 创建一个向量量化层
            dim=512,  # 设置输入维度为512
            codebook_size=64,  # 设置码书大小为64
        codebook_dim=32,  # 设置代码簿的维度为32
        commitment_weight=0.1,  # 设置捆绑权重为0.1
        decay=0.85,  # 设置衰减率为0.85
        heads=32,  # 设置注意力头的数量为32
        kmeans_iters=20,  # 设置K均值聚类的迭代次数为20
        separate_codebook_per_head=True,  # 设置每个注意力头是否有单独的代码簿为True
        stochastic_sample_codes=True,  # 设置是否随机采样代码为True
        threshold_ema_dead_code=2,  # 设置EMA死代码的阈值为2
    )
    self.out_feature_net = nn.Linear(512, hidden_channels)  # 创建一个线性层，输入维度为512，输出维度为隐藏通道数

    self.encoder = attentions_onnx.Encoder(  # 创建一个注意力编码器
        hidden_channels,  # 隐藏通道数
        filter_channels,  # 过滤器通道数
        n_heads,  # 注意力头的数量
        n_layers,  # 层数
        kernel_size,  # 卷积核大小
        p_dropout,  # 丢弃概率
        gin_channels=self.gin_channels,  # GIN通道数
    )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)  # 创建一个一维卷积层，输入通道数为hidden_channels，输出通道数为out_channels * 2，卷积核大小为1

    def forward(
        self, x, x_lengths, tone, language, bert, ja_bert, en_bert, emo, g=None
    ):
        x_mask = torch.ones_like(x).unsqueeze(0)  # 创建一个与x相同大小的全1张量，并在第0维度上增加一个维度
        bert_emb = self.bert_proj(bert.transpose(0, 1).unsqueeze(0)).transpose(1, 2)  # 对bert进行转置操作，然后通过bert_proj进行线性变换，并对结果进行再次转置
        ja_bert_emb = self.ja_bert_proj(ja_bert.transpose(0, 1).unsqueeze(0)).transpose(
            1, 2
        )  # 对ja_bert进行转置操作，然后通过ja_bert_proj进行线性变换，并对结果进行再次转置
        en_bert_emb = self.en_bert_proj(en_bert.transpose(0, 1).unsqueeze(0)).transpose(
            1, 2
        )  # 对en_bert进行转置操作，然后通过en_bert_proj进行线性变换，并对结果进行再次转置
        emo_emb = self.in_feature_net(emo.transpose(0, 1))  # 对emo进行转置操作，然后通过in_feature_net进行线性变换
        emo_emb, _, _ = self.emo_vq(emo_emb.unsqueeze(1))  # 对emo_emb进行维度扩展，然后通过emo_vq进行处理
        emo_emb = self.out_feature_net(emo_emb)  # 通过out_feature_net进行线性变换

        x = (
            self.emb(x)  # 对输入x进行嵌入操作
            + self.tone_emb(tone)  # 添加音调嵌入到输入张量中
            + self.language_emb(language)  # 添加语言嵌入到输入张量中
            + bert_emb  # 添加 BERT 嵌入到输入张量中
            + ja_bert_emb  # 添加日语 BERT 嵌入到输入张量中
            + en_bert_emb  # 添加英语 BERT 嵌入到输入张量中
            + emo_emb  # 添加情感嵌入到输入张量中
        ) * math.sqrt(
            self.hidden_channels  # 对输入张量进行缩放以平衡梯度大小
        )  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # 转置输入张量的维度
        x_mask = x_mask.to(x.dtype)  # 将掩码张量转换为与输入张量相同的数据类型

        x = self.encoder(x * x_mask, x_mask, g=g)  # 使用编码器对输入张量进行编码
        stats = self.proj(x) * x_mask  # 对编码后的张量进行投影并应用掩码

        m, logs = torch.split(stats, self.out_channels, dim=1)  # 将投影后的张量拆分为均值和对数方差
        return x, m, logs, x_mask  # 返回编码后的张量、均值、对数方差和掩码


class ResidualCouplingBlock(nn.Module):  # 定义一个残差耦合块
    def __init__(
        self,  # 定义一个初始化方法，self代表类的实例对象
        channels,  # 输入通道数
        hidden_channels,  # 隐藏层通道数
        kernel_size,  # 卷积核大小
        dilation_rate,  # 膨胀率
        n_layers,  # 神经网络层数
        n_flows=4,  # 流的数量，默认为4
        gin_channels=0,  # 输入通道数，默认为0
    ):
        super().__init__()  # 调用父类的初始化方法
        self.channels = channels  # 初始化输入通道数
        self.hidden_channels = hidden_channels  # 初始化隐藏层通道数
        self.kernel_size = kernel_size  # 初始化卷积核大小
        self.dilation_rate = dilation_rate  # 初始化膨胀率
        self.n_layers = n_layers  # 初始化神经网络层数
        self.n_flows = n_flows  # 初始化流的数量
        self.gin_channels = gin_channels  # 初始化输入通道数

        self.flows = nn.ModuleList()  # 初始化一个空的神经网络模块列表
        for i in range(n_flows):
            # 将ResidualCouplingLayer模块添加到流程中
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            # 将Flip模块添加到流程中
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=True):
        # 如果不是反向传播
        if not reverse:
            # 对于每个流程中的模块，依次进行前向传播
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            # 如果是反向传播，对于每个流程中的模块，依次进行反向传播
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)  # 调用flow函数，传入参数x, x_mask, g, reverse，并将结果赋值给变量x
        return x  # 返回变量x的值


class PosteriorEncoder(nn.Module):  # 定义一个名为PosteriorEncoder的类，继承自nn.Module
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()  # 调用父类的构造函数
        self.in_channels = in_channels  # 初始化类的属性in_channels
        self.out_channels = out_channels  # 初始化类的属性out_channels
        self.hidden_channels = hidden_channels  # 初始化类的属性hidden_channels
        self.kernel_size = kernel_size  # 初始化类的属性kernel_size
        self.dilation_rate = dilation_rate  # 设置类的属性 dilation_rate 为传入的 dilation_rate 值
        self.n_layers = n_layers  # 设置类的属性 n_layers 为传入的 n_layers 值
        self.gin_channels = gin_channels  # 设置类的属性 gin_channels 为传入的 gin_channels 值

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)  # 创建一个 1 维卷积层，输入通道数为 in_channels，输出通道数为 hidden_channels，卷积核大小为 1
        self.enc = modules.WN(  # 创建一个 WaveNet 模块
            hidden_channels,  # 输入通道数为 hidden_channels
            kernel_size,  # 卷积核大小为 kernel_size
            dilation_rate,  # 膨胀率为 dilation_rate
            n_layers,  # 层数为 n_layers
            gin_channels=gin_channels,  # 全局信息通道数为 gin_channels
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)  # 创建一个 1 维卷积层，输入通道数为 hidden_channels，输出通道数为 out_channels * 2，卷积核大小为 1

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )  # 根据输入序列长度 x_lengths 和输入张量 x 的维度创建一个掩码张量，并转换为与输入张量 x 相同的数据类型
        x = self.pre(x) * x_mask  # 对输入张量 x 进行预处理，并乘以掩码张量
        x = self.enc(x, x_mask, g=g)  # 使用 WaveNet 模块对输入张量 x 进行编码，传入掩码张量和全局信息 g
        stats = self.proj(x) * x_mask  # 使用self.proj对输入x进行处理，然后与x_mask相乘得到stats
        m, logs = torch.split(stats, self.out_channels, dim=1)  # 将stats按照self.out_channels进行分割，得到m和logs
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask  # 根据m和logs计算z，其中包括随机数和指数运算
        return z, m, logs, x_mask  # 返回计算得到的z、m、logs和输入的x_mask


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super(Generator, self).__init__()  # 调用父类的构造函数进行初始化
        self.num_kernels = len(resblock_kernel_sizes)  # 计算resblock_kernel_sizes的长度并赋值给self.num_kernels
        self.num_upsamples = len(upsample_rates)  # 计算上采样率列表的长度，并赋值给self.num_upsamples
        self.conv_pre = Conv1d(  # 创建一个一维卷积层，用于预处理
            initial_channel, upsample_initial_channel, 7, 1, padding=3  # 设置输入通道数、输出通道数、卷积核大小、步长和填充
        )
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2  # 根据resblock的值选择不同的残差块类型

        self.ups = nn.ModuleList()  # 创建一个空的ModuleList，用于存储上采样层
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):  # 遍历上采样率和卷积核大小的列表
            self.ups.append(  # 将下面创建的上采样层添加到self.ups中
                weight_norm(  # 对下面创建的卷积层进行权重归一化
                    ConvTranspose1d(  # 创建一个一维转置卷积层，用于上采样
                        upsample_initial_channel // (2**i),  # 设置输入通道数
                        upsample_initial_channel // (2 ** (i + 1)),  # 设置输出通道数
                        k,  # 设置卷积核大小
                        u,  # 设置上采样率
                        padding=(k - u) // 2,  # 设置填充
                    )
                )
            )
        # 创建一个空的 nn.ModuleList() 用于存储残差块
        self.resblocks = nn.ModuleList()
        # 遍历上采样层列表，计算初始通道数并创建残差块
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            # 遍历残差块的卷积核大小和膨胀大小，创建并添加残差块到 ModuleList 中
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        # 创建一个 1x1 卷积层，用于后处理
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        # 对上采样层进行初始化权重
        self.ups.apply(init_weights)

        # 如果有条件输入通道数不为 0，则创建一个 1x1 卷积层用于条件输入
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        # 对输入进行预处理卷积
        x = self.conv_pre(x)
        # 如果有条件输入，则将条件输入进行卷积并与输入相加
        if g is not None:
            x = x + self.cond(g)

        # 遍历上采样次数
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)  # 使用 Leaky ReLU 激活函数处理输入 x
            x = self.ups[i](x)  # 使用上采样层处理输入 x
            xs = None  # 初始化变量 xs 为 None
            for j in range(self.num_kernels):  # 遍历 num_kernels 次
                if xs is None:  # 如果 xs 为 None
                    xs = self.resblocks[i * self.num_kernels + j](x)  # 使用残差块处理输入 x，并赋值给 xs
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)  # 如果 xs 不为 None，则将残差块处理输入 x 的结果加到 xs 上
            x = xs / self.num_kernels  # 将 xs 除以 num_kernels 赋值给 x
        x = F.leaky_relu(x)  # 使用 Leaky ReLU 激活函数处理输入 x
        x = self.conv_post(x)  # 使用卷积后处理输入 x
        x = torch.tanh(x)  # 使用双曲正切函数处理输入 x

        return x  # 返回处理后的 x

    def remove_weight_norm(self):  # 定义移除权重归一化的方法
        print("Removing weight norm...")  # 打印信息
        for layer in self.ups:  # 遍历 self.ups 中的层
            remove_weight_norm(layer)  # 移除层的权重归一化
        for layer in self.resblocks:  # 遍历 self.resblocks 中的层
            layer.remove_weight_norm()
```
这行代码是一个注释，它没有实际的代码作用。

```
class DiscriminatorP(torch.nn.Module):
```
定义了一个名为DiscriminatorP的类，它是torch.nn.Module的子类。

```
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
```
定义了DiscriminatorP类的初始化方法，它接受period、kernel_size、stride和use_spectral_norm等参数。

```
        super(DiscriminatorP, self).__init__()
```
调用了父类的初始化方法。

```
        self.period = period
        self.use_spectral_norm = use_spectral_norm
```
将传入的period和use_spectral_norm参数赋值给类的属性。

```
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
```
根据use_spectral_norm的值选择使用weight_norm或spectral_norm，并将其赋值给norm_f。

```
        self.convs = nn.ModuleList(
```
创建了一个空的nn.ModuleList对象，并将其赋值给self.convs。

```
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
```
在self.convs中添加了一个元素，这个元素是一个使用norm_f包装的Conv2d对象。

                # 使用 Conv2d 函数创建一个卷积层，输入通道数为32，输出通道数为128，卷积核大小为(kernel_size, 1)，步长为(stride, 1)，填充为(get_padding(kernel_size, 1), 0)
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                # 使用 Conv2d 函数创建一个卷积层，输入通道数为128，输出通道数为512，卷积核大小为(kernel_size, 1)，步长为(stride, 1)，填充为(get_padding(kernel_size, 1), 0)
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                # 使用 Conv2d 函数创建一个卷积层，输入通道数为...
                        512,  # 设置输出通道数为 512
                        1024,  # 设置输入通道数为 1024
                        (kernel_size, 1),  # 设置卷积核大小为 kernel_size x 1
                        (stride, 1),  # 设置步长为 stride x 1
                        padding=(get_padding(kernel_size, 1), 0),  # 设置填充大小为通过函数 get_padding 计算得到的值和 0
                    )
                ),
                norm_f(  # 对卷积结果进行归一化处理
                    Conv2d(  # 创建一个二维卷积层
                        1024,  # 设置输入通道数为 1024
                        1024,  # 设置输出通道数为 1024
                        (kernel_size, 1),  # 设置卷积核大小为 kernel_size x 1
                        1,  # 设置步长为 1
                        padding=(get_padding(kernel_size, 1), 0),  # 设置填充大小为通过函数 get_padding 计算得到的值和 0
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))  # 创建一个二维卷积层，并对卷积结果进行归一化处理
    def forward(self, x):
        fmap = []  # 用于存储每一层卷积后的特征图

        # 1d to 2d
        b, c, t = x.shape  # 获取输入张量的维度信息
        if t % self.period != 0:  # 如果输入序列长度不是周期的整数倍，需要进行填充
            n_pad = self.period - (t % self.period)  # 计算需要填充的长度
            x = F.pad(x, (0, n_pad), "reflect")  # 使用反射填充方式对输入进行填充
            t = t + n_pad  # 更新填充后的序列长度
        x = x.view(b, c, t // self.period, self.period)  # 将输入序列转换为二维形式，以便进行卷积操作

        for layer in self.convs:  # 遍历每一层卷积
            x = layer(x)  # 对输入进行卷积操作
            x = F.leaky_relu(x, modules.LRELU_SLOPE)  # 使用LeakyReLU激活函数对卷积结果进行激活
            fmap.append(x)  # 将卷积后的特征图添加到fmap列表中
        x = self.conv_post(x)  # 对最后一层卷积进行处理
        fmap.append(x)  # 将处理后的结果添加到fmap列表中
        x = torch.flatten(x, 1, -1)  # 将最终的特征图展平为一维张量

        return x, fmap  # 返回最终的结果张量和每一层卷积后的特征图列表
class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        # 定义权重归一化函数，根据 use_spectral_norm 参数选择 weight_norm 或 spectral_norm
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        # 定义卷积层列表，每个卷积层使用 norm_f 函数进行权重归一化
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),  # 输入通道数为1，输出通道数为16，卷积核大小为15
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),  # 输入通道数为16，输出通道数为64，卷积核大小为41
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),  # 输入通道数为64，输出通道数为256，卷积核大小为41
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),  # 输入通道数为256，输出通道数为1024，卷积核大小为41
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),  # 输入通道数为1024，输出通道数为1024，卷积核大小为41
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),  # 输入通道数为1024，输出通道数为1024，卷积核大小为5
            ]
        )
        # 定义后续的卷积层，使用 norm_f 函数进行权重归一化
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))  # 输入通道数为1024，输出通道数为1，卷积核大小为3

    def forward(self, x):
        fmap = []  # 定义一个空列表用于存储特征图
        for layer in self.convs:  # 遍历self.convs中的每一层
            x = layer(x)  # 将输入x传入当前层进行计算
            x = F.leaky_relu(x, modules.LRELU_SLOPE)  # 使用Leaky ReLU激活函数对x进行处理
            fmap.append(x)  # 将处理后的x添加到fmap列表中
        x = self.conv_post(x)  # 将x传入self.conv_post进行计算
        fmap.append(x)  # 将计算后的x添加到fmap列表中
        x = torch.flatten(x, 1, -1)  # 将x展平为一维张量

        return x, fmap  # 返回x和fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):  # 初始化函数，接受一个布尔类型的参数use_spectral_norm
        super(MultiPeriodDiscriminator, self).__init__()  # 调用父类的初始化函数
        periods = [2, 3, 5, 7, 11]  # 定义一个包含多个整数的列表periods

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]  # 创建一个包含一个DiscriminatorS对象的列表discs
        discs = discs + [  # 将下面的列表添加到discs列表中
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods  # 创建一个包含多个DiscriminatorP对象的列表，根据periods中的整数进行初始化
        ]
        # 初始化一个 nn.ModuleList 类型的对象，用于存储多个鉴别器
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        # 初始化空列表，用于存储真实图像的鉴别器输出
        y_d_rs = []
        # 初始化空列表，用于存储生成图像的鉴别器输出
        y_d_gs = []
        # 初始化空列表，用于存储真实图像的特征图
        fmap_rs = []
        # 初始化空列表，用于存储生成图像的特征图
        fmap_gs = []
        # 遍历所有鉴别器
        for i, d in enumerate(self.discriminators):
            # 对真实图像进行鉴别器前向传播，获取鉴别器输出和特征图
            y_d_r, fmap_r = d(y)
            # 对生成图像进行鉴别器前向传播，获取鉴别器输出和特征图
            y_d_g, fmap_g = d(y_hat)
            # 将真实图像的鉴别器输出存入列表
            y_d_rs.append(y_d_r)
            # 将生成图像的鉴别器输出存入列表
            y_d_gs.append(y_d_g)
            # 将真实图像的特征图存入列表
            fmap_rs.append(fmap_r)
            # 将生成图像的特征图存入列表
            fmap_gs.append(fmap_g)

        # 返回真实图像的鉴别器输出列表、生成图像的鉴别器输出列表、真实图像的特征图列表、生成图像的特征图列表
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    # 初始化函数，设置类的初始属性
    def __init__(self, spec_channels, gin_channels=0):
        # 调用父类的初始化函数
        super().__init__()
        # 设置类的属性
        self.spec_channels = spec_channels
        # 定义卷积层的滤波器
        ref_enc_filters = [32, 32, 64, 64, 128, 128]
        # 获取滤波器的数量
        K = len(ref_enc_filters)
        # 定义滤波器
        filters = [1] + ref_enc_filters
        # 定义卷积层
        convs = [
            # 使用权重归一化的卷积层
            weight_norm(
                # 2维卷积层
                nn.Conv2d(
                    # 输入通道数
                    in_channels=filters[i],
                    # 输出通道数
                    out_channels=filters[i + 1],
                    # 卷积核大小
                    kernel_size=(3, 3),
                    # 步长
                    stride=(2, 2),
                    # 填充
                    padding=(1, 1),
                )
        )
        for i in range(K)  # 循环K次，K的值未知
    ]
    self.convs = nn.ModuleList(convs)  # 创建一个包含卷积层的模块列表

    out_channels = self.calculate_channels(spec_channels, 3, 2, 1, K)  # 计算输出通道数
    self.gru = nn.GRU(
        input_size=ref_enc_filters[-1] * out_channels,  # GRU的输入大小
        hidden_size=256 // 2,  # GRU的隐藏层大小
        batch_first=True,  # 输入数据的维度顺序
    )
    self.proj = nn.Linear(128, gin_channels)  # 创建一个线性层

def forward(self, inputs, mask=None):
    N = inputs.size(0)  # 获取输入数据的批量大小
    out = inputs.view(N, 1, -1, self.spec_channels)  # 将输入数据重塑为指定形状
    for conv in self.convs:  # 遍历卷积层列表
        out = conv(out)  # 对输入数据进行卷积操作
        # out = wn(out)  # 对输出数据进行权重归一化操作
        out = F.relu(out)  # 使用ReLU激活函数处理out张量，将小于0的值置为0

        out = out.transpose(1, 2)  # 将out张量的维度1和维度2进行转置

        T = out.size(1)  # 获取out张量的维度1的大小
        N = out.size(0)  # 获取out张量的维度0的大小

        out = out.contiguous().view(N, T, -1)  # 将out张量转换为连续的张量，并改变其形状为(N, T, -1)

        self.gru.flatten_parameters()  # 将GRU层的参数展平，以便进行后续操作

        memory, out = self.gru(out)  # 使用GRU层处理out张量，得到memory和out

        return self.proj(out.squeeze(0))  # 对out张量进行压缩并通过self.proj进行处理，返回结果

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):  # 对于n_convs的次数进行循环
            L = (L - kernel_size + 2 * pad) // stride + 1  # 根据给定的公式计算L的值
        return L  # 返回计算后的L的值
    # 初始化函数，用于创建Synthesizer对象
    def __init__(
        self,
        n_vocab,  # 词汇表的大小
        spec_channels,  # 频谱通道数
        segment_size,  # 分段大小
        inter_channels,  # 内部通道数
        hidden_channels,  # 隐藏层通道数
        filter_channels,  # 过滤器通道数
        n_heads,  # 多头注意力机制的头数
        n_layers,  # 层数
        kernel_size,  # 卷积核大小
        p_dropout,  # 丢弃概率
        resblock,  # 是否使用残差块
        resblock_kernel_sizes,  # 残差块的卷积核大小
        resblock_dilation_sizes,  # 残差块的扩张大小
        upsample_rates,  # 上采样率
        upsample_initial_channel,  # 上采样初始通道数
        upsample_kernel_sizes,  # 定义变量 upsample_kernel_sizes
        n_speakers=256,  # 定义变量 n_speakers 并设置默认值为 256
        gin_channels=256,  # 定义变量 gin_channels 并设置默认值为 256
        use_sdp=True,  # 定义变量 use_sdp 并设置默认值为 True
        n_flow_layer=4,  # 定义变量 n_flow_layer 并设置默认值为 4
        n_layers_trans_flow=4,  # 定义变量 n_layers_trans_flow 并设置默认值为 4
        flow_share_parameter=False,  # 定义变量 flow_share_parameter 并设置默认值为 False
        use_transformer_flow=True,  # 定义变量 use_transformer_flow 并设置默认值为 True
        **kwargs,  # 接收其他未命名的参数并存储在字典 kwargs 中
    ):
        super().__init__()  # 调用父类的构造函数
        self.n_vocab = n_vocab  # 将参数 n_vocab 赋值给实例变量 self.n_vocab
        self.spec_channels = spec_channels  # 将参数 spec_channels 赋值给实例变量 self.spec_channels
        self.inter_channels = inter_channels  # 将参数 inter_channels 赋值给实例变量 self.inter_channels
        self.hidden_channels = hidden_channels  # 将参数 hidden_channels 赋值给实例变量 self.hidden_channels
        self.filter_channels = filter_channels  # 将参数 filter_channels 赋值给实例变量 self.filter_channels
        self.n_heads = n_heads  # 将参数 n_heads 赋值给实例变量 self.n_heads
        self.n_layers = n_layers  # 将参数 n_layers 赋值给实例变量 self.n_layers
        self.kernel_size = kernel_size  # 将参数 kernel_size 赋值给实例变量 self.kernel_size
        self.p_dropout = p_dropout  # 将参数 p_dropout 赋值给实例变量 self.p_dropout
        # 设置resblock参数
        self.resblock = resblock
        # 设置resblock_kernel_sizes参数
        self.resblock_kernel_sizes = resblock_kernel_sizes
        # 设置resblock_dilation_sizes参数
        self.resblock_dilation_sizes = resblock_dilation_sizes
        # 设置upsample_rates参数
        self.upsample_rates = upsample_rates
        # 设置upsample_initial_channel参数
        self.upsample_initial_channel = upsample_initial_channel
        # 设置upsample_kernel_sizes参数
        self.upsample_kernel_sizes = upsample_kernel_sizes
        # 设置segment_size参数
        self.segment_size = segment_size
        # 设置n_speakers参数
        self.n_speakers = n_speakers
        # 设置gin_channels参数
        self.gin_channels = gin_channels
        # 设置n_layers_trans_flow参数
        self.n_layers_trans_flow = n_layers_trans_flow
        # 设置use_spk_conditioned_encoder参数
        self.use_spk_conditioned_encoder = kwargs.get("use_spk_conditioned_encoder", True)
        # 设置use_sdp参数
        self.use_sdp = use_sdp
        # 设置use_noise_scaled_mas参数
        self.use_noise_scaled_mas = kwargs.get("use_noise_scaled_mas", False)
        # 设置mas_noise_scale_initial参数
        self.mas_noise_scale_initial = kwargs.get("mas_noise_scale_initial", 0.01)
        # 设置noise_scale_delta参数
        self.noise_scale_delta = kwargs.get("noise_scale_delta", 2e-6)
        # 设置current_mas_noise_scale参数
        self.current_mas_noise_scale = self.mas_noise_scale_initial
        # 如果使用spk_conditioned_encoder并且gin_channels大于0，则设置enc_gin_channels参数
        if self.use_spk_conditioned_encoder and gin_channels > 0:
            self.enc_gin_channels = gin_channels
        # 使用TextEncoder类初始化self.enc_p，传入参数n_vocab, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, self.n_speakers, self.enc_gin_channels
        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            self.n_speakers,
            gin_channels=self.enc_gin_channels,
        )
        # 使用Generator类初始化self.dec，传入参数inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
        gin_channels=gin_channels,  # 为PosteriorEncoder和TransformerCouplingBlock指定gin_channels参数
    )
    self.enc_q = PosteriorEncoder(  # 创建PosteriorEncoder对象
        spec_channels,  # 指定spec_channels参数
        inter_channels,  # 指定inter_channels参数
        hidden_channels,  # 指定hidden_channels参数
        5,  # 指定第五个参数
        1,  # 指定第六个参数
        16,  # 指定第七个参数
        gin_channels=gin_channels,  # 为gin_channels参数指定值
    )
    if use_transformer_flow:  # 如果use_transformer_flow为True
        self.flow = TransformerCouplingBlock(  # 创建TransformerCouplingBlock对象
            inter_channels,  # 指定inter_channels参数
            hidden_channels,  # 指定hidden_channels参数
            filter_channels,  # 指定filter_channels参数
            n_heads,  # 指定n_heads参数
            n_layers_trans_flow,  # 指定n_layers_trans_flow参数
            5,  # 指定第七个参数
            p_dropout,  # 指定p_dropout参数
        n_flow_layer,  # 定义变量n_flow_layer，表示流层的数量
        gin_channels=gin_channels,  # 定义变量gin_channels，表示GIN模型的通道数
        share_parameter=flow_share_parameter,  # 定义变量share_parameter，表示流共享参数
    )
else:  # 如果条件不满足
    self.flow = ResidualCouplingBlock(  # 定义self.flow，使用ResidualCouplingBlock模块
        inter_channels,  # 定义变量inter_channels，表示交互通道数
        hidden_channels,  # 定义变量hidden_channels，表示隐藏通道数
        5,  # 定义值为5的变量，表示某种参数
        1,  # 定义值为1的变量，表示某种参数
        n_flow_layer,  # 使用变量n_flow_layer
        gin_channels=gin_channels,  # 使用变量gin_channels
    )
self.sdp = StochasticDurationPredictor(  # 定义self.sdp，使用StochasticDurationPredictor模块
    hidden_channels,  # 使用变量hidden_channels
    192,  # 定义值为192的变量，表示某种参数
    3,  # 定义值为3的变量，表示某种参数
    0.5,  # 定义值为0.5的变量，表示某种参数
    4,  # 定义值为4的变量，表示某种参数
    gin_channels=gin_channels  # 使用变量gin_channels
)
self.dp = DurationPredictor(  # 定义self.dp，使用DurationPredictor模块
    hidden_channels,  # 使用变量hidden_channels
    256,  # 定义值为256的变量，表示某种参数
    3,  # 定义值为3的变量，表示某种参数
    0.5,  # 定义值为0.5的变量，表示某种参数
    gin_channels=gin_channels  # 使用变量gin_channels
)
        # 如果说话者数量大于等于1，则创建一个嵌入层，嵌入层的大小为n_speakers x gin_channels
        if n_speakers >= 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
        # 如果说话者数量小于1，则创建一个参考编码器，输入通道数为spec_channels，输出通道数为gin_channels
        else:
            self.ref_enc = ReferenceEncoder(spec_channels, gin_channels)

    # 导出模型为ONNX格式
    def export_onnx(
        self,
        path,
        max_len=None,
        sdp_ratio=0,
        y=None,
    ):
        # 设置噪声比例为0.667
        noise_scale = 0.667
        # 设置长度比例为1
        length_scale = 1
        # 设置噪声比例权重为0.8
        noise_scale_w = 0.8
        # 创建一个长为4的张量，包含元素[0, 97, ...]
        x = (
            torch.LongTensor(
                [
                    0,
                    97,
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 以二进制模式打开文件，并将其内容封装成字节流
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建一个 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名列表，读取每个文件的数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
        # 创建一个大小为 x.shape[1] x 1024 的随机张量，存储在 CPU 上
        bert = torch.randn(size=(x.shape[1], 1024)).cpu()
        # 创建一个大小为 x.shape[1] x 1024 的随机张量，存储在 CPU 上
        ja_bert = torch.randn(size=(x.shape[1], 1024)).cpu()
        # 创建一个大小为 x.shape[1] x 1024 的随机张量，存储在 CPU 上
        en_bert = torch.randn(size=(x.shape[1], 1024)).cpu()
        if self.n_speakers > 0:  # 检查是否有说话者数量大于0
            g = self.emb_g(sid).unsqueeze(-1)  # 从emb_g模型中获取sid对应的张量，并在最后一个维度上增加一个维度
            torch.onnx.export(  # 使用torch.onnx.export函数将模型导出为ONNX格式
                self.emb_g,  # 要导出的模型
                (sid),  # 模型的输入
                f"onnx/{path}/{path}_emb.onnx",  # 导出的ONNX文件路径
                input_names=["sid"],  # 输入张量的名称
                output_names=["g"],  # 输出张量的名称
                verbose=True,  # 是否打印详细信息
            )
        else:  # 如果没有说话者数量大于0
            g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)  # 从ref_enc模型中获取y的转置张量，并在最后一个维度上增加一个维度

        emo = torch.randn(512, 1)  # 生成一个512x1的随机张量

        torch.onnx.export(  # 使用torch.onnx.export函数将模型导出为ONNX格式
            self.enc_p,  # 要导出的模型
            (x, x_lengths, tone, language, bert, ja_bert, en_bert, emo, g),  # 模型的输入
            f"onnx/{path}/{path}_enc_p.onnx",  # 导出的ONNX文件路径
            input_names=[  # 输入张量的名称
                "x",  # 输入数据的名称
                "x_lengths",  # 输入数据的长度
                "t",  # 时间步
                "language",  # 语言信息
                "bert_0",  # BERT模型的第一个输出
                "bert_1",  # BERT模型的第二个输出
                "bert_2",  # BERT模型的第三个输出
                "emo",  # 情感信息
                "g",  # 其他信息
            ],
            output_names=["xout", "m_p", "logs_p", "x_mask"],  # 输出数据的名称
            dynamic_axes={
                "x": [0, 1],  # 输入数据的动态轴
                "t": [0, 1],  # 时间步的动态轴
                "language": [0, 1],  # 语言信息的动态轴
                "bert_0": [0],  # BERT模型第一个输出的动态轴
                "bert_1": [0],  # BERT模型第二个输出的动态轴
                "bert_2": [0],  # BERT模型第三个输出的动态轴
                "xout": [0, 2],  # 输出数据的动态轴
                "m_p": [0, 2],  # 输出数据的动态轴
                "logs_p": [0, 2],  # 设置参数 logs_p 的值为列表 [0, 2]
                "x_mask": [0, 2],  # 设置参数 x_mask 的值为列表 [0, 2]
            },
            verbose=True,  # 设置参数 verbose 的值为 True
            opset_version=16,  # 设置参数 opset_version 的值为 16
        )

        x, m_p, logs_p, x_mask = self.enc_p(  # 调用 self.enc_p 方法，将返回的结果分别赋值给 x, m_p, logs_p, x_mask
            x, x_lengths, tone, language, bert, ja_bert, en_bert, emo, g  # 方法的参数列表
        )

        zinput = (  # 创建 zinput 变量
            torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype)  # 生成指定大小的随机张量，并将其移动到指定设备上
            * noise_scale_w  # 乘以 noise_scale_w
        )
        torch.onnx.export(  # 调用 torch.onnx.export 方法
            self.sdp,  # 导出的模型
            (x, x_mask, zinput, g),  # 方法的输入参数列表
            f"onnx/{path}/{path}_sdp.onnx",  # 导出的文件路径
            input_names=["x", "x_mask", "zin", "g"],  # 输入参数的名称列表
        torch.onnx.export(
            self.dp,  # 导出self.dp模型
            (x, x_mask, g),  # 输入参数
            f"onnx/{path}/{path}_dp.onnx",  # 导出的ONNX文件路径
            input_names=["x", "x_mask", "g"],  # 输入参数的名称
            output_names=["logw"],  # 输出参数的名称
            dynamic_axes={"x": [0, 2], "x_mask": [0, 2], "logw": [0, 2]},  # 动态轴
            verbose=True,  # 是否显示详细信息
        )
        torch.onnx.export(
            self.dp,  # 导出self.dp模型
            (x, x_mask, g),  # 输入参数
            f"onnx/{path}/{path}_dp.onnx",  # 导出的ONNX文件路径
            input_names=["x", "x_mask", "g"],  # 输入参数的名称
            output_names=["logw"],  # 输出参数的名称
            dynamic_axes={"x": [0, 2], "x_mask": [0, 2], "logw": [0, 2]},  # 动态轴
            verbose=True,  # 是否显示详细信息
        )
        logw = self.sdp(x, x_mask, zinput, g=g) * (sdp_ratio) + self.dp(  # 计算logw
            x, x_mask, g=g  # 输入参数
        ) * (1 - sdp_ratio)  # 计算logw
        w = torch.exp(logw) * x_mask * length_scale  # 计算w
        w_ceil = torch.ceil(w)  # 对w进行向上取整
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()  # 计算y_lengths
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(  # 计算y_mask
        x_mask.dtype  # 获取 x_mask 的数据类型

        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)  # 创建注意力掩码，用于计算注意力权重
        attn = commons.generate_path(w_ceil, attn_mask)  # 生成路径

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # 计算注意力加权的 m_p，将结果进行转置操作
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # 计算注意力加权的 logs_p，将结果进行转置操作

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale  # 计算 z_p
        torch.onnx.export(
            self.flow,
            (z_p, y_mask, g),
            f"onnx/{path}/{path}_flow.onnx",
            input_names=["z_p", "y_mask", "g"],
            output_names=["z"],
            dynamic_axes={"z_p": [0, 2], "y_mask": [0, 2], "z": [0, 2]}  # 导出 ONNX 模型
# 设置 verbose 参数为 True，以便在执行过程中输出详细信息
verbose=True,

# 使用 self.flow 方法对 z_p 和 y_mask 进行处理，得到 z
z = self.flow(z_p, y_mask, g=g, reverse=True)

# 从 z 中提取出符合条件的部分，赋值给 z_in
z_in = (z * y_mask)[:, :, :max_len]

# 使用 torch.onnx.export 方法将 self.dec 模型导出为 ONNX 格式的文件
torch.onnx.export(
    self.dec,  # 要导出的模型
    (z_in, g),  # 模型的输入
    f"onnx/{path}/{path}_dec.onnx",  # 导出的文件路径
    input_names=["z_in", "g"],  # 输入的名称
    output_names=["o"],  # 输出的名称
    dynamic_axes={"z_in": [0, 2], "o": [0, 2]},  # 动态轴的设置
    verbose=True,  # 设置 verbose 参数为 True，以便在执行过程中输出详细信息
)

# 使用 self.dec 方法对 (z * y_mask)[:, :, :max_len] 和 g 进行处理，得到 o
o = self.dec((z * y_mask)[:, :, :max_len], g=g)
```