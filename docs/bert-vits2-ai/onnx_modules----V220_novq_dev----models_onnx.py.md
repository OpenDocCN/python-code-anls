# `d:/src/tocomm/Bert-VITS2\onnx_modules\V220_novq_dev\models_onnx.py`

```
import math  # 导入数学库，用于数学运算
import torch  # 导入PyTorch库，用于构建神经网络模型
from torch import nn  # 从PyTorch库中导入神经网络模块
from torch.nn import functional as F  # 从PyTorch库中导入函数模块，并重命名为F

import commons  # 导入自定义的commons模块
import modules  # 导入自定义的modules模块
from . import attentions_onnx  # 从当前目录下导入attentions_onnx模块

from torch.nn import Conv1d, ConvTranspose1d, Conv2d  # 从PyTorch库中导入一维卷积、一维转置卷积和二维卷积模块
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm  # 从PyTorch库中导入权重归一化、移除权重归一化和谱归一化模块
from commons import init_weights, get_padding  # 从自定义的commons模块中导入初始化权重和获取填充模块
from .text import symbols, num_tones, num_languages  # 从当前目录下的text模块中导入symbols、num_tones和num_languages变量

class DurationDiscriminator(nn.Module):  # 定义一个名为DurationDiscriminator的神经网络模型类，继承自nn.Module
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):  # 初始化函数，接受输入通道数、滤波器通道数、卷积核大小、dropout概率和全局信息通道数作为参数
        super().__init__()  # 调用父类的初始化函数
        self.in_channels = in_channels  # 设置输入通道数
        self.filter_channels = filter_channels  # 设置滤波器通道数
        self.kernel_size = kernel_size  # 设置卷积核大小
        self.p_dropout = p_dropout  # 设置dropout概率
        self.gin_channels = gin_channels  # 设置输入通道数

        self.drop = nn.Dropout(p_dropout)  # 初始化dropout层
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )  # 初始化第一个卷积层
        self.norm_1 = modules.LayerNorm(filter_channels)  # 初始化第一个归一化层
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )  # 初始化第二个卷积层
        self.norm_2 = modules.LayerNorm(filter_channels)  # 初始化第二个归一化层
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)  # 初始化持续时间投影层

        self.pre_out_conv_1 = nn.Conv1d(
            2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )  # 初始化预输出卷积层1
        )
        self.pre_out_norm_1 = modules.LayerNorm(filter_channels)  # 创建一个LayerNorm层，用于对filter_channels进行归一化处理
        self.pre_out_conv_2 = nn.Conv1d(  # 创建一个一维卷积层
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_2 = modules.LayerNorm(filter_channels)  # 创建另一个LayerNorm层，用于对filter_channels进行归一化处理

        if gin_channels != 0:  # 如果gin_channels不为0
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)  # 创建一个一维卷积层

        self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1), nn.Sigmoid())  # 创建一个包含线性层和Sigmoid激活函数的序列模块

    def forward_probability(self, x, x_mask, dur, g=None):  # 定义前向传播函数，接受输入x、x_mask、dur和g
        dur = self.dur_proj(dur)  # 对dur进行投影处理
        x = torch.cat([x, dur], dim=1)  # 在维度1上对x和dur进行拼接
        x = self.pre_out_conv_1(x * x_mask)  # 对x乘以x_mask后，通过pre_out_conv_1进行一维卷积
        x = torch.relu(x)  # 对x进行ReLU激活函数处理
        x = self.pre_out_norm_1(x)  # 对x进行归一化处理
        x = self.drop(x)  # 对x进行dropout处理
        x = self.pre_out_conv_2(x * x_mask)  # 对x乘以x_mask后，通过pre_out_conv_2进行一维卷积
        x = torch.relu(x)  # 使用 PyTorch 的 relu 函数对输入 x 进行激活函数处理
        x = self.pre_out_norm_2(x)  # 使用预定义的神经网络层对输入 x 进行处理
        x = self.drop(x)  # 使用预定义的神经网络层对输入 x 进行 dropout 处理
        x = x * x_mask  # 将输入 x 与掩码 x_mask 相乘
        x = x.transpose(1, 2)  # 对输入 x 进行维度转置操作
        output_prob = self.output_layer(x)  # 使用预定义的神经网络层对输入 x 进行处理，得到输出概率
        return output_prob  # 返回输出概率

    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        x = torch.detach(x)  # 使用 PyTorch 的 detach 函数对输入 x 进行处理
        if g is not None:
            g = torch.detach(g)  # 使用 PyTorch 的 detach 函数对输入 g 进行处理
            x = x + self.cond(g)  # 将输入 x 与经过条件处理的输入 g 相加
        x = self.conv_1(x * x_mask)  # 使用预定义的卷积神经网络层对输入 x 与掩码 x_mask 进行处理
        x = torch.relu(x)  # 使用 PyTorch 的 relu 函数对输入 x 进行激活函数处理
        x = self.norm_1(x)  # 使用预定义的神经网络层对输入 x 进行处理
        x = self.drop(x)  # 使用预定义的神经网络层对输入 x 进行 dropout 处理
        x = self.conv_2(x * x_mask)  # 使用预定义的卷积神经网络层对输入 x 与掩码 x_mask 进行处理
        x = torch.relu(x)  # 使用 PyTorch 的 relu 函数对输入 x 进行激活函数处理
        x = self.norm_2(x)  # 使用预定义的神经网络层对输入 x 进行处理
        x = self.drop(x)  # 对输入数据进行丢弃操作，去除一部分数据，以减少过拟合的可能性

        output_probs = []  # 初始化一个空列表，用于存储后续计算得到的输出概率
        for dur in [dur_r, dur_hat]:  # 遍历持续时间列表
            output_prob = self.forward_probability(x, x_mask, dur, g)  # 调用 forward_probability 方法计算输出概率
            output_probs.append(output_prob)  # 将计算得到的输出概率添加到列表中

        return output_probs  # 返回输出概率列表


class TransformerCouplingBlock(nn.Module):  # 定义一个名为 TransformerCouplingBlock 的类，继承自 nn.Module
    def __init__(  # 初始化方法，用于初始化类的属性
        self,  # 类的实例对象
        channels,  # 输入通道数
        hidden_channels,  # 隐藏层通道数
        filter_channels,  # 过滤器通道数
        n_heads,  # 多头注意力机制的头数
        n_layers,  # 层数
        kernel_size,  # 卷积核大小
        p_dropout,  # 丢弃概率
        n_flows=4,  # 定义了一个名为n_flows的参数，值为4
        gin_channels=0,  # 定义了一个名为gin_channels的参数，值为0
        share_parameter=False,  # 定义了一个名为share_parameter的参数，值为False
    ):
        super().__init__()  # 调用父类的初始化方法
        self.channels = channels  # 初始化一个名为channels的属性，值为传入的参数
        self.hidden_channels = hidden_channels  # 初始化一个名为hidden_channels的属性，值为传入的参数
        self.kernel_size = kernel_size  # 初始化一个名为kernel_size的属性，值为传入的参数
        self.n_layers = n_layers  # 初始化一个名为n_layers的属性，值为传入的参数
        self.n_flows = n_flows  # 初始化一个名为n_flows的属性，值为传入的参数
        self.gin_channels = gin_channels  # 初始化一个名为gin_channels的属性，值为传入的参数

        self.flows = nn.ModuleList()  # 初始化一个名为flows的属性，值为一个空的nn.ModuleList对象

        self.wn = (  # 初始化一个名为wn的属性
            attentions_onnx.FFT(  # 调用attentions_onnx模块中的FFT方法
                hidden_channels,  # 传入hidden_channels参数
                filter_channels,  # 传入filter_channels参数
                n_heads,  # 传入n_heads参数
                n_layers,  # 传入n_layers参数
                kernel_size,  # 定义变量 kernel_size，表示卷积核的大小
                p_dropout,  # 定义变量 p_dropout，表示dropout的概率
                isflow=True,  # 定义变量 isflow，表示是否为流模式，默认为True
                gin_channels=self.gin_channels,  # 定义变量 gin_channels，表示输入的通道数，使用self.gin_channels作为默认值
            )
            if share_parameter  # 如果 share_parameter 为真
            else None  # 否则为None
        )

        for i in range(n_flows):  # 循环n_flows次
            self.flows.append(  # 将新的TransformerCouplingLayer对象添加到self.flows列表中
                modules.TransformerCouplingLayer(  # 创建TransformerCouplingLayer对象
                    channels,  # 定义变量 channels，表示输入的通道数
                    hidden_channels,  # 定义变量 hidden_channels，表示隐藏层的通道数
                    kernel_size,  # 定义变量 kernel_size，表示卷积核的大小
                    n_layers,  # 定义变量 n_layers，表示Transformer的层数
                    n_heads,  # 定义变量 n_heads，表示Transformer的头数
                    p_dropout,  # 定义变量 p_dropout，表示dropout的概率
                    filter_channels,  # 定义变量 filter_channels，表示滤波器的通道数
                    mean_only=True,  # 定义变量 mean_only，表示是否只使用均值，默认为True
                    wn_sharing_parameter=self.wn,  # 设置参数 wn_sharing_parameter 为 self.wn
                    gin_channels=self.gin_channels,  # 设置参数 gin_channels 为 self.gin_channels
                )
            )
            self.flows.append(modules.Flip())  # 将 modules.Flip() 添加到 self.flows 列表中

    def forward(self, x, x_mask, g=None, reverse=True):
        if not reverse:  # 如果不是反向传播
            for flow in self.flows:  # 遍历 self.flows 列表中的每个元素
                x, _ = flow(x, x_mask, g=g, reverse=reverse)  # 调用 flow 函数进行前向传播
        else:  # 如果是反向传播
            for flow in reversed(self.flows):  # 遍历 self.flows 列表中的每个元素（反向）
                x = flow(x, x_mask, g=g, reverse=reverse)  # 调用 flow 函数进行反向传播
        return x  # 返回 x

class StochasticDurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels,  # 输入通道数
        filter_channels,  # 过滤通道数
        kernel_size,  # 卷积核大小
        p_dropout,  # 丢弃概率
        n_flows=4,  # 流的数量，默认为4
        gin_channels=0,  # gin通道数，默认为0
    ):
        super().__init__()  # 调用父类的构造函数
        filter_channels = in_channels  # 这行代码需要从将来的版本中移除
        self.in_channels = in_channels  # 输入通道数
        self.filter_channels = filter_channels  # 过滤通道数
        self.kernel_size = kernel_size  # 卷积核大小
        self.p_dropout = p_dropout  # 丢弃概率
        self.n_flows = n_flows  # 流的数量
        self.gin_channels = gin_channels  # gin通道数

        self.log_flow = modules.Log()  # 创建Log模块
        self.flows = nn.ModuleList()  # 创建模块列表
        self.flows.append(modules.ElementwiseAffine(2))  # 将ElementwiseAffine模块添加到模块列表中
        for i in range(n_flows):  # 循环n_flows次
            self.flows.append(  # 将以下模块添加到模块列表中
        modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)  # 创建一个包含2个输入通道的卷积流模块，设置滤波器通道数、核大小和层数
    )
    self.flows.append(modules.Flip())  # 将Flip模块添加到flows列表中

self.post_pre = nn.Conv1d(1, filter_channels, 1)  # 创建一个1维卷积层，用于后处理的预处理
self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)  # 创建一个1维卷积层，用于后处理的投影
self.post_convs = modules.DDSConv(  # 创建一个DDSConv模块，用于后处理的卷积
    filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
)
self.post_flows = nn.ModuleList()  # 创建一个空的模块列表，用于后处理的流
self.post_flows.append(modules.ElementwiseAffine(2))  # 将ElementwiseAffine模块添加到后处理的流中
for i in range(4):  # 循环4次
    self.post_flows.append(
        modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)  # 将包含2个输入通道的卷积流模块添加到后处理的流中
    )
    self.post_flows.append(modules.Flip())  # 将Flip模块添加到后处理的流中

self.pre = nn.Conv1d(in_channels, filter_channels, 1)  # 创建一个1维卷积层，用于预处理
self.proj = nn.Conv1d(filter_channels, filter_channels, 1)  # 创建一个1维卷积层，用于投影
self.convs = modules.DDSConv(  # 创建一个DDSConv模块，用于卷积
        filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )  # 创建一个WaveGlow模型的初始化函数，设置滤波器通道数、卷积核大小、层数和丢失率

        if gin_channels != 0:  # 如果输入的全局条件通道数不为0
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)  # 创建一个1维卷积层作为条件网络

    def forward(self, x, x_mask, z, g=None):  # WaveGlow模型的前向传播函数，输入为x、x_mask、z和g
        x = torch.detach(x)  # 将输入张量x的梯度分离
        x = self.pre(x)  # 将输入x传入预处理网络
        if g is not None:  # 如果全局条件g不为空
            g = torch.detach(g)  # 将全局条件g的梯度分离
            x = x + self.cond(g)  # 将条件网络对全局条件g的输出加到输入x上
        x = self.convs(x, x_mask)  # 将处理后的输入x和掩码x_mask传入卷积网络
        x = self.proj(x) * x_mask  # 将卷积网络的输出传入投影层并乘以掩码x_mask

        flows = list(reversed(self.flows))  # 将模型中的流层反转并转换为列表
        flows = flows[:-2] + [flows[-1]]  # 移除一个无用的流层
        for flow in flows:  # 遍历流层列表
            z = flow(z, x_mask, g=x, reverse=True)  # 将输入z、掩码x_mask和条件g传入流层进行反向传播
        z0, z1 = torch.split(z, [1, 1], 1)  # 将输出z按照指定维度分割为z0和z1
        logw = z0  # 将z0赋值给logw
        return logw
```
这行代码是一个函数的返回语句，它返回变量logw的值。

```
class DurationPredictor(nn.Module):
```
这行代码定义了一个名为DurationPredictor的类，它继承自nn.Module类。

```
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
```
这是DurationPredictor类的构造函数，它接受in_channels、filter_channels、kernel_size、p_dropout和可选的gin_channels作为参数。

```
        super().__init__()
```
这行代码调用了父类nn.Module的构造函数，确保正确地初始化了DurationPredictor类。

```
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels
```
这些行代码将构造函数的参数值分别赋给DurationPredictor类的成员变量。

```
        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
```
这些行代码创建了Dropout、Conv1d和LayerNorm的实例，并将它们分别赋给DurationPredictor类的成员变量。
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )  # 创建一个一维卷积层，用于处理输入数据，设置输入通道数、输出通道数、卷积核大小和填充大小

        self.norm_2 = modules.LayerNorm(filter_channels)  # 创建一个用于归一化的层，对输入数据进行标准化处理

        self.proj = nn.Conv1d(filter_channels, 1, 1)  # 创建一个一维卷积层，用于将处理后的数据投影到输出维度为1的空间

        if gin_channels != 0:  # 如果条件输入通道数不为0
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)  # 创建一个一维卷积层，用于处理条件输入数据

    def forward(self, x, x_mask, g=None):  # 定义前向传播函数，接受输入数据x、输入数据的掩码x_mask和条件输入数据g（可选）
        x = torch.detach(x)  # 将输入数据x从计算图中分离出来
        if g is not None:  # 如果条件输入数据g不为空
            g = torch.detach(g)  # 将条件输入数据g从计算图中分离出来
            x = x + self.cond(g)  # 将条件输入数据g经过一维卷积处理后的结果加到输入数据x上
        x = self.conv_1(x * x_mask)  # 对输入数据x按照掩码进行卷积处理
        x = torch.relu(x)  # 对卷积处理后的数据进行ReLU激活函数处理
        x = self.norm_1(x)  # 对处理后的数据进行归一化处理
        x = self.drop(x)  # 对处理后的数据进行丢弃操作
        x = self.conv_2(x * x_mask)  # 对处理后的数据再次进行卷积处理
        x = torch.relu(x)  # 对卷积处理后的数据进行ReLU激活函数处理
        x = self.norm_2(x)  # 对输入数据进行归一化处理
        x = self.drop(x)  # 对输入数据进行丢弃处理
        x = self.proj(x * x_mask)  # 对输入数据进行投影处理
        return x * x_mask  # 返回处理后的数据

class Bottleneck(nn.Sequential):
    def __init__(self, in_dim, hidden_dim):
        c_fc1 = nn.Linear(in_dim, hidden_dim, bias=False)  # 创建一个线性层，输入维度为in_dim，输出维度为hidden_dim，不使用偏置
        c_fc2 = nn.Linear(in_dim, hidden_dim, bias=False)  # 创建另一个线性层，输入维度为in_dim，输出维度为hidden_dim，不使用偏置
        super().__init__(*[c_fc1, c_fc2])  # 调用父类的初始化方法，将创建的线性层作为参数传入

class Block(nn.Module):
    def __init__(self, in_dim, hidden_dim) -> None:
        super().__init__()  # 调用父类的初始化方法
        self.norm = nn.LayerNorm(in_dim)  # 创建一个LayerNorm层，输入维度为in_dim
        self.mlp = MLP(in_dim, hidden_dim)  # 创建一个MLP模型，输入维度为in_dim，隐藏层维度为hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mlp(self.norm(x))  # 将输入 x 经过归一化处理后，通过多层感知机（MLP）进行处理，并将结果与原始输入相加
        return x  # 返回处理后的结果


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.c_fc1 = nn.Linear(in_dim, hidden_dim, bias=False)  # 创建一个全连接层，输入维度为 in_dim，输出维度为 hidden_dim，不使用偏置
        self.c_fc2 = nn.Linear(in_dim, hidden_dim, bias=False)  # 创建另一个全连接层，输入维度为 in_dim，输出维度为 hidden_dim，不使用偏置
        self.c_proj = nn.Linear(hidden_dim, in_dim, bias=False)  # 创建一个全连接层，输入维度为 hidden_dim，输出维度为 in_dim，不使用偏置

    def forward(self, x: torch.Tensor):
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)  # 将输入 x 分别经过第一个全连接层和 SILU 激活函数处理，然后与第二个全连接层处理后的结果相乘
        x = self.c_proj(x)  # 将上一步处理后的结果经过全连接层处理
        return x  # 返回处理后的结果


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab,  # 词汇量，用于定义模型的输入维度
        out_channels,  # 输出通道数，定义模型输出的通道数
        hidden_channels,  # 隐藏层通道数，定义模型隐藏层的通道数
        filter_channels,  # 过滤器通道数，定义模型中过滤器的通道数
        n_heads,  # 多头注意力机制中的头数
        n_layers,  # 模型中的层数
        kernel_size,  # 卷积核大小
        p_dropout,  # dropout概率，用于正则化
        n_speakers,  # 说话者数量，用于定义模型的输入维度
        gin_channels=0,  # GIN模型中的输入通道数，默认为0
    ):
        super().__init__()  # 调用父类的构造函数
        self.n_vocab = n_vocab  # 初始化词汇量
        self.out_channels = out_channels  # 初始化输出通道数
        self.hidden_channels = hidden_channels  # 初始化隐藏层通道数
        self.filter_channels = filter_channels  # 初始化过滤器通道数
        self.n_heads = n_heads  # 初始化多头注意力机制中的头数
        self.n_layers = n_layers  # 初始化模型中的层数
        self.kernel_size = kernel_size  # 初始化卷积核大小
        self.p_dropout = p_dropout  # 初始化dropout概率
        self.gin_channels = gin_channels  # 设置对象的属性 gin_channels
        self.emb = nn.Embedding(len(symbols), hidden_channels)  # 创建一个嵌入层，用于将符号映射到隐藏通道
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)  # 对嵌入层的权重进行正态分布初始化
        self.tone_emb = nn.Embedding(num_tones, hidden_channels)  # 创建一个嵌入层，用于将音调映射到隐藏通道
        nn.init.normal_(self.tone_emb.weight, 0.0, hidden_channels**-0.5)  # 对嵌入层的权重进行正态分布初始化
        self.language_emb = nn.Embedding(num_languages, hidden_channels)  # 创建一个嵌入层，用于将语言映射到隐藏通道
        nn.init.normal_(self.language_emb.weight, 0.0, hidden_channels**-0.5)  # 对嵌入层的权重进行正态分布初始化
        self.bert_proj = nn.Conv1d(1024, hidden_channels, 1)  # 创建一个卷积层，用于对输入进行一维卷积
        self.ja_bert_proj = nn.Conv1d(1024, hidden_channels, 1)  # 创建一个卷积层，用于对输入进行一维卷积
        self.en_bert_proj = nn.Conv1d(1024, hidden_channels, 1)  # 创建一个卷积层，用于对输入进行一维卷积
        # self.emo_proj = nn.Linear(1024, 1024)  # 创建一个线性层，用于对输入进行线性变换
        # self.emo_quantizer = nn.ModuleList()  # 创建一个模块列表，用于存储量化器模块
        # for i in range(0, n_speakers):  # 遍历说话者数量
        #    self.emo_quantizer.append(  # 向量量化器模块添加新的量化器
        #        VectorQuantize(
        #            dim=1024,
        #            codebook_size=10,
        #            decay=0.8,
        #            commitment_weight=1.0,
        #            learnable_codebook=True,
        #            ema_update=False,  # 设置 ema_update 参数为 False
        #        )
        #    )
        # self.emo_q_proj = nn.Linear(1024, hidden_channels)  # 使用 nn.Linear 创建一个线性变换，输入维度为 1024，输出维度为 hidden_channels
        self.n_speakers = n_speakers  # 初始化 self.n_speakers 为 n_speakers
        self.emo_proj = nn.Linear(512, hidden_channels)  # 使用 nn.Linear 创建一个线性变换，输入维度为 512，输出维度为 hidden_channels

        self.encoder = attentions_onnx.Encoder(  # 初始化 self.encoder 为 attentions_onnx.Encoder 类的实例
            hidden_channels,  # 隐藏层维度
            filter_channels,  # 过滤器通道数
            n_heads,  # 多头注意力机制的头数
            n_layers,  # 编码器层数
            kernel_size,  # 卷积核大小
            p_dropout,  # 丢弃概率
            gin_channels=self.gin_channels,  # gin_channels 参数设置为 self.gin_channels
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)  # 使用 nn.Conv1d 创建一个一维卷积层，输入通道数为 hidden_channels，输出通道数为 out_channels * 2，卷积核大小为 1

    def forward(  # 定义前向传播函数
        self, x, x_lengths, tone, language, bert, ja_bert, en_bert, emo, g=None  # 输入参数包括 x, x_lengths, tone, language, bert, ja_bert, en_bert, emo, g
        ):  # 定义一个函数，参数为 x, tone, language, bert, ja_bert, en_bert, emo
        x_mask = torch.ones_like(x).unsqueeze(0)  # 创建一个与 x 相同大小的全 1 张量，并在第 0 维度上增加一个维度
        bert_emb = self.bert_proj(bert.transpose(0, 1).unsqueeze(0)).transpose(1, 2)  # 对 bert 进行线性变换并转置，然后在第 0 维度上增加一个维度，并再次转置
        ja_bert_emb = self.ja_bert_proj(ja_bert.transpose(0, 1).unsqueeze(0)).transpose(1, 2)  # 对 ja_bert 进行线性变换并转置，然后在第 0 维度上增加一个维度，并再次转置
        en_bert_emb = self.en_bert_proj(en_bert.transpose(0, 1).unsqueeze(0)).transpose(1, 2)  # 对 en_bert 进行线性变换并转置，然后在第 0 维度上增加一个维度，并再次转置

        x = (
            self.emb(x)  # 对 x 进行嵌入操作
            + self.tone_emb(tone)  # 对 tone 进行嵌入操作
            + self.language_emb(language)  # 对 language 进行嵌入操作
            + bert_emb  # 加上 bert_emb
            + ja_bert_emb  # 加上 ja_bert_emb
            + en_bert_emb  # 加上 en_bert_emb
            + self.emo_proj(emo)  # 对 emo 进行线性变换
        ) * math.sqrt(
            self.hidden_channels  # 乘以 self.hidden_channels 的平方根
        )  # [b, t, h]  # 将张量 x 沿着第一个维度进行转置，将第一个维度和最后一个维度交换位置，得到新的张量
        x = torch.transpose(x, 1, -1)  # [b, h, t]  # 将张量 x_mask 的数据类型转换为与张量 x 相同的数据类型
        x_mask = x_mask.to(x.dtype)

        x = self.encoder(x * x_mask, x_mask, g=g)  # 使用编码器对输入张量 x 进行编码
        stats = self.proj(x) * x_mask  # 使用投影层对编码后的张量 x 进行投影，并乘以掩码张量 x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)  # 将投影后的张量 stats 沿着第一个维度进行分割，分割成 m 和 logs 两部分
        return x, m, logs, x_mask  # 返回编码后的张量 x、m、logs 和掩码张量 x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,  # 初始化残差耦合块的参数，包括通道数、隐藏层通道数、卷积核大小、扩张率、层数和流数
        gin_channels=0,  # 设置默认参数 gin_channels 为 0
    ):
        super().__init__()  # 调用父类的构造函数
        self.channels = channels  # 初始化对象的属性 channels
        self.hidden_channels = hidden_channels  # 初始化对象的属性 hidden_channels
        self.kernel_size = kernel_size  # 初始化对象的属性 kernel_size
        self.dilation_rate = dilation_rate  # 初始化对象的属性 dilation_rate
        self.n_layers = n_layers  # 初始化对象的属性 n_layers
        self.n_flows = n_flows  # 初始化对象的属性 n_flows
        self.gin_channels = gin_channels  # 初始化对象的属性 gin_channels

        self.flows = nn.ModuleList()  # 创建一个空的 nn.ModuleList 对象，用于存储流程
        for i in range(n_flows):  # 遍历 n_flows 次
            self.flows.append(  # 向 flows 中添加元素
                modules.ResidualCouplingLayer(  # 创建 ResidualCouplingLayer 对象
                    channels,  # 传入参数 channels
                    hidden_channels,  # 传入参数 hidden_channels
                    kernel_size,  # 传入参数 kernel_size
                    dilation_rate,  # 传入参数 dilation_rate
                    n_layers,  # 传入参数 n_layers
                    gin_channels=gin_channels,  # 设置参数gin_channels的值为gin_channels
                    mean_only=True,  # 设置参数mean_only的值为True
                )
            )
            self.flows.append(modules.Flip())  # 将modules.Flip()添加到self.flows列表中

    def forward(self, x, x_mask, g=None, reverse=True):  # 定义前向传播函数，接受输入x, x_mask, g和reverse参数
        if not reverse:  # 如果reverse参数为False
            for flow in self.flows:  # 遍历self.flows列表中的每个flow
                x, _ = flow(x, x_mask, g=g, reverse=reverse)  # 调用flow的前向传播函数，更新x的值
        else:  # 如果reverse参数为True
            for flow in reversed(self.flows):  # 遍历self.flows列表中的每个flow（倒序）
                x = flow(x, x_mask, g=g, reverse=reverse)  # 调用flow的前向传播函数，更新x的值
        return x  # 返回更新后的x


class PosteriorEncoder(nn.Module):  # 定义PosteriorEncoder类，继承自nn.Module类
    def __init__(
        self,
        in_channels,  # 定义参数in_channels
        out_channels,  # 输出通道数
        hidden_channels,  # 隐藏层通道数
        kernel_size,  # 卷积核大小
        dilation_rate,  # 膨胀率
        n_layers,  # 卷积层数
        gin_channels=0,  # 输入图的通道数，默认为0

        super().__init__()  # 调用父类的初始化方法
        self.in_channels = in_channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数
        self.hidden_channels = hidden_channels  # 隐藏层通道数
        self.kernel_size = kernel_size  # 卷积核大小
        self.dilation_rate = dilation_rate  # 膨胀率
        self.n_layers = n_layers  # 卷积层数
        self.gin_channels = gin_channels  # 输入图的通道数

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)  # 创建一个1维卷积层，用于数据预处理
        self.enc = modules.WN(  # 创建一个WaveNet模块
            hidden_channels,  # 输入通道数
            kernel_size,  # 卷积核大小
        dilation_rate,  # 定义了卷积层的膨胀率
        n_layers,  # 定义了网络的层数
        gin_channels=gin_channels,  # 定义了输入的通道数，默认为gin_channels
    )
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)  # 定义了一个1维卷积层，用于投影

def forward(self, x, x_lengths, g=None):
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
        x.dtype
    )  # 根据输入的长度创建一个掩码，用于屏蔽填充部分
    x = self.pre(x) * x_mask  # 对输入进行预处理，并乘以掩码
    x = self.enc(x, x_mask, g=g)  # 对输入进行编码
    stats = self.proj(x) * x_mask  # 使用投影层对编码结果进行处理，并乘以掩码
    m, logs = torch.split(stats, self.out_channels, dim=1)  # 将处理后的结果分割成均值和标准差
    z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask  # 生成服从正态分布的随机数，并乘以掩码
    return z, m, logs, x_mask  # 返回生成的随机数、均值、标准差和掩码


class Generator(torch.nn.Module):
    def __init__(
        self,  # 定义一个类的初始化方法，self代表类的实例
        initial_channel,  # 初始通道数
        resblock,  # 残差块类型，1或2
        resblock_kernel_sizes,  # 残差块的卷积核大小
        resblock_dilation_sizes,  # 残差块的扩张大小
        upsample_rates,  # 上采样率
        upsample_initial_channel,  # 上采样的初始通道数
        upsample_kernel_sizes,  # 上采样的卷积核大小
        gin_channels=0,  # 输入通道数，默认为0
    ):
        super(Generator, self).__init__()  # 调用父类的初始化方法
        self.num_kernels = len(resblock_kernel_sizes)  # 计算残差块的数量
        self.num_upsamples = len(upsample_rates)  # 计算上采样的数量
        self.conv_pre = Conv1d(  # 定义一个一维卷积层
            initial_channel, upsample_initial_channel, 7, 1, padding=3  # 输入通道数、输出通道数、卷积核大小、步长、填充
        )
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2  # 根据resblock的值选择不同的残差块类型

        self.ups = nn.ModuleList()  # 定义一个空的模块列表
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):  # 遍历上采样率和卷积核大小
            self.ups.append(  # 将经过权重归一化的一维转置卷积层添加到self.ups列表中
                weight_norm(  # 对卷积层进行权重归一化处理
                    ConvTranspose1d(  # 创建一维转置卷积层
                        upsample_initial_channel // (2**i),  # 输入通道数
                        upsample_initial_channel // (2 ** (i + 1)),  # 输出通道数
                        k,  # 卷积核大小
                        u,  # 上采样步长
                        padding=(k - u) // 2,  # 填充大小
                    )
                )
            )

        self.resblocks = nn.ModuleList()  # 创建一个空的ModuleList用于存储残差块
        for i in range(len(self.ups)):  # 遍历self.ups列表
            ch = upsample_initial_channel // (2 ** (i + 1))  # 计算通道数
            for j, (k, d) in enumerate(  # 遍历resblock_kernel_sizes和resblock_dilation_sizes
                zip(resblock_kernel_sizes, resblock_dilation_sizes)  # 将resblock_kernel_sizes和resblock_dilation_sizes进行压缩
            ):
                self.resblocks.append(resblock(ch, k, d))  # 将创建的残差块添加到self.resblocks中
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)  # 创建一个 1 维卷积层，用于处理输入数据
        self.ups.apply(init_weights)  # 对上采样层进行初始化权重操作

        if gin_channels != 0:  # 如果输入的条件通道不为 0
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)  # 创建一个 1 维卷积层，用于处理条件输入数据

    def forward(self, x, g=None):  # 定义前向传播函数，接受输入数据 x 和条件输入数据 g
        x = self.conv_pre(x)  # 对输入数据进行卷积处理
        if g is not None:  # 如果条件输入数据不为空
            x = x + self.cond(g)  # 将条件输入数据经过卷积处理后的结果与输入数据相加

        for i in range(self.num_upsamples):  # 循环进行上采样操作
            x = F.leaky_relu(x, modules.LRELU_SLOPE)  # 对输入数据进行 Leaky ReLU 激活函数处理
            x = self.ups[i](x)  # 对输入数据进行上采样操作
            xs = None  # 初始化一个变量 xs
            for j in range(self.num_kernels):  # 循环进行卷积核处理
                if xs is None:  # 如果 xs 为空
                    xs = self.resblocks[i * self.num_kernels + j](x)  # 对输入数据进行残差块处理并赋值给 xs
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)  # 将输入数据经过残差块处理后的结果与 xs 相加
        x = xs / self.num_kernels  # 将输入xs除以self.num_kernels，得到x
        x = F.leaky_relu(x)  # 使用Leaky ReLU激活函数处理x
        x = self.conv_post(x)  # 使用self.conv_post对x进行卷积操作
        x = torch.tanh(x)  # 使用双曲正切函数处理x

        return x  # 返回处理后的x

    def remove_weight_norm(self):
        print("Removing weight norm...")  # 打印信息，表示正在移除权重归一化
        for layer in self.ups:  # 遍历self.ups中的每一层
            remove_weight_norm(layer)  # 调用remove_weight_norm函数移除权重归一化
        for layer in self.resblocks:  # 遍历self.resblocks中的每一层
            layer.remove_weight_norm()  # 调用layer的remove_weight_norm方法移除权重归一化


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()  # 调用父类的构造函数进行初始化
        self.period = period  # 初始化self.period为传入的period参数
        self.use_spectral_norm = use_spectral_norm  # 初始化self.use_spectral_norm为传入的use_spectral_norm参数
        # 根据 use_spectral_norm 的值选择使用 weight_norm 还是 spectral_norm
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        # 创建包含卷积层的 nn.ModuleList
        self.convs = nn.ModuleList(
            [
                # 使用 norm_f 对第一个卷积层进行归一化
                norm_f(
                    Conv2d(
                        1,  # 输入通道数
                        32,  # 输出通道数
                        (kernel_size, 1),  # 卷积核大小
                        (stride, 1),  # 步长
                        padding=(get_padding(kernel_size, 1), 0),  # 填充
                    )
                ),
                # 使用 norm_f 对第二个卷积层进行归一化
                norm_f(
                    Conv2d(
                        32,  # 输入通道数
                        128,  # 输出通道数
                        (kernel_size, 1),  # 卷积核大小
                        (stride, 1),  # 步长
                        padding=(get_padding(kernel_size, 1), 0),  # 填充
                    )
                )
                ),
                # 使用 Conv2d 函数创建一个 128 到 512 的卷积层，设置卷积核大小为 (kernel_size, 1)，步长为 (stride, 1)，填充为 (get_padding(kernel_size, 1), 0)
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                # 使用 Conv2d 函数创建一个 512 到 1024 的卷积层，设置卷积核大小为 (kernel_size, 1)，步长为 (stride, 1)，填充为 (get_padding(kernel_size, 1), 0)
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                # 使用 norm_f 函数对上一层的输出进行归一化处理
                norm_f(
                    Conv2d(
                        1024,  # 输入通道数
                        1024,  # 输出通道数
                        (kernel_size, 1),  # 卷积核大小，这里是一个元组，表示高和宽的大小
                        1,  # 步长
                        padding=(get_padding(kernel_size, 1), 0),  # 填充
                    )
                ),  # 定义一个二维卷积层，并添加到列表中
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))  # 定义一个二维卷积层，并进行归一化处理

    def forward(self, x):
        fmap = []  # 初始化一个空列表用于存储特征图

        # 1d to 2d
        b, c, t = x.shape  # 获取输入张量的维度信息
        if t % self.period != 0:  # 如果时间维度不能整除周期
            n_pad = self.period - (t % self.period)  # 计算需要填充的长度
            x = F.pad(x, (0, n_pad), "reflect")  # 对输入张量进行反射填充
        t = t + n_pad  # 将变量t增加n_pad的值
        x = x.view(b, c, t // self.period, self.period)  # 重新调整张量x的形状

        for layer in self.convs:  # 遍历self.convs中的每一层
            x = layer(x)  # 对x进行卷积操作
            x = F.leaky_relu(x, modules.LRELU_SLOPE)  # 对x进行Leaky ReLU激活函数处理
            fmap.append(x)  # 将处理后的x添加到fmap列表中
        x = self.conv_post(x)  # 对x进行卷积操作
        fmap.append(x)  # 将处理后的x添加到fmap列表中
        x = torch.flatten(x, 1, -1)  # 对x进行展平操作

        return x, fmap  # 返回处理后的x和fmap列表


class DiscriminatorS(torch.nn.Module):  # 定义名为DiscriminatorS的类，继承自torch.nn.Module类
    def __init__(self, use_spectral_norm=False):  # 定义初始化方法，接受一个布尔类型的参数use_spectral_norm
        super(DiscriminatorS, self).__init__()  # 调用父类的初始化方法
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm  # 根据use_spectral_norm的值选择不同的归一化函数
        self.convs = nn.ModuleList(  # 创建一个包含卷积层的ModuleList
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),  # 使用Conv1d函数创建一个卷积层，输入通道数为1，输出通道数为16，卷积核大小为15，步长为1，填充为7
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),  # 使用Conv1d函数创建一个卷积层，输入通道数为16，输出通道数为64，卷积核大小为41，步长为4，分组数为4，填充为20
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),  # 使用Conv1d函数创建一个卷积层，输入通道数为64，输出通道数为256，卷积核大小为41，步长为4，分组数为16，填充为20
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),  # 使用Conv1d函数创建一个卷积层，输入通道数为256，输出通道数为1024，卷积核大小为41，步长为4，分组数为64，填充为20
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),  # 使用Conv1d函数创建一个卷积层，输入通道数为1024，输出通道数为1024，卷积核大小为41，步长为4，分组数为256，填充为20
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),  # 使用Conv1d函数创建一个卷积层，输入通道数为1024，输出通道数为1024，卷积核大小为5，步长为1，填充为2
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))  # 使用Conv1d函数创建一个卷积层，输入通道数为1024，输出通道数为1，卷积核大小为3，步长为1，填充为1

    def forward(self, x):
        fmap = []

        for layer in self.convs:  # 遍历self.convs中的每个卷积层
            x = layer(x)  # 将输入x传入当前卷积层进行卷积操作
            x = F.leaky_relu(x, modules.LRELU_SLOPE)  # 对卷积结果进行Leaky ReLU激活函数处理
            fmap.append(x)  # 将处理后的结果添加到fmap列表中
        x = self.conv_post(x)  # 将处理后的结果再传入最后一个卷积层进行卷积操作
        fmap.append(x)  # 将最终处理后的结果添加到fmap列表中
        x = torch.flatten(x, 1, -1)  # 将最终处理后的结果展平成一维张量
class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]  # 定义一个包含多个整数的列表

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]  # 创建一个包含单个DiscriminatorS对象的列表
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]  # 将包含多个DiscriminatorP对象的列表与之前的列表合并
        self.discriminators = nn.ModuleList(discs)  # 创建一个包含所有discs对象的ModuleList

    def forward(self, y, y_hat):
        y_d_rs = []  # 创建一个空列表
        y_d_gs = []  # 创建一个空列表
        fmap_rs = []  # 创建一个空列表
        fmap_gs = []  # 创建一个空列表
        for i, d in enumerate(self.discriminators):
            # 使用enumerate函数遍历self.discriminators列表，i为索引，d为元素
            y_d_r, fmap_r = d(y)
            # 调用d函数，传入y参数，将返回值分别赋给y_d_r和fmap_r
            y_d_g, fmap_g = d(y_hat)
            # 调用d函数，传入y_hat参数，将返回值分别赋给y_d_g和fmap_g
            y_d_rs.append(y_d_r)
            # 将y_d_r添加到y_d_rs列表中
            y_d_gs.append(y_d_g)
            # 将y_d_g添加到y_d_gs列表中
            fmap_rs.append(fmap_r)
            # 将fmap_r添加到fmap_rs列表中
            fmap_gs.append(fmap_g)
            # 将fmap_g添加到fmap_gs列表中

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
        # 返回y_d_rs, y_d_gs, fmap_rs, fmap_gs这四个列表


class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self, spec_channels, gin_channels=0):
        # 初始化函数，接受spec_channels和gin_channels两个参数
        super().__init__()
        # 调用父类的初始化函数
        self.spec_channels = spec_channels
        # 将参数spec_channels赋给self.spec_channels
        # 定义参考编码器的滤波器数量
        ref_enc_filters = [32, 32, 64, 64, 128, 128]
        # 获取参考编码器滤波器数量的长度
        K = len(ref_enc_filters)
        # 创建滤波器列表，包括输入通道数和参考编码器滤波器数量
        filters = [1] + ref_enc_filters
        # 创建卷积层列表，每一层都是带有权重归一化的卷积层
        convs = [
            weight_norm(
                nn.Conv2d(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                )
            )
            for i in range(K)
        ]
        # 将卷积层列表转换为模块列表
        self.convs = nn.ModuleList(convs)
        # 计算输出通道数
        out_channels = self.calculate_channels(spec_channels, 3, 2, 1, K)
        # 创建 GRU 层
        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1] * out_channels,  # 设置输入大小为参考编码器最后一个滤波器的数量乘以输出通道数
            hidden_size=256 // 2,  # 设置隐藏层大小为256的一半
            batch_first=True,  # 设置输入数据的第一个维度为batch大小
        )
        self.proj = nn.Linear(128, gin_channels)  # 创建一个线性层，输入大小为128，输出大小为gin_channels

    def forward(self, inputs, mask=None):
        N = inputs.size(0)  # 获取输入数据的batch大小
        out = inputs.view(N, 1, -1, self.spec_channels)  # 将输入数据重塑为[N, 1, Ty, n_freqs]的形状
        for conv in self.convs:  # 遍历卷积层列表
            out = conv(out)  # 对输入数据进行卷积操作
            # out = wn(out)
            out = F.relu(out)  # 对卷积后的数据进行ReLU激活函数操作，得到[N, 128, Ty//2^K, n_mels//2^K]的输出

        out = out.transpose(1, 2)  # 将输出数据进行维度转置，得到[N, Ty//2^K, 128, n_mels//2^K]的形状
        T = out.size(1)  # 获取转置后的输出数据的第二个维度大小
        N = out.size(0)  # 获取转置后的输出数据的第一个维度大小
        out = out.contiguous().view(N, T, -1)  # 将输出数据连续化，得到[N, Ty//2^K, 128*n_mels//2^K]的形状

        self.gru.flatten_parameters()  # 将GRU层的参数展平，以便进行前向传播
        memory, out = self.gru(out)  # 使用GRU模型处理输入数据out，并将处理后的结果保存到memory中，同时将输出保存到out中 [1, N, 128]

        return self.proj(out.squeeze(0))  # 将out中的数据进行压缩，然后通过self.proj进行投影处理，返回处理后的结果

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):  # 循环n_convs次
            L = (L - kernel_size + 2 * pad) // stride + 1  # 根据给定的公式计算L的值
        return L  # 返回计算后的L的值


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        n_vocab,
        spec_channels,
        segment_size,
        inter_channels,  # 用于定义Transformer模型中间层的通道数
        hidden_channels,  # 用于定义Transformer模型隐藏层的通道数
        filter_channels,  # 用于定义Transformer模型中的滤波器通道数
        n_heads,  # 用于定义Transformer模型中的注意力头数
        n_layers,  # 用于定义Transformer模型中的层数
        kernel_size,  # 用于定义卷积核的大小
        p_dropout,  # 用于定义Dropout层的概率
        resblock,  # 用于定义是否使用残差块
        resblock_kernel_sizes,  # 用于定义残差块中卷积核的大小
        resblock_dilation_sizes,  # 用于定义残差块中卷积核的扩张大小
        upsample_rates,  # 用于定义上采样的比例
        upsample_initial_channel,  # 用于定义上采样初始通道数
        upsample_kernel_sizes,  # 用于定义上采样中卷积核的大小
        n_speakers=256,  # 用于定义说话人的数量，默认为256
        gin_channels=256,  # 用于定义Gated Information Network的通道数，默认为256
        use_sdp=True,  # 用于定义是否使用Self Distillation Process，默认为True
        n_flow_layer=4,  # 用于定义Flow模型中的层数
        n_layers_trans_flow=4,  # 用于定义Transformer Flow模型中的层数
        flow_share_parameter=False,  # 用于定义Flow模型中是否共享参数，默认为False
        use_transformer_flow=True,  # 用于定义是否使用Transformer Flow模型，默认为True
**kwargs,  # 接收任意数量的关键字参数，存储在kwargs字典中
):
    super().__init__()  # 调用父类的构造方法
    self.n_vocab = n_vocab  # 初始化n_vocab属性
    self.spec_channels = spec_channels  # 初始化spec_channels属性
    self.inter_channels = inter_channels  # 初始化inter_channels属性
    self.hidden_channels = hidden_channels  # 初始化hidden_channels属性
    self.filter_channels = filter_channels  # 初始化filter_channels属性
    self.n_heads = n_heads  # 初始化n_heads属性
    self.n_layers = n_layers  # 初始化n_layers属性
    self.kernel_size = kernel_size  # 初始化kernel_size属性
    self.p_dropout = p_dropout  # 初始化p_dropout属性
    self.resblock = resblock  # 初始化resblock属性
    self.resblock_kernel_sizes = resblock_kernel_sizes  # 初始化resblock_kernel_sizes属性
    self.resblock_dilation_sizes = resblock_dilation_sizes  # 初始化resblock_dilation_sizes属性
    self.upsample_rates = upsample_rates  # 初始化upsample_rates属性
    self.upsample_initial_channel = upsample_initial_channel  # 初始化upsample_initial_channel属性
    self.upsample_kernel_sizes = upsample_kernel_sizes  # 初始化upsample_kernel_sizes属性
    self.segment_size = segment_size  # 初始化segment_size属性
    self.n_speakers = n_speakers  # 初始化n_speakers属性
        self.gin_channels = gin_channels  # 设置对象的属性gin_channels为传入的参数gin_channels
        self.n_layers_trans_flow = n_layers_trans_flow  # 设置对象的属性n_layers_trans_flow为传入的参数n_layers_trans_flow
        self.use_spk_conditioned_encoder = kwargs.get("use_spk_conditioned_encoder", True)  # 设置对象的属性use_spk_conditioned_encoder为传入的参数kwargs中的"use_spk_conditioned_encoder"，如果不存在则默认为True
        self.use_sdp = use_sdp  # 设置对象的属性use_sdp为传入的参数use_sdp
        self.use_noise_scaled_mas = kwargs.get("use_noise_scaled_mas", False)  # 设置对象的属性use_noise_scaled_mas为传入的参数kwargs中的"use_noise_scaled_mas"，如果不存在则默认为False
        self.mas_noise_scale_initial = kwargs.get("mas_noise_scale_initial", 0.01)  # 设置对象的属性mas_noise_scale_initial为传入的参数kwargs中的"mas_noise_scale_initial"，如果不存在则默认为0.01
        self.noise_scale_delta = kwargs.get("noise_scale_delta", 2e-6)  # 设置对象的属性noise_scale_delta为传入的参数kwargs中的"noise_scale_delta"，如果不存在则默认为2e-6
        self.current_mas_noise_scale = self.mas_noise_scale_initial  # 设置对象的属性current_mas_noise_scale为mas_noise_scale_initial的值
        if self.use_spk_conditioned_encoder and gin_channels > 0:  # 如果use_spk_conditioned_encoder为True且gin_channels大于0
            self.enc_gin_channels = gin_channels  # 设置对象的属性enc_gin_channels为gin_channels的值
        self.enc_p = TextEncoder(  # 设置对象的属性enc_p为TextEncoder类的实例化对象
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
        p_dropout,  # 模型中的 dropout 概率
        self.n_speakers,  # 模型中的说话者数量
        gin_channels=self.enc_gin_channels,  # GIN 模块的输入通道数
    )
    self.dec = Generator(
        inter_channels,  # 生成器中的中间通道数
        resblock,  # 使用的残差块类型
        resblock_kernel_sizes,  # 残差块的卷积核大小
        resblock_dilation_sizes,  # 残差块的膨胀大小
        upsample_rates,  # 上采样率
        upsample_initial_channel,  # 上采样初始通道数
        upsample_kernel_sizes,  # 上采样的卷积核大小
        gin_channels=gin_channels,  # GIN 模块的输入通道数
    )
    self.enc_q = PosteriorEncoder(
        spec_channels,  # 音频频谱的通道数
        inter_channels,  # 编码器中的中间通道数
        hidden_channels,  # 编码器中的隐藏层通道数
        5,  # 编码器中的卷积层数量
        1,  # 编码器中的卷积核大小
            16,  # 设置一个参数为16
            gin_channels=gin_channels,  # 使用给定的gin_channels参数
        )
        if use_transformer_flow:  # 如果use_transformer_flow为True
            self.flow = TransformerCouplingBlock(  # 创建一个TransformerCouplingBlock对象
                inter_channels,  # 使用给定的inter_channels参数
                hidden_channels,  # 使用给定的hidden_channels参数
                filter_channels,  # 使用给定的filter_channels参数
                n_heads,  # 使用给定的n_heads参数
                n_layers_trans_flow,  # 使用给定的n_layers_trans_flow参数
                5,  # 设置一个参数为5
                p_dropout,  # 使用给定的p_dropout参数
                n_flow_layer,  # 使用给定的n_flow_layer参数
                gin_channels=gin_channels,  # 使用给定的gin_channels参数
                share_parameter=flow_share_parameter,  # 使用给定的flow_share_parameter参数
            )
        else:  # 如果use_transformer_flow为False
            self.flow = ResidualCouplingBlock(  # 创建一个ResidualCouplingBlock对象
                inter_channels,  # 使用给定的inter_channels参数
                hidden_channels,  # 使用给定的hidden_channels参数
                5,  # 设置参数为5
                1,  # 设置参数为1
                n_flow_layer,  # 设置参数为n_flow_layer
                gin_channels=gin_channels,  # 设置参数gin_channels为gin_channels
            )
        self.sdp = StochasticDurationPredictor(
            hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels  # 初始化StochasticDurationPredictor对象
        )
        self.dp = DurationPredictor(
            hidden_channels, 256, 3, 0.5, gin_channels=gin_channels  # 初始化DurationPredictor对象
        )

        if n_speakers >= 1:  # 如果说话者数量大于等于1
            self.emb_g = nn.Embedding(n_speakers, gin_channels)  # 创建Embedding对象
        else:
            self.ref_enc = ReferenceEncoder(spec_channels, gin_channels)  # 创建ReferenceEncoder对象

    def export_onnx(
        self,
        path,  # 导出ONNX模型的路径
        max_len=None,  # 定义一个参数 max_len，表示最大长度，默认为 None
        sdp_ratio=0,   # 定义一个参数 sdp_ratio，表示 sdp 比率，默认为 0
        y=None,        # 定义一个参数 y，表示标签，默认为 None
    ):
        noise_scale = 0.667  # 定义一个变量 noise_scale，表示噪声比例
        length_scale = 1      # 定义一个变量 length_scale，表示长度比例
        noise_scale_w = 0.8    # 定义一个变量 noise_scale_w，表示噪声比例 w
        x = (                  # 定义一个变量 x，表示一个长列表
            torch.LongTensor(   # 创建一个长整型张量
                [
                    0,           # 列表中的元素
                    97,          # 列表中的元素
                    0,           # 列表中的元素
                    8,           # 列表中的元素
                    0,           # 列表中的元素
                    78,          # 列表中的元素
                    0,           # 列表中的元素
                    8,           # 列表中的元素
                    0,           # 列表中的元素
                    76,          # 列表中的元素
# 创建一个空的字节流对象
bio = BytesIO()
# 将给定的列表中的每个元素转换为字节，并将其写入字节流对象中
for num in [0, 37, 0, 40, 0, 97, 0, 8, 0, 23, 0, 8, 0, 74, 0, 26, 0, 104, 0]:
    bio.write(num.to_bytes(2, 'little'))
        )
        .unsqueeze(0)  # 在第0维度上增加一个维度
        .cpu()  # 将数据移动到 CPU 上进行处理
    )
    tone = torch.zeros_like(x).cpu()  # 创建一个与 x 相同大小的全零张量，并移动到 CPU 上
    language = torch.zeros_like(x).cpu()  # 创建一个与 x 相同大小的全零张量，并移动到 CPU 上
    x_lengths = torch.LongTensor([x.shape[1]]).cpu()  # 创建一个包含 x.shape[1] 的长整型张量，并移动到 CPU 上
    sid = torch.LongTensor([0]).cpu()  # 创建一个包含 0 的长整型张量，并移动到 CPU 上
    bert = torch.randn(size=(x.shape[1], 1024)).cpu()  # 创建一个大小为 (x.shape[1], 1024) 的随机张量，并移动到 CPU 上
    ja_bert = torch.randn(size=(x.shape[1], 1024)).cpu()  # 创建一个大小为 (x.shape[1], 1024) 的随机张量，并移动到 CPU 上
    en_bert = torch.randn(size=(x.shape[1], 1024)).cpu()  # 创建一个大小为 (x.shape[1], 1024) 的随机张量，并移动到 CPU 上

    if self.n_speakers > 0:  # 如果说话者数量大于0
        g = self.emb_g(sid).unsqueeze(-1)  # 从 self.emb_g 中获取 g，并在最后一个维度上增加一个维度
        torch.onnx.export(  # 使用 torch.onnx.export 导出模型
            self.emb_g,  # 要导出的模型
            (sid),  # 模型的输入
            f"onnx/{path}/{path}_emb.onnx",  # 导出的文件路径
            input_names=["sid"],  # 输入的名称
            output_names=["g"],  # 输出的名称
        verbose=True,  # 设置为True时，会输出详细的导出过程信息
    )
else:  # 如果条件不满足
    g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)  # 对y进行转置并进行编码处理

emo = torch.randn(512, 1)  # 生成一个大小为512x1的随机张量

torch.onnx.export(  # 使用torch.onnx.export函数导出模型
    self.enc_p,  # 待导出的模型
    (x, x_lengths, tone, language, bert, ja_bert, en_bert, emo, g),  # 输入参数
    f"onnx/{path}/{path}_enc_p.onnx",  # 导出的ONNX文件路径
    input_names=[  # 输入参数的名称列表
        "x",
        "x_lengths",
        "t",
        "language",
        "bert_0",
        "bert_1",
        "bert_2",
        "emo",
# 定义输入参数
input_names=["x", "t", "language", "bert_0", "bert_1", "bert_2"]
# 定义输出参数
output_names=["xout", "m_p", "logs_p", "x_mask"]
# 定义动态轴
dynamic_axes={
    "x": [0, 1],
    "t": [0, 1],
    "language": [0, 1],
    "bert_0": [0],
    "bert_1": [0],
    "bert_2": [0],
    "xout": [0, 2],
    "m_p": [0, 2],
    "logs_p": [0, 2],
    "x_mask": [0, 2],
}
# 设置为详细模式
verbose=True
# 设置操作集版本
opset_version=16
        )

        # 创建一个随机的输入张量 zinput，形状与 x 相同，但是只有两个通道
        zinput = (
            torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype)
            * noise_scale_w
        )
        # 使用 torch.onnx.export 函数将 self.sdp 模型导出为 ONNX 格式的文件
        torch.onnx.export(
            self.sdp,  # 要导出的模型
            (x, x_mask, zinput, g),  # 模型的输入
            f"onnx/{path}/{path}_sdp.onnx",  # 导出的文件路径
            input_names=["x", "x_mask", "zin", "g"],  # 输入张量的名称
            output_names=["logw"],  # 输出张量的名称
            dynamic_axes={"x": [0, 2], "x_mask": [0, 2], "zin": [0, 2], "logw": [0, 2]},  # 动态维度
            verbose=True,  # 是否打印详细信息
        )
        # 使用 torch.onnx.export 函数将 self.dp 模型导出为 ONNX 格式的文件
        torch.onnx.export(
            self.dp,  # 要导出的模型
            (x, x_mask, g),  # 模型的输入
            f"onnx/{path}/{path}_dp.onnx",  # 导出的文件路径
        input_names=["x", "x_mask", "g"],  # 定义输入的名称列表
        output_names=["logw"],  # 定义输出的名称列表
        dynamic_axes={"x": [0, 2], "x_mask": [0, 2], "logw": [0, 2]},  # 定义动态轴
        verbose=True,  # 设置是否显示详细信息
    )
    logw = self.sdp(x, x_mask, zinput, g=g) * (sdp_ratio) + self.dp(
        x, x_mask, g=g
    ) * (1 - sdp_ratio)  # 计算logw的值
    w = torch.exp(logw) * x_mask * length_scale  # 计算w的值
    w_ceil = torch.ceil(w)  # 对w进行向上取整
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()  # 计算y_lengths的值
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
        x_mask.dtype
    )  # 生成y_mask
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)  # 生成attn_mask
    attn = commons.generate_path(w_ceil, attn_mask)  # 生成attn
    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
        1, 2
    )  # [b, t', t], [b, t, d] -> [b, d, t']  # 计算m_p的值
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # 计算注意力权重和对数概率的矩阵乘法，然后进行转置操作，得到结果矩阵的转置

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale  # 根据均值m_p和对数概率logs_p生成服从正态分布的随机数z_p

        torch.onnx.export(
            self.flow,
            (z_p, y_mask, g),
            f"onnx/{path}/{path}_flow.onnx",
            input_names=["z_p", "y_mask", "g"],
            output_names=["z"],
            dynamic_axes={"z_p": [0, 2], "y_mask": [0, 2], "z": [0, 2]},
            verbose=True,
        )  # 将模型self.flow导出为ONNX格式，并指定输入输出的名称和动态轴

        z = self.flow(z_p, y_mask, g=g, reverse=True)  # 使用模型self.flow对z_p进行反向传播得到z

        z_in = (z * y_mask)[:, :, :max_len]  # 对z乘以y_mask并截取前max_len个元素，得到z_in

        torch.onnx.export(
            self.dec,
            (z_in, g),  # 从元组中解包 z_in 和 g
            f"onnx/{path}/{path}_dec.onnx",  # 使用 f-string 构建 ONNX 文件的路径
            input_names=["z_in", "g"],  # 设置输入节点的名称为 "z_in" 和 "g"
            output_names=["o"],  # 设置输出节点的名称为 "o"
            dynamic_axes={"z_in": [0, 2], "o": [0, 2]},  # 设置动态轴的范围
            verbose=True,  # 设置为详细模式
        )
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)  # 调用 self.dec 方法并传入参数
```