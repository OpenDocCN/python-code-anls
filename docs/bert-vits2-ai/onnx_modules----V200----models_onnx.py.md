# `Bert-VITS2\onnx_modules\V200\models_onnx.py`

```
# 导入数学库
import math
# 导入 PyTorch 库
import torch
# 从 PyTorch 库中导入 nn 模块
from torch import nn
# 从 PyTorch 的 nn 模块中导入 functional 模块并重命名为 F
from torch.nn import functional as F
# 导入自定义的 commons 模块
import commons
# 导入自定义的 modules 模块
import modules
# 从当前目录下的 attentions_onnx 模块中导入所有内容
from . import attentions_onnx
# 从 PyTorch 的 nn 模块中导入 Conv1d、ConvTranspose1d、Conv2d 模块
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
# 从 PyTorch 的 nn.utils 模块中导入 weight_norm、remove_weight_norm、spectral_norm 模块
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
# 从 commons 模块中导入 init_weights、get_padding 函数
from commons import init_weights, get_padding
# 从当前目录下的 text 模块中导入 symbols、num_tones、num_languages 变量
from .text import symbols, num_tones, num_languages

# 定义 DurationDiscriminator 类，继承自 nn.Module 类
class DurationDiscriminator(nn.Module):  # vits2
    # 初始化函数，接受输入通道数、滤波器通道数、卷积核大小、丢弃概率、条件输入通道数作为参数
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        # 调用父类的初始化函数
        super().__init__()

        # 初始化类的属性
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        # 创建丢弃层，用于随机丢弃输入张量的部分元素
        self.drop = nn.Dropout(p_dropout)
        # 创建第一个卷积层，用于提取输入张量的特征
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # 创建第一个归一化层，用于对卷积层的输出进行归一化处理
        self.norm_1 = modules.LayerNorm(filter_channels)
        # 创建第二个卷积层，用于进一步提取特征
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # 创建第二个归一化层，用于对第二个卷积层的输出进行归一化处理
        self.norm_2 = modules.LayerNorm(filter_channels)
        # 创建持续时间投影层，用于将持续时间映射到特征空间
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        # 创建预输出的第一个卷积层，用于进一步提取特征
        self.pre_out_conv_1 = nn.Conv1d(
            2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # 创建预输出的第一个归一化层，用于对预输出的第一个卷积层的输出进行归一化处理
        self.pre_out_norm_1 = modules.LayerNorm(filter_channels)
        # 创建预输出的第二个卷积层，用于进一步提取特征
        self.pre_out_conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # 创建预输出的第二个归一化层，用于对预输出的第二个卷积层的输出进行归一化处理
        self.pre_out_norm_2 = modules.LayerNorm(filter_channels)

        # 如果条件输入通道数不为 0，则创建条件卷积层，用于处理条件输入
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        # 创建输出层，用于将特征映射到概率空间
        self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1), nn.Sigmoid())
    # 计算前向传播的概率，给定输入 x、输入掩码 x_mask、持续时间 dur 和条件 g（可选）
    def forward_probability(self, x, x_mask, dur, g=None):
        # 使用持续时间的投影进行处理
        dur = self.dur_proj(dur)
        # 将输入 x 和持续时间 dur 连接起来
        x = torch.cat([x, dur], dim=1)
        # 使用预处理卷积层 1 进行处理
        x = self.pre_out_conv_1(x * x_mask)
        # 使用 ReLU 激活函数
        x = torch.relu(x)
        # 使用预处理规范化层 1 进行处理
        x = self.pre_out_norm_1(x)
        # 使用 dropout 进行处理
        x = self.drop(x)
        # 使用预处理卷积层 2 进行处理
        x = self.pre_out_conv_2(x * x_mask)
        # 使用 ReLU 激活函数
        x = torch.relu(x)
        # 使用预处理规范化层 2 进行处理
        x = self.pre_out_norm_2(x)
        # 使用 dropout 进行处理
        x = self.drop(x)
        # 将输入掩码应用到 x 上
        x = x * x_mask
        # 转置 x 的维度
        x = x.transpose(1, 2)
        # 使用输出层进行处理，得到输出概率
        output_prob = self.output_layer(x)
        # 返回输出概率
        return output_prob
    
    # 前向传播函数，给定输入 x、输入掩码 x_mask、真实持续时间 dur_r、预测持续时间 dur_hat 和条件 g（可选）
    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        # 将输入 x 转换为不需要梯度的张量
        x = torch.detach(x)
        # 如果条件 g 存在
        if g is not None:
            # 将条件 g 转换为不需要梯度的张量
            g = torch.detach(g)
            # 将条件 g 作为输入 x 的条件信息
            x = x + self.cond(g)
        # 使用卷积层 1 进行处理
        x = self.conv_1(x * x_mask)
        # 使用 ReLU 激活函数
        x = torch.relu(x)
        # 使用规范化层 1 进行处理
        x = self.norm_1(x)
        # 使用 dropout 进行处理
        x = self.drop(x)
        # 使用卷积层 2 进行处理
        x = self.conv_2(x * x_mask)
        # 使用 ReLU 激活函数
        x = torch.relu(x)
        # 使用规范化层 2 进行处理
        x = self.norm_2(x)
        # 使用 dropout 进行处理
        x = self.drop(x)
    
        # 存储输出概率的列表
        output_probs = []
        # 对于真实持续时间 dur_r 和预测持续时间 dur_hat
        for dur in [dur_r, dur_hat]:
            # 计算前向传播的概率，并添加到输出概率列表中
            output_prob = self.forward_probability(x, x_mask, dur, g)
            output_probs.append(output_prob)
    
        # 返回输出概率列表
        return output_probs
# 定义一个 TransformerCouplingBlock 类，继承自 nn.Module
class TransformerCouplingBlock(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        n_flows=4,
        gin_channels=0,
        share_parameter=False,
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化类的属性
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        # 创建一个空的 nn.ModuleList 用于存储流的模块
        self.flows = nn.ModuleList()

        # 如果 share_parameter 为 True，则创建一个 FFT 模块，否则为 None
        self.wn = (
            attentions_onnx.FFT(
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                isflow=True,
                gin_channels=self.gin_channels,
            )
            if share_parameter
            else None
        )

        # 循环创建 n_flows 个 TransformerCouplingLayer 和 Flip 模块，并添加到 flows 中
        for i in range(n_flows):
            self.flows.append(
                modules.TransformerCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    n_layers,
                    n_heads,
                    p_dropout,
                    filter_channels,
                    mean_only=True,
                    wn_sharing_parameter=self.wn,
                    gin_channels=self.gin_channels,
                )
            )
            self.flows.append(modules.Flip())

    # 前向传播函数，接受输入 x, x_mask, g 和 reverse 参数
    def forward(self, x, x_mask, g=None, reverse=True):
        # 如果 reverse 为 False，则按顺序对 flows 中的模块进行前向传播
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        # 如果 reverse 为 True，则按相反顺序对 flows 中的模块进行前向传播
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        # 返回前向传播的结果 x
        return x


# 定义一个 StochasticDurationPredictor 类，继承自 nn.Module
class StochasticDurationPredictor(nn.Module):
    # 初始化函数，设置模型的参数
    def __init__(
        self,
        in_channels,
        filter_channels,
        kernel_size,
        p_dropout,
        n_flows=4,
        gin_channels=0,
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 将 filter_channels 设置为 in_channels，这行代码需要在将来的版本中移除
        filter_channels = in_channels  # it needs to be removed from future version.
        # 设置模型的各个参数
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        # 初始化 Log 模块
        self.log_flow = modules.Log()
        # 初始化流模块列表
        self.flows = nn.ModuleList()
        # 添加 ElementwiseAffine 模块到流模块列表
        self.flows.append(modules.ElementwiseAffine(2))
        # 循环创建 n_flows 个 ConvFlow 模块和 Flip 模块，添加到流模块列表
        for i in range(n_flows):
            self.flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.flows.append(modules.Flip())

        # 初始化 post_pre 模块
        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        # 初始化 post_proj 模块
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        # 初始化 post_convs 模块
        self.post_convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        # 初始化 post_flows 模块列表
        self.post_flows = nn.ModuleList()
        # 添加 ElementwiseAffine 模块到 post_flows 模块列表
        self.post_flows.append(modules.ElementwiseAffine(2))
        # 循环创建 4 个 ConvFlow 模块和 Flip 模块，添加到 post_flows 模块列表
        for i in range(4):
            self.post_flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.post_flows.append(modules.Flip())

        # 初始化 pre 模块
        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        # 初始化 proj 模块
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        # 初始化 convs 模块
        self.convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        # 如果 gin_channels 不为 0，则初始化 cond 模块
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)
    # 定义一个前向传播函数，接受输入 x, x_mask, z, g（可选）
    def forward(self, x, x_mask, z, g=None):
        # 将输入 x 转换为不需要梯度的张量
        x = torch.detach(x)
        # 将输入 x 通过预处理层
        x = self.pre(x)
        # 如果 g 不为空
        if g is not None:
            # 将 g 转换为不需要梯度的张量
            g = torch.detach(g)
            # 将输入 x 加上条件层对 g 的处理结果
            x = x + self.cond(g)
        # 将输入 x 通过卷积层，使用 x_mask 进行掩码
        x = self.convs(x, x_mask)
        # 将卷积层的输出通过投影层，并乘以 x_mask
        x = self.proj(x) * x_mask

        # 将流列表进行反转
        flows = list(reversed(self.flows))
        # 移除最后两个元素，并添加最后一个元素，去除一个无用的 vflow
        flows = flows[:-2] + [flows[-1]]
        # 遍历流列表中的每个流
        for flow in flows:
            # 对 z 进行流的逆操作，使用 x_mask 进行掩码，g 为 x，reverse 为 True
            z = flow(z, x_mask, g=x, reverse=True)
        # 将 z 按照指定维度分割成 z0 和 z1
        z0, z1 = torch.split(z, [1, 1], 1)
        # 将 z0 作为 logw 返回
        logw = z0
        return logw
# 定义一个名为 DurationPredictor 的神经网络模型类
class DurationPredictor(nn.Module):
    # 初始化函数，定义模型的参数
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()

        # 初始化模型的各个参数
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        # 初始化模型的各个层
        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        # 如果有条件输入通道，则初始化条件输入层
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    # 前向传播函数，定义模型的前向计算过程
    def forward(self, x, x_mask, g=None):
        # 对输入数据进行去梯度操作
        x = torch.detach(x)
        # 如果有条件输入，则对条件输入进行去梯度操作，并将其加到输入数据上
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        # 第一卷积层
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        # 第二卷积层
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        # 投影层
        x = self.proj(x * x_mask)
        # 返回结果
        return x * x_mask


# 定义一个名为 TextEncoder 的神经网络模型类
class TextEncoder(nn.Module):
    # 初始化函数，定义模型的参数
    def __init__(
        self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        gin_channels=0,
        # 调用父类的构造函数进行初始化
        super().__init__()
        # 初始化属性：词汇量大小
        self.n_vocab = n_vocab
        # 初始化属性：输出通道数
        self.out_channels = out_channels
        # 初始化属性：隐藏通道数
        self.hidden_channels = hidden_channels
        # 初始化属性：过滤器通道数
        self.filter_channels = filter_channels
        # 初始化属性：注意力头数
        self.n_heads = n_heads
        # 初始化属性：层数
        self.n_layers = n_layers
        # 初始化属性：卷积核大小
        self.kernel_size = kernel_size
        # 初始化属性：丢弃概率
        self.p_dropout = p_dropout
        # 初始化属性：GIN通道数
        self.gin_channels = gin_channels
        # 创建一个词嵌入层，词汇表大小为symbols的长度，隐藏通道数为hidden_channels
        self.emb = nn.Embedding(len(symbols), hidden_channels)
        # 对词嵌入层的权重进行正态分布初始化
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        # 创建一个音调嵌入层，音调数量为num_tones，隐藏通道数为hidden_channels
        self.tone_emb = nn.Embedding(num_tones, hidden_channels)
        # 对音调嵌入层的权重进行正态分布初始化
        nn.init.normal_(self.tone_emb.weight, 0.0, hidden_channels**-0.5)
        # 创建一个语言嵌入层，语言数量为num_languages，隐藏通道数为hidden_channels
        self.language_emb = nn.Embedding(num_languages, hidden_channels)
        # 对语言嵌入层的权重进行正态分布初始化
        nn.init.normal_(self.language_emb.weight, 0.0, hidden_channels**-0.5)
        # 创建一个1维卷积层，输入通道数为1024，输出通道数为hidden_channels，卷积核大小为1
        self.bert_proj = nn.Conv1d(1024, hidden_channels, 1)
        # 创建一个1维卷积层，输入通道数为1024，输出通道数为hidden_channels，卷积核大小为1
        self.ja_bert_proj = nn.Conv1d(1024, hidden_channels, 1)
        # 创建一个1维卷积层，输入通道数为1024，输出通道数为hidden_channels，卷积核大小为1
        self.en_bert_proj = nn.Conv1d(1024, hidden_channels, 1)
        # 创建一个编码器对象，参数为隐藏通道数、过滤器通道数、注意力头数、层数、卷积核大小、丢弃概率、GIN通道数
        self.encoder = attentions_onnx.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.gin_channels,
        )
        # 创建一个1维卷积层，输入通道数为hidden_channels，输出通道数为out_channels * 2，卷积核大小为1
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
    # 定义一个前向传播函数，接受输入 x、x_lengths、tone、language、bert、ja_bert、en_bert 和可选参数 g
    def forward(self, x, x_lengths, tone, language, bert, ja_bert, en_bert, g=None):
        # 创建一个与 x 相同大小的全 1 张量，并增加一个维度
        x_mask = torch.ones_like(x).unsqueeze(0)
        # 对 bert 进行投影，并转置后增加一个维度
        bert_emb = self.bert_proj(bert.transpose(0, 1).unsqueeze(0)).transpose(1, 2)
        # 对 ja_bert 进行投影，并转置后增加一个维度
        ja_bert_emb = self.ja_bert_proj(ja_bert.transpose(0, 1).unsqueeze(0)).transpose(1, 2)
        # 对 en_bert 进行投影，并转置后增加一个维度
        en_bert_emb = self.en_bert_proj(en_bert.transpose(0, 1).unsqueeze(0)).transpose(1, 2)
        # 将输入 x 经过嵌入层、tone 嵌入层、language 嵌入层、bert 嵌入层、ja_bert 嵌入层和en_bert 嵌入层的处理，并乘以一个数值
        x = (
            self.emb(x)
            + self.tone_emb(tone)
            + self.language_emb(language)
            + bert_emb
            + ja_bert_emb
            + en_bert_emb
        ) * math.sqrt(
            self.hidden_channels
        )  # [b, t, h]
        # 将 x 进行转置，交换第 1 和最后一个维度
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        # 将 x_mask 转换为与 x 相同的数据类型
        x_mask = x_mask.to(x.dtype)

        # 将 x 传入编码器进行处理，得到输出 x
        x = self.encoder(x * x_mask, x_mask, g=g)
        # 对输出 x 进行投影，得到统计数据 stats
        stats = self.proj(x) * x_mask

        # 将统计数据 stats 按照通道数分割成 m 和 logs
        m, logs = torch.split(stats, self.out_channels, dim=1)
        # 返回输出 x、m、logs 和 x_mask
        return x, m, logs, x_mask
# 定义残差耦合块的神经网络模块
class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        # 初始化残差耦合块的参数
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        # 创建流模块的列表
        self.flows = nn.ModuleList()
        # 根据 n_flows 的数量循环创建 ResidualCouplingLayer 和 Flip 模块
        for i in range(n_flows):
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
            self.flows.append(modules.Flip())

    # 定义前向传播函数
    def forward(self, x, x_mask, g=None, reverse=True):
        # 如果不是反向传播
        if not reverse:
            # 对每个流模块进行前向传播
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            # 如果是反向传播，对每个流模块进行反向传播
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        # 返回结果
        return x


# 定义后验编码器的神经网络模块
class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    # 初始化函数，设置模型的输入通道数、输出通道数、隐藏通道数、卷积核大小、膨胀率、层数和输入通道数
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels,
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 设置输入通道数
        self.in_channels = in_channels
        # 设置输出通道数
        self.out_channels = out_channels
        # 设置隐藏通道数
        self.hidden_channels = hidden_channels
        # 设置卷积核大小
        self.kernel_size = kernel_size
        # 设置膨胀率
        self.dilation_rate = dilation_rate
        # 设置层数
        self.n_layers = n_layers
        # 设置输入通道数
        self.gin_channels = gin_channels

        # 创建一个 1x1 的卷积层，将输入通道数转换为隐藏通道数
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        # 创建一个 WaveNet 模块
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        # 创建一个 1x1 的卷积层，将隐藏通道数转换为输出通道数的两倍
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    # 前向传播函数，接收输入数据 x、数据长度 x_lengths 和全局特征 g
    def forward(self, x, x_lengths, g=None):
        # 生成输入数据的掩码，保留有效数据部分
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        # 对输入数据进行预处理，并乘以掩码
        x = self.pre(x) * x_mask
        # 将预处理后的数据输入 WaveNet 模块进行编码
        x = self.enc(x, x_mask, g=g)
        # 将编码后的数据投影到输出通道数的两倍
        stats = self.proj(x) * x_mask
        # 将投影后的数据分割为均值 m 和标准差 logs
        m, logs = torch.split(stats, self.out_channels, dim=1)
        # 生成服从正态分布的随机数，乘以标准差 logs，加上均值 m，并乘以掩码
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        # 返回生成的 z、均值 m、标准差 logs 和掩码
        return z, m, logs, x_mask
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
        # 初始化生成器模型
        super(Generator, self).__init__()
        # 记录残差块的数量
        self.num_kernels = len(resblock_kernel_sizes)
        # 记录上采样的数量
        self.num_upsamples = len(upsample_rates)
        # 创建一个一维卷积层，用于预处理
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        # 根据输入的残差块类型选择不同的残差块模块
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        # 创建一个模块列表，用于存储上采样层
        self.ups = nn.ModuleList()
        # 遍历上采样率和卷积核大小，创建上采样层
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        # 创建一个模块列表，用于存储残差块
        self.resblocks = nn.ModuleList()
        # 遍历上采样层，创建对应数量的残差块
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        # 创建一个一维卷积层，用于后处理
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        # 初始化上采样层的权重
        self.ups.apply(init_weights)

        # 如果有条件输入，创建一个一维卷积层
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
    # 定义一个前向传播函数，接受输入 x 和条件 g（可选）
    def forward(self, x, g=None):
        # 对输入 x 进行预处理卷积
        x = self.conv_pre(x)
        # 如果条件 g 不为空，则将条件 g 作用于 x
        if g is not None:
            x = x + self.cond(g)

        # 循环进行上采样操作
        for i in range(self.num_upsamples):
            # 对 x 进行 LeakyReLU 激活函数处理
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            # 对 x 进行上采样操作
            x = self.ups[i](x)
            xs = None
            # 循环进行残差块操作
            for j in range(self.num_kernels):
                # 如果 xs 为空，则将当前残差块作用于 x
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                # 如果 xs 不为空，则将当前残差块的输出加到 xs 上
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            # 对 xs 进行均值处理
            x = xs / self.num_kernels
        # 对 x 进行 LeakyReLU 激活函数处理
        x = F.leaky_relu(x)
        # 对 x 进行后处理卷积
        x = self.conv_post(x)
        # 对 x 进行 tanh 激活函数处理
        x = torch.tanh(x)

        # 返回处理后的 x
        return x

    # 定义一个移除权重归一化的函数
    def remove_weight_norm(self):
        # 打印移除权重归一化的提示信息
        print("Removing weight norm...")
        # 遍历所有上采样层，移除权重归一化
        for layer in self.ups:
            remove_weight_norm(layer)
        # 遍历所有残差块，移除权重归一化
        for layer in self.resblocks:
            layer.remove_weight_norm()
class DiscriminatorP(torch.nn.Module):
    # 定义判别器类，继承自 nn.Module
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        # 初始化函数，接受周期、卷积核大小、步长和是否使用谱归一化作为参数
        super(DiscriminatorP, self).__init__()
        # 调用父类的初始化函数
        self.period = period
        # 设置周期属性
        self.use_spectral_norm = use_spectral_norm
        # 设置是否使用谱归一化属性
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        # 根据是否使用谱归一化选择相应的归一化函数
        self.convs = nn.ModuleList(
            # 定义卷积层列表
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
                # 第一层卷积
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                # 第二层卷积
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                # 第三层卷积
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                # 第四层卷积
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                # 第五层卷积
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        # 定义最后一层卷积
    # 定义一个前向传播函数，接受输入 x
    def forward(self, x):
        # 初始化特征图列表
        fmap = []

        # 将输入从1维转换为2维
        b, c, t = x.shape
        # 如果时间步 t 不能被周期 period 整除，进行填充
        if t % self.period != 0:  # pad first
            # 计算需要填充的数量
            n_pad = self.period - (t % self.period)
            # 对输入进行反射填充
            x = F.pad(x, (0, n_pad), "reflect")
            # 更新时间步数
            t = t + n_pad
        # 将输入重塑为指定形状
        x = x.view(b, c, t // self.period, self.period)

        # 遍历卷积层列表
        for layer in self.convs:
            # 对输入进行卷积操作
            x = layer(x)
            # 对卷积结果进行 LeakyReLU 激活
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            # 将卷积结果添加到特征图列表中
            fmap.append(x)
        # 对卷积后的结果进行后续卷积操作
        x = self.conv_post(x)
        # 将卷积后的结果添加到特征图列表中
        fmap.append(x)
        # 对结果进行展平操作
        x = torch.flatten(x, 1, -1)

        # 返回展平后的结果和特征图列表
        return x, fmap
class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        # 定义权重归一化函数，如果使用谱归一化则使用谱归一化函数
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        # 定义卷积层列表
        self.convs = nn.ModuleList(
            [
                # 使用权重归一化函数对输入进行卷积操作
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        # 定义后续的卷积层
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for layer in self.convs:
            # 对输入进行卷积操作
            x = layer(x)
            # 使用LeakyReLU激活函数
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        # 对卷积结果进行后续卷积操作
        x = self.conv_post(x)
        fmap.append(x)
        # 对结果进行展平操作
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            # 创建多个不同周期的DiscriminatorS对象
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        # 将所有Discriminator对象组成列表
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            # 对真实音频和生成音频分别进行判别
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    # 参考编码器模块，输入为mels，维度为[N, Ty/r, n_mels*r]
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self, spec_channels, gin_channels=0):
        super().__init__()
        self.spec_channels = spec_channels
        ref_enc_filters = [32, 32, 64, 64, 128, 128]  # 定义卷积层的滤波器数量
        K = len(ref_enc_filters)  # 获取滤波器数量
        filters = [1] + ref_enc_filters  # 构建滤波器列表
        convs = [  # 创建卷积层列表
            weight_norm(  # 对权重进行归一化
                nn.Conv2d(  # 创建二维卷积层
                    in_channels=filters[i],  # 输入通道数
                    out_channels=filters[i + 1],  # 输出通道数
                    kernel_size=(3, 3),  # 卷积核大小
                    stride=(2, 2),  # 步长
                    padding=(1, 1),  # 填充
                )
            )
            for i in range(K)  # 遍历滤波器数量
        ]
        self.convs = nn.ModuleList(convs)  # 将卷积层列表转换为模块列表
        # self.wns = nn.ModuleList([weight_norm(num_features=ref_enc_filters[i]) for i in range(K)]) # noqa: E501

        out_channels = self.calculate_channels(spec_channels, 3, 2, 1, K)  # 计算输出通道数
        self.gru = nn.GRU(  # 创建GRU层
            input_size=ref_enc_filters[-1] * out_channels,  # 输入大小
            hidden_size=256 // 2,  # 隐藏层大小
            batch_first=True,  # 批量维度是否在第一维
        )
        self.proj = nn.Linear(128, gin_channels)  # 创建线性层

    def forward(self, inputs, mask=None):
        N = inputs.size(0)  # 获取输入数据的批量大小
        out = inputs.view(N, 1, -1, self.spec_channels)  # 将输入数据重塑为四维张量
        for conv in self.convs:  # 遍历卷积层
            out = conv(out)  # 进行卷积操作
            # out = wn(out)
            out = F.relu(out)  # 使用ReLU激活函数

        out = out.transpose(1, 2)  # 转置张量维度
        T = out.size(1)  # 获取转置后的张量维度
        N = out.size(0)  # 获取转置后的张量维度
        out = out.contiguous().view(N, T, -1)  # 将张量重塑为三维张量

        self.gru.flatten_parameters()  # 展平GRU层的参数
        memory, out = self.gru(out)  # 运行GRU层
        return self.proj(out.squeeze(0))  # 返回线性层的输出

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):  # 遍历卷积层数量
            L = (L - kernel_size + 2 * pad) // stride + 1  # 计算输出大小
        return L  # 返回输出大小
class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        n_vocab,  # 词汇表大小
        spec_channels,  # 频谱通道数
        segment_size,  # 分段大小
        inter_channels,  # 中间通道数
        hidden_channels,  # 隐藏通道数
        filter_channels,  # 过滤器通道数
        n_heads,  # 多头注意力机制头数
        n_layers,  # 层数
        kernel_size,  # 卷积核大小
        p_dropout,  # 丢弃概率
        resblock,  # 是否使用残差块
        resblock_kernel_sizes,  # 残差块卷积核大小
        resblock_dilation_sizes,  # 残差块膨胀大小
        upsample_rates,  # 上采样率
        upsample_initial_channel,  # 上采样初始通道数
        upsample_kernel_sizes,  # 上采样卷积核大小
        n_speakers=256,  # 说话人数，默认为256
        gin_channels=256,  # GIN通道数，默认为256
        use_sdp=True,  # 是否使用SDP，默认为True
        n_flow_layer=4,  # 流层数，默认为4
        n_layers_trans_flow=4,  # 转换流层数，默认为4
        flow_share_parameter=False,  # 流共享参数，默认为False
        use_transformer_flow=True,  # 是否使用Transformer流，默认为True
        **kwargs,  # 其他参数
    def export_onnx(
        self,
        path,  # 导出路径
        max_len=None,  # 最大长度，默认为None
        sdp_ratio=0,  # SDP比率，默认为0
        y=None,  # 目标值，默认为None
```