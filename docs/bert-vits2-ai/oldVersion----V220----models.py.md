# `d:/src/tocomm/Bert-VITS2\oldVersion\V220\models.py`

```
import math  # 导入 math 模块，用于数学运算
import torch  # 导入 torch 模块，用于构建神经网络
from torch import nn  # 从 torch 模块中导入 nn 模块，用于构建神经网络
from torch.nn import functional as F  # 从 torch.nn 模块中导入 functional 模块，并重命名为 F，用于定义神经网络的各种功能

import commons  # 导入自定义的 commons 模块
import modules  # 导入自定义的 modules 模块
import attentions  # 导入自定义的 attentions 模块
import monotonic_align  # 导入自定义的 monotonic_align 模块

from torch.nn import Conv1d, ConvTranspose1d, Conv2d  # 从 torch.nn 模块中导入 Conv1d、ConvTranspose1d、Conv2d 类，用于定义卷积层
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm  # 从 torch.nn.utils 模块中导入 weight_norm、remove_weight_norm、spectral_norm 函数，用于对神经网络参数进行归一化处理

from commons import init_weights, get_padding  # 从 commons 模块中导入 init_weights、get_padding 函数，用于初始化神经网络参数和获取填充值
from text import symbols, num_tones, num_languages  # 从 text 模块中导入 symbols、num_tones、num_languages 变量，用于处理文本信息

from vector_quantize_pytorch import VectorQuantize  # 导入自定义的 VectorQuantize 类，用于向量量化处理


class DurationDiscriminator(nn.Module):  # 定义 DurationDiscriminator 类，继承自 nn.Module 类，用于实现持续时间鉴别器（vits2）
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 初始化各个参数
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        # 初始化一个丢弃层，用于随机丢弃输入张量中的一些元素
        self.drop = nn.Dropout(p_dropout)
        # 初始化一个一维卷积层，用于对输入进行一维卷积操作
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # 初始化一个层归一化层，用于对输入进行层归一化操作
        self.norm_1 = modules.LayerNorm(filter_channels)
        # 初始化第二个一维卷积层
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # 初始化第二个层归一化层
        self.norm_2 = modules.LayerNorm(filter_channels)
        # 创建一个卷积层，输入通道数为1，输出通道数为filter_channels，卷积核大小为1
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        # 创建一个卷积层，输入通道数为2 * filter_channels，输出通道数为filter_channels，卷积核大小为kernel_size，填充大小为kernel_size // 2
        self.pre_out_conv_1 = nn.Conv1d(
            2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # 创建一个 LayerNorm 模块，对filter_channels个特征进行归一化
        self.pre_out_norm_1 = modules.LayerNorm(filter_channels)
        # 创建一个卷积层，输入通道数为filter_channels，输出通道数为filter_channels，卷积核大小为kernel_size，填充大小为kernel_size // 2
        self.pre_out_conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # 创建一个 LayerNorm 模块，对filter_channels个特征进行归一化
        self.pre_out_norm_2 = modules.LayerNorm(filter_channels)

        # 如果 gin_channels 不为0，则创建一个卷积层，输入通道数为gin_channels，输出通道数为in_channels，卷积核大小为1
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        # 创建一个包含线性层和Sigmoid激活函数的序列模块，将filter_channels个特征映射到1个输出
        self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1), nn.Sigmoid())

    def forward_probability(self, x, x_mask, dur, g=None):
        # 将dur输入到dur_proj卷积层中进行处理
        dur = self.dur_proj(dur)
        # 将x和dur在通道维度上拼接
        x = torch.cat([x, dur], dim=1)
        # 将x乘以x_mask后输入到pre_out_conv_1卷积层中进行处理
        x = self.pre_out_conv_1(x * x_mask)
        x = torch.relu(x)  # 使用 PyTorch 的 relu 函数对输入 x 进行激活函数处理
        x = self.pre_out_norm_1(x)  # 使用预定义的神经网络层对输入 x 进行规范化处理
        x = self.drop(x)  # 使用预定义的神经网络层对输入 x 进行 dropout 处理
        x = self.pre_out_conv_2(x * x_mask)  # 使用预定义的神经网络层对输入 x 与掩码 x_mask 进行卷积处理
        x = torch.relu(x)  # 使用 PyTorch 的 relu 函数对输入 x 进行激活函数处理
        x = self.pre_out_norm_2(x)  # 使用预定义的神经网络层对输入 x 进行规范化处理
        x = self.drop(x)  # 使用预定义的神经网络层对输入 x 进行 dropout 处理
        x = x * x_mask  # 将输入 x 与掩码 x_mask 进行逐元素相乘
        x = x.transpose(1, 2)  # 对输入 x 进行维度转置操作
        output_prob = self.output_layer(x)  # 使用预定义的神经网络层对输入 x 进行输出层处理，得到输出概率
        return output_prob  # 返回输出概率作为前向传播的结果

    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        x = torch.detach(x)  # 使用 PyTorch 的 detach 函数将输入 x 从计算图中分离出来
        if g is not None:  # 如果条件 g 不为空
            g = torch.detach(g)  # 使用 PyTorch 的 detach 函数将条件 g 从计算图中分离出来
            x = x + self.cond(g)  # 将输入 x 与条件 g 经过预定义的神经网络层相加
        x = self.conv_1(x * x_mask)  # 使用预定义的神经网络层对输入 x 与掩码 x_mask 进行卷积处理
        x = torch.relu(x)  # 使用 PyTorch 的 relu 函数对输入 x 进行激活函数处理
        x = self.norm_1(x)  # 使用预定义的神经网络层对输入 x 进行规范化处理
        x = self.drop(x)  # 对输入数据进行 dropout 处理
        x = self.conv_2(x * x_mask)  # 使用卷积层对输入数据进行处理，并应用掩码
        x = torch.relu(x)  # 对处理后的数据应用 ReLU 激活函数
        x = self.norm_2(x)  # 对处理后的数据进行归一化
        x = self.drop(x)  # 再次对数据进行 dropout 处理

        output_probs = []  # 初始化一个空列表用于存储输出概率
        for dur in [dur_r, dur_hat]:  # 遍历持续时间列表
            output_prob = self.forward_probability(x, x_mask, dur, g)  # 调用 forward_probability 方法计算输出概率
            output_probs.append(output_prob)  # 将计算得到的输出概率添加到列表中

        return output_probs  # 返回输出概率列表


class TransformerCouplingBlock(nn.Module):  # 定义 TransformerCouplingBlock 类
    def __init__(  # 初始化方法，接受 channels, hidden_channels, filter_channels 等参数
        self,
        channels,
        hidden_channels,
        filter_channels,
        n_heads,  # 定义变量n_heads，表示注意力头的数量
        n_layers,  # 定义变量n_layers，表示层的数量
        kernel_size,  # 定义变量kernel_size，表示卷积核的大小
        p_dropout,  # 定义变量p_dropout，表示dropout的概率
        n_flows=4,  # 定义变量n_flows，默认值为4，表示流的数量
        gin_channels=0,  # 定义变量gin_channels，默认值为0，表示GIN的通道数
        share_parameter=False,  # 定义变量share_parameter，默认值为False，表示是否共享参数
    ):
        super().__init__()  # 调用父类的构造函数
        self.channels = channels  # 初始化变量channels
        self.hidden_channels = hidden_channels  # 初始化变量hidden_channels
        self.kernel_size = kernel_size  # 初始化变量kernel_size
        self.n_layers = n_layers  # 初始化变量n_layers
        self.n_flows = n_flows  # 初始化变量n_flows
        self.gin_channels = gin_channels  # 初始化变量gin_channels

        self.flows = nn.ModuleList()  # 初始化变量flows为一个空的ModuleList

        self.wn = (  # 初始化变量wn
            attentions.FFT(  # 调用attentions模块中的FFT函数
                hidden_channels,  # 隐藏层的通道数
                filter_channels,  # 过滤器的通道数
                n_heads,  # 注意力头的数量
                n_layers,  # 层数
                kernel_size,  # 卷积核大小
                p_dropout,  # 丢弃概率
                isflow=True,  # 是否为流模式
                gin_channels=self.gin_channels,  # GIN模型的输入通道数
            )
            if share_parameter  # 如果共享参数
            else None  # 否则为None
        )

        for i in range(n_flows):  # 循环n_flows次
            self.flows.append(  # 向flows列表中添加元素
                modules.TransformerCouplingLayer(  # 使用TransformerCouplingLayer模块
                    channels,  # 通道数
                    hidden_channels,  # 隐藏层的通道数
                    kernel_size,  # 卷积核大小
                    n_layers,  # 层数
                    n_heads,  # 定义变量 n_heads，表示注意力头的数量
                    p_dropout,  # 定义变量 p_dropout，表示丢弃概率
                    filter_channels,  # 定义变量 filter_channels，表示滤波器的通道数
                    mean_only=True,  # 定义变量 mean_only，表示是否只计算均值
                    wn_sharing_parameter=self.wn,  # 定义变量 wn_sharing_parameter，表示权重归一化共享参数
                    gin_channels=self.gin_channels,  # 定义变量 gin_channels，表示GIN模型的通道数
                )
            )
            self.flows.append(modules.Flip())  # 将 Flip 模块添加到 flows 列表中

    def forward(self, x, x_mask, g=None, reverse=False):  # 定义 forward 方法，接受输入 x、x_mask、g 和 reverse 参数
        if not reverse:  # 如果不是反向传播
            for flow in self.flows:  # 遍历 flows 列表中的每个 flow
                x, _ = flow(x, x_mask, g=g, reverse=reverse)  # 对输入 x 进行流操作
        else:  # 如果是反向传播
            for flow in reversed(self.flows):  # 遍历 flows 列表中的每个 flow（反向）
                x = flow(x, x_mask, g=g, reverse=reverse)  # 对输入 x 进行反向流操作
        return x  # 返回处理后的输入 x
class StochasticDurationPredictor(nn.Module):  # 定义一个名为StochasticDurationPredictor的类，继承自nn.Module
    def __init__(  # 初始化方法，接受参数
        self,
        in_channels,  # 输入通道数
        filter_channels,  # 过滤器通道数
        kernel_size,  # 卷积核大小
        p_dropout,  # 丢弃概率
        n_flows=4,  # 流的数量，默认为4
        gin_channels=0,  # gin通道数，默认为0
    ):
        super().__init__()  # 调用父类的初始化方法
        filter_channels = in_channels  # 这行代码需要从将来的版本中移除。将filter_channels设置为in_channels
        self.in_channels = in_channels  # 设置对象的in_channels属性为传入的in_channels
        self.filter_channels = filter_channels  # 设置对象的filter_channels属性为传入的filter_channels
        self.kernel_size = kernel_size  # 设置对象的kernel_size属性为传入的kernel_size
        self.p_dropout = p_dropout  # 设置对象的p_dropout属性为传入的p_dropout
        self.n_flows = n_flows  # 设置对象的n_flows属性为传入的n_flows
        self.gin_channels = gin_channels  # 设置对象的gin_channels属性为传入的gin_channels

        self.log_flow = modules.Log()  # 创建一个Log模块并将其赋值给对象的log_flow属性
        self.flows = nn.ModuleList()  # 创建一个空的神经网络模块列表，用于存储流模块
        self.flows.append(modules.ElementwiseAffine(2))  # 向流模块列表中添加一个ElementwiseAffine模块
        for i in range(n_flows):  # 循环n_flows次
            self.flows.append(  # 向流模块列表中添加以下模块
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)  # 添加一个ConvFlow模块
            )
            self.flows.append(modules.Flip())  # 向流模块列表中添加一个Flip模块

        self.post_pre = nn.Conv1d(1, filter_channels, 1)  # 创建一个1维卷积层，用于后处理
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)  # 创建一个1维卷积层，用于后处理
        self.post_convs = modules.DDSConv(  # 创建一个DDSConv模块，用于后处理
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        self.post_flows = nn.ModuleList()  # 创建一个空的神经网络模块列表，用于存储后处理流模块
        self.post_flows.append(modules.ElementwiseAffine(2))  # 向后处理流模块列表中添加一个ElementwiseAffine模块
        for i in range(4):  # 循环4次
            self.post_flows.append(  # 向后处理流模块列表中添加以下模块
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)  # 添加一个ConvFlow模块
            )
            self.post_flows.append(modules.Flip())  # 向后处理流模块列表中添加一个Flip模块
        self.pre = nn.Conv1d(in_channels, filter_channels, 1)  # 创建一个一维卷积层，用于对输入数据进行预处理
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)  # 创建一个一维卷积层，用于对输入数据进行投影
        self.convs = modules.DDSConv(  # 创建一个自定义的DDSConv模块，用于对输入数据进行卷积操作
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        if gin_channels != 0:  # 如果输入数据的通道数不为0
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)  # 创建一个一维卷积层，用于对条件数据进行处理

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)  # 将输入数据进行分离，使其不再与计算图关联
        x = self.pre(x)  # 对输入数据进行预处理
        if g is not None:  # 如果条件数据不为空
            g = torch.detach(g)  # 将条件数据进行分离，使其不再与计算图关联
            x = x + self.cond(g)  # 将条件数据经过卷积层处理后加到输入数据上
        x = self.convs(x, x_mask)  # 对输入数据进行卷积操作
        x = self.proj(x) * x_mask  # 对卷积后的数据进行投影并乘以掩码

        if not reverse:  # 如果不是反向操作
            flows = self.flows  # 获取流程
            assert w is not None  # 断言 w 不为空

            logdet_tot_q = 0  # 初始化 logdet_tot_q 为 0
            h_w = self.post_pre(w)  # 使用 self.post_pre 方法处理 w，得到 h_w
            h_w = self.post_convs(h_w, x_mask)  # 使用 self.post_convs 方法处理 h_w，传入 x_mask
            h_w = self.post_proj(h_w) * x_mask  # 使用 self.post_proj 方法处理 h_w，并乘以 x_mask
            e_q = (
                torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype)
                * x_mask
            )  # 生成一个符合正态分布的随机张量 e_q，大小为 (w.size(0), 2, w.size(2))，并乘以 x_mask
            z_q = e_q  # 将 e_q 赋值给 z_q
            for flow in self.post_flows:  # 遍历 self.post_flows
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))  # 使用 flow 处理 z_q，传入 x_mask 和 (x + h_w)，并得到 z_q 和 logdet_q
                logdet_tot_q += logdet_q  # 将 logdet_q 累加到 logdet_tot_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)  # 将 z_q 按照 [1, 1] 在维度 1 上分割成 z_u 和 z1
            u = torch.sigmoid(z_u) * x_mask  # 对 z_u 进行 sigmoid 操作，并乘以 x_mask
            z0 = (w - u) * x_mask  # 计算 z0，即 (w - u) 乘以 x_mask
            logdet_tot_q += torch.sum(
                (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2]
            )  # 将 (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) 乘以 x_mask 后在维度 [1, 2] 上求和，并累加到 logdet_tot_q
            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2])  # 计算logq的值，使用torch.sum对指定维度求和
                - logdet_tot_q  # 减去logdet_tot_q的值
            )

            logdet_tot = 0  # 初始化logdet_tot为0
            z0, logdet = self.log_flow(z0, x_mask)  # 调用self.log_flow方法，得到z0和logdet的值
            logdet_tot += logdet  # 将logdet加到logdet_tot上
            z = torch.cat([z0, z1], 1)  # 将z0和z1按照维度1进行拼接
            for flow in flows:  # 遍历flows列表
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)  # 调用flow方法，得到z和logdet的值
                logdet_tot = logdet_tot + logdet  # 将logdet加到logdet_tot上
            nll = (
                torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2])  # 计算nll的值，使用torch.sum对指定维度求和
                - logdet_tot  # 减去logdet_tot的值
            )
            return nll + logq  # 返回nll加上logq的值
        else:
            flows = list(reversed(self.flows))  # 将self.flows列表进行反转
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow  # 将flows列表中倒数第二个元素删除，并将最后一个元素添加进去
            z = (
                torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype)  # 生成一个与输入张量相同大小的随机张量，并将其转移到指定设备上
                * noise_scale  # 乘以噪声比例
            )
            for flow in flows:  # 遍历flows列表中的每个元素
                z = flow(z, x_mask, g=x, reverse=reverse)  # 使用flow函数对z进行变换
            z0, z1 = torch.split(z, [1, 1], 1)  # 将z张量按照指定维度进行分割
            logw = z0  # 将z0赋值给logw
            return logw  # 返回logw


class DurationPredictor(nn.Module):  # 定义一个名为DurationPredictor的类，继承自nn.Module
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):  # 初始化函数，接受输入通道数、滤波器通道数、卷积核大小、丢弃概率和gin通道数等参数
        super().__init__()  # 调用父类的初始化函数

        self.in_channels = in_channels  # 初始化输入通道数
        self.filter_channels = filter_channels  # 初始化滤波器通道数
        self.kernel_size = kernel_size  # 初始化卷积核大小
        self.p_dropout = p_dropout  # 设置类的属性 p_dropout 为传入的 p_dropout 值
        self.gin_channels = gin_channels  # 设置类的属性 gin_channels 为传入的 gin_channels 值

        self.drop = nn.Dropout(p_dropout)  # 创建一个丢弃层对象，丢弃概率为 p_dropout
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )  # 创建一个一维卷积层对象，输入通道数为 in_channels，输出通道数为 filter_channels，卷积核大小为 kernel_size，填充为 kernel_size // 2
        self.norm_1 = modules.LayerNorm(filter_channels)  # 创建一个层归一化对象，输入通道数为 filter_channels
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )  # 创建另一个一维卷积层对象，输入通道数为 filter_channels，输出通道数为 filter_channels，卷积核大小为 kernel_size，填充为 kernel_size // 2
        self.norm_2 = modules.LayerNorm(filter_channels)  # 创建另一个层归一化对象，输入通道数为 filter_channels
        self.proj = nn.Conv1d(filter_channels, 1, 1)  # 创建一个一维卷积层对象，输入通道数为 filter_channels，输出通道数为 1，卷积核大小为 1

        if gin_channels != 0:  # 如果 gin_channels 不等于 0
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)  # 创建一个一维卷积层对象，输入通道数为 gin_channels，输出通道数为 in_channels，卷积核大小为 1

    def forward(self, x, x_mask, g=None):  # 定义前向传播函数，输入参数为 x, x_mask, g
        x = torch.detach(x)  # 将输入 x 转换为不需要梯度的张量
        if g is not None:  # 如果 g 不为空
            g = torch.detach(g)  # 将张量 g 分离出来，使其不再跟踪梯度
            x = x + self.cond(g)  # 将 x 和 self.cond(g) 相加，更新 x 的值
        x = self.conv_1(x * x_mask)  # 使用 self.conv_1 对 x 乘以 x_mask 进行卷积操作
        x = torch.relu(x)  # 对 x 进行 ReLU 激活函数操作
        x = self.norm_1(x)  # 对 x 进行归一化操作
        x = self.drop(x)  # 对 x 进行 dropout 操作
        x = self.conv_2(x * x_mask)  # 使用 self.conv_2 对 x 乘以 x_mask 进行卷积操作
        x = torch.relu(x)  # 对 x 进行 ReLU 激活函数操作
        x = self.norm_2(x)  # 对 x 进行归一化操作
        x = self.drop(x)  # 对 x 进行 dropout 操作
        x = self.proj(x * x_mask)  # 使用 self.proj 对 x 乘以 x_mask 进行投影操作
        return x * x_mask  # 返回 x 乘以 x_mask 的结果


class Bottleneck(nn.Sequential):
    def __init__(self, in_dim, hidden_dim):
        c_fc1 = nn.Linear(in_dim, hidden_dim, bias=False)  # 创建一个线性层 c_fc1
        c_fc2 = nn.Linear(in_dim, hidden_dim, bias=False)  # 创建一个线性层 c_fc2
        super().__init__(*[c_fc1, c_fc2])  # 将 c_fc1 和 c_fc2 作为参数传递给父类 nn.Sequential 的初始化方法
# 定义一个名为Block的类，继承自nn.Module
class Block(nn.Module):
    # 初始化方法，接受输入维度和隐藏层维度作为参数
    def __init__(self, in_dim, hidden_dim) -> None:
        super().__init__()  # 调用父类的初始化方法
        self.norm = nn.LayerNorm(in_dim)  # 创建LayerNorm层，用于对输入进行归一化
        self.mlp = MLP(in_dim, hidden_dim)  # 创建MLP对象，用于进行多层感知机的计算

    # 前向传播方法，接受输入张量x，返回张量
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mlp(self.norm(x))  # 将输入x经过LayerNorm和MLP计算后与原始输入相加
        return x  # 返回计算结果的张量


# 定义一个名为MLP的类，继承自nn.Module
class MLP(nn.Module):
    # 初始化方法，接受输入维度和隐藏层维度作为参数
    def __init__(self, in_dim, hidden_dim):
        super().__init__()  # 调用父类的初始化方法
        self.c_fc1 = nn.Linear(in_dim, hidden_dim, bias=False)  # 创建全连接层，用于进行线性变换
        self.c_fc2 = nn.Linear(in_dim, hidden_dim, bias=False)  # 创建全连接层，用于进行线性变换
        self.c_proj = nn.Linear(hidden_dim, in_dim, bias=False)  # 创建全连接层，用于进行线性变换

    # 前向传播方法，接受输入张量x，返回张量
    def forward(self, x: torch.Tensor):
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)  # 使用激活函数SILU对输入进行处理，然后将结果与第二个全连接层的输出相乘
        x = self.c_proj(x)  # 将处理后的结果传递给c_proj进行处理
        return x  # 返回处理后的结果


class TextEncoder(nn.Module):
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
        n_speakers,
        gin_channels=0,
    ):
        super().__init__()  # 调用父类的构造函数进行初始化
        self.n_vocab = n_vocab  # 设置类的属性 n_vocab 为传入的参数 n_vocab
        self.out_channels = out_channels  # 设置类的属性 out_channels 为传入的参数 out_channels
        self.hidden_channels = hidden_channels  # 设置类的属性 hidden_channels 为传入的参数 hidden_channels
        self.filter_channels = filter_channels  # 设置类的属性 filter_channels 为传入的参数 filter_channels
        self.n_heads = n_heads  # 设置类的属性 n_heads 为传入的参数 n_heads
        self.n_layers = n_layers  # 设置类的属性 n_layers 为传入的参数 n_layers
        self.kernel_size = kernel_size  # 设置类的属性 kernel_size 为传入的参数 kernel_size
        self.p_dropout = p_dropout  # 设置类的属性 p_dropout 为传入的参数 p_dropout
        self.gin_channels = gin_channels  # 设置类的属性 gin_channels 为传入的参数 gin_channels
        self.emb = nn.Embedding(len(symbols), hidden_channels)  # 创建一个嵌入层，将 symbols 的长度作为输入维度，hidden_channels 作为输出维度
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)  # 对嵌入层的权重进行正态分布初始化
        self.tone_emb = nn.Embedding(num_tones, hidden_channels)  # 创建一个嵌入层，将 num_tones 作为输入维度，hidden_channels 作为输出维度
        nn.init.normal_(self.tone_emb.weight, 0.0, hidden_channels**-0.5)  # 对嵌入层的权重进行正态分布初始化
        self.language_emb = nn.Embedding(num_languages, hidden_channels)  # 创建一个嵌入层，将 num_languages 作为输入维度，hidden_channels 作为输出维度
        nn.init.normal_(self.language_emb.weight, 0.0, hidden_channels**-0.5)  # 对嵌入层的权重进行正态分布初始化
        self.bert_proj = nn.Conv1d(1024, hidden_channels, 1)  # 创建一个 1 维卷积层，输入通道数为 1024，输出通道数为 hidden_channels，卷积核大小为 1
        self.ja_bert_proj = nn.Conv1d(1024, hidden_channels, 1)  # 创建一个 1 维卷积层，输入通道数为 1024，输出通道数为 hidden_channels，卷积核大小为 1
        self.en_bert_proj = nn.Conv1d(1024, hidden_channels, 1)  # 创建一个 1 维卷积层，输入通道数为 1024，输出通道数为 hidden_channels，卷积核大小为 1
        # self.emo_proj = nn.Linear(512, hidden_channels)  # 创建一个全连接层，输入维度为 512，输出维度为 hidden_channels
        self.in_feature_net = nn.Sequential(  # 创建一个包含多个层的神经网络模块
            # 输入被假定为已经归一化的嵌入
            nn.Linear(512, 1028, bias=False),  # 创建一个线性层，输入维度为512，输出维度为1028，不使用偏置
            nn.GELU(),  # 使用 GELU 激活函数
            nn.LayerNorm(1028),  # 对输入进行 Layer Normalization
            *[Block(1028, 512) for _ in range(1)],  # 创建一个包含1个Block的列表，每个Block的输入维度为1028，输出维度为512
            nn.Linear(1028, 512, bias=False),  # 创建一个线性层，输入维度为1028，输出维度为512，不使用偏置
            # 在传递给VQ之前进行归一化？
            # nn.GELU(),
            # nn.LayerNorm(512),
        )
        self.emo_vq = VectorQuantize(
            dim=512,  # 嵌入的维度
            codebook_size=64,  # 代码簿的大小
            codebook_dim=32,  # 每个代码簿的维度
            commitment_weight=0.1,  # 承诺损失的权重
            decay=0.85,  # 代码簿更新的衰减率
            heads=32,  # 多头注意力机制的头数
            kmeans_iters=20,  # K均值聚类的迭代次数
            separate_codebook_per_head=True,  # 每个头使用单独的代码簿
            stochastic_sample_codes=True,  # 随机采样代码
        threshold_ema_dead_code=2,  # 设置阈值用于指示死代码的移动平均值
    )
    self.out_feature_net = nn.Linear(512, hidden_channels)  # 创建一个线性层，输入维度为512，输出维度为隐藏层维度

    self.encoder = attentions.Encoder(  # 创建一个注意力编码器
        hidden_channels,  # 隐藏层维度
        filter_channels,  # 过滤器通道数
        n_heads,  # 注意力头数
        n_layers,  # 编码器层数
        kernel_size,  # 卷积核大小
        p_dropout,  # 丢弃概率
        gin_channels=self.gin_channels,  # GIN通道数
    )
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)  # 创建一个一维卷积层，输入通道数为隐藏层维度，输出通道数为输出通道数的两倍，卷积核大小为1

def forward(
    self, x, x_lengths, tone, language, bert, ja_bert, en_bert, emo, sid, g=None
):
    sid = sid.cpu()  # 将sid移动到CPU上
    bert_emb = self.bert_proj(bert).transpose(1, 2)  # 使用bert_proj对bert进行投影，并将结果进行转置
        # 使用 self.ja_bert_proj 对日语 BERT 输出进行投影，并进行转置操作
        ja_bert_emb = self.ja_bert_proj(ja_bert).transpose(1, 2)
        # 使用 self.en_bert_proj 对英语 BERT 输出进行投影，并进行转置操作
        en_bert_emb = self.en_bert_proj(en_bert).transpose(1, 2)
        # 使用 self.in_feature_net 对情感向量进行处理
        emo_emb = self.in_feature_net(emo)
        # 将情感向量进行扩展，并通过 self.emo_vq 进行量化
        emo_emb, _, loss_commit = self.emo_vq(emo_emb.unsqueeze(1))
        # 计算量化损失的均值
        loss_commit = loss_commit.mean()
        # 使用 self.out_feature_net 对情感向量进行处理
        emo_emb = self.out_feature_net(emo_emb)
        # 对输入进行一系列的处理和加权求和操作
        x = (
            self.emb(x)
            + self.tone_emb(tone)
            + self.language_emb(language)
            + bert_emb
            + ja_bert_emb
            + en_bert_emb
            + emo_emb
        ) * math.sqrt(
            self.hidden_channels
        )  # [b, t, h]
        # 对输入进行维度转置操作
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        # 生成输入的 mask
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
        x.dtype
```
这行代码的作用是获取变量x的数据类型。

```
        )

        x = self.encoder(x * x_mask, x_mask, g=g)
```
这行代码的作用是使用self.encoder对x进行编码，并将结果赋值给变量x。

```
        stats = self.proj(x) * x_mask
```
这行代码的作用是使用self.proj对x进行投影，并将结果乘以x_mask赋值给变量stats。

```
        m, logs = torch.split(stats, self.out_channels, dim=1)
```
这行代码的作用是使用torch.split对stats进行分割，分割成m和logs两部分。

```
        return x, m, logs, x_mask, loss_commit
```
这行代码的作用是返回变量x, m, logs, x_mask, loss_commit。

```
class ResidualCouplingBlock(nn.Module):
```
这行代码的作用是定义一个名为ResidualCouplingBlock的类，继承自nn.Module。

```
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
```
这段代码是ResidualCouplingBlock类的构造函数，用于初始化类的属性。参数包括channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows, gin_channels等。
        ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置对象的通道数
        self.channels = channels
        # 设置对象的隐藏通道数
        self.hidden_channels = hidden_channels
        # 设置对象的卷积核大小
        self.kernel_size = kernel_size
        # 设置对象的膨胀率
        self.dilation_rate = dilation_rate
        # 设置对象的层数
        self.n_layers = n_layers
        # 设置对象的流数
        self.n_flows = n_flows
        # 设置对象的GIN通道数
        self.gin_channels = gin_channels

        # 创建一个空的模块列表
        self.flows = nn.ModuleList()
        # 遍历流的数量，为每个流添加ResidualCouplingLayer模块
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
    mean_only=True,  # 设置参数 mean_only 为 True
)
self.flows.append(modules.Flip())  # 将 modules.Flip() 添加到 self.flows 列表中

def forward(self, x, x_mask, g=None, reverse=False):  # 定义 forward 方法，接受参数 x, x_mask, g, reverse
    if not reverse:  # 如果 reverse 不为 True
        for flow in self.flows:  # 遍历 self.flows 列表中的元素
            x, _ = flow(x, x_mask, g=g, reverse=reverse)  # 调用 flow 方法，传入参数 x, x_mask, g, reverse
    else:  # 如果 reverse 为 True
        for flow in reversed(self.flows):  # 遍历 self.flows 列表中的元素（倒序）
            x = flow(x, x_mask, g=g, reverse=reverse)  # 调用 flow 方法，传入参数 x, x_mask, g, reverse
    return x  # 返回 x

class PosteriorEncoder(nn.Module):  # 定义 PosteriorEncoder 类，继承自 nn.Module
    def __init__(  # 定义初始化方法
        self,
        in_channels,  # 输入通道数
        out_channels,  # 输出通道数
        hidden_channels,  # 隐藏层的通道数
        kernel_size,  # 卷积核的大小
        dilation_rate,  # 膨胀率
        n_layers,  # 神经网络的层数
        gin_channels=0,  # 输入的全局信息通道数，默认为0
    ):
        super().__init__()  # 调用父类的构造函数
        self.in_channels = in_channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数
        self.hidden_channels = hidden_channels  # 隐藏层的通道数
        self.kernel_size = kernel_size  # 卷积核的大小
        self.dilation_rate = dilation_rate  # 膨胀率
        self.n_layers = n_layers  # 神经网络的层数
        self.gin_channels = gin_channels  # 输入的全局信息通道数

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)  # 创建一个1维卷积层，用于数据预处理
        self.enc = modules.WN(  # 创建一个WaveNet模块
            hidden_channels,  # 隐藏层的通道数
            kernel_size,  # 卷积核的大小
            dilation_rate,  # 膨胀率
            n_layers,  # 定义神经网络的层数
            gin_channels=gin_channels,  # 定义输入的通道数
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)  # 创建一个卷积层，用于将输入数据投影到指定维度

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype  # 根据输入数据的长度创建一个掩码，用于屏蔽无效数据
        )
        x = self.pre(x) * x_mask  # 对输入数据进行预处理，并根据掩码屏蔽无效数据
        x = self.enc(x, x_mask, g=g)  # 对预处理后的数据进行编码
        stats = self.proj(x) * x_mask  # 使用卷积层对编码后的数据进行投影，并根据掩码屏蔽无效数据
        m, logs = torch.split(stats, self.out_channels, dim=1)  # 将投影后的数据按照指定维度进行分割
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask  # 根据分割后的数据生成新的数据
        return z, m, logs, x_mask  # 返回生成的数据、均值、标准差以及掩码信息


class Generator(torch.nn.Module):
    def __init__(
        self,  # 定义生成器的初始化函数，包括参数的设置
        initial_channel,  # 定义初始通道数
        resblock,  # 定义残差块类型
        resblock_kernel_sizes,  # 定义残差块内核大小
        resblock_dilation_sizes,  # 定义残差块扩张大小
        upsample_rates,  # 定义上采样率
        upsample_initial_channel,  # 定义上采样初始通道数
        upsample_kernel_sizes,  # 定义上采样内核大小
        gin_channels=0,  # 定义GIN通道数，默认为0
    ):
        super(Generator, self).__init__()  # 调用父类的构造函数
        self.num_kernels = len(resblock_kernel_sizes)  # 计算残差块内核大小的数量
        self.num_upsamples = len(upsample_rates)  # 计算上采样率的数量
        self.conv_pre = Conv1d(  # 定义预处理卷积层
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2  # 根据resblock的值选择不同的残差块类型

        self.ups = nn.ModuleList()  # 定义上采样模块列表
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):  # 遍历上采样率和内核大小
            self.ups.append(  # 添加上采样模块到上采样模块列表
                weight_norm(  # 对卷积层进行权重归一化处理
                    ConvTranspose1d(  # 一维转置卷积层
                        upsample_initial_channel // (2**i),  # 输入通道数
                        upsample_initial_channel // (2 ** (i + 1)),  # 输出通道数
                        k,  # 卷积核大小
                        u,  # 步长
                        padding=(k - u) // 2,  # 填充
                    )
                )
            )

        self.resblocks = nn.ModuleList()  # 创建一个空的神经网络模块列表
        for i in range(len(self.ups)):  # 遍历上采样层
            ch = upsample_initial_channel // (2 ** (i + 1))  # 计算通道数
            for j, (k, d) in enumerate(  # 遍历残差块的卷积核大小和膨胀大小
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))  # 将残差块添加到神经网络模块列表中

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)  # 创建一个一维卷积层，用于后处理
        self.ups.apply(init_weights)  # 对 self.ups 中的模块应用 init_weights 函数来初始化权重

        if gin_channels != 0:  # 如果 gin_channels 不等于 0
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)  # 创建一个 1 维卷积层，输入通道数为 gin_channels，输出通道数为 upsample_initial_channel，卷积核大小为 1

    def forward(self, x, g=None):  # 定义前向传播函数，输入参数为 x 和 g（可选）
        x = self.conv_pre(x)  # 对输入 x 进行预处理卷积操作
        if g is not None:  # 如果 g 不为空
            x = x + self.cond(g)  # 将 g 经过 self.cond 卷积层处理后的结果与 x 相加

        for i in range(self.num_upsamples):  # 循环执行 num_upsamples 次
            x = F.leaky_relu(x, modules.LRELU_SLOPE)  # 对 x 进行 LeakyReLU 激活函数操作
            x = self.ups[i](x)  # 对 x 进行上采样操作
            xs = None  # 初始化 xs 为 None
            for j in range(self.num_kernels):  # 循环执行 num_kernels 次
                if xs is None:  # 如果 xs 为 None
                    xs = self.resblocks[i * self.num_kernels + j](x)  # 将 x 经过 resblocks 中的模块处理后的结果赋给 xs
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)  # 将 x 经过 resblocks 中的模块处理后的结果与 xs 相加
            x = xs / self.num_kernels  # 将 xs 除以 num_kernels 赋给 x
        x = F.leaky_relu(x)  # 使用 Leaky ReLU 激活函数处理输入 x
        x = self.conv_post(x)  # 使用 self.conv_post 对象对输入 x 进行卷积操作
        x = torch.tanh(x)  # 使用双曲正切函数处理输入 x

        return x  # 返回处理后的 x

    def remove_weight_norm(self):
        print("Removing weight norm...")  # 打印信息，表示正在移除权重归一化
        for layer in self.ups:  # 遍历 self.ups 中的层
            remove_weight_norm(layer)  # 调用 remove_weight_norm 函数移除权重归一化
        for layer in self.resblocks:  # 遍历 self.resblocks 中的层
            layer.remove_weight_norm()  # 调用 layer.remove_weight_norm 方法移除权重归一化


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()  # 调用父类的构造函数
        self.period = period  # 初始化 period 属性
        self.use_spectral_norm = use_spectral_norm  # 初始化 use_spectral_norm 属性
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm  # 根据 use_spectral_norm 的值选择 weight_norm 或 spectral_norm 赋值给 norm_f
        # 创建一个 nn.ModuleList 对象，用于存储多个卷积层
        self.convs = nn.ModuleList(
            [
                # 添加第一个卷积层，输入通道数为1，输出通道数为32，卷积核大小为(kernel_size, 1)，步长为(stride, 1)，填充为(get_padding(kernel_size, 1), 0)
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                # 添加第二个卷积层，输入通道数为32，输出通道数为128，卷积核大小为(kernel_size, 1)，步长为(stride, 1)，填充为(get_padding(kernel_size, 1), 0)
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
# 使用 norm_f 函数对 Conv2d 进行规范化处理
Conv2d(
    128,  # 输入通道数
    512,  # 输出通道数
    (kernel_size, 1),  # 卷积核大小
    (stride, 1),  # 步长
    padding=(get_padding(kernel_size, 1), 0),  # 填充
)

# 使用 norm_f 函数对 Conv2d 进行规范化处理
Conv2d(
    512,  # 输入通道数
    1024,  # 输出通道数
    (kernel_size, 1),  # 卷积核大小
    (stride, 1),  # 步长
    padding=(get_padding(kernel_size, 1), 0),  # 填充
)

# 使用 norm_f 函数对 Conv2d 进行规范化处理
Conv2d(
                        1024,  # 定义输出通道数为1024
                        1024,  # 定义输入通道数为1024
                        (kernel_size, 1),  # 定义卷积核大小为(kernel_size, 1)
                        1,  # 定义步长为1
                        padding=(get_padding(kernel_size, 1), 0),  # 定义填充大小
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))  # 定义一个卷积层

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape  # 获取输入张量的维度信息
        if t % self.period != 0:  # 如果时间维度不能整除self.period
            n_pad = self.period - (t % self.period)  # 计算需要填充的长度
            x = F.pad(x, (0, n_pad), "reflect")  # 在时间维度上进行反射填充
            t = t + n_pad  # 更新时间维度的长度
        x = x.view(b, c, t // self.period, self.period)  # 重新调整张量的形状，将其转换为指定的维度

        for layer in self.convs:  # 遍历卷积层列表
            x = layer(x)  # 对输入张量进行卷积操作
            x = F.leaky_relu(x, modules.LRELU_SLOPE)  # 使用Leaky ReLU激活函数处理张量
            fmap.append(x)  # 将处理后的张量添加到特征图列表中
        x = self.conv_post(x)  # 对输入张量进行另一次卷积操作
        fmap.append(x)  # 将处理后的张量添加到特征图列表中
        x = torch.flatten(x, 1, -1)  # 将输入张量展平为一维张量

        return x, fmap  # 返回处理后的张量和特征图列表


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):  # 初始化方法，接受一个布尔类型的参数
        super(DiscriminatorS, self).__init__()  # 调用父类的初始化方法
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm  # 根据参数选择使用哪种规范化方法
        self.convs = nn.ModuleList(  # 创建一个包含卷积层的模块列表
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),  # 使用规范化方法对输入进行一维卷积操作
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),  # 使用Conv1d进行卷积操作，输入通道数为16，输出通道数为64，卷积核大小为41，步长为4，分组数为4，填充大小为20
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),  # 使用Conv1d进行卷积操作，输入通道数为64，输出通道数为256，卷积核大小为41，步长为4，分组数为16，填充大小为20
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),  # 使用Conv1d进行卷积操作，输入通道数为256，输出通道数为1024，卷积核大小为41，步长为4，分组数为64，填充大小为20
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),  # 使用Conv1d进行卷积操作，输入通道数为1024，输出通道数为1024，卷积核大小为41，步长为4，分组数为256，填充大小为20
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),  # 使用Conv1d进行卷积操作，输入通道数为1024，输出通道数为1024，卷积核大小为5，步长为1，填充大小为2
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))  # 使用Conv1d进行卷积操作，输入通道数为1024，输出通道数为1，卷积核大小为3，步长为1，填充大小为1

    def forward(self, x):
        fmap = []

        for layer in self.convs:  # 遍历self.convs中的每个卷积层
            x = layer(x)  # 对输入x进行卷积操作
            x = F.leaky_relu(x, modules.LRELU_SLOPE)  # 使用LeakyReLU激活函数对卷积结果进行激活
            fmap.append(x)  # 将激活后的结果添加到fmap列表中
        x = self.conv_post(x)  # 对输入x进行最后一层卷积操作
        fmap.append(x)  # 将最后一层卷积的结果添加到fmap列表中
        x = torch.flatten(x, 1, -1)  # 对最后一层卷积的结果进行展平操作
        return x, fmap
```
这行代码是一个函数的返回语句，返回变量x和fmap。

```
class MultiPeriodDiscriminator(torch.nn.Module):
```
定义了一个名为MultiPeriodDiscriminator的类，继承自torch.nn.Module。

```
    def __init__(self, use_spectral_norm=False):
```
类的初始化方法，接受一个名为use_spectral_norm的布尔类型参数。

```
        super(MultiPeriodDiscriminator, self).__init__()
```
调用父类的初始化方法。

```
        periods = [2, 3, 5, 7, 11]
```
创建一个包含5个整数的列表。

```
        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
```
创建一个名为discs的列表，其中包含一个DiscriminatorS类的实例。

```
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
```
将包含DiscriminatorS实例的列表与一个包含使用periods列表中的值创建的DiscriminatorP实例的列表相加。

```
        self.discriminators = nn.ModuleList(discs)
```
创建一个包含discs列表中所有元素的ModuleList。

```
    def forward(self, y, y_hat):
```
定义了一个名为forward的方法，接受y和y_hat两个参数。

```
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
```
创建了四个空列表。

```
        for i, d in enumerate(self.discriminators):
```
对self.discriminators列表中的元素进行遍历，i为索引，d为元素。
            y_d_r, fmap_r = d(y)  # 调用函数d，将y作为参数，返回y_d_r和fmap_r
            y_d_g, fmap_g = d(y_hat)  # 调用函数d，将y_hat作为参数，返回y_d_g和fmap_g
            y_d_rs.append(y_d_r)  # 将y_d_r添加到y_d_rs列表中
            y_d_gs.append(y_d_g)  # 将y_d_g添加到y_d_gs列表中
            fmap_rs.append(fmap_r)  # 将fmap_r添加到fmap_rs列表中
            fmap_gs.append(fmap_g)  # 将fmap_g添加到fmap_gs列表中

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs  # 返回y_d_rs, y_d_gs, fmap_rs, fmap_gs四个列表


class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self, spec_channels, gin_channels=0):  # 初始化ReferenceEncoder类的实例
        super().__init__()  # 调用父类的初始化方法
        self.spec_channels = spec_channels  # 将参数spec_channels赋值给self.spec_channels
        ref_enc_filters = [32, 32, 64, 64, 128, 128]  # 定义ref_enc_filters列表并赋值
        K = len(ref_enc_filters)  # 获取ref_enc_filters列表的长度，赋值给变量K
        filters = [1] + ref_enc_filters  # 创建一个新的列表，包含1和ref_enc_filters列表的所有元素
        convs = [  # 创建一个卷积层列表
            weight_norm(  # 对卷积层进行权重归一化
                nn.Conv2d(  # 创建一个二维卷积层
                    in_channels=filters[i],  # 输入通道数为filters[i]
                    out_channels=filters[i + 1],  # 输出通道数为filters[i+1]
                    kernel_size=(3, 3),  # 卷积核大小为3x3
                    stride=(2, 2),  # 步长为2x2
                    padding=(1, 1),  # 填充为1x1
                )
            )
            for i in range(K)  # 循环K次，创建K个卷积层
        ]
        self.convs = nn.ModuleList(convs)  # 将卷积层列表转换为nn.ModuleList类型，赋值给self.convs
        # self.wns = nn.ModuleList([weight_norm(num_features=ref_enc_filters[i]) for i in range(K)]) # noqa: E501
        # 创建一个包含权重归一化的列表，赋值给self.wns

        out_channels = self.calculate_channels(spec_channels, 3, 2, 1, K)  # 调用calculate_channels方法计算输出通道数，赋值给out_channels
        self.gru = nn.GRU(  # 创建一个GRU层
            input_size=ref_enc_filters[-1] * out_channels,  # 输入大小为ref_enc_filters[-1]乘以out_channels
            hidden_size=256 // 2,  # 定义隐藏层的大小为256的一半
            batch_first=True,  # 设置输入数据的维度顺序为(batch, seq_len, feature)
        )
        self.proj = nn.Linear(128, gin_channels)  # 创建一个线性变换层，输入维度为128，输出维度为gin_channels

    def forward(self, inputs, mask=None):
        N = inputs.size(0)  # 获取输入数据的batch大小
        out = inputs.view(N, 1, -1, self.spec_channels)  # 将输入数据重塑为指定形状 [N, 1, Ty, n_freqs]
        for conv in self.convs:  # 遍历卷积层列表
            out = conv(out)  # 对输入数据进行卷积操作
            # out = wn(out)
            out = F.relu(out)  # 对卷积后的数据进行ReLU激活函数操作，得到非线性输出 [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # 调换张量维度顺序 [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)  # 获取调换维度后的张量的第二个维度大小
        N = out.size(0)  # 获取调换维度后的张量的第一个维度大小
        out = out.contiguous().view(N, T, -1)  # 将张量重塑为指定形状 [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()  # 将GRU层的参数展平
        memory, out = self.gru(out)  # 对输入数据进行GRU操作，得到输出out和内部记忆memory
        return self.proj(out.squeeze(0))
```
这行代码的作用是将out中的多余维度去除，并将结果返回给self.proj。

```python
    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L
```
这个函数的作用是根据给定的参数计算输出的通道数，并返回结果。

```python
class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """
```
这是一个类的定义，用于创建一个名为SynthesizerTrn的类，用于训练合成器。

```python
    def __init__(
        self,
        n_vocab,
        spec_channels,
        segment_size,
        inter_channels,
```
这是SynthesizerTrn类的初始化函数，它接受n_vocab、spec_channels、segment_size和inter_channels作为参数。
        hidden_channels,  # 隐藏层的通道数
        filter_channels,  # 过滤器的通道数
        n_heads,  # 注意力头的数量
        n_layers,  # 层数
        kernel_size,  # 卷积核大小
        p_dropout,  # 丢弃概率
        resblock,  # 是否使用残差块
        resblock_kernel_sizes,  # 残差块的卷积核大小
        resblock_dilation_sizes,  # 残差块的扩张大小
        upsample_rates,  # 上采样率
        upsample_initial_channel,  # 上采样初始通道数
        upsample_kernel_sizes,  # 上采样的卷积核大小
        n_speakers=256,  # 说话者数量，默认为256
        gin_channels=256,  # GIN模块的通道数，默认为256
        use_sdp=True,  # 是否使用SDP，默认为True
        n_flow_layer=4,  # 流层的数量，默认为4
        n_layers_trans_flow=4,  # 转换流的层数，默认为4
        flow_share_parameter=False,  # 流共享参数，默认为False
        use_transformer_flow=True,  # 是否使用Transformer流，默认为True
        **kwargs  # 其他参数
        ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置词汇表大小
        self.n_vocab = n_vocab
        # 设置特殊通道数
        self.spec_channels = spec_channels
        # 设置内部通道数
        self.inter_channels = inter_channels
        # 设置隐藏通道数
        self.hidden_channels = hidden_channels
        # 设置过滤器通道数
        self.filter_channels = filter_channels
        # 设置注意力头数
        self.n_heads = n_heads
        # 设置层数
        self.n_layers = n_layers
        # 设置卷积核大小
        self.kernel_size = kernel_size
        # 设置丢失率
        self.p_dropout = p_dropout
        # 设置是否使用残差块
        self.resblock = resblock
        # 设置残差块的卷积核大小
        self.resblock_kernel_sizes = resblock_kernel_sizes
        # 设置残差块的扩张大小
        self.resblock_dilation_sizes = resblock_dilation_sizes
        # 设置上采样率
        self.upsample_rates = upsample_rates
        # 设置上采样初始通道数
        self.upsample_initial_channel = upsample_initial_channel
        # 设置上采样卷积核大小
        self.upsample_kernel_sizes = upsample_kernel_sizes
        # 设置段大小
        self.segment_size = segment_size
        # 设置说话者数量
        self.n_speakers = n_speakers
        # 设置GIN通道数
        self.gin_channels = gin_channels
        # 设置属性 self.n_layers_trans_flow 为传入的 n_layers_trans_flow
        self.n_layers_trans_flow = n_layers_trans_flow
        # 设置属性 self.use_spk_conditioned_encoder 为传入参数中的 use_spk_conditioned_encoder，如果参数中没有则默认为 True
        self.use_spk_conditioned_encoder = kwargs.get("use_spk_conditioned_encoder", True)
        # 设置属性 self.use_sdp 为传入的 use_sdp
        self.use_sdp = use_sdp
        # 设置属性 self.use_noise_scaled_mas 为传入参数中的 use_noise_scaled_mas，如果参数中没有则默认为 False
        self.use_noise_scaled_mas = kwargs.get("use_noise_scaled_mas", False)
        # 设置属性 self.mas_noise_scale_initial 为传入参数中的 mas_noise_scale_initial，如果参数中没有则默认为 0.01
        self.mas_noise_scale_initial = kwargs.get("mas_noise_scale_initial", 0.01)
        # 设置属性 self.noise_scale_delta 为传入参数中的 noise_scale_delta，如果参数中没有则默认为 2e-6
        self.noise_scale_delta = kwargs.get("noise_scale_delta", 2e-6)
        # 设置属性 self.current_mas_noise_scale 为 self.mas_noise_scale_initial
        self.current_mas_noise_scale = self.mas_noise_scale_initial
        # 如果使用了 spk_conditioned_encoder 并且 gin_channels 大于 0，则设置属性 self.enc_gin_channels 为传入的 gin_channels
        if self.use_spk_conditioned_encoder and gin_channels > 0:
            self.enc_gin_channels = gin_channels
        # 创建 TextEncoder 对象并设置为属性 self.enc_p
        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        self.n_speakers,  # 初始化属性 n_speakers
        gin_channels=self.enc_gin_channels,  # 初始化属性 gin_channels，并赋值为 enc_gin_channels
    )
    self.dec = Generator(  # 初始化属性 dec，并赋值为 Generator 类的实例化对象
        inter_channels,  # 参数1：中间通道数
        resblock,  # 参数2：残差块
        resblock_kernel_sizes,  # 参数3：残差块的卷积核大小
        resblock_dilation_sizes,  # 参数4：残差块的膨胀大小
        upsample_rates,  # 参数5：上采样率
        upsample_initial_channel,  # 参数6：上采样初始通道数
        upsample_kernel_sizes,  # 参数7：上采样的卷积核大小
        gin_channels=gin_channels,  # 参数8：gin_channels 属性的值
    )
    self.enc_q = PosteriorEncoder(  # 初始化属性 enc_q，并赋值为 PosteriorEncoder 类的实例化对象
        spec_channels,  # 参数1：频谱通道数
        inter_channels,  # 参数2：中间通道数
        hidden_channels,  # 参数3：隐藏通道数
        5,  # 参数4：固定值 5
        1,  # 参数5：固定值 1
        16,  # 参数6：固定值 16
        gin_channels=gin_channels,  # 传递参数 gin_channels 给 TransformerCouplingBlock 或 ResidualCouplingBlock
    )
    if use_transformer_flow:  # 如果 use_transformer_flow 为真
        self.flow = TransformerCouplingBlock(  # 创建一个 TransformerCouplingBlock 对象
            inter_channels,  # 传递参数 inter_channels 给 TransformerCouplingBlock
            hidden_channels,  # 传递参数 hidden_channels 给 TransformerCouplingBlock
            filter_channels,  # 传递参数 filter_channels 给 TransformerCouplingBlock
            n_heads,  # 传递参数 n_heads 给 TransformerCouplingBlock
            n_layers_trans_flow,  # 传递参数 n_layers_trans_flow 给 TransformerCouplingBlock
            5,  # 传递参数 5 给 TransformerCouplingBlock
            p_dropout,  # 传递参数 p_dropout 给 TransformerCouplingBlock
            n_flow_layer,  # 传递参数 n_flow_layer 给 TransformerCouplingBlock
            gin_channels=gin_channels,  # 传递参数 gin_channels 给 TransformerCouplingBlock
            share_parameter=flow_share_parameter,  # 传递参数 flow_share_parameter 给 TransformerCouplingBlock
        )
    else:  # 如果 use_transformer_flow 为假
        self.flow = ResidualCouplingBlock(  # 创建一个 ResidualCouplingBlock 对象
            inter_channels,  # 传递参数 inter_channels 给 ResidualCouplingBlock
            hidden_channels,  # 传递参数 hidden_channels 给 ResidualCouplingBlock
            5,  # 传递参数 5 给 ResidualCouplingBlock
                1,  # 设置参数1
                n_flow_layer,  # 设置参数n_flow_layer
                gin_channels=gin_channels,  # 设置参数gin_channels为默认值
            )
        self.sdp = StochasticDurationPredictor(  # 初始化StochasticDurationPredictor对象
            hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels  # 设置StochasticDurationPredictor对象的参数
        )
        self.dp = DurationPredictor(  # 初始化DurationPredictor对象
            hidden_channels, 256, 3, 0.5, gin_channels=gin_channels  # 设置DurationPredictor对象的参数
        )

        if n_speakers >= 1:  # 如果说话者数量大于等于1
            self.emb_g = nn.Embedding(n_speakers, gin_channels)  # 初始化nn.Embedding对象
        else:  # 否则
            self.ref_enc = ReferenceEncoder(spec_channels, gin_channels)  # 初始化ReferenceEncoder对象

    def forward(  # 定义前向传播函数
        self,
        x,  # 输入数据x
        x_lengths,  # 输入数据x的长度
        y,  # 输入的目标语音特征
        y_lengths,  # 目标语音特征的长度
        sid,  # 说话人的标识
        tone,  # 音调信息
        language,  # 语言信息
        bert,  # BERT 编码
        ja_bert,  # 日语的 BERT 编码
        en_bert,  # 英语的 BERT 编码
        emo=None,  # 情感信息，默认为 None
    ):
        if self.n_speakers > 0:  # 如果存在多个说话人
            g = self.emb_g(sid).unsqueeze(-1)  # 通过说话人标识获取对应的说话人嵌入，并在最后添加一个维度
        else:  # 如果只有单个说话人
            g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)  # 通过目标语音特征获取参考编码，并在最后添加一个维度
        x, m_p, logs_p, x_mask, loss_commit = self.enc_p(
            x, x_lengths, tone, language, bert, ja_bert, en_bert, emo, sid, g=g
        )  # 使用编码器 P 对输入进行编码
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)  # 使用编码器 Q 对目标语音特征进行编码
        z_p = self.flow(z, y_mask, g=g)  # 使用流模型对编码后的目标语音特征进行处理
        with torch.no_grad():  # 使用 torch.no_grad() 上下文管理器，确保在此范围内的操作不会被记录在计算图中，不会进行梯度计算
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # 计算 logs_p 的负指数，得到 s_p_sq_r，用于后续计算
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )  # 计算负交叉熵的第一部分，对 logs_p 进行操作
            neg_cent2 = torch.matmul(
                -0.5 * (z_p**2).transpose(1, 2), s_p_sq_r
            )  # 计算负交叉熵的第二部分，进行矩阵乘法操作
            neg_cent3 = torch.matmul(
                z_p.transpose(1, 2), (m_p * s_p_sq_r)
            )  # 计算负交叉熵的第三部分，进行矩阵乘法操作
            neg_cent4 = torch.sum(
                -0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True
            )  # 计算负交叉熵的第四部分，对 m_p 和 s_p_sq_r 进行操作
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4  # 将四部分结果相加得到最终的负交叉熵
            if self.use_noise_scaled_mas:  # 如果使用了噪声缩放的 MAS
                epsilon = (
                    torch.std(neg_cent)  # 计算 neg_cent 的标准差
                    * torch.randn_like(neg_cent)  # 生成与 neg_cent 相同形状的随机数，乘以标准差得到 epsilon
            # 计算注意力掩码，将输入的掩码扩展为三维张量
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            # 计算注意力权重，使用负中心化的注意力分布和注意力掩码
            attn = (
                monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )

        # 计算注意力权重的总和
        w = attn.sum(2)

        # 使用注意力权重计算长度加权的自注意力
        l_length_sdp = self.sdp(x, x_mask, w, g=g)
        # 将长度加权的自注意力除以输入掩码的总和
        l_length_sdp = l_length_sdp / torch.sum(x_mask)

        # 计算注意力权重的对数
        logw_ = torch.log(w + 1e-6) * x_mask
        # 使用差分隐私机制处理注意力权重的对数
        logw = self.dp(x, x_mask, g=g)
        # 计算差分隐私机制处理后的注意力权重对数的损失
        l_length_dp = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(
            x_mask
        )  # for averaging  # 用于求平均

        l_length = l_length_dp + l_length_sdp  # 计算 l_length，将 l_length_dp 和 l_length_sdp 相加

        # expand prior  # 扩展先验
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)  # 计算 m_p
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)  # 计算 logs_p

        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )  # 通过 commons.rand_slice_segments 函数对 z 进行随机切片，得到 z_slice 和 ids_slice
        o = self.dec(z_slice, g=g)  # 使用 z_slice 和 g 作为参数调用 self.dec 函数，得到 o
        return (
            o,
            l_length,
            attn,
            ids_slice,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),  # 返回一个包含多个元素的元组
    def infer(  # 定义一个名为infer的方法，用于推断
        self,  # 方法的第一个参数是self，表示类的实例对象
        x,  # 输入参数x，用于推断
        x_lengths,  # 输入参数x_lengths，表示x的长度
        sid,  # 输入参数sid，表示sid
        tone,  # 输入参数tone，表示音调
        language,  # 输入参数language，表示语言
        bert,  # 输入参数bert，表示bert
        ja_bert,  # 输入参数ja_bert，表示日语的bert
        en_bert,  # 输入参数en_bert，表示英语的bert
        emo=None,  # 输入参数emo，默认值为None，表示情绪
        noise_scale=0.667,  # 输入参数noise_scale，默认值为0.667，表示噪音比例
        length_scale=1,  # 输入参数length_scale，默认值为1，表示长度比例
        noise_scale_w=0.8,  # 输入参数noise_scale_w，默认值为0.8，表示噪音比例w
        max_len=None,  # 输入参数max_len，默认值为None，表示最大长度
        sdp_ratio=0,  # 设置默认的 sdp_ratio 为 0
        y=None,  # 初始化 y 为 None

    ):
        # x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, tone, language, bert)  # 调用 self.enc_p 方法，传入参数并获取返回值
        # g = self.gst(y)  # 调用 self.gst 方法，传入参数 y 并获取返回值
        if self.n_speakers > 0:  # 如果 self.n_speakers 大于 0
            g = self.emb_g(sid).unsqueeze(-1)  # 调用 self.emb_g 方法，传入参数 sid 并获取返回值，然后在最后一个维度上增加一个维度
        else:  # 否则
            g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)  # 调用 self.ref_enc 方法，传入参数 y 的转置，并在最后一个维度上增加一个维度
        x, m_p, logs_p, x_mask, _ = self.enc_p(  # 调用 self.enc_p 方法，传入多个参数并获取返回值
            x, x_lengths, tone, language, bert, ja_bert, en_bert, emo, sid, g=g
        )
        logw = self.sdp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w) * (  # 调用 self.sdp 方法，传入多个参数并进行计算
            sdp_ratio  # 乘以 sdp_ratio
        ) + self.dp(x, x_mask, g=g) * (1 - sdp_ratio)  # 调用 self.dp 方法，传入多个参数并进行计算
        w = torch.exp(logw) * x_mask * length_scale  # 计算 w
        w_ceil = torch.ceil(w)  # 对 w 进行向上取整
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()  # 对 w_ceil 按指定维度求和并进行限制最小值为 1，然后转换为整型
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(  # 调用 commons.sequence_mask 方法，传入参数并获取返回值，然后在第二个维度上增加一个维度，并转换为指定类型
            x_mask.dtype
        )
        # 创建注意力掩码，将输入和输出的掩码相乘得到注意力掩码
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        # 使用生成路径函数生成路径
        attn = commons.generate_path(w_ceil, attn_mask)

        # 计算新的位置编码
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        # 计算新的位置编码的对数
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        # 添加噪声并进行缩放
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        # 使用流函数进行反向传播
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        # 使用解码器进行解码
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        # 返回解码结果、注意力、输出掩码和位置编码相关信息
        return o, attn, y_mask, (z, z_p, m_p, logs_p)
```