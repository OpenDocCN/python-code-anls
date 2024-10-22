# `.\diffusers\models\unets\unet_1d_blocks.py`

```py
# 版权声明，指定版权所有者及其保留权利
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证，版本 2.0（“许可证”）进行许可；
# 除非遵守许可证，否则不得使用此文件。
# 可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面同意，否则根据许可证分发的软件是按“现状”基础提供的，
# 不提供任何形式的明示或暗示的担保或条件。
# 请参阅许可证以获取管理权限和限制的具体条款。
import math  # 导入数学库，以便使用数学函数
from typing import Optional, Tuple, Union  # 导入类型注解工具

import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 的函数式神经网络接口
from torch import nn  # 从 PyTorch 导入神经网络模块

from ..activations import get_activation  # 导入自定义激活函数获取工具
from ..resnet import Downsample1D, ResidualTemporalBlock1D, Upsample1D, rearrange_dims  # 导入自定义的 ResNet 组件


class DownResnetBlock1D(nn.Module):  # 定义一个一维下采样的 ResNet 模块
    def __init__(  # 初始化方法，定义模块的参数
        self,
        in_channels: int,  # 输入通道数
        out_channels: Optional[int] = None,  # 输出通道数（可选）
        num_layers: int = 1,  # 残差层数，默认为1
        conv_shortcut: bool = False,  # 是否使用卷积快捷连接
        temb_channels: int = 32,  # 时间嵌入通道数
        groups: int = 32,  # 组数
        groups_out: Optional[int] = None,  # 输出组数（可选）
        non_linearity: Optional[str] = None,  # 非线性激活函数（可选）
        time_embedding_norm: str = "default",  # 时间嵌入的归一化方式
        output_scale_factor: float = 1.0,  # 输出缩放因子
        add_downsample: bool = True,  # 是否添加下采样层
    ):
        super().__init__()  # 调用父类初始化方法
        self.in_channels = in_channels  # 设置输入通道数
        out_channels = in_channels if out_channels is None else out_channels  # 如果未指定输出通道数，则设置为输入通道数
        self.out_channels = out_channels  # 设置输出通道数
        self.use_conv_shortcut = conv_shortcut  # 保存是否使用卷积快捷连接的标志
        self.time_embedding_norm = time_embedding_norm  # 设置时间嵌入的归一化方式
        self.add_downsample = add_downsample  # 保存是否添加下采样层的标志
        self.output_scale_factor = output_scale_factor  # 设置输出缩放因子

        if groups_out is None:  # 如果未指定输出组数
            groups_out = groups  # 设置输出组数为输入组数

        # 始终至少有一个残差块
        resnets = [ResidualTemporalBlock1D(in_channels, out_channels, embed_dim=temb_channels)]  # 创建第一个残差块

        for _ in range(num_layers):  # 根据指定的层数添加残差块
            resnets.append(ResidualTemporalBlock1D(out_channels, out_channels, embed_dim=temb_channels))  # 添加后续的残差块

        self.resnets = nn.ModuleList(resnets)  # 将残差块列表转换为 PyTorch 模块列表

        if non_linearity is None:  # 如果未指定非线性激活函数
            self.nonlinearity = None  # 设置为 None
        else:
            self.nonlinearity = get_activation(non_linearity)  # 获取指定的激活函数

        self.downsample = None  # 初始化下采样层为 None
        if add_downsample:  # 如果需要添加下采样层
            self.downsample = Downsample1D(out_channels, use_conv=True, padding=1)  # 创建下采样层
    # 定义前向传播函数，接收隐藏状态和可选的时间嵌入，返回处理后的隐藏状态和输出状态
    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 初始化一个空元组用于存储输出状态
        output_states = ()
    
        # 使用第一个残差网络处理输入的隐藏状态和时间嵌入
        hidden_states = self.resnets[0](hidden_states, temb)
        # 遍历后续的残差网络，逐个处理隐藏状态
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states, temb)
    
        # 将当前的隐藏状态添加到输出状态元组中
        output_states += (hidden_states,)
    
        # 如果非线性激活函数存在，则应用于隐藏状态
        if self.nonlinearity is not None:
            hidden_states = self.nonlinearity(hidden_states)
    
        # 如果下采样层存在，则对隐藏状态进行下采样处理
        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)
    
        # 返回处理后的隐藏状态和输出状态
        return hidden_states, output_states
# 定义一个一维的上采样残差块类，继承自 nn.Module
class UpResnetBlock1D(nn.Module):
    # 初始化方法，定义输入输出通道、层数等参数
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        out_channels: Optional[int] = None,  # 输出通道数，默认为 None
        num_layers: int = 1,  # 残差层数，默认为 1
        temb_channels: int = 32,  # 时间嵌入通道数
        groups: int = 32,  # 分组数
        groups_out: Optional[int] = None,  # 输出分组数，默认为 None
        non_linearity: Optional[str] = None,  # 非线性激活函数，默认为 None
        time_embedding_norm: str = "default",  # 时间嵌入归一化方式，默认为 "default"
        output_scale_factor: float = 1.0,  # 输出缩放因子
        add_upsample: bool = True,  # 是否添加上采样层，默认为 True
    ):
        # 调用父类初始化方法
        super().__init__()
        self.in_channels = in_channels  # 保存输入通道数
        # 如果输出通道数为 None，则设置为输入通道数
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels  # 保存输出通道数
        self.time_embedding_norm = time_embedding_norm  # 保存时间嵌入归一化方式
        self.add_upsample = add_upsample  # 保存是否添加上采样层
        self.output_scale_factor = output_scale_factor  # 保存输出缩放因子

        # 如果输出分组数为 None，则设置为输入分组数
        if groups_out is None:
            groups_out = groups

        # 初始化至少一个残差块
        resnets = [ResidualTemporalBlock1D(2 * in_channels, out_channels, embed_dim=temb_channels)]

        # 根据层数添加残差块
        for _ in range(num_layers):
            resnets.append(ResidualTemporalBlock1D(out_channels, out_channels, embed_dim=temb_channels))

        # 将残差块列表转换为 nn.ModuleList
        self.resnets = nn.ModuleList(resnets)

        # 根据非线性激活函数的设置，初始化激活函数
        if non_linearity is None:
            self.nonlinearity = None
        else:
            self.nonlinearity = get_activation(non_linearity)

        # 初始化上采样层为 None
        self.upsample = None
        # 如果需要添加上采样层，则初始化它
        if add_upsample:
            self.upsample = Upsample1D(out_channels, use_conv_transpose=True)

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态
        res_hidden_states_tuple: Optional[Tuple[torch.Tensor, ...]] = None,  # 残差隐藏状态元组，默认为 None
        temb: Optional[torch.Tensor] = None,  # 时间嵌入，默认为 None
    ) -> torch.Tensor:
        # 如果有残差隐藏状态，则将其与当前隐藏状态拼接
        if res_hidden_states_tuple is not None:
            res_hidden_states = res_hidden_states_tuple[-1]  # 取最后一个残差状态
            hidden_states = torch.cat((hidden_states, res_hidden_states), dim=1)  # 拼接操作

        # 通过第一个残差块处理隐藏状态
        hidden_states = self.resnets[0](hidden_states, temb)
        # 依次通过后续的残差块处理隐藏状态
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states, temb)

        # 如果有非线性激活函数，则应用它
        if self.nonlinearity is not None:
            hidden_states = self.nonlinearity(hidden_states)

        # 如果有上采样层，则应用它
        if self.upsample is not None:
            hidden_states = self.upsample(hidden_states)

        # 返回最终的隐藏状态
        return hidden_states


# 定义一个值函数中间块类，继承自 nn.Module
class ValueFunctionMidBlock1D(nn.Module):
    # 初始化方法，定义输入输出通道和嵌入维度
    def __init__(self, in_channels: int, out_channels: int, embed_dim: int):
        # 调用父类初始化方法
        super().__init__()
        self.in_channels = in_channels  # 保存输入通道数
        self.out_channels = out_channels  # 保存输出通道数
        self.embed_dim = embed_dim  # 保存嵌入维度

        # 初始化第一个残差块
        self.res1 = ResidualTemporalBlock1D(in_channels, in_channels // 2, embed_dim=embed_dim)
        # 初始化第一个下采样层
        self.down1 = Downsample1D(out_channels // 2, use_conv=True)
        # 初始化第二个残差块
        self.res2 = ResidualTemporalBlock1D(in_channels // 2, in_channels // 4, embed_dim=embed_dim)
        # 初始化第二个下采样层
        self.down2 = Downsample1D(out_channels // 4, use_conv=True)
    # 定义前向传播函数，接受输入张量和可选的嵌入张量，返回输出张量
        def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
            # 将输入张量 x 通过第一个残差块处理，可能使用嵌入张量 temb
            x = self.res1(x, temb)
            # 将处理后的张量 x 通过第一个下采样层进行下采样
            x = self.down1(x)
            # 将下采样后的张量 x 通过第二个残差块处理，可能使用嵌入张量 temb
            x = self.res2(x, temb)
            # 将处理后的张量 x 通过第二个下采样层进行下采样
            x = self.down2(x)
            # 返回最终处理后的张量 x
            return x
# 定义一个中间分辨率的时间块类，继承自 nn.Module
class MidResTemporalBlock1D(nn.Module):
    # 初始化方法，定义该类的参数
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        out_channels: int,  # 输出通道数
        embed_dim: int,  # 嵌入维度
        num_layers: int = 1,  # 层数，默认值为 1
        add_downsample: bool = False,  # 是否添加下采样，默认为 False
        add_upsample: bool = False,  # 是否添加上采样，默认为 False
        non_linearity: Optional[str] = None,  # 非线性激活函数的类型，默认为 None
    ):
        # 调用父类构造函数
        super().__init__()
        # 设置输入通道数
        self.in_channels = in_channels
        # 设置输出通道数
        self.out_channels = out_channels
        # 设置是否添加下采样
        self.add_downsample = add_downsample

        # 至少会有一个残差网络
        resnets = [ResidualTemporalBlock1D(in_channels, out_channels, embed_dim=embed_dim)]

        # 根据层数添加残差网络层
        for _ in range(num_layers):
            resnets.append(ResidualTemporalBlock1D(out_channels, out_channels, embed_dim=embed_dim))

        # 将残差网络层列表转换为 nn.ModuleList
        self.resnets = nn.ModuleList(resnets)

        # 根据是否提供非线性激活函数初始化相应属性
        if non_linearity is None:
            self.nonlinearity = None
        else:
            self.nonlinearity = get_activation(non_linearity)

        # 初始化上采样层为 None
        self.upsample = None
        # 如果添加上采样，则创建上采样层
        if add_upsample:
            self.upsample = Upsample1D(out_channels, use_conv=True)

        # 初始化下采样层为 None
        self.downsample = None
        # 如果添加下采样，则创建下采样层
        if add_downsample:
            self.downsample = Downsample1D(out_channels, use_conv=True)

        # 如果同时添加了上采样和下采样，抛出错误
        if self.upsample and self.downsample:
            raise ValueError("Block cannot downsample and upsample")

    # 定义前向传播方法
    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        # 通过第一个残差网络处理隐藏状态
        hidden_states = self.resnets[0](hidden_states, temb)
        # 遍历其余的残差网络进行处理
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states, temb)

        # 如果有上采样层，则执行上采样
        if self.upsample:
            hidden_states = self.upsample(hidden_states)
        # 如果有下采样层，则执行下采样
        if self.downsample:
            self.downsample = self.downsample(hidden_states)

        # 返回处理后的隐藏状态
        return hidden_states


# 定义输出卷积块类，继承自 nn.Module
class OutConv1DBlock(nn.Module):
    # 初始化方法，定义该类的参数
    def __init__(self, num_groups_out: int, out_channels: int, embed_dim: int, act_fn: str):
        # 调用父类构造函数
        super().__init__()
        # 创建第一层 1D 卷积，kernel_size 为 5，padding 为 2
        self.final_conv1d_1 = nn.Conv1d(embed_dim, embed_dim, 5, padding=2)
        # 创建 GroupNorm 层，指定组数和嵌入维度
        self.final_conv1d_gn = nn.GroupNorm(num_groups_out, embed_dim)
        # 根据激活函数名称获取激活函数
        self.final_conv1d_act = get_activation(act_fn)
        # 创建第二层 1D 卷积，kernel_size 为 1
        self.final_conv1d_2 = nn.Conv1d(embed_dim, out_channels, 1)

    # 定义前向传播方法
    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 通过第一层卷积处理隐藏状态
        hidden_states = self.final_conv1d_1(hidden_states)
        # 调整维度
        hidden_states = rearrange_dims(hidden_states)
        # 通过 GroupNorm 层处理
        hidden_states = self.final_conv1d_gn(hidden_states)
        # 再次调整维度
        hidden_states = rearrange_dims(hidden_states)
        # 通过激活函数处理
        hidden_states = self.final_conv1d_act(hidden_states)
        # 通过第二层卷积处理
        hidden_states = self.final_conv1d_2(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states


# 定义输出值函数块类，继承自 nn.Module
class OutValueFunctionBlock(nn.Module):
    # 初始化方法，设置全连接层维度和激活函数
        def __init__(self, fc_dim: int, embed_dim: int, act_fn: str = "mish"):
            # 调用父类的初始化方法
            super().__init__()
            # 创建一个模块列表，包含线性层和激活函数
            self.final_block = nn.ModuleList(
                [
                    # 第一个线性层，将输入维度从 fc_dim + embed_dim 转换到 fc_dim // 2
                    nn.Linear(fc_dim + embed_dim, fc_dim // 2),
                    # 获取指定的激活函数
                    get_activation(act_fn),
                    # 第二个线性层，将输入维度从 fc_dim // 2 转换到 1
                    nn.Linear(fc_dim // 2, 1),
                ]
            )
    
        # 前向传播方法，接受隐藏状态和额外的嵌入信息
        def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
            # 重塑隐藏状态，使其成为二维张量
            hidden_states = hidden_states.view(hidden_states.shape[0], -1)
            # 将重塑后的隐藏状态与额外的嵌入信息在最后一个维度上连接
            hidden_states = torch.cat((hidden_states, temb), dim=-1)
            # 遍历 final_block 中的每一层，逐层处理隐藏状态
            for layer in self.final_block:
                hidden_states = layer(hidden_states)
    
            # 返回最终的隐藏状态
            return hidden_states
# 定义包含不同内核函数系数的字典
_kernels = {
    # 线性内核的系数
    "linear": [1 / 8, 3 / 8, 3 / 8, 1 / 8],
    # 三次插值内核的系数
    "cubic": [-0.01171875, -0.03515625, 0.11328125, 0.43359375, 0.43359375, 0.11328125, -0.03515625, -0.01171875],
    # Lanczos3 内核的系数
    "lanczos3": [
        0.003689131001010537,
        0.015056144446134567,
        -0.03399861603975296,
        -0.066637322306633,
        0.13550527393817902,
        0.44638532400131226,
        0.44638532400131226,
        0.13550527393817902,
        -0.066637322306633,
        -0.03399861603975296,
        0.015056144446134567,
        0.003689131001010537,
    ],
}

# 定义一维下采样的模块
class Downsample1d(nn.Module):
    # 初始化方法，接受内核类型和填充模式
    def __init__(self, kernel: str = "linear", pad_mode: str = "reflect"):
        super().__init__()
        # 保存填充模式
        self.pad_mode = pad_mode
        # 根据内核类型创建一维内核的张量
        kernel_1d = torch.tensor(_kernels[kernel])
        # 计算填充大小
        self.pad = kernel_1d.shape[0] // 2 - 1
        # 注册内核张量为缓冲区
        self.register_buffer("kernel", kernel_1d)

    # 前向传播方法，处理输入的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 对输入张量进行填充
        hidden_states = F.pad(hidden_states, (self.pad,) * 2, self.pad_mode)
        # 创建权重张量，用于卷积操作
        weight = hidden_states.new_zeros([hidden_states.shape[1], hidden_states.shape[1], self.kernel.shape[0]])
        # 生成索引，用于选择权重
        indices = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        # 扩展内核张量以适应权重张量的形状
        kernel = self.kernel.to(weight)[None, :].expand(hidden_states.shape[1], -1)
        # 将内核填充到权重张量的对角线上
        weight[indices, indices] = kernel
        # 使用一维卷积对输入张量进行处理并返回结果
        return F.conv1d(hidden_states, weight, stride=2)

# 定义一维上采样的模块
class Upsample1d(nn.Module):
    # 初始化方法，接受内核类型和填充模式
    def __init__(self, kernel: str = "linear", pad_mode: str = "reflect"):
        super().__init__()
        # 保存填充模式
        self.pad_mode = pad_mode
        # 根据内核类型创建一维内核的张量，并乘以2以扩展作用
        kernel_1d = torch.tensor(_kernels[kernel]) * 2
        # 计算填充大小
        self.pad = kernel_1d.shape[0] // 2 - 1
        # 注册内核张量为缓冲区
        self.register_buffer("kernel", kernel_1d)

    # 前向传播方法，处理输入的张量
    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 对输入张量进行填充
        hidden_states = F.pad(hidden_states, ((self.pad + 1) // 2,) * 2, self.pad_mode)
        # 创建权重张量，用于反卷积操作
        weight = hidden_states.new_zeros([hidden_states.shape[1], hidden_states.shape[1], self.kernel.shape[0]])
        # 生成索引，用于选择权重
        indices = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        # 扩展内核张量以适应权重张量的形状
        kernel = self.kernel.to(weight)[None, :].expand(hidden_states.shape[1], -1)
        # 将内核填充到权重张量的对角线上
        weight[indices, indices] = kernel
        # 使用一维反卷积对输入张量进行处理并返回结果
        return F.conv_transpose1d(hidden_states, weight, stride=2, padding=self.pad * 2 + 1)

# 定义一维自注意力模块
class SelfAttention1d(nn.Module):
    # 初始化方法，接受输入通道数、头数和丢弃率
    def __init__(self, in_channels: int, n_head: int = 1, dropout_rate: float = 0.0):
        super().__init__()
        # 保存输入通道数
        self.channels = in_channels
        # 创建分组归一化层
        self.group_norm = nn.GroupNorm(1, num_channels=in_channels)
        # 保存头数
        self.num_heads = n_head

        # 定义查询、键、值的线性变换
        self.query = nn.Linear(self.channels, self.channels)
        self.key = nn.Linear(self.channels, self.channels)
        self.value = nn.Linear(self.channels, self.channels)

        # 定义注意力投影的线性变换
        self.proj_attn = nn.Linear(self.channels, self.channels, bias=True)

        # 创建丢弃层
        self.dropout = nn.Dropout(dropout_rate, inplace=True)
    # 将输入投影张量进行转置以适应多头注意力机制
    def transpose_for_scores(self, projection: torch.Tensor) -> torch.Tensor:
        # 获取新的形状，将最后一个维度分割为头数和每个头的维度
        new_projection_shape = projection.size()[:-1] + (self.num_heads, -1)
        # 移动头的位置，调整形状从 (B, T, H * D) 变为 (B, T, H, D)，再变为 (B, H, T, D)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        # 返回调整后的张量
        return new_projection
    
    # 前向传播函数，处理输入的隐藏状态
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 保存输入的隐藏状态以进行残差连接
        residual = hidden_states
        # 获取批量大小、通道维度和序列长度
        batch, channel_dim, seq = hidden_states.shape
    
        # 应用分组归一化到隐藏状态
        hidden_states = self.group_norm(hidden_states)
        # 转置隐藏状态的维度以便后续处理
        hidden_states = hidden_states.transpose(1, 2)
    
        # 通过查询、键和值的线性层投影隐藏状态
        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)
    
        # 转置查询、键和值的投影以适应注意力机制
        query_states = self.transpose_for_scores(query_proj)
        key_states = self.transpose_for_scores(key_proj)
        value_states = self.transpose_for_scores(value_proj)
    
        # 计算缩放因子以防止梯度消失
        scale = 1 / math.sqrt(math.sqrt(key_states.shape[-1]))
    
        # 计算注意力得分，进行矩阵乘法
        attention_scores = torch.matmul(query_states * scale, key_states.transpose(-1, -2) * scale)
        # 计算注意力概率分布
        attention_probs = torch.softmax(attention_scores, dim=-1)
    
        # 计算注意力输出
        hidden_states = torch.matmul(attention_probs, value_states)
    
        # 调整输出的维度顺序
        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        # 获取新的隐藏状态形状以匹配通道数
        new_hidden_states_shape = hidden_states.size()[:-2] + (self.channels,)
        # 重塑隐藏状态的形状
        hidden_states = hidden_states.view(new_hidden_states_shape)
    
        # 计算下一步的隐藏状态
        hidden_states = self.proj_attn(hidden_states)
        # 再次转置隐藏状态的维度
        hidden_states = hidden_states.transpose(1, 2)
        # 应用 dropout 正则化
        hidden_states = self.dropout(hidden_states)
    
        # 将最终输出与残差相加
        output = hidden_states + residual
    
        # 返回最终输出
        return output
# 定义残差卷积块类，继承自 nn.Module
class ResConvBlock(nn.Module):
    # 初始化函数，定义输入、中间和输出通道，以及是否为最后一层的标志
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, is_last: bool = False):
        # 调用父类初始化方法
        super().__init__()
        # 设置是否为最后一层的标志
        self.is_last = is_last
        # 检查输入通道和输出通道是否相同，决定是否需要卷积跳跃连接
        self.has_conv_skip = in_channels != out_channels

        # 如果需要卷积跳跃连接，则定义 1D 卷积层
        if self.has_conv_skip:
            self.conv_skip = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        # 定义第一个卷积层，卷积核大小为 5，使用填充保持尺寸
        self.conv_1 = nn.Conv1d(in_channels, mid_channels, 5, padding=2)
        # 定义第一个组归一化层
        self.group_norm_1 = nn.GroupNorm(1, mid_channels)
        # 定义第一个 GELU 激活函数
        self.gelu_1 = nn.GELU()
        # 定义第二个卷积层，卷积核大小为 5，使用填充保持尺寸
        self.conv_2 = nn.Conv1d(mid_channels, out_channels, 5, padding=2)

        # 如果不是最后一层，则定义第二个组归一化层和激活函数
        if not self.is_last:
            self.group_norm_2 = nn.GroupNorm(1, out_channels)
            self.gelu_2 = nn.GELU()

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 如果有卷积跳跃连接，则对输入进行跳跃连接处理，否则直接使用输入
        residual = self.conv_skip(hidden_states) if self.has_conv_skip else hidden_states

        # 依次通过第一个卷积层、组归一化层和激活函数处理隐藏状态
        hidden_states = self.conv_1(hidden_states)
        hidden_states = self.group_norm_1(hidden_states)
        hidden_states = self.gelu_1(hidden_states)
        # 通过第二个卷积层处理隐藏状态
        hidden_states = self.conv_2(hidden_states)

        # 如果不是最后一层，则继续通过第二个组归一化层和激活函数处理
        if not self.is_last:
            hidden_states = self.group_norm_2(hidden_states)
            hidden_states = self.gelu_2(hidden_states)

        # 将处理后的隐藏状态与残差相加，得到最终输出
        output = hidden_states + residual
        # 返回最终输出
        return output


# 定义 UNet 中间块类，继承自 nn.Module
class UNetMidBlock1D(nn.Module):
    # 初始化函数，定义中间通道、输入通道和可选输出通道
    def __init__(self, mid_channels: int, in_channels: int, out_channels: Optional[int] = None):
        # 调用父类初始化方法
        super().__init__()

        # 如果未指定输出通道，则将输出通道设为输入通道
        out_channels = in_channels if out_channels is None else out_channels

        # 定义下采样模块，使用立方插值
        self.down = Downsample1d("cubic")
        # 创建包含多个残差卷积块的列表
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]
        # 创建包含自注意力模块的列表
        attentions = [
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(out_channels, out_channels // 32),
        ]
        # 定义上采样模块，使用立方插值
        self.up = Upsample1d(kernel="cubic")

        # 将自注意力模块和残差卷积块转换为模块列表
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
    # 定义前向传播函数，接受隐藏状态和可选的时间嵌入
        def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
            # 将隐藏状态通过下采样模块处理
            hidden_states = self.down(hidden_states)
            # 遍历注意力层和残差网络，对隐藏状态进行处理
            for attn, resnet in zip(self.attentions, self.resnets):
                # 先通过残差网络处理隐藏状态
                hidden_states = resnet(hidden_states)
                # 然后通过注意力层处理隐藏状态
                hidden_states = attn(hidden_states)
    
            # 将隐藏状态通过上采样模块处理
            hidden_states = self.up(hidden_states)
    
            # 返回最终的隐藏状态
            return hidden_states
# 定义一个一维的注意力下采样块，继承自 nn.Module
class AttnDownBlock1D(nn.Module):
    # 初始化方法，定义输入和输出通道数及中间通道数
    def __init__(self, out_channels: int, in_channels: int, mid_channels: Optional[int] = None):
        # 调用父类的初始化方法
        super().__init__()
        # 如果中间通道数为 None，则设置为输出通道数
        mid_channels = out_channels if mid_channels is None else mid_channels

        # 创建一个下采样模块，采用三次插值法
        self.down = Downsample1d("cubic")
        # 定义残差卷积块的列表
        resnets = [
            # 第一个残差卷积块，输入通道为 in_channels，输出和中间通道为 mid_channels
            ResConvBlock(in_channels, mid_channels, mid_channels),
            # 第二个残差卷积块，输入和输出通道为 mid_channels
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            # 第三个残差卷积块，输入通道为 mid_channels，输出通道为 out_channels
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]
        # 定义注意力模块的列表
        attentions = [
            # 第一个自注意力模块，输入通道为 mid_channels，输出通道为 mid_channels // 32
            SelfAttention1d(mid_channels, mid_channels // 32),
            # 第二个自注意力模块，输入通道为 mid_channels，输出通道为 mid_channels // 32
            SelfAttention1d(mid_channels, mid_channels // 32),
            # 第三个自注意力模块，输入通道为 out_channels，输出通道为 out_channels // 32
            SelfAttention1d(out_channels, out_channels // 32),
        ]

        # 将注意力模块列表封装成 nn.ModuleList
        self.attentions = nn.ModuleList(attentions)
        # 将残差卷积块列表封装成 nn.ModuleList
        self.resnets = nn.ModuleList(resnets)

    # 前向传播方法，定义输入和输出
    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 对隐藏状态进行下采样
        hidden_states = self.down(hidden_states)

        # 遍历残差卷积块和注意力模块
        for resnet, attn in zip(self.resnets, self.attentions):
            # 通过残差卷积块处理隐藏状态
            hidden_states = resnet(hidden_states)
            # 通过注意力模块处理隐藏状态
            hidden_states = attn(hidden_states)

        # 返回处理后的隐藏状态和一个包含隐藏状态的元组
        return hidden_states, (hidden_states,)


# 定义一个一维的下采样块，继承自 nn.Module
class DownBlock1D(nn.Module):
    # 初始化方法，定义输入和输出通道数及中间通道数
    def __init__(self, out_channels: int, in_channels: int, mid_channels: Optional[int] = None):
        # 调用父类的初始化方法
        super().__init__()
        # 如果中间通道数为 None，则设置为输出通道数
        mid_channels = out_channels if mid_channels is None else mid_channels

        # 创建一个下采样模块，采用三次插值法
        self.down = Downsample1d("cubic")
        # 定义残差卷积块的列表
        resnets = [
            # 第一个残差卷积块，输入通道为 in_channels，输出和中间通道为 mid_channels
            ResConvBlock(in_channels, mid_channels, mid_channels),
            # 第二个残差卷积块，输入和输出通道为 mid_channels
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            # 第三个残差卷积块，输入通道为 mid_channels，输出通道为 out_channels
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]

        # 将残差卷积块列表封装成 nn.ModuleList
        self.resnets = nn.ModuleList(resnets)

    # 前向传播方法，定义输入和输出
    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 对隐藏状态进行下采样
        hidden_states = self.down(hidden_states)

        # 遍历残差卷积块
        for resnet in self.resnets:
            # 通过残差卷积块处理隐藏状态
            hidden_states = resnet(hidden_states)

        # 返回处理后的隐藏状态和一个包含隐藏状态的元组
        return hidden_states, (hidden_states,)


# 定义一个没有跳过连接的一维下采样块，继承自 nn.Module
class DownBlock1DNoSkip(nn.Module):
    # 初始化方法，定义输入和输出通道数及中间通道数
    def __init__(self, out_channels: int, in_channels: int, mid_channels: Optional[int] = None):
        # 调用父类的初始化方法
        super().__init__()
        # 如果中间通道数为 None，则设置为输出通道数
        mid_channels = out_channels if mid_channels is None else mid_channels

        # 定义残差卷积块的列表
        resnets = [
            # 第一个残差卷积块，输入通道为 in_channels，输出和中间通道为 mid_channels
            ResConvBlock(in_channels, mid_channels, mid_channels),
            # 第二个残差卷积块，输入和输出通道为 mid_channels
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            # 第三个残差卷积块，输入通道为 mid_channels，输出通道为 out_channels
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]

        # 将残差卷积块列表封装成 nn.ModuleList
        self.resnets = nn.ModuleList(resnets)

    # 前向传播方法，定义输入和输出
    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 将隐藏状态和 temb 在通道维度上拼接
        hidden_states = torch.cat([hidden_states, temb], dim=1)
        # 遍历残差卷积块
        for resnet in self.resnets:
            # 通过残差卷积块处理隐藏状态
            hidden_states = resnet(hidden_states)

        # 返回处理后的隐藏状态和一个包含隐藏状态的元组
        return hidden_states, (hidden_states,)


# 定义一个一维的注意力上采样块，继承自 nn.Module
class AttnUpBlock1D(nn.Module):
    # 初始化方法，用于创建类的实例，设置输入、输出和中间通道数
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        # 调用父类初始化方法
        super().__init__()
        # 如果中间通道数未提供，则将其设置为输出通道数
        mid_channels = out_channels if mid_channels is None else mid_channels
    
        # 创建残差卷积块列表，配置输入、中间和输出通道数
        resnets = [
            ResConvBlock(2 * in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]
        # 创建自注意力层列表，配置通道数
        attentions = [
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(out_channels, out_channels // 32),
        ]
    
        # 将注意力层添加到模块列表中，以便在前向传播中使用
        self.attentions = nn.ModuleList(attentions)
        # 将残差卷积块添加到模块列表中，以便在前向传播中使用
        self.resnets = nn.ModuleList(resnets)
        # 初始化上采样层，使用立方插值
        self.up = Upsample1d(kernel="cubic")
    
    # 前向传播方法，定义输入张量和输出张量之间的计算
    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 获取残差隐藏状态元组中的最后一个状态
        res_hidden_states = res_hidden_states_tuple[-1]
        # 将隐藏状态与残差隐藏状态在通道维度上拼接
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
    
        # 遍历残差块和注意力层并依次处理隐藏状态
        for resnet, attn in zip(self.resnets, self.attentions):
            # 使用残差块处理隐藏状态
            hidden_states = resnet(hidden_states)
            # 使用注意力层处理隐藏状态
            hidden_states = attn(hidden_states)
    
        # 对处理后的隐藏状态进行上采样
        hidden_states = self.up(hidden_states)
    
        # 返回最终的隐藏状态
        return hidden_states
# 定义一维上采样块的类，继承自 nn.Module
class UpBlock1D(nn.Module):
    # 初始化方法，接收输入通道、输出通道和中间通道的参数
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        # 调用父类初始化方法
        super().__init__()
        # 如果中间通道为 None，则将其设置为输入通道数
        mid_channels = in_channels if mid_channels is None else mid_channels

        # 定义包含三个残差卷积块的列表
        resnets = [
            ResConvBlock(2 * in_channels, mid_channels, mid_channels),  # 第一个残差块，输入通道是输入通道的两倍
            ResConvBlock(mid_channels, mid_channels, mid_channels),      # 第二个残差块，输入输出通道均为中间通道
            ResConvBlock(mid_channels, mid_channels, out_channels),      # 第三个残差块，输出通道为目标输出通道
        ]

        # 将残差块列表转换为 nn.ModuleList
        self.resnets = nn.ModuleList(resnets)
        # 定义一维上采样层，使用立方插值核
        self.up = Upsample1d(kernel="cubic")

    # 前向传播方法，接收隐藏状态和残差隐藏状态元组
    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 获取最后一个残差隐藏状态
        res_hidden_states = res_hidden_states_tuple[-1]
        # 将隐藏状态和残差隐藏状态在通道维度上连接
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        # 遍历每个残差块进行前向传播
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        # 对隐藏状态进行上采样
        hidden_states = self.up(hidden_states)

        # 返回上采样后的隐藏状态
        return hidden_states


# 定义不使用跳过连接的一维上采样块的类，继承自 nn.Module
class UpBlock1DNoSkip(nn.Module):
    # 初始化方法，接收输入通道、输出通道和中间通道的参数
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        # 调用父类初始化方法
        super().__init__()
        # 如果中间通道为 None，则将其设置为输入通道数
        mid_channels = in_channels if mid_channels is None else mid_channels

        # 定义包含三个残差卷积块的列表
        resnets = [
            ResConvBlock(2 * in_channels, mid_channels, mid_channels),  # 第一个残差块，输入通道是输入通道的两倍
            ResConvBlock(mid_channels, mid_channels, mid_channels),      # 第二个残差块，输入输出通道均为中间通道
            ResConvBlock(mid_channels, mid_channels, out_channels, is_last=True),  # 第三个残差块，输出通道为目标输出通道，标记为最后一个块
        ]

        # 将残差块列表转换为 nn.ModuleList
        self.resnets = nn.ModuleList(resnets)

    # 前向传播方法，接收隐藏状态和残差隐藏状态元组
    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 获取最后一个残差隐藏状态
        res_hidden_states = res_hidden_states_tuple[-1]
        # 将隐藏状态和残差隐藏状态在通道维度上连接
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        # 遍历每个残差块进行前向传播
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        # 返回处理后的隐藏状态
        return hidden_states


# 定义各种下采样块的类型
DownBlockType = Union[DownResnetBlock1D, DownBlock1D, AttnDownBlock1D, DownBlock1DNoSkip]
# 定义各种中间块的类型
MidBlockType = Union[MidResTemporalBlock1D, ValueFunctionMidBlock1D, UNetMidBlock1D]
# 定义各种输出块的类型
OutBlockType = Union[OutConv1DBlock, OutValueFunctionBlock]
# 定义各种上采样块的类型
UpBlockType = Union[UpResnetBlock1D, UpBlock1D, AttnUpBlock1D, UpBlock1DNoSkip]


# 根据类型获取对应的下采样块
def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
) -> DownBlockType:
    # 如果指定的下采样块类型为 DownResnetBlock1D，返回相应的块
    if down_block_type == "DownResnetBlock1D":
        return DownResnetBlock1D(
            in_channels=in_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
        )
    # 如果指定的下采样块类型为 DownBlock1D，返回相应的块
    elif down_block_type == "DownBlock1D":
        return DownBlock1D(out_channels=out_channels, in_channels=in_channels)
    # 检查下采样块类型是否为 "AttnDownBlock1D"
    elif down_block_type == "AttnDownBlock1D":
        # 返回一个 AttnDownBlock1D 对象，传入输出和输入通道参数
        return AttnDownBlock1D(out_channels=out_channels, in_channels=in_channels)
    # 检查下采样块类型是否为 "DownBlock1DNoSkip"
    elif down_block_type == "DownBlock1DNoSkip":
        # 返回一个 DownBlock1DNoSkip 对象，传入输出和输入通道参数
        return DownBlock1DNoSkip(out_channels=out_channels, in_channels=in_channels)
    # 如果下采样块类型不匹配，抛出一个值错误异常
    raise ValueError(f"{down_block_type} does not exist.")
# 根据给定的上采样块类型，创建并返回对应的上采样块实例
def get_up_block(
    # 上采样块类型
    up_block_type: str, 
    # 网络层数
    num_layers: int, 
    # 输入通道数
    in_channels: int, 
    # 输出通道数
    out_channels: int, 
    # 时间嵌入通道数
    temb_channels: int, 
    # 是否添加上采样
    add_upsample: bool
) -> UpBlockType:
    # 检查上采样块类型是否为 "UpResnetBlock1D"
    if up_block_type == "UpResnetBlock1D":
        # 创建并返回 UpResnetBlock1D 实例
        return UpResnetBlock1D(
            # 设置输入通道数
            in_channels=in_channels,
            # 设置网络层数
            num_layers=num_layers,
            # 设置输出通道数
            out_channels=out_channels,
            # 设置时间嵌入通道数
            temb_channels=temb_channels,
            # 设置是否添加上采样
            add_upsample=add_upsample,
        )
    # 检查上采样块类型是否为 "UpBlock1D"
    elif up_block_type == "UpBlock1D":
        # 创建并返回 UpBlock1D 实例
        return UpBlock1D(in_channels=in_channels, out_channels=out_channels)
    # 检查上采样块类型是否为 "AttnUpBlock1D"
    elif up_block_type == "AttnUpBlock1D":
        # 创建并返回 AttnUpBlock1D 实例
        return AttnUpBlock1D(in_channels=in_channels, out_channels=out_channels)
    # 检查上采样块类型是否为 "UpBlock1DNoSkip"
    elif up_block_type == "UpBlock1DNoSkip":
        # 创建并返回 UpBlock1DNoSkip 实例
        return UpBlock1DNoSkip(in_channels=in_channels, out_channels=out_channels)
    # 抛出错误，表示该上采样块类型不存在
    raise ValueError(f"{up_block_type} does not exist.")


# 根据给定的中间块类型，创建并返回对应的中间块实例
def get_mid_block(
    # 中间块类型
    mid_block_type: str,
    # 网络层数
    num_layers: int,
    # 输入通道数
    in_channels: int,
    # 中间通道数
    mid_channels: int,
    # 输出通道数
    out_channels: int,
    # 嵌入维度
    embed_dim: int,
    # 是否添加下采样
    add_downsample: bool,
) -> MidBlockType:
    # 检查中间块类型是否为 "MidResTemporalBlock1D"
    if mid_block_type == "MidResTemporalBlock1D":
        # 创建并返回 MidResTemporalBlock1D 实例
        return MidResTemporalBlock1D(
            # 设置网络层数
            num_layers=num_layers,
            # 设置输入通道数
            in_channels=in_channels,
            # 设置输出通道数
            out_channels=out_channels,
            # 设置嵌入维度
            embed_dim=embed_dim,
            # 设置是否添加下采样
            add_downsample=add_downsample,
        )
    # 检查中间块类型是否为 "ValueFunctionMidBlock1D"
    elif mid_block_type == "ValueFunctionMidBlock1D":
        # 创建并返回 ValueFunctionMidBlock1D 实例
        return ValueFunctionMidBlock1D(in_channels=in_channels, out_channels=out_channels, embed_dim=embed_dim)
    # 检查中间块类型是否为 "UNetMidBlock1D"
    elif mid_block_type == "UNetMidBlock1D":
        # 创建并返回 UNetMidBlock1D 实例
        return UNetMidBlock1D(in_channels=in_channels, mid_channels=mid_channels, out_channels=out_channels)
    # 抛出错误，表示该中间块类型不存在
    raise ValueError(f"{mid_block_type} does not exist.")


# 根据给定的输出块类型，创建并返回对应的输出块实例
def get_out_block(
    # 输出块类型
    *, out_block_type: str, 
    # 输出组数
    num_groups_out: int, 
    # 嵌入维度
    embed_dim: int, 
    # 输出通道数
    out_channels: int, 
    # 激活函数类型
    act_fn: str, 
    # 全连接层维度
    fc_dim: int
) -> Optional[OutBlockType]:
    # 检查输出块类型是否为 "OutConv1DBlock"
    if out_block_type == "OutConv1DBlock":
        # 创建并返回 OutConv1DBlock 实例
        return OutConv1DBlock(num_groups_out, out_channels, embed_dim, act_fn)
    # 检查输出块类型是否为 "ValueFunction"
    elif out_block_type == "ValueFunction":
        # 创建并返回 OutValueFunctionBlock 实例
        return OutValueFunctionBlock(fc_dim, embed_dim, act_fn)
    # 如果输出块类型不匹配，返回 None
    return None
```