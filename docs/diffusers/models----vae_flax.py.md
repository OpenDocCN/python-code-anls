# `.\diffusers\models\vae_flax.py`

```py
# 版权所有 2024 The HuggingFace Team. 保留所有权利。
#
# 根据 Apache License, Version 2.0 (以下称为“许可证”) 进行授权；
# 除非遵守该许可证，否则您不得使用此文件。
# 您可以在以下网址获得许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，软件
# 根据该许可证分发是按“原样”基础进行的，
# 不提供任何形式的明示或暗示的担保或条件。
# 有关许可证规定的权限和限制，请参见许可证。

# JAX 实现 VQGAN，来源于 taming-transformers https://github.com/CompVis/taming-transformers

import math  # 导入数学库，提供数学函数
from functools import partial  # 从 functools 模块导入 partial 函数，用于部分应用函数
from typing import Tuple  # 从 typing 模块导入 Tuple，用于类型注解

import flax  # 导入 flax 库，支持神经网络构建
import flax.linen as nn  # 从 flax 导入 linen 模块，简化神经网络层的创建
import jax  # 导入 jax 库，支持高效数值计算
import jax.numpy as jnp  # 从 jax 导入 numpy 模块，提供类似于 NumPy 的数组操作
from flax.core.frozen_dict import FrozenDict  # 导入 FrozenDict，提供不可变字典的实现

from ..configuration_utils import ConfigMixin, flax_register_to_config  # 导入配置相关的混合类和注册函数
from ..utils import BaseOutput  # 从 utils 模块导入 BaseOutput 基类
from .modeling_flax_utils import FlaxModelMixin  # 从 modeling_flax_utils 导入 FlaxModelMixin


@flax.struct.dataclass  # 使用 flax 的数据类装饰器，自动生成类的初始化和其他方法
class FlaxDecoderOutput(BaseOutput):  # 定义解码器输出类，继承自 BaseOutput
    """
    解码方法的输出。

    参数:
        sample (`jnp.ndarray` 的形状为 `(batch_size, num_channels, height, width)`):
            模型最后一层的解码输出样本。
        dtype (`jnp.dtype`, *可选*, 默认为 `jnp.float32`):
            参数的 `dtype`。
    """

    sample: jnp.ndarray  # 定义解码样本为 jnp.ndarray 类型


@flax.struct.dataclass  # 使用 flax 的数据类装饰器
class FlaxAutoencoderKLOutput(BaseOutput):  # 定义自动编码器 KL 输出类，继承自 BaseOutput
    """
    自动编码器 KL 编码方法的输出。

    参数:
        latent_dist (`FlaxDiagonalGaussianDistribution`):
            编码器的输出表示为 FlaxDiagonalGaussianDistribution 的均值和对数方差。
            `FlaxDiagonalGaussianDistribution` 允许从分布中采样潜在变量。
    """

    latent_dist: "FlaxDiagonalGaussianDistribution"  # 定义潜在分布类型为 FlaxDiagonalGaussianDistribution


class FlaxUpsample2D(nn.Module):  # 定义 2D 上采样层的 Flax 实现，继承自 nn.Module
    """
    Flax 实现的 2D 上采样层

    参数:
        in_channels (`int`):
            输入通道数
        dtype (:obj:`jnp.dtype`, *可选*, 默认为 jnp.float32):
            参数的 `dtype`
    """

    in_channels: int  # 定义输入通道数为整型
    dtype: jnp.dtype = jnp.float32  # 定义参数类型，默认为 jnp.float32

    def setup(self):  # 定义设置方法，在模块初始化时调用
        self.conv = nn.Conv(  # 创建卷积层
            self.in_channels,  # 设置输入通道数
            kernel_size=(3, 3),  # 设置卷积核大小为 3x3
            strides=(1, 1),  # 设置卷积步幅为 1
            padding=((1, 1), (1, 1)),  # 设置填充方式
            dtype=self.dtype,  # 设置卷积层参数的类型
        )

    def __call__(self, hidden_states):  # 定义模块的前向传播方法
        batch, height, width, channels = hidden_states.shape  # 解包输入形状为批量大小、高度、宽度和通道数
        hidden_states = jax.image.resize(  # 对输入进行上采样
            hidden_states,  # 输入的隐藏状态
            shape=(batch, height * 2, width * 2, channels),  # 设置输出形状为原来的两倍
            method="nearest",  # 使用最近邻插值法进行上采样
        )
        hidden_states = self.conv(hidden_states)  # 通过卷积层处理上采样后的状态
        return hidden_states  # 返回处理后的隐藏状态


class FlaxDownsample2D(nn.Module):  # 定义 2D 下采样层的 Flax 实现，继承自 nn.Module
    """
    Flax 实现的 2D 下采样层
    # 参数说明文档
        Args:
            in_channels (`int`):
                输入通道数
            dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
                参数数据类型
    
        # 声明输入通道数为整数类型
        in_channels: int
        # 声明数据类型，默认值为 jnp.float32
        dtype: jnp.dtype = jnp.float32
    
        # 设置方法，用于初始化卷积层
        def setup(self):
            # 创建卷积层，指定输入通道、卷积核大小、步幅和填充方式
            self.conv = nn.Conv(
                self.in_channels,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="VALID",
                dtype=self.dtype,
            )
    
        # 调用方法，接收隐藏状态作为输入
        def __call__(self, hidden_states):
            # 定义填充的尺寸，增加高度和宽度维度的边界
            pad = ((0, 0), (0, 1), (0, 1), (0, 0))  # pad height and width dim
            # 对输入的隐藏状态进行填充
            hidden_states = jnp.pad(hidden_states, pad_width=pad)
            # 将填充后的隐藏状态输入卷积层进行处理
            hidden_states = self.conv(hidden_states)
            # 返回处理后的隐藏状态
            return hidden_states
# 定义 Flax 实现的 2D Resnet Block 类，继承自 nn.Module
class FlaxResnetBlock2D(nn.Module):
    """
    Flax 实现的 2D Resnet Block。

    参数:
        in_channels (`int`):
            输入通道数
        out_channels (`int`):
            输出通道数
        dropout (:obj:`float`, *可选*, 默认为 0.0):
            Dropout 率
        groups (:obj:`int`, *可选*, 默认为 `32`):
            用于分组归一化的组数。
        use_nin_shortcut (:obj:`bool`, *可选*, 默认为 `None`):
            是否使用 `nin_shortcut`。这会在 ResNet 块内部激活一个新层
        dtype (:obj:`jnp.dtype`, *可选*, 默认为 jnp.float32):
            参数数据类型
    """

    # 定义输入通道数，输出通道数，dropout 率，分组数，是否使用 nin_shortcut 和数据类型
    in_channels: int
    out_channels: int = None
    dropout: float = 0.0
    groups: int = 32
    use_nin_shortcut: bool = None
    dtype: jnp.dtype = jnp.float32

    # 设置方法，用于初始化各层
    def setup(self):
        # 如果未指定输出通道数，则设置为输入通道数
        out_channels = self.in_channels if self.out_channels is None else self.out_channels

        # 初始化第一层分组归一化
        self.norm1 = nn.GroupNorm(num_groups=self.groups, epsilon=1e-6)
        # 初始化第一层卷积
        self.conv1 = nn.Conv(
            out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        # 初始化第二层分组归一化
        self.norm2 = nn.GroupNorm(num_groups=self.groups, epsilon=1e-6)
        # 初始化 dropout 层
        self.dropout_layer = nn.Dropout(self.dropout)
        # 初始化第二层卷积
        self.conv2 = nn.Conv(
            out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        # 根据输入和输出通道数判断是否使用 nin_shortcut
        use_nin_shortcut = self.in_channels != out_channels if self.use_nin_shortcut is None else self.use_nin_shortcut

        # 初始化快捷连接卷积层
        self.conv_shortcut = None
        if use_nin_shortcut:
            # 如果需要使用 nin_shortcut，则初始化其卷积层
            self.conv_shortcut = nn.Conv(
                out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                dtype=self.dtype,
            )

    # 前向传播方法，接受隐状态和确定性标志
    def __call__(self, hidden_states, deterministic=True):
        # 保存输入作为残差
        residual = hidden_states
        # 通过第一层归一化
        hidden_states = self.norm1(hidden_states)
        # 应用 Swish 激活函数
        hidden_states = nn.swish(hidden_states)
        # 通过第一层卷积
        hidden_states = self.conv1(hidden_states)

        # 通过第二层归一化
        hidden_states = self.norm2(hidden_states)
        # 再次应用 Swish 激活函数
        hidden_states = nn.swish(hidden_states)
        # 应用 dropout
        hidden_states = self.dropout_layer(hidden_states, deterministic)
        # 通过第二层卷积
        hidden_states = self.conv2(hidden_states)

        # 如果使用快捷连接，则通过卷积层处理残差
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        # 返回隐状态与残差的和
        return hidden_states + residual


# 定义 Flax 实现的基于卷积的多头注意力块类，继承自 nn.Module
class FlaxAttentionBlock(nn.Module):
    r"""
    Flax 基于卷积的多头注意力块，用于扩散模型的 VAE。
    # 定义参数文档
    Parameters:
        channels (:obj:`int`):  # 输入通道数
            Input channels
        num_head_channels (:obj:`int`, *optional*, defaults to `None`):  # 注意力头的数量（可选，默认值为None）
            Number of attention heads
        num_groups (:obj:`int`, *optional*, defaults to `32`):  # 用于组归一化的组数（可选，默认值为32）
            The number of groups to use for group norm
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):  # 参数的数据类型（可选，默认值为jnp.float32）
            Parameters `dtype`

    """

    channels: int  # 定义输入通道数类型
    num_head_channels: int = None  # 定义注意力头的数量，默认值为None
    num_groups: int = 32  # 定义组归一化的组数，默认值为32
    dtype: jnp.dtype = jnp.float32  # 定义参数的数据类型，默认值为jnp.float32

    def setup(self):  # 设置方法
        # 计算注意力头的数量，如果未定义则默认为1
        self.num_heads = self.channels // self.num_head_channels if self.num_head_channels is not None else 1

        # 定义稠密层的部分，使用指定的通道和数据类型
        dense = partial(nn.Dense, self.channels, dtype=self.dtype)

        # 创建组归一化层，使用指定的组数和小常数
        self.group_norm = nn.GroupNorm(num_groups=self.num_groups, epsilon=1e-6)
        # 创建查询、键和值的稠密层
        self.query, self.key, self.value = dense(), dense(), dense()
        # 创建投影注意力的稠密层
        self.proj_attn = dense()

    def transpose_for_scores(self, projection):  # 转置以适应注意力头
        # 定义新的投影形状，插入头的维度
        new_projection_shape = projection.shape[:-1] + (self.num_heads, -1)
        # 将头的维度移动到第二个位置（B, T, H * D）->（B, T, H, D）
        new_projection = projection.reshape(new_projection_shape)
        # （B, T, H, D）->（B, H, T, D）
        new_projection = jnp.transpose(new_projection, (0, 2, 1, 3))
        return new_projection  # 返回转置后的投影

    def __call__(self, hidden_states):  # 定义调用方法
        residual = hidden_states  # 保存输入的残差
        batch, height, width, channels = hidden_states.shape  # 获取输入的形状

        hidden_states = self.group_norm(hidden_states)  # 对隐藏状态进行组归一化

        # 重新调整隐藏状态的形状以适应注意力机制
        hidden_states = hidden_states.reshape((batch, height * width, channels))

        query = self.query(hidden_states)  # 计算查询
        key = self.key(hidden_states)  # 计算键
        value = self.value(hidden_states)  # 计算值

        # 转置查询、键和值以适应注意力计算
        query = self.transpose_for_scores(query)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        # 计算注意力权重
        scale = 1 / math.sqrt(math.sqrt(self.channels / self.num_heads))  # 计算缩放因子
        attn_weights = jnp.einsum("...qc,...kc->...qk", query * scale, key * scale)  # 计算注意力权重
        attn_weights = nn.softmax(attn_weights, axis=-1)  # 对注意力权重进行归一化

        # 根据注意力权重聚合值
        hidden_states = jnp.einsum("...kc,...qk->...qc", value, attn_weights)

        hidden_states = jnp.transpose(hidden_states, (0, 2, 1, 3))  # 转置隐藏状态
        new_hidden_states_shape = hidden_states.shape[:-2] + (self.channels,)  # 定义新的隐藏状态形状
        hidden_states = hidden_states.reshape(new_hidden_states_shape)  # 重新调整形状

        hidden_states = self.proj_attn(hidden_states)  # 通过投影注意力层处理隐藏状态
        hidden_states = hidden_states.reshape((batch, height, width, channels))  # 还原到原始形状
        hidden_states = hidden_states + residual  # 加上残差
        return hidden_states  # 返回处理后的隐藏状态
# 定义一个基于 Flax 和 Resnet 的二维编码器块，用于扩散式变分自编码器
class FlaxDownEncoderBlock2D(nn.Module):
    r"""
    Flax Resnet blocks-based Encoder block for diffusion-based VAE.

    Parameters:
        in_channels (:obj:`int`):
            输入通道数
        out_channels (:obj:`int`):
            输出通道数
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout 率
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Resnet 层块的数量
        resnet_groups (:obj:`int`, *optional*, defaults to `32`):
            Resnet 块组归一化使用的组数
        add_downsample (:obj:`bool`, *optional*, defaults to `True`):
            是否添加下采样层
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            参数的数据类型
    """

    # 初始化输入和输出通道、dropout 率、层数、组数、是否添加下采样和数据类型
    in_channels: int
    out_channels: int
    dropout: float = 0.0
    num_layers: int = 1
    resnet_groups: int = 32
    add_downsample: bool = True
    dtype: jnp.dtype = jnp.float32

    # 设置函数，用于构建模块的内部结构
    def setup(self):
        # 创建一个空列表用于存放 Resnet 块
        resnets = []
        # 遍历设置的层数，构建 Resnet 块
        for i in range(self.num_layers):
            # 如果是第一层，使用输入通道，否则使用输出通道
            in_channels = self.in_channels if i == 0 else self.out_channels

            # 创建一个 Resnet 块实例
            res_block = FlaxResnetBlock2D(
                in_channels=in_channels,
                out_channels=self.out_channels,
                dropout=self.dropout,
                groups=self.resnet_groups,
                dtype=self.dtype,
            )
            # 将创建的 Resnet 块添加到列表中
            resnets.append(res_block)
        # 将所有 Resnet 块存储到实例变量中
        self.resnets = resnets

        # 如果需要添加下采样层，则创建下采样模块
        if self.add_downsample:
            self.downsamplers_0 = FlaxDownsample2D(self.out_channels, dtype=self.dtype)

    # 前向传播函数，用于处理输入的隐藏状态
    def __call__(self, hidden_states, deterministic=True):
        # 依次通过每个 Resnet 块处理隐藏状态
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, deterministic=deterministic)

        # 如果需要下采样，则调用下采样层处理隐藏状态
        if self.add_downsample:
            hidden_states = self.downsamplers_0(hidden_states)

        # 返回处理后的隐藏状态
        return hidden_states


# 定义一个基于 Flax 和 Resnet 的二维解码器块，用于扩散式变分自编码器
class FlaxUpDecoderBlock2D(nn.Module):
    r"""
    Flax Resnet blocks-based Decoder block for diffusion-based VAE.

    Parameters:
        in_channels (:obj:`int`):
            输入通道数
        out_channels (:obj:`int`):
            输出通道数
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout 率
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Resnet 层块的数量
        resnet_groups (:obj:`int`, *optional*, defaults to `32`):
            Resnet 块组归一化使用的组数
        add_upsample (:obj:`bool`, *optional*, defaults to `True`):
            是否添加上采样层
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            参数的数据类型
    """

    # 初始化输入和输出通道、dropout 率、层数、组数、是否添加上采样和数据类型
    in_channels: int
    out_channels: int
    dropout: float = 0.0
    num_layers: int = 1
    resnet_groups: int = 32
    add_upsample: bool = True
    dtype: jnp.dtype = jnp.float32
    # 设置方法，初始化 ResNet 模块
        def setup(self):
            # 创建一个空列表，用于存储 ResNet 块
            resnets = []
            # 遍历指定数量的层
            for i in range(self.num_layers):
                # 根据层索引确定输入通道数，第一层使用 in_channels，其余层使用 out_channels
                in_channels = self.in_channels if i == 0 else self.out_channels
                # 创建一个 ResNet 块并初始化其参数
                res_block = FlaxResnetBlock2D(
                    in_channels=in_channels,  # 输入通道数
                    out_channels=self.out_channels,  # 输出通道数
                    dropout=self.dropout,  # dropout 概率
                    groups=self.resnet_groups,  # 组数
                    dtype=self.dtype,  # 数据类型
                )
                # 将创建的 ResNet 块添加到列表中
                resnets.append(res_block)
    
            # 将创建的 ResNet 块列表赋值给实例变量
            self.resnets = resnets
    
            # 如果需要添加上采样层，则初始化上采样层
            if self.add_upsample:
                self.upsamplers_0 = FlaxUpsample2D(self.out_channels, dtype=self.dtype)
    
        # 前向传播方法，处理隐藏状态
        def __call__(self, hidden_states, deterministic=True):
            # 逐个通过 ResNet 块处理隐藏状态
            for resnet in self.resnets:
                hidden_states = resnet(hidden_states, deterministic=deterministic)
    
            # 如果需要上采样，则应用上采样层
            if self.add_upsample:
                hidden_states = self.upsamplers_0(hidden_states)
    
            # 返回处理后的隐藏状态
            return hidden_states
# 定义 FlaxUNetMidBlock2D 类，继承自 nn.Module
class FlaxUNetMidBlock2D(nn.Module):
    r"""
    Flax Unet 中间块模块。

    参数：
        in_channels (:obj:`int`):
            输入通道数
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout 率
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Resnet 层块的数量
        resnet_groups (:obj:`int`, *optional*, defaults to `32`):
            Resnet 和注意力块的组归一化使用的组数
        num_attention_heads (:obj:`int`, *optional*, defaults to `1`):
            每个注意力块的注意力头数量
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            参数的数据类型
    """

    # 定义类属性，包含输入通道数、dropout 率等
    in_channels: int
    dropout: float = 0.0
    num_layers: int = 1
    resnet_groups: int = 32
    num_attention_heads: int = 1
    dtype: jnp.dtype = jnp.float32

    # 设置模块的初始化方法
    def setup(self):
        # 计算 Resnet 组数，若未指定则取输入通道数的四分之一与 32 的最小值
        resnet_groups = self.resnet_groups if self.resnet_groups is not None else min(self.in_channels // 4, 32)

        # 至少有一个 Resnet 层块
        resnets = [
            FlaxResnetBlock2D(
                in_channels=self.in_channels,  # 输入通道数
                out_channels=self.in_channels,  # 输出通道数
                dropout=self.dropout,  # dropout 率
                groups=resnet_groups,  # 组数
                dtype=self.dtype,  # 数据类型
            )
        ]

        # 初始化注意力块列表
        attentions = []

        # 创建多个层块
        for _ in range(self.num_layers):
            # 创建一个注意力块并添加到列表中
            attn_block = FlaxAttentionBlock(
                channels=self.in_channels,  # 通道数
                num_head_channels=self.num_attention_heads,  # 注意力头数量
                num_groups=resnet_groups,  # 组数
                dtype=self.dtype,  # 数据类型
            )
            attentions.append(attn_block)  # 将注意力块添加到列表

            # 创建一个 Resnet 层块并添加到列表中
            res_block = FlaxResnetBlock2D(
                in_channels=self.in_channels,  # 输入通道数
                out_channels=self.in_channels,  # 输出通道数
                dropout=self.dropout,  # dropout 率
                groups=resnet_groups,  # 组数
                dtype=self.dtype,  # 数据类型
            )
            resnets.append(res_block)  # 将 Resnet 层块添加到列表

        # 将生成的 Resnet 层块和注意力块存储为类属性
        self.resnets = resnets
        self.attentions = attentions

    # 定义模块的前向调用方法
    def __call__(self, hidden_states, deterministic=True):
        # 使用第一个 Resnet 层块处理隐藏状态
        hidden_states = self.resnets[0](hidden_states, deterministic=deterministic)
        # 遍历注意力块和 Resnet 层块进行处理
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states)  # 应用注意力块
            hidden_states = resnet(hidden_states, deterministic=deterministic)  # 应用 Resnet 层块

        # 返回处理后的隐藏状态
        return hidden_states


# 定义 FlaxEncoder 类，继承自 nn.Module
class FlaxEncoder(nn.Module):
    r"""
    Flax 实现的 VAE 编码器。

    该模型是 Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    子类。可将其用作常规 Flax linen 模块，并参考 Flax 文档以获取与
    一般用法和行为相关的所有事项。

    最后，该模型支持固有的 JAX 特性，例如：
    - [即时编译 (JIT)](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    # 自动微分相关链接
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    # 向量化相关链接
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    # 并行化相关链接
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    # 参数说明
    Parameters:
        # 输入通道数，默认为 3
        in_channels (:obj:`int`, *optional*, defaults to 3):
            Input channels
        # 输出通道数，默认为 3
        out_channels (:obj:`int`, *optional*, defaults to 3):
            Output channels
        # 下采样块类型，默认为 `(DownEncoderBlock2D)`
        down_block_types (:obj:`Tuple[str]`, *optional*, defaults to `(DownEncoderBlock2D)`):
            DownEncoder block type
        # 每个块的输出通道数元组，默认为 `(64,)`
        block_out_channels (:obj:`Tuple[str]`, *optional*, defaults to `(64,)`):
            Tuple containing the number of output channels for each block
        # 每个块的 ResNet 层数，默认为 2
        layers_per_block (:obj:`int`, *optional*, defaults to `2`):
            Number of Resnet layer for each block
        # 归一化分组数，默认为 32
        norm_num_groups (:obj:`int`, *optional*, defaults to `32`):
            norm num group
        # 激活函数类型，默认为 `silu`
        act_fn (:obj:`str`, *optional*, defaults to `silu`):
            Activation function
        # 是否将最后的输出通道数加倍，默认为 False
        double_z (:obj:`bool`, *optional*, defaults to `False`):
            Whether to double the last output channels
        # 参数数据类型，默认为 jnp.float32
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """

    # 设置默认输入通道数为 3
    in_channels: int = 3
    # 设置默认输出通道数为 3
    out_channels: int = 3
    # 设置默认下采样块类型
    down_block_types: Tuple[str] = ("DownEncoderBlock2D",)
    # 设置每个块的默认输出通道数
    block_out_channels: Tuple[int] = (64,)
    # 设置每个块的默认层数为 2
    layers_per_block: int = 2
    # 设置默认归一化分组数为 32
    norm_num_groups: int = 32
    # 设置默认激活函数为 "silu"
    act_fn: str = "silu"
    # 设置默认是否加倍输出通道数为 False
    double_z: bool = False
    # 设置默认数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 设置模型的各个层
    def setup(self):
        # 获取输出通道的数量
        block_out_channels = self.block_out_channels
        # 输入层，定义卷积操作
        self.conv_in = nn.Conv(
            block_out_channels[0],  # 输入通道数
            kernel_size=(3, 3),  # 卷积核大小
            strides=(1, 1),  # 步幅
            padding=((1, 1), (1, 1)),  # 填充方式
            dtype=self.dtype,  # 数据类型
        )

        # 下采样部分
        down_blocks = []  # 初始化下采样块列表
        output_channel = block_out_channels[0]  # 当前输出通道
        for i, _ in enumerate(self.down_block_types):  # 遍历下采样块类型
            input_channel = output_channel  # 当前输入通道
            output_channel = block_out_channels[i]  # 更新输出通道
            is_final_block = i == len(block_out_channels) - 1  # 检查是否为最后一个块

            # 创建下采样块
            down_block = FlaxDownEncoderBlock2D(
                in_channels=input_channel,  # 输入通道数
                out_channels=output_channel,  # 输出通道数
                num_layers=self.layers_per_block,  # 块内层数
                resnet_groups=self.norm_num_groups,  # 归一化组数
                add_downsample=not is_final_block,  # 是否添加下采样
                dtype=self.dtype,  # 数据类型
            )
            down_blocks.append(down_block)  # 将下采样块添加到列表
        self.down_blocks = down_blocks  # 保存下采样块

        # 中间层
        self.mid_block = FlaxUNetMidBlock2D(
            in_channels=block_out_channels[-1],  # 输入通道数为最后一个块的输出通道
            resnet_groups=self.norm_num_groups,  # 归一化组数
            num_attention_heads=None,  # 注意力头数（未使用）
            dtype=self.dtype,  # 数据类型
        )

        # 结束层
        conv_out_channels = 2 * self.out_channels if self.double_z else self.out_channels  # 输出通道数
        self.conv_norm_out = nn.GroupNorm(num_groups=self.norm_num_groups, epsilon=1e-6)  # 归一化层
        self.conv_out = nn.Conv(
            conv_out_channels,  # 输出通道数
            kernel_size=(3, 3),  # 卷积核大小
            strides=(1, 1),  # 步幅
            padding=((1, 1), (1, 1)),  # 填充方式
            dtype=self.dtype,  # 数据类型
        )

    # 前向传播方法
    def __call__(self, sample, deterministic: bool = True):
        # 输入层处理
        sample = self.conv_in(sample)  # 对输入样本应用卷积

        # 下采样处理
        for block in self.down_blocks:  # 遍历下采样块
            sample = block(sample, deterministic=deterministic)  # 处理样本

        # 中间层处理
        sample = self.mid_block(sample, deterministic=deterministic)  # 对样本应用中间块

        # 结束层处理
        sample = self.conv_norm_out(sample)  # 应用归一化
        sample = nn.swish(sample)  # 使用 Swish 激活函数
        sample = self.conv_out(sample)  # 应用最后的卷积层

        return sample  # 返回处理后的样本
# 定义 FlaxDecoder 类，继承自 nn.Module
class FlaxDecoder(nn.Module):
    r"""
    Flax 实现的 VAE 解码器。

    该模型是 Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    的子类。可将其作为常规 Flax linen 模块使用，并参考 Flax 文档以了解所有相关的
    使用和行为。

    最后，该模型支持固有的 JAX 特性，例如：
    - [即时编译 (JIT)](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [自动微分](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [向量化](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [并行化](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    参数：
        in_channels (:obj:`int`, *可选*, 默认为 3):
            输入通道数
        out_channels (:obj:`int`, *可选*, 默认为 3):
            输出通道数
        up_block_types (:obj:`Tuple[str]`, *可选*, 默认为 `(UpDecoderBlock2D)`):
            UpDecoder 块类型
        block_out_channels (:obj:`Tuple[str]`, *可选*, 默认为 `(64,)`):
            包含每个块输出通道数量的元组
        layers_per_block (:obj:`int`, *可选*, 默认为 `2`):
            每个块的 Resnet 层数量
        norm_num_groups (:obj:`int`, *可选*, 默认为 `32`):
            规范的组数量
        act_fn (:obj:`str`, *可选*, 默认为 `silu`):
            激活函数
        double_z (:obj:`bool`, *可选*, 默认为 `False`):
            是否加倍最后的输出通道数
        dtype (:obj:`jnp.dtype`, *可选*, 默认为 jnp.float32):
            参数的 `dtype`
    """

    # 定义输入通道数，默认为 3
    in_channels: int = 3
    # 定义输出通道数，默认为 3
    out_channels: int = 3
    # 定义 UpDecoder 块类型，默认为一个元组，包含 "UpDecoderBlock2D"
    up_block_types: Tuple[str] = ("UpDecoderBlock2D",)
    # 定义每个块的输出通道数量，默认为一个元组，包含 64
    block_out_channels: int = (64,)
    # 定义每个块的层数，默认为 2
    layers_per_block: int = 2
    # 定义规范的组数量，默认为 32
    norm_num_groups: int = 32
    # 定义激活函数，默认为 "silu"
    act_fn: str = "silu"
    # 定义参数的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 初始化设置方法
    def setup(self):
        # 获取输出通道数
        block_out_channels = self.block_out_channels

        # 输入层，将 z 转换为 block_in
        self.conv_in = nn.Conv(
            block_out_channels[-1],  # 输入通道数为输出通道数列表的最后一个元素
            kernel_size=(3, 3),  # 卷积核大小为 3x3
            strides=(1, 1),  # 步幅为 1
            padding=((1, 1), (1, 1)),  # 上下左右各填充 1 像素
            dtype=self.dtype,  # 数据类型
        )

        # 中间层
        self.mid_block = FlaxUNetMidBlock2D(
            in_channels=block_out_channels[-1],  # 输入通道数为输出通道数列表的最后一个元素
            resnet_groups=self.norm_num_groups,  # 归一化组数
            num_attention_heads=None,  # 注意力头数设为 None
            dtype=self.dtype,  # 数据类型
        )

        # 上采样
        reversed_block_out_channels = list(reversed(block_out_channels))  # 反转输出通道数列表
        output_channel = reversed_block_out_channels[0]  # 当前输出通道数为反转列表的第一个元素
        up_blocks = []  # 初始化上采样块列表
        for i, _ in enumerate(self.up_block_types):  # 遍历上采样块类型
            prev_output_channel = output_channel  # 保存前一个输出通道数
            output_channel = reversed_block_out_channels[i]  # 更新当前输出通道数

            is_final_block = i == len(block_out_channels) - 1  # 检查是否为最后一个块

            # 创建上采样解码块
            up_block = FlaxUpDecoderBlock2D(
                in_channels=prev_output_channel,  # 输入通道数为前一个输出通道数
                out_channels=output_channel,  # 输出通道数
                num_layers=self.layers_per_block + 1,  # 层数为每个块的层数加一
                resnet_groups=self.norm_num_groups,  # 归一化组数
                add_upsample=not is_final_block,  # 如果不是最后一个块则添加上采样
                dtype=self.dtype,  # 数据类型
            )
            up_blocks.append(up_block)  # 将上采样块添加到列表
            prev_output_channel = output_channel  # 更新前一个输出通道数

        self.up_blocks = up_blocks  # 将上采样块列表赋值给实例变量

        # 结束层
        self.conv_norm_out = nn.GroupNorm(num_groups=self.norm_num_groups, epsilon=1e-6)  # 归一化层
        self.conv_out = nn.Conv(
            self.out_channels,  # 输出通道数
            kernel_size=(3, 3),  # 卷积核大小为 3x3
            strides=(1, 1),  # 步幅为 1
            padding=((1, 1), (1, 1)),  # 上下左右各填充 1 像素
            dtype=self.dtype,  # 数据类型
        )

    # 前向传播方法
    def __call__(self, sample, deterministic: bool = True):
        # 将 z 转换为 block_in
        sample = self.conv_in(sample)  # 通过输入卷积层处理样本

        # 中间层
        sample = self.mid_block(sample, deterministic=deterministic)  # 通过中间块处理样本

        # 上采样
        for block in self.up_blocks:  # 遍历所有上采样块
            sample = block(sample, deterministic=deterministic)  # 处理样本

        sample = self.conv_norm_out(sample)  # 通过归一化层处理样本
        sample = nn.swish(sample)  # 应用 Swish 激活函数
        sample = self.conv_out(sample)  # 通过输出卷积层处理样本

        return sample  # 返回处理后的样本
# 定义一个类表示对角高斯分布
class FlaxDiagonalGaussianDistribution(object):
    # 初始化函数，接受参数和一个可选的确定性标志
    def __init__(self, parameters, deterministic=False):
        # 将参数拆分为均值和对数方差，最后一维用于通道最后的情况
        self.mean, self.logvar = jnp.split(parameters, 2, axis=-1)
        # 限制对数方差在-30到20之间
        self.logvar = jnp.clip(self.logvar, -30.0, 20.0)
        # 设置确定性标志
        self.deterministic = deterministic
        # 计算标准差
        self.std = jnp.exp(0.5 * self.logvar)
        # 计算方差
        self.var = jnp.exp(self.logvar)
        # 如果是确定性模式，则将方差和标准差设置为均值的零张量
        if self.deterministic:
            self.var = self.std = jnp.zeros_like(self.mean)

    # 从分布中采样
    def sample(self, key):
        # 使用均值和标准差生成样本
        return self.mean + self.std * jax.random.normal(key, self.mean.shape)

    # 计算KL散度
    def kl(self, other=None):
        # 如果是确定性模式，返回零
        if self.deterministic:
            return jnp.array([0.0])

        # 如果没有提供其他分布，计算与标准正态分布的KL散度
        if other is None:
            return 0.5 * jnp.sum(self.mean**2 + self.var - 1.0 - self.logvar, axis=[1, 2, 3])

        # 计算两个分布之间的KL散度
        return 0.5 * jnp.sum(
            jnp.square(self.mean - other.mean) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar,
            axis=[1, 2, 3],
        )

    # 计算负对数似然
    def nll(self, sample, axis=[1, 2, 3]):
        # 如果是确定性模式，返回零
        if self.deterministic:
            return jnp.array([0.0])

        # 计算2π的对数
        logtwopi = jnp.log(2.0 * jnp.pi)
        # 计算负对数似然
        return 0.5 * jnp.sum(logtwopi + self.logvar + jnp.square(sample - self.mean) / self.var, axis=axis)

    # 返回分布的众数
    def mode(self):
        return self.mean


# 使用装饰器将类注册到配置中
@flax_register_to_config
# 定义一个Flax自编码器类，使用KL损失解码潜在表示
class FlaxAutoencoderKL(nn.Module, FlaxModelMixin, ConfigMixin):
    r"""
    Flax实现的变分自编码器（VAE）模型，带有KL损失以解码潜在表示。

    该模型继承自[`FlaxModelMixin`]。请查看超类文档以了解所有模型的通用方法
    （如下载或保存）。

    该模型是Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    子类。将其用作常规Flax Linen模块，并参考Flax文档了解其
    一般用法和行为。

    该模型支持JAX的固有特性，例如以下内容：

    - [即时（JIT）编译](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [自动微分](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [矢量化](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [并行化](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)
    # 参数说明
        Parameters:
            in_channels (`int`, *optional*, defaults to 3):  # 输入图像的通道数，默认为3
                Number of channels in the input image.
            out_channels (`int`, *optional*, defaults to 3):  # 输出图像的通道数，默认为3
                Number of channels in the output.
            down_block_types (`Tuple[str]`, *optional*, defaults to `(DownEncoderBlock2D)`):  # 下采样模块类型的元组，默认为 DownEncoderBlock2D
                Tuple of downsample block types.
            up_block_types (`Tuple[str]`, *optional*, defaults to `(UpDecoderBlock2D)`):  # 上采样模块类型的元组，默认为 UpDecoderBlock2D
                Tuple of upsample block types.
            block_out_channels (`Tuple[str]`, *optional*, defaults to `(64,)`):  # 每个模块的输出通道数的元组，默认为 64
                Tuple of block output channels.
            layers_per_block (`int`, *optional*, defaults to `2`):  # 每个模块中的 ResNet 层数，默认为 2
                Number of ResNet layer for each block.
            act_fn (`str`, *optional*, defaults to `silu`):  # 使用的激活函数，默认为 silu
                The activation function to use.
            latent_channels (`int`, *optional*, defaults to `4`):  # 潜在空间中的通道数，默认为 4
                Number of channels in the latent space.
            norm_num_groups (`int`, *optional*, defaults to `32`):  # 归一化的组数，默认为 32
                The number of groups for normalization.
            sample_size (`int`, *optional*, defaults to 32):  # 输入样本的大小，默认为 32
                Sample input size.
            scaling_factor (`float`, *optional*, defaults to 0.18215):  # 用于缩放潜在空间的标准差，默认为 0.18215
                The component-wise standard deviation of the trained latent space computed using the first batch of the
                training set. This is used to scale the latent space to have unit variance when training the diffusion
                model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
                diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
                / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
                Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
            dtype (`jnp.dtype`, *optional*, defaults to `jnp.float32`):  # 参数的数据类型，默认为 jnp.float32
                The `dtype` of the parameters.
        """  # 结束参数说明
    
        in_channels: int = 3  # 定义输入通道数，默认值为3
        out_channels: int = 3  # 定义输出通道数，默认值为3
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",)  # 定义下采样模块类型，默认使用 DownEncoderBlock2D
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",)  # 定义上采样模块类型，默认使用 UpDecoderBlock2D
        block_out_channels: Tuple[int] = (64,)  # 定义模块输出通道数，默认为 64
        layers_per_block: int = 1  # 定义每个模块的 ResNet 层数，默认为 1
        act_fn: str = "silu"  # 定义激活函数，默认为 silu
        latent_channels: int = 4  # 定义潜在空间通道数，默认为 4
        norm_num_groups: int = 32  # 定义归一化组数，默认为 32
        sample_size: int = 32  # 定义样本输入大小，默认为 32
        scaling_factor: float = 0.18215  # 定义缩放因子，默认为 0.18215
        dtype: jnp.dtype = jnp.float32  # 定义参数数据类型，默认为 jnp.float32
    # 设置模型的编码器和解码器等组件
    def setup(self):
        # 初始化编码器，配置输入和输出通道及其他参数
        self.encoder = FlaxEncoder(
            in_channels=self.config.in_channels,
            out_channels=self.config.latent_channels,
            down_block_types=self.config.down_block_types,
            block_out_channels=self.config.block_out_channels,
            layers_per_block=self.config.layers_per_block,
            act_fn=self.config.act_fn,
            norm_num_groups=self.config.norm_num_groups,
            double_z=True,
            dtype=self.dtype,
        )
        # 初始化解码器，配置输入和输出通道及其他参数
        self.decoder = FlaxDecoder(
            in_channels=self.config.latent_channels,
            out_channels=self.config.out_channels,
            up_block_types=self.config.up_block_types,
            block_out_channels=self.config.block_out_channels,
            layers_per_block=self.config.layers_per_block,
            norm_num_groups=self.config.norm_num_groups,
            act_fn=self.config.act_fn,
            dtype=self.dtype,
        )
        # 初始化量化卷积，配置输入通道和卷积参数
        self.quant_conv = nn.Conv(
            2 * self.config.latent_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            dtype=self.dtype,
        )
        # 初始化后量化卷积，配置输入通道和卷积参数
        self.post_quant_conv = nn.Conv(
            self.config.latent_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            dtype=self.dtype,
        )

    # 初始化权重，返回冻结的参数字典
    def init_weights(self, rng: jax.Array) -> FrozenDict:
        # 初始化输入张量，设置样本形状
        sample_shape = (1, self.in_channels, self.sample_size, self.sample_size)
        sample = jnp.zeros(sample_shape, dtype=jnp.float32)

        # 将随机数生成器分割为参数、丢弃和高斯随机数生成器
        params_rng, dropout_rng, gaussian_rng = jax.random.split(rng, 3)
        rngs = {"params": params_rng, "dropout": dropout_rng, "gaussian": gaussian_rng}

        # 初始化并返回参数
        return self.init(rngs, sample)["params"]

    # 编码样本，返回潜在分布
    def encode(self, sample, deterministic: bool = True, return_dict: bool = True):
        # 调整样本维度顺序
        sample = jnp.transpose(sample, (0, 2, 3, 1))

        # 使用编码器生成隐藏状态
        hidden_states = self.encoder(sample, deterministic=deterministic)
        # 通过量化卷积处理隐藏状态
        moments = self.quant_conv(hidden_states)
        # 创建潜在分布
        posterior = FlaxDiagonalGaussianDistribution(moments)

        # 根据 return_dict 决定返回的格式
        if not return_dict:
            return (posterior,)

        return FlaxAutoencoderKLOutput(latent_dist=posterior)

    # 解码潜在变量，返回生成的样本
    def decode(self, latents, deterministic: bool = True, return_dict: bool = True):
        # 检查潜在变量的通道数，必要时调整维度顺序
        if latents.shape[-1] != self.config.latent_channels:
            latents = jnp.transpose(latents, (0, 2, 3, 1))

        # 通过后量化卷积处理潜在变量
        hidden_states = self.post_quant_conv(latents)
        # 使用解码器生成隐藏状态
        hidden_states = self.decoder(hidden_states, deterministic=deterministic)

        # 调整隐藏状态维度顺序
        hidden_states = jnp.transpose(hidden_states, (0, 3, 1, 2))

        # 根据 return_dict 决定返回的格式
        if not return_dict:
            return (hidden_states,)

        return FlaxDecoderOutput(sample=hidden_states)
    # 定义一个可调用的函数，用于处理样本，带有一些可选参数
        def __call__(self, sample, sample_posterior=False, deterministic: bool = True, return_dict: bool = True):
            # 编码输入样本，获取后验分布，参数控制编码行为
            posterior = self.encode(sample, deterministic=deterministic, return_dict=return_dict)
            # 如果需要样本后验分布
            if sample_posterior:
                # 创建一个高斯分布的随机数生成器
                rng = self.make_rng("gaussian")
                # 从后验分布中采样隐状态
                hidden_states = posterior.latent_dist.sample(rng)
            else:
                # 获取后验分布的模态值作为隐状态
                hidden_states = posterior.latent_dist.mode()
    
            # 解码隐状态，返回解码后的样本
            sample = self.decode(hidden_states, return_dict=return_dict).sample
    
            # 如果不需要以字典形式返回结果
            if not return_dict:
                # 返回解码后的样本元组
                return (sample,)
    
            # 返回一个包含解码样本的输出对象
            return FlaxDecoderOutput(sample=sample)
```