# `.\diffusers\models\controlnet_flax.py`

```py
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入类型提示
from typing import Optional, Tuple, Union

# 导入 Flax 库
import flax
import flax.linen as nn
# 导入 JAX 库
import jax
import jax.numpy as jnp
# 从 Flax 导入冻结字典
from flax.core.frozen_dict import FrozenDict

# 导入配置相关的工具
from ..configuration_utils import ConfigMixin, flax_register_to_config
# 导入基础输出类
from ..utils import BaseOutput
# 导入时间步嵌入和时间步类
from .embeddings_flax import FlaxTimestepEmbedding, FlaxTimesteps
# 导入 Flax 模型混合类
from .modeling_flax_utils import FlaxModelMixin
# 导入 UNet 2D 块
from .unets.unet_2d_blocks_flax import (
    FlaxCrossAttnDownBlock2D,
    FlaxDownBlock2D,
    FlaxUNetMidBlock2DCrossAttn,
)

# 定义 FlaxControlNetOutput 数据类，继承自 BaseOutput
@flax.struct.dataclass
class FlaxControlNetOutput(BaseOutput):
    """
    The output of [`FlaxControlNetModel`].
    该类表示 FlaxControlNetModel 的输出。

    Args:
        down_block_res_samples (`jnp.ndarray`): 下层块的结果样本
        mid_block_res_sample (`jnp.ndarray`): 中间块的结果样本
    """

    # 定义下层块结果样本的类型
    down_block_res_samples: jnp.ndarray
    # 定义中间块结果样本的类型
    mid_block_res_sample: jnp.ndarray


# 定义 FlaxControlNetConditioningEmbedding 模块
class FlaxControlNetConditioningEmbedding(nn.Module):
    # 定义输入的条件嵌入通道数
    conditioning_embedding_channels: int
    # 定义每个块的输出通道数
    block_out_channels: Tuple[int, ...] = (16, 32, 96, 256)
    # 定义数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 设置模块的组件
    def setup(self) -> None:
        # 创建输入卷积层，输出通道数为第一个块的输出通道数
        self.conv_in = nn.Conv(
            self.block_out_channels[0],
            kernel_size=(3, 3),  # 卷积核大小
            padding=((1, 1), (1, 1)),  # 填充方式
            dtype=self.dtype,  # 数据类型
        )

        # 初始化块列表
        blocks = []
        # 遍历每对相邻的块输出通道数
        for i in range(len(self.block_out_channels) - 1):
            # 获取当前输入通道数
            channel_in = self.block_out_channels[i]
            # 获取下一个输出通道数
            channel_out = self.block_out_channels[i + 1]
            # 创建第一个卷积层
            conv1 = nn.Conv(
                channel_in,  # 输入通道数
                kernel_size=(3, 3),  # 卷积核大小
                padding=((1, 1), (1, 1)),  # 填充方式
                dtype=self.dtype,  # 数据类型
            )
            # 将卷积层添加到块列表
            blocks.append(conv1)
            # 创建第二个卷积层，带有步幅
            conv2 = nn.Conv(
                channel_out,  # 输出通道数
                kernel_size=(3, 3),  # 卷积核大小
                strides=(2, 2),  # 步幅
                padding=((1, 1), (1, 1)),  # 填充方式
                dtype=self.dtype,  # 数据类型
            )
            # 将卷积层添加到块列表
            blocks.append(conv2)
        # 将所有块存储为类的属性
        self.blocks = blocks

        # 创建输出卷积层，输出通道数为条件嵌入通道数
        self.conv_out = nn.Conv(
            self.conditioning_embedding_channels,  # 输出通道数
            kernel_size=(3, 3),  # 卷积核大小
            padding=((1, 1), (1, 1)),  # 填充方式
            kernel_init=nn.initializers.zeros_init(),  # 权重初始化为零
            bias_init=nn.initializers.zeros_init(),  # 偏置初始化为零
            dtype=self.dtype,  # 数据类型
        )
    # 定义调用方法，接收条件输入并返回处理后的嵌入
        def __call__(self, conditioning: jnp.ndarray) -> jnp.ndarray:
            # 通过输入卷积层处理条件输入，生成嵌入
            embedding = self.conv_in(conditioning)
            # 应用 SiLU 激活函数到嵌入
            embedding = nn.silu(embedding)
    
            # 遍历所有块进行嵌入的逐层处理
            for block in self.blocks:
                # 通过当前块处理嵌入
                embedding = block(embedding)
                # 再次应用 SiLU 激活函数到嵌入
                embedding = nn.silu(embedding)
    
            # 通过输出卷积层处理嵌入，得到最终结果
            embedding = self.conv_out(embedding)
    
            # 返回最终处理后的嵌入
            return embedding
# 注册类到 Flax 配置管理
@flax_register_to_config
# 定义 FlaxControlNetModel 类，继承自 nn.Module, FlaxModelMixin 和 ConfigMixin
class FlaxControlNetModel(nn.Module, FlaxModelMixin, ConfigMixin):
    r"""
    一个 ControlNet 模型。

    该模型继承自 [`FlaxModelMixin`]。请查看超类文档以了解它为所有模型实现的通用方法
    （例如下载或保存）。

    此模型也是 Flax Linen [`flax.linen.Module`](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    的子类。可以将其用作常规的 Flax Linen 模块，并参考 Flax 文档以了解与其
    一般用法和行为相关的所有事项。

    支持 JAX 的固有特性，例如：

    - [即时编译 (JIT)](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [自动微分](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [向量化](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [并行化](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    参数:
        sample_size (`int`, *可选*):
            输入样本的大小。
        in_channels (`int`, *可选*, 默认为 4):
            输入样本中的通道数。
        down_block_types (`Tuple[str]`, *可选*, 默认为 `("FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxDownBlock2D")`):
            使用的下采样块的元组。
        block_out_channels (`Tuple[int]`, *可选*, 默认为 `(320, 640, 1280, 1280)`):
            每个块的输出通道元组。
        layers_per_block (`int`, *可选*, 默认为 2):
            每个块的层数。
        attention_head_dim (`int` 或 `Tuple[int]`, *可选*, 默认为 8):
            注意力头的维度。
        num_attention_heads (`int` 或 `Tuple[int]`, *可选*):
            注意力头的数量。
        cross_attention_dim (`int`, *可选*, 默认为 768):
            跨注意力特征的维度。
        dropout (`float`, *可选*, 默认为 0):
            下采样、上采样和瓶颈块的 dropout 概率。
        flip_sin_to_cos (`bool`, *可选*, 默认为 `True`):
            是否在时间嵌入中将 sin 转换为 cos。
        freq_shift (`int`, *可选*, 默认为 0): 应用于时间嵌入的频率偏移。
        controlnet_conditioning_channel_order (`str`, *可选*, 默认为 `rgb`):
            条件图像的通道顺序。如果是 `bgr`，将转换为 `rgb`。
        conditioning_embedding_out_channels (`tuple`, *可选*, 默认为 `(16, 32, 96, 256)`):
            `conditioning_embedding` 层中每个块的输出通道元组。
    """

    # 设置输入样本的默认大小
    sample_size: int = 32
    # 设置输入样本的默认通道数
    in_channels: int = 4
    # 定义下采样块的类型元组，包括三次 CrossAttnDownBlock2D 和一次 DownBlock2D
    down_block_types: Tuple[str, ...] = (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    )
    # 定义是否仅使用交叉注意力，默认为 False
    only_cross_attention: Union[bool, Tuple[bool, ...]] = False
    # 定义每个块的输出通道数的元组
    block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280)
    # 定义每个块的层数，默认为 2
    layers_per_block: int = 2
    # 定义注意力头的维度，默认为 8
    attention_head_dim: Union[int, Tuple[int, ...]] = 8
    # 可选的注意力头数量，默认为 None
    num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None
    # 定义交叉注意力的维度，默认为 1280
    cross_attention_dim: int = 1280
    # 定义 dropout 概率，默认为 0.0
    dropout: float = 0.0
    # 定义是否使用线性投影，默认为 False
    use_linear_projection: bool = False
    # 定义数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 定义是否翻转正弦和余弦的布尔值，默认为 True
    flip_sin_to_cos: bool = True
    # 定义频率偏移，默认为 0
    freq_shift: int = 0
    # 定义 ControlNet 条件通道的顺序，默认为 "rgb"
    controlnet_conditioning_channel_order: str = "rgb"
    # 定义条件嵌入输出通道数的元组
    conditioning_embedding_out_channels: Tuple[int, ...] = (16, 32, 96, 256)

    # 初始化权重的方法，接收一个随机数生成器
    def init_weights(self, rng: jax.Array) -> FrozenDict:
        # 初始化输入张量的形状
        sample_shape = (1, self.in_channels, self.sample_size, self.sample_size)
        # 创建一个全零的样本张量
        sample = jnp.zeros(sample_shape, dtype=jnp.float32)
        # 创建一个全为 1 的时间步张量
        timesteps = jnp.ones((1,), dtype=jnp.int32)
        # 创建一个全零的编码器隐藏状态张量
        encoder_hidden_states = jnp.zeros((1, 1, self.cross_attention_dim), dtype=jnp.float32)
        # 创建 ControlNet 条件的形状
        controlnet_cond_shape = (1, 3, self.sample_size * 8, self.sample_size * 8)
        # 创建一个全零的 ControlNet 条件张量
        controlnet_cond = jnp.zeros(controlnet_cond_shape, dtype=jnp.float32)

        # 将 rng 分成两个部分，一个用于参数，一个用于 dropout
        params_rng, dropout_rng = jax.random.split(rng)
        # 创建一个包含参数和 dropout 随机数生成器的字典
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 调用初始化方法，返回包含参数的字典
        return self.init(rngs, sample, timesteps, encoder_hidden_states, controlnet_cond)["params"]

    # 定义可调用方法，接收样本、时间步、编码器隐藏状态等参数
    def __call__(
        self,
        sample: jnp.ndarray,
        timesteps: Union[jnp.ndarray, float, int],
        encoder_hidden_states: jnp.ndarray,
        controlnet_cond: jnp.ndarray,
        # 定义条件缩放因子，默认为 1.0
        conditioning_scale: float = 1.0,
        # 定义是否返回字典，默认为 True
        return_dict: bool = True,
        # 定义是否处于训练模式，默认为 False
        train: bool = False,
```