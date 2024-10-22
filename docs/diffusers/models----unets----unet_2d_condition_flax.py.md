# `.\diffusers\models\unets\unet_2d_condition_flax.py`

```py
# 版权声明，表明该文件的版权所有者及相关信息
# 
# 根据 Apache License 2.0 版本的许可协议
# 除非遵守该许可协议，否则不得使用本文件
# 可以在以下地址获取许可证副本
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面协议另有约定，软件在“按现状”基础上分发，
# 不提供任何明示或暗示的保证或条件
# 请参阅许可证了解管理权限和限制的具体条款
from typing import Dict, Optional, Tuple, Union  # 从 typing 模块导入类型注释工具

import flax  # 导入 flax 库用于构建神经网络
import flax.linen as nn  # 从 flax 中导入 linen 模块，方便定义神经网络层
import jax  # 导入 jax 库用于高效数值计算
import jax.numpy as jnp  # 导入 jax 的 numpy 模块，提供张量操作功能
from flax.core.frozen_dict import FrozenDict  # 从 flax 导入 FrozenDict，用于不可变字典

from ...configuration_utils import ConfigMixin, flax_register_to_config  # 导入配置相关工具
from ...utils import BaseOutput  # 导入基础输出类
from ..embeddings_flax import FlaxTimestepEmbedding, FlaxTimesteps  # 导入时间步嵌入相关类
from ..modeling_flax_utils import FlaxModelMixin  # 导入模型混合类
from .unet_2d_blocks_flax import (  # 导入 UNet 的不同构建块
    FlaxCrossAttnDownBlock2D,  # 导入交叉注意力下采样块
    FlaxCrossAttnUpBlock2D,  # 导入交叉注意力上采样块
    FlaxDownBlock2D,  # 导入下采样块
    FlaxUNetMidBlock2DCrossAttn,  # 导入中间块，带有交叉注意力
    FlaxUpBlock2D,  # 导入上采样块
)


@flax.struct.dataclass  # 使用 flax 的数据类装饰器
class FlaxUNet2DConditionOutput(BaseOutput):  # 定义 UNet 条件输出类，继承自基础输出类
    """
    [`FlaxUNet2DConditionModel`] 的输出。

    参数：
        sample (`jnp.ndarray` 的形状为 `(batch_size, num_channels, height, width)`):
            基于 `encoder_hidden_states` 输入的隐藏状态输出。模型最后一层的输出。
    """

    sample: jnp.ndarray  # 定义输出样本，数据类型为 jnp.ndarray


@flax_register_to_config  # 使用装饰器将模型注册到配置中
class FlaxUNet2DConditionModel(nn.Module, FlaxModelMixin, ConfigMixin):  # 定义条件 UNet 模型类，继承多个混合类
    r"""
    一个条件 2D UNet 模型，接收噪声样本、条件状态和时间步，并返回样本形状的输出。

    此模型继承自 [`FlaxModelMixin`]。请查看超类文档以了解其通用方法
    （例如下载或保存）。

    此模型也是 Flax Linen 的 [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    子类。将其作为常规 Flax Linen 模块使用，具体使用和行为请参阅 Flax 文档。

    支持以下 JAX 特性：
    - [即时编译 (JIT)](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [自动微分](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [向量化](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [并行化](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)
    # 参数说明部分
    Parameters:
        # 输入样本的大小，类型为整型，选填参数
        sample_size (`int`, *optional*):
            The size of the input sample.
        # 输入样本的通道数，类型为整型，默认为4
        in_channels (`int`, *optional*, defaults to 4):
            The number of channels in the input sample.
        # 输出的通道数，类型为整型，默认为4
        out_channels (`int`, *optional*, defaults to 4):
            The number of channels in the output.
        # 使用的下采样块的元组，类型为字符串元组，默认为特定的下采样块
        down_block_types (`Tuple[str]`, *optional*, defaults to `("FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxDownBlock2D")`):
            The tuple of downsample blocks to use.
        # 使用的上采样块的元组，类型为字符串元组，默认为特定的上采样块
        up_block_types (`Tuple[str]`, *optional*, defaults to `("FlaxUpBlock2D", "FlaxCrossAttnUpBlock2D", "FlaxCrossAttnUpBlock2D", "FlaxCrossAttnUpBlock2D")`):
            The tuple of upsample blocks to use.
        # UNet中间块的类型，类型为字符串，默认为"UNetMidBlock2DCrossAttn"
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock2DCrossAttn"`):
            Block type for middle of UNet, it can be one of `UNetMidBlock2DCrossAttn`. If `None`, the mid block layer
            is skipped.
        # 每个块的输出通道的元组，类型为整型元组，默认为特定的输出通道
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        # 每个块的层数，类型为整型，默认为2
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        # 注意力头的维度，可以是整型或整型元组，默认为8
        attention_head_dim (`int` or `Tuple[int]`, *optional*, defaults to 8):
            The dimension of the attention heads.
        # 注意力头的数量，可以是整型或整型元组，选填参数
        num_attention_heads (`int` or `Tuple[int]`, *optional*):
            The number of attention heads.
        # 交叉注意力特征的维度，类型为整型，默认为768
        cross_attention_dim (`int`, *optional*, defaults to 768):
            The dimension of the cross attention features.
        # dropout的概率，类型为浮点数，默认为0
        dropout (`float`, *optional*, defaults to 0):
            Dropout probability for down, up and bottleneck blocks.
        # 是否在时间嵌入中将正弦转换为余弦，类型为布尔值，默认为True
        flip_sin_to_cos (`bool`, *optional*, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        # 应用于时间嵌入的频率偏移，类型为整型，默认为0
        freq_shift (`int`, *optional*, defaults to 0): The frequency shift to apply to the time embedding.
        # 是否启用内存高效的注意力机制，类型为布尔值，默认为False
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            Enable memory efficient attention as described [here](https://arxiv.org/abs/2112.05682).
        # 是否将头维度拆分为新的轴进行自注意力计算，类型为布尔值，默认为False
        split_head_dim (`bool`, *optional*, defaults to `False`):
            Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
            enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
    """
    
    # 定义样本大小，默认为32
    sample_size: int = 32
    # 定义输入通道数，默认为4
    in_channels: int = 4
    # 定义输出通道数，默认为4
    out_channels: int = 4
    # 定义下采样块的类型元组
    down_block_types: Tuple[str, ...] = (
        "CrossAttnDownBlock2D",  # 第一个下采样块
        "CrossAttnDownBlock2D",  # 第二个下采样块
        "CrossAttnDownBlock2D",  # 第三个下采样块
        "DownBlock2D",           # 第四个下采样块
    )
    # 定义上采样块的类型元组
    up_block_types: Tuple[str, ...] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")
    # 定义中间块类型，默认为"UNetMidBlock2DCrossAttn"
    mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn"
    # 定义是否只使用交叉注意力，默认为False
    only_cross_attention: Union[bool, Tuple[bool]] = False
    # 定义每个块的输出通道元组
    block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280)
    # 每个块的层数设为 2
    layers_per_block: int = 2
    # 注意力头的维度设为 8
    attention_head_dim: Union[int, Tuple[int, ...]] = 8
    # 可选的注意力头数量，默认为 None
    num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None
    # 跨注意力的维度设为 1280
    cross_attention_dim: int = 1280
    # dropout 比率设为 0.0
    dropout: float = 0.0
    # 是否使用线性投影，默认为 False
    use_linear_projection: bool = False
    # 数据类型设为 float32
    dtype: jnp.dtype = jnp.float32
    # flip_sin_to_cos 设为 True
    flip_sin_to_cos: bool = True
    # 频移设为 0
    freq_shift: int = 0
    # 是否使用内存高效的注意力，默认为 False
    use_memory_efficient_attention: bool = False
    # 是否拆分头维度，默认为 False
    split_head_dim: bool = False
    # 每个块的变换层数设为 1
    transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1
    # 可选的附加嵌入类型，默认为 None
    addition_embed_type: Optional[str] = None
    # 可选的附加时间嵌入维度，默认为 None
    addition_time_embed_dim: Optional[int] = None
    # 附加嵌入类型的头数量设为 64
    addition_embed_type_num_heads: int = 64
    # 可选的投影类嵌入输入维度，默认为 None
    projection_class_embeddings_input_dim: Optional[int] = None

    # 初始化权重函数，接受随机数生成器作为参数
    def init_weights(self, rng: jax.Array) -> FrozenDict:
        # 初始化输入张量的形状
        sample_shape = (1, self.in_channels, self.sample_size, self.sample_size)
        # 创建全零的输入样本
        sample = jnp.zeros(sample_shape, dtype=jnp.float32)
        # 创建全一的时间步张量
        timesteps = jnp.ones((1,), dtype=jnp.int32)
        # 初始化编码器的隐藏状态
        encoder_hidden_states = jnp.zeros((1, 1, self.cross_attention_dim), dtype=jnp.float32)

        # 分割随机数生成器，用于参数和 dropout
        params_rng, dropout_rng = jax.random.split(rng)
        # 创建随机数字典
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 初始化附加条件关键字参数
        added_cond_kwargs = None
        # 判断嵌入类型是否为 "text_time"
        if self.addition_embed_type == "text_time":
            # 通过反向计算获取期望的文本嵌入维度
            is_refiner = (
                5 * self.config.addition_time_embed_dim + self.config.cross_attention_dim
                == self.config.projection_class_embeddings_input_dim
            )
            # 确定微条件的数量
            num_micro_conditions = 5 if is_refiner else 6

            # 计算文本嵌入维度
            text_embeds_dim = self.config.projection_class_embeddings_input_dim - (
                num_micro_conditions * self.config.addition_time_embed_dim
            )

            # 计算时间 ID 的通道数和维度
            time_ids_channels = self.projection_class_embeddings_input_dim - text_embeds_dim
            time_ids_dims = time_ids_channels // self.addition_time_embed_dim
            # 创建附加条件关键字参数字典
            added_cond_kwargs = {
                "text_embeds": jnp.zeros((1, text_embeds_dim), dtype=jnp.float32),
                "time_ids": jnp.zeros((1, time_ids_dims), dtype=jnp.float32),
            }
        # 返回初始化后的参数字典
        return self.init(rngs, sample, timesteps, encoder_hidden_states, added_cond_kwargs)["params"]

    # 定义调用函数，接收多个输入参数
    def __call__(
        self,
        sample: jnp.ndarray,
        timesteps: Union[jnp.ndarray, float, int],
        encoder_hidden_states: jnp.ndarray,
        # 可选的附加条件关键字参数
        added_cond_kwargs: Optional[Union[Dict, FrozenDict]] = None,
        # 可选的下块附加残差
        down_block_additional_residuals: Optional[Tuple[jnp.ndarray, ...]] = None,
        # 可选的中块附加残差
        mid_block_additional_residual: Optional[jnp.ndarray] = None,
        # 是否返回字典，默认为 True
        return_dict: bool = True,
        # 是否为训练模式，默认为 False
        train: bool = False,
```