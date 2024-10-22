# `.\diffusers\models\embeddings_flax.py`

```py
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# 该文件的使用需要遵循 Apache 2.0 许可证
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何明示或暗示的担保或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# 查看许可证以了解特定权限和限制
import math  # 导入数学库以进行数学运算

import flax.linen as nn  # 导入Flax库中的神经网络模块
import jax.numpy as jnp  # 导入JAX的numpy模块以进行数值计算


def get_sinusoidal_embeddings(
    timesteps: jnp.ndarray,  # 定义输入参数 timesteps 为一维 JAX 数组
    embedding_dim: int,  # 定义输出嵌入的维度
    freq_shift: float = 1,  # 频率偏移的默认值为1
    min_timescale: float = 1,  # 最小时间尺度的默认值
    max_timescale: float = 1.0e4,  # 最大时间尺度的默认值
    flip_sin_to_cos: bool = False,  # 是否翻转正弦和余弦
    scale: float = 1.0,  # 缩放因子的默认值
) -> jnp.ndarray:  # 函数返回一个 JAX 数组
    """Returns the positional encoding (same as Tensor2Tensor).
    
    返回位置编码，类似于Tensor2Tensor

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
        输入为一维张量，N个索引，每个批次元素一个
        These may be fractional.
        embedding_dim: The number of output channels.
        嵌入的通道数
        min_timescale: The smallest time unit (should probably be 0.0).
        最小时间单位
        max_timescale: The largest time unit.
        最大时间单位
    Returns:
        a Tensor of timing signals [N, num_channels]
        返回时间信号的张量 [N, num_channels]
    """
    assert timesteps.ndim == 1, "Timesteps should be a 1d-array"  # 检查 timesteps 是否为一维数组
    assert embedding_dim % 2 == 0, f"Embedding dimension {embedding_dim} should be even"  # 检查嵌入维度是否为偶数
    num_timescales = float(embedding_dim // 2)  # 计算时间尺度的数量
    log_timescale_increment = math.log(max_timescale / min_timescale) / (num_timescales - freq_shift)  # 计算对数时间尺度增量
    inv_timescales = min_timescale * jnp.exp(jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment)  # 计算反时间尺度
    emb = jnp.expand_dims(timesteps, 1) * jnp.expand_dims(inv_timescales, 0)  # 计算嵌入

    # scale embeddings
    scaled_time = scale * emb  # 对嵌入进行缩放

    if flip_sin_to_cos:  # 如果需要翻转正弦和余弦
        signal = jnp.concatenate([jnp.cos(scaled_time), jnp.sin(scaled_time)], axis=1)  # 拼接余弦和正弦信号
    else:  # 否则
        signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1)  # 拼接正弦和余弦信号
    signal = jnp.reshape(signal, [jnp.shape(timesteps)[0], embedding_dim])  # 重塑信号的形状
    return signal  # 返回信号


class FlaxTimestepEmbedding(nn.Module):  # 定义时间步嵌入模块
    r"""
    Time step Embedding Module. Learns embeddings for input time steps.
    时间步嵌入模块。学习输入时间步的嵌入

    Args:
        time_embed_dim (`int`, *optional*, defaults to `32`):
                Time step embedding dimension
                时间步嵌入维度
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
                Parameters `dtype`
                参数的数据类型
    """

    time_embed_dim: int = 32  # 设置时间嵌入维度的默认值为32
    dtype: jnp.dtype = jnp.float32  # 设置参数的数据类型的默认值为jnp.float32

    @nn.compact  # 指示该方法为紧凑的神经网络模块
    def __call__(self, temb):  # 定义模块的调用方法，接收输入参数 temb
        temb = nn.Dense(self.time_embed_dim, dtype=self.dtype, name="linear_1")(temb)  # 第一个全连接层
        temb = nn.silu(temb)  # 应用Silu激活函数
        temb = nn.Dense(self.time_embed_dim, dtype=self.dtype, name="linear_2")(temb)  # 第二个全连接层
        return temb  # 返回处理后的temb


class FlaxTimesteps(nn.Module):  # 定义时间步模块
    r"""
    # 包装类，用于生成正弦时间步嵌入，详细说明见 https://arxiv.org/abs/2006.11239
    
    # 参数：
    #     dim (`int`, *可选*, 默认为 `32`):
    #             时间步嵌入的维度
        dim: int = 32  # 定义时间步嵌入的维度，默认值为 32
        flip_sin_to_cos: bool = False  # 定义是否将正弦值转换为余弦值，默认为 False
        freq_shift: float = 1  # 定义频率偏移量，默认为 1
    
        @nn.compact  # 表示这是一个紧凑模式的神经网络层，适合 JAX 使用
        def __call__(self, timesteps):  # 定义调用方法，接受时间步作为输入
            return get_sinusoidal_embeddings(  # 调用函数生成正弦嵌入
                timesteps,  # 输入的时间步
                embedding_dim=self.dim,  # 嵌入维度设置为实例属性 dim
                flip_sin_to_cos=self.flip_sin_to_cos,  # 设置是否翻转正弦到余弦
                freq_shift=self.freq_shift  # 设置频率偏移量
            )  # 返回生成的正弦嵌入
```