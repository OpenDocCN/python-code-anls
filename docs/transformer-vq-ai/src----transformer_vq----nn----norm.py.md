# `transformer_vq\src\transformer_vq\nn\norm.py`

```py
# 导入必要的库
import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
# 导入自定义的数据类型 Dtype
from transformer_vq.nn.types import Dtype

# 定义 LayerNorm 类，继承自 nn.Module
class LayerNorm(nn.Module):
    # 初始化函数，接受输入维度和参数数据类型
    input_dim: int
    param_dtype: Dtype
    center: bool = False  # 默认情况下为 rms 层归一化
    norm: bool = True
    gain: bool = True
    bias: bool = True

    # 设置函数，用于初始化参数
    def setup(self):
        # 初始化参数的参数列表
        initializer_args = [[self.input_dim], self.param_dtype]
        # 如果需要 gain 参数，则初始化参数 g
        if self.gain:
            self.g = self.param("g", jax.nn.initializers.ones, *initializer_args)
        # 如果需要 bias 参数，则初始化参数 b
        if self.bias:
            self.b = self.param("b", jax.nn.initializers.zeros, *initializer_args)

    # 调用函数，实现 LayerNorm 的前向传播
    def __call__(self, x, eps=1e-6):
        # 检查输入 x 的形状是否符合要求
        chex.assert_shape(x, (..., self.input_dim))
        # 获取输入 x 的数据类型
        dtype = x.dtype
        # 将输入 x 转换为 jnp.float32 类型
        x = x.astype(jnp.float32)
        # 如果需要 center 参数，则对 x 进行中心化处理
        if self.center:
            x -= jnp.mean(x, axis=-1, keepdims=True)
        # 如果需要 norm 参数，则对 x 进行归一化处理
        if self.norm:
            x *= jax.lax.rsqrt(eps + jnp.mean(jnp.square(x), axis=-1, keepdims=True))
        # 构造广播形状
        broadcast_shape = [1 for _ in range(x.ndim - 1)] + [self.input_dim]
        # 如果需要 gain 参数，则对 x 进行缩放处理
        if self.gain:
            x *= jnp.reshape(self.g, broadcast_shape)
        # 如果需要 bias 参数，则对 x 进行偏置处理
        if self.bias:
            x += jnp.reshape(self.b, broadcast_shape)
        # 返回数据类型为 dtype 的 x
        return x.astype(dtype)
```