# `transformer_vq\src\transformer_vq\nn\pe.py`

```py
# 导入必要的模块
import dataclasses
import chex
import flax.linen as nn
import jax.numpy as jnp
from transformer_vq.nn.types import TransformerConfig

# 定义函数，生成 sinusoid 嵌入
def get_sinusoid_embs(length, width, lam, flip, start=0):
    # 生成位置序列
    pos_seq = start + jnp.arange(length)
    # 检查位置序列的形状
    chex.assert_shape(pos_seq, [length])
    # 计算频率的倒数
    inv_lams = 1 / (lam ** (jnp.arange(0, width, 2) / width))
    # 计算正弦值和余弦值
    pre = pos_seq[..., None] * inv_lams[None, ...]
    sin = jnp.sin(pre)
    cos = jnp.cos(pre)
    # 拼接正弦值和余弦值
    cat = jnp.concatenate([sin, cos], axis=-1)
    # 检查拼接后的形状
    chex.assert_shape(cat, [length, width])
    # 如果不需要翻转，直接返回拼接后的值
    if not flip:
        return cat
    # 如果需要翻转，返回翻转后的值
    return jnp.flip(cat, axis=0)

# 定义一个类 ScaledSin，继承自 nn.Module
class ScaledSin(nn.Module):
    # 类的配置信息
    config: TransformerConfig

    # 初始化函数
    def setup(self):
        # 应用配置信息
        self.apply_config()
        # 初始化缩放参数
        self.scale = self.param("scale", self.b_init, [], jnp.float32)

    # 应用配置信息的函数
    def apply_config(self):
        # 遍历配置信息，将其作为类的属性
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    # 类的调用函数
    def __call__(self, length, offset):
        # 生成 sinusoid 嵌入
        embs = get_sinusoid_embs(
            length=length, start=offset, width=self.d_model, lam=self.pe_lam, flip=False
        )
        # 返回缩放后的嵌入
        return (self.scale * embs).astype(self.dtype)
```