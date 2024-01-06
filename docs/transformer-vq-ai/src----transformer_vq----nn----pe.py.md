# `transformer_vq\src\transformer_vq\nn\pe.py`

```
# 导入必要的模块
import dataclasses  # 用于创建不可变的类
import chex  # 用于进行数组形状的检查
import flax.linen as nn  # 用于构建神经网络模型
import jax.numpy as jnp  # 用于进行数值计算

from transformer_vq.nn.types import TransformerConfig  # 从指定模块中导入指定类型

# 定义一个函数，用于生成正弦和余弦的嵌入
def get_sinusoid_embs(length, width, lam, flip, start=0):
    # 生成位置序列
    pos_seq = start + jnp.arange(length)
    # 检查位置序列的形状是否符合要求
    chex.assert_shape(pos_seq, [length])
    # 计算指数衰减率的倒数
    inv_lams = 1 / (lam ** (jnp.arange(0, width, 2) / width))
    # 计算正弦值
    pre = pos_seq[..., None] * inv_lams[None, ...]
    sin = jnp.sin(pre)
    # 计算余弦值
    cos = jnp.cos(pre)
    # 将正弦值和余弦值拼接在一起
    cat = jnp.concatenate([sin, cos], axis=-1)
    # 检查拼接后的数组形状是否符合要求
    chex.assert_shape(cat, [length, width])
    # 如果不需要翻转，则返回拼接后的数组
    if not flip:
        return cat
    # 返回沿着指定轴翻转输入数组
    return jnp.flip(cat, axis=0)


class ScaledSin(nn.Module):
    # 参考 w. hua 等人的论文，年份为 2022
    config: TransformerConfig

    def setup(self):
        # 应用配置参数
        self.apply_config()
        # 创建一个名为 scale 的参数，初始值为 self.b_init，数据类型为 jnp.float32
        self.scale = self.param("scale", self.b_init, [], jnp.float32)

    def apply_config(self):
        # 将配置参数应用到模块中
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    def __call__(self, length, offset):
        # 获取正弦嵌入向量，长度为 length，起始位置为 offset，宽度为 self.d_model，lam 为 self.pe_lam，不翻转
        embs = get_sinusoid_embs(
            length=length, start=offset, width=self.d_model, lam=self.pe_lam, flip=False
        )
        # 返回缩放后的正弦嵌入向量，数据类型转换为 self.dtype
        return (self.scale * embs).astype(self.dtype)
抱歉，我无法为您提供代码注释，因为您没有提供需要注释的代码。如果您有任何代码需要解释，请提供给我，我将竭诚为您服务。
```