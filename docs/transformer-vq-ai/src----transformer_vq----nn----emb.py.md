# `transformer_vq\src\transformer_vq\nn\emb.py`

```
# 导入必要的模块
import dataclasses
import flax.linen as nn
import jax.numpy as jnp
from transformer_vq.nn.types import TransformerConfig

# 定义一个名为Embeddings的类，继承自nn.Module
class Embeddings(nn.Module):
    # 类的构造函数，接受一个TransformerConfig类型的参数config
    config: TransformerConfig

    # 初始化函数，用于设置类的初始状态
    def setup(self):
        # 调用apply_config方法，应用配置
        self.apply_config()
        # 初始化嵌入层的参数
        emb_args = [self.e_init, [self.n_vocab, self.d_model], self.param_dtype]
        self.embs = self.param("embs", *emb_args)
        # 初始化偏置参数
        bias_out_args = [self.b_init, [self.n_vocab], self.param_dtype]
        self.bias_out = self.param("bias_out", *bias_out_args)

    # 应用配置的方法
    def apply_config(self):
        # 遍历config对象的属性和值
        for k, v in dataclasses.asdict(self.config).items():
    # 设置对象的属性，将属性名和属性值作为参数传入
    setattr(self, k, v)

    # 定义对象的调用方法，接受参数 x
    def __call__(self, x):
        # 从 self.embs 中取出指定索引的元素，构成新的数组
        x = jnp.take_along_axis(
            self.embs[None, ...], x[..., None].astype(jnp.int32), axis=1
        )
        # 将数组转换为指定的数据类型
        return x.astype(self.dtype)

    # 定义对象的 logits 方法，接受参数 x
    def logits(self, x):
        # 将数组转换为 jnp.float32 类型
        x = x.astype(jnp.float32)
        # 计算两个数组的矩阵乘法
        x = jnp.dot(x, self.embs.T.astype(jnp.float32))
        # 将偏置项加到 x 上
        x += self.bias_out.astype(jnp.float32)[None, None, ...]
        return x
```