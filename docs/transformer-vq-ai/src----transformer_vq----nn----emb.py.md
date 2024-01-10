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

    # 初始化函数，用于设置模型的参数
    def setup(self):
        # 调用apply_config方法，将config中的属性设置为类的属性
        self.apply_config()
        # 初始化词嵌入矩阵的参数
        emb_args = [self.e_init, [self.n_vocab, self.d_model], self.param_dtype]
        self.embs = self.param("embs", *emb_args)
        # 初始化输出偏置的参数
        bias_out_args = [self.b_init, [self.n_vocab], self.param_dtype]
        self.bias_out = self.param("bias_out", *bias_out_args)

    # 将config中的属性设置为类的属性
    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    # 类的调用方法，接受一个输入x，返回经过处理的x
    def __call__(self, x):
        # 从词嵌入矩阵中取出对应索引的词向量
        x = jnp.take_along_axis(
            self.embs[None, ...], x[..., None].astype(jnp.int32), axis=1
        )
        return x.astype(self.dtype)

    # 计算输出的logits值
    def logits(self, x):
        # 将输入x转换为float32类型
        x = x.astype(jnp.float32)
        # 计算logits值
        x = jnp.dot(x, self.embs.T.astype(jnp.float32))
        x += self.bias_out.astype(jnp.float32)[None, None, ...]
        return x
```