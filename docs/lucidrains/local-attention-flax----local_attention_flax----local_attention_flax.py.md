# `.\lucidrains\local-attention-flax\local_attention_flax\local_attention_flax.py`

```
# 导入必要的库
import flax.linen as nn
from jax import numpy as np
from einops import rearrange

# 定义全局变量，用于掩码操作
ATTN_MASK_VALUE = -1e10

# 定义一个名为LocalAttention的类，继承自nn.Module
class LocalAttention(nn.Module):
    # 初始化函数，接受dim（维度）、window_size（窗口大小）、heads（头数，默认为8）、dim_head（每个头的维度，默认为64）
    dim: int
    window_size: int
    heads: int = 8
    dim_head: int = 64

    # 定义__call__方法，用于实现类的调用
    @nn.compact
    def __call__(self, x):
        # 获取输入张量x的维度信息
        n, h, dim_head, wsz = x.shape[0], self.heads, self.dim_head, self.window_size
        # 断言，确保序列长度必须能被窗口大小整除
        assert (n % wsz) == 0, 'sequence length must be divisible by the window size'
        # 计算缩放因子
        scale = dim_head ** -0.5
        # 计算窗口数量
        window = n // wsz

        # 将输入张量x通过全连接层映射为qkv
        qkv = nn.Dense(features = 3 * h * dim_head, use_bias = False)(x)
        # 将qkv分割为q、k、v
        q, k, v = np.split(qkv, 3, axis = -1)
        # 重排q、k、v的维度
        q, k, v = map(lambda t: rearrange(t, '(w n) (h d) -> h w n d', w = window, h = h), (q, k, v))

        # 对k、v进行填充
        k, v = map(lambda t: np.pad(t, ((0, 0), (1, 0), (0, 0), (0, 0)), constant_values = 0.), (k ,v))
        # 对k、v进行拼接
        k, v = map(lambda t: np.concatenate((t[:, :-1], t[:, 1:]), axis = 2), (k, v))

        # 计算注意力分数
        sim = np.einsum('h w i d, h w j d -> h w i j', q, k) * scale

        # 创建掩码
        mask = np.tril(np.ones((wsz, wsz * 2)), wsz)
        # 将掩码应用到注意力分数上
        sim = np.where(mask, sim, ATTN_MASK_VALUE)

        # 计算注意力权重
        attn = nn.softmax(sim, axis = -1)
        # 计算输出张量
        out = np.einsum('h w i j, h w j d -> h w i d', attn, v)
        # 重排输出张量的维度
        out = rearrange(out, 'h w n d -> (w n) (h d)')
        # 通过全连接层映射输出张量
        out =  nn.Dense(features = self.dim)(out)
        # 返回输出张量
        return out
```