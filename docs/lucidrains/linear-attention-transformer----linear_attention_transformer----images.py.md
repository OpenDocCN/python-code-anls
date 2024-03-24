# `.\lucidrains\linear-attention-transformer\linear_attention_transformer\images.py`

```py
import torch
from torch import nn

class ImageLinearAttention(nn.Module):
    # 定义图像线性注意力模块
    def __init__(self, chan, chan_out = None, kernel_size = 1, padding = 0, stride = 1, key_dim = 64, value_dim = 64, heads = 8, norm_queries = True):
        super().__init__()
        self.chan = chan
        # 如果输出通道数未指定，则设置为输入通道数
        chan_out = chan if chan_out is None else chan_out

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.heads = heads

        self.norm_queries = norm_queries

        conv_kwargs = {'padding': padding, 'stride': stride}
        # 创建用于查询的卷积层
        self.to_q = nn.Conv2d(chan, key_dim * heads, kernel_size, **conv_kwargs)
        # 创建用于键的卷积层
        self.to_k = nn.Conv2d(chan, key_dim * heads, kernel_size, **conv_kwargs)
        # 创建用于值的卷积层
        self.to_v = nn.Conv2d(chan, value_dim * heads, kernel_size, **conv_kwargs)

        out_conv_kwargs = {'padding': padding}
        # 创建输出卷积层
        self.to_out = nn.Conv2d(value_dim * heads, chan_out, kernel_size, **out_conv_kwargs)

    def forward(self, x, context = None):
        b, c, h, w, k_dim, heads = *x.shape, self.key_dim, self.heads

        q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))

        q, k, v = map(lambda t: t.reshape(b, heads, -1, h * w), (q, k, v))

        q, k = map(lambda x: x * (self.key_dim ** -0.25), (q, k))

        if context is not None:
            context = context.reshape(b, c, 1, -1)
            ck, cv = self.to_k(context), self.to_v(context)
            ck, cv = map(lambda t: t.reshape(b, heads, k_dim, -1), (ck, cv))
            k = torch.cat((k, ck), dim=3)
            v = torch.cat((v, cv), dim=3)

        k = k.softmax(dim=-1)

        if self.norm_queries:
            q = q.softmax(dim=-2)

        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhdn,bhde->bhen', q, context)
        out = out.reshape(b, -1, h, w)
        out = self.to_out(out)
        return out
```