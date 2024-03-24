# `.\lucidrains\isab-pytorch\isab_pytorch\isab_pytorch.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块和 einsum 函数
from torch import nn, einsum
# 从 torch.nn.functional 中导入 F 模块
import torch.nn.functional as F

# 从 einops 库中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat

# 定义一个辅助函数，用于检查值是否存在
def exists(val):
    return val is not None

# 定义一个注意力机制类
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        and_self_attend = False
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.and_self_attend = and_self_attend

        # 定义将输入转换为查询向量的线性层
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        # 定义将输入转换为键值对的线性层
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        # 定义将输出转换为最终输出的线性层
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        context,
        mask = None
    ):
        h, scale = self.heads, self.scale

        if self.and_self_attend:
            # 如果需要自注意力机制，则将上下文信息与输入拼接在一起
            context = torch.cat((x, context), dim = -2)

            if exists(mask):
                # 对 mask 进行填充，使其与输入的维度相匹配
                mask = F.pad(mask, (x.shape[-2], 0), value = True)

        # 将输入 x 转换为查询向量 q，将上下文信息转换为键值对 k 和 v
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        # 将查询向量 q、键 k、值 v 重排维度，以适应注意力计算
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        # 计算点积注意力得分
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale

        if exists(mask):
            # 对注意力得分进行 mask 处理
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b n -> b 1 1 n')
            dots.masked_fill_(~mask, mask_value)

        # 对注意力得分进行 softmax 操作，得到注意力权重
        attn = dots.softmax(dim = -1)
        # 根据注意力权重计算输出
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # 重排输出维度，返回最终输出
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

# 定义一个独立的多头自注意力块类
class ISAB(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        num_latents = None,
        latent_self_attend = False
    ):
        super().__init__()
        # 如果存在 latents 数量，则初始化为随机张量，否则为 None
        self.latents = nn.Parameter(torch.randn(num_latents, dim)) if exists(num_latents) else None
        # 定义第一个注意力机制，用于处理 latents 和输入 x
        self.attn1 = Attention(dim, heads, and_self_attend = latent_self_attend)
        # 定义第二个注意力机制，用于处理输入 x 和 latents
        self.attn2 = Attention(dim, heads)

    def forward(self, x, latents = None, mask = None):
        b, *_ = x.shape

        # 确保 latents 参数存在性与 latents 属性的一致性
        assert exists(latents) ^ exists(self.latents), 'you can only either learn the latents within the module, or pass it in externally'
        latents = latents if exists(latents) else self.latents

        if latents.ndim == 2:
            # 如果 latents 是二维张量，则重复扩展为与输入 x 相同的 batch 维度
            latents = repeat(latents, 'n d -> b n d', b = b)

        # 使用第一个注意力机制处理 latents 和输入 x，得到 latents
        latents = self.attn1(latents, x, mask = mask)
        # 使用第二个注意力机制处理输入 x 和 latents，得到输出
        out     = self.attn2(x, latents)

        return out, latents
```