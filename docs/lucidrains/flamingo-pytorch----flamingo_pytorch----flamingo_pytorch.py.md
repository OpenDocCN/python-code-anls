# `.\lucidrains\flamingo-pytorch\flamingo_pytorch\flamingo_pytorch.py`

```py
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many

def exists(val):
    return val is not None

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),  # 对输入进行 Layer Normalization
        nn.Linear(dim, inner_dim, bias = False),  # 线性变换，将输入维度转换为 inner_dim
        nn.GELU(),  # GELU 激活函数
        nn.Linear(inner_dim, dim, bias = False)  # 线性变换，将 inner_dim 转换为 dim
    )

class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)  # 对媒体数据进行 Layer Normalization
        self.norm_latents = nn.LayerNorm(dim)  # 对潜在数据进行 Layer Normalization

        self.to_q = nn.Linear(dim, inner_dim, bias = False)  # 线性变换，将输入维度转换为 inner_dim
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)  # 线性变换，将输入维度转换为 inner_dim * 2
        self.to_out = nn.Linear(inner_dim, dim, bias = False)  # 线性变换，将 inner_dim 转换为 dim

    def forward(self, x, latents):
        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """
        x = self.norm_media(x)  # 对媒体数据进行 Layer Normalization
        latents = self.norm_latents(latents)  # 对潜在数据进行 Layer Normalization

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(latents)  # 对潜在数据进行线性变换

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim = -2)  # 拼接媒体数据和潜在数据
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)  # 将拼接后的数据进行线性变换并分割为 key 和 value

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h = h)  # 重排数据维度

        q = q * self.scale  # 缩放 q

        # attention

        sim = einsum('... i d, ... j d  -> ... i j', q, k)  # 计算注意力分数

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()  # 对注意力分数进行处理
        attn = sim.softmax(dim = -1)  # 计算注意力权��

        out = einsum('... i j, ... j d -> ... i d', attn, v)  # 计算输出
        out = rearrange(out, 'b h t n d -> b t n (h d)', h = h)  # 重排输出维度
        return self.to_out(out)  # 返回输出数据

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        num_media_embeds = 4,
        ff_mult = 4
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))  # 初始化潜在数据
        self.media_pos_emb = nn.Parameter(torch.randn(num_media_embeds, 1, dim))  # 初始化媒体位置嵌入

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),  # 添加 PerceiverAttention 层
                FeedForward(dim = dim, mult = ff_mult)  # 添加 FeedForward 层
            ]))

        self.norm = nn.LayerNorm(dim)  # 对数据进行 Layer Normalization

    def forward(self, x):
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')  # 重排输入数据维度

        times = x.shape[1]
        x = x + self.media_pos_emb[:times]  # 将媒体位置嵌入加到输入数据上

        latents = repeat(self.latents, 'n d -> b m n d', b = x.shape[0], m = x.shape[1])  # 重复潜在数据

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents  # 使用 PerceiverAttention 层
            latents = ff(latents) + latents  # 使用 FeedForward 层

        return self.norm(latents)  # 对输出数据进行 Layer Normalization

# gated cross attention

class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        only_attend_immediate_media = True
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)  # 对数据进行 Layer Normalization

        self.to_q = nn.Linear(dim, inner_dim, bias = False)  # 线性变换，将输入维度转换为 inner_dim
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)  # 线性变换，将输入维度转换为 inner_dim * 2
        self.to_out = nn.Linear(inner_dim, dim, bias = False)  # 线性变换，将 inner_dim 转换为 dim

        # whether for text to only attend to immediate preceding image, or all images

        self.only_attend_immediate_media = only_attend_immediate_media  # 是否只关注紧邻的图像

    def forward(
        self,
        x,
        media,
        media_locations = None
        ):
            # 获取媒体数据的形状信息
            b, t, m = media.shape[:3]
            # 获取头数
            h = self.heads

            # 对输入进行归一化处理
            x = self.norm(x)

            # 将输入转换为查询向量
            q = self.to_q(x)
            # 重新排列媒体数据的维度
            media = rearrange(media, 'b t n d -> b (t n) d')

            # 将媒体数据转换为键值对
            k, v = self.to_kv(media).chunk(2, dim = -1)
            # 重新排列多个张量的维度
            q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = h)

            # 对查询向量进行缩放
            q = q * self.scale

            # 计算查询向量和键向量之间的相似度
            sim = einsum('... i d, ... j d -> ... i j', q, k)

            if exists(media_locations):
                # 计算文本时间
                text_time = media_locations.cumsum(dim = -1) # 在每个 True 布尔值处，增加时间计数器（相对于媒体时间）
                media_time = torch.arange(t, device = x.device) + 1

                # 如果只关注最近的图像，则文本时间必须等于媒体时间
                # 否则，只要文本时间大于媒体时间（如果关注所有先前的图像/媒体）
                mask_op = torch.eq if self.only_attend_immediate_media else torch.ge

                # 创建文本到媒体的掩码
                text_to_media_mask = mask_op(rearrange(text_time, 'b i -> b 1 i 1'), repeat(media_time, 'j -> 1 1 1 (j m)', m = m))
                sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

            # 对相似度进行归一化处理
            sim = sim - sim.amax(dim = -1, keepdim = True).detach()
            attn = sim.softmax(dim = -1)

            if exists(media_locations) and self.only_attend_immediate_media:
                # 需要将没有前置媒体的文本的注意力置零
                text_without_media_mask = text_time == 0
                text_without_media_mask = rearrange(text_without_media_mask, 'b i -> b 1 i 1')
                attn = attn.masked_fill(text_without_media_mask, 0.)

            # 计算输出
            out = einsum('... i j, ... j d -> ... i d', attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            return self.to_out(out)
# 定义一个 GatedCrossAttentionBlock 类，继承自 nn.Module
class GatedCrossAttentionBlock(nn.Module):
    # 初始化函数，接受一些参数
    def __init__(
        self,
        *,
        dim,                    # 输入维度
        dim_head = 64,          # 每个头的维度
        heads = 8,              # 多头注意力的头数
        ff_mult = 4,            # FeedForward 层的倍数
        only_attend_immediate_media = True  # 是否只关注直接媒体
    ):
        super().__init__()
        # 创建 MaskedCrossAttention 对象，用于计算交叉注意力
        self.attn = MaskedCrossAttention(dim = dim, dim_head = dim_head, heads = heads, only_attend_immediate_media = only_attend_immediate_media)
        # 创建一个可学习的参数，用于门控交叉注意力
        self.attn_gate = nn.Parameter(torch.tensor([0.]))

        # 创建 FeedForward 对象，用于前馈神经网络
        self.ff = FeedForward(dim, mult = ff_mult)
        # 创建一个可学习的参数，用于门控前馈神经网络
        self.ff_gate = nn.Parameter(torch.tensor([0.]))

    # 前向传播函数
    def forward(
        self,
        x,                      # 输入张量
        media,                  # 媒体张量，由感知器重新采样编码 - (batch, time, latents, dim)
        media_locations = None  # 表示媒体位置的布尔张量 - (batch, sequence)
    ):
        # 计算交叉注意力并应用门控
        x = self.attn(x, media, media_locations = media_locations) * self.attn_gate.tanh() + x
        # 应用前馈神经网络并应用门控
        x = self.ff(x) * self.ff_gate.tanh()  + x
        # 返回结果张量
        return x
```