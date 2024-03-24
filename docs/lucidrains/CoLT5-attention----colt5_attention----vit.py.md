# `.\lucidrains\CoLT5-attention\colt5_attention\vit.py`

```
import torch
from torch import nn

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange, Reduce

from colt5_attention.transformer_block import (
    ConditionalRoutedImageAttention,
    ConditionalRoutedFeedForward
)

# helpers

# 定义一个函数，如果输入参数是元组则返回元组，否则返回元组包含输入参数的元组
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# 定义一个函数，生成二维位置编码的正弦和余弦值
def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    # 获取 patches 的形状信息
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    # 生成网格坐标
    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    # 确保特征维度是4的倍数
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    # 计算 omega 值
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    # 计算位置编码
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    pe = pe.type(dtype)
    return rearrange(pe, '(h w) d -> h w d', h = h, w = w)

# classes

# 定义一个 Transformer 类
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        attn_num_heavy_tokens_q,
        attn_num_heavy_tokens_kv,
        attn_light_dim_head,
        attn_light_heads,
        attn_light_window_size,
        attn_heavy_dim_head,
        attn_heavy_heads,
        ff_num_heavy_tokens,
        ff_light_mult,
        ff_heavy_mult,
        router_straight_through = True,
        router_kwargs: dict = {},
        router_use_triton = False,
        flash_attn = True,
        attn_num_routed_kv = 1
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):

            # 创建 ConditionalRoutedFeedForward 实例
            ff = ConditionalRoutedFeedForward(
                dim,
                num_heavy_tokens = ff_num_heavy_tokens,
                light_ff_mult = ff_light_mult,
                heavy_ff_mult = ff_heavy_mult,
                router_straight_through = router_straight_through,
                router_kwargs = router_kwargs,
                use_triton = router_use_triton
            )

            # 创建 ConditionalRoutedImageAttention 实例
            attn = ConditionalRoutedImageAttention(
                dim,
                num_heavy_tokens_q = attn_num_heavy_tokens_q,
                num_heavy_tokens_kv = attn_num_heavy_tokens_kv,
                num_routed_kv = attn_num_routed_kv,
                light_dim_head = attn_light_dim_head,
                light_heads = attn_light_heads,
                light_window_size = attn_light_window_size,
                heavy_dim_head = attn_heavy_dim_head,
                heavy_heads = attn_heavy_heads,
                router_straight_through = router_straight_through,
                router_kwargs = router_kwargs,
                use_triton = router_use_triton,
                use_flash_attn = flash_attn,
                channel_first = False,
                use_null_q_tokens = True
            )

            self.layers.append(nn.ModuleList([attn, ff]))

    # 前向传播函数
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x

            x, ps = pack([x], 'b * d')
            x = ff(x) + x            
            x, = unpack(x, ps, 'b * d')

        return x

# 定义一个 ConditionalRoutedViT 类
class ConditionalRoutedViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        attn_num_heavy_tokens_q,
        attn_num_heavy_tokens_kv,
        attn_heavy_dim_head,
        attn_heavy_heads,
        attn_light_dim_head,
        attn_light_heads,
        attn_light_window_size,
        ff_num_heavy_tokens,
        ff_heavy_mult,
        ff_light_mult,
        channels = 3,
        router_straight_through = True,
        router_kwargs: dict = {},
        router_use_triton = False,
        flash_attn = True,
        attn_num_routed_kv = 1,
        default_coor_descent_eps = 1.
    # 定义一个继承自 nn.Module 的类，用于实现图像的分块处理和Transformer处理
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 获取图像的高度和宽度
        image_height, image_width = pair(image_size)
        # 获取分块的高度和宽度
        patch_height, patch_width = pair(patch_size)

        # 断言图像的高度和宽度能够被分块的高度和宽度整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # 计算分块的数量
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # 计算每个分块的维度
        patch_dim = channels * patch_height * patch_width

        # 定义一个序列模块，用于将图像分块转换为嵌入向量
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # 设置路由器参数，包括epsilon值
        router_kwargs = {'eps': default_coor_descent_eps, **router_kwargs}

        # 创建Transformer模块
        self.transformer = Transformer(
            dim,
            depth,
            attn_num_heavy_tokens_q,
            attn_num_heavy_tokens_kv,
            attn_light_dim_head,
            attn_light_heads,
            attn_light_window_size,
            attn_heavy_dim_head,
            attn_heavy_heads,
            ff_num_heavy_tokens,
            ff_light_mult,
            ff_heavy_mult,
            router_straight_through,
            router_kwargs,
            router_use_triton,
            flash_attn,
            attn_num_routed_kv
        )

        # 定义一个线性头部模块，用于分类
        self.linear_head = nn.Sequential(
            Reduce('b h w c -> b c', 'mean'),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    # 前向传播函数
    def forward(self, img):
        # 获取图像的高度、宽度和数据类型
        *_, h, w, dtype = *img.shape, img.dtype

        # 将图像转换为嵌入向量
        x = self.to_patch_embedding(img)
        # 添加位置编码
        x = x + posemb_sincos_2d(x)        

        # 使用Transformer处理嵌入向量
        x = self.transformer(x)

        # 使用线性头部进行分类
        return self.linear_head(x)
```