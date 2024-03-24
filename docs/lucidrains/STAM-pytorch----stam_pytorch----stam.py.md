# `.\lucidrains\STAM-pytorch\stam_pytorch\stam.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块和 einsum 函数
from torch import nn, einsum

# 从 einops 库中导入 rearrange 和 repeat 函数，以及 torch 模块中的 Rearrange 类
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# 定义 PreNorm 类，继承自 nn.Module 类
class PreNorm(nn.Module):
    # 初始化函数，接受维度 dim 和函数 fn 作为参数
    def __init__(self, dim, fn):
        super().__init__()
        # 初始化 LayerNorm 层
        self.norm = nn.LayerNorm(dim)
        # 将传入的函数赋值给 fn
        self.fn = fn
    # 前向传播函数，接受输入 x 和关键字参数 kwargs
    def forward(self, x, **kwargs):
        # 对输入 x 进行 LayerNorm 处理后，再传入函数 fn 进行处理
        return self.fn(self.norm(x), **kwargs)

# 定义 FeedForward 类，继承自 nn.Module 类
class FeedForward(nn.Module):
    # 初始化函数，接受维度 dim、隐藏层维度 hidden_dim 和 dropout 参数（默认为 0.）
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        # 定义神经网络结构
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    # 前向传播函数，接受输入 x
    def forward(self, x):
        # 将输入 x 传入神经网络结构中
        return self.net(x)

# 定义 Attention 类，继承自 nn.Module 类
class Attention(nn.Module):
    # 初始化函数，接受维度 dim、头数 heads、头维度 dim_head 和 dropout 参数（默认为 0.）
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        # 定义线性层，用于计算 Q、K、V
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        # 定义输出层
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    # 前向传播函数，接受输入 x
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        # 将输入 x 通过线性层得到 Q、K、V，并分割为三部分
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # 计算注意力权重
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        # 计算输出
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 定义 Transformer 类，继承自 nn.Module 类
class Transformer(nn.Module):
    # 初始化函数，接受维度 dim、层数 depth、头数 heads、头维度 dim_head、MLP维度 mlp_dim 和 dropout 参数（默认为 0.）
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        # 构建多层 Transformer 结构
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    # 前向传播函数，接受输入 x
    def forward(self, x):
        # 遍历每一层 Transformer 结构
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# 定义 STAM 类，继承自 nn.Module 类
class STAM(nn.Module):
    # 初始化函数，接受多个参数，包括维度 dim、图像大小 image_size、patch 大小 patch_size、帧数 num_frames、类别数 num_classes 等
    def __init__(
        self,
        *,
        dim,
        image_size,
        patch_size,
        num_frames,
        num_classes,
        space_depth,
        space_heads,
        space_mlp_dim,
        time_depth,
        time_heads,
        time_mlp_dim,
        space_dim_head = 64,
        time_dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        # 定义图像块到嵌入向量的映射
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b f c (h p1) (w p2) -> b f (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        # 定义位置嵌入向量
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_cls_token = nn.Parameter(torch.randn(1, dim))
        self.time_cls_token = nn.Parameter(torch.randn(1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # 定义空间 Transformer 和时间 Transformer
        self.space_transformer = Transformer(dim, space_depth, space_heads, space_dim_head, space_mlp_dim, dropout)
        self.time_transformer = Transformer(dim, time_depth, time_heads, time_dim_head, time_mlp_dim, dropout)
        self.mlp_head = nn.Linear(dim, num_classes)
    # 定义一个前向传播函数，接受视频数据作为输入
    def forward(self, video):
        # 将视频数据转换为补丁嵌入
        x = self.to_patch_embedding(video)
        b, f, n, *_ = x.shape

        # 连接空间的CLS标记

        # 重复空间的CLS标记，以匹配补丁嵌入的维度
        space_cls_tokens = repeat(self.space_cls_token, 'n d -> b f n d', b = b, f = f)
        # 在空间的CLS标记和补丁嵌入之间进行连接
        x = torch.cat((space_cls_tokens, x), dim = -2)

        # 位置嵌入

        # 添加位置嵌入到补丁嵌入中
        x += self.pos_embedding[:, :, :(n + 1)]
        # 对结果进行dropout处理
        x = self.dropout(x)

        # 空间注意力

        # 重新排列张量的维度，以便输入到空间变换器中
        x = rearrange(x, 'b f ... -> (b f) ...')
        # 使用空间变换器处理数据
        x = self.space_transformer(x)
        # 从每个帧中选择CLS标记
        x = rearrange(x[:, 0], '(b f) ... -> b f ...', b = b)

        # 连接时间的CLS标记

        # 重复时间的CLS标记，以匹配补丁嵌入的维度
        time_cls_tokens = repeat(self.time_cls_token, 'n d -> b n d', b = b)
        # 在时间的CLS标记和空间注意力结果之间进行连接
        x = torch.cat((time_cls_tokens, x), dim = -2)

        # 时间注意力

        # 使用时间变换器处理数据
        x = self.time_transformer(x)

        # 最终的多层感知机

        # 从每个样本中选择第一个元素，并通过多层感知机处理
        return self.mlp_head(x[:, 0])
```