# `.\lucidrains\vit-pytorch\vit_pytorch\vivit.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn

# 从 einops 库中导入 rearrange, repeat, reduce 函数
from einops import rearrange, repeat, reduce
# 从 einops.layers.torch 库中导入 Rearrange 类
from einops.layers.torch import Rearrange

# 辅助函数

# 判断值是否存在的函数
def exists(val):
    return val is not None

# 将输入转换为元组的函数
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# 类

# 前馈神经网络类
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        # 定义神经网络结构
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# 注意力机制类
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 变换器类
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# 视觉变换器类
class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        dim,
        spatial_depth,
        temporal_depth,
        heads,
        mlp_dim,
        pool = 'cls',
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.
    ):
        # 调用父类的构造函数
        super().__init__()
        # 解构图像尺寸和图像块尺寸
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        # 断言图像高度和宽度能够被图像块高度和宽度整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        # 断言帧数能够被帧块大小整除
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        # 计算图像块数量和帧块数量
        num_image_patches = (image_height // patch_height) * (image_width // patch_width)
        num_frame_patches = (frames // frame_patch_size)

        # 计算图像块维度
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        # 断言池化类型为'cls'或'mean'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # 根据池化类型设置是否使用全局平均池化
        self.global_average_pool = pool == 'mean'

        # 定义将图像块转换为嵌入向量的层序列
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f (h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        # 初始化位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frame_patches, num_image_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # 初始化空间和时间的CLS token参数
        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None

        # 初始化空间和时间的Transformer模型
        self.spatial_transformer = Transformer(dim, spatial_depth, heads, dim_head, mlp_dim, dropout)
        self.temporal_transformer = Transformer(dim, temporal_depth, heads, dim_head, mlp_dim, dropout)

        # 设置池化类型和转换为潜在空间的层
        self.pool = pool
        self.to_latent = nn.Identity()

        # 定义MLP头部
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, video):
        # 将视频转换为图像块嵌入向量
        x = self.to_patch_embedding(video)
        b, f, n, _ = x.shape

        # 添加位置嵌入
        x = x + self.pos_embedding[:, :f, :n]

        # 如果存在空间CLS token，则添加到输入中
        if exists(self.spatial_cls_token):
            spatial_cls_tokens = repeat(self.spatial_cls_token, '1 1 d -> b f 1 d', b = b, f = f)
            x = torch.cat((spatial_cls_tokens, x), dim = 2)

        # 应用Dropout
        x = self.dropout(x)

        # 重排张量形状以便空间注意力
        x = rearrange(x, 'b f n d -> (b f) n d')

        # 在空间上进行注意力计算
        x = self.spatial_transformer(x)

        # 重排张量形状以便后续处理
        x = rearrange(x, '(b f) n d -> b f n d', b = b)

        # 剔除空间CLS token或进行全局平均池化以便时间注意力
        x = x[:, :, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b f d', 'mean')

        # 如果存在时间CLS token，则添加到输入中
        if exists(self.temporal_cls_token):
            temporal_cls_tokens = repeat(self.temporal_cls_token, '1 1 d-> b 1 d', b = b)
            x = torch.cat((temporal_cls_tokens, x), dim = 1)

        # 在时间上进行注意力计算
        x = self.temporal_transformer(x)

        # 剔除时间CLS token或进行全局平均池化
        x = x[:, 0] if not self.global_average_pool else reduce(x, 'b f d -> b d', 'mean')

        # 转换为潜在空间并返回MLP头部的输出
        x = self.to_latent(x)
        return self.mlp_head(x)
```