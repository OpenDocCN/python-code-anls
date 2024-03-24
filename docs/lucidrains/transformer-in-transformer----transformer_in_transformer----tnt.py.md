# `.\lucidrains\transformer-in-transformer\transformer_in_transformer\tnt.py`

```py
# 导入 torch 库
import torch
# 导入 torch 中的函数库
import torch.nn.functional as F
# 从 torch 中导入 nn 和 einsum 模块
from torch import nn, einsum

# 从 einops 中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat
# 从 einops.layers.torch 中导入 Rearrange 类

# 辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 如果值存在则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 判断值是否可以被除数整除
def divisible_by(val, divisor):
    return (val % divisor) == 0

# 计算展开后的输出尺寸
def unfold_output_size(image_size, kernel_size, stride, padding):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)

# 类

# 预处理层
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        # 使用 LayerNorm 对输入进行归一化
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        # 对输入进行归一化后，传入下一层处理
        return self.fn(self.norm(x), **kwargs)

# 前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        # 神经网络结构：全连接层 -> GELU 激活函数 -> Dropout -> 全连接层
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        # 前馈神经网络的前向传播
        return self.net(x)

# 注意力机制
class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads =  heads
        self.scale = dim_head ** -0.5

        # 将输入转换为查询、键、值
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        # 输出层结构：全连接层 -> Dropout
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, d, h = *x.shape, self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        # 计算注意力分数
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        # 计算输出
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# 主类

class TNT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_dim,
        pixel_dim,
        patch_size,
        pixel_size,
        depth,
        num_classes,
        channels = 3,
        heads = 8,
        dim_head = 64,
        ff_dropout = 0.,
        attn_dropout = 0.,
        unfold_args = None
    # 初始化函数，设置模型参数
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 检查图像大小是否能被分块大小整除
        assert divisible_by(image_size, patch_size), 'image size must be divisible by patch size'
        # 检查分块大小是否能被像素大小整除
        assert divisible_by(patch_size, pixel_size), 'patch size must be divisible by pixel size for now'

        # 计算分块令牌的数量
        num_patch_tokens = (image_size // patch_size) ** 2

        # 设置模型参数
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_tokens = nn.Parameter(torch.randn(num_patch_tokens + 1, patch_dim))

        # 设置默认的展开参数
        unfold_args = default(unfold_args, (pixel_size, pixel_size, 0))
        unfold_args = (*unfold_args, 0) if len(unfold_args) == 2 else unfold_args
        kernel_size, stride, padding = unfold_args

        # 计算像素宽度和像素数量
        pixel_width = unfold_output_size(patch_size, kernel_size, stride, padding)
        num_pixels = pixel_width ** 2

        # 定义将像素转换为令牌的模块
        self.to_pixel_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> (b h w) c p1 p2', p1 = patch_size, p2 = patch_size),
            nn.Unfold(kernel_size = kernel_size, stride = stride, padding = padding),
            Rearrange('... c n -> ... n c'),
            nn.Linear(channels * kernel_size ** 2, pixel_dim)
        )

        # 初始化分块位置编码和像素位置编码
        self.patch_pos_emb = nn.Parameter(torch.randn(num_patch_tokens + 1, patch_dim))
        self.pixel_pos_emb = nn.Parameter(torch.randn(num_pixels, pixel_dim))

        # 创建模型层
        layers = nn.ModuleList([])
        for _ in range(depth):

            # 定义将像素转换为分块的模块
            pixel_to_patch = nn.Sequential(
                nn.LayerNorm(pixel_dim),
                Rearrange('... n d -> ... (n d)'),
                nn.Linear(pixel_dim * num_pixels, patch_dim),
            )

            # 添加模型层
            layers.append(nn.ModuleList([
                PreNorm(pixel_dim, Attention(dim = pixel_dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                PreNorm(pixel_dim, FeedForward(dim = pixel_dim, dropout = ff_dropout)),
                pixel_to_patch,
                PreNorm(patch_dim, Attention(dim = patch_dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                PreNorm(patch_dim, FeedForward(dim = patch_dim, dropout = ff_dropout)),
            ]))

        # 设置模型层和 MLP 头部
        self.layers = layers
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, num_classes)
        )

    # 前向传播函数
    def forward(self, x):
        # 获取输入张量的形状和模型参数
        b, _, h, w, patch_size, image_size = *x.shape, self.patch_size, self.image_size
        # 检查输入的高度和宽度是否能被分块大小整除
        assert divisible_by(h, patch_size) and divisible_by(w, patch_size), f'height {h} and width {w} of input must be divisible by the patch size'

        # 计算分块的数量
        num_patches_h = h // patch_size
        num_patches_w = w // patch_size
        n = num_patches_w * num_patches_h

        # 将输入张量转换为像素令牌和分块令牌
        pixels = self.to_pixel_tokens(x)
        patches = repeat(self.patch_tokens[:(n + 1)], 'n d -> b n d', b = b)

        # 添加分块位置编码和像素位置编码
        patches += rearrange(self.patch_pos_emb[:(n + 1)], 'n d -> () n d')
        pixels += rearrange(self.pixel_pos_emb, 'n d -> () n d')

        # 遍历模型层，进行注意力和前馈计算
        for pixel_attn, pixel_ff, pixel_to_patch_residual, patch_attn, patch_ff in self.layers:

            pixels = pixel_attn(pixels) + pixels
            pixels = pixel_ff(pixels) + pixels

            patches_residual = pixel_to_patch_residual(pixels)

            patches_residual = rearrange(patches_residual, '(b h w) d -> b (h w) d', h = num_patches_h, w = num_patches_w)
            patches_residual = F.pad(patches_residual, (0, 0, 1, 0), value = 0) # cls token gets residual of 0
            patches = patches + patches_residual

            patches = patch_attn(patches) + patches
            patches = patch_ff(patches) + patches

        # 提取分类令牌并通过 MLP 头部进行分类预测
        cls_token = patches[:, 0]
        return self.mlp_head(cls_token)
```