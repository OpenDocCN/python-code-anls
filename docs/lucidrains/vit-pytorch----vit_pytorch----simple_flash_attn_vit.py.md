# `.\lucidrains\vit-pytorch\vit_pytorch\simple_flash_attn_vit.py`

```
# 导入必要的库
from collections import namedtuple
from packaging import version

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

# 定义常量
Config = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# 定义辅助函数

# 将输入转换为元组
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# 生成二维位置编码的正弦和余弦值
def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# 主类

class Attend(nn.Module):
    def __init__(self, use_flash = False):
        super().__init__()
        self.use_flash = use_flash
        assert not (use_flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # 确定 CUDA 和 CPU 的高效注意力配置

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            self.cuda_config = Config(True, False, False)
        else:
            self.cuda_config = Config(False, True, True)

    def flash_attn(self, q, k, v):
        config = self.cuda_config if q.is_cuda else self.cpu_config

        # Flash Attention - https://arxiv.org/abs/2205.14135
        
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(q, k, v)

        return out

    def forward(self, q, k, v):
        n, device, scale = q.shape[-2], q.device, q.shape[-1] ** -0.5

        if self.use_flash:
            return self.flash_attn(q, k, v)

        # 相似度

        sim = einsum("b h i d, b j d -> b h i j", q, k) * scale

        # 注意力

        attn = sim.softmax(dim=-1)

        # 聚合值

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        return out

# 类

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, use_flash = True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = Attend(use_flash = use_flash)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    # 初始化 Transformer 模型
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, use_flash):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个空的层列表
        self.layers = nn.ModuleList([])
        # 根据深度循环创建多个 Transformer 层
        for _ in range(depth):
            # 每个 Transformer 层包含注意力机制和前馈神经网络
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, use_flash = use_flash),
                FeedForward(dim, mlp_dim)
            ]))
    
    # 前向传播函数
    def forward(self, x):
        # 遍历每个 Transformer 层
        for attn, ff in self.layers:
            # 使用注意力机制处理输入，并将结果与输入相加
            x = attn(x) + x
            # 使用前馈神经网络处理输入，并将结果与输入相加
            x = ff(x) + x
        # 返回处理后的结果
        return x
# 定义一个简单的ViT模型，继承自nn.Module类
class SimpleViT(nn.Module):
    # 初始化函数，接收一系列参数来构建ViT模型
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, use_flash = True):
        super().__init__()
        # 获取图像的高度和宽度
        image_height, image_width = pair(image_size)
        # 获取patch的高度和宽度
        patch_height, patch_width = pair(patch_size)

        # 断言图像的高度和宽度必须能够被patch的高度和宽度整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # 计算patch的数量
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # 计算每个patch的维度
        patch_dim = channels * patch_height * patch_width

        # 将图像转换为patch嵌入
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # 创建Transformer模块
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, use_flash)

        # 将嵌入转换为潜在空间
        self.to_latent = nn.Identity()
        # 线性头部，用于分类
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    # 前向传播函数，接收图像作为输入
    def forward(self, img):
        # 获取图像的形状和数据类型
        *_, h, w, dtype = *img.shape, img.dtype

        # 将图像转换为patch嵌入
        x = self.to_patch_embedding(img)
        # 生成位置编码
        pe = posemb_sincos_2d(x)
        # 将位置编码加到嵌入中
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        # 经过Transformer模块处理
        x = self.transformer(x)
        # 对所有patch的输出取平均值
        x = x.mean(dim = 1)

        # 转换为潜在空间
        x = self.to_latent(x)
        # 使用线性头部进行分类
        return self.linear_head(x)
```