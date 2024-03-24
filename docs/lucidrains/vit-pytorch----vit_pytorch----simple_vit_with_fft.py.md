# `.\lucidrains\vit-pytorch\vit_pytorch\simple_vit_with_fft.py`

```py
# 导入 torch 库
import torch
# 从 torch.fft 中导入 fft2 函数
from torch.fft import fft2
# 从 torch 中导入 nn 模块
from torch import nn

# 从 einops 库中导入 rearrange、reduce、pack、unpack 函数
from einops import rearrange, reduce, pack, unpack
# 从 einops.layers.torch 中导入 Rearrange 类

# 辅助函数

# 如果输入 t 是元组，则返回 t，否则返回 (t, t)
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# 生成二维位置编码的函数
def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    # 生成网格坐标
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    # 确保特征维度是4的倍数
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    # 计算 omega
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    # 计算位置编码
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# 类

# 前馈神经网络类
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

# 注意力机制类
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Transformer 类
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# SimpleViT 类
class SimpleViT(nn.Module):
    # 初始化函数，设置模型参数
    def __init__(self, *, image_size, patch_size, freq_patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        # 调用父类初始化函数
        super().__init__()
        # 获取图像的高度和宽度
        image_height, image_width = pair(image_size)
        # 获取 patch 的高度和宽度
        patch_height, patch_width = pair(patch_size)
        # 获取频域 patch 的高度和宽度
        freq_patch_height, freq_patch_width = pair(freq_patch_size)

        # 断言图像的高度和宽度能够被 patch 的高度和宽度整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        # 断言图像的高度和宽度能够被频域 patch 的高度和宽度整除
        assert image_height % freq_patch_height == 0 and image_width % freq_patch_width == 0, 'Image dimensions must be divisible by the freq patch size.'

        # 计算 patch 的维度
        patch_dim = channels * patch_height * patch_width
        # 计算频域 patch 的维度
        freq_patch_dim = channels * 2 * freq_patch_height * freq_patch_width

        # 将图像转换为 patch 的嵌入向量
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # 将频域 patch 转换为嵌入向量
        self.to_freq_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) ri -> b (h w) (p1 p2 ri c)", p1 = freq_patch_height, p2 = freq_patch_width),
            nn.LayerNorm(freq_patch_dim),
            nn.Linear(freq_patch_dim, dim),
            nn.LayerNorm(dim)
        )

        # 生成位置编码
        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        )

        # 生成频域位置编码
        self.freq_pos_embedding = posemb_sincos_2d(
            h = image_height // freq_patch_height,
            w = image_width // freq_patch_width,
            dim = dim
        )

        # 创建 Transformer 模型
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        # 池化方式为平均池化
        self.pool = "mean"
        # 转换为潜在空间的操作
        self.to_latent = nn.Identity()

        # 线性层，用于分类
        self.linear_head = nn.Linear(dim, num_classes)

    # 前向传播函数
    def forward(self, img):
        # 获取设备和数据类型
        device, dtype = img.device, img.dtype

        # 将图像转换为 patch 的嵌入向量
        x = self.to_patch_embedding(img)
        # 对图像进行二维傅里叶变换
        freqs = torch.view_as_real(fft2(img))

        # 将频域 patch 转换为嵌入向量
        f = self.to_freq_embedding(freqs)

        # 添加位置编码
        x += self.pos_embedding.to(device, dtype = dtype)
        f += self.freq_pos_embedding.to(device, dtype = dtype)

        # 打包数据
        x, ps = pack((f, x), 'b * d')

        # 使用 Transformer 进行特征提取
        x = self.transformer(x)

        # 解包数据
        _, x = unpack(x, ps, 'b * d')
        # 对特征进行池化操作
        x = reduce(x, 'b n d -> b d', 'mean')

        # 转换为潜在空间
        x = self.to_latent(x)
        # 使用线性层进行分类
        return self.linear_head(x)
# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 创建一个简单的ViT模型实例，指定参数包括类别数、图像大小、patch大小、频率patch大小、维度、深度、头数、MLP维度
    vit = SimpleViT(
        num_classes = 1000,
        image_size = 256,
        patch_size = 8,
        freq_patch_size = 8,
        dim = 1024,
        depth = 1,
        heads = 8,
        mlp_dim = 2048,
    )

    # 生成一个8个样本的随机张量，每个样本包含3个通道，大小为256x256
    images = torch.randn(8, 3, 256, 256)

    # 将图像输入ViT模型，得到输出logits
    logits = vit(images)
```