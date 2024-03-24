# `.\lucidrains\vit-pytorch\vit_pytorch\vit_3d.py`

```
import torch  # 导入 PyTorch 库
from torch import nn  # 从 PyTorch 库中导入 nn 模块

from einops import rearrange, repeat  # 从 einops 库中导入 rearrange 和 repeat 函数
from einops.layers.torch import Rearrange  # 从 einops 库中导入 Torch 版的 Rearrange 类

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)  # 如果 t 是元组则返回 t，否则返回 (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),  # 对输入进行 Layer Normalization
            nn.Linear(dim, hidden_dim),  # 线性变换
            nn.GELU(),  # GELU 激活函数
            nn.Dropout(dropout),  # Dropout 层
            nn.Linear(hidden_dim, dim),  # 线性变换
            nn.Dropout(dropout)  # Dropout 层
        )
    def forward(self, x):
        return self.net(x)  # 前向传播

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)  # 对输入进行 Layer Normalization
        self.attend = nn.Softmax(dim = -1)  # Softmax 层
        self.dropout = nn.Dropout(dropout)  # Dropout 层

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)  # 线性变换，用于计算 Q、K、V

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  # 线性变换
            nn.Dropout(dropout)  # Dropout 层
        ) if project_out else nn.Identity()  # 如果 project_out 为真则使用 nn.Sequential，否则使用 nn.Identity

    def forward(self, x):
        x = self.norm(x)  # Layer Normalization
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # 将线性变换后的结果切分成 Q、K、V
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)  # 重排 Q、K、V 的维度

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # 计算 Q、K 的点积

        attn = self.attend(dots)  # 注意力权重
        attn = self.dropout(attn)  # Dropout

        out = torch.matmul(attn, v)  # 加权求和
        out = rearrange(out, 'b h n d -> b n (h d)')  # 重排输出维度
        return self.to_out(out)  # 返回输出

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),  # 注意力层
                FeedForward(dim, mlp_dim, dropout = dropout)  # 前馈神经网络层
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # 注意力层输出与输入相加
            x = ff(x) + x  # 前馈神经网络层输出与输入相加
        return x  # 返回输出

class ViT(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),  # 重排图像补丁的维度
            nn.LayerNorm(patch_dim),  # 对输入进行 Layer Normalization
            nn.Linear(patch_dim, dim),  # 线性变换
            nn.LayerNorm(dim),  # 对输入进行 Layer Normalization
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 位置编码
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 类别标记
        self.dropout = nn.Dropout(emb_dropout)  # Dropout 层

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)  # Transformer 模块

        self.pool = pool  # 池化方式
        self.to_latent = nn.Identity()  # 转换为潜在空间的恒等映射

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),  # 对输入进行 Layer Normalization
            nn.Linear(dim, num_classes)  # 线性变换
        )  # MLP 头部
    # 前向传播函数，接收视频数据作为输入
    def forward(self, video):
        # 将视频数据转换为补丁嵌入
        x = self.to_patch_embedding(video)
        # 获取批量大小、补丁数量和嵌入维度
        b, n, _ = x.shape

        # 重复类别标记以匹配批量大小
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # 将类别标记与补丁嵌入拼接在一起
        x = torch.cat((cls_tokens, x), dim=1)
        # 添加位置嵌入到输入中
        x += self.pos_embedding[:, :(n + 1)]
        # 对输入进行 dropout 处理
        x = self.dropout(x)

        # 使用 Transformer 处理输入数据
        x = self.transformer(x)

        # 根据池化方式计算输出
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # 将输出转换为潜在空间
        x = self.to_latent(x)
        # 使用 MLP 头部处理潜在空间的输出
        return self.mlp_head(x)
```