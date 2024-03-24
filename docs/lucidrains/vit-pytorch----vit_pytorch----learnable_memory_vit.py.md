# `.\lucidrains\vit-pytorch\vit_pytorch\learnable_memory_vit.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 torch.nn 模块中导入 functional 模块
import torch.nn.functional as F

# 从 einops 库中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat
# 从 einops.layers.torch 库中导入 Rearrange 类
from einops.layers.torch import Rearrange

# 辅助函数

# 判断变量是否存在的函数
def exists(val):
    return val is not None

# 将输入转换为元组的函数
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# 控制层是否冻结的函数

# 设置模块参数是否需要梯度的函数
def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

# 冻结所有层的函数
def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

# 解冻所有层的函数
def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)

# 类

# 前馈神经网络类
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
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

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask = None, memories = None):
        x = self.norm(x)

        x_kv = x # input for key / values projection

        if exists(memories):
            # add memories to key / values if it is passed in
            memories = repeat(memories, 'n d -> b n d', b = x.shape[0]) if memories.ndim == 2 else memories
            x_kv = torch.cat((x_kv, memories), dim = 1)

        qkv = (self.to_q(x), *self.to_kv(x_kv).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if exists(attn_mask):
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Transformer 类
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, attn_mask = None, memories = None):
        for ind, (attn, ff) in enumerate(self.layers):
            layer_memories = memories[ind] if exists(memories) else None

            x = attn(x, attn_mask = attn_mask, memories = layer_memories) + x
            x = ff(x) + x
        return x

# ViT ��
class ViT(nn.Module):
    # 初始化函数，设置模型参数和结构
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        # 调用父类的初始化函数
        super().__init__()
        # 获取图像的高度和宽度
        image_height, image_width = pair(image_size)
        # 获取patch的高度和宽度
        patch_height, patch_width = pair(patch_size)

        # 断言图像的高度和宽度能够被patch的高度和宽度整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # 计算patch的数量
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # 计算每个patch的维度
        patch_dim = channels * patch_height * patch_width
        # 断言池化类型只能是'cls'或'mean'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # 定义将图像转换为patch嵌入的层
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        # 初始化位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # 初始化类别标记参数
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # 初始化dropout层
        self.dropout = nn.Dropout(emb_dropout)

        # 初始化transformer模型
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # 定义MLP头部
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    # 将图像转换为tokens
    def img_to_tokens(self, img):
        # 将图像转换为patch嵌入
        x = self.to_patch_embedding(img)

        # 重复类别标记，拼接到patch嵌入中
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = x.shape[0])
        x = torch.cat((cls_tokens, x), dim = 1)

        # 添加位置嵌入并进行dropout
        x += self.pos_embedding
        x = self.dropout(x)
        return x

    # 前向传播函数
    def forward(self, img):
        # 将图像转换为tokens
        x = self.img_to_tokens(img)        

        # 使用transformer模型处理tokens
        x = self.transformer(x)

        # 获取类别标记的输出
        cls_tokens = x[:, 0]
        return self.mlp_head(cls_tokens)
# 适配器模块，具有每层可学习的记忆、记忆 CLS 标记和可学习的适配器头部

class Adapter(nn.Module):
    def __init__(
        self,
        *,
        vit,
        num_memories_per_layer = 10,
        num_classes = 2,   
    ):
        super().__init__()
        assert isinstance(vit, ViT)

        # 提取一些需要的模型变量

        dim = vit.cls_token.shape[-1]
        layers = len(vit.transformer.layers)
        num_patches = vit.pos_embedding.shape[-2]

        self.vit = vit

        # 冻结 ViT 主干 - 只有记忆会被微调

        freeze_all_layers_(vit)

        # 可学习的参数

        self.memory_cls_token = nn.Parameter(torch.randn(dim))
        self.memories_per_layer = nn.Parameter(torch.randn(layers, num_memories_per_layer, dim))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        # 专门的注意力掩码以保留原始 ViT 的输出
        # 它允许记忆 CLS 标记关注所有其他标记（和可学习的记忆层标记），但反之亦然

        attn_mask = torch.ones((num_patches, num_patches), dtype = torch.bool)
        attn_mask = F.pad(attn_mask, (1, num_memories_per_layer), value = False)  # 主要标记不能关注每层的可学习记忆
        attn_mask = F.pad(attn_mask, (0, 0, 1, 0), value = True)                  # 记忆 CLS 标记可以关注所有内容
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, img):
        b = img.shape[0]

        tokens = self.vit.img_to_tokens(img)

        # 添加任务特定的记忆标记

        memory_cls_tokens = repeat(self.memory_cls_token, 'd -> b 1 d', b = b)
        tokens = torch.cat((memory_cls_tokens, tokens), dim = 1)        

        # 通过变压器传递记忆以及图像标记进行关注

        out = self.vit.transformer(tokens, memories = self.memories_per_layer, attn_mask = self.attn_mask)

        # 提取记忆 CLS 标记

        memory_cls_tokens = out[:, 0]

        # 通过任务特定的适配器头部传递

        return self.mlp_head(memory_cls_tokens)
```