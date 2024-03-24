# `.\lucidrains\vit-pytorch\vit_pytorch\cross_vit.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块和 einsum 函数
from torch import nn, einsum
# 从 torch.nn 模块中导入 functional 模块并重命名为 F
import torch.nn.functional as F

# 从 einops 库中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat
# 从 einops.layers.torch 库中导入 Rearrange 类
from einops.layers.torch import Rearrange

# 辅助函数

# 判断变量是否存在的函数
def exists(val):
    return val is not None

# 如果变量存在则返回该变量，否则返回默认值的函数
def default(val, d):
    return val if exists(val) else d

# 前馈神经网络

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

# 注意力机制

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

    def forward(self, x, context = None, kv_include_self = False):
        b, n, _, h = *x.shape, self.heads
        x = self.norm(x)
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context), dim = 1) # 交叉注意力需要 CLS 标记包含自身作为键/值

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Transformer 编码器，用于小和大补丁

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
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

# 投影 CLS 标记，以防小和大补丁标记具有不同的维度

class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x

# 交叉���意力 Transformer

class CrossTransformer(nn.Module):
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectInOut(sm_dim, lg_dim, Attention(lg_dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                ProjectInOut(lg_dim, sm_dim, Attention(sm_dim, heads = heads, dim_head = dim_head, dropout = dropout))
            ]))
    # 定义一个前向传播函数，接受两个输入：sm_tokens和lg_tokens
    def forward(self, sm_tokens, lg_tokens):
        # 将输入的sm_tokens和lg_tokens分别拆分为(sm_cls, sm_patch_tokens)和(lg_cls, lg_patch_tokens)
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))

        # 遍历self.layers中的每一层，每一层包含sm_attend_lg和lg_attend_sm
        for sm_attend_lg, lg_attend_sm in self.layers:
            # 对sm_cls进行注意力计算，使用lg_patch_tokens作为上下文，kv_include_self设置为True，然后加上原始sm_cls
            sm_cls = sm_attend_lg(sm_cls, context=lg_patch_tokens, kv_include_self=True) + sm_cls
            # 对lg_cls进行注意力计算，使用sm_patch_tokens作为上下文，kv_include_self设置为True，然后加上原始lg_cls
            lg_cls = lg_attend_sm(lg_cls, context=sm_patch_tokens, kv_include_self=True) + lg_cls

        # 将sm_cls和sm_patch_tokens在维度1上拼接起来
        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim=1)
        # 将lg_cls和lg_patch_tokens在维度1上拼接起来
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim=1)
        # 返回拼接后的sm_tokens和lg_tokens
        return sm_tokens, lg_tokens
# 定义多尺度编码器类
class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,  # 编码器深度
        sm_dim,  # 小尺度维度
        lg_dim,  # 大尺度维度
        sm_enc_params,  # 小尺度编码器参数
        lg_enc_params,  # 大尺度编码器参数
        cross_attn_heads,  # 跨尺度注意力头数
        cross_attn_depth,  # 跨尺度注意力深度
        cross_attn_dim_head = 64,  # 跨尺度注意力头维度
        dropout = 0.  # 丢弃率
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim = sm_dim, dropout = dropout, **sm_enc_params),  # 小尺度变换器
                Transformer(dim = lg_dim, dropout = dropout, **lg_enc_params),  # 大尺度变换器
                CrossTransformer(sm_dim = sm_dim, lg_dim = lg_dim, depth = cross_attn_depth, heads = cross_attn_heads, dim_head = cross_attn_dim_head, dropout = dropout)  # 跨尺度变换器
            ]))

    def forward(self, sm_tokens, lg_tokens):
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens), lg_enc(lg_tokens)  # 小尺度编码器和大尺度编码器
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)  # 跨尺度注意力

        return sm_tokens, lg_tokens

# 基于补丁的图像到标记嵌入器类
class ImageEmbedder(nn.Module):
    def __init__(
        self,
        *,
        dim,  # 维度
        image_size,  # 图像尺寸
        patch_size,  # 补丁尺寸
        dropout = 0.  # 丢弃率
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),  # 图像转换为补丁
            nn.LayerNorm(patch_dim),  # 层归一化
            nn.Linear(patch_dim, dim),  # 线性变换
            nn.LayerNorm(dim)  # 层归一化
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 位置嵌入
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 类别标记
        self.dropout = nn.Dropout(dropout)  # 丢弃层

    def forward(self, img):
        x = self.to_patch_embedding(img)  # 图像转换为补丁嵌入
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)  # 重复类别标记
        x = torch.cat((cls_tokens, x), dim=1)  # 拼接类别标记和补丁嵌入
        x += self.pos_embedding[:, :(n + 1)]  # 加上位置嵌入

        return self.dropout(x)  # 返回结果经过丢弃层处理

# 跨ViT类
class CrossViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,  # 图像尺寸
        num_classes,  # 类别数
        sm_dim,  # 小尺度维度
        lg_dim,  # 大尺度维度
        sm_patch_size = 12,  # 小尺度补丁尺寸
        sm_enc_depth = 1,  # 小尺度编码器深度
        sm_enc_heads = 8,  # 小尺度编码器头数
        sm_enc_mlp_dim = 2048,  # 小尺度编码器MLP维度
        sm_enc_dim_head = 64,  # 小尺度编码器头维度
        lg_patch_size = 16,  # 大尺度补丁尺寸
        lg_enc_depth = 4,  # 大尺度编码器深度
        lg_enc_heads = 8,  # 大尺度编码器头数
        lg_enc_mlp_dim = 2048,  # 大尺度编码器MLP维度
        lg_enc_dim_head = 64,  # 大尺度编码器头维度
        cross_attn_depth = 2,  # 跨尺度注意力深度
        cross_attn_heads = 8,  # 跨尺度注意力头数
        cross_attn_dim_head = 64,  # 跨尺度注意力头维度
        depth = 3,  # 深度
        dropout = 0.1,  # 丢弃率
        emb_dropout = 0.1  # 嵌入丢弃率
    # 初始化函数，继承父类的初始化方法
    def __init__(
        super().__init__()
        # 创建小尺寸图像嵌入器对象
        self.sm_image_embedder = ImageEmbedder(dim = sm_dim, image_size = image_size, patch_size = sm_patch_size, dropout = emb_dropout)
        # 创建大尺寸图像嵌入器对象
        self.lg_image_embedder = ImageEmbedder(dim = lg_dim, image_size = image_size, patch_size = lg_patch_size, dropout = emb_dropout)

        # 创建多尺度编码器对象
        self.multi_scale_encoder = MultiScaleEncoder(
            depth = depth,
            sm_dim = sm_dim,
            lg_dim = lg_dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            sm_enc_params = dict(
                depth = sm_enc_depth,
                heads = sm_enc_heads,
                mlp_dim = sm_enc_mlp_dim,
                dim_head = sm_enc_dim_head
            ),
            lg_enc_params = dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                mlp_dim = lg_enc_mlp_dim,
                dim_head = lg_enc_dim_head
            ),
            dropout = dropout
        )

        # 创建小尺寸MLP头部对象
        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        # 创建大尺寸MLP头部对象
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))

    # 前向传播函数
    def forward(self, img):
        # 获取小尺寸图像嵌入
        sm_tokens = self.sm_image_embedder(img)
        # 获取大尺寸图像嵌入
        lg_tokens = self.lg_image_embedder(img)

        # 多尺度编码器处理小尺寸和大尺寸图像嵌入
        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)

        # 提取小尺寸和大尺寸的类别特征
        sm_cls, lg_cls = map(lambda t: t[:, 0], (sm_tokens, lg_tokens))

        # 小尺寸MLP头部处理小尺寸类别特征
        sm_logits = self.sm_mlp_head(sm_cls)
        # 大尺寸MLP头部处理大尺寸类别特征
        lg_logits = self.lg_mlp_head(lg_cls)

        # 返回小尺寸和大尺寸类别特征的加和
        return sm_logits + lg_logits
```