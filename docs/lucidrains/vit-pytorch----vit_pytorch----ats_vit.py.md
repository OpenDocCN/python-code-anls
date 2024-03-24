# `.\lucidrains\vit-pytorch\vit_pytorch\ats_vit.py`

```py
# 导入 torch 库
import torch
# 导入 torch 中的函数库
import torch.nn.functional as F
# 从 torch.nn.utils.rnn 中导入 pad_sequence 函数
from torch.nn.utils.rnn import pad_sequence
# 从 torch 中导入 nn、einsum 模块
from torch import nn, einsum
# 从 einops 中导入 rearrange、repeat 函数和 Rearrange 类
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# 辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 将输入转换为元组
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# 自适应令牌采样函数和类

# 计算输入张量的自然对数，避免输入为 0 时出现错误
def log(t, eps = 1e-6):
    return torch.log(t + eps)

# 生成服从 Gumbel 分布的随机数
def sample_gumbel(shape, device, dtype, eps = 1e-6):
    u = torch.empty(shape, device = device, dtype = dtype).uniform_(0, 1)
    return -log(-log(u, eps), eps)

# 在指定维度上对输入张量进行批量索引选择
def batched_index_select(values, indices, dim = 1):
    # 获取值张量和索引张量的维度信息
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    # 将索引张量扩展到与值张量相同的维度
    indices = indices[(..., *((None,) * len(value_dims))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

# 自适应令牌采样类
class AdaptiveTokenSampling(nn.Module):
    def __init__(self, output_num_tokens, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.output_num_tokens = output_num_tokens
    # 定义一个前向传播函数，接收注意力值、数值、掩码作为输入
    def forward(self, attn, value, mask):
        # 获取注意力值的头数、输出的标记数、eps值、设备和数据类型
        heads, output_num_tokens, eps, device, dtype = attn.shape[1], self.output_num_tokens, self.eps, attn.device, attn.dtype

        # 获取CLS标记到所有其他标记的注意力值
        cls_attn = attn[..., 0, 1:]

        # 计算数值的范数，用于加权得分，如论文中所述
        value_norms = value[..., 1:, :].norm(dim=-1)

        # 通过数值的范数加权注意力得分，对所有头求和
        cls_attn = einsum('b h n, b h n -> b n', cls_attn, value_norms)

        # 归一化为1
        normed_cls_attn = cls_attn / (cls_attn.sum(dim=-1, keepdim=True) + eps)

        # 不使用逆变换采样，而是反转softmax并使用gumbel-max采样
        pseudo_logits = log(normed_cls_attn)

        # 为gumbel-max采样屏蔽伪对数
        mask_without_cls = mask[:, 1:]
        mask_value = -torch.finfo(attn.dtype).max / 2
        pseudo_logits = pseudo_logits.masked_fill(~mask_without_cls, mask_value)

        # 扩展k次，k为自适应采样数
        pseudo_logits = repeat(pseudo_logits, 'b n -> b k n', k=output_num_tokens)
        pseudo_logits = pseudo_logits + sample_gumbel(pseudo_logits.shape, device=device, dtype=dtype)

        # gumbel-max采样并加一以保留0用于填充/掩码
        sampled_token_ids = pseudo_logits.argmax(dim=-1) + 1

        # 使用torch.unique计算唯一值，然后从右侧填充序列
        unique_sampled_token_ids_list = [torch.unique(t, sorted=True) for t in torch.unbind(sampled_token_ids)]
        unique_sampled_token_ids = pad_sequence(unique_sampled_token_ids_list, batch_first=True)

        # 基于填充计算新的掩码
        new_mask = unique_sampled_token_ids != 0

        # CLS标记永远不会被屏蔽（得到True值）
        new_mask = F.pad(new_mask, (1, 0), value=True)

        # 在前面添加一个0标记ID以保留CLS注意力得分
        unique_sampled_token_ids = F.pad(unique_sampled_token_ids, (1, 0), value=0)
        expanded_unique_sampled_token_ids = repeat(unique_sampled_token_ids, 'b n -> b h n', h=heads)

        # 收集新的注意力得分
        new_attn = batched_index_select(attn, expanded_unique_sampled_token_ids, dim=2)

        # 返回采样的注意力得分、新掩码（表示填充）以及采样的标记索引（用于残差）
        return new_attn, new_mask, unique_sampled_token_ids
# 定义前馈神经网络类
class FeedForward(nn.Module):
    # 初始化函数，定义网络结构
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        # 使用 nn.Sequential 定义网络层次结构
        self.net = nn.Sequential(
            nn.LayerNorm(dim),  # Layer normalization
            nn.Linear(dim, hidden_dim),  # 线性变换
            nn.GELU(),  # GELU 激活函数
            nn.Dropout(dropout),  # Dropout 正则化
            nn.Linear(hidden_dim, dim),  # 线性变换
            nn.Dropout(dropout)  # Dropout 正则化
        )
    # 前向传播函数
    def forward(self, x):
        return self.net(x)

# 定义注意力机制类
class Attention(nn.Module):
    # 初始化函数，定义注意力机制结构
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., output_num_tokens = None):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)  # Layer normalization
        self.attend = nn.Softmax(dim = -1)  # Softmax 注意力权重计算
        self.dropout = nn.Dropout(dropout)  # Dropout 正则化

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)  # 线性变换

        self.output_num_tokens = output_num_tokens
        self.ats = AdaptiveTokenSampling(output_num_tokens) if exists(output_num_tokens) else None

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  # 线性变换
            nn.Dropout(dropout)  # Dropout 正则化
        )

    # 前向传播函数
    def forward(self, x, *, mask):
        num_tokens = x.shape[1]

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if exists(mask):
            dots_mask = rearrange(mask, 'b i -> b 1 i 1') * rearrange(mask, 'b j -> b 1 1 j')
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(~dots_mask, mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        sampled_token_ids = None

        # 如果启用了自适应令牌采样，并且令牌数量大于输出令牌数量
        if exists(self.output_num_tokens) and (num_tokens - 1) > self.output_num_tokens:
            attn, mask, sampled_token_ids = self.ats(attn, v, mask = mask)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), mask, sampled_token_ids

# 定义 Transformer 类
class Transformer(nn.Module):
    # 初始化函数，定义 Transformer 结构
    def __init__(self, dim, depth, max_tokens_per_depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        assert len(max_tokens_per_depth) == depth, 'max_tokens_per_depth must be a tuple of length that is equal to the depth of the transformer'
        assert sorted(max_tokens_per_depth, reverse = True) == list(max_tokens_per_depth), 'max_tokens_per_depth must be in decreasing order'
        assert min(max_tokens_per_depth) > 0, 'max_tokens_per_depth must have at least 1 token at any layer'

        self.layers = nn.ModuleList([])
        for _, output_num_tokens in zip(range(depth), max_tokens_per_depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, output_num_tokens = output_num_tokens, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    # 定义前向传播函数，接受输入张量 x
    def forward(self, x):
        # 获取输入张量 x 的形状的前两个维度大小和设备信息
        b, n, device = *x.shape[:2], x.device

        # 使用掩码来跟踪填充位置，以便在采样标记时移除重复项
        mask = torch.ones((b, n), device=device, dtype=torch.bool)

        # 创建一个包含从 0 到 n-1 的张量，设备信息与输入张量 x 一致
        token_ids = torch.arange(n, device=device)
        token_ids = repeat(token_ids, 'n -> b n', b=b)

        # 遍历每个注意力层和前馈层
        for attn, ff in self.layers:
            # 调用注意力层的前向传播函数，获取注意力输出、更新后的掩码和采样的标记
            attn_out, mask, sampled_token_ids = attn(x, mask=mask)

            # 当进行标记采样时，需要使用采样的标记 id 从输入张量中选择对应的标记
            if exists(sampled_token_ids):
                x = batched_index_select(x, sampled_token_ids, dim=1)
                token_ids = batched_index_select(token_ids, sampled_token_ids, dim=1)

            # 更新输入张量，加上注意力输出
            x = x + attn_out

            # 经过前馈层处理后再加上原始输入，得到最终输出
            x = ff(x) + x

        # 返回最终输出张量和标记 id
        return x, token_ids
class ViT(nn.Module):
    # 定义 ViT 模型类，继承自 nn.Module
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, max_tokens_per_depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        # 初始化函数，接收参数 image_size, patch_size, num_classes, dim, depth, max_tokens_per_depth, heads, mlp_dim, channels, dim_head, dropout, emb_dropout
        super().__init__()
        # 调用父类的初始化函数

        image_height, image_width = pair(image_size)
        # 获取图像的高度和宽度
        patch_height, patch_width = pair(patch_size)
        # 获取补丁的高度和宽度

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        # 断言，确保图像的尺寸能够被补丁的尺寸整除

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # 计算补丁的数量
        patch_dim = channels * patch_height * patch_width
        # 计算每个补丁的维度

        self.to_patch_embedding = nn.Sequential(
            # 定义将图像转换为补丁嵌入的序列
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            # 重新排列图像的通道和补丁的维度
            nn.LayerNorm(patch_dim),
            # 对每个补丁进行 LayerNorm
            nn.Linear(patch_dim, dim),
            # 线性变换将每个补丁的维度映射到指定的维度 dim
            nn.LayerNorm(dim)
            # 对映射后的维度进行 LayerNorm
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # 初始化位置嵌入参数
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # 初始化类别标记参数
        self.dropout = nn.Dropout(emb_dropout)
        # 定义丢弃层，用于嵌入的丢弃

        self.transformer = Transformer(dim, depth, max_tokens_per_depth, heads, dim_head, mlp_dim, dropout)
        # 初始化 Transformer 模型

        self.mlp_head = nn.Sequential(
            # 定义 MLP 头部
            nn.LayerNorm(dim),
            # 对输入进行 LayerNorm
            nn.Linear(dim, num_classes)
            # 线性变换将维度映射到类别数量
        )

    def forward(self, img, return_sampled_token_ids = False):
        # 定义前向传播函数，接收图像和是否返回采样的令牌 ID

        x = self.to_patch_embedding(img)
        # 将图像转换为补丁嵌入
        b, n, _ = x.shape
        # 获取 x 的形状信息

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # 重复类别标记，使其与补丁嵌入的形状相同
        x = torch.cat((cls_tokens, x), dim=1)
        # 拼接类别标记和补丁嵌入
        x += self.pos_embedding[:, :(n + 1)]
        # 添加位置嵌入
        x = self.dropout(x)
        # 对输入进行丢弃

        x, token_ids = self.transformer(x)
        # 使用 Transformer 进行转换

        logits = self.mlp_head(x[:, 0])
        # 使用 MLP 头部生成输出

        if return_sampled_token_ids:
            # 如果需要返回采样的令牌 ID
            token_ids = token_ids[:, 1:] - 1
            # 移除类别标记并减去 1 以使 -1 成为填充
            return logits, token_ids
            # 返回输出和令牌 ID

        return logits
        # 返回输出
```