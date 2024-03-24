# `.\lucidrains\x-clip\x_clip\x_clip.py`

```
# 导入数学库
import math
# 导入复制库
import copy
# 导入上下文管理器
from contextlib import contextmanager
# 导入偏函数和装饰器
from functools import partial, wraps

# 导入 PyTorch 库
import torch
import torch.nn.functional as F
import torch.distributed as distributed
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint

# 导入 einops 库
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

# 导入自定义模块
from x_clip.mlm import MLM
from x_clip.visual_ssl import SimSiam, SimCLR
from x_clip.distributed import all_gather

# 辅助函数

# 返回输入本身
def identity(t, *args, **kwargs):
    return t

# 检查值是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    return val if exists(val) else d

# 空上下文管理器
@contextmanager
def null_context():
    yield

# 返回指定数据类型的最大负值
def max_neg_value(dtype):
    return -torch.finfo(dtype).max

# 将输入转换为元组
def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

# 计算带掩码的均值
def masked_mean(t, mask, dim = 1, eps = 1e-6):
    t = t.masked_fill(~mask, 0.)
    numer = t.sum(dim = dim)
    denom = mask.sum(dim = dim).clamp(min = eps)
    return numer / denom

# 将输入张量的指定维度填充到指定长度
def pad_dim_to(t, length, dim = 0):
    pad_length = length - t.shape[dim]
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length))

# 计算输入张量的对数
def log(t, eps = 1e-20):
    return torch.log(t + eps)

# 计算输入张量的 L2 范数
def l2norm(t):
    return F.normalize(t, dim = -1)

# 提取输入张量的对角线元素
def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device = device)
    j_range = torch.arange(j, device = device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d = num_diag_el)

# 检查点辅助函数

# 使函数支持检查点
def make_checkpointable(fn):
    @wraps(fn)
    def inner(*args):
        input_needs_grad = any([isinstance(el, torch.Tensor) and el.requires_grad for el in args])

        if not input_needs_grad:
            return fn(*args)

        return checkpoint(fn, *args)

    return inner

# 关键字参数辅助函数

# 从字典中选择指定键的值并弹出这些键
def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

# 根据条件将字典分组
def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

# 检查字符串是否以指定前缀开头
def string_begins_with(prefix, str):
    return str.startswith(prefix)

# 根据前缀将字典分组
def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

# 根据前缀将字典分组并去除前缀
def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items()))
    return kwargs_without_prefix, kwargs

# 辅助类

# 重排图像维度
class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h = int(math.sqrt(x.shape[1])))

# 层归一化
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = -1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = -1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

# 预归一化
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

# 补丁丢弃

class PatchDropout(nn.Module):
    def __init__(self, prob):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
    # 定义一个前向传播函数，用于在训练时进行部分丢弃
    def forward(self, x, force_keep_all = False):
        # 如果不在训练模式下，或者概率为0，或者强制保留所有元素，则直接返回输入
        if not self.training or self.prob == 0. or force_keep_all:
            return x

        # 获取输入张量的形状信息和设备信息
        b, n, _, device = *x.shape, x.device

        # 创建一个包含0到b-1的整数张量，用于索引每个样本
        batch_indices = torch.arange(b, device = device)
        # 重新排列张量维度，将其变为二维张量
        batch_indices = rearrange(batch_indices, '... -> ... 1')
        # 计算应该保留的补丁数量，至少保留一个补丁
        num_patches_keep = max(1, int(n * (1 - self.prob)))
        # 生成服从标准正态分布的随机数，然后在每个样本的补丁中选择要保留的补丁索引
        patch_indices_keep = torch.randn(b, n, device = device).topk(num_patches_keep, dim = -1).indices

        # 返回保留的补丁数据
        return x[batch_indices, patch_indices_keep]
# 定义旋转位置嵌入类
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 计算频率的倒数
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        # 将频率的倒数作为缓冲区
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        # 获取频率的倒数
        inv_freq = self.inv_freq
        # 生成序列长度的张量
        t = torch.arange(seq_len, device=device).type_as(inv_freq)
        # 计算频率
        freqs = torch.einsum('i , j -> i j', t, inv_freq)
        # 拼接频率，返回结果
        return torch.cat((freqs, freqs), dim=-1)

# 旋转半个张量
def rotate_half(x):
    # 重新排列张量
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    # 拆分张量
    x1, x2 = x.unbind(dim=-2)
    # 拼接旋转后的张量
    return torch.cat((-x2, x1), dim=-1)

# 应用旋转位置嵌入
def apply_rotary_pos_emb(freqs, t):
    # 获取旋转维度
    rot_dim = freqs.shape[-1]
    # 拆分张量
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    # 应用旋转位置嵌入
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    # 拼接结果
    return torch.cat((t, t_pass), dim=-1)

# GEGLU模块
class GEGLU(nn.Module):
    def forward(self, x):
        # 拆分张量
        x, gate = x.chunk(2, dim=-1)
        # 返回GEGLU激活后的结果
        return x * F.gelu(gate)

# 前馈神经网络模块
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2, bias=False),
            GEGLU(),
            LayerNorm(inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias=False)
        )

    def forward(self, x):
        # 返回前馈神经网络的结果
        return self.net(x)

# 注意力机制模块
class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, causal=False, dropout=0.):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias=False), LayerNorm(dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, rotary_pos_emb=None):
        h, device, scale = self.heads, x.device, self.scale

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        q = q * self.scale

        if exists(rotary_pos_emb):
            apply_rotary = partial(apply_rotary_pos_emb, rotary_pos_emb)
            q, k, v = map(apply_rotary, (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.type(sim.dtype)

        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Transformer模块
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head=64,
        heads=8,
        causal=False,
        attn_dropout=0.,
        ff_dropout=0.,
        ff_mult=4,
        checkpoint_during_training=False
    ):
        super().__init__()
        self.checkpoint_during_training = checkpoint_during_training

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim=dim, dim_head=dim_head, heads=heads, causal=causal, dropout=attn_dropout)),
                PreNorm(dim, FeedForward(dim=dim, mult=ff_mult)),
            ]))

        self.norm_in = LayerNorm(dim)
        self.norm_out = LayerNorm(dim)
    # 定义一个前向传播函数，接受输入 x，旋转位置嵌入 rotary_pos_emb 和掩码 mask
    def forward(
        self,
        x,
        rotary_pos_emb = None,
        mask = None
    ):
        # 检查是否可以在训练期间进行检查点，如果可以则调用 make_checkpointable 函数，否则调用 identity 函数
        can_checkpoint = self.training and self.checkpoint_during_training
        checkpoint_fn = make_checkpointable if can_checkpoint else identity

        # 对输入 x 进行归一化处理
        x = self.norm_in(x)

        # 遍历每个注意力层和前馈层
        for attn, ff in self.layers:
            # 对注意力层和前馈层应用检查点函数
            attn, ff = map(checkpoint_fn, (attn, ff))

            # 执行注意力层操作，并将结果与输入 x 相加
            x = attn(x, mask, rotary_pos_emb) + x
            # 执行前馈层操作，并将结果与输入 x 相加
            x = ff(x) + x

        # 对最终输出 x 进行归一化处理
        return self.norm_out(x)
# 定义文本转换器类，继承自 nn.Module
class TextTransformer(nn.Module):
    # 初始化函数，接受一系列参数
    def __init__(
        self,
        dim,
        *,
        num_tokens,
        max_seq_len,
        dim_head,
        rotary_pos_emb = None,
        causal = False,
        **kwargs
    ):
        super().__init__()
        # 创建一个词嵌入层，将输入的标记转换为指定维度的向量
        self.token_emb = nn.Embedding(num_tokens, dim)

        # 创建绝对位置编码层，用于处理绝对位置信息
        self.abs_pos_emb = nn.Embedding(max_seq_len, dim) if not rotary_pos_emb else None
        # 创建旋转位置编码层，用于处理旋转位置信息
        self.rotary_pos_emb = RotaryEmbedding(min(dim_head, 32)) if rotary_pos_emb else None

        # 创建一个类别标记参数，用于处理因果关系
        self.cls_token = nn.Parameter(torch.randn(dim)) if not causal else None

        # 创建一个 Transformer 模型，用于文本转换
        self.transformer = Transformer(dim, dim_head = dim_head, causal = causal, **kwargs)

    # 前向传播函数，接受输入 x 和掩码 mask
    def forward(self, x, mask = None):
        # 获取输入 x 的形状和设备信息
        b, n, device = *x.shape, x.device

        # 将输入 x 转换为词嵌入向量
        x = self.token_emb(x)

        # 如果存在绝对位置编码层，则添加绝对位置编码信息
        if exists(self.abs_pos_emb):
            pos_emb = self.abs_pos_emb(torch.arange(n, device = device))
            x = x + rearrange(pos_emb, 'n d -> 1 n d')

        rotary_pos_emb = None
        # 如果存在旋转位置编码层，则获取旋转位置编码信息
        if exists(self.rotary_pos_emb):
            rotary_pos_emb = self.rotary_pos_emb(n + 1, device = device)

        # 如果存在类别标记参数，则添加类别标记到输入 x 中
        if exists(self.cls_token):
            cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b = b)
            x = torch.cat((cls_tokens, x), dim = 1)

            # 如果存在掩码，则在掩码前面填充一个值为 True 的元素
            if exists(mask):
                mask = F.pad(mask, (1, 0), value = True)

        # 使用 Transformer 模型进行转换
        out = self.transformer(x, mask = mask, rotary_pos_emb = rotary_pos_emb)
        return out

# 定义视觉转换器类，继承自 nn.Module
class VisionTransformer(nn.Module):
    # 初始化函数，接受一系列参数
    def __init__(
        self,
        dim,
        *,
        image_size,
        patch_size,
        channels,
        patch_dropout = 0.5,
        **kwargs
    ):
        super().__init__()
        # 断言图像尺寸必须能够被补丁大小整除
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        # 创建一个将图像转换为标记序列的模块
        self.to_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim)
        )

        # 创建位置编码层，用于处理位置信息
        self.pos_emb = nn.Embedding(num_patches, dim)
        # 创建补丁丢弃模块，用于随机丢弃补丁
        self.patch_dropout = PatchDropout(patch_dropout)

        # 创建一个 Transformer 模型，用于视觉转换
        self.transformer = Transformer(dim, **kwargs)

        # 创建一个将输出转换为类别标记的模块
        self.to_cls_tokens = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, dim, bias = False),
            Rearrange('b d -> b 1 d')
        )

    # 前向传播函数，接受输入 x 和是否保留所有补丁的标志
    def forward(
        self,
        x,
        keep_all_patches = False
    ):
        device = x.device

        # 将输入 x 转换为标记序列
        x = self.to_tokens(x)
        b, n, _ = x.shape

        # 添加位置编码信息到输入 x 中
        pos_emb = self.pos_emb(torch.arange(n, device = device))
        x = x + rearrange(pos_emb, 'n d -> 1 n d')

        # 对输入 x 进行补丁丢弃处理
        x = self.patch_dropout(x, force_keep_all = keep_all_patches)

        # 使用 Transformer 模型进行转换
        out = self.transformer(x)

        # 将输出转换为类别标记并返回
        cls_tokens = self.to_cls_tokens(out)
        return torch.cat((cls_tokens, out), dim = 1)

# 定义模型前向传播函数，接受一系列参数
def model_forward_with_context(
    *,
    fn,
    args,
    freeze,
):
    # 根据是否冻结模型选择上下文
    encoding_context = null_context if not freeze else torch.no_grad

    # 在指定上下文中执行模型前向传播
    with encoding_context():
        enc = fn(*args)

        # 如果冻结模型，则将输出张量断开梯度
        if freeze:
            enc.detach_()

    return enc

# 主要的 CLIP 类，继承自 nn.Module
class CLIP(nn.Module):
    # 初始化函数，设置各种参数的默认取值
    def __init__(
        self,
        *,
        image_encoder = None,  # 图像编码器，默认为None
        text_encoder = None,   # 文本编码器，默认为None
        dim_text = 512,        # 文本维度，默认为512
        dim_image = 512,       # 图像维度，默认为512
        dim_latent = 512,      # 潜在空间维度，默认为512
        num_text_tokens = 10000,  # 文本标记数量，默认为10000
        text_enc_depth = 6,        # 文本编码器深度，默认为6
        text_seq_len = 256,        # 文本序列长度，默认为256
        text_heads = 8,            # 文本头数，默认为8
        text_dim_head = 64,        # 文本头维度，默认为64
        text_has_cls_token = True, # 文本是否包含CLS标记，默认为True
        text_pad_id = 0,           # 文本填充标记，默认为0
        text_rotary_pos_emb = False,  # 是否使用旋转位置编码，默认为False
        text_causal_mask = False,     # 是否使用因果掩码，默认为False
        text_eos_id = None,           # 文本结束标记，默认为None
        text_encode_without_mask = False,  # 是否在不使用掩码的情况下编码文本，默认为False
        visual_enc_depth = 6,        # 图像编码器深度，默认为6
        visual_heads = 8,            # 图像头数，默认为8
        visual_dim_head = 64,        # 图像头维度，默认为64
        visual_image_size = 256,     # 图像大小，默认为256
        visual_patch_size = 32,      # 图像块大小，默认为32
        visual_patch_dropout = 0.5,  # 图像块丢弃率，默认为0.5
        visual_has_cls_token = True, # 图像是否包含CLS标记，默认为True
        channels = 3,                # 通道数，默认为3
        use_all_token_embeds = False,  # 是否使用所有标记嵌入，默认为False
        downsample_image_embeds = False,  # 是否降采样图像嵌入，默认为False
        decoupled_contrastive_learning = False,  # 是否解耦对比学习，默认为False
        extra_latent_projection = False,         # 是否使用额外的潜在投影，默认为False
        use_mlm = False,                         # 是否使用MLM，默认为False
        text_ssl_loss_weight = 0.05,             # 文本SSL损失权重，默认为0.05
        use_visual_ssl = False,                  # 是否使用视觉SSL，默认为False
        visual_ssl = None,                       # 视觉SSL，默认为None
        visual_ssl_type = 'simsiam',             # 视觉SSL类型，默认为'simsiam'
        visual_ssl_hidden_layer = -1,            # 视觉SSL隐藏层，默认为-1
        simclr_temperature = 0.1,                # SimCLR温度，默认为0.1
        image_ssl_loss_weight = 0.05,            # 图像SSL损失权重，默认为0.05
        multiview_loss_weight = 0.1,             # 多视图损失权重，默认为0.1
        checkpoint_during_training = False,      # 训练期间是否检查点，默认为False
        sim_reg_loss_weight = 0.,                # 相似性正则化损失权重，默认为0.0
        **kwargs
    def forward(
        self,
        text,
        image,
        return_loss = False,                # 是否返回损失，默认为False
        return_encodings = False,           # 是否返回编码，默认为False
        return_latents = False,             # 是否返回潜在空间，默认为False
        freeze_image_encoder = False,       # 如果设置为True，则图像编码器不会被训练，由LiT论文提出
        freeze_text_encoder = False,        # 如果设置为True，则文本编码器不会被训练
        text_to_image = True,               # 在额外投影打开的情况下，根据模态方向返回不同的相似性值
        aug_text = None,                    # 增强文本（用于多视图）
        aug_image = None                    # 增强图像（用于多视图）
```