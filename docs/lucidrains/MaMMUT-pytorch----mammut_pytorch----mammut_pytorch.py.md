# `.\lucidrains\MaMMUT-pytorch\mammut_pytorch\mammut_pytorch.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 einsum, nn 模块
from torch import einsum, nn
# 从 torch 库中导入 F 模块
import torch.nn.functional as F
# 从 torch 库中导入 distributed 模块
import torch.distributed as dist
# 从 torch 库中导入 Function 模块
from torch.autograd import Function

# 从 einops 库中导入 rearrange, repeat 函数
from einops import rearrange, repeat

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回该变量，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 判断一个数是否可以被另一个数整除
def divisible_by(numer, denom):
    return (numer % denom) == 0

# 分布式

# 在指定维度上对张量进行填充，使其达到指定长度
def pad_dim_to(t, length, dim = 0):
    pad_length = length - t.shape[dim]
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length)

# 对所有进程中的张量进行收集
def all_gather_variable_batch(t):
    device, rank, world_size = t.device, dist.get_rank(), dist.get_world_size()

    size = torch.tensor(t.shape[0], device = device, dtype = torch.long)
    sizes = [torch.empty_like(size, device = device, dtype = torch.long) for i in range(world_size)]
    dist.all_gather(sizes, size)

    sizes = torch.stack(sizes)
    max_size = sizes.amax().item()

    padded_t = pad_dim_to(t, max_size, dim = 0)
    gathered_tensors = [torch.empty_like(padded_t, device = device, dtype = padded_t.dtype) for i in range(world_size)]
    dist.all_gather(gathered_tensors, padded_t)

    gathered_tensor = torch.cat(gathered_tensors)
    seq = torch.arange(max_size, device = device)

    mask = rearrange(seq, 'j -> 1 j') < rearrange(sizes, 'i -> i 1')
    mask = rearrange(mask, 'i j -> (i j)')

    gathered_tensor = gathered_tensor[mask]
    sizes = sizes.tolist()

    return gathered_tensor, sizes

# 自定义的 AllGather 函数
class AllGather(Function):
    @staticmethod
    def forward(ctx, x):
        assert dist.is_initialized() and dist.get_world_size() > 1
        x, batch_sizes = all_gather_variable_batch(x)
        ctx.batch_sizes = batch_sizes
        return x

    @staticmethod
    def backward(ctx, grads):
        batch_sizes, rank = ctx.batch_sizes, dist.get_rank()
        grads_by_rank = grads.split(batch_sizes, dim = 0)
        return grads_by_rank[rank]

# 应用自定义的 AllGather 函数
all_gather = AllGather.apply

# 归一化
# 使用不带偏置的 layernorm，这是 PyTorch 不提供的功能

# Layernorm 类
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# 残差连接

# Residual 类
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

# 转换为潜变量

# EmbedToLatents 类
class EmbedToLatents(nn.Module):
    def __init__(self, dim, dim_latents):
        super().__init__()
        self.to_latents = nn.Linear(dim, dim_latents, bias=False)

    def forward(self, x):
        latents = self.to_latents(x)
        return F.normalize(latents, dim=-1)

# 旋转位置嵌入
# https://arxiv.org/abs/2104.09864

# RotaryEmbedding 类
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

# 将张量旋转一半
def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)

# 应用旋转位置嵌入
def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())

# 经典的 Noam Shazeer 论文，这里使用 SwiGLU 代替更流行的 GEGLU 用于门控前馈
# https://arxiv.org/abs/2002.05202

# SwiGLU 类
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

# 并行注意力和前馈，带有残差连接
# 定义一个并行Transformer块的类
class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.norm = LayerNorm(dim)  # 初始化LayerNorm

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))  # 定义融合维度

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)  # 初始化RotaryEmbedding

        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)  # 线性变换
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)  # 线性变换

        self.ff_out = nn.Sequential(
            SwiGLU(),  # SwiGLU激活函数
            nn.Linear(ff_inner_dim, dim, bias=False)  # 线性变换
        )

        # 用于缓存因果掩码和旋转嵌入
        self.mask = None
        self.pos_emb = None

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n].to(device)

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)  # 生成上三角掩码
        self.mask = mask
        return mask

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n].to(device)

        pos_emb = self.rotary_emb(n, device=device)  # 获取旋转嵌入
        self.pos_emb = pos_emb
        return pos_emb

    def forward(self, x, attn_mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device, h = x.shape[1], x.device, self.heads

        # pre layernorm
        x = self.norm(x)  # LayerNorm

        # attention queries, keys, values, and feedforward inner
        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)  # 拆分线性变换结果

        # split heads
        q = rearrange(q, "b n (h d) -> b h n d", h=h)  # 重排张量形状

        # rotary embeddings
        positions = self.get_rotary_embedding(n, device)  # 获取旋转嵌入
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))  # 应用旋转嵌入

        # scale
        q = q * self.scale  # 缩放

        # similarity
        sim = einsum("b h i d, b j d -> b h i j", q, k)  # 计算相似度

        # causal mask
        causal_mask = self.get_mask(n, device)  # 获取因果掩码
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)  # 应用掩码

        # extra attention mask - for masking out attention from text CLS token to padding
        if exists(attn_mask):
            attn_mask = rearrange(attn_mask, 'b i j -> b 1 i j')  # 重排注意力掩码
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)  # 应用额外的掩码

        # attention
        attn = sim.softmax(dim=-1)  # softmax计算注意力权重

        # aggregate values
        out = einsum("b h i j, b j d -> b h i d", attn, v)  # 聚合值

        # merge heads
        out = rearrange(out, "b h n d -> b n (h d)")  # 合并头部
        return self.attn_out(out) + self.ff_out(ff)  # 返回注意力输出和前馈输出

# cross attention - using multi-query + one-headed key / values as in PaLM w/ optional parallel feedforward
class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=8,
        parallel_ff=False,
        ff_mult=4,
        norm_context=False
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化头数和缩放因子
        self.heads = heads
        self.scale = dim_head ** -0.5
        # 计算内部维度
        inner_dim = heads * dim_head
        # 设置上下文维度
        context_dim = default(context_dim, dim)

        # 初始化 LayerNorm 层
        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(context_dim) if norm_context else nn.Identity()

        # 初始化线性变换层
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # 是否有并行前馈
        ff_inner_dim = ff_mult * dim

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_inner_dim * 2, bias=False),
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        ) if parallel_ff else None

    def forward(self, x, context):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # 预先 LayerNorm，用于查询和上下文
        x = self.norm(x)
        context = self.context_norm(context)

        # 获取查询
        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # 缩放
        q = q * self.scale

        # 获取键/值
        k, v = self.to_kv(context).chunk(2, dim=-1)

        # 查询/键相似度
        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # 注意力
        attn = sim.softmax(dim=-1)

        # 聚合
        out = einsum('b h i j, b j d -> b h i d', attn, v)

        # 合并和组合头
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        # 添加并行前馈（用于多模态层）
        if exists(self.ff):
            out = out + self.ff(x)

        return out
# 定义一个名为 MaMMUT 的类，继承自 nn.Module 类
class MaMMUT(nn.Module):
    # 初始化函数，接收多个参数
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        depth,
        cross_attend_every=1,
        cross_attend_layers=None,
        dim_latents=None,
        image_dim=None,
        num_img_queries=256,
        dim_head=64,
        heads=8,
        ff_mult=4,
        img_encoder=None,
        caption_loss_weight=1.,
        contrastive_loss_weight=1.,
        pad_id=0
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化类的属性
        self.dim = dim

        self.pad_id = pad_id
        self.caption_loss_weight = caption_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight

        # token embeddings

        # 创建一个嵌入层，用于将 token 映射为指定维度的向量
        self.token_emb = nn.Embedding(num_tokens, dim)
        # 创建一个 nn.Parameter 对象，用于存储文本的分类标记
        self.text_cls_token = nn.Parameter(torch.randn(dim))

        # image encoder

        # 设置图像编码器
        self.img_encoder = img_encoder

        # attention pooling for image tokens

        # 创建一个 nn.Parameter 对象，用于存储图像查询向量
        self.img_queries = nn.Parameter(torch.randn(num_img_queries + 1, dim)) # num image queries for multimodal, but 1 extra CLS for contrastive learning
        # 创建一个交叉注意力池化层，用于处理图像 token
        self.img_attn_pool = CrossAttention(dim=dim, context_dim=image_dim, dim_head=dim_head, heads=heads, norm_context=True)

        # 创建 LayerNorm 层，用于规范化图像注意力池化结果
        self.img_attn_pool_norm = LayerNorm(dim)
        # 创建 LayerNorm 层，用于规范化文本分类标记
        self.text_cls_norm = LayerNorm(dim)

        # to latents

        # 设置潜在空间的维度
        dim_latents = default(dim_latents, dim)
        # 创建将图像嵌入转换为潜在空间的层
        self.img_to_latents = EmbedToLatents(dim, dim_latents)
        # 创建将文本嵌入转换为潜在空间的层
        self.text_to_latents = EmbedToLatents(dim, dim_latents)

        # contrastive learning temperature

        # 创建一个 nn.Parameter 对象，用于存储对比学习的温度参数
        self.temperature = nn.Parameter(torch.Tensor([1.]))

        # layers

        # 创建一个空的 nn.ModuleList 对象，用于存储多个层
        self.layers = nn.ModuleList([])

        # 循环创建指定数量的层
        for ind in range(depth):
            layer = ind + 1

            has_cross_attn = divisible_by(layer, cross_attend_every)

            if exists(cross_attend_layers):
                assert isinstance(cross_attend_layers, tuple)
                has_cross_attn = layer in cross_attend_layers

            # 将每一层的处理逻辑添加到 layers 中
            self.layers.append(nn.ModuleList([
                Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)),
                Residual(CrossAttention(dim=dim, dim_head=dim_head, heads=heads, parallel_ff=True, ff_mult=ff_mult)) if has_cross_attn else None
            ]))

        # to logits

        # 创建一个序列，包含规范化层和线性层，用于生成输出 logits
        self.to_logits = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_tokens, bias=False)
        )

        # they used embedding weight tied projection out to logits, not common, but works

        # 将线性层的权重与嵌入层的权重绑定在一起
        self.to_logits[-1].weight = self.token_emb.weight
        # 初始化嵌入层的权重
        nn.init.normal_(self.token_emb.weight, std=0.02)

        # is data parallel
        # 检查是否启用了数据并行处理
        self.is_data_parallel = dist.is_initialized() and dist.get_world_size() > 1

    # 定义一个方法，用于将文本嵌入
    def embed_text(self, text):
        # 获取文本的批量大小和设备信息
        batch, device = text.shape[0], text.device

        seq = text.shape[1]

        # 获取文本的 token 嵌入
        text_tokens = self.token_emb(text)

        # append text cls tokens

        # 重复文本分类标记，拼接到文本 token 后面
        text_cls_tokens = repeat(self.text_cls_token, 'd -> b 1 d', b=batch)
        text_tokens = torch.cat((text_tokens, text_cls_tokens), dim=-2)

        # create specific mask for text cls token at the end
        # to prevent it from attending to padding

        # 创建特定的掩码，用于防止文本分类标记与填充部分进行注意力交互
        cls_mask = rearrange(text!=self.pad_id, 'b j -> b 1 j')
        attn_mask = F.pad(cls_mask, (0, 1, seq, 0), value=True)

        # go through layers, but do not cross attend

        # 遍历层，但不进行交叉注意力
        for attn_ff, _ in self.layers:
            text_tokens = attn_ff(text_tokens, attn_mask=attn_mask)

        # get text cls token

        # 获取文本分类标记和文本 token
        text_tokens, text_cls_tokens = text_tokens[:, :-1], text_tokens[:, -1]
        # 规范化文本分类标记
        text_embeds = self.text_cls_norm(text_cls_tokens)
        return text_embeds, text_tokens
    # 将图像嵌入到嵌入向量中
    def embed_image(self, images=None, image_tokens=None):
        # 将图像编码为嵌入向量
        # 使用在初始化时传入的 img_encoder
        # 也可以接受预先计算的图像 tokens

        # 确保 images 和 image_tokens 不能同时存在
        assert not (exists(images) and exists(image_tokens))

        if exists(images):
            # 确保存在 self.img_encoder，用于自动图像编码
            assert exists(self.img_encoder), 'img_encoder must be passed in for automatic image encoding'
            image_tokens = self.img_encoder(images)

        # 注意力池化图像 tokens

        img_queries = repeat(self.img_queries, 'n d -> b n d', b=image_tokens.shape[0])
        img_queries = self.img_attn_pool(img_queries, image_tokens)
        img_queries = self.img_attn_pool_norm(img_queries)

        return img_queries[:, 0], img_queries[:, 1:]

    # 前向传播函数
    def forward(
        self,
        text,
        text_mask = None,
        images=None,
        image_tokens=None,
        labels=None,
        return_loss=False,
        return_embeddings=False
    ):
        batch, device = text.shape[0], text.device

        if return_loss and not exists(labels):
            text, labels = text[:, :-1], text[:, 1:]

        text_embeds, _ = self.embed_text(text)

        image_embeds, image_tokens = self.embed_image(images=images, image_tokens=image_tokens)

        # 如果研究人员需要返回嵌入向量，则返回嵌入向量
        if return_embeddings:
            return text_embeds, image_embeds

        # 经过各层处理

        text_tokens = self.token_emb(text)

        for attn_ff, cross_attn in self.layers:
            text_tokens = attn_ff(text_tokens)

            if exists(cross_attn):
                text_tokens = cross_attn(text_tokens, image_tokens)

        logits = self.to_logits(text_tokens)

        if not return_loss:
            return logits

        # 缩写

        ce = F.cross_entropy

        # 计算标题损失（交叉熵损失）

        logits = rearrange(logits, 'b n c -> b c n')
        caption_loss = ce(logits, labels, ignore_index=self.pad_id)
        caption_loss = caption_loss * self.caption_loss_weight

        # 嵌入到潜变量

        text_latents = self.text_to_latents(text_embeds)
        image_latents = self.img_to_latents(image_embeds)

        # 如果使用数据并行，需要从所有机器中收集所有潜变量

        if self.is_data_parallel:
            latents = torch.stack((text_latents, image_latents), dim = 1)
            latents = all_gather(latents)
            text_latents, image_latents = latents.unbind(dim = 1)

        # 计算对比损失

        sim = einsum('i d, j d -> i j', text_latents, image_latents)
        sim = sim * self.temperature.exp()
        contrastive_labels = torch.arange(batch, device=device)

        contrastive_loss = (ce(sim, contrastive_labels) + ce(sim.t(), contrastive_labels)) * 0.5
        contrastive_loss = contrastive_loss * self.contrastive_loss_weight

        return caption_loss + contrastive_loss
```