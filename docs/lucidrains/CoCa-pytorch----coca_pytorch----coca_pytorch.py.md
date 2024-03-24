# `.\lucidrains\CoCa-pytorch\coca_pytorch\coca_pytorch.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 einsum, nn 模块
from torch import einsum, nn
# 从 torch 库中导入 F 模块
import torch.nn.functional as F
# 从 torch.autograd 库中导入 Function 模块
from torch.autograd import Function
# 从 torch.distributed 库中导入 dist 模块
import torch.distributed as dist

# 从 einops 库中导入 rearrange, repeat 函数

# helper functions

# 定义函数 exists，判断变量是否存在
def exists(val):
    return val is not None

# 定义函数 default，如果变量存在则返回其值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# distributed

# 定义函数 pad_dim_to，将张量在指定维度上填充到指定长度
def pad_dim_to(t, length, dim = 0):
    pad_length = length - t.shape[dim]
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length)

# 定义函数 all_gather_variable_batch，用于在分布式环境中收集所有张量的批次
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

# 定义类 AllGather，用于在分布式环境中收集所有张量
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

# 将 AllGather 类应用到张量上
all_gather = AllGather.apply

# normalization
# they use layernorm without bias, something that pytorch does not offer

# 定义类 LayerNorm，用于实现 Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# residual

# 定义类 Residual，用于实现残差连接
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

# to latents

# 定义类 EmbedToLatents，用于将输入转换为潜在空间
class EmbedToLatents(nn.Module):
    def __init__(self, dim, dim_latents):
        super().__init__()
        self.to_latents = nn.Linear(dim, dim_latents, bias=False)

    def forward(self, x):
        latents = self.to_latents(x)
        return F.normalize(latents, dim=-1)

# rotary positional embedding
# https://arxiv.org/abs/2104.09864

# 定义类 RotaryEmbedding，用于实现旋转位置嵌���
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

# 定义函数 rotate_half，用于旋转张量的一半
def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)

# 定义函数 apply_rotary_pos_emb，应用旋转位置嵌入到张量上
def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())

# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202

# 定义类 SwiGLU，用于实现 SwiGLU 激活函数
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

# parallel attention and feedforward with residual
# discovered by Wang et al + EleutherAI from GPT-J fame

# 定义类 ParallelTransformerBlock，用于实现并行的注意力和前馈网络块
class ParallelTransformerBlock(nn.Module):
    # 初始化函数，设置模型参数
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        # 调用父类的初始化函数
        super().__init__()
        # 对输入进行归一化处理
        self.norm = LayerNorm(dim)

        # 计算注意力机制和前馈网络的内部维度
        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))

        # 设置头数和缩放因子
        self.heads = heads
        self.scale = dim_head**-0.5
        # 初始化旋转嵌入
        self.rotary_emb = RotaryEmbedding(dim_head)

        # 定义融合的注意力机制和前馈网络的投影层
        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)

        # 前馈网络输出层
        self.ff_out = nn.Sequential(
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )

        # 用于缓存因果掩码和旋转嵌入
        self.mask = None
        self.pos_emb = None

    # 获取因果掩码
    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n].to(device)

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.mask = mask
        return mask

    # 获取旋转嵌入
    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n].to(device)

        pos_emb = self.rotary_emb(n, device=device)
        self.pos_emb = pos_emb
        return pos_emb

    # 前向传播函数
    def forward(self, x, attn_mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device, h = x.shape[1], x.device, self.heads

        # 预先归一化处理
        x = self.norm(x)

        # 获取注意力机制的查询、键、值和前馈网络的内部表示
        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # 分割头部
        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        # 旋转嵌入
        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # 缩放
        q = q * self.scale

        # 相似度计算
        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # 因果掩码
        causal_mask = self.get_mask(n, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # 额外的注意力掩码
        if exists(attn_mask):
            attn_mask = rearrange(attn_mask, 'b i j -> b 1 i j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        # 注意力计算
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # 聚合值
        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # 合并头部
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.attn_out(out) + self.ff_out(ff)
# 定义交叉注意力模块，使用多查询 + 单头键/值，类似于 PaLM，可选择并行前馈
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
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # 是否使用并行前馈
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

        # 预层归一化，用于查询和上下文
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
        sim = sim - sim.amax(dim=-1, keepdim=True)
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

# transformer
class CoCa(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        unimodal_depth,
        multimodal_depth,
        dim_latents = None,
        image_dim = None,
        num_img_queries=256,
        dim_head=64,
        heads=8,
        ff_mult=4,
        img_encoder=None,
        caption_loss_weight=1.,
        contrastive_loss_weight=1.,
        pad_id=0
    # 初始化函数，设置模型的参数
    def __init__(
        self,
        dim,
        num_tokens,
        pad_id,
        caption_loss_weight,
        contrastive_loss_weight,
        img_encoder,
        num_img_queries,
        image_dim,
        dim_head,
        heads,
        dim_latents,
        unimodal_depth,
        multimodal_depth,
        ff_mult
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 设置模型的维度
        self.dim = dim

        # 设置填充标识符和损失权重
        self.pad_id = pad_id
        self.caption_loss_weight = caption_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight

        # token embeddings

        # 创建 token embeddings 层
        self.token_emb = nn.Embedding(num_tokens, dim)
        # 创建文本分类标记
        self.text_cls_token = nn.Parameter(torch.randn(dim))

        # image encoder

        # 设置图像编码器
        self.img_encoder = img_encoder

        # attention pooling for image tokens

        # 创建图像查询参数
        self.img_queries = nn.Parameter(torch.randn(num_img_queries + 1, dim)) # num image queries for multimodal, but 1 extra CLS for contrastive learning
        # 创建图像注意力池化层
        self.img_attn_pool = CrossAttention(dim=dim, context_dim=image_dim, dim_head=dim_head, heads=heads, norm_context=True)

        # 图像注意力池化层的归一化
        self.img_attn_pool_norm = LayerNorm(dim)
        # 文本分类标记的归一化
        self.text_cls_norm = LayerNorm(dim)

        # to latents

        # 设置潜变量的维度
        dim_latents = default(dim_latents, dim)
        # 图像到潜变量的映射
        self.img_to_latents = EmbedToLatents(dim, dim_latents)
        # 文本到潜变量的映射
        self.text_to_latents = EmbedToLatents(dim, dim_latents)

        # 对比学习的温度参数
        self.temperature = nn.Parameter(torch.Tensor([1.]))

        # unimodal layers

        # 创建单模态层
        self.unimodal_layers = nn.ModuleList([])
        for ind in range(unimodal_depth):
            self.unimodal_layers.append(
                Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)),
            )

        # multimodal layers

        # 创建多模态层
        self.multimodal_layers = nn.ModuleList([])
        for ind in range(multimodal_depth):
            self.multimodal_layers.append(nn.ModuleList([
                Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)),
                Residual(CrossAttention(dim=dim, dim_head=dim_head, heads=heads, parallel_ff=True, ff_mult=ff_mult))
            ]))

        # to logits

        # 创建输出层
        self.to_logits = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_tokens, bias=False)
        )

        # 将嵌入权重与投影层权重绑定
        self.to_logits[-1].weight = self.token_emb.weight
        # 初始化嵌入权重
        nn.init.normal_(self.token_emb.weight, std=0.02)

        # 是否处于数据并行设置中
        self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

    # 嵌入文本
    def embed_text(self, text):
        # 获取批次大小和设备
        batch, device = text.shape[0], text.device

        # 获取序列长度
        seq = text.shape[1]

        # 获取文本的 token embeddings
        text_tokens = self.token_emb(text)

        # 添加文本分类标记
        text_cls_tokens = repeat(self.text_cls_token, 'd -> b 1 d', b=batch)
        text_tokens = torch.cat((text_tokens, text_cls_tokens), dim=-2)

        # 创建文本分类标记的特定掩码，防止其与填充部分进行注意力
        cls_mask = rearrange(text!=self.pad_id, 'b j -> b 1 j')
        attn_mask = F.pad(cls_mask, (0, 1, seq, 0), value=True)

        # 经过单模态层
        for attn_ff in self.unimodal_layers:
            text_tokens = attn_ff(text_tokens, attn_mask=attn_mask)

        # 获取文本分类标记
        text_tokens, text_cls_tokens = text_tokens[:, :-1], text_tokens[:, -1]
        text_embeds = self.text_cls_norm(text_cls_tokens)
        return text_embeds, text_tokens
    # 将图像嵌入到嵌入向量中
    def embed_image(self, images=None, image_tokens=None):
        # 将图像编码为嵌入向量
        # 使用在初始化时传入的 img_encoder
        # 也可以接受预先计算的图像标记

        # 确保图像和图像标记不同时存在
        assert not (exists(images) and exists(image_tokens))

        if exists(images):
            # 确保存在 self.img_encoder，用于自动图像编码
            assert exists(self.img_encoder), 'img_encoder must be passed in for automatic image encoding'
            image_tokens = self.img_encoder(images)

        # 注意力池化图像标记

        img_queries = repeat(self.img_queries, 'n d -> b n d', b=image_tokens.shape[0])
        img_queries = self.img_attn_pool(img_queries, image_tokens)
        img_queries = self.img_attn_pool_norm(img_queries)

        return img_queries[:, 0], img_queries[:, 1:]

    def forward(
        self,
        text,
        images=None,
        image_tokens=None,
        labels=None,
        return_loss=False,
        return_embeddings=False
    ):
        batch, device = text.shape[0], text.device

        if return_loss and not exists(labels):
            text, labels = text[:, :-1], text[:, 1:]

        text_embeds, text_tokens = self.embed_text(text)

        image_embeds, image_tokens = self.embed_image(images=images, image_tokens=image_tokens)

        # 如果研究人员需要返回嵌入向量，则返回嵌入向量

        if return_embeddings:
            return text_embeds, image_embeds

        # 经过多模态层

        for attn_ff, cross_attn in self.multimodal_layers:
            text_tokens = attn_ff(text_tokens)
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

        # 可能进行分布式全收集

        if self.is_distributed:
            latents = torch.stack((text_latents, image_latents), dim=1)
            latents = all_gather(latents)
            text_latents, image_latents = latents.unbind(dim=1)

        # 计算对比损失

        sim = einsum('i d, j d -> i j', text_latents, image_latents)
        sim = sim * self.temperature.exp()
        contrastive_labels = torch.arange(batch, device=device)

        contrastive_loss = (ce(sim, contrastive_labels) + ce(sim.t(), contrastive_labels)) * 0.5
        contrastive_loss = contrastive_loss * self.contrastive_loss_weight

        return caption_loss + contrastive_loss
```