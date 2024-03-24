# `.\lucidrains\RETRO-pytorch\retro_pytorch\retro_pytorch.py`

```py
# 导入必要的库
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum

# 导入自定义的库
from retro_pytorch.retrieval import BERT_VOCAB_SIZE
from einops import rearrange, repeat

# 常量定义
MIN_DIM_HEAD = 32

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回其值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 判断一个数是否可以被另一个数整除
def divisible_by(val, divisor):
    return (val / divisor).is_integer()

# 将变量转换为元组
def cast_tuple(val, num = 1):
    return val if isinstance(val, tuple) else ((val,) * num)

# 初始化深度网络参数
def deepnorm_init(transformer, beta, module_name_match_list = ['.ff.', '.to_v', '.to_out']):
    for name, module in transformer.named_modules():
        if type(module) != nn.Linear:
            continue

        needs_beta_gain = any(map(lambda substr: substr in name, module_name_match_list))
        gain = beta if needs_beta_gain else 1
        nn.init.xavier_normal_(module.weight.data, gain = gain)

        if exists(module.bias):
            nn.init.constant_(module.bias.data, 0)

# 归一化

# RMS归一化类
class RMSNorm(nn.Module):
    def __init__(
        self,
        dim,
        *,
        eps = 1e-8,
        gated = False
    ):
        super().__init__()
        self.eps = eps
        self.scale = dim ** -0.5
        self.gamma = nn.Parameter(torch.ones(dim))
        self.weight = nn.Parameter(torch.ones(dim)) if gated else None

    def forward(self, x):
        norm = x.norm(keepdim = True, dim = -1) * self.scale
        out = (x / norm.clamp(min = self.eps)) * self.gamma

        if not exists(self.weight):
            return out

        return out * (x * self.weight).sigmoid()

# 前向和后向归一化残差包装模块

# 前向归一化类
class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm_klass = RMSNorm):
        super().__init__()
        self.fn = fn
        self.norm = norm_klass(dim)

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs) + x

# 后向归一化类
class PostNorm(nn.Module):
    def __init__(self, dim, fn, scale_residual = 1, norm_klass = RMSNorm):
        super().__init__()
        self.fn = fn
        self.scale_residual = scale_residual
        self.norm = norm_klass(dim)

    def forward(self, x, *args, **kwargs):
        residual = x * self.scale_residual
        out = self.fn(x, *args, **kwargs) + residual
        return self.norm(out)

# 位置嵌入

# 旋转嵌入类
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len, *, device, offset = 0):
        seq = torch.arange(max_seq_len, device = device) + offset
        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim = -1)
        return rearrange(emb, 'n d -> 1 1 n d')

# 旋转半个位置
def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

# 应用旋转位置嵌入
def apply_rotary_pos_emb(t, freqs):
    seq_len, rot_dim = t.shape[-2], freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim = -1)

# 前馈网络

# 前馈网络类
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(mult * dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x):
        return self.ff(x)

# 注意力机制

# 注意力类
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim = None,
        dim_head = 64,
        heads = 8,
        causal = False,
        dropout = 0.,
        null_kv = False
    # 初始化函数，设置模型参数
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        causal = False,
        context_dim = None,
        null_kv = False
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 设置上下文维度，默认为输入维度
        context_dim = default(context_dim, dim)

        # 设置头数和缩放因子
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads

        # 设置dropout层
        self.dropout = nn.Dropout(dropout)

        # 线性变换层，将输入转换为查询、键、值
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias = False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        # 允许对空值进行注意力计算，以防止注意力破坏
        self.null_k = nn.Parameter(torch.randn(inner_dim)) if null_kv else None
        self.null_v = nn.Parameter(torch.randn(inner_dim)) if null_kv else None

    # 前向传播函数
    def forward(self, x, mask = None, context = None, pos_emb = None):
        # 获取输入张量的形状、设备、头数和缩放因子
        b, device, h, scale = x.shape[0], x.device, self.heads, self.scale

        # 获取键值对输入，默认为输入张量
        kv_input = default(context, x)

        # 分别对输入进行线性变换得到查询、键、值
        q, k, v = self.to_q(x), self.to_k(kv_input), self.to_v(kv_input)

        # 将查询、键、值按头数拆分
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 对查询进行缩放
        q = q * scale

        # 应用相对位置编码（旋转嵌入）
        if exists(pos_emb):
            q_pos_emb, k_pos_emb = cast_tuple(pos_emb, num = 2)
            q = apply_rotary_pos_emb(q, q_pos_emb)
            k = apply_rotary_pos_emb(k, k_pos_emb)

        # 添加空键/值
        if exists(self.null_k):
            nk, nv = self.null_k, self.null_v
            nk, nv = map(lambda t: repeat(t, '(h d) -> b h 1 d', b = b, h = h), (nk, nv))
            k = torch.cat((nk, k), dim = -2)
            v = torch.cat((nv, v), dim = -2)

        # 计算查询键相似度
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # 掩码
        mask_value = -torch.finfo(sim.dtype).max
        if exists(mask):
            if exists(self.null_k):
                mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)

        # 如果是因果注意力，进行掩码
        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones(i, j, device = device, dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        # 注意力计算
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # 聚合
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # 合并头部
        out = rearrange(out, 'b h n d -> b n (h d)')

        # 线性变换输出
        return self.to_out(out)
class ChunkedCrossAttention(nn.Module):
    def __init__(
        self,
        chunk_size,
        **kwargs
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.cross_attn = Attention(null_kv = True, **kwargs)

    def forward(self, x, *, context_mask = None, context, pos_emb = None):
        # derive variables
        chunk_size = self.chunk_size

        b, n, num_chunks, num_retrieved = x.shape[0], x.shape[-2], *context.shape[-4:-2]

        # if sequence length less than chunk size, do an early return
        if n < self.chunk_size:
            return torch.zeros_like(x)

        # causal padding
        causal_padding = chunk_size - 1

        x = F.pad(x, (0, 0, -causal_padding, causal_padding), value = 0.)

        # remove sequence which is ahead of the neighbors retrieved (during inference)
        seq_index = (n // chunk_size) * chunk_size
        x, x_remainder = x[:, :seq_index], x[:, seq_index:]

        seq_remain_len = x_remainder.shape[-2]

        # take care of rotary positional embedding
        # make sure queries positions are properly shifted to the future
        q_pos_emb, k_pos_emb = pos_emb
        q_pos_emb = F.pad(q_pos_emb, (0, 0, -causal_padding, causal_padding), value = 0.)

        k_pos_emb = repeat(k_pos_emb, 'b h n d -> b h (r n) d', r = num_retrieved)
        pos_emb = (q_pos_emb, k_pos_emb)

        # reshape so we have chunk to chunk attention, without breaking causality
        x = rearrange(x, 'b (k n) d -> (b k) n d', k = num_chunks)
        context = rearrange(context, 'b k r n d -> (b k) (r n) d')

        if exists(context_mask):
            context_mask = rearrange(context_mask, 'b k r n -> (b k) (r n)')

        # cross attention
        out = self.cross_attn(x, context = context, mask = context_mask, pos_emb = pos_emb)

        # reshape back to original sequence
        out = rearrange(out, '(b k) n d -> b (k n) d', b = b)

        # pad back to original, with 0s at the beginning (which will be added to the residual and be fine)
        out = F.pad(out, (0, 0, causal_padding, -causal_padding + seq_remain_len), value = 0.)
        return out

# encoder and decoder classes

class Encoder(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        context_dim = None,
        causal = False,
        heads = 8,
        dim_head = 64,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        final_norm = True,
        cross_attn_layers = None,
        post_norm = False,
        output_dim = None,
        norm_klass = RMSNorm,
        scale_residual = 1.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        # partial rotary embeddings, which is better than full rotary
        # Wang and Komatsuzaki et al https://github.com/kingoflolz/mesh-transformer-jax/

        rotary_emb_dim = min(dim_head, MIN_DIM_HEAD)
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim)

        wrapper = partial(PreNorm, dim, norm_klass = norm_klass) if not post_norm else partial(PostNorm, dim, scale_residual = scale_residual, norm_klass = norm_klass)

        for layer_num in range(1, depth + 1):
            has_cross_attn = not exists(cross_attn_layers) or layer_num in cross_attn_layers

            self.layers.append(nn.ModuleList([
                wrapper(Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, causal = causal)),
                wrapper(Attention(dim = dim, context_dim = context_dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)) if has_cross_attn else None,
                wrapper(FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)),
            ]))

        self.norm_out = norm_klass(dim) if final_norm and not post_norm else nn.Identity()
        self.project_out = nn.Linear(dim, output_dim) if exists(output_dim) else nn.Identity()
    # 定义一个前向传播函数，接受输入 x 和关键字参数 mask 和 chunked_seq
    def forward(self, x, *, mask = None, chunked_seq):
        # 获取输入 x 的设备信息、分块大小和序列长度
        device, chunk_size, seq_len = x.device, x.shape[-2], chunked_seq.shape[-2]

        # 生成查询位置编码
        q_pos_emb = self.rotary_pos_emb(chunk_size, device = device)
        # 生成键值位置编码
        k_pos_emb = self.rotary_pos_emb(seq_len, device = device)

        # 遍历每个层中的注意力、交叉注意力和前馈网络
        for attn, cross_attn, ff in self.layers:
            # 使用注意力机制处理输入 x，传入位置编码 q_pos_emb
            x = attn(x, mask = mask, pos_emb = q_pos_emb)

            # 如果存在交叉注意力层
            if exists(cross_attn):
                # 使用交叉注意力处理输入 x，传入上下文 chunked_seq 和位置编码 q_pos_emb、k_pos_emb
                x = cross_attn(x, context = chunked_seq, pos_emb = (q_pos_emb, k_pos_emb))

            # 使用前馈网络处理输入 x
            x = ff(x)

        # 对处理后的 x 进行输出层的归一化
        x = self.norm_out(x)
        # 对归一化后的 x 进行输出投影
        return self.project_out(x)
class Decoder(nn.Module):
    # 定义解码器类
    def __init__(
        self,
        dim,
        *,
        depth,
        heads = 8,
        dim_head = 64,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        final_norm = True,
        cross_attn_layers = None,
        chunk_size = 64,
        post_norm = False,
        norm_klass = RMSNorm,
        scale_residual = 1.
    ):
        # 初始化函数，设置解码器的参数
        super().__init__()
        self.layers = nn.ModuleList([])

        # 部分旋转嵌入，比完整旋转更好
        # 王和小松崎等人 https://github.com/kingoflolz/mesh-transformer-jax/
        rotary_emb_dim = min(dim_head, MIN_DIM_HEAD)
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim)

        wrapper = partial(PreNorm, dim, norm_klass = norm_klass) if not post_norm else partial(PostNorm, dim, scale_residual = scale_residual, norm_klass = norm_klass)

        self.chunk_size = chunk_size

        for layer_num in range(1, depth + 1):
            has_cross_attn = not exists(cross_attn_layers) or layer_num in cross_attn_layers

            self.layers.append(nn.ModuleList([
                wrapper(Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, causal = True)),
                wrapper(ChunkedCrossAttention(chunk_size = chunk_size, dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)) if has_cross_attn else None,
                wrapper(FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)),
            ]))

        self.norm_out = norm_klass(dim) if final_norm and not post_norm else nn.Identity()

    def forward(self, x, *, encoder = None, encoder_retrieved_mask = None, context_mask = None, retrieved = None):
        # 前向传播函数，接收输入 x 和其他参数
        device, seq_len = x.device, x.shape[-2]
        self_attn_pos_emb = self.rotary_pos_emb(seq_len, device = device)

        # 计算序列索引
        num_seq_chunks = seq_len // self.chunk_size
        seq_index = num_seq_chunks * self.chunk_size

        # 在检索的块上进行旋转位置
        if exists(retrieved):
            num_chunks, num_neighbors, chunk_size = retrieved.shape[-4:-1]

            cross_attn_q_pos_emb = self.rotary_pos_emb(self.chunk_size, device = device, offset = self.chunk_size - 1)  # 需要添加额外的块大小，因为它将被移位
            cross_attn_k_pos_emb = self.rotary_pos_emb(chunk_size, device = device)

            cross_attn_pos_emb = (cross_attn_q_pos_emb, cross_attn_k_pos_emb)

        # ��踪检索的标记是否已编码
        retrieved_encoded = False

        # 遍历解码器层
        for attn, cross_attn, ff in self.layers:
            x = attn(x, pos_emb = self_attn_pos_emb)

            if exists(cross_attn) and exists(retrieved):
                if not retrieved_encoded:
                    retrieved = rearrange(retrieved, 'b k r n d -> (b k r) n d')
                    seq_as_context = repeat(x[:, :seq_index], 'b (k n) d -> (b k r) n d', n = self.chunk_size, r = num_neighbors)

                    retrieved = encoder(retrieved, mask = encoder_retrieved_mask, chunked_seq = seq_as_context)
                    retrieved = rearrange(retrieved, '(b k r) n d -> b k r n d', k = num_chunks, r = num_neighbors)
                    retrieved_encoded = True

                x = cross_attn(
                    x,
                    context = retrieved,
                    context_mask = context_mask,
                    pos_emb = cross_attn_pos_emb
                )

            x = ff(x)

        return self.norm_out(x)

# 主类
class RETRO(nn.Module):
    # 定义主类
    # 初始化模型参数
    def __init__(
        self,
        *,
        num_tokens = BERT_VOCAB_SIZE,  # 设置词汇表大小，默认为BERT词汇表大小
        max_seq_len = 2048,  # 设置最大序列长度，默认为2048
        enc_dim = 896,  # 设置编码器维度，默认为896
        enc_depth = 2,  # 设置编码器深度，默认为2
        enc_cross_attn_layers = None,  # 设置编码器交叉注意力层，默认为None
        dec_depth = 12,  # 设置解码器深度，默认为12
        dec_cross_attn_layers = (1, 3, 6, 9),  # 设置解码器交叉注意力层，默认为(1, 3, 6, 9)
        heads = 8,  # 设置头数，默认为8
        dec_dim = 768,  # 设置解码器维度，默认为768
        dim_head = 64,  # 设置每个头的维度，默认为64
        enc_attn_dropout = 0.,  # 设置编码器注意力机制的dropout，默认为0
        enc_ff_dropout = 0.,  # 设置编码器前馈网络的dropout，默认为0
        dec_attn_dropout = 0.,  # 设置解码器注意力机制的dropout，默认为0
        dec_ff_dropout = 0.,  # 设置解码器前馈网络的dropout，默认为0
        chunk_size = 64,  # 设置块大小，默认为64
        pad_id = 0,  # 设置填充ID，默认为0
        enc_scale_residual = None,  # 设置编码器残差缩放，默认为None
        dec_scale_residual = None,  # 设置解码器残差缩放，默认为None
        norm_klass = None,  # 设置规范化类，默认为None
        gated_rmsnorm = False,  # 设置是否使用门控RMSNorm，默认为False
        use_deepnet = False  # 设置是否使用深度网络，默认为False
    ):
        super().__init__()
        assert dim_head >= MIN_DIM_HEAD, f'dimension per head must be greater than {MIN_DIM_HEAD}'  # 断言每个头的维度必须大于等于最小维度
        self.seq_len = max_seq_len  # 设置序列长度为最大序列长度
        self.pad_id = pad_id  # 设置填充ID

        self.token_emb = nn.Embedding(num_tokens, enc_dim)  # 创建词嵌入层
        self.pos_emb = nn.Embedding(max_seq_len, enc_dim)  # 创建位置嵌入层

        self.chunk_size = chunk_size  # 设置块大小

        self.to_decoder_model_dim = nn.Linear(enc_dim, dec_dim) if enc_dim != dec_dim else nn.Identity()  # 创建线性层，用于编码器到解码器维度转换

        # for deepnet, residual scales
        # follow equation in Figure 2. in https://arxiv.org/abs/2203.00555

        norm_klass = default(norm_klass, RMSNorm)  # 设置规范化类为默认值或RMSNorm

        if use_deepnet:
            enc_scale_residual = default(enc_scale_residual, 0.81 * ((enc_depth ** 4) * dec_depth) ** .0625)  # 如果使用深度网络，则设置编码器残��缩放
            dec_scale_residual = default(dec_scale_residual, (3 * dec_depth) ** 0.25)  # 如果使用深度网络，则设置解码器残差缩放
            norm_klass = nn.LayerNorm  # 如果使用深度网络，则设置规范化类为LayerNorm

        # allow for gated rmsnorm

        if gated_rmsnorm:
            norm_klass = partial(RMSNorm, gated = True)  # 如果使用门控RMSNorm，则设置规范化类为带有门控的RMSNorm

        # define encoder and decoders

        self.encoder = Encoder(
            dim = enc_dim,
            context_dim = dec_dim,
            dim_head = dim_head,
            depth = enc_depth,
            attn_dropout = enc_attn_dropout,
            ff_dropout = enc_ff_dropout,
            cross_attn_layers = enc_cross_attn_layers,
            post_norm = use_deepnet,
            norm_klass = norm_klass,
            scale_residual = enc_scale_residual,
            output_dim = dec_dim
        )  # 定义编码器

        self.decoder = Decoder(
            dim = dec_dim,
            depth = dec_depth,
            dim_head = dim_head,
            attn_dropout = dec_attn_dropout,
            ff_dropout = dec_ff_dropout,
            cross_attn_layers = dec_cross_attn_layers,
            chunk_size = chunk_size,
            post_norm = use_deepnet,
            norm_klass = norm_klass,
            scale_residual = dec_scale_residual
        )  # 定义解码器

        self.to_logits = nn.Linear(dec_dim, num_tokens)  # 创建线性层，用于将解码器输出映射到词汇表大小

        # deepnet has special init of weight matrices

        if use_deepnet:
            deepnorm_init(self.encoder, 0.87 * ((enc_depth ** 4) * dec_depth) ** -0.0625)  # 如果使用深度网络，则初始化编码器
            deepnorm_init(self.decoder, (12 * dec_depth) ** -0.25)  # 如果使用深度网络，则初始化解码器

    def forward_without_retrieval(
        self,
        seq
    ):
        # embed sequence

        embed = self.token_emb(seq)  # 对序列进行词嵌入
        embed = embed[:, :self.seq_len]  # 截取指定长度的嵌入序列

        # get absolute positional embedding

        pos_emb = self.pos_emb(torch.arange(embed.shape[1], device = embed.device))  # 获取绝对位置嵌入
        pos_emb = rearrange(pos_emb, 'n d -> 1 n d')  # 重新排列位置嵌入
        embed = embed + pos_emb  # 将位置嵌入加到词嵌入上

        embed = self.to_decoder_model_dim(embed)  # 将嵌入转换到解码器模型维度
        embed = self.decoder(embed)  # 解码器处理嵌入序列

        # project to logits

        return self.to_logits(embed)  # 将解码器输出映射到词汇表大小

    def forward(
        self,
        seq,
        retrieved = None,
        return_loss = False
        """
        b - batch
        n - sequence length / chunk length
        k - number of chunks
        d - feature dimension
        r - num retrieved neighbors
        """

        # 如果没有提供retrieved参数，则直接调用forward_without_retrieval方法
        if not exists(retrieved):
            return self.forward_without_retrieval(seq)

        # 断言只有在训练时才能返回损失
        assert not (return_loss and not self.training), 'must be training if returning loss'

        # 假设填充标记ID（通常为0）需要被屏蔽掉
        mask = retrieved != self.pad_id

        # 处理一些用户输入
        if retrieved.ndim == 3:
            # 重新排列retrieved的维度，将'n'维度变为1
            retrieved = rearrange(retrieved, 'b k n -> b k 1 n') # 1 neighbor retrieved

        # 如果需要返回损失，则推导标签
        if return_loss:
            seq, labels = seq[:, :-1], seq[:, 1:]

        # 定义变量
        n, num_chunks, num_neighbors, chunk_size, retrieved_shape, device = seq.shape[-1], *retrieved.shape[-3:], retrieved.shape, seq.device

        # 断言检查retrieved输入的chunk_size必须大于等于RETRO初始化时指定的chunk_size
        assert chunk_size >= self.chunk_size, 'chunk size of retrieval input must be greater or equal to the designated chunk_size on RETRO initialization'

        # 计算序列需要的chunk数量，并检查传入的num_chunks是否符合要求
        num_seq_chunks = n // self.chunk_size
        assert num_chunks == num_seq_chunks, f'sequence requires {num_seq_chunks} retrieved chunks, but only {num_chunks} passed in'

        # 计算还未获取k个最近邻的序列索引
        seq_index = num_seq_chunks * self.chunk_size

        # 对序列和retrieved chunks进行嵌入
        embed = self.token_emb(seq)
        retrieved = self.token_emb(retrieved)

        # 获取绝对位置嵌入
        pos_emb = self.pos_emb(torch.arange(n, device=device))
        pos_emb = rearrange(pos_emb, 'n d -> 1 n d')
        embed = embed + pos_emb

        # 如果需要，处理编码器和解码器的掩码
        encoder_retrieved_mask = decoder_retrieved_mask = None
        if exists(mask):
            assert mask.shape == retrieved_shape, 'retrieval mask must be of the same shape as the retrieval tokens'
            encoder_retrieved_mask = rearrange(mask, 'b k r n -> (b k r) n')
            decoder_retrieved_mask = mask

        # 如果需要，将序列嵌入和retrieved嵌入投影到解码器维度
        embed = self.to_decoder_model_dim(embed)

        # 解码
        embed = self.decoder(
            embed,
            encoder=self.encoder,
            context_mask=decoder_retrieved_mask,
            encoder_retrieved_mask=encoder_retrieved_mask,
            retrieved=retrieved
        )

        # 投影到logits
        logits = self.to_logits(embed)

        # 如果不需要返回损失，则返回logits
        if not return_loss:
            return logits

        # 计算交叉熵损失
        loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels, ignore_index=self.pad_id)
        return loss
```