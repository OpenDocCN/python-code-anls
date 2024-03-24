# `.\lucidrains\linear-attention-transformer\linear_attention_transformer\linear_attention_transformer.py`

```
# 导入 PyTorch 库
import torch
# 导入 PyTorch 中的函数库
import torch.nn.functional as F
# 从 torch 中导入 nn, einsum 模块
from torch import nn, einsum
# 导入 math 库
import math
# 从 operator 中导入 mul 函数
from operator import mul
# 从 math 中导入 gcd 函数
from math import gcd
# 从 collections 中导入 namedtuple 模块
from collections import namedtuple
# 从 functools 中导入 partial, reduce 函数
from functools import partial, reduce

# 导入自定义模块
from local_attention import LocalAttention
from linformer import LinformerSelfAttention
from product_key_memory import PKM
from axial_positional_embedding import AxialPositionalEmbedding
from linear_attention_transformer.reversible import ReversibleSequence, SequentialSequence
from einops import rearrange, repeat

# 定义 namedtuple 类型 LinformerSettings
LinformerSettings = namedtuple('LinformerSettings', ['k'])
# 定义 namedtuple 类型 LinformerContextSettings
LinformerContextSettings = namedtuple('LinformerContextSettings', ['seq_len', 'k'])

# 辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(value, d):
    return d if not exists(value) else value

# 返回固定值的函数
def always(value):
    return lambda *args, **kwargs: value

# 将值转换为元组
def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val

# 安全除法
def safe_div(n, d, eps = 1e-6):
    return n.div_(d + eps)

# 最小公倍数
def lcm(*numbers):
    return int(reduce(lambda x, y: int((x * y) / gcd(x, y)), numbers, 1)

# 合并张量的维度
def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)

# 扩展张量的维度
def expand_dim(t, dim, k, unsqueeze=True):
    if unsqueeze:
        t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

# 在指定索引处分割张量
def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

# 获取张量的最小负值
def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

# 辅助类

# 预归一化
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# 分块
class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim = self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim = self.dim)

# 输入输出投影
class ProjectInOut(nn.Module):
    def __init__(self, fn, dim_in, dim_out, project_out = True):
        super().__init__()
        self.fn = fn
        self.project_in = nn.Linear(dim_in, dim_out)
        self.project_out = nn.Linear(dim_out, dim_in) if project_out else nn.Identity()

    def forward(self, x, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, **kwargs)
        x = self.project_out(x)
        return x

# 令牌移位辅助类

# 移位函数
def shift(t, amount, mask = None):
    if amount == 0:
        return t

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    return F.pad(t, (0, 0, amount, -amount), value = 0.)

# 预移位令牌
class PreShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim = -1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args, mask = mask), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim = -1)
        return self.fn(x, **kwargs)

# 位置嵌入

# 绝对位置嵌入
class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)
    # 定义一个前向传播函数，接受输入张量 x
    def forward(self, x):
        # 生成一个与输入张量 x 的第二维相同长度的张量 t，元素为从 0 到 x.shape[1]-1
        t = torch.arange(x.shape[1], device=x.device)
        # 使用嵌入层 emb 对 t 进行嵌入操作，得到一个新的张量，维度为 [1, t的长度, 嵌入维度]
        return self.emb(t)[None, :, :]
# 定义固定位置嵌入类，用于生成固定位置嵌入
class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        # 计算频率的倒数，用于生成正弦和余弦位置编码
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        # 生成位置向量
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        # 计算正弦和余弦位置编码
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        # 将正弦和余弦位置编码拼接在一起
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        # 将位置嵌入注册为缓冲区
        self.register_buffer('emb', emb)

    def forward(self, x):
        # 返回位置嵌入
        return self.emb[None, :x.shape[1], :].to(x)

# 旋转位置嵌入的辅助函数
# 将输入张量中的每两个元素进行旋转
def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

# 应用旋转位置嵌入
def apply_rotory_pos_emb(q, k, sinu_pos):
    # 重新排列正弦位置编码的形状
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j = 2)
    sin, cos = sinu_pos.unbind(dim = -2)
    # 将正弦和余弦位置编码重复到与输入张量相同的形状
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j = 2), (sin, cos))
    # 应用旋转位置嵌入到查询和键中
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k

# 前馈神经网络
# GELU激活函数
class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))

# 如果PyTorch中存在GELU函数，则使用PyTorch中的GELU，否则使用自定义的GELU_
GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

# 前馈神经网络类
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        # 第一个线性层
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        # 第二个线性层
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x

# 自注意力层
# 线性注意力函数
def linear_attn(q, k, v, kv_mask = None):
    dim = q.shape[-1]

    if exists(kv_mask):
        mask_value = max_neg_value(q)
        mask = kv_mask[:, None, :, None]
        k = k.masked_fill_(~mask, mask_value)
        v = v.masked_fill_(~mask, 0.)
        del mask

    q = q.softmax(dim=-1)
    k = k.softmax(dim=-2)

    q = q * dim ** -0.5

    context = einsum('bhnd,bhne->bhde', k, v)
    attn = einsum('bhnd,bhde->bhne', q, context)
    return attn.reshape(*q.shape)

# 因果线性注意力函数
def causal_linear_attn(q, k, v, kv_mask = None, bucket_size = None, eps = 1e-3):
    b, h, n, e, dtype = *q.shape, q.dtype
    bucket_size = default(bucket_size, 64)
    bucket_size = max(bucket_size, 1)
    assert bucket_size == 0 or (n % bucket_size) == 0, f'sequence length {n} must be divisible by the bucket size {bucket_size} for causal linear attention'

    q = q.softmax(dim=-1)
    k = torch.exp(k).type(dtype).clone()

    q = q * e ** -0.5

    if exists(kv_mask):
        mask = kv_mask[:, None, :, None]
        k = k.masked_fill_(~mask, 0.)
        v = v.masked_fill_(~mask, 0.)
        del mask

    bucket_fn = lambda x: x.reshape(*x.shape[:-2], -1, bucket_size, e)
    b_q, b_k, b_v = map(bucket_fn, (q, k, v))

    b_k_sum = b_k.sum(dim=-2)
    b_k_cumsum = b_k_sum.cumsum(dim = -2).type(dtype)

    context = einsum('bhund,bhune->bhude', b_k, b_v)
    context = context.cumsum(dim = -3).type(dtype)

    if bucket_size > 1:
        context = F.pad(context, (0, 0, 0, 0, 1, 0), value = 0.)
        context, _ = split_at_index(2, -1, context)

        b_k_cumsum = F.pad(b_k_cumsum, (0, 0, 1, 0), value = 0.)
        b_k_cumsum, _ = split_at_index(2, -1, b_k_cumsum)

    D_inv = 1. / einsum('bhud,bhund->bhun', b_k_cumsum, b_q).clamp(min = eps)
    attn = einsum('bhund,bhude,bhun->bhune', b_q, context, D_inv)
    return attn.reshape(*q.shape)

# 自注意力层类
class SelfAttention(nn.Module):
    # 初始化函数，设置模型参数
    def __init__(self, dim, heads, causal = False, dim_head = None, blindspot_size = 1, n_local_attn_heads = 0, local_attn_window_size = 128, receives_context = False, dropout = 0., attn_dropout = 0.):
        # 调用父类初始化函数
        super().__init__()
        # 检查维度是否可以被头数整除
        assert dim_head or (dim % heads) == 0, 'embedding dimension must be divisible by number of heads'
        # 设置每个头的维度
        d_heads = default(dim_head, dim // heads)

        # 初始化模型参数
        self.heads = heads
        self.d_heads = d_heads
        self.receives_context = receives_context

        # 设置全局注意力头数和函数
        self.global_attn_heads = heads - n_local_attn_heads
        self.global_attn_fn = linear_attn if not causal else partial(causal_linear_attn, bucket_size = blindspot_size)

        # 设置局部注意力头数和局部注意力对象
        self.local_attn_heads = n_local_attn_heads
        self.local_attn  = LocalAttention(local_attn_window_size, causal = causal, dropout = attn_dropout)

        # 线性变换得到查询、键、值
        self.to_q = nn.Linear(dim, d_heads * heads, bias = False)

        kv_heads = heads

        self.kv_heads = kv_heads
        self.to_k = nn.Linear(dim, d_heads * kv_heads, bias = False)
        self.to_v = nn.Linear(dim, d_heads * kv_heads, bias = False)

        # 线性变换得到输出
        self.to_out = nn.Linear(d_heads * heads, dim)
        self.dropout = nn.Dropout(dropout)

    # 前向传播函数
    def forward(self, x, input_mask = None, context = None, context_mask = None, pos_emb = None, **kwargs):
        # 如果模型需要上下文信息但未提供，则报错
        assert not (self.receives_context and not exists(context)), 'context must be supplied if self attention is in receives context mode'

        # 根据是否需要上下文信息，获取查询、键、值
        if not self.receives_context:
            q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))
        else:
            q, k, v = (self.to_q(x), self.to_k(context), self.to_v(context))

        b, t, e, h, dh = *q.shape, self.heads, self.d_heads

        # 合并头部维度
        merge_heads = lambda x: x.reshape(*x.shape[:2], -1, dh).transpose(1, 2)

        q, k, v = map(merge_heads, (q, k, v))

        # 如果存在位置编码且不需要上下文信息，则应用旋转位置编码
        if exists(pos_emb) and not self.receives_context:
            q, k = apply_rotory_pos_emb(q, k, pos_emb)

        out = []

        # 分割索引函数，用于分割局部和全局注意力
        split_index_fn = partial(split_at_index, 1, self.local_attn_heads)

        (lq, q), (lk, k), (lv, v) = map(split_index_fn, (q, k, v))

        has_local, has_global = map(lambda x: x.shape[1] > 0, (lq, q))

        # 如果存在局部注意力，则计算局部注意力
        if has_local:
            local_out = self.local_attn(lq, lk, lv, input_mask = input_mask)
            out.append(local_out)

        # 如果存在全局注意力，则计算全局注意力
        if has_global:
            kv_mask = input_mask if not self.receives_context else context_mask
            global_out = self.global_attn_fn(q, k, v, kv_mask = kv_mask)
            out.append(global_out)

        # 拼接注意力结果并返回
        attn = torch.cat(out, dim=1)
        attn = attn.transpose(1, 2).reshape(b, t, -1)
        return self.dropout(self.to_out(attn))
# 定义 FoldAxially 类，用于将输入张量按轴进行折叠
class FoldAxially(nn.Module):
    def __init__(self, axial_dim, fn):
        super().__init__()
        self.fn = fn
        self.axial_dim = axial_dim
    # 前向传播函数，对输入张量进行处理
    def forward(self, x, input_mask = None, **kwargs):
        # 获取输入张量的形状信息
        b, t, d, ax = *x.shape, self.axial_dim
        # 将输入张量按轴进行折叠和转置
        x = x.reshape(b, -1, ax, d).transpose(1, 2).reshape(b * ax, -1, d)

        # 初始化 mask 为 None
        mask = None
        # 如果输入的 mask 存在
        if exists(input_mask):
            # 将 mask 按轴进行折叠和转置
            mask = input_mask.reshape(b, -1, ax).transpose(1, 2).reshape(b * ax, -1)

        # 对折叠后的张量进行处理
        x = self.fn(x, input_mask = mask, **kwargs)
        # 将处理后的张量还原为原始形状
        x = x.reshape(b, ax, -1, d).transpose(1, 2).reshape(b, t, d)
        return x

# 定义 LinearAttentionTransformer 类，用于实现线性注意力变换器
class LinearAttentionTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        max_seq_len,
        heads = 8,
        dim_head = None,
        bucket_size = 64,
        causal = False,
        ff_chunks = 1,
        ff_glu = False,
        ff_dropout = 0.,
        attn_layer_dropout = 0.,
        attn_dropout = 0.,
        reversible = False,
        blindspot_size = 1,
        n_local_attn_heads = 0,
        local_attn_window_size = 128,
        receives_context = False,
        attend_axially = False,
        pkm_layers = tuple(),
        pkm_num_keys = 128,
        linformer_settings = None,
        context_linformer_settings = None,
        shift_tokens = False
        ):
            # 调用父类的构造函数
            super().__init__()
            # 断言条件，确保 Linformer 自注意力层仅用于非因果网络
            assert not (causal and exists(linformer_settings)), 'Linformer self attention layer can only be used for non-causal networks'
            # 断言条件，确保 Linformer 自注意力设置是 LinformerSettings 命名元组
            assert not exists(linformer_settings) or isinstance(linformer_settings, LinformerSettings), 'Linformer self-attention settings must be a LinformerSettings namedtuple'
            # 断言条件，确保 Linformer 上下文自注意力设置是 LinformerSettings 命名元组
            assert not exists(context_linformer_settings) or isinstance(context_linformer_settings, LinformerContextSettings), 'Linformer contextual self-attention settings must be a LinformerSettings namedtuple'

            # 如果 n_local_attn_heads 不是元组，则将其转换为深度个相同元素的元组
            if type(n_local_attn_heads) is not tuple:
                n_local_attn_heads = tuple([n_local_attn_heads] * depth)

            # 断言条件，确保本地注意力头元组的长度与深度相同
            assert len(n_local_attn_heads) == depth, 'local attention heads tuple must have the same length as the depth'
            # 断言条件，确保每个本地注意力头数小于最大头数
            assert all([(local_heads <= heads) for local_heads in n_local_attn_heads]), 'number of local attn heads must be less than the maximum number of heads'

            # 初始化层列表
            layers = nn.ModuleList([])

            # 遍历深度和本地注意力头数
            for ind, local_heads in zip(range(depth), n_local_attn_heads):
                # 计算层编号
                layer_num = ind + 1
                # 检查是否使用 PKM
                use_pkm = layer_num in cast_tuple(pkm_layers)

                # 如果不使用 Linformer 设置，则创建 SelfAttention 对象
                if not exists(linformer_settings):
                    attn = SelfAttention(dim, heads, causal, dim_head = dim_head, blindspot_size = blindspot_size, n_local_attn_heads = local_heads, local_attn_window_size = local_attn_window_size, dropout = attn_layer_dropout, attn_dropout= attn_dropout)
                # 否则创建 LinformerSelfAttention 对象
                else:
                    attn = LinformerSelfAttention(dim, max_seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout, **linformer_settings._asdict())

                # 如果需要移动标记，则进行标记移动
                if shift_tokens:
                    shifts = (1, 0, -1) if not causal else (1, 0)
                    attn, parallel_net = map(partial(PreShiftTokens, shifts), (attn, parallel_net))

                # 将 SelfAttention 和 FeedForward 添加到层列表中
                layers.append(nn.ModuleList([
                    PreNorm(dim, attn),
                    PreNorm(dim, parallel_net)
                ]))

                # 如果需要轴向关注，则添加到层列表中
                if attend_axially:
                    layers.append(nn.ModuleList([
                        PreNorm(dim, FoldAxially(local_attn_window_size, SelfAttention(dim, heads, causal, dropout = attn_layer_dropout, attn_dropout= attn_dropout))),
                        PreNorm(dim, Chunk(ff_chunks, FeedForward(dim, glu = ff_glu, dropout= ff_dropout), along_dim = 1))
                    ]))

                # 如果接收上下文，则添加到层列表中
                if receives_context:
                    if not exists(context_linformer_settings):
                        attn = SelfAttention(dim, heads, dim_head = dim_head, dropout = attn_layer_dropout, attn_dropout= attn_dropout, receives_context = True)
                    else:
                        attn = LinformerSelfAttention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout, **context_linformer_settings._asdict())

                    layers.append(nn.ModuleList([
                        PreNorm(dim, attn),
                        PreNorm(dim, Chunk(ff_chunks, FeedForward(dim, glu = ff_glu, dropout= ff_dropout), along_dim = 1))
                    ]))

            # 根据是否可逆选择执行类型
            execute_type = ReversibleSequence if reversible else SequentialSequence

            # 设置轴向层和上下文层
            axial_layer = ((True, False),) if attend_axially else tuple()
            attn_context_layer = ((True, False),) if receives_context else tuple()
            route_attn = ((True, False), *axial_layer, *attn_context_layer) * depth
            route_context = ((False, False), *axial_layer, *attn_context_layer) * depth

            # 根据接收上下文情况设置路由映射
            context_route_map = {'context': route_context, 'context_mask': route_context} if receives_context else {}
            attn_route_map = {'input_mask': route_attn, 'pos_emb': route_attn}
            # 创建层序列对象
            self.layers = execute_type(layers, args_route = {**attn_route_map, **context_route_map})

            # 计算填充到的倍数
            self.pad_to_multiple = lcm(
                1 if not causal else blindspot_size,
                1 if all([(h == 0) for h in n_local_attn_heads]) else local_attn_window_size
            )
    # 定义一个 forward 方法，用于前向传播计算
    def forward(self, x, **kwargs):
        # 调用 self.layers 方法，传入输入 x 和其他参数 kwargs，返回计算结果
        return self.layers(x, **kwargs)
class LinearAttentionTransformerLM(nn.Module):
    # 定义线性注意力变换器语言模型类
    def __init__(
        self,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        heads = 8,
        dim_head = 64,
        causal = False,
        emb_dim = None,
        reversible = False,
        ff_chunks = 1,
        ff_glu = False,
        ff_dropout = 0.,
        attn_layer_dropout = 0.,
        attn_dropout = 0.,
        blindspot_size = 1,
        n_local_attn_heads = 0,
        local_attn_window_size = 128,
        return_embeddings = False,
        receives_context = False,
        pkm_layers = tuple(),
        pkm_num_keys = 128,
        attend_axially = False,
        linformer_settings = None,
        context_linformer_settings = None,
        use_axial_pos_emb = True,
        use_rotary_emb = False,
        shift_tokens = False
    ):
        # 初始化函数，接受多个参数
        assert n_local_attn_heads == 0 or (max_seq_len % local_attn_window_size) == 0, 'max sequence length must be divisible by the local attention window size'
        # 断言语句，确保本地注意力头数为0或最大序列长度能被本地注意力窗口大小整除
        super().__init__()
        # 调用父类的初始化函数

        emb_dim = default(emb_dim, dim)
        # 如果emb_dim为None，则使用dim作为默认值
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        # 创建一个嵌入层，用于将输入的token转换为向量表示

        if use_rotary_emb:
            self.pos_emb = FixedPositionalEmbedding(emb_dim, max_seq_len)
            self.layer_pos_emb = FixedPositionalEmbedding(dim_head, max_seq_len)
        elif use_axial_pos_emb:
            self.pos_emb = AxialPositionalEmbedding(emb_dim, axial_shape=(math.ceil(max_seq_len / local_attn_window_size), local_attn_window_size))
            self.layer_pos_emb = always(None)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len)
            self.layer_pos_emb = always(None)
        # 根据不同的位置编码方式，初始化位置编码层

        self.transformer = LinearAttentionTransformer(dim, depth, max_seq_len, heads = heads, dim_head = dim_head, causal = causal, ff_chunks = ff_chunks, ff_glu = ff_glu, ff_dropout = ff_dropout, attn_layer_dropout = attn_layer_dropout, attn_dropout = attn_dropout, reversible = reversible, blindspot_size = blindspot_size, n_local_attn_heads = n_local_attn_heads, local_attn_window_size = local_attn_window_size, receives_context = receives_context, pkm_layers = pkm_layers, pkm_num_keys = pkm_num_keys, attend_axially = attend_axially, linformer_settings = linformer_settings, context_linformer_settings = context_linformer_settings, shift_tokens = shift_tokens)
        # 创建线性注意力变换器模型

        if emb_dim != dim:
            self.transformer = ProjectInOut(self.transformer, emb_dim, dim, project_out = not return_embeddings)
        # 如果emb_dim不等于dim，则使用ProjectInOut函数将维度转换为dim

        self.norm = nn.LayerNorm(emb_dim)
        # 创建一个LayerNorm层，用于归一化
        self.out = nn.Linear(emb_dim, num_tokens) if not return_embeddings else nn.Identity()
        # 创建一个线性层，用于输出结果

    def forward(self, x, **kwargs):
        # 前向传播函数，接受输入x和关键字参数kwargs
        x = self.token_emb(x)
        # 将输入x通过token_emb转换为向量表示
        x = x + self.pos_emb(x).type(x.type())
        # 将位置编码加到输入x上

        layer_pos_emb = self.layer_pos_emb(x)
        # 获取层级位置编码
        x = self.transformer(x, pos_emb = layer_pos_emb, **kwargs)
        # 使用transformer处理输入x和位置编码
        x = self.norm(x)
        # 对输出进行归一化
        return self.out(x)
        # 返回输出结果
```