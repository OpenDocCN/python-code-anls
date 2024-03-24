# `.\lucidrains\memory-transformer-xl\memory_transformer_xl\memory_transformer_xl.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 torch 库中导入 nn.functional 模块，并重命名为 F
import torch.nn.functional as F

# 从 mogrifier 模块中导入 Mogrifier 类
from mogrifier import Mogrifier

# 导入 math 库
import math
# 从 collections 模块中导入 namedtuple 类
from collections import namedtuple
# 从 functools 模块中导入 partial 函数
from functools import partial
# 从 inspect 模块中导入 isfunction 函数
from inspect import isfunction

# 定义一个名为 Memory 的命名元组，包含 short 和 long 两个字段
Memory = namedtuple('Memory', ['short', 'long'])

# 定义辅助函数

# 返回一个字典，包含输入张量的数据类型和设备信息
def to(t):
    return {'dtype': t.dtype, 'device': t.device}

# 如果输入元素 el 不是元组，则将其转换为元组
def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)

# 如果输入值 x 不为 None，则返回 x，否则返回 val 或 val() 的结果（如果 val 是函数）
def default(x, val):
    if x is not None:
        return x
    return val if not isfunction(val) else val()

# 返回输入张量的最小负值
def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

# 在指定维度上重新塑形张量
def reshape_dim(t, dim, split_dims):
    shape = list(t.shape)
    num_dims = len(shape)
    dim = (dim + num_dims) % num_dims
    shape[dim:dim+1] = split_dims
    return t.reshape(shape)

# 在指定维度上将张量拆分为两部分
def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

# 在指定维度上创建一个先进先出队列
def queue_fifo(*args, length, dim=-2):
    queue = torch.cat(args, dim=dim)
    if length > 0:
        return split_at_index(dim, -length, queue)

    device = queue.device
    shape = list(queue.shape)
    shape[dim] = 0
    return queue, torch.empty(shape, device=device)

# 将输入张量在最后一个维度上进行循环移位
def shift(x):
    *_, i, j = x.shape
    zero_pad = torch.zeros((*_, i, i), **to(x))
    x = torch.cat([x, zero_pad], -1)
    l = i + j - 1
    x = x.view(*_, -1)
    zero_pad = torch.zeros(*_, -x.size(-1) % l, **to(x))
    shifted = torch.cat([x, zero_pad], -1).view(*_, -1, l)
    return shifted[..., :i, i - 1:]

# 迭代张量的第一个维度
def iterate_tensor(t):
    length = t.shape[0]
    for ind in range(length):
        yield t[ind]

# 初始化具有指定形状和维度的参数张量
def init_parameter(shape, dim):
    t = torch.zeros(shape)
    std = 1 / math.sqrt(dim)
    t.uniform_(-std, std)
    return nn.Parameter(t)

# 定义辅助类

# 定义一个具有残差连接的模块
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 定义一个具有预层归一化的模块
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# 定义神经调制的双稳态循环单元和其他门控类

# 定义一个神经调制的双稳态循环单元类
class nBRC(nn.Module):
    def __init__(self, dims, hidden_dims):
        super().__init__()
        self.Ua = nn.Linear(dims, hidden_dims)
        self.Wa = nn.Linear(dims, hidden_dims)
        self.Uc = nn.Linear(dims, hidden_dims)
        self.Wc = nn.Linear(dims, hidden_dims)
        self.U  = nn.Linear(dims, hidden_dims)

    def forward(self, x, h):
        l = lambda linear, tensor: F.linear(tensor, linear.weight.clone(), linear.bias.clone())

        a = 1 + torch.tanh(l(self.Ua, x) + l(self.Wa, h))
        c = torch.sigmoid(l(self.Uc, x) + l(self.Wc, h))
        return c * h + (1 - c) * torch.tanh(l(self.U, x) + a * h)

# 定义一个门控类，使用 GRU 作为门控单元
class GRUGating(nn.Module):
    def __init__(self, dim, fn, mogrify=False):
        super().__init__()
        self.dim = dim
        self.fn = fn
        self.gru = nBRC(dim, dim)
        self.mogrify = Mogrifier(dim, factorize_k=dim // 4) if mogrify else None

    def forward(self, x, **kwargs):
        shape = x.shape
        dim = self.dim

        y = self.fn(x, **kwargs)

        if self.mogrify is not None:
            y, x = self.mogrify(y, x)

        gated_output = self.gru(
            y.reshape(-1, dim),
            x.reshape(-1, dim)
        )

        return gated_output.reshape(shape)

# feedforward

# 定义 GELU 激活函数类
class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))

# 如果 nn 模块中存在 GELU 函数，则使用 nn.GELU，否则使用自定义的 GELU_ 函数
GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

# 定义一个前馈神经网络类
class FeedForward(nn.Module):
    # 初始化神经网络模块，设置输入维度、倍数、dropout率、激活函数和是否使用GLU
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        # 调用父类的初始化方法
        super().__init__()
        # 设置默认激活函数为GELU
        activation = default(activation, GELU)

        # 是否使用GLU
        self.glu = glu
        # 第一层线性变换，输入维度为dim，输出维度为dim * mult * (2 if glu else 1)
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        # 激活函数
        self.act = activation()
        # dropout层
        self.dropout = nn.Dropout(dropout)
        # 第二层线性变换，输入维度为dim * mult，输出维度为dim
        self.w2 = nn.Linear(dim * mult, dim)

    # 前向传播函数
    def forward(self, x, **kwargs):
        # 如果不使用GLU
        if not self.glu:
            # 第一层线性变换
            x = self.w1(x)
            # 激活函数
            x = self.act(x)
        else:
            # 使用GLU
            # 将第一层线性变换的输出分成两部分
            x, v = self.w1(x).chunk(2, dim=-1)
            # 激活函数作用在其中一部分上，另一部分保持不变
            x = self.act(x) * v

        # dropout层
        x = self.dropout(x)
        # 第二层线性变换
        x = self.w2(x)
        # 返回结果
        return x
# 定义自注意力机制类
class SelfAttention(nn.Module):
    def __init__(self, dim, seq_len, mem_len, lmem_len, heads = 8, attn_dropout = 0., dropout = 0., memory_attn_dropout = 0., one_kv_head = False, num_mem_kv = 4):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.heads = heads
        self.dim_head = dim // heads
        self.seq_len = seq_len
        self.mem_len = mem_len
        self.lmem_len = lmem_len
        self.scale = self.dim_head ** (-0.5)

        self.to_q = nn.Linear(dim, dim, bias = False)

        kv_dim = self.dim_head if one_kv_head else dim
        self.to_kv = nn.Linear(dim, kv_dim * 2, bias = False)
        self.to_out = nn.Linear(dim, dim)

        self.mem_kv = init_parameter((1, num_mem_kv, dim), dim)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout = nn.Dropout(dropout)

        self.memory_attn_dropout = nn.Dropout(memory_attn_dropout)

    def forward(self, x, memories = None, pos_emb = None, input_mask = None, calc_memory = True, **kwargs):
        b, t, e, h, dim_h = *x.shape, self.heads, self.dim_head

        memories = default(memories, (None, None))
        mem, lmem = memories

        init_mem = lambda: torch.empty(b, 0, e, **to(x))
        mem = default(mem, init_mem)
        lmem = default(lmem, init_mem)
        mem_kv = self.mem_kv.expand(b, -1, -1)

        mem_len, lmem_len, mem_kv_len = map(lambda t: t.shape[1], (mem, lmem, mem_kv))

        q = self.to_q(x)

        kv_input = torch.cat((mem_kv, lmem, mem, x), dim=1)
        kv_len = kv_input.shape[1]
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        merge_heads = lambda x: reshape_dim(x, -1, (-1, dim_h)).transpose(1, 2)
        q, k, v = map(merge_heads, (q, k, v))

        k, v = map(lambda x: x.expand(-1, h, -1, -1), (k, v))

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = max_neg_value(dots)

        if pos_emb is not None:
            pos_emb = pos_emb[:, -kv_len:].type(q.dtype)
            pos_dots = torch.einsum('bhid,hjd->bhij', q, pos_emb) * self.scale
            pos_dots = shift(pos_dots)
            pos_dots = F.pad(pos_dots, (dots.shape[-1] - pos_dots.shape[-1], 0), value = 0.)
            dots = dots + pos_dots

        if input_mask is not None:
            mask = input_mask[:, None, :, None] * input_mask[:, None, None, :]
            mask = F.pad(mask, (mem_len + lmem_len + mem_kv_len, 0), value = True)
            dots.masked_fill_(~mask, mask_value)

        total_mem_len = mem_len + lmem_len + mem_kv_len
        mask = torch.ones(t, t + total_mem_len, **to(x)).triu_(diagonal = 1 + total_mem_len).bool()
        dots.masked_fill_(mask[None, None, ...], mask_value)

        attn = dots.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(b, t, -1)
        out = self.to_out(out)

        return self.dropout(out)

# 定义线性注意力机制函数
def linear_attn(q, k, v):
    q, k = q.softmax(dim=-1), k.softmax(dim=-2)
    context = torch.einsum('bhnd,bhne->bhde', k, v)
    out = torch.einsum('bhnd,bhde->bhne', q, context)
    return out

# 定义全连接注意力机制函数
def full_attn(q, k, v):
    dots = torch.einsum('bhid,bhjd->bhij', q, k) * q.shape[-1] ** -0.5
    dots = dots.softmax(dim=-1)
    out = torch.einsum('bhij,bhjd->bhid', dots, v)
    return out

# 定义线性自注意力类
class LinearSelfAttention(nn.Module):
    def __init__(self, dim, depth, heads = 8):
        super().__init__()
        self.dim_head = dim // heads
        self.norm = nn.LayerNorm(dim, elementwise_affine = False)

        self.to_q = init_parameter((dim, dim), dim)
        self.to_kv = init_parameter((dim, 2 * dim), dim)
        self.to_out = init_parameter((dim, dim), dim)
    # 定义一个前向传播函数，接受输入 x 和隐藏状态 hiddens，默认为 None
    def forward(self, x, hiddens = None):
        # 获取头部维度
        dim_head = self.dim_head
        # 复制权重矩阵 w_q, w_kv, w_out
        w_q, w_kv, w_out = map(torch.clone, (self.to_q, self.to_kv, self.to_out))
        
        # 对输入 x 进行归一化处理
        normed_lmem = self.norm(x)
        # 计算查询向量 q
        q = torch.einsum('bnd,de->bne', normed_lmem, w_q)

        # 将输入 x 和隐藏状态 hiddens 拼接在一起作为键值对输入
        kv_input = torch.cat((normed_lmem, hiddens), dim=1)
        # 计算键 k 和值 v
        k, v = torch.einsum('bnd,de->bne', kv_input, w_kv).chunk(2, dim=-1)

        # 将查询 q、键 k、值 v 进行维度重塑和转置
        q, k, v = map(lambda t: reshape_dim(t, -1, (-1, dim_head)).transpose(-2, -3), (q, k, v))

        # 使用线性注意力函数计算输出
        out = linear_attn(q, k, v)

        # 将输出进行维度转置和重塑，使其形状与输入 x 相同
        out = out.transpose(2, 3).reshape_as(x)
        # 使用权重矩阵 w_out 对输出进行线性变换
        out = torch.einsum('bnd,de->bne', out, w_out)
        # 返回处理后的输出
        return out
# 定义一个内存注意力网络的类，继承自 nn.Module
class MemoryAttentionNetwork(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(self, dim, num_memory_depth, mem_len, lmem_len, heads = 4, num_attn_steps = 2, num_mem_kv = 4, mem_write_iters = 2):
        super().__init__()
        # 初始化内存深度、内存长度和长期内存长度等属性
        self.num_memory_depth = num_memory_depth
        self.mem_len = mem_len
        self.lmem_len = lmem_len

        self.dim = dim
        dim_head = dim // heads
        self.dim_head = dim_head

        # 初始化深度嵌入、初始长期内存和长期内存位置嵌入等参数
        self.depth_emb = init_parameter((num_memory_depth, 1, 1, 1), dim)
        self.init_lmem = init_parameter((1, 1, dim), dim)
        self.lmem_pos_emb = init_parameter((1, lmem_len, dim), dim)

        self.mem_kv = init_parameter((1, num_mem_kv, dim), dim)

        # 初始化自注意力层和门控循环单元
        self.attn = LinearSelfAttention(dim, num_memory_depth, heads = heads)
        self.gate = nBRC(dim, dim)
        self.mem_write_iters = mem_write_iters

    # 前向传播函数，接受多个参数
    def forward(self, lmem, smem, hiddens, detach_lmem = False):
        batch, dim, dim_head, mem_depth, lmem_len = lmem.shape[0], self.dim, self.dim_head, self.num_memory_depth, self.lmem_len

        # 适当地分离隐藏状态，并在给定截断信号时分离长期内存
        hiddens = hiddens.detach()

        if detach_lmem:
            lmem = lmem.detach()

        # 如果没有提供长期内存状态，则初始化长期内存状态
        if lmem is None or lmem.shape[1] == 0:
            lmem = self.init_lmem.clone().expand(batch, lmem_len, -1)

        # 使用高效的线性注意力更新长期内存
        next_lmem = lmem + self.lmem_pos_emb

        hiddens_and_smem = torch.cat((smem, hiddens), dim=-2)
        all_hiddens = (hiddens_and_smem + self.depth_emb).transpose(0, 1).reshape(batch, -1, dim)
        all_hiddens = torch.cat((all_hiddens, self.mem_kv.expand(batch, -1, -1)), dim=1)

        # 迭代执行内存写入操作
        for _ in range(self.mem_write_iters):
            attn_out = self.attn(next_lmem, hiddens = all_hiddens)
            next_lmem = self.gate(attn_out, next_lmem)

        # FIFO队列短期内存
        _, next_mem = queue_fifo(smem, hiddens, length = self.mem_len, dim = 2)

        # 返回更新后的短期内存和长期内存
        return Memory(short = next_mem.detach(), long = next_lmem)

# transformer

class MemoryTransformerXL(nn.Module):
    # 初始化模型参数
    def __init__(self, num_tokens, dim, seq_len, depth, emb_dim = None, memory_layers = None, mem_len = None, lmem_len = None, heads = 8, gru_gated_residual = True, mogrify_gru = False, attn_dropout = 0., ff_glu = False, ff_dropout = 0., attn_layer_dropout = 0., one_kv_head = False, num_mem_kv = 0, mem_write_iters = 2):
        super().__init__()
        # 设置默认的嵌入维度
        emb_dim = default(emb_dim, dim)
        # 设置默认的短期记忆长度
        mem_len = default(mem_len, seq_len)
        # 设置默认的长期记忆长度
        lmem_len = default(lmem_len, mem_len)

        # 设置默认的记忆层
        memory_layers = default(memory_layers, list(range(1, depth + 1)))

        # 检查所有指定的记忆层是否有效
        assert all([layer > 0 and layer <= depth for layer in memory_layers]), 'one of the indicated memory layers is invalid'

        # 初始化模型参数
        self.mem_len = mem_len
        self.seq_len = seq_len

        self.depth = depth
        self.memory_layers = list(memory_layers)

        # 创建 token 的嵌入层
        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        # 将嵌入维度转换为模型维度
        self.to_model_dim = nn.Identity() if emb_dim == dim else nn.Linear(emb_dim, dim)

        seq_and_mem_len = seq_len + mem_len + lmem_len
        # 创建位置编码参数
        self.pos_emb = nn.Parameter(torch.zeros(heads, seq_and_mem_len, dim // heads))
        
        # 创建输出层
        self.to_logits = nn.Sequential(
            nn.Identity() if emb_dim == dim else nn.Linear(dim, emb_dim),
            nn.Linear(emb_dim, num_tokens)
        )

        # 根据是否使用 GRU 门控残差来选择包装器
        wrapper = partial(GRUGating, dim, mogrify = mogrify_gru) if gru_gated_residual else Residual

        # 创建注意力层和前馈层
        self.attn_layers = nn.ModuleList([wrapper(PreNorm(dim, SelfAttention(dim, seq_len, mem_len, lmem_len, heads, dropout = attn_layer_dropout, attn_dropout = attn_dropout, one_kv_head = one_kv_head, num_mem_kv = num_mem_kv))) for _ in range(depth)])
        self.ff_layers = nn.ModuleList([wrapper(PreNorm(dim, FeedForward(dim, dropout = ff_dropout, glu = ff_glu))) for _ in range(depth)])

        # 创建记忆网络
        self.memory_network = MemoryAttentionNetwork(dim, len(self.memory_layers), mem_len, lmem_len, num_mem_kv = num_mem_kv, mem_write_iters = mem_write_iters)

    # 前向传播函数
    def forward(self, x, memories = None, mask = None, detach_lmem = False):
        # 对输入进行 token 嵌入
        x = self.token_emb(x)
        x = self.to_model_dim(x)
        b, t, d = x.shape

        # 检查输入序列长度是否超过最大序列长度
        assert t <= self.seq_len, f'input contains a sequence length {t} that is greater than the designated maximum sequence length {self.seq_len}'

        memories = default(memories, (None, None))
        mem, lmem = memories

        num_memory_layers = len(self.memory_layers)

        # 初始化记忆
        mem = default(mem, lambda: torch.empty(num_memory_layers, b, 0, d, **to(x)))
        lmem = default(lmem, lambda: torch.empty(b, 0, d, **to(x)))

        mem_len, lmem_len = map(lambda t: t.shape[2], (mem, lmem))
        total_len = mem_len + lmem_len + self.seq_len

        # 获取位置编码
        pos_emb = self.pos_emb[:, (self.seq_len - t):total_len]

        mem_iter = iterate_tensor(mem)

        hiddens = []

        # 遍历注意力层和前馈层
        for ind, (attn, ff) in enumerate(zip(self.attn_layers, self.ff_layers)):
            layer_num = ind + 1
            use_memory = layer_num in self.memory_layers
            memories = (next(mem_iter), lmem) if use_memory else None

            if use_memory:
                hiddens.append(x)

            x = attn(x, memories = memories, input_mask = mask, pos_emb = pos_emb)
            x = ff(x)

        hiddens = torch.stack(hiddens)
        out = self.to_logits(x)

        # 计算下一个记忆状态
        # 只有在输入序列长度达到最大时才将隐藏状态推送到短期记忆中

        if t < self.mem_len:
            return out, Memory(short = mem, long = lmem)

        next_memory = self.memory_network(lmem, mem, hiddens, detach_lmem = detach_lmem)
        return out, next_memory
```