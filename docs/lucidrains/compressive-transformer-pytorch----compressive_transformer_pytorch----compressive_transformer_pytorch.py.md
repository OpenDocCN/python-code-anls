# `.\lucidrains\compressive-transformer-pytorch\compressive_transformer_pytorch\compressive_transformer_pytorch.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 torch.nn 模块中导入 F 函数
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

# 定义 Memory 命名元组
Memory = namedtuple('Memory', ['mem', 'compressed_mem'])

# 辅助函数

# 定义 to 函数，返回包含数据类型和设备信息的字典
def to(t):
    return {'dtype': t.dtype, 'device': t.device}

# 定义 cast_tuple 函数，将元素转换为元组
def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)

# 定义 default 函数，如果 x 不为 None，则返回 x，否则返回 val 或 val() 的结果
def default(x, val):
    if x is not None:
        return x
    return val if not isfunction(val) else val()

# 定义 max_neg_value 函数，返回给定张量的最大负值
def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

# 定义 reshape_dim 函数，根据给定维度和分割维度对张量进行重塑
def reshape_dim(t, dim, split_dims):
    shape = list(t.shape)
    num_dims = len(shape)
    dim = (dim + num_dims) % num_dims
    shape[dim:dim+1] = split_dims
    return t.reshape(shape)

# 定义 split_at_index 函数，根据给定维度和索引将张量分割成两部分
def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

# 定义 queue_fifo 函数，实现先进先出队列操作
def queue_fifo(*args, length, dim=-2):
    queue = torch.cat(args, dim=dim)
    if length > 0:
        return split_at_index(dim, -length, queue)

    device = queue.device
    shape = list(queue.shape)
    shape[dim] = 0
    return queue, torch.empty(shape, device=device)

# 定义 shift 函数，实现张量的位移操作
def shift(x):
    *_, i, j = x.shape
    zero_pad = torch.zeros((*_, i, i), **to(x))
    x = torch.cat([x, zero_pad], -1)
    l = i + j - 1
    x = x.view(*_, -1)
    zero_pad = torch.zeros(*_, -x.size(-1) % l, **to(x))
    shifted = torch.cat([x, zero_pad], -1).view(*_, -1, l)
    return shifted[..., :i, i - 1:]

# 定义 iterate_tensor 函数，实现对张量的迭代操作
def iterate_tensor(t):
    length = t.shape[0]
    for ind in range(length):
        yield t[ind]

# full attention 用于计算辅助重构损失

# 定义 full_attn 函数，实现全连接注意力机制
def full_attn(q, k, v, dropout_fn=None):
    *_, dim = q.shape
    dots = torch.einsum('bhid,bhjd->bhij', q, k) * (dim ** -0.5)
    attn = dots.softmax(dim=-1)
    if dropout_fn is not None:
        attn = dropout_fn(attn)
    return torch.einsum('bhij,bhjd->bhid', attn, v)

# 辅助类

# 定义 Residual 类，实现残差连接
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        out = cast_tuple(out)
        ret = (out[0] + x), *out[1:]
        return ret

# 定义 GRUGating 类，实现 GRU 门控机制
class GRUGating(nn.Module):
    def __init__(self, dim, fn, mogrify=False):
        super().__init__()
        self.dim = dim
        self.fn = fn
        self.gru = nn.GRUCell(dim, dim)
        self.mogrify = Mogrifier(dim, factorize_k=dim // 4) if mogrify else None

    def forward(self, x, **kwargs):
        batch, dim = x.shape[0], self.dim
        out = self.fn(x, **kwargs)
        (y, *rest) = cast_tuple(out)

        if self.mogrify is not None:
            y, x = self.mogrify(y, x)

        gated_output = self.gru(
            y.reshape(-1, dim),
            x.reshape(-1, dim)
        )

        gated_output = gated_output.reshape(batch, -1, dim)
        ret = gated_output, *rest
        return ret

# 定义 PreNorm 类，实现预层归一化
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# 定义 ConvCompress 类，实现卷积压缩
class ConvCompress(nn.Module):
    def __init__(self, dim, ratio=4):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, ratio, stride=ratio)

    def forward(self, mem):
        mem = mem.transpose(1, 2)
        compressed_mem = self.conv(mem)
        return compressed_mem.transpose(1, 2)

# feedforward

# 定义 GELU_ 类，实现 GELU 激活函数
class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))

# 如果 nn 模块中存在 GELU 函数，则使用 nn.GELU，否则使用 GELU_ 类
GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

# 定义 FeedForward 类
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
# 定义 SelfAttention 类，继承自 nn.Module
class SelfAttention(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(self, dim, seq_len, mem_len, cmem_len, cmem_ratio = 4, heads = 8, attn_dropout = 0., dropout = 0., reconstruction_attn_dropout = 0.):
        super().__init__()
        # 断言确保维度能够被头数整除
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        # 初始化各个参数
        self.heads = heads
        self.dim_head = dim // heads
        self.seq_len = seq_len
        self.mem_len = mem_len
        self.cmem_len = cmem_len
        self.cmem_ratio = cmem_ratio
        self.scale = self.dim_head ** (-0.5)

        # 创建 ConvCompress 对象，用于压缩记忆
        self.compress_mem_fn = ConvCompress(dim, cmem_ratio)

        # 创建线性层，用于计算查询、键和值
        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_kv = nn.Linear(dim, dim * 2, bias = False)
        self.to_out = nn.Linear(dim, dim)

        # 创建 Dropout 层，用于注意力机制的 dropout 和整体的 dropout
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout = nn.Dropout(dropout)

        # 创建 Dropout 层，用于重构注意力机制的 dropout
        self.reconstruction_attn_dropout = nn.Dropout(reconstruction_attn_dropout)
    # 定义前向传播函数，接受输入 x 和一些可选参数
    def forward(self, x, memories = None, pos_emb = None, input_mask = None, calc_memory = True, **kwargs):
        # 获取输入 x 的形状信息
        b, t, e, h, dim_h = *x.shape, self.heads, self.dim_head

        # 初始化记忆
        memories = default(memories, (None, None))
        mem, cmem = memories

        # 初始化空的记忆
        init_empty_mem = lambda: torch.empty(b, 0, e, **to(x))
        mem = default(mem, init_empty_mem)
        cmem = default(cmem, init_empty_mem)

        # 获取记忆的长度
        mem_len = mem.shape[1]
        cmem_len = cmem.shape[1]

        # 计算查询向量 q
        q = self.to_q(x)

        # 将记忆和输入 x 连接起来，获取键值对 k, v
        kv_input = torch.cat((cmem, mem, x), dim=1)
        kv_len = kv_input.shape[1]
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        # 合并多头注意力的维度
        merge_heads = lambda x: reshape_dim(x, -1, (-1, dim_h)).transpose(1, 2)
        q, k, v = map(merge_heads, (q, k, v))

        # 扩展键值对 k, v 的维度
        k, v = map(lambda x: x.expand(-1, h, -1, -1), (k, v))

        # 计算点积注意力
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = max_neg_value(dots)

        # 添加位置编码
        if pos_emb is not None:
            pos_emb = pos_emb[:, -kv_len:].type(q.dtype)
            pos_dots = torch.einsum('bhid,hjd->bhij', q, pos_emb) * self.scale
            pos_dots = shift(pos_dots)
            dots = dots + pos_dots

        # 添加输入掩码
        if input_mask is not None:
            mask = input_mask[:, None, :, None] * input_mask[:, None, None, :]
            mask = F.pad(mask, (mem_len + cmem_len, 0), value = True)
            dots.masked_fill_(~mask, mask_value)

        # 创建掩码矩阵
        total_mem_len = mem_len + cmem_len
        mask = torch.ones(t, t + total_mem_len, **to(x)).triu_(diagonal = 1 + total_mem_len).bool()
        dots.masked_fill_(mask[None, None, ...], mask_value)

        # 计算注意力权重
        attn = dots.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # 计算输出
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(b, t, -1)
        logits = self.to_out(out)
        logits = self.dropout(logits)

        # 复制记忆
        new_mem = mem
        new_cmem = cmem
        aux_loss = torch.zeros(1, requires_grad = True, **to(q))

        # 如果序列长度小于设定值或者不需要计算记忆，则直接返回结果
        if self.seq_len > t or not calc_memory:
            return logits, Memory(new_mem, new_cmem), aux_loss

        # 计算记忆和压缩记忆
        old_mem, new_mem = queue_fifo(mem, x, length = self.mem_len, dim = 1)
        old_mem_padding = old_mem.shape[1] % self.cmem_ratio

        # 对旧记忆进行填充
        if old_mem_padding != 0:
            old_mem = F.pad(old_mem, (0, 0, old_mem_padding, 0), value = 0.)

        # 如果旧记忆为空或者压缩记忆长度小于等于0，则直接返回结果
        if old_mem.shape[1] == 0 or self.cmem_len <= 0:
            return logits, Memory(new_mem, new_cmem), aux_loss

        # 压缩记忆
        compressed_mem = self.compress_mem_fn(old_mem.detach())
        old_cmem, new_cmem = split_at_index(1, -self.cmem_len, torch.cat((cmem, compressed_mem), dim=1))

        # 如果不处于训练状态，则直接返回结果
        if not self.training:
            return logits, Memory(new_mem, new_cmem), aux_loss

        # 计算训练时的压缩记忆辅助损失
        self.to_kv.weight.detach_()

        cmem_k, cmem_v = self.to_kv(compressed_mem).chunk(2, dim=-1)
        cmem_k, cmem_v = map(merge_heads, (cmem_k, cmem_v))
        cmem_k, cmem_v = map(lambda x: x.expand(-1, h, -1, -1), (cmem_k, cmem_v))

        old_mem_range = slice(- min(mem_len, self.mem_len) - self.seq_len, -self.seq_len)
        old_mem_k, old_mem_v = map(lambda x: x[:, :, old_mem_range].clone(), (k, v))

        q, old_mem_k, old_mem_v = map(torch.detach, (q, old_mem_k, old_mem_v))

        attn_fn = partial(full_attn, dropout_fn = self.reconstruction_attn_dropout)

        aux_loss = F.mse_loss(
            attn_fn(q, old_mem_k, old_mem_v),
            attn_fn(q, cmem_k, cmem_v)
        )

        return logits, Memory(new_mem, new_cmem), aux_loss
# 定义一个压缩变换器类，继承自 nn.Module
class CompressiveTransformer(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        num_tokens,  # 标记的数量
        dim,  # 维度
        seq_len,  # 序列长度
        depth,  # 深度
        emb_dim = None,  # 嵌入维度，默认为 None
        memory_layers = None,  # 记忆层，默认为 None
        enhanced_recurrence = True,  # 增强循环，默认为 True
        mem_len = None,  # 记忆长度，默认为 None
        cmem_len = None,  # 压缩记忆长度，默认为 None
        cmem_ratio = 4,  # 压缩记忆比率，默认为 4
        heads = 8,  # 头数，默认为 8
        gru_gated_residual = True,  # GRU 门控残差，默认为 True
        mogrify_gru = False,  # Mogrify GRU，默认为 False
        attn_dropout = 0.,  # 注意力丢弃率，默认为 0
        ff_glu = False,  # FeedForward GLU，默认为 False
        ff_dropout = 0.,  # FeedForward 丢弃率，默认为 0
        attn_layer_dropout = 0.,  # 注意力层丢弃率，默认为 0
        reconstruction_attn_dropout = 0.,  # 重构注意力丢弃率，默认为 0
        reconstruction_loss_weight = 1.  # 重构损失权重，默认为 1
    ):
        super().__init__()  # 调用父类的初始化函数
        emb_dim = default(emb_dim, dim)  # 如果嵌入维度为 None，则使用维度
        mem_len = default(mem_len, seq_len)  # 如果记忆长度为 None，则使用序列长度
        cmem_len = default(cmem_len, mem_len // cmem_ratio)  # 如果压缩记忆长度为 None，则使用记忆长度除以压缩比率
        memory_layers = default(memory_layers, list(range(1, depth + 1)))  # 如果记忆层为 None，则使用范围为 1 到深度的列表

        assert mem_len >= seq_len, 'length of memory should be at least the sequence length'  # 断言记忆长度至少应该等于序列长度
        assert cmem_len >= (mem_len // cmem_ratio), f'length of compressed memory should be at least the memory length divided by the compression ratio {int(mem_len // cmem_ratio)}'  # 断言压缩记忆长度至少应该等于记忆长度除以压缩比率
        assert all([layer > 0 and layer <= depth for layer in memory_layers]), 'one of the indicated memory layers is invalid'  # 断言所有指定的记忆层都在有效范围内

        self.seq_len = seq_len  # 保存序列长度

        self.depth = depth  # 保存深度
        self.memory_layers = list(memory_layers)  # 保存记忆层列表
        self.enhanced_recurrence = enhanced_recurrence  # 保存增强循环标志

        self.token_emb = nn.Embedding(num_tokens, emb_dim)  # 创建标记嵌入层
        self.to_model_dim = nn.Identity() if emb_dim == dim else nn.Linear(emb_dim, dim)  # 如果嵌入维度等于维度，则使用恒等映射，否则使用线性映射

        seq_and_mem_len = seq_len + mem_len + cmem_len  # 计算序列和记忆长度之和
        self.pos_emb = nn.Parameter(torch.zeros(heads, seq_and_mem_len, dim // heads))  # 创建位置嵌入参数

        self.to_logits = nn.Sequential(
            nn.Identity() if emb_dim == dim else nn.Linear(dim, emb_dim),  # 如果嵌入维度等于维度，则使用恒等映射，否则使用线性映射
            nn.Linear(emb_dim, num_tokens)  # 线性映射到标记数量
        )

        wrapper = partial(GRUGating, dim, mogrify = mogrify_gru) if gru_gated_residual else Residual  # 根据 GRU 门控残差标志选择包装器

        self.attn_layers = nn.ModuleList([wrapper(PreNorm(dim, SelfAttention(dim, seq_len, mem_len, cmem_len, cmem_ratio, heads, dropout = attn_layer_dropout, attn_dropout = attn_dropout, reconstruction_attn_dropout = reconstruction_attn_dropout))) for _ in range(depth)])  # 创建注意力层列表
        self.ff_layers = nn.ModuleList([wrapper(PreNorm(dim, FeedForward(dim, dropout = ff_dropout, glu = ff_glu))) for _ in range(depth)])  # 创建前馈层列表

        self.reconstruction_loss_weight = reconstruction_loss_weight  # 保存重构损失权重
    # 前向传播函数，接受输入 x，记忆 memories 和掩码 mask
    def forward(self, x, memories = None, mask = None):
        # 对输入进行 token embedding
        x = self.token_emb(x)
        # 调整输入维度到模型维度
        x = self.to_model_dim(x)
        b, t, d = x.shape

        # 断言输入序列长度不超过指定的最大序列长度
        assert t <= self.seq_len, f'input contains a sequence length {t} that is greater than the designated maximum sequence length {self.seq_len}'

        # 初始化记忆
        memories = default(memories, (None, None))
        mem, cmem = memories

        num_memory_layers = len(self.memory_layers)
        # 初始化空记忆
        init_empty_mem = lambda: torch.empty(num_memory_layers, b, 0, d, **to(x))
        mem = default(mem, init_empty_mem)
        cmem = default(cmem, init_empty_mem)

        total_len = mem.shape[2] + cmem.shape[2] + self.seq_len
        # 获取位置编码
        pos_emb = self.pos_emb[:, (self.seq_len - t):total_len]

        next_mem = []
        next_cmem = []
        aux_loss = torch.tensor(0., requires_grad = True, **to(x))

        # 如果启用增强循环
        if self.enhanced_recurrence:
            mem = torch.roll(mem, -1, 0)
            cmem = torch.roll(cmem, -1, 0)

        # 迭代记忆
        mem_iter, cmem_iter = map(iterate_tensor, (mem, cmem))

        # 遍历注意力层和前馈层
        for ind, (attn, ff) in enumerate(zip(self.attn_layers, self.ff_layers)):
            layer_num = ind + 1

            use_memory = layer_num in self.memory_layers
            memories = (next(mem_iter), next(cmem_iter)) if use_memory else None

            # 执行注意力机制和前馈网络
            x, (mem_out, cmem_out), layer_aux_loss = attn(x, memories = memories, calc_memory = use_memory, input_mask = mask, pos_emb = pos_emb)
            x,  = ff(x)

            aux_loss = aux_loss + layer_aux_loss

            # 如果不使用记忆，则跳过
            if not use_memory:
                continue

            next_mem.append(mem_out)
            next_cmem.append(cmem_out)

        # 获取输出结果
        out = self.to_logits(x)

        # 将下一步记忆和压缩记忆堆叠并分离梯度
        next_mem, next_cmem = map(torch.stack, (next_mem, next_cmem))
        next_mem, next_cmem = map(torch.detach, (next_mem, next_cmem))

        # 计算辅助损失
        aux_loss = aux_loss * self.reconstruction_loss_weight / num_memory_layers
        # 返回输出、记忆和辅助损失
        return out, Memory(mem = next_mem, compressed_mem = next_cmem), aux_loss
```