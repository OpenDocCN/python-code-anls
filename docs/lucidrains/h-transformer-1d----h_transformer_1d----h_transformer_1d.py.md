# `.\lucidrains\h-transformer-1d\h_transformer_1d\h_transformer_1d.py`

```
# 从 math 模块中导入 log2 和 ceil 函数
# 从 functools 模块中导入 wraps 函数
import torch
# 从 torch 模块中导入 nn, einsum, diagonal 和 nn.functional 模块
from torch import nn, einsum, diagonal
import torch.nn.functional as F
# 从 h_transformer_1d.reversible 模块中导入 ReversibleSequence 和 SequentialSequence 类
from h_transformer_1d.reversible import ReversibleSequence, SequentialSequence
# 从 rotary_embedding_torch 模块中导入 apply_rotary_emb 和 RotaryEmbedding 类
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding
# 从 einops 模块中导入 rearrange, reduce, repeat 函数

# helpers

# 定义函数 exists，判断变量是否存在
def exists(val):
    return val is not None

# 定义函数 masked_aggregate，对张量进行聚合操作
def masked_aggregate(tensor, mask = None, dim = -1, average = True):
    if not exists(mask):
        fn = torch.sum if not average else torch.mean
        return fn(tensor, dim = dim)

    diff_len = len(tensor.shape) - len(mask.shape)
    mask = mask[(..., *((None,) * diff_len))]
    tensor = tensor.masked_fill(~mask, 0.)

    total_el = mask.sum(dim = dim)
    agg = tensor.sum(dim = dim)

    if average:
        agg = agg / total_el.clamp(min = 1.)

    agg.masked_fill_(total_el == 0, 0.)
    return agg

# 定义函数 shift，对张量进行平移操作
def shift(t, amount, mask = None):
    if amount == 0:
        return t

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    return F.pad(t, (0, 0, amount, -amount), value = 0.)

# helper classes

# 定义类 PreNorm，实现预层归一化
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# 定义类 FeedForward，实现前馈神经网络
class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        *,
        mult = 4
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

# token shifting

# 定义类 PreShiftTokens，实现令牌平移
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

# hierarchical attention helper functions

# 定义函数 cast_for_op，将张量转换为指定类型进行操作
def cast_for_op(cast_type, fn):
    @wraps(fn)
    def inner(t, *args, **kwargs):
        orig_type = t.dtype
        t = t.type(cast_type)
        out = fn(t, *args, **kwargs)
        out = out.type(orig_type)
        return out
    return inner

# 定义函数 flip_every_two，交换张量中每两个元素的位置
def flip_every_two(t):
    t = rearrange(t, 'b (n r) ... -> b n r ...', r = 2)
    t = torch.flip(t, dims = (2,))                          # so we pay attention to the off-diagonal blocks in the attention matrix
    t = rearrange(t, 'b n r ... -> b (n r) ...')
    return t

# attention

# 定义类 HAttention1D，实现一维注意力机制
class HAttention1D(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads = 8,
        dim_head = 64,
        block_size = 16,
        pos_emb = None,
        eps = 1e-8,
        **kwargs
    ):
        super().__init__()
        self.eps = eps
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.block_size = block_size
        inner_dim = heads * dim_head

        self.pos_emb = pos_emb
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

# causal attention

# 定义类 CausalHAttention1D，实现一维因果注意力机制
class CausalHAttention1D(nn.Module):
    def __init__(
        self,
        dim,
        *,
        max_seq_len,
        heads = 8,
        dim_head = 64,
        block_size = 16,
        eps = 1e-8,
        pos_emb = None
        ):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化注意力机制的参数
        self.eps = eps
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.block_size = block_size
        inner_dim = heads * dim_head

        # 设置位置编码
        self.pos_emb = pos_emb

        # 线性变换，将输入维度转换为内部维度的三倍，用于计算查询、键、值
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        # 线性变换，将内部维度转换为输出维度
        self.to_out = nn.Linear(inner_dim, dim)

        # 推导出掩码

        # 计算级别数量
        num_levels = int(log2(max_seq_len // block_size)) - 1
        root_seq = torch.arange(max_seq_len)
        seqs = [root_seq]
        seq = root_seq

        # 生成掩码序列
        for ind in range(num_levels):
            seq = rearrange(seq, '(n r) -> n r', r = 2)
            seq = seq.max(dim = -1).values
            expanded_mask_seq = repeat(seq, 'n -> (n r)', r = (2 ** (ind + 1)))
            seqs.append(expanded_mask_seq)

        # 将生成的掩码序列堆叠起来
        seq_keys = torch.stack(seqs, dim = 0)
        # 创建掩码，用于屏蔽无效位置
        mask = seq_keys > rearrange(root_seq, 'n -> () n')
        # 将掩码作为缓冲区注册到模型中
        self.register_buffer('mask', mask)
# 主类定义

class HTransformer1D(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,          # 标记的数量
        dim,                 # 向量维度
        depth,               # 深度
        max_seq_len,         # 最大序列长度
        causal = False,      # 是否因果
        heads = 8,           # 多头注意力的头数
        dim_head = 64,       # 每个头的维度
        ff_mult = 4,         # FeedForward 层的倍数
        block_size = 128,    # 块的大小，即 Nr
        pos_emb = None,      # 位置编码
        reversible = False,  # 是否可逆
        shift_tokens = False # 是否移动标记
    ):
        super().__init__()
        assert (max_seq_len % block_size) == 0, 'maximum sequence length must be divisible by the block size'
        num_blocks = max_seq_len // block_size
        assert log2(max_seq_len // block_size).is_integer(), f'number of blocks {num_blocks} must be a power of 2'

        self.token_emb = nn.Embedding(num_tokens, dim)  # 标记嵌入层
        self.pos_emb = RotaryEmbedding(dim = dim_head)   # 位置编码
        self.max_seq_len = max_seq_len

        layers = nn.ModuleList([])  # 模块列表

        attn_class = CausalHAttention1D if causal else HAttention1D  # 根据是否因果选择不同的注意力类
        attn_kwargs = dict(max_seq_len = max_seq_len) if causal else dict()  # 如果是因果，传入最大序列长度参数

        shift_token_ranges = (0, 1) if shift_tokens else (-1, 0, 1)  # 如果移动标记，设置移动范围

        for ind in range(depth):
            attn = attn_class(dim, dim_head = dim_head, heads = heads, block_size = block_size, pos_emb = self.pos_emb, **attn_kwargs)  # 创建注意力层
            ff = FeedForward(dim, mult = ff_mult)  # 创建 FeedForward 层

            if shift_tokens:
                attn, ff = map(lambda t: PreShiftTokens(shift_token_ranges, t), (attn, ff))  # 如果移动标记，对注意力和 FeedForward 层进行预移动标记处理

            attn, ff = map(lambda t: PreNorm(dim, t), (attn, ff))  # 对注意力和 FeedForward 层进行预归一化处理
            layers.append(nn.ModuleList([attn ,ff]))  # 将注意力和 FeedForward 层添加到模块列表中

        execute_type = ReversibleSequence if reversible else SequentialSequence  # 根据是否可逆选择不同的执行类型
        route_attn = ((True, False),) * depth  # 设置注意力路由
        attn_route_map = {'mask': route_attn}  # 设置注意力路由映射

        self.layers = execute_type(layers, args_route = {**attn_route_map})  # 创建执行类型的层

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),  # 归��化层
            nn.Linear(dim, num_tokens)  # 线性层，输出标记数量
        )

    def forward(self, x, mask = None):
        b, n, device = *x.shape, x.device  # 获取输入张量的形状和设备信息
        assert n <= self.max_seq_len, 'sequence length must be less than the maximum sequence length'  # 断言序列长度必须小于等于最大序列长度
        x = self.token_emb(x)  # 标记嵌入
        x = self.layers(x, mask = mask)  # 执行层
        return self.to_logits(x)  # 输出预测结果
```