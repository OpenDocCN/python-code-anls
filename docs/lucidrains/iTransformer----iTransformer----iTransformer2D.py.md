# `.\lucidrains\iTransformer\iTransformer\iTransformer2D.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn, einsum, Tensor
from torch import nn, einsum, Tensor
# 从 torch.nn 库中导入 Module, ModuleList
from torch.nn import Module, ModuleList
# 从 torch.nn.functional 库中导入 F
import torch.nn.functional as F

# 从 beartype 库中导入 beartype
from beartype import beartype
# 从 beartype.typing 库中导入 Optional, Union, Tuple
from beartype.typing import Optional, Union, Tuple

# 导入 einops 库中的 rearrange, reduce, repeat, pack, unpack
from einops import rearrange, reduce, repeat, pack, unpack
# 从 einops.layers.torch 库中导入 Rearrange
from einops.layers.torch import Rearrange

# 从 iTransformer.attend 模块中导入 Attend 类
from iTransformer.attend import Attend
# 从 iTransformer.revin 模块中导入 RevIN 类
from iTransformer.revin import RevIN

# 从 gateloop_transformer 模块中导入 SimpleGateLoopLayer 类
from gateloop_transformer import SimpleGateLoopLayer
# 从 rotary_embedding_torch 模块中导入 RotaryEmbedding 类

# 定义 helper functions

# 判断变量是否存在
def exists(v):
    return v is not None

# 如果变量存在则返回变量，否则返回默认值
def default(v, d):
    return v if exists(v) else d

# 将输入的张量 t 按照指定的 pattern 进行打包
def pack_one(t, pattern):
    return pack([t], pattern)

# 将输入的张量 t 按照指定的 pattern 进行解包
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# 返回输入的张量 t
def identity(t, *args, **kwargs):
    return t

# 判断 num 是否能被 den 整除
def divisible_by(num, den):
    return (num % den) == 0

# 将输入的变量 t 转换为元组形式
def cast_tuple(t):
    return (t,) if not isinstance(t, tuple) else t

# 定义 attention 类

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        heads = 4,
        dropout = 0.,
        causal = False,
        flash = True,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.rotary_emb = rotary_emb

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 3, h = heads)
        )

        self.to_v_gates = nn.Sequential(
            nn.Linear(dim, dim_inner, bias = False),
            nn.SiLU(),
            Rearrange('b n (h d) -> b h n d', h = heads)
        )

        self.attend = Attend(flash = flash, dropout = dropout, causal = causal)

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )

    @beartype
    def forward(self, x):
        q, k, v = self.to_qkv(x)

        if exists(self.rotary_emb):
            q, k = map(self.rotary_emb.rotate_queries_or_keys, (q, k))

        out = self.attend(q, k, v)

        out = out * self.to_v_gates(x)
        return self.to_out(out)

# 定义 GEGLU 类

class GEGLU(Module):
    def forward(self, x):
        x, gate = rearrange(x, '... (r d) -> r ... d', r = 2)
        return x * F.gelu(gate)

# 定义 FeedForward 函数

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_inner, dim)
    )

# 定义 transformer block 类

class TransformerBlock(Module):
    def __init__(
        self,
        *,
        dim,
        causal = False,
        dim_head = 32,
        heads = 8,
        ff_mult = 4,
        flash_attn = True,
        attn_dropout = 0.,
        ff_dropout = 0.,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.rotary_emb = rotary_emb

        self.attn = Attention(flash = flash_attn, rotary_emb = rotary_emb, causal = causal, dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        self.ff = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, x, rotary_emb: Optional[RotaryEmbedding] = None):

        x = self.attn(x) + x
        x = self.attn_norm(x)

        x = self.ff(x) + x
        x = self.ff_norm(x)

        return x

# 定义主类 iTransformer2D

class iTransformer2D(Module):
    @beartype
    # 初始化函数，设置模型的各种参数
    def __init__(
        self,
        *,
        num_variates: int,  # 变量数量
        lookback_len: int,  # 回溯长度
        num_time_tokens: int,  # 时间标记数量
        depth: int,  # 模型深度
        dim: int,  # 维度
        pred_length: Union[int, Tuple[int, ...]],  # 预测长度
        dim_head = 32,  # 头部维度
        heads = 4,  # 头部数量
        attn_dropout = 0.,  # 注意力机制的dropout
        ff_mult = 4,  # FeedForward 层的倍数
        ff_dropout = 0.,  # FeedForward 层的dropout
        num_mem_tokens = 4,  # 记忆标记数量
        use_reversible_instance_norm = False,  # 是否使用可逆实例归一化
        reversible_instance_norm_affine = True,  # 可逆实例归一化的可学习参数
        flash_attn = True  # 是否使用 Flash Attention
    ):
        super().__init__()
        assert divisible_by(lookback_len, num_time_tokens)  # 断言回溯长度可以被时间标记数量整除
        assert num_time_tokens >= 2  # 断言时间标记数量至少为2

        self.num_variates = num_variates  # 设置变量数量
        self.lookback_len = lookback_len  # 设置回溯长度
        self.num_time_tokens = num_time_tokens  # 设置时间标记数量

        self.mem_tokens = nn.Parameter(torch.randn(num_mem_tokens, dim)) if num_mem_tokens > 0 else None  # 设置记忆标记

        pred_length = cast_tuple(pred_length)  # 将预测长度转换为元组
        self.pred_length = pred_length  # 设置预测长度

        self.reversible_instance_norm = RevIN(num_variates, affine = reversible_instance_norm_affine) if use_reversible_instance_norm else None  # 设置可逆实例归一化

        rotary_emb = RotaryEmbedding(dim_head)  # 创建旋转嵌入对象

        self.layers = ModuleList([])  # 创建模型层列表

        block_kwargs = dict(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            ff_mult = ff_mult,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            flash_attn = flash_attn
        )

        # 循环创建模型层
        for _ in range(depth):
            self.layers.append(ModuleList([
                SimpleGateLoopLayer(dim = dim),
                TransformerBlock(causal = True, rotary_emb = rotary_emb, **block_kwargs),
                TransformerBlock(causal = False, **block_kwargs)
            ]))

        # 创建变量标记转换层
        self.to_variate_token = nn.Sequential(
            nn.Linear(lookback_len, dim),
            nn.LayerNorm(dim)
        )

        time_kernel_size = lookback_len // num_time_tokens  # 计算时间卷积核大小

        # 创建时间标记转换层
        self.to_time_tokens = nn.Sequential(
            Rearrange('b v n -> (b v) 1 n'),
            nn.ConstantPad1d((time_kernel_size, 0), value = 0.),
            nn.Conv1d(1, dim, time_kernel_size * 2),
            Rearrange('(b v) d t -> b v t d', v = num_variates),
            nn.LayerNorm(dim)
        )

        self.pred_heads = ModuleList([])  # 创建预测头列表

        # 循环创建预测头
        for one_pred_length in pred_length:
            head = nn.Sequential(
                nn.Linear(dim, one_pred_length),
                Rearrange('b v n -> b n v')
            )

            self.pred_heads.append(head)

    @beartype
    # 前向传播函数
    def forward(
        self,
        x: Tensor,  # 输入张量
        targets: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None  # 目标张量
    ):
        """
        einstein notation

        b - batch
        n - time
        v - variate
        t - number of time tokens
        """

        # 检查是否存在记忆令牌
        has_mem = exists(self.mem_tokens)
        # 断言输入张量的形状符合预期
        assert x.shape[1:] == (self.lookback_len, self.num_variates)

        # 将输入张量重新排列，将时间维度放在最后
        x = rearrange(x, 'b n v -> b v n')

        # 如果存在可逆实例归一化，则对输入张量进行处理
        if exists(self.reversible_instance_norm):
            x, reverse_fn = self.reversible_instance_norm(x)

        # 推导每个变量的时间令牌 't'

        t = self.to_time_tokens(x)

        # 'v' 将是变量池令牌，与 iTransformer 中的每个变量令牌相同

        v = self.to_variate_token(x)

        # 将时间和变量令牌组合成二维特征图，包含变量和时间

        x, variate_pool_token_ps = pack((t, v), 'b v * d')

        # 记忆令牌

        if has_mem:
            m = repeat(self.mem_tokens, 'm d -> b m t d', b = x.shape[0], t = x.shape[-2])
            x, mem_ps = pack([m, x], 'b * t d')

        # 注意力和前馈层

        for gateloop_block, time_attn_block, variate_attn_block in self.layers:
            x, ps = pack_one(x, '* t d')

            # gateloop block
            x = gateloop_block(x) + x

            # 每个变量的时间上的因果关注
            x = time_attn_block(x)

            x = unpack_one(x, ps, '* t d')

            x = rearrange(x, 'b v t d -> b t v d')
            x, ps = pack_one(x, '* v d')

            # 全局变量关注（如反向 Transformer 论文中）
            x = variate_attn_block(x)

            x = unpack_one(x, ps, '* v d')
            x = rearrange(x, 'b t v d -> b v t d')

        # 剥离记忆令牌

        if has_mem:
            _, x = unpack(x, mem_ps, 'b * t d')

        # 获取原始的变量池令牌

        _, v = unpack(x, variate_pool_token_ps, 'b v * d')

        # 如果需要，进行可逆实例归一化

        if exists(self.reversible_instance_norm):
            v = reverse_fn(v)

        # 预测多个时间步

        pred_list = [fn(v) for fn in self.pred_heads]

        # 如果传入了目标值，则计算损失

        if exists(targets):
            targets = cast_tuple(targets)
            assert len(targets) == len(pred_list)

            assert self.training
            mse_loss = 0.
            for target, pred in zip(targets, pred_list):
                assert target.shape == pred.shape

                mse_loss = mse_loss + F.mse_loss(target, pred)

            return mse_loss

        if len(pred_list) == 0:
            return pred_list[0]

        pred_dict = dict(zip(self.pred_length, pred_list))
        return pred_dict
```