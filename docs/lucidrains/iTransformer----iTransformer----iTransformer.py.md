# `.\lucidrains\iTransformer\iTransformer\iTransformer.py`

```py
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

# 从 einops 库中导入 rearrange, reduce, repeat, pack, unpack
from einops import rearrange, reduce, repeat, pack, unpack
# 从 einops.layers.torch 库中导入 Rearrange
from einops.layers.torch import Rearrange

# 从 iTransformer.attend 模块中导入 Attend 类
from iTransformer.attend import Attend
# 从 iTransformer.revin 模块中导入 RevIN 类

# 辅助函数

# 判断变量是否存在
def exists(v):
    return v is not None

# 如果变量存在则返回变量，否则返回默认值
def default(v, d):
    return v if exists(v) else d

# 返回输入本身
def identity(t, *args, **kwargs):
    return t

# 将输入转换为元组
def cast_tuple(t):
    return (t,) if not isinstance(t, tuple) else t

# 注意力机制

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        heads = 4,
        dropout = 0.,
        flash = True
    ):
        super().__init__()
        # 缩放因子
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        # 将输入转换为查询、键、值
        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 3, h = heads)
        )

        # 将输入转换为值门控制
        self.to_v_gates = nn.Sequential(
            nn.Linear(dim, dim_inner, bias = False),
            nn.SiLU(),
            Rearrange('b n (h d) -> b h n d', h = heads)
        )

        # 注意力机制
        self.attend = Attend(flash = flash, dropout = dropout)

        # 输出层
        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        q, k, v = self.to_qkv(x)

        out = self.attend(q, k, v)

        out = out * self.to_v_gates(x)
        return self.to_out(out)

# 前馈神经网络

class GEGLU(Module):
    def forward(self, x):
        x, gate = rearrange(x, '... (r d) -> r ... d', r = 2)
        return x * F.gelu(gate)

# 创建前馈神经网络
def FeedForward(dim, mult = 4, dropout = 0.):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_inner, dim)
    )

# 主类

class iTransformer(Module):
    @beartype
    def __init__(
        self,
        *,
        num_variates: int,
        lookback_len: int,
        depth: int,
        dim: int,
        num_tokens_per_variate = 1,
        pred_length: Union[int, Tuple[int, ...]],
        dim_head = 32,
        heads = 4,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        num_mem_tokens = 4,
        use_reversible_instance_norm = False,
        reversible_instance_norm_affine = False,
        flash_attn = True
    ):
        # 初始化函数，设置模型的变量数和回溯长度
        super().__init__()
        self.num_variates = num_variates
        self.lookback_len = lookback_len

        # 初始化内存令牌参数
        self.mem_tokens = nn.Parameter(torch.randn(num_mem_tokens, dim)) if num_mem_tokens > 0 else None

        # 处理预测长度
        pred_length = cast_tuple(pred_length)
        self.pred_length = pred_length

        # 初始化可逆实例归一化层
        self.reversible_instance_norm = RevIN(num_variates, affine = reversible_instance_norm_affine) if use_reversible_instance_norm else None
        self.num_tokens_per_variate = num_tokens_per_variate

        # 初始化模型的层
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = flash_attn),
                nn.LayerNorm(dim),
                FeedForward(dim, mult = ff_mult, dropout = ff_dropout),
                nn.LayerNorm(dim)
            ]))

        # 初始化 MLP 输入层
        self.mlp_in = nn.Sequential(
            nn.Linear(lookback_len, dim * num_tokens_per_variate),
            Rearrange('b v (n d) -> b (v n) d', n = num_tokens_per_variate),
            nn.LayerNorm(dim)
        )

        # 初始化预测头部
        self.pred_heads = ModuleList([])

        for one_pred_length in pred_length:
            head = nn.Sequential(
                Rearrange('b (v n) d -> b v (n d)', n = num_tokens_per_variate),
                nn.Linear(dim * num_tokens_per_variate, one_pred_length),
                Rearrange('b v n -> b n v')
            )

            self.pred_heads.append(head)

    @beartype
    def forward(
        self,
        x: Tensor,
        targets: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ):
        """
        einstein notation

        b - batch
        n - time
        v - variate
        t - num tokens per variate
        """
        t = self.num_tokens_per_variate

        has_mem = exists(self.mem_tokens)
        assert x.shape[1:] == (self.lookback_len, self.num_variates)

        # 将输入数据重新排列
        x = rearrange(x, 'b n v -> b v n')

        if exists(self.reversible_instance_norm):
            x, reverse_fn = self.reversible_instance_norm(x)

        x = self.mlp_in(x)

        # 内存令牌

        if has_mem:
            m = repeat(self.mem_tokens, 'm d -> b m d', b = x.shape[0])
            x, mem_ps = pack([m, x], 'b * d')

        # 注意力和前馈层

        for attn, attn_post_norm, ff, ff_post_norm in self.layers:
            x = attn(x) + x
            x = attn_post_norm(x)
            x = ff(x) + x
            x = ff_post_norm(x)

        # 剪切出内存令牌

        if has_mem:
            _, x = unpack(x, mem_ps, 'b * d')

        # 如果需要可逆实例归一化

        if exists(self.reversible_instance_norm):
            x = rearrange(x, 'b (n t) d -> t b n d', t = t)
            x = reverse_fn(x)
            x = rearrange(x, 't b n d -> b (n t) d', t = t)

        # 多次预测

        pred_list = [fn(x) for fn in self.pred_heads]

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