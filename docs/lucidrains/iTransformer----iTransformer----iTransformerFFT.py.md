# `.\lucidrains\iTransformer\iTransformer\iTransformerFFT.py`

```py
# 导入 torch 库
import torch
# 从 torch.fft 模块中导入 fft 函数
from torch.fft import fft
# 从 torch 模块中导入 nn、einsum、Tensor
from torch import nn, einsum, Tensor
# 从 torch.nn 模块中导入 Module、ModuleList
from torch.nn import Module, ModuleList
# 从 torch.nn.functional 模块中导入 F
import torch.nn.functional as F

# 从 beartype 库中导入 beartype 函数
from beartype import beartype
# 从 beartype.typing 模块中导入 Optional、Union、Tuple
from beartype.typing import Optional, Union, Tuple

# 从 einops 库中导入 rearrange、reduce、repeat、pack、unpack
from einops import rearrange, reduce, repeat, pack, unpack
# 从 einops.layers.torch 模块中导入 Rearrange
from einops.layers.torch import Rearrange

# 从 iTransformer.attend 模块中导入 Attend 类
from iTransformer.attend import Attend
# 从 iTransformer.revin 模块中导入 RevIN 类

# 定义 helper functions

# 判断变量是否存在
def exists(v):
    return v is not None

# 如果变量存在则返回变量，否则返回默认值
def default(v, d):
    return v if exists(v) else d

# 返回输入的值
def identity(t, *args, **kwargs):
    return t

# 如果输入不是元组，则转换为元组
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
        flash = True
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 3, h = heads)
        )

        self.to_v_gates = nn.Sequential(
            nn.Linear(dim, dim_inner, bias = False),
            nn.SiLU(),
            Rearrange('b n (h d) -> b h n d', h = heads)
        )

        self.attend = Attend(flash = flash, dropout = dropout)

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

# 定义 feedforward 类

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

# 定义主类 iTransformerFFT

class iTransformerFFT(Module):
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
    # 定义模型的初始化方法
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置模型的变量数和回溯长度
        self.num_variates = num_variates
        self.lookback_len = lookback_len

        # 如果存在记忆令牌数量大于0，则使用随机初始化的参数
        self.mem_tokens = nn.Parameter(torch.randn(num_mem_tokens, dim)) if num_mem_tokens > 0 else None

        # 将预测长度转换为元组形式
        pred_length = cast_tuple(pred_length)
        self.pred_length = pred_length

        # 如果使用可逆实例归一化，则初始化RevIN对象
        self.reversible_instance_norm = RevIN(num_variates, affine = reversible_instance_norm_affine) if use_reversible_instance_norm else None

        # 初始化模型的层列表
        self.layers = ModuleList([])
        # 根据深度循环添加多个层
        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = flash_attn),
                nn.LayerNorm(dim),
                FeedForward(dim, mult = ff_mult, dropout = ff_dropout),
                nn.LayerNorm(dim)
            ]))

        # 定义MLP输入层
        self.mlp_in = nn.Sequential(
            nn.Linear(lookback_len, dim * num_tokens_per_variate),
            Rearrange('b v (n d) -> b (v n) d', n = num_tokens_per_variate),
            nn.LayerNorm(dim)
        )

        # 定义FFT-MLP输入层
        self.fft_mlp_in = nn.Sequential(
            Rearrange('b v n c -> b v (n c)'),
            nn.Linear(lookback_len * 2, dim * num_tokens_per_variate),
            Rearrange('b v (n d) -> b (v n) d', n = num_tokens_per_variate),
            nn.LayerNorm(dim)   
        )

        # 初始化预测头列表
        self.pred_heads = ModuleList([])

        # 针对每个预测长度，添加一个预测头
        for one_pred_length in pred_length:
            head = nn.Sequential(
                Rearrange('b (v n) d -> b v (n d)', n = num_tokens_per_variate),
                nn.Linear(dim * num_tokens_per_variate, one_pred_length),
                Rearrange('b v n -> b n v')
            )

            self.pred_heads.append(head)

    # 定义模型的前向传播方法
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
        """
        # 检查是否存在记忆令牌
        has_mem = exists(self.mem_tokens)
        # 断言输入张量的形状符合预期
        assert x.shape[1:] == (self.lookback_len, self.num_variates)

        # 论文的关键在于将变量视为注意力中的空间维度
        # 如果成功复制论文，有很多改进的机会

        # 重新排列输入张量的维度，将时间维度放在最后
        x = rearrange(x, 'b n v -> b v n')

        # 对输入张量进行傅立叶变换
        x_fft = fft(x)
        # 将傅立叶变换后的结果转换为实部和虚部
        x_fft = torch.view_as_real(x_fft)

        # 如果存在可逆实例归一化，则对输入张量进行归一化
        if exists(self.reversible_instance_norm):
            x, reverse_fn = self.reversible_instance_norm(x)

        # 将输入张量投影到变量令牌中，对时间和傅立叶域都进行投影
        x = self.mlp_in(x)
        x_fft = self.fft_mlp_in(x_fft)

        # 将傅立叶变换后的结果放在左侧，以便稍后拼接
        x, fft_ps = pack([x_fft, x], 'b * d')

        # 记忆令牌
        if has_mem:
            # 重复记忆令牌以匹配输入张量的批次维度
            m = repeat(self.mem_tokens, 'm d -> b m d', b = x.shape[0])
            x, mem_ps = pack([m, x], 'b * d')

        # 注意力和前馈层
        for attn, attn_post_norm, ff, ff_post_norm in self.layers:
            x = attn(x) + x
            x = attn_post_norm(x)
            x = ff(x) + x
            x = ff_post_norm(x)

        # 拼接出记忆令牌
        if has_mem:
            _, x = unpack(x, mem_ps, 'b * d')

        # 拼接出傅立叶令牌
        x_fft, x = unpack(x, fft_ps, 'b * d')

        # 如果需要，进行可逆实例归一化
        if exists(self.reversible_instance_norm):
            x = reverse_fn(x)

        # 预测多次
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

        # 如果预测列表为空，则返回第一个预测值
        if len(pred_list) == 0:
            return pred_list[0]

        # 将预测结果与预测长度组成字典返回
        pred_dict = dict(zip(self.pred_length, pred_list))
        return pred_dict
```