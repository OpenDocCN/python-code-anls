# `.\lucidrains\gateloop-transformer\gateloop_transformer\simplified_gate_loop.py`

```py
# 导入所需模块
from functools import partial
import torch
from torch import nn, Tensor
from torch.nn import Module
from typing import Tuple
from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange
from gateloop_transformer.gateloop_transformer import RMSNorm
from gateloop_transformer.associative_scan import associative_scan

# 检查变量是否存在的函数
def exists(v):
    return v is not None

# 绝对值截断函数，用于处理小于给定阈值的值
def abs_clamp_eps(t, eps = 1e-20):
    sign = torch.sign(t)
    return sign * t.abs().clamp(min = eps)

# 使用 Heinsen 序列进行关联扫描
def heinsen_associative_scan(a, kv, eps = 1e-20):
    log_a = a.clamp(min = eps).log()
    log_kv = abs_clamp_eps(kv, eps = eps).to(dtype = torch.complex64).log()
    a_star = torch.cumsum(log_a, dim = 1)
    log_x0_plus_b_star = torch.logcumsumexp(log_kv - a_star, dim = 1)
    log_x = a_star + log_x0_plus_b_star
    return a_star.exp().real, log_x.exp().real

# 使用 TorchScript 实现的二进制运算函数
@torch.jit.script
def binary_operator(
    a: Tuple[Tensor, Tensor],
    b: Tuple[Tensor, Tensor]
):
    a_i, kv_i = a
    a_j, kv_j = b
    return a_j * a_i, torch.addcmul(kv_j, a_j, kv_i)

# 门循环操作符
def gate_loop_operator(q, kv, a, cache = None, heinsen = False):
    if exists(cache):
        cache_a, cache_kv = cache
        a, a_ps = pack([cache_a, a], 'b * d')
        kv, kv_ps = pack([cache_kv, kv], 'b * d')

    if heinsen:
        a, kv = heinsen_associative_scan(a, kv)
    else:
        a, kv = associative_scan(binary_operator, (a, kv))

    if exists(cache):
        _, a = unpack(a, a_ps, 'b * d')
        _, kv = unpack(kv, kv_ps, 'b * d')

    return q * kv, (a[:, -1], kv[:, -1])

# 使用 JAX 实现的门循环操作符
def get_jax_gate_loop_operator():
    try:
        from jax import jit, numpy as jnp
        from jax.lax import associative_scan
        from jax2torch import jax2torch
    except ImportError as e:
        print(f'jax and jax2torch must be installed - `pip install jax2torch`')

    @jit
    def jax_gate_loop_operator(q, kv, a, cache = None):
        def binary_operator(e_i, e_j):
            a_i, kv_i = e_i
            a_j, kv_j = e_j
            return a_j * a_i, a_j * kv_i + kv_j

        if exists(cache):
            cache_a, cache_kv = cache
            a, a_ps = pack([cache_a, a], 'b * d')
            kv, kv_ps = pack([cache_kv, kv], 'b * d')

        _, y = associative_scan(binary_operator, (a, kv), axis = 1)

        if exists(cache):
            _, a = unpack(a, a_ps, 'b * d')
            _, kv = unpack(kv, kv_ps, 'b * d')

        return q * y, (a[:, -1], kv[:, -1])

    return jax2torch(jax_gate_loop_operator)

# 简单的门循环层
class SimpleGateLoopLayer(Module):
    """
    简化的门循环层，用于补充注意力机制
    参考 https://github.com/lucidrains/mega-pytorch
    """

    def __init__(
        self,
        dim,
        prenorm = True,
        use_heinsen = False,
        use_jax_associative_scan = False,
        post_ln = False,
        reverse = False
    ):
        # 调用父类的构造函数
        super().__init__()
        # 断言确保 use_heinsen 和 use_jax_associative_scan 中至多只有一个为真
        assert (int(use_heinsen) + int(use_jax_associative_scan)) <= 1

        # 如果 prenorm 为真，则使用 RMSNorm 进行归一化，否则使用 nn.Identity()
        self.norm = RMSNorm(dim) if prenorm else nn.Identity()

        # 初始化维度
        self.dim = dim

        # 将输入映射到 q, k, v，并进行线性变换
        self.to_qkva = nn.Sequential(
            nn.Linear(dim, dim * 3, bias = False),
            Rearrange('b n (qkva d) -> qkva (b d) n 1', qkva = 3)
        )

        # 设置是否使用 Heinsen 或 JAX 的关联扫描
        self.use_heinsen = use_heinsen
        self.use_jax = use_jax_associative_scan

        # 根据使用的扫描方式选择相应的 gate_loop_fn
        if use_jax_associative_scan:
            self.gate_loop_fn = get_jax_gate_loop_operator()
        elif use_heinsen:
            self.gate_loop_fn = partial(gate_loop_operator, heinsen = True)
        else:
            self.gate_loop_fn = gate_loop_operator

        # 如果 post_ln 为真，则使用 nn.LayerNorm(dim) 进行归一化，否则使用 nn.Identity()
        self.maybe_post_ln = nn.LayerNorm(dim) if post_ln else nn.Identity()
        # 将输出进行头部分割
        self.split_heads = Rearrange('(b d) n 1 -> b n d', d = dim)

        # 设置是否反转序列
        self.reverse = reverse

    # 前向传播函数
    def forward(
        self,
        x,
        cache = None,
        return_cache = False
    ):
        # 如果需要反转序列，则对输入进行反转
        if self.reverse:
            x = torch.flip(x, dims = (-2,))

        # 对输入进行归一化
        x = self.norm(x)

        # 将输入映射到 q, k, v
        q, kv, a = self.to_qkva(x)

        # 使用 gate_loop_fn 进行计算
        out, cache = self.gate_loop_fn(q, kv, a.sigmoid(), cache = cache)

        # 将输出进行头部分割
        out = self.split_heads(out)
        # 对输出进行归一化
        out = self.maybe_post_ln(out)

        # 如果需要反转序列，则对输出进行反转
        if self.reverse:
            out = torch.flip(out, dims = (-2,))

        # 如果不需要返回 cache，则直接返回输出
        if not return_cache:
            return out

        # 断言确保只有在非反转序列时才能缓存
        assert not self.reverse, 'caching only works with non-reversed seq'

        # 返回输出和 cache
        return out, cache
```