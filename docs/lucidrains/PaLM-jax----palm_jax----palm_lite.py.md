# `.\lucidrains\PaLM-jax\palm_jax\palm_lite.py`

```
# 从 math 模块中导入 log2 和 floor 函数
# 从 typing 模块中导入 List 和 Tuple 类型
import numpy as onp
# 从 jax 模块中导入 random, jit, nn, lax, numpy 模块，并将 numpy 模块重命名为 np
from jax import random, jit, nn, lax, numpy as np
# 从 jax.numpy 模块中导入 einsum 函数
from jax.numpy import einsum
# 从 equinox 模块中导入 Module, static_field 类
from equinox import Module, static_field
# 从 einops 模块中导入 rearrange, repeat 函数

# 定义 RMSNorm 类，继承自 Module 类
class RMSNorm(Module):
    # 定义类属性 gamma, scale, eps
    gamma: np.ndarray
    scale: float = static_field()
    eps: float = static_field()

    # 初始化方法，接受 dim 和 eps 两个参数
    def __init__(self, dim, eps = 1e-5):
        # 初始化 gamma 为全为 1 的数组
        self.gamma = np.ones((dim,))
        self.eps = eps
        self.scale = dim ** 0.5

    # 定义 __call__ 方法，接受参数 x
    def __call__(self, x):
        # 计算 x 的平方和，并在最后一个维度上保持维度
        sum_of_squares = np.sum(np.square(x), axis = -1, keepdims = True)
        # 计算 sum_of_squares 加上 eps 的平方根的倒数
        inv_norm = lax.rsqrt(sum_of_squares + self.eps)
        # 返回 inv_norm 乘以 x 乘以 gamma 乘以 scale 的结果
        return inv_norm * x * self.gamma * self.scale

# 定义 get_alibi_slopes 函数，接受 heads 参数
def get_alibi_slopes(heads):
    # 定义内部函数 get_slopes_power_of_2，接受 n 参数
    def get_slopes_power_of_2(n):
        # 计算起始值 start
        start = (2 ** (-2 ** -(log2(n) - 3)))
        ratio = start
        # 返回等比数列
        return [start*ratio**i for i in range(n)]

    # 如果 heads 的对数是整数
    if log2(heads).is_integer():
        # 返回 get_slopes_power_of_2(heads) 的结果
        return get_slopes_power_of_2(heads)

    # 计算最接近 heads 的 2 的幂次方
    closest_power_of_2 = 2 ** floor(log2(heads))
    # 返回 get_slopes_power_of_2(closest_power_of_2) 和 get_slopes_power_of_2(2 * closest_power_of_2) 的结果
    return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

# 定义 calc_alibi_bias 函数，接受 seq_len 和 heads 两个参数
def calc_alibi_bias(seq_len, heads):
    # 获取斜率
    slopes = get_alibi_slopes(heads)
    # 重排 slopes 数组的维度
    slopes = rearrange(onp.array(slopes), 'h -> h 1 1')
    # 生成偏置
    bias = rearrange(onp.arange(seq_len), 'j -> 1 1 j')
    return slopes * bias

# 定义 ParallelTransformerBlock 类，继承自 Module 类
class ParallelTransformerBlock(Module):
    # 定义类属性 norm, wi, attn_wo, ff_wo, heads, fused_dims, scale, mask_value
    norm: Module
    wi: np.ndarray
    attn_wo: np.ndarray
    ff_wo: np.ndarray
    heads: int = static_field()
    fused_dims: Tuple[int] = static_field()
    scale: float = static_field()
    mask_value: float = static_field()

    # 初始化方法，接受 dim, dim_head, heads, key, ff_mult, mask_value 参数
    def __init__(
        self,
        dim,
        dim_head,
        heads,
        key,
        ff_mult = 4,
        mask_value = -1e10
    ):
        # 计算注意力内部维度和前馈内部维度
        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        # 初始化 norm 为 RMSNorm 类的实例
        self.norm = RMSNorm(dim)
        self.fused_dims = (attn_inner_dim, dim_head, ff_inner_dim, ff_inner_dim)

        # 初始化 wi, attn_wo, ff_wo 为随机正态分布的数组
        self.wi = random.normal(key, (dim, sum(self.fused_dims)))
        self.attn_wo = random.normal(key, (attn_inner_dim, dim))
        self.ff_wo = random.normal(key, (ff_inner_dim, dim))

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.mask_value = mask_value

    # 定义 __call__ 方法，接受 x 和 attn_bias 两个参数
    def __call__(self, x, *, attn_bias):
        # 获取 x 的倒数第二个维度的大小和分割索引
        n, split_indices = x.shape[-2], onp.cumsum(self.fused_dims[:-1])

        # 对 x 进行归一化
        x = self.norm(x)

        # 融合注意力和前馈的投影

        q, kv, ff, ff_gate = np.split(x @ self.wi, split_indices, axis = -1)

        # 分割出头部

        q = rearrange(q, '... n (h d) -> ... h n d', h = self.heads)

        # 缩放

        q *= self.scale

        # 相似度

        sim = einsum('... h i d, ... j d -> ... h i j', q, kv)

        # 因果掩码

        sim = sim + attn_bias

        # 注意力

        attn = nn.softmax(sim, axis = -1)

        # 聚合值

        out = einsum('... h i j, ... j d -> ... h i d', attn, kv)

        # 合并头部

        out = rearrange(out, '... h n d -> ... n (h d)')

        # 前馈输出

        attn_out = out @ self.attn_wo

        ff_out = (ff * nn.swish(ff_gate)) @ self.ff_wo

        # 合并头部输出

        return attn_out + ff_out

# 主类

class PaLM(Module):
    # 定义类属性 embedding, norm, layers, attn_bias
    embedding: np.ndarray
    norm: Module
    layers: List[List[Module]]
    attn_bias: onp.ndarray = static_field()

    # 初始化方法，接受 num_tokens, dim, dim_head, depth, heads, key, ff_mult, max_seq_len, mask_value 参数
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        dim_head,
        depth,
        heads,
        key,
        ff_mult = 4,
        max_seq_len = 2048,
        mask_value = -1e10
        self.embedding = random.normal(key, (num_tokens, dim)) * 0.02
        # 初始化嵌入矩阵，使用正态分布生成随机值，并乘以0.02

        causal_mask = onp.tril(onp.ones((max_seq_len, max_seq_len)))
        # 创建一个下三角矩阵作为因果掩码
        alibi_bias = calc_alibi_bias(max_seq_len, heads = heads)
        # 计算alibi偏置
        self.attn_bias = np.where(causal_mask, repeat(alibi_bias, 'h 1 j -> h i j', i = max_seq_len), mask_value)
        # 根据因果掩码和alibi偏置生成注意力偏置矩阵

        self.layers = [ParallelTransformerBlock(dim = dim, dim_head = dim_head, heads = heads, key = key, ff_mult = ff_mult) for _ in range(depth)]
        # 创建多个并行Transformer块
        self.norm = RMSNorm(dim)
        # 初始化RMS归一化层

    @jit
    def __call__(self, x):
        # 定义类的调用方法，输入x
        n = x.shape[-1]
        # 获取输入x的最后一个维度大小
        x = self.embedding[x]
        # 使用嵌入矩阵将输入x转换为嵌入向量

        attn_bias = self.attn_bias[..., :n, :n]
        # 获取与输入长度相关的注意力偏置

        for block in self.layers:
            # 遍历每个Transformer块
            x = block(x, attn_bias = attn_bias) + x
            # 对输入x进行Transformer块的处理，并将结果与原始输入相加

        x = self.norm(x)
        # 对处理后的结果进行RMS归一化
        return x @ self.embedding.transpose()
        # 返回结果与嵌入矩阵的转置矩阵的乘积
```