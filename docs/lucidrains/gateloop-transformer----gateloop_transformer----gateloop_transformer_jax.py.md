# `.\lucidrains\gateloop-transformer\gateloop_transformer\gateloop_transformer_jax.py`

```py
# 导入必要的模块和函数
from typing import List, Tuple, Callable
from jax import random, jit, nn, lax, numpy as np
from jax.lax import associative_scan
from equinox import Module, static_field

# linear

# 定义线性层模块
class Linear(Module):
    weight: np.ndarray
    bias: np.ndarray

    def __init__(self, dim_in, dim_out, *, key):
        # 使用随机数生成权重和偏置
        weight_key, bias_key = random.split(key)
        self.weight = random.normal(weight_key, (dim_in, dim_out))
        self.bias = random.normal(bias_key, (dim_out,))

    def __call__(self, x, *, key = None):
        # 计算线性变换
        return x @ self.weight + self.bias

# rmsnorm

# 定义 RMSNorm 模块
class RMSNorm(Module):
    scale: float = static_field()
    eps: float = static_field()
    gamma: np.ndarray

    def __init__(self, dim, eps = 1e-5):
        # 初始化参数
        self.eps = eps
        self.scale = dim ** 0.5
        self.gamma = np.ones((dim,))

    def __call__(self, x):
        # 计算 RMSNorm
        sum_of_squares = np.sum(np.square(x), axis = -1, keepdims = True)
        inv_norm = lax.rsqrt(sum_of_squares + self.eps)
        return inv_norm * x * self.gamma * self.scale

# gate loop layer

# 定义门循环操作符
def gate_loop_operator(k, v, q, a):
    kv = k * v + 0.j

    def binary_operator(e_i, e_j):
        a_i, kv_i = e_i
        a_j, kv_j = e_j
        return a_j * a_i, a_j * kv_i + kv_j

    # 使用关联扫描计算门循环
    _, y = associative_scan(binary_operator, (a, kv), axis = 1)

    return q * np.real(y)

# 定义门循环模块
class GateLoop(Module):
    norm: RMSNorm
    wq: np.ndarray
    wk: np.ndarray
    wv: np.ndarray
    wa: np.ndarray
    wg: np.ndarray
    wo: np.ndarray

    def __init__(
        self,
        dim,
        key
    ):
        """
        q - query
        k - key
        v - value
        a - state transition
        g - gating with silu activation
        o - output
        """

        # 使用随机数生成参数
        q_key, k_key, v_key, a_key, g_key, o_key = random.split(key, 6)

        self.norm = RMSNorm(dim)

        self.wq = random.normal(q_key, (dim, dim))
        self.wk = random.normal(k_key, (dim, dim))
        self.wv = random.normal(v_key, (dim, dim))
        self.wa = random.normal(a_key, (dim, dim * 2))
        self.wg = random.normal(g_key, (dim, dim))
        self.wo = random.normal(o_key, (dim, dim))

    def __call__(self, x):
        x = self.norm(x)

        q = x @ self.wq
        k = x @ self.wk
        v = x @ self.wv
        a = x @ self.wa
        g = x @ self.wg

        # 构成复杂状态转换
        a_real, a_imag = np.split(a, 2, axis = -1)
        a_complex = lax.complex(a_real, a_imag)

        magnitude, phase = np.abs(a_complex), np.angle(a_complex)
        magnitude = nn.sigmoid(magnitude)

        a_complex = magnitude * np.exp(1j * phase)

        # 使用复杂状态进行关联扫描
        y = gate_loop_operator(k, v, q, a_complex)

        # 使用 ReTNet 的 silu gating
        y = y * nn.silu(g)

        o = y @ self.wo

        return o

# basic feedforward with pre-rmsnorm

# 定义带有 RMSNorm 的基本前馈模块
class FeedForward(Module):
    norm: RMSNorm
    proj_in: Linear
    proj_out: Linear

    def __init__(
        self,
        *,
        dim,
        key,
        mult = 4
    ):
        self.norm = RMSNorm(dim)
        self.proj_in = Linear(dim, dim * mult, key = key)
        self.proj_out = Linear(dim * mult, dim, key = key)

    def __call__(self, x):
        x = self.norm(x)
        x = self.proj_in(x)
        x = nn.gelu(x)
        x = self.proj_out(x)
        return x

# main class

# 定义门循环变换器模块
class GateLoopTransformer(Module):
    embedding: np.ndarray
    norm: Module
    layers: List[Tuple[GateLoop, FeedForward]]

    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        key,
        ff_mult = 4
    # 初始化嵌入矩阵，使用正态分布随机初始化，乘以0.02
    self.embedding = random.normal(key, (num_tokens, dim)) * 0.02

    # 初始化层列表
    layers = []

    # 循环创建深度次数的GateLoop和FeedForward层，并添加到层列表中
    for _ in range(depth):
        gateloop = GateLoop(dim = dim, key = key)
        ff = FeedForward(dim = dim, mult = ff_mult, key = key)
        layers.append((gateloop, ff))

    # 将创建的层列表赋值给self.layers
    self.layers = layers

    # 初始化RMSNorm层
    self.norm = RMSNorm(dim)

@jit
def __call__(self, x):
    # 通过嵌入矩阵获取输入x的嵌入向量
    x = self.embedding[x]

    # 遍历每一层，依次进行GateLoop和FeedForward操作
    for gateloop, ff in self.layers:
        x = gateloop(x) + x
        x = ff(x) + x

    # 对输出进行归一化处理
    x = self.norm(x)

    # 计算logits，即输出结果
    logits = x @ self.embedding.transpose()

    return logits
# 如果当前脚本被直接运行
if __name__ == '__main__':
    # 导入 jax 库
    import jax
    # 使用 PRNGKey 创建一个随机种子
    key = jax.random.PRNGKey(0)

    # 创建一个 GateLoopTransformer 模型实例
    model = GateLoopTransformer(
        num_tokens = 20000,
        dim = 512,
        depth = 12,
        key = key
    )

    # 生成一个长度为 1024 的随机整数序列
    seq = jax.random.randint(key, (1024,), 0, 20000)
    # 使用模型对序列进行推理，得到输出 logits
    logits = model(seq)

    # 打印 logits 的形状
    print(logits.shape) # (1024, 20000)
```