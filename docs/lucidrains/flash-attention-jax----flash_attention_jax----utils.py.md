# `.\lucidrains\flash-attention-jax\flash_attention_jax\utils.py`

```
# 导入 JAX 库
import jax
# 导入 partial 函数
from functools import partial
# 导入 JAX 中的 numpy 模块
import jax.numpy as jnp
# 从 JAX 中导入 random 模块
from jax import random
# 从 JAX 中导入 value_and_grad 函数

# 定义一个装饰器函数，用于计算函数值和梯度
def value_and_grad_wrapper(fn, **kwargs):
    # 使用 partial 函数将 value_and_grad 函数应用到 fn 函数上
    @partial(value_and_grad, **kwargs)
    def inner(*args, **kwargs):
        # 返回 fn 函数的和
        return jnp.sum(fn(*args, **kwargs))
    return inner

# 定义计算两个张量之间差异的函数
def diff(t1, t2):
    # 返回两个张量之间的最大绝对值差
    return jnp.max(jnp.abs(t1 - t2))

# 定义 PRNGKey 生成器函数
def PRNGKeyGenerator(seed = 42):
    # 使用给定种子创建 PRNGKey
    key = random.PRNGKey(seed)
    # 生成子密钥
    while True:
        sub_key, key = random.split(key)
        yield sub_key

# 定义计算两个函数值和梯度之间差异的函数
def value_and_grad_difference(
    fn1,
    fn2,
    seed = 42,
    batch = 2,
    heads = 4,
    q_seq_len = 4096,
    k_seq_len = 8192,
    add_key_mask = True,
    dim = 512
):
    # 创建 PRNGKey 生成器
    key_gen = PRNGKeyGenerator(seed)

    # 生成随机正态分布的张量 q, k, v
    q = random.normal(next(key_gen), (batch, heads, q_seq_len, dim))
    k = random.normal(next(key_gen), (batch, heads, k_seq_len, dim))
    v = random.normal(next(key_gen), (batch, heads, k_seq_len, dim))

    # 生成随机的 key_mask
    key_mask = random.randint(next(key_gen), (batch, k_seq_len), 0, 2) == 1

    # 使用 partial 函数将 value_and_grad_wrapper 函数应用到 fn1, fn2 上
    fn1_value_and_grad, fn2_value_and_grad = map(partial(value_and_grad_wrapper, argnums = (0, 1, 2)), (fn1, fn2))

    # 将参数 q, k, v 和 key_mask（如果需要）传递给函数 fn1 和 fn2，并计算函数值和梯度
    args = (q, k, v)
    if add_key_mask:
        args = (*args, key_mask)

    # 计算 fn1 和 fn2 的函数值和梯度
    o1, grads1 = fn1_value_and_grad(*args)
    o2, grads2 = fn2_value_and_grad(*args)

    # 返回函数值之间的差异和梯度之间的差异
    return diff(o1, o2), [diff(*args) for args in zip(grads1, grads2)]
```