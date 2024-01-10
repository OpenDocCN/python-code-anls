# `transformer_vq\src\transformer_vq\nn\prob.py`

```
import jax  # 导入 jax 库
import jax.numpy as jnp  # 导入 jax 库中的 numpy 模块，并重命名为 jnp

# 定义一个函数 nucleus，接受 logits 和 p 两个参数
def nucleus(logits, p):
    n_vocab = logits.shape[-1]  # 获取 logits 的最后一个维度大小，即词汇表的大小
    # 计算概率
    probs = jax.nn.softmax(logits, axis=-1)
    # 对概率进行升序排序，并获取其 argsort 索引
    sorted_probs = jnp.sort(probs, axis=-1)
    sort_indices = jnp.argsort(probs, axis=-1)
    cumulative_sorted_probs = jnp.cumsum(sorted_probs, axis=-1)
    # 创建一个 nucleus 掩码，用于对排序后的概率进行筛选
    # 掩码接受最大概率的 tokens，使其总和小于或等于 p，并始终包括最大概率的 token
    m1 = jnp.greater(cumulative_sorted_probs, 1.0 - (p - 1e-4))  # "is tail > 1-p"?
    m2 = jnp.equal(
        jnp.arange(n_vocab),
        jnp.full(fill_value=n_vocab - 1, shape=[n_vocab]),
    )
    mask_for_sorted = jnp.logical_or(m1, m2).astype(jnp.int32)
    # 对掩码进行非排序，以便它适用于非排序顺序中的 token logits
    unsort_indices = jnp.argsort(sort_indices, axis=-1)
    mask = jnp.take_along_axis(mask_for_sorted, unsort_indices, axis=-1)
    # 掩盖非核心 logits
    masked_logits = logits * mask - 1e10 * (1 - mask)
    return masked_logits  # 返回掩盖后的 logits
```