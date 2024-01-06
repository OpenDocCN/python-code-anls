# `transformer_vq\src\transformer_vq\nn\prob.py`

```
# 导入 jax 库
import jax
# 导入 jax 库中的 numpy 模块并重命名为 jnp
import jax.numpy as jnp

# 定义一个函数 nucleus，接受 logits 和 p 作为参数
def nucleus(logits, p):
    # 获取 logits 的最后一个维度的大小，即词汇表的大小
    n_vocab = logits.shape[-1]
    # 计算 logits 的 softmax 概率
    probs = jax.nn.softmax(logits, axis=-1)
    # 对概率进行排序，并获取排序后的索引
    sorted_probs = jnp.sort(probs, axis=-1)
    sort_indices = jnp.argsort(probs, axis=-1)
    # 计算累积概率
    cumulative_sorted_probs = jnp.cumsum(sorted_probs, axis=-1)
    # 创建一个 nucleus 掩码，用于对排序后的概率进行筛选
    # 掩码接受累积概率小于等于 p 的最大概率，同时始终包括最大概率的标记
    m1 = jnp.greater(cumulative_sorted_probs, 1.0 - (p - 1e-4))  # "is tail > 1-p"?
    m2 = jnp.equal(
        jnp.arange(n_vocab),
        jnp.full(fill_value=n_vocab - 1, shape=[n_vocab]),
    )
# 创建一个逻辑或的掩码，将其转换为整数类型
mask_for_sorted = jnp.logical_or(m1, m2).astype(jnp.int32)
# 对排序后的索引进行排序，以便将其应用于非排序顺序的标记logits
unsort_indices = jnp.argsort(sort_indices, axis=-1)
# 通过使用unsort_indices对mask_for_sorted进行索引，创建一个掩码
mask = jnp.take_along_axis(mask_for_sorted, unsort_indices, axis=-1)
# 屏蔽非核心logits
masked_logits = logits * mask - 1e10 * (1 - mask)
# 返回屏蔽后的logits
return masked_logits
```