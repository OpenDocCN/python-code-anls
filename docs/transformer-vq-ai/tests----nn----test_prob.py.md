# `transformer_vq\tests\nn\test_prob.py`

```
# 导入需要的库
import jax
import jax.numpy as jnp
import numpy as np
from transformer_vq.nn.prob import nucleus

# 定义一个测试函数，用于测试argsort_inversion函数
def test_argsort_inversion() -> None:
    # 生成一个服从正态分布的随机数组
    x = jax.random.normal(jax.random.PRNGKey(0), shape=[100, 10], dtype=jnp.float32)
    # 对数组进行排序
    sorted_x = jnp.sort(x, axis=-1)
    # 获取排序后的索引
    sort_indices = jnp.argsort(x, axis=-1)
    # 获取排序后的索引的逆序
    unsort_indices = jnp.argsort(sort_indices, axis=-1)
    # 根据逆序索引对排序后的数组进行还原
    x_ = jnp.take_along_axis(sorted_x, unsort_indices, axis=-1)
    # 断言还原后的数组与原数组相等
    np.testing.assert_allclose(actual=x_, desired=x)

# 定义一个测试函数，用于测试nucleus函数
def test_nucleus() -> None:
    # 创建一个概率数组
    probs = jnp.array(
        [0.05, 0.05, 0.05, 0.09, 0.11, 0.15, 0.30, 0.20], dtype=jnp.float32
    )
# 计算logits的自然对数
logits = jnp.log(probs)

# 使用nucleus函数对logits进行处理，保留概率分布中累积概率小于0.25的部分
actual = nucleus(logits=logits, p=0.25)
expected = jnp.array([-1e10] * 6 + [logits[-2]] + [-1e10])
np.testing.assert_allclose(actual=actual, desired=expected)

# 使用nucleus函数对logits进行处理，保留概率分布中累积概率小于0.31的部分
actual = nucleus(logits=logits, p=0.31)
expected = jnp.concatenate([jnp.array([-1e10] * 6), logits[-2:]], axis=-1)
np.testing.assert_allclose(actual=actual, desired=expected)

# 使用nucleus函数对logits进行处理，保留概率分布中累积概率小于0.65的部分
actual = nucleus(logits=logits, p=0.65)
expected = jnp.concatenate([jnp.array([-1e10] * 5), logits[-3:]], axis=-1)
np.testing.assert_allclose(actual=actual, desired=expected)

# 使用nucleus函数对logits进行处理，保留概率分布中累积概率小于0.75的部分
actual = nucleus(logits=logits, p=0.75)
expected = jnp.concatenate([jnp.array([-1e10] * 4), logits[-4:]], axis=-1)
np.testing.assert_allclose(actual=actual, desired=expected)

# 使用nucleus函数对logits进行处理，保留概率分布中累积概率小于0.85的部分
actual = nucleus(logits=logits, p=0.85)
expected = jnp.concatenate([jnp.array([-1e10] * 3), logits[-5:]], axis=-1)
# 使用 NumPy 测试库中的 assert_allclose 函数，比较实际值和期望值是否在允许误差范围内
np.testing.assert_allclose(actual=actual, desired=expected)

# 调用 nucleus 函数，传入 logits 参数和概率值 0.90，返回实际值
actual = nucleus(logits=logits, p=0.90)
# 生成期望值，使用 jnp.concatenate 函数连接数组，生成新的数组
expected = jnp.concatenate([jnp.array([-1e10] * 2), logits[-6:]], axis=-1)
# 使用 NumPy 测试库中的 assert_allclose 函数，比较实际值和期望值是否在允许误差范围内
np.testing.assert_allclose(actual=actual, desired=expected)

# 依次类推，对不同概率值下的实际值和期望值进行比较
```