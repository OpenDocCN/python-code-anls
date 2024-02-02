# `transformer_vq\tests\nn\test_prob.py`

```py
# 导入需要的库
import jax
import jax.numpy as jnp
import numpy as np
# 从transformer_vq.nn.prob模块中导入nucleus函数

# 定义测试函数test_argsort_inversion
def test_argsort_inversion() -> None:
    # 生成服从正态分布的随机数矩阵
    x = jax.random.normal(jax.random.PRNGKey(0), shape=[100, 10], dtype=jnp.float32)
    # 对矩阵按最后一个维度进行排序
    sorted_x = jnp.sort(x, axis=-1)
    # 返回排序后的索引
    sort_indices = jnp.argsort(x, axis=-1)
    # 返回排序后的索引的逆序
    unsort_indices = jnp.argsort(sort_indices, axis=-1)
    # 根据逆序索引重新排列矩阵
    x_ = jnp.take_along_axis(sorted_x, unsort_indices, axis=-1)
    # 断言重新排列后的矩阵与原矩阵相等
    np.testing.assert_allclose(actual=x_, desired=x)

# 定义测试函数test_nucleus
def test_nucleus() -> None:
    # 定义概率数组
    probs = jnp.array(
        [0.05, 0.05, 0.05, 0.09, 0.11, 0.15, 0.30, 0.20], dtype=jnp.float32
    )
    # 计算概率的对数
    logits = jnp.log(probs)

    # 测试nucleus函数，p=0.25
    actual = nucleus(logits=logits, p=0.25)
    expected = jnp.array([-1e10] * 6 + [logits[-2]] + [-1e10])
    np.testing.assert_allclose(actual=actual, desired=expected)

    # 测试nucleus函数，p=0.31
    actual = nucleus(logits=logits, p=0.31)
    expected = jnp.concatenate([jnp.array([-1e10] * 6), logits[-2:]], axis=-1)
    np.testing.assert_allclose(actual=actual, desired=expected)

    # 测试nucleus函数，p=0.65
    actual = nucleus(logits=logits, p=0.65)
    expected = jnp.concatenate([jnp.array([-1e10] * 5), logits[-3:]], axis=-1)
    np.testing.assert_allclose(actual=actual, desired=expected)

    # 测试nucleus函数，p=0.75
    actual = nucleus(logits=logits, p=0.75)
    expected = jnp.concatenate([jnp.array([-1e10] * 4), logits[-4:]], axis=-1)
    np.testing.assert_allclose(actual=actual, desired=expected)

    # 测试nucleus函数，p=0.85
    actual = nucleus(logits=logits, p=0.85)
    expected = jnp.concatenate([jnp.array([-1e10] * 3), logits[-5:]], axis=-1)
    np.testing.assert_allclose(actual=actual, desired=expected)

    # 测试nucleus函数，p=0.90
    actual = nucleus(logits=logits, p=0.90)
    expected = jnp.concatenate([jnp.array([-1e10] * 2), logits[-6:]], axis=-1)
    np.testing.assert_allclose(actual=actual, desired=expected)

    # 测试nucleus函数，p=0.95
    actual = nucleus(logits=logits, p=0.95)
    expected = jnp.concatenate([jnp.array([-1e10] * 1), logits[-7:]], axis=-1)
    np.testing.assert_allclose(actual=actual, desired=expected)

    # 测试nucleus函数，p=0.99
    actual = nucleus(logits=logits, p=0.99)
    # 将变量logits的值赋给变量expected，用于后续的断言比较
    expected = logits
    # 使用 NumPy 测试库进行断言比较，检查变量actual和变量expected的值是否全部接近
    np.testing.assert_allclose(actual=actual, desired=expected)
    
    # 调用nucleus函数，传入logits作为参数，设置p值为1.0，将返回值赋给变量actual
    actual = nucleus(logits=logits, p=1.0)
    # 将变量logits的值赋给变量expected，用于后续的断言比较
    expected = logits
    # 使用 NumPy 测试库进行断言比较，检查变量actual和变量expected的值是否全部接近
    np.testing.assert_allclose(actual=actual, desired=expected)
```