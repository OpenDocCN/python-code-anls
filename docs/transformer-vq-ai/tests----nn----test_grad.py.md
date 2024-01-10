# `transformer_vq\tests\nn\test_grad.py`

```
# 导入 jax 库
import jax
# 导入 jax 库中的 numpy 模块，并重命名为 jnp
import jax.numpy as jnp
# 导入 numpy 库，并重命名为 np
import numpy as np
# 从 transformer_vq.nn.grad 模块中导入 sg 函数
from transformer_vq.nn.grad import sg
# 从 transformer_vq.nn.grad 模块中导入 st 函数
from transformer_vq.nn.grad import st

# 定义测试 sg 函数的函数
def test_sg():
    # 生成服从正态分布的随机数作为输入
    input_ = jax.random.normal(jax.random.PRNGKey(0), [10], dtype=jnp.float32)
    # 断言 sg 函数的输出与输入相等
    np.testing.assert_allclose(actual=sg(input_), desired=input_)
    # 断言 sg 函数的雅可比矩阵与全零矩阵相等
    np.testing.assert_allclose(
        actual=jax.jacobian(sg)(input_), desired=np.zeros([10, 10])
    )

# 定义测试 st 函数的函数
def test_st():
    # 生成服从正态分布的随机数作为输入
    input_ = jax.random.normal(jax.random.PRNGKey(0), [10], dtype=jnp.float32)
    # 断言 st 函数的输出与全零矩阵相等
    np.testing.assert_allclose(actual=st(input_), desired=np.zeros_like(input_))
    # 断言 st 函数的雅可比矩阵与单位矩阵相等
    np.testing.assert_allclose(actual=jax.jacobian(st)(input_), desired=np.eye(10))
```