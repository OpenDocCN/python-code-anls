# `D:\src\scipysrc\scipy\scipy\special\tests\test_powm1.py`

```
# 导入 pytest 库，用于编写和运行测试
import pytest
# 导入 numpy 库并重命名为 np，用于科学计算
import numpy as np
# 从 numpy.testing 中导入 assert_allclose 函数，用于检查数组或浮点数是否在给定的误差范围内相等
from numpy.testing import assert_allclose
# 从 scipy.special 中导入 powm1 函数，用于计算 pow(x, y) - 1，其中 pow 是指数函数
from scipy.special import powm1

# 以下是用于测试 powm1 函数的一系列测试用例，每个测试用例包含输入 x, y 和期望的输出 expected，以及相对误差 rtol

# 测试用例列表，每个元组包含 x, y, 期望输出 expected，和相对误差 rtol
powm1_test_cases = [
    (1.25, 0.75, 0.18217701125396976, 1e-15),
    (2.0, 1e-7, 6.931472045825965e-08, 1e-15),
    (25.0, 5e-11, 1.6094379125636148e-10, 1e-15),
    (0.99996, 0.75, -3.0000150002530058e-05, 1e-15),
    (0.9999999999990905, 20, -1.81898940353014e-11, 1e-15),
    (-1.25, 751.0, -6.017550852453444e+72, 2e-15)
]

# 使用 pytest.mark.parametrize 装饰器，将参数化测试应用于 test_powm1 函数
@pytest.mark.parametrize('x, y, expected, rtol', powm1_test_cases)
def test_powm1(x, y, expected, rtol):
    # 调用 powm1 函数计算结果
    p = powm1(x, y)
    # 使用 assert_allclose 检查计算结果 p 是否与期望值 expected 在相对误差 rtol 内相等
    assert_allclose(p, expected, rtol=rtol)

# 更多的参数化测试用例，用于测试特定情况下 powm1 函数的精确输出

@pytest.mark.parametrize('x, y, expected',
                         [(0.0, 0.0, 0.0),
                          (0.0, -1.5, np.inf),
                          (0.0, 1.75, -1.0),
                          (-1.5, 2.0, 1.25),
                          (-1.5, 3.0, -4.375),
                          (np.nan, 0.0, 0.0),
                          (1.0, np.nan, 0.0),
                          (1.0, np.inf, 0.0),
                          (1.0, -np.inf, 0.0),
                          (np.inf, 7.5, np.inf),
                          (np.inf, -7.5, -1.0),
                          (3.25, np.inf, np.inf),
                          (np.inf, np.inf, np.inf),
                          (np.inf, -np.inf, -1.0),
                          (np.inf, 0.0, 0.0),
                          (-np.inf, 0.0, 0.0),
                          (-np.inf, 2.0, np.inf),
                          (-np.inf, 3.0, -np.inf),
                          (-1.0, float(2**53 - 1), -2.0)])
def test_powm1_exact_cases(x, y, expected):
    # 测试一些特定情况下 powm1 函数的精确输出
    p = powm1(x, y)
    # 使用 assert 检查计算结果 p 是否等于期望值 expected
    assert p == expected

@pytest.mark.parametrize('x, y',
                         [(-1.25, 751.03),
                          (-1.25, np.inf),
                          (np.nan, np.nan),
                          (-np.inf, -np.inf),
                          (-np.inf, 2.5)])
def test_powm1_return_nan(x, y):
    # 测试一些特定情况下 powm1 函数返回 NaN 的情况
    p = powm1(x, y)
    # 使用 assert 检查 p 是否为 NaN
    assert np.isnan(p)
```