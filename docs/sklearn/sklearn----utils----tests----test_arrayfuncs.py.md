# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_arrayfuncs.py`

```
# 导入所需的库和模块
import numpy as np
import pytest

# 导入 sklearn 库中的测试工具和数组处理函数
from sklearn.utils._testing import assert_allclose
from sklearn.utils.arrayfuncs import _all_with_any_reduction_axis_1, min_pos


# 定义测试函数 test_min_pos()
def test_min_pos():
    # 检查 min_pos 函数返回的值为正，并且在 float 和 double 数据类型下一致
    X = np.random.RandomState(0).randn(100)

    min_double = min_pos(X)  # 计算 double 类型下的最小正数
    min_float = min_pos(X.astype(np.float32))  # 计算 float 类型下的最小正数

    # 断言两者的值接近
    assert_allclose(min_double, min_float)
    # 断言最小正数大于等于 0
    assert min_double >= 0


# 使用 pytest 提供的参数化装饰器，测试不同数据类型下 min_pos 的行为
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_min_pos_no_positive(dtype):
    # 检查当输入数组中所有元素 <= 0 时，min_pos 函数返回输入数据类型的最大表示值 (#19328)
    X = np.full(100, -1.0).astype(dtype, copy=False)

    # 断言 min_pos 返回值等于对应数据类型的最大值
    assert min_pos(X) == np.finfo(dtype).max


# 使用 pytest 参数化装饰器测试多种数据类型和取值的情况下 _all_with_any_reduction_axis_1 函数的行为
@pytest.mark.parametrize(
    "dtype", [np.int16, np.int32, np.int64, np.float32, np.float64]
)
@pytest.mark.parametrize("value", [0, 1.5, -1])
def test_all_with_any_reduction_axis_1(dtype, value):
    # 检查当没有任何行等于 `value` 时，_all_with_any_reduction_axis_1 返回 False
    X = np.arange(12, dtype=dtype).reshape(3, 4)
    assert not _all_with_any_reduction_axis_1(X, value=value)

    # 使一行的所有元素等于 `value`
    X[1, :] = value
    # 断言此时 _all_with_any_reduction_axis_1 返回 True
    assert _all_with_any_reduction_axis_1(X, value=value)
```