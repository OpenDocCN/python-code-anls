# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_missing.py`

```
# 导入所需的库
import numpy as np  # 导入 NumPy 库，用于处理数值计算
import pytest  # 导入 Pytest 库，用于编写和运行单元测试

# 导入 sklearn 库中的函数用于检查标量是否为 NaN
from sklearn.utils._missing import is_scalar_nan


# 使用 Pytest 的 parametrize 装饰器为测试函数 test_is_scalar_nan 提供多组参数化输入
@pytest.mark.parametrize(
    "value, result",
    [
        (float("nan"), True),                 # 测试 float("nan")，预期结果为 True
        (np.nan, True),                       # 测试 np.nan，预期结果为 True
        (float(np.nan), True),                # 测试 float(np.nan)，预期结果为 True
        (np.float32(np.nan), True),           # 测试 np.float32(np.nan)，预期结果为 True
        (np.float64(np.nan), True),           # 测试 np.float64(np.nan)，预期结果为 True
        (0, False),                           # 测试整数 0，预期结果为 False
        (0.0, False),                         # 测试浮点数 0.0，预期结果为 False
        (None, False),                        # 测试 None，预期结果为 False
        ("", False),                          # 测试空字符串 ""，预期结果为 False
        ("nan", False),                       # 测试字符串 "nan"，预期结果为 False
        ([np.nan], False),                    # 测试包含 np.nan 的列表，预期结果为 False
        (9867966753463435747313673, False),   # 测试大整数，预期结果为 False（Python int that overflows with C type）
    ],
)
# 测试函数，检查 is_scalar_nan 函数对各种输入值的处理是否符合预期
def test_is_scalar_nan(value, result):
    assert is_scalar_nan(value) is result  # 断言 is_scalar_nan 的返回值与预期结果相等
    assert isinstance(is_scalar_nan(value), bool)  # 断言 is_scalar_nan 返回的结果是 Python 的布尔类型
```