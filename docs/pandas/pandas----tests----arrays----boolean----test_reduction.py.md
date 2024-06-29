# `D:\src\scipysrc\pandas\pandas\tests\arrays\boolean\test_reduction.py`

```
# 导入所需的库
import numpy as np
import pytest

# 导入 pandas 库，并使用 pd 别名
import pandas as pd


# 定义一个测试用的 fixture，返回一个包含布尔值和缺失值的数组
@pytest.fixture
def data():
    """Fixture returning boolean array, with valid and missing values."""
    return pd.array(
        [True, False] * 4 + [np.nan] + [True, False] * 44 + [np.nan] + [True, False],
        dtype="boolean",
    )


# 使用 pytest 的 parametrize 装饰器定义多组参数化测试数据
@pytest.mark.parametrize(
    "values, exp_any, exp_all, exp_any_noskip, exp_all_noskip",
    [
        ([True, pd.NA], True, True, True, pd.NA),
        ([False, pd.NA], False, False, pd.NA, False),
        ([pd.NA], False, True, pd.NA, pd.NA),
        ([], False, True, False, True),
        # GH-33253: all True / all False values buggy with skipna=False
        ([True, True], True, True, True, True),
        ([False, False], False, False, False, False),
    ],
)
def test_any_all(values, exp_any, exp_all, exp_any_noskip, exp_all_noskip):
    # 将 pd.NA 转换为对应的 numpy 布尔类型
    exp_any = pd.NA if exp_any is pd.NA else np.bool_(exp_any)
    exp_all = pd.NA if exp_all is pd.NA else np.bool_(exp_all)
    exp_any_noskip = pd.NA if exp_any_noskip is pd.NA else np.bool_(exp_any_noskip)
    exp_all_noskip = pd.NA if exp_all_noskip is pd.NA else np.bool_(exp_all_noskip)

    # 对于 pd.array 和 pd.Series 进行迭代测试
    for con in [pd.array, pd.Series]:
        a = con(values, dtype="boolean")
        # 断言任意值和所有值的结果是否与期望一致
        assert a.any() is exp_any
        assert a.all() is exp_all
        assert a.any(skipna=False) is exp_any_noskip
        assert a.all(skipna=False) is exp_all_noskip

        # 使用 numpy 的 np.any 和 np.all 进行进一步断言
        assert np.any(a.any()) is exp_any
        assert np.all(a.all()) is exp_all


# 定义测试函数，验证数据的归约操作的返回类型
def test_reductions_return_types(dropna, data, all_numeric_reductions):
    op = all_numeric_reductions
    s = pd.Series(data)
    if dropna:
        s = s.dropna()

    # 根据操作类型进行不同的断言
    if op in ("sum", "prod"):
        assert isinstance(getattr(s, op)(), np.int_)
    elif op == "count":
        # 在 32 位编译中，返回类型为 intc，但不是 intp
        assert isinstance(getattr(s, op)(), np.integer)
    elif op in ("min", "max"):
        assert isinstance(getattr(s, op)(), np.bool_)
    else:
        # 对于 "mean", "std", "var", "median", "kurt", "skew"，返回类型应为 np.float64
        assert isinstance(getattr(s, op)(), np.float64)
```