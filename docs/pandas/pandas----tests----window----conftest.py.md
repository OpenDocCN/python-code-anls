# `D:\src\scipysrc\pandas\pandas\tests\window\conftest.py`

```
# 导入 datetime 和 timedelta 类
from datetime import (
    datetime,
    timedelta,
)

# 导入 numpy 库，并使用 np 别名
import numpy as np

# 导入 pytest 库
import pytest

# 导入 pandas.util._test_decorators 模块，并使用 td 别名
import pandas.util._test_decorators as td

# 从 pandas 库导入 DataFrame、Series、bdate_range 函数
from pandas import (
    DataFrame,
    Series,
    bdate_range,
)


# 创建参数化 fixture，用于返回 True 或 False
@pytest.fixture(params=[True, False])
def raw(request):
    """raw keyword argument for rolling.apply"""
    return request.param


# 创建参数化 fixture，包含多个字符串选项
@pytest.fixture(
    params=[
        "sum",
        "mean",
        "median",
        "max",
        "min",
        "var",
        "std",
        "kurt",
        "skew",
        "count",
        "sem",
    ]
)
def arithmetic_win_operators(request):
    return request.param


# 创建参数化 fixture，用于返回 True 或 False
@pytest.fixture(params=[True, False])
def center(request):
    return request.param


# 创建参数化 fixture，可以返回 None 或整数 1
@pytest.fixture(params=[None, 1])
def min_periods(request):
    return request.param


# 创建参数化 fixture，用于返回 True 或 False
@pytest.fixture(params=[True, False])
def adjust(request):
    """adjust keyword argument for ewm"""
    return request.param


# 创建参数化 fixture，用于返回 True 或 False
@pytest.fixture(params=[True, False])
def ignore_na(request):
    """ignore_na keyword argument for ewm"""
    return request.param


# 创建参数化 fixture，用于返回 True 或 False
@pytest.fixture(params=[True, False])
def numeric_only(request):
    """numeric_only keyword argument"""
    return request.param


# 创建参数化 fixture，包含多个元组选项和字符串选项
@pytest.fixture(
    params=[
        pytest.param("numba", marks=[td.skip_if_no("numba"), pytest.mark.single_cpu]),
        "cython",
    ]
)
def engine(request):
    """engine keyword argument for rolling.apply"""
    return request.param


# 创建参数化 fixture，包含多个元组选项
@pytest.fixture(
    params=[
        pytest.param(
            ("numba", True), marks=[td.skip_if_no("numba"), pytest.mark.single_cpu]
        ),
        ("cython", True),
        ("cython", False),
    ]
)
def engine_and_raw(request):
    """engine and raw keyword arguments for rolling.apply"""
    return request.param


# 创建参数化 fixture，包含不同类型的时间间隔选项
@pytest.fixture(params=["1 day", timedelta(days=1), np.timedelta64(1, "D")])
def halflife_with_times(request):
    """Halflife argument for EWM when times is specified."""
    return request.param


# 创建 fixture，返回一个随机生成的 Series 对象
@pytest.fixture
def series():
    """Make mocked series as fixture."""
    arr = np.random.default_rng(2).standard_normal(100)
    locs = np.arange(20, 40)
    arr[locs] = np.nan
    series = Series(arr, index=bdate_range(datetime(2009, 1, 1), periods=100))
    return series


# 创建 fixture，返回一个随机生成的 DataFrame 对象
@pytest.fixture
def frame():
    """Make mocked frame as fixture."""
    return DataFrame(
        np.random.default_rng(2).standard_normal((100, 10)),
        index=bdate_range(datetime(2009, 1, 1), periods=100),
    )


# 创建参数化 fixture，用于返回 None 或整数
@pytest.fixture(params=[None, 1, 2, 5, 10])
def step(request):
    """step keyword argument for rolling window operations."""
    return request.param
```