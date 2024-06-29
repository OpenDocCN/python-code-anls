# `D:\src\scipysrc\pandas\pandas\tests\groupby\conftest.py`

```
# 导入必要的库：numpy 和 pytest
import numpy as np
import pytest

# 从 pandas 库中导入以下模块和类：
# DataFrame: 用于创建和操作数据帧
# Index: 用于创建索引对象
# Series: 用于创建和操作序列对象
# date_range: 用于生成日期范围
from pandas import (
    DataFrame,
    Index,
    Series,
    date_range,
)

# 从 pandas.core.groupby.base 模块中导入以下函数：
# reduction_kernels: 包含所有的分组聚合函数的名称
# transformation_kernels: 包含所有的分组变换函数的名称
from pandas.core.groupby.base import (
    reduction_kernels,
    transformation_kernels,
)


@pytest.fixture
def df():
    """
    返回一个 DataFrame 对象，包含列 A, B, C, D：
    - A 和 B 是字符串列
    - C 和 D 是随机生成的标准正态分布数据列
    """
    return DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
            "C": np.random.default_rng(2).standard_normal(8),
            "D": np.random.default_rng(2).standard_normal(8),
        }
    )


@pytest.fixture
def ts():
    """
    返回一个 Series 对象，包含随机生成的标准正态分布数据：
    - 索引为工作日频率，从 '2000-01-01' 开始，共 30 天
    """
    return Series(
        np.random.default_rng(2).standard_normal(30),
        index=date_range("2000-01-01", periods=30, freq="B"),
    )


@pytest.fixture
def tsframe():
    """
    返回一个 DataFrame 对象，包含随机生成的标准正态分布数据：
    - 列名为 ['A', 'B', 'C', 'D']
    - 索引为工作日频率，从 '2000-01-01' 开始，共 30 天
    """
    return DataFrame(
        np.random.default_rng(2).standard_normal((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=30, freq="B"),
    )


@pytest.fixture
def three_group():
    """
    返回一个 DataFrame 对象，包含列 A, B, C, D, E, F：
    - A 和 B 是字符串列
    - C 是字符串列，包含 'dull' 和 'shiny'
    - D, E, F 是随机生成的标准正态分布数据列
    """
    return DataFrame(
        {
            "A": [
                "foo",
                "foo",
                "foo",
                "foo",
                "bar",
                "bar",
                "bar",
                "bar",
                "foo",
                "foo",
                "foo",
            ],
            "B": [
                "one",
                "one",
                "one",
                "two",
                "one",
                "one",
                "one",
                "two",
                "two",
                "two",
                "one",
            ],
            "C": [
                "dull",
                "dull",
                "shiny",
                "dull",
                "dull",
                "shiny",
                "shiny",
                "dull",
                "shiny",
                "shiny",
                "shiny",
            ],
            "D": np.random.default_rng(2).standard_normal(11),
            "E": np.random.default_rng(2).standard_normal(11),
            "F": np.random.default_rng(2).standard_normal(11),
        }
    )


@pytest.fixture
def slice_test_df():
    """
    返回一个 DataFrame 对象，包含列 Index, Group, Value：
    - Index 列作为索引
    - Group 列包含字符串 'a', 'b', 'c'
    - Value 列包含类似 'a0_at_0' 的字符串
    """
    data = [
        [0, "a", "a0_at_0"],
        [1, "b", "b0_at_1"],
        [2, "a", "a1_at_2"],
        [3, "b", "b1_at_3"],
        [4, "c", "c0_at_4"],
        [5, "a", "a2_at_5"],
        [6, "a", "a3_at_6"],
        [7, "a", "a4_at_7"],
    ]
    df = DataFrame(data, columns=["Index", "Group", "Value"])
    return df.set_index("Index")


@pytest.fixture
def slice_test_grouped(slice_test_df):
    """
    返回一个按 Group 列分组后的 GroupBy 对象
    """
    return slice_test_df.groupby("Group", as_index=False)


@pytest.fixture(params=sorted(reduction_kernels))
def reduction_func(request):
    """
    逐个返回分组聚合函数的名称字符串
    """
    return request.param


@pytest.fixture(params=sorted(transformation_kernels))
def transformation_func(request):
    """
    逐个返回分组变换函数的名称字符串
    """
    yield request.param
    # 返回函数参数 request.param 的值作为函数的返回值
    return request.param
@pytest.fixture(params=sorted(reduction_kernels) + sorted(transformation_kernels))
def groupby_func(request):
    """Fixture function yielding both aggregation and transformation functions."""
    return request.param

@pytest.fixture(
    params=[
        ("mean", {}),
        ("var", {"ddof": 1}),
        ("var", {"ddof": 0}),
        ("std", {"ddof": 1}),
        ("std", {"ddof": 0}),
        ("sum", {}),
        ("min", {}),
        ("max", {}),
        ("sum", {"min_count": 2}),
        ("min", {"min_count": 2}),
        ("max", {"min_count": 2}),
    ],
    ids=[
        "mean",
        "var_1",
        "var_0",
        "std_1",
        "std_0",
        "sum",
        "min",
        "max",
        "sum-min_count",
        "min-min_count",
        "max-min_count",
    ],
)
def numba_supported_reductions(request):
    """Fixture providing parameterized data for reductions supported with engine='numba'."""
    return request.param
```