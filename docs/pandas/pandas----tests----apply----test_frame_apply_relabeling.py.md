# `D:\src\scipysrc\pandas\pandas\tests\apply\test_frame_apply_relabeling.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于单元测试

from pandas.compat.numpy import np_version_gte1p25  # 导入 NumPy 兼容性模块，用于版本比较

import pandas as pd  # 导入 Pandas 库，用于数据处理
import pandas._testing as tm  # 导入 Pandas 内部测试模块

def test_agg_relabel():
    # GH 26513
    df = pd.DataFrame({"A": [1, 2, 1, 2], "B": [1, 2, 3, 4], "C": [3, 4, 5, 6]})

    # 简单情况，一个列，一个函数
    result = df.agg(foo=("B", "sum"))
    expected = pd.DataFrame({"B": [10]}, index=pd.Index(["foo"]))
    tm.assert_frame_equal(result, expected)

    # 同一列测试不同方法
    result = df.agg(foo=("B", "sum"), bar=("B", "min"))
    expected = pd.DataFrame({"B": [10, 1]}, index=pd.Index(["foo", "bar"]))
    tm.assert_frame_equal(result, expected)


def test_agg_relabel_multi_columns_multi_methods():
    # GH 26513, 测试多列多方法
    df = pd.DataFrame({"A": [1, 2, 1, 2], "B": [1, 2, 3, 4], "C": [3, 4, 5, 6]})
    result = df.agg(
        foo=("A", "sum"),
        bar=("B", "mean"),
        cat=("A", "min"),
        dat=("B", "max"),
        f=("A", "max"),
        g=("C", "min"),
    )
    expected = pd.DataFrame(
        {
            "A": [6.0, np.nan, 1.0, np.nan, 2.0, np.nan],
            "B": [np.nan, 2.5, np.nan, 4.0, np.nan, np.nan],
            "C": [np.nan, np.nan, np.nan, np.nan, np.nan, 3.0],
        },
        index=pd.Index(["foo", "bar", "cat", "dat", "f", "g"]),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.xfail(np_version_gte1p25, reason="name of min now equals name of np.min")
def test_agg_relabel_partial_functions():
    # GH 26513, 测试 partial, functools 或更复杂的情况
    df = pd.DataFrame({"A": [1, 2, 1, 2], "B": [1, 2, 3, 4], "C": [3, 4, 5, 6]})
    result = df.agg(foo=("A", np.mean), bar=("A", "mean"), cat=("A", min))
    expected = pd.DataFrame(
        {"A": [1.5, 1.5, 1.0]}, index=pd.Index(["foo", "bar", "cat"])
    )
    tm.assert_frame_equal(result, expected)

    result = df.agg(
        foo=("A", min),
        bar=("A", np.min),
        cat=("B", max),
        dat=("C", "min"),
        f=("B", np.sum),
        kk=("B", lambda x: min(x)),
    )
    expected = pd.DataFrame(
        {
            "A": [1.0, 1.0, np.nan, np.nan, np.nan, np.nan],
            "B": [np.nan, np.nan, 4.0, np.nan, 10.0, 1.0],
            "C": [np.nan, np.nan, np.nan, 3.0, np.nan, np.nan],
        },
        index=pd.Index(["foo", "bar", "cat", "dat", "f", "kk"]),
    )
    tm.assert_frame_equal(result, expected)


def test_agg_namedtuple():
    # GH 26513
    df = pd.DataFrame({"A": [0, 1], "B": [1, 2]})
    result = df.agg(
        foo=pd.NamedAgg("B", "sum"),
        bar=pd.NamedAgg("B", "min"),
        cat=pd.NamedAgg(column="B", aggfunc="count"),
        fft=pd.NamedAgg("B", aggfunc="max"),
    )

    expected = pd.DataFrame(
        {"B": [3, 1, 2, 2]}, index=pd.Index(["foo", "bar", "cat", "fft"])
    )
    tm.assert_frame_equal(result, expected)
    # 使用 DataFrame 的 agg 方法进行聚合操作，生成一个包含指定聚合结果的 Series 对象
    result = df.agg(
        foo=pd.NamedAgg("A", "min"),  # 使用 "A" 列的最小值作为 "foo" 列的聚合结果
        bar=pd.NamedAgg(column="B", aggfunc="max"),  # 使用 "B" 列的最大值作为 "bar" 列的聚合结果
        cat=pd.NamedAgg(column="A", aggfunc="max"),  # 使用 "A" 列的最大值作为 "cat" 列的聚合结果
    )
    
    # 创建一个预期的 DataFrame 对象，包含指定的数据和索引
    expected = pd.DataFrame(
        {"A": [0.0, np.nan, 1.0], "B": [np.nan, 2.0, np.nan]},  # 数据为指定的字典结构
        index=pd.Index(["foo", "bar", "cat"]),  # 指定索引为给定的 Index 对象
    )
    
    # 使用测试工具比较 result 和 expected 两个 DataFrame 对象是否相等
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数 test_reconstruct_func
def test_reconstruct_func():
    # GH 28472, 测试以确保 reconstruct_func 不会被移动；
    # 其他库（例如 dask）使用此方法
    # 调用 Pandas 核心模块中的 reconstruct_func 函数，传入参数 "min"
    result = pd.core.apply.reconstruct_func("min")
    # 预期的返回结果
    expected = (False, "min", None, None)
    # 使用测试工具（tm）来比较 result 和 expected 是否相等
    tm.assert_equal(result, expected)
```