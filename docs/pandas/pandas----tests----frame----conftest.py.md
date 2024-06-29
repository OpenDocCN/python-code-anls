# `D:\src\scipysrc\pandas\pandas\tests\frame\conftest.py`

```
# 导入所需的库
import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 库

from pandas import (  # 从 pandas 库中导入以下内容：
    DataFrame,  # 数据框
    Index,  # 索引
    NaT,  # 表示缺失日期时间的特殊值
    date_range,  # 生成日期范围的函数
)


@pytest.fixture
def datetime_frame() -> DataFrame:
    """
    Fixture for DataFrame of floats with DatetimeIndex

    Columns are ['A', 'B', 'C', 'D']
    """
    return DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),  # 使用正态分布生成 10 行 4 列的随机浮点数数据
        columns=Index(list("ABCD"), dtype=object),  # 指定列名为 ['A', 'B', 'C', 'D']，数据类型为对象
        index=date_range("2000-01-01", periods=10, freq="B"),  # 生成从 2000-01-01 开始的 10 个工作日日期作为索引
    )


@pytest.fixture
def float_string_frame():
    """
    Fixture for DataFrame of floats and strings with index of unique strings

    Columns are ['A', 'B', 'C', 'D', 'foo'].
    """
    df = DataFrame(
        np.random.default_rng(2).standard_normal((30, 4)),  # 使用正态分布生成 30 行 4 列的随机浮点数数据
        index=Index([f"foo_{i}" for i in range(30)], dtype=object),  # 生成包含 30 个唯一字符串的索引
        columns=Index(list("ABCD"), dtype=object),  # 指定列名为 ['A', 'B', 'C', 'D']，数据类型为对象
    )
    df["foo"] = "bar"  # 添加一列名为 'foo'，所有行设置为字符串 'bar'
    return df


@pytest.fixture
def mixed_float_frame():
    """
    Fixture for DataFrame of different float types with index of unique strings

    Columns are ['A', 'B', 'C', 'D'].
    """
    df = DataFrame(
        {
            col: np.random.default_rng(2).random(30, dtype=dtype)  # 使用指定类型的随机数填充各列
            for col, dtype in zip(
                list("ABCD"), ["float32", "float32", "float32", "float64"]
            )
        },
        index=Index([f"foo_{i}" for i in range(30)], dtype=object),  # 生成包含 30 个唯一字符串的索引
    )
    df["C"] = df["C"].astype("float16")  # 将列 'C' 的数据类型转换为 float16（仅支持少数的 numpy 随机数类型）
    return df


@pytest.fixture
def mixed_int_frame():
    """
    Fixture for DataFrame of different int types with index of unique strings

    Columns are ['A', 'B', 'C', 'D'].
    """
    return DataFrame(
        {
            col: np.ones(30, dtype=dtype)  # 使用指定类型的随机数填充各列（均为1）
            for col, dtype in zip(list("ABCD"), ["int32", "uint64", "uint8", "int64"])
        },
        index=Index([f"foo_{i}" for i in range(30)], dtype=object),  # 生成包含 30 个唯一字符串的索引
    )


@pytest.fixture
def timezone_frame():
    """
    Fixture for DataFrame of date_range Series with different time zones

    Columns are ['A', 'B', 'C']; some entries are missing

               A                         B                         C
    0 2013-01-01 2013-01-01 00:00:00-05:00 2013-01-01 00:00:00+01:00
    1 2013-01-02                       NaT                       NaT
    2 2013-01-03 2013-01-03 00:00:00-05:00 2013-01-03 00:00:00+01:00
    """
    df = DataFrame(
        {
            "A": date_range("20130101", periods=3),  # 生成包含 3 个日期的时间序列，作为列 'A'
            "B": date_range("20130101", periods=3, tz="US/Eastern"),  # 生成包含 3 个日期的带时区（美国东部时区）时间序列，作为列 'B'
            "C": date_range("20130101", periods=3, tz="CET"),  # 生成包含 3 个日期的带时区（中欧时区）时间序列，作为列 'C'
        }
    )
    df.iloc[1, 1] = NaT  # 将第二行第二列的值设置为 NaT（Not a Time，表示缺失日期时间）
    df.iloc[1, 2] = NaT  # 将第二行第三列的值设置为 NaT（表示缺失日期时间）
    return df
```