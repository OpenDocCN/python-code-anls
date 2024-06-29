# `D:\src\scipysrc\pandas\pandas\tests\io\parser\dtypes\test_empty.py`

```
"""
Tests dtype specification during parsing
for all of the parsers defined in parsers.py
"""

# 导入所需的模块和库
from io import StringIO  # 导入StringIO类，用于模拟文件读取

import numpy as np  # 导入NumPy库，用于数据处理
import pytest  # 导入pytest库，用于测试框架

from pandas import (  # 导入Pandas库中的多个类和函数
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    concat,
)
import pandas._testing as tm  # 导入Pandas内部测试模块

skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")  # 标记测试跳过pyarrow的装饰器


@skip_pyarrow  # CSV解析错误：空的CSV文件或块
def test_dtype_all_columns_empty(all_parsers):
    # 见gh-12048
    parser = all_parsers
    result = parser.read_csv(StringIO("A,B"), dtype=str)

    expected = DataFrame({"A": [], "B": []}, dtype=str)
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV解析错误：空的CSV文件或块
def test_empty_pass_dtype(all_parsers):
    parser = all_parsers

    data = "one,two"
    result = parser.read_csv(StringIO(data), dtype={"one": "u1"})

    expected = DataFrame(
        {"one": np.empty(0, dtype="u1"), "two": np.empty(0, dtype=object)},
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV解析错误：空的CSV文件或块
def test_empty_with_index_pass_dtype(all_parsers):
    parser = all_parsers

    data = "one,two"
    result = parser.read_csv(
        StringIO(data), index_col=["one"], dtype={"one": "u1", 1: "f"}
    )

    expected = DataFrame(
        {"two": np.empty(0, dtype="f")}, index=Index([], dtype="u1", name="one")
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV解析错误：空的CSV文件或块
def test_empty_with_multi_index_pass_dtype(all_parsers):
    parser = all_parsers

    data = "one,two,three"
    result = parser.read_csv(
        StringIO(data), index_col=["one", "two"], dtype={"one": "u1", 1: "f8"}
    )

    exp_idx = MultiIndex.from_arrays(
        [np.empty(0, dtype="u1"), np.empty(0, dtype=np.float64)],
        names=["one", "two"],
    )
    expected = DataFrame({"three": np.empty(0, dtype=object)}, index=exp_idx)
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV解析错误：空的CSV文件或块
def test_empty_with_mangled_column_pass_dtype_by_names(all_parsers):
    parser = all_parsers

    data = "one,one"
    result = parser.read_csv(StringIO(data), dtype={"one": "u1", "one.1": "f"})

    expected = DataFrame(
        {"one": np.empty(0, dtype="u1"), "one.1": np.empty(0, dtype="f")},
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV解析错误：空的CSV文件或块
def test_empty_with_mangled_column_pass_dtype_by_indexes(all_parsers):
    parser = all_parsers

    data = "one,one"
    result = parser.read_csv(StringIO(data), dtype={0: "u1", 1: "f"})

    expected = DataFrame(
        {"one": np.empty(0, dtype="u1"), "one.1": np.empty(0, dtype="f")},
    )
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV解析错误：空的CSV文件或块
def test_empty_with_dup_column_pass_dtype_by_indexes(all_parsers):
    # 见gh-9424
    parser = all_parsers
    # 创建一个期望的数据框架 `expected`，使用 concat 函数将两个空 Series 水平拼接成一个 DataFrame
    expected = concat(
        [Series([], name="one", dtype="u1"), Series([], name="one.1", dtype="f")],
        axis=1,
    )
    
    # 准备一个简单的 CSV 数据字符串 `data`，包含列名 "one,one"
    data = "one,one"
    
    # 使用 parser 对象的 read_csv 方法解析给定的 CSV 字符串，并指定第一列为无符号整数，第二列为浮点数
    result = parser.read_csv(StringIO(data), dtype={0: "u1", 1: "f"})
    
    # 使用 tm 模块的 assert_frame_equal 函数比较解析结果 `result` 和期望的数据框架 `expected` 是否相等
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于测试处理空数据和重复列名时的行为，特别是使用索引传递 dtype 参数时是否引发异常
def test_empty_with_dup_column_pass_dtype_by_indexes_raises(all_parsers):
    # 使用 pytest 标记测试用例，引用 issue gh-9424
    parser = all_parsers
    # 创建期望的 DataFrame，包含两个空的 Series，分别命名为 "one" 和 "one.1"，数据类型分别为 "u1" 和 "f"
    expected = concat(
        [Series([], name="one", dtype="u1"), Series([], name="one.1", dtype="f")],
        axis=1,
    )
    # 将期望的 DataFrame 的索引转换为对象类型
    expected.index = expected.index.astype(object)

    # 使用 pytest 的上下文管理语句，期望引发 ValueError 异常，异常信息包含 "Duplicate names"
    with pytest.raises(ValueError, match="Duplicate names"):
        # 调用 parser 对象的 read_csv 方法，传递一个空字符串，指定列名为 ["one", "one"]，并指定第0列和第1列的数据类型分别为 "u1" 和 "f"
        parser.read_csv(StringIO(""), names=["one", "one"], dtype={0: "u1", 1: "f"})


# 使用 pytest.mark.parametrize 装饰器定义参数化测试用例，测试不同的 dtype 参数对 read_csv 方法的影响
@pytest.mark.parametrize(
    "dtype,expected",
    [
        # 测试 dtype 为 np.float64 时的预期结果，期望创建一个包含列 "a" 和 "b" 的空 DataFrame，数据类型为 np.float64
        (np.float64, DataFrame(columns=["a", "b"], dtype=np.float64)),
        (
            "category",
            DataFrame({"a": Categorical([]), "b": Categorical([])}),
        ),  # 测试 dtype 为 "category" 时的预期结果，期望创建一个包含两列的 DataFrame，数据类型为 Categorical
        (
            {"a": "category", "b": "category"},
            DataFrame({"a": Categorical([]), "b": Categorical([])}),
        ),  # 测试 dtype 为 {"a": "category", "b": "category"} 时的预期结果，期望创建一个包含两列的 DataFrame，数据类型为 Categorical
        ("datetime64[ns]", DataFrame(columns=["a", "b"], dtype="datetime64[ns]")),  # 测试 dtype 为 "datetime64[ns]" 时的预期结果，期望创建一个包含列 "a" 和 "b" 的空 DataFrame，数据类型为 datetime64[ns]
        (
            "timedelta64[ns]",
            DataFrame(
                {
                    "a": Series([], dtype="timedelta64[ns]"),
                    "b": Series([], dtype="timedelta64[ns]"),
                },
            ),
        ),  # 测试 dtype 为 "timedelta64[ns]" 时的预期结果，期望创建一个包含两列的 DataFrame，数据类型为 timedelta64[ns]
        (
            {"a": np.int64, "b": np.int32},
            DataFrame(
                {"a": Series([], dtype=np.int64), "b": Series([], dtype=np.int32)},
            ),
        ),  # 测试 dtype 为 {"a": np.int64, "b": np.int32} 时的预期结果，期望创建一个包含两列的 DataFrame，分别数据类型为 np.int64 和 np.int32
        (
            {0: np.int64, 1: np.int32},
            DataFrame(
                {"a": Series([], dtype=np.int64), "b": Series([], dtype=np.int32)},
            ),
        ),  # 测试 dtype 为 {0: np.int64, 1: np.int32} 时的预期结果，期望创建一个包含两列的 DataFrame，分别数据类型为 np.int64 和 np.int32
        (
            {"a": np.int64, 1: np.int32},
            DataFrame(
                {"a": Series([], dtype=np.int64), "b": Series([], dtype=np.int32)},
            ),
        ),  # 测试 dtype 为 {"a": np.int64, 1: np.int32} 时的预期结果，期望创建一个包含两列的 DataFrame，分别数据类型为 np.int64 和 np.int32
    ],
)
# 使用 @skip_pyarrow 装饰器标记这个测试函数，注释为 CSV 解析错误：空的 CSV 文件或块
def test_empty_dtype(all_parsers, dtype, expected):
    # 引用 issue gh-14712
    parser = all_parsers
    # 定义一个简单的包含 "a,b" 字符串数据的变量
    data = "a,b"

    # 调用 parser 对象的 read_csv 方法，传递 data 字符串作为数据源，指定 header=0 和给定的 dtype 参数
    result = parser.read_csv(StringIO(data), header=0, dtype=dtype)
    # 使用 tm.assert_frame_equal 方法断言 result 和 expected 的 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
```