# `D:\src\scipysrc\pandas\pandas\tests\io\parser\dtypes\test_categorical.py`

```
"""
Tests dtype specification during parsing
for all of the parsers defined in parsers.py
"""

# 导入所需的库和模块
from io import StringIO
import os

import numpy as np
import pytest

# 导入 pandas 库及其相关模块
from pandas._libs import parsers as libparsers
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Timestamp,
)
import pandas._testing as tm

# 忽略特定警告信息的 pytest 标记
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

# 对于使用 pyarrow 的测试，标记为预期失败
xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")

# 参数化测试函数，测试分类数据类型
@xfail_pyarrow  # AssertionError: Attributes of DataFrame.iloc[:, 0] are different
@pytest.mark.parametrize(
    "dtype",
    [
        "category",
        CategoricalDtype(),
        {"a": "category", "b": "category", "c": CategoricalDtype()},
    ],
)
def test_categorical_dtype(all_parsers, dtype):
    # see gh-10153
    # 获取所有解析器的实例
    parser = all_parsers
    # 准备测试数据
    data = """a,b,c
1,a,3.4
1,a,3.4
2,b,4.5"""
    # 预期的 DataFrame 结果
    expected = DataFrame(
        {
            "a": Categorical(["1", "1", "2"]),
            "b": Categorical(["a", "a", "b"]),
            "c": Categorical(["3.4", "3.4", "4.5"]),
        }
    )
    # 使用解析器读取数据并进行测试
    actual = parser.read_csv(StringIO(data), dtype=dtype)
    # 断言实际结果与预期结果是否相等
    tm.assert_frame_equal(actual, expected)

# 参数化测试函数，测试单列的分类数据类型
@pytest.mark.parametrize("dtype", [{"b": "category"}, {1: "category"}])
def test_categorical_dtype_single(all_parsers, dtype, request):
    # see gh-10153
    # 获取所有解析器的实例
    parser = all_parsers
    # 准备测试数据
    data = """a,b,c
1,a,3.4
1,a,3.4
2,b,4.5"""
    # 预期的 DataFrame 结果
    expected = DataFrame(
        {"a": [1, 1, 2], "b": Categorical(["a", "a", "b"]), "c": [3.4, 3.4, 4.5]}
    )
    # 如果使用的是 pyarrow 引擎，标记为预期失败以避免测试失败
    if parser.engine == "pyarrow":
        mark = pytest.mark.xfail(
            strict=False,
            reason="Flaky test sometimes gives object dtype instead of Categorical",
        )
        request.applymarker(mark)

    # 使用解析器读取数据并进行测试
    actual = parser.read_csv(StringIO(data), dtype=dtype)
    # 断言实际结果与预期结果是否相等
    tm.assert_frame_equal(actual, expected)

# 预期失败的测试函数，测试无序的分类数据类型
@xfail_pyarrow  # AssertionError: Attributes of DataFrame.iloc[:, 0] are different
def test_categorical_dtype_unsorted(all_parsers):
    # see gh-10153
    # 获取所有解析器的实例
    parser = all_parsers
    # 准备测试数据
    data = """a,b,c
1,b,3.4
1,b,3.4
2,a,4.5"""
    # 预期的 DataFrame 结果
    expected = DataFrame(
        {
            "a": Categorical(["1", "1", "2"]),
            "b": Categorical(["b", "b", "a"]),
            "c": Categorical(["3.4", "3.4", "4.5"]),
        }
    )
    # 使用解析器读取数据并进行测试
    actual = parser.read_csv(StringIO(data), dtype="category")
    # 断言实际结果与预期结果是否相等
    tm.assert_frame_equal(actual, expected)

# 预期失败的测试函数，测试包含缺失值的分类数据类型
@xfail_pyarrow  # AssertionError: Attributes of DataFrame.iloc[:, 0] are different
def test_categorical_dtype_missing(all_parsers):
    # see gh-10153
    # 获取所有解析器的实例
    parser = all_parsers
    # 准备测试数据
    data = """a,b,c
1,b,3.4
1,nan,3.4
2,a,4.5"""
    # 预期的 DataFrame 结果
    expected = DataFrame(
        {
            "a": Categorical(["1", "1", "2"]),
            "b": Categorical(["b", np.nan, "a"]),
            "c": Categorical(["3.4", "3.4", "4.5"]),
        }
    )
    # 使用解析器读取数据并进行测试
    actual = parser.read_csv(StringIO(data), dtype="category")
    # 断言实际结果与预期结果是否相等
    tm.assert_frame_equal(actual, expected)
@xfail_pyarrow  # 标记为预期失败，因为 DataFrame.iloc[:, 0] 的属性不同导致 AssertionError
@pytest.mark.slow
def test_categorical_dtype_high_cardinality_numeric(all_parsers, monkeypatch):
    # 见 gh-18186
    # 曾是 C 解析器的问题，与 DEFAULT_BUFFER_HEURISTIC 有关
    parser = all_parsers
    heuristic = 2**5
    data = np.sort([str(i) for i in range(heuristic + 1)])
    expected = DataFrame({"a": Categorical(data, ordered=True)})
    with monkeypatch.context() as m:
        m.setattr(libparsers, "DEFAULT_BUFFER_HEURISTIC", heuristic)
        actual = parser.read_csv(StringIO("a\n" + "\n".join(data)), dtype="category")
    # 对实际数据重新排序分类，确保顺序正确
    actual["a"] = actual["a"].cat.reorder_categories(
        np.sort(actual.a.cat.categories), ordered=True
    )
    # 断言实际输出与预期结果相等
    tm.assert_frame_equal(actual, expected)


def test_categorical_dtype_utf16(all_parsers, csv_dir_path):
    # 见 gh-10153
    pth = os.path.join(csv_dir_path, "utf16_ex.txt")
    parser = all_parsers
    encoding = "utf-16"
    sep = "\t"

    # 读取预期的 DataFrame，应用分类数据类型
    expected = parser.read_csv(pth, sep=sep, encoding=encoding)
    expected = expected.apply(Categorical)

    # 读取实际的 DataFrame，指定数据类型为分类
    actual = parser.read_csv(pth, sep=sep, encoding=encoding, dtype="category")
    # 断言实际输出与预期结果相等
    tm.assert_frame_equal(actual, expected)


def test_categorical_dtype_chunksize_infer_categories(all_parsers):
    # 见 gh-10153
    parser = all_parsers
    data = """a,b
1,a
1,b
1,b
2,c"""
    expecteds = [
        DataFrame({"a": [1, 1], "b": Categorical(["a", "b"])}),
        DataFrame({"a": [1, 2], "b": Categorical(["b", "c"])}, index=[2, 3]),
    ]

    # 对于 pyarrow 引擎，不支持 'chunksize' 选项
    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), dtype={"b": "category"}, chunksize=2)
        return

    # 用 chunksize=2 读取 CSV 数据，指定 'b' 列的数据类型为分类
    with parser.read_csv(
        StringIO(data), dtype={"b": "category"}, chunksize=2
    ) as actuals:
        for actual, expected in zip(actuals, expecteds):
            # 断言实际输出与预期结果相等
            tm.assert_frame_equal(actual, expected)


def test_categorical_dtype_chunksize_explicit_categories(all_parsers):
    # 见 gh-10153
    parser = all_parsers
    data = """a,b
1,a
1,b
1,b
2,c"""
    cats = ["a", "b", "c"]
    expecteds = [
        DataFrame({"a": [1, 1], "b": Categorical(["a", "b"], categories=cats)}),
        DataFrame(
            {"a": [1, 2], "b": Categorical(["b", "c"], categories=cats)},
            index=[2, 3],
        ),
    ]
    dtype = CategoricalDtype(cats)

    # 对于 pyarrow 引擎，不支持 'chunksize' 选项
    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), dtype={"b": dtype}, chunksize=2)
        return

    # 用 chunksize=2 读取 CSV 数据，指定 'b' 列的数据类型为自定义的分类数据类型
    with parser.read_csv(StringIO(data), dtype={"b": dtype}, chunksize=2) as actuals:
        for actual, expected in zip(actuals, expecteds):
            # 断言实际输出与预期结果相等
            tm.assert_frame_equal(actual, expected)


def test_categorical_dtype_latin1(all_parsers, csv_dir_path):
    pass  # 此函数暂未实现，用于测试 Latin1 编码的情况
    # 将文件路径csv_dir_path和文件名"unicode_series.csv"连接成完整的文件路径
    pth = os.path.join(csv_dir_path, "unicode_series.csv")
    
    # 使用all_parsers作为解析器对象
    parser = all_parsers
    
    # 设置编码格式为"latin-1"
    encoding = "latin-1"
    
    # 使用指定的解析器和编码读取CSV文件内容，没有指定列头(header=None)
    expected = parser.read_csv(pth, header=None, encoding=encoding)
    
    # 将读取的第二列数据转换为分类数据类型
    expected[1] = Categorical(expected[1])
    
    # 再次使用相同的解析器和编码读取CSV文件内容，并指定第二列为"category"数据类型
    actual = parser.read_csv(pth, header=None, encoding=encoding, dtype={1: "category"})
    
    # 使用测试框架中的函数检查实际读取的数据框(actual)与预期的数据框(expected)是否相等
    tm.assert_frame_equal(actual, expected)
@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize(
    "categories",
    [["a", "b", "c"], ["a", "c", "b"], ["a", "b", "c", "d"], ["c", "b", "a"]],
)
# 定义测试函数，参数化测试数据集合和是否有序的标志
def test_categorical_category_dtype(all_parsers, categories, ordered):
    # 获取所有解析器对象
    parser = all_parsers
    # 准备测试数据
    data = """a,b
1,a
1,b
1,b
2,c"""
    # 期望的数据框架
    expected = DataFrame(
        {
            "a": [1, 1, 1, 2],
            "b": Categorical(
                ["a", "b", "b", "c"], categories=categories, ordered=ordered
            ),
        }
    )

    # 指定数据类型，b列为分类数据类型
    dtype = {"b": CategoricalDtype(categories=categories, ordered=ordered)}
    # 使用解析器读取CSV数据，并进行断言比较
    result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)


# 定义测试函数，测试无序分类数据类型
def test_categorical_category_dtype_unsorted(all_parsers):
    # 获取所有解析器对象
    parser = all_parsers
    # 准备测试数据
    data = """a,b
1,a
1,b
1,b
2,c"""
    # 指定b列的数据类型为分类，但是未指定categories
    dtype = CategoricalDtype(["c", "b", "a"])
    # 期望的数据框架
    expected = DataFrame(
        {
            "a": [1, 1, 1, 2],
            "b": Categorical(["a", "b", "b", "c"], categories=["c", "b", "a"]),
        }
    )

    # 使用解析器读取CSV数据，并进行断言比较
    result = parser.read_csv(StringIO(data), dtype={"b": dtype})
    tm.assert_frame_equal(result, expected)


# 定义测试函数，测试数值类型被强制转换为分类类型
def test_categorical_coerces_numeric(all_parsers):
    # 获取所有解析器对象
    parser = all_parsers
    # 指定b列的数据类型为分类，指定的categories为[1, 2, 3]
    dtype = {"b": CategoricalDtype([1, 2, 3])}

    # 准备测试数据
    data = "b\n1\n1\n2\n3"
    # 期望的数据框架
    expected = DataFrame({"b": Categorical([1, 1, 2, 3])})

    # 使用解析器读取CSV数据，并进行断言比较
    result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)


# 定义测试函数，测试日期时间类型被强制转换为分类类型
def test_categorical_coerces_datetime(all_parsers):
    # 获取所有解析器对象
    parser = all_parsers
    # 准备日期时间索引
    dti = pd.DatetimeIndex(["2017-01-01", "2018-01-01", "2019-01-01"], freq=None)
    # 指定b列的数据类型为分类，指定的categories为日期时间索引
    dtype = {"b": CategoricalDtype(dti)}

    # 准备测试数据
    data = "b\n2017-01-01\n2018-01-01\n2019-01-01"
    # 期望的数据框架
    expected = DataFrame({"b": Categorical(dtype["b"].categories)})

    # 使用解析器读取CSV数据，并进行断言比较
    result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)


# 定义测试函数，测试时间戳类型被强制转换为分类类型
def test_categorical_coerces_timestamp(all_parsers):
    # 获取所有解析器对象
    parser = all_parsers
    # 指定b列的数据类型为分类，指定的categories为时间戳对象列表
    dtype = {"b": CategoricalDtype([Timestamp("2014")])}

    # 准备测试数据
    data = "b\n2014-01-01\n2014-01-01"
    # 期望的数据框架
    expected = DataFrame({"b": Categorical([Timestamp("2014")] * 2)})

    # 使用解析器读取CSV数据，并进行断言比较
    result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)


# 定义测试函数，测试时间增量类型被强制转换为分类类型
def test_categorical_coerces_timedelta(all_parsers):
    # 获取所有解析器对象
    parser = all_parsers
    # 指定b列的数据类型为分类，指定的categories为时间增量对象列表
    dtype = {"b": CategoricalDtype(pd.to_timedelta(["1h", "2h", "3h"]))}

    # 准备测试数据
    data = "b\n1h\n2h\n3h"
    # 期望的数据框架
    expected = DataFrame({"b": Categorical(dtype["b"].categories)})

    # 使用解析器读取CSV数据，并进行断言比较
    result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        "b\nTrue\nFalse\nNA\nFalse",
        "b\ntrue\nfalse\nNA\nfalse",
        "b\nTRUE\nFALSE\nNA\nFALSE",
        "b\nTrue\nFalse\nNA\nFALSE",
    ],
)
# 定义测试函数，测试布尔类型被强制转换为分类类型
def test_categorical_dtype_coerces_boolean(all_parsers, data):
    # see gh-20498
    # 获取所有解析器对象
    parser = all_parsers
    # 指定b列的数据类型为分类，指定的categories为布尔值列表[False, True]
    dtype = {"b": CategoricalDtype([False, True])}
    # 创建一个预期的 DataFrame，包含一列名为 'b'，数据类型为 Categorical，内容为 [True, False, None, False]
    expected = DataFrame({"b": Categorical([True, False, None, False])})

    # 使用 parser 对象读取给定的 CSV 数据（从 StringIO 对象中读取），并指定数据类型为 dtype
    result = parser.read_csv(StringIO(data), dtype=dtype)
    # 使用测试工具 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于测试处理非预期分类数据的情况
def test_categorical_unexpected_categories(all_parsers):
    # 将参数 all_parsers 赋给变量 parser
    parser = all_parsers
    # 定义一个字典 dtype，包含一个键值对，键为 "b"，值为一个指定分类数据类型对象，其中包含分类列表 ["a", "b", "d", "e"]
    dtype = {"b": CategoricalDtype(["a", "b", "d", "e"])}

    # 定义测试数据字符串 data，包含文本 "b\nd\na\nc\nd"，其中包含一个未预期的字符 'c'
    data = "b\nd\na\nc\nd"  # Unexpected c

    # 创建预期结果的 DataFrame 对象 expected，包含一个列 "b"，其值为一个分类对象 Categorical，其中包含列表 ['d', 'a', 'c', 'd']，使用预定义的分类类型 dtype["b"]
    expected = DataFrame({"b": Categorical(list("dacd"), dtype=dtype["b"])})

    # 调用解析器对象 parser 的 read_csv 方法，读取数据并根据预定义的数据类型 dtype 进行解析
    result = parser.read_csv(StringIO(data), dtype=dtype)

    # 使用测试工具 tm.assert_frame_equal 检查结果 result 是否与预期结果 expected 相等
    tm.assert_frame_equal(result, expected)
```