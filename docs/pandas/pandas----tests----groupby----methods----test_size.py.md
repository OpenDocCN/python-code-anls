# `D:\src\scipysrc\pandas\pandas\tests\groupby\methods\test_size.py`

```
# 导入所需的库和模块
import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 库，用于测试

import pandas.util._test_decorators as td  # 导入 pandas 的测试装饰器模块

from pandas import (  # 从 pandas 库中导入以下对象：
    DataFrame,  # DataFrame 对象，用于操作数据帧
    Index,  # Index 对象，用于处理索引
    PeriodIndex,  # PeriodIndex 对象，用于处理时期索引
    Series,  # Series 对象，用于操作序列数据
)
import pandas._testing as tm  # 导入 pandas 测试模块

# 使用 pytest 的 parametrize 装饰器来定义多组参数化测试用例
@pytest.mark.parametrize("by", ["A", "B", ["A", "B"]])
def test_size(df, by):
    grouped = df.groupby(by=by)  # 根据指定的列或列组进行分组
    result = grouped.size()  # 计算每个分组的大小（元素个数）
    # 遍历分组结果，检查每个分组的大小是否与分组的实际元素个数相符
    for key, group in grouped:
        assert result[key] == len(group)


@pytest.mark.parametrize("by", ["A", "B", ["A", "B"]])
def test_size_sort(sort, by):
    # 创建一个具有随机数据的 DataFrame 对象
    df = DataFrame(np.random.default_rng(2).choice(20, (1000, 3)), columns=list("ABC"))
    left = df.groupby(by=by, sort=sort).size()  # 根据指定列进行分组并计算分组大小
    right = df.groupby(by=by, sort=sort)["C"].apply(lambda a: a.shape[0])  # 获取每个分组中特定列的大小
    # 使用 pandas 的断言函数检查两个 Series 对象是否相等，忽略列名检查
    tm.assert_series_equal(left, right, check_names=False)


def test_size_series_dataframe():
    # 创建一个空的 DataFrame 对象
    df = DataFrame(columns=["A", "B"])
    # 创建一个空的 Series 对象
    out = Series(dtype="int64", index=Index([], name="A"))
    # 使用 pandas 的断言函数检查分组后的大小是否与预期的 Series 对象相等
    tm.assert_series_equal(df.groupby("A").size(), out)


def test_size_groupby_all_null():
    # 创建一个包含全部空值分组的 DataFrame 对象
    df = DataFrame({"A": [None, None]})
    result = df.groupby("A").size()  # 计算每个分组的大小
    # 创建一个空的 Series 对象作为预期结果
    expected = Series(dtype="int64", index=Index([], name="A"))
    # 使用 pandas 的断言函数检查分组后的大小是否与预期的 Series 对象相等
    tm.assert_series_equal(result, expected)


def test_size_period_index():
    # 创建一个包含时期索引的 Series 对象
    ser = Series([1], index=PeriodIndex(["2000"], name="A", freq="D"))
    grp = ser.groupby(level="A")  # 根据索引级别进行分组
    result = grp.size()  # 计算每个分组的大小
    # 使用 pandas 的断言函数检查分组后的大小是否与原始 Series 对象相等
    tm.assert_series_equal(result, ser)


def test_size_on_categorical(as_index):
    # 创建一个包含分类数据的 DataFrame 对象
    df = DataFrame([[1, 1], [2, 2]], columns=["A", "B"])
    df["A"] = df["A"].astype("category")  # 将指定列转换为分类类型
    # 根据多列进行分组，并计算分组后的大小
    result = df.groupby(["A", "B"], as_index=as_index, observed=False).size()
    # 创建一个预期的 DataFrame 对象，包含分组后的大小信息
    expected = DataFrame(
        [[1, 1, 1], [1, 2, 0], [2, 1, 0], [2, 2, 1]], columns=["A", "B", "size"]
    )
    expected["A"] = expected["A"].astype("category")
    if as_index:
        expected = expected.set_index(["A", "B"])["size"].rename(None)
    # 使用 pandas 的断言函数检查分组后的结果是否与预期的 DataFrame 对象相等
    tm.assert_equal(result, expected)


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
def test_size_series_masked_type_returns_Int64(dtype):
    # 创建一个指定类型的 Series 对象
    ser = Series([1, 1, 1], index=["a", "a", "b"], dtype=dtype)
    result = ser.groupby(level=0).size()  # 根据索引级别进行分组，并计算每个分组的大小
    # 创建一个预期的 Series 对象，包含分组后的大小信息
    expected = Series([2, 1], dtype="Int64", index=["a", "b"])
    # 使用 pandas 的断言函数检查分组后的结果是否与预期的 Series 对象相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "dtype",
    [
        object,
        pytest.param("string[pyarrow_numpy]", marks=td.skip_if_no("pyarrow")),
        pytest.param("string[pyarrow]", marks=td.skip_if_no("pyarrow")),
    ],
)
def test_size_strings(dtype):
    # 创建一个包含字符串数据的 DataFrame 对象
    df = DataFrame({"a": ["a", "a", "b"], "b": "a"}, dtype=dtype)
    result = df.groupby("a")["b"].size()  # 根据指定列进行分组，并计算每个分组的大小
    # 根据不同的数据类型，选择预期的大小数据类型
    exp_dtype = "Int64" if dtype == "string[pyarrow]" else "int64"
    # 创建一个预期的 Series 对象，包含数值 [2, 1]
    # 指定索引为 ["a", "b"]，索引名称为 "a"，索引数据类型为 dtype 变量的值
    # Series 的名称为 "b"，数据类型为 exp_dtype 变量的值
    expected = Series(
        [2, 1],
        index=Index(["a", "b"], name="a", dtype=dtype),
        name="b",
        dtype=exp_dtype,
    )
    # 使用测试模块 tm 的方法来比较 result 和 expected 两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)
```