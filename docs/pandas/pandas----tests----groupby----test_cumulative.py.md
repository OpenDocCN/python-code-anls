# `D:\src\scipysrc\pandas\pandas\tests\groupby\test_cumulative.py`

```
import numpy as np
import pytest

# 从 pandas.errors 模块导入 UnsupportedFunctionCall 异常类
from pandas.errors import UnsupportedFunctionCall
# 导入 pandas.util._test_decorators 模块，命名为 td
import pandas.util._test_decorators as td

# 导入 pandas 库，并从中导入 DataFrame 和 Series 类
import pandas as pd
from pandas import (
    DataFrame,
    Series,
)
# 导入 pandas._testing 模块，命名为 tm
import pandas._testing as tm


# 定义测试用例的参数化 fixture，用于测试 cummin 和 cummax 方法
@pytest.fixture(
    params=[np.int32, np.int64, np.float32, np.float64, "Int64", "Float64"],
    ids=["np.int32", "np.int64", "np.float32", "np.float64", "Int64", "Float64"],
)
def dtypes_for_minmax(request):
    """
    Fixture of dtypes with min and max values used for testing
    cummin and cummax
    """
    # 获取参数值
    dtype = request.param

    np_type = dtype
    # 处理特殊的 dtype "Int64" 和 "Float64"
    if dtype == "Int64":
        np_type = np.int64
    elif dtype == "Float64":
        np_type = np.float64

    # 根据数据类型获取最小值和最大值
    min_val = (
        np.iinfo(np_type).min
        if np.dtype(np_type).kind == "i"
        else np.finfo(np_type).min
    )
    max_val = (
        np.iinfo(np_type).max
        if np.dtype(np_type).kind == "i"
        else np.finfo(np_type).max
    )

    # 返回元组，包含数据类型及其最小值和最大值
    return (dtype, min_val, max_val)


# 测试函数：验证 groupby 方法结合 cumprod 的结果是否正确
def test_groupby_cumprod():
    # GH 4095
    # 创建 DataFrame 对象
    df = DataFrame({"key": ["b"] * 10, "value": 2})

    # 使用 groupby 和 cumprod 计算累积乘积
    actual = df.groupby("key")["value"].cumprod()
    # 期望的结果，使用 apply 和 lambda 表达式实现
    expected = df.groupby("key", group_keys=False)["value"].apply(lambda x: x.cumprod())
    expected.name = "value"
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(actual, expected)

    # 创建另一个 DataFrame 对象
    df = DataFrame({"key": ["b"] * 100, "value": 2})
    # 将 "value" 列的数据类型转换为 float
    df["value"] = df["value"].astype(float)
    # 再次使用 groupby 和 cumprod 计算累积乘积
    actual = df.groupby("key")["value"].cumprod()
    expected = df.groupby("key", group_keys=False)["value"].apply(lambda x: x.cumprod())
    expected.name = "value"
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(actual, expected)


# 测试函数：验证 groupby 方法结合 cumprod 在溢出情况下的处理是否正确
@pytest.mark.skip_ubsan
def test_groupby_cumprod_overflow():
    # GH#37493 如果溢出，则返回与 numpy 一致的垃圾值
    df = DataFrame({"key": ["b"] * 4, "value": 100_000})
    # 使用 groupby 和 cumprod 计算累积乘积
    actual = df.groupby("key")["value"].cumprod()
    # 期望的结果，手动创建 Series 对象
    expected = Series(
        [100_000, 10_000_000_000, 1_000_000_000_000_000, 7766279631452241920],
        name="value",
    )
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(actual, expected)

    # 使用 apply 和 lambda 表达式实现的 numpy_result
    numpy_result = df.groupby("key", group_keys=False)["value"].apply(
        lambda x: x.cumprod()
    )
    numpy_result.name = "value"
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(actual, numpy_result)


# 测试函数：验证 cumprod 方法对其他列的影响
def test_groupby_cumprod_nan_influences_other_columns():
    # GH#48064
    # 创建 DataFrame 对象
    df = DataFrame(
        {
            "a": 1,
            "b": [1, np.nan, 2],
            "c": [1, 2, 3.0],
        }
    )
    # 使用 cumprod 方法，只计算数值型列，跳过 NaN 值
    result = df.groupby("a").cumprod(numeric_only=True, skipna=False)
    # 期望的结果，手动创建 DataFrame 对象
    expected = DataFrame({"b": [1, np.nan, np.nan], "c": [1, 2, 6.0]})
    # 断言两个 DataFrame 对象是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数：验证 cummin 方法的正确性
def test_cummin(dtypes_for_minmax):
    # 获取测试用例的数据类型
    dtype = dtypes_for_minmax[0]

    # GH 15048
    # 创建基础的 DataFrame 对象
    base_df = DataFrame({"A": [1, 1, 1, 1, 2, 2, 2, 2], "B": [3, 4, 3, 2, 2, 3, 2, 1]})
    # 期望的最小值列表
    expected_mins = [3, 3, 3, 2, 2, 2, 2, 1]

    # 将 DataFrame 对象转换为指定的数据类型
    df = base_df.astype(dtype)
    # 创建期望的 DataFrame 对象
    expected = DataFrame({"B": expected_mins}).astype(dtype)
    # 使用 groupby 和 cummin 计算累积最小值
    result = df.groupby("A").cummin()
    # 断言两个 DataFrame 对象是否相等
    tm.assert_frame_equal(result, expected)
    # 对 DataFrame df 按列"A"进行分组，保留分组键，并对每个分组的列"B"应用累计最小值函数，结果转换为DataFrame
    result = df.groupby("A", group_keys=False).B.apply(lambda x: x.cummin()).to_frame()
    # 使用测试工具 tm 来比较 result 和预期结果 expected 是否相等
    tm.assert_frame_equal(result, expected)
# 使用给定的数据类型和最小值参数
def test_cummin_min_value_for_dtype(dtypes_for_minmax):
    # 获取数据类型
    dtype = dtypes_for_minmax[0]
    # 获取最小值
    min_val = dtypes_for_minmax[1]

    # GH 15048
    # 创建基础数据框架，包含列'A'和列'B'，并初始化数据
    base_df = DataFrame({"A": [1, 1, 1, 1, 2, 2, 2, 2], "B": [3, 4, 3, 2, 2, 3, 2, 1]})
    # 预期的最小值列表
    expected_mins = [3, 3, 3, 2, 2, 2, 2, 1]
    # 创建预期结果的数据框架，仅包含'B'列，并转换为指定数据类型
    expected = DataFrame({"B": expected_mins}).astype(dtype)
    # 将基础数据框架转换为指定数据类型的数据框架
    df = base_df.astype(dtype)
    # 设置特定位置的'B'列数据为最小值
    df.loc[[2, 6], "B"] = min_val
    df.loc[[1, 5], "B"] = min_val + 1
    # 更新预期结果中的数据，确保特定位置的'B'列不被舍入为最小值
    expected.loc[[2, 3, 6, 7], "B"] = min_val
    expected.loc[[1, 5], "B"] = min_val + 1  # 应保持为最小值+1，不被舍入为最小值
    # 对数据框架按'A'列分组，并对'B'列应用累计最小值操作，生成结果
    result = df.groupby("A").cummin()
    # 使用精确比较检查，断言结果与预期一致
    tm.assert_frame_equal(result, expected, check_exact=True)
    # 生成预期结果的另一种方式，对数据框架按'A'列分组，然后对'B'列应用累计最小值操作，转换为数据框架
    expected = (
        df.groupby("A", group_keys=False).B.apply(lambda x: x.cummin()).to_frame()
    )
    # 使用精确比较检查，断言结果与预期一致
    tm.assert_frame_equal(result, expected, check_exact=True)


def test_cummin_nan_in_some_values(dtypes_for_minmax):
    # 明确转换为浮点数，以避免设置nan时的隐式转换
    base_df = DataFrame({"A": [1, 1, 1, 1, 2, 2, 2, 2], "B": [3, 4, 3, 2, 2, 3, 2, 1]})
    base_df = base_df.astype({"B": "float"})
    # 在特定位置设置nan值
    base_df.loc[[0, 2, 4, 6], "B"] = np.nan
    # 创建预期结果的数据框架，包含'B'列，并设置预期值，包括nan
    expected = DataFrame({"B": [np.nan, 4, np.nan, 2, np.nan, 3, np.nan, 1]})
    # 对数据框架按'A'列分组，并对'B'列应用累计最小值操作，生成结果
    result = base_df.groupby("A").cummin()
    # 使用精确比较检查，断言结果与预期一致
    tm.assert_frame_equal(result, expected)
    # 生成预期结果的另一种方式，对数据框架按'A'列分组，然后对'B'列应用累计最小值操作，转换为数据框架
    expected = (
        base_df.groupby("A", group_keys=False).B.apply(lambda x: x.cummin()).to_frame()
    )
    # 使用精确比较检查，断言结果与预期一致
    tm.assert_frame_equal(result, expected)


def test_cummin_datetime():
    # GH 15561
    # 创建包含'a'列和'b'列的数据框架，并将'b'列转换为日期时间类型
    df = DataFrame({"a": [1], "b": pd.to_datetime(["2001"])})
    # 创建预期结果的序列，包含预期的日期时间值
    expected = Series(pd.to_datetime("2001"), index=[0], name="b")
    # 对数据框架按'a'列分组，并对'b'列应用累计最小值操作，生成结果
    result = df.groupby("a")["b"].cummin()
    # 使用序列比较检查，断言结果与预期一致
    tm.assert_series_equal(expected, result)


def test_cummin_getattr_series():
    # GH 15635
    # 创建包含'a'列和'b'列的数据框架
    df = DataFrame({"a": [1, 2, 1], "b": [1, 2, 2]})
    # 对数据框架按'a'列分组，并对'b'列应用累计最小值操作，生成结果
    result = df.groupby("a").b.cummin()
    # 创建预期结果的序列，包含'b'列的预期值
    expected = Series([1, 2, 1], name="b")
    # 使用序列比较检查，断言结果与预期一致
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("method", ["cummin", "cummax"])
@pytest.mark.parametrize("dtype", ["UInt64", "Int64", "Float64", "float", "boolean"])
def test_cummin_max_all_nan_column(method, dtype):
    # 创建包含'A'列和'B'列全为nan的基础数据框架，并将'B'列转换为指定数据类型
    base_df = DataFrame({"A": [1, 1, 1, 1, 2, 2, 2, 2], "B": [np.nan] * 8})
    base_df["B"] = base_df["B"].astype(dtype)
    # 对数据框架按'A'列分组
    grouped = base_df.groupby("A")

    # 创建预期结果的数据框架，包含'B'列全为nan，数据类型为指定类型
    expected = DataFrame({"B": [np.nan] * 8}, dtype=dtype)
    # 对分组后的数据框架应用累计方法(method参数指定)，生成结果
    result = getattr(grouped, method)()
    # 使用精确比较检查，断言结果与预期一致
    tm.assert_frame_equal(expected, result)

    # 对分组后的'B'列应用累计方法(method参数指定)，生成结果，转换为数据框架
    result = getattr(grouped["B"], method)().to_frame()
    # 使用精确比较检查，断言结果与预期一致
    tm.assert_frame_equal(expected, result)


def test_cummax(dtypes_for_minmax):
    # 获取数据类型
    dtype = dtypes_for_minmax[0]

    # GH 15048
    # 创建基础数据框架，包含列'A'和列'B'，并初始化数据
    base_df = DataFrame({"A": [1, 1, 1, 1, 2, 2, 2, 2], "B": [3, 4, 3, 2, 2, 3, 2, 1]})
    # 预期的最大值列表
    expected_maxs = [3, 4, 4, 4, 2, 3, 3, 3]

    # 将基础数据框架转换为指定数据类型的数据框架
    df = base_df.astype(dtype)

    # 创建预期结果的数据框架，仅包含'B'列，并转换为指定数据类型
    expected = DataFrame({"B": expected_maxs}).astype(dtype)
    # 对数据框架按'A'列分组，并对'B'列应用累计最大值操作，生成结果
    result = df.groupby("A").cummax()
    # 使用精确比较检查，断言结果与预期一致
    tm.assert_frame_equal(result, expected)
    # 使用 Pandas DataFrame 对象 df，按列 "A" 分组，并且不生成分组键作为索引，对列 "B" 应用 cummax() 函数，
    # 并将结果转换为 DataFrame，存储在 result 变量中
    result = df.groupby("A", group_keys=False).B.apply(lambda x: x.cummax()).to_frame()
    
    # 使用测试工具函数 tm.assert_frame_equal() 检查 result 变量是否与预期的 DataFrame expected 相等
    tm.assert_frame_equal(result, expected)
# 测试累积最大值和最小值对指定数据类型的影响
def test_cummax_min_value_for_dtype(dtypes_for_minmax):
    # 从数据类型列表中选择第一个作为当前测试的数据类型
    dtype = dtypes_for_minmax[0]
    # 从数据类型列表中选择第三个作为最大值
    max_val = dtypes_for_minmax[2]

    # GH 15048
    # 创建一个基础数据框，包含两列"A"和"B"，分别有指定的整数数据
    base_df = DataFrame({"A": [1, 1, 1, 1, 2, 2, 2, 2], "B": [3, 4, 3, 2, 2, 3, 2, 1]})
    # 预期的最大值列表
    expected_maxs = [3, 4, 4, 4, 2, 3, 3, 3]

    # 将基础数据框转换为指定的数据类型
    df = base_df.astype(dtype)
    # 将"B"列中第2行和第6行的值设为最大值
    df.loc[[2, 6], "B"] = max_val
    # 创建预期结果数据框，只包含"B"列，并转换为指定的数据类型
    expected = DataFrame({"B": expected_maxs}).astype(dtype)
    # 将预期结果数据框中第2行、第3行、第6行和第7行的"B"列值设为最大值
    expected.loc[[2, 3, 6, 7], "B"] = max_val

    # 对"A"列分组并计算累积最大值
    result = df.groupby("A").cummax()
    # 使用测试框架验证结果与预期是否相等
    tm.assert_frame_equal(result, expected)

    # 重新计算预期结果，使用lambda函数应用于"B"列的累积最大值，不分组关键字
    expected = (
        df.groupby("A", group_keys=False).B.apply(lambda x: x.cummax()).to_frame()
    )
    # 使用测试框架验证结果与预期是否相等
    tm.assert_frame_equal(result, expected)


# 测试在部分值中包含NaN的累积最大值
def test_cummax_nan_in_some_values(dtypes_for_minmax):
    # Test nan in some values
    # 显式将整个数据框转换为浮点类型，避免设置NaN时的隐式转换
    base_df = DataFrame({"A": [1, 1, 1, 1, 2, 2, 2, 2], "B": [3, 4, 3, 2, 2, 3, 2, 1]})
    base_df = base_df.astype({"B": "float"})
    # 将"B"列中第0行、第2行、第4行和第6行的值设为NaN
    base_df.loc[[0, 2, 4, 6], "B"] = np.nan
    # 创建预期结果数据框，只包含"B"列，并包含NaN值
    expected = DataFrame({"B": [np.nan, 4, np.nan, 4, np.nan, 3, np.nan, 3]})
    # 对"A"列分组并计算累积最大值
    result = base_df.groupby("A").cummax()
    # 使用测试框架验证结果与预期是否相等
    tm.assert_frame_equal(result, expected)

    # 重新计算预期结果，使用lambda函数应用于"B"列的累积最大值，不分组关键字
    expected = (
        base_df.groupby("A", group_keys=False).B.apply(lambda x: x.cummax()).to_frame()
    )
    # 使用测试框架验证结果与预期是否相等
    tm.assert_frame_equal(result, expected)


# 测试在日期时间数据上的累积最大值
def test_cummax_datetime():
    # GH 15561
    # 创建一个包含单个日期时间列"b"的数据框
    df = DataFrame({"a": [1], "b": pd.to_datetime(["2001"])})
    # 创建预期结果序列，与单个日期时间值相同
    expected = Series(pd.to_datetime("2001"), index=[0], name="b")

    # 对"a"列分组并计算"b"列的累积最大值
    result = df.groupby("a")["b"].cummax()
    # 使用测试框架验证结果与预期是否相等
    tm.assert_series_equal(expected, result)


# 测试在数据框中使用getattr方法获取系列的累积最大值
def test_cummax_getattr_series():
    # GH 15635
    # 创建一个包含两列"a"和"b"的数据框
    df = DataFrame({"a": [1, 2, 1], "b": [2, 1, 1]})
    # 计算"a"列分组下"b"列的累积最大值
    result = df.groupby("a").b.cummax()
    # 创建预期结果序列，包含累积最大值的结果
    expected = Series([2, 1, 2], name="b")
    # 使用测试框架验证结果与预期是否相等
    tm.assert_series_equal(result, expected)


# 测试在int64数据类型的实现边界下的累积最大值
def test_cummax_i8_at_implementation_bound():
    # 以前最小值被视为NPY_NAT+1而不是NPY_NAT，适用于int64数据类型 GH#46382
    # 创建一个包含三列"A"、"B"和"C"的数据框，其中"B"列包含具有特定值的Series对象
    ser = Series([pd.NaT._value + n for n in range(5)])
    df = DataFrame({"A": 1, "B": ser, "C": ser._values.view("M8[ns]")})
    # 对"A"列分组并计算所有列的累积最大值
    gb = df.groupby("A")

    # 计算累积最大值
    res = gb.cummax()
    # 创建预期结果数据框，只包含"B"和"C"列
    exp = df[["B", "C"]]
    # 使用测试框架验证结果与预期是否相等
    tm.assert_frame_equal(res, exp)


# 测试在skipna=False时cummin和cummax方法的使用
@pytest.mark.parametrize("method", ["cummin", "cummax"])
@pytest.mark.parametrize("dtype", ["float", "Int64", "Float64"])
@pytest.mark.parametrize(
    "groups,expected_data",
    [
        ([1, 1, 1], [1, None, None]),
        ([1, 2, 3], [1, None, 2]),
        ([1, 3, 3], [1, None, None]),
    ],
)
def test_cummin_max_skipna(method, dtype, groups, expected_data):
    # GH-34047
    # 创建一个包含单列"a"的数据框，其中包含指定的数据类型
    df = DataFrame({"a": Series([1, None, 2], dtype=dtype)})
    # 创建一个原始数据框的副本
    orig = df.copy()
    # 使用"a"列分组并获取累积最小或最大值
    gb = df.groupby(groups)["a"]

    # 计算累积最小或最大值，根据传入的方法名
    result = getattr(gb, method)(skipna=False)
    # 创建预期结果序列，包含指定数据类型的期望数据
    expected = Series(expected_data, dtype=dtype, name="a")

    # 检查结果与预期是否相等，同时确保未意外更改原始数据框
    tm.assert_frame_equal(df, orig)
    tm.assert_series_equal(result, expected)
# 使用 pytest 模块的 parametrize 装饰器来定义多个参数化测试用例，测试累计最小值和最大值方法
@pytest.mark.parametrize("method", ["cummin", "cummax"])
def test_cummin_max_skipna_multiple_cols(method):
    # 确保 "a" 列中的缺失值不会导致 "b" 列被 NaN 填充
    df = DataFrame({"a": [np.nan, 2.0, 2.0], "b": [2.0, 2.0, 2.0]})
    # 根据多列进行分组，并选择需要应用方法的列
    gb = df.groupby([1, 1, 1])[["a", "b"]]

    # 调用指定的累计方法（cummin 或 cummax），并禁止跳过 NaN 值
    result = getattr(gb, method)(skipna=False)
    # 期望的结果 DataFrame
    expected = DataFrame({"a": [np.nan, np.nan, np.nan], "b": [2.0, 2.0, 2.0]})

    # 断言结果与期望相等
    tm.assert_frame_equal(result, expected)


# 使用 pytest 模块的 parametrize 装饰器来定义多个参数化测试用例，测试累计乘积和累计求和方法
@pytest.mark.parametrize("func", ["cumprod", "cumsum"])
def test_numpy_compat(func):
    # 见 GitHub issue #12811
    df = DataFrame({"A": [1, 2, 1], "B": [1, 2, 3]})
    # 根据列 "A" 进行分组
    g = df.groupby("A")

    msg = "numpy operations are not valid with groupby"

    # 测试在 groupby 对象上调用指定的函数时抛出异常 UnsupportedFunctionCall，并匹配指定的错误信息
    with pytest.raises(UnsupportedFunctionCall, match=msg):
        getattr(g, func)(1, 2, 3)
    with pytest.raises(UnsupportedFunctionCall, match=msg):
        getattr(g, func)(foo=1)


# 使用 td.skip_if_32bit 装饰器来跳过 32 位系统的测试，并使用 pytest.mark.parametrize 装饰器定义多个参数化测试用例
@pytest.mark.parametrize("method", ["cummin", "cummax"])
@pytest.mark.parametrize(
    "dtype,val", [("UInt64", np.iinfo("uint64").max), ("Int64", 2**53 + 1)]
)
def test_nullable_int_not_cast_as_float(method, dtype, val):
    data = [val, pd.NA]
    df = DataFrame({"grp": [1, 1], "b": data}, dtype=dtype)
    # 根据 "grp" 列进行分组
    grouped = df.groupby("grp")

    # 对分组后的数据应用累计方法（cummin 或 cummax）
    result = grouped.transform(method)
    # 期望的结果 DataFrame
    expected = DataFrame({"b": data}, dtype=dtype)

    # 断言结果与期望相等
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，测试 Cython API 的使用情况
def test_cython_api2(as_index):
    # 这里采用了快速的应用路径

    # 测试累计求和方法（GitHub issue #5614）
    # GitHub issue #5755 - 累计求和是一种转换器，应该忽略 as_index 参数
    df = DataFrame([[1, 2, np.nan], [1, np.nan, 9], [3, 4, 9]], columns=["A", "B", "C"])
    # 期望的结果 DataFrame
    expected = DataFrame([[2, np.nan], [np.nan, 9], [4, 9]], columns=["B", "C"])
    # 根据 "A" 列进行分组，并根据传入的 as_index 参数应用累计求和方法
    result = df.groupby("A", as_index=as_index).cumsum()
    # 断言结果与期望相等
    tm.assert_frame_equal(result, expected)
```