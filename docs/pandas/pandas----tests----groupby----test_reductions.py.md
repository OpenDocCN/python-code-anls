# `D:\src\scipysrc\pandas\pandas\tests\groupby\test_reductions.py`

```
# 导入内置模块 builtins
import builtins
# 导入 datetime 模块并使用别名 dt
import datetime as dt
# 从 string 模块中导入 ascii_lowercase 变量
from string import ascii_lowercase

# 导入 numpy 库并使用别名 np
import numpy as np
# 导入 pytest 库
import pytest

# 从 pandas._libs.tslibs 模块中导入 iNaT
from pandas._libs.tslibs import iNaT

# 从 pandas.core.dtypes.common 模块中导入 pandas_dtype
from pandas.core.dtypes.common import pandas_dtype
# 从 pandas.core.dtypes.missing 模块中导入 na_value_for_dtype
from pandas.core.dtypes.missing import na_value_for_dtype

# 导入 pandas 库并使用别名 pd
import pandas as pd
# 从 pandas 中导入特定类和函数
from pandas import (
    DataFrame,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
    isna,
)
# 导入 pandas 测试模块，并使用别名 tm
import pandas._testing as tm
# 从 pandas.util 中导入 _test_decorators，并使用别名 td
from pandas.util import _test_decorators as td


# 使用 pytest 的参数化装饰器定义测试函数，测试不同数据类型的基本聚合操作
@pytest.mark.parametrize("dtype", ["int64", "int32", "float64", "float32"])
def test_basic_aggregations(dtype):
    # 创建一个 Series 对象，数据为 np.arange(9) // 3，索引为 np.arange(9)，指定数据类型为 dtype
    data = Series(np.arange(9) // 3, index=np.arange(9), dtype=dtype)

    # 创建一个索引为 np.arange(9) 的数组 index，并对其进行随机重排
    index = np.arange(9)
    np.random.default_rng(2).shuffle(index)
    data = data.reindex(index)

    # 对数据按照 x // 3 进行分组，不保留分组键
    grouped = data.groupby(lambda x: x // 3, group_keys=False)

    # 遍历分组后的结果，验证每个分组的长度是否为 3
    for k, v in grouped:
        assert len(v) == 3

    # 对分组后的数据应用 np.mean 聚合函数
    agged = grouped.aggregate(np.mean)
    # 验证分组键为 1 的聚合结果是否为 1
    assert agged[1] == 1

    # 使用 grouped.agg(np.mean) 获取聚合结果，并使用测试工具函数验证两个 Series 是否相等
    expected = grouped.agg(np.mean)
    tm.assert_series_equal(agged, expected)  # shorthand
    tm.assert_series_equal(agged, grouped.mean())

    # 对分组后的数据进行求和，并验证结果是否与预期相等
    result = grouped.sum()
    expected = grouped.agg(np.sum)
    if dtype == "int32":
        # 当 dtype 为 "int32" 时，将预期结果转换为 int32 类型，因为 NumPy 的 sum 返回 int64
        expected = expected.astype("int32")
    tm.assert_series_equal(result, expected)

    # 对分组后的数据应用 lambda 函数进行复杂的转换操作，并验证结果是否与预期相等
    expected = grouped.apply(lambda x: x * x.sum())
    transformed = grouped.transform(lambda x: x * x.sum())
    # 验证索引为 7 的结果是否为 12
    assert transformed[7] == 12
    tm.assert_series_equal(transformed, expected)

    # 根据数据值进行分组，并应用 np.mean 聚合函数，并验证结果是否与预期相等，不检查索引类型
    value_grouped = data.groupby(data)
    result = value_grouped.aggregate(np.mean)
    tm.assert_series_equal(result, agged, check_index_type=False)

    # 复杂聚合操作，应用多个聚合函数 np.mean 和 np.std，并验证是否抛出预期的异常
    agged = grouped.aggregate([np.mean, np.std])

    msg = r"nested renamer is not supported"
    # 使用 pytest 的 assert 语法验证是否抛出指定类型的异常，并匹配指定的消息
    with pytest.raises(pd.errors.SpecificationError, match=msg):
        grouped.aggregate({"one": np.mean, "two": np.std})

    # 边界情况，应用 lambda 函数，验证是否抛出预期的异常类型 Exception
    msg = "Must produce aggregated value"
    with pytest.raises(Exception, match=msg):
        grouped.aggregate(lambda x: x * 2)


# 使用 pytest 的参数化装饰器定义测试函数，测试布尔类型的分组聚合操作
@pytest.mark.parametrize(
    "vals",
    [
        ["foo", "bar", "baz"],
        ["foo", "", ""],
        ["", "", ""],
        [1, 2, 3],
        [1, 0, 0],
        [0, 0, 0],
        [1.0, 2.0, 3.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [True, True, True],
        [True, False, False],
        [False, False, False],
        [np.nan, np.nan, np.nan],
    ],
)
def test_groupby_bool_aggs(skipna, all_boolean_reductions, vals):
    # 创建 DataFrame 对象，包含键为 'key' 的列和值为 vals 的列 'val'
    df = DataFrame({"key": ["a"] * 3 + ["b"] * 3, "val": vals * 2})

    # 使用 Python 内置函数 getattr 获取 all_boolean_reductions 指定的函数，并计算期望值
    exp = getattr(builtins, all_boolean_reductions)(vals)

    # 处理缺失数据的边界情况，如果 skipna 为 True，且 vals 全部为缺失值，并且 all_boolean_reductions 为 'any'，则期望值为 False
    if skipna and all(isna(vals)) and all_boolean_reductions == "any":
        exp = False

    # 创建期望的 DataFrame，列为 ['val']，索引为 ['a', 'b']
    expected = DataFrame(
        [exp] * 2, columns=["val"], index=pd.Index(["a", "b"], name="key")
    )
    # 使用 getattr 函数获取 DataFrame 按 "key" 列分组后的对象，并调用指定的 all_boolean_reductions 函数
    result = getattr(df.groupby("key"), all_boolean_reductions)(skipna=skipna)
    # 断言结果 DataFrame 和期望的结果相等
    tm.assert_frame_equal(result, expected)
def test_any():
    # 创建一个 DataFrame 包含不同类型的数据
    df = DataFrame(
        [[1, 2, "foo"], [1, np.nan, "bar"], [3, np.nan, "baz"]],
        columns=["A", "B", "C"],
    )
    # 创建预期的 DataFrame，用于测试预期结果
    expected = DataFrame(
        [[True, True], [False, True]], columns=["B", "C"], index=[1, 3]
    )
    # 设置预期 DataFrame 的索引名称
    expected.index.name = "A"
    # 对 DataFrame 进行按列 A 分组并应用 any() 聚合函数
    result = df.groupby("A").any()
    # 使用测试框架检查结果是否与预期一致
    tm.assert_frame_equal(result, expected)


def test_bool_aggs_dup_column_labels(all_boolean_reductions):
    # GH#21668
    # 创建具有重复列标签的 DataFrame
    df = DataFrame([[True, True]], columns=["a", "a"])
    # 根据列 [0] 对 DataFrame 进行分组
    grp_by = df.groupby([0])
    # 对分组后的结果应用指定的所有布尔缩减操作
    result = getattr(grp_by, all_boolean_reductions)()
    # 设置预期结果为重新设置轴的 DataFrame
    expected = df.set_axis(np.array([0]))
    # 使用测试框架检查结果是否与预期一致
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        [False, False, False],
        [True, True, True],
        [pd.NA, pd.NA, pd.NA],
        [False, pd.NA, False],
        [True, pd.NA, True],
        [True, pd.NA, False],
    ],
)
def test_masked_kleene_logic(all_boolean_reductions, skipna, data):
    # GH#37506
    # 创建布尔类型的 Series
    ser = Series(data, dtype="boolean")
    # 计算整个系列上的预期数据
    expected_data = getattr(ser, all_boolean_reductions)(skipna=skipna)
    expected = Series(expected_data, index=np.array([0]), dtype="boolean")
    # 对 Series 根据多个列 [0, 0, 0] 进行分组并应用布尔缩减操作
    result = ser.groupby([0, 0, 0]).agg(all_boolean_reductions, skipna=skipna)
    # 使用测试框架检查结果是否与预期一致
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "dtype1,dtype2,exp_col1,exp_col2",
    [
        (
            "float",
            "Float64",
            np.array([True], dtype=bool),
            pd.array([pd.NA], dtype="boolean"),
        ),
        (
            "Int64",
            "float",
            pd.array([pd.NA], dtype="boolean"),
            np.array([True], dtype=bool),
        ),
        (
            "Int64",
            "Int64",
            pd.array([pd.NA], dtype="boolean"),
            pd.array([pd.NA], dtype="boolean"),
        ),
        (
            "Float64",
            "boolean",
            pd.array([pd.NA], dtype="boolean"),
            pd.array([pd.NA], dtype="boolean"),
        ),
    ],
)
def test_masked_mixed_types(dtype1, dtype2, exp_col1, exp_col2):
    # GH#37506
    # 创建包含混合数据类型的 DataFrame
    data = [1.0, np.nan]
    df = DataFrame(
        {"col1": pd.array(data, dtype=dtype1), "col2": pd.array(data, dtype=dtype2)}
    )
    # 对 DataFrame 根据列 [1, 1] 进行分组并应用 "all" 布尔缩减操作
    result = df.groupby([1, 1]).agg("all", skipna=False)
    # 创建预期的 DataFrame
    expected = DataFrame({"col1": exp_col1, "col2": exp_col2}, index=np.array([1]))
    # 使用测试框架检查结果是否与预期一致
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
def test_masked_bool_aggs_skipna(
    all_boolean_reductions, dtype, skipna, frame_or_series
):
    # GH#40585
    # 创建具有特定数据类型的 DataFrame 或 Series
    obj = frame_or_series([pd.NA, 1], dtype=dtype)
    expected_res = True
    if not skipna and all_boolean_reductions == "all":
        expected_res = pd.NA
    # 创建预期结果的 DataFrame 或 Series
    expected = frame_or_series([expected_res], index=np.array([1]), dtype="boolean")
    # 使用给定的对象 `obj` 进行分组操作，分组依据为列表 `[1, 1]`，并对每个分组应用指定的布尔函数归约操作 `all_boolean_reductions`，跳过 NaN 值（如果设置为 `skipna=True`）。
    result = obj.groupby([1, 1]).agg(all_boolean_reductions, skipna=skipna)
    # 使用测试模块中的断言方法 `assert_equal` 检查 `result` 是否等于预期的结果 `expected`。
    tm.assert_equal(result, expected)
# 使用 pytest.mark.parametrize 装饰器，为 test_object_type_missing_vals 函数定义多组参数化测试用例
@pytest.mark.parametrize(
    "bool_agg_func,data,expected_res",
    [
        ("any", [pd.NA, np.nan], False),                  # 情况1：使用 'any' 聚合函数，数据包含 NA 和 np.nan，期望结果为 False
        ("any", [pd.NA, 1, np.nan], True),               # 情况2：使用 'any' 聚合函数，数据包含 NA, 1 和 np.nan，期望结果为 True
        ("all", [pd.NA, pd.NaT], True),                  # 情况3：使用 'all' 聚合函数，数据包含 NA 和 NaT，期望结果为 True
        ("all", [pd.NA, False, pd.NaT], False),          # 情况4：使用 'all' 聚合函数，数据包含 NA, False 和 NaT，期望结果为 False
    ],
)
def test_object_type_missing_vals(bool_agg_func, data, expected_res, frame_or_series):
    # GH#37501
    # 使用给定的数据创建对象 obj，并指定对象类型为 'object'
    obj = frame_or_series(data, dtype=object)
    # 对 obj 执行分组操作，并应用指定的聚合函数 bool_agg_func
    result = obj.groupby([1] * len(data)).agg(bool_agg_func)
    # 创建预期结果对象 expected，其值为 expected_res，索引为 [1]，数据类型为 'bool'
    expected = frame_or_series([expected_res], index=np.array([1]), dtype="bool")
    # 断言 result 和 expected 是否相等
    tm.assert_equal(result, expected)


# 定义测试函数 test_object_NA_raises_with_skipna_false，验证在 skipna=False 时处理 NA 引发 TypeError 异常
def test_object_NA_raises_with_skipna_false(all_boolean_reductions):
    # GH#37501
    # 创建包含一个 NA 值的 Series 对象 ser，数据类型为 'object'
    ser = Series([pd.NA], dtype=object)
    # 使用 pytest.raises 检测是否引发 TypeError 异常，异常信息包含 "boolean value of NA is ambiguous"
    with pytest.raises(TypeError, match="boolean value of NA is ambiguous"):
        # 对 ser 执行分组操作，并应用 all_boolean_reductions 函数，skipna=False
        ser.groupby([1]).agg(all_boolean_reductions, skipna=False)


# 定义测试函数 test_empty，验证对空数据的处理
def test_empty(frame_or_series, all_boolean_reductions):
    # GH 45231
    # 根据 frame_or_series 的类型选择不同的参数创建对象 kwargs
    kwargs = {"columns": ["a"]} if frame_or_series is DataFrame else {"name": "a"}
    # 使用 kwargs 创建对象 obj，数据类型为 'object'
    obj = frame_or_series(**kwargs, dtype=object)
    # 对 obj 的分组结果应用 all_boolean_reductions 函数
    result = getattr(obj.groupby(obj.index), all_boolean_reductions)()
    # 创建预期结果对象 expected，与 obj 类型相同，数据类型为 'bool'
    expected = frame_or_series(**kwargs, dtype=bool)
    # 断言 result 和 expected 是否相等
    tm.assert_equal(result, expected)


# 使用 pytest.mark.parametrize 装饰器，为 test_idxmin_idxmax_extremes 函数定义参数化测试用例
@pytest.mark.parametrize("how", ["idxmin", "idxmax"])
def test_idxmin_idxmax_extremes(how, any_real_numpy_dtype):
    # GH#57040
    # 如果 any_real_numpy_dtype 是 int 或 float 类型，则无需测试
    if any_real_numpy_dtype is int or any_real_numpy_dtype is float:
        # 不执行后续操作
        return
    # 根据 any_real_numpy_dtype 类型选择使用 np.iinfo 或 np.finfo 函数
    info = np.iinfo if "int" in any_real_numpy_dtype else np.finfo
    # 获取 any_real_numpy_dtype 类型的最小值和最大值
    min_value = info(any_real_numpy_dtype).min
    max_value = info(any_real_numpy_dtype).max
    # 创建 DataFrame 对象 df，包含两列 'a' 和 'b'
    df = DataFrame(
        {"a": [2, 1, 1, 2], "b": [min_value, max_value, max_value, min_value]},
        dtype=any_real_numpy_dtype,
    )
    # 对 df 执行分组操作，按列 'a' 进行分组
    gb = df.groupby("a")
    # 对分组对象 gb 应用 how 参数指定的聚合函数
    result = getattr(gb, how)()
    # 创建预期结果 DataFrame 对象 expected，包含列 'b'，索引为 [1, 2]，数据类型为 any_real_numpy_dtype
    expected = DataFrame(
        {"b": [1, 0]}, index=pd.Index([1, 2], name="a", dtype=any_real_numpy_dtype)
    )
    # 断言 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 使用 pytest.mark.parametrize 装饰器，为 test_idxmin_idxmax_extremes_skipna 函数定义参数化测试用例
@pytest.mark.parametrize("how", ["idxmin", "idxmax"])
def test_idxmin_idxmax_extremes_skipna(skipna, how, float_numpy_dtype):
    # GH#57040
    # 获取 float_numpy_dtype 类型的最小值和最大值
    min_value = np.finfo(float_numpy_dtype).min
    max_value = np.finfo(float_numpy_dtype).max
    # 创建 DataFrame 对象 df，包含两列 'a' 和 'b'
    df = DataFrame(
        {
            "a": Series(np.repeat(range(1, 6), repeats=2), dtype="intp"),
            "b": Series(
                [
                    np.nan,
                    min_value,
                    np.nan,
                    max_value,
                    min_value,
                    np.nan,
                    max_value,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                dtype=float_numpy_dtype,
            ),
        },
    )
    # 对 df 执行分组操作，按列 'a' 进行分组
    gb = df.groupby("a")

    # 如果 skipna=False，则预期引发 ValueError 异常
    if not skipna:
        msg = f"DataFrameGroupBy.{how} with skipna=False"
        with pytest.raises(ValueError, match=msg):
            getattr(gb, how)(skipna=skipna)
        return
    # 使用 getattr 函数从对象 gb 中获取名为 how 的方法，并传入 skipna 参数进行调用，得到计算结果
    result = getattr(gb, how)(skipna=skipna)
    # 创建一个预期的 DataFrame 对象，其中包含一列名为 'b' 的数据，和索引为整数范围 [1, 5] 的列 'a'
    expected = DataFrame(
        {"b": [1, 3, 4, 6, np.nan]}, index=pd.Index(range(1, 6), name="a", dtype="intp")
    )
    # 使用 tm.assert_frame_equal 函数比较计算结果 result 和预期结果 expected 是否相等
    tm.assert_frame_equal(result, expected)
# 使用 pytest.mark.parametrize 装饰器，定义多个参数化测试用例
@pytest.mark.parametrize(
    "func, values",
    [
        ("idxmin", {"c_int": [0, 2], "c_float": [1, 3], "c_date": [1, 2]}),
        ("idxmax", {"c_int": [1, 3], "c_float": [0, 2], "c_date": [0, 3]}),
    ],
)
# 使用 pytest.mark.parametrize 装饰器，定义单个参数化测试用例
@pytest.mark.parametrize("numeric_only", [True, False])
def test_idxmin_idxmax_returns_int_types(func, values, numeric_only):
    # GH 25444
    # 创建一个 DataFrame 对象，包含不同类型的数据列
    df = DataFrame(
        {
            "name": ["A", "A", "B", "B"],
            "c_int": [1, 2, 3, 4],
            "c_float": [4.02, 3.03, 2.04, 1.05],
            "c_date": ["2019", "2018", "2016", "2017"],
        }
    )
    # 将 'c_date' 列转换为 datetime 类型
    df["c_date"] = pd.to_datetime(df["c_date"])
    # 在 'c_date' 列上应用时区 'US/Pacific'
    df["c_date_tz"] = df["c_date"].dt.tz_localize("US/Pacific")
    # 计算时间差并存储在 'c_timedelta' 列
    df["c_timedelta"] = df["c_date"] - df["c_date"].iloc[0]
    # 将 'c_date' 列转换为周期 'W' 并存储在 'c_period' 列
    df["c_period"] = df["c_date"].dt.to_period("W")
    # 将 'c_int' 列转换为 Int64 类型并存储在 'c_Integer' 列
    df["c_Integer"] = df["c_int"].astype("Int64")
    # 将 'c_float' 列转换为 Float64 类型并存储在 'c_Floating' 列
    df["c_Floating"] = df["c_float"].astype("Float64")

    # 调用 getattr 方法获取指定函数（如 idxmin 或 idxmax）的结果
    result = getattr(df.groupby("name"), func)(numeric_only=numeric_only)

    # 创建期望结果的 DataFrame
    expected = DataFrame(values, index=pd.Index(["A", "B"], name="name"))
    # 如果 numeric_only 为 True，则移除 'c_date' 列
    if numeric_only:
        expected = expected.drop(columns=["c_date"])
    # 否则，将 'c_date' 列重命名为 'c_date_tz', 'c_timedelta', 'c_period'
    else:
        expected["c_date_tz"] = expected["c_date"]
        expected["c_timedelta"] = expected["c_date"]
        expected["c_period"] = expected["c_date"]
    # 将 'c_int' 列重命名为 'c_Integer'，'c_float' 列重命名为 'c_Floating'
    expected["c_Integer"] = expected["c_int"]
    expected["c_Floating"] = expected["c_float"]

    # 使用 assert_frame_equal 函数比较实际结果和期望结果
    tm.assert_frame_equal(result, expected)


# 使用 pytest.mark.parametrize 装饰器，定义多个参数化测试用例
@pytest.mark.parametrize(
    "data",
    [
        (
            Timestamp("2011-01-15 12:50:28.502376"),
            Timestamp("2011-01-20 12:50:28.593448"),
        ),
        (24650000000000001, 24650000000000002),
    ],
)
# 使用 pytest.mark.parametrize 装饰器，定义单个参数化测试用例
@pytest.mark.parametrize("method", ["count", "min", "max", "first", "last"])
def test_groupby_non_arithmetic_agg_int_like_precision(method, data):
    # GH#6620, GH#9311
    # 创建一个 DataFrame 对象，包含 'a' 列和 'b' 列
    df = DataFrame({"a": [1, 1], "b": data})

    # 根据 'a' 列进行分组
    grouped = df.groupby("a")
    # 调用 getattr 方法获取指定函数（如 count、min、max 等）的结果
    result = getattr(grouped, method)()
    # 根据不同的聚合函数，设置期望值
    if method == "count":
        expected_value = 2
    elif method == "first":
        expected_value = data[0]
    elif method == "last":
        expected_value = data[1]
    else:
        expected_value = getattr(df["b"], method)()
    # 创建期望结果的 DataFrame
    expected = DataFrame({"b": [expected_value]}, index=pd.Index([1], name="a"))

    # 使用 assert_frame_equal 函数比较实际结果和期望结果
    tm.assert_frame_equal(result, expected)


# 使用 pytest.mark.parametrize 装饰器，定义单个参数化测试用例
@pytest.mark.parametrize("how", ["first", "last"])
def test_first_last_skipna(any_real_nullable_dtype, sort, skipna, how):
    # GH#57019
    # 获取适合 'any_real_nullable_dtype' 的 NA 值
    na_value = na_value_for_dtype(pandas_dtype(any_real_nullable_dtype))
    # 创建一个 DataFrame 对象，包含 'a'、'b'、'c' 列，并设置数据类型为 'any_real_nullable_dtype'
    df = DataFrame(
        {
            "a": [2, 1, 1, 2, 3, 3],
            "b": [na_value, 3.0, na_value, 4.0, np.nan, np.nan],
            "c": [na_value, 3.0, na_value, 4.0, np.nan, np.nan],
        },
        dtype=any_real_nullable_dtype,
    )
    # 根据 'a' 列进行分组，并根据 'sort' 参数进行排序
    gb = df.groupby("a", sort=sort)
    # 使用 getattr 方法获取指定函数（如 first 或 last）的方法
    method = getattr(gb, how)
    # 调用指定方法，根据 'skipna' 参数进行计算
    result = method(skipna=skipna)
    # 根据传入的参数 `how` 和 `skipna` 选择对应的 ilocs 列表，用于后续的数据索引
    ilocs = {
        ("first", True): [3, 1, 4],
        ("first", False): [0, 1, 4],
        ("last", True): [3, 1, 5],
        ("last", False): [3, 2, 5],
    }[how, skipna]
    
    # 根据 ilocs 列表从数据框 `df` 中提取相应的行，并将列 "a" 设置为索引
    expected = df.iloc[ilocs].set_index("a")
    
    # 如果 sort 参数为真，则对期望的数据框按索引排序
    if sort:
        expected = expected.sort_index()
    
    # 使用测试工具 `tm` 来比较 `result` 和 `expected` 的数据框是否相等
    tm.assert_frame_equal(result, expected)
def test_groupby_mean_no_overflow():
    # 回归测试 (#22487)，验证是否修复了整数溢出的问题
    df = DataFrame(
        {
            "user": ["A", "A", "A", "A", "A"],
            "connections": [4970, 4749, 4719, 4704, 18446744073699999744],
            # 创建包含用户和连接数的DataFrame
        }
    )
    assert df.groupby("user")["connections"].mean()["A"] == 3689348814740003840
    # 断言检查分组后用户"A"的连接数均值是否等于特定的预期值


def test_mean_on_timedelta():
    # GitHub问题 #17382，测试时间间隔数据的均值计算
    df = DataFrame({"time": pd.to_timedelta(range(10)), "cat": ["A", "B"] * 5})
    # 创建包含时间间隔和类别的DataFrame
    result = df.groupby("cat")["time"].mean()
    # 计算按类别分组后时间间隔的均值
    expected = Series(
        pd.to_timedelta([4, 5]), name="time", index=pd.Index(["A", "B"], name="cat")
    )
    tm.assert_series_equal(result, expected)
    # 断言检查计算结果与预期结果是否一致


def test_cython_median():
    arr = np.random.default_rng(2).standard_normal(1000)
    arr[::2] = np.nan
    df = DataFrame(arr)
    # 创建包含随机数和NaN值的DataFrame

    labels = np.random.default_rng(2).integers(0, 50, size=1000).astype(float)
    labels[::17] = np.nan
    # 创建包含随机标签和NaN值的数组

    result = df.groupby(labels).median()
    # 按标签分组计算每组的中位数
    exp = df.groupby(labels).agg(np.nanmedian)
    tm.assert_frame_equal(result, exp)
    # 断言检查计算结果与使用np.nanmedian聚合的结果是否一致

    df = DataFrame(np.random.default_rng(2).standard_normal((1000, 5)))
    rs = df.groupby(labels).agg(np.median)
    xp = df.groupby(labels).median()
    tm.assert_frame_equal(rs, xp)
    # 重新测试，检查使用不同方法计算中位数的结果是否一致


def test_median_empty_bins(observed):
    df = DataFrame(np.random.default_rng(2).integers(0, 44, 500))
    # 创建包含随机整数的DataFrame

    grps = range(0, 55, 5)
    bins = pd.cut(df[0], grps)
    # 使用cut函数将数据分组为区间

    result = df.groupby(bins, observed=observed).median()
    # 按照分组后的区间计算每组的中位数
    expected = df.groupby(bins, observed=observed).agg(lambda x: x.median())
    tm.assert_frame_equal(result, expected)
    # 断言检查计算结果与使用lambda函数计算的中位数是否一致


def test_max_min_non_numeric():
    # GitHub问题 #2700，测试在包含非数值列时的最大最小值计算
    aa = DataFrame({"nn": [11, 11, 22, 22], "ii": [1, 2, 3, 4], "ss": 4 * ["mama"]})
    # 创建包含数值和非数值列的DataFrame

    result = aa.groupby("nn").max()
    assert "ss" in result
    # 按nn列分组计算最大值，并断言检查结果是否包含非数值列"ss"

    result = aa.groupby("nn").max(numeric_only=False)
    assert "ss" in result
    # 按nn列分组计算最大值，包括非数值列，并断言检查结果是否包含非数值列"ss"

    result = aa.groupby("nn").min()
    assert "ss" in result
    # 按nn列分组计算最小值，并断言检查结果是否包含非数值列"ss"

    result = aa.groupby("nn").min(numeric_only=False)
    assert "ss" in result
    # 按nn列分组计算最小值，包括非数值列，并断言检查结果是否包含非数值列"ss"


def test_max_min_object_multiple_columns():
    # GitHub问题 #41111，测试在包含对象类型多列时的最大最小值计算
    df = DataFrame(
        {
            "A": [1, 1, 2, 2, 3],
            "B": [1, "foo", 2, "bar", False],
            "C": ["a", "b", "c", "d", "e"],
        }
    )
    df._consolidate_inplace()  # 应该已经是合并的，但是再次检查
    assert len(df._mgr.blocks) == 2
    # 创建包含数值和对象类型多列的DataFrame，并确保数据块已合并

    gb = df.groupby("A")

    result = gb[["C"]].max()
    # 按A列分组计算列C的最大值
    ei = pd.Index([1, 2, 3], name="A")
    expected = DataFrame({"C": ["b", "d", "e"]}, index=ei)
    tm.assert_frame_equal(result, expected)
    # 断言检查计算结果与预期结果是否一致

    result = gb[["C"]].min()
    # 按A列分组计算列C的最小值
    ei = pd.Index([1, 2, 3], name="A")
    expected = DataFrame({"C": ["a", "c", "e"]}, index=ei)
    tm.assert_frame_equal(result, expected)
    # 断言检查计算结果与预期结果是否一致


def test_min_date_with_nans():
    # GitHub问题 #26321
    # 使用 pandas 库将字符串日期转换为日期对象，只保留日期部分
    dates = pd.to_datetime(
        Series(["2019-05-09", "2019-05-09", "2019-05-09"]), format="%Y-%m-%d"
    ).dt.date
    
    # 创建一个 DataFrame 对象，包含三列数据：a 列包含 NaN 值和字符串 "1"，b 列包含整数 0, 1, 1，c 列为之前转换得到的日期对象
    df = DataFrame({"a": [np.nan, "1", np.nan], "b": [0, 1, 1], "c": dates})
    
    # 对 DataFrame 进行按列 'b' 分组，并取出 'c' 列的最小日期值，结果不作为索引
    result = df.groupby("b", as_index=False)["c"].min()["c"]
    
    # 创建一个预期的 Series 对象，包含两个日期值，与之前的日期格式化保持一致
    expected = pd.to_datetime(
        Series(["2019-05-09", "2019-05-09"], name="c"), format="%Y-%m-%d"
    ).dt.date
    
    # 使用测试工具库 tm 检查 result 和 expected 的 Series 对象是否相等
    tm.assert_series_equal(result, expected)
    
    # 再次对 DataFrame 按列 'b' 分组，并取出 'c' 列的最小日期值，结果作为索引
    result = df.groupby("b")["c"].min()
    
    # 设置预期结果的索引名称为 'b'
    expected.index.name = "b"
    
    # 使用测试工具库 tm 检查 result 和 expected 的 Series 对象是否相等
    tm.assert_series_equal(result, expected)
def test_max_inat():
    # GH#40767 dont interpret iNaT as NaN
    # 创建包含整数和iNaT值的Series对象
    ser = Series([1, iNaT])
    # 创建一个包含整数值的NumPy数组
    key = np.array([1, 1], dtype=np.int64)
    # 使用key对ser进行分组
    gb = ser.groupby(key)

    # 计算每个组的最大值，要求至少有2个非缺失值
    result = gb.max(min_count=2)
    # 期望结果是一个Series对象，指定了预期的整数值
    expected = Series({1: 1}, dtype=np.int64)
    # 断言结果与预期相等
    tm.assert_series_equal(result, expected, check_exact=True)

    # 计算每个组的最小值，要求至少有2个非缺失值
    result = gb.min(min_count=2)
    # 期望结果是一个Series对象，指定了预期的iNaT值
    expected = Series({1: iNaT}, dtype=np.int64)
    # 断言结果与预期相等
    tm.assert_series_equal(result, expected, check_exact=True)

    # 因为条目数量不足，结果被掩盖为NaN
    result = gb.min(min_count=3)
    # 期望结果是一个Series对象，含有NaN值
    expected = Series({1: np.nan})
    # 断言结果与预期相等
    tm.assert_series_equal(result, expected, check_exact=True)


def test_max_inat_not_all_na():
    # GH#40767 dont interpret iNaT as NaN

    # 确保iNaT+1不会被四舍五入成iNaT
    ser = Series([1, iNaT, 2, iNaT + 1])
    # 使用列表对ser进行分组
    gb = ser.groupby([1, 2, 3, 3])
    # 计算每个组的最小值，要求至少有2个非缺失值
    result = gb.min(min_count=2)

    # 注意：在转换为float64时，iNaT + 1映射到iNaT，即会有信息损失
    expected = Series({1: np.nan, 2: np.nan, 3: iNaT + 1})
    expected.index = expected.index.astype(int)
    # 断言结果与预期相等
    tm.assert_series_equal(result, expected, check_exact=True)


@pytest.mark.parametrize("func", ["min", "max"])
def test_groupby_aggregate_period_column(func):
    # GH 31471
    # 创建分组键和时间周期范围
    groups = [1, 2]
    periods = pd.period_range("2020", periods=2, freq="Y")
    # 创建包含分组键和时间周期的DataFrame对象
    df = DataFrame({"a": groups, "b": periods})

    # 使用getattr调用相应的聚合函数（最小值或最大值）
    result = getattr(df.groupby("a")["b"], func)()
    # 创建预期结果的索引
    idx = pd.Index([1, 2], name="a")
    # 创建预期的Series对象
    expected = Series(periods, index=idx, name="b")

    # 断言结果与预期相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", ["min", "max"])
def test_groupby_aggregate_period_frame(func):
    # GH 31471
    # 创建分组键和时间周期范围
    groups = [1, 2]
    periods = pd.period_range("2020", periods=2, freq="Y")
    # 创建包含分组键和时间周期的DataFrame对象
    df = DataFrame({"a": groups, "b": periods})

    # 使用getattr调用相应的聚合函数（最小值或最大值）
    result = getattr(df.groupby("a"), func)()
    # 创建预期结果的索引
    idx = pd.Index([1, 2], name="a")
    # 创建预期的DataFrame对象
    expected = DataFrame({"b": periods}, index=idx)

    # 断言结果与预期相等
    tm.assert_frame_equal(result, expected)


def test_aggregate_numeric_object_dtype():
    # https://github.com/pandas-dev/pandas/issues/39329
    # 简化的情况：多个对象列其中一个全为NaN -> 推断为float时会被分割
    # 创建包含键、字符和NaN值的DataFrame对象
    df = DataFrame(
        {"key": ["A", "A", "B", "B"], "col1": list("abcd"), "col2": [np.nan] * 4},
    ).astype(object)
    # 对键进行分组并计算每组的最小值
    result = df.groupby("key").min()
    # 创建预期的DataFrame对象，设置索引和对象类型
    expected = (
        DataFrame(
            {"key": ["A", "B"], "col1": ["a", "c"], "col2": [np.nan, np.nan]},
        )
        .set_index("key")
        .astype(object)
    )
    # 断言结果与预期相等
    tm.assert_frame_equal(result, expected)

    # 同样的情况，但是使用数字
    df = DataFrame(
        {"key": ["A", "A", "B", "B"], "col1": list("abcd"), "col2": range(4)},
    ).astype(object)
    # 对键进行分组并计算每组的最小值
    result = df.groupby("key").min()
    # 创建预期的DataFrame对象，设置索引和对象类型
    expected = (
        DataFrame({"key": ["A", "B"], "col1": ["a", "c"], "col2": [0, 2]})
        .set_index("key")
        .astype(object)
    )
    # 断言结果与预期相等
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于测试对分类数据进行聚合操作时的索引丢失情况
def test_aggregate_categorical_lost_index(func: str):
    # 创建一个有序的分类序列，包含单个值 'b'
    ds = Series(["b"], dtype="category").cat.as_ordered()
    # 创建一个数据框，包含两列'A'和'B'，其中'A'列为1997，'B'列使用上面定义的分类序列
    df = DataFrame({"A": [1997], "B": ds})
    # 对数据框按列'A'进行分组，对'B'列应用传入的聚合函数'func'
    result = df.groupby("A").agg({"B": func})
    # 创建一个预期结果的数据框，包含单列'B'，值为 ['b']，索引为包含1997的索引
    expected = DataFrame({"B": ["b"]}, index=pd.Index([1997], name="A"))

    # 确保'B'列保持有序分类的数据类型
    expected["B"] = expected["B"].astype(ds.dtype)

    # 使用测试工具函数，比较实际结果和预期结果的数据框是否相等
    tm.assert_frame_equal(result, expected)


# 使用参数化测试，测试不同数据类型（'Int64', 'Int32', 'Float64', 'Float32', 'boolean'）的分组最小值和最大值计算
@pytest.mark.parametrize("dtype", ["Int64", "Int32", "Float64", "Float32", "boolean"])
def test_groupby_min_max_nullable(dtype):
    # 根据数据类型选择不同的时间戳
    if dtype == "Int64":
        ts = 1618556707013635762  # 避免精度损失
    elif dtype == "boolean":
        ts = 0
    else:
        ts = 4.0

    # 创建一个数据框，包含'id'列和'ts'列，其中'ts'列使用指定的数据类型
    df = DataFrame({"id": [2, 2], "ts": [ts, ts + 1]})
    df["ts"] = df["ts"].astype(dtype)

    # 根据'id'列对数据框进行分组
    gb = df.groupby("id")

    # 计算分组后每组的最小值
    result = gb.min()
    # 创建预期的最小值数据框，只包含第一行，索引为'id'
    expected = df.iloc[:1].set_index("id")
    tm.assert_frame_equal(result, expected)

    # 计算分组后每组的最大值
    res_max = gb.max()
    # 创建预期的最大值数据框，只包含第二行，索引为'id'
    expected_max = df.iloc[1:].set_index("id")
    tm.assert_frame_equal(res_max, expected_max)

    # 使用指定的最小值计算非空值
    result2 = gb.min(min_count=3)
    # 创建预期的非空最小值数据框，只包含一列'ts'，值为pd.NA，索引与expected相同，数据类型为dtype
    expected2 = DataFrame({"ts": [pd.NA]}, index=expected.index, dtype=dtype)
    tm.assert_frame_equal(result2, expected2)

    # 使用指定的最大值计算非空值
    res_max2 = gb.max(min_count=3)
    tm.assert_frame_equal(res_max2, expected2)

    # 包含NA值的情况
    df2 = DataFrame({"id": [2, 2, 2], "ts": [ts, pd.NA, ts + 1]})
    df2["ts"] = df2["ts"].astype(dtype)
    gb2 = df2.groupby("id")

    # 计算分组后每组的最小值
    result3 = gb2.min()
    tm.assert_frame_equal(result3, expected)

    # 计算分组后每组的最大值
    res_max3 = gb2.max()
    tm.assert_frame_equal(res_max3, expected_max)

    # 使用指定的最小值计算至少包含100个非空值
    result4 = gb2.min(min_count=100)
    tm.assert_frame_equal(result4, expected2)

    # 使用指定的最大值计算至少包含100个非空值
    res_max4 = gb2.max(min_count=100)
    tm.assert_frame_equal(res_max4, expected2)


# 测试当分组为空时，计算最小和最大值是否正确处理
def test_min_max_nullable_uint64_empty_group():
    # 创建一个分类变量，包含10个值为0的类别
    cat = pd.Categorical([0] * 10, categories=[0, 1])
    # 创建一个数据框，包含'A'列和'B'列，其中'B'列包含从0开始的10个uint64类型的整数
    df = DataFrame({"A": cat, "B": pd.array(np.arange(10, dtype=np.uint64))})
    # 根据'A'列对数据框进行分组，观察空分组
    gb = df.groupby("A", observed=False)

    # 计算分组后每组的最小值
    res = gb.min()

    # 创建预期的最小值数据框，包含单列'B'，值为[0, pd.NA]，索引为分类变量的类别
    idx = pd.CategoricalIndex([0, 1], dtype=cat.dtype, name="A")
    expected = DataFrame({"B": pd.array([0, pd.NA], dtype="UInt64")}, index=idx)
    tm.assert_frame_equal(res, expected)

    # 计算分组后每组的最大值
    res = gb.max()
    # 修改预期的最大值数据框，第一行第一列的值改为9
    expected.iloc[0, 0] = 9
    tm.assert_frame_equal(res, expected)


# 使用参数化测试，测试对分类变量进行分组后计算第一个、最后一个、最小值和最大值的结果
@pytest.mark.parametrize("func", ["first", "last", "min", "max"])
def test_groupby_min_max_categorical(func):
    # 创建一个数据框，包含'col1'列和'col2'列，'col1'列为有序分类，'col2'列为有序分类
    df = DataFrame(
        {
            "col1": pd.Categorical(["A"], categories=list("AB"), ordered=True),
            "col2": pd.Categorical([1], categories=[1, 2], ordered=True),
            "value": 0.1,
        }
    )
    # 对数据框按'col1'列进行分组，不考虑观察到的所有类别
    result = getattr(df.groupby("col1", observed=False), func)()

    # 创建一个有序分类索引，包含数据'A'和'B'，索引名称为'col1'
    idx = pd.CategoricalIndex(data=["A", "B"], name="col1", ordered=True)
    # 创建一个 DataFrame 对象，其中包含两列数据：
    # - "col2" 列使用 pd.Categorical 类型，包含两个值：1 和 None，设置了类别 [1, 2] 并指定为有序
    # - "value" 列包含两个数值：0.1 和 None
    # 设置 DataFrame 的索引为预定义的变量 idx
    expected = DataFrame(
        {
            "col2": pd.Categorical([1, None], categories=[1, 2], ordered=True),
            "value": [0.1, None],
        },
        index=idx,
    )
    
    # 使用 pandas.testing 模块中的 assert_frame_equal 函数来比较变量 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize("func", ["min", "max"])
# 使用 pytest 的 parametrize 标记来多次运行测试函数，分别测试 "min" 和 "max" 函数
def test_min_empty_string_dtype(func):
    # GH#55619
    # 如果没有安装 pyarrow 模块，则跳过当前测试用例
    pytest.importorskip("pyarrow")
    
    # 定义字符串数据类型为 "string[pyarrow_numpy]"
    dtype = "string[pyarrow_numpy]"
    
    # 创建一个空的 DataFrame，指定数据类型为 dtype
    df = DataFrame({"a": ["a"], "b": "a", "c": "a"}, dtype=dtype).iloc[:0]
    
    # 对空的 DataFrame 按 "a" 列进行分组，然后调用传入的函数（min 或 max）
    result = getattr(df.groupby("a"), func)()
    
    # 创建预期结果的 DataFrame
    expected = DataFrame(
        columns=["b", "c"], dtype=dtype, index=pd.Index([], dtype=dtype, name="a")
    )
    
    # 使用 pytest 的 assert_frame_equal 函数比较实际结果和预期结果是否一致
    tm.assert_frame_equal(result, expected)


def test_max_nan_bug():
    # 创建一个包含 NaN 值的 DataFrame
    df = DataFrame(
        {
            "Unnamed: 0": ["-04-23", "-05-06", "-05-07"],
            "Date": [
                "2013-04-23 00:00:00",
                "2013-05-06 00:00:00",
                "2013-05-07 00:00:00",
            ],
            "app": Series([np.nan, np.nan, "OE"]),
            "File": ["log080001.log", "log.log", "xlsx"],
        }
    )
    
    # 根据 "Date" 列对 DataFrame 进行分组
    gb = df.groupby("Date")
    
    # 对分组后的结果获取 "File" 列的最大值
    r = gb[["File"]].max()
    
    # 获取 "File" 列的最大值，并转换为 DataFrame
    e = gb["File"].max().to_frame()
    
    # 使用 pytest 的 assert_frame_equal 函数比较实际结果和预期结果是否一致
    tm.assert_frame_equal(r, e)
    
    # 断言结果中没有 NaN 值
    assert not r["File"].isna().any()


@pytest.mark.slow
@pytest.mark.parametrize("with_nan", [True, False])
@pytest.mark.parametrize("keys", [["joe"], ["joe", "jim"]])
# 使用 pytest 的 parametrize 标记来多次运行测试函数，测试不同的参数组合
def test_series_groupby_nunique(sort, dropna, as_index, with_nan, keys):
    # 定义 n 和 m 的值
    n = 100
    m = 10
    
    # 创建日期范围为 10 天的日期序列
    days = date_range("2015-08-23", periods=10)
    
    # 创建一个包含随机数据的 DataFrame
    df = DataFrame(
        {
            "jim": np.random.default_rng(2).choice(list(ascii_lowercase), n),
            "joe": np.random.default_rng(2).choice(days, n),
            "julie": np.random.default_rng(2).integers(0, m, n),
        }
    )
    
    # 如果 with_nan 为 True，则将特定位置的值设为 NaN
    if with_nan:
        df = df.astype({"julie": float})  # 显式转换以避免下面的隐式转换
        df.loc[1::17, "jim"] = None
        df.loc[3::37, "joe"] = None
        df.loc[7::19, "julie"] = None
        df.loc[8::19, "julie"] = None
        df.loc[9::19, "julie"] = None
    
    # 备份原始的 DataFrame
    original_df = df.copy()
    
    # 根据指定的 keys 列进行分组，as_index 和 sort 参数根据测试用例的参数决定
    gr = df.groupby(keys, as_index=as_index, sort=sort)
    
    # 对分组后的 "julie" 列计算唯一值的数量
    left = gr["julie"].nunique(dropna=dropna)
    
    # 再次根据指定的 keys 列进行分组，计算 "julie" 列唯一值的数量，并转换为 Series
    gr = df.groupby(keys, as_index=as_index, sort=sort)
    right = gr["julie"].apply(Series.nunique, dropna=dropna)
    
    # 如果不使用 as_index，则重置右侧的索引
    if not as_index:
        right = right.reset_index(drop=True)
    
    # 根据 as_index 参数选择使用 assert_series_equal 或 assert_frame_equal 比较结果
    if as_index:
        tm.assert_series_equal(left, right, check_names=False)
    else:
        tm.assert_frame_equal(left, right, check_names=False)
    
    # 使用 assert_frame_equal 检查 DataFrame 是否与原始备份相同
    tm.assert_frame_equal(df, original_df)


def test_nunique():
    # 创建一个包含多列的 DataFrame
    df = DataFrame({"A": list("abbacc"), "B": list("abxacc"), "C": list("abbacx")})

    # 创建预期结果的 DataFrame
    expected = DataFrame({"A": list("abc"), "B": [1, 2, 1], "C": [1, 1, 2]})
    
    # 根据 "A" 列进行分组，计算每列的唯一值数量，结果为 DataFrame
    result = df.groupby("A", as_index=False).nunique()
    
    # 使用 pytest 的 assert_frame_equal 函数比较实际结果和预期结果是否一致
    tm.assert_frame_equal(result, expected)

    # 使用 "A" 列作为索引，计算每列的唯一值数量，结果为 Series
    expected.index = list("abc")
    expected.index.name = "A"
    expected = expected.drop(columns="A")
    result = df.groupby("A").nunique()
    
    # 使用 pytest 的 assert_frame_equal 函数比较实际结果和预期结果是否一致
    tm.assert_frame_equal(result, expected)

    # 替换 "x" 为 None，然后根据 "A" 列进行分组，计算每列的唯一值数量，设置 dropna=False
    result = df.replace({"x": None}).groupby("A").nunique(dropna=False)
    
    # 使用 pytest 的 assert_frame_equal 函数比较实际结果和预期结果是否一致
    tm.assert_frame_equal(result, expected)
    # 创建预期的 DataFrame，包含列 B 和 C，索引为 ['a', 'b', 'c']，并将索引命名为 'A'
    expected = DataFrame({"B": [1] * 3, "C": [1] * 3}, index=list("abc"))
    expected.index.name = "A"
    # 使用 df 中的 'x' 列进行替换操作，将 None 替换为 NaN，并对结果按 'A' 列进行分组，并计算每组的唯一值数量
    result = df.replace({"x": None}).groupby("A").nunique()
    # 使用测试模块 tm 中的 assert_frame_equal 方法比较 result 和预期的 expected DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
def test_nunique_with_object():
    # GH 11077
    # 创建一个包含数据的 DataFrame 对象，列名为 amount, id, name
    data = DataFrame(
        [
            [100, 1, "Alice"],
            [200, 2, "Bob"],
            [300, 3, "Charlie"],
            [-400, 4, "Dan"],
            [500, 5, "Edith"],
        ],
        columns=["amount", "id", "name"],
    )

    # 对 DataFrame 进行分组，按列'id'和'amount'分组，统计'name'列的唯一值数量
    result = data.groupby(["id", "amount"])["name"].nunique()

    # 从 DataFrame 的'id'和'amount'列创建一个 MultiIndex 对象
    index = MultiIndex.from_arrays([data.id, data.amount])

    # 创建一个期望的 Series，其值为每个组中唯一'name'值的数量，索引为上面创建的 MultiIndex
    expected = Series([1] * 5, name="name", index=index)

    # 使用测试工具比较结果 Series 和期望 Series 是否相等
    tm.assert_series_equal(result, expected)


def test_nunique_with_empty_series():
    # GH 12553
    # 创建一个空的 Series 对象，名称为'name'，数据类型为 object
    data = Series(name="name", dtype=object)

    # 对 Series 进行分组，按 level 0 分组，统计唯一值的数量
    result = data.groupby(level=0).nunique()

    # 创建一个期望的 Series，其名称为'name'，数据类型为'int64'
    expected = Series(name="name", dtype="int64")

    # 使用测试工具比较结果 Series 和期望 Series 是否相等
    tm.assert_series_equal(result, expected)


def test_nunique_with_timegrouper():
    # GH 13453
    # 创建一个包含时间和数据的 DataFrame 对象，将时间列设为索引
    test = DataFrame(
        {
            "time": [
                Timestamp("2016-06-28 09:35:35"),
                Timestamp("2016-06-28 16:09:30"),
                Timestamp("2016-06-28 16:46:28"),
            ],
            "data": ["1", "2", "3"],
        }
    ).set_index("time")

    # 对 DataFrame 进行分组，按每小时('h')分组，统计'data'列的唯一值数量
    result = test.groupby(pd.Grouper(freq="h"))["data"].nunique()

    # 创建一个期望的 Series，使用 apply 函数统计每个组中'data'列唯一值的数量
    expected = test.groupby(pd.Grouper(freq="h"))["data"].apply(Series.nunique)

    # 使用测试工具比较结果 Series 和期望 Series 是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "key, data, dropna, expected",
    [
        (
            ["x", "x", "x"],
            [Timestamp("2019-01-01"), pd.NaT, Timestamp("2019-01-01")],
            True,
            # 创建一个包含时间和 NaT 值的 DataFrame 对象
            Series([1], index=pd.Index(["x"], name="key"), name="data"),
        ),
        (
            ["x", "x", "x"],
            [dt.date(2019, 1, 1), pd.NaT, dt.date(2019, 1, 1)],
            True,
            # 创建一个包含日期和 NaT 值的 DataFrame 对象
            Series([1], index=pd.Index(["x"], name="key"), name="data"),
        ),
        (
            ["x", "x", "x", "y", "y"],
            [
                dt.date(2019, 1, 1),
                pd.NaT,
                dt.date(2019, 1, 1),
                pd.NaT,
                dt.date(2019, 1, 1),
            ],
            False,
            # 创建一个包含日期和 NaT 值的 DataFrame 对象，不删除 NaT 值
            Series([2, 2], index=pd.Index(["x", "y"], name="key"), name="data"),
        ),
        (
            ["x", "x", "x", "x", "y"],
            [
                dt.date(2019, 1, 1),
                pd.NaT,
                dt.date(2019, 1, 1),
                pd.NaT,
                dt.date(2019, 1, 1),
            ],
            False,
            # 创建一个包含日期和 NaT 值的 DataFrame 对象，不删除 NaT 值
            Series([2, 1], index=pd.Index(["x", "y"], name="key"), name="data"),
        ),
    ],
)
def test_nunique_with_NaT(key, data, dropna, expected):
    # GH 27951
    # 创建一个包含键和数据的 DataFrame 对象
    df = DataFrame({"key": key, "data": data})

    # 对 DataFrame 进行分组，按'key'列分组，统计'data'列的唯一值数量，可以选择是否删除 NaT 值
    result = df.groupby(["key"])["data"].nunique(dropna=dropna)

    # 使用测试工具比较结果 Series 和期望 Series 是否相等
    tm.assert_series_equal(result, expected)


def test_nunique_preserves_column_level_names():
    # GH 23222
    # 创建一个包含数据的 DataFrame 对象，只有一列'A'，列名为'level_0'
    test = DataFrame([1, 2, 2], columns=pd.Index(["A"], name="level_0"))

    # 对 DataFrame 进行分组，按每个值进行分组，统计每个组的唯一值数量
    result = test.groupby([0, 0, 0]).nunique()

    # 创建一个期望的 DataFrame，其值为每个组中唯一值的数量，列名与测试 DataFrame 的列名相同
    expected = DataFrame([2], index=np.array([0]), columns=test.columns)
    # 使用测试框架中的 assert_frame_equal 函数比较两个数据框架（DataFrame）的内容是否相等，并断言它们相等。
    tm.assert_frame_equal(result, expected)
def test_nunique_transform_with_datetime():
    # GH 35109 - transform with nunique on datetimes results in integers
    # 创建一个包含日期范围的数据帧，日期从"2008-12-31"到"2009-01-02"
    df = DataFrame(date_range("2008-12-31", "2009-01-02"), columns=["date"])
    # 对数据帧按照指定的分组键进行分组，并对每组中的日期列执行唯一值计数的转换操作
    result = df.groupby([0, 0, 1])["date"].transform("nunique")
    # 期望的结果是一个日期列的系列，显示每个组中唯一日期的数量
    expected = Series([2, 2, 1], name="date")
    # 使用测试工具函数验证结果是否与期望值相等
    tm.assert_series_equal(result, expected)


def test_empty_categorical(observed):
    # GH#21334
    # 创建一个包含单个整数的系列，并将其转换为分类类型
    cat = Series([1]).astype("category")
    # 从分类系列中选择空切片，生成一个空的系列
    ser = cat[:0]
    # 对空系列进行分组，指定是否观察到所有可能的分组键
    gb = ser.groupby(ser, observed=observed)
    # 对分组对象执行唯一值计数的操作
    result = gb.nunique()
    # 根据观察到的标志，期望结果是一个空的系列或者包含一个零值的系列
    if observed:
        expected = Series([], index=cat[:0], dtype="int64")
    else:
        expected = Series([0], index=cat, dtype="int64")
    # 使用测试工具函数验证结果是否与期望值相等
    tm.assert_series_equal(result, expected)


def test_intercept_builtin_sum():
    s = Series([1.0, 2.0, np.nan, 3.0])
    # 对系列按照指定的多个分组键进行分组
    grouped = s.groupby([0, 1, 2, 2])

    # GH#53425
    # 对分组对象应用内置的求和函数
    result = grouped.agg(builtins.sum)
    # GH#53425
    # 对分组对象应用内置的求和函数
    result2 = grouped.apply(builtins.sum)
    # 期望结果是一个包含指定索引和数值的系列，验证结果是否与期望值相等
    expected = Series([1.0, 2.0, np.nan], index=np.array([0, 1, 2]))
    tm.assert_series_equal(result, expected)
    tm.assert_series_equal(result2, expected)


@pytest.mark.parametrize("min_count", [0, 10])
def test_groupby_sum_mincount_boolean(min_count):
    b = True
    a = False
    na = np.nan
    dfg = pd.array([b, b, na, na, a, a, b], dtype="boolean")

    # 创建一个包含布尔值的数据帧
    df = DataFrame({"A": [1, 1, 2, 2, 3, 3, 1], "B": dfg})
    # 对数据帧按照'A'列进行分组，并对'B'列执行求和操作，指定最小计数
    result = df.groupby("A").sum(min_count=min_count)
    # 根据最小计数值为0或者10，期望结果是一个包含特定索引和值的数据帧
    if min_count == 0:
        expected = DataFrame(
            {"B": pd.array([3, 0, 0], dtype="Int64")},
            index=pd.Index([1, 2, 3], name="A"),
        )
        tm.assert_frame_equal(result, expected)
    else:
        expected = DataFrame(
            {"B": pd.array([pd.NA] * 3, dtype="Int64")},
            index=pd.Index([1, 2, 3], name="A"),
        )
        tm.assert_frame_equal(result, expected)


def test_groupby_sum_below_mincount_nullable_integer():
    # https://github.com/pandas-dev/pandas/issues/32861
    # 创建一个包含可空整数的数据帧
    df = DataFrame({"a": [0, 1, 2], "b": [0, 1, 2], "c": [0, 1, 2]}, dtype="Int64")
    # 对数据帧按照'a'列进行分组
    grouped = df.groupby("a")
    # 创建一个整数索引
    idx = pd.Index([0, 1, 2], name="a", dtype="Int64")

    # 对分组对象中的'b'列执行求和操作，指定最小计数为2
    result = grouped["b"].sum(min_count=2)
    # 期望结果是一个包含NA值的系列，验证结果是否与期望值相等
    expected = Series([pd.NA] * 3, dtype="Int64", index=idx, name="b")
    tm.assert_series_equal(result, expected)

    # 对分组对象执行求和操作，指定最小计数为2
    result = grouped.sum(min_count=2)
    # 期望结果是一个包含NA值的数据帧，验证结果是否与期望值相等
    expected = DataFrame({"b": [pd.NA] * 3, "c": [pd.NA] * 3}, dtype="Int64", index=idx)
    tm.assert_frame_equal(result, expected)


def test_groupby_sum_timedelta_with_nat():
    # GH#42659
    # 创建一个包含日期时间和时间间隔的数据帧
    df = DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [pd.Timedelta("1D"), pd.Timedelta("2D"), pd.Timedelta("3D"), pd.NaT],
        }
    )
    td3 = pd.Timedelta(days=3)

    # 对数据帧按照'a'列进行分组
    gb = df.groupby("a")

    # 对分组对象中的'b'列执行求和操作
    res = gb.sum()
    # 期望结果是一个包含特定索引和值的数据帧，验证结果是否与期望值相等
    expected = DataFrame({"b": [td3, td3]}, index=pd.Index([1, 2], name="a"))
    tm.assert_frame_equal(res, expected)

    # 对分组对象中'b'列执行求和操作，并指定最小计数为2
    res = gb["b"].sum()
    tm.assert_series_equal(res, expected["b"])

    # 对分组对象中'b'列执行求和操作，并指定最小计数为2
    res = gb["b"].sum(min_count=2)
    # 创建一个 Series 对象 expected，包含两个元素 td3 和 pd.NaT，数据类型为 'm8[ns]'，名称为 'b'，索引与 expected.index 一致
    expected = Series([td3, pd.NaT], dtype="m8[ns]", name="b", index=expected.index)
    # 使用测试工具函数 tm.assert_series_equal 比较 res 和 expected 两个 Series 对象是否相等
    tm.assert_series_equal(res, expected)
@pytest.mark.parametrize(
    "dtype", ["int8", "int16", "int32", "int64", "float32", "float64", "uint64"]
)
@pytest.mark.parametrize(
    "method,data",
    [
        ("first", {"df": [{"a": 1, "b": 1}, {"a": 2, "b": 3}]}),
        ("last", {"df": [{"a": 1, "b": 2}, {"a": 2, "b": 4}]}),
        ("min", {"df": [{"a": 1, "b": 1}, {"a": 2, "b": 3}]}),
        ("max", {"df": [{"a": 1, "b": 2}, {"a": 2, "b": 4}]}),
        ("count", {"df": [{"a": 1, "b": 2}, {"a": 2, "b": 2}], "out_type": "int64"}),
    ],
)
def test_groupby_non_arithmetic_agg_types(dtype, method, data):
    # 用于测试非算术聚合操作的函数，参数化测试不同的数据类型和聚合方法
    # GH9311, GH6620：对应的GitHub issue号
    df = DataFrame(
        [{"a": 1, "b": 1}, {"a": 1, "b": 2}, {"a": 2, "b": 3}, {"a": 2, "b": 4}]
    )

    # 将DataFrame中'b'列的数据类型转换为指定的dtype
    df["b"] = df.b.astype(dtype)

    # 如果data中没有'args'键，则初始化为空列表
    if "args" not in data:
        data["args"] = []

    # 确定输出类型为指定的data["out_type"]，否则默认为dtype
    if "out_type" in data:
        out_type = data["out_type"]
    else:
        out_type = dtype

    # 从data中取出期望的DataFrame数据，并创建对应的DataFrame对象
    exp = data["df"]
    df_out = DataFrame(exp)

    # 将DataFrame中'b'列的数据类型转换为指定的out_type
    df_out["b"] = df_out.b.astype(out_type)
    # 将DataFrame的索引设置为'a'
    df_out.set_index("a", inplace=True)

    # 根据'a'列对df进行分组
    grpd = df.groupby("a")
    # 使用getattr动态调用grpd对象的method方法，并传入data["args"]作为参数
    t = getattr(grpd, method)(*data["args"])
    # 断言t与预期的df_out相等
    tm.assert_frame_equal(t, df_out)


def scipy_sem(*args, **kwargs):
    from scipy.stats import sem

    return sem(*args, ddof=1, **kwargs)


@pytest.mark.parametrize(
    "op,targop",
    [
        ("mean", np.mean),
        ("median", np.median),
        ("std", np.std),
        ("var", np.var),
        ("sum", np.sum),
        ("prod", np.prod),
        ("min", np.min),
        ("max", np.max),
        ("first", lambda x: x.iloc[0]),
        ("last", lambda x: x.iloc[-1]),
        ("count", np.size),
        pytest.param("sem", scipy_sem, marks=td.skip_if_no("scipy")),
    ],
)
def test_ops_general(op, targop):
    # 通用操作的测试函数，参数化测试不同的操作和目标函数
    df = DataFrame(np.random.default_rng(2).standard_normal(1000))
    labels = np.random.default_rng(2).integers(0, 50, size=1000).astype(float)

    # 对df根据labels进行分组，然后应用op指定的操作
    result = getattr(df.groupby(labels), op)()
    kwargs = {"ddof": 1, "axis": 0} if op in ["std", "var"] else {}
    # 使用targop对df按labels进行聚合操作，得到预期的DataFrame对象
    expected = df.groupby(labels).agg(targop, **kwargs)
    # 断言result与expected相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "values",
    [
        {
            "a": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "b": [1, pd.NA, 2, 1, pd.NA, 2, 1, pd.NA, 2],
        },
        {"a": [1, 1, 2, 2, 3, 3], "b": [1, 2, 1, 2, 1, 2]},
    ],
)
@pytest.mark.parametrize("function", ["mean", "median", "var"])
def test_apply_to_nullable_integer_returns_float(values, function):
    # 测试对可空整数应用函数时的返回结果类型为浮点数
    # https://github.com/pandas-dev/pandas/issues/32219：相关的GitHub issue号
    output = 0.5 if function == "var" else 1.5
    arr = np.array([output] * 3, dtype=float)
    idx = pd.Index([1, 2, 3], name="a", dtype="Int64")
    # 创建预期的DataFrame对象
    expected = DataFrame({"b": arr}, index=idx).astype("Float64")

    # 根据'a'列对values创建DataFrame对象，并进行分组
    groups = DataFrame(values, dtype="Int64").groupby("a")

    # 对groups对象应用function指定的操作
    result = getattr(groups, function)()
    # 断言result与expected相等
    tm.assert_frame_equal(result, expected)

    # 对groups对象进行agg操作，应用function指定的聚合函数
    result = groups.agg(function)
    # 断言result与expected相等
    tm.assert_frame_equal(result, expected)

    # 对groups对象进行agg操作，应用所有指定的聚合函数
    result = groups.agg([function])
    # 将结果DataFrame的列标签设置为多级索引，其中包含单一级别的元组("b", function)
    expected.columns = MultiIndex.from_tuples([("b", function)])
    # 使用测试框架中的函数来断言两个DataFrame对象是否相等
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize(
    "op",
    [
        "sum",
        "prod",
        "min",
        "max",
        "median",
        "mean",
        "skew",
        "std",
        "var",
        "sem",
    ],
)
# 定义测试函数，参数化 op 变量，遍历不同的操作类型
def test_regression_allowlist_methods(op, skipna, sort):
    # GH6944
    # GH 17537
    # 显式测试允许列表中的方法
    # 创建一个包含单个元素的 DataFrame 对象
    frame = DataFrame([0])

    # 根据索引级别分组 DataFrame
    grouped = frame.groupby(level=0, sort=sort)

    if op == "skew":
        # 对于 skew 操作，使用 skipna 参数
        # 调用分组后对象的 skew 方法
        result = getattr(grouped, op)(skipna=skipna)
        # 期望结果是对每个分组应用 skew 操作后的结果
        expected = frame.groupby(level=0).apply(lambda h: getattr(h, op)(skipna=skipna))
        if sort:
            # 如果需要排序，则对期望结果进行索引排序
            expected = expected.sort_index()
        # 断言结果与期望相等
        tm.assert_frame_equal(result, expected)
    else:
        # 对于其他操作，直接调用对应的分组后对象方法
        result = getattr(grouped, op)()
        # 期望结果是对每个分组应用操作后的结果
        expected = frame.groupby(level=0).apply(lambda h: getattr(h, op)())
        if sort:
            # 如果需要排序，则对期望结果进行索引排序
            expected = expected.sort_index()
        # 断言结果与期望相等
        tm.assert_frame_equal(result, expected)


def test_groupby_prod_with_int64_dtype():
    # GH#46573
    # 创建一个包含整型数据的 DataFrame 对象
    data = [
        [1, 11],
        [1, 41],
        [1, 17],
        [1, 37],
        [1, 7],
        [1, 29],
        [1, 31],
        [1, 2],
        [1, 3],
        [1, 43],
        [1, 5],
        [1, 47],
        [1, 19],
        [1, 88],
    ]
    df = DataFrame(data, columns=["A", "B"], dtype="int64")
    # 对 DataFrame 按照 A 列分组并计算乘积，重新设置索引
    result = df.groupby(["A"]).prod().reset_index()
    # 期望结果是乘积后的 DataFrame
    expected = DataFrame({"A": [1], "B": [180970905912331920]}, dtype="int64")
    tm.assert_frame_equal(result, expected)


def test_groupby_std_datetimelike():
    # GH#48481
    # 创建时间间隔序列
    tdi = pd.timedelta_range("1 Day", periods=10000)
    ser = Series(tdi)
    # 每隔五个元素乘以 2，以获得不同分组的不同标准差
    ser[::5] *= 2

    # 创建 DataFrame 对象
    df = ser.to_frame("A").copy()

    # 添加两列 B 和 C，分别与时间戳相关联
    df["B"] = ser + Timestamp(0)
    df["C"] = ser + Timestamp(0, tz="UTC")
    # 最后一行包含 NaT
    df.iloc[-1] = pd.NaT

    # 根据多个列分组 DataFrame
    gb = df.groupby(list(range(5)) * 2000)

    # 计算标准差
    result = gb.std()

    # 注意：这里的结果不完全等同于 [gb.get_group(i).std() for i in gb.groups] 的结果，
    # 但是与处理 int64 数据时得到的浮点数误差相匹配 xref GH#51332
    td1 = pd.Timedelta("2887 days 11:21:02.326710176")
    td4 = pd.Timedelta("2886 days 00:42:34.664668096")
    exp_ser = Series([td1 * 2, td1, td1, td1, td4], index=np.arange(5))
    expected = DataFrame({"A": exp_ser, "B": exp_ser, "C": exp_ser})
    tm.assert_frame_equal(result, expected)
```