# `D:\src\scipysrc\pandas\pandas\tests\base\test_value_counts.py`

```
# 导入所需的库
import collections  # 导入 collections 库，用于计数操作
from datetime import timedelta  # 导入 datetime 模块中的 timedelta 类

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

import pandas as pd  # 导入 Pandas 库，用于数据处理和分析
from pandas import (  # 从 Pandas 库中导入多个类和函数
    DatetimeIndex,
    Index,
    Interval,
    IntervalIndex,
    MultiIndex,
    Series,
    Timedelta,
    TimedeltaIndex,
    array,
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块
from pandas.tests.base.common import allow_na_ops  # 从 Pandas 测试模块中导入 allow_na_ops 函数


@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
def test_value_counts(index_or_series_obj):
    obj = index_or_series_obj
    obj = np.repeat(obj, range(1, len(obj) + 1))  # 使用 NumPy 的 repeat 函数复制数据
    result = obj.value_counts()  # 对对象进行值计数

    counter = collections.Counter(obj)  # 使用 Counter 对象计数
    expected = Series(dict(counter.most_common()), dtype=np.int64, name="count")  # 创建预期的 Series 对象

    if obj.dtype != np.float16:  # 如果对象的数据类型不是 np.float16
        expected.index = expected.index.astype(obj.dtype)  # 将预期结果的索引转换为对象的数据类型
    else:
        with pytest.raises(NotImplementedError, match="float16 indexes are not "):  # 捕获 NotImplementedError 异常
            expected.index.astype(obj.dtype)  # 尝试转换索引为对象的数据类型
        return  # 返回，测试中止

    if isinstance(expected.index, MultiIndex):  # 如果预期结果的索引是 MultiIndex 类型
        expected.index.names = obj.names  # 设置预期结果索引的名称为对象的名称
    else:
        expected.index.name = obj.name  # 设置预期结果的单索引名称为对象的名称

    if not isinstance(result.dtype, np.dtype):  # 如果结果的数据类型不是 np.dtype 类型
        if getattr(obj.dtype, "storage", "") == "pyarrow":  # 如果对象的数据类型存储方式为 pyarrow
            expected = expected.astype("int64[pyarrow]")  # 将预期结果转换为指定的数据类型
        else:
            # 即 IntegerDtype
            expected = expected.astype("Int64")  # 将预期结果转换为 Int64 类型

    tm.assert_series_equal(result, expected)  # 使用 Pandas 测试模块的 assert_series_equal 函数比较结果


@pytest.mark.parametrize("null_obj", [np.nan, None])
@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
def test_value_counts_null(null_obj, index_or_series_obj):
    orig = index_or_series_obj
    obj = orig.copy()  # 复制原始对象

    if not allow_na_ops(obj):  # 如果对象不允许 NA 操作
        pytest.skip("type doesn't allow for NA operations")  # 跳过测试
    elif len(obj) < 1:  # 如果对象长度小于 1
        pytest.skip("Test doesn't make sense on empty data")  # 跳过测试，因为空数据时测试无意义
    elif isinstance(orig, MultiIndex):  # 如果原始对象是 MultiIndex 类型
        pytest.skip(f"MultiIndex can't hold '{null_obj}'")  # 跳过测试，因为 MultiIndex 不能包含特定类型的值

    values = obj._values  # 获取对象的值数组
    values[0:2] = null_obj  # 将数组中的前两个值设置为 null_obj

    klass = type(obj)  # 获取对象的类型
    repeated_values = np.repeat(values, range(1, len(values) + 1))  # 使用 NumPy 的 repeat 函数复制数据
    obj = klass(repeated_values, dtype=obj.dtype)  # 创建新对象，使用复制后的数据和原始数据类型

    # 因为 np.nan == np.nan is False, 但 None == None is True
    # np.nan 会被重复计数，而 None 不会
    counter = collections.Counter(obj.dropna())  # 使用 Counter 对象对删除 NA 值后的对象进行计数
    expected = Series(dict(counter.most_common()), dtype=np.int64, name="count")  # 创建预期的 Series 对象

    if obj.dtype != np.float16:  # 如果对象的数据类型不是 np.float16
        expected.index = expected.index.astype(obj.dtype)  # 将预期结果的索引转换为对象的数据类型
    else:
        with pytest.raises(NotImplementedError, match="float16 indexes are not "):  # 捕获 NotImplementedError 异常
            expected.index.astype(obj.dtype)  # 尝试转换索引为对象的数据类型
        return  # 返回，测试中止

    expected.index.name = obj.name  # 设置预期结果的索引名称为对象的名称

    result = obj.value_counts()  # 对对象进行值计数

    if not isinstance(result.dtype, np.dtype):  # 如果结果的数据类型不是 np.dtype 类型
        if getattr(obj.dtype, "storage", "") == "pyarrow":  # 如果对象的数据类型存储方式为 pyarrow
            expected = expected.astype("int64[pyarrow]")  # 将预期结果转换为指定的数据类型
        else:
            # 即 IntegerDtype
            expected = expected.astype("Int64")  # 将预期结果转换为 Int64 类型

    tm.assert_series_equal(result, expected)  # 使用 Pandas 测试模块的 assert_series_equal 函数比较结果
    # 将空对象作为键，赋值为3，添加到预期结果字典中
    expected[null_obj] = 3
    
    # 对对象进行值计数，包括NaN值，返回一个包含计数结果的Series对象
    result = obj.value_counts(dropna=False)
    
    # 对预期结果字典按索引排序，返回排序后的新字典
    expected = expected.sort_index()
    
    # 对计数结果的Series对象按索引排序，返回排序后的新Series对象
    result = result.sort_index()
    
    # 使用测试模块中的函数，比较并断言两个Series对象是否相等
    tm.assert_series_equal(result, expected)
# 定义一个函数用于测试值计数操作，这里包含两个参数：
# index_or_series：用于创建 Series 或 Index 的类
# using_infer_string：一个布尔值，指示是否使用推断字符串
def test_value_counts_inferred(index_or_series, using_infer_string):
    # 从参数 index_or_series 中获取类对象 klass
    klass = index_or_series
    # 创建一个包含字符串的列表 s_values
    s_values = ["a", "b", "b", "b", "b", "c", "d", "d", "a", "a"]
    # 根据 s_values 创建 Series 对象 s
    s = klass(s_values)
    # 创建预期结果 Series 对象 expected，包含各值的计数和索引
    expected = Series([4, 3, 2, 1], index=["b", "a", "d", "c"], name="count")
    # 断言 s.value_counts() 的结果与 expected 相等
    tm.assert_series_equal(s.value_counts(), expected)

    # 如果 s 是 Index 类型的实例
    if isinstance(s, Index):
        # 创建期望的 Index 对象 exp，包含 s_values 中唯一的对象数组
        exp = Index(np.unique(np.array(s_values, dtype=np.object_)))
        # 断言 s.unique() 的结果与 exp 相等
        tm.assert_index_equal(s.unique(), exp)
    else:
        # 创建期望的对象数组 exp，包含 s_values 中唯一的对象数组
        exp = np.unique(np.array(s_values, dtype=np.object_))
        # 如果 using_infer_string 为真，则将 exp 转换为 array 类型
        if using_infer_string:
            exp = array(exp)
        # 断言 s.unique() 的结果与 exp 相等
        tm.assert_equal(s.unique(), exp)

    # 断言 s 的唯一值数量为 4
    assert s.nunique() == 4
    # 获取值计数的直方图，不进行排序，并且按值进行排序后再断言
    hist = s.value_counts(sort=False).sort_values()
    # 创建预期的 Series 对象 expected，包含按索引排序后的值计数
    expected = Series([3, 1, 4, 2], index=list("acbd"), name="count").sort_values()
    # 断言 hist 与 expected 相等
    tm.assert_series_equal(hist, expected)

    # 按升序排序获取值计数的直方图
    hist = s.value_counts(ascending=True)
    # 创建预期的 Series 对象 expected，包含按升序索引排序后的值计数
    expected = Series([1, 2, 3, 4], index=list("cdab"), name="count")
    # 断言 hist 与 expected 相等
    tm.assert_series_equal(hist, expected)

    # 获取相对频率的值计数的直方图
    hist = s.value_counts(normalize=True)
    # 创建预期的 Series 对象 expected，包含按索引排序后的相对频率
    expected = Series(
        [0.4, 0.3, 0.2, 0.1], index=["b", "a", "d", "c"], name="proportion"
    )
    # 断言 hist 与 expected 相等
    tm.assert_series_equal(hist, expected)


# 定义一个函数用于测试值计数的 bins 参数
def test_value_counts_bins(index_or_series, using_infer_string):
    # 从参数 index_or_series 中获取类对象 klass
    klass = index_or_series
    # 创建一个包含字符串的列表 s_values
    s_values = ["a", "b", "b", "b", "b", "c", "d", "d", "a", "a"]
    # 根据 s_values 创建 Series 对象 s
    s = klass(s_values)

    # 测试 bins 参数只能用于数值数据的断言
    msg = "bins argument only works with numeric data"
    with pytest.raises(TypeError, match=msg):
        s.value_counts(bins=1)

    # 创建一个包含整数的 Series 对象 s1
    s1 = Series([1, 1, 2, 3])
    # 使用 bins 参数获取值计数的结果 res1，并断言其与预期结果 exp1 相等
    res1 = s1.value_counts(bins=1)
    exp1 = Series({Interval(0.997, 3.0): 4}, name="count")
    tm.assert_series_equal(res1, exp1)
    # 使用 bins 参数获取值计数的归一化结果 res1n，并断言其与预期结果 exp1n 相等
    res1n = s1.value_counts(bins=1, normalize=True)
    exp1n = Series({Interval(0.997, 3.0): 1.0}, name="proportion")
    tm.assert_series_equal(res1n, exp1n)

    # 如果 s1 是 Index 类型的实例，则断言 s1.unique() 结果与期望的 Index 对象相等
    if isinstance(s1, Index):
        tm.assert_index_equal(s1.unique(), Index([1, 2, 3]))
    else:
        # 创建预期的整数数组 exp，包含 s1 中唯一的整数值
        exp = np.array([1, 2, 3], dtype=np.int64)
        # 断言 s1.unique() 的结果与 exp 相等
        tm.assert_numpy_array_equal(s1.unique(), exp)

    # 断言 s1 的唯一值数量为 3
    assert s1.nunique() == 3

    # 使用 bins 参数获取值计数的结果 res4，将索引转换为区间索引 intervals，然后断言其与预期结果 exp4 相等
    res4 = s1.value_counts(bins=4, dropna=True)
    intervals = IntervalIndex.from_breaks([0.997, 1.5, 2.0, 2.5, 3.0])
    exp4 = Series([2, 1, 1, 0], index=intervals.take([0, 1, 3, 2]), name="count")
    tm.assert_series_equal(res4, exp4)

    # 使用 bins 参数获取值计数的结果 res4，将索引转换为区间索引 intervals，然后断言其与预期结果 exp4 相等
    res4 = s1.value_counts(bins=4, dropna=False)
    intervals = IntervalIndex.from_breaks([0.997, 1.5, 2.0, 2.5, 3.0])
    exp4 = Series([2, 1, 1, 0], index=intervals.take([0, 1, 3, 2]), name="count")
    tm.assert_series_equal(res4, exp4)

    # 使用 bins 参数获取值计数的归一化结果 res4n，将索引转换为区间索引 intervals，然后断言其与预期结果 exp4n 相等
    res4n = s1.value_counts(bins=4, normalize=True)
    exp4n = Series(
        [0.5, 0.25, 0.25, 0], index=intervals.take([0, 1, 3, 2]), name="proportion"
    )
    tm.assert_series_equal(res4n, exp4n)

    # 处理缺失值的情况
    # 定义一个包含字符串和NaN值的列表
    s_values = ["a", "b", "b", "b", np.nan, np.nan, "d", "d", "a", "a", "b"]
    # 使用给定的类（klass）构造一个 Series 对象 s
    s = klass(s_values)
    # 预期的 Series 对象，展示每个值的出现次数，并指定索引和名称
    expected = Series([4, 3, 2], index=["b", "a", "d"], name="count")
    # 断言 s 的 value_counts 方法生成的 Series 与预期结果 expected 相等
    tm.assert_series_equal(s.value_counts(), expected)

    # 如果 s 是 Index 类型
    if isinstance(s, Index):
        # 预期的唯一值索引，包括 "a", "b", NaN, "d"
        exp = Index(["a", "b", np.nan, "d"])
        # 断言 s 的 unique 方法生成的 Index 与预期结果 exp 相等
        tm.assert_index_equal(s.unique(), exp)
    else:
        # 否则，预期的唯一值数组，包括 "a", "b", NaN, "d"，数据类型为 object
        exp = np.array(["a", "b", np.nan, "d"], dtype=object)
        # 如果使用推断的字符串类型标志
        if using_infer_string:
            # 将 exp 转换为数组类型
            exp = array(exp)
        # 断言 s 的 unique 方法生成的数组与预期结果 exp 相等
        tm.assert_equal(s.unique(), exp)
    # 断言 s 的唯一值数量为 3
    assert s.nunique() == 3

    # 根据 klass 的类型，构造一个空的 Series 对象 s
    s = klass({}) if klass is dict else klass({}, dtype=object)
    # 预期的空 Series 对象，数据类型为 np.int64，名称为 "count"
    expected = Series([], dtype=np.int64, name="count")
    # 断言 s 的 value_counts 方法生成的 Series 与预期结果 expected 相等，忽略索引类型检查
    tm.assert_series_equal(s.value_counts(), expected, check_index_type=False)
    # 根据原始数据类型的不同，返回的数据类型可能不同
    # 如果 s 是 Index 类型
    if isinstance(s, Index):
        # 断言 s 的 unique 方法生成的 Index 与空 Index 相等，允许不精确匹配
        tm.assert_index_equal(s.unique(), Index([]), exact=False)
    else:
        # 断言 s 的 unique 方法生成的数组与空数组相等，忽略数据类型检查
        tm.assert_numpy_array_equal(s.unique(), np.array([]), check_dtype=False)

    # 断言 s 的唯一值数量为 0
    assert s.nunique() == 0
# 定义测试函数，用于测试 datetime64 数据类型的值计数方法
def test_value_counts_datetime64(index_or_series, unit):
    # 获取传入参数的类别
    klass = index_or_series

    # 创建一个 DataFrame，包含 person_id、dt 和 food 列
    df = pd.DataFrame(
        {
            "person_id": ["xxyyzz", "xxyyzz", "xxyyzz", "xxyyww", "foofoo", "foofoo"],
            "dt": pd.to_datetime(
                [
                    "2010-01-01",
                    "2010-01-01",
                    "2010-01-01",
                    "2009-01-01",
                    "2008-09-09",
                    "2008-09-09",
                ]
            ).as_unit(unit),  # 将日期时间数据转换为指定单位的 datetime64 类型
            "food": ["PIE", "GUM", "EGG", "EGG", "PIE", "GUM"],
        }
    )

    # 从 DataFrame 中提取 dt 列，并复制为 Series 类型
    s = klass(df["dt"].copy())
    s.name = None  # 清空 Series 的名称
    # 创建一个 datetime64 数组作为索引
    idx = pd.to_datetime(
        ["2010-01-01 00:00:00", "2008-09-09 00:00:00", "2009-01-01 00:00:00"]
    ).as_unit(unit)
    # 创建预期的 Series，表示每个唯一值的计数
    expected_s = Series([3, 2, 1], index=idx, name="count")
    # 断言 Series 的值计数结果与预期结果相等
    tm.assert_series_equal(s.value_counts(), expected_s)

    # 创建预期的 datetime64 数组
    expected = array(
        np.array(
            ["2010-01-01 00:00:00", "2009-01-01 00:00:00", "2008-09-09 00:00:00"],
            dtype=f"datetime64[{unit}]",
        )
    )
    # 获取 Series 的唯一值数组
    result = s.unique()
    # 根据不同的类别进行断言
    if isinstance(s, Index):
        tm.assert_index_equal(result, DatetimeIndex(expected))
    else:
        tm.assert_extension_array_equal(result, expected)

    # 断言 Series 的唯一值数量为 3
    assert s.nunique() == 3

    # 处理包含 NaT（Not a Time）的情况
    s = df["dt"].copy()
    s = klass(list(s.values) + [pd.NaT] * 4)  # 添加 NaT 到 Series 中
    if klass is Series:
        s = s.dt.as_unit(unit)  # 转换为指定单位的 datetime64 类型
    else:
        s = s.as_unit(unit)

    # 计算值的计数
    result = s.value_counts()
    # 断言结果的索引类型为 datetime64
    assert result.index.dtype == f"datetime64[{unit}]"
    # 创建包含 NaT 计数的预期结果
    tm.assert_series_equal(result, expected_s)

    # 计算包含 NaN 的值的计数
    result = s.value_counts(dropna=False)
    # 创建包含 NaT 计数的预期结果
    expected_s = pd.concat(
        [
            Series([4], index=DatetimeIndex([pd.NaT]).as_unit(unit), name="count"),
            expected_s,
        ]
    )
    # 断言结果与预期结果相等
    tm.assert_series_equal(result, expected_s)

    # 断言 Series 的数据类型为 datetime64
    assert s.dtype == f"datetime64[{unit}]"
    # 获取 Series 的唯一值数组
    unique = s.unique()
    # 断言唯一值数组的数据类型为 datetime64
    assert unique.dtype == f"datetime64[{unit}]"

    # 使用索引断言唯一值数组的结果
    if isinstance(s, Index):
        exp_idx = DatetimeIndex(expected.tolist() + [pd.NaT]).as_unit(unit)
        tm.assert_index_equal(unique, exp_idx)
    else:
        tm.assert_extension_array_equal(unique[:3], expected)
        assert pd.isna(unique[3])  # 断言唯一值数组的第四个值为 NaN

    # 断言 Series 的唯一值数量为 3
    assert s.nunique() == 3
    # 断言包含 NaN 的唯一值数量为 4
    assert s.nunique(dropna=False) == 4


# 定义测试函数，用于测试 timedelta64 数据类型的值计数方法
def test_value_counts_timedelta64(index_or_series, unit):
    # timedelta64 类型的单位为纳秒
    klass = index_or_series

    # 创建一个表示一天时间间隔的 TimedeltaIndex
    day = Timedelta(timedelta(1)).as_unit(unit)
    tdi = TimedeltaIndex([day], name="dt").as_unit(unit)

    # 创建包含一天时间间隔的数组
    tdvals = np.zeros(6, dtype=f"m8[{unit}]") + day
    td = klass(tdvals, name="dt")

    # 计算时间间隔值的计数
    result = td.value_counts()
    # 创建预期的 Series，表示每个时间间隔的计数
    expected_s = Series([6], index=tdi, name="count")
    # 断言结果与预期相等
    tm.assert_series_equal(result, expected_s)

    # 获取时间间隔值的唯一值数组
    expected = tdi
    result = td.unique()
    # 根据类别进行断言
    if isinstance(td, Index):
        tm.assert_index_equal(result, expected)
    else:
        # 如果不是第一种情况，则执行以下代码块
        # 使用测试工具函数比较结果和期望值的扩展数组是否相等
        tm.assert_extension_array_equal(result, expected._values)

    # 创建一个包含6个元素的NumPy数组，每个元素都是由day的日期和unit组成的时间戳
    td2 = day + np.zeros(6, dtype=f"m8[{unit}]")
    # 使用类klass将上述数组包装成一个对象，并命名为"dt"
    td2 = klass(td2, name="dt")
    # 对对象td2进行值计数，返回结果作为Series对象
    result2 = td2.value_counts()
    # 使用测试工具函数比较result2和期望的Series对象是否相等
    tm.assert_series_equal(result2, expected_s)
# 测试函数，用于验证带有 NaN 值的 value_counts 方法的行为
def test_value_counts_with_nan(dropna, index_or_series):
    # GH31944: 问题跟踪编号
    # 将传入的 index_or_series 赋值给 klass
    klass = index_or_series
    # 定义一个包含 True、pd.NA、np.nan 的列表作为 values
    values = [True, pd.NA, np.nan]
    # 使用 klass 创建一个对象 obj，其值为 values
    obj = klass(values)
    # 调用对象的 value_counts 方法，传入 dropna 参数，并将结果赋值给 res
    res = obj.value_counts(dropna=dropna)
    # 根据 dropna 的值生成预期的 Series 对象 expected
    if dropna is True:
        expected = Series([1], index=Index([True], dtype=obj.dtype), name="count")
    else:
        expected = Series([1, 1, 1], index=[True, pd.NA, np.nan], name="count")
    # 使用测试框架的 assert_series_equal 函数比较 res 和 expected
    tm.assert_series_equal(res, expected)


# 测试函数，用于验证对象推断的对象值计数行为（已弃用）
def test_value_counts_object_inference_deprecated():
    # GH#56161: 问题跟踪编号
    # 创建一个带有时区信息的日期范围 dti
    dti = pd.date_range("2016-01-01", periods=3, tz="UTC")
    # 将 dti 转换为 object 类型，并赋值给 idx
    idx = dti.astype(object)
    # 调用 idx 的 value_counts 方法，将结果赋值给 res
    res = idx.value_counts()
    # 获取 dti 的 value_counts 结果，并将其索引类型转换为 object 类型，赋值给 exp
    exp = dti.value_counts()
    exp.index = exp.index.astype(object)
    # 使用测试框架的 assert_series_equal 函数比较 res 和 exp
    tm.assert_series_equal(res, exp)
```