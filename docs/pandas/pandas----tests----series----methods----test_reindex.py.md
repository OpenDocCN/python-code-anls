# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_reindex.py`

```
# 导入必要的库：numpy用于数值计算，pytest用于单元测试
import numpy as np
import pytest

# 导入 pandas 库中的特定模块和类
from pandas._config import using_pyarrow_string_dtype
from pandas import (
    NA,
    Categorical,
    Float64Dtype,
    Index,
    MultiIndex,
    NaT,
    Period,
    PeriodIndex,
    RangeIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    isna,
)
# 导入 pandas 内部测试模块
import pandas._testing as tm

# 标记为预期失败的测试用例，原因是箭头库的共享内存功能不兼容
@pytest.mark.xfail(
    using_pyarrow_string_dtype(), reason="share memory doesn't work for arrow"
)
def test_reindex(datetime_series, string_series):
    # 重新索引为相同索引的字符串系列
    identity = string_series.reindex(string_series.index)

    # 断言新创建的系列与原系列共享内存
    assert np.may_share_memory(string_series.index, identity.index)

    # 断言新创建的索引对象与原索引对象相同
    assert identity.index.is_(string_series.index)
    # 断言新创建的索引对象与原索引对象完全相同
    assert identity.index.identical(string_series.index)

    # 从字符串系列中选择子索引范围
    subIndex = string_series.index[10:20]
    subSeries = string_series.reindex(subIndex)

    # 验证子系列中每个值是否与原系列相同
    for idx, val in subSeries.items():
        assert val == string_series[idx]

    # 从日期时间系列中选择子索引范围
    subIndex2 = datetime_series.index[10:20]
    subTS = datetime_series.reindex(subIndex2)

    # 验证子时间序列中每个值是否与原时间序列相同
    for idx, val in subTS.items():
        assert val == datetime_series[idx]

    # 对于选择的子索引范围内的日期时间系列，验证是否全部为 NaN
    stuffSeries = datetime_series.reindex(subIndex)
    assert np.isnan(stuffSeries).all()

    # 针对非连续索引的情况，从日期时间系列中选择每隔一个值的子序列
    nonContigIndex = datetime_series.index[::2]
    subNonContig = datetime_series.reindex(nonContigIndex)
    for idx, val in subNonContig.items():
        assert val == datetime_series[idx]

    # 返回一个原索引相同的时间序列的副本
    result = datetime_series.reindex()
    assert result is not datetime_series


# 测试重新索引处理 NaN 值的情况
def test_reindex_nan():
    # 创建一个包含 NaN 索引的 Series 对象
    ts = Series([2, 3, 5, 7], index=[1, 4, np.nan, 8])

    # 创建用于重新索引的 NaN 值列表和对应的索引转换关系列表
    i, j = [np.nan, 1, np.nan, 8, 4, np.nan], [2, 0, 2, 3, 1, 2]
    # 断言重新索引后的 Series 对象与预期的结果相等
    tm.assert_series_equal(ts.reindex(i), ts.iloc[j])

    # 将索引类型转换为对象类型
    ts.index = ts.index.astype("object")

    # 使用 reindex 方法对 NaN 值进行重新索引，跳过索引类型检查
    tm.assert_series_equal(ts.reindex(i), ts.iloc[j], check_index_type=False)


# 测试在 Series 中添加 NaT 值进行重新索引
def test_reindex_series_add_nat():
    # 创建一个时间范围
    rng = date_range("1/1/2000 00:00:00", periods=10, freq="10s")
    series = Series(rng)

    # 对时间序列进行重新索引，扩展索引范围到 15
    result = series.reindex(range(15))
    # 断言结果的数据类型为日期时间类型
    assert np.issubdtype(result.dtype, np.dtype("M8[ns]"))

    # 验证结果中最后 5 个值是否全部为 NaN
    mask = result.isna()
    assert mask[-5:].all()
    # 验证结果中除最后 5 个值外的其他值是否不含 NaN
    assert not mask[:-5].any()


# 测试带有日期时间的重新索引操作
def test_reindex_with_datetimes():
    # 创建一个包含随机数的时间序列
    rng = date_range("1/1/2000", periods=20)
    ts = Series(np.random.default_rng(2).standard_normal(20), index=rng)

    # 对时间序列进行重新索引，选择索引范围为 [5:10]
    result = ts.reindex(list(ts.index[5:10]))
    expected = ts[5:10]
    # 将预期结果的索引频率设为 None 进行比较
    expected.index = expected.index._with_freq(None)
    # 断言重新索引后的结果与预期结果相等
    tm.assert_series_equal(result, expected)

    # 使用索引列表对时间序列进行重新索引，选择索引范围为 [5:10]
    result = ts[list(ts.index[5:10])]
    # 断言重新索引后的结果与预期结果相等
    tm.assert_series_equal(result, expected)


# 测试边界情况：空 Series 对象的重新索引
def test_reindex_corner(datetime_series):
    # 创建一个空索引的 Series 对象
    empty = Series(index=[])
    # 使用 pad 方法对空 Series 对象进行重新索引，确认功能正常
    empty.reindex(datetime_series.index, method="pad")  # it works

    # 使用 pad 方法对空 Series 对象进行重新索引，得到重新索引后的结果
    reindexed = empty.reindex(datetime_series.index, method="pad")

    # 通过传递非索引对象进行测试
    # pass non-Index
    # 使用 datetime_series 的索引重新索引 datetime_series，生成一个重新索引的 Series
    reindexed = datetime_series.reindex(list(datetime_series.index))
    
    # 将 datetime_series 的索引频率设置为 None
    datetime_series.index = datetime_series.index._with_freq(None)
    
    # 使用 pytest 的 assert_series_equal 函数比较 datetime_series 和 reindexed Series 是否相等
    tm.assert_series_equal(datetime_series, reindexed)

    # 指定一个错误的填充方法 'foo'，并期待引发 ValueError 异常，异常消息匹配特定的正则表达式消息
    ts = datetime_series[::2]
    msg = (
        r"Invalid fill method\. Expecting pad \(ffill\), backfill "
        r"\(bfill\) or nearest\. Got foo"
    )
    with pytest.raises(ValueError, match=msg):
        # 重新索引 ts Series，使用了错误的填充方法 'foo'，预期会引发 ValueError 异常
        ts.reindex(datetime_series.index, method="foo")
def test_reindex_pad():
    # 创建一个包含整数序列的 Series 对象，数据类型为 int64
    s = Series(np.arange(10), dtype="int64")
    # 从 s 中按照步长为 2 取出元素，创建新的 Series 对象 s2
    s2 = s[::2]

    # 使用“pad”方法重新索引 s2 到 s 的索引上，生成重新索引后的 Series 对象 reindexed
    reindexed = s2.reindex(s.index, method="pad")
    # 使用“ffill”方法重新索引 s2 到 s 的索引上，生成重新索引后的 Series 对象 reindexed2
    reindexed2 = s2.reindex(s.index, method="ffill")
    # 检查 reindexed 和 reindexed2 是否相等
    tm.assert_series_equal(reindexed, reindexed2)

    # 创建预期结果的 Series 对象 expected
    expected = Series([0, 0, 2, 2, 4, 4, 6, 6, 8, 8])
    # 检查 reindexed 是否与 expected 相等
    tm.assert_series_equal(reindexed, expected)


def test_reindex_pad2():
    # GH4604
    # 创建一个包含整数的 Series 对象 s，指定索引为字符串列表
    s = Series([1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])
    # 新的索引列表
    new_index = ["a", "g", "c", "f"]
    # 创建预期结果的 Series 对象 expected
    expected = Series([1, 1, 3, 3.0], index=new_index)

    # 执行 reindex 操作后，使用 ffill 方法填充缺失值，结果存储在 result 中
    result = s.reindex(new_index).ffill()
    # 检查 result 是否与 expected 相等
    tm.assert_series_equal(result, expected)

    # 再次执行 reindex 操作后，使用 ffill 方法填充缺失值，结果存储在 result 中
    result = s.reindex(new_index).ffill()
    # 检查 result 是否与 expected 相等
    tm.assert_series_equal(result, expected)

    # 更新预期结果的 Series 对象 expected
    expected = Series([1, 5, 3, 5], index=new_index)
    # 执行 reindex 操作后，使用 ffill 方法填充缺失值，结果存储在 result 中
    result = s.reindex(new_index, method="ffill")
    # 检查 result 是否与 expected 相等
    tm.assert_series_equal(result, expected)


def test_reindex_inference():
    # 推断出新的数据类型
    # 创建一个包含布尔值的 Series 对象 s，指定索引为字符列表 "abcd"
    s = Series([True, False, False, True], index=list("abcd"))
    # 新的索引列表
    new_index = "agc"
    # 执行 reindex 操作后，使用 ffill 方法填充缺失值，结果存储在 result 中
    result = s.reindex(list(new_index)).ffill()
    # 创建预期结果的 Series 对象 expected，指定数据类型为 object
    expected = Series([True, True, False], index=list(new_index), dtype=object)
    # 检查 result 是否与 expected 相等
    tm.assert_series_equal(result, expected)


def test_reindex_downcasting():
    # GH4618 shifted series downcasting
    # 创建一个包含布尔值的 Series 对象 s，索引为整数范围
    s = Series(False, index=range(5))
    # 执行 shift 操作后，使用 bfill 方法填充缺失值，结果存储在 result 中
    result = s.shift(1).bfill()
    # 创建预期结果的 Series 对象 expected，指定数据类型为 object
    expected = Series(False, index=range(5), dtype=object)
    # 检查 result 是否与 expected 相等
    tm.assert_series_equal(result, expected)


def test_reindex_nearest():
    # 创建一个包含整数序列的 Series 对象 s，数据类型为 int64
    s = Series(np.arange(10, dtype="int64"))
    # 目标重新索引的目标值列表
    target = [0.1, 0.9, 1.5, 2.0]
    # 执行 reindex 操作，使用 nearest 方法填充缺失值，结果存储在 result 中
    result = s.reindex(target, method="nearest")
    # 创建预期结果的 Series 对象 expected，将目标值四舍五入并转换为 int64 类型
    expected = Series(np.around(target).astype("int64"), target)
    # 检查 result 是否与 expected 相等
    tm.assert_series_equal(expected, result)

    # 执行 reindex 操作，使用 nearest 方法填充缺失值，指定容忍度为 0.2，结果存储在 result 中
    result = s.reindex(target, method="nearest", tolerance=0.2)
    # 创建预期结果的 Series 对象 expected
    expected = Series([0, 1, np.nan, 2], target)
    # 检查 result 是否与 expected 相等
    tm.assert_series_equal(expected, result)

    # 执行 reindex 操作，使用 nearest 方法填充缺失值，指定容忍度列表，结果存储在 result 中
    result = s.reindex(target, method="nearest", tolerance=[0.3, 0.01, 0.4, 3])
    # 创建预期结果的 Series 对象 expected
    expected = Series([0, np.nan, np.nan, 2], target)
    # 检查 result 是否与 expected 相等
    tm.assert_series_equal(expected, result)


def test_reindex_int(datetime_series):
    # 从 datetime_series 中按照步长为 2 取出元素，创建新的 Series 对象 ts
    ts = datetime_series[::2]
    # 创建一个与 ts 索引相同的整数全为零的 Series 对象 int_ts
    int_ts = Series(np.zeros(len(ts), dtype=int), index=ts.index)

    # 执行 reindex 操作后，结果存储在 reindexed_int 中
    reindexed_int = int_ts.reindex(datetime_series.index)

    # 如果引入了 NaN 值
    assert reindexed_int.dtype == np.float64

    # 如果没有引入 NaN 值
    reindexed_int = int_ts.reindex(int_ts.index[::2])
    assert reindexed_int.dtype == np.dtype(int)


def test_reindex_bool(datetime_series):
    # 创建一个与 datetime_series 索引相同的布尔值全为零的 Series 对象 bool_ts
    ts = datetime_series[::2]
    bool_ts = Series(np.zeros(len(ts), dtype=bool), index=ts.index)

    # 执行 reindex 操作后，结果存储在 reindexed_bool 中
    reindexed_bool = bool_ts.reindex(datetime_series.index)

    # 如果引入了 NaN 值
    assert reindexed_bool.dtype == np.object_

    # 如果没有引入 NaN 值
    reindexed_bool = bool_ts.reindex(bool_ts.index[::2])
    # 使用断言确保 reindexed_bool 的数据类型是 numpy 中的布尔型
    assert reindexed_bool.dtype == np.bool_
def test_reindex_bool_pad(datetime_series):
    # 从索引5开始切片时间序列
    ts = datetime_series[5:]
    # 创建一个布尔类型的 Series，长度与切片后的时间序列相同，索引与时间序列相同，值全部为 False
    bool_ts = Series(np.zeros(len(ts), dtype=bool), index=ts.index)
    # 使用“pad”方法重新索引填充布尔类型的 Series，索引与原时间序列相同
    filled_bool = bool_ts.reindex(datetime_series.index, method="pad")
    # 断言填充的结果中前5个值是否全部为缺失值
    assert isna(filled_bool[:5]).all()


def test_reindex_categorical():
    # 创建一个日期范围为2000年1月1日起，3个周期的索引
    index = date_range("20000101", periods=3)

    # 对一个分类数据 Series 进行重新索引到一个无效的分类索引
    s = Series(["a", "b", "c"], dtype="category")
    result = s.reindex(index)
    # 创建一个期望的 Series，其值为 NaN，分类为 ["a", "b", "c"]，索引与输入的 index 相同
    expected = Series(
        Categorical(values=[np.nan, np.nan, np.nan], categories=["a", "b", "c"])
    )
    expected.index = index
    # 断言结果与期望是否相等
    tm.assert_series_equal(result, expected)

    # 部分重新索引
    expected = Series(Categorical(values=["b", "c"], categories=["a", "b", "c"]))
    expected.index = [1, 2]
    result = s.reindex([1, 2])
    tm.assert_series_equal(result, expected)

    expected = Series(Categorical(values=["c", np.nan], categories=["a", "b", "c"]))
    expected.index = [2, 3]
    result = s.reindex([2, 3])
    tm.assert_series_equal(result, expected)


def test_reindex_astype_order_consistency():
    # GH#17444
    # 创建一个索引为 [2, 0, 1] 的 Series
    ser = Series([1, 2, 3], index=[2, 0, 1])
    new_index = [0, 1, 2]
    temp_dtype = "category"
    new_dtype = str
    # 先重新索引，然后转换数据类型为 temp_dtype，再转换数据类型为 new_dtype
    result = ser.reindex(new_index).astype(temp_dtype).astype(new_dtype)
    # 期望先转换为 temp_dtype，再重新索引，最后转换为 new_dtype
    expected = ser.astype(temp_dtype).reindex(new_index).astype(new_dtype)
    # 断言结果与期望是否相等
    tm.assert_series_equal(result, expected)


def test_reindex_fill_value():
    # -----------------------------------------------------------
    # 浮点数
    floats = Series([1.0, 2.0, 3.0])
    result = floats.reindex([1, 2, 3])
    # 创建一个期望的 Series，包含填充后的结果，索引为 [1, 2, 3]
    expected = Series([2.0, 3.0, np.nan], index=[1, 2, 3])
    # 断言结果与期望是否相等
    tm.assert_series_equal(result, expected)

    result = floats.reindex([1, 2, 3], fill_value=0)
    # 创建一个期望的 Series，包含填充后的结果，索引为 [1, 2, 3]，填充值为 0
    expected = Series([2.0, 3.0, 0], index=[1, 2, 3])
    # 断言结果与期望是否相等
    tm.assert_series_equal(result, expected)

    # -----------------------------------------------------------
    # 整数
    ints = Series([1, 2, 3])

    result = ints.reindex([1, 2, 3])
    # 创建一个期望的 Series，包含填充后的结果，索引为 [1, 2, 3]，填充值为 NaN
    expected = Series([2.0, 3.0, np.nan], index=[1, 2, 3])
    # 断言结果与期望是否相等
    tm.assert_series_equal(result, expected)

    # 不进行数据类型的上转换
    result = ints.reindex([1, 2, 3], fill_value=0)
    # 创建一个期望的 Series，包含填充后的结果，索引为 [1, 2, 3]，填充值为 0，数据类型为整数
    expected = Series([2, 3, 0], index=[1, 2, 3])
    # 断言结果与期望是否相等，并且结果的数据类型是整数类型
    assert issubclass(result.dtype.type, np.integer)
    tm.assert_series_equal(result, expected)

    # -----------------------------------------------------------
    # 对象
    objects = Series([1, 2, 3], dtype=object)

    result = objects.reindex([1, 2, 3])
    # 创建一个期望的 Series，包含填充后的结果，索引为 [1, 2, 3]，填充值为 NaN，数据类型为对象
    expected = Series([2, 3, np.nan], index=[1, 2, 3], dtype=object)
    # 断言结果与期望是否相等
    tm.assert_series_equal(result, expected)

    result = objects.reindex([1, 2, 3], fill_value="foo")
    # 创建一个期望的 Series，包含填充后的结果，索引为 [1, 2, 3]，填充值为 "foo"，数据类型为对象
    expected = Series([2, 3, "foo"], index=[1, 2, 3], dtype=object)
    # 断言结果与期望是否相等
    tm.assert_series_equal(result, expected)

    # ------------------------------------------------------------
    # 布尔值
    bools = Series([True, False, True])

    result = bools.reindex([1, 2, 3])
    # 创建一个预期的 Series 对象，包含布尔值和 NaN，索引为指定的整数列表，数据类型为 object
    expected = Series([False, True, np.nan], index=[1, 2, 3], dtype=object)
    # 使用测试工具模块 tm 来比较 result 和预期的 Series 对象是否相等
    tm.assert_series_equal(result, expected)
    
    # 对布尔值 Series 对象进行重新索引，索引为指定的整数列表，填充值为 False
    result = bools.reindex([1, 2, 3], fill_value=False)
    # 创建一个预期的 Series 对象，包含布尔值，索引为指定的整数列表
    expected = Series([False, True, False], index=[1, 2, 3])
    # 使用测试工具模块 tm 来比较 result 和预期的 Series 对象是否相等
    tm.assert_series_equal(result, expected)
@pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[ns]"])
# 参数化测试，针对不同的dtype进行测试
@pytest.mark.parametrize("fill_value", ["string", 0, Timedelta(0)])
# 参数化测试，针对不同的fill_value进行测试，包括字符串、整数0和Timedelta对象

def test_reindex_fill_value_datetimelike_upcast(dtype, fill_value):
    # 测试函数，验证填充值对于日期时间类数据类型的提升行为
    # 如果dtype是"timedelta64[ns]"并且fill_value是Timedelta(0)，则进行下面的替换
    if dtype == "timedelta64[ns]" and fill_value == Timedelta(0):
        # 使用与该dtype不兼容的标量进行测试
        fill_value = Timestamp(0)

    # 创建一个包含NaT的Series对象，指定dtype
    ser = Series([NaT], dtype=dtype)

    # 对Series进行重新索引，填充值为fill_value
    result = ser.reindex([0, 1], fill_value=fill_value)
    # 期望的结果Series，填充值根据输入的fill_value决定
    expected = Series([NaT, fill_value], index=[0, 1], dtype=object)
    # 使用测试工具tm.assert_series_equal进行结果验证
    tm.assert_series_equal(result, expected)


def test_reindex_datetimeindexes_tz_naive_and_aware():
    # 测试函数，验证在不同时区情况下的日期时间索引的重新索引行为
    # 创建一个带时区的日期时间索引
    idx = date_range("20131101", tz="America/Chicago", periods=7)
    # 创建一个新的日期时间索引
    newidx = date_range("20131103", periods=10, freq="h")
    # 创建一个Series对象，使用带时区的日期时间索引
    s = Series(range(7), index=idx)
    # 预期产生TypeError异常，匹配特定的错误消息
    msg = (
        r"Cannot compare dtypes datetime64\[ns, America/Chicago\] "
        r"and datetime64\[ns\]"
    )
    with pytest.raises(TypeError, match=msg):
        # 对Series进行重新索引，使用ffill方法
        s.reindex(newidx, method="ffill")


def test_reindex_empty_series_tz_dtype():
    # 测试函数，验证空Series对象的重新索引行为，带有特定的dtype
    # 通过指定dtype创建一个空的Series对象
    result = Series(dtype="datetime64[ns, UTC]").reindex([0, 1])
    # 预期的结果Series，填充值为NaT
    expected = Series([NaT] * 2, dtype="datetime64[ns, UTC]")
    # 使用测试工具tm.assert_equal进行结果验证
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "p_values, o_values, values, expected_values",
    [
        (
            [Period("2019Q1", "Q-DEC"), Period("2019Q2", "Q-DEC")],
            [Period("2019Q1", "Q-DEC"), Period("2019Q2", "Q-DEC"), "All"],
            [1.0, 1.0],
            [1.0, 1.0, np.nan],
        ),
        (
            [Period("2019Q1", "Q-DEC"), Period("2019Q2", "Q-DEC")],
            [Period("2019Q1", "Q-DEC"), Period("2019Q2", "Q-DEC")],
            [1.0, 1.0],
            [1.0, 1.0],
        ),
    ],
)
# 参数化测试，验证在PeriodIndex与Index对象之间重新索引行为的正确性
def test_reindex_periodindex_with_object(p_values, o_values, values, expected_values):
    # 创建一个PeriodIndex对象
    period_index = PeriodIndex(p_values)
    # 创建一个Index对象
    object_index = Index(o_values)

    # 创建一个Series对象，使用PeriodIndex作为索引
    ser = Series(values, index=period_index)
    # 对Series进行重新索引，使用Index对象作为索引
    result = ser.reindex(object_index)
    # 预期的结果Series，使用Index对象作为索引
    expected = Series(expected_values, index=object_index)
    # 使用测试工具tm.assert_series_equal进行结果验证
    tm.assert_series_equal(result, expected)


def test_reindex_too_many_args():
    # 测试函数，验证当传递多余的参数给reindex函数时的行为
    # 创建一个Series对象
    ser = Series([1, 2])
    # 预期产生TypeError异常，匹配特定的错误消息
    msg = r"reindex\(\) takes from 1 to 2 positional arguments but 3 were given"
    with pytest.raises(TypeError, match=msg):
        # 对Series进行重新索引，传递了3个参数而不是1或2个
        ser.reindex([2, 3], False)


def test_reindex_double_index():
    # 测试函数，验证当传递重复的索引参数给reindex函数时的行为
    # 创建一个Series对象
    ser = Series([1, 2])
    # 预期产生TypeError异常，匹配特定的错误消息
    msg = r"reindex\(\) got multiple values for argument 'index'"
    with pytest.raises(TypeError, match=msg):
        # 对Series进行重新索引，传递了重复的索引参数
        ser.reindex([2, 3], index=[3, 4])


def test_reindex_no_posargs():
    # 测试函数，验证当不传递位置参数给reindex函数时的行为
    # 创建一个Series对象
    ser = Series([1, 2])
    # 对Series进行重新索引，传递了索引参数但没有位置参数
    result = ser.reindex(index=[1, 0])
    # 预期的结果Series，索引顺序被调整
    expected = Series([2, 1], index=[1, 0])
    # 使用测试工具tm.assert_series_equal进行结果验证
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("values", [[["a"], ["x"]], [[], []]])
# 参数化测试，针对不同的values进行测试
def test_reindex_empty_with_level(values):
    # 测试函数，验证在多级索引为空时的重新索引行为
    # GH41170
    # 创建一个 Pandas Series 对象
    ser = Series(
        # 用从 values 数组创建的 MultiIndex 作为索引，索引标签类型为 'object'
        range(len(values[0])), index=MultiIndex.from_arrays(values), dtype="object"
    )
    # 使用 reindex 方法重新索引 Series，保留 level 0 的索引标签为 "b"，其他标签会被丢弃
    result = ser.reindex(np.array(["b"]), level=0)
    # 创建一个预期的 Pandas Series 对象，其索引为 MultiIndex，第一级索引为 "b"，第二级索引为 values[1]
    expected = Series(
        index=MultiIndex(levels=[["b"], values[1]], codes=[[], []]), dtype="object"
    )
    # 使用测试工具 tm.assert_series_equal 比较 result 和 expected 两个 Series 对象
    tm.assert_series_equal(result, expected)
# 测试函数：测试重新索引时遇到缺失类别的情况
def test_reindex_missing_category():
    # 标识 GitHub 问题号 18185
    # 创建一个包含整数的 Series 对象，并指定数据类型为“category”
    ser = Series([1, 2, 3, 1], dtype="category")
    # 设置错误消息的正则表达式，用于检查 TypeError 异常
    msg = r"Cannot setitem on a Categorical with a new category \(-1\)"
    # 使用 pytest 检查是否会引发 TypeError 异常，且异常消息符合预期
    with pytest.raises(TypeError, match=msg):
        # 对 Series 对象进行重新索引，填充值为 -1
        ser.reindex([1, 2, 3, 4, 5], fill_value=-1)


# 测试函数：测试使用 Float64Dtype 类型的 Series 对象进行重新索引
def test_reindexing_with_float64_NA_log():
    # 标识 GitHub 问题号 47055
    # 创建一个包含浮点数和缺失值 NA 的 Series 对象，数据类型为 Float64Dtype
    s = Series([1.0, NA], dtype=Float64Dtype())
    # 对 Series 对象进行重新索引，指定新的索引范围为 0 到 2
    s_reindex = s.reindex(range(3))
    # 获取重新索引后的值的原始数据
    result = s_reindex.values._data
    # 期望的结果数组，包含了 1、NaN、NaN
    expected = np.array([1, np.nan, np.nan])
    # 使用测试模块 tm 检查两个数组是否相等
    tm.assert_numpy_array_equal(result, expected)
    # 使用 tm.assert_produces_warning(None) 确保在以下代码段中不会引发警告
    with tm.assert_produces_warning(None):
        # 对重新索引后的 Series 对象取对数
        result_log = np.log(s_reindex)
        # 期望的对数结果 Series 对象，包含了 0、NaN、NaN，数据类型为 Float64Dtype
        expected_log = Series([0, np.nan, np.nan], dtype=Float64Dtype())
        # 使用 tm.assert_series_equal 检查两个 Series 对象是否相等
        tm.assert_series_equal(result_log, expected_log)


# 测试函数：使用指定的 dtype 进行重新索引，扩展非纳秒级（nonnano）时间和时间间隔
@pytest.mark.parametrize("dtype", ["timedelta64", "datetime64"])
def test_reindex_expand_nonnano_nat(dtype):
    # 标识 GitHub 问题号 53497
    # 创建一个包含特定 dtype 的 Series 对象，其值为数组 [1]，dtype 由参数 dtype 决定
    ser = Series(np.array([1], dtype=f"{dtype}[s]"))
    # 对 Series 对象进行重新索引，新的索引范围为 RangeIndex(2)
    result = ser.reindex(RangeIndex(2))
    # 期望的结果 Series 对象，包含了 [1, np.<dtype>('nat', 's')] 的数组
    expected = Series(
        np.array([1, getattr(np, dtype)("nat", "s")], dtype=f"{dtype}[s]")
    )
    # 使用 tm.assert_series_equal 检查两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)
```