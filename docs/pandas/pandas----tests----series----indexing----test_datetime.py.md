# `D:\src\scipysrc\pandas\pandas\tests\series\indexing\test_datetime.py`

```
# 引入所需的库和模块

from datetime import (
    datetime,  # 导入datetime类，用于处理日期时间
    timedelta,  # 导入timedelta类，用于处理时间间隔
)
import re  # 导入re模块，用于正则表达式操作

from dateutil.tz import (
    gettz,  # 导入gettz函数，用于获取时区对象
    tzutc,  # 导入tzutc函数，用于表示UTC时区对象
)
import numpy as np  # 导入numpy库，用于数值计算
import pytest  # 导入pytest库，用于单元测试框架

from pandas._libs import index as libindex  # 导入pandas内部索引库

import pandas as pd  # 导入pandas库，用于数据分析
from pandas import (
    DataFrame,  # 导入DataFrame类，用于处理二维数据
    Series,  # 导入Series类，用于处理一维数据
    Timestamp,  # 导入Timestamp类，用于处理时间戳
    date_range,  # 导入date_range函数，用于生成日期范围
    period_range,  # 导入period_range函数，用于生成周期范围
)
import pandas._testing as tm  # 导入pandas内部测试工具模块


def test_fancy_getitem():
    # 创建一个日期范围，频率为每月第一个星期五，从2005年1月1日到2010年1月1日
    dti = date_range(
        freq="WOM-1FRI", start=datetime(2005, 1, 1), end=datetime(2010, 1, 1)
    )

    # 创建一个Series对象，索引为上述日期范围，值为索引位置
    s = Series(np.arange(len(dti)), index=dti)

    # 断言：通过不同的方式获取索引为"1/2/2009"的值是否为48
    assert s["1/2/2009"] == 48
    assert s["2009-1-2"] == 48
    assert s[datetime(2009, 1, 2)] == 48
    assert s[Timestamp(datetime(2009, 1, 2))] == 48

    # 使用pytest验证索引为"2009-1-3"的情况是否会引发KeyError异常
    with pytest.raises(KeyError, match=r"^'2009-1-3'$"):
        s["2009-1-3"]

    # 使用pandas测试工具tm，验证切片操作的结果是否相等
    tm.assert_series_equal(
        s["3/6/2009":"2009-06-05"], s[datetime(2009, 3, 6) : datetime(2009, 6, 5)]
    )


def test_fancy_setitem():
    # 创建一个日期范围，频率为每月第一个星期五，从2005年1月1日到2010年1月1日
    dti = date_range(
        freq="WOM-1FRI", start=datetime(2005, 1, 1), end=datetime(2010, 1, 1)
    )

    # 创建一个Series对象，索引为上述日期范围，值为索引位置
    s = Series(np.arange(len(dti)), index=dti)

    # 设置索引为"1/2/2009"的值为-2，验证是否设置成功
    s["1/2/2009"] = -2
    assert s.iloc[48] == -2

    # 将索引为"1/2/2009"到"2009-06-05"的值设置为-3，验证是否设置成功
    s["1/2/2009":"2009-06-05"] = -3
    assert (s[48:54] == -3).all()


@pytest.mark.parametrize("tz_source", ["pytz", "dateutil"])
def test_getitem_setitem_datetime_tz(tz_source):
    if tz_source == "pytz":
        # 导入pytz库，如果不存在则跳过测试
        pytz = pytest.importorskip(tz_source)
        tzget = pytz.timezone  # 设置时区获取函数为pytz.timezone
    else:
        # 对于dateutil的特殊处理，设置时区获取函数为gettz，并处理UTC时区
        tzget = lambda x: tzutc() if x == "UTC" else gettz(x)

    N = 50
    # 创建一个具有时区信息的日期范围，频率为每小时，从1990年1月1日开始，共50个时间点，时区为US/Eastern
    rng = date_range("1/1/1990", periods=N, freq="h", tz=tzget("US/Eastern"))
    ts = Series(np.random.default_rng(2).standard_normal(N), index=rng)

    # 测试Timestamp带时区信息的处理，验证设置和获取操作是否正确
    result = ts.copy()
    result["1990-01-01 09:00:00+00:00"] = 0
    result["1990-01-01 09:00:00+00:00"] = ts.iloc[4]
    tm.assert_series_equal(result, ts)

    result = ts.copy()
    result["1990-01-01 03:00:00-06:00"] = 0
    result["1990-01-01 03:00:00-06:00"] = ts.iloc[4]
    tm.assert_series_equal(result, ts)

    # 使用datetime对象进行设置和获取操作，验证结果是否正确
    result = ts.copy()
    result[datetime(1990, 1, 1, 9, tzinfo=tzget("UTC"))] = 0
    result[datetime(1990, 1, 1, 9, tzinfo=tzget("UTC"))] = ts.iloc[4]
    tm.assert_series_equal(result, ts)

    result = ts.copy()
    dt = Timestamp(1990, 1, 1, 3).tz_localize(tzget("US/Central"))
    dt = dt.to_pydatetime()
    result[dt] = 0
    result[dt] = ts.iloc[4]
    tm.assert_series_equal(result, ts)


def test_getitem_setitem_datetimeindex():
    N = 50
    # 创建一个具有时区信息的日期范围，频率为每小时，从1990年1月1日开始，共50个时间点，时区为US/Eastern
    rng = date_range("1/1/1990", periods=N, freq="h", tz="US/Eastern")
    ts = Series(np.random.default_rng(2).standard_normal(N), index=rng)

    # 验证使用字符串格式的时间戳获取数据的正确性
    result = ts["1990-01-01 04:00:00"]
    expected = ts.iloc[4]
    assert result == expected

    # 验证使用字符串格式的时间戳设置数据的正确性
    result = ts.copy()
    result["1990-01-01 04:00:00"] = 0
    # 将时间序列中 "1990-01-01 04:00:00" 对应的值设置为 ts 的第五个元素的值
    result["1990-01-01 04:00:00"] = ts.iloc[4]
    # 断言两个时间序列是否相等
    tm.assert_series_equal(result, ts)

    # 从时间序列中选择时间范围为 "1990-01-01 04:00:00" 到 "1990-01-01 07:00:00" 的子序列
    result = ts["1990-01-01 04:00:00":"1990-01-01 07:00:00"]
    # 期望的结果是时间序列 ts 中从索引位置 4 到 7 的子序列
    expected = ts[4:8]
    # 断言两个时间序列是否相等
    tm.assert_series_equal(result, expected)

    # 复制时间序列 ts 到 result
    result = ts.copy()
    # 将时间序列中 "1990-01-01 04:00:00" 到 "1990-01-01 07:00:00" 的值设为 0
    result["1990-01-01 04:00:00":"1990-01-01 07:00:00"] = 0
    # 将时间序列中 "1990-01-01 04:00:00" 到 "1990-01-01 07:00:00" 的值设置为 ts 中从索引位置 4 到 7 的值
    result["1990-01-01 04:00:00":"1990-01-01 07:00:00"] = ts[4:8]
    # 断言两个时间序列是否相等
    tm.assert_series_equal(result, ts)

    # 定义左边界 lb 和右边界 rb
    lb = "1990-01-01 04:00:00"
    rb = "1990-01-01 07:00:00"
    # 使用 lb 和 rb 来筛选时间序列 ts 中符合条件的数据
    result = ts[(ts.index >= lb) & (ts.index <= rb)]
    # 期望的结果是时间序列 ts 中从索引位置 4 到 7 的子序列
    expected = ts[4:8]
    # 断言两个时间序列是否相等
    tm.assert_series_equal(result, expected)

    # 定义左边界 lb 和右边界 rb，带有时区信息
    lb = "1990-01-01 04:00:00-0500"
    rb = "1990-01-01 07:00:00-0500"
    # 使用 lb 和 rb 来筛选时间序列 ts 中符合条件的数据
    result = ts[(ts.index >= lb) & (ts.index <= rb)]
    # 期望的结果是时间序列 ts 中从索引位置 4 到 7 的子序列
    expected = ts[4:8]
    # 断言两个时间序列是否相等
    tm.assert_series_equal(result, expected)

    # 定义错误消息字符串
    msg = "Cannot compare tz-naive and tz-aware datetime-like objects"
    # 创建 naive datetime 对象
    naive = datetime(1990, 1, 1, 4)
    # 迭代处理 naive 对象、Timestamp 包装的 naive 对象和 np.datetime64(ns) 格式的 naive 对象
    for key in [naive, Timestamp(naive), np.datetime64(naive, "ns")]:
        # 断言尝试访问时间序列 ts 中不存在的键会引发 KeyError 异常，并且错误消息匹配预期的消息
        with pytest.raises(KeyError, match=re.escape(repr(key))):
            ts[key]

    # 复制时间序列 ts 到 result
    result = ts.copy()
    # 将时间序列中 naive 对应的值设为 ts 的第五个元素的值
    result[naive] = ts.iloc[4]
    # 断言 result 的索引数据类型为 object
    assert result.index.dtype == object
    # 断言 result 的除最后一个索引外的所有索引与 rng 转换为 object 类型后的索引相等
    tm.assert_index_equal(result.index[:-1], rng.astype(object))
    # 断言 result 的最后一个索引与 naive 相等
    assert result.index[-1] == naive

    # 定义错误消息字符串
    msg = "Cannot compare tz-naive and tz-aware datetime-like objects"
    # 断言尝试访问时间序列 ts 中不符合时区兼容性要求的时间范围时，会引发 TypeError 异常，并且错误消息匹配预期的消息
    with pytest.raises(TypeError, match=msg):
        ts[naive : datetime(1990, 1, 1, 7)]

    # 复制时间序列 ts 到 result
    result = ts.copy()
    # 断言尝试在时间序列 result 中设置不符合时区兼容性要求的时间范围时，会引发 TypeError 异常，并且错误消息匹配预期的消息
    with pytest.raises(TypeError, match=msg):
        result[naive : datetime(1990, 1, 1, 7)] = 0
    # 断言尝试在时间序列 result 中设置不符合时区兼容性要求的时间范围时，会引发 TypeError 异常，并且错误消息匹配预期的消息
    with pytest.raises(TypeError, match=msg):
        result[naive : datetime(1990, 1, 1, 7)] = 99
    # 由于设置项目失败，因此 result 应该与 ts 相匹配
    tm.assert_series_equal(result, ts)

    # 定义左边界 lb 和右边界 rb
    lb = naive
    rb = datetime(1990, 1, 1, 7)
    # 定义错误消息字符串
    msg = r"Invalid comparison between dtype=datetime64\[ns, US/Eastern\] and datetime"
    # 断言尝试比较不符合时区兼容性要求的时间范围时，会引发 TypeError 异常，并且错误消息匹配预期的消息
    with pytest.raises(TypeError, match=msg):
        ts[(ts.index >= lb) & (ts.index <= rb)]

    # 为左边界 lb 和右边界 rb 添加时区信息后，筛选时间序列 ts 中符合条件的数据
    lb = Timestamp(naive).tz_localize(rng.tzinfo)
    rb = Timestamp(datetime(1990, 1, 1, 7)).tz_localize(rng.tzinfo)
    result = ts[(ts.index >= lb) & (ts.index <= rb)]
    # 期望的结果是时间序列 ts 中从索引位置 4 到 7 的子序列
    expected = ts[4:8]
    # 断言两个时间序列是否相等
    tm.assert_series_equal(result, expected)

    # 从时间序列 ts 中选择索引位置为 4 的单个元素
    result = ts[ts.index[4]]
    # 期望的结果是时间序列 ts 中索引位置为 4 的元素
    expected = ts.iloc[4]
    # 断言结果与期望相等
    assert result == expected

    # 从时间序列 ts 中选择索引位置从 4 到 7 的子序列
    result = ts[ts.index[4:8]]
    # 期望的结果是时间序列 ts 中从索引位置 4 到 7 的子序列
    expected = ts[4:8]
    # 断言两个时间序列是否相等
    tm.assert_series_equal(result, expected)
    # 复制时间序列 `ts` 并赋值给 `result`
    result = ts.copy()
    
    # 将 `result` 中索引从第5到第8个（不包括第8个）的位置置为0
    result[ts.index[4:8]] = 0
    
    # 使用位置索引在 `result` 中的第5到第8行（不包括第8行）替换为 `ts` 中对应位置的数据
    result.iloc[4:8] = ts.iloc[4:8]
    
    # 断言 `result` 和 `ts` 序列相等
    tm.assert_series_equal(result, ts)
    
    # 也测试部分日期切片的情况
    # 从时间序列 `ts` 中取出日期为 "1990-01-02" 的数据并赋给 `result`
    result = ts["1990-01-02"]
    
    # 期望的结果是从 `ts` 中索引从24到48的数据
    expected = ts[24:48]
    
    # 断言 `result` 和 `expected` 序列相等
    tm.assert_series_equal(result, expected)
    
    # 复制时间序列 `ts` 并赋值给 `result`
    result = ts.copy()
    
    # 将 `result` 中日期为 "1990-01-02" 的位置置为0
    result["1990-01-02"] = 0
    
    # 将 `ts` 中索引从24到48的数据赋给 `result` 中日期为 "1990-01-02" 的位置
    result["1990-01-02"] = ts[24:48]
    
    # 断言 `result` 和 `ts` 序列相等
    tm.assert_series_equal(result, ts)
# 定义一个测试函数，用于测试基于 PeriodIndex 的索引和赋值操作
def test_getitem_setitem_periodindex():
    # 设定时间范围，生成 PeriodIndex，频率为每小时
    N = 50
    rng = period_range("1/1/1990", periods=N, freq="h")
    # 创建一个 Series 对象，其值为 N 个标准正态分布随机数，索引为 rng
    ts = Series(np.random.default_rng(2).standard_normal(N), index=rng)

    # 测试单个时间点的索引
    result = ts["1990-01-01 04"]
    expected = ts.iloc[4]
    assert result == expected

    # 复制 Series 对象
    result = ts.copy()
    # 对特定时间点赋值为 0
    result["1990-01-01 04"] = 0
    # 再将其值设置为原始 Series 对象中对应索引位置的值
    result["1990-01-01 04"] = ts.iloc[4]
    # 断言两个 Series 对象相等
    tm.assert_series_equal(result, ts)

    # 测试时间范围内的切片索引
    result = ts["1990-01-01 04":"1990-01-01 07"]
    expected = ts[4:8]
    tm.assert_series_equal(result, expected)

    # 复制 Series 对象
    result = ts.copy()
    # 将特定时间范围内的值赋值为 0
    result["1990-01-01 04":"1990-01-01 07"] = 0
    # 再将其值设置为原始 Series 对象中对应切片范围的值
    result["1990-01-01 04":"1990-01-01 07"] = ts[4:8]
    # 断言两个 Series 对象相等
    tm.assert_series_equal(result, ts)

    # 设定左右边界时间点
    lb = "1990-01-01 04"
    rb = "1990-01-01 07"
    # 使用复杂条件对索引进行筛选
    result = ts[(ts.index >= lb) & (ts.index <= rb)]
    expected = ts[4:8]
    tm.assert_series_equal(result, expected)

    # GH 2782
    # 通过索引访问特定位置的值
    result = ts[ts.index[4]]
    expected = ts.iloc[4]
    assert result == expected

    # 通过索引访问切片范围的值
    result = ts[ts.index[4:8]]
    expected = ts[4:8]
    tm.assert_series_equal(result, expected)

    # 复制 Series 对象
    result = ts.copy()
    # 将特定索引切片范围的值赋值为 0
    result[ts.index[4:8]] = 0
    # 再将其值设置为原始 Series 对象中对应切片范围的值
    result.iloc[4:8] = ts.iloc[4:8]
    # 断言两个 Series 对象相等
    tm.assert_series_equal(result, ts)


# 测试 DateTime 索引操作
def test_datetime_indexing():
    # 创建日期范围索引
    index = date_range("1/1/2000", "1/7/2000")
    # 将索引重复 3 次
    index = index.repeat(3)

    # 创建 Series 对象，长度与索引相同
    s = Series(len(index), index=index)
    # 创建一个时间戳
    stamp = Timestamp("1/8/2000")

    # 使用 pytest 断言捕获 KeyError 异常，验证时间戳不存在时的索引行为
    with pytest.raises(KeyError, match=re.escape(repr(stamp))):
        s[stamp]
    # 将时间戳对应位置的值赋值为 0
    s[stamp] = 0
    # 断言时间戳位置的值为 0
    assert s[stamp] == 0

    # 创建 Series 对象，并反转索引顺序
    s = Series(len(index), index=index)
    s = s[::-1]

    # 使用 pytest 断言捕获 KeyError 异常，验证时间戳不存在时的索引行为
    with pytest.raises(KeyError, match=re.escape(repr(stamp))):
        s[stamp]
    # 将时间戳对应位置的值赋值为 0
    s[stamp] = 0
    # 断言时间戳位置的值为 0
    assert s[stamp] == 0


# 测试时间序列中的重复日期索引
def test_indexing_with_duplicate_datetimeindex(
    rand_series_with_duplicate_datetimeindex,
):
    # 获取具有重复日期索引的随机时间序列
    ts = rand_series_with_duplicate_datetimeindex

    # 获取索引中的唯一日期
    uniques = ts.index.unique()
    for date in uniques:
        # 测试单个日期的索引
        result = ts[date]

        # 创建一个布尔掩码，用于筛选特定日期的值
        mask = ts.index == date
        # 计算特定日期的值出现次数
        total = (ts.index == date).sum()
        # 根据值出现次数选择预期结果
        expected = ts[mask]
        if total > 1:
            tm.assert_series_equal(result, expected)
        else:
            tm.assert_almost_equal(result, expected.iloc[0])

        # 复制时间序列对象
        cp = ts.copy()
        # 将特定日期的值赋值为 0
        cp[date] = 0
        # 创建预期结果的 Series 对象，将特定日期的值设置为 0
        expected = Series(np.where(mask, 0, ts), index=ts.index)
        tm.assert_series_equal(cp, expected)

    # 创建一个新的日期索引
    key = datetime(2000, 1, 6)
    # 使用 pytest 断言捕获 KeyError 异常，验证不存在的日期索引行为
    with pytest.raises(KeyError, match=re.escape(repr(key))):
        ts[key]

    # 在时间序列中添加新的日期索引及其对应的值
    ts[datetime(2000, 1, 6)] = 0
    # 断言新添加的日期索引位置的值为 0
    assert ts[datetime(2000, 1, 6)] == 0


# 测试超过大小截断的 loc 和 getitem 操作
def test_loc_getitem_over_size_cutoff(monkeypatch):
    # #1821

    # 使用 monkeypatch 设置 libindex 模块中的 _SIZE_CUTOFF 属性为 1000
    monkeypatch.setattr(libindex, "_SIZE_CUTOFF", 1000)

    # 创建大量非周期性日期时间的列表
    dates = []
    sec = timedelta(seconds=1)
    half_sec = timedelta(microseconds=500000)
    d = datetime(2011, 12, 5, 20, 30)
    n = 1100
    # 使用循环生成日期列表，每次迭代增加不同的时间间隔
    for i in range(n):
        dates.append(d)                         # 将当前日期d添加到列表中
        dates.append(d + sec)                   # 添加当前日期d加上sec秒后的日期
        dates.append(d + sec + half_sec)        # 添加当前日期d加上sec秒和half_sec秒后的日期
        dates.append(d + sec + sec + half_sec)  # 添加当前日期d加上sec秒、sec秒和half_sec秒后的日期
        d += 3 * sec                            # 更新日期d，增加3倍的sec秒

    # 在日期列表中随机选择位置，将选中位置后的日期重复复制
    duplicate_positions = np.random.default_rng(2).integers(0, len(dates) - 1, 20)
    for p in duplicate_positions:
        dates[p + 1] = dates[p]                # 将位置p后的日期复制到位置p+1处

    # 使用随机数生成DataFrame，索引为日期列表dates，列为'A', 'B', 'C', 'D'，数据为标准正态分布随机数
    df = DataFrame(
        np.random.default_rng(2).standard_normal((len(dates), 4)),
        index=dates,
        columns=list("ABCD"),
    )

    # 计算特定位置的时间戳
    pos = n * 3
    timestamp = df.index[pos]
    # 断言特定时间戳在DataFrame的索引中存在
    assert timestamp in df.index

    # 使用.loc方法获取特定时间戳的行，并断言结果不为空
    df.loc[timestamp]
    assert len(df.loc[[timestamp]]) > 0
def test_indexing_over_size_cutoff_period_index(monkeypatch):
    # GH 27136
    # 设置 _SIZE_CUTOFF 属性为 1000
    monkeypatch.setattr(libindex, "_SIZE_CUTOFF", 1000)

    # 设置 periods 的数量为 1100，生成一个日期范围索引
    n = 1100
    idx = period_range("1/1/2000", freq="min", periods=n)
    
    # 断言 idx._engine.over_size_threshold 是 True
    assert idx._engine.over_size_threshold

    # 创建一个 Series，其值为随机标准正态分布数列，索引为 idx
    s = Series(np.random.default_rng(2).standard_normal(len(idx)), index=idx)

    # 设置 pos 为 n - 1，获取该索引位置的时间戳
    pos = n - 1
    timestamp = idx[pos]
    
    # 断言 timestamp 在 s 的索引中
    assert timestamp in s.index

    # 访问 s 中时间戳为 timestamp 的数据
    s[timestamp]
    
    # 断言 s.loc[[timestamp]] 的长度大于 0
    assert len(s.loc[[timestamp]]) > 0


def test_indexing_unordered():
    # GH 2437
    # 创建一个日期范围索引 rng，从 "2011-01-01" 到 "2011-01-15"
    rng = date_range(start="2011-01-01", end="2011-01-15")
    
    # 创建一个 Series ts，其值为随机数列，索引为 rng
    ts = Series(np.random.default_rng(2).random(len(rng)), index=rng)
    
    # 创建一个乱序的 Series ts2，包含 ts 的子集
    ts2 = pd.concat([ts[0:4], ts[-4:], ts[4:-4]])

    # 遍历 ts 的索引
    for t in ts.index:
        # 获取 ts 中 t 索引位置的值
        expected = ts[t]
        # 获取 ts2 中 t 索引位置的值
        result = ts2[t]
        
        # 断言 ts 和 ts2 中 t 索引位置的值相等
        assert expected == result

    # 定义一个比较函数 compare
    def compare(slobj):
        # 对 ts2 中 slobj 的子集进行排序
        result = ts2[slobj].copy()
        result = result.sort_index()
        
        # 获取 ts 中 slobj 的子集
        expected = ts[slobj]
        # 将 expected 的索引频率设为 None
        expected.index = expected.index._with_freq(None)
        
        # 使用 tm.assert_series_equal 比较 result 和 expected
        tm.assert_series_equal(result, expected)

    # 遍历不同的切片 key
    for key in [
        slice("2011-01-01", "2011-01-15"),
        slice("2010-12-30", "2011-01-15"),
        slice("2011-01-01", "2011-01-16"),
        # 部分范围
        slice("2011-01-01", "2011-01-6"),
        slice("2011-01-06", "2011-01-8"),
        slice("2011-01-06", "2011-01-12"),
    ]:
        # 使用 pytest.raises 断言会抛出 KeyError，并包含特定的错误信息
        with pytest.raises(
            KeyError, match="Value based partial slicing on non-monotonic"
        ):
            compare(key)

    # 获取 ts2 中 "2011" 年份的子集，并进行排序
    result = ts2["2011"].sort_index()
    
    # 获取 ts 中 "2011" 年份的子集
    expected = ts["2011"]
    # 将 expected 的索引频率设为 None
    expected.index = expected.index._with_freq(None)
    
    # 使用 tm.assert_series_equal 比较 result 和 expected
    tm.assert_series_equal(result, expected)


def test_indexing_unordered2():
    # diff freq
    # 创建一个日期范围索引 rng，从 datetime(2005, 1, 1) 开始，20 个周期，频率为 "ME"
    rng = date_range(datetime(2005, 1, 1), periods=20, freq="ME")
    
    # 创建一个 Series ts，其值为 0 到 19 的整数，索引为 rng，并随机乱序
    ts = Series(np.arange(len(rng)), index=rng)
    ts = ts.take(np.random.default_rng(2).permutation(20))

    # 获取 ts 中 "2005" 年份的子集
    result = ts["2005"]
    
    # 遍历 result 的索引
    for t in result.index:
        # 断言 t 的年份为 2005
        assert t.year == 2005


def test_indexing():
    # 创建一个日期范围索引 idx，从 "2001-1-1" 开始，20 个周期，频率为 "ME"
    idx = date_range("2001-1-1", periods=20, freq="ME")
    
    # 创建一个 Series ts，其值为随机数列，索引为 idx
    ts = Series(np.random.default_rng(2).random(len(idx)), index=idx)

    # 获取 "2001" 年份的子集
    result = ts["2001"]
    
    # 使用 tm.assert_series_equal 比较 result 和 ts 的前 12 个元素
    tm.assert_series_equal(result, ts.iloc[:12])

    # 创建一个 DataFrame df，包含一个列 "A"，其值为 ts
    df = DataFrame({"A": ts})

    # 在 2.0 之前，df["2001"] 行为是对行进行切片操作，现在应抛出 KeyError
    with pytest.raises(KeyError, match="2001"):
        df["2001"]

    # 设置 ts 为一个新的 Series，其值为随机数列，索引为 idx
    ts = Series(np.random.default_rng(2).random(len(idx)), index=idx)
    expected = ts.copy()

    # 将 expected 的前 12 个元素设置为 1
    expected.iloc[:12] = 1
    
    # 将 ts 中 "2001" 年份的子集设置为 1
    ts["2001"] = 1
    
    # 使用 tm.assert_series_equal 比较 ts 和 expected
    tm.assert_series_equal(ts, expected)

    expected = df.copy()

    # 将 expected 的前 12 行第 0 列设置为 1
    expected.iloc[:12, 0] = 1
    
    # 将 df 中 "2001" 行的 "A" 列设置为 1
    df.loc["2001", "A"] = 1
    
    # 使用 tm.assert_frame_equal 比较 df 和 expected
    tm.assert_frame_equal(df, expected)


def test_getitem_str_month_with_datetimeindex():
    # GH3546 (not including times on the last day)
    # 创建一个时间范围，从 "2013-05-31 00:00" 到 "2013-05-31 23:00"，每小时频率
    idx = date_range(start="2013-05-31 00:00", end="2013-05-31 23:00", freq="h")
    # 创建一个时间序列，其索引为上面创建的时间范围，值为索引位置的连续整数
    ts = Series(range(len(idx)), index=idx)
    # 从时间序列中选择 "2013-05" 这个月份的数据作为期望值
    expected = ts["2013-05"]
    # 使用测试工具比较期望值和时间序列，确保它们相等
    tm.assert_series_equal(expected, ts)
    
    # 创建一个时间范围，从 "2013-05-31 00:00" 到 "2013-05-31 23:59"，每秒频率
    idx = date_range(start="2013-05-31 00:00", end="2013-05-31 23:59", freq="s")
    # 创建一个时间序列，其索引为上面创建的时间范围，值为索引位置的连续整数
    ts = Series(range(len(idx)), index=idx)
    # 从时间序列中选择 "2013-05" 这个月份的数据作为期望值
    expected = ts["2013-05"]
    # 使用测试工具比较期望值和时间序列，确保它们相等
    tm.assert_series_equal(expected, ts)
# 定义一个测试函数，测试从具有 DateTimeIndex 的 Series 中获取带有年份字符串的元素
def test_getitem_str_year_with_datetimeindex():
    # 创建一个时间戳索引列表，包含两个时间戳对象
    idx = [
        Timestamp("2013-05-31 00:00"),
        Timestamp(datetime(2013, 5, 31, 23, 59, 59, 999999)),
    ]
    # 根据索引创建一个 Series 对象，其值为索引的位置
    ts = Series(range(len(idx)), index=idx)
    # 期望的结果是根据年份字符串索引获取的 Series 对象
    expected = ts["2013"]
    # 使用测试框架验证期望结果与实际结果是否相等
    tm.assert_series_equal(expected, ts)


# 定义一个测试函数，测试从具有 DateTimeIndex 的 DataFrame 中获取带有秒级精度字符串的元素
def test_getitem_str_second_with_datetimeindex():
    # GH14826，用秒为单位的字符串/日期时间对象进行索引
    # 创建一个 5x5 的随机数 DataFrame
    df = DataFrame(
        np.random.default_rng(2).random((5, 5)),
        columns=["open", "high", "low", "close", "volume"],
        # 创建一个带有时区信息的日期范围索引
        index=date_range("2012-01-02 18:01:00", periods=5, tz="US/Central", freq="s"),
    )

    # 这是一个单独的日期时间，因此将引发 KeyError 异常
    with pytest.raises(KeyError, match=r"^'2012-01-02 18:01:02'$"):
        df["2012-01-02 18:01:02"]

    # 期望的异常消息，包含特定格式的时间戳和时区信息
    msg = r"Timestamp\('2012-01-02 18:01:02-0600', tz='US/Central'\)"
    # 使用 pytest 验证特定异常消息是否被抛出
    with pytest.raises(KeyError, match=msg):
        df[df.index[2]]


# 定义一个测试函数，测试将 datetime 对象与全部为 None 的 Series 进行比较
def test_compare_datetime_with_all_none():
    # GH#54870
    # 创建一个包含日期字符串的 Series 对象，指定数据类型为 datetime64[ns]
    ser = Series(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    # 创建一个全部为 None 的 Series 对象
    ser2 = Series([None, None])
    # 执行日期比较操作，结果是一个布尔值的 Series 对象
    result = ser > ser2
    # 期望的结果是一个布尔值的 Series 对象，全部为 False
    expected = Series([False, False])
    # 使用测试框架验证期望结果与实际结果是否相等
    tm.assert_series_equal(result, expected)
```