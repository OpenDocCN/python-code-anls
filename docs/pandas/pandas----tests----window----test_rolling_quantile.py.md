# `D:\src\scipysrc\pandas\pandas\tests\window\test_rolling_quantile.py`

```
from functools import partial  # 导入 functools 模块的 partial 函数，用于创建偏函数

import numpy as np  # 导入 NumPy 库，并用 np 作为别名
import pytest  # 导入 pytest 测试框架

from pandas import (  # 从 pandas 库中导入以下模块：
    DataFrame,  # DataFrame 数据结构，用于处理二维数据
    Series,  # Series 数据结构，用于处理一维数据
    concat,  # 用于连接 pandas 对象的函数
    isna,  # 检查是否为缺失值的函数
    notna,  # 检查是否不是缺失值的函数
)
import pandas._testing as tm  # 导入 pandas 内部测试模块，使用 tm 作为别名

from pandas.tseries import offsets  # 从 pandas 的 tseries 模块中导入 offsets

# 计算给定数组的百分位数
def scoreatpercentile(a, per):
    values = np.sort(a, axis=0)  # 对数组 a 按第 0 轴（行）进行排序

    idx = int(per / 1.0 * (values.shape[0] - 1))  # 计算百分位数所对应的索引

    if idx == values.shape[0] - 1:
        retval = values[-1]  # 如果 idx 是最后一个索引，则返回最后一个值
    else:
        qlow = idx / (values.shape[0] - 1)  # 计算较低的百分位数的比例
        qhig = (idx + 1) / (values.shape[0] - 1)  # 计算较高的百分位数的比例
        vlow = values[idx]  # 较低百分位数所对应的值
        vhig = values[idx + 1]  # 较高百分位数所对应的值
        # 根据百分位数的比例计算插值
        retval = vlow + (vhig - vlow) * (per - qlow) / (qhig - qlow)

    return retval  # 返回计算得到的百分位数值


# 使用 pytest 的参数化功能，对 Series 进行测试
@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_series(series, q, step):
    compare_func = partial(scoreatpercentile, per=q)  # 创建一个比较函数的偏函数
    result = series.rolling(50, step=step).quantile(q)  # 对 Series 应用滚动窗口计算百分位数
    assert isinstance(result, Series)  # 断言结果是一个 Series 对象
    end = range(0, len(series), step or 1)[-1] + 1  # 计算测试结束位置
    # 使用测试模块进行准确度比较
    tm.assert_almost_equal(result.iloc[-1], compare_func(series[end - 50 : end]))


# 使用 pytest 的参数化功能，对 DataFrame 进行测试
@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_frame(raw, frame, q, step):
    compare_func = partial(scoreatpercentile, per=q)  # 创建一个比较函数的偏函数
    result = frame.rolling(50, step=step).quantile(q)  # 对 DataFrame 应用滚动窗口计算百分位数
    assert isinstance(result, DataFrame)  # 断言结果是一个 DataFrame 对象
    end = range(0, len(frame), step or 1)[-1] + 1  # 计算测试结束位置
    # 使用测试模块进行 Series 对象的准确度比较
    tm.assert_series_equal(
        result.iloc[-1, :],
        frame.iloc[end - 50 : end, :].apply(compare_func, axis=0, raw=raw),
        check_names=False,
    )


# 使用 pytest 的参数化功能，测试时间规则下的 Series 对象
@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_time_rule_series(series, q):
    compare_func = partial(scoreatpercentile, per=q)  # 创建一个比较函数的偏函数
    win = 25  # 窗口大小为 25
    ser = series[::2].resample("B").mean()  # 对 Series 进行下采样，并计算每个工作日的平均值
    series_result = ser.rolling(window=win, min_periods=10).quantile(q)  # 应用滚动窗口计算百分位数
    last_date = series_result.index[-1]  # 获取结果的最后一个日期索引
    prev_date = last_date - 24 * offsets.BDay()  # 计算前一个日期索引

    trunc_series = series[::2].truncate(prev_date, last_date)  # 截取时间范围内的 Series 数据
    # 使用测试模块进行准确度比较
    tm.assert_almost_equal(series_result.iloc[-1], compare_func(trunc_series))


# 使用 pytest 的参数化功能，测试时间规则下的 DataFrame 对象
@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_time_rule_frame(raw, frame, q):
    compare_func = partial(scoreatpercentile, per=q)  # 创建一个比较函数的偏函数
    win = 25  # 窗口大小为 25
    frm = frame[::2].resample("B").mean()  # 对 DataFrame 进行下采样，并计算每个工作日的平均值
    frame_result = frm.rolling(window=win, min_periods=10).quantile(q)  # 应用滚动窗口计算百分位数
    last_date = frame_result.index[-1]  # 获取结果的最后一个日期索引
    prev_date = last_date - 24 * offsets.BDay()  # 计算前一个日期索引

    trunc_frame = frame[::2].truncate(prev_date, last_date)  # 截取时间范围内的 DataFrame 数据
    # 使用测试模块进行 Series 对象的准确度比较
    tm.assert_series_equal(
        frame_result.xs(last_date),
        trunc_frame.apply(compare_func, raw=raw),
        check_names=False,
    )


# 使用 pytest 的参数化功能，测试包含 NaN 值的情况
@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_nans(q):
    compare_func = partial(scoreatpercentile, per=q)  # 创建一个比较函数的偏函数
    obj = Series(np.random.default_rng(2).standard_normal(50))  # 创建一个带有随机数据的 Series 对象
    obj[:10] = np.nan  # 将前 10 个元素设置为 NaN
    obj[-10:] = np.nan  # 将最后 10 个元素设置为 NaN

    result = obj.rolling(50, min_periods=30).quantile(q)  # 应用滚动窗口计算百分位数
    tm.assert_almost_equal(result.iloc[-1], compare_func(obj[10:-10]))  # 使用测试模块进行准确度比较

    # min_periods 参数正常工作
    # 对对象 `obj` 进行滚动窗口大小为20，最小有效周期为15的分位数计算，并将结果存储在 `result` 中
    result = obj.rolling(20, min_periods=15).quantile(q)
    
    # 断言：检查 `result` 中第23个元素是否为缺失值
    assert isna(result.iloc[23])
    # 断言：检查 `result` 中第24个元素是否不是缺失值
    assert not isna(result.iloc[24])
    
    # 断言：检查 `result` 中倒数第6个元素是否不是缺失值
    assert not isna(result.iloc[-6])
    # 断言：检查 `result` 中倒数第5个元素是否是缺失值
    assert isna(result.iloc[-5])
    
    # 创建一个包含20个标准正态随机数的 `Series` 对象 `obj2`
    obj2 = Series(np.random.default_rng(2).standard_normal(20))
    # 对 `obj2` 进行滚动窗口大小为10，最小有效周期为5的分位数计算，并将结果存储在 `result` 中
    result = obj2.rolling(10, min_periods=5).quantile(q)
    
    # 断言：检查 `result` 中第3个元素是否为缺失值
    assert isna(result.iloc[3])
    # 断言：检查 `result` 中第4个元素是否不是缺失值
    assert notna(result.iloc[4])
    
    # 使用对象 `obj` 进行两次滚动窗口大小为20的分位数计算，其中最小有效周期分别为0和1
    result0 = obj.rolling(20, min_periods=0).quantile(q)
    result1 = obj.rolling(20, min_periods=1).quantile(q)
    # 使用断言检查两次计算结果是否几乎相等
    tm.assert_almost_equal(result0, result1)
# 使用 pytest.mark.parametrize 装饰器，为 test_min_periods 函数参数化测试用例
@pytest.mark.parametrize("minp", [0, 99, 100])
@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_min_periods(series, minp, q, step):
    # 调用 rolling 方法计算滚动窗口的 quantile，min_periods 参数设置最小有效期
    result = series.rolling(len(series) + 1, min_periods=minp, step=step).quantile(q)
    expected = series.rolling(len(series), min_periods=minp, step=step).quantile(q)
    # 检查结果和期望的 NaN 掩码是否一致
    nan_mask = isna(result)
    tm.assert_series_equal(nan_mask, isna(expected))

    # 取反 NaN 掩码
    nan_mask = ~nan_mask
    # 检查非 NaN 部分的数值是否几乎相等
    tm.assert_almost_equal(result[nan_mask], expected[nan_mask])


# 使用 pytest.mark.parametrize 装饰器，为 test_center 函数参数化测试用例
@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_center(q):
    # 创建包含随机数据和 NaN 的 Series 对象
    obj = Series(np.random.default_rng(2).standard_normal(50))
    obj[:10] = np.nan
    obj[-10:] = np.nan

    # 计算中心滚动窗口的 quantile
    result = obj.rolling(20, center=True).quantile(q)
    # 期望的结果是扩展 NaN 值后的滚动窗口 quantile
    expected = (
        concat([obj, Series([np.nan] * 9)])
        .rolling(20)
        .quantile(q)
        .iloc[9:]
        .reset_index(drop=True)
    )
    # 检查结果是否与期望相等
    tm.assert_series_equal(result, expected)


# 使用 pytest.mark.parametrize 装饰器，为 test_center_reindex_series 函数参数化测试用例
@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_center_reindex_series(series, q):
    # 创建额外的索引列表 s
    # shifter index
    s = [f"x{x:d}" for x in range(12)]

    # 对 Series 进行重新索引，然后计算中心滚动窗口的 quantile
    series_xp = (
        series.reindex(list(series.index) + s)
        .rolling(window=25)
        .quantile(q)
        .shift(-12)
        .reindex(series.index)
    )

    # 直接对 Series 计算中心滚动窗口的 quantile
    series_rs = series.rolling(window=25, center=True).quantile(q)
    # 检查两者是否相等
    tm.assert_series_equal(series_xp, series_rs)


# 使用 pytest.mark.parametrize 装饰器，为 test_center_reindex_frame 函数参数化测试用例
@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_center_reindex_frame(frame, q):
    # 创建额外的索引列表 s
    # shifter index
    s = [f"x{x:d}" for x in range(12)]

    # 对 DataFrame 进行重新索引，然后计算中心滚动窗口的 quantile
    frame_xp = (
        frame.reindex(list(frame.index) + s)
        .rolling(window=25)
        .quantile(q)
        .shift(-12)
        .reindex(frame.index)
    )
    # 直接对 DataFrame 计算中心滚动窗口的 quantile
    frame_rs = frame.rolling(window=25, center=True).quantile(q)
    # 检查两者是否相等
    tm.assert_frame_equal(frame_xp, frame_rs)
```