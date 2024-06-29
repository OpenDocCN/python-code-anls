# `D:\src\scipysrc\pandas\pandas\tests\window\test_rolling_skew_kurt.py`

```
# 导入必要的模块和函数
from functools import partial  # 导入偏函数功能
import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 测试框架

from pandas import (  # 从 pandas 库中导入多个子模块和函数
    DataFrame,  # 数据框对象
    Series,  # 系列对象
    concat,  # 数据合并函数
    isna,  # 判断缺失值函数
    notna,  # 判断非缺失值函数
)
import pandas._testing as tm  # 导入 pandas 的测试模块

from pandas.tseries import offsets  # 导入 pandas 时间序列的偏移量功能


@pytest.mark.parametrize("sp_func, roll_func", [["kurtosis", "kurt"], ["skew", "skew"]])
def test_series(series, sp_func, roll_func):
    # 导入 scipy.stats 库，并检查是否可用；如果不可用，则跳过测试
    sp_stats = pytest.importorskip("scipy.stats")

    # 创建比较函数，使用 scipy 统计函数中指定的函数名和不偏参数设置为 False
    compare_func = partial(getattr(sp_stats, sp_func), bias=False)
    
    # 对传入的 series 对象进行滚动计算，并获取结果
    result = getattr(series.rolling(50), roll_func)()
    
    # 断言结果类型为 Series 类型
    assert isinstance(result, Series)
    
    # 使用测试模块中的函数，比较结果的最后一个值与部分数据应用比较函数的结果的近似性
    tm.assert_almost_equal(result.iloc[-1], compare_func(series[-50:]))


@pytest.mark.parametrize("sp_func, roll_func", [["kurtosis", "kurt"], ["skew", "skew"]])
def test_frame(raw, frame, sp_func, roll_func):
    # 导入 scipy.stats 库，并检查是否可用；如果不可用，则跳过测试
    sp_stats = pytest.importorskip("scipy.stats")

    # 创建比较函数，使用 scipy 统计函数中指定的函数名和不偏参数设置为 False
    compare_func = partial(getattr(sp_stats, sp_func), bias=False)
    
    # 对传入的 frame 对象进行滚动计算，并获取结果
    result = getattr(frame.rolling(50), roll_func)()
    
    # 断言结果类型为 DataFrame 类型
    assert isinstance(result, DataFrame)
    
    # 使用测试模块中的函数，比较结果的最后一行与部分数据应用比较函数的结果的系列的相等性
    tm.assert_series_equal(
        result.iloc[-1, :],
        frame.iloc[-50:, :].apply(compare_func, axis=0, raw=raw),
        check_names=False,
    )


@pytest.mark.parametrize("sp_func, roll_func", [["kurtosis", "kurt"], ["skew", "skew"]])
def test_time_rule_series(series, sp_func, roll_func):
    # 导入 scipy.stats 库，并检查是否可用；如果不可用，则跳过测试
    sp_stats = pytest.importorskip("scipy.stats")

    # 创建比较函数，使用 scipy 统计函数中指定的函数名和不偏参数设置为 False
    compare_func = partial(getattr(sp_stats, sp_func), bias=False)
    
    # 设定窗口大小
    win = 25
    
    # 对 series 对象进行时间重采样到工作日平均值，并进行滚动计算
    ser = series[::2].resample("B").mean()
    series_result = getattr(ser.rolling(window=win, min_periods=10), roll_func)()
    
    # 获取结果的最后日期
    last_date = series_result.index[-1]
    
    # 获取上一个日期
    prev_date = last_date - 24 * offsets.BDay()

    # 截取上述日期范围内的 series 数据
    trunc_series = series[::2].truncate(prev_date, last_date)
    
    # 使用测试模块中的函数，比较结果的最后一个值与截取数据应用比较函数的结果的近似性
    tm.assert_almost_equal(series_result.iloc[-1], compare_func(trunc_series))


@pytest.mark.parametrize("sp_func, roll_func", [["kurtosis", "kurt"], ["skew", "skew"]])
def test_time_rule_frame(raw, frame, sp_func, roll_func):
    # 导入 scipy.stats 库，并检查是否可用；如果不可用，则跳过测试
    sp_stats = pytest.importorskip("scipy.stats")

    # 创建比较函数，使用 scipy 统计函数中指定的函数名和不偏参数设置为 False
    compare_func = partial(getattr(sp_stats, sp_func), bias=False)
    
    # 设定窗口大小
    win = 25
    
    # 对 frame 对象进行时间重采样到工作日平均值，并进行滚动计算
    frm = frame[::2].resample("B").mean()
    frame_result = getattr(frm.rolling(window=win, min_periods=10), roll_func)()
    
    # 获取结果的最后日期
    last_date = frame_result.index[-1]
    
    # 获取上一个日期
    prev_date = last_date - 24 * offsets.BDay()

    # 截取上述日期范围内的 frame 数据
    trunc_frame = frame[::2].truncate(prev_date, last_date)
    
    # 使用测试模块中的函数，比较结果的最后一行与截取数据应用比较函数的结果的系列的相等性
    tm.assert_series_equal(
        frame_result.xs(last_date),
        trunc_frame.apply(compare_func, raw=raw),
        check_names=False,
    )


@pytest.mark.parametrize("sp_func, roll_func", [["kurtosis", "kurt"], ["skew", "skew"]])
def test_nans(sp_func, roll_func):
    # 导入 scipy.stats 库，并检查是否可用；如果不可用，则跳过测试
    sp_stats = pytest.importorskip("scipy.stats")

    # 创建比较函数，使用 scipy 统计函数中指定的函数名和不偏参数设置为 False
    compare_func = partial(getattr(sp_stats, sp_func), bias=False)
    
    # 创建随机数据 series 对象
    obj = Series(np.random.default_rng(2).standard_normal(50))
    
    # 将数据 series 中的前 10 个和后 10 个元素设置为 NaN
    obj[:10] = np.nan
    obj[-10:] = np.nan

    # 对 obj 对象进行滚动计算，并获取结果
    result = getattr(obj.rolling(50, min_periods=30), roll_func)()
    
    # 使用测试模块中的函数，比较结果的最后一个值与部分数据应用比较函数的结果的近似性
    tm.assert_almost_equal(result.iloc[-1], compare_func(obj[10:-10]))

    # 检查 min_periods 参数是否正确工作
    # 调用对象的 rolling 方法进行滚动计算，窗口大小为20，最小有效数据点数为15，然后应用指定的滚动函数
    result = getattr(obj.rolling(20, min_periods=15), roll_func)()
    # 断言结果的第23行是否为缺失值
    assert isna(result.iloc[23])
    # 断言结果的第24行是否不是缺失值
    assert not isna(result.iloc[24])

    # 断言结果的倒数第6行是否不是缺失值
    assert not isna(result.iloc[-6])
    # 断言结果的倒数第5行是否是缺失值
    assert isna(result.iloc[-5])

    # 创建一个包含20个随机标准正态分布值的序列对象
    obj2 = Series(np.random.default_rng(2).standard_normal(20))
    # 对 obj2 应用 rolling 方法进行滚动计算，窗口大小为10，最小有效数据点数为5，然后应用指定的滚动函数
    result = getattr(obj2.rolling(10, min_periods=5), roll_func)()
    # 断言结果的第3行是否为缺失值
    assert isna(result.iloc[3])
    # 断言结果的第4行是否不是缺失值
    assert notna(result.iloc[4])

    # 对象 obj 进行滚动计算，窗口大小为20，最小有效数据点数为0，然后应用指定的滚动函数
    result0 = getattr(obj.rolling(20, min_periods=0), roll_func)()
    # 对象 obj 进行滚动计算，窗口大小为20，最小有效数据点数为1，然后应用指定的滚动函数
    result1 = getattr(obj.rolling(20, min_periods=1), roll_func)()
    # 使用测试工具断言 result0 和 result1 的值近似相等
    tm.assert_almost_equal(result0, result1)
@pytest.mark.parametrize("minp", [0, 99, 100])
@pytest.mark.parametrize("roll_func", ["kurt", "skew"])
# 定义测试函数，使用参数化测试来测试不同的最小周期和滚动函数
def test_min_periods(series, minp, roll_func, step):
    # 调用对象的 rolling 方法，设置滚动窗口长度为 len(series) + 1，最小周期为 minp，步长为 step，并调用 roll_func 函数
    result = getattr(
        series.rolling(len(series) + 1, min_periods=minp, step=step), roll_func
    )()
    # 期望结果是调用对象的 rolling 方法，设置滚动窗口长度为 len(series)，最小周期为 minp，步长为 step，并调用 roll_func 函数
    expected = getattr(
        series.rolling(len(series), min_periods=minp, step=step), roll_func
    )()
    # 检查结果中的 NaN 值，并断言与期望结果的 NaN 位置相同
    nan_mask = isna(result)
    tm.assert_series_equal(nan_mask, isna(expected))

    # 反转 NaN 掩码，表示非 NaN 的位置
    nan_mask = ~nan_mask
    # 断言在非 NaN 的位置，结果与期望结果几乎相等
    tm.assert_almost_equal(result[nan_mask], expected[nan_mask])


@pytest.mark.parametrize("roll_func", ["kurt", "skew"])
# 定义测试函数，使用参数化测试来测试不同的滚动函数
def test_center(roll_func):
    # 创建一个包含随机标准正态分布数据的 Series 对象，长度为 50
    obj = Series(np.random.default_rng(2).standard_normal(50))
    # 将前 10 个和后 10 个数据设置为 NaN
    obj[:10] = np.nan
    obj[-10:] = np.nan

    # 调用对象的 rolling 方法，设置滚动窗口长度为 20，中心对齐，然后调用 roll_func 函数
    result = getattr(obj.rolling(20, center=True), roll_func)()
    # 创建一个包含 obj 和 9 个 NaN 值的 Series 对象，并进行 rolling 操作，设置滚动窗口长度为 20，然后调用 roll_func 函数
    expected = (
        getattr(concat([obj, Series([np.nan] * 9)]).rolling(20), roll_func)()
        .iloc[9:]
        .reset_index(drop=True)
    )
    # 断言结果与期望结果的 Series 对象相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("roll_func", ["kurt", "skew"])
# 定义测试函数，使用参数化测试来测试不同的滚动函数
def test_center_reindex_series(series, roll_func):
    # 创建一个索引为 ["x0", "x1", ..., "x11"] 的列表
    s = [f"x{x:d}" for x in range(12)]

    # 对 series 执行 reindex 操作，将索引扩展至 series.index + s，并进行滚动操作，窗口长度为 25，然后调用 roll_func 函数
    series_xp = (
        getattr(
            series.reindex(list(series.index) + s).rolling(window=25),
            roll_func,
        )()
        .shift(-12)
        .reindex(series.index)
    )
    # 对 series 执行滚动操作，窗口长度为 25，中心对齐，然后调用 roll_func 函数
    series_rs = getattr(series.rolling(window=25, center=True), roll_func)()
    # 断言 series_xp 和 series_rs 的 Series 对象相等
    tm.assert_series_equal(series_xp, series_rs)


@pytest.mark.slow
@pytest.mark.parametrize("roll_func", ["kurt", "skew"])
# 定义测试函数，使用参数化测试来测试不同的滚动函数，标记为慢速测试
def test_center_reindex_frame(frame, roll_func):
    # 创建一个索引为 ["x0", "x1", ..., "x11"] 的列表
    s = [f"x{x:d}" for x in range(12)]

    # 对 frame 执行 reindex 操作，将索引扩展至 frame.index + s，并进行滚动操作，窗口长度为 25，然后调用 roll_func 函数
    frame_xp = (
        getattr(
            frame.reindex(list(frame.index) + s).rolling(window=25),
            roll_func,
        )()
        .shift(-12)
        .reindex(frame.index)
    )
    # 对 frame 执行滚动操作，窗口长度为 25，中心对齐，然后调用 roll_func 函数
    frame_rs = getattr(frame.rolling(window=25, center=True), roll_func)()
    # 断言 frame_xp 和 frame_rs 的 DataFrame 对象相等
    tm.assert_frame_equal(frame_xp, frame_rs)


# 定义测试函数，测试滚动操作的偏度的边缘情况
def test_rolling_skew_edge_cases(step):
    # 创建期望结果，前 4 个值为 NaN，第 5 个值为 0.0
    expected = Series([np.nan] * 4 + [0.0])[::step]
    # 创建 Series 对象，包含 5 个相同的值，计算滚动窗口为 5 的偏度
    d = Series([1] * 5)
    x = d.rolling(window=5, step=step).skew()
    # 断言期望结果与计算结果的 Series 对象相等
    tm.assert_series_equal(expected, x)

    # 创建期望结果，包含 5 个 NaN 值
    expected = Series([np.nan] * 5)[::step]
    # 创建包含随机标准正态分布数据的 Series 对象，计算滚动窗口为 2 的偏度
    d = Series(np.random.default_rng(2).standard_normal(5))
    x = d.rolling(window=2, step=step).skew()
    # 断言期望结果与计算结果的 Series 对象相等
    tm.assert_series_equal(expected, x)

    # 创建 Series 对象，包含特定数据，计算滚动窗口为 4 的偏度
    d = Series([-1.50837035, -0.1297039, 0.19501095, 1.73508164, 0.41941401])
    # 创建期望结果，前 3 个值为 NaN，第 4 和第 5 个值为具体的偏度值
    expected = Series([np.nan, np.nan, np.nan, 0.177994, 1.548824])[::step]
    x = d.rolling(window=4, step=step).skew()
    # 断言期望结果与计算结果的 Series 对象相等
    tm.assert_series_equal(expected, x)


# 定义测试函数，测试滚动操作的峰度的边缘情况
def test_rolling_kurt_edge_cases(step):
    # 创建期望结果，前 4 个值为 NaN，第 5 个值为 -3.0
    expected = Series([np.nan] * 4 + [-3.0])[::step]

    # 创建 Series 对象，包含 5 个相同的值，计算滚动窗口为 5 的峰度
    d = Series([1] * 5)
    # 计算滚动窗口大小为5的数据序列 d 的峰度，并使用步长 step 进行滚动
    x = d.rolling(window=5, step=step).kurt()
    # 断言滚动峰度序列 x 与期望的序列 expected 相等
    tm.assert_series_equal(expected, x)

    # 生成全为 NaN 的序列（窗口太小）
    expected = Series([np.nan] * 5)[::step]
    # 生成一个包含5个随机标准正态分布样本的数据序列 d
    d = Series(np.random.default_rng(2).standard_normal(5))
    # 计算滚动窗口大小为3的数据序列 d 的峰度，并使用步长 step 进行滚动
    x = d.rolling(window=3, step=step).kurt()
    # 断言滚动峰度序列 x 与期望的序列 expected 相等
    tm.assert_series_equal(expected, x)

    # 生成具有特定数值的数据序列 d
    d = Series([-1.50837035, -0.1297039, 0.19501095, 1.73508164, 0.41941401])
    # 生成期望的滚动峰度序列 expected，包含 [NaN, NaN, NaN, 1.224307, 2.671499]（根据步长选择）
    expected = Series([np.nan, np.nan, np.nan, 1.224307, 2.671499])[::step]
    # 计算滚动窗口大小为4的数据序列 d 的峰度，并使用步长 step 进行滚动
    x = d.rolling(window=4, step=step).kurt()
    # 断言滚动峰度序列 x 与期望的序列 expected 相等
    tm.assert_series_equal(expected, x)
# 定义一个函数，用于测试滚动偏度（rolling skew）等于值时的情况
def test_rolling_skew_eq_value_fperr(step):
    # #18804 更新：所有值相等时，滚动偏度应返回 NaN
    # #46717 更新：所有值相等时，滚动偏度应返回 0 而不是 NaN
    # 创建一个 Series 对象，包含 15 个相等的浮点数 1.1，并计算滚动窗口为 10 的滚动偏度
    a = Series([1.1] * 15).rolling(window=10, step=step).skew()
    # 断言：索引大于等于 9 的部分滚动偏度应全部为 0
    assert (a[a.index >= 9] == 0).all()
    # 断言：索引小于 9 的部分滚动偏度应全部为 NaN
    assert a[a.index < 9].isna().all()


# 定义一个函数，用于测试滚动峰度（rolling kurt）等于值时的情况
def test_rolling_kurt_eq_value_fperr(step):
    # #18804 更新：所有值相等时，滚动峰度应返回 NaN
    # #46717 更新：所有值相等时，滚动峰度应返回 -3 而不是 NaN
    # 创建一个 Series 对象，包含 15 个相等的浮点数 1.1，并计算滚动窗口为 10 的滚动峰度
    a = Series([1.1] * 15).rolling(window=10, step=step).kurt()
    # 断言：索引大于等于 9 的部分滚动峰度应全部为 -3
    assert (a[a.index >= 9] == -3).all()
    # 断言：索引小于 9 的部分滚动峰度应全部为 NaN
    assert a[a.index < 9].isna().all()
```