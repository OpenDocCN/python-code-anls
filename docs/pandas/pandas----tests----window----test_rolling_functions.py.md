# `D:\src\scipysrc\pandas\pandas\tests\window\test_rolling_functions.py`

```
# 导入datetime模块中的datetime类，用于处理日期和时间
from datetime import datetime

# 导入numpy库，并使用np作为别名
import numpy as np

# 导入pytest库，用于编写和运行测试用例
import pytest

# 导入pandas库中的_test_decorators模块，用于测试装饰器
import pandas.util._test_decorators as td

# 从pandas库中导入多个类和函数，包括DataFrame、DatetimeIndex、Series、concat、isna、notna
from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    concat,
    isna,
    notna,
)

# 导入pandas库中的_testing模块，用于测试
import pandas._testing as tm

# 从pandas库中的tseries模块中导入offsets对象，用于时间序列的偏移量
from pandas.tseries import offsets


# 使用pytest.mark.parametrize装饰器标记的测试函数，对series对象执行滚动计算
@pytest.mark.parametrize(
    "compare_func, roll_func, kwargs",
    [
        [np.mean, "mean", {}],  # 比较函数为np.mean，滚动函数为'mean'，无附加参数
        [np.nansum, "sum", {}],  # 比较函数为np.nansum，滚动函数为'sum'，无附加参数
        [
            lambda x: np.isfinite(x).astype(float).sum(),
            "count",
            {},  # 比较函数为lambda函数，计算有限值数量，滚动函数为'count'，无附加参数
        ],
        [np.median, "median", {}],  # 比较函数为np.median，滚动函数为'median'，无附加参数
        [np.min, "min", {}],  # 比较函数为np.min，滚动函数为'min'，无附加参数
        [np.max, "max", {}],  # 比较函数为np.max，滚动函数为'max'，无附加参数
        [lambda x: np.std(x, ddof=1), "std", {}],  # 比较函数为lambda函数，计算有偏标准差，滚动函数为'std'，无附加参数
        [lambda x: np.std(x, ddof=0), "std", {"ddof": 0}],  # 比较函数为lambda函数，计算无偏标准差，滚动函数为'std'，附加参数ddof=0
        [lambda x: np.var(x, ddof=1), "var", {}],  # 比较函数为lambda函数，计算有偏方差，滚动函数为'var'，无附加参数
        [lambda x: np.var(x, ddof=0), "var", {"ddof": 0}],  # 比较函数为lambda函数，计算无偏方差，滚动函数为'var'，附加参数ddof=0
    ],
)
def test_series(series, compare_func, roll_func, kwargs, step):
    # 执行滚动计算，并返回结果
    result = getattr(series.rolling(50, step=step), roll_func)(**kwargs)
    # 断言结果为Series类型
    assert isinstance(result, Series)
    # 计算滚动窗口的结束位置
    end = range(0, len(series), step or 1)[-1] + 1
    # 断言结果的最后一个值与比较函数应用于原始series的最后50个值的结果接近
    tm.assert_almost_equal(result.iloc[-1], compare_func(series[end - 50 : end]))


# 使用pytest.mark.parametrize装饰器标记的测试函数，对frame对象执行滚动计算
@pytest.mark.parametrize(
    "compare_func, roll_func, kwargs",
    [
        [np.mean, "mean", {}],  # 比较函数为np.mean，滚动函数为'mean'，无附加参数
        [np.nansum, "sum", {}],  # 比较函数为np.nansum，滚动函数为'sum'，无附加参数
        [
            lambda x: np.isfinite(x).astype(float).sum(),
            "count",
            {},  # 比较函数为lambda函数，计算有限值数量，滚动函数为'count'，无附加参数
        ],
        [np.median, "median", {}],  # 比较函数为np.median，滚动函数为'median'，无附加参数
        [np.min, "min", {}],  # 比较函数为np.min，滚动函数为'min'，无附加参数
        [np.max, "max", {}],  # 比较函数为np.max，滚动函数为'max'，无附加参数
        [lambda x: np.std(x, ddof=1), "std", {}],  # 比较函数为lambda函数，计算有偏标准差，滚动函数为'std'，无附加参数
        [lambda x: np.std(x, ddof=0), "std", {"ddof": 0}],  # 比较函数为lambda函数，计算无偏标准差，滚动函数为'std'，附加参数ddof=0
        [lambda x: np.var(x, ddof=1), "var", {}],  # 比较函数为lambda函数，计算有偏方差，滚动函数为'var'，无附加参数
        [lambda x: np.var(x, ddof=0), "var", {"ddof": 0}],  # 比较函数为lambda函数，计算无偏方差，滚动函数为'var'，附加参数ddof=0
    ],
)
def test_frame(raw, frame, compare_func, roll_func, kwargs, step):
    # 执行滚动计算，并返回结果
    result = getattr(frame.rolling(50, step=step), roll_func)(**kwargs)
    # 断言结果为DataFrame类型
    assert isinstance(result, DataFrame)
    # 计算滚动窗口的结束位置
    end = range(0, len(frame), step or 1)[-1] + 1
    # 断言结果的最后一行与比较函数应用于原始frame最后50行的结果的Series相等，不检查名称
    tm.assert_series_equal(
        result.iloc[-1, :],
        frame.iloc[end - 50 : end, :].apply(compare_func, axis=0, raw=raw),
        check_names=False,
    )


# 使用pytest.mark.parametrize装饰器标记的测试函数，对series对象应用时间规则的滚动计算
@pytest.mark.parametrize(
    "compare_func, roll_func, kwargs, minp",
    [
        [np.mean, "mean", {}, 10],  # 比较函数为np.mean，滚动函数为'mean'，最小观测点数为10
        [np.nansum, "sum", {}, 10],  # 比较函数为np.nansum，滚动函数为'sum'，最小观测点数为10
        [lambda x: np.isfinite(x).astype(float).sum(), "count", {}, 0],  # 比较函数为lambda函数，计算有限值数量，滚动函数为'count'，最小观测点数为0
        [np.median, "median", {}, 10],  # 比较函数为np.median，滚动函数为'median'，最小观测点数为10
        [np.min, "min", {}, 10],  # 比较函数为np.min，滚动函数为'min'，最小观测点数为10
        [np.max, "max", {}, 10],  # 比较函数为np.max，滚动函数为'max'，最小观测点数为10
        [lambda x: np.std(x, ddof=1), "std", {}, 10],  # 比较函数为lambda函数，计算有偏标准差，滚动函数为'std'，最小观测点数为10
        [lambda x: np.std(x, ddof=0), "std",
    # 计算前一日期，减去一个工作日的时间间隔
    prev_date = last_date - 24 * offsets.BDay()
    
    # 对系列数据进行每隔两个数据点取样，然后截断到指定的时间范围内
    trunc_series = series[::2].truncate(prev_date, last_date)
    
    # 使用断言确保系列结果的最后一个值接近于使用比较函数计算出的值
    tm.assert_almost_equal(series_result.iloc[-1], compare_func(trunc_series))
@pytest.mark.parametrize(
    "compare_func, roll_func, kwargs, minp",
    [
        [np.mean, "mean", {}, 10],  # 参数化测试：使用np.mean计算平均值，窗口大小为10
        [np.nansum, "sum", {}, 10],  # 参数化测试：使用np.nansum计算总和，窗口大小为10
        [lambda x: np.isfinite(x).astype(float).sum(), "count", {}, 0],  # 参数化测试：计算非NaN值的数量，窗口大小为0
        [np.median, "median", {}, 10],  # 参数化测试：使用np.median计算中位数，窗口大小为10
        [np.min, "min", {}, 10],  # 参数化测试：使用np.min计算最小值，窗口大小为10
        [np.max, "max", {}, 10],  # 参数化测试：使用np.max计算最大值，窗口大小为10
        [lambda x: np.std(x, ddof=1), "std", {}, 10],  # 参数化测试：计算样本标准差，窗口大小为10
        [lambda x: np.std(x, ddof=0), "std", {"ddof": 0}, 10],  # 参数化测试：计算总体标准差，窗口大小为10
        [lambda x: np.var(x, ddof=1), "var", {}, 10],  # 参数化测试：计算样本方差，窗口大小为10
        [lambda x: np.var(x, ddof=0), "var", {"ddof": 0}, 10],  # 参数化测试：计算总体方差，窗口大小为10
    ],
)
def test_time_rule_frame(raw, frame, compare_func, roll_func, kwargs, minp):
    win = 25
    frm = frame[::2].resample("B").mean()  # 对frame进行降采样并计算均值
    frame_result = getattr(frm.rolling(window=win, min_periods=minp), roll_func)(  # 应用滚动函数到降采样后的数据
        **kwargs
    )
    last_date = frame_result.index[-1]  # 获取结果的最后一个日期
    prev_date = last_date - 24 * offsets.BDay()  # 计算前一个日期

    trunc_frame = frame[::2].truncate(prev_date, last_date)  # 根据日期范围截取frame
    tm.assert_series_equal(
        frame_result.xs(last_date),
        trunc_frame.apply(compare_func, raw=raw),  # 比较滚动函数的结果和截取frame应用比较函数的结果
        check_names=False,
    )


@pytest.mark.parametrize(
    "compare_func, roll_func, kwargs",
    [
        [np.mean, "mean", {}],  # 参数化测试：使用np.mean计算平均值
        [np.nansum, "sum", {}],  # 参数化测试：使用np.nansum计算总和
        [np.median, "median", {}],  # 参数化测试：使用np.median计算中位数
        [np.min, "min", {}],  # 参数化测试：使用np.min计算最小值
        [np.max, "max", {}],  # 参数化测试：使用np.max计算最大值
        [lambda x: np.std(x, ddof=1), "std", {}],  # 参数化测试：计算样本标准差
        [lambda x: np.std(x, ddof=0), "std", {"ddof": 0}],  # 参数化测试：计算总体标准差
        [lambda x: np.var(x, ddof=1), "var", {}],  # 参数化测试：计算样本方差
        [lambda x: np.var(x, ddof=0), "var", {"ddof": 0}],  # 参数化测试：计算总体方差
    ],
)
def test_nans(compare_func, roll_func, kwargs):
    obj = Series(np.random.default_rng(2).standard_normal(50))  # 创建一个包含50个正态分布随机数的Series
    obj[:10] = np.nan  # 将前10个元素设为NaN
    obj[-10:] = np.nan  # 将最后10个元素设为NaN

    result = getattr(obj.rolling(50, min_periods=30), roll_func)(**kwargs)  # 应用滚动函数到Series
    tm.assert_almost_equal(result.iloc[-1], compare_func(obj[10:-10]))  # 比较滚动函数的最后结果与参考函数的结果

    # 测试min_periods参数是否正常工作
    result = getattr(obj.rolling(20, min_periods=15), roll_func)(**kwargs)
    assert pd.isna(result.iloc[23])
    assert not pd.isna(result.iloc[24])

    assert not pd.isna(result.iloc[-6])
    assert pd.isna(result.iloc[-5])

    obj2 = Series(np.random.default_rng(2).standard_normal(20))  # 创建另一个Series
    result = getattr(obj2.rolling(10, min_periods=5), roll_func)(**kwargs)
    assert pd.isna(result.iloc[3])
    assert pd.notna(result.iloc[4])

    if roll_func != "sum":
        result0 = getattr(obj.rolling(20, min_periods=0), roll_func)(**kwargs)
        result1 = getattr(obj.rolling(20, min_periods=1), roll_func)(**kwargs)
        tm.assert_almost_equal(result0, result1)


def test_nans_count():
    obj = Series(np.random.default_rng(2).standard_normal(50))  # 创建一个包含50个正态分布随机数的Series
    obj[:10] = np.nan  # 将前10个元素设为NaN
    obj[-10:] = np.nan  # 将最后10个元素设为NaN
    result = obj.rolling(50, min_periods=30).count()  # 计算非NaN值的数量
    tm.assert_almost_equal(
        result.iloc[-1], np.isfinite(obj[10:-10]).astype(float).sum()  # 比较计算结果和参考值
    )
    # 一个包含多个统计函数及其参数的列表
    [
        # 统计函数 "mean"，无参数
        ["mean", {}],
        # 统计函数 "sum"，无参数
        ["sum", {}],
        # 统计函数 "median"，无参数
        ["median", {}],
        # 统计函数 "min"，无参数
        ["min", {}],
        # 统计函数 "max"，无参数
        ["max", {}],
        # 统计函数 "std"，无参数
        ["std", {}],
        # 统计函数 "std"，参数包含 "ddof" 为 0
        ["std", {"ddof": 0}],
        # 统计函数 "var"，无参数
        ["var", {}],
        # 统计函数 "var"，参数包含 "ddof" 为 0
        ["var", {"ddof": 0}],
    ],
@pytest.mark.parametrize(
    "roll_func, kwargs, minp",
    [
        ["mean", {}, 15],
        ["sum", {}, 15],
        ["count", {}, 0],
        ["median", {}, 15],
        ["min", {}, 15],
        ["max", {}, 15],
        ["std", {}, 15],
        ["std", {"ddof": 0}, 15],
        ["var", {}, 15],
        ["var", {"ddof": 0}, 15],
    ],
)
def test_center(roll_func, kwargs, minp):
    # 创建一个包含50个标准正态分布值的Series对象
    obj = Series(np.random.default_rng(2).standard_normal(50))
    # 将前10个值和后10个值设置为NaN
    obj[:10] = np.nan
    obj[-10:] = np.nan

    # 执行滚动窗口操作，计算指定函数（mean/sum/count等）的结果
    result = getattr(obj.rolling(20, min_periods=minp, center=True), roll_func)(
        **kwargs
    )
    # 期望的结果，基于包含NaN值的Series对象执行相同的滚动窗口操作
    expected = (
        getattr(
            concat([obj, Series([np.nan] * 9)]).rolling(20, min_periods=minp), roll_func
        )(**kwargs)
        .iloc[9:]  # 删除前9个NaN值
        .reset_index(drop=True)  # 重置索引
    )
    # 断言两个Series对象是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "roll_func, kwargs, minp, fill_value",
    [
        ["mean", {}, 10, None],
        ["sum", {}, 10, None],
        ["count", {}, 0, 0],
        ["median", {}, 10, None],
        ["min", {}, 10, None],
        ["max", {}, 10, None],
        ["std", {}, 10, None],
        ["std", {"ddof": 0}, 10, None],
        ["var", {}, 10, None],
        ["var", {"ddof": 0}, 10, None],
    ],
)
def test_center_reindex_series(series, roll_func, kwargs, minp, fill_value):
    # 创建一个包含shifted index的列表
    s = [f"x{x:d}" for x in range(12)]

    # 执行重新索引操作后的滚动窗口计算，计算指定函数（mean/sum/count等）的结果
    series_xp = (
        getattr(
            series.reindex(list(series.index) + s).rolling(window=25, min_periods=minp),
            roll_func,
        )(**kwargs)
        .shift(-12)  # 将结果向前移动12个位置
        .reindex(series.index)  # 恢复原始索引
    )
    # 执行滚动窗口计算，计算指定函数（mean/sum/count等）的结果
    series_rs = getattr(
        series.rolling(window=25, min_periods=minp, center=True), roll_func
    )(**kwargs)
    # 如果填充值不为None，则使用指定的填充值填充NaN值
    if fill_value is not None:
        series_xp = series_xp.fillna(fill_value)
    # 断言两个Series对象是否相等
    tm.assert_series_equal(series_xp, series_rs)
    [
        # 创建包含统计函数信息的列表，每个元素是一个包含以下内容的列表：
        # - 统计函数名称
        # - 参数字典（当前为空字典）
        # - 第一个数据输入值（初始为10）
        # - 第二个数据输入值（初始为None）
        ["mean", {}, 10, None],
        ["sum", {}, 10, None],
        ["count", {}, 0, 0],
        ["median", {}, 10, None],
        ["min", {}, 10, None],
        ["max", {}, 10, None],
        ["std", {}, 10, None],
        ["std", {"ddof": 0}, 10, None],
        ["var", {}, 10, None],
        ["var", {"ddof": 0}, 10, None],
    ],
# 定义一个测试函数，用于测试数据框架的重新索引和滚动操作
def test_center_reindex_frame(frame, roll_func, kwargs, minp, fill_value):
    # 创建一个包含12个字符串的列表，用于作为后续索引使用
    s = [f"x{x:d}" for x in range(12)]

    # 对数据框架进行重新索引和滚动操作，并调用特定的滚动函数
    frame_xp = (
        getattr(
            frame.reindex(list(frame.index) + s).rolling(window=25, min_periods=minp),
            roll_func,
        )(**kwargs)
        .shift(-12)  # 将结果向前偏移12个位置
        .reindex(frame.index)  # 根据原索引重新排列结果
    )
    
    # 使用滚动函数对数据框架进行滚动操作，并指定滚动参数
    frame_rs = getattr(
        frame.rolling(window=25, min_periods=minp, center=True), roll_func
    )(**kwargs)
    
    # 如果指定了填充值，则用填充值填充缺失值
    if fill_value is not None:
        frame_xp = frame_xp.fillna(fill_value)
    
    # 断言两个数据框架相等
    tm.assert_frame_equal(frame_xp, frame_rs)


# 使用pytest的参数化装饰器，定义一系列滚动函数的测试用例
@pytest.mark.parametrize(
    "f",
    [
        lambda x: x.rolling(window=10, min_periods=5).cov(x, pairwise=False),
        lambda x: x.rolling(window=10, min_periods=5).corr(x, pairwise=False),
        lambda x: x.rolling(window=10, min_periods=5).max(),
        lambda x: x.rolling(window=10, min_periods=5).min(),
        lambda x: x.rolling(window=10, min_periods=5).sum(),
        lambda x: x.rolling(window=10, min_periods=5).mean(),
        lambda x: x.rolling(window=10, min_periods=5).std(),
        lambda x: x.rolling(window=10, min_periods=5).var(),
        lambda x: x.rolling(window=10, min_periods=5).skew(),
        lambda x: x.rolling(window=10, min_periods=5).kurt(),
        lambda x: x.rolling(window=10, min_periods=5).quantile(q=0.5),
        lambda x: x.rolling(window=10, min_periods=5).median(),
        lambda x: x.rolling(window=10, min_periods=5).apply(sum, raw=False),
        lambda x: x.rolling(window=10, min_periods=5).apply(sum, raw=True),
        # 使用pytest.param装饰器标记特定的测试用例，指定跳过条件
        pytest.param(
            lambda x: x.rolling(win_type="boxcar", window=10, min_periods=5).mean(),
            marks=td.skip_if_no("scipy"),
        ),
    ],
)
# 定义测试滚动函数的窗口非收缩行为
def test_rolling_functions_window_non_shrinkage(f):
    # 创建一个包含4个整数的序列，并预期所有结果为NaN的序列
    s = Series(range(4))
    s_expected = Series(np.nan, index=s.index)
    
    # 创建一个包含4行2列数据的数据框架，并预期所有结果为NaN的数据框架
    df = DataFrame([[1, 5], [3, 2], [3, 9], [-1, 0]], columns=["A", "B"])
    df_expected = DataFrame(np.nan, index=df.index, columns=df.columns)

    # 分别对序列和数据框架应用滚动函数，并断言结果与预期相等
    s_result = f(s)
    tm.assert_series_equal(s_result, s_expected)

    df_result = f(df)
    tm.assert_frame_equal(df_result, df_expected)


# 定义一个测试函数，验证在滚动窗口为1时的最大值计算和重采样操作
def test_rolling_max_gh6297(step):
    """Replicate result expected in GH #6297"""
    # 创建一个包含日期时间索引的序列，其中包含6个时间点的数据
    indices = [datetime(1975, 1, i) for i in range(1, 6)]
    # 添加一个时间点，使得某一天有两个数据点
    indices.append(datetime(1975, 1, 3, 6, 0))
    series = Series(range(1, 7), index=indices)
    # 将整数值映射为浮点数值
    series = series.map(lambda x: float(x))
    # 按时间排序数据
    series = series.sort_index()

    # 创建预期的序列，包含期望的最大值计算结果
    expected = Series(
        [1.0, 2.0, 6.0, 4.0, 5.0],
        index=DatetimeIndex([datetime(1975, 1, i, 0) for i in range(1, 6)], freq="D"),
    )[::step]
    
    # 对序列进行重采样，并计算滚动窗口为1时的最大值
    x = series.resample("D").max().rolling(window=1, step=step).max()
    tm.assert_series_equal(expected, x)


# 定义一个测试函数，验证滚动最大值和重采样操作
def test_rolling_max_resample(step):
    indices = [datetime(1975, 1, i) for i in range(1, 6)]
    # 为了在最后一天有3个数据点（4、10和20），添加特定的时间点到索引列表中
    indices.append(datetime(1975, 1, 5, 1))
    indices.append(datetime(1975, 1, 5, 2))
    # 创建一个时间序列，包含值为0到4，以及额外的值10和20，使用indices作为索引
    series = Series(list(range(5)) + [10, 20], index=indices)
    # 将序列中的整数值转换为浮点数
    series = series.map(lambda x: float(x))
    # 按照时间顺序对序列进行排序
    series = series.sort_index()

    # 期望的结果是按照指定的频率和索引创建的时间序列，使用step参数进行切片
    expected = Series(
        [0.0, 1.0, 2.0, 3.0, 20.0],
        index=DatetimeIndex([datetime(1975, 1, i, 0) for i in range(1, 6)], freq="D"),
    )[::step]
    # 对原始序列进行按天重采样，并使用最大值进行滚动计算，步长为step
    x = series.resample("D").max().rolling(window=1, step=step).max()
    # 使用断言来验证预期的序列与计算结果的一致性
    tm.assert_series_equal(expected, x)

    # 现在指定中位数（10.0）
    expected = Series(
        [0.0, 1.0, 2.0, 3.0, 10.0],
        index=DatetimeIndex([datetime(1975, 1, i, 0) for i in range(1, 6)], freq="D"),
    )[::step]
    # 对原始序列进行按天重采样，并使用中位数进行滚动计算，步长为step
    x = series.resample("D").median().rolling(window=1, step=step).max()
    # 使用断言来验证预期的序列与计算结果的一致性
    tm.assert_series_equal(expected, x)

    # 现在指定均值（(4+10+20)/3）
    v = (4.0 + 10.0 + 20.0) / 3.0
    expected = Series(
        [0.0, 1.0, 2.0, 3.0, v],
        index=DatetimeIndex([datetime(1975, 1, i, 0) for i in range(1, 6)], freq="D"),
    )[::step]
    # 对原始序列进行按天重采样，并使用均值进行滚动计算，步长为step
    x = series.resample("D").mean().rolling(window=1, step=step).max()
    # 使用断言来验证预期的序列与计算结果的一致性
    tm.assert_series_equal(expected, x)
def test_rolling_min_resample(step):
    # 创建日期时间索引，从1975年1月1日到1月5日，并在最后一天添加额外数据点
    indices = [datetime(1975, 1, i) for i in range(1, 6)]
    indices.append(datetime(1975, 1, 5, 1))  # 额外数据点
    indices.append(datetime(1975, 1, 5, 2))  # 额外数据点
    # 创建时间序列，包含值从0到4，以及额外的值10和20，使用日期时间索引
    series = Series(list(range(5)) + [10, 20], index=indices)
    # 将所有值转换为浮点数
    series = series.map(lambda x: float(x))
    # 按时间顺序排序时间序列
    series = series.sort_index()

    # 创建预期的结果时间序列，每步长为step，计算每日最小值
    expected = Series(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        index=DatetimeIndex([datetime(1975, 1, i, 0) for i in range(1, 6)], freq="D"),
    )[::step]
    # 对时间序列进行按日重采样，并计算滚动窗口大小为1的最小值
    r = series.resample("D").min().rolling(window=1, step=step)
    # 使用测试工具比较预期结果和实际结果的最小值序列
    tm.assert_series_equal(expected, r.min())


def test_rolling_median_resample():
    # 创建日期时间索引，从1975年1月1日到1月5日，并在最后一天添加额外数据点
    indices = [datetime(1975, 1, i) for i in range(1, 6)]
    indices.append(datetime(1975, 1, 5, 1))  # 额外数据点
    indices.append(datetime(1975, 1, 5, 2))  # 额外数据点
    # 创建时间序列，包含值从0到4，以及额外的值10和20，使用日期时间索引
    series = Series(list(range(5)) + [10, 20], index=indices)
    # 将所有值转换为浮点数
    series = series.map(lambda x: float(x))
    # 按时间顺序排序时间序列
    series = series.sort_index()

    # 创建预期的结果时间序列，每日计算中位数
    expected = Series(
        [0.0, 1.0, 2.0, 3.0, 10],
        index=DatetimeIndex([datetime(1975, 1, i, 0) for i in range(1, 6)], freq="D"),
    )
    # 对时间序列进行按日重采样，并计算滚动窗口大小为1的中位数
    x = series.resample("D").median().rolling(window=1).median()
    # 使用测试工具比较预期结果和实际结果的中位数序列
    tm.assert_series_equal(expected, x)


def test_rolling_median_memory_error():
    # GH11722
    # 创建一个包含20000个随机标准正态分布值的时间序列，计算滚动窗口大小为2的中位数
    n = 20000
    Series(np.random.default_rng(2).standard_normal(n)).rolling(
        window=2, center=False
    ).median()
    Series(np.random.default_rng(2).standard_normal(n)).rolling(
        window=2, center=False
    ).median()


def test_rolling_min_max_numeric_types(any_real_numpy_dtype):
    # GH12373

    # 只测试这些操作不会抛出异常，并且返回类型为float64。其他测试将覆盖定量正确性
    # 创建一个包含0到19的numpy数组，数据类型为any_real_numpy_dtype，计算滚动窗口大小为5的最大值
    result = (
        DataFrame(np.arange(20, dtype=any_real_numpy_dtype)).rolling(window=5).max()
    )
    # 断言结果的第一个列的数据类型为float64
    assert result.dtypes[0] == np.dtype("f8")
    # 创建一个包含0到19的numpy数组，数据类型为any_real_numpy_dtype，计算滚动窗口大小为5的最小值
    result = (
        DataFrame(np.arange(20, dtype=any_real_numpy_dtype)).rolling(window=5).min()
    )
    # 断言结果的第一个列的数据类型为float64
    assert result.dtypes[0] == np.dtype("f8")


@pytest.mark.parametrize(
    "f",
    [
        # 返回滚动窗口中的非空观测数，窗口大小为10，从0开始计算
        lambda x: x.rolling(window=10, min_periods=0).count(),
        # 返回滚动窗口中的协方差矩阵，窗口大小为10，至少包含5个观测
        lambda x: x.rolling(window=10, min_periods=5).cov(x, pairwise=False),
        # 返回滚动窗口中的相关系数矩阵，窗口大小为10，至少包含5个观测
        lambda x: x.rolling(window=10, min_periods=5).corr(x, pairwise=False),
        # 返回滚动窗口中的最大值，窗口大小为10，至少包含5个观测
        lambda x: x.rolling(window=10, min_periods=5).max(),
        # 返回滚动窗口中的最小值，窗口大小为10，至少包含5个观测
        lambda x: x.rolling(window=10, min_periods=5).min(),
        # 返回滚动窗口中的总和，窗口大小为10，至少包含5个观测
        lambda x: x.rolling(window=10, min_periods=5).sum(),
        # 返回滚动窗口中的均值，窗口大小为10，至少包含5个观测
        lambda x: x.rolling(window=10, min_periods=5).mean(),
        # 返回滚动窗口中的标准差，窗口大小为10，至少包含5个观测
        lambda x: x.rolling(window=10, min_periods=5).std(),
        # 返回滚动窗口中的方差，窗口大小为10，至少包含5个观测
        lambda x: x.rolling(window=10, min_periods=5).var(),
        # 返回滚动窗口中的偏度，窗口大小为10，至少包含5个观测
        lambda x: x.rolling(window=10, min_periods=5).skew(),
        # 返回滚动窗口中的峰度，窗口大小为10，至少包含5个观测
        lambda x: x.rolling(window=10, min_periods=5).kurt(),
        # 返回滚动窗口中的中位数，窗口大小为10，至少包含5个观测
        lambda x: x.rolling(window=10, min_periods=5).quantile(0.5),
        # 返回滚动窗口中的中位数，窗口大小为10，至少包含5个观测
        lambda x: x.rolling(window=10, min_periods=5).median(),
        # 应用指定函数（这里为求和）到滚动窗口中的值，raw=False意味着传递数组到函数
        lambda x: x.rolling(window=10, min_periods=5).apply(sum, raw=False),
        # 应用指定函数（这里为求和）到滚动窗口中的值，raw=True意味着传递原始的ndarray到函数
        lambda x: x.rolling(window=10, min_periods=5).apply(sum, raw=True),
        # 使用boxcar权重函数计算滚动窗口中的均值，窗口大小为10，至少包含5个观测
        pytest.param(
            lambda x: x.rolling(win_type="boxcar", window=10, min_periods=5).mean(),
            marks=td.skip_if_no("scipy"),
        ),
    ],
)
def test_moment_functions_zero_length(f):
    # 定义一个测试函数，用于测试处理长度为零的情况
    # GH 8056 是 GitHub 上的一个问题编号
    # 创建一个空的 Series 对象，数据类型为 float64
    s = Series(dtype=np.float64)
    # 预期的结果与 s 相同
    s_expected = s
    # 创建一个空的 DataFrame 对象
    df1 = DataFrame()
    # 预期的结果与 df1 相同
    df1_expected = df1
    # 创建一个包含一列名为 "a" 的 DataFrame 对象，并将其类型转换为 float64
    df2 = DataFrame(columns=["a"])
    df2["a"] = df2["a"].astype("float64")
    # 预期的结果与 df2 相同
    df2_expected = df2

    # 对 s 应用函数 f，得到结果 s_result，并验证其与预期结果 s_expected 相等
    s_result = f(s)
    tm.assert_series_equal(s_result, s_expected)

    # 对 df1 应用函数 f，得到结果 df1_result，并验证其与预期结果 df1_expected 相等
    df1_result = f(df1)
    tm.assert_frame_equal(df1_result, df1_expected)

    # 对 df2 应用函数 f，得到结果 df2_result，并验证其与预期结果 df2_expected 相等
    df2_result = f(df2)
    tm.assert_frame_equal(df2_result, df2_expected)
```