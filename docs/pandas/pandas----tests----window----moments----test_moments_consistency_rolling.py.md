# `D:\src\scipysrc\pandas\pandas\tests\window\moments\test_moments_consistency_rolling.py`

```
# 导入必要的库
import numpy as np
import pytest

# 从 pandas 库中导入 Series 类
from pandas import Series

# 导入 pandas 内部测试模块
import pandas._testing as tm


# 检查参数 x 是否没有 NaN 值
def no_nans(x):
    return x.notna().all().all()


# 检查参数 x 是否全为 NaN 值
def all_na(x):
    return x.isnull().all().all()


# 定义一个 pytest 的 fixture，用于提供滚动一致性测试的参数组合
@pytest.fixture(params=[(1, 0), (5, 1)])
def rolling_consistency_cases(request):
    """window, min_periods"""
    return request.param


# 参数化测试：测试滚动应用函数求和的一致性
@pytest.mark.parametrize("f", [lambda v: Series(v).sum(), np.nansum, np.sum])
def test_rolling_apply_consistency_sum(
    request, all_data, rolling_consistency_cases, center, f
):
    window, min_periods = rolling_consistency_cases

    # 对于 np.sum 函数，如果数据集 all_data 中既不全是 NaN 也不全为空且 min_periods 大于 0，
    # 标记该测试为预期失败，因为 np.sum 对 NaN 值的处理行为不同
    if f is np.sum:
        if not no_nans(all_data) and not (
            all_na(all_data) and not all_data.empty and min_periods > 0
        ):
            request.applymarker(
                pytest.mark.xfail(reason="np.sum has different behavior with NaNs")
            )

    # 执行滚动求和计算
    rolling_f_result = all_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).sum()

    # 执行滚动应用函数 f（可能是 Series(v).sum()、np.nansum 或 np.sum）
    rolling_apply_f_result = all_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).apply(func=f, raw=True)

    # 使用测试模块中的方法验证两者结果的一致性
    tm.assert_equal(rolling_f_result, rolling_apply_f_result)


# 参数化测试：测试滚动计算方差的一致性
@pytest.mark.parametrize("ddof", [0, 1])
def test_moments_consistency_var(all_data, rolling_consistency_cases, center, ddof):
    window, min_periods = rolling_consistency_cases

    # 计算滚动方差
    var_x = all_data.rolling(window=window, min_periods=min_periods, center=center).var(
        ddof=ddof
    )

    # 断言方差不为负数
    assert not (var_x < 0).any().any()

    if ddof == 0:
        # 对于无偏方差，检查 biased var(x) == mean(x^2) - mean(x)^2
        mean_x = all_data.rolling(
            window=window, min_periods=min_periods, center=center
        ).mean()
        mean_x2 = (
            (all_data * all_data)
            .rolling(window=window, min_periods=min_periods, center=center)
            .mean()
        )
        tm.assert_equal(var_x, mean_x2 - (mean_x * mean_x))


# 参数化测试：测试滚动计算常数序列方差的一致性
@pytest.mark.parametrize("ddof", [0, 1])
def test_moments_consistency_var_constant(
    consistent_data, rolling_consistency_cases, center, ddof
):
    window, min_periods = rolling_consistency_cases

    # 计算常数序列的计数和方差
    count_x = consistent_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).count()
    var_x = consistent_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).var(ddof=ddof)

    # 断言方差不大于 0
    assert not (var_x > 0).any().any()

    # 预期的结果应当是常数序列的 NaN 值
    expected = consistent_data * np.nan
    expected[count_x >= max(min_periods, 1)] = 0.0
    if ddof == 1:
        expected[count_x < 2] = np.nan
    tm.assert_equal(var_x, expected)


# 参数化测试：测试滚动计算方差、标准差、协方差的一致性
@pytest.mark.parametrize("ddof", [0, 1])
def test_rolling_consistency_var_std_cov(
    all_data, rolling_consistency_cases, center, ddof
):
    window, min_periods = rolling_consistency_cases

    # 计算滚动方差
    var_x = all_data.rolling(window=window, min_periods=min_periods, center=center).var(
        ddof=ddof
    )

    # 断言方差不为负数
    assert not (var_x < 0).any().any()
    # 计算滚动窗口内数据的标准差
    std_x = all_data.rolling(window=window, min_periods=min_periods, center=center).std(
        ddof=ddof
    )
    # 断言：确保标准差非负
    assert not (std_x < 0).any().any()

    # 检查方差与标准差的平方是否相等
    tm.assert_equal(var_x, std_x * std_x)

    # 计算滚动窗口内数据的协方差
    cov_x_x = all_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).cov(all_data, ddof=ddof)
    # 断言：确保协方差非负
    assert not (cov_x_x < 0).any().any()

    # 检查方差与自己的协方差是否相等
    tm.assert_equal(var_x, cov_x_x)
# 使用 pytest.mark.parametrize 装饰器，为测试函数 test_rolling_consistency_series_cov_corr 参数化，参数是 ddof，分别为 0 和 1
@pytest.mark.parametrize("ddof", [0, 1])
# 定义测试函数 test_rolling_consistency_series_cov_corr，参数包括 series_data、rolling_consistency_cases、center、ddof
def test_rolling_consistency_series_cov_corr(
    series_data, rolling_consistency_cases, center, ddof
):
    # 解包 rolling_consistency_cases 参数为 window 和 min_periods
    window, min_periods = rolling_consistency_cases

    # 计算 series_data + series_data 的滚动方差，使用窗口大小 window 和最小有效数据点数 min_periods，以 center 为中心
    var_x_plus_y = (
        (series_data + series_data)
        .rolling(window=window, min_periods=min_periods, center=center)
        .var(ddof=ddof)
    )
    # 计算 series_data 的滚动方差，使用窗口大小 window 和最小有效数据点数 min_periods，以 center 为中心
    var_x = series_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).var(ddof=ddof)
    # 计算 series_data 的滚动方差，使用窗口大小 window 和最小有效数据点数 min_periods，以 center 为中心
    var_y = series_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).var(ddof=ddof)
    # 计算 series_data 与自身的滚动协方差，使用窗口大小 window 和最小有效数据点数 min_periods，以 center 为中心
    cov_x_y = series_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).cov(series_data, ddof=ddof)
    # 检查 cov(x, y) 是否等于 (var(x+y) - var(x) - var(y)) / 2
    tm.assert_equal(cov_x_y, 0.5 * (var_x_plus_y - var_x - var_y))

    # 检查 corr(x, y) 是否等于 cov(x, y) / (std(x) * std(y))
    corr_x_y = series_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).corr(series_data)
    # 计算 series_data 的滚动标准差，使用窗口大小 window 和最小有效数据点数 min_periods，以 center 为中心
    std_x = series_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).std(ddof=ddof)
    # 计算 series_data 的滚动标准差，使用窗口大小 window 和最小有效数据点数 min_periods，以 center 为中心
    std_y = series_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).std(ddof=ddof)
    # 检查 corr(x, y) 是否等于 cov(x, y) / (std(x) * std(y))
    tm.assert_equal(corr_x_y, cov_x_y / (std_x * std_y))

    if ddof == 0:
        # 检查有偏的 cov(x, y) 是否等于 mean(x*y) - mean(x)*mean(y)
        # 计算 series_data 的滚动均值，使用窗口大小 window 和最小有效数据点数 min_periods，以 center 为中心
        mean_x = series_data.rolling(
            window=window, min_periods=min_periods, center=center
        ).mean()
        # 计算 series_data 的滚动均值，使用窗口大小 window 和最小有效数据点数 min_periods，以 center 为中心
        mean_y = series_data.rolling(
            window=window, min_periods=min_periods, center=center
        ).mean()
        # 计算 series_data * series_data 的滚动均值，使用窗口大小 window 和最小有效数据点数 min_periods，以 center 为中心
        mean_x_times_y = (
            (series_data * series_data)
            .rolling(window=window, min_periods=min_periods, center=center)
            .mean()
        )
        # 检查 cov(x, y) 是否等于 mean(x*y) - mean(x)*mean(y)
        tm.assert_equal(cov_x_y, mean_x_times_y - (mean_x * mean_y))


# 定义测试函数 test_rolling_consistency_mean，参数包括 all_data、rolling_consistency_cases、center
def test_rolling_consistency_mean(all_data, rolling_consistency_cases, center):
    # 解包 rolling_consistency_cases 参数为 window 和 min_periods
    window, min_periods = rolling_consistency_cases

    # 计算 all_data 的滚动均值，使用窗口大小 window 和最小有效数据点数 min_periods，以 center 为中心
    result = all_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).mean()
    # 计算预期的 all_data 的滚动均值，使用窗口大小 window 和最小有效数据点数 min_periods，以 center 为中心
    expected = (
        all_data.rolling(window=window, min_periods=min_periods, center=center)
        .sum()
        .divide(
            all_data.rolling(
                window=window, min_periods=min_periods, center=center
            ).count()
        )
    )
    # 检查计算结果是否与预期一致，预期结果的类型转换为 "float64"
    tm.assert_equal(result, expected.astype("float64"))


# 定义测试函数 test_rolling_consistency_constant，参数包括 consistent_data、rolling_consistency_cases、center
def test_rolling_consistency_constant(
    consistent_data, rolling_consistency_cases, center
):
    # 解包 rolling_consistency_cases 参数为 window 和 min_periods
    window, min_periods = rolling_consistency_cases

    # 计算 consistent_data 的滚动计数，使用窗口大小 window 和最小有效数据点数 min_periods，以 center 为中心
    count_x = consistent_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).count()
    # 计算 consistent_data 的滚动均值，使用窗口大小 window 和最小有效数据点数 min_periods，以 center 为中心
    mean_x = consistent_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).mean()
    # 检查一系列与自身的相关性是否为 1 或 NaN
    # 计算数据的滚动窗口相关系数
    corr_x_x = consistent_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).corr(consistent_data)

    # 计算数据的最大值
    exp = (
        consistent_data.max()
        if isinstance(consistent_data, Series)
        else consistent_data.max().max()
    )

    # 创建一个与 consistent_data 形状相同的空数组 expected，并将其填充为 NaN
    expected = consistent_data * np.nan
    
    # 将大于等于指定最小周期（min_periods 或者 1，取较大者）的部分数据设置为 exp
    expected[count_x >= max(min_periods, 1)] = exp
    
    # 使用断言确认 mean_x 与 expected 相等
    tm.assert_equal(mean_x, expected)

    # 将 expected 数组全部设置为 NaN，用于检查常数系列与自身的相关性是否为 NaN
    expected[:] = np.nan
    
    # 使用断言确认 corr_x_x（常数系列与自身的相关系数）与 expected 相等
    tm.assert_equal(corr_x_x, expected)
def test_rolling_consistency_var_debiasing_factors(
    all_data, rolling_consistency_cases, center
):
    window, min_periods = rolling_consistency_cases  # 从 rolling_consistency_cases 中获取窗口大小和最小周期数

    # 计算不偏方差
    var_unbiased_x = all_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).var()
    
    # 计算有偏方差
    var_biased_x = all_data.rolling(
        window=window, min_periods=min_periods, center=center
    ).var(ddof=0)
    
    # 计算修正因子
    var_debiasing_factors_x = (
        all_data.rolling(window=window, min_periods=min_periods, center=center)
        .count()  # 计算窗口内非缺失值的数量
        .divide(
            (
                all_data.rolling(
                    window=window, min_periods=min_periods, center=center
                ).count()  # 再次计算窗口内非缺失值的数量
                - 1.0  # 减去1，用于修正偏差
            ).replace(0.0, np.nan)  # 将分母为0的情况替换为 NaN，避免除零错误
        )
    )
    
    # 断言修正后的不偏方差等于有偏方差乘以修正因子
    tm.assert_equal(var_unbiased_x, var_biased_x * var_debiasing_factors_x)
```