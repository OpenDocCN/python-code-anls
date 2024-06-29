# `D:\src\scipysrc\pandas\pandas\tests\window\moments\test_moments_consistency_expanding.py`

```
import numpy as np
import pytest

from pandas import Series
import pandas._testing as tm


# 检查给定的 Series x 是否没有 NaN 值
def no_nans(x):
    return x.notna().all().all()


# 检查给定的 Series x 是否全是 NaN 值
def all_na(x):
    return x.isnull().all().all()


# 测试函数，参数化使用不同的函数 f 进行测试
@pytest.mark.parametrize("f", [lambda v: Series(v).sum(), np.nansum, np.sum])
def test_expanding_apply_consistency_sum_nans(request, all_data, min_periods, f):
    if f is np.sum:
        # 如果数据中存在 NaN 值，并且不是全是 NaN，并且数据不为空且最小期数大于 0
        if not no_nans(all_data) and not (
            all_na(all_data) and not all_data.empty and min_periods > 0
        ):
            # 标记为预期失败，原因是 np.sum 在处理 NaN 时行为不同
            request.applymarker(
                pytest.mark.xfail(reason="np.sum has different behavior with NaNs")
            )
    # 计算数据的累积和
    expanding_f_result = all_data.expanding(min_periods=min_periods).sum()
    # 使用给定的函数 f 对数据进行累积应用
    expanding_apply_f_result = all_data.expanding(min_periods=min_periods).apply(
        func=f, raw=True
    )
    # 断言两种计算结果应该相等
    tm.assert_equal(expanding_f_result, expanding_apply_f_result)


# 测试函数，参数化使用不同的 ddof 进行方差一致性测试
@pytest.mark.parametrize("ddof", [0, 1])
def test_moments_consistency_var(all_data, min_periods, ddof):
    # 计算数据的累积方差
    var_x = all_data.expanding(min_periods=min_periods).var(ddof=ddof)
    # 断言方差不应小于 0
    assert not (var_x < 0).any().any()

    if ddof == 0:
        # 当 ddof 为 0 时，检查有偏方差是否等于均值的平方减均值平方的结果
        mean_x2 = (all_data * all_data).expanding(min_periods=min_periods).mean()
        mean_x = all_data.expanding(min_periods=min_periods).mean()
        tm.assert_equal(var_x, mean_x2 - (mean_x * mean_x))


# 测试函数，参数化使用不同的 ddof 进行常数数据的方差一致性测试
@pytest.mark.parametrize("ddof", [0, 1])
def test_moments_consistency_var_constant(consistent_data, min_periods, ddof):
    # 计算数据的累积计数
    count_x = consistent_data.expanding(min_periods=min_periods).count()
    # 计算数据的累积方差
    var_x = consistent_data.expanding(min_periods=min_periods).var(ddof=ddof)

    # 断言方差不应大于 0
    assert not (var_x > 0).any().any()
    # 期望的方差结果，常数系列的方差应全为 0
    expected = consistent_data * np.nan
    expected[count_x >= max(min_periods, 1)] = 0.0
    if ddof == 1:
        expected[count_x < 2] = np.nan
    tm.assert_equal(var_x, expected)


# 测试函数，参数化使用不同的 ddof 进行方差、标准差和协方差的一致性测试
@pytest.mark.parametrize("ddof", [0, 1])
def test_expanding_consistency_var_std_cov(all_data, min_periods, ddof):
    # 计算数据的累积方差
    var_x = all_data.expanding(min_periods=min_periods).var(ddof=ddof)
    # 断言方差不应小于 0
    assert not (var_x < 0).any().any()

    # 计算数据的累积标准差
    std_x = all_data.expanding(min_periods=min_periods).std(ddof=ddof)
    # 断言标准差不应小于 0
    assert not (std_x < 0).any().any()

    # 检查方差是否等于标准差的平方
    tm.assert_equal(var_x, std_x * std_x)

    # 计算数据与自身的累积协方差
    cov_x_x = all_data.expanding(min_periods=min_periods).cov(all_data, ddof=ddof)
    # 断言协方差不应小于 0
    assert not (cov_x_x < 0).any().any()

    # 检查方差是否等于数据与自身的协方差
    tm.assert_equal(var_x, cov_x_x)


# 测试函数，参数化使用不同的 ddof 进行系列数据的协方差和相关性一致性测试
@pytest.mark.parametrize("ddof", [0, 1])
def test_expanding_consistency_series_cov_corr(series_data, min_periods, ddof):
    # 计算序列数据加和后的累积方差
    var_x_plus_y = (
        (series_data + series_data).expanding(min_periods=min_periods).var(ddof=ddof)
    )
    # 计算序列数据的累积方差
    var_x = series_data.expanding(min_periods=min_periods).var(ddof=ddof)
    var_y = series_data.expanding(min_periods=min_periods).var(ddof=ddof)
    # 计算扩展期间的累积协方差 cov(x, y)，考虑最小期间数和自由度修正
    cov_x_y = series_data.expanding(min_periods=min_periods).cov(series_data, ddof=ddof)
    # 检查是否满足 cov(x, y) == (var(x+y) - var(x) - var(y)) / 2 的关系
    tm.assert_equal(cov_x_y, 0.5 * (var_x_plus_y - var_x - var_y))

    # 计算扩展期间的累积相关系数 corr(x, y)，考虑最小期间数
    corr_x_y = series_data.expanding(min_periods=min_periods).corr(series_data)
    # 计算扩展期间的标准差 std(x) 和 std(y)，考虑自由度修正
    std_x = series_data.expanding(min_periods=min_periods).std(ddof=ddof)
    std_y = series_data.expanding(min_periods=min_periods).std(ddof=ddof)
    # 检查是否满足 corr(x, y) == cov(x, y) / (std(x) * std(y)) 的关系
    tm.assert_equal(corr_x_y, cov_x_y / (std_x * std_y))

    if ddof == 0:
        # 检查是否满足偏倚估计下的协方差关系 biased cov(x, y) == mean(x*y) - mean(x)*mean(y)
        mean_x = series_data.expanding(min_periods=min_periods).mean()
        mean_y = series_data.expanding(min_periods=min_periods).mean()
        # 计算扩展期间内 x*y 的均值
        mean_x_times_y = (
            (series_data * series_data).expanding(min_periods=min_periods).mean()
        )
        # 检查是否满足偏倚估计下的协方差关系
        tm.assert_equal(cov_x_y, mean_x_times_y - (mean_x * mean_y))
# 定义函数，测试数据的扩展一致性的均值
def test_expanding_consistency_mean(all_data, min_periods):
    # 计算数据的扩展均值，考虑最小周期
    result = all_data.expanding(min_periods=min_periods).mean()
    # 计算期望值，为数据扩展总和除以数据扩展计数
    expected = (
        all_data.expanding(min_periods=min_periods).sum()
        / all_data.expanding(min_periods=min_periods).count()
    )
    # 断言结果与期望相等，且结果类型为 float64
    tm.assert_equal(result, expected.astype("float64"))


# 定义函数，测试一致性数据的扩展常数
def test_expanding_consistency_constant(consistent_data, min_periods):
    # 计算数据扩展后的计数
    count_x = consistent_data.expanding().count()
    # 计算数据扩展后的均值，考虑最小周期
    mean_x = consistent_data.expanding(min_periods=min_periods).mean()
    # 检查序列与自身的相关性，应为 1 或 NaN
    corr_x_x = consistent_data.expanding(min_periods=min_periods).corr(consistent_data)

    # 如果 consistent_data 是 Series 类型，取其最大值，否则取其内部序列的最大值
    exp = (
        consistent_data.max()
        if isinstance(consistent_data, Series)
        else consistent_data.max().max()
    )

    # 检查常数序列的均值预期结果
    expected = consistent_data * np.nan
    expected[count_x >= max(min_periods, 1)] = exp
    # 断言计算的均值与期望值相等
    tm.assert_equal(mean_x, expected)

    # 检查常数序列与自身的相关性，预期结果应为 NaN
    expected[:] = np.nan
    tm.assert_equal(corr_x_x, expected)


# 定义函数，测试数据的扩展一致性的方差去偏因子
def test_expanding_consistency_var_debiasing_factors(all_data, min_periods):
    # 计算无偏方差
    var_unbiased_x = all_data.expanding(min_periods=min_periods).var()
    # 计算有偏方差
    var_biased_x = all_data.expanding(min_periods=min_periods).var(ddof=0)
    # 计算方差去偏因子
    var_debiasing_factors_x = all_data.expanding().count() / (
        all_data.expanding().count() - 1.0
    ).replace(0.0, np.nan)
    # 断言无偏方差等于有偏方差乘以方差去偏因子
    tm.assert_equal(var_unbiased_x, var_biased_x * var_debiasing_factors_x)
```