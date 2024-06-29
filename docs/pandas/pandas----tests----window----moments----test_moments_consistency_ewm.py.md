# `D:\src\scipysrc\pandas\pandas\tests\window\moments\test_moments_consistency_ewm.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试

from pandas import (  # 从 Pandas 库中导入以下几个对象
    DataFrame,  # 数据帧对象，用于处理二维表格数据
    Series,  # 系列对象，用于处理一维标签数组
    concat,  # 用于沿指定轴连接 Pandas 对象的函数
)
import pandas._testing as tm  # 导入 Pandas 内部测试工具模块


def create_mock_weights(obj, com, adjust, ignore_na):
    # 创建模拟权重函数，根据输入对象的类型（DataFrame 或 Series）生成相应权重
    if isinstance(obj, DataFrame):  # 如果输入对象是 DataFrame 类型
        if not len(obj.columns):  # 如果 DataFrame 没有列
            return DataFrame(index=obj.index, columns=obj.columns)  # 返回一个空的 DataFrame
        # 否则，对 DataFrame 的每一列调用 create_mock_series_weights 函数，并连接结果
        w = concat(
            [
                create_mock_series_weights(
                    obj.iloc[:, i], com=com, adjust=adjust, ignore_na=ignore_na
                )
                for i in range(len(obj.columns))
            ],
            axis=1,
        )
        w.index = obj.index  # 设置生成的权重的索引与输入 DataFrame 的索引一致
        w.columns = obj.columns  # 设置生成的权重的列名与输入 DataFrame 的列名一致
        return w  # 返回生成的权重 DataFrame
    else:  # 如果输入对象是 Series 类型
        return create_mock_series_weights(obj, com, adjust, ignore_na)  # 调用 create_mock_series_weights 处理 Series 对象


def create_mock_series_weights(s, com, adjust, ignore_na):
    # 创建模拟系列权重函数，根据输入系列生成相应权重
    w = Series(np.nan, index=s.index, name=s.name)  # 创建一个 NaN 值的 Series 权重对象
    alpha = 1.0 / (1.0 + com)  # 计算指数加权平均的衰减因子 alpha
    if adjust:  # 如果需要调整
        count = 0  # 初始化计数器
        for i in range(len(s)):  # 遍历系列对象的每个元素
            if s.iat[i] == s.iat[i]:  # 如果元素不是 NaN
                w.iat[i] = pow(1.0 / (1.0 - alpha), count)  # 计算权重值
                count += 1  # 更新计数器
            elif not ignore_na:  # 如果元素是 NaN 且不忽略 NaN 值
                count += 1  # 更新计数器
    else:  # 如果不需要调整
        sum_wts = 0.0  # 初始化权重和
        prev_i = -1  # 初始化前一个有效元素的索引
        count = 0  # 初始化计数器
        for i in range(len(s)):  # 遍历系列对象的每个元素
            if s.iat[i] == s.iat[i]:  # 如果元素不是 NaN
                if prev_i == -1:  # 如果是第一个有效元素
                    w.iat[i] = 1.0  # 直接赋值权重为 1.0
                else:  # 如果不是第一个有效元素
                    w.iat[i] = alpha * sum_wts / pow(1.0 - alpha, count - prev_i)  # 计算权重值
                sum_wts += w.iat[i]  # 更新权重和
                prev_i = count  # 更新前一个有效元素的索引
                count += 1  # 更新计数器
            elif not ignore_na:  # 如果元素是 NaN 且不忽略 NaN 值
                count += 1  # 更新计数器
    return w  # 返回生成的权重 Series 对象


def test_ewm_consistency_mean(all_data, adjust, ignore_na, min_periods):
    # 测试指数加权移动平均的一致性（均值）
    com = 3.0  # 设置衰减因子

    # 计算实际结果
    result = all_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).mean()
    # 创建模拟权重
    weights = create_mock_weights(all_data, com=com, adjust=adjust, ignore_na=ignore_na)
    # 期望结果为加权后的累积和除以加权累积和
    expected = all_data.multiply(weights).cumsum().divide(weights.cumsum()).ffill()
    expected[
        all_data.expanding().count() < (max(min_periods, 1) if min_periods else 1)
    ] = np.nan
    # 使用测试工具模块检查结果是否一致
    tm.assert_equal(result, expected.astype("float64"))


def test_ewm_consistency_consistent(consistent_data, adjust, ignore_na, min_periods):
    # 测试指数加权移动平均的一致性（一致性）
    com = 3.0  # 设置衰减因子

    # 计算展开计数和指数加权移动平均值
    count_x = consistent_data.expanding().count()
    mean_x = consistent_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).mean()
    # 检查系列与自身的相关性为 1 或 NaN
    corr_x_x = consistent_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).corr(consistent_data)
    exp = (
        consistent_data.max()
        if isinstance(consistent_data, Series)
        else consistent_data.max().max()
    )
    # 检查常数系列的均值
    expected = consistent_data * np.nan
    expected[count_x >= max(min_periods, 1)] = exp
    # 使用测试工具模块检查结果是否一致
    tm.assert_equal(mean_x, expected)
    # 将 expected 数组中的所有元素设置为 NaN
    expected[:] = np.nan
    # 使用测试框架中的 assert_equal 函数，比较 corr_x_x 和 expected 是否相等
    tm.assert_equal(corr_x_x, expected)
def test_ewm_consistency_var_debiasing_factors(
    all_data, adjust, ignore_na, min_periods
):
    # 设置指数加权移动平均窗口参数的常数值
    com = 3.0

    # 检查方差去偏因子
    # 计算不偏方差
    var_unbiased_x = all_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).var(bias=False)
    # 计算有偏方差
    var_biased_x = all_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).var(bias=True)

    # 创建模拟权重
    weights = create_mock_weights(all_data, com=com, adjust=adjust, ignore_na=ignore_na)
    # 计算权重的累积和和累积平方和
    cum_sum = weights.cumsum().ffill()
    cum_sum_sq = (weights * weights).cumsum().ffill()
    # 计算分子
    numerator = cum_sum * cum_sum
    # 计算分母
    denominator = numerator - cum_sum_sq
    # 处理分母为非正数的情况
    denominator[denominator <= 0.0] = np.nan
    # 计算方差去偏因子
    var_debiasing_factors_x = numerator / denominator

    # 断言验证方差无偏结果与有偏结果乘以去偏因子的结果一致性
    tm.assert_equal(var_unbiased_x, var_biased_x * var_debiasing_factors_x)


@pytest.mark.parametrize("bias", [True, False])
def test_moments_consistency_var(all_data, adjust, ignore_na, min_periods, bias):
    # 设置指数加权移动平均窗口参数的常数值
    com = 3.0

    # 计算指数加权移动平均的均值
    mean_x = all_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).mean()
    # 计算指数加权移动平均的方差
    var_x = all_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).var(bias=bias)
    # 断言验证方差非负
    assert not (var_x < 0).any().any()

    if bias:
        # 验证有偏方差等于均值的平方的期望减去均值的平方
        mean_x2 = (
            (all_data * all_data)
            .ewm(com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na)
            .mean()
        )
        tm.assert_equal(var_x, mean_x2 - (mean_x * mean_x))


@pytest.mark.parametrize("bias", [True, False])
def test_moments_consistency_var_constant(
    consistent_data, adjust, ignore_na, min_periods, bias
):
    # 设置指数加权移动平均窗口参数的常数值
    com = 3.0
    # 计算累积计数
    count_x = consistent_data.expanding(min_periods=min_periods).count()
    # 计算指数加权移动平均的方差
    var_x = consistent_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).var(bias=bias)

    # 断言验证常数序列的方差为0
    assert not (var_x > 0).any().any()
    # 设置预期结果为常数乘以 NaN
    expected = consistent_data * np.nan
    expected[count_x >= max(min_periods, 1)] = 0.0
    if not bias:
        expected[count_x < 2] = np.nan
    tm.assert_equal(var_x, expected)


@pytest.mark.parametrize("bias", [True, False])
def test_ewm_consistency_std(all_data, adjust, ignore_na, min_periods, bias):
    # 设置指数加权移动平均窗口参数的常数值
    com = 3.0
    # 计算指数加权移动平均的方差
    var_x = all_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).var(bias=bias)
    # 断言验证方差非负
    assert not (var_x < 0).any().any()

    # 计算指数加权移动平均的标准差
    std_x = all_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).std(bias=bias)
    # 断言验证标准差非负
    assert not (std_x < 0).any().any()

    # 断言验证方差等于标准差的平方
    tm.assert_equal(var_x, std_x * std_x)

    # 计算指数加权移动平均的协方差
    cov_x_x = all_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).cov(all_data, bias=bias)
    # 断言验证协方差非负
    assert not (cov_x_x < 0).any().any()
    # 检查变量 x 的方差是否等于 x 与自身的协方差
    tm.assert_equal(var_x, cov_x_x)
# 使用参数化测试，测试指数加权移动平均（EWMA）在计算方差、协方差和相关性时的一致性
@pytest.mark.parametrize("bias", [True, False])
def test_ewm_consistency_series_cov_corr(
    series_data, adjust, ignore_na, min_periods, bias
):
    # 设置 EWMA 的参数 com
    com = 3.0

    # 计算序列数据与自身的加权移动平均的方差
    var_x_plus_y = (
        (series_data + series_data)
        .ewm(com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na)
        .var(bias=bias)
    )

    # 计算序列数据的加权移动平均的方差
    var_x = series_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).var(bias=bias)

    # 同上，计算另一个序列数据的加权移动平均的方差
    var_y = series_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).var(bias=bias)

    # 计算序列数据之间的加权移动平均的协方差
    cov_x_y = series_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).cov(series_data, bias=bias)

    # 检查 cov(x, y) 是否满足关系：cov(x, y) == (var(x+y) - var(x) - var(y)) / 2
    tm.assert_equal(cov_x_y, 0.5 * (var_x_plus_y - var_x - var_y))

    # 检查 corr(x, y) 是否满足关系：corr(x, y) == cov(x, y) / (std(x) * std(y))
    corr_x_y = series_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).corr(series_data)

    # 计算序列数据的标准差
    std_x = series_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).std(bias=bias)
    std_y = series_data.ewm(
        com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
    ).std(bias=bias)

    # 检查 corr(x, y) 是否满足关系：corr(x, y) == cov(x, y) / (std(x) * std(y))
    tm.assert_equal(corr_x_y, cov_x_y / (std_x * std_y))

    if bias:
        # 检查有偏的 cov(x, y) 是否满足关系：cov(x, y) == mean(x*y) - mean(x)*mean(y)
        mean_x = series_data.ewm(
            com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
        ).mean()
        mean_y = series_data.ewm(
            com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na
        ).mean()

        # 计算序列数据与自身乘积的加权移动平均的均值
        mean_x_times_y = (
            (series_data * series_data)
            .ewm(com=com, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na)
            .mean()
        )

        # 检查 cov(x, y) 是否满足关系：cov(x, y) == mean(x*y) - mean(x)*mean(y)
        tm.assert_equal(cov_x_y, mean_x_times_y - (mean_x * mean_y))
```