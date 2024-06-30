# `D:\src\scipysrc\scipy\scipy\stats\tests\test_entropy.py`

```
import math
import pytest
from pytest import raises as assert_raises

import numpy as np
from numpy.testing import assert_allclose

from scipy import stats
from scipy.conftest import array_api_compatible
from scipy._lib._array_api import xp_assert_close, xp_assert_equal, xp_assert_less

class TestEntropy:
    @array_api_compatible
    def test_entropy_positive(self, xp):
        # See ticket #497
        # 创建包含概率分布的数组 pk 和 qk
        pk = xp.asarray([0.5, 0.2, 0.3])
        qk = xp.asarray([0.1, 0.25, 0.65])
        # 计算自熵和交叉熵
        eself = stats.entropy(pk, pk)
        edouble = stats.entropy(pk, qk)
        # 使用断言检查自熵应该为零，交叉熵应该小于零
        xp_assert_equal(eself, xp.asarray(0.))
        xp_assert_less(-edouble, xp.asarray(0.))

    @array_api_compatible
    def test_entropy_base(self, xp):
        # 创建全为1的数组 pk
        pk = xp.ones(16)
        # 计算以不同基数为底的熵
        S = stats.entropy(pk, base=2.)
        # 使用断言检查计算结果是否接近期望值
        xp_assert_less(xp.abs(S - 4.), xp.asarray(1.e-5))

        # 创建全为1的数组 qk，并将一部分元素替换为2
        qk = xp.ones(16)
        qk = xp.where(xp.arange(16) < 8, xp.asarray(2.), qk)
        # 计算 pk 和 qk 的熵，并使用不同基数的熵来比较
        S = stats.entropy(pk, qk)
        S2 = stats.entropy(pk, qk, base=2.)
        # 使用断言检查相对误差是否在可接受范围内
        xp_assert_less(xp.abs(S/S2 - math.log(2.)), xp.asarray(1.e-5))

    @array_api_compatible
    def test_entropy_zero(self, xp):
        # 测试对于包含零元素的数组 x 的熵计算
        x = xp.asarray([0., 1., 2.])
        # 使用断言检查熵计算结果是否接近期望值
        xp_assert_close(stats.entropy(x),
                        xp.asarray(0.63651416829481278))

    @array_api_compatible
    def test_entropy_2d(self, xp):
        # 创建二维数组 pk 和 qk，计算它们之间的熵
        pk = xp.asarray([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
        qk = xp.asarray([[0.2, 0.1], [0.3, 0.6], [0.5, 0.3]])
        # 使用断言检查计算得到的熵是否接近期望值数组
        xp_assert_close(stats.entropy(pk, qk),
                        xp.asarray([0.1933259, 0.18609809]))

    @array_api_compatible
    def test_entropy_2d_zero(self, xp):
        # 创建包含零元素的二维数组 pk 和 qk，测试熵计算
        pk = xp.asarray([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
        qk = xp.asarray([[0.0, 0.1], [0.3, 0.6], [0.5, 0.3]])
        # 使用断言检查熵计算结果是否接近期望值数组，注意处理无穷大的情况
        xp_assert_close(stats.entropy(pk, qk),
                        xp.asarray([xp.inf, 0.18609809]))

        # 将 pk 的部分元素替换为零，再次计算熵
        pk = xp.asarray([[0.0, 0.2], [0.6, 0.3], [0.3, 0.5]])
        # 使用断言检查熵计算结果是否接近期望值数组
        xp_assert_close(stats.entropy(pk, qk),
                        xp.asarray([0.17403988, 0.18609809]))

    @array_api_compatible
    def test_entropy_base_2d_nondefault_axis(self, xp):
        # 创建二维数组 pk，沿非默认轴计算熵
        pk = xp.asarray([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
        # 使用断言检查沿指定轴计算得到的熵是否接近期望值数组
        xp_assert_close(stats.entropy(pk, axis=1),
                        xp.asarray([0.63651417, 0.63651417, 0.66156324]))

    @array_api_compatible
    def test_entropy_2d_nondefault_axis(self, xp):
        # 创建二维数组 pk 和 qk，沿非默认轴计算它们之间的熵
        pk = xp.asarray([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
        qk = xp.asarray([[0.2, 0.1], [0.3, 0.6], [0.5, 0.3]])
        # 使用断言检查沿指定轴计算得到的熵是否接近期望值数组
        xp_assert_close(stats.entropy(pk, qk, axis=1),
                        xp.asarray([0.23104906, 0.23104906, 0.12770641]))

    @array_api_compatible
    # 这里应该继续完整地添加后续的测试方法，但由于截断，无法提供完整代码
    # 测试 stats.entropy 函数对于不兼容形状的输入是否会引发 ValueError 异常
    def test_entropy_raises_value_error(self, xp):
        # 创建两个 NumPy 数组，分别表示概率分布 pk 和 qk
        pk = xp.asarray([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
        qk = xp.asarray([[0.1, 0.2], [0.6, 0.3]])
        # 定义错误信息字符串
        message = "Array shapes are incompatible for broadcasting."
        # 使用 pytest 检查 stats.entropy 函数是否会引发 ValueError 异常，并验证错误信息
        with pytest.raises(ValueError, match=message):
            stats.entropy(pk, qk)

    # 使用装饰器指示此测试兼容特定的数组 API
    @array_api_compatible
    # 测试 stats.entropy 函数在指定 axis=0 时与默认参数计算结果是否相等
    def test_base_entropy_with_axis_0_is_equal_to_default(self, xp):
        # 创建一个 NumPy 数组表示概率分布 pk
        pk = xp.asarray([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
        # 断言 stats.entropy 函数在 axis=0 和默认参数下计算结果是否相近
        xp_assert_close(stats.entropy(pk, axis=0),
                        stats.entropy(pk))

    # 使用装饰器指示此测试兼容特定的数组 API
    @array_api_compatible
    # 测试 stats.entropy 函数在指定 axis=0 时与指定的概率分布 qk 计算结果是否相等
    def test_entropy_with_axis_0_is_equal_to_default(self, xp):
        # 创建两个 NumPy 数组，分别表示概率分布 pk 和 qk
        pk = xp.asarray([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
        qk = xp.asarray([[0.2, 0.1], [0.3, 0.6], [0.5, 0.3]])
        # 断言 stats.entropy 函数在 axis=0 和指定的概率分布下计算结果是否相近
        xp_assert_close(stats.entropy(pk, qk, axis=0),
                        stats.entropy(pk, qk))

    # 使用装饰器指示此测试兼容特定的数组 API
    @array_api_compatible
    # 测试 stats.entropy 函数在转置 pk 后与指定 axis=1 计算结果是否相等
    def test_base_entropy_transposed(self, xp):
        # 创建一个 NumPy 数组表示概率分布 pk
        pk = xp.asarray([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
        # 断言 stats.entropy 函数在转置 pk 后与 axis=1 计算结果是否相近
        xp_assert_close(stats.entropy(pk.T),
                        stats.entropy(pk, axis=1))

    # 使用装饰器指示此测试兼容特定的数组 API
    @array_api_compatible
    # 测试 stats.entropy 函数在转置 pk 和 qk 后与指定 axis=1 计算结果是否相等
    def test_entropy_transposed(self, xp):
        # 创建两个 NumPy 数组，分别表示概率分布 pk 和 qk
        pk = xp.asarray([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
        qk = xp.asarray([[0.2, 0.1], [0.3, 0.6], [0.5, 0.3]])
        # 断言 stats.entropy 函数在转置 pk 和 qk 后与 axis=1 计算结果是否相近
        xp_assert_close(stats.entropy(pk.T, qk.T),
                        stats.entropy(pk, qk, axis=1))

    # 使用装饰器指示此测试兼容特定的数组 API
    @array_api_compatible
    # 测试 stats.entropy 函数对于具有不同形状的输入是否正确广播
    def test_entropy_broadcasting(self, xp):
        # 创建一个随机数生成器 rng，并生成 NumPy 数组 x 和 y
        rng = np.random.default_rng(74187315492831452)
        x = xp.asarray(rng.random(3))
        y = xp.asarray(rng.random((2, 1)))
        # 使用 axis=-1 计算 stats.entropy 函数的结果，并验证广播结果
        res = stats.entropy(x, y, axis=-1)
        xp_assert_equal(res[0], stats.entropy(x, y[0, ...]))
        xp_assert_equal(res[1], stats.entropy(x, y[1, ...]))

    # 使用装饰器指示此测试兼容特定的数组 API
    @array_api_compatible
    # 测试 stats.entropy 函数在形状不匹配的情况下是否会引发 ValueError 异常
    def test_entropy_shape_mismatch(self, xp):
        # 创建两个 NumPy 数组，分别具有不同的形状
        x = xp.ones((10, 1, 12))
        y = xp.ones((11, 2))
        # 定义错误信息字符串
        message = "Array shapes are incompatible for broadcasting."
        # 使用 pytest 检查 stats.entropy 函数是否会引发 ValueError 异常，并验证错误信息
        with pytest.raises(ValueError, match=message):
            stats.entropy(x, y)

    # 使用装饰器指示此测试兼容特定的数组 API
    @array_api_compatible
    # 测试 stats.entropy 函数的输入参数是否被正确验证
    def test_input_validation(self, xp):
        # 创建一个 NumPy 数组 x，其中 base 参数为负数
        x = xp.ones(10)
        # 定义错误信息字符串
        message = "`base` must be a positive number."
        # 使用 pytest 检查 stats.entropy 函数是否会引发 ValueError 异常，并验证错误信息
        with pytest.raises(ValueError, match=message):
            stats.entropy(x, base=-2)
class TestDifferentialEntropy:
    """
    Vasicek results are compared with the R package vsgoftest.

    # library(vsgoftest)
    #
    # samp <- c(<values>)
    # entropy.estimate(x = samp, window = <window_length>)

    """

    def test_differential_entropy_vasicek(self):
        # 使用随机种子为0生成100个标准正态分布的随机数
        random_state = np.random.RandomState(0)
        values = random_state.standard_normal(100)

        # 计算使用 Vasicek 方法的差分熵
        entropy = stats.differential_entropy(values, method='vasicek')
        assert_allclose(entropy, 1.342551, rtol=1e-6)

        # 使用 Vasicek 方法和窗口长度为1计算差分熵
        entropy = stats.differential_entropy(values, window_length=1,
                                             method='vasicek')
        assert_allclose(entropy, 1.122044, rtol=1e-6)

        # 使用 Vasicek 方法和窗口长度为8计算差分熵
        entropy = stats.differential_entropy(values, window_length=8,
                                             method='vasicek')
        assert_allclose(entropy, 1.349401, rtol=1e-6)

    def test_differential_entropy_vasicek_2d_nondefault_axis(self):
        # 使用随机种子为0生成一个3行100列的标准正态分布随机数组
        random_state = np.random.RandomState(0)
        values = random_state.standard_normal((3, 100))

        # 沿着第二个轴（每行）使用 Vasicek 方法计算差分熵
        entropy = stats.differential_entropy(values, axis=1, method='vasicek')
        assert_allclose(
            entropy,
            [1.342551, 1.341826, 1.293775],
            rtol=1e-6,
        )

        # 沿着第二个轴（每行）使用 Vasicek 方法和窗口长度为1计算差分熵
        entropy = stats.differential_entropy(values, axis=1, window_length=1,
                                             method='vasicek')
        assert_allclose(
            entropy,
            [1.122044, 1.102944, 1.129616],
            rtol=1e-6,
        )

        # 沿着第二个轴（每行）使用 Vasicek 方法和窗口长度为8计算差分熵
        entropy = stats.differential_entropy(values, axis=1, window_length=8,
                                             method='vasicek')
        assert_allclose(
            entropy,
            [1.349401, 1.338514, 1.292332],
            rtol=1e-6,
        )

    def test_differential_entropy_raises_value_error(self):
        # 使用随机种子为0生成一个3行100列的标准正态分布随机数组
        random_state = np.random.RandomState(0)
        values = random_state.standard_normal((3, 100))

        # 定义错误消息模板
        error_str = (
            r"Window length \({window_length}\) must be positive and less "
            r"than half the sample size \({sample_size}\)."
        )

        sample_size = values.shape[1]

        # 对于不同的窗口长度，测试是否抛出 ValueError 异常
        for window_length in {-1, 0, sample_size//2, sample_size}:
            formatted_error_str = error_str.format(
                window_length=window_length,
                sample_size=sample_size,
            )

            # 使用 assert_raises 检查是否抛出指定异常和匹配的错误消息
            with assert_raises(ValueError, match=formatted_error_str):
                stats.differential_entropy(
                    values,
                    window_length=window_length,
                    axis=1,
                )

    def test_base_differential_entropy_with_axis_0_is_equal_to_default(self):
        # 使用随机种子为0生成一个100行3列的标准正态分布随机数组
        random_state = np.random.RandomState(0)
        values = random_state.standard_normal((100, 3))

        # 沿着第一个轴（每列）计算差分熵
        entropy = stats.differential_entropy(values, axis=0)
        # 计算默认情况下的差分熵（沿着第一个轴）
        default_entropy = stats.differential_entropy(values)
        # 断言两者的结果应该非常接近
        assert_allclose(entropy, default_entropy)
    # 定义一个单元测试方法，用于测试转置后的差分熵计算
    def test_base_differential_entropy_transposed(self):
        # 创建一个具有固定随机种子的随机数生成器对象
        random_state = np.random.RandomState(0)
        # 生成一个形状为 (3, 100) 的标准正态分布随机数数组
        values = random_state.standard_normal((3, 100))

        # 断言转置后的差分熵与按指定轴计算的差分熵结果相等
        assert_allclose(
            stats.differential_entropy(values.T).T,
            stats.differential_entropy(values, axis=1),
        )

    # 定义一个单元测试方法，用于测试输入验证的异常情况
    def test_input_validation(self):
        # 生成一个包含 10 个随机数的一维数组
        x = np.random.rand(10)

        # 预期的错误消息
        message = "`base` must be a positive number or `None`."
        # 使用 pytest 检查是否会抛出 ValueError 异常并匹配预期的错误消息
        with pytest.raises(ValueError, match=message):
            stats.differential_entropy(x, base=-2)

        # 再次设置预期的错误消息
        message = "`method` must be one of..."
        # 使用 pytest 检查是否会抛出 ValueError 异常并匹配预期的错误消息
        with pytest.raises(ValueError, match=message):
            stats.differential_entropy(x, method='ekki-ekki')

    # 使用 pytest 的参数化装饰器定义多组参数来测试一致性
    @pytest.mark.parametrize('method', ['vasicek', 'van es',
                                        'ebrahimi', 'correa'])
    def test_consistency(self, method):
        # 检验估计方法是否一致的测试
        # 如果方法是 'correa'，则设定样本数为 10000，否则设为 1000000
        n = 10000 if method == 'correa' else 1000000
        # 生成一个大小为 n 的正态分布随机数样本，使用指定的随机种子
        rvs = stats.norm.rvs(size=n, random_state=0)
        # 计算正态分布的熵的期望值
        expected = stats.norm.entropy()
        # 计算使用指定方法的差分熵估计值
        res = stats.differential_entropy(rvs, method=method)
        # 断言估计值与期望值在一定容忍范围内的接近程度
        assert_allclose(res, expected, rtol=0.005)

    # 使用字典定义多个方法的 RMSE 和标准差的参考值
    # values from differential_entropy reference [6], table 1, n=50, m=7
    norm_rmse_std_cases = {  # method: (RMSE, STD)
                           'vasicek': (0.198, 0.109),
                           'van es': (0.212, 0.110),
                           'correa': (0.135, 0.112),
                           'ebrahimi': (0.128, 0.109)
                           }

    # 使用 pytest 的参数化装饰器定义多组参数来测试 RMSE 和标准差
    @pytest.mark.parametrize('method, expected',
                             list(norm_rmse_std_cases.items()))
    def test_norm_rmse_std(self, method, expected):
        # 测试估计方法的 RMSE 和标准差是否与参考值匹配，同时也测试向量化
        reps, n, m = 10000, 50, 7
        rmse_expected, std_expected = expected
        # 生成大小为 (reps, n) 的正态分布随机数样本，使用指定的随机种子
        rvs = stats.norm.rvs(size=(reps, n), random_state=0)
        # 计算正态分布的熵的真实值
        true_entropy = stats.norm.entropy()
        # 计算使用指定方法和窗口长度的差分熵估计值
        res = stats.differential_entropy(rvs, window_length=m,
                                         method=method, axis=-1)
        # 断言估计值的均方根误差与预期的 RMSE 值在一定容忍范围内的接近程度
        assert_allclose(np.sqrt(np.mean((res - true_entropy)**2)),
                        rmse_expected, atol=0.005)
        # 断言估计值的标准差与预期的标准差在一定容忍范围内的接近程度
        assert_allclose(np.std(res), std_expected, atol=0.002)

    # 使用字典定义多个方法的 RMSE 和标准差的参考值
    # values from differential_entropy reference [6], table 2, n=50, m=7
    expon_rmse_std_cases = {  # method: (RMSE, STD)
                            'vasicek': (0.194, 0.148),
                            'van es': (0.179, 0.149),
                            'correa': (0.155, 0.152),
                            'ebrahimi': (0.151, 0.148)
                            }

    # 使用 pytest 的参数化装饰器定义多组参数来测试 RMSE 和标准差
    @pytest.mark.parametrize('method, expected',
                             list(expon_rmse_std_cases.items()))
    # 定义一个测试方法，用于验证估计器的均方根误差（RMSE）和标准偏差是否与参考文献 [6] 中给出的期望值匹配。
    # 顺便测试向量化功能。
    def test_expon_rmse_std(self, method, expected):
        # 设定重复次数、样本大小和特征数
        reps, n, m = 10000, 50, 7
        # 从期望值元组中解包得到 RMSE 和标准偏差的期望值
        rmse_expected, std_expected = expected
        # 生成服从指数分布的随机变量矩阵，使用给定的随机种子
        rvs = stats.expon.rvs(size=(reps, n), random_state=0)
        # 计算真实的熵值（指数分布的熵）
        true_entropy = stats.expon.entropy()
        # 计算不同熵估计方法的熵值，返回一个数组
        res = stats.differential_entropy(rvs, window_length=m,
                                         method=method, axis=-1)
        # 断言数组 res 与真实熵值的均方根误差是否在给定的误差范围内
        assert_allclose(np.sqrt(np.mean((res - true_entropy)**2)),
                        rmse_expected, atol=0.005)
        # 断言数组 res 的标准偏差是否在给定的误差范围内
        assert_allclose(np.std(res), std_expected, atol=0.002)

    # 使用 pytest 的参数化装饰器，定义自动化测试方法，测试不同的参数组合
    @pytest.mark.parametrize('n, method', [(8, 'van es'),
                                           (12, 'ebrahimi'),
                                           (1001, 'vasicek')])
    def test_method_auto(self, n, method):
        # 生成服从正态分布的随机变量数组，使用给定的随机种子
        rvs = stats.norm.rvs(size=(n,), random_state=0)
        # 分别计算使用默认方法和指定方法的差分熵
        res1 = stats.differential_entropy(rvs)
        res2 = stats.differential_entropy(rvs, method=method)
        # 断言两种方法计算的差分熵是否相等
        assert res1 == res2
```