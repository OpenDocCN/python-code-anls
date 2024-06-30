# `D:\src\scipysrc\scipy\scipy\stats\tests\test_hypotests.py`

```
from itertools import product  # 导入itertools模块中的product函数，用于生成迭代器的笛卡尔积

import numpy as np  # 导入NumPy库并简写为np，用于科学计算
import random  # 导入random模块，用于生成随机数
import functools  # 导入functools模块，用于高阶函数操作
import pytest  # 导入pytest库，用于编写和运行测试用例
from numpy.testing import (assert_, assert_equal, assert_allclose,
                           assert_almost_equal)  # 导入NumPy测试工具中的断言方法，用于测试数组内容是否相等
from pytest import raises as assert_raises  # 导入pytest库中的raises函数并简写为assert_raises，用于检查是否抛出异常

import scipy.stats as stats  # 导入SciPy库中的stats模块，用于统计函数
from scipy.stats import distributions  # 导入SciPy库中的distributions模块，用于概率分布
from scipy.stats._hypotests import (epps_singleton_2samp, cramervonmises,
                                    _cdf_cvm, cramervonmises_2samp,
                                    _pval_cvm_2samp_exact, barnard_exact,
                                    boschloo_exact)  # 导入SciPy库中的假设检验和统计测试相关的函数

from scipy.stats._mannwhitneyu import mannwhitneyu, _mwu_state  # 导入SciPy库中的Mann-Whitney U检验相关函数
from .common_tests import check_named_results  # 从当前目录下的common_tests模块中导入check_named_results函数
from scipy._lib._testutils import _TestPythranFunc  # 导入SciPy库中的测试工具函数
from scipy.stats._axis_nan_policy import SmallSampleWarning, too_small_1d_not_omit  # 导入SciPy库中的警告类和警告信息

class TestEppsSingleton:
    def test_statistic_1(self):
        # Goerg & Kaiser中的第一个示例，也出现在Epps & Singleton的原始论文中。
        # 注意：具体数值可能会有所不同，因为四分位数范围的计算方式可能不同。
        x = np.array([-0.35, 2.55, 1.73, 0.73, 0.35,
                      2.69, 0.46, -0.94, -0.37, 12.07])  # 定义NumPy数组x，包含浮点数值
        y = np.array([-1.15, -0.15, 2.48, 3.25, 3.71,
                      4.29, 5.00, 7.74, 8.38, 8.60])  # 定义NumPy数组y，包含浮点数值
        w, p = epps_singleton_2samp(x, y)  # 调用epps_singleton_2samp函数计算统计量w和p值
        assert_almost_equal(w, 15.14, decimal=1)  # 使用assert_almost_equal断言检查w的值是否接近15.14，精确到小数点后一位
        assert_almost_equal(p, 0.00442, decimal=3)  # 使用assert_almost_equal断言检查p的值是否接近0.00442，精确到小数点后三位

    def test_statistic_2(self):
        # Goerg & Kaiser中的第二个示例，同样不完全匹配。
        x = np.array((0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 5, 5, 5, 5, 6, 10,
                      10, 10, 10))  # 定义NumPy数组x，包含整数值
        y = np.array((10, 4, 0, 5, 10, 10, 0, 5, 6, 7, 10, 3, 1, 7, 0, 8, 1,
                      5, 8, 10))  # 定义NumPy数组y，包含整数值
        w, p = epps_singleton_2samp(x, y)  # 调用epps_singleton_2samp函数计算统计量w和p值
        assert_allclose(w, 8.900, atol=0.001)  # 使用assert_allclose断言检查w的值是否接近8.900，允许误差为0.001
        assert_almost_equal(p, 0.06364, decimal=3)  # 使用assert_almost_equal断言检查p的值是否接近0.06364，精确到小数点后三位

    def test_epps_singleton_array_like(self):
        np.random.seed(1234)  # 设置随机种子，以便结果可重现
        x, y = np.arange(30), np.arange(28)  # 定义NumPy数组x和y，分别包含0到29和0到27的整数

        w1, p1 = epps_singleton_2samp(list(x), list(y))  # 调用epps_singleton_2samp函数计算统计量w1和p1值
        w2, p2 = epps_singleton_2samp(tuple(x), tuple(y))  # 调用epps_singleton_2samp函数计算统计量w2和p2值
        w3, p3 = epps_singleton_2samp(x, y)  # 调用epps_singleton_2samp函数计算统计量w3和p3值

        assert_(w1 == w2 == w3)  # 使用assert_断言检查w1、w2和w3是否相等
        assert_(p1 == p2 == p3)  # 使用assert_断言检查p1、p2和p3是否相等

    def test_epps_singleton_size(self):
        # 如果样本包含少于5个元素，则发出警告
        x, y = (1, 2, 3, 4), np.arange(10)  # 定义NumPy数组x和y，分别包含整数1到4和0到9

        with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):  # 使用pytest.warns捕获SmallSampleWarning警告，并匹配消息too_small_1d_not_omit
            res = epps_singleton_2samp(x, y)  # 调用epps_singleton_2samp函数计算统计量和p值
            assert_equal(res.statistic, np.nan)  # 使用assert_equal断言检查统计量是否为NaN
            assert_equal(res.pvalue, np.nan)  # 使用assert_equal断言检查p值是否为NaN

    def test_epps_singleton_nonfinite(self):
        # 如果存在非有限值，则抛出错误
        x, y = (1, 2, 3, 4, 5, np.inf), np.arange(10)  # 定义NumPy数组x和y，包含整数1到5和无穷大，以及整数0到9

        assert_raises(ValueError, epps_singleton_2samp, x, y)  # 使用assert_raises断言检查调用epps_singleton_2samp函数时是否会引发ValueError异常
    # 定义一个测试方法，用于测试某个功能或函数
    def test_names(self):
        # 创建两个 NumPy 数组 x 和 y，分别包含 20 和 30 个连续整数
        x, y = np.arange(20), np.arange(30)
        # 调用 epps_singleton_2samp 函数对数组 x 和 y 进行统计分析，返回结果存储在 res 中
        res = epps_singleton_2samp(x, y)
        # 定义一个元组 attributes 包含 'statistic' 和 'pvalue'，用于检查返回结果的命名属性
        attributes = ('statistic', 'pvalue')
        # 调用 check_named_results 函数检查 res 是否包含指定的命名属性
        check_named_results(res, attributes)
class TestCvm:
    # 从Csorgo / Faraway的1996年论文中的表1中获取Cramér-von Mises统计量的累积分布函数（CDF）的预期值。

    def test_cdf_4(self):
        assert_allclose(
                _cdf_cvm([0.02983, 0.04111, 0.12331, 0.94251], 4),
                [0.01, 0.05, 0.5, 0.999],
                atol=1e-4)

    def test_cdf_10(self):
        assert_allclose(
                _cdf_cvm([0.02657, 0.03830, 0.12068, 0.56643], 10),
                [0.01, 0.05, 0.5, 0.975],
                atol=1e-4)

    def test_cdf_1000(self):
        assert_allclose(
                _cdf_cvm([0.02481, 0.03658, 0.11889, 1.16120], 1000),
                [0.01, 0.05, 0.5, 0.999],
                atol=1e-4)

    def test_cdf_inf(self):
        assert_allclose(
                _cdf_cvm([0.02480, 0.03656, 0.11888, 1.16204]),
                [0.01, 0.05, 0.5, 0.999],
                atol=1e-4)

    def test_cdf_support(self):
        # 累积分布函数（CDF）在区间 [1/(12*n), n/3] 上有支持
        assert_equal(_cdf_cvm([1/(12*533), 533/3], 533), [0, 1])
        assert_equal(_cdf_cvm([1/(12*(27 + 1)), (27 + 1)/3], 27), [0, 1])

    def test_cdf_large_n(self):
        # 测试大样本时，渐近CDF与实际CDF之间的接近程度
        assert_allclose(
                _cdf_cvm([0.02480, 0.03656, 0.11888, 1.16204, 100], 10000),
                _cdf_cvm([0.02480, 0.03656, 0.11888, 1.16204, 100]),
                atol=1e-4)

    def test_large_x(self):
        # 对于大的 x 值和 n 值，用于计算CDF的级数收敛较慢。
        # 这导致了R包goftest和作为scipy实现基础的MAPLE代码中的错误。
        # 注意：对于 x >= 1000/3 和 n = 1000，CDF = 1
        assert_(0.99999 < _cdf_cvm(333.3, 1000) < 1.0)
        assert_(0.99999 < _cdf_cvm(333.3) < 1.0)

    def test_low_p(self):
        # _cdf_cvm 可能返回大于 1 的值。在这种情况下，我们将 p 值设为零。
        n = 12
        res = cramervonmises(np.ones(n)*0.8, 'norm')
        assert_(_cdf_cvm(res.statistic, n) > 1.0)
        assert_equal(res.pvalue, 0)

    @pytest.mark.parametrize('x', [(), [1.5]])
    def test_invalid_input(self, x):
        with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
            res = cramervonmises(x, "norm")
            assert_equal(res.statistic, np.nan)
            assert_equal(res.pvalue, np.nan)
    def test_values_R(self):
        # compared against R package goftest, version 1.1.1
        # 使用 cramervonmises 函数计算输入数据的 Cramér-von Mises 统计量和 p 值，以正态分布为参照
        res = cramervonmises([-1.7, 2, 0, 1.3, 4, 0.1, 0.6], "norm")
        # 断言计算得到的统计量接近于 0.288156，允许误差为 1e-6
        assert_allclose(res.statistic, 0.288156, atol=1e-6)
        # 断言计算得到的 p 值接近于 0.1453465，允许误差为 1e-6
        assert_allclose(res.pvalue, 0.1453465, atol=1e-6)

        # 使用 cramervonmises 函数计算输入数据的 Cramér-von Mises 统计量和 p 值，以正态分布为参照，
        # 并指定正态分布的均值为 3，标准差为 1.5
        res = cramervonmises([-1.7, 2, 0, 1.3, 4, 0.1, 0.6], "norm", (3, 1.5))
        # 断言计算得到的统计量接近于 0.9426685，允许误差为 1e-6
        assert_allclose(res.statistic, 0.9426685, atol=1e-6)
        # 断言计算得到的 p 值接近于 0.002026417，允许误差为 1e-6
        assert_allclose(res.pvalue, 0.002026417, atol=1e-6)

        # 使用 cramervonmises 函数计算输入数据的 Cramér-von Mises 统计量和 p 值，以指数分布为参照
        res = cramervonmises([1, 2, 5, 1.4, 0.14, 11, 13, 0.9, 7.5], "expon")
        # 断言计算得到的统计量接近于 0.8421854，允许误差为 1e-6
        assert_allclose(res.statistic, 0.8421854, atol=1e-6)
        # 断言计算得到的 p 值接近于 0.004433406，允许误差为 1e-6
        assert_allclose(res.pvalue, 0.004433406, atol=1e-6)

    def test_callable_cdf(self):
        # 定义输入数据和参数
        x, args = np.arange(5), (1.4, 0.7)
        # 使用 cramervonmises 函数计算输入数据的 Cramér-von Mises 统计量和 p 值，
        # 第二个参数为概率分布的累积分布函数 (CDF)，这里使用指数分布的 CDF
        r1 = cramervonmises(x, distributions.expon.cdf)
        # 使用 cramervonmises 函数计算输入数据的 Cramér-von Mises 统计量和 p 值，以指数分布为参照
        r2 = cramervonmises(x, "expon")
        # 断言两次计算得到的统计量和 p 值相等
        assert_equal((r1.statistic, r1.pvalue), (r2.statistic, r2.pvalue))

        # 使用 cramervonmises 函数计算输入数据的 Cramér-von Mises 统计量和 p 值，
        # 第二个参数为概率分布的累积分布函数 (CDF)，这里使用 beta 分布的 CDF，并指定参数 args
        r1 = cramervonmises(x, distributions.beta.cdf, args)
        # 使用 cramervonmises 函数计算输入数据的 Cramér-von Mises 统计量和 p 值，以 beta 分布为参照，并指定参数 args
        r2 = cramervonmises(x, "beta", args)
        # 断言两次计算得到的统计量和 p 值相等
        assert_equal((r1.statistic, r1.pvalue), (r2.statistic, r2.pvalue))
# 定义一个测试类 TestMannWhitneyU，用于测试 Mann-Whitney U 检验的功能

# 所有的魔数均来自于 R 的 wilcox.test 函数，除非另有说明
# https://rdrr.io/r/stats/wilcox.test.html

class TestMannWhitneyU:

    # --- Test Input Validation ---

    # 使用 pytest 的参数化装饰器，为测试方法 test_empty 提供多组参数
    @pytest.mark.parametrize('kwargs_update', [{'x': []}, {'y': []},
                                               {'x': [], 'y': []}])
    def test_empty(self, kwargs_update):
        # 创建两个 NumPy 数组作为通用且有效的输入
        x = np.array([1, 2])
        y = np.array([3, 4])
        kwargs = dict(x=x, y=y)
        # 更新 kwargs 字典以包含参数化的更新
        kwargs.update(kwargs_update)
        # 使用 pytest 的 warn 断言检查是否会引发 SmallSampleWarning 警告，并匹配特定的字符串 "too_small_1d_not_omit"
        with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
            # 调用 mannwhitneyu 函数，并断言返回的 statistic 和 pvalue 均为 NaN
            res = mannwhitneyu(**kwargs)
            assert_equal(res.statistic, np.nan)
            assert_equal(res.pvalue, np.nan)

    # 测试输入验证功能
    def test_input_validation(self):
        # 创建两个 NumPy 数组作为通用且有效的输入
        x = np.array([1, 2])
        y = np.array([3, 4])
        # 使用 assert_raises 断言检查是否会引发 ValueError，并匹配特定的错误信息字符串
        with assert_raises(ValueError, match="`use_continuity` must be one"):
            mannwhitneyu(x, y, use_continuity='ekki')
        with assert_raises(ValueError, match="`alternative` must be one of"):
            mannwhitneyu(x, y, alternative='ekki')
        with assert_raises(ValueError, match="`axis` must be an integer"):
            mannwhitneyu(x, y, axis=1.5)
        with assert_raises(ValueError, match="`method` must be one of"):
            mannwhitneyu(x, y, method='ekki')
    def test_auto(self):
        # Test that default method ('auto') chooses intended method
        
        np.random.seed(1)
        n = 8  # threshold to switch from exact to asymptotic
        
        # both inputs are smaller than threshold; should use exact
        x = np.random.rand(n-1)
        y = np.random.rand(n-1)
        auto = mannwhitneyu(x, y)
        asymptotic = mannwhitneyu(x, y, method='asymptotic')
        exact = mannwhitneyu(x, y, method='exact')
        assert auto.pvalue == exact.pvalue
        assert auto.pvalue != asymptotic.pvalue
        
        # one input is smaller than threshold; should use exact
        x = np.random.rand(n-1)
        y = np.random.rand(n+1)
        auto = mannwhitneyu(x, y)
        asymptotic = mannwhitneyu(x, y, method='asymptotic')
        exact = mannwhitneyu(x, y, method='exact')
        assert auto.pvalue == exact.pvalue
        assert auto.pvalue != asymptotic.pvalue
        
        # other input is smaller than threshold; should use exact
        auto = mannwhitneyu(y, x)
        asymptotic = mannwhitneyu(x, y, method='asymptotic')
        exact = mannwhitneyu(x, y, method='exact')
        assert auto.pvalue == exact.pvalue
        assert auto.pvalue != asymptotic.pvalue
        
        # both inputs are larger than threshold; should use asymptotic
        x = np.random.rand(n+1)
        y = np.random.rand(n+1)
        auto = mannwhitneyu(x, y)
        asymptotic = mannwhitneyu(x, y, method='asymptotic')
        exact = mannwhitneyu(x, y, method='exact')
        assert auto.pvalue != exact.pvalue
        assert auto.pvalue == asymptotic.pvalue
        
        # both inputs are smaller than threshold, but there is a tie
        # should use asymptotic
        x = np.random.rand(n-1)
        y = np.random.rand(n-1)
        y[3] = x[3]
        auto = mannwhitneyu(x, y)
        asymptotic = mannwhitneyu(x, y, method='asymptotic')
        exact = mannwhitneyu(x, y, method='exact')
        assert auto.pvalue != exact.pvalue
        assert auto.pvalue == asymptotic.pvalue



    # --- Test Basic Functionality ---
    
    x = [210.052110, 110.190630, 307.918612]
    y = [436.08811482466416, 416.37397329768191, 179.96975939463582,
         197.8118754228619, 34.038757281225756, 138.54220550921517,
         128.7769351470246, 265.92721427951852, 275.6617533155341,
         592.34083395416258, 448.73177590617018, 300.61495185038905,
         187.97508449019588]

    # This test was written for mann_whitney_u in gh-4933.
    # Originally, the p-values for alternatives were swapped;
    # this has been corrected and the tests have been refactored for
    # compactness, but otherwise the tests are unchanged.
    # R code for comparison, e.g.:
    # options(digits = 16)
    # x = c(210.052110, 110.190630, 307.918612)
    # y = c(436.08811482466416, 416.37397329768191, 179.96975939463582,
    #       197.8118754228619, 34.038757281225756, 138.54220550921517,
    #       128.7769351470246, 265.92721427951852, 275.6617533155341,
    #       592.34083395416258, 448.73177590617018, 300.61495185038905,
    #       187.97508449019588)
    # 定义基本测试用例，每个测试用例包括一个字典和一个预期结果元组
    cases_basic = [[{"alternative": 'two-sided', "method": "asymptotic"},
                    (16, 0.6865041817876)],
                   [{"alternative": 'less', "method": "asymptotic"},
                    (16, 0.3432520908938)],
                   [{"alternative": 'greater', "method": "asymptotic"},
                    (16, 0.7047591913255)],
                   [{"alternative": 'two-sided', "method": "exact"},
                    (16, 0.7035714285714)],
                   [{"alternative": 'less', "method": "exact"},
                    (16, 0.3517857142857)],
                   [{"alternative": 'greater', "method": "exact"},
                    (16, 0.6946428571429)]]

    # 使用 pytest 的参数化装饰器标记，为每组参数化的测试数据执行测试函数
    @pytest.mark.parametrize(("kwds", "expected"), cases_basic)
    def test_basic(self, kwds, expected):
        # 调用 mannwhitneyu 函数，传入 self.x 和 self.y 作为参数，以及 kwds 字典作为关键字参数
        res = mannwhitneyu(self.x, self.y, **kwds)
        # 断言测试结果与期望结果在数值上的近似程度
        assert_allclose(res, expected)

    # 定义包含连续性校正参数的测试用例集合
    cases_continuity = [[{"alternative": 'two-sided', "use_continuity": True},
                         (23, 0.6865041817876)],
                        [{"alternative": 'less', "use_continuity": True},
                         (23, 0.7047591913255)],
                        [{"alternative": 'greater', "use_continuity": True},
                         (23, 0.3432520908938)],
                        [{"alternative": 'two-sided', "use_continuity": False},
                         (23, 0.6377328900502)],
                        [{"alternative": 'less', "use_continuity": False},
                         (23, 0.6811335549749)],
                        [{"alternative": 'greater', "use_continuity": False},
                         (23, 0.3188664450251)]]

    # 使用 pytest 的参数化装饰器标记，为每组连续性校正参数化的测试数据执行测试函数
    @pytest.mark.parametrize(("kwds", "expected"), cases_continuity)
    def test_continuity(self, kwds, expected):
        # 在交换 self.x 和 self.y 的情况下调用 mannwhitneyu 函数，传入指定的关键字参数
        # method='asymptotic' 时 exact=FALSE，use_continuity=False 时 correct=FALSE
        res = mannwhitneyu(self.y, self.x, method='asymptotic', **kwds)
        # 断言测试结果与期望结果在数值上的近似程度
        assert_allclose(res, expected)
    # 定义一个测试方法，用于测试 tie correction 是否与 R 的 wilcox.test 一致
    def test_tie_correct(self):
        # 创建数据集 x 和 y，并引入一些小的变化
        x = [1, 2, 3, 4]  # 定义一个列表 x
        y0 = np.array([1, 2, 3, 4, 5])  # 创建一个 NumPy 数组 y0
        dy = np.array([0, 1, 0, 1, 0])*0.01  # 创建一个 NumPy 数组 dy
        dy2 = np.array([0, 0, 1, 0, 0])*0.01  # 创建一个 NumPy 数组 dy2
        # 通过对 y0 和一些偏差进行操作，生成多个 y 的变体
        y = [y0-0.01, y0-dy, y0-dy2, y0, y0+dy2, y0+dy, y0+0.01]
        # 使用 mannwhitneyu 函数计算统计量和 p 值，指定轴向和方法参数
        res = mannwhitneyu(x, y, axis=-1, method="asymptotic")
        # 预期的 U 统计量的值
        U_expected = [10, 9, 8.5, 8, 7.5, 7, 6]
        # 预期的 p 值列表
        p_expected = [1, 0.9017048037317, 0.804080657472, 0.7086240584439,
                      0.6197963884941, 0.5368784563079, 0.3912672792826]
        # 使用 assert_equal 函数断言实际 U 统计量与预期值相等
        assert_equal(res.statistic, U_expected)
        # 使用 assert_allclose 函数断言实际 p 值列表与预期值列表相近
        assert_allclose(res.pvalue, p_expected)

    # --- Test Exact Distribution of U ---

    # 下面是 U 的精确分布的累积分布函数的标定值，引自参考文献 [1] 的第 52 页（Mann-Whitney 原始文献）
    pn3 = {1: [0.25, 0.5, 0.75], 2: [0.1, 0.2, 0.4, 0.6],
           3: [0.05, .1, 0.2, 0.35, 0.5, 0.65]}
    pn4 = {1: [0.2, 0.4, 0.6], 2: [0.067, 0.133, 0.267, 0.4, 0.6],
           3: [0.028, 0.057, 0.114, 0.2, .314, 0.429, 0.571],
           4: [0.014, 0.029, 0.057, 0.1, 0.171, 0.243, 0.343, 0.443, 0.557]}
    pm5 = {1: [0.167, 0.333, 0.5, 0.667],
           2: [0.047, 0.095, 0.19, 0.286, 0.429, 0.571],
           3: [0.018, 0.036, 0.071, 0.125, 0.196, 0.286, 0.393, 0.5, 0.607],
           4: [0.008, 0.016, 0.032, 0.056, 0.095, 0.143,
               0.206, 0.278, 0.365, 0.452, 0.548],
           5: [0.004, 0.008, 0.016, 0.028, 0.048, 0.075, 0.111,
               0.155, 0.21, 0.274, 0.345, .421, 0.5, 0.579]}
    pm6 = {1: [0.143, 0.286, 0.428, 0.571],
           2: [0.036, 0.071, 0.143, 0.214, 0.321, 0.429, 0.571],
           3: [0.012, 0.024, 0.048, 0.083, 0.131,
               0.19, 0.274, 0.357, 0.452, 0.548],
           4: [0.005, 0.01, 0.019, 0.033, 0.057, 0.086, 0.129,
               0.176, 0.238, 0.305, 0.381, 0.457, 0.543]}  # 上一个列表的最后一个元素 0.543，已从 0.545 修改过来；我假设这是个错字
           # 5: [0.002, 0.004, 0.009, 0.015, 0.026, 0.041, 0.063, 0.089,
           #     0.123, 0.165, 0.214, 0.268, 0.331, 0.396, 0.465, 0.535],
           # 6: [0.001, 0.002, 0.004, 0.008, 0.013, 0.021, 0.032, 0.047,
           #     0.066, 0.09, 0.12, 0.155, 0.197, 0.242, 0.294, 0.350,
           #     0.409, 0.469, 0.531]}
    # 定义精确分布测试方法
    def test_exact_distribution(self):
        # 创建包含各个表格的字典
        p_tables = {3: self.pn3, 4: self.pn4, 5: self.pm5, 6: self.pm6}
        # 遍历字典中的每个表格
        for n, table in p_tables.items():
            # 遍历每个表格中的每个项目
            for m, p in table.items():
                # 检查 p 值与表格中的值是否接近
                u = np.arange(0, len(p))
                # 设置形状参数并计算累积分布函数 (CDF)
                _mwu_state.set_shapes(m, n)
                assert_allclose(_mwu_state.cdf(k=u), p, atol=1e-3)

                # 检查恒等式 CDF + SF - PMF = 1
                # （在此实现中，SF(U) 包括 PMF(U)）
                u2 = np.arange(0, m*n+1)
                assert_allclose(_mwu_state.cdf(k=u2)
                                + _mwu_state.sf(k=u2)
                                - _mwu_state.pmf(k=u2), 1)

                # 检查关于 U 平均值的对称性，即 pmf(U) = pmf(m*n-U)
                pmf = _mwu_state.pmf(k=u2)
                assert_allclose(pmf, pmf[::-1])

                # 检查关于 m, n 交换的对称性
                _mwu_state.set_shapes(n, m)
                pmf2 = _mwu_state.pmf(k=u2)
                assert_allclose(pmf, pmf2)

    # 测试渐近行为
    def test_asymptotic_behavior(self):
        np.random.seed(0)

        # 对于小样本，渐近检验不太准确
        x = np.random.rand(5)
        y = np.random.rand(5)
        res1 = mannwhitneyu(x, y, method="exact")
        res2 = mannwhitneyu(x, y, method="asymptotic")
        assert res1.statistic == res2.statistic
        assert np.abs(res1.pvalue - res2.pvalue) > 1e-2

        # 对于大样本，它们基本一致
        x = np.random.rand(40)
        y = np.random.rand(40)
        res1 = mannwhitneyu(x, y, method="exact")
        res2 = mannwhitneyu(x, y, method="asymptotic")
        assert res1.statistic == res2.statistic
        assert np.abs(res1.pvalue - res2.pvalue) < 1e-3

    # --- 测试边缘情况 ---

    # 测试精确方法下 U 等于均值的情况
    def test_exact_U_equals_mean(self):
        # 测试 U == m*n/2 的情况，使用精确方法
        # 如果没有特殊处理，双边 p 值 > 1，因为两个单边 p 值 > 0.5
        res_l = mannwhitneyu([1, 2, 3], [1.5, 2.5], alternative="less",
                             method="exact")
        res_g = mannwhitneyu([1, 2, 3], [1.5, 2.5], alternative="greater",
                             method="exact")
        assert_equal(res_l.pvalue, res_g.pvalue)
        assert res_l.pvalue > 0.5

        # 对于双边情况，期望结果是 (3, 1)
        res = mannwhitneyu([1, 2, 3], [1.5, 2.5], alternative="two-sided",
                           method="exact")
        assert_equal(res, (3, 1))
        # U == m*n/2 的情况在渐近测试中被 test_gh_2118 测试
        # 渐近测试复杂的原因与连续性修正有关。
    cases_scalar = [[{"alternative": 'two-sided', "method": "asymptotic"},
                     (0, 1)],
                    [{"alternative": 'less', "method": "asymptotic"},
                     (0, 0.5)],
                    [{"alternative": 'greater', "method": "asymptotic"},
                     (0, 0.977249868052)],
                    [{"alternative": 'two-sided', "method": "exact"}, (0, 1)],
                    [{"alternative": 'less', "method": "exact"}, (0, 0.5)],
                    [{"alternative": 'greater', "method": "exact"}, (0, 1)]]


# 定义了多个测试用例，每个测试用例包含参数字典和期望的测试结果
cases_scalar = [
    [{"alternative": 'two-sided', "method": "asymptotic"}, (0, 1)],
    [{"alternative": 'less', "method": "asymptotic"}, (0, 0.5)],
    [{"alternative": 'greater', "method": "asymptotic"}, (0, 0.977249868052)],
    [{"alternative": 'two-sided', "method": "exact"}, (0, 1)],
    [{"alternative": 'less', "method": "exact"}, (0, 0.5)],
    [{"alternative": 'greater', "method": "exact"}, (0, 1)]
]

@pytest.mark.parametrize(("kwds", "result"), cases_scalar)
def test_scalar_data(self, kwds, result):
    # 对标量数据进行测试，验证 mannwhitneyu 函数的输出是否符合预期结果
    assert_allclose(mannwhitneyu(1, 2, **kwds), result)

def test_equal_scalar_data(self):
    # 当两个标量相等时，根据不同的方法进行测试
    # 在渐近方法中，当两个标量相等时，使用 0.5/1 进行近似。对于 'less' 和 'greater' 替代假设，R 给出 pvalue=1.0，但对于 'two-sided' 给出 NA。尽管如此，这里不需要特殊处理以匹配该行为。
    assert_equal(mannwhitneyu(1, 1, method="exact"), (0.5, 1))
    assert_equal(mannwhitneyu(1, 1, method="asymptotic"), (0.5, 1))

    # 如果不进行连续性修正，结果会变为 0/0，这在数学上是未定义的
    assert_equal(mannwhitneyu(1, 1, method="asymptotic",
                              use_continuity=False), (0.5, np.nan))

# --- Test Enhancements / Bug Reports ---

@pytest.mark.parametrize("method", ["asymptotic", "exact"])
    def test_gh_12837_11113(self, method):
        # 测试广播可行性的行为是否恰当：
        # 输出形状是否正确，所有值是否与逐一测试样本时相同。
        # 测试 gh-12837 和 gh-11113（对 n-d 输入的请求）是否解决
        np.random.seed(0)

        # 除了 axis = -3 外，数组是可广播的
        axis = -3
        m, n = 7, 10  # 样本大小
        x = np.random.rand(m, 3, 8)
        y = np.random.rand(6, n, 1, 8) + 0.1
        res = mannwhitneyu(x, y, method=method, axis=axis)

        shape = (6, 3, 8)  # 给定输入的适当输出形状
        assert res.pvalue.shape == shape
        assert res.statistic.shape == shape

        # 为简单起见，将测试的轴移动到末尾
        x, y = np.moveaxis(x, axis, -1), np.moveaxis(y, axis, -1)

        x = x[None, ...]  # 给 x 添加一个零维度
        assert x.ndim == y.ndim

        x = np.broadcast_to(x, shape + (m,))
        y = np.broadcast_to(y, shape + (n,))
        assert x.shape[:-1] == shape
        assert y.shape[:-1] == shape

        # 循环遍历样本对
        statistics = np.zeros(shape)
        pvalues = np.zeros(shape)
        for indices in product(*[range(i) for i in shape]):
            xi = x[indices]
            yi = y[indices]
            temp = mannwhitneyu(xi, yi, method=method)
            statistics[indices] = temp.statistic
            pvalues[indices] = temp.pvalue

        np.testing.assert_equal(res.pvalue, pvalues)
        np.testing.assert_equal(res.statistic, statistics)

    def test_gh_11355(self):
        # 测试 NaN/Inf 在输入中的正确行为
        x = [1, 2, 3, 4]
        y = [3, 6, 7, 8, 9, 3, 2, 1, 4, 4, 5]
        res1 = mannwhitneyu(x, y)

        # Inf 不是问题。这是一个秩测试，它是最大的值
        y[4] = np.inf
        res2 = mannwhitneyu(x, y)

        assert_equal(res1.statistic, res2.statistic)
        assert_equal(res1.pvalue, res2.pvalue)

        # 默认情况下，NaN 应该传播
        y[4] = np.nan
        res3 = mannwhitneyu(x, y)
        assert_equal(res3.statistic, np.nan)
        assert_equal(res3.pvalue, np.nan)

    cases_11355 = [([1, 2, 3, 4],
                    [3, 6, 7, 8, np.inf, 3, 2, 1, 4, 4, 5],
                    10, 0.1297704873477),
                   ([1, 2, 3, 4],
                    [3, 6, 7, 8, np.inf, np.inf, 2, 1, 4, 4, 5],
                    8.5, 0.08735617507695),
                   ([1, 2, np.inf, 4],
                    [3, 6, 7, 8, np.inf, 3, 2, 1, 4, 4, 5],
                    17.5, 0.5988856695752),
                   ([1, 2, np.inf, 4],
                    [3, 6, 7, 8, np.inf, np.inf, 2, 1, 4, 4, 5],
                    16, 0.4687165824462),
                   ([1, np.inf, np.inf, 4],
                    [3, 6, 7, 8, np.inf, np.inf, 2, 1, 4, 4, 5],
                    24.5, 0.7912517950119)]
    @pytest.mark.parametrize(("x", "y", "statistic", "pvalue"), cases_11355)
    def test_gh_11355b(self, x, y, statistic, pvalue):
        # 使用 pytest 的参数化功能，对多组测试数据进行测试
        res = mannwhitneyu(x, y, method='asymptotic')
        # 断言得到的统计量与预期统计量非常接近，允许的误差为1e-12
        assert_allclose(res.statistic, statistic, atol=1e-12)
        # 断言得到的 p 值与预期 p 值非常接近，允许的误差为1e-12
        assert_allclose(res.pvalue, pvalue, atol=1e-12)

    cases_9184 = [[True, "less", "asymptotic", 0.900775348204],
                  [True, "greater", "asymptotic", 0.1223118025635],
                  [True, "two-sided", "asymptotic", 0.244623605127],
                  [False, "less", "asymptotic", 0.8896643190401],
                  [False, "greater", "asymptotic", 0.1103356809599],
                  [False, "two-sided", "asymptotic", 0.2206713619198],
                  [True, "less", "exact", 0.8967698967699],
                  [True, "greater", "exact", 0.1272061272061],
                  [True, "two-sided", "exact", 0.2544122544123]]

    @pytest.mark.parametrize(("use_continuity", "alternative",
                              "method", "pvalue_exp"), cases_9184)
    def test_gh_9184(self, use_continuity, alternative, method, pvalue_exp):
        # 使用 pytest 的参数化功能，对多组测试数据进行测试
        # 对应于 GitHub issue #9184 的测试，这可能只是文档上的问题。请查看
        # 文档以确认 mannwhitneyu 函数正确注明统计量是第一个样本（x）的。
        # 无论如何，检查提供的案例是否与 R 的输出一致。
        # R 代码：
        # options(digits=16)
        # x <- c(0.80, 0.83, 1.89, 1.04, 1.45, 1.38, 1.91, 1.64, 0.73, 1.46)
        # y <- c(1.15, 0.88, 0.90, 0.74, 1.21)
        # wilcox.test(x, y, alternative = "less", exact = FALSE)
        # wilcox.test(x, y, alternative = "greater", exact = FALSE)
        # wilcox.test(x, y, alternative = "two.sided", exact = FALSE)
        # wilcox.test(x, y, alternative = "less", exact = FALSE,
        #             correct=FALSE)
        # wilcox.test(x, y, alternative = "greater", exact = FALSE,
        #             correct=FALSE)
        # wilcox.test(x, y, alternative = "two.sided", exact = FALSE,
        #             correct=FALSE)
        # wilcox.test(x, y, alternative = "less", exact = TRUE)
        # wilcox.test(x, y, alternative = "greater", exact = TRUE)
        # wilcox.test(x, y, alternative = "two.sided", exact = TRUE)
        statistic_exp = 35
        x = (0.80, 0.83, 1.89, 1.04, 1.45, 1.38, 1.91, 1.64, 0.73, 1.46)
        y = (1.15, 0.88, 0.90, 0.74, 1.21)
        # 运行 mannwhitneyu 函数，传入参数并获取结果
        res = mannwhitneyu(x, y, use_continuity=use_continuity,
                           alternative=alternative, method=method)
        # 断言得到的统计量与预期统计量相等
        assert_equal(res.statistic, statistic_exp)
        # 断言得到的 p 值与预期 p 值非常接近，使用默认的误差容限
        assert_allclose(res.pvalue, pvalue_exp)
    def test_gh_4067(self):
        # 测试当所有输入都是 NaN 时的正确行为，默认情况下是传播 NaN
        a = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        b = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        res = mannwhitneyu(a, b)
        assert_equal(res.statistic, np.nan)
        assert_equal(res.pvalue, np.nan)

    # 所有案例都与 R 中的 wilcox.test 进行了对比，例如：
    # options(digits=16)
    # x = c(1, 2, 3)
    # y = c(1.5, 2.5)
    # wilcox.test(x, y, exact=FALSE, alternative='less')

    cases_2118 = [[[1, 2, 3], [1.5, 2.5], "greater", (3, 0.6135850036578)],
                  [[1, 2, 3], [1.5, 2.5], "less", (3, 0.6135850036578)],
                  [[1, 2, 3], [1.5, 2.5], "two-sided", (3, 1.0)],
                  [[1, 2, 3], [2], "greater", (1.5, 0.681324055883)],
                  [[1, 2, 3], [2], "less", (1.5, 0.681324055883)],
                  [[1, 2, 3], [2], "two-sided", (1.5, 1)],
                  [[1, 2], [1, 2], "greater", (2, 0.667497228949)],
                  [[1, 2], [1, 2], "less", (2, 0.667497228949)],
                  [[1, 2], [1, 2], "two-sided", (2, 1)]]

    @pytest.mark.parametrize(["x", "y", "alternative", "expected"], cases_2118)
    def test_gh_2118(self, x, y, alternative, expected):
        # 测试在方法为渐近法时，当 U == m*n/2 时的情况
        # 应用连续性修正可能导致 p 值大于 1
        res = mannwhitneyu(x, y, use_continuity=True, alternative=alternative,
                           method="asymptotic")
        assert_allclose(res, expected, rtol=1e-12)

    def test_gh19692_smaller_table(self):
        # 在 gh-19692 中，我们注意到计算 p 值时使用的缓存形状取决于输入的顺序，
        # 因为样本大小 n1 和 n2 发生了变化。这表明存在不必要的缓存增长和冗余计算。
        # 检查这一问题是否得到解决。
        rng = np.random.default_rng(7600451795963068007)
        m, n = 5, 11
        x = rng.random(size=m)
        y = rng.random(size=n)
        _mwu_state.reset()  # 重置缓存
        res = stats.mannwhitneyu(x, y, method='exact')
        shape = _mwu_state.configurations.shape
        assert shape[-1] == min(res.statistic, m*n - res.statistic) + 1
        stats.mannwhitneyu(y, x, method='exact')
        assert shape == _mwu_state.configurations.shape  # 当大小反转时保持相同

        # 此外，我们没有充分利用零假设分布的对称性。
        # 确保在 `k > m*n/2` 时不显式评估零假设分布。
        _mwu_state.reset()  # 重置缓存
        stats.mannwhitneyu(x, 0*y, method='exact', alternative='greater')
        shape = _mwu_state.configurations.shape
        assert shape[-1] == 1  # k 是最小可能值
        stats.mannwhitneyu(0*x, y, method='exact', alternative='greater')
        assert shape == _mwu_state.configurations.shape
    # 使用 pytest 的参数化功能，为三种不同的 alternative 值分别运行测试
    @pytest.mark.parametrize('alternative', ['less', 'greater', 'two-sided'])
    def test_permutation_method(self, alternative):
        # 创建一个指定种子的随机数生成器对象
        rng = np.random.default_rng(7600451795963068007)
        # 生成一个 2x5 的随机数组 x
        x = rng.random(size=(2, 5))
        # 生成一个 2x6 的随机数组 y
        y = rng.random(size=(2, 6))
        # 使用 Mann-Whitney U 检验计算统计量和 p 值，使用 PermutationMethod 方法
        res = stats.mannwhitneyu(x, y, method=stats.PermutationMethod(),
                                 alternative=alternative, axis=1)
        # 使用 Mann-Whitney U 检验计算统计量和 p 值，使用 'exact' 方法
        res2 = stats.mannwhitneyu(x, y, method='exact',
                                  alternative=alternative, axis=1)
        # 断言两次计算的统计量非常接近，相对误差小于 1e-15
        assert_allclose(res.statistic, res2.statistic, rtol=1e-15)
        # 断言两次计算的 p 值非常接近，相对误差小于 1e-15
        assert_allclose(res.pvalue, res2.pvalue, rtol=1e-15)
class TestSomersD(_TestPythranFunc):
    # 设置测试方法的初始化
    def setup_method(self):
        # 定义测试中使用的数据类型为所有整数和所有浮点数的集合
        self.dtypes = self.ALL_INTEGER + self.ALL_FLOAT
        # 设置测试用例的输入参数字典
        self.arguments = {
            0: (np.arange(10), self.ALL_INTEGER + self.ALL_FLOAT),
            1: (np.arange(10), self.ALL_INTEGER + self.ALL_FLOAT)
        }
        # 从参数中提取输入数组
        input_array = [self.arguments[idx][0] for idx in self.arguments]
        
        # 使用 functools.partial 函数冻结 alternative 参数为 'two-sided'
        # 这样 self.partialfunc 可以简单地是 stats.somersd，因为 alternative 是一个可选参数
        self.partialfunc = functools.partial(stats.somersd, alternative='two-sided')
        
        # 计算期望结果
        self.expected = self.partialfunc(*input_array)

    # 定义 Pythran 函数的测试方法
    def pythranfunc(self, *args):
        # 调用 partialfunc 方法计算结果
        res = self.partialfunc(*args)
        # 断言结果的统计量与期望值的统计量非常接近
        assert_allclose(res.statistic, self.expected.statistic, atol=1e-15)
        # 断言结果的 p 值与期望值的 p 值非常接近
        assert_allclose(res.pvalue, self.expected.pvalue, atol=1e-15)

    # 定义关键字参数的 Pythran 函数测试方法
    def test_pythranfunc_keywords(self):
        # 不指定可选关键字参数的情况下计算 somersd
        table = [[27, 25, 14, 7, 0], [7, 14, 18, 35, 12], [1, 3, 2, 7, 17]]
        res1 = stats.somersd(table)
        
        # 使用默认值指定可选关键字参数的情况下计算 somersd
        optional_args = self.get_optional_args(stats.somersd)
        res2 = stats.somersd(table, **optional_args)
        
        # 断言两种情况下的统计量非常接近
        assert_allclose(res1.statistic, res2.statistic, atol=1e-15)
        # 断言两种情况下的 p 值非常接近
        assert_allclose(res1.pvalue, res2.pvalue, atol=1e-15)

    # 定义不对称性的测试方法
    def test_asymmetry(self):
        # 测试 somersd 相对于输入顺序的不对称性，并验证约定的正确性
        # 第一个输入是行变量且是独立的数据来自维基百科
        x = [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 1, 2,
             2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
        y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        
        # 与 SAS FREQ 的结果进行交叉验证
        d_cr = 0.272727272727270
        d_rc = 0.342857142857140
        p = 0.092891940883700  # 两个方向的相同 p 值
        res = stats.somersd(x, y)
        
        # 断言结果的统计量非常接近预期值 d_cr
        assert_allclose(res.statistic, d_cr, atol=1e-15)
        # 断言结果的 p 值非常接近预期值 p
        assert_allclose(res.pvalue, p, atol=1e-4)
        # 断言结果的表形状为 (3, 2)
        assert_equal(res.table.shape, (3, 2))
        
        # 再次交叉验证但是输入顺序相反的情况
        res = stats.somersd(y, x)
        # 断言结果的统计量非常接近预期值 d_rc
        assert_allclose(res.statistic, d_rc, atol=1e-15)
        # 断言结果的 p 值非常接近预期值 p
        assert_allclose(res.pvalue, p, atol=1e-15)
        # 断言结果的表形状为 (2, 3)
        assert_equal(res.table.shape, (2, 3))
    def test_somers_original(self):
        # test against Somers' original paper [1]

        # Table 5A
        # Somers' convention was column IV
        table = np.array([[8, 2], [6, 5], [3, 4], [1, 3], [2, 3]])
        # Our convention (and that of SAS FREQ) is row IV
        table = table.T  # 转置表格，以匹配我们的约定
        dyx = 129/340
        assert_allclose(stats.somersd(table).statistic, dyx)

        # table 7A - d_yx = 1
        table = np.array([[25, 0], [85, 0], [0, 30]])
        dxy, dyx = 3300/5425, 3300/3300
        assert_allclose(stats.somersd(table).statistic, dxy)
        assert_allclose(stats.somersd(table.T).statistic, dyx)  # 测试转置表格的统计量

        # table 7B - d_yx < 0
        table = np.array([[25, 0], [0, 30], [85, 0]])
        dyx = -1800/3300
        assert_allclose(stats.somersd(table.T).statistic, dyx)  # 测试转置表格的统计量

    def test_contingency_table_with_zero_rows_cols(self):
        # test that zero rows/cols in contingency table don't affect result

        N = 100
        shape = 4, 6
        size = np.prod(shape)

        np.random.seed(0)
        s = stats.multinomial.rvs(N, p=np.ones(size)/size).reshape(shape)
        res = stats.somersd(s)

        s2 = np.insert(s, 2, np.zeros(shape[1]), axis=0)  # 在第2行插入全零行
        res2 = stats.somersd(s2)

        s3 = np.insert(s, 2, np.zeros(shape[0]), axis=1)  # 在第2列插入全零列
        res3 = stats.somersd(s3)

        s4 = np.insert(s2, 2, np.zeros(shape[0]+1), axis=1)  # 在第2列插入全零列
        res4 = stats.somersd(s4)

        # Cross-check with result from SAS FREQ:
        assert_allclose(res.statistic, -0.116981132075470, atol=1e-15)
        assert_allclose(res.statistic, res2.statistic)
        assert_allclose(res.statistic, res3.statistic)
        assert_allclose(res.statistic, res4.statistic)

        assert_allclose(res.pvalue, 0.156376448188150, atol=1e-15)
        assert_allclose(res.pvalue, res2.pvalue)
        assert_allclose(res.pvalue, res3.pvalue)
        assert_allclose(res.pvalue, res4.pvalue)
    # 定义一个测试函数，用于测试无效的列联表情况
    def test_invalid_contingency_tables(self):
        # 设定样本总数为100
        N = 100
        # 设定列联表的形状为4行6列
        shape = 4, 6
        # 计算列联表的总元素个数
        size = np.prod(shape)

        # 设置随机数种子为0
        np.random.seed(0)
        # 生成一个有效的列联表 s，其元素服从多项分布
        s = stats.multinomial.rvs(N, p=np.ones(size)/size).reshape(shape)

        # 创建一个比 s 中所有元素小2的列联表 s5
        s5 = s - 2
        # 设定错误信息字符串
        message = "All elements of the contingency table must be non-negative"
        # 使用 assert_raises 检查是否抛出 ValueError 异常，并匹配特定错误信息
        with assert_raises(ValueError, match=message):
            stats.somersd(s5)

        # 创建一个比 s 中所有元素大0.01的列联表 s6
        s6 = s + 0.01
        # 更新错误信息字符串
        message = "All elements of the contingency table must be integer"
        # 使用 assert_raises 检查是否抛出 ValueError 异常，并匹配特定错误信息
        with assert_raises(ValueError, match=message):
            stats.somersd(s6)

        # 更新错误信息字符串，检查至少有两个非零元素的列联表
        message = ("At least two elements of the contingency "
                   "table must be nonzero.")
        # 使用 assert_raises 检查是否抛出 ValueError 异常，并匹配特定错误信息
        with assert_raises(ValueError, match=message):
            stats.somersd([[]])

        # 检查只有一个元素的列联表
        with assert_raises(ValueError, match=message):
            stats.somersd([[1]])

        # 创建一个全零的3x3列联表 s7
        s7 = np.zeros((3, 3))
        # 使用 assert_raises 检查是否抛出 ValueError 异常，并匹配特定错误信息
        with assert_raises(ValueError, match=message):
            stats.somersd(s7)

        # 修改 s7 的一个元素为1，依然检查是否抛出上述的 ValueError 异常
        s7[0, 1] = 1
        # 使用 assert_raises 检查是否抛出 ValueError 异常，并匹配特定错误信息
        with assert_raises(ValueError, match=message):
            stats.somersd(s7)

    # 定义一个测试函数，用于检查只有排名影响结果的情况
    def test_only_ranks_matter(self):
        # 只有输入数据的排名应该影响结果
        x = [1, 2, 3]
        x2 = [-1, 2.1, np.inf]
        y = [3, 2, 1]
        y2 = [0, -0.5, -np.inf]
        # 计算两组数据的 Somers' D 统计量
        res = stats.somersd(x, y)
        res2 = stats.somersd(x2, y2)
        # 断言两组数据的统计量和 p 值相等
        assert_equal(res.statistic, res2.statistic)
        assert_equal(res.pvalue, res2.pvalue)

    # 定义一个测试函数，用于检查列联表是否正确返回
    def test_contingency_table_return(self):
        # 检查列联表是否正确返回
        x = np.arange(10)
        y = np.arange(10)
        # 计算 x 和 y 的 Somers' D 统计量
        res = stats.somersd(x, y)
        # 断言返回的列联表是否为单位矩阵
        assert_equal(res.table, np.eye(10))
    def test_somersd_alternative(self):
        # 测试 alternative 参数，使用渐近方法（由于平局）
        
        # 基于 scipy.stats.test_stats.TestCorrSpearman2::test_alternative

        # 定义两组数据
        x1 = [1, 2, 3, 4, 5]
        x2 = [5, 6, 7, 8, 7]

        # 强正相关
        expected = stats.somersd(x1, x2, alternative="two-sided")
        assert expected.statistic > 0

        # 排名相关系数 > 0 -> 较大的 "less" p 值
        res = stats.somersd(x1, x2, alternative="less")
        assert_equal(res.statistic, expected.statistic)
        assert_allclose(res.pvalue, 1 - (expected.pvalue / 2))

        # 排名相关系数 > 0 -> 较小的 "greater" p 值
        res = stats.somersd(x1, x2, alternative="greater")
        assert_equal(res.statistic, expected.statistic)
        assert_allclose(res.pvalue, expected.pvalue / 2)

        # 翻转排名相关方向
        x2.reverse()

        # 强负相关
        expected = stats.somersd(x1, x2, alternative="two-sided")
        assert expected.statistic < 0

        # 排名相关系数 < 0 -> 较大的 "greater" p 值
        res = stats.somersd(x1, x2, alternative="greater")
        assert_equal(res.statistic, expected.statistic)
        assert_allclose(res.pvalue, 1 - (expected.pvalue / 2))

        # 排名相关系数 < 0 -> 较小的 "less" p 值
        res = stats.somersd(x1, x2, alternative="less")
        assert_equal(res.statistic, expected.statistic)
        assert_allclose(res.pvalue, expected.pvalue / 2)

        # 引发 ValueError，测试不支持的 alternative 值
        with pytest.raises(ValueError, match="`alternative` must be..."):
            stats.somersd(x1, x2, alternative="ekki-ekki")

    @pytest.mark.parametrize("positive_correlation", (False, True))
    def test_somersd_perfect_correlation(self, positive_correlation):
        # 在添加 `alternative` 参数之前，完美相关性被视为特殊情况。
        # 现在它被视为任何其他情况，但确保没有除以零的警告或相关错误

        # 定义两组数据
        x1 = np.arange(10)
        x2 = x1 if positive_correlation else np.flip(x1)
        expected_statistic = 1 if positive_correlation else -1

        # 完美相关 -> 极小的 "two-sided" p 值 (0)
        res = stats.somersd(x1, x2, alternative="two-sided")
        assert res.statistic == expected_statistic
        assert res.pvalue == 0

        # 排名相关系数 > 0 -> 较大的 "less" p 值 (1)，或者 (0)
        res = stats.somersd(x1, x2, alternative="less")
        assert res.statistic == expected_statistic
        assert res.pvalue == (1 if positive_correlation else 0)

        # 排名相关系数 > 0 -> 较小的 "greater" p 值 (0)，或者 (1)
        res = stats.somersd(x1, x2, alternative="greater")
        assert res.statistic == expected_statistic
        assert res.pvalue == (0 if positive_correlation else 1)
    def test_somersd_large_inputs_gh18132(self):
        # 定义测试函数，用于测试在可能发生溢出的大输入情况下是否给出了预期的输出。
        # 这在二进制输入的情况下进行测试。参见 gh-18126.

        # 生成随机的类别列表 1-2（二进制）
        classes = [1, 2]
        n_samples = 10 ** 6
        random.seed(6272161)
        # 从类别列表中随机选择 n_samples 次，作为输入数据 x 和 y
        x = random.choices(classes, k=n_samples)
        y = random.choices(classes, k=n_samples)

        # 获取用于比较的值：来自 sklearn 的输出
        # from sklearn import metrics
        # val_auc_sklearn = metrics.roc_auc_score(x, y)
        # 将其转换为基尼系数（Gini = (AUC*2)-1）
        # val_sklearn = 2 * val_auc_sklearn - 1
        val_sklearn = -0.001528138777036947

        # 计算 Somers' D 统计量，其应该与 val_sklearn 的结果在机器精度范围内相等
        val_scipy = stats.somersd(x, y).statistic
        # 使用 assert_allclose 函数断言 val_sklearn 和 val_scipy 在给定的绝对容差范围内相等
        assert_allclose(val_sklearn, val_scipy, atol=1e-15)
# 定义一个名为 TestBarnardExact 的测试类，用于展示 barnard_exact() 函数的正确工作。
class TestBarnardExact:
    """Some tests to show that barnard_exact() works correctly."""
    
    # 使用 pytest 的 parametrize 装饰器为 test_precise 方法提供多组参数化输入和预期输出
    @pytest.mark.parametrize(
        "input_sample,expected",
        [
            ([[43, 40], [10, 39]], (3.555406779643, 0.000362832367)),
            ([[100, 2], [1000, 5]], (-1.776382925679, 0.135126970878)),
            ([[2, 7], [8, 2]], (-2.518474945157, 0.019210815430)),
            ([[5, 1], [10, 10]], (1.449486150679, 0.156277546306)),
            ([[5, 15], [20, 20]], (-1.851640199545, 0.066363501421)),
            ([[5, 16], [20, 25]], (-1.609639949352, 0.116984852192)),
            ([[10, 5], [10, 1]], (-1.449486150679, 0.177536588915)),
            ([[5, 0], [1, 4]], (2.581988897472, 0.013671875000)),
            ([[0, 1], [3, 2]], (-1.095445115010, 0.509667991877)),
            ([[0, 2], [6, 4]], (-1.549193338483, 0.197019618792)),
            ([[2, 7], [8, 2]], (-2.518474945157, 0.019210815430)),
        ],
    )
    # 定义 test_precise 方法，用于精确检查 barnard_exact 函数的结果是否符合预期
    def test_precise(self, input_sample, expected):
        """The expected values have been generated by R, using a resolution
        for the nuisance parameter of 1e-6 :
        ```R
        library(Barnard)
        options(digits=10)
        barnard.test(43, 40, 10, 39, dp=1e-6, pooled=TRUE)
        ```
        """
        # 调用 barnard_exact 函数计算输入样本的统计量和 p 值
        res = barnard_exact(input_sample)
        statistic, pvalue = res.statistic, res.pvalue
        # 使用 assert_allclose 断言函数确保统计量和 p 值与预期值非常接近
        assert_allclose([statistic, pvalue], expected)

    # 使用 pytest 的 parametrize 装饰器为 test_pooled_param 方法提供多组参数化输入和预期输出
    @pytest.mark.parametrize(
        "input_sample,expected",
        [
            ([[43, 40], [10, 39]], (3.920362887717, 0.000289470662)),
            ([[100, 2], [1000, 5]], (-1.139432816087, 0.950272080594)),
            ([[2, 7], [8, 2]], (-3.079373904042, 0.020172119141)),
            ([[5, 1], [10, 10]], (1.622375939458, 0.150599922226)),
            ([[5, 15], [20, 20]], (-1.974771239528, 0.063038448651)),
            ([[5, 16], [20, 25]], (-1.722122973346, 0.133329494287)),
            ([[10, 5], [10, 1]], (-1.765469659009, 0.250566655215)),
            ([[5, 0], [1, 4]], (5.477225575052, 0.007812500000)),
            ([[0, 1], [3, 2]], (-1.224744871392, 0.509667991877)),
            ([[0, 2], [6, 4]], (-1.732050807569, 0.197019618792)),
            ([[2, 7], [8, 2]], (-3.079373904042, 0.020172119141)),
        ],
    )
    # 定义 test_pooled_param 方法，用于检查 barnard_exact 函数在不合并参数情况下的结果是否符合预期
    def test_pooled_param(self, input_sample, expected):
        """The expected values have been generated by R, using a resolution
        for the nuisance parameter of 1e-6 :
        ```R
        library(Barnard)
        options(digits=10)
        barnard.test(43, 40, 10, 39, dp=1e-6, pooled=FALSE)
        ```
        """
        # 调用 barnard_exact 函数计算输入样本的统计量和 p 值，不合并参数
        res = barnard_exact(input_sample, pooled=False)
        statistic, pvalue = res.statistic, res.pvalue
        # 使用 assert_allclose 断言函数确保统计量和 p 值与预期值非常接近
        assert_allclose([statistic, pvalue], expected)
    def test_raises(self):
        # 测试当输入的nuisances数量不正确时是否会引发错误。
        error_msg = (
            "Number of points `n` must be strictly positive, found 0"
        )
        # 使用 assert_raises 确保抛出 ValueError 异常，并检查错误消息是否匹配预期
        with assert_raises(ValueError, match=error_msg):
            barnard_exact([[1, 2], [3, 4]], n=0)

        # 测试当输入的形状不正确时是否会引发错误。
        error_msg = "The input `table` must be of shape \\(2, 2\\)."
        # 使用 assert_raises 确保抛出 ValueError 异常，并检查错误消息是否匹配预期
        with assert_raises(ValueError, match=error_msg):
            barnard_exact(np.arange(6).reshape(2, 3))

        # 测试当输入的表格中存在负值时是否会引发错误。
        error_msg = "All values in `table` must be nonnegative."
        # 使用 assert_raises 确保抛出 ValueError 异常，并检查错误消息是否匹配预期
        with assert_raises(ValueError, match=error_msg):
            barnard_exact([[-1, 2], [3, 4]])

        # 测试当输入的 alternative 参数不正确时是否会引发错误。
        error_msg = (
            "`alternative` should be one of {'two-sided', 'less', 'greater'},"
            " found .*"
        )
        # 使用 assert_raises 确保抛出 ValueError 异常，并检查错误消息是否匹配预期
        with assert_raises(ValueError, match=error_msg):
            barnard_exact([[1, 2], [3, 4]], "not-correct")

    @pytest.mark.parametrize(
        "input_sample,expected",
        [
            ([[0, 0], [4, 3]], (1.0, 0)),
        ],
    )
    def test_edge_cases(self, input_sample, expected):
        # 执行 Barnard 精确性检验，并获取结果
        res = barnard_exact(input_sample)
        # 检查返回结果中的统计量和 p 值是否与预期相符
        statistic, pvalue = res.statistic, res.pvalue
        assert_equal(pvalue, expected[0])
        assert_equal(statistic, expected[1])

    @pytest.mark.parametrize(
        "input_sample,expected",
        [
            ([[0, 5], [0, 10]], (1.0, np.nan)),
            ([[5, 0], [10, 0]], (1.0, np.nan)),
        ],
    )
    def test_row_or_col_zero(self, input_sample, expected):
        # 执行 Barnard 精确性检验，并获取结果
        res = barnard_exact(input_sample)
        # 检查返回结果中的统计量和 p 值是否与预期相符
        statistic, pvalue = res.statistic, res.pvalue
        assert_equal(pvalue, expected[0])
        assert_equal(statistic, expected[1])

    @pytest.mark.parametrize(
        "input_sample,expected",
        [
            ([[2, 7], [8, 2]], (-2.518474945157, 0.009886140845)),
            ([[7, 200], [300, 8]], (-21.320036698460, 0.0)),
            ([[21, 28], [1957, 6]], (-30.489638143953, 0.0)),
        ],
    )
    @pytest.mark.parametrize("alternative", ["greater", "less"])
    # 定义一个测试方法，用于比较统计数据的假设检验结果
    def test_less_greater(self, input_sample, expected, alternative):
        """
        "The expected values have been generated by R, using a resolution
        for the nuisance parameter of 1e-6 :
        ```R
        library(Barnard)
        options(digits=10)
        a = barnard.test(2, 7, 8, 2, dp=1e-6, pooled=TRUE)
        a$p.value[1]
        ```
        In this test, we are using the "one-sided" return value `a$p.value[1]`
        to test our pvalue.
        """
        
        # 从期望值元组中获取统计值和小于某个值的预期 p 值
        expected_stat, less_pvalue_expect = expected
        
        # 如果选择了“greater”替代假设，则对输入样本进行反向排序，并调整期望统计值的符号
        if alternative == "greater":
            input_sample = np.array(input_sample)[:, ::-1]
            expected_stat = -expected_stat
        
        # 调用 barnard_exact 函数进行精确巴纳德检验，得到统计量和 p 值
        res = barnard_exact(input_sample, alternative=alternative)
        statistic, pvalue = res.statistic, res.pvalue
        
        # 使用 assert_allclose 断言函数检查统计量和 p 值是否接近期望值，允许的误差为 1e-7
        assert_allclose(
            [statistic, pvalue], [expected_stat, less_pvalue_expect], atol=1e-7
        )
class TestBoschlooExact:
    """Some tests to show that boschloo_exact() works correctly."""

    ATOL = 1e-7  # 定义误差容限为 1e-7

    @pytest.mark.parametrize(
        "input_sample,expected",
        [  # 参数化测试数据和期望结果
            ([[2, 7], [8, 2]], (0.01852173, 0.009886142)),
            ([[5, 1], [10, 10]], (0.9782609, 0.9450994)),
            ([[5, 16], [20, 25]], (0.08913823, 0.05827348)),
            ([[10, 5], [10, 1]], (0.1652174, 0.08565611)),
            ([[5, 0], [1, 4]], (1, 1)),
            ([[0, 1], [3, 2]], (0.5, 0.34375)),
            ([[2, 7], [8, 2]], (0.01852173, 0.009886142)),
            ([[7, 12], [8, 3]], (0.06406797, 0.03410916)),
            ([[10, 24], [25, 37]], (0.2009359, 0.1512882)),
        ],
    )
    def test_less(self, input_sample, expected):
        """The expected values have been generated by R, using a resolution
        for the nuisance parameter of 1e-8 :
        ```R
        library(Exact)
        options(digits=10)
        data <- matrix(c(43, 10, 40, 39), 2, 2, byrow=TRUE)
        a = exact.test(data, method="Boschloo", alternative="less",
                       tsmethod="central", np.interval=TRUE, beta=1e-8)
        ```
        """
        # 调用 boschloo_exact 函数，计算统计量和 p 值
        res = boschloo_exact(input_sample, alternative="less")
        statistic, pvalue = res.statistic, res.pvalue
        # 使用 assert_allclose 断言检查统计量和 p 值与期望值的接近性
        assert_allclose([statistic, pvalue], expected, atol=self.ATOL)

    @pytest.mark.parametrize(
        "input_sample,expected",
        [  # 参数化测试数据和期望结果
            ([[43, 40], [10, 39]], (0.0002875544, 0.0001615562)),
            ([[2, 7], [8, 2]], (0.9990149, 0.9918327)),
            ([[5, 1], [10, 10]], (0.1652174, 0.09008534)),
            ([[5, 15], [20, 20]], (0.9849087, 0.9706997)),
            ([[5, 16], [20, 25]], (0.972349, 0.9524124)),
            ([[5, 0], [1, 4]], (0.02380952, 0.006865367)),
            ([[0, 1], [3, 2]], (1, 1)),
            ([[0, 2], [6, 4]], (1, 1)),
            ([[2, 7], [8, 2]], (0.9990149, 0.9918327)),
            ([[7, 12], [8, 3]], (0.9895302, 0.9771215)),
            ([[10, 24], [25, 37]], (0.9012936, 0.8633275)),
        ],
    )
    def test_greater(self, input_sample, expected):
        """The expected values have been generated by R, using a resolution
        for the nuisance parameter of 1e-8 :
        ```R
        library(Exact)
        options(digits=10)
        data <- matrix(c(43, 10, 40, 39), 2, 2, byrow=TRUE)
        a = exact.test(data, method="Boschloo", alternative="greater",
                       tsmethod="central", np.interval=TRUE, beta=1e-8)
        ```
        """
        # 调用 boschloo_exact 函数，计算统计量和 p 值
        res = boschloo_exact(input_sample, alternative="greater")
        statistic, pvalue = res.statistic, res.pvalue
        # 使用 assert_allclose 断言检查统计量和 p 值与期望值的接近性
        assert_allclose([statistic, pvalue], expected, atol=self.ATOL)
    # 使用 pytest 的 parametrize 装饰器来定义多组输入参数和预期输出
    @pytest.mark.parametrize(
        "input_sample,expected",
        [
            # 第一组测试数据
            ([[43, 40], [10, 39]], (0.0002875544, 0.0003231115)),
            # 第二组测试数据
            ([[2, 7], [8, 2]], (0.01852173, 0.01977228)),
            # 第三组测试数据
            ([[5, 1], [10, 10]], (0.1652174, 0.1801707)),
            # 第四组测试数据
            ([[5, 16], [20, 25]], (0.08913823, 0.116547)),
            # 第五组测试数据
            ([[5, 0], [1, 4]], (0.02380952, 0.01373073)),
            # 第六组测试数据
            ([[0, 1], [3, 2]], (0.5, 0.6875)),
            # 第七组测试数据
            ([[2, 7], [8, 2]], (0.01852173, 0.01977228)),
            # 第八组测试数据
            ([[7, 12], [8, 3]], (0.06406797, 0.06821831)),
        ],
    )
    # 定义测试方法 test_two_sided，用于测试双侧 boschloo_exact 算法的结果是否符合预期
    def test_two_sided(self, input_sample, expected):
        """The expected values have been generated by R, using a resolution
        for the nuisance parameter of 1e-8 :
        ```R
        library(Exact)
        options(digits=10)
        data <- matrix(c(43, 10, 40, 39), 2, 2, byrow=TRUE)
        a = exact.test(data, method="Boschloo", alternative="two.sided",
                       tsmethod="central", np.interval=TRUE, beta=1e-8)
        ```
        """
        # 调用 boschloo_exact 函数计算测试样本的统计量和 p 值
        res = boschloo_exact(input_sample, alternative="two-sided", n=64)
        # 设置断言，验证计算出的统计量和 p 值与预期值在允许误差范围内的相似性
        statistic, pvalue = res.statistic, res.pvalue
        assert_allclose([statistic, pvalue], expected, atol=self.ATOL)
    
    # 定义测试方法 test_raises，用于验证 boschloo_exact 函数在异常情况下是否能正确抛出异常
    def test_raises(self):
        # 测试当 nuisances 数量输入错误时是否抛出 ValueError 异常
        error_msg = (
            "Number of points `n` must be strictly positive, found 0"
        )
        with assert_raises(ValueError, match=error_msg):
            boschloo_exact([[1, 2], [3, 4]], n=0)
    
        # 测试当输入形状错误时是否抛出 ValueError 异常
        error_msg = "The input `table` must be of shape \\(2, 2\\)."
        with assert_raises(ValueError, match=error_msg):
            boschloo_exact(np.arange(6).reshape(2, 3))
    
        # 测试所有值必须为非负数时是否抛出 ValueError 异常
        error_msg = "All values in `table` must be nonnegative."
        with assert_raises(ValueError, match=error_msg):
            boschloo_exact([[-1, 2], [3, 4]])
    
        # 测试错误的 alternative 参数是否能正确抛出 ValueError 异常
        error_msg = (
            r"`alternative` should be one of \('two-sided', 'less', "
            r"'greater'\), found .*"
        )
        with assert_raises(ValueError, match=error_msg):
            boschloo_exact([[1, 2], [3, 4]], "not-correct")
    
    # 使用 pytest 的 parametrize 装饰器定义另一组输入参数和预期输出
    @pytest.mark.parametrize(
        "input_sample,expected",
        [
            # 第一组特殊情况测试数据：所有元素为 0
            ([[0, 5], [0, 10]], (np.nan, np.nan)),
            # 第二组特殊情况测试数据：部分行或列全部为 0
            ([[5, 0], [10, 0]], (np.nan, np.nan)),
        ],
    )
    # 定义测试方法 test_row_or_col_zero，验证在行或列全为 0 时的 boschloo_exact 函数返回是否符合预期
    def test_row_or_col_zero(self, input_sample, expected):
        # 调用 boschloo_exact 函数计算统计量和 p 值
        res = boschloo_exact(input_sample)
        statistic, pvalue = res.statistic, res.pvalue
        # 断言验证 p 值和统计量是否符合预期
        assert_equal(pvalue, expected[0])
        assert_equal(statistic, expected[1])
    # 定义一个测试方法，用于验证双边检验中的 p 值不会超过 1，即使是两个单边检验中最小 p 值的两倍。参见 gh-15345。
    def test_two_sided_gt_1(self):
        # 构建一个二维表格，用于进行精确的 Boschloo 检验
        tbl = [[1, 1], [13, 12]]
        # 使用 Boschloo 方法进行单边检验，并获取其 p 值
        pl = boschloo_exact(tbl, alternative='less').pvalue
        pg = boschloo_exact(tbl, alternative='greater').pvalue
        # 断言条件：两个单边检验中最小 p 值的两倍不大于 1
        assert 2*min(pl, pg) > 1
        # 使用 Boschloo 方法进行双边检验，并获取其 p 值
        pt = boschloo_exact(tbl, alternative='two-sided').pvalue
        # 断言条件：双边检验的 p 值应等于 1.0
        assert pt == 1.0

    # 使用参数化测试，验证 Boschloo 方法的统计量与 Fisher 精确检验的 p 值（用于单边检验）是否一致。参见 gh-15345。
    @pytest.mark.parametrize("alternative", ("less", "greater"))
    def test_against_fisher_exact(self, alternative):
        # 构建一个二维表格，用于进行精确的 Boschloo 检验
        tbl = [[2, 7], [8, 2]]
        # 使用 Boschloo 方法进行检验，并获取其统计量
        boschloo_stat = boschloo_exact(tbl, alternative=alternative).statistic
        # 使用 Fisher 精确检验进行单边检验，并获取其 p 值
        fisher_p = stats.fisher_exact(tbl, alternative=alternative)[1]
        # 断言条件：验证 Boschloo 方法的统计量与 Fisher 精确检验的 p 值是否接近
        assert_allclose(boschloo_stat, fisher_p)
class TestCvm_2samp:
    # 使用 pytest 的 parametrize 装饰器对 test_too_small_input 方法进行参数化测试
    @pytest.mark.parametrize('args', [([], np.arange(5)),
                                      (np.arange(5), [1])])
    def test_too_small_input(self, args):
        # 测试在输入过小时是否触发警告 SmallSampleWarning，并匹配给定的警告信息
        with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
            # 调用 cramervonmises_2samp 函数计算结果
            res = cramervonmises_2samp(*args)
            # 断言计算结果的 statistic 属性为 NaN
            assert_equal(res.statistic, np.nan)
            # 断言计算结果的 pvalue 属性为 NaN
            assert_equal(res.pvalue, np.nan)

    # 测试当输入参数无效时是否触发 ValueError 异常，并匹配给定的错误消息
    def test_invalid_input(self):
        y = np.arange(5)
        msg = 'method must be either auto, exact or asymptotic'
        with pytest.raises(ValueError, match=msg):
            cramervonmises_2samp(y, y, 'xyz')

    # 对列表输入进行测试，分别使用列表和数组形式的输入，比较两者的计算结果
    def test_list_input(self):
        x = [2, 3, 4, 7, 6]
        y = [0.2, 0.7, 12, 18]
        r1 = cramervonmises_2samp(x, y)
        r2 = cramervonmises_2samp(np.array(x), np.array(y))
        # 断言两种输入方式得到的 statistic 和 pvalue 相等
        assert_equal((r1.statistic, r1.pvalue), (r2.statistic, r2.pvalue))

    # 进行 Conover 文献中的例子测试，验证结果与文献中给出的值是否接近
    def test_example_conover(self):
        # Example 2 in Section 6.2 of W.J. Conover: Practical Nonparametric
        # Statistics, 1971.
        x = [7.6, 8.4, 8.6, 8.7, 9.3, 9.9, 10.1, 10.6, 11.2]
        y = [5.2, 5.7, 5.9, 6.5, 6.8, 8.2, 9.1, 9.8, 10.8, 11.3, 11.5, 12.3,
             12.5, 13.4, 14.6]
        # 计算 Cramer-von-Mises 统计量和 p 值，并断言其接近文献中的值
        r = cramervonmises_2samp(x, y)
        assert_allclose(r.statistic, 0.262, atol=1e-3)
        assert_allclose(r.pvalue, 0.18, atol=1e-2)

    # 使用 parametrize 装饰器对 test_exact_pvalue 方法进行参数化测试
    @pytest.mark.parametrize('statistic, m, n, pval',
                             [(710, 5, 6, 48./462),
                              (1897, 7, 7, 117./1716),
                              (576, 4, 6, 2./210),
                              (1764, 6, 7, 2./1716)])
    def test_exact_pvalue(self, statistic, m, n, pval):
        # 使用文献 Anderson: On the distribution of the two-sample Cramer-von-Mises
        # criterion, 1962 中的精确值进行断言
        assert_equal(_pval_cvm_2samp_exact(statistic, m, n), pval)

    # 标记为 xslow 的测试，用于大样本的检验，确保计算的 p 值不为 0、1 或 NaN
    @pytest.mark.xslow
    def test_large_sample(self):
        # 对于大样本，统计量 U 会非常大，进行 p 值的检查
        np.random.seed(4367)
        x = distributions.norm.rvs(size=1000000)
        y = distributions.norm.rvs(size=900000)
        r = cramervonmises_2samp(x, y)
        assert_(0 < r.pvalue < 1)
        r = cramervonmises_2samp(x, y+0.1)
        assert_(0 < r.pvalue < 1)

    # 比较精确方法和渐近方法的结果是否一致
    def test_exact_vs_asymptotic(self):
        np.random.seed(0)
        x = np.random.rand(7)
        y = np.random.rand(8)
        r1 = cramervonmises_2samp(x, y, method='exact')
        r2 = cramervonmises_2samp(x, y, method='asymptotic')
        assert_equal(r1.statistic, r2.statistic)
        assert_allclose(r1.pvalue, r2.pvalue, atol=1e-2)
    # 定义一个测试方法，用于测试 `cramervonmises_2samp` 函数的自动选择方法功能
    def test_method_auto(self):
        # 创建一个包含 0 到 19 的 NumPy 数组
        x = np.arange(20)
        # 创建一个包含三个浮点数的列表
        y = [0.5, 4.7, 13.1]
        # 使用确切方法计算 Cramér-von Mises 统计量和 p 值
        r1 = cramervonmises_2samp(x, y, method='exact')
        # 使用自动选择方法计算 Cramér-von Mises 统计量和 p 值
        r2 = cramervonmises_2samp(x, y, method='auto')
        # 断言两次计算的 p 值相等
        assert_equal(r1.pvalue, r2.pvalue)
        
        # 如果一个样本包含超过 20 个观测值，切换到渐近方法
        # 扩展 NumPy 数组以包含 0 到 21
        x = np.arange(21)
        # 使用渐近方法计算 Cramér-von Mises 统计量和 p 值
        r1 = cramervonmises_2samp(x, y, method='asymptotic')
        # 再次使用自动选择方法计算 Cramér-von Mises 统计量和 p 值
        r2 = cramervonmises_2samp(x, y, method='auto')
        # 断言两次计算的 p 值相等
        assert_equal(r1.pvalue, r2.pvalue)

    # 定义一个测试方法，用于测试 `cramervonmises_2samp` 函数处理相同输入的情况
    def test_same_input(self):
        # 确保可以处理平凡的边缘情况
        # 注意 `_cdf_cvm_inf(0)` 返回 nan。通过在统计量非常小的情况下返回 pvalue=1 来避免 nan
        # 创建包含 0 到 14 的 NumPy 数组
        x = np.arange(15)
        # 对相同的输入进行 Cramér-von Mises 测试
        res = cramervonmises_2samp(x, x)
        # 断言统计量和 p 值分别为 (0.0, 1.0)
        assert_equal((res.statistic, res.pvalue), (0.0, 1.0))
        
        # 检查确切的 p 值
        # 对前四个元素相同的输入进行 Cramér-von Mises 测试
        res = cramervonmises_2samp(x[:4], x[:4])
        # 断言统计量和 p 值分别为 (0.0, 1.0)
        assert_equal((res.statistic, res.pvalue), (0.0, 1.0))
class TestTukeyHSD:
    # 定义测试类 TestTukeyHSD，用于 Tukey HSD 多重比较方法的测试

    data_same_size = ([24.5, 23.5, 26.4, 27.1, 29.9],
                      [28.4, 34.2, 29.5, 32.2, 30.1],
                      [26.1, 28.3, 24.3, 26.2, 27.8])
    # 相同大小的数据样本，每个元素是一个列表，表示每组的数据

    data_diff_size = ([24.5, 23.5, 26.28, 26.4, 27.1, 29.9, 30.1, 30.1],
                      [28.4, 34.2, 29.5, 32.2, 30.1],
                      [26.1, 28.3, 24.3, 26.2, 27.8])
    # 不同大小的数据样本，每个元素是一个列表，表示每组的数据

    extreme_size = ([24.5, 23.5, 26.4],
                    [28.4, 34.2, 29.5, 32.2, 30.1, 28.4, 34.2, 29.5, 32.2, 30.1],
                    [26.1, 28.3, 24.3, 26.2, 27.8])
    # 极端大小差异的数据样本，每个元素是一个列表，表示每组的数据

    sas_same_size = """
    Comparison LowerCL Difference UpperCL Significance
    2 - 3    0.6908830568    4.34    7.989116943        1
    2 - 1    0.9508830568    4.6     8.249116943     1
    3 - 2    -7.989116943    -4.34    -0.6908830568    1
    3 - 1    -3.389116943    0.26    3.909116943        0
    1 - 2    -8.249116943    -4.6    -0.9508830568    1
    1 - 3    -3.909116943    -0.26    3.389116943        0
    """
    # 相同大小数据样本的 Tukey HSD 结果，每行包含比较组、置信区间、差异、显著性

    sas_diff_size = """
    Comparison LowerCL Difference UpperCL Significance
    2 - 1    0.2679292645    3.645    7.022070736        1
    2 - 3    0.5934764007    4.34    8.086523599        1
    1 - 2    -7.022070736    -3.645    -0.2679292645    1
    1 - 3    -2.682070736    0.695    4.072070736        0
    3 - 2    -8.086523599    -4.34    -0.5934764007    1
    3 - 1    -4.072070736    -0.695    2.682070736        0
    """
    # 不同大小数据样本的 Tukey HSD 结果，每行包含比较组、置信区间、差异、显著性

    sas_extreme = """
    Comparison LowerCL Difference UpperCL Significance
    2 - 3    1.561605075        4.34    7.118394925        1
    2 - 1    2.740784879        6.08    9.419215121        1
    3 - 2    -7.118394925    -4.34    -1.561605075    1
    3 - 1    -1.964526566    1.74    5.444526566        0
    1 - 2    -9.419215121    -6.08    -2.740784879    1
    1 - 3    -5.444526566    -1.74    1.964526566        0
    """
    # 极端大小差异数据样本的 Tukey HSD 结果，每行包含比较组、置信区间、差异、显著性

    @pytest.mark.parametrize("data,res_expect_str,atol",
                             ((data_same_size, sas_same_size, 1e-4),
                              (data_diff_size, sas_diff_size, 1e-4),
                              (extreme_size, sas_extreme, 1e-10),
                              ),
                             ids=["equal size sample",
                                  "unequal sample size",
                                  "extreme sample size differences"])
    # 使用 pytest 的 parametrize 装饰器定义参数化测试，测试数据、期望结果字符串、误差容忍度
    def test_compare_sas(self, data, res_expect_str, atol):
        '''
        SAS code used to generate results for each sample:
        DATA ACHE;
        INPUT BRAND RELIEF;
        CARDS;
        1 24.5
        ...
        3 27.8
        ;
        ods graphics on;   ODS RTF;ODS LISTING CLOSE;
           PROC ANOVA DATA=ACHE;
           CLASS BRAND;
           MODEL RELIEF=BRAND;
           MEANS BRAND/TUKEY CLDIFF;
           TITLE 'COMPARE RELIEF ACROSS MEDICINES  - ANOVA EXAMPLE';
           ods output  CLDiffs =tc;
        proc print data=tc;
            format LowerCL 17.16 UpperCL 17.16 Difference 17.16;
            title "Output with many digits";
        RUN;
        QUIT;
        ODS RTF close;
        ODS LISTING;
        '''
        
        # 将期望结果字符串转换为 NumPy 数组，用于断言比较
        res_expect = np.asarray(res_expect_str.replace(" - ", " ").split()[5:],
                                dtype=float).reshape((6, 6))
        
        # 使用给定数据进行 Tukey HSD 检验
        res_tukey = stats.tukey_hsd(*data)
        
        # 计算 Tukey HSD 结果的置信区间
        conf = res_tukey.confidence_interval()
        
        # 遍历比较结果
        for i, j, l, s, h, sig in res_expect:
            i, j = int(i) - 1, int(j) - 1
            # 断言置信区间的下界值与期望值相近
            assert_allclose(conf.low[i, j], l, atol=atol)
            # 断言 Tukey 统计值与期望值相近
            assert_allclose(res_tukey.statistic[i, j], s, atol=atol)
            # 断言置信区间的上界值与期望值相近
            assert_allclose(conf.high[i, j], h, atol=atol)
            # 断言 Tukey 的 p 值是否小于等于 0.05 与期望的二元结果一致
            assert_allclose((res_tukey.pvalue[i, j] <= .05), sig == 1)

    matlab_sm_siz = """
        1    2    -8.2491590248597    -4.6    -0.9508409751403    0.0144483269098
        1    3    -3.9091590248597    -0.26    3.3891590248597    0.9803107240900
        2    3    0.6908409751403    4.34    7.9891590248597    0.0203311368795
        """

    matlab_diff_sz = """
        1    2    -7.02207069748501    -3.645    -0.26792930251500 0.03371498443080
        1    3    -2.68207069748500    0.695    4.07207069748500 0.85572267328807
        2    3    0.59347644287720    4.34    8.08652355712281 0.02259047020620
        """

    # 使用 pytest 的参数化装饰器标记多组测试数据和期望结果
    @pytest.mark.parametrize("data,res_expect_str,atol",
                             ((data_same_size, matlab_sm_siz, 1e-12),
                              (data_diff_size, matlab_diff_sz, 1e-7)),
                             ids=["equal size sample",
                                  "unequal size sample"])
    def test_compare_matlab(self, data, res_expect_str, atol):
        """
        与 MATLAB 的结果进行比较测试

        vals = [24.5, 23.5,  26.4, 27.1, 29.9, 28.4, 34.2, 29.5, 32.2, 30.1,
         26.1, 28.3, 24.3, 26.2, 27.8]
        names = {'zero', 'zero', 'zero', 'zero', 'zero', 'one', 'one', 'one',
         'one', 'one', 'two', 'two', 'two', 'two', 'two'}
        [p,t,stats] = anova1(vals,names,"off");
        [c,m,h,nms] = multcompare(stats, "CType","hsd");
        """
        # 将期望的结果字符串转换为 NumPy 数组
        res_expect = np.asarray(res_expect_str.split(),
                                dtype=float).reshape((3, 6))
        # 使用给定的数据计算 Tukey 的 HSD 测试结果
        res_tukey = stats.tukey_hsd(*data)
        # 计算置信区间
        conf = res_tukey.confidence_interval()
        # 遍历比较结果
        for i, j, l, s, h, p in res_expect:
            i, j = int(i) - 1, int(j) - 1
            # 断言置信区间的下限与期望值接近
            assert_allclose(conf.low[i, j], l, atol=atol)
            # 断言 Tukey 统计量与期望值接近
            assert_allclose(res_tukey.statistic[i, j], s, atol=atol)
            # 断言置信区间的上限与期望值接近
            assert_allclose(conf.high[i, j], h, atol=atol)
            # 断言 Tukey 的 p 值与期望值接近

            assert_allclose(res_tukey.pvalue[i, j], p, atol=atol)

    def test_compare_r(self):
        """
        对比 R 中的结果和 p 值进行测试:
        来自: https://www.rdocumentation.org/packages/stats/versions/3.6.2/
        topics/TukeyHSD
        > require(graphics)
        > summary(fm1 <- aov(breaks ~ tension, data = warpbreaks))
        > TukeyHSD(fm1, "tension", ordered = TRUE)
        > plot(TukeyHSD(fm1, "tension"))
        Tukey multiple comparisons of means
        95% family-wise confidence level
        factor levels have been ordered
        Fit: aov(formula = breaks ~ tension, data = warpbreaks)
        $tension
        """
        # R 的结果字符串
        str_res = """
                diff        lwr      upr     p adj
        2 - 3  4.722222 -4.8376022 14.28205 0.4630831
        1 - 3 14.722222  5.1623978 24.28205 0.0014315
        1 - 2 10.000000  0.4401756 19.55982 0.0384598
        """
        # 将字符串转换为 NumPy 数组以获取期望的结果
        res_expect = np.asarray(str_res.replace(" - ", " ").split()[5:],
                                dtype=float).reshape((3, 6))
        # 提供的数据用于测试
        data = ([26, 30, 54, 25, 70, 52, 51, 26, 67,
                 27, 14, 29, 19, 29, 31, 41, 20, 44],
                [18, 21, 29, 17, 12, 18, 35, 30, 36,
                 42, 26, 19, 16, 39, 28, 21, 39, 29],
                [36, 21, 24, 18, 10, 43, 28, 15, 26,
                 20, 21, 24, 17, 13, 15, 15, 16, 28])

        # 使用提供的数据计算 Tukey 的 HSD 测试结果
        res_tukey = stats.tukey_hsd(*data)
        # 计算置信区间
        conf = res_tukey.confidence_interval()
        # 遍历比较结果
        for i, j, s, l, h, p in res_expect:
            i, j = int(i) - 1, int(j) - 1
            # 断言置信区间的下限与期望值接近，atol 根据 R 中结果的精度设置
            assert_allclose(conf.low[i, j], l, atol=1e-7)
            # 断言 Tukey 统计量与期望值接近，atol 根据 R 中结果的精度设置
            assert_allclose(res_tukey.statistic[i, j], s, atol=1e-6)
            # 断言置信区间的上限与期望值接近，atol 根据 R 中结果的精度设置
            assert_allclose(conf.high[i, j], h, atol=1e-5)
            # 断言 Tukey 的 p 值与期望值接近，atol 根据 R 中结果的精度设置
            assert_allclose(res_tukey.pvalue[i, j], p, atol=1e-7)
    def test_engineering_stat_handbook(self):
        '''
        Example sourced from:
        https://www.itl.nist.gov/div898/handbook/prc/section4/prc471.htm
        '''
        # 定义四个数据组
        group1 = [6.9, 5.4, 5.8, 4.6, 4.0]
        group2 = [8.3, 6.8, 7.8, 9.2, 6.5]
        group3 = [8.0, 10.5, 8.1, 6.9, 9.3]
        group4 = [5.8, 3.8, 6.1, 5.6, 6.2]
        # 进行 Tukey 的 HSD 分析
        res = stats.tukey_hsd(group1, group2, group3, group4)
        # 获取置信区间
        conf = res.confidence_interval()
        # 设定下界矩阵
        lower = np.asarray([
            [0, 0, 0, -2.25],
            [.29, 0, -2.93, .13],
            [1.13, 0, 0, .97],
            [0, 0, 0, 0]])
        # 设定上界矩阵
        upper = np.asarray([
            [0, 0, 0, 1.93],
            [4.47, 0, 1.25, 4.31],
            [5.31, 0, 0, 5.15],
            [0, 0, 0, 0]])

        # 遍历需要检查的索引对，进行数值比较
        for (i, j) in [(1, 0), (2, 0), (0, 3), (1, 2), (2, 3)]:
            # 断言置信区间的下界值与预期的下界值接近
            assert_allclose(conf.low[i, j], lower[i, j], atol=1e-2)
            # 断言置信区间的上界值与预期的上界值接近
            assert_allclose(conf.high[i, j], upper[i, j], atol=1e-2)

    def test_rand_symm(self):
        # 测试结果的一些预期身份
        np.random.seed(1234)
        # 生成一个随机数据集
        data = np.random.rand(3, 100)
        # 进行 Tukey 的 HSD 分析
        res = stats.tukey_hsd(*data)
        # 获取置信区间
        conf = res.confidence_interval()
        # 置信区间应该是彼此对称的负值
        assert_equal(conf.low, -conf.high.T)
        # `high` 和 `low` 中心对角线应该相同，因为自身比较的平均差异为 0
        assert_equal(np.diagonal(conf.high), conf.high[0, 0])
        assert_equal(np.diagonal(conf.low), conf.low[0, 0])
        # 统计数组应该是反对称的，对角线上为零
        assert_equal(res.statistic, -res.statistic.T)
        assert_equal(np.diagonal(res.statistic), 0)
        # p 值应该是对称的，在与自身比较时为 1
        assert_equal(res.pvalue, res.pvalue.T)
        assert_equal(np.diagonal(res.pvalue), 1)

    def test_no_inf(self):
        # 检查是否会抛出异常，要求数据不能包含无穷大
        with assert_raises(ValueError, match="...must be finite."):
            stats.tukey_hsd([1, 2, 3], [2, np.inf], [6, 7, 3])

    def test_is_1d(self):
        # 检查是否会抛出异常，要求数据必须是一维的
        with assert_raises(ValueError, match="...must be one-dimensional"):
            stats.tukey_hsd([[1, 2], [2, 3]], [2, 5], [5, 23, 6])

    def test_no_empty(self):
        # 检查是否会抛出异常，要求数据列表不能为空
        with assert_raises(ValueError, match="...must be greater than one"):
            stats.tukey_hsd([], [2, 5], [4, 5, 6])

    @pytest.mark.parametrize("nargs", (0, 1))
    def test_not_enough_treatments(self, nargs):
        # 检查是否会抛出异常，要求处理的数据量必须大于 1
        with assert_raises(ValueError, match="...more than 1 treatment."):
            stats.tukey_hsd(*([[23, 7, 3]] * nargs))

    @pytest.mark.parametrize("cl", [-.5, 0, 1, 2])
    def test_conf_level_invalid(self, cl):
        # 检查是否会抛出异常，要求置信水平必须在 0 到 1 之间
        with assert_raises(ValueError, match="must be between 0 and 1"):
            r = stats.tukey_hsd([23, 7, 3], [3, 4], [9, 4])
            r.confidence_interval(cl)
    # 定义一个测试方法，用于验证两组数据的 t 检验结果
    def test_2_args_ttest(self):
        # 使用 Tukey 的 HSD 方法计算两组数据之间的差异显著性检验结果
        res_tukey = stats.tukey_hsd(*self.data_diff_size[:2])
        # 使用独立样本 t 检验计算两组数据之间的差异显著性检验结果
        res_ttest = stats.ttest_ind(*self.data_diff_size[:2])
        # 断言两种方法计算得到的 p 值是否接近
        assert_allclose(res_ttest.pvalue, res_tukey.pvalue[0, 1])
        assert_allclose(res_ttest.pvalue, res_tukey.pvalue[1, 0])
    @pytest.mark.parametrize("c1, n1, c2, n2, p_expect", (
        # 从文献 [1] 中的示例，第 6 章示例：示例 1
        [0, 100, 3, 100, 0.0884],
        [2, 100, 6, 100, 0.1749]
    ))
    # 定义测试函数，用于测试统计模块中的泊松均值检验
    def test_paper_examples(self, c1, n1, c2, n2, p_expect):
        # 调用泊松均值检验函数，返回检验结果对象
        res = stats.poisson_means_test(c1, n1, c2, n2)
        # 断言检验结果的 p 值近似等于预期值，允许的误差为 1e-4
        assert_allclose(res.pvalue, p_expect, atol=1e-4)

    @pytest.mark.parametrize("c1, n1, c2, n2, p_expect, alt, d", (
        # 这些测试案例是使用原始作者的 Fortran 代码生成的，
        # 使用稍微修改的 Fortran 版本，可以在 https://github.com/nolanbconaway/poisson-etest 找到，
        # 添加了额外的测试。
        [20, 10, 20, 10, 0.9999997568929630, 'two-sided', 0],
        [10, 10, 10, 10, 0.9999998403241203, 'two-sided', 0],
        [50, 15, 1, 1, 0.09920321053409643, 'two-sided', .05],
        [3, 100, 20, 300, 0.12202725450896404, 'two-sided', 0],
        [3, 12, 4, 20, 0.40416087318539173, 'greater', 0],
        [4, 20, 3, 100, 0.008053640402974236, 'greater', 0],
        # 发表的论文中不包括 `less` 选项，
        # 因此通过交换参数顺序和 alternative="greater" 计算
        [4, 20, 3, 10, 0.3083216325432898, 'less', 0],
        [1, 1, 50, 15, 0.09322998607245102, 'less', 0]
    ))
    # 定义测试函数，用于验证泊松均值检验在不同情况下的正确性
    def test_fortran_authors(self, c1, n1, c2, n2, p_expect, alt, d):
        # 调用泊松均值检验函数，返回检验结果对象，指定备择假设和差异值
        res = stats.poisson_means_test(c1, n1, c2, n2, alternative=alt, diff=d)
        # 断言检验结果的 p 值近似等于预期值，允许的绝对误差为 2e-6，相对误差为 1e-16
        assert_allclose(res.pvalue, p_expect, atol=2e-6, rtol=1e-16)

    # 定义测试函数，用于验证在极端情况下泊松均值检验的结果
    def test_different_results(self):
        # 由于 Fortran 实现在较高的计数和观测数时会出现问题，因此我们预期结果会有所不同。
        # 通过观察，我们可以推断 p 值接近于 1。
        count1, count2 = 10000, 10000
        nobs1, nobs2 = 10000, 10000
        # 调用泊松均值检验函数，返回检验结果对象
        res = stats.poisson_means_test(count1, nobs1, count2, nobs2)
        # 断言检验结果的 p 值近似等于 1
        assert_allclose(res.pvalue, 1)

    # 定义测试函数，验证修正了原始 Fortran 代码中已知错误的情况
    def test_less_than_zero_lambda_hat2(self):
        # 展示修正了原始 Fortran 代码中已知问题的行为。
        # p 值应该明显接近于 1。
        count1, count2 = 0, 0
        nobs1, nobs2 = 1, 1
        # 调用泊松均值检验函数，返回检验结果对象
        res = stats.poisson_means_test(count1, nobs1, count2, nobs2)
        # 断言检验结果的 p 值近似等于 1
        assert_allclose(res.pvalue, 1)
    # 定义测试函数，验证输入的有效性
    def test_input_validation(self):
        # 初始化计数器和观测数
        count1, count2 = 0, 0
        nobs1, nobs2 = 1, 1

        # 测试非整数事件
        message = '`k1` and `k2` must be integers.'
        # 断言抛出 TypeError 异常，并匹配预期的错误信息
        with assert_raises(TypeError, match=message):
            stats.poisson_means_test(.7, nobs1, count2, nobs2)
        with assert_raises(TypeError, match=message):
            stats.poisson_means_test(count1, nobs1, .7, nobs2)

        # 测试负数事件数
        message = '`k1` and `k2` must be greater than or equal to 0.'
        # 断言抛出 ValueError 异常，并匹配预期的错误信息
        with assert_raises(ValueError, match=message):
            stats.poisson_means_test(-1, nobs1, count2, nobs2)
        with assert_raises(ValueError, match=message):
            stats.poisson_means_test(count1, nobs1, -1, nobs2)

        # 测试负数样本大小
        message = '`n1` and `n2` must be greater than 0.'
        # 断言抛出 ValueError 异常，并匹配预期的错误信息
        with assert_raises(ValueError, match=message):
            stats.poisson_means_test(count1, -1, count2, nobs2)
        with assert_raises(ValueError, match=message):
            stats.poisson_means_test(count1, nobs1, count2, -1)

        # 测试负的差异值
        message = 'diff must be greater than or equal to 0.'
        # 断言抛出 ValueError 异常，并匹配预期的错误信息
        with assert_raises(ValueError, match=message):
            stats.poisson_means_test(count1, nobs1, count2, nobs2, diff=-1)

        # 测试无效的备择假设
        message = 'Alternative must be one of ...'
        # 断言抛出 ValueError 异常，并匹配预期的错误信息
        with assert_raises(ValueError, match=message):
            stats.poisson_means_test(1, 2, 1, 2, alternative='error')
class TestBWSTest:

    def test_bws_input_validation(self):
        # 使用种子创建随机数生成器
        rng = np.random.default_rng(4571775098104213308)

        # 生成两个长度为7的随机一维数组 x 和 y
        x, y = rng.random(size=(2, 7))

        # 指定输入验证失败时的错误信息
        message = '`x` and `y` must be exactly one-dimensional.'
        # 断言调用 bws_test 函数时会引发 ValueError 异常，并匹配指定的错误信息
        with pytest.raises(ValueError, match=message):
            stats.bws_test([x, x], [y, y])

        message = '`x` and `y` must not contain NaNs.'
        with pytest.raises(ValueError, match=message):
            stats.bws_test([np.nan], y)

        message = '`x` and `y` must be of nonzero size.'
        with pytest.raises(ValueError, match=message):
            stats.bws_test(x, [])

        message = 'alternative` must be one of...'
        with pytest.raises(ValueError, match=message):
            stats.bws_test(x, y, alternative='ekki-ekki')

        message = 'method` must be an instance of...'
        with pytest.raises(ValueError, match=message):
            stats.bws_test(x, y, method=42)


    def test_against_published_reference(self):
        # 对比参考文献中的示例2，用于 bws_test 函数测试
        # 参考文献：https://link.springer.com/content/pdf/10.1007/BF02762032.pdf
        x = [1, 2, 3, 4, 6, 7, 8]
        y = [5, 9, 10, 11, 12, 13, 14]
        # 调用 bws_test 函数计算统计量和 p 值
        res = stats.bws_test(x, y, alternative='two-sided')
        # 断言计算结果与预期值接近
        assert_allclose(res.statistic, 5.132, atol=1e-3)
        assert_equal(res.pvalue, 10/3432)


    @pytest.mark.parametrize(('alternative', 'statistic', 'pvalue'),
                             [('two-sided', 1.7510204081633, 0.1264422777777),
                              ('less', -1.7510204081633, 0.05754662004662),
                              ('greater', -1.7510204081633, 0.9424533799534)])
    def test_against_R(self, alternative, statistic, pvalue):
        # 与 R 语言中 BWStest 包的 bws_test 函数进行对比测试
        # 使用随机数生成器创建两个长度为7的随机数组 x 和 y
        rng = np.random.default_rng(4571775098104213308)
        x, y = rng.random(size=(2, 7))
        # 调用 bws_test 函数计算统计量和 p 值，使用参数化测试参数
        res = stats.bws_test(x, y, alternative=alternative)
        # 断言计算结果与预期值接近，设置允许的误差范围
        assert_allclose(res.statistic, statistic, rtol=1e-13)
        assert_allclose(res.pvalue, pvalue, atol=1e-2, rtol=1e-1)

    @pytest.mark.parametrize(('alternative', 'statistic', 'pvalue'),
                             [('two-sided', 1.142629265891, 0.2903950180801),
                              ('less', 0.99629665877411, 0.8545660222131),
                              ('greater', 0.99629665877411, 0.1454339777869)])
    def test_against_R(self, alternative, statistic, pvalue):
        # 与 R 语言中 BWStest 包的 bws_test 函数进行对比测试
        # 使用随机数生成器创建两个长度为7的随机数组 x 和 y
        rng = np.random.default_rng(4571775098104213308)
        x, y = rng.random(size=(2, 7))
        # 调用 bws_test 函数计算统计量和 p 值，使用参数化测试参数
        res = stats.bws_test(x, y, alternative=alternative)
        # 断言计算结果与预期值接近，设置允许的误差范围
        assert_allclose(res.statistic, statistic, rtol=1e-13)
        assert_allclose(res.pvalue, pvalue, atol=1e-2, rtol=1e-1)
    def test_against_R_imbalanced(self, alternative, statistic, pvalue):
        # Test against R library BWStest function bws_test
        # library(BWStest)
        # options(digits=16)
        # x = c(...)
        # y = c(...)
        # bws_test(x, y, alternative='two.sided')
        
        # 使用指定的随机种子创建随机数生成器对象
        rng = np.random.default_rng(5429015622386364034)
        
        # 生成两个长度分别为 9 和 8 的随机数组
        x = rng.random(size=9)
        y = rng.random(size=8)
        
        # 调用 bws_test 函数进行统计检验，检验两组数据的差异性
        res = stats.bws_test(x, y, alternative=alternative)
        
        # 断言检验统计量的近似相等性
        assert_allclose(res.statistic, statistic, rtol=1e-13)
        
        # 断言 p 值的近似相等性
        assert_allclose(res.pvalue, pvalue, atol=1e-2, rtol=1e-1)

    def test_method(self):
        # Test that `method` parameter has the desired effect
        
        # 使用指定的随机种子创建随机数生成器对象
        rng = np.random.default_rng(1520514347193347862)
        
        # 生成两个长度为 10 的随机数组
        x, y = rng.random(size=(2, 10))

        # 使用置换方法进行 bws_test 的测试
        method = stats.PermutationMethod(n_resamples=10, random_state=rng)
        res1 = stats.bws_test(x, y, method=method)

        # 断言生成的置换分布数组长度为 10
        assert len(res1.null_distribution) == 10

        # 再次使用相同的置换方法进行 bws_test 测试，检验置换分布的一致性
        res2 = stats.bws_test(x, y, method=method)

        # 断言两个置换分布数组的近似相等性
        assert_allclose(res1.null_distribution, res2.null_distribution)

        # 使用不同的随机种子创建随机数生成器对象
        rng = np.random.default_rng(5205143471933478621)
        method = stats.PermutationMethod(n_resamples=10, random_state=rng)
        res3 = stats.bws_test(x, y, method=method)

        # 断言不同随机种子生成的置换分布数组不完全相等
        assert not np.allclose(res3.null_distribution, res1.null_distribution)

    def test_directions(self):
        # Sanity check of the sign of the one-sided statistic
        
        # 使用指定的随机种子创建随机数生成器对象
        rng = np.random.default_rng(1520514347193347862)
        
        # 生成一个长度为 5 的随机数组，并生成另一个数组作为其偏移量
        x = rng.random(size=5)
        y = x - 1

        # 进行单侧检验，检验统计量大于 0 的情况
        res = stats.bws_test(x, y, alternative='greater')
        assert res.statistic > 0
        assert_equal(res.pvalue, 1 / len(res.null_distribution))

        # 进行单侧检验，检验统计量小于 0 的情况
        res = stats.bws_test(x, y, alternative='less')
        assert res.statistic > 0
        assert_equal(res.pvalue, 1)

        # 进行单侧检验，检验统计量小于 0 的情况（反向比较）
        res = stats.bws_test(y, x, alternative='less')
        assert res.statistic < 0
        assert_equal(res.pvalue, 1 / len(res.null_distribution))

        # 进行单侧检验，检验统计量大于 0 的情况（反向比较）
        res = stats.bws_test(y, x, alternative='greater')
        assert res.statistic < 0
        assert_equal(res.pvalue, 1)
```