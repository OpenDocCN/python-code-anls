# `D:\src\scipysrc\scipy\scipy\stats\tests\test_morestats.py`

```
# Author:  Travis Oliphant, 2002
#
# Further enhancements and tests added by numerous SciPy developers.
#
# 导入数学库
import math
# 导入警告处理模块
import warnings
# 导入系统相关模块
import sys
# 导入函数工具模块中的部分功能
from functools import partial

# 导入numpy库，并重命名为np
import numpy as np
# 导入numpy测试模块
import numpy.testing
# 从numpy随机模块中导入随机状态生成器
from numpy.random import RandomState
# 从numpy测试模块中导入多个断言函数
from numpy.testing import (assert_array_equal, assert_almost_equal,
                           assert_array_less, assert_array_almost_equal,
                           assert_, assert_allclose, assert_equal,
                           suppress_warnings)
# 导入pytest测试框架
import pytest
# 从pytest中导入断言异常函数，并重命名为assert_raises
from pytest import raises as assert_raises
# 导入正则表达式模块
import re
# 导入scipy库中的optimize、stats、special模块
from scipy import optimize, stats, special
# 导入scipy统计模块中的更多统计功能
from scipy.stats._morestats import _abw_state, _get_As_weibull, _Avals_weibull
# 导入common_tests模块中的check_named_results函数
from .common_tests import check_named_results
# 从_hypotests模块中导入_wilcoxon_distr和_get_wilcoxon_distr2函数
from .._hypotests import _get_wilcoxon_distr, _get_wilcoxon_distr2
# 导入_scipy.stats._binomtest模块中的_binary_search_for_binom_tst函数
from scipy.stats._binomtest import _binary_search_for_binom_tst
# 导入_scipy.stats._distr_params模块中的distcont函数
from scipy.stats._distr_params import distcont
# 导入_scipy.stats._axis_nan_policy模块中的多个警告类和函数
from scipy.stats._axis_nan_policy import (SmallSampleWarning, too_small_nd_omit,
                                          too_small_1d_omit, too_small_1d_not_omit)

# 从scipy.conftest模块中导入array_api_compatible装饰器
from scipy.conftest import array_api_compatible
# 从scipy._lib._array_api模块中导入多个数组API相关函数和标志
from scipy._lib._array_api import (array_namespace, xp_assert_close, xp_assert_less,
                                   xp_assert_equal, is_numpy)

# 使用pytest.mark.skip_xp_backends装饰器标记变量
skip_xp_backends = pytest.mark.skip_xp_backends

# 将distcont变量转换为字典类型
distcont = dict(distcont)  # type: ignore

# Matplotlib不是scipy的依赖项，但在probplot中可以选择使用，因此进行检查
try:
    # 尝试导入matplotlib库
    import matplotlib
    # 设置matplotlib的后端为'Agg'
    matplotlib.rcParams['backend'] = 'Agg'
    # 导入matplotlib.pyplot模块，并重命名为plt
    import matplotlib.pyplot as plt
    # 设置标志表明已成功导入matplotlib
    have_matplotlib = True
except Exception:
    # 如果导入失败，则标志为未成功导入matplotlib
    have_matplotlib = False


# test data gear.dat from NIST for Levene and Bartlett test
# https://www.itl.nist.gov/div898/handbook/eda/section3/eda3581.htm
# 定义用于Levene和Bartlett测试的NIST中的gear.dat测试数据
g1 = [1.006, 0.996, 0.998, 1.000, 0.992, 0.993, 1.002, 0.999, 0.994, 1.000]
g2 = [0.998, 1.006, 1.000, 1.002, 0.997, 0.998, 0.996, 1.000, 1.006, 0.988]
g3 = [0.991, 0.987, 0.997, 0.999, 0.995, 0.994, 1.000, 0.999, 0.996, 0.996]
g4 = [1.005, 1.002, 0.994, 1.000, 0.995, 0.994, 0.998, 0.996, 1.002, 0.996]
g5 = [0.998, 0.998, 0.982, 0.990, 1.002, 0.984, 0.996, 0.993, 0.980, 0.996]
g6 = [1.009, 1.013, 1.009, 0.997, 0.988, 1.002, 0.995, 0.998, 0.981, 0.996]
g7 = [0.990, 1.004, 0.996, 1.001, 0.998, 1.000, 1.018, 1.010, 0.996, 1.002]
g8 = [0.998, 1.000, 1.006, 1.000, 1.002, 0.996, 0.998, 0.996, 1.002, 1.006]
g9 = [1.002, 0.998, 0.996, 0.995, 0.996, 1.004, 1.004, 0.998, 0.999, 0.991]
g10 = [0.991, 0.995, 0.984, 0.994, 0.997, 0.997, 0.991, 0.998, 1.004, 0.997]


# The loggamma RVS stream is changing due to gh-13349; this version
# preserves the old stream so that tests don't change.
# 定义_old_loggamma_rvs函数，用于保持旧版loggamma随机变量生成器的兼容性
def _old_loggamma_rvs(*args, **kwargs):
    # 使用stats.gamma.rvs函数生成loggamma随机变量，然后返回其自然对数
    return np.log(stats.gamma.rvs(*args, **kwargs))


# 定义TestBayes_mvs测试类，用于测试贝叶斯统计相关的函数
class TestBayes_mvs:
   `
    def test_basic(self):
        # 测试基本功能，数据样本列表
        data = [6, 9, 12, 7, 8, 8, 13]
        # 调用 stats.bayes_mvs 函数计算数据的贝叶斯估计，返回均值、方差和标准差
        mean, var, std = stats.bayes_mvs(data)
        # 断言均值统计值与期望值 9.0 的接近程度
        assert_almost_equal(mean.statistic, 9.0)
        # 断言均值的区间值与期望区间的接近程度，容忍误差 1e-6
        assert_allclose(mean.minmax, (7.103650222492964, 10.896349777507034),
                        rtol=1e-6)

        # 断言方差统计值与期望值 10.0 的接近程度
        assert_almost_equal(var.statistic, 10.0)
        # 断言方差的区间值与期望区间的接近程度，容忍误差 1e-9
        assert_allclose(var.minmax, (3.1767242068607087, 24.45910381334018),
                        rtol=1e-09)

        # 断言标准差统计值与期望值 2.9724954732045084 的接近程度，小数精度为 14 位
        assert_almost_equal(std.statistic, 2.9724954732045084, decimal=14)
        # 断言标准差的区间值与期望区间的接近程度，容忍误差 1e-14
        assert_allclose(std.minmax, (1.7823367265645145, 4.9456146050146312),
                        rtol=1e-14)

    def test_empty_input(self):
        # 断言对空列表输入调用 stats.bayes_mvs 函数会引发 ValueError 异常
        assert_raises(ValueError, stats.bayes_mvs, [])

    def test_result_attributes(self):
        # 创建一个数组 x，包含从 0 到 14 的整数
        x = np.arange(15)
        # 定义结果属性元组，包含 'statistic' 和 'minmax'
        attributes = ('statistic', 'minmax')
        # 调用 stats.bayes_mvs 函数计算数组 x 的贝叶斯估计结果
        res = stats.bayes_mvs(x)

        # 遍历结果，检查每个结果对象的属性是否包含 'statistic' 和 'minmax'
        for i in res:
            check_named_results(i, attributes)
class TestMvsdist:
    # 定义测试类 TestMvsdist，用于测试 stats.mvsdist 函数

    def test_basic(self):
        # 定义测试方法 test_basic，测试基本功能
        data = [6, 9, 12, 7, 8, 8, 13]
        # 调用 stats.mvsdist 函数计算均值、方差、标准差
        mean, var, std = stats.mvsdist(data)

        # 断言均值的平均值接近于 9.0
        assert_almost_equal(mean.mean(), 9.0)
        
        # 断言均值的置信区间（90%）接近于 (7.103650222492964, 10.896349777507034)，相对误差限制在 1e-14
        assert_allclose(mean.interval(0.9), (7.103650222492964,
                                             10.896349777507034), rtol=1e-14)

        # 断言方差的平均值接近于 10.0
        assert_almost_equal(var.mean(), 10.0)
        
        # 断言方差的置信区间（90%）接近于 (3.1767242068607087, 24.45910381334018)，相对误差限制在 1e-09
        assert_allclose(var.interval(0.9), (3.1767242068607087,
                                            24.45910381334018), rtol=1e-09)

        # 断言标准差的平均值接近于 2.9724954732045084，小数精度为 14
        assert_almost_equal(std.mean(), 2.9724954732045084, decimal=14)
        
        # 断言标准差的置信区间（90%）接近于 (1.7823367265645145, 4.9456146050146312)，相对误差限制在 1e-14
        assert_allclose(std.interval(0.9), (1.7823367265645145,
                                            4.9456146050146312), rtol=1e-14)

    def test_empty_input(self):
        # 定义测试方法 test_empty_input，测试空输入情况
        # 断言调用 stats.mvsdist 函数时会引发 ValueError 异常，因为输入数据为空列表
        assert_raises(ValueError, stats.mvsdist, [])

    def test_bad_arg(self):
        # 定义测试方法 test_bad_arg，测试参数错误的情况
        # 当数据点少于两个时，断言调用 stats.mvsdist 函数会引发 ValueError 异常
        data = [1]
        assert_raises(ValueError, stats.mvsdist, data)

    def test_warns(self):
        # 定义测试方法 test_warns，测试警告
        # 回归测试 gh-5270，确保没有不必要的除零警告
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            # 调用 stats.mvsdist 函数计算各数据的均值，希望不会引发 RuntimeWarning
            [x.mean() for x in stats.mvsdist([1, 2, 3])]
            [x.mean() for x in stats.mvsdist([1, 2, 3, 4, 5])]


class TestShapiro:
    # 定义测试类 TestShapiro，用于测试 Shapiro-Wilk 正态性检验
    # 定义测试函数 test_basic，用于测试 Shapiro-Wilk 正态性检验
    def test_basic(self):
        # 第一个数据集 x1
        x1 = [0.11, 7.87, 4.61, 10.14, 7.95, 3.14, 0.46,
              4.43, 0.21, 4.75, 0.71, 1.52, 3.24,
              0.93, 0.42, 4.97, 9.53, 4.55, 0.47, 6.66]
        # 进行 Shapiro-Wilk 正态性检验，返回统计量 W 和 p 值
        w, pw = stats.shapiro(x1)
        # 用于断言检验结果的统计量 W 是否接近预期值，保留小数点后 6 位
        assert_almost_equal(w, 0.90047299861907959, decimal=6)
        # 同上，断言检验结果中的统计量（通过 shapiro_test 对象）是否接近预期值
        assert_almost_equal(shapiro_test.statistic, 0.90047299861907959, decimal=6)
        # 断言 p 值是否接近预期值，保留小数点后 6 位
        assert_almost_equal(pw, 0.042089745402336121, decimal=6)
        # 同上，断言检验结果中的 p 值是否接近预期值
        assert_almost_equal(shapiro_test.pvalue, 0.042089745402336121, decimal=6)

        # 第二个数据集 x2
        x2 = [1.36, 1.14, 2.92, 2.55, 1.46, 1.06, 5.27, -1.11,
              3.48, 1.10, 0.88, -0.51, 1.46, 0.52, 6.20, 1.69,
              0.08, 3.67, 2.81, 3.49]
        # 进行 Shapiro-Wilk 正态性检验，返回统计量 W 和 p 值
        w, pw = stats.shapiro(x2)
        # 断言检验结果的统计量 W 是否接近预期值，保留小数点后 6 位
        assert_almost_equal(w, 0.9590270, decimal=6)
        # 同上，断言检验结果中的统计量是否接近预期值
        assert_almost_equal(shapiro_test.statistic, 0.9590270, decimal=6)
        # 断言 p 值是否接近预期值，保留小数点后 3 位
        assert_almost_equal(pw, 0.52460, decimal=3)
        # 同上，断言检验结果中的 p 值是否接近预期值
        assert_almost_equal(shapiro_test.pvalue, 0.52460, decimal=3)

        # 第三个数据集 x3，使用正态分布生成
        x3 = stats.norm.rvs(loc=5, scale=3, size=100, random_state=12345678)
        # 进行 Shapiro-Wilk 正态性检验，返回统计量 W 和 p 值
        w, pw = stats.shapiro(x3)
        # 断言检验结果的统计量 W 是否接近预期值，保留小数点后 6 位
        assert_almost_equal(w, 0.9772805571556091, decimal=6)
        # 同上，断言检验结果中的统计量是否接近预期值
        assert_almost_equal(shapiro_test.statistic, 0.9772805571556091, decimal=6)
        # 断言 p 值是否接近预期值，保留小数点后 3 位
        assert_almost_equal(pw, 0.08144091814756393, decimal=3)
        # 同上，断言检验结果中的 p 值是否接近预期值
        assert_almost_equal(shapiro_test.pvalue, 0.08144091814756393, decimal=3)

        # 第四个数据集 x4，来自原始论文
        x4 = [0.139, 0.157, 0.175, 0.256, 0.344, 0.413, 0.503, 0.577, 0.614,
              0.655, 0.954, 1.392, 1.557, 1.648, 1.690, 1.994, 2.174, 2.206,
              3.245, 3.510, 3.571, 4.354, 4.980, 6.084, 8.351]
        # 预期的统计量 W 和 p 值
        W_expected = 0.83467
        p_expected = 0.000914
        # 进行 Shapiro-Wilk 正态性检验，返回统计量 W 和 p 值
        w, pw = stats.shapiro(x4)
        # 断言检验结果的统计量 W 是否接近预期值，保留小数点后 4 位
        assert_almost_equal(w, W_expected, decimal=4)
        # 同上，断言检验结果中的统计量是否接近预期值
        assert_almost_equal(shapiro_test.statistic, W_expected, decimal=4)
        # 断言 p 值是否接近预期值，保留小数点后 5 位
        assert_almost_equal(pw, p_expected, decimal=5)
        # 同上，断言检验结果中的 p 值是否接近预期值
        assert_almost_equal(shapiro_test.pvalue, p_expected, decimal=5)
    # 定义一个测试方法，用于测试二维数据的 Shapiro-Wilk 正态性检验
    def test_2d(self):
        # 第一个二维数据集
        x1 = [[0.11, 7.87, 4.61, 10.14, 7.95, 3.14, 0.46,
               4.43, 0.21, 4.75], [0.71, 1.52, 3.24,
               0.93, 0.42, 4.97, 9.53, 4.55, 0.47, 6.66]]
        # 执行 Shapiro-Wilk 正态性检验，返回统计量 w 和 p 值 pw
        w, pw = stats.shapiro(x1)
        # 再次执行 Shapiro-Wilk 正态性检验，保存结果到 shapiro_test
        shapiro_test = stats.shapiro(x1)
        # 断言检验的统计量 w 接近预期值 0.90047299861907959
        assert_almost_equal(w, 0.90047299861907959, decimal=6)
        # 断言 shapiro_test 的统计量 statistic 与预期值接近
        assert_almost_equal(shapiro_test.statistic, 0.90047299861907959, decimal=6)
        # 断言检验的 p 值 pw 接近预期值 0.042089745402336121
        assert_almost_equal(pw, 0.042089745402336121, decimal=6)
        # 断言 shapiro_test 的 p 值接近预期值
        assert_almost_equal(shapiro_test.pvalue, 0.042089745402336121, decimal=6)

        # 第二个二维数据集
        x2 = [[1.36, 1.14, 2.92, 2.55, 1.46, 1.06, 5.27, -1.11,
               3.48, 1.10], [0.88, -0.51, 1.46, 0.52, 6.20, 1.69,
               0.08, 3.67, 2.81, 3.49]]
        # 执行 Shapiro-Wilk 正态性检验，返回统计量 w 和 p 值 pw
        w, pw = stats.shapiro(x2)
        # 再次执行 Shapiro-Wilk 正态性检验，保存结果到 shapiro_test
        shapiro_test = stats.shapiro(x2)
        # 断言检验的统计量 w 接近预期值 0.9590270
        assert_almost_equal(w, 0.9590270, decimal=6)
        # 断言 shapiro_test 的统计量 statistic 与预期值接近
        assert_almost_equal(shapiro_test.statistic, 0.9590270, decimal=6)
        # 断言检验的 p 值 pw 接近预期值 0.52460
        assert_almost_equal(pw, 0.52460, decimal=3)
        # 断言 shapiro_test 的 p 值接近预期值
        assert_almost_equal(shapiro_test.pvalue, 0.52460, decimal=3)

    # 参数化测试方法，用于测试不足的样本数据情况
    @pytest.mark.parametrize('x', ([], [1], [1, 2]))
    def test_not_enough_values(self, x):
        # 使用 pytest 的 warn 断言，检测是否发出了 SmallSampleWarning 警告
        with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
            # 执行 Shapiro-Wilk 正态性检验，预期返回 statistic 和 pvalue 均为 NaN
            res = stats.shapiro(x)
            # 断言返回的 statistic 为 NaN
            assert_equal(res.statistic, np.nan)
            # 断言返回的 pvalue 为 NaN
            assert_equal(res.pvalue, np.nan)

    # 测试方法，用于测试包含 NaN 输入的情况
    def test_nan_input(self):
        # 创建一个包含 NaN 的数组
        x = np.arange(10.)
        x[9] = np.nan

        # 执行 Shapiro-Wilk 正态性检验，返回统计量 w 和 p 值 pw
        w, pw = stats.shapiro(x)
        # 再次执行 Shapiro-Wilk 正态性检验，保存结果到 shapiro_test
        shapiro_test = stats.shapiro(x)
        # 断言返回的统计量 w 为 NaN
        assert_equal(w, np.nan)
        # 断言 shapiro_test 的统计量 statistic 为 NaN
        assert_equal(shapiro_test.statistic, np.nan)
        # 由于统计量为 NaN，p 值也应为 NaN
        assert_almost_equal(pw, np.nan)
        # 断言 shapiro_test 的 p 值为 NaN
        assert_almost_equal(shapiro_test.pvalue, np.nan)

    # 测试方法，用于验证 GitHub 问题 #14462 的情况
    def test_gh14462(self):
        # 执行 Box-Cox 变换，并获取转换后的值和最大对数似然值
        trans_val, maxlog = stats.boxcox([122500, 474400, 110400])
        # 执行 Shapiro-Wilk 正态性检验，返回统计量和 p 值
        res = stats.shapiro(trans_val)

        # 参考值来自 R 语言的执行结果
        ref = (0.86468431705371, 0.2805581751566)
        # 使用 assert_allclose 检查实际结果与参考值的接近程度
        assert_allclose(res, ref, rtol=1e-5)
    # 定义一个测试方法，用于验证在长度为3的输入情况下，gh-18322 报告的问题，即 p-value 可能为负值。
    def test_length_3_gh18322(self):
        # 调用 stats 模块的 shapiro 函数，传入长度为3的数值列表，返回统计结果
        res = stats.shapiro([0.6931471805599453, 0.0, 0.0])
        # 断言检查返回的 p-value 是否大于等于 0
        assert res.pvalue >= 0

        # 对于输入为 [-0.7746653110021126, -0.4344432067942129, 1.8157053280290931] 的情况，
        # R 的 shapiro.test 函数未能准确计算 p-value。检查 stats.shapiro 函数使用的公式是否正确。
        # 下面是 R 中的相关代码，用于验证这一点：
        # options(digits=16)
        # x = c(-0.7746653110021126, -0.4344432067942129, 1.8157053280290931)
        # shapiro.test(x)
        x = [-0.7746653110021126, -0.4344432067942129, 1.8157053280290931]
        # 调用 stats 模块的 shapiro 函数，传入新的数值列表 x，返回统计结果
        res = stats.shapiro(x)
        # 断言检查返回的统计量 statistic 是否接近于预期值 0.84658770645509
        assert_allclose(res.statistic, 0.84658770645509)
        # 断言检查返回的 p-value 是否在接受范围内，使用相对误差 1e-6 进行比较
        assert_allclose(res.pvalue, 0.2313666489882, rtol=1e-6)
class TestAnderson:
    # 测试正常情况下的 Anderson-Darling 检验
    def test_normal(self):
        # 使用种子 1234567890 初始化随机数生成器
        rs = RandomState(1234567890)
        # 生成两个服从指数分布和正态分布的随机数样本
        x1 = rs.standard_exponential(size=50)
        x2 = rs.standard_normal(size=50)
        # 进行 Anderson-Darling 检验，返回统计量 A、临界值 crit 和显著性 sig
        A, crit, sig = stats.anderson(x1)
        # 断言检验统计量 A 应小于所有临界值 crit 中除最后一个之外的值
        assert_array_less(crit[:-1], A)
        A, crit, sig = stats.anderson(x2)
        # 断言检验统计量 A 应小于 crit 中倒数第二个和最后一个临界值之间的所有值
        assert_array_less(A, crit[-2:])

        # 创建一个全为 1 的数组，并将第一个元素设为 0
        v = np.ones(10)
        v[0] = 0
        # 进行 Anderson-Darling 检验，返回统计量 A、临界值 crit 和显著性 sig
        A, crit, sig = stats.anderson(v)
        # 断言检验统计量 A 应接近预期值 3.208057
        # 预期值的计算可以独立于 scipy 进行，例如在 R 中的计算结果如下：
        #   > library(nortest)
        #   > v <- rep(1, 10)
        #   > v[1] <- 0
        #   > result <- ad.test(v)
        #   > result$statistic
        #          A
        #   3.208057
        assert_allclose(A, 3.208057)

    # 测试指数分布的 Anderson-Darling 检验
    def test_expon(self):
        rs = RandomState(1234567890)
        x1 = rs.standard_exponential(size=50)
        x2 = rs.standard_normal(size=50)
        # 进行 Anderson-Darling 检验，指定分布类型为指数分布
        A, crit, sig = stats.anderson(x1, 'expon')
        # 断言检验统计量 A 应小于 crit 中倒数第二个和最后一个临界值之间的所有值
        assert_array_less(A, crit[-2:])
        # 忽略所有的运行时警告
        with np.errstate(all='ignore'):
            A, crit, sig = stats.anderson(x2, 'expon')
        # 断言检验统计量 A 应大于最后一个临界值 crit[-1]
        assert_(A > crit[-1])

    # 测试 Gumbel 分布的 Anderson-Darling 检验
    def test_gumbel(self):
        # 回归测试 gh-6306，修复该问题前，此案例可能会返回 a2=inf
        v = np.ones(100)
        v[0] = 0.0
        # 进行 Gumbel 分布的 Anderson-Darling 检验
        a2, crit, sig = stats.anderson(v, 'gumbel')
        # 对统计量的简要重新计算
        n = len(v)
        xbar, s = stats.gumbel_l.fit(v)
        logcdf = stats.gumbel_l.logcdf(v, xbar, s)
        logsf = stats.gumbel_l.logsf(v, xbar, s)
        i = np.arange(1, n+1)
        expected_a2 = -n - np.mean((2*i - 1) * (logcdf + logsf[::-1]))
        # 断言检验统计量 a2 应接近预期值 expected_a2
        assert_allclose(a2, expected_a2)

    # 测试异常参数的 Anderson-Darling 检验
    def test_bad_arg(self):
        # 断言 ValueError 异常应被触发，因为参数 dist='plate_of_shrimp' 无效
        assert_raises(ValueError, stats.anderson, [1], dist='plate_of_shrimp')

    # 测试 Anderson-Darling 检验结果的属性
    def test_result_attributes(self):
        rs = RandomState(1234567890)
        x = rs.standard_exponential(size=50)
        # 进行 Anderson-Darling 检验，返回结果对象 res
        res = stats.anderson(x)
        # 检查结果对象 res 的属性是否包含 'statistic'、'critical_values' 和 'significance_level'
        attributes = ('statistic', 'critical_values', 'significance_level')
        check_named_results(res, attributes)

    # 测试 Gumbel 分布类型参数 'gumbel_r' 和 'gumbel_l' 的有效性
    def test_gumbel_l(self):
        # 回归测试 gh-2592, gh-6337
        rs = RandomState(1234567890)
        x = rs.gumbel(size=100)
        # 分别进行 'gumbel' 和 'gumbel_l' 类型的 Anderson-Darling 检验
        A1, crit1, sig1 = stats.anderson(x, 'gumbel')
        A2, crit2, sig2 = stats.anderson(x, 'gumbel_l')
        # 断言两次检验结果 A2 应接近 A1
        assert_allclose(A2, A1)
    def test_gumbel_r(self):
        # gh-2592, gh-6337
        # 添加对 'gumbel_r' 和 'gumbel_l' 作为有效输入的支持。
        # 使用种子 1234567890 初始化随机数生成器
        rs = RandomState(1234567890)
        # 生成服从 Gumbel 分布的随机数数组
        x1 = rs.gumbel(size=100)
        # 创建一个包含 100 个元素且全部为 1 的数组
        x2 = np.ones(100)
        # 由于常量数组是一个特殊情况，会导致 gumbel_r.fit 出错，因此修改 x2 中的一个值
        x2[0] = 0.996
        # 计算 x1 的 Anderson-Darling 统计量及关键值
        A1, crit1, sig1 = stats.anderson(x1, 'gumbel_r')
        # 计算 x2 的 Anderson-Darling 统计量及关键值
        A2, crit2, sig2 = stats.anderson(x2, 'gumbel_r')

        # 断言：A1 应小于 crit1 的倒数第二个和最后一个值
        assert_array_less(A1, crit1[-2:])
        # 断言：A2 应大于 crit2 的最后一个值
        assert_(A2 > crit2[-1])

    def test_weibull_min_case_A(self):
        # 使用 `anderson` 参考文献 [7] 中的数据和参考值
        x = np.array([225, 171, 198, 189, 189, 135, 162, 135, 117, 162])
        # 对 x 进行 Weibull 最小值分布的 Anderson-Darling 测试
        res = stats.anderson(x, 'weibull_min')
        # 提取拟合结果的参数
        m, loc, scale = res.fit_result.params
        # 断言：拟合结果 (m, loc, scale) 应接近 (2.38, 99.02, 78.23)，相对误差不超过 2e-3
        assert_allclose((m, loc, scale), (2.38, 99.02, 78.23), rtol=2e-3)
        # 断言：统计量应接近 0.260，相对误差不超过 1e-3
        assert_allclose(res.statistic, 0.260, rtol=1e-3)
        # 断言：统计量应小于临界值数组的第一个值
        assert res.statistic < res.critical_values[0]

        # 计算 Weibull 分布的形状参数 c
        c = 1 / m  # 约为 0.42
        # 断言：c 应接近 1/2.38，相对误差不超过 2e-3
        assert_allclose(c, 1/2.38, rtol=2e-3)
        # 在 c=0.4 和 c=0.45 之间线性插值，对应 _Avals_weibull 中的 -3 和 -2 索引
        As40 = _Avals_weibull[-3]
        As45 = _Avals_weibull[-2]
        As_ref = As40 + (c - 0.4)/(0.45 - 0.4) * (As45 - As40)
        # 断言：所有临界值应大于插值计算得到的 As_ref，精度误差不超过 1e-3
        assert np.all(res.critical_values > As_ref)
        # 断言：临界值数组应与插值计算得到的 As_ref 接近，绝对误差不超过 1e-3
        assert_allclose(res.critical_values, As_ref, atol=1e-3)

    def test_weibull_min_case_B(self):
        # 来自 `anderson` 参考文献 [7]
        x = np.array([74, 57, 48, 29, 502, 12, 70, 21,
                      29, 386, 59, 27, 153, 26, 326])
        # 断言：当拟合过程收敛到异常值时，应抛出 ValueError 异常，且异常消息应包含指定的信息
        message = "Maximum likelihood estimation has converged to "
        with pytest.raises(ValueError, match=message):
            stats.anderson(x, 'weibull_min')

    def test_weibull_warning_error(self):
        # 检查当观测值过少时是否会出现警告消息
        # 同时，这也是拟合过程中可能出现错误的示例
        x = -np.array([225, 75, 57, 168, 107, 12, 61, 43, 29])
        # 断言：应触发 UserWarning 警告，且警告消息应包含指定的信息
        wmessage = "Critical values of the test statistic are given for the..."
        # 断言：应触发 ValueError 异常，且异常消息应包含指定的信息
        emessage = "An error occurred while fitting the Weibull distribution..."
        wcontext = pytest.warns(UserWarning, match=wmessage)
        econtext = pytest.raises(ValueError, match=emessage)
        # 在警告和异常上下文中执行 Anderson-Darling 测试
        with wcontext, econtext:
            stats.anderson(x, 'weibull_min')

    @pytest.mark.parametrize('distname',
                             ['norm', 'expon', 'gumbel_l', 'extreme1',
                              'gumbel', 'gumbel_r', 'logistic', 'weibull_min'])
    # 测试 anderson 函数对于指定分布名是否返回 FitResult 对象
    def test_anderson_fit_params(self, distname):
        # 使用指定的随机数生成器创建随机数生成器对象
        rng = np.random.default_rng(330691555377792039)
        # 如果 distname 在 {'extreme1', 'gumbel'} 中，则使用 'gumbel_l' 作为真实分布名，否则使用 distname
        real_distname = ('gumbel_l' if distname in {'extreme1', 'gumbel'}
                         else distname)
        # 根据真实的分布名获取对应的概率分布对象
        dist = getattr(stats, real_distname)
        # 从 distcont 字典中获取真实分布名对应的参数
        params = distcont[real_distname]
        # 从指定分布中生成随机样本
        x = dist.rvs(*params, size=1000, random_state=rng)
        # 对生成的随机样本进行 anderson 检验
        res = stats.anderson(x, distname)
        # 断言返回的 FitResult 对象的 success 属性为真
        assert res.fit_result.success

    # 测试 _get_As_weibull 函数对于不同参数值返回的结果是否正确
    def test_anderson_weibull_As(self):
        m = 1  # 当 m < 2 时，使用特定的 _Avals_weibull 最后一个元素
        assert_equal(_get_As_weibull(1/m), _Avals_weibull[-1])
        m = np.inf  # 当 m 为无穷大时，使用特定的 _Avals_weibull 第一个元素
        assert_equal(_get_As_weibull(1/m), _Avals_weibull[0])
class TestAndersonKSamp:
    def test_example1a(self):
        # Example data from Scholz & Stephens (1987), originally
        # published in Lehmann (1995, Nonparametrics, Statistical
        # Methods Based on Ranks, p. 309)
        # Pass a mixture of lists and arrays
        t1 = [38.7, 41.5, 43.8, 44.5, 45.5, 46.0, 47.7, 58.0]
        t2 = np.array([39.2, 39.3, 39.7, 41.4, 41.8, 42.9, 43.3, 45.8])
        t3 = np.array([34.0, 35.0, 39.0, 40.0, 43.0, 43.0, 44.0, 45.0])
        t4 = np.array([34.0, 34.8, 34.8, 35.4, 37.2, 37.8, 41.2, 42.8])

        # Perform Anderson-Darling k-sample test on the data
        Tk, tm, p = stats.anderson_ksamp((t1, t2, t3, t4), midrank=False)

        # Check the computed test statistic Tk against the expected value
        assert_almost_equal(Tk, 4.449, 3)
        # Check the first 5 elements of the computed critical values tm
        assert_array_almost_equal([0.4985, 1.3237, 1.9158, 2.4930, 3.2459],
                                  tm[0:5], 4)
        # Check the computed p-value p against the expected value with a tolerance
        assert_allclose(p, 0.0021, atol=0.00025)

    def test_example1b(self):
        # Example data from Scholz & Stephens (1987), originally
        # published in Lehmann (1995, Nonparametrics, Statistical
        # Methods Based on Ranks, p. 309)
        # Pass arrays
        t1 = np.array([38.7, 41.5, 43.8, 44.5, 45.5, 46.0, 47.7, 58.0])
        t2 = np.array([39.2, 39.3, 39.7, 41.4, 41.8, 42.9, 43.3, 45.8])
        t3 = np.array([34.0, 35.0, 39.0, 40.0, 43.0, 43.0, 44.0, 45.0])
        t4 = np.array([34.0, 34.8, 34.8, 35.4, 37.2, 37.8, 41.2, 42.8])

        # Perform Anderson-Darling k-sample test on the data with midrank=True
        Tk, tm, p = stats.anderson_ksamp((t1, t2, t3, t4), midrank=True)

        # Check the computed test statistic Tk against the expected value
        assert_almost_equal(Tk, 4.480, 3)
        # Check the first 5 elements of the computed critical values tm
        assert_array_almost_equal([0.4985, 1.3237, 1.9158, 2.4930, 3.2459],
                                  tm[0:5], 4)
        # Check the computed p-value p against the expected value with a tolerance
        assert_allclose(p, 0.0020, atol=0.00025)

    @pytest.mark.xslow
    # 定义一个测试方法，用于测试 anderson_ksamp 函数的功能
    def test_example2a(self):
        # Example data taken from an earlier technical report of
        # Scholz and Stephens
        # Pass lists instead of arrays
        # 准备多个样本数据，每个样本是一个整数列表
        t1 = [194, 15, 41, 29, 33, 181]
        t2 = [413, 14, 58, 37, 100, 65, 9, 169, 447, 184, 36, 201, 118]
        t3 = [34, 31, 18, 18, 67, 57, 62, 7, 22, 34]
        t4 = [90, 10, 60, 186, 61, 49, 14, 24, 56, 20, 79, 84, 44, 59, 29,
              118, 25, 156, 310, 76, 26, 44, 23, 62]
        t5 = [130, 208, 70, 101, 208]
        t6 = [74, 57, 48, 29, 502, 12, 70, 21, 29, 386, 59, 27]
        t7 = [55, 320, 56, 104, 220, 239, 47, 246, 176, 182, 33]
        t8 = [23, 261, 87, 7, 120, 14, 62, 47, 225, 71, 246, 21, 42, 20, 5,
              12, 120, 11, 3, 14, 71, 11, 14, 11, 16, 90, 1, 16, 52, 95]
        t9 = [97, 51, 11, 4, 141, 18, 142, 68, 77, 80, 1, 16, 106, 206, 82,
              54, 31, 216, 46, 111, 39, 63, 18, 191, 18, 163, 24]
        t10 = [50, 44, 102, 72, 22, 39, 3, 15, 197, 188, 79, 88, 46, 5, 5, 36,
               22, 139, 210, 97, 30, 23, 13, 14]
        t11 = [359, 9, 12, 270, 603, 3, 104, 2, 438]
        t12 = [50, 254, 5, 283, 35, 12]
        t13 = [487, 18, 100, 7, 98, 5, 85, 91, 43, 230, 3, 130]
        t14 = [102, 209, 14, 57, 54, 32, 67, 59, 134, 152, 27, 14, 230, 66,
               61, 34]

        # 将所有样本数据组成一个元组
        samples = (t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14)
        
        # 调用 anderson_ksamp 函数进行多样本检验，midrank 参数设置为 False
        Tk, tm, p = stats.anderson_ksamp(samples, midrank=False)
        
        # 断言 Tk 的值约等于 3.288，允许误差为 3
        assert_almost_equal(Tk, 3.288, 3)
        
        # 断言 tm 数组的前五个元素约等于 [0.5990, 1.3269, 1.8052, 2.2486, 2.8009]
        assert_array_almost_equal([0.5990, 1.3269, 1.8052, 2.2486, 2.8009],
                                  tm[0:5], 4)
        
        # 断言 p 值约等于 0.0041，允许的绝对误差为 0.00025
        assert_allclose(p, 0.0041, atol=0.00025)

        # 使用指定种子创建一个随机数生成器对象 rng
        rng = np.random.default_rng(6989860141921615054)
        
        # 使用 PermutationMethod 类创建一个 method 对象，设置重采样次数为 9999，随机数种子为 rng
        method = stats.PermutationMethod(n_resamples=9999, random_state=rng)
        
        # 再次调用 anderson_ksamp 函数，传入 samples、midrank 参数为 False 和 method 参数
        res = stats.anderson_ksamp(samples, midrank=False, method=method)
        
        # 断言 res 对象的 statistic 属性等于 Tk
        assert_array_equal(res.statistic, Tk)
        
        # 断言 res 对象的 critical_values 属性等于 tm
        assert_array_equal(res.critical_values, tm)
        
        # 断言 res 对象的 pvalue 属性约等于 p，允许的绝对误差为 6e-4
        assert_allclose(res.pvalue, p, atol=6e-4)
    # 测试示例2b：检查 anderson_ksamp 函数的输出是否符合预期

    def test_example2b(self):
        # Example data taken from an earlier technical report of
        # Scholz and Stephens
        # 定义多个示例数据集
        t1 = [194, 15, 41, 29, 33, 181]
        t2 = [413, 14, 58, 37, 100, 65, 9, 169, 447, 184, 36, 201, 118]
        t3 = [34, 31, 18, 18, 67, 57, 62, 7, 22, 34]
        t4 = [90, 10, 60, 186, 61, 49, 14, 24, 56, 20, 79, 84, 44, 59, 29,
              118, 25, 156, 310, 76, 26, 44, 23, 62]
        t5 = [130, 208, 70, 101, 208]
        t6 = [74, 57, 48, 29, 502, 12, 70, 21, 29, 386, 59, 27]
        t7 = [55, 320, 56, 104, 220, 239, 47, 246, 176, 182, 33]
        t8 = [23, 261, 87, 7, 120, 14, 62, 47, 225, 71, 246, 21, 42, 20, 5,
              12, 120, 11, 3, 14, 71, 11, 14, 11, 16, 90, 1, 16, 52, 95]
        t9 = [97, 51, 11, 4, 141, 18, 142, 68, 77, 80, 1, 16, 106, 206, 82,
              54, 31, 216, 46, 111, 39, 63, 18, 191, 18, 163, 24]
        t10 = [50, 44, 102, 72, 22, 39, 3, 15, 197, 188, 79, 88, 46, 5, 5, 36,
               22, 139, 210, 97, 30, 23, 13, 14]
        t11 = [359, 9, 12, 270, 603, 3, 104, 2, 438]
        t12 = [50, 254, 5, 283, 35, 12]
        t13 = [487, 18, 100, 7, 98, 5, 85, 91, 43, 230, 3, 130]
        t14 = [102, 209, 14, 57, 54, 32, 67, 59, 134, 152, 27, 14, 230, 66,
               61, 34]

        # 调用 anderson_ksamp 函数计算统计量 Tk, 检验量 tm 和 p 值
        Tk, tm, p = stats.anderson_ksamp((t1, t2, t3, t4, t5, t6, t7, t8,
                                          t9, t10, t11, t12, t13, t14),
                                         midrank=True)

        # 断言检查统计量 Tk 的近似值是否在容忍误差范围内
        assert_almost_equal(Tk, 3.294, 3)
        
        # 断言检查检验量 tm 的前五个元素是否与预期数组几乎相等
        assert_array_almost_equal([0.5990, 1.3269, 1.8052, 2.2486, 2.8009],
                                  tm[0:5], 4)
        
        # 断言检查 p 值是否在容忍误差范围内
        assert_allclose(p, 0.0041, atol=0.00025)

    # 测试不足样本量时是否能引发 ValueError 异常
    def test_not_enough_samples(self):
        assert_raises(ValueError, stats.anderson_ksamp, np.ones(5))

    # 测试没有不同观察值时是否能引发 ValueError 异常
    def test_no_distinct_observations(self):
        assert_raises(ValueError, stats.anderson_ksamp,
                      (np.ones(5), np.ones(5)))

    # 测试空样本时是否能引发 ValueError 异常
    def test_empty_sample(self):
        assert_raises(ValueError, stats.anderson_ksamp, (np.ones(5), []))

    # 测试结果的属性是否如预期设置
    def test_result_attributes(self):
        # Pass a mixture of lists and arrays
        # 定义包含列表和数组的混合样本
        t1 = [38.7, 41.5, 43.8, 44.5, 45.5, 46.0, 47.7, 58.0]
        t2 = np.array([39.2, 39.3, 39.7, 41.4, 41.8, 42.9, 43.3, 45.8])
        
        # 调用 anderson_ksamp 函数计算统计量，不使用 midrank 参数
        res = stats.anderson_ksamp((t1, t2), midrank=False)

        # 定义要检查的结果属性列表
        attributes = ('statistic', 'critical_values', 'significance_level')
        
        # 调用自定义函数 check_named_results 检查结果对象是否具有指定的属性
        check_named_results(res, attributes)

        # 断言检查结果对象的 significance_level 属性是否等于其 p 值
        assert_equal(res.significance_level, res.pvalue)
class TestAnsari:

    def test_small(self):
        x = [1, 2, 3, 3, 4]
        y = [3, 2, 6, 1, 6, 1, 4, 1]
        # 使用 suppress_warnings 上下文管理器，忽略特定警告
        with suppress_warnings() as sup:
            # 过滤特定类型的警告消息
            sup.filter(UserWarning, "Ties preclude use of exact statistic.")
            # 调用 stats.ansari 计算 Ansari-Bradley 检验的统计量 W 和 p 值
            W, pval = stats.ansari(x, y)
        # 断言 W 的值接近于期望值 23.5，精确到小数点后 11 位
        assert_almost_equal(W, 23.5, 11)
        # 断言 pval 的值接近于期望值 0.13499256881897437，精确到小数点后 11 位
        assert_almost_equal(pval, 0.13499256881897437, 11)

    def test_approx(self):
        ramsay = np.array((111, 107, 100, 99, 102, 106, 109, 108, 104, 99,
                           101, 96, 97, 102, 107, 113, 116, 113, 110, 98))
        parekh = np.array((107, 108, 106, 98, 105, 103, 110, 105, 104,
                           100, 96, 108, 103, 104, 114, 114, 113, 108,
                           106, 99))
        # 使用 suppress_warnings 上下文管理器，忽略特定警告
        with suppress_warnings() as sup:
            # 过滤特定类型的警告消息
            sup.filter(UserWarning, "Ties preclude use of exact statistic.")
            # 调用 stats.ansari 计算 Ansari-Bradley 检验的统计量 W 和 p 值
            W, pval = stats.ansari(ramsay, parekh)
        # 断言 W 的值接近于期望值 185.5，精确到小数点后 11 位
        assert_almost_equal(W, 185.5, 11)
        # 断言 pval 的值接近于期望值 0.18145819972867083，精确到小数点后 11 位
        assert_almost_equal(pval, 0.18145819972867083, 11)

    def test_exact(self):
        # 调用 stats.ansari 计算 Ansari-Bradley 检验的统计量 W 和 p 值
        W, pval = stats.ansari([1, 2, 3, 4], [15, 5, 20, 8, 10, 12])
        # 断言 W 的值接近于期望值 10.0，精确到小数点后 11 位
        assert_almost_equal(W, 10.0, 11)
        # 断言 pval 的值接近于期望值 0.533333333333333333，精确到小数点后 7 位
        assert_almost_equal(pval, 0.533333333333333333, 7)

    @pytest.mark.parametrize('args', [([], [1]), ([1], [])])
    def test_bad_arg(self, args):
        # 使用 pytest.warns 断言捕获 SmallSampleWarning 警告，并匹配 too_small_1d_not_omit 字符串
        with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
            # 调用 stats.ansari 并传入参数 args，预期返回 np.nan 的统计量和 p 值
            res = stats.ansari(*args)
            # 断言返回的统计量为 np.nan
            assert_equal(res.statistic, np.nan)
            # 断言返回的 p 值为 np.nan
            assert_equal(res.pvalue, np.nan)

    def test_result_attributes(self):
        x = [1, 2, 3, 3, 4]
        y = [3, 2, 6, 1, 6, 1, 4, 1]
        # 使用 suppress_warnings 上下文管理器，忽略特定警告
        with suppress_warnings() as sup:
            # 过滤特定类型的警告消息
            sup.filter(UserWarning, "Ties preclude use of exact statistic.")
            # 调用 stats.ansari 计算 Ansari-Bradley 检验的结果
            res = stats.ansari(x, y)
        # 定义需要检查的属性
        attributes = ('statistic', 'pvalue')
        # 使用 check_named_results 函数检查 res 对象的属性是否与 attributes 相符
        check_named_results(res, attributes)

    def test_bad_alternative(self):
        # 'alternative' 参数值无效时，应该引发 ValueError 异常
        x1 = [1, 2, 3, 4]
        x2 = [5, 6, 7, 8]
        match = "'alternative' must be 'two-sided'"
        # 使用 assert_raises 断言捕获 ValueError 异常，并匹配特定错误信息 match
        with assert_raises(ValueError, match=match):
            # 调用 stats.ansari，并传入无效的 alternative 参数值 'foo'
            stats.ansari(x1, x2, alternative='foo')
    def test_alternative_exact(self):
        x1 = [-5, 1, 5, 10, 15, 20, 25]  # 定义第一个样本数据，高比例，loc=10
        x2 = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]  # 定义第二个样本数据，低比例，loc=10
        
        # 调用 Ansari-Bradley 检验函数计算统计量和 p 值，默认 alternative='two-sided'
        statistic, pval = stats.ansari(x1, x2)
        
        # 指定 alternative='less' 计算 p 值
        pval_l = stats.ansari(x1, x2, alternative='less').pvalue
        
        # 指定 alternative='greater' 计算 p 值
        pval_g = stats.ansari(x1, x2, alternative='greater').pvalue
        
        # 断言两个单侧检验的结果
        assert pval_l > 0.95
        assert pval_g < 0.05  # 显著水平
        
        # 进一步检查 p 值是否满足一些条件
        prob = _abw_state.pmf(statistic, len(x1), len(x2))
        assert_allclose(pval_g + pval_l, 1 + prob, atol=1e-12)
        
        # 检查一侧 p 值是否等于双侧 p 值的一半，另一侧 p 值是否是其补集
        assert_allclose(pval_g, pval/2, atol=1e-12)
        assert_allclose(pval_l, 1 + prob - pval/2, atol=1e-12)
        
        # 确认交换 x 和 y 后结果是否反转
        pval_l_reverse = stats.ansari(x2, x1, alternative='less').pvalue
        pval_g_reverse = stats.ansari(x2, x1, alternative='greater').pvalue
        assert pval_l_reverse < 0.05
        assert pval_g_reverse > 0.95

    @pytest.mark.parametrize(
        'x, y, alternative, expected',
        # 测试设计确保在 Ansari-Bradley 测试的 exact 模式中涵盖 if-else 语句。
        [([1, 2, 3, 4], [5, 6, 7, 8], 'less', 0.6285714285714),
         ([1, 2, 3, 4], [5, 6, 7, 8], 'greater', 0.6285714285714),
         ([1, 2, 3], [4, 5, 6, 7, 8], 'less', 0.8928571428571),
         ([1, 2, 3], [4, 5, 6, 7, 8], 'greater', 0.2857142857143),
         ([1, 2, 3, 4, 5], [6, 7, 8], 'less', 0.2857142857143),
         ([1, 2, 3, 4, 5], [6, 7, 8], 'greater', 0.8928571428571)]
    )
    def test_alternative_exact_with_R(self, x, y, alternative, expected):
        # 在任意数据上使用 R 进行测试
        # 第三个测试案例使用的示例 R 代码:
        # ```R
        # > options(digits=16)
        # > x <- c(1,2,3)
        # > y <- c(4,5,6,7,8)
        # > ansari.test(x, y, alternative='less', exact=TRUE)
        #
        #     Ansari-Bradley test
        #
        # data:  x and y
        # AB = 6, p-value = 0.8928571428571
        # alternative hypothesis: true ratio of scales is less than 1
        #
        # ```
        
        # 使用给定的 alternative 参数计算 p 值并断言结果
        pval = stats.ansari(x, y, alternative=alternative).pvalue
        assert_allclose(pval, expected, atol=1e-12)
    `
        # 定义一个测试用例，用于检验替代近似方法
        def test_alternative_approx(self):
            # 生成两组服从正态分布的随机数，分别设置均值和方差
            x1 = stats.norm.rvs(0, 5, size=100, random_state=123)
            x2 = stats.norm.rvs(0, 2, size=100, random_state=123)
            
            # 当样本大小 m > 55 或 n > 55 时，测试应自动切换到近似方法
            pval_l = stats.ansari(x1, x2, alternative='less').pvalue
            pval_g = stats.ansari(x1, x2, alternative='greater').pvalue
            
            # 断言检查，验证近似结果与预期非常接近
            assert_allclose(pval_l, 1.0, atol=1e-12)
            assert_allclose(pval_g, 0.0, atol=1e-12)
            
            # 另外检查单侧 p 值是否等于双侧 p 值的一半，并且另一个单侧 p 值是其补集
            x1 = stats.norm.rvs(0, 2, size=60, random_state=123)
            x2 = stats.norm.rvs(0, 1.5, size=60, random_state=123)
            pval = stats.ansari(x1, x2).pvalue
            pval_l = stats.ansari(x1, x2, alternative='less').pvalue
            pval_g = stats.ansari(x1, x2, alternative='greater').pvalue
            
            # 断言检查，验证单侧 p 值是否符合预期
            assert_allclose(pval_g, pval/2, atol=1e-12)
            assert_allclose(pval_l, 1-pval/2, atol=1e-12)
@array_api_compatible
class TestBartlett:
    # Bartlett 检验的测试类，用于检验方差齐性的统计方法

    def test_data(self, xp):
        # 测试数据是否符合 Bartlett 检验的预期结果
        args = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]
        args = [xp.asarray(arg) for arg in args]  # 将每个参数转换为对应的数组表示
        T, pval = stats.bartlett(*args)  # 进行 Bartlett 检验，计算统计量 T 和 p 值
        xp_assert_close(T, xp.asarray(20.78587342806484))  # 断言 T 的计算结果与预期值相近
        xp_assert_close(pval, xp.asarray(0.0136358632781))  # 断言 p 值的计算结果与预期值相近

    def test_too_few_args(self, xp):
        # 测试当输入参数过少时是否能正确抛出 ValueError 异常
        message = "Must enter at least two input sample vectors."
        with pytest.raises(ValueError, match=message):
            stats.bartlett(xp.asarray([1.]))

    def test_result_attributes(self, xp):
        # 测试 Bartlett 检验返回结果的属性是否符合预期
        args = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]
        args = [xp.asarray(arg) for arg in args]  # 将每个参数转换为对应的数组表示
        res = stats.bartlett(*args)  # 进行 Bartlett 检验
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes, xp=xp)  # 检查返回结果是否包含指定的属性

    @pytest.mark.skip_xp_backends(
        "jax.numpy", cpu_only=True,
        reasons=['`var` incorrect when `correction > n` (google/jax#21330)'])
    @pytest.mark.usefixtures("skip_xp_backends")
    def test_empty_arg(self, xp):
        # 测试当输入参数中包含空数组时的行为
        args = (g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, [])
        args = [xp.asarray(arg) for arg in args]  # 将每个参数转换为对应的数组表示
        if is_numpy(xp):
            with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
                res = stats.bartlett(*args)  # 在 NumPy 下，测试是否会产生警告
        else:
            with np.testing.suppress_warnings() as sup:
                # 在非 NumPy 下，设置忽略特定警告
                sup.filter(RuntimeWarning, "invalid value encountered")
                sup.filter(UserWarning, r"var\(\): degrees of freedom is <= 0.")
                sup.filter(RuntimeWarning, "Degrees of freedom <= 0 for slice")
                res = stats.bartlett(*args)  # 执行 Bartlett 检验
        NaN = xp.asarray(xp.nan)
        xp_assert_equal(res.statistic, NaN)  # 断言统计量为 NaN
        xp_assert_equal(res.pvalue, NaN)  # 断言 p 值为 NaN


class TestLevene:

    def test_data(self):
        # 测试数据是否符合 Levene 检验的预期结果
        args = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]
        W, pval = stats.levene(*args)  # 进行 Levene 检验，计算统计量 W 和 p 值
        assert_almost_equal(W, 1.7059176930008939, 7)  # 断言 W 的计算结果与预期值相近
        assert_almost_equal(pval, 0.0990829755522, 7)  # 断言 p 值的计算结果与预期值相近

    def test_trimmed1(self):
        # 测试当 center='trimmed' 时，与 center='mean' 相比结果是否一致，proportiontocut=0
        W1, pval1 = stats.levene(g1, g2, g3, center='mean')
        W2, pval2 = stats.levene(g1, g2, g3, center='trimmed',
                                 proportiontocut=0.0)
        assert_almost_equal(W1, W2)  # 断言两种中心方法得到的 W 值相等
        assert_almost_equal(pval1, pval2)  # 断言两种中心方法得到的 p 值相等
    # 定义一个名为 test_trimmed2 的测试方法，使用 self 参数，表示这是一个类的方法
    def test_trimmed2(self):
        # 定义两个列表 x 和 y，包含浮点数数据
        x = [1.2, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0]
        y = [0.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 200.0]
        # 使用随机种子 1234 初始化 numpy 随机数生成器
        np.random.seed(1234)
        # 对列表 x 进行随机排列
        x2 = np.random.permutation(x)

        # 使用 center='trimmed' 参数调用 stats.levene 函数，计算 Levene 检验的统计量 W0 和 p 值 pval0
        W0, pval0 = stats.levene(x, y, center='trimmed', proportiontocut=0.125)
        # 使用 center='trimmed' 参数再次调用 stats.levene 函数，计算 Levene 检验的统计量 W1 和 p 值 pval1
        W1, pval1 = stats.levene(x2, y, center='trimmed', proportiontocut=0.125)
        # 对 x 和 y 列表的部分数据进行切片，然后使用 center='mean' 参数调用 stats.levene 函数
        # 计算 Levene 检验的统计量 W2 和 p 值 pval2
        W2, pval2 = stats.levene(x[1:-1], y[1:-1], center='mean')
        
        # 断言：W0 和 W2 应该近似相等
        assert_almost_equal(W0, W2)
        # 断言：W1 和 W2 应该近似相等
        assert_almost_equal(W1, W2)
        # 断言：pval1 和 pval2 应该近似相等
        assert_almost_equal(pval1, pval2)

    # 定义一个名为 test_equal_mean_median 的测试方法，使用 self 参数
    def test_equal_mean_median(self):
        # 创建一个包含等间隔数值的 numpy 数组 x
        x = np.linspace(-1, 1, 21)
        # 使用随机种子 1234 初始化 numpy 随机数生成器
        np.random.seed(1234)
        # 对数组 x 进行随机排列，得到数组 x2
        x2 = np.random.permutation(x)
        # 创建数组 y，其中元素为 x 中每个元素的立方
        y = x**3
        
        # 使用 center='mean' 参数调用 stats.levene 函数，计算 Levene 检验的统计量 W1 和 p 值 pval1
        W1, pval1 = stats.levene(x, y, center='mean')
        # 使用 center='median' 参数调用 stats.levene 函数，计算 Levene 检验的统计量 W2 和 p 值 pval2
        W2, pval2 = stats.levene(x2, y, center='median')
        
        # 断言：W1 和 W2 应该近似相等
        assert_almost_equal(W1, W2)
        # 断言：pval1 和 pval2 应该近似相等
        assert_almost_equal(pval1, pval2)

    # 定义一个名为 test_bad_keyword 的测试方法，使用 self 参数
    def test_bad_keyword(self):
        # 创建一个包含等间隔数值的 numpy 数组 x
        x = np.linspace(-1, 1, 21)
        # 断言：调用 stats.levene 函数时使用不存在的参数 portiontocut 会抛出 TypeError 异常
        assert_raises(TypeError, stats.levene, x, x, portiontocut=0.1)

    # 定义一个名为 test_bad_center_value 的测试方法，使用 self 参数
    def test_bad_center_value(self):
        # 创建一个包含等间隔数值的 numpy 数组 x
        x = np.linspace(-1, 1, 21)
        # 断言：调用 stats.levene 函数时使用不合法的 center='trim' 参数会抛出 ValueError 异常
        assert_raises(ValueError, stats.levene, x, x, center='trim')

    # 定义一个名为 test_too_few_args 的测试方法，使用 self 参数
    def test_too_few_args(self):
        # 断言：调用 stats.levene 函数时提供的参数列表中包含不足两个元素的列表 [1] 会抛出 ValueError 异常
        assert_raises(ValueError, stats.levene, [1])

    # 定义一个名为 test_result_attributes 的测试方法，使用 self 参数
    def test_result_attributes(self):
        # 创建一个名为 args 的列表，内容为 g1, g2, g3, ..., g10
        args = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]
        # 调用 stats.levene 函数，传递 args 列表中的元素作为参数，并获取返回值 res
        res = stats.levene(*args)
        # 定义一个元组 attributes，包含两个字符串 'statistic' 和 'pvalue'
        attributes = ('statistic', 'pvalue')
        # 调用 check_named_results 函数，验证 res 是否包含指定的属性
        check_named_results(res, attributes)

    # 定义一个名为 test_1d_input 的测试方法，使用 self 参数
    # 临时修复问题 #9252：仅接受 1 维输入
    def test_1d_input(self):
        # 创建一个二维 numpy 数组 x
        x = np.array([[1, 2], [3, 4]])
        # 断言：调用 stats.levene 函数时传递的参数 g1 和 x 不是 1 维数组，会抛出 ValueError 异常
        assert_raises(ValueError, stats.levene, g1, x)
class TestBinomTest:
    """Tests for stats.binomtest."""

    # Expected results here are from R binom.test, e.g.
    # options(digits=16)
    # binom.test(484, 967, p=0.48)
    #

    def test_two_sided_pvalues1(self):
        # `tol` could be stricter on most architectures, but the value
        # here is limited by accuracy of `binom.cdf` for large inputs on
        # Linux_Python_37_32bit_full and aarch64
        rtol = 1e-10  # aarch64 observed rtol: 1.5e-11
        # 调用 stats 模块中的 binomtest 函数计算二项分布测试的结果
        res = stats.binomtest(10079999, 21000000, 0.48)
        # 使用 assert_allclose 断言函数检查计算结果的 p-value 是否接近于 1.0
        assert_allclose(res.pvalue, 1.0, rtol=rtol)
        res = stats.binomtest(10079990, 21000000, 0.48)
        assert_allclose(res.pvalue, 0.9966892187965, rtol=rtol)
        res = stats.binomtest(10080009, 21000000, 0.48)
        assert_allclose(res.pvalue, 0.9970377203856, rtol=rtol)
        res = stats.binomtest(10080017, 21000000, 0.48)
        assert_allclose(res.pvalue, 0.9940754817328, rtol=1e-9)

    def test_two_sided_pvalues2(self):
        rtol = 1e-10  # no aarch64 failure with 1e-15, preemptive bump
        # 继续调用 stats 模块中的 binomtest 函数进行不同的二项分布测试
        res = stats.binomtest(9, n=21, p=0.48)
        assert_allclose(res.pvalue, 0.6689672431939, rtol=rtol)
        res = stats.binomtest(4, 21, 0.48)
        assert_allclose(res.pvalue, 0.008139563452106, rtol=rtol)
        res = stats.binomtest(11, 21, 0.48)
        assert_allclose(res.pvalue, 0.8278629664608, rtol=rtol)
        res = stats.binomtest(7, 21, 0.48)
        assert_allclose(res.pvalue, 0.1966772901718, rtol=rtol)
        res = stats.binomtest(3, 10, .5)
        assert_allclose(res.pvalue, 0.34375, rtol=rtol)
        res = stats.binomtest(2, 2, .4)
        assert_allclose(res.pvalue, 0.16, rtol=rtol)
        res = stats.binomtest(2, 4, .3)
        assert_allclose(res.pvalue, 0.5884, rtol=rtol)

    def test_edge_cases(self):
        rtol = 1e-10  # aarch64 observed rtol: 1.33e-15
        # 进行特殊边界条件的二项分布测试
        res = stats.binomtest(484, 967, 0.5)
        assert_allclose(res.pvalue, 1, rtol=rtol)
        res = stats.binomtest(3, 47, 3/47)
        assert_allclose(res.pvalue, 1, rtol=rtol)
        res = stats.binomtest(13, 46, 13/46)
        assert_allclose(res.pvalue, 1, rtol=rtol)
        res = stats.binomtest(15, 44, 15/44)
        assert_allclose(res.pvalue, 1, rtol=rtol)
        res = stats.binomtest(7, 13, 0.5)
        assert_allclose(res.pvalue, 1, rtol=rtol)
        res = stats.binomtest(6, 11, 0.5)
        assert_allclose(res.pvalue, 1, rtol=rtol)
    def test_binary_srch_for_binom_tst(self):
        # Test that old behavior of binomtest is maintained
        # by the new binary search method in cases where d
        # exactly equals the input on one side.
        
        # 设定二项分布的参数
        n = 10
        p = 0.5
        k = 3
        
        # 对于 k > PMF 的众数的情况进行第一轮测试
        i = np.arange(np.ceil(p * n), n+1)
        d = stats.binom.pmf(k, n, p)
        
        # 旧的计算 y 的方式，可能与 R 保持一致
        y1 = np.sum(stats.binom.pmf(i, n, p) <= d, axis=0)
        
        # 使用二分搜索的新方法
        ix = _binary_search_for_binom_tst(lambda x1:
                                          -stats.binom.pmf(x1, n, p),
                                          -d, np.ceil(p * n), n)
        
        # 计算新的 y
        y2 = n - ix + int(d == stats.binom.pmf(ix, n, p))
        
        # 断言两种计算方式的结果近似相等
        assert_allclose(y1, y2, rtol=1e-9)
        
        # 现在测试另一侧的情况
        k = 7
        i = np.arange(np.floor(p * n) + 1)
        d = stats.binom.pmf(k, n, p)
        
        # 旧的计算 y 的方式
        y1 = np.sum(stats.binom.pmf(i, n, p) <= d, axis=0)
        
        # 使用二分搜索的新方法
        ix = _binary_search_for_binom_tst(lambda x1:
                                          stats.binom.pmf(x1, n, p),
                                          d, 0, np.floor(p * n))
        
        # 计算新的 y
        y2 = ix + 1
        
        # 断言两种计算方式的结果近似相等
        assert_allclose(y1, y2, rtol=1e-9)

    # 这里的期望结果来自于 R 3.6.2 的 binom.test
    @pytest.mark.parametrize('alternative, pval, ci_low, ci_high',
                             [('less', 0.148831050443,
                               0.0, 0.2772002496709138),
                              ('greater', 0.9004695898947,
                               0.1366613252458672, 1.0),
                              ('two-sided', 0.2983720970096,
                               0.1266555521019559, 0.2918426890886281)])
    def test_confidence_intervals1(self, alternative, pval, ci_low, ci_high):
        # 执行二项检验并验证置信区间
        res = stats.binomtest(20, n=100, p=0.25, alternative=alternative)
        
        # 断言置信区间计算的 p 值接近预期值
        assert_allclose(res.pvalue, pval, rtol=1e-12)
        
        # 断言检验统计量的值等于 0.2
        assert_equal(res.statistic, 0.2)
        
        # 计算置信区间
        ci = res.proportion_ci(confidence_level=0.95)
        
        # 断言置信区间的上下界接近预期值
        assert_allclose((ci.low, ci.high), (ci_low, ci_high), rtol=1e-12)

    # 这里的期望结果同样来自于 R 3.6.2 的 binom.test.
    @pytest.mark.parametrize('alternative, pval, ci_low, ci_high',
                             [('less',
                               0.005656361, 0.0, 0.1872093),
                              ('greater',
                               0.9987146, 0.008860761, 1.0),
                              ('two-sided',
                               0.01191714, 0.006872485, 0.202706269)])
    # 使用 stats 模块的 binomtest 函数进行二项分布检验，设置参数并获取结果
    res = stats.binomtest(3, n=50, p=0.2, alternative=alternative)
    # 断言检查 p 值是否接近给定值 pval，允许的相对误差为 1e-6
    assert_allclose(res.pvalue, pval, rtol=1e-6)
    # 断言检查统计量是否等于 0.06
    assert_equal(res.statistic, 0.06)
    # 计算置信区间，指定置信水平为 0.99
    ci = res.proportion_ci(confidence_level=0.99)
    # 断言检查置信区间的下限和上限是否分别接近给定的 ci_low 和 ci_high，允许的相对误差为 1e-6
    assert_allclose((ci.low, ci.high), (ci_low, ci_high), rtol=1e-6)

# 使用 pytest 的参数化装饰器，设置多组测试参数
@pytest.mark.parametrize('alternative, pval, ci_high',
                         [('less', 0.05631351, 0.2588656),
                          ('greater', 1.0, 1.0),
                          ('two-sided', 0.07604122, 0.3084971)])
# 定义测试方法，测试二项分布情况下的置信区间，具体参数由装饰器提供
def test_confidence_interval_exact_k0(self, alternative, pval, ci_high):
    # 使用 stats 模块的 binomtest 函数进行二项分布检验，设置参数并获取结果
    res = stats.binomtest(0, 10, p=0.25, alternative=alternative)
    # 断言检查 p 值是否接近给定值 pval，允许的相对误差为 1e-6
    assert_allclose(res.pvalue, pval, rtol=1e-6)
    # 计算置信区间，指定置信水平为 0.95
    ci = res.proportion_ci(confidence_level=0.95)
    # 断言检查置信区间的下限是否等于 0.0
    assert_equal(ci.low, 0.0)
    # 断言检查置信区间的上限是否接近给定的 ci_high，允许的相对误差为 1e-6
    assert_allclose(ci.high, ci_high, rtol=1e-6)

# 使用 pytest 的参数化装饰器，设置多组测试参数
@pytest.mark.parametrize('alternative, pval, ci_low',
                         [('less', 1.0, 0.0),
                          ('greater', 9.536743e-07, 0.7411344),
                          ('two-sided', 9.536743e-07, 0.6915029)])
# 定义测试方法，测试二项分布情况下的置信区间，具体参数由装饰器提供
def test_confidence_interval_exact_k_is_n(self, alternative, pval, ci_low):
    # 使用 stats 模块的 binomtest 函数进行二项分布检验，设置参数并获取结果
    res = stats.binomtest(10, 10, p=0.25, alternative=alternative)
    # 断言检查 p 值是否接近给定值 pval，允许的相对误差为 1e-6
    assert_allclose(res.pvalue, pval, rtol=1e-6)
    # 计算置信区间，指定置信水平为 0.95
    ci = res.proportion_ci(confidence_level=0.95)
    # 断言检查置信区间的上限是否等于 1.0
    assert_equal(ci.high, 1.0)
    # 断言检查置信区间的下限是否接近给定的 ci_low，允许的相对误差为 1e-6
    assert_allclose(ci.low, ci_low, rtol=1e-6)
    @pytest.mark.parametrize(
        'k, alternative, corr, conf, ci_low, ci_high',
        # 参数化测试：定义多组参数，用于不同情况下的置信区间计算测试
        [[3, 'two-sided', True, 0.95, 0.08094782, 0.64632928],  # 示例参数组1
         [3, 'two-sided', True, 0.99, 0.0586329, 0.7169416],   # 示例参数组2
         [3, 'two-sided', False, 0.95, 0.1077913, 0.6032219],  # 示例参数组3
         [3, 'two-sided', False, 0.99, 0.07956632, 0.6799753], # 示例参数组4
         [3, 'less', True, 0.95, 0.0, 0.6043476],               # 示例参数组5
         [3, 'less', True, 0.99, 0.0, 0.6901811],               # 示例参数组6
         [3, 'less', False, 0.95, 0.0, 0.5583002],              # 示例参数组7
         [3, 'less', False, 0.99, 0.0, 0.6507187],              # 示例参数组8
         [3, 'greater', True, 0.95, 0.09644904, 1.0],           # 示例参数组9
         [3, 'greater', True, 0.99, 0.06659141, 1.0],           # 示例参数组10
         [3, 'greater', False, 0.95, 0.1268766, 1.0],           # 示例参数组11
         [3, 'greater', False, 0.99, 0.08974147, 1.0],          # 示例参数组12
         [0, 'two-sided', True, 0.95, 0.0, 0.3445372],          # 示例参数组13
         [0, 'two-sided', False, 0.95, 0.0, 0.2775328],         # 示例参数组14
         [0, 'less', True, 0.95, 0.0, 0.2847374],               # 示例参数组15
         [0, 'less', False, 0.95, 0.0, 0.212942],               # 示例参数组16
         [0, 'greater', True, 0.95, 0.0, 1.0],                  # 示例参数组17
         [0, 'greater', False, 0.95, 0.0, 1.0],                 # 示例参数组18
         [10, 'two-sided', True, 0.95, 0.6554628, 1.0],         # 示例参数组19
         [10, 'two-sided', False, 0.95, 0.7224672, 1.0],        # 示例参数组20
         [10, 'less', True, 0.95, 0.0, 1.0],                    # 示例参数组21
         [10, 'less', False, 0.95, 0.0, 1.0],                   # 示例参数组22
         [10, 'greater', True, 0.95, 0.7152626, 1.0],           # 示例参数组23
         [10, 'greater', False, 0.95, 0.787058, 1.0]]           # 示例参数组24
    )
    # 测试函数：测试置信区间的计算是否符合预期
    def test_ci_wilson_method(self, k, alternative, corr, conf,
                              ci_low, ci_high):
        # 调用二项分布测试函数计算结果
        res = stats.binomtest(k, n=10, p=0.1, alternative=alternative)
        # 根据参数判断使用的置信区间计算方法
        if corr:
            method = 'wilsoncc'
        else:
            method = 'wilson'
        # 计算置信区间
        ci = res.proportion_ci(confidence_level=conf, method=method)
        # 断言置信区间的上下界是否在预期范围内
        assert_allclose((ci.low, ci.high), (ci_low, ci_high), rtol=1e-6)

    def test_estimate_equals_hypothesized_prop(self):
        # 测试特殊情况：估计比例等于假设比例时的情况
        # 当 alternative 参数为 'two-sided' 时，p 值为 1
        res = stats.binomtest(4, 16, 0.25)
        assert_equal(res.statistic, 0.25)
        assert_equal(res.pvalue, 1.0)

    @pytest.mark.parametrize('k, n', [(0, 0), (-1, 2)])
    # 参数化测试：测试不合法的 k 和 n 的情况
    def test_invalid_k_n(self, k, n):
        with pytest.raises(ValueError,
                           match="must be an integer not less than"):
            stats.binomtest(k, n)

    def test_invalid_k_too_big(self):
        # 测试不合法的 k 大于 n 的情况
        with pytest.raises(ValueError,
                           match=r"k \(11\) must not be greater than n \(10\)."):
            stats.binomtest(11, 10, 0.25)

    def test_invalid_k_wrong_type(self):
        # 测试不合法的 k 类型错误的情况
        with pytest.raises(TypeError,
                           match="k must be an integer."):
            stats.binomtest([10, 11], 21, 0.25)
    # 定义测试函数，用于测试不合法的 p 值范围
    def test_invalid_p_range(self):
        # 设置错误消息，验证是否会引发 ValueError 异常
        message = r'p \(-0.5\) must be in range...'
        # 使用 pytest 来检查是否会抛出指定消息的 ValueError 异常
        with pytest.raises(ValueError, match=message):
            # 调用 binomtest 函数，传入不合法的 p 值
            stats.binomtest(50, 150, p=-0.5)
        # 设置另一个错误消息，验证是否会引发 ValueError 异常
        message = r'p \(1.5\) must be in range...'
        # 使用 pytest 来检查是否会抛出指定消息的 ValueError 异常
        with pytest.raises(ValueError, match=message):
            # 再次调用 binomtest 函数，传入不合法的 p 值
            stats.binomtest(50, 150, p=1.5)

    # 定义测试函数，用于测试不合法的置信水平值
    def test_invalid_confidence_level(self):
        # 调用 binomtest 函数，获取结果对象
        res = stats.binomtest(3, n=10, p=0.1)
        # 设置错误消息，验证是否会引发 ValueError 异常
        message = r"confidence_level \(-1\) must be in the interval"
        # 使用 pytest 来检查是否会抛出指定消息的 ValueError 异常
        with pytest.raises(ValueError, match=message):
            # 调用 proportion_ci 方法，传入不合法的置信水平值
            res.proportion_ci(confidence_level=-1)

    # 定义测试函数，用于测试不合法的置信区间计算方法
    def test_invalid_ci_method(self):
        # 调用 binomtest 函数，获取结果对象
        res = stats.binomtest(3, n=10, p=0.1)
        # 使用 pytest 来检查是否会抛出指定消息的 ValueError 异常
        with pytest.raises(ValueError, match=r"method \('plate of shrimp'\) must be"):
            # 调用 proportion_ci 方法，传入不合法的计算方法名称
            res.proportion_ci(method="plate of shrimp")

    # 定义测试函数，用于测试不合法的备择假设
    def test_invalid_alternative(self):
        # 使用 pytest 来检查是否会抛出指定消息的 ValueError 异常
        with pytest.raises(ValueError, match=r"alternative \('ekki'\) not..."):
            # 调用 binomtest 函数，传入不合法的备择假设
            stats.binomtest(3, n=10, p=0.1, alternative='ekki')

    # 定义测试函数，用于测试别名功能是否正常
    def test_alias(self):
        # 调用 binomtest 函数，获取结果对象
        res = stats.binomtest(3, n=10, p=0.1)
        # 使用 assert_equal 来验证两个值是否相等
        assert_equal(res.proportion_estimate, res.statistic)

    # 标记为跳过测试，如果系统位数不大于 32 位
    @pytest.mark.skipif(sys.maxsize <= 2**32, reason="32-bit does not overflow")
    def test_boost_overflow_raises(self):
        # 使用 pytest 来检查是否会抛出指定异常
        # 检查 Boost.Math 错误策略在 Python 中是否会引发 OverflowError
        with pytest.raises(OverflowError, match='Error in function...'):
            # 调用 binomtest 函数，传入可能引发溢出异常的参数
            stats.binomtest(5, 6, p=sys.float_info.min)
class TestFligner:

    def test_data(self):
        # numbers from R: fligner.test in package stats
        # 创建一个包含整数序列的 NumPy 数组 x1
        x1 = np.arange(5)
        # 断言 fligner 函数对 x1 和 x1 的平方进行测试后的近似数组相等性
        assert_array_almost_equal(stats.fligner(x1, x1**2),
                                  (3.2282229927203536, 0.072379187848207877),
                                  11)

    def test_trimmed1(self):
        # Perturb input to break ties in the transformed data
        # See https://github.com/scipy/scipy/pull/8042 for more details
        # 使用随机数种子为 123 创建一个随机数生成器 rs
        rs = np.random.RandomState(123)

        def _perturb(g):
            # 将 g 中的每个元素加上一个微小的随机扰动
            return (np.asarray(g) + 1e-10 * rs.randn(len(g))).tolist()

        # 对 g1、g2、g3 进行扰动处理
        g1_ = _perturb(g1)
        g2_ = _perturb(g2)
        g3_ = _perturb(g3)
        # 测试 center='trimmed' 时与 center='mean' 相比，是否得到相同结果
        Xsq1, pval1 = stats.fligner(g1_, g2_, g3_, center='mean')
        Xsq2, pval2 = stats.fligner(g1_, g2_, g3_, center='trimmed',
                                    proportiontocut=0.0)
        # 断言两种不同 center 参数设置下的结果近似相等
        assert_almost_equal(Xsq1, Xsq2)
        assert_almost_equal(pval1, pval2)

    def test_trimmed2(self):
        # 创建两个包含浮点数的列表 x 和 y
        x = [1.2, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0]
        y = [0.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 200.0]
        # 使用 center='trimmed' 进行测试
        Xsq1, pval1 = stats.fligner(x, y, center='trimmed',
                                    proportiontocut=0.125)
        # 在这里对数据进行修剪，然后使用 center='mean'
        Xsq2, pval2 = stats.fligner(x[1:-1], y[1:-1], center='mean')
        # 断言两种不同处理方式下的结果近似相等
        assert_almost_equal(Xsq1, Xsq2)
        assert_almost_equal(pval1, pval2)

    # The following test looks reasonable at first, but fligner() uses the
    # function stats.rankdata(), and in one of the cases in this test,
    # there are ties, while in the other (because of normal rounding
    # errors) there are not.  This difference leads to differences in the
    # third significant digit of W.
    #
    #def test_equal_mean_median(self):
    #    x = np.linspace(-1,1,21)
    #    y = x**3
    #    W1, pval1 = stats.fligner(x, y, center='mean')
    #    W2, pval2 = stats.fligner(x, y, center='median')
    #    assert_almost_equal(W1, W2)
    #    assert_almost_equal(pval1, pval2)

    def test_bad_keyword(self):
        # 创建一个包含等差数列的 NumPy 数组 x
        x = np.linspace(-1, 1, 21)
        # 断言 fligner 函数在给定错误的关键字参数时引发 TypeError
        assert_raises(TypeError, stats.fligner, x, x, portiontocut=0.1)

    def test_bad_center_value(self):
        # 创建一个包含等差数列的 NumPy 数组 x
        x = np.linspace(-1, 1, 21)
        # 断言 fligner 函数在给定错误的 center 参数值时引发 ValueError
        assert_raises(ValueError, stats.fligner, x, x, center='trim')

    def test_bad_num_args(self):
        # 断言 fligner 函数在参数数量不足时引发 ValueError
        assert_raises(ValueError, stats.fligner, [1])

    def test_empty_arg(self):
        # 创建一个包含整数序列的 NumPy 数组 x
        x = np.arange(5)
        # 断言 fligner 函数在处理空参数时会引发警告，并返回 NaN
        with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
            res = stats.fligner(x, x**2, [])
            assert_equal(res.statistic, np.nan)
            assert_equal(res.pvalue, np.nan)


def mood_cases_with_ties():
    # 生成具有内部和样本之间的随机关系的随机 `x` 和 `y` 数组。期望结果来自 SAS 中的 (统计量, p 值)。
    expected_results = [(-1.76658511464992, .0386488678399305),
                        (-.694031428192304, .2438312498647250),
                        (-1.15093525352151, .1248794365836150)]
    # 定义用于生成随机数的种子列表
    seeds = [23453254, 1298352315, 987234597]
    # 对每个种子进行循环迭代
    for si, seed in enumerate(seeds):
        # 使用指定种子创建 NumPy 随机数生成器对象
        rng = np.random.default_rng(seed)
        # 生成长度为 100 的随机数组 `xy`
        xy = rng.random(100)
        # 生成随机索引以创建 ties（重复值）
        tie_ind = rng.integers(low=0, high=99, size=5)
        # 为每个索引生成随机数量的 ties
        num_ties_per_ind = rng.integers(low=1, high=5, size=5)
        # 在每个 `tie_ind` 处，将接下来的 `n` 个索引的值设置为相等
        for i, n in zip(tie_ind, num_ties_per_ind):
            for j in range(i + 1, i + n):
                xy[j] = xy[i]
        # 在将 `xy` 分割为 `x, y` 之前，打乱 `xy` 的顺序
        rng.shuffle(xy)
        # 将 `xy` 分割为两个数组 `x` 和 `y`
        x, y = np.split(xy, 2)
        # 生成器返回 `x, y` 及其余参数，用于后续的统计分析
        yield x, y, 'less', *expected_results[si]
# 定义一个测试类 TestMood，用于测试 Mood 方法的功能
class TestMood:

    # 使用 pytest 的 parametrize 装饰器，传入多组参数化测试数据，来测试 mood_cases_with_ties 返回的数据
    @pytest.mark.parametrize("x,y,alternative,stat_expect,p_expect",
                             mood_cases_with_ties())
    def test_against_SAS(self, x, y, alternative, stat_expect, p_expect):
        """
        Example code used to generate SAS output:
        DATA myData;
        INPUT X Y;
        CARDS;
        1 0
        1 1
        1 2
        1 3
        1 4
        2 0
        2 1
        2 4
        2 9
        2 16
        ods graphics on;
        proc npar1way mood data=myData ;
           class X;
            ods output  MoodTest=mt;
        proc contents data=mt;
        proc print data=mt;
          format     Prob1 17.16 Prob2 17.16 Statistic 17.16 Z 17.16 ;
            title "Mood Two-Sample Test";
        proc print data=myData;
            title "Data for above results";
          run;
        """
        # 调用 stats 模块的 mood 函数，计算统计量和 p 值
        statistic, pvalue = stats.mood(x, y, alternative=alternative)
        # 使用 assert_allclose 断言函数，验证计算得到的统计量和 p 值与预期值的接近程度
        assert_allclose(stat_expect, statistic, atol=1e-16)
        assert_allclose(p_expect, pvalue, atol=1e-16)

    # 使用 pytest 的 parametrize 装饰器，传入多组参数化测试数据，来测试不同 alternative 下的预期输出
    @pytest.mark.parametrize("alternative, expected",
                             [('two-sided', (1.019938533549930,
                                             .3077576129778760)),
                              ('less', (1.019938533549930,
                                        1 - .1538788064889380)),
                              ('greater', (1.019938533549930,
                                           .1538788064889380))])
    def test_against_SAS_2(self, alternative, expected):
        # Code to run in SAS in above function
        # 定义两组数据 x 和 y，用于测试不同 alternative 下的 Mood 方法的预期输出
        x = [111, 107, 100, 99, 102, 106, 109, 108, 104, 99,
             101, 96, 97, 102, 107, 113, 116, 113, 110, 98]
        y = [107, 108, 106, 98, 105, 103, 110, 105, 104, 100,
             96, 108, 103, 104, 114, 114, 113, 108, 106, 99]
        # 调用 stats 模块的 mood 函数，计算统计量和 p 值
        res = stats.mood(x, y, alternative=alternative)
        # 使用 assert_allclose 断言函数，验证计算得到的结果与预期的接近程度
        assert_allclose(res, expected)

    # 测试当参数顺序改变时，Mood 方法的统计量 z 应该改变符号，p 值不应该改变
    def test_mood_order_of_args(self):
        # 设定随机种子以便重现结果
        np.random.seed(1234)
        # 生成两组随机数据 x1 和 x2
        x1 = np.random.randn(10, 1)
        x2 = np.random.randn(15, 1)
        # 分别调用 stats 模块的 mood 函数，计算统计量 z 和 p 值
        z1, p1 = stats.mood(x1, x2)
        z2, p2 = stats.mood(x2, x1)
        # 使用 assert_array_almost_equal 断言函数，验证 z1 和 p1 与 -z2 和 p2 的接近程度
        assert_array_almost_equal([z1, p1], [-z2, p2])
    def test_mood_with_axis_none(self):
        # Test with axis = None, compare with results from R

        # Sample data for group 1
        x1 = [-0.626453810742332, 0.183643324222082, -0.835628612410047,
              1.59528080213779, 0.329507771815361, -0.820468384118015,
              0.487429052428485, 0.738324705129217, 0.575781351653492,
              -0.305388387156356, 1.51178116845085, 0.389843236411431,
              -0.621240580541804, -2.2146998871775, 1.12493091814311,
              -0.0449336090152309, -0.0161902630989461, 0.943836210685299,
              0.821221195098089, 0.593901321217509]

        # Sample data for group 2
        x2 = [-0.896914546624981, 0.184849184646742, 1.58784533120882,
              -1.13037567424629, -0.0802517565509893, 0.132420284381094,
              0.707954729271733, -0.23969802417184, 1.98447393665293,
              -0.138787012119665, 0.417650750792556, 0.981752777463662,
              -0.392695355503813, -1.03966897694891, 1.78222896030858,
              -2.31106908460517, 0.878604580921265, 0.035806718015226,
              1.01282869212708, 0.432265154539617, 2.09081920524915,
              -1.19992581964387, 1.58963820029007, 1.95465164222325,
              0.00493777682814261, -2.45170638784613, 0.477237302613617,
              -0.596558168631403, 0.792203270299649, 0.289636710177348]

        # Convert lists to numpy arrays
        x1 = np.array(x1)
        x2 = np.array(x2)

        # Reshape arrays to match required shape
        x1.shape = (10, 2)
        x2.shape = (15, 2)

        # Perform mood test with axis=None and assert the result
        assert_array_almost_equal(stats.mood(x1, x2, axis=None),
                                  [-1.31716607555, 0.18778296257])

    def test_mood_2d(self):
        # Test if the results of mood test in 2-D case are consistent with the
        # R result for the same inputs. Numbers from R mood.test().

        # Number of variables
        ny = 5

        # Set random seed for reproducibility
        np.random.seed(1234)

        # Generate random data for two groups
        x1 = np.random.randn(10, ny)
        x2 = np.random.randn(15, ny)

        # Perform mood test on the entire data arrays
        z_vectest, pval_vectest = stats.mood(x1, x2)

        # Assert the results for each variable separately
        for j in range(ny):
            assert_array_almost_equal([z_vectest[j], pval_vectest[j]],
                                      stats.mood(x1[:, j], x2[:, j]))

        # Transpose arrays to test axis handling
        x1 = x1.transpose()
        x2 = x2.transpose()

        # Perform mood test with axis=1 (along columns) and assert results
        z_vectest, pval_vectest = stats.mood(x1, x2, axis=1)
        for i in range(ny):
            assert_array_almost_equal([z_vectest[i], pval_vectest[i]],
                                      stats.mood(x1[i, :], x2[i, :]))
    def test_mood_3d(self):
        # 定义一个形状为 (10, 5, 6) 的三维数组
        shape = (10, 5, 6)
        # 设置随机种子为 1234
        np.random.seed(1234)
        # 生成两个形状为 shape 的随机数组 x1 和 x2
        x1 = np.random.randn(*shape)
        x2 = np.random.randn(*shape)

        # 遍历三个轴
        for axis in range(3):
            # 对 x1 和 x2 沿着当前轴进行 mood 检验
            z_vectest, pval_vectest = stats.mood(x1, x2, axis=axis)
            # 对三维数组进行检验，检验结果应该与从三维数组中取出一维数组后进行的检验结果相同
            axes_idx = ([1, 2], [0, 2], [0, 1])  # 两个不等于当前轴的轴索引
            for i in range(shape[axes_idx[axis][0]]):
                for j in range(shape[axes_idx[axis][1]]):
                    # 根据当前轴不同取出不同的切片
                    if axis == 0:
                        slice1 = x1[:, i, j]
                        slice2 = x2[:, i, j]
                    elif axis == 1:
                        slice1 = x1[i, :, j]
                        slice2 = x2[i, :, j]
                    else:
                        slice1 = x1[i, j, :]
                        slice2 = x2[i, j, :]

                    # 断言当前位置的 mood 检验结果应与切片后的 mood 检验结果相同
                    assert_array_almost_equal([z_vectest[i, j],
                                               pval_vectest[i, j]],
                                              stats.mood(slice1, slice2))

    def test_mood_bad_arg(self):
        # 测试当参数长度之和小于 3 时是否会触发警告
        with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
            res = stats.mood([1], [])
            assert_equal(res.statistic, np.nan)
            assert_equal(res.pvalue, np.nan)

    def test_mood_alternative(self):
        # 使用固定的随机种子生成两个大小为 100 的正态分布样本 x 和 y
        np.random.seed(0)
        x = stats.norm.rvs(scale=0.75, size=100)
        y = stats.norm.rvs(scale=1.25, size=100)

        # 分别计算两组数据的 mood 检验结果，使用不同的 alternative 参数
        stat1, p1 = stats.mood(x, y, alternative='two-sided')
        stat2, p2 = stats.mood(x, y, alternative='less')
        stat3, p3 = stats.mood(x, y, alternative='greater')

        # 断言三种 alternative 下的统计量应相等
        assert stat1 == stat2 == stat3
        # 检查 p 值是否在一定精度范围内接近 0
        assert_allclose(p1, 0, atol=1e-7)
        # 检查 less alternative 下的 p 值是否近似等于 two-sided alternative 的一半
        assert_allclose(p2, p1/2)
        # 检查 greater alternative 下的 p 值是否近似等于 1 减去 two-sided alternative 的一半
        assert_allclose(p3, 1 - p1/2)

        # 测试当 alternative 参数为非法值时是否会引发 ValueError 异常
        with pytest.raises(ValueError, match="`alternative` must be..."):
            stats.mood(x, y, alternative='ekki-ekki')

    @pytest.mark.parametrize("alternative", ['two-sided', 'less', 'greater'])
    def test_result(self, alternative):
        # 使用指定随机种子生成两个正态分布的一维数组 x1 和 x2
        rng = np.random.default_rng(265827767938813079281100964083953437622)
        x1 = rng.standard_normal((10, 1))
        x2 = rng.standard_normal((15, 1))

        # 对 x1 和 x2 进行 mood 检验，使用参数化的 alternative 参数
        res = stats.mood(x1, x2, alternative=alternative)
        # 断言返回的统计量和 p 值应与 res 中的值相等
        assert_equal((res.statistic, res.pvalue), res)
# 定义一个测试类 TestProbplot，用于测试概率图相关的统计函数
class TestProbplot:

    # 测试基本的概率图函数
    def test_basic(self):
        # 生成服从标准正态分布的随机样本
        x = stats.norm.rvs(size=20, random_state=12345)
        # 计算概率图的数据点
        osm, osr = stats.probplot(x, fit=False)
        # 预期的标准化残差值
        osm_expected = [-1.8241636, -1.38768012, -1.11829229, -0.91222575,
                        -0.73908135, -0.5857176, -0.44506467, -0.31273668,
                        -0.18568928, -0.06158146, 0.06158146, 0.18568928,
                        0.31273668, 0.44506467, 0.5857176, 0.73908135,
                        0.91222575, 1.11829229, 1.38768012, 1.8241636]
        # 断言标准化残差和预期值的近似性
        assert_allclose(osr, np.sort(x))
        assert_allclose(osm, osm_expected)

        # 使用 fit=True 参数计算拟合后的概率图
        res, res_fit = stats.probplot(x, fit=True)
        # 预期的拟合后残差值
        res_fit_expected = [1.05361841, 0.31297795, 0.98741609]
        # 断言拟合后残差和预期值的近似性
        assert_allclose(res_fit, res_fit_expected)

    # 测试 sparams 关键字参数
    def test_sparams_keyword(self):
        # 生成服从标准正态分布的较大随机样本
        x = stats.norm.rvs(size=100, random_state=123456)
        # 检查 sparams=None, sparams=0 和 sparams=() 时的概率图
        osm1, osr1 = stats.probplot(x, sparams=None, fit=False)
        osm2, osr2 = stats.probplot(x, sparams=0, fit=False)
        osm3, osr3 = stats.probplot(x, sparams=(), fit=False)
        # 断言不同参数设置下的标准化残差的近似性
        assert_allclose(osm1, osm2)
        assert_allclose(osm1, osm3)
        assert_allclose(osr1, osr2)
        assert_allclose(osr1, osr3)
        # 使用 (loc, scale) 参数计算标准正态分布的概率图

    # 测试 dist 关键字参数
    def test_dist_keyword(self):
        # 生成服从标准正态分布的随机样本
        x = stats.norm.rvs(size=20, random_state=12345)
        # 使用 t 分布拟合的概率图
        osm1, osr1 = stats.probplot(x, fit=False, dist='t', sparams=(3,))
        osm2, osr2 = stats.probplot(x, fit=False, dist=stats.t, sparams=(3,))
        # 断言使用不同参数设置时 t 分布概率图的标准化残差的近似性
        assert_allclose(osm1, osm2)
        assert_allclose(osr1, osr2)

        # 断言错误的分布名称会引发 ValueError 异常
        assert_raises(ValueError, stats.probplot, x, dist='wrong-dist-name')
        # 断言错误的参数类型会引发 AttributeError 异常
        assert_raises(AttributeError, stats.probplot, x, dist=[])

        # 自定义分布类，模拟一个类似分布的对象
        class custom_dist:
            """Some class that looks just enough like a distribution."""
            # 定义分位点函数
            def ppf(self, q):
                return stats.norm.ppf(q, loc=2)

        # 使用自定义分布对象进行概率图计算
        osm1, osr1 = stats.probplot(x, sparams=(2,), fit=False)
        osm2, osr2 = stats.probplot(x, dist=custom_dist(), fit=False)
        # 断言自定义分布对象和参数化分布的概率图的标准化残差的近似性
        assert_allclose(osm1, osm2)
        assert_allclose(osr1, osr2)

    # 标记：如果没有 matplotlib，跳过测试
    @pytest.mark.skipif(not have_matplotlib, reason="no matplotlib")
    # 定义一个测试方法，用于验证 `plot` 关键字参数的不同组合是否能产生一致的结果
    def test_plot_kwarg(self):
        # 创建一个新的 Matplotlib 图形对象
        fig = plt.figure()
        # 向图形对象添加一个子图
        fig.add_subplot(111)
        # 生成一个服从自由度为 3 的 t 分布的随机样本数组 x
        x = stats.t.rvs(3, size=100, random_state=7654321)
        # 对数组 x 进行概率图分析，并将结果绘制在当前的 Matplotlib 图形对象中
        res1, fitres1 = stats.probplot(x, plot=plt)
        # 关闭当前的 Matplotlib 图形对象
        plt.close()
        # 对数组 x 进行概率图分析，但不进行绘制
        res2, fitres2 = stats.probplot(x, plot=None)
        # 对数组 x 进行概率图分析，并指定不拟合直线，将结果绘制在当前的 Matplotlib 图形对象中
        res3 = stats.probplot(x, fit=False, plot=plt)
        # 关闭当前的 Matplotlib 图形对象
        plt.close()
        # 对数组 x 进行概率图分析，不进行拟合直线，也不进行绘制
        res4 = stats.probplot(x, fit=False, plot=None)
        # 断言所有的概率图分析结果都具有相同的长度为 2
        assert_(len(res1) == len(res2) == len(res3) == len(res4) == 2)
        # 断言各组概率图分析结果在数值上非常接近
        assert_allclose(res1, res2)
        assert_allclose(res1, res3)
        assert_allclose(res1, res4)
        # 断言不拟合直线的结果 fitres1 和 fitres2 在数值上非常接近
        assert_allclose(fitres1, fitres2)

        # 验证 `plot` 参数能够接受 Matplotlib 的 Axes 对象作为参数
        # 创建一个新的 Matplotlib 图形对象
        fig = plt.figure()
        # 向图形对象添加一个子图，并将返回的 Axes 对象赋给变量 ax
        ax = fig.add_subplot(111)
        # 对数组 x 进行概率图分析，不进行拟合直线，将结果绘制在指定的 Axes 对象上
        stats.probplot(x, fit=False, plot=ax)
        # 关闭当前的 Matplotlib 图形对象
        plt.close()

    # 定义一个测试方法，用于验证当指定一个无效的分布时是否能触发 ValueError 异常
    def test_probplot_bad_args(self):
        # 断言调用 `stats.probplot` 方法时，传入一个无效的分布参数会触发 ValueError 异常
        assert_raises(ValueError, stats.probplot, [1], dist="plate_of_shrimp")

    # 定义一个测试方法，用于验证当传入空数组时 `stats.probplot` 的行为是否正确
    def test_empty(self):
        # 断言对空数组调用 `stats.probplot` 且不拟合直线时，返回的结果应该是空数组的概率图分析结果
        assert_equal(stats.probplot([], fit=False),
                     (np.array([]), np.array([])))
        # 断言对空数组调用 `stats.probplot` 且进行拟合直线时，返回的结果应该包含空数组的概率图分析结果
        # 和 NaN 值构成的元组
        assert_equal(stats.probplot([], fit=True),
                     ((np.array([]), np.array([])),
                      (np.nan, np.nan, 0.0)))

    # 定义一个测试方法，用于验证当传入长度为 1 的数组时 `stats.probplot` 的行为是否正确
    def test_array_of_size_one(self):
        # 使用 `np.errstate` 来忽略可能出现的无效数值警告
        with np.errstate(invalid='ignore'):
            # 断言对长度为 1 的数组调用 `stats.probplot` 且进行拟合直线时，返回的结果应该包含
            # 由一个点构成的概率图分析结果和 NaN 值构成的元组
            assert_equal(stats.probplot([1], fit=True),
                         ((np.array([0.]), np.array([1])),
                          (np.nan, np.nan, 0.0)))
class TestWilcoxon:
    # 定义测试类 TestWilcoxon

    def test_wilcoxon_bad_arg(self):
        # 测试当给定的两个参数长度不同或者 zero_method 未知时是否能引发 ValueError 异常
        assert_raises(ValueError, stats.wilcoxon, [1], [1, 2])
        assert_raises(ValueError, stats.wilcoxon, [1, 2], [1, 2], "dummy")
        assert_raises(ValueError, stats.wilcoxon, [1, 2], [1, 2],
                      alternative="dummy")
        assert_raises(ValueError, stats.wilcoxon, [1]*10, mode="xyz")

    def test_zero_diff(self):
        # 测试在 x - y == 0 时，pratt 和 wilcox 是否能引发 ValueError 异常
        x = np.arange(20)
        assert_raises(ValueError, stats.wilcoxon, x, x, "wilcox",
                      mode="approx")
        assert_raises(ValueError, stats.wilcoxon, x, x, "pratt",
                      mode="approx")
        # 当 zero_method == "zsplit" 时，ranksum 是 n*(n+1)/2 的一半
        assert_equal(stats.wilcoxon(x, x, "zsplit", mode="approx"),
                     (20*21/4, 1.0))

    def test_pratt(self):
        # 针对 gh-6805 的回归测试：检查与 R 包 coin (wilcoxsign_test) 报告的 p 值匹配性
        x = [1, 2, 3, 4]
        y = [1, 2, 3, 5]
        with suppress_warnings() as sup:
            sup.filter(UserWarning, message="Sample size too small")
            # 使用 zero_method="pratt" 和 mode="approx" 调用 wilcoxon 函数
            res = stats.wilcoxon(x, y, zero_method="pratt", mode="approx")
        assert_allclose(res, (0.0, 0.31731050786291415))

    def test_wilcoxon_arg_type(self):
        # 测试能够接受列表作为参数，解决问题 6070
        arr = [1, 2, 3, 0, -1, 3, 1, 2, 1, 1, 2]

        _ = stats.wilcoxon(arr, zero_method="pratt", mode="approx")
        _ = stats.wilcoxon(arr, zero_method="zsplit", mode="approx")
        _ = stats.wilcoxon(arr, zero_method="wilcox", mode="approx")
    # 定义测试函数，用于测试 Wilcoxon 签名秩检验的准确性
    def test_accuracy_wilcoxon(self):
        # 定义频率列表
        freq = [1, 4, 16, 15, 8, 4, 5, 1, 2]
        # 定义数字范围
        nums = range(-4, 5)
        # 根据频率和数字范围生成数组 x，将每个数字按照对应频率重复
        x = np.concatenate([[u] * v for u, v in zip(nums, freq)])
        # 创建一个与 x 大小相同的全零数组 y
        y = np.zeros(x.size)

        # 执行 Wilcoxon 签名秩检验，使用 Pratt 方法和近似模式
        T, p = stats.wilcoxon(x, y, "pratt", mode="approx")
        # 断言检验统计量 T 的值近似于 423
        assert_allclose(T, 423)
        # 断言 p 值近似于 0.0031724568006762576
        assert_allclose(p, 0.0031724568006762576)

        # 重复上述步骤，使用 Z 分裂方法
        T, p = stats.wilcoxon(x, y, "zsplit", mode="approx")
        assert_allclose(T, 441)
        assert_allclose(p, 0.0032145343172473055)

        # 再次重复上述步骤，使用 Wilcoxon 方法
        T, p = stats.wilcoxon(x, y, "wilcox", mode="approx")
        assert_allclose(T, 327)
        assert_allclose(p, 0.00641346115861)

        # 测试 'correction' 选项，使用 R 计算的值进行检验
        # 对新的 x, y 进行赋值
        x = np.array([120, 114, 181, 188, 180, 146, 121, 191, 132, 113, 127, 112])
        y = np.array([133, 143, 119, 189, 112, 199, 198, 113, 115, 121, 142, 187])
        # 执行 Wilcoxon 签名秩检验，不进行校正
        T, p = stats.wilcoxon(x, y, correction=False, mode="approx")
        # 断言检验统计量 T 的值为 34
        assert_equal(T, 34)
        # 断言 p 值近似于 0.6948866，相对误差小于 1e-6
        assert_allclose(p, 0.6948866, rtol=1e-6)
        # 执行 Wilcoxon 签名秩检验，进行校正
        T, p = stats.wilcoxon(x, y, correction=True, mode="approx")
        assert_equal(T, 34)
        assert_allclose(p, 0.7240817, rtol=1e-6)

    # 定义测试函数，用于检验 Wilcoxon 签名秩检验的结果属性
    def test_wilcoxon_result_attributes(self):
        # 定义输入数组 x 和 y
        x = np.array([120, 114, 181, 188, 180, 146, 121, 191, 132, 113, 127, 112])
        y = np.array([133, 143, 119, 189, 112, 199, 198, 113, 115, 121, 142, 187])
        # 执行 Wilcoxon 签名秩检验，不进行校正
        res = stats.wilcoxon(x, y, correction=False, mode="approx")
        # 定义期望的结果属性列表
        attributes = ('statistic', 'pvalue')
        # 检查返回结果的命名属性是否符合预期
        check_named_results(res, attributes)

    # 定义测试函数，用于检验 Wilcoxon 签名秩检验是否具有 z 统计量
    def test_wilcoxon_has_zstatistic(self):
        # 生成随机数种子
        rng = np.random.default_rng(89426135444)
        # 生成大小为 15 的随机数组 x 和 y
        x, y = rng.random(15), rng.random(15)

        # 执行 Wilcoxon 签名秩检验，使用近似模式
        res = stats.wilcoxon(x, y, mode="approx")
        # 计算参考值 ref，应为 stats.norm.ppf(res.pvalue/2)
        ref = stats.norm.ppf(res.pvalue/2)
        # 断言返回结果具有 z 统计量，并且其值近似于参考值 ref
        assert_allclose(res.zstatistic, ref)

        # 再次执行 Wilcoxon 签名秩检验，使用精确模式，检验结果不应具有 z 统计量
        res = stats.wilcoxon(x, y, mode="exact")
        assert not hasattr(res, 'zstatistic')

        # 第三次执行 Wilcoxon 签名秩检验，未指定模式，默认为精确模式，结果不应具有 z 统计量
        res = stats.wilcoxon(x, y)
        assert not hasattr(res, 'zstatistic')

    # 定义测试函数，用于检验 Wilcoxon 签名秩检验在出现并列情况时的表现
    def test_wilcoxon_tie(self):
        # 回归测试 gh-2391
        # 对应的 R 代码为：
        #   > result = wilcox.test(rep(0.1, 10), exact=FALSE, correct=FALSE)
        #   > result$p.value
        #   [1] 0.001565402
        #   > result = wilcox.test(rep(0.1, 10), exact=FALSE, correct=TRUE)
        #   > result$p.value
        #   [1] 0.001904195
        # 执行 Wilcoxon 签名秩检验，输入全为 0.1 的数组，使用近似模式
        stat, p = stats.wilcoxon([0.1] * 10, mode="approx")
        # 预期的 p 值
        expected_p = 0.001565402
        # 断言检验统计量 stat 的值为 0
        assert_equal(stat, 0)
        # 断言 p 值近似于预期的值，相对误差小于 1e-6
        assert_allclose(p, expected_p, rtol=1e-6)

        # 再次执行 Wilcoxon 签名秩检验，使用校正选项，近似模式
        stat, p = stats.wilcoxon([0.1] * 10, correction=True, mode="approx")
        # 预期的 p 值
        expected_p = 0.001904195
        # 断言检验统计量 stat 的值为 0
        assert_equal(stat, 0)
        # 断言 p 值近似于预期的值，相对误差小于 1e-6
        assert_allclose(p, expected_p, rtol=1e-6)
    def test_onesided(self):
        # 对照 "R version 3.4.1 (2017-06-30)" 进行测试
        # 定义两组数据 x 和 y
        x = [125, 115, 130, 140, 140, 115, 140, 125, 140, 135]
        y = [110, 122, 125, 120, 140, 124, 123, 137, 135, 145]

        # 忽略特定警告信息
        with suppress_warnings() as sup:
            sup.filter(UserWarning, message="Sample size too small")
            # 使用 Wilcoxon 符号秩检验计算统计量 w 和 p 值，设定单侧检验（小于），近似模式
            w, p = stats.wilcoxon(x, y, alternative="less", mode="approx")
        # 断言统计量 w 的值为 27
        assert_equal(w, 27)
        # 断言 p 值的近似值为 0.7031847，精确到小数点后六位
        assert_almost_equal(p, 0.7031847, decimal=6)

        with suppress_warnings() as sup:
            sup.filter(UserWarning, message="Sample size too small")
            # 使用 Wilcoxon 符号秩检验计算统计量 w 和 p 值，设定单侧检验（小于），进行校正，近似模式
            w, p = stats.wilcoxon(x, y, alternative="less", correction=True,
                                  mode="approx")
        # 断言统计量 w 的值为 27
        assert_equal(w, 27)
        # 断言 p 值的近似值为 0.7233656，精确到小数点后六位
        assert_almost_equal(p, 0.7233656, decimal=6)

        with suppress_warnings() as sup:
            sup.filter(UserWarning, message="Sample size too small")
            # 使用 Wilcoxon 符号秩检验计算统计量 w 和 p 值，设定单侧检验（大于），近似模式
            w, p = stats.wilcoxon(x, y, alternative="greater", mode="approx")
        # 断言统计量 w 的值为 27
        assert_equal(w, 27)
        # 断言 p 值的近似值为 0.2968153，精确到小数点后六位
        assert_almost_equal(p, 0.2968153, decimal=6)

        with suppress_warnings() as sup:
            sup.filter(UserWarning, message="Sample size too small")
            # 使用 Wilcoxon 符号秩检验计算统计量 w 和 p 值，设定单侧检验（大于），进行校正，近似模式
            w, p = stats.wilcoxon(x, y, alternative="greater", correction=True,
                                  mode="approx")
        # 断言统计量 w 的值为 27
        assert_equal(w, 27)
        # 断言 p 值的近似值为 0.3176447，精确到小数点后六位
        assert_almost_equal(p, 0.3176447, decimal=6)

    def test_exact_basic(self):
        # 对于 n 从 1 到 50 的每个值进行测试
        for n in range(1, 51):
            # 获取 Wilcoxon 分布的概率质量函数 pmf1 和 pmf2
            pmf1 = _get_wilcoxon_distr(n)
            pmf2 = _get_wilcoxon_distr2(n)
            # 断言 pmf1 的长度应为 n*(n+1)/2 + 1
            assert_equal(n*(n+1)/2 + 1, len(pmf1))
            # 断言 pmf1 的所有元素之和为 1
            assert_equal(sum(pmf1), 1)
            # 断言 pmf1 和 pmf2 在精度内相等
            assert_array_almost_equal(pmf1, pmf2)
    # 定义一个测试方法，用于测试精确 p 值的计算

    def test_exact_pval(self):
        # 使用 "R version 3.4.1 (2017-06-30)" 计算的预期值

        # 创建包含一维数组的 NumPy 数组 x，表示样本 x 的数据
        x = np.array([1.81, 0.82, 1.56, -0.48, 0.81, 1.28, -1.04, 0.23,
                      -0.75, 0.14])
        # 创建包含一维数组的 NumPy 数组 y，表示样本 y 的数据
        y = np.array([0.71, 0.65, -0.2, 0.85, -1.1, -0.45, -0.84, -0.24,
                      -0.68, -0.76])
        
        # 计算两样本 Wilcoxon 符号秩检验的 p 值，双侧检验模式
        _, p = stats.wilcoxon(x, y, alternative="two-sided", mode="exact")
        # 断言 p 值近似等于给定值，精度为小数点后六位
        assert_almost_equal(p, 0.1054688, decimal=6)
        
        # 计算两样本 Wilcoxon 符号秩检验的 p 值，小于检验模式
        _, p = stats.wilcoxon(x, y, alternative="less", mode="exact")
        # 断言 p 值近似等于给定值，精度为小数点后六位
        assert_almost_equal(p, 0.9580078, decimal=6)
        
        # 计算两样本 Wilcoxon 符号秩检验的 p 值，大于检验模式
        _, p = stats.wilcoxon(x, y, alternative="greater", mode="exact")
        # 断言 p 值近似等于给定值，精度为小数点后六位
        assert_almost_equal(p, 0.05273438, decimal=6)

        # 创建包含从 0 到 19 的一维数组的 NumPy 数组 x
        x = np.arange(0, 20) + 0.5
        # 创建包含从 20 到 1 的一维数组的 NumPy 数组 y
        y = np.arange(20, 0, -1)
        
        # 计算两样本 Wilcoxon 符号秩检验的 p 值，双侧检验模式
        _, p = stats.wilcoxon(x, y, alternative="two-sided", mode="exact")
        # 断言 p 值近似等于给定值，精度为小数点后六位
        assert_almost_equal(p, 0.8694878, decimal=6)
        
        # 计算两样本 Wilcoxon 符号秩检验的 p 值，小于检验模式
        _, p = stats.wilcoxon(x, y, alternative="less", mode="exact")
        # 断言 p 值近似等于给定值，精度为小数点后六位
        assert_almost_equal(p, 0.4347439, decimal=6)
        
        # 计算两样本 Wilcoxon 符号秩检验的 p 值，大于检验模式
        _, p = stats.wilcoxon(x, y, alternative="greater", mode="exact")
        # 断言 p 值近似等于给定值，精度为小数点后六位
        assert_almost_equal(p, 0.5795889, decimal=6)

    # 这些输入被选择以使 W 统计量处于分布的中心（当支持的长度为奇数时），
    # 或处于中心左侧的值（当支持的长度为偶数时）。此外，所选数字使得
    # W 统计量等于正值的总和。

    @pytest.mark.parametrize('x', [[-1, -2, 3],
                                   [-1, 2, -3, -4, 5],
                                   [-1, -2, 3, -4, -5, -6, 7, 8]])
    # 参数化测试方法，测试不同输入对于 test_exact_p_1 方法的影响
    def test_exact_p_1(self, x):
        # 使用 Wilcoxon 符号秩检验计算给定样本 x 的 W 值和 p 值
        w, p = stats.wilcoxon(x)
        # 将 x 转换为 NumPy 数组
        x = np.array(x)
        # 计算真实的 W 值，即 x 中大于 0 的元素之和
        wtrue = x[x > 0].sum()
        # 断言计算得到的 W 值与真实的 W 值相等
        assert_equal(w, wtrue)
        # 断言 p 值等于 1
        assert_equal(p, 1)

    def test_auto(self):
        # 如果没有绑定并且 n <= 25，则自动默认为精确模式
        x = np.arange(0, 25) + 0.5
        y = np.arange(25, 0, -1)
        # 断言两种调用方式下的 Wilcoxon 符号秩检验结果相等
        assert_equal(stats.wilcoxon(x, y),
                     stats.wilcoxon(x, y, mode="exact"))

        # 如果存在绑定（即 d = x-y 中有零），则切换到近似模式
        d = np.arange(0, 13)
        with suppress_warnings() as sup:
            # 忽略特定的警告信息
            sup.filter(UserWarning, message="Exact p-value calculation")
            # 使用 Wilcoxon 符号秩检验计算给定数据 d 的 W 值和 p 值
            w, p = stats.wilcoxon(d)
        # 断言使用近似模式计算的结果与直接指定近似模式的结果相等
        assert_equal(stats.wilcoxon(d, mode="approx"), (w, p))

        # 对于样本数量大于 25，使用近似模式
        d = np.arange(1, 52)
        # 断言两种调用方式下的 Wilcoxon 符号秩检验结果相等
        assert_equal(stats.wilcoxon(d), stats.wilcoxon(d, mode="approx"))

    @pytest.mark.parametrize('size', [3, 5, 10])
    # 参数化测试方法，测试不同大小的输入对于测试的影响
    def test_permutation_method(self, size):
        # 使用指定种子初始化随机数生成器
        rng = np.random.default_rng(92348034828501345)
        # 生成指定大小的随机数组
        x = rng.random(size=size)
        # 使用置换法进行 Wilcoxon 符号秩检验
        res = stats.wilcoxon(x, method=stats.PermutationMethod())
        # 使用精确法进行 Wilcoxon 符号秩检验作为参考
        ref = stats.wilcoxon(x, method='exact')
        # 断言检验统计量相等
        assert_equal(res.statistic, ref.statistic)
        # 断言 p 值相等
        assert_equal(res.pvalue, ref.pvalue)

        # 扩展数据集大小并重新生成随机数种子
        x = rng.random(size=size*10)
        rng = np.random.default_rng(59234803482850134)
        # 使用置换法和指定的重采样次数进行 Wilcoxon 符号秩检验
        pm = stats.PermutationMethod(n_resamples=99, random_state=rng)
        ref = stats.wilcoxon(x, method=pm)
        rng = np.random.default_rng(59234803482850134)
        pm = stats.PermutationMethod(n_resamples=99, random_state=rng)
        # 使用置换法进行 Wilcoxon 符号秩检验
        res = stats.wilcoxon(x, method=pm)

        # 断言四舍五入后的 p 值与原始 p 值相等（用于验证重采样次数的使用）
        assert_equal(np.round(res.pvalue, 2), res.pvalue)
        # 断言 p 值相等（用于验证随机数种子的使用）
        assert_equal(res.pvalue, ref.pvalue)

    def test_method_auto_nan_propagate_ND_length_gt_50_gh20591(self):
        # 当 method 不是 'approx'，nan_policy 是 'propagate'，且多维数组切片包含 NaN 时，
        # wilcoxon 的结果对象可能会根据条件返回部分切片的 zstatistic，而其他切片则不返回。
        # 这可能导致错误，因为 apply_along_axis 可能会创建不规则数组。
        # 检查这个问题是否已经解决。
        rng = np.random.default_rng(235889269872456)
        # 创建一个大小为 (51, 2) 的正态分布数组，长度超过精确阈值
        A = rng.normal(size=(51, 2))
        # 将数组中某个位置设为 NaN
        A[5, 1] = np.nan
        # 使用默认方法进行 Wilcoxon 符号秩检验
        res = stats.wilcoxon(A)
        # 使用 'approx' 方法作为参考进行 Wilcoxon 符号秩检验
        ref = stats.wilcoxon(A, method='approx')
        # 断言两个对象在数值上接近
        assert_allclose(res, ref)
        # 断言参考对象具有 zstatistic 属性
        assert hasattr(ref, 'zstatistic')
        # 断言结果对象没有 zstatistic 属性
        assert not hasattr(res, 'zstatistic')

    @pytest.mark.parametrize('method', ['exact', 'approx'])
    def test_symmetry_gh19872_gh20752(self, method):
        # 检查单侧精确检验是否遵守所需的对称性。Bug 报告在 gh-19872 和 gh-20752 中；
        # gh-19872 中的示例较为简洁。
        var1 = [62, 66, 61, 68, 74, 62, 68, 62, 55, 59]
        var2 = [71, 71, 69, 61, 75, 71, 77, 72, 62, 65]
        # 使用指定方法进行 Wilcoxon 符号秩检验，alternative='less'
        ref = stats.wilcoxon(var1, var2, alternative='less', method=method)
        # 使用指定方法进行 Wilcoxon 符号秩检验，alternative='greater'
        res = stats.wilcoxon(var2, var1, alternative='greater', method=method)
        # 计算最大可能的检验统计量
        max_statistic = len(var1) * (len(var1) + 1) / 2
        # 断言检验统计量是整数
        assert int(res.statistic) != res.statistic
        # 断言两个检验统计量的数值接近
        assert_allclose(max_statistic - res.statistic, ref.statistic, rtol=1e-15)
        # 断言两个 p 值的数值接近
        assert_allclose(res.pvalue, ref.pvalue, rtol=1e-15)
# 数据用于 k 统计测试，来自于 R 语言 kStatistics 包的示例
# https://cran.r-project.org/web/packages/kStatistics/kStatistics.pdf
# 查看 nKS "Examples"

# 声明一个包含数据的列表 x_kstat，用于后续的统计测试
x_kstat = [16.34, 10.76, 11.84, 13.55, 15.85, 18.20, 7.51, 10.22, 12.52, 14.68,
           16.08, 19.43, 8.12, 11.20, 12.95, 14.77, 16.83, 19.80, 8.55, 11.58,
           12.10, 15.02, 16.83, 16.98, 19.92, 9.47, 11.68, 13.41, 15.35, 19.11]

# 使用 array_api_compatible 装饰器标记 TestKstat 类
@array_api_compatible
class TestKstat:
    # 定义测试方法 test_moments_normal_distribution，接受参数 xp
    def test_moments_normal_distribution(self, xp):
        # 设置随机数种子
        np.random.seed(32149)
        # 生成长度为 12345 的随机标准正态分布数据，并转换为 xp 的数组形式
        data = xp.asarray(np.random.randn(12345), dtype=xp.float64)
        # 计算数据的 k 统计量的前四个时刻，并转换为 xp 的数组
        moments = xp.asarray([stats.kstat(data, n) for n in [1, 2, 3, 4]])

        # 预期的 k 统计量结果，与 moments 进行比较，允许相对误差 rtol=1e-4
        expected = xp.asarray([0.011315, 1.017931, 0.05811052, 0.0754134],
                              dtype=data.dtype)
        xp_assert_close(moments, expected, rtol=1e-4)

        # 测试与 stats.moment 的等价性
        m1 = stats.moment(data, order=1)
        m2 = stats.moment(data, order=2)
        m3 = stats.moment(data, order=3)
        xp_assert_close(xp.asarray((m1, m2, m3)), expected[:-1], atol=0.02, rtol=1e-2)

    # 定义测试空输入的方法 test_empty_input，接受参数 xp
    def test_empty_input(self, xp):
        # 如果 xp 是 numpy，测试将会引发 SmallSampleWarning 警告
        if is_numpy(xp):
            with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
                res = stats.kstat(xp.asarray([]))
        else:
            # 对于 array_api_strict，忽略无效值错误
            with np.errstate(invalid='ignore'):
                res = stats.kstat(xp.asarray([]))
        # 断言结果与 xp 的 NaN 数组相等
        xp_assert_equal(res, xp.asarray(xp.nan))

    # 定义测试包含 NaN 输入的方法 test_nan_input，接受参数 xp
    def test_nan_input(self, xp):
        # 生成一个长度为 10 的浮点数数组，将索引为 6 的值设置为 NaN
        data = xp.arange(10.)
        data = xp.where(data == 6, xp.asarray(xp.nan), data)

        # 断言 k 统计量计算结果为 NaN
        xp_assert_equal(stats.kstat(data), xp.asarray(xp.nan))

    # 使用参数化标记，参数 n 取值为 [0, 4.001] 中的每个值
    @pytest.mark.parametrize('n', [0, 4.001])
    def test_kstat_bad_arg(self, n, xp):
        # 如果 n 大于 4 或小于 1，则抛出 ValueError 异常
        data = xp.arange(10)
        message = 'k-statistics only supported for 1<=n<=4'
        with pytest.raises(ValueError, match=message):
            stats.kstat(data, n=n)

    # 使用参数化标记，参数 case 包含不同的测试值对 (n, ref)
    @pytest.mark.parametrize('case', [(1, 14.02166666666667),
                                      (2, 12.65006954022974),
                                      (3, -1.447059503280798),
                                      (4, -141.6682291883626)])
    def test_against_R(self, case, xp):
        # 测试与 R kStatistics 计算的参考值对比
        # 对于每个 case，计算数据 x_kstat 的 k 统计量，并与 ref 进行比较
        n, ref = case
        res = stats.kstat(xp.asarray(x_kstat), n)
        xp_assert_close(res, xp.asarray(ref))


# 使用 array_api_compatible 装饰器标记 TestKstatVar 类
@array_api_compatible
class TestKstatVar:
    # 定义一个测试函数，测试对空输入的情况
    def test_empty_input(self, xp):
        # 创建一个空的 NumPy 数组
        x = xp.asarray([])
        # 检查当前使用的数组库是否为 NumPy
        if is_numpy(xp):
            # 如果是 NumPy，预期会发出警告 SmallSampleWarning，且匹配给定的警告消息
            with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
                # 计算统计量 kstatvar，并存储结果
                res = stats.kstatvar(x)
        else:
            # 如果不是 NumPy，设置无效值的错误状态为忽略，用于 array_api_strict
            with np.errstate(invalid='ignore'):
                # 计算统计量 kstatvar，并存储结果
                res = stats.kstatvar(x)
        # 使用 xp_assert_equal 检查计算结果与预期结果是否相等（均为 NaN）
        xp_assert_equal(res, xp.asarray(xp.nan))

    # 定义一个测试函数，测试对含 NaN 输入的情况
    def test_nan_input(self, xp):
        # 创建一个包含 NaN 的数据数组
        data = xp.arange(10.)
        data = xp.where(data == 6, xp.asarray(xp.nan), data)
        # 调用统计函数 kstat，预期结果为 NaN
        xp_assert_equal(stats.kstat(data), xp.asarray(xp.nan))

    # 标记为跳过的测试函数，仅在使用 NumPy 时有效，理由是 `n` 的输入验证与后端无关
    @skip_xp_backends(np_only=True,
                      reasons=['input validation of `n` does not depend on backend'])
    @pytest.mark.usefixtures("skip_xp_backends")
    def test_bad_arg(self):
        # 检查当 `n` 不为 1 或 2 时，是否会引发 ValueError 异常
        data = [1]
        n = 10
        message = 'Only n=1 or n=2 supported.'
        with pytest.raises(ValueError, match=message):
            # 调用统计函数 kstatvar，预期会引发 ValueError 异常
            stats.kstatvar(data, n=n)

    # 定义一个测试函数，与 R 和 MathWorld 的结果进行比较
    def test_against_R_mathworld(self, xp):
        # 在 https://mathworld.wolfram.com/k-Statistic.html 中找到的参考值进行测试
        n = len(x_kstat)
        k2 = 12.65006954022974  # 参考 TestKstat 中源代码
        k4 = -141.6682291883626

        # 计算统计量 kstatvar，使用 n=1 的预期结果与参考值进行比较
        res = stats.kstatvar(xp.asarray(x_kstat), 1)
        ref = k2 / n
        xp_assert_close(res, xp.asarray(ref))

        # 计算统计量 kstatvar，使用 n=2 的预期结果与参考值进行比较
        ref = (2*k2**2*n + (n-1)*k4) / (n * (n+1))
        xp_assert_close(res, xp.asarray(ref))
class TestPpccPlot:
    # 定义测试类 TestPpccPlot，用于测试 ppcc_plot 函数的各种情况

    def setup_method(self):
        # 在每个测试方法执行前设置初始条件
        self.x = _old_loggamma_rvs(5, size=500, random_state=7654321) + 5
        # 使用 _old_loggamma_rvs 生成长度为 500 的随机数列，并加上偏移量 5，赋给 self.x

    def test_basic(self):
        # 测试基本功能是否正常
        N = 5
        svals, ppcc = stats.ppcc_plot(self.x, -10, 10, N=N)
        # 调用 ppcc_plot 函数，计算 self.x 的 ppcc 值和对应的 svals
        ppcc_expected = [0.21139644, 0.21384059, 0.98766719, 0.97980182,
                         0.93519298]
        # 预期的 ppcc 值列表
        assert_allclose(svals, np.linspace(-10, 10, num=N))
        # 检查 svals 是否等于 -10 到 10 等间隔的 N 个值
        assert_allclose(ppcc, ppcc_expected)
        # 检查计算得到的 ppcc 是否与预期的 ppcc_expected 接近

    def test_dist(self):
        # 测试指定分布的功能
        svals1, ppcc1 = stats.ppcc_plot(self.x, -10, 10, dist='tukeylambda')
        # 使用字符串 'tukeylambda' 指定分布，并计算对应的 ppcc 值和 svals
        svals2, ppcc2 = stats.ppcc_plot(self.x, -10, 10,
                                        dist=stats.tukeylambda)
        # 使用 tukeylambda 分布对象来计算 ppcc 值和 svals
        assert_allclose(svals1, svals2, rtol=1e-20)
        # 检查两种方式计算的 svals 是否非常接近
        assert_allclose(ppcc1, ppcc2, rtol=1e-20)
        # 检查两种方式计算的 ppcc 是否非常接近
        svals3, ppcc3 = stats.ppcc_plot(self.x, -10, 10)
        # 测试默认情况下使用 'tukeylambda' 分布
        assert_allclose(svals1, svals3, rtol=1e-20)
        # 检查使用默认 'tukeylambda' 分布时的 svals 是否与第一次计算结果接近
        assert_allclose(ppcc1, ppcc3, rtol=1e-20)
        # 检查使用默认 'tukeylambda' 分布时的 ppcc 是否与第一次计算结果接近

    @pytest.mark.skipif(not have_matplotlib, reason="no matplotlib")
    def test_plot_kwarg(self):
        # 测试使用 matplotlib.pyplot 模块的功能
        fig = plt.figure()
        ax = fig.add_subplot(111)
        stats.ppcc_plot(self.x, -20, 20, plot=plt)
        # 使用 matplotlib 绘制 ppcc 图
        fig.delaxes(ax)
        # 删除子图

        ax = fig.add_subplot(111)
        stats.ppcc_plot(self.x, -20, 20, plot=ax)
        # 使用 Matplotlib Axes 对象绘制 ppcc 图
        plt.close()
        # 关闭图形

    def test_invalid_inputs(self):
        # 测试无效输入时的行为
        assert_raises(ValueError, stats.ppcc_plot, self.x, 1, 0)
        # 当 b 小于等于 a 时，应该抛出 ValueError 异常

        assert_raises(ValueError, stats.ppcc_plot, [1, 2, 3], 0, 1,
                      dist="plate_of_shrimp")
        # 当指定无效的分布时，应该抛出 ValueError 异常

    def test_empty(self):
        # 测试空数组输入时的行为
        svals, ppcc = stats.ppcc_plot([], 0, 1)
        # 对空数组进行 ppcc 计算
        assert_allclose(svals, np.linspace(0, 1, num=80))
        # 检查 svals 是否等于 0 到 1 之间的 80 个等间距值
        assert_allclose(ppcc, np.zeros(80, dtype=float))
        # 检查 ppcc 是否全为 0，并且类型为 float


class TestPpccMax:
    # 定义测试类 TestPpccMax，用于测试 ppcc_max 函数的各种情况

    def test_ppcc_max_bad_arg(self):
        # 测试给定无效分布时的行为
        data = [1]
        assert_raises(ValueError, stats.ppcc_max, data, dist="plate_of_shrimp")
        # 当指定无效的分布时，应该抛出 ValueError 异常

    def test_ppcc_max_basic(self):
        # 测试 ppcc_max 的基本功能
        x = stats.tukeylambda.rvs(-0.7, loc=2, scale=0.5, size=10000,
                                  random_state=1234567) + 1e4
        # 使用 tukeylambda 分布生成随机数列 x
        assert_almost_equal(stats.ppcc_max(x), -0.71215366521264145, decimal=7)
        # 检查计算得到的 ppcc_max 值是否接近给定的值，精确到小数点后 7 位
    def test_dist(self):
        x = stats.tukeylambda.rvs(-0.7, loc=2, scale=0.5, size=10000,
                                  random_state=1234567) + 1e4

        # 测试可以通过名称和对象两种方式指定分布
        max1 = stats.ppcc_max(x, dist='tukeylambda')
        max2 = stats.ppcc_max(x, dist=stats.tukeylambda)
        # 断言确保两种指定方式得到的结果几乎相等
        assert_almost_equal(max1, -0.71215366521264145, decimal=5)
        assert_almost_equal(max2, -0.71215366521264145, decimal=5)

        # 测试默认情况下使用 'tukeylambda' 分布
        max3 = stats.ppcc_max(x)
        # 断言确保默认情况下的结果几乎为预期值
        assert_almost_equal(max3, -0.71215366521264145, decimal=5)

    def test_brack(self):
        x = stats.tukeylambda.rvs(-0.7, loc=2, scale=0.5, size=10000,
                                  random_state=1234567) + 1e4
        # 断言应该引发 ValueError，因为 brack 参数不是长度为2的元组
        assert_raises(ValueError, stats.ppcc_max, x, brack=(0.0, 1.0, 0.5))

        # 断言使用正确的 brack 参数范围得到几乎预期的结果
        assert_almost_equal(stats.ppcc_max(x, brack=(0, 1)),
                            -0.71215366521264145, decimal=7)

        # 断言使用另一组 brack 参数范围得到几乎预期的结果
        assert_almost_equal(stats.ppcc_max(x, brack=(-2, 2)),
                            -0.71215366521264145, decimal=7)
class TestBoxcox_llf:
    
    # 定义测试类 TestBoxcox_llf

    def test_basic(self):
        # 基本测试用例
        x = stats.norm.rvs(size=10000, loc=10, random_state=54321)
        # 从正态分布中生成10000个随机数，均值为10，随机种子为54321
        lmbda = 1
        # 定义 lambda 参数为1
        llf = stats.boxcox_llf(lmbda, x)
        # 计算 Box-Cox 对数似然函数
        llf_expected = -x.size / 2. * np.log(np.sum(x.std()**2))
        # 计算预期的对数似然函数值
        assert_allclose(llf, llf_expected)
        # 使用 assert_allclose 函数比较计算得到的 llf 和预期值 llf_expected

    def test_array_like(self):
        # 测试处理类似数组的输入
        x = stats.norm.rvs(size=100, loc=10, random_state=54321)
        # 从正态分布中生成100个随机数，均值为10，随机种子为54321
        lmbda = 1
        # 定义 lambda 参数为1
        llf = stats.boxcox_llf(lmbda, x)
        # 计算 Box-Cox 对数似然函数
        llf2 = stats.boxcox_llf(lmbda, list(x))
        # 使用列表形式的 x 计算 Box-Cox 对数似然函数
        assert_allclose(llf, llf2, rtol=1e-12)
        # 使用 assert_allclose 函数比较两个计算得到的 llf 和 llf2，允许相对误差为1e-12

    def test_2d_input(self):
        # 测试处理二维输入
        # 注意：boxcox_llf() 已经可以处理二维输入（某种程度上），因此保持这种方式。
        #       boxcox() 无法处理二维输入，因为 brent() 返回一个标量。
        x = stats.norm.rvs(size=100, loc=10, random_state=54321)
        # 从正态分布中生成100个随机数，均值为10，随机种子为54321
        lmbda = 1
        # 定义 lambda 参数为1
        llf = stats.boxcox_llf(lmbda, x)
        # 计算 Box-Cox 对数似然函数
        llf2 = stats.boxcox_llf(lmbda, np.vstack([x, x]).T)
        # 使用堆叠后的 x 计算 Box-Cox 对数似然函数
        assert_allclose([llf, llf], llf2, rtol=1e-12)
        # 使用 assert_allclose 函数比较两个计算得到的 llf 和 llf2，允许相对误差为1e-12

    def test_empty(self):
        # 测试空输入的情况
        assert_(np.isnan(stats.boxcox_llf(1, [])))
        # 使用 assert_ 断言检查 stats.boxcox_llf(1, []) 是否为 NaN

    def test_gh_6873(self):
        # gh-6873 的回归测试
        # 此示例源自 gh-7534，gh-6873 的一个重复。
        data = [198.0, 233.0, 233.0, 392.0]
        # 数据列表
        llf = stats.boxcox_llf(-8, data)
        # 使用指定的 lambda 参数计算 Box-Cox 对数似然函数
        # 预期值是用 mpmath 计算的
        assert_allclose(llf, -17.93934208579061)
        # 使用 assert_allclose 函数比较计算得到的 llf 和预期值

    def test_instability_gh20021(self):
        # gh-20021 的不稳定性测试
        data = [2003, 1950, 1997, 2000, 2009]
        # 数据列表
        llf = stats.boxcox_llf(1e-8, data)
        # 使用极小的 lambda 参数计算 Box-Cox 对数似然函数
        # 预期值是用 mpsci 计算的，设置 mpmath.mp.dps=100
        assert_allclose(llf, -15.32401272869016598)
        # 使用 assert_allclose 函数比较计算得到的 llf 和预期值

# 这是来自 GitHub 用户 Qukaiyi 的数据，作为 boxcox 失败的示例数据集。
_boxcox_data = [
    15957, 112079, 1039553, 711775, 173111, 307382, 183155, 53366, 760875,
    207500, 160045, 473714, 40194, 440319, 133261, 265444, 155590, 36660,
    904939, 55108, 138391, 339146, 458053, 63324, 1377727, 1342632, 41575,
    68685, 172755, 63323, 368161, 199695, 538214, 167760, 388610, 398855,
    1001873, 364591, 1320518, 194060, 194324, 2318551, 196114, 64225, 272000,
    198668, 123585, 86420, 1925556, 695798, 88664, 46199, 759135, 28051,
    345094, 1977752, 51778, 82746, 638126, 2560910, 45830, 140576, 1603787,
    57371, 548730, 5343629, 2298913, 998813, 2156812, 423966, 68350, 145237,
    131935, 1600305, 342359, 111398, 1409144, 281007, 60314, 242004, 113418,
    246211, 61940, 95858, 957805, 40909, 307955, 174159, 124278, 241193,
    872614, 304180, 146719, 64361, 87478, 509360, 167169, 933479, 620561,
    483333, 97416, 143518, 286905, 597837, 2556043, 89065, 69944, 196858,
    88883, 49379, 916265, 1527392, 626954, 54415, 89013, 2883386, 106096,
    402697, 45578, 349852, 140379, 34648, 757343, 1305442, 2054757, 121232,
    606048, 101492, 51426, 1820833, 83412, 136349, 1379924, 505977, 1303486,
    95853, 146451, 285422, 2205423, 259020, 45864, 684547, 182014, 784334,
]
    # 定义一个长列表，包含了大量整数数据
    174793, 563068, 170745, 1195531, 63337, 71833, 199978, 2330904, 227335,
    898280, 75294, 2011361, 116771, 157489, 807147, 1321443, 1148635, 2456524,
    81839, 1228251, 97488, 1051892, 75397, 3009923, 2732230, 90923, 39735,
    132433, 225033, 337555, 1204092, 686588, 1062402, 40362, 1361829, 1497217,
    150074, 551459, 2019128, 39581, 45349, 1117187, 87845, 1877288, 164448,
    10338362, 24942, 64737, 769946, 2469124, 2366997, 259124, 2667585, 29175,
    56250, 74450, 96697, 5920978, 838375, 225914, 119494, 206004, 430907,
    244083, 219495, 322239, 407426, 618748, 2087536, 2242124, 4736149, 124624,
    406305, 240921, 2675273, 4425340, 821457, 578467, 28040, 348943, 48795,
    145531, 52110, 1645730, 1768364, 348363, 85042, 2673847, 81935, 169075,
    367733, 135474, 383327, 1207018, 93481, 5934183, 352190, 636533, 145870,
    55659, 146215, 73191, 248681, 376907, 1606620, 169381, 81164, 246390,
    236093, 885778, 335969, 49266, 381430, 307437, 350077, 34346, 49340,
    84715, 527120, 40163, 46898, 4609439, 617038, 2239574, 159905, 118337,
    120357, 430778, 3799158, 3516745, 54198, 2970796, 729239, 97848, 6317375,
    887345, 58198, 88111, 867595, 210136, 1572103, 1420760, 574046, 845988,
    509743, 397927, 1119016, 189955, 3883644, 291051, 126467, 1239907, 2556229,
    411058, 657444, 2025234, 1211368, 93151, 577594, 4842264, 1531713, 305084,
    479251, 20591, 1466166, 137417, 897756, 594767, 3606337, 32844, 82426,
    1294831, 57174, 290167, 322066, 813146, 5671804, 4425684, 895607, 450598,
    1048958, 232844, 56871, 46113, 70366, 701618, 97739, 157113, 865047,
    194810, 1501615, 1765727, 38125, 2733376, 40642, 437590, 127337, 106310,
    4167579, 665303, 809250, 1210317, 45750, 1853687, 348954, 156786, 90793,
    1885504, 281501, 3902273, 359546, 797540, 623508, 3672775, 55330, 648221,
    266831, 90030, 7118372, 735521, 1009925, 283901, 806005, 2434897, 94321,
    309571, 4213597, 2213280, 120339, 64403, 8155209, 1686948, 4327743,
    1868312, 135670, 3189615, 1569446, 706058, 58056, 2438625, 520619, 105201,
    141961, 179990, 1351440, 3148662, 2804457, 2760144, 70775, 33807, 1926518,
    2362142, 186761, 240941, 97860, 1040429, 1431035, 78892, 484039, 57845,
    724126, 3166209, 175913, 159211, 1182095, 86734, 1921472, 513546, 326016,
    1891609
    def test_fixed_lmbda(self):
        # 生成一个数组，包含从 _old_loggamma_rvs 函数返回的值
        x = _old_loggamma_rvs(5, size=50, random_state=12345) + 5
        # 使用指定的 lambda 值进行 Box-Cox 变换
        xt = stats.boxcox(x, lmbda=1)
        # 断言变换后的值与 x - 1 接近
        assert_allclose(xt, x - 1)
        # 使用指定的 lambda 值进行 Box-Cox 变换
        xt = stats.boxcox(x, lmbda=-1)
        # 断言变换后的值与 1 - 1/x 接近
        assert_allclose(xt, 1 - 1/x)

        # 使用指定的 lambda 值进行 Box-Cox 变换
        xt = stats.boxcox(x, lmbda=0)
        # 断言变换后的值与 ln(x) 接近
        assert_allclose(xt, np.log(x))

        # 也测试数组输入是否有效
        xt = stats.boxcox(list(x), lmbda=0)
        # 断言变换后的值与 ln(x) 接近
        assert_allclose(xt, np.log(x))

        # 测试常数输入是否被接受；参见 gh-12225
        xt = stats.boxcox(np.ones(10), 2)
        # 断言变换后的值为一个长度为 10 的零数组
        assert_equal(xt, np.zeros(10))

    def test_lmbda_None(self):
        # 从正态分布的随机变量开始，执行反向变换以检查优化函数是否接近正确答案
        lmbda = 2.5
        x = stats.norm.rvs(loc=10, size=50000, random_state=1245)
        x_inv = (x * lmbda + 1)**(-lmbda)
        xt, maxlog = stats.boxcox(x_inv)

        # 断言 maxlog 接近 -1 / lmbda，精确度为两位小数
        assert_almost_equal(maxlog, -1 / lmbda, decimal=2)

    def test_alpha(self):
        rng = np.random.RandomState(1234)
        x = _old_loggamma_rvs(5, size=50, random_state=rng) + 5

        # 一些常规的 alpha 值，用于小样本大小
        _, _, interval = stats.boxcox(x, alpha=0.75)
        # 断言置信区间接近给定值
        assert_allclose(interval, [4.004485780226041, 5.138756355035744])
        _, _, interval = stats.boxcox(x, alpha=0.05)
        # 断言置信区间接近给定值
        assert_allclose(interval, [1.2138178554857557, 8.209033272375663])

        # 尝试一些极端值，确保不超出 N=500 的限制
        x = _old_loggamma_rvs(7, size=500, random_state=rng) + 15
        _, _, interval = stats.boxcox(x, alpha=0.001)
        # 断言置信区间接近给定值
        assert_allclose(interval, [0.3988867, 11.40553131])
        _, _, interval = stats.boxcox(x, alpha=0.999)
        # 断言置信区间接近给定值
        assert_allclose(interval, [5.83316246, 5.83735292])

    def test_boxcox_bad_arg(self):
        # 如果任何数据值为负数，则引发 ValueError
        x = np.array([-1, 2])
        assert_raises(ValueError, stats.boxcox, x)
        # 如果数据是常数，则引发 ValueError
        assert_raises(ValueError, stats.boxcox, np.array([1]))
        # 如果数据不是一维的，则引发 ValueError
        assert_raises(ValueError, stats.boxcox, np.array([[1], [2]]))

    def test_empty(self):
        # 断言对空输入执行 Box-Cox 变换后返回的数组形状为 (0,)
        assert_(stats.boxcox([]).shape == (0,))

    def test_gh_6873(self):
        # gh-6873 的回归测试
        y, lam = stats.boxcox(_boxcox_data)
        # lam 的预期值是通过 R 库 'car' 中的 powerTransform 函数计算得出的
        # 保留大约五个有效数字的精度
        assert_allclose(lam, -0.051654, rtol=1e-5)

    @pytest.mark.parametrize("bounds", [(-1, 1), (1.1, 2), (-2, -1.1)])
    def test_bounded_optimizer_within_bounds(self, bounds):
        # 定义一个使用指定边界的自定义优化器函数
        def optimizer(fun):
            # 调用 scipy.optimize.minimize_scalar 函数进行标量优化，使用了给定的边界和方法
            return optimize.minimize_scalar(fun, bounds=bounds,
                                            method="bounded")

        # 对数据进行 Box-Cox 变换，使用自定义的优化器
        _, lmbda = stats.boxcox(_boxcox_data, lmbda=None, optimizer=optimizer)
        # 断言得到的 lambda 参数在指定的边界之内
        assert bounds[0] < lmbda < bounds[1]

    def test_bounded_optimizer_against_unbounded_optimizer(self):
        # 测试设置优化器边界是否能排除无界优化器的解

        # 获取无界解
        _, lmbda = stats.boxcox(_boxcox_data, lmbda=None)

        # 设置公差和围绕解的边界
        bounds = (lmbda + 0.1, lmbda + 1)
        options = {'xatol': 1e-12}

        def optimizer(fun):
            # 调用 scipy.optimize.minimize_scalar 函数进行标量优化，使用了给定的边界、方法和选项
            return optimize.minimize_scalar(fun, bounds=bounds,
                                            method="bounded", options=options)

        # 检查有界解。应该激活较低的边界。
        _, lmbda_bounded = stats.boxcox(_boxcox_data, lmbda=None,
                                        optimizer=optimizer)
        # 断言有界解与无界解不同
        assert lmbda_bounded != lmbda
        # 断言有界解接近于边界的较低值
        assert_allclose(lmbda_bounded, bounds[0])

    @pytest.mark.parametrize("optimizer", ["str", (1, 2), 0.1])
    def test_bad_optimizer_type_raises_error(self, optimizer):
        # 检查如果传递字符串、元组或浮点数作为优化器，是否会引发错误
        with pytest.raises(ValueError, match="`optimizer` must be a callable"):
            # 调用 Box-Cox 变换函数，使用了非法的优化器类型
            stats.boxcox(_boxcox_data, lmbda=None, optimizer=optimizer)

    def test_bad_optimizer_value_raises_error(self):
        # 检查如果优化器函数不返回 OptimizeResult 对象，是否会引发错误

        # 定义一个始终返回 1 的测试函数
        def optimizer(fun):
            return 1

        message = "return an object containing the optimal `lmbda`"
        with pytest.raises(ValueError, match=message):
            # 调用 Box-Cox 变换函数，使用了无效的优化器函数
            stats.boxcox(_boxcox_data, lmbda=None, optimizer=optimizer)

    @pytest.mark.parametrize(
            "bad_x", [np.array([1, -42, 12345.6]), np.array([np.nan, 42, 1])]
        )
    def test_negative_x_value_raises_error(self, bad_x):
        """Test boxcox_normmax raises ValueError if x contains non-positive values."""
        message = "only positive, finite, real numbers"
        with pytest.raises(ValueError, match=message):
            # 调用 boxcox_normmax 函数，传递包含非正数值的 x，应该引发错误
            stats.boxcox_normmax(bad_x)

    @pytest.mark.parametrize('x', [
        # 尝试触发幂表达式的溢出。
        np.array([2003.0, 1950.0, 1997.0, 2000.0, 2009.0,
                  2009.0, 1980.0, 1999.0, 2007.0, 1991.0]),
        # 尝试使用大的 lambda 触发溢出。
        np.array([2003.0, 1950.0, 1997.0, 2000.0, 2009.0]),
        # 尝试使用大数据触发溢出。
        np.array([2003.0e200, 1950.0e200, 1997.0e200, 2000.0e200, 2009.0e200])
    ])
    # 定义一个测试方法，用于测试输入数据是否会导致溢出
    def test_overflow(self, x):
        # 使用 pytest 来捕获 UserWarning 类型的警告，并匹配包含特定文本 "The optimal lambda is" 的警告消息
        with pytest.warns(UserWarning, match="The optimal lambda is"):
            # 对输入数据 x 应用 Box-Cox 变换，返回变换后的数据 xt_bc 和计算得到的 lambda 值 lam_bc
            xt_bc, lam_bc = stats.boxcox(x)
            # 断言变换后的数据 xt_bc 中的所有元素都是有限的
            assert np.all(np.isfinite(xt_bc))
class TestBoxcoxNormmax:
    # 设置测试方法的初始化，生成一个带有随机状态的数据集 self.x
    def setup_method(self):
        self.x = _old_loggamma_rvs(5, size=50, random_state=12345) + 5

    # 测试使用 pearsonr 方法计算最大对数似然时的结果
    def test_pearsonr(self):
        maxlog = stats.boxcox_normmax(self.x)
        assert_allclose(maxlog, 1.804465, rtol=1e-6)

    # 测试使用 mle 方法计算最大对数似然时的结果，并验证 boxcox() 方法的结果与之一致
    def test_mle(self):
        maxlog = stats.boxcox_normmax(self.x, method='mle')
        assert_allclose(maxlog, 1.758101, rtol=1e-6)

        _, maxlog_boxcox = stats.boxcox(self.x)
        assert_allclose(maxlog_boxcox, maxlog)

    # 测试使用 all 方法计算所有方法下的最大对数似然时的结果
    def test_all(self):
        maxlog_all = stats.boxcox_normmax(self.x, method='all')
        assert_allclose(maxlog_all, [1.804465, 1.758101], rtol=1e-6)

    # 参数化测试，测试不同优化器和边界条件下的最大对数似然值
    @pytest.mark.parametrize("method", ["mle", "pearsonr", "all"])
    @pytest.mark.parametrize("bounds", [(-1, 1), (1.1, 2), (-2, -1.1)])
    def test_bounded_optimizer_within_bounds(self, method, bounds):

        # 定义一个优化器函数，使用 scipy.optimize.minimize_scalar 进行标量最小化
        def optimizer(fun):
            return optimize.minimize_scalar(fun, bounds=bounds,
                                            method="bounded")

        # 测试使用不同方法和优化器条件下的最大对数似然值
        maxlog = stats.boxcox_normmax(self.x, method=method,
                                      optimizer=optimizer)
        assert np.all(bounds[0] < maxlog)
        assert np.all(maxlog < bounds[1])

    # 标记为慢速测试，测试用户自定义优化器的效果
    @pytest.mark.slow
    def test_user_defined_optimizer(self):
        # 测试使用非基于 scipy.optimize.minimize 的优化器得到的最大对数似然值
        lmbda = stats.boxcox_normmax(self.x)
        lmbda_rounded = np.round(lmbda, 5)
        lmbda_range = np.linspace(lmbda_rounded-0.01, lmbda_rounded+0.01, 1001)

        # 定义一个自定义的优化器函数，通过暴力搜索在一定范围内寻找最小值
        class MyResult:
            pass

        def optimizer(fun):
            objs = []
            for lmbda in lmbda_range:
                objs.append(fun(lmbda))
            res = MyResult()
            res.x = lmbda_range[np.argmin(objs)]
            return res

        # 测试使用自定义优化器条件下得到的最大对数似然值，并验证其接近预期值
        lmbda2 = stats.boxcox_normmax(self.x, optimizer=optimizer)
        assert lmbda2 != lmbda
        assert_allclose(lmbda2, lmbda, 1e-5)

    # 测试使用用户自定义优化器和 brack 参数时是否会引发错误
    def test_user_defined_optimizer_and_brack_raises_error(self):
        optimizer = optimize.minimize_scalar

        # 测试使用默认的 brack=None 和用户定义的 optimizer 是否正常工作
        stats.boxcox_normmax(self.x, brack=None, optimizer=optimizer)

        # 测试使用用户自定义的 brack 参数和 optimizer 是否会抛出错误
        with pytest.raises(ValueError, match="`brack` must be None if "
                                             "`optimizer` is given"):
            stats.boxcox_normmax(self.x, brack=(-2.0, 2.0),
                                 optimizer=optimizer)
    # 使用 pytest 的 parametrize 装饰器，为 test_overflow 函数参数化测试数据
    @pytest.mark.parametrize(
        'x', ([2003.0, 1950.0, 1997.0, 2000.0, 2009.0],
              [0.50000471, 0.50004979, 0.50005902, 0.50009312, 0.50001632]))
    # 定义 test_overflow 测试函数，接受参数 x
    def test_overflow(self, x):
        # 设置警告信息
        message = "The optimal lambda is..."
        # 使用 pytest 的 warns 上下文管理器，检查是否触发 UserWarning 并匹配特定消息
        with pytest.warns(UserWarning, match=message):
            # 调用 stats 模块的 boxcox_normmax 函数，使用 'mle' 方法计算最佳 lambda
            lmbda = stats.boxcox_normmax(x, method='mle')
        # 断言转换后的数据是否都是有限的
        assert np.isfinite(special.boxcox(x, lmbda)).all()
        # 设置最大值的安全因子为 10000，用于 boxcox_normmax 函数
        ymax = np.finfo(np.float64).max / 10000
        # 根据 lambda 值选择 x 中的最大值或最小值
        x_treme = np.max(x) if lmbda > 0 else np.min(x)
        # 使用 boxcox 函数对 x_treme 进行转换
        y_extreme = special.boxcox(x_treme, lmbda)
        # 断言转换结果与 ymax 乘以 lambda 的符号的近似相等性
        assert_allclose(y_extreme, ymax * np.sign(lmbda))

    # 定义 test_negative_ymax 测试函数，测试当 ymax 为负数时是否触发 ValueError
    def test_negative_ymax(self):
        with pytest.raises(ValueError, match="`ymax` must be strictly positive"):
            # 调用 stats 模块的 boxcox_normmax 函数，传入负数 ymax
            stats.boxcox_normmax(self.x, ymax=-1)

    # 使用 pytest 的 parametrize 装饰器，为 test_user_defined_ymax_input_float64_32 函数参数化测试数据
    @pytest.mark.parametrize("x", [
        # 测试 float64 中的正数溢出
        np.array([2003.0, 1950.0, 1997.0, 2000.0, 2009.0], dtype=np.float64),
        # 测试 float64 中的负数溢出
        np.array([0.50000471, 0.50004979, 0.50005902, 0.50009312, 0.50001632], dtype=np.float64),
        # 测试 float32 中的正数溢出
        np.array([200.3, 195.0, 199.7, 200.0, 200.9], dtype=np.float32),
        # 测试 float32 中的负数溢出
        np.array([2e-30, 1e-30, 1e-30, 1e-30, 1e-30, 1e-30], dtype=np.float32),
    ])
    # 使用 pytest 的 parametrize 装饰器，为 ymax 参数参数化测试数据
    @pytest.mark.parametrize("ymax", [1e10, 1e30, None])
    # TODO: 在修复溢出问题后，添加 method 参数 "pearsonr"
    @pytest.mark.parametrize("method", ["mle"])
    # 定义 test_user_defined_ymax_input_float64_32 测试函数，接受参数 x, ymax, method
    def test_user_defined_ymax_input_float64_32(self, x, ymax, method):
        # 测试转换数据的最大值接近 ymax
        with pytest.warns(UserWarning, match="The optimal lambda is"):
            # 根据 method 和 ymax 参数调用 stats 模块的 boxcox_normmax 函数计算 lambda
            kwarg = {'ymax': ymax} if ymax is not None else {}
            lmb = stats.boxcox_normmax(x, method=method, **kwarg)
            # 根据 lambda 值选择 x 中的最小值和最大值
            x_treme = [np.min(x), np.max(x)]
            # 计算转换后的最大值的绝对值
            ymax_res = max(abs(stats.boxcox(x_treme, lmb)))
            # 如果 ymax 为 None，使用 x 的数据类型的最大值除以 10000 作为 ymax
            if ymax is None:
                ymax = np.finfo(x.dtype).max / 10000
            # 断言计算得到的 ymax 与实际结果的近似性，相对误差限制为 1e-5
            assert_allclose(ymax, ymax_res, rtol=1e-5)

    # 使用 pytest 的 parametrize 装饰器，为 test_user_defined_ymax_input_float64_32 函数参数化测试数据
    @pytest.mark.parametrize("x", [
        # 测试 float32 中的正数溢出但不在 float64 中
        [200.3, 195.0, 199.7, 200.0, 200.9],
        # 测试 float32 中的负数溢出但不在 float64 中
        [2e-30, 1e-30, 1e-30, 1e-30, 1e-30, 1e-30],
    ])
    # TODO: 在修复溢出问题后，添加 method 参数 "pearsonr"
    @pytest.mark.parametrize("method", ["mle"])
    # 定义一个测试函数，用于测试 boxcox_normmax 函数对不同数据类型和方法的行为
    def test_user_defined_ymax_inf(self, x, method):
        # 将输入数据转换为 numpy 数组，其中 x_32 是 float32 类型，x_64 是 float64 类型
        x_32 = np.asarray(x, dtype=np.float32)
        x_64 = np.asarray(x, dtype=np.float64)

        # 断言在 float32 类型下可能会溢出，但在 float64 类型下不会溢出，产生 UserWarning
        with pytest.warns(UserWarning, match="The optimal lambda is"):
            stats.boxcox_normmax(x_32, method=method)
        
        # 调用 boxcox_normmax 函数计算 float64 类型数据的最优 lambda
        stats.boxcox_normmax(x_64, method=method)

        # 计算 ymax 为无穷时的真实最优 lambda，并比较两种数据类型下的结果
        lmb_32 = stats.boxcox_normmax(x_32, ymax=np.inf, method=method)
        lmb_64 = stats.boxcox_normmax(x_64, ymax=np.inf, method=method)
        # 断言两种数据类型下计算出的最优 lambda 接近，相对误差在 1e-2 范围内
        assert_allclose(lmb_32, lmb_64, rtol=1e-2)
class TestBoxcoxNormplot:
    # 测试类 TestBoxcoxNormplot，用于测试 boxcox_normplot 函数的各种情况

    def setup_method(self):
        # 在每个测试方法运行前设置初始化条件
        self.x = _old_loggamma_rvs(5, size=500, random_state=7654321) + 5
        # 生成长度为 500 的随机数据，应用 loggamma 分布进行变换，并加上常数 5

    def test_basic(self):
        # 基本测试方法，测试 boxcox_normplot 函数的基本功能
        N = 5
        # 设定 N 的值为 5
        lmbdas, ppcc = stats.boxcox_normplot(self.x, -10, 10, N=N)
        # 使用 boxcox_normplot 函数生成 lmbdas 和 ppcc，指定参数范围和 N 的值
        ppcc_expected = [0.57783375, 0.83610988, 0.97524311, 0.99756057,
                         0.95843297]
        # 预期的 ppcc 值
        assert_allclose(lmbdas, np.linspace(-10, 10, num=N))
        # 断言 lmbdas 应接近于从 -10 到 10 等间距生成的 N 个值
        assert_allclose(ppcc, ppcc_expected)
        # 断言 ppcc 应接近于预期的 ppcc_expected 值列表

    @pytest.mark.skipif(not have_matplotlib, reason="no matplotlib")
    def test_plot_kwarg(self):
        # 测试带有 plot 关键字参数的情况，使用 matplotlib.pyplot 模块进行检查
        fig = plt.figure()
        ax = fig.add_subplot(111)
        stats.boxcox_normplot(self.x, -20, 20, plot=plt)
        # 调用 boxcox_normplot 函数，使用 plt 对象进行绘图
        fig.delaxes(ax)
        # 删除图形中的 ax 坐标轴对象

        # 检查是否接受 Matplotlib Axes 对象
        ax = fig.add_subplot(111)
        stats.boxcox_normplot(self.x, -20, 20, plot=ax)
        # 调用 boxcox_normplot 函数，使用 ax 对象进行绘图
        plt.close()
        # 关闭当前图形

    def test_invalid_inputs(self):
        # 测试无效输入的情况
        # lb 参数必须大于 la 参数
        assert_raises(ValueError, stats.boxcox_normplot, self.x, 1, 0)
        # x 参数不能包含负值
        assert_raises(ValueError, stats.boxcox_normplot, [-1, 1], 0, 1)

    def test_empty(self):
        # 测试空输入的情况
        assert_(stats.boxcox_normplot([], 0, 1).size == 0)
        # 断言调用 boxcox_normplot 函数，传入空列表 []，返回的结果 size 应该为 0


class TestYeojohnson_llf:
    # 测试类 TestYeojohnson_llf，用于测试 yeojohnson_llf 函数的各种情况

    def test_array_like(self):
        # 测试类数组输入的情况
        x = stats.norm.rvs(size=100, loc=0, random_state=54321)
        # 生成均值为 0 的正态分布随机数据，长度为 100
        lmbda = 1
        # 设定 lmbda 的值为 1
        llf = stats.yeojohnson_llf(lmbda, x)
        # 调用 yeojohnson_llf 函数计算 llf
        llf2 = stats.yeojohnson_llf(lmbda, list(x))
        # 再次调用 yeojohnson_llf 函数，传入 x 的列表形式计算 llf2
        assert_allclose(llf, llf2, rtol=1e-12)
        # 断言 llf 和 llf2 应接近，相对误差小于 1e-12

    def test_2d_input(self):
        # 测试二维输入的情况
        x = stats.norm.rvs(size=100, loc=10, random_state=54321)
        # 生成均值为 10 的正态分布随机数据，长度为 100
        lmbda = 1
        # 设定 lmbda 的值为 1
        llf = stats.yeojohnson_llf(lmbda, x)
        # 调用 yeojohnson_llf 函数计算 llf
        llf2 = stats.yeojohnson_llf(lmbda, np.vstack([x, x]).T)
        # 再次调用 yeojohnson_llf 函数，传入 x 的垂直堆叠的转置形式计算 llf2
        assert_allclose([llf, llf], llf2, rtol=1e-12)
        # 断言 [llf, llf] 和 llf2 应接近，相对误差小于 1e-12

    def test_empty(self):
        # 测试空输入的情况
        assert_(np.isnan(stats.yeojohnson_llf(1, [])))
        # 断言调用 yeojohnson_llf 函数，传入空列表 []，返回的结果应为 NaN


class TestYeojohnson:
    # 测试类 TestYeojohnson，用于测试 yeojohnson 函数的各种情况
    # 定义一个测试方法，测试指定参数lambda的情况
    def test_fixed_lmbda(self):
        # 创建一个伪随机数生成器，用于生成可重现的随机数序列
        rng = np.random.RandomState(12345)

        # 测试正数输入情况
        # 生成服从旧版本loggamma分布的随机数，并加上5
        x = _old_loggamma_rvs(5, size=50, random_state=rng) + 5
        # 断言所有生成的随机数大于0
        assert np.all(x > 0)
        # 对x应用Yeojohnson变换，lambda参数为1，期望输出与输入一致
        xt = stats.yeojohnson(x, lmbda=1)
        assert_allclose(xt, x)
        # 对x应用Yeojohnson变换，lambda参数为-1，期望输出满足特定计算公式
        xt = stats.yeojohnson(x, lmbda=-1)
        assert_allclose(xt, 1 - 1 / (x + 1))
        # 对x应用Yeojohnson变换，lambda参数为0，期望输出满足特定计算公式
        xt = stats.yeojohnson(x, lmbda=0)
        assert_allclose(xt, np.log(x + 1))
        # 对x应用Yeojohnson变换，lambda参数为1，期望输出与输入一致
        xt = stats.yeojohnson(x, lmbda=1)
        assert_allclose(xt, x)

        # 测试负数输入情况
        # 生成服从旧版本loggamma分布的随机数，并减去5
        x = _old_loggamma_rvs(5, size=50, random_state=rng) - 5
        # 断言所有生成的随机数小于0
        assert np.all(x < 0)
        # 对x应用Yeojohnson变换，lambda参数为2，期望输出满足特定计算公式
        xt = stats.yeojohnson(x, lmbda=2)
        assert_allclose(xt, -np.log(-x + 1))
        # 对x应用Yeojohnson变换，lambda参数为1，期望输出与输入一致
        xt = stats.yeojohnson(x, lmbda=1)
        assert_allclose(xt, x)
        # 对x应用Yeojohnson变换，lambda参数为3，期望输出满足特定计算公式
        xt = stats.yeojohnson(x, lmbda=3)
        assert_allclose(xt, 1 / (-x + 1) - 1)

        # 测试既有正数又有负数输入情况
        # 生成服从旧版本loggamma分布的随机数，并减去2
        x = _old_loggamma_rvs(5, size=50, random_state=rng) - 2
        # 断言并非所有随机数都小于0
        assert not np.all(x < 0)
        # 断言并非所有随机数都大于等于0
        assert not np.all(x >= 0)
        # 筛选出正数部分
        pos = x >= 0
        # 对正数部分应用Yeojohnson变换，lambda参数为1，期望输出与输入一致
        xt = stats.yeojohnson(x, lmbda=1)
        assert_allclose(xt[pos], x[pos])
        # 对正数部分应用Yeojohnson变换，lambda参数为-1，期望输出满足特定计算公式
        xt = stats.yeojohnson(x, lmbda=-1)
        assert_allclose(xt[pos], 1 - 1 / (x[pos] + 1))
        # 对正数部分应用Yeojohnson变换，lambda参数为0，期望输出满足特定计算公式
        xt = stats.yeojohnson(x, lmbda=0)
        assert_allclose(xt[pos], np.log(x[pos] + 1))
        # 对正数部分应用Yeojohnson变换，lambda参数为1，期望输出与输入一致
        xt = stats.yeojohnson(x, lmbda=1)
        assert_allclose(xt[pos], x[pos])

        # 筛选出负数部分
        neg = ~pos
        # 对负数部分应用Yeojohnson变换，lambda参数为2，期望输出满足特定计算公式
        xt = stats.yeojohnson(x, lmbda=2)
        assert_allclose(xt[neg], -np.log(-x[neg] + 1))
        # 对负数部分应用Yeojohnson变换，lambda参数为1，期望输出与输入一致
        xt = stats.yeojohnson(x, lmbda=1)
        assert_allclose(xt[neg], x[neg])
        # 对负数部分应用Yeojohnson变换，lambda参数为3，期望输出满足特定计算公式
        xt = stats.yeojohnson(x, lmbda=3)
        assert_allclose(xt[neg], 1 / (-x[neg] + 1) - 1)
    # 测试以 None 为 lambda 参数的情况
    def test_lmbda_None(self, lmbda):
        # 从正态分布随机变量开始，进行逆变换以检查优化函数是否接近正确答案。

        # 定义逆变换函数
        def _inverse_transform(x, lmbda):
            # 创建与 x 形状相同的零数组
            x_inv = np.zeros(x.shape, dtype=x.dtype)
            # 标记 x 中大于等于 0 的位置
            pos = x >= 0

            # 当 x >= 0 时
            if abs(lmbda) < np.spacing(1.):
                # 如果 lambda 接近于 0，使用指数函数进行逆变换
                x_inv[pos] = np.exp(x[pos]) - 1
            else:  # 如果 lambda 不等于 0
                # 使用 Yeo-Johnson 变换公式进行逆变换
                x_inv[pos] = np.power(x[pos] * lmbda + 1, 1 / lmbda) - 1

            # 当 x < 0 时
            if abs(lmbda - 2) > np.spacing(1.):
                # 如果 lambda 不接近 2，使用另一种 Yeo-Johnson 变换公式进行逆变换
                x_inv[~pos] = 1 - np.power(-(2 - lmbda) * x[~pos] + 1,
                                           1 / (2 - lmbda))
            else:  # 如果 lambda 等于 2
                # 使用特定公式进行逆变换
                x_inv[~pos] = 1 - np.exp(-x[~pos])

            return x_inv

        # 设置随机种子和样本数
        n_samples = 20000
        np.random.seed(1234567)
        # 从正态分布生成随机样本 x
        x = np.random.normal(loc=0, scale=1, size=(n_samples))

        # 对随机样本 x 进行逆变换
        x_inv = _inverse_transform(x, lmbda)
        # 对逆变换后的结果 x_inv 进行 Yeo-Johnson 变换，返回变换后的结果和 lambda 值
        xt, maxlog = stats.yeojohnson(x_inv)

        # 断言 Yeo-Johnson 变换后的 lambda 值接近预期值 lmbda
        assert_allclose(maxlog, lmbda, atol=1e-2)

        # 断言变换后的数据 xt 与原始数据 x 的归一化误差接近于 0
        assert_almost_equal(0, np.linalg.norm(x - xt) / n_samples, decimal=2)
        # 断言变换后的数据 xt 的均值接近于 0
        assert_almost_equal(0, xt.mean(), decimal=1)
        # 断言变换后的数据 xt 的标准差接近于 1
        assert_almost_equal(1, xt.std(), decimal=1)
    def test_input_high_variance(self):
        # 对于 GitHub 问题编号 10821 的非回归测试
        # 创建包含高方差输入的数组 x
        x = np.array([3251637.22, 620695.44, 11642969.00, 2223468.22,
                      85307500.00, 16494389.89, 917215.88, 11642969.00,
                      2145773.87, 4962000.00, 620695.44, 651234.50,
                      1907876.71, 4053297.88, 3251637.22, 3259103.08,
                      9547969.00, 20631286.23, 12807072.08, 2383819.84,
                      90114500.00, 17209575.46, 12852969.00, 2414609.99,
                      2170368.23])
        # 使用 Yeojohnson 变换对 x 进行转换
        xt_yeo, lam_yeo = stats.yeojohnson(x)
        # 使用 Box-Cox 变换对 x 做平移后转换
        xt_box, lam_box = stats.boxcox(x + 1)
        # 断言两种变换的结果在指定的相对误差范围内相等
        assert_allclose(xt_yeo, xt_box, rtol=1e-6)
        # 断言两种变换的 lambda 值在指定的相对误差范围内相等
        assert_allclose(lam_yeo, lam_box, rtol=1e-6)

    @pytest.mark.parametrize('x', [
        np.array([1.0, float("nan"), 2.0]),
        np.array([1.0, float("inf"), 2.0]),
        np.array([1.0, -float("inf"), 2.0]),
        np.array([-1.0, float("nan"), float("inf"), -float("inf"), 1.0])
    ])
    def test_nonfinite_input(self, x):
        # 使用 pytest 框架测试非有限输入的情况
        with pytest.raises(ValueError, match='Yeo-Johnson input must be finite'):
            # 断言 Yeo-Johnson 变换对于非有限输入会引发 ValueError
            xt_yeo, lam_yeo = stats.yeojohnson(x)

    @pytest.mark.parametrize('x', [
        # 尝试触发幂表达式中的溢出
        np.array([2003.0, 1950.0, 1997.0, 2000.0, 2009.0,
                  2009.0, 1980.0, 1999.0, 2007.0, 1991.0]),
        # 尝试使用一个大的最优 lambda 触发溢出
        np.array([2003.0, 1950.0, 1997.0, 2000.0, 2009.0]),
        # 尝试使用大数据触发溢出
        np.array([2003.0e200, 1950.0e200, 1997.0e200, 2000.0e200, 2009.0e200])
    ])
    def test_overflow(self, x):
        # 对于 GitHub 问题编号 18389 的非回归测试
        # 定义优化函数，用于参数化 lambda_yeo
        def optimizer(fun, lam_yeo):
            # 使用 optimize.fminbound 进行优化
            out = optimize.fminbound(fun, -lam_yeo, lam_yeo, xtol=1.48e-08)
            result = optimize.OptimizeResult()
            result.x = out
            return result

        # 在 NumPy 错误状态下运行以下代码块
        with np.errstate(all="raise"):
            # 对输入 x 进行 Yeo-Johnson 变换
            xt_yeo, lam_yeo = stats.yeojohnson(x)
            # 使用 Box-Cox 变换并利用 optimizer 函数对 lambda 进行优化
            xt_box, lam_box = stats.boxcox(
                x + 1, optimizer=partial(optimizer, lam_yeo=lam_yeo))
            # 断言 Yeo-Johnson 变换后的数据方差是有限的
            assert np.isfinite(np.var(xt_yeo))
            # 断言 Box-Cox 变换后的数据方差是有限的
            assert np.isfinite(np.var(xt_box))
            # 断言两种变换的 lambda 值在指定的相对误差范围内相等
            assert_allclose(lam_yeo, lam_box, rtol=1e-6)
            # 断言两种变换的结果在较宽松的相对误差范围内相等
            assert_allclose(xt_yeo, xt_box, rtol=1e-4)

    @pytest.mark.parametrize('x', [
        np.array([2003.0, 1950.0, 1997.0, 2000.0, 2009.0,
                  2009.0, 1980.0, 1999.0, 2007.0, 1991.0]),
        np.array([2003.0, 1950.0, 1997.0, 2000.0, 2009.0])
    ])
    @pytest.mark.parametrize('scale', [1, 1e-12, 1e-32, 1e-150, 1e32, 1e200])
    @pytest.mark.parametrize('sign', [1, -1])
    def test_overflow_underflow_signed_data(self, x, scale, sign):
        # 针对 gh-18389 的非回归测试
        # 设置错误状态处理，确保所有错误都会抛出异常
        with np.errstate(all="raise"):
            # 对数据进行 Yeo-Johnson 转换，处理数据溢出和下溢
            xt_yeo, lam_yeo = stats.yeojohnson(sign * x * scale)
            # 断言转换后的数据符号与原始数据符号相同
            assert np.all(np.sign(sign * x) == np.sign(xt_yeo))
            # 断言 lam_yeo 是有限的
            assert np.isfinite(lam_yeo)
            # 断言转换后的数据方差是有限的
            assert np.isfinite(np.var(xt_yeo))

    @pytest.mark.parametrize('x', [
        np.array([0, 1, 2, 3]),
        np.array([0, -1, 2, -3]),
        np.array([0, 0, 0])
    ])
    @pytest.mark.parametrize('sign', [1, -1])
    @pytest.mark.parametrize('brack', [None, (-2, 2)])
    def test_integer_signed_data(self, x, sign, brack):
        # 设置错误状态处理，确保所有错误都会抛出异常
        with np.errstate(all="raise"):
            # 将整数数组乘以符号值，得到有符号整数
            x_int = sign * x
            # 将有符号整数数组转换为浮点数数组
            x_float = x_int.astype(np.float64)
            # 计算整数数据的 Yeo-Johnson 转换的最大 lambda 值
            lam_yeo_int = stats.yeojohnson_normmax(x_int, brack=brack)
            # 对整数数据进行 Yeo-Johnson 转换
            xt_yeo_int = stats.yeojohnson(x_int, lmbda=lam_yeo_int)
            # 计算浮点数数据的 Yeo-Johnson 转换的最大 lambda 值
            lam_yeo_float = stats.yeojohnson_normmax(x_float, brack=brack)
            # 对浮点数数据进行 Yeo-Johnson 转换
            xt_yeo_float = stats.yeojohnson(x_float, lmbda=lam_yeo_float)
            # 断言整数数据转换后的符号与原始数据符号相同
            assert np.all(np.sign(x_int) == np.sign(xt_yeo_int))
            # 断言整数数据的 Yeo-Johnson 转换的 lambda 值是有限的
            assert np.isfinite(lam_yeo_int)
            # 断言整数数据转换后的方差是有限的
            assert np.isfinite(np.var(xt_yeo_int))
            # 断言整数数据和浮点数数据的最大 lambda 值相等
            assert lam_yeo_int == lam_yeo_float
            # 断言整数数据和浮点数数据的 Yeo-Johnson 转换结果相等
            assert np.all(xt_yeo_int == xt_yeo_float)
class TestYeojohnsonNormmax:
    # 定义测试类 TestYeojohnsonNormmax

    def setup_method(self):
        # 设置测试方法，初始化 self.x
        self.x = _old_loggamma_rvs(5, size=50, random_state=12345) + 5

    def test_mle(self):
        # 测试最大似然估计方法
        maxlog = stats.yeojohnson_normmax(self.x)
        # 断言最大似然估计的结果与预期值接近
        assert_allclose(maxlog, 1.876393, rtol=1e-6)

    def test_darwin_example(self):
        # 测试来自原始论文 "A new family of power transformations to
        # improve normality or symmetry" by Yeo and Johnson 的示例
        x = [6.1, -8.4, 1.0, 2.0, 0.7, 2.9, 3.5, 5.1, 1.8, 3.6, 7.0, 3.0, 9.3,
             7.5, -6.0]
        # 计算 Yeo-Johnson 转换的最大似然估计值
        lmbda = stats.yeojohnson_normmax(x)
        # 断言计算结果与预期值接近
        assert np.allclose(lmbda, 1.305, atol=1e-3)


@array_api_compatible
class TestCircFuncs:
    # 定义测试类 TestCircFuncs，支持数组 API 兼容性

    # 在 gh-5747 中，R 包 `circular` 用于计算圆形方差的参考值，例如：
    # library(circular)
    # options(digits=16)
    # x = c(0, 2*pi/3, 5*pi/3)
    # var.circular(x)
    @pytest.mark.parametrize("test_func,expected",
                             [(stats.circmean, 0.167690146),
                              (stats.circvar, 0.006455174000787767),
                              (stats.circstd, 6.520702116)])
    def test_circfuncs(self, test_func, expected, xp):
        # 参数化测试，使用不同的统计函数和预期值
        x = xp.asarray([355., 5., 2., 359., 10., 350.])
        # 断言使用统计函数计算结果与预期值接近
        xp_assert_close(test_func(x, high=360), xp.asarray(expected))

    def test_circfuncs_small(self, xp):
        # 测试小范围输入情况
        # 默认的容差不适用于这里，因为参考值是近似值。确保所有数组类型都以 float64 工作，以避免需要分别设置 float32 和 float64 的容差。
        x = xp.asarray([20, 21, 22, 18, 19, 20.5, 19.2], dtype=xp.float64)
        M1 = xp.mean(x)
        M2 = stats.circmean(x, high=360)
        # 断言使用 circmean 函数计算结果与 numpy 的平均值接近
        xp_assert_close(M2, M1, rtol=1e-5)

        # 使用 torch 的 var 和 std 函数进行测试，ddof=1，因此我们需要 array_api_compat 的 torch
        xp_test = array_namespace(x)
        V1 = xp_test.var(x*xp.pi/180, correction=0)
        # 对于小的变化，circvar 大约是线性方差的一半
        V1 = V1 / 2.
        V2 = stats.circvar(x, high=360)
        # 断言使用 circvar 函数计算结果与 numpy 的方差接近
        xp_assert_close(V2, V1, rtol=1e-4)

        S1 = xp_test.std(x, correction=0)
        S2 = stats.circstd(x, high=360)
        # 断言使用 circstd 函数计算结果与 numpy 的标准差接近
        xp_assert_close(S2, S1, rtol=1e-4)

    @pytest.mark.parametrize("test_func, numpy_func",
                             [(stats.circmean, np.mean),
                              (stats.circvar, np.var),
                              (stats.circstd, np.std)])
    def test_circfuncs_close(self, test_func, numpy_func, xp):
        # 参数化测试，使用 numpy 的函数作为参考
        # circfuncs 应该处理非常相似的输入 (gh-12740)
        x = np.asarray([0.12675364631578953] * 10 + [0.12675365920187928] * 100)
        circstat = test_func(xp.asarray(x))
        normal = xp.asarray(numpy_func(x))
        # 断言 circfuncs 的计算结果与 numpy 的结果接近
        xp_assert_close(circstat, normal, atol=2e-8)

    @pytest.mark.parametrize('circfunc', [stats.circmean,
                                          stats.circvar,
                                          stats.circstd])
    # 定义一个测试方法，用于测试循环均值函数在不同轴上的表现
    def test_circmean_axis(self, xp, circfunc):
        # 创建一个二维数组，表示角度数据，范围在 0 到 360 之间
        x = xp.asarray([[355, 5, 2, 359, 10, 350],
                        [351, 7, 4, 352, 9, 349],
                        [357, 9, 8, 358, 4, 356.]])
        # 计算整个数组的循环均值，不指定轴
        res = circfunc(x, high=360)
        # 将数组展开为一维，计算其循环均值作为参考值
        ref = circfunc(xp.reshape(x, (-1,)), high=360)
        # 断言计算结果与参考值接近
        xp_assert_close(res, xp.asarray(ref))

        # 指定轴为 1，计算每行的循环均值
        res = circfunc(x, high=360, axis=1)
        # 循环计算每行的循环均值作为参考值
        ref = [circfunc(x[i, :], high=360) for i in range(x.shape[0])]
        # 断言计算结果与参考值接近
        xp_assert_close(res, xp.asarray(ref))

        # 指定轴为 0，计算每列的循环均值
        res = circfunc(x, high=360, axis=0)
        # 循环计算每列的循环均值作为参考值
        ref = [circfunc(x[:, i], high=360) for i in range(x.shape[1])]
        # 断言计算结果与参考值接近
        xp_assert_close(res, xp.asarray(ref))

    # 使用参数化测试，测试各种循环统计函数在数组样式数据上的表现
    @pytest.mark.parametrize("test_func,expected",
                             [(stats.circmean, 0.167690146),
                              (stats.circvar, 0.006455174270186603),
                              (stats.circstd, 6.520702116)])
    def test_circfuncs_array_like(self, test_func, expected, xp):
        # 创建一个一维数组，表示角度数据
        x = xp.asarray([355, 5, 2, 359, 10, 350.])
        # 断言循环统计函数计算结果与期望值接近
        xp_assert_close(test_func(x, high=360), xp.asarray(expected))

    # 使用参数化测试，测试空数组情况下循环统计函数的行为
    @pytest.mark.parametrize("test_func", [stats.circmean, stats.circvar,
                                           stats.circstd])
    def test_empty(self, test_func, xp):
        # 指定数组元素类型为 float64
        dtype = xp.float64
        # 创建一个空的数组
        x = xp.asarray([], dtype=dtype)
        # 如果使用的是 NumPy，期望会有一个 SmallSampleWarning 警告
        if is_numpy(xp):
            with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
                # 执行循环统计函数，并捕获警告
                res = test_func(x)
        else:
            with np.testing.suppress_warnings() as sup:
                # for array_api_strict
                # 过滤掉 "Mean of empty slice" 和 "invalid value encountered" 的警告
                sup.filter(RuntimeWarning, "Mean of empty slice")
                sup.filter(RuntimeWarning, "invalid value encountered")
                # 执行循环统计函数
                res = test_func(x)
        # 断言计算结果为 NaN
        xp_assert_equal(res, xp.asarray(xp.nan, dtype=dtype))

    # 使用参数化测试，测试包含 NaN 值的数组在循环统计函数中的行为
    @pytest.mark.parametrize("test_func", [stats.circmean, stats.circvar,
                                           stats.circstd])
    def test_nan_propagate(self, test_func, xp):
        # 创建一个包含 NaN 值的数组
        x = xp.asarray([355, 5, 2, 359, 10, 350, np.nan])
        # 断言循环统计函数处理 NaN 值后的结果为 NaN
        xp_assert_equal(test_func(x, high=360), xp.asarray(xp.nan))

    # 使用参数化测试，测试循环统计函数在不同情况下的行为
    @pytest.mark.parametrize("test_func,expected",
                             [(stats.circmean,
                               {None: np.nan, 0: 355.66582264, 1: 0.28725053}),
                              (stats.circvar,
                               {None: np.nan,
                                0: 0.002570671054089924,
                                1: 0.005545914017677123}),
                              (stats.circstd,
                               {None: np.nan, 0: 4.11093193, 1: 6.04265394})])
    # 测试函数：验证在给定测试函数和期望结果的情况下，对输入数组中的 NaN 值进行处理
    def test_nan_propagate_array(self, test_func, expected, xp):
        # 创建输入数组 x，包含整数和 NaN 值
        x = xp.asarray([[355, 5, 2, 359, 10, 350, 1],
                        [351, 7, 4, 352, 9, 349, np.nan],
                        [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
        # 遍历期望结果中的轴
        for axis in expected.keys():
            # 调用测试函数，处理数组 x，指定高阈值为 360，轴为当前循环的 axis
            out = test_func(x, high=360, axis=axis)
            # 如果轴为 None，验证输出是否全为 NaN
            if axis is None:
                xp_assert_equal(out, xp.asarray(xp.nan))
            else:
                # 验证输出的第一个元素是否与期望一致
                xp_assert_close(out[0], xp.asarray(expected[axis]))
                # 验证除第一个元素外的其他元素是否全为 NaN
                xp_assert_equal(out[1:], xp.full_like(out[1:], xp.nan))

    # 测试函数：验证对标量进行 circmean 计算的准确性
    def test_circmean_scalar(self, xp):
        # 创建包含单个标量值的数组 x
        x = xp.asarray(1.)[()]
        # 计算标量值的平均值 M2
        M2 = stats.circmean(x)
        # 验证计算结果 M2 是否与输入值 M1 接近
        xp_assert_close(M2, x, rtol=1e-5)

    # 测试函数：验证对范围进行 circmean 计算的准确性
    def test_circmean_range(self, xp):
        # 执行回归测试以验证 circmean 在给定高低阈值时的结果是否在预期范围内
        m = stats.circmean(xp.arange(0, 2, 0.1), xp.pi, -xp.pi)
        # 验证计算结果 m 是否小于 pi
        xp_assert_less(m, xp.asarray(xp.pi))
        # 验证计算结果 -m 是否小于 pi
        xp_assert_less(-m, xp.asarray(xp.pi))

    # 测试函数：验证在处理 numpy uint8 数据类型时 circmean 的准确性
    def test_circfuncs_uint8(self, xp):
        # 执行回归测试以验证 circmean 在处理 numpy uint8 数据时的结果
        x = xp.asarray([150, 10], dtype=xp.uint8)
        # 验证 circmean 的计算结果是否与预期值接近
        xp_assert_close(stats.circmean(x, high=180), xp.asarray(170.0))
        xp_assert_close(stats.circvar(x, high=180), xp.asarray(0.2339555554617))
        xp_assert_close(stats.circstd(x, high=180), xp.asarray(20.91551378))

    # 测试函数：验证对单个数字进行 circstd 计算的准确性
    def test_circstd_zero(self, xp):
        # 对单个数字进行 circstd 计算，预期结果应返回正零
        y = stats.circstd(xp.asarray([0]))
        assert math.copysign(1.0, y) == 1.0

    # 测试函数：验证在处理极小输入时 circmean 的精确性
    def test_circmean_accuracy_tiny_input(self, xp):
        # 对于非常小的 x，使得 sin(x) == x 和 cos(x) == 1.0，在数值上 circmean(x) 应返回 x
        # 此测试验证这一点。
        #
        # 此测试的目的不是展示 circmean() 对某些输入在最后一位的精确性，因为这既不被保证也不特别有用。
        # 相反，这是一个"白盒"健全性检查，确保在 (high - low) 和 (2 * pi) 之间的转换不会引入不必要的精度损失。

        # 创建一个包含极小值的数组 x
        x = xp.linspace(1e-9, 1e-8, 100)
        # 验证在极小值情况下 sin(x) 是否等于 x，cos(x) 是否等于 1.0
        assert xp.all(xp.sin(x) == x) and xp.all(xp.cos(x) == 1.0)

        # 标记 x 中不等式 x * (2 * pi) / (2 * pi) != x 的位置
        m = (x * (2 * xp.pi) / (2 * xp.pi)) != x
        assert xp.any(m)
        # 选择 x 中满足条件的值
        x = x[m]

        # 对数组 x 的每一列计算 circmean，并验证结果是否与 x 相等
        y = stats.circmean(x[:, None], axis=1)
        assert xp.all(y == x)
    # 定义一个测试方法，用于验证 circmean() 函数在处理大量数据时不会由于过早旋转而引入数值精度损失。
    # 这是一个白盒测试，意在通过提供一个非常大的输入 x 来检测 (x - low) == x 的数值关系。
    # x 是一个由 xp.asarray() 创建的数组，数据类型为 xp.float64，值为 1e17。
    x = xp.asarray(1e17, dtype=xp.float64)

    # 计算 x 的正弦和余弦，然后使用 arctan2 函数计算角度，返回的值存储在 y 中。
    y = math.atan2(xp.sin(x), xp.cos(x))  # -2.6584887370946806

    # 将 y 转换为 xp.float64 类型后，存储在 expected 变量中。
    expected = xp.asarray(y, dtype=xp.float64)

    # 调用 stats 模块中的 circmean() 函数，对 x 进行处理，指定高度为 xp.pi，低度为 -xp.pi。
    actual = stats.circmean(x, high=xp.pi, low=-xp.pi)

    # 使用 xp_assert_close 函数断言 actual 和 expected 的接近程度。
    # 相对误差（rtol）为 1e-15，绝对误差（atol）为 0.0。
    xp_assert_close(actual, expected, rtol=1e-15, atol=0.0)
# 定义一个测试类 TestCircFuncsNanPolicy，用于测试圆形统计函数的 NaN 处理策略
class TestCircFuncsNanPolicy:

    # `nan_policy` 由 `_axis_nan_policy` 装饰器实现，该装饰器目前还不兼容数组 API。
    # 一旦兼容了数组 API，通用测试将会比这些更强大，因此这些参数化测试将不再必要。
    # 所以目前不需要使这些函数兼容数组 API；等到时机成熟时，它们可以被移除。
    @pytest.mark.parametrize("test_func,expected",
                             [(stats.circmean,
                               {None: 359.4178026893944,
                                0: np.array([353.0, 6.0, 3.0, 355.5, 9.5,
                                             349.5]),
                                1: np.array([0.16769015, 358.66510252])}),
                              (stats.circvar,
                               {None: 0.008396678483192477,
                                0: np.array([1.9997969, 0.4999873, 0.4999873,
                                             6.1230956, 0.1249992, 0.1249992]
                                            )*(np.pi/180)**2,
                                1: np.array([0.006455174270186603,
                                             0.01016767581393285])}),
                              (stats.circstd,
                               {None: 7.440570778057074,
                                0: np.array([2.00020313, 1.00002539, 1.00002539,
                                             3.50108929, 0.50000317,
                                             0.50000317]),
                                1: np.array([6.52070212, 8.19138093])})])
    # 定义测试方法 test_nan_omit_array，参数化测试不同的圆形统计函数和期望值
    def test_nan_omit_array(self, test_func, expected):
        # 定义输入数组 x，包含 NaN 值
        x = np.array([[355, 5, 2, 359, 10, 350, np.nan],
                      [351, 7, 4, 352, 9, 349, np.nan],
                      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
        # 对于每个轴，执行测试
        for axis in expected.keys():
            if axis is None:
                # 对于 axis 为 None 的情况，使用指定的圆形统计函数计算，忽略 NaN 值
                out = test_func(x, high=360, nan_policy='omit', axis=axis)
                # 断言计算结果与期望值的接近程度
                assert_allclose(out, expected[axis], rtol=1e-7)
            else:
                # 对于指定的 axis，使用指定的圆形统计函数计算，忽略 NaN 值
                # 使用 pytest.warns 检查警告信息，匹配特定的警告内容
                with pytest.warns(SmallSampleWarning, match=too_small_nd_omit):
                    out = test_func(x, high=360, nan_policy='omit', axis=axis)
                    # 断言前面部分结果与期望值的接近程度
                    assert_allclose(out[:-1], expected[axis], rtol=1e-7)
                    # 断言最后一个值为 NaN
                    assert_(np.isnan(out[-1]))

    # 参数化测试单个数列的圆形统计函数及其期望值
    @pytest.mark.parametrize("test_func,expected",
                             [(stats.circmean, 0.167690146),
                              (stats.circvar, 0.006455174270186603),
                              (stats.circstd, 6.520702116)])
    # 定义测试方法 test_nan_omit
    def test_nan_omit(self, test_func, expected):
        # 定义输入列表 x，包含 NaN 值
        x = [355, 5, 2, 359, 10, 350, np.nan]
        # 断言单个数列的圆形统计函数计算结果与期望值的接近程度
        assert_allclose(test_func(x, high=360, nan_policy='omit'),
                        expected, rtol=1e-7)

    # 参数化测试不同的圆形统计函数
    @pytest.mark.parametrize("test_func", [stats.circmean, stats.circvar,
                                           stats.circstd])
    # 定义一个测试函数，测试在所有值为 NaN 的情况下，函数 test_func 的行为
    def test_nan_omit_all(self, test_func):
        # 创建一个包含多个 NaN 值的列表
        x = [np.nan, np.nan, np.nan, np.nan, np.nan]
        # 使用 pytest 来检测是否会出现 SmallSampleWarning 警告，且警告信息匹配 too_small_1d_omit
        with pytest.warns(SmallSampleWarning, match=too_small_1d_omit):
            # 断言调用 test_func 对于 x 的处理结果中是否包含 NaN
            assert_(np.isnan(test_func(x, nan_policy='omit')))

    # 使用 parametrize 标记，为以下测试函数（circmean、circvar、circstd）参数化
    @pytest.mark.parametrize("test_func", [stats.circmean, stats.circvar,
                                           stats.circstd])
    # 测试在指定轴上所有值为 NaN 的情况下，函数 test_func 的行为
    def test_nan_omit_all_axis(self, test_func):
        # 使用 pytest 来检测是否会出现 SmallSampleWarning 警告，且警告信息匹配 too_small_nd_omit
        with pytest.warns(SmallSampleWarning, match=too_small_nd_omit):
            # 创建一个包含多个 NaN 值的二维数组 x
            x = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan],
                          [np.nan, np.nan, np.nan, np.nan, np.nan]])
            # 调用 test_func 对 x 进行处理，指定 nan_policy='omit'，沿 axis=1（行）进行操作
            out = test_func(x, nan_policy='omit', axis=1)
            # 断言处理结果中是否全部为 NaN
            assert_(np.isnan(out).all())
            # 断言处理后的结果数组长度是否为 2
            assert_(len(out) == 2)

    # 使用 parametrize 标记，为以下测试函数（circmean、circvar、circstd）参数化
    @pytest.mark.parametrize("x",
                             [[355, 5, 2, 359, 10, 350, np.nan],
                              np.array([[355, 5, 2, 359, 10, 350, np.nan],
                                        [351, 7, 4, 352, np.nan, 9, 349]])])
    @pytest.mark.parametrize("test_func", [stats.circmean, stats.circvar,
                                           stats.circstd])
    # 测试当遇到无效的 nan_policy='raise' 时，是否会引发 ValueError 异常
    def test_nan_raise(self, test_func, x):
        # 断言调用 test_func 对于 x，设置 high=360 和 nan_policy='raise' 时，是否会引发 ValueError 异常
        assert_raises(ValueError, test_func, x, high=360, nan_policy='raise')

    # 使用 parametrize 标记，为以下测试函数（circmean、circvar、circstd）参数化
    @pytest.mark.parametrize("x",
                             [[355, 5, 2, 359, 10, 350, np.nan],
                              np.array([[355, 5, 2, 359, 10, 350, np.nan],
                                        [351, 7, 4, 352, np.nan, 9, 349]])])
    @pytest.mark.parametrize("test_func", [stats.circmean, stats.circvar,
                                           stats.circstd])
    # 测试当遇到无效的 nan_policy='foobar' 时，是否会引发 ValueError 异常
    def test_bad_nan_policy(self, test_func, x):
        # 断言调用 test_func 对于 x，设置 high=360 和 nan_policy='foobar' 时，是否会引发 ValueError 异常
        assert_raises(ValueError, test_func, x, high=360, nan_policy='foobar')
# 定义一个测试类 TestMedianTest
class TestMedianTest:

    # 定义一个测试方法，测试当样本数量不足时是否会引发 ValueError
    def test_bad_n_samples(self):
        # median_test 需要至少两个样本。
        assert_raises(ValueError, stats.median_test, [1, 2, 3])

    # 定义一个测试方法，测试空样本是否会引发 ValueError
    def test_empty_sample(self):
        # 每个样本必须至少包含一个值。
        assert_raises(ValueError, stats.median_test, [], [1, 2, 3])

    # 定义一个测试方法，测试当设置 ties="ignore" 时空样本是否会引发 ValueError
    def test_empty_when_ties_ignored(self):
        # 总中位数为 1，并且第一个参数中的所有值都等于总中位数。
        # 在 ties="ignore" 的情况下，这些值会被忽略，导致第一个样本在效果上为空。
        # 这应该会引发 ValueError。
        assert_raises(ValueError, stats.median_test,
                      [1, 1, 1, 1], [2, 0, 1], [2, 0], ties="ignore")

    # 定义一个测试方法，测试当默认设置 ties="below" 时是否会引发 ValueError
    def test_empty_contingency_row(self):
        # 总中位数为 1，且默认 ties="below"，所有样本中的值都被计算为低于总中位数。
        # 这会导致列联表中出现一行全为零的情况，这是一个错误。
        assert_raises(ValueError, stats.median_test, [1, 1, 1], [1, 1, 1])

        # 当 ties="above" 时，所有的值被计算为高于总中位数。
        assert_raises(ValueError, stats.median_test, [1, 1, 1], [1, 1, 1],
                      ties="above")

    # 定义一个测试方法，测试当 ties 参数设置错误时是否会引发 ValueError
    def test_bad_ties(self):
        assert_raises(ValueError, stats.median_test, [1, 2, 3], [4, 5],
                      ties="foo")

    # 定义一个测试方法，测试当 nan_policy 参数设置错误时是否会引发 ValueError
    def test_bad_nan_policy(self):
        assert_raises(ValueError, stats.median_test, [1, 2, 3], [4, 5],
                      nan_policy='foobar')

    # 定义一个测试方法，测试当传入未知关键字参数时是否会引发 TypeError
    def test_bad_keyword(self):
        assert_raises(TypeError, stats.median_test, [1, 2, 3], [4, 5],
                      foo="foo")

    # 定义一个简单的测试方法，测试基本的 median_test 功能
    def test_simple(self):
        x = [1, 2, 3]
        y = [1, 2, 3]
        stat, p, med, tbl = stats.median_test(x, y)

        # 中位数是浮点数，但这个相等测试应该是安全的。
        assert_equal(med, 2.0)

        # 断言数组 tbl 的值与预期的行列联表相等
        assert_array_equal(tbl, [[1, 1], [2, 2]])

        # 预期值与行列联表相等，因此统计量应该为 0，p 值应该为 1。
        assert_equal(stat, 0)
        assert_equal(p, 1)

    # 定义一个测试方法，测试不同 ties 选项下的 median_test 计算
    def test_ties_options(self):
        x = [1, 2, 3, 4]
        y = [5, 6]
        z = [7, 8, 9]
        # 总中位数为 5。

        # 默认的 'ties' 选项是 "below"。
        stat, p, m, tbl = stats.median_test(x, y, z)
        assert_equal(m, 5)
        assert_equal(tbl, [[0, 1, 3], [4, 1, 0]])

        # 当 ties="ignore" 时的测试。
        stat, p, m, tbl = stats.median_test(x, y, z, ties="ignore")
        assert_equal(m, 5)
        assert_equal(tbl, [[0, 1, 3], [4, 0, 0]])

        # 当 ties="above" 时的测试。
        stat, p, m, tbl = stats.median_test(x, y, z, ties="above")
        assert_equal(m, 5)
        assert_equal(tbl, [[0, 2, 3], [4, 0, 0]])
    def test_nan_policy_options(self):
        # 创建包含 NaN 值的两个列表作为输入数据
        x = [1, 2, np.nan]
        y = [4, 5, 6]
        
        # 使用 'propagate' 策略计算中位数检验的统计量和 p 值
        mt1 = stats.median_test(x, y, nan_policy='propagate')
        
        # 使用 'omit' 策略计算中位数检验的统计量、p 值、中位数和频数表
        s, p, m, t = stats.median_test(x, y, nan_policy='omit')

        # 断言第一次中位数检验的结果应为 (NaN, NaN, NaN, None)
        assert_equal(mt1, (np.nan, np.nan, np.nan, None))
        
        # 断言第二次中位数检验的统计量接近 0.31250000000000006
        assert_allclose(s, 0.31250000000000006)
        
        # 断言第二次中位数检验的 p 值接近 0.57615012203057869
        assert_allclose(p, 0.57615012203057869)
        
        # 断言第二次中位数检验的中位数为 4.0
        assert_equal(m, 4.0)
        
        # 断言第二次中位数检验的频数表
        assert_equal(t, np.array([[0, 2], [2, 1]]))
        
        # 断言使用 'raise' 策略时应引发 ValueError
        assert_raises(ValueError, stats.median_test, x, y, nan_policy='raise')

    def test_basic(self):
        # 测试基本的中位数检验功能

        # 创建两个输入列表
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8]

        # 计算中位数检验的统计量、p 值、中位数和频数表
        stat, p, m, tbl = stats.median_test(x, y)
        
        # 断言中位数为 4
        assert_equal(m, 4)
        
        # 断言频数表
        assert_equal(tbl, [[1, 2], [4, 2]])

        # 使用 chi2_contingency 函数计算期望的统计量和 p 值
        exp_stat, exp_p, dof, e = stats.chi2_contingency(tbl)
        
        # 断言计算得到的统计量和期望的统计量接近
        assert_allclose(stat, exp_stat)
        
        # 断言计算得到的 p 值和期望的 p 值接近
        assert_allclose(p, exp_p)

        # 使用 lambda_=0 参数调用中位数检验
        stat, p, m, tbl = stats.median_test(x, y, lambda_=0)
        
        # 断言中位数为 4
        assert_equal(m, 4)
        
        # 断言频数表
        assert_equal(tbl, [[1, 2], [4, 2]])

        # 使用 chi2_contingency 函数计算期望的统计量和 p 值，同时传递 lambda_=0 参数
        exp_stat, exp_p, dof, e = stats.chi2_contingency(tbl, lambda_=0)
        
        # 断言计算得到的统计量和期望的统计量接近
        assert_allclose(stat, exp_stat)
        
        # 断言计算得到的 p 值和期望的 p 值接近
        assert_allclose(p, exp_p)

        # 禁用校正参数进行中位数检验
        stat, p, m, tbl = stats.median_test(x, y, correction=False)
        
        # 断言中位数为 4
        assert_equal(m, 4)
        
        # 断言频数表
        assert_equal(tbl, [[1, 2], [4, 2]])

        # 使用 chi2_contingency 函数计算期望的统计量和 p 值，同时禁用校正参数
        exp_stat, exp_p, dof, e = stats.chi2_contingency(tbl, correction=False)
        
        # 断言计算得到的统计量和期望的统计量接近
        assert_allclose(stat, exp_stat)
        
        # 断言计算得到的 p 值和期望的 p 值接近
        assert_allclose(p, exp_p)

    @pytest.mark.parametrize("correction", [False, True])
    def test_result(self, correction):
        # 测试中位数检验的结果

        # 创建两个输入列表
        x = [1, 2, 3]
        y = [1, 2, 3]

        # 执行中位数检验，根据 correction 参数进行校正
        res = stats.median_test(x, y, correction=correction)
        
        # 断言结果的统计量、p 值、中位数和频数表
        assert_equal((res.statistic, res.pvalue, res.median, res.table), res)
class TestDirectionalStats:
    # Reference implementations are not available
    # 测试方向统计正确性的类

    def test_directional_stats_correctness(self):
        # Data from Fisher: Dispersion on a sphere, 1953 and
        # Mardia and Jupp, Directional Statistics.
        # 使用 Fisher 和 Mardia 提供的数据进行方向统计的正确性测试

        # Declination and inclination angles converted to radians
        decl = -np.deg2rad(np.array([343.2, 62., 36.9, 27., 359.,
                                     5.7, 50.4, 357.6, 44.]))
        incl = -np.deg2rad(np.array([66.1, 68.7, 70.1, 82.1, 79.5,
                                     73., 69.3, 58.8, 51.4]))

        # Convert angles to Cartesian coordinates
        data = np.stack((np.cos(incl) * np.cos(decl),
                         np.cos(incl) * np.sin(decl),
                         np.sin(incl)),
                        axis=1)
        
        # Calculate directional statistics for the data
        dirstats = stats.directional_stats(data)
        directional_mean = dirstats.mean_direction
        mean_rounded = np.round(directional_mean, 4)

        # Reference mean direction provided for assertion
        reference_mean = np.array([0.2984, -0.1346, -0.9449])
        assert_allclose(mean_rounded, reference_mean)

    @pytest.mark.parametrize('angles, ref', [
        ([-np.pi/2, np.pi/2], 1.),
        ([0, 2*np.pi], 0.)
    ])
    def test_directional_stats_2d_special_cases(self, angles, ref):
        if callable(ref):
            ref = ref(angles)
        
        # Convert angles to 2D Cartesian coordinates
        data = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        
        # Calculate directional statistics and compare with reference
        res = 1 - stats.directional_stats(data).mean_resultant_length
        assert_allclose(res, ref)

    def test_directional_stats_2d(self):
        # Test that for circular data directional_stats
        # yields the same result as circmean/circvar
        # 测试对于圆形数据，directional_stats 和 circmean/circvar 的结果一致性
        
        # Generate random circular data in radians
        rng = np.random.default_rng(0xec9a6899d5a2830e0d1af479dbe1fd0c)
        testdata = 2 * np.pi * rng.random((1000, ))
        
        # Convert random data to 2D Cartesian coordinates
        testdata_vector = np.stack((np.cos(testdata),
                                    np.sin(testdata)),
                                   axis=1)
        
        # Calculate directional statistics
        dirstats = stats.directional_stats(testdata_vector)
        directional_mean = dirstats.mean_direction
        
        # Convert mean direction angle to match circmean's output format
        directional_mean_angle = np.arctan2(directional_mean[1],
                                            directional_mean[0])
        directional_mean_angle = directional_mean_angle % (2*np.pi)
        
        # Compare directional mean with circmean of the test data
        circmean = stats.circmean(testdata)
        assert_allclose(circmean, directional_mean_angle)

        # Compare directional variance with circvar of the test data
        directional_var = 1 - dirstats.mean_resultant_length
        circular_var = stats.circvar(testdata)
        assert_allclose(directional_var, circular_var)

    def test_directional_mean_higher_dim(self):
        # test that directional_stats works for higher dimensions
        # here a 4D array is reduced over axis = 2
        # 测试 directional_stats 在高维度数据上的工作，这里对一个4维数组沿 axis = 2 进行统计
        
        # Example 2D data array
        data = np.array([[0.8660254, 0.5, 0.],
                         [0.8660254, -0.5, 0.]])
        
        # Tile the 2D data to create a 4D array
        full_array = np.tile(data, (2, 2, 2, 1))
        
        # Expected mean direction for the 4D array
        expected = np.array([[[1., 0., 0.],
                              [1., 0., 0.]],
                             [[1., 0., 0.],
                              [1., 0., 0.]]])
        
        # Calculate directional statistics over axis 2 of the 4D array
        dirstats = stats.directional_stats(full_array, axis=2)
        
        # Assert that the calculated mean direction matches the expected result
        assert_allclose(expected, dirstats.mean_direction)
    # 测试函数，验证列表和 NumPy 数组作为输入时是否产生相同的结果
    def test_directional_stats_list_ndarray_input(self):
        # 定义输入数据：包括两个三维向量的列表
        data = [[0.8660254, 0.5, 0.], [0.8660254, -0.5, 0]]
        # 将列表转换为 NumPy 数组
        data_array = np.asarray(data)
        # 对输入数据调用 directional_stats 函数，返回结果
        res = stats.directional_stats(data)
        # 对 NumPy 数组调用 directional_stats 函数，返回参考结果
        ref = stats.directional_stats(data_array)
        # 断言两次调用的平均方向是否接近
        assert_allclose(res.mean_direction, ref.mean_direction)
        # 断言两次调用的平均结果长度是否接近
        assert_allclose(res.mean_resultant_length,
                        res.mean_resultant_length)

    # 测试函数，验证一维数据是否会引发 ValueError
    def test_directional_stats_1d_error(self):
        # 定义输入数据：一维全为 1 的 NumPy 数组
        data = np.ones((5, ))
        # 期望引发的错误消息
        message = (r"samples must at least be two-dimensional. "
                   r"Instead samples has shape: (5,)")
        # 使用 pytest 的断言来验证是否会引发 ValueError，并检查错误消息是否匹配
        with pytest.raises(ValueError, match=re.escape(message)):
            stats.directional_stats(data)

    # 测试函数，验证不同输入下 directional stats 计算是否产生相同结果
    def test_directional_stats_normalize(self):
        # 定义未归一化输入数据：包括两个三维向量的 NumPy 数组
        data = np.array([[0.8660254, 0.5, 0.],
                         [1.7320508, -1., 0.]])
        # 对未归一化数据调用 directional_stats 函数，返回结果
        res = stats.directional_stats(data, normalize=True)
        # 对数据进行归一化处理
        normalized_data = data / np.linalg.norm(data, axis=-1,
                                                keepdims=True)
        # 对归一化后的数据调用 directional_stats 函数，返回参考结果
        ref = stats.directional_stats(normalized_data,
                                      normalize=False)
        # 断言两次调用的平均方向是否接近
        assert_allclose(res.mean_direction, ref.mean_direction)
        # 断言两次调用的平均结果长度是否接近
        assert_allclose(res.mean_resultant_length,
                        ref.mean_resultant_length)
class TestFDRControl:
    # 定义测试类 TestFDRControl，用于测试 false_discovery_control 函数的行为

    def test_input_validation(self):
        # 测试输入验证函数
        message = "`ps` must include only numbers between 0 and 1"
        # 设置错误信息，用于验证是否引发 ValueError 异常
        with pytest.raises(ValueError, match=message):
            # 使用 pytest 检查是否引发 ValueError，并匹配指定的错误信息
            stats.false_discovery_control([-1, 0.5, 0.7])
        with pytest.raises(ValueError, match=message):
            stats.false_discovery_control([0.5, 0.7, 2])
        with pytest.raises(ValueError, match=message):
            stats.false_discovery_control([0.5, 0.7, np.nan])

        message = "Unrecognized `method` 'YAK'"
        # 设置错误信息，用于验证是否引发 ValueError 异常
        with pytest.raises(ValueError, match=message):
            stats.false_discovery_control([0.5, 0.7, 0.9], method='YAK')

        message = "`axis` must be an integer or `None`"
        # 设置错误信息，用于验证是否引发 ValueError 异常
        with pytest.raises(ValueError, match=message):
            stats.false_discovery_control([0.5, 0.7, 0.9], axis=1.5)
        with pytest.raises(ValueError, match=message):
            stats.false_discovery_control([0.5, 0.7, 0.9], axis=(1, 2))

    def test_against_TileStats(self):
        # 与 TileStats 的比较测试
        # 参考 false_discovery_control 的参考 [3]
        ps = [0.005, 0.009, 0.019, 0.022, 0.051, 0.101, 0.361, 0.387]
        # 设置测试用的概率值列表
        res = stats.false_discovery_control(ps)
        # 调用 false_discovery_control 函数计算结果
        ref = [0.036, 0.036, 0.044, 0.044, 0.082, 0.135, 0.387, 0.387]
        # 设置预期结果列表
        assert_allclose(res, ref, atol=1e-3)
        # 使用 assert_allclose 检查计算结果是否与预期结果在指定的误差范围内相等

    @pytest.mark.parametrize("case",
                             [([0.24617028, 0.01140030, 0.05652047, 0.06841983,
                                0.07989886, 0.01841490, 0.17540784, 0.06841983,
                                0.06841983, 0.25464082], 'bh'),
                              ([0.72102493, 0.03339112, 0.16554665, 0.20039952,
                                0.23402122, 0.05393666, 0.51376399, 0.20039952,
                                0.20039952, 0.74583488], 'by')])
    # 参数化测试，对不同的 case 执行相同的测试函数
    def test_against_R(self, case):
        # 与 R 的比较测试
        # 测试与 p.adjust 函数的对比，例如 p = c(0.22155325, 0.00114003,..., 0.0364813 , 0.25464082)
        # p.adjust(p, "BY")
        ref, method = case
        # 设置参考值和方法
        rng = np.random.default_rng(6134137338861652935)
        # 创建随机数生成器
        ps = stats.loguniform.rvs(1e-3, 0.5, size=10, random_state=rng)
        ps[3] = ps[7]  # force a tie
        # 修改第三个和第七个元素，强制它们相等
        res = stats.false_discovery_control(ps, method=method)
        # 调用 false_discovery_control 函数计算结果
        assert_allclose(res, ref, atol=1e-6)
        # 使用 assert_allclose 检查计算结果是否与预期结果在指定的误差范围内相等

    def test_axis_None(self):
        # 测试 axis 参数为 None 的情况
        rng = np.random.default_rng(6134137338861652935)
        # 创建随机数生成器
        ps = stats.loguniform.rvs(1e-3, 0.5, size=(3, 4, 5), random_state=rng)
        # 生成符合对数均匀分布的随机数数组
        res = stats.false_discovery_control(ps, axis=None)
        # 调用 false_discovery_control 函数计算结果，axis 设置为 None
        ref = stats.false_discovery_control(ps.ravel())
        # 计算展开 ps 后的参考结果
        assert_equal(res, ref)
        # 使用 assert_equal 检查计算结果是否与预期结果完全相等

    @pytest.mark.parametrize("axis", [0, 1, -1])
    # 参数化测试，对不同的 axis 值执行相同的测试函数
    def test_axis(self, axis):
        # 测试指定 axis 值的情况
        rng = np.random.default_rng(6134137338861652935)
        # 创建随机数生成器
        ps = stats.loguniform.rvs(1e-3, 0.5, size=(3, 4, 5), random_state=rng)
        # 生成符合对数均匀分布的随机数数组
        res = stats.false_discovery_control(ps, axis=axis)
        # 调用 false_discovery_control 函数计算结果，指定 axis
        ref = np.apply_along_axis(stats.false_discovery_control, axis, ps)
        # 使用 np.apply_along_axis 计算沿指定轴的参考结果
        assert_equal(res, ref)
        # 使用 assert_equal 检查计算结果是否与预期结果完全相等
    # 定义测试函数 test_edge_cases，用于测试极端情况
    def test_edge_cases(self):
        # 断言调用 stats.false_discovery_control 函数，传入参数 [0.25]，预期返回结果与 [0.25] 相等
        assert_array_equal(stats.false_discovery_control([0.25]), [0.25])
        # 断言调用 stats.false_discovery_control 函数，传入参数 0.25，预期返回结果与 0.25 相等
        assert_array_equal(stats.false_discovery_control(0.25), 0.25)
        # 断言调用 stats.false_discovery_control 函数，传入空列表 []，预期返回结果为空列表 []
        assert_array_equal(stats.false_discovery_control([]), [])
# 使用装饰器确保此类兼容数组 API
@array_api_compatible
# 定义测试类 TestCommonAxis
class TestCommonAxis:
    # 这里的注释指出了 `test_axis_nan_policy` 中 `axis` 测试更加详尽，
    # 但这些测试在数组 API 下还没有运行。此类放在 `test_morestats` 中，
    # 而不是 `test_axis_nan_policy` 中，因为当前数组 API CI 作业中没有运行 `test_axis_nan_policy` 的原因。

    # 使用 pytest 的参数化装饰器，传入不同的测试用例给 `test_axis` 方法
    @pytest.mark.parametrize('case', [(stats.sem, {}),
                                      (stats.kstat, {'n': 4}),
                                      (stats.kstat, {'n': 2}),
                                      (stats.variation, {})])
    # 定义测试方法 `test_axis`，接受参数 `case` 和 `xp`
    def test_axis(self, case, xp):
        # 从参数 `case` 中解包出函数 `fun` 和关键字参数 `kwargs`
        fun, kwargs = case
        # 使用默认的随机数生成器创建 `rng`
        rng = np.random.default_rng(24598245982345)
        # 生成一个 `6x7` 的随机数组，并将其转换为 `xp` 对象
        x = xp.asarray(rng.random((6, 7)))

        # 在 `axis=0` 上调用函数 `fun`，计算结果 `res`
        res = fun(x, **kwargs, axis=0)
        # 生成参考结果 `ref`，使用列表推导式逐列计算 `fun(x[:, i], **kwargs)`，并转换为 `xp` 对象
        ref = xp.asarray([fun(x[:, i], **kwargs) for i in range(x.shape[1])])
        # 使用 `xp_assert_close` 检验 `res` 和 `ref` 的近似程度

        # 在 `axis=1` 上调用函数 `fun`，计算结果 `res`
        res = fun(x, **kwargs, axis=1)
        # 生成参考结果 `ref`，使用列表推导式逐行计算 `fun(x[i, :], **kwargs)`，并转换为 `xp` 对象
        ref = xp.asarray([fun(x[i, :], **kwargs) for i in range(x.shape[0])])
        # 使用 `xp_assert_close` 检验 `res` 和 `ref` 的近似程度

        # 在 `axis=None` 上调用函数 `fun`，计算结果 `res`
        res = fun(x, **kwargs, axis=None)
        # 生成参考结果 `ref`，将 `x` 重塑为一维数组，并应用 `fun` 函数，然后转换为 `xp` 对象
        ref = fun(xp.reshape(x, (-1,)), **kwargs)
        # 使用 `xp_assert_close` 检验 `res` 和 `ref` 的近似程度
```