# `D:\src\scipysrc\scipy\scipy\stats\tests\test_continuous_basic.py`

```
import sys  # 导入sys模块，用于系统相关的参数和功能
import numpy as np  # 导入NumPy库，并用np作为别名
import numpy.testing as npt  # 导入NumPy的测试模块，并用npt作为别名
import pytest  # 导入pytest测试框架
from pytest import raises as assert_raises  # 从pytest中导入raises函数，并用assert_raises作为别名
from scipy.integrate import IntegrationWarning  # 从SciPy库中导入IntegrationWarning
import itertools  # 导入itertools模块，用于创建迭代器的函数

from scipy import stats  # 从SciPy库中导入stats模块
from .common_tests import (check_normalization, check_moment,  # 导入自定义模块common_tests中的函数
                           check_mean_expect,
                           check_var_expect, check_skew_expect,
                           check_kurt_expect, check_entropy,
                           check_private_entropy, check_entropy_vect_scale,
                           check_edge_support, check_named_args,
                           check_random_state_property,
                           check_meth_dtype, check_ppf_dtype,
                           check_cmplx_deriv,
                           check_pickling, check_rvs_broadcast,
                           check_freezing, check_munp_expect,)
from scipy.stats._distr_params import distcont  # 导入SciPy的_distr_params模块中的distcont
from scipy.stats._distn_infrastructure import rv_continuous_frozen  # 导入SciPy的_distn_infrastructure模块中的rv_continuous_frozen

"""
Test all continuous distributions.

Parameters were chosen for those distributions that pass the
Kolmogorov-Smirnov test.  This provides safe parameters for each
distributions so that we can perform further testing of class methods.

These tests currently check only/mostly for serious errors and exceptions,
not for numerically exact results.
"""

# Note that you need to add new distributions you want tested
# to _distr_params

DECIMAL = 5  # 指定测试精度为小数点后5位  # 从0增加到5的精度

_IS_32BIT = (sys.maxsize < 2**32)  # 检查当前系统是否为32位系统，返回布尔值

# Sets of tests to skip.
# Entries sorted by speed (very slow to slow).
# xslow took > 1s; slow took > 0.5s

xslow_test_cont_basic = {'studentized_range', 'kstwo', 'ksone', 'vonmises', 'kappa4',  # 定义需要跳过的测试集合，这些测试较慢
                         'recipinvgauss', 'vonmises_line', 'gausshyper',
                         'rel_breitwigner', 'norminvgauss'}
slow_test_cont_basic = {'crystalball', 'powerlognorm', 'pearson3'}

# test_moments is already marked slow
xslow_test_moments = {'studentized_range', 'ksone', 'vonmises', 'vonmises_line',  # 需要跳过的矩测试集合
                      'recipinvgauss', 'kstwo', 'kappa4'}

slow_fit_mle = {'exponweib', 'genexpon', 'genhyperbolic', 'johnsonsb',  # 慢速的最大似然估计集合
                'kappa4', 'powerlognorm', 'tukeylambda'}
xslow_fit_mle = {'gausshyper', 'ncf', 'ncx2', 'recipinvgauss', 'vonmises_line'}
xfail_fit_mle = {'ksone', 'kstwo', 'trapezoid', 'truncpareto', 'irwinhall'}  # 失败的最大似然估计集合
skip_fit_mle = {'levy_stable', 'studentized_range'}  # 需要完全跳过的最大似然估计集合（运行时间超过10分钟）
slow_fit_mm = {'chi2', 'expon', 'lognorm', 'loguniform', 'powerlaw', 'reciprocal'}  # 慢速的最大似然估计集合
xslow_fit_mm = {'argus', 'beta', 'exponpow', 'gausshyper', 'gengamma',  # 极慢速的最大似然估计集合
                'genhalflogistic', 'geninvgauss', 'gompertz', 'halfgennorm',
                'johnsonsb', 'kstwobign', 'ncx2', 'norminvgauss', 'truncnorm',
                'truncweibull_min', 'wrapcauchy'}
# 用于忽略在拟合过程中不适合的概率分布名称集合，这些分布较慢或者在特定测试中表现不佳
xfail_fit_mm = {'alpha', 'betaprime', 'bradford', 'burr', 'burr12', 'cauchy',
                'crystalball', 'exponweib', 'f', 'fisk', 'foldcauchy', 'genextreme',
                'genpareto', 'halfcauchy', 'invgamma', 'irwinhall', 'jf_skew_t',
                'johnsonsu', 'kappa3', 'kappa4', 'levy', 'levy_l', 'loglaplace',
                'lomax', 'mielke', 'ncf', 'nct', 'pareto', 'powerlognorm', 'powernorm',
                'rel_breitwigner',  'skewcauchy', 't', 'trapezoid', 'truncexpon',
                'truncpareto', 'tukeylambda', 'vonmises', 'vonmises_line'}

# 用于忽略在拟合过程中速度较慢的概率分布名称集合
skip_fit_mm = {'genexpon', 'genhyperbolic', 'ksone', 'kstwo', 'levy_stable',
               'recipinvgauss', 'studentized_range'}  # far too slow (>10min)

# 这些概率分布在复杂导数测试中失败
# 这里的“失败”意味着产生错误的结果和/或引发异常，具体取决于对应特殊函数实现的细节
# 参考 https://github.com/scipy/scipy/pull/4979 进行讨论
fails_cmplx = {'argus', 'beta', 'betaprime', 'chi', 'chi2', 'cosine',
               'dgamma', 'dweibull', 'erlang', 'f', 'foldcauchy', 'gamma',
               'gausshyper', 'gengamma', 'genhyperbolic',
               'geninvgauss', 'gennorm', 'genpareto',
               'halfcauchy', 'halfgennorm', 'invgamma', 'irwinhall', 'jf_skew_t',
               'ksone', 'kstwo', 'kstwobign', 'levy_l', 'loggamma',
               'logistic', 'loguniform', 'maxwell', 'nakagami',
               'ncf', 'nct', 'ncx2', 'norminvgauss', 'pearson3',
               'powerlaw', 'rdist', 'reciprocal', 'rice',
               'skewnorm', 't', 'truncweibull_min',
               'tukeylambda', 'vonmises', 'vonmises_line',
               'rv_histogram_instance', 'truncnorm', 'studentized_range',
               'johnsonsb', 'halflogistic', 'rel_breitwigner'}

# 速度较慢的测试方法集合，用于使用列表时
slow_with_lists = {'studentized_range'}

# 包含直方图测试实例的列表，分为等宽和非等宽箱子
histogram_test_instances = []
case1 = {'a': [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6,
               6, 6, 6, 7, 7, 7, 8, 8, 9], 'bins': 8}  # equal width bins
case2 = {'a': [1, 1], 'bins': [0, 1, 10]}  # unequal width bins
# 对每个案例和密度进行排列组合，创建直方图并转换为分布对象后添加到测试实例列表中
for case, density in itertools.product([case1, case2], [True, False]):
    _hist = np.histogram(**case, density=density)
    _rv_hist = stats.rv_histogram(_hist, density=density)
    histogram_test_instances.append((_rv_hist, tuple()))

# 用于测试连续分布基本特性的测试用例生成器函数
def cases_test_cont_basic():
    for distname, arg in distcont[:] + histogram_test_instances:
        if distname == 'levy_stable':  # 在单独的测试中失败，因此跳过
            continue
        if distname in slow_test_cont_basic:
            yield pytest.param(distname, arg, marks=pytest.mark.slow)  # 标记为慢速测试
        elif distname in xslow_test_cont_basic:
            yield pytest.param(distname, arg, marks=pytest.mark.xslow)  # 标记为非常慢测试
        else:
            yield distname, arg
# 使用参数化测试，循环执行每一对 (distname, arg) 组合作为测试用例
@pytest.mark.parametrize('distname,arg', cases_test_cont_basic())
# 对 sn 参数进行参数化，固定为 [500]
@pytest.mark.parametrize('sn', [500])
def test_cont_basic(distname, arg, sn):
    try:
        # 尝试获取名为 distname 的概率分布函数对象
        distfn = getattr(stats, distname)
    except TypeError:
        # 如果出现 TypeError，将 distname 视为已存在的分布实例，命名为 'rv_histogram_instance'
        distfn = distname
        distname = 'rv_histogram_instance'

    # 创建随机数生成器对象，固定种子 765456
    rng = np.random.RandomState(765456)
    # 生成 size 为 sn 的随机样本数据
    rvs = distfn.rvs(size=sn, *arg, random_state=rng)
    # 获取分布的均值 m 和方差 v
    m, v = distfn.stats(*arg)

    # 如果 distname 不在 {'laplace_asymmetric'} 中，则检查样本均值和方差
    if distname not in {'laplace_asymmetric'}:
        check_sample_meanvar_(m, v, rvs)

    # 检查累积分布函数和反函数的正确性
    check_cdf_ppf(distfn, arg, distname)
    # 检查生存函数和反函数的正确性
    check_sf_isf(distfn, arg, distname)
    # 检查累积分布函数和生存函数的互补性
    check_cdf_sf(distfn, arg, distname)
    # 检查反函数和反函数的互补性
    check_ppf_isf(distfn, arg, distname)
    # 检查概率密度函数
    check_pdf(distfn, arg, distname)
    # 检查概率密度函数和对数概率密度函数的一致性
    check_pdf_logpdf(distfn, arg, distname)
    # 检查概率密度函数和对数概率密度函数在端点处的值
    check_pdf_logpdf_at_endpoints(distfn, arg, distname)
    # 检查累积分布函数和对数累积分布函数的一致性
    check_cdf_logcdf(distfn, arg, distname)
    # 检查生存函数和对数生存函数的一致性
    check_sf_logsf(distfn, arg, distname)
    # 检查反函数的广播功能
    check_ppf_broadcast(distfn, arg, distname)

    # 设置显著性水平 alpha 为 0.01
    alpha = 0.01
    # 如果 distname 是 'rv_histogram_instance'，则针对其累积分布函数进行分布检查
    if distname == 'rv_histogram_instance':
        check_distribution_rvs(distfn.cdf, arg, alpha, rvs)
    # 否则，对 distname 不是 'geninvgauss' 的分布进行分布检查
    elif distname != 'geninvgauss':
        check_distribution_rvs(distname, arg, alpha, rvs)

    # 设置默认的 loc 和 scale 参数
    locscale_defaults = (0, 1)
    # 定义需要检查的方法列表
    meths = [distfn.pdf, distfn.logpdf, distfn.cdf, distfn.logcdf,
             distfn.logsf]
    # 确保 x 参数在支持范围内
    spec_x = {'weibull_max': -0.5, 'levy_l': -0.5,
              'pareto': 1.5, 'truncpareto': 3.2, 'tukeylambda': 0.3,
              'rv_histogram_instance': 5.0}
    # 根据 distname 获取特定的 x 值
    x = spec_x.get(distname, 0.5)
    # 对特定分布进行参数修正
    if distname == 'invweibull':
        arg = (1,)
    elif distname == 'ksone':
        arg = (3,)

    # 检查命名参数的正确性
    check_named_args(distfn, x, arg, locscale_defaults, meths)
    # 检查随机数生成器的属性
    check_random_state_property(distfn, arg)

    # 如果 distname 在 ['rel_breitwigner'] 中并且当前环境是 32 位系统，跳过测试
    if distname in ['rel_breitwigner'] and _IS_32BIT:
        pytest.skip("fails on Linux 32-bit")
    else:
        # 否则，进行对象的序列化测试
        check_pickling(distfn, arg)
    # 检查冻结分布的正确性
    check_freezing(distfn, arg)

    # 计算分布的熵
    if distname not in ['kstwobign', 'kstwo', 'ncf']:
        check_entropy(distfn, arg, distname)

    # 如果分布没有参数，检查分布的向量熵
    if distfn.numargs == 0:
        check_vecentropy(distfn, arg)

    # 如果分布的熵不等于通用连续随机变量的熵且 distname 不是 'vonmises'，检查私有熵
    if (distfn.__class__._entropy != stats.rv_continuous._entropy
            and distname != 'vonmises'):
        check_private_entropy(distfn, arg, stats.rv_continuous)

    # 使用上下文管理器，忽略特定警告类型，并进行熵向量缩放测试
    with npt.suppress_warnings() as sup:
        sup.filter(IntegrationWarning, "The occurrence of roundoff error")
        sup.filter(IntegrationWarning, "Extremely bad integrand")
        sup.filter(RuntimeWarning, "invalid value")
        check_entropy_vect_scale(distfn, arg)

    # 检查支持范围的正确性
    check_retrieving_support(distfn, arg)
    # 检查边界支持的正确性
    check_edge_support(distfn, arg)

    # 检查方法的返回数据类型
    check_meth_dtype(distfn, arg, meths)
    # 检查反函数的返回数据类型
    check_ppf_dtype(distfn, arg)

    # 如果 distname 不在 fails_cmplx 列表中，检查复杂导数
    if distname not in fails_cmplx:
        check_cmplx_deriv(distfn, arg)
    # 如果分布名称不是 'truncnorm'，则调用 check_ppf_private 函数来检查和执行相关操作
    if distname != 'truncnorm':
        check_ppf_private(distfn, arg, distname)
def cases_test_cont_basic_fit():
    # 定义一些用于标记测试的 pytest 标记
    slow = pytest.mark.slow
    xslow = pytest.mark.xslow
    fail = pytest.mark.skip(reason="Test fails and may be slow.")
    skip = pytest.mark.skip(reason="Test too slow to run to completion (>10m).")

    # 对于每个分布和参数的组合，以及直方图测试实例，生成测试用例
    for distname, arg in distcont[:] + histogram_test_instances:
        # 遍历两种拟合方法：MLE（极大似然估计）和MM（矩法）
        for method in ["MLE", "MM"]:
            # 对于是否修正参数的两种情况，生成测试参数
            for fix_args in [True, False]:
                # 根据不同的情况应用不同的 pytest 标记
                if method == 'MLE' and distname in slow_fit_mle:
                    yield pytest.param(distname, arg, method, fix_args, marks=slow)
                    continue
                if method == 'MLE' and distname in xslow_fit_mle:
                    yield pytest.param(distname, arg, method, fix_args, marks=xslow)
                    continue
                if method == 'MLE' and distname in xfail_fit_mle:
                    yield pytest.param(distname, arg, method, fix_args, marks=fail)
                    continue
                if method == 'MLE' and distname in skip_fit_mle:
                    yield pytest.param(distname, arg, method, fix_args, marks=skip)
                    continue
                if method == 'MM' and distname in slow_fit_mm:
                    yield pytest.param(distname, arg, method, fix_args, marks=slow)
                    continue
                if method == 'MM' and distname in xslow_fit_mm:
                    yield pytest.param(distname, arg, method, fix_args, marks=xslow)
                    continue
                if method == 'MM' and distname in xfail_fit_mm:
                    yield pytest.param(distname, arg, method, fix_args, marks=fail)
                    continue
                if method == 'MM' and distname in skip_fit_mm:
                    yield pytest.param(distname, arg, method, fix_args, marks=skip)
                    continue

                # 默认情况下生成测试参数
                yield distname, arg, method, fix_args


def test_cont_basic_fit_cases():
    # 检查分布名称不应同时出现在多个MLE或MM集合中
    assert (len(xslow_fit_mle.union(xfail_fit_mle).union(skip_fit_mle)) ==
            len(xslow_fit_mle) + len(xfail_fit_mle) + len(skip_fit_mle))
    assert (len(xslow_fit_mm.union(xfail_fit_mm).union(skip_fit_mm)) ==
            len(xslow_fit_mm) + len(xfail_fit_mm) + len(skip_fit_mm))


@pytest.mark.parametrize('distname, arg, method, fix_args',
                         cases_test_cont_basic_fit())
@pytest.mark.parametrize('n_fit_samples', [200])
def test_cont_basic_fit(distname, arg, n_fit_samples, method, fix_args):
    try:
        # 尝试获取给定名称的分布函数
        distfn = getattr(stats, distname)
    except TypeError:
        # 如果失败，则使用名称本身作为函数
        distfn = distname

    # 创建一个指定种子的随机数生成器
    rng = np.random.RandomState(765456)
    # 生成随机变量样本
    rvs = distfn.rvs(size=n_fit_samples, *arg, random_state=rng)
    # 根据 fix_args 的值检查拟合参数
    if fix_args:
        check_fit_args_fix(distfn, arg, rvs, method)
    else:
        check_fit_args(distfn, arg, rvs, method)

@pytest.mark.parametrize('distname,arg', cases_test_cont_basic_fit())
def test_rvs_scalar(distname, arg):
    # 当参数为标量时，rvs 应返回一个标量 (gh-12428)
    # 尝试从 `stats` 模块中获取名为 `distname` 的分布函数对象
    try:
        distfn = getattr(stats, distname)
    # 如果 `distname` 不是一个类型错误（TypeError）引发的异常，将 `distname` 直接赋值给 `distfn`
    except TypeError:
        distfn = distname
        # 将 `distname` 设置为字符串 'rv_histogram_instance'
        distname = 'rv_histogram_instance'

    # 断言 `distfn.rvs(*arg)` 返回的值是标量（scalar）
    assert np.isscalar(distfn.rvs(*arg))
    # 断言 `distfn.rvs(*arg, size=())` 返回的值是标量（scalar）
    assert np.isscalar(distfn.rvs(*arg, size=()))
    # 断言 `distfn.rvs(*arg, size=None)` 返回的值是标量（scalar）
    assert np.isscalar(distfn.rvs(*arg, size=None))
# 测试 levy_stable 随机数生成器的状态属性
def test_levy_stable_random_state_property():
    # levy_stable 只实现了 rvs() 方法，在 test_cont_basic() 的主循环中被跳过。
    # 在这里我们只对 levy_stable 应用 check_random_state_property 进行测试。
    check_random_state_property(stats.levy_stable, (0.5, 0.1))


# 函数用例 cases_test_moments()
def cases_test_moments():
    # 失败的正则化测试集合
    fail_normalization = set()
    # 失败的高阶矩测试集合，包括 'ncf'
    fail_higher = {'ncf'}
    # 失败的矩测试集合，例如 'johnsonsu'，generic `munp` 在 johnsonsu 分布上不精确
    fail_moment = {'johnsonsu'}

    # 遍历 distcont 列表和 histogram_test_instances
    for distname, arg in distcont[:] + histogram_test_instances:
        # 如果分布名称为 'levy_stable'，则跳过此次循环
        if distname == 'levy_stable':
            continue

        # 如果 distname 在 xslow_test_moments 中，则使用 pytest 的 xslow 标记
        if distname in xslow_test_moments:
            yield pytest.param(distname, arg, True, True, True, True,
                               marks=pytest.mark.xslow(reason="too slow"))
            continue

        # 判断 distname 是否不在失败的三个测试集合中
        cond1 = distname not in fail_normalization
        cond2 = distname not in fail_higher
        cond3 = distname not in fail_moment

        # 创建 marks 列表，目前未使用，可以用来为特定分布的测试添加超时限制
        marks = list()
        # 示例中展示了如何为 'skewnorm' 分布添加超时限制
        # marks.append(pytest.mark.timeout(300)) if distname == 'skewnorm' else None

        # 生成测试参数，包括分布名称、参数、三个条件是否满足、是否预期测试失败的标志
        yield pytest.param(distname, arg, cond1, cond2, cond3,
                           False, marks=marks)

        # 如果有任何一个条件不满足，则再次运行测试
        if not cond1 or not cond2 or not cond3:
            # 两次运行分布，一次跳过 not_ok 部分，一次包括 not_ok 部分但标记为 knownfail
            yield pytest.param(distname, arg, True, True, True, True,
                               marks=[pytest.mark.xfail] + marks)


# 使用 pytest 的 slow 标记，参数化执行 cases_test_moments() 生成的测试用例
@pytest.mark.slow
@pytest.mark.parametrize('distname,arg,normalization_ok,higher_ok,moment_ok,'
                         'is_xfailing',
                         cases_test_moments())
def test_moments(distname, arg, normalization_ok, higher_ok, moment_ok,
                 is_xfailing):
    try:
        # 尝试获取名为 distname 的分布函数
        distfn = getattr(stats, distname)
    except TypeError:
        # 如果出现 TypeError，将 distname 赋值给 distfn，并将 distname 设为 'rv_histogram_instance'
        distfn = distname
        distname = 'rv_histogram_instance'
    # 使用 npt.suppress_warnings() 上下文管理器来忽略特定类型的警告
    with npt.suppress_warnings() as sup:
        # 过滤特定的积分警告，告知积分可能发散或收敛缓慢
        sup.filter(IntegrationWarning,
                   "The integral is probably divergent, or slowly convergent.")
        # 过滤最大子分割数警告
        sup.filter(IntegrationWarning,
                   "The maximum number of subdivisions.")
        # 过滤算法不收敛警告
        sup.filter(IntegrationWarning,
                   "The algorithm does not converge.")

        # 如果测试预期失败，则过滤积分警告
        if is_xfailing:
            sup.filter(IntegrationWarning)

        # 计算分布的统计量：均值（m）、方差（v）、偏度（s）、峰度（k）
        m, v, s, k = distfn.stats(*arg, moments='mvsk')

        # 忽略所有 numpy 的错误（如溢出或除零），在此上下文中设置
        with np.errstate(all="ignore"):
            # 如果归一化正常，则检查分布的归一化
            if normalization_ok:
                check_normalization(distfn, arg, distname)

            # 如果高级检查正常，则依次检查均值、偏度、方差、峰度的期望
            if higher_ok:
                check_mean_expect(distfn, arg, m, distname)
                check_skew_expect(distfn, arg, m, v, s, distname)
                check_var_expect(distfn, arg, m, v, distname)
                check_kurt_expect(distfn, arg, m, v, k, distname)
                check_munp_expect(distfn, arg, distname)

        # 检查分布的位置和尺度参数
        check_loc_scale(distfn, arg, m, v, distname)

        # 如果矩检查正常，则检查分布的矩
        if moment_ok:
            check_moment(distfn, arg, m, v, distname)
# 使用 pytest.mark.parametrize 装饰器，为 test_rvs_broadcast 函数提供多个参数化的测试用例
@pytest.mark.parametrize('dist,shape_args', distcont)
def test_rvs_broadcast(dist, shape_args):
    # 如果分布是 'gausshyper' 或 'studentized_range'，跳过测试并显示消息
    if dist in ['gausshyper', 'studentized_range']:
        pytest.skip("too slow")

    # 如果分布是 'rel_breitwigner' 并且系统是 32 位，则跳过测试并显示消息
    if dist in ['rel_breitwigner'] and _IS_32BIT:
        # gh18414
        pytest.skip("fails on Linux 32-bit")

    # 如果 shape_only 是 True，表示分布的 _rvs 方法使用多个随机数生成随机变量
    # 这意味着使用广播或非平凡大小的情况下，结果可能不同于使用 numpy.vectorize 版本的 rvs()
    # 因此只能比较结果的形状而不是值
    # 分布是否在以下列表中是分布的实现细节，不是要求。如果分布的 rvs() 方法实现变化，这个测试可能也需要改变。
    shape_only = dist in ['argus', 'betaprime', 'dgamma', 'dweibull',
                          'exponnorm', 'genhyperbolic', 'geninvgauss',
                          'levy_stable', 'nct', 'norminvgauss', 'rice',
                          'skewnorm', 'semicircular', 'gennorm', 'loggamma']

    # 获取分布函数对象
    distfunc = getattr(stats, dist)
    # 设置 loc 参数为长度为 2 的零向量
    loc = np.zeros(2)
    # 设置 scale 参数为形状为 (3, 1) 的全一数组
    scale = np.ones((3, 1))
    # 获取分布函数的参数个数
    nargs = distfunc.numargs
    # 初始化参数列表
    allargs = []
    # 初始化形状列表为 [3, 2]
    bshape = [3, 2]

    # 生成形状参数的参数...
    for k in range(nargs):
        # 创建形状为 (k+4,) + (1,)*(k+2) 的 shp
        shp = (k + 4,) + (1,)*(k + 2)
        # 将 shape_args[k] 的值扩展为 shp 的形状，加入到参数列表中
        allargs.append(shape_args[k]*np.ones(shp))
        # 将 k+4 插入到 bshape 列表的开头
        bshape.insert(0, k + 4)
    # 添加 loc 和 scale 到参数列表中
    allargs.extend([loc, scale])
    # bshape 包含了当 loc、scale 和形状参数都进行广播时的预期形状。

    # 调用 check_rvs_broadcast 函数，对分布函数进行广播随机变量测试
    check_rvs_broadcast(distfunc, dist, allargs, bshape, shape_only, 'd')


# 以下是预期的 SF、CDF、PDF 值，使用 mpmath 和设定 mpmath.mp.dps = 50 计算，并在 20 位输出精度下显示：
#
# def ks(x, n):
#     x = mpmath.mpf(x)
#     logp = -mpmath.power(6.0*n*x+1.0, 2)/18.0/n
#     sf, cdf = mpmath.exp(logp), -mpmath.expm1(logp)
#     pdf = (6.0*n*x+1.0) * 2 * sf/3
#     print(mpmath.nstr(sf, 20), mpmath.nstr(cdf, 20), mpmath.nstr(pdf, 20))
#
# 测试使用 1/n < x < 1-1/n 和 n > 1e6 条件来进行渐近计算。
# 较大的 x 会有较小的 sf。
@pytest.mark.parametrize('x,n,sf,cdf,pdf,rtol',
                         [(2.0e-5, 1000000000,
                           0.44932297307934442379, 0.55067702692065557621,
                           35946.137394996276407, 5e-15),
                          (2.0e-9, 1000000000,
                           0.99999999061111115519, 9.3888888448132728224e-9,
                           8.6666665852962971765, 5e-14),
                          (5.0e-4, 1000000000,
                           7.1222019433090374624e-218, 1.0,
                           1.4244408634752704094e-211, 5e-14)])
def test_gh17775_regression(x, n, sf, cdf, pdf, rtol):
    # Regression test for gh-17775. In scipy 1.9.3 and earlier,
    # 回归测试，针对 gh-17775。在 Scipy 1.9.3 及更早版本中，
    pass  # 空语句，暂未实现具体的测试代码
    # 这些测试将失败。
    #
    # KS 检验中一个渐进的尾部概率密度函数约为 e^(-(6nx+1)^2 / 18n)
    # 给定一个大的 32 位整数 n，在 C 语言实现中，6n 可能会溢出。
    # 破损行为的示例：
    # ksone.sf(2.0e-5, 1000000000) == 0.9374359693473666
    # ks 是 stats 中的 ksone 模块
    ks = stats.ksone
    # 创建包含 ks.sf(x, n), ks.cdf(x, n), ks.pdf(x, n) 的 NumPy 数组
    vals = np.array([ks.sf(x, n), ks.cdf(x, n), ks.pdf(x, n)])
    # 期望的值数组
    expected = np.array([sf, cdf, pdf])
    # 使用相对容差检查 vals 和 expected 是否全部接近
    npt.assert_allclose(vals, expected, rtol=rtol)
    # sf 和 cdf 的和必须为 1.0
    npt.assert_equal(vals[0] + vals[1], 1.0)
    # 检查对 sf 的反函数是否接近 x（使用更低的容差）
    npt.assert_allclose([ks.isf(sf, n)], [x], rtol=1e-8)
def test_rvs_gh2069_regression():
    # Regression tests for gh-2069.  In scipy 0.17 and earlier,
    # these tests would fail.
    #
    # A typical example of the broken behavior:
    # >>> norm.rvs(loc=np.zeros(5), scale=np.ones(5))
    # array([-2.49613705, -2.49613705, -2.49613705, -2.49613705, -2.49613705])
    
    # 使用特定的随机种子创建一个随机数生成器实例
    rng = np.random.RandomState(123)
    
    # 生成服从正态分布的随机数，均值为0，标准差为1
    vals = stats.norm.rvs(loc=np.zeros(5), scale=1, random_state=rng)
    
    # 计算生成的随机数之间的差值
    d = np.diff(vals)
    
    # 断言所有的差值不应该为零，如果有任何一个为零则抛出错误信息
    npt.assert_(np.all(d != 0), "All the values are equal, but they shouldn't be!")
    
    # 使用不同的参数再次生成随机数，并进行相同的差值检查
    vals = stats.norm.rvs(loc=0, scale=np.ones(5), random_state=rng)
    d = np.diff(vals)
    npt.assert_(np.all(d != 0), "All the values are equal, but they shouldn't be!")
    
    # 再次使用不同的参数生成随机数，并进行差值检查
    vals = stats.norm.rvs(loc=np.zeros(5), scale=np.ones(5), random_state=rng)
    d = np.diff(vals)
    npt.assert_(np.all(d != 0), "All the values are equal, but they shouldn't be!")
    
    # 使用多维均值数组和标准差为1生成随机数，并进行差值检查
    vals = stats.norm.rvs(loc=np.array([[0], [0]]), scale=np.ones(5),
                          random_state=rng)
    d = np.diff(vals.ravel())
    npt.assert_(np.all(d != 0), "All the values are equal, but they shouldn't be!")

    # 断言使用不合法参数调用正态分布生成函数会引发值错误异常
    assert_raises(ValueError, stats.norm.rvs, [[0, 0], [0, 0]],
                  [[1, 1], [1, 1]], 1)
    
    # 断言使用不合法参数调用 gamma 分布生成函数会引发值错误异常
    assert_raises(ValueError, stats.gamma.rvs, [2, 3, 4, 5], 0, 1, (2, 2))
    
    # 断言使用不合法参数调用 gamma 分布生成函数会引发值错误异常
    assert_raises(ValueError, stats.gamma.rvs, [1, 1, 1, 1], [0, 0, 0, 0],
                  [[1], [2]], (4,))


def test_nomodify_gh9900_regression():
    # Regression test for gh-9990
    # Prior to gh-9990, calls to stats.truncnorm._cdf() use what ever was
    # set inside the stats.truncnorm instance during stats.truncnorm.cdf().
    # This could cause issues with multi-threaded code.
    # Since then, the calls to cdf() are not permitted to modify the global
    # stats.truncnorm instance.
    
    # 将 stats.truncnorm 模块赋值给变量 tn
    tn = stats.truncnorm
    
    # 使用右半截尾正态分布，检查 cdf 和 _cdf 返回相同的结果
    npt.assert_almost_equal(tn.cdf(1, 0, np.inf),
                            0.6826894921370859)
    
    # 使用右半截尾正态分布，检查 cdf 和 _cdf 返回相同的结果
    npt.assert_almost_equal(tn._cdf([1], [0], [np.inf]),
                            0.6826894921370859)
    
    # 使用左半截尾正态分布，检查 cdf 和 _cdf 返回相同的结果
    npt.assert_almost_equal(tn.cdf(-1, -np.inf, 0),
                            0.31731050786291415)
    
    # 使用左半截尾正态分布，检查 cdf 和 _cdf 返回相同的结果
    npt.assert_almost_equal(tn._cdf([-1], [-np.inf], [0]),
                            0.31731050786291415)
    
    # 检查右半截尾正态分布的 _cdf 是否未发生变化
    npt.assert_almost_equal(tn._cdf([1], [0], [np.inf]),
                            0.6826894921370859)  # Not 1.6826894921370859
    
    # 检查右半截尾正态分布的 cdf 是否未发生变化
    npt.assert_almost_equal(tn.cdf(1, 0, np.inf),
                            0.6826894921370859)
    
    # 检查左半截尾正态分布的 _cdf 是否未发生变化
    npt.assert_almost_equal(tn._cdf([-1], [-np.inf], [0]),
                            0.31731050786291415)  # Not -0.6826894921370859
    
    # 检查左半截尾正态分布的 cdf 是否未发生变化
    npt.assert_almost_equal(tn.cdf(1, -np.inf, 0),
                            1)  # Not 1.6826894921370859
    # 使用 numpy.testing.assert_almost_equal 函数来验证累积分布函数的计算结果是否接近期望值
    npt.assert_almost_equal(tn.cdf(-1, -np.inf, 0),
                            0.31731050786291415)  # 预期结果是 0.31731050786291415，而非 -0.6826894921370859
def test_broadcast_gh9990_regression():
    # 回归测试，针对 GitHub issue-9990
    # x 值为 7 仅在所提供的 4 个分布的支持范围内。在 9990 之前，传递给 stats.reciprocal._cdf 的一个数组将有 4 个元素，
    # 但之前由 stats.reciprocal_argcheck() 存储的数组将有 6 个元素，导致广播错误。
    a = np.array([1, 2, 3, 4, 5, 6])
    b = np.array([8, 16, 1, 32, 1, 48])
    ans = [stats.reciprocal.cdf(7, _a, _b) for _a, _b in zip(a,b)]
    npt.assert_array_almost_equal(stats.reciprocal.cdf(7, a, b), ans)

    ans = [stats.reciprocal.cdf(1, _a, _b) for _a, _b in zip(a,b)]
    npt.assert_array_almost_equal(stats.reciprocal.cdf(1, a, b), ans)

    ans = [stats.reciprocal.cdf(_a, _a, _b) for _a, _b in zip(a,b)]
    npt.assert_array_almost_equal(stats.reciprocal.cdf(a, a, b), ans)

    ans = [stats.reciprocal.cdf(_b, _a, _b) for _a, _b in zip(a,b)]
    npt.assert_array_almost_equal(stats.reciprocal.cdf(b, a, b), ans)


def test_broadcast_gh7933_regression():
    # 检查广播是否正常工作
    stats.truncnorm.logpdf(
        np.array([3.0, 2.0, 1.0]),
        a=(1.5 - np.array([6.0, 5.0, 4.0])) / 3.0,
        b=np.inf,
        loc=np.array([6.0, 5.0, 4.0]),
        scale=3.0
    )


def test_gh2002_regression():
    # 添加一个检查，确保在只有部分 x 值与某些形状参数兼容的情况下广播正常工作。
    x = np.r_[-2:2:101j]
    a = np.r_[-np.ones(50), np.ones(51)]
    expected = [stats.truncnorm.pdf(_x, _a, np.inf) for _x, _a in zip(x, a)]
    ans = stats.truncnorm.pdf(x, a, np.inf)
    npt.assert_array_almost_equal(ans, expected)


def test_gh1320_regression():
    # 检查 gh-1320 中的第一个示例现在是否正常工作。
    c = 2.62
    stats.genextreme.ppf(0.5, np.array([[c], [c + 0.5]]))
    # gh-1320 中的其他示例似乎在一段时间前就停止工作了。
    # ans = stats.genextreme.moment(2, np.array([c, c + 0.5]))
    # expected = np.array([25.50105963, 115.11191437])
    # stats.genextreme.moment(5, np.array([[c], [c + 0.5]]))
    # stats.genextreme.moment(5, np.array([c, c + 0.5]))


def test_method_of_moments():
    # 来自 https://en.wikipedia.org/wiki/Method_of_moments_(statistics) 的示例
    np.random.seed(1234)
    x = [0, 0, 0, 0, 1]
    a = 1/5 - 2*np.sqrt(3)/5
    b = 1/5 + 2*np.sqrt(3)/5
    # 强制使用矩方法（uniform.fit 被覆盖）
    loc, scale = super(type(stats.uniform), stats.uniform).fit(x, method="MM")
    npt.assert_almost_equal(loc, a, decimal=4)
    npt.assert_almost_equal(loc+scale, b, decimal=4)


def check_sample_meanvar_(popmean, popvar, sample):
    if np.isfinite(popmean):
        check_sample_mean(sample, popmean)
    if np.isfinite(popvar):
        check_sample_var(sample, popvar)


def check_sample_mean(sample, popmean):
    # 检查样本均值与总体均值之间的不太可能的差异
    prob = stats.ttest_1samp(sample, popmean).pvalue
    # 确保 prob 的值大于 0.01，如果不是，则会引发 AssertionError 异常。
    assert prob > 0.01
# 检查样本与总体方差的置信区间是否重叠，用于验证总体均值是否在由样本引导的置信区间内。
def check_sample_var(sample, popvar):
    # 使用bootstrap方法计算样本方差的置信区间
    res = stats.bootstrap(
        (sample,),
        lambda x, axis: x.var(ddof=1, axis=axis),
        confidence_level=0.995,
    )
    conf = res.confidence_interval
    low, high = conf.low, conf.high
    # 断言总体方差是否在计算出的置信区间内
    assert low <= popvar <= high


# 检查累积分布函数（CDF）与分位函数（PPF）之间的一致性
def check_cdf_ppf(distfn, arg, msg):
    values = [0.001, 0.5, 0.999]
    # 检验CDF与PPF之间的往返关系是否近似相等
    npt.assert_almost_equal(distfn.cdf(distfn.ppf(values, *arg), *arg),
                            values, decimal=DECIMAL, err_msg=msg +
                            ' - cdf-ppf roundtrip')


# 检查生存函数（SF）与逆生存函数（ISF）之间的一致性
def check_sf_isf(distfn, arg, msg):
    # 检验SF与ISF之间的往返关系是否近似相等
    npt.assert_almost_equal(distfn.sf(distfn.isf([0.1, 0.5, 0.9], *arg), *arg),
                            [0.1, 0.5, 0.9], decimal=DECIMAL, err_msg=msg +
                            ' - sf-isf roundtrip')


# 检查CDF与SF之间的关系
def check_cdf_sf(distfn, arg, msg):
    # 检验CDF与1-SF之间的关系是否近似相等
    npt.assert_almost_equal(distfn.cdf([0.1, 0.9], *arg),
                            1.0 - distfn.sf([0.1, 0.9], *arg),
                            decimal=DECIMAL, err_msg=msg +
                            ' - cdf-sf relationship')


# 检查PPF与ISF之间的关系
def check_ppf_isf(distfn, arg, msg):
    p = np.array([0.1, 0.9])
    # 检验PPF与ISF之间的关系是否近似相等
    npt.assert_almost_equal(distfn.isf(p, *arg), distfn.ppf(1-p, *arg),
                            decimal=DECIMAL, err_msg=msg +
                            ' - ppf-isf relationship')


# 检查概率密度函数（PDF）与累积分布函数（CDF）的导数之间的关系
def check_pdf(distfn, arg, msg):
    # 比较PDF在中位数处的值与CDF的数值导数的数值近似
    median = distfn.ppf(0.5, *arg)
    eps = 1e-6
    pdfv = distfn.pdf(median, *arg)
    if (pdfv < 1e-4) or (pdfv > 1e4):
        # 避免检查PDF接近零或非常大的情况（奇点）
        median = median + 0.1
        pdfv = distfn.pdf(median, *arg)
    cdfdiff = (distfn.cdf(median + eps, *arg) -
               distfn.cdf(median - eps, *arg))/eps/2.0
    # 断言PDF值与CDF的数值导数之间的数值近似
    msg += ' - cdf-pdf relationship'
    npt.assert_almost_equal(pdfv, cdfdiff, decimal=DECIMAL, err_msg=msg)


# 检查概率密度函数（PDF）与其对数的关系
def check_pdf_logpdf(distfn, args, msg):
    # 比较几个点上的PDF与对数PDF的关系
    points = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    vals = distfn.ppf(points, *args)
    vals = vals[np.isfinite(vals)]
    pdf = distfn.pdf(vals, *args)
    logpdf = distfn.logpdf(vals, *args)
    pdf = pdf[(pdf != 0) & np.isfinite(pdf)]
    logpdf = logpdf[np.isfinite(logpdf)]
    # 断言PDF与对数PDF之间的数值近似
    msg += " - logpdf-log(pdf) relationship"
    npt.assert_almost_equal(np.log(pdf), logpdf, decimal=7, err_msg=msg)


# 检查概率密度函数（PDF）与其对数的关系，特别是在端点处
def check_pdf_logpdf_at_endpoints(distfn, args, msg):
    # 比较端点处PDF与对数PDF的关系
    points = np.array([0, 1])
    vals = distfn.ppf(points, *args)
    vals = vals[np.isfinite(vals)]
    pdf = distfn.pdf(vals, *args)
    # 计算概率分布函数在给定值上的对数概率密度值
    logpdf = distfn.logpdf(vals, *args)
    # 过滤掉 PDF 值为零或非有限的部分
    pdf = pdf[(pdf != 0) & np.isfinite(pdf)]
    # 过滤掉对数概率密度值中非有限的部分
    logpdf = logpdf[np.isfinite(logpdf)]
    # 将一条信息添加到消息中，描述对数概率密度与概率密度之间的关系
    msg += " - logpdf-log(pdf) relationship"
    # 使用断言检查对数概率密度与概率密度的对应关系是否近似相等
    npt.assert_almost_equal(np.log(pdf), logpdf, decimal=7, err_msg=msg)
# 检查生存函数与生存函数的对数在几个点上的比较
def check_sf_logsf(distfn, args, msg):
    # 定义比较的点
    points = np.array([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
    # 使用分位数函数计算给定点的值
    vals = distfn.ppf(points, *args)
    # 过滤掉非有限值
    vals = vals[np.isfinite(vals)]
    # 计算生存函数（Survival Function）
    sf = distfn.sf(vals, *args)
    # 计算生存函数的对数（Log Survival Function）
    logsf = distfn.logsf(vals, *args)
    # 过滤掉生存函数值为零的点
    sf = sf[sf != 0]
    # 过滤掉对数生存函数中的非有限值
    logsf = logsf[np.isfinite(logsf)]
    # 更新消息以包含对数生存函数与生存函数的对数之间的关系
    msg += " - logsf-log(sf) relationship"
    # 断言对数生存函数与生存函数的对数在小数点后7位上几乎相等
    npt.assert_almost_equal(np.log(sf), logsf, decimal=7, err_msg=msg)


# 检查累积分布函数与累积分布函数的对数在几个点上的比较
def check_cdf_logcdf(distfn, args, msg):
    # 定义比较的点
    points = np.array([0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
    # 使用分位数函数计算给定点的值
    vals = distfn.ppf(points, *args)
    # 过滤掉非有限值
    vals = vals[np.isfinite(vals)]
    # 计算累积分布函数（Cumulative Distribution Function）
    cdf = distfn.cdf(vals, *args)
    # 计算累积分布函数的对数（Log Cumulative Distribution Function）
    logcdf = distfn.logcdf(vals, *args)
    # 过滤掉累积分布函数值为零的点
    cdf = cdf[cdf != 0]
    # 过滤掉对数累积分布函数中的非有限值
    logcdf = logcdf[np.isfinite(logcdf)]
    # 更新消息以包含对数累积分布函数与累积分布函数的对数之间的关系
    msg += " - logcdf-log(cdf) relationship"
    # 断言对数累积分布函数与累积分布函数的对数在小数点后7位上几乎相等
    npt.assert_almost_equal(np.log(cdf), logcdf, decimal=7, err_msg=msg)


# 检查多个参数集合下的百分位点函数
def check_ppf_broadcast(distfn, arg, msg):
    # 设定重复次数
    num_repeats = 5
    # 如果参数集合非空，则生成对应的参数列表
    args = [] * num_repeats
    if arg:
        args = [np.array([_] * num_repeats) for _ in arg]

    # 计算中位数
    median = distfn.ppf(0.5, *arg)
    # 计算多个参数集合下的中位数
    medians = distfn.ppf(0.5, *args)
    # 更新消息以表明进行了多个参数集合下的百分位点计算
    msg += " - ppf multiple"
    # 断言多个参数集合下的中位数应该几乎相等
    npt.assert_almost_equal(medians, [median] * num_repeats, decimal=7, err_msg=msg)


# 检查分布的随机变量样本的K-S检验
def check_distribution_rvs(dist, args, alpha, rvs):
    # dist 可以是一个累积分布函数，或者是 scipy.stats 中分布的名称
    # args 是 scipy.stats.dist(*args) 的参数
    # alpha 是显著性水平，通常约为 0.01
    # rvs 是随机变量的数组
    # 从 scipy.stats.tests 中测试
    # 这个版本重复使用现有的随机变量
    # 执行 Kolmogorov-Smirnov 检验，检验 rvs 是否符合分布
    D, pval = stats.kstest(rvs, dist, args=args, N=1000)
    if (pval < alpha):
        # 如果 K-S 检验不通过，可能是样本不符合分布
        # 生成新的样本，再次进行 K-S 检验
        D, pval = stats.kstest(dist, dist, args=args, N=1000)
        # 断言新样本通过 K-S 检验
        npt.assert_(pval > alpha, "D = " + str(D) + "; pval = " + str(pval) +
                    "; alpha = " + str(alpha) + "\nargs = " + str(args))


# 检查 ve-entropy 方法是否与 _entropy 方法一致
def check_vecentropy(distfn, args):
    # 断言 ve-entropy 方法与 _entropy 方法返回的结果应该相等
    npt.assert_equal(distfn.vecentropy(*args), distfn._entropy(*args))


# 检查 loc 和 scale 数组，捕捉 gh-13580 类似的 bug，其中 loc 和 scale 数组不正确广播形状的情况
def check_loc_scale(distfn, arg, m, v, msg):
    # 定义 loc 和 scale 数组
    loc, scale = np.array([10.0, 20.0]), np.array([10.0, 20.0])
    # 计算分布的期望值和方差
    mt, vt = distfn.stats(*arg, loc=loc, scale=scale)
    # 断言分布的期望值满足 loc 和 scale 的线性变换
    npt.assert_allclose(m*scale + loc, mt)
    # 断言分布的方差满足 scale 的平方倍
    npt.assert_allclose(v*scale*scale, vt)


# 检查 _ppf 方法是否失败，因为没有定义 self.nb
def check_ppf_private(distfn, arg, msg):
    # 调用 _ppf 方法，验证其是否因为缺少 self.nb 定义而失败
    ppfs = distfn._ppf(np.array([0.1, 0.5, 0.9]), *arg)
    # 使用 NumPy 函数 np.isnan 检查数组 ppfs 中是否存在 NaN（Not a Number）值
    npt.assert_(not np.any(np.isnan(ppfs)), msg + 'ppf private is nan')
# 定义函数，用于检查分布函数的支持域是否正确
def check_retrieving_support(distfn, args):
    # 设置默认的 loc 和 scale 值
    loc, scale = 1, 2
    # 调用分布函数的 support 方法获取支持域
    supp = distfn.support(*args)
    # 带有指定 loc 和 scale 参数调用分布函数的 support 方法获取支持域
    supp_loc_scale = distfn.support(*args, loc=loc, scale=scale)
    # 使用 numpy 近似相等断言函数检查计算结果
    npt.assert_almost_equal(np.array(supp)*scale + loc,
                            np.array(supp_loc_scale))


# 定义函数，用于检查分布函数的拟合参数是否正确
def check_fit_args(distfn, arg, rvs, method):
    # 忽略所有的运行时警告，并且将特定警告类型过滤掉
    with np.errstate(all='ignore'), npt.suppress_warnings() as sup:
        sup.filter(category=RuntimeWarning,
                   message="The shape parameter of the erlang")
        sup.filter(category=RuntimeWarning,
                   message="floating point number truncated")
        # 调用分布函数的 fit 方法进行参数拟合
        vals = distfn.fit(rvs, method=method)
        vals2 = distfn.fit(rvs, optimizer='powell', method=method)
    # 只检查返回值的长度，精确性在 test_fit.py 中测试
    npt.assert_(len(vals) == 2+len(arg))
    npt.assert_(len(vals2) == 2+len(arg))


# 定义函数，用于检查分布函数的拟合参数是否正确（固定 floc 和 fscale）
def check_fit_args_fix(distfn, arg, rvs, method):
    # 忽略所有的运行时警告，并且将特定警告类型过滤掉
    with np.errstate(all='ignore'), npt.suppress_warnings() as sup:
        sup.filter(category=RuntimeWarning,
                   message="The shape parameter of the erlang")

        # 使用固定的 floc 和 fscale 参数调用分布函数的 fit 方法进行参数拟合
        vals = distfn.fit(rvs, floc=0, method=method)
        vals2 = distfn.fit(rvs, fscale=1, method=method)
        # 检查返回值的长度及具体数值
        npt.assert_(len(vals) == 2+len(arg))
        npt.assert_(vals[-2] == 0)
        npt.assert_(vals2[-1] == 1)
        npt.assert_(len(vals2) == 2+len(arg))
        # 如果参数列表 arg 的长度大于 0，进一步检查返回值中特定位置的数值
        if len(arg) > 0:
            vals3 = distfn.fit(rvs, f0=arg[0], method=method)
            npt.assert_(len(vals3) == 2+len(arg))
            npt.assert_(vals3[0] == arg[0])
        # 如果参数列表 arg 的长度大于 1，进一步检查返回值中特定位置的数值
        if len(arg) > 1:
            vals4 = distfn.fit(rvs, f1=arg[1], method=method)
            npt.assert_(len(vals4) == 2+len(arg))
            npt.assert_(vals4[1] == arg[1])
        # 如果参数列表 arg 的长度大于 2，进一步检查返回值中特定位置的数值
        if len(arg) > 2:
            vals5 = distfn.fit(rvs, f2=arg[2], method=method)
            npt.assert_(len(vals5) == 2+len(arg))
            npt.assert_(vals5[2] == arg[2])


# 定义生成器函数，用于生成测试用例参数
def cases_test_methods_with_lists():
    # 遍历 distcont 列表中的分布名称和参数列表
    for distname, arg in distcont:
        # 如果分布名称在 slow_with_lists 列表中，则标记为慢速测试
        if distname in slow_with_lists:
            yield pytest.param(distname, arg, marks=pytest.mark.slow)
        else:
            yield distname, arg


# 使用参数化装饰器定义测试函数，测试分布函数的方法（接受列表作为参数）
@pytest.mark.parametrize('method', ['pdf', 'logpdf', 'cdf', 'logcdf',
                                    'sf', 'logsf', 'ppf', 'isf'])
@pytest.mark.parametrize('distname, args', cases_test_methods_with_lists())
def test_methods_with_lists(method, distname, args):
    # 获取指定分布函数对象及其方法
    dist = getattr(stats, distname)
    f = getattr(dist, method)
    # 对于特定分布及方法，设置输入参数列表 x
    if distname == 'invweibull' and method.startswith('log'):
        x = [1.5, 2]
    else:
        x = [0.1, 0.2]

    # 将每个参数扩展成二维列表
    shape2 = [[a]*2 for a in args]
    loc = [0, 0.1]
    scale = [1, 1.01]
    # 调用分布函数的方法计算结果，并使用近似相等断言检查
    result = f(x, *shape2, loc=loc, scale=scale)
    npt.assert_allclose(result,
                        [f(*v) for v in zip(x, *shape2, loc, scale)],
                        rtol=1e-14, atol=5e-14)
def test_burr_fisk_moment_gh13234_regression():
    # 计算 Burr 分布的矩，并验证返回结果是否为浮点数
    vals0 = stats.burr.moment(1, 5, 4)
    assert isinstance(vals0, float)

    # 计算 Fisk 分布的矩，并验证返回结果是否为浮点数
    vals1 = stats.fisk.moment(1, 8)
    assert isinstance(vals1, float)


def test_moments_with_array_gh12192_regression():
    # 使用数组 loc 和标量 scale，计算正态分布的矩的期望值，并进行断言比较
    vals0 = stats.norm.moment(order=1, loc=np.array([1, 2, 3]), scale=1)
    expected0 = np.array([1., 2., 3.])
    npt.assert_equal(vals0, expected0)

    # 使用数组 loc 和无效的标量 scale，计算正态分布的矩的期望值，并进行断言比较
    vals1 = stats.norm.moment(order=1, loc=np.array([1, 2, 3]), scale=-1)
    expected1 = np.array([np.nan, np.nan, np.nan])
    npt.assert_equal(vals1, expected1)

    # 使用数组 loc 和数组 scale（包含无效条目），计算正态分布的矩的期望值，并进行断言比较
    vals2 = stats.norm.moment(order=1, loc=np.array([1, 2, 3]),
                              scale=[-3, 1, 0])
    expected2 = np.array([np.nan, 2., np.nan])
    npt.assert_equal(vals2, expected2)

    # 当 loc == 0 且 scale < 0 时，计算正态分布的矩的期望值，并进行断言比较
    vals3 = stats.norm.moment(order=2, loc=0, scale=-4)
    expected3 = np.nan
    npt.assert_equal(vals3, expected3)
    assert isinstance(vals3, expected3.__class__)

    # 使用数组 loc（包含 0 条目）和数组 scale（包含无效条目），计算正态分布的矩的期望值，并进行断言比较
    vals4 = stats.norm.moment(order=2, loc=[1, 0, 2], scale=[3, -4, -5])
    expected4 = np.array([10., np.nan, np.nan])
    npt.assert_equal(vals4, expected4)

    # 当所有 loc == 0 且数组 scale 包含无效条目时，计算正态分布的矩的期望值，并进行断言比较
    vals5 = stats.norm.moment(order=2, loc=[0, 0, 0], scale=[5., -2, 100.])
    expected5 = np.array([25., np.nan, 10000.])
    npt.assert_equal(vals5, expected5)

    # 当所有 loc == 0 且所有 scale < 0 时，计算正态分布的矩的期望值，并进行断言比较
    vals6 = stats.norm.moment(order=2, loc=[0, 0, 0], scale=[-5., -2, -100.])
    expected6 = np.array([np.nan, np.nan, np.nan])
    npt.assert_equal(vals6, expected6)

    # 使用标量参数 df、loc 和 scale，计算卡方分布的矩的期望值，并进行断言比较
    vals7 = stats.chi.moment(order=2, df=1, loc=0, scale=0)
    expected7 = np.nan
    npt.assert_equal(vals7, expected7)
    assert isinstance(vals7, expected7.__class__)

    # 使用数组参数 df、标量 loc 和标量 scale，计算卡方分布的矩的期望值，并进行断言比较
    vals8 = stats.chi.moment(order=2, df=[1, 2, 3], loc=0, scale=0)
    expected8 = np.array([np.nan, np.nan, np.nan])
    npt.assert_equal(vals8, expected8)

    # 使用数组参数 df、数组 loc 和数组 scale，计算卡方分布的矩的期望值，并进行断言比较
    vals9 = stats.chi.moment(order=2, df=[1, 2, 3], loc=[1., 0., 2.],
                             scale=[1., -3., 0.])
    expected9 = np.array([3.59576912, np.nan, np.nan])
    npt.assert_allclose(vals9, expected9, rtol=1e-8)

    # 当 n > 4、所有 loc != 0 且所有 scale != 0 时，计算正态分布的矩的期望值，并进行断言比较
    vals10 = stats.norm.moment(5, [1., 2.], [1., 2.])
    expected10 = np.array([26., 832.])
    npt.assert_allclose(vals10, expected10, rtol=1e-13)

    # 测试广播功能及更多情况
    a = [-1.1, 0, 1, 2.2, np.pi]
    b = [-1.1, 0, 1, 2.2, np.pi]
    loc = [-1.1, 0, np.sqrt(2)]
    scale = [-2.1, 0, 1, 2.2, np.pi]

    a = np.array(a).reshape((-1, 1, 1, 1))
    b = np.array(b).reshape((-1, 1, 1))
    loc = np.array(loc).reshape((-1, 1))
    scale = np.array(scale)
    # 计算 Beta 分布的二阶矩，使用 stats.beta.moment 函数
    vals11 = stats.beta.moment(order=2, a=a, b=b, loc=loc, scale=scale)

    # 使用 np.broadcast_arrays 对 a, b, loc, scale 进行广播，使它们具有相同的形状
    a, b, loc, scale = np.broadcast_arrays(a, b, loc, scale)

    # 遍历广播后的 a 数组的每个元素及其对应的索引
    for i in np.ndenumerate(a):
        # 设置 numpy 的错误状态，忽略无效值和除以零的警告
        with np.errstate(invalid='ignore', divide='ignore'):
            # 获取当前索引值 i[0]
            i = i[0]  # just get the index
            # 使用当前索引对应的参数计算 Beta 分布的二阶矩，与直接使用标量输入的函数进行比较
            expected = stats.beta.moment(order=2, a=a[i], b=b[i],
                                         loc=loc[i], scale=scale[i])
            # 断言计算得到的 vals11[i] 等于预期值 expected
            np.testing.assert_equal(vals11[i], expected)
def test_broadcasting_in_moments_gh12192_regression():
    # 计算正态分布的一阶矩，对于 loc 参数为数组，scale 参数为标量的情况
    vals0 = stats.norm.moment(order=1, loc=np.array([1, 2, 3]), scale=[[1]])
    expected0 = np.array([[1., 2., 3.]])
    # 断言计算结果与预期结果相等
    npt.assert_equal(vals0, expected0)
    # 断言结果的形状与预期结果的形状相等
    assert vals0.shape == expected0.shape

    # 计算正态分布的一阶矩，对于 loc 参数为二维数组，scale 参数为一维数组的情况
    vals1 = stats.norm.moment(order=1, loc=np.array([[1], [2], [3]]),
                              scale=[1, 2, 3])
    expected1 = np.array([[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]])
    # 断言计算结果与预期结果相等
    npt.assert_equal(vals1, expected1)
    # 断言结果的形状与预期结果的形状相等
    assert vals1.shape == expected1.shape

    # 计算卡方分布的一阶矩，对于 df 参数为一维数组，loc 和 scale 参数为标量的情况
    vals2 = stats.chi.moment(order=1, df=[1., 2., 3.], loc=0., scale=1.)
    expected2 = np.array([0.79788456, 1.25331414, 1.59576912])
    # 断言计算结果与预期结果在相对误差容限下（rtol=1e-8）相近
    npt.assert_allclose(vals2, expected2, rtol=1e-8)
    # 断言结果的形状与预期结果的形状相等
    assert vals2.shape == expected2.shape

    # 计算卡方分布的一阶矩，对于 df 参数为二维数组，loc 和 scale 参数分别为数组的情况
    vals3 = stats.chi.moment(order=1, df=[[1.], [2.], [3.]], loc=[0., 1., 2.],
                             scale=[-1., 0., 3.])
    expected3 = np.array([[np.nan, np.nan, 4.39365368],
                          [np.nan, np.nan, 5.75994241],
                          [np.nan, np.nan, 6.78730736]])
    # 断言计算结果与预期结果在相对误差容限下（rtol=1e-8）相近
    npt.assert_allclose(vals3, expected3, rtol=1e-8)
    # 断言结果的形状与预期结果的形状相等
    assert vals3.shape == expected3.shape


@pytest.mark.slow
def test_kappa3_array_gh13582():
    # https://github.com/scipy/scipy/pull/15140#issuecomment-994958241
    # 定义形状参数列表和矩参数列表
    shapes = [0.5, 1.5, 2.5, 3.5, 4.5]
    moments = 'mvsk'
    # 计算 kappa3 分布的统计量，返回结果为二维数组
    res = np.array([[stats.kappa3.stats(shape, moments=moment)
                   for shape in shapes] for moment in moments])
    # 计算 kappa3 分布的统计量，返回结果为一维数组
    res2 = np.array(stats.kappa3.stats(shapes, moments=moments))
    # 断言两种计算方式的结果相等
    npt.assert_allclose(res, res2)


@pytest.mark.xslow
def test_kappa4_array_gh13582():
    # 定义 h 和 k 参数的数组
    h = np.array([-0.5, 2.5, 3.5, 4.5, -3])
    k = np.array([-0.5, 1, -1.5, 0, 3.5])
    moments = 'mvsk'
    # 计算 kappa4 分布的统计量，返回结果为二维数组
    res = np.array([[stats.kappa4.stats(h[i], k[i], moments=moment)
                   for i in range(5)] for moment in moments])
    # 计算 kappa4 分布的统计量，返回结果为一维数组
    res2 = np.array(stats.kappa4.stats(h, k, moments=moments))
    # 断言两种计算方式的结果相等
    npt.assert_allclose(res, res2)

    # https://github.com/scipy/scipy/pull/15250#discussion_r775112913
    # 定义 h 和 k 参数的数组
    h = np.array([-1, -1/4, -1/4, 1, -1, 0])
    k = np.array([1, 1, 1/2, -1/3, -1, 0])
    # 计算 kappa4 分布的统计量，返回结果为二维数组
    res = np.array([[stats.kappa4.stats(h[i], k[i], moments=moment)
                   for i in range(6)] for moment in moments])
    # 计算 kappa4 分布的统计量，返回结果为一维数组
    res2 = np.array(stats.kappa4.stats(h, k, moments=moments))
    # 断言两种计算方式的结果相等
    npt.assert_allclose(res, res2)

    # https://github.com/scipy/scipy/pull/15250#discussion_r775115021
    # 定义 h 和 k 参数的数组
    h = np.array([-1, -0.5, 1])
    k = np.array([-1, -0.5, 0, 1])[:, None]
    # 计算 kappa4 分布的统计量，返回结果为三维数组
    res2 = np.array(stats.kappa4.stats(h, k, moments=moments))
    # 断言结果的形状为 (4, 4, 3)
    assert res2.shape == (4, 4, 3)


def test_frozen_attributes():
    # gh-14827 报告所有冻结分布都同时具有 pmf 和 pdf 属性；连续分布应具有 pdf 属性，离散分布应具有 pmf 属性。
    message = "'rv_continuous_frozen' object has no attribute"
    # 使用 pytest 检查连续分布的冻结对象是否缺少 pmf 属性
    with pytest.raises(AttributeError, match=message):
        stats.norm().pmf
    # 使用 pytest 检查连续分布的冻结对象是否缺少 logpmf 属性
    with pytest.raises(AttributeError, match=message):
        stats.norm().logpmf
    # 将 stats.norm 的 pmf 属性设置为字符串 "herring"
    stats.norm.pmf = "herring"
    # 使用 stats.norm 创建一个冻结的正态分布对象
    frozen_norm = stats.norm()
    # 断言 frozen_norm 是 rv_continuous_frozen 类的实例
    assert isinstance(frozen_norm, rv_continuous_frozen)
    # 删除 stats.norm 的 pmf 属性
    delattr(stats.norm, 'pmf')
# 定义测试函数，用于测试特定的概率分布函数在不同输入条件下的行为
def test_skewnorm_pdf_gh16038():
    # 使用默认随机数生成器创建一个随机数对象
    rng = np.random.default_rng(0)
    # 初始化输入变量 x 和参数 a
    x, a = -np.inf, 0
    # 断言 skewnorm.pdf 函数在 x = -inf，a = 0 时的返回值等于标准正态分布的概率密度函数在 x = -inf 时的返回值
    npt.assert_equal(stats.skewnorm.pdf(x, a), stats.norm.pdf(x))
    
    # 生成一个 3x3 的随机数组 x 和参数数组 a
    x, a = rng.random(size=(3, 3)), rng.random(size=(3, 3))
    # 生成一个 3x3 的随机布尔掩码数组，用于在参数数组 a 中随机置零元素
    mask = rng.random(size=(3, 3)) < 0.5
    a[mask] = 0
    # 提取符合掩码条件的 x 中的元素，形成一维数组 x_norm
    x_norm = x[mask]
    # 计算 skewnorm.pdf 在输入 x, a 下的结果
    res = stats.skewnorm.pdf(x, a)
    # 断言结果数组 res 中符合掩码条件的元素等于标准正态分布的概率密度函数在 x_norm 上的结果
    npt.assert_equal(res[mask], stats.norm.pdf(x_norm))
    # 断言结果数组 res 中不符合掩码条件的元素等于 skewnorm.pdf 在相应 x, a 下的计算结果
    npt.assert_equal(res[~mask], stats.skewnorm.pdf(x[~mask], a[~mask]))


# 参数化测试，针对单一输入的函数应该返回单一输出的情况
scalar_out = [['rvs', []], ['pdf', [0]], ['logpdf', [0]], ['cdf', [0]],
              ['logcdf', [0]], ['sf', [0]], ['logsf', [0]], ['ppf', [0]],
              ['isf', [0]], ['moment', [1]], ['entropy', []], ['expect', []],
              ['median', []], ['mean', []], ['std', []], ['var', []]]
scalars_out = [['interval', [0.95]], ['support', []], ['stats', ['mv']]]


@pytest.mark.parametrize('case', scalar_out + scalars_out)
def test_scalar_for_scalar(case):
    # 针对某些 rv_continuous 函数返回 0 维数组而非 NumPy 标量的问题进行测试
    # 防止出现回归问题
    method_name, args = case
    # 获取 stats.norm 对象中的指定方法名的方法
    method = getattr(stats.norm(), method_name)
    # 调用方法并获取结果
    res = method(*args)
    # 如果 case 属于 scalar_out 中的情况，则断言返回结果为 NumPy 标量
    if case in scalar_out:
        assert isinstance(res, np.number)
    else:
        # 否则，断言返回结果的第一个和第二个元素分别为 NumPy 标量
        assert isinstance(res[0], np.number)
        assert isinstance(res[1], np.number)


def test_scalar_for_scalar2():
    # 测试那些不是冻结分布属性的方法
    # 对给定数据集使用 stats.norm.fit 方法，并断言返回结果的第一个和第二个元素为 NumPy 标量
    res = stats.norm.fit([1, 2, 3])
    assert isinstance(res[0], np.number)
    assert isinstance(res[1], np.number)
    
    # 对给定数据集使用 stats.norm.fit_loc_scale 方法，并断言返回结果的第一个和第二个元素为 NumPy 标量
    res = stats.norm.fit_loc_scale([1, 2, 3])
    assert isinstance(res[0], np.number)
    assert isinstance(res[1], np.number)
    
    # 对给定数据集使用 stats.norm.nnlf 方法，并断言返回结果为 NumPy 标量
    res = stats.norm.nnlf((0, 1), [1, 2, 3])
    assert isinstance(res, np.number)
```