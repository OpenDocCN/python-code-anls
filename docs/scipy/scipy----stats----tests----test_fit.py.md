# `D:\src\scipysrc\scipy\scipy\stats\tests\test_fit.py`

```
import os
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
import pytest
from scipy import stats
from scipy.optimize import differential_evolution

from .test_continuous_basic import distcont
from scipy.stats._distn_infrastructure import FitError
from scipy.stats._distr_params import distdiscrete
from scipy.stats import goodness_of_fit

# 不是一个正式的统计收敛测试，仅仅验证估计值和真实值之间的差异是否太大
# 这些是用来尝试的样本大小
fit_sizes = [1000, 5000, 10000]

# 估计值和真实值之间的百分比阈值，用于判断是否失败
thresh_percent = 0.25

# 估计值和真实值之间的最小差异阈值，低于此值则判定测试失败
thresh_min = 0.75

# MLE 拟合失败的分布列表
mle_failing_fits = [
        'gausshyper',
        'genexpon',
        'gengamma',
        'irwinhall',
        'kappa4',
        'ksone',
        'kstwo',
        'ncf',
        'ncx2',
        'truncexpon',
        'tukeylambda',
        'vonmises',
        'levy_stable',
        'trapezoid',
        'truncweibull_min',
        'studentized_range',
]

# 这些通过但是非常慢 (>1秒)
mle_Xslow_fits = ['betaprime', 'crystalball', 'exponweib', 'f', 'geninvgauss',
                  'jf_skew_t', 'recipinvgauss', 'rel_breitwigner', 'vonmises_line']

# 当所有参数被拟合时，这些分布的MLE拟合方法表现不佳，所以将位置参数固定在0进行测试
mle_use_floc0 = [
    'burr',
    'chi',
    'chi2',
    'mielke',
    'pearson3',
    'genhalflogistic',
    'rdist',
    'pareto',
    'powerlaw',  # distfn.nnlf(est2, rvs) > distfn.nnlf(est1, rvs) otherwise
    'powerlognorm',
    'wrapcauchy',
    'rel_breitwigner',
]

# MM 拟合失败的分布列表
mm_failing_fits = ['alpha', 'betaprime', 'burr', 'burr12', 'cauchy', 'chi',
                   'chi2', 'crystalball', 'dgamma', 'dweibull', 'f',
                   'fatiguelife', 'fisk', 'foldcauchy', 'genextreme',
                   'gengamma', 'genhyperbolic', 'gennorm', 'genpareto',
                   'halfcauchy', 'invgamma', 'invweibull', 'irwinhall', 'jf_skew_t',
                   'johnsonsu', 'kappa3', 'ksone', 'kstwo', 'levy', 'levy_l',
                   'levy_stable', 'loglaplace', 'lomax', 'mielke', 'nakagami',
                   'ncf', 'nct', 'ncx2', 'pareto', 'powerlognorm', 'powernorm',
                   'rel_breitwigner', 'skewcauchy', 't', 'trapezoid', 'triang',
                   'truncpareto', 'truncweibull_min', 'tukeylambda',
                   'studentized_range']

# 这些也许不会失败，但是测试耗时很长
mm_XXslow_fits = ['argus', 'exponpow', 'exponweib', 'gausshyper', 'genexpon',
                  'genhalflogistic', 'halfgennorm', 'gompertz', 'johnsonsb',
                  'kappa4', 'kstwobign', 'recipinvgauss',
                  'truncexpon', 'vonmises', 'vonmises_line']

# 这些通过但是非常慢 (>1秒)
mm_Xslow_fits = ['wrapcauchy']

# 失败的拟合分布类型字典，包含MLE和MM两种情况
failing_fits = {"MM": mm_failing_fits + mm_XXslow_fits, "MLE": mle_failing_fits}
xslow_fits = {"MM": mm_Xslow_fits, "MLE": mle_Xslow_fits}
# 定义一个字典，包含使用 MM 和 MLE 方法的 Xslow 拟合函数

fail_interval_censored = {"truncpareto"}
# 包含一个字符串集合，表示需要跳过拟合测试的截断 Pareto 分布

# Don't run the fit test on these:
# 不要在以下分布上运行拟合测试：
skip_fit = [
    'erlang',  # Subclass of gamma, generates a warning.
    'genhyperbolic', 'norminvgauss',  # too slow
]

def cases_test_cont_fit():
    # this tests the closeness of the estimated parameters to the true
    # parameters with fit method of continuous distributions
    # Note: is slow, some distributions don't converge with sample
    # size <= 10000
    # 这个函数测试连续分布的拟合方法估计参数与真实参数之间的接近程度。
    # 注意：速度较慢，某些分布在样本大小 <= 10000 时不会收敛。
    for distname, arg in distcont:
        if distname not in skip_fit:
            yield distname, arg

@pytest.mark.slow
@pytest.mark.parametrize('distname,arg', cases_test_cont_fit())
@pytest.mark.parametrize('method', ["MLE", "MM"])
def test_cont_fit(distname, arg, method):
    run_xfail = int(os.getenv('SCIPY_XFAIL', default=False))
    run_xslow = int(os.getenv('SCIPY_XSLOW', default=False))

    if distname in failing_fits[method] and not run_xfail:
        # The generic `fit` method can't be expected to work perfectly for all
        # distributions, data, and guesses. Some failures are expected.
        # 通用的 `fit` 方法不能期望对所有分布、数据和猜测都能完美工作。一些失败是可以预期的。
        msg = "Failure expected; set environment variable SCIPY_XFAIL=1 to run."
        pytest.xfail(msg)

    if distname in xslow_fits[method] and not run_xslow:
        # Very slow; set environment variable SCIPY_XSLOW=1 to run.
        # 非常慢；设置环境变量 SCIPY_XSLOW=1 来运行。
        msg = "Very slow; set environment variable SCIPY_XSLOW=1 to run."
        pytest.skip(msg)

    distfn = getattr(stats, distname)

    truearg = np.hstack([arg, [0.0, 1.0]])
    diffthreshold = np.max(np.vstack([truearg*thresh_percent,
                                      np.full(distfn.numargs+2, thresh_min)]),
                           0)
    # 对于每个给定的 fit_size 进行迭代
    for fit_size in fit_sizes:
        # 设定随机种子以保证结果的可复现性
        np.random.seed(1234)

        # 忽略所有的数值警告
        with np.errstate(all='ignore'):
            # 从分布 distfn 中生成指定大小的随机变量样本 rvs
            rvs = distfn.rvs(size=fit_size, *arg)
            
            # 根据方法选择是否设置 floc=0 的关键字参数
            if method == 'MLE' and distfn.name in mle_use_floc0:
                kwds = {'floc': 0}
            else:
                kwds = {}
            
            # 使用 distfn.fit 方法拟合 rvs 数据，得到参数估计值 est
            est = distfn.fit(rvs, method=method, **kwds)
            
            # 如果方法为 'MLE'，则进行关于 CensoredData 的简单测试
            if method == 'MLE':
                # 创建 CensoredData 对象 data1，用于检查数据是否包含截断数据
                data1 = stats.CensoredData(rvs)
                
                # 使用 distfn.fit 方法拟合 CensoredData 对象 data1，得到参数估计值 est1
                est1 = distfn.fit(data1, **kwds)
                
                # 检查拟合结果 est1 是否与 est 相同，若不同则引发错误
                msg = ('Different results fitting uncensored data wrapped as'
                       f' CensoredData: {distfn.name}: est={est} est1={est1}')
                assert_allclose(est1, est, rtol=1e-10, err_msg=msg)
            
            # 如果方法为 'MLE'，并且 distname 不在 fail_interval_censored 中
            if method == 'MLE' and distname not in fail_interval_censored:
                # 将 rvs 中的前 nic 个值转换为区间截断值
                nic = 15
                interval = np.column_stack((rvs, rvs))
                interval[:nic, 0] *= 0.99
                interval[:nic, 1] *= 1.01
                interval.sort(axis=1)
                
                # 创建 CensoredData 对象 data2，用于拟合区间截断数据
                data2 = stats.CensoredData(interval=interval)
                
                # 使用 distfn.fit 方法拟合 CensoredData 对象 data2，得到参数估计值 est2
                est2 = distfn.fit(data2, **kwds)
                
                # 检查拟合结果 est2 是否与 est 相同，若不同则引发错误
                msg = ('Different results fitting interval-censored'
                       f' data: {distfn.name}: est={est} est2={est2}')
                assert_allclose(est2, est, rtol=0.05, err_msg=msg)

        # 计算估计值 est 与真实参数 truearg 之间的差异
        diff = est - truearg

        # 为位置参数设置阈值
        diffthreshold[-2] = np.max([np.abs(rvs.mean())*thresh_percent,
                                    thresh_min])

        # 如果估计值 est 中存在 NaN，则引发错误
        if np.any(np.isnan(est)):
            raise AssertionError('nan returned in fit')
        else:
            # 如果所有估计值 est 的绝对差都在设定的阈值内，则结束循环
            if np.all(np.abs(diff) <= diffthreshold):
                break
    else:
        # 如果循环正常结束而未找到满足条件的拟合结果，则引发错误并打印详细信息
        txt = 'parameter: %s\n' % str(truearg)
        txt += 'estimated: %s\n' % str(est)
        txt += 'diff     : %s\n' % str(diff)
        raise AssertionError('fit not very good in %s\n' % distfn.name + txt)
# 定义函数 _check_loc_scale_mle_fit，用于检查特定概率分布的最大似然估计 (MLE) 的位置参数和尺度参数是否符合预期
def _check_loc_scale_mle_fit(name, data, desired, atol=None):
    # 从 scipy.stats 模块中获取指定名称的概率分布函数对象
    d = getattr(stats, name)
    # 使用给定数据进行最大似然估计，提取出位置参数和尺度参数的估计结果
    actual = d.fit(data)[-2:]
    # 断言位置参数和尺度参数的估计结果与期望值在指定容差范围内相近，否则抛出 AssertionError
    assert_allclose(actual, desired, atol=atol,
                    err_msg='poor mle fit of (loc, scale) in %s' % name)


# 定义函数 test_non_default_loc_scale_mle_fit，测试非默认位置和尺度参数的最大似然估计
def test_non_default_loc_scale_mle_fit():
    # 提供测试数据
    data = np.array([1.01, 1.78, 1.78, 1.78, 1.88, 1.88, 1.88, 2.00])
    # 分别对均匀分布和指数分布进行非默认位置和尺度参数的最大似然估计检验
    _check_loc_scale_mle_fit('uniform', data, [1.01, 0.99], 1e-3)
    _check_loc_scale_mle_fit('expon', data, [1.01, 0.73875], 1e-3)


# 定义函数 test_expon_fit，测试指数分布的最大似然估计
def test_expon_fit():
    """gh-6167"""
    # 提供测试数据
    data = [0, 0, 0, 0, 2, 2, 2, 2]
    # 对指数分布进行最大似然估计，指定分布的位置参数为 0
    phat = stats.expon.fit(data, floc=0)
    # 断言估计结果与期望值在指定容差范围内相近，否则抛出 AssertionError
    assert_allclose(phat, [0, 1.0], atol=1e-3)


# 定义函数 test_fit_error，测试对 fit 函数异常情况的处理
def test_fit_error():
    # 提供测试数据，包含 29 个零和 21 个一的组合
    data = np.concatenate([np.zeros(29), np.ones(21)])
    # 设置期望的错误消息
    message = "Optimization converged to parameters that are..."
    # 使用 pytest.raises 检测 FitError 异常，同时期望触发 RuntimeWarning 警告
    with pytest.raises(FitError, match=message), \
            pytest.warns(RuntimeWarning):
        stats.beta.fit(data)


# 使用 pytest.mark.parametrize 对不同概率分布进行 nnlf 和相关方法的测试
@pytest.mark.parametrize("dist, params",
                         [(stats.norm, (0.5, 2.5)),  # type: ignore[attr-defined]
                          (stats.binom, (10, 0.3, 2))])  # type: ignore[attr-defined]
def test_nnlf_and_related_methods(dist, params):
    # 创建随机数生成器对象
    rng = np.random.default_rng(983459824)

    # 根据概率分布是否具有 pdf 或者 logpmf 方法来选择 logpxf 函数
    if hasattr(dist, 'pdf'):
        logpxf = dist.logpdf
    else:
        logpxf = dist.logpmf

    # 从指定概率分布生成随机样本数据
    x = dist.rvs(*params, size=100, random_state=rng)
    # 计算参考值，即负对数似然函数的期望值
    ref = -logpxf(x, *params).sum()
    # 计算 nnlf 和 _penalized_nnlf 方法的结果，并与参考值进行断言比较
    res1 = dist.nnlf(params, x)
    res2 = dist._penalized_nnlf(params, x)
    assert_allclose(res1, ref)
    assert_allclose(res2, ref)


# 定义函数 cases_test_fit_mle，测试不同概率分布的最大似然估计是否存在问题或者超时
def cases_test_fit_mle():
    # 列出默认测试失败或超时的概率分布集合
    skip_basic_fit = {'argus', 'irwinhall', 'foldnorm', 'truncpareto',
                      'truncweibull_min', 'ksone', 'levy_stable',
                      'studentized_range', 'kstwo', 'arcsine'}

    # 列出慢速测试失败或超时的概率分布集合
    slow_basic_fit = {'alpha', 'betaprime', 'binom', 'bradford', 'burr12',
                      'chi', 'crystalball', 'dweibull', 'erlang', 'exponnorm',
                      'exponpow', 'f', 'fatiguelife', 'fisk', 'foldcauchy', 'gamma',
                      'genexpon', 'genextreme', 'gennorm', 'genpareto',
                      'gompertz', 'halfgennorm', 'invgamma', 'invgauss', 'invweibull',
                      'jf_skew_t', 'johnsonsb', 'johnsonsu', 'kappa3',
                      'kstwobign', 'loglaplace', 'lognorm', 'lomax', 'mielke',
                      'nakagami', 'nbinom', 'norminvgauss',
                      'pareto', 'pearson3', 'powerlaw', 'powernorm',
                      'randint', 'rdist', 'recipinvgauss', 'rice', 'skewnorm',
                      't', 'uniform', 'weibull_max', 'weibull_min', 'wrapcauchy'}

    # 保持列表中的概率分布名称按字母顺序排列
    # 定义一个包含概率分布名称的集合，用于基本拟合测试中标记为“极慢”的分布
    xslow_basic_fit = {'beta', 'betabinom', 'betanbinom', 'burr', 'exponweib',
                       'gausshyper', 'gengamma', 'genhalflogistic',
                       'genhyperbolic', 'geninvgauss',
                       'hypergeom', 'kappa4', 'loguniform',
                       'ncf', 'nchypergeom_fisher', 'nchypergeom_wallenius',
                       'nct', 'ncx2', 'nhypergeom',
                       'powerlognorm', 'reciprocal', 'rel_breitwigner',
                       'skellam', 'trapezoid', 'triang', 'truncnorm',
                       'tukeylambda', 'vonmises', 'zipfian'}
    
    # 遍历由混合的离散和连续分布字典组成的列表
    for dist in dict(distdiscrete + distcont):
        # 如果分布在跳过基本拟合的集合中，或者分布不是字符串类型，则标记为“单独测试”
        if dist in skip_basic_fit or not isinstance(dist, str):
            reason = "tested separately"
            # 使用 pytest 的参数化功能跳过此测试，并添加跳过原因
            yield pytest.param(dist, marks=pytest.mark.skip(reason=reason))
        # 如果分布在慢速基本拟合的集合中，则标记为“太慢（>= 0.25秒）”
        elif dist in slow_basic_fit:
            reason = "too slow (>= 0.25s)"
            # 使用 pytest 的参数化功能标记为慢速测试，并添加原因
            yield pytest.param(dist, marks=pytest.mark.slow(reason=reason))
        # 如果分布在极慢基本拟合的集合中，则标记为“太慢（>= 1.0秒）”
        elif dist in xslow_basic_fit:
            reason = "too slow (>= 1.0s)"
            # 使用 pytest 的参数化功能标记为极慢测试，并添加原因
            yield pytest.param(dist, marks=pytest.mark.xslow(reason=reason))
        else:
            # 否则，正常执行测试
            yield dist
# 定义一个函数，用于测试拟合模型的均方误差（MSE）
def cases_test_fit_mse():
    # 这些测试用例因为运行速度很慢，可能无法通过测试
    skip_basic_fit = {'levy_stable', 'studentized_range', 'ksone', 'skewnorm',
                      'irwinhall',  # 在运行时会挂起
                      'norminvgauss',  # 超级慢（大约1小时），但能通过
                      'kstwo',  # 非常慢（大约25分钟），但能通过
                      'geninvgauss',  # 相当慢（大约4分钟），但能通过
                      'gausshyper', 'genhyperbolic',  # 集成警告
                      'tukeylambda',  # 接近，但不满足容差
                      'vonmises',  # 可能有负的累积分布函数；表现不佳
                      'argus'}  # 不满足容差；单独测试

    # 请保持以下列表按字母顺序排列...
    slow_basic_fit = {'alpha', 'anglit', 'arcsine', 'betabinom', 'bradford',
                      'chi', 'chi2', 'crystalball', 'dweibull',
                      'erlang', 'exponnorm', 'exponpow', 'exponweib',
                      'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm',
                      'gamma', 'genexpon', 'genextreme', 'genhalflogistic',
                      'genlogistic', 'genpareto', 'gompertz',
                      'hypergeom', 'invweibull',
                      'johnsonsu', 'kappa3', 'kstwobign',
                      'laplace_asymmetric', 'loggamma', 'loglaplace',
                      'lognorm', 'lomax',
                      'maxwell', 'nhypergeom',
                      'pareto', 'powernorm', 'randint', 'recipinvgauss',
                      'semicircular',
                      't', 'triang', 'truncexpon', 'truncpareto',
                      'uniform',
                      'wald', 'weibull_max', 'weibull_min', 'wrapcauchy'}

    # 请保持以下列表按字母顺序排列...
    xslow_basic_fit = {'argus', 'beta', 'betaprime', 'burr', 'burr12',
                       'dgamma', 'f', 'gengamma', 'gennorm',
                       'halfgennorm', 'invgamma', 'invgauss', 'jf_skew_t',
                       'johnsonsb', 'kappa4', 'loguniform', 'mielke',
                       'nakagami', 'ncf', 'nchypergeom_fisher',
                       'nchypergeom_wallenius', 'nct', 'ncx2',
                       'pearson3', 'powerlaw', 'powerlognorm',
                       'rdist', 'reciprocal', 'rel_breitwigner', 'rice',
                       'trapezoid', 'truncnorm', 'truncweibull_min',
                       'vonmises_line', 'zipfian'}

    warns_basic_fit = {'skellam'}  # 可以在gh-14901解决后删除此标记
    # 遍历由离散和连续分布合并而成的分布列表
    for dist in dict(distdiscrete + distcont):
        # 如果分布在跳过基本拟合的列表中，或者不是字符串类型，则跳过测试
        if dist in skip_basic_fit or not isinstance(dist, str):
            reason = "Fails. Oh well."
            # 使用 pytest 的标记跳过测试，并添加跳过原因
            yield pytest.param(dist, marks=pytest.mark.skip(reason=reason))
        # 如果分布在慢速基本拟合列表中，添加慢速测试标记
        elif dist in slow_basic_fit:
            reason = "too slow (>= 0.25s)"
            yield pytest.param(dist, marks=pytest.mark.slow(reason=reason))
        # 如果分布在非常慢速基本拟合列表中，添加非常慢速测试标记
        elif dist in xslow_basic_fit:
            reason = "too slow (>= 1.0s)"
            yield pytest.param(dist, marks=pytest.mark.xslow(reason=reason))
        # 如果分布在警告基本拟合列表中，忽略运行时警告
        elif dist in warns_basic_fit:
            mark = pytest.mark.filterwarnings('ignore::RuntimeWarning')
            yield pytest.param(dist, marks=mark)
        else:
            # 否则正常执行测试
            yield dist
# 定义一个生成器函数，用于测试分布的起始拟合情况
def cases_test_fitstart():
    # 遍历包含分布名称和形状的字典的项目
    for distname, shapes in dict(distcont).items():
        # 如果 distname 不是字符串类型，或者在 {'studentized_range', 'recipinvgauss'} 中，跳过此次循环
        if (not isinstance(distname, str) or
                distname in {'studentized_range', 'recipinvgauss'}):  # slow
            continue
        # 返回 distname 和 shapes，作为生成器的一部分
        yield distname, shapes


# 使用 pytest 的 parametrize 装饰器为 test_fitstart 函数参数化测试
@pytest.mark.parametrize('distname, shapes', cases_test_fitstart())
def test_fitstart(distname, shapes):
    # 根据 distname 获取对应的分布对象
    dist = getattr(stats, distname)
    # 使用种子值 216342614 创建随机数生成器
    rng = np.random.default_rng(216342614)
    # 生成长度为 10 的随机数据
    data = rng.random(10)

    # 忽略无效值和除以零的错误
    with np.errstate(invalid='ignore', divide='ignore'):  # irrelevant to test
        # 调用分布对象的 _fitstart 方法进行初步拟合
        guess = dist._fitstart(data)

    # 断言调用分布对象的 _argcheck 方法，传入 guess 的倒数第三个元素作为参数
    assert dist._argcheck(*guess[:-2])


# 定义一个辅助函数，用于断言给定分布对象的负对数似然函数值是否小于或接近于给定参数值
def assert_nlff_less_or_close(dist, data, params1, params0, rtol=1e-7, atol=0,
                              nlff_name='nnlf'):
    # 根据 nlff_name 获取分布对象的负对数似然函数方法
    nlff = getattr(dist, nlff_name)
    # 分别计算使用 params1 和 params0 参数时的负对数似然函数值
    nlff1 = nlff(params1, data)
    nlff0 = nlff(params0, data)
    # 如果 nlff1 不小于 nlff0，则触发异常
    if not (nlff1 < nlff0):
        np.testing.assert_allclose(nlff1, nlff0, rtol=rtol, atol=atol)


# 定义一个测试类 TestFit，用于测试分布的拟合过程
class TestFit:
    # 设置默认使用二项分布作为测试的分布对象
    dist = stats.binom  # type: ignore[attr-defined]
    # 设定随机数生成器的种子值
    seed = 654634816187
    # 使用给定种子值创建随机数生成器对象
    rng = np.random.default_rng(seed)
    # 生成长度为 100 的二项分布随机样本数据
    data = stats.binom.rvs(5, 0.5, size=100, random_state=rng)  # type: ignore[attr-defined]  # noqa: E501
    # 设置二项分布参数 n 和 p 的取值范围
    shape_bounds_a = [(1, 10), (0, 1)]
    # 设置二项分布参数 n 和 p 的取值范围，使用字典形式
    shape_bounds_d = {'n': (1, 10), 'p': (0, 1)}
    # 设置比较浮点数时的容差值
    atol = 5e-2
    rtol = 1e-2
    # 将容差值存储在字典中
    tols = {'atol': atol, 'rtol': rtol}

    # 定义一个方法 opt，用于调用 differential_evolution 函数
    def opt(self, *args, **kwds):
        return differential_evolution(*args, seed=0, **kwds)

    # 定义一个测试方法 test_dist_iv，用于测试分布对象和数据是否有效
    def test_dist_iv(self):
        message = "`dist` must be an instance of..."
        # 断言调用 stats.fit 方法时，当 dist 不是一个有效的分布对象时会触发 ValueError 异常
        with pytest.raises(ValueError, match=message):
            stats.fit(10, self.data, self.shape_bounds_a)

    # 定义一个测试方法 test_data_iv，用于测试数据是否有效
    def test_data_iv(self):
        message = "`data` must be exactly one-dimensional."
        # 断言调用 stats.fit 方法时，当 data 不是一维数组时会触发 ValueError 异常
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, [[1, 2, 3]], self.shape_bounds_a)

        message = "All elements of `data` must be finite numbers."
        # 断言调用 stats.fit 方法时，当 data 中包含非有限数值（如 NaN 或无穷大）时会触发 ValueError 异常
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, [1, 2, 3, np.nan], self.shape_bounds_a)
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, [1, 2, 3, np.inf], self.shape_bounds_a)
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, ['1', '2', '3'], self.shape_bounds_a)
    # 定义一个测试方法，用于测试参数边界情况
    def test_bounds_iv(self):
        # 准备测试警告信息
        message = "Bounds provided for the following unrecognized..."
        # 设置一个参数边界字典
        shape_bounds = {'n': (1, 10), 'p': (0, 1), '1': (0, 10)}
        # 使用 pytest 来捕获 RuntimeWarning，确保特定警告消息被触发
        with pytest.warns(RuntimeWarning, match=message):
            # 调用 stats 模块的 fit 方法，传入分布、数据和参数边界
            stats.fit(self.dist, self.data, shape_bounds)

        # 准备测试异常信息
        message = "Each element of a `bounds` sequence must be a tuple..."
        # 设置一个错误的参数边界列表
        shape_bounds = [(1, 10, 3), (0, 1)]
        # 使用 pytest 来捕获 ValueError，确保特定异常消息被触发
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, shape_bounds)

        # 准备测试异常信息
        message = "Each element of `bounds` must be a tuple specifying..."
        # 设置另一个错误的参数边界列表
        shape_bounds = [(1, 10, 3), (0, 1, 0.5)]
        # 使用 pytest 来捕获 ValueError，确保特定异常消息被触发
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, shape_bounds)
        
        # 准备测试异常信息
        message = "A `bounds` sequence must contain at least 2 elements..."
        # 设置一个包含不足两个元素的参数边界列表
        shape_bounds = [(1, 10)]
        # 使用 pytest 来捕获 ValueError，确保特定异常消息被触发
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, shape_bounds)
        
        # 准备测试异常信息
        message = "A `bounds` sequence may not contain more than 3 elements..."
        # 设置一个包含超过三个元素的参数边界列表
        bounds = [(1, 10), (1, 10), (1, 10), (1, 10)]
        # 使用 pytest 来捕获 ValueError，确保特定异常消息被触发
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, bounds)
        
        # 准备测试异常信息
        message = "There are no values for `p` on the interval..."
        # 设置一个包含不合法值的参数边界字典
        shape_bounds = {'n': (1, 10), 'p': (1, 0)}
        # 使用 pytest 来捕获 ValueError，确保特定异常消息被触发
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, shape_bounds)
        
        # 准备测试异常信息
        message = "There are no values for `n` on the interval..."
        # 设置一个不合法的参数边界列表
        shape_bounds = [(10, 1), (0, 1)]
        # 使用 pytest 来捕获 ValueError，确保特定异常消息被触发
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, shape_bounds)
        
        # 准备测试异常信息
        message = "There are no integer values for `n` on the interval..."
        # 设置一个包含非整数值的参数边界列表
        shape_bounds = [(1.4, 1.6), (0, 1)]
        # 使用 pytest 来捕获 ValueError，确保特定异常消息被触发
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, shape_bounds)
        
        # 准备测试异常信息
        message = "The intersection of user-provided bounds for `n`"
        # 使用 pytest 来捕获 ValueError，确保特定异常消息被触发
        with pytest.raises(ValueError, match=message):
            # 调用 stats 模块的 fit 方法，传入分布和数据，但未提供参数边界
            stats.fit(self.dist, self.data)
        # 设置一个非法的参数边界列表
        shape_bounds = [(-np.inf, np.inf), (0, 1)]
        # 使用 pytest 来捕获 ValueError，确保特定异常消息被触发
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, shape_bounds)
    # 定义测试函数 test_guess_iv，用于测试猜测参数 `guess` 的不同情况下的行为
    def test_guess_iv(self):
        # 设置测试消息字符串，用于匹配运行时警告
        message = "Guesses provided for the following unrecognized..."
        # 定义一个有效的猜测字典
        guess = {'n': 1, 'p': 0.5, '1': 255}
        # 使用 pytest.warns 检查是否会触发 RuntimeWarning，并匹配消息
        with pytest.warns(RuntimeWarning, match=message):
            # 调用 stats.fit 方法，传入分布、数据、形状边界和猜测字典作为参数
            stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)

        # 设置另一个测试消息字符串，用于匹配 ValueError
        message = "Each element of `guess` must be a scalar..."
        
        # 定义多个无效的猜测类型，检查是否会引发 ValueError 并匹配消息
        guess = {'n': 1, 'p': 'hi'}
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)
        guess = [1, 'f']
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)
        guess = [[1, 2]]
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)

        # 设置另一个测试消息字符串，用于匹配 ValueError
        message = "A `guess` sequence must contain at least 2..."

        # 定义一个长度不足的猜测列表，检查是否会引发 ValueError 并匹配消息
        guess = [1]
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)

        # 设置另一个测试消息字符串，用于匹配 ValueError
        message = "A `guess` sequence may not contain more than 3..."

        # 定义一个长度超过限制的猜测列表，检查是否会引发 ValueError 并匹配消息
        guess = [1, 2, 3, 4]
        with pytest.raises(ValueError, match=message):
            stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)

        # 设置另一个测试消息字符串，用于匹配 RuntimeWarning
        message = "Guess for parameter `n` rounded.*|Guess for parameter `p` clipped..."

        # 定义一个包含四舍五入和剪切的猜测字典，检查是否会触发 RuntimeWarning 并匹配消息
        guess = {'n': 4.5, 'p': -0.5}
        with pytest.warns(RuntimeWarning, match=message):
            stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)

        # 设置另一个测试消息字符串，用于匹配 RuntimeWarning
        message = "Guess for parameter `loc` rounded..."

        # 定义一个包含四舍五入的猜测列表，检查是否会触发 RuntimeWarning 并匹配消息
        guess = [5, 0.5, 0.5]
        with pytest.warns(RuntimeWarning, match=message):
            stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)

        # 设置另一个测试消息字符串，用于匹配 RuntimeWarning
        message = "Guess for parameter `p` clipped..."

        # 定义一个包含剪切的猜测字典，检查是否会触发 RuntimeWarning 并匹配消息
        guess = {'n': 5, 'p': -0.5}
        with pytest.warns(RuntimeWarning, match=message):
            stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)

        # 设置另一个测试消息字符串，用于匹配 RuntimeWarning
        message = "Guess for parameter `loc` clipped..."

        # 定义一个包含剪切的猜测列表，检查是否会触发 RuntimeWarning 并匹配消息
        guess = [5, 0.5, 1]
        with pytest.warns(RuntimeWarning, match=message):
            stats.fit(self.dist, self.data, self.shape_bounds_d, guess=guess)
    # 定义一个用于基本拟合测试的方法，接受分布名称和方法作为参数
    def basic_fit_test(self, dist_name, method):

        # 设置样本数量
        N = 5000
        # 将连续和离散分布数据合并到字典中
        dist_data = dict(distcont + distdiscrete)
        # 创建随机数生成器对象，使用给定种子值
        rng = np.random.default_rng(self.seed)
        # 获取指定名称的概率分布函数
        dist = getattr(stats, dist_name)
        # 获取分布的形状参数
        shapes = np.array(dist_data[dist_name])
        # 初始化边界数组
        bounds = np.empty((len(shapes) + 2, 2), dtype=np.float64)
        # 设置形状参数的边界
        bounds[:-2, 0] = shapes / 10. ** np.sign(shapes)
        bounds[:-2, 1] = shapes * 10. ** np.sign(shapes)
        # 设置loc的边界
        bounds[-2] = (0, 10)
        # 设置scale的边界
        bounds[-1] = (1e-16, 10)
        # 从loc的边界中随机选择一个值
        loc = rng.uniform(*bounds[-2])
        # 从scale的边界中随机选择一个值
        scale = rng.uniform(*bounds[-1])
        # 创建参考参数列表
        ref = list(dist_data[dist_name]) + [loc, scale]

        # 如果分布有pmf属性，则调整参考参数和边界
        if getattr(dist, 'pmf', False):
            ref = ref[:-1]
            ref[-1] = np.floor(loc)
            # 生成符合指定分布的随机样本数据
            data = dist.rvs(*ref, size=N, random_state=rng)
            bounds = bounds[:-1]
        # 如果分布有pdf属性，则生成符合指定分布的随机样本数据
        if getattr(dist, 'pdf', False):
            data = dist.rvs(*ref, size=N, random_state=rng)

        # 忽略运行时警告，如溢出警告
        with npt.suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "overflow encountered")
            # 对指定分布进行参数拟合，返回拟合结果
            res = stats.fit(dist, data, bounds, method=method,
                            optimizer=self.opt)

        # 定义不同拟合方法的名称映射
        nlff_names = {'mle': 'nnlf', 'mse': '_penalized_nlpsf'}
        # 获取当前方法对应的名称
        nlff_name = nlff_names[method]
        # 断言拟合结果与预期值的差异小于或等于给定容差值
        assert_nlff_less_or_close(dist, data, res.params, ref, **self.tols,
                                  nlff_name=nlff_name)

    # 使用参数化测试框架，对每个dist_name执行基本拟合测试（最大似然估计）
    @pytest.mark.parametrize("dist_name", cases_test_fit_mle())
    def test_basic_fit_mle(self, dist_name):
        self.basic_fit_test(dist_name, "mle")

    # 使用参数化测试框架，对每个dist_name执行基本拟合测试（均方误差）
    @pytest.mark.parametrize("dist_name", cases_test_fit_mse())
    def test_basic_fit_mse(self, dist_name):
        self.basic_fit_test(dist_name, "mse")

    # 测试arcsine分布
    def test_arcsine(self):
        # 无法保证所有分布都能适应所有数据，因为具有任意边界条件
        # 这个分布恰好在某些情况下失败，尝试稍微不同的设置
        N = 1000
        # 创建随机数生成器对象，使用给定种子值
        rng = np.random.default_rng(self.seed)
        # 获取arcsine分布的概率分布对象
        dist = stats.arcsine
        # 设置形状参数
        shapes = (1., 2.)
        # 生成符合arcsine分布的随机样本数据
        data = dist.rvs(*shapes, size=N, random_state=rng)
        # 设置形状参数的边界条件
        shape_bounds = {'loc': (0.1, 10), 'scale': (0.1, 10)}
        # 对数据进行参数拟合，返回拟合结果
        res = stats.fit(dist, data, shape_bounds, optimizer=self.opt)
        # 断言拟合结果与预期值的差异小于或等于给定容差值
        assert_nlff_less_or_close(dist, data, res.params, shapes, **self.tols)

    # 使用参数化测试框架，对每种方法（mle、mse）执行基本拟合测试
    @pytest.mark.parametrize("method", ('mle', 'mse'))
    def test_argus(self, method):
        # 不保证所有分布都能适合所有数据，特定范围内的分布可能会失败
        # 尝试一些稍微不同的东西
        N = 1000
        rng = np.random.default_rng(self.seed)
        # 使用 Argus 分布
        dist = stats.argus
        shapes = (1., 2., 3.)
        # 生成 Argus 分布的随机变量数据
        data = dist.rvs(*shapes, size=N, random_state=rng)
        # 定义参数的边界
        shape_bounds = {'chi': (0.1, 10), 'loc': (0.1, 10), 'scale': (0.1, 10)}
        # 拟合分布到数据，使用指定的优化器和方法
        res = stats.fit(dist, data, shape_bounds, optimizer=self.opt, method=method)

        # 断言拟合结果符合指定的数值和形状容差
        assert_nlff_less_or_close(dist, data, res.params, shapes, **self.tols)

    def test_foldnorm(self):
        # 不保证所有分布都能适合所有数据，特定范围内的分布可能会失败
        # 尝试一些稍微不同的东西
        N = 1000
        rng = np.random.default_rng(self.seed)
        # 使用 Folded Normal 分布
        dist = stats.foldnorm
        shapes = (1.952125337355587, 2., 3.)
        # 生成 Folded Normal 分布的随机变量数据
        data = dist.rvs(*shapes, size=N, random_state=rng)
        # 定义参数的边界
        shape_bounds = {'c': (0.1, 10), 'loc': (0.1, 10), 'scale': (0.1, 10)}
        # 拟合分布到数据，使用指定的优化器
        res = stats.fit(dist, data, shape_bounds, optimizer=self.opt)

        # 断言拟合结果符合指定的数值和形状容差
        assert_nlff_less_or_close(dist, data, res.params, shapes, **self.tols)

    def test_truncpareto(self):
        # 不保证所有分布都能适合所有数据，特定范围内的分布可能会失败
        # 尝试一些稍微不同的东西
        N = 1000
        rng = np.random.default_rng(self.seed)
        # 使用 Truncated Pareto 分布
        dist = stats.truncpareto
        shapes = (1.8, 5.3, 2.3, 4.1)
        # 生成 Truncated Pareto 分布的随机变量数据
        data = dist.rvs(*shapes, size=N, random_state=rng)
        # 定义参数的边界
        shape_bounds = [(0.1, 10)]*4
        # 拟合分布到数据，使用指定的优化器
        res = stats.fit(dist, data, shape_bounds, optimizer=self.opt)

        # 断言拟合结果符合指定的数值和形状容差
        assert_nlff_less_or_close(dist, data, res.params, shapes, **self.tols)

    @pytest.mark.fail_slow(5)
    def test_truncweibull_min(self):
        # 不保证所有分布都能适合所有数据，特定范围内的分布可能会失败
        # 尝试一些稍微不同的东西
        N = 1000
        rng = np.random.default_rng(self.seed)
        # 使用 Truncated Weibull Minimum 分布
        dist = stats.truncweibull_min
        shapes = (2.5, 0.25, 1.75, 2., 3.)
        # 生成 Truncated Weibull Minimum 分布的随机变量数据
        data = dist.rvs(*shapes, size=N, random_state=rng)
        # 定义参数的边界
        shape_bounds = [(0.1, 10)]*5
        # 拟合分布到数据，使用指定的优化器
        res = stats.fit(dist, data, shape_bounds, optimizer=self.opt)

        # 断言拟合结果符合指定的数值和形状容差
        assert_nlff_less_or_close(dist, data, res.params, shapes, **self.tols)
    def test_missing_shape_bounds(self):
        # 某些分布在某个参数上有一个小的定义域，例如二项分布中的 $p \in [0, 1]$
        # 用户不需要提供这些范围，因为用户的范围（无）与分布的定义域的交集是有限的
        N = 1000
        rng = np.random.default_rng(self.seed)

        # 使用二项分布初始化 dist 变量
        dist = stats.binom
        n, p, loc = 10, 0.65, 0
        # 生成二项分布的随机数据
        data = dist.rvs(n, p, loc=loc, size=N, random_state=rng)
        # 设置形状边界为 {'n': [0, 20]}，检查数组也是可以的
        shape_bounds = {'n': np.array([0, 20])}
        # 对数据进行拟合并返回拟合结果
        res = stats.fit(dist, data, shape_bounds, optimizer=self.opt)
        # 断言拟合参数接近于 (n, p, loc)，使用预设的容差
        assert_allclose(res.params, (n, p, loc), **self.tols)

        # 使用伯努利分布初始化 dist 变量
        dist = stats.bernoulli
        p, loc = 0.314159, 0
        # 生成伯努利分布的随机数据
        data = dist.rvs(p, loc=loc, size=N, random_state=rng)
        # 对数据进行拟合并返回拟合结果
        res = stats.fit(dist, data, optimizer=self.opt)
        # 断言拟合参数接近于 (p, loc)，使用预设的容差
        assert_allclose(res.params, (p, loc), **self.tols)

    def test_fit_only_loc_scale(self):
        # 仅拟合 loc 参数
        N = 5000
        rng = np.random.default_rng(self.seed)

        # 使用正态分布初始化 dist 变量
        dist = stats.norm
        loc, scale = 1.5, 1
        # 生成正态分布的随机数据
        data = dist.rvs(loc=loc, size=N, random_state=rng)
        # 设置 loc 参数的边界为 (0, 5)
        loc_bounds = (0, 5)
        bounds = {'loc': loc_bounds}
        # 对数据进行拟合并返回拟合结果
        res = stats.fit(dist, data, bounds, optimizer=self.opt)
        # 断言拟合参数接近于 (loc, scale)，使用预设的容差
        assert_allclose(res.params, (loc, scale), **self.tols)

        # 仅拟合 scale 参数
        loc, scale = 0, 2.5
        # 生成正态分布的随机数据
        data = dist.rvs(scale=scale, size=N, random_state=rng)
        # 设置 scale 参数的边界为 (0.01, 5)
        scale_bounds = (0.01, 5)
        bounds = {'scale': scale_bounds}
        # 对数据进行拟合并返回拟合结果
        res = stats.fit(dist, data, bounds, optimizer=self.opt)
        # 断言拟合参数接近于 (loc, scale)，使用预设的容差
        assert_allclose(res.params, (loc, scale), **self.tols)

        # 同时拟合 loc 和 scale 参数
        dist = stats.norm
        loc, scale = 1.5, 2.5
        # 生成正态分布的随机数据
        data = dist.rvs(loc=loc, scale=scale, size=N, random_state=rng)
        # 设置 loc 和 scale 参数的边界
        bounds = {'loc': loc_bounds, 'scale': scale_bounds}
        # 对数据进行拟合并返回拟合结果
        res = stats.fit(dist, data, bounds, optimizer=self.opt)
        # 断言拟合参数接近于 (loc, scale)，使用预设的容差
        assert_allclose(res.params, (loc, scale), **self.tols)

    def test_everything_fixed(self):
        N = 5000
        rng = np.random.default_rng(self.seed)

        # 使用正态分布初始化 dist 变量
        dist = stats.norm
        loc, scale = 1.5, 2.5
        # 生成正态分布的随机数据
        data = dist.rvs(loc=loc, scale=scale, size=N, random_state=rng)

        # 默认情况下 loc 和 scale 被固定为 0 和 1
        res = stats.fit(dist, data)
        # 断言拟合参数接近于 (0, 1)，使用预设的容差
        assert_allclose(res.params, (0, 1), **self.tols)

        # 显式地固定 loc 和 scale 参数
        bounds = {'loc': (loc, loc), 'scale': (scale, scale)}
        # 对数据进行拟合并返回拟合结果
        res = stats.fit(dist, data, bounds)
        # 断言拟合参数接近于 (loc, scale)，使用预设的容差
        assert_allclose(res.params, (loc, scale), **self.tols)

        # 在整理过程中 n 被固定
        dist = stats.binom
        n, p, loc = 10, 0.65, 0
        # 生成二项分布的随机数据
        data = dist.rvs(n, p, loc=loc, size=N, random_state=rng)
        # 设置形状边界为 {'n': (0, 20), 'p': (0.65, 0.65)}
        shape_bounds = {'n': (0, 20), 'p': (0.65, 0.65)}
        # 对数据进行拟合并返回拟合结果
        res = stats.fit(dist, data, shape_bounds, optimizer=self.opt)
        # 断言拟合参数接近于 (n, p, loc)，使用预设的容差
        assert_allclose(res.params, (n, p, loc), **self.tols)
    def test_failure(self):
        # 设定数据量 N 为 5000
        N = 5000
        # 使用给定的种子值初始化随机数生成器 rng
        rng = np.random.default_rng(self.seed)

        # 设定分布为负二项分布，形状参数为 (5, 0.5)
        dist = stats.nbinom
        shapes = (5, 0.5)
        # 生成符合指定分布的随机数据，大小为 N，使用给定的随机数生成器 rng
        data = dist.rvs(*shapes, size=N, random_state=rng)

        # 断言数据的最小值为 0
        assert data.min() == 0
        # 在位置下限为 0.5的情况下，似然为零
        # 设定参数边界
        bounds = [(0, 30), (0, 1), (0.5, 10)]
        # 使用指定边界拟合数据并返回结果
        res = stats.fit(dist, data, bounds)
        # 设定消息字符串
        message = "Optimization converged to parameter values that are"
        # 断言结果消息以给定的消息字符串开头
        assert res.message.startswith(message)
        # 断言拟合未成功
        assert res.success is False

    @pytest.mark.xslow
    def test_guess(self):
        # 测试猜测是否帮助 DE 找到期望的解决方案
        # 设定数据量 N 为 2000
        N = 2000
        # 使用指定种子初始化随机数生成器 rng
        rng = np.random.default_rng(196390444561)
        # 设定分布为超几何分布
        dist = stats.nhypergeom
        params = (20, 7, 12, 0)
        # 设定参数边界
        bounds = [(2, 200), (0.7, 70), (1.2, 120), (0, 10)]

        # 生成符合指定分布的随机数据，大小为 N，使用给定的随机数生成器 rng
        data = dist.rvs(*params, size=N, random_state=rng)

        # 使用指定边界和优化器进行数据拟合并返回结果
        res = stats.fit(dist, data, bounds, optimizer=self.opt)
        # 断言结果参数不完全接近给定的参数，使用指定的容差
        assert not np.allclose(res.params, params, **self.tols)

        # 使用指定的猜测参数进行数据拟合并返回结果
        res = stats.fit(dist, data, bounds, guess=params, optimizer=self.opt)
        # 断言结果参数完全接近给定的参数，使用指定的容差
        assert_allclose(res.params, params, **self.tols)

    def test_mse_accuracy_1(self):
        # 测试最大间距估计与维基百科示例的比较
        data = [2, 4]
        # 设定分布为指数分布
        dist = stats.expon
        # 设定参数边界
        bounds = {'loc': (0, 0), 'scale': (1e-8, 10)}
        # 使用最大似然估计法拟合数据并返回结果
        res_mle = stats.fit(dist, data, bounds=bounds, method='mle')
        # 断言结果参数 scale 接近 3，使用指定的容差
        assert_allclose(res_mle.params.scale, 3, atol=1e-3)
        # 使用均方误差估计法拟合数据并返回结果
        res_mse = stats.fit(dist, data, bounds=bounds, method='mse')
        # 断言结果参数 scale 接近 3.915，使用指定的容差
        assert_allclose(res_mse.params.scale, 3.915, atol=1e-3)

    def test_mse_accuracy_2(self):
        # 测试最大间距估计与维基百科示例的比较
        # 使用指定种子初始化随机数生成器 rng
        rng = np.random.default_rng(9843212616816518964)

        # 设定分布为均匀分布
        dist = stats.uniform
        n = 10
        # 生成符合指定参数的随机数据，大小为 n，使用给定的随机数生成器 rng
        data = dist(3, 6).rvs(size=n, random_state=rng)
        # 设定参数边界
        bounds = {'loc': (0, 10), 'scale': (1e-8, 10)}
        # 使用均方误差估计法拟合数据并返回结果
        res = stats.fit(dist, data, bounds=bounds, method='mse')
        # 设定参考值
        x = np.sort(data)
        a = (n*x[0] - x[-1])/(n - 1)
        b = (n*x[-1] - x[0])/(n - 1)
        ref = a, b-a  # (3.6081133632151503, 5.509328130317254)
        # 断言结果参数接近给定的参考值，使用指定的相对容差
        assert_allclose(res.params, ref, rtol=1e-4)
# 定义一个名为 examgrades 的列表，包含了一系列考试成绩数据
examgrades = [65, 61, 81, 88, 69, 89, 55, 84, 86, 84, 71, 81, 84, 81, 78, 67,
              96, 66, 73, 75, 59, 71, 69, 63, 79, 76, 63, 85, 87, 88, 80, 71,
              65, 84, 71, 75, 81, 79, 64, 65, 84, 77, 70, 75, 84, 75, 73, 92,
              90, 79, 80, 71, 73, 71, 58, 79, 73, 64, 77, 82, 81, 59, 54, 82,
              57, 79, 79, 73, 74, 82, 63, 64, 73, 69, 87, 68, 81, 73, 83, 73,
              80, 73, 73, 71, 66, 78, 64, 74, 68, 67, 75, 75, 80, 85, 74, 76,
              80, 77, 93, 70, 86, 80, 81, 83, 68, 60, 85, 64, 74, 82, 81, 77,
              66, 85, 75, 81, 69, 60, 83, 72]

# 定义一个名为 TestGoodnessOfFit 的类，用于测试拟合优度检验
class TestGoodnessOfFit:

    # 定义一个测试方法 test_gof_iv，测试在不合法输入时是否抛出预期异常
    def test_gof_iv(self):
        # 将 dist 设置为 stats.norm，表示使用正态分布进行测试
        dist = stats.norm
        # 设置 x 为一个列表 [1, 2, 3]
        x = [1, 2, 3]

        # 定义一个异常消息，用于验证异常抛出时的匹配检查
        message = r"`dist` must be a \(non-frozen\) instance of..."
        with pytest.raises(TypeError, match=message):
            goodness_of_fit(stats.norm(), x)

        # 设置另一个异常消息，验证在不合法输入时是否正确抛出异常
        message = "`data` must be a one-dimensional array of numbers."
        with pytest.raises(ValueError, match=message):
            goodness_of_fit(dist, [[1, 2, 3]])

        # 设置第三个异常消息，用于验证异常抛出时的匹配检查
        message = "`statistic` must be one of..."
        with pytest.raises(ValueError, match=message):
            goodness_of_fit(dist, x, statistic='mm')

        # 设置第四个异常消息，验证在不合法输入时是否正确抛出异常
        message = "`n_mc_samples` must be an integer."
        with pytest.raises(TypeError, match=message):
            goodness_of_fit(dist, x, n_mc_samples=1000.5)

        # 设置第五个异常消息，用于验证异常抛出时的匹配检查
        message = "'herring' cannot be used to seed a"
        with pytest.raises(ValueError, match=message):
            goodness_of_fit(dist, x, random_state='herring')

    # 定义一个测试方法 test_against_ks，验证使用 Kolmogorov-Smirnov 检验方法的拟合结果
    def test_against_ks(self):
        # 使用指定的随机数生成器创建 rng 对象
        rng = np.random.default_rng(8517426291317196949)
        # 设置 x 为前面定义的 examgrades 列表
        x = examgrades
        # 计算 x 的均值和标准差（未知参数）
        known_params = {'loc': np.mean(x), 'scale': np.std(x, ddof=1)}
        # 进行拟合优度检验，使用 Kolmogorov-Smirnov 检验，指定随机数生成器 rng
        res = goodness_of_fit(stats.norm, x, known_params=known_params,
                              statistic='ks', random_state=rng)
        # 计算参考值，使用 Kolmogorov-Smirnov 检验的理论分布
        ref = stats.kstest(x, stats.norm(**known_params).cdf, method='exact')
        # 断言拟合统计量与参考值的接近程度，预期约为 0.0848
        assert_allclose(res.statistic, ref.statistic)  # ~0.0848
        # 断言 p 值与参考值的接近程度，允许的误差为 5e-3，预期约为 0.335
        assert_allclose(res.pvalue, ref.pvalue, atol=5e-3)  # ~0.335

    # 定义一个测试方法 test_against_lilliefors，验证使用 Lilliefors 检验方法的拟合结果
    def test_against_lilliefors(self):
        # 使用指定的随机数生成器创建 rng 对象
        rng = np.random.default_rng(2291803665717442724)
        # 设置 x 为前面定义的 examgrades 列表
        x = examgrades
        # 进行拟合优度检验，使用 Lilliefors 检验，指定随机数生成器 rng
        res = goodness_of_fit(stats.norm, x, statistic='ks', random_state=rng)
        # 计算 x 的均值和标准差（未知参数）
        known_params = {'loc': np.mean(x), 'scale': np.std(x, ddof=1)}
        # 计算参考值，使用 Kolmogorov-Smirnov 检验的理论分布
        ref = stats.kstest(x, stats.norm(**known_params).cdf, method='exact')
        # 断言拟合统计量与参考值的接近程度，预期约为 0.0848
        assert_allclose(res.statistic, ref.statistic)  # ~0.0848
        # 断言 p 值与预期值的接近程度，允许的误差为 5e-3，预期约为 0.0348
        assert_allclose(res.pvalue, 0.0348, atol=5e-3)
    # 测试用例：使用 Cramér-von Mises 统计量检验正态分布拟合效果
    def test_against_cvm(self):
        # 创建随机数生成器对象，并指定种子
        rng = np.random.default_rng(8674330857509546614)
        # 使用 examgrades 数据进行测试
        x = examgrades
        # 已知的参数：均值为 x 的均值，标准差为 x 的样本标准差
        known_params = {'loc': np.mean(x), 'scale': np.std(x, ddof=1)}
        # 调用 goodness_of_fit 函数，使用 Cramér-von Mises 统计量，传入已知参数和随机数种子
        res = goodness_of_fit(stats.norm, x, known_params=known_params,
                              statistic='cvm', random_state=rng)
        # 使用 stats.cramervonmises 计算参考值
        ref = stats.cramervonmises(x, stats.norm(**known_params).cdf)
        # 断言检验统计量的近似值是否接近参考值，期望误差在 0.090 左右
        assert_allclose(res.statistic, ref.statistic)  # ~0.090
        # 断言 p 值的近似值是否接近参考值，期望误差在 5e-3 左右，期望值约为 0.636
        assert_allclose(res.pvalue, ref.pvalue, atol=5e-3)  # ~0.636

    # 测试用例：使用 Anderson-Darling 统计量检验正态分布拟合效果（案例 0）
    def test_against_anderson_case_0(self):
        # 创建随机数生成器对象，并指定种子
        rng = np.random.default_rng(7384539336846690410)
        # 使用 1 到 100 的数组作为测试数据
        x = np.arange(1, 101)
        # 已知的参数：均值为 45.01575354024957，标准差为 30
        known_params = {'loc': 45.01575354024957, 'scale': 30}
        # 调用 goodness_of_fit 函数，使用 Anderson-Darling 统计量，传入已知参数和随机数种子
        res = goodness_of_fit(stats.norm, x, known_params=known_params,
                              statistic='ad', random_state=rng)
        # 断言检验统计量的近似值是否接近期望值 2.492
        assert_allclose(res.statistic, 2.492)  # See [1] Table 1A 1.0
        # 断言 p 值的近似值是否接近期望值 0.05，期望误差在 5e-3 左右
        assert_allclose(res.pvalue, 0.05, atol=5e-3)

    # 测试用例：使用 Anderson-Darling 统计量检验正态分布拟合效果（案例 1）
    def test_against_anderson_case_1(self):
        # 创建随机数生成器对象，并指定种子
        rng = np.random.default_rng(5040212485680146248)
        # 使用 1 到 100 的数组作为测试数据
        x = np.arange(1, 101)
        # 已知的参数：标准差为 29.957112639101933，均值将通过 root_scalar 找到
        known_params = {'scale': 29.957112639101933}
        # 调用 goodness_of_fit 函数，使用 Anderson-Darling 统计量，传入已知参数和随机数种子
        res = goodness_of_fit(stats.norm, x, known_params=known_params,
                              statistic='ad', random_state=rng)
        # 断言检验统计量的近似值是否接近期望值 0.908
        assert_allclose(res.statistic, 0.908)  # See [1] Table 1B 1.1
        # 断言 p 值的近似值是否接近期望值 0.1，期望误差在 5e-3 左右
        assert_allclose(res.pvalue, 0.1, atol=5e-3)

    # 测试用例：使用 Anderson-Darling 统计量检验正态分布拟合效果（案例 2）
    def test_against_anderson_case_2(self):
        # 创建随机数生成器对象，并指定种子
        rng = np.random.default_rng(726693985720914083)
        # 使用 1 到 100 的数组作为测试数据
        x = np.arange(1, 101)
        # 已知的参数：均值为 44.5680212261933，标准差将通过 root_scalar 找到
        known_params = {'loc': 44.5680212261933}
        # 调用 goodness_of_fit 函数，使用 Anderson-Darling 统计量，传入已知参数和随机数种子
        res = goodness_of_fit(stats.norm, x, known_params=known_params,
                              statistic='ad', random_state=rng)
        # 断言检验统计量的近似值是否接近期望值 2.904
        assert_allclose(res.statistic, 2.904)  # See [1] Table 1B 1.2
        # 断言 p 值的近似值是否接近期望值 0.025，期望误差在 5e-3 左右
        assert_allclose(res.pvalue, 0.025, atol=5e-3)

    # 测试用例：使用 Anderson-Darling 统计量检验斜态正态分布拟合效果（案例 3）
    def test_against_anderson_case_3(self):
        # 创建随机数生成器对象，并指定种子
        rng = np.random.default_rng(6763691329830218206)
        # 生成斜态正态分布的随机样本，均值为 1，标准差为 2，斜度为 1.4477847789132101
        x = stats.skewnorm.rvs(1.4477847789132101, loc=1, scale=2, size=100,
                               random_state=rng)
        # 调用 goodness_of_fit 函数，使用 Anderson-Darling 统计量，传入斜态正态分布的样本和随机数种子
        res = goodness_of_fit(stats.norm, x, statistic='ad', random_state=rng)
        # 断言检验统计量的近似值是否接近期望值 0.559
        assert_allclose(res.statistic, 0.559)  # See [1] Table 1B 1.2
        # 断言 p 值的近似值是否接近期望值 0.15，期望误差在 5e-3 左右
        assert_allclose(res.pvalue, 0.15, atol=5e-3)
    def test_against_anderson_gumbel_r(self):
        # 使用特定的随机数生成器创建一个 RNG 对象
        rng = np.random.default_rng(7302761058217743)
        # 生成 Gumbel 分布的随机样本数据
        x = stats.genextreme(0.051896837188595134, loc=0.5,
                             scale=1.5).rvs(size=1000, random_state=rng)
        # 对生成的样本数据进行拟合检验，使用 Anderson-Darling 统计量
        res = goodness_of_fit(stats.gumbel_r, x, statistic='ad',
                              random_state=rng)
        # 使用 Anderson-Darling 检验作为参考进行比较
        ref = stats.anderson(x, dist='gumbel_r')
        # 断言拟合统计量接近参考值的临界值
        assert_allclose(res.statistic, ref.critical_values[0])
        # 断言 p 值接近参考值的显著性水平
        assert_allclose(res.pvalue, ref.significance_level[0]/100, atol=5e-3)

    def test_against_filliben_norm(self):
        # 对正态分布进行拟合检验，参考 `stats.fit` 中的文献 [7] 第 8 节 "Example"
        rng = np.random.default_rng(8024266430745011915)
        y = [6, 1, -4, 8, -2, 5, 0]
        known_params = {'loc': 0, 'scale': 1}
        # 使用 Filliben 统计量对给定数据进行拟合检验
        res = stats.goodness_of_fit(stats.norm, y, known_params=known_params,
                                    statistic="filliben", random_state=rng)
        # 由于 Filliben 计算中的舍入误差，略微的差异可以被接受
        assert_allclose(res.statistic, 0.98538, atol=1e-4)
        # 断言 p 值在特定范围内
        assert 0.75 < res.pvalue < 0.9

        # 使用 R 的 ppcc 库进行比较:
        # library(ppcc)
        # options(digits=16)
        # x <- c(6, 1, -4, 8, -2, 5, 0)
        # set.seed(100)
        # ppccTest(x, "qnorm", ppos="Filliben")
        # 使用 R 的结果进行比较
        assert_allclose(res.statistic, 0.98540957187084, rtol=2e-5)
        assert_allclose(res.pvalue, 0.8875, rtol=2e-3)

    def test_filliben_property(self):
        # Filliben 统计量应该与数据的位置和尺度无关
        rng = np.random.default_rng(8535677809395478813)
        x = rng.normal(loc=10, scale=0.5, size=100)
        # 对正态分布进行 Filliben 统计量的拟合检验
        res = stats.goodness_of_fit(stats.norm, x,
                                    statistic="filliben", random_state=rng)
        known_params = {'loc': 0, 'scale': 1}
        # 使用不同的参数对同一数据进行 Filliben 统计量的拟合检验
        ref = stats.goodness_of_fit(stats.norm, x, known_params=known_params,
                                    statistic="filliben", random_state=rng)
        # 断言两次拟合的统计量非常接近
        assert_allclose(res.statistic, ref.statistic, rtol=1e-15)

    @pytest.mark.parametrize('case', [(25, [.928, .937, .950, .958, .966]),
                                      (50, [.959, .965, .972, .977, .981]),
                                      (95, [.977, .979, .983, .986, .989])])
    # 定义一个测试方法，用于检验正态分布拟合结果与 Filliben 标准的一致性，参考文献 [7] 表格 1
    def test_against_filliben_norm_table(self, case):
        # 使用种子为 504569995557928957 的随机数生成器创建随机数生成器对象
        rng = np.random.default_rng(504569995557928957)
        # 从随机数生成器生成长度为 n 的随机数序列 x
        n, ref = case
        x = rng.random(n)
        # 指定正态分布的已知参数 loc=0, scale=1
        known_params = {'loc': 0, 'scale': 1}
        # 对 x 应用正态分布的拟合检验，使用 Filliben 统计量，指定随机数生成器 rng
        res = stats.goodness_of_fit(stats.norm, x, known_params=known_params,
                                    statistic="filliben", random_state=rng)
        # 定义百分位数数组
        percentiles = np.array([0.005, 0.01, 0.025, 0.05, 0.1])
        # 计算 res.null_distribution 的百分位数对应的分数
        res = stats.scoreatpercentile(res.null_distribution, percentiles*100)
        # 使用 assert_allclose 断言检验 res 是否接近于 ref，允许的绝对误差为 2e-3
        assert_allclose(res, ref, atol=2e-3)

    # 标记为 pytest 的慢速测试
    @pytest.mark.xslow
    # 参数化测试方法，对不同的 case 进行参数化测试
    @pytest.mark.parametrize('case', [(5, 0.95772790260469, 0.4755),
                                      (6, 0.95398832257958, 0.3848),
                                      (7, 0.9432692889277, 0.2328)])
    # 定义一个测试方法，用于检验 Rayleigh 分布拟合结果与 PPCC 套件中 Filliben 方法的一致性
    def test_against_ppcc(self, case):
        # 解包 case 中的参数 n, ref_statistic, ref_pvalue
        n, ref_statistic, ref_pvalue = case
        # 使用种子为 7777775561439803116 的随机数生成器创建随机数生成器对象
        rng = np.random.default_rng(7777775561439803116)
        # 从正态分布中生成 n 个随机数 x
        x = rng.normal(size=n)
        # 对 x 应用 Rayleigh 分布的拟合检验，使用 Filliben 统计量，指定随机数生成器 rng
        res = stats.goodness_of_fit(stats.rayleigh, x, statistic="filliben",
                                    random_state=rng)
        # 使用 assert_allclose 断言检验 res.statistic 是否接近于 ref_statistic，允许的相对误差为 1e-4
        assert_allclose(res.statistic, ref_statistic, rtol=1e-4)
        # 使用 assert_allclose 断言检验 res.pvalue 是否接近于 ref_pvalue，允许的绝对误差为 1.5e-2
        assert_allclose(res.pvalue, ref_pvalue, atol=1.5e-2)
    def test_params_effects(self):
        # 确保 `guessed_params`、`fit_params` 和 `known_params` 有预期的效果

        # 使用指定种子初始化随机数生成器
        rng = np.random.default_rng(9121950977643805391)
        
        # 从 skewnorm 分布生成 50 个随机数
        x = stats.skewnorm.rvs(-5.044559778383153, loc=1, scale=2, size=50,
                               random_state=rng)

        # 展示 `guessed_params` 不适用于猜测，但 `fit_params` 和 `known_params` 符合提供的拟合
        guessed_params = {'c': 13.4}
        fit_params = {'scale': 13.73}
        known_params = {'loc': -13.85}

        # 再次使用相同种子初始化随机数生成器
        rng = np.random.default_rng(9121950977643805391)

        # 进行拟合和拟合质量检验，返回拟合结果
        res1 = goodness_of_fit(stats.weibull_min, x, n_mc_samples=2,
                               guessed_params=guessed_params,
                               fit_params=fit_params,
                               known_params=known_params, random_state=rng)

        # 断言检查拟合结果的特定参数是否与预期值接近
        assert not np.allclose(res1.fit_result.params.c, 13.4)
        assert_equal(res1.fit_result.params.scale, 13.73)
        assert_equal(res1.fit_result.params.loc, -13.85)

        # 展示改变猜测会改变被拟合的参数，并且会改变空分布
        guessed_params = {'c': 2}
        rng = np.random.default_rng(9121950977643805391)

        # 再次进行拟合和拟合质量检验，返回拟合结果
        res2 = goodness_of_fit(stats.weibull_min, x, n_mc_samples=2,
                               guessed_params=guessed_params,
                               fit_params=fit_params,
                               known_params=known_params, random_state=rng)

        # 断言检查新的拟合结果与之前的拟合结果参数是否不接近
        assert not np.allclose(res2.fit_result.params.c,
                               res1.fit_result.params.c, rtol=1e-8)
        assert not np.allclose(res2.null_distribution,
                               res1.null_distribution, rtol=1e-8)
        assert_equal(res2.fit_result.params.scale, 13.73)
        assert_equal(res2.fit_result.params.loc, -13.85)

        # 如果将所有参数都设置为 `fit_params` 和 `known_params`，
        # 它们都会固定为这些值，但空分布会变化
        fit_params = {'c': 13.4, 'scale': 13.73}
        rng = np.random.default_rng(9121950977643805391)

        # 再次进行拟合和拟合质量检验，返回拟合结果
        res3 = goodness_of_fit(stats.weibull_min, x, n_mc_samples=2,
                               guessed_params=guessed_params,
                               fit_params=fit_params,
                               known_params=known_params, random_state=rng)

        # 断言检查新的拟合结果的特定参数是否等于预期值
        assert_equal(res3.fit_result.params.c, 13.4)
        assert_equal(res3.fit_result.params.scale, 13.73)
        assert_equal(res3.fit_result.params.loc, -13.85)
        assert not np.allclose(res3.null_distribution, res1.null_distribution)
    def test_custom_statistic(self):
        # 测试支持自定义统计函数的功能

        # 参考文献:
        # [1] Pyke, R. (1965).  "Spacings".  Journal of the Royal Statistical
        #     Society: Series B (Methodological), 27(3): 395-436.
        # [2] Burrows, P. M. (1979).  "Selected Percentage Points of
        #     Greenwood's Statistics".  Journal of the Royal Statistical
        #     Society. Series A (General), 142(2): 256-258.

        # 使用 Greenwood 统计量进行说明；参见 [1, p.402].
        def greenwood(dist, data, *, axis):
            # 对数据沿指定轴进行排序
            x = np.sort(data, axis=axis)
            # 计算累积分布函数
            y = dist.cdf(x)
            # 计算差分
            d = np.diff(y, axis=axis, prepend=0, append=1)
            # 计算平方和
            return np.sum(d ** 2, axis=axis)

        # 在完全指定的零分布上运行蒙特卡罗测试，样本大小为 5，并将模拟分位数与
        # [2, Table 1, column (n = 5)] 中给出的精确分位数进行比较。
        rng = np.random.default_rng(9121950977643805391)
        # 生成指数分布的随机变量
        data = stats.expon.rvs(size=5, random_state=rng)
        # 进行拟合优度检验，使用自定义的统计函数 greenwood
        result = goodness_of_fit(stats.expon, data,
                                 known_params={'loc': 0, 'scale': 1},
                                 statistic=greenwood, random_state=rng)
        # 设置期望的分位数
        p = [.01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99]
        exact_quantiles = [
            .183863, .199403, .210088, .226040, .239947, .253677, .268422,
            .285293, .306002, .334447, .382972, .432049, .547468]
        # 计算模拟的分位数
        simulated_quantiles = np.quantile(result.null_distribution, p)
        # 断言模拟的分位数与期望的分位数在给定的容差范围内接近
        assert_allclose(simulated_quantiles, exact_quantiles, atol=0.005)
class TestFitResult:
    # 定义测试类 TestFitResult
    def test_plot_iv(self):
        # 定义测试方法 test_plot_iv，self 参数表示类的实例
        rng = np.random.default_rng(1769658657308472721)
        # 创建随机数生成器 rng，指定种子以确保结果可重现

        data = stats.norm.rvs(0, 1, size=100, random_state=rng)
        # 生成服从标准正态分布的随机数据，size=100 表示生成 100 个数据点

        def optimizer(*args, **kwargs):
            # 定义内部函数 optimizer，用于优化参数设置
            return differential_evolution(*args, **kwargs, seed=rng)
            # 调用全局函数 differential_evolution 进行参数优化，设置随机种子为 rng

        bounds = [(0, 30), (0, 1)]
        # 设置参数范围 bounds，每个参数有一个最小值和最大值

        res = stats.fit(stats.norm, data, bounds, optimizer=optimizer)
        # 使用 stats 模块的 fit 函数拟合正态分布到数据，使用 optimizer 进行优化

        try:
            import matplotlib  # noqa: F401
            # 尝试导入 matplotlib 模块，确保其可用性
            message = r"`plot_type` must be one of \{'..."
            # 设置错误消息，指示 plot_type 参数必须是特定集合中的一个
            with pytest.raises(ValueError, match=message):
                # 使用 pytest 检查是否会引发 ValueError 异常，并匹配指定的错误消息
                res.plot(plot_type='llama')
                # 调用拟合结果对象的 plot 方法，传递不支持的 plot_type 参数
        except (ModuleNotFoundError, ImportError):
            # 捕获可能的导入错误异常
            message = r"matplotlib must be installed to use method `plot`."
            # 设置错误消息，指示需要安装 matplotlib 才能使用 plot 方法
            with pytest.raises(ModuleNotFoundError, match=message):
                # 使用 pytest 检查是否会引发 ModuleNotFoundError 异常，并匹配指定的错误消息
                res.plot(plot_type='llama')
                # 调用拟合结果对象的 plot 方法，传递不支持的 plot_type 参数
```