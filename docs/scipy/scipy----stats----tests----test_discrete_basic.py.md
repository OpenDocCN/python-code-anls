# `D:\src\scipysrc\scipy\scipy\stats\tests\test_discrete_basic.py`

```
import numpy.testing as npt  # 导入numpy测试模块，别名为npt
from numpy.testing import assert_allclose  # 导入numpy的assert_allclose函数，用于比较数组是否接近

import numpy as np  # 导入numpy库，别名为np
import pytest  # 导入pytest测试框架

from scipy import stats  # 导入scipy的stats模块
from .common_tests import (check_normalization, check_moment,  # 导入自定义测试函数
                           check_mean_expect, check_var_expect,
                           check_skew_expect, check_kurt_expect,
                           check_entropy, check_private_entropy,
                           check_edge_support, check_named_args,
                           check_random_state_property, check_pickling,
                           check_rvs_broadcast, check_freezing,)
from scipy.stats._distr_params import distdiscrete, invdistdiscrete  # 导入离散分布参数相关模块
from scipy.stats._distn_infrastructure import rv_discrete_frozen  # 导入离散冻结分布模块

vals = ([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])  # 定义vals变量，包含两个列表作为离散分布的可能取值

distdiscrete += [[stats.rv_discrete(values=vals), ()]]  # 将包含离散分布和空元组的列表添加到distdiscrete中

# 对于以下分布，只在测试模式full下运行test_discrete_basic
distslow = {'zipfian', 'nhypergeom'}

# 覆盖`check_cdf_ppf`的ULPs调整数
roundtrip_cdf_ppf_exceptions = {'nbinom': 30}

def cases_test_discrete_basic():
    seen = set()  # 创建空集合seen，用于存储已经遍历的分布名称
    for distname, arg in distdiscrete:  # 遍历distdiscrete中的分布名和参数对
        if distname in distslow:  # 如果分布名在distslow中
            yield pytest.param(distname, arg, distname, marks=pytest.mark.slow)  # 生成一个带有标记的pytest参数化测试
        else:
            yield distname, arg, distname not in seen  # 否则生成普通的测试参数
        seen.add(distname)  # 将当前分布名加入seen集合中


@pytest.mark.parametrize('distname,arg,first_case', cases_test_discrete_basic())
def test_discrete_basic(distname, arg, first_case):
    try:
        distfn = getattr(stats, distname)  # 尝试获取stats模块中对应名称的分布函数
    except TypeError:
        distfn = distname  # 如果出现类型错误，将distname作为分布函数
        distname = 'sample distribution'  # 将分布名设为'sample distribution'
    np.random.seed(9765456)  # 设置随机种子为9765456
    rvs = distfn.rvs(size=2000, *arg)  # 生成指定分布的随机变量
    supp = np.unique(rvs)  # 获取随机变量的唯一值作为支持集合
    m, v = distfn.stats(*arg)  # 计算分布的统计信息（均值和方差）
    check_cdf_ppf(distfn, arg, supp, distname + ' cdf_ppf')  # 调用检查CDF和PPF的函数

    check_pmf_cdf(distfn, arg, distname)  # 检查PMF和CDF的函数
    check_oth(distfn, arg, supp, distname + ' oth')  # 检查其他功能的函数
    check_edge_support(distfn, arg)  # 检查边界支持的函数

    alpha = 0.01  # 显著性水平设为0.01
    check_discrete_chisquare(distfn, arg, rvs, alpha, distname + ' chisquare')  # 检查离散分布的卡方检验

    if first_case:  # 如果是第一个测试案例
        locscale_defaults = (0,)  # 默认位置和尺度设定为0
        meths = [distfn.pmf, distfn.logpmf, distfn.cdf, distfn.logcdf,  # 定义需要检查的方法列表
                 distfn.logsf]
        spec_k = {'randint': 11, 'hypergeom': 4, 'bernoulli': 0,  # 指定特定分布的参数k值
                  'nchypergeom_wallenius': 6}
        k = spec_k.get(distname, 1)  # 获取分布名称对应的k值，如果不存在则默认为1
        check_named_args(distfn, k, arg, locscale_defaults, meths)  # 检查命名参数的函数
        if distname != 'sample distribution':  # 如果不是样本分布
            check_scale_docstring(distfn)  # 检查尺度文档字符串的函数
        check_random_state_property(distfn, arg)  # 检查随机状态属性的函数
        check_pickling(distfn, arg)  # 检查序列化的函数
        check_freezing(distfn, arg)  # 检查冻结的函数

        # 熵
        check_entropy(distfn, arg, distname)  # 检查熵的函数
        if distfn.__class__._entropy != stats.rv_discrete._entropy:  # 如果定义的熵不等于默认的离散随机变量的熵
            check_private_entropy(distfn, arg, stats.rv_discrete)  # 检查私有熵的函数
@pytest.mark.parametrize('distname,arg', distdiscrete)
def test_moments(distname, arg):
    try:
        # 尝试获取名为distname的概率分布函数
        distfn = getattr(stats, distname)
    except TypeError:
        # 如果失败，将distname作为函数本身，并更改distname为'sample distribution'
        distfn = distname
        distname = 'sample distribution'
    # 获取概率分布函数的统计量 m, v, s, k
    m, v, s, k = distfn.stats(*arg, moments='mvsk')
    # 检查概率分布函数的归一化
    check_normalization(distfn, arg, distname)

    # 比较 `stats` 方法和 `moment` 方法
    check_moment(distfn, arg, m, v, distname)
    # 检查均值期望
    check_mean_expect(distfn, arg, m, distname)
    # 检查方差期望
    check_var_expect(distfn, arg, m, v, distname)
    # 检查偏度期望
    check_skew_expect(distfn, arg, m, v, s, distname)
    with np.testing.suppress_warnings() as sup:
        # 如果概率分布函数是'zipf'或'betanbinom'，忽略运行时警告
        if distname in ['zipf', 'betanbinom']:
            sup.filter(RuntimeWarning)
        # 检查峰度期望
        check_kurt_expect(distfn, arg, m, v, k, distname)

    # 冻结分布的矩
    check_moment_frozen(distfn, arg, m, 1)
    check_moment_frozen(distfn, arg, v+m*m, 2)


@pytest.mark.parametrize('dist,shape_args', distdiscrete)
def test_rvs_broadcast(dist, shape_args):
    # 如果shape_only为True，表示分布的_rvs方法使用多个随机数生成随机变量
    # 这意味着使用广播或非平凡大小的rvs的结果不一定与使用numpy.vectorize版本的rvs()相同
    # 因此，我们只能比较结果的形状，而不能比较值。
    # 下列分布的存在与否是分布实现的细节，不是要求。如果分布的rvs()方法的实现发生变化，可能需要修改此测试。
    shape_only = dist in ['betabinom', 'betanbinom', 'skellam', 'yulesimon',
                          'dlaplace', 'nchypergeom_fisher',
                          'nchypergeom_wallenius']

    try:
        # 尝试获取名为dist的概率分布函数
        distfunc = getattr(stats, dist)
    except TypeError:
        # 如果失败，将dist作为函数本身，并使用字符串表示rv_discrete的分布情况
        distfunc = dist
        dist = f'rv_discrete(values=({dist.xk!r}, {dist.pk!r}))'
    # 创建一个长度为2的零数组
    loc = np.zeros(2)
    # 获取概率分布函数的参数个数
    nargs = distfunc.numargs
    allargs = []
    bshape = []
    # 生成形状参数的参数值...
    for k in range(nargs):
        shp = (k + 3,) + (1,)*(k + 1)
        param_val = shape_args[k]
        allargs.append(np.full(shp, param_val))
        bshape.insert(0, shp[0])
    allargs.append(loc)
    bshape.append(loc.size)
    # bshape保存当loc、scale和形状参数一起广播时的预期形状。
    check_rvs_broadcast(
        distfunc, dist, allargs, bshape, shape_only, [np.dtype(int)]
    )


@pytest.mark.parametrize('dist,args', distdiscrete)
def test_ppf_with_loc(dist, args):
    try:
        # 尝试获取名为dist的概率分布函数
        distfn = getattr(stats, dist)
    except TypeError:
        # 如果失败，将dist作为函数本身
        distfn = dist
    # 使用种子1942349设置随机数生成器的种子
    np.random.seed(1942349)
    # 定义负、零和正重定位数组
    re_locs = [np.random.randint(-10, -1), 0, np.random.randint(1, 10)]
    _a, _b = distfn.support(*args)
    # 对于每个 loc 在 re_locs 列表中的值进行迭代
    for loc in re_locs:
        # 使用 numpy.testing.assert_array_equal 函数来断言两个数组是否相等
        npt.assert_array_equal(
            # 构造第一个数组，元素为 [_a-1+loc, _b+loc]
            [_a-1+loc, _b+loc],
            # 构造第二个数组，元素为 [distfn.ppf(0.0, *args, loc=loc), distfn.ppf(1.0, *args, loc=loc)]
            [distfn.ppf(0.0, *args, loc=loc), distfn.ppf(1.0, *args, loc=loc)]
        )
@pytest.mark.parametrize('dist, args', distdiscrete)
# 使用 distdiscrete 中的每个参数对分布函数进行参数化测试
def test_isf_with_loc(dist, args):
    try:
        distfn = getattr(stats, dist)
    except TypeError:
        distfn = dist
    # 尝试获取 stats 模块中名为 dist 的函数或分布对象，如果失败则直接使用 dist
    np.random.seed(1942349)
    # 设置随机数种子为 1942349
    re_locs = [np.random.randint(-10, -1), 0, np.random.randint(1, 10)]
    # 生成包含负数、零和正数的随机整数列表作为 loc 参数的测试值
    _a, _b = distfn.support(*args)
    # 调用 distfn 的 support 方法获取支持范围并分配给 _a 和 _b
    for loc in re_locs:
        expected = _b + loc, _a - 1 + loc
        # 计算预期的结果作为边界的修正值
        res = distfn.isf(0., *args, loc=loc), distfn.isf(1., *args, loc=loc)
        # 调用 distfn 的 isf 方法计算逆累积分布函数（Inverse Survival Function）
        # 以验证预期结果与实际结果是否相等
        npt.assert_array_equal(expected, res)
        # 使用 NumPy 测试工具 npt 检查预期和实际结果数组是否完全相等
    # 测试广播行为
    re_locs = [np.random.randint(-10, -1, size=(5, 3)),
               np.zeros((5, 3)),
               np.random.randint(1, 10, size=(5, 3))]
    # 生成包含不同形状的随机整数数组作为 loc 参数的测试值
    _a, _b = distfn.support(*args)
    # 再次获取支持范围并分配给 _a 和 _b
    for loc in re_locs:
        expected = _b + loc, _a - 1 + loc
        # 计算预期的结果作为边界的修正值
        res = distfn.isf(0., *args, loc=loc), distfn.isf(1., *args, loc=loc)
        # 调用 distfn 的 isf 方法计算逆累积分布函数，验证预期结果与实际结果是否相等
        npt.assert_array_equal(expected, res)
        # 使用 NumPy 测试工具 npt 检查预期和实际结果数组是否完全相等


def check_cdf_ppf(distfn, arg, supp, msg):
    # supp 假设为 distfn 支持的整数数组，但不一定包含所有支持的整数
    # 此测试假设分布函数在其支持范围内的任何值的 PMF 都大于 1e-8

    # 计算支持数组 supp 的累积分布函数值
    cdf_supp = distfn.cdf(supp, *arg)
    # 在极少数情况下，由于有限精度计算，ppf(cdf(supp)) 可能会导致一个元素偏离一位
    # 我们通过减少几个 ULP 来避免这种情况
    n_ulps = roundtrip_cdf_ppf_exceptions.get(distfn.name, 15)
    cdf_supp0 = cdf_supp - n_ulps*np.spacing(cdf_supp)
    # 使用 NumPy 测试工具 npt 检查 distfn 的 ppf 方法的计算结果与 supp 是否完全相等
    npt.assert_array_equal(distfn.ppf(cdf_supp0, *arg),
                           supp, msg + '-roundtrip')
    # 再次使用 distfn 的 ppf 方法，但这次将 cdf 值减少 1e-8
    npt.assert_array_equal(distfn.ppf(distfn.cdf(supp, *arg) - 1e-8, *arg),
                           supp, msg + '-roundtrip')

    if not hasattr(distfn, 'xk'):
        _a, _b = distfn.support(*arg)
        # 获取支持范围并分配给 _a 和 _b
        supp1 = supp[supp < _b]
        # 选择小于 _b 的支持数组元素
        npt.assert_array_equal(distfn.ppf(distfn.cdf(supp1, *arg) + 1e-8, *arg),
                               supp1 + distfn.inc, msg + ' ppf-cdf-next')


def check_pmf_cdf(distfn, arg, distname):
    if hasattr(distfn, 'xk'):
        index = distfn.xk
    else:
        startind = int(distfn.ppf(0.01, *arg) - 1)
        index = list(range(startind, startind + 10))
    # 计算 distfn 在 index 处的累积分布函数值和概率质量函数值的累计和
    cdfs = distfn.cdf(index, *arg)
    pmfs_cum = distfn.pmf(index, *arg).cumsum()

    atol, rtol = 1e-10, 1e-10
    if distname == 'skellam':    # 对 ncx2 进行精度调整
        atol, rtol = 1e-5, 1e-5
    # 使用 NumPy 测试工具 npt 检查累积分布函数值与概率质量函数值的累计和是否在给定的误差范围内
    npt.assert_allclose(cdfs - cdfs[0], pmfs_cum - pmfs_cum[0],
                        atol=atol, rtol=rtol)

    # 还要检查非整数 k 处的 pmf 是否为零
    k = np.asarray(index)
    k_shifted = k[:-1] + np.diff(k)/2
    # 使用 NumPy 测试工具 npt 检查 distfn 在 k_shifted 处的 pmf 值是否为零
    npt.assert_equal(distfn.pmf(k_shifted, *arg), 0)
    # 设置 loc 变量为 0.5，用于指定分布函数的位置参数
    loc = 0.5
    # 使用指定的分布函数 distfn，传入 loc 及其余参数来创建分布对象
    dist = distfn(loc=loc, *arg)
    # 断言两个数组的所有元素在指定误差范围内相等，用于验证概率质量函数的准确性
    npt.assert_allclose(dist.pmf(k[1:] + loc), np.diff(dist.cdf(k + loc)))
    # 断言指定数组中的元素与零相等，用于验证偏移后的概率质量函数的值是否为零
    npt.assert_equal(dist.pmf(k_shifted + loc), 0)
def check_moment_frozen(distfn, arg, m, k):
    # 使用 npt.assert_allclose 检查分布函数的 k 阶矩是否接近于 m
    npt.assert_allclose(distfn(*arg).moment(k), m,
                        atol=1e-10, rtol=1e-10)


def check_oth(distfn, arg, supp, msg):
    # 检查 distfn 的其他方法
    npt.assert_allclose(distfn.sf(supp, *arg), 1. - distfn.cdf(supp, *arg),
                        atol=1e-10, rtol=1e-10)

    # 在区间 [0.01, 0.99] 上均匀生成 20 个点的数组 q
    q = np.linspace(0.01, 0.99, 20)
    npt.assert_allclose(distfn.isf(q, *arg), distfn.ppf(1. - q, *arg),
                        atol=1e-10, rtol=1e-10)

    # 计算中位数的补生存函数值，并进行断言
    median_sf = distfn.isf(0.5, *arg)
    npt.assert_(distfn.sf(median_sf - 1, *arg) > 0.5)
    npt.assert_(distfn.cdf(median_sf + 1, *arg) > 0.5)


def check_discrete_chisquare(distfn, arg, rvs, alpha, msg):
    """Perform chisquare test for random sample of a discrete distribution

    Parameters
    ----------
    distname : string
        name of distribution function
    arg : sequence
        parameters of distribution
    alpha : float
        significance level, threshold for p-value

    Returns
    -------
    result : bool
        0 if test passes, 1 if test fails

    """
    # 设置最小质量 wsupp 为 0.05
    wsupp = 0.05

    # 构建具有最小质量 `wsupp` 的区间
    # 区间左半开，与 cdf 差异一样
    _a, _b = distfn.support(*arg)
    lo = int(max(_a, -1000))
    high = int(min(_b, 1000)) + 1
    distsupport = range(lo, high)
    last = 0
    distsupp = [lo]
    distmass = []
    for ii in distsupport:
        current = distfn.cdf(ii, *arg)
        if current - last >= wsupp - 1e-14:
            distsupp.append(ii)
            distmass.append(current - last)
            last = current
            if current > (1 - wsupp):
                break
    if distsupp[-1] < _b:
        distsupp.append(_b)
        distmass.append(1 - last)
    distsupp = np.array(distsupp)
    distmass = np.array(distmass)

    # 将区间转换为右半开，以便直方图使用
    histsupp = distsupp + 1e-8
    histsupp[0] = _a

    # 计算样本频率并进行卡方检验
    freq, hsupp = np.histogram(rvs, histsupp)
    chis, pval = stats.chisquare(np.array(freq), len(rvs)*distmass)

    # 断言检验的 p 值大于 alpha
    npt.assert_(
        pval > alpha,
        f'chisquare - test for {msg} at arg = {str(arg)} with pval = {str(pval)}'
    )


def check_scale_docstring(distfn):
    if distfn.__doc__ is not None:
        # 如果文档字符串不为空，断言中不应包含 "scale" 字符串
        npt.assert_('scale' not in distfn.__doc__)


@pytest.mark.parametrize('method', ['pmf', 'logpmf', 'cdf', 'logcdf',
                                    'sf', 'logsf', 'ppf', 'isf'])
@pytest.mark.parametrize('distname, args', distdiscrete)
def test_methods_with_lists(method, distname, args):
    # 测试离散分布是否能够接受 Python 列表作为参数
    try:
        dist = getattr(stats, distname)
    except TypeError:
        return
    if method in ['ppf', 'isf']:
        z = [0.1, 0.2]
    else:
        z = [0, 1]
    p2 = [[p]*2 for p in args]
    # 定义列表 loc 包含元素 [0, 1]，用作分布函数的位置参数
    loc = [0, 1]
    # 使用概率质量函数 (pmf) 计算给定 z 值的概率质量函数的结果，
    # 参数 *p2 表示 p2 是一个可迭代对象，传递给 pmf 函数作为额外参数
    result = dist.pmf(z, *p2, loc=loc)
    # 使用 numpy.testing.assert_allclose 函数断言 result 与生成的列表进行比较，
    # 生成的列表通过将 z 值与 p2 中的参数和 loc 结合传递给 pmf 函数来计算
    npt.assert_allclose(result,
                        [dist.pmf(*v) for v in zip(z, *p2, loc)],
                        rtol=1e-15, atol=1e-15)
# 使用 pytest 的 parametrize 装饰器，为 gh-13280 issue 编写回归测试
@pytest.mark.parametrize('distname, args', invdistdiscrete)
def test_cdf_gh13280_regression(distname, args):
    # 当形状参数无效时，测试是否输出 NaN
    dist = getattr(stats, distname)
    x = np.arange(-2, 15)
    vals = dist.cdf(x, *args)
    expected = np.nan
    npt.assert_equal(vals, expected)


def cases_test_discrete_integer_shapes():
    # 只有在拟合时才允许是整数的分布参数，但作为 PDF 等的输入可以是实数
    integrality_exceptions = {'nbinom': {'n'}, 'betanbinom': {'n'}}

    seen = set()
    for distname, shapes in distdiscrete:
        if distname in seen:
            continue
        seen.add(distname)

        try:
            dist = getattr(stats, distname)
        except TypeError:
            continue

        shape_info = dist._shape_info()

        for i, shape in enumerate(shape_info):
            if (shape.name in integrality_exceptions.get(distname, set()) or
                    not shape.integrality):
                continue

            yield distname, shape.name, shapes


@pytest.mark.parametrize('distname, shapename, shapes',
                         cases_test_discrete_integer_shapes())
def test_integer_shapes(distname, shapename, shapes):
    # 测试整数形状参数的情况
    dist = getattr(stats, distname)
    shape_info = dist._shape_info()
    shape_names = [shape.name for shape in shape_info]
    i = shape_names.index(shapename)  # 这个参数必须是整数

    shapes_copy = list(shapes)

    valid_shape = shapes[i]
    invalid_shape = valid_shape - 0.5  # 任意非整数值
    new_valid_shape = valid_shape - 1
    shapes_copy[i] = [[valid_shape], [invalid_shape], [new_valid_shape]]

    a, b = dist.support(*shapes)
    x = np.round(np.linspace(a, b, 5))

    pmf = dist.pmf(x, *shapes_copy)
    assert not np.any(np.isnan(pmf[0, :]))
    assert np.all(np.isnan(pmf[1, :]))
    assert not np.any(np.isnan(pmf[2, :]))


def test_frozen_attributes():
    # gh-14827 报告所有冻结分布都具有 pmf 和 pdf 属性；连续分布应有 pdf，离散分布应有 pmf
    message = "'rv_discrete_frozen' object has no attribute"
    with pytest.raises(AttributeError, match=message):
        stats.binom(10, 0.5).pdf
    with pytest.raises(AttributeError, match=message):
        stats.binom(10, 0.5).logpdf
    stats.binom.pdf = "herring"
    frozen_binom = stats.binom(10, 0.5)
    assert isinstance(frozen_binom, rv_discrete_frozen)
    delattr(stats.binom, 'pdf')


@pytest.mark.parametrize('distname, shapes', distdiscrete)
def test_interval(distname, shapes):
    # gh-11026 报告当 `confidence=1` 时 `interval` 返回值不正确，但左端点超过分布的支持范围
    # 确认这是所有分布的预期行为
    # 如果 distname 是字符串类型，则从 stats 模块中获取对应的分布对象
    if isinstance(distname, str):
        dist = getattr(stats, distname)
    # 如果 distname 不是字符串类型，则直接使用传入的 distname 作为分布对象
    else:
        dist = distname
    
    # 获取分布的支持区间（support），并将结果解包到 a 和 b 中
    a, b = dist.support(*shapes)
    
    # 使用 npt.assert_equal 函数验证分布对象的 percent point function (ppf) 的计算结果是否符合预期
    npt.assert_equal(dist.ppf([0, 1], *shapes), (a-1, b))
    
    # 使用 npt.assert_equal 函数验证分布对象的 inverse survival function (isf) 的计算结果是否符合预期
    npt.assert_equal(dist.isf([1, 0], *shapes), (a-1, b))
    
    # 使用 npt.assert_equal 函数验证分布对象的置信区间（interval）计算结果是否符合预期
    npt.assert_equal(dist.interval(1, *shapes), (a-1, b))
@pytest.mark.xfail_on_32bit("Sensible to machine precision")
# 标记为在32位系统上预期的失败，注释说明为机器精度敏感

def test_rv_sample():
    # Thoroughly test rv_sample and check that gh-3758 is resolved
    # 彻底测试 rv_sample 函数，并检查 gh-3758 是否已解决

    # Generate a random discrete distribution
    rng = np.random.default_rng(98430143469)
    # 使用种子98430143469初始化随机数生成器
    xk = np.sort(rng.random(10) * 10)
    # 生成长度为10的随机数组，乘以10并排序，作为离散分布的取值
    pk = rng.random(10)
    # 生成长度为10的随机数组，作为离散分布的概率
    pk /= np.sum(pk)
    # 将概率数组归一化
    dist = stats.rv_discrete(values=(xk, pk))
    # 使用给定的取值和概率创建离散随机变量

    # Generate points to the left and right of xk
    xk_left = (np.array([0] + xk[:-1].tolist()) + xk)/2
    # 计算 xk 左侧的点
    xk_right = (np.array(xk[1:].tolist() + [xk[-1]+1]) + xk)/2
    # 计算 xk 右侧的点

    # Generate points to the left and right of cdf
    cdf2 = np.cumsum(pk)
    # 计算累积分布函数
    cdf2_left = (np.array([0] + cdf2[:-1].tolist()) + cdf2)/2
    # 计算累积分布函数左侧的点
    cdf2_right = (np.array(cdf2[1:].tolist() + [1]) + cdf2)/2
    # 计算累积分布函数右侧的点

    # support - leftmost and rightmost xk
    a, b = dist.support()
    # 获取离散随机变量的支持区间
    assert_allclose(a, xk[0])
    assert_allclose(b, xk[-1])

    # pmf - supported only on the xk
    assert_allclose(dist.pmf(xk), pk)
    assert_allclose(dist.pmf(xk_right), 0)
    assert_allclose(dist.pmf(xk_left), 0)

    # logpmf is log of the pmf; log(0) = -np.inf
    with np.errstate(divide='ignore'):
        assert_allclose(dist.logpmf(xk), np.log(pk))
        assert_allclose(dist.logpmf(xk_right), -np.inf)
        assert_allclose(dist.logpmf(xk_left), -np.inf)

    # cdf - the cumulative sum of the pmf
    assert_allclose(dist.cdf(xk), cdf2)
    assert_allclose(dist.cdf(xk_right), cdf2)
    assert_allclose(dist.cdf(xk_left), [0]+cdf2[:-1].tolist())

    with np.errstate(divide='ignore'):
        assert_allclose(dist.logcdf(xk), np.log(dist.cdf(xk)),
                        atol=1e-15)
        assert_allclose(dist.logcdf(xk_right), np.log(dist.cdf(xk_right)),
                        atol=1e-15)
        assert_allclose(dist.logcdf(xk_left), np.log(dist.cdf(xk_left)),
                        atol=1e-15)

    # sf is 1-cdf
    assert_allclose(dist.sf(xk), 1-dist.cdf(xk))
    assert_allclose(dist.sf(xk_right), 1-dist.cdf(xk_right))
    assert_allclose(dist.sf(xk_left), 1-dist.cdf(xk_left))

    with np.errstate(divide='ignore'):
        assert_allclose(dist.logsf(xk), np.log(dist.sf(xk)),
                        atol=1e-15)
        assert_allclose(dist.logsf(xk_right), np.log(dist.sf(xk_right)),
                        atol=1e-15)
        assert_allclose(dist.logsf(xk_left), np.log(dist.sf(xk_left)),
                        atol=1e-15)

    # ppf
    assert_allclose(dist.ppf(cdf2), xk)
    assert_allclose(dist.ppf(cdf2_left), xk)
    assert_allclose(dist.ppf(cdf2_right)[:-1], xk[1:])
    assert_allclose(dist.ppf(0), a - 1)
    assert_allclose(dist.ppf(1), b)

    # isf
    sf2 = dist.sf(xk)
    assert_allclose(dist.isf(sf2), xk)
    assert_allclose(dist.isf(1-cdf2_left), dist.ppf(cdf2_left))
    assert_allclose(dist.isf(1-cdf2_right), dist.ppf(cdf2_right))
    assert_allclose(dist.isf(0), b)
    assert_allclose(dist.isf(1), a - 1)

    # interval is (ppf(alpha/2), isf(alpha/2))
    ps = np.linspace(0.01, 0.99, 10)
    # 计算双侧置信区间的上下限值
    int2 = dist.ppf(ps/2), dist.isf(ps/2)
    # 断言双侧置信区间是否与给定概率值对应的区间一致
    assert_allclose(dist.interval(1-ps), int2)
    # 断言零概率对应的置信区间是否与中位数一致
    assert_allclose(dist.interval(0), dist.median())
    # 断言完全概率对应的置信区间是否与给定参数范围一致
    assert_allclose(dist.interval(1), (a-1, b))

    # 中位数等同于累积分布函数的逆函数在0.5处的值
    med2 = dist.ppf(0.5)
    # 断言中位数是否与计算得到的值一致
    assert_allclose(dist.median(), med2)

    # 根据定义计算平均值、方差、偏度和峰度
    mean2 = np.sum(xk*pk)
    var2 = np.sum((xk - mean2)**2 * pk)
    skew2 = np.sum((xk - mean2)**3 * pk) / var2**(3/2)
    kurt2 = np.sum((xk - mean2)**4 * pk) / var2**2 - 3
    # 断言分布对象
```