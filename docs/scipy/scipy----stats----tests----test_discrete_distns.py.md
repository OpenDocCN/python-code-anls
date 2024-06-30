# `D:\src\scipysrc\scipy\scipy\stats\tests\test_discrete_distns.py`

```
# 导入 pytest 库，用于测试
# 导入 itertools 库，用于生成迭代器的工具函数
import pytest
import itertools

# 从 scipy.stats 中导入多个分布函数
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
                         bernoulli, boltzmann, skellam, zipf, zipfian, binom,
                         nbinom, nchypergeom_fisher, nchypergeom_wallenius,
                         randint)

# 导入 numpy 库，并从中导入部分测试函数
import numpy as np
from numpy.testing import (
    assert_almost_equal, assert_equal, assert_allclose, suppress_warnings
)

# 从 scipy.special 中导入 binom 函数
from scipy.special import binom as special_binom
# 从 scipy.optimize 中导入 root_scalar 函数
from scipy.optimize import root_scalar
# 从 scipy.integrate 中导入 quad 函数


# 使用 pytest.mark.parametrize 装饰器，定义了多个参数化测试用例
# 这些测试用例用于测试 hypergeom.cdf 函数的准确性
@pytest.mark.parametrize('k, M, n, N, expected, rtol',
                         [(3, 10, 4, 5,
                           0.9761904761904762, 1e-15),
                          (107, 10000, 3000, 215,
                           0.9999999997226765, 1e-15),
                          (10, 10000, 3000, 215,
                           2.681682217692179e-21, 5e-11)])
def test_hypergeom_cdf(k, M, n, N, expected, rtol):
    # 计算超几何分布的累积分布函数值
    p = hypergeom.cdf(k, M, n, N)
    # 使用 assert_allclose 函数断言计算值 p 与期望值 expected 的接近程度
    assert_allclose(p, expected, rtol=rtol)


# 使用 pytest.mark.parametrize 装饰器，定义了多个参数化测试用例
# 这些测试用例用于测试 hypergeom.sf 函数的准确性
@pytest.mark.parametrize('k, M, n, N, expected, rtol',
                         [(25, 10000, 3000, 215,
                           0.9999999999052958, 1e-15),
                          (125, 10000, 3000, 215,
                           1.4416781705752128e-18, 5e-11)])
def test_hypergeom_sf(k, M, n, N, expected, rtol):
    # 计算超几何分布的生存函数值
    p = hypergeom.sf(k, M, n, N)
    # 使用 assert_allclose 函数断言计算值 p 与期望值 expected 的接近程度
    assert_allclose(p, expected, rtol=rtol)


def test_hypergeom_logpmf():
    # 对称性测试
    k = 5
    N = 50
    K = 10
    n = 5
    # 计算超几何分布的对数概率质量函数值
    logpmf1 = hypergeom.logpmf(k, N, K, n)
    logpmf2 = hypergeom.logpmf(n - k, N, N - K, n)
    logpmf3 = hypergeom.logpmf(K - k, N, K, N - n)
    logpmf4 = hypergeom.logpmf(k, N, n, K)
    # 使用 assert_almost_equal 函数断言四组计算值的接近程度
    assert_almost_equal(logpmf1, logpmf2, decimal=12)
    assert_almost_equal(logpmf1, logpmf3, decimal=12)
    assert_almost_equal(logpmf1, logpmf4, decimal=12)

    # 测试相关分布
    k = 1
    N = 10
    K = 7
    n = 1
    # 计算超几何分布和伯努利分布的对数概率质量函数值
    hypergeom_logpmf = hypergeom.logpmf(k, N, K, n)
    bernoulli_logpmf = bernoulli.logpmf(k, K/N)
    # 使用 assert_almost_equal 函数断言两者计算值的接近程度
    assert_almost_equal(hypergeom_logpmf, bernoulli_logpmf, decimal=12)


def test_nhypergeom_pmf():
    M, n, r = 45, 13, 8
    k = 6
    # 计算负超几何分布的概率质量函数值
    NHG = nhypergeom.pmf(k, M, n, r)
    # 计算超几何分布的概率质量函数值，并进行调整以匹配负超几何分布
    HG = hypergeom.pmf(k, M, n, k+r-1) * (M - n - (r-1)) / (M - (k+r-1))
    # 使用 assert_allclose 函数断言两者的接近程度
    assert_allclose(HG, NHG, rtol=1e-10)


def test_nhypergeom_pmfcdf():
    M = 8
    n = 3
    r = 4
    support = np.arange(n+1)
    # 计算负超几何分布的概率质量函数和累积分布函数值
    pmf = nhypergeom.pmf(support, M, n, r)
    cdf = nhypergeom.cdf(support, M, n, r)
    # 使用 assert_allclose 函数验证 pmf 数组与指定的数值数组之间的近似相等性
    assert_allclose(pmf, [1/14, 3/14, 5/14, 5/14], rtol=1e-13)
    # 使用 assert_allclose 函数验证 cdf 数组与指定的数值数组之间的近似相等性
    assert_allclose(cdf, [1/14, 4/14, 9/14, 1.0], rtol=1e-13)
def test_nhypergeom_r0():
    # 使用 `r = 0` 进行测试。
    M = 10
    n = 3
    r = 0
    # 调用 nhypergeom 的 pmf 方法计算超几何分布的概率质量函数
    pmf = nhypergeom.pmf([[0, 1, 2, 0], [1, 2, 0, 3]], M, n, r)
    # 使用 assert_allclose 检查计算结果与预期值的接近程度
    assert_allclose(pmf, [[1, 0, 0, 1], [0, 0, 1, 0]], rtol=1e-13)


def test_nhypergeom_rvs_shape():
    # 检查当给定的大小比广播参数的维度还多时，rvs 方法返回的数组形状是否正确。
    x = nhypergeom.rvs(22, [7, 8, 9], [[12], [13]], size=(5, 1, 2, 3))
    assert x.shape == (5, 1, 2, 3)


def test_nhypergeom_accuracy():
    # 检查 nhypergeom.rvs 在 gh-13431 修复后是否与逆变换抽样给出相同的值
    np.random.seed(0)
    x = nhypergeom.rvs(22, 7, 11, size=100)
    np.random.seed(0)
    p = np.random.uniform(size=100)
    y = nhypergeom.ppf(p, 22, 7, 11)
    # 使用 assert_equal 检查两个数组是否相等
    assert_equal(x, y)


def test_boltzmann_upper_bound():
    k = np.arange(-3, 5)

    N = 1
    # 使用 boltzmann 的 pmf 方法计算玻尔兹曼分布的概率质量函数
    p = boltzmann.pmf(k, 0.123, N)
    expected = k == 0
    assert_equal(p, expected)

    lam = np.log(2)
    N = 3
    # 使用 boltzmann 的 pmf 方法计算玻尔兹曼分布的概率质量函数，同时使用 assert_allclose 检查接近程度
    p = boltzmann.pmf(k, lam, N)
    expected = [0, 0, 0, 4/7, 2/7, 1/7, 0, 0]
    assert_allclose(p, expected, rtol=1e-13)

    # 使用 boltzmann 的 cdf 方法计算玻尔兹曼分布的累积分布函数，同时使用 assert_allclose 检查接近程度
    c = boltzmann.cdf(k, lam, N)
    expected = [0, 0, 0, 4/7, 6/7, 1, 1, 1]
    assert_allclose(c, expected, rtol=1e-13)


def test_betabinom_a_and_b_unity():
    # 测试极限情况，即 betabinom(n, 1, 1) 是从 0 到 n 的离散均匀分布
    n = 20
    k = np.arange(n + 1)
    # 使用 betabinom 的 pmf 方法计算贝塔二项分布的概率质量函数
    p = betabinom(n, 1, 1).pmf(k)
    expected = np.repeat(1 / (n + 1), n + 1)
    assert_almost_equal(p, expected)


@pytest.mark.parametrize('dtypes', itertools.product(*[(int, float)]*3))
def test_betabinom_stats_a_and_b_integers_gh18026(dtypes):
    # gh-18026 报告说当一些参数是整数时，betabinom 的峰度计算会失败。检查是否已解决这个问题。
    n_type, a_type, b_type = dtypes
    n, a, b = n_type(10), a_type(2), b_type(3)
    # 使用 assert_allclose 检查计算出的峰度值与预期值的接近程度
    assert_allclose(betabinom.stats(n, a, b, moments='k'), -0.6904761904761907)


def test_betabinom_bernoulli():
    # 测试极限情况，即 betabinom(1, a, b) = bernoulli(a / (a + b))
    a = 2.3
    b = 0.63
    k = np.arange(2)
    # 使用 betabinom 的 pmf 方法计算贝塔二项分布的概率质量函数，并与伯努利分布的概率质量函数进行比较
    p = betabinom(1, a, b).pmf(k)
    expected = bernoulli(a / (a + b)).pmf(k)
    assert_almost_equal(p, expected)


def test_issue_10317():
    alpha, n, p = 0.9, 10, 1
    # 使用 assert_equal 检查 nbinom 的区间函数的计算结果是否符合预期
    assert_equal(nbinom.interval(confidence=alpha, n=n, p=p), (0, 0))


def test_issue_11134():
    alpha, n, p = 0.95, 10, 0
    # 使用 assert_equal 检查 binom 的区间函数的计算结果是否符合预期
    assert_equal(binom.interval(confidence=alpha, n=n, p=p), (0, 0))


def test_issue_7406():
    np.random.seed(0)
    # 使用 assert_equal 检查 binom 的 ppf 方法在给定参数下的计算结果是否符合预期
    assert_equal(binom.ppf(np.random.rand(10), 0, 0.5), 0)

    # 还需检查端点情况（q=0, q=1）的计算是否正确
    assert_equal(binom.ppf(0, 0, 0.5), -1)
    assert_equal(binom.ppf(1, 0, 0.5), 0)


def test_issue_5122():
    p = 0
    n = np.random.randint(100, size=10)

    x = 0
    # 使用 assert_equal 检查 binom 的 ppf 方法在给定参数下的计算结果是否符合预期
    ppf = binom.ppf(x, n, p)
    assert_equal(ppf, -1)

    x = np.linspace(0.01, 0.99, 10)
    # 使用 assert_equal 检查 binom 的 ppf 方法在给定参数下的计算结果是否符合预期
    ppf = binom.ppf(x, n, p)
    # 断言检查 ppf 是否等于 0
    assert_equal(ppf, 0)

    # 设置变量 x 为 1
    x = 1
    # 使用二项分布的逆累积分布函数计算 ppf
    ppf = binom.ppf(x, n, p)
    # 再次断言检查 ppf 是否等于 n
    assert_equal(ppf, n)
def test_issue_1603():
    # 断言测试：使用二项分布的分位点函数来验证当 n=1000, p 在 1e-3 到 1e-100 之间时，结果为0的概率为0.01
    assert_equal(binom(1000, np.logspace(-3, -100)).ppf(0.01), 0)


def test_issue_5503():
    # 设置概率值 p 为 0.5
    p = 0.5
    # 生成一个 numpy 数组 x，其元素为对数空间中从 10^3 到 10^14 之间的 12 个数
    x = np.logspace(3, 14, 12)
    # 断言测试：使用二项分布的累积分布函数来验证在参数为 (x, 2*x, p) 的情况下，得到的结果接近于 0.5，允许误差为 1e-2
    assert_allclose(binom.cdf(x, 2*x, p), 0.5, atol=1e-2)


@pytest.mark.parametrize('x, n, p, cdf_desired', [
    (300, 1000, 3/10, 0.51559351981411995636),
    (3000, 10000, 3/10, 0.50493298381929698016),
    (30000, 100000, 3/10, 0.50156000591726422864),
    (300000, 1000000, 3/10, 0.50049331906666960038),
    (3000000, 10000000, 3/10, 0.50015600124585261196),
    (30000000, 100000000, 3/10, 0.50004933192735230102),
    (30010000, 100000000, 3/10, 0.98545384016570790717),
    (29990000, 100000000, 3/10, 0.01455017177985268670),
    (29950000, 100000000, 3/10, 5.02250963487432024943e-28),
])
def test_issue_5503pt2(x, n, p, cdf_desired):
    # 断言测试：使用二项分布的累积分布函数来验证在参数为 (x, n, p) 的情况下，得到的累积概率密度函数值接近于预期的 cdf_desired
    assert_allclose(binom.cdf(x, n, p), cdf_desired)


def test_issue_5503pt3():
    # 断言测试：使用二项分布的累积分布函数来验证在参数为 (2, 10^12, 10^-12) 的情况下，得到的结果接近于 0.91969860292869777384
    assert_allclose(binom.cdf(2, 10**12, 10**-12), 0.91969860292869777384)


def test_issue_6682():
    # 断言测试：使用负二项分布的生存函数来验证在参数为 (250, 50, 32/63) 的情况下，得到的生存函数值接近于 1.460458510976452e-35
    assert_allclose(nbinom.sf(250, 50, 32./63.), 1.460458510976452e-35)


def test_issue_19747():
    # 断言测试：验证在 nbinom.logcdf 函数中，当 k 取负值时不会引发错误
    result = nbinom.logcdf([5, -1, 1], 5, 0.5)
    reference = [-0.47313352, -np.inf, -2.21297293]
    assert_allclose(result, reference)


def test_boost_divide_by_zero_issue_15101():
    # 设置参数值 n=1000, p=0.01, k=996
    n = 1000
    p = 0.01
    k = 996
    # 断言测试：使用二项分布的概率质量函数来验证在参数为 (k, n, p) 的情况下，得到的概率接近于 0.0
    assert_allclose(binom.pmf(k, n, p), 0.0)


def test_skellam_gh11474():
    # 设置 mu 为包含多个值的列表
    mu = [1, 10, 100, 1000, 5000, 5050, 5100, 5250, 6000]
    # 计算 Skellam 分布的累积分布函数，使用相同的 mu 值作为参数
    cdf = skellam.cdf(0, mu, mu)
    # R 中生成的预期值
    cdf_expected = [0.6542541612768356, 0.5448901559424127, 0.5141135799745580,
                    0.5044605891382528, 0.5019947363350450, 0.5019848365953181,
                    0.5019750827993392, 0.5019466621805060, 0.5018209330219539]
    # 断言测试：验证计算得到的 Skellam 分布累积分布函数值与预期值接近
    assert_allclose(cdf, cdf_expected)


class TestZipfian:
    def test_zipfian_asymptotic(self):
        # 设置参数值 a=6.5, N=10000000
        a = 6.5
        N = 10000000
        # 生成整数数组 k，其元素为 1 到 20
        k = np.arange(1, 21)
        # 断言测试：验证 Zipfian 分布在极限情况下（即 N 趋向于无穷大），其概率质量函数、累积分布函数和生存函数接近于相应的 Zipf 分布值
        assert_allclose(zipfian.pmf(k, a, N), zipf.pmf(k, a))
        assert_allclose(zipfian.cdf(k, a, N), zipf.cdf(k, a))
        assert_allclose(zipfian.sf(k, a, N), zipf.sf(k, a))
        assert_allclose(zipfian.stats(a, N, moments='msvk'),
                        zipf.stats(a, moments='msvk'))
    def test_zipfian_continuity(self):
        # 检验 Zipfian 分布中的连续性
        # 当 a = 1 时，切换计算调和和的方法
        alt1, agt1 = 0.99999999, 1.00000001
        N = 30
        k = np.arange(1, N + 1)
        # 验证概率质量函数在不同参数下的近似性
        assert_allclose(zipfian.pmf(k, alt1, N), zipfian.pmf(k, agt1, N),
                        rtol=5e-7)
        # 验证累积分布函数在不同参数下的近似性
        assert_allclose(zipfian.cdf(k, alt1, N), zipfian.cdf(k, agt1, N),
                        rtol=5e-7)
        # 验证生存函数在不同参数下的近似性
        assert_allclose(zipfian.sf(k, alt1, N), zipfian.sf(k, agt1, N),
                        rtol=5e-7)
        # 验证统计量在不同参数下的近似性
        assert_allclose(zipfian.stats(alt1, N, moments='msvk'),
                        zipfian.stats(agt1, N, moments='msvk'), rtol=5e-7)

    def test_zipfian_R(self):
        # 与 R 语言中的 VGAM 包进行测试比对
        # np.random.seed(0)  # 确保随机数种子
        k = np.random.randint(1, 20, size=10)
        a = np.random.rand(10) * 10 + 1
        n = np.random.randint(1, 100, size=10)
        # 预期的概率质量函数值
        pmf = [8.076972e-03, 2.950214e-05, 9.799333e-01, 3.216601e-06,
               3.158895e-04, 3.412497e-05, 4.350472e-10, 2.405773e-06,
               5.860662e-06, 1.053948e-04]
        # 预期的累积分布函数值
        cdf = [0.8964133, 0.9998666, 0.9799333, 0.9999995, 0.9998584,
               0.9999458, 1.0000000, 0.9999920, 0.9999977, 0.9998498]
        # 跳过第一个点；对于低 a 和 n，zipUC 方法不准确
        # 验证概率质量函数的近似性，跳过第一个值
        assert_allclose(zipfian.pmf(k, a, n)[1:], pmf[1:], rtol=1e-6)
        # 验证累积分布函数的近似性，跳过第一个值
        assert_allclose(zipfian.cdf(k, a, n)[1:], cdf[1:], rtol=5e-5)

    np.random.seed(0)
    naive_tests = np.vstack((np.logspace(-2, 1, 10),
                             np.random.randint(2, 40, 10))).T

    @pytest.mark.parametrize("a, n", naive_tests)
    def test_zipfian_naive(self, a, n):
        # 对简单实现进行测试

        @np.vectorize
        def Hns(n, s):
            """计算调和级数的简单实现"""
            return (1/np.arange(1, n+1)**s).sum()

        @np.vectorize
        def pzip(k, a, n):
            """Zipf分布概率质量函数的简单实现"""
            if k < 1 or k > n:
                return 0.
            else:
                return 1 / k**a / Hns(n, a)

        k = np.arange(n+1)
        pmf = pzip(k, a, n)
        cdf = np.cumsum(pmf)
        mean = np.average(k, weights=pmf)
        var = np.average((k - mean)**2, weights=pmf)
        std = var**0.5
        skew = np.average(((k-mean)/std)**3, weights=pmf)
        kurtosis = np.average(((k-mean)/std)**4, weights=pmf) - 3
        # 断言确保Zipf分布的概率质量函数正确
        assert_allclose(zipfian.pmf(k, a, n), pmf)
        # 断言确保Zipf分布的累积分布函数正确
        assert_allclose(zipfian.cdf(k, a, n), cdf)
        # 断言确保Zipf分布的统计特性正确
        assert_allclose(zipfian.stats(a, n, moments="mvsk"),
                        [mean, var, skew, kurtosis])

    def test_pmf_integer_k(self):
        k = np.arange(0, 1000)
        k_int32 = k.astype(np.int32)
        dist = zipfian(111, 22)
        pmf = dist.pmf(k)
        pmf_k_int32 = dist.pmf(k_int32)
        # 断言确保使用整数k的情况下概率质量函数不变
        assert_equal(pmf, pmf_k_int32)
class TestNCH:
    np.random.seed(2)  # 设置随机种子为2，以确保结果可重复
    shape = (2, 4, 3)
    max_m = 100
    m1 = np.random.randint(1, max_m, size=shape)    # 红球数量，随机生成在1到max_m之间的整数数组
    m2 = np.random.randint(1, max_m, size=shape)    # 白球数量，随机生成在1到max_m之间的整数数组
    N = m1 + m2                                     # 总球数量，红球数量加白球数量
    n = randint.rvs(0, N, size=N.shape)             # 抽取的球数量，随机从0到N中抽取整数，形状与N相同
    xl = np.maximum(0, n-m2)                        # 支持的下界，为0和n-m2的最大值
    xu = np.minimum(n, m1)                          # 支持的上界，为n和m1的最小值
    x = randint.rvs(xl, xu, size=xl.shape)          # 实际抽取到的球的数量，从xl到xu中随机抽取整数，形状与xl相同
    odds = np.random.rand(*x.shape)*2               # 赔率，随机生成与x形状相同的随机数数组，乘以2

    # 当传入函数名（字符串）时，测试输出更易读
    @pytest.mark.parametrize('dist_name',
                             ['nchypergeom_fisher', 'nchypergeom_wallenius'])
    def test_nch_hypergeom(self, dist_name):
        # 当赔率为1时，两个非中心超几何分布均退化为超几何分布
        dists = {'nchypergeom_fisher': nchypergeom_fisher,
                 'nchypergeom_wallenius': nchypergeom_wallenius}
        dist = dists[dist_name]
        x, N, m1, n = self.x, self.N, self.m1, self.n
        assert_allclose(dist.pmf(x, N, m1, n, odds=1),
                        hypergeom.pmf(x, N, m1, n))

    def test_nchypergeom_fisher_naive(self):
        # 测试与一个非常简单的实现进行比较
        x, N, m1, n, odds = self.x, self.N, self.m1, self.n, self.odds

        @np.vectorize
        def pmf_mean_var(x, N, m1, n, w):
            # nchypergeom_fisher的简单实现
            m2 = N - m1
            xl = np.maximum(0, n-m2)  # 支持的下界
            xu = np.minimum(n, m1)    # 支持的上界

            def f(x):
                t1 = special_binom(m1, x)
                t2 = special_binom(m2, n - x)
                return t1 * t2 * w**x

            def P(k):
                return sum(f(y)*y**k for y in range(xl, xu + 1))

            P0 = P(0)
            P1 = P(1)
            P2 = P(2)
            pmf = f(x) / P0
            mean = P1 / P0
            var = P2 / P0 - (P1 / P0)**2
            return pmf, mean, var

        pmf, mean, var = pmf_mean_var(x, N, m1, n, odds)
        assert_allclose(nchypergeom_fisher.pmf(x, N, m1, n, odds), pmf)
        assert_allclose(nchypergeom_fisher.stats(N, m1, n, odds, moments='m'),
                        mean)
        assert_allclose(nchypergeom_fisher.stats(N, m1, n, odds, moments='v'),
                        var)
    def test_wallenius_against_mpmath(self):
        # 使用 mpmath 预计算数据，因为上述的简单实现不可靠。参见 gh-13330 的源代码。
        M = 50
        n = 30
        N = 20
        odds = 2.25
        # 预期结果，使用 mpmath 计算得出。
        sup = np.arange(21)
        pmf = np.array([3.699003068656875e-20,
                        5.89398584245431e-17,
                        2.1594437742911123e-14,
                        3.221458044649955e-12,
                        2.4658279241205077e-10,
                        1.0965862603981212e-08,
                        3.057890479665704e-07,
                        5.622818831643761e-06,
                        7.056482841531681e-05,
                        0.000618899425358671,
                        0.003854172932571669,
                        0.01720592676256026,
                        0.05528844897093792,
                        0.12772363313574242,
                        0.21065898367825722,
                        0.24465958845359234,
                        0.1955114898110033,
                        0.10355390084949237,
                        0.03414490375225675,
                        0.006231989845775931,
                        0.0004715577304677075])
        mean = 14.808018384813426
        var = 2.6085975877923717

        # 检查 nchypergeom_wallenius.pmf 在 sup 上的返回值是否与预期的 pmf 相近，
        # 相对误差和绝对误差均设为较小的值。
        assert_allclose(nchypergeom_wallenius.pmf(sup, M, n, N, odds), pmf,
                        rtol=1e-13, atol=1e-13)
        # 检查 nchypergeom_wallenius.mean 是否与预期的均值相近，
        # 相对误差设为较小的值。
        assert_allclose(nchypergeom_wallenius.mean(M, n, N, odds),
                        mean, rtol=1e-13)
        # 检查 nchypergeom_wallenius.var 是否与预期的方差相近，
        # 相对误差设为较小的值。
        assert_allclose(nchypergeom_wallenius.var(M, n, N, odds),
                        var, rtol=1e-11)
@pytest.mark.parametrize("mu, q, expected",
                         [[10, 120, -1.240089881791596e-38],
                          [1500, 0, -86.61466680572661]])
def test_nbinom_11465(mu, q, expected):
    # 定义测试函数 test_nbinom_11465，用于测试负二项分布的对数累积分布函数在极端尾部的情况
    size = 20
    n, p = size, size/(size+mu)
    # 在 R 中:
    # options(digits=16)
    # pnbinom(mu=10, size=20, q=120, log.p=TRUE)
    # 计算负二项分布的参数 n 和 p
    assert_allclose(nbinom.logcdf(q, n, p), expected)


def test_gh_17146():
    # 验证离散分布在非整数 x 处返回 PMF 为零
    # 参考 gh-17146
    x = np.linspace(0, 1, 11)
    p = 0.8
    # 使用伯努利分布计算给定概率 p 的 PMF
    pmf = bernoulli(p).pmf(x)
    i = (x % 1 == 0)
    # 验证结果是否接近预期值
    assert_allclose(pmf[-1], p)
    assert_allclose(pmf[0], 1-p)
    assert_equal(pmf[~i], 0)


class TestBetaNBinom:
    @pytest.mark.parametrize('x, n, a, b, ref',
                            [[5, 5e6, 5, 20, 1.1520944824139114e-107],
                            [100, 50, 5, 20, 0.002855762954310226],
                            [10000, 1000, 5, 20, 1.9648515726019154e-05]])
    def test_betanbinom_pmf(self, x, n, a, b, ref):
        # 测试贝塔负二项分布的概率质量函数在分布尾部的准确性
        # 参考值通过 mpmath 计算得出
        # from mpmath import mp
        # mp.dps = 500
        # def betanbinom_pmf(k, n, a, b):
        #     k = mp.mpf(k)
        #     a = mp.mpf(a)
        #     b = mp.mpf(b)
        #     n = mp.mpf(n)
        #     return float(mp.binomial(n + k - mp.one, k)
        #                  * mp.beta(a + n, b + k) / mp.beta(a, b))
        # 验证结果是否接近参考值
        assert_allclose(betanbinom.pmf(x, n, a, b), ref, rtol=1e-10)


    @pytest.mark.parametrize('n, a, b, ref',
                            [[10000, 5000, 50, 0.12841520515722202],
                            [10, 9, 9, 7.9224400871459695],
                            [100, 1000, 10, 1.5849602176622748]])
    def test_betanbinom_kurtosis(self, n, a, b, ref):
        # 参考值通过 mpmath 计算得出
        # from mpmath import mp
        # def kurtosis_betanegbinom(n, a, b):
        #     n = mp.mpf(n)
        #     a = mp.mpf(a)
        #     b = mp.mpf(b)
        #     four = mp.mpf(4.)
        #     mean = n * b / (a - mp.one)
        #     var = (n * b * (n + a - 1.) * (a + b - 1.)
        #            / ((a - 2.) * (a - 1.)**2.))
        #     def f(k):
        #         return (mp.binomial(n + k - mp.one, k)
        #                 * mp.beta(a + n, b + k) / mp.beta(a, b)
        #                 * (k - mean)**four)
        #     fourth_moment = mp.nsum(f, [0, mp.inf])
        #     return float(fourth_moment/var**2 - 3.)
        # 验证结果是否接近参考值
        assert_allclose(betanbinom.stats(n, a, b, moments="k"),
                        ref, rtol=3e-15)
    def test_gh20692(self):
        # 定义一个测试函数，用于验证使用 int32 类型数据生成的概率质量函数（pmf）与使用 double 类型数据生成的输出相同
        k = np.arange(0, 1000)  # 创建一个从 0 到 999 的 numpy 数组 k
        k_int32 = k.astype(np.int32)  # 将 k 转换为 np.int32 类型，得到 k_int32
        dist = zipf(9)  # 创建一个 Zipf 分布对象 dist，参数为 9
        pmf = dist.pmf(k)  # 计算 k 对应的概率质量函数 pmf
        pmf_k_int32 = dist.pmf(k_int32)  # 计算 k_int32 对应的概率质量函数 pmf_k_int32
        assert_equal(pmf, pmf_k_int32)  # 断言 pmf 与 pmf_k_int32 相等
```