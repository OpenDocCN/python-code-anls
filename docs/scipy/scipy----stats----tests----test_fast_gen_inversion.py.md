# `D:\src\scipysrc\scipy\scipy\stats\tests\test_fast_gen_inversion.py`

```
import pytest  # 导入 pytest 模块，用于编写和运行测试用例
import warnings  # 导入 warnings 模块，用于处理警告信息
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.testing import (assert_array_equal, assert_allclose,  # 从 NumPy 测试模块中导入断言函数
                           suppress_warnings)
from copy import deepcopy  # 导入 deepcopy 函数，用于深拷贝对象
from scipy.stats.sampling import FastGeneratorInversion  # 导入 FastGeneratorInversion 类
from scipy import stats  # 导入 SciPy 统计模块


def test_bad_args():
    # 测试不合法的参数
    with pytest.raises(ValueError, match="loc must be scalar"):
        FastGeneratorInversion(stats.norm(loc=(1.2, 1.3)))

    with pytest.raises(ValueError, match="scale must be scalar"):
        FastGeneratorInversion(stats.norm(scale=[1.5, 5.7]))

    with pytest.raises(ValueError, match="'test' cannot be used to seed"):
        FastGeneratorInversion(stats.norm(), random_state="test")

    msg = "Each of the 1 shape parameters must be a scalar"
    with pytest.raises(ValueError, match=msg):
        FastGeneratorInversion(stats.gamma([1.3, 2.5]))

    with pytest.raises(ValueError, match="`dist` must be a frozen"):
        FastGeneratorInversion("xy")

    with pytest.raises(ValueError, match="Distribution 'truncnorm' is not"):
        FastGeneratorInversion(stats.truncnorm(1.3, 4.5))


def test_random_state():
    # 测试随机数种子的设置和使用
    # 固定种子
    gen = FastGeneratorInversion(stats.norm(), random_state=68734509)
    x1 = gen.rvs(size=10)
    gen.random_state = 68734509
    x2 = gen.rvs(size=10)
    assert_array_equal(x1, x2)

    # 使用生成器作为随机数种子
    urng = np.random.default_rng(20375857)
    gen = FastGeneratorInversion(stats.norm(), random_state=urng)
    x1 = gen.rvs(size=10)
    gen.random_state = np.random.default_rng(20375857)
    x2 = gen.rvs(size=10)
    assert_array_equal(x1, x2)

    # 使用 RandomState 对象作为随机数种子
    urng = np.random.RandomState(2364)
    gen = FastGeneratorInversion(stats.norm(), random_state=urng)
    x1 = gen.rvs(size=10)
    gen.random_state = np.random.RandomState(2364)
    x2 = gen.rvs(size=10)
    assert_array_equal(x1, x2)

    # evaluate_error 方法调用时，不应影响由 rvs 方法使用的随机数种子
    gen = FastGeneratorInversion(stats.norm(), random_state=68734509)
    x1 = gen.rvs(size=10)
    _ = gen.evaluate_error(size=5)  # 这会生成 5 个均匀分布的随机数
    x2 = gen.rvs(size=10)
    gen.random_state = 68734509
    x3 = gen.rvs(size=20)
    assert_array_equal(x2, x3[10:])


dists_with_params = [
    ("alpha", (3.5,)),  # alpha 分布，参数为 (3.5,)
    ("anglit", ()),  # anglit 分布，无参数
    ("argus", (3.5,)),  # argus 分布，参数为 (3.5,)
    ("argus", (5.1,)),  # argus 分布，参数为 (5.1,)
    ("beta", (1.5, 0.9)),  # beta 分布，参数为 (1.5, 0.9)
    ("cosine", ()),  # cosine 分布，无参数
    ("betaprime", (2.5, 3.3)),  # betaprime 分布，参数为 (2.5, 3.3)
    ("bradford", (1.2,)),  # bradford 分布，参数为 (1.2,)
    ("burr", (1.3, 2.4)),  # burr 分布，参数为 (1.3, 2.4)
    ("burr12", (0.7, 1.2)),  # burr12 分布，参数为 (0.7, 1.2)
    ("cauchy", ()),  # cauchy 分布，无参数
    ("chi2", (3.5,)),  # chi2 分布，参数为 (3.5,)
    ("chi", (4.5,)),  # chi 分布，参数为 (4.5,)
    ("crystalball", (0.7, 1.2)),  # crystalball 分布，参数为 (0.7, 1.2)
    ("expon", ()),  # expon 分布，无参数
    ("gamma", (1.5,)),  # gamma 分布，参数为 (1.5,)
    ("gennorm", (2.7,)),  # gennorm 分布，参数为 (2.7,)
    ("gumbel_l", ()),  # gumbel_l 分布，无参数
    ("gumbel_r", ()),  # gumbel_r 分布，无参数
    ("hypsecant", ()),  # hypsecant 分布，无参数
    ("invgauss", (3.1,)),  # invgauss 分布，参数为 (3.1,)
    ("invweibull", (1.5,)),  # invweibull 分布，参数为 (1.5,)
    ("laplace", ()),  # laplace 分布，无参数
    ("logistic", ()),  # logistic 分布，无参数
    ("maxwell", ()),  # maxwell 分布，无参数
    ("moyal", ()),  # moyal 分布，无参数
    ("norm", ()),  # norm 分布，无参数
    ("pareto", (1.3,)),  # pareto 分布，参数为 (1.3,)
    ("powerlaw", (7.6,)),  # powerlaw 分布，参数为 (7.6,)
    ("rayleigh", ())  # rayleigh 分布，无参数
]
    # 元组 ("semicircular", ())，含有一个元素 "semicircular" 和一个空元组 ()
    ("semicircular", ()),
    # 元组 ("t", (5.7,))，含有一个元素 "t" 和一个包含一个浮点数 5.7 的元组 (5.7,)
    ("t", (5.7,)),
    # 元组 ("wald", ())，含有一个元素 "wald" 和一个空元组 ()
    ("wald", ()),
    # 元组 ("weibull_max", (2.4,))，含有一个元素 "weibull_max" 和一个包含一个浮点数 2.4 的元组 (2.4,)
    ("weibull_max", (2.4,)),
    # 元组 ("weibull_min", (1.2,))，含有一个元素 "weibull_min" 和一个包含一个浮点数 1.2 的元组 (1.2,)
    ("weibull_min", (1.2,)),
@pytest.mark.parametrize(("distname, args"), dists_with_params)
def test_rvs_and_ppf(distname, args):
    # 使用参数化测试，对每个分布名称和参数组合执行以下测试
    # 设置随机数生成器的种子
    urng = np.random.default_rng(9807324628097097)
    # 使用给定的分布名称和参数创建一个 scipy.stats 的随机变量对象 rng1
    rng1 = getattr(stats, distname)(*args)
    # 从 rng1 中生成随机变量 rvs1
    rvs1 = rng1.rvs(size=500, random_state=urng)
    # 使用 FastGeneratorInversion 类创建另一个随机变量对象 rng2，与 rng1 使用相同的种子
    rng2 = FastGeneratorInversion(rng1, random_state=urng)
    # 从 rng2 中生成随机变量 rvs2
    rvs2 = rng2.rvs(size=500)
    # 断言两个随机变量序列 rvs1 和 rvs2 的 Cramér-von Mises 检验的 p 值大于 0.01
    assert stats.cramervonmises_2samp(rvs1, rvs2).pvalue > 0.01

    # 检查分布的百分位点函数 ppf
    q = [0.001, 0.1, 0.5, 0.9, 0.999]
    # 使用 assert_allclose 检查 rng1 和 rng2 的百分位点函数 ppf 的值是否在给定的误差范围内
    assert_allclose(rng1.ppf(q), rng2.ppf(q), atol=1e-10)


@pytest.mark.parametrize(("distname, args"), dists_with_params)
def test_u_error(distname, args):
    # 使用参数化测试，对每个分布名称和参数组合执行以下测试
    # 根据分布名称和参数创建一个 scipy.stats 的分布对象 dist
    dist = getattr(stats, distname)(*args)
    # 使用 FastGeneratorInversion 类创建随机数生成器对象 rng
    with suppress_warnings() as sup:
        # 过滤由 UNU.RAN 抛出的警告
        sup.filter(RuntimeWarning)
        rng = FastGeneratorInversion(dist)
    # 计算评估误差，其中 u_error 表示 U 误差
    u_error, x_error = rng.evaluate_error(
        size=10_000, random_state=9807324628097097, x_error=False
    )
    # 断言 U 误差小于等于 1e-10
    assert u_error <= 1e-10


@pytest.mark.xslow
@pytest.mark.xfail(reason="geninvgauss CDF is not accurate")
def test_geninvgauss_uerror():
    # 根据指定的 geninvgauss 分布参数创建分布对象 dist
    dist = stats.geninvgauss(3.2, 1.5)
    # 使用 FastGeneratorInversion 类创建随机数生成器对象 rng
    rng = FastGeneratorInversion(dist)
    # 计算评估误差，其中 err[0] 表示 U 误差
    err = rng.evaluate_error(size=10_000, random_state=67982)
    # 断言 U 误差小于 1e-10
    assert err[0] < 1e-10


# TODO: add more distributions
@pytest.mark.fail_slow(5)
@pytest.mark.parametrize(("distname, args"), [("beta", (0.11, 0.11))])
def test_error_extreme_params(distname, args):
    # 执行使用极端参数进行测试的函数
    # 根据分布名称和参数创建一个 scipy.stats 的分布对象 dist
    with suppress_warnings() as sup:
        # 过滤由 UNU.RAN 抛出的警告
        sup.filter(RuntimeWarning)
        dist = getattr(stats, distname)(*args)
        # 使用 FastGeneratorInversion 类创建随机数生成器对象 rng
        rng = FastGeneratorInversion(dist)
    # 计算评估误差，其中 u_error 表示 U 误差
    u_error, x_error = rng.evaluate_error(
        size=10_000, random_state=980732462809709732623, x_error=True
    )
    # 如果 U 误差大于等于 2.5 * 1e-10，则断言 x_error 小于 1e-9
    if u_error >= 2.5 * 1e-10:
        assert x_error < 1e-9


def test_evaluate_error_inputs():
    # 测试 evaluate_error 方法的输入参数
    # 使用标准正态分布创建 FastGeneratorInversion 对象 gen
    gen = FastGeneratorInversion(stats.norm())
    # 断言 evaluate_error 方法对于非整数 size 参数会引发 ValueError 异常
    with pytest.raises(ValueError, match="size must be an integer"):
        gen.evaluate_error(size=3.5)
    # 断言 evaluate_error 方法对于非整数 size 参数会引发 ValueError 异常
    with pytest.raises(ValueError, match="size must be an integer"):
        gen.evaluate_error(size=(3, 3))


def test_rvs_ppf_loc_scale():
    # 测试随机变量生成和百分位点函数在给定 loc 和 scale 下的行为
    loc, scale = 3.5, 2.3
    # 使用给定的 loc 和 scale 创建标准正态分布对象 dist
    dist = stats.norm(loc=loc, scale=scale)
    # 使用 FastGeneratorInversion 类创建随机数生成器对象 rng，指定种子为 1234
    rng = FastGeneratorInversion(dist, random_state=1234)
    # 生成 rng 中的随机变量 r
    r = rng.rvs(size=1000)
    # 对 r 进行重新缩放，以进行 Cramér-von Mises 检验
    r_rescaled = (r - loc) / scale
    # 断言重新缩放后的 r 对于正态分布的 Cramér-von Mises 检验的 p 值大于 0.01
    assert stats.cramervonmises(r_rescaled, "norm").pvalue > 0.01
    # 检查百分位点函数 _ppf 和 ppf 的结果是否在给定的误差范围内
    q = [0.001, 0.1, 0.5, 0.9, 0.999]
    assert_allclose(rng._ppf(q), rng.ppf(q), atol=1e-10)


def test_domain():
    # 测试 domain 参数是否正确传递给 UNU.RAN 生成器
    # 使用标准正态分布创建 FastGeneratorInversion 对象 rng，并指定 domain=(-1, 1)
    rng = FastGeneratorInversion(stats.norm(), domain=(-1, 1))
    # 生成 rng 中的随机变量 r
    r = rng.rvs(size=100)
    # 断言确保 r 的最小值大于等于 -1 且小于 r 的最大值，同时 r 的最大值小于等于 1
    assert -1 <= r.min() < r.max() <= 1
    
    # 设置正态分布的参数 loc 和 scale
    loc, scale = 3.5, 1.3
    # 使用 loc 和 scale 创建一个正态分布对象
    dist = stats.norm(loc=loc, scale=scale)
    # 使用 FastGeneratorInversion 类生成一个服从给定分布和指定定义域的随机数生成器
    rng = FastGeneratorInversion(dist, domain=(-1.5, 2))
    # 生成符合指定分布和定义域的随机样本，大小为 100
    r = rng.rvs(size=100)
    # 计算新的区间下界 lb 和上界 ub，根据 loc 和 scale 的定义域扩展
    lb, ub = loc - scale * 1.5, loc + scale * 2
    # 断言确保 r 的最小值大于等于 lb 且小于 r 的最大值，同时 r 的最大值小于等于 ub
    assert lb <= r.min() < r.max() <= ub
@pytest.mark.parametrize(("distname, args, expected"),
                         [("beta", (3.5, 2.5), (0, 1)),
                          ("norm", (), (-np.inf, np.inf))])
def test_support(distname, args, expected):
    # 测试支持区间是否在截断和位置/尺度应用后更新
    # 使用 beta 分布，因为它是变换后的 betaprime 分布，所以正确考虑支持区间很重要
    # （即 beta 的支持区间为 (0,1)，而 betaprime 的支持区间为 (0, inf)）
    dist = getattr(stats, distname)(*args)
    rng = FastGeneratorInversion(dist)
    assert_array_equal(rng.support(), expected)
    rng.loc = 1
    rng.scale = 2
    assert_array_equal(rng.support(), 1 + 2*np.array(expected))


@pytest.mark.parametrize(("distname, args"),
                         [("beta", (3.5, 2.5)), ("norm", ())])
def test_support_truncation(distname, args):
    # 类似的截断测试
    dist = getattr(stats, distname)(*args)
    rng = FastGeneratorInversion(dist, domain=(0.5, 0.7))
    assert_array_equal(rng.support(), (0.5, 0.7))
    rng.loc = 1
    rng.scale = 2
    assert_array_equal(rng.support(), (1 + 2 * 0.5, 1 + 2 * 0.7))


def test_domain_shift_truncation():
    # 标准正态分布的中心是零，应该移动到域的左端点
    # 如果不是这样，UNURAN 中的 PINV 将会引发警告，因为中心不在域内
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rng = FastGeneratorInversion(stats.norm(), domain=(1, 2))
    r = rng.rvs(size=100)
    assert 1 <= r.min() < r.max() <= 2


def test_non_rvs_methods_with_domain():
    # 首先，将截断正态分布与 stats.truncnorm 进行比较
    rng = FastGeneratorInversion(stats.norm(), domain=(2.3, 3.2))
    trunc_norm = stats.truncnorm(2.3, 3.2)
    # 取在域内外的值
    x = (2.0, 2.4, 3.0, 3.4)
    p = (0.01, 0.5, 0.99)
    assert_allclose(rng._cdf(x), trunc_norm.cdf(x))
    assert_allclose(rng._ppf(p), trunc_norm.ppf(p))
    loc, scale = 2, 3
    rng.loc = 2
    rng.scale = 3
    trunc_norm = stats.truncnorm(2.3, 3.2, loc=loc, scale=scale)
    x = np.array(x) * scale + loc
    assert_allclose(rng._cdf(x), trunc_norm.cdf(x))
    assert_allclose(rng._ppf(p), trunc_norm.ppf(p))

    # 使用 beta 分布进行另一个合理性检查
    # 在这种情况下，使用正确的域是很重要的，因为 beta 是 betaprime 的变换，它们具有不同的支持区间
    rng = FastGeneratorInversion(stats.beta(2.5, 3.5), domain=(0.3, 0.7))
    rng.loc = 2
    rng.scale = 2.5
    # 支持区间是 (2.75, 3.75) (2 + 2.5 * 0.3, 2 + 2.5 * 0.7)
    assert_array_equal(rng.support(), (2.75, 3.75))
    x = np.array([2.74, 2.76, 3.74, 3.76])
    # cdf 在域外应为零
    y_cdf = rng._cdf(x)
    assert_array_equal((y_cdf[0], y_cdf[3]), (0, 1))
    assert np.min(y_cdf[1:3]) > 0
    # ppf 需要将 0 和 1 映射到边界
    # 使用 assert_allclose 函数验证 rng._ppf(y_cdf) 返回的结果是否接近 (2.75, 2.76, 3.74, 3.75)
    assert_allclose(rng._ppf(y_cdf), (2.75, 2.76, 3.74, 3.75))
# 测试非反向采样方法（不考虑域）的函数
def test_non_rvs_methods_without_domain():
    # 创建标准正态分布对象
    norm_dist = stats.norm()
    # 使用快速反向生成器初始化对象，传入标准正态分布对象
    rng = FastGeneratorInversion(norm_dist)
    # 在 [-3, 3] 范围内生成均匀间隔的 10 个数
    x = np.linspace(-3, 3, num=10)
    # 待测试的累积分布函数（_cdf）应与标准正态分布的累积分布函数（cdf）在 x 点处相等
    assert_allclose(rng._cdf(x), norm_dist.cdf(x))
    # 待测试的百分点函数（_ppf）应与标准正态分布的百分点函数（ppf）在 p 点处相等
    p = (0.01, 0.5, 0.99)
    assert_allclose(rng._ppf(p), norm_dist.ppf(p))
    # 设置新的 loc 和 scale 值
    loc, scale = 0.5, 1.3
    rng.loc = loc
    rng.scale = scale
    # 使用新的 loc 和 scale 值创建标准正态分布对象
    norm_dist = stats.norm(loc=loc, scale=scale)
    # 重新测试更新后的 _cdf 和 _ppf 方法
    assert_allclose(rng._cdf(x), norm_dist.cdf(x))
    assert_allclose(rng._ppf(p), norm_dist.ppf(p))

@pytest.mark.parametrize(("domain, x"),
                         [(None, 0.5),
                         ((0, 1), 0.5),
                         ((0, 1), 1.5)])
# 测试标量输入的函数
def test_scalar_inputs(domain, x):
    """ pdf, cdf 等函数应将标量值映射到标量值。
    检查包括和不包括域的情况，因为域会影响 pdf, cdf 等函数。
    使用域内外的 x 值进行检查 """
    # 使用标准正态分布对象和指定的域初始化快速反向生成器
    rng = FastGeneratorInversion(stats.norm(), domain=domain)
    # 检查 _cdf 返回值是否是标量
    assert np.isscalar(rng._cdf(x))
    # 检查 _ppf 返回值是否是标量
    assert np.isscalar(rng._ppf(0.5))

# 测试设置域参数对大 chi 值情况下的函数
def test_domain_argus_large_chi():
    # 对于大的 chi 值，使用 Gamma 分布并且需要转换域。
    # 这是为了确保转换工作正常的测试。
    chi, lb, ub = 5.5, 0.25, 0.75
    # 使用 Argus 分布和指定的域初始化快速反向生成器
    rng = FastGeneratorInversion(stats.argus(chi), domain=(lb, ub))
    # 设置随机种子
    rng.random_state = 4574
    # 生成大小为 500 的随机数样本
    r = rng.rvs(size=500)
    # 断言样本的最小值在 [lb, ub) 范围内
    assert lb <= r.min() < r.max() <= ub
    # 使用条件累积分布函数进行拟合优度检验
    cdf = stats.argus(chi).cdf
    prob = cdf(ub) - cdf(lb)
    assert stats.cramervonmises(r, lambda x: cdf(x) / prob).pvalue > 0.05

# 测试设置 loc 和 scale 参数的函数
def test_setting_loc_scale():
    # 使用标准正态分布对象和指定的随机种子初始化快速反向生成器
    rng = FastGeneratorInversion(stats.norm(), random_state=765765864)
    # 生成大小为 1000 的随机数样本 r1
    r1 = rng.rvs(size=1000)
    # 设置 loc 和 scale 参数
    rng.loc = 3.0
    rng.scale = 2.5
    # 重新生成大小为 1000 的随机数样本 r2
    r2 = rng.rvs(1000)
    # 断言经过重新缩放的 r2 应当再次服从标准正态分布
    assert stats.cramervonmises_2samp(r1, (r2 - 3) / 2.5).pvalue > 0.05
    # 将 loc 和 scale 参数重置为默认值 loc=0, scale=1
    rng.loc = 0
    rng.scale = 1
    # 再次生成大小为 1000 的随机数样本 r2
    r2 = rng.rvs(1000)
    # 断言 r1 和 r2 应当服从相同的分布
    assert stats.cramervonmises_2samp(r1, r2).pvalue > 0.05

# 测试忽略形状参数范围的函数
def test_ignore_shape_range():
    # 创建一个消息字符串
    msg = "No generator is defined for the shape parameters"
    # 使用 t 分布和默认设置引发 ValueError 异常，断言异常消息与 msg 匹配
    with pytest.raises(ValueError, match=msg):
        rng = FastGeneratorInversion(stats.t(0.03))
    # 忽略推荐的形状参数范围初始化快速反向生成器
    rng = FastGeneratorInversion(stats.t(0.03), ignore_shape_range=True)
    # 我们可以忽略形状参数的推荐范围，
    # 但可以预期在这种情况下 u-误差会变得过大
    u_err, _ = rng.evaluate_error(size=1000, random_state=234)
    assert u_err >= 1e-6

@pytest.mark.xfail_on_32bit(
    "NumericalInversePolynomial.qrvs fails for Win 32-bit"
)
# QRVS 测试类
class TestQRVS:
    def test_input_validation(self):
        # 创建一个 FastGeneratorInversion 实例，使用正态分布作为统计模型
        gen = FastGeneratorInversion(stats.norm())

        # 检查 qmc_engine 是否为 FastGeneratorInversion 的实例，若不是则引发 ValueError 异常
        match = "`qmc_engine` must be an instance of..."
        with pytest.raises(ValueError, match=match):
            gen.qrvs(qmc_engine=0)

        # 检查 d 是否与 qmc_engine 的维度一致，若不一致则引发 ValueError 异常
        match = "`d` must be consistent with dimension of `qmc_engine`."
        with pytest.raises(ValueError, match=match):
            gen.qrvs(d=3, qmc_engine=stats.qmc.Halton(2))

    qrngs = [None, stats.qmc.Sobol(1, seed=0), stats.qmc.Halton(3, seed=0)]
    # `size=None` 不应该改变形状，`size=1` 应该增加形状
    sizes = [
        (None, tuple()),  # None 对应空元组形状
        (1, (1,)),        # 1 对应 (1,) 形状
        (4, (4,)),        # 4 对应 (4,) 形状
        ((4,), (4,)),     # (4,) 对应 (4,) 形状
        ((2, 4), (2, 4)),  # (2, 4) 对应 (2, 4) 形状
    ]
    # `d=None` 或 `d=1` 不应该增加形状
    ds = [(None, tuple()), (1, tuple()), (3, (3,))]

    @pytest.mark.parametrize("qrng", qrngs)
    @pytest.mark.parametrize("size_in, size_out", sizes)
    @pytest.mark.parametrize("d_in, d_out", ds)
    def test_QRVS_shape_consistency(self, qrng, size_in, size_out,
                                    d_in, d_out):
        # 创建一个 FastGeneratorInversion 实例，使用正态分布作为统计模型
        gen = FastGeneratorInversion(stats.norm())

        # 如果 d 和 qrng.d 不一致，引发 ValueError 异常
        if d_in is not None and qrng is not None and qrng.d != d_in:
            match = "`d` must be consistent with dimension of `qmc_engine`."
            with pytest.raises(ValueError, match=match):
                gen.qrvs(size_in, d=d_in, qmc_engine=qrng)
            return

        # 如果 d 由 qrng 决定，且 qrng 不为 None 且 qrng.d 不为 1，则 d_out 被设为 (qrng.d,)
        if d_in is None and qrng is not None and qrng.d != 1:
            d_out = (qrng.d,)

        # 预期的形状为 size_out + d_out
        shape_expected = size_out + d_out

        # 复制一个 qrng 实例
        qrng2 = deepcopy(qrng)
        # 调用 gen 的 qrvs 方法生成随机变量样本
        qrvs = gen.qrvs(size=size_in, d=d_in, qmc_engine=qrng)

        # 如果 size_in 不为 None，则断言 qrvs 的形状与 shape_expected 相同
        if size_in is not None:
            assert qrvs.shape == shape_expected

        # 如果 qrng2 不为 None，则生成均匀分布样本 uniform，并计算其对应的正态分布反函数结果 qrvs2
        if qrng2 is not None:
            uniform = qrng2.random(np.prod(size_in) or 1)
            qrvs2 = stats.norm.ppf(uniform).reshape(shape_expected)
            # 断言 qrvs 与 qrvs2 在容差范围内相等
            assert_allclose(qrvs, qrvs2, atol=1e-12)
    def test_QRVS_size_tuple(self):
        # 定义测试方法 test_QRVS_size_tuple，用于验证 QMCEngine 生成的样本形状是否正确
        # QMCEngine 生成的样本始终为形状 (n, d)。当 `size` 是一个元组时，
        # 我们在调用 qmc_engine.random 时设定 `n = prod(size)`，对样本进行转换，
        # 并将其 reshape 到最终的维度。在 reshape 时需要小心，
        # 因为 QMCEngine 返回的样本的“列”大致是独立的，但是每列内部的元素不是。
        # 我们需要确保在 reshape 过程中不会混淆：qrvs[..., i] 应该保持与 qrvs[..., i+1] 的“独立”性，
        # 但是 qrvs[..., i] 内部的元素应该来自相同的低差异序列。

        gen = FastGeneratorInversion(stats.norm())  # 创建一个用于快速生成反演的对象 gen，使用正态分布

        size = (3, 4)  # 定义一个元组 size，表示样本的维度为 (3, 4)
        d = 5  # 定义维度 d 为 5

        qrng = stats.qmc.Halton(d, seed=0)  # 创建一个 QMC 引擎对象 qrng，使用 Halton 序列，种子为 0
        qrng2 = stats.qmc.Halton(d, seed=0)  # 创建另一个 QMC 引擎对象 qrng2，使用 Halton 序列，种子为 0

        uniform = qrng2.random(np.prod(size))  # 生成一个形状为 size 的均匀随机数序列 uniform

        qrvs = gen.qrvs(size=size, d=d, qmc_engine=qrng)  # 使用 gen 对象的 qrvs 方法生成 QMC 引擎生成的样本 qrvs
        qrvs2 = stats.norm.ppf(uniform)  # 对 uniform 序列应用正态分布的反函数，生成 qrvs2

        for i in range(d):
            sample = qrvs[..., i]  # 获取 qrvs 中第 i 列的样本
            sample2 = qrvs2[:, i].reshape(size)  # 获取 qrvs2 中第 i 列的样本，并 reshape 到 size
            assert_allclose(sample, sample2, atol=1e-12)  # 断言 sample 和 sample2 的近似程度，允许误差为 1e-12
def test_burr_overflow():
    # 定义一个测试函数，用于测试 burr 分布在使用 math.exp 而不是 np.exp 时可能引起溢出错误的情况。
    # 直接实现 PDF 为 x**(-c-1) / (1+x**(-c))**(d+1) 时，也可能在设置过程中引发溢出错误。
    args = (1.89128135, 0.30195177)
    # 使用 suppress_warnings 上下文管理器来抑制警告
    with suppress_warnings() as sup:
        # 过滤掉潜在的溢出警告
        sup.filter(RuntimeWarning)
        # 使用 stats.burr(*args) 创建一个 burr 分布的快速生成器
        gen = FastGeneratorInversion(stats.burr(*args))
    # 使用生成器评估误差，random_state 设为 4326
    u_error, _ = gen.evaluate_error(random_state=4326)
    # 断言 u_error 应小于等于 1e-10
    assert u_error <= 1e-10
```