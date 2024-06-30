# `D:\src\scipysrc\scipy\scipy\stats\tests\test_sampling.py`

```
# 引入线程处理模块
import threading
# 引入 pickle 序列化模块
import pickle
# 引入 pytest 测试框架
import pytest
# 引入深拷贝函数
from copy import deepcopy
# 引入平台信息模块
import platform
# 引入系统相关模块
import sys
# 引入数学计算模块
import math
# 引入 numpy 数组处理模块
import numpy as np
# 引入 numpy 测试工具函数
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
# 引入 scipy 中的统计抽样模块
from scipy.stats.sampling import (
    TransformedDensityRejection,
    DiscreteAliasUrn,
    DiscreteGuideTable,
    NumericalInversePolynomial,
    NumericalInverseHermite,
    RatioUniforms,
    SimpleRatioUniforms,
    UNURANError
)
# 引入 pytest 的异常断言函数
from pytest import raises as assert_raises
# 引入 scipy 的统计模块
from scipy import stats
# 引入 scipy 的特殊函数模块
from scipy import special
# 引入 scipy 中的卡方检验和克拉默-冯·米泽统计模块
from scipy.stats import chisquare, cramervonmises
# 引入 scipy 中的分布参数模块
from scipy.stats._distr_params import distdiscrete, distcont
# 引入 scipy 库内部工具函数
from scipy._lib._util import check_random_state

# 共同的测试数据：这些数据可以在所有测试之间共享。

# 在所有连续方法之间共享的正态分布类
class StandardNormal:
    def pdf(self, x):
        # NumericalInverseHermite 需要的归一化常数
        return 1./np.sqrt(2.*np.pi) * np.exp(-0.5 * x*x)

    def dpdf(self, x):
        return 1./np.sqrt(2.*np.pi) * -x * np.exp(-0.5 * x*x)

    def cdf(self, x):
        return special.ndtr(x)

# 所有方法的列表，包括方法名和参数字典
all_methods = [
    ("TransformedDensityRejection", {"dist": StandardNormal()}),
    ("DiscreteAliasUrn", {"dist": [0.02, 0.18, 0.8]}),
    ("DiscreteGuideTable", {"dist": [0.02, 0.18, 0.8]}),
    ("NumericalInversePolynomial", {"dist": StandardNormal()}),
    ("NumericalInverseHermite", {"dist": StandardNormal()}),
    ("SimpleRatioUniforms", {"dist": StandardNormal(), "mode": 0})
]

# 如果运行环境是 PyPy 并且版本低于 7.3.10，设置浮点错误消息
if (sys.implementation.name == 'pypy'
        and sys.implementation.version < (7, 3, 10)):
    floaterr = r"unsupported operand type for float\(\): 'list'"
else:
    floaterr = r"must be real number, not list"

# 确保在传递无效回调时 UNU.RAN 内部发生错误。
# 不同的生成器抛出不同的错误消息。
# 因此，如果出现 `UNURANError`，我们不验证错误消息。
bad_pdfs_common = [
    # 负的概率密度函数
    (lambda x: -x, UNURANError, r"..."),
    # 返回错误的类型
    (lambda x: [], TypeError, floaterr),
    # 函数内部未定义的名称
    (lambda x: foo, NameError, r"name 'foo' is not defined"),  # type: ignore[name-defined]  # noqa: F821, E501
    # 返回无穷大值 => 溢出错误
    (lambda x: np.inf, UNURANError, r"..."),
    # NaN 值 => UNU.RAN 内部错误
    (lambda x: np.nan, UNURANError, r"..."),
    # PDF 签名错误
    (lambda: 1.0, TypeError, r"takes 0 positional arguments but 1 was given")
]

# 对 dpdf 采用相同的方法
bad_dpdf_common = [
    # 返回无穷大值
    (lambda x: np.inf, UNURANError, r"..."),
    # NaN 值 => UNU.RAN 内部错误
    (lambda x: np.nan, UNURANError, r"..."),
    # 返回错误的类型
    (lambda x: [], TypeError, floaterr),
    # 函数内部未定义的名称

    # Infinite value returned.
    (lambda x: np.inf, UNURANError, r"..."),
    # NaN value => internal error in UNU.RAN
    (lambda x: np.nan, UNURANError, r"..."),
    # Returning wrong type
    (lambda x: [], TypeError, floaterr),
    # Undefined name inside the function
    (
        lambda x: foo,  # 定义一个匿名函数，参数为 x，但尝试访问未定义的变量 foo，会引发 NameError
        NameError,  # 指定匿名函数可能会引发的异常类型为 NameError
        r"name 'foo' is not defined"  # 异常信息字符串，描述 NameError 的具体情况
    ),  # type: ignore[name-defined]  # 告知类型检查系统忽略对未定义变量的检查 # noqa: F821, E501

    # signature of dPDF wrong
    (
        lambda: 1.0,  # 定义一个不带参数的匿名函数，返回值为 1.0
        TypeError,   # 指定匿名函数可能会引发的异常类型为 TypeError
        r"takes 0 positional arguments but 1 was given"  # 异常信息字符串，描述 TypeError 的具体情况
    )
# 错误的 logpdfs 共同点：常见问题列表
bad_logpdfs_common = [
    # 返回错误的类型
    (lambda x: [], TypeError, floaterr),
    # 函数内部使用了未定义的名称
    (lambda x: foo, NameError, r"name 'foo' is not defined"),  # type: ignore[name-defined]  # noqa: F821, E501
    # 返回无穷大值 => 溢出错误
    (lambda x: np.inf, UNURANError, r"..."),
    # 返回 NaN 值 => UNU.RAN 内部错误
    (lambda x: np.nan, UNURANError, r"..."),
    # logpdf 签名错误
    (lambda: 1.0, TypeError, r"takes 0 positional arguments but 1 was given")
]


# 错误的 pv_common：常见问题列表
bad_pv_common = [
    # 必须至少包含一个元素
    ([], r"must contain at least one element"),
    # 维度数错误（预期为 1 维，实际为 2 维）
    ([[1.0, 0.0]], r"wrong number of dimensions \(expected 1, got 2\)"),
    # 包含非有限/非 NaN 值
    ([0.2, 0.4, np.nan, 0.8], r"must contain only finite / non-nan values"),
    # 包含非有限/非 NaN 值
    ([0.2, 0.4, np.inf, 0.8], r"must contain only finite / non-nan values"),
    # 必须至少包含一个非零值
    ([0.0, 0.0], r"must contain at least one non-zero value"),
]


# 错误的 sized_domains：域大小错误列表
bad_sized_domains = [
    # 域中元素数大于 2
    ((1, 2, 3), ValueError, r"must be a length 2 tuple"),
    # 空域
    ((), ValueError, r"must be a length 2 tuple")
]

# 错误的 domains：域值错误列表
bad_domains = [
    # 左值大于等于右值
    ((2, 1), UNURANError, r"left >= right"),
    # 左值等于右值
    ((1, 1), UNURANError, r"left >= right"),
]

# 无穷大和 NaN 值存在的 domains
inf_nan_domains = [
    # 左值大于等于右值
    ((10, 10), UNURANError, r"left >= right"),
    # 左值为正无穷大，右值为负无穷大
    ((np.inf, np.inf), UNURANError, r"left >= right"),
    # 左值为负无穷大，右值为正无穷大
    ((-np.inf, -np.inf), UNURANError, r"left >= right"),
    # 左值为正无穷大，右值为负无穷大
    ((np.inf, -np.inf), UNURANError, r"left >= right"),
    # 域中包含 NaN 值
    ((-np.inf, np.nan), ValueError, r"only non-nan values"),
    ((np.nan, np.inf), ValueError, r"only non-nan values")
]

# 存在 NaN 值的 domains，某些分布不支持无穷尾部，不要将 NaN 值与无穷大混合
nan_domains = [
    # 域中包含 NaN 值
    ((0, np.nan), ValueError, r"only non-nan values"),
    ((np.nan, np.nan), ValueError, r"only non-nan values")
]


# 所有方法都应当针对 NaN、错误大小和错误值的 domains 抛出错误。
@pytest.mark.parametrize("domain, err, msg",
                         bad_domains + bad_sized_domains +
                         nan_domains)  # type: ignore[operator]
@pytest.mark.parametrize("method, kwargs", all_methods)
def test_bad_domain(domain, err, msg, method, kwargs):
    # 获取对应方法
    Method = getattr(stats.sampling, method)
    # 断言应当抛出指定类型的错误并匹配特定消息
    with pytest.raises(err, match=msg):
        Method(**kwargs, domain=domain)


@pytest.mark.parametrize("method, kwargs", all_methods)
def test_random_state(method, kwargs):
    # 获取对应方法
    Method = getattr(stats.sampling, method)

    # 简单种子适用于任何版本的 NumPy
    seed = 123
    rng1 = Method(**kwargs, random_state=seed)
    rng2 = Method(**kwargs, random_state=seed)
    assert_equal(rng1.rvs(100), rng2.rvs(100))

    # 使用全局种子
    np.random.seed(123)
    rng1 = Method(**kwargs)
    rvs1 = rng1.rvs(100)
    np.random.seed(None)
    # 使用传入的关键字参数和固定的随机种子创建 Method 的实例 rng2
    rng2 = Method(**kwargs, random_state=123)
    # 使用 rng2 实例生成 100 个随机变量
    rvs2 = rng2.rvs(100)
    # 断言 rvs1 和 rvs2 应当相等，用于验证生成的随机数是否一致
    assert_equal(rvs1, rvs2)

    # 创建一个新的 NumPy 随机状态 seed1，使用 MT19937 算法和种子 123
    seed1 = np.random.RandomState(np.random.MT19937(123))
    # 使用 seed1 创建一个新的 Generator 实例 seed2，同样使用 MT19937 算法和种子 123
    seed2 = np.random.Generator(np.random.MT19937(123))
    # 使用 seed1 作为随机状态创建 Method 的实例 rng1
    rng1 = Method(**kwargs, random_state=seed1)
    # 使用 seed2 作为随机状态创建 Method 的实例 rng2
    rng2 = Method(**kwargs, random_state=seed2)
    # 断言 rng1 和 rng2 生成的 100 个随机变量应当相等，用于验证随机数生成器的一致性
    assert_equal(rng1.rvs(100), rng2.rvs(100))
`
def test_set_random_state():
    # 创建 TransformedDensityRejection 对象 rng1，使用标准正态分布，并设置随机状态为 123
    rng1 = TransformedDensityRejection(StandardNormal(), random_state=123)
    # 创建 TransformedDensityRejection 对象 rng2，使用标准正态分布，不设置随机状态
    rng2 = TransformedDensityRejection(StandardNormal())
    # 设置 rng2 的随机状态为 123
    rng2.set_random_state(123)
    # 验证 rng1 和 rng2 的 100 个随机样本是否相同
    assert_equal(rng1.rvs(100), rng2.rvs(100))
    # 创建 TransformedDensityRejection 对象 rng，使用标准正态分布，设置随机状态为 123
    rng = TransformedDensityRejection(StandardNormal(), random_state=123)
    # 生成 rng 的 100 个随机样本
    rvs1 = rng.rvs(100)
    # 设置 rng 的随机状态为 123
    rng.set_random_state(123)
    # 生成 rng 的 100 个随机样本
    rvs2 = rng.rvs(100)
    # 验证 rvs1 和 rvs2 是否相同
    assert_equal(rvs1, rvs2)

def test_threading_behaviour():
    # 测试 API 是否线程安全，验证锁机制和 `PyErr_Occurred` 是否正确
    errors = {"err1": None, "err2": None}

    # 定义一个 Distribution 类，包含 PDF 和 DPDF 方法
    class Distribution:
        def __init__(self, pdf_msg):
            self.pdf_msg = pdf_msg

        def pdf(self, x):
            # 当 x 在 (49.9, 50.0) 之间时，抛出 ValueError 异常
            if 49.9 < x < 50.0:
                raise ValueError(self.pdf_msg)
            return x

        def dpdf(self, x):
            return 1

    # 定义 func1 函数，创建 Distribution 对象，TransformedDensityRejection 对象并生成 100000 个样本
    def func1():
        dist = Distribution('foo')
        rng = TransformedDensityRejection(dist, domain=(10, 100),
                                          random_state=12)
        try:
            rng.rvs(100000)
        except ValueError as e:
            errors['err1'] = e.args[0]

    # 定义 func2 函数，创建 Distribution 对象，TransformedDensityRejection 对象并生成 100000 个样本
    def func2():
        dist = Distribution('bar')
        rng = TransformedDensityRejection(dist, domain=(10, 100),
                                          random_state=2)
        try:
            rng.rvs(100000)
        except ValueError as e:
            errors['err2'] = e.args[0]

    # 创建线程 t1 和 t2，分别运行 func1 和 func2
    t1 = threading.Thread(target=func1)
    t2 = threading.Thread(target=func2)

    # 启动线程 t1 和 t2
    t1.start()
    t2.start()

    # 等待线程 t1 和 t2 执行完成
    t1.join()
    t2.join()

    # 验证错误字典 errors 中的 err1 和 err2 是否正确
    assert errors['err1'] == 'foo'
    assert errors['err2'] == 'bar'

@pytest.mark.parametrize("method, kwargs", all_methods)
def test_pickle(method, kwargs):
    # 获取 stats.sampling 模块中的方法 Method
    Method = getattr(stats.sampling, method)
    # 创建 Method 对象 rng1，使用给定的关键字参数 kwargs 和随机状态 123
    rng1 = Method(**kwargs, random_state=123)
    # 使用 pickle 将 rng1 对象序列化
    obj = pickle.dumps(rng1)
    # 使用 pickle 将序列化后的对象反序列化为 rng2
    rng2 = pickle.loads(obj)
    # 验证 rng1 和 rng2 的 100 个随机样本是否相同
    assert_equal(rng1.rvs(100), rng2.rvs(100))

@pytest.mark.parametrize("size", [None, 0, (0, ), 1, (10, 3), (2, 3, 4, 5),
                                  (0, 0), (0, 1)])
def test_rvs_size(size):
    # 创建 TransformedDensityRejection 对象 rng，使用标准正态分布
    rng = TransformedDensityRejection(StandardNormal())
    # 根据 size 参数测试 rvs 方法的返回值
    if size is None:
        # 当 size 为 None 时，验证返回值是否为标量
        assert np.isscalar(rng.rvs(size))
    else:
        if np.isscalar(size):
            size = (size, )
        # 验证 rng.rvs(size) 的形状是否与 size 相同
        assert rng.rvs(size).shape == size

def test_with_scipy_distribution():
    # 测试 setup 是否与 SciPy 的 rv_frozen 分布兼容
    dist = stats.norm()
    urng = np.random.default_rng(0)
    rng = NumericalInverseHermite(dist, random_state=urng)
    u = np.linspace(0, 1, num=100)
    # 检查随机样本与分布统计值的一致性
    check_cont_samples(rng, dist, dist.stats())
    # 验证分位点函数的结果是否与分布的 PPF 函数相同
    assert_allclose(dist.ppf(u), rng.ppf(u))
    # 测试是否支持 `loc` 和 `scale` 参数
    dist = stats.norm(loc=10., scale=5.)
    rng = NumericalInverseHermite(dist, random_state=urng)
    # 检查连续分布的样本生成是否正确
    check_cont_samples(rng, dist, dist.stats())
    # 断言连续分布的百分位点函数是否正确
    assert_allclose(dist.ppf(u), rng.ppf(u))
    # 切换到处理离散分布
    dist = stats.binom(10, 0.2)
    # 使用别名法生成离散分布的随机数生成器
    rng = DiscreteAliasUrn(dist, random_state=urng)
    # 获取离散分布的定义域
    domain = dist.support()
    # 计算离散分布在定义域上的概率质量函数
    pv = dist.pmf(np.arange(domain[0], domain[1]+1))
    # 检查离散分布的样本生成是否正确
    check_discr_samples(rng, pv, dist.stats())
def check_cont_samples(rng, dist, mv_ex, rtol=1e-7, atol=1e-1):
    # 从随机数生成器 `rng` 中生成 100000 个随机变量样本
    rvs = rng.rvs(100000)
    # 计算样本的均值和方差
    mv = rvs.mean(), rvs.var()
    # 只有当方差有限时才测试矩的匹配性
    if np.isfinite(mv_ex[1]):
        # 断言样本的均值和方差与期望值 `mv_ex` 在指定的相对误差和绝对误差范围内匹配
        assert_allclose(mv, mv_ex, rtol=rtol, atol=atol)
    # 从随机数生成器 `rng` 中再生成 500 个随机变量样本
    rvs = rng.rvs(500)
    # 将分布对象 `dist` 的累积分布函数向量化
    dist.cdf = np.vectorize(dist.cdf)
    # 使用 Cramer Von Mises 检验拟合优度，返回 p 值
    pval = cramervonmises(rvs, dist.cdf).pvalue
    # 断言 p 值大于 0.1
    assert pval > 0.1


def check_discr_samples(rng, pv, mv_ex, rtol=1e-3, atol=1e-1):
    # 从随机数生成器 `rng` 中生成 100000 个随机变量样本
    rvs = rng.rvs(100000)
    # 测试前几个矩是否匹配
    mv = rvs.mean(), rvs.var()
    # 断言样本的均值和方差与期望值 `mv_ex` 在指定的相对误差和绝对误差范围内匹配
    assert_allclose(mv, mv_ex, rtol=rtol, atol=atol)
    # 归一化概率向量 `pv`
    pv = pv / pv.sum()
    # 使用卡方检验检验拟合优度
    obs_freqs = np.zeros_like(pv)
    _, freqs = np.unique(rvs, return_counts=True)
    freqs = freqs / freqs.sum()
    obs_freqs[:freqs.size] = freqs
    pval = chisquare(obs_freqs, pv).pvalue
    # 断言 p 值大于 0.1
    assert pval > 0.1


def test_warning_center_not_in_domain():
    # 如果提供的中心或计算的中心不在分布的定义域内，UNURAN 将发出警告
    msg = "102 : center moved into domain of distribution"
    # 使用 pytest 检测是否产生 RuntimeWarning，并匹配警告信息 `msg`
    with pytest.warns(RuntimeWarning, match=msg):
        NumericalInversePolynomial(StandardNormal(), center=0, domain=(3, 5))
    with pytest.warns(RuntimeWarning, match=msg):
        NumericalInversePolynomial(StandardNormal(), domain=(3, 5))


@pytest.mark.parametrize('method', ["SimpleRatioUniforms",
                                    "NumericalInversePolynomial",
                                    "TransformedDensityRejection"])
def test_error_mode_not_in_domain(method):
    # 如果模式不在定义域内，UNURAN 将引发错误
    # 这种行为与中心不在定义域内的情况不同。模式应该是精确值，中心可以是近似值
    Method = getattr(stats.sampling, method)
    msg = "17 : mode not in domain"
    # 使用 pytest 检测是否引发 UNURANError，并匹配错误信息 `msg`
    with pytest.raises(UNURANError, match=msg):
        Method(StandardNormal(), mode=0, domain=(3, 5))


@pytest.mark.parametrize('method', ["NumericalInverseHermite",
                                    "NumericalInversePolynomial"])
class TestQRVS:
    def test_input_validation(self, method):
        # 检验输入的合法性
        match = "`qmc_engine` must be an instance of..."
        # 使用 pytest 检测是否引发 ValueError，并匹配错误信息 `match`
        with pytest.raises(ValueError, match=match):
            Method = getattr(stats.sampling, method)
            gen = Method(StandardNormal())
            gen.qrvs(qmc_engine=0)

        # QMCEngine 和旧版本 NumPy 的问题
        Method = getattr(stats.sampling, method)
        gen = Method(StandardNormal())

        match = "`d` must be consistent with dimension of `qmc_engine`."
        # 使用 pytest 检测是否引发 ValueError，并匹配错误信息 `match`
        with pytest.raises(ValueError, match=match):
            gen.qrvs(d=3, qmc_engine=stats.qmc.Halton(2))

    qrngs = [None, stats.qmc.Sobol(1, seed=0), stats.qmc.Halton(3, seed=0)]
    # `size=None` should not add anything to the shape, `size=1` should
    # 定义一个包含多种大小和维度参数的列表
    sizes = [(None, tuple()), (1, (1,)), (4, (4,)),
             ((4,), (4,)), ((2, 4), (2, 4))]  # type: ignore

    # 定义一组测试参数，每个元素包含输入输出的大小信息
    # `d=None` 和 `d=1` 都不应该改变形状
    ds = [(None, tuple()), (1, tuple()), (3, (3,))]

    @pytest.mark.parametrize('qrng', qrngs)
    @pytest.mark.parametrize('size_in, size_out', sizes)
    @pytest.mark.parametrize('d_in, d_out', ds)
    # 定义测试方法，参数化多个参数
    def test_QRVS_shape_consistency(self, qrng, size_in, size_out,
                                    d_in, d_out, method):
        # 检查操作系统是否为 Windows 32 位
        w32 = sys.platform == "win32" and platform.architecture()[0] == "32bit"
        # 如果是 Windows 32 位且使用的方法是 "NumericalInversePolynomial"，标记为预期失败
        if w32 and method == "NumericalInversePolynomial":
            pytest.xfail("NumericalInversePolynomial.qrvs fails for Win "
                         "32-bit")

        # 创建标准正态分布对象
        dist = StandardNormal()
        # 根据方法名称从 stats.sampling 模块中获取相应的方法类
        Method = getattr(stats.sampling, method)
        # 使用标准正态分布对象创建生成器
        gen = Method(dist)

        # 如果 d_in 不为 None，并且 qrng 不为 None，且 qrng.d 与 d_in 不一致，引发 ValueError 异常
        if d_in is not None and qrng is not None and qrng.d != d_in:
            match = "`d` must be consistent with dimension of `qmc_engine`."
            with pytest.raises(ValueError, match=match):
                gen.qrvs(size_in, d=d_in, qmc_engine=qrng)
            return

        # 如果 d_in 为 None，并且 qrng 不为 None，且 qrng.d 不为 1，则 d_out 被设置为 (qrng.d,)
        if d_in is None and qrng is not None and qrng.d != 1:
            d_out = (qrng.d,)

        # 预期的输出形状为 size_out 加上 d_out
        shape_expected = size_out + d_out

        # 深拷贝 qrng 对象
        qrng2 = deepcopy(qrng)
        # 生成随机变量样本集
        qrvs = gen.qrvs(size=size_in, d=d_in, qmc_engine=qrng)
        # 如果 size_in 不为 None，断言 qrvs 的形状应该等于 shape_expected
        if size_in is not None:
            assert qrvs.shape == shape_expected

        # 如果 qrng2 不为 None，进行进一步的比较
        if qrng2 is not None:
            # 生成均匀分布样本
            uniform = qrng2.random(np.prod(size_in) or 1)
            # 使用逆正态分布函数对均匀分布样本进行转换，并重塑为 shape_expected 形状
            qrvs2 = stats.norm.ppf(uniform).reshape(shape_expected)
            # 使用 assert_allclose 函数比较 qrvs 和 qrvs2 是否近似相等
            assert_allclose(qrvs, qrvs2, atol=1e-12)
    # 定义一个测试方法，用于测试 QMC 引擎生成的样本的大小是否正确
    def test_QRVS_size_tuple(self, method):
        # QMCEngine 生成的样本形状总是 (n, d)。当 `size` 是一个元组时，
        # 我们在调用 qmc_engine.random 时将 `n = prod(size)` 设置为样本的大小，
        # 然后对样本进行转换，并将其重新塑形为最终的维度。在重新塑形时，
        # 我们需要小心，因为 QMCEngine 返回的样本的“列”在某种程度上是独立的，
        # 但是每列内部的元素不是独立的。我们需要确保在重新塑形时不要混淆这一点：
        # qrvs[..., i] 应该保持与 qrvs[..., i+1] 的“列”独立，
        # 但是 qrvs[..., i] 内部的元素应该来自同一个低失配序列。

        # 创建一个标准正态分布对象
        dist = StandardNormal()
        # 根据给定方法名称动态获取对应的抽样方法类
        Method = getattr(stats.sampling, method)
        # 使用标准正态分布对象创建抽样方法实例
        gen = Method(dist)

        # 定义样本的大小为 (3, 4)
        size = (3, 4)
        # 定义维度 d 为 5
        d = 5
        # 使用 Halton 序列生成器创建 QMC 引擎 qrng 和 qrng2
        qrng = stats.qmc.Halton(d, seed=0)
        qrng2 = stats.qmc.Halton(d, seed=0)

        # 使用 qrng2 生成随机均匀分布的样本
        uniform = qrng2.random(np.prod(size))

        # 使用 QMC 引擎 qrng 生成 QMC 随机变量样本 qrvs
        qrvs = gen.qrvs(size=size, d=d, qmc_engine=qrng)
        # 使用标准正态分布对象 stats.norm 转换均匀分布样本 uniform 为样本 qrvs2
        qrvs2 = stats.norm.ppf(uniform)

        # 对每个维度进行比较，确保 qrvs[..., i] 与 qrvs2[:, i].reshape(size) 接近
        for i in range(d):
            sample = qrvs[..., i]
            sample2 = qrvs2[:, i].reshape(size)
            # 使用 assert_allclose 函数检查两个样本的接近程度，设置公差为 1e-12
            assert_allclose(sample, sample2, atol=1e-12)
class TestTransformedDensityRejection:
    # 定义了一个测试类 TestTransformedDensityRejection

    # Simple Custom Distribution
    # 简单的自定义分布类 dist0
    class dist0:
        # 概率密度函数
        def pdf(self, x):
            return 3/4 * (1-x*x)
            # 返回 3/4 * (1 - x^2)

        # 概率密度函数的导数
        def dpdf(self, x):
            return 3/4 * (-2*x)
            # 返回 3/4 * (-2x)

        # 累积分布函数
        def cdf(self, x):
            return 3/4 * (x - x**3/3 + 2/3)
            # 返回 3/4 * (x - x^3/3 + 2/3)

        # 支持区间
        def support(self):
            return -1, 1
            # 返回支持区间 [-1, 1]

    # Standard Normal Distribution
    # 标准正态分布类 dist1
    class dist1:
        # 概率密度函数
        def pdf(self, x):
            return stats.norm._pdf(x / 0.1)
            # 返回标准正态分布在 x / 0.1 处的概率密度值

        # 概率密度函数的导数
        def dpdf(self, x):
            return -x / 0.01 * stats.norm._pdf(x / 0.1)
            # 返回标准正态分布在 x / 0.1 处的概率密度函数导数值

        # 累积分布函数
        def cdf(self, x):
            return stats.norm._cdf(x / 0.1)
            # 返回标准正态分布在 x / 0.1 处的累积分布函数值

    # pdf with piecewise linear function as transformed density
    # with T = -1/sqrt with shift. Taken from UNU.RAN test suite
    # (from file t_tdr_ps.c)
    # 带有分段线性函数的变换密度的概率密度函数，其中 T = -1/sqrt，带有位移。取自 UNU.RAN 测试套件的 t_tdr_ps.c 文件
    class dist2:
        # 构造函数，接收一个位移参数 shift
        def __init__(self, shift):
            self.shift = shift

        # 概率密度函数
        def pdf(self, x):
            x -= self.shift
            y = 1. / (abs(x) + 1.)
            return 0.5 * y * y
            # 返回变换后的密度函数值

        # 概率密度函数的导数
        def dpdf(self, x):
            x -= self.shift
            y = 1. / (abs(x) + 1.)
            y = y * y * y
            return y if (x < 0.) else -y
            # 返回变换后的密度函数导数值

        # 累积分布函数
        def cdf(self, x):
            x -= self.shift
            if x <= 0.:
                return 0.5 / (1. - x)
            else:
                return 1. - 0.5 / (1. + x)
            # 返回变换后的累积分布函数值

    # 定义了一个 dists 列表，包含四个分布对象
    dists = [dist0(), dist1(), dist2(0.), dist2(10000.)]

    # exact mean and variance of the distributions in the list dists
    # dists 列表中各分布的精确均值和方差
    mv0 = [0., 4./15.]
    mv1 = [0., 0.01]
    mv2 = [0., np.inf]
    mv3 = [10000., np.inf]
    mvs = [mv0, mv1, mv2, mv3]

    @pytest.mark.parametrize("dist, mv_ex",
                             zip(dists, mvs))
    # 参数化测试，参数为 dist 和对应的精确均值方差 mv_ex
    def test_basic(self, dist, mv_ex):
        with suppress_warnings() as sup:
            # 过滤 UNU.RAN 抛出的警告
            sup.filter(RuntimeWarning)
            rng = TransformedDensityRejection(dist, random_state=42)
            # 使用 TransformedDensityRejection 类初始化 rng 对象
        check_cont_samples(rng, dist, mv_ex)
        # 调用 check_cont_samples 函数检查连续样本

    # PDF 0 everywhere => bad construction points
    # 各处的概率密度函数值为 0 => 构造点错误
    bad_pdfs = [(lambda x: 0, UNURANError, r"50 : bad construction points.")]
    bad_pdfs += bad_pdfs_common  # type: ignore[arg-type]

    @pytest.mark.parametrize("pdf, err, msg", bad_pdfs)
    # 参数化测试，参数为 pdf、错误类型 err 和错误消息 msg
    def test_bad_pdf(self, pdf, err, msg):
        class dist:
            pass
        dist.pdf = pdf
        dist.dpdf = lambda x: 1  # an arbitrary dPDF
        # 设置 dist 类的 pdf 方法和 dpdf 方法
        with pytest.raises(err, match=msg):
            TransformedDensityRejection(dist)
            # 检查使用 TransformedDensityRejection 类初始化 dist 对象时是否抛出指定错误

    @pytest.mark.parametrize("dpdf, err, msg", bad_dpdf_common)
    # 参数化测试，参数为 dpdf、错误类型 err 和错误消息 msg
    def test_bad_dpdf(self, dpdf, err, msg):
        class dist:
            pass
        dist.pdf = lambda x: x
        dist.dpdf = dpdf
        # 设置 dist 类的 pdf 方法和 dpdf 方法
        with pytest.raises(err, match=msg):
            TransformedDensityRejection(dist, domain=(1, 10))
            # 检查使用 TransformedDensityRejection 类初始化 dist 对象时是否抛出指定错误

    # test domains with inf + nan in them. need to write a custom test for
    # this because not all methods support infinite tails.
    # 测试包含无穷大和 NaN 的域。需要编写自定义测试，因为不是所有的方法都支持无穷尾部。
    @pytest.mark.parametrize("domain, err, msg", inf_nan_domains)
    # 测试函数：检查在给定域中，创建 TransformedDensityRejection 对象时是否引发指定错误
    def test_inf_nan_domains(self, domain, err, msg):
        # 使用 pytest 检查是否会引发特定错误类型，并匹配错误消息
        with pytest.raises(err, match=msg):
            TransformedDensityRejection(StandardNormal(), domain=domain)

    # 参数化测试：检查当 construction_points 为非正整数时是否引发 ValueError
    @pytest.mark.parametrize("construction_points", [-1, 0, 0.1])
    def test_bad_construction_points_scalar(self, construction_points):
        # 使用 pytest 检查是否会引发 ValueError，并匹配错误消息
        with pytest.raises(ValueError, match=r"`construction_points` must be "
                                             r"a positive integer."):
            TransformedDensityRejection(
                StandardNormal(), construction_points=construction_points
            )

    # 测试函数：检查当 construction_points 是空数组或非严格单调递增时是否引发 ValueError 或 RuntimeWarning
    def test_bad_construction_points_array(self):
        # 空数组情况
        construction_points = []
        with pytest.raises(ValueError, match=r"`construction_points` must "
                                             r"either be a "
                                             r"scalar or a non-empty array."):
            TransformedDensityRejection(
                StandardNormal(), construction_points=construction_points
            )

        # 不严格单调递增情况
        construction_points = [1, 1, 1, 1, 1, 1]
        with pytest.warns(RuntimeWarning, match=r"33 : starting points not "
                                                r"strictly monotonically "
                                                r"increasing"):
            TransformedDensityRejection(
                StandardNormal(), construction_points=construction_points
            )

        # 包含 NaN 的情况
        construction_points = [np.nan, np.nan, np.nan]
        with pytest.raises(UNURANError, match=r"50 : bad construction "
                                              r"points."):
            TransformedDensityRejection(
                StandardNormal(), construction_points=construction_points
            )

        # 超出指定域的情况
        construction_points = [-10, 10]
        with pytest.warns(RuntimeWarning, match=r"50 : starting point out of "
                                                r"domain"):
            TransformedDensityRejection(
                StandardNormal(), domain=(-3, 3),
                construction_points=construction_points
            )

    # 参数化测试：检查当 c 不在允许范围内时是否引发 ValueError
    @pytest.mark.parametrize("c", [-1., np.nan, np.inf, 0.1, 1.])
    def test_bad_c(self, c):
        # 使用 pytest 检查是否会引发 ValueError，并匹配错误消息
        msg = r"`c` must either be -0.5 or 0."
        with pytest.raises(ValueError, match=msg):
            TransformedDensityRejection(StandardNormal(), c=-1.)

    # 参数化测试数据：u 的各种可能取值，用于后续的参数化测试
    u = [np.linspace(0, 1, num=1000), [], [[]], [np.nan],
         [-np.inf, np.nan, np.inf], 0,
         [[np.nan, 0.5, 0.1], [0.2, 0.4, np.inf], [-2, 3, 4]]]

    # 参数化测试：检查 TransformedDensityRejection 对象在不同 u 值下的行为
    @pytest.mark.parametrize("u", u)
    # 定义一个测试方法，用于验证 ppf_hat 方法的准确性
    def test_ppf_hat(self, u):
        # 增大 max_squeeze_hat_ratio，以提高 ppf_hat 的精确性
        rng = TransformedDensityRejection(StandardNormal(),
                                          max_squeeze_hat_ratio=0.9999)
        # 屏蔽特定的 RuntimeWarning，以避免与 NaN 比较时引发警告
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in greater")
            sup.filter(RuntimeWarning, "invalid value encountered in "
                                       "greater_equal")
            sup.filter(RuntimeWarning, "invalid value encountered in less")
            sup.filter(RuntimeWarning, "invalid value encountered in "
                                       "less_equal")
            # 计算 rng 对象的 ppf_hat 方法的返回值
            res = rng.ppf_hat(u)
            # 期望的结果通过 stats.norm.ppf(u) 获得
            expected = stats.norm.ppf(u)
        # 使用 assert_allclose 检查 res 和 expected 的近似程度
        assert_allclose(res, expected, rtol=1e-3, atol=1e-5)
        # 检查 res 和 expected 的形状是否相同
        assert res.shape == expected.shape

    # 定义另一个测试方法，用于验证处理异常分布情况的 TransformedDensityRejection 类
    def test_bad_dist(self):
        # 定义一个空的分布类 dist
        class dist:
            ...

        # 出现 ValueError 时的期望异常信息
        msg = r"`pdf` required but not found."
        # 使用 pytest.raises 来检查是否引发了预期的 ValueError 异常，并匹配异常信息
        with pytest.raises(ValueError, match=msg):
            TransformedDensityRejection(dist)

        # 定义另一个分布类 dist，其中缺少 dpdf 方法
        class dist:
            # 定义一个简单的 pdf 方法，但未定义 dpdf 方法
            pdf = lambda x: 1-x*x  # noqa: E731

        # 出现 ValueError 时的期望异常信息
        msg = r"`dpdf` required but not found."
        # 使用 pytest.raises 来检查是否引发了预期的 ValueError 异常，并匹配异常信息
        with pytest.raises(ValueError, match=msg):
            TransformedDensityRejection(dist)
class TestDiscreteAliasUrn:
    # DAU fails on these probably because of large domains and small
    # computation errors in PMF. Mean/SD match but chi-squared test fails.
    basic_fail_dists = {
        'nchypergeom_fisher',  # numerical errors on tails
        'nchypergeom_wallenius',  # numerical errors on tails
        'randint'  # fails on 32-bit ubuntu
    }

    @pytest.mark.parametrize("distname, params", distdiscrete)
    def test_basic(self, distname, params):
        # Check if the distribution name is in the set of distributions known to fail
        if distname in self.basic_fail_dists:
            msg = ("DAU fails on these probably because of large domains "
                   "and small computation errors in PMF.")
            # Skip the test with the provided message
            pytest.skip(msg)
        
        # Determine the distribution object based on the name or object itself
        if not isinstance(distname, str):
            dist = distname
        else:
            dist = getattr(stats, distname)
        
        # Initialize the distribution object with parameters
        dist = dist(*params)
        
        # Get the domain (support) of the distribution
        domain = dist.support()
        
        # Check if the domain is finite
        if not np.isfinite(domain[1] - domain[0]):
            # Skip the test if the domain has infinite tails
            pytest.skip("DAU only works with a finite domain.")
        
        # Generate an array of integers within the distribution's domain
        k = np.arange(domain[0], domain[1]+1)
        
        # Calculate the probability mass function (PMF) at each point in k
        pv = dist.pmf(k)
        
        # Compute the mean and variance using the distribution's stats method
        mv_ex = dist.stats('mv')
        
        # Initialize a DiscreteAliasUrn random number generator with fixed seed
        rng = DiscreteAliasUrn(dist, random_state=42)
        
        # Check discrete samples against expected PMF and statistics
        check_discr_samples(rng, pv, mv_ex)

    # Can't use bad_pmf_common here as we evaluate PMF early on to avoid
    # unhelpful errors from UNU.RAN.
    bad_pmf = [
        # inf returned
        (lambda x: np.inf, ValueError,
         r"must contain only finite / non-nan values"),
        # nan returned
        (lambda x: np.nan, ValueError,
         r"must contain only finite / non-nan values"),
        # all zeros
        (lambda x: 0.0, ValueError,
         r"must contain at least one non-zero value"),
        # Undefined name inside the function
        (lambda x: foo, NameError,  # type: ignore[name-defined]  # noqa: F821
         r"name 'foo' is not defined"),
        # Returning wrong type.
        (lambda x: [], ValueError,
         r"setting an array element with a sequence."),
        # probabilities < 0
        (lambda x: -x, UNURANError,
         r"50 : probability < 0"),
        # signature of PMF wrong
        (lambda: 1.0, TypeError,
         r"takes 0 positional arguments but 1 was given")
    ]

    @pytest.mark.parametrize("pmf, err, msg", bad_pmf)
    def test_bad_pmf(self, pmf, err, msg):
        # Define a dummy class for the distribution with a specified PMF function
        class dist:
            pass
        
        # Assign the provided PMF lambda function to the pmf attribute of dist
        dist.pmf = pmf
        
        # Ensure that the test raises the expected error with the given message
        with pytest.raises(err, match=msg):
            # Attempt to create a DiscreteAliasUrn instance with the defined distribution
            DiscreteAliasUrn(dist, domain=(1, 10))

    @pytest.mark.parametrize("pv", [[0.18, 0.02, 0.8],
                                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    # 使用给定的概率向量 `pv`，将其转换为 numpy 数组，并确保数据类型为 float64
    def test_sampling_with_pv(self, pv):
        pv = np.asarray(pv, dtype=np.float64)
        # 使用给定的概率向量创建一个 DiscreteAliasUrn 随机数生成器对象 `rng`
        rng = DiscreteAliasUrn(pv, random_state=123)
        # 生成 100,000 个随机样本
        rng.rvs(100_000)
        # 根据概率向量重新归一化 `pv`
        pv = pv / pv.sum()
        # 创建变量数组 `variates`，范围从 0 到 `pv` 的长度
        variates = np.arange(0, len(pv))
        # 测试前几个矩匹配是否正确
        m_expected = np.average(variates, weights=pv)
        v_expected = np.average((variates - m_expected) ** 2, weights=pv)
        # 预期的均值和方差
        mv_expected = m_expected, v_expected
        # 检查离散样本是否符合预期
        check_discr_samples(rng, pv, mv_expected)

    # 使用参数化测试，对不良的概率向量 `pv` 进行测试，并验证是否抛出预期的 ValueError 异常
    @pytest.mark.parametrize("pv, msg", bad_pv_common)
    def test_bad_pv(self, pv, msg):
        with pytest.raises(ValueError, match=msg):
            DiscreteAliasUrn(pv)

    # 测试当输入域包含无限值时是否会抛出预期的 ValueError 异常
    # DAU 不支持无限的尾部，因此当域中存在 inf 时应该抛出错误
    inf_domain = [(-np.inf, np.inf), (np.inf, np.inf), (-np.inf, -np.inf),
                  (0, np.inf), (-np.inf, 0)]
    @pytest.mark.parametrize("domain", inf_domain)
    def test_inf_domain(self, domain):
        with pytest.raises(ValueError, match=r"must be finite"):
            DiscreteAliasUrn(stats.binom(10, 0.2), domain=domain)

    # 测试当相对于全部概率向量的 urn 大小小于 1 时是否会发出 RuntimeWarning 警告
    def test_bad_urn_factor(self):
        with pytest.warns(RuntimeWarning, match=r"relative urn size < 1."):
            DiscreteAliasUrn([0.5, 0.5], urn_factor=-1)

    # 测试当未提供 `domain` 参数而概率向量不可用时，是否会抛出预期的 ValueError 异常
    def test_bad_args(self):
        # 错误消息的期望文本
        msg = (r"`domain` must be provided when the "
               r"probability vector is not available.")
        # 定义一个虚拟的概率分布类 `dist`
        class dist:
            def pmf(self, x):
                return x
        # 使用 pytest 检查是否抛出预期的 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            DiscreteAliasUrn(dist)

    # 测试特定问题 #GH19359 中的情况，使用 softmax 函数生成概率向量 `pv`
    def test_gh19359(self):
        pv = special.softmax(np.ones((1533,)))
        # 使用该概率向量创建一个 DiscreteAliasUrn 随机数生成器对象 `rng`
        rng = DiscreteAliasUrn(pv, random_state=42)
        # 检查生成的离散样本是否正确
        check_discr_samples(rng, pv, (1532 / 2, (1532**2 - 1) / 12),
                            rtol=5e-3)
class TestNumericalInversePolynomial:
    # 定义一个测试类 TestNumericalInversePolynomial，用于测试数值反函数多项式

    # Simple Custom Distribution
    # 简单的自定义分布类 dist0
    class dist0:
        def pdf(self, x):
            # 概率密度函数，返回 3/4 * (1-x*x)
            return 3/4 * (1-x*x)

        def cdf(self, x):
            # 累积分布函数，返回 3/4 * (x - x**3/3 + 2/3)
            return 3/4 * (x - x**3/3 + 2/3)

        def support(self):
            # 支持区间，返回 (-1, 1)
            return -1, 1

    # Standard Normal Distribution
    # 标准正态分布类 dist1
    class dist1:
        def pdf(self, x):
            # 概率密度函数，返回 stats.norm._pdf(x / 0.1)
            return stats.norm._pdf(x / 0.1)

        def cdf(self, x):
            # 累积分布函数，返回 stats.norm._cdf(x / 0.1)
            return stats.norm._cdf(x / 0.1)

    # Sin 2 distribution
    # Sin 2 分布类 dist2
    #          /  0.05 + 0.45*(1 +sin(2 Pi x))  if |x| <= 1
    #  f(x) = <
    #          \  0        otherwise
    # 取自 UNU.RAN 测试套件（来自文件 t_pinv.c）
    class dist2:
        def pdf(self, x):
            # 概率密度函数，返回 0.05 + 0.45 * (1 + np.sin(2*np.pi*x))
            return 0.05 + 0.45 * (1 + np.sin(2*np.pi*x))

        def cdf(self, x):
            # 累积分布函数，返回 (0.05*(x + 1) +
            #                   0.9*(1. + 2.*np.pi*(1 + x) - np.cos(2.*np.pi*x)) /
            #                   (4.*np.pi))
            return (0.05*(x + 1) +
                    0.9*(1. + 2.*np.pi*(1 + x) - np.cos(2.*np.pi*x)) /
                    (4.*np.pi))

        def support(self):
            # 支持区间，返回 (-1, 1)
            return -1, 1

    # Sin 10 distribution
    # Sin 10 分布类 dist3
    #          /  0.05 + 0.45*(1 +sin(2 Pi x))  if |x| <= 5
    #  f(x) = <
    #          \  0        otherwise
    # 取自 UNU.RAN 测试套件（来自文件 t_pinv.c）
    class dist3:
        def pdf(self, x):
            # 概率密度函数，返回 0.2 * (0.05 + 0.45 * (1 + np.sin(2*np.pi*x)))
            return 0.2 * (0.05 + 0.45 * (1 + np.sin(2*np.pi*x)))

        def cdf(self, x):
            # 累积分布函数，返回 x/10. + 0.5 + 0.09/(2*np.pi) * (np.cos(10*np.pi) -
            #                                                    np.cos(2*np.pi*x))
            return x/10. + 0.5 + 0.09/(2*np.pi) * (np.cos(10*np.pi) -
                                                   np.cos(2*np.pi*x))

        def support(self):
            # 支持区间，返回 (-5, 5)
            return -5, 5

    # 将四个分布类实例化为 dists 列表
    dists = [dist0(), dist1(), dist2(), dist3()]

    # exact mean and variance of the distributions in the list dists
    # dists 列表中分布的精确均值和方差
    mv0 = [0., 4./15.]
    mv1 = [0., 0.01]
    mv2 = [-0.45/np.pi, 2/3*0.5 - 0.45**2/np.pi**2]
    mv3 = [-0.45/np.pi, 0.2 * 250/3 * 0.5 - 0.45**2/np.pi**2]
    mvs = [mv0, mv1, mv2, mv3]

    @pytest.mark.parametrize("dist, mv_ex",
                             zip(dists, mvs))
    def test_basic(self, dist, mv_ex):
        # 测试基本功能函数
        # 创建 NumericalInversePolynomial 类的实例 rng，使用 dist 和随机种子 42
        rng = NumericalInversePolynomial(dist, random_state=42)
        # 检查连续样本函数，传入 rng 实例、dist 和预期的均值和方差 mv_ex
        check_cont_samples(rng, dist, mv_ex)

    @pytest.mark.xslow
    @pytest.mark.parametrize("distname, params", distcont)
    # 定义一个测试方法，用于测试给定分布的反演多项式数值逆方法
    def test_basic_all_scipy_dists(self, distname, params):

        # 非常慢的分布，一些断言由于微小的数值差异而失败。
        # 可以通过更改种子或增加 u_resolution 来避免这些问题。
        very_slow_dists = ['anglit', 'gausshyper', 'kappa4',
                           'ksone', 'kstwo', 'levy_l',
                           'levy_stable', 'studentized_range',
                           'trapezoid', 'triang', 'vonmises']

        # 对于这些分布，PINV 太慢，测试将跳过。
        fail_dists = ['chi2', 'fatiguelife', 'gibrat',
                      'halfgennorm', 'lognorm', 'ncf',
                      'ncx2', 'pareto', 't']

        # 对于这些分布，跳过样本矩与真实矩之间的一致性检查。
        # 由于样本矩的高方差，我们不能期望它们能通过测试。
        skip_sample_moment_check = ['rel_breitwigner']

        # 如果 distname 在非常慢的分布列表中，则跳过测试
        if distname in very_slow_dists:
            pytest.skip(f"PINV too slow for {distname}")

        # 如果 distname 在失败的分布列表中，则跳过测试
        if distname in fail_dists:
            pytest.skip(f"PINV fails for {distname}")

        # 根据 distname 获取分布对象
        dist = (getattr(stats, distname)
                if isinstance(distname, str)
                else distname)

        # 使用给定参数初始化分布对象
        dist = dist(*params)

        # 使用随机种子初始化 NumericalInversePolynomial 对象
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            rng = NumericalInversePolynomial(dist, random_state=42)

        # 如果 distname 在跳过样本矩检查的列表中，则直接返回，不执行后续检查
        if distname in skip_sample_moment_check:
            return

        # 检查连续样本的一致性
        check_cont_samples(rng, dist, [dist.mean(), dist.var()])

    # 使用常见的错误 PDF 进行测试
    @pytest.mark.parametrize("pdf, err, msg", bad_pdfs_common)
    def test_bad_pdf(self, pdf, err, msg):
        # 定义一个空的类 dist
        class dist:
            pass
        
        # 设置 dist 的 pdf 属性
        dist.pdf = pdf

        # 测试是否会引发特定的错误，并匹配相应的消息
        with pytest.raises(err, match=msg):
            NumericalInversePolynomial(dist, domain=[0, 5])

    # 使用常见的错误对数 PDF 进行测试
    @pytest.mark.parametrize("logpdf, err, msg", bad_logpdfs_common)
    def test_bad_logpdf(self, logpdf, err, msg):
        # 定义一个空的类 dist
        class dist:
            pass
        
        # 设置 dist 的 logpdf 属性
        dist.logpdf = logpdf

        # 测试是否会引发特定的错误，并匹配相应的消息
        with pytest.raises(err, match=msg):
            NumericalInversePolynomial(dist, domain=[0, 5])

    # 测试包含无穷和 NaN 的域。需要编写自定义测试，因为不是所有方法都支持无限尾部。
    @pytest.mark.parametrize("domain, err, msg", inf_nan_domains)
    def test_inf_nan_domains(self, domain, err, msg):
        # 测试是否会引发特定的错误，并匹配相应的消息
        with pytest.raises(err, match=msg):
            NumericalInversePolynomial(StandardNormal(), domain=domain)
    u = [
        # 测试分位数为0和1时返回负无穷和正无穷，并检查0到1之间等距点的PPF的正确性。
        np.linspace(0, 1, num=10000),
        # 测试空数组的PPF方法
        [], [[]],
        # 测试NaN和无穷大返回NaN的结果。
        [np.nan], [-np.inf, np.nan, np.inf],
        # 测试标量输入返回标量结果。
        0,
        # 测试包含NaN、大于1和小于0的值以及一些有效值的数组。
        [[np.nan, 0.5, 0.1], [0.2, 0.4, np.inf], [-2, 3, 4]]
    ]

    @pytest.mark.parametrize("u", u)
    def test_ppf(self, u):
        # 创建标准正态分布对象
        dist = StandardNormal()
        # 使用数值逆多项式方法创建随机数生成器对象，设置u_resolution为1e-14
        rng = NumericalInversePolynomial(dist, u_resolution=1e-14)
        # 旧版本的NumPy对NaN进行比较时会抛出RuntimeWarnings
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in greater")
            sup.filter(RuntimeWarning, "invalid value encountered in "
                                       "greater_equal")
            sup.filter(RuntimeWarning, "invalid value encountered in less")
            sup.filter(RuntimeWarning, "invalid value encountered in "
                                       "less_equal")
            # 计算逆累积分布函数的结果
            res = rng.ppf(u)
            # 使用stats.norm计算标准正态分布的逆累积分布函数的期望结果
            expected = stats.norm.ppf(u)
        # 检查计算结果与期望结果的接近程度
        assert_allclose(res, expected, rtol=1e-11, atol=1e-11)
        # 检查结果的形状与期望结果的形状是否一致
        assert res.shape == expected.shape

    x = [np.linspace(-10, 10, num=10000), [], [[]], [np.nan],
         [-np.inf, np.nan, np.inf], 0,
         [[np.nan, 0.5, 0.1], [0.2, 0.4, np.inf], [-np.inf, 3, 4]]]

    @pytest.mark.parametrize("x", x)
    def test_cdf(self, x):
        # 创建标准正态分布对象
        dist = StandardNormal()
        # 使用数值逆多项式方法创建随机数生成器对象，设置u_resolution为1e-14
        rng = NumericalInversePolynomial(dist, u_resolution=1e-14)
        # 旧版本的NumPy对NaN进行比较时会抛出RuntimeWarnings
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in greater")
            sup.filter(RuntimeWarning, "invalid value encountered in "
                                       "greater_equal")
            sup.filter(RuntimeWarning, "invalid value encountered in less")
            sup.filter(RuntimeWarning, "invalid value encountered in "
                                       "less_equal")
            # 计算累积分布函数的结果
            res = rng.cdf(x)
            # 使用stats.norm计算标准正态分布的累积分布函数的期望结果
            expected = stats.norm.cdf(x)
        # 检查计算结果与期望结果的接近程度
        assert_allclose(res, expected, rtol=1e-11, atol=1e-11)
        # 检查结果的形状与期望结果的形状是否一致
        assert res.shape == expected.shape

    @pytest.mark.slow
    def test_u_error(self):
        # 创建标准正态分布对象
        dist = StandardNormal()
        # 使用数值逆多项式方法创建随机数生成器对象，设置u_resolution为1e-10
        rng = NumericalInversePolynomial(dist, u_resolution=1e-10)
        # 计算u_error的最大误差和平均绝对误差
        max_error, mae = rng.u_error()
        # 断言最大误差小于1e-10
        assert max_error < 1e-10
        # 断言平均绝对误差小于等于最大误差
        assert mae <= max_error
        # 使用数值逆多项式方法创建随机数生成器对象，设置u_resolution为1e-14
        rng = NumericalInversePolynomial(dist, u_resolution=1e-14)
        # 计算u_error的最大误差和平均绝对误差
        max_error, mae = rng.u_error()
        # 断言最大误差小于1e-14
        assert max_error < 1e-14
        # 断言平均绝对误差小于等于最大误差
        assert mae <= max_error
    # 定义一组不良的订单参数，包括整数、浮点数、无穷大和非数值
    bad_orders = [1, 4.5, 20, np.inf, np.nan]
    
    # 定义一组不良的u分辨率参数，包括极小值、浮点数、无穷大和非数值
    bad_u_resolution = [1e-20, 1e-1, np.inf, np.nan]

    # 使用pytest的参数化装饰器，对每个不良订单参数进行测试
    @pytest.mark.parametrize("order", bad_orders)
    def test_bad_orders(self, order):
        # 创建标准正态分布对象
        dist = StandardNormal()

        # 设置异常信息
        msg = r"`order` must be an integer in the range \[3, 17\]."
        
        # 使用pytest的断言检查是否抛出特定异常
        with pytest.raises(ValueError, match=msg):
            NumericalInversePolynomial(dist, order=order)

    # 使用pytest的参数化装饰器，对每个不良u分辨率参数进行测试
    @pytest.mark.parametrize("u_resolution", bad_u_resolution)
    def test_bad_u_resolution(self, u_resolution):
        # 设置异常信息
        msg = r"`u_resolution` must be between 1e-15 and 1e-5."
        
        # 使用pytest的断言检查是否抛出特定异常
        with pytest.raises(ValueError, match=msg):
            NumericalInversePolynomial(StandardNormal(),
                                       u_resolution=u_resolution)

    # 测试未指定pdf或logpdf方法时是否抛出异常
    def test_bad_args(self):

        # 定义一个不良的概率分布类，缺少pdf和logpdf方法
        class BadDist:
            def cdf(self, x):
                return stats.norm._cdf(x)

        # 创建BadDist对象
        dist = BadDist()
        
        # 设置异常信息
        msg = r"Either of the methods `pdf` or `logpdf` must be specified"
        
        # 使用pytest的断言检查是否抛出特定异常
        with pytest.raises(ValueError, match=msg):
            rng = NumericalInversePolynomial(dist)

        # 使用标准正态分布对象创建NumericalInversePolynomial对象
        dist = StandardNormal()
        rng = NumericalInversePolynomial(dist)
        
        # 设置异常信息
        msg = r"`sample_size` must be greater than or equal to 1000."
        
        # 使用pytest的断言检查是否抛出特定异常
        with pytest.raises(ValueError, match=msg):
            rng.u_error(10)

        # 定义一个概率分布类，仅包含pdf方法
        class Distribution:
            def pdf(self, x):
                return np.exp(-0.5 * x*x)

        # 创建Distribution对象
        dist = Distribution()
        rng = NumericalInversePolynomial(dist)
        
        # 设置异常信息
        msg = r"Exact CDF required but not found."
        
        # 使用pytest的断言检查是否抛出特定异常
        with pytest.raises(ValueError, match=msg):
            rng.u_error()

    # 测试logpdf和pdf方法的一致性
    def test_logpdf_pdf_consistency(self):
        # 1. 检查PINV是否只能与pdf和logpdf方法一起工作
        # 2. 检查生成的ppf是否相同（在小的容差范围内）

        # 定义一个空的自定义分布类
        class MyDist:
            pass

        # 使用pdf方法创建具有pdf方法的MyDist对象
        dist_pdf = MyDist()
        dist_pdf.pdf = lambda x: math.exp(-x*x/2)
        rng1 = NumericalInversePolynomial(dist_pdf)

        # 使用logpdf方法创建具有logpdf方法的MyDist对象
        dist_logpdf = MyDist()
        dist_logpdf.logpdf = lambda x: -x*x/2
        rng2 = NumericalInversePolynomial(dist_logpdf)

        # 创建一个从1e-5到1-1e-5的等间距序列
        q = np.linspace(1e-5, 1-1e-5, num=100)
        
        # 使用numpy的assert_allclose函数检查rng1.ppf(q)和rng2.ppf(q)是否在容差范围内相等
        assert_allclose(rng1.ppf(q), rng2.ppf(q))
class TestNumericalInverseHermite:
    # 定义两个概率分布类 dist0 和 dist1，分别实现概率密度函数、概率密度函数的导数、累积分布函数和支持域

    #         /  (1 +sin(2 Pi x))/2  if |x| <= 1
    # f(x) = <
    #         \  0        otherwise
    # 参考自 UNU.RAN 测试套件 (文件 t_hinv.c)
    class dist0:
        def pdf(self, x):
            # 返回概率密度函数值
            return 0.5*(1. + np.sin(2.*np.pi*x))

        def dpdf(self, x):
            # 返回概率密度函数的导数值
            return np.pi*np.cos(2.*np.pi*x)

        def cdf(self, x):
            # 返回累积分布函数值
            return (1. + 2.*np.pi*(1 + x) - np.cos(2.*np.pi*x)) / (4.*np.pi)

        def support(self):
            # 返回支持域
            return -1, 1

    #         /  Max(sin(2 Pi x), 0) * Pi/2  if -1 < x < 0.5
    # f(x) = <
    #         \  0        otherwise
    # 参考自 UNU.RAN 测试套件 (文件 t_hinv.c)
    class dist1:
        def pdf(self, x):
            if (x <= -0.5):
                # 当 x <= -0.5 时返回概率密度函数值
                return np.sin((2. * np.pi) * x) * 0.5 * np.pi
            if (x < 0.):
                # 当 x < 0 且 x > -0.5 时返回概率密度函数值
                return 0.
            if (x <= 0.5):
                # 当 x <= 0.5 且 x > 0 时返回概率密度函数值
                return np.sin((2. * np.pi) * x) * 0.5 * np.pi

        def dpdf(self, x):
            if (x <= -0.5):
                # 当 x <= -0.5 时返回概率密度函数的导数值
                return np.cos((2. * np.pi) * x) * np.pi * np.pi
            if (x < 0.):
                # 当 x < 0 且 x > -0.5 时返回概率密度函数的导数值
                return 0.
            if (x <= 0.5):
                # 当 x <= 0.5 且 x > 0 时返回概率密度函数的导数值
                return np.cos((2. * np.pi) * x) * np.pi * np.pi

        def cdf(self, x):
            if (x <= -0.5):
                # 当 x <= -0.5 时返回累积分布函数值
                return 0.25 * (1 - np.cos((2. * np.pi) * x))
            if (x < 0.):
                # 当 x < 0 且 x > -0.5 时返回累积分布函数值
                return 0.5
            if (x <= 0.5):
                # 当 x <= 0.5 且 x > 0 时返回累积分布函数值
                return 0.75 - 0.25 * np.cos((2. * np.pi) * x)

        def support(self):
            # 返回支持域
            return -1, 0.5

    # 初始化两个概率分布对象 dist0 和 dist1 的列表
    dists = [dist0(), dist1()]

    # dists 列表中分布的精确均值和方差
    mv0 = [-1/(2*np.pi), 1/3 - 1/(4*np.pi*np.pi)]
    mv1 = [-1/4, 3/8 - 1/(2*np.pi*np.pi) - 1/16]
    mvs = [mv0, mv1]

    @pytest.mark.parametrize("dist, mv_ex",
                             zip(dists, mvs))
    @pytest.mark.parametrize("order", [3, 5])
    # 测试 NumericalInverseHermite 类在给定分布和阶数下的基本功能
    def test_basic(self, dist, mv_ex, order):
        rng = NumericalInverseHermite(dist, order=order, random_state=42)
        check_cont_samples(rng, dist, mv_ex)

    # 测试包含无穷和 NaN 的域。由于不是所有方法都支持无穷尾部，因此需要编写自定义测试
    @pytest.mark.parametrize("domain, err, msg", inf_nan_domains)
    def test_inf_nan_domains(self, domain, err, msg):
        # 使用 pytest 的断言捕获特定错误和消息
        with pytest.raises(err, match=msg):
            NumericalInverseHermite(StandardNormal(), domain=domain)
    # 定义测试函数，用于测试所有的 SciPy 概率分布
    def basic_test_all_scipy_dists(self, distname, shapes):
        # 慢速分布列表，这些分布测试可能较慢
        slow_dists = {'ksone', 'kstwo', 'levy_stable', 'skewnorm'}
        # 失败的分布列表，这些分布通常因为累积分布函数（CDF）或概率密度函数（PDF）不准确而失败
        fail_dists = {'beta', 'gausshyper', 'geninvgauss', 'ncf', 'nct',
                      'norminvgauss', 'genhyperbolic', 'studentized_range',
                      'vonmises', 'kappa4', 'invgauss', 'wald'}

        # 如果指定的分布在慢速分布列表中，则跳过测试
        if distname in slow_dists:
            pytest.skip("Distribution is too slow")
        # 如果指定的分布在失败的分布列表中，则标记为预期失败，并附上具体原因的文档链接
        if distname in fail_dists:
            pytest.xfail("Fails - usually due to inaccurate CDF/PDF")

        # 设置随机数种子，以确保测试结果的可重复性
        np.random.seed(0)

        # 根据分布名称和参数形状创建分布对象
        dist = getattr(stats, distname)(*shapes)
        # 创建 NumericalInverseHermite 对象，用于数值反演 Hermite 过程
        fni = NumericalInverseHermite(dist)

        # 生成随机样本数据
        x = np.random.rand(10)
        # 计算百分位点函数值的相对误差
        p_tol = np.max(np.abs(dist.ppf(x)-fni.ppf(x))/np.abs(dist.ppf(x)))
        # 计算累积分布函数值的相对误差
        u_tol = np.max(np.abs(dist.cdf(fni.ppf(x)) - x))

        # 断言百分位点函数值的相对误差小于 1e-8
        assert p_tol < 1e-8
        # 断言累积分布函数值的相对误差小于 1e-12
        assert u_tol < 1e-12

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    @pytest.mark.xslow
    @pytest.mark.parametrize(("distname", "shapes"), distcont)
    # 测试函数，针对所有 SciPy 分布的基本测试
    def test_basic_all_scipy_dists(self, distname, shapes):
        # 如果分布名称是 "truncnorm"，则跳过该测试，因为它已单独测试
        # if distname == "truncnorm":
        #     pytest.skip("Tested separately")
        self.basic_test_all_scipy_dists(distname, shapes)

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    # 测试函数，针对 GitHub 问题 #17155 的特定测试
    def test_basic_truncnorm_gh17155(self):
        # 对 "truncnorm" 分布进行基本测试，参数为 (0.1, 2)
        self.basic_test_all_scipy_dists("truncnorm", (0.1, 2))

    # 测试输入验证功能
    def test_input_validation(self):
        # 使用正则表达式匹配字符串 "`order` must be either 1, 3, or 5."，验证是否引发 ValueError 异常
        match = r"`order` must be either 1, 3, or 5."
        with pytest.raises(ValueError, match=match):
            NumericalInverseHermite(StandardNormal(), order=2)

        # 验证对于没有累积分布函数的分布 "norm"，是否引发 ValueError 异常
        match = "`cdf` required but not found"
        with pytest.raises(ValueError, match=match):
            NumericalInverseHermite("norm")

        # 验证对于无法将字符串转换为浮点数的情况，是否引发 ValueError 异常
        match = "could not convert string to float"
        with pytest.raises(ValueError, match=match):
            NumericalInverseHermite(StandardNormal(),
                                    u_resolution='ekki')

    # 定义随机数生成器列表
    rngs = [None, 0, np.random.RandomState(0)]
    # 添加新的随机数生成器对象到列表中，用于类型忽略的标注
    rngs.append(np.random.default_rng(0))  # type: ignore
    # 定义不同尺寸的输入输出元组列表
    sizes = [(None, tuple()), (8, (8,)), ((4, 5, 6), (4, 5, 6))]

    @pytest.mark.parametrize('rng', rngs)
    @pytest.mark.parametrize('size_in, size_out', sizes)
    # 测试随机变量生成的功能
    def test_RVS(self, rng, size_in, size_out):
        # 创建标准正态分布对象
        dist = StandardNormal()
        # 创建 NumericalInverseHermite 对象，用于数值反演 Hermite 过程
        fni = NumericalInverseHermite(dist)

        # 深度复制随机数生成器对象
        rng2 = deepcopy(rng)
        # 生成随机变量样本数据
        rvs = fni.rvs(size=size_in, random_state=rng)
        # 如果指定了输入尺寸，则断言生成的随机变量数据的形状与期望的输出尺寸相同
        if size_in is not None:
            assert rvs.shape == size_out

        # 如果存在第二个随机数生成器对象，则验证其生成的均匀分布样本数据与标准正态分布样本数据的一致性
        if rng2 is not None:
            rng2 = check_random_state(rng2)
            uniform = rng2.uniform(size=size_in)
            rvs2 = stats.norm.ppf(uniform)
            assert_allclose(rvs, rvs2)
    def test_inaccurate_CDF(self):
        # CDF function with inaccurate tail cannot be inverted; see gh-13319
        # https://github.com/scipy/scipy/pull/13319#discussion_r626188955
        # 定义两个形状参数
        shapes = (2.3098496451481823, 0.6268795430096368)
        # 定义匹配模式用于检测警告信息
        match = ("98 : one or more intervals very short; possibly due to "
                 "numerical problems with a pole or very flat tail")

        # 使用默认的容差值，预期会发生运行时警告
        with pytest.warns(RuntimeWarning, match=match):
            # 创建 NumericalInverseHermite 对象，用于处理 beta 分布的反函数
            NumericalInverseHermite(stats.beta(*shapes))

        # 使用较粗的容差值，预期不会产生错误
        NumericalInverseHermite(stats.beta(*shapes), u_resolution=1e-8)

    def test_custom_distribution(self):
        # 创建标准正态分布的 NumericalInverseHermite 对象
        dist1 = StandardNormal()
        fni1 = NumericalInverseHermite(dist1)

        # 创建标准正态分布的 NumericalInverseHermite 对象
        dist2 = stats.norm()
        fni2 = NumericalInverseHermite(dist2)

        # 断言两个对象产生的随机变量值接近
        assert_allclose(fni1.rvs(random_state=0), fni2.rvs(random_state=0))

    # 定义一系列测试用例中的输入 u
    u = [
        # 检查 PPF 方法对于在 0.02 到 0.98 之间均匀分布的点的正确性
        np.linspace(0., 1., num=10000),
        # 测试空数组的 PPF 方法
        [], [[]],
        # 测试包含 NaN 和 Inf 的数组，期望返回 NaN
        [np.nan], [-np.inf, np.nan, np.inf],
        # 测试标量输入是否返回标量输出
        0,
        # 测试包含 NaN、大于 1 和小于 0 的值以及一些有效值的数组
        [[np.nan, 0.5, 0.1], [0.2, 0.4, np.inf], [-2, 3, 4]]
    ]

    @pytest.mark.parametrize("u", u)
    def test_ppf(self, u):
        # 创建标准正态分布的 NumericalInverseHermite 对象
        dist = StandardNormal()
        # 创建用于处理 PPF 的 NumericalInverseHermite 对象
        rng = NumericalInverseHermite(dist, u_resolution=1e-12)
        # 使用 suppress_warnings 上下文管理器来抑制 NaN 相关的运行时警告
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in greater")
            sup.filter(RuntimeWarning, "invalid value encountered in "
                                       "greater_equal")
            sup.filter(RuntimeWarning, "invalid value encountered in less")
            sup.filter(RuntimeWarning, "invalid value encountered in "
                                       "less_equal")
            # 计算 PPF 方法的结果
            res = rng.ppf(u)
            # 使用 scipy 的标准正态分布的 PPF 方法作为预期结果
            expected = stats.norm.ppf(u)
        # 断言计算结果与预期结果的接近程度
        assert_allclose(res, expected, rtol=1e-9, atol=3e-10)
        # 断言计算结果与预期结果的形状一致
        assert res.shape == expected.shape

    @pytest.mark.slow
    def test_u_error(self):
        # 创建标准正态分布的 NumericalInverseHermite 对象
        dist = StandardNormal()
        # 创建用于处理 u 误差的 NumericalInverseHermite 对象
        rng = NumericalInverseHermite(dist, u_resolution=1e-10)
        # 计算 u 误差的最大值和平均绝对误差
        max_error, mae = rng.u_error()
        # 断言最大误差小于给定的阈值
        assert max_error < 1e-10
        # 断言平均绝对误差小于等于最大误差
        assert mae <= max_error
        # 使用 suppress_warnings 上下文管理器来抑制 u 分辨率过小的警告
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            # 使用更小的 u 分辨率创建 NumericalInverseHermite 对象
            rng = NumericalInverseHermite(dist, u_resolution=1e-14)
        # 重新计算 u 误差的最大值和平均绝对误差
        max_error, mae = rng.u_error()
        # 断言最大误差小于给定的阈值
        assert max_error < 1e-14
        # 断言平均绝对误差小于等于最大误差
        assert mae <= max_error
class TestDiscreteGuideTable:
    # 定义一些基础失败的分布名称集合，这些分布存在数值错误的风险
    basic_fail_dists = {
        'nchypergeom_fisher',  # 在尾部存在数值错误
        'nchypergeom_wallenius',  # 在尾部存在数值错误
        'randint'  # 在32位的Ubuntu上会失败
    }

    # 测试当guide_factor大于3时是否会触发警告
    def test_guide_factor_gt3_raises_warning(self):
        pv = [0.1, 0.3, 0.6]
        urng = np.random.default_rng()
        # 断言在创建DiscreteGuideTable对象时会发出RuntimeWarning警告
        with pytest.warns(RuntimeWarning):
            DiscreteGuideTable(pv, random_state=urng, guide_factor=7)

    # 测试当guide_factor为零时是否会触发警告
    def test_guide_factor_zero_raises_warning(self):
        pv = [0.1, 0.3, 0.6]
        urng = np.random.default_rng()
        # 断言在创建DiscreteGuideTable对象时会发出RuntimeWarning警告
        with pytest.warns(RuntimeWarning):
            DiscreteGuideTable(pv, random_state=urng, guide_factor=0)

    # 测试当guide_factor为负数时是否会触发警告
    def test_negative_guide_factor_raises_warning(self):
        # UNU.RAN包装器会自动处理这种情况，但它已经发出了有用的警告
        # 这里我们测试是否会触发警告
        pv = [0.1, 0.3, 0.6]
        urng = np.random.default_rng()
        # 断言在创建DiscreteGuideTable对象时会发出RuntimeWarning警告
        with pytest.warns(RuntimeWarning):
            DiscreteGuideTable(pv, random_state=urng, guide_factor=-1)

    # 使用参数化测试，测试不同的离散分布
    @pytest.mark.parametrize("distname, params", distdiscrete)
    def test_basic(self, distname, params):
        if distname in self.basic_fail_dists:
            msg = ("DGT fails on these probably because of large domains "
                   "and small computation errors in PMF.")
            # 跳过这些分布的测试，因为它们可能由于大域和PMF中的小计算错误而失败
            pytest.skip(msg)

        if not isinstance(distname, str):
            dist = distname
        else:
            dist = getattr(stats, distname)

        dist = dist(*params)
        domain = dist.support()

        if not np.isfinite(domain[1] - domain[0]):
            # DGT仅适用于有限域。因此，跳过具有无限尾部的分布。
            pytest.skip("DGT only works with a finite domain.")

        k = np.arange(domain[0], domain[1]+1)
        pv = dist.pmf(k)
        mv_ex = dist.stats('mv')
        rng = DiscreteGuideTable(dist, random_state=42)
        # 检查离散样本的生成
        check_discr_samples(rng, pv, mv_ex)

    u = [
        # 测试在0到1之间等间距点的PPF方法的正确性。
        np.linspace(0, 1, num=10000),
        # 测试空数组的PPF方法
        [], [[]],
        # 测试如果nans和infs返回nan结果。
        [np.nan], [-np.inf, np.nan, np.inf],
        # 测试如果输入为标量则返回标量。
        0,
        # 测试包含nans、大于1和小于0的值以及一些有效值的数组。
        [[np.nan, 0.5, 0.1], [0.2, 0.4, np.inf], [-2, 3, 4]]
    ]

    # 参数化测试，测试不同的输入u
    @pytest.mark.parametrize('u', u)
    # 定义一个测试方法，用于测试离散指导表的百分位点计算功能
    def test_ppf(self, u):
        # 定义二项分布的参数 n 和 p
        n, p = 4, 0.1
        # 创建二项分布对象
        dist = stats.binom(n, p)
        # 使用指定随机状态创建离散指导表对象
        rng = DiscreteGuideTable(dist, random_state=42)

        # 忽略与 NaN 相关的运行时警告
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in greater")
            sup.filter(RuntimeWarning, "invalid value encountered in greater_equal")
            sup.filter(RuntimeWarning, "invalid value encountered in less")
            sup.filter(RuntimeWarning, "invalid value encountered in less_equal")

            # 计算指定概率 u 的百分位点
            res = rng.ppf(u)
            # 计算预期的二项分布百分位点
            expected = stats.binom.ppf(u, n, p)
        # 断言计算结果的形状与预期相同
        assert_equal(res.shape, expected.shape)
        # 断言计算结果与预期结果相等
        assert_equal(res, expected)

    # 使用参数化测试装饰器标记，测试离散指导表在给定不良参数值时是否能正确抛出 ValueError 异常
    @pytest.mark.parametrize("pv, msg", bad_pv_common)
    def test_bad_pv(self, pv, msg):
        # 使用 pytest 断言检查是否抛出指定的 ValueError 异常和匹配的错误消息
        with pytest.raises(ValueError, match=msg):
            DiscreteGuideTable(pv)

    # 使用参数化测试装饰器标记，测试离散指导表在无限域情况下是否能正确抛出 ValueError 异常
    # DGT 不支持无限尾部，因此当域中包含无限值时，应该抛出错误
    inf_domain = [(-np.inf, np.inf), (np.inf, np.inf), (-np.inf, -np.inf),
                  (0, np.inf), (-np.inf, 0)]

    @pytest.mark.parametrize("domain", inf_domain)
    def test_inf_domain(self, domain):
        # 使用 pytest 断言检查是否抛出指定的 ValueError 异常，且错误消息包含 "must be finite"
        with pytest.raises(ValueError, match=r"must be finite"):
            DiscreteGuideTable(stats.binom(10, 0.2), domain=domain)
class TestSimpleRatioUniforms:
    # pdf with piecewise linear function as transformed density
    # with T = -1/sqrt with shift. Taken from UNU.RAN test suite
    # (from file t_srou.c)
    # 定义一个包含转换密度的分段线性函数的概率密度函数
    # T = -1/sqrt with shift。取自UNU.RAN测试套件中的t_srou.c文件
    class dist:
        def __init__(self, shift):
            # 初始化函数，设置平移值
            self.shift = shift
            self.mode = shift

        def pdf(self, x):
            # 计算概率密度函数的值
            x -= self.shift
            y = 1. / (abs(x) + 1.)
            return 0.5 * y * y

        def cdf(self, x):
            # 计算累积分布函数的值
            x -= self.shift
            if x <= 0.:
                return 0.5 / (1. - x)
            else:
                return 1. - 0.5 / (1. + x)

    dists = [dist(0.), dist(10000.)]

    # exact mean and variance of the distributions in the list dists
    # 分布列表dists中分布的精确均值和方差
    mv1 = [0., np.inf]
    mv2 = [10000., np.inf]
    mvs = [mv1, mv2]

    @pytest.mark.parametrize("dist, mv_ex",
                             zip(dists, mvs))
    def test_basic(self, dist, mv_ex):
        # 使用SimpleRatioUniforms生成随机数生成器rng，设置种子为42
        rng = SimpleRatioUniforms(dist, mode=dist.mode, random_state=42)
        # 检查连续样本生成的结果是否符合期望的分布特性
        check_cont_samples(rng, dist, mv_ex)
        # 使用SimpleRatioUniforms生成随机数生成器rng，设置种子为42，并指定cdf_at_mode参数
        rng = SimpleRatioUniforms(dist, mode=dist.mode,
                                  cdf_at_mode=dist.cdf(dist.mode),
                                  random_state=42)
        # 再次检查连续样本生成的结果是否符合期望的分布特性
        check_cont_samples(rng, dist, mv_ex)

    # test domains with inf + nan in them. need to write a custom test for
    # this because not all methods support infinite tails.
    # 测试包含无穷和NaN的区域。需要为此编写自定义测试，因为并非所有方法都支持无限尾部
    @pytest.mark.parametrize("domain, err, msg", inf_nan_domains)
    def test_inf_nan_domains(self, domain, err, msg):
        # 使用pytest断言来检查是否会抛出预期的异常
        with pytest.raises(err, match=msg):
            SimpleRatioUniforms(StandardNormal(), domain=domain)

    def test_bad_args(self):
        # 测试当pdf_area < 0时是否会抛出ValueError异常
        with pytest.raises(ValueError, match=r"`pdf_area` must be > 0"):
            SimpleRatioUniforms(StandardNormal(), mode=0, pdf_area=-1)


class TestRatioUniforms:
    def test_rv_generation(self):
        # 使用KS检验来检查rvs的分布是否符合期望
        # 正态分布
        f = stats.norm.pdf
        v = np.sqrt(f(np.sqrt(2))) * np.sqrt(2)
        u = np.sqrt(f(0))
        # 使用RatioUniforms生成正态分布的随机数生成器gen，设置种子为12345
        gen = RatioUniforms(f, umax=u, vmin=-v, vmax=v, random_state=12345)
        # 断言KS检验的p值是否大于0.25
        assert_equal(stats.kstest(gen.rvs(2500), 'norm')[1] > 0.25, True)

        # 指数分布
        # 使用RatioUniforms生成指数分布的随机数生成器gen，设置种子为12345
        gen = RatioUniforms(lambda x: np.exp(-x), umax=1,
                            vmin=0, vmax=2*np.exp(-1), random_state=12345)
        # 断言KS检验的p值是否大于0.25
        assert_equal(stats.kstest(gen.rvs(1000), 'expon')[1] > 0.25, True)
    def test_shape(self):
        # 测试返回值的形状是否与 size 参数相关
        f = stats.norm.pdf
        # 计算 f(np.sqrt(2)) 的平方根，然后再开平方
        v = np.sqrt(f(np.sqrt(2))) * np.sqrt(2)
        # 计算 f(0) 的开平方
        u = np.sqrt(f(0))

        # 创建 RatioUniforms 对象 gen1, gen2, gen3，使用相同的随机种子
        gen1 = RatioUniforms(f, umax=u, vmin=-v, vmax=v, random_state=1234)
        gen2 = RatioUniforms(f, umax=u, vmin=-v, vmax=v, random_state=1234)
        gen3 = RatioUniforms(f, umax=u, vmin=-v, vmax=v, random_state=1234)
        # 从 gen1, gen2, gen3 中生成随机变量 r1, r2, r3
        r1, r2, r3 = gen1.rvs(3), gen2.rvs((3,)), gen3.rvs((3, 1))
        # 断言 r1 和 r2 相等
        assert_equal(r1, r2)
        # 断言 r2 和 r3 扁平化后相等
        assert_equal(r2, r3.flatten())
        # 断言 r1 的形状为 (3,)
        assert_equal(r1.shape, (3,))
        # 断言 r3 的形状为 (3, 1)
        assert_equal(r3.shape, (3, 1))

        # 创建 RatioUniforms 对象 gen4, gen5，使用相同的随机种子
        gen4 = RatioUniforms(f, umax=u, vmin=-v, vmax=v, random_state=12)
        gen5 = RatioUniforms(f, umax=u, vmin=-v, vmax=v, random_state=12)
        # 从 gen4, gen5 中生成随机变量 r4, r5
        r4, r5 = gen4.rvs(size=(3, 3, 3)), gen5.rvs(size=27)
        # 断言 r4 扁平化后与 r5 相等
        assert_equal(r4.flatten(), r5)
        # 断言 r4 的形状为 (3, 3, 3)
        assert_equal(r4.shape, (3, 3, 3))

        # 创建 RatioUniforms 对象 gen6, gen7, gen8，使用相同的随机种子
        gen6 = RatioUniforms(f, umax=u, vmin=-v, vmax=v, random_state=1234)
        gen7 = RatioUniforms(f, umax=u, vmin=-v, vmax=v, random_state=1234)
        gen8 = RatioUniforms(f, umax=u, vmin=-v, vmax=v, random_state=1234)
        # 从 gen6, gen7, gen8 中生成随机变量 r6, r7, r8
        r6, r7, r8 = gen6.rvs(), gen7.rvs(1), gen8.rvs((1,))
        # 断言 r6 和 r7 相等
        assert_equal(r6, r7)
        # 断言 r7 和 r8 相等
        assert_equal(r7, r8)

    def test_random_state(self):
        f = stats.norm.pdf
        # 计算 f(np.sqrt(2)) 的平方根，然后再开平方
        v = np.sqrt(f(np.sqrt(2))) * np.sqrt(2)
        # 计算 f(0) 的开平方
        umax = np.sqrt(f(0))
        # 创建 RatioUniforms 对象 gen1，使用指定的随机种子
        gen1 = RatioUniforms(f, umax=umax, vmin=-v, vmax=v, random_state=1234)
        # 生成随机变量 r1
        r1 = gen1.rvs(10)
        # 设置随机种子为 1234
        np.random.seed(1234)
        # 创建 RatioUniforms 对象 gen2，使用默认的随机种子
        gen2 = RatioUniforms(f, umax=umax, vmin=-v, vmax=v)
        # 生成随机变量 r2
        r2 = gen2.rvs(10)
        # 断言 r1 和 r2 相等
        assert_equal(r1, r2)

    def test_exceptions(self):
        f = stats.norm.pdf
        # 需要保证 vmin < vmax
        with assert_raises(ValueError, match="vmin must be smaller than vmax"):
            # 创建 RatioUniforms 对象时应该抛出 ValueError 异常
            RatioUniforms(pdf=f, umax=1, vmin=3, vmax=1)
        with assert_raises(ValueError, match="vmin must be smaller than vmax"):
            # 创建 RatioUniforms 对象时应该抛出 ValueError 异常
            RatioUniforms(pdf=f, umax=1, vmin=1, vmax=1)
        # 需要保证 umax > 0
        with assert_raises(ValueError, match="umax must be positive"):
            # 创建 RatioUniforms 对象时应该抛出 ValueError 异常
            RatioUniforms(pdf=f, umax=-1, vmin=1, vmax=3)
        with assert_raises(ValueError, match="umax must be positive"):
            # 创建 RatioUniforms 对象时应该抛出 ValueError 异常
            RatioUniforms(pdf=f, umax=0, vmin=1, vmax=3)
```