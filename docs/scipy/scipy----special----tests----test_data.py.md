# `D:\src\scipysrc\scipy\scipy\special\tests\test_data.py`

```
import importlib.resources  # 导入用于访问资源的模块

import numpy as np  # 导入 NumPy 库
from numpy.testing import suppress_warnings  # 导入用于抑制警告的函数
import pytest  # 导入 pytest 测试框架

from scipy.special import (  # 导入 SciPy 中的特殊函数
    lpn, lpmn, lpmv, lqn, lqmn, sph_harm, eval_legendre, eval_hermite,
    eval_laguerre, eval_genlaguerre, binom, cbrt, expm1, log1p, zeta,
    jn, jv, jvp, yn, yv, yvp, iv, ivp, kn, kv, kvp,
    gamma, gammaln, gammainc, gammaincc, gammaincinv, gammainccinv, digamma,
    beta, betainc, betaincinv, poch,
    ellipe, ellipeinc, ellipk, ellipkm1, ellipkinc,
    elliprc, elliprd, elliprf, elliprg, elliprj,
    erf, erfc, erfinv, erfcinv, exp1, expi, expn,
    bdtrik, btdtr, btdtri, btdtria, btdtrib, chndtr, gdtr, gdtrc, gdtrix, gdtrib,
    nbdtrik, pdtrik, owens_t,
    mathieu_a, mathieu_b, mathieu_cem, mathieu_sem, mathieu_modcem1,
    mathieu_modsem1, mathieu_modcem2, mathieu_modsem2,
    ellip_harm, ellip_harm_2, spherical_jn, spherical_yn, wright_bessel
)
from scipy.integrate import IntegrationWarning  # 导入积分警告类

from scipy.special._testutils import FuncData  # 导入用于测试的函数数据类


# The npz files are generated, and hence may live in the build dir. We can only
# access them through `importlib.resources`, not an explicit path from `__file__`
_datadir = importlib.resources.files('scipy.special.tests.data')  # 获取数据文件目录路径

_boost_npz = _datadir.joinpath('boost.npz')  # 构建 Boost 数据文件路径对象
with importlib.resources.as_file(_boost_npz) as f:  # 打开 Boost 数据文件
    DATASETS_BOOST = np.load(f)  # 加载 Boost 数据集

_gsl_npz = _datadir.joinpath('gsl.npz')  # 构建 GSL 数据文件路径对象
with importlib.resources.as_file(_gsl_npz) as f:  # 打开 GSL 数据文件
    DATASETS_GSL = np.load(f)  # 加载 GSL 数据集

_local_npz = _datadir.joinpath('local.npz')  # 构建本地数据文件路径对象
with importlib.resources.as_file(_local_npz) as f:  # 打开本地数据文件
    DATASETS_LOCAL = np.load(f)  # 加载本地数据集


def data(func, dataname, *a, **kw):
    kw.setdefault('dataname', dataname)
    return FuncData(func, DATASETS_BOOST[dataname], *a, **kw)  # 返回 Boost 数据集中的函数数据


def data_gsl(func, dataname, *a, **kw):
    kw.setdefault('dataname', dataname)
    return FuncData(func, DATASETS_GSL[dataname], *a, **kw)  # 返回 GSL 数据集中的函数数据


def data_local(func, dataname, *a, **kw):
    kw.setdefault('dataname', dataname)
    return FuncData(func, DATASETS_LOCAL[dataname], *a, **kw)  # 返回本地数据集中的函数数据


def ellipk_(k):
    return ellipk(k*k)  # 计算健全第一类完全椭圆积分


def ellipkinc_(f, k):
    return ellipkinc(f, k*k)  # 计算健全第一类不完全椭圆积分


def ellipe_(k):
    return ellipe(k*k)  # 计算健全第二类完全椭圆积分


def ellipeinc_(f, k):
    return ellipeinc(f, k*k)  # 计算健全第二类不完全椭圆积分


def zeta_(x):
    return zeta(x, 1.)  # 计算 Riemann zeta 函数


def assoc_legendre_p_boost_(nu, mu, x):
    # the boost test data is for integer orders only
    return lpmv(mu, nu.astype(int), x)  # 计算关联 Legendre 函数（使用 Boost 测试数据）


def legendre_p_via_assoc_(nu, x):
    return lpmv(0, nu, x)  # 计算 Legendre 函数（通过关联 Legendre 函数实现）


def lpn_(n, x):
    return lpn(n.astype('l'), x)[0][-1]  # 计算 Laguerre 多项式


def lqn_(n, x):
    return lqn(n.astype('l'), x)[0][-1]  # 计算 Laguerre 函数


def legendre_p_via_lpmn(n, x):
    return lpmn(0, n, x)[0][0,-1]  # 通过 lpmn 计算 Legendre 函数


def legendre_q_via_lqmn(n, x):
    return lqmn(0, n, x)[0][0,-1]  # 通过 lqmn 计算 Legendre 函数


def mathieu_ce_rad(m, q, x):
    return mathieu_cem(m, q, x*180/np.pi)[0]  # 计算 Mathieu 函数的角度版本


def mathieu_se_rad(m, q, x):
    return mathieu_sem(m, q, x*180/np.pi)[0]  # 计算 Mathieu 函数的角度版本


def mathieu_mc1_scaled(m, q, x):
    # GSL follows a different normalization.
    # We follow Abramowitz & Stegun, they apparently something else.
    # 计算 Mathieu 函数，与 GSL 和 Abramowitz & Stegun 的归一化相关
    # 调用 mathieu_modcem1 函数计算结果，返回第一个元素
    return mathieu_modcem1(m, q, x)[0] * np.sqrt(np.pi/2)
def mathieu_ms1_scaled(m, q, x):
    return mathieu_modsem1(m, q, x)[0] * np.sqrt(np.pi/2)
# 调用 mathieu_modsem1 函数计算 Mathieu 函数的第一类偶解，返回其值乘以 sqrt(pi/2)

def mathieu_mc2_scaled(m, q, x):
    return mathieu_modcem2(m, q, x)[0] * np.sqrt(np.pi/2)
# 调用 mathieu_modcem2 函数计算 Mathieu 函数的第二类偶解，返回其值乘以 sqrt(pi/2)

def mathieu_ms2_scaled(m, q, x):
    return mathieu_modsem2(m, q, x)[0] * np.sqrt(np.pi/2)
# 调用 mathieu_modsem2 函数计算 Mathieu 函数的第一类奇解，返回其值乘以 sqrt(pi/2)

def eval_legendre_ld(n, x):
    return eval_legendre(n.astype('l'), x)
# 调用 eval_legendre 函数计算勒让德多项式，将 n 转换为长整型后作为参数传入

def eval_legendre_dd(n, x):
    return eval_legendre(n.astype('d'), x)
# 调用 eval_legendre 函数计算勒让德多项式，将 n 转换为双精度浮点型后作为参数传入

def eval_hermite_ld(n, x):
    return eval_hermite(n.astype('l'), x)
# 调用 eval_hermite 函数计算厄米多项式，将 n 转换为长整型后作为参数传入

def eval_laguerre_ld(n, x):
    return eval_laguerre(n.astype('l'), x)
# 调用 eval_laguerre 函数计算拉盖尔多项式，将 n 转换为长整型后作为参数传入

def eval_laguerre_dd(n, x):
    return eval_laguerre(n.astype('d'), x)
# 调用 eval_laguerre 函数计算拉盖尔多项式，将 n 转换为双精度浮点型后作为参数传入

def eval_genlaguerre_ldd(n, a, x):
    return eval_genlaguerre(n.astype('l'), a, x)
# 调用 eval_genlaguerre 函数计算广义拉盖尔多项式，将 n 转换为长整型后作为参数传入

def eval_genlaguerre_ddd(n, a, x):
    return eval_genlaguerre(n.astype('d'), a, x)
# 调用 eval_genlaguerre 函数计算广义拉盖尔多项式，将 n 转换为双精度浮点型后作为参数传入

def bdtrik_comp(y, n, p):
    return bdtrik(1-y, n, p)
# 调用 bdtrik 函数计算二项分布的累积分布函数的补函数，传入参数 y 的补数

def btdtri_comp(a, b, p):
    return btdtri(a, b, 1-p)
# 调用 btdtri 函数计算贝塔分布的分位函数，传入参数 p 的补数

def btdtria_comp(p, b, x):
    return btdtria(1-p, b, x)
# 调用 btdtria 函数计算贝塔分布的分位函数，传入参数 p 的补数

def btdtrib_comp(a, p, x):
    return btdtrib(a, 1-p, x)
# 调用 btdtrib 函数计算贝塔分布的分位函数，传入参数 p 的补数

def gdtr_(p, x):
    return gdtr(1.0, p, x)
# 调用 gdtr 函数计算 Gamma 分布的累积分布函数，使用参数 p 的补数

def gdtrc_(p, x):
    return gdtrc(1.0, p, x)
# 调用 gdtrc 函数计算 Gamma 分布的补函数，使用参数 p 的补数

def gdtrix_(b, p):
    return gdtrix(1.0, b, p)
# 调用 gdtrix 函数计算 Gamma 分布的分位函数，使用参数 p 的补数

def gdtrix_comp(b, p):
    return gdtrix(1.0, b, 1-p)
# 调用 gdtrix 函数计算 Gamma 分布的分位函数，传入参数 p 的补数

def gdtrib_(p, x):
    return gdtrib(1.0, p, x)
# 调用 gdtrib 函数计算 Gamma 分布的分位函数，使用参数 p 的补数

def gdtrib_comp(p, x):
    return gdtrib(1.0, 1-p, x)
# 调用 gdtrib 函数计算 Gamma 分布的分位函数，传入参数 p 的补数

def nbdtrik_comp(y, n, p):
    return nbdtrik(1-y, n, p)
# 调用 nbdtrik 函数计算负二项分布的累积分布函数的补函数，传入参数 y 的补数

def pdtrik_comp(p, m):
    return pdtrik(1-p, m)
# 调用 pdtrik 函数计算泊松分布的累积分布函数的补函数，传入参数 p 的补数

def poch_(z, m):
    return 1.0 / poch(z, m)
# 调用 poch 函数计算 Pochhammer 符号，返回其倒数

def poch_minus(z, m):
    return 1.0 / poch(z, -m)
# 调用 poch 函数计算 Pochhammer 符号，将 m 取负数后返回其倒数

def spherical_jn_(n, x):
    return spherical_jn(n.astype('l'), x)
# 调用 spherical_jn 函数计算球贝塞尔函数，将 n 转换为长整型后作为参数传入

def spherical_yn_(n, x):
    return spherical_yn(n.astype('l'), x)
# 调用 spherical_yn 函数计算球贝塞尔函数的第二类形式，将 n 转换为长整型后作为参数传入

def sph_harm_(m, n, theta, phi):
    y = sph_harm(m, n, theta, phi)
    return (y.real, y.imag)
# 调用 sph_harm 函数计算球谐函数，返回实部和虚部构成的元组

def cexpm1(x, y):
    z = expm1(x + 1j*y)
    return z.real, z.imag
# 调用 expm1 函数计算 exp(z) - 1，其中 z 是复数 x + i*y，返回其实部和虚部构成的元组

def clog1p(x, y):
    z = log1p(x + 1j*y)
    return z.real, z.imag
# 调用 log1p 函数计算 log(1 + z)，其中 z 是复数 x + i*y，返回其实部和虚部构成的元组


@pytest.mark.parametrize('test', BOOST_TESTS, ids=repr)
def test_boost(test):
    # Filter deprecation warnings of any deprecated functions.
    if test.func in [btdtr, btdtri, btdtri_comp]:
        with pytest.deprecated_call():
            _test_factory(test)
    else:
        _test_factory(test)
# 使用 pytest.mark.parametrize 运行测试 BOOST_TESTS 中的每个测试，根据 test.func 的值过滤部分废弃函数的警告，调用 _test_factory 进行测试
# 定义包含 GSL 测试数据的列表
GSL_TESTS = [
        # 使用 mathieu_a 函数生成 mathieu_ab 数据集的测试数据，传入参数和误差容忍度
        data_gsl(mathieu_a, 'mathieu_ab', (0, 1), 2, rtol=1e-13, atol=1e-13),
        # 使用 mathieu_b 函数生成 mathieu_ab 数据集的测试数据，传入参数和误差容忍度
        data_gsl(mathieu_b, 'mathieu_ab', (0, 1), 3, rtol=1e-13, atol=1e-13),

        # GSL 输出的精度也受限制...
        # 使用 mathieu_ce_rad 函数生成 mathieu_ce_se 数据集的测试数据，传入参数和误差容忍度
        data_gsl(mathieu_ce_rad, 'mathieu_ce_se', (0, 1, 2), 3, rtol=1e-7, atol=1e-13),
        # 使用 mathieu_se_rad 函数生成 mathieu_ce_se 数据集的测试数据，传入参数和误差容忍度
        data_gsl(mathieu_se_rad, 'mathieu_ce_se', (0, 1, 2), 4, rtol=1e-7, atol=1e-13),

        # 使用 mathieu_mc1_scaled 函数生成 mathieu_mc_ms 数据集的测试数据，传入参数和误差容忍度
        data_gsl(mathieu_mc1_scaled, 'mathieu_mc_ms',
                 (0, 1, 2), 3, rtol=1e-7, atol=1e-13),
        # 使用 mathieu_ms1_scaled 函数生成 mathieu_mc_ms 数据集的测试数据，传入参数和误差容忍度
        data_gsl(mathieu_ms1_scaled, 'mathieu_mc_ms',
                 (0, 1, 2), 4, rtol=1e-7, atol=1e-13),

        # 使用 mathieu_mc2_scaled 函数生成 mathieu_mc_ms 数据集的测试数据，传入参数和误差容忍度
        data_gsl(mathieu_mc2_scaled, 'mathieu_mc_ms',
                 (0, 1, 2), 5, rtol=1e-7, atol=1e-13),
        # 使用 mathieu_ms2_scaled 函数生成 mathieu_mc_ms 数据集的测试数据，传入参数和误差容忍度
        data_gsl(mathieu_ms2_scaled, 'mathieu_mc_ms',
                 (0, 1, 2), 6, rtol=1e-7, atol=1e-13),
]


# 使用 GSL_TESTS 列表中的测试数据作为参数化测试，每个测试用例使用其表现的字符串表示作为标识符
@pytest.mark.parametrize('test', GSL_TESTS, ids=repr)
def test_gsl(test):
    # 调用 _test_factory 函数执行测试
    _test_factory(test)


# 定义包含本地测试数据的列表
LOCAL_TESTS = [
    # 使用 ellipkinc 函数生成 ellipkinc_neg_m 数据集的测试数据，传入参数和误差容忍度
    data_local(ellipkinc, 'ellipkinc_neg_m', (0, 1), 2),
    # 使用 ellipkm1 函数生成 ellipkm1 数据集的测试数据，传入参数和误差容忍度
    data_local(ellipkm1, 'ellipkm1', 0, 1),
    # 使用 ellipeinc 函数生成 ellipeinc_neg_m 数据集的测试数据，传入参数和误差容忍度
    data_local(ellipeinc, 'ellipeinc_neg_m', (0, 1), 2),
    # 使用 clog1p 函数生成 log1p_expm1_complex 数据集的测试数据，传入参数和误差容忍度
    data_local(clog1p, 'log1p_expm1_complex', (0,1), (2,3), rtol=1e-14),
    # 使用 cexpm1 函数生成 log1p_expm1_complex 数据集的测试数据，传入参数和误差容忍度
    data_local(cexpm1, 'log1p_expm1_complex', (0,1), (4,5), rtol=1e-14),
    # 使用 gammainc 函数生成 gammainc 数据集的测试数据，传入参数和误差容忍度
    data_local(gammainc, 'gammainc', (0, 1), 2, rtol=1e-12),
    # 使用 gammaincc 函数生成 gammaincc 数据集的测试数据，传入参数和误差容忍度
    data_local(gammaincc, 'gammaincc', (0, 1), 2, rtol=1e-11),
    # 使用 ellip_harm_2 函数生成 ellip 数据集的测试数据，传入参数和误差容忍度
    data_local(ellip_harm_2, 'ellip',(0, 1, 2, 3, 4), 6, rtol=1e-10, atol=1e-13),
    # 使用 ellip_harm 函数生成 ellip 数据集的测试数据，传入参数和误差容忍度
    data_local(ellip_harm, 'ellip',(0, 1, 2, 3, 4), 5, rtol=1e-10, atol=1e-13),
    # 使用 wright_bessel 函数生成 wright_bessel 数据集的测试数据，传入参数和误差容忍度
    data_local(wright_bessel, 'wright_bessel', (0, 1, 2), 3, rtol=1e-11),
]


# 使用 LOCAL_TESTS 列表中的测试数据作为参数化测试，每个测试用例使用其表现的字符串表示作为标识符
@pytest.mark.parametrize('test', LOCAL_TESTS, ids=repr)
def test_local(test):
    # 调用 _test_factory 函数执行测试
    _test_factory(test)


# 定义 _test_factory 函数，接收测试数据和数据类型作为参数，执行测试
def _test_factory(test, dtype=np.float64):
    """Boost test"""
    # 使用 suppress_warnings 上下文管理器，忽略 IntegrationWarning 并输出相关信息
    with suppress_warnings() as sup:
        sup.filter(IntegrationWarning, "The occurrence of roundoff error is detected")
        # 忽略所有 NumPy 的错误
        with np.errstate(all='ignore'):
            # 调用测试对象的 check 方法进行测试，传入指定的数据类型
            test.check(dtype=dtype)
```