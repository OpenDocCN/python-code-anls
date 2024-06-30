# `D:\src\scipysrc\scipy\scipy\special\tests\test_cdflib.py`

```
"""
Test cdflib functions versus mpmath, if available.

The following functions still need tests:

- ncfdtr
- ncfdtri
- ncfdtridfn
- ncfdtridfd
- ncfdtrinc
- nbdtrik
- nbdtrin
- pdtrik
- nctdtr
- nctdtrit
- nctdtridf
- nctdtrinc

"""
# 导入必要的模块
import itertools  # 导入 itertools 模块，用于迭代工具
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.testing import assert_equal, assert_allclose  # 导入 NumPy 测试模块中的断言函数
import pytest  # 导入 PyTest 测试框架

import scipy.special as sp  # 导入 SciPy 中的 special 模块，用于数学特殊函数
from scipy.special._testutils import (  # 导入 SciPy 测试工具中的特定函数和类
    MissingModule, check_version, FuncData)
from scipy.special._mptestutils import (  # 导入 SciPy 的特定测试工具函数和类
    Arg, IntArg, get_args, mpf2float, assert_mpmath_equal)

try:
    import mpmath  # 尝试导入 mpmath 数学库
except ImportError:
    mpmath = MissingModule('mpmath')  # 若导入失败，则创建一个缺失模块的占位符对象


class ProbArg:
    """Generate a set of probabilities on [0, 1]."""

    def __init__(self):
        # Include the endpoints for compatibility with Arg et. al.
        self.a = 0  # 设定概率区间起始值为 0
        self.b = 1  # 设定概率区间结束值为 1

    def values(self, n):
        """Return an array containing approximately n numbers."""
        m = max(1, n//3)
        v1 = np.logspace(-30, np.log10(0.3), m)  # 生成指数分布的数值，填充到数组 v1
        v2 = np.linspace(0.3, 0.7, m + 1, endpoint=False)[1:]  # 生成线性分布的数值，填充到数组 v2
        v3 = 1 - np.logspace(np.log10(0.3), -15, m)  # 生成逆指数分布的数值，填充到数组 v3
        v = np.r_[v1, v2, v3]  # 合并 v1、v2 和 v3 数组，并去重
        return np.unique(v)  # 返回去重后的唯一数值数组


class EndpointFilter:
    def __init__(self, a, b, rtol, atol):
        self.a = a  # 设定区间起始值
        self.b = b  # 设定区间结束值
        self.rtol = rtol  # 相对误差容忍度
        self.atol = atol  # 绝对误差容忍度

    def __call__(self, x):
        mask1 = np.abs(x - self.a) < self.rtol*np.abs(self.a) + self.atol  # 创建条件掩码 mask1，用于检查 x 是否接近于 self.a
        mask2 = np.abs(x - self.b) < self.rtol*np.abs(self.b) + self.atol  # 创建条件掩码 mask2，用于检查 x 是否接近于 self.b
        return np.where(mask1 | mask2, False, True)  # 返回一个布尔数组，表示 x 是否在指定误差范围之外


class _CDFData:
    def __init__(self, spfunc, mpfunc, index, argspec, spfunc_first=True,
                 dps=20, n=5000, rtol=None, atol=None,
                 endpt_rtol=None, endpt_atol=None):
        self.spfunc = spfunc  # 初始化 SciPy 函数
        self.mpfunc = mpfunc  # 初始化 mpmath 函数
        self.index = index  # 函数索引
        self.argspec = argspec  # 参数规范
        self.spfunc_first = spfunc_first  # 是否先使用 SciPy 函数
        self.dps = dps  # mpmath 的小数位数精度
        self.n = n  # 生成数值的数量

        if not isinstance(argspec, list):
            self.endpt_rtol = None
            self.endpt_atol = None
        elif endpt_rtol is not None or endpt_atol is not None:
            if isinstance(endpt_rtol, list):
                self.endpt_rtol = endpt_rtol
            else:
                self.endpt_rtol = [endpt_rtol]*len(self.argspec)
            if isinstance(endpt_atol, list):
                self.endpt_atol = endpt_atol
            else:
                self.endpt_atol = [endpt_atol]*len(self.argspec)
        else:
            self.endpt_rtol = None
            self.endpt_atol = None
    def idmap(self, *args):
        # 如果设置了spfunc_first标志，先调用spfunc进行计算
        if self.spfunc_first:
            res = self.spfunc(*args)
            # 如果结果是NaN，则直接返回NaN
            if np.isnan(res):
                return np.nan
            # 将参数转换为列表，更新其中的索引位置的参数值为计算结果
            args = list(args)
            args[self.index] = res
            # 设置精度为self.dps，使用mpfunc计算结果，取实部并转换为浮点数
            with mpmath.workdps(self.dps):
                res = self.mpfunc(*tuple(args))
                # 虚部是无效的，只取实部进行处理
                res = mpf2float(res.real)
        else:
            # 设置精度为self.dps，使用mpfunc直接计算结果，取实部并转换为浮点数
            with mpmath.workdps(self.dps):
                res = self.mpfunc(*args)
                res = mpf2float(res.real)
            # 将参数转换为列表，更新其中的索引位置的参数值为计算结果，然后调用spfunc进行计算
            args = list(args)
            args[self.index] = res
            res = self.spfunc(*tuple(args))
        return res

    def get_param_filter(self):
        # 如果没有设置endpt_rtol和endpt_atol，则返回None
        if self.endpt_rtol is None and self.endpt_atol is None:
            return None

        # 初始化一个空列表filters来存储EndpointFilter对象
        filters = []
        # 遍历endpt_rtol、endpt_atol和argspec的元组，生成EndpointFilter对象，并添加到filters列表中
        for rtol, atol, spec in zip(self.endpt_rtol, self.endpt_atol, self.argspec):
            if rtol is None and atol is None:
                filters.append(None)
                continue
            elif rtol is None:
                rtol = 0.0
            elif atol is None:
                atol = 0.0

            filters.append(EndpointFilter(spec.a, spec.b, rtol, atol))
        return filters

    def check(self):
        # 生成参数的值
        args = get_args(self.argspec, self.n)
        # 获取参数过滤器
        param_filter = self.get_param_filter()
        # 参数列为所有列的索引，结果列数为参数数组的列数
        param_columns = tuple(range(args.shape[1]))
        result_columns = args.shape[1]
        # 将原始参数数组args与索引位置的参数列合并成一个新的参数数组
        args = np.hstack((args, args[:, self.index].reshape(args.shape[0], 1)))
        # 创建FuncData对象，并调用其check方法进行检查
        FuncData(self.idmap, args,
                 param_columns=param_columns, result_columns=result_columns,
                 rtol=self.rtol, atol=self.atol, vectorized=False,
                 param_filter=param_filter).check()
# 定义一个函数，用于创建并检查一个 _CDFData 对象
def _assert_inverts(*a, **kw):
    d = _CDFData(*a, **kw)  # 创建 _CDFData 对象 d，并传入参数 *a, **kw
    d.check()  # 调用 _CDFData 对象的 check 方法


# 定义一个函数，计算二项分布的累积分布函数
def _binomial_cdf(k, n, p):
    k, n, p = mpmath.mpf(k), mpmath.mpf(n), mpmath.mpf(p)  # 将 k, n, p 转换为 mpmath 的多精度浮点数
    if k <= 0:
        return mpmath.mpf(0)  # 如果 k 小于等于 0，返回累积分布函数值 0
    elif k >= n:
        return mpmath.mpf(1)  # 如果 k 大于等于 n，返回累积分布函数值 1

    onemp = mpmath.fsub(1, p, exact=True)  # 计算 1 - p
    return mpmath.betainc(n - k, k + 1, x2=onemp, regularized=True)  # 使用 beta 不完全函数计算二项分布的累积分布函数


# 定义一个函数，计算 F 分布的累积分布函数
def _f_cdf(dfn, dfd, x):
    if x < 0:
        return mpmath.mpf(0)  # 如果 x 小于 0，返回累积分布函数值 0
    dfn, dfd, x = mpmath.mpf(dfn), mpmath.mpf(dfd), mpmath.mpf(x)  # 将 dfn, dfd, x 转换为 mpmath 的多精度浮点数
    ub = dfn * x / (dfn * x + dfd)  # 计算 F 分布的上界
    res = mpmath.betainc(dfn / 2, dfd / 2, x2=ub, regularized=True)  # 使用 beta 不完全函数计算 F 分布的累积分布函数
    return res  # 返回累积分布函数值


# 定义一个函数，计算学生 t 分布的累积分布函数
def _student_t_cdf(df, t, dps=None):
    if dps is None:
        dps = mpmath.mp.dps  # 如果未指定精度，使用默认精度
    with mpmath.workdps(dps):  # 设置精度
        df, t = mpmath.mpf(df), mpmath.mpf(t)  # 将 df, t 转换为 mpmath 的多精度浮点数
        fac = mpmath.hyp2f1(0.5, 0.5 * (df + 1), 1.5, -t**2 / df)  # 计算超几何函数
        fac *= t * mpmath.gamma(0.5 * (df + 1))  # 乘以 gamma 函数值
        fac /= mpmath.sqrt(mpmath.pi * df) * mpmath.gamma(0.5 * df)  # 除以 sqrt(pi * df) * gamma(0.5 * df)
        return 0.5 + fac  # 返回累积分布函数值


# 定义一个函数，计算非中心卡方分布的概率密度函数
def _noncentral_chi_pdf(t, df, nc):
    res = mpmath.besseli(df / 2 - 1, mpmath.sqrt(nc * t))  # 计算修正贝塞尔函数
    res *= mpmath.exp(-(t + nc) / 2) * (t / nc)**(df / 4 - 1 / 2) / 2  # 计算非中心卡方分布的概率密度函数
    return res  # 返回概率密度函数值


# 定义一个函数，计算非中心卡方分布的累积分布函数
def _noncentral_chi_cdf(x, df, nc, dps=None):
    if dps is None:
        dps = mpmath.mp.dps  # 如果未指定精度，使用默认精度
    x, df, nc = mpmath.mpf(x), mpmath.mpf(df), mpmath.mpf(nc)  # 将 x, df, nc 转换为 mpmath 的多精度浮点数
    with mpmath.workdps(dps):  # 设置精度
        res = mpmath.quad(lambda t: _noncentral_chi_pdf(t, df, nc), [0, x])  # 使用数值积分计算非中心卡方分布的累积分布函数
        return res  # 返回累积分布函数值


# 定义一个函数，计算 Tukey Lambda 分布的分位数
def _tukey_lmbda_quantile(p, lmbda):
    # 当 lmbda 不等于 0 时
    return (p**lmbda - (1 - p)**lmbda) / lmbda  # 返回 Tukey Lambda 分布的分位数
    # 测试函数，用于验证 sp.gdtria 的逆运算是否正确
    def test_gdtria(self):
        # 调用 _assert_inverts 函数进行验证
        _assert_inverts(
            sp.gdtria,  # 待测试的函数对象
            # 使用 mpmath.gammainc 函数作为参考，计算累积分布函数
            lambda a, b, x: mpmath.gammainc(b, b=a*x, regularized=True),
            0,  # 参数索引，指定在 _assert_inverts 中 a 的位置
            [ProbArg(),  # 可能值对象
             Arg(0, 1e3, inclusive_a=False),  # a 的取值范围
             Arg(0, 1e4, inclusive_a=False)],  # x 的取值范围
            rtol=1e-7,  # 相对误差容限
            endpt_atol=[None, 1e-7, 1e-10])  # 端点绝对误差容限数组

    # 测试函数，用于验证 sp.gdtrib 的逆运算是否正确
    def test_gdtrib(self):
        # 使用较小的 a 和 x 值，因为 mpmath 不会在其他情况下收敛
        _assert_inverts(
            sp.gdtrib,  # 待测试的函数对象
            # 使用 mpmath.gammainc 函数作为参考，计算累积分布函数
            lambda a, b, x: mpmath.gammainc(b, b=a*x, regularized=True),
            1,  # 参数索引，指定在 _assert_inverts 中 a 的位置
            [Arg(0, 1e2, inclusive_a=False),  # a 的取值范围
             ProbArg(),  # 可能值对象
             Arg(0, 1e3, inclusive_a=False)],  # x 的取值范围
            rtol=1e-5)  # 相对误差容限

    # 测试函数，用于验证 sp.gdtrix 的逆运算是否正确
    def test_gdtrix(self):
        _assert_inverts(
            sp.gdtrix,  # 待测试的函数对象
            # 使用 mpmath.gammainc 函数作为参考，计算累积分布函数
            lambda a, b, x: mpmath.gammainc(b, b=a*x, regularized=True),
            2,  # 参数索引，指定在 _assert_inverts 中 a 的位置
            [Arg(0, 1e3, inclusive_a=False),  # a 的取值范围
             Arg(0, 1e3, inclusive_a=False),  # b 的取值范围
             ProbArg()],  # 可能值对象
            rtol=1e-7,  # 相对误差容限
            endpt_atol=[None, 1e-7, 1e-10])  # 端点绝对误差容限数组

    # 测试函数，用于验证 sp.nrdtrimn 的逆运算是否正确
    def test_nrdtrimn(self):
        _assert_inverts(
            sp.nrdtrimn,  # 待测试的函数对象
            # 使用 mpmath.ncdf 函数作为参考，计算累积分布函数
            lambda x, y, z: mpmath.ncdf(z, x, y),
            0,  # 参数索引，指定在 _assert_inverts 中 x 的位置
            [ProbArg(),  # 可能值对象
             Arg(0.1, np.inf, inclusive_a=False, inclusive_b=False),  # y 的取值范围
             Arg(-1e10, 1e10)],  # z 的取值范围
            rtol=1e-5)  # 相对误差容限

    # 测试函数，用于验证 sp.nrdtrisd 的逆运算是否正确
    def test_nrdtrisd(self):
        _assert_inverts(
            sp.nrdtrisd,  # 待测试的函数对象
            # 使用 mpmath.ncdf 函数作为参考，计算累积分布函数
            lambda x, y, z: mpmath.ncdf(z, x, y),
            1,  # 参数索引，指定在 _assert_inverts 中 y 的位置
            [Arg(-np.inf, 10, inclusive_a=False, inclusive_b=False),  # x 的取值范围
             ProbArg(),  # 可能值对象
             Arg(10, 1e100)],  # z 的取值范围
            rtol=1e-5)  # 相对误差容限

    # 测试函数，用于验证 sp.stdtr 的逆运算是否正确
    def test_stdtr(self):
        # 断言 sp.stdtr 与 _student_t_cdf 结果相等
        assert_mpmath_equal(
            sp.stdtr,  # 待测试的函数对象
            _student_t_cdf,  # 参考函数对象
            [IntArg(1, 100), Arg(1e-10, np.inf)], rtol=1e-7)  # 参数列表及误差容限

    # 标记为预期失败的测试函数，用于验证 sp.stdtridf 的逆运算是否正确
    @pytest.mark.xfail(run=False)
    def test_stdtridf(self):
        _assert_inverts(
            sp.stdtridf,  # 待测试的函数对象
            _student_t_cdf,  # 参考函数对象
            0, [ProbArg(), Arg()], rtol=1e-7)  # 参数索引及误差容限

    # 测试函数，用于验证 sp.stdtrit 的逆运算是否正确
    def test_stdtrit(self):
        _assert_inverts(
            sp.stdtrit,  # 待测试的函数对象
            _student_t_cdf,  # 参考函数对象
            1,  # 参数索引，指定在 _assert_inverts 中 a 的位置
            [IntArg(1, 100), ProbArg()],  # 参数列表
            rtol=1e-7,  # 相对误差容限
            endpt_atol=[None, 1e-10])  # 端点绝对误差容限数组

    # 测试函数，用于验证 sp.chdtriv 的逆运算是否正确
    def test_chdtriv(self):
        _assert_inverts(
            sp.chdtriv,  # 待测试的函数对象
            # 使用 mpmath.gammainc 函数作为参考，计算累积分布函数
            lambda v, x: mpmath.gammainc(v/2, b=x/2, regularized=True),
            0,  # 参数索引，指定在 _assert_inverts 中 v 的位置
            [ProbArg(),  # 可能值对象
             IntArg(1, 100)],  # x 的取值范围
            rtol=1e-4)  # 相对误差容限

    # 标记为预期失败的测试函数
    @pytest.mark.xfail(run=False)
    # 使用更大的 atol，因为 mpmath 执行数值积分
    _assert_inverts(
        sp.chndtridf,                    # 调用 _assert_inverts 函数，测试 sp.chndtridf 函数
        _noncentral_chi_cdf,             # 使用 _noncentral_chi_cdf 函数作为比较参照
        1,                               # 第一个参数为 1
        [Arg(0, 100, inclusive_a=False),  # 参数列表：0 到 100 之间的 Arg 对象，左边不包括
         ProbArg(),                      # 概率参数对象
         Arg(0, 100, inclusive_a=False)], # 参数列表：0 到 100 之间的 Arg 对象，左边不包括
        n=1000,                           # 执行 1000 次测试
        rtol=1e-4,                        # 相对误差容限为 1e-4
        atol=1e-15                        # 绝对误差容限为 1e-15
    )

@pytest.mark.xfail(run=False)
def test_chndtrinc(self):
    # 使用更大的 atol，因为 mpmath 执行数值积分
    _assert_inverts(
        sp.chndtrinc,                    # 调用 _assert_inverts 函数，测试 sp.chndtrinc 函数
        _noncentral_chi_cdf,             # 使用 _noncentral_chi_cdf 函数作为比较参照
        2,                               # 第一个参数为 2
        [Arg(0, 100, inclusive_a=False),  # 参数列表：0 到 100 之间的 Arg 对象，左边不包括
         IntArg(1, 100),                 # 整数参数对象，范围为 1 到 100
         ProbArg()],                     # 概率参数对象
        n=1000,                           # 执行 1000 次测试
        rtol=1e-4,                        # 相对误差容限为 1e-4
        atol=1e-15                        # 绝对误差容限为 1e-15
    )

def test_chndtrix(self):
    # 使用更大的 atol，因为 mpmath 执行数值积分
    _assert_inverts(
        sp.chndtrix,                     # 调用 _assert_inverts 函数，测试 sp.chndtrix 函数
        _noncentral_chi_cdf,             # 使用 _noncentral_chi_cdf 函数作为比较参照
        0,                               # 第一个参数为 0
        [ProbArg(),                      # 概率参数对象
         IntArg(1, 100),                 # 整数参数对象，范围为 1 到 100
         Arg(0, 100, inclusive_a=False)],# 参数列表：0 到 100 之间的 Arg 对象，左边不包括
        n=1000,                           # 执行 1000 次测试
        rtol=1e-4,                        # 相对误差容限为 1e-4
        atol=1e-15,                       # 绝对误差容限为 1e-15
        endpt_atol=[1e-6, None, None]     # 终点绝对误差容限分别为 1e-6，无，无
    )

def test_tklmbda_zero_shape(self):
    # 当 lmbda = 0 时，CDF 有一个简单的闭式形式
    one = mpmath.mpf(1)
    assert_mpmath_equal(
        lambda x: sp.tklmbda(x, 0),      # 调用 sp.tklmbda 函数，参数为 x 和 0
        lambda x: one/(mpmath.exp(-x) + one),  # 期望值函数表达式
        [Arg()],                         # 参数列表包含 Arg 对象
        rtol=1e-7                        # 相对误差容限为 1e-7
    )

def test_tklmbda_neg_shape(self):
    _assert_inverts(
        sp.tklmbda,                      # 调用 _assert_inverts 函数，测试 sp.tklmbda 函数
        _tukey_lmbda_quantile,           # 使用 _tukey_lmbda_quantile 函数作为比较参照
        0,                               # 第一个参数为 0
        [ProbArg(),                      # 概率参数对象
         Arg(-25, 0, inclusive_b=False)],# 参数列表：-25 到 0 之间的 Arg 对象，右边不包括
        spfunc_first=False,              # 不首先执行 spfunc 参数
        rtol=1e-5,                       # 相对误差容限为 1e-5
        endpt_atol=[1e-9, 1e-5]          # 终点绝对误差容限分别为 1e-9，1e-5
    )

@pytest.mark.xfail(run=False)
def test_tklmbda_pos_shape(self):
    _assert_inverts(
        sp.tklmbda,                      # 调用 _assert_inverts 函数，测试 sp.tklmbda 函数
        _tukey_lmbda_quantile,           # 使用 _tukey_lmbda_quantile 函数作为比较参照
        0,                               # 第一个参数为 0
        [ProbArg(),                      # 概率参数对象
         Arg(0, 100, inclusive_a=False)],# 参数列表：0 到 100 之间的 Arg 对象，左边不包括
        spfunc_first=False,              # 不首先执行 spfunc 参数
        rtol=1e-5                        # 相对误差容限为 1e-5
    )

@pytest.mark.parametrize('lmbda', [0.5, 1.0, 8.0])
def test_tklmbda_lmbda1(self, lmbda):
    bound = 1/lmbda
    assert_equal(
        sp.tklmbda([-bound, bound], lmbda),  # 调用 sp.tklmbda 函数，测试输入范围 [-bound, bound] 和 lmbda
        [0.0, 1.0]                          # 期望返回的结果列表
    )
# 定义一个包含函数名称和参数个数的列表
funcs = [
    ("btdtria", 3),    # 函数名为'btdtria'，参数个数为3
    ("btdtrib", 3),    # 函数名为'btdtrib'，参数个数为3
    ("bdtrik", 3),     # 函数名为'bdtrik'，参数个数为3
    ("bdtrin", 3),     # 函数名为'bdtrin'，参数个数为3
    ("chdtriv", 2),    # 函数名为'chdtriv'，参数个数为2
    ("chndtr", 3),     # 函数名为'chndtr'，参数个数为3
    ("chndtrix", 3),   # 函数名为'chndtrix'，参数个数为3
    ("chndtridf", 3),  # 函数名为'chndtridf'，参数个数为3
    ("chndtrinc", 3),  # 函数名为'chndtrinc'，参数个数为3
    ("fdtridfd", 3),   # 函数名为'fdtridfd'，参数个数为3
    ("ncfdtr", 4),     # 函数名为'ncfdtr'，参数个数为4
    ("ncfdtri", 4),    # 函数名为'ncfdtri'，参数个数为4
    ("ncfdtridfn", 4), # 函数名为'ncfdtridfn'，参数个数为4
    ("ncfdtridfd", 4), # 函数名为'ncfdtridfd'，参数个数为4
    ("ncfdtrinc", 4),  # 函数名为'ncfdtrinc'，参数个数为4
    ("gdtrix", 3),     # 函数名为'gdtrix'，参数个数为3
    ("gdtrib", 3),     # 函数名为'gdtrib'，参数个数为3
    ("gdtria", 3),     # 函数名为'gdtria'，参数个数为3
    ("nbdtrik", 3),    # 函数名为'nbdtrik'，参数个数为3
    ("nbdtrin", 3),    # 函数名为'nbdtrin'，参数个数为3
    ("nrdtrimn", 3),   # 函数名为'nrdtrimn'，参数个数为3
    ("nrdtrisd", 3),   # 函数名为'nrdtrisd'，参数个数为3
    ("pdtrik", 2),     # 函数名为'pdtrik'，参数个数为2
    ("stdtr", 2),      # 函数名为'stdtr'，参数个数为2
    ("stdtrit", 2),    # 函数名为'stdtrit'，参数个数为2
    ("stdtridf", 2),   # 函数名为'stdtridf'，参数个数为2
    ("nctdtr", 3),     # 函数名为'nctdtr'，参数个数为3
    ("nctdtrit", 3),   # 函数名为'nctdtrit'，参数个数为3
    ("nctdtridf", 3),  # 函数名为'nctdtridf'，参数个数为3
    ("nctdtrinc", 3),  # 函数名为'nctdtrinc'，参数个数为3
    ("tklmbda", 2),    # 函数名为'tklmbda'，参数个数为2
]

# 使用 pytest 的参数化功能，为每个函数和参数组合生成测试用例
@pytest.mark.parametrize('func,numargs', funcs, ids=[x[0] for x in funcs])
def test_nonfinite(func, numargs):
    # 使用指定的种子创建随机数生成器
    rng = np.random.default_rng(1701299355559735)
    # 获取 scipy 库中对应名称的函数对象
    func = getattr(sp, func)
    # 生成参数选择列表，包括每个参数的浮点数、NaN、正无穷和负无穷
    args_choices = [(float(x), np.nan, np.inf, -np.inf) for x in rng.random(numargs)]

    # 对每个参数组合进行迭代测试
    for args in itertools.product(*args_choices):
        # 调用函数，传入参数
        res = func(*args)

        # 如果任何一个参数是 NaN，则结果应该是 NaN
        if any(np.isnan(x) for x in args):
            assert_equal(res, np.nan)
        else:
            # 其他情况下，结果应该返回某个值（但不会引发异常或导致挂起）
            pass

# 测试特定的函数 chndtrix，验证问题编号 gh-2158 已解决
def test_chndtrix_gh2158():
    res = sp.chndtrix(0.999999, 2, np.arange(20.)+1e-6)

    # 用 R 生成的期望结果
    res_exp = [27.63103493142305, 35.25728589950540, 39.97396073236288,
               43.88033702110538, 47.35206403482798, 50.54112500166103,
               53.52720257322766, 56.35830042867810, 59.06600769498512,
               61.67243118946381, 64.19376191277179, 66.64228141346548,
               69.02756927200180, 71.35726934749408, 73.63759723904816,
               75.87368842650227, 78.06984431185720, 80.22971052389806,
               82.35640899964173, 84.45263768373256]
    assert_allclose(res, res_exp)

# 标记为在 32 位系统上预期失败的测试用例，指定失败原因
@pytest.mark.xfail_on_32bit("32bit fails due to algorithm threshold")
def test_nctdtr_gh19896():
    # 测试 gh-19896 是否已解决，与 Fortran 代码中的 SciPy 1.11 结果进行比较
    dfarr = [0.98, 9.8, 98, 980]
    pnoncarr = [-3.8, 0.38, 3.8, 38]
    tarr = [0.0015, 0.15, 1.5, 15]
    # 预期的结果数组，包含数值的列表
    resarr = [0.9999276519560749, 0.9999276519560749, 0.9999908831755221,
              0.9999990265452424, 0.3524153312279712, 0.39749697267251416,
              0.7168629634895805, 0.9656246449259646, 7.234804392512006e-05,
              7.234804392512006e-05, 0.03538804607509127, 0.795482701508521,
              0.0, 0.0, 0.0,
              0.011927908523093889, 0.9999276519560749, 0.9999276519560749,
              0.9999997441133123, 1.0, 0.3525155979118013,
              0.4076312014048369, 0.8476794017035086, 0.9999999297116268,
              7.234804392512006e-05, 7.234804392512006e-05, 0.013477443099785824,
              0.9998501512331494, 0.0, 0.0,
              0.0, 6.561112613212572e-07, 0.9999276519560749,
              0.9999276519560749, 0.9999999313496014, 1.0,
              0.3525281784865706, 0.40890253001898014, 0.8664672830017024,
              1.0, 7.234804392512006e-05, 7.234804392512006e-05,
              0.010990889489704836, 1.0, 0.0,
              0.0, 0.0, 0.0,
              0.9999276519560749, 0.9999276519560749, 0.9999999418789304,
              1.0, 0.35252945487817355, 0.40903153246690993,
              0.8684247068528264, 1.0, 7.234804392512006e-05,
              7.234804392512006e-05, 0.01075068918582911, 1.0,
              0.0, 0.0, 0.0, 0.0]
    
    # 实际结果数组，初始化为空列表
    actarr = []
    
    # 使用 itertools 的 product 函数遍历 dfarr, pnoncarr, tarr 的所有组合，并计算 sp.nctdtr 函数的结果，添加到 actarr 中
    for df, p, t in itertools.product(dfarr, pnoncarr, tarr):
        actarr += [sp.nctdtr(df, p, t)]
    
    # 断言，验证 actarr 和 resarr 的接近程度，允许的相对误差（relative tolerance）设为 1e-6，绝对误差（absolute tolerance）设为 0.0
    assert_allclose(actarr, resarr, rtol=1e-6, atol=0.0)
# 定义一个测试函数，用于测试问题编号为 gh-19896 的bug是否已解决。
def test_nctdtrinc_gh19896():
    # 定义三个测试用例的数组，分别是自由度数组、概率数组和 t 值数组
    dfarr = [0.001, 0.98, 9.8, 98, 980, 10000, 98, 9.8, 0.98, 0.001]
    parr = [0.001, 0.1, 0.3, 0.8, 0.999, 0.001, 0.1, 0.3, 0.8, 0.999]
    tarr = [0.0015, 0.15, 1.5, 15, 300, 0.0015, 0.15, 1.5, 15, 300]
    # 预期结果数组，包含预期的计算结果
    desired = [3.090232306168629, 1.406141304556198, 2.014225177124157,
               13.727067118283456, 278.9765683871208, 3.090232306168629,
               1.4312427877936222, 2.014225177124157, 3.712743137978295,
               -3.086951096691082]
    # 调用 sp.nctdtrinc 函数进行实际计算
    actual = sp.nctdtrinc(dfarr, parr, tarr)
    # 使用 assert_allclose 函数断言实际结果和预期结果在一定的相对误差和绝对误差下相等
    assert_allclose(actual, desired, rtol=5e-12, atol=0.0)


# 定义测试函数，用于测试 stdtr 和 stdtrit 函数对负无穷的处理
def test_stdtr_stdtrit_neg_inf():
    # 断言 stdtr 函数在参数为负无穷时返回 NaN 数组
    assert np.all(np.isnan(sp.stdtr(-np.inf, [-np.inf, -1.0, 0.0, 1.0, np.inf])))
    # 断言 stdtrit 函数在参数为负无穷时返回 NaN 数组
    assert np.all(np.isnan(sp.stdtrit(-np.inf, [0.0, 0.25, 0.5, 0.75, 1.0])))


# 定义测试函数，用于测试 bdtrik 和 nbdtrik 函数在处理正无穷时的情况
def test_bdtrik_nbdtrik_inf():
    # 创建一个包含 NaN、负无穷、负数、0、正数、正无穷的 NumPy 数组 y
    y = np.array(
        [np.nan,-np.inf,-10.0, -1.0, 0.0, .00001, .5, 0.9999, 1.0, 10.0, np.inf])
    # 将 y 转换为列向量
    y = y[:,None]
    # 创建一个包含 NaN、负无穷、负数、0、正数、正无穷的概率数组 p
    p = np.atleast_2d(
        [np.nan, -np.inf, -10.0, -1.0, 0.0, .00001, .5, 1.0, np.inf])
    # 断言调用 bdtrik 函数时返回 NaN 数组
    assert np.all(np.isnan(sp.bdtrik(y, np.inf, p)))
    # 断言调用 nbdtrik 函数时返回 NaN 数组
    assert np.all(np.isnan(sp.nbdtrik(y, np.inf, p)))
```