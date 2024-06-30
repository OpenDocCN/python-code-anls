# `D:\src\scipysrc\scipy\scipy\stats\_levy_stable\__init__.py`

```
# 导入警告模块，用于处理警告信息
import warnings
# 导入 functools 模块中的 partial 函数，用于创建偏函数
from functools import partial

# 导入 NumPy 库，并将其重命名为 np，用于数值计算
import numpy as np

# 从 SciPy 库中导入 optimize 模块，用于优化算法
from scipy import optimize
# 从 SciPy 库中导入 integrate 模块，用于数值积分
from scipy import integrate
# 从 SciPy 库中导入 _quadrature 模块中的 _builtincoeffs 函数
from scipy.integrate._quadrature import _builtincoeffs
# 从 SciPy 库中导入 interpolate 模块，用于插值
from scipy import interpolate
# 从 SciPy 库中导入 interpolate 模块中的 RectBivariateSpline 类
from scipy.interpolate import RectBivariateSpline
# 导入 scipy.special 模块，并将其重命名为 sc，用于特殊函数的计算
import scipy.special as sc
# 从 SciPy 库中导入 _lib 模块中的 _lazywhere 函数
from scipy._lib._util import _lazywhere
# 从当前包中的 _distn_infrastructure 模块中导入 rv_continuous 类和 _ShapeInfo 类
from .._distn_infrastructure import rv_continuous, _ShapeInfo
# 从当前包中的 _continuous_distns 模块中导入 uniform、expon、_norm_pdf 和 _norm_cdf 函数
from .._continuous_distns import uniform, expon, _norm_pdf, _norm_cdf
# 从当前包中的 levyst 模块中导入 Nolan 类
from .levyst import Nolan
# 从 SciPy 库中导入 _lib 模块中的 inherit_docstring_from 函数，用于继承文档字符串

# 将 levy_stable、levy_stable_gen 和 pdf_from_cf_with_fft 添加到 __all__ 中，用于模块导出时的限制
__all__ = ["levy_stable", "levy_stable_gen", "pdf_from_cf_with_fft"]

# 稳定分布以不同的参数化方式存在，一些对于数值计算有利，而其他一些则因其位置/尺度感知性而有用。
#
# 在这里我们遵循 [NO] 约定（见下面 levy_stable_gen 的文档字符串中的参考文献）。
#
# S0 / Z0 / x0（也称为 Zoleterav's M）
# S1 / Z1 / x1
#
# 其中 S* 表示参数化方式，Z* 表示标准化版本，其中 gamma = 1，delta = 0，x* 表示变量。
#
# SciPy 的原始 Stable 是一个随机变量生成器。它使用 S1，但不幸的是它不是位置/尺度感知的。

# 默认的数值积分容差
# 用于 piecewise 中的 epsrel，同时在 dni 中用于 epsrel 和 epsabs
# （在 dni 中需要 epsabs，因为加权积分要求 epsabs > 0）
_QUAD_EPS = 1.2e-14


def _Phi_Z0(alpha, t):
    """Calculate Phi_Z0(alpha, t) for stable distributions."""
    return (
        -np.tan(np.pi * alpha / 2) * (np.abs(t) ** (1 - alpha) - 1)
        if alpha != 1
        else -2.0 * np.log(np.abs(t)) / np.pi
    )


def _Phi_Z1(alpha, t):
    """Calculate Phi_Z1(alpha, t) for stable distributions."""
    return (
        np.tan(np.pi * alpha / 2)
        if alpha != 1
        else -2.0 * np.log(np.abs(t)) / np.pi
    )


def _cf(Phi, t, alpha, beta):
    """Characteristic function for stable distributions."""
    return np.exp(
        -(np.abs(t) ** alpha) * (1 - 1j * beta * np.sign(t) * Phi(alpha, t))
    )


# 使用 _Phi_Z0 函数创建一个偏函数 _cf_Z0，用于计算 Z0 稳定分布的特征函数
_cf_Z0 = partial(_cf, _Phi_Z0)
# 使用 _Phi_Z1 函数创建一个偏函数 _cf_Z1，用于计算 Z1 稳定分布的特征函数
_cf_Z1 = partial(_cf, _Phi_Z1)


def _pdf_single_value_cf_integrate(Phi, x, alpha, beta, **kwds):
    """Calculate probability density function using characteristic function
    integration for stable distributions.
    """
    # 获取 kwds 中的 quad_eps 参数，如果不存在则使用默认的 _QUAD_EPS
    quad_eps = kwds.get("quad_eps", _QUAD_EPS)

    def integrand1(t):
        """First part of the integrand for stable distribution PDF calculation."""
        if t == 0:
            return 0
        return np.exp(-(t ** alpha)) * (
            np.cos(beta * (t ** alpha) * Phi(alpha, t))
        )

    def integrand2(t):
        """Second part of the integrand for stable distribution PDF calculation."""
        if t == 0:
            return 0
        return np.exp(-(t ** alpha)) * (
            np.sin(beta * (t ** alpha) * Phi(alpha, t))
        )
    # 在计算积分过程中忽略无效值的错误
    with np.errstate(invalid="ignore"):
        # 使用 integrate.quad 函数计算第一个积分，并返回积分值和其他输出
        int1, *ret1 = integrate.quad(
            integrand1,     # 积分函数的第一个参数：被积函数
            0,              # 积分下限
            np.inf,         # 积分上限为无穷大
            weight="cos",   # 权重函数为余弦
            wvar=x,         # 权重函数的自变量
            limit=1000,     # 积分过程的最大迭代次数
            epsabs=quad_eps,    # 绝对误差的容许值
            epsrel=quad_eps,    # 相对误差的容许值
            full_output=1,  # 返回完整的积分输出信息
        )

        # 使用 integrate.quad 函数计算第二个积分，并返回积分值和其他输出
        int2, *ret2 = integrate.quad(
            integrand2,     # 积分函数的第一个参数：被积函数
            0,              # 积分下限
            np.inf,         # 积分上限为无穷大
            weight="sin",   # 权重函数为正弦
            wvar=x,         # 权重函数的自变量
            limit=1000,     # 积分过程的最大迭代次数
            epsabs=quad_eps,    # 绝对误差的容许值
            epsrel=quad_eps,    # 相对误差的容许值
            full_output=1,  # 返回完整的积分输出信息
        )

    # 返回两个积分值之和除以 π
    return (int1 + int2) / np.pi
_pdf_single_value_cf_integrate_Z0 = partial(
    _pdf_single_value_cf_integrate, _Phi_Z0
)
_pdf_single_value_cf_integrate_Z1 = partial(
    _pdf_single_value_cf_integrate, _Phi_Z1
)

# 定义一个偏函数，用于调用_pdf_single_value_cf_integrate函数，固定第二个参数为_Phi_Z0
# 这里的_Z0表示对应的某种条件或参数设置

def _nolan_round_x_near_zeta(x0, alpha, zeta, x_tol_near_zeta):
    """Round x close to zeta for Nolan's method in [NO]."""
    # "8. When |x0-beta*tan(pi*alpha/2)| is small, the
    # computations of the density and cumulative have numerical problems.
    # The program works around this by setting
    # z = beta*tan(pi*alpha/2) when
    # |z-beta*tan(pi*alpha/2)| < tol(5)*alpha**(1/alpha).
    # (The bound on the right is ad hoc, to get reasonable behavior
    # when alpha is small)."
    # 当|x0-beta*tan(pi*alpha/2)|很小时，密度和累积计算会出现数值问题。
    # 程序通过设置z = beta*tan(pi*alpha/2)，当|z-beta*tan(pi*alpha/2)| < tol(5)*alpha**(1/alpha)时，
    # 来解决这个问题。（右侧的界限是临时的，用于在alpha很小时获得合理的行为）
    if np.abs(x0 - zeta) < x_tol_near_zeta * alpha ** (1 / alpha):
        x0 = zeta
    return x0

# 定义一个函数，用于将x0近似到zeta附近，以用于Nolan的方法中的处理

def _nolan_round_difficult_input(
    x0, alpha, beta, zeta, x_tol_near_zeta, alpha_tol_near_one
):
    """Round difficult input values for Nolan's method in [NO]."""

    # following Nolan's STABLE,
    # "1. When 0 < |alpha-1| < 0.005, the program has numerical problems
    # evaluating the pdf and cdf. The current version of the program sets
    # alpha=1 in these cases. This approximation is not bad in the S0
    # parameterization."
    if np.abs(alpha - 1) < alpha_tol_near_one:
        alpha = 1.0
    # 当0 < |alpha-1| < 0.005时，程序在计算pdf和cdf时会出现数值问题。
    # 程序在这些情况下设置alpha=1。在S0参数化中，这个近似是不错的。

    # "2. When alpha=1 and |beta| < 0.005, the program has numerical
    # problems. The current version sets beta=0."
    # We seem to have addressed this through re-expression of g(theta) here
    # 当alpha=1且|beta| < 0.005时，程序会出现数值问题。当前版本将beta=0。

    x0 = _nolan_round_x_near_zeta(x0, alpha, zeta, x_tol_near_zeta)
    return x0, alpha, beta

# 定义一个函数，用于处理Nolan方法中的困难输入值的舍入处理

def _pdf_single_value_piecewise_Z1(x, alpha, beta, **kwds):
    # convert from Nolan's S_1 (aka S) to S_0 (aka Zolaterev M)
    # parameterization
    zeta = -beta * np.tan(np.pi * alpha / 2.0)
    x0 = x + zeta if alpha != 1 else x
    return _pdf_single_value_piecewise_Z0(x0, alpha, beta, **kwds)

# 定义一个函数，将Nolan的S_1参数化转换为S_0参数化，然后调用_pdf_single_value_piecewise_Z0函数

def _pdf_single_value_piecewise_Z0(x0, alpha, beta, **kwds):

    quad_eps = kwds.get("quad_eps", _QUAD_EPS)
    x_tol_near_zeta = kwds.get("piecewise_x_tol_near_zeta", 0.005)
    alpha_tol_near_one = kwds.get("piecewise_alpha_tol_near_one", 0.005)

    zeta = -beta * np.tan(np.pi * alpha / 2.0)
    x0, alpha, beta = _nolan_round_difficult_input(
        x0, alpha, beta, zeta, x_tol_near_zeta, alpha_tol_near_one
    )

    # some other known distribution pdfs / analytical cases
    # TODO: add more where possible with test coverage,
    # eg https://en.wikipedia.org/wiki/Stable_distribution#Other_analytic_cases
    if alpha == 2.0:
        # normal
        return _norm_pdf(x0 / np.sqrt(2)) / np.sqrt(2)

# 定义一个函数，计算pdf在S_0参数化下的值，处理一些已知的分布和解析情况
    elif alpha == 0.5 and beta == 1.0:
        # levy 分布的概率密度函数
        # 由于 S(1/2, 1, gamma, delta; <x>) ==
        # S(1/2, 1, gamma, gamma + delta; <x0>).
        _x = x0 + 1
        if _x <= 0:
            return 0

        # 计算 Levy 分布的概率密度函数
        return 1 / np.sqrt(2 * np.pi * _x) / _x * np.exp(-1 / (2 * _x))
    elif alpha == 0.5 and beta == 0.0 and x0 != 0:
        # 解析解 [HO] 的概率密度函数
        S, C = sc.fresnel([1 / np.sqrt(2 * np.pi * np.abs(x0))])
        arg = 1 / (4 * np.abs(x0))
        # 计算解析解 [HO] 的概率密度函数
        return (
            np.sin(arg) * (0.5 - S[0]) + np.cos(arg) * (0.5 - C[0])
        ) / np.sqrt(2 * np.pi * np.abs(x0) ** 3)
    elif alpha == 1.0 and beta == 0.0:
        # Cauchy 分布的概率密度函数
        return 1 / (1 + x0 ** 2) / np.pi

    # 对于其它情况，使用分段函数计算概率密度函数
    return _pdf_single_value_piecewise_post_rounding_Z0(
        x0, alpha, beta, quad_eps, x_tol_near_zeta
    )
# 使用Nolan在[NO]中详细描述的方法计算概率密度函数（PDF）。

def _pdf_single_value_piecewise_post_rounding_Z0(x0, alpha, beta, quad_eps,
                                                 x_tol_near_zeta):
    """Calculate pdf using Nolan's methods as detailed in [NO]."""

    # 使用Nolan类初始化_nolan对象，获取zeta、xi、c2和g值
    _nolan = Nolan(alpha, beta, x0)
    zeta = _nolan.zeta
    xi = _nolan.xi
    c2 = _nolan.c2
    g = _nolan.g

    # 如果需要，根据x_tol_near_zeta再次将x0舍入到zeta。由于浮点数差异，zeta可能已重新计算。
    x0 = _nolan_round_x_near_zeta(x0, alpha, zeta, x_tol_near_zeta)

    # 处理Nolan的初始情况逻辑
    if x0 == zeta:
        # 当x0等于zeta时，返回特定公式的值
        return (
            sc.gamma(1 + 1 / alpha)
            * np.cos(xi)
            / np.pi
            / ((1 + zeta ** 2) ** (1 / alpha / 2))
        )
    elif x0 < zeta:
        # 当x0小于zeta时，根据特定逻辑递归调用此函数
        return _pdf_single_value_piecewise_post_rounding_Z0(
            -x0, alpha, -beta, quad_eps, x_tol_near_zeta
        )

    # 根据Nolan的推断，现在可以假设
    #   当alpha != 1时，x0 > zeta
    #   当alpha == 1时，beta != 0

    # 避免在空集上计算积分，使用np.isclose处理macos的浮点数差异
    if np.isclose(-xi, np.pi / 2, rtol=1e-014, atol=1e-014):
        return 0.0

    # 定义积分函数integrand(theta)
    def integrand(theta):
        # 限制任何导致g_1 < 0的数值问题接近theta的极限
        g_1 = g(theta)
        if not np.isfinite(g_1) or g_1 < 0:
            g_1 = 0
        return g_1 * np.exp(-g_1)

    # 使用np.errstate忽略所有错误状态
    with np.errstate(all="ignore"):
        # 使用optimize.bisect查找peak值
        peak = optimize.bisect(
            lambda t: g(t) - 1, -xi, np.pi / 2, xtol=quad_eps
        )

        # integrand可能非常尖锐，因此需要强制QUADPACK在其支持内评估函数
        #

        # 最后，在
        #   ~exp(-100), ~exp(-10), ~exp(-5), ~exp(-1)
        # 添加额外的样本点，以改善QUADPACK对快速下降尾部行为的检测
        # （这个选择相当随意）
        tail_points = [
            optimize.bisect(lambda t: g(t) - exp_height, -xi, np.pi / 2)
            for exp_height in [100, 10, 5]
            # exp_height = 1由peak处理
        ]
        intg_points = [0, peak] + tail_points
        # 使用integrate.quad计算积分，返回intg和其他结果
        intg, *ret = integrate.quad(
            integrand,
            -xi,
            np.pi / 2,
            points=intg_points,
            limit=100,
            epsrel=quad_eps,
            epsabs=0,
            full_output=1,
        )

    # 返回结果c2乘以积分intg
    return c2 * intg


# 将x从Nolan的S_1（也称为S）转换为S_0（也称为Zolaterev M）参数化
def _cdf_single_value_piecewise_Z1(x, alpha, beta, **kwds):
    zeta = -beta * np.tan(np.pi * alpha / 2.0)
    x0 = x + zeta if alpha != 1 else x

    # 调用_cdf_single_value_piecewise_Z0函数，返回其结果
    return _cdf_single_value_piecewise_Z0(x0, alpha, beta, **kwds)


# 仅设置_quad_eps和piecewise_x_tol_near_zeta关键字参数的_pdf_single_value_piecewise_post_rounding_Z0函数的包装器
def _cdf_single_value_piecewise_Z0(x0, alpha, beta, **kwds):
    quad_eps = kwds.get("quad_eps", _QUAD_EPS)
    x_tol_near_zeta = kwds.get("piecewise_x_tol_near_zeta", 0.005)
    # 从关键字参数中获取 "piecewise_alpha_tol_near_one" 的值，默认为 0.005
    alpha_tol_near_one = kwds.get("piecewise_alpha_tol_near_one", 0.005)

    # 计算 zeta 值，根据给定的公式 zeta = -beta * tan(pi * alpha / 2.0)
    zeta = -beta * np.tan(np.pi * alpha / 2.0)
    
    # 调用函数 _nolan_round_difficult_input 处理输入参数 x0, alpha, beta, zeta, x_tol_near_zeta, alpha_tol_near_one
    x0, alpha, beta = _nolan_round_difficult_input(
        x0, alpha, beta, zeta, x_tol_near_zeta, alpha_tol_near_one
    )

    # 根据 alpha 的值判断分布类型并返回对应的累积分布函数值
    if alpha == 2.0:
        # 如果 alpha 等于 2.0，代表正态分布，返回正态分布的累积分布函数值
        return _norm_cdf(x0 / np.sqrt(2))
    elif alpha == 0.5 and beta == 1.0:
        # 如果 alpha 等于 0.5，beta 等于 1.0，代表李维分布，根据李维分布的特定公式计算返回值
        _x = x0 + 1
        if _x <= 0:
            return 0
        return sc.erfc(np.sqrt(0.5 / _x))
    elif alpha == 1.0 and beta == 0.0:
        # 如果 alpha 等于 1.0，beta 等于 0.0，代表柯西分布，返回柯西分布的累积分布函数值
        return 0.5 + np.arctan(x0) / np.pi

    # 如果不是以上特定分布类型，则调用函数 _cdf_single_value_piecewise_post_rounding_Z0 计算分段后的累积分布函数值并返回
    return _cdf_single_value_piecewise_post_rounding_Z0(
        x0, alpha, beta, quad_eps, x_tol_near_zeta
    )
# 使用Nolan在文档[NO]中详述的方法计算累积分布函数（cdf）。
def _cdf_single_value_piecewise_post_rounding_Z0(x0, alpha, beta, quad_eps,
                                                 x_tol_near_zeta):
    """Calculate cdf using Nolan's methods as detailed in [NO]."""
    # 创建Nolan对象，用于处理参数alpha、beta和x0
    _nolan = Nolan(alpha, beta, x0)
    # 获取Nolan对象中计算得到的zeta值
    zeta = _nolan.zeta
    # 获取Nolan对象中计算得到的xi值
    xi = _nolan.xi
    # 获取Nolan对象中计算得到的c1值
    c1 = _nolan.c1
    # 获取Nolan对象中计算得到的c3值
    c3 = _nolan.c3
    # 获取Nolan对象中计算得到的g函数
    g = _nolan.g

    # 如果需要，将x0舍入到zeta附近，因为zeta可能由于浮点数差异而重新计算。
    # 参考：https://github.com/scipy/scipy/pull/18133
    x0 = _nolan_round_x_near_zeta(x0, alpha, zeta, x_tol_near_zeta)

    # 处理Nolan的初始情况逻辑
    if (alpha == 1 and beta < 0) or x0 < zeta:
        # 注意：Nolan的论文在这里有一个错误！
        # 他声明 F(x) = 1 - F(x, alpha, -beta)，但这显然是不正确的，因为在这种情况下 F(-infty) 应为 1.0
        # 实际上，在 alpha != 1 且 x0 < zeta 的情况下是正确的。
        return 1 - _cdf_single_value_piecewise_post_rounding_Z0(
            -x0, alpha, -beta, quad_eps, x_tol_near_zeta
        )
    elif x0 == zeta:
        # 当x0等于zeta时，返回特定的计算结果
        return 0.5 - xi / np.pi

    # 根据Nolan的方法，我们可以假设以下情况
    #   当 alpha != 1 时，x0 > zeta
    #   当 alpha == 1 时，beta > 0

    # 避免在空集上计算积分
    # 使用 isclose，因为macOS存在浮点数差异
    if np.isclose(-xi, np.pi / 2, rtol=1e-014, atol=1e-014):
        return c1

    # 定义积分的被积函数
    def integrand(theta):
        g_1 = g(theta)
        return np.exp(-g_1)

    with np.errstate(all="ignore"):
        # 在必要时缩小支持区间
        left_support = -xi
        right_support = np.pi / 2
        if alpha > 1:
            # 当 alpha > 1 时，被积函数在区间内是单调递增的
            if integrand(-xi) != 0.0:
                res = optimize.minimize(
                    integrand,
                    (-xi,),
                    method="L-BFGS-B",
                    bounds=[(-xi, np.pi / 2)],
                )
                left_support = res.x[0]
        else:
            # 当 alpha <= 1 时，被积函数在区间内是单调递减的
            if integrand(np.pi / 2) != 0.0:
                res = optimize.minimize(
                    integrand,
                    (np.pi / 2,),
                    method="L-BFGS-B",
                    bounds=[(-xi, np.pi / 2)],
                )
                right_support = res.x[0]

        # 执行数值积分
        intg, *ret = integrate.quad(
            integrand,
            left_support,
            right_support,
            points=[left_support, right_support],
            limit=100,
            epsrel=quad_eps,
            epsabs=0,
            full_output=1,
        )

    # 返回最终的cdf计算结果
    return c1 + c3 * intg


# 使用Nolan在文档[NO]中详述的方法模拟随机变量
def _rvs_Z1(alpha, beta, size=None, random_state=None):
    """Simulate random variables using Nolan's methods as detailed in [NO].
    """
    # 定义函数 alpha1func，计算 alpha 等于 1 时的结果
    def alpha1func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
        return (
            2
            / np.pi
            * (
                (np.pi / 2 + bTH) * tanTH
                - beta * np.log((np.pi / 2 * W * cosTH) / (np.pi / 2 + bTH))
            )
        )

    # 定义函数 beta0func，计算 beta 等于 0 时的结果
    def beta0func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
        return (
            W
            / (cosTH / np.tan(aTH) + np.sin(TH))
            * ((np.cos(aTH) + np.sin(aTH) * tanTH) / W) ** (1.0 / alpha)
        )

    # 定义函数 otherwise，处理 alpha 不等于 1 或 beta 不等于 0 的情况
    def otherwise(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
        # alpha 不等于 1 且 beta 不等于 0
        val0 = beta * np.tan(np.pi * alpha / 2)
        th0 = np.arctan(val0) / alpha
        val3 = W / (cosTH / np.tan(alpha * (th0 + TH)) + np.sin(TH))
        res3 = val3 * (
            (
                np.cos(aTH)
                + np.sin(aTH) * tanTH
                - val0 * (np.sin(aTH) - np.cos(aTH) * tanTH)
            )
            / W
        ) ** (1.0 / alpha)
        return res3

    # 定义函数 alphanot1func，根据 alpha 和 beta 的值选择对应的函数进行计算
    def alphanot1func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
        res = _lazywhere(
            beta == 0,
            (alpha, beta, TH, aTH, bTH, cosTH, tanTH, W),
            beta0func,
            f2=otherwise,
        )
        return res

    # 将 alpha、beta、TH、W 等变量广播到相同的大小
    alpha = np.broadcast_to(alpha, size)
    beta = np.broadcast_to(beta, size)
    # 生成均匀分布的随机数 TH
    TH = uniform.rvs(
        loc=-np.pi / 2.0, scale=np.pi, size=size, random_state=random_state
    )
    # 生成指数分布的随机数 W
    W = expon.rvs(size=size, random_state=random_state)
    # 计算 aTH、bTH、cosTH 和 tanTH
    aTH = alpha * TH
    bTH = beta * TH
    cosTH = np.cos(TH)
    tanTH = np.tan(TH)
    # 根据 alpha 的值选择相应的函数进行计算
    res = _lazywhere(
        alpha == 1,
        (alpha, beta, TH, aTH, bTH, cosTH, tanTH, W),
        alpha1func,
        f2=alphanot1func,
    )
    # 返回计算结果
    return res
def _fitstart_S0(data):
    # 调用 _fitstart_S1 函数获取 alpha, beta, delta1, gamma 参数
    alpha, beta, delta1, gamma = _fitstart_S1(data)

    # 根据 S1 参数化到 S0 参数化的转换公式，详见 [NO] 文章
    # 注意只有 delta 发生变化
    if alpha != 1:
        # 当 alpha 不等于 1 时，根据公式计算 delta0
        delta0 = delta1 + beta * gamma * np.tan(np.pi * alpha / 2.0)
    else:
        # 当 alpha 等于 1 时，根据公式计算 delta0
        delta0 = delta1 + 2 * beta * gamma * np.log(gamma) / np.pi

    # 返回计算得到的参数 alpha, beta, delta0, gamma
    return alpha, beta, delta0, gamma


def _fitstart_S1(data):
    # 使用 McCullock 1986 方法 - 稳定分布参数的简单一致估计
    # fmt: off
    # 表格 III 和 IV
    nu_alpha_range = [2.439, 2.5, 2.6, 2.7, 2.8, 3, 3.2, 3.5, 4,
                      5, 6, 8, 10, 15, 25]
    nu_beta_range = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1]

    # 表格 III - alpha = psi_1(nu_alpha, nu_beta)
    alpha_table = np.array([
        [2.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.000],
        [1.916, 1.924, 1.924, 1.924, 1.924, 1.924, 1.924],
        [1.808, 1.813, 1.829, 1.829, 1.829, 1.829, 1.829],
        [1.729, 1.730, 1.737, 1.745, 1.745, 1.745, 1.745],
        [1.664, 1.663, 1.663, 1.668, 1.676, 1.676, 1.676],
        [1.563, 1.560, 1.553, 1.548, 1.547, 1.547, 1.547],
        [1.484, 1.480, 1.471, 1.460, 1.448, 1.438, 1.438],
        [1.391, 1.386, 1.378, 1.364, 1.337, 1.318, 1.318],
        [1.279, 1.273, 1.266, 1.250, 1.210, 1.184, 1.150],
        [1.128, 1.121, 1.114, 1.101, 1.067, 1.027, 0.973],
        [1.029, 1.021, 1.014, 1.004, 0.974, 0.935, 0.874],
        [0.896, 0.892, 0.884, 0.883, 0.855, 0.823, 0.769],
        [0.818, 0.812, 0.806, 0.801, 0.780, 0.756, 0.691],
        [0.698, 0.695, 0.692, 0.689, 0.676, 0.656, 0.597],
        [0.593, 0.590, 0.588, 0.586, 0.579, 0.563, 0.513]]).T
    # 转置是因为使用 `RectBivariateSpline` 进行插值时，`nu_beta` 作为 `x`，`nu_alpha` 作为 `y`

    # 表格 IV - beta = psi_2(nu_alpha, nu_beta)
    beta_table = np.array([
        [0, 2.160, 1.000, 1.000, 1.000, 1.000, 1.000],
        [0, 1.592, 3.390, 1.000, 1.000, 1.000, 1.000],
        [0, 0.759, 1.800, 1.000, 1.000, 1.000, 1.000],
        [0, 0.482, 1.048, 1.694, 1.000, 1.000, 1.000],
        [0, 0.360, 0.760, 1.232, 2.229, 1.000, 1.000],
        [0, 0.253, 0.518, 0.823, 1.575, 1.000, 1.000],
        [0, 0.203, 0.410, 0.632, 1.244, 1.906, 1.000],
        [0, 0.165, 0.332, 0.499, 0.943, 1.560, 1.000],
        [0, 0.136, 0.271, 0.404, 0.689, 1.230, 2.195],
        [0, 0.109, 0.216, 0.323, 0.539, 0.827, 1.917],
        [0, 0.096, 0.190, 0.284, 0.472, 0.693, 1.759],
        [0, 0.082, 0.163, 0.243, 0.412, 0.601, 1.596],
        [0, 0.074, 0.147, 0.220, 0.377, 0.546, 1.482],
        [0, 0.064, 0.128, 0.191, 0.330, 0.478, 1.362],
        [0, 0.056, 0.112, 0.167, 0.285, 0.428, 1.274]]).T

    # 表格 V 和 VII
    # 这些表格按照递减的 `alpha_range` 排序，因此需要根据 `RectBivariateSpline` 的要求进行反转。
    # 定义 alpha 的范围，逆序排列
    alpha_range = [2, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1,
                   1, 0.9, 0.8, 0.7, 0.6, 0.5][::-1]
    
    # 定义 beta 的范围
    beta_range = [0, 0.25, 0.5, 0.75, 1]

    # 表 V - 根据 alpha 和 beta 计算 nu_c = psi_3(alpha, beta) 的表格
    nu_c_table = np.array([
        [1.908, 1.908, 1.908, 1.908, 1.908],
        [1.914, 1.915, 1.916, 1.918, 1.921],
        [1.921, 1.922, 1.927, 1.936, 1.947],
        [1.927, 1.930, 1.943, 1.961, 1.987],
        [1.933, 1.940, 1.962, 1.997, 2.043],
        [1.939, 1.952, 1.988, 2.045, 2.116],
        [1.946, 1.967, 2.022, 2.106, 2.211],
        [1.955, 1.984, 2.067, 2.188, 2.333],
        [1.965, 2.007, 2.125, 2.294, 2.491],
        [1.980, 2.040, 2.205, 2.435, 2.696],
        [2.000, 2.085, 2.311, 2.624, 2.973],
        [2.040, 2.149, 2.461, 2.886, 3.356],
        [2.098, 2.244, 2.676, 3.265, 3.912],
        [2.189, 2.392, 3.004, 3.844, 4.775],
        [2.337, 2.634, 3.542, 4.808, 6.247],
        [2.588, 3.073, 4.534, 6.636, 9.144]])[::-1].T
    # 转置是因为使用 `RectBivariateSpline` 进行插值时，`beta` 作为 `x` 轴，`alpha` 作为 `y` 轴

    # 表 VII - 根据 alpha 和 beta 计算 nu_zeta = psi_5(alpha, beta) 的表格
    nu_zeta_table = np.array([
        [0, 0.000, 0.000, 0.000, 0.000],
        [0, -0.017, -0.032, -0.049, -0.064],
        [0, -0.030, -0.061, -0.092, -0.123],
        [0, -0.043, -0.088, -0.132, -0.179],
        [0, -0.056, -0.111, -0.170, -0.232],
        [0, -0.066, -0.134, -0.206, -0.283],
        [0, -0.075, -0.154, -0.241, -0.335],
        [0, -0.084, -0.173, -0.276, -0.390],
        [0, -0.090, -0.192, -0.310, -0.447],
        [0, -0.095, -0.208, -0.346, -0.508],
        [0, -0.098, -0.223, -0.380, -0.576],
        [0, -0.099, -0.237, -0.424, -0.652],
        [0, -0.096, -0.250, -0.469, -0.742],
        [0, -0.089, -0.262, -0.520, -0.853],
        [0, -0.078, -0.272, -0.581, -0.997],
        [0, -0.061, -0.279, -0.659, -1.198]])[::-1].T
    # fmt: on

    # 使用 RectBivariateSpline 创建 psi_1 插值函数，根据 nu_beta_range 和 nu_alpha_range
    psi_1 = RectBivariateSpline(nu_beta_range, nu_alpha_range,
                                alpha_table, kx=1, ky=1, s=0)

    # 定义 psi_1_1 函数，用于返回 psi_1 的值，当 nu_beta > 0 时，返回正值，否则返回负值
    def psi_1_1(nu_beta, nu_alpha):
        return psi_1(nu_beta, nu_alpha) \
            if nu_beta > 0 else psi_1(-nu_beta, nu_alpha)

    # 使用 RectBivariateSpline 创建 psi_2 插值函数，根据 nu_beta_range 和 nu_alpha_range
    psi_2 = RectBivariateSpline(nu_beta_range, nu_alpha_range,
                                beta_table, kx=1, ky=1, s=0)

    # 定义 psi_2_1 函数，用于返回 psi_2 的值，当 nu_beta > 0 时，返回正值，否则返回负值
    def psi_2_1(nu_beta, nu_alpha):
        return psi_2(nu_beta, nu_alpha) \
            if nu_beta > 0 else -psi_2(-nu_beta, nu_alpha)

    # 使用 RectBivariateSpline 创建 phi_3 插值函数，根据 beta_range 和 alpha_range
    phi_3 = RectBivariateSpline(beta_range, alpha_range, nu_c_table,
                                kx=1, ky=1, s=0)

    # 定义 phi_3_1 函数，用于返回 phi_3 的值，当 beta > 0 时，返回正值，否则返回负值
    def phi_3_1(beta, alpha):
        return phi_3(beta, alpha) if beta > 0 else phi_3(-beta, alpha)

    # 使用 RectBivariateSpline 创建 phi_5 插值函数，根据 beta_range 和 alpha_range
    phi_5 = RectBivariateSpline(beta_range, alpha_range, nu_zeta_table,
                                kx=1, ky=1, s=0)

    # 定义 phi_5_1 函数，用于返回 phi_5 的值，当 beta > 0 时，返回正值，否则返回负值
    def phi_5_1(beta, alpha):
        return phi_5(beta, alpha) if beta > 0 else -phi_5(-beta, alpha)

    # 计算数据 data 的分位数
    p05 = np.percentile(data, 5)
    p50 = np.percentile(data, 50)
    p95 = np.percentile(data, 95)
   `
    # 计算数据的 25% 分位数
    p25 = np.percentile(data, 25)
    # 计算数据的 75% 分位数
    p75 = np.percentile(data, 75)

    # 计算 nu_alpha，根据 p95 和 p05 与 p75 和 p25 的差值
    nu_alpha = (p95 - p05) / (p75 - p25)
    # 计算 nu_beta，根据 p95、p05 和 p50 的值
    nu_beta = (p95 + p05 - 2 * p50) / (p95 - p05)

    # 判断 nu_alpha 是否大于等于 2.439
    if nu_alpha >= 2.439:
        # 获取浮点数的最小正数值
        eps = np.finfo(float).eps
        # 使用 psi_1_1 函数计算 alpha，并限制在 eps 到 2.0 之间
        alpha = np.clip(psi_1_1(nu_beta, nu_alpha)[0, 0], eps, 2.)
        # 使用 psi_2_1 函数计算 beta，并限制在 -1.0 到 1.0 之间
        beta = np.clip(psi_2_1(nu_beta, nu_alpha)[0, 0], -1.0, 1.0)
    else:
        # 如果 nu_alpha 小于 2.439，直接设定 alpha 为 2.0
        alpha = 2.0
        # 根据 nu_beta 的符号设定 beta
        beta = np.sign(nu_beta)
    # 计算常数 c，使用 phi_3_1 函数，基于 beta 和 alpha 的值
    c = (p75 - p25) / phi_3_1(beta, alpha)[0, 0]
    # 计算 zeta，使用 phi_5_1 函数，基于 beta 和 alpha 的值
    zeta = p50 + c * phi_5_1(beta, alpha)[0, 0]
    # 计算 delta，根据 alpha 的值确定不同的公式
    delta = zeta - beta * c * np.tan(np.pi * alpha / 2.) if alpha != 1. else zeta

    # 返回 alpha、beta、delta 和 c 的值
    return (alpha, beta, delta, c)
# 定义一个 Levy 稳定分布的连续随机变量类，继承自 rv_continuous 类
class levy_stable_gen(rv_continuous):
    r"""A Levy-stable continuous random variable.

    %(before_notes)s

    See Also
    --------
    levy, levy_l, cauchy, norm

    Notes
    -----
    The distribution for `levy_stable` has characteristic function:

    .. math::

        \varphi(t, \alpha, \beta, c, \mu) =
        e^{it\mu -|ct|^{\alpha}(1-i\beta\operatorname{sign}(t)\Phi(\alpha, t))}

    where two different parameterizations are supported. The first :math:`S_1`:

    .. math::

        \Phi = \begin{cases}
                \tan \left({\frac {\pi \alpha }{2}}\right)&\alpha \neq 1\\
                -{\frac {2}{\pi }}\log |t|&\alpha =1
                \end{cases}

    The second :math:`S_0`:

    .. math::

        \Phi = \begin{cases}
                -\tan \left({\frac {\pi \alpha }{2}}\right)(|ct|^{1-\alpha}-1)
                &\alpha \neq 1\\
                -{\frac {2}{\pi }}\log |ct|&\alpha =1
                \end{cases}

    The probability density function for `levy_stable` is:

    .. math::

        f(x) = \frac{1}{2\pi}\int_{-\infty}^\infty \varphi(t)e^{-ixt}\,dt

    where :math:`-\infty < t < \infty`. This integral does not have a known
    closed form.

    `levy_stable` generalizes several distributions.  Where possible, they
    should be used instead.  Specifically, when the shape parameters
    assume the values in the table below, the corresponding equivalent
    distribution should be used.

    =========  ========  ===========
    ``alpha``  ``beta``   Equivalent
    =========  ========  ===========
     1/2       -1        `levy_l`
     1/2       1         `levy`
     1         0         `cauchy`
     2         any       `norm` (with ``scale=sqrt(2)``)
    =========  ========  ===========

    Evaluation of the pdf uses Nolan's piecewise integration approach with the
    Zolotarev :math:`M` parameterization by default. There is also the option
    to use direct numerical integration of the standard parameterization of the
    characteristic function or to evaluate by taking the FFT of the
    characteristic function.

    The default method can changed by setting the class variable
    ``levy_stable.pdf_default_method`` to one of 'piecewise' for Nolan's
    approach, 'dni' for direct numerical integration, or 'fft-simpson' for the
    FFT based approach. For the sake of backwards compatibility, the methods
    'best' and 'zolotarev' are equivalent to 'piecewise' and the method
    'quadrature' is equivalent to 'dni'.

    The parameterization can be changed  by setting the class variable
    ``levy_stable.parameterization`` to either 'S0' or 'S1'.
    The default is 'S1'.

    To improve performance of piecewise and direct numerical integration one
    can specify ``levy_stable.quad_eps`` (defaults to 1.2e-14). This is used
    as both the absolute and relative quadrature tolerance for direct numerical
    integration and as the relative quadrature tolerance for the piecewise
    method. One can also specify ``levy_stable.piecewise_x_tol_near_zeta``
    (defaults to 0.005) for how close x is to zeta before it is considered the
    same as x [NO]. The exact check is
    ``abs(x0 - zeta) < piecewise_x_tol_near_zeta*alpha**(1/alpha)``. One can
    also specify ``levy_stable.piecewise_alpha_tol_near_one`` (defaults to
    0.005) for how close alpha is to 1 before being considered equal to 1.


    # 可以指定 `levy_stable.piecewise_x_tol_near_zeta` 参数（默认为 0.005）来确定 x 距离 zeta 多近算作相等
    # 精确检查条件为 `abs(x0 - zeta) < piecewise_x_tol_near_zeta*alpha**(1/alpha)`
    # 也可以指定 `levy_stable.piecewise_alpha_tol_near_one` 参数（默认为 0.005）来确定 alpha 距离 1 多近算作相等


    To increase accuracy of FFT calculation one can specify
    ``levy_stable.pdf_fft_grid_spacing`` (defaults to 0.001) and
    ``pdf_fft_n_points_two_power`` (defaults to None which means a value is
    calculated that sufficiently covers the input range).


    # 若要提高 FFT 计算的精度，可以指定 `levy_stable.pdf_fft_grid_spacing` 参数（默认为 0.001）
    # 和 `pdf_fft_n_points_two_power` 参数（默认为 None，即自动计算一个足够覆盖输入范围的值）


    Further control over FFT calculation is available by setting
    ``pdf_fft_interpolation_degree`` (defaults to 3) for spline order and
    ``pdf_fft_interpolation_level`` for determining the number of points to use
    in the Newton-Cotes formula when approximating the characteristic function
    (considered experimental).


    # 可以通过设置 `pdf_fft_interpolation_degree` 参数（默认为 3）来控制样条插值的阶数
    # 和 `pdf_fft_interpolation_level` 参数来确定在近似特征函数时使用的点数
    # （考虑为实验性质）


    Evaluation of the cdf uses Nolan's piecewise integration approach with the
    Zolatarev :math:`S_0` parameterization by default. There is also the option
    to evaluate through integration of an interpolated spline of the pdf
    calculated by means of the FFT method. The settings affecting FFT
    calculation are the same as for pdf calculation. The default cdf method can
    be changed by setting ``levy_stable.cdf_default_method`` to either
    'piecewise' or 'fft-simpson'.  For cdf calculations the Zolatarev method is
    superior in accuracy, so FFT is disabled by default.


    # 默认情况下，CDF 的评估使用 Nolan 的分段积分方法和 Zolatarev 的参数化 :math:`S_0`。
    # 还可以选择通过 FFT 方法计算的 PDF 的插值样条来进行评估。
    # 影响 FFT 计算的设置与 PDF 计算相同。
    # 可以通过将 `levy_stable.cdf_default_method` 设置为 'piecewise' 或 'fft-simpson' 来更改默认的 CDF 方法。
    # 对于 CDF 计算，Zolatarev 方法在精度上更优，因此默认情况下禁用 FFT。


    Fitting estimate uses quantile estimation method in [MC]. MLE estimation of
    parameters in fit method uses this quantile estimate initially. Note that
    MLE doesn't always converge if using FFT for pdf calculations; this will be
    the case if alpha <= 1 where the FFT approach doesn't give good
    approximations.


    # 拟合估计使用 [MC] 中的分位数估计方法。
    # 拟合方法中参数的 MLE 估计最初使用此分位数估计。
    # 注意，如果使用 FFT 进行 PDF 计算，MLE 并不总是收敛；
    # 当 alpha <= 1 时，FFT 方法不能给出良好的近似。


    Any non-missing value for the attribute
    ``levy_stable.pdf_fft_min_points_threshold`` will set
    ``levy_stable.pdf_default_method`` to 'fft-simpson' if a valid
    default method is not otherwise set.


    # 对于属性 `levy_stable.pdf_fft_min_points_threshold` 的任何非缺失值，
    # 如果没有设置有效的默认方法，将把 `levy_stable.pdf_default_method` 设置为 'fft-simpson'。


    .. warning::

        For pdf calculations FFT calculation is considered experimental.

        For cdf calculations FFT calculation is considered experimental. Use
        Zolatarev's method instead (default).


    # 警告:
    # 对于 PDF 计算，FFT 计算被视为实验性质。
    # 对于 CDF 计算，FFT 计算被视为实验性质。默认情况下，请使用 Zolatarev 方法。


    The probability density above is defined in the "standardized" form. To
    shift and/or scale the distribution use the ``loc`` and ``scale``
    parameters.
    Generally ``%(name)s.pdf(x, %(shapes)s, loc, scale)`` is identically
    equivalent to ``%(name)s.pdf(y, %(shapes)s) / scale`` with
    ``y = (x - loc) / scale``, except in the ``S1`` parameterization if
    ``alpha == 1``.  In that case ``%(name)s.pdf(x, %(shapes)s, loc, scale)``
    is identically equivalent to ``%(name)s.pdf(y, %(shapes)s) / scale`` with
    ``y = (x - loc - 2 * beta * scale * np.log(scale) / np.pi) / scale``.


    # 上述概率密度以“标准化”形式定义。要进行分布的平移和/或缩放，请使用 `loc` 和 `scale` 参数。
    # 通常情况下， ``%(name)s.pdf(x, %(shapes)s, loc, scale)`` 等同于 ``%(name)s.pdf(y, %(shapes)s) / scale``,
    # 其中 ``y = (x - loc) / scale``, 除了在 ``S1`` 参数化中如果 ``alpha == 1`` 时。
    # 在这种情况下， ``%(name)s.pdf(x, %(shapes)s, loc, scale)`` 等同于 ``%(name)s.pdf(y, %(shapes)s) / scale``,
    # 其中 ``y = (x - loc - 2 * beta * scale * np.log(scale) / np.pi) / scale``。
    """
    See [NO2]_ Definition 1.8 for more information.
    Note that shifting the location of a distribution
    does not make it a "noncentral" distribution.

    References
    ----------
    .. [MC] McCulloch, J., 1986. Simple consistent estimators of stable
        distribution parameters. Communications in Statistics - Simulation and
        Computation 15, 11091136.
    .. [WZ] Wang, Li and Zhang, Ji-Hong, 2008. Simpson's rule based FFT method
        to compute densities of stable distribution.
    .. [NO] Nolan, J., 1997. Numerical Calculation of Stable Densities and
        distributions Functions.
    .. [NO2] Nolan, J., 2018. Stable Distributions: Models for Heavy Tailed
        Data.
    .. [HO] Hopcraft, K. I., Jakeman, E., Tanner, R. M. J., 1999. Lévy random
        walks with fluctuating step number and multiscale behavior.

    %(example)s

    """
    # Configurable options as class variables
    # (accessible from self by attribute lookup).
    parameterization = "S1"
    pdf_default_method = "piecewise"
    cdf_default_method = "piecewise"
    quad_eps = _QUAD_EPS  # 用于数值积分的精度参数
    piecewise_x_tol_near_zeta = 0.005  # 在 zeta 附近的分段函数近似的 x 容差
    piecewise_alpha_tol_near_one = 0.005  # 在 alpha 接近 1 时的分段函数近似的 alpha 容差
    pdf_fft_min_points_threshold = None  # FFT 方法计算概率密度函数所需的最小点数阈值
    pdf_fft_grid_spacing = 0.001  # FFT 方法计算概率密度函数的网格间距
    pdf_fft_n_points_two_power = None  # FFT 方法计算概率密度函数时的点数，应为 2 的幂次方
    pdf_fft_interpolation_level = 3  # FFT 方法计算概率密度函数的插值级别
    pdf_fft_interpolation_degree = 3  # FFT 方法计算概率密度函数的插值次数

    def _argcheck(self, alpha, beta):
        return (alpha > 0) & (alpha <= 2) & (beta <= 1) & (beta >= -1)
        # 检查参数 alpha 和 beta 是否在合法范围内的函数

    def _shape_info(self):
        ialpha = _ShapeInfo("alpha", False, (0, 2), (False, True))
        ibeta = _ShapeInfo("beta", False, (-1, 1), (True, True))
        return [ialpha, ibeta]
        # 返回 alpha 和 beta 参数的形状信息列表

    def _parameterization(self):
        allowed = ("S0", "S1")
        pz = self.parameterization
        if pz not in allowed:
            raise RuntimeError(
                f"Parameterization '{pz}' in supported list: {allowed}"
            )
        return pz
        # 检查并返回参数化类型，如果不在允许的列表中则引发错误

    @inherit_docstring_from(rv_continuous)
    def rvs(self, *args, **kwds):
        X1 = super().rvs(*args, **kwds)

        kwds.pop("discrete", None)
        kwds.pop("random_state", None)
        (alpha, beta), delta, gamma, size = self._parse_args_rvs(*args, **kwds)

        # shift location for this parameterisation (S1)
        X1 = np.where(
            alpha == 1.0, X1 + 2 * beta * gamma * np.log(gamma) / np.pi, X1
        )
        # 根据参数化类型对生成的随机变量进行位置调整

        if self._parameterization() == "S0":
            return np.where(
                alpha == 1.0,
                X1 - (beta * 2 * gamma * np.log(gamma) / np.pi),
                X1 - gamma * beta * np.tan(np.pi * alpha / 2.0),
            )
        elif self._parameterization() == "S1":
            return X1
        # 根据参数化类型返回相应的随机变量

    def _rvs(self, alpha, beta, size=None, random_state=None):
        return _rvs_Z1(alpha, beta, size, random_state)
        # 返回符合指定稳定分布参数的随机变量

    @inherit_docstring_from(rv_continuous)
    def pdf(self, x, *args, **kwds):
        # 覆盖基类版本以更正 S1 参数化的位置
        if self._parameterization() == "S0":
            # 如果参数化方式为 S0，则调用基类的 pdf 方法
            return super().pdf(x, *args, **kwds)
        elif self._parameterization() == "S1":
            # 解析参数
            (alpha, beta), delta, gamma = self._parse_args(*args, **kwds)
            # 如果 alpha 的所有值都不等于 1，则调用基类的 pdf 方法
            if np.all(np.reshape(alpha, (1, -1))[0, :] != 1):
                return super().pdf(x, *args, **kwds)
            else:
                # 对参数进行广播以匹配输入数据的形状
                x = np.reshape(x, (1, -1))[0, :]
                x, alpha, beta = np.broadcast_arrays(x, alpha, beta)

                # 组合数据为 (x, alpha, beta) 的三维数组
                data_in = np.dstack((x, alpha, beta))[0]
                # 初始化输出数据的空数组
                data_out = np.empty(shape=(len(data_in), 1))
                # 找到唯一的 alpha, beta 参数对
                uniq_param_pairs = np.unique(data_in[:, 1:], axis=0)
                # 遍历每个参数对
                for pair in uniq_param_pairs:
                    _alpha, _beta = pair
                    # 根据参数值计算 delta
                    _delta = (
                        delta + 2 * _beta * gamma * np.log(gamma) / np.pi
                        if _alpha == 1.0
                        else delta
                    )
                    # 创建数据掩码以过滤具有相同参数对的数据
                    data_mask = np.all(data_in[:, 1:] == pair, axis=-1)
                    # 从输入数据中提取符合当前参数对的 x 值
                    _x = data_in[data_mask, 0]
                    # 计算当前参数对下的概率密度值，并将结果放入输出数组
                    data_out[data_mask] = (
                        super()
                        .pdf(_x, _alpha, _beta, loc=_delta, scale=gamma)
                        .reshape(len(_x), 1)
                    )
                # 将输出数据转置为行向量
                output = data_out.T[0]
                # 如果输出形状为 (1,)，则返回单个值
                if output.shape == (1,):
                    return output[0]
                return output

    @inherit_docstring_from(rv_continuous)
    # 覆盖基类版本以修正 S1 参数化的位置
    # 注意：这与上面的 pdf() 几乎完全相同
    def cdf(self, x, *args, **kwds):
        if self._parameterization() == "S0":
            # 如果参数化为 S0，则调用基类的 cdf 方法
            return super().cdf(x, *args, **kwds)
        elif self._parameterization() == "S1":
            # 如果参数化为 S1，则解析参数并进行处理
            (alpha, beta), delta, gamma = self._parse_args(*args, **kwds)
            if np.all(np.reshape(alpha, (1, -1))[0, :] != 1):
                # 如果 alpha 不全为 1，则调用基类的 cdf 方法
                return super().cdf(x, *args, **kwds)
            else:
                # 对于 S1 参数化的正确位置处理
                x = np.reshape(x, (1, -1))[0, :]
                x, alpha, beta = np.broadcast_arrays(x, alpha, beta)

                # 将 x, alpha, beta 组合成一个数组
                data_in = np.dstack((x, alpha, beta))[0]
                # 创建一个空数组来存储输出
                data_out = np.empty(shape=(len(data_in), 1))

                # 找到唯一的 alpha, beta 对，并分组
                uniq_param_pairs = np.unique(data_in[:, 1:], axis=0)
                for pair in uniq_param_pairs:
                    _alpha, _beta = pair
                    # 根据 alpha 的值更新 delta
                    _delta = (
                        delta + 2 * _beta * gamma * np.log(gamma) / np.pi
                        if _alpha == 1.0
                        else delta
                    )
                    # 创建一个数据掩码，用于选择特定的 alpha, beta 对应的数据
                    data_mask = np.all(data_in[:, 1:] == pair, axis=-1)
                    _x = data_in[data_mask, 0]
                    # 计算每组数据的 cdf，并将结果存储到 data_out 中
                    data_out[data_mask] = (
                        super()
                        .cdf(_x, _alpha, _beta, loc=_delta, scale=gamma)
                        .reshape(len(_x), 1)
                    )

                # 将输出转置为行向量
                output = data_out.T[0]
                # 如果输出的形状为 (1,)，则返回其第一个元素
                if output.shape == (1,):
                    return output[0]
                return output

    # 根据参数化类型选择合适的 _fitstart 函数并返回其结果
    def _fitstart(self, data):
        if self._parameterization() == "S0":
            _fitstart = _fitstart_S0
        elif self._parameterization() == "S1":
            _fitstart = _fitstart_S1
        return _fitstart(data)

    # 计算给定 alpha 和 beta 的统计量并返回
    def _stats(self, alpha, beta):
        # 根据 alpha 的值计算 mu, mu2, g1, g2
        mu = 0 if alpha > 1 else np.nan
        mu2 = 2 if alpha == 2 else np.inf
        g1 = 0.0 if alpha == 2.0 else np.nan
        g2 = 0.0 if alpha == 2.0 else np.nan
        return mu, mu2, g1, g2
# 使用 Cotes 数字，参见 http://oeis.org/A100642 的序列
Cotes_table = np.array(
    [[], [1]] + [v[2] for v in _builtincoeffs.values()], dtype=object
)

# 构建 Cotes 数组，用于特征函数的傅立叶变换及 Newton-Cotes 积分
Cotes = np.array(
    [
        np.pad(r, (0, len(Cotes_table) - 1 - len(r)), mode='constant')
        for r in Cotes_table
    ]
)

def pdf_from_cf_with_fft(cf, h=0.01, q=9, level=3):
    """Calculates pdf from characteristic function.

    Uses fast Fourier transform with Newton-Cotes integration following [WZ].
    Defaults to using Simpson's method (3-point Newton-Cotes integration).

    Parameters
    ----------
    cf : callable
        Single argument function from float -> complex expressing a
        characteristic function for some distribution.
    h : Optional[float]
        Step size for Newton-Cotes integration. Default: 0.01
    q : Optional[int]
        Use 2**q steps when performing Newton-Cotes integration.
        The infinite integral in the inverse Fourier transform will then
        be restricted to the interval [-2**q * h / 2, 2**q * h / 2]. Setting
        the number of steps equal to a power of 2 allows the fft to be
        calculated in O(n*log(n)) time rather than O(n**2).
        Default: 9
    level : Optional[int]
        Calculate integral using n-point Newton-Cotes integration for
        n = level. The 3-point Newton-Cotes formula corresponds to Simpson's
        rule. Default: 3

    Returns
    -------
    x_l : ndarray
        Array of points x at which pdf is estimated. 2**q equally spaced
        points from -pi/h up to but not including pi/h.
    density : ndarray
        Estimated values of pdf corresponding to cf at points in x_l.

    References
    ----------
    .. [WZ] Wang, Li and Zhang, Ji-Hong, 2008. Simpson's rule based FFT method
        to compute densities of stable distribution.
    """
    n = level
    N = 2**q
    steps = np.arange(0, N)
    L = N * h / 2
    x_l = np.pi * (steps - N / 2) / L

    if level > 1:
        # 创建索引以用于多点 Newton-Cotes 积分
        indices = np.arange(n).reshape(n, 1)
        # 计算傅立叶变换及积分
        s1 = np.sum(
            (-1) ** steps * Cotes[n, indices] * np.fft.fft(
                (-1)**steps * cf(-L + h * steps + h * indices / (n - 1))
            ) * np.exp(
                1j * np.pi * indices / (n - 1)
                - 2 * 1j * np.pi * indices * steps /
                (N * (n - 1))
            ),
            axis=0
        )
    else:
        # 单点 Newton-Cotes 积分
        s1 = (-1) ** steps * Cotes[n, 0] * np.fft.fft(
            (-1) ** steps * cf(-L + h * steps)
        )

    # 计算概率密度函数的估计值
    density = h * s1 / (2 * np.pi * np.sum(Cotes[n]))
    return (x_l, density)

# 创建稳定分布的 Levy 稳定分布生成器
levy_stable = levy_stable_gen(name="levy_stable")
```