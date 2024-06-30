# `D:\src\scipysrc\sympy\sympy\physics\control\lti.py`

```
from typing import Type  # 引入 Type 类型提示
from sympy import Interval, numer, Rational, solveset  # 从 sympy 库导入 Interval, numer, Rational, solveset
from sympy.core.add import Add  # 从 sympy.core.add 模块导入 Add 类
from sympy.core.basic import Basic  # 从 sympy.core.basic 模块导入 Basic 类
from sympy.core.containers import Tuple  # 从 sympy.core.containers 模块导入 Tuple 类
from sympy.core.evalf import EvalfMixin  # 从 sympy.core.evalf 模块导入 EvalfMixin 类
from sympy.core.expr import Expr  # 从 sympy.core.expr 模块导入 Expr 类
from sympy.core.function import expand  # 从 sympy.core.function 模块导入 expand 函数
from sympy.core.logic import fuzzy_and  # 从 sympy.core.logic 模块导入 fuzzy_and 函数
from sympy.core.mul import Mul  # 从 sympy.core.mul 模块导入 Mul 类
from sympy.core.numbers import I, pi, oo  # 从 sympy.core.numbers 模块导入 I, pi, oo 常量
from sympy.core.power import Pow  # 从 sympy.core.power 模块导入 Pow 类
from sympy.core.singleton import S  # 从 sympy.core.singleton 模块导入 S 单例
from sympy.core.symbol import Dummy, Symbol  # 从 sympy.core.symbol 模块导入 Dummy, Symbol 类
from sympy.functions import Abs  # 从 sympy.functions 模块导入 Abs 函数
from sympy.core.sympify import sympify, _sympify  # 从 sympy.core.sympify 模块导入 sympify, _sympify 函数
from sympy.matrices import Matrix, ImmutableMatrix, ImmutableDenseMatrix, eye, ShapeError, zeros  # 从 sympy.matrices 模块导入 Matrix, ImmutableMatrix, ImmutableDenseMatrix, eye, ShapeError, zeros
from sympy.functions.elementary.exponential import exp, log  # 从 sympy.functions.elementary.exponential 模块导入 exp, log 函数
from sympy.matrices.expressions import MatMul, MatAdd  # 从 sympy.matrices.expressions 模块导入 MatMul, MatAdd 类
from sympy.polys import Poly, rootof  # 从 sympy.polys 模块导入 Poly, rootof 函数
from sympy.polys.polyroots import roots  # 从 sympy.polys.polyroots 模块导入 roots 函数
from sympy.polys.polytools import cancel, degree  # 从 sympy.polys.polytools 模块导入 cancel, degree 函数
from sympy.series import limit  # 从 sympy.series 模块导入 limit 函数
from sympy.utilities.misc import filldedent  # 从 sympy.utilities.misc 模块导入 filldedent 函数
from sympy.solvers.ode.systems import linodesolve  # 从 sympy.solvers.ode.systems 模块导入 linodesolve 函数
from sympy.solvers.solveset import linsolve, linear_eq_to_matrix  # 从 sympy.solvers.solveset 模块导入 linsolve, linear_eq_to_matrix 函数

from mpmath.libmp.libmpf import prec_to_dps  # 从 mpmath.libmp.libmpf 模块导入 prec_to_dps 函数

__all__ = ['TransferFunction', 'Series', 'MIMOSeries', 'Parallel', 'MIMOParallel',  # 设置模块公开接口
    'Feedback', 'MIMOFeedback', 'TransferFunctionMatrix', 'StateSpace', 'gbt', 'bilinear', 'forward_diff', 'backward_diff',
    'phase_margin', 'gain_margin']

def _roots(poly, var):
    """ like roots, but works on higher-order polynomials. """
    # 计算多项式的所有根（包括重根），返回一个列表
    r = roots(poly, var, multiple=True)
    # 获取多项式的次数
    n = degree(poly)
    # 如果返回的根数不等于多项式的次数，则使用 rootof 函数补充缺失的根
    if len(r) != n:
        r = [rootof(poly, var, k) for k in range(n)]
    return r

def gbt(tf, sample_per, alpha):
    r"""
    Returns falling coefficients of H(z) from numerator and denominator.

    Explanation
    ===========

    Where H(z) is the corresponding discretized transfer function,
    discretized with the generalised bilinear transformation method.
    H(z) is obtained from the continuous transfer function H(s)
    by substituting $s(z) = \frac{z-1}{T(\alpha z + (1-\alpha))}$ into H(s), where T is the
    sample period.
    Coefficients are falling, i.e. $H(z) = \frac{az+b}{cz+d}$ is returned
    as [a, b], [c, d].

    Examples
    ========

    >>> from sympy.physics.control.lti import TransferFunction, gbt
    >>> from sympy.abc import s, L, R, T

    >>> tf = TransferFunction(1, s*L + R, s)
    >>> numZ, denZ = gbt(tf, T, 0.5)
    >>> numZ
    [T/(2*(L + R*T/2)), T/(2*(L + R*T/2))]
    >>> denZ
    [1, (-L + R*T/2)/(L + R*T/2)]

    >>> numZ, denZ = gbt(tf, T, 0)
    >>> numZ
    [T/L]
    >>> denZ
    [1, (-L + R*T)/L]

    >>> numZ, denZ = gbt(tf, T, 1)
    >>> numZ
    [T/(L + R*T), 0]
    >>> denZ
    [1, -L/(L + R*T)]

    >>> numZ, denZ = gbt(tf, T, 0.3)
    >>> numZ
    [3*T/(10*(L + 3*R*T/10)), 7*T/(10*(L + 3*R*T/10))]
    >>> denZ
    [1, (-L + 7*R*T/10)/(L + 3*R*T/10)]

    References
    ==========

    """
    """
    # 如果传入的传递函数不是单输入单输出（SISO）系统，则抛出未实现错误
    if not tf.is_SISO:
        raise NotImplementedError("Not implemented for MIMO systems.")

    # 将采样周期设置为 T，与样本周期相同
    T = sample_per  # and sample period T
    
    # 将传递函数 tf 的方差赋给 s
    s = tf.var
    # 将 s 赋给虚拟离散变量 z
    z = s

    # 获取传递函数 tf 的分子多项式的所有系数
    np = tf.num.as_poly(s).all_coeffs()
    # 获取传递函数 tf 的分母多项式的所有系数
    dp = tf.den.as_poly(s).all_coeffs()
    
    # 将 alpha 转换为最接近其值的有限精度有理数，限定分母不超过1000
    alpha = Rational(alpha).limit_denominator(1000)

    # 计算多项式的阶数 N，取 np 和 dp 中较大长度的减一
    N = max(len(np), len(dp)) - 1

    # 计算传递函数的分子的离散系数
    num = Add(*[ T**(N-i) * c * (z-1)**i * (alpha * z + 1 - alpha)**(N-i) for c, i in zip(np[::-1], range(len(np))) ])
    
    # 计算传递函数的分母的离散系数
    den = Add(*[ T**(N-i) * c * (z-1)**i * (alpha * z + 1 - alpha)**(N-i) for c, i in zip(dp[::-1], range(len(dp))) ])

    # 获取分子的离散系数多项式的所有系数
    num_coefs = num.as_poly(z).all_coeffs()
    # 获取分母的离散系数多项式的所有系数
    den_coefs = den.as_poly(z).all_coeffs()

    # 将分母系数的首项作为参数 para
    para = den_coefs[0]
    # 将分子系数除以 para 得到标准化后的分子系数列表
    num_coefs = [coef / para for coef in num_coefs]
    # 将分母系数除以 para 得到标准化后的分母系数列表
    den_coefs = [coef / para for coef in den_coefs]

    # 返回标准化后的分子系数和分母系数
    return num_coefs, den_coefs
    """
def bilinear(tf, sample_per):
    r"""
    Returns falling coefficients of H(z) from numerator and denominator.

    Explanation
    ===========

    Where H(z) is the corresponding discretized transfer function,
    discretized with the bilinear transform method.
    H(z) is obtained from the continuous transfer function H(s)
    by substituting $s(z) = \frac{2}{T}\frac{z-1}{z+1}$ into H(s), where T is the
    sample period.
    Coefficients are falling, i.e. $H(z) = \frac{az+b}{cz+d}$ is returned
    as [a, b], [c, d].

    Examples
    ========

    >>> from sympy.physics.control.lti import TransferFunction, bilinear
    >>> from sympy.abc import s, L, R, T

    >>> tf = TransferFunction(1, s*L + R, s)
    >>> numZ, denZ = bilinear(tf, T)
    >>> numZ
    [T/(2*(L + R*T/2)), T/(2*(L + R*T/2))]
    >>> denZ
    [1, (-L + R*T/2)/(L + R*T/2)]
    """
    # 使用 bilinear transform 方法计算离散化的传递函数系数
    return gbt(tf, sample_per, S.Half)

def forward_diff(tf, sample_per):
    r"""
    Returns falling coefficients of H(z) from numerator and denominator.

    Explanation
    ===========

    Where H(z) is the corresponding discretized transfer function,
    discretized with the forward difference transform method.
    H(z) is obtained from the continuous transfer function H(s)
    by substituting $s(z) = \frac{z-1}{T}$ into H(s), where T is the
    sample period.
    Coefficients are falling, i.e. $H(z) = \frac{az+b}{cz+d}$ is returned
    as [a, b], [c, d].

    Examples
    ========

    >>> from sympy.physics.control.lti import TransferFunction, forward_diff
    >>> from sympy.abc import s, L, R, T

    >>> tf = TransferFunction(1, s*L + R, s)
    >>> numZ, denZ = forward_diff(tf, T)
    >>> numZ
    [T/L]
    >>> denZ
    [1, (-L + R*T)/L]
    """
    # 使用 forward difference transform 方法计算离散化的传递函数系数
    return gbt(tf, sample_per, S.Zero)

def backward_diff(tf, sample_per):
    r"""
    Returns falling coefficients of H(z) from numerator and denominator.

    Explanation
    ===========

    Where H(z) is the corresponding discretized transfer function,
    discretized with the backward difference transform method.
    H(z) is obtained from the continuous transfer function H(s)
    by substituting $s(z) =  \frac{z-1}{Tz}$ into H(s), where T is the
    sample period.
    Coefficients are falling, i.e. $H(z) = \frac{az+b}{cz+d}$ is returned
    as [a, b], [c, d].

    Examples
    ========

    >>> from sympy.physics.control.lti import TransferFunction, backward_diff
    >>> from sympy.abc import s, L, R, T

    >>> tf = TransferFunction(1, s*L + R, s)
    >>> numZ, denZ = backward_diff(tf, T)
    >>> numZ
    [T/(L + R*T), 0]
    >>> denZ
    [1, -L/(L + R*T)]
    """
    # 使用 backward difference transform 方法计算离散化的传递函数系数
    return gbt(tf, sample_per, S.One)

def phase_margin(system):
    r"""
    Returns the phase margin of a continuous time system.
    Only applicable to Transfer Functions which can generate valid bode plots.

    Raises
    ======

    NotImplementedError
        When time delay terms are present in the system.
    """
    # 返回连续时间系统的相位余量
    # 仅适用于可以生成有效 bode 图的传递函数
    raise NotImplementedError("When time delay terms are present in the system.")
    from sympy.functions import arg  # 导入 sympy 库中的 arg 函数，用于计算复数的辐角

    if not isinstance(system, SISOLinearTimeInvariant):
        raise ValueError("Margins are only applicable for SISO LTI systems.")
        # 如果传入的系统不是 SISO 线性时不变系统，则抛出 ValueError 异常

    _w = Dummy("w", real=True)  # 创建一个实数域的虚拟变量 _w
    repl = I*_w  # 定义替换表达式 repl，将 _w 替换为复数单位虚数乘以 _w
    expr = system.to_expr()  # 将系统转换为表达式形式
    len_free_symbols = len(expr.free_symbols)  # 计算表达式中自由符号的数量
    if expr.has(exp):
        raise NotImplementedError("Margins for systems with Time delay terms are not supported.")
        # 如果表达式中包含 exp 函数，抛出 NotImplementedError 异常，暂不支持带有时延项的系统
    elif len_free_symbols > 1:
        raise ValueError("Extra degree of freedom found. Make sure"
            " that there are no free symbols in the dynamical system other"
            " than the variable of Laplace transform.")
        # 如果表达式中自由符号数量大于1，抛出 ValueError 异常，确保动态系统中除 Laplace 变换变量外没有其他自由符号

    w_expr = expr.subs({system.var: repl})  # 将系统表达式中的系统变量替换为 repl

    mag = 20*log(Abs(w_expr), 10)  # 计算 w_expr 的幅值的对数，以 10 为底
    mag_sol = list(solveset(mag, _w, Interval(0, oo, left_open=True)))  # 解幅值对数的方程，得到解集

    if (len(mag_sol) == 0):
      pm = S(-180)
    else:
      wcp = mag_sol[0]  # 取第一个解作为临界频率 wcp
      pm = ((arg(w_expr)*S(180)/pi).subs({_w:wcp}) + S(180)) % 360
      # 计算相位裕度，使用 arg 函数计算 w_expr 的辐角并转换为角度制，最后将结果调整到 0 到 360 度之间

    if(pm >= 180):
        pm = pm - 360  # 如果相位裕度大于等于 180 度，则减去 360 度

    return pm  # 返回计算得到的相位裕度
def gain_margin(system):
    r"""
    Returns the gain margin of a continuous time system.
    Only applicable to Transfer Functions which can generate valid bode plots.

    Raises
    ======

    NotImplementedError
        When time delay terms are present in the system.

    ValueError
        When a SISO LTI system is not passed.

        When more than one free symbol is present in the system.
        The only variable in the transfer function should be
        the variable of the Laplace transform.

    Examples
    ========

    >>> from sympy.physics.control import TransferFunction, gain_margin
    >>> from sympy.abc import s

    >>> tf = TransferFunction(1, s**3 + 2*s**2 + s, s)
    >>> gain_margin(tf)
    20*log(2)/log(10)
    >>> gain_margin(tf).n()
    6.02059991327962

    >>> tf1 = TransferFunction(s**3, s**2 + 5*s, s)
    >>> gain_margin(tf1)
    oo

    See Also
    ========

    phase_margin

    References
    ==========

    https://en.wikipedia.org/wiki/Bode_plot

    """
    # 检查传入的系统是否为 SISO (Single Input Single Output) 线性时不变系统
    if not isinstance(system, SISOLinearTimeInvariant):
        raise ValueError("Margins are only applicable for SISO LTI systems.")

    # 定义虚拟变量 _w 用于替换 Laplace 变换中的 s*j
    _w = Dummy("w", real=True)
    repl = I*_w
    # 将系统转换为表达式形式
    expr = system.to_expr()
    # 获取表达式中的自由变量数量
    len_free_symbols = len(expr.free_symbols)
    
    # 如果表达式中包含 exp，则不支持带有时延项的系统
    if expr.has(exp):
        raise NotImplementedError("Margins for systems with Time delay terms are not supported.")
    # 如果表达式中的自由变量超过一个，则抛出错误
    elif len_free_symbols > 1:
        raise ValueError("Extra degree of freedom found. Make sure"
                         " that there are no free symbols in the dynamical system other"
                         " than the variable of Laplace transform.")

    # 将 Laplace 变换的变量替换为虚拟变量 _w
    w_expr = expr.subs({system.var: repl})

    # 计算增益裕量
    mag = 20*log(Abs(w_expr), 10)
    phase = w_expr
    # 解决相位的实部和虚部，并找到解集
    phase_sol = list(solveset(numer(phase.as_real_imag()[1].cancel()), _w, Interval(0, oo, left_open=True)))

    # 如果相位解为空集，则增益裕量为无穷大
    if len(phase_sol) == 0:
        gm = oo
    else:
        wcg = phase_sol[0]
        gm = -mag.subs({_w: wcg})

    return gm


class LinearTimeInvariant(Basic, EvalfMixin):
    """A common class for all the Linear Time-Invariant Dynamical Systems."""

    _clstype: Type

    # Users should not directly interact with this class.
    def __new__(cls, *system, **kwargs):
        if cls is LinearTimeInvariant:
            raise NotImplementedError('The LTICommon class is not meant to be used directly.')
        return super(LinearTimeInvariant, cls).__new__(cls, *system, **kwargs)

    @classmethod
    def _check_args(cls, args):
        # 至少需要传入一个参数
        if not args:
            raise ValueError("At least 1 argument must be passed.")
        # 所有参数必须是指定类型的对象
        if not all(isinstance(arg, cls._clstype) for arg in args):
            raise TypeError(f"All arguments must be of type {cls._clstype}.")
        # 确保所有传输函数使用相同的复变量进行 Laplace 变换
        var_set = {arg.var for arg in args}
        if len(var_set) != 1:
            raise ValueError(filldedent(f"""
                All transfer functions should use the same complex variable
                of the Laplace transform. {len(var_set)} different
                values found."""))
    # 定义一个属性方法，用于判断当前的线性时不变系统是否是 SISO（Single Input Single Output）系统。
    @property
    def is_SISO(self):
        """Returns `True` if the passed LTI system is SISO else returns False."""
        # 返回存储在实例变量 _is_SISO 中的布尔值，表示系统是否是 SISO 类型。
        return self._is_SISO
class SISOLinearTimeInvariant(LinearTimeInvariant):
    """A common class for all the SISO Linear Time-Invariant Dynamical Systems."""
    # Users should not directly interact with this class.

    @property
    def num_inputs(self):
        """Return the number of inputs for SISOLinearTimeInvariant."""
        return 1

    @property
    def num_outputs(self):
        """Return the number of outputs for SISOLinearTimeInvariant."""
        return 1

    _is_SISO = True  # Indicates this is a Single-Input Single-Output system


class MIMOLinearTimeInvariant(LinearTimeInvariant):
    """A common class for all the MIMO Linear Time-Invariant Dynamical Systems."""
    # Users should not directly interact with this class.
    _is_SISO = False  # Indicates this is not a Single-Input Single-Output system


SISOLinearTimeInvariant._clstype = SISOLinearTimeInvariant
MIMOLinearTimeInvariant._clstype = MIMOLinearTimeInvariant


def _check_other_SISO(func):
    def wrapper(*args, **kwargs):
        if not isinstance(args[-1], SISOLinearTimeInvariant):
            return NotImplemented
        else:
            return func(*args, **kwargs)
    return wrapper


def _check_other_MIMO(func):
    def wrapper(*args, **kwargs):
        if not isinstance(args[-1], MIMOLinearTimeInvariant):
            return NotImplemented
        else:
            return func(*args, **kwargs)
    return wrapper


class TransferFunction(SISOLinearTimeInvariant):
    r"""
    A class for representing LTI (Linear, time-invariant) systems that can be strictly described
    by ratio of polynomials in the Laplace transform complex variable. The arguments
    are ``num``, ``den``, and ``var``, where ``num`` and ``den`` are numerator and
    denominator polynomials of the ``TransferFunction`` respectively, and the third argument is
    a complex variable of the Laplace transform used by these polynomials of the transfer function.
    ``num`` and ``den`` can be either polynomials or numbers, whereas ``var``
    has to be a :py:class:`~.Symbol`.

    Explanation
    ===========

    Generally, a dynamical system representing a physical model can be described in terms of Linear
    Ordinary Differential Equations like -

            $\small{b_{m}y^{\left(m\right)}+b_{m-1}y^{\left(m-1\right)}+\dots+b_{1}y^{\left(1\right)}+b_{0}y=
            a_{n}x^{\left(n\right)}+a_{n-1}x^{\left(n-1\right)}+\dots+a_{1}x^{\left(1\right)}+a_{0}x}$

    Here, $x$ is the input signal and $y$ is the output signal and superscript on both is the order of derivative
    (not exponent). Derivative is taken with respect to the independent variable, $t$. Also, generally $m$ is greater
    than $n$.

    It is not feasible to analyse the properties of such systems in their native form therefore, we use
    mathematical tools like Laplace transform to get a better perspective. Taking the Laplace transform
    of both the sides in the equation (at zero initial conditions), we get -

            $\small{\mathcal{L}[b_{m}y^{\left(m\right)}+b_{m-1}y^{\left(m-1\right)}+\dots+b_{1}y^{\left(1\right)}+b_{0}y]=
            \mathcal{L}[a_{n}x^{\left(n\right)}+a_{n-1}x^{\left(n-1\right)}+\dots+a_{1}x^{\left(1\right)}+a_{0}x]}$

    Using the linearity property of Laplace transform and also considering zero initial conditions
    (i.e. $\small{y(0^{-}) = 0}$, $\small{y'(0^{-}) = 0}$ and so on), the equation
    above gets translated to -

            $\small{b_{m}\mathcal{L}[y^{\left(m\right)}]+\dots+b_{1}\mathcal{L}[y^{\left(1\right)}]+b_{0}\mathcal{L}[y]=
            a_{n}\mathcal{L}[x^{\left(n\right)}]+\dots+a_{1}\mathcal{L}[x^{\left(1\right)}]+a_{0}\mathcal{L}[x]}$

    Now, applying Derivative property of Laplace transform,

            $\small{b_{m}s^{m}\mathcal{L}[y]+\dots+b_{1}s\mathcal{L}[y]+b_{0}\mathcal{L}[y]=
            a_{n}s^{n}\mathcal{L}[x]+\dots+a_{1}s\mathcal{L}[x]+a_{0}\mathcal{L}[x]}$

    Here, the superscript on $s$ is **exponent**. Note that the zero initial conditions assumption, mentioned above, is very important
    and cannot be ignored otherwise the dynamical system cannot be considered time-independent and the simplified equation above
    cannot be reached.

    Collecting $\mathcal{L}[y]$ and $\mathcal{L}[x]$ terms from both the sides and taking the ratio
    $\frac{ \mathcal{L}\left\{y\right\} }{ \mathcal{L}\left\{x\right\} }$, we get the typical rational form of transfer
    function.

    The numerator of the transfer function is, therefore, the Laplace transform of the output signal
    (The signals are represented as functions of time) and similarly, the denominator
    of the transfer function is the Laplace transform of the input signal. It is also a convention
    to denote the input and output signal's Laplace transform with capital alphabets like shown below.

            $H(s) = \frac{Y(s)}{X(s)} = \frac{ \mathcal{L}\left\{y(t)\right\} }{ \mathcal{L}\left\{x(t)\right\} }$

    $s$, also known as complex frequency, is a complex variable in the Laplace domain. It corresponds to the
    equivalent variable $t$, in the time domain. Transfer functions are sometimes also referred to as the Laplace
    transform of the system's impulse response. Transfer function, $H$, is represented as a rational
    function in $s$ like,

            $H(s) =\ \frac{a_{n}s^{n}+a_{n-1}s^{n-1}+\dots+a_{1}s+a_{0}}{b_{m}s^{m}+b_{m-1}s^{m-1}+\dots+b_{1}s+b_{0}}$

    Parameters
    ==========

    num : Expr, Number
        The numerator polynomial of the transfer function.
    den : Expr, Number
        The denominator polynomial of the transfer function.
    var : Symbol
        Complex variable of the Laplace transform used by the
        polynomials of the transfer function.

    Raises
    ======

    TypeError
        When ``var`` is not a Symbol or when ``num`` or ``den`` is not a
        number or a polynomial.
    ValueError
        When ``den`` is zero.

    Examples
    ========

    >>> from sympy.abc import s, p, a
    >>> from sympy.physics.control.lti import TransferFunction

    # 创建一个传递函数对象 tf1，分子为 s + a，分母为 s**2 + s + 1，变量为 s
    >>> tf1 = TransferFunction(s + a, s**2 + s + 1, s)
    # 显示 tf1 的值
    >>> tf1
    TransferFunction(a + s, s**2 + s + 1, s)
    # 显示 tf1 的分子
    >>> tf1.num
    a + s
    # 显示 tf1 的分母
    >>> tf1.den
    s**2 + s + 1
    # 显示 tf1 的变量
    >>> tf1.var
    s
    # 显示 tf1 的所有参数
    >>> tf1.args
    (a + s, s**2 + s + 1, s)

    # 可以使用任意复数变量作为 ``var``
    >>> tf2 = TransferFunction(a*p**3 - a*p**2 + s*p, p + a**2, p)
    >>> tf2
    TransferFunction(a*p**3 - a*p**2 + p*s, a**2 + p, p)
    >>> tf3 = TransferFunction((p + 3)*(p - 1), (p - 1)*(p + 5), p)
    >>> tf3
    TransferFunction((p - 1)*(p + 3), (p - 1)*(p + 5), p)

    # 使用 ``-`` 操作符来对传递函数取负
    >>> tf4 = TransferFunction(-a + s, p**2 + s, p)
    >>> -tf4
    TransferFunction(a - s, p**2 + s, p)
    >>> tf5 = TransferFunction(s**4 - 2*s**3 + 5*s + 4, s + 4, s)
    >>> -tf5
    TransferFunction(-s**4 + 2*s**3 - 5*s - 4, s + 4, s)

    # 可以使用浮点数或整数（或其他常数）作为分子和分母
    >>> tf6 = TransferFunction(1/2, 4, s)
    >>> tf6.num
    0.500000000000000
    >>> tf6.den
    4
    >>> tf6.var
    s
    >>> tf6.args
    (0.5, 4, s)

    # 可以使用 ``**`` 操作符对传递函数取整数次幂
    >>> tf7 = TransferFunction(s + a, s - a, s)
    >>> tf7**3
    TransferFunction((a + s)**3, (-a + s)**3, s)
    >>> tf7**0
    TransferFunction(1, 1, s)
    >>> tf8 = TransferFunction(p + 4, p - 3, p)
    >>> tf8**-1
    TransferFunction(p - 3, p + 4, p)

    # 传递函数的加法、减法和乘法可以形成未评估的 ``Series`` 或 ``Parallel`` 对象
    >>> tf9 = TransferFunction(s + 1, s**2 + s + 1, s)
    >>> tf10 = TransferFunction(s - p, s + 3, s)
    >>> tf11 = TransferFunction(4*s**2 + 2*s - 4, s - 1, s)
    >>> tf12 = TransferFunction(1 - s, s**2 + 4, s)
    >>> tf9 + tf10
    Parallel(TransferFunction(s + 1, s**2 + s + 1, s), TransferFunction(-p + s, s + 3, s))
    >>> tf10 - tf11
    Parallel(TransferFunction(-p + s, s + 3, s), TransferFunction(-4*s**2 - 2*s + 4, s - 1, s))
    >>> tf9 * tf10
    Series(TransferFunction(s + 1, s**2 + s + 1, s), TransferFunction(-p + s, s + 3, s))
    >>> tf10 - (tf9 + tf12)
    Parallel(TransferFunction(-p + s, s + 3, s), TransferFunction(-s - 1, s**2 + s + 1, s), TransferFunction(s - 1, s**2 + 4, s))
    >>> tf10 - (tf9 * tf12)
    Parallel(TransferFunction(-p + s, s + 3, s), Series(TransferFunction(-1, 1, s), TransferFunction(s + 1, s**2 + s + 1, s), TransferFunction(1 - s, s**2 + 4, s)))
    >>> tf11 * tf10 * tf9
    Series(TransferFunction(4*s**2 + 2*s - 4, s - 1, s), TransferFunction(-p + s, s + 3, s), TransferFunction(s + 1, s**2 + s + 1, s))
    >>> tf9 * tf11 + tf10 * tf12
    # 定义一个并行结构，包含两个串联结构：第一个串联结构包括两个传递函数，第二个串联结构也包括两个传递函数
    Parallel(
        Series(TransferFunction(s + 1, s**2 + s + 1, s), TransferFunction(4*s**2 + 2*s - 4, s - 1, s)),
        Series(TransferFunction(-p + s, s + 3, s), TransferFunction(1 - s, s**2 + 4, s))
    )
    >>> (tf9 + tf12) * (tf10 + tf11)
    # 返回一个串联结构，其中包含两个并行结构：第一个并行结构包括传递函数和传递函数，第二个并行结构也包括传递函数和传递函数
    Series(
        Parallel(TransferFunction(s + 1, s**2 + s + 1, s), TransferFunction(1 - s, s**2 + 4, s)),
        Parallel(TransferFunction(-p + s, s + 3, s), TransferFunction(4*s**2 + 2*s - 4, s - 1, s))
    )

    These unevaluated ``Series`` or ``Parallel`` objects can convert into the
    resultant transfer function using ``.doit()`` method or by ``.rewrite(TransferFunction)``.

    >>> ((tf9 + tf10) * tf12).doit()
    # 返回一个转移函数，其分子是 (1 - s)*((-p + s)*(s**2 + s + 1) + (s + 1)*(s + 3))，分母是 (s + 3)*(s**2 + 4)*(s**2 + s + 1)
    TransferFunction((1 - s)*((-p + s)*(s**2 + s + 1) + (s + 1)*(s + 3)), (s + 3)*(s**2 + 4)*(s**2 + s + 1), s)
    >>> (tf9 * tf10 - tf11 * tf12).rewrite(TransferFunction)
    # 返回一个重写的转移函数，其分子是 -(1 - s)*(s + 3)*(s**2 + s + 1)*(4*s**2 + 2*s - 4) + (-p + s)*(s - 1)*(s + 1)*(s**2 + 4)，分母是 (s - 1)*(s + 3)*(s**2 + 4)*(s**2 + s + 1)

    See Also
    ========

    Feedback, Series, Parallel

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Transfer_function
    .. [2] https://en.wikipedia.org/wiki/Laplace_transform

    """
    # 定义一个新的类方法 `__new__`，用于创建 TransferFunction 对象，参数包括分子 `num`、分母 `den` 和变量 `var`
    def __new__(cls, num, den, var):
        # 将分子和分母转换成符号表达式
        num, den = _sympify(num), _sympify(den)

        # 如果变量不是符号类型，抛出类型错误异常
        if not isinstance(var, Symbol):
            raise TypeError("Variable input must be a Symbol.")

        # 如果分母为零，抛出值错误异常
        if den == 0:
            raise ValueError("TransferFunction cannot have a zero denominator.")

        # 检查分子和分母类型是否支持转移函数的创建，如果不支持则抛出类型错误异常
        if (((isinstance(num, (Expr, TransferFunction, Series, Parallel)) and num.has(Symbol)) or num.is_number) and
            ((isinstance(den, (Expr, TransferFunction, Series, Parallel)) and den.has(Symbol)) or den.is_number)):
            return super(TransferFunction, cls).__new__(cls, num, den, var)

        else:
            raise TypeError("Unsupported type for numerator or denominator of TransferFunction.")

    @classmethod
    # 返回类方法修饰符，用于创建 TransferFunction 对象的类方法
    @classmethod
    def from_coeff_lists(cls, num_list, den_list, var):
        r"""
        Creates a new ``TransferFunction`` efficiently from a list of coefficients.

        Parameters
        ==========

        num_list : Sequence
            Sequence comprising of numerator coefficients.
        den_list : Sequence
            Sequence comprising of denominator coefficients.
        var : Symbol
            Complex variable of the Laplace transform used by the
            polynomials of the transfer function.

        Raises
        ======

        ZeroDivisionError
            When the constructed denominator is zero.

        Examples
        ========

        >>> from sympy.abc import s, p
        >>> from sympy.physics.control.lti import TransferFunction
        >>> num = [1, 0, 2]
        >>> den = [3, 2, 2, 1]
        >>> tf = TransferFunction.from_coeff_lists(num, den, s)
        >>> tf
        TransferFunction(s**2 + 2, 3*s**3 + 2*s**2 + 2*s + 1, s)

        # Create a Transfer Function with more than one variable
        >>> tf1 = TransferFunction.from_coeff_lists([p, 1], [2*p, 0, 4], s)
        >>> tf1
        TransferFunction(p*s + 1, 2*p*s**2 + 4, s)

        """
        # Reverse the coefficient lists to align with ascending powers of var
        num_list = num_list[::-1]
        den_list = den_list[::-1]

        # Compute powers of var corresponding to coefficients in num_list and den_list
        num_var_powers = [var**i for i in range(len(num_list))]
        den_var_powers = [var**i for i in range(len(den_list))]

        # Construct the numerator and denominator polynomials
        _num = sum(coeff * var_power for coeff, var_power in zip(num_list, num_var_powers))
        _den = sum(coeff * var_power for coeff, var_power in zip(den_list, den_var_powers))

        # Check if the denominator is zero and raise an exception if so
        if _den == 0:
            raise ZeroDivisionError("TransferFunction cannot have a zero denominator.")

        # Return a new instance of TransferFunction using computed numerator and denominator
        return cls(_num, _den, var)
    @property
    def num(self):
        """
        Returns the numerator polynomial of the transfer function.

        Examples
        ========

        >>> from sympy.abc import s, p
        >>> from sympy.physics.control.lti import TransferFunction
        >>> G1 = TransferFunction(s**2 + p*s + 3, s - 4, s)
        >>> G1.num
        p*s + s**2 + 3
        >>> G2 = TransferFunction((p + 5)*(p - 3), (p - 3)*(p + 1), p)
        >>> G2.num
        (p - 3)*(p + 5)

        """
        # 返回传递函数的分子多项式
        return self.args[0]

    @property
    def den(self):
        """
        Returns the denominator polynomial of the transfer function.

        Examples
        ========

        >>> from sympy.abc import s, p
        >>> from sympy.physics.control.lti import TransferFunction
        >>> G1 = TransferFunction(s + 4, p**3 - 2*p + 4, s)
        >>> G1.den
        p**3 - 2*p + 4
        >>> G2 = TransferFunction(3, 4, s)
        >>> G2.den
        4

        """
        # 返回传递函数的分母多项式
        return self.args[1]

    @property
    def var(self):
        """
        返回 Laplace 变换中多项式的复杂变量，用于传递函数。

        示例
        ========

        >>> from sympy.abc import s, p
        >>> from sympy.physics.control.lti import TransferFunction
        >>> G1 = TransferFunction(p**2 + 2*p + 4, p - 6, p)
        >>> G1.var
        p
        >>> G2 = TransferFunction(0, s - 5, s)
        >>> G2.var
        s

        """
        # 返回第三个参数，即传递函数中的变量
        return self.args[2]

    def _eval_subs(self, old, new):
        # 使用新的变量替换传递函数的分子和分母
        arg_num = self.num.subs(old, new)
        arg_den = self.den.subs(old, new)
        # 返回一个新的传递函数对象，使用新的变量
        argnew = TransferFunction(arg_num, arg_den, self.var)
        return self if old == self.var else argnew

    def _eval_evalf(self, prec):
        # 对传递函数进行数值求解
        return TransferFunction(
            self.num._eval_evalf(prec),
            self.den._eval_evalf(prec),
            self.var)

    def _eval_simplify(self, **kwargs):
        # 简化传递函数，将其化简为最简形式
        tf = cancel(Mul(self.num, 1/self.den, evaluate=False), expand=False).as_numer_denom()
        num_, den_ = tf[0], tf[1]
        return TransferFunction(num_, den_, self.var)

    def _eval_rewrite_as_StateSpace(self, *args):
        """
        返回等效的状态空间模型，用于传递函数模型。
        状态空间模型将以可控制的规范形式返回。

        与状态空间到传递函数模型的转换不同，传递函数到状态空间模型的转换并不唯一。
        对于给定的传递函数模型，可能存在多个状态空间表示。

        示例
        ========

        >>> from sympy.abc import s
        >>> from sympy.physics.control import TransferFunction, StateSpace
        >>> tf = TransferFunction(s**2 + 1, s**3 + 2*s + 10, s)
        >>> tf.rewrite(StateSpace)
        StateSpace(Matrix([
        [  0,  1, 0],
        [  0,  0, 1],
        [-10, -2, 0]]), Matrix([
        [0],
        [0],
        [1]]), Matrix([[1, 0, 1]]), Matrix([[0]]))

        """
        # 如果传递函数不是适当的，抛出值错误
        if not self.is_proper:
            raise ValueError("Transfer Function must be proper.")

        # 获取分子和分母多项式对象
        num_poly = Poly(self.num, self.var)
        den_poly = Poly(self.den, self.var)
        n = den_poly.degree()

        # 获取多项式的系数
        num_coeffs = num_poly.all_coeffs()
        den_coeffs = den_poly.all_coeffs()
        diff = n - num_poly.degree()
        num_coeffs = [0]*diff + num_coeffs

        # 构建状态空间的 A 矩阵
        a = den_coeffs[1:]
        a_mat = Matrix([[(-1)*coefficient/den_coeffs[0] for coefficient in reversed(a)]])
        vert = zeros(n-1, 1)
        mat = eye(n-1)
        A = vert.row_join(mat)
        A = A.col_join(a_mat)

        # 构建状态空间的 B 矩阵
        B = zeros(n, 1)
        B[n-1] = 1

        # 构建状态空间的 C 矩阵
        i = n
        C = []
        while(i > 0):
            C.append(num_coeffs[i] - den_coeffs[i]*num_coeffs[0])
            i -= 1
        C = Matrix([C])

        # 构建状态空间的 D 矩阵
        D = Matrix([num_coeffs[0]])

        return StateSpace(A, B, C, D)
    def expand(self):
        """
        返回分子和分母展开形式的传递函数。

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction
        >>> G1 = TransferFunction((a - s)**2, (s**2 + a)**2, s)
        >>> G1.expand()
        TransferFunction(a**2 - 2*a*s + s**2, a**2 + 2*a*s**2 + s**4, s)
        >>> G2 = TransferFunction((p + 3*b)*(p - b), (p - b)*(p + 2*b), p)
        >>> G2.expand()
        TransferFunction(-3*b**2 + 2*b*p + p**2, -2*b**2 + b*p + p**2, p)

        """
        # 使用 sympy 中的 expand 函数分别展开分子和分母
        return TransferFunction(expand(self.num), expand(self.den), self.var)

    def dc_gain(self):
        """
        计算当频率趋近于零时响应的增益。

        对于纯积分器系统，直流增益为无穷大。

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction
        >>> tf1 = TransferFunction(s + 3, s**2 - 9, s)
        >>> tf1.dc_gain()
        -1/3
        >>> tf2 = TransferFunction(p**2, p - 3 + p**3, p)
        >>> tf2.dc_gain()
        0
        >>> tf3 = TransferFunction(a*p**2 - b, s + b, s)
        >>> tf3.dc_gain()
        (a*p**2 - b)/b
        >>> tf4 = TransferFunction(1, s, s)
        >>> tf4.dc_gain()
        oo

        """
        # 计算传递函数的分子乘以分母的倒数，并在频率趋近于零时求极限
        m = Mul(self.num, Pow(self.den, -1, evaluate=False), evaluate=False)
        return limit(m, self.var, 0)

    def poles(self):
        """
        返回传递函数的极点。

        Examples
        ========

        >>> from sympy.abc import s, p, a
        >>> from sympy.physics.control.lti import TransferFunction
        >>> tf1 = TransferFunction((p + 3)*(p - 1), (p - 1)*(p + 5), p)
        >>> tf1.poles()
        [-5, 1]
        >>> tf2 = TransferFunction((1 - s)**2, (s**2 + 1)**2, s)
        >>> tf2.poles()
        [I, I, -I, -I]
        >>> tf3 = TransferFunction(s**2, a*s + p, s)
        >>> tf3.poles()
        [-p/a]

        """
        # 使用 sympy 中的 Poly 类和 _roots 函数计算传递函数的极点
        return _roots(Poly(self.den, self.var), self.var)

    def zeros(self):
        """
        返回传递函数的零点。

        Examples
        ========

        >>> from sympy.abc import s, p, a
        >>> from sympy.physics.control.lti import TransferFunction
        >>> tf1 = TransferFunction((p + 3)*(p - 1), (p - 1)*(p + 5), p)
        >>> tf1.zeros()
        [-3, 1]
        >>> tf2 = TransferFunction((1 - s)**2, (s**2 + 1)**2, s)
        >>> tf2.zeros()
        [1, 1]
        >>> tf3 = TransferFunction(s**2, a*s + p, s)
        >>> tf3.zeros()
        [0, 0]

        """
        # 使用 sympy 中的 Poly 类和 _roots 函数计算传递函数的零点
        return _roots(Poly(self.num, self.var), self.var)
    def eval_frequency(self, other):
        """
        Returns the system response at any point in the real or complex plane.

        Examples
        ========

        >>> from sympy.abc import s, p, a
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy import I
        >>> tf1 = TransferFunction(1, s**2 + 2*s + 1, s)
        >>> omega = 0.1
        >>> tf1.eval_frequency(I*omega)
        1/(0.99 + 0.2*I)
        >>> tf2 = TransferFunction(s**2, a*s + p, s)
        >>> tf2.eval_frequency(2)
        4/(2*a + p)
        >>> tf2.eval_frequency(I*2)
        -4/(2*I*a + p)
        """
        # 计算传递函数在给定复数或实数位置的系统响应的数值表达式
        arg_num = self.num.subs(self.var, other)  # 替换传递函数的分子中的变量
        arg_den = self.den.subs(self.var, other)  # 替换传递函数的分母中的变量
        argnew = TransferFunction(arg_num, arg_den, self.var).to_expr()  # 将替换后的分子和分母重新组合成传递函数表达式
        return argnew.expand()  # 展开并返回表达式

    def is_stable(self):
        """
        Returns True if the transfer function is asymptotically stable; else False.

        This would not check the marginal or conditional stability of the system.

        Examples
        ========

        >>> from sympy.abc import s, p, a
        >>> from sympy import symbols
        >>> from sympy.physics.control.lti import TransferFunction
        >>> q, r = symbols('q, r', negative=True)
        >>> tf1 = TransferFunction((1 - s)**2, (s + 1)**2, s)
        >>> tf1.is_stable()
        True
        >>> tf2 = TransferFunction((1 - p)**2, (s**2 + 1)**2, s)
        >>> tf2.is_stable()
        False
        >>> tf3 = TransferFunction(4, q*s - r, s)
        >>> tf3.is_stable()
        False
        >>> tf4 = TransferFunction(p + 1, a*p - s**2, p)
        >>> tf4.is_stable() is None   # Not enough info about the symbols to determine stability
        True

        """
        # 检查传递函数是否渐近稳定，并返回布尔值 True 或 False
        return fuzzy_and(pole.as_real_imag()[0].is_negative for pole in self.poles())

    def __add__(self, other):
        if isinstance(other, (TransferFunction, Series)):
            if not self.var == other.var:
                raise ValueError(filldedent("""
                    All the transfer functions should use the same complex variable
                    of the Laplace transform."""))
            return Parallel(self, other)  # 返回两个传递函数并联后的结果
        elif isinstance(other, Parallel):
            if not self.var == other.var:
                raise ValueError(filldedent("""
                    All the transfer functions should use the same complex variable
                    of the Laplace transform."""))
            arg_list = list(other.args)
            return Parallel(self, *arg_list)  # 返回当前传递函数和其他并联传递函数的结果
        else:
            raise ValueError("TransferFunction cannot be added with {}.".format(type(other)))  # 抛出错误，无法与其他类型进行加法运算

    def __radd__(self, other):
        return self + other  # 反向加法运算，调用正向加法运算逻辑
    # 定义减法运算符重载函数，用于处理对象和另一个对象之间的减法操作
    def __sub__(self, other):
        # 如果 `other` 是 `TransferFunction` 或 `Series` 的实例
        if isinstance(other, (TransferFunction, Series)):
            # 检查两个对象是否使用相同的复变量
            if not self.var == other.var:
                raise ValueError(filldedent("""
                    All the transfer functions should use the same complex variable
                    of the Laplace transform."""))
            # 返回一个新的 `Parallel` 对象，包含自身和 `other` 的负值
            return Parallel(self, -other)
        # 如果 `other` 是 `Parallel` 的实例
        elif isinstance(other, Parallel):
            # 检查两个对象是否使用相同的复变量
            if not self.var == other.var:
                raise ValueError(filldedent("""
                    All the transfer functions should use the same complex variable
                    of the Laplace transform."""))
            # 创建一个列表，包含 `other` 的所有参数的负值
            arg_list = [-i for i in list(other.args)]
            # 返回一个新的 `Parallel` 对象，包含自身和 `other` 参数的负值
            return Parallel(self, *arg_list)
        else:
            # 如果 `other` 类型不匹配，抛出异常
            raise ValueError("{} cannot be subtracted from a TransferFunction."
                .format(type(other)))

    # 定义右向减法运算符重载函数，用于处理对象和自身之间的减法操作
    def __rsub__(self, other):
        # 返回自身对象的负值与 `other` 的和
        return -self + other

    # 定义乘法运算符重载函数，用于处理对象和另一个对象之间的乘法操作
    def __mul__(self, other):
        # 如果 `other` 是 `TransferFunction` 或 `Parallel` 的实例
        if isinstance(other, (TransferFunction, Parallel)):
            # 检查两个对象是否使用相同的复变量
            if not self.var == other.var:
                raise ValueError(filldedent("""
                    All the transfer functions should use the same complex variable
                    of the Laplace transform."""))
            # 返回一个新的 `Series` 对象，包含自身和 `other` 的乘积
            return Series(self, other)
        # 如果 `other` 是 `Series` 的实例
        elif isinstance(other, Series):
            # 检查两个对象是否使用相同的复变量
            if not self.var == other.var:
                raise ValueError(filldedent("""
                    All the transfer functions should use the same complex variable
                    of the Laplace transform."""))
            # 创建一个列表，包含 `other` 的所有参数
            arg_list = list(other.args)
            # 返回一个新的 `Series` 对象，包含自身和 `other` 的所有参数
            return Series(self, *arg_list)
        else:
            # 如果 `other` 类型不匹配，抛出异常
            raise ValueError("TransferFunction cannot be multiplied with {}."
                .format(type(other)))

    # 右向乘法运算符重载函数与 `__mul__` 相同
    __rmul__ = __mul__
    def __truediv__(self, other):
        # 如果 `other` 是 TransferFunction 类型
        if isinstance(other, TransferFunction):
            # 检查是否使用相同的复变量进行 Laplace 变换
            if not self.var == other.var:
                raise ValueError(filldedent("""
                    All the transfer functions should use the same complex variable
                    of the Laplace transform."""))
            # 返回一个 Series 对象，表示两个传递函数的串联
            return Series(self, TransferFunction(other.den, other.num, self.var))
        # 如果 `other` 是 Parallel 类型且包含两个参数
        elif (isinstance(other, Parallel) and len(other.args
                ) == 2 and isinstance(other.args[0], TransferFunction)
            and isinstance(other.args[1], (Series, TransferFunction))):
            # 检查是否使用相同的复变量进行 Laplace 变换
            if not self.var == other.var:
                raise ValueError(filldedent("""
                    Both TransferFunction and Parallel should use the
                    same complex variable of the Laplace transform."""))
            # 如果 `other.args[1]` 是 `self`，表示控制系统的单位反馈
            if other.args[1] == self:
                return Feedback(self, other.args[0])
            # 获取 `other.args[1]` 的参数列表
            other_arg_list = list(other.args[1].args) if isinstance(
                other.args[1], Series) else other.args[1]
            # 如果 `other_arg_list` 与 `other.args[1]` 相等
            if other_arg_list == other.args[1]:
                return Feedback(self, other_arg_list)
            # 如果 `self` 存在于 `other_arg_list` 中，则移除它
            elif self in other_arg_list:
                other_arg_list.remove(self)
            else:
                # 返回 `self` 与 `Series` 对象的串联
                return Feedback(self, Series(*other_arg_list))

            # 如果 `other_arg_list` 只有一个元素
            if len(other_arg_list) == 1:
                return Feedback(self, *other_arg_list)
            else:
                # 返回 `self` 与 `Series` 对象的串联
                return Feedback(self, Series(*other_arg_list))
        else:
            # 如果 `other` 不符合上述任何条件，抛出异常
            raise ValueError("TransferFunction cannot be divided by {}.".
                format(type(other)))

    __rtruediv__ = __truediv__

    def __pow__(self, p):
        # 将指数 `p` 转换为符号表达式
        p = sympify(p)
        # 如果 `p` 不是整数，抛出异常
        if not p.is_Integer:
            raise ValueError("Exponent must be an integer.")
        # 如果 `p` 是零
        if p is S.Zero:
            # 返回一个单位传递函数对象
            return TransferFunction(1, 1, self.var)
        # 如果 `p` 大于零
        elif p > 0:
            # 分子和分母的指数分别为 `p`
            num_, den_ = self.num**p, self.den**p
        else:
            # 取 `p` 的绝对值
            p = abs(p)
            # 分子和分母的指数分别为 `p`
            num_, den_ = self.den**p, self.num**p

        # 返回一个根据指数 `p` 计算得到的传递函数对象
        return TransferFunction(num_, den_, self.var)

    def __neg__(self):
        # 返回当前传递函数的负数
        return TransferFunction(-self.num, self.den, self.var)

    @property
    def is_proper(self):
        """
        Returns True if degree of the numerator polynomial is less than
        or equal to degree of the denominator polynomial, else False.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction
        >>> tf1 = TransferFunction(b*s**2 + p**2 - a*p + s, b - p**2, s)
        >>> tf1.is_proper
        False
        >>> tf2 = TransferFunction(p**2 - 4*p, p**3 + 3*p + 2, p)
        >>> tf2.is_proper
        True

        """
        # 返回分子多项式的次数是否小于等于分母多项式的次数
        return degree(self.num, self.var) <= degree(self.den, self.var)

    @property
    def is_strictly_proper(self):
        """
        Returns True if degree of the numerator polynomial is strictly less
        than degree of the denominator polynomial, else False.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction
        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
        >>> tf1.is_strictly_proper
        False
        >>> tf2 = TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)
        >>> tf2.is_strictly_proper
        True

        """
        # 检查分子多项式的次数是否严格小于分母多项式的次数，并返回结果
        return degree(self.num, self.var) < degree(self.den, self.var)

    @property
    def is_biproper(self):
        """
        Returns True if degree of the numerator polynomial is equal to
        degree of the denominator polynomial, else False.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction
        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
        >>> tf1.is_biproper
        True
        >>> tf2 = TransferFunction(p**2, p + a, p)
        >>> tf2.is_biproper
        False

        """
        # 检查分子多项式的次数是否等于分母多项式的次数，并返回结果
        return degree(self.num, self.var) == degree(self.den, self.var)

    def to_expr(self):
        """
        Converts a ``TransferFunction`` object to SymPy Expr.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy import Expr
        >>> tf1 = TransferFunction(s, a*s**2 + 1, s)
        >>> tf1.to_expr()
        s/(a*s**2 + 1)
        >>> isinstance(_, Expr)
        True
        >>> tf2 = TransferFunction(1, (p + 3*b)*(b - p), p)
        >>> tf2.to_expr()
        1/((b - p)*(3*b + p))
        >>> tf3 = TransferFunction((s - 2)*(s - 3), (s - 1)*(s - 2)*(s - 3), s)
        >>> tf3.to_expr()
        ((s - 3)*(s - 2))/(((s - 3)*(s - 2)*(s - 1)))

        """
        # 如果分子不是常数1，则返回分子乘以分母的倒数；否则返回分母的倒数
        if self.num != 1:
            return Mul(self.num, Pow(self.den, -1, evaluate=False), evaluate=False)
        else:
            return Pow(self.den, -1, evaluate=False)
# 定义一个函数 `_flatten_args`，用于扁平化参数列表
def _flatten_args(args, _cls):
    # 创建临时空列表，用于存放处理后的参数
    temp_args = []
    # 遍历参数列表中的每个参数
    for arg in args:
        # 如果参数是指定类别 `_cls` 的实例
        if isinstance(arg, _cls):
            # 将该参数的子参数展开后添加到临时列表中
            temp_args.extend(arg.args)
        else:
            # 如果参数不是指定类别的实例，直接添加到临时列表中
            temp_args.append(arg)
    # 将临时列表转换为元组并返回
    return tuple(temp_args)


# 定义一个函数 `_dummify_args`，用于生成虚拟参数列表和映射字典
def _dummify_args(_arg, var):
    # 创建空字典，用于存放虚拟参数与实际参数的映射关系
    dummy_dict = {}
    # 创建空列表，用于存放生成的虚拟参数列表
    dummy_arg_list = []

    # 遍历传入的参数列表 `_arg` 中的每个参数
    for arg in _arg:
        # 创建一个新的虚拟参数对象 `_s`
        _s = Dummy()
        # 将虚拟参数 `_s` 与实际参数 `var` 建立映射关系，并添加到映射字典中
        dummy_dict[_s] = var
        # 使用虚拟参数 `_s` 替换参数列表中的实际参数 `var`，生成虚拟参数并添加到虚拟参数列表中
        dummy_arg = arg.subs({var: _s})
        dummy_arg_list.append(dummy_arg)

    # 返回生成的虚拟参数列表和映射字典
    return dummy_arg_list, dummy_dict


# 定义一个类 `Series`，继承自 `SISOLinearTimeInvariant`
class Series(SISOLinearTimeInvariant):
    # 表示一个由 SISO 系统组成的串联配置的类

    # 构造方法，用于初始化对象
    def __init__(self, *args, evaluate=False):
        # 调用父类的构造方法
        super().__init__(*args)
        # 设置是否进行求值的标志
        self.evaluate = evaluate

    # 文档字符串，用于描述类的作用、参数及异常
    r"""
    A class for representing a series configuration of SISO systems.

    Parameters
    ==========

    args : SISOLinearTimeInvariant
        SISO systems in a series configuration.
    evaluate : Boolean, Keyword
        When passed ``True``, returns the equivalent
        ``Series(*args).doit()``. Set to ``False`` by default.

    Raises
    ======

    ValueError
        When no argument is passed.

        ``var`` attribute is not same for every system.
    TypeError
        Any of the passed ``*args`` has unsupported type

        A combination of SISO and MIMO systems is
        passed. There should be homogeneity in the
        type of systems passed, SISO in this case.
    """

    # 示例和说明见下面的代码和注释
    Examples
    ========

    >>> from sympy.abc import s, p, a, b
    >>> from sympy import Matrix
    >>> from sympy.physics.control.lti import TransferFunction, Series, Parallel, StateSpace
    >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
    >>> tf2 = TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)
    >>> tf3 = TransferFunction(p**2, p + s, s)
    >>> S1 = Series(tf1, tf2)
    >>> S1
    Series(TransferFunction(a*p**2 + b*s, -p + s, s), TransferFunction(s**3 - 2, s**4 + 5*s + 6, s))
    >>> S1.var
    s
    >>> S2 = Series(tf2, Parallel(tf3, -tf1))
    >>> S2
    Series(TransferFunction(s**3 - 2, s**4 + 5*s + 6, s), Parallel(TransferFunction(p**2, p + s, s), TransferFunction(-a*p**2 - b*s, -p + s, s)))
    >>> S2.var
    s
    >>> S3 = Series(Parallel(tf1, tf2), Parallel(tf2, tf3))
    >>> S3
    Series(Parallel(TransferFunction(a*p**2 + b*s, -p + s, s), TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)), Parallel(TransferFunction(s**3 - 2, s**4 + 5*s + 6, s), TransferFunction(p**2, p + s, s)))
    >>> S3.var
    s

    You can get the resultant transfer function by using ``.doit()`` method:

    >>> S3 = Series(tf1, tf2, -tf3)
    >>> S3.doit()
    TransferFunction(-p**2*(s**3 - 2)*(a*p**2 + b*s), (-p + s)*(p + s)*(s**4 + 5*s + 6), s)
    >>> S4 = Series(tf2, Parallel(tf1, -tf3))
    >>> S4.doit()
    TransferFunction((s**3 - 2)*(-p**2*(-p + s) + (p + s)*(a*p**2 + b*s)), (-p + s)*(p + s)*(s**4 + 5*s + 6), s)

    You can also connect StateSpace which results in SISO

    >>> A1 = Matrix([[-1]])
    >>> B1 = Matrix([[1]])
    >>> C1 = Matrix([[-1]])
    >>> D1 = Matrix([1])
    >>> A2 = Matrix([[0]])
    >>> B2 = Matrix([[1]])
    >>> C2 = Matrix([[1]])
    >>> D2 = Matrix([[0]])
    >>> ss1 = StateSpace(A1, B1, C1, D1)
    >>> ss2 = StateSpace(A2, B2, C2, D2)
    # 创建 StateSpace 对象 ss2，使用给定的矩阵 A2、B2、C2、D2 初始化

    >>> S5 = Series(ss1, ss2)
    # 创建 Series 对象 S5，将 ss1 和 ss2 进行串联连接

    >>> S5
    # 显示 Series 对象 S5 的描述信息
    Series(StateSpace(Matrix([[-1]]), Matrix([[1]]), Matrix([[-1]]), Matrix([[1]])), StateSpace(Matrix([[0]]), Matrix([[1]]), Matrix([[1]]), Matrix([[0]])))

    >>> S5.doit()
    # 对 Series 对象 S5 执行 doit() 方法，计算其最简形式

    StateSpace(Matrix([
    [-1,  0],
    [-1, 0]]), Matrix([
    [1],
    [1]]), Matrix([[0, 1]]), Matrix([[0]]))
    # 显示 Series 对象 S5 经过 doit() 方法计算后的 StateSpace 表示

    Notes
    =====

    All the transfer functions should use the same complex variable
    ``var`` of the Laplace transform.
    # 所有传递函数应使用相同的复变量 ``var``，用于拉普拉斯变换。

    See Also
    ========

    MIMOSeries, Parallel, TransferFunction, Feedback
    # 参见 MIMOSeries, Parallel, TransferFunction, Feedback 相关内容

    """
    def __new__(cls, *args, evaluate=False):

        args = _flatten_args(args, Series)
        # 将传入的参数扁平化处理，使用 Series 类来处理

        # 对于 StateSpace 的串联连接
        if args and any(isinstance(arg, StateSpace) or (hasattr(arg, 'is_StateSpace_object')
                                            and arg.is_StateSpace_object)for arg in args):
            # 检查是否为 SISO 系统
            if (args[0].num_inputs == 1) and (args[-1].num_outputs == 1):
                # 检查互连性
                for i in range(1, len(args)):
                    if args[i].num_inputs != args[i-1].num_outputs:
                        raise ValueError(filldedent("""Systems with incompatible inputs and outputs
                            cannot be connected in Series."""))
                        # 抛出值错误，说明输入输出不兼容，无法串联连接。
                cls._is_series_StateSpace = True
                # 设定为 StateSpace 的串联连接
            else:
                raise ValueError("To use Series connection for MIMO systems use MIMOSeries instead.")
                # 抛出值错误，说明对于 MIMO 系统，请使用 MIMOSeries。
        else:
            cls._is_series_StateSpace = False
            cls._check_args(args)
            # 检查传入参数的合法性

        obj = super().__new__(cls, *args)
        # 调用父类的 __new__ 方法创建对象

        return obj.doit() if evaluate else obj
        # 如果 evaluate 参数为 True，则执行 doit() 方法后返回对象，否则直接返回对象

    @property
    def var(self):
        """
        Returns the complex variable used by all the transfer functions.

        Examples
        ========

        >>> from sympy.abc import p
        >>> from sympy.physics.control.lti import TransferFunction, Series, Parallel
        >>> G1 = TransferFunction(p**2 + 2*p + 4, p - 6, p)
        >>> G2 = TransferFunction(p, 4 - p, p)
        >>> G3 = TransferFunction(0, p**4 - 1, p)
        >>> Series(G1, G2).var
        p
        >>> Series(-G3, Parallel(G1, G2)).var
        p

        """
        return self.args[0].var
        # 返回被所有传递函数使用的复变量
    # 定义方法 `doit`，用于处理串联连接后的传递函数或状态空间对象
    def doit(self, **hints):
        """
        返回在串联连接后评估得到的传递函数或状态空间对象。

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction, Series
        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
        >>> tf2 = TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)
        >>> Series(tf2, tf1).doit()
        TransferFunction((s**3 - 2)*(a*p**2 + b*s), (-p + s)*(s**4 + 5*s + 6), s)
        >>> Series(-tf1, -tf2).doit()
        TransferFunction((2 - s**3)*(-a*p**2 - b*s), (-p + s)*(s**4 + 5*s + 6), s)

        Notes
        =====

        如果串联连接仅包含传递函数组件，则返回等效的传递函数。但是，如果参数中使用了状态空间对象，
        则返回状态空间对象。

        """
        # 检查系统是否为状态空间对象
        if self._is_series_StateSpace:
            # 返回等效的状态空间模型
            res = self.args[0]
            if not isinstance(res, StateSpace):
                res = res.doit().rewrite(StateSpace)
            for arg in self.args[1:]:
                if not isinstance(arg, StateSpace):
                    arg = arg.doit().rewrite(StateSpace)
                else:
                    arg = arg.doit()
                arg = arg.doit()
                res = arg * res
                return res

        # 提取各参数的分子和分母，计算结果传递函数的分子和分母
        _num_arg = (arg.doit().num for arg in self.args)
        _den_arg = (arg.doit().den for arg in self.args)
        res_num = Mul(*_num_arg, evaluate=True)
        res_den = Mul(*_den_arg, evaluate=True)
        return TransferFunction(res_num, res_den, self.var)

    # 将当前对象重写为传递函数的形式
    def _eval_rewrite_as_TransferFunction(self, *args, **kwargs):
        if self._is_series_StateSpace:
            return self.doit().rewrite(TransferFunction)[0][0]
        return self.doit()

    # 定义并重载二元加法运算符 `__add__`
    @_check_other_SISO
    def __add__(self, other):
        # 如果 `other` 是并联连接对象，则将当前对象与其它参数进行并联连接
        if isinstance(other, Parallel):
            arg_list = list(other.args)
            return Parallel(self, *arg_list)

        # 否则，将当前对象与 `other` 进行并联连接
        return Parallel(self, other)

    # 重载反向加法运算符 `__radd__`，与 `__add__` 相同
    __radd__ = __add__

    # 定义并重载二元减法运算符 `__sub__`
    @_check_other_SISO
    def __sub__(self, other):
        # 返回当前对象与 `other` 的相反数之和
        return self + (-other)

    # 重载反向减法运算符 `__rsub__`
    def __rsub__(self, other):
        # 返回 `other` 与当前对象的相反数之和
        return -self + other

    # 定义并重载二元乘法运算符 `__mul__`
    @_check_other_SISO
    def __mul__(self, other):
        # 将当前对象与 `other` 进行串联连接
        arg_list = list(self.args)
        return Series(*arg_list, other)
    def __truediv__(self, other):
        # 如果除数是 TransferFunction 类型，则返回新的 Series 对象
        if isinstance(other, TransferFunction):
            return Series(*self.args, TransferFunction(other.den, other.num, other.var))
        # 如果除数是 Series 类型，则分别转换成 TransferFunction 后进行除法操作
        elif isinstance(other, Series):
            tf_self = self.rewrite(TransferFunction)
            tf_other = other.rewrite(TransferFunction)
            return tf_self / tf_other
        # 如果除数是 Parallel 类型且符合特定条件，则进行特定处理
        elif (isinstance(other, Parallel) and len(other.args) == 2
            and isinstance(other.args[0], TransferFunction) and isinstance(other.args[1], Series)):

            # 检查变量是否一致，若不一致则抛出异常
            if not self.var == other.var:
                raise ValueError(filldedent("""
                    All the transfer functions should use the same complex variable
                    of the Laplace transform."""))
            # 求出各自参数的差集
            self_arg_list = set(self.args)
            other_arg_list = set(other.args[1].args)
            res = list(self_arg_list ^ other_arg_list)
            # 根据差集的长度返回不同的 Feedback 对象
            if len(res) == 0:
                return Feedback(self, other.args[0])
            elif len(res) == 1:
                return Feedback(self, *res)
            else:
                return Feedback(self, Series(*res))
        else:
            # 若不符合以上条件则抛出异常
            raise ValueError("This transfer function expression is invalid.")

    def __neg__(self):
        # 返回当前 Series 对象的负值
        return Series(TransferFunction(-1, 1, self.var), self)

    def to_expr(self):
        """Returns the equivalent ``Expr`` object."""
        # 将当前 Series 对象转换为 SymPy 的表达式对象
        return Mul(*(arg.to_expr() for arg in self.args), evaluate=False)

    @property
    def is_proper(self):
        """
        Returns True if degree of the numerator polynomial of the resultant transfer
        function is less than or equal to degree of the denominator polynomial of
        the same, else False.
        
        返回结果转移函数的分子多项式的次数是否小于或等于其分母多项式的次数，是则返回 True，否则返回 False。

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction, Series
        >>> tf1 = TransferFunction(b*s**2 + p**2 - a*p + s, b - p**2, s)
        >>> tf2 = TransferFunction(p**2 - 4*p, p**3 + 3*s + 2, s)
        >>> tf3 = TransferFunction(s, s**2 + s + 1, s)
        >>> S1 = Series(-tf2, tf1)
        >>> S1.is_proper
        False
        >>> S2 = Series(tf1, tf2, tf3)
        >>> S2.is_proper
        True

        """
        # 调用 doit 方法并检查结果的 proper 属性
        return self.doit().is_proper
    def is_strictly_proper(self):
        """
        Returns True if degree of the numerator polynomial of the resultant transfer
        function is strictly less than degree of the denominator polynomial of
        the same, else False.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction, Series
        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
        >>> tf2 = TransferFunction(s**3 - 2, s**2 + 5*s + 6, s)
        >>> tf3 = TransferFunction(1, s**2 + s + 1, s)
        >>> S1 = Series(tf1, tf2)
        >>> S1.is_strictly_proper
        False
        >>> S2 = Series(tf1, tf2, tf3)
        >>> S2.is_strictly_proper
        True

        """
        # 调用 doit() 方法计算结果，然后检查是否是严格适当的传递函数
        return self.doit().is_strictly_proper

    @property
    def is_biproper(self):
        r"""
        Returns True if degree of the numerator polynomial of the resultant transfer
        function is equal to degree of the denominator polynomial of
        the same, else False.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction, Series
        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
        >>> tf2 = TransferFunction(p, s**2, s)
        >>> tf3 = TransferFunction(s**2, 1, s)
        >>> S1 = Series(tf1, -tf2)
        >>> S1.is_biproper
        False
        >>> S2 = Series(tf2, tf3)
        >>> S2.is_biproper
        True

        """
        # 调用 doit() 方法计算结果，然后检查是否是双适当的传递函数
        return self.doit().is_biproper

    @property
    def is_StateSpace_object(self):
        # 返回对象是否是串联状态空间对象的布尔值
        return self._is_series_StateSpace
# 定义一个函数 `_mat_mul_compatible`，用于检查多个对象的输出和输入是否兼容以进行矩阵乘法
def _mat_mul_compatible(*args):
    """To check whether shapes are compatible for matrix mul."""
    # 返回所有相邻对象的输出数量是否等于下一个对象的输入数量
    return all(args[i].num_outputs == args[i+1].num_inputs for i in range(len(args)-1))


# 定义一个类 `MIMOSeries`，用于表示一系列 MIMO 系统的配置，继承自 `MIMOLinearTimeInvariant`
class MIMOSeries(MIMOLinearTimeInvariant):
    """
    A class for representing a series configuration of MIMO systems.

    Parameters
    ==========

    args : MIMOLinearTimeInvariant
        MIMO systems in a series configuration.
    evaluate : Boolean, Keyword
        When passed ``True``, returns the equivalent
        ``MIMOSeries(*args).doit()``. Set to ``False`` by default.

    Raises
    ======

    ValueError
        When no argument is passed.

        ``var`` attribute is not same for every system.

        ``num_outputs`` of the MIMO system is not equal to the
        ``num_inputs`` of its adjacent MIMO system. (Matrix
        multiplication constraint, basically)
    TypeError
        Any of the passed ``*args`` has unsupported type

        A combination of SISO and MIMO systems is
        passed. There should be homogeneity in the
        type of systems passed, MIMO in this case.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import MIMOSeries, TransferFunctionMatrix, StateSpace
    >>> from sympy import Matrix, pprint
    >>> mat_a = Matrix([[5*s], [5]])  # 2 Outputs 1 Input
    >>> mat_b = Matrix([[5, 1/(6*s**2)]])  # 1 Output 2 Inputs
    >>> mat_c = Matrix([[1, s], [5/s, 1]])  # 2 Outputs 2 Inputs
    >>> tfm_a = TransferFunctionMatrix.from_Matrix(mat_a, s)
    >>> tfm_b = TransferFunctionMatrix.from_Matrix(mat_b, s)
    >>> tfm_c = TransferFunctionMatrix.from_Matrix(mat_c, s)
    >>> MIMOSeries(tfm_c, tfm_b, tfm_a)
    MIMOSeries(TransferFunctionMatrix(((TransferFunction(1, 1, s), TransferFunction(s, 1, s)), (TransferFunction(5, s, s), TransferFunction(1, 1, s)))), TransferFunctionMatrix(((TransferFunction(5, 1, s), TransferFunction(1, 6*s**2, s)),)), TransferFunctionMatrix(((TransferFunction(5*s, 1, s),), (TransferFunction(5, 1, s),))))
    >>> pprint(_, use_unicode=False)  #  For Better Visualization
    [5*s]                 [1  s]
    [---]    [5   1  ]    [-  -]
    [ 1 ]    [-  ----]    [1  1]
    [   ]   *[1     2]   *[    ]
    [ 5 ]    [   6*s ]{t} [5  1]
    [ - ]                 [-  -]
    [ 1 ]{t}              [s  1]{t}
    >>> MIMOSeries(tfm_c, tfm_b, tfm_a).doit()
    TransferFunctionMatrix(((TransferFunction(150*s**4 + 25*s, 6*s**3, s), TransferFunction(150*s**4 + 5*s, 6*s**2, s)), (TransferFunction(150*s**3 + 25, 6*s**3, s), TransferFunction(150*s**3 + 5, 6*s**2, s))))
    >>> pprint(_, use_unicode=False)  # (2 Inputs -A-> 2 Outputs) -> (2 Inputs -B-> 1 Output) -> (1 Input -C-> 2 Outputs) is equivalent to (2 Inputs -Series Equivalent-> 2 Outputs).
    [     4              4      ]
    [150*s  + 25*s  150*s  + 5*s]
    [-------------  ------------]
    [        3             2    ]
    [     6*s           6*s     ]
    [                           ]
    """

    # 初始化方法，接受一系列 MIMO 系统对象和一个可选的布尔参数 evaluate，默认为 False
    def __init__(self, *args, evaluate=False):
        # 调用父类的初始化方法
        super().__init__(*args)
        # 如果 evaluate 参数为 True，则调用 doit() 方法
        if evaluate:
            self.doit()

    # 实现 doit 方法，返回进行系列等效处理后的 TransferFunctionMatrix 对象
    def doit(self):
        # 检查传入的 MIMO 系统列表是否符合矩阵乘法的要求
        if not _mat_mul_compatible(*self.args):
            # 如果不符合，抛出 ValueError 异常
            raise ValueError("num_outputs of the MIMO system is not equal to the num_inputs of its adjacent MIMO system.")
        
        # 返回处理后的 TransferFunctionMatrix 对象
        return TransferFunctionMatrix(*self.args)
    """
    Define the MIMOSeries class constructor with optional evaluation parameter.

    Parameters
    ==========
    cls : class
        The class object of MIMOSeries.
    *args : StateSpace
        Variable length argument list, each element being a StateSpace object or convertible to it.
    evaluate : bool, optional
        Flag indicating whether to evaluate the resulting object immediately (default is False).

    Returns
    =======
    obj : MIMOSeries
        A new instance of MIMOSeries representing the series connection of StateSpace systems.

    Raises
    ======
    ValueError
        If the arguments are incompatible in terms of input and output connections between systems.

    Notes
    =====
    All the transfer function matrices should use the same complex variable ``var`` of the Laplace transform.

    ``MIMOSeries(A, B)`` is not equivalent to ``A*B``. It is always in the reverse order, that is ``B*A``.

    See Also
    ========
    Series, MIMOParallel
    """
    def __new__(cls, *args, evaluate=False):

        # Check if any argument is a StateSpace object or can be converted to it
        if args and any(isinstance(arg, StateSpace) or (hasattr(arg, 'is_StateSpace_object')
                                            and arg.is_StateSpace_object) for arg in args):
            # Check compatibility of inputs and outputs among connected systems
            for i in range(1, len(args)):
                if args[i].num_inputs != args[i - 1].num_outputs:
                    raise ValueError(filldedent("""Systems with incompatible inputs and outputs
                        cannot be connected in MIMOSeries."""))
            # Create a new instance of the class with the provided arguments
            obj = super().__new__(cls, *args)
            cls._is_series_StateSpace = True
        else:
            # Check if arguments are valid for matrix multiplication
            cls._check_args(args)
            cls._is_series_StateSpace = False

            if _mat_mul_compatible(*args):
                # Create a new instance of the class with the provided arguments
                obj = super().__new__(cls, *args)
            else:
                # Raise an error if the number of input and output signals are not compatible
                raise ValueError(filldedent("""
                    Number of input signals do not match the number
                    of output signals of adjacent systems for some args."""))

        # Evaluate the resulting object if requested
        return obj.doit() if evaluate else obj
    # 返回用于所有传递函数的复杂变量。
    
    Examples
    ========
    示例
    
    >>> from sympy.abc import p
    >>> from sympy.physics.control.lti import TransferFunction, MIMOSeries, TransferFunctionMatrix
    >>> G1 = TransferFunction(p**2 + 2*p + 4, p - 6, p)
    >>> G2 = TransferFunction(p, 4 - p, p)
    >>> G3 = TransferFunction(0, p**4 - 1, p)
    >>> tfm_1 = TransferFunctionMatrix([[G1, G2, G3]])
    >>> tfm_2 = TransferFunctionMatrix([[G1], [G2], [G3]])
    >>> MIMOSeries(tfm_2, tfm_1).var
    p
    
    @property
    def var(self):
        """
        返回所有传递函数使用的复杂变量。
    
        示例
        ========
    
        >>> from sympy.abc import p
        >>> from sympy.physics.control.lti import TransferFunction, MIMOSeries, TransferFunctionMatrix
        >>> G1 = TransferFunction(p**2 + 2*p + 4, p - 6, p)
        >>> G2 = TransferFunction(p, 4 - p, p)
        >>> G3 = TransferFunction(0, p**4 - 1, p)
        >>> tfm_1 = TransferFunctionMatrix([[G1, G2, G3]])
        >>> tfm_2 = TransferFunctionMatrix([[G1], [G2], [G3]])
        >>> MIMOSeries(tfm_2, tfm_1).var
        p
    
        """
        return self.args[0].var
    
    @property
    def num_inputs(self):
        """返回串联系统的输入信号数量。"""
        return self.args[0].num_inputs
    
    @property
    def num_outputs(self):
        """返回串联系统的输出信号数量。"""
        return self.args[-1].num_outputs
    
    @property
    def shape(self):
        """返回等效 MIMO 系统的形状。"""
        return self.num_outputs, self.num_inputs
    
    @property
    def is_StateSpace_object(self):
        """返回是否为状态空间对象的布尔值。"""
        return self._is_series_StateSpace
    def doit(self, cancel=False, **kwargs):
        """
        返回在串联配置中评估多输入多输出系统后得到的结果。对于传递函数系统，返回传递函数矩阵，
        对于状态空间系统，返回相应的状态空间系统。

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction, MIMOSeries, TransferFunctionMatrix
        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
        >>> tf2 = TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)
        >>> tfm1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf2]])
        >>> tfm2 = TransferFunctionMatrix([[tf2, tf1], [tf1, tf1]])
        >>> MIMOSeries(tfm2, tfm1).doit()
        TransferFunctionMatrix(((TransferFunction(2*(-p + s)*(s**3 - 2)*(a*p**2 + b*s)*(s**4 + 5*s + 6), (-p + s)**2*(s**4 + 5*s + 6)**2, s), TransferFunction((-p + s)**2*(s**3 - 2)*(a*p**2 + b*s) + (-p + s)*(a*p**2 + b*s)**2*(s**4 + 5*s + 6), (-p + s)**3*(s**4 + 5*s + 6), s)), (TransferFunction((-p + s)*(s**3 - 2)**2*(s**4 + 5*s + 6) + (s**3 - 2)*(a*p**2 + b*s)*(s**4 + 5*s + 6)**2, (-p + s)*(s**4 + 5*s + 6)**3, s), TransferFunction(2*(s**3 - 2)*(a*p**2 + b*s), (-p + s)*(s**4 + 5*s + 6), s))))

        """
        if self._is_series_StateSpace:
            # 如果是串联的状态空间模型，返回等效的状态空间模型
            res = self.args[0]
            if not isinstance(res, StateSpace):
                res = res.doit().rewrite(StateSpace)
            for arg in self.args[1:]:
                if not isinstance(arg, StateSpace):
                    arg = arg.doit().rewrite(StateSpace)
                else:
                    arg = arg.doit()
                res = arg * res
            return res

        _arg = (arg.doit()._expr_mat for arg in reversed(self.args))

        if cancel:
            # 如果需要取消操作，计算并返回相乘的矩阵
            res = MatMul(*_arg, evaluate=True)
            return TransferFunctionMatrix.from_Matrix(res, self.var)

        _dummy_args, _dummy_dict = _dummify_args(_arg, self.var)
        # 计算并返回替换后的传递函数矩阵
        res = MatMul(*_dummy_args, evaluate=True)
        temp_tfm = TransferFunctionMatrix.from_Matrix(res, self.var)
        return temp_tfm.subs(_dummy_dict)

    def _eval_rewrite_as_TransferFunctionMatrix(self, *args, **kwargs):
        if self._is_series_StateSpace:
            # 如果是串联的状态空间模型，以传递函数形式重写并返回结果
            return self.doit().rewrite(TransferFunction)
        return self.doit()

    @_check_other_MIMO
    def __add__(self, other):
        if isinstance(other, MIMOParallel):
            # 如果其他操作数是并联的多输入多输出系统，将自身与其它操作数并联并返回
            arg_list = list(other.args)
            return MIMOParallel(self, *arg_list)
        return MIMOParallel(self, other)

    __radd__ = __add__

    @_check_other_MIMO
    def __sub__(self, other):
        # 重载减法运算符，返回自身与负操作数的和
        return self + (-other)

    def __rsub__(self, other):
        # 重载右侧减法运算符，返回负自身与操作数的和
        return -self + other

    @_check_other_MIMO
    # 定义 MIMOSeries 类的乘法运算符重载方法
    def __mul__(self, other):
        # 如果 other 是 MIMOSeries 类型的对象
        if isinstance(other, MIMOSeries):
            # 将当前对象的参数转换为列表
            self_arg_list = list(self.args)
            # 将 other 对象的参数转换为列表
            other_arg_list = list(other.args)
            # 返回一个新的 MIMOSeries 对象，参数顺序为 other 的参数后跟 self 的参数
            return MIMOSeries(*other_arg_list, *self_arg_list)  # A*B = MIMOSeries(B, A)
        
        # 如果 other 不是 MIMOSeries 类型的对象，则将当前对象的参数转换为列表
        arg_list = list(self.args)
        # 返回一个新的 MIMOSeries 对象，参数顺序为 other 和当前对象的参数
        return MIMOSeries(other, *arg_list)

    # 定义 MIMOSeries 类的负号运算符重载方法
    def __neg__(self):
        # 将当前对象的参数转换为列表
        arg_list = list(self.args)
        # 将参数列表中的第一个参数取负值
        arg_list[0] = -arg_list[0]
        # 返回一个新的 MIMOSeries 对象，参数不变
        return MIMOSeries(*arg_list)
class Parallel(SISOLinearTimeInvariant):
    r"""
    A class for representing a parallel configuration of SISO systems.

    Parameters
    ==========

    args : SISOLinearTimeInvariant
        SISO systems in a parallel arrangement.
    evaluate : Boolean, Keyword
        When passed ``True``, returns the equivalent
        ``Parallel(*args).doit()``. Set to ``False`` by default.

    Raises
    ======

    ValueError
        When no argument is passed.

        ``var`` attribute is not same for every system.
    TypeError
        Any of the passed ``*args`` has unsupported type

        A combination of SISO and MIMO systems is
        passed. There should be homogeneity in the
        type of systems passed.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import s, p, a, b
    >>> from sympy.physics.control.lti import TransferFunction, Parallel, Series, StateSpace
    >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
    >>> tf2 = TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)
    >>> tf3 = TransferFunction(p**2, p + s, s)
    >>> P1 = Parallel(tf1, tf2)
    >>> P1
    Parallel(TransferFunction(a*p**2 + b*s, -p + s, s), TransferFunction(s**3 - 2, s**4 + 5*s + 6, s))
    >>> P1.var
    s
    >>> P2 = Parallel(tf2, Series(tf3, -tf1))
    >>> P2
    Parallel(TransferFunction(s**3 - 2, s**4 + 5*s + 6, s), Series(TransferFunction(p**2, p + s, s), TransferFunction(-a*p**2 - b*s, -p + s, s)))
    >>> P2.var
    s
    >>> P3 = Parallel(Series(tf1, tf2), Series(tf2, tf3))
    >>> P3
    Parallel(Series(TransferFunction(a*p**2 + b*s, -p + s, s), TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)), Series(TransferFunction(s**3 - 2, s**4 + 5*s + 6, s), TransferFunction(p**2, p + s, s)))
    >>> P3.var
    s

    You can get the resultant transfer function by using ``.doit()`` method:

    >>> Parallel(tf1, tf2, -tf3).doit()
    TransferFunction(-p**2*(-p + s)*(s**4 + 5*s + 6) + (-p + s)*(p + s)*(s**3 - 2) + (p + s)*(a*p**2 + b*s)*(s**4 + 5*s + 6), (-p + s)*(p + s)*(s**4 + 5*s + 6), s)
    >>> Parallel(tf2, Series(tf1, -tf3)).doit()
    TransferFunction(-p**2*(a*p**2 + b*s)*(s**4 + 5*s + 6) + (-p + s)*(p + s)*(s**3 - 2), (-p + s)*(p + s)*(s**4 + 5*s + 6), s)

    Parallel can be used to connect SISO ``StateSpace`` systems together.

    >>> A1 = Matrix([[-1]])
    >>> B1 = Matrix([[1]])
    >>> C1 = Matrix([[-1]])
    >>> D1 = Matrix([1])
    >>> A2 = Matrix([[0]])
    >>> B2 = Matrix([[1]])
    >>> C2 = Matrix([[1]])
    >>> D2 = Matrix([[0]])
    >>> ss1 = StateSpace(A1, B1, C1, D1)
    >>> ss2 = StateSpace(A2, B2, C2, D2)
    >>> P4 = Parallel(ss1, ss2)
    >>> P4
    Parallel(StateSpace(Matrix([[-1]]), Matrix([[1]]), Matrix([[-1]]), Matrix([[1]])), StateSpace(Matrix([[0]]), Matrix([[1]]), Matrix([[1]]), Matrix([[0]])))

    ``doit()`` can be used to find ``StateSpace`` equivalent for the system containing ``StateSpace`` objects.

    >>> P4.doit()
    StateSpace(Matrix([
    [-1, 0],
    [ 0, 0]]), Matrix([
    [1],
    """

    This section defines a class named `TransferFunction` which represents a transfer function in control theory.

    Notes
    =====

    All the transfer functions should use the same complex variable
    ``var`` of the Laplace transform.

    See Also
    ========

    Series, TransferFunction, Feedback

    """

    def __new__(cls, *args, evaluate=False):
        """
        Creates a new instance of TransferFunction.

        Parameters
        ==========

        args : tuple
            Arguments to initialize the TransferFunction.
        evaluate : bool, optional
            If True, evaluates the object immediately.

        Notes
        =====

        This constructor checks for StateSpace parallel connections and handles them appropriately.

        Raises
        ======

        ValueError
            If MIMO (Multiple Input Multiple Output) systems are detected within parallel connections.

        Returns
        =======

        TransferFunction object
            Returns an instance of TransferFunction.

        """
        # Flatten the arguments to handle nested Parallel connections
        args = _flatten_args(args, Parallel)
        
        # Check for StateSpace parallel connection
        if args and any(isinstance(arg, StateSpace) or (hasattr(arg, 'is_StateSpace_object')
                                                        and arg.is_StateSpace_object) for arg in args):
            # Check if all are SISO (Single Input Single Output)
            if all(arg.is_SISO for arg in args):
                cls._is_parallel_StateSpace = True
            else:
                raise ValueError("To use Parallel connection for MIMO systems use MIMOParallel instead.")
        else:
            cls._is_parallel_StateSpace = False
            cls._check_args(args)
        
        # Create the object using the superclass constructor
        obj = super().__new__(cls, *args)

        # Evaluate the object if `evaluate` flag is True
        return obj.doit() if evaluate else obj

    @property
    def var(self):
        """
        Returns the complex variable used by all the transfer functions.

        Examples
        ========

        >>> from sympy.abc import p
        >>> from sympy.physics.control.lti import TransferFunction, Parallel, Series
        >>> G1 = TransferFunction(p**2 + 2*p + 4, p - 6, p)
        >>> G2 = TransferFunction(p, 4 - p, p)
        >>> G3 = TransferFunction(0, p**4 - 1, p)
        >>> Parallel(G1, G2).var
        p
        >>> Parallel(-G3, Series(G1, G2)).var
        p

        """
        return self.args[0].var
    def doit(self, **hints):
        """
        Returns the resultant transfer function or state space obtained by
        parallel connection of transfer functions or state space objects.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction, Parallel
        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
        >>> tf2 = TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)
        >>> Parallel(tf2, tf1).doit()
        TransferFunction((-p + s)*(s**3 - 2) + (a*p**2 + b*s)*(s**4 + 5*s + 6), (-p + s)*(s**4 + 5*s + 6), s)
        >>> Parallel(-tf1, -tf2).doit()
        TransferFunction((2 - s**3)*(-p + s) + (-a*p**2 - b*s)*(s**4 + 5*s + 6), (-p + s)*(s**4 + 5*s + 6), s)

        """
        if self._is_parallel_StateSpace:
            # 如果是并联的状态空间模型，则返回等效的状态空间模型
            res = self.args[0].doit()
            if not isinstance(res, StateSpace):
                res = res.rewrite(StateSpace)
            for arg in self.args[1:]:
                if not isinstance(arg, StateSpace):
                    arg = arg.doit().rewrite(StateSpace)
                res += arg
            return res

        _arg = (arg.doit().to_expr() for arg in self.args)
        res = Add(*_arg).as_numer_denom()
        return TransferFunction(*res, self.var)

    def _eval_rewrite_as_TransferFunction(self, *args, **kwargs):
        if self._is_parallel_StateSpace:
            # 如果是并联的状态空间模型，则调用 doit 方法并转换为传递函数形式
            return self.doit().rewrite(TransferFunction)[0][0]
        return self.doit()

    @_check_other_SISO
    def __add__(self, other):
        # 并联操作的加法重载
        self_arg_list = list(self.args)
        return Parallel(*self_arg_list, other)

    __radd__ = __add__

    @_check_other_SISO
    def __sub__(self, other):
        # 并联操作的减法重载
        return self + (-other)

    def __rsub__(self, other):
        # 反向减法重载
        return -self + other

    @_check_other_SISO
    def __mul__(self, other):
        # 并联操作的乘法重载

        if isinstance(other, Series):
            arg_list = list(other.args)
            return Series(self, *arg_list)

        return Series(self, other)

    def __neg__(self):
        # 负运算的重载
        return Series(TransferFunction(-1, 1, self.var), self)

    def to_expr(self):
        """Returns the equivalent ``Expr`` object."""
        # 返回等效的表达式对象
        return Add(*(arg.to_expr() for arg in self.args), evaluate=False)

    @property
    def is_proper(self):
        """
        Returns True if degree of the numerator polynomial of the resultant transfer
        function is less than or equal to degree of the denominator polynomial of
        the same, else False.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction, Parallel
        >>> tf1 = TransferFunction(b*s**2 + p**2 - a*p + s, b - p**2, s)
        >>> tf2 = TransferFunction(p**2 - 4*p, p**3 + 3*s + 2, s)
        >>> tf3 = TransferFunction(s, s**2 + s + 1, s)
        >>> P1 = Parallel(-tf2, tf1)
        >>> P1.is_proper
        False
        >>> P2 = Parallel(tf2, tf3)
        >>> P2.is_proper
        True

        """
        # 调用 doit 方法计算并返回当前对象的 proper 属性值
        return self.doit().is_proper

    @property
    def is_strictly_proper(self):
        """
        Returns True if degree of the numerator polynomial of the resultant transfer
        function is strictly less than degree of the denominator polynomial of
        the same, else False.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction, Parallel
        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
        >>> tf2 = TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)
        >>> tf3 = TransferFunction(s, s**2 + s + 1, s)
        >>> P1 = Parallel(tf1, tf2)
        >>> P1.is_strictly_proper
        False
        >>> P2 = Parallel(tf2, tf3)
        >>> P2.is_strictly_proper
        True

        """
        # 调用 doit 方法计算并返回当前对象的 strictly_proper 属性值
        return self.doit().is_strictly_proper

    @property
    def is_biproper(self):
        """
        Returns True if degree of the numerator polynomial of the resultant transfer
        function is equal to degree of the denominator polynomial of
        the same, else False.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction, Parallel
        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
        >>> tf2 = TransferFunction(p**2, p + s, s)
        >>> tf3 = TransferFunction(s, s**2 + s + 1, s)
        >>> P1 = Parallel(tf1, -tf2)
        >>> P1.is_biproper
        True
        >>> P2 = Parallel(tf2, tf3)
        >>> P2.is_biproper
        False

        """
        # 调用 doit 方法计算并返回当前对象的 biproper 属性值
        return self.doit().is_biproper

    @property
    def is_StateSpace_object(self):
        # 返回当前对象的 _is_parallel_StateSpace 属性值
        return self._is_parallel_StateSpace
class MIMOParallel(MIMOLinearTimeInvariant):
    r"""
    A class for representing a parallel configuration of MIMO systems.

    Parameters
    ==========

    args : MIMOLinearTimeInvariant
        MIMO Systems in a parallel arrangement.
    evaluate : Boolean, Keyword
        When passed ``True``, returns the equivalent
        ``MIMOParallel(*args).doit()``. Set to ``False`` by default.

    Raises
    ======

    ValueError
        When no argument is passed.

        ``var`` attribute is not same for every system.

        All MIMO systems passed do not have same shape.
    TypeError
        Any of the passed ``*args`` has unsupported type

        A combination of SISO and MIMO systems is
        passed. There should be homogeneity in the
        type of systems passed, MIMO in this case.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunctionMatrix, MIMOParallel, StateSpace
    >>> from sympy import Matrix, pprint
    >>> expr_1 = 1/s
    >>> expr_2 = s/(s**2-1)
    >>> expr_3 = (2 + s)/(s**2 - 1)
    >>> expr_4 = 5
    >>> tfm_a = TransferFunctionMatrix.from_Matrix(Matrix([[expr_1, expr_2], [expr_3, expr_4]]), s)
    >>> tfm_b = TransferFunctionMatrix.from_Matrix(Matrix([[expr_2, expr_1], [expr_4, expr_3]]), s)
    >>> tfm_c = TransferFunctionMatrix.from_Matrix(Matrix([[expr_3, expr_4], [expr_1, expr_2]]), s)

    # Create an instance of MIMOParallel with given TransferFunctionMatrix objects
    >>> MIMOParallel(tfm_a, tfm_b, tfm_c)
    MIMOParallel(TransferFunctionMatrix(((TransferFunction(1, s, s), TransferFunction(s, s**2 - 1, s)), (TransferFunction(s + 2, s**2 - 1, s), TransferFunction(5, 1, s)))), TransferFunctionMatrix(((TransferFunction(s, s**2 - 1, s), TransferFunction(1, s, s)), (TransferFunction(5, 1, s), TransferFunction(s + 2, s**2 - 1, s)))), TransferFunctionMatrix(((TransferFunction(s + 2, s**2 - 1, s), TransferFunction(5, 1, s)), (TransferFunction(1, s, s), TransferFunction(s, s**2 - 1, s)))))

    # Pretty-print the resulting TransferFunctionMatrix for better visualization
    >>> pprint(_, use_unicode=False)

    [  1       s   ]      [  s       1   ]      [s + 2     5   ]
    [  -     ------]      [------    -   ]      [------    -   ]
    [  s      2    ]      [ 2        s   ]      [ 2        1   ]
    [        s  - 1]      [s  - 1        ]      [s  - 1        ]
    [              ]    + [              ]    + [              ]
    [s + 2     5   ]      [  5     s + 2 ]      [  1       s   ]
    [------    -   ]      [  -     ------]      [  -     ------]
    [ 2        1   ]      [  1      2    ]      [  s      2    ]
    [s  - 1        ]{t}   [        s  - 1]{t}   [        s  - 1]{t}

    # Perform the symbolic computation and return the resulting TransferFunctionMatrix
    >>> MIMOParallel(tfm_a, tfm_b, tfm_c).doit()
    TransferFunctionMatrix(((TransferFunction(s**2 + s*(2*s + 2) - 1, s*(s**2 - 1), s), TransferFunction(2*s**2 + 5*s*(s**2 - 1) - 1, s*(s**2 - 1), s)), (TransferFunction(s**2 + s*(s + 2) + 5*s*(s**2 - 1) - 1, s*(s**2 - 1), s), TransferFunction(5*s**2 + 2*s - 3, s**2 - 1, s))))

    # Pretty-print the resulting TransferFunctionMatrix for better visualization
    >>> pprint(_, use_unicode=False)
    # 创建一个并联连接的 MIMO（多输入多输出）系统，使用输入的 StateSpace 对象 ss1 和 ss2
    p1 = MIMOParallel(ss1, ss2)
    
    # 输出 p1 的字符串表示，显示两个 StateSpace 对象的并联连接
    p1
    
    
    
    # 调用 doit() 方法，将 MIMOParallel 对象 p1 转换为一个合适的 StateSpace 对象
    p1.doit()
    
    # 返回一个新的 StateSpace 对象，其描述了原始 MIMOParallel 对象所代表的系统
    def __new__(cls, *args, evaluate=False):
        # 将参数展开为扁平结构，以处理并行连接
        args = _flatten_args(args, MIMOParallel)

        # 对于 StateSpace 并行连接
        if args and any(isinstance(arg, StateSpace) or (hasattr(arg, 'is_StateSpace_object')
                                    and arg.is_StateSpace_object) for arg in args):
            # 检查所有系统的输入和输出是否兼容
            if any(arg.num_inputs != args[0].num_inputs or arg.num_outputs != args[0].num_outputs
                   for arg in args[1:]):
                raise ShapeError("Systems with incompatible inputs and outputs cannot be "
                                 "connected in MIMOParallel.")
            cls._is_parallel_StateSpace = True
        else:
            # 检查参数的形状是否相等
            cls._check_args(args)
            if any(arg.shape != args[0].shape for arg in args):
                raise TypeError("Shape of all the args is not equal.")
            cls._is_parallel_StateSpace = False
        # 调用父类的构造方法创建对象
        obj = super().__new__(cls, *args)
        
        # 如果 evaluate 为 True，则立即计算结果
        return obj.doit() if evaluate else obj

    @property
    def var(self):
        """
        Returns the complex variable used by all the systems.

        Examples
        ========

        >>> from sympy.abc import p
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, MIMOParallel
        >>> G1 = TransferFunction(p**2 + 2*p + 4, p - 6, p)
        >>> G2 = TransferFunction(p, 4 - p, p)
        >>> G3 = TransferFunction(0, p**4 - 1, p)
        >>> G4 = TransferFunction(p**2, p**2 - 1, p)
        >>> tfm_a = TransferFunctionMatrix([[G1, G2], [G3, G4]])
        >>> tfm_b = TransferFunctionMatrix([[G2, G1], [G4, G3]])
        >>> MIMOParallel(tfm_a, tfm_b).var
        p

        """
        # 返回并行系统中第一个系统的复杂变量
        return self.args[0].var

    @property
    def num_inputs(self):
        """Returns the number of input signals of the parallel system."""
        # 返回并行系统中第一个系统的输入信号数
        return self.args[0].num_inputs

    @property
    def num_outputs(self):
        """Returns the number of output signals of the parallel system."""
        # 返回并行系统中第一个系统的输出信号数
        return self.args[0].num_outputs

    @property
    def shape(self):
        """Returns the shape of the equivalent MIMO system."""
        # 返回等效 MIMO 系统的形状，即输出信号数和输入信号数的元组
        return self.num_outputs, self.num_inputs

    @property
    def is_StateSpace_object(self):
        # 返回当前对象是否为 StateSpace 对象的并行组合
        return self._is_parallel_StateSpace
    # 定义方法 `doit`，用于计算并返回并联配置中多输入多输出系统的传递函数矩阵或状态空间
    def doit(self, **hints):
        """
        返回并联配置中多输入多输出系统评估后的传递函数矩阵或状态空间。

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction, MIMOParallel, TransferFunctionMatrix
        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
        >>> tf2 = TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)
        >>> tfm_1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
        >>> tfm_2 = TransferFunctionMatrix([[tf2, tf1], [tf1, tf2]])
        >>> MIMOParallel(tfm_1, tfm_2).doit()
        TransferFunctionMatrix(((TransferFunction((-p + s)*(s**3 - 2) + (a*p**2 + b*s)*(s**4 + 5*s + 6), (-p + s)*(s**4 + 5*s + 6), s), TransferFunction((-p + s)*(s**3 - 2) + (a*p**2 + b*s)*(s**4 + 5*s + 6), (-p + s)*(s**4 + 5*s + 6), s)), (TransferFunction((-p + s)*(s**3 - 2) + (a*p**2 + b*s)*(s**4 + 5*s + 6), (-p + s)*(s**4 + 5*s + 6), s), TransferFunction((-p + s)*(s**3 - 2) + (a*p**2 + b*s)*(s**4 + 5*s + 6), (-p + s)*(s**4 + 5*s + 6), s))))

        """
        # 如果是并联状态空间模型，则返回等效的状态空间模型
        if self._is_parallel_StateSpace:
            # 取第一个参数作为初始值
            res = self.args[0]
            # 如果初始值不是状态空间模型，则转换为状态空间模型
            if not isinstance(res, StateSpace):
                res = res.doit().rewrite(StateSpace)
            # 对于剩余的参数，如果不是状态空间模型，则转换为状态空间模型后相加
            for arg in self.args[1:]:
                if not isinstance(arg, StateSpace):
                    arg = arg.doit().rewrite(StateSpace)
                else:
                    arg = arg.doit()
                res += arg
            return res
        # 对每个参数调用 `doit` 方法并生成表达式矩阵，然后求和生成 `MatAdd` 对象
        _arg = (arg.doit()._expr_mat for arg in self.args)
        res = MatAdd(*_arg, evaluate=True)
        # 返回转换函数矩阵，基于结果矩阵和变量 `self.var`
        return TransferFunctionMatrix.from_Matrix(res, self.var)

    # 尝试以 `TransferFunctionMatrix` 形式重写对象
    def _eval_rewrite_as_TransferFunctionMatrix(self, *args, **kwargs):
        if self._is_parallel_StateSpace:
            return self.doit().rewrite(TransferFunction)
        return self.doit()

    # 定义特殊方法 `__add__` 用于支持 MIMO 并联系统的加法操作
    @_check_other_MIMO
    def __add__(self, other):
        # 将自身的参数列表转为列表并添加 `other` 参数后返回新的并联对象
        self_arg_list = list(self.args)
        return MIMOParallel(*self_arg_list, other)

    # 定义特殊方法 `__radd__`，使其与 `__add__` 相同
    __radd__ = __add__

    # 定义特殊方法 `__sub__` 用于支持 MIMO 并联系统的减法操作
    @_check_other_MIMO
    def __sub__(self, other):
        # 返回 `self + (-other)` 的结果
        return self + (-other)

    # 定义特殊方法 `__rsub__`，使其返回 `-self + other` 的结果
    def __rsub__(self, other):
        return -self + other

    # 定义特殊方法 `__mul__` 用于支持 MIMO 并联系统的乘法操作
    def __mul__(self, other):
        # 如果 `other` 是 `MIMOSeries` 类型，则将 `self` 添加到 `other` 的参数列表中返回新的系列对象
        if isinstance(other, MIMOSeries):
            arg_list = list(other.args)
            return MIMOSeries(*arg_list, self)

        # 否则，将 `other` 和 `self` 作为参数创建新的系列对象返回
        return MIMOSeries(other, self)

    # 定义特殊方法 `__neg__`，返回 `self` 参数列表中每个参数的负数的并联对象
    def __neg__(self):
        arg_list = [-arg for arg in list(self.args)]
        return MIMOParallel(*arg_list)
# 定义一个反馈系统的类，继承自 TransferFunction
class Feedback(TransferFunction):
    """
    A class for representing closed-loop feedback interconnection between two
    SISO input/output systems.
    用于表示两个SISO输入/输出系统之间的闭环反馈连接的类。

    The first argument, ``sys1``, is the feedforward part of the closed-loop
    system or in simple words, the dynamical model representing the process
    to be controlled.
    第一个参数“sys1”是闭环系统的前馈部分，或者简单来说，表示待控制的动态模型。

    The second argument, ``sys2``, is the feedback system
    and controls the fed back signal to ``sys1``.
    第二个参数“sys2”是反馈系统，控制反馈信号到“sys1”。

    Both ``sys1`` and ``sys2`` can either be ``Series`` or ``TransferFunction`` objects.
    “sys1”和“sys2”都可以是“Series”或“TransferFunction”对象。

    Parameters
    ==========

    sys1 : Series, TransferFunction
        The feedforward path system.
        前馈路径系统。

    sys2 : Series, TransferFunction, optional
        The feedback path system (often a feedback controller).
        It is the model sitting on the feedback path.
        反馈路径系统（通常是反馈控制器），位于反馈路径上的模型。

        If not specified explicitly, the sys2 is
        assumed to be unit (1.0) transfer function.
        如果没有显式指定，则默认为单位（1.0）传递函数。

    sign : int, optional
        The sign of feedback. Can either be ``1``
        (for positive feedback) or ``-1`` (for negative feedback).
        Default value is `-1`.
        反馈的符号。可以是“1”（正反馈）或“-1”（负反馈）。默认值是“-1”。

    Raises
    ======

    ValueError
        When ``sys1`` and ``sys2`` are not using the
        same complex variable of the Laplace transform.
        当“sys1”和“sys2”不使用相同的拉普拉斯变换复数变量时。

        When a combination of ``sys1`` and ``sys2`` yields
        zero denominator.
        当“sys1”和“sys2”的组合导致零分母时。

    TypeError
        When either ``sys1`` or ``sys2`` is not a ``Series`` or a
        ``TransferFunction`` object.
        当“sys1”或“sys2”不是“Series”或“TransferFunction”对象时。

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunction, Feedback
    >>> plant = TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s)
    >>> controller = TransferFunction(5*s - 10, s + 7, s)
    >>> F1 = Feedback(plant, controller)
    >>> F1
    Feedback(TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s), TransferFunction(5*s - 10, s + 7, s), -1)
    >>> F1.var
    s
    >>> F1.args
    (TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s), TransferFunction(5*s - 10, s + 7, s), -1)

    You can get the feedforward and feedback path systems by using ``.sys1`` and ``.sys2`` respectively.
    可以通过使用“.sys1”和“.sys2”来获取前馈路径和反馈路径系统。

    >>> F1.sys1
    TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s)
    >>> F1.sys2
    TransferFunction(5*s - 10, s + 7, s)

    You can get the resultant closed loop transfer function obtained by negative feedback
    interconnection using ``.doit()`` method.
    可以使用“.doit()”方法获得通过负反馈互联获得的闭环传递函数。

    >>> F1.doit()
    TransferFunction((s + 7)*(s**2 - 4*s + 2)*(3*s**2 + 7*s - 3), ((s + 7)*(s**2 - 4*s + 2) + (5*s - 10)*(3*s**2 + 7*s - 3))*(s**2 - 4*s + 2), s)
    >>> G = TransferFunction(2*s**2 + 5*s + 1, s**2 + 2*s + 3, s)
    >>> C = TransferFunction(5*s + 10, s + 10, s)
    >>> F2 = Feedback(G*C, TransferFunction(1, 1, s))
    >>> F2.doit()
    TransferFunction((s + 10)*(5*s + 10)*(s**2 + 2*s + 3)*(2*s**2 + 5*s + 1), (s + 10)*((s + 10)*(s**2 + 2*s + 3) + (5*s + 10)*(2*s**2 + 5*s + 1))*(s**2 + 2*s + 3), s)

    To negate a ``Feedback`` object, the ``-`` operator can be prepended:
    要否定一个“Feedback”对象，可以前置“-”运算符：

    >>> -F1
    """
    Feedback(TransferFunction(-3*s**2 - 7*s + 3, s**2 - 4*s + 2, s), TransferFunction(10 - 5*s, s + 7, s), -1)
    >>> -F2
    Feedback(Series(TransferFunction(-1, 1, s), TransferFunction(2*s**2 + 5*s + 1, s**2 + 2*s + 3, s), TransferFunction(5*s + 10, s + 10, s)), TransferFunction(-1, 1, s), -1)

    See Also
    ========

    MIMOFeedback, Series, Parallel

    """
    # Feedback类的构造函数，用于创建反馈系统
    def __new__(cls, sys1, sys2=None, sign=-1):
        # 如果sys2为空，则默认为TransferFunction(1, 1, sys1.var)
        if not sys2:
            sys2 = TransferFunction(1, 1, sys1.var)

        # 检查sys1和sys2的类型是否为TransferFunction、Series或Feedback的实例，否则引发TypeError异常
        if not (isinstance(sys1, (TransferFunction, Series, Feedback))
            and isinstance(sys2, (TransferFunction, Series, Feedback))):
            raise TypeError("Unsupported type for `sys1` or `sys2` of Feedback.")

        # 检查sign的值是否为-1或1，否则引发ValueError异常
        if sign not in [-1, 1]:
            raise ValueError("""
                Unsupported type for feedback. `sign` arg should
                either be 1 (positive feedback loop) or -1
                (negative feedback loop).""")

        # 如果等效系统的表达式简化后等于sign，则引发ValueError异常
        if Mul(sys1.to_expr(), sys2.to_expr()).simplify() == sign:
            raise ValueError("The equivalent system will have zero denominator.")

        # 检查sys1和sys2是否使用相同的复数变量，否则引发ValueError异常
        if sys1.var != sys2.var:
            raise ValueError("""
                Both `sys1` and `sys2` should be using the
                same complex variable.""")

        # 调用父类的构造函数创建Feedback对象
        return super(TransferFunction, cls).__new__(cls, sys1, sys2, _sympify(sign))

    @property
    def sys1(self):
        """
        Returns the feedforward system of the feedback interconnection.

        Examples
        ========

        >>> from sympy.abc import s, p
        >>> from sympy.physics.control.lti import TransferFunction, Feedback
        >>> plant = TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s)
        >>> controller = TransferFunction(5*s - 10, s + 7, s)
        >>> F1 = Feedback(plant, controller)
        >>> F1.sys1
        TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s)
        >>> G = TransferFunction(2*s**2 + 5*s + 1, p**2 + 2*p + 3, p)
        >>> C = TransferFunction(5*p + 10, p + 10, p)
        >>> P = TransferFunction(1 - s, p + 2, p)
        >>> F2 = Feedback(TransferFunction(1, 1, p), G*C*P)
        >>> F2.sys1
        TransferFunction(1, 1, p)

        """
        # 返回反馈系统中的前馈系统（sys1）
        return self.args[0]

    @property
    def sys2(self):
        """
        返回反馈互连的反馈控制器。

        Examples
        ========

        >>> from sympy.abc import s, p
        >>> from sympy.physics.control.lti import TransferFunction, Feedback
        >>> plant = TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s)
        >>> controller = TransferFunction(5*s - 10, s + 7, s)
        >>> F1 = Feedback(plant, controller)
        >>> F1.sys2
        TransferFunction(5*s - 10, s + 7, s)
        >>> G = TransferFunction(2*s**2 + 5*s + 1, p**2 + 2*p + 3, p)
        >>> C = TransferFunction(5*p + 10, p + 10, p)
        >>> P = TransferFunction(1 - s, p + 2, p)
        >>> F2 = Feedback(TransferFunction(1, 1, p), G*C*P)
        >>> F2.sys2
        Series(TransferFunction(2*s**2 + 5*s + 1, p**2 + 2*p + 3, p), TransferFunction(5*p + 10, p + 10, p), TransferFunction(1 - s, p + 2, p))

        """
        return self.args[1]



    @property
    def var(self):
        """
        返回用于反馈互连中所有传递函数的拉普拉斯变换的复变量。

        Examples
        ========

        >>> from sympy.abc import s, p
        >>> from sympy.physics.control.lti import TransferFunction, Feedback
        >>> plant = TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s)
        >>> controller = TransferFunction(5*s - 10, s + 7, s)
        >>> F1 = Feedback(plant, controller)
        >>> F1.var
        s
        >>> G = TransferFunction(2*s**2 + 5*s + 1, p**2 + 2*p + 3, p)
        >>> C = TransferFunction(5*p + 10, p + 10, p)
        >>> P = TransferFunction(1 - s, p + 2, p)
        >>> F2 = Feedback(TransferFunction(1, 1, p), G*C*P)
        >>> F2.var
        p

        """
        return self.sys1.var



    @property
    def sign(self):
        """
        返回 MIMO 反馈模型的类型。返回 ``1`` 表示正反馈，返回 ``-1`` 表示负反馈。
        """
        return self.args[2]



    @property
    def num(self):
        """
        返回闭环反馈系统的分子。
        """
        return self.sys1



    @property
    def den(self):
        """
        返回闭环反馈模型的分母。

        如果反馈类型是正反馈，则返回并联的单位函数与负的串联传递函数。
        如果反馈类型是负反馈，则返回并联的单位函数与串联传递函数。

        """
        unit = TransferFunction(1, 1, self.var)
        arg_list = list(self.sys1.args) if isinstance(self.sys1, Series) else [self.sys1]
        if self.sign == 1:
            return Parallel(unit, -Series(self.sys2, *arg_list))
        return Parallel(unit, Series(self.sys2, *arg_list))



    @property
    def sensitivity(self):
        """
        Returns the sensitivity function of the feedback loop.

        Sensitivity of a Feedback system is the ratio
        of change in the open loop gain to the change in
        the closed loop gain.

        .. note::
            This method would not return the complementary
            sensitivity function.

        Examples
        ========

        >>> from sympy.abc import p
        >>> from sympy.physics.control.lti import TransferFunction, Feedback
        >>> C = TransferFunction(5*p + 10, p + 10, p)
        >>> P = TransferFunction(1 - p, p + 2, p)
        >>> F_1 = Feedback(P, C)
        >>> F_1.sensitivity
        1/((1 - p)*(5*p + 10)/((p + 2)*(p + 10)) + 1)

        """

        # 计算并返回反馈系统的敏感函数
        return 1/(1 - self.sign*self.sys1.to_expr()*self.sys2.to_expr())

    def doit(self, cancel=False, expand=False, **hints):
        """
        Returns the resultant transfer function obtained by the
        feedback interconnection.

        Examples
        ========

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction, Feedback
        >>> plant = TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s)
        >>> controller = TransferFunction(5*s - 10, s + 7, s)
        >>> F1 = Feedback(plant, controller)
        >>> F1.doit()
        TransferFunction((s + 7)*(s**2 - 4*s + 2)*(3*s**2 + 7*s - 3), ((s + 7)*(s**2 - 4*s + 2) + (5*s - 10)*(3*s**2 + 7*s - 3))*(s**2 - 4*s + 2), s)
        >>> G = TransferFunction(2*s**2 + 5*s + 1, s**2 + 2*s + 3, s)
        >>> F2 = Feedback(G, TransferFunction(1, 1, s))
        >>> F2.doit()
        TransferFunction((s**2 + 2*s + 3)*(2*s**2 + 5*s + 1), (s**2 + 2*s + 3)*(3*s**2 + 7*s + 4), s)

        Use kwarg ``expand=True`` to expand the resultant transfer function.
        Use ``cancel=True`` to cancel out the common terms in numerator and
        denominator.

        >>> F2.doit(cancel=True, expand=True)
        TransferFunction(2*s**2 + 5*s + 1, 3*s**2 + 7*s + 4, s)
        >>> F2.doit(expand=True)
        TransferFunction(2*s**4 + 9*s**3 + 17*s**2 + 17*s + 3, 3*s**4 + 13*s**3 + 27*s**2 + 29*s + 12, s)

        """
        # 如果 self.sys1 是 Series 类型，则将其参数作为列表 arg_list
        arg_list = list(self.sys1.args) if isinstance(self.sys1, Series) else [self.sys1]
        # 计算 self.sys1 的传递函数并设置一个单位传递函数
        F_n, unit = self.sys1.doit(), TransferFunction(1, 1, self.sys1.var)
        # 根据 self.sign 的正负选择平行或串联连接
        if self.sign == -1:
            F_d = Parallel(unit, Series(self.sys2, *arg_list)).doit()
        else:
            F_d = Parallel(unit, -Series(self.sys2, *arg_list)).doit()

        # 构建结果传递函数对象 _resultant_tf
        _resultant_tf = TransferFunction(F_n.num * F_d.den, F_n.den * F_d.num, F_n.var)

        # 如果设置了 cancel 标志，则简化 _resultant_tf
        if cancel:
            _resultant_tf = _resultant_tf.simplify()

        # 如果设置了 expand 标志，则展开 _resultant_tf
        if expand:
            _resultant_tf = _resultant_tf.expand()

        return _resultant_tf

    def _eval_rewrite_as_TransferFunction(self, num, den, sign, **kwargs):
        """
        Rewrite the object as a TransferFunction.

        This function is used to rewrite the object in a specific form.
        """
        return self.doit()
    # 将 Feedback 对象转换为 SymPy 表达式
    def to_expr(self):
        """
        Converts a ``Feedback`` object to SymPy Expr.

        Examples
        ========

        >>> from sympy.abc import s, a, b
        >>> from sympy.physics.control.lti import TransferFunction, Feedback
        >>> from sympy import Expr
        >>> tf1 = TransferFunction(a+s, 1, s)
        >>> tf2 = TransferFunction(b+s, 1, s)
        >>> fd1 = Feedback(tf1, tf2)
        >>> fd1.to_expr()
        (a + s)/((a + s)*(b + s) + 1)
        >>> isinstance(_, Expr)
        True
        """

        # 调用 Feedback 对象的 doit 方法，将其转换为表达式，并返回
        return self.doit().to_expr()

    # 重载负号运算符，返回两个系统的负反馈对象
    def __neg__(self):
        return Feedback(-self.sys1, -self.sys2, self.sign)
# 定义一个函数 `_is_invertible`，用于检查给定的一对 MIMO 系统是否可逆。
def _is_invertible(a, b, sign):
    # 构造单位矩阵，a.num_outputs 是 a 对象的输出数量，sign 控制反馈方向，a.doit()._expr_mat 是 a 的表达式矩阵，b.doit()._expr_mat 是 b 的表达式矩阵
    _mat = eye(a.num_outputs) - sign*(a.doit()._expr_mat)*(b.doit()._expr_mat)
    # 计算矩阵 _mat 的行列式
    _det = _mat.det()

    # 返回行列式不等于零表示可逆
    return _det != 0


# 定义一个类 MIMOFeedback，表示两个 MIMO 输入/输出系统之间的闭环反馈连接
class MIMOFeedback(MIMOLinearTimeInvariant):
    """
    一个用于表示两个 MIMO 输入/输出系统之间闭环反馈连接的类。

    Parameters
    ==========

    sys1 : MIMOSeries, TransferFunctionMatrix
        放置在正馈路径上的 MIMO 系统。
    sys2 : MIMOSeries, TransferFunctionMatrix
        放置在反馈路径上的系统（通常是反馈控制器）。
    sign : int, optional
        反馈的符号。可以是 ``1``（正反馈）或 ``-1``（负反馈）。
        默认值为 `-1`。

    Raises
    ======

    ValueError
        当 ``sys1`` 和 ``sys2`` 使用不同的拉普拉斯变换复数变量时。
        正向路径模型的输入/输出数量与反馈路径的输出/输入不相等。
        当 ``sys1`` 和 ``sys2`` 的乘积不是方阵时。
        当等效的 MIMO 系统不可逆时。

    TypeError
        当 ``sys1`` 或 ``sys2`` 不是 ``MIMOSeries`` 或 ``TransferFunctionMatrix`` 对象时。

    Examples
    ========

    >>> from sympy import Matrix, pprint
    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunctionMatrix, MIMOFeedback
    >>> plant_mat = Matrix([[1, 1/s], [0, 1]])
    >>> controller_mat = Matrix([[10, 0], [0, 10]])  # Constant Gain
    >>> plant = TransferFunctionMatrix.from_Matrix(plant_mat, s)
    >>> controller = TransferFunctionMatrix.from_Matrix(controller_mat, s)
    >>> feedback = MIMOFeedback(plant, controller)  # Negative Feedback (default)
    >>> pprint(feedback, use_unicode=False)
    /    [1  1]    [10  0 ]   \-1   [1  1]
    |    [-  -]    [--  - ]   |     [-  -]
    |    [1  s]    [1   1 ]   |     [1  s]
    |I + [    ]   *[      ]   |   * [    ]
    |    [0  1]    [0   10]   |     [0  1]
    |    [-  -]    [-   --]   |     [-  -]
    \    [1  1]{t} [1   1 ]{t}/     [1  1]{t}

    要获得等效系统矩阵，使用 ``doit`` 或 ``rewrite`` 方法。

    >>> pprint(feedback.doit(), use_unicode=False)
    [1     1  ]
    [--  -----]
    [11  121*s]
    [         ]
    [0    1   ]
    [-    --  ]
    [1    11  ]{t}

    要对 ``MIMOFeedback`` 对象求负，使用 ``-`` 运算符。

    >>> neg_feedback = -feedback
    >>> pprint(neg_feedback.doit(), use_unicode=False)
    [-1    -1  ]
    [---  -----]
    [11   121*s]
    [          ]
    [ 0    -1  ]
    [ -    --- ]
    [ 1    11  ]{t}

    See Also
    ========

    Feedback, MIMOSeries, MIMOParallel

    """
    def __new__(cls, sys1, sys2, sign=-1):
        # 检查输入参数是否为 TransferFunctionMatrix 或 MIMOSeries 类型，否则抛出类型错误异常
        if not (isinstance(sys1, (TransferFunctionMatrix, MIMOSeries))
            and isinstance(sys2, (TransferFunctionMatrix, MIMOSeries))):
            raise TypeError("Unsupported type for `sys1` or `sys2` of MIMO Feedback.")

        # 检查 sys1 和 sys2 的输入输出端口数是否相等，否则抛出值错误异常
        if sys1.num_inputs != sys2.num_outputs or \
            sys1.num_outputs != sys2.num_inputs:
            raise ValueError(filldedent("""
                Product of `sys1` and `sys2` must
                yield a square matrix."""))

        # 检查 sign 参数是否为 -1 或 1，否则抛出值错误异常
        if sign not in (-1, 1):
            raise ValueError(filldedent("""
                Unsupported type for feedback. `sign` arg should
                either be 1 (positive feedback loop) or -1
                (negative feedback loop)."""))

        # 检查 sys1 和 sys2 是否可逆，否则抛出值错误异常
        if not _is_invertible(sys1, sys2, sign):
            raise ValueError("Non-Invertible system inputted.")
        
        # 检查 sys1 和 sys2 是否使用相同的复数变量，否则抛出值错误异常
        if sys1.var != sys2.var:
            raise ValueError(filldedent("""
                Both `sys1` and `sys2` should be using the
                same complex variable."""))

        # 调用父类的 __new__ 方法创建新的实例
        return super().__new__(cls, sys1, sys2, _sympify(sign))

    @property
    def sys1(self):
        r"""
        Returns the system placed on the feedforward path of the MIMO feedback interconnection.

        Examples
        ========

        >>> from sympy import pprint
        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, MIMOFeedback
        >>> tf1 = TransferFunction(s**2 + s + 1, s**2 - s + 1, s)
        >>> tf2 = TransferFunction(1, s, s)
        >>> tf3 = TransferFunction(1, 1, s)
        >>> sys1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
        >>> sys2 = TransferFunctionMatrix([[tf3, tf3], [tf3, tf2]])
        >>> F_1 = MIMOFeedback(sys1, sys2, 1)
        >>> F_1.sys1
        TransferFunctionMatrix(((TransferFunction(s**2 + s + 1, s**2 - s + 1, s), TransferFunction(1, s, s)), (TransferFunction(1, s, s), TransferFunction(s**2 + s + 1, s**2 - s + 1, s))))
        >>> pprint(_, use_unicode=False)
        [ 2                    ]
        [s  + s + 1      1     ]
        [----------      -     ]
        [ 2              s     ]
        [s  - s + 1            ]
        [                      ]
        [             2        ]
        [    1       s  + s + 1]
        [    -       ----------]
        [    s        2        ]
        [            s  - s + 1]{t}

        """
        # 返回该对象的第一个参数，即位于 MIMO 反馈互连的前向路径上的系统
        return self.args[0]
    def sys2(self):
        r"""
        Returns the feedback controller of the MIMO feedback interconnection.

        Examples
        ========

        >>> from sympy import pprint
        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, MIMOFeedback
        >>> tf1 = TransferFunction(s**2, s**3 - s + 1, s)
        >>> tf2 = TransferFunction(1, s, s)
        >>> tf3 = TransferFunction(1, 1, s)
        >>> sys1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
        >>> sys2 = TransferFunctionMatrix([[tf1, tf3], [tf3, tf2]])
        >>> F_1 = MIMOFeedback(sys1, sys2)
        >>> F_1.sys2
        TransferFunctionMatrix(((TransferFunction(s**2, s**3 - s + 1, s), TransferFunction(1, 1, s)), (TransferFunction(1, 1, s), TransferFunction(1, s, s))))
        >>> pprint(_, use_unicode=False)
        [     2       ]
        [    s       1]
        [----------  -]
        [ 3          1]
        [s  - s + 1   ]
        [             ]
        [    1       1]
        [    -       -]
        [    1       s]{t}

        """
        return self.args[1]



    @property
    def var(self):
        r"""
        Returns the complex variable of the Laplace transform used by all
        the transfer functions involved in the MIMO feedback loop.

        Examples
        ========

        >>> from sympy.abc import p
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, MIMOFeedback
        >>> tf1 = TransferFunction(p, 1 - p, p)
        >>> tf2 = TransferFunction(1, p, p)
        >>> tf3 = TransferFunction(1, 1, p)
        >>> sys1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
        >>> sys2 = TransferFunctionMatrix([[tf1, tf3], [tf3, tf2]])
        >>> F_1 = MIMOFeedback(sys1, sys2, 1)  # Positive feedback
        >>> F_1.var
        p

        """
        return self.sys1.var



    @property
    def sign(self):
        r"""
        Returns the type of feedback interconnection of two models. ``1``
        for Positive and ``-1`` for Negative.
        """
        return self.args[2]
    def sensitivity(self):
        r"""
        Returns the sensitivity function matrix of the feedback loop.

        Sensitivity of a closed-loop system is the ratio of change
        in the open loop gain to the change in the closed loop gain.

        .. note::
            This method would not return the complementary
            sensitivity function.

        Examples
        ========

        >>> from sympy import pprint
        >>> from sympy.abc import p
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, MIMOFeedback
        >>> tf1 = TransferFunction(p, 1 - p, p)
        >>> tf2 = TransferFunction(1, p, p)
        >>> tf3 = TransferFunction(1, 1, p)
        >>> sys1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])
        >>> sys2 = TransferFunctionMatrix([[tf1, tf3], [tf3, tf2]])
        >>> F_1 = MIMOFeedback(sys1, sys2, 1)  # Positive feedback
        >>> F_2 = MIMOFeedback(sys1, sys2)  # Negative feedback
        >>> pprint(F_1.sensitivity, use_unicode=False)
        [   4      3      2               5      4      2           ]
        [- p  + 3*p  - 4*p  + 3*p - 1    p  - 2*p  + 3*p  - 3*p + 1 ]
        [----------------------------  -----------------------------]
        [  4      3      2              5      4      3      2      ]
        [ p  + 3*p  - 8*p  + 8*p - 3   p  + 3*p  - 8*p  + 8*p  - 3*p]
        [                                                           ]
        [       4    3    2                  3      2               ]
        [      p  - p  - p  + p           3*p  - 6*p  + 4*p - 1     ]
        [ --------------------------    --------------------------  ]
        [  4      3      2               4      3      2            ]
        [ p  + 3*p  - 8*p  + 8*p - 3    p  + 3*p  - 8*p  + 8*p - 3  ]
        >>> pprint(F_2.sensitivity, use_unicode=False)
        [ 4      3      2           5      4      2          ]
        [p  - 3*p  + 2*p  + p - 1  p  - 2*p  + 3*p  - 3*p + 1]
        [------------------------  --------------------------]
        [   4      3                   5      4      2       ]
        [  p  - 3*p  + 2*p - 1        p  - 3*p  + 2*p  - p   ]
        [                                                    ]
        [     4    3    2               4      3             ]
        [    p  - p  - p  + p        2*p  - 3*p  + 2*p - 1   ]
        [  -------------------       ---------------------   ]
        [   4      3                   4      3              ]
        [  p  - 3*p  + 2*p - 1        p  - 3*p  + 2*p - 1    ]

        """
        # 计算系统1和系统2的表达式矩阵
        _sys1_mat = self.sys1.doit()._expr_mat
        _sys2_mat = self.sys2.doit()._expr_mat

        # 返回灵敏度函数矩阵，即反馈回路的灵敏度函数
        return (eye(self.sys1.num_inputs) - \
            self.sign*_sys1_mat*_sys2_mat).inv()

    def _eval_rewrite_as_TransferFunctionMatrix(self, sys1, sys2, sign, **kwargs):
        # 将对象重写为TransferFunctionMatrix的形式
        return self.doit()

    def __neg__(self):
        # 返回当前对象的相反数，即负反馈的MIMOFeedback对象
        return MIMOFeedback(-self.sys1, -self.sys2, self.sign)
# 定义一个私有方法，将 ImmutableMatrix 转换为 TransferFunctionMatrix，以提高效率
def _to_TFM(mat, var):
    # 使用 lambda 表达式定义一个函数，将每个表达式转换为 TransferFunction 对象，并传入变量 var
    to_tf = lambda expr: TransferFunction.from_rational_expression(expr, var)
    # 使用列表推导式，对 mat 中的每一行（转换为列表）中的每个表达式，都调用 to_tf 函数进行转换
    arg = [[to_tf(expr) for expr in row] for row in mat.tolist()]
    # 返回构建好的 TransferFunctionMatrix 对象
    return TransferFunctionMatrix(arg)


class TransferFunctionMatrix(MIMOLinearTimeInvariant):
    """
    用于表示多输入多输出（MIMO）系统的传递函数矩阵的类，是单输入单输出（SISO）传递函数的一般化。

    它是传递函数（``TransferFunction``）、SISO-``Series`` 或 SISO-``Parallel`` 的矩阵。
    只有一个参数 ``arg``，也是强制性参数。
    ``arg`` 应严格为列表的列表类型，其中包含传递函数或可简化为传递函数。

    参数
    ==========

    arg : 嵌套的 ``List``（严格）。
        用户应输入一个嵌套列表，包含 ``TransferFunction``、``Series`` 和/或 ``Parallel`` 对象。

    示例
    ========

    .. note::
        可使用 ``pprint()`` 来更好地可视化 ``TransferFunctionMatrix`` 对象。

    >>> from sympy.abc import s, p, a
    >>> from sympy import pprint
    >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, Series, Parallel
    >>> tf_1 = TransferFunction(s + a, s**2 + s + 1, s)
    >>> tf_2 = TransferFunction(p**4 - 3*p + 2, s + p, s)
    >>> tf_3 = TransferFunction(3, s + 2, s)
    >>> tf_4 = TransferFunction(-a + p, 9*s - 9, s)
    >>> tfm_1 = TransferFunctionMatrix([[tf_1], [tf_2], [tf_3]])
    >>> tfm_1
    TransferFunctionMatrix(((TransferFunction(a + s, s**2 + s + 1, s),), (TransferFunction(p**4 - 3*p + 2, p + s, s),), (TransferFunction(3, s + 2, s),)))
    >>> tfm_1.var
    s
    >>> tfm_1.num_inputs
    1
    >>> tfm_1.num_outputs
    3
    >>> tfm_1.shape
    (3, 1)
    >>> tfm_1.args
    (((TransferFunction(a + s, s**2 + s + 1, s),), (TransferFunction(p**4 - 3*p + 2, p + s, s),), (TransferFunction(3, s + 2, s),)),)
    >>> tfm_2 = TransferFunctionMatrix([[tf_1, -tf_3], [tf_2, -tf_1], [tf_3, -tf_2]])
    >>> tfm_2
    TransferFunctionMatrix(((TransferFunction(a + s, s**2 + s + 1, s), TransferFunction(-3, s + 2, s)), (TransferFunction(p**4 - 3*p + 2, p + s, s), TransferFunction(-a - s, s**2 + s + 1, s)), (TransferFunction(3, s + 2, s), TransferFunction(-p**4 + 3*p - 2, p + s, s))))
    >>> pprint(tfm_2, use_unicode=False)  # pretty-printing for better visualization
    [   a + s           -3       ]
    [ ----------       -----     ]
    [  2               s + 2     ]
    [ s  + s + 1                 ]
    [                            ]
    [ 4                          ]
    [p  - 3*p + 2      -a - s    ]
    [------------    ----------  ]
    [   p + s         2          ]
    [                s  + s + 1  ]
    [                            ]
    [                 4          ]
    [     3        - p  + 3*p - 2]
    """
    pass
    # 创建一个 TransferFunctionMatrix 对象 tfm_2，并且可以进行转置操作，用于交换输入和输出的传递函数
    tfm_2 = TransferFunctionMatrix(((TransferFunction(a + s, s**2 + s + 1, s), TransferFunction(p**4 - 3*p + 2, p + s, s), TransferFunction(3, s + 2, s)),
                                   (TransferFunction(-3, s + 2, s), TransferFunction(-a - s, s**2 + s + 1, s), TransferFunction(-p**4 + 3*p - 2, p + s, s))))

    # 打印 tfm_2 对象，使用 pprint 进行输出，use_unicode=False 表示不使用 Unicode 进行输出
    pprint(_, use_unicode=False)
    # 显示 tfm_2 的转置结果，交换了输入和输出的传递函数
    [             4                          ]
    [  a + s     p  - 3*p + 2        3       ]
    [----------  ------------      -----     ]
    [ 2             p + s          s + 2     ]
    [s  + s + 1                              ]
    [                                        ]
    [                             4          ]
    [   -3          -a - s     - p  + 3*p - 2]
    [  -----      ----------   --------------]
    [  s + 2       2               p + s     ]
    [             s  + s + 1                 ]

    # 创建四个 TransferFunction 对象 tf_5, tf_6, tf_7, tf_8 分别用于不同的传递函数表示
    >>> tf_5 = TransferFunction(5, s, s)
    >>> tf_6 = TransferFunction(5*s, (2 + s**2), s)
    >>> tf_7 = TransferFunction(5, (s*(2 + s**2)), s)
    >>> tf_8 = TransferFunction(5, 1, s)

    # 创建一个 TransferFunctionMatrix 对象 tfm_3，包含两行两列的传递函数
    >>> tfm_3 = TransferFunctionMatrix([[tf_5, tf_6], [tf_7, tf_8]])
    # 打印 tfm_3 对象，使用 pprint 进行输出，use_unicode=False 表示不使用 Unicode 进行输出
    >>> pprint(tfm_3, use_unicode=False)
    [    5        5*s  ]
    [    -       ------]
    [    s        2    ]
    [            s  + 2]
    [                  ]
    [    5         5   ]
    [----------    -   ]
    [  / 2    \    1   ]
    [s*\s  + 2/        ]

    # 访问 tfm_3 的属性 var，显示其中使用的自变量
    >>> tfm_3.var
    s
    # 访问 tfm_3 的形状属性，显示其矩阵的行列数
    >>> tfm_3.shape
    (2, 2)
    # 访问 tfm_3 的 num_outputs 属性，显示其输出的数量
    >>> tfm_3.num_outputs
    2
    # 访问 tfm_3 的 num_inputs 属性，显示其输入的数量
    >>> tfm_3.num_inputs
    2
    # 访问 tfm_3 的 args 属性，显示传递函数矩阵的具体组成
    >>> tfm_3.args
    (((TransferFunction(5, s, s), TransferFunction(5*s, s**2 + 2, s)), (TransferFunction(5, s*(s**2 + 2), s), TransferFunction(5, 1, s))),)

    # 使用索引访问 TransferFunctionMatrix tfm_3 中的特定 TransferFunction 对象
    >>> tfm_3[1, 0]  # 返回位于第2行第1列的 TransferFunction 对象
    TransferFunction(5, s*(s**2 + 2), s)
    >>> tfm_3[0, 0]  # 返回位于第1行第1列的 TransferFunction 对象
    TransferFunction(5, s, s)
    >>> tfm_3[:, 0]  # 返回第1列的 TransferFunctionMatrix 对象
    TransferFunctionMatrix(((TransferFunction(5, s, s),), (TransferFunction(5, s*(s**2 + 2), s),)))
    >>> pprint(_, use_unicode=False)
    [    5     ]
    [    -     ]
    [    s     ]
    [          ]
    [    5     ]
    [----------]
    [  / 2    \]
    [s*\s  + 2/]

    # 返回第1行的 TransferFunctionMatrix 对象
    >>> tfm_3[0, :]  # 返回第1行的 TransferFunctionMatrix 对象
    TransferFunctionMatrix(((TransferFunction(5, s, s), TransferFunction(5*s, s**2 + 2, s)),))
    >>> pprint(_, use_unicode=False)
    [5   5*s  ]
    [-  ------]
    [s   2    ]
    [   s  + 2]
    # To negate a transfer function matrix, use the `-` operator prepended to the matrix:
    >>> tfm_4 = TransferFunctionMatrix([[tf_2], [-tf_1], [tf_3]])
    # Negates each transfer function in tfm_4, creating a new TransferFunctionMatrix
    >>> -tfm_4
    TransferFunctionMatrix(((TransferFunction(-p**4 + 3*p - 2, p + s, s),), (TransferFunction(a + s, s**2 + s + 1, s),), (TransferFunction(-3, s + 2, s),)))
    
    # Create a new TransferFunctionMatrix tfm_5 and apply negation to a specific element.
    >>> tfm_5 = TransferFunctionMatrix([[tf_1, tf_2], [tf_3, -tf_1]])
    # Negates the element tf_1 within tfm_5, keeping other elements unchanged.
    >>> -tfm_5
    TransferFunctionMatrix(((TransferFunction(-a - s, s**2 + s + 1, s), TransferFunction(-p**4 + 3*p - 2, p + s, s)), (TransferFunction(-3, s + 2, s), TransferFunction(a + s, s**2 + s + 1, s))))
    
    # The method subs() returns a new TransferFunctionMatrix with specified substitutions applied to its elements.
    # It does not modify the original TransferFunctionMatrix object.
    
    >>> tfm_2.subs(p, 2)  # Substitute p with 2 in all elements of tfm_2.
    TransferFunctionMatrix(((TransferFunction(a + s, s**2 + s + 1, s), TransferFunction(-3, s + 2, s)), (TransferFunction(12, s + 2, s), TransferFunction(-a - s, s**2 + s + 1, s)), (TransferFunction(3, s + 2, s), TransferFunction(-12, s + 2, s))))
    >>> pprint(_, use_unicode=False)
    [  a + s        -3     ]
    [----------    -----   ]
    [ 2            s + 2   ]
    [s  + s + 1            ]
    [                      ]
    [    12        -a - s  ]
    [  -----     ----------]
    [  s + 2      2        ]
    [            s  + s + 1]
    [                      ]
    [    3          -12    ]
    [  -----       -----   ]
    [  s + 2       s + 2   ]{t}
    
    # Print the state of tfm_2 after the substitution operation, showing no change in tfm_2 itself.
    >>> pprint(tfm_2, use_unicode=False)
    [   a + s           -3       ]
    [ ----------       -----     ]
    [  2               s + 2     ]
    [ s  + s + 1                 ]
    [                            ]
    [ 4                          ]
    [p  - 3*p + 2      -a - s    ]
    [------------    ----------  ]
    [   p + s         2          ]
    [                s  + s + 1  ]
    [                            ]
    [                 4          ]
    [     3        - p  + 3*p - 2]
    [   -----      --------------]
    [   s + 2          p + s     ]{t}
    
    # The subs() method also supports substituting multiple variables simultaneously.
    >>> tfm_2.subs({p: 2, a: 1})  # Substitute p with 2 and a with 1 in tfm_2.
    TransferFunctionMatrix(((TransferFunction(s + 1, s**2 + s + 1, s), TransferFunction(-3, s + 2, s)), (TransferFunction(12, s + 2, s), TransferFunction(-s - 1, s**2 + s + 1, s)), (TransferFunction(3, s + 2, s), TransferFunction(-12, s + 2, s))))
    >>> pprint(_, use_unicode=False)
    [  s + 1        -3     ]
    [----------    -----   ]
    [ 2            s + 2   ]
    [s  + s + 1            ]
    [                      ]
    [    12        -s - 1  ]
    [  -----     ----------]
    [  s + 2      2        ]
    [            s  + s + 1]
    [                      ]
    [    3          -12    ]
    [  -----       -----   ]
    [  s + 2       s + 2   ]{t}
    # 创建一个包含多个 TransferFunction 对象的 TransferFunctionMatrix 对象 tfm_8，包含了一个子列表 [tf_3]
    tfm_8 = TransferFunctionMatrix([[tf_3]])
    
    # 创建一个包含一个 TransferFunction 对象的 TransferFunctionMatrix 对象 tfm_9，包含了一个子列表 [-tf_3]
    tfm_9 = TransferFunctionMatrix([[-tf_3]])
    
    # 创建一个包含多个 TransferFunction 对象的 TransferFunctionMatrix 对象 tfm_10，包含了一个子列表 [tf_1], [tf_2], [tf_4]
    tfm_10 = TransferFunctionMatrix([[tf_1], [tf_2], [tf_4]])
    
    # 创建一个包含多个 TransferFunction 对象的 TransferFunctionMatrix 对象 tfm_11，包含了一个子列表 [tf_4], [-tf_1]
    tfm_11 = TransferFunctionMatrix([[tf_4], [-tf_1]])
    
    # 创建一个包含多个 TransferFunction 对象的 TransferFunctionMatrix 对象 tfm_12，包含了一个子列表 [tf_4, -tf_1, tf_3], [-tf_2, -tf_4, -tf_3]
    tfm_12 = TransferFunctionMatrix([[tf_4, -tf_1, tf_3], [-tf_2, -tf_4, -tf_3]])
    
    # 计算 tfm_8 和 tfm_10 的加法结果，返回一个新的 TransferFunctionMatrix 对象
    tfm_8 + tfm_10
    # 创建一个 MIMOParallel 对象，包含一个 TransferFunctionMatrix，其中有三个 TransferFunction
    MIMOParallel(
        TransferFunctionMatrix((
            (TransferFunction(3, s + 2, s),),                       # 第一个 TransferFunction: 分子为 3，分母为 s + 2
            (TransferFunction(p**4 - 3*p + 2, p + s, s),),         # 第二个 TransferFunction: 分子为 p^4 - 3*p + 2，分母为 p + s
            (TransferFunction(-a - s, s**2 + s + 1, s),)           # 第三个 TransferFunction: 分子为 -a - s，分母为 s^2 + s + 1
        )),
        TransferFunctionMatrix((
            (TransferFunction(a + s, s**2 + s + 1, s),),           # 第一个 TransferFunction: 分子为 a + s，分母为 s^2 + s + 1
            (TransferFunction(p**4 - 3*p + 2, p + s, s),),         # 第二个 TransferFunction: 分子为 p^4 - 3*p + 2，分母为 p + s
            (TransferFunction(-a + p, 9*s - 9, s),)                # 第三个 TransferFunction: 分子为 -a + p，分母为 9*s - 9
        ))
    )
    # 打印 MIMOParallel 对象，使用非 Unicode 字符集
    >>> pprint(_, use_unicode=False)
    [     3      ]      [   a + s    ]
    [   -----    ]      [ ---------- ]
    [   s + 2    ]      [  2         ]
    [            ]      [ s  + s + 1 ]
    [ 4          ]      [            ]
    [p  - 3*p + 2]      [ 4          ]
    [------------]    + [p  - 3*p + 2]
    [   p + s    ]      [------------]
    [            ]      [   p + s    ]
    [   -a - s   ]      [            ]
    [ ---------- ]      [   -a + p   ]
    [  2         ]      [  -------   ]
    [ s  + s + 1 ]{t}   [  9*s - 9   ]{t}

    # 计算 tfm_10 和 tfm_8 的负值
    >>> -tfm_10 - tfm_8
    # 创建一个 MIMOParallel 对象，包含一个 TransferFunctionMatrix，其中有三个 TransferFunction
    MIMOParallel(
        TransferFunctionMatrix((
            (TransferFunction(-a - s, s**2 + s + 1, s),),           # 第一个 TransferFunction: 分子为 -a - s，分母为 s^2 + s + 1
            (TransferFunction(-p**4 + 3*p - 2, p + s, s),),         # 第二个 TransferFunction: 分子为 -p^4 + 3*p - 2，分母为 p + s
            (TransferFunction(a - p, 9*s - 9, s),)                  # 第三个 TransferFunction: 分子为 a - p，分母为 9*s - 9
        )),
        TransferFunctionMatrix((
            (TransferFunction(-3, s + 2, s),),                       # 第一个 TransferFunction: 分子为 -3，分母为 s + 2
            (TransferFunction(-p**4 + 3*p - 2, p + s, s),),         # 第二个 TransferFunction: 分子为 -p^4 + 3*p - 2，分母为 p + s
            (TransferFunction(a + s, s**2 + s + 1, s),)            # 第三个 TransferFunction: 分子为 a + s，分母为 s^2 + s + 1
        ))
    )
    # 打印 MIMOParallel 对象，使用非 Unicode 字符集
    >>> pprint(_, use_unicode=False)
    [    -a - s    ]      [     -3       ]
    [  ----------  ]      [    -----     ]
    [   2          ]      [    s + 2     ]
    [  s  + s + 1  ]      [              ]
    [              ]      [   4          ]
    [   4          ]      [- p  + 3*p - 2]
    [- p  + 3*p - 2]    + [--------------]
    [--------------]      [    p + s     ]
    [    p + s     ]      [              ]
    [              ]      [    a + s     ]
    [    a - p     ]      [  ----------  ]
    [   -------    ]      [   2          ]
    [   9*s - 9    ]{t}   [  s  + s + 1  ]{t}

    # 计算 tfm_12 和 tfm_8 的乘积
    >>> tfm_12 * tfm_8
    # 创建一个 MIMOSeries 对象，包含一个 TransferFunctionMatrix，其中有三个 TransferFunction
    MIMOSeries(
        TransferFunctionMatrix((
            (TransferFunction(3, s + 2, s),),                       # 第一个 TransferFunction: 分子为 3，分母为 s + 2
            (TransferFunction(p**4 - 3*p + 2, p + s, s),),         # 第二个 TransferFunction: 分子为 p^4 - 3*p + 2，分母为 p + s
            (TransferFunction(-a - s, s**2 + s + 1, s),)           # 第三个 TransferFunction: 分子为 -a - s，分母为 s^2 + s + 1
        )),
        TransferFunctionMatrix((
            (TransferFunction(-a + p, 9*s - 9, s),                   # 第一个 TransferFunction: 分子为 -a + p，分母为 9*s - 9
             TransferFunction(-a - s, s**2 + s + 1, s),             # 第二个 TransferFunction: 分子为 -a - s，分母为 s^2 + s + 1
             TransferFunction(3, s + 2, s)),                        # 第三个 TransferFunction: 分子为 3，分母为 s + 2
            (TransferFunction(-p**4 + 3*p - 2, p + s, s),           # 第一个 TransferFunction: 分子为 -p^4 + 3*p - 2，分母为 p + s
             TransferFunction(a - p, 9*s - 9, s),                    # 第二个 TransferFunction: 分子为 a - p，分母为 9*s - 9
             TransferFunction(-3, s + 2, s))                         # 第三个 TransferFunction: 分子为 -3，分母为 s + 2
        ))
    )
    # 打印 MIMOSeries 对象，使用非 Unicode 字符集
    >>> pprint(_, use_unicode=False)
                                           [     3      ]
                                           [   -----    ]
    [    -a + p        -a - s      3  ]    [   s + 2    ]
    [   -------      ----------  -----]    [            ]
    [   9*s - 9       2          s + 2]    [ 4          ]
    [                s  + s + 1       ]    [p  - 3*p + 2]
    [                                 ]   *[------------]
    [   4                             ]    [   p + s    ]
    [- p  + 3*p - 2    a - p      -3  ]    [            ]
    [--------------   -------    -----]    [   -a - s   ]
    # 定义一个复杂的传递函数表达式，表示为一个 MIMOSeries 对象
    MIMOSeries(
        TransferFunctionMatrix((
            (TransferFunction(-3, s + 2, s),),
        )),
        TransferFunctionMatrix((
            (TransferFunction(3, s + 2, s),),
            (TransferFunction(p**4 - 3*p + 2, p + s, s),),
            (TransferFunction(-a - s, s**2 + s + 1, s),)
        )),
        TransferFunctionMatrix((
            (TransferFunction(-a + p, 9*s - 9, s),
             TransferFunction(-a - s, s**2 + s + 1, s),
             TransferFunction(3, s + 2, s)),
            (TransferFunction(-p**4 + 3*p - 2, p + s, s),
             TransferFunction(a - p, 9*s - 9, s),
             TransferFunction(-3, s + 2, s))
        ))
    )

    # 使用 pprint 函数打印上述传递函数表达式的文本表示，不使用 Unicode 字符
    pprint(_, use_unicode=False)

    # 下面是另一个复杂的传递函数表达式，表示为一个 MIMOParallel 对象
    MIMOParallel(
        TransferFunctionMatrix((
            (TransferFunction(a + s, s**2 + s + 1, s),),
            (TransferFunction(p**4 - 3*p + 2, p + s, s),),
            (TransferFunction(-a + p, 9*s - 9, s),)
        )),
        MIMOSeries(
            TransferFunctionMatrix((
                (TransferFunction(-3, s + 2, s),),
            )),
            TransferFunctionMatrix((
                (TransferFunction(3, s + 2, s),),
                (TransferFunction(p**4 - 3*p + 2, p + s, s),),
                (TransferFunction(-a - s, s**2 + s + 1, s),)
            ))
        )
    )

    # 使用 pprint 函数打印上述传递函数表达式的文本表示，不使用 Unicode 字符
    pprint(_, use_unicode=False)

    # 对之前定义的三个传递函数表达式进行求值，计算其结果
    (-tfm_8 + tfm_10 + tfm_8*tfm_9).doit()
    TransferFunctionMatrix(((TransferFunction((a + s)*(s + 2)**3 - 3*(s + 2)**2*(s**2 + s + 1) - 9*(s + 2)*(s**2 + s + 1), (s + 2)**3*(s**2 + s + 1), s),), (TransferFunction((p + s)*(-3*p**4 + 9*p - 6), (p + s)**2*(s + 2), s),), (TransferFunction((-a + p)*(s + 2)*(s**2 + s + 1)**2 + (a + s)*(s + 2)*(9*s - 9)*(s**2 + s + 1) + (3*a + 3*s)*(9*s - 9)*(s**2 + s + 1), (s + 2)*(9*s - 9)*(s**2 + s + 1)**2, s),)))
    >>> (-tfm_12 * -tfm_8 * -tfm_9).rewrite(TransferFunctionMatrix)
    TransferFunctionMatrix(((TransferFunction(3*(-3*a + 3*p)*(p + s)*(s + 2)*(s**2 + s + 1)**2 + 3*(-3*a - 3*s)*(p + s)*(s + 2)*(9*s - 9)*(s**2 + s + 1) + 3*(a + s)*(s + 2)**2*(9*s - 9)*(-p**4 + 3*p - 2)*(s**2 + s + 1), (p + s)*(s + 2)**3*(9*s - 9)*(s**2 + s + 1)**2, s),), (TransferFunction(3*(-a + p)*(p + s)*(s + 2)**2*(-p**4 + 3*p - 2)*(s**2 + s + 1) + 3*(3*a + 3*s)*(p + s)**2*(s + 2)*(9*s - 9) + 3*(p + s)*(s + 2)*(9*s - 9)*(-3*p**4 + 9*p - 6)*(s**2 + s + 1), (p + s)**2*(s + 2)**3*(9*s - 9)*(s**2 + s + 1), s),)))

    See Also
    ========

    TransferFunction, MIMOSeries, MIMOParallel, Feedback

    """
    创建一个 TransferFunctionMatrix 对象，其参数是一个包含多个 TransferFunction 对象的嵌套元组。
    这些 TransferFunction 对象描述了复杂的线性时不变系统的传递函数。

    def __new__(cls, arg):
        expr_mat_arg = []
        try:
            var = arg[0][0].var
        except TypeError:
            raise ValueError(filldedent("""
                `arg` param in TransferFunctionMatrix should
                strictly be a nested list containing TransferFunction
                objects."""))
        
        # 遍历参数 arg 中的每一行和每个元素，确保它们是有效的 TransferFunction 对象
        for row in arg:
            temp = []
            for element in row:
                # 检查每个元素是否是 SISOLinearTimeInvariant 类型
                if not isinstance(element, SISOLinearTimeInvariant):
                    raise TypeError(filldedent("""
                        Each element is expected to be of
                        type `SISOLinearTimeInvariant`."""))

                # 检查每个 TransferFunction 对象是否使用相同的复杂变量 var
                if var != element.var:
                    raise ValueError(filldedent("""
                        Conflicting value(s) found for `var`. All TransferFunction
                        instances in TransferFunctionMatrix should use the same
                        complex variable in Laplace domain."""))

                # 将每个 TransferFunction 对象转换为表达式形式并添加到临时列表中
                temp.append(element.to_expr())
            expr_mat_arg.append(temp)

        # 如果参数 arg 是 tuple、list 或 Tuple 类型，则将其转换为嵌套的 Tuple 对象
        if isinstance(arg, (tuple, list, Tuple)):
            arg = Tuple(*(Tuple(*r, sympify=False) for r in arg), sympify=False)

        # 调用父类的 __new__ 方法创建 TransferFunctionMatrix 对象
        obj = super(TransferFunctionMatrix, cls).__new__(cls, arg)
        # 使用表达式列表创建不可变矩阵对象 _expr_mat
        obj._expr_mat = ImmutableMatrix(expr_mat_arg)
        return obj
    def from_Matrix(cls, matrix, var):
        """
        从 SymPy 的 Expr 对象组成的矩阵高效创建一个新的 TransferFunctionMatrix。

        Parameters
        ==========

        matrix : ImmutableMatrix
            包含 Expr/Number 元素的不可变矩阵。
        var : Symbol
            Laplace 变换中将被 TransferFunctionMatrix 中所有 TransferFunction 对象使用的复数变量。

        Examples
        ========

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunctionMatrix
        >>> from sympy import Matrix, pprint
        >>> M = Matrix([[s, 1/s], [1/(s+1), s]])
        >>> M_tf = TransferFunctionMatrix.from_Matrix(M, s)
        >>> pprint(M_tf, use_unicode=False)
        [  s    1]
        [  -    -]
        [  1    s]
        [        ]
        [  1    s]
        [-----  -]
        [s + 1  1]{t}
        >>> M_tf.elem_poles()
        [[[], [0]], [[-1], []]]
        >>> M_tf.elem_zeros()
        [[[0], []], [[], [0]]]

        """
        return _to_TFM(matrix, var)

    @property
    def var(self):
        """
        返回所有传递函数矩阵中的传递函数或 Series/Parallel 对象所使用的复数变量。

        Examples
        ========

        >>> from sympy.abc import p, s
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, Series, Parallel
        >>> G1 = TransferFunction(p**2 + 2*p + 4, p - 6, p)
        >>> G2 = TransferFunction(p, 4 - p, p)
        >>> G3 = TransferFunction(0, p**4 - 1, p)
        >>> G4 = TransferFunction(s + 1, s**2 + s + 1, s)
        >>> S1 = Series(G1, G2)
        >>> S2 = Series(-G3, Parallel(G2, -G1))
        >>> tfm1 = TransferFunctionMatrix([[G1], [G2], [G3]])
        >>> tfm1.var
        p
        >>> tfm2 = TransferFunctionMatrix([[-S1, -S2], [S1, S2]])
        >>> tfm2.var
        p
        >>> tfm3 = TransferFunctionMatrix([[G4]])
        >>> tfm3.var
        s

        """
        return self.args[0][0][0].var

    @property
    def num_inputs(self):
        """
        返回系统的输入数量。

        Examples
        ========

        >>> from sympy.abc import s, p
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix
        >>> G1 = TransferFunction(s + 3, s**2 - 3, s)
        >>> G2 = TransferFunction(4, s**2, s)
        >>> G3 = TransferFunction(p**2 + s**2, p - 3, s)
        >>> tfm_1 = TransferFunctionMatrix([[G2, -G1, G3], [-G2, -G1, -G3]])
        >>> tfm_1.num_inputs
        3

        See Also
        ========

        num_outputs

        """
        return self._expr_mat.shape[1]
    def num_outputs(self):
        """
        返回系统输出的数量。

        Examples
        ========

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunctionMatrix
        >>> from sympy import Matrix
        >>> M_1 = Matrix([[s], [1/s]])
        >>> TFM = TransferFunctionMatrix.from_Matrix(M_1, s)
        >>> print(TFM)
        TransferFunctionMatrix(((TransferFunction(s, 1, s),), (TransferFunction(1, s, s),)))
        >>> TFM.num_outputs
        2

        See Also
        ========

        num_inputs

        """
        return self._expr_mat.shape[0]

    @property
    def shape(self):
        """
        返回传递函数矩阵的形状，即 ``(输出数量, 输入数量)``。

        Examples
        ========

        >>> from sympy.abc import s, p
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix
        >>> tf1 = TransferFunction(p**2 - 1, s**4 + s**3 - p, p)
        >>> tf2 = TransferFunction(1 - p, p**2 - 3*p + 7, p)
        >>> tf3 = TransferFunction(3, 4, p)
        >>> tfm1 = TransferFunctionMatrix([[tf1, -tf2]])
        >>> tfm1.shape
        (1, 2)
        >>> tfm2 = TransferFunctionMatrix([[-tf2, tf3], [tf1, -tf1]])
        >>> tfm2.shape
        (2, 2)

        """
        return self._expr_mat.shape

    def __neg__(self):
        """
        返回传递函数矩阵的负矩阵。

        """
        neg = -self._expr_mat
        return _to_TFM(neg, self.var)

    @_check_other_MIMO
    def __add__(self, other):
        """
        将当前传递函数矩阵与另一个对象相加，返回新的并联多输入多输出系统。

        """
        if not isinstance(other, MIMOParallel):
            return MIMOParallel(self, other)
        other_arg_list = list(other.args)
        return MIMOParallel(self, *other_arg_list)

    @_check_other_MIMO
    def __sub__(self, other):
        """
        将当前传递函数矩阵与另一个对象相减，返回新的并联多输入多输出系统的负矩阵。

        """
        return self + (-other)

    @_check_other_MIMO
    def __mul__(self, other):
        """
        将当前传递函数矩阵与另一个对象相乘，返回新的串联多输入多输出系统。

        """
        if not isinstance(other, MIMOSeries):
            return MIMOSeries(other, self)
        other_arg_list = list(other.args)
        return MIMOSeries(*other_arg_list, self)

    def __getitem__(self, key):
        """
        返回传递函数矩阵的特定元素。

        """
        trunc = self._expr_mat.__getitem__(key)
        if isinstance(trunc, ImmutableMatrix):
            return _to_TFM(trunc, self.var)
        return TransferFunction.from_rational_expression(trunc, self.var)

    def transpose(self):
        """
        返回传递函数矩阵的转置（交换输入和输出层）。

        """
        transposed_mat = self._expr_mat.transpose()
        return _to_TFM(transposed_mat, self.var)
    def elem_poles(self):
        """
        Returns the poles of each element of the ``TransferFunctionMatrix``.

        .. note::
            Actual poles of a MIMO system are NOT the poles of individual elements.

        Examples
        ========

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix
        >>> tf_1 = TransferFunction(3, (s + 1), s)
        >>> tf_2 = TransferFunction(s + 6, (s + 1)*(s + 2), s)
        >>> tf_3 = TransferFunction(s + 3, s**2 + 3*s + 2, s)
        >>> tf_4 = TransferFunction(s + 2, s**2 + 5*s - 10, s)
        >>> tfm_1 = TransferFunctionMatrix([[tf_1, tf_2], [tf_3, tf_4]])
        >>> tfm_1
        TransferFunctionMatrix(((TransferFunction(3, s + 1, s), TransferFunction(s + 6, (s + 1)*(s + 2), s)), (TransferFunction(s + 3, s**2 + 3*s + 2, s), TransferFunction(s + 2, s**2 + 5*s - 10, s))))
        >>> tfm_1.elem_poles()
        [[[-1], [-2, -1]], [[-2, -1], [-5/2 + sqrt(65)/2, -sqrt(65)/2 - 5/2]]]

        See Also
        ========

        elem_zeros

        """
        # 返回一个二维列表，其中每个元素是对应位置元素的极点列表
        return [[element.poles() for element in row] for row in self.doit().args[0]]

    def elem_zeros(self):
        """
        Returns the zeros of each element of the ``TransferFunctionMatrix``.

        .. note::
            Actual zeros of a MIMO system are NOT the zeros of individual elements.

        Examples
        ========

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix
        >>> tf_1 = TransferFunction(3, (s + 1), s)
        >>> tf_2 = TransferFunction(s + 6, (s + 1)*(s + 2), s)
        >>> tf_3 = TransferFunction(s + 3, s**2 + 3*s + 2, s)
        >>> tf_4 = TransferFunction(s**2 - 9*s + 20, s**2 + 5*s - 10, s)
        >>> tfm_1 = TransferFunctionMatrix([[tf_1, tf_2], [tf_3, tf_4]])
        >>> tfm_1
        TransferFunctionMatrix(((TransferFunction(3, s + 1, s), TransferFunction(s + 6, (s + 1)*(s + 2), s)), (TransferFunction(s + 3, s**2 + 3*s + 2, s), TransferFunction(s**2 - 9*s + 20, s**2 + 5*s - 10, s))))
        >>> tfm_1.elem_zeros()
        [[[], [-6]], [[-3], [4, 5]]]

        See Also
        ========

        elem_poles

        """
        # 返回一个二维列表，其中每个元素是对应位置元素的零点列表
        return [[element.zeros() for element in row] for row in self.doit().args[0]]
    def eval_frequency(self, other):
        """
        Evaluates system response of each transfer function in the ``TransferFunctionMatrix`` at any point in the real or complex plane.

        Examples
        ========

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix
        >>> from sympy import I
        >>> tf_1 = TransferFunction(3, (s + 1), s)
        >>> tf_2 = TransferFunction(s + 6, (s + 1)*(s + 2), s)
        >>> tf_3 = TransferFunction(s + 3, s**2 + 3*s + 2, s)
        >>> tf_4 = TransferFunction(s**2 - 9*s + 20, s**2 + 5*s - 10, s)
        >>> tfm_1 = TransferFunctionMatrix([[tf_1, tf_2], [tf_3, tf_4]])
        >>> tfm_1
        TransferFunctionMatrix(((TransferFunction(3, s + 1, s), TransferFunction(s + 6, (s + 1)*(s + 2), s)), (TransferFunction(s + 3, s**2 + 3*s + 2, s), TransferFunction(s**2 - 9*s + 20, s**2 + 5*s - 10, s))))
        >>> tfm_1.eval_frequency(2)
        Matrix([
        [   1, 2/3],
        [5/12, 3/2]])
        >>> tfm_1.eval_frequency(I*2)
        Matrix([
        [   3/5 - 6*I/5,                -I],
        [3/20 - 11*I/20, -101/74 + 23*I/74]])
        """
        # 将 self._expr_mat 中的变量 self.var 替换为给定的 other，得到替换后的矩阵 mat
        mat = self._expr_mat.subs(self.var, other)
        # 展开矩阵中的表达式
        return mat.expand()

    def _flat(self):
        """Returns flattened list of args in TransferFunctionMatrix"""
        # 返回 TransferFunctionMatrix 中所有元素的扁平化列表
        return [elem for tup in self.args[0] for elem in tup]

    def _eval_evalf(self, prec):
        """Calls evalf() on each transfer function in the transfer function matrix"""
        # 将每个传递函数矩阵中的元素应用 evalf() 方法，以指定的精度 prec 进行数值评估
        dps = prec_to_dps(prec)
        mat = self._expr_mat.applyfunc(lambda a: a.evalf(n=dps))
        # 将评估后的结果重新转换为 TransferFunctionMatrix 类型
        return _to_TFM(mat, self.var)

    def _eval_simplify(self, **kwargs):
        """Simplifies the transfer function matrix"""
        # 对传递函数矩阵中的每个元素应用化简操作
        simp_mat = self._expr_mat.applyfunc(lambda a: cancel(a, expand=False))
        # 将化简后的结果重新转换为 TransferFunctionMatrix 类型
        return _to_TFM(simp_mat, self.var)

    def expand(self, **hints):
        """Expands the transfer function matrix"""
        # 对传递函数矩阵中的每个元素应用展开操作
        expand_mat = self._expr_mat.expand(**hints)
        # 将展开后的结果重新转换为 TransferFunctionMatrix 类型
        return _to_TFM(expand_mat, self.var)
# 定义 StateSpace 类，继承自 LinearTimeInvariant 类
class StateSpace(LinearTimeInvariant):
    """
    State space model (ssm) of a linear, time invariant control system.

    表示线性、时不变控制系统的标准状态空间模型，其中 A、B、C、D 是状态空间矩阵。
    这构成了线性控制系统的方程：
        (1) x'(t) = A * x(t) + B * u(t);    x 属于 R^n , u 属于 R^k
        (2) y(t)  = C * x(t) + D * u(t);    y 属于 R^m
    其中 u(t) 是任意输入信号，y(t) 是相应输出，x(t) 是系统的状态。

    Parameters
    ==========

    A : Matrix
        状态空间模型的状态矩阵。
    B : Matrix
        状态空间模型的输入到状态矩阵。
    C : Matrix
        状态空间模型的状态到输出矩阵。
    D : Matrix
        状态空间模型的馈送矩阵。

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.physics.control import StateSpace

    通过四个矩阵创建 StateSpaceModel 的最简单方式是：

    >>> A = Matrix([[1, 2], [1, 0]])
    >>> B = Matrix([1, 1])
    >>> C = Matrix([[0, 1]])
    >>> D = Matrix([0])
    >>> StateSpace(A, B, C, D)
    StateSpace(Matrix([
    [1, 2],
    [1, 0]]), Matrix([
    [1],
    [1]]), Matrix([[0, 1]]), Matrix([[0]]))


    也可以使用更少的矩阵。其余部分将填充为最少的零：

    >>> StateSpace(A, B)
    StateSpace(Matrix([
    [1, 2],
    [1, 0]]), Matrix([
    [1],
    [1]]), Matrix([[0, 0]]), Matrix([[0]]))


    See Also
    ========

    TransferFunction, TransferFunctionMatrix

    References
    ==========
    .. [1] https://en.wikipedia.org/wiki/State-space_representation
    .. [2] https://in.mathworks.com/help/control/ref/ss.html

    """
    def __new__(cls, A=None, B=None, C=None, D=None):
        # 如果未提供 A，则创建一个 1x1 的零矩阵
        if A is None:
            A = zeros(1)
        # 如果未提供 B，则创建一个与 A 行数相同的列向量的零矩阵
        if B is None:
            B = zeros(A.rows, 1)
        # 如果未提供 C，则创建一个与 A 列数相同的行向量的零矩阵
        if C is None:
            C = zeros(1, A.cols)
        # 如果未提供 D，则创建一个与 C 行数和 B 列数相同的零矩阵
        if D is None:
            D = zeros(C.rows, B.cols)

        # 将输入的 A、B、C、D 转换为 sympy 的矩阵类型
        A = _sympify(A)
        B = _sympify(B)
        C = _sympify(C)
        D = _sympify(D)

        # 检查输入的 A、B、C、D 是否都是 ImmutableDenseMatrix 类型
        if (isinstance(A, ImmutableDenseMatrix) and isinstance(B, ImmutableDenseMatrix) and
            isinstance(C, ImmutableDenseMatrix) and isinstance(D, ImmutableDenseMatrix)):
            # 检查状态矩阵 A 是否为方阵
            if A.rows != A.cols:
                raise ShapeError("Matrix A must be a square matrix.")

            # 检查状态矩阵 A 和输入矩阵 B 的行数是否相同
            if A.rows != B.rows:
                raise ShapeError("Matrices A and B must have the same number of rows.")

            # 检查输出矩阵 C 和传递矩阵 D 的行数是否相同
            if C.rows != D.rows:
                raise ShapeError("Matrices C and D must have the same number of rows.")

            # 检查状态矩阵 A 和输出矩阵 C 的列数是否相同
            if A.cols != C.cols:
                raise ShapeError("Matrices A and C must have the same number of columns.")

            # 检查输入矩阵 B 和传递矩阵 D 的列数是否相同
            if B.cols != D.cols:
                raise ShapeError("Matrices B and D must have the same number of columns.")

            # 创建 StateSpace 对象，并设置内部变量
            obj = super(StateSpace, cls).__new__(cls, A, B, C, D)
            obj._A = A
            obj._B = B
            obj._C = C
            obj._D = D

            # 确定系统是 SISO 还是 MIMO
            num_outputs = D.rows
            num_inputs = D.cols
            if num_inputs == 1 and num_outputs == 1:
                obj._is_SISO = True
                obj._clstype = SISOLinearTimeInvariant
            else:
                obj._is_SISO = False
                obj._clstype = MIMOLinearTimeInvariant

            return obj

        else:
            # 如果输入的 A、B、C、D 不全是 sympy 矩阵类型，则抛出 TypeError
            raise TypeError("A, B, C and D inputs must all be sympy Matrices.")

    @property
    def state_matrix(self):
        """
        Returns the state matrix of the model.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[1, 2], [1, 0]])
        >>> B = Matrix([1, 1])
        >>> C = Matrix([[0, 1]])
        >>> D = Matrix([0])
        >>> ss = StateSpace(A, B, C, D)
        >>> ss.state_matrix
        Matrix([
        [1, 2],
        [1, 0]])

        """
        return self._A

    @property
    def input_matrix(self):
        """
        Returns the input matrix of the model.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[1, 2], [1, 0]])
        >>> B = Matrix([1, 1])
        >>> C = Matrix([[0, 1]])
        >>> D = Matrix([0])
        >>> ss = StateSpace(A, B, C, D)
        >>> ss.input_matrix
        Matrix([
        [1],
        [1]])

        """
        # 返回模型的输入矩阵
        return self._B

    @property
    def output_matrix(self):
        """
        Returns the output matrix of the model.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[1, 2], [1, 0]])
        >>> B = Matrix([1, 1])
        >>> C = Matrix([[0, 1]])
        >>> D = Matrix([0])
        >>> ss = StateSpace(A, B, C, D)
        >>> ss.output_matrix
        Matrix([[0, 1]])

        """
        # 返回模型的输出矩阵
        return self._C

    @property
    def feedforward_matrix(self):
        """
        Returns the feedforward matrix of the model.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[1, 2], [1, 0]])
        >>> B = Matrix([1, 1])
        >>> C = Matrix([[0, 1]])
        >>> D = Matrix([0])
        >>> ss = StateSpace(A, B, C, D)
        >>> ss.feedforward_matrix
        Matrix([[0]])

        """
        # 返回模型的前馈矩阵
        return self._D

    @property
    def num_states(self):
        """
        Returns the number of states of the model.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[1, 2], [1, 0]])
        >>> B = Matrix([1, 1])
        >>> C = Matrix([[0, 1]])
        >>> D = Matrix([0])
        >>> ss = StateSpace(A, B, C, D)
        >>> ss.num_states
        2

        """
        # 返回模型的状态数
        return self._A.rows

    @property
    def num_inputs(self):
        """
        Returns the number of inputs of the model.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[1, 2], [1, 0]])
        >>> B = Matrix([1, 1])
        >>> C = Matrix([[0, 1]])
        >>> D = Matrix([0])
        >>> ss = StateSpace(A, B, C, D)
        >>> ss.num_inputs
        1

        """
        # 返回模型的输入数量
        return self._D.cols

    @property
    def num_outputs(self):
        """
        Returns the number of outputs of the model.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[1, 2], [1, 0]])
        >>> B = Matrix([1, 1])
        >>> C = Matrix([[0, 1]])
        >>> D = Matrix([0])
        >>> ss = StateSpace(A, B, C, D)
        >>> ss.num_outputs
        1

        """
        # 返回模型的输出数量
        return self._D.rows
    # 定义一个方法 `shape`，用于返回等效 StateSpace 系统的形状信息
    def shape(self):
        """Returns the shape of the equivalent StateSpace system."""
        # 返回当前对象的输出数量和输入数量作为形状信息
        return self.num_outputs, self.num_inputs
    def dsolve(self, initial_conditions=None, input_vector=None, var=Symbol('t')):
        r"""
        Returns `y(t)` or output of StateSpace given by the solution of equations:
            x'(t) = A * x(t) + B * u(t)
            y(t)  = C * x(t) + D * u(t)

        Parameters
        ============

        initial_conditions : Matrix
            The initial conditions of `x` state vector. If not provided, it defaults to a zero vector.
        input_vector : Matrix
            The input vector for state space. If not provided, it defaults to a zero vector.
        var : Symbol
            The symbol representing time. If not provided, it defaults to `t`.

        Examples
        ==========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[-2, 0], [1, -1]])
        >>> B = Matrix([[1], [0]])
        >>> C = Matrix([[2, 1]])
        >>> ip = Matrix([5])
        >>> i = Matrix([0, 0])
        >>> ss = StateSpace(A, B, C)
        >>> ss.dsolve(input_vector=ip, initial_conditions=i).simplify()
        Matrix([[15/2 - 5*exp(-t) - 5*exp(-2*t)/2]])

        If no input is provided it defaults to solving the system with zero initial conditions and zero input.

        >>> ss.dsolve()
        Matrix([[0]])

        References
        ==========
        .. [1] https://web.mit.edu/2.14/www/Handouts/StateSpaceResponse.pdf
        .. [2] https://docs.sympy.org/latest/modules/solvers/ode.html#sympy.solvers.ode.systems.linodesolve
        """

        # Check if 'var' is an instance of Symbol; raise an error if not
        if not isinstance(var, Symbol):
            raise ValueError("Variable for representing time must be a Symbol.")

        # Set default initial conditions if not provided
        if not initial_conditions:
            initial_conditions = zeros(self._A.shape[0], 1)
        # Raise ShapeError if initial conditions vector does not match the state matrix size
        elif initial_conditions.shape != (self._A.shape[0], 1):
            raise ShapeError("Initial condition vector should have the same number of "
                             "rows as the state matrix.")

        # Set default input vector if not provided
        if not input_vector:
            input_vector = zeros(self._B.shape[1], 1)
        # Raise ShapeError if input vector does not match the input matrix size
        elif input_vector.shape != (self._B.shape[1], 1):
            raise ShapeError("Input vector should have the same number of "
                             "columns as the input matrix.")

        # Solve the linear ordinary differential equations using linodesolve
        sol = linodesolve(A=self._A, t=var, b=self._B*input_vector, type='type2', doit=True)
        mat1 = Matrix(sol)
        
        # Replace the variable 'var' with 0 in the solution matrix
        mat2 = mat1.replace(var, 0)

        # Get all free symbols from mat2 not in self._A, self._B, or input_vector
        free1 = self._A.free_symbols | self._B.free_symbols | input_vector.free_symbols
        free2 = mat2.free_symbols
        
        # Calculate dummy symbols that are in mat2 but not in free1
        dummy_symbols = list(free2 - free1)

        # Convert mat2 to a coefficient matrix for linear equations
        r1, r2 = linear_eq_to_matrix(mat2, dummy_symbols)

        # Solve the linear equations (r1 * x = -r2 - initial_conditions)
        s = linsolve((r1, initial_conditions + r2))
        
        # Retrieve the first solution tuple
        res_tuple = next(iter(s))

        # Substitute the values from res_tuple back into mat1
        for ind, v in enumerate(res_tuple):
            mat1 = mat1.replace(dummy_symbols[ind], v)

        # Calculate and return the resulting expression: C * mat1 + D * input_vector
        res = self._C * mat1 + self._D * input_vector
        return res
    def _eval_evalf(self, prec):
        """
        根据给定的精度，将数值表达式评估为浮点数。
        """
        dps = prec_to_dps(prec)
        # 使用指定的精度对矩阵的元素进行数值评估
        return StateSpace(
            self._A.evalf(n=dps),
            self._B.evalf(n=dps),
            self._C.evalf(n=dps),
            self._D.evalf(n=dps))

    def _eval_rewrite_as_TransferFunction(self, *args):
        """
        返回状态空间模型的等效传递函数表示。

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import TransferFunction, StateSpace
        >>> A = Matrix([[-5, -1], [3, -1]])
        >>> B = Matrix([2, 5])
        >>> C = Matrix([[1, 2]])
        >>> D = Matrix([0])
        >>> ss = StateSpace(A, B, C, D)
        >>> ss.rewrite(TransferFunction)
        [[TransferFunction(12*s + 59, s**2 + 6*s + 8, s)]]

        """
        s = Symbol('s')
        n = self._A.shape[0]
        I = eye(n)
        # 计算状态空间模型的传递函数表达式
        G = self._C * (s * I - self._A).solve(self._B) + self._D
        G = G.simplify()
        # 将每个表达式转换为对应的传递函数对象
        to_tf = lambda expr: TransferFunction.from_rational_expression(expr, s)
        # 生成传递函数对象的矩阵表示
        tf_mat = [[to_tf(expr) for expr in sublist] for sublist in G.tolist()]
        return tf_mat
    def __add__(self, other):
        """
        Add two State Space systems (parallel connection).

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A1 = Matrix([[1]])
        >>> B1 = Matrix([[2]])
        >>> C1 = Matrix([[-1]])
        >>> D1 = Matrix([[-2]])
        >>> A2 = Matrix([[-1]])
        >>> B2 = Matrix([[-2]])
        >>> C2 = Matrix([[1]])
        >>> D2 = Matrix([[2]])
        >>> ss1 = StateSpace(A1, B1, C1, D1)
        >>> ss2 = StateSpace(A2, B2, C2, D2)
        >>> ss1 + ss2
        StateSpace(Matrix([
        [1,  0],
        [0, -1]]), Matrix([
        [ 2],
        [-2]]), Matrix([[-1, 1]]), Matrix([[0]]))

        """
        # 检查是否为标量
        if isinstance(other, (int, float, complex, Symbol)):
            A = self._A
            B = self._B
            C = self._C
            D = self._D.applyfunc(lambda element: element + other)

        else:
            # 检查系统的性质
            if not isinstance(other, StateSpace):
                raise ValueError("Addition is only supported for 2 State Space models.")
            # 检查系统的维度
            elif ((self.num_inputs != other.num_inputs) or (self.num_outputs != other.num_outputs)):
                raise ShapeError("Systems with incompatible inputs and outputs cannot be added.")

            # 构建新的状态空间系统
            m1 = (self._A).row_join(zeros(self._A.shape[0], other._A.shape[-1]))
            m2 = zeros(other._A.shape[0], self._A.shape[-1]).row_join(other._A)
            A = m1.col_join(m2)
            B = self._B.col_join(other._B)
            C = self._C.row_join(other._C)
            D = self._D + other._D

        return StateSpace(A, B, C, D)

    def __radd__(self, other):
        """
        Right add two State Space systems.

        Examples
        ========

        >>> from sympy.physics.control import StateSpace
        >>> s = StateSpace()
        >>> 5 + s
        StateSpace(Matrix([[0]]), Matrix([[0]]), Matrix([[0]]), Matrix([[5]]))

        """
        return self + other

    def __sub__(self, other):
        """
        Subtract two State Space systems.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A1 = Matrix([[1]])
        >>> B1 = Matrix([[2]])
        >>> C1 = Matrix([[-1]])
        >>> D1 = Matrix([[-2]])
        >>> A2 = Matrix([[-1]])
        >>> B2 = Matrix([[-2]])
        >>> C2 = Matrix([[1]])
        >>> D2 = Matrix([[2]])
        >>> ss1 = StateSpace(A1, B1, C1, D1)
        >>> ss2 = StateSpace(A2, B2, C2, D2)
        >>> ss1 - ss2
        StateSpace(Matrix([
        [1,  0],
        [0, -1]]), Matrix([
        [ 2],
        [-2]]), Matrix([[-1, -1]]), Matrix([[-4]]))

        """
        # 调用 __add__ 方法来实现减法
        return self + (-other)
    def __rsub__(self, other):
        """
        Right subtract two State Space systems.

        Examples
        ========

        >>> from sympy.physics.control import StateSpace
        >>> s = StateSpace()
        >>> 5 - s
        StateSpace(Matrix([[0]]), Matrix([[0]]), Matrix([[0]]), Matrix([[5]]))

        """
        # 返回右减操作的结果，实际上是 other + (-self)
        return other + (-self)

    def __neg__(self):
        """
        Returns the negation of the state space model.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[-5, -1], [3, -1]])
        >>> B = Matrix([2, 5])
        >>> C = Matrix([[1, 2]])
        >>> D = Matrix([0])
        >>> ss = StateSpace(A, B, C, D)
        >>> -ss
        StateSpace(Matrix([
        [-5, -1],
        [ 3, -1]]), Matrix([
        [2],
        [5]]), Matrix([[-1, -2]]), Matrix([[0]]))

        """
        # 返回状态空间模型的负值，即取负后的矩阵 C 和 D
        return StateSpace(self._A, self._B, -self._C, -self._D)

    def __mul__(self, other):
        """
        Multiplication of two State Space systems (serial connection).

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[-5, -1], [3, -1]])
        >>> B = Matrix([2, 5])
        >>> C = Matrix([[1, 2]])
        >>> D = Matrix([0])
        >>> ss = StateSpace(A, B, C, D)
        >>> ss*5
        StateSpace(Matrix([
        [-5, -1],
        [ 3, -1]]), Matrix([
        [2],
        [5]]), Matrix([[5, 10]]), Matrix([[0]]))

        """
        # 检查是否为标量
        if isinstance(other, (int, float, complex, Symbol)):
            # 如果是标量，对 C 和 D 进行标量乘法
            A = self._A
            B = self._B
            C = self._C.applyfunc(lambda element: element * other)
            D = self._D.applyfunc(lambda element: element * other)

        else:
            # 如果不是标量，检查是否为 StateSpace 类型
            if not isinstance(other, StateSpace):
                raise ValueError("Multiplication is only supported for 2 State Space models.")
            # 检查系统维度是否兼容
            elif self.num_inputs != other.num_outputs:
                raise ShapeError("Systems with incompatible inputs and outputs cannot be multiplied.")

            # 构造串联连接的状态空间模型
            m1 = (other._A).row_join(zeros(other._A.shape[0], self._A.shape[1]))
            m2 = (self._B * other._C).row_join(self._A)

            A = m1.col_join(m2)
            B = (other._B).col_join(self._B * other._D)
            C = (self._D * other._C).row_join(self._C)
            D = self._D * other._D

        return StateSpace(A, B, C, D)
    def __rmul__(self, other):
        """
        Right multiply two StateSpace systems.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[-5, -1], [3, -1]])
        >>> B = Matrix([2, 5])
        >>> C = Matrix([[1, 2]])
        >>> D = Matrix([0])
        >>> ss = StateSpace(A, B, C, D)
        >>> 5*ss
        StateSpace(Matrix([
        [-5, -1],
        [ 3, -1]]), Matrix([
        [10],
        [25]]), Matrix([[1, 2]]), Matrix([[0]]))

        """
        # 如果 `other` 是整数、浮点数、复数或符号类型
        if isinstance(other, (int, float, complex, Symbol)):
            # 获取当前对象的状态空间矩阵和输出矩阵
            A = self._A
            C = self._C
            # 将输入矩阵和传入的 `other` 相乘，并返回新的输入矩阵
            B = self._B.applyfunc(lambda element: element * other)
            # 将输出矩阵和传入的 `other` 相乘，并返回新的输出矩阵
            D = self._D.applyfunc(lambda element: element * other)
            # 返回一个新的 StateSpace 对象，使用修改后的状态空间矩阵和输出矩阵
            return StateSpace(A, B, C, D)
        else:
            # 如果 `other` 不是支持的类型，则调用原始的乘法操作
            return self * other

    def __repr__(self):
        # 获取当前对象的状态空间矩阵和输入、输出矩阵的字符串表示
        A_str = self._A.__repr__()
        B_str = self._B.__repr__()
        C_str = self._C.__repr__()
        D_str = self._D.__repr__()

        # 返回一个字符串，描述当前 StateSpace 对象的状态空间模型
        return f"StateSpace(\n{A_str},\n\n{B_str},\n\n{C_str},\n\n{D_str})"


    def append(self, other):
        """
        Returns the first model appended with the second model. The order is preserved.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A1 = Matrix([[1]])
        >>> B1 = Matrix([[2]])
        >>> C1 = Matrix([[-1]])
        >>> D1 = Matrix([[-2]])
        >>> A2 = Matrix([[-1]])
        >>> B2 = Matrix([[-2]])
        >>> C2 = Matrix([[1]])
        >>> D2 = Matrix([[2]])
        >>> ss1 = StateSpace(A1, B1, C1, D1)
        >>> ss2 = StateSpace(A2, B2, C2, D2)
        >>> ss1.append(ss2)
        StateSpace(Matrix([
        [1,  0],
        [0, -1]]), Matrix([
        [2,  0],
        [0, -2]]), Matrix([
        [-1, 0],
        [ 0, 1]]), Matrix([
        [-2, 0],
        [ 0, 2]]))

        """
        # 计算合并后的状态空间模型的维度
        n = self.num_states + other.num_states
        m = self.num_inputs + other.num_inputs
        p = self.num_outputs + other.num_outputs

        # 初始化合并后的状态空间模型的矩阵
        A = zeros(n, n)
        B = zeros(n, m)
        C = zeros(p, n)
        D = zeros(p, m)

        # 将第一个模型的状态空间矩阵和输入矩阵复制到合并后的模型中相应位置
        A[:self.num_states, :self.num_states] = self._A
        B[:self.num_states, :self.num_inputs] = self._B
        C[:self.num_outputs, :self.num_states] = self._C
        D[:self.num_outputs, :self.num_inputs] = self._D

        # 将第二个模型的状态空间矩阵和输入矩阵复制到合并后的模型中相应位置
        A[self.num_states:, self.num_states:] = other._A
        B[self.num_states:, self.num_inputs:] = other._B
        C[self.num_outputs:, self.num_states:] = other._C
        D[self.num_outputs:, self.num_inputs:] = other._D

        # 返回一个新的 StateSpace 对象，表示合并后的模型
        return StateSpace(A, B, C, D)
    def observability_matrix(self):
        """
        Returns the observability matrix of the state space model:
            [C, C * A^1, C * A^2, .. , C * A^(n-1)]; A in R^(n x n), C in R^(m x k)

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[-1.5, -2], [1, 0]])
        >>> B = Matrix([0.5, 0])
        >>> C = Matrix([[0, 1]])
        >>> D = Matrix([1])
        >>> ss = StateSpace(A, B, C, D)
        >>> ob = ss.observability_matrix()
        >>> ob
        Matrix([
        [0, 1],
        [1, 0]])

        References
        ==========
        .. [1] https://in.mathworks.com/help/control/ref/statespacemodel.obsv.html

        """
        n = self.num_states  # 获取状态空间模型的状态数
        ob = self._C  # 初始时，观测矩阵为 C
        for i in range(1, n):
            ob = ob.col_join(self._C * self._A**i)  # 将 C * A^i 添加到观测矩阵中

        return ob  # 返回构建好的观测矩阵

    def observable_subspace(self):
        """
        Returns the observable subspace of the state space model.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[-1.5, -2], [1, 0]])
        >>> B = Matrix([0.5, 0])
        >>> C = Matrix([[0, 1]])
        >>> D = Matrix([1])
        >>> ss = StateSpace(A, B, C, D)
        >>> ob_subspace = ss.observable_subspace()
        >>> ob_subspace
        [Matrix([
        [0],
        [1]]), Matrix([
        [1],
        [0]])]

        """
        return self.observability_matrix().columnspace()  # 返回观测矩阵的列空间，即可观测子空间

    def is_observable(self):
        """
        Returns if the state space model is observable.

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[-1.5, -2], [1, 0]])
        >>> B = Matrix([0.5, 0])
        >>> C = Matrix([[0, 1]])
        >>> D = Matrix([1])
        >>> ss = StateSpace(A, B, C, D)
        >>> ss.is_observable()
        True

        """
        return self.observability_matrix().rank() == self.num_states  # 判断观测矩阵的秩是否等于状态数，以确定是否可观测

    def controllability_matrix(self):
        """
        Returns the controllability matrix of the system:
            [B, A * B, A^2 * B, .. , A^(n-1) * B]; A in R^(n x n), B in R^(n x m)

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.physics.control import StateSpace
        >>> A = Matrix([[-1.5, -2], [1, 0]])
        >>> B = Matrix([0.5, 0])
        >>> C = Matrix([[0, 1]])
        >>> D = Matrix([1])
        >>> ss = StateSpace(A, B, C, D)
        >>> ss.controllability_matrix()
        Matrix([
        [0.5, -0.75],
        [  0,   0.5]])

        References
        ==========
        .. [1] https://in.mathworks.com/help/control/ref/statespacemodel.ctrb.html

        """
        co = self._B  # 初始时，控制能力矩阵为 B
        n = self._A.shape[0]  # 获取状态矩阵 A 的行数，即状态空间的维度
        for i in range(1, n):
            co = co.row_join(((self._A)**i) * self._B)  # 将 A^i * B 添加到控制能力矩阵中

        return co  # 返回构建好的控制能力矩阵
    def controllable_subspace(self):
        """
        Returns the controllable subspace of the state space model.
        """

        # 调用controllability_matrix方法获取控制性矩阵的列空间
        return self.controllability_matrix().columnspace()

    def is_controllable(self):
        """
        Returns if the state space model is controllable.
        """

        # 检查控制性矩阵的秩是否等于状态数量，判断模型是否可控
        return self.controllability_matrix().rank() == self.num_states
```