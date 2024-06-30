# `D:\src\scipysrc\sympy\sympy\core\evalf.py`

```
"""
Adaptive numerical evaluation of SymPy expressions, using mpmath
for mathematical functions.
"""
# 从未来版本导入类型提示的注解
from __future__ import annotations
from typing import Tuple as tTuple, Optional, Union as tUnion, Callable, List, Dict as tDict, Type, TYPE_CHECKING, \
    Any, overload

# 导入标准的数学库
import math

# 导入 mpmath 库及其特定模块
import mpmath.libmp as libmp
from mpmath import (
    make_mpc, make_mpf, mp, mpc, mpf, nsum, quadts, quadosc, workprec)
from mpmath import inf as mpmath_inf
from mpmath.libmp import (from_int, from_man_exp, from_rational, fhalf,
        fnan, finf, fninf, fnone, fone, fzero, mpf_abs, mpf_add,
        mpf_atan, mpf_atan2, mpf_cmp, mpf_cos, mpf_e, mpf_exp, mpf_log, mpf_lt,
        mpf_mul, mpf_neg, mpf_pi, mpf_pow, mpf_pow_int, mpf_shift, mpf_sin,
        mpf_sqrt, normalize, round_nearest, to_int, to_str)
from mpmath.libmp import bitcount as mpmath_bitcount
from mpmath.libmp.backend import MPZ
from mpmath.libmp.libmpc import _infs_nan
from mpmath.libmp.libmpf import dps_to_prec, prec_to_dps

# 导入本地的 SymPy 模块
from .sympify import sympify
from .singleton import S
from sympy.external.gmpy import SYMPY_INTS
from sympy.utilities.iterables import is_sequence
from sympy.utilities.lambdify import lambdify
from sympy.utilities.misc import as_int

# 如果是类型检查模式，导入额外的 SymPy 类型
if TYPE_CHECKING:
    from sympy.core.expr import Expr
    from sympy.core.add import Add
    from sympy.core.mul import Mul
    from sympy.core.power import Pow
    from sympy.core.symbol import Symbol
    from sympy.integrals.integrals import Integral
    from sympy.concrete.summations import Sum
    from sympy.concrete.products import Product
    from sympy.functions.elementary.exponential import exp, log
    from sympy.functions.elementary.complexes import Abs, re, im
    from sympy.functions.elementary.integers import ceiling, floor
    from sympy.functions.elementary.trigonometric import atan
    from .numbers import Float, Rational, Integer, AlgebraicNumber, Number

# 计算常数 lg(10)
LG10 = math.log2(10)
# 定义四舍五入函数
rnd = round_nearest


def bitcount(n):
    """Return smallest integer, b, such that |n|/2**b < 1.
    """
    # 使用 mpmath 库计算整数 n 的二进制位数
    return mpmath_bitcount(abs(int(n)))

# 以下是一些常用的占位符值，用于表示指数和精度级别，例如精确数的指数。必须注意避免将它们传递给 mpmath 函数或在最终结果中返回它们。
# 正无穷
INF = float(mpmath_inf)
# 负无穷
MINUS_INF = float(-mpmath_inf)

# 默认最大精度约为 100 位数字。真正的专家将其设置为正无穷。
DEFAULT_MAXPREC = 333


class PrecisionExhausted(ArithmeticError):
    pass

#----------------------------------------------------------------------------#
#                                                                            #
#              Helper functions for arithmetic and complex parts             #
#                                                                            #
#----------------------------------------------------------------------------#

"""
An mpf value tuple is a tuple of integers (sign, man, exp, bc)
representing a floating-point number: [1, -1][sign]*man*2**exp where
"""
# MPF_TUP 是一个类型别名，表示一个包含四个整数的元组，用于表示 mpf 值
MPF_TUP = tTuple[int, int, int, int]  # mpf value tuple

# TMP_RES 是一个类型别名，表示临时结果的类型，可能是以下几种之一：
# 1. 包含两个非零的 MPF_TUP 元组（近似数值）
# 2. 包含两个 None（表示精确零）
# 3. 包含两个整数（表示对应复数部分的估计相对精度）
# 4. 'ComplexInfinity'（表示无穷大）
TMP_RES = Any  # temporary result, should be some variant of
# tUnion[tTuple[Optional[MPF_TUP], Optional[MPF_TUP],
#               Optional[int], Optional[int]],
#        'ComplexInfinity']
# but mypy reports error because it doesn't know as we know
# 1. re and re_acc are either both None or both MPF_TUP
# 2. sometimes the result can't be zoo

# OPT_DICT 是一个类型别名，表示内部 evalf 函数中的 "options" 参数的类型，
# 它是一个字符串到任意类型的字典
OPT_DICT = tDict[str, Any]


def fastlog(x: Optional[MPF_TUP]) -> tUnion[int, Any]:
    """Fast approximation of log2(x) for an mpf value tuple x.

    Explanation
    ===========

    快速计算 mpf 值元组 x 的 log2(x) 近似值。
    计算结果为指数加上尾数的宽度。这是一个近似值，原因如下：
    1）它给出 ceil(log2(abs(x))) 的值；
    2）如果 x 是 2 的精确幂，它将比实际值多 1。
    虽然可以通过检测奇数尾数为1来纠正（这表明处理的是精确的 2 的幂次方），
    但这会降低速度，不是必需的，因为这仅作为 x 中位数数量的近似值使用。
    正确的返回值可以写成 "x[2] + (x[3] if x[1] != 1 else 0)"。
        因为 mpf 元组的尾数始终是奇数，所以不需要检查尾数是否是2的倍数
    （如果是，结果将比实际值多1）。

    Examples
    ========

    >>> from sympy import log
    >>> from sympy.core.evalf import fastlog, bitcount
    >>> s, m, e = 0, 5, 1
    >>> bc = bitcount(m)
    >>> n = [1, -1][s]*m*2**e
    >>> n, (log(n)/log(2)).evalf(2), fastlog((s, m, e, bc))
    (10, 3.3, 4)
    """

    # 如果 x 为空或者等于 fzero，则返回负无穷
    if not x or x == fzero:
        return MINUS_INF
    # 返回 x 的指数加上尾数的宽度
    return x[2] + x[3]


def pure_complex(v: 'Expr', or_real=False) -> tuple['Number', 'Number'] | None:
    """Return a and b if v matches a + I*b where b is not zero and
    a and b are Numbers, else None. If `or_real` is True then 0 will
    be returned for `b` if `v` is a real number.

    Examples
    ========

    >>> from sympy.core.evalf import pure_complex
    >>> from sympy import sqrt, I, S
    >>> a, b, surd = S(2), S(3), sqrt(2)
    >>> pure_complex(a)
    >>> pure_complex(a, or_real=True)
    (2, 0)
    >>> pure_complex(surd)
    >>> pure_complex(a + b*I)
    (2, 3)
    >>> pure_complex(I)
    """

    # 如果 v 符合形式 a + I*b，并且 b 不为零，返回 (a, b)
    # 其中 a 和 b 是数值类型，否则返回 None。
    # 如果 or_real 为 True，并且 v 是实数，则返回 (a, 0)
    return None
    (0, 1)
    """
    # 解压元组 v 并将其分配给变量 h 和 t
    h, t = v.as_coeff_Add()
    
    # 如果 t 不为空，则继续执行下面的条件判断
    if t:
        # 尝试将 t 拆分为系数和乘积项
        c, i = t.as_coeff_Mul()
        
        # 如果乘积项 i 是虚数单位 S.ImaginaryUnit
        if i is S.ImaginaryUnit:
            # 返回 h（实数部分）和 c（虚数部分）
            return h, c
    
    # 如果 or_real 参数为真，则返回 h 和零值
    elif or_real:
        return h, S.Zero
    
    # 其他情况返回 None
    return None
# 定义一个类型别名，表示一个具有特定结构的元组
SCALED_ZERO_TUP = tTuple[List[int], int, int, int]

# 函数装饰器，用于指定函数的重载形式，声明了两个重载版本的函数 scaled_zero
@overload
def scaled_zero(mag: SCALED_ZERO_TUP, sign=1) -> MPF_TUP:
    ...

@overload
def scaled_zero(mag: int, sign=1) -> tTuple[SCALED_ZERO_TUP, int]:
    ...

# 实际的函数定义，接受一个整数或者一个特定结构的元组 mag 作为参数，并返回一个元组或者 MPF_TUP
def scaled_zero(mag: tUnion[SCALED_ZERO_TUP, int], sign=1) -> tUnion[MPF_TUP, tTuple[SCALED_ZERO_TUP, int]]:
    """Return an mpf representing a power of two with magnitude ``mag``
    and -1 for precision. Or, if ``mag`` is a scaled_zero tuple, then just
    remove the sign from within the list that it was initially wrapped
    in.

    Examples
    ========

    >>> from sympy.core.evalf import scaled_zero
    >>> from sympy import Float
    >>> z, p = scaled_zero(100)
    >>> z, p
    (([0], 1, 100, 1), -1)
    >>> ok = scaled_zero(z)
    >>> ok
    (0, 1, 100, 1)
    >>> Float(ok)
    1.26765060022823e+30
    >>> Float(ok, p)
    0.e+30
    >>> ok, p = scaled_zero(100, -1)
    >>> Float(scaled_zero(ok), p)
    -0.e+30
    """
    # 如果 mag 是一个符合特定结构的元组且其中的数值全为零，则返回去掉符号后的列表
    if isinstance(mag, tuple) and len(mag) == 4 and iszero(mag, scaled=True):
        return (mag[0][0],) + mag[1:]
    # 如果 mag 是整数类型
    elif isinstance(mag, SYMPY_INTS):
        # 检查符号是否为有效值
        if sign not in [-1, 1]:
            raise ValueError('sign must be +/-1')
        # 调用 mpf_shift 函数，生成一个表示 2 的幂的数值，并设置精度为 -1
        rv, p = mpf_shift(fone, mag), -1
        # 根据符号设置返回值的列表
        s = 0 if sign == 1 else 1
        rv = ([s],) + rv[1:]
        return rv, p
    else:
        raise ValueError('scaled zero expects int or scaled_zero tuple.')

# 函数，判断给定的数值是否为零
def iszero(mpf: tUnion[MPF_TUP, SCALED_ZERO_TUP, None], scaled=False) -> Optional[bool]:
    # 如果 scaled 参数为 False，则判断 mpf 是否为空或者其第一个和最后一个元素是否为零
    if not scaled:
        return not mpf or not mpf[1] and not mpf[-1]
    # 否则，判断 mpf 是否存在且其第一个元素为列表且第二个和最后一个元素均为 1
    return mpf and isinstance(mpf[0], list) and mpf[1] == mpf[-1] == 1

# 函数，计算复数的相对精度
def complex_accuracy(result: TMP_RES) -> tUnion[int, Any]:
    """
    Returns relative accuracy of a complex number with given accuracies
    for the real and imaginary parts. The relative accuracy is defined
    in the complex norm sense as ||z|+|error|| / |z| where error
    is equal to (real absolute error) + (imag absolute error)*i.

    The full expression for the (logarithmic) error can be approximated
    easily by using the max norm to approximate the complex norm.

    In the worst case (re and im equal), this is wrong by a factor
    sqrt(2), or by log2(sqrt(2)) = 0.5 bit.
    """
    # 如果结果是复数无穷大，则返回无穷大
    if result is S.ComplexInfinity:
        return INF
    # 解包结果元组
    re, im, re_acc, im_acc = result
    # 如果虚部为零
    if not im:
        # 如果实部也为零，则返回无穷大
        if not re:
            return INF
        # 否则返回实部的精度
        return re_acc
    # 如果实部为零，返回虚部的精度
    if not re:
        return im_acc
    # 计算实部和虚部的尺寸
    re_size = fastlog(re)
    im_size = fastlog(im)
    # 计算绝对误差
    absolute_error = max(re_size - re_acc, im_size - im_acc)
    # 计算相对误差
    relative_error = absolute_error - max(re_size, im_size)
    return -relative_error

# 函数，计算表达式的绝对值
def get_abs(expr: 'Expr', prec: int, options: OPT_DICT) -> TMP_RES:
    # 调用 evalf 函数计算表达式的浮点数表示，精度为 prec+2，并带有指定的选项
    result = evalf(expr, prec + 2, options)
    # 如果结果是复数无穷大，则返回特定的结果
    if result is S.ComplexInfinity:
        return finf, None, prec, None
    # 否则，返回解包后的结果元组
    re, im, re_acc, im_acc = result
    # 如果 re 为假值（None 或 False），则交换 re 和 im 变量及其对应的累积变量
    if not re:
        re, re_acc, im, im_acc = im, im_acc, re, re_acc
    
    # 如果 im 不为空（非空值），则执行以下逻辑
    if im:
        # 如果表达式是一个数字
        if expr.is_number:
            # 计算表达式的绝对值并返回，同时计算精度为 prec + 2
            abs_expr, _, acc, _ = evalf(abs(N(expr, prec + 2)),
                                        prec + 2, options)
            return abs_expr, None, acc, None
        else:
            # 如果 options 中包含 'subs'，则调用 libmp.mpc_abs 计算复数的绝对值
            if 'subs' in options:
                return libmp.mpc_abs((re, im), prec), None, re_acc, None
            # 否则，直接返回表达式的绝对值，精度为 prec
            return abs(expr), None, prec, None
    
    # 如果 im 为空但 re 不为空，则执行以下逻辑
    elif re:
        # 返回 re 的绝对值
        return mpf_abs(re), None, re_acc, None
    
    # 如果 im 和 re 都为空，则返回四个 None
    else:
        return None, None, None, None
# 获取复数表达式的实部或虚部的函数，根据参数 `no` 的值来决定返回实部或虚部
def get_complex_part(expr: 'Expr', no: int, prec: int, options: OPT_DICT) -> TMP_RES:
    """no = 0 for real part, no = 1 for imaginary part"""
    # 设置工作精度为指定的精度值
    workprec = prec
    # 初始化迭代次数计数器
    i = 0
    while 1:
        # 使用 evalf 函数计算表达式在给定精度下的数值
        res = evalf(expr, workprec, options)
        # 如果结果为复数无穷大，则返回特定的结果
        if res is S.ComplexInfinity:
            return fnan, None, prec, None
        # 获取指定部分（实部或虚部）的值和精度
        value, accuracy = res[no::2]
        # XXX 是否最后一个正确？考虑 re((1+I)**2).n()
        # 如果值为空（0 或 None）、精度满足要求或者值的绝对值大于指定精度，则返回结果
        if (not value) or accuracy >= prec or -value[2] > prec:
            return value, None, accuracy, None
        # 增加工作精度，每次增加最小30或2的幂次方
        workprec += max(30, 2**i)
        i += 1


# 计算绝对值表达式的函数
def evalf_abs(expr: 'Abs', prec: int, options: OPT_DICT) -> TMP_RES:
    return get_abs(expr.args[0], prec, options)


# 计算实部表达式的函数
def evalf_re(expr: 're', prec: int, options: OPT_DICT) -> TMP_RES:
    return get_complex_part(expr.args[0], 0, prec, options)


# 计算虚部表达式的函数
def evalf_im(expr: 'im', prec: int, options: OPT_DICT) -> TMP_RES:
    return get_complex_part(expr.args[0], 1, prec, options)


# 处理复数最终结果的函数，根据实部和虚部的情况返回相应结果
def finalize_complex(re: MPF_TUP, im: MPF_TUP, prec: int) -> TMP_RES:
    # 如果实部和虚部都为零，则抛出异常
    if re == fzero and im == fzero:
        raise ValueError("got complex zero with unknown accuracy")
    # 如果实部为零，则返回虚部
    elif re == fzero:
        return None, im, None, prec
    # 如果虚部为零，则返回实部
    elif im == fzero:
        return re, None, prec, None

    # 计算实部和虚部的大小对数
    size_re = fastlog(re)
    size_im = fastlog(im)
    # 根据实部和虚部的大小决定精度
    if size_re > size_im:
        re_acc = prec
        im_acc = prec + min(-(size_re - size_im), 0)
    else:
        im_acc = prec
        re_acc = prec + min(-(size_im - size_re), 0)
    return re, im, re_acc, im_acc


# 去除结果中微小的实部或虚部的函数
def chop_parts(value: TMP_RES, prec: int) -> TMP_RES:
    """
    Chop off tiny real or complex parts.
    """
    # 如果值为复数无穷大，则直接返回该值
    if value is S.ComplexInfinity:
        return value
    # 获取实部、虚部及其精度
    re, im, re_acc, im_acc = value
    # 方法1：根据绝对值去除微小的实部
    if re and re not in _infs_nan and (fastlog(re) < -prec + 4):
        re, re_acc = None, None
    # 方法1：根据绝对值去除微小的虚部
    if im and im not in _infs_nan and (fastlog(im) < -prec + 4):
        im, im_acc = None, None
    # 方法2：如果实部和虚部都存在，则根据精度和大小差异去除微小部分
    if re and im:
        delta = fastlog(re) - fastlog(im)
        if re_acc < 2 and (delta - re_acc <= -prec + 4):
            re, re_acc = None, None
        if im_acc < 2 and (delta - im_acc >= prec - 4):
            im, im_acc = None, None
    return re, im, re_acc, im_acc


# 检查结果精度是否达到指定精度的函数
def check_target(expr: 'Expr', result: TMP_RES, prec: int):
    # 计算复数结果的精度
    a = complex_accuracy(result)
    # 如果精度小于指定精度，则抛出精度不足的异常
    if a < prec:
        raise PrecisionExhausted("Failed to distinguish the expression: \n\n%s\n\n"
            "from zero. Try simplifying the input, using chop=True, or providing "
            "a higher maxn for evalf" % (expr))


# 获取整数部分的函数，根据参数 no 的值决定计算表达式的上取整或下取整
def get_integer_part(expr: 'Expr', no: int, options: OPT_DICT, return_ints=False) -> \
        tUnion[TMP_RES, tTuple[int, int]]:
    """
    With no = 1, computes ceiling(expr)
    With no = -1, computes floor(expr)

    Note: this function either gives the exact result or signals failure.
    """
    from sympy.functions.elementary.complexes import re, im
    # 假设表达式的大小可能小于大约2^30
    assumed_size = 30
    # 使用给定的精度和选项计算表达式的值
    result = evalf(expr, assumed_size, options)
    # 如果结果是复数无穷大，则抛出值错误异常
    if result is S.ComplexInfinity:
        raise ValueError("Cannot get integer part of Complex Infinity")
    # 将结果解包为四个部分：实部的整数部分、虚部的整数部分、实部的精度、虚部的精度
    ire, iim, ire_acc, iim_acc = result

    # 现在已知大小，可以计算需要多少额外精度（如果有的话）以达到最接近的整数
    if ire and iim:
        gap = max(fastlog(ire) - ire_acc, fastlog(iim) - iim_acc)
    elif ire:
        gap = fastlog(ire) - ire_acc
    elif iim:
        gap = fastlog(iim) - iim_acc
    else:
        # 如果结果是零，根据返回整数的选项返回适当的值或者返回None
        if return_ints:
            return 0, 0
        else:
            return None, None, None, None

    # 定义一个边界值
    margin = 10

    # 如果计算得到的间隙大于等于边界值的负值
    if gap >= -margin:
        # 计算新的精度，确保包含所需的额外精度
        prec = margin + assumed_size + gap
        # 使用新的精度再次计算表达式的值
        ire, iim, ire_acc, iim_acc = evalf(
            expr, prec, options)
    else:
        # 否则，使用默认的预设精度
        prec = assumed_size

    # 现在可以轻松地找到最接近的整数，但是要找到 floor/ceil，还必须计算与最接近整数的差值的正负性（如果非常接近可能会失败）。
    # 定义函数 calc_part，计算表达式的整数部分
    def calc_part(re_im: 'Expr', nexpr: MPF_TUP):
        # 从模块中导入 Add 类
        from .add import Add
        # 解包 nexpr 获取指数信息
        _, _, exponent, _ = nexpr
        # 检查指数是否为整数
        is_int = exponent == 0
        # 将 nexpr 转换为整数
        nint = int(to_int(nexpr, rnd))

        # 如果指数为整数
        if is_int:
            # 确保我们有足够的精度来区分 nint 和传递给 calc_part 的表达式的 re_im 部分
            ire, iim, ire_acc, iim_acc = evalf(
                re_im - nint, 10, options)  # 不需要太多精度
            assert not iim
            # 计算 ire 的大小，-ve 表示 ire 小于 1
            size = -fastlog(ire) + 2
            # 如果大小超过预设精度 prec，则重新计算 ire
            if size > prec:
                ire, iim, ire_acc, iim_acc = evalf(
                    re_im, size, options)
                assert not iim
                nexpr = ire
            nint = int(to_int(nexpr, rnd))
            # 获取新的指数信息
            _, _, new_exp, _ = ire
            is_int = new_exp == 0

        # 如果不是整数
        if not is_int:
            # 如果存在替换选项，并且所有的替换值都包含整数 re/im 部分，则替换它们到表达式中
            s = options.get('subs', False)
            if s:
                # 使用 strict=False 与 as_int 一起，因为我们接受 2.0 == 2
                def is_int_reim(x):
                    """检查是否为整数或整数加虚部乘积的形式。"""
                    try:
                        as_int(x, strict=False)
                        return True
                    except ValueError:
                        try:
                            [as_int(i, strict=False) for i in x.as_real_imag()]
                            return True
                        except ValueError:
                            return False

                # 如果所有替换值都符合整数或整数加虚部乘积的形式，则进行替换
                if all(is_int_reim(v) for v in s.values()):
                    re_im = re_im.subs(s)

            # 将 re_im 加上 -nint，但不进行求值
            re_im = Add(re_im, -nint, evaluate=False)
            # 对 re_im 进行评估，获取其值和精度
            x, _, x_acc, _ = evalf(re_im, 10, options)
            # 检查是否达到目标精度，如果达不到且 re_im 不为零，则抛出 PrecisionExhausted 异常
            try:
                check_target(re_im, (x, None, x_acc, None), 3)
            except PrecisionExhausted:
                if not re_im.equals(0):
                    raise PrecisionExhausted
                x = fzero
            # 如果 x 或 fzero 与 no 相等，则将 nint 增加 no
            nint += int(no*(mpf_cmp(x or fzero, fzero) == no))

        # 将 nint 转换为符号表示
        nint = from_int(nint)
        # 返回计算结果和 INF
        return nint, INF

    # 初始化 re_, im_, re_acc, im_acc
    re_, im_, re_acc, im_acc = None, None, None, None

    # 如果存在 ire，则计算 re_ 和 re_acc
    if ire:
        re_, re_acc = calc_part(re(expr, evaluate=False), ire)
    # 如果存在 iim，则计算 im_ 和 im_acc
    if iim:
        im_, im_acc = calc_part(im(expr, evaluate=False), iim)

    # 如果需要返回整数，则返回 re_ 和 im_ 的整数表示
    if return_ints:
        return int(to_int(re_ or fzero)), int(to_int(im_ or fzero))
    # 否则返回 re_, im_, re_acc, im_acc
    return re_, im_, re_acc, im_acc
# 定义一个函数 evalf_ceiling，用于计算给定表达式的上整函数值
# 返回结果是调用 get_integer_part 函数处理后的结果
def evalf_ceiling(expr: 'ceiling', prec: int, options: OPT_DICT) -> TMP_RES:
    return get_integer_part(expr.args[0], 1, options)


# 定义一个函数 evalf_floor，用于计算给定表达式的下整函数值
# 返回结果是调用 get_integer_part 函数处理后的结果
def evalf_floor(expr: 'floor', prec: int, options: OPT_DICT) -> TMP_RES:
    return get_integer_part(expr.args[0], -1, options)


# 定义一个函数 evalf_float，用于返回给定浮点数表达式的精确浮点数表示
# 返回结果是浮点数的元组表示 (_mpf_, None, prec, None)
def evalf_float(expr: 'Float', prec: int, options: OPT_DICT) -> TMP_RES:
    return expr._mpf_, None, prec, None


# 定义一个函数 evalf_rational，用于返回给定有理数表达式的浮点数表示
# 返回结果是调用 from_rational 函数处理后的结果，以及精度信息
def evalf_rational(expr: 'Rational', prec: int, options: OPT_DICT) -> TMP_RES:
    return from_rational(expr.p, expr.q, prec), None, prec, None


# 定义一个函数 evalf_integer，用于返回给定整数表达式的浮点数表示
# 返回结果是调用 from_int 函数处理后的结果，以及精度信息
def evalf_integer(expr: 'Integer', prec: int, options: OPT_DICT) -> TMP_RES:
    return from_int(expr.p, prec), None, prec, None


#----------------------------------------------------------------------------#
#                                                                            #
#                            Arithmetic operations                           #
#                                                                            #
#----------------------------------------------------------------------------#


# 定义函数 add_terms，用于将一个由(mp浮点数值, 精度)组成的列表进行求和
# 返回结果根据情况有四种可能：
# - 如果没有非零项，则返回 None, None
# - 如果只有一项，则返回这一项
# - 如果求和后得到零（例如由于精度问题产生的），返回 scaled_zero
# - 否则返回一个根据 target_prec 缩放的元组，表示求和结果的浮点数值和精度
#
# 这个函数需要对特殊值如 NaN 和无穷大进行特殊处理，并且确保返回的浮点数元组被标准化到目标精度
def add_terms(terms: list, prec: int, target_prec: int) -> \
        tTuple[tUnion[MPF_TUP, SCALED_ZERO_TUP, None], Optional[int]]:
    """
    Helper for evalf_add. Adds a list of (mpfval, accuracy) terms.

    Returns
    =======

    - None, None if there are no non-zero terms;
    - terms[0] if there is only 1 term;
    - scaled_zero if the sum of the terms produces a zero by cancellation
      e.g. mpfs representing 1 and -1 would produce a scaled zero which need
      special handling since they are not actually zero and they are purposely
      malformed to ensure that they cannot be used in anything but accuracy
      calculations;
    - a tuple that is scaled to target_prec that corresponds to the
      sum of the terms.

    The returned mpf tuple will be normalized to target_prec; the input
    prec is used to define the working precision.

    XXX explain why this is needed and why one cannot just loop using mpf_add
    """

    terms = [t for t in terms if not iszero(t[0])]
    if not terms:
        return None, None
    elif len(terms) == 1:
        return terms[0]

    # see if any argument is NaN or oo and thus warrants a special return
    special = []
    from .numbers import Float
    for t in terms:
        arg = Float._new(t[0], 1)
        if arg is S.NaN or arg.is_infinite:
            special.append(arg)
    if special:
        from .add import Add
        rv = evalf(Add(*special), prec + 4, {})
        return rv[0], rv[2]

    working_prec = 2*prec
    sum_man, sum_exp = 0, 0
    absolute_err: List[int] = []
    # 遍历 terms 列表中的每个元素，其中每个元素是一个元组 (x, accuracy)
    for x, accuracy in terms:
        # 解包元组 x，包含符号位 sign，尾数 man，指数 exp，位计数 bc
        sign, man, exp, bc = x
        # 如果符号位为真（非零），则将尾数 man 取负
        if sign:
            man = -man
        # 计算绝对误差并添加到 absolute_err 列表中
        absolute_err.append(bc + exp - accuracy)
        # 计算 delta 为当前项的指数 exp 减去总和的指数 sum_exp
        delta = exp - sum_exp
        # 如果当前项的指数大于等于总和的指数
        if exp >= sum_exp:
            # x 显著大于现有总和？
            # 首先：进行快速测试
            if ((delta > working_prec) and
                ((not sum_man) or
                 delta - bitcount(abs(sum_man)) > working_prec)):
                # 更新总和的尾数为当前项的尾数 man
                sum_man = man
                # 更新总和的指数为当前项的指数 exp
                sum_exp = exp
            else:
                # 否则，将当前项的尾数 man 左移 delta 位后加到总和的尾数 sum_man 中
                sum_man += (man << delta)
        else:
            # 如果当前项的指数小于总和的指数，取其绝对值作为 delta
            delta = -delta
            # x 显著小于现有总和？
            if delta - bc > working_prec:
                # 如果当前总和的尾数为零，则更新总和的尾数为当前项的尾数 man 和指数 exp
                if not sum_man:
                    sum_man, sum_exp = man, exp
            else:
                # 否则，将当前总和的尾数 sum_man 左移 delta 位后加上当前项的尾数 man
                sum_man = (sum_man << delta) + man
                # 更新总和的指数为当前项的指数 exp
                sum_exp = exp
    # 计算绝对误差的最大值
    absolute_error = max(absolute_err)
    # 如果当前总和的尾数为零，则返回一个经过缩放的零，其误差为 absolute_error
    if not sum_man:
        return scaled_zero(absolute_error)
    # 如果当前总和的尾数为负数，则将总和的符号 sum_sign 设置为 1，并取其绝对值作为总和的尾数
    if sum_man < 0:
        sum_sign = 1
        sum_man = -sum_man
    else:
        # 否则，总和的符号 sum_sign 设置为 0
        sum_sign = 0
    # 计算总和的位计数 sum_bc
    sum_bc = bitcount(sum_man)
    # 计算总和的精度 sum_accuracy
    sum_accuracy = sum_exp + sum_bc - absolute_error
    # 将归一化后的总和 (sum_sign, sum_man, sum_exp, sum_bc, target_prec, rnd) 和总和的精度 sum_accuracy 返回
    r = normalize(sum_sign, sum_man, sum_exp, sum_bc, target_prec,
        rnd), sum_accuracy
    return r
# 计算加法表达式的数值评估结果
def evalf_add(v: 'Add', prec: int, options: OPT_DICT) -> TMP_RES:
    # 尝试将表达式转化为纯复数形式
    res = pure_complex(v)
    if res:
        # 如果表达式是纯复数，拆分为实部和虚部进行评估
        h, c = res
        re, _, re_acc, _ = evalf(h, prec, options)  # 评估实部
        im, _, im_acc, _ = evalf(c, prec, options)  # 评估虚部
        return re, im, re_acc, im_acc

    oldmaxprec = options.get('maxprec', DEFAULT_MAXPREC)

    i = 0
    target_prec = prec
    while 1:
        options['maxprec'] = min(oldmaxprec, 2*prec)

        # 对加法表达式中的每个参数进行递归评估
        terms = [evalf(arg, prec + 10, options) for arg in v.args]
        n = terms.count(S.ComplexInfinity)  # 统计无穷大的数量
        if n >= 2:
            return fnan, None, prec, None  # 如果有两个以上的无穷大，返回 NaN
        # 将参数分组并求和实部和虚部
        re, re_acc = add_terms(
            [a[0::2] for a in terms if isinstance(a, tuple) and a[0]], prec, target_prec)
        im, im_acc = add_terms(
            [a[1::2] for a in terms if isinstance(a, tuple) and a[1]], prec, target_prec)
        if n == 1:
            if re in (finf, fninf, fnan) or im in (finf, fninf, fnan):
                return fnan, None, prec, None  # 如果其中一个部分为无穷大，返回 NaN
            return S.ComplexInfinity  # 如果只有一个无穷大，返回复数无穷大

        # 计算复数的精确度
        acc = complex_accuracy((re, im, re_acc, im_acc))
        if acc >= target_prec:
            if options.get('verbose'):
                print("ADD: wanted", target_prec, "accurate bits, got", re_acc, im_acc)
            break
        else:
            if (prec - target_prec) > options['maxprec']:
                break

            # 调整精度并进行重新计算
            prec = prec + max(10 + 2**i, target_prec - acc)
            i += 1
            if options.get('verbose'):
                print("ADD: restarting with prec", prec)

    options['maxprec'] = oldmaxprec
    # 如果实部或虚部为零，进行零值处理
    if iszero(re, scaled=True):
        re = scaled_zero(re)
    if iszero(im, scaled=True):
        im = scaled_zero(im)
    return re, im, re_acc, im_acc


# 计算乘法表达式的数值评估结果
def evalf_mul(v: 'Mul', prec: int, options: OPT_DICT) -> TMP_RES:
    res = pure_complex(v)
    if res:
        # 如果表达式是纯复数 h*I 形式，只评估虚部
        _, h = res
        im, _, im_acc, _ = evalf(h, prec, options)
        return None, im, None, im_acc

    args = list(v.args)

    # 检查是否有 NaN 或无穷大的参数，需要特殊处理
    has_zero = False
    special = []
    from .numbers import Float
    for arg in args:
        result = evalf(arg, prec, options)
        if result is S.ComplexInfinity:
            special.append(result)
            continue
        if result[0] is None:
            if result[1] is None:
                has_zero = True
            continue
        num = Float._new(result[0], 1)
        if num is S.NaN:
            return fnan, None, prec, None  # 如果有 NaN，返回 NaN
        if num.is_infinite:
            special.append(num)
    if special:
        if has_zero:
            return fnan, None, prec, None  # 如果有零和无穷大，返回 NaN
        from .mul import Mul
        return evalf(Mul(*special), prec + 4, {})  # 对特殊情况进行深入评估
    if has_zero:
        return None, None, None, None  # 如果有零，返回空值

    # 考虑到舍入误差，在实数乘法中并不会损失精度。在复数情况下也是如此。
    # 总精度；但是实部或虚部的单独精度可能较低。
    acc = prec

    # XXX: 很大的过估计
    working_prec = prec + len(args) + 5

    # 空乘积是1
    start = man, exp, bc = MPZ(1), 0, 1

    # 首先，我们将所有纯实数或纯虚数相乘。
    # direction 告诉我们结果应该乘以 I**direction；
    # 所有其他数字放入 complex_factors 中，在第一阶段之后相乘。
    last = len(args)
    direction = 0
    args.append(S.One)
    complex_factors = []

    for i, arg in enumerate(args):
        if i != last and pure_complex(arg):
            args[-1] = (args[-1]*arg).expand()
            continue
        elif i == last and arg is S.One:
            continue
        re, im, re_acc, im_acc = evalf(arg, working_prec, options)
        if re and im:
            complex_factors.append((re, im, re_acc, im_acc))
            continue
        elif re:
            (s, m, e, b), w_acc = re, re_acc
        elif im:
            (s, m, e, b), w_acc = im, im_acc
            direction += 1
        else:
            return None, None, None, None
        direction += 2*s
        man *= m
        exp += e
        bc += b
        while bc > 3*working_prec:
            man >>= working_prec
            exp += working_prec
            bc -= working_prec
        acc = min(acc, w_acc)
    sign = (direction & 2) >> 1
    if not complex_factors:
        v = normalize(sign, man, exp, bitcount(man), prec, rnd)
        # 乘以 i
        if direction & 1:
            return None, v, None, acc
        else:
            return v, None, acc, None
    else:
        # initialize with the first term
        # 如果(man, exp, bc)不等于start，则存在实部；赋予虚部
        if (man, exp, bc) != start:
            re, im = (sign, man, exp, bitcount(man)), (0, MPZ(0), 0, 0)
            i0 = 0
        else:
            # 如果没有真实部分要开始（除了起始的1）
            # 从复数因子列表中取出第一个复数因子
            wre, wim, wre_acc, wim_acc = complex_factors[0]
            # 计算当前精度与复数的精度的最小值
            acc = min(acc,
                      complex_accuracy((wre, wim, wre_acc, wim_acc)))
            re = wre
            im = wim
            i0 = 1

        for wre, wim, wre_acc, wim_acc in complex_factors[i0:]:
            # acc 是乘积的总体精度；我们不计算精确的乘积精度。
            # 计算当前精度与复数的精度的最小值
            acc = min(acc,
                      complex_accuracy((wre, wim, wre_acc, wim_acc)))

            use_prec = working_prec
            # 计算乘积的实部和虚部
            A = mpf_mul(re, wre, use_prec)
            B = mpf_mul(mpf_neg(im), wim, use_prec)
            C = mpf_mul(re, wim, use_prec)
            D = mpf_mul(im, wre, use_prec)
            re = mpf_add(A, B, use_prec)
            im = mpf_add(C, D, use_prec)
        if options.get('verbose'):
            # 如果设置了verbose选项，打印精确位数的信息
            print("MUL: wanted", prec, "accurate bits, got", acc)
        # multiply by I
        # 根据direction的值进行乘以I操作
        if direction & 1:
            re, im = mpf_neg(im), re
        # 返回实部、虚部以及精度acc
        return re, im, acc, acc
# 定义一个函数 evalf_pow，用于计算幂运算的数值结果
def evalf_pow(v: 'Pow', prec: int, options) -> TMP_RES:
    # 设置目标精度为传入的精度值
    target_prec = prec
    # 将输入的 Pow 对象 v 拆分为底数 base 和指数 exp
    base, exp = v.args

    # 处理指数为整数的情况，优化处理速度并且能更好地处理实部/虚部为零的情况
    if exp.is_Integer:
        p: int = exp.p  # type: ignore  # 获取整数指数的值
        # 对于指数为零的情况，直接返回 1 和相关空值
        if not p:
            return fone, None, prec, None
        # 当指数 p 不为零时，基数 base 的求值需要增加一定的精度，特别是当 p 较大时
        prec += int(math.log2(abs(p)))
        # 对基数 base 进行求值，精度增加 5
        result = evalf(base, prec + 5, options)
        # 如果结果为无穷大，则根据指数 p 的正负进行不同的处理
        if result is S.ComplexInfinity:
            if p < 0:
                return None, None, None, None  # 返回空值
            return result  # 返回无穷大

        # 将求得的结果分解为实部和虚部
        re, im, re_acc, im_acc = result
        # 当实部非零且虚部为零时，进行实数的整数幂运算
        if re and not im:
            return mpf_pow_int(re, p, target_prec), None, target_prec, None
        # 当虚部非零且实部为零时，按照复数的幂运算规则处理
        if im and not re:
            z = mpf_pow_int(im, p, target_prec)
            case = p % 4
            if case == 0:
                return z, None, target_prec, None
            if case == 1:
                return None, z, None, target_prec
            if case == 2:
                return mpf_neg(z), None, target_prec, None
            if case == 3:
                return None, mpf_neg(z), None, target_prec
        # 当实部为零时，根据指数 p 的正负进行不同的处理
        if not re:
            if p < 0:
                return S.ComplexInfinity  # 返回复数无穷大
            return None, None, None, None  # 返回空值
        # 对于一般的复数进行任意整数幂运算
        re, im = libmp.mpc_pow_int((re, im), p, prec)
        # 假设输入的数据精确无误，返回最终的复数结果
        return finalize_complex(re, im, target_prec)

    # 处理非整数指数的情况
    result = evalf(base, prec + 5, options)
    # 当结果为无穷大时，如果指数为有理数且为负数，则返回空值，否则抛出未实现的错误
    if result is S.ComplexInfinity:
        if exp.is_Rational:
            if exp < 0:
                return None, None, None, None
            return result
        raise NotImplementedError

    # 处理指数为半的情况，即求平方根
    if exp is S.Half:
        xre, xim, _, _ = result
        # 对于复数进行平方根运算
        if xim:
            re, im = libmp.mpc_sqrt((xre or fzero, xim), prec)
            return finalize_complex(re, im, prec)
        # 当实部为零时，返回空值
        if not xre:
            return None, None, None, None
        # 当实部为负数时，返回其绝对值的平方根
        if mpf_lt(xre, fzero):
            return None, mpf_sqrt(mpf_neg(xre), prec), None, prec
        # 当实部为正数时，返回其平方根
        return mpf_sqrt(xre, prec), None, prec, None

    # 对于其他情况，首先评估指数以确定其大小，以决定必须使用的工作精度
    prec += 10
    result = evalf(exp, prec, options)
    # 当结果为无穷大时，返回 NaN 和相关空值
    if result is S.ComplexInfinity:
        return fnan, None, prec, None
    yre, yim, _, _ = result
    # 特殊情况处理：指数为零时，返回 1 和相关空值
    if not (yre or yim):
        return fone, None, prec, None

    # 计算 yre 的对数值
    ysize = fastlog(yre)
    # 如果 ysize 太大，重新开始计算
    # 注意：prec + ysize 可能会超过 maxprec
    if ysize > 5:
        # 将 prec 增加 ysize
        prec += ysize
        # 对指数表达式进行评估，获取实部和虚部
        yre, yim, _, _ = evalf(exp, prec, options)

    # 纯指数函数；不需要评估基数
    if base is S.Exp1:
        if yim:
            # 计算复数指数函数 e^(yre + i*yim)，返回实部和虚部
            re, im = libmp.mpc_exp((yre or fzero, yim), prec)
            return finalize_complex(re, im, target_prec)
        # 计算实数指数函数 e^yre
        return mpf_exp(yre, target_prec), None, target_prec, None

    # 对基数进行评估，获取其实部和虚部
    xre, xim, _, _ = evalf(base, prec + 5, options)

    # 处理 0 的任意次幂
    if not (xre or xim):
        if yim:
            # 若指数为复数，返回 NaN
            return fnan, None, prec, None
        if yre[0] == 1:  # y < 0
            # 若指数为负数，返回正无穷
            return S.ComplexInfinity
        # 若指数为零或正数，返回 None
        return None, None, None, None

    # 计算 (real ** complex) 或 (complex ** complex)
    if yim:
        # 计算复数底数的复数次幂
        re, im = libmp.mpc_pow(
            (xre or fzero, xim or fzero), (yre or fzero, yim),
            target_prec)
        return finalize_complex(re, im, target_prec)
    # 计算复数底数的实数次幂
    if xim:
        re, im = libmp.mpc_pow_mpf((xre or fzero, xim), yre, target_prec)
        return finalize_complex(re, im, target_prec)
    # 计算负实数底数的实数次幂
    elif mpf_lt(xre, fzero):
        re, im = libmp.mpc_pow_mpf((xre, fzero), yre, target_prec)
        return finalize_complex(re, im, target_prec)
    # 计算正实数底数的实数次幂
    else:
        return mpf_pow(xre, yre, target_prec), None, target_prec, None
#----------------------------------------------------------------------------#
#                                                                            #
#                            Special functions                               #
#                                                                            #
#----------------------------------------------------------------------------#

# 对指数函数进行评估，返回临时结果
def evalf_exp(expr: 'exp', prec: int, options: OPT_DICT) -> TMP_RES:
    # 导入指数函数的Pow类
    from .power import Pow
    # 调用evalf_pow函数处理Pow对象的计算，禁用求值
    return evalf_pow(Pow(S.Exp1, expr.exp, evaluate=False), prec, options)


# 对三角函数进行评估，处理复数参数的sin和cos函数
def evalf_trig(v: 'Expr', prec: int, options: OPT_DICT) -> TMP_RES:
    """
    This function handles sin and cos of complex arguments.

    TODO: should also handle tan of complex arguments.
    """
    # 导入正弦和余弦函数
    from sympy.functions.elementary.trigonometric import cos, sin
    # 根据表达式类型选择相应的数学函数
    if isinstance(v, cos):
        func = mpf_cos
    elif isinstance(v, sin):
        func = mpf_sin
    else:
        raise NotImplementedError
    # 获取函数的参数
    arg = v.args[0]
    # 增加20位额外的精度，防止需要重新开始的情况
    xprec = prec + 20
    # 对参数进行evalf评估，获取实部和虚部的结果
    re, im, re_acc, im_acc = evalf(arg, xprec, options)
    # 如果有虚部
    if im:
        # 如果选项中包含替换，则对参数进行替换
        if 'subs' in options:
            v = v.subs(options['subs'])
        # 对表达式的评估结果进行evalf重新评估，返回实部和精度
        return evalf(v._eval_evalf(prec), prec, options)
    # 如果实部为零
    if not re:
        # 如果是余弦函数，返回1.0，否则返回空值
        if isinstance(v, cos):
            return fone, None, prec, None
        elif isinstance(v, sin):
            return None, None, None, None
        else:
            raise NotImplementedError
    # 对于三角函数，我们关注参数的固定点（绝对）精度
    xsize = fastlog(re)
    # 参数大小小于1.0，可以直接计算
    if xsize < 1:
        return func(re, prec, rnd), None, prec, None
    # 参数非常大
    if xsize >= 10:
        # 增加参数大小后的精度
        xprec = prec + xsize
        # 再次评估参数的实部和虚部
        re, im, re_acc, im_acc = evalf(arg, xprec, options)
    # 需要重复评估，因为参数非常接近pi（或pi/2）的整数倍，接近根的情况
    while 1:
        # 计算三角函数的值
        y = func(re, prec, rnd)
        # 计算结果的大小
        ysize = fastlog(y)
        # 计算与所需精度的差距
        gap = -ysize
        # 计算精度
        accuracy = (xprec - xsize) - gap
        # 如果精度小于所需精度
        if accuracy < prec:
            # 如果选项中包含详细信息，则打印信息
            if options.get('verbose'):
                print("SIN/COS", accuracy, "wanted", prec, "gap", gap)
                print(to_str(y, 10))
            # 如果超过最大精度选项，则返回y，否则增加精度
            if xprec > options.get('maxprec', DEFAULT_MAXPREC):
                return y, None, accuracy, None
            xprec += gap
            # 再次评估参数的实部和虚部
            re, im, re_acc, im_acc = evalf(arg, xprec, options)
            continue
        else:
            return y, None, prec, None


# 对对数函数进行评估
def evalf_log(expr: 'log', prec: int, options: OPT_DICT) -> TMP_RES:
    # 如果表达式参数大于1个，则调用doit方法处理表达式，然后重新评估
    if len(expr.args)>1:
        expr = expr.doit()
        return evalf(expr, prec, options)
    # 获取对数函数的参数
    arg = expr.args[0]
    # 增加10位额外精度
    workprec = prec + 10
    # 对参数进行评估，获取结果
    result = evalf(arg, workprec, options)
    # 如果结果是复无穷大，则直接返回该结果
    if result is S.ComplexInfinity:
        return result
    # 解构结果元组，获取实部、虚部、精度、未使用的部分
    xre, xim, xacc, _ = result

    # evalf 函数在 chop=True 时可能返回 NoneType
    # issue 18516, 19623
    # 如果 xre 和 xim 均为 None，则进行特殊处理
    if xre is xim is None:
        # Dear reviewer, I do not know what -inf is;
        # it looks to be (1, 0, -789, -3)
        # but I'm not sure in general,
        # so we just let mpmath figure
        # it out by taking log of 0 directly.
        # It would be better to return -inf instead.
        xre = fzero

    # 如果存在虚部 xim，则进行如下计算
    if xim:
        from sympy.functions.elementary.complexes import Abs
        from sympy.functions.elementary.exponential import log

        # XXX: use get_abs etc instead
        # 计算参数的绝对值的对数
        re = evalf_log(
            log(Abs(arg, evaluate=False), evaluate=False), prec, options)
        # 计算参数的反正切值
        im = mpf_atan2(xim, xre or fzero, prec)
        return re[0], im, re[2], prec

    # 计算是否存在虚数项
    imaginary_term = (mpf_cmp(xre, fzero) < 0)

    # 计算实部的对数
    re = mpf_log(mpf_abs(xre), prec, rnd)
    # 计算对数的快速近似值
    size = fastlog(re)
    # 如果精度减去快速计算的大小大于工作精度，并且 re 不为零，则进行如下处理
    if prec - size > workprec and re != fzero:
        from .add import Add
        # 实际上我们需要精确计算 1+x，而不是 x
        add = Add(S.NegativeOne, arg, evaluate=False)
        # 计算精确加法的结果
        xre, xim, _, _ = evalf_add(add, prec, options)
        # 计算第二个精度
        prec2 = workprec - fastlog(xre)
        # xre 现在是 x - 1，因此在这里添加 1 返回计算 x
        re = mpf_log(mpf_abs(mpf_add(xre, fone, prec2)), prec, rnd)

    # 实部精确度
    re_acc = prec

    # 如果存在虚数项，则返回实部、π、实部精确度、精度
    if imaginary_term:
        return re, mpf_pi(prec), re_acc, prec
    # 否则返回实部、None、实部精确度、None
    else:
        return re, None, re_acc, None
# 对给定的atan函数参数求值，获取其实际数值和误差估计
def evalf_atan(v: 'atan', prec: int, options: OPT_DICT) -> TMP_RES:
    # 从参数列表中获取第一个参数
    arg = v.args[0]
    # 调用evalf函数对参数进行精确求值，返回实部、虚部以及对应的精确度和选项
    xre, xim, reacc, imacc = evalf(arg, prec + 5, options)
    # 如果实部和虚部均为空，则返回四个None
    if xre is xim is None:
        return (None,)*4
    # 如果虚部不为空，目前不支持该情况，抛出NotImplementedError
    if xim:
        raise NotImplementedError
    # 对实部应用mpf_atan函数进行反正切计算，返回计算结果、None、指定的精度和None
    return mpf_atan(xre, prec, rnd), None, prec, None


# 对替换字典中的所有Float条目进行精度更改
def evalf_subs(prec: int, subs: dict) -> dict:
    """ Change all Float entries in `subs` to have precision prec. """
    newsubs = {}
    for a, b in subs.items():
        # 将每个值转换为SymPy对象
        b = S(b)
        # 如果值是Float类型，则调用_eval_evalf函数以指定精度进行求值
        if b.is_Float:
            b = b._eval_evalf(prec)
        newsubs[a] = b
    return newsubs


# 对Piecewise表达式进行求值
def evalf_piecewise(expr: 'Expr', prec: int, options: OPT_DICT) -> TMP_RES:
    from .numbers import Float, Integer
    # 如果选项中包含'subs'键，则使用evalf_subs函数替换表达式中的符号
    if 'subs' in options:
        expr = expr.subs(evalf_subs(prec, options['subs']))
        # 复制选项字典，并移除'subs'键
        newopts = options.copy()
        del newopts['subs']
        # 如果表达式具有'func'属性，则调用evalf函数对其进行精确求值
        if hasattr(expr, 'func'):
            return evalf(expr, prec, newopts)
        # 如果表达式是float类型，则转换为Float对象再调用evalf函数
        if isinstance(expr, float):
            return evalf(Float(expr), prec, newopts)
        # 如果表达式是int类型，则转换为Integer对象再调用evalf函数
        if isinstance(expr, int):
            return evalf(Integer(expr), prec, newopts)

    # 如果仍然有未定义的符号，则抛出NotImplementedError
    raise NotImplementedError


# 对代数数进行精确数值求解
def evalf_alg_num(a: 'AlgebraicNumber', prec: int, options: OPT_DICT) -> TMP_RES:
    # 调用a.to_root()方法获取代数数的根并进行精确求解
    return evalf(a.to_root(), prec, options)


#----------------------------------------------------------------------------#
#                                                                            #
#                            High-level operations                           #
#                                                                            #
#----------------------------------------------------------------------------#


# 将输入转换为mpmath中的复数或实数对象
def as_mpmath(x: Any, prec: int, options: OPT_DICT) -> tUnion[mpc, mpf]:
    from .numbers import Infinity, NegativeInfinity, Zero
    # 使用sympify函数将输入转换为SymPy对象
    x = sympify(x)
    # 如果输入是Zero类型或等于0.0，则返回mpf对象表示0
    if isinstance(x, Zero) or x == 0.0:
        return mpf(0)
    # 如果输入是Infinity类型，则返回mpf对象表示正无穷
    if isinstance(x, Infinity):
        return mpf('inf')
    # 如果输入是NegativeInfinity类型，则返回mpf对象表示负无穷
    if isinstance(x, NegativeInfinity):
        return mpf('-inf')
    # 否则调用evalf函数对输入进行精确求解，并将结果转换为mpmath对象
    result = evalf(x, prec, options)
    return quad_to_mpmath(result)


# 对积分表达式进行精确数值积分
def do_integral(expr: 'Integral', prec: int, options: OPT_DICT) -> TMP_RES:
    # 获取积分表达式的函数、积分变量和积分下限、上限
    func = expr.args[0]
    x, xlow, xhigh = expr.args[1]
    # 如果积分上限等于积分下限，则将它们都设为0
    if xlow == xhigh:
        xlow = xhigh = 0
    # 如果积分变量不在函数的自由符号中，则只计算上下限的差异
    # 如果存在公共的符号会在取差分时相消，则使用这种差异
    elif x not in func.free_symbols:
        if xhigh.free_symbols & xlow.free_symbols:
            diff = xhigh - xlow
            if diff.is_number:
                xlow, xhigh = 0, diff

    # 获取当前的最大精度限制，将其设置为指定精度的两倍
    oldmaxprec = options.get('maxprec', DEFAULT_MAXPREC)
    options['maxprec'] = min(oldmaxprec, 2*prec)
    with workprec(prec + 5):
        # 设置精度为 prec + 5
        xlow = as_mpmath(xlow, prec + 15, options)
        # 将 xlow 转换为 mpmath 格式，精度为 prec + 15
        xhigh = as_mpmath(xhigh, prec + 15, options)
        # 将 xhigh 转换为 mpmath 格式，精度为 prec + 15

        # Integration is like summation, and we can phone home from
        # the integrand function to update accuracy summation style
        # Note that this accuracy is inaccurate, since it fails
        # to account for the variable quadrature weights,
        # but it is better than nothing
        # 积分类似于求和，我们可以从被积函数中返回结果来更新精度的求和方式。
        # 需要注意的是，此处的精度不准确，因为它未考虑变量的积分权重，但总比什么都不做好。

        from sympy.functions.elementary.trigonometric import cos, sin
        from .symbol import Wild
        # 导入符号计算库中的三角函数 cos 和 sin，以及当前文件夹下的 symbol 模块中的 Wild 符号

        have_part = [False, False]
        max_real_term: tUnion[float, int] = MINUS_INF
        max_imag_term: tUnion[float, int] = MINUS_INF

        def f(t: 'Expr') -> tUnion[mpc, mpf]:
            nonlocal max_real_term, max_imag_term
            # 定义内部函数 f，参数 t 是表达式类型，返回值可以是 mpc 或 mpf 类型
            re, im, re_acc, im_acc = evalf(func, mp.prec, {'subs': {x: t}})
            # 调用 evalf 函数计算 func 在精度 mp.prec 下的值，其中 x 替换为 t

            have_part[0] = re or have_part[0]
            have_part[1] = im or have_part[1]
            # 更新 have_part 列表，记录 re 和 im 是否非零

            max_real_term = max(max_real_term, fastlog(re))
            max_imag_term = max(max_imag_term, fastlog(im))
            # 更新 max_real_term 和 max_imag_term，记录 re 和 im 的对数值（如果为正数）

            if im:
                return mpc(re or fzero, im)
            return mpf(re or fzero)
            # 如果存在虚部 im，则返回 mpc(re, im)，否则返回 mpf(re)

        if options.get('quad') == 'osc':
            # 如果选项中有 'quad' 键且其值为 'osc'
            A = Wild('A', exclude=[x])
            B = Wild('B', exclude=[x])
            D = Wild('D')
            m = func.match(cos(A*x + B)*D)
            # 尝试匹配 func 是否符合 cos(A*x + B)*D 的形式
            if not m:
                m = func.match(sin(A*x + B)*D)
                # 如果不符合，则尝试匹配 sin(A*x + B)*D 的形式
            if not m:
                raise ValueError("An integrand of the form sin(A*x+B)*f(x) "
                  "or cos(A*x+B)*f(x) is required for oscillatory quadrature")
                # 如果都不匹配，则抛出 ValueError 异常

            period = as_mpmath(2*S.Pi/m[A], prec + 15, options)
            # 计算周期 period，使用 m[A] 的值计算，精度为 prec + 15
            result = quadosc(f, [xlow, xhigh], period=period)
            # 使用 quadosc 函数进行振荡积分计算，积分区间为 [xlow, xhigh]

            # XXX: quadosc does not do error detection yet
            # quadosc 目前还没有误差检测功能
            quadrature_error = MINUS_INF
            # 设置 quadrature_error 初始值为负无穷大
        else:
            result, quadrature_err = quadts(f, [xlow, xhigh], error=1)
            # 使用 quadts 函数进行通常的积分计算，积分区间为 [xlow, xhigh]，设置误差参数为 1
            quadrature_error = fastlog(quadrature_err._mpf_)
            # 计算积分误差的对数值

    options['maxprec'] = oldmaxprec
    # 恢复选项中的 maxprec 值为旧值 oldmaxprec

    if have_part[0]:
        re: Optional[MPF_TUP] = result.real._mpf_
        re_acc: Optional[int]
        if re == fzero:
            re_s, re_acc = scaled_zero(int(-max(prec, max_real_term, quadrature_error)))
            # 调整精度以适应实部 re
            re = scaled_zero(re_s)  # 在 evalf_integral 中处理得当
        else:
            re_acc = int(-max(max_real_term - fastlog(re) - prec, quadrature_error))
            # 计算实部 re 的精度

    else:
        re, re_acc = None, None

    if have_part[1]:
        im: Optional[MPF_TUP] = result.imag._mpf_
        im_acc: Optional[int]
        if im == fzero:
            im_s, im_acc = scaled_zero(int(-max(prec, max_imag_term, quadrature_error)))
            # 调整精度以适应虚部 im
            im = scaled_zero(im_s)  # 在 evalf_integral 中处理得当
        else:
            im_acc = int(-max(max_imag_term - fastlog(im) - prec, quadrature_error))
            # 计算虚部 im 的精度

    else:
        im, im_acc = None, None

    result = re, im, re_acc, im_acc
    # 更新结果值

    return result
    # 返回最终的结果
# 定义函数，用于计算积分表达式的数值近似
def evalf_integral(expr: 'Integral', prec: int, options: OPT_DICT) -> TMP_RES:
    # 获取积分的限制条件
    limits = expr.limits
    # 如果限制条件不符合预期（不是单一积分或者格式不正确），则抛出未实现的错误
    if len(limits) != 1 or len(limits[0]) != 3:
        raise NotImplementedError
    # 设置工作精度为指定精度
    workprec = prec
    i = 0
    # 获取最大精度，如果未指定则为无穷大
    maxprec = options.get('maxprec', INF)
    # 进入循环，尝试达到期望的精度
    while 1:
        # 执行积分计算，获取结果
        result = do_integral(expr, workprec, options)
        # 计算结果的复杂精度
        accuracy = complex_accuracy(result)
        # 如果达到了预期的精度，则退出循环
        if accuracy >= prec:  # achieved desired precision
            break
        # 如果无法再增加精度，则退出循环
        if workprec >= maxprec:  # can't increase accuracy any more
            break
        # 如果精度为-1，则可能答案确实为零，或者我们还未增加足够的精度，因此加倍工作精度，以避免过长时间达到最大精度
        if accuracy == -1:
            workprec *= 2
        else:
            # 否则，增加工作精度，增加量为预设精度或2的幂
            workprec += max(prec, 2**i)
        # 确保工作精度不超过最大精度
        workprec = min(workprec, maxprec)
        i += 1
    # 返回积分计算结果
    return result


# 定义函数，用于检查级数的收敛性
def check_convergence(numer: 'Expr', denom: 'Expr', n: 'Symbol') -> tTuple[int, Any, Any]:
    """
    返回元组 (h, g, p) 其中：
    -- h 是：
        > 0 表示以 1/factorial(n)**h 收敛
        < 0 表示以 factorial(n)**(-h) 发散
        = 0 表示几何或多项式的收敛或发散

    -- abs(g) 是：
        > 1 表示以 1/h**n 的几何收敛
        < 1 表示以 h**n 的几何发散
        = 1 表示多项式的收敛或发散

        (g < 0 表示交替级数)

    -- p 是：
        > 1 表示以 1/n**h 的多项式收敛
        <= 1 表示以 n**(-h) 的多项式发散
    """
    # 导入多项式工具类 Poly
    from sympy.polys.polytools import Poly
    # 创建分子和分母的多项式对象
    npol = Poly(numer, n)
    dpol = Poly(denom, n)
    # 计算分子和分母的次数
    p = npol.degree()
    q = dpol.degree()
    # 计算级数的收敛速率
    rate = q - p
    # 如果速率不为零，则返回速率及空值
    if rate:
        return rate, None, None
    # 计算分母的常数项与分子的常数项比值
    constant = dpol.LC() / npol.LC()
    # 导入数值比较工具函数 equal_valued
    from .numbers import equal_valued
    # 如果常数项绝对值不等于1，则返回速率和常数项及空值
    if not equal_valued(abs(constant), 1):
        return rate, constant, None
    # 如果分子和分母的次数均为0，则返回速率、常数项和0
    if npol.degree() == dpol.degree() == 0:
        return rate, constant, 0
    # 获取分子和分母的系数，计算其差值除以分母的常数项
    pc = npol.all_coeffs()[1]
    qc = dpol.all_coeffs()[1]
    return rate, constant, (qc - pc)/dpol.LC()


# 定义函数，用于求解快速收敛的无穷超几何级数
def hypsum(expr: 'Expr', n: 'Symbol', start: int, prec: int) -> mpf:
    """
    对给定的无穷超几何级数求和，其通项由表达式 expr 给出，例如 e = hypsum(1/factorial(n), n)。
    连续项之比必须是整数多项式的商。
    """
    # 导入浮点数工具 Float 和数值比较工具 equal_valued
    from .numbers import Float, equal_valued
    # 导入超简化工具 hypersimp
    from sympy.simplify.simplify import hypersimp

    # 如果精度为无穷大，则抛出未实现的错误
    if prec == float('inf'):
        raise NotImplementedError('does not support inf prec')

    # 如果起始点不为零，则将表达式中的 n 替换为 n + start
    if start:
        expr = expr.subs(n, n + start)
    # 对表达式进行超几何简化
    hs = hypersimp(expr, n)
    # 如果无法进行超几何简化，则抛出未实现的错误
    if hs is None:
        raise NotImplementedError("a hypergeometric series is required")
    # 将超几何简化后的表达式化为分子和分母
    num, den = hs.as_numer_denom()

    # 使用 lambdify 将分子和分母转换为关于 n 的函数
    func1 = lambdify(n, num)
    func2 = lambdify(n, den)

    # 调用 check_convergence 函数，获取级数的收敛性指标 h, g, p
    h, g, p = check_convergence(num, den, n)
    # 如果指数 h 小于 0，则抛出值错误异常，表示和发散，类似于 (n!)^(-h)
    if h < 0:
        raise ValueError("Sum diverges like (n!)^%i" % (-h))

    # 计算表达式在 n=0 处的项
    term = expr.subs(n, 0)
    # 如果 term 不是有理数，则抛出未实现错误，表示不支持非有理数项的功能
    if not term.is_Rational:
        raise NotImplementedError("Non rational term functionality is not implemented.")

    # 如果指数 h 大于 0，或者 h 等于 0 且底数 g 的绝对值大于 1，则直接求和
    if h > 0 or (h == 0 and abs(g) > 1):
        # 将 term 转换为多精度整数 MPZ，并左移 prec 位，以提高精度
        term = (MPZ(term.p) << prec) // term.q
        s = term
        k = 1
        # 循环直到 term 的绝对值小于 5
        while abs(term) > 5:
            # 计算新的 term
            term *= MPZ(func1(k - 1))
            term //= MPZ(func2(k - 1))
            s += term
            k += 1
        # 将 s 转换回标准浮点数表示，精度为 -prec
        return from_man_exp(s, -prec)
    else:
        # 判断是否为交替序列（alt），即 g 是否小于 0
        alt = g < 0
        # 如果底数 g 的绝对值小于 1，则抛出值错误异常，表示和发散，类似于 (|g|)^n
        if abs(g) < 1:
            raise ValueError("Sum diverges like (%i)^n" % abs(1/g))
        # 如果指数 p 小于 1，或者 p 等于 1 且不是交替序列，则抛出值错误异常，表示和发散，类似于 n^(-p)
        if p < 1 or (equal_valued(p, 1) and not alt):
            raise ValueError("Sum diverges like n^%i" % (-p))
        
        # 现在已知是多项式收敛：使用 Richardson 外推方法
        vold = None
        ndig = prec_to_dps(prec)
        while True:
            # 需要至少使用四倍精度，因为外推过程中会有大量抵消；还需要检查答案确保达到所需精度。
            prec2 = 4 * prec
            term0 = (MPZ(term.p) << prec2) // term.q

            def summand(k, _term=[term0]):
                if k:
                    k = int(k)
                    _term[0] *= MPZ(func1(k - 1))
                    _term[0] //= MPZ(func2(k - 1))
                return make_mpf(from_man_exp(_term[0], -prec2))

            # 在给定精度下进行 Richardson 外推求和
            with workprec(prec):
                v = nsum(summand, [0, mpmath_inf], method='richardson')
            vf = Float(v, ndig)
            # 如果 vold 不为 None 且与 vf 相等，则结束循环
            if vold is not None and vold == vf:
                break
            # 每次将精度加倍
            prec += prec

            vold = vf

        # 返回结果 v 的多精度浮点数表示
        return v._mpf_
def evalf_prod(expr: 'Product', prec: int, options: OPT_DICT) -> TMP_RES:
    # 检查所有限制条件是否是整数，如果是，则直接求值
    if all((l[1] - l[2]).is_Integer for l in expr.limits):
        result = evalf(expr.doit(), prec=prec, options=options)
    else:
        # 导入求和相关模块
        from sympy.concrete.summations import Sum
        # 将乘积表达式重写为对应的求和表达式，并求值
        result = evalf(expr.rewrite(Sum), prec=prec, options=options)
    return result


def evalf_sum(expr: 'Sum', prec: int, options: OPT_DICT) -> TMP_RES:
    # 导入浮点数相关模块
    from .numbers import Float
    # 如果有替换选项，则进行表达式替换
    if 'subs' in options:
        expr = expr.subs(options['subs'])
    # 获取表达式的函数部分和限制条件
    func = expr.function
    limits = expr.limits
    # 如果限制条件的数量不为1，或者每个限制条件不为3，则抛出未实现的错误
    if len(limits) != 1 or len(limits[0]) != 3:
        raise NotImplementedError
    # 如果函数为零函数，则返回空值
    if func.is_zero:
        return None, None, prec, None
    # 增加精度10，尝试进行求和计算
    prec2 = prec + 10
    try:
        n, a, b = limits[0]
        # 如果上限不是无穷大，下限是负无穷大，或者下限不是整数，则抛出未实现的错误
        if b is not S.Infinity or a is S.NegativeInfinity or a != int(a):
            raise NotImplementedError
        # 如果可以，使用快速超几何求和方法
        v = hypsum(func, n, int(a), prec2)
        delta = prec - fastlog(v)
        # 如果快速对数小于-10，使用精度修正进行超几何求和
        if fastlog(v) < -10:
            v = hypsum(func, n, int(a), delta)
        return v, None, min(prec, delta), None
    except NotImplementedError:
        # 对于一般的级数，使用欧拉-麦克劳林求和方法
        eps = Float(2.0)**(-prec)
        for i in range(1, 5):
            m = n = 2**i * prec
            s, err = expr.euler_maclaurin(m=m, n=n, eps=eps,
                eval_integral=False)
            err = err.evalf()
            if err is S.NaN:
                raise NotImplementedError
            if err <= eps:
                break
        # 对错误进行快速对数求值，并返回实部和虚部精度
        err = fastlog(evalf(abs(err), 20, options)[0])
        re, im, re_acc, im_acc = evalf(s, prec2, options)
        if re_acc is None:
            re_acc = -err
        if im_acc is None:
            im_acc = -err
        return re, im, re_acc, im_acc


#----------------------------------------------------------------------------#
#                                                                            #
#                            符号接口                                        #
#                                                                            #
#----------------------------------------------------------------------------#

def evalf_symbol(x: 'Expr', prec: int, options: OPT_DICT) -> TMP_RES:
    # 从选项中获取符号替换的值
    val = options['subs'][x]
    # 如果值是浮点数，则返回其值，否则进行缓存操作
    if isinstance(val, mpf):
        if not val:
            return None, None, None, None
        return val._mpf_, None, prec, None
    else:
        # 如果缓存中不存在符号的值，则进行计算并缓存
        if '_cache' not in options:
            options['_cache'] = {}
        cache = options['_cache']
        cached, cached_prec = cache.get(x, (None, MINUS_INF))
        if cached_prec >= prec:
            return cached
        v = evalf(sympify(val), prec, options)
        cache[x] = (v, prec)
        return v


evalf_table: tDict[Type['Expr'], Callable[['Expr', int, OPT_DICT], TMP_RES]] = {}


def _create_evalf_table():
    # 全局创建一个空的评估函数表
    global evalf_table
    # 导入乘积相关模块
    from sympy.concrete.products import Product
    # 导入 SymPy 库中的具体模块或函数
    from sympy.concrete.summations import Sum                  # 导入 Sum 类
    from .add import Add                                       # 从当前包中导入 Add 类
    from .mul import Mul                                       # 从当前包中导入 Mul 类
    from .numbers import Exp1, Float, Half, ImaginaryUnit, Integer, NaN, NegativeOne, One, Pi, Rational, \
        Zero, ComplexInfinity, AlgebraicNumber                  # 从当前包中导入多个数学常量和类型
    from .power import Pow                                     # 从当前包中导入 Pow 类
    from .symbol import Dummy, Symbol                           # 从当前包中导入 Dummy 和 Symbol 类
    from sympy.functions.elementary.complexes import Abs, im, re  # 导入复数运算相关函数
    from sympy.functions.elementary.exponential import exp, log  # 导入指数和对数函数
    from sympy.functions.elementary.integers import ceiling, floor  # 导入取上下整函数
    from sympy.functions.elementary.piecewise import Piecewise  # 导入分段函数
    from sympy.functions.elementary.trigonometric import atan, cos, sin  # 导入三角函数
    from sympy.integrals.integrals import Integral             # 导入积分函数
    
    # 定义一个评估函数的映射表，将不同类型或函数映射到对应的评估函数
    evalf_table = {
        Symbol: evalf_symbol,                                  # 对于 Symbol 类型，使用 evalf_symbol 函数评估
        Dummy: evalf_symbol,                                   # 对于 Dummy 类型，也使用 evalf_symbol 函数评估
        Float: evalf_float,                                    # 对于 Float 类型，使用 evalf_float 函数评估
        Rational: evalf_rational,                              # 对于 Rational 类型，使用 evalf_rational 函数评估
        Integer: evalf_integer,                                # 对于 Integer 类型，使用 evalf_integer 函数评估
        Zero: lambda x, prec, options: (None, None, prec, None),  # 对于 Zero 类型，返回一个特定的元组
        One: lambda x, prec, options: (fone, None, prec, None),  # 对于 One 类型，返回一个特定的元组
        Half: lambda x, prec, options: (fhalf, None, prec, None),  # 对于 Half 类型，返回一个特定的元组
        Pi: lambda x, prec, options: (mpf_pi(prec), None, prec, None),  # 对于 Pi 类型，返回一个特定的元组
        Exp1: lambda x, prec, options: (mpf_e(prec), None, prec, None),  # 对于 Exp1 类型，返回一个特定的元组
        ImaginaryUnit: lambda x, prec, options: (None, fone, None, prec),  # 对于 ImaginaryUnit 类型，返回一个特定的元组
        NegativeOne: lambda x, prec, options: (fnone, None, prec, None),  # 对于 NegativeOne 类型，返回一个特定的元组
        ComplexInfinity: lambda x, prec, options: S.ComplexInfinity,  # 对于 ComplexInfinity 类型，返回特定的值
        NaN: lambda x, prec, options: (fnan, None, prec, None),  # 对于 NaN 类型，返回一个特定的元组
    
        exp: evalf_exp,                                         # 对于 exp 函数，使用 evalf_exp 函数评估
    
        cos: evalf_trig,                                        # 对于 cos 函数，使用 evalf_trig 函数评估
        sin: evalf_trig,                                        # 对于 sin 函数，使用 evalf_trig 函数评估
    
        Add: evalf_add,                                         # 对于 Add 类，使用 evalf_add 函数评估
        Mul: evalf_mul,                                         # 对于 Mul 类，使用 evalf_mul 函数评估
        Pow: evalf_pow,                                         # 对于 Pow 类，使用 evalf_pow 函数评估
    
        log: evalf_log,                                         # 对于 log 函数，使用 evalf_log 函数评估
        atan: evalf_atan,                                       # 对于 atan 函数，使用 evalf_atan 函数评估
        Abs: evalf_abs,                                         # 对于 Abs 函数，使用 evalf_abs 函数评估
    
        re: evalf_re,                                           # 对于 re 函数，使用 evalf_re 函数评估
        im: evalf_im,                                           # 对于 im 函数，使用 evalf_im 函数评估
        floor: evalf_floor,                                     # 对于 floor 函数，使用 evalf_floor 函数评估
        ceiling: evalf_ceiling,                                 # 对于 ceiling 函数，使用 evalf_ceiling 函数评估
    
        Integral: evalf_integral,                               # 对于 Integral 类，使用 evalf_integral 函数评估
        Sum: evalf_sum,                                         # 对于 Sum 类，使用 evalf_sum 函数评估
        Product: evalf_prod,                                    # 对于 Product 类，使用 evalf_prod 函数评估
        Piecewise: evalf_piecewise,                             # 对于 Piecewise 类，使用 evalf_piecewise 函数评估
    
        AlgebraicNumber: evalf_alg_num,                         # 对于 AlgebraicNumber 类，使用 evalf_alg_num 函数评估
    }
def evalf(x: 'Expr', prec: int, options: OPT_DICT) -> TMP_RES:
    """
    Evaluate the ``Expr`` instance, ``x``
    to a binary precision of ``prec``. This
    function is supposed to be used internally.

    Parameters
    ==========

    x : Expr
        The formula to evaluate to a float.
    prec : int
        The binary precision that the output should have.
    options : dict
        A dictionary with the same entries as
        ``EvalfMixin.evalf`` and in addition,
        ``maxprec`` which is the maximum working precision.

    Returns
    =======

    An optional tuple, ``(re, im, re_acc, im_acc)``
    which are the real, imaginary, real accuracy
    and imaginary accuracy respectively. ``re`` is
    an mpf value tuple and so is ``im``. ``re_acc``
    and ``im_acc`` are ints.

    NB: all these return values can be ``None``.
    If all values are ``None``, then that represents 0.
    Note that 0 is also represented as ``fzero = (0, 0, 0, 0)``.
    """
    from sympy.functions.elementary.complexes import re as re_, im as im_
    
    try:
        # 从预定义的评估函数表中获取与 x 类型匹配的评估函数 rf
        rf = evalf_table[type(x)]
        # 使用 rf 函数对 x 进行评估，得到结果 r
        r = rf(x, prec, options)
    except KeyError:
        # 如果在评估函数表中找不到 x 的类型，则使用普通的 evalf 方法
        if 'subs' in options:
            # 如果 options 中包含 'subs' 键，则使用 evalf_subs 转换 x
            x = x.subs(evalf_subs(prec, options['subs']))
        # 尝试调用 x 对象的 _eval_evalf 方法，获取其评估结果 xe
        xe = x._eval_evalf(prec)
        if xe is None:
            # 如果 xe 为 None，则抛出 NotImplementedError
            raise NotImplementedError
        # 检查 xe 是否具有 as_real_imag 方法
        as_real_imag = getattr(xe, "as_real_imag", None)
        if as_real_imag is None:
            # 如果 xe 没有 as_real_imag 方法，则抛出 NotImplementedError
            raise NotImplementedError  # e.g. FiniteSet(-1.0, 1.0).evalf()
        # 获取 xe 的实部和虚部
        re, im = as_real_imag()
        # 检查实部或虚部是否包含 sympy 库中的 re_ 或 im_ 符号
        if re.has(re_) or im.has(im_):
            # 如果实部或虚部包含 re_ 或 im_ 符号，则抛出 NotImplementedError
            raise NotImplementedError
        # 如果实部为 0.0，则将其设置为 None
        if re == 0.0:
            re = None
            reprec = None
        elif re.is_number:
            # 如果实部为数值，则将其转换为指定精度的 mpf 值
            re = re._to_mpmath(prec, allow_ints=False)._mpf_
            reprec = prec
        else:
            # 否则抛出 NotImplementedError
            raise NotImplementedError
        # 如果虚部为 0.0，则将其设置为 None
        if im == 0.0:
            im = None
            imprec = None
        elif im.is_number:
            # 如果虚部为数值，则将其转换为指定精度的 mpf 值
            im = im._to_mpmath(prec, allow_ints=False)._mpf_
            imprec = prec
        else:
            # 否则抛出 NotImplementedError
            raise NotImplementedError
        # 组装结果元组 r
        r = re, im, reprec, imprec
    
    if options.get("verbose"):
        # 如果 options 中包含 'verbose' 键，则打印输入和输出信息
        print("### input", x)
        print("### output", to_str(r[0] or fzero, 50) if isinstance(r, tuple) else r)
        print("### raw", r)  # 打印原始评估结果 r
        print()
    
    chop = options.get('chop', False)
    if chop:
        # 如果 options 中包含 'chop' 键，则进行舍入处理
        if chop is True:
            chop_prec = prec
        else:
            # 根据给定的容差转换为相应的精度
            chop_prec = int(round(-3.321*math.log10(chop) + 2.5))
            if chop_prec == 3:
                chop_prec -= 1
        # 对结果 r 进行舍入处理
        r = chop_parts(r, chop_prec)
    
    if options.get("strict"):
        # 如果 options 中包含 'strict' 键，则检查目标的正确性
        check_target(x, r, prec)
    
    return r
    """`
# 将 ``evalf`` 返回的四元组转换为 ``mpf`` 或 ``mpc`` 对象
    """Turn the quad returned by ``evalf`` into an ``mpf`` or ``mpc``. """
    # 根据上下文选择创建 mpc 对象的函数，默认使用 make_mpc
    mpc = make_mpc if ctx is None else ctx.make_mpc
    # 根据上下文选择创建 mpf 对象的函数，默认使用 make_mpf
    mpf = make_mpf if ctx is None else ctx.make_mpf
    # 如果四元组 q 的实部是复数无穷，抛出 NotImplementedError 异常
    if q is S.ComplexInfinity:
        raise NotImplementedError
    # 将四元组 q 解包为实部 re 和虚部 im，以及两个未使用的值 _
    re, im, _, _ = q
    # 如果虚部 im 不为零
    if im:
        # 如果实部 re 为零，将实部设为 fzero
        if not re:
            re = fzero
        # 返回 mpc 对象，实部和虚部为 (re, im)
        return mpc((re, im))
    # 如果虚部 im 为零且实部 re 不为零
    elif re:
        # 返回 mpf 对象，实部为 re
        return mpf(re)
    # 如果实部 re 和虚部 im 都为零，返回 mpf 对象，实部为 fzero
    else:
        return mpf(fzero)
class EvalfMixin:
    """Mixin class adding evalf capability."""

    __slots__ = ()  # type: tTuple[str, ...]

    n = evalf  # 设置 n 属性为 evalf 函数的引用

    def _evalf(self, prec):
        """Helper for evalf. Does the same thing but takes binary precision"""
        r = self._eval_evalf(prec)  # 调用 _eval_evalf 方法，传入精度参数 prec
        if r is None:
            r = self
        return r

    def _eval_evalf(self, prec):
        return  # _eval_evalf 方法暂时未实现，返回 None

    def _to_mpmath(self, prec, allow_ints=True):
        # mpmath functions accept ints as input
        errmsg = "cannot convert to mpmath number"
        if allow_ints and self.is_Integer:  # 如果 allow_ints 为 True 并且 self 是整数
            return self.p  # 返回整数部分
        if hasattr(self, '_as_mpf_val'):  # 如果对象有 _as_mpf_val 方法
            return make_mpf(self._as_mpf_val(prec))  # 调用 _as_mpf_val 方法，并将其转换为 mpmath 的浮点数表示
        try:
            result = evalf(self, prec, {})  # 调用 evalf 函数计算结果
            return quad_to_mpmath(result)  # 将结果转换为 mpmath 的表示形式
        except NotImplementedError:
            v = self._eval_evalf(prec)  # 调用 _eval_evalf 方法获取数值
            if v is None:
                raise ValueError(errmsg)  # 如果获取的值是 None，则抛出值错误异常
            if v.is_Float:
                return make_mpf(v._mpf_)  # 如果 v 是浮点数，则返回其 mpmath 表示
            # Number + Number*I is also fine
            re, im = v.as_real_imag()  # 获取 v 的实部和虚部
            if allow_ints and re.is_Integer:
                re = from_int(re.p)  # 如果实部是整数，则转换为 mpmath 的整数表示
            elif re.is_Float:
                re = re._mpf_  # 否则，转换为 mpmath 的浮点数表示
            else:
                raise ValueError(errmsg)  # 如果不能转换，则抛出值错误异常
            if allow_ints and im.is_Integer:
                im = from_int(im.p)  # 如果虚部是整数，则转换为 mpmath 的整数表示
            elif im.is_Float:
                im = im._mpf_  # 否则，转换为 mpmath 的浮点数表示
            else:
                raise ValueError(errmsg)  # 如果不能转换，则抛出值错误异常
            return make_mpc((re, im))  # 构造复数表示并返回


def N(x, n=15, **options):
    r"""
    Calls x.evalf(n, \*\*options).

    Explanations
    ============

    Both .n() and N() are equivalent to .evalf(); use the one that you like better.
    See also the docstring of .evalf() for information on the options.

    Examples
    ========

    >>> from sympy import Sum, oo, N
    >>> from sympy.abc import k
    >>> Sum(1/k**k, (k, 1, oo))
    Sum(k**(-k), (k, 1, oo))
    >>> N(_, 4)
    1.291

    """
    # by using rational=True, any evaluation of a string
    # will be done using exact values for the Floats
    return sympify(x, rational=True).evalf(n, **options)  # 将 x 转换为 sympy 表达式，并调用 evalf 方法计算数值


def _evalf_with_bounded_error(x: 'Expr', eps: 'Optional[Expr]' = None,
                              m: int = 0,
                              options: Optional[OPT_DICT] = None) -> TMP_RES:
    """
    Evaluate *x* to within a bounded absolute error.

    Parameters
    ==========

    x : Expr
        The quantity to be evaluated.
    eps : Expr, None, optional (default=None)
        Positive real upper bound on the acceptable error.
    m : int, optional (default=0)
        If *eps* is None, then use 2**(-m) as the upper bound on the error.
    options: OPT_DICT
        As in the ``evalf`` function.

    Returns
    =======

    A tuple ``(re, im, re_acc, im_acc)``, as returned by ``evalf``.

    See Also
    ========

    evalf

    """
    # 检查 eps 是否已指定
    if eps is not None:
        # 检查 eps 是否为有理数或浮点数且大于0，否则抛出数值错误异常
        if not (eps.is_Rational or eps.is_Float) or not eps > 0:
            raise ValueError("eps must be positive")
        # 计算 1/eps 的浮点数结果，并进行评估，获取其有效数字和指数
        r, _, _, _ = evalf(1/eps, 1, {})
        # 使用快速对数函数计算 r 的对数
        m = fastlog(r)

    # 对输入 x 进行浮点数评估，获取其有效数字和指数
    c, d, _, _ = evalf(x, 1, {})
    # 注释：如果 x = a + b*I，则 |a| <= 2|c| 且 |b| <= 2|d|，等号仅在零情况下成立。
    # 如果 a 非零，则 |c| = 2**nc（其中 nc 是某个整数），且 c 具有比特计数 1。
    # 因此，2**fastlog(c) = 2**(nc+1) = 2|c| 是 |a| 的一个上界。对于 b 和 d 也是类似。
    nr, ni = fastlog(c), fastlog(d)
    # 计算有效数字的最大值加1
    n = max(nr, ni) + 1
    # 如果 x 是 0，则 n 是 MINUS_INF，且 p 将为 1。否则，
    # n - 1 位可越过 a 和 b 的整数部分，+1 考虑到 |x|/max(|a|, |b|) 的最大值不超过 sqrt(2) 的因子。
    p = max(1, m + n + 1)

    # 如果 options 为 None，则设为空字典
    options = options or {}
    # 返回对 x 进行浮点数评估的结果
    return evalf(x, p, options)
```