# `D:\src\scipysrc\sympy\sympy\polys\domains\mpelements.py`

```
"""Real and complex elements. """

# 导入必要的模块和类
from sympy.external.gmpy import MPQ
from sympy.polys.domains.domainelement import DomainElement
from sympy.utilities import public

# 导入特定的数学计算模块和函数
from mpmath.ctx_mp_python import PythonMPContext, _mpf, _mpc, _constant
from mpmath.libmp import (MPZ_ONE, fzero, fone, finf, fninf, fnan,
    round_nearest, mpf_mul, repr_dps, int_types,
    from_int, from_float, from_str, to_rational)

# 声明一个公共类 RealElement，继承自 _mpf 和 DomainElement
@public
class RealElement(_mpf, DomainElement):
    """An element of a real domain. """

    # 定义只读属性 _mpf_
    __slots__ = ('__mpf__',)
    _mpf_ = property(lambda self: self.__mpf__, lambda self, val: setattr(self, '__mpf__', val))

    # 返回该元素所属的父类对象
    def parent(self):
        return self.context._parent

# 声明一个公共类 ComplexElement，继承自 _mpc 和 DomainElement
@public
class ComplexElement(_mpc, DomainElement):
    """An element of a complex domain. """

    # 定义只读属性 _mpc_
    __slots__ = ('__mpc__',)
    _mpc_ = property(lambda self: self.__mpc__, lambda self, val: setattr(self, '__mpc__', val))

    # 返回该元素所属的父类对象
    def parent(self):
        return self.context._parent

# 使用 object.__new__ 来创建一个新对象
new = object.__new__

# 声明一个公共类 MPContext，继承自 PythonMPContext
@public
class MPContext(PythonMPContext):

    # 初始化方法，接收精度、十进制小数点、容差和真实标记等参数
    def __init__(ctx, prec=53, dps=None, tol=None, real=False):
        ctx._prec_rounding = [prec, round_nearest]  # 设置精度和舍入方式

        # 根据传入的 dps 或 prec 设置精度
        if dps is None:
            ctx._set_prec(prec)
        else:
            ctx._set_dps(dps)

        ctx.mpf = RealElement  # 设置实数元素类
        ctx.mpc = ComplexElement  # 设置复数元素类
        ctx.mpf._ctxdata = [ctx.mpf, new, ctx._prec_rounding]  # 设置实数元素类的上下文数据
        ctx.mpc._ctxdata = [ctx.mpc, new, ctx._prec_rounding]  # 设置复数元素类的上下文数据

        # 根据 real 参数设置上下文
        if real:
            ctx.mpf.context = ctx
        else:
            ctx.mpc.context = ctx

        ctx.constant = _constant  # 设置常数对象
        ctx.constant._ctxdata = [ctx.mpf, new, ctx._prec_rounding]  # 设置常数对象的上下文数据
        ctx.constant.context = ctx

        ctx.types = [ctx.mpf, ctx.mpc, ctx.constant]  # 定义支持的数据类型列表
        ctx.trap_complex = True  # 开启复数异常捕获
        ctx.pretty = True  # 开启输出美化

        # 根据 tol 参数设置容差值
        if tol is None:
            ctx.tol = ctx._make_tol()
        elif tol is False:
            ctx.tol = fzero
        else:
            ctx.tol = ctx._convert_tol(tol)

        ctx.tolerance = ctx.make_mpf(ctx.tol)  # 转换容差值为实数元素

        # 根据容差值设置最大分母
        if not ctx.tolerance:
            ctx.max_denom = 1000000
        else:
            ctx.max_denom = int(1 / ctx.tolerance)

        # 初始化一些常见数学常数
        ctx.zero = ctx.make_mpf(fzero)
        ctx.one = ctx.make_mpf(fone)
        ctx.j = ctx.make_mpc((fzero, fone))
        ctx.inf = ctx.make_mpf(finf)
        ctx.ninf = ctx.make_mpf(fninf)
        ctx.nan = ctx.make_mpf(fnan)

    # 创建容差值的内部方法
    def _make_tol(ctx):
        hundred = (0, 25, 2, 5)
        eps = (0, MPZ_ONE, 1 - ctx.prec, 1)
        return mpf_mul(hundred, eps)

    # 创建容差值的公共方法
    def make_tol(ctx):
        return ctx.make_mpf(ctx._make_tol())
    # 将给定的容差值 tol 转换为正确的格式
    def _convert_tol(ctx, tol):
        # 如果 tol 是整数类型，则转换为对应的浮点数精确表示
        if isinstance(tol, int_types):
            return from_int(tol)
        # 如果 tol 是浮点数，则直接使用其浮点数精确表示
        if isinstance(tol, float):
            return from_float(tol)
        # 如果 tol 具有 _mpf_ 属性，返回其内部的多精度浮点数表示
        if hasattr(tol, "_mpf_"):
            return tol._mpf_
        # 否则，根据上下文的精度和舍入方式，将 tol 解析为多精度浮点数
        prec, rounding = ctx._prec_rounding
        if isinstance(tol, str):
            return from_str(tol, prec, rounding)
        # 如果无法识别 tol 的类型，则抛出 ValueError 异常
        raise ValueError("expected a real number, got %s" % tol)

    # 在无法从给定的 x 和 strings 创建多精度浮点数时，抛出 TypeError 异常
    def _convert_fallback(ctx, x, strings):
        raise TypeError("cannot create mpf from " + repr(x))

    # 返回上下文中的表示数字的字符串，基于当前的精确度
    @property
    def _repr_digits(ctx):
        return repr_dps(ctx._prec)

    # 返回上下文中的表示数字的字符串，基于当前的小数点位数
    @property
    def _str_digits(ctx):
        return ctx._dps

    # 将多精度浮点数 s 转换为最接近的有理数 p/q 对，如果 limit=True，则限制 q 不超过上下文中的最大分母
    def to_rational(ctx, s, limit=True):
        p, q = to_rational(s._mpf_)

        # 对于 GROUND_TYPES=flint 情况下，如果使用了 gmpy2，则 p 是 gmpy2.mpz 实例，需要转换为整数
        p = int(p)

        # 如果 limit=False 或者 q <= ctx.max_denom，则直接返回 p, q
        if not limit or q <= ctx.max_denom:
            return p, q

        # 否则，使用连分数法求解最接近的有理数
        p0, q0, p1, q1 = 0, 1, 1, 0
        n, d = p, q

        while True:
            a = n // d
            q2 = q0 + a * q1
            if q2 > ctx.max_denom:
                break
            p0, q0, p1, q1 = p1, q1, p0 + a * p1, q2
            n, d = d, n - a * d

        k = (ctx.max_denom - q0) // q1

        # 根据计算得到的 p0, q0, p1, q1 计算边界值
        number = MPQ(p, q)
        bound1 = MPQ(p0 + k * p1, q0 + k * q1)
        bound2 = MPQ(p1, q1)

        # 比较边界值和原始值 s 的差异，返回最接近的有理数的分子和分母
        if not bound2 or not bound1:
            return p, q
        elif abs(bound2 - number) <= abs(bound1 - number):
            return bound2.numerator, bound2.denominator
        else:
            return bound1.numerator, bound1.denominator

    # 判断两个多精度浮点数 s 和 t 是否几乎相等，根据相对误差 rel_eps 和绝对误差 abs_eps 进行判断
    def almosteq(ctx, s, t, rel_eps=None, abs_eps=None):
        t = ctx.convert(t)
        # 如果 abs_eps 和 rel_eps 都为 None，则使用上下文中的容差值或生成默认的容差值
        if abs_eps is None and rel_eps is None:
            rel_eps = abs_eps = ctx.tolerance or ctx.make_tol()
        # 如果 abs_eps 为 None，则将 rel_eps 转换为对应的浮点数
        if abs_eps is None:
            abs_eps = ctx.convert(rel_eps)
        # 如果 rel_eps 为 None，则将 abs_eps 转换为对应的浮点数
        elif rel_eps is None:
            rel_eps = ctx.convert(abs_eps)
        # 计算 s 和 t 的绝对差异
        diff = abs(s - t)
        # 如果绝对差异小于等于 abs_eps，则判定为几乎相等
        if diff <= abs_eps:
            return True
        # 否则，计算相对误差并与 rel_eps 进行比较
        abss = abs(s)
        abst = abs(t)
        if abss < abst:
            err = diff / abst
        else:
            err = diff / abss
        return err <= rel_eps
```