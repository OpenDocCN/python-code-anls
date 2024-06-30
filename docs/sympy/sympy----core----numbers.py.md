# `D:\src\scipysrc\sympy\sympy\core\numbers.py`

```
# 导入未来版本中的注释语法，允许在类型提示中使用字符串形式的类型名
from __future__ import annotations

# 导入数学库中的特定模块和函数
import numbers
import decimal
import fractions
import math

# 从本地的 containers 模块中导入 Tuple 类型
from .containers import Tuple

# 从本地的 sympify 模块中导入多个函数和异常
from .sympify import (SympifyError, _sympy_converter, sympify, _convert_numpy_types,
              _sympify, _is_numpy_instance)

# 从本地的 singleton 模块中导入 S 和 Singleton 类
from .singleton import S, Singleton

# 从本地的 basic 模块中导入 Basic 类
from .basic import Basic

# 从本地的 expr 模块中导入 Expr 和 AtomicExpr 类
from .expr import Expr, AtomicExpr

# 从本地的 evalf 模块中导入 pure_complex 函数
from .evalf import pure_complex

# 从本地的 cache 模块中导入 cacheit 和 clear_cache 函数
from .cache import cacheit, clear_cache

# 从本地的 decorators 模块中导入 _sympifyit 装饰器
from .decorators import _sympifyit

# 从本地的 intfunc 模块中导入多个函数
from .intfunc import num_digits, igcd, ilcm, mod_inverse, integer_nthroot

# 从本地的 logic 模块中导入 fuzzy_not 函数
from .logic import fuzzy_not

# 从本地的 kind 模块中导入 NumberKind 类
from .kind import NumberKind

# 从本地的 sorting 模块中导入 ordered 函数
from .sorting import ordered

# 从 sympy.external.gmpy 模块中导入 SYMPY_INTS, gmpy, flint
from sympy.external.gmpy import SYMPY_INTS, gmpy, flint

# 从 sympy.multipledispatch 模块中导入 dispatch 装饰器
from sympy.multipledispatch import dispatch

# 导入 mpmath 库
import mpmath

# 从 mpmath.libmp 模块中导入 bitcount 和 round_nearest 函数
import mpmath.libmp as mlib

# 从 mpmath.libmp.backend 模块中导入 MPZ 类
from mpmath.libmp.backend import MPZ

# 从 mpmath.libmp 模块中导入 mpf_pow, mpf_pi, mpf_e, phi_fixed 函数
from mpmath.libmp import mpf_pow, mpf_pi, mpf_e, phi_fixed

# 从 mpmath.ctx_mp_python 模块中导入 mpnumeric 类
from mpmath.ctx_mp_python import mpnumeric

# 从 mpmath.libmp.libmpf 模块中导入多个函数和常量
from mpmath.libmp.libmpf import (
    finf as _mpf_inf, fninf as _mpf_ninf,
    fnan as _mpf_nan, fzero, _normalize as mpf_normalize,
    prec_to_dps, dps_to_prec
)

# 从 sympy.utilities.misc 模块中导入 debug 函数
from sympy.utilities.misc import debug

# 从本地的 parameters 模块中导入 global_parameters
from .parameters import global_parameters

# 计算常数 log(2)
_LOG2 = math.log(2)

# 定义一个函数 comp，用于比较 z1 和 z2 的误差是否小于等于指定的容差 tol
def comp(z1, z2, tol=None):
    r"""Return a bool indicating whether the error between z1 and z2
    is $\le$ ``tol``.

    Examples
    ========

    If ``tol`` is ``None`` then ``True`` will be returned if
    :math:`|z1 - z2|\times 10^p \le 5` where $p$ is minimum value of the
    decimal precision of each value.

    >>> from sympy import comp, pi
    >>> pi4 = pi.n(4); pi4
    3.142
    >>> comp(_, 3.142)
    True
    >>> comp(pi4, 3.141)
    False
    >>> comp(pi4, 3.143)
    False

    A comparison of strings will be made
    if ``z1`` is a Number and ``z2`` is a string or ``tol`` is ''.

    >>> comp(pi4, 3.1415)
    True
    >>> comp(pi4, 3.1415, '')
    False

    When ``tol`` is provided and $z2$ is non-zero and
    :math:`|z1| > 1` the error is normalized by :math:`|z1|`:

    >>> abs(pi4 - 3.14)/pi4
    0.000509791731426756
    >>> comp(pi4, 3.14, .001)  # difference less than 0.1%
    True
    >>> comp(pi4, 3.14, .0005)  # difference less than 0.1%
    False

    When :math:`|z1| \le 1` the absolute error is used:

    >>> 1/pi4
    0.3183
    >>> abs(1/pi4 - 0.3183)/(1/pi4)
    3.07371499106316e-5
    >>> abs(1/pi4 - 0.3183)
    9.78393554684764e-6
    >>> comp(1/pi4, 0.3183, 1e-5)
    True

    To see if the absolute error between ``z1`` and ``z2`` is less
    than or equal to ``tol``, call this as ``comp(z1 - z2, 0, tol)``
    or ``comp(z1 - z2, tol=tol)``:

    >>> abs(pi4 - 3.14)
    0.00160156249999988
    >>> comp(pi4 - 3.14, 0, .002)
    True
    >>> comp(pi4 - 3.14, 0, .001)
    False
    """
    # 如果 z2 是字符串，则要求 z1 必须是一个 Number 类型的对象
    if isinstance(z2, str):
        if not pure_complex(z1, or_real=True):
            raise ValueError('when z2 is a str z1 must be a Number')
        return str(z1) == z2
    # 如果 z1 为假值，则交换 z1 和 z2 的位置
    if not z1:
        z1, z2 = z2, z1
    # 如果 z1 为假值（例如 None、空字符串、空列表等），返回 True
    if not z1:
        return True
    
    # 如果 tol 为假值（例如空字符串、None），执行以下操作
    if not tol:
        # 将 z1 和 z2 分别赋值给 a 和 b
        a, b = z1, z2
        
        # 如果 tol 是空字符串，返回 a 和 b 的字符串表示是否相等的布尔值
        if tol == '':
            return str(a) == str(b)
        
        # 如果 tol 是 None，将 a 和 b 转换为符号表达式
        if tol is None:
            a, b = sympify(a), sympify(b)
            
            # 如果 a 或 b 不是数字，抛出值错误异常
            if not all(i.is_number for i in (a, b)):
                raise ValueError('expecting 2 numbers')
            
            # 获取 a 和 b 中所有的浮点数
            fa = a.atoms(Float)
            fb = b.atoms(Float)
            
            # 如果 fa 和 fb 都为空集，表示没有浮点数，精确比较 a 和 b 是否相等
            if not fa and not fb:
                return a == b
            
            # 将 a 转换为纯复数
            for _ in range(2):
                ca = pure_complex(a, or_real=True)
                
                # 如果 ca 不存在且 fa 非空，将 a 转换为最小精度的浮点数，并重新尝试转换为纯复数
                if not ca:
                    if fa:
                        a = a.n(prec_to_dps(min(i._prec for i in fa)))
                        ca = pure_complex(a, or_real=True)
                        break
                    else:
                        fa, fb = fb, fa
                        a, b = b, a
            
            # 将 b 转换为纯复数
            cb = pure_complex(b)
            
            # 如果 cb 不存在且 fb 非空，将 b 转换为最小精度的浮点数，并重新尝试转换为纯复数
            if not cb and fb:
                b = b.n(prec_to_dps(min(i._prec for i in fb)))
                cb = pure_complex(b, or_real=True)
            
            # 如果 ca 和 cb 都存在且至少有一个是复数部分，逐个比较它们的复数部分
            if ca and cb and (ca[1] or cb[1]):
                return all(comp(i, j) for i, j in zip(ca, cb))
            
            # 计算容差 tol，使用最小精度的 10 的 tol 次方倍数
            tol = 10**prec_to_dps(min(a._prec, getattr(b, '_prec', a._prec)))
            
            # 比较 a 和 b 的差的绝对值乘以容差 tol 是否小于等于 5
            return int(abs(a - b)*tol) <= 5
    
    # 计算 z1 和 z2 的差的绝对值
    diff = abs(z1 - z2)
    
    # 计算 z1 的绝对值
    az1 = abs(z1)
    
    # 如果 z2 存在且 az1 大于 1，则返回 z1 和 z2 的差的绝对值除以 z1 的绝对值是否小于等于容差 tol
    if z2 and az1 > 1:
        return diff / az1 <= tol
    else:
        # 否则，直接比较 z1 和 z2 的差的绝对值是否小于等于容差 tol
        return diff <= tol
# 返回给定 mpf 元组根据指定精度归一化后的结果。在检查指数部分为零时，确定是否应返回零的情况下，mpf_norm 总是假定这是零，但实际可能不是，因为 "+inf"、"-inf" 和 "nan" 的 mpf 值的尾数也是零。
def mpf_norm(mpf, prec):
    """Return the mpf tuple normalized appropriately for the indicated
    precision after doing a check to see if zero should be returned or
    not when the mantissa is 0. ``mpf_normlize`` always assumes that this
    is zero, but it may not be since the mantissa for mpf's values "+inf",
    "-inf" and "nan" have a mantissa of zero, too.

    Note: this is not intended to validate a given mpf tuple, so sending
    mpf tuples that were not created by mpmath may produce bad results. This
    is only a wrapper to ``mpf_normalize`` which provides the check for non-
    zero mpfs that have a 0 for the mantissa.
    """
    sign, man, expt, bc = mpf
    if not man:
        # hack for mpf_normalize which does not do this;
        # it assumes that if man is zero the result is 0
        # (see issue 6639)
        if not bc:
            return fzero
        else:
            # don't change anything; this should already
            # be a well formed mpf tuple
            return mpf

    # Necessary if mpmath is using the gmpy backend
    from mpmath.libmp.backend import MPZ
    rv = mpf_normalize(sign, MPZ(man), expt, bc, prec, rnd)
    return rv

# 错误字典，用于设置 SymPy 是否应该在除以零时引发异常
_errdict = {"divide": False}


def seterr(divide=False):
    """
    Should SymPy raise an exception on 0/0 or return a nan?

    divide == True .... raise an exception
    divide == False ... return nan
    """
    if _errdict["divide"] != divide:
        clear_cache()
        _errdict["divide"] = divide


def _as_integer_ratio(p):
    neg_pow, man, expt, _ = getattr(p, '_mpf_', mpmath.mpf(p)._mpf_)
    p = [1, -1][neg_pow % 2]*man
    if expt < 0:
        q = 2**-expt
    else:
        q = 1
        p *= 2**expt
    return int(p), int(q)


def _decimal_to_Rational_prec(dec):
    """Convert an ordinary decimal instance to a Rational."""
    if not dec.is_finite():
        raise TypeError("dec must be finite, got %s." % dec)
    s, d, e = dec.as_tuple()
    prec = len(d)
    if e >= 0:  # it's an integer
        rv = Integer(int(dec))
    else:
        s = (-1)**s
        d = sum(di*10**i for i, di in enumerate(reversed(d)))
        rv = Rational(s*d, 10**-e)
    return rv, prec

# 字符串映射表，用于将字符串中的数字字符替换为 None
_dig = str.maketrans(dict.fromkeys('1234567890'))

def _literal_float(s):
    """return True if s is space-trimmed number literal else False

    Python allows underscore as digit separators: there must be a
    digit on each side. So neither a leading underscore nor a
    double underscore are valid as part of a number. A number does
    not have to precede the decimal point, but there must be a
    digit before the optional "e" or "E" that begins the signs
    exponent of the number which must be an integer, perhaps with
    underscore separators.

    SymPy allows space as a separator; if the calling routine replaces
    them with underscores then the same semantics will be enforced
    for them as for underscores: there can only be 1 *between* digits.
    """
    # 返回 True 如果 s 是去除空格的数字字面量，否则返回 False

    # Python 允许使用下划线作为数字分隔符：每一侧必须有一个数字。
    # 因此，前导下划线或双下划线都不是数字的一部分。
    # 数字不必在小数点之前，但是在可选的 "e" 或 "E" 之前必须有一个数字，它开头的指数必须是一个整数，可能有下划线分隔符。

    # SymPy 允许空格作为分隔符；如果调用例程用下划线替换它们，那么它们的语义将与下划线相同：只能在数字之间有 1 个分隔符。
    Python允许使用下划线作为数字的分隔符
    """
    # 浮点数字符串验证函数
    # 不检查 float(s) 的错误，因为无法确定 s 是否恶意。可能可以写一个正则表达式来处理，但大多数人可能无法理解它。
    # 参数 s 是待验证的浮点数字符串
    
    # 拆分字符串 s，获取尾数和指数部分
    parts = s.split('e')
    # 如果拆分后的部分大于2，则不符合浮点数格式，返回 False
    if len(parts) > 2:
        return False
    # 如果拆分后的部分等于2，则有尾数和指数
    if len(parts) == 2:
        m, e = parts
        # 如果指数部分以 '+' 或 '-' 开头，则去掉符号
        if e.startswith(tuple('+-')):
            e = e[1:]
        # 如果指数部分为空，则不符合浮点数格式，返回 False
        if not e:
            return False
    # 如果拆分后的部分等于1，则整个字符串为尾数，指数默认为 '1'
    else:
        m, e = s, '1'
    
    # 拆分尾数部分，获取整数部分和小数部分
    parts = m.split('.')
    # 如果拆分后的部分大于2，则不符合浮点数格式，返回 False
    if len(parts) > 2:
        return False
    # 如果拆分后的部分等于2，则有整数部分和小数部分
    elif len(parts) == 2:
        i, f = parts
    # 如果拆分后的部分等于1，则整个字符串为整数部分，小数部分默认为 '1'
    else:
        i, f = m, '1'
    
    # 如果整数部分和小数部分同时为空，则不符合浮点数格式，返回 False
    if not i and not f:
        return False
    # 如果整数部分以 '+' 或 '-' 开头，则去掉符号
    if i and i[0] in '+-':
        i = i[1:]
    # 如果整数部分为空，则默认为 '1'，用于处理如 -.3e4 这样的情况
    if not i:
        i = '1'
    # 如果小数部分为空，则默认为 '1'
    f = f or '1'
    
    # 检查整数部分 i、小数部分 f 和指数部分 e 中的每个组，确保只包含数字且不为空
    for n in (i, f, e):
        for g in n.split('_'):
            # 如果任何组为空或包含非数字字符，则不符合浮点数格式，返回 False
            if not g or g.translate(_dig):
                return False
    
    # 如果所有检查通过，则认为字符串 s 符合浮点数格式，返回 True
    return True
    """
# (a,b) -> gcd(a,b)

# TODO caching with decorator, but not to degrade performance

# 表示SymPy中的原子数的基类
class Number(AtomicExpr):
    """Represents atomic numbers in SymPy.

    Explanation
    ===========

    Floating point numbers are represented by the Float class.
    Rational numbers (of any size) are represented by the Rational class.
    Integer numbers (of any size) are represented by the Integer class.
    Float and Rational are subclasses of Number; Integer is a subclass
    of Rational.

    For example, ``2/3`` is represented as ``Rational(2, 3)`` which is
    a different object from the floating point number obtained with
    Python division ``2/3``. Even for numbers that are exactly
    represented in binary, there is a difference between how two forms,
    such as ``Rational(1, 2)`` and ``Float(0.5)``, are used in SymPy.
    The rational form is to be preferred in symbolic computations.

    Other kinds of numbers, such as algebraic numbers ``sqrt(2)`` or
    complex numbers ``3 + 4*I``, are not instances of Number class as
    they are not atomic.

    See Also
    ========

    Float, Integer, Rational
    """
    is_commutative = True
    is_number = True
    is_Number = True

    __slots__ = ()

    # Used to make max(x._prec, y._prec) return x._prec when only x is a float
    _prec = -1

    kind = NumberKind

    def __new__(cls, *obj):
        # 处理输入参数，根据类型创建相应的数值对象
        if len(obj) == 1:
            obj = obj[0]

        if isinstance(obj, Number):
            return obj
        if isinstance(obj, SYMPY_INTS):
            return Integer(obj)
        if isinstance(obj, tuple) and len(obj) == 2:
            return Rational(*obj)
        if isinstance(obj, (float, mpmath.mpf, decimal.Decimal)):
            return Float(obj)
        if isinstance(obj, str):
            _obj = obj.lower()  # float('INF') == float('inf')
            if _obj == 'nan':
                return S.NaN
            elif _obj == 'inf':
                return S.Infinity
            elif _obj == '+inf':
                return S.Infinity
            elif _obj == '-inf':
                return S.NegativeInfinity
            val = sympify(obj)
            if isinstance(val, Number):
                return val
            else:
                raise ValueError('String "%s" does not denote a Number' % obj)
        # 如果输入类型不在预期的范围内，则抛出异常
        msg = "expected str|int|long|float|Decimal|Number object but got %r"
        raise TypeError(msg % type(obj).__name__)

    def could_extract_minus_sign(self):
        # 检查是否可以提取负号
        return bool(self.is_extended_negative)

    def invert(self, other, *gens, **args):
        # 反转操作，根据other的类型选择相应的操作
        from sympy.polys.polytools import invert
        if getattr(other, 'is_number', True):
            return mod_inverse(self, other)
        return invert(self, other, *gens, **args)
    # 导入符号运算库中的符号函数
    def __divmod__(self, other):
        from sympy.functions.elementary.complexes import sign

        # 尝试将参数转换为数字，处理无穷大和NaN的情况
        try:
            other = Number(other)
            if self.is_infinite or S.NaN in (self, other):
                return (S.NaN, S.NaN)
        except TypeError:
            return NotImplemented

        # 若除数为零，抛出零除错误
        if not other:
            raise ZeroDivisionError('modulo by zero')

        # 处理整数除法的情况
        if self.is_Integer and other.is_Integer:
            return Tuple(*divmod(self.p, other.p))
        # 处理与浮点数的除法
        elif isinstance(other, Float):
            rat = self/Rational(other)
        else:
            rat = self/other

        # 若除数为有限数，则计算商和余数
        if other.is_finite:
            # 计算整数部分
            w = int(rat) if rat >= 0 else int(rat) - 1
            r = self - other*w
            # 若余数与除数浮点数相等，增加整数部分并重置余数为零
            if r == Float(other):
                w += 1
                r = 0
            # 处理当self或other为浮点数时的情况，确保余数为浮点数
            if isinstance(self, Float) or isinstance(other, Float):
                r = Float(r)  # 以防w或r为0
        else:
            # 处理无穷大和符号相同的情况
            w = 0 if not self or (sign(self) == sign(other)) else -1
            r = other if w else self

        # 返回商和余数的元组
        return Tuple(w, r)

    # 反向除法运算符的实现，调用内置divmod方法
    def __rdivmod__(self, other):
        try:
            other = Number(other)
        except TypeError:
            return NotImplemented
        return divmod(other, self)

    # 返回符号表达式的多精度浮点值的估算，至少精确到prec位
    def _as_mpf_val(self, prec):
        """Evaluation of mpf tuple accurate to at least prec bits."""
        raise NotImplementedError('%s needs ._as_mpf_val() method' %
            (self.__class__.__name__))

    # 返回浮点数表达式的指定精度的新浮点数
    def _eval_evalf(self, prec):
        return Float._new(self._as_mpf_val(prec), prec)

    # 返回浮点操作的多精度浮点值及其精度
    def _as_mpf_op(self, prec):
        prec = max(prec, self._prec)
        return self._as_mpf_val(prec), prec

    # 将符号表达式转换为Python内置的浮点数类型
    def __float__(self):
        return mlib.to_float(self._as_mpf_val(53))

    # 抛出未实现的异常，要求具体类实现.floor()方法
    def floor(self):
        raise NotImplementedError('%s needs .floor() method' %
            (self.__class__.__name__))

    # 抛出未实现的异常，要求具体类实现.ceiling()方法
    def ceiling(self):
        raise NotImplementedError('%s needs .ceiling() method' %
            (self.__class__.__name__))

    # 返回.floor()方法的别名，用于向下取整
    def __floor__(self):
        return self.floor()

    # 返回.ceiling()方法的别名，用于向上取整
    def __ceil__(self):
        return self.ceiling()

    # 返回自身的共轭复数
    def _eval_conjugate(self):
        return self

    # 返回关于给定符号的级数展开的顺序对象
    def _eval_order(self, *symbols):
        from sympy.series.order import Order
        # Order(5, x, y) -> Order(1,x,y)
        return Order(S.One, *symbols)

    # 实现符号替换，如果old为-self，则返回-new，否则返回self
    def _eval_subs(self, old, new):
        if old == -self:
            return -new
        return self  # 没有其他可能性

    # 返回类的排序关键字
    @classmethod
    def class_key(cls):
        return 1, 0, 'Number'

    # 缓存化的排序关键字，用于快速排序操作
    @cacheit
    def sort_key(self, order=None):
        return self.class_key(), (0, ()), (), self

    # 使用装饰器指定_sympifyit('other', NotImplemented)，暂未提供具体实现
    @_sympifyit('other', NotImplemented)
    # 定义特殊方法 __add__，用于处理对象与数值相加的操作
    def __add__(self, other):
        # 检查 other 是否为 Number 类型，并且全局参数中允许评估表达式
        if isinstance(other, Number) and global_parameters.evaluate:
            # 如果 other 是 S.NaN，则返回 S.NaN
            if other is S.NaN:
                return S.NaN
            # 如果 other 是 S.Infinity，则返回 S.Infinity
            elif other is S.Infinity:
                return S.Infinity
            # 如果 other 是 S.NegativeInfinity，则返回 S.NegativeInfinity
            elif other is S.NegativeInfinity:
                return S.NegativeInfinity
        # 调用父类的 __add__ 方法处理其他情况
        return AtomicExpr.__add__(self, other)

    # 使用装饰器定义特殊方法 __sub__，用于处理对象与数值相减的操作
    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        # 检查 other 是否为 Number 类型，并且全局参数中允许评估表达式
        if isinstance(other, Number) and global_parameters.evaluate:
            # 如果 other 是 S.NaN，则返回 S.NaN
            if other is S.NaN:
                return S.NaN
            # 如果 other 是 S.Infinity，则返回 S.NegativeInfinity
            elif other is S.Infinity:
                return S.NegativeInfinity
            # 如果 other 是 S.NegativeInfinity，则返回 S.Infinity
            elif other is S.NegativeInfinity:
                return S.Infinity
        # 调用父类的 __sub__ 方法处理其他情况
        return AtomicExpr.__sub__(self, other)

    # 使用装饰器定义特殊方法 __mul__，用于处理对象与数值或元组相乘的操作
    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        # 检查 other 是否为 Number 类型，并且全局参数中允许评估表达式
        if isinstance(other, Number) and global_parameters.evaluate:
            # 如果 other 是 S.NaN，则返回 S.NaN
            if other is S.NaN:
                return S.NaN
            # 如果 other 是 S.Infinity，则根据 self 的情况返回相应值
            elif other is S.Infinity:
                if self.is_zero:
                    return S.NaN
                elif self.is_positive:
                    return S.Infinity
                else:
                    return S.NegativeInfinity
            # 如果 other 是 S.NegativeInfinity，则根据 self 的情况返回相应值
            elif other is S.NegativeInfinity:
                if self.is_zero:
                    return S.NaN
                elif self.is_positive:
                    return S.NegativeInfinity
                else:
                    return S.Infinity
        # 如果 other 是元组类型，则返回 NotImplemented
        elif isinstance(other, Tuple):
            return NotImplemented
        # 调用父类的 __mul__ 方法处理其他情况
        return AtomicExpr.__mul__(self, other)

    # 定义特殊方法 __truediv__，用于处理对象与数值相除的操作
    @_sympifyit('other', NotImplemented)
    def __truediv__(self, other):
        # 检查 other 是否为 Number 类型，并且全局参数中允许评估表达式
        if isinstance(other, Number) and global_parameters.evaluate:
            # 如果 other 是 S.NaN，则返回 S.NaN
            if other is S.NaN:
                return S.NaN
            # 如果 other 是 S.Infinity 或 S.NegativeInfinity，则返回 S.Zero
            elif other in (S.Infinity, S.NegativeInfinity):
                return S.Zero
        # 调用父类的 __truediv__ 方法处理其他情况
        return AtomicExpr.__truediv__(self, other)

    # 定义特殊方法 __eq__，抛出 NotImplementedError 异常
    def __eq__(self, other):
        raise NotImplementedError('%s needs .__eq__() method' %
            (self.__class__.__name__))

    # 定义特殊方法 __ne__，抛出 NotImplementedError 异常
    def __ne__(self, other):
        raise NotImplementedError('%s needs .__ne__() method' %
            (self.__class__.__name__))

    # 定义特殊方法 __lt__，抛出 NotImplementedError 异常
    def __lt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s < %s" % (self, other))
        raise NotImplementedError('%s needs .__lt__() method' %
            (self.__class__.__name__))

    # 定义特殊方法 __le__，抛出 NotImplementedError 异常
    def __le__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s <= %s" % (self, other))
        raise NotImplementedError('%s needs .__le__() method' %
            (self.__class__.__name__))
    def __gt__(self, other):
        try:
            # 尝试将 other 转换为符号表达式
            other = _sympify(other)
        except SympifyError:
            # 如果转换失败，抛出类型错误异常
            raise TypeError("Invalid comparison %s > %s" % (self, other))
        # 返回 other 是否小于 self 的结果
        return _sympify(other).__lt__(self)

    def __ge__(self, other):
        try:
            # 尝试将 other 转换为符号表达式
            other = _sympify(other)
        except SympifyError:
            # 如果转换失败，抛出类型错误异常
            raise TypeError("Invalid comparison %s >= %s" % (self, other))
        # 返回 other 是否小于等于 self 的结果
        return _sympify(other).__le__(self)

    def __hash__(self):
        # 调用父类的哈希函数
        return super().__hash__()

    def is_constant(self, *wrt, **flags):
        # 符号表达式默认是常数，无论传入的参数如何
        return True

    def as_coeff_mul(self, *deps, rational=True, **kwargs):
        # 将符号表达式表示为系数乘积形式
        # 如果表达式是有理数或者 rational 参数为 False，则直接返回自身和空的乘数元组
        if self.is_Rational or not rational:
            return self, ()
        elif self.is_negative:
            # 如果表达式是负数，则返回 -1 和去掉负号后的表达式
            return S.NegativeOne, (-self,)
        # 否则返回 1 和包含自身的乘数元组
        return S.One, (self,)

    def as_coeff_add(self, *deps):
        # 将符号表达式表示为系数加和形式
        # 如果表达式是有理数，则返回自身和空的加数元组
        if self.is_Rational:
            return self, ()
        # 否则返回 0 和包含自身的加数元组
        return S.Zero, (self,)

    def as_coeff_Mul(self, rational=False):
        """Efficiently extract the coefficient of a product."""
        # 高效提取表达式的乘积形式系数
        if not rational:
            # 如果不考虑有理数，则返回自身和 1
            return self, S.One
        # 否则返回 1 和自身
        return S.One, self

    def as_coeff_Add(self, rational=False):
        """Efficiently extract the coefficient of a summation."""
        # 高效提取表达式的加和形式系数
        if not rational:
            # 如果不考虑有理数，则返回自身和 0
            return self, S.Zero
        # 否则返回 0 和自身
        return S.Zero, self

    def gcd(self, other):
        """Compute GCD of `self` and `other`. """
        # 计算 self 和 other 的最大公约数
        from sympy.polys.polytools import gcd
        return gcd(self, other)

    def lcm(self, other):
        """Compute LCM of `self` and `other`. """
        # 计算 self 和 other 的最小公倍数
        from sympy.polys.polytools import lcm
        return lcm(self, other)

    def cofactors(self, other):
        """Compute GCD and cofactors of `self` and `other`. """
        # 计算 self 和 other 的最大公约数以及其余子
        from sympy.polys.polytools import cofactors
        return cofactors(self, other)
# 定义一个浮点数类 Float，继承自 Number 类
class Float(Number):
    """Represent a floating-point number of arbitrary precision.

    Examples
    ========

    >>> from sympy import Float
    >>> Float(3.5)
    3.50000000000000
    >>> Float(3)
    3.00000000000000

    Creating Floats from strings (and Python ``int`` and ``long``
    types) will give a minimum precision of 15 digits, but the
    precision will automatically increase to capture all digits
    entered.

    >>> Float(1)
    1.00000000000000
    >>> Float(10**20)
    100000000000000000000.
    >>> Float('1e20')
    100000000000000000000.

    However, *floating-point* numbers (Python ``float`` types) retain
    only 15 digits of precision:

    >>> Float(1e20)
    1.00000000000000e+20
    >>> Float(1.23456789123456789)
    1.23456789123457

    It may be preferable to enter high-precision decimal numbers
    as strings:

    >>> Float('1.23456789123456789')
    1.23456789123456789

    The desired number of digits can also be specified:

    >>> Float('1e-3', 3)
    0.00100
    >>> Float(100, 4)
    100.0

    Float can automatically count significant figures if a null string
    is sent for the precision; spaces or underscores are also allowed. (Auto-
    counting is only allowed for strings, ints and longs).

    >>> Float('123 456 789.123_456', '')
    123456789.123456
    >>> Float('12e-3', '')
    0.012
    >>> Float(3, '')
    3.

    If a number is written in scientific notation, only the digits before the
    exponent are considered significant if a decimal appears, otherwise the
    "e" signifies only how to move the decimal:

    >>> Float('60.e2', '')  # 2 digits significant
    6.0e+3
    >>> Float('60e2', '')  # 4 digits significant
    6000.
    >>> Float('600e-2', '')  # 3 digits significant
    6.00

    Notes
    =====

    Floats are inexact by their nature unless their value is a binary-exact
    value.

    >>> approx, exact = Float(.1, 1), Float(.125, 1)

    For calculation purposes, evalf needs to be able to change the precision
    but this will not increase the accuracy of the inexact value. The
    following is the most accurate 5-digit approximation of a value of 0.1
    that had only 1 digit of precision:

    >>> approx.evalf(5)
    0.099609

    By contrast, 0.125 is exact in binary (as it is in base 10) and so it
    can be passed to Float or evalf to obtain an arbitrary precision with
    matching accuracy:

    >>> Float(exact, 5)
    0.12500
    >>> exact.evalf(20)
    0.12500000000000000000

    Trying to make a high-precision Float from a float is not disallowed,
    but one must keep in mind that the *underlying float* (not the apparent
    decimal value) is being obtained with high precision. For example, 0.3
    does not have a finite binary representation. The closest rational is
    the fraction 5404319552844595/2**54. So if you try to obtain a Float of
    0.3 to 20 digits of precision you will not see the same thing as 0.3
    followed by 19 zeros:

    >>> Float(0.3, 20)
    """
    # 定义一个 Python 类的槽（slots），限定只能存储指定的属性
    __slots__ = ('_mpf_', '_prec')
    
    # Float 类中的一个元组属性，存储具体的浮点数值和精度信息
    _mpf_: tuple[int, int, int, int]
    
    # Float 类的类属性，用于指示该类对象是一个浮点数
    is_Float = True
    
    # 以下属性用于描述 Float 类对象的数值性质
    is_rational = None       # 是否为有理数
    is_irrational = None     # 是否为无理数
    is_number = True         # 是否为数值
    
    is_real = True           # 是否为实数
    is_extended_real = True  # 是否为扩展实数（包括无穷大和无穷小）
    
    # 以下是一个字符串的转换表，用于在操作中移除非数字字符
    _remove_non_digits = str.maketrans(dict.fromkeys("-+_."))
    
    # 定义一个类方法，用于创建 Float 类的实例
    @classmethod
    def _new(cls, _mpf_, _prec, zero=True):
        # 创建新的 Float 对象的特殊情况处理
        if zero and _mpf_ == fzero:
            return S.Zero  # 如果 zero 为 True 且 _mpf_ 为 fzero，则返回整数 0
        elif _mpf_ == _mpf_nan:
            return S.NaN  # 如果 _mpf_ 为 _mpf_nan，则返回符号常量 NaN
        elif _mpf_ == _mpf_inf:
            return S.Infinity  # 如果 _mpf_ 为 _mpf_inf，则返回符号常量 Infinity
        elif _mpf_ == _mpf_ninf:
            return S.NegativeInfinity  # 如果 _mpf_ 为 _mpf_ninf，则返回符号常量 NegativeInfinity

        # 创建一个新的 Expr 对象并初始化
        obj = Expr.__new__(cls)
        obj._mpf_ = mpf_norm(_mpf_, _prec)  # 规范化 _mpf_ 到指定精度 _prec
        obj._prec = _prec  # 设置对象的精度属性
        return obj  # 返回创建的对象实例

    def __getnewargs_ex__(self):
        # 返回对象实例的可哈希参数和关键字参数
        sign, man, exp, bc = self._mpf_
        arg = (sign, hex(man)[2:], exp, bc)  # 将 _mpf_ 转换为可哈希的参数形式
        kwargs = {'precision': self._prec}  # 设置关键字参数 precision
        return ((arg,), kwargs)  # 返回参数元组和关键字参数字典

    def _hashable_content(self):
        # 返回对象的可哈希内容，包括 _mpf_ 和 _prec
        return (self._mpf_, self._prec)

    def floor(self):
        # 返回对象的下取整结果作为整数类型
        return Integer(int(mlib.to_int(
            mlib.mpf_floor(self._mpf_, self._prec))))

    def ceiling(self):
        # 返回对象的上取整结果作为整数类型
        return Integer(int(mlib.to_int(
            mlib.mpf_ceil(self._mpf_, self._prec))))

    def __floor__(self):
        # 返回对象的下取整结果
        return self.floor()

    def __ceil__(self):
        # 返回对象的上取整结果
        return self.ceiling()

    @property
    def num(self):
        # 返回对象的数值部分作为 mpmath 的 mpf 类型
        return mpmath.mpf(self._mpf_)

    def _as_mpf_val(self, prec):
        # 将对象转换为指定精度 prec 的 mpf 值
        rv = mpf_norm(self._mpf_, prec)  # 规范化对象的 _mpf_ 到指定精度 prec
        if rv != self._mpf_ and self._prec == prec:
            debug(self._mpf_, rv)  # 如果规范化后的值不等于原始 _mpf_，并且精度相同，则进行调试输出
        return rv  # 返回规范化后的值

    def _as_mpf_op(self, prec):
        # 返回对象的 _mpf_ 值和指定精度 prec 中的较大者
        return self._mpf_, max(prec, self._prec)

    def _eval_is_finite(self):
        # 判断对象是否为有限数
        if self._mpf_ in (_mpf_inf, _mpf_ninf):
            return False  # 如果 _mpf_ 是正无穷或负无穷，则返回 False
        return True  # 否则返回 True

    def _eval_is_infinite(self):
        # 判断对象是否为无穷数
        if self._mpf_ in (_mpf_inf, _mpf_ninf):
            return True  # 如果 _mpf_ 是正无穷或负无穷，则返回 True
        return False  # 否则返回 False

    def _eval_is_integer(self):
        # 判断对象是否为整数
        if self._mpf_ == fzero:
            return True  # 如果 _mpf_ 为零，则返回 True
        if not int_valued(self):
            return False  # 如果对象不具有整数值特性，则返回 False

    def _eval_is_negative(self):
        # 判断对象是否为负数
        if self._mpf_ in (_mpf_ninf, _mpf_inf):
            return False  # 如果 _mpf_ 是正无穷或负无穷，则返回 False
        return self.num < 0  # 否则比较对象的数值部分是否小于零，返回结果

    def _eval_is_positive(self):
        # 判断对象是否为正数
        if self._mpf_ in (_mpf_ninf, _mpf_inf):
            return False  # 如果 _mpf_ 是正无穷或负无穷，则返回 False
        return self.num > 0  # 否则比较对象的数值部分是否大于零，返回结果

    def _eval_is_extended_negative(self):
        # 判断对象是否为扩展负数
        if self._mpf_ == _mpf_ninf:
            return True  # 如果 _mpf_ 是负无穷，则返回 True
        if self._mpf_ == _mpf_inf:
            return False  # 如果 _mpf_ 是正无穷，则返回 False
        return self.num < 0  # 否则比较对象的数值部分是否小于零，返回结果

    def _eval_is_extended_positive(self):
        # 判断对象是否为扩展正数
        if self._mpf_ == _mpf_inf:
            return True  # 如果 _mpf_ 是正无穷，则返回 True
        if self._mpf_ == _mpf_ninf:
            return False  # 如果 _mpf_ 是负无穷，则返回 False
        return self.num > 0  # 否则比较对象的数值部分是否大于零，返回结果

    def _eval_is_zero(self):
        # 判断对象是否为零
        return self._mpf_ == fzero  # 返回 _mpf_ 是否等于零的比较结果

    def __bool__(self):
        # 判断对象是否为真（非零）
        return self._mpf_ != fzero  # 返回 _mpf_ 是否不等于零的比较结果

    def __neg__(self):
        # 返回对象的相反数
        if not self:
            return self  # 如果对象为零，则返回自身
        return Float._new(mlib.mpf_neg(self._mpf_), self._prec)  # 否则返回相反数的 Float 对象

    @_sympifyit('other', NotImplemented)
    # 定义 __add__ 方法，用于实现加法操作
    def __add__(self, other):
        # 如果 other 是 Number 类型且全局参数 evaluate 为真
        if isinstance(other, Number) and global_parameters.evaluate:
            # 将 other 转换为 MPF 操作数，使用当前对象的精度
            rhs, prec = other._as_mpf_op(self._prec)
            # 调用 mlib.mpf_add 执行 MPF 加法操作，并返回新的 Float 对象
            return Float._new(mlib.mpf_add(self._mpf_, rhs, prec, rnd), prec)
        # 否则调用 Number 类的 __add__ 方法
        return Number.__add__(self, other)

    # 应用 _sympifyit 装饰器，定义 __sub__ 方法，实现减法操作
    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        # 如果 other 是 Number 类型且全局参数 evaluate 为真
        if isinstance(other, Number) and global_parameters.evaluate:
            # 将 other 转换为 MPF 操作数，使用当前对象的精度
            rhs, prec = other._as_mpf_op(self._prec)
            # 调用 mlib.mpf_sub 执行 MPF 减法操作，并返回新的 Float 对象
            return Float._new(mlib.mpf_sub(self._mpf_, rhs, prec, rnd), prec)
        # 否则调用 Number 类的 __sub__ 方法
        return Number.__sub__(self, other)

    # 应用 _sympifyit 装饰器，定义 __mul__ 方法，实现乘法操作
    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        # 如果 other 是 Number 类型且全局参数 evaluate 为真
        if isinstance(other, Number) and global_parameters.evaluate:
            # 将 other 转换为 MPF 操作数，使用当前对象的精度
            rhs, prec = other._as_mpf_op(self._prec)
            # 调用 mlib.mpf_mul 执行 MPF 乘法操作，并返回新的 Float 对象
            return Float._new(mlib.mpf_mul(self._mpf_, rhs, prec, rnd), prec)
        # 否则调用 Number 类的 __mul__ 方法
        return Number.__mul__(self, other)

    # 应用 _sympifyit 装饰器，定义 __truediv__ 方法，实现真除法操作
    @_sympifyit('other', NotImplemented)
    def __truediv__(self, other):
        # 如果 other 是 Number 类型且不为零且全局参数 evaluate 为真
        if isinstance(other, Number) and other != 0 and global_parameters.evaluate:
            # 将 other 转换为 MPF 操作数，使用当前对象的精度
            rhs, prec = other._as_mpf_op(self._prec)
            # 调用 mlib.mpf_div 执行 MPF 真除法操作，并返回新的 Float 对象
            return Float._new(mlib.mpf_div(self._mpf_, rhs, prec, rnd), prec)
        # 否则调用 Number 类的 __truediv__ 方法
        return Number.__truediv__(self, other)

    # 应用 _sympifyit 装饰器，定义 __mod__ 方法，实现取模操作
    @_sympifyit('other', NotImplemented)
    def __mod__(self, other):
        # 如果 other 是 Rational 类型且分母不为1且全局参数 evaluate 为真
        if isinstance(other, Rational) and other.q != 1 and global_parameters.evaluate:
            # 使用 Rational 类的 __mod__ 方法计算取模，然后以当前精度创建 Float 对象
            return Float(Rational.__mod__(Rational(self), other),
                         precision=self._prec)
        # 如果 other 是 Float 类型且全局参数 evaluate 为真
        if isinstance(other, Float) and global_parameters.evaluate:
            # 计算 self 除以 other 的结果 r
            r = self/other
            # 如果 r 是整数值，则返回精度为 self 和 other 最大的 Float 对象
            if int_valued(r):
                return Float(0, precision=max(self._prec, other._prec))
        # 如果 other 是 Number 类型且全局参数 evaluate 为真
        if isinstance(other, Number) and global_parameters.evaluate:
            # 将 other 转换为 MPF 操作数，使用当前对象的精度
            rhs, prec = other._as_mpf_op(self._prec)
            # 调用 mlib.mpf_mod 执行 MPF 取模操作，并返回新的 Float 对象
            return Float._new(mlib.mpf_mod(self._mpf_, rhs, prec, rnd), prec)
        # 否则调用 Number 类的 __mod__ 方法
        return Number.__mod__(self, other)

    # 应用 _sympifyit 装饰器，定义 __rmod__ 方法，实现反向取模操作
    @_sympifyit('other', NotImplemented)
    def __rmod__(self, other):
        # 如果 other 是 Float 类型且全局参数 evaluate 为真
        if isinstance(other, Float) and global_parameters.evaluate:
            # 调用 other 对象的 __mod__ 方法，并将 self 作为参数传入
            return other.__mod__(self)
        # 如果 other 是 Number 类型且全局参数 evaluate 为真
        if isinstance(other, Number) and global_parameters.evaluate:
            # 将 other 转换为 MPF 操作数，使用当前对象的精度
            rhs, prec = other._as_mpf_op(self._prec)
            # 调用 mlib.mpf_mod 执行 MPF 反向取模操作，并返回新的 Float 对象
            return Float._new(mlib.mpf_mod(rhs, self._mpf_, prec, rnd), prec)
        # 否则调用 Number 类的 __rmod__ 方法
        return Number.__rmod__(self, other)
    def _eval_power(self, expt):
        """
        expt is symbolic object but not equal to 0, 1

        (-p)**r -> exp(r*log(-p)) -> exp(r*(log(p) + I*Pi)) ->
                  -> p**r*(sin(Pi*r) + cos(Pi*r)*I)
        """
        # 检查当前对象是否与 0 相等
        if equal_valued(self, 0):
            # 如果指数 expt 是正数，返回自身
            if expt.is_extended_positive:
                return self
            # 如果指数 expt 是负数，返回复无穷
            if expt.is_extended_negative:
                return S.ComplexInfinity
        # 如果 expt 是数字类型
        if isinstance(expt, Number):
            # 如果 expt 是整数
            if isinstance(expt, Integer):
                # 使用精度 self._prec，计算 self 的整数次幂并返回浮点数结果
                prec = self._prec
                return Float._new(
                    mlib.mpf_pow_int(self._mpf_, expt.p, prec, rnd), prec)
            # 如果 expt 是有理数，并且其分子为 1，分母为奇数，并且 self 是负数
            elif isinstance(expt, Rational) and \
                    expt.p == 1 and expt.q % 2 and self.is_negative:
                # 返回复数幂 (-1)**expt * (-self)**expt 的结果
                return Pow(S.NegativeOne, expt, evaluate=False)*(
                    -self)._eval_power(expt)
            # 将 expt 转换为多精度浮点数操作，使用指定精度 self._prec
            expt, prec = expt._as_mpf_op(self._prec)
            mpfself = self._mpf_
            try:
                # 计算 self 的 expt 次幂并返回浮点数结果
                y = mpf_pow(mpfself, expt, prec, rnd)
                return Float._new(y, prec)
            except mlib.ComplexResult:
                # 如果计算结果为复数，计算其实部和虚部的幂并返回复数结果
                re, im = mlib.mpc_pow(
                    (mpfself, fzero), (expt, fzero), prec, rnd)
                return Float._new(re, prec) + \
                    Float._new(im, prec)*S.ImaginaryUnit

    def __abs__(self):
        # 返回当前对象的绝对值的浮点数表示
        return Float._new(mlib.mpf_abs(self._mpf_), self._prec)

    def __int__(self):
        # 如果当前对象的多精度浮点数值为零，返回整数 0
        if self._mpf_ == fzero:
            return 0
        # 否则返回当前对象的多精度浮点数值转换为整数的结果
        return int(mlib.to_int(self._mpf_))  # 使用 round_fast = round_down

    def __eq__(self, other):
        # 如果 other 是 float 类型，转换为 Float 对象
        if isinstance(other, float):
            other = Float(other)
        # 调用基类的相等性比较方法判断当前对象是否等于 other
        return Basic.__eq__(self, other)

    def __ne__(self, other):
        # 调用 __eq__ 方法判断当前对象是否不等于 other
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return eq
        else:
            return not eq

    def __hash__(self):
        # 将当前对象转换为浮点数值
        float_val = float(self)
        # 如果浮点数值不是无穷大，则返回其哈希值
        if not math.isinf(float_val):
            return hash(float_val)
        # 否则调用基类的哈希方法返回哈希值
        return Basic.__hash__(self)
    # 定义一个私有方法 `_Frel`，用于实现符号对象与其他对象之间的比较操作
    def _Frel(self, other, op):
        try:
            # 尝试将 `other` 转换为符号对象
            other = _sympify(other)
        except SympifyError:
            # 如果转换失败，返回 NotImplemented
            return NotImplemented
        
        # 如果 `other` 是有理数
        if other.is_Rational:
            # 计算 self * other.q <?> other.p，同时保持精度
            '''
            >>> f = Float(.1,2)
            >>> i = 1234567890
            >>> (f*i)._mpf_
            (0, 471, 18, 9)
            >>> mlib.mpf_mul(f._mpf_, mlib.from_int(i))
            (0, 505555550955, -12, 39)
            '''
            smpf = mlib.mpf_mul(self._mpf_, mlib.from_int(other.q))
            ompf = mlib.from_int(other.p)
            return _sympify(bool(op(smpf, ompf)))
        
        # 如果 `other` 是浮点数
        elif other.is_Float:
            return _sympify(bool(
                        op(self._mpf_, other._mpf_)))
        
        # 如果 `other` 可比较且不是无穷大
        elif other.is_comparable and other not in (
                S.Infinity, S.NegativeInfinity):
            # 将 `other` 转换为数值，精度为 `self._prec`
            other = other.evalf(prec_to_dps(self._prec))
            if other._prec > 1:
                # 如果 `other` 是数值
                if other.is_Number:
                    return _sympify(bool(
                        op(self._mpf_, other._as_mpf_val(self._prec))))
    
    # 实现符号对象与其他对象的大于比较操作
    def __gt__(self, other):
        if isinstance(other, NumberSymbol):
            return other.__lt__(self)
        rv = self._Frel(other, mlib.mpf_gt)
        if rv is None:
            return Expr.__gt__(self, other)
        return rv

    # 实现符号对象与其他对象的大于等于比较操作
    def __ge__(self, other):
        if isinstance(other, NumberSymbol):
            return other.__le__(self)
        rv = self._Frel(other, mlib.mpf_ge)
        if rv is None:
            return Expr.__ge__(self, other)
        return rv

    # 实现符号对象与其他对象的小于比较操作
    def __lt__(self, other):
        if isinstance(other, NumberSymbol):
            return other.__gt__(self)
        rv = self._Frel(other, mlib.mpf_lt)
        if rv is None:
            return Expr.__lt__(self, other)
        return rv

    # 实现符号对象与其他对象的小于等于比较操作
    def __le__(self, other):
        if isinstance(other, NumberSymbol):
            return other.__ge__(self)
        rv = self._Frel(other, mlib.mpf_le)
        if rv is None:
            return Expr.__le__(self, other)
        return rv

    # 实现符号对象与其他对象的近似相等比较操作
    def epsilon_eq(self, other, epsilon="1e-15"):
        return abs(self - other) < Float(epsilon)

    # 实现符号对象的格式化输出方法
    def __format__(self, format_spec):
        return format(decimal.Decimal(str(self)), format_spec)
# 将 float 和 decimal.Decimal 类型的对象映射到 Float 类型的 _sympy_converter 字典中
_sympy_converter[float] = _sympy_converter[decimal.Decimal] = Float

# 为了在 Sage 中能良好运行，将 Float 赋值给 RealNumber 变量
RealNumber = Float

# Rational 类，表示任意大小的有理数 p/q
class Rational(Number):
    """Represents rational numbers (p/q) of any size.

    Examples
    ========

    >>> from sympy import Rational, nsimplify, S, pi
    >>> Rational(1, 2)
    1/2

    Rational is unprejudiced in accepting input. If a float is passed, the
    underlying value of the binary representation will be returned:

    >>> Rational(.5)
    1/2
    >>> Rational(.2)
    3602879701896397/18014398509481984

    If the simpler representation of the float is desired then consider
    limiting the denominator to the desired value or convert the float to
    a string (which is roughly equivalent to limiting the denominator to
    10**12):

    >>> Rational(str(.2))
    1/5
    >>> Rational(.2).limit_denominator(10**12)
    1/5

    An arbitrarily precise Rational is obtained when a string literal is
    passed:

    >>> Rational("1.23")
    123/100
    >>> Rational('1e-2')
    1/100
    >>> Rational(".1")
    1/10
    >>> Rational('1e-2/3.2')
    1/320

    The conversion of other types of strings can be handled by
    the sympify() function, and conversion of floats to expressions
    or simple fractions can be handled with nsimplify:

    >>> S('.[3]')  # repeating digits in brackets
    1/3
    >>> S('3**2/10')  # general expressions
    9/10
    >>> nsimplify(.3)  # numbers that have a simple form
    3/10

    But if the input does not reduce to a literal Rational, an error will
    be raised:

    >>> Rational(pi)
    Traceback (most recent call last):
    ...
    TypeError: invalid input: pi


    Low-level
    ---------

    Access numerator and denominator as .p and .q:

    >>> r = Rational(3, 4)
    >>> r
    3/4
    >>> r.p
    3
    >>> r.q
    4

    Note that p and q return integers (not SymPy Integers) so some care
    is needed when using them in expressions:

    >>> r.p/r.q
    0.75

    If an unevaluated Rational is desired, ``gcd=1`` can be passed and
    this will keep common divisors of the numerator and denominator
    from being eliminated. It is not possible, however, to leave a
    negative value in the denominator.

    >>> Rational(2, 4, gcd=1)
    2/4
    >>> Rational(2, -4, gcd=1).q
    4

    See Also
    ========
    sympy.core.sympify.sympify, sympy.simplify.simplify.nsimplify
    """
    # 指示该类是实数、有理数和数字
    is_real = True
    is_integer = False
    is_rational = True
    is_number = True

    # 用于存储实例化对象的属性名列表
    __slots__ = ('p', 'q')

    # 定义属性 p 和 q，分别表示有理数的分子和分母，均为整数
    p: int
    q: int

    # 表示该类为有理数
    is_Rational = True

    # 使用缓存装饰器缓存计算结果
    @cacheit
    # 定义一个特殊方法 __new__，用于创建 Rational 对象
    def __new__(cls, p, q=None, gcd=None):
        # 如果 q 为 None
        if q is None:
            # 如果 p 是 Rational 类的实例，则直接返回 p
            if isinstance(p, Rational):
                return p
            
            # 如果 p 是 SYMPY_INTS 类型的实例
            if isinstance(p, SYMPY_INTS):
                pass
            else:
                # 如果 p 是 float 或 Float 类型
                if isinstance(p, (float, Float)):
                    # 将 p 转换为有理数并返回
                    return Rational(*_as_integer_ratio(p))

                # 如果 p 不是 str 类型
                if not isinstance(p, str):
                    try:
                        # 尝试用 sympify 转换 p
                        p = sympify(p)
                    except (SympifyError, SyntaxError):
                        pass  # 如果转换失败，则在下面会抛出错误
                else:
                    # 如果 p 中包含多个 '/'，则抛出类型错误
                    if p.count('/') > 1:
                        raise TypeError('invalid input: %s' % p)
                    # 去除 p 中的空格
                    p = p.replace(' ', '')
                    # 分割 p 中最后一个 '/' 前后的内容
                    pq = p.rsplit('/', 1)
                    if len(pq) == 2:
                        p, q = pq
                        # 将 p 和 q 转换为分数
                        fp = fractions.Fraction(p)
                        fq = fractions.Fraction(q)
                        p = fp/fq
                    try:
                        # 尝试将 p 转换为分数
                        p = fractions.Fraction(p)
                    except ValueError:
                        pass  # 如果转换失败，则在下面会抛出错误
                    else:
                        # 如果成功转换为分数，则返回 Rational 对象
                        return Rational(p.numerator, p.denominator, 1)

                # 如果 p 不是 Rational 类型，则抛出类型错误
                if not isinstance(p, Rational):
                    raise TypeError('invalid input: %s' % p)

            # 设置默认值 q 和 gcd
            q = 1
            gcd = 1
        
        # 初始化 Q 为 1
        Q = 1

        # 如果 p 不是 SYMPY_INTS 类型
        if not isinstance(p, SYMPY_INTS):
            # 将 p 转换为有理数并更新 Q
            p = Rational(p)
            Q *= p.q  # Q 乘以 p 的分母
            p = p.p   # p 更新为 p 的分子
        else:
            # 如果 p 是 SYMPY_INTS 类型，则将 p 转换为整数
            p = int(p)

        # 如果 q 不是 SYMPY_INTS 类型
        if not isinstance(q, SYMPY_INTS):
            # 将 q 转换为有理数并更新 p 和 Q
            q = Rational(q)
            p *= q.q  # p 乘以 q 的分母
            Q *= q.p  # Q 乘以 q 的分子
        else:
            # 如果 q 是 SYMPY_INTS 类型，则将 q 转换为整数并更新 Q
            Q *= int(q)
        
        # q 更新为 Q
        q = Q

        # 现在 p 和 q 都是整数
        # 如果 q 等于 0
        if q == 0:
            # 如果 p 等于 0
            if p == 0:
                # 如果 _errdict["divide"] 为 True，则抛出零除零的 ValueError
                if _errdict["divide"]:
                    raise ValueError("Indeterminate 0/0")
                else:
                    # 否则返回 S.NaN
                    return S.NaN
            # 返回 S.ComplexInfinity
            return S.ComplexInfinity
        
        # 如果 q 小于 0，则将 q 和 p 取反
        if q < 0:
            q = -q
            p = -p
        
        # 如果 gcd 未提供，则计算 p 的绝对值与 q 的最大公约数
        if not gcd:
            gcd = igcd(abs(p), q)
        
        # 如果最大公约数大于 1，则将 p 和 q 同时除以最大公约数
        if gcd > 1:
            p //= gcd
            q //= gcd
        
        # 如果 q 等于 1，则返回 Integer 类型的 p
        if q == 1:
            return Integer(p)
        
        # 如果 p 等于 1 且 q 等于 2，则返回 S.Half
        if p == 1 and q == 2:
            return S.Half
        
        # 创建 Expr 类的新实例 obj
        obj = Expr.__new__(cls)
        obj.p = p  # 设置 obj 的分子
        obj.q = q  # 设置 obj 的分母
        return obj  # 返回 obj

    # 定义一个方法 limit_denominator，用于获取最接近 self 的有理数，并限制分母不超过 max_denominator
    def limit_denominator(self, max_denominator=1000000):
        """Closest Rational to self with denominator at most max_denominator.

        Examples
        ========

        >>> from sympy import Rational
        >>> Rational('3.141592653589793').limit_denominator(10)
        22/7
        >>> Rational('3.141592653589793').limit_denominator(100)
        311/99

        """
        # 将 self 转换为分数对象 f
        f = fractions.Fraction(self.p, self.q)
        # 返回将 f 限制分母后的 Rational 对象
        return Rational(f.limit_denominator(fractions.Fraction(int(max_denominator))))

    # 定义一个特殊方法 __getnewargs__，返回包含 self.p 和 self.q 的元组
    def __getnewargs__(self):
        return (self.p, self.q)

    # 定义一个方法 _hashable_content，返回包含 self.p 和 self.q 的元组
    def _hashable_content(self):
        return (self.p, self.q)
    # 判断当前有理数是否大于零
    def _eval_is_positive(self):
        return self.p > 0

    # 判断当前有理数是否等于零
    def _eval_is_zero(self):
        return self.p == 0

    # 实现有理数的负数运算
    def __neg__(self):
        return Rational(-self.p, self.q)

    # 实现有理数与其他对象的加法运算
    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        # 如果全局参数允许评估
        if global_parameters.evaluate:
            # 如果其他对象是整数类型
            if isinstance(other, Integer):
                return Rational(self.p + self.q*other.p, self.q, 1)
            # 如果其他对象是有理数类型
            elif isinstance(other, Rational):
                # TODO: 可能可以进一步优化这部分代码
                return Rational(self.p*other.q + self.q*other.p, self.q*other.q)
            # 如果其他对象是浮点数类型
            elif isinstance(other, Float):
                return other + self
            # 其他类型则调用父类的加法运算
            else:
                return Number.__add__(self, other)
        # 如果不允许评估则调用父类的加法运算
        return Number.__add__(self, other)
    __radd__ = __add__

    # 实现有理数与其他对象的减法运算
    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        # 如果全局参数允许评估
        if global_parameters.evaluate:
            # 如果其他对象是整数类型
            if isinstance(other, Integer):
                return Rational(self.p - self.q*other.p, self.q, 1)
            # 如果其他对象是有理数类型
            elif isinstance(other, Rational):
                return Rational(self.p*other.q - self.q*other.p, self.q*other.q)
            # 如果其他对象是浮点数类型
            elif isinstance(other, Float):
                return -other + self
            # 其他类型则调用父类的减法运算
            else:
                return Number.__sub__(self, other)
        # 如果不允许评估则调用父类的减法运算
        return Number.__sub__(self, other)
    
    # 实现其他对象减去有理数的反向运算
    @_sympifyit('other', NotImplemented)
    def __rsub__(self, other):
        # 如果全局参数允许评估
        if global_parameters.evaluate:
            # 如果其他对象是整数类型
            if isinstance(other, Integer):
                return Rational(self.q*other.p - self.p, self.q, 1)
            # 如果其他对象是有理数类型
            elif isinstance(other, Rational):
                return Rational(self.q*other.p - self.p*other.q, self.q*other.q)
            # 如果其他对象是浮点数类型
            elif isinstance(other, Float):
                return -self + other
            # 其他类型则调用父类的反向减法运算
            else:
                return Number.__rsub__(self, other)
        # 如果不允许评估则调用父类的反向减法运算
        return Number.__rsub__(self, other)
    
    # 实现有理数与其他对象的乘法运算
    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        # 如果全局参数允许评估
        if global_parameters.evaluate:
            # 如果其他对象是整数类型
            if isinstance(other, Integer):
                return Rational(self.p*other.p, self.q, igcd(other.p, self.q))
            # 如果其他对象是有理数类型
            elif isinstance(other, Rational):
                return Rational(self.p*other.p, self.q*other.q, igcd(self.p, other.q)*igcd(self.q, other.p))
            # 如果其他对象是浮点数类型
            elif isinstance(other, Float):
                return other*self
            # 其他类型则调用父类的乘法运算
            else:
                return Number.__mul__(self, other)
        # 如果不允许评估则调用父类的乘法运算
        return Number.__mul__(self, other)
    __rmul__ = __mul__

    # 实现有理数与其他对象的除法运算
    @_sympifyit('other', NotImplemented)
    # 定义对象的真除操作符重载方法
    def __truediv__(self, other):
        # 如果全局参数 evaluate 为真，则执行以下逻辑
        if global_parameters.evaluate:
            # 如果 other 是 Integer 类型
            if isinstance(other, Integer):
                # 如果 self.p 存在且 other.p 等于零，则返回正无穷
                if self.p and other.p == S.Zero:
                    return S.ComplexInfinity
                else:
                    # 否则返回有理数，分子为 self.p，分母为 self.q*other.p，并计算它们的最大公约数
                    return Rational(self.p, self.q*other.p, igcd(self.p, other.p))
            # 如果 other 是 Rational 类型
            elif isinstance(other, Rational):
                # 返回有理数，分子为 self.p*other.q，分母为 self.q*other.p，并计算它们的最大公约数
                return Rational(self.p*other.q, self.q*other.p, igcd(self.p, other.p)*igcd(self.q, other.q))
            # 如果 other 是 Float 类型
            elif isinstance(other, Float):
                # 返回 self 乘以 1/other 的结果
                return self*(1/other)
            # 其他情况，调用 Number 类的真除操作符方法
            else:
                return Number.__truediv__(self, other)
        # 如果全局参数 evaluate 不为真，则调用 Number 类的真除操作符方法
        return Number.__truediv__(self, other)

    # 使用装饰器 _sympifyit 将 other 转换为符号类型，如果无法转换则返回 NotImplemented
    @_sympifyit('other', NotImplemented)
    # 定义对象的反向真除操作符重载方法
    def __rtruediv__(self, other):
        # 如果全局参数 evaluate 为真，则执行以下逻辑
        if global_parameters.evaluate:
            # 如果 other 是 Integer 类型
            if isinstance(other, Integer):
                # 返回有理数，分子为 other.p*self.q，分母为 self.p，并计算它们的最大公约数
                return Rational(other.p*self.q, self.p, igcd(self.p, other.p))
            # 如果 other 是 Rational 类型
            elif isinstance(other, Rational):
                # 返回有理数，分子为 other.p*self.q，分母为 other.q*self.p，并计算它们的最大公约数
                return Rational(other.p*self.q, other.q*self.p, igcd(self.p, other.p)*igcd(self.q, other.q))
            # 如果 other 是 Float 类型
            elif isinstance(other, Float):
                # 返回 other 乘以 1/self 的结果
                return other*(1/self)
            # 其他情况，调用 Number 类的反向真除操作符方法
            else:
                return Number.__rtruediv__(self, other)
        # 如果全局参数 evaluate 不为真，则调用 Number 类的反向真除操作符方法
        return Number.__rtruediv__(self, other)

    # 使用装饰器 _sympifyit 将 other 转换为符号类型，如果无法转换则返回 NotImplemented
    @_sympifyit('other', NotImplemented)
    # 定义对象的取模操作符重载方法
    def __mod__(self, other):
        # 如果全局参数 evaluate 为真，则执行以下逻辑
        if global_parameters.evaluate:
            # 如果 other 是 Rational 类型
            if isinstance(other, Rational):
                # 计算取模操作，分子为 self.p*other.q - n*other.p*self.q，分母为 self.q*other.q
                n = (self.p*other.q) // (other.p*self.q)
                return Rational(self.p*other.q - n*other.p*self.q, self.q*other.q)
            # 如果 other 是 Float 类型
            if isinstance(other, Float):
                # 使用 Rational 类的取模操作符方法计算取模，然后使用 other 的精度来四舍五入结果
                return Float(self.__mod__(Rational(other)),
                             precision=other._prec)
            # 其他情况，调用 Number 类的取模操作符方法
            return Number.__mod__(self, other)
        # 如果全局参数 evaluate 不为真，则调用 Number 类的取模操作符方法
        return Number.__mod__(self, other)

    # 使用装饰器 _sympifyit 将 other 转换为符号类型，如果无法转换则返回 NotImplemented
    @_sympifyit('other', NotImplemented)
    # 定义对象的反向取模操作符重载方法
    def __rmod__(self, other):
        # 如果 other 是 Rational 类型
        if isinstance(other, Rational):
            # 调用 Rational 类的反向取模操作符方法
            return Rational.__mod__(other, self)
        # 其他情况，调用 Number 类的反向取模操作符方法
        return Number.__rmod__(self, other)
    # 定义用于计算幂操作的方法，接受一个指数参数 expt
    def _eval_power(self, expt):
        # 检查指数 expt 是否是 Number 类型
        if isinstance(expt, Number):
            # 如果 expt 是 Float 类型
            if isinstance(expt, Float):
                # 调用 _eval_evalf 方法计算浮点精度后的自身的 expt 次方
                return self._eval_evalf(expt._prec) ** expt
            # 如果 expt 是负数
            if expt.is_extended_negative:
                # (3/4)**-2 -> (4/3)**2 的特殊情况处理
                ne = -expt
                # 如果 ne 是 S.One
                if (ne is S.One):
                    # 返回有理数 self.q/self.p
                    return Rational(self.q, self.p)
                # 如果 self 是负数
                if self.is_negative:
                    # 返回 S.NegativeOne**expt * (self.q/-self.p)**ne
                    return S.NegativeOne ** expt * Rational(self.q, -self.p) ** ne
                else:
                    # 返回 self.q/self.p 的 ne 次方
                    return Rational(self.q, self.p) ** ne
            # 如果 expt 是正无穷
            if expt is S.Infinity:  # -oo 已被负数检查捕获
                # 如果 self.p > self.q
                if self.p > self.q:
                    # 返回正无穷 S.Infinity
                    return S.Infinity
                # 如果 self.p < -self.q
                if self.p < -self.q:
                    # 返回正无穷加上虚数单位乘以正无穷 S.Infinity + S.Infinity * S.ImaginaryUnit
                    return S.Infinity + S.Infinity * S.ImaginaryUnit
                # 返回零 S.Zero
                return S.Zero
            # 如果 expt 是整数类型
            if isinstance(expt, Integer):
                # 处理 (4/3)**2 -> 4**2 / 3**2 的情况
                return Rational(self.p ** expt.p, self.q ** expt.p, 1)
            # 如果 expt 是有理数类型
            if isinstance(expt, Rational):
                # 计算有理数 expt.p // expt.q 的整数部分
                intpart = expt.p // expt.q
                # 如果 intpart 不为零
                if intpart:
                    # 将 intpart 加一
                    intpart += 1
                    # 计算余数部分 remfracpart
                    remfracpart = intpart * expt.q - expt.p
                    # 创建有理数 ratfracpart
                    ratfracpart = Rational(remfracpart, expt.q)
                    # 如果 self.p 不等于 1
                    if self.p != 1:
                        # 返回 Integer(self.p)**expt * Integer(self.q)**ratfracpart * Rational(1, self.q**intpart, 1)
                        return Integer(self.p) ** expt * Integer(self.q) ** ratfracpart * Rational(1, self.q ** intpart, 1)
                    # 返回 Integer(self.q)**ratfracpart * Rational(1, self.q**intpart, 1)
                    return Integer(self.q) ** ratfracpart * Rational(1, self.q ** intpart, 1)
                else:
                    # 计算余数部分 remfracpart
                    remfracpart = expt.q - expt.p
                    # 创建有理数 ratfracpart
                    ratfracpart = Rational(remfracpart, expt.q)
                    # 如果 self.p 不等于 1
                    if self.p != 1:
                        # 返回 Integer(self.p)**expt * Integer(self.q)**ratfracpart * Rational(1, self.q, 1)
                        return Integer(self.p) ** expt * Integer(self.q) ** ratfracpart * Rational(1, self.q, 1)
                    # 返回 Integer(self.q)**ratfracpart * Rational(1, self.q, 1)
                    return Integer(self.q) ** ratfracpart * Rational(1, self.q, 1)

        # 如果 self 是负数并且 expt 是偶数
        if self.is_extended_negative and expt.is_even:
            # 返回 (-self)**expt
            return (-self) ** expt

        # 默认返回空值
        return

    # 将 Rational 转换为浮点数值，精度为 prec
    def _as_mpf_val(self, prec):
        return mlib.from_rational(self.p, self.q, prec, rnd)

    # 使用 mpmath 将 Rational 转换为浮点数
    def _mpmath_(self, prec, rnd):
        return mpmath.make_mpf(mlib.from_rational(self.p, self.q, prec, rnd))

    # 返回有理数的绝对值
    def __abs__(self):
        return Rational(abs(self.p), self.q)

    # 将有理数转换为整数
    def __int__(self):
        # 分别获取 p 和 q
        p, q = self.p, self.q
        # 如果 p 小于 0
        if p < 0:
            # 返回 -int(-p//q)
            return -int(-p // q)
        # 返回 int(p//q)
        return int(p // q)

    # 返回有理数的下取整
    def floor(self):
        return Integer(self.p // self.q)

    # 返回有理数的上取整
    def ceiling(self):
        return -Integer(-self.p // self.q)

    # 将有理数转换为下取整
    def __floor__(self):
        return self.floor()

    # 将有理数转换为上取整
    def __ceil__(self):
        return self.ceiling()
    def __eq__(self, other):
        try:
            # 尝试将参数转换为符号表达式
            other = _sympify(other)
        except SympifyError:
            # 如果转换失败，则返回NotImplemented
            return NotImplemented
        # 如果other不是Number类型
        if not isinstance(other, Number):
            # 若self为非零数则返回False，否则根据S(0) == S.false或S(0) == False返回True或False
            return False
        # 若self为零则根据other是否为非零数返回True或False
        if not self:
            return not other
        # 若other为数学符号类型
        if other.is_NumberSymbol:
            # 若other为无理数则返回False，否则比较other与self的等价性
            if other.is_irrational:
                return False
            return other.__eq__(self)
        # 若other为有理数类型
        if other.is_Rational:
            # 有理数总是以简化形式存在，所以直接比较其分子和分母是否相等
            return self.p == other.p and self.q == other.q
        # 其他情况返回False
        return False

    def __ne__(self, other):
        # 返回非相等的结果
        return not self == other

    def _Rrel(self, other, attr):
        # 如果想比较self < other，传入self, other, __gt__
        try:
            # 尝试将参数转换为符号表达式
            other = _sympify(other)
        except SympifyError:
            # 如果转换失败，则返回NotImplemented
            return NotImplemented
        # 若other为数值类型
        if other.is_Number:
            op = None
            s, o = self, other
            # 如果other是数学符号类型或浮点数类型或有理数类型，分别获取相应的比较操作符
            if other.is_NumberSymbol:
                op = getattr(o, attr)
            elif other.is_Float:
                op = getattr(o, attr)
            elif other.is_Rational:
                s, o = Integer(s.p*o.q), Integer(s.q*o.p)
                op = getattr(o, attr)
            # 若获取到操作符，则执行操作
            if op:
                return op(s)
            # 如果o是数值并且是扩展实数，则返回Integer(s.p), s.q*o
            if o.is_number and o.is_extended_real:
                return Integer(s.p), s.q*o

    def __gt__(self, other):
        # 比较self是否大于other
        rv = self._Rrel(other, '__lt__')
        # 如果结果为None，则返回self, other；否则返回Expr.__gt__(*rv)
        if rv is None:
            rv = self, other
        elif not isinstance(rv, tuple):
            return rv
        return Expr.__gt__(*rv)

    def __ge__(self, other):
        # 比较self是否大于或等于other
        rv = self._Rrel(other, '__le__')
        # 如果结果为None，则返回self, other；否则返回Expr.__ge__(*rv)
        if rv is None:
            rv = self, other
        elif not isinstance(rv, tuple):
            return rv
        return Expr.__ge__(*rv)

    def __lt__(self, other):
        # 比较self是否小于other
        rv = self._Rrel(other, '__gt__')
        # 如果结果为None，则返回self, other；否则返回Expr.__lt__(*rv)
        if rv is None:
            rv = self, other
        elif not isinstance(rv, tuple):
            return rv
        return Expr.__lt__(*rv)

    def __le__(self, other):
        # 比较self是否小于或等于other
        rv = self._Rrel(other, '__ge__')
        # 如果结果为None，则返回self, other；否则返回Expr.__le__(*rv)
        if rv is None:
            rv = self, other
        elif not isinstance(rv, tuple):
            return rv
        return Expr.__le__(*rv)

    def __hash__(self):
        # 返回基类的哈希值
        return super().__hash__()

    def factors(self, limit=None, use_trial=True, use_rho=False,
                use_pm1=False, verbose=False, visual=False):
        """A wrapper to factorint which return factors of self that are
        smaller than limit (or cheap to compute). Special methods of
        factoring are disabled by default so that only trial division is used.
        """
        # 导入factorrat函数并调用，返回self的因子
        from sympy.ntheory.factor_ import factorrat

        return factorrat(self, limit=limit, use_trial=use_trial,
                      use_rho=use_rho, use_pm1=use_pm1,
                      verbose=verbose).copy()
    # 返回分数的分子属性
    @property
    def numerator(self):
        return self.p

    # 返回分数的分母属性
    @property
    def denominator(self):
        return self.q

    # 计算当前有理数与另一个数的最大公约数
    @_sympifyit('other', NotImplemented)
    def gcd(self, other):
        if isinstance(other, Rational):
            # 如果另一个数是有理数且为零，则返回它自身
            if other == S.Zero:
                return other
            # 返回当前有理数与另一个有理数的最大公约数构成的新有理数
            return Rational(
                igcd(self.p, other.p),
                ilcm(self.q, other.q))
        # 如果另一个数不是有理数，则调用父类 Number 的 gcd 方法处理
        return Number.gcd(self, other)

    # 计算当前有理数与另一个数的最小公倍数
    @_sympifyit('other', NotImplemented)
    def lcm(self, other):
        if isinstance(other, Rational):
            # 返回当前有理数与另一个有理数的最小公倍数构成的新有理数
            return Rational(
                self.p // igcd(self.p, other.p) * other.p,
                igcd(self.q, other.q))
        # 如果另一个数不是有理数，则调用父类 Number 的 lcm 方法处理
        return Number.lcm(self, other)

    # 将当前有理数表示为分子和分母的整数形式
    def as_numer_denom(self):
        return Integer(self.p), Integer(self.q)

    # 将当前有理数提取为它的内容和原始值的元组
    def as_content_primitive(self, radical=False, clear=True):
        """Return the tuple (R, self/R) where R is the positive Rational
        extracted from self.

        Examples
        ========

        >>> from sympy import S
        >>> (S(-3)/2).as_content_primitive()
        (3/2, -1)

        See docstring of Expr.as_content_primitive for more examples.
        """
        # 如果当前有理数不是零
        if self:
            # 如果当前有理数为正数，则返回它自身和 1 的元组
            if self.is_positive:
                return self, S.One
            # 如果当前有理数为负数，则返回它的绝对值和 -1 的元组
            return -self, S.NegativeOne
        # 如果当前有理数为零，则返回 1 和它自身的元组
        return S.One, self

    # 从乘法表达式中高效提取当前有理数的系数
    def as_coeff_Mul(self, rational=False):
        """Efficiently extract the coefficient of a product."""
        return self, S.One

    # 从加法表达式中高效提取当前有理数的系数
    def as_coeff_Add(self, rational=False):
        """Efficiently extract the coefficient of a summation."""
        return self, S.Zero
# Integer 类继承自 Rational 类，表示任意大小的整数。

class Integer(Rational):
    """Represents integer numbers of any size.
    
    Examples
    ========
    
    >>> from sympy import Integer
    >>> Integer(3)
    3
    
    If a float or a rational is passed to Integer, the fractional part
    will be discarded; the effect is of rounding toward zero.
    
    >>> Integer(3.8)
    3
    >>> Integer(-3.8)
    -3
    
    A string is acceptable input if it can be parsed as an integer:
    
    >>> Integer("9" * 20)
    99999999999999999999
    
    It is rarely needed to explicitly instantiate an Integer, because
    Python integers are automatically converted to Integer when they
    are used in SymPy expressions.
    """

    q = 1  # 设置 q 为 1
    is_integer = True  # 声明 is_integer 为 True
    is_number = True  # 声明 is_number 为 True

    is_Integer = True  # 声明 is_Integer 为 True

    __slots__ = ()  # 使用空元组作为 __slots__，节省对象内存

    def _as_mpf_val(self, prec):
        # 返回精度为 prec 的多精度浮点数值
        return mlib.from_int(self.p, prec, rnd)

    def _mpmath_(self, prec, rnd):
        # 将整数转换为 mpmath 中的多精度浮点数
        return mpmath.make_mpf(self._as_mpf_val(prec))

    @cacheit
    def __new__(cls, i):
        if isinstance(i, str):
            i = i.replace(' ', '')  # 如果 i 是字符串，去除空格
        # 尝试将 i 转换为整数类型，如果失败则抛出 TypeError
        try:
            ival = int(i)
        except TypeError:
            raise TypeError(
                "Argument of Integer should be of numeric type, got %s." % i)
        # 只处理良好行为的整数类型，例如将 numpy.int32 实例转换
        if ival == 1:
            return S.One  # 返回 SymPy 中的 S.One，表示整数 1
        if ival == -1:
            return S.NegativeOne  # 返回 SymPy 中的 S.NegativeOne，表示整数 -1
        if ival == 0:
            return S.Zero  # 返回 SymPy 中的 S.Zero，表示整数 0
        obj = Expr.__new__(cls)  # 使用 Expr 类的 __new__ 方法创建对象
        obj.p = ival  # 将整数值赋给对象的属性 p
        return obj

    def __getnewargs__(self):
        return (self.p,)  # 返回包含整数值的元组，用于对象的深复制

    # Arithmetic operations are here for efficiency

    def __int__(self):
        return self.p  # 返回对象的整数值

    def floor(self):
        return Integer(self.p)  # 返回一个新的 Integer 对象，表示向下取整的结果

    def ceiling(self):
        return Integer(self.p)  # 返回一个新的 Integer 对象，表示向上取整的结果

    def __floor__(self):
        return self.floor()  # 向下取整的魔术方法实现

    def __ceil__(self):
        return self.ceiling()  # 向上取整的魔术方法实现

    def __neg__(self):
        return Integer(-self.p)  # 返回当前对象相反数的 Integer 对象

    def __abs__(self):
        if self.p >= 0:
            return self  # 如果对象的整数值大于等于 0，则返回对象本身
        else:
            return Integer(-self.p)  # 否则返回当前对象的绝对值的 Integer 对象

    def __divmod__(self, other):
        if isinstance(other, Integer) and global_parameters.evaluate:
            return Tuple(*(divmod(self.p, other.p)))  # 如果 other 是 Integer 类型且 evaluate 全局参数为 True，则返回整数除法的商和余数的元组
        else:
            return Number.__divmod__(self, other)  # 否则调用 Number 类的 __divmod__ 方法处理
    # 定义一个特殊方法，用于实现右除法运算
    def __rdivmod__(self, other):
        # 如果 `other` 是整数并且全局参数 `evaluate` 为真
        if isinstance(other, int) and global_parameters.evaluate:
            # 返回包含商和余数的元组
            return Tuple(*(divmod(other, self.p)))
        else:
            try:
                # 尝试将 `other` 转换为 `Number` 类型
                other = Number(other)
            except TypeError:
                # 如果转换失败，抛出类型错误，提示不支持的操作类型
                msg = "unsupported operand type(s) for divmod(): '%s' and '%s'"
                oname = type(other).__name__
                sname = type(self).__name__
                raise TypeError(msg % (oname, sname))
            # 否则调用 `Number` 类的 `__divmod__` 方法进行操作
            return Number.__divmod__(other, self)

    # TODO make it decorator + bytecodehacks?
    # 定义一个特殊方法，用于实现加法运算
    def __add__(self, other):
        # 如果全局参数 `evaluate` 为真
        if global_parameters.evaluate:
            # 根据 `other` 的类型进行不同的加法操作
            if isinstance(other, int):
                return Integer(self.p + other)
            elif isinstance(other, Integer):
                return Integer(self.p + other.p)
            elif isinstance(other, Rational):
                return Rational(self.p*other.q + other.p, other.q, 1)
            # 否则调用 `Rational` 类的 `__add__` 方法进行操作
            return Rational.__add__(self, other)
        else:
            # 如果 `evaluate` 为假，则返回 `Add` 类的实例
            return Add(self, other)

    # 定义一个特殊方法，用于实现右加法运算
    def __radd__(self, other):
        # 如果全局参数 `evaluate` 为真
        if global_parameters.evaluate:
            # 根据 `other` 的类型进行不同的加法操作
            if isinstance(other, int):
                return Integer(other + self.p)
            elif isinstance(other, Rational):
                return Rational(other.p + self.p*other.q, other.q, 1)
            # 否则调用 `Rational` 类的 `__radd__` 方法进行操作
            return Rational.__radd__(self, other)
        # 否则调用 `Rational` 类的 `__radd__` 方法进行操作
        return Rational.__radd__(self, other)

    # 定义一个特殊方法，用于实现减法运算
    def __sub__(self, other):
        # 如果全局参数 `evaluate` 为真
        if global_parameters.evaluate:
            # 根据 `other` 的类型进行不同的减法操作
            if isinstance(other, int):
                return Integer(self.p - other)
            elif isinstance(other, Integer):
                return Integer(self.p - other.p)
            elif isinstance(other, Rational):
                return Rational(self.p*other.q - other.p, other.q, 1)
            # 否则调用 `Rational` 类的 `__sub__` 方法进行操作
            return Rational.__sub__(self, other)
        # 否则调用 `Rational` 类的 `__sub__` 方法进行操作
        return Rational.__sub__(self, other)

    # 定义一个特殊方法，用于实现右减法运算
    def __rsub__(self, other):
        # 如果全局参数 `evaluate` 为真
        if global_parameters.evaluate:
            # 根据 `other` 的类型进行不同的减法操作
            if isinstance(other, int):
                return Integer(other - self.p)
            elif isinstance(other, Rational):
                return Rational(other.p - self.p*other.q, other.q, 1)
            # 否则调用 `Rational` 类的 `__rsub__` 方法进行操作
            return Rational.__rsub__(self, other)
        # 否则调用 `Rational` 类的 `__rsub__` 方法进行操作
        return Rational.__rsub__(self, other)

    # 定义一个特殊方法，用于实现乘法运算
    def __mul__(self, other):
        # 如果全局参数 `evaluate` 为真
        if global_parameters.evaluate:
            # 根据 `other` 的类型进行不同的乘法操作
            if isinstance(other, int):
                return Integer(self.p*other)
            elif isinstance(other, Integer):
                return Integer(self.p*other.p)
            elif isinstance(other, Rational):
                return Rational(self.p*other.p, other.q, igcd(self.p, other.q))
            # 否则调用 `Rational` 类的 `__mul__` 方法进行操作
            return Rational.__mul__(self, other)
        # 否则调用 `Rational` 类的 `__mul__` 方法进行操作
        return Rational.__mul__(self, other)
    # 定义自定义类的乘法右运算符重载函数
    def __rmul__(self, other):
        # 检查全局参数 evaluate 是否为真
        if global_parameters.evaluate:
            # 如果 other 是整数，则返回 Integer 类型结果
            if isinstance(other, int):
                return Integer(other*self.p)
            # 如果 other 是有理数 Rational，则返回相应的有理数结果
            elif isinstance(other, Rational):
                return Rational(other.p*self.p, other.q, igcd(self.p, other.q))
            # 对于其他类型，调用 Rational 类的 __rmul__ 方法处理
            return Rational.__rmul__(self, other)
        # 如果 evaluate 不为真，则调用 Rational 类的 __rmul__ 方法处理
        return Rational.__rmul__(self, other)

    # 定义自定义类的取模运算符重载函数
    def __mod__(self, other):
        # 检查全局参数 evaluate 是否为真
        if global_parameters.evaluate:
            # 如果 other 是整数，则返回 Integer 类型结果
            if isinstance(other, int):
                return Integer(self.p % other)
            # 如果 other 是 Integer 类型，则返回 Integer 类型结果
            elif isinstance(other, Integer):
                return Integer(self.p % other.p)
            # 对于其他类型，调用 Rational 类的 __mod__ 方法处理
            return Rational.__mod__(self, other)
        # 如果 evaluate 不为真，则调用 Rational 类的 __mod__ 方法处理
        return Rational.__mod__(self, other)

    # 定义自定义类的反向取模运算符重载函数
    def __rmod__(self, other):
        # 检查全局参数 evaluate 是否为真
        if global_parameters.evaluate:
            # 如果 other 是整数，则返回 Integer 类型结果
            if isinstance(other, int):
                return Integer(other % self.p)
            # 如果 other 是 Integer 类型，则返回 Integer 类型结果
            elif isinstance(other, Integer):
                return Integer(other.p % self.p)
            # 对于其他类型，调用 Rational 类的 __rmod__ 方法处理
            return Rational.__rmod__(self, other)
        # 如果 evaluate 不为真，则调用 Rational 类的 __rmod__ 方法处理
        return Rational.__rmod__(self, other)

    # 定义自定义类的相等比较运算符重载函数
    def __eq__(self, other):
        # 如果 other 是整数，则比较 self.p 和 other
        if isinstance(other, int):
            return (self.p == other)
        # 如果 other 是 Integer 类型，则比较 self.p 和 other.p
        elif isinstance(other, Integer):
            return (self.p == other.p)
        # 对于其他类型，调用 Rational 类的 __eq__ 方法处理
        return Rational.__eq__(self, other)

    # 定义自定义类的不等比较运算符重载函数
    def __ne__(self, other):
        # 返回相反的相等比较结果
        return not self == other

    # 定义自定义类的大于比较运算符重载函数
    def __gt__(self, other):
        try:
            # 尝试将 other 转换为 sympy 对象
            other = _sympify(other)
        except SympifyError:
            # 转换失败时返回 NotImplemented
            return NotImplemented
        # 如果 other 是整数，则比较 self.p 和 other.p 的大小
        if other.is_Integer:
            return _sympify(self.p > other.p)
        # 对于其他情况，调用 Rational 类的 __gt__ 方法处理
        return Rational.__gt__(self, other)

    # 定义自定义类的小于比较运算符重载函数
    def __lt__(self, other):
        try:
            # 尝试将 other 转换为 sympy 对象
            other = _sympify(other)
        except SympifyError:
            # 转换失败时返回 NotImplemented
            return NotImplemented
        # 如果 other 是整数，则比较 self.p 和 other.p 的大小
        if other.is_Integer:
            return _sympify(self.p < other.p)
        # 对于其他情况，调用 Rational 类的 __lt__ 方法处理
        return Rational.__lt__(self, other)

    # 定义自定义类的大于等于比较运算符重载函数
    def __ge__(self, other):
        try:
            # 尝试将 other 转换为 sympy 对象
            other = _sympify(other)
        except SympifyError:
            # 转换失败时返回 NotImplemented
            return NotImplemented
        # 如果 other 是整数，则比较 self.p 和 other.p 的大小
        if other.is_Integer:
            return _sympify(self.p >= other.p)
        # 对于其他情况，调用 Rational 类的 __ge__ 方法处理
        return Rational.__ge__(self, other)

    # 定义自定义类的小于等于比较运算符重载函数
    def __le__(self, other):
        try:
            # 尝试将 other 转换为 sympy 对象
            other = _sympify(other)
        except SympifyError:
            # 转换失败时返回 NotImplemented
            return NotImplemented
        # 如果 other 是整数，则比较 self.p 和 other.p 的大小
        if other.is_Integer:
            return _sympify(self.p <= other.p)
        # 对于其他情况，调用 Rational 类的 __le__ 方法处理
        return Rational.__le__(self, other)

    # 定义自定义类的哈希计算方法
    def __hash__(self):
        # 返回 self.p 的哈希值
        return hash(self.p)

    # 定义自定义类的索引方法
    def __index__(self):
        # 返回 self.p 的值
        return self.p

    ########################################

    # 定义自定义类的判断是否为奇数方法
    def _eval_is_odd(self):
        # 返回 self.p 是否为奇数的布尔值
        return bool(self.p % 2)

    # 定义自定义类的判断是否为素数方法
    def _eval_is_prime(self):
        # 导入 sympy.ntheory.primetest 模块中的 isprime 函数，并调用其判断是否为素数
        from sympy.ntheory.primetest import isprime
        return isprime(self)

    # 定义自定义类的判断是否为合数方法
    def _eval_is_composite(self):
        # 如果 self 大于 1，则返回 self.is_prime 的非模糊结果
        if self > 1:
            return fuzzy_not(self.is_prime)
        else:
            # 否则返回 False
            return False

    # 定义自定义类的返回自身与 1 的元组方法
    def as_numer_denom(self):
        return self, S.One

    # 使用 _sympifyit 装饰器定义的函数
    @_sympifyit('other', NotImplemented)
    # 实现整数除法的魔术方法，要求右侧操作数必须是 Expr 类的实例
    def __floordiv__(self, other):
        # 如果右侧操作数不是 Expr 类的实例，则返回 NotImplemented
        if not isinstance(other, Expr):
            return NotImplemented
        # 如果右侧操作数是 Integer 类的实例，执行整数除法操作
        if isinstance(other, Integer):
            return Integer(self.p // other)
        # 对于其他情况，使用 divmod 函数执行整数除法，返回结果的商
        return divmod(self, other)[0]

    # 实现右侧整数除法的魔术方法
    def __rfloordiv__(self, other):
        # 将右侧操作数转换为 Integer 类的实例，执行整数除法操作
        return Integer(Integer(other).p // self.p)

    # 下面的位运算操作 (__lshift__, __rlshift__, ..., __invert__) 仅为 Integer 类型定义，
    # 不支持一般的 SymPy 表达式。这是为了与 numbers.Integral ABC 兼容，后者仅在 numbers.Integral 的实例之间定义了这些操作。
    # 因此，这些方法明确检查整数类型，而不使用 sympify，因为它们不应接受任意的符号表达式，
    # 并且没有类似于 numbers.Integral 位运算的符号模拟。
    
    # 实现左移操作的魔术方法
    def __lshift__(self, other):
        # 如果右侧操作数是 int、Integer 或 numbers.Integral 类型，则执行左移操作
        if isinstance(other, (int, Integer, numbers.Integral)):
            return Integer(self.p << int(other))
        else:
            return NotImplemented

    # 实现右侧左移操作的魔术方法
    def __rlshift__(self, other):
        # 如果右侧操作数是 int 或 numbers.Integral 类型，则执行右侧左移操作
        if isinstance(other, (int, numbers.Integral)):
            return Integer(int(other) << self.p)
        else:
            return NotImplemented

    # 实现右移操作的魔术方法
    def __rshift__(self, other):
        # 如果右侧操作数是 int、Integer 或 numbers.Integral 类型，则执行右移操作
        if isinstance(other, (int, Integer, numbers.Integral)):
            return Integer(self.p >> int(other))
        else:
            return NotImplemented

    # 实现右侧右移操作的魔术方法
    def __rrshift__(self, other):
        # 如果右侧操作数是 int 或 numbers.Integral 类型，则执行右侧右移操作
        if isinstance(other, (int, numbers.Integral)):
            return Integer(int(other) >> self.p)
        else:
            return NotImplemented

    # 实现按位与操作的魔术方法
    def __and__(self, other):
        # 如果右侧操作数是 int、Integer 或 numbers.Integral 类型，则执行按位与操作
        if isinstance(other, (int, Integer, numbers.Integral)):
            return Integer(self.p & int(other))
        else:
            return NotImplemented

    # 实现右侧按位与操作的魔术方法
    def __rand__(self, other):
        # 如果右侧操作数是 int 或 numbers.Integral 类型，则执行右侧按位与操作
        if isinstance(other, (int, numbers.Integral)):
            return Integer(int(other) & self.p)
        else:
            return NotImplemented

    # 实现按位异或操作的魔术方法
    def __xor__(self, other):
        # 如果右侧操作数是 int、Integer 或 numbers.Integral 类型，则执行按位异或操作
        if isinstance(other, (int, Integer, numbers.Integral)):
            return Integer(self.p ^ int(other))
        else:
            return NotImplemented

    # 实现右侧按位异或操作的魔术方法
    def __rxor__(self, other):
        # 如果右侧操作数是 int 或 numbers.Integral 类型，则执行右侧按位异或操作
        if isinstance(other, (int, numbers.Integral)):
            return Integer(int(other) ^ self.p)
        else:
            return NotImplemented

    # 实现按位或操作的魔术方法
    def __or__(self, other):
        # 如果右侧操作数是 int、Integer 或 numbers.Integral 类型，则执行按位或操作
        if isinstance(other, (int, Integer, numbers.Integral)):
            return Integer(self.p | int(other))
        else:
            return NotImplemented

    # 实现右侧按位或操作的魔术方法
    def __ror__(self, other):
        # 如果右侧操作数是 int 或 numbers.Integral 类型，则执行右侧按位或操作
        if isinstance(other, (int, numbers.Integral)):
            return Integer(int(other) | self.p)
        else:
            return NotImplemented

    # 实现按位取反操作的魔术方法
    def __invert__(self):
        # 返回当前整数的按位取反结果
        return Integer(~self.p)
# 定义一个全局变量，将 int 类型映射到 sympy 的 Integer 类型
_sympy_converter[int] = Integer

# 表示代数数的类，在 SymPy 中用于表示代数数。
class AlgebraicNumber(Expr):
    """
    用于在 SymPy 中表示代数数的类。

    符号上，此类的实例表示一个元素 $\alpha \in \mathbb{Q}(\theta) \hookrightarrow \mathbb{C}$。
    即，代数数 $\alpha$ 被表示为特定数域 $\mathbb{Q}(\theta)$ 中的一个元素，这个数域中包含了一个
    特定的复数，它是一个多项式的根（定义了嵌入 $\mathbb{Q}(\theta) \hookrightarrow \mathbb{C}$）。

    形式上，原始元 $\theta$ 由两个数据点给出：（1）它的最小多项式（定义了 $\mathbb{Q}(\theta)$），
    和（2）一个特定的复数，它是这个多项式的一个根（定义了嵌入 $\mathbb{Q}(\theta) \hookrightarrow \mathbb{C}$）。
    最后，我们所表示的代数数 $\alpha$ 由 $\theta$ 的一个多项式的系数给出。
    """

    # 限制实例只能包含以下指定的属性，以节省内存
    __slots__ = ('rep', 'root', 'alias', 'minpoly', '_own_minpoly')

    # 标志位，指示这是一个 AlgebraicNumber 类的实例
    is_AlgebraicNumber = True
    # 标志位，指示这是一个代数数
    is_algebraic = True
    # 标志位，指示这是一个数字
    is_number = True

    # 表示数字类型的枚举值
    kind = NumberKind

    # 可选的别名符号不是自由的。
    # 实际上，alias 应该是一个 Str 类型，但某些方法期望它是 Expr 类的一个实例。
    # 这里定义一个空集合，用于存放自由符号集合
    free_symbols: set[Basic] = set()

    def __hash__(self):
        return super().__hash__()

    # 重写父类的 __hash__ 方法

    def _eval_evalf(self, prec):
        return self.as_expr()._evalf(prec)

    # 获取代数数的数值近似

    @property
    def is_aliased(self):
        """Returns ``True`` if ``alias`` was set. """
        return self.alias is not None

    # 检查是否设置了别名

    def as_poly(self, x=None):
        """Create a Poly instance from ``self``. """
        from sympy.polys.polytools import Poly, PurePoly
        if x is not None:
            return Poly.new(self.rep, x)
        else:
            if self.alias is not None:
                return Poly.new(self.rep, self.alias)
            else:
                from .symbol import Dummy
                return PurePoly.new(self.rep, Dummy('x'))

    # 将代数数转换为多项式

    def as_expr(self, x=None):
        """Create a Basic expression from ``self``. """
        return self.as_poly(x or self.root).as_expr().expand()

    # 将代数数转换为基本表达式

    def coeffs(self):
        """Returns all SymPy coefficients of an algebraic number. """
        return [ self.rep.dom.to_sympy(c) for c in self.rep.all_coeffs() ]

    # 返回代数数的所有 SymPy 系数

    def native_coeffs(self):
        """Returns all native coefficients of an algebraic number. """
        return self.rep.all_coeffs()

    # 返回代数数的所有本地系数

    def to_algebraic_integer(self):
        """Convert ``self`` to an algebraic integer. """
        from sympy.polys.polytools import Poly

        f = self.minpoly

        if f.LC() == 1:
            return self

        coeff = f.LC()**(f.degree() - 1)
        poly = f.compose(Poly(f.gen/f.LC()))

        minpoly = poly*coeff
        root = f.LC()*self.root

        return AlgebraicNumber((minpoly, root), self.coeffs())

    # 将代数数转换为代数整数
    # 定义一个方法用于简化表达式
    def _eval_simplify(self, **kwargs):
        # 从 sympy.polys.rootoftools 模块导入 CRootOf 类
        from sympy.polys.rootoftools import CRootOf
        # 从 sympy.polys 模块导入 minpoly 函数
        from sympy.polys import minpoly
        # 从 kwargs 参数中获取 measure 和 ratio
        measure, ratio = kwargs['measure'], kwargs['ratio']
        # 遍历 self.minpoly 的所有根，排除 CRootOf 类型的根
        for r in [r for r in self.minpoly.all_roots() if r.func != CRootOf]:
            # 如果 minpoly(self.root - r) 是 Symbol 类型
            if minpoly(self.root - r).is_Symbol:
                # 如果 r 比 self.root 更简单，根据比例 ratio 判断
                if measure(r) < ratio * measure(self.root):
                    # 返回一个新的代数数对象 AlgebraicNumber
                    return AlgebraicNumber(r)
        # 如果没有更简单的根，则返回自身
        return self

    # 定义一个方法用于生成同一数域中的另一个元素
    def field_element(self, coeffs):
        r"""
        形成同一数域中的另一个元素。

        Explanation
        ===========

        如果我们表示 $\alpha \in \mathbb{Q}(\theta)$，形成同一数域中的另一个元素
        $\beta \in \mathbb{Q}(\theta)$。

        Parameters
        ==========

        coeffs : list, :py:class:`~.ANP`
            像类 :py:meth:`constructor<.AlgebraicNumber.__new__>` 中的 *coeffs* 参数一样，
            用原始元素的多项式定义新元素。

            如果是列表，则元素应为整数或有理数。
            如果是 :py:class:`~.ANP`，我们使用它的系数（使用其 :py:meth:`~.ANP.to_list()` 方法）。

        Examples
        ========

        >>> from sympy import AlgebraicNumber, sqrt
        >>> a = AlgebraicNumber(sqrt(5), [-1, 1])
        >>> b = a.field_element([3, 2])
        >>> print(a)
        1 - sqrt(5)
        >>> print(b)
        2 + 3*sqrt(5)
        >>> print(b.primitive_element() == a.primitive_element())
        True

        See Also
        ========

        AlgebraicNumber
        """
        # 返回一个新的代数数对象 AlgebraicNumber
        return AlgebraicNumber(
            (self.minpoly, self.root), coeffs=coeffs, alias=self.alias)

    @property
    def is_primitive_element(self):
        r"""
        判断这个代数数 $\alpha \in \mathbb{Q}(\theta)$ 是否等于其数域的原始元素 $\theta$。
        """
        # 获取系数 c
        c = self.coeffs()
        # 如果 self.minpoly 是线性的，返回 True
        # 第二种情况发生在 self.minpoly 是线性的时候：
        return c == [1, 0] or c == [self.root]

    # 定义一个方法用于获取数域 $\mathbb{Q}(\theta)$ 的原始元素 $\theta$
    def primitive_element(self):
        r"""
        获取代数数 $\alpha$ 所属数域 $\mathbb{Q}(\theta)$ 的原始元素 $\theta$。

        Returns
        =======

        AlgebraicNumber

        """
        # 如果是原始元素，直接返回自身
        if self.is_primitive_element:
            return self
        # 否则生成一个新的同一数域中的另一个元素
        return self.field_element([1, 0])
    def to_primitive_element(self, radicals=True):
        r"""
        Convert ``self`` to an :py:class:`~.AlgebraicNumber` instance that is
        equal to its own primitive element.

        Explanation
        ===========

        If we represent $\alpha \in \mathbb{Q}(\theta)$, $\alpha \neq \theta$,
        construct a new :py:class:`~.AlgebraicNumber` that represents
        $\alpha \in \mathbb{Q}(\alpha)$.

        Examples
        ========

        >>> from sympy import sqrt, to_number_field
        >>> from sympy.abc import x
        >>> a = to_number_field(sqrt(2), sqrt(2) + sqrt(3))

        The :py:class:`~.AlgebraicNumber` ``a`` represents the number
        $\sqrt{2}$ in the field $\mathbb{Q}(\sqrt{2} + \sqrt{3})$. Rendering
        ``a`` as a polynomial,

        >>> a.as_poly().as_expr(x)
        x**3/2 - 9*x/2

        reflects the fact that $\sqrt{2} = \theta^3/2 - 9 \theta/2$, where
        $\theta = \sqrt{2} + \sqrt{3}$.

        ``a`` is not equal to its own primitive element. Its minpoly

        >>> a.minpoly.as_poly().as_expr(x)
        x**4 - 10*x**2 + 1

        is that of $\theta$.

        Converting to a primitive element,

        >>> a_prim = a.to_primitive_element()
        >>> a_prim.minpoly.as_poly().as_expr(x)
        x**2 - 2

        we obtain an :py:class:`~.AlgebraicNumber` whose ``minpoly`` is that of
        the number itself.

        Parameters
        ==========

        radicals : boolean, optional (default=True)
            If ``True``, then we will try to return an
            :py:class:`~.AlgebraicNumber` whose ``root`` is an expression
            in radicals. If that is not possible (or if *radicals* is
            ``False``), ``root`` will be a :py:class:`~.ComplexRootOf`.

        Returns
        =======

        AlgebraicNumber
            An algebraic number instance representing the primitive element.

        See Also
        ========

        is_primitive_element
            Method to check if the algebraic number is already a primitive element.

        """
        # 如果已经是原始元素，则直接返回自身
        if self.is_primitive_element:
            return self
        # 获取当前代数数的最小多项式
        m = self.minpoly_of_element()
        # 将当前代数数转换为对应的根
        r = self.to_root(radicals=radicals)
        # 返回一个新的代数数实例，表示为 (最小多项式, 根)
        return AlgebraicNumber((m, r))

    def minpoly_of_element(self):
        r"""
        Compute the minimal polynomial for this algebraic number.

        Explanation
        ===========

        Recall that we represent an element $\alpha \in \mathbb{Q}(\theta)$.
        Our instance attribute ``self.minpoly`` is the minimal polynomial for
        our primitive element $\theta$. This method computes the minimal
        polynomial for $\alpha$.

        """
        # 如果当前代数数的自身最小多项式尚未计算，则计算它
        if self._own_minpoly is None:
            # 如果当前代数数已经是一个原始元素，则使用其自身的最小多项式
            if self.is_primitive_element:
                self._own_minpoly = self.minpoly
            else:
                # 否则，计算其对应原始元素的最小多项式
                from sympy.polys.numberfields.minpoly import minpoly
                theta = self.primitive_element()
                self._own_minpoly = minpoly(self.as_expr(theta), polys=True)
        # 返回当前代数数的最小多项式
        return self._own_minpoly
    # 如果当前对象是原始元素并且其根不是代数数，则直接返回根对象
    if self.is_primitive_element and not isinstance(self.root, AlgebraicNumber):
        return self.root
    # 如果没有传入预先计算的最小多项式，则计算当前对象的最小多项式
    m = minpoly or self.minpoly_of_element()
    # 获取最小多项式的所有根
    roots = m.all_roots(radicals=radicals)
    # 如果只有一个根，则直接返回该根
    if len(roots) == 1:
        return roots[0]
    # 将当前对象转换为表达式
    ex = self.as_expr()
    # 遍历所有根
    for b in roots:
        # 如果某个根与当前对象的表达式相同，则返回该根
        if m.same_root(b, ex):
            return b
class RationalConstant(Rational):
    """
    Abstract base class for rationals with specific behaviors

    Derived classes must define class attributes p and q and should probably all
    be singletons.
    """
    __slots__ = ()

    def __new__(cls):
        # 创建一个新的对象实例，并返回
        return AtomicExpr.__new__(cls)


class IntegerConstant(Integer):
    __slots__ = ()

    def __new__(cls):
        # 创建一个新的对象实例，并返回
        return AtomicExpr.__new__(cls)


class Zero(IntegerConstant, metaclass=Singleton):
    """The number zero.

    Zero is a singleton, and can be accessed by ``S.Zero``

    Examples
    ========

    >>> from sympy import S, Integer
    >>> Integer(0) is S.Zero
    True
    >>> 1/S.Zero
    zoo

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Zero
    """

    p = 0
    q = 1
    is_positive = False
    is_negative = False
    is_zero = True
    is_number = True
    is_comparable = True

    __slots__ = ()

    def __getnewargs__(self):
        # 返回一个空元组，表示该对象实例的构造参数为空
        return ()

    @staticmethod
    def __abs__():
        # 返回单例对象 S.Zero 的绝对值，即 S.Zero 本身
        return S.Zero

    @staticmethod
    def __neg__():
        # 返回单例对象 S.Zero 的负值，即 S.Zero 本身
        return S.Zero

    def _eval_power(self, expt):
        # 根据指数 expt 的性质进行幂运算的求值
        if expt.is_extended_positive:
            return self
        if expt.is_extended_negative:
            return S.ComplexInfinity
        if expt.is_extended_real is False:
            return S.NaN
        if expt.is_zero:
            return S.One

        # 处理幂指数为乘法表达式的情况
        coeff, terms = expt.as_coeff_Mul()
        if coeff.is_negative:
            return S.ComplexInfinity**terms
        if coeff is not S.One:  # 如果存在要丢弃的数字
            return self**terms

    def _eval_order(self, *symbols):
        # 计算在给定变量（symbols）下的顺序，对于 Zero 总是返回自身
        return self

    def __bool__(self):
        # 将 Zero 转换为布尔值，总是返回 False
        return False


class One(IntegerConstant, metaclass=Singleton):
    """The number one.

    One is a singleton, and can be accessed by ``S.One``.

    Examples
    ========

    >>> from sympy import S, Integer
    >>> Integer(1) is S.One
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/1_%28number%29
    """
    is_number = True
    is_positive = True

    p = 1
    q = 1

    __slots__ = ()

    def __getnewargs__(self):
        # 返回一个空元组，表示该对象实例的构造参数为空
        return ()

    @staticmethod
    def __abs__():
        # 返回单例对象 S.One 的绝对值，即 S.One 本身
        return S.One

    @staticmethod
    def __neg__():
        # 返回单例对象 S.One 的负值，即 S.NegativeOne
        return S.NegativeOne

    def _eval_power(self, expt):
        # 根据指数 expt 的性质进行幂运算的求值，对于 One 总是返回自身
        return self

    def _eval_order(self, *symbols):
        # 计算在给定变量（symbols）下的顺序，对于 One 不返回任何值
        return

    @staticmethod
    def factors(limit=None, use_trial=True, use_rho=False, use_pm1=False,
                verbose=False, visual=False):
        # 根据参数返回因子，对于 visual=True 的情况返回 S.One，其他情况返回空字典
        if visual:
            return S.One
        else:
            return {}


class NegativeOne(IntegerConstant, metaclass=Singleton):
    """The number negative one.

    NegativeOne is a singleton, and can be accessed by ``S.NegativeOne``.

    Examples
    ========
    # 导入必要的符号计算库中的 S 和 Integer 对象
    >>> from sympy import S, Integer
    # 检查 Integer(-1) 是否与 S.NegativeOne 是同一对象
    >>> Integer(-1) is S.NegativeOne
    # 返回 True 表示它们是同一对象

    # 本类用于表示数字 -1
    """
    # 标记此对象为数值类型
    is_number = True

    # 初始化 p 和 q，分别为 -1 和 1
    p = -1
    q = 1

    # 定义一个空的特殊属性 __slots__，用于限制对象的属性
    __slots__ = ()

    # 返回空元组，用于支持对象的序列化和反序列化
    def __getnewargs__(self):
        return ()

    # 定义静态方法 __abs__，返回 S.One，表示绝对值
    @staticmethod
    def __abs__():
        return S.One

    # 定义静态方法 __neg__，返回 S.One，表示取负值
    @staticmethod
    def __neg__():
        return S.One

    # 定义方法 _eval_power，处理指数运算
    def _eval_power(self, expt):
        # 如果指数 expt 是奇数，返回 S.NegativeOne
        if expt.is_odd:
            return S.NegativeOne
        # 如果指数 expt 是偶数，返回 S.One
        if expt.is_even:
            return S.One
        # 如果 expt 是数字类型
        if isinstance(expt, Number):
            # 如果 expt 是浮点数类型
            if isinstance(expt, Float):
                return Float(-1.0)**expt
            # 如果 expt 是 S.NaN（不是一个数字）
            if expt is S.NaN:
                return S.NaN
            # 如果 expt 是正无穷或负无穷
            if expt in (S.Infinity, S.NegativeInfinity):
                return S.NaN
            # 如果 expt 是 S.Half（分数的一半）
            if expt is S.Half:
                return S.ImaginaryUnit
            # 如果 expt 是有理数类型
            if isinstance(expt, Rational):
                # 如果 expt 的分母是 2
                if expt.q == 2:
                    return S.ImaginaryUnit**Integer(expt.p)
                # 否则，使用指数 expt.p 和 expt.q 进行计算
                i, r = divmod(expt.p, expt.q)
                if i:
                    return self**i*self**Rational(r, expt.q)
        # 默认返回 None，表示未定义
        return
class Half(RationalConstant, metaclass=Singleton):
    """The rational number 1/2.

    Half is a singleton, and can be accessed by ``S.Half``.

    Examples
    ========

    >>> from sympy import S, Rational
    >>> Rational(1, 2) is S.Half
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/One_half
    """
    # 声明 Half 是一个数值常量
    is_number = True

    # 分子和分母分别为 1 和 2
    p = 1
    q = 2

    # 使用 __slots__ 机制，节省对象内存
    __slots__ = ()

    # 返回空元组，用于创建新实例时的参数
    def __getnewargs__(self):
        return ()

    # 返回 S.Half，即 Half 的绝对值
    @staticmethod
    def __abs__():
        return S.Half


class Infinity(Number, metaclass=Singleton):
    r"""Positive infinite quantity.

    Explanation
    ===========

    In real analysis the symbol `\infty` denotes an unbounded
    limit: `x\to\infty` means that `x` grows without bound.

    Infinity is often used not only to define a limit but as a value
    in the affinely extended real number system.  Points labeled `+\infty`
    and `-\infty` can be added to the topological space of the real numbers,
    producing the two-point compactification of the real numbers.  Adding
    algebraic properties to this gives us the extended real numbers.

    Infinity is a singleton, and can be accessed by ``S.Infinity``,
    or can be imported as ``oo``.

    Examples
    ========

    >>> from sympy import oo, exp, limit, Symbol
    >>> 1 + oo
    oo
    >>> 42/oo
    0
    >>> x = Symbol('x')
    >>> limit(exp(x), x, oo)
    oo

    See Also
    ========

    NegativeInfinity, NaN

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Infinity
    """

    # 定义 Infinity 的数学特性
    is_commutative = True
    is_number = True
    is_complex = False
    is_extended_real = True
    is_infinite = True
    is_comparable = True
    is_extended_positive = True
    is_prime = False

    # 使用 __slots__ 机制，节省对象内存
    __slots__ = ()

    # 创建新实例时调用 AtomicExpr.__new__
    def __new__(cls):
        return AtomicExpr.__new__(cls)

    # 返回 LaTeX 表示为 \infty
    def _latex(self, printer):
        return r"\infty"

    # 替换表达式中的旧值为新值
    def _eval_subs(self, old, new):
        if self == old:
            return new

    # 返回浮点表示为正无穷
    def _eval_evalf(self, prec=None):
        return Float('inf')

    # 计算浮点表示，与 _eval_evalf 保持一致
    def evalf(self, prec=None, **options):
        return self._eval_evalf(prec)

    # 实现 Infinity 与数值相加的操作
    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other in (S.NegativeInfinity, S.NaN):
                return S.NaN
            return self
        return Number.__add__(self, other)
    __radd__ = __add__

    # 实现 Infinity 与数值相减的操作
    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other in (S.Infinity, S.NaN):
                return S.NaN
            return self
        return Number.__sub__(self, other)

    # 实现数值减去 Infinity 的操作
    @_sympifyit('other', NotImplemented)
    def __rsub__(self, other):
        return (-self).__add__(other)

    # 继续实现其他运算方法
    # 定义乘法运算符重载方法，处理自身与其他对象的乘法操作
    def __mul__(self, other):
        # 检查其他对象是否为数字类型且全局参数允许求值
        if isinstance(other, Number) and global_parameters.evaluate:
            # 如果其他对象为零或者NaN，则返回NaN
            if other.is_zero or other is S.NaN:
                return S.NaN
            # 如果其他对象是扩展正数，则返回自身
            if other.is_extended_positive:
                return self
            # 否则返回负无穷
            return S.NegativeInfinity
        # 如果不满足条件，则调用父类 Number 的乘法运算
        return Number.__mul__(self, other)
    # 右乘法运算符与左乘法运算符相同
    __rmul__ = __mul__

    # 定义真除法运算符重载方法，处理自身与其他对象的真除法操作
    @_sympifyit('other', NotImplemented)
    def __truediv__(self, other):
        # 检查其他对象是否为数字类型且全局参数允许求值
        if isinstance(other, Number) and global_parameters.evaluate:
            # 如果其他对象是正负无穷或者NaN，则返回NaN
            if other is S.Infinity or \
                other is S.NegativeInfinity or \
                    other is S.NaN:
                return S.NaN
            # 如果其他对象是非负扩展数，则返回自身
            if other.is_extended_nonnegative:
                return self
            # 否则返回负无穷
            return S.NegativeInfinity
        # 如果不满足条件，则调用父类 Number 的真除法运算
        return Number.__truediv__(self, other)

    # 定义绝对值运算符重载方法，始终返回正无穷
    def __abs__(self):
        return S.Infinity

    # 定义负数运算符重载方法，始终返回负无穷
    def __neg__(self):
        return S.NegativeInfinity

    # 定义幂运算的内部求值方法，根据指数的类型返回不同结果
    def _eval_power(self, expt):
        """
        ``expt`` 是一个符号对象，但不等于0或1。

        ================ ======= ==============================
        表达式             结果    备注
        ================ ======= ==============================
        ``oo ** nan``    ``nan``
        ``oo ** -p``     ``0``   ``p`` 是数，``oo``
        ================ ======= ==============================

        参见
        ========
        Pow
        NaN
        NegativeInfinity

        """
        # 如果指数是正扩展数，则返回正无穷
        if expt.is_extended_positive:
            return S.Infinity
        # 如果指数是负扩展数，则返回零
        if expt.is_extended_negative:
            return S.Zero
        # 如果指数是NaN，则返回NaN
        if expt is S.NaN:
            return S.NaN
        # 如果指数是复杂无穷，则返回NaN
        if expt is S.ComplexInfinity:
            return S.NaN
        # 如果指数不是扩展实数且是数值类型，则根据其实部进行判断
        if expt.is_extended_real is False and expt.is_number:
            from sympy.functions.elementary.complexes import re
            expt_real = re(expt)
            # 如果实部是正数，则返回复杂无穷
            if expt_real.is_positive:
                return S.ComplexInfinity
            # 如果实部是负数，则返回零
            if expt_real.is_negative:
                return S.Zero
            # 如果实部是零，则返回NaN
            if expt_real.is_zero:
                return S.NaN

            # 否则返回自身的指数运算结果
            return self**expt.evalf()

    # 返回浮点数值的内部方法，始终返回 mlib.finf
    def _as_mpf_val(self, prec):
        return mlib.finf

    # 定义哈希运算符重载方法，调用父类的哈希运算
    def __hash__(self):
        return super().__hash__()

    # 定义相等运算符重载方法，用于判断与其他对象是否相等于正无穷
    def __eq__(self, other):
        return other is S.Infinity or other == float('inf')

    # 定义不等运算符重载方法，用于判断与其他对象是否不等于正无穷
    def __ne__(self, other):
        return other is not S.Infinity and other != float('inf')

    # 大于运算符重载委托给父类 Expr 的实现
    __gt__ = Expr.__gt__
    # 大于等于运算符重载委托给父类 Expr 的实现
    __ge__ = Expr.__ge__
    # 小于运算符重载委托给父类 Expr 的实现
    __lt__ = Expr.__lt__
    # 小于等于运算符重载委托给父类 Expr 的实现
    __le__ = Expr.__le__

    # 定义取模运算符重载方法，如果其他对象不是表达式则返回NotImplemented
    @_sympifyit('other', NotImplemented)
    def __mod__(self, other):
        if not isinstance(other, Expr):
            return NotImplemented
        return S.NaN

    # 右取模运算符与左取模运算符相同
    __rmod__ = __mod__

    # floor 方法，返回自身
    def floor(self):
        return self

    # ceiling 方法，返回自身
    def ceiling(self):
        return self
oo = S.Infinity
# 定义 oo 为正无穷大

class NegativeInfinity(Number, metaclass=Singleton):
    """Negative infinite quantity.

    NegativeInfinity is a singleton, and can be accessed
    by ``S.NegativeInfinity``.

    See Also
    ========

    Infinity
    """

    is_extended_real = True
    is_complex = False
    is_commutative = True
    is_infinite = True
    is_comparable = True
    is_extended_negative = True
    is_number = True
    is_prime = False

    __slots__ = ()

    def __new__(cls):
        # 创建 NegativeInfinity 类的新实例
        return AtomicExpr.__new__(cls)

    def _latex(self, printer):
        # 返回 LaTeX 格式的负无穷表示
        return r"-\infty"

    def _eval_subs(self, old, new):
        # 如果当前对象等于 old，则替换为 new
        if self == old:
            return new

    def _eval_evalf(self, prec=None):
        # 返回浮点数形式的负无穷大
        return Float('-inf')

    def evalf(self, prec=None, **options):
        # 调用 _eval_evalf 方法返回浮点数形式的负无穷大
        return self._eval_evalf(prec)

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        # 处理与其他数的加法操作
        if isinstance(other, Number) and global_parameters.evaluate:
            if other in (S.Infinity, S.NaN):
                return S.NaN
            return self
        return Number.__add__(self, other)
    __radd__ = __add__

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        # 处理与其他数的减法操作
        if isinstance(other, Number) and global_parameters.evaluate:
            if other in (S.NegativeInfinity, S.NaN):
                return S.NaN
            return self
        return Number.__sub__(self, other)

    @_sympifyit('other', NotImplemented)
    def __rsub__(self, other):
        # 处理其他数减去负无穷大的情况
        return (-self).__add__(other)

    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        # 处理与其他数的乘法操作
        if isinstance(other, Number) and global_parameters.evaluate:
            if other.is_zero or other is S.NaN:
                return S.NaN
            if other.is_extended_positive:
                return self
            return S.Infinity
        return Number.__mul__(self, other)
    __rmul__ = __mul__

    @_sympifyit('other', NotImplemented)
    def __truediv__(self, other):
        # 处理与其他数的除法操作
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.Infinity or \
                other is S.NegativeInfinity or \
                    other is S.NaN:
                return S.NaN
            if other.is_extended_nonnegative:
                return self
            return S.Infinity
        return Number.__truediv__(self, other)

    def __abs__(self):
        # 返回正无穷大的绝对值
        return S.Infinity

    def __neg__(self):
        # 返回正无穷大的负值
        return S.Infinity
    # 定义一个方法用于计算负无穷的指数运算结果
    def _eval_power(self, expt):
        """
        ``expt`` is symbolic object but not equal to 0 or 1.

        ================ ======= ==============================
        Expression       Result  Notes
        ================ ======= ==============================
        ``(-oo) ** nan`` ``nan``
        ``(-oo) ** oo``  ``nan``
        ``(-oo) ** -oo`` ``nan``
        ``(-oo) ** e``   ``oo``  ``e`` is positive even integer
        ``(-oo) ** o``   ``-oo`` ``o`` is positive odd integer
        ================ ======= ==============================

        See Also
        ========

        Infinity
        Pow
        NaN

        """
        # 如果指数是一个数值类型
        if expt.is_number:
            # 如果指数是 NaN 或者正无穷或负无穷，返回 NaN
            if expt is S.NaN or expt is S.Infinity or expt is S.NegativeInfinity:
                return S.NaN

            # 如果指数是正的偶数整数，返回正无穷
            if isinstance(expt, Integer) and expt.is_extended_positive:
                if expt.is_odd:
                    return S.NegativeInfinity
                else:
                    return S.Infinity

            # 计算 Infinity ** expt 和 (-1) ** expt 的结果
            inf_part = S.Infinity**expt
            s_part = S.NegativeOne**expt
            # 如果 Infinity ** expt 等于 0 并且 (-1) ** expt 是有限的，则返回 Infinity ** expt
            if inf_part == 0 and s_part.is_finite:
                return inf_part
            # 如果 Infinity ** expt 是复无穷并且 (-1) ** expt 是有限且不为零，则返回复无穷
            if (inf_part is S.ComplexInfinity and s_part.is_finite and not s_part.is_zero):
                return S.ComplexInfinity
            # 否则返回 (-1) ** expt * Infinity ** expt 的乘积
            return s_part * inf_part

    # 返回负无穷的 MPF（多重精度浮点数）值
    def _as_mpf_val(self, prec):
        return mlib.fninf

    # 计算当前对象的哈希值
    def __hash__(self):
        return super().__hash__()

    # 判断当前对象是否等于另一个对象（是否为负无穷）
    def __eq__(self, other):
        return other is S.NegativeInfinity or other == float('-inf')

    # 判断当前对象是否不等于另一个对象（是否不为负无穷）
    def __ne__(self, other):
        return other is not S.NegativeInfinity and other != float('-inf')

    # 定义当前对象的大于运算符
    __gt__ = Expr.__gt__
    # 定义当前对象的大于等于运算符
    __ge__ = Expr.__ge__
    # 定义当前对象的小于运算符
    __lt__ = Expr.__lt__
    # 定义当前对象的小于等于运算符
    __le__ = Expr.__le__

    # 定义当前对象的模运算（返回 NaN）
    @_sympifyit('other', NotImplemented)
    def __mod__(self, other):
        if not isinstance(other, Expr):
            return NotImplemented
        return S.NaN

    # 定义模运算的反向操作（返回 NaN）
    __rmod__ = __mod__

    # 返回当前对象的 floor 结果（即自身）
    def floor(self):
        return self

    # 返回当前对象的 ceiling 结果（即自身）
    def ceiling(self):
        return self

    # 返回当前对象表示为幂的字典形式（{-1: 1, ∞: 1}）
    def as_powers_dict(self):
        return {S.NegativeOne: 1, S.Infinity: 1}
class NaN(Number, metaclass=Singleton):
    """
    Not a Number.

    Explanation
    ===========

    This serves as a place holder for numeric values that are indeterminate.
    Most operations on NaN, produce another NaN.  Most indeterminate forms,
    such as ``0/0`` or ``oo - oo`` produce NaN.  Two exceptions are ``0**0``
    and ``oo**0``, which all produce ``1`` (this is consistent with Python's
    float).

    NaN is loosely related to floating point nan, which is defined in the
    IEEE 754 floating point standard, and corresponds to the Python
    ``float('nan')``.  Differences are noted below.

    NaN is mathematically not equal to anything else, even NaN itself.  This
    explains the initially counter-intuitive results with ``Eq`` and ``==`` in
    the examples below.

    NaN is not comparable so inequalities raise a TypeError.  This is in
    contrast with floating point nan where all inequalities are false.

    NaN is a singleton, and can be accessed by ``S.NaN``, or can be imported
    as ``nan``.

    Examples
    ========

    >>> from sympy import nan, S, oo, Eq
    >>> nan is S.NaN
    True
    >>> oo - oo
    nan
    >>> nan + 1
    nan
    >>> Eq(nan, nan)   # mathematical equality
    False
    >>> nan == nan     # structural equality
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/NaN

    """
    is_commutative = True  # NaN is commutative in operations
    is_extended_real = None  # NaN is not an extended real number
    is_real = None  # NaN is not a real number
    is_rational = None  # NaN is not a rational number
    is_algebraic = None  # NaN is not an algebraic number
    is_transcendental = None  # NaN is not a transcendental number
    is_integer = None  # NaN is not an integer
    is_comparable = False  # NaN is not comparable with other numbers
    is_finite = None  # NaN is not a finite number
    is_zero = None  # NaN is not zero
    is_prime = None  # NaN is not a prime number
    is_positive = None  # NaN is not a positive number
    is_negative = None  # NaN is not a negative number
    is_number = True  # NaN is considered a number

    __slots__ = ()  # No additional attributes for NaN

    def __new__(cls):
        return AtomicExpr.__new__(cls)  # Create a new instance of NaN

    def _latex(self, printer):
        return r"\text{NaN}"  # LaTeX representation of NaN

    def __neg__(self):
        return self  # Unary negation of NaN returns itself

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        return self  # Addition with NaN returns NaN

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        return self  # Subtraction involving NaN returns NaN

    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        return self  # Multiplication with NaN returns NaN

    @_sympifyit('other', NotImplemented)
    def __truediv__(self, other):
        return self  # Division with NaN returns NaN

    def floor(self):
        return self  # NaN floored is still NaN

    def ceiling(self):
        return self  # NaN ceiled is still NaN

    def _as_mpf_val(self, prec):
        return _mpf_nan  # Return NaN as a multi-precision float value

    def __hash__(self):
        return super().__hash__()  # Hash method for NaN

    def __eq__(self, other):
        # NaN is structurally equal to another NaN
        return other is S.NaN  # NaN equals another NaN structurally

    def __ne__(self, other):
        return other is not S.NaN  # NaN is not equal to any other object

    # Expr will _sympify and raise TypeError
    __gt__ = Expr.__gt__  # Greater than comparison raises TypeError
    __ge__ = Expr.__ge__  # Greater than or equal comparison raises TypeError
    __lt__ = Expr.__lt__  # Less than comparison raises TypeError
    __le__ = Expr.__le__  # Less than or equal comparison raises TypeError

nan = S.NaN  # Define nan as a shortcut for S.NaN

@dispatch(NaN, Expr)  # Dispatch decorator for NaN and Expr type
def _eval_is_eq(a, b):  # Function to evaluate equality between NaN and Expr
    return False  # NaN is not equal to any Expr instance


class ComplexInfinity(AtomicExpr, metaclass=Singleton):
    r"""Complex infinity.

    Explanation
    # 在复分析中，符号`\tilde\infty`代表一个具有无限大小但复数相位未确定的量。
    
    # ComplexInfinity是一个单例对象，可以通过`S.ComplexInfinity`访问，也可以作为`zoo`导入使用。
    
    # 下面是一些示例，展示了使用ComplexInfinity的情况：
    # 1. 导入zoo后，zoo与任何数相加结果仍为zoo。
    # 2. 任何数除以zoo的结果为0。
    # 3. zoo与zoo相加结果为NaN（不是一个数）。
    # 4. zoo乘以zoo的结果仍为zoo。
    
    # 另见：
    # Infinity
    
    class ComplexInfinity:
        # ComplexInfinity是可交换的
        is_commutative = True
        # 它是无限的
        is_infinite = True
        # 它是一个数
        is_number = True
        # 它不是质数
        is_prime = False
        # 它不是复数
        is_complex = False
        # 它不是扩展实数
        is_extended_real = False
    
        # 类型为NumberKind
        kind = NumberKind
    
        # 禁止动态添加新属性
        __slots__ = ()
    
        # 创建一个新的ComplexInfinity对象
        def __new__(cls):
            return AtomicExpr.__new__(cls)
    
        # 返回LaTeX表示为`\tilde{\infty}`
        def _latex(self, printer):
            return r"\tilde{\infty}"
    
        # 返回绝对值为正无穷Infinity
        @staticmethod
        def __abs__():
            return S.Infinity
    
        # 返回自身，即向下取整也是自身
        def floor(self):
            return self
    
        # 返回自身，即向上取整也是自身
        def ceiling(self):
            return self
    
        # 返回负的ComplexInfinity
        @staticmethod
        def __neg__():
            return S.ComplexInfinity
    
        # 计算幂操作
        def _eval_power(self, expt):
            # 如果指数是ComplexInfinity，则结果为NaN
            if expt is S.ComplexInfinity:
                return S.NaN
            
            # 如果指数是一个数
            if isinstance(expt, Number):
                # 如果指数是零，则结果为NaN
                if expt.is_zero:
                    return S.NaN
                else:
                    # 如果指数为正，则结果为ComplexInfinity
                    if expt.is_positive:
                        return S.ComplexInfinity
                    else:
                        # 如果指数为负，则结果为零
                        return S.Zero
zoo = S.ComplexInfinity


# 将变量zoo设置为复无穷大的符号常量S.ComplexInfinity
zoo = S.ComplexInfinity



class NumberSymbol(AtomicExpr):


# 定义一个名为NumberSymbol的类，继承自AtomicExpr类
class NumberSymbol(AtomicExpr):



    is_commutative = True
    is_finite = True
    is_number = True


# 定义NumberSymbol类的三个布尔类型的类属性
    is_commutative = True  # 可交换
    is_finite = True        # 有限
    is_number = True        # 数值



    __slots__ = ()


# 使用空元组设置NumberSymbol类的__slots__属性，以优化内存使用
    __slots__ = ()



    is_NumberSymbol = True


# 设置NumberSymbol类的布尔类型类属性is_NumberSymbol为True
    is_NumberSymbol = True



    kind = NumberKind


# 将类属性kind设置为NumberKind，可能是一个枚举或类，未在此处定义
    kind = NumberKind



    def __new__(cls):
        return AtomicExpr.__new__(cls)


# 实现类的特殊方法__new__，创建并返回一个新的AtomicExpr类实例
    def __new__(cls):
        return AtomicExpr.__new__(cls)



    def approximation(self, number_cls):
        """ Return an interval with number_cls endpoints
        that contains the value of NumberSymbol.
        If not implemented, then return None.
        """


# 方法approximation，接受一个number_cls参数，返回包含NumberSymbol值的number_cls端点的区间
# 如果未实现，则返回None
    def approximation(self, number_cls):
        """ 返回一个带有number_cls端点的区间，包含NumberSymbol的值。
        如果未实现，则返回None。
        """

    def _eval_evalf(self, prec):
        return Float._new(self._as_mpf_val(prec), prec)


# 实现_eval_evalf方法，返回使用精度prec调用_as_mpf_val方法后创建的Float对象
    def _eval_evalf(self, prec):
        return Float._new(self._as_mpf_val(prec), prec)



    def __eq__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if self is other:
            return True
        if other.is_Number and self.is_irrational:
            return False

        return False    # NumberSymbol != non-(Number|self)


# 实现__eq__方法，用于比较对象是否相等
    def __eq__(self, other):
        try:
            other = _sympify(other)  # 尝试将other转换为符号表达式
        except SympifyError:
            return NotImplemented  # 如果转换失败，返回NotImplemented
        if self is other:  # 如果self和other引用同一对象
            return True
        if other.is_Number and self.is_irrational:  # 如果other是数值且self是无理数
            return False

        return False    # NumberSymbol != non-(Number|self)



    def __ne__(self, other):
        return not self == other


# 实现__ne__方法，用于比较对象是否不相等
    def __ne__(self, other):
        return not self == other



    def __le__(self, other):
        if self is other:
            return S.true
        return Expr.__le__(self, other)


# 实现__le__方法，用于小于等于比较
    def __le__(self, other):
        if self is other:  # 如果self和other引用同一对象
            return S.true
        return Expr.__le__(self, other)  # 否则调用父类Expr的__le__方法比较

    def __ge__(self, other):
        if self is other:
            return S.true
        return Expr.__ge__(self, other)


# 实现__ge__方法，用于大于等于比较
    def __ge__(self, other):
        if self is other:  # 如果self和other引用同一对象
            return S.true
        return Expr.__ge__(self, other)  # 否则调用父类Expr的__ge__方法比较



    def __int__(self):
        # subclass with appropriate return value
        raise NotImplementedError


# 实现__int__方法，用于将对象转换为整数，但在此类中未实际实现，而是抛出NotImplementedError异常
    def __int__(self):
        # subclass with appropriate return value
        raise NotImplementedError



    def __hash__(self):
        return super().__hash__()


# 实现__hash__方法，返回由父类AtomicExpr的__hash__方法计算的哈希值
    def __hash__(self):
        return super().__hash__()



class Exp1(NumberSymbol, metaclass=Singleton):


# 定义一个名为Exp1的类，继承自NumberSymbol类，并使用Singleton作为元类
class Exp1(NumberSymbol, metaclass=Singleton):



    r"""The `e` constant.

    Explanation
    ===========

    The transcendental number `e = 2.718281828\ldots` is the base of the
    natural logarithm and of the exponential function, `e = \exp(1)`.
    Sometimes called Euler's number or Napier's constant.

    Exp1 is a singleton, and can be accessed by ``S.Exp1``,
    or can be imported as ``E``.

    Examples
    ========

    >>> from sympy import exp, log, E
    >>> E is exp(1)
    True
    >>> log(E)
    1

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/E_%28mathematical_constant%29
    """


# 类文档字符串，描述了Exp1类代表的数学常数e的含义、性质和用法
    r"""The `e` constant.

    Explanation
    ===========

    The transcendental number `e = 2.718281828\ldots` is the base of the
    natural logarithm and of the exponential function, `e = \exp(1)`.
    Sometimes called Euler's number or Napier's constant.

    Exp1 is a singleton, and can be accessed by ``S.Exp1``,
    or can be imported as ``E``.

    Examples
    ========

    >>> from sympy import exp, log, E
    >>> E is exp(1)
    True
    >>> log(E)
    1

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/E_%28mathematical_constant%29
    """



    is_real = True
    is_positive = True
    is_negative = False  # XXX Forces is_negative/is_nonnegative
    is_irrational = True
    is_number = True
    is_algebraic = False
    is_transcendental = True


# 定义Exp1类的多个布尔类型的类属性，描述了该数学常数的性质
    is_real = True              # 实数
    is_positive = True          # 正数
    is_negative = False         # 负数，强制为is_negative/is_nonnegative
    is_irrational = True        # 无理数
    is_number = True            # 数值
    is_algebraic = False        # 代数数
    is_transcendental = True    # 超越数



    __slots__ = ()


# 使用空元组设置Exp1类的__slots__属性，以优化内存使用
    __slots__ = ()



    def _latex(self, printer):
        return r"e"


# 实现_latex方法，返回数学表达式的LaTeX表示，此处为返回字符串"e"
    def _latex(self, printer):
        return r"e"



    @staticmethod
    def __abs__():
        return S.Exp1


# 实现静态方法__abs__，返回Exp1常数的绝对值，即
    # 如果参数是一个数值对象
    def _eval_power_exp_is_pow(self, arg):
        # 检查参数是否为一个数值
        if arg.is_Number:
            # 如果参数是正无穷大，返回正无穷大
            if arg is oo:
                return oo
            # 如果参数是负无穷大，返回零
            elif arg == -oo:
                return S.Zero
        
        # 导入对数函数，用于后续判断
        from sympy.functions.elementary.exponential import log
        # 如果参数是对数函数对象，返回其参数
        if isinstance(arg, log):
            return arg.args[0]

        # 不自动展开幂次和乘法（参见问题 3351）：
        elif not arg.is_Add:
            # 定义虚数乘以无穷大的结果
            Ioo = I*oo
            # 如果参数是虚数乘以正负无穷大，返回 NaN
            if arg in [Ioo, -Ioo]:
                return nan
            
            # 提取参数中的 pi*i 系数
            coeff = arg.coeff(pi*I)
            if coeff:
                # 如果 2*coeff 是整数
                if (2*coeff).is_integer:
                    # 如果 coeff 是偶数，返回 1
                    if coeff.is_even:
                        return S.One
                    # 如果 coeff 是奇数，返回 -1
                    elif coeff.is_odd:
                        return S.NegativeOne
                    # 如果 coeff + 1/2 是偶数，返回 -i
                    elif (coeff + S.Half).is_even:
                        return -I
                    # 如果 coeff + 1/2 是奇数，返回 i
                    elif (coeff + S.Half).is_odd:
                        return I
                # 如果 coeff 是有理数
                elif coeff.is_Rational:
                    # 对 coeff 取模 2，限制在 [0, 2*pi) 范围内
                    ncoeff = coeff % 2
                    # 如果 ncoeff 大于 1，调整到 (-pi, pi] 范围内
                    if ncoeff > 1:
                        ncoeff -= 2
                    # 如果 ncoeff 不等于 coeff，返回 e 的 ncoeff*pi*i 次方
                    if ncoeff != coeff:
                        return S.Exp1**(ncoeff*S.Pi*S.ImaginaryUnit)

            # 警告：risch.py 中的代码对此处的更改非常敏感
            # 查找单个对数因子
            coeff, terms = arg.as_coeff_Mul()

            # 但是不能乘以无穷大
            if coeff in (oo, -oo):
                return
            
            coeffs, log_term = [coeff], None
            for term in Mul.make_args(terms):
                # 如果项是对数函数
                if isinstance(term, log):
                    # 如果还没有找到对数因子，记录下这个对数的参数
                    if log_term is None:
                        log_term = term.args[0]
                    else:
                        return
                elif term.is_comparable:
                    # 如果项可以比较大小，将其添加到系数列表中
                    coeffs.append(term)
                else:
                    return

            # 返回 log_term 的 Mul(*coeffs) 次方，如果没有对数因子则返回 None
            return log_term**Mul(*coeffs) if log_term else None
        
        # 如果参数是加法表达式
        elif arg.is_Add:
            # 初始化结果列表和添加列表
            out = []
            add = []
            argchanged = False
            for a in arg.args:
                # 如果 a 是 S.One，直接添加到添加列表中
                if a is S.One:
                    add.append(a)
                    continue
                # 计算 self 的 a 次方
                newa = self**a
                # 如果 newa 是 Pow 类型，并且基数是 self
                if isinstance(newa, Pow) and newa.base is self:
                    # 如果指数不等于 a，将指数添加到添加列表中，并标记参数已更改
                    if newa.exp != a:
                        add.append(newa.exp)
                        argchanged = True
                    else:
                        add.append(a)
                else:
                    # 将计算结果添加到输出列表中
                    out.append(newa)
            # 如果输出列表不为空或者参数已更改
            if out or argchanged:
                # 返回乘积的乘积和指数的 Pow 对象
                return Mul(*out)*Pow(self, Add(*add), evaluate=False)
        
        # 如果参数是矩阵
        elif arg.is_Matrix:
            # 计算参数的指数函数
            return arg.exp()

    # 将对象重写为 sin 函数的表达式
    def _eval_rewrite_as_sin(self, **kwargs):
        # 导入正弦函数
        from sympy.functions.elementary.trigonometric import sin
        # 返回 sin(i + π/2) - i*sin(i)
        return sin(I + S.Pi/2) - I*sin(I)
    # 定义一个方法 `_eval_rewrite_as_cos`，用于重写表达式为余弦形式
    def _eval_rewrite_as_cos(self, **kwargs):
        # 导入 sympy 库中的余弦函数
        from sympy.functions.elementary.trigonometric import cos
        # 返回一个复数 I 的余弦值加上复数 I 乘以 (I 加上 π/2) 的余弦值的表达式
        return cos(I) + I*cos(I + S.Pi/2)
# 将 S.Exp1 赋值给变量 E
E = S.Exp1

# 定义 Pi 类，表示数学常数 π
class Pi(NumberSymbol, metaclass=Singleton):
    r"""The `\pi` constant.

    Explanation
    ===========

    The transcendental number `\pi = 3.141592654\ldots` represents the ratio
    of a circle's circumference to its diameter, the area of the unit circle,
    the half-period of trigonometric functions, and many other things
    in mathematics.

    Pi is a singleton, and can be accessed by ``S.Pi``, or can
    be imported as ``pi``.

    Examples
    ========

    >>> from sympy import S, pi, oo, sin, exp, integrate, Symbol
    >>> S.Pi
    pi
    >>> pi > 3
    True
    >>> pi.is_irrational
    True
    >>> x = Symbol('x')
    >>> sin(x + 2*pi)
    sin(x)
    >>> integrate(exp(-x**2), (x, -oo, oo))
    sqrt(pi)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pi
    """

    # 常数特性
    is_real = True
    is_positive = True
    is_negative = False
    is_irrational = True
    is_number = True
    is_algebraic = False
    is_transcendental = True

    __slots__ = ()

    # 返回 LaTeX 格式的表示
    def _latex(self, printer):
        return r"\pi"

    # 返回 Pi 的绝对值
    @staticmethod
    def __abs__():
        return S.Pi

    # 将 Pi 转换为整数，返回 3
    def __int__(self):
        return 3

    # 返回 Pi 的 MPF（多精度浮点数）值
    def _as_mpf_val(self, prec):
        return mpf_pi(prec)

    # 返回 Pi 的近似区间
    def approximation_interval(self, number_cls):
        if issubclass(number_cls, Integer):
            return (Integer(3), Integer(4))
        elif issubclass(number_cls, Rational):
            return (Rational(223, 71, 1), Rational(22, 7, 1))

# 将 S.Pi 赋值给变量 pi
pi = S.Pi

# 定义 GoldenRatio 类，表示黄金比率 φ
class GoldenRatio(NumberSymbol, metaclass=Singleton):
    r"""The golden ratio, `\phi`.

    Explanation
    ===========

    `\phi = \frac{1 + \sqrt{5}}{2}` is an algebraic number.  Two quantities
    are in the golden ratio if their ratio is the same as the ratio of
    their sum to the larger of the two quantities, i.e. their maximum.

    GoldenRatio is a singleton, and can be accessed by ``S.GoldenRatio``.

    Examples
    ========

    >>> from sympy import S
    >>> S.GoldenRatio > 1
    True
    >>> S.GoldenRatio.expand(func=True)
    1/2 + sqrt(5)/2
    >>> S.GoldenRatio.is_irrational
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Golden_ratio
    """

    # 特性
    is_real = True
    is_positive = True
    is_negative = False
    is_irrational = True
    is_number = True
    is_algebraic = True
    is_transcendental = False

    __slots__ = ()

    # 返回 LaTeX 格式的表示
    def _latex(self, printer):
        return r"\phi"

    # 将 GoldenRatio 转换为整数，返回 1
    def __int__(self):
        return 1

    # 返回 GoldenRatio 的 MPF（多精度浮点数）值
    def _as_mpf_val(self, prec):
         # XXX track down why this has to be increased
        rv = mlib.from_man_exp(phi_fixed(prec + 10), -prec - 10)
        return mpf_norm(rv, prec)

    # 扩展 GoldenRatio 的函数表示
    def _eval_expand_func(self, **hints):
        from sympy.functions.elementary.miscellaneous import sqrt
        return S.Half + S.Half*sqrt(5)
    # 定义一个方法用于生成数值类的近似区间
    def approximation_interval(self, number_cls):
        # 如果传入的类是 Integer 的子类，则返回一个区间 (1, 2)
        if issubclass(number_cls, Integer):
            return (S.One, Rational(2))
        # 如果传入的类是 Rational 的子类，则暂时不返回任何值
        elif issubclass(number_cls, Rational):
            pass

    # 将 _eval_rewrite_as_sqrt 方法重写为 _eval_expand_func 方法
    _eval_rewrite_as_sqrt = _eval_expand_func
class TribonacciConstant(NumberSymbol, metaclass=Singleton):
    r"""The tribonacci constant.

    Explanation
    ===========

    The tribonacci numbers are like the Fibonacci numbers, but instead
    of starting with two predetermined terms, the sequence starts with
    three predetermined terms and each term afterwards is the sum of the
    preceding three terms.

    The tribonacci constant is the ratio toward which adjacent tribonacci
    numbers tend. It is a root of the polynomial `x^3 - x^2 - x - 1 = 0`,
    and also satisfies the equation `x + x^{-3} = 2`.

    TribonacciConstant is a singleton, and can be accessed
    by ``S.TribonacciConstant``.

    Examples
    ========

    >>> from sympy import S
    >>> S.TribonacciConstant > 1
    True
    >>> S.TribonacciConstant.expand(func=True)
    1/3 + (19 - 3*sqrt(33))**(1/3)/3 + (3*sqrt(33) + 19)**(1/3)/3
    >>> S.TribonacciConstant.is_irrational
    True
    >>> S.TribonacciConstant.n(20)
    1.8392867552141611326

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Generalizations_of_Fibonacci_numbers#Tribonacci_numbers
    """

    # 设置常数属性
    is_real = True
    is_positive = True
    is_negative = False
    is_irrational = True
    is_number = True
    is_algebraic = True
    is_transcendental = False

    __slots__ = ()  # 使用空元组限制实例的动态属性

    # LaTeX 输出方法
    def _latex(self, printer):
        return r"\text{TribonacciConstant}"

    # 将实例转换为整数
    def __int__(self):
        return 1

    # 返回 MPF (多倍精度浮点数) 值
    def _as_mpf_val(self, prec):
        return self._eval_evalf(prec)._mpf_

    # 返回浮点数近似值
    def _eval_evalf(self, prec):
        rv = self._eval_expand_func(function=True)._eval_evalf(prec + 4)
        return Float(rv, precision=prec)

    # 返回函数的展开形式
    def _eval_expand_func(self, **hints):
        from sympy.functions.elementary.miscellaneous import cbrt, sqrt
        return (1 + cbrt(19 - 3*sqrt(33)) + cbrt(19 + 3*sqrt(33))) / 3

    # 根据数字类型返回适当的近似区间
    def approximation_interval(self, number_cls):
        if issubclass(number_cls, Integer):
            return (S.One, Rational(2))
        elif issubclass(number_cls, Rational):
            pass

    # 重写为基于平方根的形式
    _eval_rewrite_as_sqrt = _eval_expand_func


class EulerGamma(NumberSymbol, metaclass=Singleton):
    r"""The Euler-Mascheroni constant.

    Explanation
    ===========

    `\gamma = 0.5772157\ldots` (also called Euler's constant) is a mathematical
    constant recurring in analysis and number theory.  It is defined as the
    limiting difference between the harmonic series and the
    natural logarithm:

    .. math:: \gamma = \lim\limits_{n\to\infty}
              \left(\sum\limits_{k=1}^n\frac{1}{k} - \ln n\right)

    EulerGamma is a singleton, and can be accessed by ``S.EulerGamma``.

    Examples
    ========

    >>> from sympy import S
    >>> S.EulerGamma.is_irrational
    >>> S.EulerGamma > 0
    True
    >>> S.EulerGamma > 1
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler%E2%80%93Mascheroni_constant
    """

    # 设置常数属性
    is_real = True
    is_positive = True
    is_negative = False
    is_irrational = None  # 缺失 is_irrational 属性
    # 是否是无理数，初始为 None
    is_irrational = None
    # 是否是数值，初始为 True
    is_number = True

    # 定义空的 __slots__，表示没有额外的实例属性
    __slots__ = ()

    # 返回 LaTeX 表示，这里返回固定字符串 "\gamma"
    def _latex(self, printer):
        return r"\gamma"

    # 将对象转换为整数，这里始终返回整数 0
    def __int__(self):
        return 0

    # 返回对象的多精度浮点表示
    def _as_mpf_val(self, prec):
         # XXX track down why this has to be increased
        # 调用 mlib 库的 euler_fixed 方法，精度为 prec+10
        v = mlib.libhyper.euler_fixed(prec + 10)
        # 将返回值 v 转换为多精度浮点数表示，并进行规范化
        rv = mlib.from_man_exp(v, -prec - 10)
        return mpf_norm(rv, prec)

    # 返回逼近区间
    def approximation_interval(self, number_cls):
        # 如果 number_cls 是 Integer 的子类，返回 (0, 1)
        if issubclass(number_cls, Integer):
            return (S.Zero, S.One)
        # 如果 number_cls 是 Rational 的子类，返回 (1/2, 3/5)
        elif issubclass(number_cls, Rational):
            return (S.Half, Rational(3, 5, 1))
# 定义一个表示Catalan常数的类，继承自NumberSymbol，使用Singleton元类
class Catalan(NumberSymbol, metaclass=Singleton):
    r"""Catalan's constant.

    Explanation
    ===========

    $G = 0.91596559\ldots$ is given by the infinite series

    .. math:: G = \sum_{k=0}^{\infty} \frac{(-1)^k}{(2k+1)^2}

    Catalan is a singleton, and can be accessed by ``S.Catalan``.

    Examples
    ========

    >>> from sympy import S
    >>> S.Catalan.is_irrational
    >>> S.Catalan > 0
    True
    >>> S.Catalan > 1
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Catalan%27s_constant
    """

    # Catalan常数是实数
    is_real = True
    # Catalan常数是正数
    is_positive = True
    # Catalan常数不是负数
    is_negative = False
    # 是否为无理数未确定
    is_irrational = None
    # Catalan常数是数值
    is_number = True

    # 使用__slots__限制实例的属性
    __slots__ = ()

    # 将Catalan常数转换为整数返回0
    def __int__(self):
        return 0

    # 使用mpmath计算Catalan常数的多精度浮点数表示
    def _as_mpf_val(self, prec):
        # XXX track down why this has to be increased
        v = mlib.catalan_fixed(prec + 10)
        rv = mlib.from_man_exp(v, -prec - 10)
        return mpf_norm(rv, prec)

    # 给出Catalan常数的近似区间
    def approximation_interval(self, number_cls):
        if issubclass(number_cls, Integer):
            return (S.Zero, S.One)
        elif issubclass(number_cls, Rational):
            return (Rational(9, 10, 1), S.One)

    # 重写为Sum表达式，使用负一的幂和分母为(2k+1)^2的求和来表示Catalan常数
    def _eval_rewrite_as_Sum(self, k_sym=None, symbols=None, **hints):
        if (k_sym is not None) or (symbols is not None):
            return self
        from .symbol import Dummy
        from sympy.concrete.summations import Sum
        k = Dummy('k', integer=True, nonnegative=True)
        return Sum(S.NegativeOne**k / (2*k+1)**2, (k, 0, S.Infinity))

    # 返回Catalan常数的LaTeX表示为"G"
    def _latex(self, printer):
        return "G"


# 定义一个表示虚数单位的类，继承自AtomicExpr，使用Singleton元类
class ImaginaryUnit(AtomicExpr, metaclass=Singleton):
    r"""The imaginary unit, `i = \sqrt{-1}`.

    I is a singleton, and can be accessed by ``S.I``, or can be
    imported as ``I``.

    Examples
    ========

    >>> from sympy import I, sqrt
    >>> sqrt(-1)
    I
    >>> I*I
    -1
    >>> 1/I
    -I

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Imaginary_unit
    """

    # 虚数单位是可交换的
    is_commutative = True
    # 虚数单位是虚数
    is_imaginary = True
    # 虚数单位是有限的
    is_finite = True
    # 虚数单位是数值
    is_number = True
    # 虚数单位是代数的
    is_algebraic = True
    # 虚数单位不是超越数的
    is_transcendental = False

    # 数字的种类为NumberKind
    kind = NumberKind

    # 使用__slots__限制实例的属性
    __slots__ = ()

    # 返回虚数单位的LaTeX表示
    def _latex(self, printer):
        return printer._settings['imaginary_unit_latex']

    # 虚数单位的绝对值为1
    @staticmethod
    def __abs__():
        return S.One

    # 返回虚数单位的浮点数近似值
    def _eval_evalf(self, prec):
        return self

    # 返回虚数单位的共轭，即返回-S.ImaginaryUnit
    def _eval_conjugate(self):
        return -S.ImaginaryUnit
    # 定义一个用于计算复数幂次的函数，参数为幂指数 expt
    def _eval_power(self, expt):
        """
        b is I = sqrt(-1)
        e is symbolic object but not equal to 0, 1

        I**r -> (-1)**(r/2) -> exp(r/2*Pi*I) -> sin(Pi*r/2) + cos(Pi*r/2)*I, r is decimal
        I**0 mod 4 -> 1
        I**1 mod 4 -> I
        I**2 mod 4 -> -1
        I**3 mod 4 -> -I
        """

        # 如果指数 expt 是整数类型
        if isinstance(expt, Integer):
            # 对指数取模 4
            expt = expt % 4
            # 根据不同的模值返回不同的复数幂结果
            if expt == 0:
                return S.One
            elif expt == 1:
                return S.ImaginaryUnit
            elif expt == 2:
                return S.NegativeOne
            elif expt == 3:
                return -S.ImaginaryUnit
        
        # 如果指数 expt 是有理数类型
        if isinstance(expt, Rational):
            # 将 expt 拆分成整数部分 i 和余数部分 r
            i, r = divmod(expt, 2)
            # 计算幂次为 r 的虚数单位
            rv = Pow(S.ImaginaryUnit, r, evaluate=False)
            # 如果整数部分 i 是奇数，则乘以 -1
            if i % 2:
                return Mul(S.NegativeOne, rv, evaluate=False)
            # 否则直接返回 rv
            return rv

    # 返回 -1 和 1/2 作为基数和指数
    def as_base_exp(self):
        return S.NegativeOne, S.Half

    # 返回一个元组，包含 Float(0)._mpf_ 和 Float(1)._mpf_
    @property
    def _mpc_(self):
        return (Float(0)._mpf_, Float(1)._mpf_)
I = S.ImaginaryUnit


# 将S.ImaginaryUnit赋值给变量I，表示虚数单位



def int_valued(x):
    """return True only for a literal Number whose internal
    representation as a fraction has a denominator of 1,
    else False, i.e. integer, with no fractional part.
    """
    if isinstance(x, (SYMPY_INTS, int)):
        return True
    if type(x) is float:
        return x.is_integer()
    if isinstance(x, Integer):
        return True
    if isinstance(x, Float):
        # x = s*m*2**p; _mpf_ = s,m,e,p
        return x._mpf_[2] >= 0
    return False  # or add new types to recognize


# 判断参数x是否表示整数值，返回True或False
# - 如果x是SYMPY_INTS、int类型之一，返回True
# - 如果x是float类型，则检查其是否为整数
# - 如果x是Integer类型，返回True
# - 如果x是Float类型，检查其内部表示中指数部分是否大于等于0
# - 对于其他类型，返回False



def equal_valued(x, y):
    """Compare expressions treating plain floats as rationals.

    Examples
    ========

    >>> from sympy import S, symbols, Rational, Float
    >>> from sympy.core.numbers import equal_valued
    >>> equal_valued(1, 2)
    False
    >>> equal_valued(1, 1)
    True

    In SymPy expressions with Floats compare unequal to corresponding
    expressions with rationals:

    >>> x = symbols('x')
    >>> x**2 == x**2.0
    False

    However an individual Float compares equal to a Rational:

    >>> Rational(1, 2) == Float(0.5)
    False

    In a future version of SymPy this might change so that Rational and Float
    compare unequal. This function provides the behavior currently expected of
    ``==`` so that it could still be used if the behavior of ``==`` were to
    change in future.

    >>> equal_valued(1, 1.0) # Float vs Rational
    True
    >>> equal_valued(S(1).n(3), S(1).n(5)) # Floats of different precision
    True

    Explanation
    ===========

    In future SymPy verions Float and Rational might compare unequal and floats
    with different precisions might compare unequal. In that context a function
    is needed that can check if a number is equal to 1 or 0 etc. The idea is
    that instead of testing ``if x == 1:`` if we want to accept floats like
    ``1.0`` as well then the test can be written as ``if equal_valued(x, 1):``
    or ``if equal_valued(x, 2):``. Since this function is intended to be used
    in situations where one or both operands are expected to be concrete
    numbers like 1 or 0 the function does not recurse through the args of any
    compound expression to compare any nested floats.

    References
    ==========

    .. [1] https://github.com/sympy/sympy/pull/20033
    """
    x = _sympify(x)
    y = _sympify(y)

    # Handle everything except Float/Rational first
    if not x.is_Float and not y.is_Float:
        return x == y
    elif x.is_Float and y.is_Float:
        # Compare values without regard for precision
        return x._mpf_ == y._mpf_
    elif x.is_Float:
        x, y = y, x
    if not x.is_Rational:
        return False

    # Now y is Float and x is Rational. A simple approach at this point would
    # just be x == Rational(y) but if y has a large exponent creating a
    # Rational could be prohibitively expensive.

    sign, man, exp, _ = y._mpf_
    p, q = x.p, x.q

    if sign:
        man = -man


# 比较表达式x和y，将普通的float视为有理数处理
# - 如果x和y都不是Float类型，直接比较它们
# - 如果x和y都是Float类型，比较它们的内部表示是否相等
# - 如果x是Float而y不是，则交换x和y再进行比较
# - 如果x不是有理数Rational，则返回False
# - 在特定情况下，比较一个有理数和一个浮点数，通过比较内部表示来确定它们是否相等
    # 如果指数 exp 等于 0
    if exp == 0:
        # 当 y 是奇整数时，返回条件判断 q == 1 and man == p
        return q == 1 and man == p
    # 如果指数 exp 大于 0
    elif exp > 0:
        # 当 y 是偶整数时
        # 如果 q 不等于 1，则返回 False
        if q != 1:
            return False
        # 如果 p 的二进制位数不等于 man 的二进制位数加上 exp，则返回 False
        if p.bit_length() != man.bit_length() + exp:
            return False
        # 返回条件判断 man 左移 exp 位是否等于 p
        return man << exp == p
    # 如果指数 exp 小于 0
    else:
        # 当 y 是非整数时，需要满足 p == man 和 q == 2 的负 exp 次方
        # 如果 p 不等于 man，则返回 False
        if p != man:
            return False
        # 计算负指数 neg_exp = -exp
        neg_exp = -exp
        # 如果 q 的二进制位数减 1 不等于 neg_exp，则返回 False
        if q.bit_length() - 1 != neg_exp:
            return False
        # 返回条件判断 1 左移 neg_exp 位是否等于 q
        return (1 << neg_exp) == q
def all_close(expr1, expr2, rtol=1e-5, atol=1e-8):
    """Return True if expr1 and expr2 are numerically close.

    The expressions must have the same structure, but any Rational, Integer, or
    Float numbers they contain are compared approximately using rtol and atol.
    Any other parts of expressions are compared exactly. However, allowance is
    made to allow for the additive and multiplicative identities.

    Relative tolerance is measured with respect to expr2 so when used in
    testing expr2 should be the expected correct answer.

    Examples
    ========

    >>> from sympy import exp
    >>> from sympy.abc import x, y
    >>> from sympy.core.numbers import all_close
    >>> expr1 = 0.1*exp(x - y)
    >>> expr2 = exp(x - y)/10
    >>> expr1
    0.1*exp(x - y)
    >>> expr2
    exp(x - y)/10
    >>> expr1 == expr2
    False
    >>> all_close(expr1, expr2)
    True

    Identities are automatically supplied:

    >>> all_close(x, x + 1e-10)
    True
    >>> all_close(x, 1.0*x)
    True
    >>> all_close(x, 1.0*x + 1e-10)
    True

    """
    NUM_TYPES = (Rational, Float)

    def _all_close(obj1, obj2):
        """Recursively check if obj1 and obj2 are numerically close."""
        if type(obj1) == type(obj2) and isinstance(obj1, (list, tuple)):
            if len(obj1) != len(obj2):
                return False
            return all(_all_close(e1, e2) for e1, e2 in zip(obj1, obj2))
        else:
            return _all_close_expr(_sympify(obj1), _sympify(obj2))

    def _all_close_expr(expr1, expr2):
        """Check if expr1 and expr2 are numerically close."""
        num1 = isinstance(expr1, NUM_TYPES)
        num2 = isinstance(expr2, NUM_TYPES)
        if num1 != num2:
            return False
        elif num1:
            return _close_num(expr1, expr2)
        if expr1.is_Add or expr1.is_Mul or expr2.is_Add or expr2.is_Mul:
            return _all_close_ac(expr1, expr2)
        if expr1.func != expr2.func or len(expr1.args) != len(expr2.args):
            return False
        args = zip(expr1.args, expr2.args)
        return all(_all_close_expr(a1, a2) for a1, a2 in args)

    def _close_num(num1, num2):
        """Check if num1 and num2 are close within given tolerances."""
        return bool(abs(num1 - num2) <= atol + rtol*abs(num2))
    def _all_close_ac(expr1, expr2):
        # 比较具有结合交换操作符的表达式，通过检查所有项具有等价系数（总是 Rational 或 Float 类型）来进行近似相等性比较
        if expr1.is_Mul or expr2.is_Mul:
            # 如果表达式中包含乘法项，使用 as_coeff_mul 方法自动获取系数和乘法项
            c1, e1 = expr1.as_coeff_mul(rational=False)
            c2, e2 = expr2.as_coeff_mul(rational=False)
            # 检查系数是否近似相等
            if not _close_num(c1, c2):
                return False
            # 将乘法项转换为集合，找出共同的乘法项
            s1 = set(e1)
            s2 = set(e2)
            common = s1 & s2
            s1 -= common
            s2 -= common
            # 如果没有剩余的乘法项，说明完全匹配
            if not s1:
                return True
            # 如果乘法项中没有包含 Float 类型的因子，则不匹配
            if not any(i.has(Float) for j in (s1, s2) for i in j):
                return False
            # 对乘法项进行排序和匹配，例如 x != x**1.0, exp(x) != exp(1.0*x) 等情况
            s1 = [i.as_base_exp() for i in ordered(s1)]
            s2 = [i.as_base_exp() for i in ordered(s2)]
            unmatched = list(range(len(s1)))
            for be1 in s1:
                for i in unmatched:
                    be2 = s2[i]
                    # 检查基数和指数是否近似相等
                    if _all_close(be1, be2):
                        unmatched.remove(i)
                        break
                else:
                    return False
            # 检查是否所有的乘法项都匹配成功
            return not(unmatched)
        # 如果表达式中包含加法项，则使用 as_coefficients_dict 方法获取系数字典
        assert expr1.is_Add or expr2.is_Add
        cd1 = expr1.as_coefficients_dict()
        cd2 = expr2.as_coefficients_dict()
        # 检查常数项系数是否近似相等
        if not _close_num(cd1[1], cd2[1]):
            return False
        # 检查系数字典长度是否相等
        if len(cd1) != len(cd2):
            return False
        # 遍历第一个系数字典的键
        for k in list(cd1):
            # 如果第二个系数字典也包含该键，则比较它们的值是否近似相等
            if k in cd2:
                if not _close_num(cd1.pop(k), cd2.pop(k)):
                    return False
            # 如果在第二个系数字典中找不到相同的键，则不匹配
        else:
            # 如果第一个系数字典为空，则说明匹配成功
            if not cd1:
                return True
        # 对于第一个系数字典中剩余的键，与第二个系数字典中的键进行匹配
        for k1 in cd1:
            for k2 in cd2:
                if _all_close_expr(k1, k2):
                    # 找到匹配的键
                    # XXX 可能存在多个匹配的情况，但这里没有考虑哪个更好的情况
                    if not _close_num(cd1[k1], cd2[k2]):
                        return False
                    break
            else:
                # 没有找到匹配的键
                return False
        # 所有键和对应值都匹配成功
        return True

    return _all_close(expr1, expr2)
# 使用 @dispatch 装饰器为 _eval_is_eq 函数添加多态支持，接受元组和数字作为参数
@dispatch(Tuple, Number) # type:ignore
# 声明 _eval_is_eq 函数，用于与其他对象比较，此处忽略 F811 错误
def _eval_is_eq(self, other): # noqa: F811
    # 总是返回 False，暗示当前对象不等于其他对象
    return False


# 定义一个函数 sympify_fractions，用于将 fractions.Fraction 类型转换为 SymPy 的 Rational 类型
def sympify_fractions(f):
    return Rational(f.numerator, f.denominator, 1)

# 将 sympify_fractions 函数注册到 _sympy_converter 字典中，以支持 fractions.Fraction 类型的转换
_sympy_converter[fractions.Fraction] = sympify_fractions


# 如果 gmpy 模块可用，则定义以下两个函数用于将 gmpy.mpz 和 gmpy.mpq 类型转换为 SymPy 的 Integer 和 Rational 类型
if gmpy is not None:

    def sympify_mpz(x):
        return Integer(int(x))

    def sympify_mpq(x):
        return Rational(int(x.numerator), int(x.denominator))

    # 将 sympify_mpz 和 sympify_mpq 函数注册到 _sympy_converter 字典中，以支持 gmpy.mpz 和 gmpy.mpq 类型的转换
    _sympy_converter[type(gmpy.mpz(1))] = sympify_mpz
    _sympy_converter[type(gmpy.mpq(1, 2))] = sympify_mpq


# 如果 flint 模块可用，则定义以下两个函数用于将 flint.fmpz 和 flint.fmpq 类型转换为 SymPy 的 Integer 和 Rational 类型
if flint is not None:

    def sympify_fmpz(x):
        return Integer(int(x))

    def sympify_fmpq(x):
        return Rational(int(x.numerator), int(x.denominator))

    # 将 sympify_fmpz 和 sympify_fmpq 函数注册到 _sympy_converter 字典中，以支持 flint.fmpz 和 flint.fmpq 类型的转换
    _sympy_converter[type(flint.fmpz(1))] = sympify_fmpz
    _sympy_converter[type(flint.fmpq(1, 2))] = sympify_fmpq


# 定义一个函数 sympify_mpmath，用于将 mpmath 中的 mpnumeric 对象转换为 SymPy 的 Expr 类型
def sympify_mpmath(x):
    return Expr._from_mpmath(x, x.context.prec)

# 将 sympify_mpmath 函数注册到 _sympy_converter 字典中，以支持 mpmath 中的 mpnumeric 对象的转换
_sympy_converter[mpnumeric] = sympify_mpmath


# 定义一个函数 sympify_complex，用于将 complex 类型转换为 SymPy 的复数类型表示
def sympify_complex(a):
    # 分别对实部和虚部进行 sympify 转换，然后返回复数表示
    real, imag = list(map(sympify, (a.real, a.imag)))
    return real + S.ImaginaryUnit*imag

# 将 sympify_complex 函数注册到 _sympy_converter 字典中，以支持 complex 类型的转换
_sympy_converter[complex] = sympify_complex


# 导入并设置 Mul 类的 identity 属性为 One 对象，用于表示乘法中的单位元
from .power import Pow
from .mul import Mul
Mul.identity = One()

# 导入并设置 Add 类的 identity 属性为 Zero 对象，用于表示加法中的零元
from .add import Add
Add.identity = Zero()


# 定义一个函数 _register_classes，用于注册 SymPy 中的数值类型到相应的抽象层次
def _register_classes():
    numbers.Number.register(Number)
    numbers.Real.register(Float)
    numbers.Rational.register(Rational)
    numbers.Integral.register(Integer)

# 调用 _register_classes 函数，注册 SymPy 中的数值类型
_register_classes()


# 定义一个元组 _illegal，包含 SymPy 中非法的特殊值
_illegal = (S.NaN, S.Infinity, S.NegativeInfinity, S.ComplexInfinity)
```