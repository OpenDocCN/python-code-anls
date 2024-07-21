# `.\pytorch\torch\utils\_sympy\numbers.py`

```
# mypy: allow-untyped-defs
# 导入 mpmath.libmp 中的 mlib 模块（未指定类型的导入）
import mpmath.libmp as mlib  # type: ignore[import-untyped]

# 导入 sympy 库及其需要的模块
import sympy
from sympy import Expr
from sympy.core.decorators import _sympifyit
from sympy.core.expr import AtomicExpr
from sympy.core.numbers import Number
from sympy.core.parameters import global_parameters
from sympy.core.singleton import S, Singleton

# 定义一个名为 IntInfinity 的类，继承自 Number，使用 Singleton 元类
class IntInfinity(Number, metaclass=Singleton):
    r"""Positive integer infinite quantity.

    Integer infinity is a value in an extended integers which
    is greater than all other integers.  We distinguish it from
    sympy's existing notion of infinity in that it reports that
    it is_integer.

    Infinity is a singleton, and can be accessed by ``S.IntInfinity``,
    or can be imported as ``int_oo``.
    """

    # NB: We can't actually mark this as infinite, as integer and infinite are
    # inconsistent assumptions in sympy.  We also report that we are complex,
    # different from sympy.oo

    # 标志这个类是整数（is_integer）和可交换（is_commutative）
    is_integer = True
    is_commutative = True
    is_number = True
    is_extended_real = True
    is_comparable = True
    is_extended_positive = True
    is_prime = False

    # 确保在处理操作符时优先于普通数字
    _op_priority = 100.0

    # 确保类实例没有额外的实例属性
    __slots__ = ()

    # 构造方法，返回一个 AtomicExpr 类型的新实例
    def __new__(cls):
        return AtomicExpr.__new__(cls)

    # 返回对象的字符串表示形式
    def _sympystr(self, printer):
        return "int_oo"

    # 替换方法，用于符号代换
    def _eval_subs(self, old, new):
        if self == old:
            return new

    # 定义加法操作
    @_sympifyit("other", NotImplemented)
    def __add__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.NegativeInfinity:
                return S.NegativeInfinity
            if other in (S.NegativeIntInfinity, S.NaN):
                return S.NaN
            return self
        return Number.__add__(self, other)

    # 右加法操作
    __radd__ = __add__

    # 定义减法操作
    @_sympifyit("other", NotImplemented)
    def __sub__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.Infinity:
                return S.NegativeInfinity
            if other in (S.IntInfinity, S.NaN):
                return S.NaN
            return self
        return Number.__sub__(self, other)

    # 右减法操作
    @_sympifyit("other", NotImplemented)
    def __rsub__(self, other):
        return (-self).__add__(other)

    # 定义乘法操作
    @_sympifyit("other", NotImplemented)
    def __mul__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other.is_zero or other is S.NaN:
                return S.NaN
            if other.is_extended_positive:
                return self
            return S.NegativeIntInfinity
        return Number.__mul__(self, other)

    # 右乘法操作
    __rmul__ = __mul__

    # 剩余方法省略
    # 定义 __truediv__ 方法，处理对象与数值相除的操作
    def __truediv__(self, other):
        # 如果 other 是数字类型且全局参数 evaluate 为真
        if isinstance(other, Number) and global_parameters.evaluate:
            # 如果 other 是以下特定值之一，则返回 NaN
            if other in (
                S.Infinity,
                S.IntInfinity,
                S.NegativeInfinity,
                S.NegativeIntInfinity,
                S.NaN,
            ):
                return S.NaN
            # 如果 other 是非负数，则返回正无穷，表示 truediv 结果为浮点数
            if other.is_extended_nonnegative:
                return S.Infinity
            # 否则返回负无穷，表示 truediv 结果为浮点数
            return S.NegativeInfinity
        # 如果不满足上述条件，调用父类 Number 的 __truediv__ 方法处理
        return Number.__truediv__(self, other)

    # 定义 __abs__ 方法，返回正无穷
    def __abs__(self):
        return S.IntInfinity

    # 定义 __neg__ 方法，返回负无穷
    def __neg__(self):
        return S.NegativeIntInfinity

    # 定义 _eval_power 方法，处理对象的幂运算
    def _eval_power(self, expt):
        # 如果指数 expt 是扩展正数，返回正无穷
        if expt.is_extended_positive:
            return S.IntInfinity
        # 如果指数 expt 是扩展负数，返回零
        if expt.is_extended_negative:
            return S.Zero
        # 如果指数 expt 是 NaN，返回 NaN
        if expt is S.NaN:
            return S.NaN
        # 如果指数 expt 是 ComplexInfinity，返回 NaN
        if expt is S.ComplexInfinity:
            return S.NaN
        # 如果指数 expt 不是扩展实数且是数字，则进一步处理
        if expt.is_extended_real is False and expt.is_number:
            from sympy.functions.elementary.complexes import re

            # 取出指数 expt 的实部
            expt_real = re(expt)
            # 如果实部为正数，返回正无穷
            if expt_real.is_positive:
                return S.ComplexInfinity
            # 如果实部为负数，返回零
            if expt_real.is_negative:
                return S.Zero
            # 如果实部为零，返回 NaN
            if expt_real.is_zero:
                return S.NaN

            # 否则返回自身的 expt 次幂的数值估计结果
            return self ** expt.evalf()

    # 定义 _as_mpf_val 方法，返回 mlib.finf
    def _as_mpf_val(self, prec):
        return mlib.finf

    # 定义 __hash__ 方法，调用父类的哈希方法
    def __hash__(self):
        return super().__hash__()

    # 定义 __eq__ 方法，判断对象是否等于正无穷
    def __eq__(self, other):
        return other is S.IntInfinity

    # 定义 __ne__ 方法，判断对象是否不等于正无穷
    def __ne__(self, other):
        return other is not S.IntInfinity

    # 定义 __gt__ 方法，判断对象是否大于给定值
    def __gt__(self, other):
        if other is S.Infinity:
            return sympy.false  # sympy.oo > int_oo
        elif other is S.IntInfinity:
            return sympy.false  # 与 sympy.oo 保持一致
        else:
            return sympy.true

    # 定义 __ge__ 方法，判断对象是否大于或等于给定值
    def __ge__(self, other):
        if other is S.Infinity:
            return sympy.false  # sympy.oo > int_oo
        elif other is S.IntInfinity:
            return sympy.true  # 与 sympy.oo 保持一致
        else:
            return sympy.true

    # 定义 __lt__ 方法，判断对象是否小于给定值
    def __lt__(self, other):
        if other is S.Infinity:
            return sympy.true  # sympy.oo > int_oo
        elif other is S.IntInfinity:
            return sympy.false  # 与 sympy.oo 保持一致
        else:
            return sympy.false

    # 定义 __le__ 方法，判断对象是否小于或等于给定值
    def __le__(self, other):
        if other is S.Infinity:
            return sympy.true  # sympy.oo > int_oo
        elif other is S.IntInfinity:
            return sympy.true  # 与 sympy.oo 保持一致
        else:
            return sympy.false

    # 使用 _sympifyit 装饰器定义 __mod__ 方法，处理对象与其他表达式的取模运算
    @_sympifyit("other", NotImplemented)
    def __mod__(self, other):
        # 如果 other 不是表达式类型，返回 NotImplemented
        if not isinstance(other, Expr):
            return NotImplemented
        # 否则返回 NaN
        return S.NaN

    # 定义 __rmod__ 方法，与 __mod__ 方法功能相同
    __rmod__ = __mod__

    # 定义 floor 方法，返回自身对象
    def floor(self):
        return self

    # 定义 ceiling 方法，返回自身对象
    def ceiling(self):
        return self
# 定义一个名为 int_oo 的全局变量，其值为 S.IntInfinity，表示正无穷大

class NegativeIntInfinity(Number, metaclass=Singleton):
    """Negative integer infinite quantity.

    NegativeInfinity is a singleton, and can be accessed
    by ``S.NegativeInfinity``.

    See Also
    ========

    IntInfinity
    """

    # 确保我们在普通数字之前被分派执行
    _op_priority = 100.0

    # 表明这是一个整数
    is_integer = True
    # 表明这是一个扩展实数
    is_extended_real = True
    # 表明这是可交换的
    is_commutative = True
    # 表明这是可比较的
    is_comparable = True
    # 表明这是扩展负数
    is_extended_negative = True
    # 表明这是一个数字
    is_number = True
    # 表明这不是一个质数
    is_prime = False

    # 限制类实例中允许的属性，节省内存
    __slots__ = ()

    # 创建类的新实例，继承自 AtomicExpr
    def __new__(cls):
        return AtomicExpr.__new__(cls)

    # 替换对象表达式中的旧表达式为新表达式
    def _eval_subs(self, old, new):
        if self == old:
            return new

    # 返回对象的字符串表示，用于打印输出
    def _sympystr(self, printer):
        return "-int_oo"

    """
    def _eval_evalf(self, prec=None):
        return Float('-inf')

    def evalf(self, prec=None, **options):
        return self._eval_evalf(prec)
    """

    # 符号计算：将对象与其他对象相加
    @_sympifyit("other", NotImplemented)
    def __add__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.Infinity:
                return S.Infinity
            if other in (S.IntInfinity, S.NaN):
                return S.NaN
            return self
        return Number.__add__(self, other)

    # 右加法运算符重载
    __radd__ = __add__

    # 符号计算：将对象与其他对象相减
    @_sympifyit("other", NotImplemented)
    def __sub__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.NegativeInfinity:
                return S.Infinity
            if other in (S.NegativeIntInfinity, S.NaN):
                return S.NaN
            return self
        return Number.__sub__(self, other)

    # 右减法运算符重载
    @_sympifyit("other", NotImplemented)
    def __rsub__(self, other):
        return (-self).__add__(other)

    # 符号计算：将对象与其他对象相乘
    @_sympifyit("other", NotImplemented)
    def __mul__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other.is_zero or other is S.NaN:
                return S.NaN
            if other.is_extended_positive:
                return self
            return S.IntInfinity
        return Number.__mul__(self, other)

    # 右乘法运算符重载
    __rmul__ = __mul__

    # 符号计算：将对象与其他对象进行真除运算
    @_sympifyit("other", NotImplemented)
    def __truediv__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other in (
                S.Infinity,
                S.IntInfinity,
                S.NegativeInfinity,
                S.NegativeIntInfinity,
                S.NaN,
            ):
                return S.NaN
            if other.is_extended_nonnegative:
                return self
            return S.Infinity  # truediv returns float
        return Number.__truediv__(self, other)

    # 返回对象的绝对值
    def __abs__(self):
        return S.IntInfinity

    # 返回对象的负值
    def __neg__(self):
        return S.IntInfinity
    # 定义一个方法来评估幂操作的结果，参数为指数 expt
    def _eval_power(self, expt):
        # 如果指数是一个数值
        if expt.is_number:
            # 如果指数在以下特定值中，返回 NaN
            if expt in (
                S.NaN,
                S.Infinity,
                S.NegativeInfinity,
                S.IntInfinity,
                S.NegativeIntInfinity,
            ):
                return S.NaN

            # 如果指数是 sympy.Integer 类型且是扩展正数
            if isinstance(expt, sympy.Integer) and expt.is_extended_positive:
                # 如果指数是奇数，返回 NegativeIntInfinity；否则返回 IntInfinity
                if expt.is_odd:
                    return S.NegativeIntInfinity
                else:
                    return S.IntInfinity

            # 计算 Inf 为底数的指数幂，S 为底数为 -1 的指数幂
            inf_part = S.IntInfinity ** expt
            s_part = S.NegativeOne ** expt
            # 如果 Inf 部分为 0 并且 S 部分是有限的，返回 Inf 部分
            if inf_part == 0 and s_part.is_finite:
                return inf_part
            # 如果 Inf 部分是 ComplexInfinity 并且 S 部分是有限的且非零，返回 ComplexInfinity
            if (
                inf_part is S.ComplexInfinity
                and s_part.is_finite
                and not s_part.is_zero
            ):
                return S.ComplexInfinity
            # 否则返回 S 部分乘以 Inf 部分的结果
            return s_part * inf_part

    # 将对象转换为指定精度 prec 的 mpf 值
    def _as_mpf_val(self, prec):
        return mlib.fninf

    # 计算对象的哈希值
    def __hash__(self):
        return super().__hash__()

    # 检查对象是否等于 S.NegativeIntInfinity
    def __eq__(self, other):
        return other is S.NegativeIntInfinity

    # 检查对象是否不等于 S.NegativeIntInfinity
    def __ne__(self, other):
        return other is not S.NegativeIntInfinity

    # 比较对象是否大于给定的 other 对象
    def __gt__(self, other):
        if other is S.NegativeInfinity:
            return sympy.true  # -sympy.oo < -int_oo
        elif other is S.NegativeIntInfinity:
            return sympy.false  # 与 sympy.oo 的一致性
        else:
            return sympy.false

    # 比较对象是否大于等于给定的 other 对象
    def __ge__(self, other):
        if other is S.NegativeInfinity:
            return sympy.true  # -sympy.oo < -int_oo
        elif other is S.NegativeIntInfinity:
            return sympy.true  # 与 sympy.oo 的一致性
        else:
            return sympy.false

    # 比较对象是否小于给定的 other 对象
    def __lt__(self, other):
        if other is S.NegativeInfinity:
            return sympy.false  # -sympy.oo < -int_oo
        elif other is S.NegativeIntInfinity:
            return sympy.false  # 与 sympy.oo 的一致性
        else:
            return sympy.true

    # 比较对象是否小于等于给定的 other 对象
    def __le__(self, other):
        if other is S.NegativeInfinity:
            return sympy.false  # -sympy.oo < -int_oo
        elif other is S.NegativeIntInfinity:
            return sympy.true  # 与 sympy.oo 的一致性
        else:
            return sympy.true

    # 定义对象的模运算方法，当 other 不是表达式时返回 NotImplemented
    @_sympifyit("other", NotImplemented)
    def __mod__(self, other):
        if not isinstance(other, Expr):
            return NotImplemented
        return S.NaN

    # 右侧模运算与左侧模运算等效
    __rmod__ = __mod__

    # 返回对象本身，表示 floor 操作
    def floor(self):
        return self

    # 返回对象本身，表示 ceiling 操作
    def ceiling(self):
        return self

    # 返回对象作为幂的字典表示，包含 S.NegativeOne 和 S.IntInfinity
    def as_powers_dict(self):
        return {S.NegativeOne: 1, S.IntInfinity: 1}
```