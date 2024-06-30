# `D:\src\scipysrc\sympy\sympy\polys\domains\modularinteger.py`

```
"""Implementation of :class:`ModularInteger` class. """

from __future__ import annotations  # 导入将来版本的注释语法支持
from typing import Any  # 导入 Any 类型的支持

import operator  # 导入 operator 模块

from sympy.polys.polyutils import PicklableWithSlots  # 从 sympy.polys.polyutils 导入 PicklableWithSlots
from sympy.polys.polyerrors import CoercionFailed  # 从 sympy.polys.polyerrors 导入 CoercionFailed
from sympy.polys.domains.domainelement import DomainElement  # 从 sympy.polys.domains.domainelement 导入 DomainElement

from sympy.utilities import public  # 从 sympy.utilities 导入 public
from sympy.utilities.exceptions import sympy_deprecation_warning  # 从 sympy.utilities.exceptions 导入 sympy_deprecation_warning

@public  # 装饰器，将类声明为公共类
class ModularInteger(PicklableWithSlots, DomainElement):
    """A class representing a modular integer. """

    mod, dom, sym, _parent = None, None, None, None  # 类属性初始化为 None

    __slots__ = ('val',)  # 使用 __slots__ 机制，限制只能有 'val' 这一个实例属性

    def parent(self):
        return self._parent  # 返回 _parent 属性的值

    def __init__(self, val):
        if isinstance(val, self.__class__):
            self.val = val.val % self.mod  # 如果 val 是 ModularInteger 类的实例，则取其 val 属性并对 mod 取模赋给 self.val
        else:
            self.val = self.dom.convert(val) % self.mod  # 否则，将 val 转换为 dom 类型再取 mod 取模赋给 self.val

    def modulus(self):
        return self.mod  # 返回 mod 属性的值

    def __hash__(self):
        return hash((self.val, self.mod))  # 返回一个哈希值，基于 (self.val, self.mod) 组成的元组

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.val)  # 返回一个字符串表示，包含类名和 self.val 的值

    def __str__(self):
        return "%s mod %s" % (self.val, self.mod)  # 返回一个字符串表示，格式为 "self.val mod self.mod"

    def __int__(self):
        return int(self.val)  # 将 self.val 转换为整数并返回

    def to_int(self):
        """Converts the ModularInteger to an integer.

        Deprecated method with a warning message suggesting alternatives.
        """
        sympy_deprecation_warning(
            """ModularInteger.to_int() is deprecated.

            Use int(a) or K = GF(p) and K.to_int(a) instead of a.to_int().
            """,
            deprecated_since_version="1.13",
            active_deprecations_target="modularinteger-to-int",
        )

        if self.sym:
            if self.val <= self.mod // 2:
                return self.val  # 如果 self.sym 为真且 self.val <= self.mod // 2，则返回 self.val
            else:
                return self.val - self.mod  # 否则，返回 self.val - self.mod
        else:
            return self.val  # 如果 self.sym 为假，则返回 self.val

    def __pos__(self):
        return self  # 返回自身实例

    def __neg__(self):
        return self.__class__(-self.val)  # 返回一个新的 ModularInteger 实例，其值为 -self.val

    @classmethod
    def _get_val(cls, other):
        """Class method to get the value of another instance or convert another value."""
        if isinstance(other, cls):
            return other.val  # 如果 other 是 cls 类型的实例，则返回其 val 属性
        else:
            try:
                return cls.dom.convert(other)  # 尝试将 other 转换为 cls.dom 类型并返回
            except CoercionFailed:
                return None  # 转换失败则返回 None

    def __add__(self, other):
        """Addition operator overload."""
        val = self._get_val(other)

        if val is not None:
            return self.__class__(self.val + val)  # 如果 val 不为 None，则返回一个新的 ModularInteger 实例，其值为 self.val + val
        else:
            return NotImplemented  # 否则返回 NotImplemented

    def __radd__(self, other):
        return self.__add__(other)  # 右加法运算，与 __add__ 方法相同

    def __sub__(self, other):
        """Subtraction operator overload."""
        val = self._get_val(other)

        if val is not None:
            return self.__class__(self.val - val)  # 如果 val 不为 None，则返回一个新的 ModularInteger 实例，其值为 self.val - val
        else:
            return NotImplemented  # 否则返回 NotImplemented

    def __rsub__(self, other):
        return (-self).__add__(other)  # 右减法运算，等同于 -self 加 other

    def __mul__(self, other):
        """Multiplication operator overload."""
        val = self._get_val(other)

        if val is not None:
            return self.__class__(self.val * val)  # 如果 val 不为 None，则返回一个新的 ModularInteger 实例，其值为 self.val * val
        else:
            return NotImplemented  # 否则返回 NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)  # 右乘法运算，与 __mul__ 方法相同
    # 定义特殊方法 __truediv__，处理对象除法操作
    def __truediv__(self, other):
        # 获取其他操作数的数值表示
        val = self._get_val(other)

        # 如果操作数有效，则返回一个新的对象，其值为当前对象值乘以操作数的倒数
        if val is not None:
            return self.__class__(self.val * self._invert(val))
        else:
            return NotImplemented

    # 定义特殊方法 __rtruediv__，处理反向除法操作
    def __rtruediv__(self, other):
        # 返回当前对象调用 invert() 方法后再调用 __mul__ 方法与操作数进行乘法操作的结果
        return self.invert().__mul__(other)

    # 定义特殊方法 __mod__，处理对象取模操作
    def __mod__(self, other):
        # 获取其他操作数的数值表示
        val = self._get_val(other)

        # 如果操作数有效，则返回一个新的对象，其值为当前对象值对操作数取模的结果
        if val is not None:
            return self.__class__(self.val % val)
        else:
            return NotImplemented

    # 定义特殊方法 __rmod__，处理反向取模操作
    def __rmod__(self, other):
        # 获取其他操作数的数值表示
        val = self._get_val(other)

        # 如果操作数有效，则返回一个新的对象，其值为操作数对当前对象值取模的结果
        if val is not None:
            return self.__class__(val % self.val)
        else:
            return NotImplemented

    # 定义特殊方法 __pow__，处理对象的乘幂操作
    def __pow__(self, exp):
        # 如果指数为0，则返回一个新对象，其值为当前对象的单位元素
        if not exp:
            return self.__class__(self.dom.one)

        # 如果指数为负数，则获取当前对象的倒数的值和指数的绝对值
        if exp < 0:
            val, exp = self.invert().val, -exp
        else:
            val = self.val

        # 返回一个新对象，其值为当前对象值的 val 次幂对 mod 取模的结果
        return self.__class__(pow(val, int(exp), self.mod))

    # 定义比较辅助方法 _compare，用于比较当前对象和另一个对象使用给定的操作符 op 进行比较
    def _compare(self, other, op):
        # 获取其他操作数的数值表示
        val = self._get_val(other)

        # 如果操作数无效，则返回 NotImplemented
        if val is None:
            return NotImplemented

        # 返回操作符 op 对当前对象值与操作数值对 mod 取模的比较结果
        return op(self.val, val % self.mod)

    # 定义过时的比较辅助方法 _compare_deprecated，用于比较当前对象和另一个对象使用给定的操作符 op 进行比较，并显示过时警告
    def _compare_deprecated(self, other, op):
        # 获取其他操作数的数值表示
        val = self._get_val(other)

        # 如果操作数无效，则返回 NotImplemented
        if val is None:
            return NotImplemented

        # 发出 sympy 过时警告，指出模整数的有序比较已被弃用
        sympy_deprecation_warning(
            """Ordered comparisons with modular integers are deprecated.

            Use e.g. int(a) < int(b) instead of a < b.
            """,
            deprecated_since_version="1.13",
            active_deprecations_target="modularinteger-compare",
            stacklevel=4,
        )

        # 返回操作符 op 对当前对象值与操作数值对 mod 取模的比较结果
        return op(self.val, val % self.mod)

    # 定义特殊方法 __eq__，处理对象的等于比较操作
    def __eq__(self, other):
        # 使用操作符 eq 比较当前对象和其他对象
        return self._compare(other, operator.eq)

    # 定义特殊方法 __ne__，处理对象的不等于比较操作
    def __ne__(self, other):
        # 使用操作符 ne 比较当前对象和其他对象
        return self._compare(other, operator.ne)

    # 定义特殊方法 __lt__，处理对象的小于比较操作，使用过时的比较方法并显示过时警告
    def __lt__(self, other):
        # 使用操作符 lt 比较当前对象和其他对象，并显示过时警告
        return self._compare_deprecated(other, operator.lt)

    # 定义特殊方法 __le__，处理对象的小于等于比较操作，使用过时的比较方法并显示过时警告
    def __le__(self, other):
        # 使用操作符 le 比较当前对象和其他对象，并显示过时警告
        return self._compare_deprecated(other, operator.le)

    # 定义特殊方法 __gt__，处理对象的大于比较操作，使用过时的比较方法并显示过时警告
    def __gt__(self, other):
        # 使用操作符 gt 比较当前对象和其他对象，并显示过时警告
        return self._compare_deprecated(other, operator.gt)

    # 定义特殊方法 __ge__，处理对象的大于等于比较操作，使用过时的比较方法并显示过时警告
    def __ge__(self, other):
        # 使用操作符 ge 比较当前对象和其他对象，并显示过时警告
        return self._compare_deprecated(other, operator.ge)

    # 定义特殊方法 __bool__，处理对象的布尔转换操作
    def __bool__(self):
        # 返回当前对象值的布尔值
        return bool(self.val)

    # 类方法 _invert，用于计算给定值的倒数，并返回一个新的类实例
    @classmethod
    def _invert(cls, value):
        # 调用 dom 类的 invert 方法计算倒数，使用 mod 属性取模，返回倒数值
        return cls.dom.invert(value, cls.mod)

    # 方法 invert，返回当前对象值的倒数，并返回一个新的类实例
    def invert(self):
        # 调用 _invert 方法计算当前对象值的倒数，并返回新的类实例
        return self.__class__(self._invert(self.val))
# 缓存模式的整数类，以元组为键，存储类型为 ModularInteger 的子类
_modular_integer_cache: dict[tuple[Any, Any, Any], type[ModularInteger]] = {}

# 定义 ModularIntegerFactory 函数，用于创建特定整数模数的自定义类
def ModularIntegerFactory(_mod, _dom, _sym, parent):
    """Create custom class for specific integer modulus."""
    try:
        # 尝试将 _mod 转换为 _dom 对象的类型
        _mod = _dom.convert(_mod)
    except CoercionFailed:
        # 如果转换失败，则标志为不可用
        ok = False
    else:
        # 否则标志为可用
        ok = True

    # 如果不可用或者 _mod 不是正整数，则引发 ValueError
    if not ok or _mod < 1:
        raise ValueError("modulus must be a positive integer, got %s" % _mod)

    # 根据 _mod, _dom, _sym 构建键
    key = _mod, _dom, _sym

    try:
        # 尝试从缓存中获取对应键的类
        cls = _modular_integer_cache[key]
    except KeyError:
        # 如果缓存中不存在，则动态创建一个新的类
        class cls(ModularInteger):
            mod, dom, sym = _mod, _dom, _sym
            _parent = parent

        # 根据 _sym 决定类的命名
        if _sym:
            cls.__name__ = "SymmetricModularIntegerMod%s" % _mod
        else:
            cls.__name__ = "ModularIntegerMod%s" % _mod

        # 将新建的类存入缓存
        _modular_integer_cache[key] = cls

    # 返回创建或者从缓存中获取的类
    return cls
```