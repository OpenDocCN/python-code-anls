# `D:\src\scipysrc\sympy\sympy\sets\ordinals.py`

```
from sympy.core import Basic, Integer
import operator

class OmegaPower(Basic):
    """
    Represents ordinal exponential and multiplication terms, one of the
    building blocks of the :class:`Ordinal` class.
    In ``OmegaPower(a, b)``, ``a`` represents exponent and ``b`` represents multiplicity.
    """

    def __new__(cls, a, b):
        # 如果 b 是整数，则转换为 Integer 类型
        if isinstance(b, int):
            b = Integer(b)
        # 如果 b 不是 Integer 类型或者小于等于 0，则引发类型错误
        if not isinstance(b, Integer) or b <= 0:
            raise TypeError("multiplicity must be a positive integer")

        # 如果 a 不是 Ordinal 类型，则尝试将其转换为 Ordinal 类型
        if not isinstance(a, Ordinal):
            a = Ordinal.convert(a)

        # 创建 OmegaPower 对象
        return Basic.__new__(cls, a, b)

    @property
    def exp(self):
        # 返回 OmegaPower 对象的指数部分
        return self.args[0]

    @property
    def mult(self):
        # 返回 OmegaPower 对象的乘数部分
        return self.args[1]

    def _compare_term(self, other, op):
        # 比较两个 OmegaPower 对象的大小关系
        if self.exp == other.exp:
            return op(self.mult, other.mult)
        else:
            return op(self.exp, other.exp)

    def __eq__(self, other):
        # 比较两个 OmegaPower 对象是否相等
        if not isinstance(other, OmegaPower):
            try:
                other = OmegaPower(0, other)
            except TypeError:
                return NotImplemented
        return self.args == other.args

    def __hash__(self):
        # 返回 OmegaPower 对象的哈希值
        return Basic.__hash__(self)

    def __lt__(self, other):
        # 比较两个 OmegaPower 对象的小于关系
        if not isinstance(other, OmegaPower):
            try:
                other = OmegaPower(0, other)
            except TypeError:
                return NotImplemented
        return self._compare_term(other, operator.lt)


class Ordinal(Basic):
    """
    Represents ordinals in Cantor normal form.

    Internally, this class is just a list of instances of OmegaPower.

    Examples
    ========
    >>> from sympy import Ordinal, OmegaPower
    >>> from sympy.sets.ordinals import omega
    >>> w = omega
    >>> w.is_limit_ordinal
    True
    >>> Ordinal(OmegaPower(w + 1, 1), OmegaPower(3, 2))
    w**(w + 1) + w**3*2
    >>> 3 + w
    w
    >>> (w + 1) * w
    w**2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Ordinal_arithmetic
    """

    def __new__(cls, *terms):
        # 创建 Ordinal 对象
        obj = super().__new__(cls, *terms)
        # 检查所有 OmegaPower 对象的指数部分是否按降序排列
        powers = [i.exp for i in obj.args]
        if not all(powers[i] >= powers[i+1] for i in range(len(powers) - 1)):
            raise ValueError("powers must be in decreasing order")
        return obj

    @property
    def terms(self):
        # 返回 Ordinal 对象的所有 OmegaPower 组成的列表
        return self.args

    @property
    def leading_term(self):
        # 返回 Ordinal 对象的首个 OmegaPower 对象
        if self == ord0:
            raise ValueError("ordinal zero has no leading term")
        return self.terms[0]

    @property
    def trailing_term(self):
        # 返回 Ordinal 对象的末尾 OmegaPower 对象
        if self == ord0:
            raise ValueError("ordinal zero has no trailing term")
        return self.terms[-1]

    @property
    def is_successor_ordinal(self):
        # 判断 Ordinal 对象是否为后继序数
        try:
            return self.trailing_term.exp == ord0
        except ValueError:
            return False

    @property
    # 检查当前 Ordinal 对象是否为极限序数
    def is_limit_ordinal(self):
        try:
            # 返回 trailing_term 的指数是否不等于 ord0 (零序数)
            return not self.trailing_term.exp == ord0
        except ValueError:
            # 如果出现 ValueError，返回 False
            return False

    @property
    # 返回当前 Ordinal 对象的次数 (degree)，即 leading_term 的指数
    def degree(self):
        return self.leading_term.exp

    @classmethod
    # 将整数值转换为 Ordinal 对象
    def convert(cls, integer_value):
        if integer_value == 0:
            # 如果整数值为 0，返回 ord0
            return ord0
        # 否则返回一个 OmegaPower 对象，指数为 integer_value
        return Ordinal(OmegaPower(0, integer_value))

    def __eq__(self, other):
        if not isinstance(other, Ordinal):
            try:
                # 如果 other 不是 Ordinal 对象，尝试将其转换为 Ordinal
                other = Ordinal.convert(other)
            except TypeError:
                # 如果转换失败，返回 NotImplemented
                return NotImplemented
        # 比较两个 Ordinal 对象的 terms 是否相等
        return self.terms == other.terms

    def __hash__(self):
        # 返回 args 的哈希值
        return hash(self.args)

    def __lt__(self, other):
        if not isinstance(other, Ordinal):
            try:
                # 如果 other 不是 Ordinal 对象，尝试将其转换为 Ordinal
                other = Ordinal.convert(other)
            except TypeError:
                # 如果转换失败，返回 NotImplemented
                return NotImplemented
        # 逐项比较 self 和 other 的 terms
        for term_self, term_other in zip(self.terms, other.terms):
            if term_self != term_other:
                # 如果有任何一项不相等，返回其比较结果
                return term_self < term_other
        # 如果所有项相等，则比较 terms 的长度
        return len(self.terms) < len(other.terms)

    def __le__(self, other):
        # 判断是否 self 等于 other 或 self 小于 other
        return (self == other or self < other)

    def __gt__(self, other):
        # 判断是否 self 大于 other
        return not self <= other

    def __ge__(self, other):
        # 判断是否 self 大于等于 other
        return not self < other

    def __str__(self):
        net_str = ""
        plus_count = 0
        if self == ord0:
            # 如果当前对象为 ord0，则返回字符串 'ord0'
            return 'ord0'
        for i in self.terms:
            if plus_count:
                net_str += " + "

            if i.exp == ord0:
                net_str += str(i.mult)
            elif i.exp == 1:
                net_str += 'w'
            elif len(i.exp.terms) > 1 or i.exp.is_limit_ordinal:
                net_str += 'w**(%s)'%i.exp
            else:
                net_str += 'w**%s'%i.exp

            if not i.mult == 1 and not i.exp == ord0:
                net_str += '*%s'%i.mult

            plus_count += 1
        return(net_str)

    __repr__ = __str__

    def __add__(self, other):
        if not isinstance(other, Ordinal):
            try:
                # 如果 other 不是 Ordinal 对象，尝试将其转换为 Ordinal
                other = Ordinal.convert(other)
            except TypeError:
                # 如果转换失败，返回 NotImplemented
                return NotImplemented
        if other == ord0:
            # 如果 other 是 ord0，返回 self
            return self
        a_terms = list(self.terms)
        b_terms = list(other.terms)
        r = len(a_terms) - 1
        b_exp = other.degree
        while r >= 0 and a_terms[r].exp < b_exp:
            r -= 1
        if r < 0:
            terms = b_terms
        elif a_terms[r].exp == b_exp:
            # 如果 a_terms[r] 的指数等于 b_exp，合并这两项
            sum_term = OmegaPower(b_exp, a_terms[r].mult + other.leading_term.mult)
            terms = a_terms[:r] + [sum_term] + b_terms[1:]
        else:
            terms = a_terms[:r+1] + b_terms
        # 返回一个新的 Ordinal 对象，其 terms 为合并后的结果
        return Ordinal(*terms)

    def __radd__(self, other):
        if not isinstance(other, Ordinal):
            try:
                # 如果 other 不是 Ordinal 对象，尝试将其转换为 Ordinal
                other = Ordinal.convert(other)
            except TypeError:
                # 如果转换失败，返回 NotImplemented
                return NotImplemented
        # 返回 other 加上 self 的结果
        return other + self
    # 定义乘法运算符重载方法，用于实现自定义类型 Ordinal 的乘法操作
    def __mul__(self, other):
        # 检查参数 other 是否为 Ordinal 类型，如果不是尝试转换成 Ordinal
        if not isinstance(other, Ordinal):
            try:
                other = Ordinal.convert(other)
            except TypeError:
                # 如果无法转换，返回 NotImplemented 表示不支持此操作
                return NotImplemented
        
        # 如果 self 或 other 中包含 ord0，返回 ord0
        if ord0 in (self, other):
            return ord0
        
        # 获取 self 的指数部分
        a_exp = self.degree
        # 获取 self 的主导项的乘法因子
        a_mult = self.leading_term.mult
        
        # 初始化用于存储结果的列表
        summation = []
        
        # 如果 other 是极限序数
        if other.is_limit_ordinal:
            # 遍历 other 的每一项
            for arg in other.terms:
                # 将 OmegaPower 对象加入 summation 列表，指数为 a_exp + arg.exp，乘法因子为 arg.mult
                summation.append(OmegaPower(a_exp + arg.exp, arg.mult))
        else:
            # 如果 other 不是极限序数，遍历 other 的所有项除了最后一项
            for arg in other.terms[:-1]:
                # 将 OmegaPower 对象加入 summation 列表，指数为 a_exp + arg.exp，乘法因子为 arg.mult
                summation.append(OmegaPower(a_exp + arg.exp, arg.mult))
            
            # 获取 other 的末尾项的乘法因子
            b_mult = other.trailing_term.mult
            # 将 OmegaPower 对象加入 summation 列表，指数为 a_exp，乘法因子为 a_mult * b_mult
            summation.append(OmegaPower(a_exp, a_mult * b_mult))
            # 将 self 的除了第一项外的所有项加入 summation 列表
            summation += list(self.terms[1:])
        
        # 返回一个新的 Ordinal 对象，其项由 summation 列表中的 OmegaPower 对象组成
        return Ordinal(*summation)

    # 定义右乘法运算符重载方法，用于实现自定义类型 Ordinal 的反向乘法操作
    def __rmul__(self, other):
        # 检查参数 other 是否为 Ordinal 类型，如果不是尝试转换成 Ordinal
        if not isinstance(other, Ordinal):
            try:
                other = Ordinal.convert(other)
            except TypeError:
                # 如果无法转换，返回 NotImplemented 表示不支持此操作
                return NotImplemented
        
        # 调用 other 的乘法运算符重载方法，传入 self 作为参数，实现反向乘法
        return other * self

    # 定义乘幂运算符重载方法，用于实现自定义类型 Ordinal 的乘幂操作
    def __pow__(self, other):
        # 如果 self 不等于 omega，返回 NotImplemented 表示不支持此操作
        if not self == omega:
            return NotImplemented
        
        # 返回一个新的 Ordinal 对象，其项为 OmegaPower 对象，指数为 other，乘法因子为 1
        return Ordinal(OmegaPower(other, 1))
class OrdinalZero(Ordinal):
    """The ordinal zero.

    OrdinalZero can be imported as ``ord0``.
    """
    pass


class OrdinalOmega(Ordinal):
    """The ordinal omega which forms the base of all ordinals in cantor normal form.

    OrdinalOmega can be imported as ``omega``.

    Examples
    ========

    >>> from sympy.sets.ordinals import omega
    >>> omega + omega
    w*2
    """

    def __new__(cls):
        # 创建一个新的 OrdinalOmega 对象，调用父类的构造函数
        return Ordinal.__new__(cls)

    @property
    def terms(self):
        # 返回一个包含单个 OmegaPower(1, 1) 的元组
        return (OmegaPower(1, 1),)


# 创建 OrdinalZero 对象，表示零序数
ord0 = OrdinalZero()
# 创建 OrdinalOmega 对象，表示无限序数 omega
omega = OrdinalOmega()
```