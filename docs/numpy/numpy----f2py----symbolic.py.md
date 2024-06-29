# `.\numpy\numpy\f2py\symbolic.py`

```
"""
Fortran/C symbolic expressions

References:
- J3/21-007: Draft Fortran 202x. https://j3-fortran.org/doc/year/21/21-007.pdf

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""

# To analyze Fortran expressions to solve dimensions specifications,
# for instances, we implement a minimal symbolic engine for parsing
# expressions into a tree of expression instances. As a first
# instance, we care only about arithmetic expressions involving
# integers and operations like addition (+), subtraction (-),
# multiplication (*), division (Fortran / is Python //, Fortran // is
# concatenate), and exponentiation (**).  In addition, .pyf files may
# contain C expressions that support here is implemented as well.
#
# TODO: support logical constants (Op.BOOLEAN)
# TODO: support logical operators (.AND., ...)
# TODO: support defined operators (.MYOP., ...)
#

__all__ = ['Expr']

import re
import warnings
from enum import Enum
from math import gcd

class Language(Enum):
    """
    Used as Expr.tostring language argument.
    """
    Python = 0
    Fortran = 1
    C = 2

class Op(Enum):
    """
    Used as Expr op attribute.
    """
    INTEGER = 10
    REAL = 12
    COMPLEX = 15
    STRING = 20
    ARRAY = 30
    SYMBOL = 40
    TERNARY = 100
    APPLY = 200
    INDEXING = 210
    CONCAT = 220
    RELATIONAL = 300
    TERMS = 1000
    FACTORS = 2000
    REF = 3000
    DEREF = 3001

class RelOp(Enum):
    """
    Used in Op.RELATIONAL expression to specify the function part.
    """
    EQ = 1
    NE = 2
    LT = 3
    LE = 4
    GT = 5
    GE = 6

    @classmethod
    def fromstring(cls, s, language=Language.C):
        """
        Convert a string representation of a relational operator to its corresponding RelOp value.
        """
        if language is Language.Fortran:
            return {'.eq.': RelOp.EQ, '.ne.': RelOp.NE,
                    '.lt.': RelOp.LT, '.le.': RelOp.LE,
                    '.gt.': RelOp.GT, '.ge.': RelOp.GE}[s.lower()]
        return {'==': RelOp.EQ, '!=': RelOp.NE, '<': RelOp.LT,
                '<=': RelOp.LE, '>': RelOp.GT, '>=': RelOp.GE}[s]

    def tostring(self, language=Language.C):
        """
        Convert the RelOp value to its string representation based on the specified language.
        """
        if language is Language.Fortran:
            return {RelOp.EQ: '.eq.', RelOp.NE: '.ne.',
                    RelOp.LT: '.lt.', RelOp.LE: '.le.',
                    RelOp.GT: '.gt.', RelOp.GE: '.ge.'}[self]
        return {RelOp.EQ: '==', RelOp.NE: '!=',
                RelOp.LT: '<', RelOp.LE: '<=',
                RelOp.GT: '>', RelOp.GE: '>='}[self]

class ArithOp(Enum):
    """
    Used in Op.APPLY expression to specify the function part.
    """
    POS = 1
    NEG = 2
    ADD = 3
    SUB = 4
    MUL = 5
    DIV = 6
    POW = 7

class OpError(Exception):
    """
    Exception raised for errors in Op operations.
    """
    pass

class Precedence(Enum):
    """
    Used as Expr.tostring precedence argument.
    """
    ATOM = 0
    POWER = 1
    UNARY = 2
    PRODUCT = 3
    SUM = 4
    LT = 6
"""
This code defines a set of classes and enums related to symbolic expressions in Fortran and C. Here's the commented version:


"""
Fortran/C symbolic expressions

References:
- J3/21-007: Draft Fortran 202x. https://j3-fortran.org/doc/year/21/21-007.pdf

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""

# To analyze Fortran expressions to solve dimensions specifications,
# for instances, we implement a minimal symbolic engine for parsing
# expressions into a tree of expression instances. As a first
# instance, we care only about arithmetic expressions involving
# integers and operations like addition (+), subtraction (-),
# multiplication (*), division (Fortran / is Python //, Fortran // is
# concatenate), and exponentiation (**).  In addition, .pyf files may
# contain C expressions that support here is implemented as well.
#
# TODO: support logical constants (Op.BOOLEAN)
# TODO: support logical operators (.AND., ...)
# TODO: support defined operators (.MYOP., ...)
#

__all__ = ['Expr']

import re
import warnings
from enum import Enum
from math import gcd

class Language(Enum):
    """
    Enum to represent programming languages for symbolic expressions.
    """
    Python = 0
    Fortran = 1
    C = 2

class Op(Enum):
    """
    Enum to represent types of operations in symbolic expressions.
    """
    INTEGER = 10
    REAL = 12
    COMPLEX = 15
    STRING = 20
    ARRAY = 30
    SYMBOL = 40
    TERNARY = 100
    APPLY = 200
    INDEXING = 210
    CONCAT = 220
    RELATIONAL = 300
    TERMS = 1000
    FACTORS = 2000
    REF = 3000
    DEREF = 3001

class RelOp(Enum):
    """
    Enum to represent relational operators.
    """
    EQ = 1
    NE = 2
    LT = 3
    LE = 4
    GT = 5
    GE = 6

    @classmethod
    def fromstring(cls, s, language=Language.C):
        """
        Convert a string representation of a relational operator to its corresponding RelOp value.
        """
        if language is Language.Fortran:
            return {'.eq.': RelOp.EQ, '.ne.': RelOp.NE,
                    '.lt.': RelOp.LT, '.le.': RelOp.LE,
                    '.gt.': RelOp.GT, '.ge.': RelOp.GE}[s.lower()]
        return {'==': RelOp.EQ, '!=': RelOp.NE, '<': RelOp.LT,
                '<=': RelOp.LE, '>': RelOp.GT, '>=': RelOp.GE}[s]

    def tostring(self, language=Language.C):
        """
        Convert the RelOp value to its string representation based on the specified language.
        """
        if language is Language.Fortran:
            return {RelOp.EQ: '.eq.', RelOp.NE: '.ne.',
                    RelOp.LT: '.lt.', RelOp.LE: '.le.',
                    RelOp.GT: '.gt.', RelOp.GE: '.ge.'}[self]
        return {RelOp.EQ: '==', RelOp.NE: '!=',
                RelOp.LT: '<', RelOp.LE: '<=',
                RelOp.GT: '>', RelOp.GE: '>='}[self]

class ArithOp(Enum):
    """
    Enum to represent arithmetic operations.
    """
    POS = 1
    NEG = 2
    ADD = 3
    SUB = 4
    MUL = 5
    DIV = 6
    POW = 7

class OpError(Exception):
    """
    Exception raised for errors in Op operations.
    """
    pass

class Precedence(Enum):
    """
    Enum to represent precedence levels in expression parsing.
    """
    ATOM = 0
    POWER = 1
    UNARY = 2
    PRODUCT = 3
    SUM = 4
    LT = 6


This Python module defines enums and classes necessary for handling symbolic expressions in Fortran and C, including various types of operations, relational operators, arithmetic operations, and precedence levels. The code also includes utility methods for converting between enum values and their string representations, specifically tailored for different programming languages.
    # 定义常量 EQ，其值为 7，表示等于操作
    EQ = 7
    # 定义常量 LAND，其值为 11，表示逻辑与操作
    LAND = 11
    # 定义常量 LOR，其值为 12，表示逻辑或操作
    LOR = 12
    # 定义常量 TERNARY，其值为 13，表示三元操作符
    TERNARY = 13
    # 定义常量 ASSIGN，其值为 14，表示赋值操作
    ASSIGN = 14
    # 定义常量 TUPLE，其值为 15，表示元组操作
    TUPLE = 15
    # 定义常量 NONE，其值为 100，表示空值或未定义状态
    NONE = 100
# 定义一个包含单个整数类型 int 的元组
integer_types = (int,)
# 定义一个包含整数和浮点数类型 int, float 的元组
number_types = (int, float)


def _pairs_add(d, k, v):
    # 内部实用方法，用于更新字典 d 中键 k 对应的值 v
    c = d.get(k)
    if c is None:
        d[k] = v
    else:
        # 如果 c 不为 None，则将其增加 v，更新到字典中
        c = c + v
        # 如果更新后的值 c 不为 0，则更新字典中的键 k
        if c:
            d[k] = c
        else:
            # 如果 c 为 0，则从字典中删除键 k
            del d[k]


class ExprWarning(UserWarning):
    # 定义一个继承自 UserWarning 的 ExprWarning 类
    pass


def ewarn(message):
    # 发出一个 ExprWarning 警告，警告消息为 message
    warnings.warn(message, ExprWarning, stacklevel=2)


class Expr:
    """Represents a Fortran expression as a op-data pair.

    Expr instances are hashable and sortable.
    """
    
    @staticmethod
    def parse(s, language=Language.C):
        """Parse a Fortran expression to a Expr.

        解析 Fortran 表达式 s 到一个 Expr 对象。
        """
        return fromstring(s, language=language)
    # 初始化方法，接受操作符 op 和数据 data 作为参数
    def __init__(self, op, data):
        # 确保 op 是 Op 类的实例
        assert isinstance(op, Op)

        # 对不同的操作符进行不同的数据有效性检查
        if op is Op.INTEGER:
            # 如果操作符是 Op.INTEGER，则数据应为一个包含两个元素的元组，第一个是数值对象，第二个是种类值（默认为4）
            assert isinstance(data, tuple) and len(data) == 2
            assert isinstance(data[0], int)
            assert isinstance(data[1], (int, str)), data
        elif op is Op.REAL:
            # 如果操作符是 Op.REAL，则数据应为一个包含两个元素的元组，第一个是浮点数对象，第二个是种类值（默认为4）
            assert isinstance(data, tuple) and len(data) == 2
            assert isinstance(data[0], float)
            assert isinstance(data[1], (int, str)), data
        elif op is Op.COMPLEX:
            # 如果操作符是 Op.COMPLEX，则数据应为一个包含两个元素的元组，都是常量表达式
            assert isinstance(data, tuple) and len(data) == 2
        elif op is Op.STRING:
            # 如果操作符是 Op.STRING，则数据应为一个包含两个元素的元组，第一个是带引号的字符串，第二个是种类值（默认为1）
            assert isinstance(data, tuple) and len(data) == 2
            # 确保第一个元素是字符串且以双引号或单引号包围，或者是一对 '@@'
            assert (isinstance(data[0], str)
                    and data[0][::len(data[0])-1] in ('""', "''", '@@'))
            assert isinstance(data[1], (int, str)), data
        elif op is Op.SYMBOL:
            # 如果操作符是 Op.SYMBOL，则数据可以是任意可散列对象
            assert hash(data) is not None
        elif op in (Op.ARRAY, Op.CONCAT):
            # 如果操作符是 Op.ARRAY 或 Op.CONCAT，则数据应为一个表达式的元组
            assert isinstance(data, tuple)
            assert all(isinstance(item, Expr) for item in data), data
        elif op in (Op.TERMS, Op.FACTORS):
            # 如果操作符是 Op.TERMS 或 Op.FACTORS，则数据应为一个字典，其值为非零整数的 Python 字典
            assert isinstance(data, dict)
        elif op is Op.APPLY:
            # 如果操作符是 Op.APPLY，则数据应为一个包含三个元素的元组，第一个元素是函数对象，第二个元素是操作数元组，第三个元素是关键字操作数字典
            assert isinstance(data, tuple) and len(data) == 3
            # 确保函数对象可散列
            assert hash(data[0]) is not None
            assert isinstance(data[1], tuple)
            assert isinstance(data[2], dict)
        elif op is Op.INDEXING:
            # 如果操作符是 Op.INDEXING，则数据应为一个包含两个元素的元组，第一个元素是对象，第二个元素是索引
            assert isinstance(data, tuple) and len(data) == 2
            # 确保对象可散列
            assert hash(data[0]) is not None
        elif op is Op.TERNARY:
            # 如果操作符是 Op.TERNARY，则数据应为一个包含三个元素的元组，分别是条件表达式、表达式1和表达式2
            assert isinstance(data, tuple) and len(data) == 3
        elif op in (Op.REF, Op.DEREF):
            # 如果操作符是 Op.REF 或 Op.DEREF，则数据应为一个 Expr 实例
            assert isinstance(data, Expr)
        elif op is Op.RELATIONAL:
            # 如果操作符是 Op.RELATIONAL，则数据应为一个包含三个元素的元组，分别是关系运算符、左操作数和右操作数
            assert isinstance(data, tuple) and len(data) == 3
        else:
            # 如果操作符未知或缺少有效性检查，则引发 NotImplementedError 异常
            raise NotImplementedError(
                f'unknown op or missing sanity check: {op}')

        # 将操作符和数据存储在对象实例的属性中
        self.op = op
        self.data = data
    def __eq__(self, other):
        # 检查是否是同一类型的表达式，并比较操作符和数据是否相等
        return (isinstance(other, Expr)
                and self.op is other.op
                and self.data == other.data)

    def __hash__(self):
        # 根据表达式的操作符和数据生成哈希值
        if self.op in (Op.TERMS, Op.FACTORS):
            data = tuple(sorted(self.data.items()))
        elif self.op is Op.APPLY:
            data = self.data[:2] + tuple(sorted(self.data[2].items()))
        else:
            data = self.data
        return hash((self.op, data))

    def __lt__(self, other):
        # 比较表达式的大小
        if isinstance(other, Expr):
            if self.op is not other.op:
                return self.op.value < other.op.value
            if self.op in (Op.TERMS, Op.FACTORS):
                return (tuple(sorted(self.data.items()))
                        < tuple(sorted(other.data.items())))
            if self.op is Op.APPLY:
                if self.data[:2] != other.data[:2]:
                    return self.data[:2] < other.data[:2]
                return tuple(sorted(self.data[2].items())) < tuple(
                    sorted(other.data[2].items()))
            return self.data < other.data
        return NotImplemented

    def __le__(self, other):
        # 检查表达式是否小于或等于另一个表达式
        return self == other or self < other

    def __gt__(self, other):
        # 检查表达式是否大于另一个表达式
        return not (self <= other)

    def __ge__(self, other):
        # 检查表达式是否大于或等于另一个表达式
        return not (self < other)

    def __repr__(self):
        # 返回表达式的规范字符串表示
        return f'{type(self).__name__}({self.op}, {self.data!r})'

    def __str__(self):
        # 返回表达式的字符串表示
        return self.tostring()

    def __pos__(self):
        # 正号操作，返回自身
        return self

    def __neg__(self):
        # 负号操作，返回自身乘以-1
        return self * -1

    def __add__(self, other):
        # 加法操作
        other = as_expr(other)
        if isinstance(other, Expr):
            if self.op is other.op:
                if self.op in (Op.INTEGER, Op.REAL):
                    # 对整数或实数执行加法
                    return as_number(
                        self.data[0] + other.data[0],
                        max(self.data[1], other.data[1]))
                if self.op is Op.COMPLEX:
                    # 对复数执行加法
                    r1, i1 = self.data
                    r2, i2 = other.data
                    return as_complex(r1 + r2, i1 + i2)
                if self.op is Op.TERMS:
                    # 对项执行加法
                    r = Expr(self.op, dict(self.data))
                    for k, v in other.data.items():
                        _pairs_add(r.data, k, v)
                    return normalize(r)
            if self.op is Op.COMPLEX and other.op in (Op.INTEGER, Op.REAL):
                return self + as_complex(other)
            elif self.op in (Op.INTEGER, Op.REAL) and other.op is Op.COMPLEX:
                return as_complex(self) + other
            elif self.op is Op.REAL and other.op is Op.INTEGER:
                return self + as_real(other, kind=self.data[1])
            elif self.op is Op.INTEGER and other.op is Op.REAL:
                return as_real(self, kind=other.data[1]) + other
            return as_terms(self) + as_terms(other)
        return NotImplemented
    # 定义特殊方法 __radd__，实现右加运算
    def __radd__(self, other):
        # 如果 other 是数值类型，则将其转换为表达式后执行加法运算
        if isinstance(other, number_types):
            return as_number(other) + self
        # 如果 other 不是数值类型，则返回 NotImplemented，表示无法处理此操作
        return NotImplemented

    # 定义特殊方法 __sub__，实现减法运算
    def __sub__(self, other):
        # 返回 self 与 -other 的加法结果，实现减法运算
        return self + (-other)

    # 定义特殊方法 __rsub__，实现右减运算
    def __rsub__(self, other):
        # 如果 other 是数值类型，则将其转换为表达式后执行减法运算
        if isinstance(other, number_types):
            return as_number(other) - self
        # 如果 other 不是数值类型，则返回 NotImplemented，表示无法处理此操作
        return NotImplemented

    # 定义特殊方法 __mul__，实现乘法运算
    def __mul__(self, other):
        # 将 other 转换为表达式类型
        other = as_expr(other)
        # 如果 other 是表达式对象
        if isinstance(other, Expr):
            # 如果两个表达式的操作符相同
            if self.op is other.op:
                # 处理不同操作符的乘法运算
                if self.op in (Op.INTEGER, Op.REAL):
                    # 对整数和实数执行乘法运算
                    return as_number(self.data[0] * other.data[0],
                                     max(self.data[1], other.data[1]))
                elif self.op is Op.COMPLEX:
                    # 对复数执行乘法运算
                    r1, i1 = self.data
                    r2, i2 = other.data
                    return as_complex(r1 * r2 - i1 * i2, r1 * i2 + r2 * i1)

                # 对因子操作进行乘法运算
                if self.op is Op.FACTORS:
                    r = Expr(self.op, dict(self.data))
                    for k, v in other.data.items():
                        _pairs_add(r.data, k, v)
                    return normalize(r)
                elif self.op is Op.TERMS:
                    r = Expr(self.op, {})
                    for t1, c1 in self.data.items():
                        for t2, c2 in other.data.items():
                            _pairs_add(r.data, t1 * t2, c1 * c2)
                    return normalize(r)

            # 处理复数与整数/实数之间的乘法运算
            if self.op is Op.COMPLEX and other.op in (Op.INTEGER, Op.REAL):
                return self * as_complex(other)
            elif other.op is Op.COMPLEX and self.op in (Op.INTEGER, Op.REAL):
                return as_complex(self) * other
            elif self.op is Op.REAL and other.op is Op.INTEGER:
                return self * as_real(other, kind=self.data[1])
            elif self.op is Op.INTEGER and other.op is Op.REAL:
                return as_real(self, kind=other.data[1]) * other

            # 处理项操作之间的乘法运算
            if self.op is Op.TERMS:
                return self * as_terms(other)
            elif other.op is Op.TERMS:
                return as_terms(self) * other

            # 如果均不匹配上述条件，则将两个表达式都转换为因子操作进行乘法运算
            return as_factors(self) * as_factors(other)
        # 如果 other 不是表达式类型，则返回 NotImplemented，表示无法处理此操作
        return NotImplemented

    # 定义特殊方法 __rmul__，实现右乘运算
    def __rmul__(self, other):
        # 如果 other 是数值类型，则将其转换为表达式后执行乘法运算
        if isinstance(other, number_types):
            return as_number(other) * self
        # 如果 other 不是数值类型，则返回 NotImplemented，表示无法处理此操作
        return NotImplemented
    def __pow__(self, other):
        other = as_expr(other)  # 将参数转换为表达式对象
        if isinstance(other, Expr):  # 如果参数是表达式对象
            if other.op is Op.INTEGER:  # 如果表达式操作是整数
                exponent = other.data[0]  # 获取整数指数
                if exponent == 0:
                    return as_number(1)  # 返回 1，任何数的 0 次方为 1
                if exponent == 1:
                    return self  # 返回自身，任何数的 1 次方为其本身
                if exponent > 0:
                    if self.op is Op.FACTORS:  # 如果自身是因子类型的表达式对象
                        r = Expr(self.op, {})  # 创建一个新的因子类型表达式对象
                        for k, v in self.data.items():
                            r.data[k] = v * exponent  # 对每个因子进行指数运算
                        return normalize(r)  # 规范化并返回结果
                    return self * (self ** (exponent - 1))  # 递归计算正整数次方
                elif exponent != -1:
                    return (self ** (-exponent)) ** -1  # 处理负指数
                return Expr(Op.FACTORS, {self: exponent})  # 返回带有自身和指数的新因子表达式
            return as_apply(ArithOp.POW, self, other)  # 如果不是整数操作，则应用 POW 操作符
        return NotImplemented  # 如果参数不是表达式对象，则返回 Not Implemented

    def __truediv__(self, other):
        other = as_expr(other)  # 将参数转换为表达式对象
        if isinstance(other, Expr):  # 如果参数是表达式对象
            # Fortran 的 / 操作符与 Python 的 / 不同:
            # - 对于整数操作数，/ 是截断操作
            return normalize(as_apply(ArithOp.DIV, self, other))  # 应用 DIV 操作符并规范化结果
        return NotImplemented  # 如果参数不是表达式对象，则返回 Not Implemented

    def __rtruediv__(self, other):
        other = as_expr(other)  # 将参数转换为表达式对象
        if isinstance(other, Expr):  # 如果参数是表达式对象
            return other / self  # 执行反向真除操作
        return NotImplemented  # 如果参数不是表达式对象，则返回 Not Implemented

    def __floordiv__(self, other):
        other = as_expr(other)  # 将参数转换为表达式对象
        if isinstance(other, Expr):  # 如果参数是表达式对象
            # Fortran 的 // 操作符与 Python 的 // 不同:
            # - 对于字符串操作数，// 是连接操作
            return normalize(Expr(Op.CONCAT, (self, other)))  # 创建连接操作的表达式并规范化结果
        return NotImplemented  # 如果参数不是表达式对象，则返回 Not Implemented

    def __rfloordiv__(self, other):
        other = as_expr(other)  # 将参数转换为表达式对象
        if isinstance(other, Expr):  # 如果参数是表达式对象
            return other // self  # 执行反向地板除法操作
        return NotImplemented  # 如果参数不是表达式对象，则返回 Not Implemented

    def __call__(self, *args, **kwargs):
        # 在 Fortran 中，括号 () 用于函数调用和索引操作。
        #
        # TODO: 实现一个方法来决定何时 __call__ 应该返回一个索引表达式。
        return as_apply(self, *map(as_expr, args),
                        **dict((k, as_expr(v)) for k, v in kwargs.items()))

    def __getitem__(self, index):
        # 用于支持可能包含在 .pyf 文件中的 C 索引操作。
        index = as_expr(index)  # 将索引转换为表达式对象
        if not isinstance(index, tuple):
            index = index,  # 如果索引不是元组，则转换为单元素元组
        if len(index) > 1:
            ewarn(f'C-index should be a single expression but got `{index}`')  # 发出警告，C 索引应该是单一表达式
        return Expr(Op.INDEXING, (self,) + index)  # 创建索引操作的表达式并返回
    def traverse(self, visit, *args, **kwargs):
        """遍历表达式树并应用访问函数。

        访问函数应用于带有给定 args 和 kwargs 的表达式。

        如果访问函数返回非空表达式，则遍历调用返回该表达式；
        否则，返回一个新的经过标准化处理的表达式，其中包含遍历-访问子表达式。
        """
        # 调用访问函数，并根据返回结果判断是否直接返回
        result = visit(self, *args, **kwargs)
        if result is not None:
            return result

        # 根据表达式的操作类型进行不同的处理
        if self.op in (Op.INTEGER, Op.REAL, Op.STRING, Op.SYMBOL):
            return self
        elif self.op in (Op.COMPLEX, Op.ARRAY, Op.CONCAT, Op.TERNARY):
            # 对于复杂操作类型，递归遍历处理每个子项
            return normalize(Expr(self.op, tuple(
                item.traverse(visit, *args, **kwargs)
                for item in self.data)))
        elif self.op in (Op.TERMS, Op.FACTORS):
            # 对于项或因子操作类型，遍历处理字典中的每一项
            data = {}
            for k, v in self.data.items():
                k = k.traverse(visit, *args, **kwargs)
                v = (v.traverse(visit, *args, **kwargs)
                     if isinstance(v, Expr) else v)
                if k in data:
                    v = data[k] + v
                data[k] = v
            return normalize(Expr(self.op, data))
        elif self.op is Op.APPLY:
            # 对于函数应用操作类型，遍历处理对象、操作数和关键字操作数
            obj = self.data[0]
            func = (obj.traverse(visit, *args, **kwargs)
                    if isinstance(obj, Expr) else obj)
            operands = tuple(operand.traverse(visit, *args, **kwargs)
                             for operand in self.data[1])
            kwoperands = dict((k, v.traverse(visit, *args, **kwargs))
                              for k, v in self.data[2].items())
            return normalize(Expr(self.op, (func, operands, kwoperands)))
        elif self.op is Op.INDEXING:
            # 对于索引操作类型，遍历处理对象和索引
            obj = self.data[0]
            obj = (obj.traverse(visit, *args, **kwargs)
                   if isinstance(obj, Expr) else obj)
            indices = tuple(index.traverse(visit, *args, **kwargs)
                            for index in self.data[1:])
            return normalize(Expr(self.op, (obj,) + indices))
        elif self.op in (Op.REF, Op.DEREF):
            # 对于引用和解引用操作类型，遍历处理数据
            return normalize(Expr(self.op,
                                  self.data.traverse(visit, *args, **kwargs)))
        elif self.op is Op.RELATIONAL:
            # 对于关系操作类型，遍历处理左操作数和右操作数
            rop, left, right = self.data
            left = left.traverse(visit, *args, **kwargs)
            right = right.traverse(visit, *args, **kwargs)
            return normalize(Expr(self.op, (rop, left, right)))
        
        # 若操作类型未被处理，则抛出未实现错误
        raise NotImplementedError(f'traverse method for {self.op}')

    def contains(self, other):
        """检查 self 是否包含 other。"""
        found = []

        def visit(expr, found=found):
            if found:
                return expr
            elif expr == other:
                found.append(1)
                return expr

        self.traverse(visit)

        # 返回是否找到匹配项的布尔值
        return len(found) != 0
    def symbols(self):
        """Return a set of symbols contained in self.
        """
        found = set()  # 初始化一个空集合用于存放找到的符号

        def visit(expr, found=found):
            if expr.op is Op.SYMBOL:  # 如果表达式的操作是符号类型
                found.add(expr)  # 将符号添加到集合中

        self.traverse(visit)  # 调用对象的 traverse 方法，传入 visit 函数进行遍历

        return found  # 返回找到的符号集合

    def polynomial_atoms(self):
        """Return a set of expressions used as atoms in polynomial self.
        """
        found = set()  # 初始化一个空集合用于存放找到的原子表达式

        def visit(expr, found=found):
            if expr.op is Op.FACTORS:  # 如果表达式的操作是因子类型
                for b in expr.data:  # 遍历因子表达式列表
                    b.traverse(visit)  # 递归调用 traverse 方法继续遍历子表达式
                return expr  # 返回当前表达式对象
            if expr.op in (Op.TERMS, Op.COMPLEX):  # 如果操作是项或复合类型，则直接返回
                return
            if expr.op is Op.APPLY and isinstance(expr.data[0], ArithOp):  # 如果是应用操作且第一个数据项是算术操作
                if expr.data[0] is ArithOp.POW:  # 如果是乘方操作
                    expr.data[1][0].traverse(visit)  # 遍历乘方操作的指数部分
                    return expr  # 返回当前表达式对象
                return
            if expr.op in (Op.INTEGER, Op.REAL):  # 如果是整数或实数类型
                return expr  # 直接返回表达式对象

            found.add(expr)  # 将当前表达式对象添加到集合中

            if expr.op in (Op.INDEXING, Op.APPLY):  # 如果是索引或应用操作
                return expr  # 返回当前表达式对象

        self.traverse(visit)  # 调用对象的 traverse 方法，传入 visit 函数进行遍历

        return found  # 返回找到的原子表达式集合

    def linear_solve(self, symbol):
        """Return a, b such that a * symbol + b == self.

        If self is not linear with respect to symbol, raise RuntimeError.
        """
        b = self.substitute({symbol: as_number(0)})  # 将 symbol 替换为 0，得到常数项 b
        ax = self - b  # 计算 self 减去常数项 b，得到 a * symbol
        a = ax.substitute({symbol: as_number(1)})  # 将 symbol 替换为 1，得到系数 a

        zero, _ = as_numer_denom(a * symbol - ax)  # 计算 a * symbol - ax 的分子，并获取分母

        if zero != as_number(0):  # 如果分子不为零
            raise RuntimeError(f'not a {symbol}-linear equation:'
                               f' {a} * {symbol} + {b} == {self}')  # 抛出运行时错误，指示不是线性方程
        return a, b  # 返回计算得到的系数 a 和常数项 b
# 将给定对象标准化并应用基本的求值方法
def normalize(obj):
    # 如果对象不是表达式类型，则直接返回该对象
    if not isinstance(obj, Expr):
        return obj

    # 处理表达式类型为 TERMS 的情况
    if obj.op is Op.TERMS:
        d = {}
        # 遍历表达式数据中的每个项和系数
        for t, c in obj.data.items():
            # 如果系数为0，则跳过该项
            if c == 0:
                continue
            # 如果项的操作类型为 COMPLEX 且系数不为1，则更新项和系数
            if t.op is Op.COMPLEX and c != 1:
                t = t * c
                c = 1
            # 如果项的操作类型为 TERMS，则进一步处理其数据项
            if t.op is Op.TERMS:
                for t1, c1 in t.data.items():
                    _pairs_add(d, t1, c1 * c)
            else:
                _pairs_add(d, t, c)
        # 如果字典 d 为空，则返回数字0
        if len(d) == 0:
            # TODO: 确定正确的类型
            return as_number(0)
        # 如果字典 d 只包含一个项，则返回该项
        elif len(d) == 1:
            (t, c), = d.items()
            if c == 1:
                return t
        # 否则返回一个表达式对象，操作类型为 TERMS，数据为字典 d
        return Expr(Op.TERMS, d)

    # 处理表达式类型为 FACTORS 的情况
    if obj.op is Op.FACTORS:
        coeff = 1
        d = {}
        # 遍历表达式数据中的每个因子和指数
        for b, e in obj.data.items():
            # 如果指数为0，则跳过该因子
            if e == 0:
                continue
            # 如果因子的操作类型为 TERMS，且指数为正整数大于1，则展开整数幂
            if b.op is Op.TERMS and isinstance(e, integer_types) and e > 1:
                b = b * (b ** (e - 1))
                e = 1

            # 处理因子操作类型为 INTEGER 或 REAL 的情况
            if b.op in (Op.INTEGER, Op.REAL):
                if e == 1:
                    coeff *= b.data[0]
                elif e > 0:
                    coeff *= b.data[0] ** e
                else:
                    _pairs_add(d, b, e)
            # 处理因子操作类型为 FACTORS 的情况
            elif b.op is Op.FACTORS:
                if e > 0 and isinstance(e, integer_types):
                    for b1, e1 in b.data.items():
                        _pairs_add(d, b1, e1 * e)
                else:
                    _pairs_add(d, b, e)
            else:
                _pairs_add(d, b, e)
        
        # 如果字典 d 为空或者系数为0，则返回数字0
        if len(d) == 0 or coeff == 0:
            # TODO: 确定正确的类型
            assert isinstance(coeff, number_types)
            return as_number(coeff)
        # 如果字典 d 只包含一个项，则根据情况返回该项或者表达式对象
        elif len(d) == 1:
            (b, e), = d.items()
            if e == 1:
                t = b
            else:
                t = Expr(Op.FACTORS, d)
            if coeff == 1:
                return t
            return Expr(Op.TERMS, {t: coeff})
        # 根据系数是否为1返回表达式对象，操作类型为 FACTORS 或 TERMS
        elif coeff == 1:
            return Expr(Op.FACTORS, d)
        else:
            return Expr(Op.TERMS, {Expr(Op.FACTORS, d): coeff})
    # 如果对象的操作符是 Op.APPLY，且第一个数据元素是 ArithOp.DIV
    if obj.op is Op.APPLY and obj.data[0] is ArithOp.DIV:
        # 将被除数和除数分配给相应变量
        dividend, divisor = obj.data[1]
        # 将被除数和除数分别转换为项和系数
        t1, c1 = as_term_coeff(dividend)
        t2, c2 = as_term_coeff(divisor)
        
        # 如果系数 c1 和 c2 都是整数类型，则计算它们的最大公约数，并进行化简
        if isinstance(c1, integer_types) and isinstance(c2, integer_types):
            g = gcd(c1, c2)
            c1, c2 = c1//g, c2//g
        else:
            # 如果 c1 或 c2 不是整数类型，则进行浮点数的除法
            c1, c2 = c1/c2, 1
        
        # 如果 t1 的操作符是 Op.APPLY，且第一个数据元素是 ArithOp.DIV
        if t1.op is Op.APPLY and t1.data[0] is ArithOp.DIV:
            # 计算分子和分母
            numer = t1.data[1][0] * c1
            denom = t1.data[1][1] * t2 * c2
            # 返回应用 ArithOp.DIV 操作符的表达式
            return as_apply(ArithOp.DIV, numer, denom)
        
        # 如果 t2 的操作符是 Op.APPLY，且第一个数据元素是 ArithOp.DIV
        if t2.op is Op.APPLY and t2.data[0] is ArithOp.DIV:
            # 计算分子和分母
            numer = t2.data[1][1] * t1 * c1
            denom = t2.data[1][0] * c2
            # 返回应用 ArithOp.DIV 操作符的表达式
            return as_apply(ArithOp.DIV, numer, denom)
        
        # 将 t1 和 t2 分解为因子，并构建因子字典
        d = dict(as_factors(t1).data)
        for b, e in as_factors(t2).data.items():
            _pairs_add(d, b, -e)
        
        # 分别构建分子和分母的字典，正负指数对应分子和分母
        numer, denom = {}, {}
        for b, e in d.items():
            if e > 0:
                numer[b] = e
            else:
                denom[b] = -e
        
        # 将分子和分母字典构建为 Op.FACTORS 操作的表达式，然后进行归一化
        numer = normalize(Expr(Op.FACTORS, numer)) * c1
        denom = normalize(Expr(Op.FACTORS, denom)) * c2
        
        # 如果分母是整数或实数类型，并且值为 1，则返回分子
        if denom.op in (Op.INTEGER, Op.REAL) and denom.data[0] == 1:
            # TODO: denom kind not used
            return numer
        
        # 返回应用 ArithOp.DIV 操作符的表达式
        return as_apply(ArithOp.DIV, numer, denom)
    
    # 如果对象的操作符是 Op.CONCAT
    if obj.op is Op.CONCAT:
        # 初始化列表，包含第一个元素
        lst = [obj.data[0]]
        # 遍历剩余元素
        for s in obj.data[1:]:
            last = lst[-1]
            # 如果最后一个元素和当前元素都是字符串，且可以拼接
            if (
                    last.op is Op.STRING
                    and s.op is Op.STRING
                    and last.data[0][0] in '"\''
                    and s.data[0][0] == last.data[0][-1]
            ):
                # 创建新的字符串表达式，并替换最后一个元素
                new_last = as_string(last.data[0][:-1] + s.data[0][1:],
                                     max(last.data[1], s.data[1]))
                lst[-1] = new_last
            else:
                # 否则直接添加当前元素到列表中
                lst.append(s)
        
        # 如果列表只有一个元素，则返回该元素
        if len(lst) == 1:
            return lst[0]
        
        # 返回 Op.CONCAT 操作的表达式，包含所有拼接后的元素
        return Expr(Op.CONCAT, tuple(lst))
    
    # 如果对象的操作符是 Op.TERNARY
    if obj.op is Op.TERNARY:
        # 将条件、表达式1和表达式2进行归一化处理
        cond, expr1, expr2 = map(normalize, obj.data)
        # 如果条件是整数类型，则根据条件的值返回相应表达式
        if cond.op is Op.INTEGER:
            return expr1 if cond.data[0] else expr2
        # 否则返回 Op.TERNARY 操作的表达式，包含归一化后的条件和表达式
        return Expr(Op.TERNARY, (cond, expr1, expr2))
    
    # 默认情况下，直接返回对象本身
    return obj
# 将非 Expr 类型的对象转换为 Expr 对象。
def as_expr(obj):
    if isinstance(obj, complex):
        # 如果 obj 是复数类型，则调用 as_complex 转换为复数表达式
        return as_complex(obj.real, obj.imag)
    if isinstance(obj, number_types):
        # 如果 obj 是数字类型，则调用 as_number 转换为数字表达式
        return as_number(obj)
    if isinstance(obj, str):
        # 如果 obj 是字符串类型，则将其应用 repr 函数转换为带有引号的字符串表达式
        return as_string(repr(obj))
    if isinstance(obj, tuple):
        # 如果 obj 是元组类型，则逐个调用 as_expr 转换为表达式元组
        return tuple(map(as_expr, obj))
    # 对于其他类型的对象，直接返回该对象
    return obj


# 将对象转换为 SYMBOL 表达式（变量或未解析表达式）。
def as_symbol(obj):
    return Expr(Op.SYMBOL, obj)


# 将对象转换为 INTEGER 或 REAL 常量。
def as_number(obj, kind=4):
    if isinstance(obj, int):
        # 如果 obj 是整数类型，则返回 INTEGER 表达式
        return Expr(Op.INTEGER, (obj, kind))
    if isinstance(obj, float):
        # 如果 obj 是浮点数类型，则返回 REAL 表达式
        return Expr(Op.REAL, (obj, kind))
    if isinstance(obj, Expr):
        # 如果 obj 已经是表达式类型，则检查其类型，如果是 INTEGER 或 REAL 直接返回
        if obj.op in (Op.INTEGER, Op.REAL):
            return obj
    # 对于无法转换的情况，抛出 OpError 异常
    raise OpError(f'cannot convert {obj} to INTEGER or REAL constant')


# 将对象转换为 INTEGER 常量。
def as_integer(obj, kind=4):
    if isinstance(obj, int):
        return Expr(Op.INTEGER, (obj, kind))
    if isinstance(obj, Expr):
        if obj.op is Op.INTEGER:
            return obj
    raise OpError(f'cannot convert {obj} to INTEGER constant')


# 将对象转换为 REAL 常量。
def as_real(obj, kind=4):
    if isinstance(obj, int):
        # 如果 obj 是整数类型，则转换为 REAL 表达式
        return Expr(Op.REAL, (float(obj), kind))
    if isinstance(obj, float):
        # 如果 obj 是浮点数类型，则转换为 REAL 表达式
        return Expr(Op.REAL, (obj, kind))
    if isinstance(obj, Expr):
        # 如果 obj 是表达式类型，则根据其类型进行转换
        if obj.op is Op.REAL:
            return obj
        elif obj.op is Op.INTEGER:
            # 如果 obj 是 INTEGER 表达式，则将其转换为 REAL 表达式
            return Expr(Op.REAL, (float(obj.data[0]), kind))
    # 对于无法转换的情况，抛出 OpError 异常
    raise OpError(f'cannot convert {obj} to REAL constant')


# 将对象转换为 STRING 表达式（字符串字面常量）。
def as_string(obj, kind=1):
    return Expr(Op.STRING, (obj, kind))


# 将对象转换为 ARRAY 表达式（数组常量）。
def as_array(obj):
    if isinstance(obj, Expr):
        # 如果 obj 已经是表达式类型，则将其转换为元组
        obj = obj,
    return Expr(Op.ARRAY, obj)


# 将对象转换为 COMPLEX 表达式（复数字面常量）。
def as_complex(real, imag=0):
    return Expr(Op.COMPLEX, (as_expr(real), as_expr(imag)))


# 将对象转换为 APPLY 表达式（函数调用、构造函数等）。
def as_apply(func, *args, **kwargs):
    return Expr(Op.APPLY,
                (func, tuple(map(as_expr, args)),
                 dict((k, as_expr(v)) for k, v in kwargs.items())))


# 将对象转换为 TERNARY 表达式（三元表达式 cond?expr1:expr2）。
def as_ternary(cond, expr1, expr2):
    return Expr(Op.TERNARY, (cond, expr1, expr2))


# 将对象转换为引用表达式。
def as_ref(expr):
    return Expr(Op.REF, expr)


# 将对象转换为解引用表达式。
def as_deref(expr):
    return Expr(Op.DEREF, expr)


# 返回等于（==）关系表达式。
def as_eq(left, right):
    return Expr(Op.RELATIONAL, (RelOp.EQ, left, right))
# 返回一个不等于关系的表达式对象
def as_ne(left, right):
    return Expr(Op.RELATIONAL, (RelOp.NE, left, right))

# 返回一个小于关系的表达式对象
def as_lt(left, right):
    return Expr(Op.RELATIONAL, (RelOp.LT, left, right))

# 返回一个小于等于关系的表达式对象
def as_le(left, right):
    return Expr(Op.RELATIONAL, (RelOp.LE, left, right))

# 返回一个大于关系的表达式对象
def as_gt(left, right):
    return Expr(Op.RELATIONAL, (RelOp.GT, left, right))

# 返回一个大于等于关系的表达式对象
def as_ge(left, right):
    return Expr(Op.RELATIONAL, (RelOp.GE, left, right))

# 将给定的表达式对象转换为TERMS表达式对象
def as_terms(obj):
    """Return expression as TERMS expression.
    """
    if isinstance(obj, Expr):
        obj = normalize(obj)
        # 如果表达式已经是TERMS类型，则直接返回
        if obj.op is Op.TERMS:
            return obj
        # 如果表达式是INTEGER类型，则转换为TERMS类型
        if obj.op is Op.INTEGER:
            return Expr(Op.TERMS, {as_integer(1, obj.data[1]): obj.data[0]})
        # 如果表达式是REAL类型，则转换为TERMS类型
        if obj.op is Op.REAL:
            return Expr(Op.TERMS, {as_real(1, obj.data[1]): obj.data[0]})
        # 否则，默认将表达式转换为一个包含单个项的TERMS表达式
        return Expr(Op.TERMS, {obj: 1})
    # 如果不是Expr类型，则抛出异常
    raise OpError(f'cannot convert {type(obj)} to terms Expr')

# 将给定的表达式对象转换为FACTORS表达式对象
def as_factors(obj):
    """Return expression as FACTORS expression.
    """
    if isinstance(obj, Expr):
        obj = normalize(obj)
        # 如果表达式已经是FACTORS类型，则直接返回
        if obj.op is Op.FACTORS:
            return obj
        # 如果表达式是TERMS类型，且只包含一个项，则转换为FACTORS类型
        if obj.op is Op.TERMS:
            if len(obj.data) == 1:
                (term, coeff), = obj.data.items()
                if coeff == 1:
                    return Expr(Op.FACTORS, {term: 1})
                return Expr(Op.FACTORS, {term: 1, Expr.number(coeff): 1})
        # 如果表达式是APPLY类型，并且是除法操作，则转换为FACTORS类型
        if (obj.op is Op.APPLY
            and obj.data[0] is ArithOp.DIV
            and not obj.data[2]):
            return Expr(Op.FACTORS, {obj.data[1][0]: 1, obj.data[1][1]: -1})
        # 否则，默认将表达式转换为一个包含单个因子的FACTORS表达式
        return Expr(Op.FACTORS, {obj: 1})
    # 如果不是Expr类型，则抛出异常
    raise OpError(f'cannot convert {type(obj)} to terms Expr')

# 将给定的表达式对象转换为项-系数对形式
def as_term_coeff(obj):
    """Return expression as term-coefficient pair.
    """
    if isinstance(obj, Expr):
        obj = normalize(obj)
        # 如果表达式是INTEGER类型，则转换为项-系数对形式
        if obj.op is Op.INTEGER:
            return as_integer(1, obj.data[1]), obj.data[0]
        # 如果表达式是REAL类型，则转换为项-系数对形式
        if obj.op is Op.REAL:
            return as_real(1, obj.data[1]), obj.data[0]
        # 如果表达式是TERMS类型，且只包含一个项，则返回项和系数
        if obj.op is Op.TERMS:
            if len(obj.data) == 1:
                (term, coeff), = obj.data.items()
                return term, coeff
            # TODO: 找到系数的最大公约数
        # 如果表达式是APPLY类型，并且是除法操作，则转换为项-系数对形式
        if obj.op is Op.APPLY and obj.data[0] is ArithOp.DIV:
            t, c = as_term_coeff(obj.data[1][0])
            return as_apply(ArithOp.DIV, t, obj.data[1][1]), c
        # 否则，默认将表达式直接返回，并假设系数为1
        return obj, 1
    # 如果不是Expr类型，则抛出异常
    raise OpError(f'cannot convert {type(obj)} to term and coeff')

# 将给定的表达式对象转换为数值-分母对形式
def as_numer_denom(obj):
    """Return expression as numer-denom pair.
    """
    # 检查对象是否属于表达式类
    if isinstance(obj, Expr):
        # 对象规范化，确保其符合标准形式
        obj = normalize(obj)
        # 检查操作符是否属于以下类型之一：整数、实数、复数、符号、索引、三元操作符
        if obj.op in (Op.INTEGER, Op.REAL, Op.COMPLEX, Op.SYMBOL,
                      Op.INDEXING, Op.TERNARY):
            # 返回对象和数值 1
            return obj, as_number(1)
        # 如果操作符是应用操作
        elif obj.op is Op.APPLY:
            # 如果对象表示的是除法且第三个数据为空
            if obj.data[0] is ArithOp.DIV and not obj.data[2]:
                # 将操作数转换为分子和分母，然后返回其交叉乘积
                numers, denoms = map(as_numer_denom, obj.data[1])
                return numers[0] * denoms[1], numers[1] * denoms[0]
            # 返回对象和数值 1
            return obj, as_number(1)
        # 如果操作符是项操作
        elif obj.op is Op.TERMS:
            # 初始化数值列表
            numers, denoms = [], []
            # 遍历项数据中的每一项及其系数
            for term, coeff in obj.data.items():
                # 将每个项转换为分子和分母形式
                n, d = as_numer_denom(term)
                # 根据系数调整分子值，并添加到分子列表中
                n = n * coeff
                numers.append(n)
                # 添加分母到分母列表中
                denoms.append(d)
            # 初始化数值并计算总分子和总分母
            numer, denom = as_number(0), as_number(1)
            for i in range(len(numers)):
                n = numers[i]
                for j in range(len(numers)):
                    if i != j:
                        # 对非当前项的分母应用乘法
                        n *= denoms[j]
                # 添加到总分子中
                numer += n
                # 对所有分母应用乘法
                denom *= denoms[i]
            # 如果总分母为整数或实数且为负数，则调整总分子和总分母的符号
            if denom.op in (Op.INTEGER, Op.REAL) and denom.data[0] < 0:
                numer, denom = -numer, -denom
            # 返回最终计算结果的分子和分母
            return numer, denom
        # 如果操作符是因子操作
        elif obj.op is Op.FACTORS:
            # 初始化分子和分母为 1
            numer, denom = as_number(1), as_number(1)
            # 遍历因子数据中的每一对底数和指数
            for b, e in obj.data.items():
                # 将每个底数转换为分子和分母形式
                bnumer, bdenom = as_numer_denom(b)
                # 根据指数值调整分子和分母
                if e > 0:
                    numer *= bnumer ** e
                    denom *= bdenom ** e
                elif e < 0:
                    numer *= bdenom ** (-e)
                    denom *= bnumer ** (-e)
            # 返回最终计算结果的分子和分母
            return numer, denom
    # 如果对象类型无法转换为分子和分母，则引发操作错误异常
    raise OpError(f'cannot convert {type(obj)} to numer and denom')
def _counter():
    # Used internally to generate unique dummy symbols
    counter = 0
    while True:
        counter += 1
        yield counter

# Initialize a global counter generator
COUNTER = _counter()

def eliminate_quotes(s):
    """Replace quoted substrings of input string.

    Return a new string and a mapping of replacements.
    """
    d = {}

    def repl(m):
        kind, value = m.groups()[:2]
        if kind:
            # remove trailing underscore
            kind = kind[:-1]
        # Determine if the quote is single or double and create a unique key
        p = {"'": "SINGLE", '"': "DOUBLE"}[value[0]]
        k = f'{kind}@__f2py_QUOTES_{p}_{COUNTER.__next__()}@'
        d[k] = value
        return k

    # Replace quoted substrings in the input string 's' using the repl function
    new_s = re.sub(r'({kind}_|)({single_quoted}|{double_quoted})'.format(
        kind=r'\w[\w\d_]*',
        single_quoted=r"('([^'\\]|(\\.))*')",
        double_quoted=r'("([^"\\]|(\\.))*")'),
        repl, s)

    # Ensure no quotes remain in the new string
    assert '"' not in new_s
    assert "'" not in new_s

    return new_s, d

def insert_quotes(s, d):
    """Inverse of eliminate_quotes.
    """
    # Replace the unique keys back with their original quoted values in string 's'
    for k, v in d.items():
        kind = k[:k.find('@')]
        if kind:
            kind += '_'
        s = s.replace(k, kind + v)
    return s

def replace_parenthesis(s):
    """Replace substrings of input that are enclosed in parenthesis.

    Return a new string and a mapping of replacements.
    """
    left, right = None, None
    mn_i = len(s)

    # Iterate through possible parenthesis pairs to find the first occurrence
    for left_, right_ in (('(/', '/)'), '()', '{}', '[]'):
        i = s.find(left_)
        if i == -1:
            continue
        if i < mn_i:
            mn_i = i
            left, right = left_, right_

    # If no parenthesis pairs are found, return original string and empty dictionary
    if left is None:
        return s, {}

    i = mn_i
    j = s.find(right, i)

    # Ensure balanced parenthesis within the found pair
    while s.count(left, i + 1, j) != s.count(right, i + 1, j):
        j = s.find(right, j + 1)
        if j == -1:
            raise ValueError(f'Mismatch of {left+right} parenthesis in {s!r}')

    # Determine the type of parenthesis and create a unique key
    p = {'(': 'ROUND', '[': 'SQUARE', '{': 'CURLY', '(/': 'ROUNDDIV'}[left]
    k = f'@__f2py_PARENTHESIS_{p}_{COUNTER.__next__()}@'
    v = s[i+len(left):j]

    # Recursively replace parenthesis in the remainder of the string
    r, d = replace_parenthesis(s[j+len(right):])
    d[k] = v
    return s[:i] + k + r, d

def _get_parenthesis_kind(s):
    assert s.startswith('@__f2py_PARENTHESIS_'), s
    return s.split('_')[4]

def unreplace_parenthesis(s, d):
    """Inverse of replace_parenthesis.
    """
    # Replace unique keys with their respective parenthesis and enclosed values
    for k, v in d.items():
        p = _get_parenthesis_kind(k)
        left = dict(ROUND='(', SQUARE='[', CURLY='{', ROUNDDIV='(/')[p]
        right = dict(ROUND=')', SQUARE=']', CURLY='}', ROUNDDIV='/)')[p]
        s = s.replace(k, left + v + right)
    return s

def fromstring(s, language=Language.C):
    """Create an expression from a string.

    This is a "lazy" parser, that is, only arithmetic operations are
    resolved, non-arithmetic operations are treated as symbols.
    """
    # 使用指定的语言(language)创建一个 _FromStringWorker 实例，并解析字符串 s
    r = _FromStringWorker(language=language).parse(s)
    # 如果解析结果 r 是 Expr 类型的对象，则直接返回它
    if isinstance(r, Expr):
        return r
    # 如果解析结果 r 不是 Expr 类型的对象，则抛出 ValueError 异常，指示解析失败
    raise ValueError(f'failed to parse `{s}` to Expr instance: got `{r}`')
class _Pair:
    # Internal class to represent a pair of expressions

    def __init__(self, left, right):
        # Constructor to initialize a _Pair object with left and right expressions
        self.left = left
        self.right = right

    def substitute(self, symbols_map):
        # Method to substitute expressions with symbols from symbols_map
        left, right = self.left, self.right
        if isinstance(left, Expr):
            left = left.substitute(symbols_map)
        if isinstance(right, Expr):
            right = right.substitute(symbols_map)
        return _Pair(left, right)

    def __repr__(self):
        # Returns a string representation of the _Pair object
        return f'{type(self).__name__}({self.left}, {self.right})'


class _FromStringWorker:

    def __init__(self, language=Language.C):
        # Constructor to initialize _FromStringWorker object with optional language parameter
        self.original = None
        self.quotes_map = None
        self.language = language

    def finalize_string(self, s):
        # Method to finalize string by inserting quotes according to quotes_map
        return insert_quotes(s, self.quotes_map)

    def parse(self, inp):
        # Method to parse input string inp
        self.original = inp
        # Eliminate quotes from inp, store unquoted version in unquoted, quotes mapping in quotes_map
        unquoted, self.quotes_map = eliminate_quotes(inp)
        # Process the unquoted string and return the result
        return self.process(unquoted)
```