# `D:\src\scipysrc\sympy\sympy\core\tests\test_expr.py`

```
# 导入 sympy.assumptions.refine 模块中的 refine 函数
from sympy.assumptions.refine import refine
# 导入 sympy.concrete.summations 模块中的 Sum 类
from sympy.concrete.summations import Sum
# 导入 sympy.core.add 模块中的 Add 类
from sympy.core.add import Add
# 导入 sympy.core.basic 模块中的 Basic 类
from sympy.core.basic import Basic
# 导入 sympy.core.containers 模块中的 Tuple 类
from sympy.core.containers import Tuple
# 导入 sympy.core.expr 模块中的多个类
from sympy.core.expr import (ExprBuilder, unchanged, Expr,
    UnevaluatedExpr)
# 导入 sympy.core.function 模块中的多个函数相关类
from sympy.core.function import (Function, expand, WildFunction,
    AppliedUndef, Derivative, diff, Subs)
# 导入 sympy.core.mul 模块中的 Mul 类
from sympy.core.mul import Mul
# 导入 sympy.core.numbers 模块中的多个数学常数和符号类
from sympy.core.numbers import (NumberSymbol, E, zoo, oo, Float, I,
    Rational, nan, Integer, Number, pi, _illegal)
# 导入 sympy.core.power 模块中的 Pow 类
from sympy.core.power import Pow
# 导入 sympy.core.relational 模块中的比较类
from sympy.core.relational import Ge, Lt, Gt, Le
# 导入 sympy.core.singleton 模块中的 S 单例对象
from sympy.core.singleton import S
# 导入 sympy.core.sorting 模块中的 default_sort_key 函数
from sympy.core.sorting import default_sort_key
# 导入 sympy.core.symbol 模块中的多个符号相关类和函数
from sympy.core.symbol import Symbol, symbols, Dummy, Wild
# 导入 sympy.core.sympify 模块中的 sympify 函数
from sympy.core.sympify import sympify
# 导入 sympy.functions.combinatorial.factorials 模块中的 factorial 函数
from sympy.functions.combinatorial.factorials import factorial
# 导入 sympy.functions.elementary.exponential 模块中的指数函数
from sympy.functions.elementary.exponential import exp_polar, exp, log
# 导入 sympy.functions.elementary.miscellaneous 模块中的 sqrt 和 Max 函数
from sympy.functions.elementary.miscellaneous import sqrt, Max
# 导入 sympy.functions.elementary.piecewise 模块中的 Piecewise 类
from sympy.functions.elementary.piecewise import Piecewise
# 导入 sympy.functions.elementary.trigonometric 模块中的三角函数
from sympy.functions.elementary.trigonometric import tan, sin, cos
# 导入 sympy.functions.special.delta_functions 模块中的单位阶跃和 Dirac δ 函数
from sympy.functions.special.delta_functions import (Heaviside,
    DiracDelta)
# 导入 sympy.functions.special.error_functions 模块中的 Si 函数
from sympy.functions.special.error_functions import Si
# 导入 sympy.functions.special.gamma_functions 模块中的 gamma 函数
from sympy.functions.special.gamma_functions import gamma
# 导入 sympy.integrals.integrals 模块中的 integrate 和 Integral 类
from sympy.integrals.integrals import integrate, Integral
# 导入 sympy.physics.secondquant 模块中的 FockState 类
from sympy.physics.secondquant import FockState
# 导入 sympy.polys.partfrac 模块中的 apart 函数
from sympy.polys.partfrac import apart
# 导入 sympy.polys.polytools 模块中的 factor、cancel 和 Poly 函数
from sympy.polys.polytools import factor, cancel, Poly
# 导入 sympy.polys.rationaltools 模块中的 together 函数
from sympy.polys.rationaltools import together
# 导入 sympy.series.order 模块中的 O 类
from sympy.series.order import O
# 导入 sympy.sets.sets 模块中的 FiniteSet 类
from sympy.sets.sets import FiniteSet
# 导入 sympy.simplify.combsimp 模块中的 combsimp 函数
from sympy.simplify.combsimp import combsimp
# 导入 sympy.simplify.gammasimp 模块中的 gammasimp 函数
from sympy.simplify.gammasimp import gammasimp
# 导入 sympy.simplify.powsimp 模块中的 powsimp 函数
from sympy.simplify.powsimp import powsimp
# 导入 sympy.simplify.radsimp 模块中的 collect 和 radsimp 函数
from sympy.simplify.radsimp import collect, radsimp
# 导入 sympy.simplify.ratsimp 模块中的 ratsimp 函数
from sympy.simplify.ratsimp import ratsimp
# 导入 sympy.simplify.simplify 模块中的 simplify 和 nsimplify 函数
from sympy.simplify.simplify import simplify, nsimplify
# 导入 sympy.simplify.trigsimp 模块中的 trigsimp 函数
from sympy.simplify.trigsimp import trigsimp
# 导入 sympy.tensor.indexed 模块中的 Indexed 类
from sympy.tensor.indexed import Indexed
# 导入 sympy.physics.units 模块中的 meter 单位
from sympy.physics.units import meter

# 导入 sympy.testing.pytest 模块中的 raises 和 XFAIL 装饰器
from sympy.testing.pytest import raises, XFAIL

# 导入 sympy.abc 模块中定义的符号变量
from sympy.abc import a, b, c, n, t, u, x, y, z

# 定义三个函数符号 f, g, h
f, g, h = symbols('f,g,h', cls=Function)

# 定义一个 DummyNumber 类，实现了 SymPy 所需的最小功能
class DummyNumber:
    """
    Minimal implementation of a number that works with SymPy.

    If one has a Number class (e.g. Sage Integer, or some other custom class)
    that one wants to work well with SymPy, one has to implement at least the
    methods of this class DummyNumber, resp. its subclasses I5 and F1_1.

    Basically, one just needs to implement either __int__() or __float__() and
    then one needs to make sure that the class works with Python integers and
    with itself.
    """

    # 实现了右加操作，如果操作数是 int 或 float 类型，则返回加法结果
    def __radd__(self, a):
        if isinstance(a, (int, float)):
            return a + self.number
        return NotImplemented

    # 实现了加法操作，如果操作数是 int、float 或 DummyNumber 类型，则返回加法结果
    def __add__(self, a):
        if isinstance(a, (int, float, DummyNumber)):
            return self.number + a
        return NotImplemented
    # 实现反向减法运算符的特殊方法
    def __rsub__(self, a):
        # 如果传入的参数是整数、浮点数，则返回参数减去对象的数值
        if isinstance(a, (int, float)):
            return a - self.number
        # 否则返回 Not Implemented，表示不支持该操作
        return NotImplemented

    # 实现减法运算符的特殊方法
    def __sub__(self, a):
        # 如果传入的参数是整数、浮点数或 DummyNumber 对象，则返回对象的数值减去参数
        if isinstance(a, (int, float, DummyNumber)):
            return self.number - a
        # 否则返回 Not Implemented，表示不支持该操作
        return NotImplemented

    # 实现反向乘法运算符的特殊方法
    def __rmul__(self, a):
        # 如果传入的参数是整数或浮点数，则返回参数乘以对象的数值
        if isinstance(a, (int, float)):
            return a * self.number
        # 否则返回 Not Implemented，表示不支持该操作
        return NotImplemented

    # 实现乘法运算符的特殊方法
    def __mul__(self, a):
        # 如果传入的参数是整数、浮点数或 DummyNumber 对象，则返回对象的数值乘以参数
        if isinstance(a, (int, float, DummyNumber)):
            return self.number * a
        # 否则返回 Not Implemented，表示不支持该操作
        return NotImplemented

    # 实现反向真除法运算符的特殊方法
    def __rtruediv__(self, a):
        # 如果传入的参数是整数或浮点数，则返回参数除以对象的数值
        if isinstance(a, (int, float)):
            return a / self.number
        # 否则返回 Not Implemented，表示不支持该操作
        return NotImplemented

    # 实现真除法运算符的特殊方法
    def __truediv__(self, a):
        # 如果传入的参数是整数、浮点数或 DummyNumber 对象，则返回对象的数值除以参数
        if isinstance(a, (int, float, DummyNumber)):
            return self.number / a
        # 否则返回 Not Implemented，表示不支持该操作
        return NotImplemented

    # 实现反向乘方运算符的特殊方法
    def __rpow__(self, a):
        # 如果传入的参数是整数或浮点数，则返回参数的对象乘方
        if isinstance(a, (int, float)):
            return a ** self.number
        # 否则返回 Not Implemented，表示不支持该操作
        return NotImplemented

    # 实现乘方运算符的特殊方法
    def __pow__(self, a):
        # 如果传入的参数是整数、浮点数或 DummyNumber 对象，则返回对象的数值乘以参数的乘方
        if isinstance(a, (int, float, DummyNumber)):
            return self.number ** a
        # 否则返回 Not Implemented，表示不支持该操作
        return NotImplemented

    # 实现正号运算符的特殊方法
    def __pos__(self):
        # 直接返回对象的数值
        return self.number

    # 实现负号运算符的特殊方法
    def __neg__(self):
        # 返回对象数值的负值
        return - self.number
class I5(DummyNumber):
    number = 5

    # 返回对象的整数表示
    def __int__(self):
        return self.number


class F1_1(DummyNumber):
    number = 1.1

    # 返回对象的浮点数表示
    def __float__(self):
        return self.number

i5 = I5()   # 创建一个 I5 类的实例
f1_1 = F1_1()   # 创建一个 F1_1 类的实例

# 基本的 SymPy 对象列表
basic_objs = [
    Rational(2),    # 使用整数创建有理数对象
    Float("1.3"),   # 使用字符串创建浮点数对象
    x,              # 变量 x
    y,              # 变量 y
    pow(x, y)*y,    # x 的 y 次方乘以 y
]

# 所有支持的对象列表，包括基本对象和自定义类的实例
all_objs = basic_objs + [
    5,      # 整数对象
    5.5,    # 浮点数对象
    i5,     # I5 类的实例
    f1_1    # F1_1 类的实例
]

def dotest(s):
    # 对所有对象的组合进行测试
    for xo in all_objs:
        for yo in all_objs:
            s(xo, yo)   # 调用传入的函数 s，传入两个对象 xo 和 yo 进行操作
    return True

def test_basic():
    def j(a, b):
        x = a       # 将 a 赋值给 x
        x = +a      # 正数运算符
        x = -a      # 负数运算符
        x = a + b   # 加法运算
        x = a - b   # 减法运算
        x = a*b     # 乘法运算
        x = a/b     # 除法运算
        x = a**b    # 幂运算
        del x       # 删除变量 x
    assert dotest(j)    # 断言调用 dotest 函数返回 True

def test_ibasic():
    def s(a, b):
        x = a       # 将 a 赋值给 x
        x += b      # 增强赋值加法运算
        x = a       # 将 a 赋值给 x
        x -= b      # 增强赋值减法运算
        x = a       # 将 a 赋值给 x
        x *= b      # 增强赋值乘法运算
        x = a       # 将 a 赋值给 x
        x /= b      # 增强赋值除法运算
    assert dotest(s)    # 断言调用 dotest 函数返回 True

class NonBasic:
    '''This class represents an object that knows how to implement binary
    operations like +, -, etc with Expr but is not a subclass of Basic itself.
    The NonExpr subclass below does subclass Basic but not Expr.

    For both NonBasic and NonExpr it should be possible for them to override
    Expr.__add__ etc because Expr.__add__ should be returning NotImplemented
    for non Expr classes. Otherwise Expr.__add__ would create meaningless
    objects like Add(Integer(1), FiniteSet(2)) and it wouldn't be possible for
    other classes to override these operations when interacting with Expr.
    '''

    # 以下是各种算术和比较运算符的实现，返回特殊操作对象 SpecialOp
    def __add__(self, other):
        return SpecialOp('+', self, other)

    def __radd__(self, other):
        return SpecialOp('+', other, self)

    def __sub__(self, other):
        return SpecialOp('-', self, other)

    def __rsub__(self, other):
        return SpecialOp('-', other, self)

    def __mul__(self, other):
        return SpecialOp('*', self, other)

    def __rmul__(self, other):
        return SpecialOp('*', other, self)

    def __truediv__(self, other):
        return SpecialOp('/', self, other)

    def __rtruediv__(self, other):
        return SpecialOp('/', other, self)

    def __floordiv__(self, other):
        return SpecialOp('//', self, other)

    def __rfloordiv__(self, other):
        return SpecialOp('//', other, self)

    def __mod__(self, other):
        return SpecialOp('%', self, other)

    def __rmod__(self, other):
        return SpecialOp('%', other, self)

    def __divmod__(self, other):
        return SpecialOp('divmod', self, other)

    def __rdivmod__(self, other):
        return SpecialOp('divmod', other, self)

    def __pow__(self, other):
        return SpecialOp('**', self, other)

    def __rpow__(self, other):
        return SpecialOp('**', other, self)

    def __lt__(self, other):
        return SpecialOp('<', self, other)

    def __gt__(self, other):
        return SpecialOp('>', self, other)

    def __le__(self, other):
        return SpecialOp('<=', self, other)
    # 定义一个特殊操作方法，用于检查当前对象是否大于等于另一个对象
    def __ge__(self, other):
        # 创建一个特殊操作对象，表示当前对象与另一个对象的大于等于关系
        return SpecialOp('>=', self, other)
# 定义一个名为 NonExpr 的类，继承自 Basic 和 NonBasic 类
class NonExpr(Basic, NonBasic):
    '''Like NonBasic above except this is a subclass of Basic but not Expr'''
    pass


# 定义一个名为 SpecialOp 的类
class SpecialOp():
    '''Represents the results of operations with NonBasic and NonExpr'''
    
    # 自定义 __new__ 方法用于创建实例
    def __new__(cls, op, arg1, arg2):
        # 使用 object.__new__ 创建一个实例对象
        obj = object.__new__(cls)
        # 设置实例的 args 属性为 (op, arg1, arg2)
        obj.args = (op, arg1, arg2)
        return obj


# 定义一个名为 NonArithmetic 的类，继承自 Basic 类
class NonArithmetic(Basic):
    '''Represents a Basic subclass that does not support arithmetic operations'''
    pass


# 定义一个名为 test_cooperative_operations 的函数，用于测试 Expr 类在二元操作中的协作性
def test_cooperative_operations():
    '''Tests that Expr uses binary operations cooperatively.

    In particular it should be possible for non-Expr classes to override
    binary operators like +, - etc when used with Expr instances. This should
    work for non-Expr classes whether they are Basic subclasses or not. Also
    non-Expr classes that do not define binary operators with Expr should give
    TypeError.
    '''
    
    # 一组 Expr 类的子类实例
    exprs = [
        Expr(),
        S.Zero,
        S.One,
        S.Infinity,
        S.NegativeInfinity,
        S.ComplexInfinity,
        S.Half,
        Float(0.5),
        Integer(2),
        Symbol('x'),
        Mul(2, Symbol('x')),
        Add(2, Symbol('x')),
        Pow(2, Symbol('x')),
    ]
    for e in exprs:
        # 遍历表达式列表中的每一个表达式对象
        # 为了测试这些类能否重写与各种 Expr 类型结合的算术操作。
        for ne in [NonBasic(), NonExpr()]:
            # 对于每个 NonBasic 和 NonExpr 实例 ne，执行以下操作：

            results = [
                (ne + e, ('+', ne, e)),  # 执行 ne + e，期望结果是一个特殊操作对象，参数为 ('+', ne, e)
                (e + ne, ('+', e, ne)),  # 执行 e + ne，期望结果是一个特殊操作对象，参数为 ('+', e, ne)
                (ne - e, ('-', ne, e)),  # 执行 ne - e，期望结果是一个特殊操作对象，参数为 ('-', ne, e)
                (e - ne, ('-', e, ne)),  # 执行 e - ne，期望结果是一个特殊操作对象，参数为 ('-', e, ne)
                (ne * e, ('*', ne, e)),  # 执行 ne * e，期望结果是一个特殊操作对象，参数为 ('*', ne, e)
                (e * ne, ('*', e, ne)),  # 执行 e * ne，期望结果是一个特殊操作对象，参数为 ('*', e, ne)
                (ne / e, ('/', ne, e)),  # 执行 ne / e，期望结果是一个特殊操作对象，参数为 ('/', ne, e)
                (e / ne, ('/', e, ne)),  # 执行 e / ne，期望结果是一个特殊操作对象，参数为 ('/', e, ne)
                (ne // e, ('//', ne, e)),  # 执行 ne // e，期望结果是一个特殊操作对象，参数为 ('//', ne, e)
                (e // ne, ('//', e, ne)),  # 执行 e // ne，期望结果是一个特殊操作对象，参数为 ('//', e, ne)
                (ne % e, ('%', ne, e)),  # 执行 ne % e，期望结果是一个特殊操作对象，参数为 ('%', ne, e)
                (e % ne, ('%', e, ne)),  # 执行 e % ne，期望结果是一个特殊操作对象，参数为 ('%', e, ne)
                (divmod(ne, e), ('divmod', ne, e)),  # 执行 divmod(ne, e)，期望结果是一个特殊操作对象，参数为 ('divmod', ne, e)
                (divmod(e, ne), ('divmod', e, ne)),  # 执行 divmod(e, ne)，期望结果是一个特殊操作对象，参数为 ('divmod', e, ne)
                (ne ** e, ('**', ne, e)),  # 执行 ne ** e，期望结果是一个特殊操作对象，参数为 ('**', ne, e)
                (e ** ne, ('**', e, ne)),  # 执行 e ** ne，期望结果是一个特殊操作对象，参数为 ('**', e, ne)
                (e < ne, ('>', ne, e)),  # 执行 e < ne，期望结果是一个特殊操作对象，参数为 ('>', ne, e)
                (ne < e, ('<', ne, e)),  # 执行 ne < e，期望结果是一个特殊操作对象，参数为 ('<', ne, e)
                (e > ne, ('<', ne, e)),  # 执行 e > ne，期望结果是一个特殊操作对象，参数为 ('<', ne, e)
                (ne > e, ('>', ne, e)),  # 执行 ne > e，期望结果是一个特殊操作对象，参数为 ('>', ne, e)
                (e <= ne, ('>=', ne, e)),  # 执行 e <= ne，期望结果是一个特殊操作对象，参数为 ('>=', ne, e)
                (ne <= e, ('<=', ne, e)),  # 执行 ne <= e，期望结果是一个特殊操作对象，参数为 ('<=', ne, e)
                (e >= ne, ('<=', ne, e)),  # 执行 e >= ne，期望结果是一个特殊操作对象，参数为 ('<=', ne, e)
                (ne >= e, ('>=', ne, e)),  # 执行 ne >= e，期望结果是一个特殊操作对象，参数为 ('>=', ne, e)
            ]

            for res, args in results:
                # 对于每个结果和其参数，使用断言检查结果的类型和参数是否符合预期
                assert type(res) is SpecialOp and res.args == args

        # 这些类不支持与 Expr 类型的二元操作。每个操作应该在与任何 Expr 类型结合时引发 TypeError 异常。
        for na in [NonArithmetic(), object()]:
            # 对于每个 NonArithmetic 实例 na 或普通对象实例，执行以下操作：

            raises(TypeError, lambda : e + na)  # 断言执行 e + na 会引发 TypeError 异常
            raises(TypeError, lambda : na + e)  # 断言执行 na + e 会引发 TypeError 异常
            raises(TypeError, lambda : e - na)  # 断言执行 e - na 会引发 TypeError 异常
            raises(TypeError, lambda : na - e)  # 断言执行 na - e 会引发 TypeError 异常
            raises(TypeError, lambda : e * na)  # 断言执行 e * na 会引发 TypeError 异常
            raises(TypeError, lambda : na * e)  # 断言执行 na * e 会引发 TypeError 异常
            raises(TypeError, lambda : e / na)  # 断言执行 e / na 会引发 TypeError 异常
            raises(TypeError, lambda : na / e)  # 断言执行 na / e 会引发 TypeError 异常
            raises(TypeError, lambda : e // na)  # 断言执行 e // na 会引发 TypeError 异常
            raises(TypeError, lambda : na // e)  # 断言执行 na // e 会引发 TypeError 异常
            raises(TypeError, lambda : e % na)  # 断言执行 e % na 会引发 TypeError 异常
            raises(TypeError, lambda : na % e)  # 断言执行 na % e 会引发 TypeError 异常
            raises(TypeError, lambda : divmod(e, na))  # 断言执行 divmod(e, na) 会引发 TypeError 异常
            raises(TypeError, lambda : divmod(na, e))  # 断言执行 divmod(na, e) 会引发 TypeError 异常
            raises(TypeError, lambda : e ** na)  # 断言执行 e ** na 会引发 TypeError 异常
            raises(TypeError, lambda : na ** e)  # 断言执行 na ** e 会引发 TypeError 异常
            raises(TypeError, lambda : e > na)  # 断言执行 e > na 会引发 TypeError 异常
            raises(TypeError, lambda : na > e)  # 断言执行 na > e 会引发 TypeError 异常
            raises(TypeError, lambda : e < na)  # 断言执行 e < na 会引发 TypeError 异常
            raises(TypeError, lambda : na < e)  # 断言执行 na < e 会引发 TypeError 异常
            raises(TypeError, lambda : e >= na)  # 断言执行 e >= na 会引发 TypeError 异常
            raises(TypeError, lambda : na >= e)  # 断言执行 na >= e 会引发 TypeError 异常
            raises(TypeError, lambda : e <= na)  # 断言执行 e <= na 会引发 TypeError 异常
            raises(TypeError, lambda : na <= e)  # 断言执行 na <= e 会引发 TypeError 异常
# 测试关系运算符功能
def test_relational():
    # 导入符号运算库中的 Lt（小于）函数
    from sympy.core.relational import Lt
    # 断言 pi 小于 3 是假
    assert (pi < 3) is S.false
    # 断言 pi 小于等于 3 是假
    assert (pi <= 3) is S.false
    # 断言 pi 大于 3 是真
    assert (pi > 3) is S.true
    # 断言 pi 大于等于 3 是真
    assert (pi >= 3) is S.true
    # 断言 -pi 小于 3 是真
    assert (-pi < 3) is S.true
    # 断言 -pi 小于等于 3 是真
    assert (-pi <= 3) is S.true
    # 断言 -pi 大于 3 是假
    assert (-pi > 3) is S.false
    # 断言 -pi 大于等于 3 是假
    assert (-pi >= 3) is S.false
    # 创建一个实数符号 r
    r = Symbol('r', real=True)
    # 断言 r - 2 小于 r - 3 是假
    assert (r - 2 < r - 3) is S.false
    # 断言复数运算表达式 Lt(x + I, x + I + 2) 的函数等于 Lt（小于）函数
    assert Lt(x + I, x + I + 2).func == Lt  # issue 8288


# 测试关系运算符假设
def test_relational_assumptions():
    # 创建非负数符号 m1
    m1 = Symbol("m1", nonnegative=False)
    # 创建非正数符号 m2
    m2 = Symbol("m2", positive=False)
    # 创建非正数符号 m3
    m3 = Symbol("m3", nonpositive=False)
    # 创建非负数符号 m4
    m4 = Symbol("m4", negative=False)
    # 断言 m1 小于 0 等价于 Lt(m1, 0)
    assert (m1 < 0) == Lt(m1, 0)
    # 断言 m2 小于等于 0 等价于 Le(m2, 0)
    assert (m2 <= 0) == Le(m2, 0)
    # 断言 m3 大于 0 等价于 Gt(m3, 0)
    assert (m3 > 0) == Gt(m3, 0)
    # 断言 m4 大于等于 0 等价于 Ge(m4, 0)
    assert (m4 >= 0) == Ge(m4, 0)
    # 创建一个实数符号 m1，并使其非负
    m1 = Symbol("m1", nonnegative=False, real=True)
    # 创建一个实数符号 m2，并使其正数
    m2 = Symbol("m2", positive=False, real=True)
    # 创建一个实数符号 m3，并使其非正数
    m3 = Symbol("m3", nonpositive=False, real=True)
    # 创建一个实数符号 m4，并使其负数
    m4 = Symbol("m4", negative=False, real=True)
    # 断言 m1 小于 0 是真
    assert (m1 < 0) is S.true
    # 断言 m2 小于等于 0 是真
    assert (m2 <= 0) is S.true
    # 断言 m3 大于 0 是真
    assert (m3 > 0) is S.true
    # 断言 m4 大于等于 0 是真
    assert (m4 >= 0) is S.true
    # 创建一个负数符号 m1
    m1 = Symbol("m1", negative=True)
    # 创建一个非正数符号 m2
    m2 = Symbol("m2", nonpositive=True)
    # 创建一个正数符号 m3
    m3 = Symbol("m3", positive=True)
    # 创建一个非负数符号 m4
    m4 = Symbol("m4", nonnegative=True)
    # 断言 m1 小于 0 是真
    assert (m1 < 0) is S.true
    # 断言 m2 小于等于 0 是真
    assert (m2 <= 0) is S.true
    # 断言 m3 大于 0 是真
    assert (m3 > 0) is S.true
    # 断言 m4 大于等于 0 是真
    assert (m4 >= 0) is S.true
    # 创建一个实数符号 m1，并使其非负
    m1 = Symbol("m1", negative=False, real=True)
    # 创建一个实数符号 m2，并使其非正数
    m2 = Symbol("m2", nonpositive=False, real=True)
    # 创建一个实数符号 m3，并使其正数
    m3 = Symbol("m3", positive=False, real=True)
    # 创建一个实数符号 m4，并使其非负数
    m4 = Symbol("m4", nonnegative=False, real=True)
    # 断言 m1 小于 0 是假
    assert (m1 < 0) is S.false
    # 断言 m2 小于等于 0 是假
    assert (m2 <= 0) is S.false
    # 断言 m3 大于 0 是假
    assert (m3 > 0) is S.false
    # 断言 m4 大于等于 0 是假
    assert (m4 >= 0) is S.false


# 测试基本对象非字符串运算
def test_basic_nostr():
    # 遍历基本对象列表
    for obj in basic_objs:
        # 断言将对象与字符串 '1' 相加会引发 TypeError
        raises(TypeError, lambda: obj + '1')
        # 断言将对象与字符串 '1' 相减会引发 TypeError
        raises(TypeError, lambda: obj - '1')
        # 如果对象是整数 2
        if obj == 2:
            # 断言对象乘以字符串 '1' 等于字符串 '11'
            assert obj * '1' == '11'
        else:
            # 断言将对象与字符串 '1' 相乘会引发 TypeError
            raises(TypeError, lambda: obj * '1')
        # 断言将对象与字符串 '1' 相除会引发 TypeError
        raises(TypeError, lambda: obj / '1')
        # 断言将对象的幂运算与字符串 '1' 相乘会引发 TypeError
        raises(TypeError, lambda: obj ** '1')


# 测试一致阶级的级数展开
def test_series_expansion_for_uniform_order():
    # 断言 (1/x + y + x) 在 x=0 处展开到 0 阶等于 1/x + O(1, x)
    assert (1/x + y + x).series(x, 0, 0) == 1/x + O(1, x)
    # 断言 (1/x + y + x) 在 x=0 处展开到 1 阶等于 1/x + y + O(x)
    assert (1/x + y + x).series(x, 0, 1) == 1/x + y + O(x)
    # 断言 (1/x + 1 + x) 在 x=0 处展开到 0 阶等于 1/x + O(1, x)
    assert (1/x + 1 + x).series(x, 0, 0) == 1/x + O(1, x)
    # 断言 (1/x + 1 + x) 在 x=0 处展开到 1 阶等于 1/x + 1 + O(x)
    assert (1/x + 1 + x).series(x, 0, 1) == 1/x + 1 + O(x)
    # 断言 (1/x + x) 在 x=0 处展开到 0 阶等于 1/x + O(1, x)
    assert (1/x + x).series(x, 0, 0) == 1/x + O(1, x)
    # 断言 (1/x + y + y*x + x) 在 x=0 处展开到 0 阶等于 1/x + O(1, x)
    assert (1/x + y
    # 断言：确保表达式 (1/x + 1 + x + x**2) 的主导项系数为 -1
    assert (1/x + 1 + x + x**2).leadterm(x)[1] == -1
    
    # 断言：确保表达式 (x**2 + 1/x) 的主导项系数为 -1
    assert (x**2 + 1/x).leadterm(x)[1] == -1
    
    # 断言：确保表达式 (1 + x**2) 的主导项系数为 0
    assert (1 + x**2).leadterm(x)[1] == 0
    
    # 断言：确保表达式 (x + 1) 的主导项系数为 0
    assert (x + 1).leadterm(x)[1] == 0
    
    # 断言：确保表达式 (x + x**2) 的主导项系数为 1
    assert (x + x**2).leadterm(x)[1] == 1
    
    # 断言：确保表达式 (x**2) 的主导项系数为 2
    assert (x**2).leadterm(x)[1] == 2
# 测试函数，用于验证 sympy 表达式对象的 as_leading_term 方法
def test_as_leading_term():
    # 验证针对不同表达式，as_leading_term 方法返回预期的主导项
    assert (3 + 2*x**(log(3)/log(2) - 1)).as_leading_term(x) == 3
    assert (1/x**2 + 1 + x + x**2).as_leading_term(x) == 1/x**2
    assert (1/x + 1 + x + x**2).as_leading_term(x) == 1/x
    assert (x**2 + 1/x).as_leading_term(x) == 1/x
    assert (1 + x**2).as_leading_term(x) == 1
    assert (x + 1).as_leading_term(x) == 1
    assert (x + x**2).as_leading_term(x) == x
    assert (x**2).as_leading_term(x) == x**2
    assert (x + oo).as_leading_term(x) is oo

    # 验证对于不合法的参数，抛出 ValueError 异常
    raises(ValueError, lambda: (x + 1).as_leading_term(1))

    # https://github.com/sympy/sympy/issues/21177
    e = -3*x + (x + Rational(3, 2) - sqrt(3)*S.ImaginaryUnit/2)**2\
        - Rational(3, 2) + 3*sqrt(3)*S.ImaginaryUnit/2
    # 验证复杂表达式的主导项计算结果
    assert e.as_leading_term(x) == \
        (12*sqrt(3)*x - 12*S.ImaginaryUnit*x)/(4*sqrt(3) + 12*S.ImaginaryUnit)

    # https://github.com/sympy/sympy/issues/21245
    e = 1 - x - x**2
    d = (1 + sqrt(5))/2
    # 验证替换变量后的表达式的主导项计算结果
    assert e.subs(x, y + 1/d).as_leading_term(y) == \
        (-576*sqrt(5)*y - 1280*y)/(256*sqrt(5) + 576)


# 测试函数，验证 leadterm 方法的使用
def test_leadterm2():
    # 验证 leadterm 方法返回预期的主导项及其次数
    assert (x*cos(1)*cos(1 + sin(1)) + sin(1 + sin(1))).leadterm(x) == \
           (sin(1 + sin(1)), 0)


# 测试函数，验证 leadterm 方法对简单表达式的主导项计算
def test_leadterm3():
    # 验证 leadterm 方法返回预期的主导项及其次数
    assert (y + z + x).leadterm(x) == (y + z, 0)


# 测试函数，验证 as_leading_term 方法对复杂表达式的主导项计算
def test_as_leading_term2():
    # 验证 as_leading_term 方法返回预期的主导项
    assert (x*cos(1)*cos(1 + sin(1)) + sin(1 + sin(1))).as_leading_term(x) == \
        sin(1 + sin(1))


# 测试函数，验证 as_leading_term 方法对包含常数和变量的表达式的主导项计算
def test_as_leading_term3():
    # 验证 as_leading_term 方法返回预期的主导项
    assert (2 + pi + x).as_leading_term(x) == 2 + pi
    assert (2*x + pi*x + x**2).as_leading_term(x) == 2*x + pi*x


# 测试函数，验证 as_leading_term 方法对包含符号和整数的表达式的主导项计算
def test_as_leading_term4():
    # 验证对于特定符号的表达式，as_leading_term 方法返回预期的主导项
    n = Symbol('n', integer=True, positive=True)
    r = -n**3/(2*n**2 + 4*n + 2) - n**2/(n**2 + 2*n + 1) + \
        n**2/(n + 1) - n/(2*n**2 + 4*n + 2) + n/(n*x + x) + 2*n/(n + 1) - \
        1 + 1/(n*x + x) + 1/(n + 1) - 1/x
    # 验证 as_leading_term 方法对复杂表达式的主导项计算，并取消通分
    assert r.as_leading_term(x).cancel() == n/2


# 测试函数，验证 as_leading_term 方法在类实例中的使用
def test_as_leading_term_stub():
    # 定义一个简单的类 foo
    class foo(Function):
        pass
    # 验证对于 foo 类的实例，as_leading_term 方法的行为
    assert foo(1/x).as_leading_term(x) == foo(1/x)
    assert foo(1).as_leading_term(x) == foo(1)
    # 验证当调用 as_leading_term 方法时抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: foo(x).as_leading_term(x))


# 测试函数，验证 as_leading_term 方法对导数和积分的处理
def test_as_leading_term_deriv_integral():
    # 验证对于导数的 as_leading_term 方法返回预期的主导项
    assert Derivative(x ** 3, x).as_leading_term(x) == 3*x**2
    assert Derivative(x ** 3, y).as_leading_term(x) == 0

    # 验证对于积分的 as_leading_term 方法返回预期的主导项
    assert Integral(x ** 3, x).as_leading_term(x) == x**4/4
    assert Integral(x ** 3, y).as_leading_term(x) == y*x**3

    # 验证对于特定函数的导数的 as_leading_term 方法返回预期的主导项
    assert Derivative(exp(x), x).as_leading_term(x) == 1
    assert Derivative(log(x), x).as_leading_term(x) == (1/x).as_leading_term(x)


# 测试函数，验证 atoms 方法的使用
def test_atoms():
    # 验证 atoms 方法返回表达式中的所有符号
    assert x.atoms() == {x}
    assert (1 + x).atoms() == {x, S.One}

    # 验证 atoms 方法根据参数过滤返回特定类型的符号
    assert (1 + 2*cos(x)).atoms(Symbol) == {x}
    assert (1 + 2*cos(x)).atoms(Symbol, Number) == {S.One, S(2), x}

    # 验证 atoms 方法对包含幂次的表达式返回所有符号
    assert (2*(x**(y**x))).atoms() == {S(2), x, y}

    # 验证 atoms 方法对常数的返回结果
    assert S.Half.atoms() == {S.Half}
    assert S.Half.atoms(Symbol) == set()

    # 验证 atoms 方法对包含无穷大的函数的处理
    assert sin(oo).atoms(oo) == set()

    # 验证 atoms 方法对多项式对象返回所有符号
    assert Poly(0, x).atoms() == {S.Zero, x}
    # 断言：对于多项式 Poly(1, x)，检查其原子集合是否为 {S.One, x}
    assert Poly(1, x).atoms() == {S.One, x}

    # 断言：对于多项式 Poly(x, x)，检查其原子集合是否为 {x}
    assert Poly(x, x).atoms() == {x}
    # 断言：对于多项式 Poly(x, x, y)，检查其原子集合是否为 {x, y}
    assert Poly(x, x, y).atoms() == {x, y}
    # 断言：对于多项式 Poly(x + y, x, y)，检查其原子集合是否为 {x, y}
    assert Poly(x + y, x, y).atoms() == {x, y}
    # 断言：对于多项式 Poly(x + y, x, y, z)，检查其原子集合是否为 {x, y, z}
    assert Poly(x + y, x, y, z).atoms() == {x, y, z}
    # 断言：对于多项式 Poly(x + y*t, x, y, z)，检查其原子集合是否为 {t, x, y, z}
    assert Poly(x + y*t, x, y, z).atoms() == {t, x, y, z}

    # 断言：对于复数表达式 (I*pi)，检查其原子集合中包含 NumberSymbol 类型的元素 {pi}
    assert (I*pi).atoms(NumberSymbol) == {pi}
    # 断言：对于复数表达式 (I*pi)，检查其原子集合中同时包含 NumberSymbol 和 I 类型的元素
    assert (I*pi).atoms(NumberSymbol, I) == \
        (I*pi).atoms(I, NumberSymbol) == {pi, I}

    # 断言：对于表达式 exp(exp(x))，检查其原子集合中包含指定的函数类型 exp(exp(x)) 和 exp(x)
    assert exp(exp(x)).atoms(exp) == {exp(exp(x)), exp(x)}
    # 断言：对于复杂表达式 1 + x*(2 + y) + exp(3 + z)，检查其原子集合中包含 Add 类型的元素
    assert (1 + x*(2 + y) + exp(3 + z)).atoms(Add) == \
        {1 + x*(2 + y) + exp(3 + z), 2 + y, 3 + z}

    # issue 6132 的测试
    e = (f(x) + sin(x) + 2)
    # 断言：对于表达式 e，检查其原子集合中包含 AppliedUndef 类型的元素 {f(x)}
    assert e.atoms(AppliedUndef) == \
        {f(x)}
    # 断言：对于表达式 e，检查其原子集合中同时包含 AppliedUndef 和 Function 类型的元素 {f(x), sin(x)}
    assert e.atoms(AppliedUndef, Function) == \
        {f(x), sin(x)}
    # 断言：对于表达式 e，检查其原子集合中包含 Function 类型的元素 {f(x), sin(x)}
    assert e.atoms(Function) == \
        {f(x), sin(x)}
    # 断言：对于表达式 e，检查其原子集合中同时包含 AppliedUndef 和 Number 类型的元素 {f(x), S(2)}
    assert e.atoms(AppliedUndef, Number) == \
        {f(x), S(2)}
    # 断言：对于表达式 e，检查其原子集合中同时包含 Function 和 Number 类型的元素 {S(2), sin(x), f(x)}
    assert e.atoms(Function, Number) == \
        {S(2), sin(x), f(x)}
# 定义测试函数，用于检查是否多项式
def test_is_polynomial():
    # 定义符号变量 k，其为非负整数
    k = Symbol('k', nonnegative=True, integer=True)

    # 断言：2 是关于 x, y, z 的多项式，返回 True
    assert Rational(2).is_polynomial(x, y, z) is True
    # 断言：π 是关于 x, y, z 的多项式，返回 True
    assert (S.Pi).is_polynomial(x, y, z) is True

    # 断言：x 是关于 x 的多项式，返回 True
    assert x.is_polynomial(x) is True
    # 断言：x 是关于 y 的多项式，返回 True
    assert x.is_polynomial(y) is True

    # 断言：x^2 是关于 x 的多项式，返回 True
    assert (x**2).is_polynomial(x) is True
    # 断言：x^2 是关于 y 的多项式，返回 True
    assert (x**2).is_polynomial(y) is True

    # 断言：x^(-2) 不是关于 x 的多项式，返回 False
    assert (x**(-2)).is_polynomial(x) is False
    # 断言：x^(-2) 是关于 y 的多项式，返回 True
    assert (x**(-2)).is_polynomial(y) is True

    # 断言：2^x 不是关于 x 的多项式，返回 False
    assert (2**x).is_polynomial(x) is False
    # 断言：2^x 是关于 y 的多项式，返回 True
    assert (2**x).is_polynomial(y) is True

    # 断言：x^k 不是关于 x 的多项式，返回 False
    assert (x**k).is_polynomial(x) is False
    # 断言：x^k 不是关于 k 的多项式，返回 False
    assert (x**k).is_polynomial(k) is False
    # 断言：x^x 不是关于 x 的多项式，返回 False
    assert (x**x).is_polynomial(x) is False
    # 断言：k^k 不是关于 k 的多项式，返回 False
    assert (k**k).is_polynomial(k) is False
    # 断言：k^x 不是关于 k 的多项式，返回 False
    assert (k**x).is_polynomial(k) is False

    # 断言：x^(-k) 不是关于 x 的多项式，返回 False
    assert (x**(-k)).is_polynomial(x) is False
    # 断言：(2*x)^k 不是关于 x 的多项式，返回 False
    assert ((2*x)**k).is_polynomial(x) is False

    # 断言：x^2 + 3*x - 8 是关于 x 的多项式，返回 True
    assert (x**2 + 3*x - 8).is_polynomial(x) is True
    # 断言：x^2 + 3*x - 8 是关于 y 的多项式，返回 True
    assert (x**2 + 3*x - 8).is_polynomial(y) is True

    # 断言：x^2 + 3*x - 8 是关于任意变量的多项式，返回 True
    assert (x**2 + 3*x - 8).is_polynomial() is True

    # 断言：sqrt(x) 不是关于 x 的多项式，返回 False
    assert sqrt(x).is_polynomial(x) is False
    # 断言：(sqrt(x))^3 不是关于 x 的多项式，返回 False
    assert (sqrt(x)**3).is_polynomial(x) is False

    # 断言：x^2 + 3*x*sqrt(y) - 8 是关于 x 的多项式，返回 True
    assert (x**2 + 3*x*sqrt(y) - 8).is_polynomial(x) is True
    # 断言：x^2 + 3*x*sqrt(y) - 8 不是关于 y 的多项式，返回 False
    assert (x**2 + 3*x*sqrt(y) - 8).is_polynomial(y) is False

    # 断言：(x^2)*(y^2) + x*(y^2) + y*x + exp(2) 是关于任意变量的多项式，返回 True
    assert ((x**2)*(y**2) + x*(y**2) + y*x + exp(2)).is_polynomial() is True
    # 断言：(x^2)*(y^2) + x*(y^2) + y*x + exp(x) 不是关于任意变量的多项式，返回 False
    assert ((x**2)*(y**2) + x*(y**2) + y*x + exp(x)).is_polynomial() is False

    # 断言：(x^2)*(y^2) + x*(y^2) + y*x + exp(2) 是关于 x, y 的多项式，返回 True
    assert ((x**2)*(y**2) + x*(y**2) + y*x + exp(2)).is_polynomial(x, y) is True
    # 断言：(x^2)*(y^2) + x*(y^2) + y*x + exp(x) 不是关于 x, y 的多项式，返回 False
    assert ((x**2)*(y**2) + x*(y**2) + y*x + exp(x)).is_polynomial(x, y) is False

    # 断言：1/f(x) + 1 不是关于 f(x) 的多项式，返回 False
    assert (1/f(x) + 1).is_polynomial(f(x)) is False


# 定义测试函数，用于检查是否有理函数
def test_is_rational_function():
    # 断言：整数 1 是有理函数，返回 True
    assert Integer(1).is_rational_function() is True
    # 断言：整数 1 是关于 x 的有理函数，返回 True
    assert Integer(1).is_rational_function(x) is True

    # 断言：分数 17/54 是有理函数，返回 True
    assert Rational(17, 54).is_rational_function() is True
    # 断言：分数 17/54 是关于 x 的有理函数，返回 True
    assert Rational(17, 54).is_rational_function(x) is True

    # 断言：12/x 是有理函数，返回 True
    assert (12/x).is_rational_function() is True
    # 断言：12/x 是关于 x 的有理函数，返回 True
    assert (12/x).is_rational_function(x) is True

    # 断言：x/y 是有理函数，返回 True
    assert (x/y).is_rational_function() is True
    # 断言：x/y 是关于 x 的有理函数，返回 True
    assert (x/y).is_rational_function(x) is True
    # 断言：x/y 是关于 x, y 的有理函数，返回 True
    assert (x/y).is_rational_function(x, y) is True

    # 断言：x^2 + 1/(x*y) 是有理函数，返回 True
    assert (x**2 + 1/x/y).is_rational_function() is True
    # 断言：x^2 + 1/(x*y) 是关于 x 的有理函数，返回 True
    assert (x**2 + 1/x/y).is_rational_function(x) is True
    # 断言：x^2 + 1/(x*y) 是关于 x, y 的有理函数，返回 True
    assert (x**2 + 1/x/y).is_rational_function(x, y) is True

    # 断言：sin(y)/x 不是有理函数，返回 False
    assert (sin(y)/x).is_rational_function() is False
    # 断言：sin(y)/x 不是关于 y 的有理函数，返回 False
    assert (sin(y)/x).is_rational_function(y) is False
    # 断言：sin(y)/x 是关于 x 的有理函数，返回 True
    assert (sin(y)/x).is_rational_function(x) is True
    # 断言：sin(y)/x 不是关于 x, y 的有理函数，返回 False
    assert (sin(y)/x).is_rational_function(x, y) is False

    # 对于 _illegal 中的
    # 断言 g 对象的 is_meromorphic 方法在参数 x 和 zoo 下返回 False
    assert g.is_meromorphic(x, zoo) is False
    
    # 创建整数符号变量 n
    n = Symbol('n', integer=True)
    # 构建表达式 e，包括 sin(1/x)**n*x
    e = sin(1/x)**n*x
    # 断言 e 对象的 is_meromorphic 方法在参数 x 和 0 下返回 False
    assert e.is_meromorphic(x, 0) is False
    # 断言 e 对象的 is_meromorphic 方法在参数 x 和 1 下返回 True
    assert e.is_meromorphic(x, 1) is True
    # 断言 e 对象的 is_meromorphic 方法在参数 x 和 zoo 下返回 False
    assert e.is_meromorphic(x, zoo) is False
    
    # 更新表达式 e，包括 log(x)**pi
    e = log(x)**pi
    # 断言 e 对象的 is_meromorphic 方法在参数 x 和 0 下返回 False
    assert e.is_meromorphic(x, 0) is False
    # 断言 e 对象的 is_meromorphic 方法在参数 x 和 1 下返回 False
    assert e.is_meromorphic(x, 1) is False
    # 断言 e 对象的 is_meromorphic 方法在参数 x 和 2 下返回 True
    assert e.is_meromorphic(x, 2) is True
    # 断言 e 对象的 is_meromorphic 方法在参数 x 和 zoo 下返回 False
    assert e.is_meromorphic(x, zoo) is False
    
    # 断言 log(x)**a 的 is_meromorphic 方法在参数 x 和 0 下返回 False
    assert (log(x)**a).is_meromorphic(x, 0) is False
    # 断言 log(x)**a 的 is_meromorphic 方法在参数 x 和 1 下返回 False
    assert (log(x)**a).is_meromorphic(x, 1) is False
    # 断言 a**log(x) 的 is_meromorphic 方法在参数 x 和 0 下返回 None
    assert (a**log(x)).is_meromorphic(x, 0) is None
    # 断言 3**log(x) 的 is_meromorphic 方法在参数 x 和 0 下返回 False
    assert (3**log(x)).is_meromorphic(x, 0) is False
    # 断言 3**log(x) 的 is_meromorphic 方法在参数 x 和 1 下返回 True
    assert (3**log(x)).is_meromorphic(x, 1) is True
def test_is_algebraic_expr():
    # 检验平方根是否为代数表达式，期望返回True
    assert sqrt(3).is_algebraic_expr(x) is True
    # 检验平方根是否为代数表达式，期望返回True
    assert sqrt(3).is_algebraic_expr() is True

    # 创建一个复杂的代数表达式
    eq = ((1 + x**2)/(1 - y**2))**(S.One/3)
    # 检验表达式是否为代数表达式，期望返回True
    assert eq.is_algebraic_expr(x) is True
    # 检验表达式是否为代数表达式，期望返回True
    assert eq.is_algebraic_expr(y) is True

    # 检验包含平方根和幂运算的表达式是否为代数表达式，期望返回True
    assert (sqrt(x) + y**(S(2)/3)).is_algebraic_expr(x) is True
    # 检验包含平方根和幂运算的表达式是否为代数表达式，期望返回True
    assert (sqrt(x) + y**(S(2)/3)).is_algebraic_expr(y) is True
    # 检验包含平方根和幂运算的表达式是否为代数表达式，期望返回True
    assert (sqrt(x) + y**(S(2)/3)).is_algebraic_expr() is True

    # 检验不是代数表达式的表达式，期望返回False
    assert (cos(y)/sqrt(x)).is_algebraic_expr() is False
    # 检验不是代数表达式的表达式，期望返回True
    assert (cos(y)/sqrt(x)).is_algebraic_expr(x) is True
    # 检验不是代数表达式的表达式，期望返回False
    assert (cos(y)/sqrt(x)).is_algebraic_expr(y) is False
    # 检验不是代数表达式的表达式，期望返回False
    assert (cos(y)/sqrt(x)).is_algebraic_expr(x, y) is False


def test_SAGE1():
    # 参考 https://github.com/sympy/sympy/issues/3346
    # 定义一个自定义整数类 MyInt
    class MyInt:
        # _sympy_ 方法返回整数值 5
        def _sympy_(self):
            return Integer(5)
    m = MyInt()
    # 计算 2 * m，期望结果为 10
    e = Rational(2)*m
    assert e == 10

    # 测试类型错误，预期会抛出 TypeError 异常
    raises(TypeError, lambda: Rational(2)*MyInt)


def test_SAGE2():
    # 定义一个自定义整数类 MyInt
    class MyInt:
        # __int__ 方法返回整数值 5
        def __int__(self):
            return 5
    assert sympify(MyInt()) == 5
    # 计算 2 * MyInt()，期望结果为 10
    e = Rational(2)*MyInt()
    assert e == 10

    # 测试类型错误，预期会抛出 TypeError 异常
    raises(TypeError, lambda: Rational(2)*MyInt)


def test_SAGE3():
    # 定义一个自定义符号类 MySymbol
    class MySymbol:
        # __rmul__ 方法定义右乘操作
        def __rmul__(self, other):
            return ('mys', other, self)

    o = MySymbol()
    # 执行 x * o，预期结果是元组 ('mys', x, o)
    e = x*o

    assert e == ('mys', x, o)


def test_len():
    e = x*y
    # 检查表达式 e 的参数数量，期望为 2
    assert len(e.args) == 2
    e = x + y + z
    # 检查表达式 e 的参数数量，期望为 3
    assert len(e.args) == 3


def test_doit():
    a = Integral(x**2, x)

    # 检查对积分对象 a 进行 doit() 操作后返回的类型是否不是 Integral 类型
    assert isinstance(a.doit(), Integral) is False

    # 检查对积分对象 a 进行 doit(integrals=True) 操作后返回的类型是否不是 Integral 类型
    assert isinstance(a.doit(integrals=True), Integral) is False
    # 检查对积分对象 a 进行 doit(integrals=False) 操作后返回的类型是否是 Integral 类型
    assert isinstance(a.doit(integrals=False), Integral) is True

    # 检查 (2 * Integral(x, x)).doit() 的计算结果是否为 x**2
    assert (2*Integral(x, x)).doit() == x**2


def test_attribute_error():
    # 测试 x 对象调用不存在的 cos() 方法时是否会抛出 AttributeError 异常
    raises(AttributeError, lambda: x.cos())
    # 测试 x 对象调用不存在的 sin() 方法时是否会抛出 AttributeError 异常
    raises(AttributeError, lambda: x.sin())
    # 测试 x 对象调用不存在的 exp() 方法时是否会抛出 AttributeError 异常
    raises(AttributeError, lambda: x.exp())


def test_args():
    # 检查乘积表达式 x * y 的参数顺序是否为 (x, y) 或 (y, x) 中的一种
    assert (x*y).args in ((x, y), (y, x))
    # 检查加法表达式 x + y 的参数顺序是否为 (x, y) 或 (y, x) 中的一种
    assert (x + y).args in ((x, y), (y, x))
    # 检查乘积加常数表达式 x*y + 1 的参数顺序是否为 (x*y, 1) 或 (1, x*y) 中的一种
    assert (x*y + 1).args in ((x*y, 1), (1, x*y))
    # 检查 sin(x*y) 表达式的参数是否为 (x*y,)
    assert sin(x*y).args == (x*y,)
    # 检查 sin(x*y) 表达式的第一个参数是否为 x*y
    assert sin(x*y).args[0] == x*y
    # 检查 x**y 表达式的参数是否为 (x, y)
    assert (x**y).args == (x, y)
    # 检查 x**y 表达式的第一个参数是否为 x
    assert (x**y).args[0] == x
    # 检查 x**y 表达式的第二个参数是否为 y
    assert (x**y).args[1] == y


def test_noncommutative_expand_issue_3757():
    # 定义非交换符号 A, B, C
    A, B, C = symbols('A,B,C', commutative=False)
    # 检查 A*B - B*A 是否不等于零
    assert A*B - B*A != 0
    # 检查 A*(A + B)*B 的展开结果是否为 A**2*B + A*B**2
    assert (A*(A + B)*B).expand() == A**2*B + A*B**2
    # 检查 A*(A + B + C)*B 的展开结果是否为 A**2*B + A*B**2 + A*C*B


def test_as_numer_denom():
    a, b, c = symbols('a, b, c')

    # 检查 nan 的分子分母表示是否为 (nan, 1)
    assert nan.as_numer_denom() == (nan, 1)
    # 检查 oo 的分子分母表示是否为 (oo, 1)
    assert oo.as_numer_denom() == (oo, 1)
    # 检查 -oo 的分子分母表示是否为 (-oo, 1)
    assert (-oo).as_numer_denom() == (-oo, 1)
    # 检查 zoo 的分子分母表示是否为 (zoo, 1)
    assert zoo.as_numer_denom() == (zoo, 1)
    # 检查 -zoo 的分子分母表示是否为 (zoo,
    # 断言表达式 S.Half.as_numer_denom() 返回 (1, 2)
    assert S.Half.as_numer_denom() == (1, 2)
    # 断言表达式 (1/y**2).as_numer_denom() 返回 (1, y**2)
    assert (1/y**2).as_numer_denom() == (1, y**2)
    # 断言表达式 (x/y**2).as_numer_denom() 返回 (x, y**2)
    assert (x/y**2).as_numer_denom() == (x, y**2)
    # 断言表达式 ((x**2 + 1)/y).as_numer_denom() 返回 (x**2 + 1, y)
    assert ((x**2 + 1)/y).as_numer_denom() == (x**2 + 1, y)
    # 断言表达式 (x*(y + 1)/y**7).as_numer_denom() 返回 (x*(y + 1), y**7)
    assert (x*(y + 1)/y**7).as_numer_denom() == (x*(y + 1), y**7)
    # 断言表达式 (x**-2).as_numer_denom() 返回 (1, x**2)
    assert (x**-2).as_numer_denom() == (1, x**2)
    # 断言表达式 (a/x + b/2/x + c/3/x).as_numer_denom() 返回 (6*a + 3*b + 2*c, 6*x)
    assert (a/x + b/2/x + c/3/x).as_numer_denom() == (6*a + 3*b + 2*c, 6*x)
    # 断言表达式 (a/x + b/2/x + c/3/y).as_numer_denom() 返回 (2*c*x + y*(6*a + 3*b), 6*x*y)
    assert (a/x + b/2/x + c/3/y).as_numer_denom() == (2*c*x + y*(6*a + 3*b), 6*x*y)
    # 断言表达式 (a/x + b/2/x + c/.5/x).as_numer_denom() 返回 (2*a + b + 4.0*c, 2*x)
    assert (a/x + b/2/x + c/.5/x).as_numer_denom() == (2*a + b + 4.0*c, 2*x)
    # 断言表达式 int(log(Add(*[Dummy()/i/x for i in range(1, 705)]).as_numer_denom()[1]/x).n(4)) 返回 705
    # 这个断言应在几秒钟内完成
    assert int(log(Add(*[Dummy()/i/x for i in range(1, 705)]).as_numer_denom()[1]/x).n(4)) == 705
    # 针对特定的无穷大值进行断言
    for i in [S.Infinity, S.NegativeInfinity, S.ComplexInfinity]:
        # 断言表达式 (i + x/3).as_numer_denom() 返回 (x + i, 3)
        assert (i + x/3).as_numer_denom() == (x + i, 3)
    # 断言表达式 (S.Infinity + x/3 + y/4).as_numer_denom() 返回 (4*x + 3*y + S.Infinity, 12)
    assert (S.Infinity + x/3 + y/4).as_numer_denom() == (4*x + 3*y + S.Infinity, 12)
    # 断言表达式 (oo*x + zoo*y).as_numer_denom() 返回 (zoo*y + oo*x, 1)
    assert (oo*x + zoo*y).as_numer_denom() == (zoo*y + oo*x, 1)

    # 定义符号 A, B, C，其中 commutative=False
    A, B, C = symbols('A,B,C', commutative=False)

    # 断言表达式 (A*B*C**-1).as_numer_denom() 返回 (A*B*C**-1, 1)
    assert (A*B*C**-1).as_numer_denom() == (A*B*C**-1, 1)
    # 断言表达式 (A*B*C**-1/x).as_numer_denom() 返回 (A*B*C**-1, x)
    assert (A*B*C**-1/x).as_numer_denom() == (A*B*C**-1, x)
    # 断言表达式 (C**-1*A*B).as_numer_denom() 返回 (C**-1*A*B, 1)
    assert (C**-1*A*B).as_numer_denom() == (C**-1*A*B, 1)
    # 断言表达式 (C**-1*A*B/x).as_numer_denom() 返回 (C**-1*A*B, x)
    assert (C**-1*A*B/x).as_numer_denom() == (C**-1*A*B, x)
    # 断言表达式 ((A*B*C)**-1).as_numer_denom() 返回 ((A*B*C)**-1, 1)
    assert ((A*B*C)**-1).as_numer_denom() == ((A*B*C)**-1, 1)
    # 断言表达式 ((A*B*C)**-1/x).as_numer_denom() 返回 ((A*B*C)**-1, x)
    assert ((A*B*C)**-1/x).as_numer_denom() == ((A*B*C)**-1, x)

    # 断言表达式 Add(0, (x + y)/z/-2, evaluate=False).as_numer_denom() 返回 (-x - y, 2*z)
    # 在处理过程中，这个表达式从 Add 转换为 Mul
    assert Add(0, (x + y)/z/-2, evaluate=False).as_numer_denom() == (-x - y, 2*z)
# 定义测试函数 test_trunc
def test_trunc():
    # 导入 math 模块
    import math
    # 符号变量定义
    x, y = symbols('x y')
    
    # 断言：对整数 2 进行截断，应返回自身
    assert math.trunc(2) == 2
    # 断言：对浮点数 4.57 进行截断，应返回整数部分 4
    assert math.trunc(4.57) == 4
    # 断言：对负数 -5.79 进行截断，应返回整数部分 -5
    assert math.trunc(-5.79) == -5
    # 断言：对常数 pi 进行截断，应返回整数部分 3
    assert math.trunc(pi) == 3
    # 断言：对 log(7) 进行截断，应返回整数部分 1
    assert math.trunc(log(7)) == 1
    # 断言：对 exp(5) 进行截断，应返回整数部分 148
    assert math.trunc(exp(5)) == 148
    # 断言：对 cos(pi) 进行截断，应返回整数部分 -1
    assert math.trunc(cos(pi)) == -1
    # 断言：对 sin(5) 进行截断，应返回整数部分 0
    assert math.trunc(sin(5)) == 0

    # 断言：对符号变量 x 执行截断应引发 TypeError 异常
    raises(TypeError, lambda: math.trunc(x))
    # 断言：对表达式 x + y**2 执行截断应引发 TypeError 异常
    raises(TypeError, lambda: math.trunc(x + y**2))
    # 断言：对正无穷 oo 执行截断应引发 TypeError 异常
    raises(TypeError, lambda: math.trunc(oo))


# 定义测试函数 test_as_independent
def test_as_independent():
    # 断言：零作为 x 的独立项，无论如何都应返回 (0, 0)
    assert S.Zero.as_independent(x, as_Add=True) == (0, 0)
    # 断言：零作为 x 的独立项，无论如何都应返回 (0, 0)
    assert S.Zero.as_independent(x, as_Add=False) == (0, 0)
    # 断言：对表达式 2*x*sin(x) + y + x，以 x 为基准的独立项应为 (y, x + 2*x*sin(x))
    assert (2*x*sin(x) + y + x).as_independent(x) == (y, x + 2*x*sin(x))
    # 断言：对表达式 2*x*sin(x) + y + x，以 y 为基准的独立项应为 (x + 2*x*sin(x), y)
    assert (2*x*sin(x) + y + x).as_independent(y) == (x + 2*x*sin(x), y)

    # 断言：对表达式 2*x*sin(x) + y + x，以 x 和 y 为基准的独立项应为 (0, y + x + 2*x*sin(x))
    assert (2*x*sin(x) + y + x).as_independent(x, y) == (0, y + x + 2*x*sin(x))

    # 断言：对表达式 x*sin(x)*cos(y)，以 x 为基准的独立项应为 (cos(y), x*sin(x))
    assert (x*sin(x)*cos(y)).as_independent(x) == (cos(y), x*sin(x))
    # 断言：对表达式 x*sin(x)*cos(y)，以 y 为基准的独立项应为 (x*sin(x), cos(y))
    assert (x*sin(x)*cos(y)).as_independent(y) == (x*sin(x), cos(y))

    # 断言：对表达式 x*sin(x)*cos(y)，以 x 和 y 为基准的独立项应为 (1, x*sin(x)*cos(y))
    assert (x*sin(x)*cos(y)).as_independent(x, y) == (1, x*sin(x)*cos(y))

    # 断言：对表达式 sin(x)，以 x 为基准的独立项应为 (1, sin(x))
    assert (sin(x)).as_independent(x) == (1, sin(x))
    # 断言：对表达式 sin(x)，以 y 为基准的独立项应为 (sin(x), 1)
    assert (sin(x)).as_independent(y) == (sin(x), 1)

    # 断言：对表达式 2*sin(x)，以 x 为基准的独立项应为 (2, sin(x))
    assert (2*sin(x)).as_independent(x) == (2, sin(x))
    # 断言：对表达式 2*sin(x)，以 y 为基准的独立项应为 (2*sin(x), 1)
    assert (2*sin(x)).as_independent(y) == (2*sin(x), 1)

    # issue 4903 = 1766b
    n1, n2, n3 = symbols('n1 n2 n3', commutative=False)
    # 断言：对表达式 n1 + n1*n2，以 n2 为基准的独立项应为 (n1, n1*n2)
    assert (n1 + n1*n2).as_independent(n2) == (n1, n1*n2)
    # 断言：对表达式 n2*n1 + n1*n2，以 n2 为基准的独立项应为 (0, n1*n2 + n2*n1)
    assert (n2*n1 + n1*n2).as_independent(n2) == (0, n1*n2 + n2*n1)
    # 断言：对表达式 n1*n2*n1，以 n2 为基准的独立项应为 (n1, n2*n1)
    assert (n1*n2*n1).as_independent(n2) == (n1, n2*n1)
    # 断言：对表达式 n1*n2*n1，以 n1 为基准的独立项应为 (1, n1*n2*n1)
    assert (n1*n2*n1).as_independent(n1) == (1, n1*n2*n1)

    # 断言：对表达式 3*x，以 x 为基准的独立项应为 (0, 3*x)
    assert (3*x).as_independent(x, as_Add=True) == (0, 3*x)
    # 断言：对表达式 3*x，以 x 为基准的独立项应为 (3, x)
    assert (3*x).as_independent(x, as_Add=False) == (3, x)
    # 断言：对表达式 3 + x，以 x 为基准的独立项应为 (3, x)
    assert (3 + x).as_independent(x, as_Add=True) == (3, x)
    # 断言：对表达式 3 + x，以 x 为基准的独立项应为 (1, 3 + x)
    assert (3 + x).as_independent(x, as_Add=False) == (1, 3 + x)

    # issue 5479
    # 断言：对表达式 3*x，以 Symbol 为基准的独立项应为 (3, x)
    assert (3*x).as_independent(Symbol) == (3, x)

    # issue 5648
    # 断言：对表达式 n1*x*y，以 x 为基准的独立项应为 (n1*y, x)
    assert (n1*x*y).as_independent(x) == (n1*y, x)
    # 断言：对表达式 (x + n1)*(x - y)，以 x 为基准的独立项应为 (1, (x + n1)*(x - y))
    assert ((x + n1)*(x - y)).as_independent(x) == (1, (x + n1)*(x - y))
    # 断言：对表达式 (x + n1)*(x - y
    # 断言，验证表达式 (x*y).as_independent(z, as_Add=True) == (x*y, 0)
    assert (x*y).as_independent(z, as_Add=True) == (x*y, 0)
@XFAIL
# 标记这个测试函数为期望失败的测试
def test_call_2():
    # TODO UndefinedFunction does not subclass Expr
    # 断言表达式的期望结果，这里检查表达式中函数 f 的使用情况
    assert (2*f)(x) == 2*f(x)


def test_replace():
    # 创建一个复杂的数学表达式 e
    e = log(sin(x)) + tan(sin(x**2))

    # 使用 replace 方法替换 e 中的 sin 函数为 cos 函数
    assert e.replace(sin, cos) == log(cos(x)) + tan(cos(x**2))
    
    # 使用 replace 方法替换 e 中的 sin 函数为 lambda 表达式 sin(2*a)
    assert e.replace(
        sin, lambda a: sin(2*a)) == log(sin(2*x)) + tan(sin(2*x**2))

    # 使用 Wild 对象创建通配符模式
    a = Wild('a')
    b = Wild('b')

    # 使用 replace 方法替换 e 中符合模式 sin(a) 的部分为 cos(a)
    assert e.replace(sin(a), cos(a)) == log(cos(x)) + tan(cos(x**2))
    
    # 使用 replace 方法替换 e 中符合模式 sin(a) 的部分为 lambda 表达式 sin(2*a)
    assert e.replace(
        sin(a), lambda a: sin(2*a)) == log(sin(2*x)) + tan(sin(2*x**2))
    
    # 测试 exact=True 的精确替换
    assert (2*x).replace(a*x + b, b - a, exact=True) == 2*x
    assert (2*x).replace(a*x + b, b - a) == 2*x
    assert (2*x).replace(a*x + b, b - a, exact=False) == 2/x
    assert (2*x).replace(a*x + b, lambda a, b: b - a, exact=True) == 2*x
    assert (2*x).replace(a*x + b, lambda a, b: b - a) == 2*x
    assert (2*x).replace(a*x + b, lambda a, b: b - a, exact=False) == 2/x

    # 创建另一个复杂的表达式 g
    g = 2*sin(x**3)

    # 使用 replace 方法替换 g 中所有数值的部分，将每个数值的平方
    assert g.replace(
        lambda expr: expr.is_Number, lambda expr: expr**2) == 4*sin(x**9)

    # 使用 replace 方法替换 cos(x) 为 sin(x)，并返回替换结果及映射
    assert cos(x).replace(cos, sin, map=True) == (sin(x), {cos(x): sin(x)})
    assert sin(x).replace(cos, sin) == sin(x)

    # 定义条件和替换函数
    cond, func = lambda x: x.is_Mul, lambda x: 2*x
    
    # 使用 replace 方法替换 x*y 符合条件的部分为 2*x*y，并返回替换结果及映射
    assert (x*y).replace(cond, func, map=True) == (2*x*y, {x*y: 2*x*y})
    
    # 使用 replace 方法替换 x*(1 + x*y) 符合条件的部分，返回替换结果及映射
    assert (x*(1 + x*y)).replace(cond, func, map=True) == \
        (2*x*(2*x*y + 1), {x*(2*x*y + 1): 2*x*(2*x*y + 1), x*y: 2*x*y})
    
    # 使用 replace 方法替换 y*sin(x) 的 sin 部分为 sin(x)/y，返回替换结果及映射
    assert (y*sin(x)).replace(sin, lambda expr: sin(expr)/y, map=True) == \
        (sin(x), {sin(x): sin(x)/y})
    
    # 如果非同时替换，将 y*sin(x) 的 sin 部分替换为 sin(x)/y = sin(x)/y
    assert (y*sin(x)).replace(sin, lambda expr: sin(expr)/y,
        simultaneous=False) == sin(x)/y
    
    # 使用 replace 方法替换 x**2 + O(x**3) 的 Pow 部分，返回替换结果
    assert (x**2 + O(x**3)).replace(Pow, lambda b, e: b**e/e
        ) == x**2/2 + O(x**3)
    
    # 使用 replace 方法替换 x**2 + O(x**3) 的 Pow 部分，非同时替换，返回替换结果
    assert (x**2 + O(x**3)).replace(Pow, lambda b, e: b**e/e,
        simultaneous=False) == x**2/2 + O(x**3)
    
    # 使用 replace 方法替换 x*(x*y + 3) 符合条件的部分，返回替换结果
    assert (x*(x*y + 3)).replace(lambda x: x.is_Mul, lambda x: 2 + x) == \
        x*(x*y + 5) + 2
    
    # 创建一个复杂的表达式 e
    e = (x*y + 1)*(2*x*y + 1) + 1
    
    # 使用 replace 方法替换 e 符合条件的部分，返回替换结果及映射
    assert e.replace(cond, func, map=True) == (
        2*((2*x*y + 1)*(4*x*y + 1)) + 1,
        {2*x*y: 4*x*y, x*y: 2*x*y, (2*x*y + 1)*(4*x*y + 1):
        2*((2*x*y + 1)*(4*x*y + 1))})
    
    # 使用 replace 方法将 x 替换为 y，返回替换结果
    assert x.replace(x, y) == y
    
    # 使用 replace 方法将 (x + 1) 的部分 1 替换为 2，返回替换结果
    assert (x + 1).replace(1, 2) == x + 2

    # 创建符号 n1, n2, n3，非可交换
    n1, n2, n3 = symbols('n1:4', commutative=False)
    
    # 使用 replace 方法替换 n1*f(n2) 中的 f 函数，返回替换结果
    assert (n1*f(n2)).replace(f, lambda x: x) == n1*n2
    
    # 使用 replace 方法替换 n3*f(n2) 中的 f 函数，返回替换结果
    assert (n3*f(n2)).replace(f, lambda x: x) == n3*n2
    
    # issue 16725 测试
    assert S.Zero.replace(Wild('x'), 1) == 1
    # 使用 exact=True，用户覆盖默认决策 False
    assert S.Zero.replace(Wild('x'), 1, exact=True) == 0


def test_find():
    # 创建一个包含多种数学元素的表达式 expr
    expr = (x + y + 2 + sin(3*x))

    # 使用 find 方法查找表达式中的整数，返回找到的结果集合
    assert expr.find(lambda u: u.is_Integer) == {S(2), S(3)}
    
    # 使用 find 方法查找表达式中的符号，返回找到的结果集合
    assert expr.find(lambda u: u.is_Symbol) == {x, y}

    # 使用 find 方法查找表达式中的整数，group=True，返回找到的结果集合及计数
    assert expr.find(lambda u: u.is_Integer, group=True) == {S(2): 1, S(3): 1}
    # 使用 assert 语句来检查表达式的查找结果是否符合预期
    assert expr.find(lambda u: u.is_Symbol, group=True) == {x: 2, y: 1}
    
    # 使用 assert 语句来检查表达式中整数类型的元素是否被正确查找到
    assert expr.find(Integer) == {S(2), S(3)}
    # 使用 assert 语句来检查表达式中符号类型的元素是否被正确查找到
    assert expr.find(Symbol) == {x, y}
    
    # 使用 assert 语句来检查在开启 group=True 模式下，整数类型的元素是否被正确查找到
    assert expr.find(Integer, group=True) == {S(2): 1, S(3): 1}
    # 使用 assert 语句来检查在开启 group=True 模式下，符号类型的元素是否被正确查找到
    assert expr.find(Symbol, group=True) == {x: 2, y: 1}
    
    # 创建一个通配符 'a'
    a = Wild('a')
    
    # 定义一个包含多个嵌套 sin 函数调用的表达式
    expr = sin(sin(x)) + sin(x) + cos(x) + x
    
    # 使用 assert 语句来检查 lambda 函数查找嵌套 sin 函数的结果是否符合预期
    assert expr.find(lambda u: type(u) is sin) == {sin(x), sin(sin(x))}
    # 使用 assert 语句来检查 lambda 函数在开启 group=True 模式下查找嵌套 sin 函数的结果是否符合预期
    assert expr.find(lambda u: type(u) is sin, group=True) == {sin(x): 2, sin(sin(x)): 1}
    
    # 使用 assert 语句来检查通配符 a 查找嵌套 sin(a) 的结果是否符合预期
    assert expr.find(sin(a)) == {sin(x), sin(sin(x))}
    # 使用 assert 语句来检查通配符 a 在开启 group=True 模式下查找嵌套 sin(a) 的结果是否符合预期
    assert expr.find(sin(a), group=True) == {sin(x): 2, sin(sin(x)): 1}
    
    # 使用 assert 语句来检查查找 sin 函数的结果是否符合预期
    assert expr.find(sin) == {sin(x), sin(sin(x))}
    # 使用 assert 语句来检查在开启 group=True 模式下查找 sin 函数的结果是否符合预期
    assert expr.find(sin, group=True) == {sin(x): 2, sin(sin(x)): 1}
# 定义一个测试函数，用于测试表达式的统计功能
def test_count():
    # 定义表达式
    expr = (x + y + 2 + sin(3*x))

    # 断言：统计整数类型的对象数量，期望结果为2
    assert expr.count(lambda u: u.is_Integer) == 2
    # 断言：统计符号类型的对象数量，期望结果为3
    assert expr.count(lambda u: u.is_Symbol) == 3

    # 断言：统计整数类型（Integer类）的对象数量，期望结果为2
    assert expr.count(Integer) == 2
    # 断言：统计符号类型（Symbol类）的对象数量，期望结果为3
    assert expr.count(Symbol) == 3
    # 断言：统计数字2的数量，期望结果为1
    assert expr.count(2) == 1

    # 创建一个Wild类对象'a'
    a = Wild('a')

    # 断言：统计sin函数的数量，期望结果为1
    assert expr.count(sin) == 1
    # 断言：统计具体sin(a)形式的数量，期望结果为1
    assert expr.count(sin(a)) == 1
    # 断言：通过判断类型来统计sin函数的数量，期望结果为1
    assert expr.count(lambda u: type(u) is sin) == 1

    # 断言：在f(x)中统计f(x)的数量，期望结果为1
    assert f(x).count(f(x)) == 1
    # 断言：在f(x).diff(x)中统计f(x)的数量，期望结果为1
    assert f(x).diff(x).count(f(x)) == 1
    # 断言：在f(x).diff(x)中统计x的数量，期望结果为2
    assert f(x).diff(x).count(x) == 2


# 定义测试函数，测试基础的has功能
def test_has_basics():
    # 创建一个Wild类对象'p'
    p = Wild('p')

    # 断言：sin(x)中是否包含符号x，期望结果为True
    assert sin(x).has(x)
    # 断言：sin(x)中是否包含sin函数，期望结果为True
    assert sin(x).has(sin)
    # 断言：sin(x)中是否包含符号y，期望结果为False
    assert not sin(x).has(y)
    # 断言：sin(x)中是否包含cos函数，期望结果为False
    assert not sin(x).has(cos)
    # 断言：f(x)中是否包含符号x，期望结果为True
    assert f(x).has(x)
    # 断言：f(x)中是否包含f函数，期望结果为True
    assert f(x).has(f)
    # 断言：f(x)中是否包含符号y，期望结果为False
    assert not f(x).has(y)
    # 断言：f(x)中是否包含函数g，期望结果为False
    assert not f(x).has(g)

    # 断言：f(x).diff(x)中是否包含符号x，期望结果为True
    assert f(x).diff(x).has(x)
    # 断言：f(x).diff(x)中是否包含函数f，期望结果为True
    assert f(x).diff(x).has(f)
    # 断言：f(x).diff(x)中是否包含Derivative类，期望结果为True
    assert f(x).diff(x).has(Derivative)
    # 断言：f(x).diff(x)中是否包含符号y，期望结果为False
    assert not f(x).diff(x).has(y)
    # 断言：f(x).diff(x)中是否包含函数g，期望结果为False
    assert not f(x).diff(x).has(g)
    # 断言：f(x).diff(x)中是否包含sin函数，期望结果为False
    assert not f(x).diff(x).has(sin)

    # 断言：x**2中是否包含符号Symbol，期望结果为True
    assert (x**2).has(Symbol)
    # 断言：x**2中是否包含Wild类，期望结果为False
    assert not (x**2).has(Wild)
    # 断言：2*p中是否包含Wild类，期望结果为True
    assert (2*p).has(Wild)

    # 断言：x中是否不包含任何东西，期望结果为False
    assert not x.has()

    # 断言：S(1)中是否包含Wild类，期望结果为False
    assert not S(1).has(Wild)
    # 断言：x中是否包含Wild类，期望结果为False
    assert not x.has(Wild)


# 定义测试函数，测试has功能的多个参数形式
def test_has_multiple():
    # 定义复杂表达式f
    f = x**2*y + sin(2**t + log(z))

    # 断言：f中是否包含符号x，期望结果为True
    assert f.has(x)
    # 断言：f中是否包含符号y，期望结果为True
    assert f.has(y)
    # 断言：f中是否包含符号z，期望结果为True
    assert f.has(z)
    # 断言：f中是否包含符号t，期望结果为True
    assert f.has(t)

    # 断言：f中是否不包含符号u，期望结果为False
    assert not f.has(u)

    # 断言：f中是否同时包含符号x, y, z, t，期望结果为True
    assert f.has(x, y, z, t)
    # 断言：f中是否同时包含符号x, y, z, t, u，期望结果为True
    assert f.has(x, y, z, t, u)

    # 定义整数对象i
    i = Integer(4400)

    # 断言：i中是否不包含符号x，期望结果为True
    assert not i.has(x)

    # 断言：i*x**i中是否包含符号x，期望结果为True
    assert (i*x**i).has(x)
    # 断言：i*y**i中是否包含符号x，期望结果为False
    assert not (i*y**i).has(x)
    # 断言：i*y**i中是否同时包含符号x, y，期望结果为True
    assert (i*y**i).has(x, y)
    # 断言：i*y**i中是否同时包含符号x, z，期望结果为False
    assert not (i*y**i).has(x, z)


# 定义测试函数，测试Piecewise对象的has功能
def test_has_piecewise():
    # 定义表达式f和Piecewise对象p
    f = (x*y + 3/y)**(3 + 2)
    p = Piecewise((g(x), x < -1), (1, x <= 1), (f, True))

    # 断言：p中是否包含符号x，期望结果为True
    assert p.has(x)
    # 断言：p中是否包含符号y，期望结果为True
    assert p.has(y)
    # 断言：p中是否不包含符号z，期望结果为False
    assert not p.has(z)
    # 断言：p中是否包含数字1，期望结果为True
    assert p.has(1)
    # 断言：p中是否包含数字3，期望结果为True
    assert p.has(3)
    # 断言：p中是否不包含数字4，期望结果为False
    assert not p.has(4)
    # 断言：p中是否包含表达式f，期望结果为True
    assert p.has(f)
    # 断言：p中是否包含函数g，期望结果为True
    assert p.has(g)
    # 断言：p中是否不包含函数h，期望结果为False
    assert not p.has(h)


# 定义测试函数，测试has功能的迭代使用
def test_has_iterative():
    # 创建三个非交
    # 使用 Tuple 类创建一个元组对象，包含两个元素：f(x) 和 g(x)，然后检查元组中是否包含 f(x)
    assert Tuple(f(x), g(x)).has(f(x))
    # XXX to be deprecated
    # 这个断言即将被弃用，可能是因为 Tuple 类或相关方法将不再支持某些功能或用法

    # 下面的断言测试 Tuple 类的 has 方法对不同参数的处理：
    # 断言 Tuple(f, g) 不包含 x
    #assert not Tuple(f, g).has(x)
    # 断言 Tuple(f, g) 包含 f
    #assert Tuple(f, g).has(f)
    # 断言 Tuple(f, g) 不包含 h
    #assert not Tuple(f, g).has(h)

    # 断言 Tuple(True) 包含 True
    assert Tuple(True).has(True)
    # 断言 Tuple(True) 包含 S.true（可能是某种特定的 True 值）
    assert Tuple(True).has(S.true)
    # 断言 Tuple(True) 不包含整数 1
    assert not Tuple(True).has(1)
def test_has_units():
    # 从 sympy.physics.units 模块导入单位 m 和 s
    from sympy.physics.units import m, s

    # 检查表达式 (x*m/s) 是否包含变量 x
    assert (x*m/s).has(x)
    # 检查表达式 (x*m/s) 是否同时包含变量 y 和 z
    assert (x*m/s).has(y, z) is False


def test_has_polys():
    # 创建多项式 poly = x**2 + x*y*sin(z)，其中 x, y, z, t 是变量
    poly = Poly(x**2 + x*y*sin(z), x, y, t)

    # 检查多项式 poly 是否包含变量 x
    assert poly.has(x)
    # 检查多项式 poly 是否同时包含变量 x, y, z
    assert poly.has(x, y, z)
    # 检查多项式 poly 是否同时包含变量 x, y, z, t
    assert poly.has(x, y, z, t)


def test_has_physics():
    # 创建 FockState 对象，检查其是否包含变量 x
    assert FockState((x, y)).has(x)


def test_as_poly_as_expr():
    # 创建表达式 f = x**2 + 2*x*y
    f = x**2 + 2*x*y

    # 将 f 转换为多项式，再转换回表达式，应该得到原表达式 f
    assert f.as_poly().as_expr() == f
    # 将 f 转换为以 x, y 为变量的多项式，再转换回表达式，应该得到原表达式 f
    assert f.as_poly(x, y).as_expr() == f

    # 将 f + sin(x) 转换为以 x, y 为变量的多项式，应该返回 None
    assert (f + sin(x)).as_poly(x, y) is None

    # 创建 Poly 对象 p，检查其转换为多项式后是否等于自身
    p = Poly(f, x, y)
    assert p.as_poly() == p

    # 测试特定问题 https://github.com/sympy/sympy/issues/20610
    # S(2) 不可转换为多项式，应返回 None
    assert S(2).as_poly() is None
    # sqrt(2) 不可转换为多项式，应返回 None
    assert sqrt(2).as_poly(extension=True) is None

    # 测试在 Tuple 中的元素不可转换为多项式的情况，应引发 AttributeError
    raises(AttributeError, lambda: Tuple(x, x).as_poly(x))
    raises(AttributeError, lambda: Tuple(x ** 2, x, y).as_poly(x))


def test_nonzero():
    # 测试各种类型的对象是否为非零
    assert bool(S.Zero) is False
    assert bool(S.One) is True
    assert bool(x) is True
    assert bool(x + y) is True
    assert bool(x - x) is False
    assert bool(x*y) is True
    assert bool(x*1) is True
    assert bool(x*0) is False


def test_is_number():
    # 测试各种对象是否为数值类型
    assert Float(3.14).is_number is True
    assert Integer(737).is_number is True
    assert Rational(3, 2).is_number is True
    assert Rational(8).is_number is True
    assert x.is_number is False
    assert (2*x).is_number is False
    assert (x + y).is_number is False
    assert log(2).is_number is True
    assert log(x).is_number is False
    assert (2 + log(2)).is_number is True
    assert (8 + log(2)).is_number is True
    assert (2 + log(x)).is_number is False
    assert (8 + log(2) + x).is_number is False
    assert (1 + x**2/x - x).is_number is True
    assert Tuple(Integer(1)).is_number is False
    assert Add(2, x).is_number is False
    assert Mul(3, 4).is_number is True
    assert Pow(log(2), 2).is_number is True
    assert oo.is_number is True
    g = WildFunction('g')
    assert g.is_number is False
    assert (2*g).is_number is False
    assert (x**2).subs(x, 3).is_number is True

    # 测试在 Basic 子类中扩展 .is_number 的可行性
    class A(Basic):
        pass
    a = A()
    assert a.is_number is False


def test_as_coeff_add():
    # 测试各种对象的 .as_coeff_add() 方法
    assert S(2).as_coeff_add() == (2, ())
    assert S(3.0).as_coeff_add() == (0, (S(3.0),))
    assert S(-3.0).as_coeff_add() == (0, (S(-3.0),))
    assert x.as_coeff_add() == (0, (x,))
    assert (x - 1).as_coeff_add() == (-1, (x,))
    assert (x + 1).as_coeff_add() == (1, (x,))
    assert (x + 2).as_coeff_add() == (2, (x,))
    assert (x + y).as_coeff_add(y) == (x, (y,))
    assert (3*x).as_coeff_add(y) == (3*x, ())
    # 不展开表达式
    e = (x + y)**2
    assert e.as_coeff_add(y) == (0, (e,))


def test_as_coeff_mul():
    # 测试各种对象的 .as_coeff_mul() 方法
    assert S(2).as_coeff_mul() == (2, ())
    assert S(3.0).as_coeff_mul() == (1, (S(3.0),))
    assert S(-3.0).as_coeff_mul() == (-1, (S(3.0),))
    assert S(-3.0).as_coeff_mul(rational=False) == (-S(3.0), ())
    assert x.as_coeff_mul() == (1, (x,))
    # 断言，负数乘以 x 的系数分解为 (-1, (x,))
    assert (-x).as_coeff_mul() == (-1, (x,))
    # 断言，2 乘以 x 的系数分解为 (2, (x,))
    assert (2*x).as_coeff_mul() == (2, (x,))
    # 断言，x*y 的关于 y 的系数分解为 (x, (y,))
    assert (x*y).as_coeff_mul(y) == (x, (y,))
    # 断言，3 + x 的系数分解为 (1, (3 + x,))
    assert (3 + x).as_coeff_mul() == (1, (3 + x,))
    # 断言，3 + x 的关于 y 的系数分解为 (3 + x, ())
    assert (3 + x).as_coeff_mul(y) == (3 + x, ())
    # 不进行展开
    # 创建 exp(x + y) 的对象 e
    e = exp(x + y)
    # 断言，e 关于 y 的系数分解为 (1, (e,))
    assert e.as_coeff_mul(y) == (1, (e,))
    # 创建 2**(x + y) 的对象 e
    e = 2**(x + y)
    # 断言，e 关于 y 的系数分解为 (1, (e,))
    assert e.as_coeff_mul(y) == (1, (e,))
    # 断言，1.1*x 的非有理数系数分解为 (1.1, (x,))
    assert (1.1*x).as_coeff_mul(rational=False) == (1.1, (x,))
    # 断言，1.1*x 的系数分解为 (1, (1.1, x))
    assert (1.1*x).as_coeff_mul() == (1, (1.1, x))
    # 断言，-oo*x 的有理数系数分解为 (-1, (oo, x))
    assert (-oo*x).as_coeff_mul(rational=True) == (-1, (oo, x))
# 定义一个测试函数，用于测试符号表达式的系数和指数提取功能
def test_as_coeff_exponent():
    # 断言：提取 3*x**4 的系数和指数应该是 (3, 4)
    assert (3*x**4).as_coeff_exponent(x) == (3, 4)
    # 断言：提取 2*x**3 的系数和指数应该是 (2, 3)
    assert (2*x**3).as_coeff_exponent(x) == (2, 3)
    # 断言：提取 4*x**2 的系数和指数应该是 (4, 2)
    assert (4*x**2).as_coeff_exponent(x) == (4, 2)
    # 断言：提取 6*x 的系数和指数应该是 (6, 1)
    assert (6*x**1).as_coeff_exponent(x) == (6, 1)
    # 断言：提取 3*x**0 的系数和指数应该是 (3, 0)
    assert (3*x**0).as_coeff_exponent(x) == (3, 0)
    # 断言：提取 2*x**0 的系数和指数应该是 (2, 0)
    assert (2*x**0).as_coeff_exponent(x) == (2, 0)
    # 断言：提取 1*x**0 的系数和指数应该是 (1, 0)
    assert (1*x**0).as_coeff_exponent(x) == (1, 0)
    # 断言：提取 0*x**0 的系数和指数应该是 (0, 0)
    assert (0*x**0).as_coeff_exponent(x) == (0, 0)
    # 断言：提取 -1*x**0 的系数和指数应该是 (-1, 0)
    assert (-1*x**0).as_coeff_exponent(x) == (-1, 0)
    # 断言：提取 -2*x**0 的系数和指数应该是 (-2, 0)
    assert (-2*x**0).as_coeff_exponent(x) == (-2, 0)
    # 断言：提取 2*x**3 + pi*x**3 的系数和指数应该是 (2 + pi, 3)
    assert (2*x**3 + pi*x**3).as_coeff_exponent(x) == (2 + pi, 3)
    # 断言：对符号表达式 f(x) 求导，并提取其关于 f(x) 的系数和指数应该是 (fx, 0)
    # issue 4784
    D = Derivative
    fx = D(f(x), x)
    assert fx.as_coeff_exponent(f(x)) == (fx, 0)


# 定义一个测试函数，用于测试符号表达式的乘法提取和加法提取功能
def test_extractions():
    # 对每个基数进行测试
    for base in (2, S.Exp1):
        # 断言：从 Pow(base**x, 3) 中提取 base**x 应该得到 base**(2*x)
        assert Pow(base**x, 3, evaluate=False).extract_multiplicatively(base**x) == base**(2*x)
        # 断言：从 base**(5*x) 中提取 base**(3*x) 应该得到 base**(2*x)
        assert (base**(5*x)).extract_multiplicatively(base**(3*x)) == base**(2*x)
    # 断言：从 (x*y)**3 中提取 x**2 * y 应该得到 x * y**2
    assert ((x*y)**3).extract_multiplicatively(x**2 * y) == x*y**2
    # 断言：从 (x*y)**3 中提取 x**4 * y 应该得到 None
    assert ((x*y)**3).extract_multiplicatively(x**4 * y) is None
    # 断言：从 2*x 中提取 2 应该得到 x
    assert (2*x).extract_multiplicatively(2) == x
    # 断言：从 2*x 中提取 3 应该得到 None
    assert (2*x).extract_multiplicatively(3) is None
    # 断言：从 2*x 中提取 -1 应该得到 None
    assert (2*x).extract_multiplicatively(-1) is None
    # 断言：从 S.Half*x 中提取 3 应该得到 x/6
    assert (S.Half*x).extract_multiplicatively(3) == x/6
    # 断言：从 sqrt(x) 中提取 x 应该得到 None
    assert (sqrt(x)).extract_multiplicatively(x) is None
    # 断言：从 sqrt(x) 中提取 1/x 应该得到 None
    assert (sqrt(x)).extract_multiplicatively(1/x) is None
    # 断言：从 x 中提取 -x 应该得到 None
    assert x.extract_multiplicatively(-x) is None
    # 断言：从 -2 - 4*I 中提取 -2 应该得到 1 + 2*I
    assert (-2 - 4*I).extract_multiplicatively(-2) == 1 + 2*I
    # 断言：从 -2 - 4*I 中提取 3 应该得到 None
    assert (-2 - 4*I).extract_multiplicatively(3) is None
    # 断言：从 -2*x - 4*y - 8 中提取 -2 应该得到 x + 2*y + 4
    assert (-2*x - 4*y - 8).extract_multiplicatively(-2) == x + 2*y + 4
    # 断言：从 -2*x*y - 4*x**2*y 中提取 -2*y 应该得到 2*x**2 + x
    assert (-2*x*y - 4*x**2*y).extract_multiplicatively(-2*y) == 2*x**2 + x
    # 断言：从 2*x*y + 4*x**2*y 中提取 2*y 应该得到 2*x**2 + x
    assert (2*x*y + 4*x**2*y).extract_multiplicatively(2*y) == 2*x**2 + x
    # 断言：从 -4*y**2*x 中提取 -3*y 应该得到 None
    assert (-4*y**2*x).extract_multiplicatively(-3*y) is None
    # 断言：从 2*x 中提取 1 应该得到 2*x
    assert (2*x).extract_multiplicatively(1) == 2*x
    # 断言：从 -oo 中提取 5 应该得到 -oo
    assert (-oo).extract_multiplicatively(5) is -oo
    # 断言：从 oo 中提取 5 应该得到 oo
    assert (oo).extract_multiplicatively(5) is oo

    # 断言：从 ((x*y)**3) 中提取 1 应该得到 None
    assert ((x*y)**3).extract_additively(1) is None
    # 断言：从 x + 1 中提取 x 应该得到 1
    assert (x + 1).extract_additively(x) == 1
    # 断言：从 x + 1 中提取 2*x 应该得到 None
    assert (x + 1).extract_additively(2*x) is None
    # 断言：从 x + 1 中提取 -x 应该得到 None
    assert (x + 1).extract_additively(-x) is None
    # 断言：从 -x + 1 中提取 2*x 应该得到 None
    assert (-x + 1).extract_additively(2*x) is None
    # 断言：从 2*x + 3 中提取 x 应该得到 x + 3
    assert (2*x + 3).extract_additively(x) == x + 3
    # 断言：从 2*x + 3 中提取 2 应该得到 2*x + 1
    assert (2*x + 3).extract_additively(2) == 2*x + 1
    # 断言：从 2*x + 3 中提取 3 应该得到 2*x
    assert (2*x + 3).extract_additively(3) == 2*x
    # 断言：从 2*x + 3 中提取 -
    # 断言：对于表达式 S(2*x - 3)，试图从中提取出 x + 1 的加法项，预期返回 None
    assert S(2*x - 3).extract_additively(x + 1) is None

    # 断言：对于表达式 S(2*x - 3)，试图从中提取出 y + z 的加法项，预期返回 None
    assert S(2*x - 3).extract_additively(y + z) is None

    # 断言：对于表达式 ((a + 1)*x*4 + y)，试图从中提取出 x 的加法项，并展开后应该等于 4*a*x + 3*x + y
    assert ((a + 1)*x*4 + y).extract_additively(x).expand() == \
        4*a*x + 3*x + y

    # 断言：对于表达式 ((a + 1)*x*4 + 3*y)，试图从中提取出 x + 2*y 的加法项，并展开后应该等于 4*a*x + 3*x + y
    assert ((a + 1)*x*4 + 3*y).extract_additively(x + 2*y).expand() == \
        4*a*x + 3*x + y

    # 断言：对于表达式 y*(x + 1)，试图从中提取出 x + 1 的加法项，预期返回 None
    assert (y*(x + 1)).extract_additively(x + 1) is None

    # 断言：对于表达式 ((y + 1)*(x + 1) + 3)，试图从中提取出 x + 1 的加法项，并期望结果为 y*(x + 1) + 3
    assert ((y + 1)*(x + 1) + 3).extract_additively(x + 1) == \
        y*(x + 1) + 3

    # 断言：对于表达式 ((x + y)*(x + 1) + x + y + 3)，试图从中提取出 x + y 的加法项，并期望结果为 x*(x + y) + 3
    assert ((x + y)*(x + 1) + x + y + 3).extract_additively(x + y) == \
        x*(x + y) + 3

    # 断言：对于表达式 (x + y + 2*((x + y)*(x + 1)) + 3)，试图从中提取出 (x + y)*(x + 1) 的加法项，并期望结果为 x + y + (x + 1)*(x + y) + 3
    assert (x + y + 2*((x + y)*(x + 1)) + 3).extract_additively((x + y)*(x + 1)) == \
        x + y + (x + 1)*(x + y) + 3

    # 断言：对于表达式 ((y + 1)*(x + 2*y + 1) + 3)，试图从中提取出 y + 1 的加法项，并期望结果为 (x + 2*y)*(y + 1) + 3
    assert ((y + 1)*(x + 2*y + 1) + 3).extract_additively(y + 1) == \
        (x + 2*y)*(y + 1) + 3

    # 断言：对于表达式 (-x - x*I)，试图从中提取出 -x 的加法项，并期望结果为 -I*x
    assert (-x - x*I).extract_additively(-x) == -I*x

    # 断言：对于表达式 (4*x*(y + 1) + y)，试图从中提取出 x 的加法项，并期望结果为 x*(4*y + 3) + y
    assert (4*x*(y + 1) + y).extract_additively(x) == x*(4*y + 3) + y

    # 创建一个符号变量 n，其被限定为整数
    n = Symbol("n", integer=True)

    # 断言：对于表达式 Integer(-3)，判断是否能提取负号，预期为 True
    assert (Integer(-3)).could_extract_minus_sign() is True

    # 断言：对于表达式 (-n*x + x)，判断是否能提取负号，预期结果与 (n*x - x) 的提取负号结果不同
    assert (-n*x + x).could_extract_minus_sign() != \
        (n*x - x).could_extract_minus_sign()

    # 断言：对于表达式 (x - y)，判断是否能提取负号，预期结果与 (-x + y) 的提取负号结果不同
    assert (x - y).could_extract_minus_sign() != \
        (-x + y).could_extract_minus_sign()

    # 断言：对于表达式 (1 - x - y)，判断是否能提取负号，预期为 True
    assert (1 - x - y).could_extract_minus_sign() is True

    # 断言：对于表达式 (1 - x + y)，判断是否能提取负号，预期为 False
    assert (1 - x + y).could_extract_minus_sign() is False

    # 断言：对于表达式 ((-x - x*y)/y)，判断是否能提取负号，预期为 False
    assert ((-x - x*y)/y).could_extract_minus_sign() is False

    # 断言：对于表达式 ((x + x*y)/(-y))，判断是否能提取负号，预期为 True
    assert ((x + x*y)/(-y)).could_extract_minus_sign() is True

    # 断言：对于表达式 ((x + x*y)/y)，判断是否能提取负号，预期为 False
    assert ((x + x*y)/y).could_extract_minus_sign() is False

    # 断言：对于表达式 ((-x - y)/(x + y))，判断是否能提取负号，预期为 False
    assert ((-x - y)/(x + y)).could_extract_minus_sign() is False

    # 定义一个类 sign_invariant，继承自 Function 和 Expr，且 nargs 为 1
    class sign_invariant(Function, Expr):
        nargs = 1

        # 重写 __neg__ 方法，返回自身
        def __neg__(self):
            return self

    # 创建一个 sign_invariant 类的实例 foo，传入参数 x
    foo = sign_invariant(x)

    # 断言：对于实例 foo，判断其与其负数是否相等，预期为 False
    assert foo == -foo

    # 断言：对于实例 foo，判断其是否能提取负号，预期为 False
    assert foo.could_extract_minus_sign() is False

    # 断言：对于表达式 (x - y)，判断是否能提取负号，预期为 False
    assert (x - y).could_extract_minus_sign() is False

    # 断言：对于表达式 (-x + y)，判断是否能提取负号，预期为 True
    assert (-x + y).could_extract_minus_sign() is True

    # 断言：对于表达式 (x - 1)，判断是否能提取负号，预期为 False
    assert (x - 1).could_extract_minus_sign() is False

    # 断言：对于表达式 (1 - x)，判断是否能提取负号，预期为 True
    assert (1 - x).could_extract_minus_sign() is True

    # 断言：对于表达式 (sqrt(2) - 1)，判断是否能提取负号，预期为 True
    assert (sqrt(2) - 1).could_extract_minus_sign() is True

    # 断言：对于表达式 (1 - sqrt(2))，判断是否能提取负号，预期为 False
    assert (1 - sqrt(2)).could_extract_minus_sign() is False

    # 断言：检查结果是否是规范形式
    eq = (3*x + 15*y).extract_multiplicatively(3)
    assert eq.args == eq.func(*eq.args).args
# 定义测试函数，用于测试 NaN (Not a Number) 类型的值的抽取功能
def test_nan_extractions():
    # 对于给定的四个值 (1, 0, I, nan)，验证 NaN 对象的加法抽取操作返回 None
    for r in (1, 0, I, nan):
        assert nan.extract_additively(r) is None
        # 验证 NaN 对象的乘法抽取操作返回 None
        assert nan.extract_multiplicatively(r) is None


# 定义测试函数，用于测试表达式中的系数操作
def test_coeff():
    # 验证表达式 (x + 1) 中 x + 1 的系数为 1
    assert (x + 1).coeff(x + 1) == 1
    # 验证表达式 3*x 中 0 的系数为 0
    assert (3*x).coeff(0) == 0
    # 验证表达式 z*(1 + x)*x**2 中 1 + x 的系数为 z*x**2
    assert (z*(1 + x)*x**2).coeff(1 + x) == z*x**2
    # 验证表达式 1 + 2*x*x**(1 + x) 中 x*x**(1 + x) 的系数为 2
    assert (1 + 2*x*x**(1 + x)).coeff(x*x**(1 + x)) == 2
    # 验证表达式 1 + 2*x**(y + z) 中 x**(y + z) 的系数为 2
    assert (1 + 2*x**(y + z)).coeff(x**(y + z)) == 2
    # 验证表达式 3 + 2*x + 4*x**2 中 1 的系数为 0
    assert (3 + 2*x + 4*x**2).coeff(1) == 0
    # 验证表达式 3 + 2*x + 4*x**2 中 -1 的系数为 0
    assert (3 + 2*x + 4*x**2).coeff(-1) == 0
    # 验证表达式 3 + 2*x + 4*x**2 中 x 的系数为 2
    assert (3 + 2*x + 4*x**2).coeff(x) == 2
    # 验证表达式 3 + 2*x + 4*x**2 中 x**2 的系数为 4
    assert (3 + 2*x + 4*x**2).coeff(x**2) == 4
    # 验证表达式 3 + 2*x + 4*x**2 中 x**3 的系数为 0
    assert (3 + 2*x + 4*x**2).coeff(x**3) == 0

    # 验证表达式 -x/8 + x*y 中 x 的系数为 Rational(-1, 8) + y
    assert (-x/8 + x*y).coeff(x) == Rational(-1, 8) + y
    # 验证表达式 -x/8 + x*y 中 -x 的系数为 S.One/8
    assert (-x/8 + x*y).coeff(-x) == S.One/8
    # 验证表达式 4*x 中 2*x 的系数为 0
    assert (4*x).coeff(2*x) == 0
    # 验证表达式 2*x 中 2*x 的系数为 1
    assert (2*x).coeff(2*x) == 1
    # 验证表达式 -oo*x 中 x*oo 的系数为 -1
    assert (-oo*x).coeff(x*oo) == -1
    # 验证表达式 10*x 中 x 的系数为 0
    assert (10*x).coeff(x, 0) == 0
    # 验证表达式 10*x 中 10*x 的系数为 0
    assert (10*x).coeff(10*x, 0) == 0

    # 定义非交换符号 n1, n2，并验证表达式 n1*n2 中 n1 的系数为 1
    n1, n2 = symbols('n1 n2', commutative=False)
    assert (n1*n2).coeff(n1) == 1
    # 验证表达式 n1*n2 中 n2 的系数为 n1
    assert (n1*n2).coeff(n2) == n1
    # 验证表达式 n1*n2 + x*n1 中 n1 的系数为 1
    assert (n1*n2 + x*n1).coeff(n1) == 1  # 1*n1*(n2+x)
    # 验证表达式 n2*n1 + x*n1 中 n1 的系数为 n2 + x
    assert (n2*n1 + x*n1).coeff(n1) == n2 + x
    # 验证表达式 n2*n1 + x*n1**2 中 n1 的系数为 n2
    assert (n2*n1 + x*n1**2).coeff(n1) == n2
    # 验证表达式 n1**x 中 n1 的系数为 0
    assert (n1**x).coeff(n1) == 0
    # 验证表达式 n1*n2 + n2*n1 中 n1 的系数为 0
    assert (n1*n2 + n2*n1).coeff(n1) == 0
    # 验证表达式 2*(n1 + n2)*n2 中 n1 + n2 的系数为 n2
    assert (2*(n1 + n2)*n2).coeff(n1 + n2, right=1) == n2
    # 验证表达式 2*(n1 + n2)*n2 中 n1 + n2 的系数为 2
    assert (2*(n1 + n2)*n2).coeff(n1 + n2, right=0) == 2

    # 验证表达式 2*f(x) + 3*f(x).diff(x) 中 f(x) 的系数为 2
    assert (2*f(x) + 3*f(x).diff(x)).coeff(f(x)) == 2

    # 定义表达式 expr 和 expr2，并验证其系数操作
    expr = z*(x + y)**2
    expr2 = z*(x + y)**2 + z*(2*x + 2*y)**2
    # 验证表达式 expr 中 z 的系数为 (x + y)**2
    assert expr.coeff(z) == (x + y)**2
    # 验证表达式 expr 中 x + y 的系数为 0
    assert expr.coeff(x + y) == 0
    # 验证表达式 expr2 中 z 的系数为 (x + y)**2 + (2*x + 2*y)**2
    assert expr2.coeff(z) == (x + y)**2 + (2*x + 2*y)**2

    # 验证表达式 x + y + 3*z 中 1 的系数为 x + y
    assert (x + y + 3*z).coeff(1) == x + y
    # 验证表达式 -x + 2*y 中 -1 的系数为 x
    assert (-x + 2*y).coeff(-1) == x
    # 验证表达式 x - 2*y 中 -1 的系数为 2*y
    assert (x - 2*y).coeff(-1) == 2*y
    # 验证表达式 3 + 2*x + 4*x**2 中 1 的系数为 0
    assert (3 + 2*x + 4*x**2).coeff(1) == 0
    # 验证表达式 -x - 2*y 中 2 的系数为 -y
    assert (-x - 2*y).coeff(2) == -y
    # 验证表达式 x + sqrt(2)*x 中 sqrt(2) 的系数为 x
    assert (x + sqrt(2)*x).coeff(sqrt(2)) == x
    # 验证表达式 3 + 2*x + 4*x**2 中 x 的系数为 2
    assert (3 + 2*x + 4*x**2).coeff(x) == 2
    # 验证表达式 3 + 2*x + 4*x**2 中 x**2 的系数为 4
    assert (3 +
    # 断言：检查在表达式 x*y 中，z 的系数是否为 0，预期结果为 x*y
    assert (x*y).coeff(z, 0) == x*y
    
    # 断言：检查在表达式 x*n + y*n + z*m 中，n 的系数是否为 x + y
    assert (x*n + y*n + z*m).coeff(n) == x + y
    
    # 断言：检查在表达式 n*m + n*o + o*l 中，右边相对于 n 的系数是否为 m + o
    assert (n*m + n*o + o*l).coeff(n, right=True) == m + o
    
    # 断言：检查在表达式 x*n*m*n + y*n*m*o + z*l 中，右边相对于 m 的系数是否为 x*n + y*o
    assert (x*n*m*n + y*n*m*o + z*l).coeff(m, right=True) == x*n + y*o
    
    # 断言：检查在表达式 x*n*m*n + x*n*m*o + z*l 中，右边相对于 m 的系数是否为 n + o
    assert (x*n*m*n + x*n*m*o + z*l).coeff(m, right=True) == n + o
    
    # 断言：检查在表达式 x*n*m*n + x*n*m*o + z*l 中，左边相对于 m 的系数是否为 x*n
    assert (x*n*m*n + x*n*m*o + z*l).coeff(m) == x*n
# 测试 g 中 psi(r) 对 r 的一阶导数的系数是否为 2/r
def test_coeff2():
    # 定义符号 r 和 kappa
    r, kappa = symbols('r, kappa')
    # 定义函数 psi(r)
    psi = Function("psi")
    # 计算 g 的表达式
    g = 1/r**2 * (2*r*psi(r).diff(r, 1) + r**2 * psi(r).diff(r, 2))
    # 展开 g 的表达式
    g = g.expand()
    # 断言 psi(r) 对 r 的一阶导数的系数是否为 2/r
    assert g.coeff(psi(r).diff(r)) == 2/r


# 测试 g 中 psi(r) 对 r 的二阶导数的系数是否为 1
def test_coeff2_0():
    # 定义符号 r 和 kappa
    r, kappa = symbols('r, kappa')
    # 定义函数 psi(r)
    psi = Function("psi")
    # 计算 g 的表达式
    g = 1/r**2 * (2*r*psi(r).diff(r, 1) + r**2 * psi(r).diff(r, 2))
    # 展开 g 的表达式
    g = g.expand()
    
    # 断言 psi(r) 对 r 的二阶导数的系数是否为 1
    assert g.coeff(psi(r).diff(r, 2)) == 1


# 测试 expr 中 z 的系数
def test_coeff_expand():
    # 定义符号 x, y, z
    expr = z*(x + y)**2
    expr2 = z*(x + y)**2 + z*(2*x + 2*y)**2
    # 断言 expr 中 z 的系数是否为 (x + y)**2
    assert expr.coeff(z) == (x + y)**2
    # 断言 expr2 中 z 的系数是否为 (x + y)**2 + (2*x + 2*y)**2
    assert expr2.coeff(z) == (x + y)**2 + (2*x + 2*y)**2


# 测试 x 的积分
def test_integrate():
    # 断言 x 的积分是否为 x**2/2
    assert x.integrate(x) == x**2/2
    # 断言 x 在区间 [0, 1] 上的积分是否为 1/2
    assert x.integrate((x, 0, 1)) == S.Half


# 测试 as_base_exp 方法
def test_as_base_exp():
    # 断言 x 的 as_base_exp 方法返回 (x, 1)
    assert x.as_base_exp() == (x, S.One)
    # 断言 (x*y*z) 的 as_base_exp 方法返回 (x*y*z, 1)
    assert (x*y*z).as_base_exp() == (x*y*z, S.One)
    # 断言 (x + y + z) 的 as_base_exp 方法返回 (x + y + z, 1)
    assert (x + y + z).as_base_exp() == (x + y + z, S.One)
    # 断言 ((x + y)**z) 的 as_base_exp 方法返回 (x + y, z)
    assert ((x + y)**z).as_base_exp() == (x + y, z)


# 测试相关方法是否存在于对象中
def test_issue_4963():
    # 断言 Mul(x, y) 对象具有属性 "is_commutative"
    assert hasattr(Mul(x, y), "is_commutative")
    # 断言 Mul(x, y, evaluate=False) 对象具有属性 "is_commutative"
    assert hasattr(Mul(x, y, evaluate=False), "is_commutative")
    # 断言 Pow(x, y) 对象具有属性 "is_commutative"
    assert hasattr(Pow(x, y), "is_commutative")
    # 断言 Pow(x, y, evaluate=False) 对象具有属性 "is_commutative"
    assert hasattr(Pow(x, y, evaluate=False), "is_commutative")
    # 定义表达式 expr
    expr = Mul(Pow(2, 2, evaluate=False), 3, evaluate=False) + 1
    # 断言 expr 对象具有属性 "is_commutative"
    assert hasattr(expr, "is_commutative")


# 测试 nsimplify 方法
def test_action_verbs():
    # 断言 nsimplify 方法对 1/(exp(3*pi*x/5) + 1) 的简化是否正确
    assert nsimplify(1/(exp(3*pi*x/5) + 1)) == (1/(exp(3*pi*x/5) + 1)).nsimplify()
    # 断言 ratsimp 方法对 1/x + 1/y 的简化是否正确
    assert ratsimp(1/x + 1/y) == (1/x + 1/y).ratsimp()
    # 断言 trigsimp 方法对 log(x) 的简化是否正确
    assert trigsimp(log(x), deep=True) == (log(x)).trigsimp(deep=True)
    # 断言 radsimp 方法对 1/(2 + sqrt(2)) 的简化是否正确
    assert radsimp(1/(2 + sqrt(2))) == (1/(2 + sqrt(2))).radsimp()
    # 断言 radsimp 方法对 1/(a + b*sqrt(c)) 的简化是否正确，symbolic 参数设置为 False
    assert radsimp(1/(a + b*sqrt(c)), symbolic=False) == (1/(a + b*sqrt(c))).radsimp(symbolic=False)
    # 断言 powsimp 方法对 x**y*x**z*y**z 的简化是否正确，combine 参数设置为 'all'
    assert powsimp(x**y*x**z*y**z, combine='all') == (x**y*x**z*y**z).powsimp(combine='all')
    # 断言 (x**t*y**t).powsimp 方法使用 force=True 参数得到正确的结果
    assert (x**t*y**t).powsimp(force=True) == (x*y)**t
    # 断言 simplify 方法对 x**y*x**z*y**z 的简化是否正确
    assert simplify(x**y*x**z*y**z) == (x**y*x**z*y**z).simplify()
    # 断言 together 方法对 1/x + 1/y 的合并是否正确
    assert together(1/x + 1/y) == (1/x + 1/y).together()
    # 断言 collect 方法对 a*x**2 + b*x**2 + a*x - b*x + c 的收集是否正确，以 x 为参数
    assert collect(a*x**2 + b*x**2 + a*x - b*x + c, x) == (a*x**2 + b*x**2 + a*x - b*x + c).collect(x)
    # 断言 apart 方法对 y/(y + 2)/(y + 1) 的分解是否正确，以 y 为参数
    assert apart(y/(y + 2)/(y + 1), y) == (y/(y + 2)/(y + 1)).apart(y)
    # 断言 combsimp 方法对 y/(x + 2)/(x + 1) 的组合简化是否正确
    assert combsimp(y/(x + 2)/(x + 1)) == (y/(x + 2)/(x + 1)).combsimp()
    # 断言 gammasimp 方法对 gamma(x)/gamma(x-5) 的 gamma 函数简化是否正确
    assert gammasimp(gamma(x)/gamma(x-5)) == (gamma(x)/gamma(x-5)).gammasimp()
    # 断言 factor 方法对 x**2 + 5*x + 6 的因式分解是否正确
    assert factor(x**2 + 5*x + 6) == (x**2 + 5*x + 6).factor()
    # 断言 refine 方法对 sqrt(x**2) 的细化是否正确
    assert refine(sqrt(x**2)) == sqrt(x**2).refine()
    # 断言 cancel 方法对 (x**2 + 5*x + 6)/(x + 2) 的取消是否正确
    assert cancel((x**2 + 5*x + 6)/(x + 2)) == ((x**2 + 5*x + 6)/(x + 2)).cancel()


# 测试 as_powers_dict 方法
def test_as_powers_dict():
    # 断言 x 的 as_powers_dict 方法返回 {x: 1}
    assert x.as_powers_dict() == {x: 1}
    # 断言 (x**y*z) 的 as_powers_dict 方法返回 {x: y, z: 1}
    assert (x**y*z).as
    # 使用 SymPy 的 Add 类构造表达式，并以字典形式返回每个项的系数
    assert [Add(3*x, 2*x, y, 3, evaluate=False).as_coefficients_dict()[i]
            for i in check] == [3, 5, 1, 0, 3]

    # 使用 SymPy 的 Mul 类构造乘法表达式，并以字典形式返回每个项的系数
    assert [(3*x*y).as_coefficients_dict()[i] for i in check] == \
        [0, 0, 0, 3, 0]

    # 使用 SymPy 的 Mul 类构造带有浮点数的乘法表达式，并以字典形式返回每个项的系数
    assert [(3.0*x*y).as_coefficients_dict()[i] for i in check] == \
        [0, 0, 0, 3.0, 0]

    # 使用 SymPy 的 Mul 类构造带有浮点数的乘法表达式，并返回它的系数，结果应为 0
    assert (3.0*x*y).as_coefficients_dict()[3.0*x*y] == 0

    # 构造一个复杂的代数表达式 eq
    eq = x*(x + 1)*a + x*b + c/x
    
    # 返回表达式 eq 中关于 x 的各项的系数字典
    assert eq.as_coefficients_dict(x) == {x: b, 1/x: c, x*(x + 1): a}

    # 对表达式 eq 展开后，返回展开后各项关于 x 的系数字典
    assert eq.expand().as_coefficients_dict(x) == {x**2: a, x: a + b, 1/x: c}

    # 返回符号 x 自身的系数，应为 1
    assert x.as_coefficients_dict() == {x: S.One}
# 测试函数，用于检验符号代数操作的特定函数和方法
def test_args_cnc():
    # 创建一个非交换符号A
    A = symbols('A', commutative=False)
    # 测试符号表达式的参数和非交换部分
    assert (x + A).args_cnc() == \
        [[], [x + A]]  # 返回空列表和包含 x + A 的列表
    assert (x + a).args_cnc() == \
        [[a + x], []]  # 返回包含 a + x 的列表和空列表
    assert (x*a).args_cnc() == \
        [[a, x], []]  # 返回包含 a 和 x 的列表和空列表
    assert (x*y*A*(A + 1)).args_cnc(cset=True) == \
        [{x, y}, [A, 1 + A]]  # 返回包含{x, y}的集合和包含A和1 + A的列表
    assert Mul(x, x, evaluate=False).args_cnc(cset=True, warn=False) == \
        [{x}, []]  # 返回包含{x}的集合和空列表
    assert Mul(x, x**2, evaluate=False).args_cnc(cset=True, warn=False) == \
        [{x, x**2}, []]  # 返回包含{x, x**2}的集合和空列表
    raises(ValueError, lambda: Mul(x, x, evaluate=False).args_cnc(cset=True))
    assert Mul(x, y, x, evaluate=False).args_cnc() == \
        [[x, y, x], []]  # 返回包含{x, y, x}的列表和空列表
    # 始终将 -1 与前导数字分开
    assert (-1.*x).args_cnc() == [[-1, 1.0, x], []]  # 返回包含[-1, 1.0, x]的列表和空列表


# 测试函数，用于检验符号表达式的原始参数
def test_new_rawargs():
    n = Symbol('n', commutative=False)
    a = x + n
    assert a.is_commutative is False  # 检查表达式 a 是否为可交换
    assert a._new_rawargs(x).is_commutative  # 检查通过参数 x 生成的新表达式是否可交换
    assert a._new_rawargs(x, y).is_commutative  # 检查通过参数 x, y 生成的新表达式是否可交换
    assert a._new_rawargs(x, n).is_commutative is False  # 检查通过参数 x, n 生成的新表达式是否可交换
    assert a._new_rawargs(x, y, n).is_commutative is False  # 检查通过参数 x, y, n 生成的新表达式是否可交换
    m = x*n
    assert m.is_commutative is False  # 检查表达式 m 是否为可交换
    assert m._new_rawargs(x).is_commutative  # 检查通过参数 x 生成的新表达式是否可交换
    assert m._new_rawargs(n).is_commutative is False  # 检查通过参数 n 生成的新表达式是否可交换
    assert m._new_rawargs(x, y).is_commutative  # 检查通过参数 x, y 生成的新表达式是否可交换
    assert m._new_rawargs(x, n).is_commutative is False  # 检查通过参数 x, n 生成的新表达式是否可交换
    assert m._new_rawargs(x, y, n).is_commutative is False  # 检查通过参数 x, y, n 生成的新表达式是否可交换

    assert m._new_rawargs(x, n, reeval=False).is_commutative is False  # 检查通过参数 x, n 生成的新表达式是否可交换，且重新评估设置为 False
    assert m._new_rawargs(S.One) is S.One  # 检查通过参数 S.One 生成的新表达式是否为 S.One


# 测试函数，用于检验问题 #5226 的情况
def test_issue_5226():
    assert Add(evaluate=False) == 0  # 检查未评估的 Add 对象是否等于 0
    assert Mul(evaluate=False) == 1  # 检查未评估的 Mul 对象是否等于 1
    assert Mul(x + y, evaluate=False).is_Add  # 检查未评估的 Mul 对象是否为 Add 类型


# 测试函数，用于检验符号对象的自由符号
def test_free_symbols():
    # free_symbols 应返回对象的自由符号集合
    assert S.One.free_symbols == set()  # 检查常数 S.One 的自由符号集合是否为空集
    assert x.free_symbols == {x}  # 检查符号 x 的自由符号集合是否为 {x}
    assert Integral(x, (x, 1, y)).free_symbols == {y}  # 检查积分表达式的自由符号集合是否为 {y}
    assert (-Integral(x, (x, 1, y))).free_symbols == {y}  # 检查带负号的积分表达式的自由符号集合是否为 {y}
    assert meter.free_symbols == set()  # 检查常数 meter 的自由符号集合是否为空集
    assert (meter**x).free_symbols == {x}  # 检查幂运算表达式的自由符号集合是否为 {x}


# 测试函数，用于检验符号对象是否包含特定自由符号
def test_has_free():
    assert x.has_free(x)  # 检查符号 x 是否包含自由符号 x
    assert not x.has_free(y)  # 检查符号 x 是否包含自由符号 y
    assert (x + y).has_free(x)  # 检查表达式 x + y 是否包含自由符号 x
    assert (x + y).has_free(*(x, z))  # 检查表达式 x + y 是否包含自由符号 x 和 z
    assert f(x).has_free(x)  # 检查函数 f(x) 是否包含自由符号 x
    assert f(x).has_free(f(x))  # 检查函数 f(x) 是否包含自由符号 f(x)
    assert Integral(f(x), (f(x), 1, y)).has_free(y)  # 检查积分表达式是否包含自由符号 y
    assert not Integral(f(x), (f(x), 1, y)).has_free(x)  # 检查积分表达式是否不包含自由符号 x
    assert not Integral(f(x), (f(x), 1, y)).has_free(f(x))  # 检查积分表达式是否不包含自由符号 f(x)
    # 简单提取
    assert (x + 1 + y).has_free(x + 1)  # 检查表达式 x + 1 + y 是否包含子表达式 x + 1 的自由符号
    assert not (x + 2 + y).has_free(x + 1)  # 检查表达式 x + 2 + y 是否不包含子表达式 x + 1 的自由符号
    assert (2 + 3*x*y).has_free(3*x)  # 检查表达式 2 + 3*x*y 是否包含子表达式 3*x 的自由符号
    raises(TypeError, lambda: x.has_free({x, y}))  # 检查在类型错误时是否会引发异常
    s = FiniteSet(1, 2)
    assert Piecewise((s, x > 3), (4, True)).has_free(s)  # 检查分段函数表达式是否包含自由符号 s
    assert not Piecewise((1, x > 3), (4, True)).has_free(s)  # 检查分段函数表达式是否不包含自由符号 s
    # 不能形成这些的集合，但后备将处理
    raises(TypeError, lambda: x.has_free(y, []))  # 检查在类型错误时是否会引发异常


# 测试函数，用于检验表达式是否包含指
    # 调用 `raises` 函数来验证特定函数在给定参数下是否会引发指定的异常
    raises(TypeError, lambda: x.has_xfree([x]))
# 测试函数，用于验证符号计算库中的乘法运算
def test_issue_5300():
    # 创建一个符号变量 x，设置为非可交换
    x = Symbol('x', commutative=False)
    # 验证表达式 x*sqrt(2)/sqrt(6) 是否等于 x*sqrt(3)/3
    assert x*sqrt(2)/sqrt(6) == x*sqrt(3)/3


# 测试函数，用于验证符号计算库中的整数除法运算
def test_floordiv():
    # 导入整数取整函数
    from sympy.functions.elementary.integers import floor
    # 验证 x // y 是否等于 floor(x / y)
    assert x // y == floor(x / y)


# 测试函数，验证在不同类型数值和符号表达式中，as_coeff_Mul() 方法的行为
def test_as_coeff_Mul():
    # 验证整数 3 的 as_coeff_Mul() 方法返回 (3, 1)
    assert Integer(3).as_coeff_Mul() == (Integer(3), Integer(1))
    # 验证有理数 3/4 的 as_coeff_Mul() 方法返回 (3/4, 1)
    assert Rational(3, 4).as_coeff_Mul() == (Rational(3, 4), Integer(1))
    # 验证浮点数 5.0 的 as_coeff_Mul() 方法返回 (5.0, 1)
    assert Float(5.0).as_coeff_Mul() == (Float(5.0), Integer(1))
    # 验证浮点数 0.0 的 as_coeff_Mul() 方法返回 (0.0, 1)
    assert Float(0.0).as_coeff_Mul() == (Float(0.0), Integer(1))

    # 验证整数 3 乘以符号变量 x 的 as_coeff_Mul() 方法返回 (3, x)
    assert (Integer(3)*x).as_coeff_Mul() == (Integer(3), x)
    # 验证有理数 3/4 乘以符号变量 x 的 as_coeff_Mul() 方法返回 (3/4, x)
    assert (Rational(3, 4)*x).as_coeff_Mul() == (Rational(3, 4), x)
    # 验证浮点数 5.0 乘以符号变量 x 的 as_coeff_Mul() 方法返回 (5.0, x)
    assert (Float(5.0)*x).as_coeff_Mul() == (Float(5.0), x)

    # 验证整数 3 乘以符号变量 x 和 y 的乘积的 as_coeff_Mul() 方法返回 (3, x*y)
    assert (Integer(3)*x*y).as_coeff_Mul() == (Integer(3), x*y)
    # 验证有理数 3/4 乘以符号变量 x 和 y 的乘积的 as_coeff_Mul() 方法返回 (3/4, x*y)
    assert (Rational(3, 4)*x*y).as_coeff_Mul() == (Rational(3, 4), x*y)
    # 验证浮点数 5.0 乘以符号变量 x 和 y 的乘积的 as_coeff_Mul() 方法返回 (5.0, x*y)
    assert (Float(5.0)*x*y).as_coeff_Mul() == (Float(5.0), x*y)

    # 验证单个符号变量 x 的 as_coeff_Mul() 方法返回 (1, x)
    assert (x).as_coeff_Mul() == (S.One, x)
    # 验证符号变量 x 和 y 的乘积的 as_coeff_Mul() 方法返回 (1, x*y)
    assert (x*y).as_coeff_Mul() == (S.One, x*y)
    # 验证负无穷乘以符号变量 x 的有理数系数的 as_coeff_Mul() 方法返回 (-1, oo*x)
    assert (-oo*x).as_coeff_Mul(rational=True) == (-1, oo*x)


# 测试函数，验证在不同类型数值和符号表达式中，as_coeff_Add() 方法的行为
def test_as_coeff_Add():
    # 验证整数 3 的 as_coeff_Add() 方法返回 (3, 0)
    assert Integer(3).as_coeff_Add() == (Integer(3), Integer(0))
    # 验证有理数 3/4 的 as_coeff_Add() 方法返回 (3/4, 0)
    assert Rational(3, 4).as_coeff_Add() == (Rational(3, 4), Integer(0))
    # 验证浮点数 5.0 的 as_coeff_Add() 方法返回 (5.0, 0)
    assert Float(5.0).as_coeff_Add() == (Float(5.0), Integer(0))

    # 验证整数 3 加上符号变量 x 的 as_coeff_Add() 方法返回 (3, x)
    assert (Integer(3) + x).as_coeff_Add() == (Integer(3), x)
    # 验证有理数 3/4 加上符号变量 x 的 as_coeff_Add() 方法返回 (3/4, x)
    assert (Rational(3, 4) + x).as_coeff_Add() == (Rational(3, 4), x)
    # 验证浮点数 5.0 加上符号变量 x 的 as_coeff_Add() 方法返回 (5.0, x)
    assert (Float(5.0) + x).as_coeff_Add() == (Float(5.0), x)
    # 验证有理数 5.0 加上符号变量 x 的 as_coeff_Add() 方法返回 (0, 5.0 + x)
    assert (Float(5.0) + x).as_coeff_Add(rational=True) == (0, Float(5.0) + x)

    # 验证整数 3 加上符号变量 x 和 y 的和的 as_coeff_Add() 方法返回 (3, x + y)
    assert (Integer(3) + x + y).as_coeff_Add() == (Integer(3), x + y)
    # 验证有理数 3/4 加上符号变量 x 和 y 的和的 as_coeff_Add() 方法返回 (3/4, x + y)
    assert (Rational(3, 4) + x + y).as_coeff_Add() == (Rational(3, 4), x + y)
    # 验证浮点数 5.0 加上符号变量 x 和 y 的和的 as_coeff_Add() 方法返回 (5.0, x + y)
    assert (Float(5.0) + x + y).as_coeff_Add() == (Float(5.0), x + y)

    # 验证单个符号变量 x 的 as_coeff_Add() 方法返回 (0, x)
    assert (x).as_coeff_Add() == (S.Zero, x)
    # 验证符号变量 x 和 y 的乘积的 as_coeff_Add() 方法返回 (0, x*y)
    assert (x*y).as_coeff_Add() == (S.Zero, x*y)


# 测试函数，验证符号表达式排序函数 default_sort_key 的行为
def test_expr_sorting():
    # 定义一系列表达式列表
    exprs = [1/x**2, 1/x, sqrt(sqrt(x)), sqrt(x), x, sqrt(x)**3, x**2]
    # 验证按照 default_sort_key 排序后的表达式列表与原列表相同
    assert sorted(exprs, key=default_sort_key) == exprs

    exprs = [x, 2*x, 2*x**2, 2*x**3, x**n, 2*x**n, sin(x), sin(x)**n,
             sin(x**2), cos(x), cos(x**2), tan(x)]
    # 验证按照 default_sort_key 排序后的表达式列表与原列表相同
    assert sorted(exprs, key=default_sort_key) == exprs

    exprs = [x + 1, x**2 + x + 1, x**3 + x**2 + x + 1]
    # 验证按照 default_sort_key 排序后的表达式列表与原列表相同
    assert sorted(exprs, key=default_sort_key) == exprs

    exprs = [S(4), x - 3*I/2, x + 3*I/2, x - 4*I + 1, x + 4*I + 1]
    # 验证按照 default_sort_key 排序后的表达式列表与原列表相同
    # 定义一个包含两个子列表的列表，用于测试排序函数
    exprs = [[1, 2], [1, 2, 3]]
    # 断言排序后的结果与原始列表相同，使用自定义的排序键函数 default_sort_key
    assert sorted(exprs, key=default_sort_key) == exprs
    
    # 定义一个包含两个字典的列表，用于测试排序函数
    exprs = [{x: -y}, {x: y}]
    # 断言排序后的结果与原始列表相同，使用自定义的排序键函数 default_sort_key
    assert sorted(exprs, key=default_sort_key) == exprs
    
    # 定义一个包含两个集合的列表，用于测试排序函数
    exprs = [{1}, {1, 2}]
    # 断言排序后的结果与原始列表相同，使用自定义的排序键函数 default_sort_key
    assert sorted(exprs, key=default_sort_key) == exprs
    
    # 将列表表达式 exprs 分解为两个变量 a 和 b，并同时赋值
    a, b = exprs = [Dummy('x'), Dummy('x')]
    # 断言排序后的结果与原始列表相同，使用自定义的排序键函数 default_sort_key
    assert sorted([b, a], key=default_sort_key) == exprs
# 定义函数 test_as_ordered_factors，用于测试 as_ordered_factors 方法
def test_as_ordered_factors():

    # 断言调用 x 对象的 as_ordered_factors 方法返回 [x]
    assert x.as_ordered_factors() == [x]

    # 断言调用复合表达式 2*x*x**n*sin(x)*cos(x) 的 as_ordered_factors 方法返回 [Integer(2), x, x**n, sin(x), cos(x)]
    assert (2*x*x**n*sin(x)*cos(x)).as_ordered_factors() \
        == [Integer(2), x, x**n, sin(x), cos(x)]

    # 准备参数列表 args
    args = [f(1), f(2), f(3), f(1, 2, 3), g(1), g(2), g(3), g(1, 2, 3)]
    # 构建乘积表达式 expr
    expr = Mul(*args)

    # 断言调用 expr 的 as_ordered_factors 方法返回 args 列表
    assert expr.as_ordered_factors() == args

    # 定义符号 A, B，其中 B 为非交换符号
    A, B = symbols('A,B', commutative=False)

    # 断言调用 A*B 的 as_ordered_factors 方法返回 [A, B]
    assert (A*B).as_ordered_factors() == [A, B]
    # 断言调用 B*A 的 as_ordered_factors 方法返回 [B, A]
    assert (B*A).as_ordered_factors() == [B, A]


# 定义函数 test_as_ordered_terms，用于测试 as_ordered_terms 方法
def test_as_ordered_terms():

    # 断言调用 x 对象的 as_ordered_terms 方法返回 [x]
    assert x.as_ordered_terms() == [x]

    # 断言调用复合表达式 sin(x)**2*cos(x) + sin(x)*cos(x)**2 + 1 的 as_ordered_terms 方法返回 [sin(x)**2*cos(x), sin(x)*cos(x)**2, 1]
    assert (sin(x)**2*cos(x) + sin(x)*cos(x)**2 + 1).as_ordered_terms() \
        == [sin(x)**2*cos(x), sin(x)*cos(x)**2, 1]

    # 准备参数列表 args
    args = [f(1), f(2), f(3), f(1, 2, 3), g(1), g(2), g(3), g(1, 2, 3)]
    # 构建加法表达式 expr
    expr = Add(*args)

    # 断言调用 expr 的 as_ordered_terms 方法返回 args 列表
    assert expr.as_ordered_terms() == args

    # 断言调用表达式 1 + 4*sqrt(3)*pi*x 的 as_ordered_terms 方法返回 [4*pi*x*sqrt(3), 1]
    assert (1 + 4*sqrt(3)*pi*x).as_ordered_terms() == [4*pi*x*sqrt(3), 1]

    # 各种复数形式的排序断言
    assert ( 2 + 3*I).as_ordered_terms() == [2, 3*I]
    assert (-2 + 3*I).as_ordered_terms() == [-2, 3*I]
    assert ( 2 - 3*I).as_ordered_terms() == [2, -3*I]
    assert (-2 - 3*I).as_ordered_terms() == [-2, -3*I]
    assert ( 4 + 3*I).as_ordered_terms() == [4, 3*I]
    assert (-4 + 3*I).as_ordered_terms() == [-4, 3*I]
    assert ( 4 - 3*I).as_ordered_terms() == [4, -3*I]
    assert (-4 - 3*I).as_ordered_terms() == [-4, -3*I]

    # 定义表达式 e
    e = x**2*y**2 + x*y**4 + y + 2

    # 不同排序方式的断言
    assert e.as_ordered_terms(order="lex") == [x**2*y**2, x*y**4, y, 2]
    assert e.as_ordered_terms(order="grlex") == [x*y**4, x**2*y**2, y, 2]
    assert e.as_ordered_terms(order="rev-lex") == [2, y, x*y**4, x**2*y**2]
    assert e.as_ordered_terms(order="rev-grlex") == [2, y, x**2*y**2, x*y**4]

    # 定义符号 k
    k = symbols('k')
    # 断言调用 k 的 as_ordered_terms 方法返回 ([(k, ((1.0, 0.0), (1,), ()))], [k])
    assert k.as_ordered_terms(data=True) == ([(k, ((1.0, 0.0), (1,), ()))], [k])


# 定义函数 test_sort_key_atomic_expr，用于测试 sort_key 方法
def test_sort_key_atomic_expr():
    # 导入 m, s 单位
    from sympy.physics.units import m, s
    # 断言对 [-m, s] 列表使用 sort_key 方法排序后返回 [-m, s]
    assert sorted([-m, s], key=lambda arg: arg.sort_key()) == [-m, s]


# 定义函数 test_eval_interval，用于测试 _eval_interval 方法
def test_eval_interval():
    # 断言对 exp(x) 对象调用 _eval_interval 方法返回 exp(1) - exp(0)
    assert exp(x)._eval_interval(*Tuple(x, 0, 1)) == exp(1) - exp(0)

    # issue 4199 的测试
    a = x/y
    raises(NotImplementedError, lambda: a._eval_interval(x, S.Zero, oo)._eval_interval(y, oo, S.Zero))
    raises(NotImplementedError, lambda: a._eval_interval(x, S.Zero, oo)._eval_interval(y, S.Zero, oo))
    a = x - y
    raises(NotImplementedError, lambda: a._eval_interval(x, S.One, oo)._eval_interval(y, oo, S.One))
    raises(ValueError, lambda: x._eval_interval(x, None, None))
    a = -y*Heaviside(x - y)
    assert a._eval_interval(x, -oo, oo) == -y
    assert a._eval_interval(x, oo, -oo) == y


# 定义函数 test_eval_interval_zoo，用于测试 _eval_interval 方法处理无穷值
def test_eval_interval_zoo():
    # 测试当 _eval_interval 方法返回无穷大时的情况
    assert Si(1/x)._eval_interval(x, S.Zero, S.One) == -pi/2 + Si(1)


# 定义函数 test_primitive，用于测试 primitive 方法
def test_primitive():
    # 断言调用 3*(x + 1)**2 的 primitive 方法返回 (3, (x + 1)**2)
    assert (3*(x + 1)**2).primitive() == (3, (x + 1)**2)
    # 断言调用 6*x + 2 的 primitive 方法返回 (2, 3*x + 1)
    assert (6*x + 2).primitive() == (2, 3*x + 1)
    # 断言调用 x/2 + 3 的 primitive 方法返回 (S.Half, x + 6)
    assert (x/2 + 3).primitive() == (S.Half, x + 6)
    # 构建乘积表达式 eq
    eq = (6*x + 2)*(x/2 + 3)
    # 断言调用 eq 的 primitive 方法返回 (1, eq)
    assert eq.primitive()[0] == 1
    # 构建幂表达式 eq
    eq = (2 + 2*x)**2
    # 断言调用 eq 的 primitive 方法返回 1
    assert eq.primitive()[0] == 1
    # 断言：验证表达式 (4.0*x).primitive() 的结果是否为 (1, 4.0*x)
    assert (4.0*x).primitive() == (1, 4.0*x)
    
    # 断言：验证表达式 (4.0*x + y/2).primitive() 的结果是否为 (S.Half, 8.0*x + y)
    assert (4.0*x + y/2).primitive() == (S.Half, 8.0*x + y)
    
    # 断言：验证表达式 (-2*x).primitive() 的结果是否为 (2, -x)
    assert (-2*x).primitive() == (2, -x)
    
    # 断言：验证表达式 Add(5*z/7, 0.5*x, 3*y/2, evaluate=False).primitive() 的结果是否为 (S.One/14, 7.0*x + 21*y + 10*z)
    assert Add(5*z/7, 0.5*x, 3*y/2, evaluate=False).primitive() == \
        (S.One/14, 7.0*x + 21*y + 10*z)
    
    # 遍历：对于列表中的每个元素 i，分别进行以下断言
    for i in [S.Infinity, S.NegativeInfinity, S.ComplexInfinity]:
        # 断言：验证表达式 (i + x/3).primitive() 的结果是否为 (S.One/3, i + x)
        assert (i + x/3).primitive() == \
            (S.One/3, i + x)
    
    # 断言：验证表达式 (S.Infinity + 2*x/3 + 4*y/7).primitive() 的结果是否为 (S.One/21, 14*x + 12*y + oo)
    assert (S.Infinity + 2*x/3 + 4*y/7).primitive() == \
        (S.One/21, 14*x + 12*y + oo)
    
    # 断言：验证表达式 S.Zero.primitive() 的结果是否为 (S.One, S.Zero)
    assert S.Zero.primitive() == (S.One, S.Zero)
# 定义测试函数 `test_issue_5843`，用于检查问题 5843 的相关功能
def test_issue_5843():
    # 尝试执行一个未定义的变量 x，会导致 NameError
    a = 1 + x
    # 使用 assert 断言，验证表达式的正确性，这里是测试某个方法的返回值
    assert (2*a).extract_multiplicatively(a) == 2
    assert (4*a).extract_multiplicatively(2*a) == 2
    assert ((3*a)*(2*a)).extract_multiplicatively(a) == 6*a


# 定义测试函数 `test_is_constant`，用于检查 is_constant 方法的行为
def test_is_constant():
    # 导入 sympy.solvers.solvers 模块中的 checksol 函数
    from sympy.solvers.solvers import checksol
    # 使用 assert 断言，验证 Sum 对象是否为常数，期望结果为 True
    assert Sum(x, (x, 1, 10)).is_constant() is True
    # 使用 assert 断言，验证 Sum 对象是否为常数，期望结果为 False
    assert Sum(x, (x, 1, n)).is_constant() is False
    # 使用 assert 断言，验证 Sum 对象是否对指定变量为常数，期望结果为 True
    assert Sum(x, (x, 1, n)).is_constant(y) is True
    # 使用 assert 断言，验证 Sum 对象是否对指定变量为常数，期望结果为 False
    assert Sum(x, (x, 1, n)).is_constant(n) is False
    # 使用 assert 断言，验证 Sum 对象是否对指定变量为常数，期望结果为 True
    assert Sum(x, (x, 1, n)).is_constant(x) is True
    # 创建一个等式对象 eq
    eq = a*cos(x)**2 + a*sin(x)**2 - a
    # 使用 assert 断言，验证等式对象是否为常数，期望结果为 True
    assert eq.is_constant() is True
    # 使用 assert 断言，验证在给定变量取特定值时等式成立
    assert eq.subs({x: pi, a: 2}) == eq.subs({x: pi, a: 3}) == 0
    # 使用 assert 断言，验证符号 x 是否为常数，期望结果为 False
    assert x.is_constant() is False
    # 使用 assert 断言，验证符号 x 对指定变量是否为常数，期望结果为 True
    assert x.is_constant(y) is True
    # 使用 assert 断言，验证 log(x/y) 是否为常数，期望结果为 False
    assert log(x/y).is_constant() is False

    # 使用 assert 断言，验证 checksol 函数的返回值，期望结果为 False
    assert checksol(x, x, Sum(x, (x, 1, n))) is False
    # 使用 assert 断言，验证 checksol 函数的返回值，期望结果为 False
    assert checksol(x, x, Sum(x, (x, 1, n))) is False
    # 使用 assert 断言，验证 f(1) 的 is_constant 属性
    assert f(1).is_constant
    # 使用 assert 断言，验证 checksol 函数的返回值，期望结果为 False
    assert checksol(x, x, f(x)) is False

    # 使用 assert 断言，验证 Pow 对象是否为常数，期望结果为 True，这里是幂运算的特例
    assert Pow(x, S.Zero, evaluate=False).is_constant() is True  # == 1
    # 使用 assert 断言，验证 Pow 对象是否为常数，期望结果为 False，这里是幂运算的特例
    assert Pow(S.Zero, x, evaluate=False).is_constant() is False  # == 0 or 1
    # 使用 assert 断言，验证 (2**x) 是否为常数，期望结果为 False
    assert (2**x).is_constant() is False
    # 使用 assert 断言，验证 Pow 对象是否为常数，期望结果为 True
    assert Pow(S(2), S(3), evaluate=False).is_constant() is True

    # 创建两个带有 zero=True 属性的符号对象
    z1, z2 = symbols('z1 z2', zero=True)
    # 使用 assert 断言，验证表达式是否为常数，期望结果为 True
    assert (z1 + 2*z2).is_constant() is True

    # 使用 assert 断言，验证 meter 对象是否为常数，期望结果为 True
    assert meter.is_constant() is True
    # 使用 assert 断言，验证 (3*meter) 是否为常数，期望结果为 True
    assert (3*meter).is_constant() is True
    # 使用 assert 断言，验证 (x*meter) 是否为常数，期望结果为 False
    assert (x*meter).is_constant() is False


# 定义测试函数 `test_equals`，用于检查 equals 方法的行为
def test_equals():
    # 使用 assert 断言，验证一个复杂表达式是否等于 0
    assert (-3 - sqrt(5) + (-sqrt(10)/2 - sqrt(2)/2)**2).equals(0)
    # 使用 assert 断言，验证两个表达式是否相等
    assert (x**2 - 1).equals((x + 1)*(x - 1))
    # 使用 assert 断言，验证三角函数平方和等于 1
    assert (cos(x)**2 + sin(x)**2).equals(1)
    # 使用 assert 断言，验证带参数的三角函数平方和等于 a
    assert (a*cos(x)**2 + a*sin(x)**2).equals(a)
    # 定义一个数学表达式 r
    r = sqrt(2)
    # 使用 assert 断言，验证一个复杂的数学表达式是否等于 0
    assert (-1/(r + r*x) + 1/r/(1 + x)).equals(0)
    # 使用 assert 断言，验证阶乘函数的等式
    assert factorial(x + 1).equals((x + 1)*factorial(x))
    # 使用 assert 断言，验证两个数学表达式是否不相等
    assert sqrt(3).equals(2*sqrt(3)) is False
    # 使用 assert 断言，验证两个数学表达式是否不相等
    assert (sqrt(5)*sqrt(3)).equals(sqrt(3)) is False
    # 使用 assert 断言，验证两个数学表达式是否不相等
    assert (sqrt(5) + sqrt(3)).equals(0) is False
    # 使用 assert 断言，验证两个数学表达式是否不相等
    assert (sqrt(5) + pi).equals(0) is False
    # 使用 assert 断言，验证 meter 对象是否等于 0
    assert meter.equals(0) is False
    # 使用 assert 断言，验证 (3*meter**2) 是否等于 0
    assert (3*meter**2).equals(0) is False
    # 创建一个复杂的等式对象 eq，用于测试 equals 方法
    eq = -(-1)**(S(3)/4)*6**(S.One/4) + (-6)**(S.One/4)*I
    # 如果等式对象不等于 0，则执行下一条断言
    if eq != 0:
        # 使用 assert 断言，验证复杂等式是否等于 0
        assert eq.equals(0)
    # 使用 assert 断言，验证 sqrt(x) 是否等于 0
    assert sqrt(x).equals(0) is False

    # 定义一个复杂的积分表达式 i 和其预期的解析结果 ans
    i = 2*sqrt(2)*x**(S(5)/2)*(1 + 1/(2*x))**(S(5)/2)/5 + \
        2*sqrt(2)*x**(S(3)/2)*(1 + 1/(2*x))**(S(5)/2)/(-6 - 3/x)
    ans = sqrt(2*x + 1)*(6*x**2 + x - 1)/15
    # 计算两者的差异
    diff = i - ans
    # 使用 assert 断言，验证差异是否等于 0，期望结果为 None，但实际上应该是 False
    assert
    # 通过最小多项式或自洽性证明
    eq = sqrt(1 + sqrt(3)) + sqrt(3 + 3*sqrt(3)) - sqrt(10 + 6*sqrt(3))
    # 断言eq应该等于0
    assert eq.equals(0)
    # 计算q的值
    q = 3**Rational(1, 3) + 3
    # 计算p的值
    p = expand(q**3)**Rational(1, 3)
    # 断言p与q相等
    assert (p - q).equals(0)

    # 问题 6829
    # 定义符号q
    q = symbols('q')
    # 复杂表达式z，用q表示
    z = (q*(-sqrt(-2*(-(q - S(7)/8)**S(2)/8 - S(2197)/13824)**(S.One/3) -
    S(13)/12)/2 - sqrt((2*q - S(7)/4)/sqrt(-2*(-(q - S(7)/8)**S(2)/8 -
    S(2197)/13824)**(S.One/3) - S(13)/12) + 2*(-(q - S(7)/8)**S(2)/8 -
    S(2197)/13824)**(S.One/3) - S(13)/6)/2 - S.One/4) + q/4 + (-sqrt(-2*(-(q
    - S(7)/8)**S(2)/8 - S(2197)/13824)**(S.One/3) - S(13)/12)/2 - sqrt((2*q
    - S(7)/4)/sqrt(-2*(-(q - S(7)/8)**S(2)/8 - S(2197)/13824)**(S.One/3) -
    S(13)/12) + 2*(-(q - S(7)/8)**S(2)/8 - S(2197)/13824)**(S.One/3) -
    S(13)/6)/2 - S.One/4)**4 + (-sqrt(-2*(-(q - S(7)/8)**S(2)/8 -
    S(2197)/13824)**(S.One/3) - S(13)/12)/2 - sqrt((2*q -
    S(7)/4)/sqrt(-2*(-(q - S(7)/8)**S(2)/8 - S(2197)/13824)**(S.One/3) -
    S(13)/12) + 2*(-(q - S(7)/8)**S(2)/8 - S(2197)/13824)**(S.One/3) -
    S(13)/6)/2 - S.One/4)**3 + 2*(-sqrt(-2*(-(q - S(7)/8)**S(2)/8 -
    S(2197)/13824)**(S.One/3) - S(13)/12)/2 - sqrt((2*q -
    S(7)/4)/sqrt(-2*(-(q - S(7)/8)**S(2)/8 - S(2197)/13824)**(S.One/3) -
    S(13)/12) + 2*(-(q - S(7)/8)**S(2)/8 - S(2197)/13824)**(S.One/3) -
    S(13)/6)/2 - S.One/4)**2 - Rational(1, 3))
    # 断言z应该等于0
    assert z.equals(0)


这段代码涉及到复杂的数学表达式和符号操作，通过注释详细解释了每个表达式的作用和预期的断言结果。
def test_round():
    # 测试浮点数四舍五入到指定小数位数
    assert str(Float('0.1249999').round(2)) == '0.12'
    # 创建一个大整数
    d20 = 12345678901234567890
    # 测试整数四舍五入到指定位数，结果应当仍为整数且与原数相等
    ans = S(d20).round(2)
    assert ans.is_Integer and ans == d20
    # 测试整数四舍五入到负数位数
    ans = S(d20).round(-2)
    assert ans.is_Integer and ans == 12345678901234567900
    # 测试有理数四舍五入到指定小数位数
    assert str(S('1/7').round(4)) == '0.1429'
    # 测试无限循环小数四舍五入
    assert str(S('.[12345]').round(4)) == '0.1235'
    # 测试普通小数四舍五入
    assert str(S('.1349').round(2)) == '0.13'
    # 测试整数四舍五入到默认位数，结果应当与原数相等
    n = S(12345)
    ans = n.round()
    assert ans.is_Integer
    assert ans == n
    # 测试整数四舍五入到指定位数，结果应当与原数相等
    ans = n.round(1)
    assert ans.is_Integer
    assert ans == n
    # 测试整数四舍五入到指定位数，结果应当与原数相等
    ans = n.round(4)
    assert ans.is_Integer
    assert ans == n
    # 测试整数四舍五入到负数位数
    assert n.round(-1) == 12340

    # 测试将数值转为浮点数并四舍五入到指定位数
    r = Float(str(n)).round(-4)
    assert r == 10000.0

    # 测试整数四舍五入到更小的负数位数
    assert n.round(-5) == 0

    # 测试将数学常数加上平方根并四舍五入到指定位数
    assert str((pi + sqrt(2)).round(2)) == '4.56'
    # 测试将数学常数加上乘以常数后四舍五入到指定位数
    assert (10*(pi + sqrt(2))).round(-1) == 50.0
    # 测试在SymPy中调用round函数应抛出TypeError异常
    raises(TypeError, lambda: round(x + 2, 2))
    # 测试将浮点数四舍五入到指定位数
    assert str(S(2.3).round(1)) == '2.3'
    
    # 测试当在给定精度下，遇到以5结尾的数字，四舍五入的结果应该使最后一位偶数
    for i in range(-99, 100):
        s = str(abs(i))
        p = len(s)
        n = '0.%s5' % s
        j = p + 2
        if i < 0:
            j += 1
            n = '-' + n
        v = str(Float(n).round(p))[:j]
        if v.endswith('.'):
            continue
        L = int(v[-1])
        assert L % 2 == 0, (n, '->', v)

    # 测试浮点数四舍五入到整数
    assert (Float(.3, 3) + 2*pi).round() == 7
    # 测试浮点数乘以常数后四舍五入到整数
    assert (Float(.3, 3) + 2*pi*100).round() == 629
    # 测试复数四舍五入
    assert (pi + 2*E*I).round() == 3 + 5*I
    # 测试浮点数四舍五入到超过已知精度的位数
    assert str((Float(.03, 3) + 2*pi/100).round(5)) == '0.0928'
    assert str((Float(.03, 3) + 2*pi/100).round(4)) == '0.0928'

    # 测试零的四舍五入
    assert S.Zero.round() == 0

    # 测试超过精度要求的四舍五入
    a = (Add(1, Float('1.' + '9'*27, ''), evaluate=0))
    assert a.round(10) == Float('3.000000000000000000000000000', '')
    assert a.round(25) == Float('3.000000000000000000000000000', '')
    assert a.round(26) == Float('3.000000000000000000000000000', '')
    assert a.round(27) == Float('2.999999999999999999999999999', '')
    assert a.round(30) == Float('2.999999999999999999999999999', '')

    # 下面是注释掉的代码块，不包含在此处的注释中
    # assert a.round(10) == Float('3.0000000000', '')
    # assert a.round(25) == Float('3.0000000000000000000000000', '')
    # assert a.round(26) == Float('3.00000000000000000000000000', '')
    # XXX: round方法是否应该设置结果的精度？
    #      上述测试的旧版本是这样的，但它们仅通过了
    #      因为具有不同精度的浮点数比较相等：
    #
    # assert a.round(10) == Float('3.0000000000', '')
    # assert a.round(25) == Float('3.0000000000000000000000000', '')
    # assert a.round(26) == Float('3.00000000000000000000000000', '')
    # assert a.round(27) == Float('2.999999999999999999999999999', '')
    # assert a.round(30) == Float('2.999999999999999999999999999', '')

    # TypeError应该被抛出，因为round()方法要求参数
    raises(TypeError, lambda: x.round())
    raises(TypeError, lambda: f(1).round())

    # 对于精确的数量级为10
    assert str(S.One.round()) == '1'
    assert str(S(100).round()) == '100'

    # 应用于实部和虚部
    assert (2*pi + E*I).round() == 6 + 3*I
    assert (2*pi + I/10).round() == 6
    assert (pi/10 + 2*I).round() == 2*I
    # 左侧的实部和虚部是具有2位小数精度的Float
    # 右侧的实部和虚部具有15位小数精度，因此它们不会相等，
    # 除非我们使用字符串或比较各个部分（这将强制浮点数转换为相同的精度）或重新创建浮点数
    assert str((pi/10 + E*I).round(2)) == '0.31 + 2.72*I'
    assert str((pi/10 + E*I).round(2).as_real_imag()) == '(0.31, 2.72)'
    assert str((pi/10 + E*I).round(2)) == '0.31 + 2.72*I'

    # issue 6914
    assert (I**(I + 3)).round(3) == Float('-0.208', '')*I

    # issue 8720
    assert S(-123.6).round() == -124
    assert S(-1.5).round() == -2
    assert S(-100.5).round() == -100
    assert S(-1.5 - 10.5*I).round() == -2 - 10*I

    # issue 7961
    assert str(S(0.006).round(2)) == '0.01'
    assert str(S(0.00106).round(4)) == '0.0011'

    # issue 8147
    assert S.NaN.round() is S.NaN
    assert S.Infinity.round() is S.Infinity
    assert S.NegativeInfinity.round() is S.NegativeInfinity
    assert S.ComplexInfinity.round() is S.ComplexInfinity
    # 循环两次，i 分别为 0 和 1
    for i in range(2):
        # 将 i 转换为浮点数
        fi = float(i)

        # 断言以下条件对所有指定精度 p (-1, 0, 1) 都成立：round(i, p) 返回整数
        assert all(type(round(i, p)) is int for p in (-1, 0, 1))
        # 断言以下条件对所有指定精度 p (-1, 0, 1) 都成立：S(i).round(p) 返回整数
        assert all(S(i).round(p).is_Integer for p in (-1, 0, 1))
        # 断言以下条件对所有指定精度 p (-1, 0, 1) 都成立：round(fi, p) 返回浮点数
        assert all(type(round(fi, p)) is float for p in (-1, 0, 1))
        # 断言以下条件对所有指定精度 p (-1, 0, 1) 都成立：S(fi).round(p) 返回浮点数
        assert all(S(fi).round(p).is_Float for p in (-1, 0, 1))

        # 断言不同精度下 round(i) 返回整数
        assert type(round(i)) is int
        # 断言 S(i).round() 返回整数
        assert S(i).round().is_Integer
        # 断言不同精度下 round(fi) 返回整数
        assert type(round(fi)) is int
        # 断言 S(fi).round() 返回整数
        assert S(fi).round().is_Integer

        # issue 25698 的问题验证
        n = 6000002
        # 断言计算结果符合预期
        assert int(n*(log(n) + log(log(n)))) == 110130079
        # 计算 cos(2)^2 + sin(2)^2，应为 1
        one = cos(2)**2 + sin(2)**2
        # 计算 exp(i * pi) 的值
        eq = exp(one*I*pi)
        # 获取复数的实部和虚部
        qr, qi = eq.as_real_imag()
        # 断言虚部舍入到两位小数为 0.0
        assert qi.round(2) == 0.0
        # 断言整个表达式舍入到两位小数为 -1.0
        assert eq.round(2) == -1.0
        # 计算 one - 1/(10**120)，eq 表示这个值
        eq = one - 1/S(10**120)
        # 断言以下表达式中 S.true 为假：eq > 1 或 eq < 1
        assert S.true not in (eq > 1, eq < 1)
        # 断言 int(eq) 等于 int(.9) 等于 0
        assert int(eq) == int(.9) == 0
        # 断言 int(-eq) 等于 int(-.9) 等于 0
        assert int(-eq) == int(-.9) == 0
def test_held_expression_UnevaluatedExpr():
    # 创建符号变量 x
    x = symbols("x")
    # 创建一个 UnevaluatedExpr 对象，表示 1/x
    he = UnevaluatedExpr(1/x)
    # 创建一个 Mul 对象，表示 x * he
    e1 = x * he

    # 断言 e1 是 Mul 类型的对象
    assert isinstance(e1, Mul)
    # 断言 e1 的参数是 (x, he)
    assert e1.args == (x, he)
    # 断言 e1 的求值结果是 1
    assert e1.doit() == 1
    # 断言 UnevaluatedExpr(Derivative(x, x)) 的非深度求值结果仍然是 Derivative(x, x)
    assert UnevaluatedExpr(Derivative(x, x)).doit(deep=False) == Derivative(x, x)
    # 断言 UnevaluatedExpr(Derivative(x, x)) 的深度求值结果是 1
    assert UnevaluatedExpr(Derivative(x, x)).doit() == 1

    # 创建一个 Mul 对象，表示 x * x，但不进行求值
    xx = Mul(x, x, evaluate=False)
    # 断言 xx 不等于 x**2
    assert xx != x**2

    # 创建一个 UnevaluatedExpr 对象，表示 xx
    ue2 = UnevaluatedExpr(xx)
    # 断言 ue2 是 UnevaluatedExpr 类型的对象
    assert isinstance(ue2, UnevaluatedExpr)
    # 断言 ue2 的参数是 (xx,)
    assert ue2.args == (xx,)
    # 断言 ue2 的深度求值结果是 x**2
    assert ue2.doit() == x**2
    # 断言 ue2 的非深度求值结果是 xx
    assert ue2.doit(deep=False) == xx

    # 创建一个 Mul 对象，表示 2 * UnevaluatedExpr(2)
    x2 = UnevaluatedExpr(2) * 2
    # 断言 x2 是 Mul 类型的对象
    assert type(x2) is Mul
    # 断言 x2 的参数是 (2, UnevaluatedExpr(2))
    assert x2.args == (2, UnevaluatedExpr(2))

def test_round_exception_nostr():
    # 在 round 异常处理中，不使用表达式的字符串形式，因为它太慢了
    s = Symbol('bad')
    try:
        s.round()
    except TypeError as e:
        # 断言异常消息中不包含 'bad'
        assert 'bad' not in str(e)
    else:
        # 如果没有引发异常，抛出 AssertionError
        raise AssertionError("Did not raise")

def test_extract_branch_factor():
    # 断言 exp_polar(2.0*I*pi) 的分支因子提取结果为 (1, 1)
    assert exp_polar(2.0*I*pi).extract_branch_factor() == (1, 1)

def test_identity_removal():
    # 断言 Add.make_args(x + 0) 的结果是 (x,)
    assert Add.make_args(x + 0) == (x,)
    # 断言 Mul.make_args(x * 1) 的结果是 (x,)
    assert Mul.make_args(x * 1) == (x,)

def test_float_0():
    # 断言 Float(0.0) + 1 的结果是 Float(1.0)
    assert Float(0.0) + 1 == Float(1.0)

@XFAIL
def test_float_0_fail():
    # 断言 Float(0.0) * x 的结果是 Float(0.0)（预期失败）
    assert Float(0.0) * x == Float(0.0)
    # 断言 (x + Float(0.0)).is_Add 是 True（预期失败）
    assert (x + Float(0.0)).is_Add

def test_issue_6325():
    # 计算 ans 和 e 的差异
    ans = (b**2 + z**2 - (b*(a + b*t) + z*(c + t*z))**2 / ((a + b*t)**2 + (c + t*z)**2)) / sqrt((a + b*t)**2 + (c + t*z)**2)
    e = sqrt((a + b*t)**2 + (c + z*t)**2)
    # 断言 e 对 t 的二阶导数等于 ans
    assert diff(e, t, 2) == ans
    # 断言 e 对 t 的二阶导数等于 ans
    assert e.diff(t, 2) == ans
    # 断言 e 对 t 的二阶导数在不简化情况下不等于 ans
    assert diff(e, t, 2, simplify=False) != ans

def test_issue_7426():
    # 创建两个 Mod 对象 f1 和 f2
    f1 = a % c
    f2 = x % z
    # 断言 f1.equals(f2) 的结果是 None
    assert f1.equals(f2) is None

def test_issue_11122():
    # 创建一个符号变量 x，其值非负
    x = Symbol('x', extended_positive=False)
    # 断言 unchanged(Gt, x, 0) 结果为 True（表示 x > 0）
    assert unchanged(Gt, x, 0)
    # 断言 (x > 0) 结果为 S.false
    assert (x > 0) is S.false

    # 创建一个符号变量 x，其值非正且为实数
    x = Symbol('x', positive=False, real=True)
    # 断言 (x > 0) 结果为 S.false
    assert (x > 0) is S.false

def test_issue_10651():
    # 创建一个符号变量 x，其为实数
    x = Symbol('x', real=True)
    # 创建几个表达式对象 e1, e3, e4, e5
    e1 = (-1 + x) / (1 - x)
    e3 = (4 * x**2 - 4) / ((1 - x) * (1 + x))
    e4 = 1 / (cos(x)**2) - (tan(x))**2
    # 创建一个符号变量 x，其值为正
    x = Symbol('x', positive=True)
    e5 = (1 + x) / x
    # 断言 e1 不是常数
    assert e1.is_constant() is None
    # 断言 e3 不是常数
    assert e3.is_constant() is None
    # 断言 e4 不是常数
    assert e4.is_constant() is None
    # 断言 e5 不是常数
    assert e5.is_constant() is False

def test_issue_10161():
    # 创建一个符号变量 x，其为实数
    x = symbols('x', real=True)
    # 断言 x * abs(x) * abs(x) 等于 x**3
    assert x * abs(x) * abs(x) == x**3

def test_issue_10755():
    # 创建一个符号变量 x
    x = symbols('x')
    # 断言对 log(x) 取整会引发 TypeError
    raises(TypeError, lambda: int(log(x)))
    # 断言对 log(x) 进行四舍五入保留两位小数会引发 TypeError
    raises(TypeError, lambda: log(x).round(2))

def test_issue_11877():
    # 创建一个符号变量 x
    x = symbols('x')
    # 断言对 log(S.Half - x) 在 [0, S.Half] 区间上的积分结果为 Rational(-1, 2) - log(2)/2
    assert integrate(log(S.Half - x), (x, 0, S.Half)) == Rational(-1, 2) - log(2)/2

def test_normal():
    # 创建一个符号变量 x
    x = symbols('x')
    # 创建一个表达式 e，表示 S.Half * (1 + x)，但不进行求值
    e = Mul(S.Half, 1 + x, evaluate=False)
    # 断言 e 的正常形式是 e 本身
    assert e.normal() == e

def test_expr():
    # 创建一个符号变量 x
    x = symbols('x')
    # 断言对 tan(x) 在 x=0 处展开到无穷阶后会引发 TypeError
    raises(TypeError, lambda: tan(x).series(x, 2, oo, "+"))

def test_ExprBuilder():
    # 待补充
    # 创建一个表达式构建器对象，指定表达式类型为乘法
    eb = ExprBuilder(Mul)
    # 将变量 x 作为乘法表达式的参数添加到表达式构建器对象中两次
    eb.args.extend([x, x])
    # 断言：使用表达式构建器构建的表达式应该等于 x 的平方
    assert eb.build() == x**2
def test_issue_22020():
    # 从 sympy.parsing.sympy_parser 导入 parse_expr 函数
    from sympy.parsing.sympy_parser import parse_expr
    # 解析第一个表达式并赋给变量 x
    x = parse_expr("log((2*V/3-V)/C)/-(R+r)*C")
    # 解析第二个表达式并赋给变量 y
    y = parse_expr("log((2*V/3-V)/C)/-(R+r)*2")
    # 断言 x 不等于 y
    assert x.equals(y) is False


def test_non_string_equality():
    # 表达式不应与字符串相等
    # 定义符号变量 x
    x = symbols('x')
    # 定义符号常量 1
    one = sympify(1)
    # 断言 x 等于字符串 'x' 是假的
    assert (x == 'x') is False
    # 断言 x 不等于字符串 'x' 是真的
    assert (x != 'x') is True
    # 断言 one 等于字符串 '1' 是假的
    assert (one == '1') is False
    # 断言 one 不等于字符串 '1' 是真的
    assert (one != '1') is True
    # 断言 x + 1 等于字符串 'x + 1' 是假的
    assert (x + 1 == 'x + 1') is False
    # 断言 x + 1 不等于字符串 'x + 1' 是真的
    assert (x + 1 != 'x + 1') is True

    # 确保 == 操作不会尝试将结果表达式转换为字符串
    # （例如通过调用 sympify() 而不是 _sympify()）

    # 定义一个具有错误 __repr__ 方法的类 BadRepr
    class BadRepr:
        def __repr__(self):
            raise RuntimeError

    # 断言 x 等于 BadRepr() 是假的
    assert (x == BadRepr()) is False
    # 断言 x 不等于 BadRepr() 是真的
    assert (x != BadRepr()) is True


def test_21494():
    # 从 sympy.testing.pytest 导入 warns_deprecated_sympy 函数
    from sympy.testing.pytest import warns_deprecated_sympy

    # 测试表达式的 expr_free_symbols 方法
    with warns_deprecated_sympy():
        assert x.expr_free_symbols == {x}

    with warns_deprecated_sympy():
        assert Basic().expr_free_symbols == set()

    with warns_deprecated_sympy():
        assert S(2).expr_free_symbols == {S(2)}

    with warns_deprecated_sympy():
        assert Indexed("A", x).expr_free_symbols == {Indexed("A", x)}

    with warns_deprecated_sympy():
        assert Subs(x, x, 0).expr_free_symbols == set()


def test_Expr__eq__iterable_handling():
    # 断言 x 不等于 range(3)
    assert x != range(3)


def test_format():
    # 格式化测试
    assert '{:1.2f}'.format(S.Zero) == '0.00'
    assert '{:+3.0f}'.format(S(3)) == ' +3'
    assert '{:23.20f}'.format(pi) == ' 3.14159265358979323846'
    assert '{:50.48f}'.format(exp(sin(1))) == '2.319776824715853173956590377503266813254904772376'


def test_issue_24045():
    # 断言 powsimp(exp(a)/((c*a - c*b)*(Float(1.0)*c*a - Float(1.0)*c*b))) 不会引发异常
    assert powsimp(exp(a)/((c*a - c*b)*(Float(1.0)*c*a - Float(1.0)*c*b)))  # doesn't raise
```