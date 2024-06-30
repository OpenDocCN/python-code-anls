# `D:\src\scipysrc\sympy\sympy\core\tests\test_basic.py`

```
# 导入必要的模块和类型声明
import collections
from typing import TypeVar, Generic

# 导入 sympy 库中的具体模块和类
from sympy.assumptions.ask import Q
from sympy.core.basic import (Basic, Atom, as_Basic,
    _atomic, _aresame)
from sympy.core.containers import Tuple
from sympy.core.function import Function, Lambda
from sympy.core.numbers import I, pi, Float
from sympy.core.singleton import S
from sympy.core.symbol import symbols, Symbol, Dummy
from sympy.concrete.summations import Sum
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.functions.elementary.exponential import exp
from sympy.testing.pytest import raises, warns_deprecated_sympy

# 创建基本对象实例
b1 = Basic()
b2 = Basic(b1)
b3 = Basic(b2)
b21 = Basic(b2, b1)
T = TypeVar('T')

# 测试 _aresame 函数的不同情况
def test__aresame():
    assert not _aresame(Basic(Tuple()), Basic())
    for i, j in [(S(2), S(2.)), (1., Float(1))]:
        for do in range(2):
            assert not _aresame(Basic(i), Basic(j))
            assert not _aresame(i, j)
            i, j = j, i

# 测试 Basic 类的结构和属性
def test_structure():
    assert b21.args == (b2, b1)
    assert b21.func(*b21.args) == b21
    assert bool(b1)

# 测试 Basic 类的不可变性
def test_immutable():
    assert not hasattr(b1, '__dict__')
    with raises(AttributeError):
        b1.x = 1

# 测试 Basic 类的相等性
def test_equality():
    instances = [b1, b2, b3, b21, Basic(b1, b1, b1), Basic]
    for i, b_i in enumerate(instances):
        for j, b_j in enumerate(instances):
            assert (b_i == b_j) == (i == j)
            assert (b_i != b_j) == (i != j)

    assert Basic() != []
    assert not(Basic() == [])
    assert Basic() != 0
    assert not(Basic() == 0)

    # 自定义类 Foo 和 Bar 用于测试 Basic 类的相等性
    class Foo:
        """
        Class that is unaware of Basic, and relies on both classes returning
        the NotImplemented singleton for equivalence to evaluate to False.
        """
    
    b = Basic()
    foo = Foo()

    assert b != foo
    assert foo != b
    assert not b == foo
    assert not foo == b

    class Bar:
        """
        Class that considers itself equal to any instance of Basic, and relies
        on Basic returning the NotImplemented singleton in order to achieve
        a symmetric equivalence relation.
        """
        def __eq__(self, other):
            if isinstance(other, Basic):
                return True
            return NotImplemented

        def __ne__(self, other):
            return not self == other

    bar = Bar()

    assert b == bar
    assert bar == b
    assert not b != bar
    assert not bar != b

# 测试 matches 方法在不同实例间的匹配性
def test_matches_basic():
    instances = [Basic(b1, b1, b2), Basic(b1, b2, b1), Basic(b2, b1, b1),
                 Basic(b1, b2), Basic(b2, b1), b2, b1]
    for i, b_i in enumerate(instances):
        for j, b_j in enumerate(instances):
            if i == j:
                assert b_i.matches(b_j) == {}
            else:
                assert b_i.matches(b_j) is None
    # 断言语句，用于检查表达式的真实性，如果表达式为假，则抛出 AssertionError 异常
    assert b1.match(b1) == {}
    # 在此处断言，期望 b1.match(b1) 返回一个空字典 {}，如果不是空字典则断言失败
# 测试方法，验证对象 b21 是否包含指定的参数或类型
def test_has():
    assert b21.has(b1)  # 断言 b21 是否包含 b1
    assert b21.has(b3, b1)  # 断言 b21 是否同时包含 b3 和 b1
    assert b21.has(Basic)  # 断言 b21 是否包含 Basic 类型
    assert not b1.has(b21, b3)  # 断言 b1 是否不包含 b21 和 b3
    assert not b21.has()  # 断言 b21 是否不包含任何参数
    assert not b21.has(str)  # 断言 b21 是否不包含 str 类型
    assert not Symbol("x").has("x")  # 断言 Symbol("x") 是否不包含字符串 "x"


# 测试方法，验证对象 b21 的符号替换功能
def test_subs():
    assert b21.subs(b2, b1) == Basic(b1, b1)  # 验证将 b2 替换为 b1 后的结果
    assert b21.subs(b2, b21) == Basic(b21, b1)  # 验证将 b2 替换为 b21 后的结果
    assert b3.subs(b2, b1) == b2  # 验证将 b2 替换为 b1 后的结果

    assert b21.subs([(b2, b1), (b1, b2)]) == Basic(b2, b2)  # 验证同时替换多个参数后的结果

    assert b21.subs({b1: b2, b2: b1}) == Basic(b2, b2)  # 使用字典进行替换验证
    assert b21.subs(collections.ChainMap({b1: b2}, {b2: b1})) == Basic(b2, b2)  # 使用 ChainMap 进行替换验证
    assert b21.subs(collections.OrderedDict([(b2, b1), (b1, b2)])) == Basic(b2, b2)  # 使用 OrderedDict 进行替换验证

    raises(ValueError, lambda: b21.subs('bad arg'))  # 断言当传入非法参数时会引发 ValueError
    raises(ValueError, lambda: b21.subs(b1, b2, b3))  # 断言当传入多个参数时会引发 ValueError
    raises(ValueError, lambda: b21.subs(b1='bad arg'))  # 断言当传入字符串参数时会引发 ValueError

    assert Symbol("text").subs({"text": b1}) == b1  # 验证使用字符串进行符号替换
    assert Symbol("s").subs({"s": 1}) == 1  # 验证使用字符串进行符号替换


# 测试方法，验证对象 b21 的 atoms() 方法
def test_atoms():
    assert b21.atoms() == {Basic()}  # 验证对象 b21 的原子集合是否符合预期


# 测试方法，验证对象 b21 的 free_symbols 属性
def test_free_symbols_empty():
    assert b21.free_symbols == set()  # 验证对象 b21 的自由符号集合是否为空集合


# 测试方法，验证对象 b21 的 doit() 方法
def test_doit():
    assert b21.doit() == b21  # 验证对象 b21 的 doit() 方法返回值是否为 b21 自身
    assert b21.doit(deep=False) == b21  # 验证对象 b21 的 doit(deep=False) 方法返回值是否为 b21 自身


# 测试方法，验证对象 S 的 repr 表示
def test_S():
    assert repr(S) == 'S'  # 验证对象 S 的字符串表示是否为 'S'


# 测试方法，验证对象 b21 的 xreplace() 方法
def test_xreplace():
    assert b21.xreplace({b2: b1}) == Basic(b1, b1)  # 验证将 b2 替换为 b1 后的结果
    assert b21.xreplace({b2: b21}) == Basic(b21, b1)  # 验证将 b2 替换为 b21 后的结果
    assert b3.xreplace({b2: b1}) == b2  # 验证将 b2 替换为 b1 后的结果
    assert Basic(b1, b2).xreplace({b1: b2, b2: b1}) == Basic(b2, b1)  # 验证同时替换多个参数后的结果
    assert Atom(b1).xreplace({b1: b2}) == Atom(b1)  # 验证替换不匹配的原子后的结果
    assert Atom(b1).xreplace({Atom(b1): b2}) == b2  # 验证替换匹配的原子后的结果
    raises(TypeError, lambda: b1.xreplace())  # 断言当不传入参数时会引发 TypeError
    raises(TypeError, lambda: b1.xreplace([b1, b2]))  # 断言当传入列表参数时会引发 TypeError
    for f in (exp, Function('f')):
        assert f.xreplace({}) == f  # 验证不传入参数时的替换结果
        assert f.xreplace({}, hack2=True) == f  # 验证使用 hack2=True 参数时的替换结果
        assert f.xreplace({f: b1}) == b1  # 验证将 f 替换为 b1 后的结果
        assert f.xreplace({f: b1}, hack2=True) == b1  # 验证使用 hack2=True 参数时将 f 替换为 b1 的结果


# 测试方法，验证对象 b21 的 sorted_args 属性
def test_sorted_args():
    x = symbols('x')
    assert b21._sorted_args == b21.args  # 验证对象 b21 的排序参数与其参数列表是否一致
    raises(AttributeError, lambda: x._sorted_args)  # 断言对符号 x 访问 _sorted_args 属性会引发 AttributeError


# 测试方法，验证对象的 call() 方法
def test_call():
    x, y = symbols('x y')
    # 见 issues 5026 和 5105 的长期历史。

    raises(TypeError, lambda: sin(x)({ x : 1, sin(x) : 2}))  # 断言在无法处理的情况下会引发 TypeError
    raises(TypeError, lambda: sin(x)(1))  # 断言当传入非法参数时会引发 TypeError

    # 因为没有可调用对象，所以不会产生影响
    assert sin(x).rcall(1) == sin(x)
    assert (1 + sin(x)).rcall(1) == 1 + sin(x)

    # 在存在可调用对象时会产生影响
    l = Lambda(x, 2*x)
    assert (l + x).rcall(y) == 2*y + x
    assert (x**l).rcall(2) == x**4
    # TODO UndefinedFunction does not subclass Expr
    #f = Function('f')
    #assert (2*f)(x) == 2*f(x)
    # 断言语句，用于验证条件是否满足；此处断言：
    # Q.real & Q.positive).rcall(x) == Q.real(x) & Q.positive(x)
    # 意味着断言 Q.real & Q.positive 对象对输入 x 的调用结果，与 Q.real(x) & Q.positive(x) 的结果相等。
    # & 符号表示按位与操作，rcall 是 Q.real & Q.positive 对象的方法，用于对 x 进行调用。
    assert (Q.real & Q.positive).rcall(x) == Q.real(x) & Q.positive(x)
def test_rewrite():
    # 定义符号变量 x, y, z
    x, y, z = symbols('x y z')
    # 定义符号变量 a, b
    a, b = symbols('a b')
    # 定义表达式 f1
    f1 = sin(x) + cos(x)
    # 断言 f1 用 cos 重写后的结果
    assert f1.rewrite(cos, exp) == exp(I*x)/2 + sin(x) + exp(-I*x)/2
    # 断言 f1 用 sin 重写后的结果
    assert f1.rewrite([cos], sin) == sin(x) + sin(x + pi/2, evaluate=False)
    # 定义表达式 f2
    f2 = sin(x) + cos(y)/gamma(z)
    # 断言 f2 用 sin 重写后的结果
    assert f2.rewrite(sin, exp) == -I*(exp(I*x) - exp(-I*x))/2 + cos(y)/gamma(z)

    # 断言 f1 被重写为自身不变
    assert f1.rewrite() == f1

def test_literal_evalf_is_number_is_zero_is_comparable():
    # 定义符号变量 x
    x = symbols('x')
    # 定义函数 f
    f = Function('f')

    # issue 5033
    # 断言 f 不是数值
    assert f.is_number is False
    # issue 6646
    # 断言 f(1) 不是数值
    assert f(1).is_number is False
    # 创建积分对象 i
    i = Integral(0, (x, x, x))
    # 断言 i 的数值化结果为 0
    assert i.n() == 0
    # 断言 i 是零
    assert i.is_zero
    # 断言 i 不是数值
    assert i.is_number is False
    # 断言 i 的数值化结果为 0，且允许非严格模式
    assert i.evalf(2, strict=False) == 0

    # issue 10268
    # 定义表达式 n
    n = sin(1)**2 + cos(1)**2 - 1
    # 断言 n 不可比较
    assert n.is_comparable is False
    # 断言 n 的数值化结果不可比较
    assert n.n(2).is_comparable is False
    # 断言 n 的两次数值化结果可比较
    assert n.n(2).n(2).is_comparable


def test_as_Basic():
    # 断言 as_Basic(1) 返回 S.One
    assert as_Basic(1) is S.One
    # 断言 as_Basic(()) 返回空元组 Tuple()
    assert as_Basic(()) == Tuple()
    # 断言当输入为列表时会抛出 TypeError 异常
    raises(TypeError, lambda: as_Basic([]))


def test_atomic():
    # 将函数 g 和 h 映射为符号函数
    g, h = map(Function, 'gh')
    # 定义符号变量 x
    x = symbols('x')
    # 断言 _atomic(g(x + h(x))) 的结果
    assert _atomic(g(x + h(x))) == {g(x + h(x))}
    # 断言 _atomic(g(x + h(x)), recursive=True) 的结果
    assert _atomic(g(x + h(x)), recursive=True) == {h(x), x, g(x + h(x))}
    # 断言 _atomic(1) 的结果
    assert _atomic(1) == set()
    # 断言 _atomic(Basic(S(1), S(2))) 的结果
    assert _atomic(Basic(S(1), S(2))) == set()


def test_as_dummy():
    # 定义符号变量 u, v, x, y, z, _0, _1
    u, v, x, y, z, _0, _1 = symbols('u v x y z _0 _1')
    # 断言 Lambda(x, x + 1).as_dummy() 的结果
    assert Lambda(x, x + 1).as_dummy() == Lambda(_0, _0 + 1)
    # 断言 Lambda(x, x + _0).as_dummy() 的结果
    assert Lambda(x, x + _0).as_dummy() == Lambda(_1, _0 + _1)
    # 创建积分表达式 eq
    eq = (1 + Sum(x, (x, 1, x)))
    # 创建预期结果 ans
    ans = 1 + Sum(_0, (_0, 1, x))
    # 断言 eq.as_dummy() 的结果
    once = eq.as_dummy()
    assert once == ans
    # 断言 once.as_dummy() 的结果
    twice = once.as_dummy()
    assert twice == ans
    # 断言 Integral(x + _0, (x, x + 1), (_0, 1, 2)).as_dummy() 的结果
    assert Integral(x + _0, (x, x + 1), (_0, 1, 2)).as_dummy() == Integral(_0 + _1, (_0, x + 1), (_1, 1, 2))
    # 断言 Dummy().as_dummy().is_commutative 的结果为 True
    assert Dummy().as_dummy().is_commutative
    # 断言 Dummy(commutative=False).as_dummy().is_commutative 的结果为 False
    assert Dummy(commutative=False).as_dummy().is_commutative is False


def test_canonical_variables():
    # 定义符号变量 x, i0, i1
    x, i0, i1 = symbols('x _:2')
    # 断言 Integral(x, (x, x + 1)).canonical_variables 的结果
    assert Integral(x, (x, x + 1)).canonical_variables == {x: i0}
    # 断言 Integral(x, (x, x + 1), (i0, 1, 2)).canonical_variables 的结果
    assert Integral(x, (x, x + 1), (i0, 1, 2)).canonical_variables == {x: i0, i0: i1}
    # 断言 Integral(x, (x, x + i0)).canonical_variables 的结果
    assert Integral(x, (x, x + i0)).canonical_variables == {x: i1}


def test_replace_exceptions():
    # 导入 Wild 模块
    from sympy.core.symbol import Wild
    # 定义符号变量 x, y
    x, y = symbols('x y')
    # 创建表达式 e
    e = (x**2 + x*y)
    # 断言替换 sin 为 2 时会抛出 TypeError 异常
    raises(TypeError, lambda: e.replace(sin, 2))
    # 创建 Wild 变量 b, c
    b = Wild('b')
    c = Wild('c')
    # 断言替换 b*c 为 c.is_real 时会抛出 TypeError 异常
    raises(TypeError, lambda: e.replace(b*c, c.is_real))
    # 断言替换 b.is_real 为 1 时会抛出 TypeError 异常
    raises(TypeError, lambda: e.replace(b.is_real, 1))
    # 调用 e.replace 方法并断言其会抛出 TypeError 异常
    raises(TypeError, lambda: e.replace(lambda d: d.is_Number, 1))
# 测试 ManagedProperties 类的行为，该类已被弃用。这里我们尽力检查，
# 如果有人在使用它，则应该以之前的方式工作，但会发出弃用警告。
def test_ManagedProperties():
    # 导入 ManagedProperties 类
    from sympy.core.assumptions import ManagedProperties

    # 用于记录执行 MyMeta 类构造函数的调用
    myclasses = []

    # 自定义的元类 MyMeta，继承自 ManagedProperties
    class MyMeta(ManagedProperties):
        def __init__(cls, *args, **kwargs):
            # 在初始化时记录执行
            myclasses.append('executed')
            super().__init__(*args, **kwargs)

    # 要执行的动态代码字符串
    code = """
class MySubclass(Basic, metaclass=MyMeta):
    pass
"""

    # 使用 warns_deprecated_sympy 上下文来检测是否发出了弃用警告
    with warns_deprecated_sympy():
        # 执行动态代码字符串
        exec(code)

    # 断言 myclasses 列表应该包含 'executed'，证明 MyMeta 的构造函数被调用
    assert myclasses == ['executed']


# 测试泛型类型的定义
def test_generic():
    # 定义一个符号类 A，同时指定其为泛型类型 Generic[T] 的子类
    class A(Symbol, Generic[T]):
        pass

    # 定义类 B，它继承自 A[T]，其中 T 是泛型参数
    class B(A[T]):
        pass
```