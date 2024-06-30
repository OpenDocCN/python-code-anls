# `D:\src\scipysrc\sympy\sympy\core\tests\test_symbol.py`

```
# 导入线程模块
import threading

# 从 sympy 核心函数中导入 Function 和 UndefinedFunction 类
from sympy.core.function import Function, UndefinedFunction
# 从 sympy 核心数字模块中导入 I, Rational, pi
from sympy.core.numbers import (I, Rational, pi)
# 从 sympy 核心关系模块中导入 GreaterThan, LessThan, StrictGreaterThan, StrictLessThan 类
from sympy.core.relational import (GreaterThan, LessThan, StrictGreaterThan, StrictLessThan)
# 从 sympy 核心符号模块中导入 Dummy, Symbol, Wild, symbols 类
from sympy.core.symbol import (Dummy, Symbol, Wild, symbols)
# 从 sympy 核心 sympify 模块中导入 sympify 函数（目前不能用 S 作为别名导入）
from sympy.core.sympify import sympify
# 从 sympy 核心符号模块中导入 uniquely_named_symbol, _symbol, Str 函数
from sympy.core.symbol import uniquely_named_symbol, _symbol, Str

# 从 sympy 测试模块中导入 raises, skip_under_pyodide 函数
from sympy.testing.pytest import raises, skip_under_pyodide
# 从 sympy 核心符号模块中导入 disambiguate 函数


def test_Str():
    # 创建字符串符号 'a1' 和 'a2'，它们的值相等
    a1 = Str('a')
    a2 = Str('a')
    # 创建字符串符号 'b'
    b = Str('b')
    # 断言 'a1' 等于 'a2' 且不等于 'b'
    assert a1 == a2 != b
    # 断言尝试创建空的 Str 对象会引发 TypeError 异常
    raises(TypeError, lambda: Str())


def test_Symbol():
    # 创建符号 'a'
    a = Symbol("a")
    # 创建符号 'x1' 和 'x2'，它们的名称相同
    x1 = Symbol("x")
    x2 = Symbol("x")
    # 创建虚拟符号 'xdummy1' 和 'xdummy2'
    xdummy1 = Dummy("x")
    xdummy2 = Dummy("x")

    # 断言 'a' 不等于 'x1' 且不等于 'x2'
    assert a != x1
    assert a != x2
    # 断言 'x1' 等于 'x2'
    assert x1 == x2
    # 断言 'x1' 不等于 'xdummy1'
    assert x1 != xdummy1
    # 断言 'xdummy1' 不等于 'xdummy2'
    assert xdummy1 != xdummy2

    # 断言创建相同名称的符号对象会相等
    assert Symbol("x") == Symbol("x")
    # 断言创建相同名称的虚拟符号对象不相等
    assert Dummy("x") != Dummy("x")
    # 创建 'd' 符号的虚拟符号对象，并检查其类型
    d = symbols('d', cls=Dummy)
    assert isinstance(d, Dummy)
    # 创建 'c' 和 'd' 符号的虚拟符号对象，并检查其类型
    c, d = symbols('c,d', cls=Dummy)
    assert isinstance(c, Dummy)
    assert isinstance(d, Dummy)
    # 断言尝试创建空的 Symbol 对象会引发 TypeError 异常
    raises(TypeError, lambda: Symbol())


def test_Dummy():
    # 断言创建的两个 Dummy 对象不相等
    assert Dummy() != Dummy()


def test_Dummy_force_dummy_index():
    # 断言尝试为 Dummy 对象设置 dummy_index 引发 AssertionError 异常
    raises(AssertionError, lambda: Dummy(dummy_index=1))
    # 创建具有相同名称和 dummy_index 的两个 Dummy 对象，并断言它们相等
    assert Dummy('d', dummy_index=2) == Dummy('d', dummy_index=2)
    # 创建 dummy_index 不同的两个 Dummy 对象，并断言它们不相等
    assert Dummy('d1', dummy_index=2) != Dummy('d2', dummy_index=2)
    # 创建 dummy_index 不同的两个 Dummy 对象，并检查其不相等的情况
    d1 = Dummy('d', dummy_index=3)
    d2 = Dummy('d')
    # 断言创建的 Dummy 对象的计数值相等
    assert d1 != d2
    # 创建相同名称和 dummy_index 的两个 Dummy 对象，并断言它们相等
    d3 = Dummy('d', dummy_index=3)
    assert d1 == d3
    # 断言 Dummy 对象的计数器值相等
    assert Dummy()._count == Dummy('d', dummy_index=3)._count


def test_lt_gt():
    S = sympify
    x, y = Symbol('x'), Symbol('y')

    # 检查符号表达式 'x >= y' 是否等同于 GreaterThan(x, y)
    assert (x >= y) == GreaterThan(x, y)
    # 检查符号表达式 'x >= 0' 是否等同于 GreaterThan(x, 0)
    assert (x >= 0) == GreaterThan(x, 0)
    # 检查符号表达式 'x <= y' 是否等同于 LessThan(x, y)
    assert (x <= y) == LessThan(x, y)
    # 检查符号表达式 'x <= 0' 是否等同于 LessThan(x, 0)

    assert (x <= 0) == LessThan(x, 0)
    assert (0 >= x) == LessThan(x, 0)
    assert (S(0) >= x) == GreaterThan(0, x)
    assert (S(0) <= x) == LessThan(0, x)

    assert (x > y) == StrictGreaterThan(x, y)
    assert (x > 0) == StrictGreaterThan(x, 0)
    assert (x < y) == StrictLessThan(x, y)
    assert (x < 0) == StrictLessThan(x, 0)

    assert (0 < x) == StrictGreaterThan(x, 0)
    assert (0 > x) == StrictLessThan(x, 0)
    assert (S(0) > x) == StrictGreaterThan(0, x)
    assert (S(0) < x) == StrictLessThan(0, x)

    e = x**2 + 4*x + 1
    assert (e >= 0) == GreaterThan(e, 0)
    assert (0 <= e) == GreaterThan(e, 0)
    assert (e > 0) == StrictGreaterThan(e, 0)
    assert (0 < e) == StrictGreaterThan(e, 0)

    assert (e <= 0) == LessThan(e, 0)
    assert (0 >= e) == LessThan(e, 0)
    assert (e < 0) == StrictLessThan(e, 0)
    assert (0 > e) == StrictLessThan(e, 0)

    assert (S(0) >= e) == GreaterThan(0, e)
    assert (S(0) <= e) == LessThan(0, e)
    assert (S(0) < e) == StrictLessThan(0, e)
    # 使用断言来检查表达式 (S(0) > e) 是否成立，预期结果是严格大于0与e的比较结果
    assert (S(0) > e) == StrictGreaterThan(0, e)
def test_no_len():
    # 定义符号变量 x
    x = Symbol('x')
    # 检查对于数字来说不应该有 len 函数，期望会抛出 TypeError 异常
    raises(TypeError, lambda: len(x))


def test_ineq_unequal():
    # 定义 sympify 函数为 S
    S = sympify
    # 定义符号变量 x, y, z
    x, y, z = symbols('x,y,z')

    # 创建一个包含多个不等式表达式的元组 e
    e = (
        S(-1) >= x, S(-1) >= y, S(-1) >= z,
        S(-1) > x, S(-1) > y, S(-1) > z,
        S(-1) <= x, S(-1) <= y, S(-1) <= z,
        S(-1) < x, S(-1) < y, S(-1) < z,
        S(0) >= x, S(0) >= y, S(0) >= z,
        S(0) > x, S(0) > y, S(0) > z,
        S(0) <= x, S(0) <= y, S(0) <= z,
        S(0) < x, S(0) < y, S(0) < z,
        S('3/7') >= x, S('3/7') >= y, S('3/7') >= z,
        S('3/7') > x, S('3/7') > y, S('3/7') > z,
        S('3/7') <= x, S('3/7') <= y, S('3/7') <= z,
        S('3/7') < x, S('3/7') < y, S('3/7') < z,
        S(1.5) >= x, S(1.5) >= y, S(1.5) >= z,
        S(1.5) > x, S(1.5) > y, S(1.5) > z,
        S(1.5) <= x, S(1.5) <= y, S(1.5) <= z,
        S(1.5) < x, S(1.5) < y, S(1.5) < z,
        S(2) >= x, S(2) >= y, S(2) >= z,
        S(2) > x, S(2) > y, S(2) > z,
        S(2) <= x, S(2) <= y, S(2) <= z,
        S(2) < x, S(2) < y, S(2) < z,
        x >= -1, y >= -1, z >= -1,
        x > -1, y > -1, z > -1,
        x <= -1, y <= -1, z <= -1,
        x < -1, y < -1, z < -1,
        x >= 0, y >= 0, z >= 0,
        x > 0, y > 0, z > 0,
        x <= 0, y <= 0, z <= 0,
        x < 0, y < 0, z < 0,
        x >= 1.5, y >= 1.5, z >= 1.5,
        x > 1.5, y > 1.5, z > 1.5,
        x <= 1.5, y <= 1.5, z <= 1.5,
        x < 1.5, y < 1.5, z < 1.5,
        x >= 2, y >= 2, z >= 2,
        x > 2, y > 2, z > 2,
        x <= 2, y <= 2, z <= 2,
        x < 2, y < 2, z < 2,

        # 以下为一些比较和逻辑操作
        x >= y, x >= z, y >= x, y >= z, z >= x, z >= y,
        x > y, x > z, y > x, y > z, z > x, z > y,
        x <= y, x <= z, y <= x, y <= z, z <= x, z <= y,
        x < y, x < z, y < x, y < z, z < x, z < y,

        # 复杂的数学表达式比较
        x - pi >= y + z, y - pi >= x + z, z - pi >= x + y,
        x - pi > y + z, y - pi > x + z, z - pi > x + y,
        x - pi <= y + z, y - pi <= x + z, z - pi <= x + y,
        x - pi < y + z, y - pi < x + z, z - pi < x + y,

        # True 和 False 值的比较
        True, False
    )

    # 提取元组 e 中除最后一个元素外的所有元素，依次与之后的元素进行比较，期望它们都不相等
    left_e = e[:-1]
    for i, e1 in enumerate(left_e):
        for e2 in e[i + 1:]:
            assert e1 != e2


def test_Wild_properties():
    # 定义 sympify 函数为 S
    S = sympify
    # 定义符号变量 x, y, p, k, n 分别带有特定属性
    x = Symbol("x")
    y = Symbol("y")
    p = Symbol("p", positive=True)
    k = Symbol("k", integer=True)
    n = Symbol("n", integer=True, positive=True)

    # 给定一组符合特定条件的模式
    given_patterns = [ x, y, p, k, -k, n, -n, S(-3), S(3),
                       pi, Rational(3, 2), I ]

    # 定义一些函数，用于检查 Wild 对象的属性
    integerp = lambda k: k.is_integer
    positivep = lambda k: k.is_positive
    symbolp = lambda k: k.is_Symbol
    realp = lambda k: k.is_extended_real

    # 定义具有特定属性的 Wild 对象
    S = Wild("S", properties=[symbolp])
    R = Wild("R", properties=[realp])
    Y = Wild("Y", exclude=[x, p, k, n])
    P = Wild("P", properties=[positivep])
    K = Wild("K", properties=[integerp])
    N = Wild("N", properties=[positivep, integerp])

    # 给定一组具有特定属性的 Wild 对象
    given_wildcards = [ S, R, Y, P, K, N ]
    # 定义一个字典 goodmatch，将每个符号映射到一个包含相关值的元组
    goodmatch = {
        S: (x, y, p, k, n),  # 将符号 S 映射到元组 (x, y, p, k, n)
        R: (p, k, -k, n, -n, -3, 3, pi, Rational(3, 2)),  # 将符号 R 映射到元组 (p, k, -k, n, -n, -3, 3, pi, Rational(3, 2))
        Y: (y, -3, 3, pi, Rational(3, 2), I ),  # 将符号 Y 映射到元组 (y, -3, 3, pi, Rational(3, 2), I)
        P: (p, n, 3, pi, Rational(3, 2)),  # 将符号 P 映射到元组 (p, n, 3, pi, Rational(3, 2))
        K: (k, -k, n, -n, -3, 3),  # 将符号 K 映射到元组 (k, -k, n, -n, -3, 3)
        N: (n, 3)  # 将符号 N 映射到元组 (n, 3)
    }

    # 遍历给定的通配符列表 given_wildcards
    for A in given_wildcards:
        # 遍历给定的模式列表 given_patterns
        for pat in given_patterns:
            # 使用模式对象 pat 对通配符 A 进行匹配，返回匹配结果字典 d
            d = pat.match(A)
            # 如果模式 pat 在 goodmatch 中对应的值中
            if pat in goodmatch[A]:
                # 断言匹配结果字典 d 中的 A 键对应的值在 goodmatch[A] 中
                assert d[A] in goodmatch[A]
            else:
                # 如果模式 pat 不在 goodmatch 中对应的值中，则断言 d 为 None
                assert d is None
def test_symbols():
    # 定义三个符号对象 x, y, z
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')

    # 测试 symbols 函数的不同输入形式是否返回正确的符号对象
    assert symbols('x') == x
    assert symbols('x ') == x
    assert symbols(' x ') == x
    assert symbols('x,') == (x,)
    assert symbols('x, ') == (x,)
    assert symbols('x ,') == (x,)

    assert symbols('x , y') == (x, y)

    assert symbols('x,y,z') == (x, y, z)
    assert symbols('x y z') == (x, y, z)

    assert symbols('x,y,z,') == (x, y, z)
    assert symbols('x y z ') == (x, y, z)

    # 定义两个额外的符号对象 xyz 和 abc
    xyz = Symbol('xyz')
    abc = Symbol('abc')

    # 测试 symbols 函数对不同形式的输入字符串的处理是否正确
    assert symbols('xyz') == xyz
    assert symbols('xyz,') == (xyz,)
    assert symbols('xyz,abc') == (xyz, abc)

    assert symbols(('xyz',)) == (xyz,)
    assert symbols(('xyz,',)) == ((xyz,),)
    assert symbols(('x,y,z,',)) == ((x, y, z),)
    assert symbols(('xyz', 'abc')) == (xyz, abc)
    assert symbols(('xyz,abc',)) == ((xyz, abc),)
    assert symbols(('xyz,abc', 'x,y,z')) == ((xyz, abc), (x, y, z))

    assert symbols(('x', 'y', 'z')) == (x, y, z)
    assert symbols(['x', 'y', 'z']) == [x, y, z]
    assert symbols({'x', 'y', 'z'}) == {x, y, z}

    # 测试 symbols 函数对不合法的输入字符串是否抛出 ValueError 异常
    raises(ValueError, lambda: symbols(''))
    raises(ValueError, lambda: symbols(','))
    raises(ValueError, lambda: symbols('x,,y,,z'))
    raises(ValueError, lambda: symbols(('x', '', 'y', '', 'z')))

    # 测试 symbols 函数在带有额外参数的情况下的行为
    a, b = symbols('x,y', real=True)
    assert a.is_real and b.is_real

    # 定义另外三个符号对象 x0, x1, x2 和 y0, y1
    x0 = Symbol('x0')
    x1 = Symbol('x1')
    x2 = Symbol('x2')

    y0 = Symbol('y0')
    y1 = Symbol('y1')

    # 测试 symbols 函数对使用冒号表示法的处理是否正确
    assert symbols('x0:0') == ()
    assert symbols('x0:1') == (x0,)
    assert symbols('x0:2') == (x0, x1)
    assert symbols('x0:3') == (x0, x1, x2)

    assert symbols('x:0') == ()
    assert symbols('x:1') == (x0,)
    assert symbols('x:2') == (x0, x1)
    assert symbols('x:3') == (x0, x1, x2)

    assert symbols('x1:1') == ()
    assert symbols('x1:2') == (x1,)
    assert symbols('x1:3') == (x1, x2)

    assert symbols('x1:3,x,y,z') == (x1, x2, x, y, z)

    assert symbols('x:3,y:2') == (x0, x1, x2, y0, y1)
    assert symbols(('x:3', 'y:2')) == ((x0, x1, x2), (y0, y1))

    # 定义符号对象 a, b, c, d
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')

    # 测试 symbols 函数在复合输入下的表现
    assert symbols('x:z') == (x, y, z)
    assert symbols('a:d,x:z') == (a, b, c, d, x, y, z)
    assert symbols(('a:d', 'x:z')) == ((a, b, c, d), (x, y, z))

    # 定义符号对象 aa, ab, ac, ad
    aa = Symbol('aa')
    ab = Symbol('ab')
    ac = Symbol('ac')
    ad = Symbol('ad')

    # 测试 symbols 函数在复杂输入下的表现
    assert symbols('aa:d') == (aa, ab, ac, ad)
    assert symbols('aa:d,x:z') == (aa, ab, ac, ad, x, y, z)
    assert symbols(('aa:d','x:z')) == ((aa, ab, ac, ad), (x, y, z))

    # 测试 symbols 函数在使用类参数时的行为
    assert type(symbols(('q:2', 'u:2'), cls=Function)[0][0]) == UndefinedFunction  # issue 23532

    # issue 6675
    # 定义一个辅助函数 sym，用于测试
    def sym(s):
        return str(symbols(s))
    assert sym('a0:4') == '(a0, a1, a2, a3)'
    assert sym('a2:4,b1:3') == '(a2, a3, b1, b2)'
    assert sym('a1(2:4)') == '(a12, a13)'
    assert sym('a0:2.0:2') == '(a0.0, a0.1, a1.0, a1.1)'
    assert sym('aa:cz') == '(aaz, abz, acz)'
    # 使用 assert 断言来验证 sym 函数的输出是否符合预期
    
    assert sym('aa:c0:2') == '(aa0, aa1, ab0, ab1, ac0, ac1)'
    # 测试输入 'aa:c0:2'，预期输出为 '(aa0, aa1, ab0, ab1, ac0, ac1)'
    
    assert sym('aa:ba:b') == '(aaa, aab, aba, abb)'
    # 测试输入 'aa:ba:b'，预期输出为 '(aaa, aab, aba, abb)'
    
    assert sym('a:3b') == '(a0b, a1b, a2b)'
    # 测试输入 'a:3b'，预期输出为 '(a0b, a1b, a2b)'
    
    assert sym('a-1:3b') == '(a-1b, a-2b)'
    # 测试输入 'a-1:3b'，预期输出为 '(a-1b, a-2b)'
    
    assert sym(r'a:2\,:2' + chr(0)) == '(a0,0\x00, a0,1\x00, a1,0\x00, a1,1\x00)'
    # 测试输入 'a:2,:2'，其中包含特殊字符 \x00，预期输出为格式化后的字符串
    
    assert sym('x(:a:3)') == '(x(a0), x(a1), x(a2))'
    # 测试输入 'x(:a:3)'，预期输出为 '(x(a0), x(a1), x(a2))'
    
    assert sym('x(:c):1') == '(xa0, xb0, xc0)'
    # 测试输入 'x(:c):1'，预期输出为 '(xa0, xb0, xc0)'
    
    assert sym('x((:a)):3') == '(x(a)0, x(a)1, x(a)2)'
    # 测试输入 'x((:a)):3'，预期输出为 '(x(a)0, x(a)1, x(a)2)'
    
    assert sym('x(:a:3') == '(x(a0, x(a1, x(a2)'
    # 测试输入 'x(:a:3'，期望引发 ValueError 异常
    
    assert sym(':2') == '(0, 1)'
    # 测试输入 ':2'，预期输出为 '(0, 1)'
    
    assert sym(':b') == '(a, b)'
    # 测试输入 ':b'，预期输出为 '(a, b)'
    
    assert sym(':b:2') == '(a0, a1, b0, b1)'
    # 测试输入 ':b:2'，预期输出为 '(a0, a1, b0, b1)'
    
    assert sym(':2:2') == '(00, 01, 10, 11)'
    # 测试输入 ':2:2'，预期输出为 '(00, 01, 10, 11)'
    
    assert sym(':b:b') == '(aa, ab, ba, bb)'
    # 测试输入 ':b:b'，预期输出为 '(aa, ab, ba, bb)'
    
    # 使用 raises 断言来验证 symbols 函数对无效输入是否会引发 ValueError 异常
    
    raises(ValueError, lambda: symbols(':'))
    # 测试输入 ':'，预期引发 ValueError 异常
    
    raises(ValueError, lambda: symbols('a:'))
    # 测试输入 'a:'，预期引发 ValueError 异常
    
    raises(ValueError, lambda: symbols('::'))
    # 测试输入 '::'，预期引发 ValueError 异常
    
    raises(ValueError, lambda: symbols('a::'))
    # 测试输入 'a::'，预期引发 ValueError 异常
    
    raises(ValueError, lambda: symbols(':a:'))
    # 测试输入 ':a:'，预期引发 ValueError 异常
    
    raises(ValueError, lambda: symbols('::a'))
    # 测试输入 '::a'，预期引发 ValueError 异常
# 测试函数，用于检查 symbols 变成函数的问题（issue 3539）
def test_symbols_become_functions_issue_3539():
    # 从 sympy.abc 中导入 alpha, phi, beta, t 符号
    from sympy.abc import alpha, phi, beta, t
    # 检查 beta 函数的类型错误异常
    raises(TypeError, lambda: beta(2))
    # 检查 beta 函数的类型错误异常
    raises(TypeError, lambda: beta(2.5))
    # 检查 phi 函数的类型错误异常
    raises(TypeError, lambda: phi(2.5))
    # 检查 alpha 函数的类型错误异常
    raises(TypeError, lambda: alpha(2.5))
    # 检查 phi 函数的类型错误异常
    raises(TypeError, lambda: phi(t))


# 测试 Unicode 相关功能
def test_unicode():
    # 创建符号 xu
    xu = Symbol('x')
    # 创建符号 x
    x = Symbol('x')
    # 断言 xu 和 x 相等
    assert x == xu
    # 检查使用非法参数创建符号的类型错误异常
    raises(TypeError, lambda: Symbol(1))


# 测试唯一命名的符号和 Symbol 函数
def test_uniquely_named_symbol_and_Symbol():
    # 将 uniquely_named_symbol 函数赋给 F
    F = uniquely_named_symbol
    # 创建符号 x
    x = Symbol('x')
    # 断言 F(x) 等于 x
    assert F(x) == x
    # 断言 F('x') 等于 x
    assert F('x') == x
    # 断言 F('x', x) 的字符串表示为 'x0'
    assert str(F('x', x)) == 'x0'
    # 断言 F('x', (x + 1, 1/x)) 的字符串表示为 'x0'
    assert str(F('x', (x + 1, 1/x))) == 'x0'
    # 创建一个具有 real=True 属性的符号 _x
    _x = Symbol('x', real=True)
    # 断言 F(('x', _x)) 等于 _x
    assert F(('x', _x)) == _x
    # 断言 F((x, _x)) 等于 _x
    assert F((x, _x)) == _x
    # 断言 F('x', real=True) 的 is_real 属性为 True
    assert F('x', real=True).is_real
    # 创建符号 y
    y = Symbol('y')
    # 断言 F(('x', y), real=True) 的 is_real 属性为 True
    assert F(('x', y), real=True).is_real
    # 创建一个具有 real=True 属性的符号 r
    r = Symbol('x', real=True)
    # 断言 F(('x', r)) 的 is_real 属性为 True
    assert F(('x', r)).is_real
    # 断言 F(('x', r), real=False) 的 is_real 属性为 True
    assert F(('x', r), real=False).is_real
    # 使用比较函数，断言 F('x1', Symbol('x1'), compare=lambda i: str(i).rstrip('1')).name 为 'x0'
    assert F('x1', Symbol('x1'),
             compare=lambda i: str(i).rstrip('1')).name == 'x0'
    # 使用修改函数，断言 F('x1', Symbol('x1'), modify=lambda i: i + '_').name 为 'x1_'
    assert F('x1', Symbol('x1'),
             modify=lambda i: i + '_').name == 'x1_'
    # 断言 _symbol(x, _x) 等于 x


# 测试符号歧义消除功能
def test_disambiguate():
    # 创建多个符号变量
    x, y, y_1, _x, x_1, x_2 = symbols('x y y_1 _x x_1 x_2')
    # 创建多个 Dummy 对象组成的元组 t1
    t1 = Dummy('y'), _x, Dummy('x'), Dummy('x')
    # 创建多个 Dummy 对象组成的元组 t2
    t2 = Dummy('x'), Dummy('x')
    # 创建多个 Dummy 对象组成的元组 t3
    t3 = Dummy('x'), Dummy('y')
    # 创建多个符号变量组成的元组 t4
    t4 = x, Dummy('x')
    # 创建多个符号变量和符号对象组成的元组 t5
    t5 = Symbol('x', integer=True), x, Symbol('x_1')

    # 断言 disambiguate(*t1) 的结果与预期值相等
    assert disambiguate(*t1) == (y, x_2, x, x_1)
    # 断言 disambiguate(*t2) 的结果与预期值相等
    assert disambiguate(*t2) == (x, x_1)
    # 断言 disambiguate(*t3) 的结果与预期值相等
    assert disambiguate(*t3) == (x, y)
    # 断言 disambiguate(*t4) 的结果与预期值相等
    assert disambiguate(*t4) == (x_1, x)
    # 断言 disambiguate(*t5) 的第一个元素不等于 x
    assert disambiguate(*t5)[0] != x  # assumptions are retained

    # 创建多个 Dummy 对象组成的元组 t6
    t6 = _x, Dummy('x')/y
    # 创建多个 Dummy 对象组成的元组 t7
    t7 = y*Dummy('y'), y

    # 断言 disambiguate(*t6) 的结果与预期值相等
    assert disambiguate(*t6) == (x_1, x/y)
    # 断言 disambiguate(*t7) 的结果与预期值相等
    assert disambiguate(*t7) == (y*y_1, y_1)
    # 断言 disambiguate(Dummy('x_1'), Dummy('x_1')) 的结果为 (x_1, Symbol('x_1_1'))


@skip_under_pyodide("Cannot create threads under pyodide.")
def test_issue_gh_16734():
    # https://github.com/sympy/sympy/issues/16734

    # 创建一个符号列表 syms
    syms = list(symbols('x, y'))

    # 定义线程 thread1
    def thread1():
        # 循环创建符号 x{n}, y{n} 并赋给 syms[0], syms[1]
        for n in range(1000):
            syms[0], syms[1] = symbols(f'x{n}, y{n}')
            # 检查当前线程中的符号是否为正数
            syms[0].is_positive  # Check an assumption in this thread.
        # 将 syms[0] 置为 None
        syms[0] = None

    # 定义线程 thread2
    def thread2():
        # 当 syms[0] 不为 None 时循环比较符号
        while syms[0] is not None:
            # 比较线程中的符号
            result = (syms[0] == syms[1])  # noqa

    # 创建线程对象 thread，目标函数为 thread1
    thread = threading.Thread(target=thread1)
    # 启动线程 thread
    thread.start()
    # 执行线程 thread2
    thread2()
    # 等待线程 thread 结束
    thread.join()
```