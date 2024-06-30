# `D:\src\scipysrc\sympy\sympy\unify\tests\test_sympy.py`

```
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.logic.boolalg import And
from sympy.core.symbol import Str
from sympy.unify.core import Compound, Variable
from sympy.unify.usympy import (deconstruct, construct, unify, is_associative,
        is_commutative)
from sympy.abc import x, y, z, n

# 定义测试函数 test_deconstruct，用于测试 deconstruct 函数
def test_deconstruct():
    # 创建一个 Basic 对象，包含 S(1), S(2), S(3) 作为参数
    expr     = Basic(S(1), S(2), S(3))
    # 创建一个预期的 Compound 对象，包含 Basic 类型和 (1, 2, 3) 作为参数
    expected = Compound(Basic, (1, 2, 3))
    # 断言调用 deconstruct 函数返回值等于预期值 expected
    assert deconstruct(expr) == expected

    # 断言调用 deconstruct 函数处理整数 1 返回 1
    assert deconstruct(1) == 1
    # 断言调用 deconstruct 函数处理符号 x 返回 x
    assert deconstruct(x) == x
    # 断言调用 deconstruct 函数指定变量为 (x,) 处理符号 x 返回 Variable(x)
    assert deconstruct(x, variables=(x,)) == Variable(x)
    # 断言调用 deconstruct 函数处理 Add(1, x, evaluate=False) 返回 Compound(Add, (1, x))
    assert deconstruct(Add(1, x, evaluate=False)) == Compound(Add, (1, x))
    # 断言调用 deconstruct 函数指定变量为 (x,) 处理 Add(1, x, evaluate=False) 返回 Compound(Add, (1, Variable(x)))
    assert deconstruct(Add(1, x, evaluate=False), variables=(x,)) == \
              Compound(Add, (1, Variable(x)))

# 定义测试函数 test_construct，用于测试 construct 函数
def test_construct():
    # 创建一个 Compound 对象，包含 Basic 类型和 (S(1), S(2), S(3)) 作为参数
    expr     = Compound(Basic, (S(1), S(2), S(3)))
    # 创建一个预期的 Basic 对象，包含 S(1), S(2), S(3) 作为参数
    expected = Basic(S(1), S(2), S(3))
    # 断言调用 construct 函数返回值等于预期值 expected
    assert construct(expr) == expected

# 定义测试函数 test_nested，测试嵌套结构的处理
def test_nested():
    # 创建一个 Basic 对象，包含 S(1), Basic(S(2)), S(3) 作为参数
    expr = Basic(S(1), Basic(S(2)), S(3))
    # 创建一个预期的 Compound 对象，包含 Basic 类型和 (S(1), Compound(Basic, Tuple(2)), S(3)) 作为参数
    cmpd = Compound(Basic, (S(1), Compound(Basic, Tuple(2)), S(3)))
    # 断言调用 deconstruct 函数处理 expr 返回 cmpd
    assert deconstruct(expr) == cmpd
    # 断言调用 construct 函数处理 cmpd 返回 expr
    assert construct(cmpd) == expr

# 定义测试函数 test_unify，测试 unify 函数
def test_unify():
    # 创建一个 Basic 对象，包含 S(1), S(2), S(3) 作为参数
    expr = Basic(S(1), S(2), S(3))
    # 创建符号 a, b, c
    a, b, c = map(Symbol, 'abc')
    # 创建一个 Basic 对象，包含 a, b, c 作为参数
    pattern = Basic(a, b, c)
    # 断言调用 unify 函数处理 expr 和 pattern 返回 [{a: 1, b: 2, c: 3}]
    assert list(unify(expr, pattern, {}, (a, b, c))) == [{a: 1, b: 2, c: 3}]
    # 断言调用 unify 函数处理 expr 和 pattern，指定变量为 (a, b, c) 返回 [{a: 1, b: 2, c: 3}]
    assert list(unify(expr, pattern, variables=(a, b, c))) == \
            [{a: 1, b: 2, c: 3}]

# 定义测试函数 test_unify_variables，测试带有变量的 unify 函数
def test_unify_variables():
    # 断言调用 unify 函数处理 Basic(S(1), S(2)) 和 Basic(S(1), x) 返回 [{x: 2}]
    assert list(unify(Basic(S(1), S(2)), Basic(S(1), x), {}, variables=(x,))) == [{x: 2}]

# 定义测试函数 test_s_input，测试处理带有符号的输入
def test_s_input():
    # 创建一个 Basic 对象，包含 S(1), S(2) 作为参数
    expr = Basic(S(1), S(2))
    # 创建符号 a, b
    a, b = map(Symbol, 'ab')
    # 创建一个 Basic 对象，包含 a, b 作为参数
    pattern = Basic(a, b)
    # 断言调用 unify 函数处理 expr 和 pattern 返回 [{a: 1, b: 2}]
    assert list(unify(expr, pattern, {}, (a, b))) == [{a: 1, b: 2}]
    # 断言调用 unify 函数处理 expr 和 pattern，指定 a = 5 返回空列表
    assert list(unify(expr, pattern, {a: 5}, (a, b))) == []

# 定义函数 iterdicteq，用于比较两个字典是否相等
def iterdicteq(a, b):
    a = tuple(a)
    b = tuple(b)
    return len(a) == len(b) and all(x in b for x in a)

# 定义测试函数 test_unify_commutative，测试处理可交换运算的 unify 函数
def test_unify_commutative():
    # 创建一个 Add 对象，包含 (1, 2, 3, evaluate=False) 作为参数
    expr = Add(1, 2, 3, evaluate=False)
    # 创建符号 a, b, c
    a, b, c = map(Symbol, 'abc')
    # 创建一个 Add 对象，包含 (a, b, c, evaluate=False) 作为参数
    pattern = Add(a, b, c, evaluate=False)

    # 调用 unify 函数处理 expr 和 pattern 返回的结果
    result  = tuple(unify(expr, pattern, {}, (a, b, c)))
    # 预期的结果集合，包含所有可能的组合
    expected = ({a: 1, b: 2, c: 3},
                {a: 1, b: 3, c: 2},
                {a: 2, b: 1, c: 3},
                {a: 2, b: 3, c: 1},
                {a: 3, b: 1, c: 2},
                {a: 3, b: 2, c: 1})

    # 断言结果集合和预期结果集合相等
    assert iterdicteq(result, expected)

# 定义测试函数 test_unify_iter，测试处理迭代的 unify 函数
def test_unify_iter():
    # 创建一个 Add 对象，包含 (1, 2, 3, evaluate=False) 作为参数
    expr = Add(1, 2, 3, evaluate=False)
    # 创建符号 a, b, c
    a, b, c = map(Symbol, 'abc')
    # 创建一个 Add 对象，包含 (a, c, evaluate=False) 作为参数
    pattern = Add(a, c, evaluate=False)
    # 断言调用 is_associative 函数处理 deconstruct(pattern) 返回 True
    assert is_associative(deconstruct(pattern))
    # 断言调用 is_commutative 函数处理 deconstruct(pattern) 返回 True
    assert is_commutative(deconstruct(pattern))

    # 调用 unify 函数处理 expr 和 pattern 返回的结果
    result   = list(unify(expr, pattern, {}, (a, c)))
    # 定义期望的结果列表，包含多个字典，每个字典表示一组预期的键值对
    expected = [{a: 1, c: Add(2, 3, evaluate=False)},
                {a: 1, c: Add(3, 2, evaluate=False)},
                {a: 2, c: Add(1, 3, evaluate=False)},
                {a: 2, c: Add(3, 1, evaluate=False)},
                {a: 3, c: Add(1, 2, evaluate=False)},
                {a: 3, c: Add(2, 1, evaluate=False)},
                {a: Add(1, 2, evaluate=False), c: 3},
                {a: Add(2, 1, evaluate=False), c: 3},
                {a: Add(1, 3, evaluate=False), c: 2},
                {a: Add(3, 1, evaluate=False), c: 2},
                {a: Add(2, 3, evaluate=False), c: 1},
                {a: Add(3, 2, evaluate=False), c: 1}]
    
    # 使用 assert 语句验证 result 是否与 expected 相等，调用 iterdicteq 函数
    assert iterdicteq(result, expected)
def test_hard_match():
    # 导入所需的三角函数
    from sympy.functions.elementary.trigonometric import (cos, sin)
    # 构造一个表达式
    expr = sin(x) + cos(x)**2
    # 将 'p' 和 'q' 映射为符号对象
    p, q = map(Symbol, 'pq')
    # 构造一个模式表达式
    pattern = sin(p) + cos(p)**2
    # 使用 unify 函数尝试统一 expr 和 pattern，期望得到的映射结果包含 {p: x}
    assert list(unify(expr, pattern, {}, (p, q))) == [{p: x}]

def test_matrix():
    # 导入矩阵符号
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    # 创建矩阵符号 X 和 Y
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', 2, 2)
    Z = MatrixSymbol('Z', 2, 3)
    # 使用 unify 函数尝试统一 X 和 Y，期望得到的映射结果包含 {n: 2, Str('X'): Str('Y')}
    assert list(unify(X, Y, {}, variables=[n, Str('X')])) == [{Str('X'): Str('Y'), n: 2}]
    # 使用 unify 函数尝试统一 X 和 Z，不期望得到任何映射结果
    assert list(unify(X, Z, {}, variables=[n, Str('X')])) == []

def test_non_frankenAdds():
    # 验证在没有引起 Basic.__new__ 失败的情况下，能够正常运行 is_commutative 属性和 str 调用
    expr = x+y*2
    # 通过 construct 和 deconstruct 函数重建表达式
    rebuilt = construct(deconstruct(expr))
    # 确保可以执行 str(rebuilt) 和 rebuilt.is_commutative 而不引发错误
    str(rebuilt)
    rebuilt.is_commutative

def test_FiniteSet_commutivity():
    # 导入有限集合
    from sympy.sets.sets import FiniteSet
    # 定义符号变量
    a, b, c, x, y = symbols('a,b,c,x,y')
    # 创建集合 s 和 t
    s = FiniteSet(a, b, c)
    t = FiniteSet(x, y)
    variables = (x, y)
    # 使用 unify 函数尝试统一 s 和 t，期望得到的映射结果包含 {x: FiniteSet(a, c), y: b}
    assert {x: FiniteSet(a, c), y: b} in tuple(unify(s, t, variables=variables))

def test_FiniteSet_complex():
    # 导入有限集合
    from sympy.sets.sets import FiniteSet
    # 定义符号变量
    a, b, c, x, y, z = symbols('a,b,c,x,y,z')
    # 构造复杂的表达式和模式
    expr = FiniteSet(Basic(S(1), x), y, Basic(x, z))
    pattern = FiniteSet(a, Basic(x, b))
    variables = a, b
    # 预期的映射结果
    expected = ({b: 1, a: FiniteSet(y, Basic(x, z))},
                {b: z, a: FiniteSet(y, Basic(S(1), x))})
    # 使用 unify 函数尝试统一 expr 和 pattern，期望得到 expected 中的映射结果
    assert iterdicteq(unify(expr, pattern, variables=variables), expected)

def test_and():
    # 定义符号变量
    variables = x, y
    # 预期的映射结果
    expected = ({x: z > 0, y: n < 3},)
    # 使用 unify 函数尝试统一 (z>0) & (n<3) 和 And(x, y)，期望得到 expected 中的映射结果
    assert iterdicteq(unify((z>0) & (n<3), And(x, y), variables=variables),
                      expected)

def test_Union():
    # 导入区间
    from sympy.sets.sets import Interval
    # 使用 unify 函数尝试统一 Interval(0, 1) + Interval(10, 11) 和 Interval(0, 1) + Interval(12, 13)
    # variables=(Interval(12, 13)) 指定变量
    assert list(unify(Interval(0, 1) + Interval(10, 11),
                      Interval(0, 1) + Interval(12, 13),
                      variables=(Interval(12, 13),)))

def test_is_commutative():
    # 验证是否可交换性
    assert is_commutative(deconstruct(x+y))
    assert is_commutative(deconstruct(x*y))
    assert not is_commutative(deconstruct(x**y))

def test_commutative_in_commutative():
    # 导入符号
    from sympy.abc import a,b,c,d
    # 导入三角函数
    from sympy.functions.elementary.trigonometric import (cos, sin)
    # 构造等式和模式
    eq = sin(3)*sin(4)*sin(5) + 4*cos(3)*cos(4)
    pat = a*cos(b)*cos(c) + d*sin(b)*sin(c)
    # 使用 unify 函数尝试统一 eq 和 pat，期望得到一个映射结果
    assert next(unify(eq, pat, variables=(a,b,c,d)))
```