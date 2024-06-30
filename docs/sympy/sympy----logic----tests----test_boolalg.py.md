# `D:\src\scipysrc\sympy\sympy\logic\tests\test_boolalg.py`

```
# 从 sympy 库中导入不同模块和函数

from sympy.assumptions.ask import Q
from sympy.assumptions.refine import refine
from sympy.core.numbers import oo
from sympy.core.relational import Equality, Eq, Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions import Piecewise
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.sets.sets import Interval, Union
from sympy.sets.contains import Contains
from sympy.simplify.simplify import simplify
from sympy.logic.boolalg import (
    And, Boolean, Equivalent, ITE, Implies, Nand, Nor, Not, Or,
    POSform, SOPform, Xor, Xnor, conjuncts, disjuncts,
    distribute_or_over_and, distribute_and_over_or,
    eliminate_implications, is_nnf, is_cnf, is_dnf, simplify_logic,
    to_nnf, to_cnf, to_dnf, to_int_repr, bool_map, true, false,
    BooleanAtom, is_literal, term_to_integer,
    truth_table, as_Boolean, to_anf, is_anf, distribute_xor_over_and,
    anf_coeffs, ANFform, bool_minterm, bool_maxterm, bool_monomial,
    _check_pair, _convert_to_varsSOP, _convert_to_varsPOS, Exclusive,
    gateinputcount)
from sympy.assumptions.cnf import CNF

from sympy.testing.pytest import raises, XFAIL, slow

# 导入 itertools 库中的函数和类
from itertools import combinations, permutations, product

# 定义符号变量
A, B, C, D = symbols('A:D')
a, b, c, d, e, w, x, y, z = symbols('a:e w:z')

# 定义测试函数 test_overloading
def test_overloading():
    """测试 | 和 & 是否按预期重载"""

    # 断言操作
    assert A & B == And(A, B)
    assert A | B == Or(A, B)
    assert (A & B) | C == Or(And(A, B), C)
    assert A >> B == Implies(A, B)
    assert A << B == Implies(B, A)
    assert ~A == Not(A)
    assert A ^ B == Xor(A, B)

# 定义测试函数 test_And
def test_And():
    """测试 And 函数的行为"""

    # 基本的 And 函数测试
    assert And() is true
    assert And(A) == A
    assert And(True) is true
    assert And(False) is false
    assert And(True, True) is true
    assert And(True, False) is false
    assert And(False, False) is false
    assert And(True, A) == A
    assert And(False, A) is false
    assert And(True, True, True) is true
    assert And(True, True, A) == A
    assert And(True, False, A) is false
    assert And(1, A) == A
    raises(TypeError, lambda: And(2, A))  # 引发 TypeError 异常
    assert And(A < 1, A >= 1) is false
    e = A > 1
    assert And(e, e.canonical) == e.canonical
    g, l, ge, le = A > B, B < A, A >= B, B <= A
    assert And(g, l, ge, le) == And(ge, g)
    assert {And(*i) for i in permutations((l,g,le,ge))} == {And(ge, g)}
    assert And(And(Eq(a, 0), Eq(b, 0)), And(Ne(a, 0), Eq(c, 0))) is false

# 定义测试函数 test_Or
def test_Or():
    """测试 Or 函数的行为"""

    # 基本的 Or 函数测试
    assert Or() is false
    assert Or(A) == A
    assert Or(True) is true
    assert Or(False) is false
    assert Or(True, True) is true
    assert Or(True, False) is true
    assert Or(False, False) is false
    assert Or(True, A) is true
    assert Or(False, A) == A
    assert Or(True, False, False) is true
    assert Or(True, False, A) is true
    assert Or(False, False, A) == A
    assert Or(1, A) is true
    raises(TypeError, lambda: Or(2, A))  # 引发 TypeError 异常
    assert Or(A < 1, A >= 1) is true
    e = A > 1
    assert Or(e, e.canonical) == e
    # 使用四个比较运算符（大于、小于、大于等于、小于等于）对 A 和 B 进行比较，并将结果分别赋值给变量 g, l, ge, le
    g, l, ge, le = A > B, B < A, A >= B, B <= A
    # 使用逻辑运算符 Or 对比较结果进行逻辑或运算，确保至少有一个比较结果为 True
    assert Or(g, l, ge, le) == Or(g, ge)
# 定义测试函数 test_Xor，用于测试 Xor 函数的各种情况
def test_Xor():
    # 断言 Xor() 返回 false
    assert Xor() is false
    # 断言 Xor(A) 返回 A
    assert Xor(A) == A
    # 断言 Xor(A, A) 返回 false
    assert Xor(A, A) is false
    # 断言 Xor(True, A, A) 返回 true
    assert Xor(True, A, A) is true
    # 断言 Xor(A, A, A, A, A) 返回 A
    assert Xor(A, A, A, A, A) == A
    # 断言 Xor(True, False, False, A, B) 返回 ~Xor(A, B)
    assert Xor(True, False, False, A, B) == ~Xor(A, B)
    # 断言 Xor(True) 返回 true
    assert Xor(True) is true
    # 断言 Xor(False) 返回 false
    assert Xor(False) is false
    # 断言 Xor(True, True) 返回 false
    assert Xor(True, True) is false
    # 断言 Xor(True, False) 返回 true
    assert Xor(True, False) is true
    # 断言 Xor(False, False) 返回 false
    assert Xor(False, False) is false
    # 断言 Xor(True, A) 返回 ~A
    assert Xor(True, A) == ~A
    # 断言 Xor(False, A) 返回 A
    assert Xor(False, A) == A
    # 断言 Xor(True, False, False) 返回 true
    assert Xor(True, False, False) is true
    # 断言 Xor(True, False, A) 返回 ~A
    assert Xor(True, False, A) == ~A
    # 断言 Xor(False, False, A) 返回 A
    assert Xor(False, False, A) == A
    # 断言 Xor(A, B) 返回的类型是 Xor 类型的实例
    assert isinstance(Xor(A, B), Xor)
    # 断言 Xor(A, B, Xor(C, D)) 返回 Xor(A, B, C, D)
    assert Xor(A, B, Xor(C, D)) == Xor(A, B, C, D)
    # 断言 Xor(A, B, Xor(B, C)) 返回 Xor(A, C)
    assert Xor(A, B, Xor(B, C)) == Xor(A, C)
    # 断言 Xor(A < 1, A >= 1, B) 返回 Xor(0, 1, B) == Xor(1, 0, B)
    assert Xor(A < 1, A >= 1, B) == Xor(0, 1, B) == Xor(1, 0, B)
    # 创建变量 e 为 A > 1
    e = A > 1
    # 断言 Xor(e, e.canonical) 返回 Xor(0, 0) == Xor(1, 1)
    assert Xor(e, e.canonical) == Xor(0, 0) == Xor(1, 1)


# 定义测试函数 test_rewrite_as_And，测试 rewrite 方法将表达式重写为 And 类型
def test_rewrite_as_And():
    # 创建表达式 expr = x ^ y
    expr = x ^ y
    # 断言 expr.rewrite(And) 返回 (x | y) & (~x | ~y)
    assert expr.rewrite(And) == (x | y) & (~x | ~y)


# 定义测试函数 test_rewrite_as_Or，测试 rewrite 方法将表达式重写为 Or 类型
def test_rewrite_as_Or():
    # 创建表达式 expr = x ^ y
    expr = x ^ y
    # 断言 expr.rewrite(Or) 返回 (x & ~y) | (y & ~x)
    assert expr.rewrite(Or) == (x & ~y) | (y & ~x)


# 定义测试函数 test_rewrite_as_Nand，测试 rewrite 方法将表达式重写为 Nand 类型
def test_rewrite_as_Nand():
    # 创建表达式 expr = (y & z) | (z & ~w)
    expr = (y & z) | (z & ~w)
    # 断言 expr.rewrite(Nand) 返回 ~(~(y & z) & ~(z & ~w))
    assert expr.rewrite(Nand) == ~(~(y & z) & ~(z & ~w))


# 定义测试函数 test_rewrite_as_Nor，测试 rewrite 方法将表达式重写为 Nor 类型
def test_rewrite_as_Nor():
    # 创建表达式 expr = z & (y | ~w)
    expr = z & (y | ~w)
    # 断言 expr.rewrite(Nor) 返回 ~(~z | ~(y | ~w))
    assert expr.rewrite(Nor) == ~(~z | ~(y | ~w))


# 定义测试函数 test_Not，测试 Not 函数的各种情况
def test_Not():
    # 断言 TypeError 异常被引发，lambda: Not(True, False)
    raises(TypeError, lambda: Not(True, False))
    # 断言 Not(True) 返回 false
    assert Not(True) is false
    # 断言 Not(False) 返回 true
    assert Not(False) is true
    # 断言 Not(0) 返回 true
    assert Not(0) is true
    # 断言 Not(1) 返回 false
    assert Not(1) is false
    # 断言 Not(2) 返回 false
    assert Not(2) is false


# 定义测试函数 test_Nand，测试 Nand 函数的各种情况
def test_Nand():
    # 断言 Nand() 返回 false
    assert Nand() is false
    # 断言 Nand(A) 返回 ~A
    assert Nand(A) == ~A
    # 断言 Nand(True) 返回 false
    assert Nand(True) is false
    # 断言 Nand(False) 返回 true
    assert Nand(False) is true
    # 断言 Nand(True, True) 返回 false
    assert Nand(True, True) is false
    # 断言 Nand(True, False) 返回 true
    assert Nand(True, False) is true
    # 断言 Nand(False, False) 返回 true
    assert Nand(False, False) is true
    # 断言 Nand(True, A) 返回 ~A
    assert Nand(True, A) == ~A
    # 断言 Nand(False, A) 返回 true
    assert Nand(False, A) is true
    # 断言 Nand(True, True, True) 返回 false
    assert Nand(True, True, True) is false
    # 断言 Nand(True, True, A) 返回 ~A
    assert Nand(True, True, A) == ~A
    # 断言 Nand(True, False, A) 返回 true
    assert Nand(True, False, A) is true


# 定义测试函数 test_Nor，测试 Nor 函数的各种情况
def test_Nor():
    # 断言 Nor() 返回 true
    assert Nor() is true
    # 断言 Nor(A) 返回 ~A
    assert Nor(A) == ~A
    # 断言 Nor(True) 返回 false
    assert Nor(True) is false
    # 断言 Nor(False) 返回 true
    assert Nor(False) is true
    # 断言 Nor(True, True) 返回 false
    assert Nor(True, True) is false
    # 断言 Nor(True, False) 返回 false
    assert Nor(True, False) is false
    # 断言 Nor(False, False) 返回 true
    assert Nor(False, False) is true
    # 断言 Nor(True, A) 返回 false
    assert Nor(True, A) is false
    # 断言 Nor(False, A) 返回 ~A
    assert Nor(False, A) == ~A
    # 断言 Nor(True, True, True) 返回 false
    assert Nor(True, True, True) is false
    # 断言 Nor(True, True, A) 返回 false
    assert Nor(True, True, A) is false
    # 断言 Nor(True, False, A) 返回 false
    assert Nor(True, False, A) is false


# 定义测试函数 test_Xnor，测试 Xnor 函数的各种情况
def test_Xnor():
    # 断言 Xnor() 返回 true
    assert Xnor() is true
    # 断言 Xnor(A) 返回 ~A
    assert Xnor(A) == ~A
    # 断言 Xnor(A, A) 返回 true
    assert
    # 断言语句，验证 Implies 函数的返回值
    
    # 断言: Implies(False, True) 的返回值应为 True
    assert Implies(False, True) is true
    
    # 断言: Implies(False, False) 的返回值应为 True
    assert Implies(False, False) is true
    
    # 断言: Implies(0, A) 的返回值应为 True
    assert Implies(0, A) is true
    
    # 断言: Implies(1, 1) 的返回值应为 True
    assert Implies(1, 1) is true
    
    # 断言: Implies(1, 0) 的返回值应为 False
    assert Implies(1, 0) is false
    
    # 断言: A >> B 与 B << A 的结果应该相等
    assert (A >> B) == (B << A)
    
    # 断言: (A < 1) >> (A >= 1) 的结果应该等于 (A >= 1)
    assert ((A < 1) >> (A >= 1)) == (A >= 1)
    
    # 断言: (A < 1) >> (S.One > A) 的返回值应为 True
    assert ((A < 1) >> (S.One > A)) is true
    
    # 断言: A >> A 的返回值应为 True
    assert (A >> A) is true
# 定义一个测试函数，用于验证 Equivalent 函数的各种断言
def test_Equivalent():
    # 断言 A 和 B 等价，且 B 和 A 等价，并且 A 和 B，A 三者等价
    assert Equivalent(A, B) == Equivalent(B, A) == Equivalent(A, B, A)
    # 断言没有输入参数时 Equivalent 函数返回 true
    assert Equivalent() is true
    # 断言 A 和 A 等价，且 A 和空参数等价，均返回 true
    assert Equivalent(A, A) == Equivalent(A) is true
    # 断言 True 和 True 等价，且 False 和 False 等价，均返回 true
    assert Equivalent(True, True) == Equivalent(False, False) is true
    # 断言 True 和 False 不等价，且 False 和 True 不等价，均返回 false
    assert Equivalent(True, False) == Equivalent(False, True) is false
    # 断言 A 和 True 等价，返回 A
    assert Equivalent(A, True) == A
    # 断言 A 和 False 等价，返回 Not(A)
    assert Equivalent(A, False) == Not(A)
    # 断言 A、B 和 True 等价，返回 A & B
    assert Equivalent(A, B, True) == A & B
    # 断言 A、B 和 False 等价，返回 ~A & ~B
    assert Equivalent(A, B, False) == ~A & ~B
    # 断言 1 和 A 等价，返回 A
    assert Equivalent(1, A) == A
    # 断言 0 和 A 等价，返回 Not(A)
    assert Equivalent(0, A) == Not(A)
    # 断言 A 和 Equivalent(B, C) 等价，与 Equivalent(Equivalent(A, B), C) 不等价
    assert Equivalent(A, Equivalent(B, C)) != Equivalent(Equivalent(A, B), C)
    # 断言 A < 1 和 A >= 1 不等价，返回 false
    assert Equivalent(A < 1, A >= 1) is false
    # 断言 A < 1、A >= 1、0 不等价，返回 false
    assert Equivalent(A < 1, A >= 1, 0) is false
    # 断言 A < 1、A >= 1、1 不等价，返回 false
    assert Equivalent(A < 1, A >= 1, 1) is false
    # 断言 A < 1 和 S.One > A 等价，且 1 和 1、0 和 0 均等价
    assert Equivalent(A < 1, S.One > A) == Equivalent(1, 1) == Equivalent(0, 0)
    # 断言 Equality(A, B) 和 Equality(B, A) 等价，返回 true
    assert Equivalent(Equality(A, B), Equality(B, A)) is true


# 定义一个测试函数，用于验证 Exclusive 函数的各种断言
def test_Exclusive():
    # 断言所有参数均为 False 时，返回 true
    assert Exclusive(False, False, False) is true
    # 断言只有一个参数为 True 时，返回 true
    assert Exclusive(True, False, False) is true
    # 断言有两个参数为 True 时，返回 false
    assert Exclusive(True, True, False) is false
    # 断言所有参数均为 True 时，返回 false
    assert Exclusive(True, True, True) is false


# 定义一个测试函数，用于验证 equals 方法的各种断言
def test_equals():
    # 断言 Not(Or(A, B)) 等价于 And(Not(A), Not(B))
    assert Not(Or(A, B)).equals(And(Not(A), Not(B))) is True
    # 断言 Equivalent(A, B) 等价于 (A >> B) & (B >> A)
    assert Equivalent(A, B).equals((A >> B) & (B >> A)) is True
    # 断言 ((A | ~B) & (~A | B)) 等价于 (~A & ~B) | (A & B)
    assert ((A | ~B) & (~A | B)).equals((~A & ~B) | (A & B)) is True
    # 断言 (A >> B) 不等价于 (~A >> ~B)
    assert (A >> B).equals(~A >> ~B) is False
    # 断言 (A >> (B >> A)) 不等价于 (A >> (C >> A))
    assert (A >> (B >> A)).equals(A >> (C >> A)) is False
    # 断言 (A & B) 的 equals 方法引发 NotImplementedError
    raises(NotImplementedError, lambda: (A & B).equals(A > B))


# 定义一个测试函数，用于验证 boolalg 模块中的简化方法
def test_simplification_boolalg():
    """
    Test working of simplification methods.
    """
    # 定义两组 minterms 和相应的 POS 和 SOP 表达式进行验证
    set1 = [[0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0]]
    set2 = [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]]
    # 断言 SOPform([x, y, z], set1) 等价于 Or(And(Not(x), z), And(Not(z), x))
    assert SOPform([x, y, z], set1) == Or(And(Not(x), z), And(Not(z), x))
    # 断言 Not(SOPform([x, y, z], set2)) 等价于 Not(Or(And(Not(x), Not(z)), And(x, z)))
    assert Not(SOPform([x, y, z], set2)) == \
        Not(Or(And(Not(x), Not(z)), And(x, z)))
    # 断言 POSform([x, y, z], set1 + set2) 返回 true
    assert POSform([x, y, z], set1 + set2) is true
    # 断言 SOPform([x, y, z], set1 + set2) 返回 true
    assert SOPform([x, y, z], set1 + set2) is true
    # 断言 SOPform([Dummy(), Dummy(), Dummy()], set1 + set2) 返回 true
    assert SOPform([Dummy(), Dummy(), Dummy()], set1 + set2) is true

    # 定义 minterms 和 dontcares 进行验证
    minterms = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]]
    dontcares = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1]]
    # 断言 SOPform([w, x, y, z], minterms, dontcares) 等价于 Or(And(y, z), And(Not(w), Not(x)))
    assert (
        SOPform([w, x, y, z], minterms, dontcares) ==
        Or(And(y, z), And(Not(w), Not(x)))
    )
    # 断言 POSform([w, x, y, z], minterms, dontcares) 等价于 And(Or(Not(w), y), z)
    assert POSform([w, x, y, z], minterms, dontcares) == And(Or(Not(w), y), z)

    # 定义 minterms 和 dontcares 进行验证
    minterms = [1, 3, 7, 11, 15]
    dontcares = [0, 2, 5]
    # 断言 SOPform([w, x, y, z], minterms, dontcares) 等价于 Or(And(y, z), And(Not(w), Not(x)))
    assert (
        SOPform([w, x, y, z], minterms, dontcares) ==
        Or(And(y, z), And(Not(w), Not(x)))
    )
    # 断言 POSform([w, x, y, z], minterms, dontcares) 等价于 And(Or(Not(w), y), z)
    assert POSform([w, x, y, z], minterms, dontcares
    # 定义两个变量 minterms 和 dontcares，分别用于测试 SOPform 和 POSform 函数
    minterms = [1, {y: 1, z: 1}]
    dontcares = [0, [0, 0, 1, 0], 5]

    # 断言测试 SOPform 函数的输出是否符合预期结果
    assert (
        SOPform([w, x, y, z], minterms, dontcares) ==
        Or(And(y, z), And(Not(w), Not(x))))
    
    # 断言测试 POSform 函数的输出是否符合预期结果
    assert POSform([w, x, y, z], minterms, dontcares) == And(Or(Not(w), y), z)


    # 更新 minterms 和 dontcares 变量的值，用于进一步测试 SOPform 和 POSform 函数
    minterms = [{y: 1, z: 1}, 1]
    dontcares = [[0, 0, 0, 0]]

    # 重新定义 minterms 变量，用于测试 ValueError 是否被正确引发
    minterms = [[0, 0, 0]]
    raises(ValueError, lambda: SOPform([w, x, y, z], minterms))
    raises(ValueError, lambda: POSform([w, x, y, z], minterms))

    # 断言测试 TypeError 是否被正确引发，测试 POSform 函数传入错误的参数类型
    raises(TypeError, lambda: POSform([w, x, y, z], ["abcdefg"]))

    # 测试简化逻辑表达式的功能
    ans = And(A, Or(B, C))
    assert simplify_logic(A & (B | C)) == ans
    assert simplify_logic((A & B) | (A & C)) == ans
    assert simplify_logic(Implies(A, B)) == Or(Not(A), B)
    assert simplify_logic(Equivalent(A, B)) == \
        Or(And(A, B), And(Not(A), Not(B)))
    assert simplify_logic(And(Equality(A, 2), C)) == And(Equality(A, 2), C)
    assert simplify_logic(And(Equality(A, 2), A)) == And(Equality(A, 2), A)
    assert simplify_logic(And(Equality(A, B), C)) == And(Equality(A, B), C)
    assert simplify_logic(Or(And(Equality(A, 3), B), And(Equality(A, 3), C))) \
        == And(Equality(A, 3), Or(B, C))
    
    # 定义逻辑表达式 b 和 e，并断言其简化后的结果
    b = (~x & ~y & ~z) | (~x & ~y & z)
    e = And(A, b)
    assert simplify_logic(e) == A & ~x & ~y
    
    # 断言测试 ValueError 是否被正确引发，测试 simplify_logic 函数传入无效的 form 参数
    raises(ValueError, lambda: simplify_logic(A & (B | C), form='blabla'))
    
    # 断言测试 simplify 函数的功能
    assert simplify(Or(x <= y, And(x < y, z))) == (x <= y)
    assert simplify(Or(x <= y, And(y > x, z))) == (x <= y)
    assert simplify(Or(x >= y, And(y < x, z))) == (x >= y)

    # 检查含有九个变量或更多的表达式是否不会被简化（未使用 force-flag）
    a, b, c, d, e, f, g, h, j = symbols('a b c d e f g h j')
    expr = a & b & c & d & e & f & g & h & j | \
        a & b & c & d & e & f & g & h & ~j
    # 断言测试表达式是否成功简化
    assert simplify_logic(expr) == expr

    # 测试带有 dontcare 参数的 simplify_logic 函数
    assert simplify_logic((a & b) | c | d, dontcare=(a & b)) == c | d

    # 检查输入的正确性
    ans = SOPform([x, y], [[1, 0]])
    assert SOPform([x, y], [[1, 0]]) == ans
    assert POSform([x, y], [[1, 0]]) == ans

    # 测试 ValueError 是否被正确引发，测试 SOPform 函数传入无效的 minterms 和 dontcares 参数
    raises(ValueError, lambda: SOPform([x], [[1]], [[1]]))
    assert SOPform([x], [[1]], [[0]]) is true
    assert SOPform([x], [[0]], [[1]]) is true
    assert SOPform([x], [], []) is false

    # 测试 ValueError 是否被正确引发，测试 POSform 函数传入无效的 minterms 和 dontcares 参数
    raises(ValueError, lambda: POSform([x], [[1]], [[1]]))
    assert POSform([x], [[1]], [[0]]) is true
    assert POSform([x], [[0]], [[1]]) is true
    assert POSform([x], [], []) is false

    # 检查 simplify 函数的工作情况
    assert simplify((A & B) | (A & C)) == And(A, Or(B, C))
    assert simplify(And(x, Not(x))) == False
    assert simplify(Or(x, Not(x))) == True
    assert simplify(And(Eq(x, 0), Eq(x, y))) == And(Eq(x, 0), Eq(y, 0))
    assert And(Eq(x - 1, 0), Eq(x, y)).simplify() == And(Eq(x, 1), Eq(y, 1))
    assert And(Ne(x - 1, 0), Ne(x, y)).simplify() == And(Ne(x, 1), Ne(x, y))
    # 断言：确保表达式 And(Eq(x - 1, 0), Ne(x, y)) 简化后等于 And(Eq(x, 1), Ne(y, 1))
    assert And(Eq(x - 1, 0), Ne(x, y)).simplify() == And(Eq(x, 1), Ne(y, 1))
    
    # 断言：确保表达式 And(Eq(x - 1, 0), Eq(x, z + y), Eq(y + x, 0)) 简化后等于 And(Eq(x, 1), Eq(y, -1), Eq(z, 2))
    assert And(Eq(x - 1, 0), Eq(x, z + y), Eq(y + x, 0)).simplify() == And(Eq(x, 1), Eq(y, -1), Eq(z, 2))
    
    # 断言：确保表达式 And(Eq(x - 1, 0), Eq(x + 2, 3)) 简化后等于 Eq(x, 1)
    assert And(Eq(x - 1, 0), Eq(x + 2, 3)).simplify() == Eq(x, 1)
    
    # 断言：确保表达式 And(Ne(x - 1, 0), Ne(x + 2, 3)) 简化后等于 Ne(x, 1)
    assert And(Ne(x - 1, 0), Ne(x + 2, 3)).simplify() == Ne(x, 1)
    
    # 断言：确保表达式 And(Eq(x - 1, 0), Eq(x + 2, 2)) 简化后等于 False
    assert And(Eq(x - 1, 0), Eq(x + 2, 2)).simplify() == False
    
    # 断言：确保表达式 And(Ne(x - 1, 0), Ne(x + 2, 2)) 简化后等于 And(Ne(x, 1), Ne(x, 0))
    assert And(Ne(x - 1, 0), Ne(x + 2, 2)).simplify() == And(Ne(x, 1), Ne(x, 0))
    
    # 断言：确保表达式 simplify(Xor(x, ~x)) 简化后等于 True
    assert simplify(Xor(x, ~x)) == True
# 定义测试布尔映射功能的函数
def test_bool_map():
    """
    Test working of bool_map function.
    """

    # 布尔表达式变量和对应的真值映射
    minterms = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 0, 1, 1],
                [1, 1, 1, 1]]
    # 检查布尔映射函数的返回结果是否符合预期
    assert bool_map(Not(Not(a)), a) == (a, {a: a})
    # 检查布尔表达式转换为标准析取范式和合取范式后的结果是否符合预期
    assert bool_map(SOPform([w, x, y, z], minterms),
                    POSform([w, x, y, z], minterms)) == \
        (And(Or(Not(w), y), Or(Not(x), y), z), {x: x, w: w, z: z, y: y})
    # 检查不同布尔表达式转换为标准析取范式后是否不相等
    assert bool_map(SOPform([x, z, y], [[1, 0, 1]]),
                    SOPform([a, b, c], [[1, 0, 1]])) != False
    # 创建两个布尔函数并检查其映射结果是否符合预期
    function1 = SOPform([x, z, y], [[1, 0, 1], [0, 0, 1]])
    function2 = SOPform([a, b, c], [[1, 0, 1], [1, 0, 0]])
    assert bool_map(function1, function2) == \
        (function1, {y: a, z: b})
    # 检查异或运算和非运算结果是否为假
    assert bool_map(Xor(x, y), ~Xor(x, y)) == False
    # 检查与运算和或运算结果是否为无效
    assert bool_map(And(x, y), Or(x, y)) is None
    # issue 16179
    # 检查异或运算和非运算结果是否为假
    assert bool_map(Xor(x, y, z), ~Xor(x, y, z)) == False
    # 检查异或运算和非运算结果是否为假
    assert bool_map(Xor(a, x, y, z), ~Xor(a, x, y, z)) == False


# 定义测试混合符号和布尔值的函数
def test_bool_symbol():
    """Test that mixing symbols with boolean values
    works as expected"""

    # 检查与运算的结果是否为第一个符号
    assert And(A, True) == A
    # 检查与运算的结果是否为第一个符号
    assert And(A, True, True) == A
    # 检查与运算的结果是否为假
    assert And(A, False) is false
    # 检查与运算的结果是否为假
    assert And(A, True, False) is false
    # 检查或运算的结果是否为真
    assert Or(A, True) is true
    # 检查或运算的结果是否为第一个符号
    assert Or(A, False) == A


# 定义测试布尔类型的函数
def test_is_boolean():
    # 检查True是否是布尔类型的实例
    assert isinstance(True, Boolean) is False
    # 检查true是否是布尔类型的实例
    assert isinstance(true, Boolean) is True
    # 检查1是否等于True
    assert 1 == True
    # 检查1是否不等于true
    assert 1 != true
    # 检查1是否等于true
    assert (1 == true) is False
    # 检查0是否等于False
    assert 0 == False
    # 检查0是否不等于false
    assert 0 != false
    # 检查0是否等于false
    assert (0 == false) is False
    # 检查true是否为布尔类型的实例
    assert true.is_Boolean is True
    # 检查与运算和或运算的结果是否为布尔类型的实例
    assert (A & B).is_Boolean
    assert (A | B).is_Boolean
    # 检查非运算的结果是否为布尔类型的实例
    assert (~A).is_Boolean
    # 检查与运算和或运算的结果是否与是否是布尔类型的实例不相等
    assert (A ^ B).is_Boolean
    assert A.is_Boolean != isinstance(A, Boolean)
    # 检查A是否是布尔类型的实例
    assert isinstance(A, Boolean)


# 定义测试布尔表达式的替换函数
def test_subs():
    # 检查替换A为True后的与运算结果是否为B
    assert (A & B).subs(A, True) == B
    # 检查替换A为False后的与运算结果是否为假
    assert (A & B).subs(A, False) is false
    # 检查替换B为True后的与运算结果是否为A
    assert (A & B).subs(B, True) == A
    # 检查替换B为False后的与运算结果是否为假
    assert (A & B).subs(B, False) is false
    # 检查同时替换A为True和B为True后的与运算结果是否为真
    assert (A & B).subs({A: True, B: True}) is true
    # 检查替换A为True后的或运算结果是否为真
    assert (A | B).subs(A, True) is true
    # 检查替换A为False后的或运算结果是否为B
    assert (A | B).subs(A, False) == B
    # 检查替换B为True后的或运算结果是否为真
    assert (A | B).subs(B, True) is true
    # 检查替换B为False后的或运算结果是否为A
    assert (A | B).subs(B, False) == A
    # 检查同时替换A为True和B为True后的或运算结果是否为真
    assert (A | B).subs({A: True, B: True}) is true


"""
we test for axioms of boolean algebra
see https://en.wikipedia.org/wiki/Boolean_algebra_(structure)
"""


# 定义测试布尔代数定理的函数
def test_commutative():
    """Test for commutativity of And and Or"""
    # 将符号'A'和'B'映射为布尔类型
    A, B = map(Boolean, symbols('A,B'))

    # 检查与运算的交换律是否成立
    assert A & B == B & A
    # 检查或运算的交换律是否成立
    assert A | B == B | A


# 定义测试与运算的结合性的函数
def test_and_associativity():
    """Test for associativity of And"""

    # 检查与运算的结合性是否成立
    assert (A & B) & C == A & (B & C)


# 定义测试或运算的结合性的函数
def test_or_assicativity():
    assert ((A | B) | C) == (A | (B | C))


# 定义测试双重否定定理的函数
def test_double_negation():
    a = Boolean()
    # 检查双重否定定理是否成立
    assert ~(~a) == a


# 测试方法

# 定义测试消除含蕴含式的函数
def test_eliminate_implications():
    # 检查消除含蕴含式后的结果是否正确
    assert eliminate_implications(Implies(A, B, evaluate=False)) == (~A) | B
    # 使用断言验证消除蕴含后的表达式是否等于预期的结果
    assert eliminate_implications(
        A >> (C >> Not(B))) == Or(Or(Not(B), Not(C)), Not(A))
    
    # 使用断言验证消除等价关系后的表达式是否等于预期的结果
    assert eliminate_implications(Equivalent(A, B, C, D)) == \
        (~A | B) & (~B | C) & (~C | D) & (~D | A)
def test_conjuncts():
    # 测试函数：检查 conjuncts 函数对逻辑与操作的处理
    assert conjuncts(A & B & C) == {A, B, C}
    # 断言：A、B、C 的逻辑与应该返回集合 {A, B, C}
    assert conjuncts((A | B) & C) == {A | B, C}
    # 断言：(A 或 B) 的逻辑与 C 应该返回集合 {(A 或 B), C}
    assert conjuncts(A) == {A}
    # 断言：单个符号 A 应该返回集合 {A}
    assert conjuncts(True) == {True}
    # 断言：True 应该返回集合 {True}
    assert conjuncts(False) == {False}
    # 断言：False 应该返回集合 {False}


def test_disjuncts():
    # 测试函数：检查 disjuncts 函数对逻辑或操作的处理
    assert disjuncts(A | B | C) == {A, B, C}
    # 断言：A、B、C 的逻辑或应该返回集合 {A, B, C}
    assert disjuncts((A | B) & C) == {(A | B) & C}
    # 断言：(A 或 B) 的逻辑与 C 应该返回集合 {(A 或 B) 与 C}
    assert disjuncts(A) == {A}
    # 断言：单个符号 A 应该返回集合 {A}
    assert disjuncts(True) == {True}
    # 断言：True 应该返回集合 {True}
    assert disjuncts(False) == {False}
    # 断言：False 应该返回集合 {False}


def test_distribute():
    # 测试函数：检查 distribute_and_over_or 函数、distribute_or_over_and 函数和 distribute_xor_over_and 函数的处理
    assert distribute_and_over_or(Or(And(A, B), C)) == And(Or(A, C), Or(B, C))
    # 断言：And(A, B) 的逻辑或 C 应该分配为 And(Or(A, C), Or(B, C))
    assert distribute_or_over_and(And(A, Or(B, C))) == Or(And(A, B), And(A, C))
    # 断言：And(A, (B 或 C)) 应该分配为 Or(And(A, B), And(A, C))
    assert distribute_xor_over_and(And(A, Xor(B, C))) == Xor(And(A, B), And(A, C))
    # 断言：And(A, (B 异或 C)) 应该分配为 Xor(And(A, B), And(A, C))


def test_to_anf():
    x, y, z = symbols('x,y,z')
    assert to_anf(And(x, y)) == And(x, y)
    # 断言：And(x, y) 应该返回 And(x, y)
    assert to_anf(Or(x, y)) == Xor(x, y, And(x, y))
    # 断言：Or(x, y) 应该返回 Xor(x, y, And(x, y))
    assert to_anf(Or(Implies(x, y), And(x, y), y)) == Xor(x, True, x & y, remove_true=False)
    # 断言：Or(Implies(x, y), And(x, y), y) 应该返回 Xor(x, True, x & y, remove_true=False)
    assert to_anf(Or(Nand(x, y), Nor(x, y), Xnor(x, y), Implies(x, y))) == True
    # 断言：Or(Nand(x, y), Nor(x, y), Xnor(x, y), Implies(x, y)) 应该返回 True
    assert to_anf(Or(x, Not(y), Nor(x,z), And(x, y), Nand(y, z))) == Xor(True, And(y, z), And(x, y, z), remove_true=False)
    # 断言：Or(x, Not(y), Nor(x,z), And(x, y), Nand(y, z)) 应该返回 Xor(True, And(y, z), And(x, y, z), remove_true=False)
    assert to_anf(Xor(x, y)) == Xor(x, y)
    # 断言：Xor(x, y) 应该返回 Xor(x, y)
    assert to_anf(Not(x)) == Xor(x, True, remove_true=False)
    # 断言：Not(x) 应该返回 Xor(x, True, remove_true=False)
    assert to_anf(Nand(x, y)) == Xor(True, And(x, y), remove_true=False)
    # 断言：Nand(x, y) 应该返回 Xor(True, And(x, y), remove_true=False)
    assert to_anf(Nor(x, y)) == Xor(x, y, True, And(x, y), remove_true=False)
    # 断言：Nor(x, y) 应该返回 Xor(x, y, True, And(x, y), remove_true=False)
    assert to_anf(Implies(x, y)) == Xor(x, True, And(x, y), remove_true=False)
    # 断言：Implies(x, y) 应该返回 Xor(x, True, And(x, y), remove_true=False)
    assert to_anf(Equivalent(x, y)) == Xor(x, y, True, remove_true=False)
    # 断言：Equivalent(x, y) 应该返回 Xor(x, y, True, remove_true=False)
    assert to_anf(Nand(x | y, x >> y), deep=False) == Xor(True, And(Or(x, y), Implies(x, y)), remove_true=False)
    # 断言：Nand(x | y, x >> y) 应该返回 Xor(True, And(Or(x, y), Implies(x, y)), remove_true=False)
    assert to_anf(Nor(x ^ y, x & y), deep=False) == Xor(True, Or(Xor(x, y), And(x, y)), remove_true=False)
    # 断言：Nor(x ^ y, x & y) 应该返回 Xor(True, Or(Xor(x, y), And(x, y)), remove_true=False)
    # issue 25218
    assert to_anf(x ^ ~(x ^ y ^ ~y)) == False
    # 断言：x ^ ~(x ^ y ^ ~y) 应该返回 False


def test_to_nnf():
    assert to_nnf(true) is true
    # 断言：true 转换为 NNF 应该仍然是 true
    assert to_nnf(false) is false
    # 断言：false 转换为 NNF 应该仍然是 false
    assert to_nnf(A) == A
    # 断言：符号 A 转换为 NNF 应该仍然是 A
    assert to_nnf(A | ~A | B) is true
    # 断言：A 或 非A 或 B 转换为 NNF 应该是 true
    assert to_nnf(A & ~A & B) is false
    # 断言：A 与 非A 与 B 转换为 NNF 应该是 false
    assert to_nnf(A >> B) == ~A | B
    # 断言：A >> B 转换为 NNF 应该是 非A 或 B
    assert to_nnf(Equivalent(A, B, C)) == (~A | B) & (~B | C) & (~C | A)
    # 断言：等价(A, B, C) 转换为 NNF 应该是 (~A | B) 与 (~B | C) 与 (~C | A)
    assert to_nnf(A ^ B ^ C) == (A | B | C) & (~A | ~B | C) & (A | ~B | ~C) & (~A | B | ~C)
    # 断言：异或(A, B, C) 转换为 NNF 应该是 (A 或 B 或 C) 与 (非A 或 非B 或 C) 与 (A 或 非B 或 非C) 与 (非A 或 B 或 非C)
    assert to_nnf(ITE(A, B, C)) == (~A | B) & (A | C)
    # 断言：ITE(A, B, C) 转换为 NNF 应该是 (~A | B) 与 (A | C)
    assert to_nnf(Not(A | B | C)) == ~A & ~B & ~C
    # 断言：非(A
    # 尽管 ITE 可以容纳非布尔值，但如果试图将 ITE 转换为布尔值的 nnf，它会报错
    raises(TypeError, lambda: ITE(A < 1, [1], B).to_nnf())
def test_to_cnf():
    # 测试取反(~)后对或操作(B | C)应得到 And(Not(B), Not(C))
    assert to_cnf(~(B | C)) == And(Not(B), Not(C))
    # 测试对与操作(A & B)后对或操作(C)应得到 And(Or(A, C), Or(B, C))
    assert to_cnf((A & B) | C) == And(Or(A, C), Or(B, C))
    # 测试对蕴含操作(A >> B)应得到 (~A) | B
    assert to_cnf(A >> B) == (~A) | B
    # 测试对蕴含操作(A >> (B & C))应得到 (~A | B) & (~A | C)
    assert to_cnf(A >> (B & C)) == (~A | B) & (~A | C)
    # 测试使用参数simplify=True，对复杂表达式(A & (B | C) | ~A & (B | C))应得到简化结果B | C
    assert to_cnf(A & (B | C) | ~A & (B | C), True) == B | C
    # 测试对与操作(A & B)应得到 And(A, B)
    assert to_cnf(A & B) == And(A, B)

    # 测试等价操作(Equivalent(A, B))应得到 And(Or(A, Not(B)), Or(B, Not(A)))
    assert to_cnf(Equivalent(A, B)) == And(Or(A, Not(B)), Or(B, Not(A)))
    # 测试等价操作(Equivalent(A, B & C))应得到 (~A | B) & (~A | C) & (~B | ~C | A)
    assert to_cnf(Equivalent(A, B & C)) == (~A | B) & (~A | C) & (~B | ~C | A)
    # 测试使用参数simplify=True，对复杂等价操作(Equivalent(A, B | C))应得到简化结果And(Or(Not(B), A), Or(Not(C), A), Or(B, C, Not(A)))
    assert to_cnf(Equivalent(A, B | C), True) == And(Or(Not(B), A), Or(Not(C), A), Or(B, C, Not(A)))
    # 测试对(A + 1)应保持不变，不受函数to_cnf影响
    assert to_cnf(A + 1) == A + 1


def test_issue_18904():
    # 定义布尔变量
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15 = symbols('x1:16')
    # 创建表达式eq，包含多个与操作和或操作的组合
    eq = (( x1 & x2 & x3 & x4 & x5 & x6 & x7 & x8 & x9 )  |
        ( x1 & x2 & x3 & x4 & x5 & x6 & x7 & x10 & x9 )  |
        ( x1 & x11 & x3 & x12 & x5 & x13 & x14 & x15 & x9 ))
    # 验证to_cnf处理后的结果是否是CNF形式
    assert is_cnf(to_cnf(eq))
    # 使用simplify=True时，验证to_cnf是否会引发ValueError异常
    raises(ValueError, lambda: to_cnf(eq, simplify=True))
    # 对And和Or函数组合表达式进行测试，验证to_cnf处理后的结果是否与原始表达式相同
    for f, t in zip((And, Or), (to_cnf, to_dnf)):
        eq = f(x1, x2, x3, x4, x5, x6, x7, x8, x9)
        raises(ValueError, lambda: to_cnf(eq, simplify=True))
        assert t(eq, simplify=True, force=True) == eq


def test_issue_9949():
    # 验证to_cnf对复合逻辑表达式的处理结果是否正确
    assert is_cnf(to_cnf((b > -5) | (a > 2) & (a < 4)))


def test_to_CNF():
    # 验证CNF.CNF_to_cnf处理CNF.to_CNF(~(B | C))得到的结果是否与to_cnf(~(B | C))相同
    assert CNF.CNF_to_cnf(CNF.to_CNF(~(B | C))) == to_cnf(~(B | C))
    # 验证CNF.CNF_to_cnf处理CNF.to_CNF((A & B) | C)得到的结果是否与to_cnf((A & B) | C)相同
    assert CNF.CNF_to_cnf(CNF.to_CNF((A & B) | C)) == to_cnf((A & B) | C)
    # 验证CNF.CNF_to_cnf处理CNF.to_CNF(A >> B)得到的结果是否与to_cnf(A >> B)相同
    assert CNF.CNF_to_cnf(CNF.to_CNF(A >> B)) == to_cnf(A >> B)
    # 验证CNF.CNF_to_cnf处理CNF.to_CNF(A >> (B & C))得到的结果是否与to_cnf(A >> (B & C))相同
    assert CNF.CNF_to_cnf(CNF.to_CNF(A >> (B & C))) == to_cnf(A >> (B & C))
    # 验证CNF.CNF_to_cnf处理CNF.to_CNF(A & (B | C) | ~A & (B | C))得到的结果是否与to_cnf(A & (B | C) | ~A & (B | C))相同
    assert CNF.CNF_to_cnf(CNF.to_CNF(A & (B | C) | ~A & (B | C))) == to_cnf(A & (B | C) | ~A & (B | C))
    # 验证CNF.CNF_to_cnf处理CNF.to_CNF(A & B)得到的结果是否与to_cnf(A & B)相同
    assert CNF.CNF_to_cnf(CNF.to_CNF(A & B)) == to_cnf(A & B)


def test_to_dnf():
    # 测试取反(~)后对或操作(B | C)应得到 And(Not(B), Not(C))
    assert to_dnf(~(B | C)) == And(Not(B), Not(C))
    # 测试对与操作(A & (B | C))应得到 Or(And(A, B), And(A, C))
    assert to_dnf(A & (B | C)) == Or(And(A, B), And(A, C))
    # 测试对蕴含操作(A >> B)应得到 (~A) | B
    assert to_dnf(A >> B) == (~A) | B
    # 测试对蕴含操作(A >> (B & C))应得到 (~A) | (B & C)
    assert to_dnf(A >> (B & C)) == (~A) | (B & C)
    # 测试对或操作(A | B)应保持不变
    assert to_dnf(A | B) == A | B

    # 测试等价操作(Equivalent(A, B))应得到 Or(And(A, B), And(Not(A), Not(B)))
    assert to_dnf(Equivalent(A, B), True) == Or(And(A, B), And(Not(A), Not(B)))
    # 测试等价操作(Equivalent(A, B & C))应得到 Or(And(A, B, C), And(Not(A), Not(B)), And(Not(A), Not(C)))
    assert to_dnf(Equivalent(A, B & C), True) == Or(And(A, B, C), And(Not(A), Not(B)), And(Not(A), Not(C)))
    # 测试对(A + 1)应保持不变，不受函数to_dnf影响
    assert to_dnf(A + 1) == A + 1


def test_to_int_repr():
    # 定义布尔变量
    x, y, z = map(Boolean, symbols('x,y,z'))

    # 定义用于递归排序的函数sorted_recursive
    def sorted_recursive(arg):
        try:
            return sorted(sorted_recursive(x) for x in arg)
        except TypeError:  # 如果arg不是序列，则返回arg本身
            return arg

    # 验证to_int_repr的处理结果是否与预期相符
    assert sorted_recursive(to_int_repr([x | y, z | x], [x, y, z])) == \
        sorted_recursive([[1, 2], [1, 3]])
    assert
    # 断言：验证给定表达式 is_anf(Xor(x, y, Or(x, y))) 返回 False
    assert is_anf(Xor(x, y, Or(x, y))) is False
    # 断言：验证给定表达式 is_anf(Xor(Not(x), y)) 返回 False
    assert is_anf(Xor(Not(x), y)) is False
# 测试是否为否定标准形（NNF）
def test_is_nnf():
    # 检查常量true是否为NNF
    assert is_nnf(true) is True
    # 检查单个变量A是否为NNF
    assert is_nnf(A) is True
    # 检查否定单个变量~A是否为NNF
    assert is_nnf(~A) is True
    # 检查合取A & B是否为NNF
    assert is_nnf(A & B) is True
    # 检查包含多个合取和否定的复杂表达式是否为NNF
    assert is_nnf((A & B) | (~A & A) | (~B & B) | (~A & ~B), False) is True
    # 检查析取(A | B) & (~A | ~B)是否为NNF
    assert is_nnf((A | B) & (~A | ~B)) is True
    # 检查含有Not和Or的表达式是否为NNF
    assert is_nnf(Not(Or(A, B))) is False
    # 检查异或A ^ B是否为NNF
    assert is_nnf(A ^ B) is False
    # 检查包含多个合取和否定的复杂表达式是否为NNF
    assert is_nnf((A & B) | (~A & A) | (~B & B) | (~A & ~B), True) is False


# 测试是否为合取标准形（CNF）
def test_is_cnf():
    # 检查单个变量x是否为CNF
    assert is_cnf(x) is True
    # 检查合取x | y | z是否为CNF
    assert is_cnf(x | y | z) is True
    # 检查合取x & y & z是否为CNF
    assert is_cnf(x & y & z) is True
    # 检查复合表达式(x | y) & z是否为CNF
    assert is_cnf((x | y) & z) is True
    # 检查析取(x & y) | z是否为CNF
    assert is_cnf((x & y) | z) is False
    # 检查否定(~(x & y)) | z是否为CNF
    assert is_cnf(~(x & y) | z) is False


# 测试是否为析取标准形（DNF）
def test_is_dnf():
    # 检查单个变量x是否为DNF
    assert is_dnf(x) is True
    # 检查析取x | y | z是否为DNF
    assert is_dnf(x | y | z) is True
    # 检查析取x & y & z是否为DNF
    assert is_dnf(x & y & z) is True
    # 检查复合表达式(x & y) | z是否为DNF
    assert is_dnf((x & y) | z) is True
    # 检查合取(x | y) & z是否为DNF
    assert is_dnf((x | y) & z) is False
    # 检查否定(~(x | y)) & z是否为DNF
    assert is_dnf(~(x | y) & z) is False


# 测试条件表达式ITE（if-then-else）
def test_ITE():
    # 定义符号变量A, B, C
    A, B, C = symbols('A:C')
    # 检查条件为True时的ITE运算
    assert ITE(True, False, True) is false
    assert ITE(True, True, False) is true
    assert ITE(False, True, False) is false
    assert ITE(False, False, True) is true
    # 检查ITE返回值类型是否为ITE对象
    assert isinstance(ITE(A, B, C), ITE)

    # 测试基于变量A的条件ITE表达式
    A = True
    assert ITE(A, B, C) == B
    A = False
    assert ITE(A, B, C) == C
    B = True
    assert ITE(And(A, B), B, C) == C
    assert ITE(Or(A, False), And(B, True), False) is false
    assert ITE(x, A, B) == Not(x)
    assert ITE(x, B, A) == x
    assert ITE(1, x, y) == x
    assert ITE(0, x, y) == y
    # 检查引发异常的情况
    raises(TypeError, lambda: ITE(2, x, y))
    raises(TypeError, lambda: ITE(1, [], y))
    raises(TypeError, lambda: ITE(1, (), y))
    raises(TypeError, lambda: ITE(1, y, []))
    # 检查ITE条件为1时是否为真
    assert ITE(1, 1, 1) is S.true
    assert isinstance(ITE(1, 1, 1, evaluate=False), ITE)

    # 检查基于关系运算符的条件ITE表达式
    assert ITE(Eq(x, True), y, x) == ITE(x, y, x)
    assert ITE(Eq(x, False), y, x) == ITE(~x, y, x)
    assert ITE(Ne(x, True), y, x) == ITE(~x, y, x)
    assert ITE(Ne(x, False), y, x) == ITE(x, y, x)
    assert ITE(Eq(S.true, x), y, x) == ITE(x, y, x)
    assert ITE(Eq(S.false, x), y, x) == ITE(~x, y, x)
    assert ITE(Ne(S.true, x), y, x) == ITE(~x, y, x)
    assert ITE(Ne(S.false, x), y, x) == ITE(x, y, x)
    # 检查条件为0和1时的ITE表达式
    assert ITE(Eq(x, 0), y, x) == x
    assert ITE(Eq(x, 1), y, x) == x
    assert ITE(Ne(x, 0), y, x) == y
    assert ITE(Ne(x, 1), y, x) == y
    # 检查ITE表达式在替换变量后的值
    assert ITE(Eq(x, 0), y, z).subs(x, 0) == y
    assert ITE(Eq(x, 0), y, z).subs(x, 1) == z
    # 检查引发值错误的情况
    raises(ValueError, lambda: ITE(x > 1, y, x, z))


# 测试是否为文字（Literal）
def test_is_literal():
    # 检查常量True和False是否为文字
    assert is_literal(True) is True
    assert is_literal(False) is True
    # 检查单个变量A是否为文字
    assert is_literal(A) is True
    # 检查否定单个变量~A是否为文字
    assert is_literal(~A) is True
    # 检查析取Or(A, B)是否为文字
    assert is_literal(Or(A, B)) is False
    # 检查Q.zero(A)是否为文字
    assert is_literal(Q.zero(A)) is True
    # 检查否定Q.zero(A)是否为文字
    assert is_literal(Not(Q.zero(A))) is True
    # 再次检查析取Or(A, B)是否为文字
    assert is_literal(Or(A, B)) is False
    # 断言：检查 And(Q.zero(A), Q.zero(B)) 不是字面量
    assert is_literal(And(Q.zero(A), Q.zero(B))) is False
    
    # 断言：检查 x < 3 是字面量
    assert is_literal(x < 3)
    
    # 断言：检查 x + y < 3 不是字面量
    assert not is_literal(x + y < 3)
def test_operators():
    # 测试逻辑运算符的功能，包括 __and__、__rand__ 等
    assert True & A == A & True == A  # 确保逻辑与操作的交换律和恒等律成立
    assert False & A == A & False == False  # 确保逻辑与操作的恒等律成立
    assert A & B == And(A, B)  # 检查自定义的 And 函数是否正确实现
    assert True | A == A | True == True  # 确保逻辑或操作的交换律和恒等律成立
    assert False | A == A | False == A  # 确保逻辑或操作的恒等律成立
    assert A | B == Or(A, B)  # 检查自定义的 Or 函数是否正确实现
    assert ~A == Not(A)  # 检查取反操作是否正确实现
    assert True >> A == A << True == A  # 确保右移和左移操作的交换律和恒等律成立
    assert False >> A == A << False == True  # 确保右移和左移操作的恒等律成立
    assert A >> True == True << A == True  # 确保右移和左移操作的恒等律成立
    assert A >> False == False << A == ~A  # 确保右移和左移操作的恒等律成立
    assert A >> B == B << A == Implies(A, B)  # 检查自定义的 Implies 函数是否正确实现
    assert True ^ A == A ^ True == ~A  # 确保异或操作的交换律和恒等律成立
    assert False ^ A == A ^ False == A  # 确保异或操作的恒等律成立
    assert A ^ B == Xor(A, B)  # 检查自定义的 Xor 函数是否正确实现


def test_true_false():
    assert true is S.true  # 确保自定义的 true 与 S.true 相等
    assert false is S.false  # 确保自定义的 false 与 S.false 相等
    assert true is not True  # 确保自定义的 true 与 Python 内置的 True 不同
    assert false is not False  # 确保自定义的 false 与 Python 内置的 False 不同
    assert true  # 确保 true 在布尔上下文中被认为为真
    assert not false  # 确保 false 在布尔上下文中被认为为假
    assert true == True  # 确保自定义的 true 与 Python 内置的 True 值相等
    assert false == False  # 确保自定义的 false 与 Python 内置的 False 值相等
    assert not (true == False)  # 确保自定义的 true 与 Python 内置的 False 值不相等
    assert not (false == True)  # 确保自定义的 false 与 Python 内置的 True 值不相等
    assert not (true == false)  # 确保自定义的 true 与自定义的 false 值不相等

    assert hash(true) == hash(True)  # 确保自定义的 true 与 Python 内置的 True 哈希值相等
    assert hash(false) == hash(False)  # 确保自定义的 false 与 Python 内置的 False 哈希值相等
    assert len({true, True}) == len({false, False}) == 1  # 确保在集合中自定义的 true 和 Python 内置的 True 只出现一次

    assert isinstance(true, BooleanAtom)  # 确保自定义的 true 是 BooleanAtom 类的实例
    assert isinstance(false, BooleanAtom)  # 确保自定义的 false 是 BooleanAtom 类的实例
    # 我们不希望从 bool 类继承，因为 bool 类继承自 int。但是像 &、|、^、<<、>> 和 ~ 等操作符在 true 和 false 上的行为与在 0 和 1 上的行为不同。
    # 详见各个 And、Or 等函数的文档字符串中的例子。
    assert not isinstance(true, bool)  # 确保自定义的 true 不是 Python 内置的 bool 类的实例
    assert not isinstance(false, bool)  # 确保自定义的 false 不是 Python 内置的 bool 类的实例

    # 注意：这里使用 'is' 进行比较是重要的。我们希望这些返回 true 和 false，而不是 True 和 False

    assert Not(true) is false  # 确保取反操作得到自定义的 false
    assert Not(True) is false  # 确保取反操作得到自定义的 false
    assert Not(false) is true  # 确保取反操作得到自定义的 true
    assert Not(False) is true  # 确保取反操作得到自定义的 true
    assert ~true is false  # 确保按位取反操作得到自定义的 false
    assert ~false is true  # 确保按位取反操作得到自定义的 true

    assert all(i.simplify(1, 2) is i for i in (S.true, S.false))  # 确保简化操作后的 true 和 false 保持不变


def test_bool_as_set():
    assert ITE(y <= 0, False, y >= 1).as_set() == Interval(1, oo)  # 检查条件表达式 ITE 的集合表示
    assert And(x <= 2, x >= -2).as_set() == Interval(-2, 2)  # 检查 And 表达式的集合表示
    assert Or(x >= 2, x <= -2).as_set() == Interval(-oo, -2) + Interval(2, oo)  # 检查 Or 表达式的集合表示
    assert Not(x > 2).as_set() == Interval(-oo, 2)  # 检查 Not 表达式的集合表示
    # issue 10240
    assert Not(And(x > 2, x < 3)).as_set() == \
        Union(Interval(-oo, 2), Interval(3, oo))  # 检查复合表达式 Not(And(...)) 的集合表示
    assert true.as_set() == S.UniversalSet  # 确保 true 的集合表示是全集
    assert false.as_set() is S.EmptySet  # 确保 false 的集合表示是空集
    assert x.as_set() == S.UniversalSet  # 确保变量 x 的集合表示是全集
    assert And(Or(x < 1, x > 3), x < 2).as_set() == Interval.open(-oo, 1)  # 检查复合表达式的集合表示
    assert And(x < 1, sin(x) < 3).as_set() == (x < 1).as_set()  # 检查复合表达式的集合表示
    raises(NotImplementedError, lambda: (sin(x) < 1).as_set())  # 检查未实现的表达式的集合表示
    # watch for object morph in as_set
    assert Eq(-1, cos(2*x)**2/sin(2*x)**2).as_set() is S.EmptySet  # 检查等式表达式的集合表示


@XFAIL
def test_multivariate_bool_as_set():
    x, y = symbols('x,y')

    assert And(x >= 0, y >= 0).as_set() == Interval(0, oo)*Interval(0, oo)  # 检查多变量 And 表达式的集合表示
    assert Or(x >= 0, y >= 0).as_set() == S.Reals*S.Reals - \
        Interval(-oo, 0, True, True)*Interval(-oo, 0, True, True)  # 检查多变量 Or 表达式的集合表示
def test_all_or_nothing():
    # 定义扩展实数符号 x
    x = symbols('x', extended_real=True)
    # 创建表示无限的符号 -oo 和 oo 的比较条件
    args = x >= -oo, x <= oo
    # 使用 And 函数将比较条件组合成一个整体条件 v
    v = And(*args)
    # 检查 v 是否是 And 类型的对象
    if v.func is And:
        # 断言条件：v 的参数个数应该是原始参数个数减去 S.true 的数量
        assert len(v.args) == len(args) - args.count(S.true)
    else:
        # 如果 v 不是 And 类型，则断言 v 等于 True
        assert v == True
    # 使用 Or 函数将比较条件 args 组合成一个整体条件 v
    v = Or(*args)
    # 检查 v 是否是 Or 类型的对象
    if v.func is Or:
        # 断言条件：v 的参数个数应该是 2
        assert len(v.args) == 2
    else:
        # 如果 v 不是 Or 类型，则断言 v 等于 True
        assert v == True


def test_canonical_atoms():
    # 断言 true 的 canonical 属性等于 true 本身
    assert true.canonical == true
    # 断言 false 的 canonical 属性等于 false 本身
    assert false.canonical == false


def test_negated_atoms():
    # 断言 true 的 negated 属性等于 false
    assert true.negated == false
    # 断言 false 的 negated 属性等于 true
    assert false.negated == true


def test_issue_8777():
    # 测试 And(x > 2, x < oo) 的集合表示是否等于半开区间 Interval(2, oo, left_open=True)
    assert And(x > 2, x < oo).as_set() == Interval(2, oo, left_open=True)
    # 测试 And(x >= 1, x < oo) 的集合表示是否等于闭区间 Interval(1, oo)
    assert And(x >= 1, x < oo).as_set() == Interval(1, oo)
    # 测试 (x < oo) 的集合表示是否等于全实数区间 Interval(-oo, oo)
    assert (x < oo).as_set() == Interval(-oo, oo)
    # 测试 (x > -oo) 的集合表示是否等于全实数区间 Interval(-oo, oo)
    assert (x > -oo).as_set() == Interval(-oo, oo)


def test_issue_8975():
    # 测试 Or(And(-oo < x, x <= -2), And(2 <= x, x < oo)) 的集合表示是否等于两个区间的并集
    assert Or(And(-oo < x, x <= -2), And(2 <= x, x < oo)).as_set() == \
        Interval(-oo, -2) + Interval(2, oo)


def test_term_to_integer():
    # 测试将二进制列表 [1, 0, 1, 0, 0, 1, 0] 转换为整数是否为 82
    assert term_to_integer([1, 0, 1, 0, 0, 1, 0]) == 82
    # 测试将二进制字符串 '0010101000111001' 转换为整数是否为 10809
    assert term_to_integer('0010101000111001') == 10809


def test_issue_21971():
    # 定义符号变量 a, b, c, d
    a, b, c, d = symbols('a b c d')
    # 定义布尔表达式 f
    f = a & b & c | a & c
    # 断言 f 中的 a & c 替换为 d 后的结果
    assert f.subs(a & c, d) == b & d | d
    # 断言 f 中的 a & b & c 替换为 d 后的结果
    assert f.subs(a & b & c, d) == a & c | d

    # 定义布尔表达式 f
    f = (a | b | c) & (a | c)
    # 断言 f 中的 a | c 替换为 d 后的结果
    assert f.subs(a | c, d) == (b | d) & d
    # 断言 f 中的 a | b | c 替换为 d 后的结果
    assert f.subs(a | b | c, d) == (a | c) & d

    # 定义布尔表达式 f
    f = (a ^ b ^ c) & (a ^ c)
    # 断言 f 中的 a ^ c 替换为 d 后的结果
    assert f.subs(a ^ c, d) == (b ^ d) & d
    # 断言 f 中的 a ^ b ^ c 替换为 d 后的结果
    assert f.subs(a ^ b ^ c, d) == (a ^ c) & d


def test_truth_table():
    # 断言 And(x, y) 的真值表是否为 [False, False, False, True]
    assert list(truth_table(And(x, y), [x, y], input=False)) == \
        [False, False, False, True]
    # 断言 Or(x, y) 的真值表是否为 [False, True, True, True]
    assert list(truth_table(x | y, [x, y], input=False)) == \
        [False, True, True, True]
    # 断言 x >> y 的真值表是否为 [True, True, False, True]
    assert list(truth_table(x >> y, [x, y], input=False)) == \
        [True, True, False, True]
    # 断言 And(x, y) 的真值表是否为 [([0, 0], False), ([0, 1], False), ([1, 0], False), ([1, 1], True)]
    assert list(truth_table(And(x, y), [x, y])) == \
        [([0, 0], False), ([0, 1], False), ([1, 0], False), ([1, 1], True)]


def test_issue_8571():
    # 对 S.true 和 S.false 进行类型错误检查
    for t in (S.true, S.false):
        raises(TypeError, lambda: +t)
        raises(TypeError, lambda: -t)
        raises(TypeError, lambda: abs(t))
        # 使用 int(bool(t)) 得到 0 或 1，进行类型错误检查
        raises(TypeError, lambda: int(t))

        # 对 [S.Zero, S.One, x] 中的每一个元素 o 进行运算类型错误检查
        for o in [S.Zero, S.One, x]:
            for _ in range(2):
                raises(TypeError, lambda: o + t)
                raises(TypeError, lambda: o - t)
                raises(TypeError, lambda: o % t)
                raises(TypeError, lambda: o*t)
                raises(TypeError, lambda: o/t)
                raises(TypeError, lambda: o**t)
                # 将 o 和 t 互换顺序后再次进行运算类型错误检查
                o, t = t, o  # do again in reversed order


def test_expand_relational():
    # 定义负数符号 n 和正数符号 p, q
    n = symbols('n', negative=True)
    p, q = symbols('p q', positive=True)
    # 定义布尔表达式 r
    r = ((n + q*(-n/q + 1))/(q*(-n/q + 1)) < 0)
    # 断言 r 不等于 S.false
    assert r is not S.false
    # 断言 r 展开后等于 S.false
    assert r.expand() is S.false
    # 断言 q > 0 展开后等于 S.true
    assert (q > 0).expand() is S.true


def test_issue_12717():
    # 断言 S.true 的 is_Atom 属性为 True
    assert S.true.is_Atom == True
    # 断言 S.false 的 is_Atom 属性为 True
    assert S.false.is_Atom == True
    # 定义非零的符号变量 nz
    nz = symbols('nz', nonzero=True)
    # 断言所有给定的值经过 as_Boolean 转换后都应为真值 S.true
    assert all(as_Boolean(i) is S.true for i in (True, S.true, 1, nz))
    
    # 定义零值的符号变量 z
    z = symbols('z', zero=True)
    # 断言所有给定的值经过 as_Boolean 转换后都应为假值 S.false
    assert all(as_Boolean(i) is S.false for i in (False, S.false, 0, z))
    
    # 断言所有给定的值经过 as_Boolean 转换后应保持其原值
    assert all(as_Boolean(i) == i for i in (x, x < 0))
    
    # 遍历迭代包含不同类型元素的元组，预期每个元素作为参数传递给 as_Boolean 应引发 TypeError 异常
    for i in (2, S(2), x + 1, []):
        raises(TypeError, lambda: as_Boolean(i))
def test_binary_symbols():
    # 检查 ITE 条件是否小于 1，如果是则返回 y，否则返回 z，计算其二进制符号
    assert ITE(x < 1, y, z).binary_symbols == {y, z}
    # 遍历等式（Eq）和不等式（Ne）函数，检查二进制符号的正确性
    for f in (Eq, Ne):
        assert f(x, 1).binary_symbols == set()  # 空集合，因为没有变量
        assert f(x, True).binary_symbols == {x}  # 包含变量 x
        assert f(x, False).binary_symbols == {x}  # 包含变量 x
    # 检查 S.true 和 S.false 的二进制符号，均为空集合
    assert S.true.binary_symbols == set()
    assert S.false.binary_symbols == set()
    # 检查单独变量 x 的二进制符号，包含变量 x
    assert x.binary_symbols == {x}
    # 检查 And 表达式的二进制符号，包含变量 x 和 y
    assert And(x, Eq(y, False), Eq(z, 1)).binary_symbols == {x, y}
    # 检查 Q.prime(x) 函数的二进制符号，为空集合
    assert Q.prime(x).binary_symbols == set()
    # 检查 Q.lt(x, 1) 函数的二进制符号，为空集合
    assert Q.lt(x, 1).binary_symbols == set()
    # 检查 Q.is_true(x) 函数的二进制符号，包含变量 x
    assert Q.is_true(x).binary_symbols == {x}
    # 检查 Q.eq(x, True) 函数的二进制符号，包含变量 x
    assert Q.eq(x, True).binary_symbols == {x}
    # 再次检查 Q.prime(x) 函数的二进制符号，为空集合
    assert Q.prime(x).binary_symbols == set()


def test_BooleanFunction_diff():
    # 测试 And(x, y) 对 x 的微分，返回一个分段函数
    assert And(x, y).diff(x) == Piecewise((0, Eq(y, False)), (1, True))


def test_issue_14700():
    # 定义逻辑变量 A 到 H
    A, B, C, D, E, F, G, H = symbols('A B C D E F G H')
    # 原始逻辑表达式 q、simplified DNF 表达式 soldnf 和 simplified CNF 表达式 solcnf 的比较
    q = ((B & D & H & ~F) | (B & H & ~C & ~D) | (B & H & ~C & ~F) |
         (B & H & ~D & ~G) | (B & H & ~F & ~G) | (C & G & ~B & ~D) |
         (C & G & ~D & ~H) | (C & G & ~F & ~H) | (D & F & H & ~B) |
         (D & F & ~G & ~H) | (B & D & F & ~C & ~H) | (D & E & F & ~B & ~C) |
         (D & F & ~A & ~B & ~C) | (D & F & ~A & ~C & ~H) |
         (A & B & D & F & ~E & ~H))
    soldnf = ((B & D & H & ~F) | (D & F & H & ~B) | (B & H & ~C & ~D) |
              (B & H & ~D & ~G) | (C & G & ~B & ~D) | (C & G & ~D & ~H) |
              (C & G & ~F & ~H) | (D & F & ~G & ~H) | (D & E & F & ~C & ~H) |
              (D & F & ~A & ~C & ~H) | (A & B & D & F & ~E & ~H))
    solcnf = ((B | C | D) & (B | D | G) & (C | D | H) & (C | F | H) &
              (D | G | H) & (F | G | H) & (B | F | ~D | ~H) &
              (~B | ~D | ~F | ~H) & (D | ~B | ~C | ~G | ~H) &
              (A | H | ~C | ~D | ~F | ~G) & (H | ~C | ~D | ~E | ~F | ~G) &
              (B | E | H | ~A | ~D | ~F | ~G))
    # 简化逻辑表达式 q 到 DNF 形式
    assert simplify_logic(q, "dnf") == soldnf
    # 简化逻辑表达式 q 到 CNF 形式
    assert simplify_logic(q, "cnf") == solcnf

    # 检查给定的最小项和不关心项，生成 SOP 形式的逻辑表达式
    minterms = [[0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
                [0, 0, 1, 1], [1, 0, 1, 1]]
    dontcares = [[1, 0, 0, 0], [1, 0, 0, 1], [1, 1, 0, 0], [1, 1, 0, 1]]
    # 使用最小项和不关心项生成 SOP 形式的逻辑表达式，不应该更复杂
    assert SOPform([w, x, y, z], minterms) == (x & ~w) | (y & z & ~x)
    # 使用最小项和不关心项生成 SOP 形式的逻辑表达式，不应该更复杂
    assert SOPform([w, x, y, z], minterms, dontcares) == \
        (x & ~w) | (y & z & ~x)


def test_issue_25115():
    # 创建包含条件 x 是否包含在整数集合 S.Integers 中的对象
    cond = Contains(x, S.Integers)
    # 简化逻辑表达式 cond，应该得到相同的结果
    assert simplify_logic(cond) == cond


def test_relational_simplification():
    # 定义实数变量 w, x, y, z，虚数变量 d, e
    w, x, y, z = symbols('w x y z', real=True)
    d, e = symbols('d e', real=False)
    # 测试所有可能的符号和顺序组合
    assert Or(x >= y, x < y).simplify() == S.true
    assert Or(x >= y, y > x).simplify() == S.true
    assert Or(x >= y, -x > -y).simplify() == S.true
    assert Or(x >= y, -y < -x).simplify() == S.true
    assert Or(-x <= -y, x < y).simplify() == S.true
    assert Or(-x <= -y, -x > -y).simplify() == S.true
    # 断言：确保逻辑表达式 Or(-x <= -y, y > x) 经简化后为真
    assert Or(-x <= -y, y > x).simplify() == S.true
    # 断言：确保逻辑表达式 Or(-x <= -y, -y < -x) 经简化后为真
    assert Or(-x <= -y, -y < -x).simplify() == S.true
    # 断言：确保逻辑表达式 Or(y <= x, x < y) 经简化后为真
    assert Or(y <= x, x < y).simplify() == S.true
    # 断言：确保逻辑表达式 Or(y <= x, y > x) 经简化后为真
    assert Or(y <= x, y > x).simplify() == S.true
    # 断言：确保逻辑表达式 Or(y <= x, -x > -y) 经简化后为真
    assert Or(y <= x, -x > -y).simplify() == S.true
    # 断言：确保逻辑表达式 Or(y <= x, -y < -x) 经简化后为真
    assert Or(y <= x, -y < -x).simplify() == S.true
    # 断言：确保逻辑表达式 Or(-y >= -x, x < y) 经简化后为真
    assert Or(-y >= -x, x < y).simplify() == S.true
    # 断言：确保逻辑表达式 Or(-y >= -x, y > x) 经简化后为真
    assert Or(-y >= -x, y > x).simplify() == S.true
    # 断言：确保逻辑表达式 Or(-y >= -x, -x > -y) 经简化后为真
    assert Or(-y >= -x, -x > -y).simplify() == S.true
    # 断言：确保逻辑表达式 Or(-y >= -x, -y < -x) 经简化后为真
    assert Or(-y >= -x, -y < -x).simplify() == S.true

    # 断言：确保逻辑表达式 Or(x < y, x >= y) 经简化后为真
    assert Or(x < y, x >= y).simplify() == S.true
    # 断言：确保逻辑表达式 Or(y > x, x >= y) 经简化后为真
    assert Or(y > x, x >= y).simplify() == S.true
    # 断言：确保逻辑表达式 Or(-x > -y, x >= y) 经简化后为真
    assert Or(-x > -y, x >= y).simplify() == S.true
    # 断言：确保逻辑表达式 Or(-y < -x, x >= y) 经简化后为真
    assert Or(-y < -x, x >= y).simplify() == S.true
    # 断言：确保逻辑表达式 Or(x < y, -x <= -y) 经简化后为真
    assert Or(x < y, -x <= -y).simplify() == S.true
    # 断言：确保逻辑表达式 Or(-x > -y, -x <= -y) 经简化后为真
    assert Or(-x > -y, -x <= -y).simplify() == S.true
    # 断言：确保逻辑表达式 Or(y > x, -x <= -y) 经简化后为真
    assert Or(y > x, -x <= -y).simplify() == S.true
    # 断言：确保逻辑表达式 Or(-y < -x, -x <= -y) 经简化后为真
    assert Or(-y < -x, -x <= -y).simplify() == S.true
    # 断言：确保逻辑表达式 Or(x < y, y <= x) 经简化后为真
    assert Or(x < y, y <= x).simplify() == S.true
    # 断言：确保逻辑表达式 Or(y > x, y <= x) 经简化后为真
    assert Or(y > x, y <= x).simplify() == S.true
    # 断言：确保逻辑表达式 Or(-x > -y, y <= x) 经简化后为真
    assert Or(-x > -y, y <= x).simplify() == S.true
    # 断言：确保逻辑表达式 Or(-y < -x, y <= x) 经简化后为真
    assert Or(-y < -x, y <= x).simplify() == S.true
    # 断言：确保逻辑表达式 Or(x < y, -y >= -x) 经简化后为真
    assert Or(x < y, -y >= -x).simplify() == S.true
    # 断言：确保逻辑表达式 Or(y > x, -y >= -x) 经简化后为真
    assert Or(y > x, -y >= -x).simplify() == S.true
    # 断言：确保逻辑表达式 Or(-x > -y, -y >= -x) 经简化后为真
    assert Or(-x > -y, -y >= -x).simplify() == S.true
    # 断言：确保逻辑表达式 Or(-y < -x, -y >= -x) 经简化后为真
    assert Or(-y < -x, -y >= -x).simplify() == S.true

    # 断言：确保逻辑表达式 And(x >= y, w < z, x <= y) 经简化后为真
    assert Or(x >= y, w < z, x <= y).simplify() == S.true
    # 断言：确保逻辑表达式 And(x >= y, x < y) 经简化后为假
    assert And(x >= y, x < y).simplify() == S.false
    # 断言：确保逻辑表达式 Or(x >= y, Eq(y, x)) 经简化后为 (x >= y)
    assert Or(x >= y, Eq(y, x)).simplify() == (x >= y)
    # 断言：确保逻辑表达式 And(x >= y, Eq(y, x)) 经简化后为 Eq(x, y)
    assert And(x >= y, Eq(y, x)).simplify() == Eq(x, y)
    # 断言：确保逻辑表达式 And(Eq(x, y), x >= 1, 2 < y, y >= 5, z < y) 经简化后为 (Eq(x, y) & (x >= 1) & (y >= 5) & (y > z))
    assert And(Eq(x, y), x >= 1, 2 < y, y >= 5, z < y).simplify() == \
        (Eq(x, y) & (x >= 1) & (y >= 5) & (y > z))
    # 断言：确保逻辑表达式 Or(Eq(x, y), x >= y, w < y, z < y) 经简化后为 (x >= y) | (y > z) | (w < y)
    assert Or(Eq(x, y), x >= y, w < y, z < y).simplify() == \
        (x >= y) | (y > z) | (w < y)
    # 断言：确保逻辑表达式 And(Eq(x, y), x >= y, w < y, y >= z, z < y) 经简化后为 Eq(x, y) & (y > z) & (w < y)
    assert And(Eq(x, y), x >= y, w < y, y >= z, z < y).simplify() == \
        Eq(x, y) & (y > z) & (w < y)
    # 断言：确保逻辑表达式 And(Eq(x, y), x >= 1, 2 < y, y >= 5, z < y) 经简化后为 (Eq(x, y) & (x >= 1) & (y >= 5) & (y > z))
    assert And(Eq(x, y), x >= 1, 2 < y, y >= 5, z < y).simplify() ==
# 定义一个测试函数，用于验证 issue 8373
def test_issue_8373():
    # 创建一个实数符号 x
    x = symbols('x', real=True)
    # 断言 x < 1 或者 x > -1 简化后为真
    assert Or(x < 1, x > -1).simplify() == S.true
    # 断言 x < 1 或者 x >= 1 简化后为真
    assert Or(x < 1, x >= 1).simplify() == S.true
    # 断言 x < 1 且 x >= 1 简化后为假
    assert And(x < 1, x >= 1).simplify() == S.false
    # 断言 x <= 1 或者 x >= 1 简化后为真
    assert Or(x <= 1, x >= 1).simplify() == S.true


# 定义一个测试函数，用于验证 issue 7950
def test_issue_7950():
    # 创建一个实数符号 x
    x = symbols('x', real=True)
    # 断言 x 等于 1 且 x 等于 2 简化后为假
    assert And(Eq(x, 1), Eq(x, 2)).simplify() == S.false


# 定义一个测试函数，用于验证数值上的关系简化
@slow
def test_relational_simplification_numerically():
    # 定义一个内部函数，用于测试数值上的简化
    def test_simplification_numerically_function(original, simplified):
        # 获取原始表达式中的自由符号
        symb = original.free_symbols
        # 计算自由符号的数量
        n = len(symb)
        # 生成符号值的组合列表
        valuelist = list(set(combinations(list(range(-(n-1), n))*n, n)))
        # 遍历值列表
        for values in valuelist:
            # 创建符号到值的映射
            sublist = dict(zip(symb, values))
            # 计算原始表达式和简化表达式在特定值下的结果
            originalvalue = original.subs(sublist)
            simplifiedvalue = simplified.subs(sublist)
            # 断言原始表达式和简化表达式在特定值下的结果相等
            assert originalvalue == simplifiedvalue, "Original: {}\nand"\
                " simplified: {}\ndo not evaluate to the same value for {}"\
                "".format(original, simplified, sublist)

    # 定义实数符号列表
    w, x, y, z = symbols('w x y z', real=True)
    # 定义非实数符号列表
    d, e = symbols('d e', real=False)

    # 待测试的表达式列表
    expressions = (And(Eq(x, y), x >= y, w < y, y >= z, z < y),
                   And(Eq(x, y), x >= 1, 2 < y, y >= 5, z < y),
                   Or(Eq(x, y), x >= 1, 2 < y, y >= 5, z < y),
                   And(x >= y, Eq(y, x)),
                   Or(And(Eq(x, y), x >= y, w < y, Or(y >= z, z < y)),
                      And(Eq(x, y), x >= 1, 2 < y, y >= -1, z < y)),
                   (Eq(x, y) & Eq(d, e) & (x >= y) & (d >= e)),
                   )

    # 遍历表达式列表，进行数值简化测试
    for expression in expressions:
        test_simplification_numerically_function(expression,
                                                 expression.simplify())


# 定义一个测试函数，用于验证关系模式在数值上的简化
def test_relational_simplification_patterns_numerically():
    # 导入必要的符号和模式函数
    from sympy.core import Wild
    from sympy.logic.boolalg import _simplify_patterns_and, \
        _simplify_patterns_or, _simplify_patterns_xor
    # 创建通配符
    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    # 符号列表
    symb = [a, b, c]
    # 模式列表
    patternlists = [[And, _simplify_patterns_and()],
                    [Or, _simplify_patterns_or()],
                    [Xor, _simplify_patterns_xor()]]
    # 生成值列表
    valuelist = list(set(combinations(list(range(-2, 3))*3, 3)))
    # 跳过 +/-2 和 0 的组合，除非全为0
    valuelist = [v for v in valuelist if any(w % 2 for w in v) or not any(v)]
    # 遍历模式列表和模式
    for func, patternlist in patternlists:
        for pattern in patternlist:
            # 创建原始表达式和简化后表达式
            original = func(*pattern[0].args)
            simplified = pattern[1]
            # 遍历值列表，进行数值简化测试
            for values in valuelist:
                sublist = dict(zip(symb, values))
                originalvalue = original.xreplace(sublist)
                simplifiedvalue = simplified.xreplace(sublist)
                # 断言原始表达式和简化后表达式在特定值下的结果相等
                assert originalvalue == simplifiedvalue, "Original: {}\nand"\
                    " simplified: {}\ndo not evaluate to the same value for"\
                    "{}".format(pattern[0], simplified, sublist)


# 定义一个测试函数，用于验证 issue 16803
def test_issue_16803():
    # 这个函数暂时为空，用于未来扩展
    pass
    # 定义符号变量 n
    n = symbols('n')
    
    # 进行断言测试，验证布尔表达式在不进行简化的情况下是否保持不变，不应引发异常
    assert ((n > 3) | (n < 0) | ((n > 0) & (n < 3))).simplify() == \
        (n > 3) | (n < 0) | ((n > 0) & (n < 3))
def test_issue_17530():
    # 定义字典 r 包含两个无穷大的键值对
    r = {x: oo, y: oo}
    # 断言 Or 表达式 x + y > 0 在 r 中求值为真
    assert Or(x + y > 0, x - y < 0).subs(r)
    # 断言 And 表达式 x + y < 0 在 r 中求值为假
    assert not And(x + y < 0, x - y < 0).subs(r)
    # 断言 Or 表达式 x + y < 0 在 r 中会抛出 TypeError 异常
    raises(TypeError, lambda: Or(x + y < 0, x - y < 0).subs(r))
    # 断言 And 表达式 x + y > 0 在 r 中会抛出 TypeError 异常
    raises(TypeError, lambda: And(x + y > 0, x - y < 0).subs(r))
    # 断言 And 表达式 x + y > 0 在 r 中会抛出 TypeError 异常
    raises(TypeError, lambda: And(x + y > 0, x - y < 0).subs(r))


def test_anf_coeffs():
    # 断言调用 anf_coeffs 函数，验证给定输入的输出是否符合预期
    assert anf_coeffs([1, 0]) == [1, 1]
    assert anf_coeffs([0, 0, 0, 1]) == [0, 0, 0, 1]
    assert anf_coeffs([0, 1, 1, 1]) == [0, 1, 1, 1]
    assert anf_coeffs([1, 1, 1, 0]) == [1, 0, 0, 1]
    assert anf_coeffs([1, 0, 0, 0]) == [1, 1, 1, 1]
    assert anf_coeffs([1, 0, 0, 1]) == [1, 1, 1, 0]
    assert anf_coeffs([1, 1, 0, 1]) == [1, 0, 1, 1]


def test_ANFform():
    # 定义符号变量 x 和 y
    x, y = symbols('x,y')
    # 断言调用 ANFform 函数，验证给定输入的输出是否符合预期
    assert ANFform([x], [1, 1]) == True
    assert ANFform([x], [0, 0]) == False
    assert ANFform([x], [1, 0]) == Xor(x, True, remove_true=False)
    assert ANFform([x, y], [1, 1, 1, 0]) == \
        Xor(True, And(x, y), remove_true=False)


def test_bool_minterm():
    # 定义符号变量 x 和 y
    x, y = symbols('x,y')
    # 断言调用 bool_minterm 函数，验证给定输入的输出是否符合预期
    assert bool_minterm(3, [x, y]) == And(x, y)
    assert bool_minterm([1, 0], [x, y]) == And(Not(y), x)


def test_bool_maxterm():
    # 定义符号变量 x 和 y
    x, y = symbols('x,y')
    # 断言调用 bool_maxterm 函数，验证给定输入的输出是否符合预期
    assert bool_maxterm(2, [x, y]) == Or(Not(x), y)
    assert bool_maxterm([0, 1], [x, y]) == Or(Not(y), x)


def test_bool_monomial():
    # 定义符号变量 x 和 y
    x, y = symbols('x,y')
    # 断言调用 bool_monomial 函数，验证给定输入的输出是否符合预期
    assert bool_monomial(1, [x, y]) == y
    assert bool_monomial([1, 1], [x, y]) == And(x, y)


def test_check_pair():
    # 断言调用 _check_pair 函数，验证给定输入的输出是否符合预期
    assert _check_pair([0, 1, 0], [0, 1, 1]) == 2
    assert _check_pair([0, 1, 0], [1, 1, 1]) == -1


def test_issue_19114():
    # 定义布尔表达式 expr
    expr = (B & C) | (A & ~C) | (~A & ~B)
    # 断言使用 to_dnf 函数将表达式简化后的结果与预期结果进行比较
    # 注释：表达式是最小的，但是可能有多种最小形式
    res1 = (A & B) | (C & ~A) | (~B & ~C)
    result = to_dnf(expr, simplify=True)
    assert result in (expr, res1)


def test_issue_20870():
    # 调用 SOPform 函数，验证给定输入的输出是否符合预期
    result = SOPform([a, b, c, d], [1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 14, 15])
    expected = ((d & ~b) | (a & b & c) | (a & ~c & ~d) |
                (b & ~a & ~c) | (c & ~a & ~d))
    assert result == expected


def test_convert_to_varsSOP():
    # 断言调用 _convert_to_varsSOP 函数，验证给定输入的输出是否符合预期
    assert _convert_to_varsSOP([0, 1, 0], [x, y, z]) ==  And(Not(x), y, Not(z))
    assert _convert_to_varsSOP([3, 1, 0], [x, y, z]) ==  And(y, Not(z))


def test_convert_to_varsPOS():
    # 断言调用 _convert_to_varsPOS 函数，验证给定输入的输出是否符合预期
    assert _convert_to_varsPOS([0, 1, 0], [x, y, z]) == Or(x, Not(y), z)
    assert _convert_to_varsPOS([3, 1, 0], [x, y, z]) ==  Or(Not(y), z)


def test_gateinputcount():
    # 定义符号变量 a 到 e
    a, b, c, d, e = symbols('a:e')
    # 断言调用 gateinputcount 函数，验证给定输入的输出是否符合预期
    assert gateinputcount(And(a, b)) == 2
    assert gateinputcount(a | b & c & d ^ (e | a)) == 9
    assert gateinputcount(And(a, True)) == 0
    # 断言调用 gateinputcount 函数，传入非法参数 a*b 会抛出 TypeError 异常
    raises(TypeError, lambda: gateinputcount(a*b))


def test_refine():
    # relational
    # 断言对 x < 0 应用非操作得到假
    assert not refine(x < 0, ~(x < 0))
    # 断言对 x < 0 应用 x < 0 得到真
    assert refine(x < 0, (x < 0))
    # 断言对 x < 0 应用 0 > x 得到真
    assert refine(x < 0, (0 > x)) is S.true
    # 断言对 x < 0 应用 y < 0 会返回原始表达式 x < 0
    assert refine(x < 0, (y < 0)) == (x < 0)
    # 断言对 x <= 0 应用非操作得到假
    assert not refine(x <= 0, ~(x <= 0))
    # 断言对 x <= 0 应用 x <= 0 得到真
    assert refine(x <= 0,  (x <= 0))
    assert refine(x <= 0,  (0 >= x)) is S.true
    # 确保当 x <= 0 时，0 >= x 成立，即 x 是非正数时为真

    assert refine(x <= 0,  (y <= 0)) == (x <= 0)
    # 当 x <= 0 时，y <= 0 的条件等价于 x <= 0

    assert not refine(x > 0, ~(x > 0))
    # 确保当 x > 0 时，~(x > 0) 不成立，即 x > 0 的否定为假

    assert refine(x > 0,  (x > 0))
    # 当 x > 0 时，x > 0 的条件成立

    assert refine(x > 0,  (0 < x)) is S.true
    # 当 x > 0 时，0 < x 成立，即 x 是正数时为真

    assert refine(x > 0,  (y > 0)) == (x > 0)
    # 当 x > 0 时，y > 0 的条件等价于 x > 0

    assert not refine(x >= 0, ~(x >= 0))
    # 确保当 x >= 0 时，~(x >= 0) 不成立，即 x >= 0 的否定为假

    assert refine(x >= 0,  (x >= 0))
    # 当 x >= 0 时，x >= 0 的条件成立

    assert refine(x >= 0,  (0 <= x)) is S.true
    # 当 x >= 0 时，0 <= x 成立，即 x 是非负数时为真

    assert refine(x >= 0,  (y >= 0)) == (x >= 0)
    # 当 x >= 0 时，y >= 0 的条件等价于 x >= 0

    assert not refine(Eq(x, 0), ~(Eq(x, 0)))
    # 确保当 x 等于 0 时，~(Eq(x, 0)) 不成立，即 x 等于 0 的否定为假

    assert refine(Eq(x, 0),  (Eq(x, 0)))
    # 当 x 等于 0 时，Eq(x, 0) 的条件成立

    assert refine(Eq(x, 0),  (Eq(0, x))) is S.true
    # 当 x 等于 0 时，Eq(0, x) 成立，即 0 等于 x 为真

    assert refine(Eq(x, 0),  (Eq(y, 0))) == Eq(x, 0)
    # 当 x 等于 0 时，Eq(y, 0) 的条件等价于 x 等于 0

    assert not refine(Ne(x, 0), ~(Ne(x, 0)))
    # 确保当 x 不等于 0 时，~(Ne(x, 0)) 不成立，即 x 不等于 0 的否定为假

    assert refine(Ne(x, 0), (Ne(0, x))) is S.true
    # 当 x 不等于 0 时，Ne(0, x) 成立，即 0 不等于 x 为真

    assert refine(Ne(x, 0),  (Ne(x, 0)))
    # 当 x 不等于 0 时，Ne(x, 0) 的条件成立

    assert refine(Ne(x, 0),  (Ne(y, 0))) == (Ne(x, 0))
    # 当 x 不等于 0 时，Ne(y, 0) 的条件等价于 x 不等于 0

    # boolean functions
    assert refine(And(x > 0, y > 0), (x > 0)) == (y > 0)
    # 对于布尔函数 And(x > 0, y > 0)，当 x > 0 时，该函数等价于 y > 0

    assert refine(And(x > 0, y > 0), (x > 0) & (y > 0)) is S.true
    # 对于布尔函数 And(x > 0, y > 0)，与 (x > 0) & (y > 0) 的条件等价

    # predicates
    assert refine(Q.positive(x), Q.positive(x)) is S.true
    # 对于谓词 Q.positive(x)，当 x 是正数时该条件成立

    assert refine(Q.positive(x), Q.negative(x)) is S.false
    # 对于谓词 Q.positive(x)，当 x 是负数时该条件不成立

    assert refine(Q.positive(x), Q.real(x)) == Q.positive(x)
    # 对于谓词 Q.positive(x)，当 x 是实数时该条件等价于 x 是正数
def test_relational_threeterm_simplification_patterns_numerically():
    # 导入必要的符号计算模块
    from sympy.core import Wild
    from sympy.logic.boolalg import _simplify_patterns_and3
    # 创建三个通配符对象
    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    # 符号列表，包含三个通配符对象
    symb = [a, b, c]
    # 设定模式列表，其中包含一个 And 函数和其简化模式的元组
    patternlists = [[And, _simplify_patterns_and3()]]
    # 创建值列表，包含从-2到2的所有三元组的排列组合
    valuelist = list(set(combinations(list(range(-2, 3))*3, 3)))
    # 筛选出不包含+/-2和0的组合，除非是全0组合
    valuelist = [v for v in valuelist if any(w % 2 for w in v) or not any(v)]
    # 遍历模式列表
    for func, patternlist in patternlists:
        # 遍历简化模式列表
        for pattern in patternlist:
            # 获取原始表达式
            original = func(*pattern[0].args)
            # 获取简化后的表达式
            simplified = pattern[1]
            # 遍历值列表
            for values in valuelist:
                # 将通配符映射到当前值组成的字典
                sublist = dict(zip(symb, values))
                # 替换原始表达式中的通配符并计算结果
                originalvalue = original.xreplace(sublist)
                # 替换简化后的表达式中的通配符并计算结果
                simplifiedvalue = simplified.xreplace(sublist)
                # 断言原始值和简化值应相等，否则抛出异常
                assert originalvalue == simplifiedvalue, "Original: {}\nand"\
                    " simplified: {}\ndo not evaluate to the same value for"\
                    "{}".format(pattern[0], simplified, sublist)


def test_issue_25451():
    # 创建逻辑表达式
    x = Or(And(a, c), Eq(a, b))
    # 断言 x 是 Or 类型的表达式
    assert isinstance(x, Or)
    # 断言 x 的参数集合包含给定的 And 和 Eq 表达式
    assert set(x.args) == {And(a, c), Eq(a, b)}
```