# `D:\src\scipysrc\sympy\sympy\logic\tests\test_inference.py`

```
"""For more tests on satisfiability, see test_dimacs"""

# 导入必要的模块和函数
from sympy.assumptions.ask import Q
from sympy.core.symbol import symbols
from sympy.core.relational import Unequality
from sympy.logic.boolalg import And, Or, Implies, Equivalent, true, false
from sympy.logic.inference import literal_symbol, \
     pl_true, satisfiable, valid, entails, PropKB
from sympy.logic.algorithms.dpll import dpll, dpll_satisfiable, \
    find_pure_symbol, find_unit_clause, unit_propagate, \
    find_pure_symbol_int_repr, find_unit_clause_int_repr, \
    unit_propagate_int_repr
from sympy.logic.algorithms.dpll2 import dpll_satisfiable as dpll2_satisfiable

# 导入 Z3 相关的满足性判断函数
from sympy.logic.algorithms.z3_wrapper import z3_satisfiable

# 导入 CNF 相关模块和函数
from sympy.assumptions.cnf import CNF, EncodedCNF

# 导入测试相关的函数和类
from sympy.logic.tests.test_lra_theory import make_random_problem
from sympy.core.random import randint

# 导入测试框架相关函数
from sympy.testing.pytest import raises, skip

# 导入外部模块
from sympy.external import import_module


# 测试函数：检验 literal_symbol 函数的行为是否符合预期
def test_literal():
    A, B = symbols('A,B')
    assert literal_symbol(True) is True
    assert literal_symbol(False) is False
    assert literal_symbol(A) is A
    assert literal_symbol(~A) is A


# 测试函数：检验 find_pure_symbol 函数的行为是否符合预期
def test_find_pure_symbol():
    A, B, C = symbols('A,B,C')
    assert find_pure_symbol([A], [A]) == (A, True)
    assert find_pure_symbol([A, B], [~A | B, ~B | A]) == (None, None)
    assert find_pure_symbol([A, B, C], [ A | ~B, ~B | ~C, C | A]) == (A, True)
    assert find_pure_symbol([A, B, C], [~A | B, B | ~C, C | A]) == (B, True)
    assert find_pure_symbol([A, B, C], [~A | ~B, ~B | ~C, C | A]) == (B, False)
    assert find_pure_symbol(
        [A, B, C], [~A | B, ~B | ~C, C | A]) == (None, None)


# 测试函数：检验 find_pure_symbol_int_repr 函数的行为是否符合预期
def test_find_pure_symbol_int_repr():
    assert find_pure_symbol_int_repr([1], [{1}]) == (1, True)
    assert find_pure_symbol_int_repr([1, 2],
                [{-1, 2}, {-2, 1}]) == (None, None)
    assert find_pure_symbol_int_repr([1, 2, 3],
                [{1, -2}, {-2, -3}, {3, 1}]) == (1, True)
    assert find_pure_symbol_int_repr([1, 2, 3],
                [{-1, 2}, {2, -3}, {3, 1}]) == (2, True)
    assert find_pure_symbol_int_repr([1, 2, 3],
                [{-1, -2}, {-2, -3}, {3, 1}]) == (2, False)
    assert find_pure_symbol_int_repr([1, 2, 3],
                [{-1, 2}, {-2, -3}, {3, 1}]) == (None, None)


# 测试函数：检验 find_unit_clause 函数的行为是否符合预期
def test_unit_clause():
    A, B, C = symbols('A,B,C')
    assert find_unit_clause([A], {}) == (A, True)
    assert find_unit_clause([A, ~A], {}) == (A, True)  # 错误 ??
    assert find_unit_clause([A | B], {A: True}) == (B, True)
    assert find_unit_clause([A | B], {B: True}) == (A, True)
    assert find_unit_clause(
        [A | B | C, B | ~C, A | ~B], {A: True}) == (B, False)
    assert find_unit_clause([A | B | C, B | ~C, A | B], {A: True}) == (B, True)
    assert find_unit_clause([A | B | C, B | ~C, A ], {}) == (A, True)


# 测试函数：检验 find_unit_clause_int_repr 函数的行为是否符合预期
def test_unit_clause_int_repr():
    assert find_unit_clause_int_repr(map(set, [[1]]), {}) == (1, True)
    # 检查调用 find_unit_clause_int_repr 函数并验证返回结果是否符合预期：({1}, [-1]) => (1, True)
    assert find_unit_clause_int_repr(map(set, [[1], [-1]]), {}) == (1, True)
    # 检查调用 find_unit_clause_int_repr 函数并验证返回结果是否符合预期：({1, 2}, {1: True}) => (2, True)
    assert find_unit_clause_int_repr([{1, 2}], {1: True}) == (2, True)
    # 检查调用 find_unit_clause_int_repr 函数并验证返回结果是否符合预期：({1, 2}, {2: True}) => (1, True)
    assert find_unit_clause_int_repr([{1, 2}], {2: True}) == (1, True)
    # 检查调用 find_unit_clause_int_repr 函数并验证返回结果是否符合预期：
    # ({{1, 2, 3}, {2, -3}, {1, -2}}, {1: True}) => (2, False)
    assert find_unit_clause_int_repr(map(set, [[1, 2, 3], [2, -3], [1, -2]]), {1: True}) == (2, False)
    # 检查调用 find_unit_clause_int_repr 函数并验证返回结果是否符合预期：
    # ({{1, 2, 3}, {3, -3}, {1, 2}}, {1: True}) => (2, True)
    assert find_unit_clause_int_repr(map(set, [[1, 2, 3], [3, -3], [1, 2]]), {1: True}) == (2, True)
    
    # 定义逻辑变量 A, B, C
    A, B, C = symbols('A,B,C')
    # 检查调用 find_unit_clause 函数并验证返回结果是否符合预期：
    # ([A | B | C, B | ~C, A], {}) => (A, True)
    assert find_unit_clause([A | B | C, B | ~C, A], {}) == (A, True)
def test_unit_propagate():
    # 定义逻辑符号 A, B, C
    A, B, C = symbols('A,B,C')
    # 测试单元传播函数，验证当给定 [A | B] 和 A 时返回空列表
    assert unit_propagate([A | B], A) == []
    # 测试单元传播函数，验证当给定 [A | B, ~A | C, ~C | B, A] 和 A 时返回 [C, ~C | B, A]

def test_unit_propagate_int_repr():
    # 测试整数表示的单元传播函数，验证当给定 [{1, 2}] 和 1 时返回空列表
    assert unit_propagate_int_repr([{1, 2}], 1) == []
    # 测试整数表示的单元传播函数，验证当给定 map(set, [[1, 2], [-1, 3], [-3, 2], [1]]) 和 1 时返回 [{3}, {-3, 2}]

def test_dpll():
    """This is also tested in test_dimacs"""
    # 定义逻辑符号 A, B, C
    A, B, C = symbols('A,B,C')
    # 测试 DPLL 算法，验证当给定 [A | B], [A, B], {A: True, B: True} 时返回 {A: True, B: True}

def test_dpll_satisfiable():
    # 定义逻辑符号 A, B, C
    A, B, C = symbols('A,B,C')
    # 测试可满足性的 DPLL 算法，验证当给定 A & ~A 时返回 False
    assert dpll_satisfiable( A & ~A ) is False
    # 测试可满足性的 DPLL 算法，验证当给定 A & ~B 时返回 {A: True, B: False}
    assert dpll_satisfiable( A & ~B ) == {A: True, B: False}
    # 测试可满足性的 DPLL 算法，验证当给定 A | B 时结果属于 [{A: True}, {B: True}, {A: True, B: True}]
    assert dpll_satisfiable( A | B ) in ({A: True}, {B: True}, {A: True, B: True})
    # 测试可满足性的 DPLL 算法，验证当给定 (~A | B) & (~B | A) 时结果属于 [{A: True, B: True}, {A: False, B: False}]
    assert dpll_satisfiable( (~A | B) & (~B | A) ) in ({A: True, B: True}, {A: False, B: False})
    # 测试可满足性的 DPLL 算法，验证当给定 (A | B) & (~B | C) 时结果属于 [{A: True, B: False, C: True}, {A: True, B: True, C: True}]
    assert dpll_satisfiable( (A | B) & (~B | C) ) in ({A: True, B: False, C: True}, {A: True, B: True, C: True})
    # 测试可满足性的 DPLL 算法，验证当给定 A & B & C 时返回 {A: True, B: True, C: True}
    assert dpll_satisfiable( A & B & C  ) == {A: True, B: True, C: True}
    # 测试可满足性的 DPLL 算法，验证当给定 (A | B) & (A >> B) 时返回 {B: True}
    assert dpll_satisfiable( (A | B) & (A >> B) ) == {B: True}
    # 测试可满足性的 DPLL 算法，验证当给定 Equivalent(A, B) & A 时返回 {A: True, B: True}
    assert dpll_satisfiable( Equivalent(A, B) & A ) == {A: True, B: True}
    # 测试可满足性的 DPLL 算法，验证当给定 Equivalent(A, B) & ~A 时返回 {A: False, B: False}
    assert dpll_satisfiable( Equivalent(A, B) & ~A ) == {A: False, B: False}

def test_dpll2_satisfiable():
    # 定义逻辑符号 A, B, C
    A, B, C = symbols('A,B,C')
    # 测试改进的 DPLL 算法，验证当给定 A & ~A 时返回 False
    assert dpll2_satisfiable( A & ~A ) is False
    # 测试改进的 DPLL 算法，验证当给定 A & ~B 时返回 {A: True, B: False}
    assert dpll2_satisfiable( A & ~B ) == {A: True, B: False}
    # 测试改进的 DPLL 算法，验证当给定 A | B 时结果属于 [{A: True}, {B: True}, {A: True, B: True}]
    assert dpll2_satisfiable( A | B ) in ({A: True}, {B: True}, {A: True, B: True})
    # 测试改进的 DPLL 算法，验证当给定 (~A | B) & (~B | A) 时结果属于 [{A: True, B: True}, {A: False, B: False}]
    assert dpll2_satisfiable( (~A | B) & (~B | A) ) in ({A: True, B: True}, {A: False, B: False})
    # 测试改进的 DPLL 算法，验证当给定 (A | B) & (~B | C) 时结果属于 [{A: True, B: False, C: True}, {A: True, B: True, C: True}]
    assert dpll2_satisfiable( (A | B) & (~B | C) ) in ({A: True, B: False, C: True}, {A: True, B: True, C: True})
    # 测试改进的 DPLL 算法，验证当给定 A & B & C 时返回 {A: True, B: True, C: True}
    assert dpll2_satisfiable( A & B & C  ) == {A: True, B: True, C: True}
    # 测试改进的 DPLL 算法，验证当给定 (A | B) & (A >> B) 时结果属于 [{B: True, A: False}, {B: True, A: True}]
    assert dpll2_satisfiable( (A | B) & (A >> B) ) in ({B: True, A: False}, {B: True, A: True})
    # 测试改进的 DPLL 算法，验证当给定 Equivalent(A, B) & A 时返回 {A: True, B: True}
    assert dpll2_satisfiable( Equivalent(A, B) & A ) == {A: True, B: True}
    # 测试改进的 DPLL 算法，验证当给定 Equivalent(A, B) & ~A 时返回 {A: False, B: False}

def test_minisat22_satisfiable():
    # 定义逻辑符号 A, B, C
    A, B, C = symbols('A,B,C')
    # 定义 minisat22_satisfiable 函数为使用 minisat22 算法求解可满足性
    minisat22_satisfiable = lambda expr: satisfiable(expr, algorithm="minisat22")
    # 测试 minisat22 算法，验证当给定 A & ~A 时返回 False
    assert minisat22_satisfiable( A & ~A ) is False
    # 测试 minisat22 算法，验证当给定 A & ~B 时返回 {A: True, B: False}
    assert minisat22_satisfiable( A & ~B ) == {A: True, B: False}
    # 测试 minisat22 算法，验证当给定 A | B 时结果属于 [{A: True}, {B: False}, {A: False, B: True}, {A: True, B: True}, {A: True, B: False}]
    assert minisat22_satisfiable( A | B ) in ({A: True}, {B: False}, {A: False, B: True}, {A: True, B: True}, {A: True, B: False})
    # 测试 minisat
    # 使用 minisat22_satisfiable 函数验证表达式 (A | B) & (A >> B) 的可满足性，期望结果为 {B: True, A: False} 或 {B: True, A: True}
    assert minisat22_satisfiable( (A | B) & (A >> B) ) in ({B: True, A: False},
        {B: True, A: True})
    
    # 使用 minisat22_satisfiable 函数验证表达式 Equivalent(A, B) & A 的可满足性，期望结果为 {A: True, B: True}
    assert minisat22_satisfiable( Equivalent(A, B) & A ) == {A: True, B: True}
    
    # 使用 minisat22_satisfiable 函数验证表达式 Equivalent(A, B) & ~A 的可满足性，期望结果为 {A: False, B: False}
    assert minisat22_satisfiable( Equivalent(A, B) & ~A ) == {A: False, B: False}
# 定义测试函数 test_minisat22_minimal_satisfiable，用于测试 minisat22_satisfiable 函数的多个情况
def test_minisat22_minimal_satisfiable():
    # 使用 symbols 函数创建符号变量 A, B, C
    A, B, C = symbols('A,B,C')
    # 定义 minisat22_satisfiable 函数，使用 minisat22 算法进行求解，返回最小解（如果设置为 True）
    minisat22_satisfiable = lambda expr, minimal=True: satisfiable(expr, algorithm="minisat22", minimal=True)
    # 断言表达式 A & ~A 在 minisat22 算法下为 False
    assert minisat22_satisfiable( A & ~A ) is False
    # 断言表达式 A & ~B 在 minisat22 算法下的解为 {A: True, B: False}
    assert minisat22_satisfiable( A & ~B ) == {A: True, B: False}
    # 断言表达式 A | B 在 minisat22 算法下的解为多个可能的字典之一
    assert minisat22_satisfiable( A | B ) in ({A: True}, {B: False}, {A: False, B: True}, {A: True, B: True}, {A: True, B: False})
    # 断言表达式 (~A | B) & (~B | A) 在 minisat22 算法下的解为 {A: True, B: True} 或 {A: False, B: False}
    assert minisat22_satisfiable( (~A | B) & (~B | A) ) in ({A: True, B: True}, {A: False, B: False})
    # 断言表达式 (A | B) & (~B | C) 在 minisat22 算法下的解为多个可能的字典之一
    assert minisat22_satisfiable( (A | B) & (~B | C) ) in ({A: True, B: False, C: True},
        {A: True, B: True, C: True}, {A: False, B: True, C: True}, {A: True, B: False, C: False})
    # 断言表达式 A & B & C 在 minisat22 算法下的解为 {A: True, B: True, C: True}
    assert minisat22_satisfiable( A & B & C ) == {A: True, B: True, C: True}
    # 断言表达式 (A | B) & (A >> B) 在 minisat22 算法下的解为 {B: True, A: False} 或 {B: True, A: True}
    assert minisat22_satisfiable( (A | B) & (A >> B) ) in ({B: True, A: False},
        {B: True, A: True})
    # 断言表达式 Equivalent(A, B) & A 在 minisat22 算法下的解为 {A: True, B: True}
    assert minisat22_satisfiable( Equivalent(A, B) & A ) == {A: True, B: True}
    # 断言表达式 Equivalent(A, B) & ~A 在 minisat22 算法下的解为 {A: False, B: False}
    assert minisat22_satisfiable( Equivalent(A, B) & ~A ) == {A: False, B: False}
    # 使用 satisfiable 函数获取 A | B | C 的所有模型，返回生成器 g
    g = satisfiable((A | B | C), algorithm="minisat22", minimal=True, all_models=True)
    # 获取生成器 g 的下一个解，将解中值为 True 的键构成集合 first_solution
    sol = next(g)
    first_solution = {key for key, value in sol.items() if value}
    # 再次获取生成器 g 的下一个解，将解中值为 True 的键构成集合 second_solution
    sol = next(g)
    second_solution = {key for key, value in sol.items() if value}
    # 再次获取生成器 g 的下一个解，将解中值为 True 的键构成集合 third_solution
    sol = next(g)
    third_solution = {key for key, value in sol.items() if value}
    # 断言 first_solution 不是 second_solution 的子集
    assert not first_solution <= second_solution
    # 断言 second_solution 不是 third_solution 的子集
    assert not second_solution <= third_solution
    # 断言 first_solution 不是 third_solution 的子集
    assert not first_solution <= third_solution

# 定义测试函数 test_satisfiable，用于测试 satisfiable 函数的特定情况
def test_satisfiable():
    # 使用 symbols 函数创建符号变量 A, B, C
    A, B, C = symbols('A,B,C')
    # 断言表达式 A & (A >> B) & ~B 在默认算法下为 False
    assert satisfiable(A & (A >> B) & ~B) is False

# 定义测试函数 test_valid，用于测试 valid 函数的特定情况
def test_valid():
    # 使用 symbols 函数创建符号变量 A, B, C
    A, B, C = symbols('A,B,C')
    # 断言表达式 A >> (B >> A) 在默认算法下为 True
    assert valid(A >> (B >> A)) is True
    # 断言表达式 (A >> (B >> C)) >> ((A >> B) >> (A >> C)) 在默认算法下为 True
    assert valid((A >> (B >> C)) >> ((A >> B) >> (A >> C))) is True
    # 断言表达式 (~B >> ~A) >> (A >> B) 在默认算法下为 True
    assert valid((~B >> ~A) >> (A >> B)) is True
    # 断言表达式 A | B | C 在默认算法下为 False
    assert valid(A | B | C) is False
    # 断言表达式 A >> B 在默认算法下为 False
    assert valid(A >> B) is False

# 定义测试函数 test_pl_true，用于测试 pl_true 函数的特定情况
def test_pl_true():
    # 使用 symbols 函数创建符号变量 A, B, C
    A, B, C = symbols('A,B,C')
    # 断言 pl_true(True) 的返回值为 True
    assert pl_true(True) is True
    # 断言在指定的解析赋值下，表达式 A & B 的值为 True
    assert pl_true( A & B, {A: True, B: True}) is True
    # 断言在指定的解析赋值下，表达式 A | B 的值为 True
    assert pl_true( A | B, {A: True}) is True
    # 断言在指定的解析赋值下，表达式 A | B 的值为 True
    assert pl_true( A | B, {B: True}) is True
    # 断言在指定的解析赋值下，表达式 A | B 的值为 True
    assert pl_true( A | B, {A: None, B: True}) is True
    # 断言在指定的解析赋值下，表达式 A >> B 的值为 True
    assert pl_true( A >> B, {A: False}) is True
    # 断言在指定的解析赋值下，表达式 A | B | ~C 的值为 True
    assert pl_true( A | B | ~C, {A: False, B: True, C: True}) is True
    # 断言在指定的解析赋值下，表达式 Equivalent(A, B) 的值为 True
    assert pl_true(Equivalent(A, B), {A: False, B: False}) is True

    # 测试返回 False 的情况
    # 断言 pl_true(False) 的返回值为 False
    assert pl_true(False) is False
    # 断言在指定的解析赋值下，表达式 A & B 的值为 False
    assert pl_true( A & B, {A: False, B: False}) is False
    # 断言在指定的解析赋值下，表达式 A & B 的值为 False
    assert pl_true( A & B, {A: False}) is False
    # 断言在指定的解析赋值下，表达
    # 对 pl_true 函数进行深度测试
    assert pl_true(A | B, {A: False}, deep=True) is None
    
    # 对 pl_true 函数进行深度测试，使用 ~A 和 ~B 作为参数
    assert pl_true(~A & ~B, {A: False}, deep=True) is None
    
    # 对 pl_true 函数进行深度测试，验证 A | B 在给定条件下的结果是否为 False
    assert pl_true(A | B, {A: False, B: False}, deep=True) is False
    
    # 对 pl_true 函数进行深度测试，验证 A & B & (~A | ~B) 在给定条件下的结果是否为 False
    assert pl_true(A & B & (~A | ~B), {A: True}, deep=True) is False
    
    # 对 pl_true 函数进行深度测试，验证 (C >> A) >> (B >> A) 在给定条件下的结果是否为 True
    assert pl_true((C >> A) >> (B >> A), {C: True}, deep=True) is True
def test_pl_true_wrong_input():
    # 导入 pi 符号以进行测试
    from sympy.core.numbers import pi
    # 检查 pl_true 函数对非法输入 'John Cleese' 是否引发 ValueError 异常
    raises(ValueError, lambda: pl_true('John Cleese'))
    # 检查 pl_true 函数对复杂表达式 42 + pi + pi ** 2 是否引发 ValueError 异常
    raises(ValueError, lambda: pl_true(42 + pi + pi ** 2))
    # 检查 pl_true 函数对整数 42 是否引发 ValueError 异常
    raises(ValueError, lambda: pl_true(42))


def test_entails():
    # 定义逻辑符号 A, B, C 作为测试符号
    A, B, C = symbols('A, B, C')
    # 断言 A 隐含于 [A >> B, ~B] 是 False
    assert entails(A, [A >> B, ~B]) is False
    # 断言 B 隐含于 [Equivalent(A, B), A] 是 True
    assert entails(B, [Equivalent(A, B), A]) is True
    # 断言 (A >> B) >> (~A >> ~B) 的蕴含性是 False
    assert entails((A >> B) >> (~A >> ~B)) is False
    # 断言 (A >> B) >> (~B >> ~A) 的蕴含性是 True
    assert entails((A >> B) >> (~B >> ~A)) is True


def test_PropKB():
    # 定义逻辑符号 A, B, C 作为测试符号
    A, B, C = symbols('A,B,C')
    # 创建一个新的命题逻辑知识库
    kb = PropKB()
    # 断言 kb 对 A >> B 的询问结果是 False
    assert kb.ask(A >> B) is False
    # 断言 kb 对 A >> (B >> A) 的询问结果是 True
    assert kb.ask(A >> (B >> A)) is True
    # 向知识库 kb 中添加 A >> B 的命题
    kb.tell(A >> B)
    # 向知识库 kb 中添加 B >> C 的命题
    kb.tell(B >> C)
    # 断言 kb 对 A 的询问结果是 False
    assert kb.ask(A) is False
    # 断言 kb 对 B 的询问结果是 False
    assert kb.ask(B) is False
    # 断言 kb 对 C 的询问结果是 False
    assert kb.ask(C) is False
    # 断言 kb 对 ~A 的询问结果是 False
    assert kb.ask(~A) is False
    # 断言 kb 对 ~B 的询问结果是 False
    assert kb.ask(~B) is False
    # 断言 kb 对 ~C 的询问结果是 False
    assert kb.ask(~C) is False
    # 断言 kb 对 A >> C 的询问结果是 True
    assert kb.ask(A >> C) is True
    # 向知识库 kb 中添加 A 的命题
    kb.tell(A)
    # 断言 kb 对 A 的询问结果是 True
    assert kb.ask(A) is True
    # 断言 kb 对 B 的询问结果是 True
    assert kb.ask(B) is True
    # 断言 kb 对 C 的询问结果是 True
    assert kb.ask(C) is True
    # 断言 kb 对 ~C 的询问结果是 False
    assert kb.ask(~C) is False
    # 从知识库 kb 中撤销 A 的命题
    kb.retract(A)
    # 断言 kb 对 C 的询问结果是 False


def test_propKB_tolerant():
    """"tolerant to bad input"""
    # 创建一个新的命题逻辑知识库
    kb = PropKB()
    # 定义逻辑符号 A, B, C 作为测试符号
    A, B, C = symbols('A,B,C')
    # 断言 kb 对 B 的询问结果是 False


def test_satisfiable_non_symbols():
    # 定义逻辑符号 x, y 作为测试符号
    x, y = symbols('x y')
    # 定义一些假设条件
    assumptions = Q.zero(x*y)
    # 定义一些事实
    facts = Implies(Q.zero(x*y), Q.zero(x) | Q.zero(y))
    # 定义一个查询条件
    query = ~Q.zero(x) & ~Q.zero(y)
    # 定义一组反驳情况
    refutations = [
        {Q.zero(x): True, Q.zero(x*y): True},
        {Q.zero(y): True, Q.zero(x*y): True},
        {Q.zero(x): True, Q.zero(y): True, Q.zero(x*y): True},
        {Q.zero(x): True, Q.zero(y): False, Q.zero(x*y): True},
        {Q.zero(x): False, Q.zero(y): True, Q.zero(x*y): True}]
    # 使用 DPLL 算法断言不可满足性
    assert not satisfiable(And(assumptions, facts, query), algorithm='dpll')
    # 使用 DPLL 算法断言可满足性，结果在反驳集合中
    assert satisfiable(And(assumptions, facts, ~query), algorithm='dpll') in refutations
    # 使用 DPLL2 算法断言不可满足性
    assert not satisfiable(And(assumptions, facts, query), algorithm='dpll2')
    # 使用 DPLL2 算法断言可满足性，结果在反驳集合中
    assert satisfiable(And(assumptions, facts, ~query), algorithm='dpll2') in refutations


def test_satisfiable_bool():
    # 导入 S 单例以进行测试
    from sympy.core.singleton import S
    # 断言对于 true 布尔表达式，可满足性返回 true: true
    assert satisfiable(true) == {true: true}
    # 断言对于 S.true 布尔表达式，可满足性返回 true: true
    assert satisfiable(S.true) == {true: true}
    # 断言对于 false 布尔表达式，可满足性返回 False
    assert satisfiable(false) is False
    # 断言对于 S.false 布尔表达式，可满足性返回 False
    assert satisfiable(S.false) is False


def test_satisfiable_all_models():
    # 导入 A, B 符号以进行测试
    from sympy.abc import A, B
    # 断言对于 False 布尔表达式，所有模型返回 False
    assert next(satisfiable(False, all_models=True)) is False
    # 断言对于 (A >> ~A) & A 布尔表达式，所有模型返回 [False]
    assert list(satisfiable((A >> ~A) & A, all_models=True)) == [False]
    # 断言对于 True 布尔表达式，所有模型返回 [{true: true}]
    assert list(satisfiable(True, all_models=True)) == [{true: true}]

    # 定义一组模型
    models = [{A: True, B: False}, {A: False, B: True}]
    # 断言对于 A ^ B 布尔表达式，所有模型中的两个被移除
    result = satisfiable(A ^ B, all_models=True)
    models.remove(next(result))
    models.remove(next(result))
    # 使用 raises 断言后续模型不存在
    raises(StopIteration, lambda: next(result))
    # 断言模型列表为空
    assert not models

    # 断言对于 Equivalent(A, B) 布尔表达式，所有模型返回 [{A: False, B: False}, {A: True, B: True}]
    assert list(satisfiable(Equivalent(A, B), all_models=True)) == \
    [{A: False, B: False}, {A: True, B: True}]
    # 遍历生成 A → B 的可满足模型，并设置 all_models=True 返回所有模型
    for model in satisfiable(A >> B, all_models=True):
        # 从模型集合中移除当前模型
        models.remove(model)
    # 断言：确保模型集合现在为空
    assert not models

    # 这是一个健全性测试，检查是否只生成了所需数量的解。
    # 表达式 expr 生成了 2**100 - 1 个模型，如果一次性生成所有模型会超时。
    from sympy.utilities.iterables import numbered_symbols
    from sympy.logic.boolalg import Or
    # 创建编号符号生成器
    sym = numbered_symbols()
    # 生成包含 100 个符号的列表 X
    X = [next(sym) for i in range(100)]
    # 计算表达式 Or(*X) 的所有模型
    result = satisfiable(Or(*X), all_models=True)
    # 断言：确保可以连续获取 10 个模型
    for i in range(10):
        assert next(result)
# 定义一个测试函数，用于测试与 Z3 相关的功能
def test_z3():
    # 导入 z3 模块
    z3 = import_module("z3")

    # 如果 z3 模块未导入成功，则跳过测试
    if not z3:
        skip("z3 not installed.")

    # 定义符号变量 A, B, C 和 x, y, z
    A, B, C = symbols('A,B,C')
    x, y, z = symbols('x,y,z')

    # 断言表达式 (x >= 2) & (x < 1) 的可满足性为 False
    assert z3_satisfiable((x >= 2) & (x < 1)) is False

    # 断言布尔表达式 A & ~A 的可满足性为 False
    assert z3_satisfiable( A & ~A ) is False

    # 计算并断言布尔表达式 A & (~A | B | C) 的可满足性，并验证模型的真实性
    model = z3_satisfiable(A & (~A | B | C))
    assert bool(model) is True
    assert model[A] is True

    # 测试非线性函数的表达式 (x ** 2 >= 2) & (x < 1) & (x > -1) 的可满足性为 False
    assert z3_satisfiable((x ** 2 >= 2) & (x < 1) & (x > -1)) is False


# 定义测试函数，比较 Z3 和 LRA DPLL2 算法在随机生成的布尔公式上的结果
def test_z3_vs_lra_dpll2():
    # 导入 z3 模块
    z3 = import_module("z3")
    
    # 如果 z3 模块未导入成功，则跳过测试
    if z3 is None:
        skip("z3 not installed.")

    # 定义函数，将布尔公式转换为编码的 CNF 形式
    def boolean_formula_to_encoded_cnf(bf):
        cnf = CNF.from_prop(bf)
        enc = EncodedCNF()
        enc.from_cnf(cnf)
        return enc

    # 定义函数，生成随机的 CNF 公式
    def make_random_cnf(num_clauses=5, num_constraints=10, num_var=2):
        assert num_clauses <= num_constraints
        constraints = make_random_problem(num_variables=num_var, num_constraints=num_constraints, rational=False)
        clauses = [[cons] for cons in constraints[:num_clauses]]
        for cons in constraints[num_clauses:]:
            if isinstance(cons, Unequality):
                cons = ~cons
            i = randint(0, num_clauses-1)
            clauses[i].append(cons)

        clauses = [Or(*clause) for clause in clauses]
        cnf = And(*clauses)
        return boolean_formula_to_encoded_cnf(cnf)

    # 定义使用 LRA DPLL2 算法判断 CNF 公式可满足性的 lambda 函数
    lra_dpll2_satisfiable = lambda x: dpll2_satisfiable(x, use_lra_theory=True)

    # 进行 50 次随机测试
    for _ in range(50):
        # 生成包含 10 个子句和 15 个约束的随机 CNF 公式
        cnf = make_random_cnf(num_clauses=10, num_constraints=15, num_var=2)

        try:
            # 使用 Z3 判断 CNF 公式的可满足性
            z3_sat = z3_satisfiable(cnf)
        except z3.z3types.Z3Exception:
            continue

        # 使用 LRA DPLL2 算法判断 CNF 公式的可满足性
        lra_dpll2_sat = lra_dpll2_satisfiable(cnf) is not False

        # 断言 Z3 和 LRA DPLL2 的结果一致
        assert z3_sat == lra_dpll2_sat
```