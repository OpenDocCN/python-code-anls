# `D:\src\scipysrc\sympy\sympy\logic\algorithms\dpll.py`

```
"""Implementation of DPLL algorithm

Further improvements: eliminate calls to pl_true, implement branching rules,
efficient unit propagation.

References:
  - https://en.wikipedia.org/wiki/DPLL_algorithm
  - https://www.researchgate.net/publication/242384772_Implementations_of_the_DPLL_Algorithm
"""

# 导入所需的模块和函数
from sympy.core.sorting import default_sort_key
from sympy.logic.boolalg import Or, Not, conjuncts, disjuncts, to_cnf, \
    to_int_repr, _find_predicates
from sympy.assumptions.cnf import CNF
from sympy.logic.inference import pl_true, literal_symbol


def dpll_satisfiable(expr):
    """
    Check satisfiability of a propositional sentence.
    It returns a model rather than True when it succeeds

    >>> from sympy.abc import A, B
    >>> from sympy.logic.algorithms.dpll import dpll_satisfiable
    >>> dpll_satisfiable(A & ~B)
    {A: True, B: False}
    >>> dpll_satisfiable(A & ~A)
    False

    """
    # 如果表达式不是 CNF 格式，则转换为 CNF
    if not isinstance(expr, CNF):
        clauses = conjuncts(to_cnf(expr))
    else:
        clauses = expr.clauses
    # 检查是否存在 False 的子句，若有则返回 False
    if False in clauses:
        return False
    # 对符号进行排序，并转换为整数表示
    symbols = sorted(_find_predicates(expr), key=default_sort_key)
    symbols_int_repr = set(range(1, len(symbols) + 1))
    clauses_int_repr = to_int_repr(clauses, symbols)
    # 调用 dpll_int_repr 函数进行 DPLL 算法求解
    result = dpll_int_repr(clauses_int_repr, symbols_int_repr, {})
    # 如果没有解，则返回空
    if not result:
        return result
    # 将整数表示的结果转换为符号表示的模型
    output = {}
    for key in result:
        output.update({symbols[key - 1]: result[key]})
    return output


def dpll(clauses, symbols, model):
    """
    Compute satisfiability in a partial model.
    Clauses is an array of conjuncts.

    >>> from sympy.abc import A, B, D
    >>> from sympy.logic.algorithms.dpll import dpll
    >>> dpll([A, B, D], [A, B], {D: False})
    False

    """
    # 计算 DP 核心
    P, value = find_unit_clause(clauses, model)
    while P:
        model.update({P: value})
        symbols.remove(P)
        if not value:
            P = ~P
        clauses = unit_propagate(clauses, P)
        P, value = find_unit_clause(clauses, model)
    P, value = find_pure_symbol(symbols, clauses)
    while P:
        model.update({P: value})
        symbols.remove(P)
        if not value:
            P = ~P
        clauses = unit_propagate(clauses, P)
        P, value = find_pure_symbol(symbols, clauses)
    # 结束 DP 核心
    unknown_clauses = []
    # 对每个子句进行真值判断
    for c in clauses:
        val = pl_true(c, model)
        if val is False:
            return False
        if val is not True:
            unknown_clauses.append(c)
    # 若所有子句都满足，则返回模型
    if not unknown_clauses:
        return model
    # 如果还有未解决的子句，则尝试递归解决
    if not clauses:
        return model
    P = symbols.pop()
    model_copy = model.copy()
    model.update({P: True})
    model_copy.update({P: False})
    symbols_copy = symbols[:]
    return (dpll(unit_propagate(unknown_clauses, P), symbols, model) or
            dpll(unit_propagate(unknown_clauses, Not(P)), symbols_copy, model_copy))


def dpll_int_repr(clauses, symbols, model):
    """
    Internal function for DPLL algorithm with integer representation of symbols.

    """
    # 此处为 DPLL 算法的具体实现，未提供完整代码，需要继续实现
    Compute satisfiability in a partial model.
    Arguments are expected to be in integer representation

    >>> from sympy.logic.algorithms.dpll import dpll_int_repr
    >>> dpll_int_repr([{1}, {2}, {3}], {1, 2}, {3: False})
    False

    """
    # 计算 DPLL 算法的核心部分

    # 找到单元子句并返回
    P, value = find_unit_clause_int_repr(clauses, model)
    while P:
        # 更新模型
        model.update({P: value})
        # 从符号集合中移除 P
        symbols.remove(P)
        # 如果 value 是 False，则取 P 的相反数
        if not value:
            P = -P
        # 单元传播
        clauses = unit_propagate_int_repr(clauses, P)
        # 继续找下一个单元子句
        P, value = find_unit_clause_int_repr(clauses, model)

    # 找到纯符号并返回
    P, value = find_pure_symbol_int_repr(symbols, clauses)
    while P:
        # 更新模型
        model.update({P: value})
        # 从符号集合中移除 P
        symbols.remove(P)
        # 如果 value 是 False，则取 P 的相反数
        if not value:
            P = -P
        # 单元传播
        clauses = unit_propagate_int_repr(clauses, P)
        # 继续找下一个纯符号
        P, value = find_pure_symbol_int_repr(symbols, clauses)

    # 处理未知子句
    unknown_clauses = []
    for c in clauses:
        # 判断子句的真值
        val = pl_true_int_repr(c, model)
        if val is False:
            return False
        if val is not True:
            unknown_clauses.append(c)

    # 如果所有子句都被满足，则返回模型
    if not unknown_clauses:
        return model

    # 选择一个符号 P 并进行递归调用
    P = symbols.pop()
    model_copy = model.copy()
    model.update({P: True})
    model_copy.update({P: False})
    symbols_copy = symbols.copy()

    # 递归调用 DPLL 算法
    return (dpll_int_repr(unit_propagate_int_repr(unknown_clauses, P), symbols, model) or
            dpll_int_repr(unit_propagate_int_repr(unknown_clauses, -P), symbols_copy, model_copy))
# pl_true_int_repr 函数用于判断给定子句中的文字是否为真，根据模型给出 True、False 或 None 的结果
def pl_true_int_repr(clause, model={}):
    result = False  # 初始化结果为 False
    for lit in clause:  # 遍历子句中的每个文字
        if lit < 0:  # 如果文字是负数
            p = model.get(-lit)  # 获取其在模型中的取值
            if p is not None:
                p = not p  # 如果取值存在，则取其相反值
        else:  # 如果文字是正数
            p = model.get(lit)  # 获取其在模型中的取值
        if p is True:  # 如果取值为 True，则返回 True
            return True
        elif p is None:  # 如果取值为 None，则将结果设为 None
            result = None
    return result  # 返回判断结果


# unit_propagate 函数用于单元传播，简化 CNF 子句集合中的内容
def unit_propagate(clauses, symbol):
    output = []  # 初始化输出列表
    for c in clauses:  # 遍历每个子句
        if c.func != Or:  # 如果子句不是 Or 运算
            output.append(c)  # 直接将子句加入输出列表
            continue
        for arg in c.args:  # 遍历子句中的每个参数
            if arg == ~symbol:  # 如果参数是 symbol 的否定
                output.append(Or(*[x for x in c.args if x != ~symbol]))  # 删除包含 ~symbol 的部分
                break
            if arg == symbol:  # 如果参数是 symbol
                break  # 不再处理此子句，跳出循环
        else:  # 如果没有 break 被执行
            output.append(c)  # 将未变化的子句加入输出列表
    return output  # 返回简化后的子句集合


# unit_propagate_int_repr 函数与 unit_propagate 类似，但处理的是整数表示的参数
def unit_propagate_int_repr(clauses, s):
    negated = {-s}  # 计算 s 的否定
    return [clause - negated for clause in clauses if s not in clause]  # 返回经过单元传播简化后的子句集合


# find_pure_symbol 函数用于查找纯符号，即在未知子句中仅作为正文字或负文字出现的符号
def find_pure_symbol(symbols, unknown_clauses):
    for sym in symbols:  # 遍历每个符号
        found_pos, found_neg = False, False  # 初始化是否找到正负文字的标志为 False
        for c in unknown_clauses:  # 遍历未知子句
            if not found_pos and sym in disjuncts(c):  # 如果未找到正文字且符号在子句中作为正文字
                found_pos = True  # 标记找到正文字
            if not found_neg and Not(sym) in disjuncts(c):  # 如果未找到负文字且符号在子句中作为负文字
                found_neg = True  # 标记找到负文字
        if found_pos != found_neg:  # 如果只找到了正文字或只找到了负文字
            return sym, found_pos  # 返回符号及其取值
    return None, None  # 如果未找到纯符号，则返回 None
    

# find_pure_symbol_int_repr 函数与 find_pure_symbol 类似，但处理的是整数表示的参数
def find_pure_symbol_int_repr(symbols, unknown_clauses):
    # 函数体未提供，未能提供注释。
    pass
    # 导入从 sympy.logic.algorithms.dpll 模块中导入 find_pure_symbol_int_repr 函数
    >>> from sympy.logic.algorithms.dpll import find_pure_symbol_int_repr
    # 调用 find_pure_symbol_int_repr 函数，传入参数为 symbols 和 unknown_clauses
    >>> find_pure_symbol_int_repr({1,2,3},
    ...     [{1, -2}, {-2, -3}, {3, 1}])
    # 返回值为 (1, True)

    """
    # 将 unknown_clauses 中所有子集的并集作为 all_symbols
    all_symbols = set().union(*unknown_clauses)
    # 找到 all_symbols 中与 symbols 相交的正数符号
    found_pos = all_symbols.intersection(symbols)
    # 找到 all_symbols 中与 symbols 相交的负数符号
    found_neg = all_symbols.intersection([-s for s in symbols])
    # 遍历 found_pos 中的符号
    for p in found_pos:
        # 如果其相反数不在 found_neg 中，则返回该符号 p 和 True
        if -p not in found_neg:
            return p, True
    # 遍历 found_neg 中的符号
    for p in found_neg:
        # 如果其相反数不在 found_pos 中，则返回该符号的相反数 -p 和 False
        if -p not in found_pos:
            return -p, False
    # 如果未找到纯符号，则返回 None, None
    return None, None
# 寻找单位子句的函数，单位子句指只有一个在模型中未绑定的变量

def find_unit_clause(clauses, model):
    """
    A unit clause has only 1 variable that is not bound in the model.
    
    >>> from sympy.abc import A, B, D
    >>> from sympy.logic.algorithms.dpll import find_unit_clause
    >>> find_unit_clause([A | B | D, B | ~D, A | ~B], {A:True})
    (B, False)
    
    """
    # 遍历每个子句
    for clause in clauses:
        num_not_in_model = 0
        # 遍历子句中的每个文字
        for literal in disjuncts(clause):
            sym = literal_symbol(literal)
            # 检查文字对应的符号是否不在模型中
            if sym not in model:
                num_not_in_model += 1
                P, value = sym, not isinstance(literal, Not)
        # 如果子句中只有一个符号不在模型中，则返回该符号和对应的值
        if num_not_in_model == 1:
            return P, value
    # 如果没有找到单位子句，则返回 None
    return None, None


# 使用整数表示寻找单位子句的函数，参数预期为整数表示

def find_unit_clause_int_repr(clauses, model):
    """
    Same as find_unit_clause, but arguments are expected to be in
    integer representation.
    
    >>> from sympy.logic.algorithms.dpll import find_unit_clause_int_repr
    >>> find_unit_clause_int_repr([{1, 2, 3},
    ...     {2, -3}, {1, -2}], {1: True})
    (2, False)
    
    """
    # 计算所有已绑定的变量
    bound = set(model) | {-sym for sym in model}
    # 遍历每个子句
    for clause in clauses:
        # 找出未绑定的变量
        unbound = clause - bound
        # 如果子句中只有一个未绑定的变量，则返回该变量和对应的值
        if len(unbound) == 1:
            p = unbound.pop()
            if p < 0:
                return -p, False
            else:
                return p, True
    # 如果没有找到单位子句，则返回 None
    return None, None
```