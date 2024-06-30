# `D:\src\scipysrc\sympy\sympy\logic\inference.py`

```
# 导入从Sympy库中需要的模块和函数
from sympy.logic.boolalg import And, Not, conjuncts, to_cnf, BooleanFunction
from sympy.core.sorting import ordered
from sympy.core.sympify import sympify
from sympy.external.importtools import import_module

# 定义函数literal_symbol，用于提取布尔表达式中的符号（不包括否定）
def literal_symbol(literal):
    """
    The symbol in this literal (without the negation).

    Examples
    ========

    >>> from sympy.abc import A
    >>> from sympy.logic.inference import literal_symbol
    >>> literal_symbol(A)
    A
    >>> literal_symbol(~A)
    A

    """
    # 如果literal为True或False，则直接返回literal本身
    if literal is True or literal is False:
        return literal
    # 如果literal是符号（变量），直接返回literal
    elif literal.is_Symbol:
        return literal
    # 如果literal是否定，递归调用literal_symbol函数获取其内部的符号
    elif literal.is_Not:
        return literal_symbol(literal.args[0])
    else:
        # 如果literal不是合法的布尔文字面值，抛出ValueError异常
        raise ValueError("Argument must be a boolean literal.")

# 定义函数satisfiable，用于检查命题逻辑句子的可满足性
# 返回满足条件的模型，或对于显然为真的表达式返回{true: true}
# 当all_models设置为True时，如果表达式可满足则返回模型的生成器，否则返回包含单个元素False的生成器
def satisfiable(expr, algorithm=None, all_models=False, minimal=False, use_lra_theory=False):
    """
    Check satisfiability of a propositional sentence.
    Returns a model when it succeeds.
    Returns {true: true} for trivially true expressions.

    On setting all_models to True, if given expr is satisfiable then
    returns a generator of models. However, if expr is unsatisfiable
    then returns a generator containing the single element False.

    Examples
    ========

    >>> from sympy.abc import A, B
    >>> from sympy.logic.inference import satisfiable
    >>> satisfiable(A & ~B)
    {A: True, B: False}
    >>> satisfiable(A & ~A)
    False
    >>> satisfiable(True)
    {True: True}
    >>> next(satisfiable(A & ~A, all_models=True))
    False
    >>> models = satisfiable((A >> B) & B, all_models=True)
    >>> next(models)
    {A: False, B: True}
    >>> next(models)
    {A: True, B: True}
    >>> def use_models(models):
    ...     for model in models:
    ...         if model:
    ...             # Do something with the model.
    ...             print(model)
    ...         else:
    ...             # Given expr is unsatisfiable.
    ...             print("UNSAT")
    >>> use_models(satisfiable(A >> ~A, all_models=True))
    {A: False}
    >>> use_models(satisfiable(A ^ A, all_models=True))
    UNSAT

    """
    # 如果使用LRA理论，则只能使用dpll2算法，否则抛出异常
    if use_lra_theory:
        if algorithm is not None and algorithm != "dpll2":
            raise ValueError(f"Currently only dpll2 can handle using lra theory. {algorithm} is not handled.")
        algorithm = "dpll2"

    # 如果算法未指定或为"pycosat"，尝试导入pycosat模块并设置算法为"pycosat"，否则使用"dpll2"
    if algorithm is None or algorithm == "pycosat":
        pycosat = import_module('pycosat')
        if pycosat is not None:
            algorithm = "pycosat"
        else:
            if algorithm == "pycosat":
                raise ImportError("pycosat module is not present")
            # 如果没有安装pycosat模块，则静默地回退到"dpll2"
            algorithm = "dpll2"

    # 如果算法为"minisat22"，尝试导入pysat模块，如果失败则改用"dpll2"
    if algorithm=="minisat22":
        pysat = import_module('pysat')
        if pysat is None:
            algorithm = "dpll2"
    # 如果算法选择为 "z3"，则尝试导入 z3 模块
    if algorithm=="z3":
        z3 = import_module('z3')
        # 如果导入失败，则将算法切换至 "dpll2"
        if z3 is None:
            algorithm = "dpll2"

    # 根据选择的算法执行相应的 SAT 求解器
    if algorithm == "dpll":
        # 导入 sympy 库中的 DPLL 算法并执行求解
        from sympy.logic.algorithms.dpll import dpll_satisfiable
        return dpll_satisfiable(expr)
    elif algorithm == "dpll2":
        # 导入 sympy 库中的 DPLL2 算法并执行求解
        from sympy.logic.algorithms.dpll2 import dpll_satisfiable
        return dpll_satisfiable(expr, all_models, use_lra_theory=use_lra_theory)
    elif algorithm == "pycosat":
        # 导入 sympy 库中的 pycosat 包装器并执行求解
        from sympy.logic.algorithms.pycosat_wrapper import pycosat_satisfiable
        return pycosat_satisfiable(expr, all_models)
    elif algorithm == "minisat22":
        # 导入 sympy 库中的 minisat22 包装器并执行求解
        from sympy.logic.algorithms.minisat22_wrapper import minisat22_satisfiable
        return minisat22_satisfiable(expr, all_models, minimal)
    elif algorithm == "z3":
        # 导入 sympy 库中的 z3 包装器并执行求解
        from sympy.logic.algorithms.z3_wrapper import z3_satisfiable
        return z3_satisfiable(expr, all_models)

    # 如果没有匹配的算法，则抛出 NotImplementedError 异常
    raise NotImplementedError
# 判断命题句是否有效的函数
def valid(expr):
    # 返回非可满足性的结果
    return not satisfiable(Not(expr))


# 判断给定的赋值是否是模型的函数
def pl_true(expr, model=None, deep=False):
    # 导入符号类 Symbol
    from sympy.core.symbol import Symbol

    # 布尔值常量 True 和 False
    boolean = (True, False)

    # 内部函数，验证表达式的有效性
    def _validate(expr):
        # 如果是符号或布尔值，返回 True
        if isinstance(expr, Symbol) or expr in boolean:
            return True
        # 如果不是布尔函数，返回 False
        if not isinstance(expr, BooleanFunction):
            return False
        # 递归验证所有参数
        return all(_validate(arg) for arg in expr.args)

    # 如果表达式是布尔值，直接返回它
    if expr in boolean:
        return expr
    # 将表达式转化为符号表达式
    expr = sympify(expr)
    # 如果表达式不合法，抛出 ValueError 异常
    if not _validate(expr):
        raise ValueError("%s is not a valid boolean expression" % expr)
    # 如果模型为空，则设为默认空字典
    if not model:
        model = {}
    # 过滤模型中的无效布尔值
    model = {k: v for k, v in model.items() if v in boolean}
    # 对表达式进行模型替换
    result = expr.subs(model)
    # 如果结果为布尔值，返回其布尔值
    if result in boolean:
        return bool(result)
    # 如果 deep 参数为 True，则进行深度验证
    if deep:
        # 从结果的原子集合创建模型字典
        model = dict.fromkeys(result.atoms(), True)
        # 如果模型中的结果有效且整体有效，则返回 True
        if pl_true(result, model):
            if valid(result):
                return True
        else:
            # 如果结果不可满足，则返回 False
            if not satisfiable(result):
                return False
    # 返回 None 表示不确定结果
    return None


# 判断是否推出的函数
def entails(expr, formula_set=None):
    # 如果公式集合为空，则返回表达式的有效性
    if formula_set is None:
        return valid(expr)
    # 未实现推出关系的验证，因此返回 None
    return None
    """
    如果 formula_set 不为空（即存在逻辑公式集合）：
        将其转换为列表形式
    否则（如果 formula_set 为空）：
        初始化 formula_set 为空列表
    将逻辑表达式 expr 的否定形式添加到 formula_set 中
    返回将 formula_set 视为 And 逻辑连接的不可满足性结果的否定
    """
    if formula_set:
        formula_set = list(formula_set)
    else:
        formula_set = []
    formula_set.append(Not(expr))
    return not satisfiable(And(*formula_set))
class KB:
    """Base class for all knowledge bases"""
    # 初始化函数，创建一个空的子句集合
    def __init__(self, sentence=None):
        self.clauses_ = set()
        # 如果提供了句子作为参数，则调用tell方法将其添加到知识库中
        if sentence:
            self.tell(sentence)

    # 添加句子的抽象方法，子类需要实现具体逻辑
    def tell(self, sentence):
        raise NotImplementedError

    # 查询的抽象方法，子类需要实现具体逻辑
    def ask(self, query):
        raise NotImplementedError

    # 撤销句子的抽象方法，子类需要实现具体逻辑
    def retract(self, sentence):
        raise NotImplementedError

    # 属性方法，返回有序的子句列表
    @property
    def clauses(self):
        return list(ordered(self.clauses_))


class PropKB(KB):
    """A KB for Propositional Logic.  Inefficient, with no indexing."""

    # 实现tell方法，将句子转换为合取范式，并将其子句添加到子句集合中
    def tell(self, sentence):
        """Add the sentence's clauses to the KB

        Examples
        ========

        >>> from sympy.logic.inference import PropKB
        >>> from sympy.abc import x, y
        >>> l = PropKB()
        >>> l.clauses
        []

        >>> l.tell(x | y)
        >>> l.clauses
        [x | y]

        >>> l.tell(y)
        >>> l.clauses
        [y, x | y]

        """
        # 将句子转换为合取范式的子句，并逐个添加到子句集合中
        for c in conjuncts(to_cnf(sentence)):
            self.clauses_.add(c)

    # 实现ask方法，检查查询是否在子句集合中成立
    def ask(self, query):
        """Checks if the query is true given the set of clauses.

        Examples
        ========

        >>> from sympy.logic.inference import PropKB
        >>> from sympy.abc import x, y
        >>> l = PropKB()
        >>> l.tell(x & ~y)
        >>> l.ask(x)
        True
        >>> l.ask(y)
        False

        """
        # 判断查询是否根据子句集合成立
        return entails(query, self.clauses_)

    # 实现retract方法，从子句集合中移除句子的子句
    def retract(self, sentence):
        """Remove the sentence's clauses from the KB

        Examples
        ========

        >>> from sympy.logic.inference import PropKB
        >>> from sympy.abc import x, y
        >>> l = PropKB()
        >>> l.clauses
        []

        >>> l.tell(x | y)
        >>> l.clauses
        [x | y]

        >>> l.retract(x | y)
        >>> l.clauses
        []

        """
        # 将句子转换为合取范式的子句，并逐个从子句集合中移除
        for c in conjuncts(to_cnf(sentence)):
            self.clauses_.discard(c)
```