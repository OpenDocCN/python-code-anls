# `D:\src\scipysrc\sympy\sympy\assumptions\facts.py`

```
"""
Known facts in assumptions module.

This module defines the facts between unary predicates in ``get_known_facts()``,
and supports functions to generate the contents in
``sympy.assumptions.ask_generated`` file.
"""

from sympy.assumptions.ask import Q  # 导入 Q 对象，用于定义符号的属性
from sympy.assumptions.assume import AppliedPredicate  # 导入 AppliedPredicate 类，用于表示应用的谓词
from sympy.core.cache import cacheit  # 导入 cacheit 装饰器，用于缓存函数的返回值
from sympy.core.symbol import Symbol  # 导入 Symbol 类，用于创建符号变量
from sympy.logic.boolalg import (to_cnf, And, Not, Implies, Equivalent,  # 导入布尔逻辑相关函数和运算符
    Exclusive,)
from sympy.logic.inference import satisfiable  # 导入 satisfiable 函数，用于判断逻辑推理是否成立


@cacheit
def get_composite_predicates():
    """
    To reduce the complexity of sat solver, these predicates are
    transformed into the combination of primitive predicates.
    """
    return {
        Q.real : Q.negative | Q.zero | Q.positive,
        Q.integer : Q.even | Q.odd,
        Q.nonpositive : Q.negative | Q.zero,
        Q.nonzero : Q.negative | Q.positive,
        Q.nonnegative : Q.zero | Q.positive,
        Q.extended_real : Q.negative_infinite | Q.negative | Q.zero | Q.positive | Q.positive_infinite,
        Q.extended_positive: Q.positive | Q.positive_infinite,
        Q.extended_negative: Q.negative | Q.negative_infinite,
        Q.extended_nonzero: Q.negative_infinite | Q.negative | Q.positive | Q.positive_infinite,
        Q.extended_nonpositive: Q.negative_infinite | Q.negative | Q.zero,
        Q.extended_nonnegative: Q.zero | Q.positive | Q.positive_infinite,
        Q.complex : Q.algebraic | Q.transcendental
    }


@cacheit
def get_known_facts(x=None):
    """
    Facts between unary predicates.

    Parameters
    ==========

    x : Symbol, optional
        Placeholder symbol for unary facts. Default is ``Symbol('x')``.

    Returns
    =======

    fact : Known facts in conjugated normal form.

    """
    if x is None:
        x = Symbol('x')

    # 组合已知事实为合取范式（CNF）
    fact = And(
        get_number_facts(x),
        get_matrix_facts(x)
    )
    return fact


@cacheit
def get_number_facts(x = None):
    """
    Facts between unary number predicates.

    Parameters
    ==========

    x : Symbol, optional
        Placeholder symbol for unary facts. Default is ``Symbol('x')``.

    Returns
    =======

    fact : Known facts in conjugated normal form.

    """
    if x is None:
        x = Symbol('x')

    # 此处应有进一步的代码，用于获取与数值相关的已知事实，暂未提供
    # 定义一个逻辑表达式 `fact`，包含多个逻辑断言和推论

    fact = And(
        # 对扩展实数进行排他性判断
        Exclusive(Q.negative_infinite(x), Q.negative(x), Q.zero(x),
            Q.positive(x), Q.positive_infinite(x)),

        # 构建复平面
        Exclusive(Q.real(x), Q.imaginary(x)),
        Implies(Q.real(x) | Q.imaginary(x), Q.complex(x)),

        # 复数的其他子集
        Exclusive(Q.transcendental(x), Q.algebraic(x)),
        Equivalent(Q.real(x), Q.rational(x) | Q.irrational(x)),
        Exclusive(Q.irrational(x), Q.rational(x)),
        Implies(Q.rational(x), Q.algebraic(x)),

        # 整数
        Exclusive(Q.even(x), Q.odd(x)),
        Implies(Q.integer(x), Q.rational(x)),
        Implies(Q.zero(x), Q.even(x)),
        Exclusive(Q.composite(x), Q.prime(x)),
        Implies(Q.composite(x) | Q.prime(x), Q.integer(x) & Q.positive(x)),
        Implies(Q.even(x) & Q.positive(x) & ~Q.prime(x), Q.composite(x)),

        # Hermite 对称和反对称
        Implies(Q.real(x), Q.hermitian(x)),
        Implies(Q.imaginary(x), Q.antihermitian(x)),
        Implies(Q.zero(x), Q.hermitian(x) | Q.antihermitian(x)),

        # 定义有限性和无限性，并构建扩展实数线
        Exclusive(Q.infinite(x), Q.finite(x)),
        Implies(Q.complex(x), Q.finite(x)),
        Implies(Q.negative_infinite(x) | Q.positive_infinite(x), Q.infinite(x)),

        # 可交换性
        Implies(Q.finite(x) | Q.infinite(x), Q.commutative(x)),
    )
    # 返回逻辑表达式 `fact`
    return fact
@cacheit
# 使用装饰器 @cacheit 来缓存函数结果，避免重复计算
def get_matrix_facts(x = None):
    """
    Facts between unary matrix predicates.

    Parameters
    ==========

    x : Symbol, optional
        Placeholder symbol for unary facts. Default is ``Symbol('x')``.

    Returns
    =======

    fact : Known facts in conjugated normal form.

    """
    if x is None:
        x = Symbol('x')

    # 构建关于矩阵谓词之间关系的已知事实，以合取范式形式返回
    fact = And(
        # 矩阵
        Implies(Q.orthogonal(x), Q.positive_definite(x)),
        Implies(Q.orthogonal(x), Q.unitary(x)),
        Implies(Q.unitary(x) & Q.real_elements(x), Q.orthogonal(x)),
        Implies(Q.unitary(x), Q.normal(x)),
        Implies(Q.unitary(x), Q.invertible(x)),
        Implies(Q.normal(x), Q.square(x)),
        Implies(Q.diagonal(x), Q.normal(x)),
        Implies(Q.positive_definite(x), Q.invertible(x)),
        Implies(Q.diagonal(x), Q.upper_triangular(x)),
        Implies(Q.diagonal(x), Q.lower_triangular(x)),
        Implies(Q.lower_triangular(x), Q.triangular(x)),
        Implies(Q.upper_triangular(x), Q.triangular(x)),
        Implies(Q.triangular(x), Q.upper_triangular(x) | Q.lower_triangular(x)),
        Implies(Q.upper_triangular(x) & Q.lower_triangular(x), Q.diagonal(x)),
        Implies(Q.diagonal(x), Q.symmetric(x)),
        Implies(Q.unit_triangular(x), Q.triangular(x)),
        Implies(Q.invertible(x), Q.fullrank(x)),
        Implies(Q.invertible(x), Q.square(x)),
        Implies(Q.symmetric(x), Q.square(x)),
        Implies(Q.fullrank(x) & Q.square(x), Q.invertible(x)),
        Equivalent(Q.invertible(x), ~Q.singular(x)),
        Implies(Q.integer_elements(x), Q.real_elements(x)),
        Implies(Q.real_elements(x), Q.complex_elements(x)),
    )
    return fact


def generate_known_facts_dict(keys, fact):
    """
    Computes and returns a dictionary which contains the relations between
    unary predicates.

    Each key is a predicate, and item is two groups of predicates.
    First group contains the predicates which are implied by the key, and
    second group contains the predicates which are rejected by the key.

    All predicates in *keys* and *fact* must be unary and have same placeholder
    symbol.

    Parameters
    ==========

    keys : list of AppliedPredicate instances.
        一组应用谓词的实例列表，用于构建字典的键。

    fact : Fact between predicates in conjugated normal form.
        谓词之间的事实，以合取范式形式给出。

    Examples
    ========

    >>> from sympy import Q, And, Implies
    >>> from sympy.assumptions.facts import generate_known_facts_dict
    >>> from sympy.abc import x
    >>> keys = [Q.even(x), Q.odd(x), Q.zero(x)]
    >>> fact = And(Implies(Q.even(x), ~Q.odd(x)),
    ...     Implies(Q.zero(x), Q.even(x)))
    >>> generate_known_facts_dict(keys, fact)
    {Q.even: ({Q.even}, {Q.odd}),
     Q.odd: ({Q.odd}, {Q.even, Q.zero}),
     Q.zero: ({Q.even, Q.zero}, {Q.odd})}
    """
    # 将事实转化为合取范式
    fact_cnf = to_cnf(fact)
    # 使用单一事实查找函数构建映射
    mapping = single_fact_lookup(keys, fact_cnf)

    ret = {}
    for key, value in mapping.items():
        # 遍历映射中的每对键值对，key 是键，value 是值（可能是列表或其他可迭代对象）
        implied = set()
        # 初始化一个空集合 implied，用于存储推断出的函数集合
        rejected = set()
        # 初始化一个空集合 rejected，用于存储被拒绝的函数集合
        for expr in value:
            # 遍历值 value 中的每个表达式 expr
            if isinstance(expr, AppliedPredicate):
                # 如果 expr 是 AppliedPredicate 类型的对象
                implied.add(expr.function)
                # 将 expr 的函数添加到 implied 集合中，表示该函数是被推断出来的
            elif isinstance(expr, Not):
                # 如果 expr 是 Not 类型的对象
                pred = expr.args[0]
                # 获取 expr 的第一个参数，即被否定的谓词
                rejected.add(pred.function)
                # 将被否定的谓词的函数添加到 rejected 集合中，表示该函数被拒绝了
        ret[key.function] = (implied, rejected)
        # 将 key 对应的函数作为 ret 字典的键，将 (implied, rejected) 元组作为对应的值存入 ret 中
    return ret
    # 返回 ret 字典，其中包含了每个 key 对应的推断和拒绝的函数集合
# 使用装饰器 @cacheit 修饰的函数，用于获取所有注册到 Q 的一元谓词的键集合
@cacheit
def get_known_facts_keys():
    """
    Return every unary predicates registered to ``Q``.

    This function is used to generate the keys for
    ``generate_known_facts_dict``.
    """
    # 要排除的多元谓词集合
    exclude = {Q.eq, Q.ne, Q.gt, Q.lt, Q.ge, Q.le}

    # 存储结果的列表
    result = []
    # 遍历 Q 类中的所有属性
    for attr in Q.__class__.__dict__:
        # 忽略以双下划线开头的属性
        if attr.startswith('__'):
            continue
        # 获取属性对应的谓词
        pred = getattr(Q, attr)
        # 如果谓词在排除集合中，则跳过
        if pred in exclude:
            continue
        # 将谓词添加到结果列表中
        result.append(pred)
    return result


def single_fact_lookup(known_facts_keys, known_facts_cnf):
    # 返回用于快速查找单个事实的字典
    mapping = {}
    # 遍历已知事实的键集合
    for key in known_facts_keys:
        # 初始化键对应的集合，并添加自身作为初始元素
        mapping[key] = {key}
        # 遍历其他已知事实的键集合
        for other_key in known_facts_keys:
            # 如果不是同一个键
            if other_key != key:
                # 进行全面推理，检查键 other_key 对 key 的影响
                if ask_full_inference(other_key, key, known_facts_cnf):
                    mapping[key].add(other_key)
                # 进行全面推理，检查键 ~other_key 对 key 的影响
                if ask_full_inference(~other_key, key, known_facts_cnf):
                    mapping[key].add(~other_key)
    return mapping


def ask_full_inference(proposition, assumptions, known_facts_cnf):
    """
    Method for inferring properties about objects.
    """
    # 如果已知事实的合取与假设以及命题不可满足，则返回 False
    if not satisfiable(And(known_facts_cnf, assumptions, proposition)):
        return False
    # 如果已知事实的合取与假设以及命题的否定不可满足，则返回 True
    if not satisfiable(And(known_facts_cnf, assumptions, Not(proposition))):
        return True
    # 否则返回 None
    return None
```