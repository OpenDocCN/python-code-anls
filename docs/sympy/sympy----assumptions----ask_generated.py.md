# `D:\src\scipysrc\sympy\sympy\assumptions\ask_generated.py`

```
"""
Do NOT manually edit this file.
Instead, run ./bin/ask_update.py.
"""

# 导入所需模块和函数
from sympy.assumptions.ask import Q
from sympy.assumptions.cnf import Literal
from sympy.core.cache import cacheit

@cacheit
def get_all_known_facts():
    """
    Known facts between unary predicates as CNF clauses.
    """
    # 返回一个空字典，用于缓存已知的一元谓词间的CNF子句
    return {}

@cacheit
def get_all_known_matrix_facts():
    """
    Known facts between unary predicates for matrices as CNF clauses.
    """
    # 返回一个包含多个 frozenset 的集合，每个 frozenset 代表一条已知的矩阵一元谓词间的CNF子句
    return {
        frozenset((Literal(Q.complex_elements, False), Literal(Q.real_elements, True))),
        frozenset((Literal(Q.diagonal, False), Literal(Q.lower_triangular, True), Literal(Q.upper_triangular, True))),
        frozenset((Literal(Q.diagonal, True), Literal(Q.lower_triangular, False))),
        frozenset((Literal(Q.diagonal, True), Literal(Q.normal, False))),
        frozenset((Literal(Q.diagonal, True), Literal(Q.symmetric, False))),
        frozenset((Literal(Q.diagonal, True), Literal(Q.upper_triangular, False))),
        frozenset((Literal(Q.fullrank, False), Literal(Q.invertible, True))),
        frozenset((Literal(Q.fullrank, True), Literal(Q.invertible, False), Literal(Q.square, True))),
        frozenset((Literal(Q.integer_elements, True), Literal(Q.real_elements, False))),
        frozenset((Literal(Q.invertible, False), Literal(Q.positive_definite, True))),
        frozenset((Literal(Q.invertible, False), Literal(Q.singular, False))),
        frozenset((Literal(Q.invertible, False), Literal(Q.unitary, True))),
        frozenset((Literal(Q.invertible, True), Literal(Q.singular, True))),
        frozenset((Literal(Q.invertible, True), Literal(Q.square, False))),
        frozenset((Literal(Q.lower_triangular, False), Literal(Q.triangular, True), Literal(Q.upper_triangular, False))),
        frozenset((Literal(Q.lower_triangular, True), Literal(Q.triangular, False))),
        frozenset((Literal(Q.normal, False), Literal(Q.unitary, True))),
        frozenset((Literal(Q.normal, True), Literal(Q.square, False))),
        frozenset((Literal(Q.orthogonal, False), Literal(Q.real_elements, True), Literal(Q.unitary, True))),
        frozenset((Literal(Q.orthogonal, True), Literal(Q.positive_definite, False))),
        frozenset((Literal(Q.orthogonal, True), Literal(Q.unitary, False))),
        frozenset((Literal(Q.square, False), Literal(Q.symmetric, True))),
        frozenset((Literal(Q.triangular, False), Literal(Q.unit_triangular, True))),
        frozenset((Literal(Q.triangular, False), Literal(Q.upper_triangular, True)))
    }

@cacheit
def get_all_known_number_facts():
    """
    Known facts between unary predicates for numbers as CNF clauses.
    """
    # 返回一个空字典，用于缓存已知的数字一元谓词间的CNF子句
    return {}

@cacheit
def get_known_facts_dict():
    """
    Logical relations between unary predicates as dictionary.

    Each key is a predicate, and item is two groups of predicates.
    First group contains the predicates which are implied by the key, and
    second group contains the predicates which are rejected by the key.

    """
    # 返回一个空字典，用于缓存一元谓词间的逻辑关系
    return {}
```