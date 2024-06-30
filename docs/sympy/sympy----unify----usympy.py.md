# `D:\src\scipysrc\sympy\sympy\unify\usympy.py`

```
""" SymPy interface to Unification engine

See sympy.unify for module level docstring
See sympy.unify.core for algorithmic docstring """

# 导入必要的类和函数
from sympy.core import Basic, Add, Mul, Pow
from sympy.core.operations import AssocOp, LatticeOp
from sympy.matrices import MatAdd, MatMul, MatrixExpr
from sympy.sets.sets import Union, Intersection, FiniteSet
from sympy.unify.core import Compound, Variable, CondVariable
from sympy.unify import core

# 合法的新基本类型
basic_new_legal = [MatrixExpr]
# 允许使用 evaluate=False 的操作类型
eval_false_legal = [AssocOp, Pow, FiniteSet]
# 不允许的操作类型
illegal = [LatticeOp]

# 检查操作符是否是关联的
def sympy_associative(op):
    assoc_ops = (AssocOp, MatAdd, MatMul, Union, Intersection, FiniteSet)
    return any(issubclass(op, aop) for aop in assoc_ops)

# 检查操作符是否是可交换的
def sympy_commutative(op):
    comm_ops = (Add, MatAdd, Union, Intersection, FiniteSet)
    return any(issubclass(op, cop) for cop in comm_ops)

# 检查对象是否是关联的 Compound
def is_associative(x):
    return isinstance(x, Compound) and sympy_associative(x.op)

# 检查对象是否是可交换的 Compound
def is_commutative(x):
    if not isinstance(x, Compound):
        return False
    if sympy_commutative(x.op):
        return True
    if issubclass(x.op, Mul):
        return all(construct(arg).is_commutative for arg in x.args)

# 根据类型创建匹配函数
def mk_matchtype(typ):
    def matchtype(x):
        return (isinstance(x, typ) or
                isinstance(x, Compound) and issubclass(x.op, typ))
    return matchtype

# 将 SymPy 对象转换为 Compound
def deconstruct(s, variables=()):
    """ Turn a SymPy object into a Compound """
    if s in variables:
        return Variable(s)
    if isinstance(s, (Variable, CondVariable)):
        return s
    if not isinstance(s, Basic) or s.is_Atom:
        return s
    return Compound(s.__class__,
                    tuple(deconstruct(arg, variables) for arg in s.args))

# 将 Compound 转换为 SymPy 对象
def construct(t):
    """ Turn a Compound into a SymPy object """
    if isinstance(t, (Variable, CondVariable)):
        return t.arg
    if not isinstance(t, Compound):
        return t
    if any(issubclass(t.op, cls) for cls in eval_false_legal):
        return t.op(*map(construct, t.args), evaluate=False)
    elif any(issubclass(t.op, cls) for cls in basic_new_legal):
        return Basic.__new__(t.op, *map(construct, t.args))
    else:
        return t.op(*map(construct, t.args))

# 重建 SymPy 表达式
def rebuild(s):
    """ Rebuild a SymPy expression.

    This removes harm caused by Expr-Rules interactions.
    """
    return construct(deconstruct(s))

# 结构化统一算法，用于两个表达式/模式的统一
def unify(x, y, s=None, variables=(), **kwargs):
    """ Structural unification of two expressions/patterns.

    Examples
    ========

    >>> from sympy.unify.usympy import unify
    >>> from sympy import Basic, S
    >>> from sympy.abc import x, y, z, p, q

    >>> next(unify(Basic(S(1), S(2)), Basic(S(1), x), variables=[x]))
    {x: 2}

    >>> expr = 2*x + y + z
    >>> pattern = 2*p + q
    >>> next(unify(expr, pattern, {}, variables=(p, q)))
    {p: x, q: y + z}

    Unification supports commutative and associative matching

    >>> expr = x + y + z
    >>> pattern = p + q
    >>> len(list(unify(expr, pattern, {}, variables=(p, q))))
    """
    """
    decons = lambda x: deconstruct(x, variables)
    将 lambda 函数 decons 定义为一个匿名函数，用于将输入的表达式 x 解构成其组成部分，使用给定的变量集合 variables。
    
    s = s or {}
    如果 s 为假值（空字典），则将其赋值为空字典，否则保持不变。
    
    s = {decons(k): decons(v) for k, v in s.items()}
    使用 decons 函数对 s 字典的每个键值对进行解构，并重新构造一个新的字典 s，其中键和值都被解构后的结果替代。
    
    ds = core.unify(decons(x), decons(y), s,
                                         is_associative=is_associative,
                                         is_commutative=is_commutative,
                                         **kwargs)
    调用 core 模块中的 unify 函数，传入解构后的 x 和 y 表达式，解构后的 s 字典作为初始替换，以及其他命名参数（is_associative, is_commutative, **kwargs）。
    
    for d in ds:
        遍历 unify 函数返回的结果集合 ds。
        yield {construct(k): construct(v) for k, v in d.items()}
        对每个结果字典 d 中的键值对应用 construct 函数，重新构造成新的字典，并通过 yield 返回该字典。
    """
```