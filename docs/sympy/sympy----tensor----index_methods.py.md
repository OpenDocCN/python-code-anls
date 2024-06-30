# `D:\src\scipysrc\sympy\sympy\tensor\index_methods.py`

```
"""
Module with functions operating on IndexedBase, Indexed and Idx objects

- Check shape conformance
- Determine indices in resulting expression

etc.

Methods in this module could be implemented by calling methods on Expr
objects instead.  When things stabilize this could be a useful
refactoring.
"""

from functools import reduce  # 导入 functools 库中的 reduce 函数

from sympy.core.function import Function  # 从 sympy.core.function 模块导入 Function 类
from sympy.functions import exp, Piecewise  # 导入 exp 和 Piecewise 函数
from sympy.tensor.indexed import Idx, Indexed  # 从 sympy.tensor.indexed 模块导入 Idx 和 Indexed 类
from sympy.utilities import sift  # 导入 sift 函数

from collections import OrderedDict  # 导入 OrderedDict 类


class IndexConformanceException(Exception):
    pass  # 定义 IndexConformanceException 异常类


def _unique_and_repeated(inds):
    """
    Returns the unique and repeated indices. Also note, from the examples given below
    that the order of indices is maintained as given in the input.

    Examples
    ========

    >>> from sympy.tensor.index_methods import _unique_and_repeated
    >>> _unique_and_repeated([2, 3, 1, 3, 0, 4, 0])
    ([2, 1, 4], [3, 0])
    """
    uniq = OrderedDict()
    for i in inds:
        if i in uniq:
            uniq[i] = 0
        else:
            uniq[i] = 1
    return sift(uniq, lambda x: uniq[x], binary=True)  # 使用 sift 函数分离唯一和重复的索引


def _remove_repeated(inds):
    """
    Removes repeated objects from sequences

    Returns a set of the unique objects and a tuple of all that have been
    removed.

    Examples
    ========

    >>> from sympy.tensor.index_methods import _remove_repeated
    >>> l1 = [1, 2, 3, 2]
    >>> _remove_repeated(l1)
    ({1, 3}, (2,))

    """
    u, r = _unique_and_repeated(inds)  # 调用 _unique_and_repeated 函数获取唯一和重复的索引
    return set(u), tuple(r)  # 返回唯一索引的集合和重复索引的元组


def _get_indices_Mul(expr, return_dummies=False):
    """Determine the outer indices of a Mul object.

    Examples
    ========

    >>> from sympy.tensor.index_methods import _get_indices_Mul
    >>> from sympy.tensor.indexed import IndexedBase, Idx
    >>> i, j, k = map(Idx, ['i', 'j', 'k'])
    >>> x = IndexedBase('x')
    >>> y = IndexedBase('y')
    >>> _get_indices_Mul(x[i, k]*y[j, k])
    ({i, j}, {})
    >>> _get_indices_Mul(x[i, k]*y[j, k], return_dummies=True)
    ({i, j}, {}, (k,))

    """

    inds = list(map(get_indices, expr.args))  # 获取表达式 expr 中每个参数的索引
    inds, syms = list(zip(*inds))  # 分离索引和对称性信息

    inds = list(map(list, inds))  # 将索引转换为列表形式
    inds = list(reduce(lambda x, y: x + y, inds))  # 使用 reduce 连接所有索引列表
    inds, dummies = _remove_repeated(inds)  # 移除重复的索引

    symmetry = {}
    for s in syms:
        for pair in s:
            if pair in symmetry:
                symmetry[pair] *= s[pair]
            else:
                symmetry[pair] = s[pair]

    if return_dummies:
        return inds, symmetry, dummies  # 返回索引集合、对称性信息和虚拟索引元组
    else:
        return inds, symmetry  # 只返回索引集合和对称性信息


def _get_indices_Pow(expr):
    """Determine outer indices of a power or an exponential.

    A power is considered a universal function, so that the indices of a Pow is
    just the collection of indices present in the expression.  This may be
    viewed as a bit inconsistent in the special case:

        x[i]**2 = x[i]*x[i]                                                      (1)

    """
    """
    The above expression could have been interpreted as the contraction of x[i]
    with itself, but we choose instead to interpret it as a function

        lambda y: y**2

    applied to each element of x (a universal function in numpy terms).  In
    order to allow an interpretation of (1) as a contraction, we need
    contravariant and covariant Idx subclasses.  (FIXME: this is not yet
    implemented)

    Expressions in the base or exponent are subject to contraction as usual,
    but an index that is present in the exponent, will not be considered
    contractable with its own base.  Note however, that indices in the same
    exponent can be contracted with each other.

    Examples
    ========

    >>> from sympy.tensor.index_methods import _get_indices_Pow
    >>> from sympy import Pow, exp, IndexedBase, Idx
    >>> A = IndexedBase('A')
    >>> x = IndexedBase('x')
    >>> i, j, k = map(Idx, ['i', 'j', 'k'])
    >>> _get_indices_Pow(exp(A[i, j]*x[j]))
    ({i}, {})
    >>> _get_indices_Pow(Pow(x[i], x[i]))
    ({i}, {})
    >>> _get_indices_Pow(Pow(A[i, j]*x[j], x[i]))
    ({i}, {})

    """
    # 将表达式分解为底数和指数
    base, exp = expr.as_base_exp()
    # 获取底数的所有指标
    binds, bsyms = get_indices(base)
    # 获取指数的所有指标
    einds, esyms = get_indices(exp)

    # 合并底数和指数的所有指标，去除重复的指标
    inds = binds | einds

    # FIXME: 根据幂运算的对称性进行特殊情况的检查，否则返回空字典
    symmetries = {}

    # 返回所有指标及其对称性信息
    return inds, symmetries
# 确定 Add 对象的外部索引。

# 在一个求和表达式中，每个项必须具有相同的外部索引集合。一个有效的表达式可以是
# x(i)*y(j) - x(j)*y(i)
# 但我们不允许像下面这样的表达式：
# x(i)*y(j) - z(j)*z(j)
# FIXME: 添加对 NumPy 广播的支持

# 导入必要的库和模块
def _get_indices_Add(expr):
    """Determine outer indices of an Add object.

    In a sum, each term must have the same set of outer indices.  A valid
    expression could be

        x(i)*y(j) - x(j)*y(i)

    But we do not allow expressions like:

        x(i)*y(j) - z(j)*z(j)

    FIXME: Add support for Numpy broadcasting

    Examples
    ========

    >>> from sympy.tensor.index_methods import _get_indices_Add
    >>> from sympy.tensor.indexed import IndexedBase, Idx
    >>> i, j, k = map(Idx, ['i', 'j', 'k'])
    >>> x = IndexedBase('x')
    >>> y = IndexedBase('y')
    >>> _get_indices_Add(x[i] + x[k]*y[i, k])
    ({i}, {})

    """

    # 对表达式的每个参数应用 get_indices 函数并收集结果
    inds = list(map(get_indices, expr.args))
    # 将结果拆分为索引集合和符号信息
    inds, syms = list(zip(*inds))

    # 允许标量的广播
    non_scalars = [x for x in inds if x != set()]
    # 如果所有参数都是标量，则返回空集合和空字典
    if not non_scalars:
        return set(), {}

    # 检查所有非标量参数的索引集合是否一致
    if not all(x == non_scalars[0] for x in non_scalars[1:]):
        raise IndexConformanceException("Indices are not consistent: %s" % expr)
    
    # 检查符号信息是否一致
    if not reduce(lambda x, y: x != y or y, syms):
        symmetries = syms[0]
    else:
        # FIXME: 搜索对称性
        symmetries = {}

    return non_scalars[0], symmetries


# 确定表达式 expr 的外部索引。

# “外部”指的是不是求和索引的索引。返回一个集合和一个字典。集合包含外部索引，
# 字典包含索引对称性信息。

# 导入必要的库和模块
def get_indices(expr):
    """Determine the outer indices of expression ``expr``

    By *outer* we mean indices that are not summation indices.  Returns a set
    and a dict.  The set contains outer indices and the dict contains
    information about index symmetries.

    Examples
    ========

    >>> from sympy.tensor.index_methods import get_indices
    >>> from sympy import symbols
    >>> from sympy.tensor import IndexedBase
    >>> x, y, A = map(IndexedBase, ['x', 'y', 'A'])
    >>> i, j, a, z = symbols('i j a z', integer=True)

    The indices of the total expression is determined, Repeated indices imply a
    summation, for instance the trace of a matrix A:

    >>> get_indices(A[i, i])
    (set(), {})

    In the case of many terms, the terms are required to have identical
    outer indices.  Else an IndexConformanceException is raised.

    >>> get_indices(x[i] + A[i, j]*y[j])
    ({i}, {})

    :Exceptions:

    An IndexConformanceException means that the terms ar not compatible, e.g.

    >>> get_indices(x[i] + y[j])                #doctest: +SKIP
            (...)
    IndexConformanceException: Indices are not consistent: x(i) + y(j)


    """

    # 对表达式的每个参数应用 get_indices 函数并收集结果
    inds = list(map(get_indices, expr.args))
    # 将结果拆分为索引集合和符号信息
    inds, syms = list(zip(*inds))
    
    # 允许标量的广播
    non_scalars = [x for x in inds if x != set()]
    # 如果所有参数都是标量，则返回空集合和空字典
    if not non_scalars:
        return set(), {}

    # 检查所有非标量参数的索引集合是否一致
    if not all(x == non_scalars[0] for x in non_scalars[1:]):
        raise IndexConformanceException("Indices are not consistent: %s" % expr)
    
    # 检查符号信息是否一致
    if not reduce(lambda x, y: x != y or y, syms):
        symmetries = syms[0]
    else:
        # FIXME: 搜索对称性
        symmetries = {}

    return non_scalars[0], symmetries
    """
    # 警告：外部索引的概念递归地应用于最深层级。这意味着括号内的虚拟变量被假定首先求和，
    # 因此以下表达式可以被优雅地处理：
    #
    # >>> get_indices((x[i] + A[i, j]*y[j])*x[j])
    # ({i, j}, {})
    #
    # 这是正确的，并且看起来很方便，但是你需要小心，因为如果请求，SymPy会乐意展开乘积。
    # 结果的表达式将混合外部的 `j` 和括号内的虚拟变量，这使得它成为一个不同的表达式。
    # 为了保险起见，最好通过为所有应分开处理的收缩使用唯一的索引来避免这种歧义。

    # 递归调用以确定子表达式的索引。

    # 如果表达式是 Indexed 类型
    if isinstance(expr, Indexed):
        # 获取索引并去除重复的虚拟变量
        c = expr.indices
        inds, dummies = _remove_repeated(c)
        return inds, {}

    # 如果表达式是 None 类型
    elif expr is None:
        return set(), {}

    # 如果表达式是 Idx 类型
    elif isinstance(expr, Idx):
        return {expr}, {}

    # 如果表达式是原子类型（Atom）
    elif expr.is_Atom:
        return set(), {}

    # 通过专门的函数递归处理
    else:
        # 如果表达式是乘法类型
        if expr.is_Mul:
            return _get_indices_Mul(expr)

        # 如果表达式是加法类型
        elif expr.is_Add:
            return _get_indices_Add(expr)

        # 如果表达式是幂或者是 exp 类型
        elif expr.is_Pow or isinstance(expr, exp):
            return _get_indices_Pow(expr)

        # 如果表达式是 Piecewise 类型
        elif isinstance(expr, Piecewise):
            # FIXME: 目前不支持 Piecewise
            return set(), {}

        # 如果表达式是 Function 类型
        elif isinstance(expr, Function):
            # 通过返回参数的索引来支持类似 ufunc 的行为
            # 函数不会解释跨参数的重复索引作为求和
            ind0 = set()
            for arg in expr.args:
                ind, sym = get_indices(arg)
                ind0 |= ind
            return ind0, sym

        # 如果表达式不包含 Indexed
        elif not expr.has(Indexed):
            return set(), {}

        # 如果以上条件均不满足，抛出未实现的错误
        raise NotImplementedError(
            "FIXME: No specialized handling of type %s" % type(expr))
    """
# 定义函数 get_contraction_structure，用于确定表达式 expr 的虚拟指标并描述其结构
def get_contraction_structure(expr):
    """Determine dummy indices of ``expr`` and describe its structure

    By *dummy* we mean indices that are summation indices.

    The structure of the expression is determined and described as follows:

    1) A conforming summation of Indexed objects is described with a dict where
       the keys are summation indices and the corresponding values are sets
       containing all terms for which the summation applies.  All Add objects
       in the SymPy expression tree are described like this.

    2) For all nodes in the SymPy expression tree that are *not* of type Add, the
       following applies:

       If a node discovers contractions in one of its arguments, the node
       itself will be stored as a key in the dict.  For that key, the
       corresponding value is a list of dicts, each of which is the result of a
       recursive call to get_contraction_structure().  The list contains only
       dicts for the non-trivial deeper contractions, omitting dicts with None
       as the one and only key.

    .. Note:: The presence of expressions among the dictionary keys indicates
       multiple levels of index contractions.  A nested dict displays nested
       contractions and may itself contain dicts from a deeper level.  In
       practical calculations the summation in the deepest nested level must be
       calculated first so that the outer expression can access the resulting
       indexed object.

    Examples
    ========

    >>> from sympy.tensor.index_methods import get_contraction_structure
    >>> from sympy import default_sort_key
    >>> from sympy.tensor import IndexedBase, Idx
    >>> x, y, A = map(IndexedBase, ['x', 'y', 'A'])
    >>> i, j, k, l = map(Idx, ['i', 'j', 'k', 'l'])
    >>> get_contraction_structure(x[i]*y[i] + A[j, j])
    {(i,): {x[i]*y[i]}, (j,): {A[j, j]}}
    >>> get_contraction_structure(x[i]*y[j])
    {None: {x[i]*y[j]}}

    A multiplication of contracted factors results in nested dicts representing
    the internal contractions.

    >>> d = get_contraction_structure(x[i, i]*y[j, j])
    >>> sorted(d.keys(), key=default_sort_key)
    [None, x[i, i]*y[j, j]]

    In this case, the product has no contractions:

    >>> d[None]
    {x[i, i]*y[j, j]}

    Factors are contracted "first":

    >>> sorted(d[x[i, i]*y[j, j]], key=default_sort_key)
    [{(i,): {x[i, i]}}, {(j,): {y[j, j]}}]

    A parenthesized Add object is also returned as a nested dictionary.  The
    term containing the parenthesis is a Mul with a contraction among the
    arguments, so it will be found as a key in the result.  It stores the
    dictionary resulting from a recursive call on the Add expression.

    >>> d = get_contraction_structure(x[i]*(y[i] + A[i, j]*x[j]))
    >>> sorted(d.keys(), key=default_sort_key)
    [(A[i, j]*x[j] + y[i])*x[i], (i,)]
    >>> d[(i,)]
    {(A[i, j]*x[j] + y[i])*x[i]}
    >>> d[x[i]*(A[i, j]*x[j] + y[i])]
    [{None: {y[i]}, (j,): {A[i, j]*x[j]}}]
    """

    # 返回值为一个字典，描述了表达式 expr 的结构及其虚拟指标的信息
    return result_dict
    # 本函数用于获取表达式的收缩结构，返回一个字典，其中键表示不同的收缩方式，值为相关的表达式集合。
    
    # 如果表达式是 Indexed 类型，提取其索引并移除重复，返回一个字典，键为索引（如果存在），值为包含表达式的集合。
    if isinstance(expr, Indexed):
        junk, key = _remove_repeated(expr.indices)
        return {key or None: {expr}}
    
    # 如果表达式是原子类型（Atom），返回一个字典，键为 None，值为包含表达式的集合。
    elif expr.is_Atom:
        return {None: {expr}}
    
    # 如果表达式是乘法类型（Mul），获取其索引并返回一个字典，键为索引（如果存在），值为包含表达式的集合。
    elif expr.is_Mul:
        junk, junk, key = _get_indices_Mul(expr, return_dummies=True)
        result = {key or None: {expr}}
        nested = []
        # 对乘法表达式的每个因子进行递归处理
        for fac in expr.args:
            facd = get_contraction_structure(fac)
            if not (None in facd and len(facd) == 1):
                nested.append(facd)
        if nested:
            result[expr] = nested
        return result
    
    # 如果表达式是幂类型（Pow）或指数函数（exp），分别对其基数和指数进行递归处理
    elif expr.is_Pow or isinstance(expr, exp):
        # 获取基数和指数的收缩结构
        b, e = expr.as_base_exp()
        dbase = get_contraction_structure(b)
        dexp = get_contraction_structure(e)
        dicts = []
        for d in dbase, dexp:
            if not (None in d and len(d) == 1):
                dicts.append(d)
        result = {None: {expr}}
        if dicts:
            result[expr] = dicts
        return result
    
    # 如果表达式是加法类型（Add），对其每个项进行递归处理，并按照相同的索引将项收集到一个字典中
    elif expr.is_Add:
        result = {}
        for term in expr.args:
            d = get_contraction_structure(term)
            for key in d:
                if key in result:
                    result[key] |= d[key]
                else:
                    result[key] = d[key]
        return result
    
    # 如果表达式是分段函数（Piecewise），暂时不支持，返回一个字典，键为 None，值为表达式本身。
    elif isinstance(expr, Piecewise):
        # TODO: 目前不支持分段函数的处理
        return {None: expr}
    # 如果表达式是一个函数对象的实例
    elif isinstance(expr, Function):
        # 收集每个参数中的非平凡缩并结构
        # 我们不将在不同参数中的重复索引报告为一种缩并
        deeplist = []
        for arg in expr.args:
            # 获取参数中的缩并结构
            deep = get_contraction_structure(arg)
            # 如果不是单一的 None，并且不全是 None，则将其添加到 deeplist 中
            if not (None in deep and len(deep) == 1):
                deeplist.append(deep)
        # 初始化字典 d，将 None 映射到表达式的集合
        d = {None: {expr}}
        # 如果 deeplist 非空，则将 expr 映射到 deeplist
        if deeplist:
            d[expr] = deeplist
        # 返回构建的字典 d
        return d

    # 这个测试比较耗费资源，所以应该放在最后
    elif not expr.has(Indexed):
        # 如果表达式不包含 Indexed 对象，则返回映射 None 到表达式的集合
        return {None: {expr}}
    # 如果以上条件均不满足，则抛出未实现的错误
    raise NotImplementedError(
        "FIXME: No specialized handling of type %s" % type(expr))
```