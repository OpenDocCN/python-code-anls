# `D:\src\scipysrc\sympy\sympy\unify\core.py`

```
""" Generic Unification algorithm for expression trees with lists of children

This implementation is a direct translation of

Artificial Intelligence: A Modern Approach by Stuart Russel and Peter Norvig
Second edition, section 9.2, page 276

It is modified in the following ways:

1.  We allow associative and commutative Compound expressions. This results in
    combinatorial blowup.
2.  We explore the tree lazily.
3.  We provide generic interfaces to symbolic algebra libraries in Python.

A more traditional version can be found here
http://aima.cs.berkeley.edu/python/logic.html
"""

from sympy.utilities.iterables import kbins

class Compound:
    """ A little class to represent an interior node in the tree

    This is analogous to SymPy.Basic for non-Atoms
    """
    def __init__(self, op, args):
        self.op = op  # Operator symbol or function name
        self.args = args  # List of arguments (subtrees) for this Compound node

    def __eq__(self, other):
        return (type(self) is type(other) and self.op == other.op and
                self.args == other.args)

    def __hash__(self):
        return hash((type(self), self.op, self.args))

    def __str__(self):
        return "%s[%s]" % (str(self.op), ', '.join(map(str, self.args)))

class Variable:
    """ A Wild token """
    def __init__(self, arg):
        self.arg = arg  # Variable name or identifier

    def __eq__(self, other):
        return type(self) is type(other) and self.arg == other.arg

    def __hash__(self):
        return hash((type(self), self.arg))

    def __str__(self):
        return "Variable(%s)" % str(self.arg)

class CondVariable:
    """ A wild token that matches conditionally.

    arg   - a wild token.
    valid - an additional constraining function on a match.
    """
    def __init__(self, arg, valid):
        self.arg = arg  # Variable name or identifier
        self.valid = valid  # Additional condition for matching

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.arg == other.arg and
                self.valid == other.valid)

    def __hash__(self):
        return hash((type(self), self.arg, self.valid))

    def __str__(self):
        return "CondVariable(%s)" % str(self.arg)

def unify(x, y, s=None, **fns):
    """ Unify two expressions.

    Parameters
    ==========

        x, y - expression trees containing leaves, Compounds and Variables.
        s    - a mapping of variables to subtrees.

    Returns
    =======

        lazy sequence of mappings {Variable: subtree}

    Examples
    ========

    >>> from sympy.unify.core import unify, Compound, Variable
    >>> expr    = Compound("Add", ("x", "y"))  # Create a Compound node representing an addition
    >>> pattern = Compound("Add", ("x", Variable("a")))  # Create a pattern with a Variable node
    >>> next(unify(expr, pattern, {}))  # Attempt to unify the expression and the pattern
    {Variable(a): 'y'}  # Expected result: Variable 'a' maps to subtree 'y'
    """
    s = s or {}  # Initialize the substitution mapping if not provided

    if x == y:
        yield s  # If x and y are identical, yield the current substitution mapping
    elif isinstance(x, (Variable, CondVariable)):
        yield from unify_var(x, y, s, **fns)  # Attempt unification with a Variable or CondVariable in x
    elif isinstance(y, (Variable, CondVariable)):
        yield from unify_var(y, x, s, **fns)  # Attempt unification with a Variable or CondVariable in y
    # 如果 x 和 y 都是 Compound 类型的对象
    elif isinstance(x, Compound) and isinstance(y, Compound):
        # 获取 is_commutative 函数，如果不存在则使用 lambda 返回 False
        is_commutative = fns.get('is_commutative', lambda x: False)
        # 获取 is_associative 函数，如果不存在则使用 lambda 返回 False
        is_associative = fns.get('is_associative', lambda x: False)
        # 对 unify(x.op, y.op, s, **fns) 生成的每个解 s 中的每个子句 sop 进行迭代
        for sop in unify(x.op, y.op, s, **fns):
            # 如果 x 和 y 都是可交换的，并且是可结合的
            if is_associative(x) and is_associative(y):
                # 将 x 和 y 中参数较少的对象赋值给 a 和 b
                a, b = (x, y) if len(x.args) < len(y.args) else (y, x)
                # 如果 x 和 y 都是可交换的
                if is_commutative(x) and is_commutative(y):
                    # 生成所有可能的参数组合，用于交换操作
                    combs = allcombinations(a.args, b.args, 'commutative')
                else:
                    # 生成所有可能的参数组合，用于结合操作
                    combs = allcombinations(a.args, b.args, 'associative')
                # 对每个参数组合进行迭代
                for aaargs, bbargs in combs:
                    # 解包参数，并对每个参数进行解包操作，生成解的迭代器
                    aa = [unpack(Compound(a.op, arg)) for arg in aaargs]
                    bb = [unpack(Compound(b.op, arg)) for arg in bbargs]
                    yield from unify(aa, bb, sop, **fns)
            # 如果 x 和 y 的参数个数相同
            elif len(x.args) == len(y.args):
                # 对 x 和 y 的每个参数进行递归解统一，生成解的迭代器
                yield from unify(x.args, y.args, sop, **fns)

    # 如果 x 和 y 都是可迭代对象，并且长度相同
    elif is_args(x) and is_args(y) and len(x) == len(y):
        # 如果 x 和 y 都是空的，直接生成解 s 的迭代器
        if len(x) == 0:
            yield s
        else:
            # 对 x 和 y 的每个头部元素进行递归解统一，生成解 s 的迭代器
            for shead in unify(x[0], y[0], s, **fns):
                yield from unify(x[1:], y[1:], shead, **fns)


这段代码主要用于实现逻辑推理中的统一操作（unification），通过递归和生成器（yield from）实现对不同类型结构（Compound 和可迭代对象）的统一处理，并根据给定的函数（fns）判断是否可交换、可结合等特性来选择合适的操作和参数组合。
# 定义 unify_var 函数，用于变量统一化处理，支持变量约束和条件判断
def unify_var(var, x, s, **fns):
    # 如果变量 var 在替换字典 s 中已存在，递归处理其对应值
    if var in s:
        yield from unify(s[var], x, s, **fns)
    # 如果变量 var 和 x 形成循环引用，进行出现检查
    elif occur_check(var, x):
        pass
    # 如果 var 是条件变量且满足有效性条件，则将其与 x 关联并生成新的替换字典
    elif isinstance(var, CondVariable) and var.valid(x):
        yield assoc(s, var, x)
    # 如果 var 是普通变量，则将其与 x 关联并生成新的替换字典
    elif isinstance(var, Variable):
        yield assoc(s, var, x)

# 定义 occur_check 函数，用于检查变量 var 是否在 x 的子树中出现
def occur_check(var, x):
    """ var occurs in subtree owned by x? """
    # 如果 var 和 x 相等，则表示 var 在 x 的子树中出现
    if var == x:
        return True
    # 如果 x 是复合结构，则递归检查其各个子节点
    elif isinstance(x, Compound):
        return occur_check(var, x.args)
    # 如果 x 是可迭代对象，则递归检查其每个元素
    elif is_args(x):
        if any(occur_check(var, xi) for xi in x): return True
    return False

# 定义 assoc 函数，返回一个带有新关联的键值对的字典副本
def assoc(d, key, val):
    """ Return copy of d with key associated to val """
    d = d.copy()
    d[key] = val
    return d

# 定义 is_args 函数，用于判断对象 x 是否是传统的可迭代对象
def is_args(x):
    """ Is x a traditional iterable? """
    return type(x) in (tuple, list, set)

# 定义 unpack 函数，用于提取复合结构对象 x 的单个成员（如果可能）
def unpack(x):
    if isinstance(x, Compound) and len(x.args) == 1:
        return x.args[0]
    else:
        return x

# 定义 allcombinations 函数，重组 A 和 B 使其具有相同数量的元素
def allcombinations(A, B, ordered):
    """
    Restructure A and B to have the same number of elements.

    Parameters
    ==========

    ordered must be either 'commutative' or 'associative'.

    A and B can be rearranged so that the larger of the two lists is
    reorganized into smaller sublists.

    Examples
    ========

    >>> from sympy.unify.core import allcombinations
    >>> for x in allcombinations((1, 2, 3), (5, 6), 'associative'): print(x)
    (((1,), (2, 3)), ((5,), (6,)))
    (((1, 2), (3,)), ((5,), (6,)))

    >>> for x in allcombinations((1, 2, 3), (5, 6), 'commutative'): print(x)
        (((1,), (2, 3)), ((5,), (6,)))
        (((1, 2), (3,)), ((5,), (6,)))
        (((1,), (3, 2)), ((5,), (6,)))
        (((1, 3), (2,)), ((5,), (6,)))
        (((2,), (1, 3)), ((5,), (6,)))
        (((2, 1), (3,)), ((5,), (6,)))
        (((2,), (3, 1)), ((5,), (6,)))
        (((2, 3), (1,)), ((5,), (6,)))
        (((3,), (1, 2)), ((5,), (6,)))
        (((3, 1), (2,)), ((5,), (6,)))
        (((3,), (2, 1)), ((5,), (6,)))
        (((3, 2), (1,)), ((5,), (6,)))
    """

    if ordered == "commutative":
        ordered = 11
    if ordered == "associative":
        ordered = None
    # 将 A 和 B 中较大的列表重新组织为较小的子列表，并生成所有可能的组合
    sm, bg = (A, B) if len(A) < len(B) else (B, A)
    for part in kbins(list(range(len(bg))), len(sm), ordered=ordered):
        if bg == B:
            yield tuple((a,) for a in A), partition(B, part)
        else:
            yield partition(A, part), tuple((b,) for b in B)

# 定义 partition 函数，将元组或列表 it 按照给定的索引列表 part 进行分割
def partition(it, part):
    """ Partition a tuple/list into pieces defined by indices.

    Examples
    ========

    >>> from sympy.unify.core import partition
    >>> partition((10, 20, 30, 40), [[0, 1, 2], [3]])
    ((10, 20, 30), (40,))
    """
    return type(it)([index(it, ind) for ind in part])

# 定义 index 函数，对可索引的可迭代对象 it 进行高级索引操作
def index(it, ind):
    """ Fancy indexing into an indexable iterable (tuple, list).

    Examples
    ========

    >>> from sympy.unify.core import index
    >>> index([10, 20, 30], (1, 2, 0))
    [20, 30, 10]
    """
    return type(it)([it[i] for i in ind])
```