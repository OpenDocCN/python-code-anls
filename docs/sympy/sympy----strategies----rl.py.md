# `D:\src\scipysrc\sympy\sympy\strategies\rl.py`

```
# 导入从sympy.utilities.iterables中的sift函数
# 从当前目录的util模块导入new函数
from sympy.utilities.iterables import sift
from .util import new

# 创建一个移除身份元素的规则函数
def rm_id(isid, new=new):
    """ Create a rule to remove identities.

    isid - fn :: x -> Bool  --- whether or not this element is an identity.

    Examples
    ========

    >>> from sympy.strategies import rm_id
    >>> from sympy import Basic, S
    >>> remove_zeros = rm_id(lambda x: x==0)
    >>> remove_zeros(Basic(S(1), S(0), S(2)))
    Basic(1, 2)
    >>> remove_zeros(Basic(S(0), S(0))) # If only identites then we keep one
    Basic(0)

    See Also:
        unpack
    """
    # 内部函数，用于移除身份元素
    def ident_remove(expr):
        """ Remove identities """
        # 检查表达式中每个参数是否是身份元素
        ids = list(map(isid, expr.args))
        if sum(ids) == 0:           # 如果没有身份元素，通常情况
            return expr
        elif sum(ids) != len(ids):  # 如果至少有一个非身份元素
            # 创建一个新的表达式，只包含非身份元素
            return new(expr.__class__,
                       *[arg for arg, x in zip(expr.args, ids) if not x])
        else:
            # 如果所有参数都是身份元素，保留第一个参数
            return new(expr.__class__, expr.args[0])

    return ident_remove


# 创建一个合并相同参数的规则函数
def glom(key, count, combine):
    """ Create a rule to conglomerate identical args.

    Examples
    ========

    >>> from sympy.strategies import glom
    >>> from sympy import Add
    >>> from sympy.abc import x

    >>> key     = lambda x: x.as_coeff_Mul()[1]
    >>> count   = lambda x: x.as_coeff_Mul()[0]
    >>> combine = lambda cnt, arg: cnt * arg
    >>> rl = glom(key, count, combine)

    >>> rl(Add(x, -x, 3*x, 2, 3, evaluate=False))
    3*x + 5

    Wait, how are key, count and combine supposed to work?

    >>> key(2*x)
    x
    >>> count(2*x)
    2
    >>> combine(2, x)
    2*x
    """
    # 内部函数，用于合并相同参数
    def conglomerate(expr):
        """ Conglomerate together identical args x + x -> 2x """
        # 使用关键函数将参数分组
        groups = sift(expr.args, key)
        # 计算每个组中参数的数量
        counts = {k: sum(map(count, args)) for k, args in groups.items()}
        # 根据combine函数创建新的参数列表
        newargs = [combine(cnt, mat) for mat, cnt in counts.items()]
        # 如果新参数列表与原参数列表不同，则返回新表达式
        if set(newargs) != set(expr.args):
            return new(type(expr), *newargs)
        else:
            return expr

    return conglomerate


# 创建一个按照关键函数排序的规则函数
def sort(key, new=new):
    """ Create a rule to sort by a key function.

    Examples
    ========

    >>> from sympy.strategies import sort
    >>> from sympy import Basic, S
    >>> sort_rl = sort(str)
    >>> sort_rl(Basic(S(3), S(1), S(2)))
    Basic(1, 2, 3)
    """

    # 内部函数，用于根据关键函数对参数进行排序
    def sort_rl(expr):
        return new(expr.__class__, *sorted(expr.args, key=key))
    return sort_rl


# 创建一个将包含B的A转换为包含A的B的规则函数
def distribute(A, B):
    """ Turns an A containing Bs into a B of As

    where A, B are container types

    >>> from sympy.strategies import distribute
    >>> from sympy import Add, Mul, symbols
    >>> x, y = symbols('x,y')
    >>> dist = distribute(Mul, Add)
    >>> expr = Mul(2, x+y, evaluate=False)
    >>> expr
    2*(x + y)
    >>> dist(expr)
    2*x + 2*y
    """
    # 定义一个函数 distribute_rl，用于分发关系表达式中的 B 类型参数
    def distribute_rl(expr):
        # 遍历表达式的每个参数及其索引
        for i, arg in enumerate(expr.args):
            # 如果当前参数是 B 类型的实例
            if isinstance(arg, B):
                # 将表达式参数分成三部分：第一部分、当前 B 类型参数、剩余部分
                first, b, tail = expr.args[:i], expr.args[i], expr.args[i + 1:]
                # 返回一个新的 B 类型参数，应用于所有可能的 A 类型参数的组合
                return B(*[A(*(first + (arg,) + tail)) for arg in b.args])
        # 如果没有找到 B 类型参数，直接返回原始的表达式
        return expr
    # 返回函数 distribute_rl 本身
    return distribute_rl
# 定义函数 subs，返回一个函数 subs_rl，用于替换表达式
def subs(a, b):
    """ Replace expressions exactly """
    # 定义内部函数 subs_rl，用于替换表达式
    def subs_rl(expr):
        # 如果表达式 expr 等于 a，则返回 b
        if expr == a:
            return b
        else:
            return expr
    # 返回内部函数 subs_rl
    return subs_rl


# 定义函数 unpack，用于解包单一参数的规则
def unpack(expr):
    """ Rule to unpack singleton args

    >>> from sympy.strategies import unpack
    >>> from sympy import Basic, S
    >>> unpack(Basic(S(2)))
    2
    """
    # 如果表达式的参数长度为 1，则返回第一个参数
    if len(expr.args) == 1:
        return expr.args[0]
    else:
        return expr


# 定义函数 flatten，用于将表达式扁平化
def flatten(expr, new=new):
    """ Flatten T(a, b, T(c, d), T2(e)) to T(a, b, c, d, T2(e)) """
    # 获取表达式的类别
    cls = expr.__class__
    args = []
    # 遍历表达式的参数
    for arg in expr.args:
        # 如果参数的类别与表达式的类别相同，将其参数扩展到 args 列表中
        if arg.__class__ == cls:
            args.extend(arg.args)
        else:
            args.append(arg)
    # 使用 new 函数重新构建表达式的类别和 args 中的参数
    return new(expr.__class__, *args)


# 定义函数 rebuild，用于重建 SymPy 树结构
def rebuild(expr):
    """ Rebuild a SymPy tree.

    Explanation
    ===========

    This function recursively calls constructors in the expression tree.
    This forces canonicalization and removes ugliness introduced by the use of
    Basic.__new__
    """
    # 如果表达式是原子，则直接返回
    if expr.is_Atom:
        return expr
    else:
        # 否则，递归地调用表达式的构造函数，重新构建整个表达式树
        return expr.func(*list(map(rebuild, expr.args)))
```