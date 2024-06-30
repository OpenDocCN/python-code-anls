# `D:\src\scipysrc\sympy\sympy\printing\tree.py`

```
# 打印系统节点的漂亮格式化输出函数
def pprint_nodes(subtrees):
    """
    Prettyprints systems of nodes.

    Examples
    ========

    >>> from sympy.printing.tree import pprint_nodes
    >>> print(pprint_nodes(["a", "b1\\nb2", "c"]))
    +-a
    +-b1
    | b2
    +-c

    """
    # 定义缩进函数，根据 type 参数决定不同的缩进方式
    def indent(s, type=1):
        x = s.split("\n")
        r = "+-%s\n" % x[0]
        for a in x[1:]:
            if a == "":
                continue
            if type == 1:
                r += "| %s\n" % a
            else:
                r += "  %s\n" % a
        return r
    
    # 如果没有子树，则返回空字符串
    if not subtrees:
        return ""
    
    f = ""
    # 对于除了最后一个子树以外的每个子树，使用 type=1 的缩进方式
    for a in subtrees[:-1]:
        f += indent(a)
    # 对于最后一个子树，使用 type=2 的缩进方式
    f += indent(subtrees[-1], 2)
    return f


# 返回关于节点 "node" 的信息的函数
def print_node(node, assumptions=True):
    """
    Returns information about the "node".

    This includes class name, string representation and assumptions.

    Parameters
    ==========

    assumptions : bool, optional
        See the ``assumptions`` keyword in ``tree``
    """
    s = "%s: %s\n" % (node.__class__.__name__, str(node))

    # 如果 assumptions 参数为 True，则获取节点的假设信息
    if assumptions:
        d = node._assumptions
    else:
        d = None

    # 如果存在假设信息，则将其逐个加入到输出字符串中
    if d:
        for a in sorted(d):
            v = d[a]
            if v is None:
                continue
            s += "%s: %s\n" % (a, v)

    return s


# 返回节点 "node" 的树形表示字符串的函数
def tree(node, assumptions=True):
    """
    Returns a tree representation of "node" as a string.

    It uses print_node() together with pprint_nodes() on node.args recursively.

    Parameters
    ==========

    asssumptions : bool, optional
        The flag to decide whether to print out all the assumption data
        (such as ``is_integer``, ``is_real``) associated with the
        expression or not.

        Enabling the flag makes the result verbose, and the printed
        result may not be deterministic because of the randomness used
        in backtracing the assumptions.

    See Also
    ========

    print_tree

    """
    subtrees = []
    # 对于节点的每个参数，递归调用 tree() 函数获取其树形表示
    for arg in node.args:
        subtrees.append(tree(arg, assumptions=assumptions))
    # 构建包含节点信息和子树格式化输出的字符串
    s = print_node(node, assumptions=assumptions) + pprint_nodes(subtrees)
    return s


# 打印节点 "node" 的树形表示
def print_tree(node, assumptions=True):
    """
    Prints a tree representation of "node".

    Parameters
    ==========

    asssumptions : bool, optional
        The flag to decide whether to print out all the assumption data
        (such as ``is_integer``, ``is_real``) associated with the
        expression or not.

        Enabling the flag makes the result verbose, and the printed
        result may not be deterministic because of the randomness used
        in backtracing the assumptions.

    Examples
    ========

    >>> from sympy.printing import print_tree
    >>> from sympy import Symbol
    >>> x = Symbol('x', odd=True)
    >>> y = Symbol('y', even=True)

    Printing with full assumptions information:

    >>> print_tree(y**x)
    Pow: y**x
    +-Symbol: y
    | algebraic: True
    | commutative: True
    | complex: True
    | even: True
    | extended_real: True

    """
    # 调用 tree() 函数获取节点 "node" 的树形表示字符串，并打印出来
    subtrees = []
    for arg in node.args:
        subtrees.append(tree(arg, assumptions=assumptions))
    s = print_node(node, assumptions=assumptions) + pprint_nodes(subtrees)
    return s
    """
    打印给定节点的树形结构表示

    参数：
    - node：要打印树的根节点
    - assumptions：布尔值，控制是否打印节点的假设信息，默认为 True

    返回值：
    无，直接将树形结构打印到标准输出

    示例用法：
    >>> print_tree(y**x, assumptions=False)

    相关函数：
    - tree

    """
    print(tree(node, assumptions=assumptions))
```