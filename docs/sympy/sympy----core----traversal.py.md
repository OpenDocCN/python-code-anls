# `D:\src\scipysrc\sympy\sympy\core\traversal.py`

```
# 导入基础模块 Basic
from .basic import Basic
# 导入排序模块中的 ordered 函数
from .sorting import ordered
# 导入 sympify 函数，用于将输入转换为 SymPy 表达式
from .sympify import sympify
# 导入 iterable 函数，用于检查对象是否可迭代
from sympy.utilities.iterables import iterable


# 定义函数 iterargs，用于广度优先遍历 Basic 对象的参数
def iterargs(expr):
    """Yield the args of a Basic object in a breadth-first traversal.
    Depth-traversal stops if `arg.args` is either empty or is not
    an iterable.

    Examples
    ========

    >>> from sympy import Integral, Function
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> from sympy.core.traversal import iterargs
    >>> list(iterargs(Integral(f(x), (f(x), 1))))
    [Integral(f(x), (f(x), 1)), f(x), (f(x), 1), x, f(x), 1, x]

    See Also
    ========
    iterfreeargs, preorder_traversal
    """
    # 初始化参数列表，将表达式 expr 添加到列表中
    args = [expr]
    # 对于 args 列表中的每个元素 i，进行迭代
    for i in args:
        # 生成器：产生当前参数 i
        yield i
        # 将参数 i 的所有子参数添加到 args 列表末尾
        args.extend(i.args)


# 定义函数 iterfreeargs，用于广度优先遍历 Basic 对象的参数
def iterfreeargs(expr, _first=True):
    """Yield the args of a Basic object in a breadth-first traversal.
    Depth-traversal stops if `arg.args` is either empty or is not
    an iterable. The bound objects of an expression will be returned
    as canonical variables.

    Examples
    ========

    >>> from sympy import Integral, Function
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> from sympy.core.traversal import iterfreeargs
    >>> list(iterfreeargs(Integral(f(x), (f(x), 1))))
    [Integral(f(x), (f(x), 1)), 1]

    See Also
    ========
    iterargs, preorder_traversal
    """
    # 初始化参数列表，将表达式 expr 添加到列表中
    args = [expr]
    # 对于 args 列表中的每个元素 i，进行迭代
    for i in args:
        # 生成器：产生当前参数 i
        yield i
        # 如果 _first 为真且 i 具有 bound_symbols 属性
        if _first and hasattr(i, 'bound_symbols'):
            # 从 i 的 canonical_variables 值中获取变量集合
            void = i.canonical_variables.values()
            # 对 i 的虚拟变量进行迭代
            for i in iterfreeargs(i.as_dummy(), _first=False):
                # 如果 i 中不包含 void 中的任何变量，则产生当前参数 i
                if not i.has(*void):
                    yield i
        # 将参数 i 的所有子参数添加到 args 列表末尾
        args.extend(i.args)


# 定义类 preorder_traversal，执行一棵树的先序遍历
class preorder_traversal:
    """
    Do a pre-order traversal of a tree.

    This iterator recursively yields nodes that it has visited in a pre-order
    fashion. That is, it yields the current node then descends through the
    tree breadth-first to yield all of a node's children's pre-order
    traversal.


    For an expression, the order of the traversal depends on the order of
    .args, which in many cases can be arbitrary.

    Parameters
    ==========
    node : SymPy expression
        The expression to traverse.
    keys : (default None) sort key(s)
        The key(s) used to sort args of Basic objects. When None, args of Basic
        objects are processed in arbitrary order. If key is defined, it will
        be passed along to ordered() as the only key(s) to use to sort the
        arguments; if ``key`` is simply True then the default keys of ordered
        will be used.

    Yields
    ======
    subtree : SymPy expression
        All of the subtrees in the tree.

    Examples
    ========

    >>> from sympy import preorder_traversal, symbols
    >>> x, y, z = symbols('x y z')

    The nodes are returned in the order that they are encountered unless key
    is given; simply passing key=True will guarantee that the traversal is
    unique.
    """
    # 初始化 preorder_traversal 类的实例，执行一棵树的先序遍历
    def __init__(self, node, keys=None):
        self._skip_flag = False
        self._nodes = [node]
        self._keys = keys

    # 迭代器方法：生成器，产生先序遍历的节点
    def __iter__(self):
        while self._nodes:
            node = self._nodes.pop(0)
            yield node
            if not self._skip_flag:
                if isinstance(node, Basic):
                    # 使用 ordered 函数对节点的参数进行排序
                    args = node.args if self._keys is None else ordered(node.args, keys=self._keys)
                    # 将排序后的参数添加到 _nodes 列表头部
                    self._nodes[:0] = args

    # 设置跳过标志的方法
    def skip(self):
        self._skip_flag = True
    >>> list(preorder_traversal((x + y)*z, keys=None)) # doctest: +SKIP
    [z*(x + y), z, x + y, y, x]
    >>> list(preorder_traversal((x + y)*z, keys=True))
    [z*(x + y), z, x + y, x, y]

    """
    # 定义一个遍历器类，用于前序遍历给定节点，并可选地按照指定键排序其子节点
    def __init__(self, node, keys=None):
        # 初始化跳过标志为假
        self._skip_flag = False
        # 调用内部方法进行前序遍历，生成遍历器对象
        self._pt = self._preorder_traversal(node, keys)

    # 内部方法：执行前序遍历的生成器函数
    def _preorder_traversal(self, node, keys):
        # 生成当前节点
        yield node
        # 如果需要跳过当前节点的子树
        if self._skip_flag:
            self._skip_flag = False
            return
        # 如果节点是基本类型对象
        if isinstance(node, Basic):
            # 如果不需要按键排序并且节点有'_argset'属性
            if not keys and hasattr(node, '_argset'):
                # 使用'_argset'保持参数作为集合，如果不关心顺序，以防止不必要的排序
                args = node._argset
            else:
                # 否则按默认顺序获取参数
                args = node.args
            # 如果需要按键排序
            if keys:
                if keys != True:
                    # 使用指定的键排序参数，如果默认为False，则不进行排序
                    args = ordered(args, keys, default=False)
                else:
                    # 否则按默认顺序排序参数
                    args = ordered(args)
            # 对节点的每个参数执行递归前序遍历
            for arg in args:
                yield from self._preorder_traversal(arg, keys)
        # 如果节点是可迭代对象
        elif iterable(node):
            # 对节点的每个项执行递归前序遍历
            for item in node:
                yield from self._preorder_traversal(item, keys)

    # 方法：设置跳过标志以跳过当前节点的子树
    def skip(self):
        """
        Skip yielding current node's (last yielded node's) subtrees.

        Examples
        ========

        >>> from sympy import preorder_traversal, symbols
        >>> x, y, z = symbols('x y z')
        >>> pt = preorder_traversal((x + y*z)*z)
        >>> for i in pt:
        ...     print(i)
        ...     if i == x + y*z:
        ...             pt.skip()
        z*(x + y*z)
        z
        x + y*z
        """
        self._skip_flag = True

    # 方法：返回遍历器对象的下一个元素
    def __next__(self):
        return next(self._pt)

    # 方法：返回遍历器对象自身，使其成为一个可迭代对象
    def __iter__(self):
        return self
def use(expr, func, level=0, args=(), kwargs={}):
    """
    Use ``func`` to transform ``expr`` at the given level.

    Examples
    ========

    >>> from sympy import use, expand
    >>> from sympy.abc import x, y

    >>> f = (x + y)**2*x + 1

    >>> use(f, expand, level=2)
    x*(x**2 + 2*x*y + y**2) + 1
    >>> expand(f)
    x**3 + 2*x**2*y + x*y**2 + 1

    """
    # 定义内部函数 _use，递归处理表达式的变换
    def _use(expr, level):
        # 如果 level 为 0，则直接应用 func 函数对 expr 进行变换
        if not level:
            return func(expr, *args, **kwargs)
        else:
            # 如果 expr 是原子表达式，则不再继续向下递归
            if expr.is_Atom:
                return expr
            else:
                # 递归地降低 level 并应用 _use 函数到 expr 的每个参数上
                level -= 1
                _args = [_use(arg, level) for arg in expr.args]
                return expr.__class__(*_args)

    # 将输入的表达式转换为 SymPy 的表达式并应用 _use 函数
    return _use(sympify(expr), level)


def walk(e, *target):
    """Iterate through the args that are the given types (target) and
    return a list of the args that were traversed; arguments
    that are not of the specified types are not traversed.

    Examples
    ========

    >>> from sympy.core.traversal import walk
    >>> from sympy import Min, Max
    >>> from sympy.abc import x, y, z
    >>> list(walk(Min(x, Max(y, Min(1, z))), Min))
    [Min(x, Max(y, Min(1, z)))]
    >>> list(walk(Min(x, Max(y, Min(1, z))), Min, Max))
    [Min(x, Max(y, Min(1, z))), Max(y, Min(1, z)), Min(1, z)]

    See Also
    ========

    bottom_up
    """
    # 如果 e 是目标类型之一，生成当前节点 e 并遍历其所有参数
    if isinstance(e, target):
        yield e
        for i in e.args:
            yield from walk(i, *target)


def bottom_up(rv, F, atoms=False, nonbasic=False):
    """Apply ``F`` to all expressions in an expression tree from the
    bottom up. If ``atoms`` is True, apply ``F`` even if there are no args;
    if ``nonbasic`` is True, try to apply ``F`` to non-Basic objects.
    """
    # 获取 rv 的参数列表
    args = getattr(rv, 'args', None)
    if args is not None:
        if args:
            # 对 rv 的每个参数递归地应用 bottom_up 函数
            args = tuple([bottom_up(a, F, atoms, nonbasic) for a in args])
            # 如果参数列表发生改变，则创建新的 rv 对象
            if args != rv.args:
                rv = rv.func(*args)
            # 应用 F 函数到 rv 上
            rv = F(rv)
        elif atoms:
            # 如果 atoms 为 True，则即使没有参数也应用 F 函数到 rv 上
            rv = F(rv)
    else:
        if nonbasic:
            # 如果 nonbasic 为 True，尝试应用 F 函数到非 Basic 对象上
            try:
                rv = F(rv)
            except TypeError:
                pass

    return rv


def postorder_traversal(node, keys=None):
    """
    Do a postorder traversal of a tree.

    This generator recursively yields nodes that it has visited in a postorder
    fashion. That is, it descends through the tree depth-first to yield all of
    a node's children's postorder traversal before yielding the node itself.

    Parameters
    ==========

    node : SymPy expression
        The expression to traverse.
    keys : (default None) sort key(s)
        The key(s) used to sort args of Basic objects. When None, args of Basic
        objects are processed in arbitrary order. If key is defined, it will
        be passed along to ordered() as the only key(s) to use to sort the
        arguments; if ``key`` is simply True then the default keys of
        ``ordered`` will be used (node count and default_sort_key).

    """
    """
    如果节点是 SymPy 的基本表达式（Basic 类型），则进行后序遍历，并生成每个子树。

    Parameters
    ==========
    node : SymPy expression
        要遍历的节点或表达式。
    keys : bool or callable or None
        如果为 True，则确保遍历是唯一的；如果为 False 或者提供了自定义函数，则按指定方式排序子节点。

    Yields
    ======
    subtree : SymPy expression
        树中的所有子树。

    Examples
    ========
    
    >>> from sympy import postorder_traversal
    >>> from sympy.abc import w, x, y, z

    节点按照遇到的顺序返回，除非提供了 keys 参数；传递 key=True 将确保遍历的唯一性。

    >>> list(postorder_traversal(w + (x + y)*z)) # doctest: +SKIP
    [z, y, x, x + y, z*(x + y), w, w + z*(x + y)]
    >>> list(postorder_traversal(w + (x + y)*z, keys=True))
    [w, z, x, y, x + y, z*(x + y), w + z*(x + y)]
    """
    if isinstance(node, Basic):
        # 获取节点的所有参数
        args = node.args
        if keys:
            if keys != True:
                # 按照指定的键或默认方式排序参数
                args = ordered(args, keys, default=False)
            else:
                # 按默认方式排序参数
                args = ordered(args)
        # 对每个参数递归进行后序遍历
        for arg in args:
            yield from postorder_traversal(arg, keys)
    elif iterable(node):
        # 如果节点是可迭代的，则对每个元素递归进行后序遍历
        for item in node:
            yield from postorder_traversal(item, keys)
    # 最后生成当前节点
    yield node
```