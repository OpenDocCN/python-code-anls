# `D:\src\scipysrc\sympy\sympy\simplify\epathtools.py`

```
"""
Tools for manipulation of expressions using paths.
"""

from sympy.core import Basic  # 导入 Basic 类

class EPath:
    r"""
    Manipulate expressions using paths.

    EPath grammar in EBNF notation::

        literal   ::= /[A-Za-z_][A-Za-z_0-9]*/
        number    ::= /-?\d+/
        type      ::= literal
        attribute ::= literal "?"
        all       ::= "*"
        slice     ::= "[" number? (":" number? (":" number?)?)? "]"
        range     ::= all | slice
        query     ::= (type | attribute) ("|" (type | attribute))*
        selector  ::= range | query range?
        path      ::= "/" selector ("/" selector)*

    See the docstring of the epath() function.

    """

    __slots__ = ("_path", "_epath")  # 定义了 EPath 类的 __slots__ 属性，限定实例的属性
    # 定义一个新的类构造方法，用于创建 EPath 对象
    def __new__(cls, path):
        """Construct new EPath. """
        # 如果 path 已经是 EPath 类型的对象，则直接返回该对象
        if isinstance(path, EPath):
            return path

        # 如果 path 为空，则抛出数值错误异常
        if not path:
            raise ValueError("empty EPath")

        # 复制 path 到 _path
        _path = path

        # 如果 path 以 '/' 开头，则去除开头的 '/'，否则抛出未实现的异常
        if path[0] == '/':
            path = path[1:]
        else:
            raise NotImplementedError("non-root EPath")

        # 初始化 epath 列表
        epath = []

        # 遍历 path 按 '/' 分割后的每个选择器
        for selector in path.split('/'):
            selector = selector.strip()

            # 如果选择器为空，则抛出数值错误异常
            if not selector:
                raise ValueError("empty selector")

            # 初始化 index 为 0
            index = 0

            # 遍历选择器的每个字符，计算符合条件的字符数
            for c in selector:
                if c.isalnum() or c in ('_', '|', '?'):
                    index += 1
                else:
                    break

            # 初始化 attrs 和 types 列表
            attrs = []
            types = []

            # 如果 index 大于 0，则处理选择器中的元素和类型
            if index:
                elements = selector[:index]
                selector = selector[index:]

                # 按 '|' 分割元素，并根据是否以 '?' 结尾判断是属性还是类型
                for element in elements.split('|'):
                    element = element.strip()

                    # 如果元素为空，则抛出数值错误异常
                    if not element:
                        raise ValueError("empty element")

                    if element.endswith('?'):
                        attrs.append(element[:-1])
                    else:
                        types.append(element)

            # 初始化 span 为 None
            span = None

            # 如果选择器为 '*'，则不做处理；否则解析选择器中的范围或切片信息
            if selector == '*':
                pass
            else:
                # 如果选择器以 '[' 开头
                if selector.startswith('['):
                    try:
                        i = selector.index(']')
                    except ValueError:
                        raise ValueError("expected ']', got EOL")

                    # 解析 '[' 和 ']' 之间的内容作为 span
                    _span, span = selector[1:i], []

                    # 如果 ':' 不在 _span 中，则将 span 转为整数
                    if ':' not in _span:
                        span = int(_span)
                    else:
                        # 否则按 ':' 分割，处理成切片对象
                        for elt in _span.split(':', 3):
                            if not elt:
                                span.append(None)
                            else:
                                span.append(int(elt))

                        span = slice(*span)

                    # 去除选择器中解析过的部分
                    selector = selector[i + 1:]

                # 如果选择器不为空，则抛出数值错误异常
                if selector:
                    raise ValueError("trailing characters in selector")

            # 将处理后的 attrs、types、span 组成的元组加入 epath 列表
            epath.append((attrs, types, span))

        # 使用 object 的构造方法创建对象
        obj = object.__new__(cls)

        # 将 _path 和 epath 分别赋值给对象的 _path 和 _epath 属性
        obj._path = _path
        obj._epath = epath

        # 返回创建的对象
        return obj

    # 定义对象的字符串表示方法，返回类名和 _path 属性的字符串表示
    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self._path)

    # 定义对象方法，用于排序 expr.args 中的元素
    def _get_ordered_args(self, expr):
        """Sort ``expr.args`` using printing order. """
        # 如果 expr 是加法表达式，则按顺序返回其项
        if expr.is_Add:
            return expr.as_ordered_terms()
        # 如果 expr 是乘法表达式，则按顺序返回其因子
        elif expr.is_Mul:
            return expr.as_ordered_factors()
        else:
            # 其他情况直接返回 expr.args
            return expr.args

    # 定义对象方法，检查 expr 是否具有 attrs 中的任何属性
    def _hasattrs(self, expr, attrs) -> bool:
        """Check if ``expr`` has any of ``attrs``. """
        # 判断 expr 是否具有 attrs 中所有属性
        return all(hasattr(expr, attr) for attr in attrs)
    def _hastypes(self, expr, types):
        """Check if ``expr`` is any of ``types``. """
        # 获取表达式 expr 的类及其继承关系的类名列表
        _types = [ cls.__name__ for cls in expr.__class__.mro() ]
        # 返回表达式是否与给定类型集合 types 有交集的布尔值
        return bool(set(_types).intersection(types))

    def _has(self, expr, attrs, types):
        """Apply ``_hasattrs`` and ``_hastypes`` to ``expr``. """
        # 如果没有要检查的属性和类型，则直接返回 True
        if not (attrs or types):
            return True

        # 如果有要检查的属性 attrs，并且表达式 expr 具有这些属性，则返回 True
        if attrs and self._hasattrs(expr, attrs):
            return True

        # 如果有要检查的类型 types，并且表达式 expr 是其中任何一个类型，则返回 True
        if types and self._hastypes(expr, types):
            return True

        # 否则返回 False
        return False

    def apply(self, expr, func, args=None, kwargs=None):
        """
        Modify parts of an expression selected by a path.

        Examples
        ========

        >>> from sympy.simplify.epathtools import EPath
        >>> from sympy import sin, cos, E
        >>> from sympy.abc import x, y, z, t

        >>> path = EPath("/*/[0]/Symbol")
        >>> expr = [((x, 1), 2), ((3, y), z)]

        >>> path.apply(expr, lambda expr: expr**2)
        [((x**2, 1), 2), ((3, y**2), z)]

        >>> path = EPath("/*/*/Symbol")
        >>> expr = t + sin(x + 1) + cos(x + y + E)

        >>> path.apply(expr, lambda expr: 2*expr)
        t + sin(2*x + 1) + cos(2*x + 2*y + E)

        """
        def _apply(path, expr, func):
            # 如果路径 path 已经遍历完毕，则直接应用 func 到 expr 上并返回结果
            if not path:
                return func(expr)
            else:
                # 否则取出路径的第一个选择器 selector，并处理剩余路径 path
                selector, path = path[0], path[1:]
                attrs, types, span = selector

                # 如果表达式是 Basic 类型
                if isinstance(expr, Basic):
                    # 如果不是原子表达式，则获取其有序参数列表
                    if not expr.is_Atom:
                        args, basic = self._get_ordered_args(expr), True
                    else:
                        return expr
                # 如果是可迭代对象
                elif hasattr(expr, '__iter__'):
                    args, basic = expr, False
                else:
                    return expr

                args = list(args)

                # 处理 span 的情况
                if span is not None:
                    if isinstance(span, slice):
                        indices = range(*span.indices(len(args)))
                    else:
                        indices = [span]
                else:
                    indices = range(len(args))

                # 遍历索引，处理每个参数
                for i in indices:
                    try:
                        arg = args[i]
                    except IndexError:
                        continue

                    # 如果参数满足要求的属性 attrs 和类型 types，则递归应用 _apply 函数
                    if self._has(arg, attrs, types):
                        args[i] = _apply(path, arg, func)

                # 根据是否为基本表达式类型，返回处理后的表达式结果
                if basic:
                    return expr.func(*args)
                else:
                    return expr.__class__(args)

        _args, _kwargs = args or (), kwargs or {}
        _func = lambda expr: func(expr, *_args, **_kwargs)

        # 调用内部定义的 _apply 函数，传入路径和表达式 expr，并应用 func 函数进行修改
        return _apply(self._epath, expr, _func)
    # 定义一个方法，用于从表达式中选择路径指定的部分
    def select(self, expr):
        """
        Retrieve parts of an expression selected by a path.

        Examples
        ========

        >>> from sympy.simplify.epathtools import EPath
        >>> from sympy import sin, cos, E
        >>> from sympy.abc import x, y, z, t

        >>> path = EPath("/*/[0]/Symbol")
        >>> expr = [((x, 1), 2), ((3, y), z)]

        >>> path.select(expr)
        [x, y]

        >>> path = EPath("/*/*/Symbol")
        >>> expr = t + sin(x + 1) + cos(x + y + E)

        >>> path.select(expr)
        [x, x, y]

        """
        # 初始化一个空列表，用于存储选取的结果
        result = []

        # 定义一个内部递归函数，用于实际执行路径选择
        def _select(path, expr):
            # 如果路径为空，则将当前表达式添加到结果列表中
            if not path:
                result.append(expr)
            else:
                # 取出路径的第一个选择器和剩余路径
                selector, path = path[0], path[1:]
                attrs, types, span = selector

                # 如果表达式是一个 Basic 对象
                if isinstance(expr, Basic):
                    # 获取表达式的有序参数
                    args = self._get_ordered_args(expr)
                # 如果表达式可以迭代
                elif hasattr(expr, '__iter__'):
                    args = expr
                else:
                    return

                # 如果指定了范围 span
                if span is not None:
                    # 如果 span 是一个切片对象
                    if isinstance(span, slice):
                        args = args[span]
                    else:
                        # 尝试获取指定索引的参数
                        try:
                            args = [args[span]]
                        except IndexError:
                            return

                # 遍历当前表达式的参数
                for arg in args:
                    # 如果当前参数满足指定的属性和类型要求
                    if self._has(arg, attrs, types):
                        # 递归调用 _select 函数，继续深入下一级路径选择
                        _select(path, arg)

        # 调用内部递归函数，从根路径开始执行选择
        _select(self._epath, expr)
        # 返回最终选取的结果列表
        return result
# 定义函数 epath，用于操作由路径选择的表达式的部分
def epath(path, expr=None, func=None, args=None, kwargs=None):
    """
    Manipulate parts of an expression selected by a path.

    Explanation
    ===========

    This function allows to manipulate large nested expressions in single
    line of code, utilizing techniques to those applied in XML processing
    standards (e.g. XPath).

    If ``func`` is ``None``, :func:`epath` retrieves elements selected by
    the ``path``. Otherwise it applies ``func`` to each matching element.

    Note that it is more efficient to create an EPath object and use the select
    and apply methods of that object, since this will compile the path string
    only once.  This function should only be used as a convenient shortcut for
    interactive use.

    This is the supported syntax:

    * select all: ``/*``
          Equivalent of ``for arg in args:``.
    * select slice: ``/[0]`` or ``/[1:5]`` or ``/[1:5:2]``
          Supports standard Python's slice syntax.
    * select by type: ``/list`` or ``/list|tuple``
          Emulates ``isinstance()``.
    * select by attribute: ``/__iter__?``
          Emulates ``hasattr()``.

    Parameters
    ==========

    path : str | EPath
        A path as a string or a compiled EPath.
    expr : Basic | iterable
        An expression or a container of expressions.
    func : callable (optional)
        A callable that will be applied to matching parts.
    args : tuple (optional)
        Additional positional arguments to ``func``.
    kwargs : dict (optional)
        Additional keyword arguments to ``func``.

    Examples
    ========

    >>> from sympy.simplify.epathtools import epath
    >>> from sympy import sin, cos, E
    >>> from sympy.abc import x, y, z, t

    >>> path = "/*/[0]/Symbol"
    >>> expr = [((x, 1), 2), ((3, y), z)]

    >>> epath(path, expr)
    [x, y]
    >>> epath(path, expr, lambda expr: expr**2)
    [((x**2, 1), 2), ((3, y**2), z)]

    >>> path = "/*/*/Symbol"
    >>> expr = t + sin(x + 1) + cos(x + y + E)

    >>> epath(path, expr)
    [x, x, y]
    >>> epath(path, expr, lambda expr: 2*expr)
    t + sin(2*x + 1) + cos(2*x + 2*y + E)

    """
    # 创建一个 EPath 对象，用给定的路径 path 初始化
    _epath = EPath(path)

    # 如果没有给定表达式 expr，则直接返回 _epath 对象
    if expr is None:
        return _epath
    # 如果没有给定函数 func，则调用 _epath 对象的 select 方法
    if func is None:
        return _epath.select(expr)
    # 否则，调用 _epath 对象的 apply 方法，将 func 应用于匹配的部分
    else:
        return _epath.apply(expr, func, args, kwargs)
```