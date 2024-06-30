# `D:\src\scipysrc\sympy\sympy\printing\dot.py`

```
# 导入必要的模块和类
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.numbers import Integer, Rational, Float
from sympy.printing.repr import srepr

# 声明需要导出的函数或类的列表
__all__ = ['dotprint']

# 默认的样式定义，将基本类映射到颜色和形状的字典
default_styles = (
    (Basic, {'color': 'blue', 'shape': 'ellipse'}),
    (Expr,  {'color': 'black'})
)

# 定义用于打印的对象类型
slotClasses = (Symbol, Integer, Rational, Float)

def purestr(x, with_args=False):
    """生成一个字符串，用以精确地重建对象

    Parameters
    ==========

    with_args : boolean, optional
        如果为 ``True``，则返回的第二个参数是一个元组，包含每个子节点应用
        ``purestr`` 后的结果。

        如果为 ``False``，则没有第二个参数。

        默认为 ``False``

    Examples
    ========

    >>> from sympy import Float, Symbol, MatrixSymbol
    >>> from sympy import Integer # noqa: F401
    >>> from sympy.core.symbol import Str # noqa: F401
    >>> from sympy.printing.dot import purestr

    对基本符号对象应用 ``purestr``：
    >>> code = purestr(Symbol('x'))
    >>> code
    "Symbol('x')"
    >>> eval(code) == Symbol('x')
    True

    对基本数值对象应用 ``purestr``：
    >>> purestr(Float(2))
    "Float('2.0', precision=53)"

    对矩阵符号应用 ``purestr``：
    >>> code = purestr(MatrixSymbol('x', 2, 2))
    >>> code
    "MatrixSymbol(Str('x'), Integer(2), Integer(2))"
    >>> eval(code) == MatrixSymbol('x', 2, 2)
    True

    当 ``with_args=True`` 时：
    >>> purestr(Float(2), with_args=True)
    ("Float('2.0', precision=53)", ())
    >>> purestr(MatrixSymbol('x', 2, 2), with_args=True)
    ("MatrixSymbol(Str('x'), Integer(2), Integer(2))",
     ("Str('x')", 'Integer(2)', 'Integer(2)'))
    """
    sargs = ()
    if not isinstance(x, Basic):
        rv = str(x)
    elif not x.args:
        rv = srepr(x)
    else:
        args = x.args
        sargs = tuple(map(purestr, args))
        rv = "%s(%s)"%(type(x).__name__, ', '.join(sargs))
    if with_args:
        rv = rv, sargs
    return rv


def styleof(expr, styles=default_styles):
    """按顺序合并样式字典

    Examples
    ========

    >>> from sympy import Symbol, Basic, Expr, S
    >>> from sympy.printing.dot import styleof
    >>> styles = [(Basic, {'color': 'blue', 'shape': 'ellipse'}),
    ...           (Expr,  {'color': 'black'})]

    >>> styleof(Basic(S(1)), styles)
    {'color': 'blue', 'shape': 'ellipse'}

    >>> x = Symbol('x')
    >>> styleof(x + 1, styles)  # 这是一个表达式
    {'color': 'black', 'shape': 'ellipse'}
    """
    style = {}
    for typ, sty in styles:
        if isinstance(expr, typ):
            style.update(sty)
    return style


def attrprint(d, delimiter=', '):
    """打印属性字典

    Examples
    ========

    >>> from sympy.printing.dot import attrprint
    >>> print(attrprint({'color': 'blue', 'shape': 'ellipse'}))
    "color"="blue", "shape"="ellipse"
    """
    # 将字典 d 中的键值对格式化为字符串，每对键值对使用 delimiter 分隔，最终使用 join 连接所有字符串
    return delimiter.join('"%s"="%s"' % item for item in sorted(d.items()))
def dotnode(expr, styles=default_styles, labelfunc=str, pos=(), repeat=True):
    """ String defining a node

    Examples
    ========

    >>> from sympy.printing.dot import dotnode
    >>> from sympy.abc import x
    >>> print(dotnode(x))
    "Symbol('x')_()" ["color"="black", "label"="x", "shape"="ellipse"];
    """
    # 根据表达式的类型确定标签
    style = styleof(expr, styles)

    # 如果表达式是 SymPy 的基本对象且不是原子对象，则使用类名作为标签
    if isinstance(expr, Basic) and not expr.is_Atom:
        label = str(expr.__class__.__name__)
    else:
        label = labelfunc(expr)
    style['label'] = label

    # 获取表达式的字符串表示
    expr_str = purestr(expr)

    # 如果 repeat 为 True，则在表达式字符串后面添加位置信息
    if repeat:
        expr_str += '_%s' % str(pos)

    # 返回符合 DOT 格式的节点定义字符串
    return '"%s" [%s];' % (expr_str, attrprint(style))


def dotedges(expr, atom=lambda x: not isinstance(x, Basic), pos=(), repeat=True):
    """ List of strings for all expr->expr.arg pairs

    See the docstring of dotprint for explanations of the options.

    Examples
    ========

    >>> from sympy.printing.dot import dotedges
    >>> from sympy.abc import x
    >>> for e in dotedges(x+2):
    ...     print(e)
    "Add(Integer(2), Symbol('x'))_()" -> "Integer(2)_(0,)";
    "Add(Integer(2), Symbol('x'))_()" -> "Symbol('x')_(1,)";
    """
    # 如果表达式是原子对象，则返回空列表
    if atom(expr):
        return []
    else:
        # 获取表达式和其参数的字符串表示
        expr_str, arg_strs = purestr(expr, with_args=True)

        # 如果 repeat 为 True，则在表达式字符串和参数字符串后面添加位置信息
        if repeat:
            expr_str += '_%s' % str(pos)
            arg_strs = ['%s_%s' % (a, str(pos + (i,)))
                for i, a in enumerate(arg_strs)]

        # 返回符合 DOT 格式的边定义字符串列表
        return ['"%s" -> "%s";' % (expr_str, a) for a in arg_strs]


template = \
"""digraph{

# Graph style
%(graphstyle)s

#########
# Nodes #
#########

%(nodes)s

#########
# Edges #
#########

%(edges)s
}"""

_graphstyle = {'rankdir': 'TD', 'ordering': 'out'}

def dotprint(expr,
    styles=default_styles, atom=lambda x: not isinstance(x, Basic),
    maxdepth=None, repeat=True, labelfunc=str, **kwargs):
    """DOT description of a SymPy expression tree

    Parameters
    ==========

    styles : list of lists composed of (Class, mapping), optional
        Styles for different classes.

        The default is

        .. code-block:: python

            (
                (Basic, {'color': 'blue', 'shape': 'ellipse'}),
                (Expr,  {'color': 'black'})
            )

    atom : function, optional
        Function used to determine if an arg is an atom.

        A good choice is ``lambda x: not x.args``.

        The default is ``lambda x: not isinstance(x, Basic)``.

    maxdepth : integer, optional
        The maximum depth.

        The default is ``None``, meaning no limit.
"""
    """
    repeat : boolean, optional
        是否对常见子表达式使用不同的节点。
    
        默认为 ``True``。
    
        例如，对于带有 ``repeat=True`` 的 ``x + x*y``，将会有两个 ``x`` 节点；对于 ``repeat=False``，将只有一个节点。
    
        .. warning::
            即使像 ``Pow(x, x)`` 中的 ``x`` 在同一对象中出现两次，它仍然只会被计算一次。
            因此，当 ``repeat=False`` 时，一个对象出去的箭头数量可能不等于其参数数量。
    
    labelfunc : function, optional
        用于为给定叶节点创建标签的函数。
    
        默认为 ``str``。
    
        另一个好的选择是 ``srepr``。
    
        例如，使用 ``str`` 时，``x + 1`` 的叶节点被标记为 ``x`` 和 ``1``。使用 ``srepr`` 时，它们分别标记为 ``Symbol('x')`` 和 ``Integer(1)``。
    
    **kwargs : optional
        额外的关键字参数作为图形的样式包含在内。
    
    Examples
    ========
    
    >>> from sympy import dotprint
    >>> from sympy.abc import x
    >>> print(dotprint(x+2)) # doctest: +NORMALIZE_WHITESPACE
    digraph{
    <BLANKLINE>
    # Graph style
    "ordering"="out"
    "rankdir"="TD"
    <BLANKLINE>
    #########
    # Nodes #
    #########
    <BLANKLINE>
    "Add(Integer(2), Symbol('x'))_()" ["color"="black", "label"="Add", "shape"="ellipse"];
    "Integer(2)_(0,)" ["color"="black", "label"="2", "shape"="ellipse"];
    "Symbol('x')_(1,)" ["color"="black", "label"="x", "shape"="ellipse"];
    <BLANKLINE>
    #########
    # Edges #
    #########
    <BLANKLINE>
    "Add(Integer(2), Symbol('x'))_()" -> "Integer(2)_(0,)";
    "Add(Integer(2), Symbol('x'))_()" -> "Symbol('x')_(1,)";
    }
    
    """
    # 根据传入的参数设置初始的图形样式
    graphstyle = _graphstyle.copy()
    graphstyle.update(kwargs)
    
    nodes = []
    edges = []
    
    def traverse(e, depth, pos=()):
        # 将当前节点加入节点列表，使用 dotnode 函数生成节点，并考虑重复性
        nodes.append(dotnode(e, styles, labelfunc=labelfunc, pos=pos, repeat=repeat))
        # 如果指定了最大深度，并且达到了最大深度，则返回
        if maxdepth and depth >= maxdepth:
            return
        # 扩展边列表，使用 dotedges 函数生成边，并考虑原子性和重复性
        edges.extend(dotedges(e, atom=atom, pos=pos, repeat=repeat))
        # 递归遍历当前节点的每个参数
        [traverse(arg, depth+1, pos + (i,)) for i, arg in enumerate(e.args) if not atom(arg)]
    
    # 调用 traverse 函数开始遍历表达式 expr，初始深度为 0
    traverse(expr, 0)
    
    # 返回带有替换内容的模板字符串
    return template%{'graphstyle': attrprint(graphstyle, delimiter='\n'),
                     'nodes': '\n'.join(nodes),
                     'edges': '\n'.join(edges)}
```