# `D:\src\scipysrc\sympy\sympy\parsing\latex\__init__.py`

```
# 从 sympy.external 模块导入 import_module 函数
# 从 sympy.utilities.decorator 模块导入 doctest_depends_on 装饰器
from sympy.external import import_module
from sympy.utilities.decorator import doctest_depends_on

# 从 sympy.parsing.latex.lark 模块导入 LarkLaTeXParser、TransformToSymPyExpr、parse_latex_lark 函数
from sympy.parsing.latex.lark import LarkLaTeXParser, TransformToSymPyExpr, parse_latex_lark  # noqa

# 从当前目录的 errors 模块导入 LaTeXParsingError 类
from .errors import LaTeXParsingError  # noqa

# 定义 __doctest_requires__ 字典，指明运行 parse_latex 函数需要的外部依赖模块
__doctest_requires__ = {('parse_latex',): ['antlr4', 'lark']}

# 使用 doctest_depends_on 装饰器，指定函数 parse_latex 的依赖模块为 ('antlr4', 'lark')
@doctest_depends_on(modules=('antlr4', 'lark'))
# 定义函数 parse_latex，将输入的 LaTeX 字符串 s 转换为 SymPy 表达式 Expr
def parse_latex(s, strict=False, backend="antlr"):
    r"""Converts the input LaTeX string ``s`` to a SymPy ``Expr``.

    Parameters
    ==========

    s : str
        The LaTeX string to parse. In Python source containing LaTeX,
        *raw strings* (denoted with ``r"``, like this one) are preferred,
        as LaTeX makes liberal use of the ``\`` character, which would
        trigger escaping in normal Python strings.
    backend : str, optional
        Currently, there are two backends supported: ANTLR, and Lark.
        The default setting is to use the ANTLR backend, which can be
        changed to Lark if preferred.

        Use ``backend="antlr"`` for the ANTLR-based parser, and
        ``backend="lark"`` for the Lark-based parser.

        The ``backend`` option is case-sensitive, and must be in
        all lowercase.
    strict : bool, optional
        This option is only available with the ANTLR backend.

        If True, raise an exception if the string cannot be parsed as
        valid LaTeX. If False, try to recover gracefully from common
        mistakes.

    Examples
    ========

    >>> from sympy.parsing.latex import parse_latex
    >>> expr = parse_latex(r"\frac {1 + \sqrt {\a}} {\b}")
    >>> expr
    (sqrt(a) + 1)/b
    >>> expr.evalf(4, subs=dict(a=5, b=2))
    1.618
    >>> func = parse_latex(r"\int_1^\alpha \dfrac{\mathrm{d}t}{t}", backend="lark")
    >>> func.evalf(subs={"alpha": 2})
    0.693147180559945
    """

    # 根据 backend 参数选择合适的 LaTeX 解析后端
    if backend == "antlr":
        # 使用 import_module 函数导入 sympy.parsing.latex._parse_latex_antlr 模块
        _latex = import_module(
            'sympy.parsing.latex._parse_latex_antlr',
            import_kwargs={'fromlist': ['X']})

        # 如果成功导入 _latex 模块，则调用其 parse_latex 函数解析 LaTeX 字符串 s
        if _latex is not None:
            return _latex.parse_latex(s, strict)
    # 如果 backend 参数为 "lark"，则调用 parse_latex_lark 函数解析 LaTeX 字符串 s
    elif backend == "lark":
        return parse_latex_lark(s)
    else:
        # 如果 backend 参数既不是 "antlr" 也不是 "lark"，抛出 NotImplementedError 异常
        raise NotImplementedError(f"Using the '{backend}' backend in the LaTeX" \
                                   " parser is not supported.")
```