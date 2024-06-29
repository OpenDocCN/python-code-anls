# `D:\src\scipysrc\pandas\pandas\core\computation\eval.py`

```
"""
Top level ``eval`` module.
"""

# 从未来导入注解，以支持类型提示中的字符串字面值类型检查
from __future__ import annotations

# 导入 tokenize 模块用于分析 Python 源代码
import tokenize
# 导入 TYPE_CHECKING 用于类型检查时避免循环导入问题，以及 Any 类型用于通用类型注解
from typing import (
    TYPE_CHECKING,
    Any,
)
# 导入警告模块
import warnings

# 导入 pandas.util._exceptions 模块中的 find_stack_level 函数
from pandas.util._exceptions import find_stack_level
# 导入 pandas.util._validators 模块中的 validate_bool_kwarg 函数
from pandas.util._validators import validate_bool_kwarg

# 导入 pandas.core.dtypes.common 模块中的 is_extension_array_dtype 函数
from pandas.core.dtypes.common import is_extension_array_dtype

# 导入 pandas.core.computation.engines 模块中的 ENGINES 对象
from pandas.core.computation.engines import ENGINES
# 导入 pandas.core.computation.expr 模块中的 PARSERS 和 Expr 类
from pandas.core.computation.expr import (
    PARSERS,
    Expr,
)
# 导入 pandas.core.computation.parsing 模块中的 tokenize_string 函数
from pandas.core.computation.parsing import tokenize_string
# 导入 pandas.core.computation.scope 模块中的 ensure_scope 函数
from pandas.core.computation.scope import ensure_scope
# 导入 pandas.core.generic 模块中的 NDFrame 类
from pandas.core.generic import NDFrame

# 导入 pandas.io.formats.printing 模块中的 pprint_thing 函数
from pandas.io.formats.printing import pprint_thing

# 如果在类型检查模式下，导入 BinOp 类
if TYPE_CHECKING:
    from pandas.core.computation.ops import BinOp


# 检查并返回有效的计算引擎名称
def _check_engine(engine: str | None) -> str:
    """
    Make sure a valid engine is passed.

    Parameters
    ----------
    engine : str
        String to validate.

    Raises
    ------
    KeyError
      * If an invalid engine is passed.
    ImportError
      * If numexpr was requested but doesn't exist.

    Returns
    -------
    str
        Engine name.
    """
    # 导入 NUMEXPR_INSTALLED 和 USE_NUMEXPR 变量
    from pandas.core.computation.check import NUMEXPR_INSTALLED
    from pandas.core.computation.expressions import USE_NUMEXPR

    # 如果 engine 为 None，则根据 USE_NUMEXPR 确定使用的默认引擎
    if engine is None:
        engine = "numexpr" if USE_NUMEXPR else "python"

    # 如果 engine 不在 ENGINES 中，抛出 KeyError 异常
    if engine not in ENGINES:
        valid_engines = list(ENGINES.keys())
        raise KeyError(
            f"Invalid engine '{engine}' passed, valid engines are {valid_engines}"
        )

    # 对于 engine 为 'numexpr'，但未安装 numexpr 库时，抛出 ImportError 异常
    if engine == "numexpr" and not NUMEXPR_INSTALLED:
        raise ImportError(
            "'numexpr' is not installed or an unsupported version. Cannot use "
            "engine='numexpr' for query/eval if 'numexpr' is not installed"
        )

    # 返回有效的引擎名称
    return engine


# 检查并确保传入的解析器名称有效
def _check_parser(parser: str) -> None:
    """
    Make sure a valid parser is passed.

    Parameters
    ----------
    parser : str

    Raises
    ------
    KeyError
      * If an invalid parser is passed
    """
    # 如果 parser 不在 PARSERS 中，抛出 KeyError 异常
    if parser not in PARSERS:
        raise KeyError(
            f"Invalid parser '{parser}' passed, valid parsers are {PARSERS.keys()}"
        )


# 检查解析器是否具有正确的方法
def _check_resolvers(resolvers) -> None:
    if resolvers is not None:
        for resolver in resolvers:
            # 如果 resolver 没有 __getitem__ 方法，抛出 TypeError 异常
            if not hasattr(resolver, "__getitem__"):
                name = type(resolver).__name__
                raise TypeError(
                    f"Resolver of type '{name}' does not "
                    "implement the __getitem__ method"
                )


# 确保表达式不为空字符串
def _check_expression(expr) -> None:
    """
    Make sure an expression is not an empty string

    Parameters
    ----------
    expr : object
        An object that can be converted to a string

    Raises
    ------
    ValueError
      * If expr is an empty string
    """
    # 如果条件表达式 expr 为假（即空字符串或者其他等效的假值），则抛出 ValueError 异常
    if not expr:
        raise ValueError("expr cannot be an empty string")
# 将对象转换为表达式字符串

def _convert_expression(expr) -> str:
    """
    Convert an object to an expression.

    This function converts an object to an expression (a unicode string) and
    checks to make sure it isn't empty after conversion. This is used to
    convert operators to their string representation for recursive calls to
    :func:`~pandas.eval`.

    Parameters
    ----------
    expr : object
        The object to be converted to a string.

    Returns
    -------
    str
        The string representation of an object.

    Raises
    ------
    ValueError
        If the expression is empty.
    """
    # 使用 pprint_thing 函数将表达式转换为字符串
    s = pprint_thing(expr)
    # 检查转换后的字符串是否为空
    _check_expression(s)
    # 返回表达式的字符串表示
    return s


# 检查表达式中是否包含不支持的局部变量前缀

def _check_for_locals(expr: str, stack_level: int, parser: str) -> None:
    # 检查是否在调用堆栈的顶部
    at_top_of_stack = stack_level == 0
    # 检查是否不是 pandas 解析器
    not_pandas_parser = parser != "pandas"

    # 如果不是 pandas 解析器，则抛出异常
    if not_pandas_parser:
        msg = "The '@' prefix is only supported by the pandas parser"
    # 如果在调用堆栈的顶部，则抛出异常
    elif at_top_of_stack:
        msg = (
            "The '@' prefix is not allowed in top-level eval calls.\n"
            "please refer to your variables by name without the '@' prefix."
        )

    # 遍历表达式中的 token，检查是否包含 '@' 符号，并根据情况抛出 SyntaxError
    if at_top_of_stack or not_pandas_parser:
        for toknum, tokval in tokenize_string(expr):
            if toknum == tokenize.OP and tokval == "@":
                raise SyntaxError(msg)


# 使用不同后端评估 Python 表达式字符串

def eval(
    expr: str | BinOp,  # BinOp 不包含在文档字符串中，因为它不适用于用户
    parser: str = "pandas",
    engine: str | None = None,
    local_dict=None,
    global_dict=None,
    resolvers=(),
    level: int = 0,
    target=None,
    inplace: bool = False,
) -> Any:
    """
    Evaluate a Python expression as a string using various backends.

    The following arithmetic operations are supported: ``+``, ``-``, ``*``,
    ``/``, ``**``, ``%``, ``//`` (python engine only) along with the following
    boolean operations: ``|`` (or), ``&`` (and), and ``~`` (not).
    Additionally, the ``'pandas'`` parser allows the use of :keyword:`and`,
    :keyword:`or`, and :keyword:`not` with the same semantics as the
    corresponding bitwise operators.  :class:`~pandas.Series` and
    :class:`~pandas.DataFrame` objects are supported and behave as they would
    with plain ol' Python evaluation.

    .. warning::

        ``eval`` can run arbitrary code which can make you vulnerable to code
         injection and untrusted data.

    Parameters
    ----------
    expr : str
        The expression to evaluate. This string cannot contain any Python
        `statements
        <https://docs.python.org/3/reference/simple_stmts.html#simple-statements>`__,
        only Python `expressions
        <https://docs.python.org/3/reference/simple_stmts.html#expression-statements>`__.
    parser : str, optional
        The parser to use for evaluating the expression (default is 'pandas').
    engine : str or None, optional
        The engine to use for evaluation (default is None).
    local_dict : dict, optional
        Dictionary of local variables to use in evaluation (default is None).
    global_dict : dict, optional
        Dictionary of global variables to use in evaluation (default is None).
    resolvers : tuple, optional
        Resolvers to use for variable lookups (default is an empty tuple).
    level : int, optional
        Evaluation level (default is 0).
    target : object, optional
        Target object for evaluation (default is None).
    inplace : bool, optional
        Whether to perform the evaluation inplace (default is False).

    Returns
    -------
    Any
        The result of the evaluated expression.

    Raises
    ------
    SyntaxError
        If the expression contains unsupported '@' prefixes.
    """
    pass  # 函数体未提供，故这里暂时不需要添加注释
    parser : {'pandas', 'python'}, default 'pandas'
        # 解析器类型参数，控制从表达式构造语法树的方法
        # 默认为 'pandas'，与标准 Python 稍有不同
        # 也可以选择 'python' 解析器以保持严格的 Python 语义

    engine : {'python', 'numexpr'}, default 'numexpr'
        # 表达式求值引擎参数
        # 支持的引擎有：
        # - None：尝试使用 'numexpr'，如果失败则回退到 'python'
        # - 'numexpr'：针对大型数据帧和复杂表达式提供了大幅的性能提升
        # - 'python'：以顶层 Python 的 eval 进行操作，一般不太有用

    local_dict : dict or None, optional
        # 局部变量字典参数，可选
        # 默认为 locals() 中的变量字典

    global_dict : dict or None, optional
        # 全局变量字典参数，可选
        # 默认为 globals() 中的变量字典

    resolvers : list of dict-like or None, optional
        # 解析器列表参数，用于注入额外的命名空间供变量查找
        # 例如，DataFrame.query 方法中注入了 DataFrame.index 和 DataFrame.columns

    level : int, optional
        # 回溯堆栈帧的层数参数，大多数用户不需要更改此参数

    target : object, optional, default None
        # 赋值的目标对象参数
        # 当表达式中有变量赋值时使用，target 必须支持字符串键的项赋值
        # 如果需要返回一个副本，则 target 还必须支持 .copy() 方法

    inplace : bool, default False
        # 如果提供了 target，并且表达式会修改 target，则控制是否原地修改 target
        # 否则返回 target 的副本，并且返回 None（如果 inplace=True）

    Returns
    -------
    ndarray, numeric scalar, DataFrame, Series, or None
        # 表达式求值后的结果类型
        # 可能返回 ndarray、数值标量、DataFrame、Series 或 None

    Raises
    ------
    inplace = validate_bool_kwarg(inplace, "inplace")
    # 验证并处理 inplace 参数，确保其为布尔值

    exprs: list[str | BinOp]
    # 声明一个表达式列表，每个表达式可以是字符串或二元操作对象

    if isinstance(expr, str):
        # 如果 expr 是字符串类型
        _check_expression(expr)
        # 检查表达式的有效性
        exprs = [e.strip() for e in expr.splitlines() if e.strip() != ""]
        # 将表达式按行分割并去除首尾空白字符，形成表达式列表
    else:
        # 如果 expr 不是字符串类型
        # ops.BinOp; for internal compat, not intended to be passed by users
        exprs = [expr]
        # 将 expr 直接作为一个元素放入表达式列表中

    multi_line = len(exprs) > 1
    # 判断是否为多行表达式

    if multi_line and target is None:
        # 如果是多行表达式且目标对象为空
        raise ValueError(
            "multi-line expressions are only valid in the "
            "context of data, use DataFrame.eval"
        )
        # 抛出数值错误，说明多行表达式只能在数据的上下文中有效，应使用 DataFrame.eval

    engine = _check_engine(engine)
    # 检查并获取引擎对象

    _check_parser(parser)
    # 检查并处理解析器对象

    _check_resolvers(resolvers)
    # 检查并处理解析器对象

    ret = None
    # 初始化返回值为 None
    first_expr = True
    # 初始化首个表达式标记为 True
    target_modified = False
    # 初始化目标对象是否被修改的标记为 False

    if inplace is False:
        # 如果 inplace 参数为 False
        return target if target_modified else ret
        # 如果目标对象被修改则返回目标对象，否则返回 None
```