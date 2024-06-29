# `D:\src\scipysrc\pandas\pandas\core\computation\expr.py`

```
"""
:func:`~pandas.eval` parsers.
"""

# 引入未来的注解特性，允许在类型提示中使用字符串形式的类型名
from __future__ import annotations

# 引入标准库模块
import ast
from functools import (
    partial,
    reduce,
)
from keyword import iskeyword
import tokenize
from typing import (
    TYPE_CHECKING,
    ClassVar,
    TypeVar,
)

# 引入第三方库
import numpy as np

# 引入 pandas 错误处理模块
from pandas.errors import UndefinedVariableError

# 引入 pandas 核心通用模块
import pandas.core.common as com

# 引入 pandas 计算操作模块
from pandas.core.computation.ops import (
    ARITH_OPS_SYMS,
    BOOL_OPS_SYMS,
    CMP_OPS_SYMS,
    LOCAL_TAG,
    MATHOPS,
    REDUCTIONS,
    UNARY_OPS_SYMS,
    BinOp,
    Constant,
    Div,
    FuncNode,
    Op,
    Term,
    UnaryOp,
    is_term,
)

# 引入 pandas 计算解析模块
from pandas.core.computation.parsing import (
    clean_backtick_quoted_toks,
    tokenize_string,
)

# 引入 pandas 计算作用域模块
from pandas.core.computation.scope import Scope

# 引入 pandas 格式化输出模块
from pandas.io.formats import printing

# 如果是类型检查模式，引入 collections.abc 的 Callable 类型
if TYPE_CHECKING:
    from collections.abc import Callable


# 定义函数 _rewrite_assign，用于重写 PyTables 表达式中使用 '=' 替代 '==' 的赋值运算符
def _rewrite_assign(tok: tuple[int, str]) -> tuple[int, str]:
    """
    Rewrite the assignment operator for PyTables expressions that use ``=``
    as a substitute for ``==``.

    Parameters
    ----------
    tok : tuple of int, str
        ints correspond to the all caps constants in the tokenize module

    Returns
    -------
    tuple of int, str
        Either the input or token or the replacement values
    """
    # 解构元组，获取 token 类型和值
    toknum, tokval = tok
    # 如果 token 值是 '='，替换为 '=='
    return toknum, "==" if tokval == "=" else tokval


# 定义函数 _replace_booleans，用于将位运算符 '&' 替换为 'and'，'|' 替换为 'or'，调整运算优先级为布尔优先级
def _replace_booleans(tok: tuple[int, str]) -> tuple[int, str]:
    """
    Replace ``&`` with ``and`` and ``|`` with ``or`` so that bitwise
    precedence is changed to boolean precedence.

    Parameters
    ----------
    tok : tuple of int, str
        ints correspond to the all caps constants in the tokenize module

    Returns
    -------
    tuple of int, str
        Either the input or token or the replacement values
    """
    # 解构元组，获取 token 类型和值
    toknum, tokval = tok
    # 如果 token 类型是操作符且值为 '&'
    if toknum == tokenize.OP:
        if tokval == "&":
            # 替换为 'and'
            return tokenize.NAME, "and"
        elif tokval == "|":
            # 替换为 'or'
            return tokenize.NAME, "or"
        # 其他情况保持不变
        return toknum, tokval
    # 非操作符类型保持不变
    return toknum, tokval


# 定义函数 _replace_locals，用于将局部变量替换为语法上有效的名称
def _replace_locals(tok: tuple[int, str]) -> tuple[int, str]:
    """
    Replace local variables with a syntactically valid name.

    Parameters
    ----------
    tok : tuple of int, str
        ints correspond to the all caps constants in the tokenize module

    Returns
    -------
    tuple of int, str
        Either the input or token or the replacement values

    Notes
    -----
    This is somewhat of a hack in that we rewrite a string such as ``'@a'`` as
    ``'__pd_eval_local_a'`` by telling the tokenizer that ``__pd_eval_local_``
    is a ``tokenize.OP`` and to replace the ``'@'`` symbol with it.
    """
    # 解构元组，获取 token 类型和值
    toknum, tokval = tok
    # 如果 token 类型是操作符且值为 '@'
    if toknum == tokenize.OP and tokval == "@":
        # 替换为定义的 LOCAL_TAG
        return tokenize.OP, LOCAL_TAG
    # 其他情况保持不变
    return toknum, tokval


# 定义函数 _compose2，用于组合两个可调用对象
def _compose2(f, g):
    """
    Compose 2 callables.
    """
    # 返回一个 lambda 函数，实现对两个函数的组合调用
    return lambda *args, **kwargs: f(g(*args, **kwargs))


# 定义函数 _compose，用于组合两个或多个可调用对象
def _compose(*funcs):
    """
    Compose 2 or more callables.
    """
    """
    # 使用断言确保传入的函数列表 funcs 至少包含两个可调用对象，否则抛出异常
    assert len(funcs) > 1, "At least 2 callables must be passed to compose"
    # 使用 reduce 函数，依次将 funcs 中的函数进行组合，得到最终的组合函数
    return reduce(_compose2, funcs)
# 定义一个函数 `_preparse`，用于预处理 Python 源代码字符串
def _preparse(
    source: str,
    # 参数 `f` 是一个可调用对象，用于处理标记化后的元组 (toknum, tokval)，默认为 `_rewrite_assign`、`_replace_booleans` 和 `_replace_locals` 的组合
    f=_compose(
        _replace_locals, _replace_booleans, _rewrite_assign, clean_backtick_quoted_toks
    ),
) -> str:
    """
    Compose a collection of tokenization functions.

    Parameters
    ----------
    source : str
        A Python source code string
    f : callable
        This takes a tuple of (toknum, tokval) as its argument and returns a
        tuple with the same structure but possibly different elements. Defaults
        to the composition of ``_rewrite_assign``, ``_replace_booleans``, and
        ``_replace_locals``.

    Returns
    -------
    str
        Valid Python source code

    Notes
    -----
    The `f` parameter can be any callable that takes *and* returns input of the
    form ``(toknum, tokval)``, where ``toknum`` is one of the constants from
    the ``tokenize`` module and ``tokval`` is a string.
    """
    # 断言 `f` 必须是可调用对象
    assert callable(f), "f must be callable"
    # 返回经过标记化处理后的 Python 源代码字符串
    return tokenize.untokenize(f(x) for x in tokenize_string(source))


# 定义一个函数工厂 `_is_type`，用于创建一个类型检查函数，检查对象是否为指定类型 `t` 或类型元组
def _is_type(t):
    """
    Factory for a type checking function of type ``t`` or tuple of types.
    """
    return lambda x: isinstance(x.value, t)


# 创建类型检查函数 `_is_list` 和 `_is_str`，分别用于检查对象是否为 list 和 str 类型
_is_list = _is_type(list)
_is_str = _is_type(str)


# 创建一个不可变集合 `_all_nodes`，包含所有 AST 节点类型
_all_nodes = frozenset(
    node
    for node in (getattr(ast, name) for name in dir(ast))
    if isinstance(node, type) and issubclass(node, ast.AST)
)


# 定义函数 `_filter_nodes`，用于筛选出 AST 节点类型中是指定超类 `superclass` 的子类的节点
def _filter_nodes(superclass, all_nodes=_all_nodes):
    """
    Filter out AST nodes that are subclasses of ``superclass``.
    """
    # 生成所有满足条件的节点类型名称的不可变集合
    node_names = (node.__name__ for node in all_nodes if issubclass(node, superclass))
    return frozenset(node_names)


# 创建多个不可变集合，包含特定类型的 AST 节点名称，例如 `_mod_nodes`、`_stmt_nodes` 等
_all_node_names = frozenset(x.__name__ for x in _all_nodes)
_mod_nodes = _filter_nodes(ast.mod)
_stmt_nodes = _filter_nodes(ast.stmt)
_expr_nodes = _filter_nodes(ast.expr)
_expr_context_nodes = _filter_nodes(ast.expr_context)
_boolop_nodes = _filter_nodes(ast.boolop)
_operator_nodes = _filter_nodes(ast.operator)
_unary_op_nodes = _filter_nodes(ast.unaryop)
_cmp_op_nodes = _filter_nodes(ast.cmpop)
_comprehension_nodes = _filter_nodes(ast.comprehension)
_handler_nodes = _filter_nodes(ast.excepthandler)
_arguments_nodes = _filter_nodes(ast.arguments)
_keyword_nodes = _filter_nodes(ast.keyword)
_alias_nodes = _filter_nodes(ast.alias)


# 创建一个不可变集合 `_hacked_nodes`，包含一些特定的 AST 节点类型名称，这些类型不直接支持但需要用于解析
_hacked_nodes = frozenset(["Assign", "Module", "Expr"])


# 创建一个不可变集合 `_unsupported_expr_nodes`，包含不支持的表达式类型的名称
_unsupported_expr_nodes = frozenset(
    [
        "Yield",
        "GeneratorExp",
        "IfExp",
        "DictComp",
        "SetComp",
        "Repr",
        "Lambda",
        "Set",
        "AST",
        "Is",
        "IsNot",
    ]
)

# 创建一个不可变集合 `_unsupported_nodes`，包含所有不支持的节点类型名称
_unsupported_nodes = (
    _stmt_nodes
    | _mod_nodes
    | _handler_nodes
    | _arguments_nodes
    | _keyword_nodes
    | _alias_nodes
    | _expr_context_nodes
    | _unsupported_expr_nodes
) - _hacked_nodes
# 从一些节点中移除不支持的节点，这些节点由 `_unsupported_nodes` 定义，
# 同时将 `_base_supported_nodes` 和 `_hacked_nodes` 的并集作为基本支持节点
_base_supported_nodes = (_all_node_names - _unsupported_nodes) | _hacked_nodes
# 计算 `_unsupported_nodes` 和 `_base_supported_nodes` 的交集
intersection = _unsupported_nodes & _base_supported_nodes
# 如果存在交集，则抛出错误信息 `_msg`，说明这些节点既支持又不支持
_msg = f"cannot both support and not support {intersection}"
assert not intersection, _msg

def _node_not_implemented(node_name: str) -> Callable[..., None]:
    """
    返回一个函数，当调用时会抛出一个 NotImplementedError 异常，
    该异常说明对于传入的节点名称，相应的功能尚未实现。
    """
    def f(self, *args, **kwargs):
        raise NotImplementedError(f"'{node_name}' nodes are not implemented")
    return f

_T = TypeVar("_T")

def disallow(nodes: set[str]) -> Callable[[type[_T]], type[_T]]:
    """
    装饰器函数，用于禁止解析特定节点，如果访问到这些节点，则会抛出 NotImplementedError 异常。
    
    Returns
    -------
    callable
        返回一个装饰器函数，该函数用于处理传入的类，禁止其访问特定节点。
    """
    def disallowed(cls: type[_T]) -> type[_T]:
        # 错误："Type[_T]" 没有 "unsupported_nodes" 属性
        cls.unsupported_nodes = ()  # type: ignore[attr-defined]
        for node in nodes:
            new_method = _node_not_implemented(node)
            name = f"visit_{node}"
            # 错误："Type[_T]" 没有 "unsupported_nodes" 属性
            cls.unsupported_nodes += (name,)  # type: ignore[attr-defined]
            setattr(cls, name, new_method)
        return cls
    return disallowed

def _op_maker(op_class, op_symbol):
    """
    返回一个函数，用于创建一个 Op 子类，并已经传入了操作符。
    
    Returns
    -------
    callable
        返回一个部分应用了 Op 子类及其操作符的函数。
    """
    def f(self, node, *args, **kwargs):
        """
        返回一个部分应用了 Op 子类及其操作符的函数。
        
        Returns
        -------
        callable
            部分应用了 Op 子类及其操作符的函数。
        """
        return partial(op_class, op_symbol, *args, **kwargs)
    return f

_op_classes = {"binary": BinOp, "unary": UnaryOp}

def add_ops(op_classes):
    """
    装饰器函数，用于为类添加操作的默认实现。
    """
    def f(cls):
        for op_attr_name, op_class in op_classes.items():
            ops = getattr(cls, f"{op_attr_name}_ops")
            ops_map = getattr(cls, f"{op_attr_name}_op_nodes_map")
            for op in ops:
                op_node = ops_map[op]
                if op_node is not None:
                    made_op = _op_maker(op_class, op)
                    setattr(cls, f"visit_{op_node}", made_op)
        return cls
    return f

@disallow(_unsupported_nodes)
@add_ops(_op_classes)
class BaseExprVisitor(ast.NodeVisitor):
    """
    自定义的 AST 遍历器。如果需要，其他引擎的解析器应该继承这个类。
    
    Parameters
    ----------
    env : Scope
        环境变量
    engine : str
        引擎名称
    parser : str
        解析器名称
    """
    # preparser 是一个可调用对象，通常用于预处理字符串节点
    preparser : callable
    """

    # const_type 和 term_type 是类变量，分别指定 Term 类型的常量和术语
    const_type: ClassVar[type[Term]] = Constant
    term_type: ClassVar[type[Term]] = Term

    # 定义支持的二元操作符的符号集合
    binary_ops = CMP_OPS_SYMS + BOOL_OPS_SYMS + ARITH_OPS_SYMS
    # 将二元操作符符号映射到对应的节点名称
    binary_op_nodes = (
        "Gt",
        "Lt",
        "GtE",
        "LtE",
        "Eq",
        "NotEq",
        "In",
        "NotIn",
        "BitAnd",
        "BitOr",
        "And",
        "Or",
        "Add",
        "Sub",
        "Mult",
        None,
        "Pow",
        "FloorDiv",
        "Mod",
    )
    binary_op_nodes_map = dict(zip(binary_ops, binary_op_nodes))

    # 定义支持的一元操作符的符号集合
    unary_ops = UNARY_OPS_SYMS
    # 将一元操作符符号映射到对应的节点名称
    unary_op_nodes = "UAdd", "USub", "Invert", "Not"
    unary_op_nodes_map = dict(zip(unary_ops, unary_op_nodes))

    # 定义 AST 节点的重写映射，用于处理特定节点类型的重写
    rewrite_map = {
        ast.Eq: ast.In,
        ast.NotEq: ast.NotIn,
        ast.In: ast.In,
        ast.NotIn: ast.NotIn,
    }

    # 不支持的节点类型的元组
    unsupported_nodes: tuple[str, ...]

    def __init__(self, env, engine, parser, preparser=_preparse) -> None:
        # 初始化方法，接受环境变量、引擎、解析器和可选的预处理器函数
        self.env = env
        self.engine = engine
        self.parser = parser
        self.preparser = preparser
        self.assigner = None

    def visit(self, node, **kwargs):
        # 根据节点类型分派到相应的访问方法进行处理
        if isinstance(node, str):
            # 如果节点是字符串，先进行预处理
            clean = self.preparser(node)
            try:
                # 尝试解析清理后的字符串生成 AST
                node = ast.fix_missing_locations(ast.parse(clean))
            except SyntaxError as e:
                # 处理语法错误，检查是否包含 Python 关键字
                if any(iskeyword(x) for x in clean.split()):
                    e.msg = "Python keyword not valid identifier in numexpr query"
                raise e

        # 构造访问方法名，并获取相应的访问器
        method = f"visit_{type(node).__name__}"
        visitor = getattr(self, method)
        return visitor(node, **kwargs)

    def visit_Module(self, node, **kwargs):
        # 访问 Module 节点，确保只有一个表达式
        if len(node.body) != 1:
            raise SyntaxError("only a single expression is allowed")
        expr = node.body[0]
        return self.visit(expr, **kwargs)

    def visit_Expr(self, node, **kwargs):
        # 访问 Expr 节点，递归处理其值节点
        return self.visit(node.value, **kwargs)
    # 重写成员运算符的方法，用于处理节点、左操作数和右操作数
    def _rewrite_membership_op(self, node, left, right):
        # 获取操作符的实例
        op_instance = node.op
        # 获取操作符的类型
        op_type = type(op_instance)

        # 确保左右操作数均为项（terms），并且操作符类型在重写映射表中
        if is_term(left) and is_term(right) and op_type in self.rewrite_map:
            # 检查左右操作数是否为列表
            left_list, right_list = map(_is_list, (left, right))
            # 检查左右操作数是否为字符串
            left_str, right_str = map(_is_str, (left, right))

            # 如果表达式中存在字符串或列表
            if left_list or right_list or left_str or right_str:
                # 使用重写映射表中对应的操作符类型实例化新的操作符对象
                op_instance = self.rewrite_map[op_type]()

            # 如果右操作数是字符串，则将其从局部变量中移除，并替换为包含一个字符串的列表（一种“hack”方式）
            if right_str:
                name = self.env.add_tmp([right.value])
                right = self.term_type(name, self.env)

            # 如果左操作数是字符串，则将其从局部变量中移除，并替换为包含一个字符串的列表（一种“hack”方式）
            if left_str:
                name = self.env.add_tmp([left.value])
                left = self.term_type(name, self.env)

        # 访问并返回操作符对象
        op = self.visit(op_instance)
        return op, op_instance, left, right

    # 可能转换相等和不相等操作的方法
    def _maybe_transform_eq_ne(self, node, left=None, right=None):
        # 如果未提供左操作数，则访问节点的左子树并进行访问
        if left is None:
            left = self.visit(node.left, side="left")
        # 如果未提供右操作数，则访问节点的右子树并进行访问
        if right is None:
            right = self.visit(node.right, side="right")
        # 使用重写成员运算符方法处理节点、左操作数和右操作数
        op, op_class, left, right = self._rewrite_membership_op(node, left, right)
        # 返回处理后的操作符、操作符类型、左操作数和右操作数
        return op, op_class, left, right

    # 可能降低常数值的类型精度的方法
    def _maybe_downcast_constants(self, left, right):
        # 定义32位浮点数类型
        f32 = np.dtype(np.float32)
        # 如果左操作数是标量且具有"value"属性，且右操作数不是标量且返回类型是32位浮点数
        if (
            left.is_scalar
            and hasattr(left, "value")
            and not right.is_scalar
            and right.return_type == f32
        ):
            # 右操作数是一个float32数组，左操作数是一个标量，创建一个临时变量名，并使用np.float32(left.value)添加到环境中
            name = self.env.add_tmp(np.float32(left.value))
            left = self.term_type(name, self.env)
        # 如果右操作数是标量且具有"value"属性，且左操作数不是标量且返回类型是32位浮点数
        if (
            right.is_scalar
            and hasattr(right, "value")
            and not left.is_scalar
            and left.return_type == f32
        ):
            # 左操作数是一个float32数组，右操作数是一个标量，创建一个临时变量名，并使用np.float32(right.value)添加到环境中
            name = self.env.add_tmp(np.float32(right.value))
            right = self.term_type(name, self.env)

        # 返回处理后的左操作数和右操作数
        return left, right

    # 可能进行求值的方法
    def _maybe_eval(self, binop, eval_in_python):
        # 对于"in"和"not in"操作，在“部分”Python空间中进行求值
        # 可以在“eval”空间中求值的内容将转换为临时变量
        # 例如：[1,2] in a + 2 * b，在这种情况下，a + 2 * b将使用numexpr进行求值，并且"in"调用将使用isin在Python空间中进行求值
        return binop.evaluate(
            self.env, self.engine, self.parser, self.term_type, eval_in_python
        )
    # 定义一个方法用于可能评估二元操作符的结果
    def _maybe_evaluate_binop(
        self,
        op,
        op_class,
        lhs,
        rhs,
        eval_in_python=("in", "not in"),
        maybe_eval_in_python=("==", "!=", "<", ">", "<=", ">="),
    ):
        # 调用二元操作符计算结果
        res = op(lhs, rhs)

        # 如果结果返回类型无效，抛出类型错误异常
        if res.has_invalid_return_type:
            raise TypeError(
                f"unsupported operand type(s) for {res.op}: "
                f"'{lhs.type}' and '{rhs.type}'"
            )

        # 如果引擎不是 'pytables' 并且涉及到日期比较操作符，必须在 Python 中执行
        if self.engine != "pytables" and (
            res.op in CMP_OPS_SYMS
            and getattr(lhs, "is_datetime", False)
            or getattr(rhs, "is_datetime", False)
        ):
            # 所有日期操作必须在 Python 中执行，因为 numexpr 在 NaT 处理上存在问题
            return self._maybe_eval(res, self.binary_ops)

        # 如果结果操作符在 eval_in_python 中，则始终在 Python 中评估 "in"/"not in" 操作符
        if res.op in eval_in_python:
            return self._maybe_eval(res, eval_in_python)
        # 否则，如果不是 'pytables' 引擎，并且任一操作数的返回类型是 object，则评估 "==" 和 "!=" 操作符
        elif self.engine != "pytables":
            if (
                getattr(lhs, "return_type", None) == object
                or getattr(rhs, "return_type", None) == object
            ):
                return self._maybe_eval(res, eval_in_python + maybe_eval_in_python)
        # 返回计算结果
        return res

    # 访问二元操作节点
    def visit_BinOp(self, node, **kwargs):
        # 可能转换等于和不等于操作符
        op, op_class, left, right = self._maybe_transform_eq_ne(node)
        # 可能将常量向下转型
        left, right = self._maybe_downcast_constants(left, right)
        # 返回可能评估的二元操作结果
        return self._maybe_evaluate_binop(op, op_class, left, right)

    # 访问除法节点
    def visit_Div(self, node, **kwargs):
        # 返回一个 lambda 函数，用于表示除法操作
        return lambda lhs, rhs: Div(lhs, rhs)

    # 访问一元操作节点
    def visit_UnaryOp(self, node, **kwargs):
        # 访问操作符和操作数节点，并执行操作符
        op = self.visit(node.op)
        operand = self.visit(node.operand)
        return op(operand)

    # 访问名称节点，返回一个 Term 类型
    def visit_Name(self, node, **kwargs) -> Term:
        return self.term_type(node.id, self.env, **kwargs)

    # TODO(py314): 自 Python 3.8 弃用。在 Python 3.14 成为最低版本后移除
    # 访问 NameConstant 节点，返回一个 Term 类型
    def visit_NameConstant(self, node, **kwargs) -> Term:
        return self.const_type(node.value, self.env)

    # TODO(py314): 自 Python 3.8 弃用。在 Python 3.14 成为最低版本后移除
    # 访问 Num 节点，返回一个 Term 类型
    def visit_Num(self, node, **kwargs) -> Term:
        return self.const_type(node.value, self.env)

    # 访问常量节点，返回一个 Term 类型
    def visit_Constant(self, node, **kwargs) -> Term:
        return self.const_type(node.value, self.env)

    # TODO(py314): 自 Python 3.8 弃用。在 Python 3.14 成为最低版本后移除
    # 访问字符串节点，返回一个 Term 类型
    def visit_Str(self, node, **kwargs) -> Term:
        # 将字符串添加到环境中，并返回一个 Term 类型
        name = self.env.add_tmp(node.s)
        return self.term_type(name, self.env)

    # 访问列表节点，返回一个 Term 类型
    def visit_List(self, node, **kwargs) -> Term:
        # 将列表中每个元素访问后的结果放入临时变量，并返回一个 Term 类型
        name = self.env.add_tmp([self.visit(e)(self.env) for e in node.elts])
        return self.term_type(name, self.env)

    # 访问元组节点，与访问列表节点相同
    visit_Tuple = visit_List

    # 访问索引节点，返回其值
    def visit_Index(self, node, **kwargs):
        """df.index[4]"""
        return self.visit(node.value)
    # 访问 Subscript 节点的方法，返回一个 Term 实例
    def visit_Subscript(self, node, **kwargs) -> Term:
        # 从 pandas 库中导入 eval 函数，并重命名为 pd_eval
        from pandas import eval as pd_eval
        
        # 访问 Subscript 节点的值部分，并获取其值
        value = self.visit(node.value)
        # 访问 Subscript 节点的切片部分，并获取其值
        slobj = self.visit(node.slice)
        # 使用 pd_eval 函数对 slobj 进行求值，使用给定的环境、引擎和解析器
        result = pd_eval(
            slobj, local_dict=self.env, engine=self.engine, parser=self.parser
        )
        try:
            # 尝试使用 result 作为索引从 value.value 中获取值，假设 value 是 Term 实例
            v = value.value[result]
        except AttributeError:
            # 如果发生 AttributeError，则假设 value 是 Op 实例，再次使用 pd_eval 函数求值
            lhs = pd_eval(
                value, local_dict=self.env, engine=self.engine, parser=self.parser
            )
            # 使用 result 作为索引从 lhs 中获取值
            v = lhs[result]
        # 将 v 添加到环境中，并用生成的名字创建一个新的 Term 实例，返回该实例
        name = self.env.add_tmp(v)
        return self.term_type(name, env=self.env)

    # 访问 Slice 节点的方法，返回一个 slice 对象
    def visit_Slice(self, node, **kwargs) -> slice:
        """df.index[slice(4,6)]"""
        # 获取 slice 节点的下限部分，并将其访问结果转换为其值
        lower = node.lower
        if lower is not None:
            lower = self.visit(lower).value
        # 获取 slice 节点的上限部分，并将其访问结果转换为其值
        upper = node.upper
        if upper is not None:
            upper = self.visit(upper).value
        # 获取 slice 节点的步长部分，并将其访问结果转换为其值
        step = node.step
        if step is not None:
            step = self.visit(step).value

        # 返回根据访问结果构造的 slice 对象
        return slice(lower, upper, step)

    # 访问 Assign 节点的方法，支持单个赋值表达式
    def visit_Assign(self, node, **kwargs):
        """
        support a single assignment node, like

        c = a + b

        set the assigner at the top level, must be a Name node which
        might or might not exist in the resolvers

        """
        # 检查赋值目标数量是否为1，若不是则引发 SyntaxError
        if len(node.targets) != 1:
            raise SyntaxError("can only assign a single expression")
        # 检查赋值目标是否为 ast.Name 类型，若不是则引发 SyntaxError
        if not isinstance(node.targets[0], ast.Name):
            raise SyntaxError("left hand side of an assignment must be a single name")
        # 检查当前环境中是否存在目标对象，若不存在则引发 ValueError
        if self.env.target is None:
            raise ValueError("cannot assign without a target object")

        try:
            # 访问赋值目标节点，获取其值
            assigner = self.visit(node.targets[0], **kwargs)
        except UndefinedVariableError:
            # 若出现 UndefinedVariableError，则将赋值目标节点的 id 作为 assigner
            assigner = node.targets[0].id

        # 将 assigner 的 name 属性作为赋值器，若为 None 则引发 SyntaxError
        self.assigner = getattr(assigner, "name", assigner)
        if self.assigner is None:
            raise SyntaxError(
                "left hand side of an assignment must be a single resolvable name"
            )

        # 访问赋值节点的值部分，并返回其访问结果
        return self.visit(node.value, **kwargs)

    # 访问 Attribute 节点的方法，处理属性访问操作
    def visit_Attribute(self, node, **kwargs):
        # 获取属性的名称
        attr = node.attr
        # 获取属性所属的值节点
        value = node.value

        # 获取节点的上下文类型，若为 ast.Load 类型则继续解析
        ctx = node.ctx
        if isinstance(ctx, ast.Load):
            # 解析属性所属值节点的值
            resolved = self.visit(value).value
            try:
                # 尝试获取 resolved 对象的属性 attr
                v = getattr(resolved, attr)
                # 将获取的属性值添加到环境中，并创建一个新的 Term 实例返回
                name = self.env.add_tmp(v)
                return self.term_type(name, self.env)
            except AttributeError:
                # 若发生 AttributeError，则可能是类似 datetime.datetime 的情况，抛出异常
                if isinstance(value, ast.Name) and value.id == attr:
                    return resolved
                raise

        # 若上下文类型不是 ast.Load，则抛出 ValueError 异常
        raise ValueError(f"Invalid Attribute context {type(ctx).__name__}")
    # 处理函数调用节点的访问，支持属性调用和命名函数
    def visit_Call(self, node, side=None, **kwargs):
        # 如果函数节点是属性访问且不是特殊方法 "__call__"
        if isinstance(node.func, ast.Attribute) and node.func.attr != "__call__":
            # 访问属性节点
            res = self.visit_Attribute(node.func)
        # 如果函数节点不是命名函数
        elif not isinstance(node.func, ast.Name):
            # 抛出类型错误，仅支持命名函数
            raise TypeError("Only named functions are supported")
        else:
            try:
                # 访问函数节点
                res = self.visit(node.func)
            except UndefinedVariableError:
                # 检查是否是支持的函数名
                try:
                    # 尝试创建 FuncNode 对象
                    res = FuncNode(node.func.id)
                except ValueError:
                    # 抛出原始错误
                    raise

        # 如果未能解析出有效结果
        if res is None:
            # 抛出值错误，表示函数调用无效
            raise ValueError(
                f"Invalid function call {node.func.id}"  # type: ignore[attr-defined]
            )
        
        # 如果 res 具有"value"属性，获取其值
        if hasattr(res, "value"):
            res = res.value

        # 如果 res 是 FuncNode 类型
        if isinstance(res, FuncNode):
            # 处理函数参数
            new_args = [self.visit(arg) for arg in node.args]

            # 如果有关键字参数，抛出类型错误，该函数不支持关键字参数
            if node.keywords:
                raise TypeError(
                    f'Function "{res.name}" does not support keyword arguments'
                )

            # 调用函数并返回结果
            return res(*new_args)

        else:
            # 处理函数参数，并将环境传递给参数
            new_args = [self.visit(arg)(self.env) for arg in node.args]

            # 遍历关键字参数
            for key in node.keywords:
                if not isinstance(key, ast.keyword):
                    # 抛出值错误，表示关键字参数错误
                    raise ValueError(
                        "keyword error in function call " f"'{node.func.id}'"  # type: ignore[attr-defined]
                    )

                # 如果关键字有参数名，将其值传递到环境中
                if key.arg:
                    kwargs[key.arg] = self.visit(key.value)(self.env)

            # 调用函数并将结果添加到临时变量中，返回结果
            name = self.env.add_tmp(res(*new_args, **kwargs))
            return self.term_type(name=name, env=self.env)

    # 返回操作符 op
    def translate_In(self, op):
        return op

    # 访问比较节点，处理多种比较操作
    def visit_Compare(self, node, **kwargs):
        ops = node.ops
        comps = node.comparators

        # 基本情况：只有一个比较操作，如 a CMP b
        if len(comps) == 1:
            op = self.translate_In(ops[0])
            # 创建二元操作节点，并继续访问
            binop = ast.BinOp(op=op, left=node.left, right=comps[0])
            return self.visit(binop)

        # 递归情况：多个比较操作，如 a CMP b CMP c 等
        left = node.left
        values = []
        for op, comp in zip(ops, comps):
            # 递归访问新的比较节点
            new_node = self.visit(
                ast.Compare(comparators=[comp], left=left, ops=[self.translate_In(op)])
            )
            left = comp
            values.append(new_node)
        # 创建布尔操作节点，并继续访问
        return self.visit(ast.BoolOp(op=ast.And(), values=values))

    # 尝试访问二元操作，如果是 Op 或 Term 类型则直接返回，否则继续访问
    def _try_visit_binop(self, bop):
        if isinstance(bop, (Op, Term)):
            return bop
        return self.visit(bop)
    # 定义一个方法用于访问布尔操作（BoolOp）节点，处理节点和关键字参数
    def visit_BoolOp(self, node, **kwargs):
        # 定义内部函数visitor，处理两个操作数
        def visitor(x, y):
            # 尝试访问并处理二元操作符的左操作数和右操作数
            lhs = self._try_visit_binop(x)
            rhs = self._try_visit_binop(y)

            # 可能对节点进行等号和不等号变换，获取操作符、操作符类别、左操作数和右操作数
            op, op_class, lhs, rhs = self._maybe_transform_eq_ne(node, lhs, rhs)
            
            # 可能对二元操作符进行求值，并返回结果
            return self._maybe_evaluate_binop(op, node.op, lhs, rhs)

        # 获取BoolOp节点的操作数列表
        operands = node.values
        # 使用reduce函数依次对操作数进行visitor函数的处理，将结果返回
        return reduce(visitor, operands)
# 不支持的节点类型和Python中不支持的表达式类型的集合
_python_not_supported = frozenset(["Dict", "BoolOp", "In", "NotIn"])
# 支持的调用集合，包括在 REDUCTIONS 和 MATHOPS 中的元素
_numexpr_supported_calls = frozenset(REDUCTIONS + MATHOPS)


@disallow(
    # 使用的是不支持的节点类型集合和Python中不支持的表达式类型集合的并集，
    # 从中排除布尔操作、属性、成员关系运算符、元组等支持的节点类型
    (_unsupported_nodes | _python_not_supported)
    - (_boolop_nodes | frozenset(["BoolOp", "Attribute", "In", "NotIn", "Tuple"]))
)
class PandasExprVisitor(BaseExprVisitor):
    def __init__(
        self,
        env,
        engine,
        parser,
        preparser=partial(
            # 对源码进行预处理，包括替换本地变量、布尔值、清理反引号引用的令牌
            _preparse,
            f=_compose(_replace_locals, _replace_booleans, clean_backtick_quoted_toks),
        ),
    ) -> None:
        super().__init__(env, engine, parser, preparser)


@disallow(
    # 不支持的节点类型集合、Python中不支持的表达式类型集合以及 "Not" 的并集
    _unsupported_nodes | _python_not_supported | frozenset(["Not"])
)
class PythonExprVisitor(BaseExprVisitor):
    def __init__(
        self, env, engine, parser, preparser=lambda source, f=None: source
    ) -> None:
        super().__init__(env, engine, parser, preparser=preparser)


class Expr:
    """
    表达式对象的封装。

    Parameters
    ----------
    expr : str
        表达式字符串
    engine : str, optional, default 'numexpr'
        引擎名称，默认为 'numexpr'
    parser : str, optional, default 'pandas'
        解析器名称，默认为 'pandas'
    env : Scope, optional, default None
        作用域对象，默认为 None
    level : int, optional, default 2
        层级深度，默认为 2
    """

    env: Scope  # 作用域对象
    engine: str  # 引擎名称
    parser: str  # 解析器名称

    def __init__(
        self,
        expr,
        engine: str = "numexpr",
        parser: str = "pandas",
        env: Scope | None = None,
        level: int = 0,
    ) -> None:
        self.expr = expr
        # 如果未提供作用域对象，则创建一个新的作用域对象
        self.env = env or Scope(level=level + 1)
        self.engine = engine
        self.parser = parser
        # 根据指定的解析器类型创建访问者对象
        self._visitor = PARSERS[parser](self.env, self.engine, self.parser)
        # 解析表达式并保存其结果
        self.terms = self.parse()

    @property
    def assigner(self):
        # 获取访问者对象中的 assigner 属性，如果不存在则返回 None
        return getattr(self._visitor, "assigner", None)

    def __call__(self):
        # 调用表达式的结果并返回
        return self.terms(self.env)

    def __repr__(self) -> str:
        # 返回表达式结果的格式化字符串表示
        return printing.pprint_thing(self.terms)

    def __len__(self) -> int:
        # 返回表达式字符串的长度
        return len(self.expr)

    def parse(self):
        """
        解析表达式。
        """
        return self._visitor.visit(self.expr)

    @property
    def names(self):
        """
        获取表达式中的名称集合。
        """
        if is_term(self.terms):
            return frozenset([self.terms.name])
        return frozenset(term.name for term in com.flatten(self.terms))


# 解析器类型到相应访问者类的映射
PARSERS = {"python": PythonExprVisitor, "pandas": PandasExprVisitor}
```