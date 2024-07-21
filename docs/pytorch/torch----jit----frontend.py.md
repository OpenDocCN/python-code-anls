# `.\pytorch\torch\jit\frontend.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和类
import ast  # AST（抽象语法树）模块，用于处理Python代码的抽象语法树
import dataclasses  # 用于创建和操作数据类的模块
import inspect  # 用于检查源码和对象的模块
import re  # 正则表达式模块，用于处理字符串匹配和操作
import string  # 字符串处理模块，包含各种字符串相关的工具函数和常量
import sys  # 系统相关的参数和函数
from collections import namedtuple  # 命名元组，创建带字段名的元组子类的工厂函数
from textwrap import dedent  # 文本缩进模块，用于去除文本的公共缩进
from typing import List, Tuple  # 类型提示模块，用于类型注解

import torch  # PyTorch深度学习库
import torch.jit.annotations  # PyTorch中用于jit编译的注解模块
from torch import _jit_internal  # PyTorch中jit内部使用的模块
from torch._C._jit_tree_views import (  # PyTorch中jit树形视图的相关类和函数
    Apply, Assert, Assign, Attribute, AugAssign, BinOp, Break, ClassDef, Const, Continue,
    Decl, Def, Delete, DictComp, DictLiteral, Dots, EmptyTypeAnnotation, ExprStmt,
    FalseLiteral, For, Ident, If, ListComp, ListLiteral, NoneLiteral, Param, Pass,
    Property, Raise, Return, Select, SliceExpr, Starred, Stmt, StringLiteral, Subscript,
    TernaryIf, TrueLiteral, TupleLiteral, UnaryOp, Var, While, With, WithItem,
)
from torch._jit_internal import (  # PyTorch中jit内部使用的模块（续）
    _is_drop_fn, FunctionModifiers, is_static_fn, should_drop,
)
from torch._sources import (  # PyTorch源码相关的一些函数
    get_source_lines_and_file, make_source_context, parse_def, ParsedDef as _ParsedDef,
)
from torch.jit._dataclass_impls import DATACLASS_MAGIC_METHODS  # PyTorch中数据类的魔术方法
from torch.jit._monkeytype_config import get_qualified_name, monkeytype_trace  # PyTorch中monkeytype相关的配置函数

_IS_ASTUNPARSE_INSTALLED = False
try:
    import astunparse  # type: ignore[import]
    _IS_ASTUNPARSE_INSTALLED = True
except ImportError:
    pass

# Borrowed from cPython implementation
# https://github.com/python/cpython/blob/561612d8456cfab5672c9b445521113b847bd6b3/Lib/textwrap.py#L411#

# 以下是一些保留前缀和保留名称
_reserved_prefix = "__jit"
_reserved_names = {"print"}  # 保留的函数名集合，包含"print"

# 字母和数字组成的字符集合
_identifier_chars = set(string.ascii_lowercase + string.ascii_uppercase + string.digits)


def is_reserved_name(name):
    # 判断给定名称是否为保留名称
    return name.startswith(_reserved_prefix) or name in _reserved_names


# 不同AST节点类型对应的可读名称映射
pretty_node_names = {
    ast.FunctionDef: "function definitions",  # 函数定义
    ast.For: "for loops",  # for循环
    ast.Delete: "del statements",  # del语句
    ast.ClassDef: "class definitions",  # 类定义
    ast.With: "with statements",  # with语句
    ast.Raise: "raise statements",  # raise语句
    ast.Assert: "assertions",  # 断言语句
    ast.Import: "import statements",  # import语句
    ast.ImportFrom: "import statements",  # from...import语句
    ast.Global: "global variables",  # 全局变量声明
    ast.Break: "break statements",  # break语句
    ast.Continue: "continue statements",  # continue语句
}

# 不同AST节点类型对应的起始关键词映射
node_start_tokens = {
    ast.FunctionDef: "def",  # 函数定义起始关键词
    ast.For: "for",  # for循环起始关键词
    ast.Delete: "del",  # del语句起始关键词
    ast.ClassDef: "class",  # 类定义起始关键词
    ast.With: "with",  # with语句起始关键词
    ast.Raise: "raise",  # raise语句起始关键词
    ast.Assert: "assert",  # 断言语句起始关键词
    ast.Import: "import",  # import语句起始关键词
    ast.ImportFrom: "from",  # from...import语句起始关键词
    ast.Global: "global",  # 全局变量声明起始关键词
    ast.Break: "break",  # break语句起始关键词
    ast.Continue: "continue",  # continue语句起始关键词
}

# 更新异步相关的节点名称映射
pretty_node_names.update(
    {
        ast.AsyncFunctionDef: "async function definitions",  # 异步函数定义
        ast.AsyncFor: "async for loops",  # 异步for循环
        ast.AsyncWith: "async with statements",  # 异步with语句
        ast.Try: "try blocks",  # try语句块
        ast.Nonlocal: "nonlocal variables",  # nonlocal变量声明
    }
)

# 更新异步相关的节点起始关键词映射
node_start_tokens.update(
    {
        ast.AsyncFunctionDef: "async def",  # 异步函数定义起始关键词
        ast.AsyncFor: "async for",  # 异步for循环起始关键词
        ast.AsyncWith: "async with",  # 异步with语句起始关键词
        ast.Try: "try",  # try语句块起始关键词
        ast.Nonlocal: "nonlocal",  # nonlocal变量声明起始关键词
    }
)
    {
        # 映射 AST (抽象语法树) 中的 AsyncFunctionDef 节点到字符串 "async def"
        ast.AsyncFunctionDef: "async def",
        # 映射 AST 中的 AsyncFor 节点到字符串 "async for"
        ast.AsyncFor: "async for",
        # 映射 AST 中的 AsyncWith 节点到字符串 "async with"
        ast.AsyncWith: "async with",
        # 映射 AST 中的 Try 节点到字符串 "try"
        ast.Try: "try",
        # 映射 AST 中的 Nonlocal 节点到字符串 "nonlocal"
        ast.Nonlocal: "nonlocal",
    }
)

# 更新 pretty_node_names 字典，添加一个新的键值对
pretty_node_names.update(
    {
        ast.AnnAssign: "annotated assignments",
    }
)
# 注意：AnnAssign 没有特定的标记

# 定义一个自定义异常类 FrontendError，继承自内置异常类 Exception
class FrontendError(Exception):
    def __init__(self, source_range, msg):
        self.source_range = source_range
        self.msg = msg

        # 创建一个 torch._C.ErrorReport 实例，用于生成详细的错误报告
        # 以保证 FrontendError 抛出时的调用栈信息准确
        self.error_report = torch._C.ErrorReport(self.source_range)

    def __str__(self):
        # 返回异常的具体信息以及相关错误报告的内容
        return self.msg + self.error_report.what().lstrip()

# 定义一个特定类型的 FrontendError 子类 NotSupportedError
class NotSupportedError(FrontendError):
    pass

# 定义另一个特定类型的 FrontendError 子类 UnsupportedNodeError
class UnsupportedNodeError(NotSupportedError):
    def __init__(self, ctx, offending_node, reason=""):
        # 获取 offending_node 的类型信息
        node_type = type(offending_node)
        # 计算节点在源码中的范围长度
        range_len = len(node_start_tokens.get(node_type, " "))
        # 使用 ctx.make_range 创建节点在源码中的范围对象 source_range
        source_range = ctx.make_range(
            offending_node.lineno,
            offending_node.col_offset,
            offending_node.col_offset + range_len,
        )
        # 获取节点类型对应的名称或者特征名
        feature_name = pretty_node_names.get(node_type, node_type.__name__)
        # 组装错误信息
        msg = f"{feature_name} {reason + ' ' if reason else ''}aren't supported"
        # 调用父类构造函数初始化异常
        super().__init__(source_range, msg)

# 定义一个特定类型的 FrontendError 子类 FrontendTypeError
class FrontendTypeError(FrontendError):
    pass

# 定义函数 build_withitems，接收 ctx 和 items 作为参数
def build_withitems(ctx, items):
    # 对 items 列表中的每个元素调用 build_withitem 函数进行处理，得到处理后的列表 items
    items = [build_withitem(ctx, i) for i in items]
    # 返回处理后的列表 items
    return list(items)

# 定义函数 build_stmts，接收 ctx 和 stmts 作为参数
def build_stmts(ctx, stmts):
    # 对 stmts 列表中的每个元素调用 build_stmt 函数进行处理，得到处理后的列表 stmts
    stmts = [build_stmt(ctx, s) for s in stmts]
    # 过滤掉列表中的 None 元素，返回处理后的列表 stmts
    return list(filter(None, stmts))

# 定义函数 get_class_properties，接收 cls 和 self_name 作为参数
def get_class_properties(cls, self_name):
    """
    获取表示类属性的 Property 对象列表。

    Args:
        cls: 要获取属性的类。
        self_name: 属性应该属于的类的名称。

    Returns:
        包含 cls 属性的 Property 对象列表。这里的 Property 指的是 TreeView 的子类。
    """
    # 使用 inspect.getmembers 获取 cls 中所有的 property 属性成员
    props = inspect.getmembers(cls, predicate=lambda m: isinstance(m, property))
    # 获取属性不应编译的列表（如果有）
    unused_properties = getattr(cls, "__jit_unused_properties__", [])

    # 创建 Property 对象列表，从检查到的 property 对象中生成
    properties = []
    for prop in props:
        # 如果属性不在未使用属性列表中，并且不应该丢弃，则处理该属性
        if prop[0] not in unused_properties and not should_drop(prop[1].fget):
            # 获取属性的 getter 方法的 JIT 定义
            getter = get_jit_def(
                prop[1].fget, f"__{prop[0]}_getter", self_name=self_name
            )
            # 获取属性的 setter 方法的 JIT 定义，如果有的话
            setter = (
                get_jit_def(prop[1].fset, f"__{prop[0]}_setter", self_name=self_name)
                if prop[1].fset
                else None
            )
            # 创建 Property 对象并添加到 properties 列表中
            properties.append(
                Property(getter.range(), Ident(getter.range(), prop[0]), getter, setter)
            )

    return properties

# 定义函数 get_class_assigns，接收 ctx 和 cls_ast 作为参数
def get_class_assigns(ctx, cls_ast):
    # 初始化 assigns 列表为空
    assigns = []
    # 定义一个内部函数 maybe_build_assign，用于尝试构建赋值语句的 AST 节点
    def maybe_build_assign(builder, entry):
        nonlocal assigns  # 声明 assigns 变量为非局部变量，用于存储生成的赋值语句
        try:
            assigns.append(builder(ctx, entry))  # 调用传入的 builder 函数构建赋值语句，并添加到 assigns 列表中
        except NotSupportedError:
            pass  # 如果 builder 抛出 NotSupportedError 异常，则忽略该异常

    # 遍历类的 AST 主体中的每个条目（entry）
    for entry in cls_ast.body:
        # 如果当前条目是赋值语句 (ast.Assign 类型)，则调用 maybe_build_assign 处理
        if isinstance(entry, ast.Assign):
            maybe_build_assign(StmtBuilder.build_Assign, entry)
        # 如果当前条目是带注解的赋值语句 (ast.AnnAssign 类型)，同样调用 maybe_build_assign 处理
        elif isinstance(entry, ast.AnnAssign):
            maybe_build_assign(StmtBuilder.build_AnnAssign, entry)
    
    # 返回生成的赋值语句列表 assigns
    return assigns
def get_jit_class_def(cls, self_name):
    """获取当前类中每个方法的定义信息。

    Args:
        cls: 要获取定义的类。
        self_name: 属性所属的类的名称。

    Returns:
        torch._C._jit_tree_views.ClassDef: 表示类的对象，
            类中方法及其定义的树形表示。
    """
    # TODO: proper overriding analysis when implementing class inheritance
    # 使用 inspect.getmembers 获取类中的方法和函数，满足以下条件：
    # - 不是静态方法
    # - 方法名存在于类的字典中
    # - 不是特定标记为需要忽略的方法
    methods = inspect.getmembers(
        cls,
        predicate=lambda m: (inspect.ismethod(m) or inspect.isfunction(m))
        and not is_static_fn(cls, m.__name__)
        and m.__name__ in cls.__dict__
        and not _is_drop_fn(m),
    )

    def is_classmethod(fn):
        return inspect.ismethod(fn) and getattr(fn, "__self__", None) == cls

    # 获取并解析当前类的源代码
    sourcelines, file_lineno, filename = get_source_lines_and_file(
        cls, torch._C.ErrorReport.call_stack()
    )
    source = "".join(sourcelines)

    dedent_src = dedent(source)
    py_ast = ast.parse(dedent_src)

    class_ast = py_ast.body[0]
    assert isinstance(class_ast, ast.ClassDef)

    # 处理数据类的特殊情况。数据类使用 dataclasses 模块动态合成魔术方法，需特殊处理。
    if dataclasses.is_dataclass(cls):
        # 检测用户是否手动实现了任何魔术方法，如果实现了，则不需要合成/覆盖它们。
        overrides = {
            method.name
            for method in class_ast.body
            if isinstance(method, ast.FunctionDef)
            and method.name in DATACLASS_MAGIC_METHODS
        }
        for i, (name, _) in enumerate(methods):
            # 检查是否可以合成魔术方法
            synthesizer_fn = DATACLASS_MAGIC_METHODS.get(name)
            if synthesizer_fn and name not in overrides:
                parsed_def = synthesizer_fn(cls)
                methods[i] = name, parsed_def
                func = getattr(cls, name)
                _jit_internal.loader.cache(func, parsed_def.source)

    # 获取方法的 JIT 定义
    method_defs = [
        get_jit_def(obj, name, self_name=self_name, is_classmethod=is_classmethod(obj))
        for (name, obj) in methods
    ]
    # 获取类的属性
    properties = get_class_properties(cls, self_name)

    # 计算源代码的首行缩进长度
    leading_whitespace_len = len(source.split("\n", 1)[0]) - len(
        dedent_src.split("\n", 1)[0]
    )
    # 构建源代码上下文
    ctx = make_source_context(
        source, filename, file_lineno, leading_whitespace_len, False
    )
    # 获取类的赋值语句
    assigns = get_class_assigns(ctx, class_ast)

    # 构建类定义对象
    return build_class_def(ctx, class_ast, method_defs, properties, self_name, assigns)
    """
    Build a JIT AST (TreeView) from the given function.
    
    Args:
        fn: A function object to compile or a pre-parsed ParsedDef object
        def_name: The name to give to the resulting AST object. This is not
            always the same as `fn.__name__`, for example:
                def _forward(self):
                    ...
                forward = _forward
            In this case, the `__name__` attribute of the function object is "_forward",
            but we want the result AST to have the name "forward".
        self_name: If this function is a method, what the type name of `self` is.
    """
    # 解析函数定义，如果输入的是 _ParsedDef 对象则直接使用，否则解析成 _ParsedDef 对象
    parsed_def = parse_def(fn) if not isinstance(fn, _ParsedDef) else fn
    # 获取类型行信息，使用 torch.jit.annotations.get_type_line 函数
    type_line = torch.jit.annotations.get_type_line(parsed_def.source)
    # 获取函数定义体
    fn_def = parsed_def.ast.body[0]
    
    # 如果函数是类方法，需要在函数体内部插入一个语句，将第一个参数赋值给类
    if is_classmethod:
        arg_name = fn_def.args.args[0].arg
        # 构建赋值语句节点
        assign_stmt = ast.parse(f"{arg_name} = {self_name}").body[0]
        # 将赋值语句插入到函数定义体的开头
        fn_def.body.insert(0, assign_stmt)
    
    # 如果应该丢弃函数，则替换函数签名和函数体
    if should_drop(fn):
        # 构建一个未使用的函数定义节点
        unused_fn_def = ast.parse(
            'def unused_fn(self: Any):\n\traise RuntimeError("Cannot call @unused methods")'
        )
        # 检查节点结构是否正确
        if len(unused_fn_def.body) != 1 or not isinstance(
            unused_fn_def.body[0], ast.FunctionDef
        ):
            raise RuntimeError(
                f"Expected a single top-level function: {parsed_def.filename}:{parsed_def.file_lineno}"
            )
        unused_def = unused_fn_def.body[0]
        # 将函数定义体替换为未使用函数定义体的主体
        fn_def.body = unused_def.body
        # 清除可能存在的可变参数和关键字参数
        fn_def.args.kwarg = fn_def.args.vararg = None
        # 替换参数的类型注释为 "Any"
        for arg in fn_def.args.args + fn_def.args.kwonlyargs:
            arg.annotation = unused_def.args.args[0].annotation
        if _is_drop_fn(fn):
            # 如果是丢弃函数，则清除返回类型注释和类型注释的评论
            fn_def.returns = None
            fn_def.type_comment = None
    
    # 如果安装了 MonkeyType，从 type_trace_db 中获取所有参数的类型追踪
    type_trace_db = torch.jit._script._get_type_trace_db()
    pdt_arg_types = None
    if monkeytype_trace and not isinstance(fn, _ParsedDef):  # type: ignore[truthy-function]
        # 获取函数的完全限定名
        qualname = get_qualified_name(fn)
        # 从 type_trace_db 中获取参数类型
        pdt_arg_types = type_trace_db.get_args_types(qualname)
    
    # 调用 build_def 函数构建函数定义
    return build_def(
        parsed_def.ctx,
        fn_def,
        type_line,
        def_name,
        self_name=self_name,
        pdt_arg_types=pdt_arg_types,
    )
# TODO: more robust handling of recognizing ignore context manager
def is_torch_jit_ignore_context_manager(stmt):
    # 检查语句是否是 torch.jit.ignore 上下文管理器
    if isinstance(stmt.items[0].context_expr, ast.Call):
        # 提取 torch 部分
        function = stmt.items[0].context_expr.func
        if isinstance(function, ast.Attribute):
            attr_name = function.attr
            attr_value = function.value
            if attr_name == "_IgnoreContextManager" and isinstance(
                attr_value, ast.Attribute
            ):
                # 应最多存在两个嵌套的属性（例如 torch.jit._IgnoreContextManager）
                if attr_value.attr == "jit" and isinstance(attr_value.value, ast.Name):
                    if attr_value.value.id == "torch":
                        return True
    return False


class Builder:
    def __call__(self, ctx, node):
        # 获取与节点类名对应的构建方法，如果不存在则抛出异常
        method = getattr(self, "build_" + node.__class__.__name__, None)
        if method is None:
            raise UnsupportedNodeError(ctx, node)
        return method(ctx, node)


def build_class_def(ctx, py_def, methods, properties, self_name, assigns):
    r = ctx.make_range(
        py_def.lineno, py_def.col_offset, py_def.col_offset + len("class")
    )
    return ClassDef(
        Ident(r, self_name), [Stmt(method) for method in methods], properties, assigns
    )


def build_def(ctx, py_def, type_line, def_name, self_name=None, pdt_arg_types=None):
    body = py_def.body
    r = ctx.make_range(py_def.lineno, py_def.col_offset, py_def.col_offset + len("def"))

    param_list = build_param_list(ctx, py_def.args, self_name, pdt_arg_types)
    return_type = None
    if getattr(py_def, "returns", None) is not None:
        return_type = build_expr(ctx, py_def.returns)

    decl = Decl(r, param_list, return_type)
    is_method = self_name is not None
    if type_line is not None:
        # 解析类型注释行并合并到声明中
        type_comment_decl = torch._C.parse_type_comment(type_line)
        decl = torch._C.merge_type_from_type_comment(decl, type_comment_decl, is_method)

    return Def(Ident(r, def_name), decl, build_stmts(ctx, body))


_vararg_kwarg_err = (
    "Compiled functions can't take variable number of arguments "
    "or use keyword-only arguments with defaults"
)


def build_param_list(ctx, py_args, self_name, pdt_arg_types=None):
    if py_args.kwarg is not None:
        expr = py_args.kwarg
        ctx_range = ctx.make_range(
            expr.lineno, expr.col_offset - 1, expr.col_offset + len(expr.arg)
        )
        raise NotSupportedError(ctx_range, _vararg_kwarg_err)
    if py_args.vararg is not None:
        expr = py_args.vararg
        ctx_range = ctx.make_range(
            expr.lineno, expr.col_offset - 1, expr.col_offset + len(expr.arg)
        )
        raise NotSupportedError(ctx_range, _vararg_kwarg_err)
    # 检查是否存在关键字参数的默认值
    if len(py_args.kw_defaults) > 0:
        # kw_defaults 是关键字参数的默认值列表（默认为 None），
        # 因此它们实际上没有行号。
        for arg in py_args.kw_defaults:
            # 如果默认值不为 None，则构建表达式并获取其范围
            if arg is not None:
                ctx_range = build_expr(ctx, arg).range()
                # 抛出不支持的错误，指定错误范围和错误消息
                raise NotSupportedError(ctx_range, _vararg_kwarg_err)

    # 由分析型类型推断的参数及其类型的列表
    arg_and_types = [
        (
            arg,
            pdt_arg_types[arg.arg]
            if pdt_arg_types and bool(pdt_arg_types[arg.arg])
            else None,
        )
        for arg in py_args.args
    ]
    
    # 由分析型类型推断的关键字参数及其类型的列表
    arg_and_types_kwonlyargs = [
        (
            arg,
            pdt_arg_types[arg.arg]
            if pdt_arg_types and bool(pdt_arg_types[arg.arg])
            else None,
        )
        for arg in py_args.kwonlyargs
    ]

    # 构建参数对象的列表，不包括仅限关键字参数
    result = [
        build_param(ctx, arg, self_name, kwarg_only=False, pdt_arg_type=arg_type)
        for arg, arg_type in arg_and_types
    ]
    
    # 将仅限关键字参数添加到结果列表中
    result += [
        build_param(ctx, arg, self_name, kwarg_only=True, pdt_arg_type=arg_type)
        for arg, arg_type in arg_and_types_kwonlyargs
    ]
    
    # 返回最终构建的参数列表
    return result
# 构建参数对象的函数
def build_param(ctx, py_arg, self_name, kwarg_only, pdt_arg_type=None):
    # 获取参数名
    name = py_arg.arg
    # 创建表示参数位置的范围对象
    r = ctx.make_range(py_arg.lineno, py_arg.col_offset, py_arg.col_offset + len(name))
    # 构建参数的类型注解表达式
    if getattr(py_arg, "annotation", None) is not None:
        annotation_expr = build_expr(ctx, py_arg.annotation)
    elif pdt_arg_type:
        annotation_expr = Var(Ident(r, pdt_arg_type))
    elif self_name is not None and name == "self":
        annotation_expr = Var(Ident(r, self_name))
    else:
        annotation_expr = EmptyTypeAnnotation(r)
    # 返回参数对象
    return Param(annotation_expr, Ident(r, name), kwarg_only)


# 构建忽略上下文管理器的函数
def build_ignore_context_manager(ctx, stmt):
    # 定义输入类型和输出类型的命名元组
    InputType = namedtuple("InputType", ["name", "ann"])
    OutputType = namedtuple("OutputType", ["name", "ann"])

    def process_ins_outs(args):
        # 解析上下文管理器以确定输入和输出及其带有注解的类型
        inputs = []
        outputs = []
        for arg in args:
            var_name = arg.arg
            var_ann = arg.value.value
            var_decl_type, var_ann = var_ann.split(":")
            if var_decl_type == "inp":
                inputs.append(InputType(var_name, var_ann))
            if var_decl_type == "out":
                outputs.append(OutputType(var_name, var_ann))
        return inputs, outputs

    def create_unique_name_ext(ctx, stmt):
        # 根据完整路径文件名和原始上下文管理器的行号创建唯一名称扩展
        fn = re.sub(r"[^a-zA-Z0-9_]", "_", ctx.filename)
        return f"{fn}_{stmt.lineno}"

    def build_return_ann_stmt(outputs):
        # 构建返回类型注解字符串和返回语句字符串
        return_type_ann = ""
        return_statement_str = "return "
        if len(outputs) == 0:
            return_type_ann += " -> None"
        elif len(outputs) == 1:
            return_type_ann = " -> " + outputs[0].ann
            return_statement_str += outputs[0].name
        elif len(outputs) > 1:
            return_type_ann = " -> Tuple"
            return_type_ann += "[" + ", ".join([var.ann for var in outputs]) + "]"
            return_statement_str += ", ".join([var.name for var in outputs])
        return return_type_ann, return_statement_str

    def build_args(args):
        # 构建参数字符串
        return ", ".join([arg.name for arg in args])

    # 解析输入和输出
    inputs, outputs = process_ins_outs(stmt.items[0].context_expr.keywords)

    # 构建忽略函数的名称
    ignore_function_name = "func_ignore_" + create_unique_name_ext(ctx, stmt)
    ignore_function_str = "\ndef " + ignore_function_name
    ignore_function_str += (
        "(" + ", ".join([var.name + " :" + var.ann for var in inputs]) + ")"
    )

    # 构建返回类型注解和函数体语句
    return_ann, return_stmt = build_return_ann_stmt(outputs)
    ignore_function_str += return_ann + ": pass"

    # 创建函数Def对象，仅从声明创建
    # 解析 ignore_function_str 字符串为 AST（抽象语法树），并获取其第一个 body 元素
    ignore_function = ast.parse(ignore_function_str).body[0]

    # 将 context manager 的主体替换为 dummy function 的 body
    ignore_function.body = stmt.body  # type: ignore[attr-defined]

    # 解析 return_stmt 字符串为 AST，并获取其第一个 body 元素作为 return 语句
    return_stmt = ast.parse(return_stmt).body[0]
    # 将 return 语句追加到 dummy function 的 body 中
    ignore_function.body.append(return_stmt)  # type: ignore[attr-defined]

    # 构建 ignore_function 的字符串表示，标记为 @torch.jit.ignore
    ignore_func_str = "@torch.jit.ignore\n" + astunparse.unparse(ignore_function)
    # 将 ignore_function 注册到全局上下文中
    ignore_func_str += f'\nglobals()["{ignore_function_name}"] = {ignore_function_name}'
    exec(ignore_func_str)  # noqa: P204

    # 构建赋值语句字符串，左侧是 build_args(outputs) 的结果
    assign_str_lhs = build_args(outputs)
    # 构建赋值语句字符串，右侧是 torch.jit.frontend.<ignore_function_name>(<in_1>, <in_2>) 的结果
    assign_str_rhs = (
        f"torch.jit.frontend.{ignore_function_name}(" + build_args(inputs) + ")"
    )

    # 如果 outputs 非空，构建完整的赋值语句
    if len(outputs) > 0:
        assign_str = assign_str_lhs + " = " + assign_str_rhs
    else:
        assign_str = assign_str_rhs
    # 解析赋值语句字符串为 AST 的 body 元素
    assign_ast = ast.parse(assign_str).body[0]
    # 返回赋值语句的 AST 表示
    return assign_ast
    def get_default_args(fn):
        """
        Get a dictionary of default arguments for a function.

        Args:
            fn: Callable - The function to inspect for default arguments.
        Returns:
            (Dict[str, Any]): mapping argument names to their default values if
            :attr:`fn` is not None, else empty dictionary.
        """
        if fn is None:
            return {}

        # 获取函数的签名信息
        signature = inspect.signature(fn)

        # 构建并返回参数名到默认值的字典
        return {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }


    def get_default_args_for_class(cls):
        """
        Get default arguments for all methods in a class (except for static methods).

        Args:
            cls: type - The class type to inspect for default arguments.
        Returns:
            A Dict[str, Dict[str, Any]] which maps each method name to a Dict[str, Any]
            that maps each argument name to its default value.
        """
        # 获取类中的方法（排除静态方法，因为它们会被单独编译为独立的脚本函数）
        methods = inspect.getmembers(
            cls,
            predicate=lambda m: (inspect.ismethod(m) or inspect.isfunction(m))
            and not is_static_fn(cls, m.__name__)
            and m.__name__ in cls.__dict__,
        )

        # 获取方法的默认参数信息
        defaults = {
            method_name: get_default_args(method_impl)
            for method_name, method_impl in methods
        }

        return defaults


    class WithItemBuilder(Builder):
        @staticmethod
        def build_withitem(ctx, item):
            lineno = item.context_expr.lineno
            start = item.context_expr.col_offset
            end = start + len(pretty_node_names[ast.With])
            op_vars = item.optional_vars
            r = ctx.make_range(lineno, start, end)

            return WithItem(
                r,
                build_expr(ctx, item.context_expr),
                build_expr(ctx, op_vars) if op_vars else None,
            )


    class StmtBuilder(Builder):
        augassign_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Mod: "%",
            ast.BitOr: "|",
            ast.BitAnd: "&",
            ast.BitXor: "^",
            ast.LShift: "<<",
            ast.RShift: ">>",
            ast.Pow: "**",
        }

        @staticmethod
        def build_Expr(ctx, stmt):
            value = stmt.value
            if value.__class__.__name__ == "Str":
                # 如果语句是字符串字面表达式，则是文档字符串，忽略处理
                return None
            else:
                # 构建表达式语句并返回
                return ExprStmt(build_expr(ctx, value))

        @staticmethod
        def build_Assign(ctx, stmt):
            # 构建赋值语句的右侧表达式
            rhs = build_expr(ctx, stmt.value)
            # 构建赋值语句的左侧表达式列表
            lhs = [build_expr(ctx, x) for x in stmt.targets]
            return Assign(lhs, rhs)

        @staticmethod
    # 定义静态方法 `build_AnnAssign`，用于构建赋值语句节点
    def build_AnnAssign(ctx, stmt):
        # 如果赋值语句没有值，则抛出不支持的节点错误
        if stmt.value is None:
            raise UnsupportedNodeError(ctx, stmt, reason="without assigned value")

        # 如果赋值目标是实例属性且不在 `__init__` 方法之外，则抛出数值错误
        if (
            type(stmt.target) == ast.Attribute
            and stmt.target.value.id == "self"  # type: ignore[attr-defined]
            and ctx.funcname != "__init__"
        ):
            start = stmt.col_offset
            end = start + len(f"self.{stmt.target.attr}")
            if hasattr(stmt.annotation, "id"):
                end += len(f": {stmt.annotation.id}")
            sr = ctx.make_range(stmt.lineno, start, end)
            raise ValueError(
                "Type annotations on instance attributes must be declared in "
                f"__init__, not '{ctx.funcname}': {sr}"
            )

        # 构建赋值右侧表达式
        rhs = build_expr(ctx, stmt.value)
        # 构建赋值左侧表达式
        lhs = build_expr(ctx, stmt.target)
        # 构建赋值类型注解表达式
        the_type = build_expr(ctx, stmt.annotation)
        # 返回赋值节点对象
        return Assign([lhs], rhs, the_type)

    # 定义静态方法 `build_Delete`，用于构建删除语句节点
    @staticmethod
    def build_Delete(ctx, stmt):
        # 构建删除操作的位置范围
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("del"))
        # 构建删除语句节点，包含删除的目标表达式
        return Delete(r, [build_expr(ctx, target) for target in stmt.targets])

    # 定义静态方法 `build_Return`，用于构建返回语句节点
    @staticmethod
    def build_Return(ctx, stmt):
        # 构建返回语句的位置范围
        r = ctx.make_range(
            stmt.lineno, stmt.col_offset, stmt.col_offset + len("return")
        )
        # 构建返回语句节点，包含返回值表达式
        return Return(r, None if stmt.value is None else build_expr(ctx, stmt.value))

    # 定义静态方法 `build_Raise`，用于构建抛出异常语句节点
    @staticmethod
    def build_Raise(ctx, stmt):
        # 构建抛出异常语句的位置范围
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("raise"))
        # 构建抛出异常语句节点，包含异常表达式
        expr = build_expr(ctx, stmt.exc)
        return Raise(r, expr)

    # 定义静态方法 `build_Assert`，用于构建断言语句节点
    @staticmethod
    def build_Assert(ctx, stmt):
        # 构建断言语句的位置范围
        r = ctx.make_range(
            stmt.lineno, stmt.col_offset, stmt.col_offset + len("assert")
        )
        # 构建断言测试表达式
        test = build_expr(ctx, stmt.test)
        # 如果有消息表达式，则构建消息表达式
        msg = build_expr(ctx, stmt.msg) if stmt.msg is not None else None
        # 返回断言语句节点对象
        return Assert(r, test, msg)

    # 定义静态方法 `build_AugAssign`，用于构建增强赋值语句节点
    @staticmethod
    def build_AugAssign(ctx, stmt):
        # 构建增强赋值左侧表达式
        lhs = build_expr(ctx, stmt.target)
        # 构建增强赋值右侧表达式
        rhs = build_expr(ctx, stmt.value)
        # 获取增强赋值操作符类型
        op = type(stmt.op)
        # 如果操作符类型在增强赋值映射中，则获取对应的操作符标记
        if op in StmtBuilder.augassign_map:
            op_token = StmtBuilder.augassign_map[op]
        else:
            # 否则抛出不支持的错误，指出不支持的增强赋值类型
            raise NotSupportedError(
                find_before(ctx, rhs.range().start, "=", offsets=(-1, 0)),
                "unsupported kind of augmented assignment: " + op.__name__,
            )
        # 返回增强赋值语句节点对象
        return AugAssign(lhs, op_token, rhs)
    def build_While(ctx, stmt):
        # 如果 while 循环有 else 分支，则抛出 NotSupportedError 异常
        if stmt.orelse:
            # TODO: 尝试恢复 else 分支的位置：Python 在这种情况下没有给出有用的注释
            raise NotSupportedError(
                None, "else branches of while loops aren't supported"
            )
        # 构建并返回 While 对象，包括位置信息和条件表达式以及循环体的语句
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("while"))
        return While(r, build_expr(ctx, stmt.test), build_stmts(ctx, stmt.body))

    @staticmethod
    def build_For(ctx, stmt):
        # 构建 for 循环的位置信息
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("for"))
        # 如果 for 循环有 else 分支，则抛出 NotSupportedError 异常
        if stmt.orelse:
            raise NotSupportedError(r, "else branches of for loops aren't supported")

        # 构建并返回 For 对象，包括位置信息、目标表达式、迭代器表达式以及循环体的语句
        return For(
            r,
            [build_expr(ctx, stmt.target)],
            [build_expr(ctx, stmt.iter)],
            build_stmts(ctx, stmt.body),
        )

    @staticmethod
    def build_If(ctx, stmt):
        # 构建 if 语句的位置信息
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("if"))
        # 构建并返回 If 对象，包括位置信息、条件表达式、if分支的语句以及else分支的语句（如果存在）
        return If(
            r,
            build_expr(ctx, stmt.test),
            build_stmts(ctx, stmt.body),
            build_stmts(ctx, stmt.orelse),
        )

    @staticmethod
    def build_Print(ctx, stmt):
        # 构建 print 语句的位置信息
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("print"))
        # 如果 print 语句有非默认目标，则抛出 NotSupportedError 异常
        if stmt.dest:
            raise NotSupportedError(
                r, "print statements with non-default destinations aren't supported"
            )
        # 构建打印参数表达式列表，并返回 ExprStmt 对象
        args = [build_expr(ctx, val) for val in stmt.values]
        return ExprStmt(Apply(Var(Ident(r, "print")), args, []))

    @staticmethod
    def build_Pass(ctx, stmt):
        # 构建 pass 语句的位置信息，并返回 Pass 对象
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("pass"))
        return Pass(r)

    @staticmethod
    def build_Break(ctx, stmt):
        # 构建 break 语句的位置信息，并返回 Break 对象
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("break"))
        return Break(r)

    @staticmethod
    def build_Continue(ctx, stmt):
        # 构建 continue 语句的位置信息，并返回 Continue 对象
        r = ctx.make_range(
            stmt.lineno, stmt.col_offset, stmt.col_offset + len("continue")
        )
        return Continue(r)

    @staticmethod
    def build_With(ctx, stmt):
        # 构建 with 语句的位置信息
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len("with"))
        # 处理 torch.jit._IgnoreContextManager 上下文管理器
        if is_torch_jit_ignore_context_manager(stmt):
            # 如果没有安装 astunparse 库，则抛出 RuntimeError 异常
            if not _IS_ASTUNPARSE_INSTALLED:
                raise RuntimeError(
                    "torch.jit._IgnoreContextManager requires installing Python library `astunparse`, \
                                   please install it in your Python environment"
                )
            # 构建并返回 ignore 上下文管理器的赋值语句
            assign_ast = build_ignore_context_manager(ctx, stmt)
            return build_stmt(ctx, assign_ast)
        # 构建并返回 With 对象，包括位置信息、上下文项和 with 块的语句
        return With(r, build_withitems(ctx, stmt.items), build_stmts(ctx, stmt.body))
    # 定义 ExprBuilder 类，继承自 Builder 类
    class ExprBuilder(Builder):

        # 二元操作符映射，将 AST 中的操作符映射到对应的字符串表示
        binop_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Pow: "**",
            ast.Mod: "%",
            ast.FloorDiv: "//",
            ast.BitAnd: "&",
            ast.BitXor: "^",
            ast.BitOr: "|",
            ast.LShift: "<<",
            ast.RShift: ">>",
        }

        # MatMult 操作符特殊处理，映射为 "@"
        binop_map[ast.MatMult] = "@"

        # 单目操作符映射，将 AST 中的操作符映射到对应的字符串表示
        unop_map = {
            ast.Not: "not",
            ast.USub: "-",
            ast.Invert: "~",
        }

        # 布尔操作符映射，将 AST 中的操作符映射到对应的字符串表示
        boolop_map = {
            ast.And: "and",
            ast.Or: "or",
        }

        # 比较操作符映射，将 AST 中的操作符映射到对应的字符串表示
        cmpop_map = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.LtE: "<=",
            ast.Lt: "<",
            ast.GtE: ">=",
            ast.Gt: ">",
            ast.Is: "is",
            ast.IsNot: "is not",
            ast.In: "in",
            ast.NotIn: "not in",
        }

        # 静态方法：构建 Attribute 类型的表达式
        @staticmethod
        def build_Attribute(ctx, expr):
            # 构建基础表达式
            base = build_expr(ctx, expr.value)
            # expr.attr 只是一个字符串，没有任何注释，因此需要手动构建范围
            source = ctx.source.encode("utf-8")

            # 获取字符的函数
            def get_char(index):
                return chr(source[index])

            # 起始位置是基础表达式的末尾位置 + 1
            start_pos = base.range().end + 1
            # 跳过空白字符
            while get_char(start_pos) in string.whitespace:
                start_pos += 1
            # 结束位置是起始位置加上属性名的长度
            end_pos = start_pos + len(expr.attr)
            # 创建属性名的范围
            name_range = ctx.make_raw_range(start_pos, end_pos)
            # 返回 Select 对象，表示属性选择
            return Select(base, Ident(name_range, expr.attr))

        # 静态方法：构建 Call 类型的表达式
        @staticmethod
        def build_Call(ctx, expr):
            # 构建函数表达式
            func = build_expr(ctx, expr.func)
            # 构建参数表达式列表
            args = [build_expr(ctx, py_arg) for py_arg in expr.args]
            # 处理 starargs 参数
            if hasattr(expr, "starargs") and expr.starargs:
                stararg_expr = build_expr(ctx, expr.starargs)
                args += [Starred(stararg_expr.range(), stararg_expr)]
            kwargs = []
            # 处理关键字参数
            for kw in expr.keywords:
                kw_expr = build_expr(ctx, kw.value)
                # XXX: 在这里我们可能可以更好地确定名称的范围
                if not kw.arg:
                    raise NotSupportedError(
                        kw_expr.range(), "keyword-arg expansion is not supported"
                    )
                # 添加属性到关键字参数列表
                kwargs.append(Attribute(Ident(kw_expr.range(), kw.arg), kw_expr))
            # 返回 Apply 对象，表示函数应用
            return Apply(func, args, kwargs)

        # 静态方法：构建 Ellipsis 类型的表达式
        @staticmethod
        def build_Ellipsis(ctx, expr):
            # 创建省略号的范围
            r = ctx.make_range(
                expr.lineno, expr.col_offset, expr.col_offset + 3
            )  # len("...") == 3
            # 返回 Dots 对象，表示省略号
            return Dots(r)

        @staticmethod
    # 构建名称表达式的方法，接受上下文和表达式对象作为参数
    def build_Name(ctx, expr):
        # 创建表示表达式位置范围的对象
        r = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + len(expr.id))
        # 检查变量名是否以预留前缀开头，如果是则抛出不支持的错误
        if expr.id.startswith(_reserved_prefix):
            raise NotSupportedError(
                r,
                "names of variables used in JIT-ed functions "
                "can't start with " + _reserved_prefix,
            )
        # 如果变量名是 "True"，返回对应的 TrueLiteral 对象
        elif expr.id == "True":
            return TrueLiteral(r)
        # 如果变量名是 "False"，返回对应的 FalseLiteral 对象
        elif expr.id == "False":
            return FalseLiteral(r)
        # 如果变量名是 "None"，返回对应的 NoneLiteral 对象
        elif expr.id == "None":
            return NoneLiteral(r)
        # 如果变量名是 "Ellipsis"，返回对应的 Dots 对象
        elif expr.id == "Ellipsis":
            return Dots(r)
        # 否则，返回一个表示变量的 Var 对象
        return Var(Ident(r, expr.id))

    # 构建 NameConstant 表达式的静态方法，接受上下文和表达式对象作为参数
    @staticmethod
    def build_NameConstant(ctx, expr):
        # 创建表示表达式位置范围的对象
        r = ctx.make_range(
            expr.lineno, expr.col_offset, expr.col_offset + len(str(expr.value))
        )
        # 根据表达式的值类型返回相应的字面值对象
        if expr.value is True:
            return TrueLiteral(r)
        elif expr.value is False:
            return FalseLiteral(r)
        elif expr.value is None:
            return NoneLiteral(r)
        elif expr.value == Ellipsis:
            return Dots(r)
        else:
            # 如果表达式的值不是预期的常量类型，则抛出值错误
            raise ValueError("Name constant value unsupported: " + str(expr.value))

    # 构建 BinOp 表达式的静态方法，接受上下文和表达式对象作为参数
    @staticmethod
    def build_BinOp(ctx, expr):
        # 构建左右操作数的表达式对象
        lhs = build_expr(ctx, expr.left)
        rhs = build_expr(ctx, expr.right)
        # 获取操作符的类型
        op = type(expr.op)

        # 检查除法操作是否需要 Python 3 真除法语义，否则抛出前端错误
        if op == ast.Div and not ctx.uses_true_division:
            err_range = ctx.make_raw_range(lhs.range().end, rhs.range().start)
            raise FrontendError(
                err_range,
                "Division of ints in TorchScript uses Python 3 true "
                "division semantics. Please put `from __future__ "
                "import division` at the top of your file",
            )

        # 获取操作符对应的标记，如果不支持则抛出不支持错误
        op_token = ExprBuilder.binop_map.get(op)
        if op_token is None:
            err_range = ctx.make_raw_range(lhs.range().end, rhs.range().start)
            raise NotSupportedError(
                err_range, "unsupported binary operator: " + op.__name__
            )
        
        # 返回表示二元操作的 BinOp 对象
        return BinOp(op_token, lhs, rhs)

    # 构建 UnaryOp 表达式的静态方法，接受上下文和表达式对象作为参数
    @staticmethod
    def build_UnaryOp(ctx, expr):
        # 构建子表达式的表达式对象
        sub_expr = build_expr(ctx, expr.operand)
        # 获取操作符的类型
        op = type(expr.op)
        # 获取操作符对应的标记，如果不支持则抛出不支持错误
        op_token = ExprBuilder.unop_map.get(op)
        if op_token is None:
            raise NotSupportedError(
                expr.range(), "unsupported unary operator: " + op.__name__
            )
        
        # 创建表示表达式位置范围的对象
        r = ctx.make_range(
            expr.lineno, expr.col_offset, expr.col_offset + len(op_token)
        )
        # 返回表示一元操作的 UnaryOp 对象
        return UnaryOp(r, op_token, sub_expr)
    # 构建布尔操作表达式的方法
    def build_BoolOp(ctx, expr):
        # 检查布尔操作表达式是否至少包含两个值，否则引发断言错误
        if len(expr.values) < 2:
            raise AssertionError(
                "expected at least 2 values in BoolOp, but got " + str(len(expr.values))
            )
        # 递归构建子表达式列表
        sub_exprs = [build_expr(ctx, sub_expr) for sub_expr in expr.values]
        # 获取布尔操作符的类型并映射到相应的符号
        op = type(expr.op)
        op_token = ExprBuilder.boolop_map.get(op)
        # 如果操作符不支持，抛出不支持错误
        if op_token is None:
            err_range = ctx.make_raw_range(
                sub_exprs[0].range().end, sub_exprs[1].range().start
            )
            raise NotSupportedError(
                err_range, "unsupported boolean operator: " + op.__name__
            )
        # 从左到右依次构建二元操作表达式
        lhs = sub_exprs[0]
        for rhs in sub_exprs[1:]:
            lhs = BinOp(op_token, lhs, rhs)
        return lhs

    # 构建三元条件表达式的静态方法
    @staticmethod
    def build_IfExp(ctx, expr):
        # 返回三元条件表达式对象，包括条件、真值和假值的构建表达式
        return TernaryIf(
            build_expr(ctx, expr.test),
            build_expr(ctx, expr.body),
            build_expr(ctx, expr.orelse),
        )

    # 构建比较表达式的静态方法
    @staticmethod
    def build_Compare(ctx, expr):
        # 构建所有操作数的表达式
        operands = [build_expr(ctx, e) for e in [expr.left] + list(expr.comparators)]
        result = None
        # 逐个处理左操作数、操作符和右操作数的比较
        for lhs, op_, rhs in zip(operands, expr.ops, operands[1:]):
            op = type(op_)
            op_token = ExprBuilder.cmpop_map.get(op)
            # 构建操作符的原始范围
            r = ctx.make_raw_range(lhs.range().end, rhs.range().start)
            # 如果操作符不支持，抛出不支持错误
            if op_token is None:
                raise NotSupportedError(
                    r, "unsupported comparison operator: " + op.__name__
                )

            # 处理 `not in` 操作符的特殊情况，构建嵌套调用树视图
            if op == ast.NotIn:
                in_expr = BinOp("in", lhs, rhs)
                cmp_expr = UnaryOp(r, "not", in_expr)
            else:
                cmp_expr = BinOp(op_token, lhs, rhs)

            # 将每个比较表达式链接成逻辑与的链
            if result is None:
                result = cmp_expr
            else:
                result = BinOp("and", result, cmp_expr)
        return result

    # 构建列表表达式的静态方法
    @staticmethod
    def build_List(ctx, expr):
        # 返回列表字面量对象，包括范围和所有元素的构建表达式
        return ListLiteral(
            ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + 1),
            [build_expr(ctx, e) for e in expr.elts],
        )

    # 构建元组表达式的静态方法
    @staticmethod
    def build_Tuple(ctx, expr):
        # 返回元组字面量对象，包括范围和所有元素的构建表达式
        return TupleLiteral(
            ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + 1),
            [build_expr(ctx, e) for e in expr.elts],
        )

    # 构建字典表达式的静态方法
    @staticmethod
    def build_Dict(ctx, expr):
        # 构建字典字面量的范围
        range = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + 1)
        # 如果字典中包含键但第一个键为空，则抛出不支持的错误
        if expr.keys and not expr.keys[0]:
            raise NotSupportedError(
                range, "Dict expansion (e.g. `{**dict}`) is not supported"
            )
        # 返回字典字面量对象，包括范围、键列表和值列表的构建表达式
        return DictLiteral(
            range,
            [build_expr(ctx, e) for e in expr.keys],
            [build_expr(ctx, e) for e in expr.values],
        )
    # 定义一个静态方法，用于构建数值常量表达式
    def build_Num(ctx, expr):
        # 将表达式的值转换为字符串
        value = str(expr.value)
        # 创建表达式值所在位置的范围对象
        r = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + len(value))
        # 返回一个常量表达式对象，表示数值常量
        return Const(r, value)

    @staticmethod
    # 定义一个静态方法，用于构建常量表达式
    def build_Constant(ctx, expr):
        # 获取表达式的值
        value = expr.value
        # 如果值为None或布尔类型，则调用相应的构建方法
        if value is None or isinstance(value, bool):
            # 注意：布尔类型需要在整数类型检查之前进行，因为布尔是整数的子类
            return ExprBuilder.build_NameConstant(ctx, expr)
        # 如果值为整数、浮点数或复数，则调用构建数值常量的方法
        if isinstance(value, (int, float, complex)):
            return ExprBuilder.build_Num(ctx, expr)
        # 如果值为字符串，则调用构建字符串常量的方法
        elif isinstance(value, str):
            return ExprBuilder.build_Str(ctx, expr)
        # 如果值为省略号（Ellipsis）类型，则调用构建省略号的方法
        elif isinstance(value, type(Ellipsis)):
            return ExprBuilder.build_Ellipsis(ctx, expr)
        else:
            # 否则，抛出前端错误，表达式类型未知
            error_range = ctx.make_range(
                expr.lineno, expr.col_offset, expr.col_offset + len(str(value))
            )
            raise FrontendError(error_range, "Unknown Constant expression type")

    @staticmethod
    # 定义一个静态方法，用于构建字符串常量表达式
    def build_Str(ctx, expr):
        # 获取表达式的字符串值
        value = str(expr.value)
        # 创建表达式值所在位置的范围对象，加1是为了包括字符串的引号
        r = ctx.make_range(
            expr.lineno, expr.col_offset, expr.col_offset + len(value) + 1
        )
        # 返回一个字符串常量表达式对象
        return StringLiteral(r, value)

    @staticmethod
    # 定义一个静态方法，用于构建连接字符串表达式
    def build_JoinedStr(ctx, expr):
        # 初始化空字符串和参数列表
        s = ""
        args = []
        # 遍历连接字符串表达式的值列表
        for value in expr.values:
            # 创建当前值的位置范围对象
            r = ctx.make_range(value.lineno, value.col_offset, value.col_offset + 1)
            # 如果当前值是格式化值表达式
            if isinstance(value, ast.FormattedValue):
                # 检查是否支持转换，不支持则抛出错误
                if value.conversion != -1:
                    raise NotSupportedError(r, "Don't support conversion in JoinedStr")
                # 检查是否支持格式化，不支持则抛出错误
                if value.format_spec is not None:
                    raise NotSupportedError(r, "Don't support formatting in JoinedStr")
                # 将字符串模板增加占位符{}，并添加其值到参数列表
                s += "{}"
                args.append(build_expr(ctx, value.value))
            # 如果当前值是常量表达式，则直接连接其值到字符串
            elif isinstance(value, ast.Constant):
                s += value.value
            else:
                # 如果值不支持，则抛出不支持错误
                raise NotSupportedError(r, "Unsupported value in JoinedStr")

        # 创建连接字符串表达式的整体位置范围对象
        r = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + 1)
        # 返回应用格式化的字符串字面量对象
        return Apply(Select(StringLiteral(r, s), Ident(r, "format")), args, [])

    @staticmethod
    # 定义一个静态方法，用于构建列表推导表达式
    def build_ListComp(ctx, stmt):
        # 创建列表推导表达式整体位置范围对象
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset)
        # 检查生成器的数量，目前只支持单个生成器
        if len(stmt.generators) != 1:
            raise NotSupportedError(r, "Only a single generator is currently supported")

        # 检查推导式中是否有条件语句，目前不支持条件语句
        if len(stmt.generators[0].ifs) != 0:
            raise NotSupportedError(r, "Comprehension ifs are not supported yet")

        # 构建推导式的元素表达式、目标表达式和迭代表达式
        elt_expr = build_expr(ctx, stmt.elt)
        target_expr = build_expr(ctx, stmt.generators[0].target)
        iter_expr = build_expr(ctx, stmt.generators[0].iter)

        # 返回列表推导表达式对象
        return ListComp(r, elt_expr, target_expr, iter_expr)

    @staticmethod
    # 定义一个静态方法，用于构建生成器表达式
    def build_GeneratorExp(ctx, stmt):
        # 将生成器表达式转换为列表推导表达式并返回
        return ExprBuilder.build_ListComp(ctx, stmt)
    # 定义一个静态方法，用于构建字典推导式对象
    def build_DictComp(ctx, stmt):
        # 创建位置范围对象，标记语句的起始位置
        r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset)
        
        # 检查生成器列表长度是否为1，如果不是则抛出异常
        if len(stmt.generators) != 1:
            raise NotSupportedError(r, "Only a single generator is currently supported")
        
        # 检查生成器的条件表达式是否为空，如果不为空则抛出异常
        if len(stmt.generators[0].ifs) != 0:
            raise NotSupportedError(r, "Comprehension ifs are not supported yet")
        
        # 构建键表达式对象
        key_expr = build_expr(ctx, stmt.key)
        # 构建值表达式对象
        value_expr = build_expr(ctx, stmt.value)
        # 构建目标表达式对象（生成器的目标变量）
        target_expr = build_expr(ctx, stmt.generators[0].target)
        # 构建迭代表达式对象（生成器的迭代对象）
        iter_expr = build_expr(ctx, stmt.generators[0].iter)
        
        # 返回构建好的字典推导式对象
        return DictComp(r, key_expr, value_expr, target_expr, iter_expr)

    @staticmethod
    # 静态方法，用于构建星号表达式对象
    def build_Starred(ctx, expr):
        # 创建位置范围对象，标记表达式的起始位置
        r = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + 1)
        # 返回构建好的星号表达式对象，其中包含被星号标记的子表达式对象
        return Starred(r, build_expr(ctx, expr.value))
# 创建一个表达式构建器对象
build_expr = ExprBuilder()
# 创建一个语句构建器对象
build_stmt = StmtBuilder()
# 创建一个with语句项目构建器对象
build_withitem = WithItemBuilder()

# 在给定上下文中，在特定位置之前查找子字符串的起始位置，并返回对应的文本范围
def find_before(ctx, pos, substr, offsets=(0, 0)):
    # 在源代码中，从头到指定位置pos之间找到最后一次出现substr的位置
    new_pos = ctx.source[:pos].rindex(substr)
    # 创建并返回一个包含新位置范围的文本范围对象，考虑偏移量调整
    return ctx.make_raw_range(new_pos + offsets[0], new_pos + len(substr) + offsets[1])
```