# `.\pytorch\torch\_jit_internal.py`

```
# mypy: allow-untyped-defs
"""
The weak_script annotation needs to be here instead of inside torch/jit/ so it
can be used in other places in torch/ (namely torch.nn) without running into
circular dependency problems
"""

# 引入标准库和第三方库
import ast  # Abstract Syntax Trees (AST) 的解析和操作
import builtins  # Python 内置函数和变量的集合
import collections  # 扩展的容器数据类型
import contextlib  # 上下文管理工具
import enum  # 创建枚举类型
import inspect  # 解析对象、源码检查等功能
import io  # 用于处理流的核心工具
import pickle  # Python 对象的序列化和反序列化
import sys  # 提供与 Python 解释器相关的操作
import textwrap  # 文本包装和填充
import threading  # 多线程支持
import types  # 提供 Python 类型和类的操作支持
import typing  # Python 类型提示功能的核心支持
import warnings  # 警告处理
import weakref  # 弱引用对象的支持

# 从 typing 模块导入多个类型提示相关的符号
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    ForwardRef,
    get_args,
    get_origin,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

# 引入 torch 库
import torch

# 此处需要显式导入 torch.distributed.__init__，以避免在其他地方（如 torch.nn）中出现循环依赖问题
import torch.distributed.rpc

# 导入 torch.package._mangling 模块作为 package_mangling
import torch.package._mangling as package_mangling

# 从 torch._awaits 导入 _Await 类
from torch._awaits import _Await

# 从 torch._C 导入 _Await 类和 CFuture 类
from torch._C import _Await as CAwait, Future as CFuture

# 从 torch._sources 导入 fake_range、get_source_lines_and_file、parse_def 函数
from torch._sources import fake_range, get_source_lines_and_file, parse_def

# 从 torch.futures 导入 Future 类
from torch.futures import Future

# 检查 Python 版本是否大于等于 3.9 和 3.10，将结果保存在 IS_PY39_PLUS 和 IS_PY310_PLUS 常量中
IS_PY39_PLUS: Final[bool] = sys.version_info >= (3, 9)
IS_PY310_PLUS: Final[bool] = sys.version_info >= (3, 10)

# 定义 BuiltinUnionType 变量，根据 Python 版本选择不同的值
BuiltinUnionType: Union[Type, Tuple[Type, ...]]
if sys.version_info >= (3, 10):
    BuiltinUnionType = types.UnionType
else:
    BuiltinUnionType = ()  # trick: this makes isinstance short circuit.

# 定义 LockType 类型，根据是否能导入 _thread 模块做不同处理
LockType: Type
try:
    import _thread

    LockType = _thread.LockType
except ImportError:
    import _dummy_thread  # type: ignore[import-not-found]

    LockType = _dummy_thread.LockType

# 定义 boolean_dispatched 变量，使用弱引用字典存储 callable 对象和其对应的字典
boolean_dispatched: "weakref.WeakKeyDictionary[Callable, Dict[str, Callable]]" = (
    weakref.WeakKeyDictionary()
)  # noqa: T484

# 定义 FAKE_FILENAME_PREFIX 常量作为字符串 "__torch_jit_dataclass"
FAKE_FILENAME_PREFIX = "__torch_jit_dataclass"


class SourceLoader:
    """
    SourceLoader 类，用于加载和缓存源代码。
    """

    def __init__(self):
        self.content = {}  # 初始化内容字典

    def cache(self, fn, source):
        """
        将源代码缓存到 content 字典中。

        Args:
            fn: 文件名
            source: 源代码内容
        """
        self.content[fn] = source

    def get_source(self, fn):
        """
        获取指定文件名对应的源代码。

        Args:
            fn: 文件名

        Returns:
            文件名对应的源代码，如果不存在则返回 None。
        """
        return self.content.get(fn)


# 定义全局 loader 变量作为 SourceLoader 的实例
loader = SourceLoader()


def createResolutionCallbackFromEnv(lookup_base):
    """
    根据环境创建一个解析回调函数，用于查找限定名，并从 lookup_base 开始查找。

    Args:
        lookup_base: 查找的基础对象

    Returns:
        解析回调函数
    """
    """
    Creates a resolution callback that will look up qualified names in an
    environment, starting with `lookup_base` for the base of any qualified
    names, then proceeding down the lookup chain with the resolved object.

    You should not use this directly, it should only be used from the other
    createResolutionCallbackFrom* functions.
    """
    def lookupInModule(qualified_name, module):
        # 如果 qualified_name 中包含 "."，则将其拆分为基础和剩余部分
        if "." in qualified_name:
            base, remaining_pieces = qualified_name.split(".", maxsplit=1)
            # 获取模块中基础名称对应的值
            module_value = getattr(module, base)
            # 递归调用查找剩余部分的名称在模块中的值
            return lookupInModule(remaining_pieces, module_value)
        else:
            # 如果 qualified_name 中没有 "."，直接返回模块中的对应属性值
            return getattr(module, qualified_name)

    def parseNestedExpr(expr, module) -> Tuple[Any, int]:
        # 初始化索引 i
        i = 0
        # 循环直到表达式结束或遇到分隔符 ", [ ]"
        while i < len(expr) and expr[i] not in (",", "[", "]"):
            i += 1

        # 处理特殊情况：空元组作为下标（在类型注释 `Tuple[()]` 中使用）
        if expr[:i] == "()":
            return (), i

        # 解析基础部分表达式名称并获取对应模块中的值
        base = lookupInModule(expr[:i].strip(), module)
        # 断言基础部分不为 None，否则抛出异常
        assert base is not None, f"Unresolvable type {expr[:i]}"
        # 如果当前索引 i 等于表达式长度或当前字符不是 "["
        if i == len(expr) or expr[i] != "[":
            return base, i

        # 断言当前字符为 "["
        assert expr[i] == "["
        # 初始化部分列表
        parts = []
        # 循环解析嵌套的表达式部分直到遇到 "]"
        while expr[i] != "]":
            part_len = 0
            i += 1
            # 解析当前位置后的嵌套表达式并获取其值和解析长度
            part, part_len = parseNestedExpr(expr[i:], module)
            parts.append(part)
            i += part_len
        # 如果部分数量大于1，返回基础部分的元组形式，以及最终的索引
        if len(parts) > 1:
            return base[tuple(parts)], i + 1
        else:
            # 如果只有一个部分，返回基础部分中对应的值和最终索引
            return base[parts[0]], i + 1

    def parseExpr(expr, module):
        try:
            # 尝试解析嵌套表达式并获取其值和解析长度
            value, len_parsed = parseNestedExpr(expr, module)
            # 断言整个表达式都已解析
            assert len_parsed == len(
                expr
            ), "whole expression was not parsed, falling back to c++ parser"
            # 返回解析后的值
            return value
        except Exception:
            """
            Python 解析器在已知的单元测试中会失败，通常会优雅地回退到 C++ 解析器。
            例如，我们的单元测试中经常会出现 Python 2 风格的注解，这些类型如 int 无法从调用帧中解析。
            """
            # 解析失败时返回 None
            return None

    # 返回一个 lambda 函数，用于解析给定表达式
    return lambda expr: parseExpr(expr, lookup_base)
# 创建一个从调用者作用域中获取变量值的函数
def createResolutionCallbackFromFrame(frames_up: int = 0):
    """
    Creates a function which, given a string variable name,
    returns the value of the variable in the scope of the caller of
    the function which called createResolutionCallbackFromFrame (by default).

    This is used to enable access in-scope Python variables inside
    TorchScript fragments.

    frames_up is number of additional frames to go up on the stack.
    The default value is 0, which correspond to the frame of the caller
    of createResolutionCallbackFromFrame. Also for example, if frames_up is set
    to 1, then the frame of the caller's caller of createResolutionCallbackFromFrame
    will be taken.

    For example, the following program prints 2::

        def bar():
            cb = createResolutionCallbackFromFrame(1)
            print(cb("foo"))

        def baz():
            foo = 2
            bar()

        baz()
    """
    # 获取当前的调用栈帧
    frame = inspect.currentframe()
    i = 0
    # 向上遍历调用栈帧，直到达到指定的帧数 frames_up + 1
    while i < frames_up + 1:
        assert frame is not None
        frame = frame.f_back
        i += 1

    assert frame is not None
    # 获取局部变量和全局变量字典
    f_locals = frame.f_locals
    f_globals = frame.f_globals

    # 创建一个环境类，用于获取变量值
    class env:
        def __getattr__(self, key):
            if key in f_locals:
                return f_locals[key]
            elif key in f_globals:
                return f_globals[key]
            elif key in dir(builtins):
                return getattr(builtins, key)

    # 返回基于当前环境的解析回调函数
    return createResolutionCallbackFromEnv(env())


# 获取函数闭包中捕获的变量字典
def get_closure(fn):
    """
    Get a dictionary of closed over variables from a function
    """
    captures = {}
    captures.update(fn.__globals__)

    # 遍历函数闭包中的自由变量
    for index, captured_name in enumerate(fn.__code__.co_freevars):
        captures[captured_name] = fn.__closure__[index].cell_contents

    return captures


# [local resolution in python]
# Depending on where a variable is defined, and where it is used, we may
# or may not be able to recover its value when recursively compiling a
# script function. Remember in the general case, a module or function is
# first defined and then later scripted. This means we do not have a
# chance to capture the active frames when the function is defined. Hence any
# name resolution has to happen later on the created closure. The way
# python captures type annotations restricts what we can recover. The
# follow example illustrates the different cases:
#
#         class MyGlobalClass:
#         ...
#         def my_local_scope():
#             @torch.jit.script
#             class MyClass:
#                 ...
#             @torch.jit.script
#             class MyClassUsedAsVar:
#                 ...
#             def eg(x: MyClass, y: MyGlobalClass):
#                 a_local_capture : Foo
#                 return MyClassUsedAsVar(x)
#
# MyGlobalClass is defined in the __globals__ dictionary of function
# 'eg', so it is always recoverable. my_local_scope introduces a new local
# 定义一个函数，从闭包中创建解析回调函数
def createResolutionCallbackFromClosure(fn):
    """
    通过检查函数来创建解析回调函数，而不是查找封闭作用域
    """
    # 获取函数的闭包
    closure = get_closure(fn)

    # 定义一个类 closure_lookup
    class closure_lookup:
        # 由于 `closure` 是一个字典，在 `env_helper` 中如果所有内容都使用 `getattr` 调用更加简单
        def __getattr__(self, key):
            # 如果 `key` 在 `closure` 中，则返回对应的值
            if key in closure:
                return closure[key]
            # 如果 `key` 在 `typing` 模块中存在，则返回对应的属性
            elif hasattr(typing, key):
                return getattr(typing, key)
            # 如果 `key` 在 `builtins` 模块中存在，则返回对应的属性
            elif hasattr(builtins, key):
                return getattr(builtins, key)
            # 如果都不满足，则返回 None
            return None

    # 调用 `createResolutionCallbackFromEnv` 函数，传入 `closure_lookup` 类的实例，并返回结果
    return createResolutionCallbackFromEnv(closure_lookup())


def can_compile_class(cls) -> bool:
    """
    判断给定的类是否可以编译

    如果类型的任何函数没有代码对象，则该类型不能被编译，可能是内置的或来自 C 的绑定类型
    """
    # 如果类是被忽略的函数，则返回 False
    if is_ignored_fn(cls):
        return False

    # 忽略以下内置类的检查
    ignored_builtin_classes = (torch.nn.Module, tuple, list, Exception)
    if issubclass(cls, ignored_builtin_classes):
        return False

    # 获取类的所有属性名
    names = cls.__dict__
    # 获取类的所有函数，并且是例程（routine）的函数对象列表
    fns = [
        getattr(cls, name)
        for name in names
        if inspect.isroutine(getattr(cls, name, None))
    ]
    # 检查每个函数对象是否有 `__code__` 属性，返回一个布尔值列表
    has_code = [hasattr(fn, "__code__") for fn in fns]
    # 如果所有函数都有 `__code__` 属性，则返回 True，否则返回 False
    return all(has_code)


def get_callable_argument_names(fn) -> List[str]:
    """
    获取可调用对象 `fn` 的所有 POSITIONAL_OR_KEYWORD 参数的名称列表

    当存在其他类型的参数时返回一个空列表

    这在 `torch.jit.trace` 中用于为追踪的函数和模块分配有意义的参数名称
    """
    # 获取函数的参数签名
    sig = inspect.signature(fn)
    # 返回所有 POSITIONAL_OR_KEYWORD 类型参数的名称列表
    return [p.name for p in sig.parameters.values()
            if p.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD}]
    # 定义一个函数，接受一个可调用对象 fn 作为参数，并返回该对象的参数名列表。
    def get_argument_names(fn):
        # 尝试使用 inspect.signature 获取 fn 的签名信息，可能会出现异常，如果失败则返回空列表。
        try:
            callable_signature = inspect.signature(fn)
        except Exception:
            return []
    
        # 初始化一个空列表，用于存储参数名。
        argument_names = []
        # 遍历 callable_signature 的参数字典 items，其中包含参数名和对应的 Parameter 对象。
        for name, param in callable_signature.parameters.items():
            # 如果参数类型不是 POSITIONAL_OR_KEYWORD，即不是位置参数或关键字参数，跳过当前循环。
            if not param.kind == param.POSITIONAL_OR_KEYWORD:
                continue
    
            # 将参数名 name 添加到 argument_names 列表中。
            argument_names.append(name)
    
        # 返回最终的参数名列表。
        return argument_names
# 将 AST 节点中的类型注解转换为源代码中表示相同注解的字符串
def get_annotation_str(annotation):
    if isinstance(annotation, ast.Name):
        return annotation.id  # 返回节点的标识符
    elif isinstance(annotation, ast.Attribute):
        return ".".join([get_annotation_str(annotation.value), annotation.attr])  # 返回属性节点的完整路径
    elif isinstance(annotation, ast.Subscript):
        # 对于 Python3.9+，订阅索引不再包装在 ast.Index 中
        subscript_slice = annotation.slice if IS_PY39_PLUS else annotation.slice.value  # 获取订阅的切片表达式
        return f"{get_annotation_str(annotation.value)}[{get_annotation_str(subscript_slice)}]"  # 返回订阅的完整表示
    elif isinstance(annotation, ast.Tuple):
        return ",".join([get_annotation_str(elt) for elt in annotation.elts])  # 返回元组的完整表示
    elif isinstance(annotation, ast.Constant):
        return f"{annotation.value}"  # 返回常量节点的值

    # 如果 AST 节点类型不在处理范围内，则可能由 ScriptTypeParser 处理
    return None


# 获取函数 'fn' 上的类型注解映射
def get_type_hint_captures(fn):
    """
    Get a dictionary containing type resolution mappings necessary to resolve types
    for the literal annotations on 'fn'. These are not considered to be closed-over by fn
    and must be obtained separately (e.g. using this function).

    Args:
        fn: A callable.
    Returns:
        A Dict[str, Any] containing a mapping from the literal annotations used on
        fn to the Python objects they refer to.
    """
    # 首先尝试获取函数的源代码。我们需要解析源代码以找到实际的字符串名称，
    # 用于注解类型，因为 inspect.signature() 只会返回注解引用的类对象，而不是字符串名称。
    # 如果无法获取源代码，则返回空字典。在函数在运行时动态合成的情况下可能会发生这种情况。
    src = loader.get_source(fn)
    if src is None:
        try:
            src = inspect.getsource(fn)  # 尝试使用 inspect.getsource 获取源代码
        except OSError as e:
            raise OSError(
                f"Failed to get source for {fn} using inspect.getsource"
            ) from e

    # 收集参数名到类型的映射字典，跳过那些注解类型为字符串的参数。
    # 这些类型只能在 TorchScript 的类型注解中理解，它们指代的是在其定义中的类，
    # 但尝试在结果函数中包含这种映射会导致无限递归，因为正在编译该类。
    # 此外，ScriptTypeParser 中有处理这种情况的逻辑。
    signature = inspect.signature(fn)
    name_to_type = {
        name: parameter.annotation
        for name, parameter in signature.parameters.items()
        if parameter.annotation is not inspect.Parameter.empty
        and not isinstance(parameter.annotation, str)
    }

    # 然后，从函数声明中获取文字类型注解
    # 使用 ast 模块解析源代码（src）并进行缩进处理，以确保准确性。
    a = ast.parse(textwrap.dedent(src))

    # 确保解析后的 AST 中只包含一个函数定义，并将其赋给变量 f。
    if len(a.body) != 1 or not isinstance(a.body[0], ast.FunctionDef):
        raise RuntimeError(f"Expected {fn} to be a function")
    f = a.body[0]

    # 准备一个字典 annotation_to_type，用于存储参数的源注解与其对应的类型。
    annotation_to_type = {}

    # 遍历函数 f 的参数列表 f.args.args。
    for arg in f.args.args:
        # 获取参数的源类型注解字符串（如果存在的话）。
        arg_annotation_str = (
            get_annotation_str(arg.annotation) if arg.annotation else None
        )

        # 如果参数没有注解或者无法转换成字符串，则跳过该参数。
        if arg_annotation_str is None:
            continue

        # 将 {arg_annotation_str: type} 插入 annotation_to_type 中，如果参数名 arg_name 在 name_to_type 中存在。
        arg_name = arg.arg
        if arg_name in name_to_type:
            annotation_to_type[arg_annotation_str] = name_to_type[arg_name]

    # 如果存在有效的返回注解，也将其包含在 annotation_to_type 中。
    literal_return_annotation = get_annotation_str(f.returns)
    valid_literal_annotation = literal_return_annotation is not None
    return_annotation = signature.return_annotation
    valid_return_annotation_type = (
        return_annotation is not inspect.Parameter.empty
        and not isinstance(return_annotation, str)
    )
    if valid_literal_annotation and valid_return_annotation_type:
        annotation_to_type[literal_return_annotation] = return_annotation

    # 返回最终的 annotation_to_type 字典，其中包含了所有参数的源注解与类型的映射关系。
    return annotation_to_type
# 创建一个解析类方法的闭包变量的回调函数
def createResolutionCallbackForClassMethods(cls):
    """
    This looks at all the methods defined in a class and pulls their closed-over
    variables into a dictionary and uses that to resolve variables.
    """
    # 获取所有在类中定义的方法，存放在列表 fns 中
    fns = [
        getattr(cls, name)
        for name in cls.__dict__  # 遍历类的字典属性
        if inspect.isroutine(getattr(cls, name))  # 筛选出是方法的属性
    ]
    # 排除内建方法和没有全局作用域或类型提示的方法，以支持 Python 3.11 中的 enum.Enum 派生类
    fns = [fn for fn in fns if not inspect.isbuiltin(fn) and hasattr(fn, "__globals__")]
    # 初始化一个空字典，用于收集闭包变量
    captures = {}

    # 遍历所有方法，更新 captures 字典，包括获取闭包和类型提示的闭包变量
    for fn in fns:
        captures.update(get_closure(fn))
        captures.update(get_type_hint_captures(fn))

    # 定义一个查找函数，用于查找类中的变量
    def lookup_in_class(key):
        if key in captures:
            return captures[key]
        else:
            return getattr(builtins, key, None)

    return lookup_in_class


def boolean_dispatch(
    arg_name,
    arg_index,
    default,
    if_true,
    if_false,
    module_name,
    func_name,
):
    """
    Dispatches to either of 2 script functions based on a boolean argument.
    In TorchScript, the boolean argument must be constant so that the correct
    function to use can be determined at compile time.
    """

    # 定义一个函数 fn，根据布尔参数调度到两个脚本函数中的一个
    def fn(*args, **kwargs):
        # 默认使用 default 参数
        dispatch_flag = default
        # 如果关键字参数中存在 arg_name，使用其值作为 dispatch_flag
        if arg_name in kwargs:
            dispatch_flag = kwargs[arg_name]
        # 否则使用位置参数中的第 arg_index 个参数作为 dispatch_flag
        elif arg_index < len(args):
            dispatch_flag = args[arg_index]

        # 根据 dispatch_flag 调用对应的函数
        if dispatch_flag:
            return if_true(*args, **kwargs)
        else:
            return if_false(*args, **kwargs)

    # 设置 fn 函数的文档字符串，根据 if_true 和 if_false 函数的情况决定
    if if_true.__doc__ is None and if_false.__doc__ is not None:
        doc = if_false.__doc__
        if_true.__doc__ = doc
    elif if_false.__doc__ is None and if_true.__doc__ is not None:
        doc = if_true.__doc__
        if_false.__doc__ = doc
    elif if_false.__doc__ is None and if_true.__doc__ is None:
        # 如果两个函数都没有文档字符串，则设置为 None
        doc = None
    else:
        raise RuntimeError("only one function can have a docstring")
    fn.__doc__ = doc

    # 设置 fn 函数的模块名和函数名，如果 module_name 或 func_name 不为 None 的话
    if module_name is not None:
        fn.__module__ = module_name
    if func_name is not None:
        fn.__name__ = func_name

    # 将 fn 函数及其相关信息添加到 boolean_dispatched 字典中
    boolean_dispatched[fn] = {
        "if_true": if_true,
        "if_false": if_false,
        "index": arg_index,
        "default": default,
        "arg_name": arg_name,
    }
    return fn


class FunctionModifiers:
    """
    Used to denote the behavior of a function in TorchScript. See export() and
    ignore() for details.
    """

    # 枚举类，用于描述在 TorchScript 中函数的行为
    UNUSED = "unused (ignored and replaced with raising of an exception)"
    IGNORE = "ignore (leave as a call to Python, cannot be torch.jit.save'd)"
    EXPORT = "export (compile this function even if nothing calls it)"
    # 定义默认说明文本，用于导出函数或前向调用时编译
    DEFAULT = "default (compile if called from a exported function / forward)"
    
    # 如果此方法未被脚本化，将 Python 方法复制到脚本化模型中
    COPY_TO_SCRIPT_WRAPPER = (
        "if this method is not scripted, copy the python method onto the scripted model"
    )
    
    # _drop（函数完全被忽略，声明可能无法被脚本化）
    _DROP = "_drop (function is fully ignored, declaration can be unscriptable)"
# 定义一个装饰器函数 `export`，用于指示一个方法在 `nn.Module` 中作为编译的入口点，
# 应当被编译成一个 `ScriptModule`。

"""
This decorator indicates that a method on an ``nn.Module`` is used as an entry point into a
:class:`ScriptModule` and should be compiled.

``forward`` implicitly is assumed to be an entry point, so it does not need this decorator.
Functions and methods called from ``forward`` are compiled as they are seen
by the compiler, so they do not need this decorator either.

Example (using ``@torch.jit.export`` on a method):

.. testcode::

    import torch
    import torch.nn as nn

    class MyModule(nn.Module):
        def implicitly_compiled_method(self, x):
            return x + 99

        # `forward` is implicitly decorated with `@torch.jit.export`,
        # so adding it here would have no effect
        def forward(self, x):
            return x + 10

        @torch.jit.export
        def another_forward(self, x):
            # When the compiler sees this call, it will compile
            # `implicitly_compiled_method`
            return self.implicitly_compiled_method(x)

        def unused_method(self, x):
            return x - 20

    # `m` will contain compiled methods:
    #     `forward`
    #     `another_forward`
    #     `implicitly_compiled_method`
    # `unused_method` will not be compiled since it was not called from
    # any compiled methods and wasn't decorated with `@torch.jit.export`
    m = torch.jit.script(MyModule())
"""

def export(fn):
    # 设置函数的 `_torchscript_modifier` 属性为 `FunctionModifiers.EXPORT`
    fn._torchscript_modifier = FunctionModifiers.EXPORT
    return fn


# 定义一个装饰器函数 `unused`，用于告知编译器一个函数或方法应当被忽略，
# 并且在调用时会触发异常。这允许您在模型中保留尚不兼容 TorchScript 的代码，
# 同时可以导出您的模型。

"""
This decorator indicates to the compiler that a function or method should
be ignored and replaced with the raising of an exception. This allows you
to leave code in your model that is not yet TorchScript compatible and still
export your model.

    Example (using ``@torch.jit.unused`` on a method)::

        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            def __init__(self, use_memory_efficient):
                super().__init__()
                self.use_memory_efficient = use_memory_efficient

            @torch.jit.unused
            def memory_efficient(self, x):
                import pdb
                pdb.set_trace()
                return x + 10

            def forward(self, x):
                # Use not-yet-scriptable memory efficient mode
                if self.use_memory_efficient:
                    return self.memory_efficient(x)
                else:
                    return x + 10

        m = torch.jit.script(MyModule(use_memory_efficient=False))
        m.save("m.pt")

        m = torch.jit.script(MyModule(use_memory_efficient=True))
        # exception raised
        m(torch.rand(100))
"""

def unused(fn):
    pass
    # 如果 fn 是 property 对象，则进行处理
    if isinstance(fn, property):
        # 将 fn 赋给 prop 变量
        prop = fn
        # 设置 prop.fget 方法的 "_torchscript_modifier" 属性为 UNUSED
        setattr(
            prop.fget, "_torchscript_modifier", FunctionModifiers.UNUSED
        )

        # 如果 prop 对象有 fset 方法，则设置其 "_torchscript_modifier" 属性为 UNUSED
        if prop.fset:
            setattr(
                prop.fset, "_torchscript_modifier", FunctionModifiers.UNUSED
            )

        # 返回处理后的 property 对象 prop
        return prop

    # 如果 fn 不是 property 对象，则直接设置其 "_torchscript_modifier" 属性为 UNUSED
    fn._torchscript_modifier = FunctionModifiers.UNUSED
    # 返回处理后的 fn 函数或方法
    return fn
# No op context manager from python side
# 定义一个空操作的上下文管理器类，继承自抽象基类 contextlib.AbstractContextManager
class _IgnoreContextManager(contextlib.AbstractContextManager):
    def __init__(self, **kwargs):
        # 初始化方法，不执行任何操作
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        # 上下文管理器的退出方法，不执行任何操作
        pass


def ignore(drop=False, **kwargs):
    """
    This decorator indicates to the compiler that a function or method should
    be ignored and left as a Python function. This allows you to leave code in
    your model that is not yet TorchScript compatible. If called from TorchScript,
    ignored functions will dispatch the call to the Python interpreter. Models with ignored
    functions cannot be exported; use :func:`@torch.jit.unused <torch.jit.unused>` instead.

    Example (using ``@torch.jit.ignore`` on a method)::

        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            @torch.jit.ignore
            def debugger(self, x):
                import pdb
                pdb.set_trace()

            def forward(self, x):
                x += 10
                # The compiler would normally try to compile `debugger`,
                # but since it is `@ignore`d, it will be left as a call
                # to Python
                self.debugger(x)
                return x

        m = torch.jit.script(MyModule())

        # Error! The call `debugger` cannot be saved since it calls into Python
        m.save("m.pt")

    Example (using ``@torch.jit.ignore(drop=True)`` on a method):

    .. testcode::

        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            @torch.jit.ignore(drop=True)
            def training_method(self, x):
                import pdb
                pdb.set_trace()

            def forward(self, x):
                if self.training:
                    self.training_method(x)
                return x

        m = torch.jit.script(MyModule())

        # This is OK since `training_method` is not saved, the call is replaced
        # with a `raise`.
        m.save("m.pt")

    .. testcleanup::

        import os
        os.remove('m.pt')
    """

    if callable(drop):
        # 如果 drop 是可调用对象，则认为是函数直接作为装饰器使用
        fn = drop
        fn._torchscript_modifier = FunctionModifiers.IGNORE
        return fn

    if not isinstance(drop, bool):
        # 如果 drop 不是布尔值，则抛出运行时错误
        raise RuntimeError(
            "Argument to @torch.jit.ignore must be a bool or "
            f"a function but got {drop}"
        )

    # for backwards compat
    # 用于向后兼容，处理 drop_on_export 参数
    drop_on_export = kwargs.pop("drop_on_export", None)
    if drop_on_export:
        # 发出警告，表明 drop_on_export=True 已被弃用
        warnings.warn(
            "ignore(drop_on_export=True) has been deprecated. TorchScript will now drop the function "
            "call on compilation. Use torch.jit.unused now. {}",
            category=FutureWarning,
        )

        drop = drop_on_export
    elif drop:
        # 如果 drop 参数为真，则发出警告信息，说明 ignore(True) 已经被废弃
        warnings.warn(
            "ignore(True) has been deprecated. TorchScript will now drop the function "
            "call on compilation. Use torch.jit.unused now. {}",
            category=FutureWarning,
        )

    # 定义装饰器函数 decorator，接受一个函数 fn 作为参数
    def decorator(fn):
        # 如果 drop 参数为真，将函数 fn 标记为未使用状态（UNUSED）
        if drop:
            fn._torchscript_modifier = FunctionModifiers.UNUSED
        else:
            # 否则，将函数 fn 标记为忽略状态（IGNORE）
            fn._torchscript_modifier = FunctionModifiers.IGNORE
        # 返回被修饰后的函数 fn
        return fn

    # 返回定义好的 decorator 函数
    return decorator
# 将函数标记为 _torchscript_modifier 为 FunctionModifiers._DROP
def _drop(fn):
    fn._torchscript_modifier = FunctionModifiers._DROP
    return fn


# 将函数标记为 _torchscript_modifier 为 FunctionModifiers.COPY_TO_SCRIPT_WRAPPER
def _copy_to_script_wrapper(fn):
    fn._torchscript_modifier = FunctionModifiers.COPY_TO_SCRIPT_WRAPPER
    return fn


# 检查模块是否有导出函数
def module_has_exports(mod):
    for name in dir(mod):
        if hasattr(mod, name):
            item = getattr(mod, name)
            if callable(item):
                # 检查函数是否被标记为 FunctionModifiers.EXPORT
                if get_torchscript_modifier(item) is FunctionModifiers.EXPORT:
                    return True
    return False


# 检查函数是否应该被丢弃
# 如果函数的 _torchscript_modifier 标记为 FunctionModifiers.UNUSED 或 FunctionModifiers._DROP，则返回 True
def should_drop(fn) -> bool:
    attr = get_torchscript_modifier(fn)
    if attr is None:
        return False
    return attr is FunctionModifiers.UNUSED or attr is FunctionModifiers._DROP


# 检查函数是否被忽略
# 如果函数的 _torchscript_modifier 标记为 FunctionModifiers.UNUSED, FunctionModifiers.IGNORE 或 FunctionModifiers._DROP，则返回 True
def is_ignored_fn(fn) -> bool:
    mod = get_torchscript_modifier(fn)
    return (
        mod is FunctionModifiers.UNUSED
        or mod is FunctionModifiers.IGNORE
        or mod is FunctionModifiers._DROP
    )


# 检查函数是否被标记为 _DROP
def _is_drop_fn(fn) -> bool:
    mod = get_torchscript_modifier(fn)
    return mod is FunctionModifiers._DROP


# 检查函数是否是类的静态方法
def is_static_fn(cls, fn) -> bool:
    return isinstance(inspect.getattr_static(cls, fn, default=None), staticmethod)


# 获取类的静态方法
def get_static_fn(cls, fn):
    return inspect.getattr_static(cls, fn).__func__


# 获取函数的 _torchscript_modifier 标记
# 如果函数是方法，则获取 __func__ 属性的 _torchscript_modifier 标记
def get_torchscript_modifier(fn):
    if not callable(fn):
        return None
    if hasattr(fn, "__func__"):
        fn = fn.__func__
    return getattr(fn, "_torchscript_modifier", FunctionModifiers.DEFAULT)


# 复制函数的 _torchscript_modifier 标记
def copy_torchscript_modifier(orig, new) -> None:
    attr = get_torchscript_modifier(orig)
    if attr is None:
        return
    new._torchscript_modifier = attr


# 重载函数的注册说明
# 重载函数在此文件中注册，然后在 torch/jit/__init__.py 中编译，
# 以便可以在 nn/functional.py 中导入而不会产生循环导入
_overloaded_fns: Dict[str, List[Callable]] = {}  # noqa: T484


# 超载函数的示例用法
_OVERLOAD_EXAMPLE = """
Example usage of overload function:
@torch.jit._overload
def my_function(x: type0) -> type0: # decl 1
    pass

@torch.jit._overload
def my_function(x: type1) -> type1: # decl 2
    pass

def my_function(x):                 # implementation
    if isinstance(x, type0):
        return x
    elif isinstance(x, type1):
        return x
"""


# 获取超载函数未实现错误消息
def get_overload_no_implementation_error_message(kind, obj):
    sourcelines, file_lineno, filename = get_source_lines_and_file(obj)
    return (
        f'Implementation for the {kind} "{_qualified_name(obj)}" is missing. Please make '
        f"sure a definition is provided and defined after all overload declarations.\n"
        f'File "{filename}", line {file_lineno}:\n'
        + "".join(sourcelines)
        + "\n"
        + _OVERLOAD_EXAMPLE
    )
# 检查函数体是否符合重载要求
def _check_overload_body(func):
    try:
        # 解析函数定义，获取抽象语法树
        parsed_def = parse_def(func)
    except OSError as e:
        # 如果解析函数定义时出现 OSError，说明源代码不可用
        # 由于这只是初始检查，如果出现这种情况，只需发出警告
        warnings.warn(
            f"Unable to retrieve source for @torch.jit._overload function: {func}."
        )
        return

    # 获取函数体的语句列表
    body = parsed_def.ast.body[0].body

    # 判断是否为 pass 语句
    def is_pass(x):
        return isinstance(x, ast.Pass)

    # 判断是否为 ... 表达式
    def is_ellipsis(x):
        return (
            isinstance(x, ast.Expr)
            and isinstance(x.value, ast.Constant)
            and x.value.value is Ellipsis
        )

    # 如果函数体语句不是单一的 pass 语句或者 ... 表达式，则抛出 RuntimeError
    if len(body) != 1 or not (is_pass(body[0]) or is_ellipsis(body[0])):
        msg = (
            "Only `pass` statement or `...` can be the body of overload declaration:\n"
        )
        # 构造错误信息，包含函数定义前三行源代码和提示信息
        msg += "\n".join(parsed_def.source.split("\n")[:3])
        msg += " <- Expecting `pass` or `...` here!\n" + _OVERLOAD_EXAMPLE
        raise RuntimeError(msg)


# 将函数注册为重载函数
def _overload(func):
    # 检查函数体是否符合重载要求
    _check_overload_body(func)
    # 获取函数的限定名
    qual_name = _qualified_name(func)
    # 获取全局变量中的重载函数列表
    global _overloaded_fns
    fn_overload_list = _overloaded_fns.get(qual_name)
    # 如果列表不存在，则创建空列表并存入全局变量中
    if fn_overload_list is None:
        fn_overload_list = []
        _overloaded_fns[qual_name] = fn_overload_list
    # 将函数添加到重载函数列表中
    fn_overload_list.append(func)
    # 返回原始函数对象
    return func


# 获取给定限定名的重载函数列表
def _get_fn_overloads(qual_name):
    return _overloaded_fns.get(qual_name)


# 清除指定限定名下的重载函数列表
def _clear_fn_overloads(qual_name) -> None:
    del _overloaded_fns[qual_name]


# 获取方法所属类的名称和定义起始行号
def get_class_name_lineno(method) -> Tuple[str, int]:
    # 获取当前调用栈帧
    current_frame = inspect.currentframe()

    # 由于栈帧嵌套，依次向上遍历两次
    for i in range(2):
        assert (
            current_frame is not None
        )  # 确保当前栈帧不为 Optional[FrameType]
        current_frame = current_frame.f_back

    # 再次确保当前栈帧不为 None
    assert current_frame is not None
    # 获取当前栈帧所在函数的名称
    class_name = current_frame.f_code.co_name
    # 获取当前栈帧所在函数的起始行号
    line_no = current_frame.f_code.co_firstlineno
    return class_name, line_no


# 在应用装饰器到类方法时，方法尚未引用其所属类
# _qualified_name 不会包含它所定义的类，因此同一文件中相同名称的方法
# 会具有相同的 _qualified_name，即使它们定义在不同的类中。这个问题仅存在于 Python 2 中。
# 我们通过查看堆栈帧并标识类名来解决此问题，并在使用重载时在同一文件中具有相同名称的模块时抛出错误。
# qualified_name => class name => list[overload_functions]
_overloaded_methods: Dict[str, Dict[str, List[Callable]]] = {}  # noqa: T484


# (qualified_name, class name) => class_fileno
_overloaded_method_class_fileno: Dict[Tuple[str, str], int] = {}


# 检查方法体是否符合重载要求
def _overload_method(func):
    _check_overload_body(func)
    # 获取函数的限定名
    qual_name = _qualified_name(func)
    # 获取全局变量 _overloaded_methods 中 qual_name 对应的类名映射表，如果不存在则创建一个空字典
    class_name_map = _overloaded_methods.get(qual_name, None)
    if class_name_map is None:
        # 如果映射表不存在，则创建一个空字典
        class_name_map = {}
        # 将新创建的空字典存入 _overloaded_methods 中对应的 qual_name
        _overloaded_methods[qual_name] = class_name_map

    # 获取函数 func 所在的类名和行号
    class_name, line_no = get_class_name_lineno(func)
    # 获取 class_name_map 中 class_name 对应的方法重载列表，如果不存在则创建一个空列表
    method_overloads = class_name_map.get(class_name, None)
    if method_overloads is None:
        # 如果方法重载列表不存在，则创建一个空列表
        method_overloads = []
        # 将新创建的空列表存入 class_name_map 中对应的 class_name
        class_name_map[class_name] = method_overloads
        # 将 (qual_name, class_name) 对应的行号存入 _overloaded_method_class_fileno
        _overloaded_method_class_fileno[(qual_name, class_name)] = line_no
    else:
        # 如果方法重载列表已存在，则检查已存储的行号是否与当前行号一致，不一致则抛出异常
        existing_lineno = _overloaded_method_class_fileno[(qual_name, class_name)]
        if existing_lineno != line_no:
            raise RuntimeError(
                "Cannot currently overload the same method name in two different"
                " classes with the same name in the same module"
            )

    # 将当前函数 func 添加到 method_overloads 中，表示重载了该方法
    method_overloads.append(func)
    # 返回被重载的函数 func
    return func
# 检查给定方法是否在递归脚本中设置了 __name__ 属性，如果没有则返回 None
def _get_overloaded_methods(method, mod_class):
    if not hasattr(method, "__name__"):
        return None
    # 获取方法的限定名称
    qual_name = _qualified_name(method)
    # 获取给定方法的重载映射
    class_name_map = _overloaded_methods.get(qual_name, None)
    if class_name_map is None:
        return None
    # 获取模块类的名称对应的重载方法
    overloads = class_name_map.get(mod_class.__name__, None)
    if overloads is None:
        return None

    # 获取方法的起始行号和模块类的起始行号以及结束行号
    method_line_no = get_source_lines_and_file(method)[1]
    mod_class_fileno = get_source_lines_and_file(mod_class)[1]
    mod_end_fileno = mod_class_fileno + len(get_source_lines_and_file(mod_class)[0])
    # 检查方法是否在模块类定义的范围内，否则抛出断言错误
    if not (method_line_no >= mod_class_fileno and method_line_no <= mod_end_fileno):
        raise AssertionError(
            "Overloads are not useable when a module is redeclared within the same file: "
            + str(method)
        )
    return overloads


# 检查注解是否为元组类型
def is_tuple(ann) -> bool:
    # 如果注解直接为 Tuple，则引发异常
    if ann is Tuple:
        raise_error_container_parameter_missing("Tuple")

    # 如果注解没有 __module__ 属性，则返回 False
    if not hasattr(ann, "__module__"):
        return False

    # 获取注解的原始类型
    ann_origin = get_origin(ann)
    # 如果是 Python 3.9+ 并且注解来自于 builtins 并且是 tuple 类型，则返回 True
    if IS_PY39_PLUS and ann.__module__ == "builtins" and ann_origin is tuple:
        return True
    # 否则，如果注解来自于 typing 并且是 Tuple 类型，则返回 True
    return ann.__module__ == "typing" and (ann_origin is Tuple or ann_origin is tuple)


# 检查注解是否为列表类型
def is_list(ann) -> bool:
    if ann is List:
        raise_error_container_parameter_missing("List")

    if not hasattr(ann, "__module__"):
        return False

    ann_origin = get_origin(ann)
    if IS_PY39_PLUS and ann.__module__ == "builtins" and ann_origin is list:
        return True
    return ann.__module__ == "typing" and (ann_origin is List or ann_origin is list)


# 检查注解是否为字典类型
def is_dict(ann) -> bool:
    if ann is Dict:
        raise_error_container_parameter_missing("Dict")

    if not hasattr(ann, "__module__"):
        return False

    ann_origin = get_origin(ann)
    if IS_PY39_PLUS and ann.__module__ == "builtins" and ann_origin is dict:
        return True
    return ann.__module__ == "typing" and (ann_origin is Dict or ann_origin is dict)


# 检查注解是否为联合类型
def is_union(ann):
    if ann is Union:
        raise_error_container_parameter_missing("Union")

    # 判断注解是否为内置联合类型或者来自 typing 模块的 Union 类型
    return isinstance(ann, BuiltinUnionType) or (
        hasattr(ann, "__module__")
        and ann.__module__ == "typing"
        and (get_origin(ann) is Union)
    )


# 检查注解是否为可选类型
def is_optional(ann):
    if ann is Optional:
        raise_error_container_parameter_missing("Optional")

    # 判断注解是否为标准库的 Optional 或者是一个联合类型中包含 None 类型的情况
    def is_optional_as_optional(ann):
        return (
            hasattr(ann, "__module__")
            and ann.__module__ == "typing"
            and (get_origin(ann) is Optional)
        )

    def is_union_as_optional(ann):
        ann_args = get_args(ann)
        return len(ann_args) == 2 and (None in ann_args or type(None) in ann_args)

    return is_optional_as_optional(ann) or (is_union(ann) and is_union_as_optional(ann))


# 检查注解是否为 Future 类型
def is_future(ann) -> bool:
    # 此处代码未完整，可能需要补充
    # 如果注释变量 ann 的类型为 Future 类型（假设 Future 是一个类型），则抛出运行时错误
    if ann is Future:
        # 抛出错误，提示未包含类型信息的 Future 使用情况
        raise RuntimeError(
            "Attempted to use Future without a "
            "contained type. Please add a contained type, e.g. "
            "Future[int]"
        )
    # 返回注释变量 ann 的原始类型是否为 Future
    return get_origin(ann) is Future
# 判断给定的注解是否是 _Await 类型
def is_await(ann) -> bool:
    if ann is _Await:
        return True
    return get_origin(ann) is _Await

# 如果 torch.distributed.rpc 可用，则导入相应的模块
if torch.distributed.rpc.is_available():
    from torch._C._distributed_rpc import PyRRef
    from torch.distributed.rpc import RRef

    # 判断给定的注解是否是 RRef 类型，如果是则抛出 RuntimeError 异常
    def is_rref(ann) -> bool:
        if ann is RRef:
            raise RuntimeError(
                "Attempted to use RRef without a "
                "contained type. Please add a contained type, e.g. "
                "RRef[int]"
            )
        return get_origin(ann) is RRef

    # 判断给定的对象是否是 PyRRef 类型的实例
    def is_rref_instance(obj) -> bool:
        return isinstance(obj, PyRRef)

else:
    # 如果 torch.distributed.rpc 不可用，则定义一个函数返回 False
    def is_rref_instance(obj) -> bool:
        # If the RPC module doesn't exist then RRefs don't exist either.
        return False

# 判断给定的注解是否是 Final 类型或者 Final 类型的实例
def is_final(ann) -> bool:
    return (
        hasattr(ann, "__module__")
        and ann.__module__ in {"typing", "typing_extensions"}
        and (get_origin(ann) is Final or isinstance(ann, type(Final)))
    )

# 定义一个允许 BroadcastingList 实例进行下标访问的类
class BroadcastingListCls:
    def __getitem__(self, types):
        return

# 创建多个名为 BroadcastingList2 到 BroadcastingList6 的全局实例
BroadcastingList1 = BroadcastingListCls()
for i in range(2, 7):
    globals()[f"BroadcastingList{i}"] = BroadcastingList1

# 判断当前是否处于 TorchScript 编译状态，始终返回 False
def is_scripting() -> bool:
    r"""
    Function that returns True when in compilation and False otherwise. This
    is useful especially with the @unused decorator to leave code in your
    model that is not yet TorchScript compatible.
    .. testcode::

        import torch

        @torch.jit.unused
        def unsupported_linear_op(x):
            return x

        def linear(x):
           if torch.jit.is_scripting():
              return torch.linear(x)
           else:
              return unsupported_linear_op(x)
    """
    return False

# 获取给定对象的完全限定名（模块层次结构 + 类名）
def _qualified_name(obj, mangle_name=True) -> str:
    # 处理特殊情况，允许在类型上覆盖限定名
    # 主要用于 TorchScript 追踪，需要正确设置限定名以便在系统中显示
    if hasattr(obj, "_jit_override_qualname"):
        return obj._jit_override_qualname
    # 对于已知的特定对象类型，直接返回其限定名
    if isinstance(obj, torch._C.ScriptFunction):
        return obj.qualified_name

    if getattr(obj, "__name__", None):
        name = obj.__name__
    # 对于枚举类，使用 name 属性作为名称
    elif isinstance(obj, enum.Enum):
        name = obj.name
    else:
        raise RuntimeError("Could not get name of python class object")
    # 如果无法获取 Python 类对象的名称，则抛出运行时错误

    if name == "<lambda>":
        name = "_lambda"  # make name a valid identifier
        # 如果名称为 "<lambda>"，则将其改为 "_lambda"，使其成为有效的标识符

    module_name = obj.__module__

    # If the module is actually a torchbind module, then we should short circuit
    # 如果模块实际上是 torchbind 模块，则应该立即返回其完整限定名
    if module_name == "torch._classes":
        return obj.qualified_name

    # The Python docs are very clear that `__module__` can be None, but I can't
    # figure out when it actually would be.
    # Python 文档明确指出 `__module__` 可能为 None，但我无法确定什么情况下会是 None
    if module_name is None:
        raise RuntimeError(
            f"Could not get qualified name for class '{name}': "
            "__module__ can't be None."
        )

    # if getattr(sys.modules[module_name], name) is not obj:
    #     raise RuntimeError(f"Could not get qualified name for class '{name}': "
    #                        f"the attr {name} on module {module_name} is not the class")

    # torch.package and TorchScript have separate mangling schemes to avoid
    # name collisions from multiple packages. To avoid them interfering with
    # each other, normalize the package manging here.
    # torch.package 和 TorchScript 有单独的名称混淆方案，以避免来自多个包的名称冲突。
    # 为避免它们相互干扰，这里对包管理进行标准化处理。

    if package_mangling.is_mangled(module_name):
        module_name = module_name.replace("<", "_")
        module_name = module_name.replace(">", "_")
        # 如果模块名经过名称混淆，则替换 "<" 和 ">" 以避免干扰

    # The PythonExceptionValue C++ class in torch/csrc/jit/python/python_sugared_value.h
    # does not need mangle the python class name.
    # torch/csrc/jit/python/python_sugared_value.h 中的 PythonExceptionValue C++ 类
    # 不需要混淆 Python 类名。

    if mangle_name:
        # __main__ is a builtin module, so rewrite it to "__torch__".
        # __main__ 是一个内置模块，因此将其重写为 "__torch__"。
        if module_name == "__main__":
            module_name = "__torch__"
        else:
            # Everything else gets a "__torch__" prefix to avoid name collisions
            # with the names of user values.
            # 其它模块名加上 "__torch__" 前缀，以避免与用户值的名称冲突。
            module_name = "__torch__." + module_name

    if "." in name:
        raise RuntimeError(
            f"Could not get qualified name for class '{name}': "
            f"'{name}' is not a valid identifier"
        )
        # 如果名称中包含点号，则抛出运行时错误，因为它不是有效的标识符。

    return module_name + "." + name
    # 返回模块名和类名的完整限定名
# 尝试获取与给定函数相关联的分派函数，如果函数不可调用则返回 None
def _try_get_dispatched_fn(fn):
    if not callable(fn):
        return None
    # 返回 fn 在 boolean_dispatched 字典中的关联值（如果存在）
    return boolean_dispatched.get(fn)


# 获取命名元组的属性信息
def _get_named_tuple_properties(
    obj,
    loc: Optional[torch._C._jit_tree_views.SourceRange] = None,
    rcb=None,
):
    # 如果未提供 loc 参数，则使用 fake_range() 函数生成一个虚拟范围对象
    if loc is None:
        loc = fake_range()

    # 断言 obj 是 tuple 的子类且具有 _fields 属性
    assert issubclass(obj, tuple) and hasattr(obj, "_fields")

    # 如果 obj 具有 _field_defaults 属性，则获取默认值列表
    if hasattr(obj, "_field_defaults"):
        defaults = [
            obj._field_defaults[field]
            for field in obj._fields
            if field in obj._field_defaults
        ]
    else:
        defaults = []

    # 在 Python 3.10 中推荐使用 inspect.get_annotations 函数来获取注解信息
    # 此外，需要显式查询基类以获取未继承的注解
    if sys.version_info[:2] < (3, 10):
        # 获取 obj 的注解，如果不存在则使用空字典
        obj_annotations = getattr(obj, "__annotations__", {})
    else:
        # 使用 inspect.get_annotations 获取 obj 的注解信息
        obj_annotations = inspect.get_annotations(obj)
        # 如果注解信息为空且 obj 具有 __base__ 属性，则从基类获取注解
        if len(obj_annotations) == 0 and hasattr(obj, "__base__"):
            obj_annotations = inspect.get_annotations(obj.__base__)

    # 初始化一个空的注解列表
    annotations = []
    # 迭代命名元组中定义的所有字段
    for field in obj._fields:
        # 检查字段是否在对象注释字典中
        if field in obj_annotations:
            # 获取字段的注释类型
            field_type = obj_annotations[field]
            # 检查注释类型是否为 ForwardRef 类型，并且确保 rcb 不为空
            if isinstance(field_type, ForwardRef) and rcb is not None:
                # 使用 rcb 解析 ForwardRef 类型，获取真实的类型
                rcb_type = rcb(field_type.__forward_arg__)
                # 如果 rcb 无法找到类型，则抛出 ValueError
                if rcb_type is None:
                    raise ValueError(
                        f"Unknown type annotation: '{field_type}' in NamedTuple {obj.__name__}."
                        f" Likely due to partial support for ForwardRef parameters in NamedTuples, see #95858."
                        f" Issue occurred at {loc.highlight()}"
                    )
                # 将解析后的类型赋值给字段类型
                field_type = rcb_type
            # 将字段类型转换为 TorchScript 可识别的类型
            the_type = torch.jit.annotations.ann_to_type(field_type, loc, rcb)
            # 将转换后的类型添加到注释列表中
            annotations.append(the_type)
        else:
            # 如果字段没有注释类型，则默认为推断为 Torch Tensor 类型
            annotations.append(torch._C.TensorType.getInferred())
    # 返回对象的类名
    return type(obj).__name__, 
    # 返回对象的字段名
    obj._fields, 
    # 返回对象的注解信息
    annotations, 
    # 返回对象的默认值信息
    defaults
# 创建命名元组的函数，根据给定的非限定名称、字段名列表和默认值元组创建一个命名元组类型
def _create_named_tuple(
    t,
    unqual_name: str,
    field_names: List[str],
    defaults: Tuple[Any, ...],
):
    # 使用 collections.namedtuple 创建命名元组类型
    TupleType = collections.namedtuple(unqual_name, field_names, defaults=defaults)  # type: ignore[call-arg, no-redef, misc]
    return TupleType(*t)


# 上下文管理器，用于临时禁用 Torch 的 JIT emit hooks
@contextlib.contextmanager
def _disable_emit_hooks():
    # 获取当前的 emit hooks
    hooks = torch._C._jit_get_emit_hooks()
    # 设置 emit hooks 为 None，禁用 emit hooks
    torch._C._jit_set_emit_hooks(None, None)
    try:
        yield  # 执行上下文管理器内的代码块
    finally:
        # 恢复之前的 emit hooks
        torch._C._jit_set_emit_hooks(hooks[0], hooks[1])


# 装饰器类，用于禁用 Torch 的 JIT emit hooks
def _disable_emit_hooks_decorator(_DecoratorContextManager) -> None:  # noqa: F811
    def __enter__(self) -> None:
        # 在进入上下文管理器时，保存当前的 emit hooks
        self.hooks = torch._C._jit_get_emit_hooks()
        # 设置 emit hooks 为 None，禁用 emit hooks
        torch._C._jit_set_emit_hooks(None, None)

    def __exit__(self, *args) -> None:
        # 在退出上下文管理器时，恢复之前的 emit hooks
        torch._C._jit_set_emit_hooks(self.hooks[0], self.hooks[1])


# 检查对象是否为异常类（Exception class）
def _is_exception(obj) -> bool:
    # 如果对象不是类，则返回 False
    if not inspect.isclass(obj):
        return False
    # 判断对象是否是异常类的子类
    return issubclass(obj, Exception)


# 抛出错误，指示容器参数缺失
def raise_error_container_parameter_missing(target_type) -> None:
    if target_type == "Dict":
        # 如果目标类型是字典但未指定键值类型，则抛出 RuntimeError
        raise RuntimeError(
            "Attempted to use Dict without "
            "contained types. Please add contained type, e.g. "
            "Dict[int, int]"
        )
    # 否则，抛出指示缺失类型参数的 RuntimeError
    raise RuntimeError(
        f"Attempted to use {target_type} without a "
        "contained type. Please add a contained type, e.g. "
        f"{target_type}[int]"
    )


# 检查是否存在必需的容器类型参数（List、Tuple、Dict、Optional）
def check_args_exist(target_type) -> None:
    if target_type is List or target_type is list:
        # 如果目标类型是列表，则抛出异常表示缺少类型参数
        raise_error_container_parameter_missing("List")
    elif target_type is Tuple or target_type is tuple:
        # 如果目标类型是元组，则抛出异常表示缺少类型参数
        raise_error_container_parameter_missing("Tuple")
    elif target_type is Dict or target_type is dict:
        # 如果目标类型是字典，则抛出异常表示缺少类型参数
        raise_error_container_parameter_missing("Dict")
    elif target_type is None or target_type is Optional:
        # 如果目标类型是 None 或 Optional，则抛出异常表示缺少类型参数
        raise_error_container_parameter_missing("Optional")


# 检查空容器（空列表、空字典、空元组），发出警告
def check_empty_containers(obj) -> None:
    if obj == [] or obj == {} or obj == ():
        warnings.warn(
            "The inner type of a container is lost when "
            "calling torch.jit.isinstance in eager mode. For "
            "example, List[int] would become list and "
            "therefore falsely return True for List[float] or"
            " List[str]."
        )


# 容器类型检查器，支持 List、Dict、Tuple 和 Optional 类型
# TODO 支持未来的类型
def container_checker(obj, target_type) -> bool:
    origin_type = get_origin(target_type)
    check_args_exist(target_type)
    if origin_type is None:
        return False
    elif origin_type is list or origin_type is List:
        # 检查空容器情况，确保对象不为空
        check_empty_containers(obj)
        # 如果对象不是列表类型，直接返回 False
        if not isinstance(obj, list):
            return False
        # 获取列表元素类型
        arg_type = get_args(target_type)[0]
        # 获取列表元素的原始类型（可能为 None）
        arg_origin = get_origin(arg_type)
        # 遍历列表中的每个元素
        for el in obj:
            # 检查是否为嵌套容器，例如 List[List[str]]
            if arg_origin:  # 处理嵌套容器，例如 List[List[str]]
                # 如果嵌套容器类型检查失败，则返回 False
                if not container_checker(el, arg_type):
                    return False
            elif not isinstance(el, arg_type):
                # 如果列表元素类型检查失败，则返回 False
                return False
        # 所有元素类型检查通过，返回 True
        return True
    elif origin_type is Dict or origin_type is dict:
        # 检查空容器情况，确保对象不为空
        check_empty_containers(obj)
        # 如果对象不是字典类型，直接返回 False
        if not isinstance(obj, dict):
            return False
        # 获取字典键和值的类型
        key_type = get_args(target_type)[0]
        val_type = get_args(target_type)[1]
        # 遍历字典的键值对
        for key, val in obj.items():
            # 检查字典键的类型是否正确
            if not isinstance(key, key_type):
                return False
            # 获取字典值的原始类型（可能为 None）
            val_origin = get_origin(val_type)
            # 如果值为嵌套容器类型，检查嵌套容器类型是否正确
            if val_origin:
                if not container_checker(val, val_type):
                    return False
            elif not isinstance(val, val_type):
                # 如果字典值类型检查失败，则返回 False
                return False
        # 所有键值对类型检查通过，返回 True
        return True
    elif origin_type is Tuple or origin_type is tuple:
        # 检查空容器情况，确保对象不为空
        check_empty_containers(obj)
        # 如果对象不是元组类型，直接返回 False
        if not isinstance(obj, tuple):
            return False
        # 获取元组中每个元素的类型
        arg_types = get_args(target_type)
        # 检查元组长度是否与类型定义匹配
        if len(obj) != len(arg_types):
            return False
        # 遍历元组中的每个元素和对应的类型
        for el, el_type in zip(obj, arg_types):
            # 获取元素类型的原始类型（可能为 None）
            el_origin = get_origin(el_type)
            # 如果元素为嵌套容器类型，检查嵌套容器类型是否正确
            if el_origin:
                if not container_checker(el, el_type):
                    return False
            elif not isinstance(el, el_type):
                # 如果元组元素类型检查失败，则返回 False
                return False
        # 所有元组元素类型检查通过，返回 True
        return True
    elif origin_type is Union or issubclass(
        origin_type, BuiltinUnionType
    ):  # 处理 Optional 类型
        # 如果对象为 None，直接返回 True，因为 None 总是有效的
        if obj is None:
            return True
        # 获取 Union 类型中的所有内部类型
        inner_types = get_args(target_type)
        # 遍历所有内部类型
        for t in inner_types:
            # 获取类型的原始类型（可能为 None）
            t_origin = get_origin(t)
            # 如果类型为嵌套容器类型，则递归检查对象是否匹配该类型
            if t_origin:
                return container_checker(obj, t)
            elif isinstance(obj, t):
                # 如果对象与类型匹配，则返回 True
                return True
    # 如果以上条件都不满足，则返回 False
    return False
# 检查对象是否是给定类型或类型元组中的一种
def _isinstance(obj, target_type) -> bool:
    # 如果目标类型是容器类型
    if isinstance(target_type, collections.abc.Container):
        # 如果目标类型不是元组，抛出运行时错误
        if not isinstance(target_type, tuple):
            raise RuntimeError(
                "The second argument to "
                "`torch.jit.isinstance` must be a type "
                "or a tuple of types"
            )
        # 遍历类型元组，递归检查对象是否属于其中任一类型
        for t_type in target_type:
            if _isinstance(obj, t_type):
                return True
        return False

    # 获取类型的原始类型（可能为 None）
    origin_type = get_origin(target_type)
    # 如果存在原始类型，则使用容器检查器进行检查
    if origin_type:
        return container_checker(obj, target_type)

    # 检查处理非类型化的可选原点返回作为“none”而不是可选的情况（在 3.7-3.8 中）
    check_args_exist(target_type)

    # 处理非容器类型的情况，直接使用 isinstance 进行类型检查
    return isinstance(obj, target_type)


class _TensorExtractor(pickle.Pickler):
    def __init__(self, *args, tensors: List[torch.Tensor], **kwargs):
        super().__init__(*args, **kwargs)
        self.tensors = tensors

    def persistent_id(self, obj):
        # 如果对象是 torch.Tensor 类型，则将其添加到 tensors 列表中
        if isinstance(obj, torch.Tensor):
            self.tensors.append(obj)
            return ""
        # 对于其他类型的对象，根据类型做特定处理以保持性能和安全性
        if isinstance(obj, LockType):
            return ""
        if isinstance(obj, CFuture) or is_rref_instance(obj):
            return ""
        if isinstance(obj, CAwait):
            return ""
        if isinstance(obj, torch.cuda.Event):
            return ""
        if isinstance(obj, threading.Thread):
            return ""
        # 返回 None 表示无需对该对象进行特殊处理
        return None


def _extract_tensors(obj):
    r"""
    该函数专门从 C++ 中调用。
    参见 ``torch/csrc/jit/python/python_ivalue.h``。

    通过 pickle 提取给定对象中包含的张量。
    """
    tensors: List[torch.Tensor] = []
    # 创建 _TensorExtractor 对象来提取对象中的张量
    extractor = _TensorExtractor(io.BytesIO(), protocol=-1, tensors=tensors)
    extractor.dump(obj)
    # 返回提取到的张量列表
    return tensors


# 在 Python-3.11+ 中，类型化的枚举（例如 IntEnum）保留了之前被删除的基类方法数量的行为。
# 为了保持这种行为，需要在这里显式删除它们。
if sys.version_info > (3, 10):
    _drop(enum.Enum.__new__)
    _drop(enum.Enum.__format__)
    _drop(enum.Enum.__repr__)
    _drop(enum.Enum.__str__)
```