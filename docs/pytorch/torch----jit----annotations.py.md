# `.\pytorch\torch\jit\annotations.py`

```
# 引入必要的模块和类型定义
# mypy: allow-untyped-defs 允许未类型化的函数定义
import ast  # AST 抽象语法树模块，用于解析 Python 代码
import builtins  # 内置函数和异常模块
import dis  # 字节码分析模块，用于分析 Python 字节码
import enum  # 枚举类型模块
import inspect  # 检查模块，用于获取对象信息
import re  # 正则表达式模块
import typing  # 类型注解模块
import warnings  # 警告模块

from textwrap import dedent  # 文本包装模块，用于去除文本开头的空格
from typing import Type  # 类型注解模块的 Type 类型

import torch  # PyTorch 深度学习库

from torch._C import (
    _GeneratorType,  # Torch C++ 扩展中的生成器类型
    AnyType,  # 任意类型
    AwaitType,  # await 对象类型
    BoolType,  # 布尔类型
    ComplexType,  # 复数类型
    DeviceObjType,  # 设备对象类型
    DictType,  # 字典类型
    EnumType,  # 枚举类型
    FloatType,  # 浮点数类型
    FutureType,  # Future 对象类型
    InterfaceType,  # 接口类型
    IntType,  # 整数类型
    ListType,  # 列表类型
    NoneType,  # None 类型
    NumberType,  # 数字类型
    OptionalType,  # 可选类型
    StreamObjType,  # 流对象类型
    StringType,  # 字符串类型
    TensorType,  # 张量类型
    TupleType,  # 元组类型
    UnionType,  # 联合类型
)

from torch._sources import get_source_lines_and_file  # 获取源代码行和文件路径的函数
from .._jit_internal import (  # Torch JIT 内部函数和类的导入
    _Await,  # 自定义的 await 类型
    _qualified_name,  # 获取对象的限定名称
    Any,  # 任意类型
    BroadcastingList1,  # 广播列表类型 1
    BroadcastingList2,  # 广播列表类型 2
    BroadcastingList3,  # 广播列表类型 3
    Dict,  # 字典类型
    Future,  # Future 对象类型
    is_await,  # 判断是否为 await 对象的函数
    is_dict,  # 判断是否为字典的函数
    is_future,  # 判断是否为 Future 对象的函数
    is_ignored_fn,  # 判断是否为被忽略的函数的函数
    is_list,  # 判断是否为列表的函数
    is_optional,  # 判断是否为可选类型的函数
    is_tuple,  # 判断是否为元组的函数
    is_union,  # 判断是否为联合类型的函数
    List,  # 列表类型
    Optional,  # 可选类型
    Tuple,  # 元组类型
    Union,  # 联合类型
)
from ._state import _get_script_class  # 获取脚本类的函数

# 如果分布式 RPC 可用，导入相关模块和类
if torch.distributed.rpc.is_available():
    from torch._C import RRefType  # 远程引用类型
    from .._jit_internal import is_rref, RRef  # 判断是否为远程引用的函数和远程引用类

from torch._ops import OpOverloadPacket  # 操作重载数据包类的导入


class Module:
    def __init__(self, name, members):
        self.name = name  # 初始化模块名称
        self.members = members  # 初始化模块成员字典

    def __getattr__(self, name):
        try:
            return self.members[name]  # 返回模块成员中指定名称的值
        except KeyError:
            raise RuntimeError(
                f"Module {self.name} has no member called {name}"
            ) from None  # 如果指定的成员不存在，则抛出运行时错误


class EvalEnv:
    env = {
        "torch": Module("torch", {"Tensor": torch.Tensor}),  # 创建包含 Torch 模块和 Tensor 类型的 Module 对象
        "Tensor": torch.Tensor,  # 张量类型
        "typing": Module("typing", {"Tuple": Tuple}),  # 创建包含 typing 模块和 Tuple 类型的 Module 对象
        "Tuple": Tuple,  # 元组类型
        "List": List,  # 列表类型
        "Dict": Dict,  # 字典类型
        "Optional": Optional,  # 可选类型
        "Union": Union,  # 联合类型
        "Future": Future,  # Future 对象类型
        "Await": _Await,  # 自定义的 await 类型
    }

    def __init__(self, rcb):
        self.rcb = rcb  # 初始化回调函数
        if torch.distributed.rpc.is_available():
            self.env["RRef"] = RRef  # 如果分布式 RPC 可用，添加 RRef 到环境中

    def __getitem__(self, name):
        if name in self.env:
            return self.env[name]  # 如果在环境中找到名称，则返回其对应的值
        if self.rcb is not None:
            return self.rcb(name)  # 如果有回调函数，则调用回调函数并返回结果
        return getattr(builtins, name, None)  # 否则返回内置模块中的对应对象或者 None


def get_signature(fn, rcb, loc, is_method):
    if isinstance(fn, OpOverloadPacket):
        signature = try_real_annotations(fn.op, loc)  # 如果 fn 是 OpOverloadPacket 类型，则尝试获取其真实注解
    else:
        signature = try_real_annotations(fn, loc)  # 否则，尝试获取 fn 的真实注解
    if signature is not None and is_method:
        # 如果是方法，签名将包含 `self` 参数，但类型注释不包含 `self`，因此在这里去除它以保持一致性
        param_types, return_type = signature
        param_types = param_types[1:]  # 去除第一个参数 `self`
        signature = (param_types, return_type)  # 更新签名
    # 如果签名为空时执行以下操作
    if signature is None:
        # 初始化 type_line 和 source 为 None
        type_line, source = None, None
        try:
            # 尝试获取函数 fn 的源代码并进行缩进去除处理
            source = dedent("".join(get_source_lines_and_file(fn)[0]))
            # 获取源代码中的类型注解行
            type_line = get_type_line(source)
        except TypeError:
            # 捕获到 TypeError 异常时不做处理
            pass
        # 如果成功获取到类型注解行
        if type_line is not None:
            # 解析类型注解行，生成函数签名
            signature = parse_type_line(type_line, rcb, loc)

    # 返回函数的签名
    return signature
def is_function_or_method(the_callable):
    # 判断给定对象是否函数或方法，不包括内置函数
    return inspect.isfunction(the_callable) or inspect.ismethod(the_callable)


def is_vararg(the_callable):
    if not is_function_or_method(the_callable) and callable(the_callable):  # noqa: B004
        # 如果 `the_callable` 是类，则解糖调用以获取其签名信息
        the_callable = the_callable.__call__

    if is_function_or_method(the_callable):
        # 检查函数或方法是否有可变参数
        return inspect.getfullargspec(the_callable).varargs is not None
    else:
        return False


def get_param_names(fn, n_args):
    if isinstance(fn, OpOverloadPacket):
        fn = fn.op

    if (
        not is_function_or_method(fn)
        and callable(fn)
        and is_function_or_method(fn.__call__)
    ):  # noqa: B004
        # 解糖调用到类的 __call__ 方法
        fn = fn.__call__

    if is_function_or_method(fn):
        if is_ignored_fn(fn):
            fn = inspect.unwrap(fn)
        # 获取函数或方法的参数名列表
        return inspect.getfullargspec(fn).args
    else:
        # 如果 `fn` 不是方法或函数（可能是带有 __call__ 方法的类），返回默认参数名列表
        return [str(i) for i in range(n_args)]


def check_fn(fn, loc):
    # 确保函数定义不是类的实例化
    try:
        source = dedent("".join(get_source_lines_and_file(fn)[0]))
    except (OSError, TypeError):
        return
    if source is None:
        return

    py_ast = ast.parse(source)
    if len(py_ast.body) == 1 and isinstance(py_ast.body[0], ast.ClassDef):
        # 在脚本函数中不能实例化类
        raise torch.jit.frontend.FrontendError(
            loc,
            f"Cannot instantiate class '{py_ast.body[0].name}' in a script function",
        )
    if len(py_ast.body) != 1 or not isinstance(py_ast.body[0], ast.FunctionDef):
        # 期望只有一个顶级函数定义
        raise torch.jit.frontend.FrontendError(
            loc, "Expected a single top-level function"
        )


def _eval_no_call(stmt, glob, loc):
    """Evaluate statement as long as it does not contain any method/function calls."""
    # 编译并执行不包含任何方法/函数调用的语句
    bytecode = compile(stmt, "", mode="eval")
    for insn in dis.get_instructions(bytecode):
        if "CALL" in insn.opname:
            raise RuntimeError(
                f"Type annotation should not contain calls, but '{stmt}' does"
            )
    return eval(bytecode, glob, loc)  # type: ignore[arg-type] # noqa: P204


def parse_type_line(type_line, rcb, loc):
    """Parse a type annotation specified as a comment.

    Example inputs:
        # type: (Tensor, torch.Tensor) -> Tuple[Tensor]
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tensor
    """
    # 解析作为注释指定的类型注解
    arg_ann_str, ret_ann_str = split_type_line(type_line)

    try:
        arg_ann = _eval_no_call(arg_ann_str, {}, EvalEnv(rcb))
    except (NameError, SyntaxError) as e:
        raise RuntimeError(
            "Failed to parse the argument list of a type annotation"
        ) from e
    # 检查 arg_ann 是否为元组类型，如果不是则转换为单元素元组
    if not isinstance(arg_ann, tuple):
        arg_ann = (arg_ann,)
    
    # 尝试评估 ret_ann_str 所表示的返回类型字符串，使用空字典作为命名空间，使用指定的评估环境 EvalEnv(rcb)
    try:
        ret_ann = _eval_no_call(ret_ann_str, {}, EvalEnv(rcb))
    # 捕获可能抛出的 NameError 或 SyntaxError 异常，并将其作为 RuntimeError 的原因重新抛出
    except (NameError, SyntaxError) as e:
        raise RuntimeError(
            "Failed to parse the return type of a type annotation"
        ) from e
    
    # 根据 arg_ann 中的注解列表，转换每个注解为其对应的类型对象，并存储在 arg_types 列表中
    arg_types = [ann_to_type(ann, loc) for ann in arg_ann]
    
    # 将 ret_ann 转换为其对应的类型对象，存储在返回值的位置 loc 中
    return arg_types, ann_to_type(ret_ann, loc)
# 尝试查找包含类型注释的行
def get_type_line(source):
    # 类型注释的标志
    type_comment = "# type:"

    # 按行分割源代码
    lines = source.split("\n")
    # 枚举所有行，以便在需要时获取行号
    lines = list(enumerate(lines))
    # 筛选出包含类型注释的行
    type_lines = list(filter(lambda line: type_comment in line[1], lines))

    # 检查是否存在需要忽略的类型注释
    # 在 torch/_VF.py 中，可能需要 `type: ignore` 注释，用于 mypy 的 JIT 函数处理
    # 此类 ignore 类型注释的格式有：
    #   1) type: ignore
    #   2) type: ignore[rule-code]
    # 忽略注释必须位于行尾

    # 在空格之前添加额外的反斜杠，以避免在 .github/workflows/lint.yml 中触发的某些检查
    # 类型注释的模式匹配，用于排除 `type: ignore` 注释
    type_pattern = re.compile("# type:\\ ignore(\\[[a-zA-Z-]+\\])?$")
    type_lines = list(filter(lambda line: not type_pattern.search(line[1]), type_lines))

    if len(type_lines) == 0:
        # 捕获常见的打字错误模式，例如额外的空格、ignore 中的拼写错误等
        wrong_type_pattern = re.compile("#[\t ]*type[\t ]*(?!: ignore(\\[.*\\])?$):")
        wrong_type_lines = list(
            filter(lambda line: wrong_type_pattern.search(line[1]), lines)
        )
        if len(wrong_type_lines) > 0:
            # 抛出运行时错误，提示类型注释前缀可能无效
            raise RuntimeError(
                "The annotation prefix in line "
                + str(wrong_type_lines[0][0])
                + " is probably invalid.\nIt must be '# type:'"
                + "\nSee PEP 484 (https://www.python.org/dev/peps/pep-0484/#suggested-syntax-for-python-2-7-and-straddling-code)"  # noqa: B950
                + "\nfor examples"
            )
        return None
    elif len(type_lines) == 1:
        # 只有一行类型注释，直接返回该行，并去除首尾空白字符
        return type_lines[0][1].strip()

    # 如果有多行类型注释，解析参数类型和返回类型
    return_line = None
    parameter_type_lines = []
    for line_num, line in type_lines:
        if "# type: (...) -> " in line:
            return_line = (line_num, line)
            break
        elif type_comment in line:
            parameter_type_lines.append(line)
    if return_line is None:
        # 如果没有找到返回类型行 '# type: (...) -> ...'，则抛出运行时错误
        raise RuntimeError(
            "Return type line '# type: (...) -> ...' not found on multiline "
            "type annotation\nfor type lines:\n"
            + "\n".join([line[1] for line in type_lines])
            + "\n(See PEP 484 https://www.python.org/dev/peps/pep-0484/#suggested-syntax-for-python-2-7-and-straddling-code)"
        )

    # 从参数类型行中获取参数类型
    def get_parameter_type(line):
        # 获取类型注释后的内容，去除首尾空白字符
        item_type = line[line.find(type_comment) + len(type_comment) :]
        return item_type.strip()

    # 获取所有参数类型，并以逗号分隔
    types = map(get_parameter_type, parameter_type_lines)
    parameter_types = ", ".join(types)

    # 将返回类型行中的省略号替换为参数类型
    return return_line[1].replace("...", parameter_types)


def split_type_line(type_line):
    # 将包含类型注释的行分割成参数类型和返回类型的部分
    """
    # 计算注释中类型注解的返回值

    # 计算起始偏移量，即注释中 "# type:" 的长度
    start_offset = len("# type:")

    # 尝试查找箭头符号 "->" 的位置
    try:
        arrow_pos = type_line.index("->")
    except ValueError:
        # 如果找不到 "->"，则抛出运行时错误
        raise RuntimeError(
            "Syntax error in type annotation (couldn't find `->`)"
        ) from None

    # 返回从起始偏移量到箭头位置之间的内容（表示参数类型），以及箭头后面到结尾的内容（表示返回类型），并去除两端的空白字符
    return type_line[start_offset:arrow_pos].strip(), type_line[arrow_pos + 2 :].strip()
    """
# 尝试使用 Python 3.5+ 的注解语法获取类型信息
def try_real_annotations(fn, loc):
    try:
        # 获取函数签名对象
        sig = inspect.signature(fn)
    except ValueError:
        # 如果获取失败，返回 None
        return None

    # 收集所有参数和返回值的注解
    all_annots = [sig.return_annotation] + [
        p.annotation for p in sig.parameters.values()
    ]
    # 如果所有注解都是空的，返回 None
    if all(ann is sig.empty for ann in all_annots):
        return None

    # 将参数的类型注解转换为对应的类型
    arg_types = [ann_to_type(p.annotation, loc) for p in sig.parameters.values()]
    # 将返回值的类型注解转换为对应的类型
    return_type = ann_to_type(sig.return_annotation, loc)
    # 返回参数类型列表和返回值类型
    return arg_types, return_type


# 查找枚举类中枚举值的共同类型。如果不是同一类型，则返回 AnyType。
def get_enum_value_type(e: Type[enum.Enum], loc):
    # 获取枚举类中的所有枚举值
    enum_values: List[enum.Enum] = list(e)
    # 如果没有枚举值，抛出异常
    if not enum_values:
        raise ValueError(f"No enum values defined for: '{e.__class__}'")

    # 收集所有枚举值的类型
    types = {type(v.value) for v in enum_values}
    # 尝试将类型转换为对应的类型
    ir_types = [try_ann_to_type(t, loc) for t in types]

    # 如果枚举值的类型不同，将会在此处引发异常。
    # 虽然 Python 支持这种情况，但我们选择在此处不实现以避免复杂化逻辑，因为这是一个罕见的情况。
    # 如果你认为有必要，请报告一个功能请求。
    res = torch._C.unify_type_list(ir_types)
    # 如果无法统一类型列表，返回 AnyType
    if not res:
        return AnyType.get()
    # 否则返回统一后的类型
    return res


# 检查类型是否是 Tensor 或其子类
def is_tensor(ann):
    if issubclass(ann, torch.Tensor):
        return True

    # 检查是否是 Tensor 的具体子类
    if issubclass(
        ann,
        (
            torch.LongTensor,
            torch.DoubleTensor,
            torch.FloatTensor,
            torch.IntTensor,
            torch.ShortTensor,
            torch.HalfTensor,
            torch.CharTensor,
            torch.ByteTensor,
            torch.BoolTensor,
        ),
    ):
        # 发出警告，说明 TorchScript 将这些子类型的类型注解视为普通的 Tensor。
        # 编译时也不会强制执行 dtype 约束。
        warnings.warn(
            "TorchScript will treat type annotations of Tensor "
            "dtype-specific subtypes as if they are normal Tensors. "
            "dtype constraints are not enforced in compilation either."
        )
        return True

    return False


# 返回 None 的占位函数
def _fake_rcb(inp):
    return None


# 尝试将类型注解转换为对应的类型
def try_ann_to_type(ann, loc, rcb=None):
    # 获取类型注解的参数
    ann_args = typing.get_args(ann)  # always returns a tuple!

    # 如果类型注解为空，返回推断出的 Tensor 类型
    if ann is inspect.Signature.empty:
        return TensorType.getInferred()
    # 如果类型注解为 None，返回 None 类型
    if ann is None:
        return NoneType.get()
    # 如果类型注解是类并且是 Tensor 或其子类，返回 Tensor 类型
    if inspect.isclass(ann) and is_tensor(ann):
        return TensorType.get()
    # 如果类型注解是元组类型
    if is_tuple(ann):
        # 处理空元组类型注解 `Tuple[()]`
        if len(ann_args) == 1 and ann_args[0] == ():
            return TupleType([])
        # 返回元组类型，其中包含元组内每个类型的转换结果
        return TupleType([try_ann_to_type(a, loc) for a in ann_args])
    # 如果类型注解是列表类型
    if is_list(ann):
        # 获取列表元素类型的转换结果
        elem_type = try_ann_to_type(ann_args[0], loc)
        if elem_type:
            return ListType(elem_type)
    # 如果注解 ann 是字典类型
    if is_dict(ann):
        # 尝试将注解的第一个和第二个参数转换为类型
        key = try_ann_to_type(ann_args[0], loc)
        value = try_ann_to_type(ann_args[1], loc)
        # 如果 key 或者 value 为 None，则抛出 ValueError 异常
        if key is None:
            raise ValueError(
                f"Unknown type annotation: '{ann_args[0]}' at {loc.highlight()}"
            )
        if value is None:
            raise ValueError(
                f"Unknown type annotation: '{ann_args[1]}' at {loc.highlight()}"
            )
        # 返回字典类型
        return DictType(key, value)
    
    # 如果注解 ann 是可选类型
    if is_optional(ann):
        # 如果第二个参数是 None 类型，则 contained 等于第一个参数，否则等于第二个参数
        if issubclass(ann_args[1], type(None)):
            contained = ann_args[0]
        else:
            contained = ann_args[1]
        # 尝试将 contained 转换为类型
        valid_type = try_ann_to_type(contained, loc)
        # 如果 valid_type 为 None，则抛出 AssertionError 异常
        msg = "Unsupported annotation {} could not be resolved because {} could not be resolved. At\n{}"
        assert valid_type, msg.format(repr(ann), repr(contained), repr(loc))
        # 返回可选类型
        return OptionalType(valid_type)
    
    # 如果注解 ann 是联合类型
    if is_union(ann):
        # TODO: 这是识别 NumberType 的一种方法
        if set(ann_args) == {int, float, complex}:
            return NumberType.get()
        inner: List = []
        # 对于 typing.get_args(ann) 中的每个参数 a
        # 如果 a 是 None，则将 NoneType.get() 添加到 inner 中
        # 否则尝试将 a 转换为类型，如果无法转换则抛出 AssertionError 异常
        for a in typing.get_args(ann):
            if a is None:
                inner.append(NoneType.get())
            maybe_type = try_ann_to_type(a, loc)
            msg = "Unsupported annotation {} could not be resolved because {} could not be resolved. At\n{}"
            assert maybe_type, msg.format(repr(ann), repr(maybe_type), repr(loc))
            inner.append(maybe_type)
        # 返回联合类型
        return UnionType(inner)  # type: ignore[arg-type]
    
    # 如果 torch.distributed.rpc 可用且 ann 是 RRef 类型
    if torch.distributed.rpc.is_available() and is_rref(ann):
        # 返回 RRefType 类型
        return RRefType(try_ann_to_type(ann_args[0], loc))
    
    # 如果 ann 是 Future 类型
    if is_future(ann):
        # 返回 FutureType 类型
        return FutureType(try_ann_to_type(ann_args[0], loc))
    
    # 如果 ann 是 Awaitable 类型
    if is_await(ann):
        # 如果 ann_args 存在，则将其第一个参数尝试转换为类型，否则返回 AnyType.get()
        elementType = try_ann_to_type(ann_args[0], loc) if ann_args else AnyType.get()
        # 返回 AwaitType 类型
        return AwaitType(elementType)
    
    # 如果 ann 是 float 类型
    if ann is float:
        # 返回 FloatType 类型
        return FloatType.get()
    
    # 如果 ann 是 complex 类型
    if ann is complex:
        # 返回 ComplexType 类型
        return ComplexType.get()
    
    # 如果 ann 是 int 或者 torch.SymInt 类型
    if ann is int or ann is torch.SymInt:
        # 返回 IntType 类型
        return IntType.get()
    
    # 如果 ann 是 str 类型
    if ann is str:
        # 返回 StringType 类型
        return StringType.get()
    
    # 如果 ann 是 bool 类型
    if ann is bool:
        # 返回 BoolType 类型
        return BoolType.get()
    
    # 如果 ann 是 Any 类型
    if ann is Any:
        # 返回 AnyType 类型
        return AnyType.get()
    
    # 如果 ann 是 type(None) 类型
    if ann is type(None):
        # 返回 NoneType 类型
        return NoneType.get()
    
    # 如果 ann 是一个类且具有 "__torch_script_interface__" 属性
    if inspect.isclass(ann) and hasattr(ann, "__torch_script_interface__"):
        # 返回 InterfaceType 类型
        return InterfaceType(ann.__torch_script_interface__)
    
    # 如果 ann 是 torch.device 类型
    if ann is torch.device:
        # 返回 DeviceObjType 类型
        return DeviceObjType.get()
    
    # 如果 ann 是 torch.Generator 类型
    if ann is torch.Generator:
        # 返回 _GeneratorType 类型
        return _GeneratorType.get()
    
    # 如果 ann 是 torch.Stream 类型
    if ann is torch.Stream:
        # 返回 StreamObjType 类型
        return StreamObjType.get()
    
    # 如果 ann 是 torch.dtype 类型
    if ann is torch.dtype:
        # 返回 IntType 类型，因为 dtype 尚未绑定为自己的类型
        return IntType.get()
    # 检查注解是否是类，并且是否是枚举的子类
    if inspect.isclass(ann) and issubclass(ann, enum.Enum):
        # 如果注解是枚举类，并且还没有被脚本化，则将其递归编译成脚本类
        if _get_script_class(ann) is None:
            scripted_class = torch.jit._script._recursive_compile_class(ann, loc)
            # 获取脚本化类的完全限定名称
            name = scripted_class.qualified_name()
        else:
            # 否则，获取注解的限定名称
            name = _qualified_name(ann)
        # 返回一个枚举类型对象，包含名称、枚举值类型和枚举类的所有成员
        return EnumType(name, get_enum_value_type(ann, loc), list(ann))
    
    # 如果注解是类但不是枚举类
    if inspect.isclass(ann):
        # 尝试获取该类的脚本化版本
        maybe_script_class = _get_script_class(ann)
        if maybe_script_class is not None:
            return maybe_script_class
        # 如果该类可以被编译成脚本，则递归编译该类
        if torch._jit_internal.can_compile_class(ann):
            return torch.jit._script._recursive_compile_class(ann, loc)
    
    # 如果没有提供的RCB回调函数，则使用一个假的回调函数
    if rcb is None:
        rcb = _fake_rcb
    # 根据提供的注解解析类型，并返回解析后的类型对象
    return torch._C._resolve_type_from_object(ann, loc, rcb)
# 将注释转换为类型
def ann_to_type(ann, loc, rcb=None):
    # 尝试将注释转换为类型
    the_type = try_ann_to_type(ann, loc, rcb)
    # 如果成功转换，返回类型
    if the_type is not None:
        return the_type
    # 如果无法转换，抛出值错误，显示位置和注释内容
    raise ValueError(f"Unknown type annotation: '{ann}' at {loc.highlight()}")

# 导出的符号列表，用于模块级别的导入
__all__ = [
    "Any",  # 任意类型
    "List",  # 列表类型
    "BroadcastingList1",  # 广播列表类型1
    "BroadcastingList2",  # 广播列表类型2
    "BroadcastingList3",  # 广播列表类型3
    "Tuple",  # 元组类型
    "is_tuple",  # 判断是否为元组的函数
    "is_list",  # 判断是否为列表的函数
    "Dict",  # 字典类型
    "is_dict",  # 判断是否为字典的函数
    "is_optional",  # 判断是否为可选类型的函数
    "is_union",  # 判断是否为联合类型的函数
    "TensorType",  # 张量类型
    "TupleType",  # 元组类型
    "FloatType",  # 浮点数类型
    "ComplexType",  # 复数类型
    "IntType",  # 整数类型
    "ListType",  # 列表类型
    "StringType",  # 字符串类型
    "DictType",  # 字典类型
    "AnyType",  # 任意类型
    "Module",  # 模块类型
    # TODO: 考虑在通配符导入期间不导出这些（保留给类型；用于惯用的类型化代码。）
    "get_signature",  # 获取函数签名的函数
    "check_fn",  # 检查函数的函数
    "get_param_names",  # 获取函数参数名的函数
    "parse_type_line",  # 解析类型行的函数
    "get_type_line",  # 获取类型行的函数
    "split_type_line",  # 分割类型行的函数
    "try_real_annotations",  # 尝试真实注释的函数
    "try_ann_to_type",  # 尝试将注释转换为类型的函数
    "ann_to_type",  # 将注释转换为类型的函数
]
```