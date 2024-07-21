# `.\pytorch\torch\_library\utils.py`

```py
# 设置 mypy 以允许未类型化的定义
mypy: allow-untyped-defs

# 导入必要的模块和类
import dataclasses  # 用于数据类的装饰器
import inspect  # 用于获取调用栈信息
import sys  # 系统相关的功能
from typing import Any, Callable, Dict, Iterable, Tuple  # 引入类型提示

import torch  # PyTorch 深度学习库
from torch import _C, _utils_internal  # PyTorch 的核心 C 模块和内部工具模块
from torch._ops import OpOverload  # 引入操作重载类


@dataclasses.dataclass
class Kernel:
    """表示一个（函数，源代码位置）"""

    func: Callable  # 可调用对象
    source: str  # 源代码位置信息

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class RegistrationHandle:
    """当调用 destroy() 方法时执行某些操作"""

    def __init__(self, on_destroy: Callable):
        self._on_destroy = on_destroy  # 销毁时的回调函数

    def destroy(self) -> None:
        self._on_destroy()  # 执行销毁时的操作


def get_source(stacklevel: int) -> str:
    """获取表示调用者的字符串。

    示例："/path/to/foo.py:42"

    使用 stacklevel=1 获取调用者的源代码位置
    使用 stacklevel=2 获取调用者的调用者的源代码位置
    依此类推。
    """
    frame = inspect.getframeinfo(sys._getframe(stacklevel))
    source = f"{frame.filename}:{frame.lineno}"
    return source  # 返回调用者的源代码位置字符串


def parse_namespace(qualname: str) -> Tuple[str, str]:
    splits = qualname.split("::")
    if len(splits) != 2:
        raise ValueError(
            f"Expected `qualname` to be of the form "
            f'"namespace::name", but got {qualname}. '
            f"The qualname passed to the torch.library APIs must consist "
            f"of a namespace and a name, e.g. aten::sin"
        )
    return splits[0], splits[1]  # 返回命名空间和名称的元组


def lookup_op(qualname: str) -> OpOverload:
    namespace, name = parse_namespace(qualname)
    if "." in name:
        name, overload = name.split(".")
    else:
        overload = "default"
    ns = getattr(torch.ops, namespace)
    packet = getattr(ns, name)
    return getattr(packet, overload)  # 返回操作重载对象


def is_builtin(op: OpOverload) -> bool:
    assert isinstance(op, OpOverload)
    return op.namespace in {"aten", "prim", "prims"}  # 检查操作是否为内置操作


def is_functional_schema(schema: Any) -> bool:
    """检查模式是否为功能性模式。

    操作符是功能性的如果：
    - 它不会修改任何输入
    - 它不会返回任何输入的视图
    - 它至少有一个返回值
    """

    def is_functional(schema):
        if schema.is_mutable:
            return False
        rets = schema.returns
        is_non_mutating_view = len(rets) > 0 and any(
            r.alias_info is not None and not r.alias_info.is_write for r in rets
        )
        if is_non_mutating_view:
            return False
        if not schema.returns:
            return False
        return True

    if isinstance(schema, torch._C.FunctionSchema):
        return is_functional(schema)

    # 惰性导入，因为并非所有的 PyTorch 构建都有 torchgen
    from torchgen.model import FunctionSchema

    if isinstance(schema, str):
        schema = FunctionSchema.parse(schema)
    assert isinstance(schema, FunctionSchema)
    return is_functional(schema)  # 检查模式是否为功能性模式
# 判断给定的类型是否类似于 Tensor 列表类型
def is_tensorlist_like_type(typ: Any) -> bool:
    return (
        typ == _C.ListType(_C.TensorType.get())
        or typ == _C.ListType(_C.OptionalType(_C.TensorType.get()))
        or typ == _C.OptionalType(_C.ListType(_C.TensorType.get()))
        or typ == _C.OptionalType(_C.ListType(_C.OptionalType(_C.TensorType.get())))
    )


# 判断给定的类型是否类似于 Tensor 类型
# 应该是 torch._C.JitType，但是该注释已经损坏
def is_tensor_like_type(typ: Any) -> bool:
    return typ == _C.TensorType.get() or typ == _C.OptionalType(_C.TensorType.get())


# 检查操作是否是原位 aten 操作，即它会修改并返回第一个参数
def mutates_and_returns_first_arg(op: OpOverload):
    """Check if an op is an inplace aten op, i.e. it mutates and returns the first arg.

    TODO: torchgen/model.py's FunctionSchema.parse is the source of truth for this,
    but not all PyTorch builds have torchgen (due to the yaml dependency being weird).
    Figure this out.

    Example: add_(Tensor(a!) x, Tensor y) -> Tensor(a)
    """
    if op.namespace != "aten":
        return False
    schema = op._schema
    if not len(schema.returns) == 1:
        return False
    if schema.returns[0].alias_info is None:
        return False
    alias_set = schema.returns[0].alias_info.after_set
    if len(alias_set) != 1:
        return False
    loc = next(iter(alias_set))
    if len(schema.arguments) < 1:
        return False
    first_arg = schema.arguments[0]
    if first_arg.alias_info is None:
        return False
    if not first_arg.alias_info.is_write:
        return False
    alias_set = first_arg.alias_info.after_set
    if len(alias_set) != 1:
        return False
    if loc != next(iter(alias_set)):
        return False
    for arg in schema.arguments[1:]:
        if arg.alias_info is not None:
            return False
    return True


# 根据函数模式填充参数的默认值，返回填充后的新参数和新关键字参数
def fill_defaults(schema, args, kwargs):
    new_args = []
    new_kwargs = {}
    for i in range(len(schema.arguments)):
        info = schema.arguments[i]
        if info.kwarg_only:
            if info.name in kwargs:
                new_kwargs[info.name] = kwargs[info.name]
            else:
                new_kwargs[info.name] = info.default_value
        else:
            if i < len(args):
                new_args.append(args[i])
            else:
                new_args.append(info.default_value)
    return tuple(new_args), new_kwargs


# 将函数模式的参数与实际参数 (args, kwargs) 进行打包
# 假设 (args, kwargs) 是某个 torch._ops.OpOverload 的输入，其中 kwargs 必须是仅限关键字参数，且可以省略默认值
def zip_schema(
    schema: _C.FunctionSchema, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Iterable[Tuple[_C.Argument, Any]]:
    """zips schema.arguments and (args, kwargs) together.

    Assumes that (args, kwargs) were the inputs to some torch._ops.OpOverload:
    that is, kwargs must be keyword-only arguments and default values may be omitted.
    """
    assert len(schema.arguments) >= len(args) + len(kwargs)
    # 遍历 schema.arguments 列表的索引范围
    for i in range(len(schema.arguments)):
        # 获取当前索引位置的参数信息对象
        info = schema.arguments[i]
        # 检查是否为仅关键字参数
        if info.kwarg_only:
            # 如果参数名称在 kwargs 中存在，生成该参数信息和对应的值
            if info.name in kwargs:
                yield info, kwargs[info.name]
            # 继续下一个循环
            continue
        
        # 如果索引 i 大于等于 args 列表的长度
        if i >= len(args):
            # 参数值等于默认值的参数不会被填充
            # 如果它们后面还有参数也等于默认值，则跳过这些参数
            # 跳过这种情况继续下一个循环
            continue
        
        # 生成参数信息和 args 中对应索引位置的参数值
        yield info, args[i]
    
    # 函数返回结束
    return
# 检查是否可以生成一个简单的虚拟实现
def can_generate_trivial_fake_impl(op: OpOverload) -> bool:
    # 断言操作是 OpOverload 类的实例
    assert isinstance(op, OpOverload)
    # 如果操作是内置的，我们可以控制它们。但这可能会在自定义操作上进行输入元数据的变异（尽管很少发生）。
    # 因此返回 False。
    if is_builtin(op):
        return False
    # 获取操作的模式模式
    schema = op._schema
    # 如果操作不可变且没有返回值，则返回 False，出于谨慎的考虑
    if not schema.is_mutable:
        return False
    # 如果操作返回了任何东西，则返回 False
    if len(schema.returns) > 0:
        return False
    # 如果操作没有返回任何东西，则认为它具有一个简单的虚拟实现，返回 True
    return True


# 判断是否需要设置 Python 模块
def requires_set_python_module() -> bool:
    """如果一个操作是在 C++ 中定义的，并且通过 torch.library API 扩展到 Python，
    则返回是否要求从 C++ 中有一个 m.set_python_module("mylib.ops") 的调用，
    将 C++ 操作关联到一个 Python 模块。
    """
    # 返回 _utils_internal 模块中的 REQUIRES_SET_PYTHON_MODULE 属性，如果不存在则默认为 True
    return getattr(_utils_internal, "REQUIRES_SET_PYTHON_MODULE", True)


# 处理分发模式
def handle_dispatch_mode(curr_mode, op_overload, *args, **kwargs):
    # 断言 curr_mode 是 TorchDispatchMode 类的实例
    assert isinstance(curr_mode, torch.utils._python_dispatch.TorchDispatchMode)
    # 初始化重载类型列表
    overload_types = []
    # 展开参数列表
    args_flattened, _ = torch.utils._pytree.tree_flatten((args, kwargs.values()))
    # 遍历展开后的参数列表
    for a in args_flattened:
        # TODO: 需要仔细检查 torch_dispatch 中 "types" 参数的语义。
        # 它在 PyInterpreter.cpp 中生成，但似乎在两个地方生成，
        # 其中一种情况下我们只包含带有 python 键的张量，另一种情况下包含所有张量。
        if isinstance(a, torch.Tensor) and torch._C._dispatch_keys(a).has(
            torch._C.DispatchKey.Python
        ):
            # 如果参数是 torch.Tensor 并且具有 Python 分发键，则将其类型添加到重载类型列表中
            overload_types.append(type(a))
    # TODO: 检查我是否正确理解了这些参数（在 C++ 中，我们传入 "0000"？？）

    # 调用当前模式的 __torch_dispatch__ 方法来处理操作重载、重载类型和参数
    return curr_mode.__torch_dispatch__(op_overload, overload_types, args, kwargs)


# 判断函数模式是否具有仅关键字参数
def has_kwarg_only_args(schema: _C.FunctionSchema):
    # 检查模式中是否存在任何仅关键字参数
    return any(a.kwarg_only for a in schema.arguments)


# 判断函数模式是否具有仅关键字张量
def has_kwarg_only_tensors(schema: _C.FunctionSchema):
    # 遍历模式中的参数列表
    for a in schema.arguments:
        # 如果参数类型既不是张量类型也不是张量列表类型，则继续下一个参数
        if not (is_tensor_like_type(a.type) or is_tensorlist_like_type(a.type)):
            continue
        # 如果参数是关键字只参数，则返回 True
        if a.kwarg_only:
            return True
    # 如果没有找到关键字只张量参数，则返回 False
    return False
```