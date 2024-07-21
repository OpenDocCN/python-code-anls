# `.\pytorch\torch\_higher_order_ops\auto_functionalize.py`

```py
# mypy: allow-untyped-defs
# 引入必要的类型引用
from typing import Any, Dict, List, Optional, Tuple, Union

# 导入PyTorch相关库
import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)

# NOTE: [auto-functionalizing custom ops]
# 用户可能希望torch.compile自定义操作以改变它们的输入。
# torch.compile将自动支持此类操作，无需任何人为提供功能化内核。
# 这里是如何做到的。

# 假设我们有一个假设的操作mylib::sin_(Tensor(a!) x) -> ()
# 当FakeTensor遇到此操作时：
# - 如果架构表明它不返回任何东西，我们可以生成一个简单的FakeTensor规则（不返回任何东西）。
# - 否则，用户需要提供一个FakeTensor实现（伪实现）。

# 接下来，当Python的FunctionalTensor看到此操作时，它将通过发出一个调用来对其进行功能化：
# auto_functionalize(op, ["x"], {"x": ...}) HOP，并用此HOP的相应输出替换被突变的输入。
# 此HOP在调用时有效地运行操作的功能版本：克隆将被突变的输入，运行操作，然后返回（输出，具有新值的张量）

class AutoFunctionalized(HigherOrderOperator):
    """auto_functionalized(_mutable_op, **kwargs)

    This HOP runs a "functional" version of _mutable_op.

    Concretely, it looks at all the arguments that are mutable through
    _mutable_op's operator schema, clones those kwargs, runs
    `out = _mutable_op(**kwargs)` with the cloned values, and then returns the
    operator output concatenated with the cloned values that were mutated.

    We have some restrictions on `_mutable_op`.
    See `can_auto_functionalize` for the restrictions. We can likely lift
    many of these if users request it.

    The reason why _mutable_op is prefixed with an
    underscore is to prevent collisions with kwarg names in **kwargs.
    """

    def __init__(self):
        super().__init__("auto_functionalized")

    def __call__(
        self,
        _mutable_op: torch._ops.OpOverload,
        **kwargs: Dict[str, Any],
    ) -> Tuple[Any, Tuple[Tensor, ...]]:
        # 确保可以自动功能化给定的操作
        assert can_auto_functionalize(_mutable_op)
        # 确保kwargs是一个字典类型
        assert isinstance(kwargs, dict)
        # 调用父类的__call__方法执行操作
        return super().__call__(_mutable_op, **kwargs)


# 创建AutoFunctionalized类的实例，用于运行功能化版本的操作
auto_functionalized = AutoFunctionalized()


def can_auto_functionalize(op: torch._ops.OperatorBase) -> bool:
    # 如果操作不是OpOverload类型，则无法自动功能化
    if not isinstance(op, torch._ops.OpOverload):
        return False

    # 对于内置操作，我们控制它们。这些操作可能（在罕见情况下）执行输入元数据的突变（我们已禁止自定义操作中的此类突变）
    if torch._library.utils.is_builtin(op):
        return False
    # 获取操作的架构信息
    schema = op._schema
    # 其它限制判断逻辑可能在此添加
    # 如果 schema 不可变，则返回 False
    if not schema.is_mutable:
        return False
    
    # 将 op 对象的 schema 赋值给 schema 变量
    schema = op._schema

    # 遍历 schema 的所有参数
    for arg in schema.arguments:
        # 如果参数没有别名信息，则继续下一个参数
        if arg.alias_info is None:
            continue
        # 如果参数的别名信息表明不可写，则继续下一个参数
        if not arg.alias_info.is_write:
            continue
        # 如果参数的类型是 torch.TensorType，则继续下一个参数
        if type(arg.type) is torch.TensorType:
            continue
        # 如果参数的类型是 torch.OptionalType，并且其元素类型是 torch.TensorType，则继续下一个参数
        if (
            type(arg.type) is torch.OptionalType
            and type(arg.type.getElementType()) is torch.TensorType
        ):
            continue
        # 如果参数的类型是 torch.ListType，并且其元素类型是 torch.TensorType，则继续下一个参数
        if (
            type(arg.type) is torch.ListType
            and type(arg.type.getElementType()) is torch.TensorType
        ):
            continue
        # 目前不支持的情况：其他类型的 Tensor。这包括像 Tensor?[]、Tensor[]? 等情况
        return False

    # 如果 schema 的返回值列表长度为 1，并且其类型是 torch.NoneType，则跳过
    if len(schema.returns) == 1 and isinstance(schema.returns[0].type, torch.NoneType):
        return True
    
    # 返回值不得别名任何内容
    for ret in schema.returns:
        # 如果返回值没有别名信息，并且其类型是 torch.TensorType，则继续下一个返回值
        if ret.alias_info is None and type(ret.type) is torch.TensorType:
            continue
        # 目前不支持的情况：返回类型是 List[Tensor]
        return False
    
    # 如果 op 的名称与 "Functionalize" 对应的调度键存在对应的内核，则返回 False
    if torch._C._dispatch_has_kernel_for_dispatch_key(op.name(), "Functionalize"):
        return False
    
    # 满足所有条件，返回 True
    return True
# 使用装饰器将函数实现注册到指定的函数调度键上
@auto_functionalized.py_impl(DispatchKey.CompositeExplicitAutograd)
def auto_functionalized_dense(
    _mutable_op: torch._ops.OpOverload,
    _only_clone_these_tensors: Optional[Tuple[str, ...]] = None,
    **kwargs: Dict[str, Any],
) -> Tuple[Any, Tuple[Tensor, ...]]:
    # 复制一份关键字参数
    new_kwargs = dict(**kwargs)
    # 初始化结果列表
    result = []

    # 获取可变参数的名称列表
    _mutable_args_names = get_mutable_arg_names(_mutable_op)
    # 遍历可变参数的名称列表
    for name in _mutable_args_names:
        # 如果指定了需要克隆的张量列表，并且当前参数不在该列表中，则直接复制而非克隆
        if (
            _only_clone_these_tensors is not None
            and name not in _only_clone_these_tensors
        ):
            new_kwargs[name] = kwargs[name]
        else:
            # 否则，根据参数是否为列表及其元素是否为张量进行克隆并保留步幅
            new_kwargs[name] = (
                [clone_preserve_strides(x) for x in kwargs[name]]
                if kwargs[name] is not None and isinstance(kwargs[name], list)
                else clone_preserve_strides(kwargs[name])
                if kwargs[name] is not None
                else None
            )
        # 将处理后的参数添加到结果列表中
        result.append(new_kwargs[name])

    # 调用可变操作并传入新的关键字参数
    out = _mutable_op(**new_kwargs)

    # 如果输出是一个元组，则将其展开并将处理后的参数也一并返回
    if isinstance(out, tuple):
        return (*out, *result)  # type: ignore[return-value]
    else:
        return (out, *result)  # type: ignore[return-value]


# 使用装饰器将函数实现注册到指定的模式上
@auto_functionalized.py_impl(FakeTensorMode)
def auto_functionalized_fake(
    mode,
    _mutable_op: torch._ops.OpOverload,
    **kwargs: Dict[str, Any],
) -> Tuple[Any, Tuple[Tensor, ...]]:
    # 使用指定模式进行上下文管理
    with mode:
        # 调用自动功能化的密集操作，并返回其结果
        result = auto_functionalized_dense(_mutable_op, **kwargs)
        return result


# 使用装饰器将函数实现注册到代理调度模式上
@auto_functionalized.py_impl(ProxyTorchDispatchMode)
def auto_functionalized_proxy(
    mode,
    _mutable_op: torch._ops.OpOverload,
    **kwargs: Dict[str, Any],
) -> Tuple[Any, Tuple[Tensor, ...]]:
    # 如果跟踪模式未启用，则直接调用自动功能化操作并返回结果
    if not mode.enable_tracing:
        return auto_functionalized(_mutable_op, **kwargs)

    # 否则，使用禁用代理模式跟踪上下文
    with disable_proxy_modes_tracing():
        out = auto_functionalized(_mutable_op, **kwargs)

    # 将关键字参数映射为非代理形式
    proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)
    # 创建代理调用函数
    out_proxy = mode.tracer.create_proxy(
        "call_function",
        auto_functionalized,
        (_mutable_op,),
        proxy_kwargs,
    )
    # 跟踪张量树结构的结果
    result = track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)
    return result


# 调用自动功能化操作并注册到指定的调度键上
auto_functionalized.fallthrough(DispatchKey.AutogradCPU)
auto_functionalized.fallthrough(DispatchKey.AutogradCUDA)


def get_mutable_arg_names(op: torch._ops.OpOverload) -> List[str]:
    """
    返回根据模式发生变化的参数名称列表。
    """
    # 从模式中获取可变参数名称列表
    mutable_args_names = [
        arg.name
        for arg in op._schema.arguments
        if arg.alias_info is not None and arg.alias_info.is_write
    ]
    return mutable_args_names


def do_auto_functionalize(
    op: torch._ops.OpOverload, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Any:
    """
    通过发出对op(*args, **kwargs)的调用来功能化，替换变异的(args, kwargs)与相应的输出。
    """
    # 调用自动功能化操作并返回其结果
    outs = auto_functionalized(op, **kwargs)
    return outs
    The normalized_kwargs are just the (args, kwargs), but all in kwarg form.
    This makes handling easier for the auto_functionalized HOP.
    """
    from torch._subclasses.functional_tensor import PythonFunctionalizeAPI

    ctx = PythonFunctionalizeAPI()

    # All of the (args, kwargs), but all as kwargs. The names for the
    # args come from the schema. This makes it easier for us to work with them.
    normalized_kwargs = {}
    schema = op._schema
    for idx, arg in enumerate(schema.arguments):
        # NB: torch_dispatch kwargs are the args defined as kwarg-only in the schema
        if arg.name in kwargs:
            normalized_kwargs[arg.name] = kwargs[arg.name]
        elif idx < len(args):
            # if its out of bounds we don't need to do anything
            # as it means the the optional arg was passed with its default
            # value
            normalized_kwargs[arg.name] = args[idx]
        else:
            normalized_kwargs[arg.name] = arg.default_value

    # Unwraps tensors from normalized_kwargs
    unwrapped_kwargs = ctx.unwrap_tensors(normalized_kwargs)  # type: ignore[arg-type]
    
    # Redispatch context to the next function in the call chain
    with ctx.redispatch_to_next():
        # Executes auto-functionalized operation with unwrapped kwargs
        unwrapped_outs = auto_functionalized(
            op, **unwrapped_kwargs  # type: ignore[arg-type]
        )

    # Retrieves names of mutable arguments from operation schema
    mutable_args_names = get_mutable_arg_names(op)

    # Separates actual outputs from mutable outputs
    unwrapped_actual_out: Union[Any, Tuple[Any]] = unwrapped_outs[
        : -len(mutable_args_names)
    ]
    unwrapped_mutable_out = unwrapped_outs[-len(mutable_args_names) :]

    # Handles scenarios based on the number of return values defined in the schema
    if len(op._schema.returns) == 0:
        assert unwrapped_actual_out[0] is None
        unwrapped_actual_out = None
    elif len(op._schema.returns) == 1:
        assert len(unwrapped_actual_out) == 1
        unwrapped_actual_out = unwrapped_actual_out[0]
    else:
        assert len(unwrapped_actual_out) == len(op._schema.returns)

    # Updates original arguments with mutated outputs
    for name, unwrapped_out in zip(mutable_args_names, unwrapped_mutable_out):
        # Can be None if input was `Tensor(a!)?`
        if unwrapped_out is None:
            continue

        # Synchronizes updates between unwrapped output and original argument
        def sync_update(o, orig_arg):
            ctx.replace(orig_arg, o)
            ctx.commit_update(orig_arg)
            ctx.sync(orig_arg)

        orig_arg = normalized_kwargs[name]

        # Handles Tensor or List[Tensor] types for auto-functionalization
        if isinstance(unwrapped_out, torch.Tensor):
            sync_update(unwrapped_out, orig_arg)
        elif isinstance(unwrapped_out, list) and all(
            isinstance(o, torch.Tensor) for o in unwrapped_out
        ):
            assert len(orig_arg) == len(unwrapped_out)
            for orig_a, o in zip(orig_arg, unwrapped_out):
                sync_update(o, orig_a)
        else:
            raise RuntimeError(
                f"unsupported type for auto-functionalization: {unwrapped_out}"
            )

    # Wraps tensors back into the context
    return ctx.wrap_tensors(unwrapped_actual_out)  # type: ignore[arg-type]
# 使用装饰器将函数转换为自动功能化的实现函数
@auto_functionalized.py_functionalize_impl
# 定义自动功能化的函数，接受上下文对象 ctx 和可变操作 _mutable_op 以及其他关键字参数
def auto_functionalized_func(ctx, _mutable_op, **kwargs):
    # 解包关键字参数中的张量，使其可以被操作处理
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    # 使用上下文对象重新调度到下一个操作
    with ctx.redispatch_to_next():
        # 调用自动功能化函数，传入可变操作和解包后的关键字参数
        result = auto_functionalized(_mutable_op, **unwrapped_kwargs)
    # 将结果包装为张量，并返回
    return ctx.wrap_tensors(result)
```