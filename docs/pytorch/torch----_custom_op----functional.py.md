# `.\pytorch\torch\_custom_op\functional.py`

```py
# mypy: allow-untyped-defs
# 引入弱引用模块，用于避免循环引用导致的内存泄漏问题
import weakref

# 引入 PyTorch 模块
import torch
# 引入 PyTorch 的 _pytree 模块
import torch.utils._pytree as pytree
# 从 torch._C 中引入 _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet 类
from torch._C import _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet
# 从 torch._ops 模块中引入 OpOverload 类
from torch._ops import OpOverload
# 从 torch.library 模块中引入 Library 类
from torch.library import Library
# 从 torchgen.model 模块中引入以下类和枚举
from torchgen.model import (
    BaseTy,
    BaseType,
    FunctionSchema,
    OperatorName,
    OptionalType,
    SchemaKind,
)

# 从当前包中的 autograd 模块中引入 autograd_not_implemented 函数
from .autograd import autograd_not_implemented


def register_functional_op(
    lib: Library,
    new_op_name: str,
    mutable_op: OpOverload,
) -> None:
    """Given a mutable operator, registers the functional variant.

    This API also correctly links the functional variant with the mutable
    operator for the purposes of functionalization.

    All of the new registrations are performed on the ``lib`` passed in.

    Arguments:
        lib (Library): Should be a torch.library.Library object that has
            the same namespace as ``mutable_op``'s namespace.
            lib will be used to register the new functional op as well
            as a functionalization kernel for the ``mutable_op``
            If you don't have a library handy, use
            ``torch.library.Library(ns, 'FRAGMENT')`` to construct one.
        new_op_name (str): The name of the functional operator (without the
            namespace). If no namespace, the new functional variant will be
            accessible under ``torch.ops.{lib.ns}.new_op_name``.
        mutable_op (OpOverload): The mutable custom operator. Note
            that you may need to add a `.default` to it, like
            `torch.ops.aten.abs_.default`.

    """
    # 对可变操作符进行验证
    validate(mutable_op)
    # 构建功能化 schema
    schema = functional_schema(new_op_name, mutable_op)
    # 在库中定义 schema
    lib.define(schema)

    # 构建功能化实现
    functional_impl = construct_functional_impl(mutable_op)
    # 在库中实现功能化操作
    lib.impl(new_op_name, functional_impl, 'CompositeExplicitAutograd')

    # 获取功能化操作的默认实现
    functional_op = getattr(getattr(torch.ops, lib.ns), new_op_name).default

    # 由于没有简便的方法生成自动求导核心，使用 autograd_not_implemented 函数代替
    # 这也使得用户无法自行注册自动求导公式，如果用户不直接在其程序中使用功能化操作，这不会成为问题，
    # 但未来可能需要重新审视这一点。
    lib.impl(new_op_name, autograd_not_implemented(functional_op), 'Autograd')

    # 构建功能化内核
    f_kernel = construct_functionalization_kernel(weakref.proxy(mutable_op), functional_op)

    # 在库中实现功能化
    lib.impl(mutable_op, f_kernel, 'Functionalize')


def construct_functional_impl(mutable_op):
    """Constructs a functional implementation from a mutable operator."""
    def functional_impl(*args):
        # 定义一个函数 functional_impl，接受任意数量的参数
        # 策略:
        # - 克隆可能被改变的参数
        # - 运行 mutable_op
        # - 返回克隆后的参数作为额外的输出

        # 初始化一个空列表，用于存储克隆后的参数
        new_args = []
        # 初始化一个空列表，用于存储额外的返回值
        extra_rets = []

        # 遍历 mutable_op 返回的每个参数及其是否可写属性
        for is_write, arg in zip(mutable_args(mutable_op), args):
            # 如果参数可写
            if is_write:
                # 如果参数不为 None，则克隆参数
                cloned = arg.clone() if arg is not None else None
                # 将克隆后的参数添加到 new_args 和 extra_rets
                new_args.append(cloned)
                extra_rets.append(cloned)
            else:
                # 如果参数不可写，则直接添加到 new_args
                new_args.append(arg)

        # 调用 mutable_op 函数，传入克隆后的参数
        result = mutable_op(*new_args)

        # 如果 mutable_op 返回 None，则返回额外返回值的元组
        if result is None:
            return tuple(extra_rets)
        # 如果 mutable_op 返回一个元组，则将其与额外返回值合并后返回
        if isinstance(result, tuple):
            return (*result, *extra_rets)
        # 否则，将 result 与额外返回值合并为一个元组返回
        return (result, *extra_rets)
    # 返回定义的函数 functional_impl
    return functional_impl
# 构建一个函数化内核，接受两个函数参数mutable_op和functional_op
def construct_functionalization_kernel(mutable_op, functional_op):
    # 定义一个内部函数kernel，接受任意数量的参数*args
    def kernel(*args):
        # 检查是否所有参数都是torch.Tensor类型且不是功能性张量
        # 如果不是功能性张量，则返回mutable_op(*args)
        if pytree.tree_all_only(torch.Tensor, lambda x: not torch._is_functional_tensor(x), args):
            with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Functionalize)):
                return mutable_op(*args)

        # 检查是否所有参数都是torch.Tensor类型且是功能性张量
        # 如果有非功能性张量存在，则抛出RuntimeError异常
        if not pytree.tree_all_only(torch.Tensor, torch._is_functional_tensor, args):
            raise RuntimeError("{mutable_op}: expected all args to be FunctionalTensorWrapper")

        # 对参数进行解包，并将功能性张量转换为普通张量
        unwrapped_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and torch._is_functional_tensor(arg):
                torch._sync(arg)
                unwrapped = torch._from_functional_tensor(arg)
                unwrapped_args.append(unwrapped)
            else:
                unwrapped_args.append(arg)

        # 使用功能性操作函数functional_op处理解包后的参数
        with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Functionalize)):
            output = functional_op(*unwrapped_args)

        # 获取mutable_op的实际输出数量
        num_actual_output = len(mutable_op._schema.returns)
        # 将输出转换为功能性张量，并根据需要进行切片
        actual_output = pytree.tree_map(
            torch._to_functional_tensor, output[:num_actual_output])

        # 提取新的值用于传播，并替换相应的输入参数
        new_values_to_propagate = output[num_actual_output:]
        inputs_to_replace = [arg for is_write, arg in zip(mutable_args(mutable_op), args)
                             if is_write]
        assert len(new_values_to_propagate) == len(inputs_to_replace)
        for new_value, arg in zip(new_values_to_propagate, inputs_to_replace):
            if (arg is None and new_value is None) or (arg is not None and new_value is not None):
                continue
            torch._C._propagate_xla_data(arg, new_value)
            torch._C._replace_(arg, new_value)
            torch._C._commit_update(arg)
            torch._sync(arg)

        # 根据actual_output的长度决定返回值
        if len(actual_output) == 1:
            return actual_output[0]
        elif len(actual_output) == 0:
            return None
        return actual_output

    return kernel


# 验证函数validate，确保mutable_op是OpOverload的实例
def validate(mutable_op: OpOverload):
    if not isinstance(mutable_op, OpOverload):
        raise TypeError(
            f"register_functional_op(mutable_op): expected mutable_op to be instance of "
            f"OpOverload but got {type(mutable_op)}")

    # 解释三种“原地”或“可变”操作的类型及其约定
    # - inplace（第一个输入在原地修改并作为唯一输出返回）
    # - out=（一些参数在原地修改并作为输出返回）
    # 解析函数的 schema，从字符串形式的 mutable_op._schema 中解析出来
    schema = FunctionSchema.parse(str(mutable_op._schema))
    # 检查 schema 的类型是否为 mutable，如果不是则抛出 RuntimeError
    if not schema.kind() == SchemaKind.mutable:
        raise RuntimeError("Expected op to be mutable (as opposed to functional, inplace or out)")
    # 检查 schema 返回的每个值，如果有注解则抛出 NotImplementedError
    for ret in schema.returns:
        # construct_functionalization_kernel 简化假设
        if ret.annotation is not None:
            raise NotImplementedError(
                "NYI: register_functional_op(op) where op returns a mutated or aliased value. "
                "Please file an issue (and as a workaround, modify your operator to "
                "not return the mutated value or aliases)")
    # 检查 schema 的每个参数，如果参数是张量并且不是 BaseType(BaseTy.Tensor) 或 OptionalType(BaseType(BaseTy.Tensor))，则抛出 NotImplementedError
    for arg in schema.arguments.flat_all:
        # construct_functionalization_kernel 简化假设
        if arg.type.is_tensor_like() and (
            arg.type != BaseType(BaseTy.Tensor)
            and arg.type != OptionalType(BaseType(BaseTy.Tensor))
        ):
            raise NotImplementedError(
                "NYI: register_functional_op(op) where op has a List[Tensor] input."
                "Please file an issue.")
# 根据新的操作名和操作重载对象生成函数模式的字符串表示
def functional_schema(new_op_name, op: OpOverload):
    # 解析操作的函数模式，转换为FunctionSchema对象
    schema = FunctionSchema.parse(str(op._schema))
    # 使用新的操作名更新函数签名中的操作名
    schema = schema.signature().with_name(OperatorName.parse(new_op_name))
    # 返回更新后的函数模式字符串表示
    return str(schema)


# 根据操作重载对象获取其所有可变参数的写入标志，以元组形式返回
def mutable_args(op: OpOverload):
    # 使用生成器表达式，对每个参数检查其别名信息是否存在并确定其是否可写
    return tuple(False if arg.alias_info is None else arg.alias_info.is_write
                 for arg in op._schema.arguments)
```