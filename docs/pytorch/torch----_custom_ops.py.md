# `.\pytorch\torch\_custom_ops.py`

```py
# 设置类型检查为允许未标记的函数
# 这是一个特殊指令，告诉类型检查器（如mypy），允许在未显式标记类型的情况下定义函数
# 通常用于需要动态或特定方式定义的情况

import inspect  # 导入inspect模块，用于获取对象信息

from torch._custom_op.impl import (
    _custom_op_with_schema,  # 导入定义具有模式的自定义操作的功能
    _find_custom_op,  # 导入查找自定义操作的功能
    infer_schema,  # 导入推断操作模式的功能
    parse_qualname,  # 导入解析操作合格名称的功能
    validate_namespace,  # 导入验证命名空间的功能
)

from torch.library import get_ctx  # 从torch库中导入get_ctx函数

__all__ = [  # 定义可导出的模块成员列表
    "custom_op",  # 自定义操作函数
    "impl",  # 实现操作
    "impl_abstract",  # 实现抽象操作
    "get_ctx",  # 获取上下文
    "impl_save_for_backward",  # 为反向传播保存实现
    "impl_backward",  # 后向传播实现
]


def custom_op(qualname, func_or_schema=None):
    r"""Register a new custom operator

    In PyTorch, defining an op (short for "operator") is a two step-process:
    - we need to define the op (by providing an operator name and schema)
    - we need to implement behavior for how the operator interacts with
      various PyTorch subsystems, like CPU/CUDA Tensors, Autograd, etc.

    This entrypoint defines the custom operator (the first step)
    you must then perform the second step by calling various
    ``impl_*`` APIs.

    This API may be used as a decorator (see examples).

    For a detailed guide on custom ops, please see
    https://docs.google.com/document/d/1aGWtgxV3HppuxQAdddyPrs74_aEntpkYt9MalnCKnhk

    Arguments:
        qualname (str): Should be a string that looks like
            "namespace::operator_name". Operators in PyTorch need a namespace to
            avoid name collisions; a given operator may only be created once.
            If you are writing a Python library, we recommend the namespace to
            be the name of your top-level module.
        func_or_schema (Union[Callable, str]): Each PyTorch operator needs a
            schema that tells PyTorch the types of the inputs/outputs.
            If this is a Callable, we will automatically infer the schema from
            the type annotations on the function (see examples). Otherwise,
            if you don't want to use type annotations, you may provide us the
            schema string.

    """
    # 注册一个新的自定义操作

    # qualname参数应该是一个类似于 "namespace::operator_name" 的字符串
    # PyTorch中的操作需要一个命名空间来避免名称冲突；一个给定的操作只能创建一次

    # 如果func_or_schema是可调用对象，则自动推断模式并注册操作
    # 否则，需要提供模式字符串来定义操作

    # 详细使用示例和自定义操作指南请参考提供的文档链接
    ns, name = parse_qualname(qualname)
    # 解析限定名称，获取命名空间和名称

    validate_namespace(ns)
    # 验证命名空间的有效性

    def inner(func):
        # 定义内部函数 `inner`，接受一个函数作为参数 `func`

        if not inspect.isfunction(func):
            # 如果参数 `func` 不是一个函数对象，则抛出数值错误异常
            raise ValueError(
                f"custom_op(...)(func): Expected `func` to be a Python "
                f"function, got: {type(func)}"
            )

        if func.__name__ != name:
            # 如果函数对象的名称不等于预期的名称 `name`，则抛出数值错误异常
            raise ValueError(
                f"custom_op(qualname='{qualname}', ...)(func): expected `func` "
                f"to have name '{name}' but got '{func.__name__}'. "
                f"Please either change the name of `func` or the qualname that "
                f"is passed to `custom_op`"
            )

        # 推断函数 `func` 的模式
        schema = infer_schema(func)
        # 使用推断的模式调用 `_custom_op_with_schema` 函数，传递限定名称和推断的模式
        _custom_op_with_schema(qualname, schema)
        # 返回函数 `func`
        return func

    if func_or_schema is None:
        # 如果 `func_or_schema` 参数为空，则返回内部函数 `inner`
        return inner

    if isinstance(func_or_schema, str):
        # 如果 `func_or_schema` 是字符串类型，则调用 `_custom_op_with_schema` 函数，
        # 传递限定名称和 `func_or_schema` 字符串
        _custom_op_with_schema(qualname, func_or_schema)
    else:
        # 否则，调用内部函数 `inner`，并传递 `func_or_schema` 作为参数
        return inner(func_or_schema)
# 注册一个设备类型的自定义操作实现函数
def impl(qualname, *, device_types=("cpu", "cuda"), func=None):
    """
    Register an implementation for a device type for this custom op.

    If the op is passed multiple Tensor inputs with different device
    types, it will dispatch to the registered implementation for the highest
    priority device type among those present.
    The supported device types, in order of priority, are {'cuda', 'cpu'}.

    This API may be used as a decorator (see examples).

    For a detailed guide on custom ops, please see
    https://docs.google.com/document/d/1aGWtgxV3HppuxQAdddyPrs74_aEntpkYt9MalnCKnhk

    Arguments:
        device_types (str or Iterable[str]): the device type(s) to register the function for.
    """

    # 定义内部函数 inner，用于注册具体的实现函数
    def inner(func):
        # 查找特定名称的自定义操作，包括 Torch 库中的检查
        custom_op = _find_custom_op(qualname, also_check_torch_library=True)
        # 在自定义操作中注册指定设备类型的实现函数
        custom_op.impl(device_types, _stacklevel=3)(func)
        return func

    # 如果没有提供实现函数，则返回内部函数 inner，允许作为装饰器使用
    if func is None:
        return inner
    # 否则直接注册实现函数并返回
    return inner(func)


def impl_abstract(qualname, *, func=None):
    """
    Register an abstract implementation for this operator.

    An "abstract implementation" specifies the behavior of this operator on
    Tensors that carry no data. Given some input Tensors with certain properties
    (sizes/strides/storage_offset/device), it specifies what the properties of
    the output Tensors are.

    The abstract implementation has the same signature as the operator.
    """
    # 导入 torch.library 模块，用于自定义操作的注册
    import torch.library

    # 使用 torch.library.register_fake 注册一个假的操作（fake operation），实现功能的模拟
    return torch.library.register_fake(qualname, func, _stacklevel=2)
# 注册一个用于保存反向传播所需信息的函数。该函数告知系统在反向传播时需要保存哪些信息。
# 请参考 :func:`impl_backward` 获取更多细节。
def impl_save_for_backward(qualname, *, func=None):
    """Register a function that tells us what to save for backward.

    Please see :func:`impl_backward` for more details.
    """

    # 内部函数，用于实际注册保存反向传播信息的逻辑
    def inner(func):
        # 查找自定义操作，也可以检查 Torch 库
        custom_op = _find_custom_op(qualname, also_check_torch_library=True)
        # 调用自定义操作的实现保存反向传播信息方法，将 func 作为参数传递
        custom_op.impl_save_for_backward(_stacklevel=3)(func)
        return func

    # 如果没有传入 func 参数，则返回内部函数 inner
    if func is None:
        return inner
    # 否则直接调用内部函数 inner，传入 func 参数
    return inner(func)


# 注册一个操作符的反向传播公式
def impl_backward(qualname, output_differentiability=None, *, func=None):
    """Registers a backward formula for an operator.

    In order for an operator to work with autograd, you need to register
    a backward formula. There are two pieces to this:
    1. You must give us a function to specify what to save for backward.
       Call this the "save for backward" function.
    2. You must give us a function that computes gradients. Call this the
       "backward" function.

    Use `impl_save_for_backward` to define a "save for backward" function
    that specifies what gets saved for backward. The function should accept
    two arguments ``(inputs, output)`` and return the quantities to be saved
    for backward.

    During runtime, when you call the operator in a forwards pass, PyTorch
    will invoke the "save for backward" function with the inputs and output
    of the operator.

    Use `impl_backward` to define the "backward" function. The backward
    function must accept ``(ctx, saved, *grads)``:
    - ``ctx`` is a context object where we may provide information
    - ``saved`` is exactly what gets returned from the "save for backward"
      function
    - ``grads`` is one or more gradients. The number of gradients matches
      the number of outputs of the operator.

    The backward function must return a dict that maps the name of
    an input to the operator to its corresponding gradient. All inputs that
    were declared to be Tensors in the operator definition must be accounted
    for in the dict. The gradient may be a Tensor or None.

    For a detailed guide on custom ops, please see
    https://docs.google.com/document/d/1aGWtgxV3HppuxQAdddyPrs74_aEntpkYt9MalnCKnhk

    """

    # 内部函数，用于实际注册操作符的反向传播公式
    def inner(func):
        # 查找自定义操作，也可以检查 Torch 库
        custom_op = _find_custom_op(qualname, also_check_torch_library=True)
        # 调用自定义操作的实现反向传播方法，将 func 作为参数传递
        custom_op.impl_backward(output_differentiability, _stacklevel=3)(func)
        return func

    # 如果没有传入 func 参数，则返回内部函数 inner
    if func is None:
        return inner
    # 否则直接调用内部函数 inner，传入 func 参数
    return inner(func)


# 取消注册一个自定义操作符。仅供测试目的使用。
def _destroy(qualname):
    """De-registers a custom op. For testing purposes only"""
    # 查找自定义操作
    custom_op = _find_custom_op(qualname)
    # 调用自定义操作的销毁方法
    custom_op._destroy()
```