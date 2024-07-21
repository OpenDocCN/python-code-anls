# `.\pytorch\torch\_dynamo\external_utils.py`

```
# mypy: allow-untyped-defs
# This module contains functions that *will be allowed* by dynamo

import functools  # 导入 functools 模块，用于高阶函数的操作
from typing import List  # 导入 List 类型提示，用于声明列表类型

import torch  # 导入 PyTorch 库
import torch.utils._pytree as pytree  # 导入 PyTorch 的内部工具模块 _pytree

try:
    import numpy as np  # 尝试导入 NumPy 库
except ModuleNotFoundError:
    np = None  # type: ignore[assignment] 如果导入失败，将 np 设置为 None，忽略类型检查


def is_compiling() -> bool:
    """
    Indicates whether we are tracing/compiling with torch.compile() or torch.export().

    If need to check specifically that TorchDynamo is used, then use
    torch.compiler.is_dynamo_compiling().

    TODO(khabinov): we should deprecate this function and use one of these two:
    * torch.compiler.is_compiling(),
    * torch.compiler.is_dynamo_compiling().
    It will depend on the context where to use what.
    """
    return torch.compiler.is_compiling()  # 返回当前是否正在编译或追踪的状态


def wrap_inline(fn):
    """
    Create an extra frame around fn that is not in skipfiles
    """

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        return fn(*args, **kwargs)

    return inner  # 返回一个装饰器，用于在函数周围创建额外的帧


def call_hook(hook, *args):
    """
    Used by compiled autograd to handle hook returning None
    """
    result = hook(*args)  # 调用 hook 函数并传入参数
    if result is None:
        return args[0]  # 如果 hook 返回 None，则返回第一个参数
    return result  # 否则返回 hook 的返回值


def wrap_numpy(f):
    r"""Decorator that turns a function from ``np.ndarray``s to ``np.ndarray``s into a function
    from ``torch.Tensor``s to ``torch.Tensor``s.
    """
    if not np:
        return f  # 如果没有成功导入 NumPy，则直接返回原函数 f

    @functools.wraps(f)
    def wrap(*args, **kwargs):
        args, kwargs = pytree.tree_map_only(
            torch.Tensor, lambda x: x.numpy(), (args, kwargs)
        )
        out = f(*args, **kwargs)  # 调用被装饰的函数 f
        return pytree.tree_map_only(np.ndarray, lambda x: torch.as_tensor(x), out)

    return wrap  # 返回一个转换了数据类型的函数装饰器


class FakeBackwardCFunction:
    def __init__(
        self,
        real: torch.autograd.function.BackwardCFunction,
        saved_tensors: List[torch.Tensor],
    ):
        self.real = real  # 初始化实际的 torch.autograd.function.BackwardCFunction 对象
        self.saved_tensors = saved_tensors  # 初始化保存的张量列表

    def __getattr__(self, name):
        # route any attribute that isn't defined on this obj
        return getattr(self.real, name)  # 如果没有定义在当前对象上的属性，就从 self.real 中获取


# This function corresponds to the "eager" implementation of a lifted autograd.Function.backward
def call_backward(backward_c_function, saved_tensors, *args):
    fake = FakeBackwardCFunction(backward_c_function, saved_tensors)
    grads = fake._forward_cls.backward(fake, *args)  # type: ignore[attr-defined]

    # in eager, we wrap in a tuple when there's only one grad output
    if type(grads) is not tuple:
        grads = (grads,)

    return grads  # 返回计算的梯度，通常是一个元组


def untyped_storage_size(x: torch.Tensor):
    return x.untyped_storage().size()  # 返回张量 x 的未命名存储的大小


class FakeCompiledAutogradEngine:
    @staticmethod
    def queue_callback(final_callbacks, cb):
        final_callbacks.append(cb)  # 将回调函数 cb 加入最终回调列表 final_callbacks 中

    @staticmethod
    def exec_final_callbacks(final_callbacks):
        i = 0
        while i < len(final_callbacks):
            cb = final_callbacks[i]
            cb()  # 执行回调函数 cb
            i += 1
        final_callbacks.clear()  # 清空最终回调列表
    # 定义一个名为 _exec_final_callbacks_stub 的函数，用于执行最终回调函数的占位符
    def _exec_final_callbacks_stub():
        # 这里使用 pass 语句，表示函数暂时不执行任何操作，只是占据了函数体
        pass
# 从 backward state 中调用指定的钩子函数
def call_hook_from_backward_state(*args, bw_state, hook_name: str, **kwargs):
    return getattr(bw_state, hook_name)(*args, **kwargs)

# 从 backward state 中调用指定模块的一组钩子函数
def call_module_hooks_from_backward_state(
    _, result, *args, bw_state, hooks_name: str, module_name: str
):
    # 获取 backward state 中指定名称的模块对象
    module = getattr(bw_state, module_name)
    # 获取 backward state 中指定名称的钩子函数列表
    hooks = getattr(bw_state, hooks_name)
    # 遍历钩子函数列表，依次调用每个钩子函数，并传递模块对象、结果以及其他参数
    for hook in hooks:
        new_result = hook(module, result, *args)
        # 如果钩子函数返回了非空结果，则更新结果
        if new_result is not None:
            result = new_result
    # 返回最终结果
    return result
```