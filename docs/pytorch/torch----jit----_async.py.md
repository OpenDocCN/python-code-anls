# `.\pytorch\torch\jit\_async.py`

```
# mypy: allow-untyped-defs
"""Async API.

This module contains the API for parallelism in TorchScript, notably:
    * torch.jit.fork
    * torch.jit.wait

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""

import torch
from torch._jit_internal import Future  # 导入 Future 类型用于表示异步任务的结果
from torch.jit._builtins import _register_builtin  # 导入内置函数 _register_builtin

from torch.utils import set_module

set_module(Future, "torch.jit")  # 设置 Future 模块的命名空间为 "torch.jit"


def fork(func, *args, **kwargs):
    r"""
    Create an asynchronous task executing `func` and a reference to the value of the result of this execution.

    `fork` will return immediately, so the return value of `func` may not have been computed yet. To force completion
    of the task and access the return value invoke `torch.jit.wait` on the Future. `fork` invoked
    with a `func` which returns `T` is typed as `torch.jit.Future[T]`. `fork` calls can be arbitrarily
    nested, and may be invoked with positional and keyword arguments.
    Asynchronous execution will only occur when run in TorchScript. If run in pure python,
    `fork` will not execute in parallel. `fork` will also not execute in parallel when invoked
    while tracing, however the `fork` and `wait` calls will be captured in the exported IR Graph.

    .. warning::
        `fork` tasks will execute non-deterministically. We recommend only spawning
        parallel fork tasks for pure functions that do not modify their inputs,
        module attributes, or global state.

    Args:
        func (callable or torch.nn.Module):  A Python function or `torch.nn.Module`
            that will be invoked. If executed in TorchScript, it will execute asynchronously,
            otherwise it will not. Traced invocations of fork will be captured in the IR.
        ``*args``, ``**kwargs``: arguments to invoke `func` with.
    Returns:
        `torch.jit.Future[T]`: a reference to the execution of `func`. The value `T`
        can only be accessed by forcing completion of `func` through `torch.jit.wait`.

    Example (fork a free function):

    .. code-block:: python

        import torch
        from torch import Tensor
        def foo(a : Tensor, b : int) -> Tensor:
            return a + b
        def bar(a):
            fut : torch.jit.Future[Tensor] = torch.jit.fork(foo, a, b=2)
            return torch.jit.wait(fut)
        script_bar = torch.jit.script(bar)
        input = torch.tensor(2)
        # only the scripted version executes asynchronously
        assert script_bar(input) == bar(input)
        # trace is not run asynchronously, but fork is captured in IR
        graph = torch.jit.trace(bar, (input,)).graph
        assert "fork" in str(graph)

    Example (fork a module method):
    # 导入 torch 库，用于深度学习任务
    import torch
    # 导入 Tensor 类型，用于定义张量数据
    from torch import Tensor
    
    # 创建 AddMod 类，继承自 torch.nn.Module，用于实现加法操作
    class AddMod(torch.nn.Module):
        
        # 定义前向传播方法，接收一个 Tensor 类型的参数 a 和一个整数类型的参数 b
        def forward(self, a: Tensor, b : int):
            # 返回 a 和 b 相加的结果
            return a + b
    
    # 创建 Mod 类，继承自 torch.nn.Module，用于实现模块化的操作
    class Mod(torch.nn.Module):
        
        # 初始化方法
        def __init__(self):
            # 调用父类的初始化方法
            super(self).__init__()
            # 创建 AddMod 类的实例，作为该模块的一个成员变量
            self.mod = AddMod()
    
        # 定义前向传播方法，接收一个输入 input
        def forward(self, input):
            # 使用 torch.jit.fork 方法异步执行 self.mod 的前向传播，传入参数 a=input 和 b=2
            fut = torch.jit.fork(self.mod, a, b=2)
            # 等待异步执行的结果 fut 返回
            return torch.jit.wait(fut)
    
    # 创建一个张量 input，其值为 2
    input = torch.tensor(2)
    # 创建 Mod 类的实例 mod
    mod = Mod()
    # 使用断言验证 mod(input) 的结果与 torch.jit.script(mod).forward(input) 的结果相等
    assert mod(input) == torch.jit.script(mod).forward(input)
def wait(future):
    r"""
    Force completion of a `torch.jit.Future[T]` asynchronous task, returning the result of the task.

    See :func:`~fork` for docs and examples.
    Args:
        future (torch.jit.Future[T]): an asynchronous task reference, created through `torch.jit.fork`
    Returns:
        `T`: the return value of the completed task
    """
    # 调用 torch._C.wait 方法等待异步任务 future 完成，并返回任务的结果
    return torch._C.wait(future)


# 将 wait 函数注册为内置函数，名称为 "aten::wait"
_register_builtin(wait, "aten::wait")
```