# `.\pytorch\torch\_prims\executor.py`

```py
# mypy: allow-untyped-defs
# 引入必要的模块和函数
from typing import Callable, Optional

from torch._prims.context import TorchRefsMode  # 导入 TorchRefsMode 类

from torch.fx import GraphModule  # 导入 GraphModule 类
from torch.fx.experimental.proxy_tensor import make_fx, wrapper_and_args_for_make_fx  # 导入 make_fx 和 wrapper_and_args_for_make_fx 函数


def execute(
    gm: GraphModule,
    *args,
    executor: str = "aten",
    executor_parameters: Optional[dict] = None,
):
    """
    Prototype ATen executor.

    Just executes the context's graph.
    """

    # 根据执行器类型选择执行方式
    if executor == "aten":
        return gm.forward(*args)

    # 如果执行器类型不是 'aten'，则抛出错误
    msg = f"Received unexpected value for 'executor': {executor}. Allowed values are: aten."
    raise ValueError(msg)


def make_traced(fn: Callable):
    """
    Returns a function that, when called, will
    trace its torch operations to prims and then
    execute those prims on the requested trace executor
    (possibly lowering them to that trace executor first).

    Only supports the torch operations defined in _torch_to_reference_map
    in context.py and operations with positional args. All args must
    be tensors.
    In the near future all these restrictions will be lifted.

    Example usage:

    def foo(a, b):
      return torch.add(a, b)

    traced_foo = make_traced(foo)

    a = torch.randn((1, 2, 3, 4, 5), device='cuda')
    b = torch.randn((1, 2, 3, 4, 5), device='cuda')
    result = traced_foo(a, b, executor='aten')
    """

    def _traced(*args, executor="aten", **kwargs):
        # TODO: caching
        # 使用 wrapper_and_args_for_make_fx 函数封装函数及其参数
        wrapped, all_args = wrapper_and_args_for_make_fx(fn, args, kwargs)

        # 使用 TorchRefsMode 上下文管理器
        with TorchRefsMode():
            # 使用 make_fx 函数创建图模块
            gm = make_fx(wrapped)(all_args)
        
        # 调用 execute 函数执行图模块并返回结果
        return execute(gm, all_args, executor=executor)

    return _traced
```