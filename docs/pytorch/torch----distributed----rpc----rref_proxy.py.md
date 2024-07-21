# `.\pytorch\torch\distributed\rpc\rref_proxy.py`

```py
# mypy: allow-untyped-defs
# 从 functools 模块导入 partial 函数，用于创建偏函数
from functools import partial

# 导入 torch 库
import torch
# 从 torch.futures 模块导入 Future 类
from torch.futures import Future

# 从当前包中导入 functions 和 rpc_async 模块
from . import functions, rpc_async
# 从 constants 模块中导入 UNSET_RPC_TIMEOUT 常量
from .constants import UNSET_RPC_TIMEOUT

# 定义函数 _local_invoke，用于在本地调用远程引用 rref 的方法 func_name
def _local_invoke(rref, func_name, args, kwargs):
    return getattr(rref.local_value(), func_name)(*args, **kwargs)

# 使用 functions 模块的 async_execution 装饰器修饰函数 _local_invoke_async_execution
@functions.async_execution
def _local_invoke_async_execution(rref, func_name, args, kwargs):
    return getattr(rref.local_value(), func_name)(*args, **kwargs)

# 定义函数 _invoke_rpc，用于执行远程过程调用
def _invoke_rpc(rref, rpc_api, func_name, timeout, *args, **kwargs):
    # 定义内部函数 _rref_type_cont，处理远程引用类型和调用
    def _rref_type_cont(rref_fut):
        # 获取远程引用的类型
        rref_type = rref_fut.value()

        # 默认使用本地调用函数 _local_invoke
        _invoke_func = _local_invoke

        # 检查是否绕过 ScriptModule 类型，避免异步函数属性检查
        bypass_type = issubclass(rref_type, torch.jit.ScriptModule) or issubclass(
            rref_type, torch._C.ScriptModule
        )

        # 如果不是绕过类型，则尝试获取 func_name 对应的函数
        if not bypass_type:
            func = getattr(rref_type, func_name)
            # 如果 func 有 "_wrapped_async_rpc_function" 属性，则使用异步调用函数 _local_invoke_async_execution
            if hasattr(func, "_wrapped_async_rpc_function"):
                _invoke_func = _local_invoke_async_execution

        # 调用 rpc_api 方法，执行远程调用
        return rpc_api(
            rref.owner(),
            _invoke_func,
            args=(rref, func_name, args, kwargs),
            timeout=timeout,
        )

    # 获取远程引用的类型的 Future 对象
    rref_fut = rref._get_type(timeout=timeout, blocking=False)

    # 如果 rpc_api 不等于 rpc_async，则等待 rref_fut 完成后执行 _rref_type_cont
    if rpc_api != rpc_async:
        rref_fut.wait()
        return _rref_type_cont(rref_fut)
    else:
        # 如果 rpc_api 是 rpc_async，则创建一个 Future 对象 result
        result: Future = Future()

        # 定义内部函数 _wrap_rref_type_cont，处理 Future 对象的 then 方法
        def _wrap_rref_type_cont(fut):
            try:
                # 调用 _rref_type_cont 函数，并在其返回的 Future 对象上执行 _complete_op
                _rref_type_cont(fut).then(_complete_op)
            except BaseException as ex:
                result.set_exception(ex)

        # 定义内部函数 _complete_op，处理 Future 对象的结果或异常
        def _complete_op(fut):
            try:
                result.set_result(fut.value())
            except BaseException as ex:
                result.set_exception(ex)

        # 在 rref_fut 上调用 then 方法，执行 _wrap_rref_type_cont
        rref_fut.then(_wrap_rref_type_cont)
        # 返回 Future 对象 result，用于异步调用结果的获取
        return result


# 此类用于管理 RRefs 的代理 RPC API 调用，完全由 C++ (python_rpc_handler.cpp) 使用。
class RRefProxy:
    # 初始化方法，接收远程引用 rref、RPC API rpc_api 和超时时间 timeout
    def __init__(self, rref, rpc_api, timeout=UNSET_RPC_TIMEOUT):
        self.rref = rref
        self.rpc_api = rpc_api
        self.rpc_timeout = timeout

    # 定义 __getattr__ 方法，用于动态获取属性 func_name 并返回偏函数 _invoke_rpc
    def __getattr__(self, func_name):
        return partial(
            _invoke_rpc, self.rref, self.rpc_api, func_name, self.rpc_timeout
        )
```