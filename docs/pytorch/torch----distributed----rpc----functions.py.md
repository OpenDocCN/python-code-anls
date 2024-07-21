# `.\pytorch\torch\distributed\rpc\functions.py`

```
# 引入 functools 模块，用于函数装饰器
import functools

# 定义异步执行函数的装饰器
def async_execution(fn):
    r"""
    一个函数装饰器，表示函数的返回值保证是一个 :class:`~torch.futures.Future` 对象，
    并且该函数可以在 RPC 调用者端异步运行。具体来说，被调用方提取由装饰函数返回的
    :class:`~torch.futures.Future`，并将后续处理步骤安装为该 :class:`~torch.futures.Future` 的回调。
    安装的回调将在完成时从 :class:`~torch.futures.Future` 中读取值，并将该值作为 RPC 响应发送回去。
    这也意味着返回的 :class:`~torch.futures.Future` 只存在于被调用方，并且从不通过 RPC 发送。
    此装饰器在被装饰函数（``fn``）的执行需要暂停和恢复时（例如包含 :meth:`~torch.distributed.rpc.rpc_async`
    或等待其他信号时）非常有用。

    .. 注意:: 要启用异步执行，应用程序必须将由此装饰器返回的函数对象传递给 RPC API。
        如果 RPC 检测到由此装饰器安装的属性，则知道该函数返回一个 ``Future`` 对象，并将相应地处理它。
        然而，这并不意味着在定义函数时此装饰器必须是最外层的。例如，与 ``@staticmethod`` 或 ``@classmethod``
        结合时，``@rpc.functions.async_execution`` 需要成为内部装饰器，以允许目标函数被识别为静态或类函数。
        此目标函数仍然可以异步执行，因为在访问时，静态或类方法会保留由 ``@rpc.functions.async_execution``
        安装的属性。

    """

    # 创建装饰函数的包装器，使用 functools.wraps 来保留原始函数的元数据
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # 执行原始函数，并返回其结果
        return fn(*args, **kwargs)

    # 由于 mypy 无法声明并使用函数对象的属性（mypy#2087），因此通过类型忽略声明装饰函数的属性
    wrapper._wrapped_async_rpc_function = fn  # type: ignore[attr-defined]
    return wrapper
```