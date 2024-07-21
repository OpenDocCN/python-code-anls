# `.\pytorch\torch\_compile.py`

```py
"""
APIs related to torch.compile which lazily import torch._dynamo to avoid
circular dependencies.
"""
import functools

# 定义一个函数 _disable_dynamo，它允许在不产生循环依赖的情况下延迟导入 torch._dynamo。
def _disable_dynamo(fn=None, recursive=True):
    """
    This API should be only used inside torch, external users should still use
    torch._dynamo.disable. The main goal of this API is to avoid circular
    imports issues that is common while using _dynamo.disable inside torch
    itself.

    This API avoids it by lazily importing torch._dynamo from the import time to
    the invocation of the decorated function.
    """
    # 如果传入了函数 fn
    if fn is not None:

        @functools.wraps(fn)
        # 定义内部函数 inner，用于装饰传入的函数 fn
        def inner(*args, **kwargs):
            # 在首次调用时缓存此函数以避免添加过多开销。
            disable_fn = getattr(fn, "__dynamo_disable", None)
            # 如果没有缓存过这个函数
            if disable_fn is None:
                # 延迟导入 torch._dynamo，避免循环导入问题
                import torch._dynamo
                # 调用 torch._dynamo.disable 创建一个新的函数并缓存起来
                disable_fn = torch._dynamo.disable(fn, recursive)
                # 将生成的新函数缓存到 fn 的 __dynamo_disable 属性中
                fn.__dynamo_disable = disable_fn

            # 返回缓存的函数并执行
            return disable_fn(*args, **kwargs)

        # 返回装饰后的函数 inner
        return inner
    else:
        # 如果没有传入函数 fn，说明是作为装饰器使用，像 @_disable_dynamo(recursive=False)
        # 返回一个偏函数，参数为 _disable_dynamo 的递归值设定为传入的值 recursive
        return functools.partial(_disable_dynamo, recursive=recursive)
```