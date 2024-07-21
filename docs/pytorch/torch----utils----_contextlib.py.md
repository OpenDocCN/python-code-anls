# `.\pytorch\torch\utils\_contextlib.py`

```py
# mypy: allow-untyped-defs
# 允许不对函数进行类型注解的声明

# 提供对上下文管理器的额外实用工具，这些工具本应该是标准库的一部分，但实际上并不是

import functools  # 导入 functools 模块，用于函数操作的工具
import inspect  # 导入 inspect 模块，用于检查活动对象的工具
import warnings  # 导入 warnings 模块，用于警告控制的工具
import sys  # 导入 sys 模块，提供对解释器相关的操作的访问
from typing import Any, Callable, TypeVar, cast  # 导入类型提示相关的工具

# 用于注解 _DecoratorContextManager 装饰器的使用（例如 'no_grad' 和 'enable_grad'）
# 参考 https://mypy.readthedocs.io/en/latest/generics.html#declaring-decorators
FuncType = Callable[..., Any]
F = TypeVar('F', bound=FuncType)


def _wrap_generator(ctx_factory, func):
    """
    使用上下文管理器工厂包装每个生成器调用。

    输入应该是一个返回上下文管理器的函数，而不是上下文管理器本身，以处理一次性上下文管理器。
    """
    @functools.wraps(func)
    def generator_context(*args, **kwargs):
        gen = func(*args, **kwargs)

        # 生成器在 `yield` 处暂停和恢复执行，因此我们需要确保每次执行流程返回到包装的生成器时正确设置梯度模式，
        # 并在通过我们的 `yield` 返回给调用者时恢复梯度模式（参见 PR #49017）。
        try:
            # 向生成器发出 `None` 启动生成器
            with ctx_factory():
                response = gen.send(None)

            while True:
                try:
                    # 将响应转发给调用者并获取其下一个请求
                    request = yield response

                except GeneratorExit:
                    # 通知仍然活动的生成器其即将关闭
                    with ctx_factory():
                        gen.close()
                    raise

                except BaseException:
                    # 传播调用者抛出的异常
                    with ctx_factory():
                        response = gen.throw(*sys.exc_info())

                else:
                    # 将最后一个请求传递给生成器并获取其响应
                    with ctx_factory():
                        response = gen.send(request)

        # 我们让生成器的 `.throw` 或 `.send` 方法抛出的异常传播到我们的调用者，
        # 除了 StopIteration 之外的异常
        except StopIteration as e:
            # 生成器告知我们它已完成：获取其返回的值（如果有），并指示我们也完成了
            # 通过返回它来看待这个值（参见 Python 返回语句的文档）
            return e.value

    return generator_context


def context_decorator(ctx, func):
    """
    类似于 contextlib.ContextDecorator。

    但有以下区别：
    1. 通过包装实现而不是继承，因此适用于从 C 实现的上下文管理器，这些上下文管理器不容易从 Python 类继承
    2. 以直观的方式包装生成器（参考 https://bugs.python.org/issue37743）
    """
    检查传入的上下文管理器或工厂函数是否符合要求，并返回相应的装饰器函数。

    检查传入的 ctx 参数，如果同时是可调用对象和有效的上下文管理器（具有 __enter__ 方法），则抛出错误，因为无法确定应该只包装构造函数还是整个类的方法。

    如果 ctx 不是可调用对象，则定义一个函数 ctx_factory，用于返回 ctx。
    如果 ctx 是可调用对象，则直接使用 ctx 作为 ctx_factory。

    如果 func 是一个类（inspect.isclass(func) 返回 True），则抛出 RuntimeError，因为无法确定是应该包装类的构造函数还是所有方法，建议分别装饰每个方法。

    如果 func 是生成器函数（inspect.isgeneratorfunction(func) 返回 True），则调用 _wrap_generator 函数来包装生成器函数。

    否则，定义一个装饰器函数 decorate_context，使用 functools.wraps 装饰器保留原始函数的元数据信息，然后在函数体内使用上下文管理器 ctx_factory() 包裹 func 的执行，最后返回 func 的执行结果。

    返回 decorate_context 函数作为装饰器函数。
    """
    assert not (callable(ctx) and hasattr(ctx, '__enter__')), (
        f"Passed in {ctx} is both callable and also a valid context manager "
        "(has __enter__), making it ambiguous which interface to use.  If you "
        "intended to pass a context manager factory, rewrite your call as "
        "context_decorator(lambda: ctx()); if you intended to pass a context "
        "manager directly, rewrite your call as context_decorator(lambda: ctx)"
    )

    if not callable(ctx):
        def ctx_factory():
            return ctx
    else:
        ctx_factory = ctx

    if inspect.isclass(func):
        raise RuntimeError(
            "Cannot decorate classes; it is ambiguous whether or not only the "
            "constructor or all methods should have the context manager applied; "
            "additionally, decorating a class at definition-site will prevent "
            "use of the identifier as a conventional type.  "
            "To specify which methods to decorate, decorate each of them "
            "individually."
        )

    if inspect.isgeneratorfunction(func):
        return _wrap_generator(ctx_factory, func)

    @functools.wraps(func)
    def decorate_context(*args, **kwargs):
        with ctx_factory():
            return func(*args, **kwargs)

    return decorate_context
# 定义一个名为 _DecoratorContextManager 的类，用于将上下文管理器用作装饰器
class _DecoratorContextManager:
    """Allow a context manager to be used as a decorator."""

    # __call__ 方法使得实例可以像函数一样调用，用于处理被装饰的函数或类
    def __call__(self, orig_func: F) -> F:
        # 检查原始函数是否是类
        if inspect.isclass(orig_func):
            # 如果是类，则发出警告，类装饰将在将来的版本中禁用
            warnings.warn(
                "Decorating classes is deprecated and will be disabled in "
                "future versions. You should only decorate functions or methods. "
                "To preserve the current behavior of class decoration, you can "
                "directly decorate the `__init__` method and nothing else.",
                FutureWarning,
                stacklevel=2,
            )
            # 创建一个 lambda 函数，使得该类可以被直接调用，返回类的实例
            func = cast(F, lambda *args, **kwargs: orig_func(*args, **kwargs))
        else:
            # 如果原始函数不是类，则直接使用原始函数
            func = orig_func

        # 返回处理后的函数
        return cast(F, context_decorator(self.clone, func))

    # __enter__ 方法，用于进入上下文管理器
    def __enter__(self) -> None:
        raise NotImplementedError

    # __exit__ 方法，用于退出上下文管理器
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        raise NotImplementedError

    # clone 方法，用于返回当前类的克隆实例
    def clone(self):
        # override this method if your children class takes __init__ parameters
        return self.__class__()


# _NoParamDecoratorContextManager 类继承自 _DecoratorContextManager 类，允许上下文管理器作为不带括号的装饰器使用
class _NoParamDecoratorContextManager(_DecoratorContextManager):
    """Allow a context manager to be used as a decorator without parentheses."""

    # __new__ 方法用于创建新的实例，如果没有传入原始函数，则返回一个实例
    def __new__(cls, orig_func=None):
        if orig_func is None:
            return super().__new__(cls)
        # 如果传入了原始函数，则直接调用当前类的实例来处理原始函数
        return cls()(orig_func)
```