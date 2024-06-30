# `D:\src\scipysrc\scikit-learn\sklearn\utils\parallel.py`

```
"""Customizations of :mod:`joblib` and :mod:`threadpoolctl` tools for scikit-learn
usage.
"""

# 引入 functools 和 warnings 模块
import functools
import warnings
# 从 functools 模块中引入 update_wrapper 函数
from functools import update_wrapper

# 引入 joblib 库
import joblib
# 从 threadpoolctl 库中引入 ThreadpoolController 类
from threadpoolctl import ThreadpoolController

# 从内部模块引入配置相关函数和对象
from .._config import config_context, get_config

# 全局线程池控制器实例，用于限制本地线程数，应通过 _get_threadpool_controller 函数访问
_threadpool_controller = None


def _with_config(delayed_func, config):
    """Helper function that intends to attach a config to a delayed function."""
    # 如果 delayed_func 已经有 with_config 属性，则调用其 with_config 方法传入 config
    if hasattr(delayed_func, "with_config"):
        return delayed_func.with_config(config)
    else:
        # 否则发出警告并返回原始 delayed_func 函数
        warnings.warn(
            (
                "`sklearn.utils.parallel.Parallel` needs to be used in "
                "conjunction with `sklearn.utils.parallel.delayed` instead of "
                "`joblib.delayed` to correctly propagate the scikit-learn "
                "configuration to the joblib workers."
            ),
            UserWarning,
        )
        return delayed_func


class Parallel(joblib.Parallel):
    """Tweak of :class:`joblib.Parallel` that propagates the scikit-learn configuration.

    This subclass of :class:`joblib.Parallel` ensures that the active configuration
    (thread-local) of scikit-learn is propagated to the parallel workers for the
    duration of the execution of the parallel tasks.

    The API does not change and you can refer to :class:`joblib.Parallel`
    documentation for more details.

    .. versionadded:: 1.3
    """

    def __call__(self, iterable):
        """Dispatch the tasks and return the results.

        Parameters
        ----------
        iterable : iterable
            Iterable containing tuples of (delayed_function, args, kwargs) that should
            be consumed.

        Returns
        -------
        results : list
            List of results of the tasks.
        """
        # 捕获当前时间点的 scikit-learn 配置，因为任务可能会在不同线程中分发执行
        config = get_config()
        # 将 iterable 中的每个 delayed_func 都附上当前配置并重新组织成新的迭代器 iterable_with_config
        iterable_with_config = (
            (_with_config(delayed_func, config), args, kwargs)
            for delayed_func, args, kwargs in iterable
        )
        # 调用父类的 __call__ 方法执行任务并返回结果
        return super().__call__(iterable_with_config)


# 当 https://github.com/joblib/joblib/issues/1071 问题修复后，移除以下函数
def delayed(function):
    """Decorator used to capture the arguments of a function.

    This alternative to `joblib.delayed` is meant to be used in conjunction
    with `sklearn.utils.parallel.Parallel`. The latter captures the scikit-
    learn configuration by calling `sklearn.get_config()` in the current
    ```
    # 使用装饰器将函数包装成延迟执行的函数，以便在后台工作线程中执行
    # 被延迟执行的函数的配置在调度第一个任务之前被捕获
    # 然后在 joblib 工作者中启用并传播配置，直到延迟函数执行完毕

    # 版本变更说明：从 scikit-learn 1.3 开始，将 `delayed` 从 `sklearn.utils.fixes` 移动到 `sklearn.utils.parallel`

    Parameters
    ----------
    function : callable
        要延迟执行的函数

    Returns
    -------
    output: tuple
        包含延迟函数、位置参数和关键字参数的元组
    """
    
    # 使用 functools.wraps 装饰器保留原始函数的元数据
    @functools.wraps(function)
    def delayed_function(*args, **kwargs):
        # 返回一个元组，包含函数的包装器 _FuncWrapper，位置参数 args，和关键字参数 kwargs
        return _FuncWrapper(function), args, kwargs

    # 返回包装后的延迟执行函数
    return delayed_function
class _FuncWrapper:
    """Load the global configuration before calling the function."""

    def __init__(self, function):
        # 初始化函数包装器，接收一个函数作为参数并保存
        self.function = function
        # 更新函数包装器的属性以便于访问函数的元数据
        update_wrapper(self, self.function)

    def with_config(self, config):
        # 设置函数包装器的配置参数，并返回自身以支持链式调用
        self.config = config
        return self

    def __call__(self, *args, **kwargs):
        # 获取函数包装器的配置参数，如果不存在则发出警告并使用空字典作为默认配置
        config = getattr(self, "config", None)
        if config is None:
            warnings.warn(
                (
                    "`sklearn.utils.parallel.delayed` should be used with"
                    " `sklearn.utils.parallel.Parallel` to make it possible to"
                    " propagate the scikit-learn configuration of the current thread to"
                    " the joblib workers."
                ),
                UserWarning,
            )
            config = {}
        # 在配置上下文中执行函数，确保函数在正确的配置环境中运行
        with config_context(**config):
            return self.function(*args, **kwargs)


def _get_threadpool_controller():
    """Return the global threadpool controller instance."""
    # 获取全局线程池控制器实例，如果不存在则创建一个新的实例
    global _threadpool_controller

    if _threadpool_controller is None:
        _threadpool_controller = ThreadpoolController()

    return _threadpool_controller


def _threadpool_controller_decorator(limits=1, user_api="blas"):
    """Decorator to limit the number of threads used at the function level.

    It should be prefered over `threadpoolctl.ThreadpoolController.wrap` because this
    one only loads the shared libraries when the function is called while the latter
    loads them at import time.
    """
    # 定义装饰器函数，用于在函数级别限制线程数

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取线程池控制器实例，并在限制范围内执行函数
            controller = _get_threadpool_controller()
            with controller.limit(limits=limits, user_api=user_api):
                return func(*args, **kwargs)

        return wrapper

    return decorator
```