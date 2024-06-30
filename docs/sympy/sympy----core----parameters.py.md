# `D:\src\scipysrc\sympy\sympy\core\parameters.py`

```
"""Thread-safe global parameters"""

# 导入清除缓存函数
from .cache import clear_cache
# 导入上下文管理器
from contextlib import contextmanager
# 导入本地线程类
from threading import local

class _global_parameters(local):
    """
    Thread-local global parameters.

    Explanation
    ===========

    This class generates thread-local container for SymPy's global parameters.
    Every global parameters must be passed as keyword argument when generating
    its instance.
    A variable, `global_parameters` is provided as default instance for this class.

    WARNING! Although the global parameters are thread-local, SymPy's cache is not
    by now.
    This may lead to undesired result in multi-threading operations.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.core.cache import clear_cache
    >>> from sympy.core.parameters import global_parameters as gp

    >>> gp.evaluate
    True
    >>> x+x
    2*x

    >>> log = []
    >>> def f():
    ...     clear_cache()
    ...     gp.evaluate = False
    ...     log.append(x+x)
    ...     clear_cache()
    >>> import threading
    >>> thread = threading.Thread(target=f)
    >>> thread.start()
    >>> thread.join()

    >>> print(log)
    [x + x]

    >>> gp.evaluate
    True
    >>> x+x
    2*x

    References
    ==========

    .. [1] https://docs.python.org/3/library/threading.html

    """
    # 初始化方法，接受关键字参数并更新到实例的字典中
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    # 设置属性的方法，如果值有变化则清除缓存
    def __setattr__(self, name, value):
        if getattr(self, name) != value:
            clear_cache()
        return super().__setattr__(name, value)

# 创建全局参数对象，默认包含 evaluate=True, distribute=True, exp_is_pow=False
global_parameters = _global_parameters(evaluate=True, distribute=True, exp_is_pow=False)

class evaluate:
    """ Control automatic evaluation

    Explanation
    ===========

    This context manager controls whether or not all SymPy functions evaluate
    by default.

    Note that much of SymPy expects evaluated expressions.  This functionality
    is experimental and is unlikely to function as intended on large
    expressions.

    Examples
    ========

    >>> from sympy import evaluate
    >>> from sympy.abc import x
    >>> print(x + x)
    2*x
    >>> with evaluate(False):
    ...     print(x + x)
    x + x
    """
    # 初始化方法，接受布尔值参数
    def __init__(self, x):
        self.x = x
        self.old = []

    # 进入上下文时保存当前的 evaluate 值，并设置为新值
    def __enter__(self):
        self.old.append(global_parameters.evaluate)
        global_parameters.evaluate = self.x

    # 离开上下文时恢复原来的 evaluate 值
    def __exit__(self, exc_type, exc_val, exc_tb):
        global_parameters.evaluate = self.old.pop()

@contextmanager
def distribute(x):
    """ Control automatic distribution of Number over Add

    Explanation
    ===========

    This context manager controls whether or not Mul distribute Number over
    Add. Plan is to avoid distributing Number over Add in all of sympy. Once
    that is done, this contextmanager will be removed.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.core.parameters import distribute
    >>> print(2*(x + 1))
    2*x + 2

    """
    # 上下文管理器函数，用于控制是否在乘法中分配数字到加法中
    yield
    # 使用上下文管理器 `distribute(False)`，临时设定 `distribute` 参数为 False
    >>> with distribute(False):
    # 打印表达式 `2*(x + 1)`
    ...     print(2*(x + 1))
    # 结果是 `2*(x + 1)`
    2*(x + 1)
    """

    # 保存当前全局参数中的 `distribute` 值到变量 `old`
    old = global_parameters.distribute

    # 尝试设置全局参数 `distribute` 为 `x`
    try:
        global_parameters.distribute = x
        # 执行 yield 语句，将控制权返回给调用者
        yield
    finally:
        # 在 finally 块中恢复 `distribute` 参数为先前保存的值 `old`
        global_parameters.distribute = old
@contextmanager
def _exp_is_pow(x):
    """
    控制是否将 `e^x` 表示为 ``exp(x)`` 还是 ``Pow(E, x)`` 的上下文管理器。

    示例
    ========

    >>> from sympy import exp
    >>> from sympy.abc import x
    >>> from sympy.core.parameters import _exp_is_pow
    >>> with _exp_is_pow(True): print(type(exp(x)))
    <class 'sympy.core.power.Pow'>
    >>> with _exp_is_pow(False): print(type(exp(x)))
    exp
    """
    # 保存旧的全局参数值
    old = global_parameters.exp_is_pow

    # 清除缓存
    clear_cache()
    try:
        # 设置全局参数为新值 x
        global_parameters.exp_is_pow = x
        # 执行 yield 语句之前的代码块
        yield
    finally:
        # 无论如何，最终都要清除缓存和恢复旧的全局参数值
        clear_cache()
        global_parameters.exp_is_pow = old
```