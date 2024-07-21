# `.\pytorch\torch\profiler\itt.py`

```py
# mypy: allow-untyped-defs
# 导入 contextmanager 模块，支持上下文管理器
from contextlib import contextmanager

# 尝试导入 torch._C 模块中的 _itt 子模块，用于性能分析
try:
    from torch._C import _itt
except ImportError:
    # 如果导入失败，定义一个名为 _ITTStub 的类来模拟 _itt 功能
    class _ITTStub:
        @staticmethod
        def _fail(*args, **kwargs):
            raise RuntimeError(
                "ITT functions not installed. Are you sure you have a ITT build?"
            )

        @staticmethod
        def is_available():
            return False

        # 定义 _ITTStub 类的方法，用于模拟 ITT 功能
        rangePush = _fail
        rangePop = _fail
        mark = _fail

    # 将 _itt 变量赋值为 _ITTStub 类的实例，用于后续调用 ITT 功能时的兼容性处理
    _itt = _ITTStub()  # type: ignore[assignment]

# 声明 __all__ 变量，定义模块的公开接口列表
__all__ = ["is_available", "range_push", "range_pop", "mark", "range"]


def is_available():
    """
    检查 ITT 功能是否可用
    """
    return _itt.is_available()


def range_push(msg):
    """
    将一个范围推送到嵌套范围跨度的堆栈中。返回开始的范围的零基准深度。

    参数:
        msg (str): 与范围关联的 ASCII 消息
    """
    return _itt.rangePush(msg)


def range_pop():
    """
    从嵌套范围跨度的堆栈中弹出一个范围。返回结束的范围的零基准深度。
    """
    return _itt.rangePop()


def mark(msg):
    """
    描述发生在某个时刻的即时事件。

    参数:
        msg (str): 与事件关联的 ASCII 消息
    """
    return _itt.mark(msg)


@contextmanager
def range(msg, *args, **kwargs):
    """
    上下文管理器 / 装饰器，在其范围的开头推送一个 ITT 范围，并在结尾处弹出它。如果有额外的参数，则将它们作为参数传递给 msg.format()。

    参数:
        msg (str): 与范围关联的消息
    """
    range_push(msg.format(*args, **kwargs))
    try:
        yield
    finally:
        range_pop()
```