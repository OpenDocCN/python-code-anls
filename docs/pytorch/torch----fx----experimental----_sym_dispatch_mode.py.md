# `.\pytorch\torch\fx\experimental\_sym_dispatch_mode.py`

```py
# mypy: allow-untyped-defs
# 引入类型提示所需的模块
from typing import List, Optional, Type

# 将这些符号添加到模块的公开接口中
__all__ = ["SymDispatchMode", "handle_sym_dispatch", "sym_function_mode"]

# Optional类型的全局变量，用于保存SymDispatchMode对象或None
SYM_FUNCTION_MODE: Optional["SymDispatchMode"] = None


# SymDispatchMode用于处理PySymInt上的操作。当这些操作发生时，会在__sym_dispatch__方法中调用。
# 这与TorchDispatchMode类似，但有一些区别：
#
#   - 在TorchDispatchMode中，您得到与用户调用API时相同的参数；例如，如果调用torch.ops.aten.foo(a, b)，
#     则您在调用中得到(a, b)作为参数。在SymDispatchMode中，如果调用a + b（其中a和b是SymInts），
#     您会得到(a.node, b.node)作为参数（这些是PySymInts）。
#
#   - SymInt/PySymInt不支持FX代理支持（不像Tensor等）。因此，您必须手动调用Tracer/create_node来写入图中。
#     请参阅ProxySymDispatchMode以获取示例。
#
class SymDispatchMode:
    def __sym_dispatch__(self, func, types, args, kwargs):
        # 抽象方法，需要子类实现具体的操作
        raise NotImplementedError

    def __enter__(self):
        # 设置全局SYM_FUNCTION_MODE变量为当前的SymDispatchMode对象实例
        global SYM_FUNCTION_MODE
        old = SYM_FUNCTION_MODE
        # 如果已经有inner属性，表示该模式已经在使用中，抛出运行时错误
        if hasattr(self, "inner"):
            raise RuntimeError(
                f"{self} has already been used as a mode. Please use a fresh version"
            )
        else:
            # 将旧的SYM_FUNCTION_MODE保存在inner属性中
            self.inner = old
        SYM_FUNCTION_MODE = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出时恢复SYM_FUNCTION_MODE为之前保存的inner属性值
        global SYM_FUNCTION_MODE
        SYM_FUNCTION_MODE = self.inner


def handle_sym_dispatch(func, args, kwargs):
    # 获取当前的SYM_FUNCTION_MODE
    global SYM_FUNCTION_MODE
    mode = sym_function_mode()
    assert mode
    # 将全局SYM_FUNCTION_MODE设置为当前模式的inner属性
    SYM_FUNCTION_MODE = mode.inner
    try:
        # TODO: properly compute types
        types: List[Type] = []
        # 调用当前SYM_FUNCTION_MODE对象的__sym_dispatch__方法处理操作
        return mode.__sym_dispatch__(func, types, args, kwargs)
    finally:
        # 恢复SYM_FUNCTION_MODE为原始模式
        SYM_FUNCTION_MODE = mode


def sym_function_mode():
    # 返回当前的SYM_FUNCTION_MODE
    return SYM_FUNCTION_MODE
```