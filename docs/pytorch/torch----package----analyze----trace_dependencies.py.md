# `.\pytorch\torch\package\analyze\trace_dependencies.py`

```py
# mypy: allow-untyped-defs
# 导入 sys 模块，用于设置和获取系统相关信息
import sys
# 导入类型提示相关模块
from typing import Any, Callable, Iterable, List, Tuple

# 将 trace_dependencies 导出，使其在模块外部可见
__all__ = ["trace_dependencies"]

# 定义 trace_dependencies 函数，用于跟踪可调用对象执行过程中使用的模块
def trace_dependencies(
    callable: Callable[[Any], Any], inputs: Iterable[Tuple[Any, ...]]
) -> List[str]:
    """Trace the execution of a callable in order to determine which modules it uses.

    Args:
        callable: 要执行和跟踪的可调用对象。
        inputs: 在跟踪过程中使用的输入。对每组输入调用 'callable' 时使用的模块会合并，
            以确定打包时可调用对象使用的所有模块。

    Returns:
        执行 callable 过程中使用的所有模块名称的列表。
    """
    modules_used = set()

    # 定义记录使用模块的函数
    def record_used_modules(frame, event, arg):
        # 如果事件不是 Python 函数调用，不做处理。
        if event != "call":
            return

        # 获取被调用的函数名。
        name = frame.f_code.co_name
        module = None

        # 尝试确定函数所在的模块名：
        #   1) 检查帧的全局命名空间。
        #   2) 检查帧的局部命名空间。
        #   3) 处理类实例方法调用时，检查局部命名空间中与 "self" 对应的对象的名为 'name' 的属性。
        if name in frame.f_globals:
            module = frame.f_globals[name].__module__
        elif name in frame.f_locals:
            module = frame.f_locals[name].__module__
        elif "self" in frame.f_locals:
            method = getattr(frame.f_locals["self"], name, None)
            module = method.__module__ if method else None

        # 如果找到模块名，则添加到使用模块集合中。
        if module:
            modules_used.add(module)

    try:
        # 将 record_used_modules 函数作为分析器函数附加到系统。
        sys.setprofile(record_used_modules)

        # 使用所有输入执行可调用对象。
        for inp in inputs:
            callable(*inp)

    finally:
        # 分离分析器函数。
        sys.setprofile(None)

    # 将集合转换为列表并返回使用的模块名称。
    return list(modules_used)
```