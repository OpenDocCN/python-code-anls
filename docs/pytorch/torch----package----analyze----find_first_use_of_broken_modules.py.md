# `.\pytorch\torch\package\analyze\find_first_use_of_broken_modules.py`

```
from typing import Dict, List  # 导入类型提示工具中的字典和列表类型

from ..package_exporter import PackagingError  # 导入自定义异常类 PackagingError

__all__ = ["find_first_use_of_broken_modules"]  # 声明模块中公开的符号为 find_first_use_of_broken_modules

def find_first_use_of_broken_modules(exc: PackagingError) -> Dict[str, List[str]]:
    """
    在 PackagingError 中查找所有损坏的模块，并为每个模块返回其首次出现的依赖路径。

    例如，对于损坏的模块 m.n.o，它在处理 a.b.c 时被添加到依赖图中，然后在处理 d.e.f 时再次遇到。
    此方法将返回 {'m.n.o': ['a', 'b', 'c']}

    Args:
        exc: PackagingError 的实例，表示一个打包错误

    Returns:
        一个字典，键为损坏模块的名称，值为其首次出现的依赖路径列表。
    """

    assert isinstance(exc, PackagingError), "exception must be a PackagingError"  # 断言确保 exc 是 PackagingError 的实例

    uses = {}  # 初始化一个空字典，用于存储损坏模块及其首次出现的依赖路径
    broken_module_names = [
        m for m, attr in exc.dependency_graph.nodes.items() if attr.get("error", False)
    ]
    # 获取所有具有 "error" 属性的节点名称，这些节点表示损坏的模块

    for module_name in broken_module_names:
        path = exc.dependency_graph.first_path(module_name)
        # 对每个损坏的模块名称，调用 dependency_graph 的 first_path 方法获取其首次出现的依赖路径
        uses[module_name] = path  # 将损坏模块名称及其路径存入 uses 字典中

    return uses  # 返回存储所有损坏模块及其首次出现路径的字典
```