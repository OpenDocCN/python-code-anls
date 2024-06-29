# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\registry.pyi`

```
# 导入必要的枚举类型和模块类型
from enum import Enum
from types import ModuleType

# 定义一个枚举类，表示后端过滤器，包括交互式和非交互式
class BackendFilter(Enum):
    INTERACTIVE: int
    NON_INTERACTIVE: int

# 后端注册表类，管理后端与GUI框架之间的映射和加载
class BackendRegistry:
    # 内置后端到GUI框架的映射字典
    _BUILTIN_BACKEND_TO_GUI_FRAMEWORK: dict[str, str]
    # GUI框架到后端的映射字典
    _GUI_FRAMEWORK_TO_BACKEND: dict[str, str]

    # 是否已加载入口点的标志
    _loaded_entry_points: bool
    # 后端到GUI框架的实际映射字典
    _backend_to_gui_framework: dict[str, str]
    # 后端名称到模块名称的映射字典
    _name_to_module: dict[str, str]

    # 获取特定后端的模块名称
    def _backend_module_name(self, backend: str) -> str: ...
    # 清除所有已加载的数据
    def _clear(self) -> None: ...
    # 确保已加载所有入口点数据
    def _ensure_entry_points_loaded(self) -> None: ...
    # 根据后端获取对应的GUI框架
    def _get_gui_framework_by_loading(self, backend: str) -> str: ...
    # 读取并返回所有入口点数据
    def _read_entry_points(self) -> list[tuple[str, str]]: ...
    # 验证并存储入口点数据
    def _validate_and_store_entry_points(self, entries: list[tuple[str, str]]) -> None: ...

    # 根据GUI框架获取对应的后端名称
    def backend_for_gui_framework(self, framework: str) -> str | None: ...
    # 验证给定的后端名称是否有效
    def is_valid_backend(self, backend: str) -> bool: ...
    # 返回所有已注册的后端名称列表
    def list_all(self) -> list[str]: ...
    # 返回内置后端名称列表，可以选择根据交互性过滤
    def list_builtin(self, filter_: BackendFilter | None) -> list[str]: ...
    # 返回所有已知的GUI框架名称列表
    def list_gui_frameworks(self) -> list[str]: ...
    # 加载指定后端的模块并返回
    def load_backend_module(self, backend: str) -> ModuleType: ...
    # 解析并返回给定后端名称对应的GUI框架和后端模块名称
    def resolve_backend(self, backend: str | None) -> tuple[str, str | None]: ...
    # 解析并返回给定GUI框架或后端名称对应的后端和GUI框架名称
    def resolve_gui_or_backend(self, gui_or_backend: str | None) -> tuple[str, str | None]: ...


# 创建一个全局的后端注册表实例
backend_registry: BackendRegistry
```