# `D:\src\scipysrc\matplotlib\lib\matplotlib\backend_managers.pyi`

```py
from matplotlib import backend_tools, widgets
# 导入 matplotlib 中的工具和小部件模块

from matplotlib.backend_bases import FigureCanvasBase
# 导入 matplotlib 的基础图形画布类

from matplotlib.figure import Figure
# 导入 matplotlib 的图形类 Figure

from collections.abc import Callable, Iterable
# 导入标准库中的 Callable 和 Iterable 类型

from typing import Any, TypeVar
# 导入 typing 模块中的 Any 和 TypeVar 类型

class ToolEvent:
    name: str
    sender: Any
    tool: backend_tools.ToolBase
    data: Any
    def __init__(self, name, sender, tool, data: Any | None = ...) -> None: ...
    # 定义工具事件类，包含名称、发送者、工具和数据，支持可选的空数据

class ToolTriggerEvent(ToolEvent):
    canvasevent: ToolEvent
    def __init__(
        self,
        name,
        sender,
        tool,
        canvasevent: ToolEvent | None = ...,
        data: Any | None = ...,
    ) -> None: ...
    # 定义工具触发事件类，继承自工具事件类，包含画布事件和数据，支持可选的空画布事件和数据

class ToolManagerMessageEvent:
    name: str
    sender: Any
    message: str
    def __init__(self, name: str, sender: Any, message: str) -> None: ...
    # 定义工具管理器消息事件类，包含名称、发送者和消息文本

class ToolManager:
    keypresslock: widgets.LockDraw
    messagelock: widgets.LockDraw
    def __init__(self, figure: Figure | None = ...) -> None: ...
    # 定义工具管理器类，包含按键锁和消息锁，支持可选的图形对象

    @property
    def canvas(self) -> FigureCanvasBase | None: ...
    # 定义属性方法 canvas，返回图形画布基类或空值

    @property
    def figure(self) -> Figure | None: ...
    # 定义属性方法 figure，返回图形对象或空值

    @figure.setter
    def figure(self, figure: Figure) -> None: ...
    # 定义设置方法 figure，设置图形对象，不返回值

    def set_figure(self, figure: Figure, update_tools: bool = ...) -> None: ...
    # 定义设置图形对象的方法，支持更新工具状态，默认不更新

    def toolmanager_connect(self, s: str, func: Callable[[ToolEvent], Any]) -> int: ...
    # 定义工具管理器连接方法，接受事件类型和回调函数，返回连接标识符

    def toolmanager_disconnect(self, cid: int) -> None: ...
    # 定义工具管理器断开连接方法，接受连接标识符，无返回值

    def message_event(self, message: str, sender: Any | None = ...) -> None: ...
    # 定义消息事件方法，接受消息文本和可选的发送者，无返回值

    @property
    def active_toggle(self) -> dict[str | None, list[str] | str]: ...
    # 定义属性方法 active_toggle，返回活动切换状态的字典结构

    def get_tool_keymap(self, name: str) -> list[str]: ...
    # 定义获取工具键位映射方法，接受工具名称，返回键位列表

    def update_keymap(self, name: str, key: str | Iterable[str]) -> None: ...
    # 定义更新键位映射方法，接受工具名称和键位字符串或可迭代对象，无返回值

    def remove_tool(self, name: str) -> None: ...
    # 定义移除工具方法，接受工具名称，无返回值

    _T = TypeVar("_T", bound=backend_tools.ToolBase)
    def add_tool(self, name: str, tool: type[_T], *args, **kwargs) -> _T: ...
    # 定义添加工具方法，接受工具名称、工具类型和额外参数，返回添加的工具对象

    def trigger_tool(
        self,
        name: str | backend_tools.ToolBase,
        sender: Any | None = ...,
        canvasevent: ToolEvent | None = ...,
        data: Any | None = ...,
    ) -> None: ...
    # 定义触发工具方法，接受工具名称或工具对象、可选的发送者、画布事件和数据，无返回值

    @property
    def tools(self) -> dict[str, backend_tools.ToolBase]: ...
    # 定义属性方法 tools，返回工具名称到工具对象的字典

    def get_tool(
        self, name: str | backend_tools.ToolBase, warn: bool = ...
    ) -> backend_tools.ToolBase | None: ...
    # 定义获取工具方法，接受工具名称或工具对象和是否警告的标志，返回工具对象或空值
```