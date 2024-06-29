# `D:\src\scipysrc\matplotlib\lib\matplotlib\backend_tools.pyi`

```
import enum  # 导入枚举模块
from matplotlib import cbook  # 导入 matplotlib 的 cbook 模块
from matplotlib.axes import Axes  # 导入 matplotlib 的 Axes 类
from matplotlib.backend_bases import ToolContainerBase, FigureCanvasBase  # 导入 matplotlib 的 ToolContainerBase 和 FigureCanvasBase 类
from matplotlib.backend_managers import ToolManager, ToolEvent  # 导入 matplotlib 的 ToolManager 和 ToolEvent 类
from matplotlib.figure import Figure  # 导入 matplotlib 的 Figure 类
from matplotlib.scale import ScaleBase  # 导入 matplotlib 的 ScaleBase 类

from typing import Any  # 导入 Any 类型，用于灵活的类型注解

class Cursors(enum.IntEnum):  # 定义枚举 Cursors，继承自 enum.IntEnum
    POINTER: int  # 光标类型：指针
    HAND: int  # 光标类型：手
    SELECT_REGION: int  # 光标类型：选择区域
    MOVE: int  # 光标类型：移动
    WAIT: int  # 光标类型：等待
    RESIZE_HORIZONTAL: int  # 光标类型：水平调整大小
    RESIZE_VERTICAL: int  # 光标类型：垂直调整大小

cursors = Cursors  # 创建 Cursors 枚举的别名 cursors

class ToolBase:  # 定义工具基类 ToolBase
    @property
    def default_keymap(self) -> list[str] | None: ...  # 默认键映射列表或空值
    description: str | None  # 工具描述或空值
    image: str | None  # 工具图像路径或空值

    def __init__(self, toolmanager: ToolManager, name: str) -> None: ...  # 初始化方法，接受 ToolManager 和名称参数

    @property
    def name(self) -> str: ...  # 工具名称属性，返回字符串类型

    @property
    def toolmanager(self) -> ToolManager: ...  # 工具管理器属性，返回 ToolManager 类型

    @property
    def canvas(self) -> FigureCanvasBase | None: ...  # 画布属性，返回 FigureCanvasBase 类型或空值

    @property
    def figure(self) -> Figure | None: ...  # 图形属性，返回 Figure 类型或空值

    @figure.setter
    def figure(self, figure: Figure | None) -> None: ...  # 图形属性的 setter 方法，接受 Figure 类型或空值

    def set_figure(self, figure: Figure | None) -> None: ...  # 设置图形的方法，接受 Figure 类型或空值

    def trigger(self, sender: Any, event: ToolEvent, data: Any = ...) -> None: ...  # 触发方法，接受发送者、事件和数据参数

class ToolToggleBase(ToolBase):  # 工具切换基类，继承自 ToolBase
    radio_group: str | None  # 单选按钮组或空值
    cursor: Cursors | None  # 光标类型或空值
    default_toggled: bool  # 默认切换状态

    def __init__(self, *args, **kwargs) -> None: ...  # 初始化方法，接受任意位置参数和关键字参数

    def enable(self, event: ToolEvent | None = ...) -> None: ...  # 启用方法，接受 ToolEvent 或空值参数

    def disable(self, event: ToolEvent | None = ...) -> None: ...  # 禁用方法，接受 ToolEvent 或空值参数

    @property
    def toggled(self) -> bool: ...  # 切换状态属性，返回布尔值

    def set_figure(self, figure: Figure | None) -> None: ...  # 设置图形的方法，接受 Figure 类型或空值

class ToolSetCursor(ToolBase):  # 设置光标工具，继承自 ToolBase
    ...

class ToolCursorPosition(ToolBase):  # 光标位置工具，继承自 ToolBase
    def send_message(self, event: ToolEvent) -> None: ...  # 发送消息方法，接受 ToolEvent 参数

class RubberbandBase(ToolBase):  # 橡皮筋基类，继承自 ToolBase
    def draw_rubberband(self, *data) -> None: ...  # 绘制橡皮筋方法，接受任意数量的数据参数

    def remove_rubberband(self) -> None: ...  # 移除橡皮筋方法

class ToolQuit(ToolBase):  # 退出工具，继承自 ToolBase
    ...

class ToolQuitAll(ToolBase):  # 全部退出工具，继承自 ToolBase
    ...

class ToolGrid(ToolBase):  # 网格工具，继承自 ToolBase
    ...

class ToolMinorGrid(ToolBase):  # 次要网格工具，继承自 ToolBase
    ...

class ToolFullScreen(ToolBase):  # 全屏工具，继承自 ToolBase
    ...

class AxisScaleBase(ToolToggleBase):  # 轴缩放基类，继承自 ToolToggleBase
    def enable(self, event: ToolEvent | None = ...) -> None: ...  # 启用方法，接受 ToolEvent 或空值参数

    def disable(self, event: ToolEvent | None = ...) -> None: ...  # 禁用方法，接受 ToolEvent 或空值参数

class ToolYScale(AxisScaleBase):  # Y 轴缩放工具，继承自 AxisScaleBase
    def set_scale(self, ax: Axes, scale: str | ScaleBase) -> None: ...  # 设置缩放方法，接受 Axes 和缩放类型参数

class ToolXScale(AxisScaleBase):  # X 轴缩放工具，继承自 AxisScaleBase
    def set_scale(self, ax, scale: str | ScaleBase) -> None: ...  # 设置缩放方法，接受 Axes 和缩放类型参数

class ToolViewsPositions(ToolBase):  # 视图和位置工具，继承自 ToolBase
    views: dict[Figure | Axes, cbook.Stack]  # 视图字典，键为 Figure 或 Axes，值为 cbook.Stack
    positions: dict[Figure | Axes, cbook.Stack]  # 位置字典，键为 Figure 或 Axes，值为 cbook.Stack
    home_views: dict[Figure, dict[Axes, tuple[float, float, float, float]]]  # 初始视图字典，键为 Figure，值为 Axes 和浮点数元组的字典

    def add_figure(self, figure: Figure) -> None: ...  # 添加图形方法，接受 Figure 参数

    def clear(self, figure: Figure) -> None: ...  # 清除方法，接受 Figure 参数

    def update_view(self) -> None: ...  # 更新视图方法

    def push_current(self, figure: Figure | None = ...) -> None: ...  # 推送当前方法，接受 Figure 或空值参数

    def update_home_views(self, figure: Figure | None = ...) -> None: ...  # 更新初始视图方法，接受 Figure 或空值参数

    def home(self) -> None: ...  # 初始方法

    def back(self) -> None: ...  # 返回方法

    def forward(self) -> None: ...  # 前进方法

class ViewsPositionsBase(ToolBase):  # 视图位置基类，继承自 ToolBase
    ...
# 定义 ToolHome 类，继承自 ViewsPositionsBase 类
class ToolHome(ViewsPositionsBase):
    ...

# 定义 ToolBack 类，继承自 ViewsPositionsBase 类
class ToolBack(ViewsPositionsBase):
    ...

# 定义 ToolForward 类，继承自 ViewsPositionsBase 类
class ToolForward(ViewsPositionsBase):
    ...

# 定义 ConfigureSubplotsBase 类，继承自 ToolBase 类
class ConfigureSubplotsBase(ToolBase):
    ...

# 定义 SaveFigureBase 类，继承自 ToolBase 类
class SaveFigureBase(ToolBase):
    ...

# 定义 ZoomPanBase 类，继承自 ToolToggleBase 类
class ZoomPanBase(ToolToggleBase):
    # 定义类的属性
    base_scale: float
    scrollthresh: float
    lastscroll: float

    # 初始化方法，接受任意参数
    def __init__(self, *args) -> None:
        ...

    # 启用方法，接受 ToolEvent 类型或 None 类型参数
    def enable(self, event: ToolEvent | None = ...) -> None:
        ...

    # 禁用方法，接受 ToolEvent 类型或 None 类型参数
    def disable(self, event: ToolEvent | None = ...) -> None:
        ...

    # 滚动缩放方法，接受 ToolEvent 参数
    def scroll_zoom(self, event: ToolEvent) -> None:
        ...

# 定义 ToolZoom 类，继承自 ZoomPanBase 类
class ToolZoom(ZoomPanBase):
    ...

# 定义 ToolPan 类，继承自 ZoomPanBase 类
class ToolPan(ZoomPanBase):
    ...

# 定义 ToolHelpBase 类，继承自 ToolBase 类
class ToolHelpBase(ToolBase):
    # 静态方法，格式化快捷键序列并返回字符串
    @staticmethod
    def format_shortcut(key_sequence: str) -> str:
        ...

# 定义 ToolCopyToClipboardBase 类，继承自 ToolBase 类
class ToolCopyToClipboardBase(ToolBase):
    ...

# 定义 default_tools 变量，类型为字典，键为字符串，值为 ToolBase 类型
default_tools: dict[str, ToolBase]

# 定义 default_toolbar_tools 变量，类型为列表，元素为列表，包含字符串或字符串列表
default_toolbar_tools: list[list[str | list[str]]]

# 添加工具到工具管理器的函数，接受 toolmanager 和 tools 参数
def add_tools_to_manager(
    toolmanager: ToolManager, tools: dict[str, type[ToolBase]] = ...
) -> None:
    ...

# 添加工具到容器的函数，接受 container 和 tools 参数
def add_tools_to_container(container: ToolContainerBase, tools: list[Any] = ...) -> None:
    ...
```