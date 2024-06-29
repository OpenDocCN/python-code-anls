# `D:\src\scipysrc\matplotlib\lib\matplotlib\testing\widgets.pyi`

```py
# 导入需要的类型提示模块和字面量类型
from typing import Any, Literal

# 导入 Matplotlib 的绘图轴、事件和鼠标按钮相关模块
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event, MouseButton
from matplotlib.widgets import AxesWidget, Widget

# 定义一个返回绘图轴对象的函数，返回类型为 Axes
def get_ax() -> Axes: ...

# 定义一个什么都不做的函数，接受任意参数，无返回值
def noop(*args: Any, **kwargs: Any) -> None: ...

# 模拟创建一个事件对象的函数，接受绘图轴、鼠标按钮、坐标数据、键盘键值和步长参数，返回类型为 Event
def mock_event(
    ax: Axes,
    button: MouseButton | int | Literal["up", "down"] | None = ...,
    xdata: float = ...,
    ydata: float = ...,
    key: str | None = ...,
    step: int = ...,
) -> Event: ...

# 执行特定事件的函数，接受工具对象、事件类型、鼠标按钮、坐标数据、键盘键值和步长参数，无返回值
def do_event(
    tool: AxesWidget,
    etype: str,
    button: MouseButton | int | Literal["up", "down"] | None = ...,
    xdata: float = ...,
    ydata: float = ...,
    key: str | None = ...,
    step: int = ...,
) -> None: ...

# 模拟鼠标点击和拖拽的函数，接受工具对象、起始坐标、结束坐标和可选的键盘键值参数，无返回值
def click_and_drag(
    tool: Widget,
    start: tuple[float, float],
    end: tuple[float, float],
    key: str | None = ...,
) -> None: ...
```