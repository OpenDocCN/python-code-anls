# `D:\src\scipysrc\matplotlib\lib\matplotlib\widgets.pyi`

```py
from .artist import Artist
from .axes import Axes
from .backend_bases import FigureCanvasBase, Event, MouseEvent, MouseButton
from .collections import LineCollection
from .figure import Figure
from .lines import Line2D
from .patches import Circle, Polygon, Rectangle
from .text import Text

import PIL.Image

from collections.abc import Callable, Collection, Iterable, Sequence
from typing import Any, Literal
from numpy.typing import ArrayLike
from .typing import ColorType
import numpy as np

# 定义了一个名为 LockDraw 的类，用于控制绘图操作的锁定状态
class LockDraw:
    def __init__(self) -> None: ...
    def __call__(self, o: Any) -> None: ...
    def release(self, o: Any) -> None: ...
    def available(self, o: Any) -> bool: ...
    def isowner(self, o: Any) -> bool: ...
    def locked(self) -> bool: ...

# 定义了一个名为 Widget 的基类，表示一个基本的 GUI 控件
class Widget:
    drawon: bool
    eventson: bool
    active: bool
    # 设置控件的激活状态
    def set_active(self, active: bool) -> None: ...
    # 获取控件的激活状态
    def get_active(self) -> None: ...
    # 判断是否忽略指定的事件
    def ignore(self, event) -> bool: ...

# AxesWidget 类继承自 Widget 类，表示与坐标轴相关的 GUI 控件
class AxesWidget(Widget):
    ax: Axes
    canvas: FigureCanvasBase | None
    # 初始化方法，传入一个 Axes 对象
    def __init__(self, ax: Axes) -> None: ...
    # 连接事件到回调函数的方法
    def connect_event(self, event: Event, callback: Callable) -> None: ...
    # 断开所有事件连接的方法
    def disconnect_events(self) -> None: ...

# Button 类继承自 AxesWidget，表示一个按钮控件
class Button(AxesWidget):
    label: Text
    color: ColorType
    hovercolor: ColorType
    # 初始化方法，传入 Axes 对象、按钮文本、可选的图片和颜色信息
    def __init__(
        self,
        ax: Axes,
        label: str,
        image: ArrayLike | PIL.Image.Image | None = ...,
        color: ColorType = ...,
        hovercolor: ColorType = ...,
        *,
        useblit: bool = ...
    ) -> None: ...
    # 注册点击事件的回调函数
    def on_clicked(self, func: Callable[[Event], Any]) -> int: ...
    # 断开指定 cid 的事件连接
    def disconnect(self, cid: int) -> None: ...

# SliderBase 类继承自 AxesWidget，表示一个滑块的基础类
class SliderBase(AxesWidget):
    orientation: Literal["horizontal", "vertical"]
    closedmin: bool
    closedmax: bool
    valmin: float
    valmax: float
    valstep: float | ArrayLike | None
    drag_active: bool
    valfmt: str
    # 初始化方法，传入 Axes 对象、滑块方向、闭合最小值、闭合最大值、值的最小和最大限制、值的格式、是否处于拖拽状态以及值的步长信息
    def __init__(
        self,
        ax: Axes,
        orientation: Literal["horizontal", "vertical"],
        closedmin: bool,
        closedmax: bool,
        valmin: float,
        valmax: float,
        valfmt: str,
        dragging: Slider | None,
        valstep: float | ArrayLike | None,
    ) -> None: ...
    # 断开指定 cid 的事件连接
    def disconnect(self, cid: int) -> None: ...
    # 重置滑块的方法
    def reset(self) -> None: ...

# Slider 类继承自 SliderBase，表示一个完整的滑块控件
class Slider(SliderBase):
    slidermin: Slider | None
    slidermax: Slider | None
    val: float
    valinit: float
    track: Rectangle
    poly: Polygon
    hline: Line2D
    vline: Line2D
    label: Text
    valtext: Text
    # 初始化函数，用于创建一个新的 Slider 实例
    def __init__(
        self,
        ax: Axes,  # 传入 Axes 对象，表示图形绘制的坐标系
        label: str,  # Slider 的标签，用于显示在图形界面上
        valmin: float,  # Slider 允许的最小值
        valmax: float,  # Slider 允许的最大值
        *,
        valinit: float = ...,  # Slider 的初始值，默认为省略值
        valfmt: str | None = ...,  # 控制显示值的格式，可以为字符串或 None
        closedmin: bool = ...,  # 是否闭合最小值边界
        closedmax: bool = ...,  # 是否闭合最大值边界
        slidermin: Slider | None = ...,  # 与当前 Slider 关联的最小值 Slider 对象，可以为 None
        slidermax: Slider | None = ...,  # 与当前 Slider 关联的最大值 Slider 对象，可以为 None
        dragging: bool = ...,  # 是否启用拖拽功能
        valstep: float | ArrayLike | None = ...,  # 控制步进的数值，可以是单个浮点数、数组或 None
        orientation: Literal["horizontal", "vertical"] = ...,  # Slider 的方向，水平或垂直
        initcolor: ColorType = ...,  # 初始时 Slider 的颜色类型
        track_color: ColorType = ...,  # Slider 轨道的颜色类型
        handle_style: dict[str, Any] | None = ...,  # Slider 滑块的样式，可以是字典或 None
        **kwargs  # 其他可选参数，传递给基类构造函数
    ) -> None:  # 初始化函数无返回值
        ...

    # 设置 Slider 的当前值
    def set_val(self, val: float) -> None:
        ...

    # 注册回调函数，当 Slider 值发生变化时调用
    def on_changed(self, func: Callable[[float], Any]) -> int:
        ...
class RangeSlider(SliderBase):
    # 定义范围滑块类，继承自SliderBase

    val: tuple[float, float]
    # 当前滑块的数值范围，为元组类型

    valinit: tuple[float, float]
    # 初始滑块的数值范围，为元组类型

    track: Rectangle
    # 滑块的轨道，为矩形对象

    poly: Polygon
    # 滑块的多边形表示

    label: Text
    # 滑块的标签文本对象

    valtext: Text
    # 显示当前数值范围的文本对象

    def __init__(
        self,
        ax: Axes,
        label: str,
        valmin: float,
        valmax: float,
        *,
        valinit: tuple[float, float] | None = ...,
        valfmt: str | None = ...,
        closedmin: bool = ...,
        closedmax: bool = ...,
        dragging: bool = ...,
        valstep: float | ArrayLike | None = ...,
        orientation: Literal["horizontal", "vertical"] = ...,
        track_color: ColorType = ...,
        handle_style: dict[str, Any] | None = ...,
        **kwargs
    ) -> None:
    # 初始化函数，创建范围滑块对象

    def set_min(self, min: float) -> None:
    # 设置滑块的最小值

    def set_max(self, max: float) -> None:
    # 设置滑块的最大值

    def set_val(self, val: ArrayLike) -> None:
    # 设置滑块的当前数值范围

    def on_changed(self, func: Callable[[tuple[float, float]], Any]) -> int:
    # 注册数值改变时的回调函数，返回回调函数的 ID


class CheckButtons(AxesWidget):
    # 复选按钮组件类，继承自AxesWidget

    labels: list[Text]
    # 复选按钮的标签列表

    def __init__(
        self,
        ax: Axes,
        labels: Sequence[str],
        actives: Iterable[bool] | None = ...,
        *,
        useblit: bool = ...,
        label_props: dict[str, Any] | None = ...,
        frame_props: dict[str, Any] | None = ...,
        check_props: dict[str, Any] | None = ...,
    ) -> None:
    # 初始化函数，创建复选按钮组件对象

    def set_label_props(self, props: dict[str, Any]) -> None:
    # 设置标签的属性

    def set_frame_props(self, props: dict[str, Any]) -> None:
    # 设置框架的属性

    def set_check_props(self, props: dict[str, Any]) -> None:
    # 设置复选框的属性

    def set_active(self, index: int, state: bool | None = ...) -> None:
    # 设置指定索引位置的复选按钮状态

    def clear(self) -> None:
    # 清空复选按钮组件

    def get_status(self) -> list[bool]:
    # 获取当前所有复选按钮的状态列表

    def get_checked_labels(self) -> list[str]:
    # 获取当前被选中的复选按钮的标签列表

    def on_clicked(self, func: Callable[[str | None], Any]) -> int:
    # 注册点击事件的回调函数，返回回调函数的 ID

    def disconnect(self, cid: int) -> None:
    # 断开指定 ID 的回调函数连接


class TextBox(AxesWidget):
    # 文本框组件类，继承自AxesWidget

    label: Text
    # 文本框的标签文本对象

    text_disp: Text
    # 文本框中显示文本的文本对象

    cursor_index: int
    # 文本框中光标的位置索引

    cursor: LineCollection
    # 文本框中光标的图形表示

    color: ColorType
    # 文本框的颜色

    hovercolor: ColorType
    # 鼠标悬停在文本框上时的颜色

    capturekeystrokes: bool
    # 是否捕捉键盘输入事件

    def __init__(
        self,
        ax: Axes,
        label: str,
        initial: str = ...,
        *,
        color: ColorType = ...,
        hovercolor: ColorType = ...,
        label_pad: float = ...,
        textalignment: Literal["left", "center", "right"] = ...,
    ) -> None:
    # 初始化函数，创建文本框对象

    @property
    def text(self) -> str:
    # 获取当前文本框中的文本内容

    def set_val(self, val: str) -> None:
    # 设置文本框的文本内容

    def begin_typing(self) -> None:
    # 开始输入文本

    def stop_typing(self) -> None:
    # 停止输入文本

    def on_text_change(self, func: Callable[[str], Any]) -> int:
    # 注册文本内容改变时的回调函数，返回回调函数的 ID

    def on_submit(self, func: Callable[[str], Any]) -> int:
    # 注册文本提交时的回调函数，返回回调函数的 ID

    def disconnect(self, cid: int) -> None:
    # 断开指定 ID 的回调函数连接


class RadioButtons(AxesWidget):
    # 单选按钮组件类，继承自AxesWidget

    activecolor: ColorType
    # 激活状态时的颜色

    value_selected: str
    # 当前选中的值

    labels: list[Text]
    # 单选按钮的标签列表
    # 初始化方法，用于创建一个单选按钮组件
    def __init__(
        self,
        ax: Axes,  # 传入的 Axes 对象，用于在其上绘制单选按钮
        labels: Iterable[str],  # 单选按钮的标签列表
        active: int = ...,  # 当前选中的单选按钮的索引
        activecolor: ColorType | None = ...,  # 选中状态的颜色，可以为空
        *,
        useblit: bool = ...,  # 是否使用 blit 进行绘制
        label_props: dict[str, Any] | Sequence[dict[str, Any]] | None = ...,  # 单选按钮标签的属性字典或序列，可以为空
        radio_props: dict[str, Any] | None = ...,  # 单选按钮的属性字典，可以为空
    ) -> None: ...
    
    # 设置单选按钮标签的属性
    def set_label_props(self, props: dict[str, Any]) -> None: ...
    
    # 设置单选按钮的属性
    def set_radio_props(self, props: dict[str, Any]) -> None: ...
    
    # 设置当前活动的单选按钮索引
    def set_active(self, index: int) -> None: ...
    
    # 清除所有单选按钮
    def clear(self) -> None: ...
    
    # 绑定单击事件的回调函数，返回绑定的 ID
    def on_clicked(self, func: Callable[[str | None], Any]) -> int: ...
    
    # 断开指定 ID 的单击事件绑定
    def disconnect(self, cid: int) -> None: ...
class SubplotTool(Widget):
    # 定义 SubplotTool 类，继承自 Widget 类
    figure: Figure
    targetfig: Figure
    buttonreset: Button
    
    def __init__(self, targetfig: Figure, toolfig: Figure) -> None:
        # 初始化方法，接受两个 Figure 对象作为参数
        ...

class Cursor(AxesWidget):
    # 定义 Cursor 类，继承自 AxesWidget 类
    visible: bool
    horizOn: bool
    vertOn: bool
    useblit: bool
    lineh: Line2D
    linev: Line2D
    background: Any
    needclear: bool
    
    def __init__(
        self,
        ax: Axes,
        *,
        horizOn: bool = ...,
        vertOn: bool = ...,
        useblit: bool = ...,
        **lineprops
    ) -> None:
        # 初始化方法，接受 Axes 对象和可选的属性作为参数
        ...

    def clear(self, event: Event) -> None:
        # 清除方法，接受事件对象作为参数
        ...

    def onmove(self, event: Event) -> None:
        # 移动方法，接受事件对象作为参数
        ...

class MultiCursor(Widget):
    # 定义 MultiCursor 类，继承自 Widget 类
    axes: Sequence[Axes]
    horizOn: bool
    vertOn: bool
    visible: bool
    useblit: bool
    vlines: list[Line2D]
    hlines: list[Line2D]
    
    def __init__(
        self,
        canvas: Any,
        axes: Sequence[Axes],
        *,
        useblit: bool = ...,
        horizOn: bool = ...,
        vertOn: bool = ...,
        **lineprops
    ) -> None:
        # 初始化方法，接受画布对象、Axes 序列和可选的属性作为参数
        ...

    def connect(self) -> None:
        # 连接方法，无参数
        ...

    def disconnect(self) -> None:
        # 断开连接方法，无参数
        ...

    def clear(self, event: Event) -> None:
        # 清除方法，接受事件对象作为参数
        ...

    def onmove(self, event: Event) -> None:
        # 移动方法，接受事件对象作为参数
        ...

class _SelectorWidget(AxesWidget):
    # 定义 _SelectorWidget 类，继承自 AxesWidget 类
    onselect: Callable[[float, float], Any]
    useblit: bool
    background: Any
    validButtons: list[MouseButton]
    
    def __init__(
        self,
        ax: Axes,
        onselect: Callable[[float, float], Any],
        useblit: bool = ...,
        button: MouseButton | Collection[MouseButton] | None = ...,
        state_modifier_keys: dict[str, str] | None = ...,
        use_data_coordinates: bool = ...,
    ) -> None:
        # 初始化方法，接受 Axes 对象、回调函数和多个可选参数作为参数
        ...

    def update_background(self, event: Event) -> None:
        # 更新背景方法，接受事件对象作为参数
        ...

    def connect_default_events(self) -> None:
        # 连接默认事件方法，无参数
        ...

    def ignore(self, event: Event) -> bool:
        # 忽略事件方法，接受事件对象作为参数，返回布尔值
        ...

    def update(self) -> None:
        # 更新方法，无参数
        ...

    def press(self, event: Event) -> bool:
        # 按下方法，接受事件对象作为参数，返回布尔值
        ...

    def release(self, event: Event) -> bool:
        # 释放方法，接受事件对象作为参数，返回布尔值
        ...

    def onmove(self, event: Event) -> bool:
        # 移动方法，接受事件对象作为参数，返回布尔值
        ...

    def on_scroll(self, event: Event) -> None:
        # 滚动方法，接受事件对象作为参数
        ...

    def on_key_press(self, event: Event) -> None:
        # 按键按下方法，接受事件对象作为参数
        ...

    def on_key_release(self, event: Event) -> None:
        # 按键释放方法，接受事件对象作为参数
        ...

    def set_visible(self, visible: bool) -> None:
        # 设置可见性方法，接受布尔值作为参数
        ...

    def get_visible(self) -> bool:
        # 获取可见性方法，返回布尔值
        ...

    @property
    def visible(self) -> bool:
        # 可见性属性，返回布尔值
        ...

    def clear(self) -> None:
        # 清除方法，无参数
        ...

    @property
    def artists(self) -> tuple[Artist]:
        # 艺术家属性，返回艺术家元组
        ...

    def set_props(self, **props) -> None:
        # 设置属性方法，接受关键字参数
        ...

    def set_handle_props(self, **handle_props) -> None:
        # 设置处理属性方法，接受关键字参数
        ...

    def add_state(self, state: str) -> None:
        # 添加状态方法，接受状态字符串作为参数
        ...

    def remove_state(self, state: str) -> None:
        # 移除状态方法，接受状态字符串作为参数
        ...

class SpanSelector(_SelectorWidget):
    # 定义 SpanSelector 类，继承自 _SelectorWidget 类
    snap_values: ArrayLike | None
    onmove_callback: Callable[[float, float], Any]
    minspan: float
    grab_range: float
    drag_from_anywhere: bool
    ignore_event_outside: bool
    canvas: FigureCanvasBase | None
    # 初始化方法，为可选择区域选择工具设置各种参数和回调函数
    def __init__(
        self,
        ax: Axes,  # 绘图区对象，表示可选择区域将在其上操作
        onselect: Callable[[float, float], Any],  # 当选择完成时调用的回调函数，参数为选择区域的起始和结束位置
        direction: Literal["horizontal", "vertical"],  # 可选择的方向，水平或垂直

        # 可选关键字参数
        *,
        minspan: float = ...,  # 最小选择跨度
        useblit: bool = ...,  # 是否使用 blit 技术来增强绘图性能
        props: dict[str, Any] | None = ...,  # 用于设置可选择区域外观的属性
        onmove_callback: Callable[[float, float], Any] | None = ...,  # 当移动选择区域时调用的回调函数
        interactive: bool = ...,  # 是否允许交互式选择
        button: MouseButton | Collection[MouseButton] | None = ...,  # 触发选择的鼠标按钮
        handle_props: dict[str, Any] | None = ...,  # 用于设置选择区域手柄外观的属性
        grab_range: float = ...,  # 设置捕获选择区域的范围
        state_modifier_keys: dict[str, str] | None = ...,  # 修改选择状态的键盘按键
        drag_from_anywhere: bool = ...,  # 是否允许从任何位置拖动选择区域
        ignore_event_outside: bool = ...,  # 是否忽略区域外的事件
        snap_values: ArrayLike | None = ...,  # 吸附到指定数值的选择位置
    ) -> None: ...
    
    # 在给定的绘图区对象上创建新的坐标轴
    def new_axes(self, ax: Axes, *, _props: dict[str, Any] | None = ...) -> None: ...
    
    # 连接默认事件处理函数
    def connect_default_events(self) -> None: ...
    
    # 获取当前选择的方向，水平或垂直
    @property
    def direction(self) -> Literal["horizontal", "vertical"]: ...
    
    # 设置选择的方向，只能是水平或垂直
    @direction.setter
    def direction(self, direction: Literal["horizontal", "vertical"]) -> None: ...
    
    # 获取当前选择区域的起始和结束位置
    @property
    def extents(self) -> tuple[float, float]: ...
    
    # 设置选择区域的起始和结束位置
    @extents.setter
    def extents(self, extents: tuple[float, float]) -> None: ...
# 定义 ToolLineHandles 类，用于管理在 Axes 上的线条对象
class ToolLineHandles:
    # ax 属性用于存储该对象所属的 Axes 对象
    ax: Axes
    
    # 初始化方法，接受 Axes 对象、位置列表、方向（水平或垂直）、线条属性字典和 useblit 参数
    def __init__(
        self,
        ax: Axes,
        positions: ArrayLike,
        direction: Literal["horizontal", "vertical"],
        *,
        line_props: dict[str, Any] | None = ...,
        useblit: bool = ...,
    ) -> None: ...
    
    # 返回该对象包含的所有 Line2D 对象的元组
    @property
    def artists(self) -> tuple[Line2D]: ...
    
    # 返回该对象的位置列表
    @property
    def positions(self) -> list[float]: ...
    
    # 返回该对象的方向，可以是 "horizontal" 或 "vertical"
    @property
    def direction(self) -> Literal["horizontal", "vertical"]: ...
    
    # 设置对象的数据，更新位置
    def set_data(self, positions: ArrayLike) -> None: ...
    
    # 设置对象的可见性
    def set_visible(self, value: bool) -> None: ...
    
    # 设置对象的动画属性
    def set_animated(self, value: bool) -> None: ...
    
    # 删除该对象及其关联的内容
    def remove(self) -> None: ...
    
    # 返回距离给定坐标 (x, y) 最近的 Line2D 对象的索引和距离
    def closest(self, x: float, y: float) -> tuple[int, float]: ...

# 定义 ToolHandles 类，用于管理在 Axes 上的标记点对象
class ToolHandles:
    # ax 属性用于存储该对象所属的 Axes 对象
    ax: Axes
    
    # 初始化方法，接受 Axes 对象、x 坐标数组、y 坐标数组、标记类型、标记属性字典和 useblit 参数
    def __init__(
        self,
        ax: Axes,
        x: ArrayLike,
        y: ArrayLike,
        *,
        marker: str = ...,
        marker_props: dict[str, Any] | None = ...,
        useblit: bool = ...,
    ) -> None: ...
    
    # 返回该对象的 x 坐标数组
    @property
    def x(self) -> ArrayLike: ...
    
    # 返回该对象的 y 坐标数组
    @property
    def y(self) -> ArrayLike: ...
    
    # 返回该对象包含的所有 Line2D 对象的元组
    @property
    def artists(self) -> tuple[Line2D]: ...
    
    # 设置对象的数据，更新点的位置
    def set_data(self, pts: ArrayLike, y: ArrayLike | None = ...) -> None: ...
    
    # 设置对象的可见性
    def set_visible(self, val: bool) -> None: ...
    
    # 设置对象的动画属性
    def set_animated(self, val: bool) -> None: ...
    
    # 返回距离给定坐标 (x, y) 最近的 Line2D 对象的索引和距离
    def closest(self, x: float, y: float) -> tuple[int, float]: ...

# 定义 RectangleSelector 类，继承自 _SelectorWidget 类
class RectangleSelector(_SelectorWidget):
    # 是否允许从任意位置开始拖拽选择
    drag_from_anywhere: bool
    
    # 是否忽略在外部的事件
    ignore_event_outside: bool
    
    # 选择框的最小横向跨度和纵向跨度
    minspanx: float
    minspany: float
    
    # 选择框的坐标系，可以是 "data" 或 "pixels"
    spancoords: Literal["data", "pixels"]
    
    # 在哪个范围内开始抓取选择框
    grab_range: float
    
    # 初始化方法，接受 Axes 对象、选择事件处理函数 onselect 和多个可选参数
    def __init__(
        self,
        ax: Axes,
        onselect: Callable[[MouseEvent, MouseEvent], Any],
        *,
        minspanx: float = ...,
        minspany: float = ...,
        useblit: bool = ...,
        props: dict[str, Any] | None = ...,
        spancoords: Literal["data", "pixels"] = ...,
        button: MouseButton | Collection[MouseButton] | None = ...,
        grab_range: float = ...,
        handle_props: dict[str, Any] | None = ...,
        interactive: bool = ...,
        state_modifier_keys: dict[str, str] | None = ...,
        drag_from_anywhere: bool = ...,
        ignore_event_outside: bool = ...,
        use_data_coordinates: bool = ...,
    ) -> None: ...
    
    # 返回选择框的角落坐标数组
    @property
    def corners(self) -> tuple[np.ndarray, np.ndarray]: ...
    
    # 返回选择框边界中心点坐标数组
    @property
    def edge_centers(self) -> tuple[np.ndarray, np.ndarray]: ...
    
    # 返回选择框的中心坐标 (x, y)
    @property
    def center(self) -> tuple[float, float]: ...
    
    # 返回选择框的范围四个边界的坐标值 (xmin, xmax, ymin, ymax)
    @property
    def extents(self) -> tuple[float, float, float, float]: ...
    
    # 设置选择框的范围四个边界的坐标值
    @extents.setter
    def extents(self, extents: tuple[float, float, float, float]) -> None: ...
    
    # 返回选择框的旋转角度
    @property
    def rotation(self) -> float: ...
    
    # 设置选择框的旋转角度
    @rotation.setter
    def rotation(self, value: float) -> None: ...
    
    # 返回选择框的几何形状描述数组
    @property
    def geometry(self) -> np.ndarray: ...

# 定义 EllipseSelector 类，继承自 RectangleSelector 类
class EllipseSelector(RectangleSelector): ...

# 定义 LassoSelector 类，继承自 _SelectorWidget 类
class LassoSelector(_SelectorWidget):
    verts: None | list[tuple[float, float]]
    # 声明一个变量 verts，其类型可以是 None 或者包含多个二元组的列表，每个二元组内包含两个浮点数

    def __init__(
        self,
        ax: Axes,
        onselect: Callable[[list[tuple[float, float]]], Any],
        *,
        useblit: bool = ...,
        props: dict[str, Any] | None = ...,
        button: MouseButton | Collection[MouseButton] | None = ...,
    ) -> None: ...
    # 构造函数 __init__，初始化对象时调用
    # 参数说明：
    #   - self: 类的实例对象
    #   - ax: Axes 对象，用于绘制图形
    #   - onselect: 回调函数，接受一个包含二元组的列表作为参数，返回任意类型的值
    #   - useblit: 布尔类型，表示是否使用闪烁传输技术
    #   - props: 可以是包含字符串键和任意值的字典，或者为 None
    #   - button: 可以是 MouseButton 类型的单个值，或者 MouseButton 类型的集合，或者为 None
    # 返回值为 None，表示构造函数没有返回值
class PolygonSelector(_SelectorWidget):
    grab_range: float
    # PolygonSelector 类，继承自 _SelectorWidget

    def __init__(
        self,
        ax: Axes,
        onselect: Callable[[ArrayLike, ArrayLike], Any],
        *,
        useblit: bool = ...,
        props: dict[str, Any] | None = ...,
        handle_props: dict[str, Any] | None = ...,
        grab_range: float = ...,
        draw_bounding_box: bool = ...,
        box_handle_props: dict[str, Any] | None = ...,
        box_props: dict[str, Any] | None = ...
    ) -> None:
    # 初始化方法，设置 PolygonSelector 对象的各种属性和参数
    # ax: 绘图的 Axes 对象
    # onselect: 当选择完成时调用的回调函数，参数为两个 ArrayLike 类型的参数，返回任意类型
    # useblit: 是否使用 blitting 进行绘图优化的布尔值，默认为 ...
    # props: 针对多边形选择器的属性字典，字符串到任意类型值的映射或空值
    # handle_props: 处理器属性的字典，字符串到任意类型值的映射或空值
    # grab_range: 抓取范围，浮点数，表示多边形抓取的范围
    # draw_bounding_box: 是否绘制边界框的布尔值
    # box_handle_props: 边界框处理器属性的字典，字符串到任意类型值的映射或空值
    # box_props: 边界框的属性字典，字符串到任意类型值的映射或空值
    # 返回 None

    def onmove(self, event: Event) -> bool:
    # 当鼠标移动事件发生时调用的方法
    # event: 事件对象，描述发生的事件类型和相关信息
    # 返回布尔值，表示是否消费了事件

    @property
    def verts(self) -> list[tuple[float, float]]:
    # 属性方法，获取多边形顶点的列表
    # 返回元组列表，每个元组包含两个浮点数，表示顶点的坐标

    @verts.setter
    def verts(self, xys: Sequence[tuple[float, float]]) -> None:
    # 属性方法，设置多边形顶点的列表
    # xys: 元组序列，包含两个浮点数的元组，表示顶点的坐标
    # 返回 None


class Lasso(AxesWidget):
    useblit: bool
    background: Any
    verts: list[tuple[float, float]] | None
    line: Line2D
    callback: Callable[[list[tuple[float, float]]], Any]
    # Lasso 类，继承自 AxesWidget

    def __init__(
        self,
        ax: Axes,
        xy: tuple[float, float],
        callback: Callable[[list[tuple[float, float]]], Any],
        *,
        useblit: bool = ...,
        props: dict[str, Any] | None = ...,
    ) -> None:
    # 初始化方法，设置 Lasso 对象的各种属性和参数
    # ax: 绘图的 Axes 对象
    # xy: 元组，表示 Lasso 的起始点坐标
    # callback: 当 Lasso 完成时调用的回调函数，参数为顶点坐标列表，返回任意类型
    # useblit: 是否使用 blitting 进行绘图优化的布尔值，默认为 ...
    # props: 针对 Lasso 的属性字典，字符串到任意类型值的映射或空值
    # 返回 None

    def onrelease(self, event: Event) -> None:
    # 当鼠标释放事件发生时调用的方法
    # event: 事件对象，描述发生的事件类型和相关信息
    # 返回 None

    def onmove(self, event: Event) -> None:
    # 当鼠标移动事件发生时调用的方法
    # event: 事件对象，描述发生的事件类型和相关信息
    # 返回 None
```