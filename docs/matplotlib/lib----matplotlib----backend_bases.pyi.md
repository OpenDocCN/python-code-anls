# `D:\src\scipysrc\matplotlib\lib\matplotlib\backend_bases.pyi`

```py
# 导入枚举和整数枚举
from enum import Enum, IntEnum
# 导入操作系统接口
import os
# 导入 Matplotlib 相关模块
from matplotlib import (
    cbook,         # Matplotlib 内部工具模块
    transforms,    # 数据转换模块
    widgets,       # Matplotlib 控件模块
    _api,          # Matplotlib 内部 API
)
from matplotlib.artist import Artist            # Matplotlib 图形对象基类
from matplotlib.axes import Axes                # Matplotlib 坐标轴对象
from matplotlib.backend_managers import ToolManager  # Matplotlib 后端管理器
from matplotlib.backend_tools import Cursors, ToolBase  # Matplotlib 后端工具
from matplotlib.colorbar import Colorbar        # Matplotlib 颜色条对象
from matplotlib.figure import Figure            # Matplotlib 图形对象
from matplotlib.font_manager import FontProperties  # Matplotlib 字体管理器
from matplotlib.path import Path                # Matplotlib 路径对象
from matplotlib.texmanager import TexManager    # Matplotlib TeX 渲染管理器
from matplotlib.text import Text                # Matplotlib 文本对象
from matplotlib.transforms import (             # Matplotlib 坐标变换模块
    Bbox, BboxBase, Transform, TransformedPath
)

from collections.abc import Callable, Iterable, Sequence  # 标准库集合抽象基类
from typing import Any, IO, Literal, NamedTuple, TypeVar  # 类型提示
from numpy.typing import ArrayLike                   # NumPy 类型提示
from .typing import ColorType, LineStyleType, CapStyleType, JoinStyleType  # 相对导入的类型提示模块

# 注册后端的函数，格式参数指定格式，后端参数指定对应的画布类
def register_backend(
    format: str, backend: str | type[FigureCanvasBase], description: str | None = ...
) -> None: ...

# 根据格式获取已注册的画布类
def get_registered_canvas_class(format: str) -> type[FigureCanvasBase]: ...

# 渲染器基类，实现基本的绘图方法
class RendererBase:
    # 初始化方法
    def __init__(self) -> None: ...
    
    # 开始绘制一个组
    def open_group(self, s: str, gid: str | None = ...) -> None: ...
    
    # 结束绘制一个组
    def close_group(self, s: str) -> None: ...
    
    # 绘制路径
    def draw_path(
        self,
        gc: GraphicsContextBase,     # 图形上下文基类
        path: Path,                 # 路径对象
        transform: Transform,       # 坐标变换对象
        rgbFace: ColorType | None = ...,  # 面部颜色（可选）
    ) -> None: ...
    
    # 绘制标记
    def draw_markers(
        self,
        gc: GraphicsContextBase,     # 图形上下文基类
        marker_path: Path,          # 标记路径对象
        marker_trans: Transform,    # 标记坐标变换对象
        path: Path,                 # 路径对象
        trans: Transform,           # 坐标变换对象
        rgbFace: ColorType | None = ...,  # 面部颜色（可选）
    ) -> None: ...
    
    # 绘制路径集合
    def draw_path_collection(
        self,
        gc: GraphicsContextBase,     # 图形上下文基类
        master_transform: Transform, # 主坐标变换对象
        paths: Sequence[Path],      # 路径对象序列
        all_transforms: Sequence[ArrayLike],  # 所有坐标变换序列
        offsets: ArrayLike | Sequence[ArrayLike],  # 偏移量序列
        offset_trans: Transform,    # 偏移量坐标变换对象
        facecolors: ColorType | Sequence[ColorType],  # 面部颜色或颜色序列
        edgecolors: ColorType | Sequence[ColorType],  # 边缘颜色或颜色序列
        linewidths: float | Sequence[float],  # 线宽或线宽序列
        linestyles: LineStyleType | Sequence[LineStyleType],  # 线型或线型序列
        antialiaseds: bool | Sequence[bool],  # 抗锯齿标志或标志序列
        urls: str | Sequence[str],   # URL 或 URL 序列
        offset_position: Any,        # 偏移位置
    ) -> None: ...
    
    # 绘制四边形网格
    def draw_quad_mesh(
        self,
        gc: GraphicsContextBase,     # 图形上下文基类
        master_transform: Transform, # 主坐标变换对象
        meshWidth,                  # 网格宽度
        meshHeight,                 # 网格高度
        coordinates: ArrayLike,     # 坐标数组
        offsets: ArrayLike | Sequence[ArrayLike],  # 偏移量数组或序列
        offsetTrans: Transform,     # 偏移量坐标变换对象
        facecolors: Sequence[ColorType],  # 面部颜色序列
        antialiased: bool,          # 抗锯齿标志
        edgecolors: Sequence[ColorType] | ColorType | None,  # 边缘颜色序列、颜色或空
    ) -> None: ...
    
    # 绘制高洛德三角形
    def draw_gouraud_triangles(
        self,
        gc: GraphicsContextBase,     # 图形上下文基类
        triangles_array: ArrayLike,  # 三角形数组
        colors_array: ArrayLike,     # 颜色数组
        transform: Transform,       # 坐标变换对象
    ) -> None: ...
    
    # 获取图像放大倍率
    def get_image_magnification(self) -> float: ...
    # 定义方法：在画布上绘制图像
    def draw_image(
        self,
        gc: GraphicsContextBase,
        x: float,
        y: float,
        im: ArrayLike,
        transform: transforms.Affine2DBase | None = ...,
    ) -> None: ...
    
    # 定义方法：检查是否启用了图像合成选项
    def option_image_nocomposite(self) -> bool: ...
    
    # 定义方法：检查是否启用了图像缩放选项
    def option_scale_image(self) -> bool: ...
    
    # 定义方法：在画布上绘制文本
    def draw_tex(
        self,
        gc: GraphicsContextBase,
        x: float,
        y: float,
        s: str,
        prop: FontProperties,
        angle: float,
        *,
        mtext: Text | None = ...
    ) -> None: ...
    
    # 定义方法：在画布上绘制文本
    def draw_text(
        self,
        gc: GraphicsContextBase,
        x: float,
        y: float,
        s: str,
        prop: FontProperties,
        angle: float,
        ismath: bool | Literal["TeX"] = ...,
        mtext: Text | None = ...,
    ) -> None: ...
    
    # 定义方法：获取给定文本的宽度、高度和下降距离
    def get_text_width_height_descent(
        self, s: str, prop: FontProperties, ismath: bool | Literal["TeX"]
    ) -> tuple[float, float, float]: ...
    
    # 定义方法：检查是否启用了 Y 轴翻转
    def flipy(self) -> bool: ...
    
    # 定义方法：获取画布的宽度和高度
    def get_canvas_width_height(self) -> tuple[float, float]: ...
    
    # 定义方法：获取 TeX 管理器实例
    def get_texmanager(self) -> TexManager: ...
    
    # 定义方法：创建新的绘图上下文对象
    def new_gc(self) -> GraphicsContextBase: ...
    
    # 定义方法：将点数转换为像素数
    def points_to_pixels(self, points: ArrayLike) -> ArrayLike: ...
    
    # 定义方法：开始栅格化过程
    def start_rasterizing(self) -> None: ...
    
    # 定义方法：结束栅格化过程
    def stop_rasterizing(self) -> None: ...
    
    # 定义方法：开始过滤器应用
    def start_filter(self) -> None: ...
    
    # 定义方法：结束过滤器应用
    def stop_filter(self, filter_func) -> None: ...
class GraphicsContextBase:
    # 图形上下文基类，用于绘图属性管理和设置

    def __init__(self) -> None:
        # 初始化函数，无需执行任何操作
        ...

    def copy_properties(self, gc: GraphicsContextBase) -> None:
        # 复制另一个图形上下文的属性到当前实例
        ...

    def restore(self) -> None:
        # 恢复先前保存的图形上下文状态
        ...

    def get_alpha(self) -> float:
        # 获取当前透明度值
        ...

    def get_antialiased(self) -> int:
        # 获取当前抗锯齿设置
        ...

    def get_capstyle(self) -> Literal["butt", "projecting", "round"]:
        # 获取当前线段端点样式设置
        ...

    def get_clip_rectangle(self) -> Bbox | None:
        # 获取当前剪切矩形区域设置
        ...

    def get_clip_path(self) -> tuple[TransformedPath, Transform] | tuple[None, None]:
        # 获取当前剪切路径设置
        ...

    def get_dashes(self) -> tuple[float, ArrayLike | None]:
        # 获取当前虚线样式设置
        ...

    def get_forced_alpha(self) -> bool:
        # 检查是否强制使用透明度
        ...

    def get_joinstyle(self) -> Literal["miter", "round", "bevel"]:
        # 获取当前线段连接样式设置
        ...

    def get_linewidth(self) -> float:
        # 获取当前线宽设置
        ...

    def get_rgb(self) -> tuple[float, float, float, float]:
        # 获取当前颜色设置（RGBA）
        ...

    def get_url(self) -> str | None:
        # 获取与图形上下文关联的 URL
        ...

    def get_gid(self) -> int | None:
        # 获取图形上下文的组ID
        ...

    def get_snap(self) -> bool | None:
        # 获取是否启用对齐设置
        ...

    def set_alpha(self, alpha: float) -> None:
        # 设置透明度值
        ...

    def set_antialiased(self, b: bool) -> None:
        # 设置是否启用抗锯齿
        ...

    def set_capstyle(self, cs: CapStyleType) -> None:
        # 设置线段端点样式
        ...

    def set_clip_rectangle(self, rectangle: Bbox | None) -> None:
        # 设置剪切矩形区域
        ...

    def set_clip_path(self, path: TransformedPath | None) -> None:
        # 设置剪切路径
        ...

    def set_dashes(self, dash_offset: float, dash_list: ArrayLike | None) -> None:
        # 设置虚线样式
        ...

    def set_foreground(self, fg: ColorType, isRGBA: bool = ...) -> None:
        # 设置前景色
        ...

    def set_joinstyle(self, js: JoinStyleType) -> None:
        # 设置线段连接样式
        ...

    def set_linewidth(self, w: float) -> None:
        # 设置线宽
        ...

    def set_url(self, url: str | None) -> None:
        # 设置与图形上下文关联的 URL
        ...

    def set_gid(self, id: int | None) -> None:
        # 设置图形上下文的组ID
        ...

    def set_snap(self, snap: bool | None) -> None:
        # 设置是否启用对齐设置
        ...

    def set_hatch(self, hatch: str | None) -> None:
        # 设置图案填充样式
        ...

    def get_hatch(self) -> str | None:
        # 获取当前图案填充样式
        ...

    def get_hatch_path(self, density: float = ...) -> Path:
        # 获取图案填充的路径对象
        ...

    def get_hatch_color(self) -> ColorType:
        # 获取图案填充的颜色
        ...

    def set_hatch_color(self, hatch_color: ColorType) -> None:
        # 设置图案填充的颜色
        ...

    def get_hatch_linewidth(self) -> float:
        # 获取图案填充的线宽
        ...

    def get_sketch_params(self) -> tuple[float, float, float] | None:
        # 获取当前草图参数
        ...

    def set_sketch_params(
        self,
        scale: float | None = ...,
        length: float | None = ...,
        randomness: float | None = ...,
    ) -> None:
        # 设置草图参数
        ...


class TimerBase:
    # 计时器基类，用于定时事件管理

    callbacks: list[tuple[Callable, tuple, dict[str, Any]]]

    def __init__(
        self,
        interval: int | None = ...,
        callbacks: list[tuple[Callable, tuple, dict[str, Any]]] | None = ...,
    ) -> None:
        # 初始化函数，设置计时器的间隔和回调函数列表
        ...

    def __del__(self) -> None:
        # 析构函数，清理资源
        ...

    def start(self, interval: int | None = ...) -> None:
        # 启动计时器
        ...

    def stop(self) -> None:
        # 停止计时器
        ...

    @property
    def interval(self) -> int:
        # 获取当前计时器的间隔时间
        ...

    @interval.setter
    def interval(self, interval: int) -> None:
        # 设置计时器的间隔时间
        ...

    @property
    def single_shot(self) -> bool:
        # 检查计时器是否为单次触发模式
        ...

    @single_shot.setter
    def single_shot(self, ss: bool) -> None:
        # 设置计时器是否为单次触发模式
        ...

    def add_callback(self, func: Callable, *args, **kwargs) -> Callable:
        # 添加回调函数到计时器
        ...

    def remove_callback(self, func: Callable, *args, **kwargs) -> None:
        # 移除计时器的回调函数
        ...


class Event:
    # 事件类，用于描述事件的名称

    name: str
    # 定义一个类属性 canvas，表示图形画布
    canvas: FigureCanvasBase
    
    # 初始化方法，接受名称、图形画布和 GUI 事件作为参数
    def __init__(
        self, name: str, canvas: FigureCanvasBase, guiEvent: Any | None = ...
    ) -> None: ...
    
    # 定义一个属性 guiEvent，表示 GUI 事件
    @property
    def guiEvent(self) -> Any: ...
class DrawEvent(Event):
    renderer: RendererBase
    def __init__(
        self, name: str, canvas: FigureCanvasBase, renderer: RendererBase
    ) -> None:
        # 初始化绘图事件对象
        super().__init__(name, canvas)
        # 设置渲染器属性
        self.renderer = renderer


class ResizeEvent(Event):
    width: int
    height: int
    def __init__(self, name: str, canvas: FigureCanvasBase) -> None:
        # 初始化调整大小事件对象
        super().__init__(name, canvas)


class CloseEvent(Event):
    def __init__(self, name: str) -> None:
        # 初始化关闭事件对象
        super().__init__(name)


class LocationEvent(Event):
    lastevent: Event | None
    x: int
    y: int
    inaxes: Axes | None
    xdata: float | None
    ydata: float | None
    def __init__(
        self,
        name: str,
        canvas: FigureCanvasBase,
        x: int,
        y: int,
        guiEvent: Any | None = ...,
        *,
        modifiers: Iterable[str] | None = ...,
    ) -> None:
        # 初始化位置事件对象
        super().__init__(name, canvas)
        self.x = x
        self.y = y
        self.guiEvent = guiEvent
        self.modifiers = modifiers


class MouseButton(IntEnum):
    LEFT: int
    MIDDLE: int
    RIGHT: int
    BACK: int
    FORWARD: int

class MouseEvent(LocationEvent):
    button: MouseButton | Literal["up", "down"] | None
    key: str | None
    step: float
    dblclick: bool
    def __init__(
        self,
        name: str,
        canvas: FigureCanvasBase,
        x: int,
        y: int,
        button: MouseButton | Literal["up", "down"] | None = ...,
        key: str | None = ...,
        step: float = ...,
        dblclick: bool = ...,
        guiEvent: Any | None = ...,
        *,
        modifiers: Iterable[str] | None = ...,
    ) -> None:
        # 初始化鼠标事件对象
        super().__init__(name, canvas, x, y, guiEvent, modifiers)
        self.button = button
        self.key = key
        self.step = step
        self.dblclick = dblclick


class PickEvent(Event):
    mouseevent: MouseEvent
    artist: Artist
    def __init__(
        self,
        name: str,
        canvas: FigureCanvasBase,
        mouseevent: MouseEvent,
        artist: Artist,
        guiEvent: Any | None = ...,
        **kwargs
    ) -> None:
        # 初始化拾取事件对象
        super().__init__(name, canvas)
        self.mouseevent = mouseevent
        self.artist = artist


class KeyEvent(LocationEvent):
    key: str | None
    def __init__(
        self,
        name: str,
        canvas: FigureCanvasBase,
        key: str | None,
        x: int = ...,
        y: int = ...,
        guiEvent: Any | None = ...,
    ) -> None:
        # 初始化键盘事件对象
        super().__init__(name, canvas, x, y, guiEvent)
        self.key = key


class FigureCanvasBase:
    required_interactive_framework: str | None

    @_api.classproperty
    def manager_class(cls) -> type[FigureManagerBase]:
        # 获取管理器类的属性
        ...

    events: list[str]
    fixed_dpi: None | float
    filetypes: dict[str, str]

    @_api.classproperty
    def supports_blit(cls) -> bool:
        # 获取是否支持位块传输的属性
        ...

    figure: Figure
    manager: None | FigureManagerBase
    widgetlock: widgets.LockDraw
    mouse_grabber: None | Axes
    toolbar: None | NavigationToolbar2
    def __init__(self, figure: Figure | None = ...) -> None:
        # 初始化图形画布基类对象
        self.figure = figure
        self.manager = None
        self.widgetlock = widgets.LockDraw()
        self.mouse_grabber = None
        self.toolbar = None

    @property
    def callbacks(self) -> cbook.CallbackRegistry:
        # 获取回调函数注册表的属性
        ...

    @property
    def button_pick_id(self) -> int:
        # 获取按钮拾取ID的属性
        ...

    @property
    def scroll_pick_id(self) -> int:
        # 获取滚动条拾取ID的属性
        ...

    @classmethod
    def new_manager(cls, figure: Figure, num: int | str) -> FigureManagerBase:
        # 创建新的管理器对象
        ...

    def is_saving(self) -> bool:
        # 判断是否正在保存
        ...

    def blit(self, bbox: BboxBase | None = ...) -> None:
        # 进行位块传输
        ...

    def inaxes(self, xy: tuple[float, float]) -> Axes | None:
        # 判断指定位置是否在轴内
        ...

    def grab_mouse(self, ax: Axes) -> None:
        # 抓取鼠标焦点到指定的轴上
        ...

    def release_mouse(self, ax: Axes) -> None:
        # 释放鼠标焦点
        ...
    # 设置图形的光标样式
    def set_cursor(self, cursor: Cursors) -> None: ...

    # 绘制图形
    def draw(self, *args, **kwargs) -> None: ...

    # 在空闲时绘制图形
    def draw_idle(self, *args, **kwargs) -> None: ...

    # 获取设备像素比例
    @property
    def device_pixel_ratio(self) -> float: ...

    # 获取图形的宽度和高度
    def get_width_height(self, *, physical: bool = ...) -> tuple[int, int]: ...

    # 获取支持的文件类型及其描述的字典
    @classmethod
    def get_supported_filetypes(cls) -> dict[str, str]: ...

    # 获取支持的文件类型并按组分类
    @classmethod
    def get_supported_filetypes_grouped(cls) -> dict[str, list[str]]: ...

    # 打印图形到指定的文件
    def print_figure(
        self,
        filename: str | os.PathLike | IO,
        dpi: float | None = ...,
        facecolor: ColorType | Literal["auto"] | None = ...,
        edgecolor: ColorType | Literal["auto"] | None = ...,
        orientation: str = ...,
        format: str | None = ...,
        *,
        bbox_inches: Literal["tight"] | Bbox | None = ...,
        pad_inches: float | None = ...,
        bbox_extra_artists: list[Artist] | None = ...,
        backend: str | None = ...,
        **kwargs
    ) -> Any: ...

    # 获取默认的文件类型
    @classmethod
    def get_default_filetype(cls) -> str: ...

    # 获取默认的文件名
    def get_default_filename(self) -> str: ...

    # 切换到指定的图形后端
    _T = TypeVar("_T", bound=FigureCanvasBase)
    def switch_backends(self, FigureCanvasClass: type[_T]) -> _T: ...

    # 注册绘图事件的回调函数
    def mpl_connect(self, s: str, func: Callable[[Event], Any]) -> int: ...

    # 断开已注册的绘图事件回调函数
    def mpl_disconnect(self, cid: int) -> None: ...

    # 创建一个新的定时器对象
    def new_timer(
        self,
        interval: int | None = ...,
        callbacks: list[tuple[Callable, tuple, dict[str, Any]]] | None = ...,
    ) -> TimerBase: ...

    # 刷新绘图事件队列
    def flush_events(self) -> None: ...

    # 开始事件循环
    def start_event_loop(self, timeout: float = ...) -> None: ...

    # 停止事件循环
    def stop_event_loop(self) -> None: ...
# 处理键盘按键事件的处理程序，绑定于特定画布和工具栏（可选）
def key_press_handler(
    event: KeyEvent,
    canvas: FigureCanvasBase | None = ...,
    toolbar: NavigationToolbar2 | None = ...,
) -> None: ...

# 处理鼠标点击事件的处理程序，绑定于特定画布和工具栏（可选）
def button_press_handler(
    event: MouseEvent,
    canvas: FigureCanvasBase | None = ...,
    toolbar: NavigationToolbar2 | None = ...,
) -> None: ...

# 自定义异常类，用于非 GUI 模式下的异常处理
class NonGuiException(Exception): ...

# 图形管理基类，管理与图形画布相关的操作
class FigureManagerBase:
    canvas: FigureCanvasBase  # 图形画布对象
    num: int | str  # 图形编号，可以是整数或字符串
    key_press_handler_id: int | None  # 键盘按键处理程序的 ID，可为 None
    button_press_handler_id: int | None  # 鼠标点击处理程序的 ID，可为 None
    toolmanager: ToolManager | None  # 工具管理器对象，可为 None
    toolbar: NavigationToolbar2 | ToolContainerBase | None  # 导航工具栏对象，可为 None

    # 初始化方法，接受一个图形画布对象和图形编号作为参数
    def __init__(self, canvas: FigureCanvasBase, num: int | str) -> None: ...

    # 类方法：使用指定的图形画布类和图形创建图形管理器对象
    @classmethod
    def create_with_canvas(
        cls, canvas_class: type[FigureCanvasBase], figure: Figure, num: int | str
    ) -> FigureManagerBase: ...

    # 类方法：启动主循环
    @classmethod
    def start_main_loop(cls) -> None: ...

    # 类方法：显示图形，支持阻塞模式（可选）
    @classmethod
    def pyplot_show(cls, *, block: bool | None = ...) -> None: ...

    # 显示图形方法
    def show(self) -> None: ...

    # 销毁图形方法
    def destroy(self) -> None: ...

    # 切换全屏方法
    def full_screen_toggle(self) -> None: ...

    # 调整窗口大小方法，接受宽度和高度参数
    def resize(self, w: int, h: int) -> None: ...

    # 获取窗口标题方法
    def get_window_title(self) -> str: ...

    # 设置窗口标题方法，接受标题字符串作为参数
    def set_window_title(self, title: str) -> None: ...

# 鼠标指针样式常量定义
cursors = Cursors

# 模式枚举类，定义交互模式类型
class _Mode(str, Enum):
    NONE: str  # 无模式
    PAN: str  # 平移模式
    ZOOM: str  # 缩放模式

# 导航工具栏类，管理图形画布的导航工具
class NavigationToolbar2:
    toolitems: tuple[tuple[str, ...] | tuple[None, ...], ...]  # 工具栏按钮项元组
    canvas: FigureCanvasBase  # 关联的图形画布对象
    mode: _Mode  # 当前的交互模式

    # 初始化方法，接受一个图形画布对象作为参数
    def __init__(self, canvas: FigureCanvasBase) -> None: ...

    # 设置状态栏消息方法，接受消息字符串作为参数
    def set_message(self, s: str) -> None: ...

    # 绘制选框方法，接受事件和选框的起始点和终止点坐标作为参数
    def draw_rubberband(
        self, event: Event, x0: float, y0: float, x1: float, y1: float
    ) -> None: ...

    # 移除选框方法
    def remove_rubberband(self) -> None: ...

    # 返回主视图方法，接受可选的参数
    def home(self, *args) -> None: ...

    # 返回上一个视图方法，接受可选的参数
    def back(self, *args) -> None: ...

    # 前进到下一个视图方法，接受可选的参数
    def forward(self, *args) -> None: ...

    # 处理鼠标移动事件方法，接受事件对象作为参数
    def mouse_move(self, event: MouseEvent) -> None: ...

    # 执行平移操作方法，接受可选的参数
    def pan(self, *args) -> None: ...

    # 内部类：平移操作信息，包含按钮、轴列表和回调函数 ID
    class _PanInfo(NamedTuple):
        button: MouseButton  # 鼠标按键
        axes: list[Axes]  # 图轴列表
        cid: int  # 回调函数 ID

    # 处理平移操作按下事件方法，接受事件对象作为参数
    def press_pan(self, event: Event) -> None: ...

    # 处理平移操作拖动事件方法，接受事件对象作为参数
    def drag_pan(self, event: Event) -> None: ...

    # 处理平移操作释放事件方法，接受事件对象作为参数
    def release_pan(self, event: Event) -> None: ...

    # 执行缩放操作方法，接受可选的参数
    def zoom(self, *args) -> None: ...

    # 内部类：缩放操作信息，包含缩放方向、起始坐标、轴列表、回调函数 ID 和颜色条对象
    class _ZoomInfo(NamedTuple):
        direction: Literal["in", "out"]  # 缩放方向，放大或缩小
        start_xy: tuple[float, float]  # 起始坐标
        axes: list[Axes]  # 图轴列表
        cid: int  # 回调函数 ID
        cbar: Colorbar  # 颜色条对象

    # 处理缩放操作按下事件方法，接受事件对象作为参数
    def press_zoom(self, event: Event) -> None: ...

    # 处理缩放操作拖动事件方法，接受事件对象作为参数
    def drag_zoom(self, event: Event) -> None: ...

    # 处理缩放操作释放事件方法，接受事件对象作为参数
    def release_zoom(self, event: Event) -> None: ...

    # 保存当前状态方法，暂未实现具体功能
    def push_current(self) -> None: ...

    subplot_tool: widgets.SubplotTool  # 子图工具对象

    # 配置子图方法，接受可选的参数
    def configure_subplots(self, *args): ...

    # 保存图形方法，接受可选的参数
    def save_figure(self, *args) -> None: ...

    # 更新视图方法
    def update(self) -> None: ...

    # 设置历史记录按钮方法
    def set_history_buttons(self) -> None: ...

# 工具容器基类，管理工具管理器对象
class ToolContainerBase:
    toolmanager: ToolManager  # 工具管理器对象

    # 初始化方法，接受工具管理器对象作为参数
    def __init__(self, toolmanager: ToolManager) -> None: ...
    # 将工具添加到工具栏中的方法，接受工具对象和其分组信息作为参数
    def add_tool(self, tool: ToolBase, group: str, position: int = ...) -> None:
        ...
    
    # 触发特定名称工具的方法，传入工具的名称作为参数
    def trigger_tool(self, name: str) -> None:
        ...
    
    # 添加工具项到工具栏的方法，接受工具项的名称、分组、位置、图像、描述和是否可切换作为参数
    def add_toolitem(
        self,
        name: str,
        group: str,
        position: int,
        image: str,
        description: str,
        toggle: bool,
    ) -> None:
        ...
    
    # 切换特定名称工具项的状态（开/关），接受工具项名称和布尔值表示状态作为参数
    def toggle_toolitem(self, name: str, toggled: bool) -> None:
        ...
    
    # 移除特定名称工具项的方法，接受工具项名称作为参数
    def remove_toolitem(self, name: str) -> None:
        ...
    
    # 设置消息文本的方法，接受字符串作为消息内容参数
    def set_message(self, s: str) -> None:
        ...
# 定义一个名为 _Backend 的类，表示后端引擎
class _Backend:
    # 版本信息，字符串类型
    backend_version: str
    # FigureCanvas 可能是 FigureCanvasBase 类型或 None
    FigureCanvas: type[FigureCanvasBase] | None
    # FigureManager 表示 FigureManagerBase 类型
    FigureManager: type[FigureManagerBase]
    # mainloop 可能是 None 或者一个无参数返回任意类型的可调用对象
    mainloop: None | Callable[[], Any]

    # 类方法：根据指定参数创建新的 FigureManagerBase 对象
    @classmethod
    def new_figure_manager(cls, num: int | str, *args, **kwargs) -> FigureManagerBase: ...

    # 类方法：根据给定的 Figure 对象创建新的 FigureManagerBase 对象
    @classmethod
    def new_figure_manager_given_figure(cls, num: int | str, figure: Figure) -> FigureManagerBase: ...

    # 类方法：如果是交互式环境，执行绘图操作
    @classmethod
    def draw_if_interactive(cls) -> None: ...

    # 类方法：显示操作，可选是否阻塞
    @classmethod
    def show(cls, *, block: bool | None = ...) -> None: ...

    # 静态方法：导出当前类的类型信息
    @staticmethod
    def export(cls) -> type[_Backend]: ...

# ShowBase 类继承自 _Backend 类
class ShowBase(_Backend):
    # 实例化对象时的调用方法，可选参数 block 用于控制是否阻塞
    def __call__(self, block: bool | None = ...) -> None: ...
```