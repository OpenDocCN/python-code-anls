# `D:\src\scipysrc\matplotlib\lib\matplotlib\lines.pyi`

```py
from .artist import Artist
from .axes import Axes
from .backend_bases import MouseEvent, FigureCanvasBase
from .path import Path
from .transforms import Bbox, Transform

from collections.abc import Callable, Sequence
from typing import Any, Literal, overload
from .typing import (
    ColorType,
    DrawStyleType,
    FillStyleType,
    LineStyleType,
    CapStyleType,
    JoinStyleType,
    MarkEveryType,
    MarkerType,
)
from numpy.typing import ArrayLike

# 定义一个函数 segment_hits，接收多个 ArrayLike 类型的参数，返回一个 ArrayLike 类型的结果
def segment_hits(
    cx: ArrayLike, cy: ArrayLike, x: ArrayLike, y: ArrayLike, radius: ArrayLike
) -> ArrayLike:
    ...

# 定义 Line2D 类，继承自 Artist 类
class Line2D(Artist):
    # 定义类变量
    lineStyles: dict[str, str]  # 线条样式的字典
    drawStyles: dict[str, str]  # 绘制样式的字典
    drawStyleKeys: list[str]    # 绘制样式的键列表
    markers: dict[str | int, str]  # 标记类型的字典
    filled_markers: tuple[str, ...]  # 填充标记的元组
    fillStyles: tuple[str, ...]  # 填充样式的元组
    zorder: float  # 绘制顺序
    ind_offset: float  # 索引偏移量

    # 构造函数，初始化 Line2D 对象
    def __init__(
        self,
        xdata: ArrayLike,
        ydata: ArrayLike,
        *,
        linewidth: float | None = ...,  # 线条宽度
        linestyle: LineStyleType | None = ...,  # 线条样式
        color: ColorType | None = ...,  # 线条颜色
        gapcolor: ColorType | None = ...,  # 间隙颜色
        marker: MarkerType | None = ...,  # 标记类型
        markersize: float | None = ...,  # 标记大小
        markeredgewidth: float | None = ...,  # 标记边缘宽度
        markeredgecolor: ColorType | None = ...,  # 标记边缘颜色
        markerfacecolor: ColorType | None = ...,  # 标记填充颜色
        markerfacecoloralt: ColorType = ...,  # 替代标记填充颜色
        fillstyle: FillStyleType | None = ...,  # 填充样式
        antialiased: bool | None = ...,  # 是否抗锯齿
        dash_capstyle: CapStyleType | None = ...,  # 虚线线端样式
        solid_capstyle: CapStyleType | None = ...,  # 实线线端样式
        dash_joinstyle: JoinStyleType | None = ...,  # 虚线连接样式
        solid_joinstyle: JoinStyleType | None = ...,  # 实线连接样式
        pickradius: float = ...,  # 选择半径
        drawstyle: DrawStyleType | None = ...,  # 绘制样式
        markevery: MarkEveryType | None = ...,  # 每个标记显示频率
        **kwargs  # 其他关键字参数
    ) -> None:
        ...

    # 判断是否包含指定鼠标事件的方法，返回布尔值和包含信息的字典
    def contains(self, mouseevent: MouseEvent) -> tuple[bool, dict]:
        ...

    # 获取选择半径的方法，返回一个浮点数
    def get_pickradius(self) -> float:
        ...

    # 设置选择半径的方法，接收一个浮点数参数，无返回值
    def set_pickradius(self, pickradius: float) -> None:
        ...

    pickradius: float  # 类变量，选择半径

    # 获取填充样式的方法，返回填充样式类型
    def get_fillstyle(self) -> FillStyleType:
        ...

    stale: bool  # 类变量，过时标记

    # 设置填充样式的方法，接收一个填充样式参数，无返回值
    def set_fillstyle(self, fs: FillStyleType) -> None:
        ...

    # 设置每个标记显示频率的方法，接收一个标记频率参数，无返回值
    def set_markevery(self, every: MarkEveryType) -> None:
        ...

    # 获取每个标记显示频率的方法，返回标记频率类型
    def get_markevery(self) -> MarkEveryType:
        ...

    # 设置选择器的方法，接收一个选择器参数，无返回值
    def set_picker(
        self, p: None | bool | float | Callable[[Artist, MouseEvent], tuple[bool, dict]]
    ) -> None:
        ...

    # 获取边界框的方法，返回 Bbox 类型
    def get_bbox(self) -> Bbox:
        ...

    @overload
    # 重载方法，设置数据为一组参数的情况
    def set_data(self, args: ArrayLike) -> None:
        ...

    @overload
    # 重载方法，设置数据为两组参数的情况
    def set_data(self, x: ArrayLike, y: ArrayLike) -> None:
        ...

    # 总是重新缓存数据的方法，无返回值
    def recache_always(self) -> None:
        ...

    # 根据指定条件重新缓存数据的方法，接收一个布尔类型参数，无返回值
    def recache(self, always: bool = ...) -> None:
        ...

    # 获取是否抗锯齿的方法，返回布尔值
    def get_antialiased(self) -> bool:
        ...

    # 获取线条颜色的方法，返回颜色类型
    def get_color(self) -> ColorType:
        ...

    # 获取绘制样式的方法，返回绘制样式类型
    def get_drawstyle(self) -> DrawStyleType:
        ...

    # 获取间隙颜色的方法，返回颜色类型
    def get_gapcolor(self) -> ColorType:
        ...

    # 获取线条样式的方法，返回线条样式类型
    def get_linestyle(self) -> LineStyleType:
        ...

    # 获取线条宽度的方法，返回浮点数
    def get_linewidth(self) -> float:
        ...

    # 获取标记类型的方法，返回标记类型
    def get_marker(self) -> MarkerType:
        ...
    # 返回图形标记边缘颜色的方法
    def get_markeredgecolor(self) -> ColorType: ...
    
    # 返回图形标记边缘宽度的方法
    def get_markeredgewidth(self) -> float: ...
    
    # 返回图形标记内部颜色的方法
    def get_markerfacecolor(self) -> ColorType: ...
    
    # 返回备用图形标记内部颜色的方法
    def get_markerfacecoloralt(self) -> ColorType: ...
    
    # 返回图形标记大小的方法
    def get_markersize(self) -> float: ...
    
    # 返回图形数据的方法，可选是否返回原始数据
    def get_data(self, orig: bool = ...) -> tuple[ArrayLike, ArrayLike]: ...
    
    # 返回图形 X 轴数据的方法，可选是否返回原始数据
    def get_xdata(self, orig: bool = ...) -> ArrayLike: ...
    
    # 返回图形 Y 轴数据的方法，可选是否返回原始数据
    def get_ydata(self, orig: bool = ...) -> ArrayLike: ...
    
    # 返回图形路径的方法
    def get_path(self) -> Path: ...
    
    # 返回图形 XY 数据的方法
    def get_xydata(self) -> ArrayLike: ...
    
    # 设置图形是否抗锯齿的方法
    def set_antialiased(self, b: bool) -> None: ...
    
    # 设置图形颜色的方法
    def set_color(self, color: ColorType) -> None: ...
    
    # 设置图形绘制风格的方法
    def set_drawstyle(self, drawstyle: DrawStyleType | None) -> None: ...
    
    # 设置图形间隙颜色的方法
    def set_gapcolor(self, gapcolor: ColorType | None) -> None: ...
    
    # 设置图形线宽的方法
    def set_linewidth(self, w: float) -> None: ...
    
    # 设置图形线型的方法
    def set_linestyle(self, ls: LineStyleType) -> None: ...
    
    # 设置图形标记的方法
    def set_marker(self, marker: MarkerType) -> None: ...
    
    # 设置图形标记边缘颜色的方法
    def set_markeredgecolor(self, ec: ColorType | None) -> None: ...
    
    # 设置图形标记内部颜色的方法
    def set_markerfacecolor(self, fc: ColorType | None) -> None: ...
    
    # 设置备用图形标记内部颜色的方法
    def set_markerfacecoloralt(self, fc: ColorType | None) -> None: ...
    
    # 设置图形标记边缘宽度的方法
    def set_markeredgewidth(self, ew: float | None) -> None: ...
    
    # 设置图形标记大小的方法
    def set_markersize(self, sz: float) -> None: ...
    
    # 设置图形 X 轴数据的方法
    def set_xdata(self, x: ArrayLike) -> None: ...
    
    # 设置图形 Y 轴数据的方法
    def set_ydata(self, y: ArrayLike) -> None: ...
    
    # 设置图形线型虚线的方法
    def set_dashes(self, seq: Sequence[float] | tuple[None, None]) -> None: ...
    
    # 从其他图形对象更新当前图形对象的方法
    def update_from(self, other: Artist) -> None: ...
    
    # 设置虚线连接风格的方法
    def set_dash_joinstyle(self, s: JoinStyleType) -> None: ...
    
    # 设置实线连接风格的方法
    def set_solid_joinstyle(self, s: JoinStyleType) -> None: ...
    
    # 获取虚线连接风格的方法
    def get_dash_joinstyle(self) -> Literal["miter", "round", "bevel"]: ...
    
    # 获取实线连接风格的方法
    def get_solid_joinstyle(self) -> Literal["miter", "round", "bevel"]: ...
    
    # 设置虚线端点风格的方法
    def set_dash_capstyle(self, s: CapStyleType) -> None: ...
    
    # 设置实线端点风格的方法
    def set_solid_capstyle(self, s: CapStyleType) -> None: ...
    
    # 获取虚线端点风格的方法
    def get_dash_capstyle(self) -> Literal["butt", "projecting", "round"]: ...
    
    # 获取实线端点风格的方法
    def get_solid_capstyle(self) -> Literal["butt", "projecting", "round"]: ...
    
    # 检查图形是否为虚线的方法
    def is_dashed(self) -> bool: ...
class AxLine(Line2D):
    # AxLine 类继承自 Line2D 类，用于表示一个直线对象
    def __init__(
        self,
        xy1: tuple[float, float],
        xy2: tuple[float, float] | None,
        slope: float | None,
        **kwargs
    ) -> None:
        # 初始化函数，接受起点坐标 xy1、终点坐标 xy2、斜率 slope 和其他关键字参数
        ...

    def get_xy1(self) -> tuple[float, float] | None:
        # 返回直线的起点坐标 xy1
        ...

    def get_xy2(self) -> tuple[float, float] | None:
        # 返回直线的终点坐标 xy2
        ...

    def get_slope(self) -> float:
        # 返回直线的斜率 slope
        ...

    def set_xy1(self, x: float, y: float) -> None:
        # 设置直线的起点坐标为 (x, y)
        ...

    def set_xy2(self, x: float, y: float) -> None:
        # 设置直线的终点坐标为 (x, y)
        ...

    def set_slope(self, slope: float) -> None:
        # 设置直线的斜率为 slope
        ...


class VertexSelector:
    # 顶点选择器类
    axes: Axes  # 坐标轴对象
    line: Line2D  # Line2D 对象，表示要操作的直线
    cid: int  # 事件回调标识符
    ind: set[int]  # 选中的顶点索引集合

    def __init__(self, line: Line2D) -> None:
        # 初始化函数，接受一个 Line2D 对象作为参数
        ...

    @property
    def canvas(self) -> FigureCanvasBase:
        # 返回与顶点选择器关联的画布对象
        ...

    def process_selected(
        self, ind: Sequence[int], xs: ArrayLike, ys: ArrayLike
    ) -> None:
        # 处理选中的顶点，传入选中的索引、对应的 x 坐标集合 xs 和 y 坐标集合 ys
        ...

    def onpick(self, event: Any) -> None:
        # 处理选中事件的回调函数，接受事件对象 event
        ...


lineStyles: dict[str, str]
# 线条样式字典，映射线条样式名称到样式定义字符串的字典

lineMarkers: dict[str | int, str]
# 线条标记字典，映射标记名称或整数标记到标记样式字符串的字典

drawStyles: dict[str, str]
# 绘制样式字典，映射绘制样式名称到样式定义字符串的字典

fillStyles: tuple[FillStyleType, ...]
# 填充样式元组，包含不同填充样式的元组
```