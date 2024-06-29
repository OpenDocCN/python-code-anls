# `D:\src\scipysrc\matplotlib\lib\matplotlib\patches.pyi`

```py
from . import artist
from .axes import Axes
from .backend_bases import RendererBase, MouseEvent
from .path import Path
from .transforms import Transform, Bbox

from typing import Any, Literal, overload

import numpy as np
from numpy.typing import ArrayLike
from .typing import ColorType, LineStyleType, CapStyleType, JoinStyleType

# 导入必要的模块和类

class Patch(artist.Artist):
    zorder: float
    def __init__(
        self,
        *,
        edgecolor: ColorType | None = ...,
        facecolor: ColorType | None = ...,
        color: ColorType | None = ...,
        linewidth: float | None = ...,
        linestyle: LineStyleType | None = ...,
        antialiased: bool | None = ...,
        hatch: str | None = ...,
        fill: bool = ...,
        capstyle: CapStyleType | None = ...,
        joinstyle: JoinStyleType | None = ...,
        **kwargs,
    ) -> None:
        # 初始化 Patch 对象，设置各种属性，包括边缘颜色、填充颜色、线宽、线型等
        ...

    def get_verts(self) -> ArrayLike:
        # 返回表示 Patch 外形的顶点数组
        ...

    def contains(self, mouseevent: MouseEvent, radius: float | None = None) -> tuple[bool, dict[Any, Any]]:
        # 检查鼠标事件是否包含在 Patch 中，并返回是否包含及其它相关信息
        ...

    def contains_point(
        self, point: tuple[float, float], radius: float | None = ...
    ) -> bool:
        # 检查指定点是否包含在 Patch 中
        ...

    def contains_points(
        self, points: ArrayLike, radius: float | None = ...
    ) -> np.ndarray:
        # 检查一组点是否包含在 Patch 中，返回一个布尔型的 NumPy 数组
        ...

    def get_extents(self) -> Bbox:
        # 返回 Patch 的范围边界框（Bbox 对象）
        ...

    def get_transform(self) -> Transform:
        # 返回 Patch 的当前变换
        ...

    def get_data_transform(self) -> Transform:
        # 返回 Patch 的数据变换
        ...

    def get_patch_transform(self) -> Transform:
        # 返回 Patch 的 Patch 变换
        ...

    def get_antialiased(self) -> bool:
        # 返回是否反锯齿
        ...

    def get_edgecolor(self) -> ColorType:
        # 返回边缘颜色
        ...

    def get_facecolor(self) -> ColorType:
        # 返回填充颜色
        ...

    def get_linewidth(self) -> float:
        # 返回线宽
        ...

    def get_linestyle(self) -> LineStyleType:
        # 返回线型
        ...

    def set_antialiased(self, aa: bool | None) -> None:
        # 设置反锯齿属性
        ...

    def set_edgecolor(self, color: ColorType | None) -> None:
        # 设置边缘颜色
        ...

    def set_facecolor(self, color: ColorType | None) -> None:
        # 设置填充颜色
        ...

    def set_color(self, c: ColorType | None) -> None:
        # 设置颜色
        ...

    def set_alpha(self, alpha: float | None) -> None:
        # 设置透明度
        ...

    def set_linewidth(self, w: float | None) -> None:
        # 设置线宽
        ...

    def set_linestyle(self, ls: LineStyleType | None) -> None:
        # 设置线型
        ...

    def set_fill(self, b: bool) -> None:
        # 设置是否填充
        ...

    def get_fill(self) -> bool:
        # 返回是否填充
        ...

    fill = property(get_fill, set_fill)

    def set_capstyle(self, s: CapStyleType) -> None:
        # 设置线帽风格
        ...

    def get_capstyle(self) -> Literal["butt", "projecting", "round"]:
        # 返回线帽风格
        ...

    def set_joinstyle(self, s: JoinStyleType) -> None:
        # 设置线连接风格
        ...

    def get_joinstyle(self) -> Literal["miter", "round", "bevel"]:
        # 返回线连接风格
        ...

    def set_hatch(self, hatch: str) -> None:
        # 设置填充图案
        ...

    def get_hatch(self) -> str:
        # 返回填充图案
        ...

    def get_path(self) -> Path:
        # 返回 Patch 的路径对象
        ...

class Shadow(Patch):
    patch: Patch
    def __init__(self, patch: Patch, ox: float, oy: float, *, shade: float = ..., **kwargs) -> None:
        # 初始化 Shadow 对象，设置阴影相关属性
        ...

class Rectangle(Patch):
    angle: float
    # 初始化函数，用于创建一个矩形对象
    def __init__(
        self,
        xy: tuple[float, float],           # 矩形左下角的坐标
        width: float,                      # 矩形的宽度
        height: float,                     # 矩形的高度
        *,
        angle: float = ...,                # 可选参数，矩形的旋转角度
        rotation_point: Literal["xy", "center"] | tuple[float, float] = ...,  # 可选参数，旋转中心点
        **kwargs,
    ) -> None: ...
    
    @property
    # 获取当前的旋转中心点（属性）
    def rotation_point(self) -> Literal["xy", "center"] | tuple[float, float]: ...
    
    @rotation_point.setter
    # 设置当前的旋转中心点（属性）
    def rotation_point(
        self, value: Literal["xy", "center"] | tuple[float, float]
    ) -> None: ...
    
    # 获取矩形的左边界 x 坐标
    def get_x(self) -> float: ...
    
    # 获取矩形的底边界 y 坐标
    def get_y(self) -> float: ...
    
    # 获取矩形的左下角坐标 (x, y)
    def get_xy(self) -> tuple[float, float]: ...
    
    # 获取矩形的四个角点坐标（返回一个 NumPy 数组）
    def get_corners(self) -> np.ndarray: ...
    
    # 获取矩形的中心点坐标（返回一个 NumPy 数组）
    def get_center(self) -> np.ndarray: ...
    
    # 获取矩形的宽度
    def get_width(self) -> float: ...
    
    # 获取矩形的高度
    def get_height(self) -> float: ...
    
    # 获取矩形的旋转角度
    def get_angle(self) -> float: ...
    
    # 设置矩形的左边界 x 坐标
    def set_x(self, x: float) -> None: ...
    
    # 设置矩形的底边界 y 坐标
    def set_y(self, y: float) -> None: ...
    
    # 设置矩形的旋转角度
    def set_angle(self, angle: float) -> None: ...
    
    # 设置矩形的左下角坐标 (x, y)
    def set_xy(self, xy: tuple[float, float]) -> None: ...
    
    # 设置矩形的宽度
    def set_width(self, w: float) -> None: ...
    
    # 设置矩形的高度
    def set_height(self, h: float) -> None: ...
    
    @overload
    # 根据一个元组设置矩形的边界（左下角 x, y 和宽度、高度）
    def set_bounds(self, args: tuple[float, float, float, float], /) -> None: ...
    
    @overload
    # 根据独立的参数设置矩形的边界（左边界 x, 底边界 y, 宽度、高度）
    def set_bounds(
        self, left: float, bottom: float, width: float, height: float, /
    ) -> None: ...
    
    # 获取矩形的边界框对象
    def get_bbox(self) -> Bbox: ...
    
    # 创建一个属性 xy，用于获取和设置矩形的左下角坐标
    xy = property(get_xy, set_xy)
class RegularPolygon(Patch):
    # 定义一个正多边形类，继承自Patch类
    xy: tuple[float, float]  # 多边形中心点坐标
    numvertices: int  # 多边形顶点数
    orientation: float  # 多边形的方向角度
    radius: float  # 多边形的半径

    def __init__(
        self,
        xy: tuple[float, float],
        numVertices: int,
        *,
        radius: float = ...,  # 可选参数：多边形的半径
        orientation: float = ...,  # 可选参数：多边形的方向角度
        **kwargs,
    ) -> None: ...
    # 初始化方法，创建一个正多边形对象

class PathPatch(Patch):
    def __init__(self, path: Path, **kwargs) -> None: ...
    # 初始化方法，创建一个路径补丁对象
    def set_path(self, path: Path) -> None: ...
    # 设置路径的方法

class StepPatch(PathPatch):
    orientation: Literal["vertical", "horizontal"]  # 步骤图的方向

    def __init__(
        self,
        values: ArrayLike,
        edges: ArrayLike,
        *,
        orientation: Literal["vertical", "horizontal"] = ...,  # 步骤图的方向，垂直或水平
        baseline: float = ...,  # 基线位置
        **kwargs,
    ) -> None: ...
    # 初始化方法，创建一个步骤补丁对象

    # NamedTuple StairData, defined in body of method
    def get_data(self) -> tuple[np.ndarray, np.ndarray, float]: ...
    # 获取数据的方法，返回一个包含两个numpy数组和一个浮点数的元组
    def set_data(
        self,
        values: ArrayLike | None = ...,  # 设置数据的方法，可选参数：数据数组
        edges: ArrayLike | None = ...,  # 可选参数：边缘数组
        baseline: float | None = ...,  # 可选参数：基线位置
    ) -> None: ...
    # 设置数据的方法

class Polygon(Patch):
    def __init__(self, xy: ArrayLike, *, closed: bool = ..., **kwargs) -> None: ...
    # 初始化方法，创建一个多边形对象，需要指定顶点坐标数组和是否闭合
    def get_closed(self) -> bool: ...
    # 获取多边形是否闭合的方法
    def set_closed(self, closed: bool) -> None: ...
    # 设置多边形是否闭合的方法
    def get_xy(self) -> np.ndarray: ...
    # 获取顶点坐标数组的方法
    def set_xy(self, xy: ArrayLike) -> None: ...
    # 设置顶点坐标数组的方法
    xy = property(get_xy, set_xy)  # 属性：顶点坐标数组

class Wedge(Patch):
    center: tuple[float, float]  # 楔形图形的中心坐标
    r: float  # 楔形图形的半径
    theta1: float  # 楔形图形的起始角度
    theta2: float  # 楔形图形的终止角度
    width: float | None  # 楔形图形的宽度（可选）

    def __init__(
        self,
        center: tuple[float, float],
        r: float,
        theta1: float,
        theta2: float,
        *,
        width: float | None = ...,  # 可选参数：楔形图形的宽度
        **kwargs,
    ) -> None: ...
    # 初始化方法，创建一个楔形图形对象
    def set_center(self, center: tuple[float, float]) -> None: ...
    # 设置中心坐标的方法
    def set_radius(self, radius: float) -> None: ...
    # 设置半径的方法
    def set_theta1(self, theta1: float) -> None: ...
    # 设置起始角度的方法
    def set_theta2(self, theta2: float) -> None: ...
    # 设置终止角度的方法
    def set_width(self, width: float | None) -> None: ...
    # 设置宽度的方法

class Arrow(Patch):
    def __init__(
        self, x: float, y: float, dx: float, dy: float, *, width: float = ..., **kwargs
    ) -> None: ...
    # 初始化方法，创建一个箭头对象，需要指定起始点和方向
    def set_data(
        self,
        x: float | None = ...,  # 设置数据的方法，可选参数：起始点的x坐标
        y: float | None = ...,  # 可选参数：起始点的y坐标
        dx: float | None = ...,  # 可选参数：箭头的方向x分量
        dy: float | None = ...,  # 可选参数：箭头的方向y分量
        width: float | None = ...,  # 可选参数：箭头的宽度
    ) -> None: ...
    # 设置数据的方法

class FancyArrow(Polygon):
    def __init__(
        self,
        x: float,
        y: float,
        dx: float,
        dy: float,
        *,
        width: float = ...,  # 箭头的宽度
        length_includes_head: bool = ...,  # 箭头长度是否包含头部
        head_width: float | None = ...,  # 头部宽度（可选）
        head_length: float | None = ...,  # 头部长度（可选）
        shape: Literal["full", "left", "right"] = ...,  # 箭头形状
        overhang: float = ...,  # 覆盖长度
        head_starts_at_zero: bool = ...,  # 头部起点是否为零
        **kwargs,
    ) -> None: ...
    # 初始化方法，创建一个特殊箭头对象
    # 定义一个方法 `set_data`，用于设置对象的数据属性
    def set_data(
        self,
        *,
        x: float | None = ...,          # x 坐标，默认为 None
        y: float | None = ...,          # y 坐标，默认为 None
        dx: float | None = ...,         # x 方向速度，默认为 None
        dy: float | None = ...,         # y 方向速度，默认为 None
        width: float | None = ...,      # 线段宽度，默认为 None
        head_width: float | None = ..., # 箭头宽度，默认为 None
        head_length: float | None = ...,# 箭头长度，默认为 None
    ) -> None:                         # 返回类型为 None
        # 该方法用于设置多个数据属性，参数列表中的 `*` 表示后续参数只能以关键字方式传递
        # 每个参数的类型为 float 或 None，表示其可以接受浮点数或者 None 作为值
        ...
class CirclePolygon(RegularPolygon):
    # CirclePolygon 类继承自 RegularPolygon 类

    def __init__(
        self,
        xy: tuple[float, float],
        radius: float = ...,
        *,
        resolution: int = ...,
        **kwargs,
    ) -> None:
        # CirclePolygon 类的初始化方法
        ...


class Ellipse(Patch):
    # Ellipse 类继承自 Patch 类

    def __init__(
        self,
        xy: tuple[float, float],
        width: float,
        height: float,
        *,
        angle: float = ...,
        **kwargs,
    ) -> None:
        # Ellipse 类的初始化方法，接受椭圆的中心坐标、宽度、高度和可选的角度及其他参数
        ...

    def set_center(self, xy: tuple[float, float]) -> None:
        # 设置椭圆的中心坐标方法
        ...

    def get_center(self) -> float:
        # 获取椭圆的中心坐标方法
        ...

    center = property(get_center, set_center)
    # 椭圆中心坐标的属性

    def set_width(self, width: float) -> None:
        # 设置椭圆宽度的方法
        ...

    def get_width(self) -> float:
        # 获取椭圆宽度的方法
        ...

    width = property(get_width, set_width)
    # 椭圆宽度的属性

    def set_height(self, height: float) -> None:
        # 设置椭圆高度的方法
        ...

    def get_height(self) -> float:
        # 获取椭圆高度的方法
        ...

    height = property(get_height, set_height)
    # 椭圆高度的属性

    def set_angle(self, angle: float) -> None:
        # 设置椭圆旋转角度的方法
        ...

    def get_angle(self) -> float:
        # 获取椭圆旋转角度的方法
        ...

    angle = property(get_angle, set_angle)
    # 椭圆旋转角度的属性

    def get_corners(self) -> np.ndarray:
        # 获取椭圆的四个角的方法
        ...

    def get_vertices(self) -> list[tuple[float, float]]:
        # 获取椭圆的顶点列表的方法
        ...

    def get_co_vertices(self) -> list[tuple[float, float]]:
        # 获取椭圆的对称顶点列表的方法
        ...


class Annulus(Patch):
    # Annulus 类继承自 Patch 类

    a: float
    b: float

    def __init__(
        self,
        xy: tuple[float, float],
        r: float | tuple[float, float],
        width: float,
        angle: float = ...,
        **kwargs,
    ) -> None:
        # Annulus 类的初始化方法，接受中心坐标、半径或半径元组、宽度和其他参数
        ...

    def set_center(self, xy: tuple[float, float]) -> None:
        # 设置环形图形中心坐标的方法
        ...

    def get_center(self) -> tuple[float, float]:
        # 获取环形图形中心坐标的方法
        ...

    center = property(get_center, set_center)
    # 环形图形中心坐标的属性

    def set_width(self, width: float) -> None:
        # 设置环形图形宽度的方法
        ...

    def get_width(self) -> float:
        # 获取环形图形宽度的方法
        ...

    width = property(get_width, set_width)
    # 环形图形宽度的属性

    def set_angle(self, angle: float) -> None:
        # 设置环形图形旋转角度的方法
        ...

    def get_angle(self) -> float:
        # 获取环形图形旋转角度的方法
        ...

    angle = property(get_angle, set_angle)
    # 环形图形旋转角度的属性

    def set_semimajor(self, a: float) -> None:
        # 设置环形图形的半长轴长度的方法
        ...

    def set_semiminor(self, b: float) -> None:
        # 设置环形图形的半短轴长度的方法
        ...

    def set_radii(self, r: float | tuple[float, float]) -> None:
        # 设置环形图形的半径或半径元组的方法
        ...

    def get_radii(self) -> tuple[float, float]:
        # 获取环形图形的半径或半径元组的方法
        ...

    radii = property(get_radii, set_radii)
    # 环形图形的半径或半径元组的属性


class Circle(Ellipse):
    # Circle 类继承自 Ellipse 类

    def __init__(
        self, xy: tuple[float, float], radius: float = ..., **kwargs
    ) -> None:
        # Circle 类的初始化方法，接受圆的中心坐标、半径和其他参数
        ...

    def set_radius(self, radius: float) -> None:
        # 设置圆的半径的方法
        ...

    def get_radius(self) -> float:
        # 获取圆的半径的方法
        ...

    radius = property(get_radius, set_radius)
    # 圆的半径的属性


class Arc(Ellipse):
    # Arc 类继承自 Ellipse 类

    theta1: float
    theta2: float

    def __init__(
        self,
        xy: tuple[float, float],
        width: float,
        height: float,
        *,
        angle: float = ...,
        theta1: float = ...,
        theta2: float = ...,
        **kwargs,
    ) -> None:
        # Arc 类的初始化方法，接受圆弧的中心坐标、宽度、高度、可选的角度、起始角度和结束角度以及其他参数
        ...


def bbox_artist(
    artist: artist.Artist,
    renderer: RendererBase,
    props: dict[str, Any] | None = ...,
    fill: bool = ...,
) -> None:
    # 根据 artist 和 props 绘制边界框的方法
    ...


def draw_bbox(
    bbox: Bbox,
    renderer: RendererBase,
    color: ColorType = ...,
    trans: Transform | None = ...,
) -> None:
    # 绘制边界框的方法，接受边界框对象、渲染器、颜色和变换参数
    ...


class _Style:
    # _Style 类的定义

    def __new__(cls, stylename, **kwargs):
        # _Style 类的构造方法，接受样式名称和其他参数
        ...
    # 定义一个类方法，用于获取已注册的所有样式，返回一个字典，键是样式的名称，值是样式的类型
    @classmethod
    def get_styles(cls) -> dict[str, type]: ...
    
    # 定义一个类方法，用于以字符串形式打印已注册的所有样式信息，返回样式信息的字符串表示
    @classmethod
    def pprint_styles(cls) -> str: ...
    
    # 定义一个类方法，用于注册新的样式，接受样式的名称和类型作为参数，无返回值
    @classmethod
    def register(cls, name: str, style: type) -> None: ...
class BoxStyle(_Style):
    # BoxStyle 类，继承自 _Style 类，用于定义不同形状的图形样式

    class Square(BoxStyle):
        # Square 类，继承自 BoxStyle 类，定义一个正方形的图形样式
        pad: float
        def __init__(self, pad: float = ...) -> None:
            # 初始化方法，接受一个 pad 参数用于指定填充量
            ...

        def __call__(
            self,
            x0: float,
            y0: float,
            width: float,
            height: float,
            mutation_size: float,
        ) -> Path:
            # 实现调用方法，根据给定的位置和尺寸参数生成一个 Path 对象表示正方形的路径
            ...

    class Circle(BoxStyle):
        # Circle 类，继承自 BoxStyle 类，定义一个圆形的图形样式
        pad: float
        def __init__(self, pad: float = ...) -> None:
            # 初始化方法，接受一个 pad 参数用于指定填充量
            ...

        def __call__(
            self,
            x0: float,
            y0: float,
            width: float,
            height: float,
            mutation_size: float,
        ) -> Path:
            # 实现调用方法，根据给定的位置和尺寸参数生成一个 Path 对象表示圆形的路径
            ...

    class Ellipse(BoxStyle):
        # Ellipse 类，继承自 BoxStyle 类，定义一个椭圆形的图形样式
        pad: float
        def __init__(self, pad: float = ...) -> None:
            # 初始化方法，接受一个 pad 参数用于指定填充量
            ...

        def __call__(
            self,
            x0: float,
            y0: float,
            width: float,
            height: float,
            mutation_size: float,
        ) -> Path:
            # 实现调用方法，根据给定的位置和尺寸参数生成一个 Path 对象表示椭圆形的路径
            ...

    class LArrow(BoxStyle):
        # LArrow 类，继承自 BoxStyle 类，定义一个左箭头形状的图形样式
        pad: float
        def __init__(self, pad: float = ...) -> None:
            # 初始化方法，接受一个 pad 参数用于指定填充量
            ...

        def __call__(
            self,
            x0: float,
            y0: float,
            width: float,
            height: float,
            mutation_size: float,
        ) -> Path:
            # 实现调用方法，根据给定的位置和尺寸参数生成一个 Path 对象表示左箭头形状的路径
            ...

    class RArrow(LArrow):
        # RArrow 类，继承自 LArrow 类，定义一个右箭头形状的图形样式
        def __call__(
            self,
            x0: float,
            y0: float,
            width: float,
            height: float,
            mutation_size: float,
        ) -> Path:
            # 实现调用方法，根据给定的位置和尺寸参数生成一个 Path 对象表示右箭头形状的路径
            ...

    class DArrow(BoxStyle):
        # DArrow 类，继承自 BoxStyle 类，定义一个下箭头形状的图形样式
        pad: float
        def __init__(self, pad: float = ...) -> None:
            # 初始化方法，接受一个 pad 参数用于指定填充量
            ...

        def __call__(
            self,
            x0: float,
            y0: float,
            width: float,
            height: float,
            mutation_size: float,
        ) -> Path:
            # 实现调用方法，根据给定的位置和尺寸参数生成一个 Path 对象表示下箭头形状的路径
            ...

    class Round(BoxStyle):
        # Round 类，继承自 BoxStyle 类，定义一个圆角矩形形状的图形样式
        pad: float
        rounding_size: float | None
        def __init__(
            self, pad: float = ..., rounding_size: float | None = ...
        ) -> None:
            # 初始化方法，接受一个 pad 参数用于指定填充量，rounding_size 参数用于指定圆角大小
            ...

        def __call__(
            self,
            x0: float,
            y0: float,
            width: float,
            height: float,
            mutation_size: float,
        ) -> Path:
            # 实现调用方法，根据给定的位置、尺寸和圆角参数生成一个 Path 对象表示圆角矩形的路径
            ...

    class Round4(BoxStyle):
        # Round4 类，继承自 BoxStyle 类，定义一个四个圆角的矩形形状的图形样式
        pad: float
        rounding_size: float | None
        def __init__(
            self, pad: float = ..., rounding_size: float | None = ...
        ) -> None:
            # 初始化方法，接受一个 pad 参数用于指定填充量，rounding_size 参数用于指定圆角大小
            ...

        def __call__(
            self,
            x0: float,
            y0: float,
            width: float,
            height: float,
            mutation_size: float,
        ) -> Path:
            # 实现调用方法，根据给定的位置、尺寸和圆角参数生成一个 Path 对象表示四个圆角矩形的路径
            ...

    class Sawtooth(BoxStyle):
        # Sawtooth 类，继承自 BoxStyle 类，定义一个锯齿形状的图形样式
        pad: float
        tooth_size: float | None
        def __init__(
            self, pad: float = ..., tooth_size: float | None = ...
        ) -> None:
            # 初始化方法，接受一个 pad 参数用于指定填充量，tooth_size 参数用于指定锯齿的大小
            ...

        def __call__(
            self,
            x0: float,
            y0: float,
            width: float,
            height: float,
            mutation_size: float,
        ) -> Path:
            # 实现调用方法，根据给定的位置、尺寸和锯齿大小参数生成一个 Path 对象表示锯齿形状的路径
            ...
    # 定义一个名为 Roundtooth 的类，继承自 Sawtooth 类
    class Roundtooth(Sawtooth):
        # 定义 __call__ 方法，该方法可以被实例直接调用
        # 参数说明：
        # - x0: 浮点数，表示起始 x 坐标
        # - y0: 浮点数，表示起始 y 坐标
        # - width: 浮点数，表示宽度
        # - height: 浮点数，表示高度
        # - mutation_size: 浮点数，表示变异大小
        # 返回值：一个 Path 对象，表示路径
        def __call__(
            self,
            x0: float,
            y0: float,
            width: float,
            height: float,
            mutation_size: float,
        ) -> Path:
            ...
class ConnectionStyle(_Style):
    # 定义连接风格的基类，继承自 _Style

    class _Base(ConnectionStyle):
        # 连接风格的基类，继承自 ConnectionStyle

        def __call__(
            self,
            posA: tuple[float, float],
            posB: tuple[float, float],
            shrinkA: float = ...,
            shrinkB: float = ...,
            patchA: Patch | None = ...,
            patchB: Patch | None = ...,
        ) -> Path:
            # 定义调用操作，接受两个位置参数，可选的收缩参数和修补区域参数，
            # 返回一个 Path 对象

    class Arc3(_Base):
        # 继承自 _Base 的 Arc3 连接风格类

        rad: float
        def __init__(self, rad: float = ...) -> None:
            # 初始化方法，接受一个半径参数

        def connect(
            self, posA: tuple[float, float], posB: tuple[float, float]
        ) -> Path:
            # 连接方法，接受两个位置参数，返回一个 Path 对象

    class Angle3(_Base):
        # 继承自 _Base 的 Angle3 连接风格类

        angleA: float
        angleB: float
        def __init__(self, angleA: float = ..., angleB: float = ...) -> None:
            # 初始化方法，接受两个角度参数

        def connect(
            self, posA: tuple[float, float], posB: tuple[float, float]
        ) -> Path:
            # 连接方法，接受两个位置参数，返回一个 Path 对象

    class Angle(_Base):
        # 继承自 _Base 的 Angle 连接风格类

        angleA: float
        angleB: float
        rad: float
        def __init__(
            self, angleA: float = ..., angleB: float = ..., rad: float = ...
        ) -> None:
            # 初始化方法，接受两个角度参数和一个半径参数

        def connect(
            self, posA: tuple[float, float], posB: tuple[float, float]
        ) -> Path:
            # 连接方法，接受两个位置参数，返回一个 Path 对象

    class Arc(_Base):
        # 继承自 _Base 的 Arc 连接风格类

        angleA: float
        angleB: float
        armA: float | None
        armB: float | None
        rad: float
        def __init__(
            self,
            angleA: float = ...,
            angleB: float = ...,
            armA: float | None = ...,
            armB: float | None = ...,
            rad: float = ...,
        ) -> None:
            # 初始化方法，接受两个角度参数和两个臂长参数以及一个半径参数

        def connect(
            self, posA: tuple[float, float], posB: tuple[float, float]
        ) -> Path:
            # 连接方法，接受两个位置参数，返回一个 Path 对象

    class Bar(_Base):
        # 继承自 _Base 的 Bar 连接风格类

        armA: float
        armB: float
        fraction: float
        angle: float | None
        def __init__(
            self,
            armA: float = ...,
            armB: float = ...,
            fraction: float = ...,
            angle: float | None = ...,
        ) -> None:
            # 初始化方法，接受两个臂长参数、一个分数参数和一个角度参数（可选）

        def connect(
            self, posA: tuple[float, float], posB: tuple[float, float]
        ) -> Path:
            # 连接方法，接受两个位置参数，返回一个 Path 对象

class ArrowStyle(_Style):
    # 箭头风格类，继承自 _Style

    class _Base(ArrowStyle):
        # 箭头风格基类，继承自 ArrowStyle

        @staticmethod
        def ensure_quadratic_bezier(path: Path) -> list[float]:
            # 静态方法，接受一个 Path 对象，返回一个浮点数列表

        def transmute(
            self, path: Path, mutation_size: float, linewidth: float
        ) -> tuple[Path, bool]:
            # 转变方法，接受一个 Path 对象、变异大小和线宽参数，返回一个 Path 对象和布尔值

        def __call__(
            self,
            path: Path,
            mutation_size: float,
            linewidth: float,
            aspect_ratio: float = ...,
        ) -> tuple[Path, bool]:
            # 定义调用操作，接受一个 Path 对象、变异大小、线宽和纵横比参数（可选），
            # 返回一个 Path 对象和布尔值
    class _Curve(_Base):
        arrow: str  # 定义属性 arrow 为字符串类型，用于表示箭头
        fillbegin: bool  # 定义属性 fillbegin 为布尔类型，表示是否填充起始端
        fillend: bool  # 定义属性 fillend 为布尔类型，表示是否填充结束端
    
        def __init__(
            self,
            head_length: float = ...,  # 头部长度，默认值未指定
            head_width: float = ...,   # 头部宽度，默认值未指定
            widthA: float = ...,       # 宽度A，默认值未指定
            widthB: float = ...,       # 宽度B，默认值未指定
            lengthA: float = ...,      # 长度A，默认值未指定
            lengthB: float = ...,      # 长度B，默认值未指定
            angleA: float | None = ...,  # 角度A，可以是浮点数或者None
            angleB: float | None = ...,  # 角度B，可以是浮点数或者None
            scaleA: float | None = ...,  # 比例A，可以是浮点数或者None
            scaleB: float | None = ...,  # 比例B，可以是浮点数或者None
        ) -> None:  # 构造函数，不返回任何值，初始化对象属性为指定或默认值
            ...
    
    class Curve(_Curve):
        def __init__(self) -> None:  # 继承自 _Curve，初始化函数为空，无需额外操作
            ...
    
    class CurveA(_Curve):
        arrow: str  # 定义属性 arrow 为字符串类型，用于表示箭头
    
    class CurveB(_Curve):
        arrow: str  # 定义属性 arrow 为字符串类型，用于表示箭头
    
    class CurveAB(_Curve):
        arrow: str  # 定义属性 arrow 为字符串类型，用于表示箭头
    
    class CurveFilledA(_Curve):
        arrow: str  # 定义属性 arrow 为字符串类型，用于表示箭头
    
    class CurveFilledB(_Curve):
        arrow: str  # 定义属性 arrow 为字符串类型，用于表示箭头
    
    class CurveFilledAB(_Curve):
        arrow: str  # 定义属性 arrow 为字符串类型，用于表示箭头
    
    class BracketA(_Curve):
        arrow: str  # 定义属性 arrow 为字符串类型，用于表示箭头
    
        def __init__(
            self,
            widthA: float = ...,    # 宽度A，默认值未指定
            lengthA: float = ...,   # 长度A，默认值未指定
            angleA: float = ...     # 角度A，默认值未指定
        ) -> None:  # 构造函数，不返回任何值，初始化对象属性为指定或默认值
            ...
    
    class BracketB(_Curve):
        arrow: str  # 定义属性 arrow 为字符串类型，用于表示箭头
    
        def __init__(
            self,
            widthB: float = ...,    # 宽度B，默认值未指定
            lengthB: float = ...,   # 长度B，默认值未指定
            angleB: float = ...     # 角度B，默认值未指定
        ) -> None:  # 构造函数，不返回任何值，初始化对象属性为指定或默认值
            ...
    
    class BracketAB(_Curve):
        arrow: str  # 定义属性 arrow 为字符串类型，用于表示箭头
    
        def __init__(
            self,
            widthA: float = ...,    # 宽度A，默认值未指定
            lengthA: float = ...,   # 长度A，默认值未指定
            angleA: float = ...,    # 角度A，默认值未指定
            widthB: float = ...,    # 宽度B，默认值未指定
            lengthB: float = ...,   # 长度B，默认值未指定
            angleB: float = ...     # 角度B，默认值未指定
        ) -> None:  # 构造函数，不返回任何值，初始化对象属性为指定或默认值
            ...
    
    class BarAB(_Curve):
        arrow: str  # 定义属性 arrow 为字符串类型，用于表示箭头
    
        def __init__(
            self,
            widthA: float = ...,    # 宽度A，默认值未指定
            angleA: float = ...,    # 角度A，默认值未指定
            widthB: float = ...,    # 宽度B，默认值未指定
            angleB: float = ...     # 角度B，默认值未指定
        ) -> None:  # 构造函数，不返回任何值，初始化对象属性为指定或默认值
            ...
    
    class BracketCurve(_Curve):
        arrow: str  # 定义属性 arrow 为字符串类型，用于表示箭头
    
        def __init__(
            self,
            widthA: float = ...,    # 宽度A，默认值未指定
            lengthA: float = ...,   # 长度A，默认值未指定
            angleA: float | None = ...  # 角度A，可以是浮点数或者None
        ) -> None:  # 构造函数，不返回任何值，初始化对象属性为指定或默认值
            ...
    
    class CurveBracket(_Curve):
        arrow: str  # 定义属性 arrow 为字符串类型，用于表示箭头
    
        def __init__(
            self,
            widthB: float = ...,    # 宽度B，默认值未指定
            lengthB: float = ...,   # 长度B，默认值未指定
            angleB: float | None = ...  # 角度B，可以是浮点数或者None
        ) -> None:  # 构造函数，不返回任何值，初始化对象属性为指定或默认值
            ...
    
    class Simple(_Base):
        def __init__(
            self,
            head_length: float = ...,    # 头部长度，默认值未指定
            head_width: float = ...,     # 头部宽度，默认值未指定
            tail_width: float = ...      # 尾部宽度，默认值未指定
        ) -> None:  # 构造函数，不返回任何值，初始化对象属性为指定或默认值
            ...
    
    class Fancy(_Base):
        def __init__(
            self,
            head_length: float = ...,    # 头部长度，默认值未指定
            head_width: float = ...,     # 头部宽度，默认值未指定
            tail_width: float = ...      # 尾部宽度，默认值未指定
        ) -> None:  # 构造函数，不返回任何值，初始化对象属性为指定或默认值
            ...
    
    class Wedge(_Base):
        tail_width: float  # 定义属性 tail_width 为浮点数，表示尾部宽度
        shrink_factor: float  # 定义属性 shrink_factor 为浮点数，表示收缩因子
    
        def __init__(
            self,
            tail_width: float = ...,     # 尾部宽度，默认值未指定
            shrink_factor: float = ...   # 收缩因子，默认值未指定
        ) -> None:  # 构造函数，不返回任何值，初始化对象属性为指定或默认值
            ...
# 定义一个继承自 Patch 的 FancyBboxPatch 类
class FancyBboxPatch(Patch):
    # 初始化方法，接受一系列参数来定义 FancyBboxPatch 对象
    def __init__(
        self,
        xy: tuple[float, float],  # 左下角顶点的坐标
        width: float,             # 宽度
        height: float,            # 高度
        boxstyle: str | BoxStyle = ...,  # 箱子样式，默认为 ...
        *,
        mutation_scale: float = ...,    # 变异比例尺
        mutation_aspect: float = ...,   # 变异方面比
        **kwargs,               # 其他关键字参数
    ) -> None: ...             # 初始化方法没有具体实现

    # 设置箱子样式的方法
    def set_boxstyle(self, boxstyle: str | BoxStyle | None = ..., **kwargs) -> None: ...

    # 获取箱子样式的方法
    def get_boxstyle(self) -> BoxStyle: ...

    # 设置变异比例尺的方法
    def set_mutation_scale(self, scale: float) -> None: ...

    # 获取变异比例尺的方法
    def get_mutation_scale(self) -> float: ...

    # 设置变异方面比的方法
    def set_mutation_aspect(self, aspect: float) -> None: ...

    # 获取变异方面比的方法
    def get_mutation_aspect(self) -> float: ...

    # 获取左下角顶点 x 坐标的方法
    def get_x(self) -> float: ...

    # 获取左下角顶点 y 坐标的方法
    def get_y(self) -> float: ...

    # 获取宽度的方法
    def get_width(self) -> float: ...

    # 获取高度的方法
    def get_height(self) -> float: ...

    # 设置左下角顶点 x 坐标的方法
    def set_x(self, x: float) -> None: ...

    # 设置左下角顶点 y 坐标的方法
    def set_y(self, y: float) -> None: ...

    # 设置宽度的方法
    def set_width(self, w: float) -> None: ...

    # 设置高度的方法
    def set_height(self, h: float) -> None: ...

    # 设置边界的方法，参数为 (left, bottom, width, height)
    @overload
    def set_bounds(self, args: tuple[float, float, float, float], /) -> None: ...

    # 设置边界的方法，参数为 (left, bottom, width, height)
    @overload
    def set_bounds(
        self, left: float, bottom: float, width: float, height: float, /
    ) -> None: ...

    # 获取边界框的方法
    def get_bbox(self) -> Bbox: ...


# 定义一个继承自 Patch 的 FancyArrowPatch 类
class FancyArrowPatch(Patch):
    patchA: Patch        # 与箭头连接的 Patch A 对象
    patchB: Patch        # 与箭头连接的 Patch B 对象
    shrinkA: float       # Patch A 的缩小系数
    shrinkB: float       # Patch B 的缩小系数

    # 初始化方法，接受一系列参数来定义 FancyArrowPatch 对象
    def __init__(
        self,
        posA: tuple[float, float] | None = ...,   # 箭头起点位置，可以为 None
        posB: tuple[float, float] | None = ...,   # 箭头终点位置，可以为 None
        *,
        path: Path | None = ...,                  # 路径对象，可以为 None
        arrowstyle: str | ArrowStyle = ...,       # 箭头样式，默认为 ...
        connectionstyle: str | ConnectionStyle = ...,  # 连接样式，默认为 ...
        patchA: Patch | None = ...,               # 与箭头连接的 Patch A 对象，可以为 None
        patchB: Patch | None = ...,               # 与箭头连接的 Patch B 对象，可以为 None
        shrinkA: float = ...,                     # Patch A 的缩小系数
        shrinkB: float = ...,                     # Patch B 的缩小系数
        mutation_scale: float = ...,              # 变异比例尺
        mutation_aspect: float | None = ...,      # 变异方面比，可以为 None
        **kwargs,                                 # 其他关键字参数
    ) -> None: ...

    # 设置箭头起点和终点位置的方法
    def set_positions(
        self, posA: tuple[float, float], posB: tuple[float, float]
    ) -> None: ...

    # 设置与箭头连接的 Patch A 对象的方法
    def set_patchA(self, patchA: Patch) -> None: ...

    # 设置与箭头连接的 Patch B 对象的方法
    def set_patchB(self, patchB: Patch) -> None: ...

    # 设置连接样式的方法
    def set_connectionstyle(self, connectionstyle: str | ConnectionStyle | None = ..., **kwargs) -> None: ...

    # 获取连接样式的方法
    def get_connectionstyle(self) -> ConnectionStyle: ...

    # 设置箭头样式的方法
    def set_arrowstyle(self, arrowstyle: str | ArrowStyle | None = ..., **kwargs) -> None: ...

    # 获取箭头样式的方法
    def get_arrowstyle(self) -> ArrowStyle: ...

    # 设置变异比例尺的方法
    def set_mutation_scale(self, scale: float) -> None: ...

    # 获取变异比例尺的方法
    def get_mutation_scale(self) -> float: ...

    # 设置变异方面比的方法
    def set_mutation_aspect(self, aspect: float | None) -> None: ...

    # 获取变异方面比的方法
    def get_mutation_aspect(self) -> float: ...


# 定义一个继承自 FancyArrowPatch 的 ConnectionPatch 类
class ConnectionPatch(FancyArrowPatch):
    xy1: tuple[float, float]         # 连接点 1 的坐标
    xy2: tuple[float, float]         # 连接点 2 的坐标
    coords1: str | Transform         # 点 1 的坐标系或变换
    coords2: str | Transform | None  # 点 2 的坐标系或变换，可以为 None
    axesA: Axes | None               # Patch A 所属的 Axes 对象，可以为 None
    axesB: Axes | None               # Patch B 所属的 Axes 对象，可以为 None
    # 初始化方法，用于创建一个连接两个坐标点的注释。
    def __init__(
        self,
        xyA: tuple[float, float],              # 参数：起始点的坐标 (x, y)
        xyB: tuple[float, float],              # 参数：结束点的坐标 (x, y)
        coordsA: str | Transform,              # 参数：起始点的坐标系描述或变换
        coordsB: str | Transform | None = ..., # 参数：结束点的坐标系描述或变换（可选）
        *,
        axesA: Axes | None = ...,              # 参数：起始点所在的图轴对象（可选）
        axesB: Axes | None = ...,              # 参数：结束点所在的图轴对象（可选）
        arrowstyle: str | ArrowStyle = ...,    # 参数：箭头的风格（可选）
        connectionstyle: str | ConnectionStyle = ...,  # 参数：连接线的风格（可选）
        patchA: Patch | None = ...,            # 参数：起始点的图形补丁对象（可选）
        patchB: Patch | None = ...,            # 参数：结束点的图形补丁对象（可选）
        shrinkA: float = ...,                  # 参数：起始点的缩放比例（可选）
        shrinkB: float = ...,                  # 参数：结束点的缩放比例（可选）
        mutation_scale: float = ...,           # 参数：变异尺度（可选）
        mutation_aspect: float | None = ...,   # 参数：变异纵横比（可选）
        clip_on: bool = ...,                   # 参数：是否裁剪到图轴范围内（可选）
        **kwargs,                              # 其他关键字参数
    ) -> None:                                 # 返回值为None
        ...
    
    # 设置注释对象是否裁剪到图轴边界内部
    def set_annotation_clip(self, b: bool | None) -> None:
        ...
    
    # 获取注释对象是否被裁剪到图轴边界内部的状态
    def get_annotation_clip(self) -> bool | None:
        ...
```