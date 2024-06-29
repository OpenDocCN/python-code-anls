# `D:\src\scipysrc\matplotlib\lib\matplotlib\collections.pyi`

```
# 导入需要的模块和类
from collections.abc import Callable, Iterable, Sequence
from typing import Literal

import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.typing import ArrayLike, NDArray  # 导入 NumPy 的类型定义

from . import artist, cm, transforms  # 导入当前目录下的 artist, cm, transforms 模块
from .backend_bases import MouseEvent  # 从 backend_bases 模块中导入 MouseEvent 类
from .artist import Artist  # 从 artist 模块中导入 Artist 类
from .colors import Normalize, Colormap  # 从 colors 模块中导入 Normalize, Colormap 类
from .lines import Line2D  # 从 lines 模块中导入 Line2D 类
from .path import Path  # 从 path 模块中导入 Path 类
from .patches import Patch  # 从 patches 模块中导入 Patch 类
from .ticker import Locator, Formatter  # 从 ticker 模块中导入 Locator, Formatter 类
from .tri import Triangulation  # 从 tri 模块中导入 Triangulation 类
from .typing import ColorType, LineStyleType, CapStyleType, JoinStyleType  # 导入 typing 模块中的类型定义

class Collection(artist.Artist, cm.ScalarMappable):
    # Collection 类继承自 artist.Artist 和 cm.ScalarMappable 类

    def __init__(
        self,
        *,
        edgecolors: ColorType | Sequence[ColorType] | None = ...,
        facecolors: ColorType | Sequence[ColorType] | None = ...,
        linewidths: float | Sequence[float] | None = ...,
        linestyles: LineStyleType | Sequence[LineStyleType] = ...,
        capstyle: CapStyleType | None = ...,
        joinstyle: JoinStyleType | None = ...,
        antialiaseds: bool | Sequence[bool] | None = ...,
        offsets: tuple[float, float] | Sequence[tuple[float, float]] | None = ...,
        offset_transform: transforms.Transform | None = ...,
        norm: Normalize | None = ...,
        cmap: Colormap | None = ...,
        pickradius: float = ...,
        hatch: str | None = ...,
        urls: Sequence[str] | None = ...,
        zorder: float = ...,
        **kwargs
    ) -> None:
        # Collection 对象的初始化方法，接收多个参数，并赋予默认值或类型注释
        ...

    def get_paths(self) -> Sequence[Path]:
        # 返回 Collection 中所有图形的路径（Path 对象）的序列
        ...

    def set_paths(self, paths: Sequence[Path]) -> None:
        # 设置 Collection 中所有图形的路径（Path 对象）
        ...

    def get_transforms(self) -> Sequence[transforms.Transform]:
        # 返回用于转换 Collection 的所有图形的转换对象（Transform 对象）的序列
        ...

    def get_offset_transform(self) -> transforms.Transform:
        # 返回 Collection 中所有图形的偏移转换对象（Transform 对象）
        ...

    def set_offset_transform(self, offset_transform: transforms.Transform) -> None:
        # 设置 Collection 中所有图形的偏移转换对象（Transform 对象）
        ...

    def get_datalim(self, transData: transforms.Transform) -> transforms.Bbox:
        # 返回 Collection 在数据坐标系中的数据边界框（Bbox 对象）
        ...

    def set_pickradius(self, pickradius: float) -> None:
        # 设置 Collection 中所有图形的选取半径
        ...

    def get_pickradius(self) -> float:
        # 返回 Collection 中所有图形的选取半径
        ...

    def set_urls(self, urls: Sequence[str | None]) -> None:
        # 设置 Collection 中所有图形的超链接 URL
        ...

    def get_urls(self) -> Sequence[str | None]:
        # 返回 Collection 中所有图形的超链接 URL 的序列
        ...

    def set_hatch(self, hatch: str) -> None:
        # 设置 Collection 中所有图形的填充图案
        ...

    def get_hatch(self) -> str:
        # 返回 Collection 中所有图形的填充图案
        ...

    def set_offsets(self, offsets: ArrayLike) -> None:
        # 设置 Collection 中所有图形的偏移量
        ...

    def get_offsets(self) -> ArrayLike:
        # 返回 Collection 中所有图形的偏移量
        ...

    def set_linewidth(self, lw: float | Sequence[float]) -> None:
        # 设置 Collection 中所有图形的线宽
        ...

    def set_linestyle(self, ls: LineStyleType | Sequence[LineStyleType]) -> None:
        # 设置 Collection 中所有图形的线型
        ...

    def set_capstyle(self, cs: CapStyleType) -> None:
        # 设置 Collection 中所有图形的端点风格
        ...

    def get_capstyle(self) -> Literal["butt", "projecting", "round"] | None:
        # 返回 Collection 中所有图形的端点风格
        ...

    def set_joinstyle(self, js: JoinStyleType) -> None:
        # 设置 Collection 中所有图形的连接风格
        ...

    def get_joinstyle(self) -> Literal["miter", "round", "bevel"] | None:
        # 返回 Collection 中所有图形的连接风格
        ...

    def set_antialiased(self, aa: bool | Sequence[bool]) -> None:
        # 设置 Collection 中所有图形的抗锯齿属性
        ...

    def get_antialiased(self) -> NDArray[np.bool_]:
        # 返回 Collection 中所有图形的抗锯齿属性
        ...

    def set_color(self, c: ColorType | Sequence[ColorType]) -> None:
        # 设置 Collection 中所有图形的颜色
        ...

    def set_facecolor(self, c: ColorType | Sequence[ColorType]) -> None:
        # 设置 Collection 中所有图形的填充颜色
        ...
    # 获取图形对象的面颜色，可以返回单个颜色或颜色序列
    def get_facecolor(self) -> ColorType | Sequence[ColorType]: ...
    
    # 获取图形对象的边缘颜色，可以返回单个颜色或颜色序列
    def get_edgecolor(self) -> ColorType | Sequence[ColorType]: ...
    
    # 设置图形对象的边缘颜色，接受单个颜色或颜色序列作为参数，无返回值
    def set_edgecolor(self, c: ColorType | Sequence[ColorType]) -> None: ...
    
    # 设置图形对象的透明度，接受单个浮点数、浮点数序列或 None 作为参数，无返回值
    def set_alpha(self, alpha: float | Sequence[float] | None) -> None: ...
    
    # 获取图形对象的线宽，可以返回单个浮点数或浮点数序列
    def get_linewidth(self) -> float | Sequence[float]: ...
    
    # 获取图形对象的线型，可以返回单个线型或线型序列
    def get_linestyle(self) -> LineStyleType | Sequence[LineStyleType]: ...
    
    # 更新标量映射对象，无返回值
    def update_scalarmappable(self) -> None: ...
    
    # 获取图形对象是否填充的布尔值
    def get_fill(self) -> bool: ...
    
    # 从另一个艺术家对象更新当前对象的属性，无返回值
    def update_from(self, other: Artist) -> None: ...
class _CollectionWithSizes(Collection):
    # 继承自 Collection 类，表示带有尺寸信息的集合类
    def get_sizes(self) -> np.ndarray: ...
    # 获取尺寸信息，返回一个 NumPy 数组

    def set_sizes(self, sizes: ArrayLike | None, dpi: float = ...) -> None: ...
    # 设置尺寸信息，sizes 可以是类数组对象或者 None，dpi 是像素密度，默认值为 ...


class PathCollection(_CollectionWithSizes):
    # 表示包含路径集合的集合类，继承自 _CollectionWithSizes
    def __init__(
        self, paths: Sequence[Path], sizes: ArrayLike | None = ..., **kwargs
    ) -> None: ...
    # 初始化方法，接受路径的序列 paths 和可选的尺寸信息 sizes，以及其他关键字参数

    def set_paths(self, paths: Sequence[Path]) -> None: ...
    # 设置路径集合，传入一个路径的序列 paths

    def get_paths(self) -> Sequence[Path]: ...
    # 获取路径集合，返回一个路径的序列

    def legend_elements(
        self,
        prop: Literal["colors", "sizes"] = ...,
        num: int | Literal["auto"] | ArrayLike | Locator = ...,
        fmt: str | Formatter | None = ...,
        func: Callable[[ArrayLike], ArrayLike] = ...,
        **kwargs,
    ) -> tuple[list[Line2D], list[str]]: ...
    # 生成图例元素，prop 指定生成图例的属性（颜色或尺寸），num 指定数量或自动确定，fmt 是格式化字符串或格式化器，func 是对属性数据的处理函数，其他关键字参数会传递给生成器
    


class PolyCollection(_CollectionWithSizes):
    # 表示包含多边形集合的集合类，继承自 _CollectionWithSizes
    def __init__(
        self,
        verts: Sequence[ArrayLike],
        sizes: ArrayLike | None = ...,
        *,
        closed: bool = ...,
        **kwargs
    ) -> None: ...
    # 初始化方法，接受顶点序列 verts，可选的尺寸信息 sizes，是否闭合的标志 closed，以及其他关键字参数

    def set_verts(
        self, verts: Sequence[ArrayLike | Path], closed: bool = ...
    ) -> None: ...
    # 设置顶点集合，verts 是顶点或路径的序列，closed 指定多边形是否闭合

    def set_paths(self, verts: Sequence[Path], closed: bool = ...) -> None: ...
    # 设置路径集合，verts 是路径的序列，closed 指定多边形是否闭合

    def set_verts_and_codes(
        self, verts: Sequence[ArrayLike | Path], codes: Sequence[int]
    ) -> None: ...
    # 设置顶点集合和路径代码，verts 是顶点或路径的序列，codes 是路径的代码（例如移动、线段等）的序列



class RegularPolyCollection(_CollectionWithSizes):
    # 表示包含正多边形集合的集合类，继承自 _CollectionWithSizes
    def __init__(
        self, numsides: int, *, rotation: float = ..., sizes: ArrayLike = ..., **kwargs
    ) -> None: ...
    # 初始化方法，接受边数 numsides，旋转角度 rotation，尺寸信息 sizes 和其他关键字参数

    def get_numsides(self) -> int: ...
    # 获取正多边形的边数，返回一个整数

    def get_rotation(self) -> float: ...
    # 获取旋转角度，返回一个浮点数



class StarPolygonCollection(RegularPolyCollection):
    # 表示包含星形多边形集合的集合类，继承自 RegularPolyCollection
    ...


class AsteriskPolygonCollection(RegularPolyCollection):
    # 表示包含星号多边形集合的集合类，继承自 RegularPolyCollection
    ...


class LineCollection(Collection):
    # 表示包含线段集合的集合类，继承自 Collection
    def __init__(
        self, segments: Sequence[ArrayLike], *, zorder: float = ..., **kwargs
    ) -> None: ...
    # 初始化方法，接受线段的序列 segments，绘制顺序 zorder 和其他关键字参数

    def set_segments(self, segments: Sequence[ArrayLike] | None) -> None: ...
    # 设置线段集合，segments 是线段的序列或者 None

    def set_verts(self, segments: Sequence[ArrayLike] | None) -> None: ...
    # 设置顶点集合，segments 是顶点的序列或者 None

    def set_paths(self, segments: Sequence[ArrayLike] | None) -> None: ...
    # 设置路径集合，segments 是路径的序列或者 None

    def get_segments(self) -> list[np.ndarray]: ...
    # 获取线段集合，返回一个 NumPy 数组的列表

    def set_color(self, c: ColorType | Sequence[ColorType]) -> None: ...
    # 设置线段颜色，c 是颜色值或者颜色值的序列

    def set_colors(self, c: ColorType | Sequence[ColorType]) -> None: ...
    # 设置线段颜色，c 是颜色值或者颜色值的序列

    def set_gapcolor(self, gapcolor: ColorType | Sequence[ColorType] | None) -> None: ...
    # 设置间隙颜色，gapcolor 是颜色值或者颜色值的序列或者 None

    def get_color(self) -> ColorType | Sequence[ColorType]: ...
    # 获取线段颜色，返回一个颜色值或者颜色值的序列

    def get_colors(self) -> ColorType | Sequence[ColorType]: ...
    # 获取线段颜色，返回一个颜色值或者颜色值的序列

    def get_gapcolor(self) -> ColorType | Sequence[ColorType] | None: ...
    # 获取间隙颜色，返回一个颜色值或者颜色值的序列或者 None



class EventCollection(LineCollection):
    # 表示包含事件集合的集合类，继承自 LineCollection
    ...
    # 初始化方法，用于创建一个新的对象实例
    def __init__(
        self,
        positions: ArrayLike,  # 参数：位置信息，可以是数组或类似数组的对象
        orientation: Literal["horizontal", "vertical"] = ...,  # 参数：方向，默认可以是水平或垂直
        *,
        lineoffset: float = ...,  # 关键字参数：线条偏移量，浮点数类型
        linelength: float = ...,  # 关键字参数：线条长度，浮点数类型
        linewidth: float | Sequence[float] | None = ...,  # 关键字参数：线条宽度，可以是单个浮点数、浮点数序列或空
        color: ColorType | Sequence[ColorType] | None = ...,  # 关键字参数：线条颜色，可以是单个颜色类型、颜色类型序列或空
        linestyle: LineStyleType | Sequence[LineStyleType] = ...,  # 关键字参数：线条样式，可以是单个样式类型、样式类型序列
        antialiased: bool | Sequence[bool] | None = ...,  # 关键字参数：是否抗锯齿，可以是单个布尔值、布尔值序列或空
        **kwargs  # 其他关键字参数
    ) -> None:  # 返回值为 None，因为初始化方法不返回任何东西，只是初始化对象状态
        ...

    # 获取位置信息的方法，返回一个浮点数列表
    def get_positions(self) -> list[float]: ...

    # 设置位置信息的方法，参数为浮点数序列或空
    def set_positions(self, positions: Sequence[float] | None) -> None: ...

    # 添加位置信息的方法，参数为浮点数序列或空
    def add_positions(self, position: Sequence[float] | None) -> None: ...

    # 扩展位置信息的方法，参数为浮点数序列或空
    def extend_positions(self, position: Sequence[float] | None) -> None: ...

    # 追加位置信息的方法，参数为浮点数序列或空
    def append_positions(self, position: Sequence[float] | None) -> None: ...

    # 判断是否为水平方向的方法，返回布尔值
    def is_horizontal(self) -> bool: ...

    # 获取对象当前的方向，返回水平或垂直文本常量
    def get_orientation(self) -> Literal["horizontal", "vertical"]: ...

    # 切换对象的方向为另一种
    def switch_orientation(self) -> None: ...

    # 设置对象的方向，参数为水平或垂直文本常量
    def set_orientation(
        self, orientation: Literal["horizontal", "vertical"]
    ) -> None: ...

    # 获取线条长度的方法，返回单个浮点数或浮点数序列
    def get_linelength(self) -> float | Sequence[float]: ...

    # 设置线条长度的方法，参数为单个浮点数或浮点数序列
    def set_linelength(self, linelength: float | Sequence[float]) -> None: ...

    # 获取线条偏移量的方法，返回单个浮点数
    def get_lineoffset(self) -> float: ...

    # 设置线条偏移量的方法，参数为单个浮点数
    def set_lineoffset(self, lineoffset: float) -> None: ...

    # 获取线条宽度的方法，返回单个浮点数
    def get_linewidth(self) -> float: ...

    # 获取线条宽度的方法，返回浮点数序列
    def get_linewidths(self) -> Sequence[float]: ...

    # 获取线条颜色的方法，返回颜色类型
    def get_color(self) -> ColorType: ...
class CircleCollection(_CollectionWithSizes):
    # 继承自 _CollectionWithSizes 的圆形集合类

    def __init__(self, sizes: float | ArrayLike, **kwargs) -> None:
        # 初始化方法，接受圆形大小参数和额外关键字参数
        ...

class EllipseCollection(Collection):
    # 继承自 Collection 的椭圆集合类

    def __init__(
        self,
        widths: ArrayLike,
        heights: ArrayLike,
        angles: ArrayLike,
        *,
        units: Literal[
            "points", "inches", "dots", "width", "height", "x", "y", "xy"
        ] = ...,
        **kwargs
    ) -> None:
        # 初始化方法，接受椭圆宽度、高度、角度参数，以及单位参数和额外关键字参数
        ...

    def set_widths(self, widths: ArrayLike) -> None:
        # 设置椭圆集合的宽度数组

    def set_heights(self, heights: ArrayLike) -> None:
        # 设置椭圆集合的高度数组

    def set_angles(self, angles: ArrayLike) -> None:
        # 设置椭圆集合的角度数组

    def get_widths(self) -> ArrayLike:
        # 获取椭圆集合的宽度数组

    def get_heights(self) -> ArrayLike:
        # 获取椭圆集合的高度数组

    def get_angles(self) -> ArrayLike:
        # 获取椭圆集合的角度数组

class PatchCollection(Collection):
    # 继承自 Collection 的补丁集合类

    def __init__(
        self, patches: Iterable[Patch], *, match_original: bool = ..., **kwargs
    ) -> None:
        # 初始化方法，接受补丁对象的可迭代集合和匹配原始补丁的标志，以及额外关键字参数
        ...

    def set_paths(self, patches: Iterable[Patch]) -> None:
        # 设置补丁集合的路径数组，类型注解标记为忽略覆盖

class TriMesh(Collection):
    # 继承自 Collection 的三角网格集合类

    def __init__(self, triangulation: Triangulation, **kwargs) -> None:
        # 初始化方法，接受三角化对象和额外关键字参数
        ...

    def get_paths(self) -> list[Path]:
        # 获取三角网格集合的路径列表

    def set_paths(self) -> None:
        # 设置路径的方法，类型注解标记为忽略覆盖

    @staticmethod
    def convert_mesh_to_paths(tri: Triangulation) -> list[Path]:
        # 静态方法：将网格转换为路径列表的方法

class _MeshData:
    # 网格数据基类

    def __init__(
        self,
        coordinates: ArrayLike,
        *,
        shading: Literal["flat", "gouraud"] = ...,
    ) -> None:
        # 初始化方法，接受坐标数组和着色方式参数
        ...

    def set_array(self, A: ArrayLike | None) -> None:
        # 设置数组的方法

    def get_coordinates(self) -> ArrayLike:
        # 获取坐标数组的方法

    def get_facecolor(self) -> ColorType | Sequence[ColorType]:
        # 获取面颜色的方法

    def get_edgecolor(self) -> ColorType | Sequence[ColorType]:
        # 获取边缘颜色的方法

class QuadMesh(_MeshData, Collection):
    # 继承自 _MeshData 和 Collection 的四边形网格类

    def __init__(
        self,
        coordinates: ArrayLike,
        *,
        antialiased: bool = ...,
        shading: Literal["flat", "gouraud"] = ...,
        **kwargs
    ) -> None:
        # 初始化方法，接受坐标数组、抗锯齿标志、着色方式参数和额外关键字参数
        ...

    def get_paths(self) -> list[Path]:
        # 获取四边形网格的路径列表

    def set_paths(self) -> None:
        # 设置路径的方法，类型注解标记为忽略覆盖

    def get_datalim(self, transData: transforms.Transform) -> transforms.Bbox:
        # 获取数据边界的方法

    def get_cursor_data(self, event: MouseEvent) -> float:
        # 获取鼠标事件数据的方法

class PolyQuadMesh(_MeshData, PolyCollection):
    # 继承自 _MeshData 和 PolyCollection 的多边形四边形网格类

    def __init__(
        self,
        coordinates: ArrayLike,
        **kwargs
    ) -> None:
        # 初始化方法，接受坐标数组和额外关键字参数
        ...
```