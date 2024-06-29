# `D:\src\scipysrc\matplotlib\lib\matplotlib\figure.pyi`

```py
# 从 collections.abc 导入 Callable、Hashable 和 Iterable 类型
from collections.abc import Callable, Hashable, Iterable
# 导入 os 模块
import os
# 从 typing 模块导入 Any、IO、Literal、Sequence 和 TypeVar 类型
from typing import Any, IO, Literal, Sequence, TypeVar, overload

# 导入 numpy 库，并从 numpy.typing 模块导入 ArrayLike 类型
import numpy as np
from numpy.typing import ArrayLike

# 导入 matplotlib 库中的相关模块和类
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backend_bases import (
    FigureCanvasBase,
    MouseButton,
    MouseEvent,
    RendererBase,
)
from matplotlib.colors import Colormap, Normalize
from matplotlib.colorbar import Colorbar
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec, SubplotSpec, SubplotParams as SubplotParams
from matplotlib.image import _ImageBase, FigureImage
from matplotlib.layout_engine import LayoutEngine
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from matplotlib.text import Text
from matplotlib.transforms import Affine2D, Bbox, BboxBase, Transform
from .typing import ColorType, HashableList

# 定义一个类型变量 _T
_T = TypeVar("_T")

# 定义 FigureBase 类，继承自 Artist 类
class FigureBase(Artist):
    # 定义类属性
    artists: list[Artist]  # 存储 Artist 对象的列表
    lines: list[Line2D]  # 存储 Line2D 对象的列表
    patches: list[Patch]  # 存储 Patch 对象的列表
    texts: list[Text]  # 存储 Text 对象的列表
    images: list[_ImageBase]  # 存储 _ImageBase 对象的列表
    legends: list[Legend]  # 存储 Legend 对象的列表
    subfigs: list[SubFigure]  # 存储 SubFigure 对象的列表
    stale: bool  # 是否过时的标志位
    suppressComposite: bool | None  # 是否抑制合成的标志位

    # 构造函数，接受任意关键字参数
    def __init__(self, **kwargs) -> None: ...

    # 自动格式化 X 轴日期
    def autofmt_xdate(
        self,
        bottom: float = ...,
        rotation: int = ...,
        ha: Literal["left", "center", "right"] = ...,
        which: Literal["major", "minor", "both"] = ...,
    ) -> None: ...

    # 获取子元素列表
    def get_children(self) -> list[Artist]: ...

    # 判断是否包含鼠标事件
    def contains(self, mouseevent: MouseEvent) -> tuple[bool, dict[Any, Any]]: ...

    # 设置总标题
    def suptitle(self, t: str, **kwargs) -> Text: ...

    # 获取总标题
    def get_suptitle(self) -> str: ...

    # 设置 X 轴总标签
    def supxlabel(self, t: str, **kwargs) -> Text: ...

    # 获取 X 轴总标签
    def get_supxlabel(self) -> str: ...

    # 设置 Y 轴总标签
    def supylabel(self, t: str, **kwargs) -> Text: ...

    # 获取 Y 轴总标签
    def get_supylabel(self) -> str: ...

    # 获取边框颜色
    def get_edgecolor(self) -> ColorType: ...

    # 获取背景颜色
    def get_facecolor(self) -> ColorType: ...

    # 获取是否有边框
    def get_frameon(self) -> bool: ...

    # 设置边框宽度
    def set_linewidth(self, linewidth: float) -> None: ...

    # 获取边框宽度
    def get_linewidth(self) -> float: ...

    # 设置边框颜色
    def set_edgecolor(self, color: ColorType) -> None: ...

    # 设置背景颜色
    def set_facecolor(self, color: ColorType) -> None: ...

    # 设置是否有边框
    def set_frameon(self, b: bool) -> None: ...

    # 返回 frameon 属性
    @property
    def frameon(self) -> bool: ...

    # 设置 frameon 属性
    @frameon.setter
    def frameon(self, b: bool) -> None: ...

    # 添加 Artist 对象到图形
    def add_artist(self, artist: Artist, clip: bool = ...) -> Artist: ...

    # 添加 Axes 对象到图形，第一种重载形式
    @overload
    def add_axes(self, ax: Axes) -> Axes: ...

    # 添加 Axes 对象到图形，第二种重载形式
    @overload
    def add_axes(
        self,
        rect: tuple[float, float, float, float],
        projection: None | str = ...,
        polar: bool = ...,
        **kwargs
    ) -> Axes: ...
    @overload
    def add_subplot(self, pos: int, **kwargs) -> Axes:
        ...

    @overload
    def add_subplot(self, ax: Axes, **kwargs) -> Axes:
        ...

    @overload
    def add_subplot(self, ax: SubplotSpec, **kwargs) -> Axes:
        ...

    @overload
    def add_subplot(self, **kwargs) -> Axes:
        ...
    # 调整子图布局，可以设置子图的左边界位置
    def subplots_adjust(
        self,
        left: float | None = ...,
        # 子图底部边界位置
        bottom: float | None = ...,
        # 子图右边界位置
        right: float | None = ...,
        # 子图顶部边界位置
        top: float | None = ...,
        # 子图之间的水平间距
        wspace: float | None = ...,
        # 子图之间的垂直间距
        hspace: float | None = ...,
    ) -> None: ...
    # 对给定的一组子图，调整它们的 x 轴标签对齐
    def align_xlabels(self, axs: Iterable[Axes] | None = ...) -> None: ...
    # 对给定的一组子图，调整它们的 y 轴标签对齐
    def align_ylabels(self, axs: Iterable[Axes] | None = ...) -> None: ...
    # 对给定的一组子图，调整它们的标题对齐
    def align_titles(self, axs: Iterable[Axes] | None = ...) -> None: ...
    # 对给定的一组子图，调整它们的标签对齐
    def align_labels(self, axs: Iterable[Axes] | None = ...) -> None: ...
    # 添加一个网格规格布局到图形中，指定行数和列数
    def add_gridspec(self, nrows: int = ..., ncols: int = ..., **kwargs) -> GridSpec: ...
    # 创建子图的重载方法，支持返回一个 ndarray 的数组
    @overload
    def subfigures(
        self,
        nrows: int = ...,
        ncols: int = ...,
        squeeze: Literal[False] = ...,
        wspace: float | None = ...,
        hspace: float | None = ...,
        width_ratios: ArrayLike | None = ...,
        height_ratios: ArrayLike | None = ...,
        **kwargs
    ) -> np.ndarray: ...
    # 创建子图的重载方法，支持返回一个 ndarray 或单个 SubFigure 对象
    @overload
    def subfigures(
        self,
        nrows: int = ...,
        ncols: int = ...,
        squeeze: Literal[True] = ...,
        wspace: float | None = ...,
        hspace: float | None = ...,
        width_ratios: ArrayLike | None = ...,
        height_ratios: ArrayLike | None = ...,
        **kwargs
    ) -> np.ndarray | SubFigure: ...
    # 向图形添加一个子图对象，使用给定的 SubplotSpec 参数
    def add_subfigure(self, subplotspec: SubplotSpec, **kwargs) -> SubFigure: ...
    # 设置当前轴对象为指定的 Axes 对象
    def sca(self, a: Axes) -> Axes: ...
    # 获取当前图形的当前轴对象
    def gca(self) -> Axes: ...
    # 获取当前图形的颜色映射对象，如果存在的话
    def _gci(self) -> ScalarMappable | None: ...
    # 处理投影需求的内部方法，根据参数返回 Axes 类型和参数字典
    def _process_projection_requirements(
        self, *, axes_class=None, polar=False, projection=None, **kwargs
    ) -> tuple[type[Axes], dict[str, Any]]: ...
    # 获取默认的 bbox_extra_artists 列表，用于包含在紧凑布局中的艺术家对象
    def get_default_bbox_extra_artists(self) -> list[Artist]: ...
    # 获取紧凑布局时的 tightbbox，返回图形的边界框
    def get_tightbbox(
        self,
        renderer: RendererBase | None = ...,
        *,
        bbox_extra_artists: Iterable[Artist] | None = ...,
    ) -> Bbox: ...
    # 创建子图网格布局，根据指定的 mosaic 参数进行设置
    @overload
    def subplot_mosaic(
        self,
        mosaic: str,
        *,
        sharex: bool = ...,
        sharey: bool = ...,
        width_ratios: ArrayLike | None = ...,
        height_ratios: ArrayLike | None = ...,
        empty_sentinel: str = ...,
        subplot_kw: dict[str, Any] | None = ...,
        per_subplot_kw: dict[str | tuple[str, ...], dict[str, Any]] | None = ...,
        gridspec_kw: dict[str, Any] | None = ...,
    ) -> dict[str, Axes]: ...
    # 创建子图网格布局，根据指定的 mosaic 列表参数进行设置
    @overload
    def subplot_mosaic(
        self,
        mosaic: list[HashableList[_T]],
        *,
        sharex: bool = ...,
        sharey: bool = ...,
        width_ratios: ArrayLike | None = ...,
        height_ratios: ArrayLike | None = ...,
        empty_sentinel: _T = ...,
        subplot_kw: dict[str, Any] | None = ...,
        per_subplot_kw: dict[_T | tuple[_T, ...], dict[str, Any]] | None = ...,
        gridspec_kw: dict[str, Any] | None = ...,
    ) -> dict[_T, Axes]: ...
    # 定义一个方法 subplot_mosaic，用于创建由 mosaic 参数定义的子图布局
    def subplot_mosaic(
        self,
        mosaic: list[HashableList[Hashable]],  # mosaic 是一个列表，其中的元素是 HashableList 类型的对象，描述子图布局
        *,
        sharex: bool = ...,  # 是否共享 x 轴
        sharey: bool = ...,  # 是否共享 y 轴
        width_ratios: ArrayLike | None = ...,  # 子图列的宽度比例
        height_ratios: ArrayLike | None = ...,  # 子图行的高度比例
        empty_sentinel: Any = ...,  # 用于标识空子图位置的对象
        subplot_kw: dict[str, Any] | None = ...,  # 应用于每个子图的关键字参数
        per_subplot_kw: dict[Hashable | tuple[Hashable, ...], dict[str, Any]] | None = ...,  # 针对每个子图的特定关键字参数
        gridspec_kw: dict[str, Any] | None = ...,  # 用于设置整体子图布局的关键字参数
    ) -> dict[Hashable, Axes]:  # 返回一个字典，将每个子图的标识符映射到对应的 Axes 对象
class SubFigure(FigureBase):
    # 子图形类，继承自FigureBase基类

    figure: Figure
    # figure属性，表示所属的Figure对象

    subplotpars: SubplotParams
    # subplotpars属性，表示子图参数

    dpi_scale_trans: Affine2D
    # dpi_scale_trans属性，表示DPI缩放的仿射变换

    transFigure: Transform
    # transFigure属性，表示图形的变换

    bbox_relative: Bbox
    # bbox_relative属性，表示相对位置的边界框

    figbbox: BboxBase
    # figbbox属性，表示图形边界框基类

    bbox: BboxBase
    # bbox属性，表示边界框基类

    transSubfigure: Transform
    # transSubfigure属性，表示子图的变换

    patch: Rectangle
    # patch属性，表示矩形补丁对象

    def __init__(
        self,
        parent: Figure | SubFigure,
        subplotspec: SubplotSpec,
        *,
        facecolor: ColorType | None = ...,
        edgecolor: ColorType | None = ...,
        linewidth: float = ...,
        frameon: bool | None = ...,
        **kwargs
    ) -> None:
        # 初始化方法，接受父图形对象或子图形对象、子图规范等参数

    @property
    def canvas(self) -> FigureCanvasBase:
        # canvas属性，返回与此子图形关联的画布对象

    @property
    def dpi(self) -> float:
        # dpi属性，返回DPI值

    @dpi.setter
    def dpi(self, value: float) -> None:
        # dpi属性的setter方法，设置DPI值

    def get_dpi(self) -> float:
        # 返回当前的DPI值

    def set_dpi(self, val) -> None:
        # 设置DPI值的方法

    def get_constrained_layout(self) -> bool:
        # 获取约束布局的状态

    def get_constrained_layout_pads(
        self, relative: bool = ...
    ) -> tuple[float, float, float, float]:
        # 获取约束布局的填充值

    def get_layout_engine(self) -> LayoutEngine:
        # 获取布局引擎对象

    @property  # type: ignore[misc]
    def axes(self) -> list[Axes]:
        # axes属性，返回与此子图形关联的所有轴对象列表

    def get_axes(self) -> list[Axes]:
        # 返回与此子图形关联的所有轴对象列表

class Figure(FigureBase):
    # 图形类，继承自FigureBase基类

    figure: Figure
    # figure属性，表示所属的Figure对象

    bbox_inches: Bbox
    # bbox_inches属性，表示图形的英寸边界框

    dpi_scale_trans: Affine2D
    # dpi_scale_trans属性，表示DPI缩放的仿射变换

    bbox: BboxBase
    # bbox属性，表示边界框基类

    figbbox: BboxBase
    # figbbox属性，表示图形边界框基类

    transFigure: Transform
    # transFigure属性，表示图形的变换

    transSubfigure: Transform
    # transSubfigure属性，表示子图的变换

    patch: Rectangle
    # patch属性，表示矩形补丁对象

    subplotpars: SubplotParams
    # subplotpars属性，表示子图参数

    def __init__(
        self,
        figsize: tuple[float, float] | None = ...,
        dpi: float | None = ...,
        *,
        facecolor: ColorType | None = ...,
        edgecolor: ColorType | None = ...,
        linewidth: float = ...,
        frameon: bool | None = ...,
        subplotpars: SubplotParams | None = ...,
        tight_layout: bool | dict[str, Any] | None = ...,
        constrained_layout: bool | dict[str, Any] | None = ...,
        layout: Literal["constrained", "compressed", "tight"]
        | LayoutEngine
        | None = ...,
        **kwargs
    ) -> None:
        # 初始化方法，接受图形大小、DPI、颜色、边框、布局等参数

    def pick(self, mouseevent: MouseEvent) -> None:
        # 处理选取事件的方法

    def set_layout_engine(
        self,
        layout: Literal["constrained", "compressed", "tight", "none"]
        | LayoutEngine
        | None = ...,
        **kwargs
    ) -> None:
        # 设置布局引擎的方法

    def get_layout_engine(self) -> LayoutEngine | None:
        # 获取布局引擎对象的方法

    def _repr_html_(self) -> str | None:
        # 返回HTML表示的方法

    def show(self, warn: bool = ...) -> None:
        # 显示图形的方法

    @property  # type: ignore[misc]
    def axes(self) -> list[Axes]:
        # axes属性，返回与此图形关联的所有轴对象列表

    def get_axes(self) -> list[Axes]:
        # 返回与此图形关联的所有轴对象列表

    @property
    def dpi(self) -> float:
        # dpi属性，返回DPI值

    @dpi.setter
    def dpi(self, dpi: float) -> None:
        # dpi属性的setter方法，设置DPI值

    def get_tight_layout(self) -> bool:
        # 获取紧凑布局的状态

    def get_constrained_layout_pads(
        self, relative: bool = ...
    ) -> tuple[float, float, float, float]:
        # 获取约束布局的填充值

    def get_constrained_layout(self) -> bool:
        # 获取约束布局的状态

    canvas: FigureCanvasBase
    # canvas属性，表示与此图形关联的画布对象

    def set_canvas(self, canvas: FigureCanvasBase) -> None:
        # 设置画布对象的方法
    # 定义一个方法 `figimage`，用于在图形中添加图像，返回一个 `FigureImage` 对象
    def figimage(
        self,
        X: ArrayLike,
        xo: int = ...,
        yo: int = ...,
        alpha: float | None = ...,
        norm: str | Normalize | None = ...,
        cmap: str | Colormap | None = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        origin: Literal["upper", "lower"] | None = ...,
        resize: bool = ...,
        **kwargs
    ) -> FigureImage: ...

    # 设置图形的尺寸，可以设置宽度和高度，支持前向调整
    def set_size_inches(
        self, w: float | tuple[float, float], h: float | None = ..., forward: bool = ...
    ) -> None: ...

    # 获取图形的尺寸，返回一个包含宽度和高度的 NumPy 数组
    def get_size_inches(self) -> np.ndarray: ...

    # 获取图形的宽度
    def get_figwidth(self) -> float: ...

    # 获取图形的高度
    def get_figheight(self) -> float: ...

    # 获取图形的 DPI（每英寸点数）
    def get_dpi(self) -> float: ...

    # 设置图形的 DPI
    def set_dpi(self, val: float) -> None: ...

    # 设置图形的宽度，支持前向调整
    def set_figwidth(self, val: float, forward: bool = ...) -> None: ...

    # 设置图形的高度，支持前向调整
    def set_figheight(self, val: float, forward: bool = ...) -> None: ...

    # 清除图形中的内容，可以选择保留观察者
    def clear(self, keep_observers: bool = ...) -> None: ...

    # 绘制图形但不进行渲染
    def draw_without_rendering(self) -> None: ...

    # 绘制指定的艺术家对象 `a`
    def draw_artist(self, a: Artist) -> None: ...

    # 向图形添加轴观察者，观察者为 `func` 指定的回调函数
    def add_axobserver(self, func: Callable[[Figure], Any]) -> None: ...

    # 将图形保存为文件 `fname`
    def savefig(
        self,
        fname: str | os.PathLike | IO,
        *,
        transparent: bool | None = ...,
        **kwargs
    ) -> None: ...

    # 获取用户的鼠标点击输入，返回一个坐标元组列表
    def ginput(
        self,
        n: int = ...,
        timeout: float = ...,
        show_clicks: bool = ...,
        mouse_add: MouseButton = ...,
        mouse_pop: MouseButton = ...,
        mouse_stop: MouseButton = ...
    ) -> list[tuple[int, int]]: ...

    # 等待用户按下按钮或鼠标，可选超时时间
    def waitforbuttonpress(self, timeout: float = ...) -> None | bool: ...

    # 调整子图的布局以避免重叠
    def tight_layout(
        self,
        *,
        pad: float = ...,
        h_pad: float | None = ...,
        w_pad: float | None = ...,
        rect: tuple[float, float, float, float] | None = ...
    ) -> None: ...
# 定义一个函数 figaspect，接受一个参数 arg，可以是浮点数或类数组对象，返回一个包含两个浮点数的元组
def figaspect(arg: float | ArrayLike) -> tuple[float, float]:
    ...
```