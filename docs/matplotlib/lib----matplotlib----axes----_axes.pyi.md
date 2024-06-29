# `D:\src\scipysrc\matplotlib\lib\matplotlib\axes\_axes.pyi`

```py
# 导入必要的类和函数以支持绘图和图形处理
from matplotlib.axes._base import _AxesBase  # 导入基本坐标轴类
from matplotlib.axes._secondary_axes import SecondaryAxis  # 导入次坐标轴类

from matplotlib.artist import Artist  # 导入图形对象的基类
from matplotlib.backend_bases import RendererBase  # 导入渲染器基类
from matplotlib.collections import (  # 导入图形集合类
    Collection,
    LineCollection,
    PathCollection,
    PolyCollection,
    EventCollection,
    QuadMesh,
)
from matplotlib.colors import Colormap, Normalize  # 导入颜色映射和归一化类
from matplotlib.container import (  # 导入容器类
    BarContainer,
    ErrorbarContainer,
    StemContainer,
)
from matplotlib.contour import ContourSet, QuadContourSet  # 导入轮廓和四轮廓集类
from matplotlib.image import AxesImage, PcolorImage  # 导入图像类
from matplotlib.legend import Legend  # 导入图例类
from matplotlib.legend_handler import HandlerBase  # 导入图例处理基类
from matplotlib.lines import Line2D, AxLine  # 导入线条类
from matplotlib.mlab import GaussianKDE  # 导入高斯核密度估计类
from matplotlib.patches import (  # 导入图形补丁类
    Rectangle,
    FancyArrow,
    Polygon,
    StepPatch,
    Wedge,
)
from matplotlib.quiver import Quiver, QuiverKey, Barbs  # 导入箭头类
from matplotlib.text import Annotation, Text  # 导入文本类
from matplotlib.transforms import Transform, Bbox  # 导入坐标变换和边界框类
import matplotlib.tri as mtri  # 导入三角剖分类
import matplotlib.table as mtable  # 导入表格类
import matplotlib.stackplot as mstack  # 导入堆叠图类
import matplotlib.streamplot as mstream  # 导入流场图类

import datetime  # 导入日期时间模块
import PIL.Image  # 导入图像处理模块
from collections.abc import Callable, Iterable, Sequence  # 导入集合抽象基类
from typing import Any, Literal, overload  # 导入类型提示相关类
import numpy as np  # 导入NumPy库
from numpy.typing import ArrayLike  # 导入NumPy数组类型提示
from matplotlib.typing import ColorType, MarkerType, LineStyleType  # 导入Matplotlib类型提示

class Axes(_AxesBase):
    def get_title(self, loc: Literal["left", "center", "right"] = ...) -> str: ...
    # 获取图表标题的方法，可以指定标题的位置
    def set_title(
        self,
        label: str,
        fontdict: dict[str, Any] | None = ...,
        loc: Literal["left", "center", "right"] | None = ...,
        pad: float | None = ...,
        *,
        y: float | None = ...,
        **kwargs
    ) -> Text: ...
    # 设置图表标题的方法，包括标题文本、字体字典、位置、间距等参数

    def get_legend_handles_labels(
        self, legend_handler_map: dict[type, HandlerBase] | None = ...
    ) -> tuple[list[Artist], list[Any]]: ...
    # 获取图例的处理对象和标签列表

    legend_: Legend | None  # 图例对象或None

    @overload
    def legend(self) -> Legend: ...
    @overload
    def legend(self, handles: Iterable[Artist | tuple[Artist, ...]], labels: Iterable[str], **kwargs) -> Legend: ...
    @overload
    def legend(self, *, handles: Iterable[Artist | tuple[Artist, ...]], **kwargs) -> Legend: ...
    @overload
    def legend(self, labels: Iterable[str], **kwargs) -> Legend: ...
    @overload
    def legend(self, **kwargs) -> Legend: ...
    # 图例方法的多态实现，支持不同的参数组合形式

    def inset_axes(
        self,
        bounds: tuple[float, float, float, float],
        *,
        transform: Transform | None = ...,
        zorder: float = ...,
        **kwargs
    ) -> Axes: ...
    # 创建子坐标轴的方法，支持设置位置和其他参数

    def indicate_inset(
        self,
        bounds: tuple[float, float, float, float],
        inset_ax: Axes | None = ...,
        *,
        transform: Transform | None = ...,
        facecolor: ColorType = ...,
        edgecolor: ColorType = ...,
        alpha: float = ...,
        zorder: float = ...,
        **kwargs
    ) -> Rectangle: ...
    # 指示插入区域的方法，支持设置插入区域的边界、样式和其他参数
    ```python`
    # 定义一个方法来指示插图的缩放，返回一个矩形对象
    def indicate_inset_zoom(self, inset_ax: Axes, **kwargs) -> Rectangle: ...
    
    # 定义一个方法来添加次要的 X 轴
    def secondary_xaxis(
        self,
        location: Literal["top", "bottom"] | float,
        functions: tuple[
            Callable[[ArrayLike], ArrayLike], Callable[[ArrayLike], ArrayLike]
        ]
        | Transform
        | None = ...,
        *,
        transform: Transform | None = ...,
        **kwargs
    ) -> SecondaryAxis: ...
    
    # 定义一个方法来添加次要的 Y 轴
    def secondary_yaxis(
        self,
        location: Literal["left", "right"] | float,
        functions: tuple[
            Callable[[ArrayLike], ArrayLike], Callable[[ArrayLike], ArrayLike]
        ]
        | Transform
        | None = ...,
        *,
        transform: Transform | None = ...,
        **kwargs
    ) -> SecondaryAxis: ...
    
    # 定义一个方法来在图上添加文本
    def text(
        self,
        x: float,
        y: float,
        s: str,
        fontdict: dict[str, Any] | None = ...,
        **kwargs
    ) -> Text: ...
    
    # 定义一个方法来添加注释
    def annotate(
        self,
        text: str,
        xy: tuple[float, float],
        xytext: tuple[float, float] | None = ...,
        xycoords: str
        | Artist
        | Transform
        | Callable[[RendererBase], Bbox | Transform]
        | tuple[float, float] = ...,
        textcoords: str
        | Artist
        | Transform
        | Callable[[RendererBase], Bbox | Transform]
        | tuple[float, float]
        | None = ...,
        arrowprops: dict[str, Any] | None = ...,
        annotation_clip: bool | None = ...,
        **kwargs
    ) -> Annotation: ...
    
    # 定义一个方法来绘制水平线
    def axhline(
        self, y: float = ..., xmin: float = ..., xmax: float = ..., **kwargs
    ) -> Line2D: ...
    
    # 定义一个方法来绘制垂直线
    def axvline(
        self, x: float = ..., ymin: float = ..., ymax: float = ..., **kwargs
    ) -> Line2D: ...
    
    # TODO: 可以分离 xy2 和 slope 的签名
    # 定义一个方法来绘制轴线
    def axline(
        self,
        xy1: tuple[float, float],
        xy2: tuple[float, float] | None = ...,
        *,
        slope: float | None = ...,
        **kwargs
    ) -> AxLine: ...
    
    # 定义一个方法来绘制水平方向的跨度区域
    def axhspan(
        self, ymin: float, ymax: float, xmin: float = ..., xmax: float = ..., **kwargs
    ) -> Rectangle: ...
    
    # 定义一个方法来绘制垂直方向的跨度区域
    def axvspan(
        self, xmin: float, xmax: float, ymin: float = ..., ymax: float = ..., **kwargs
    ) -> Rectangle: ...
    
    # 定义一个方法来绘制水平线集合
    def hlines(
        self,
        y: float | ArrayLike,
        xmin: float | ArrayLike,
        xmax: float | ArrayLike,
        colors: ColorType | Sequence[ColorType] | None = ...,
        linestyles: LineStyleType = ...,
        *,
        label: str = ...,
        data=...,
        **kwargs
    ) -> LineCollection: ...
    
    # 定义一个方法来绘制垂直线集合
    def vlines(
        self,
        x: float | ArrayLike,
        ymin: float | ArrayLike,
        ymax: float | ArrayLike,
        colors: ColorType | Sequence[ColorType] | None = ...,
        linestyles: LineStyleType = ...,
        *,
        label: str = ...,
        data=...,
        **kwargs
    ) -> LineCollection: ...
    def eventplot(
        self,
        positions: ArrayLike | Sequence[ArrayLike],
        *,
        orientation: Literal["horizontal", "vertical"] = ...,
        lineoffsets: float | Sequence[float] = ...,
        linelengths: float | Sequence[float] = ...,
        linewidths: float | Sequence[float] | None = ...,
        colors: ColorType | Sequence[ColorType] | None = ...,
        alpha: float | Sequence[float] | None = ...,
        linestyles: LineStyleType | Sequence[LineStyleType] = ...,
        data=...,
        **kwargs
    ) -> EventCollection:
    """
    # 创建事件图，根据给定的位置和参数绘制事件图

    Parameters:
    - positions: 事件的位置，可以是数组或数组序列
    - orientation: 图的方向，可以是水平或垂直
    - lineoffsets: 每个事件行的偏移量，可以是单个值或序列
    - linelengths: 每个事件行的长度，可以是单个值或序列
    - linewidths: 每个事件行的线宽，可以是单个值、序列或空
    - colors: 每个事件行的颜色，可以是颜色类型或颜色序列，或空
    - alpha: 每个事件行的透明度，可以是单个值或透明度序列，或空
    - linestyles: 每个事件行的线型，可以是线型类型或线型序列
    - data: 数据源
    - **kwargs: 其他关键字参数

    Returns:
    - EventCollection: 包含事件图形对象的集合
    """
    ...

    def plot(
        self,
        *args: float | ArrayLike | str,
        scalex: bool = ...,
        scaley: bool = ...,
        data = ...,
        **kwargs
    ) -> list[Line2D]:
    """
    # 绘制标准图表，支持多种参数和样式选项

    Parameters:
    - *args: 绘图的数据，可以是浮点数、数组或字符串
    - scalex: 是否在 x 轴上缩放
    - scaley: 是否在 y 轴上缩放
    - data: 数据源
    - **kwargs: 其他关键字参数

    Returns:
    - list[Line2D]: 包含所有线条对象的列表
    """
    ...

    def plot_date(
        self,
        x: ArrayLike,
        y: ArrayLike,
        fmt: str = ...,
        tz: str | datetime.tzinfo | None = ...,
        xdate: bool = ...,
        ydate: bool = ...,
        *,
        data=...,
        **kwargs
    ) -> list[Line2D]:
    """
    # 绘制日期格式的图表，支持日期数据和格式化选项

    Parameters:
    - x: x 轴的日期数据，可以是数组或类数组
    - y: y 轴的数据，可以是数组或类数组
    - fmt: 日期格式字符串
    - tz: 时区信息
    - xdate: 是否使用日期格式化 x 轴
    - ydate: 是否使用日期格式化 y 轴
    - data: 数据源
    - **kwargs: 其他关键字参数

    Returns:
    - list[Line2D]: 包含所有线条对象的列表
    """
    ...

    def loglog(self, *args, **kwargs) -> list[Line2D]:
    """
    # 绘制双对数坐标图，支持多种参数和样式选项

    Parameters:
    - *args: 绘图的数据
    - **kwargs: 其他关键字参数

    Returns:
    - list[Line2D]: 包含所有线条对象的列表
    """
    ...

    def semilogx(self, *args, **kwargs) -> list[Line2D]:
    """
    # 绘制 x 轴对数坐标图，支持多种参数和样式选项

    Parameters:
    - *args: 绘图的数据
    - **kwargs: 其他关键字参数

    Returns:
    - list[Line2D]: 包含所有线条对象的列表
    """
    ...

    def semilogy(self, *args, **kwargs) -> list[Line2D]:
    """
    # 绘制 y 轴对数坐标图，支持多种参数和样式选项

    Parameters:
    - *args: 绘图的数据
    - **kwargs: 其他关键字参数

    Returns:
    - list[Line2D]: 包含所有线条对象的列表
    """
    ...

    def acorr(
        self, x: ArrayLike, *, data=..., **kwargs
    ) -> tuple[np.ndarray, np.ndarray, LineCollection | Line2D, Line2D | None]:
    """
    # 计算自相关图，并绘制相关系数和线条

    Parameters:
    - x: 输入数据
    - data: 数据源
    - **kwargs: 其他关键字参数

    Returns:
    - tuple[np.ndarray, np.ndarray, LineCollection | Line2D, Line2D | None]: 自相关结果和线条对象的元组
    """
    ...

    def xcorr(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        normed: bool = ...,
        detrend: Callable[[ArrayLike], ArrayLike] = ...,
        usevlines: bool = ...,
        maxlags: int = ...,
        data = ...,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray, LineCollection | Line2D, Line2D | None]:
    """
    # 计算互相关图，并绘制相关系数和线条

    Parameters:
    - x: 输入数据
    - y: 输入数据
    - normed: 是否标准化
    - detrend: 去趋势函数
    - usevlines: 是否使用垂直线
    - maxlags: 最大滞后
    - data: 数据源
    - **kwargs: 其他关键字参数

    Returns:
    - tuple[np.ndarray, np.ndarray, LineCollection | Line2D, Line2D | None]: 互相关结果和线条对象的元组
    """
    ...

    def step(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *args,
        where: Literal["pre", "post", "mid"] = ...,
        data = ...,
        **kwargs
    ) -> list[Line2D]:
    """
    # 绘制阶梯图，支持多种参数和样式选项

    Parameters:
    - x: x 轴的数据
    - y: y 轴的数据
    - where: 画线的位置，可以是 'pre'、'post' 或 'mid'
    - data: 数据源
    - **kwargs: 其他关键字参数

    Returns:
    - list[Line2D]: 包含所有线条对象的列表
    """
    ...

    def bar(
        self,
        x: float | ArrayLike,
        height: float | ArrayLike,
        width: float | ArrayLike = ...,
        bottom: float | ArrayLike | None = ...,
        *,
        align: Literal["center", "edge"] = ...,
        data = ...,
        **kwargs
    ) -> BarContainer:
    """
    # 绘制柱状图，支持多种参数和样式选项

    Parameters:
    - x: 柱状图的 x 坐标
    - height: 柱状图的高度
    - width: 柱状图的宽度
    - bottom: 柱状图的底部位置
    - align: 柱状图的对齐方式，可以是 'center' 或 'edge'
    - data: 数据源
    - **kwargs: 其他关键字参数

    Returns:
    - BarContainer: 包含所有柱状图对象的容器
    """
    ...

    def barh(
        self,
        y: float | ArrayLike,
        width: float | ArrayLike,
        height: float | ArrayLike = ...,
        left: float | ArrayLike | None = ...,
        *,
        align: Literal["center", "edge"] = ...,
        data = ...,
        **kwargs
    ) -> BarContainer:
    """
    # 绘制水平柱状图，支持多种参数和样式选项

    Parameters:
    - y: 柱状图的 y 坐标
    - width: 柱状图的
    # 定义方法签名，表明该方法返回类型为 PolyCollection
    ) -> PolyCollection: ...

    # 定义 stem 方法，用于创建 STEM 图（用于显示数据分布情况），参数说明如下：
    #   *args: 可变位置参数，接受数组类型或字符串类型的输入
    #   linefmt: 线条格式字符串或 None，默认为 None
    #   markerfmt: 标记点格式字符串或 None，默认为 None
    #   basefmt: 基线格式字符串或 None，默认为 None
    #   bottom: 底部位置的浮点数，默认为 ...
    #   label: 标签字符串或 None，默认为 None
    #   orientation: 方向，可选值为 "vertical" 或 "horizontal"
    #   data: 数据，具体格式未指定
    # 返回类型为 StemContainer
    def stem(
        self,
        *args: ArrayLike | str,
        linefmt: str | None = ...,
        markerfmt: str | None = ...,
        basefmt: str | None = ...,
        bottom: float = ...,
        label: str | None = ...,
        orientation: Literal["vertical", "horizontal"] = ...,
        data=...,
    ) -> StemContainer: ...

    # TODO: data kwarg preprocessor? （待办：data 关键字参数预处理程序？）

    # 定义 pie 方法，用于创建饼图，参数说明如下：
    #   x: 数组类型，表示饼图各部分的大小
    #   explode: 数组类型或 None，用于指定是否分离某些扇形
    #   labels: 字符串序列或 None，饼图各部分的标签
    #   colors: 颜色类型或颜色类型序列或 None，饼图各部分的颜色
    #   autopct: 字符串或回调函数(float -> str)或 None，用于在饼图中显示百分比
    #   pctdistance: 浮点数，百分比标签与圆心的距离
    #   shadow: 布尔值，是否显示阴影
    #   labeldistance: 浮点数或 None，标签与圆心的距离
    #   startangle: 浮点数，起始角度
    #   radius: 浮点数，饼图的半径
    #   counterclock: 布尔值，是否逆时针旋转
    #   wedgeprops: 字典类型或 None，楔形块的属性
    #   textprops: 字典类型或 None，文本的属性
    #   center: 浮点数元组，饼图的中心坐标
    #   frame: 布尔值，是否绘制轴框架
    #   rotatelabels: 布尔值，是否旋转标签
    #   normalize: 布尔值，是否归一化
    #   hatch: 字符串或字符串序列或 None，用于填充图案的图案样式
    #   data: 数据，具体格式未指定
    # 返回类型为 tuple[list[Wedge], list[Text]] 或 tuple[list[Wedge], list[Text], list[Text]]
    def pie(
        self,
        x: ArrayLike,
        *,
        explode: ArrayLike | None = ...,
        labels: Sequence[str] | None = ...,
        colors: ColorType | Sequence[ColorType] | None = ...,
        autopct: str | Callable[[float], str] | None = ...,
        pctdistance: float = ...,
        shadow: bool = ...,
        labeldistance: float | None = ...,
        startangle: float = ...,
        radius: float = ...,
        counterclock: bool = ...,
        wedgeprops: dict[str, Any] | None = ...,
        textprops: dict[str, Any] | None = ...,
        center: tuple[float, float] = ...,
        frame: bool = ...,
        rotatelabels: bool = ...,
        normalize: bool = ...,
        hatch: str | Sequence[str] | None = ...,
        data=...,
    ) -> tuple[list[Wedge], list[Text]] | tuple[
        list[Wedge], list[Text], list[Text]
    ]: ...

    # 定义 errorbar 方法，用于创建误差条图，参数说明如下：
    #   x: 浮点数或数组类型，表示 x 轴上的位置
    #   y: 浮点数或数组类型，表示 y 轴上的位置
    #   yerr: 浮点数或数组类型或 None，y 方向的误差条大小
    #   xerr: 浮点数或数组类型或 None，x 方向的误差条大小
    #   fmt: 字符串，表示误差条的格式
    #   ecolor: 颜色类型或 None，误差线的颜色
    #   elinewidth: 浮点数或 None，误差线的线宽
    #   capsize: 浮点数或 None，误差条的帽子大小
    #   barsabove: 布尔值，误差条是否在数据点上方
    #   lolims: 布尔值或数组类型，是否存在低限
    #   uplims: 布尔值或数组类型，是否存在上限
    #   xlolims: 布尔值或数组类型，x 轴上的低限
    #   xuplims: 布尔值或数组类型，x 轴上的上限
    #   errorevery: 整数或整数元组，每隔多少个数据点绘制一次误差条
    #   capthick: 浮点数或 None，误差条帽子的线宽
    #   data: 数据，具体格式未指定
    #   **kwargs: 其他关键字参数
    # 返回类型为 ErrorbarContainer
    def errorbar(
        self,
        x: float | ArrayLike,
        y: float | ArrayLike,
        yerr: float | ArrayLike | None = ...,
        xerr: float | ArrayLike | None = ...,
        fmt: str = ...,
        *,
        ecolor: ColorType | None = ...,
        elinewidth: float | None = ...,
        capsize: float | None = ...,
        barsabove: bool = ...,
        lolims: bool | ArrayLike = ...,
        uplims: bool | ArrayLike = ...,
        xlolims: bool | ArrayLike = ...,
        xuplims: bool | ArrayLike = ...,
        errorevery: int | tuple[int, int] = ...,
        capthick: float | None = ...,
        data=...,
        **kwargs
    ) -> ErrorbarContainer: ...
    # 绘制箱线图的方法，用于可视化数据分布和统计信息
    def boxplot(
        self,
        x: ArrayLike | Sequence[ArrayLike],
        *,
        notch: bool | None = ...,
        sym: str | None = ...,
        vert: bool | None = ...,
        orientation: Literal["vertical", "horizontal"] = ...,
        whis: float | tuple[float, float] | None = ...,
        positions: ArrayLike | None = ...,
        widths: float | ArrayLike | None = ...,
        patch_artist: bool | None = ...,
        bootstrap: int | None = ...,
        usermedians: ArrayLike | None = ...,
        conf_intervals: ArrayLike | None = ...,
        meanline: bool | None = ...,
        showmeans: bool | None = ...,
        showcaps: bool | None = ...,
        showbox: bool | None = ...,
        showfliers: bool | None = ...,
        boxprops: dict[str, Any] | None = ...,
        tick_labels: Sequence[str] | None = ...,
        flierprops: dict[str, Any] | None = ...,
        medianprops: dict[str, Any] | None = ...,
        meanprops: dict[str, Any] | None = ...,
        capprops: dict[str, Any] | None = ...,
        whiskerprops: dict[str, Any] | None = ...,
        manage_ticks: bool = ...,
        autorange: bool = ...,
        zorder: float | None = ...,
        capwidths: float | ArrayLike | None = ...,
        label: Sequence[str] | None = ...,
        data=...,
    ) -> dict[str, Any]: ...

    # 使用给定的统计数据绘制箱线图
    def bxp(
        self,
        bxpstats: Sequence[dict[str, Any]],
        positions: ArrayLike | None = ...,
        *,
        widths: float | ArrayLike | None = ...,
        vert: bool | None = ...,
        orientation: Literal["vertical", "horizontal"] = ...,
        patch_artist: bool = ...,
        shownotches: bool = ...,
        showmeans: bool = ...,
        showcaps: bool = ...,
        showbox: bool = ...,
        showfliers: bool = ...,
        boxprops: dict[str, Any] | None = ...,
        whiskerprops: dict[str, Any] | None = ...,
        flierprops: dict[str, Any] | None = ...,
        medianprops: dict[str, Any] | None = ...,
        capprops: dict[str, Any] | None = ...,
        meanprops: dict[str, Any] | None = ...,
        meanline: bool = ...,
        manage_ticks: bool = ...,
        zorder: float | None = ...,
        capwidths: float | ArrayLike | None = ...,
        label: Sequence[str] | None = ...,
    ) -> dict[str, Any]: ...

    # 绘制散点图的方法，用于可视化 x 和 y 数据之间的关系
    def scatter(
        self,
        x: float | ArrayLike,
        y: float | ArrayLike,
        s: float | ArrayLike | None = ...,
        c: ArrayLike | Sequence[ColorType] | ColorType | None = ...,
        *,
        marker: MarkerType | None = ...,
        cmap: str | Colormap | None = ...,
        norm: str | Normalize | None = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        alpha: float | None = ...,
        linewidths: float | Sequence[float] | None = ...,
        edgecolors: Literal["face", "none"] | ColorType | Sequence[ColorType] | None = ...,
        plotnonfinite: bool = ...,
        data=...,
        **kwargs
    ) -> PathCollection: ...
    def hexbin(
        self,
        x: ArrayLike,
        y: ArrayLike,
        C: ArrayLike | None = ...,
        *,
        gridsize: int | tuple[int, int] = ...,
        bins: Literal["log"] | int | Sequence[float] | None = ...,
        xscale: Literal["linear", "log"] = ...,
        yscale: Literal["linear", "log"] = ...,
        extent: tuple[float, float, float, float] | None = ...,
        cmap: str | Colormap | None = ...,
        norm: str | Normalize | None = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        alpha: float | None = ...,
        linewidths: float | None = ...,
        edgecolors: Literal["face", "none"] | ColorType = ...,
        reduce_C_function: Callable[[np.ndarray | list[float]], float] = ...,
        mincnt: int | None = ...,
        marginals: bool = ...,
        data=...,
        **kwargs
    ) -> PolyCollection:
        ...


    def arrow(
        self, x: float, y: float, dx: float, dy: float, **kwargs
    ) -> FancyArrow:
        ...


    def quiverkey(
        self, Q: Quiver, X: float, Y: float, U: float, label: str, **kwargs
    ) -> QuiverKey:
        ...


    def quiver(self, *args, data=..., **kwargs) -> Quiver:
        ...


    def barbs(self, *args, data=..., **kwargs) -> Barbs:
        ...


    def fill(self, *args, data=..., **kwargs) -> list[Polygon]:
        ...


    def fill_between(
        self,
        x: ArrayLike,
        y1: ArrayLike | float,
        y2: ArrayLike | float = ...,
        where: Sequence[bool] | None = ...,
        interpolate: bool = ...,
        step: Literal["pre", "post", "mid"] | None = ...,
        *,
        data=...,
        **kwargs
    ) -> PolyCollection:
        ...


    def fill_betweenx(
        self,
        y: ArrayLike,
        x1: ArrayLike | float,
        x2: ArrayLike | float = ...,
        where: Sequence[bool] | None = ...,
        step: Literal["pre", "post", "mid"] | None = ...,
        interpolate: bool = ...,
        *,
        data=...,
        **kwargs
    ) -> PolyCollection:
        ...


    def imshow(
        self,
        X: ArrayLike | PIL.Image.Image,
        cmap: str | Colormap | None = ...,
        norm: str | Normalize | None = ...,
        *,
        aspect: Literal["equal", "auto"] | float | None = ...,
        interpolation: str | None = ...,
        alpha: float | ArrayLike | None = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        origin: Literal["upper", "lower"] | None = ...,
        extent: tuple[float, float, float, float] | None = ...,
        interpolation_stage: Literal["data", "rgba"] | None = ...,
        filternorm: bool = ...,
        filterrad: float = ...,
        resample: bool | None = ...,
        url: str | None = ...,
        data=...,
        **kwargs
    ) -> AxesImage:
        ...
    # 定义一个方法 pcolor，用于在图中绘制伪彩色图。接受可变数量的数组参数，以及一些关键字参数。
    def pcolor(
        self,
        *args: ArrayLike,
        shading: Literal["flat", "nearest", "auto"] | None = ...,
        alpha: float | None = ...,
        norm: str | Normalize | None = ...,
        cmap: str | Colormap | None = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        data=...,
        **kwargs
    ) -> Collection: ...

    # 定义一个方法 pcolormesh，用于在图中绘制伪彩色网格。接受可变数量的数组参数，以及一些关键字参数。
    def pcolormesh(
        self,
        *args: ArrayLike,
        alpha: float | None = ...,
        norm: str | Normalize | None = ...,
        cmap: str | Colormap | None = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        shading: Literal["flat", "nearest", "gouraud", "auto"] | None = ...,
        antialiased: bool = ...,
        data=...,
        **kwargs
    ) -> QuadMesh: ...

    # 定义一个方法 pcolorfast，用于在图中快速绘制伪彩色。接受可变数量的数组参数或者两个浮点数元组，以及一些关键字参数。
    def pcolorfast(
        self,
        *args: ArrayLike | tuple[float, float],
        alpha: float | None = ...,
        norm: str | Normalize | None = ...,
        cmap: str | Colormap | None = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        data=...,
        **kwargs
    ) -> AxesImage | PcolorImage | QuadMesh: ...

    # 定义一个方法 contour，用于在图中绘制等高线图。接受可变数量的参数和一些关键字参数。
    def contour(self, *args, data=..., **kwargs) -> QuadContourSet: ...

    # 定义一个方法 contourf，用于在图中填充等高线图。接受可变数量的参数和一些关键字参数。
    def contourf(self, *args, data=..., **kwargs) -> QuadContourSet: ...

    # 定义一个方法 clabel，用于为等高线图添加标签。接受 ContourSet 对象和一些关键字参数。
    def clabel(
        self, CS: ContourSet, levels: ArrayLike | None = ..., **kwargs
    ) -> list[Text]: ...

    # 定义一个方法 hist，用于绘制直方图。接受一个或多个数组作为输入，以及一系列的关键字参数。
    def hist(
        self,
        x: ArrayLike | Sequence[ArrayLike],
        bins: int | Sequence[float] | str | None = ...,
        *,
        range: tuple[float, float] | None = ...,
        density: bool = ...,
        weights: ArrayLike | None = ...,
        cumulative: bool | float = ...,
        bottom: ArrayLike | float | None = ...,
        histtype: Literal["bar", "barstacked", "step", "stepfilled"] = ...,
        align: Literal["left", "mid", "right"] = ...,
        orientation: Literal["vertical", "horizontal"] = ...,
        rwidth: float | None = ...,
        log: bool = ...,
        color: ColorType | Sequence[ColorType] | None = ...,
        label: str | Sequence[str] | None = ...,
        stacked: bool = ...,
        data=...,
        **kwargs
    ) -> tuple[
        np.ndarray | list[np.ndarray],
        np.ndarray,
        BarContainer | Polygon | list[BarContainer | Polygon],
    ]: ...

    # 定义一个方法 stairs，用于绘制阶梯图。接受数值和一些关键字参数。
    def stairs(
        self,
        values: ArrayLike,
        edges: ArrayLike | None = ...,
        *,
        orientation: Literal["vertical", "horizontal"] = ...,
        baseline: float | ArrayLike | None = ...,
        fill: bool = ...,
        data=...,
        **kwargs
    ) -> StepPatch: ...
    def hist2d(
        self,
        x: ArrayLike,
        y: ArrayLike,
        bins: None
        | int
        | tuple[int, int]
        | ArrayLike
        | tuple[ArrayLike, ArrayLike] = ...,
        *,
        range: ArrayLike | None = ...,
        density: bool = ...,
        weights: ArrayLike | None = ...,
        cmin: float | None = ...,
        cmax: float | None = ...,
        data=...,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, QuadMesh]: ...
    """
    二维直方图绘制函数，将数据点在二维空间中分箱并生成直方图。

    参数:
    - x: x 轴数据，ArrayLike 类型
    - y: y 轴数据，ArrayLike 类型
    - bins: 指定分箱的方式，可以是整数、元组形式的分箱尺寸或者数据数组形式
    - range: 数据的范围，ArrayLike 类型或 None
    - density: 是否为密度图，bool 类型
    - weights: 每个数据点的权重，ArrayLike 类型或 None
    - cmin, cmax: 颜色映射的最小和最大值，float 类型或 None
    - data: 可选参数，额外的数据
    - **kwargs: 其他关键字参数

    返回:
    - 包含 x 、 y 、 z 和 QuadMesh 对象的元组
    """
    
    def ecdf(
        self,
        x: ArrayLike,
        weights: ArrayLike | None = ...,
        *,
        complementary: bool=...,
        orientation: Literal["vertical", "horizonatal"]=...,
        compress: bool=...,
        data=...,
        **kwargs
    ) -> Line2D: ...
    """
    绘制经验累积分布函数（ECDF）的函数。

    参数:
    - x: 待绘制的数据，ArrayLike 类型
    - weights: 每个数据点的权重，ArrayLike 类型或 None
    - complementary: 是否为补集的 ECDF，bool 类型
    - orientation: 绘图方向，垂直或水平，Literal 类型
    - compress: 是否压缩绘图，bool 类型
    - data: 可选参数，额外的数据
    - **kwargs: 其他关键字参数

    返回:
    - 包含 ECDF 曲线的 Line2D 对象
    """
    
    def psd(
        self,
        x: ArrayLike,
        *,
        NFFT: int | None = ...,
        Fs: float | None = ...,
        Fc: int | None = ...,
        detrend: Literal["none", "mean", "linear"]
        | Callable[[ArrayLike], ArrayLike]
        | None = ...,
        window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = ...,
        noverlap: int | None = ...,
        pad_to: int | None = ...,
        sides: Literal["default", "onesided", "twosided"] | None = ...,
        scale_by_freq: bool | None = ...,
        return_line: bool | None = ...,
        data=...,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, Line2D]: ...
    """
    绘制功率谱密度（PSD）估计的函数。

    参数:
    - x: 输入数据，ArrayLike 类型
    - NFFT: FFT 窗口大小，int 类型或 None
    - Fs: 采样频率，float 类型或 None
    - Fc: 截止频率，int 类型或 None
    - detrend: 数据去趋势的方式，Literal 或 Callable 类型或 None
    - window: 加窗函数，Callable 类型、ArrayLike 类型或 None
    - noverlap: 重叠窗口大小，int 类型或 None
    - pad_to: FFT 结果的填充大小，int 类型或 None
    - sides: PSD 的计算方向，Literal 类型或 None
    - scale_by_freq: 是否按频率进行缩放，bool 类型或 None
    - return_line: 是否返回 Line2D 对象，bool 类型或 None
    - data: 可选参数，额外的数据
    - **kwargs: 其他关键字参数

    返回:
    - 包含频谱和功率谱的元组，或者包含频谱、功率谱和 Line2D 对象的元组
    """
    
    def csd(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        NFFT: int | None = ...,
        Fs: float | None = ...,
        Fc: int | None = ...,
        detrend: Literal["none", "mean", "linear"]
        | Callable[[ArrayLike], ArrayLike]
        | None = ...,
        window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = ...,
        noverlap: int | None = ...,
        pad_to: int | None = ...,
        sides: Literal["default", "onesided", "twosided"] | None = ...,
        scale_by_freq: bool | None = ...,
        return_line: bool | None = ...,
        data=...,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, Line2D]: ...
    """
    绘制交叉功率谱密度（CSD）估计的函数。

    参数:
    - x: 第一个输入数据，ArrayLike 类型
    - y: 第二个输入数据，ArrayLike 类型
    - NFFT: FFT 窗口大小，int 类型或 None
    - Fs: 采样频率，float 类型或 None
    - Fc: 截止频率，int 类型或 None
    - detrend: 数据去趋势的方式，Literal 或 Callable 类型或 None
    - window: 加窗函数，Callable 类型、ArrayLike 类型或 None
    - noverlap: 重叠窗口大小，int 类型或 None
    - pad_to: FFT 结果的填充大小，int 类型或 None
    - sides: CSD 的计算方向，Literal 类型或 None
    - scale_by_freq: 是否按频率进行缩放，bool 类型或 None
    - return_line: 是否返回 Line2D 对象，bool 类型或 None
    - data: 可选参数，额外的数据
    - **kwargs: 其他关键字参数

    返回:
    - 包含频谱和交叉谱的元组，或者包含频谱、交叉谱和 Line2D 对象的元组
    """
    
    def magnitude_spectrum(
        self,
        x: ArrayLike,
        *,
        Fs: float | None = ...,
        Fc: int | None = ...,
        window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = ...,
        pad_to: int | None = ...,
        sides: Literal["default", "onesided", "twosided"] | None = ...,
        scale: Literal["default", "linear", "dB"] | None = ...,
        data=...,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray, Line2D]: ...
    """
    计算和绘制幅度谱的函数。

    参数:
    - x: 输入数据，ArrayLike 类型
    - Fs: 采样频率，float 类型或 None
    - Fc: 截止频率，int 类型或 None
    - window: 加窗函数，Callable 类型、ArrayLike 类型或 None
    - pad_to: FFT 结果的填充大小，int 类型或 None
    - sides: PSD 的计算方向，Literal 类型或 None
    - scale: 幅度谱的缩放方式，Literal 类型或 None
    - data: 可选参数，额外的数据
    - **kwargs: 其他关键字参数

    返回:
    - 包含频率、幅度谱和 Line2D 对象的元组
    """
    
    def angle_spectrum(
        self,
        x: ArrayLike,
        *,
        Fs: float | None = ...,
        Fc: int | None = ...,
        window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = ...,
        pad_to: int | None = ...,
        sides: Literal["default", "onesided", "twosided"] | None = ...,
        data=...,
        **kwargs
    ) -> tuple
    ) -> tuple[np.ndarray, np.ndarray, Line2D]:
    # 定义函数签名，返回类型为包含三个元素的元组：两个 ndarray 和一个 Line2D 对象
    def phase_spectrum(
        self,
        x: ArrayLike,
        *,
        Fs: float | None = ...,
        Fc: int | None = ...,
        window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = ...,
        pad_to: int | None = ...,
        sides: Literal["default", "onesided", "twosided"] | None = ...,
        data=...,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray, Line2D]:
    # 定义 phase_spectrum 方法，计算信号的相位谱
    def cohere(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        NFFT: int = ...,
        Fs: float = ...,
        Fc: int = ...,
        detrend: Literal["none", "mean", "linear"]
        | Callable[[ArrayLike], ArrayLike] = ...,
        window: Callable[[ArrayLike], ArrayLike] | ArrayLike = ...,
        noverlap: int = ...,
        pad_to: int | None = ...,
        sides: Literal["default", "onesided", "twosided"] = ...,
        scale_by_freq: bool | None = ...,
        data=...,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
    # 定义 cohere 方法，计算信号的相干函数
    def specgram(
        self,
        x: ArrayLike,
        *,
        NFFT: int | None = ...,
        Fs: float | None = ...,
        Fc: int | None = ...,
        detrend: Literal["none", "mean", "linear"]
        | Callable[[ArrayLike], ArrayLike]
        | None = ...,
        window: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = ...,
        noverlap: int | None = ...,
        cmap: str | Colormap | None = ...,
        xextent: tuple[float, float] | None = ...,
        pad_to: int | None = ...,
        sides: Literal["default", "onesided", "twosided"] | None = ...,
        scale_by_freq: bool | None = ...,
        mode: Literal["default", "psd", "magnitude", "angle", "phase"] | None = ...,
        scale: Literal["default", "linear", "dB"] | None = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        data=...,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, AxesImage]:
    # 定义 specgram 方法，计算信号的谱图
    def spy(
        self,
        Z: ArrayLike,
        *,
        precision: float | Literal["present"] = ...,
        marker: str | None = ...,
        markersize: float | None = ...,
        aspect: Literal["equal", "auto"] | float | None = ...,
        origin: Literal["upper", "lower"] = ...,
        **kwargs
    ) -> AxesImage:
    # 定义 spy 方法，绘制矩阵 Z 的稀疏图
    def matshow(self, Z: ArrayLike, **kwargs) -> AxesImage:
    # 定义 matshow 方法，显示矩阵 Z 的图像表示
    # 定义一个方法 `violinplot`，用于绘制小提琴图
    def violinplot(
        self,
        dataset: ArrayLike | Sequence[ArrayLike],
        positions: ArrayLike | None = ...,
        *,
        vert: bool | None = ...,
        orientation: Literal["vertical", "horizontal"] = ...,
        widths: float | ArrayLike = ...,
        showmeans: bool = ...,
        showextrema: bool = ...,
        showmedians: bool = ...,
        quantiles: Sequence[float | Sequence[float]] | None = ...,
        points: int = ...,
        bw_method: Literal["scott", "silverman"]
        | float
        | Callable[[GaussianKDE], float]
        | None = ...,
        side: Literal["both", "low", "high"] = ...,
        data=...,
    ) -> dict[str, Collection]: ...

    # 定义一个方法 `violin`，用于绘制小提琴图的数据处理
    def violin(
        self,
        vpstats: Sequence[dict[str, Any]],
        positions: ArrayLike | None = ...,
        *,
        vert: bool | None = ...,
        orientation: Literal["vertical", "horizontal"] = ...,
        widths: float | ArrayLike = ...,
        showmeans: bool = ...,
        showextrema: bool = ...,
        showmedians: bool = ...,
        side: Literal["both", "low", "high"] = ...,
    ) -> dict[str, Collection]: ...

    # 赋值给变量 `table`，表示引入的模块 `mtable` 中的 `table` 方法
    table = mtable.table

    # 赋值给变量 `stackplot`，表示引入的模块 `mstack` 中的 `stackplot` 方法
    stackplot = mstack.stackplot

    # 赋值给变量 `streamplot`，表示引入的模块 `mstream` 中的 `streamplot` 方法
    streamplot = mstream.streamplot

    # 赋值给变量 `tricontour`，表示引入的模块 `mtri` 中的 `tricontour` 方法
    tricontour = mtri.tricontour

    # 赋值给变量 `tricontourf`，表示引入的模块 `mtri` 中的 `tricontourf` 方法
    tricontourf = mtri.tricontourf

    # 赋值给变量 `tripcolor`，表示引入的模块 `mtri` 中的 `tripcolor` 方法
    tripcolor = mtri.tripcolor

    # 赋值给变量 `triplot`，表示引入的模块 `mtri` 中的 `triplot` 方法
    triplot = mtri.triplot
```