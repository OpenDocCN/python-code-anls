# `D:\src\scipysrc\matplotlib\lib\matplotlib\colorbar.pyi`

```py
# 导入 matplotlib 库中特定模块，用于自定义图形的边框样式
import matplotlib.spines as mspines
# 从 matplotlib 库中导入特定模块，包括颜色映射、集合、颜色、轮廓等
from matplotlib import cm, collections, colors, contour
# 从 matplotlib 库中导入 Axes 类，表示图表中的坐标轴
from matplotlib.axes import Axes
# 从 matplotlib 库中导入 RendererBase 类，用于渲染图形基础的抽象基类
from matplotlib.backend_bases import RendererBase
# 从 matplotlib 库中导入 Patch 类，表示图表中的图形块
from matplotlib.patches import Patch
# 从 matplotlib 库中导入 Locator 和 Formatter 类，用于设置坐标轴刻度位置和格式
from matplotlib.ticker import Locator, Formatter
# 从 matplotlib 库中导入 Bbox 类，用于表示图形对象的边界框
from matplotlib.transforms import Bbox

# 导入 numpy 库，用于科学计算和数组操作
import numpy as np
# 导入 numpy.typing 库中的 ArrayLike 类型，表示类似数组的对象
from numpy.typing import ArrayLike
# 导入 collections.abc 库中的 Sequence 类型，表示序列类型的抽象基类
from collections.abc import Sequence
# 导入 typing 库中的 Any 和 Literal 类型，用于声明函数参数和变量类型
from typing import Any, Literal, overload
# 导入自定义的 ColorType 类型，用于表示颜色数据的类型
from .typing import ColorType

# 定义 _ColorbarSpine 类，继承自 mspines.Spines 类
class _ColorbarSpine(mspines.Spines):
    # 构造函数，初始化 _ColorbarSpine 对象
    def __init__(self, axes: Axes): ...
    
    # 获取窗口边界的方法，接受可选的渲染器参数，返回 Bbox 对象
    def get_window_extent(self, renderer: RendererBase | None = ...) -> Bbox:...
    
    # 设置顶点坐标的方法，接受 ArrayLike 类型的参数 xy
    def set_xy(self, xy: ArrayLike) -> None: ...
    
    # 绘制方法，接受可选的渲染器参数，无返回值
    def draw(self, renderer: RendererBase | None) -> None:...

# 定义 Colorbar 类
class Colorbar:
    # 定义成员变量 n_rasterize，表示光栅化数量
    n_rasterize: int
    # 定义成员变量 mappable，表示可映射对象的颜色映射
    mappable: cm.ScalarMappable
    # 定义成员变量 ax，表示图表的坐标轴
    ax: Axes
    # 定义成员变量 alpha，表示透明度
    alpha: float | None
    # 定义成员变量 cmap，表示颜色映射对象
    cmap: colors.Colormap
    # 定义成员变量 norm，表示归一化对象
    norm: colors.Normalize
    # 定义成员变量 values，表示数值序列或者 None
    values: Sequence[float] | None
    # 定义成员变量 boundaries，表示边界序列或者 None
    boundaries: Sequence[float] | None
    # 定义成员变量 extend，表示颜色条的扩展方式
    extend: Literal["neither", "both", "min", "max"]
    # 定义成员变量 spacing，表示刻度间距类型
    spacing: Literal["uniform", "proportional"]
    # 定义成员变量 orientation，表示颜色条的方向
    orientation: Literal["vertical", "horizontal"]
    # 定义成员变量 drawedges，表示是否绘制边缘线
    drawedges: bool
    # 定义成员变量 extendfrac，表示扩展的分数值或者自动模式
    extendfrac: Literal["auto"] | float | Sequence[float] | None
    # 定义成员变量 extendrect，表示是否使用扩展矩形
    extendrect: bool
    # 定义成员变量 solids，表示用于填充的 QuadMesh 对象或者 None
    solids: None | collections.QuadMesh
    # 定义成员变量 solids_patches，表示用于填充的 Patch 对象列表
    solids_patches: list[Patch]
    # 定义成员变量 lines，表示线条集合的列表
    lines: list[collections.LineCollection]
    # 定义成员变量 outline，表示颜色条的边框对象
    outline: _ColorbarSpine
    # 定义成员变量 dividers，表示分隔线的集合对象
    dividers: collections.LineCollection
    # 定义成员变量 ticklocation，表示刻度位置的枚举值
    ticklocation: Literal["left", "right", "top", "bottom"]
    
    # 构造函数，初始化 Colorbar 对象
    def __init__(
        self,
        ax: Axes,
        mappable: cm.ScalarMappable | None = ...,
        *,
        cmap: str | colors.Colormap | None = ...,
        norm: colors.Normalize | None = ...,
        alpha: float | None = ...,
        values: Sequence[float] | None = ...,
        boundaries: Sequence[float] | None = ...,
        orientation: Literal["vertical", "horizontal"] | None = ...,
        ticklocation: Literal["auto", "left", "right", "top", "bottom"] = ...,
        extend: Literal["neither", "both", "min", "max"] | None = ...,
        spacing: Literal["uniform", "proportional"] = ...,
        ticks: Sequence[float] | Locator | None = ...,
        format: str | Formatter | None = ...,
        drawedges: bool = ...,
        extendfrac: Literal["auto"] | float | Sequence[float] | None = ...,
        extendrect: bool = ...,
        label: str = ...,
        location: Literal["left", "right", "top", "bottom"] | None = ...
    ) -> None: ...
    
    # locator 属性的 getter 方法，返回 Locator 对象
    @property
    def locator(self) -> Locator: ...
    
    # locator 属性的 setter 方法，接受 Locator 类型的参数 loc，无返回值
    @locator.setter
    def locator(self, loc: Locator) -> None: ...
    
    # minorlocator 属性的 getter 方法，返回 Locator 对象
    @property
    def minorlocator(self) -> Locator: ...
    
    # minorlocator 属性的 setter 方法，接受 Locator 类型的参数 loc，无返回值
    @minorlocator.setter
    def minorlocator(self, loc: Locator) -> None: ...
    
    # formatter 属性的 getter 方法，返回 Formatter 对象
    @property
    def formatter(self) -> Formatter: ...
    
    # formatter 属性的 setter 方法，接受 Formatter 类型的参数 fmt，无返回值
    @formatter.setter
    def formatter(self, fmt: Formatter) -> None: ...
    
    # minorformatter 属性的 getter 方法，返回 Formatter 对象
    @property
    def minorformatter(self) -> Formatter: ...
    
    # minorformatter 属性的 setter 方法，接受 Formatter 类型的参数 fmt，无返回值
    @minorformatter.setter
    def minorformatter(self, fmt: Formatter) -> None: ...
    # 更新图例的标准设置，接受一个颜色映射对象作为参数
    def update_normal(self, mappable: cm.ScalarMappable) -> None: ...
    
    # 添加等高线集合到图中，可选择是否擦除现有的图像
    @overload
    def add_lines(self, CS: contour.ContourSet, erase: bool = ...) -> None: ...
    
    # 添加等高线到图中，可选择不同的参数组合来指定等高线的级别、颜色和线宽
    @overload
    def add_lines(
        self,
        levels: ArrayLike,
        colors: ColorType | Sequence[ColorType],
        linewidths: float | ArrayLike,
        erase: bool = ...,
    ) -> None: ...
    
    # 更新刻度线的显示
    def update_ticks(self) -> None: ...
    
    # 设置刻度线的位置和标签
    def set_ticks(
        self,
        ticks: Sequence[float] | Locator,
        *,
        labels: Sequence[str] | None = ...,
        minor: bool = ...,
        **kwargs
    ) -> None: ...
    
    # 获取刻度线的位置
    def get_ticks(self, minor: bool = ...) -> np.ndarray: ...
    
    # 设置刻度线的标签
    def set_ticklabels(
        self,
        ticklabels: Sequence[str],
        *,
        minor: bool = ...,
        **kwargs
    ) -> None: ...
    
    # 打开次要刻度线显示
    def minorticks_on(self) -> None: ...
    
    # 关闭次要刻度线显示
    def minorticks_off(self) -> None: ...
    
    # 设置坐标轴的标签
    def set_label(self, label: str, *, loc: str | None = ..., **kwargs) -> None: ...
    
    # 设置图形元素的透明度
    def set_alpha(self, alpha: float | np.ndarray) -> None: ...
    
    # 移除图形元素
    def remove(self) -> None: ...
    
    # 处理拖拽平移功能
    def drag_pan(self, button: Any, key: Any, x: float, y: float) -> None: ...
# 将 ColorbarBase 设置为 Colorbar 类的别名
ColorbarBase = Colorbar

# 定义 make_axes 函数，用于创建一个或多个子图（Axes 对象）
def make_axes(
    parents: Axes | list[Axes] | np.ndarray,  # 参数 parents 可以是单个 Axes 对象、Axes 对象列表或 NumPy 数组
    location: Literal["left", "right", "top", "bottom"] | None = ...,  # 参数 location 表示子图的位置，可以是指定的边界或者 None
    orientation: Literal["vertical", "horizontal"] | None = ...,  # 参数 orientation 表示子图的方向，垂直或水平，或者 None
    fraction: float = ...,  # fraction 参数控制子图的尺寸比例
    shrink: float = ...,  # shrink 参数控制子图的缩小比例
    aspect: float = ...,  # aspect 参数控制子图的纵横比
    **kwargs  # 其余的关键字参数传递给创建子图的函数
) -> tuple[Axes, dict[str, Any]]: ...  # 函数返回一个元组，包含创建的 Axes 对象和一组包含其他信息的字典

# 定义 make_axes_gridspec 函数，用于在指定的 parent Axes 上创建一个子图网格
def make_axes_gridspec(
    parent: Axes,  # 参数 parent 是要将子图网格附加到的父级 Axes 对象
    *,
    location: Literal["left", "right", "top", "bottom"] | None = ...,  # 参数 location 指定子图网格的位置，可以是指定的边界或者 None
    orientation: Literal["vertical", "horizontal"] | None = ...,  # 参数 orientation 指定子图网格的方向，垂直或水平，或者 None
    fraction: float = ...,  # fraction 参数控制子图网格的尺寸比例
    shrink: float = ...,  # shrink 参数控制子图网格的缩小比例
    aspect: float = ...,  # aspect 参数控制子图网格的纵横比
    **kwargs  # 其余的关键字参数传递给创建子图网格的函数
) -> tuple[Axes, dict[str, Any]]: ...  # 函数返回一个元组，包含创建的 Axes 对象和一组包含其他信息的字典
```