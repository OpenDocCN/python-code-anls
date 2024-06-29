# `D:\src\scipysrc\matplotlib\lib\matplotlib\axes\_base.pyi`

```
# 导入 matplotlib.artist 模块的别名 martist
import matplotlib.artist as martist

# 导入 datetime 模块
import datetime
# 导入 collections.abc 模块中的 Callable、Iterable、Iterator 和 Sequence 类型
from collections.abc import Callable, Iterable, Iterator, Sequence
# 导入 matplotlib 的 cbook 模块
from matplotlib import cbook
# 导入 matplotlib.artist 模块中的 Artist 类
from matplotlib.artist import Artist
# 导入 matplotlib.axis 模块中的 XAxis、YAxis 和 Tick 类
from matplotlib.axis import XAxis, YAxis, Tick
# 导入 matplotlib.backend_bases 模块中的 RendererBase、MouseButton 和 MouseEvent 类
from matplotlib.backend_bases import RendererBase, MouseButton, MouseEvent
# 导入 matplotlib.cbook 模块中的 CallbackRegistry 类
from matplotlib.cbook import CallbackRegistry
# 导入 matplotlib.container 模块中的 Container 类
from matplotlib.container import Container
# 导入 matplotlib.collections 模块中的 Collection 类
from matplotlib.collections import Collection
# 导入 matplotlib.cm 模块中的 ScalarMappable 类
from matplotlib.cm import ScalarMappable
# 导入 matplotlib.legend 模块中的 Legend 类
from matplotlib.legend import Legend
# 导入 matplotlib.lines 模块中的 Line2D 类
from matplotlib.lines import Line2D
# 导入 matplotlib.gridspec 模块中的 SubplotSpec 和 GridSpec 类
from matplotlib.gridspec import SubplotSpec, GridSpec
# 导入 matplotlib.figure 模块中的 Figure 类
from matplotlib.figure import Figure
# 导入 matplotlib.image 模块中的 AxesImage 类
from matplotlib.image import AxesImage
# 导入 matplotlib.patches 模块中的 Patch 类
from matplotlib.patches import Patch
# 导入 matplotlib.scale 模块中的 ScaleBase 类
from matplotlib.scale import ScaleBase
# 导入 matplotlib.spines 模块中的 Spines 类
from matplotlib.spines import Spines
# 导入 matplotlib.table 模块中的 Table 类
from matplotlib.table import Table
# 导入 matplotlib.text 模块中的 Text 类
from matplotlib.text import Text
# 导入 matplotlib.transforms 模块中的 Transform 和 Bbox 类
from matplotlib.transforms import Transform, Bbox

# 导入 cycler 模块中的 Cycler 类
from cycler import Cycler

# 导入 numpy 库并指定别名 np
import numpy as np
# 导入 numpy.typing 模块中的 ArrayLike 类型
from numpy.typing import ArrayLike
# 导入 typing 模块中的 Any、Literal 和 TypeVar 类型
from typing import Any, Literal, TypeVar, overload

# 定义泛型类型变量 _T
_T = TypeVar("_T", bound=Artist)

# 定义 _axis_method_wrapper 类，用于包装轴方法的属性名和方法名
class _axis_method_wrapper:
    attr_name: str
    method_name: str
    __doc__: str
    def __init__(
        self, attr_name: str, method_name: str, *, doc_sub: dict[str, str] | None = ...
    ) -> None: ...
    def __set_name__(self, owner: Any, name: str) -> None: ...

# 定义 _AxesBase 类，继承自 martist.Artist 类
class _AxesBase(martist.Artist):
    # 定义类属性
    name: str
    patch: Patch
    spines: Spines
    fmt_xdata: Callable[[float], str] | None
    fmt_ydata: Callable[[float], str] | None
    xaxis: XAxis
    yaxis: YAxis
    bbox: Bbox
    dataLim: Bbox
    transAxes: Transform
    transScale: Transform
    transLimits: Transform
    transData: Transform
    ignore_existing_data_limits: bool
    axison: bool
    containers: list[Container]
    callbacks: CallbackRegistry
    child_axes: list[_AxesBase]
    legend_: Legend | None
    title: Text
    _projection_init: Any

    # 定义初始化方法
    def __init__(
        self,
        fig: Figure,
        *args: tuple[float, float, float, float] | Bbox | int,
        facecolor: ColorType | None = ...,
        frameon: bool = ...,
        sharex: _AxesBase | None = ...,
        sharey: _AxesBase | None = ...,
        label: Any = ...,
        xscale: str | ScaleBase | None = ...,
        yscale: str | ScaleBase | None = ...,
        box_aspect: float | None = ...,
        forward_navigation_events: bool | Literal["auto"] = ...,
        **kwargs
    ) -> None: ...
    
    # 获取子图规范（SubplotSpec）的方法
    def get_subplotspec(self) -> SubplotSpec | None: ...
    
    # 设置子图规范（SubplotSpec）的方法
    def set_subplotspec(self, subplotspec: SubplotSpec) -> None: ...
    
    # 获取网格规范（GridSpec）的方法
    def get_gridspec(self) -> GridSpec | None: ...
    
    # 设置图形对象（Figure）的方法
    def set_figure(self, fig: Figure) -> None: ...
    
    # 获取视图限制（viewLim）属性的方法
    @property
    def viewLim(self) -> Bbox: ...
    
    # 获取 X 轴变换（Transform）的方法
    def get_xaxis_transform(
        self, which: Literal["grid", "tick1", "tick2"] = ...
    ) -> Transform: ...
    
    # 获取 X 轴文本1变换（Transform）的方法
    def get_xaxis_text1_transform(
        self, pad_points: float
    ) -> tuple[
        Transform,
        Literal["center", "top", "bottom", "baseline", "center_baseline"],
        Literal["center", "left", "right"],
    ]:
    # 函数签名，定义了返回类型为一个元组，包含一个 Transform 对象和两个字面量类型的字符串
    ...

    def get_xaxis_text2_transform(
        self, pad_points
    ) -> tuple[
        Transform,
        Literal["center", "top", "bottom", "baseline", "center_baseline"],
        Literal["center", "left", "right"],
    ]:
    # 获取 x 轴文本2的变换信息
    ...

    def get_yaxis_transform(
        self, which: Literal["grid", "tick1", "tick2"] = ...
    ) -> Transform:
    # 获取 y 轴的变换信息，可以指定类型为 grid、tick1 或 tick2
    ...

    def get_yaxis_text1_transform(
        self, pad_points
    ) -> tuple[
        Transform,
        Literal["center", "top", "bottom", "baseline", "center_baseline"],
        Literal["center", "left", "right"],
    ]:
    # 获取 y 轴文本1的变换信息
    ...

    def get_yaxis_text2_transform(
        self, pad_points
    ) -> tuple[
        Transform,
        Literal["center", "top", "bottom", "baseline", "center_baseline"],
        Literal["center", "left", "right"],
    ]:
    # 获取 y 轴文本2的变换信息
    ...

    def get_position(self, original: bool = ...) -> Bbox:
    # 获取当前轴对象的位置信息，返回一个 Bbox 对象
    ...

    def set_position(
        self,
        pos: Bbox | tuple[float, float, float, float],
        which: Literal["both", "active", "original"] = ...,
    ) -> None:
    # 设置轴对象的位置信息，可以指定作用于全部、激活或原始位置
    ...

    def reset_position(self) -> None:
    # 重置轴对象的位置信息
    ...

    def set_axes_locator(
        self, locator: Callable[[_AxesBase, RendererBase], Bbox]
    ) -> None:
    # 设置轴对象的定位器函数
    ...

    def get_axes_locator(self) -> Callable[[_AxesBase, RendererBase], Bbox]:
    # 获取轴对象当前的定位器函数
    ...

    def sharex(self, other: _AxesBase) -> None:
    # 在 x 轴方向上与另一个轴对象共享属性
    ...

    def sharey(self, other: _AxesBase) -> None:
    # 在 y 轴方向上与另一个轴对象共享属性
    ...

    def clear(self) -> None:
    # 清除轴对象上的所有内容
    ...

    def cla(self) -> None:
    # 清除当前轴对象的坐标轴及其内容
    ...

    class ArtistList(Sequence[_T]):
        def __init__(
            self,
            axes: _AxesBase,
            prop_name: str,
            valid_types: type | Iterable[type] | None = ...,
            invalid_types: type | Iterable[type] | None = ...,
        ) -> None:
        # 艺术家列表类的初始化方法，接受轴对象、属性名称以及可选的有效和无效类型参数
        ...

        def __len__(self) -> int:
        # 返回艺术家列表中元素的数量
        ...

        def __iter__(self) -> Iterator[_T]:
        # 返回一个迭代器，用于遍历艺术家列表中的元素
        ...

        @overload
        def __getitem__(self, key: int) -> _T:
        # 获取艺术家列表中指定索引位置的元素
        ...

        @overload
        def __getitem__(self, key: slice) -> list[_T]:
        # 获取艺术家列表中指定切片范围的元素列表
        ...

        @overload
        def __add__(self, other: _AxesBase.ArtistList[_T]) -> list[_T]:
        # 将当前艺术家列表与另一个同类型的列表连接，并返回连接后的新列表
        ...

        @overload
        def __add__(self, other: list[Any]) -> list[Any]:
        # 将当前艺术家列表与普通列表连接，并返回连接后的新列表
        ...

        @overload
        def __add__(self, other: tuple[Any]) -> tuple[Any]:
        # 将当前艺术家列表与元组连接，并返回连接后的新元组
        ...

        @overload
        def __radd__(self, other: _AxesBase.ArtistList[_T]) -> list[_T]:
        # 反向操作符重载：将另一个同类型的艺术家列表与当前列表连接，并返回连接后的新列表
        ...

        @overload
        def __radd__(self, other: list[Any]) -> list[Any]:
        # 反向操作符重载：将普通列表与当前艺术家列表连接，并返回连接后的新列表
        ...

        @overload
        def __radd__(self, other: tuple[Any]) -> tuple[Any]:
        # 反向操作符重载：将元组与当前艺术家列表连接，并返回连接后的新元组
        ...

    @property
    def artists(self) -> _AxesBase.ArtistList[Artist]:
    # 获取当前轴对象上的艺术家列表，其中元素为 Artist 类型
    ...

    @property
    def collections(self) -> _AxesBase.ArtistList[Collection]:
    # 获取当前轴对象上的集合列表，其中元素为 Collection 类型
    ...

    @property
    def images(self) -> _AxesBase.ArtistList[AxesImage]:
    # 获取当前轴对象上的图像列表，其中元素为 AxesImage 类型
    ...

    @property
    def lines(self) -> _AxesBase.ArtistList[Line2D]:
    # 获取当前轴对象上的线条列表，其中元素为 Line2D 类型
    ...

    @property
    def patches(self) -> _AxesBase.ArtistList[Patch]:
    # 获取当前轴对象上的图形列表，其中元素为 Patch 类型
    ...
    @property
    def tables(self) -> _AxesBase.ArtistList[Table]: ...
    # 返回当前图表对象中所有表格（Table）类型的艺术家（Artist）列表，但具体实现未提供
    
    @property
    def texts(self) -> _AxesBase.ArtistList[Text]: ...
    # 返回当前图表对象中所有文本（Text）类型的艺术家（Artist）列表，但具体实现未提供
    
    def get_facecolor(self) -> ColorType: ...
    # 返回当前图表对象的背景颜色
    
    def set_facecolor(self, color: ColorType | None) -> None: ...
    # 设置当前图表对象的背景颜色，参数color可以是颜色类型或者None表示透明
    
    @overload
    def set_prop_cycle(self, cycler: Cycler) -> None: ...
    @overload
    def set_prop_cycle(self, label: str, values: Iterable[Any]) -> None: ...
    @overload
    def set_prop_cycle(self, **kwargs: Iterable[Any]) -> None: ...
    # 设置图表的属性循环（property cycle），可以接受不同的参数形式进行设置
    
    def get_aspect(self) -> float | Literal["auto"]: ...
    # 返回当前图表对象的宽高比例（aspect ratio），可以是浮点数或者"auto"
    
    def set_aspect(
        self,
        aspect: float | Literal["auto", "equal"],
        adjustable: Literal["box", "datalim"] | None = ...,
        anchor: str | tuple[float, float] | None = ...,
        share: bool = ...,
    ) -> None: ...
    # 设置当前图表对象的宽高比例（aspect ratio）及其调整方式、锚点位置和是否共享比例
    
    def get_adjustable(self) -> Literal["box", "datalim"]: ...
    # 返回当前图表对象的可调整属性，可以是"box"或"datalim"
    
    def set_adjustable(
        self, adjustable: Literal["box", "datalim"], share: bool = ...
    ) -> None: ...
    # 设置当前图表对象的可调整属性，包括调整类型和是否共享比例
    
    def get_box_aspect(self) -> float | None: ...
    # 返回当前图表对象的盒子宽高比例（box aspect ratio），可以是浮点数或None
    
    def set_box_aspect(self, aspect: float | None = ...) -> None: ...
    # 设置当前图表对象的盒子宽高比例（box aspect ratio）
    
    def get_anchor(self) -> str | tuple[float, float]: ...
    # 返回当前图表对象的锚点位置，可以是字符串或元组形式的坐标
    
    def set_anchor(
        self, anchor: str | tuple[float, float], share: bool = ...
    ) -> None: ...
    # 设置当前图表对象的锚点位置及是否共享比例
    
    def get_data_ratio(self) -> float: ...
    # 返回当前图表对象的数据宽高比例（data ratio）
    
    def apply_aspect(self, position: Bbox | None = ...) -> None: ...
    # 根据给定的位置（或自动）应用宽高比例设置
    
    @overload
    def axis(
        self,
        arg: tuple[float, float, float, float] | bool | str | None = ...,
        /,
        *,
        emit: bool = ...
    ) -> tuple[float, float, float, float]: ...
    @overload
    def axis(
        self,
        *,
        emit: bool = ...,
        xmin: float | None = ...,
        xmax: float | None = ...,
        ymin: float | None = ...,
        ymax: float | None = ...
    ) -> tuple[float, float, float, float]: ...
    # 控制当前图表对象的坐标轴设置，支持多种参数形式
    
    def get_legend(self) -> Legend: ...
    # 返回当前图表对象的图例（legend）对象
    
    def get_images(self) -> list[AxesImage]: ...
    # 返回当前图表对象中所有图片（AxesImage）对象的列表
    
    def get_lines(self) -> list[Line2D]: ...
    # 返回当前图表对象中所有线条（Line2D）对象的列表
    
    def get_xaxis(self) -> XAxis: ...
    # 返回当前图表对象的X轴（XAxis）对象
    
    def get_yaxis(self) -> YAxis: ...
    # 返回当前图表对象的Y轴（YAxis）对象
    
    def has_data(self) -> bool: ...
    # 检查当前图表对象是否包含数据，返回布尔值
    
    def add_artist(self, a: Artist) -> Artist: ...
    # 向当前图表对象添加指定的艺术家（Artist），并返回该艺术家对象
    
    def add_child_axes(self, ax: _AxesBase) -> _AxesBase: ...
    # 向当前图表对象添加子图表对象（_AxesBase），并返回该子图表对象
    
    def add_collection(
        self, collection: Collection, autolim: bool = ...
    ) -> Collection: ...
    # 向当前图表对象添加集合（Collection），并返回该集合对象
    
    def add_image(self, image: AxesImage) -> AxesImage: ...
    # 向当前图表对象添加图片（AxesImage），并返回该图片对象
    
    def add_line(self, line: Line2D) -> Line2D: ...
    # 向当前图表对象添加线条（Line2D），并返回该线条对象
    
    def add_patch(self, p: Patch) -> Patch: ...
    # 向当前图表对象添加图形补丁（Patch），并返回该补丁对象
    
    def add_table(self, tab: Table) -> Table: ...
    # 向当前图表对象添加表格（Table），并返回该表格对象
    
    def add_container(self, container: Container) -> Container: ...
    # 向当前图表对象添加容器（Container），并返回该容器对象
    
    def relim(self, visible_only: bool = ...) -> None: ...
    # 重新设置当前图表对象的数据限制（data limits），可选是否仅包括可见部分
    
    def update_datalim(
        self, xys: ArrayLike, updatex: bool = ..., updatey: bool = ...
    ) -> None: ...
    # 更新当前图表对象的数据限制（data limits），根据给定的数据点
    
    def in_axes(self, mouseevent: MouseEvent) -> bool: ...
    # 检查给定的鼠标事件是否发生在当前图表对象内，返回布尔值
    
    def get_autoscale_on(self) -> bool: ...
    # 返回当前图表对象是否自动缩放的状态，返回布尔值
    
    def set_autoscale_on(self, b: bool) -> None: ...
    # 设置当前图表对象是否自动缩放的状态，参数b为布尔值
    
    @property
    def use_sticky_edges(self) -> bool: ...
    # 返回当前图表对象是否使用粘边（sticky edges）的状态，返回布尔值
    # 定义属性 use_sticky_edges 的 setter 方法，用于设置是否使用粘边效果
    @use_sticky_edges.setter
    def use_sticky_edges(self, b: bool) -> None: ...

    # 获取 x 轴边界的边距值
    def get_xmargin(self) -> float: ...

    # 获取 y 轴边界的边距值
    def get_ymargin(self) -> float: ...

    # 设置 x 轴边界的边距值
    def set_xmargin(self, m: float) -> None: ...

    # 设置 y 轴边界的边距值
    def set_ymargin(self, m: float) -> None: ...

    # 定义 margins 方法，用于设置边界的多种方式，可以根据参数设置 x 和 y 的边距值，可能返回两个边距值的元组或 None
    # 这里可能可以通过重载（overloads）来优化实现
    def margins(
        self,
        *margins: float,
        x: float | None = ...,
        y: float | None = ...,
        tight: bool | None = ...
    ) -> tuple[float, float] | None: ...

    # 设置栅格化的顺序值
    def set_rasterization_zorder(self, z: float | None) -> None: ...

    # 获取栅格化的顺序值
    def get_rasterization_zorder(self) -> float | None: ...

    # 自动调整绘图范围
    def autoscale(
        self,
        enable: bool = ...,
        axis: Literal["both", "x", "y"] = ...,
        tight: bool | None = ...,
    ) -> None: ...

    # 自动调整视图范围
    def autoscale_view(
        self, tight: bool | None = ..., scalex: bool = ..., scaley: bool = ...
    ) -> None: ...

    # 绘制指定的 Artist 对象
    def draw_artist(self, a: Artist) -> None: ...

    # 在帧中重新绘制
    def redraw_in_frame(self) -> None: ...

    # 获取是否显示边框
    def get_frame_on(self) -> bool: ...

    # 设置是否显示边框
    def set_frame_on(self, b: bool) -> None: ...

    # 获取轴线在网格线之上还是之下
    def get_axisbelow(self) -> bool | Literal["line"]: ...

    # 设置轴线在网格线之上还是之下
    def set_axisbelow(self, b: bool | Literal["line"]) -> None: ...

    # 显示或隐藏网格线
    def grid(
        self,
        visible: bool | None = ...,
        which: Literal["major", "minor", "both"] = ...,
        axis: Literal["both", "x", "y"] = ...,
        **kwargs
    ) -> None: ...

    # 设置刻度标签的格式
    def ticklabel_format(
        self,
        *,
        axis: Literal["both", "x", "y"] = ...,
        style: Literal["", "sci", "scientific", "plain"] | None = ...,
        scilimits: tuple[int, int] | None = ...,
        useOffset: bool | float | None = ...,
        useLocale: bool | None = ...,
        useMathText: bool | None = ...
    ) -> None: ...

    # 设置定位器参数
    def locator_params(
        self, axis: Literal["both", "x", "y"] = ..., tight: bool | None = ..., **kwargs
    ) -> None: ...

    # 设置刻度参数
    def tick_params(self, axis: Literal["both", "x", "y"] = ..., **kwargs) -> None: ...

    # 关闭坐标轴
    def set_axis_off(self) -> None: ...

    # 打开坐标轴
    def set_axis_on(self) -> None: ...

    # 获取 x 轴标签
    def get_xlabel(self) -> str: ...

    # 设置 x 轴标签
    def set_xlabel(
        self,
        xlabel: str,
        fontdict: dict[str, Any] | None = ...,
        labelpad: float | None = ...,
        *,
        loc: Literal["left", "center", "right"] | None = ...,
        **kwargs
    ) -> Text: ...

    # 反转 x 轴
    def invert_xaxis(self) -> None: ...

    # 获取 x 轴边界的范围
    def get_xbound(self) -> tuple[float, float]: ...

    # 设置 x 轴边界的范围
    def set_xbound(
        self, lower: float | None = ..., upper: float | None = ...
    ) -> None: ...

    # 获取 x 轴的当前显示范围
    def get_xlim(self) -> tuple[float, float]: ...

    # 设置 x 轴的显示范围
    def set_xlim(
        self,
        left: float | tuple[float, float] | None = ...,
        right: float | None = ...,
        *,
        emit: bool = ...,
        auto: bool | None = ...,
        xmin: float | None = ...,
        xmax: float | None = ...
    ) -> tuple[float, float]: ...

    # 获取 y 轴标签
    def get_ylabel(self) -> str: ...
    # 设置 Y 轴标签文本及其字体样式、标签与图表之间的间距，支持传入关键字参数
    def set_ylabel(
        self,
        ylabel: str,
        fontdict: dict[str, Any] | None = ...,
        labelpad: float | None = ...,
        *,
        loc: Literal["bottom", "center", "top"] | None = ...,
        **kwargs
    ) -> Text: ...

    # 反转 Y 轴方向，使得最大值变为最小值，最小值变为最大值
    def invert_yaxis(self) -> None: ...

    # 获取 Y 轴的数据范围（下限和上限）
    def get_ybound(self) -> tuple[float, float]: ...

    # 设置 Y 轴的数据范围（下限和上限），支持传入 None 表示保持原值
    def set_ybound(
        self, lower: float | None = ..., upper: float | None = ...
    ) -> None: ...

    # 获取当前 Y 轴的显示范围（底部和顶部）
    def get_ylim(self) -> tuple[float, float]: ...

    # 设置 Y 轴的显示范围（底部和顶部），支持单个值或元组表示范围，emit 参数控制是否发出事件
    def set_ylim(
        self,
        bottom: float | tuple[float, float] | None = ...,
        top: float | None = ...,
        *,
        emit: bool = ...,
        auto: bool | None = ...,
        ymin: float | None = ...,
        ymax: float | None = ...
    ) -> tuple[float, float]: ...

    # 格式化给定的 X 轴数据值为字符串
    def format_xdata(self, x: float) -> str: ...

    # 格式化给定的 Y 轴数据值为字符串
    def format_ydata(self, y: float) -> str: ...

    # 格式化给定的坐标点的 X 和 Y 值为字符串
    def format_coord(self, x: float, y: float) -> str: ...

    # 开启次要刻度线显示
    def minorticks_on(self) -> None: ...

    # 关闭次要刻度线显示
    def minorticks_off(self) -> None: ...

    # 返回是否可以进行缩放操作
    def can_zoom(self) -> bool: ...

    # 返回是否可以进行平移操作
    def can_pan(self) -> bool: ...

    # 返回是否启用了导航功能
    def get_navigate(self) -> bool: ...

    # 设置是否启用导航功能
    def set_navigate(self, b: bool) -> None: ...

    # 返回是否启用了前向导航事件处理
    def get_forward_navigation_events(self) -> bool | Literal["auto"]: ...

    # 设置是否启用前向导航事件处理
    def set_forward_navigation_events(self, forward: bool | Literal["auto"]) -> None: ...

    # 返回当前导航模式，可以是 PAN（平移）或 ZOOM（缩放）
    def get_navigate_mode(self) -> Literal["PAN", "ZOOM"] | None: ...

    # 设置导航模式为 PAN（平移）或 ZOOM（缩放）
    def set_navigate_mode(self, b: Literal["PAN", "ZOOM"] | None) -> None: ...

    # 开始进行平移操作，传入鼠标点击的 X 和 Y 坐标位置以及按钮信息
    def start_pan(self, x: float, y: float, button: MouseButton) -> None: ...

    # 结束当前的平移操作
    def end_pan(self) -> None: ...

    # 拖拽进行平移操作，传入鼠标按钮信息、按键信息（可选）、X 和 Y 坐标位置
    def drag_pan(
        self, button: MouseButton, key: str | None, x: float, y: float
    ) -> None: ...

    # 返回当前 Axes 对象下的所有子元素列表
    def get_children(self) -> list[Artist]: ...

    # 检查给定点是否在当前 Axes 对象内
    def contains_point(self, point: tuple[int, int]) -> bool: ...

    # 返回默认的额外边界框艺术家列表
    def get_default_bbox_extra_artists(self) -> list[Artist]: ...

    # 获取紧凑的边界框（用于自动布局）
    def get_tightbbox(
        self,
        renderer: RendererBase | None = ...,
        *,
        call_axes_locator: bool = ...,
        bbox_extra_artists: Sequence[Artist] | None = ...,
        for_layout_only: bool = ...
    ) -> Bbox | None: ...

    # 创建一个共享 X 轴的 Axes 对象
    def twinx(self) -> _AxesBase: ...

    # 创建一个共享 Y 轴的 Axes 对象
    def twiny(self) -> _AxesBase: ...

    # 获取共享 X 轴的视图组
    def get_shared_x_axes(self) -> cbook.GrouperView: ...

    # 获取共享 Y 轴的视图组
    def get_shared_y_axes(self) -> cbook.GrouperView: ...

    # 在外侧标记图表，可选择移除内部刻度线
    def label_outer(self, remove_inner_ticks: bool = ...) -> None: ...

    # 返回 X 轴网格线列表
    def get_xgridlines(self) -> list[Line2D]: ...

    # 返回 X 轴刻度线列表，可以包括次要刻度线
    def get_xticklines(self, minor: bool = ...) -> list[Line2D]: ...

    # 返回 Y 轴网格线列表
    def get_ygridlines(self) -> list[Line2D]: ...
    # 返回所有 y 轴的刻度线对象的列表，可以选择是否包括次要刻度线
    def get_yticklines(self, minor: bool = ...) -> list[Line2D]: ...
    
    # 将图像映射对象设置为科学计数法显示
    def _sci(self, im: ScalarMappable) -> None: ...
    
    # 返回是否自动缩放 x 轴的状态
    def get_autoscalex_on(self) -> bool: ...
    
    # 返回是否自动缩放 y 轴的状态
    def get_autoscaley_on(self) -> bool: ...
    
    # 设置是否自动缩放 x 轴的状态
    def set_autoscalex_on(self, b: bool) -> None: ...
    
    # 设置是否自动缩放 y 轴的状态
    def set_autoscaley_on(self, b: bool) -> None: ...
    
    # 返回 x 轴是否被反转
    def xaxis_inverted(self) -> bool: ...
    
    # 返回当前 x 轴的刻度类型
    def get_xscale(self) -> str: ...
    
    # 设置 x 轴的刻度类型
    def set_xscale(self, value: str | ScaleBase, **kwargs) -> None: ...
    
    # 返回 x 轴的刻度位置，可以选择是否包括次要刻度
    def get_xticks(self, *, minor: bool = ...) -> np.ndarray: ...
    
    # 设置 x 轴的刻度位置及其标签
    def set_xticks(
        self,
        ticks: ArrayLike,
        labels: Iterable[str] | None = ...,
        *,
        minor: bool = ...,
        **kwargs
    ) -> list[Tick]: ...
    
    # 返回主要 x 轴刻度标签的列表
    def get_xmajorticklabels(self) -> list[Text]: ...
    
    # 返回次要 x 轴刻度标签的列表
    def get_xminorticklabels(self) -> list[Text]: ...
    
    # 返回 x 轴刻度标签的列表，可以选择是否包括次要刻度和标签类型
    def get_xticklabels(
        self, minor: bool = ..., which: Literal["major", "minor", "both"] | None = ...
    ) -> list[Text]: ...
    
    # 设置 x 轴的刻度标签
    def set_xticklabels(
        self,
        labels: Iterable[str | Text],
        *,
        minor: bool = ...,
        fontdict: dict[str, Any] | None = ...,
        **kwargs
    ) -> list[Text]: ...
    
    # 返回 y 轴是否被反转
    def yaxis_inverted(self) -> bool: ...
    
    # 返回当前 y 轴的刻度类型
    def get_yscale(self) -> str: ...
    
    # 设置 y 轴的刻度类型
    def set_yscale(self, value: str | ScaleBase, **kwargs) -> None: ...
    
    # 返回 y 轴的刻度位置，可以选择是否包括次要刻度
    def get_yticks(self, *, minor: bool = ...) -> np.ndarray: ...
    
    # 设置 y 轴的刻度位置及其标签
    def set_yticks(
        self,
        ticks: ArrayLike,
        labels: Iterable[str] | None = ...,
        *,
        minor: bool = ...,
        **kwargs
    ) -> list[Tick]: ...
    
    # 返回主要 y 轴刻度标签的列表
    def get_ymajorticklabels(self) -> list[Text]: ...
    
    # 返回次要 y 轴刻度标签的列表
    def get_yminorticklabels(self) -> list[Text]: ...
    
    # 返回 y 轴刻度标签的列表，可以选择是否包括次要刻度和标签类型
    def get_yticklabels(
        self, minor: bool = ..., which: Literal["major", "minor", "both"] | None = ...
    ) -> list[Text]: ...
    
    # 设置 y 轴的刻度标签
    def set_yticklabels(
        self,
        labels: Iterable[str | Text],
        *,
        minor: bool = ...,
        fontdict: dict[str, Any] | None = ...,
        **kwargs
    ) -> list[Text]: ...
    
    # 将 x 轴刻度标签设置为日期格式，可以指定时区
    def xaxis_date(self, tz: str | datetime.tzinfo | None = ...) -> None: ...
    
    # 将 y 轴刻度标签设置为日期格式，可以指定时区
    def yaxis_date(self, tz: str | datetime.tzinfo | None = ...) -> None: ...
```