# `D:\src\scipysrc\matplotlib\lib\matplotlib\legend.pyi`

```
# 从 matplotlib 库中导入特定模块和类
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.offsetbox import (
    DraggableOffsetBox,
)
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from matplotlib.text import Text
from matplotlib.transforms import (
    BboxBase,
    Transform,
)

# 导入 pathlib 库，用于处理路径
import pathlib
# 导入 collections.abc 库中的 Iterable 类型，用于类型检查
from collections.abc import Iterable
# 导入 typing 库中的 Any, Literal, overload 类型，用于类型标注和检查
from typing import Any, Literal, overload
# 导入当前包中的 typing 模块中的 ColorType 类型
from .typing import ColorType

# 定义 DraggableLegend 类，继承自 DraggableOffsetBox 类
class DraggableLegend(DraggableOffsetBox):
    # 类型注释：legend 属性是 Legend 类型的对象
    legend: Legend
    # 构造函数，接受 legend 参数，use_blit 参数为布尔类型，update 参数为 Literal 类型，表示"loc" 或 "bbox"
    def __init__(
        self, legend: Legend, use_blit: bool = ..., update: Literal["loc", "bbox"] = ...
    ) -> None: ...
    # 定义 finalize_offset 方法，无返回值
    def finalize_offset(self) -> None: ...

# 定义 Legend 类，继承自 Artist 类
class Legend(Artist):
    # codes 属性是 dict 类型，键是 str 类型，值是 int 类型
    codes: dict[str, int]
    # zorder 属性是 float 类型，表示绘制顺序
    zorder: float
    # prop 属性是 FontProperties 类型，表示字体属性
    prop: FontProperties
    # texts 属性是 Text 对象组成的列表
    texts: list[Text]
    # legend_handles 属性是 Artist 对象或 None 组成的列表
    legend_handles: list[Artist | None]
    # numpoints 属性是 int 类型，表示图例中每个样本点的数量
    numpoints: int
    # markerscale 属性是 float 类型，表示标记的缩放比例
    markerscale: float
    # scatterpoints 属性是 int 类型，表示散点图中的点数量
    scatterpoints: int
    # borderpad 属性是 float 类型，表示图例边界填充
    borderpad: float
    # labelspacing 属性是 float 类型，表示标签间的间距
    labelspacing: float
    # handlelength 属性是 float 类型，表示图例句柄的长度
    handlelength: float
    # handleheight 属性是 float 类型，表示图例句柄的高度
    handleheight: float
    # handletextpad 属性是 float 类型，表示图例句柄和文本之间的间距
    handletextpad: float
    # borderaxespad 属性是 float 类型，表示图例和轴之间的间距
    borderaxespad: float
    # columnspacing 属性是 float 类型，表示列之间的间距
    columnspacing: float
    # shadow 属性是 bool 类型，表示是否显示阴影
    shadow: bool
    # isaxes 属性是 bool 类型，表示是否与轴相关联
    isaxes: bool
    # axes 属性是 Axes 类型，表示图例所属的轴
    axes: Axes
    # parent 属性是 Axes 或 Figure 类型，表示图例的父级对象
    parent: Axes | Figure
    # legendPatch 属性是 FancyBboxPatch 类型，表示图例的外边框
    legendPatch: FancyBboxPatch
    def __init__(
        self,
        parent: Axes | Figure,
        handles: Iterable[Artist | tuple[Artist, ...]],
        labels: Iterable[str],
        *,
        loc: str | tuple[float, float] | int | None = ...,
        numpoints: int | None = ...,
        markerscale: float | None = ...,
        markerfirst: bool = ...,
        reverse: bool = ...,
        scatterpoints: int | None = ...,
        scatteryoffsets: Iterable[float] | None = ...,
        prop: FontProperties | dict[str, Any] | None = ...,
        fontsize: float | str | None = ...,
        labelcolor: ColorType
        | Iterable[ColorType]
        | Literal["linecolor", "markerfacecolor", "mfc", "markeredgecolor", "mec"]
        | None = ...,
        borderpad: float | None = ...,
        labelspacing: float | None = ...,
        handlelength: float | None = ...,
        handleheight: float | None = ...,
        handletextpad: float | None = ...,
        borderaxespad: float | None = ...,
        columnspacing: float | None = ...,
        ncols: int = ...,
        mode: Literal["expand"] | None = ...,
        fancybox: bool | None = ...,
        shadow: bool | dict[str, Any] | None = ...,
        title: str | None = ...,
        title_fontsize: float | None = ...,
        framealpha: float | None = ...,
        edgecolor: Literal["inherit"] | ColorType | None = ...,
        facecolor: Literal["inherit"] | ColorType | None = ...,
        bbox_to_anchor: BboxBase
        | tuple[float, float]
        | tuple[float, float, float, float]
        | None = ...,
        bbox_transform: Transform | None = ...,
        frameon: bool | None = ...,
        handler_map: dict[Artist | type, HandlerBase] | None = ...,
        title_fontproperties: FontProperties | dict[str, Any] | None = ...,
        alignment: Literal["center", "left", "right"] = ...,
        ncol: int = ...,
        draggable: bool = ...
    ) -> None:
    ...
    def contains(self, mouseevent: MouseEvent) -> tuple[bool, dict[Any, Any]]:
    ...
    def set_ncols(self, ncols: int) -> None:
    ...
    @classmethod
    def get_default_handler_map(cls) -> dict[type, HandlerBase]:
    ...
    @classmethod
    def set_default_handler_map(cls, handler_map: dict[type, HandlerBase]) -> None:
    ...
    @classmethod
    def update_default_handler_map(
        cls, handler_map: dict[type, HandlerBase]
    ) -> None:
    ...
    def get_legend_handler_map(self) -> dict[type, HandlerBase]:
    ...
    @staticmethod
    def get_legend_handler(
        legend_handler_map: dict[type, HandlerBase], orig_handle: Any
    ) -> HandlerBase | None:
    ...
    def get_children(self) -> list[Artist]:
    ...
    def get_frame(self) -> Rectangle:
    ...
    def get_lines(self) -> list[Line2D]:
    ...
    def get_patches(self) -> list[Patch]:
    ...
    def get_texts(self) -> list[Text]:
    ...
    def set_alignment(self, alignment: Literal["center", "left", "right"]) -> None:
    ...
    def get_alignment(self) -> Literal["center", "left", "right"]:
    ...



    def __init__(
        self,
        parent: Axes | Figure,
        handles: Iterable[Artist | tuple[Artist, ...]],
        labels: Iterable[str],
        *,
        loc: str | tuple[float, float] | int | None = ...,
        numpoints: int | None = ...,
        markerscale: float | None = ...,
        markerfirst: bool = ...,
        reverse: bool = ...,
        scatterpoints: int | None = ...,
        scatteryoffsets: Iterable[float] | None = ...,
        prop: FontProperties | dict[str, Any] | None = ...,
        fontsize: float | str | None = ...,
        labelcolor: ColorType
        | Iterable[ColorType]
        | Literal["linecolor", "markerfacecolor", "mfc", "markeredgecolor", "mec"]
        | None = ...,
        borderpad: float | None = ...,
        labelspacing: float | None = ...,
        handlelength: float | None = ...,
        handleheight: float | None = ...,
        handletextpad: float | None = ...,
        borderaxespad: float | None = ...,
        columnspacing: float | None = ...,
        ncols: int = ...,
        mode: Literal["expand"] | None = ...,
        fancybox: bool | None = ...,
        shadow: bool | dict[str, Any] | None = ...,
        title: str | None = ...,
        title_fontsize: float | None = ...,
        framealpha: float | None = ...,
        edgecolor: Literal["inherit"] | ColorType | None = ...,
        facecolor: Literal["inherit"] | ColorType | None = ...,
        bbox_to_anchor: BboxBase
        | tuple[float, float]
        | tuple[float, float, float, float]
        | None = ...,
        bbox_transform: Transform | None = ...,
        frameon: bool | None = ...,
        handler_map: dict[Artist | type, HandlerBase] | None = ...,
        title_fontproperties: FontProperties | dict[str, Any] | None = ...,
        alignment: Literal["center", "left", "right"] = ...,
        ncol: int = ...,
        draggable: bool = ...
    ) -> None:
        """
        Initialize the legend handler map with specified parameters.

        Parameters:
        - parent: Axes or Figure object to which the legend will be attached.
        - handles: Iterable of Artist objects or tuples of Artist objects.
        - labels: Iterable of strings representing labels for the legend entries.
        - loc: Location of the legend, specified as a string, tuple of coordinates, or integer.
        - numpoints: Number of points in the legend for scatter plots.
        - markerscale: Scale factor for legend markers.
        - markerfirst: If True, markers are placed before labels.
        - reverse: If True, legend entries are displayed in reverse order.
        - scatterpoints: Number of scatter points for scatter plot legends.
        - scatteryoffsets: Iterable of offsets for scatter points.
        - prop: Font properties or dictionary of properties for the legend text.
        - fontsize: Font size for legend text.
        - labelcolor: Color of legend labels or special values like 'linecolor', 'markerfacecolor', etc.
        - borderpad: Padding between the legend border and content.
        - labelspacing: Spacing between legend entries.
        - handlelength: Length of legend handles.
        - handleheight: Height of legend handles.
        - handletextpad: Padding between legend handle and label text.
        - borderaxespad: Padding between the axes and legend border.
        - columnspacing: Spacing between columns of multiple legends.
        - ncols: Number of columns for multiple legends.
        - mode: Expansion mode for legends.
        - fancybox: If True, use a fancy box around the legend.
        - shadow: If True or a dictionary, enable shadow for the legend.
        - title: Title of the legend.
        - title_fontsize: Font size of the legend title.
        - framealpha: Transparency of the legend frame.
        - edgecolor: Color of the legend edge or 'inherit'.
        - facecolor: Color of the legend face or 'inherit'.
        - bbox_to_anchor: Bounding box coordinates or transformation for the legend.
        - bbox_transform: Transformation for the legend bounding box.
        - frameon: If True, draw a frame around the legend.
        - handler_map: Mapping of Artist or type to HandlerBase for custom legend handling.
        - title_fontproperties: Font properties or dictionary for the legend title text.
        - alignment: Alignment of the legend ('center', 'left', 'right').
        - ncol: Number of columns for the legend.
        - draggable: If True, allow the legend to be draggable.
        """
        pass

    def contains(self, mouseevent: MouseEvent) -> tuple[bool, dict[Any, Any]]:
        """
        Check if the legend contains the mouse event.

        Parameters:
        - mouseevent: MouseEvent to check against the legend.

        Returns:
        - Tuple (bool, dict): Boolean indicating if the event is within the legend,
          and a dictionary containing additional information.
        """
        pass

    def set_ncols(self, ncols: int) -> None:
        """
        Set the number of columns for the legend.

        Parameters:
        - ncols: Number of columns to set.
        """
        pass

    @classmethod
    def get_default_handler_map(cls) -> dict[type, HandlerBase]:
        """
        Get the default handler map for legends.

        Returns:
        - Dictionary mapping types to HandlerBase instances.
        """
        pass

    @classmethod
    def set_default_handler_map(cls, handler_map: dict[type, HandlerBase]) -> None:
        """
        Set the default handler map for legends.

        Parameters:
        - handler_map: Dictionary mapping types to HandlerBase instances.
        """
        pass

    @classmethod
    def update_default_handler_map(
        cls, handler_map: dict[type,
    # 设置图例的位置，可以接受字符串、二元组（浮点数）、整数或者空值作为参数，默认为空值
    def set_loc(self, loc: str | tuple[float, float] | int | None = ...) -> None: ...

    # 设置图例的标题，接受标题字符串和字体属性对象、字符串、路径对象或者空值作为参数，默认为空值
    def set_title(
        self, title: str, prop: FontProperties | str | pathlib.Path | None = ...
    ) -> None: ...

    # 获取图例的标题文本
    def get_title(self) -> Text: ...

    # 获取图例是否显示边框的状态，返回布尔值
    def get_frame_on(self) -> bool: ...

    # 设置图例是否显示边框，接受布尔值作为参数
    def set_frame_on(self, b: bool) -> None: ...

    # 将图例的绘制框架方法设置为显示或隐藏边框的方法
    draw_frame = set_frame_on

    # 获取图例的 bbox_anchor 属性，返回 BboxBase 对象
    def get_bbox_to_anchor(self) -> BboxBase: ...

    # 设置图例的 bbox_anchor 属性，接受 BboxBase 对象、四元组浮点数、二元组浮点数或空值作为参数
    def set_bbox_to_anchor(
        self,
        bbox: BboxBase
        | tuple[float, float]
        | tuple[float, float, float, float]
        | None,
        transform: Transform | None = ...
    ) -> None: ...

    # 方法重载：设置图例是否可拖动，接受 True、False 作为状态参数，以及是否使用 blit 和更新方式
    @overload
    def set_draggable(
        self,
        state: Literal[True],
        use_blit: bool = ...,
        update: Literal["loc", "bbox"] = ...,
    ) -> DraggableLegend: ...

    # 方法重载：设置图例是否可拖动，接受 True、False 作为状态参数，以及是否使用 blit 和更新方式
    @overload
    def set_draggable(
        self,
        state: Literal[False],
        use_blit: bool = ...,
        update: Literal["loc", "bbox"] = ...,
    ) -> None: ...

    # 获取图例是否可拖动的状态，返回布尔值
    def get_draggable(self) -> bool: ...
```