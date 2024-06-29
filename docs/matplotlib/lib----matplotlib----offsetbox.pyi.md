# `D:\src\scipysrc\matplotlib\lib\matplotlib\offsetbox.pyi`

```py
# 导入需要的模块和类
import matplotlib.artist as martist
from matplotlib.backend_bases import RendererBase, Event, FigureCanvasBase
from matplotlib.colors import Colormap, Normalize
import matplotlib.text as mtext
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.image import BboxImage
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox, BboxBase, Transform

import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Callable, Sequence
from typing import Any, Literal, overload

# 调试标志
DEBUG: bool

# 定义函数 _get_packed_offsets
def _get_packed_offsets(
    widths: Sequence[float],
    total: float | None,
    sep: float | None,
    mode: Literal["fixed", "expand", "equal"] = ...,
) -> tuple[float, np.ndarray]:
    ...

# 定义 OffsetBox 类，继承自 martist.Artist
class OffsetBox(martist.Artist):
    width: float | None
    height: float | None
    
    # 初始化方法
    def __init__(self, *args, **kwargs) -> None:
        ...

    # 设置所属图形的方法
    def set_figure(self, fig: Figure) -> None:
        ...

    # 设置偏移量的方法
    def set_offset(
        self,
        xy: tuple[float, float]
        | Callable[[float, float, float, float, RendererBase], tuple[float, float]],
    ) -> None:
        ...

    # 获取偏移量的方法，重载版本1
    @overload
    def get_offset(self, bbox: Bbox, renderer: RendererBase) -> tuple[float, float]:
        ...

    # 获取偏移量的方法，重载版本2
    @overload
    def get_offset(
        self,
        width: float,
        height: float,
        xdescent: float,
        ydescent: float,
        renderer: RendererBase
    ) -> tuple[float, float]:
        ...

    # 设置宽度的方法
    def set_width(self, width: float) -> None:
        ...

    # 设置高度的方法
    def set_height(self, height: float) -> None:
        ...

    # 获取可见子元素列表的方法
    def get_visible_children(self) -> list[martist.Artist]:
        ...

    # 获取所有子元素列表的方法
    def get_children(self) -> list[martist.Artist]:
        ...

    # 获取边界框的方法
    def get_bbox(self, renderer: RendererBase) -> Bbox:
        ...

    # 获取窗口范围的方法
    def get_window_extent(self, renderer: RendererBase | None = ...) -> Bbox:
        ...

# 定义 PackerBase 类，继承自 OffsetBox
class PackerBase(OffsetBox):
    height: float | None
    width: float | None
    sep: float | None
    pad: float | None
    mode: Literal["fixed", "expand", "equal"]
    align: Literal["top", "bottom", "left", "right", "center", "baseline"]

    # 初始化方法
    def __init__(
        self,
        pad: float | None = ...,
        sep: float | None = ...,
        width: float | None = ...,
        height: float | None = ...,
        align: Literal["top", "bottom", "left", "right", "center", "baseline"] = ...,
        mode: Literal["fixed", "expand", "equal"] = ...,
        children: list[martist.Artist] | None = ...,
    ) -> None:
        ...

# 定义 VPacker 类，继承自 PackerBase
class VPacker(PackerBase):
    ...

# 定义 HPacker 类，继承自 PackerBase
class HPacker(PackerBase):
    ...

# 定义 PaddedBox 类，继承自 OffsetBox
class PaddedBox(OffsetBox):
    pad: float | None
    patch: FancyBboxPatch

    # 初始化方法
    def __init__(
        self,
        child: martist.Artist,
        pad: float | None = ...,
        *,
        draw_frame: bool = ...,
        patch_attrs: dict[str, Any] | None = ...,
    ) -> None:
        ...

    # 更新边框的方法
    def update_frame(self, bbox: Bbox, fontsize: float | None = ...) -> None:
        ...

    # 绘制边框的方法
    def draw_frame(self, renderer: RendererBase) -> None:
        ...

# 定义 DrawingArea 类，继承自 OffsetBox
class DrawingArea(OffsetBox):
    width: float
    # 定义一个浮点型变量，表示对象的高度
    height: float
    # 定义一个浮点型变量，表示对象的X轴下降
    xdescent: float
    # 定义一个浮点型变量，表示对象的Y轴下降
    ydescent: float
    # 定义一个Transform类型的变量，用于偏移转换
    offset_transform: Transform
    # 定义一个Transform类型的变量，用于DPI转换
    dpi_transform: Transform

    # 初始化方法，接受对象的宽度、高度、X轴下降、Y轴下降和是否裁剪作为参数
    def __init__(
        self,
        width: float,
        height: float,
        xdescent: float = ...,
        ydescent: float = ...,
        clip: bool = ...,
    ) -> None: ...

    # 获取属性，表示是否裁剪子对象
    @property
    def clip_children(self) -> bool: ...

    # 设置属性，用于设置是否裁剪子对象
    @clip_children.setter
    def clip_children(self, val: bool) -> None: ...

    # 获取对象的转换信息
    def get_transform(self) -> Transform: ...

    # 设置对象的偏移量，接受一个包含两个浮点数的元组作为参数
    # 注意：此方法不接受超类的所有选项
    def set_offset(self, xy: tuple[float, float]) -> None: ...  # type: ignore[override]

    # 获取对象的偏移量，返回一个包含两个浮点数的元组
    # 注意：此方法不接受超类的所有选项
    def get_offset(self) -> tuple[float, float]: ...  # type: ignore[override]

    # 添加一个艺术家对象作为子对象
    def add_artist(self, a: martist.Artist) -> None: ...
# 定义一个名为 TextArea 的类，它是 OffsetBox 的子类
class TextArea(OffsetBox):
    offset_transform: Transform  # 定义一个名为 offset_transform 的属性，类型为 Transform

    # 初始化方法，接受一个字符串 s 作为参数，还可以接受 textprops 和 multilinebaseline 作为可选参数
    def __init__(
        self,
        s: str,
        *,
        textprops: dict[str, Any] | None = ...,  # 文本属性的字典，可以为空
        multilinebaseline: bool = ...,  # 是否多行基线的布尔值
    ) -> None: ...

    # 设置文本内容的方法，接受一个字符串 s 作为参数，无返回值
    def set_text(self, s: str) -> None: ...

    # 获取当前文本内容的方法，返回一个字符串
    def get_text(self) -> str: ...

    # 设置是否多行基线的方法，接受一个布尔值 t 作为参数，无返回值
    def set_multilinebaseline(self, t: bool) -> None: ...

    # 获取是否多行基线的方法，返回一个布尔值
    def get_multilinebaseline(self) -> bool: ...

    # 覆盖父类方法，设置偏移量的方法，接受一个二元组 (x, y) 作为参数，无返回值
    def set_offset(self, xy: tuple[float, float]) -> None: ...  # type: ignore[override]

    # 覆盖父类方法，获取偏移量的方法，返回一个二元组 (x, y)
    def get_offset(self) -> tuple[float, float]: ...  # type: ignore[override]

# 定义一个名为 AuxTransformBox 的类，它是 OffsetBox 的子类
class AuxTransformBox(OffsetBox):
    aux_transform: Transform  # 定义一个名为 aux_transform 的属性，类型为 Transform
    offset_transform: Transform  # 定义一个名为 offset_transform 的属性，类型为 Transform
    ref_offset_transform: Transform  # 定义一个名为 ref_offset_transform 的属性，类型为 Transform

    # 初始化方法，接受一个 Transform 对象 aux_transform 作为参数
    def __init__(self, aux_transform: Transform) -> None: ...

    # 添加艺术家对象的方法，接受一个 martist.Artist 对象 a 作为参数，无返回值
    def add_artist(self, a: martist.Artist) -> None: ...

    # 获取当前对象的变换方法，返回一个 Transform 对象
    def get_transform(self) -> Transform: ...

    # 覆盖父类方法，设置偏移量的方法，接受一个二元组 (x, y) 作为参数，无返回值
    def set_offset(self, xy: tuple[float, float]) -> None: ...  # type: ignore[override]

    # 覆盖父类方法，获取偏移量的方法，返回一个二元组 (x, y)
    def get_offset(self) -> tuple[float, float]: ...  # type: ignore[override]

# 定义一个名为 AnchoredOffsetbox 的类，它是 OffsetBox 的子类
class AnchoredOffsetbox(OffsetBox):
    zorder: float  # 定义一个名为 zorder 的属性，类型为 float
    codes: dict[str, int]  # 定义一个名为 codes 的属性，类型为字典，键为字符串，值为整数
    loc: int  # 定义一个名为 loc 的属性，类型为整数
    borderpad: float  # 定义一个名为 borderpad 的属性，类型为 float
    pad: float  # 定义一个名为 pad 的属性，类型为 float
    prop: FontProperties  # 定义一个名为 prop 的属性，类型为 FontProperties
    patch: FancyBboxPatch  # 定义一个名为 patch 的属性，类型为 FancyBboxPatch

    # 初始化方法，接受一个字符串 loc 作为位置参数，还可以接受一系列可选参数
    def __init__(
        self,
        loc: str,
        *,
        pad: float = ...,  # 内边距的浮点数，默认值由外部决定
        borderpad: float = ...,  # 边框内边距的浮点数，默认值由外部决定
        child: OffsetBox | None = ...,  # 嵌套的 OffsetBox 对象，可以为空
        prop: FontProperties | None = ...,  # 字体属性，可以为空
        frameon: bool = ...,  # 是否显示框架的布尔值
        bbox_to_anchor: BboxBase | tuple[float, float] | tuple[float, float, float, float] | None = ...,  # 锚点的位置或变换框，可以为空
        bbox_transform: Transform | None = ...,  # 变换对象，可以为空
        **kwargs  # 其他关键字参数
    ) -> None: ...

    # 设置嵌套的 OffsetBox 对象的方法，接受一个 OffsetBox 对象或 None 作为参数，无返回值
    def set_child(self, child: OffsetBox | None) -> None: ...

    # 获取当前嵌套的 OffsetBox 对象的方法，返回一个 OffsetBox 对象或 None
    def get_child(self) -> OffsetBox | None: ...

    # 获取所有子艺术家对象的方法，返回一个 martist.Artist 对象的列表
    def get_children(self) -> list[martist.Artist]: ...

    # 获取锚点位置的变换框对象的方法，返回一个 Bbox 对象
    def get_bbox_to_anchor(self) -> Bbox: ...

    # 设置锚点位置的变换框对象的方法，接受一个 BboxBase 对象和一个可选的 Transform 对象作为参数，无返回值
    def set_bbox_to_anchor(
        self, bbox: BboxBase, transform: Transform | None = ...
    ) -> None: ...

    # 更新框架的方法，接受一个 Bbox 对象和一个可选的字体大小作为参数，无返回值
    def update_frame(self, bbox: Bbox, fontsize: float | None = ...) -> None: ...

# 定义一个名为 AnchoredText 的类，它是 AnchoredOffsetbox 的子类
class AnchoredText(AnchoredOffsetbox):
    txt: TextArea  # 定义一个名为 txt 的属性，类型为 TextArea

    # 初始化方法，接受一个字符串 s 和一个字符串 loc 作为参数，还可以接受一系列可选参数
    def __init__(
        self,
        s: str,
        loc: str,
        *,
        pad: float = ...,  # 内边距的浮点数，默认值由外部决定
        borderpad: float = ...,  # 边框内边距的浮点数，默认值由外部决定
        prop: dict[str, Any] | None = ...,  # 字体属性的字典，可以为空
        **kwargs  # 其他关键字参数
    ) -> None: ...

# 定义一个名为 OffsetImage 的类，它是 OffsetBox 的子类
class OffsetImage(OffsetBox):
    image: BboxImage  # 定义一个名为 image 的属性，类型为 BboxImage
    stale: bool  # 定义一个名为 stale 的属性，类型为布尔值

    # 初始化方法，接受一个类数组 arr 作为参数，还可以接受一系列可选参数
    def __init__(
        self,
        arr: ArrayLike,
        *,
        zoom: float = ...,  # 缩放比例的浮点数，默认值由外部决定
        cmap: Colormap | str | None = ...,  # 颜色映射或名称，可以为空
        norm: Normalize | str | None = ...,  # 标准化对象或名称，可以为空
        interpolation: str | None = ...,  # 插值方法的字符串，可以为空
        origin: Literal["upper", "lower"] | None = ...,  # 图像的原点位置，可以为空
        filternorm: bool = ...,  # 是否应用过滤器归一化的布尔值
        filterrad: float = ...,  # 过滤器半径的浮点数
        resample: bool = ...,  # 是否重采样的布尔值
        dpi_cor: bool = ...,  # 是否进行 DPI 校正的布尔值
        **kwargs  # 其他关键字参数
    ) -> None: ...
    # 设置数据到对象中，接受一个类数组或者空值
    def set_data(self, arr: ArrayLike | None) -> None: ...

    # 获取对象中存储的数据，返回类数组或空值
    def get_data(self) -> ArrayLike | None: ...

    # 设置缩放比例到对象中，接受一个浮点数参数
    def set_zoom(self, zoom: float) -> None: ...

    # 获取对象中存储的缩放比例，返回一个浮点数
    def get_zoom(self) -> float: ...

    # 获取对象的子元素列表，返回一个包含martist.Artist类型元素的列表
    def get_children(self) -> list[martist.Artist]: ...

    # 获取对象的偏移量，返回一个包含两个浮点数元素的元组
    def get_offset(self) -> tuple[float, float]: ...  # type: ignore[override]
class AnnotationBbox(martist.Artist, mtext._AnnotationBase):
    zorder: float  # 属性：图形对象的层级顺序
    offsetbox: OffsetBox  # 属性：偏移框对象，用于显示注释框
    arrowprops: dict[str, Any] | None  # 属性：箭头属性字典或空值，用于注释框的箭头设置
    xybox: tuple[float, float]  # 属性：偏移框的位置坐标
    boxcoords: str | tuple[str, str] | martist.Artist | Transform | Callable[
        [RendererBase], Bbox | Transform
    ]  # 属性：注释框坐标系描述或坐标转换函数
    arrow_patch: FancyArrowPatch | None  # 属性：箭头对象或空值
    patch: FancyBboxPatch  # 属性：边框补丁对象
    prop: FontProperties  # 属性：字体属性对象

    def __init__(
        self,
        offsetbox: OffsetBox,
        xy: tuple[float, float],
        xybox: tuple[float, float] | None = ...,
        xycoords: str
        | tuple[str, str]
        | martist.Artist
        | Transform
        | Callable[[RendererBase], Bbox | Transform] = ...,
        boxcoords: str
        | tuple[str, str]
        | martist.Artist
        | Transform
        | Callable[[RendererBase], Bbox | Transform]
        | None = ...,
        *,
        frameon: bool = ...,
        pad: float = ...,
        annotation_clip: bool | None = ...,
        box_alignment: tuple[float, float] = ...,
        bboxprops: dict[str, Any] | None = ...,
        arrowprops: dict[str, Any] | None = ...,
        fontsize: float | str | None = ...,
        **kwargs
    ) -> None:
        ...

    @property
    def xyann(self) -> tuple[float, float]:
        ...  # 方法：获取注释框的位置坐标

    @xyann.setter
    def xyann(self, xyann: tuple[float, float]) -> None:
        ...  # 方法：设置注释框的位置坐标

    @property
    def anncoords(
        self,
    ) -> str | tuple[str, str] | martist.Artist | Transform | Callable[
        [RendererBase], Bbox | Transform
    ]:
        ...  # 方法：获取注释框的坐标系描述或坐标转换函数

    @anncoords.setter
    def anncoords(
        self,
        coords: str
        | tuple[str, str]
        | martist.Artist
        | Transform
        | Callable[[RendererBase], Bbox | Transform],
    ) -> None:
        ...  # 方法：设置注释框的坐标系描述或坐标转换函数

    def get_children(self) -> list[martist.Artist]:
        ...  # 方法：获取注释框的子对象列表

    def set_figure(self, fig: Figure) -> None:
        ...  # 方法：设置注释框所属的图形对象

    def set_fontsize(self, s: str | float | None = ...) -> None:
        ...  # 方法：设置字体大小

    def get_fontsize(self) -> float:
        ...  # 方法：获取当前字体大小

    def get_tightbbox(self, renderer: RendererBase | None = ...) -> Bbox:
        ...  # 方法：获取注释框的紧凑边界框

    def update_positions(self, renderer: RendererBase) -> None:
        ...  # 方法：更新注释框的位置信息

class DraggableBase:
    ref_artist: martist.Artist  # 属性：参考图形对象
    got_artist: bool  # 属性：是否有有效的图形对象
    mouse_x: int  # 属性：鼠标的 X 坐标
    mouse_y: int  # 属性：鼠标的 Y 坐标
    background: Any  # 属性：背景对象

    @property
    def canvas(self) -> FigureCanvasBase:
        ...  # 方法：获取画布对象

    @property
    def cids(self) -> list[int]:
        ...  # 方法：获取回调函数 ID 列表

    def __init__(self, ref_artist: martist.Artist, use_blit: bool = ...) -> None:
        ...  # 方法：初始化拖动基础类

    def on_motion(self, evt: Event) -> None:
        ...  # 方法：处理鼠标移动事件

    def on_pick(self, evt: Event) -> None:
        ...  # 方法：处理鼠标选取事件

    def on_release(self, event: Event) -> None:
        ...  # 方法：处理鼠标释放事件

    def disconnect(self) -> None:
        ...  # 方法：断开事件处理器的连接

    def save_offset(self) -> None:
        ...  # 方法：保存偏移量信息

    def update_offset(self, dx: float, dy: float) -> None:
        ...  # 方法：更新偏移量信息

    def finalize_offset(self) -> None:
        ...  # 方法：完成偏移量的最终设置

class DraggableOffsetBox(DraggableBase):
    offsetbox: OffsetBox  # 属性：偏移框对象

    def __init__(
        self, ref_artist: martist.Artist, offsetbox: OffsetBox, use_blit: bool = ...
    ) -> None:
        ...  # 方法：初始化可拖动的偏移框对象

    def save_offset(self) -> None:
        ...  # 方法：保存偏移量信息
    # 更新对象在画布中的偏移量，根据给定的水平(dx)和垂直(dy)的位移
    def update_offset(self, dx: float, dy: float) -> None:
        ...
    
    # 获取对象在画布中的位置坐标，返回一个包含水平和垂直位置的元组
    def get_loc_in_canvas(self) -> tuple[float, float]:
        ...
# 定义一个可拖动的注释类，继承自DraggableBase
class DraggableAnnotation(DraggableBase):
    # 类型注解，表示annotation属性是一个mtext.Annotation对象
    annotation: mtext.Annotation
    
    # 构造函数，初始化可拖动注释对象
    def __init__(self, annotation: mtext.Annotation, use_blit: bool = ...) -> None:
        # 调用父类的构造函数进行初始化
        ...
    
    # 保存偏移量的方法，用于记录注释的偏移信息
    def save_offset(self) -> None:
        # 实现保存偏移量的逻辑
        ...
    
    # 更新偏移量的方法，用于更新注释的偏移位置
    def update_offset(self, dx: float, dy: float) -> None:
        # 根据传入的dx和dy更新注释的偏移量
        ...
```