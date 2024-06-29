# `D:\src\scipysrc\matplotlib\lib\matplotlib\text.pyi`

```py
# 导入所需模块和类
from .artist import Artist
from .backend_bases import RendererBase
from .font_manager import FontProperties
from .offsetbox import DraggableAnnotation
from .path import Path
from .patches import FancyArrowPatch, FancyBboxPatch
from .textpath import (  # noqa: reexported API
    TextPath as TextPath,
    TextToPath as TextToPath,
)
from .transforms import (
    Bbox,
    BboxBase,
    Transform,
)

from collections.abc import Callable, Iterable
from typing import Any, Literal
from .typing import ColorType

# 定义一个名为Text的类，继承自Artist类
class Text(Artist):
    zorder: float

    # 初始化方法，接收多个参数来配置文本属性
    def __init__(
        self,
        x: float = ...,
        y: float = ...,
        text: Any = ...,
        *,
        color: ColorType | None = ...,
        verticalalignment: Literal[
            "bottom", "baseline", "center", "center_baseline", "top"
        ] = ...,
        horizontalalignment: Literal["left", "center", "right"] = ...,
        multialignment: Literal["left", "center", "right"] | None = ...,
        fontproperties: str | Path | FontProperties | None = ...,
        rotation: float | Literal["vertical", "horizontal"] | None = ...,
        linespacing: float | None = ...,
        rotation_mode: Literal["default", "anchor"] | None = ...,
        usetex: bool | None = ...,
        wrap: bool = ...,
        transform_rotates_text: bool = ...,
        parse_math: bool | None = ...,
        antialiased: bool | None = ...,
        **kwargs
    ) -> None: ...
    
    # 更新文本对象的属性
    def update(self, kwargs: dict[str, Any]) -> list[Any]: ...
    
    # 获取文本对象的旋转角度
    def get_rotation(self) -> float: ...
    
    # 获取文本对象是否启用了文本旋转
    def get_transform_rotates_text(self) -> bool: ...
    
    # 设置文本对象的旋转模式
    def set_rotation_mode(self, m: None | Literal["default", "anchor"]) -> None: ...
    
    # 获取文本对象的旋转模式
    def get_rotation_mode(self) -> Literal["default", "anchor"]: ...
    
    # 设置文本对象的边界框属性
    def set_bbox(self, rectprops: dict[str, Any]) -> None: ...
    
    # 获取文本对象的边界框补丁对象
    def get_bbox_patch(self) -> None | FancyBboxPatch: ...
    
    # 更新文本对象边界框的位置和大小
    def update_bbox_position_size(self, renderer: RendererBase) -> None: ...
    
    # 获取文本对象是否启用了自动换行
    def get_wrap(self) -> bool: ...
    
    # 设置文本对象是否启用自动换行
    def set_wrap(self, wrap: bool) -> None: ...
    
    # 获取文本对象的颜色
    def get_color(self) -> ColorType: ...
    
    # 获取文本对象的字体属性
    def get_fontproperties(self) -> FontProperties: ...
    
    # 获取文本对象的字体族列表
    def get_fontfamily(self) -> list[str]: ...
    
    # 获取文本对象的字体名称
    def get_fontname(self) -> str: ...
    
    # 获取文本对象的字体样式
    def get_fontstyle(self) -> Literal["normal", "italic", "oblique"]: ...
    
    # 获取文本对象的字体大小
    def get_fontsize(self) -> float | str: ...
    
    # 获取文本对象的字体变体
    def get_fontvariant(self) -> Literal["normal", "small-caps"]: ...
    
    # 获取文本对象的字体粗细
    def get_fontweight(self) -> int | str: ...
    
    # 获取文本对象的字体拉伸程度
    def get_stretch(self) -> int | str: ...
    
    # 获取文本对象的水平对齐方式
    def get_horizontalalignment(self) -> Literal["left", "center", "right"]: ...
    
    # 获取文本对象的无单位位置坐标
    def get_unitless_position(self) -> tuple[float, float]: ...
    
    # 获取文本对象的位置坐标
    def get_position(self) -> tuple[float, float]: ...
    
    # 获取文本对象的文本内容
    def get_text(self) -> str: ...
    
    # 获取文本对象的垂直对齐方式
    def get_verticalalignment(
        self,
    ) -> Literal["bottom", "baseline", "center", "center_baseline", "top"]: ...
    
    # 获取文本对象在指定渲染器和 DPI 下的窗口边界
    def get_window_extent(
        self, renderer: RendererBase | None = ..., dpi: float | None = ...
    ) -> Bbox: ...
    # 设置文本框背景色
    def set_backgroundcolor(self, color: ColorType) -> None: ...

    # 设置文本颜色
    def set_color(self, color: ColorType) -> None: ...

    # 设置文本水平对齐方式为左、中或右
    def set_horizontalalignment(
        self, align: Literal["left", "center", "right"]
    ) -> None: ...

    # 设置文本多行对齐方式为左、中或右
    def set_multialignment(self, align: Literal["left", "center", "right"]) -> None: ...

    # 设置文本行间距
    def set_linespacing(self, spacing: float) -> None: ...

    # 设置文本字体家族，可以是单个字体名称或多个字体名称的迭代器
    def set_fontfamily(self, fontname: str | Iterable[str]) -> None: ...

    # 设置文本字体变体为普通或小型大写字母
    def set_fontvariant(self, variant: Literal["normal", "small-caps"]) -> None: ...

    # 设置文本字体风格为普通、斜体或倾斜
    def set_fontstyle(
        self, fontstyle: Literal["normal", "italic", "oblique"]
    ) -> None: ...

    # 设置文本字体大小，可以是浮点数或字符串
    def set_fontsize(self, fontsize: float | str) -> None: ...

    # 获取数学表达式的字体家族
    def get_math_fontfamily(self) -> str: ...

    # 设置数学表达式的字体家族
    def set_math_fontfamily(self, fontfamily: str) -> None: ...

    # 设置文本字体粗细，可以是整数或字符串
    def set_fontweight(self, weight: int | str) -> None: ...

    # 设置文本字体拉伸，可以是整数或字符串
    def set_fontstretch(self, stretch: int | str) -> None: ...

    # 设置文本框左上角位置坐标
    def set_position(self, xy: tuple[float, float]) -> None: ...

    # 设置文本框左上角 x 坐标
    def set_x(self, x: float) -> None: ...

    # 设置文本框左上角 y 坐标
    def set_y(self, y: float) -> None: ...

    # 设置文本框旋转角度（弧度）
    def set_rotation(self, s: float) -> None: ...

    # 设置文本框是否根据旋转变换文本
    def set_transform_rotates_text(self, t: bool) -> None: ...

    # 设置文本垂直对齐方式为底部、基线、中心、带基线的中心或顶部
    def set_verticalalignment(
        self, align: Literal["bottom", "baseline", "center", "center_baseline", "top"]
    ) -> None: ...

    # 设置文本内容
    def set_text(self, s: Any) -> None: ...

    # 设置文本字体属性，可以是字体属性对象、字符串、路径或 None
    def set_fontproperties(self, fp: FontProperties | str | Path | None) -> None: ...

    # 设置是否使用 TeX 渲染文本
    def set_usetex(self, usetex: bool | None) -> None: ...

    # 获取是否使用 TeX 渲染文本的当前状态
    def get_usetex(self) -> bool: ...

    # 设置是否解析数学表达式
    def set_parse_math(self, parse_math: bool) -> None: ...

    # 获取是否解析数学表达式的当前状态
    def get_parse_math(self) -> bool: ...

    # 设置文本字体名称，可以是单个字体名称或多个字体名称的迭代器
    def set_fontname(self, fontname: str | Iterable[str]) -> None: ...

    # 获取文本是否反锯齿化的当前状态
    def get_antialiased(self) -> bool: ...

    # 设置文本是否反锯齿化
    def set_antialiased(self, antialiased: bool) -> None: ...
class OffsetFrom:
    # 定义 OffsetFrom 类，用于计算相对于参考坐标的偏移量
    def __init__(
        self,
        artist: Artist | BboxBase | Transform,  # 初始化方法，接受一个艺术家、包围框或变换作为参考对象
        ref_coord: tuple[float, float],  # 参考坐标的元组，包含两个浮点数
        unit: Literal["points", "pixels"] = ...,  # 单位可以是 "points" 或 "pixels"，默认为省略值
    ) -> None: ...
    # 设置偏移量的单位，可以是 "points" 或 "pixels"
    def set_unit(self, unit: Literal["points", "pixels"]) -> None: ...
    # 获取偏移量的单位，返回值为 "points" 或 "pixels"
    def get_unit(self) -> Literal["points", "pixels"]: ...
    # 调用方法，根据渲染器返回相对于参考坐标的变换
    def __call__(self, renderer: RendererBase) -> Transform: ...

class _AnnotationBase:
    # _AnnotationBase 类作为注释的基础类，定义了基本的注释属性和方法
    xy: tuple[float, float]  # 注释的位置坐标，包含两个浮点数
    xycoords: str | tuple[str, str] | Artist | Transform | Callable[
        [RendererBase], Bbox | Transform
    ]  # 注释位置的坐标系描述，可以是字符串、元组、艺术家、变换或渲染器返回包围框或变换的可调用对象
    # 初始化方法，接受位置坐标和坐标系描述，还可选的注释剪切属性
    def __init__(
        self,
        xy,
        xycoords: str
        | tuple[str, str]
        | Artist
        | Transform
        | Callable[[RendererBase], Bbox | Transform] = ...,
        annotation_clip: bool | None = ...,
    ) -> None: ...
    # 设置注释是否剪切在图形内部的属性
    def set_annotation_clip(self, b: bool | None) -> None: ...
    # 获取注释是否剪切在图形内部的属性
    def get_annotation_clip(self) -> bool | None: ...
    # 设置注释是否可拖动的状态，可选参数包括状态和是否使用位块传输
    def draggable(
        self, state: bool | None = ..., use_blit: bool = ...
    ) -> DraggableAnnotation | None: ...

class Annotation(Text, _AnnotationBase):
    # Annotation 类继承自 Text 和 _AnnotationBase，表示带有注释文本的注释对象
    arrowprops: dict[str, Any] | None  # 箭头属性字典或空值
    arrow_patch: FancyArrowPatch | None  # 箭头补丁对象或空值
    # 初始化方法，接受注释文本、位置坐标、文本位置坐标、注释箭头属性、注释剪切属性和其他关键字参数
    def __init__(
        self,
        text: str,
        xy: tuple[float, float],
        xytext: tuple[float, float] | None = ...,
        xycoords: str
        | tuple[str, str]
        | Artist
        | Transform
        | Callable[[RendererBase], Bbox | Transform] = ...,
        textcoords: str
        | tuple[str, str]
        | Artist
        | Transform
        | Callable[[RendererBase], Bbox | Transform]
        | None = ...,
        arrowprops: dict[str, Any] | None = ...,
        annotation_clip: bool | None = ...,
        **kwargs
    ) -> None: ...
    # 返回注释位置的坐标系描述，可以是字符串、元组、艺术家、变换或渲染器返回包围框或变换的可调用对象
    @property
    def xycoords(
        self,
    ) -> str | tuple[str, str] | Artist | Transform | Callable[
        [RendererBase], Bbox | Transform
    ]: ...
    # 设置注释位置的坐标系描述
    @xycoords.setter
    def xycoords(
        self,
        xycoords: str
        | tuple[str, str]
        | Artist
        | Transform
        | Callable[[RendererBase], Bbox | Transform],
    ) -> None: ...
    # 返回注释的文本位置坐标
    @property
    def xyann(self) -> tuple[float, float]: ...
    # 设置注释的文本位置坐标
    @xyann.setter
    def xyann(self, xytext: tuple[float, float]) -> None: ...
    # 返回注释文本位置的坐标系描述
    def get_anncoords(
        self,
    ) -> str | tuple[str, str] | Artist | Transform | Callable[
        [RendererBase], Bbox | Transform
    ]: ...
    # 设置注释文本位置的坐标系描述
    def set_anncoords(
        self,
        coords: str
        | tuple[str, str]
        | Artist
        | Transform
        | Callable[[RendererBase], Bbox | Transform],
    ) -> None: ...
    # 返回注释文本位置的坐标系描述，可以是字符串、元组、艺术家、变换或渲染器返回包围框或变换的可调用对象
    @property
    def anncoords(
        self,
    ) -> str | tuple[str, str] | Artist | Transform | Callable[
        [RendererBase], Bbox | Transform
    ]: ...
    # 设置注释文本位置的坐标系描述
    @anncoords.setter
    def anncoords(
        self,
        coords: str
        | tuple[str, str]
        | Artist
        | Transform
        | Callable[[RendererBase], Bbox | Transform],
    ) -> None: ...
    # 定义方法 `update_positions`，接收一个 `renderer` 参数，并无返回值
    def update_positions(self, renderer: RendererBase) -> None: ...

    # 重写父类方法 `get_window_extent`，移除了 `dpi` 参数
    # 方法返回一个 `Bbox` 对象，可以接收一个 `renderer` 参数，默认为 `None`
    def get_window_extent(self, renderer: RendererBase | None = ...) -> Bbox: ...  # type: ignore[override]
```