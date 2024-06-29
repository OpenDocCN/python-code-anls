# `D:\src\scipysrc\matplotlib\lib\matplotlib\artist.pyi`

```py
from .axes._base import _AxesBase
from .backend_bases import RendererBase, MouseEvent
from .figure import Figure, SubFigure
from .path import Path
from .patches import Patch
from .patheffects import AbstractPathEffect
from .transforms import (
    BboxBase,
    Bbox,
    Transform,
    TransformedPatchPath,
    TransformedPath,
)

import numpy as np

from collections.abc import Callable, Iterable
from typing import Any, NamedTuple, TextIO, overload, TypeVar
from numpy.typing import ArrayLike

_T_Artist = TypeVar("_T_Artist", bound=Artist)

# 函数定义，未提供具体实现
def allow_rasterization(draw): ...

# 命名元组定义，表示包含 x 和 y 属性的数组样式数据对
class _XYPair(NamedTuple):
    x: ArrayLike
    y: ArrayLike

# 空类，未定义任何成员或方法
class _Unset: ...

# 艺术家类，用于绘图元素的基类
class Artist:
    # 属性：图层顺序
    zorder: float
    # 属性：陈旧状态回调函数
    stale_callback: Callable[[Artist, bool], None] | None
    # 属性：所属的图形对象，可以是 Figure 或 SubFigure，或者为空
    figure: Figure | SubFigure | None
    # 属性：裁剪框，可以是 BboxBase 或空
    clipbox: BboxBase | None

    # 构造函数，未提供具体实现
    def __init__(self) -> None: ...

    # 方法：移除当前艺术家对象
    def remove(self) -> None: ...

    # 方法：检查是否使用了单位
    def have_units(self) -> bool: ...

    # TODO units
    # 方法：转换 x 坐标的单位
    def convert_xunits(self, x): ...

    # 方法：转换 y 坐标的单位
    def convert_yunits(self, y): ...

    # 属性：获取所属的坐标系对象，可以是 _AxesBase 或空
    @property
    def axes(self) -> _AxesBase | None: ...

    # 属性：设置所属的坐标系对象
    @axes.setter
    def axes(self, new_axes: _AxesBase | None) -> None: ...

    # 属性：检查艺术家对象是否陈旧
    @property
    def stale(self) -> bool: ...

    # 属性：设置艺术家对象的陈旧状态
    @stale.setter
    def stale(self, val: bool) -> None: ...

    # 方法：获取艺术家对象在指定渲染器下的窗口范围
    def get_window_extent(self, renderer: RendererBase | None = ...) -> Bbox: ...

    # 方法：获取艺术家对象在指定渲染器下的紧凑边界框
    def get_tightbbox(self, renderer: RendererBase | None = ...) -> Bbox | None: ...

    # 方法：添加回调函数
    def add_callback(self, func: Callable[[Artist], Any]) -> int: ...

    # 方法：移除指定 ID 的回调函数
    def remove_callback(self, oid: int) -> None: ...

    # 方法：标记属性变化
    def pchanged(self) -> None: ...

    # 方法：检查是否设置了变换
    def is_transform_set(self) -> bool: ...

    # 方法：设置变换
    def set_transform(self, t: Transform | None) -> None: ...

    # 方法：获取变换对象
    def get_transform(self) -> Transform: ...

    # 方法：获取所有子艺术家对象的列表
    def get_children(self) -> list[Artist]: ...

    # TODO can these dicts be type narrowed? e.g. str keys
    # 方法：检查鼠标事件是否发生在当前艺术家对象内部，并返回布尔值和额外数据的字典
    def contains(self, mouseevent: MouseEvent) -> tuple[bool, dict[Any, Any]]: ...

    # 方法：检查当前艺术家对象是否可选取
    def pickable(self) -> bool: ...

    # 方法：处理鼠标点击事件
    def pick(self, mouseevent: MouseEvent) -> None: ...

    # 方法：设置选择器函数
    def set_picker(
        self,
        picker: None
        | bool
        | float
        | Callable[[Artist, MouseEvent], tuple[bool, dict[Any, Any]]],
    ) -> None: ...

    # 方法：获取当前选择器函数
    def get_picker(
        self,
    ) -> None | bool | float | Callable[
        [Artist, MouseEvent], tuple[bool, dict[Any, Any]]
    ]: ...

    # 方法：获取艺术家对象的 URL
    def get_url(self) -> str | None: ...

    # 方法：设置艺术家对象的 URL
    def set_url(self, url: str | None) -> None: ...

    # 方法：获取艺术家对象的组 ID
    def get_gid(self) -> str | None: ...

    # 方法：设置艺术家对象的组 ID
    def set_gid(self, gid: str | None) -> None: ...

    # 方法：获取是否启用对齐到网格的状态
    def get_snap(self) -> bool | None: ...

    # 方法：设置是否启用对齐到网格
    def set_snap(self, snap: bool | None) -> None: ...

    # 方法：获取草图参数，返回包含三个浮点数的元组或空
    def get_sketch_params(self) -> tuple[float, float, float] | None: ...

    # 方法：设置草图参数
    def set_sketch_params(
        self,
        scale: float | None = ...,
        length: float | None = ...,
        randomness: float | None = ...,
    ) -> None: ...

    # 方法：设置路径效果
    def set_path_effects(self, path_effects: list[AbstractPathEffect]) -> None: ...
    # 返回一个 AbstractPathEffect 对象的列表，表示该对象的路径效果
    def get_path_effects(self) -> list[AbstractPathEffect]: ...

    # 返回 Figure 对象或者 None，表示当前对象所属的图形对象
    def get_figure(self) -> Figure | None: ...

    # 设置该对象所属的 Figure 对象
    def set_figure(self, fig: Figure) -> None: ...

    # 设置剪切框的边界框，可以为 BboxBase 或者 None
    def set_clip_box(self, clipbox: BboxBase | None) -> None: ...

    # 设置剪切路径，可以为 Patch、Path、TransformedPath、TransformedPatchPath 或者 None
    # 可选参数 transform 表示变换对象，默认为 None
    def set_clip_path(
        self,
        path: Patch | Path | TransformedPath | TransformedPatchPath | None,
        transform: Transform | None = ...,
    ) -> None: ...

    # 返回当前对象的透明度，为浮点数或者 None
    def get_alpha(self) -> float | None: ...

    # 返回当前对象的可见状态，为布尔值
    def get_visible(self) -> bool: ...

    # 返回当前对象的动画状态，为布尔值
    def get_animated(self) -> bool: ...

    # 返回当前对象是否在布局中的状态，为布尔值
    def get_in_layout(self) -> bool: ...

    # 返回当前对象是否启用剪切功能，为布尔值
    def get_clip_on(self) -> bool: ...

    # 返回当前对象的剪切边界框，为 Bbox 对象或者 None
    def get_clip_box(self) -> Bbox | None: ...

    # 返回当前对象的剪切路径，为 Patch、Path、TransformedPath、TransformedPatchPath 或者 None
    def get_clip_path(
        self,
    ) -> Patch | Path | TransformedPath | TransformedPatchPath | None: ...

    # 返回当前对象的变换后剪切路径及其仿射变换，为元组，可能为 None 或者 (Path, Transform)
    def get_transformed_clip_path_and_affine(
        self,
    ) -> tuple[None, None] | tuple[Path, Transform]: ...

    # 设置当前对象的剪切功能状态
    def set_clip_on(self, b: bool) -> None: ...

    # 返回当前对象的光栅化状态，为布尔值
    def get_rasterized(self) -> bool: ...

    # 设置当前对象的光栅化状态
    def set_rasterized(self, rasterized: bool) -> None: ...

    # 返回当前对象的聚合过滤器，为 Callable 对象或者 None
    def get_agg_filter(self) -> Callable[[ArrayLike, float], tuple[np.ndarray, float, float]] | None: ...

    # 设置当前对象的聚合过滤器
    def set_agg_filter(
        self, filter_func: Callable[[ArrayLike, float], tuple[np.ndarray, float, float]] | None
    ) -> None: ...

    # 绘制当前对象，使用指定的渲染器
    def draw(self, renderer: RendererBase) -> None: ...

    # 设置当前对象的透明度
    def set_alpha(self, alpha: float | None) -> None: ...

    # 设置当前对象的可见状态
    def set_visible(self, b: bool) -> None: ...

    # 设置当前对象的动画状态
    def set_animated(self, b: bool) -> None: ...

    # 设置当前对象是否在布局中的状态
    def set_in_layout(self, in_layout: bool) -> None: ...

    # 返回当前对象的标签，可以是任意对象
    def get_label(self) -> object: ...

    # 设置当前对象的标签
    def set_label(self, s: object) -> None: ...

    # 返回当前对象的显示顺序值，为浮点数
    def get_zorder(self) -> float: ...

    # 设置当前对象的显示顺序值
    def set_zorder(self, level: float) -> None: ...

    # 返回当前对象的粘性边界属性，为 _XYPair 对象
    @property
    def sticky_edges(self) -> _XYPair: ...

    # 从另一个 Artist 对象更新当前对象的属性
    def update_from(self, other: Artist) -> None: ...

    # 返回当前对象的属性字典
    def properties(self) -> dict[str, Any]: ...

    # 更新当前对象的属性
    def update(self, props: dict[str, Any]) -> list[Any]: ...

    # 内部方法，更新当前对象的属性，参数类型为 Any
    def _internal_update(self, kwargs: Any) -> list[Any]: ...

    # 使用关键字参数设置当前对象的属性
    def set(self, **kwargs: Any) -> list[Any]: ...

    # 重载方法：根据条件匹配获取符合条件的 Artist 对象列表
    @overload
    def findobj(
        self,
        match: None | Callable[[Artist], bool] = ...,
        include_self: bool = ...,
    ) -> list[Artist]: ...

    # 重载方法：根据类型匹配获取符合条件的 Artist 对象列表
    @overload
    def findobj(
        self,
        match: type[_T_Artist],
        include_self: bool = ...,
    ) -> list[_T_Artist]: ...

    # 根据鼠标事件获取当前对象的光标数据
    def get_cursor_data(self, event: MouseEvent) -> Any: ...

    # 格式化光标数据为字符串形式
    def format_cursor_data(self, data: Any) -> str: ...

    # 返回当前对象是否处于鼠标悬停状态
    def get_mouseover(self) -> bool: ...

    # 设置当前对象的鼠标悬停状态
    def set_mouseover(self, mouseover: bool) -> None: ...

    # 属性：返回当前对象的鼠标悬停状态
    @property
    def mouseover(self) -> bool: ...

    # 属性设置方法：设置当前对象的鼠标悬停状态
    @mouseover.setter
    def mouseover(self, mouseover: bool) -> None: ...
class ArtistInspector:
    # oorig 属性可以是 Artist 类型或者 Artist 类型的类型对象
    oorig: Artist | type[Artist]
    # o 属性是 Artist 类型的类型对象
    o: type[Artist]
    # aliasd 属性是一个字典，键为字符串，值为字符串集合
    aliasd: dict[str, set[str]]

    # 初始化方法，接受一个 Artist 类型、Artist 类型的类型对象或者可迭代对象作为参数
    def __init__(
        self, o: Artist | type[Artist] | Iterable[Artist | type[Artist]]
    ) -> None: ...
    
    # 获取别名的方法，返回一个字典，键为字符串，值为字符串集合
    def get_aliases(self) -> dict[str, set[str]]: ...

    # 获取属性的有效值的方法，接受属性名作为参数，返回字符串或 None
    def get_valid_values(self, attr: str) -> str | None: ...

    # 获取设置器的方法，返回一个字符串列表
    def get_setters(self) -> list[str]: ...

    # 静态方法，接受可调用对象作为参数，返回该对象的参数数量
    @staticmethod
    def number_of_parameters(func: Callable) -> int: ...

    # 静态方法，接受可调用对象作为参数，判断该对象是否为别名
    @staticmethod
    def is_alias(method: Callable) -> bool: ...

    # 获取别名后的名称的方法，接受一个字符串作为参数，返回一个字符串
    def aliased_name(self, s: str) -> str: ...

    # 获取带有目标别名的名称的方法，接受两个字符串作为参数，返回一个字符串
    def aliased_name_rest(self, s: str, target: str) -> str: ...

    # 重载方法，打印设置器的方法，prop 参数可为 None 或者字符串，leadingspace 参数为整数
    @overload
    def pprint_setters(
        self, prop: None = ..., leadingspace: int = ...
    ) -> list[str]: ...

    # 重载方法，打印设置器的方法，prop 参数为字符串，leadingspace 参数为整数
    @overload
    def pprint_setters(self, prop: str, leadingspace: int = ...) -> str: ...

    # 重载方法，打印剩余设置器的方法，prop 参数可为 None 或者字符串，leadingspace 参数为整数
    @overload
    def pprint_setters_rest(
        self, prop: None = ..., leadingspace: int = ...
    ) -> list[str]: ...

    # 重载方法，打印剩余设置器的方法，prop 参数为字符串，leadingspace 参数为整数
    @overload
    def pprint_setters_rest(self, prop: str, leadingspace: int = ...) -> str: ...

    # 获取属性的方法，返回一个包含所有属性的字典
    def properties(self) -> dict[str, Any]: ...

    # 打印获取器的方法，返回一个字符串列表
    def pprint_getters(self) -> list[str]: ...

# 获取对象的属性值的方法，接受一个 Artist 对象和可选的属性名作为参数，返回任意类型
def getp(obj: Artist, property: str | None = ...) -> Any: ...

# getp 方法的别名
get = getp

# 设置对象的属性值的方法，接受一个 Artist 对象、任意数量的位置参数和关键字参数，返回一个列表或 None
def setp(obj: Artist, *args, file: TextIO | None = ..., **kwargs) -> list[Any] | None: ...

# 文档生成方法，接受一个 Artist 类型、Artist 类型的类型对象或者可迭代对象作为参数，返回一个字符串
def kwdoc(artist: Artist | type[Artist] | Iterable[Artist | type[Artist]]) -> str: ...
```