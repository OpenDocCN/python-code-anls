# `D:\src\scipysrc\matplotlib\lib\matplotlib\spines.pyi`

```
# 导入必要的模块和类型
from collections.abc import Callable, Iterator, MutableMapping
from typing import Any, Literal, TypeVar, overload

import matplotlib.patches as mpatches
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.path import Path
from matplotlib.transforms import Transform
from matplotlib.typing import ColorType

# 定义 Spine 类，继承自 mpatches.Patch 类
class Spine(mpatches.Patch):
    axes: Axes  # 拥有这个 Spine 的 Axes 对象
    spine_type: str  # Spine 的类型，如 "left", "right", "bottom", "top"
    axis: Axis | None  # 关联的 Axis 对象，可能为空

    # 初始化方法，接受 Axes 对象、Spine 类型和路径作为参数
    def __init__(self, axes: Axes, spine_type: str, path: Path, **kwargs) -> None: ...

    # 设置 Spine 为弧形，接受中心点、半径和起始角度、结束角度作为参数
    def set_patch_arc(
        self, center: tuple[float, float], radius: float, theta1: float, theta2: float
    ) -> None: ...

    # 设置 Spine 为圆形，接受中心点和半径作为参数
    def set_patch_circle(self, center: tuple[float, float], radius: float) -> None: ...

    # 设置 Spine 为直线
    def set_patch_line(self) -> None: ...

    # 获取 Spine Patch 的变换对象
    def get_patch_transform(self) -> Transform: ...

    # 获取 Spine 的路径对象
    def get_path(self) -> Path: ...

    # 注册关联的 Axis 对象
    def register_axis(self, axis: Axis) -> None: ...

    # 清除 Spine
    def clear(self) -> None: ...

    # 设置 Spine 的位置，可以是 "center"、"zero" 或者具体的位置和数值
    def set_position(
        self,
        position: Literal["center", "zero"]
        | tuple[Literal["outward", "axes", "data"], float],
    ) -> None: ...

    # 获取 Spine 的位置，可以是 "center"、"zero" 或者具体的位置和数值
    def get_position(
        self,
    ) -> Literal["center", "zero"] | tuple[
        Literal["outward", "axes", "data"], float
    ]: ...

    # 获取 Spine 的变换对象
    def get_spine_transform(self) -> Transform: ...

    # 设置 Spine 的边界，接受下限和上限作为参数
    def set_bounds(self, low: float | None = ..., high: float | None = ...) -> None: ...

    # 获取 Spine 的边界，返回下限和上限的元组
    def get_bounds(self) -> tuple[float, float]: ...

    # 类型变量 _T 为 Spine 类型，用于类方法的返回类型声明
    _T = TypeVar("_T", bound=Spine)

    # 创建线性 Spine 的类方法，接受 Axes 对象、Spine 类型和其他关键字参数
    @classmethod
    def linear_spine(
        cls: type[_T],
        axes: Axes,
        spine_type: Literal["left", "right", "bottom", "top"],
        **kwargs
    ) -> _T: ...

    # 创建弧形 Spine 的类方法，接受 Axes 对象、Spine 类型、中心点、半径、起始角度、结束角度和其他关键字参数
    @classmethod
    def arc_spine(
        cls: type[_T],
        axes: Axes,
        spine_type: Literal["left", "right", "bottom", "top"],
        center: tuple[float, float],
        radius: float,
        theta1: float,
        theta2: float,
        **kwargs
    ) -> _T: ...

    # 创建圆形 Spine 的类方法，接受 Axes 对象、中心点、半径和其他关键字参数
    @classmethod
    def circular_spine(
        cls: type[_T], axes: Axes, center: tuple[float, float], radius: float, **kwargs
    ) -> _T: ...

    # 设置 Spine 的颜色，接受颜色值或 None 作为参数
    def set_color(self, c: ColorType | None) -> None: ...

# 定义 SpinesProxy 类，代理 Spine 字典中的多个 Spine 对象
class SpinesProxy:
    # 初始化方法，接受 Spine 字典作为参数
    def __init__(self, spine_dict: dict[str, Spine]) -> None: ...

    # 获取属性的特殊方法，允许通过属性名调用 Spine 对象的方法
    def __getattr__(self, name: str) -> Callable[..., None]: ...

    # 返回对象的属性列表的特殊方法
    def __dir__(self) -> list[str]: ...

# 定义 Spines 类，继承自 MutableMapping 类，表示多个 Spine 对象的集合
class Spines(MutableMapping[str, Spine]):
    # 初始化方法，接受多个 Spine 对象作为参数
    def __init__(self, **kwargs: Spine) -> None: ...

    # 从字典创建 Spines 对象的类方法，接受 Spine 字典作为参数
    @classmethod
    def from_dict(cls, d: dict[str, Spine]) -> Spines: ...

    # 获取属性的特殊方法，允许通过属性名获取对应的 Spine 对象
    def __getattr__(self, name: str) -> Spine: ...

    # 获取指定键的值的重载方法，接受字符串键作为参数，返回对应的 Spine 对象
    @overload
    def __getitem__(self, key: str) -> Spine: ...

    # 获取指定多个键的值的重载方法，接受字符串键列表作为参数，返回 SpinesProxy 对象
    @overload
    def __getitem__(self, key: list[str]) -> SpinesProxy: ...

    # 获取指定切片范围的值的重载方法，接受切片对象作为参数，返回 SpinesProxy 对象
    @overload
    def __getitem__(self, key: slice) -> SpinesProxy: ...

    # 设置指定键的值的方法，接受字符串键和对应的 Spine 对象作为参数
    def __setitem__(self, key: str, value: Spine) -> None: ...

    # 删除指定键的值的方法，接受字符串键作为参数
    def __delitem__(self, key: str) -> None: ...

    # 返回迭代器的特殊方法，迭代 Spines 对象的键
    def __iter__(self) -> Iterator[str]: ...

    # 返回长度的特殊方法，返回 Spines 对象包含的 Spine 数量
    def __len__(self) -> int: ...
```