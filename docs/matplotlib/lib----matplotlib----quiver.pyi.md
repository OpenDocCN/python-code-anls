# `D:\src\scipysrc\matplotlib\lib\matplotlib\quiver.pyi`

```
# 导入必要的模块和类
import matplotlib.artist as martist
import matplotlib.collections as mcollections
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.transforms import Transform, Bbox

# 导入 numpy 相关类和类型
import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Sequence
from typing import Any, Literal, overload
from matplotlib.typing import ColorType

# 定义 QuiverKey 类，继承自 martist.Artist 类
class QuiverKey(martist.Artist):
    # 定义各种属性，包括对齐、标签位置、颜色等
    halign: dict[Literal["N", "S", "E", "W"], Literal["left", "center", "right"]]
    valign: dict[Literal["N", "S", "E", "W"], Literal["top", "center", "bottom"]]
    pivot: dict[Literal["N", "S", "E", "W"], Literal["middle", "tip", "tail"]]
    Q: Quiver
    X: float
    Y: float
    U: float
    angle: float
    coord: Literal["axes", "figure", "data", "inches"]
    color: ColorType | None
    label: str
    labelpos: Literal["N", "S", "E", "W"]
    labelcolor: ColorType | None
    fontproperties: dict[str, Any]
    kw: dict[str, Any]
    text: Text
    zorder: float
    
    # 初始化方法，接收一系列参数
    def __init__(
        self,
        Q: Quiver,
        X: float,
        Y: float,
        U: float,
        label: str,
        *,
        angle: float = ...,
        coordinates: Literal["axes", "figure", "data", "inches"] = ...,
        color: ColorType | None = ...,
        labelsep: float = ...,
        labelpos: Literal["N", "S", "E", "W"] = ...,
        labelcolor: ColorType | None = ...,
        fontproperties: dict[str, Any] | None = ...,
        **kwargs
    ) -> None:
        ...

    # 属性方法，返回标签的分隔距离
    @property
    def labelsep(self) -> float:
        ...

    # 方法，设置图形对象
    def set_figure(self, fig: Figure) -> None:
        ...

# 定义 Quiver 类，继承自 mcollections.PolyCollection 类
class Quiver(mcollections.PolyCollection):
    # 定义各种属性，包括坐标、向量、比例、宽度等
    X: ArrayLike
    Y: ArrayLike
    XY: ArrayLike
    U: ArrayLike
    V: ArrayLike
    Umask: ArrayLike
    N: int
    scale: float | None
    headwidth: float
    headlength: float
    headaxislength: float
    minshaft: float
    minlength: float
    units: Literal["width", "height", "dots", "inches", "x", "y", "xy"]
    scale_units: Literal["width", "height", "dots", "inches", "x", "y", "xy"] | None
    angles: Literal["uv", "xy"] | ArrayLike
    width: float | None
    pivot: Literal["tail", "middle", "tip"]
    transform: Transform
    polykw: dict[str, Any]

    # 构造方法的重载，接收多种参数
    @overload
    def __init__(
        self,
        ax: Axes,
        U: ArrayLike,
        V: ArrayLike,
        C: ArrayLike = ...,
        *,
        scale: float | None = ...,
        headwidth: float = ...,
        headlength: float = ...,
        headaxislength: float = ...,
        minshaft: float = ...,
        minlength: float = ...,
        units: Literal["width", "height", "dots", "inches", "x", "y", "xy"] = ...,
        scale_units: Literal["width", "height", "dots", "inches", "x", "y", "xy"]
        | None = ...,
        angles: Literal["uv", "xy"] | ArrayLike = ...,
        width: float | None = ...,
        color: ColorType | Sequence[ColorType] = ...,
        pivot: Literal["tail", "mid", "middle", "tip"] = ...,
        **kwargs
    ):
        ...
    # 方法声明，初始化一个箭头对象在给定的轴上，用于表示矢量场
    @overload
    def __init__(
        self,
        ax: Axes,  # 参数ax：指定的Axes对象，箭头将在其上绘制
        X: ArrayLike,  # 参数X：表示箭头起点的x坐标，可以是数组形式
        Y: ArrayLike,  # 参数Y：表示箭头起点的y坐标，可以是数组形式
        U: ArrayLike,  # 参数U：表示箭头x方向的分量，可以是数组形式
        V: ArrayLike,  # 参数V：表示箭头y方向的分量，可以是数组形式
        C: ArrayLike = ...,  # 参数C（可选）：表示箭头颜色或标量值，可以是数组形式，默认为省略值
        *,
        scale: float | None = ...,  # 参数scale（可选）：箭头大小的比例因子或None，默认为省略值
        headwidth: float = ...,  # 参数headwidth（可选）：箭头头部宽度，默认为省略值
        headlength: float = ...,  # 参数headlength（可选）：箭头头部长度，默认为省略值
        headaxislength: float = ...,  # 参数headaxislength（可选）：箭头头部轴长，默认为省略值
        minshaft: float = ...,  # 参数minshaft（可选）：箭头最小的箭杆长度，默认为省略值
        minlength: float = ...,  # 参数minlength（可选）：箭头最小的长度，默认为省略值
        units: Literal["width", "height", "dots", "inches", "x", "y", "xy"] = ...,  # 参数units（可选）：长度单位，默认为省略值
        scale_units: Literal["width", "height", "dots", "inches", "x", "y", "xy"] | None = ...,  # 参数scale_units（可选）：比例单位，默认为省略值
        angles: Literal["uv", "xy"] | ArrayLike = ...,  # 参数angles（可选）：箭头方向角度或数组形式，默认为省略值
        width: float | None = ...,  # 参数width（可选）：箭头线宽度或None，默认为省略值
        color: ColorType | Sequence[ColorType] = ...,  # 参数color（可选）：箭头颜色或颜色序列，默认为省略值
        pivot: Literal["tail", "mid", "middle", "tip"] = ...,  # 参数pivot（可选）：箭头旋转的中心点，默认为省略值
        **kwargs  # 其他关键字参数
    ) -> None:  # 返回类型为None
        ...
    
    # 获取数据限制，返回数据的包围框
    def get_datalim(self, transData: Transform) -> Bbox:
        ...
    
    # 设置箭头对象的U、V和C值
    def set_UVC(
        self, U: ArrayLike, V: ArrayLike, C: ArrayLike | None = ...
    ) -> None:
        ...
# 定义名为 Barbs 的类，继承自 mcollections.PolyCollection
class Barbs(mcollections.PolyCollection):
    # 类变量：存储每个箭头的大小的字典，键为字符串，值为浮点数
    sizes: dict[str, float]
    # 类变量：指示是否填充空箭头的布尔值
    fill_empty: bool
    # 类变量：存储箭羽增量的字典，键为字符串，值为浮点数
    barb_increments: dict[str, float]
    # 类变量：指示是否对箭头进行四舍五入的布尔值
    rounding: bool
    # 类变量：存储箭头翻转信息的 NumPy 数组
    flip: np.ndarray
    # 实例变量：存储箭头位置的数组或类数组对象
    x: ArrayLike
    # 实例变量：存储箭头位置的数组或类数组对象
    y: ArrayLike
    # 实例变量：存储箭头方向的数组或类数组对象
    u: ArrayLike
    # 实例变量：存储箭头方向的数组或类数组对象
    v: ArrayLike

    # 构造函数重载1：初始化 Barbs 对象，使用箭头位置和方向的数组及其它可选参数
    @overload
    def __init__(
        self,
        ax: Axes,
        U: ArrayLike,
        V: ArrayLike,
        C: ArrayLike = ...,
        *,
        pivot: str = ...,
        length: int = ...,
        barbcolor: ColorType | Sequence[ColorType] | None = ...,
        flagcolor: ColorType | Sequence[ColorType] | None = ...,
        sizes: dict[str, float] | None = ...,
        fill_empty: bool = ...,
        barb_increments: dict[str, float] | None = ...,
        rounding: bool = ...,
        flip_barb: bool | ArrayLike = ...,
        **kwargs
    ) -> None: ...

    # 构造函数重载2：初始化 Barbs 对象，使用箭头位置、方向和坐标的数组及其它可选参数
    @overload
    def __init__(
        self,
        ax: Axes,
        X: ArrayLike,
        Y: ArrayLike,
        U: ArrayLike,
        V: ArrayLike,
        C: ArrayLike = ...,
        *,
        pivot: str = ...,
        length: int = ...,
        barbcolor: ColorType | Sequence[ColorType] | None = ...,
        flagcolor: ColorType | Sequence[ColorType] | None = ...,
        sizes: dict[str, float] | None = ...,
        fill_empty: bool = ...,
        barb_increments: dict[str, float] | None = ...,
        rounding: bool = ...,
        flip_barb: bool | ArrayLike = ...,
        **kwargs
    ) -> None: ...

    # 方法：设置箭头的方向和长度
    def set_UVC(
        self, U: ArrayLike, V: ArrayLike, C: ArrayLike | None = ...
    ) -> None: ...

    # 方法：设置箭头的偏移量（位置）
    def set_offsets(self, xy: ArrayLike) -> None: ...
```