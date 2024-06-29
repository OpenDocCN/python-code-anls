# `D:\src\scipysrc\matplotlib\lib\matplotlib\contour.pyi`

```py
# 导入 matplotlib 库中需要的模块和类
import matplotlib.cm as cm
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import Collection, PathCollection
from matplotlib.colors import Colormap, Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.transforms import Transform, TransformedPatchPath, TransformedPath
from matplotlib.ticker import Locator, Formatter

# 导入 numpy 库中需要的类型定义和模块
from numpy.typing import ArrayLike
import numpy as np
# 导入 collections.abc 库中需要的类型定义
from collections.abc import Callable, Iterable, Sequence
# 导入 typing 库中需要的类型定义
from typing import Literal
# 导入本地的 typing 模块中的 ColorType 类型定义
from .typing import ColorType

# 定义 ContourLabeler 类
class ContourLabeler:
    # 类的属性定义
    labelFmt: str | Formatter | Callable[[float], str] | dict[float, str]
    labelManual: bool | Iterable[tuple[float, float]]
    rightside_up: bool
    labelLevelList: list[float]
    labelIndiceList: list[int]
    labelMappable: cm.ScalarMappable
    labelCValueList: list[ColorType]
    labelXYs: list[tuple[float, float]]
    
    # 方法定义：添加等高线标签
    def clabel(
        self,
        levels: ArrayLike | None = ...,
        *,
        fontsize: str | float | None = ...,
        inline: bool = ...,
        inline_spacing: float = ...,
        fmt: str | Formatter | Callable[[float], str] | dict[float, str] | None = ...,
        colors: ColorType | Sequence[ColorType] | None = ...,
        use_clabeltext: bool = ...,
        manual: bool | Iterable[tuple[float, float]] = ...,
        rightside_up: bool = ...,
        zorder: float | None = ...
    ) -> list[Text]:
        ...

    # 方法定义：打印标签
    def print_label(self, linecontour: ArrayLike, labelwidth: float) -> bool:
        ...

    # 方法定义：检查标签是否过于接近
    def too_close(self, x: float, y: float, lw: float) -> bool:
        ...

    # 方法定义：获取标签文本
    def get_text(
        self,
        lev: float,
        fmt: str | Formatter | Callable[[float], str] | dict[float, str],
    ) -> str:
        ...

    # 方法定义：定位标签位置
    def locate_label(
        self, linecontour: ArrayLike, labelwidth: float
    ) -> tuple[float, float, float]:
        ...

    # 方法定义：计算标签的旋转角度和内联选项
    def calc_label_rot_and_inline(
        self,
        slc: ArrayLike,
        ind: int,
        lw: float,
        lc: ArrayLike | None = ...,
        spacing: int = ...,
    ) -> tuple[float, list[ArrayLike]]:
        ...

    # 方法定义：添加标签
    def add_label(
        self, x: float, y: float, rotation: float, lev: float, cvalue: ColorType
    ) -> None:
        ...

    # 方法定义：使用 clabeltext 添加标签
    def add_label_clabeltext(
        self, x: float, y: float, rotation: float, lev: float, cvalue: ColorType
    ) -> None:
        ...

    # 方法定义：在附近添加标签
    def add_label_near(
        self,
        x: float,
        y: float,
        inline: bool = ...,
        inline_spacing: int = ...,
        transform: Transform | Literal[False] | None = ...,
    ) -> None:
        ...

    # 方法定义：弹出标签
    def pop_label(self, index: int = ...) -> None:
        ...

    # 方法定义：标签显示控制
    def labels(self, inline: bool, inline_spacing: int) -> None:
        ...

    # 方法定义：移除标签
    def remove(self) -> None:
        ...

# 定义 ContourSet 类，继承自 ContourLabeler 和 Collection
class ContourSet(ContourLabeler, Collection):
    # 类的属性定义
    axes: Axes
    levels: Iterable[float]
    filled: bool
    linewidths: float | ArrayLike | None
    hatches: Iterable[str | None]
    # 定义一个属性 `origin`，可以是 "upper", "lower", "image" 或者 None 类型
    origin: Literal["upper", "lower", "image"] | None
    # 定义一个属性 `extent`，是一个包含四个浮点数的元组或者 None 类型
    extent: tuple[float, float, float, float] | None
    # 定义一个属性 `colors`，可以是 ColorType 类型或者 ColorType 的序列
    colors: ColorType | Sequence[ColorType]
    # 定义一个属性 `extend`，只能是 "neither", "both", "min", "max" 四个字符串字面量中的一个
    extend: Literal["neither", "both", "min", "max"]
    # 定义一个整数属性 `nchunk`
    nchunk: int
    # 定义一个属性 `locator`，可以是 Locator 类型或者 None 类型
    locator: Locator | None
    # 定义一个布尔属性 `logscale`
    logscale: bool
    # 定义一个属性 `negative_linestyles`，可以是 None 或者包含 "solid", "dashed", "dashdot", "dotted" 字符串字面量的可迭代对象
    negative_linestyles: None | Literal[
        "solid", "dashed", "dashdot", "dotted"
    ] | Iterable[Literal["solid", "dashed", "dashdot", "dotted"]]
    # 定义一个属性 `clip_path`，可以是 Patch、Path、TransformedPath、TransformedPatchPath 或者 None 类型
    clip_path: Patch | Path | TransformedPath | TransformedPatchPath | None
    # 定义一个属性 `labelTexts`，是一个 Text 对象的列表
    labelTexts: list[Text]
    # 定义一个属性 `labelCValues`，是一个 ColorType 对象的列表

    labelCValues: list[ColorType]

    @property
    # 定义一个只读属性 `tcolors`，返回一个元组列表，每个元组包含四个浮点数
    def tcolors(self) -> list[tuple[tuple[float, float, float, float]]]: ...

    # 只对未填充的情况适用
    @property
    # 定义一个只读属性 `tlinewidths`，返回一个浮点数元组的列表
    def tlinewidths(self) -> list[tuple[float]]: ...

    @property
    # 定义一个只读属性 `allkinds`，返回一个列表，每个元素是一个列表，包含 None 或者 np.ndarray 对象
    def allkinds(self) -> list[list[np.ndarray | None]]: ...

    @property
    # 定义一个只读属性 `allsegs`，返回一个列表，每个元素是一个列表，包含 np.ndarray 对象
    def allsegs(self) -> list[list[np.ndarray]]: ...

    @property
    # 定义一个只读属性 `alpha`，可以是浮点数或者 None
    def alpha(self) -> float | None: ...

    @property
    # 定义一个只读属性 `antialiased`，是一个布尔值
    def antialiased(self) -> bool: ...

    @antialiased.setter
    # 定义 `antialiased` 属性的 setter 方法，接受一个布尔值或者布尔值的序列作为参数，不返回任何值
    def antialiased(self, aa: bool | Sequence[bool]) -> None: ...

    @property
    # 定义一个只读属性 `collections`，返回一个 PathCollection 对象的列表
    def collections(self) -> list[PathCollection]: ...

    @property
    # 定义一个只读属性 `linestyles`，可以是 None，或者 "solid", "dashed", "dashdot", "dotted" 字符串字面量的可迭代对象
    def linestyles(self) -> (
        None |
        Literal["solid", "dashed", "dashdot", "dotted"] |
        Iterable[Literal["solid", "dashed", "dashdot", "dotted"]]
    ): ...

    # 定义一个构造函数 `__init__`，接受多个参数，初始化一个对象，没有返回值
    def __init__(
        self,
        ax: Axes,
        *args,
        levels: Iterable[float] | None = ...,
        filled: bool = ...,
        linewidths: float | ArrayLike | None = ...,
        linestyles: Literal["solid", "dashed", "dashdot", "dotted"]
        | Iterable[Literal["solid", "dashed", "dashdot", "dotted"]]
        | None = ...,
        hatches: Iterable[str | None] = ...,
        alpha: float | None = ...,
        origin: Literal["upper", "lower", "image"] | None = ...,
        extent: tuple[float, float, float, float] | None = ...,
        cmap: str | Colormap | None = ...,
        colors: ColorType | Sequence[ColorType] | None = ...,
        norm: str | Normalize | None = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        extend: Literal["neither", "both", "min", "max"] = ...,
        antialiased: bool | None = ...,
        nchunk: int = ...,
        locator: Locator | None = ...,
        transform: Transform | None = ...,
        negative_linestyles: Literal["solid", "dashed", "dashdot", "dotted"]
        | Iterable[Literal["solid", "dashed", "dashdot", "dotted"]]
        | None = ...,
        clip_path: Patch | Path | TransformedPath | TransformedPatchPath | None = ...,
        **kwargs
    ) -> None: ...

    # 定义一个方法 `legend_elements`，返回一个元组，包含 Artist 对象列表和字符串列表
    def legend_elements(
        self, variable_name: str = ..., str_format: Callable[[float], str] = ...
    ) -> tuple[list[Artist], list[str]]: ...

    # 定义一个方法 `find_nearest_contour`，接受 x, y 两个浮点数参数，返回一个元组
    # 元组包含四个整数和三个浮点数，用于表示最近的等高线信息
    def find_nearest_contour(
        self, x: float, y: float, indices: Iterable[int] | None = ..., pixel: bool = ...
    ) -> tuple[int, int, int, float, float, float]: ...
# 定义一个名为 QuadContourSet 的类，该类继承自 ContourSet 类
class QuadContourSet(ContourSet):
    # 省略部分代码，详细实现未显示在这里
    ...
```