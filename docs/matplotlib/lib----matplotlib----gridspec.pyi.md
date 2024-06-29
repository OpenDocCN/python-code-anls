# `D:\src\scipysrc\matplotlib\lib\matplotlib\gridspec.pyi`

```
from typing import Any, Literal, overload  # 导入类型提示相关模块

from numpy.typing import ArrayLike  # 导入数组类型相关模块
import numpy as np  # 导入 NumPy 库

from matplotlib.axes import Axes, SubplotBase  # 导入 Matplotlib 中的 Axes 和 SubplotBase 类
from matplotlib.backend_bases import RendererBase  # 导入 Matplotlib 中的 RendererBase 类
from matplotlib.figure import Figure  # 导入 Matplotlib 中的 Figure 类
from matplotlib.transforms import Bbox  # 导入 Matplotlib 中的 Bbox 类

class GridSpecBase:
    def __init__(
        self,
        nrows: int,
        ncols: int,
        height_ratios: ArrayLike | None = ...,  # 行高比例数组或空值
        width_ratios: ArrayLike | None = ...,   # 列宽比例数组或空值
    ) -> None: ...
    @property
    def nrows(self) -> int: ...  # 返回行数
    @property
    def ncols(self) -> int: ...  # 返回列数
    def get_geometry(self) -> tuple[int, int]: ...  # 返回几何形状的元组
    def get_subplot_params(self, figure: Figure | None = ...) -> SubplotParams: ...  # 获取子图参数
    def new_subplotspec(
        self, loc: tuple[int, int], rowspan: int = ..., colspan: int = ...
    ) -> SubplotSpec: ...  # 创建新的子图规范
    def set_width_ratios(self, width_ratios: ArrayLike | None) -> None: ...  # 设置列宽比例
    def get_width_ratios(self) -> ArrayLike: ...  # 获取列宽比例
    def set_height_ratios(self, height_ratios: ArrayLike | None) -> None: ...  # 设置行高比例
    def get_height_ratios(self) -> ArrayLike: ...  # 获取行高比例
    def get_grid_positions(
        self, fig: Figure
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...  # 获取网格位置数组的元组
    @staticmethod
    def _check_gridspec_exists(figure: Figure, nrows: int, ncols: int) -> GridSpec: ...  # 静态方法：检查网格规范是否存在
    def __getitem__(
        self, key: tuple[int | slice, int | slice] | slice | int
    ) -> SubplotSpec: ...  # 获取子图规范
    @overload
    def subplots(
        self,
        *,
        sharex: bool | Literal["all", "row", "col", "none"] = ...,
        sharey: bool | Literal["all", "row", "col", "none"] = ...,
        squeeze: Literal[False],
        subplot_kw: dict[str, Any] | None = ...
    ) -> np.ndarray: ...  # 重载方法：创建子图数组
    @overload
    def subplots(
        self,
        *,
        sharex: bool | Literal["all", "row", "col", "none"] = ...,
        sharey: bool | Literal["all", "row", "col", "none"] = ...,
        squeeze: Literal[True] = ...,
        subplot_kw: dict[str, Any] | None = ...
    ) -> np.ndarray | Axes: ...  # 重载方法：创建子图数组或单个子图

class GridSpec(GridSpecBase):
    left: float | None
    bottom: float | None
    right: float | None
    top: float | None
    wspace: float | None
    hspace: float | None
    figure: Figure | None
    def __init__(
        self,
        nrows: int,
        ncols: int,
        figure: Figure | None = ...,
        left: float | None = ...,
        bottom: float | None = ...,
        right: float | None = ...,
        top: float | None = ...,
        wspace: float | None = ...,
        hspace: float | None = ...,
        width_ratios: ArrayLike | None = ...,
        height_ratios: ArrayLike | None = ...,
    ) -> None: ...  # 构造函数：初始化网格规范对象
    def update(self, **kwargs: float | None) -> None: ...  # 更新方法：更新网格规范对象的属性
    def locally_modified_subplot_params(self) -> list[str]: ...  # 返回本地修改的子图参数列表
    # 定义一个方法 `tight_layout`，用于调整图表的布局以确保子图之间的紧凑排列
    def tight_layout(
        self,
        figure: Figure,  # 参数 `figure` 表示要调整布局的图表对象
        renderer: RendererBase | None = ...,  # 参数 `renderer` 表示渲染器对象，可选，默认为 None
        pad: float = ...,  # 参数 `pad` 表示全局的填充量，用于控制子图与图表边缘的间距
        h_pad: float | None = ...,  # 参数 `h_pad` 表示水平方向的填充量，用于控制子图之间的水平间距，可以为 None
        w_pad: float | None = ...,  # 参数 `w_pad` 表示垂直方向的填充量，用于控制子图之间的垂直间距，可以为 None
        rect: tuple[float, float, float, float] | None = ...,  # 参数 `rect` 表示图表的位置和大小，可以为 None
    ) -> None:
        # 这里是方法的占位符，实际的方法体内容需要根据具体实现来补充或理解
        pass
# 定义一个继承自GridSpecBase的类GridSpecFromSubplotSpec，用于管理子图的网格规格
class GridSpecFromSubplotSpec(GridSpecBase):
    figure: Figure | None  # 定义一个Figure对象或None
    # 初始化方法，接受行数（nrows）、列数（ncols）、子图规格（subplot_spec）、水平间距（wspace）、垂直间距（hspace）、行高比例（height_ratios）、列宽比例（width_ratios）
    def __init__(
        self,
        nrows: int,
        ncols: int,
        subplot_spec: SubplotSpec,
        wspace: float | None = ...,
        hspace: float | None = ...,
        height_ratios: ArrayLike | None = ...,
        width_ratios: ArrayLike | None = ...,
    ) -> None: ...

    # 返回顶层子图规格（SubplotSpec对象）
    def get_topmost_subplotspec(self) -> SubplotSpec: ...

# 定义一个子图规格类SubplotSpec
class SubplotSpec:
    num1: int  # 定义一个整数num1
    # 初始化方法，接受网格规格（gridspec）、整数num1和可选的整数num2
    def __init__(
        self, gridspec: GridSpecBase, num1: int, num2: int | None = ...
    ) -> None: ...
    
    # 从子图参数创建SubplotSpec对象的静态方法
    @staticmethod
    def _from_subplot_args(figure, args): ...

    @property
    def num2(self) -> int:  # 返回整数num2的属性
    @num2.setter
    def num2(self, value: int) -> None: ...  # 设置整数num2的属性
    
    # 返回该子图规格所属的网格规格对象（GridSpec对象）
    def get_gridspec(self) -> GridSpec: ...
    
    # 返回子图规格的几何形状，以元组形式返回（左，底，右，顶）
    def get_geometry(self) -> tuple[int, int, int, int]: ...
    
    @property
    def rowspan(self) -> range:  # 返回行跨度的属性
    @property
    def colspan(self) -> range:  # 返回列跨度的属性
    
    # 判断该子图是否位于第一行
    def is_first_row(self) -> bool: ...
    
    # 判断该子图是否位于最后一行
    def is_last_row(self) -> bool: ...
    
    # 判断该子图是否位于第一列
    def is_first_col(self) -> bool: ...
    
    # 判断该子图是否位于最后一列
    def is_last_col(self) -> bool: ...
    
    # 获取子图在图中的位置（Bbox对象）
    def get_position(self, figure: Figure) -> Bbox: ...
    
    # 返回顶层子图规格（SubplotSpec对象）
    def get_topmost_subplotspec(self) -> SubplotSpec: ...
    
    # 判断该子图规格是否与另一个对象相等
    def __eq__(self, other: object) -> bool: ...
    
    # 返回该子图规格对象的哈希值
    def __hash__(self) -> int: ...
    
    # 创建并返回一个子网格规格对象（GridSpecFromSubplotSpec对象）
    def subgridspec(
        self, nrows: int, ncols: int, **kwargs
    ) -> GridSpecFromSubplotSpec: ...

# 定义子图参数类SubplotParams
class SubplotParams:
    # 初始化方法，接受左边距（left）、底边距（bottom）、右边距（right）、顶边距（top）、水平间距（wspace）、垂直间距（hspace）
    def __init__(
        self,
        left: float | None = ...,
        bottom: float | None = ...,
        right: float | None = ...,
        top: float | None = ...,
        wspace: float | None = ...,
        hspace: float | None = ...,
    ) -> None: ...
    
    left: float  # 左边距属性
    right: float  # 右边距属性
    bottom: float  # 底边距属性
    top: float  # 顶边距属性
    wspace: float  # 水平间距属性
    hspace: float  # 垂直间距属性
    
    # 更新子图参数的方法，接受左边距（left）、底边距（bottom）、右边距（right）、顶边距（top）、水平间距（wspace）、垂直间距（hspace）
    def update(
        self,
        left: float | None = ...,
        bottom: float | None = ...,
        right: float | None = ...,
        top: float | None = ...,
        wspace: float | None = ...,
        hspace: float | None = ...,
    ) -> None: ...
```