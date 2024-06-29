# `D:\src\scipysrc\matplotlib\lib\matplotlib\legend_handler.pyi`

```
# 导入必要的模块和类
from collections.abc import Callable, Sequence           # 导入必要的集合和类型
from matplotlib.artist import Artist                    # 导入画家类
from matplotlib.legend import Legend                    # 导入图例类
from matplotlib.offsetbox import OffsetBox              # 导入偏移框类
from matplotlib.transforms import Transform             # 导入变换类

from typing import TypeVar                               # 导入类型变量

from numpy.typing import ArrayLike                       # 导入类似数组的类型

def update_from_first_child(tgt: Artist, src: Artist) -> None: ...
    # 更新目标画家对象的属性，使用源画家对象的属性作为参考

class HandlerBase:
    def __init__(
        self,
        xpad: float = ...,                              # X轴方向的填充
        ypad: float = ...,                              # Y轴方向的填充
        update_func: Callable[[Artist, Artist], None] | None = ...,  # 更新函数的可调用对象或None
    ) -> None: ...
        # 初始化处理程序基类，设置填充和可选的更新函数

    def update_prop(
        self, legend_handle: Artist, orig_handle: Artist, legend: Legend
    ) -> None: ...
        # 更新图例句柄的属性，基于原始句柄和图例对象

    def adjust_drawing_area(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
    ) -> tuple[float, float, float, float]: ...
        # 调整绘图区域的大小和位置，返回新的位置参数元组

    def legend_artist(
        self, legend: Legend, orig_handle: Artist, fontsize: float, handlebox: OffsetBox
    ) -> Artist: ...
        # 创建图例的艺术家对象，基于原始句柄、字体大小和处理框

    def create_artists(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> Sequence[Artist]: ...
        # 创建一组艺术家对象用于图例显示，基于原始句柄和各种位置参数以及变换对象

class HandlerNpoints(HandlerBase):
    def __init__(
        self, marker_pad: float = ...,                   # 标记之间的填充
        numpoints: int | None = ...,                     # 标记点的数量或None
        **kwargs
    ) -> None: ...
        # 初始化N个点的处理程序，设置标记之间的填充和标记点的数量或None

    def get_numpoints(self, legend: Legend) -> int | None: ...
        # 获取图例对象中的标记点数量或None

    def get_xdata(
        self,
        legend: Legend,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
    ) -> tuple[ArrayLike, ArrayLike]: ...
        # 获取X轴数据，返回元组包含X轴数据和对应的标签数据

class HandlerNpointsYoffsets(HandlerNpoints):
    def __init__(
        self,
        numpoints: int | None = ...,                     # 标记点的数量或None
        yoffsets: Sequence[float] | None = ...,          # Y轴偏移序列或None
        **kwargs
    ) -> None: ...
        # 初始化带有Y轴偏移的N个点的处理程序，设置标记点的数量或None和Y轴偏移序列或None

    def get_ydata(
        self,
        legend: Legend,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
    ) -> ArrayLike: ...
        # 获取Y轴数据，返回Y轴数据的类似数组

class HandlerLine2DCompound(HandlerNpoints):
    def create_artists(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> Sequence[Artist]: ...
        # 创建复合线条的艺术家对象集合，用于图例显示，基于原始句柄和各种位置参数以及变换对象

class HandlerLine2D(HandlerNpoints):
    def create_artists(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> Sequence[Artist]: ...
        # 创建线条的艺术家对象集合，用于图例显示，基于原始句柄和各种位置参数以及变换对象

class HandlerPatch(HandlerBase):
    def __init__(self, patch_func: Callable | None = ..., **kwargs) -> None: ...
        # 初始化图形补丁的处理程序，设置可调用的图形补丁函数或None以及其他关键字参数
    # 定义一个方法 `create_artists`，用于创建艺术家对象序列，根据提供的参数进行定制化绘制。
    def create_artists(
        self,
        legend: Legend,            # 传入一个图例对象，用于在图中添加说明
        orig_handle: Artist,       # 原始艺术家对象，作为模板进行绘制
        xdescent: float,           # x 方向下降距离，影响布局
        ydescent: float,           # y 方向下降距离，影响布局
        width: float,              # 绘制对象的宽度
        height: float,             # 绘制对象的高度
        fontsize: float,           # 字体大小，影响文本的显示
        trans: Transform,          # 转换对象，用于控制绘制对象的位置和大小
    ) -> Sequence[Artist]:
        ...
class HandlerStepPatch(HandlerBase):
    # 继承自 HandlerBase 类，用于处理步进类型的图形修补
    def create_artists(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> Sequence[Artist]: ...
    # 创建图形对象的方法，返回一个艺术家对象的序列

class HandlerLineCollection(HandlerLine2D):
    # 继承自 HandlerLine2D 类，用于处理线集合类型的图形对象
    def get_numpoints(self, legend: Legend) -> int: ...
    # 获取图例中数据点的数量
    
    def create_artists(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> Sequence[Artist]: ...
    # 创建图形对象的方法，返回一个艺术家对象的序列

_T = TypeVar("_T", bound=Artist)

class HandlerRegularPolyCollection(HandlerNpointsYoffsets):
    # 继承自 HandlerNpointsYoffsets 类，处理正多边形集合类型的图形对象
    def __init__(
        self,
        yoffsets: Sequence[float] | None = ...,
        sizes: Sequence[float] | None = ...,
        **kwargs
    ) -> None: ...
    # 初始化方法，接受偏移量和大小的序列作为参数
    
    def get_numpoints(self, legend: Legend) -> int: ...
    # 获取图例中数据点的数量
    
    def get_sizes(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
    ) -> Sequence[float]: ...
    # 获取图例中数据点的大小序列
    
    def update_prop(
        self, legend_handle, orig_handle: Artist, legend: Legend
    ) -> None: ...
    # 更新图例属性的方法
    
    def create_collection(
        self,
        orig_handle: _T,
        sizes: Sequence[float] | None,
        offsets: Sequence[float] | None,
        offset_transform: Transform,
    ) -> _T: ...
    # 创建图形对象集合的方法，返回一个艺术家对象

    def create_artists(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> Sequence[Artist]: ...
    # 创建图形对象的方法，返回一个艺术家对象的序列

class HandlerPathCollection(HandlerRegularPolyCollection):
    # 继承自 HandlerRegularPolyCollection 类，用于处理路径集合类型的图形对象
    def create_collection(
        self,
        orig_handle: _T,
        sizes: Sequence[float] | None,
        offsets: Sequence[float] | None,
        offset_transform: Transform,
    ) -> _T: ...
    # 创建图形对象集合的方法，返回一个艺术家对象

class HandlerCircleCollection(HandlerRegularPolyCollection):
    # 继承自 HandlerRegularPolyCollection 类，用于处理圆圈集合类型的图形对象
    def create_collection(
        self,
        orig_handle: _T,
        sizes: Sequence[float] | None,
        offsets: Sequence[float] | None,
        offset_transform: Transform,
    ) -> _T: ...
    # 创建图形对象集合的方法，返回一个艺术家对象

class HandlerErrorbar(HandlerLine2D):
    # 继承自 HandlerLine2D 类，用于处理误差线类型的图形对象
    def __init__(
        self,
        xerr_size: float = ...,
        yerr_size: float | None = ...,
        marker_pad: float = ...,
        numpoints: int | None = ...,
        **kwargs
    ) -> None: ...
    # 初始化方法，接受误差大小、标记填充和数据点数量等参数
    
    def get_err_size(
        self,
        legend: Legend,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
    ) -> tuple[float, float]: ...
    # 获取误差大小的方法，返回一个浮点数元组
    # 定义一个方法 create_artists，用于生成艺术家对象的序列
    def create_artists(
        self,
        legend: Legend,          # 参数 legend：图例对象，用于创建艺术家对象
        orig_handle: Artist,     # 参数 orig_handle：原始艺术家对象，作为模板
        xdescent: float,         # 参数 xdescent：X轴下降量，浮点数类型
        ydescent: float,         # 参数 ydescent：Y轴下降量，浮点数类型
        width: float,            # 参数 width：宽度，浮点数类型
        height: float,           # 参数 height：高度，浮点数类型
        fontsize: float,         # 参数 fontsize：字体大小，浮点数类型
        trans: Transform,        # 参数 trans：变换对象，用于位置和形状的变换
    ) -> Sequence[Artist]:       # 返回类型为 Artist 的序列
        ...
# 继承自 HandlerNpointsYoffsets 类，处理 STEM 类型的图例处理器
class HandlerStem(HandlerNpointsYoffsets):
    # 初始化方法，接受多个参数，包括标记填充、点数、底部偏移、y轴偏移和额外关键字参数
    def __init__(
        self,
        marker_pad: float = ...,
        numpoints: int | None = ...,
        bottom: float | None = ...,
        yoffsets: Sequence[float] | None = ...,
        **kwargs
    ) -> None: ...
    
    # 获取 y 数据的方法，返回数组样式的数据
    def get_ydata(
        self,
        legend: Legend,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
    ) -> ArrayLike: ...

    # 创建艺术家对象的方法，返回艺术家对象序列
    def create_artists(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> Sequence[Artist]: ...

# 继承自 HandlerBase 类，处理元组类型的图例处理器
class HandlerTuple(HandlerBase):
    # 初始化方法，接受多个参数，包括分割数、填充和额外关键字参数
    def __init__(
        self, ndivide: int | None = ..., pad: float | None = ..., **kwargs
    ) -> None: ...
    
    # 创建艺术家对象的方法，返回艺术家对象序列
    def create_artists(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> Sequence[Artist]: ...

# 继承自 HandlerBase 类，处理多边形集合类型的图例处理器
class HandlerPolyCollection(HandlerBase):
    # 创建艺术家对象的方法，返回艺术家对象序列
    def create_artists(
        self,
        legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> Sequence[Artist]: ...
```