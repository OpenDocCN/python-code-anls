# `D:\src\scipysrc\matplotlib\lib\matplotlib\axis.pyi`

```
from collections.abc import Callable, Iterable, Sequence  # 导入必要的集合类型
import datetime  # 导入日期时间模块
from typing import Any, Literal, overload  # 引入类型提示相关模块

import numpy as np  # 导入NumPy库
from numpy.typing import ArrayLike  # 引入NumPy数组类型提示

import matplotlib.artist as martist  # 导入matplotlib的艺术家模块
from matplotlib import cbook  # 导入matplotlib的cbook模块
from matplotlib.axes import Axes  # 导入matplotlib的Axes类
from matplotlib.backend_bases import RendererBase  # 导入matplotlib的RendererBase基类
from matplotlib.lines import Line2D  # 导入matplotlib的Line2D类
from matplotlib.text import Text  # 导入matplotlib的Text类
from matplotlib.ticker import Locator, Formatter  # 导入matplotlib的Locator和Formatter类
from matplotlib.transforms import Transform, Bbox  # 导入matplotlib的Transform和Bbox类
from matplotlib.typing import ColorType  # 导入matplotlib的颜色类型提示

GRIDLINE_INTERPOLATION_STEPS: int  # 声明网格线插值步数变量

class Tick(martist.Artist):  # Tick类，继承自matplotlib的Artist类
    axes: Axes  # 坐标轴对象
    tick1line: Line2D  # 刻度线1对象
    tick2line: Line2D  # 刻度线2对象
    gridline: Line2D  # 网格线对象
    label1: Text  # 标签1对象
    label2: Text  # 标签2对象
    
    def __init__(  # Tick类的初始化方法
        self,
        axes: Axes,
        loc: float,
        *,
        size: float | None = ...,
        width: float | None = ...,
        color: ColorType | None = ...,
        tickdir: Literal["in", "inout", "out"] | None = ...,
        pad: float | None = ...,
        labelsize: float | None = ...,
        labelcolor: ColorType | None = ...,
        labelfontfamily: str | Sequence[str] | None = ...,
        zorder: float | None = ...,
        gridOn: bool | None = ...,
        tick1On: bool = ...,
        tick2On: bool = ...,
        label1On: bool = ...,
        label2On: bool = ...,
        major: bool = ...,
        labelrotation: float = ...,
        grid_color: ColorType | None = ...,
        grid_linestyle: str | None = ...,
        grid_linewidth: float | None = ...,
        grid_alpha: float | None = ...,
        **kwargs
    ) -> None: ...

    def get_tickdir(self) -> Literal["in", "inout", "out"]:  # 获取刻度方向方法
        pass

    def get_tick_padding(self) -> float:  # 获取刻度间距方法
        pass

    def get_children(self) -> list[martist.Artist]:  # 获取子对象列表方法
        pass

    stale: bool  # 刷新状态标志

    def set_pad(self, val: float) -> None:  # 设置刻度间距方法
        pass

    def get_pad(self) -> None:  # 获取刻度间距方法
        pass

    def get_loc(self) -> float:  # 获取刻度位置方法
        pass

    def set_label1(self, s: object) -> None:  # 设置标签1内容方法
        pass

    def set_label(self, s: object) -> None:  # 设置标签内容方法
        pass

    def set_label2(self, s: object) -> None:  # 设置标签2内容方法
        pass

    def set_url(self, url: str | None) -> None:  # 设置URL链接方法
        pass

    def get_view_interval(self) -> ArrayLike:  # 获取视图区间方法
        pass

    def update_position(self, loc: float) -> None:  # 更新位置方法
        pass

class XTick(Tick):  # X轴刻度类，继承自Tick类
    __name__: str  # 名称属性

    def __init__(self, *args, **kwargs) -> None: ...  # 初始化方法，继承自Tick类

    stale: bool  # 刷新状态标志

    def update_position(self, loc: float) -> None:  # 更新位置方法
        pass

    def get_view_interval(self) -> np.ndarray:  # 获取视图区间方法
        pass

class YTick(Tick):  # Y轴刻度类，继承自Tick类
    __name__: str  # 名称属性

    def __init__(self, *args, **kwargs) -> None: ...  # 初始化方法，继承自Tick类

    stale: bool  # 刷新状态标志

    def update_position(self, loc: float) -> None:  # 更新位置方法
        pass

    def get_view_interval(self) -> np.ndarray:  # 获取视图区间方法
        pass

class Ticker:  # 刻度类
    def __init__(self) -> None: ...  # 初始化方法
    @property
    def locator(self) -> Locator | None: ...  # 定位器属性
    @locator.setter
    def locator(self, locator: Locator) -> None: ...  # 定位器设置方法
    @property
    def formatter(self) -> Formatter | None: ...  # 格式化器属性
    @formatter.setter
    def formatter(self, formatter: Formatter) -> None: ...  # 格式化器设置方法

class _LazyTickList:  # 惰性刻度列表类
    ...
    # 定义一个初始化方法，接受一个布尔类型的参数 `major`，无返回值
    def __init__(self, major: bool) -> None: ...
    
    # 对于 py3.9 被弃用的情况下，将返回值类型 `Self` 替换为 `None`
    @overload
    # 定义一个重载方法 `__get__`，当 `instance` 和 `owner` 均为 `None` 时返回 `_LazyTickList` 类型对象
    def __get__(self, instance: None, owner: None) -> _LazyTickList: ...
    
    # 定义一个重载方法 `__get__`，接受一个 `instance` 参数为 `Axis` 类型，`owner` 参数为 `Axis` 类型，返回 `Tick` 类型列表
    @overload
    def __get__(self, instance: Axis, owner: type[Axis]) -> list[Tick]: ...
# 定义一个名为 Axis 的类，继承自 martist.Artist 类
class Axis(martist.Artist):
    # 类变量 OFFSETTEXTPAD，用于偏移文本的填充
    OFFSETTEXTPAD: int
    # 布尔型变量，表示标签是否为默认值
    isDefault_label: bool
    # axes 属性，表示该坐标轴所属的 Axes 对象
    axes: Axes
    # major 属性，表示主要刻度的 Ticker 对象
    major: Ticker
    # minor 属性，表示次要刻度的 Ticker 对象
    minor: Ticker
    # callbacks 属性，用于管理回调函数的注册与调用
    callbacks: cbook.CallbackRegistry
    # label 属性，表示坐标轴的标签文本对象
    label: Text
    # offsetText 属性，表示偏移文本对象，通常用于标注坐标轴的偏移信息
    offsetText: Text
    # labelpad 属性，浮点数，表示标签的填充大小
    labelpad: float
    # pickradius 属性，浮点数，表示鼠标拾取半径的大小
    pickradius: float
    
    # 构造函数，初始化 Axis 对象
    def __init__(self, axes, *, pickradius: float = ..., clear: bool = ...) -> None:
        ...

    # isDefault_majloc 属性的 getter 方法，返回是否为主要刻度的默认位置
    @property
    def isDefault_majloc(self) -> bool:
        ...

    # isDefault_majloc 属性的 setter 方法，设置是否为主要刻度的默认位置
    @isDefault_majloc.setter
    def isDefault_majloc(self, value: bool) -> None:
        ...

    # isDefault_majfmt 属性的 getter 方法，返回是否为主要刻度的默认格式
    @property
    def isDefault_majfmt(self) -> bool:
        ...

    # isDefault_majfmt 属性的 setter 方法，设置是否为主要刻度的默认格式
    @isDefault_majfmt.setter
    def isDefault_majfmt(self, value: bool) -> None:
        ...

    # isDefault_minloc 属性的 getter 方法，返回是否为次要刻度的默认位置
    @property
    def isDefault_minloc(self) -> bool:
        ...

    # isDefault_minloc 属性的 setter 方法，设置是否为次要刻度的默认位置
    @isDefault_minloc.setter
    def isDefault_minloc(self, value: bool) -> None:
        ...

    # isDefault_minfmt 属性的 getter 方法，返回是否为次要刻度的默认格式
    @property
    def isDefault_minfmt(self) -> bool:
        ...

    # isDefault_minfmt 属性的 setter 方法，设置是否为次要刻度的默认格式
    @isDefault_minfmt.setter
    def isDefault_minfmt(self, value: bool) -> None:
        ...

    # majorTicks 属性，懒加载方式获取主要刻度的列表
    majorTicks: _LazyTickList

    # minorTicks 属性，懒加载方式获取次要刻度的列表
    minorTicks: _LazyTickList

    # 获取是否移除重叠位置的方法
    def get_remove_overlapping_locs(self) -> bool:
        ...

    # 设置是否移除重叠位置的方法
    def set_remove_overlapping_locs(self, val: bool) -> None:
        ...

    # remove_overlapping_locs 属性的 getter 方法，获取是否移除重叠位置
    @property
    def remove_overlapping_locs(self) -> bool:
        ...

    # remove_overlapping_locs 属性的 setter 方法，设置是否移除重叠位置
    @remove_overlapping_locs.setter
    def remove_overlapping_locs(self, val: bool) -> None:
        ...

    # stale 属性，表示坐标轴是否过时的状态
    stale: bool

    # 设置标签的坐标位置方法
    def set_label_coords(self, x: float, y: float, transform: Transform | None = ...) -> None:
        ...

    # 获取坐标轴的变换对象方法
    def get_transform(self) -> Transform:
        ...

    # 获取坐标轴的刻度类型方法
    def get_scale(self) -> str:
        ...

    # 限制坐标轴范围方法，根据比例因子限制范围
    def limit_range_for_scale(self, vmin: float, vmax: float) -> tuple[float, float]:
        ...

    # 获取坐标轴的子对象列表方法
    def get_children(self) -> list[martist.Artist]:
        ...

    # TODO units，表示未完成的部分，计划添加单位相关功能

    # converter 属性，表示转换器对象
    converter: Any

    # units 属性，表示单位对象
    units: Any

    # 清除坐标轴的方法
    def clear(self) -> None:
        ...

    # 重置刻度方法
    def reset_ticks(self) -> None:
        ...

    # 打开次要刻度方法
    def minorticks_on(self) -> None:
        ...

    # 关闭次要刻度方法
    def minorticks_off(self) -> None:
        ...

    # 设置刻度参数方法
    def set_tick_params(self, which: Literal["major", "minor", "both"] = ..., reset: bool = ..., **kwargs) -> None:
        ...

    # 获取刻度参数方法
    def get_tick_params(self, which: Literal["major", "minor"] = ...) -> dict[str, Any]:
        ...

    # 获取视图间隔方法
    def get_view_interval(self) -> tuple[float, float]:
        ...

    # 设置视图间隔方法
    def set_view_interval(self, vmin: float, vmax: float, ignore: bool = ...) -> None:
        ...

    # 获取数据间隔方法
    def get_data_interval(self) -> tuple[float, float]:
        ...

    # 设置数据间隔方法
    def set_data_interval(self, vmin: float, vmax: float, ignore: bool = ...) -> None:
        ...

    # 获取是否反转坐标轴的方法
    def get_inverted(self) -> bool:
        ...

    # 设置是否反转坐标轴的方法
    def set_inverted(self, inverted: bool) -> None:
        ...

    # 设置默认间隔方法
    def set_default_intervals(self) -> None:
        ...

    # 获取紧凑包围框方法，用于布局计算
    def get_tightbbox(self, renderer: RendererBase | None = ..., *, for_layout_only: bool = ...) -> Bbox | None:
        ...

    # 获取刻度填充大小方法
    def get_tick_padding(self) -> float:
        ...

    # 获取网格线列表方法
    def get_gridlines(self) -> list[Line2D]:
        ...

    # 获取标签文本对象方法
    def get_label(self) -> Text:
        ...

    # 获取偏移文本对象方法
    def get_offset_text(self) -> Text:
        ...

    # 获取拾取半径方法
    def get_pickradius(self) -> float:
        ...

    # 获取主要刻度标签列表方法
    def get_majorticklabels(self) -> list[Text]:
        ...
    # 返回次要刻度标签的列表，每个标签为文本类型
    def get_minorticklabels(self) -> list[Text]: ...

    # 返回刻度标签的列表，可以选择是主要还是次要的，或者两者都包括
    def get_ticklabels(
        self, minor: bool = ..., which: Literal["major", "minor", "both"] | None = ...
    ) -> list[Text]: ...

    # 返回主要刻度线的列表，每个刻度线为 Line2D 对象
    def get_majorticklines(self) -> list[Line2D]: ...

    # 返回次要刻度线的列表，每个刻度线为 Line2D 对象
    def get_minorticklines(self) -> list[Line2D]: ...

    # 返回刻度线的列表，可以选择是主要还是次要的
    def get_ticklines(self, minor: bool = ...) -> list[Line2D]: ...

    # 返回主要刻度位置的 NumPy 数组
    def get_majorticklocs(self) -> np.ndarray: ...

    # 返回次要刻度位置的 NumPy 数组
    def get_minorticklocs(self) -> np.ndarray: ...

    # 返回刻度位置的 NumPy 数组，可以选择是主要还是次要的
    def get_ticklocs(self, *, minor: bool = ...) -> np.ndarray: ...

    # 返回刻度方向的 NumPy 数组，可以选择是主要还是次要的
    def get_ticks_direction(self, minor: bool = ...) -> np.ndarray: ...

    # 返回标签的文本内容作为字符串
    def get_label_text(self) -> str: ...

    # 返回主要定位器对象
    def get_major_locator(self) -> Locator: ...

    # 返回次要定位器对象
    def get_minor_locator(self) -> Locator: ...

    # 返回主要格式化对象
    def get_major_formatter(self) -> Formatter: ...

    # 返回次要格式化对象
    def get_minor_formatter(self) -> Formatter: ...

    # 返回主要刻度的列表，可以指定要返回的刻度数目
    def get_major_ticks(self, numticks: int | None = ...) -> list[Tick]: ...

    # 返回次要刻度的列表，可以指定要返回的刻度数目
    def get_minor_ticks(self, numticks: int | None = ...) -> list[Tick]: ...

    # 控制坐标轴的网格显示，可以选择显示主要、次要或者两者的网格
    def grid(
        self,
        visible: bool | None = ...,
        which: Literal["major", "minor", "both"] = ...,
        **kwargs
    ) -> None: ...

    # TODO units
    # 更新坐标轴的单位信息，data 参数表示数据
    def update_units(self, data): ...

    # 返回坐标轴是否有单位信息
    def have_units(self) -> bool: ...

    # 将输入的 x 值转换成坐标轴的单位
    def convert_units(self, x): ...

    # 设置坐标轴的单位
    def set_units(self, u) -> None: ...

    # 返回当前坐标轴的单位信息
    def get_units(self): ...

    # 设置坐标轴的标签文本
    def set_label_text(
        self, label: str, fontdict: dict[str, Any] | None = ..., **kwargs
    ) -> Text: ...

    # 设置主要刻度的格式化方式
    def set_major_formatter(
        self, formatter: Formatter | str | Callable[[float, float], str]
    ) -> None: ...

    # 设置次要刻度的格式化方式
    def set_minor_formatter(
        self, formatter: Formatter | str | Callable[[float, float], str]
    ) -> None: ...

    # 设置主要定位器对象
    def set_major_locator(self, locator: Locator) -> None: ...

    # 设置次要定位器对象
    def set_minor_locator(self, locator: Locator) -> None: ...

    # 设置选择点的半径
    def set_pickradius(self, pickradius: float) -> None: ...

    # 设置刻度标签的内容和属性，可以指定是主要还是次要刻度
    def set_ticklabels(
        self,
        labels: Iterable[str | Text],
        *,
        minor: bool = ...,
        fontdict: dict[str, Any] | None = ...,
        **kwargs
    ) -> list[Text]: ...

    # 设置刻度的位置和标签，可以指定是主要还是次要刻度
    def set_ticks(
        self,
        ticks: ArrayLike,
        labels: Iterable[str] | None = ...,
        *,
        minor: bool = ...,
        **kwargs
    ) -> list[Tick]: ...

    # 将坐标轴设置为日期格式，tz 参数指定时区信息
    def axis_date(self, tz: str | datetime.tzinfo | None = ...) -> None: ...

    # 返回刻度之间的间距
    def get_tick_space(self) -> int: ...

    # 返回标签的位置，可以是顶部或底部
    def get_label_position(self) -> Literal["top", "bottom"]: ...

    # 设置标签的位置，可以是顶部、底部、左侧或右侧
    def set_label_position(
        self, position: Literal["top", "bottom", "left", "right"]
    ) -> None: ...

    # 返回最小正数值
    def get_minpos(self) -> float: ...
# 定义 X 轴类，继承自 Axis 类
class XAxis(Axis):
    # 类属性 __name__，表示对象的名称，类型为 str
    __name__: str
    # 类属性 axis_name，表示轴的名称，类型为 str
    axis_name: str
    
    # 初始化方法，接受任意数量的位置参数和关键字参数，无具体实现
    def __init__(self, *args, **kwargs) -> None: ...
    
    # 类属性 label_position，表示标签位置，可以为 "bottom" 或 "top"
    label_position: Literal["bottom", "top"]
    
    # 类属性 stale，表示数据是否过时，类型为 bool
    stale: bool
    
    # 设置标签位置的方法，参数 position 可以是 "bottom" 或 "top"
    def set_label_position(self, position: Literal["bottom", "top"]) -> None: ...  # type: ignore[override]
    
    # 设置刻度位置的方法，参数 position 可以是 "top", "bottom", "both", "default", "none"
    def set_ticks_position(
        self, position: Literal["top", "bottom", "both", "default", "none"]
    ) -> None: ...
    
    # 将刻度移至顶部的方法，无具体实现
    def tick_top(self) -> None: ...
    
    # 将刻度移至底部的方法，无具体实现
    def tick_bottom(self) -> None: ...
    
    # 获取刻度位置的方法，返回值可以是 "top", "bottom", "default", "unknown"
    def get_ticks_position(self) -> Literal["top", "bottom", "default", "unknown"]: ...
    
    # 获取刻度间隔空间的方法，返回值类型为 int
    def get_tick_space(self) -> int: ...

# 定义 Y 轴类，继承自 Axis 类
class YAxis(Axis):
    # 类属性 __name__，表示对象的名称，类型为 str
    __name__: str
    # 类属性 axis_name，表示轴的名称，类型为 str
    axis_name: str
    
    # 初始化方法，接受任意数量的位置参数和关键字参数，无具体实现
    def __init__(self, *args, **kwargs) -> None: ...
    
    # 类属性 label_position，表示标签位置，可以为 "left" 或 "right"
    label_position: Literal["left", "right"]
    
    # 类属性 stale，表示数据是否过时，类型为 bool
    stale: bool
    
    # 设置标签位置的方法，参数 position 可以是 "left" 或 "right"
    def set_label_position(self, position: Literal["left", "right"]) -> None: ...  # type: ignore[override]
    
    # 设置偏移位置的方法，参数 position 可以是 "left" 或 "right"
    def set_offset_position(self, position: Literal["left", "right"]) -> None: ...
    
    # 设置刻度位置的方法，参数 position 可以是 "left", "right", "both", "default", "none"
    def set_ticks_position(
        self, position: Literal["left", "right", "both", "default", "none"]
    ) -> None: ...
    
    # 将刻度移至右侧的方法，无具体实现
    def tick_right(self) -> None: ...
    
    # 将刻度移至左侧的方法，无具体实现
    def tick_left(self) -> None: ...
    
    # 获取刻度位置的方法，返回值可以是 "left", "right", "default", "unknown"
    def get_ticks_position(self) -> Literal["left", "right", "default", "unknown"]: ...
    
    # 获取刻度间隔空间的方法，返回值类型为 int
    def get_tick_space(self) -> int: ...
```