# `D:\src\scipysrc\matplotlib\lib\matplotlib\projections\polar.pyi`

```
import matplotlib.axis as maxis
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.text import Text

import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Sequence
from typing import Any, ClassVar, Literal, overload

# 导入必要的模块和类

class PolarTransform(mtransforms.Transform):
    input_dims: int
    output_dims: int
    def __init__(
        self,
        axis: PolarAxes | None = ...,
        use_rmin: bool = ...,
        *,
        apply_theta_transforms: bool = ...,
        scale_transform: mtransforms.Transform | None = ...,
    ) -> None: ...
    # 极坐标变换类，继承自matplotlib的Transform类

    def inverted(self) -> InvertedPolarTransform: ...
    # 返回逆变换的方法

class PolarAffine(mtransforms.Affine2DBase):
    def __init__(
        self, scale_transform: mtransforms.Transform, limits: mtransforms.BboxBase
    ) -> None: ...
    # 极坐标仿射变换类，继承自matplotlib的Affine2DBase类

class InvertedPolarTransform(mtransforms.Transform):
    input_dims: int
    output_dims: int
    def __init__(
        self,
        axis: PolarAxes | None = ...,
        use_rmin: bool = ...,
        *,
        apply_theta_transforms: bool = ...,
    ) -> None: ...
    # 逆极坐标变换类，继承自matplotlib的Transform类

    def inverted(self) -> PolarTransform: ...
    # 返回正向变换的方法

class ThetaFormatter(mticker.Formatter): ...
# 角度格式化类，继承自matplotlib的Formatter类

class _AxisWrapper:
    def __init__(self, axis: maxis.Axis) -> None: ...
    # 轴包装类，封装matplotlib的Axis类

    def get_view_interval(self) -> np.ndarray: ...
    # 获取视图区间的方法

    def set_view_interval(self, vmin: float, vmax: float) -> None: ...
    # 设置视图区间的方法

    def get_minpos(self) -> float: ...
    # 获取最小位置的方法

    def get_data_interval(self) -> np.ndarray: ...
    # 获取数据区间的方法

    def set_data_interval(self, vmin: float, vmax: float) -> None: ...
    # 设置数据区间的方法

    def get_tick_space(self) -> int: ...
    # 获取刻度空间的方法

class ThetaLocator(mticker.Locator):
    base: mticker.Locator
    axis: _AxisWrapper | None
    def __init__(self, base: mticker.Locator) -> None: ...
    # 角度定位器类，继承自matplotlib的Locator类

class ThetaTick(maxis.XTick):
    def __init__(self, axes: PolarAxes, *args, **kwargs) -> None: ...
    # 角度刻度类，继承自matplotlib的XTick类

class ThetaAxis(maxis.XAxis):
    axis_name: str
    # 角度轴类，继承自matplotlib的XAxis类

class RadialLocator(mticker.Locator):
    base: mticker.Locator
    def __init__(self, base, axes: PolarAxes | None = ...) -> None: ...
    # 径向定位器类，继承自matplotlib的Locator类

class RadialTick(maxis.YTick): ...
# 径向刻度类，继承自matplotlib的YTick类

class RadialAxis(maxis.YAxis):
    axis_name: str
    # 径向轴类，继承自matplotlib的YAxis类

class _WedgeBbox(mtransforms.Bbox):
    def __init__(
        self,
        center: tuple[float, float],
        viewLim: mtransforms.Bbox,
        originLim: mtransforms.Bbox,
        **kwargs,
    ) -> None: ...
    # 楔形边界框类，继承自matplotlib的Bbox类

class PolarAxes(Axes):
    PolarTransform: ClassVar[type] = PolarTransform
    PolarAffine: ClassVar[type] = PolarAffine
    InvertedPolarTransform: ClassVar[type] = InvertedPolarTransform
    ThetaFormatter: ClassVar[type] = ThetaFormatter
    RadialLocator: ClassVar[type] = RadialLocator
    ThetaLocator: ClassVar[type] = ThetaLocator

    name: str
    use_sticky_edges: bool
    def __init__(
        self,
        *args,
        theta_offset: float = ...,
        theta_direction: float = ...,
        rlabel_position: float = ...,
        **kwargs,
    ) -> None: ...
    # 极坐标轴类，继承自matplotlib的Axes类
    # 获取 x 轴的变换对象
    def get_xaxis_transform(
        self, which: Literal["tick1", "tick2", "grid"] = ...
    ) -> mtransforms.Transform: ...
    
    # 获取 x 轴文本1的变换对象和对齐方式
    def get_xaxis_text1_transform(
        self, pad: float
    ) -> tuple[
        mtransforms.Transform,
        Literal["center", "top", "bottom", "baseline", "center_baseline"],
        Literal["center", "left", "right"],
    ]: ...
    
    # 获取 x 轴文本2的变换对象和对齐方式
    def get_xaxis_text2_transform(
        self, pad: float
    ) -> tuple[
        mtransforms.Transform,
        Literal["center", "top", "bottom", "baseline", "center_baseline"],
        Literal["center", "left", "right"],
    ]: ...
    
    # 获取 y 轴的变换对象
    def get_yaxis_transform(
        self, which: Literal["tick1", "tick2", "grid"] = ...
    ) -> mtransforms.Transform: ...
    
    # 获取 y 轴文本1的变换对象和对齐方式
    def get_yaxis_text1_transform(
        self, pad: float
    ) -> tuple[
        mtransforms.Transform,
        Literal["center", "top", "bottom", "baseline", "center_baseline"],
        Literal["center", "left", "right"],
    ]: ...
    
    # 获取 y 轴文本2的变换对象和对齐方式
    def get_yaxis_text2_transform(
        self, pad: float
    ) -> tuple[
        mtransforms.Transform,
        Literal["center", "top", "bottom", "baseline", "center_baseline"],
        Literal["center", "left", "right"],
    ]: ...
    
    # 设置极坐标图的最大角度值
    def set_thetamax(self, thetamax: float) -> None: ...
    
    # 获取极坐标图的最大角度值
    def get_thetamax(self) -> float: ...
    
    # 设置极坐标图的最小角度值
    def set_thetamin(self, thetamin: float) -> None: ...
    
    # 获取极坐标图的最小角度值
    def get_thetamin(self) -> float: ...
    
    # 设置极坐标图的角度限制范围
    @overload
    def set_thetalim(self, minval: float, maxval: float, /) -> tuple[float, float]: ...
    @overload
    def set_thetalim(self, *, thetamin: float, thetamax: float) -> tuple[float, float]: ...
    
    # 设置极坐标图的角度偏移量
    def set_theta_offset(self, offset: float) -> None: ...
    
    # 获取极坐标图的角度偏移量
    def get_theta_offset(self) -> float: ...
    
    # 设置极坐标图的零角度位置和偏移量
    def set_theta_zero_location(
        self,
        loc: Literal["N", "NW", "W", "SW", "S", "SE", "E", "NE"],
        offset: float = ...,
    ) -> None: ...
    
    # 设置极坐标图的旋转方向
    def set_theta_direction(
        self,
        direction: Literal[-1, 1, "clockwise", "counterclockwise", "anticlockwise"],
    ) -> None: ...
    
    # 获取极坐标图的旋转方向
    def get_theta_direction(self) -> Literal[-1, 1]: ...
    
    # 设置极坐标图的最大半径值
    def set_rmax(self, rmax: float) -> None: ...
    
    # 获取极坐标图的最大半径值
    def get_rmax(self) -> float: ...
    
    # 设置极坐标图的最小半径值
    def set_rmin(self, rmin: float) -> None: ...
    
    # 获取极坐标图的最小半径值
    def get_rmin(self) -> float: ...
    
    # 设置极坐标图的半径原点值
    def set_rorigin(self, rorigin: float | None) -> None: ...
    
    # 获取极坐标图的半径原点值
    def get_rorigin(self) -> float: ...
    
    # 获取极坐标图的半径符号
    def get_rsign(self) -> float: ...
    
    # 设置极坐标图的半径限制范围
    def set_rlim(
        self,
        bottom: float | tuple[float, float] | None = ...,
        top: float | None = ...,
        *,
        emit: bool = ...,
        auto: bool = ...,
        **kwargs,
    ) -> tuple[float, float]: ...
    
    # 获取极坐标图的半径标签位置
    def get_rlabel_position(self) -> float: ...
    
    # 设置极坐标图的半径标签位置
    def set_rlabel_position(self, value: float) -> None: ...
    
    # 设置极坐标图的半径缩放
    def set_rscale(self, *args, **kwargs) -> None: ...
    
    # 设置极坐标图的半径刻度
    def set_rticks(self, *args, **kwargs) -> None: ...
    # 设置雷达图的角度刻度线及其标签
    def set_thetagrids(
        self,
        angles: ArrayLike,
        labels: Sequence[str | Text] | None = ...,
        fmt: str | None = ...,
        **kwargs,
    ) -> tuple[list[Line2D], list[Text]]: ...

    # 设置雷达图的半径刻度线及其标签
    def set_rgrids(
        self,
        radii: ArrayLike,
        labels: Sequence[str | Text] | None = ...,
        angle: float | None = ...,
        fmt: str | None = ...,
        **kwargs,
    ) -> tuple[list[Line2D], list[Text]]: ...

    # 格式化鼠标指针当前位置的坐标信息为字符串
    def format_coord(self, theta: float, r: float) -> str: ...

    # 获取数据的比例，用于判断数据的缩放比例
    def get_data_ratio(self) -> float: ...

    # 判断是否可以进行缩放操作
    def can_zoom(self) -> bool: ...

    # 判断是否可以进行平移操作
    def can_pan(self) -> bool: ...

    # 开始平移操作，记录起始坐标和按钮信息
    def start_pan(self, x: float, y: float, button: int) -> None: ...

    # 结束平移操作
    def end_pan(self) -> None: ...

    # 进行平移操作，更新平移后的坐标位置
    def drag_pan(self, button: Any, key: Any, x: float, y: float) -> None: ...
```