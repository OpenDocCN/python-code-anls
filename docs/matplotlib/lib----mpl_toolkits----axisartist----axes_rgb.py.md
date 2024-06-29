# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axisartist\axes_rgb.py`

```py
from matplotlib import _api  # 导入 matplotlib 中的 _api 模块

from mpl_toolkits.axes_grid1.axes_rgb import (  # 导入 axes_rgb 模块中的 make_rgb_axes 和 RGBAxes 类
    make_rgb_axes, RGBAxes as _RGBAxes)

from .axislines import Axes  # 导入当前包中的 axislines 模块中的 Axes 类


_api.warn_deprecated(
    "3.8", name=__name__, obj_type="module", alternative="axes_grid1.axes_rgb")
# 使用 _api 模块的 warn_deprecated 函数发出警告，表示当前代码即将被弃用，推荐替代方案为 axes_grid1.axes_rgb 模块


@_api.deprecated("3.8", alternative=(
    "axes_grid1.axes_rgb.RGBAxes(..., axes_class=axislines.Axes"))
# 使用 _api 模块的 deprecated 装饰器标记这个类为弃用，提供替代的类构造方式
class RGBAxes(_RGBAxes):
    """
    Subclass of `~.axes_grid1.axes_rgb.RGBAxes` with
    ``_defaultAxesClass`` = `.axislines.Axes`.
    """
    _defaultAxesClass = Axes  # 设置默认的轴类为当前包中的 axislines 模块中的 Axes 类
```