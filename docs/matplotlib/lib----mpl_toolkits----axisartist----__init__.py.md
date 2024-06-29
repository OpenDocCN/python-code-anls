# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axisartist\__init__.py`

```py
# 导入需要的模块和类

from .axislines import Axes
from .axislines import (  # noqa: F401
    AxesZero, AxisArtistHelper, AxisArtistHelperRectlinear,
    GridHelperBase, GridHelperRectlinear, Subplot, SubplotZero)
from .axis_artist import AxisArtist, GridlinesCollection  # noqa: F401
from .grid_helper_curvelinear import GridHelperCurveLinear  # noqa: F401
from .floating_axes import FloatingAxes, FloatingSubplot  # noqa: F401
from mpl_toolkits.axes_grid1.parasite_axes import (
    host_axes_class_factory, parasite_axes_class_factory)

# 使用 parasite_axes_class_factory 创建 ParasiteAxes 类，基于 Axes 类的变体
ParasiteAxes = parasite_axes_class_factory(Axes)
# 使用 host_axes_class_factory 创建 HostAxes 类，也是基于 Axes 类的变体
HostAxes = host_axes_class_factory(Axes)
# 设置 SubplotHost 作为 HostAxes 的别名，表示宿主子图
SubplotHost = HostAxes
```