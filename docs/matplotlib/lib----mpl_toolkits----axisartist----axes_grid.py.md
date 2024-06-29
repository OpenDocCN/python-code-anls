# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axisartist\axes_grid.py`

```py
from matplotlib import _api

# 导入了matplotlib中的一个API模块，用于内部警告和废弃功能的处理

import mpl_toolkits.axes_grid1.axes_grid as axes_grid_orig
# 导入了matplotlib的axes_grid1.axes_grid模块，作为axes_grid_orig的别名

from .axislines import Axes
# 从当前包中导入axislines模块中的Axes类

# 发出警告，指示某功能在版本3.8中已被弃用
_api.warn_deprecated(
    "3.8", name=__name__, obj_type="module", alternative="axes_grid1.axes_grid")

# 使用装饰器标记该类已经被废弃，提供了一个替代选项
@_api.deprecated("3.8", alternative=(
    "axes_grid1.axes_grid.Grid(..., axes_class=axislines.Axes"))
class Grid(axes_grid_orig.Grid):
    # 将默认的Axes类设为导入的Axes类
    _defaultAxesClass = Axes

# 使用装饰器标记该类已经被废弃，提供了一个替代选项
@_api.deprecated("3.8", alternative=(
    "axes_grid1.axes_grid.ImageGrid(..., axes_class=axislines.Axes"))
class ImageGrid(axes_grid_orig.ImageGrid):
    # 将默认的Axes类设为导入的Axes类
    _defaultAxesClass = Axes

# 将ImageGrid类赋值给AxesGrid，作为别名使用
AxesGrid = ImageGrid
```