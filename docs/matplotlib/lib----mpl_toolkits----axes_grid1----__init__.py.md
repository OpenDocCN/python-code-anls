# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axes_grid1\__init__.py`

```
# 从当前包中导入特定模块和函数
from . import axes_size as Size
# 从当前包中导入 Divider、SubplotDivider 和 make_axes_locatable 函数
from .axes_divider import Divider, SubplotDivider, make_axes_locatable
# 从当前包中导入 AxesGrid、Grid 和 ImageGrid 类
from .axes_grid import AxesGrid, Grid, ImageGrid
# 从当前包中导入 host_subplot 和 host_axes 函数
from .parasite_axes import host_subplot, host_axes

# 定义导出的模块列表，这些模块可以被外部引用
__all__ = ["Size",
           "Divider", "SubplotDivider", "make_axes_locatable",
           "AxesGrid", "Grid", "ImageGrid",
           "host_subplot", "host_axes"]
```