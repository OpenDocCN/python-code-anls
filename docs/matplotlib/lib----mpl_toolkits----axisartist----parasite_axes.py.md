# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axisartist\parasite_axes.py`

```
# 导入 mpl_toolkits.axes_grid1.parasite_axes 模块中的两个函数：
# host_axes_class_factory 和 parasite_axes_class_factory
from mpl_toolkits.axes_grid1.parasite_axes import (
    host_axes_class_factory, parasite_axes_class_factory)

# 从当前包（当前模块的相对位置）导入 axislines 模块中的 Axes 类
from .axislines import Axes

# 使用 parasite_axes_class_factory 函数，基于 Axes 类创建 ParasiteAxes 类型
ParasiteAxes = parasite_axes_class_factory(Axes)

# 使用 host_axes_class_factory 函数，基于 Axes 类创建 HostAxes 和 SubplotHost 类型
HostAxes = SubplotHost = host_axes_class_factory(Axes)
```