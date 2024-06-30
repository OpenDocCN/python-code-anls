# `D:\src\scipysrc\sympy\sympy\plotting\__init__.py`

```
# 导入绘图模块中的各项功能
from .plot import plot_backends  # 导入绘图后端的相关函数
from .plot_implicit import plot_implicit  # 导入隐式绘图函数
from .textplot import textplot  # 导入文本绘图函数
from .pygletplot import PygletPlot  # 导入基于Pyglet的绘图类
from .plot import PlotGrid  # 导入绘图网格类
from .plot import (plot, plot_parametric, plot3d, plot3d_parametric_surface,
                  plot3d_parametric_line, plot_contour)  # 导入各种绘图函数

# __all__ 列表定义了模块中对外公开的名称列表，用于 from module import * 导入
__all__ = [
    'plot_backends',  # 绘图后端相关函数
    'plot_implicit',  # 隐式绘图函数
    'textplot',  # 文本绘图函数
    'PygletPlot',  # 基于Pyglet的绘图类
    'PlotGrid',  # 绘图网格类
    'plot', 'plot_parametric', 'plot3d', 'plot3d_parametric_surface',
    'plot3d_parametric_line', 'plot_contour'  # 各种绘图函数
]
```