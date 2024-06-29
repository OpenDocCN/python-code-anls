# `D:\src\scipysrc\pandas\pandas\plotting\_matplotlib\__init__.py`

```
# 导入必要的类型检查支持
from __future__ import annotations

# 引入类型检查所需的模块
from typing import TYPE_CHECKING

# 从 pandas.plotting._matplotlib.boxplot 模块中导入相关函数和类
from pandas.plotting._matplotlib.boxplot import (
    BoxPlot,          # 导入 BoxPlot 类
    boxplot,          # 导入 boxplot 函数
    boxplot_frame,    # 导入 boxplot_frame 函数
    boxplot_frame_groupby,  # 导入 boxplot_frame_groupby 函数
)

# 从 pandas.plotting._matplotlib.converter 模块中导入相关函数
from pandas.plotting._matplotlib.converter import (
    deregister,  # 导入 deregister 函数
    register,    # 导入 register 函数
)

# 从 pandas.plotting._matplotlib.core 模块中导入相关绘图类
from pandas.plotting._matplotlib.core import (
    AreaPlot,      # 导入 AreaPlot 类
    BarhPlot,      # 导入 BarhPlot 类
    BarPlot,       # 导入 BarPlot 类
    HexBinPlot,    # 导入 HexBinPlot 类
    LinePlot,      # 导入 LinePlot 类
    PiePlot,       # 导入 PiePlot 类
    ScatterPlot,   # 导入 ScatterPlot 类
)

# 从 pandas.plotting._matplotlib.hist 模块中导入相关绘图函数和类
from pandas.plotting._matplotlib.hist import (
    HistPlot,      # 导入 HistPlot 类
    KdePlot,       # 导入 KdePlot 类
    hist_frame,    # 导入 hist_frame 函数
    hist_series,   # 导入 hist_series 函数
)

# 从 pandas.plotting._matplotlib.misc 模块中导入其他杂项绘图函数
from pandas.plotting._matplotlib.misc import (
    andrews_curves,           # 导入 andrews_curves 函数
    autocorrelation_plot,     # 导入 autocorrelation_plot 函数
    bootstrap_plot,           # 导入 bootstrap_plot 函数
    lag_plot,                 # 导入 lag_plot 函数
    parallel_coordinates,     # 导入 parallel_coordinates 函数
    radviz,                   # 导入 radviz 函数
    scatter_matrix,           # 导入 scatter_matrix 函数
)

# 从 pandas.plotting._matplotlib.tools 模块中导入 table 函数
from pandas.plotting._matplotlib.tools import table

# 如果在类型检查模式下，则导入 MPLPlot 类
if TYPE_CHECKING:
    from pandas.plotting._matplotlib.core import MPLPlot

# 定义一个字典，将字符串类型的绘图类型映射到对应的绘图类
PLOT_CLASSES: dict[str, type[MPLPlot]] = {
    "line": LinePlot,         # 映射 "line" 到 LinePlot 类
    "bar": BarPlot,           # 映射 "bar" 到 BarPlot 类
    "barh": BarhPlot,         # 映射 "barh" 到 BarhPlot 类
    "box": BoxPlot,           # 映射 "box" 到 BoxPlot 类
    "hist": HistPlot,         # 映射 "hist" 到 HistPlot 类
    "kde": KdePlot,           # 映射 "kde" 到 KdePlot 类
    "area": AreaPlot,         # 映射 "area" 到 AreaPlot 类
    "pie": PiePlot,           # 映射 "pie" 到 PiePlot 类
    "scatter": ScatterPlot,   # 映射 "scatter" 到 ScatterPlot 类
    "hexbin": HexBinPlot,     # 映射 "hexbin" 到 HexBinPlot 类
}


def plot(data, kind, **kwargs):
    # 在文件顶部导入 pyplot 可能会导致在 matplotlib 2 中（转换器似乎不起作用时）出现问题
    import matplotlib.pyplot as plt

    # 如果设置了 reuse_plot 参数为 True，则尝试重用现有图形
    if kwargs.pop("reuse_plot", False):
        ax = kwargs.get("ax")
        # 如果未指定轴，并且已经存在至少一个图形，则获取当前轴
        if ax is None and len(plt.get_fignums()) > 0:
            with plt.rc_context():
                ax = plt.gca()
            kwargs["ax"] = getattr(ax, "left_ax", ax)
    # 根据绘图类型创建对应的绘图对象
    plot_obj = PLOT_CLASSES[kind](data, **kwargs)
    # 生成绘图
    plot_obj.generate()
    # 如果是交互模式，则绘制图形
    plt.draw_if_interactive()
    # 返回绘图对象的结果
    return plot_obj.result


# 定义模块的公开接口，列出可以从模块中导入的公共符号
__all__ = [
    "plot",
    "hist_series",
    "hist_frame",
    "boxplot",
    "boxplot_frame",
    "boxplot_frame_groupby",
    "table",
    "andrews_curves",
    "autocorrelation_plot",
    "bootstrap_plot",
    "lag_plot",
    "parallel_coordinates",
    "radviz",
    "scatter_matrix",
    "register",
    "deregister",
]
```