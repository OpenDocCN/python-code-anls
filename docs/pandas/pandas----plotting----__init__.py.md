# `D:\src\scipysrc\pandas\pandas\plotting\__init__.py`

```
"""
Plotting public API.

Authors of third-party plotting backends should implement a module with a
public ``plot(data, kind, **kwargs)``. The parameter `data` will contain
the data structure and can be a `Series` or a `DataFrame`. For example,
for ``df.plot()`` the parameter `data` will contain the DataFrame `df`.
In some cases, the data structure is transformed before being sent to
the backend (see PlotAccessor.__call__ in pandas/plotting/_core.py for
the exact transformations).

The parameter `kind` will be one of:

- line
- bar
- barh
- box
- hist
- kde
- area
- pie
- scatter
- hexbin

See the pandas API reference for documentation on each kind of plot.

Any other keyword argument is currently assumed to be backend specific,
but some parameters may be unified and added to the signature in the
future (e.g. `title` which should be useful for any backend).

Currently, all the Matplotlib functions in pandas are accessed through
the selected backend. For example, `pandas.plotting.boxplot` (equivalent
to `DataFrame.boxplot`) is also accessed in the selected backend. This
is expected to change, and the exact API is under discussion. But with
the current version, backends are expected to implement the next functions:

- plot (describe above, used for `Series.plot` and `DataFrame.plot`)
- hist_series and hist_frame (for `Series.hist` and `DataFrame.hist`)
- boxplot (`pandas.plotting.boxplot(df)` equivalent to `DataFrame.boxplot`)
- boxplot_frame and boxplot_frame_groupby
- register and deregister (register converters for the tick formats)
- Plots not called as `Series` and `DataFrame` methods:
  - table
  - andrews_curves
  - autocorrelation_plot
  - bootstrap_plot
  - lag_plot
  - parallel_coordinates
  - radviz
  - scatter_matrix

Use the code in pandas/plotting/_matplotib.py and
https://github.com/pyviz/hvplot as a reference on how to write a backend.

For the discussion about the API see
https://github.com/pandas-dev/pandas/issues/26747.
"""

# 从 pandas.plotting._core 导入必要的函数和类
from pandas.plotting._core import (
    PlotAccessor,                # 数据可视化访问器类，用于扩展数据结构的绘图能力
    boxplot,                    # 箱线图绘制函数
    boxplot_frame,              # DataFrame 箱线图绘制函数
    boxplot_frame_groupby,      # 分组后的 DataFrame 箱线图绘制函数
    hist_frame,                 # DataFrame 直方图绘制函数
    hist_series,                # Series 直方图绘制函数
)

# 从 pandas.plotting._misc 导入其他支持的绘图函数和相关功能
from pandas.plotting._misc import (
    andrews_curves,             # Andrews 曲线绘制函数
    autocorrelation_plot,       # 自相关图绘制函数
    bootstrap_plot,             # Bootstrap 绘制函数
    deregister as deregister_matplotlib_converters,  # 取消注册 Matplotlib 转换器函数
    lag_plot,                   # 滞后图绘制函数
    parallel_coordinates,       # 平行坐标绘制函数
    plot_params,                # 绘图参数设置函数
    radviz,                     # RadViz 绘制函数
    register as register_matplotlib_converters,      # 注册 Matplotlib 转换器函数
    scatter_matrix,             # 散点矩阵绘制函数
    table,                      # 表格绘制函数
)

# 定义公开的 API 接口，即可以被外部调用的函数和类名列表
__all__ = [
    "PlotAccessor",
    "boxplot",
    "boxplot_frame",
    "boxplot_frame_groupby",
    "hist_frame",
    "hist_series",
    "scatter_matrix",
    "radviz",
    "andrews_curves",
    "bootstrap_plot",
    "parallel_coordinates",
    "lag_plot",
    "autocorrelation_plot",
    "table",
    "plot_params",
    "register_matplotlib_converters",
    "deregister_matplotlib_converters",
]
```