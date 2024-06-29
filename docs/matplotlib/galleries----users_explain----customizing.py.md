# `D:\src\scipysrc\matplotlib\galleries\users_explain\customizing.py`

```
"""
.. redirect-from:: /users/customizing
.. redirect-from:: /tutorials/introductory/customizing

.. _customizing:

=====================================================
Customizing Matplotlib with style sheets and rcParams
=====================================================

Tips for customizing the properties and default styles of Matplotlib.

There are three ways to customize Matplotlib:

1. :ref:`Setting rcParams at runtime<customizing-with-dynamic-rc-settings>`.
2. :ref:`Using style sheets<customizing-with-style-sheets>`.
3. :ref:`Changing your matplotlibrc file<customizing-with-matplotlibrc-files>`.

Setting rcParams at runtime takes precedence over style sheets, style
sheets take precedence over :file:`matplotlibrc` files.

.. _customizing-with-dynamic-rc-settings:

Runtime rc settings
===================

You can dynamically change the default rc (runtime configuration)
settings in a python script or interactively from the python shell. All
rc settings are stored in a dictionary-like variable called
:data:`matplotlib.rcParams`, which is global to the matplotlib package.
See `matplotlib.rcParams` for a full list of configurable rcParams.
rcParams can be modified directly, for example:
"""

# 导入需要的模块和库
from cycler import cycler  # 导入cycler模块中的cycler函数

import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块，并重命名为plt
import numpy as np  # 导入numpy模块，并重命名为np

import matplotlib as mpl  # 导入matplotlib模块，并重命名为mpl

# 设置全局默认的线条宽度和线条风格
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.linestyle'] = '--'

data = np.random.randn(50)  # 生成50个随机数的数组
plt.plot(data)  # 绘制数据

# %%
# 注意，在修改通常的`~.Axes.plot`颜色时，需要修改*axes*的*prop_cycle*属性：
# 修改轴对象的prop_cycle属性，设置颜色循环为红、绿、蓝、黄
mpl.rcParams['axes.prop_cycle'] = cycler(color=['r', 'g', 'b', 'y'])
plt.plot(data)  # 绘制图形，第一种颜色为红色

# %%
# Matplotlib还提供了几个方便的函数来修改rc设置。`matplotlib.rc`可以用来同时修改
# 一个组内的多个设置，使用关键字参数：
# 设置线条组的设置，设置线宽为4，线条风格为'-.'
mpl.rc('lines', linewidth=4, linestyle='-.')
plt.plot(data)

# %%
# 临时的rc设置
# ---------------------
#
# 可以使用`matplotlib.rc_context`上下文管理器临时改变:data:`matplotlib.rcParams`对象：
# 在上下文中临时设置线条宽度为2，线条风格为':'
with mpl.rc_context({'lines.linewidth': 2, 'lines.linestyle': ':'}):
    plt.plot(data)

# %%
# `matplotlib.rc_context`也可以作为装饰器在函数内修改默认设置：
# 在函数内部使用装饰器，临时设置线条宽度为3，线条风格为'-'
@mpl.rc_context({'lines.linewidth': 3, 'lines.linestyle': '-'})
def plotting_function():
    plt.plot(data)

plotting_function()

# %%
# `matplotlib.rcdefaults`会恢复标准的Matplotlib默认设置。
#
# 在设置rcParams值时有一定的验证程度，请参阅:mod:`matplotlib.rcsetup`了解详情。

# %%
# .. _customizing-with-style-sheets:
#
# 使用样式表
# ==================
#
# 另一种改变绘图的视觉外观的方法是在所谓的样式表中设置rcParams，并使用`matplotlib.style.use`导入样式表。通过这种方式可以轻松切换
# 导入 matplotlib.pyplot 库，用于绘图和样式控制
import matplotlib.pyplot as plt

# 设置当前的绘图样式为 'ggplot'，模拟 R 语言中流行的 ggplot 包的美学风格
plt.style.use('ggplot')

# %%
# 打印当前可用的所有样式
print(plt.style.available)

# %%
# 定义自定义样式
# -----------------------
#
# 您可以创建自定义样式，并通过调用 `.style.use` 使用它们，提供样式表的路径或 URL。
#
# 例如，您可以创建 `./images/presentation.mplstyle`，包含以下内容：
#
#    axes.titlesize : 24
#    axes.labelsize : 20
#    lines.linewidth : 3
#    lines.markersize : 10
#    xtick.labelsize : 16
#    ytick.labelsize : 16
#
# 然后，当您想要将为论文设计的图形适应为演示风格时，只需添加：
#
#    >>> import matplotlib.pyplot as plt
#    >>> plt.style.use('./images/presentation.mplstyle')
#
#
# 分发样式
# -------------------
#
# 您可以将样式表包含到标准可导入的 Python 包中（例如可以在 PyPI 上分发）。如果您的包能够以 `import mypackage` 的方式导入，
# 并且在 `mypackage/__init__.py` 模块中添加了 `mypackage/presentation.mplstyle` 样式表，则可以通过
# `plt.style.use("mypackage.presentation")` 使用它。还支持子包（例如 `dotted.package.name`）。
#
# 或者，您可以通过将 `mpl_configdir/stylelib` 目录下的 `<style-name>.mplstyle` 文件添加到 Matplotlib 的样式路径中，
# 使 Matplotlib 知道您的自定义样式表。然后可以通过调用 `style.use(<style-name>)` 加载您的自定义样式表。
# 默认情况下，`mpl_configdir` 应该是 `~/.config/matplotlib`，但您可以使用 `matplotlib.get_configdir()` 检查其位置；
# 您可能需要创建此目录。您还可以通过设置 :envvar:`MPLCONFIGDIR` 环境变量更改 Matplotlib 查找 stylelib/ 文件夹的目录，
# 请参阅 :ref:`locating-matplotlib-config-dir`。
#
# 请注意，如果样式具有相同的名称，则在 `mpl_configdir/stylelib` 中的自定义样式表将覆盖由 Matplotlib 定义的样式表。
#
# 导入matplotlib.pyplot库，用于绘图操作
import matplotlib.pyplot as plt

# 使用'dark_background'风格上下文管理器，设置临时的绘图风格
with plt.style.context('dark_background'):
    # 绘制正弦函数图像，使用红色线条连接带有圆点标记的数据点
    plt.plot(np.sin(np.linspace(0, 2 * np.pi)), 'r-o')

# 显示绘制的图像
plt.show()
# ``style.use('<path>/<style-name>.mplstyle')``, 使用指定路径下的样式表来设置 matplotlib 的样式。
# 样式表中的设置优先于 :file:`matplotlibrc` 文件中的设置。

# 要显示当前活动的 :file:`matplotlibrc` 文件的加载位置，可以执行以下操作：
#
#   >>> import matplotlib
#   >>> matplotlib.matplotlib_fname()
#   '/home/foo/.config/matplotlib/matplotlibrc'
#
# 查看下面的示例 :ref:`matplotlibrc file<matplotlibrc-sample>`，以及 `matplotlib.rcParams`
# 获取完整可配置的 rcParams 列表。
#
# .. _matplotlibrc-sample:
#
# 默认的 :file:`matplotlibrc` 文件
# -------------------------------------
#
# .. literalinclude:: ../../../lib/matplotlib/mpl-data/matplotlibrc
#
#
# .. _ggplot: https://ggplot2.tidyverse.org/
# .. _R: https://www.r-project.org/
```