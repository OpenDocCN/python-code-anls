# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\stem_plot.py`

```py
"""
=========
Stem Plot
=========

`~.pyplot.stem` plots vertical lines from a baseline to the y-coordinate and
places a marker at the tip.
"""

# 导入 matplotlib.pyplot 库，并命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并命名为 np
import numpy as np

# 生成一个包含 41 个元素的数组，元素在区间 [0.1, 2*pi] 均匀分布
x = np.linspace(0.1, 2 * np.pi, 41)
# 计算 y 值，y = exp(sin(x))
y = np.exp(np.sin(x))

# 绘制 stem 图，显示出图形
plt.stem(x, y)
plt.show()

# %%
#
# 可以通过 *bottom* 参数来调整基线的位置。
# 参数 *linefmt*、*markerfmt* 和 *basefmt* 控制图形的基本格式属性。
# 不过，与 `~.pyplot.plot` 不同的是，并非所有属性都可以通过关键字参数进行配置。
# 若要进行更高级的控制，请调整 `.pyplot` 返回的线对象。

# 绘制带有自定义格式的 stem 图形，包括灰色线和菱形标记
markerline, stemlines, baseline = plt.stem(
    x, y, linefmt='grey', markerfmt='D', bottom=1.1)
# 设置标记的填充颜色为无色
markerline.set_markerfacecolor('none')
plt.show()

# %%
#
# .. admonition:: References
#
#    此示例展示了以下函数、方法、类和模块的使用：
#
#    - `matplotlib.axes.Axes.stem` / `matplotlib.pyplot.stem`
```