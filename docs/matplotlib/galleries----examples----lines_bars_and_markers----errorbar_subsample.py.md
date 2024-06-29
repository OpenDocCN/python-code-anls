# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\errorbar_subsample.py`

```py
"""
====================
Errorbar subsampling
====================

The parameter *errorevery* of `.Axes.errorbar` can be used to draw error bars
only on a subset of data points. This is particularly useful if there are many
data points with similar errors.
"""

# 导入 matplotlib.pyplot 库，并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并重命名为 np
import numpy as np

# 创建示例数据
# 生成从 0.1 到 4（不包括）的数组，步长为 0.1
x = np.arange(0.1, 4, 0.1)
# 计算 y1 和 y2 对应的指数函数值
y1 = np.exp(-1.0 * x)
y2 = np.exp(-0.5 * x)

# 创建示例的变量误差条值
# 使用不同的函数来计算 y1 和 y2 的误差值
y1err = 0.1 + 0.1 * np.sqrt(x)
y2err = 0.1 + 0.1 * np.sqrt(x/2)

# 创建一个包含三个子图的图形窗口
fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharex=True,
                                    figsize=(12, 6))

# 在第一个子图 ax0 上绘制所有的误差条
ax0.set_title('all errorbars')
ax0.errorbar(x, y1, yerr=y1err)
ax0.errorbar(x, y2, yerr=y2err)

# 在第二个子图 ax1 上绘制每隔6个数据点的误差条
ax1.set_title('only every 6th errorbar')
ax1.errorbar(x, y1, yerr=y1err, errorevery=6)
ax1.errorbar(x, y2, yerr=y2err, errorevery=6)

# 在第三个子图 ax2 上绘制从第3个数据点开始，每隔6个数据点的误差条
ax2.set_title('second series shifted by 3')
ax2.errorbar(x, y1, yerr=y1err, errorevery=(0, 6))
ax2.errorbar(x, y2, yerr=y2err, errorevery=(3, 6))

# 设置整个图形的标题
fig.suptitle('Errorbar subsampling')
# 显示图形
plt.show()
```