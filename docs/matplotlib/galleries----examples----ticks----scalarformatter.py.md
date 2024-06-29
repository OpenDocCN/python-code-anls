# `D:\src\scipysrc\matplotlib\galleries\examples\ticks\scalarformatter.py`

```py
"""
==========================
The default tick formatter
==========================

By default, tick labels are formatted using a `.ScalarFormatter`, which can be
configured via `~.axes.Axes.ticklabel_format`.  This example illustrates some
possible configurations:

- Default.
- ``useMathText=True``: Fancy formatting of mathematical expressions.
- ``useOffset=False``: Do not use offset notation; see
  `.ScalarFormatter.set_useOffset`.
"""

# 导入 matplotlib 库中的 pyplot 模块，并将其重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并将其重命名为 np
import numpy as np

# 生成一个包含从 0 到 1（不包括1），步长为0.01的数组作为x轴数据
x = np.arange(0, 1, .01)
# 创建一个9x9的子图布局，并设置总体尺寸为9x9英寸，使用“constrained”布局，设置子图之间的垂直间距为0.1
fig, axs = plt.subplots(
    3, 3, figsize=(9, 9), layout="constrained", gridspec_kw={"hspace": 0.1})

# 在每列的第一个子图中绘制图形，使用不同的函数和参数来展示数据
for col in axs.T:
    col[0].plot(x * 1e5 + 1e10, x * 1e-10 + 1e-5)
    col[1].plot(x * 1e5, x * 1e-4)
    col[2].plot(-x * 1e5 - 1e10, -x * 1e-5 - 1e-10)

# 针对每一行的第二列子图，设置tick labels使用数学文本格式化
for ax in axs[:, 1]:
    ax.ticklabel_format(useMathText=True)
# 针对每一行的第三列子图，设置tick labels不使用偏移量格式
for ax in axs[:, 2]:
    ax.ticklabel_format(useOffset=False)

# 更新全局绘图参数，设置标题的字体粗细为粗体，并且垂直位置偏移1.1
plt.rcParams.update({"axes.titleweight": "bold", "axes.titley": 1.1})
# 设置每个子图的标题
axs[0, 0].set_title("default settings")
axs[0, 1].set_title("useMathText=True")
axs[0, 2].set_title("useOffset=False")

# 显示图形
plt.show()
```