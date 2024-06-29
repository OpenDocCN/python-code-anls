# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\xcorr_acorr_demo.py`

```py
"""
===========================
Cross- and auto-correlation
===========================

Example use of cross-correlation (`~.Axes.xcorr`) and auto-correlation
(`~.Axes.acorr`) plots.
"""

# 导入 matplotlib 的 pyplot 模块，并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 模块，并重命名为 np
import numpy as np

# 设置随机种子以便结果可重现
np.random.seed(19680801)

# 生成两组服从标准正态分布的随机数序列，每组有 100 个数
x, y = np.random.randn(2, 100)

# 创建包含两个子图的图形窗口，共享 x 轴
fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)

# 在第一个子图上绘制交叉相关图（xcorr）
ax1.xcorr(x, y, usevlines=True, maxlags=50, normed=True, lw=2)
# 开启网格线显示
ax1.grid(True)
# 设置第一个子图的标题
ax1.set_title('Cross-correlation (xcorr)')

# 在第二个子图上绘制自相关图（acorr）
ax2.acorr(x, usevlines=True, normed=True, maxlags=50, lw=2)
# 开启网格线显示
ax2.grid(True)
# 设置第二个子图的标题
ax2.set_title('Auto-correlation (acorr)')

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.acorr` / `matplotlib.pyplot.acorr`
#    - `matplotlib.axes.Axes.xcorr` / `matplotlib.pyplot.xcorr`
```