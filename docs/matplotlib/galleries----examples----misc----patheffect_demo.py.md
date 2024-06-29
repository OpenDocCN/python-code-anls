# `D:\src\scipysrc\matplotlib\galleries\examples\misc\patheffect_demo.py`

```py
"""
===============
Patheffect Demo
===============

"""
# 导入 matplotlib 的 pyplot 模块，并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 模块，并重命名为 np
import numpy as np

# 导入 matplotlib 的 patheffects 模块
from matplotlib import patheffects

# 创建一个包含三个子图的图形窗口，大小为 8x3
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))

# 在第一个子图(ax1)上显示一个简单的图像
ax1.imshow([[1, 2], [2, 3]])
# 在图中添加一个带箭头的注释文本，带有路径效果（描边和阴影）
txt = ax1.annotate("test", (1., 1.), (0., 0),
                   arrowprops=dict(arrowstyle="->",
                                   connectionstyle="angle3", lw=2),
                   size=20, ha="center",
                   path_effects=[patheffects.withStroke(linewidth=3,
                                                        foreground="w")])
# 设置箭头路径效果，包括描边和正常显示
txt.arrow_patch.set_path_effects([
    patheffects.Stroke(linewidth=5, foreground="w"),
    patheffects.Normal()])

# 在第一个子图(ax1)上打开网格线，并添加路径效果（描边）
pe = [patheffects.withStroke(linewidth=3,
                             foreground="w")]
ax1.grid(True, linestyle="-", path_effects=pe)

# 创建一个 5x5 的二维数组
arr = np.arange(25).reshape((5, 5))
# 在第二个子图(ax2)上显示这个二维数组
ax2.imshow(arr)
# 在二维数组的等高线上添加路径效果（描边）
cntr = ax2.contour(arr, colors="k")
cntr.set(path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])

# 在等高线上添加标签，并设置标签的路径效果（描边）
clbls = ax2.clabel(cntr, fmt="%2.0f", use_clabeltext=True)
plt.setp(clbls, path_effects=[
    patheffects.withStroke(linewidth=3, foreground="w")])

# 在第三个子图(ax3)上绘制一条直线，并在图例中添加阴影效果
p1, = ax3.plot([0, 1], [0, 1])
leg = ax3.legend([p1], ["Line 1"], fancybox=True, loc='upper left')
leg.legendPatch.set_path_effects([patheffects.withSimplePatchShadow()])

# 显示图形窗口
plt.show()
```