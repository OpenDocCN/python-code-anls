# `D:\src\scipysrc\matplotlib\galleries\examples\pie_and_polar_charts\polar_bar.py`

```
"""
=======================
Bar chart on polar axis
=======================

Demo of bar plot on a polar axis.
"""
# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，用于数值计算
import numpy as np

# 设置随机种子以便结果可重复
np.random.seed(19680801)

# 计算扇形的角度
N = 20
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
# 计算每个扇形的半径，随机生成
radii = 10 * np.random.rand(N)
# 计算每个扇形的宽度，随机生成
width = np.pi / 4 * np.random.rand(N)
# 根据半径生成颜色，使用 viridis 色图
colors = plt.cm.viridis(radii / 10.)

# 创建极坐标子图对象
ax = plt.subplot(projection='polar')
# 绘制极坐标柱状图
ax.bar(theta, radii, width=width, bottom=0.0, color=colors, alpha=0.5)

# 展示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.bar` / `matplotlib.pyplot.bar`
#    - `matplotlib.projections.polar`
```