# `D:\src\scipysrc\matplotlib\galleries\plot_types\stats\boxplot_plot.py`

```py
"""
==========
boxplot(X)
==========
Draw a box and whisker plot.

See `~matplotlib.axes.Axes.boxplot`.
"""
# 导入 matplotlib.pyplot 库，并简写为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并简写为 np
import numpy as np

# 使用自定义样式 '_mpl-gallery'
plt.style.use('_mpl-gallery')

# 生成数据：
# 设置随机种子为 10，确保结果可重现
np.random.seed(10)
# 生成符合正态分布的随机数据，均值分别为 3、5、4，标准差分别为 1.25、1.00、1.25，形状为 (100, 3)
D = np.random.normal((3, 5, 4), (1.25, 1.00, 1.25), (100, 3))

# 绘制图形
# 创建一个图形和坐标轴对象
fig, ax = plt.subplots()
# 绘制箱线图，并设置相关参数：
# - 数据 D
# - 箱体的位置 positions=[2, 4, 6]
# - 箱体宽度 widths=1.5
# - 是否使用颜色填充 patch_artist=True
# - 不显示均值 showmeans=False
# - 不显示离群值 showfliers=False
# - 中位数线属性设置 medianprops={"color": "white", "linewidth": 0.5}
# - 箱体属性设置 boxprops={"facecolor": "C0", "edgecolor": "white", "linewidth": 0.5}
# - 涨落线属性设置 whiskerprops={"color": "C0", "linewidth": 1.5}
# - 箱顶帽属性设置 capprops={"color": "C0", "linewidth": 1.5}
VP = ax.boxplot(D, positions=[2, 4, 6], widths=1.5, patch_artist=True,
                showmeans=False, showfliers=False,
                medianprops={"color": "white", "linewidth": 0.5},
                boxprops={"facecolor": "C0", "edgecolor": "white",
                          "linewidth": 0.5},
                whiskerprops={"color": "C0", "linewidth": 1.5},
                capprops={"color": "C0", "linewidth": 1.5})

# 设置坐标轴范围和刻度
ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

# 显示图形
plt.show()
```