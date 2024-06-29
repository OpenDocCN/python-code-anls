# `D:\src\scipysrc\matplotlib\galleries\plot_types\stats\violin.py`

```py
"""
=============
violinplot(D)
=============
Make a violin plot.

See `~matplotlib.axes.Axes.violinplot`.
"""
# 导入需要的库
import matplotlib.pyplot as plt  # 导入matplotlib绘图库
import numpy as np  # 导入NumPy库用于数值计算

plt.style.use('_mpl-gallery')  # 使用自定义的matplotlib样式表

# 生成数据:
np.random.seed(10)  # 设置随机数种子以便结果可复现
D = np.random.normal((3, 5, 4), (0.75, 1.00, 0.75), (200, 3))
# 生成符合正态分布的三组数据，每组数据各自均值和标准差不同，总共200个数据点

# 绘图:
fig, ax = plt.subplots()  # 创建图形和坐标轴对象

vp = ax.violinplot(D, [2, 4, 6], widths=2,  # 绘制小提琴图，指定位置和宽度
                   showmeans=False, showmedians=False, showextrema=False)
# 隐藏小提琴图中的均值、中位数和极值点

# 样式设置:
for body in vp['bodies']:
    body.set_alpha(0.9)  # 设置小提琴的透明度为0.9，增强可视化效果
ax.set(xlim=(0, 8), xticks=np.arange(1, 8),  # 设置X轴的范围和刻度
       ylim=(0, 8), yticks=np.arange(1, 8))  # 设置Y轴的范围和刻度

plt.show()  # 显示绘制的图形
```