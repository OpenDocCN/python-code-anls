# `D:\src\scipysrc\seaborn\examples\palette_generation.py`

```
"""
Different cubehelix palettes
============================

_thumb: .4, .65
"""
# 导入所需的库
import numpy as np  # 导入 NumPy 库，用于处理数值数据
import seaborn as sns  # 导入 Seaborn 库，用于绘制统计图形
import matplotlib.pyplot as plt  # 导入 Matplotlib 库，用于绘制图表

# 设置 Seaborn 的主题样式为白色背景
sns.set_theme(style="white")

# 创建一个随机数生成器，种子为 50
rs = np.random.RandomState(50)

# 设置 Matplotlib 图表的布局，创建一个 3x3 的子图表格，每个子图大小为 9x9 英寸，并共享 x 和 y 轴
f, axes = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)

# 对每个子图进行操作，ax 为当前子图对象，s 为从 0 到 3 等间隔的 10 个值
for ax, s in zip(axes.flat, np.linspace(0, 3, 10)):

    # 使用 cubehelix 调色板创建 colormap 以在 kdeplot 中使用
    cmap = sns.cubehelix_palette(start=s, light=1, as_cmap=True)

    # 生成并绘制一个随机的双变量数据集
    x, y = rs.normal(size=(2, 50))
    sns.kdeplot(
        x=x, y=y,
        cmap=cmap, fill=True,
        clip=(-5, 5), cut=10,
        thresh=0, levels=15,
        ax=ax,
    )
    # 设置子图的坐标轴关闭
    ax.set_axis_off()

# 设置整个图表的 x 和 y 轴范围
ax.set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5))

# 调整子图的布局，使其填充整个图表区域，调整上、下、左、右的边距和子图之间的间隔
f.subplots_adjust(0, 0, 1, 1, .08, .08)
```