# `D:\src\scipysrc\seaborn\examples\scatterplot_sizes.py`

```
"""
Scatterplot with continuous hues and sizes
==========================================

_thumb: .51, .44

"""
# 导入 seaborn 库，并设置主题为白色网格样式
import seaborn as sns
sns.set_theme(style="whitegrid")

# 加载示例数据集 planets
planets = sns.load_dataset("planets")

# 使用 cubehelix 调色板生成颜色映射
cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)

# 创建关系图，显示 planets 数据集中距离与轨道周期的散点图
g = sns.relplot(
    data=planets,
    x="distance", y="orbital_period",  # 设置 x 轴为距离，y 轴为轨道周期
    hue="year", size="mass",            # 根据年份和质量设置散点颜色和大小
    palette=cmap, sizes=(10, 200),      # 设置颜色调色板和散点大小范围
)

# 设置 x 和 y 轴为对数刻度
g.set(xscale="log", yscale="log")

# 在 x 和 y 轴上绘制次要网格线，线宽为 0.25
g.ax.xaxis.grid(True, "minor", linewidth=.25)
g.ax.yaxis.grid(True, "minor", linewidth=.25)

# 去除图表左侧和底部的边框线
g.despine(left=True, bottom=True)
```