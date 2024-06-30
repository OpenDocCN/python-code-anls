# `D:\src\scipysrc\seaborn\examples\jitter_stripplot.py`

```
"""
Conditional means with observations
===================================

"""
# 导入 seaborn 和 matplotlib.pyplot 库
import seaborn as sns
import matplotlib.pyplot as plt

# 设定 seaborn 的主题风格为白色网格
sns.set_theme(style="whitegrid")

# 加载鸢尾花数据集
iris = sns.load_dataset("iris")

# 将数据集 "melt"（融化）为长格式或整洁格式的表示，以便分析
iris = iris.melt(id_vars="species", var_name="measurement")

# 初始化图形
f, ax = plt.subplots()
# 去除图形中的上部和右部边框线
sns.despine(bottom=True, left=True)

# 使用 scatterplot 显示每个观测值
sns.stripplot(
    data=iris, x="value", y="measurement", hue="species",
    dodge=True, alpha=.25, zorder=1, legend=False,
)

# 显示条件均值，通过调整每个点图在条带中的位置来对齐
sns.pointplot(
    data=iris, x="value", y="measurement", hue="species",
    dodge=.8 - .8 / 3, palette="dark", errorbar=None,
    markers="d", markersize=4, linestyle="none",
)

# 改进图例显示
sns.move_legend(
    ax, loc="lower right", ncol=3, frameon=True, columnspacing=1, handletextpad=0,
)
```