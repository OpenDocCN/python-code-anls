# `D:\src\scipysrc\seaborn\examples\paired_pointplots.py`

```
"""
Paired categorical plots
========================

"""
# 导入 seaborn 库并设置主题为白色网格风格
import seaborn as sns
sns.set_theme(style="whitegrid")

# 加载示例数据集 Titanic
titanic = sns.load_dataset("titanic")

# 设置一个绘图网格，将生存概率与多个变量进行对比
g = sns.PairGrid(titanic, y_vars="survived",
                 x_vars=["class", "sex", "who", "alone"],
                 height=5, aspect=.5)

# 在每个子图上绘制 seaborn 的点图（pointplot）
g.map(sns.pointplot, color="xkcd:plum")

# 设置 y 轴的范围在 0 到 1 之间
g.set(ylim=(0, 1))

# 去除图形的左侧脊柱
sns.despine(fig=g.fig, left=True)
```