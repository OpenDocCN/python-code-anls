# `D:\src\scipysrc\seaborn\examples\pointplot_anova.py`

```
"""
Plotting a three-way ANOVA
==========================

_thumb: .42, .5
"""
# 导入 seaborn 库，用于绘图和数据可视化
import seaborn as sns
# 设置 seaborn 主题为白色网格风格
sns.set_theme(style="whitegrid")

# 加载示例数据集 exercise，该数据集可能包含运动相关的数据
exercise = sns.load_dataset("exercise")

# 使用 catplot 绘制点图，显示脉搏（pulse）在三个分类因子（time, kind, diet）下的关系
g = sns.catplot(
    data=exercise, x="time", y="pulse", hue="kind", col="diet",
    capsize=.2, palette="YlGnBu_d", errorbar="se",
    kind="point", height=6, aspect=.75,
)
# 移除图中左侧的轴线
g.despine(left=True)
```