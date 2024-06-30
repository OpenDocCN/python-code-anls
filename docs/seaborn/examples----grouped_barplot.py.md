# `D:\src\scipysrc\seaborn\examples\grouped_barplot.py`

```
"""
Grouped barplots
================

_thumb: .36, .5
"""
# 导入 seaborn 库
import seaborn as sns
# 设置 seaborn 的主题为白色网格
sns.set_theme(style="whitegrid")

# 加载企鹅数据集
penguins = sns.load_dataset("penguins")

# 绘制按物种和性别分组的嵌套条形图
g = sns.catplot(
    data=penguins, kind="bar",  # 使用 penguins 数据集，绘制条形图
    x="species", y="body_mass_g", hue="sex",  # 横轴为物种，纵轴为体重，按性别分组
    errorbar="sd",  # 添加标准差的误差线
    palette="dark", alpha=.6, height=6  # 使用深色调色板，设置透明度和图形高度
)
# 去除左侧的边框线
g.despine(left=True)
# 设置坐标轴标签
g.set_axis_labels("", "Body mass (g)")
# 设置图例标题为空
g.legend.set_title("")
```