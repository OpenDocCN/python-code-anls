# `D:\src\scipysrc\seaborn\examples\scatterplot_matrix.py`

```
"""
Scatterplot Matrix
==================

_thumb: .3, .2
"""
# 导入 seaborn 库，并设置图形主题为 "ticks"
import seaborn as sns
sns.set_theme(style="ticks")

# 使用 seaborn 提供的示例数据集 "penguins" 加载数据框
df = sns.load_dataset("penguins")

# 绘制数据框中数值型变量两两之间的散点图矩阵，根据物种分类着色
sns.pairplot(df, hue="species")
```