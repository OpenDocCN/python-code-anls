# `D:\src\scipysrc\seaborn\examples\pair_grid_with_kde.py`

```
"""
Paired density and scatterplot matrix
=====================================

_thumb: .5, .5
"""
# 导入 seaborn 库，并设置主题样式为白色
import seaborn as sns
sns.set_theme(style="white")

# 使用 seaborn 提供的样本数据集 "penguins" 加载数据框
df = sns.load_dataset("penguins")

# 创建一个 PairGrid 对象，设置对角线不共享 y 轴
g = sns.PairGrid(df, diag_sharey=False)

# 在 PairGrid 对象的上三角区域绘制散点图，点大小为 15
g.map_upper(sns.scatterplot, s=15)

# 在 PairGrid 对象的下三角区域绘制核密度估计图
g.map_lower(sns.kdeplot)

# 在 PairGrid 对象的对角线上绘制核密度估计图，线宽为 2
g.map_diag(sns.kdeplot, lw=2)
```