# `D:\src\scipysrc\seaborn\examples\scatterplot_categorical.py`

```
"""
Scatterplot with categorical variables
======================================

_thumb: .45, .45

"""
# 导入 seaborn 库
import seaborn as sns
# 设置 seaborn 的主题和调色板风格
sns.set_theme(style="whitegrid", palette="muted")

# 加载企鹅数据集（penguins dataset）
df = sns.load_dataset("penguins")

# 绘制分类散点图，展示每个观测值
ax = sns.swarmplot(data=df, x="body_mass_g", y="sex", hue="species")
# 设置 y 轴标签为空字符串，即不显示 y 轴标签
ax.set(ylabel="")
```