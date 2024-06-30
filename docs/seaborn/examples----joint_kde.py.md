# `D:\src\scipysrc\seaborn\examples\joint_kde.py`

```
"""
Joint kernel density estimate
=============================

_thumb: .6, .4
"""
# 导入 seaborn 库，并设置样式为 "ticks"
import seaborn as sns
sns.set_theme(style="ticks")

# 加载企鹅数据集（penguins dataset）
penguins = sns.load_dataset("penguins")

# 使用核密度估计展示两个变量的联合分布
g = sns.jointplot(
    data=penguins,                      # 使用 penguins 数据集
    x="bill_length_mm", y="bill_depth_mm",  # 设置 x 轴和 y 轴变量
    hue="species",                      # 根据物种类别着色
    kind="kde",                         # 使用核密度估计方式展示联合分布
)
```