# `D:\src\scipysrc\seaborn\examples\multiple_regression.py`

```
"""
Multiple linear regression
==========================

_thumb: .45, .45
"""
# 导入 seaborn 库，并设置其主题
import seaborn as sns
sns.set_theme()

# 加载企鹅数据集 'penguins'
penguins = sns.load_dataset("penguins")

# 绘制线性回归模型，显示喙长度（bill_length_mm）与喙深度（bill_depth_mm）的关系，
# 根据物种（species）进行颜色区分，图形高度为 5
g = sns.lmplot(
    data=penguins,
    x="bill_length_mm", y="bill_depth_mm", hue="species",
    height=5
)

# 设置更具信息性的坐标轴标签，替换默认标签
g.set_axis_labels("Snoot length (mm)", "Snoot depth (mm)")
```