# `D:\src\scipysrc\seaborn\examples\regression_marginals.py`

```
"""
Linear regression with marginal distributions
=============================================

_thumb: .65, .65
"""
# 导入 seaborn 库，用于数据可视化
import seaborn as sns
# 设置 seaborn 库的主题为 darkgrid
sns.set_theme(style="darkgrid")

# 加载示例数据集 tips，该数据集包含在 seaborn 库中
tips = sns.load_dataset("tips")
# 创建一个关联图，显示 total_bill 和 tip 之间的关系，同时包括边际分布和线性回归拟合线
g = sns.jointplot(x="total_bill", y="tip", data=tips,
                  kind="reg",  # 指定绘图类型为带有线性回归拟合的关联图
                  truncate=False,  # 不截断绘图范围
                  xlim=(0, 60), ylim=(0, 12),  # 设置 x 和 y 轴的显示范围
                  color="m",  # 设置图形的颜色为紫色（magenta）
                  height=7)  # 设置图形的高度为7英寸
```