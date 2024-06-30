# `D:\src\scipysrc\seaborn\examples\marginal_ticks.py`

```
"""
Scatterplot with marginal ticks
===============================

_thumb: .66, .34
"""
# 导入 seaborn 库
import seaborn as sns
# 设置 seaborn 主题为白色，并启用颜色编码
sns.set_theme(style="white", color_codes=True)
# 加载 mpg 数据集
mpg = sns.load_dataset("mpg")

# 直接使用 JointGrid 创建自定义图表
# data=mpg 指定数据集，x="mpg" 和 y="acceleration" 分别指定 x 轴和 y 轴的数据
# space=0 设置子图间的间距为 0，ratio=17 设置子图的宽高比为 17
g = sns.JointGrid(data=mpg, x="mpg", y="acceleration", space=0, ratio=17)

# 在主图中绘制散点图，指定大小为 mpg["horsepower"]，点的大小范围为 (30, 120)
# 设置颜色为绿色，透明度为 0.6，不显示图例
g.plot_joint(sns.scatterplot, size=mpg["horsepower"], sizes=(30, 120),
             color="g", alpha=.6, legend=False)

# 在边缘图中绘制边际毛毯图，高度为 1，颜色为绿色，透明度为 0.6
g.plot_marginals(sns.rugplot, height=1, color="g", alpha=.6)
```