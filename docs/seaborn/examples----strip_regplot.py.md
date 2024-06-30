# `D:\src\scipysrc\seaborn\examples\strip_regplot.py`

```
"""
Regression fit over a strip plot
================================

_thumb: .53, .5
"""
# 导入 seaborn 库
import seaborn as sns
# 设置 seaborn 的主题样式
sns.set_theme()

# 加载名为 "mpg" 的数据集
mpg = sns.load_dataset("mpg")

# 创建一个分类图（catplot），显示 cylinders 对 acceleration 的关系，
# 并根据 weight 进行着色，原生比例为 True，绘制在 z 轴上方
sns.catplot(
    data=mpg, x="cylinders", y="acceleration", hue="weight",
    native_scale=True, zorder=1
)

# 绘制一个回归图（regplot），显示 cylinders 对 acceleration 的回归关系，
# 禁用散点图，不进行截断，使用二次多项式拟合，颜色为浅灰色
sns.regplot(
    data=mpg, x="cylinders", y="acceleration",
    scatter=False, truncate=False, order=2, color=".2",
)
```