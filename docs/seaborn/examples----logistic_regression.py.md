# `D:\src\scipysrc\seaborn\examples\logistic_regression.py`

```
"""
Faceted logistic regression
===========================

_thumb: .58, .5
"""
# 导入 seaborn 库，用于数据可视化
import seaborn as sns
# 设置 seaborn 的绘图风格为暗色网格
sns.set_theme(style="darkgrid")

# 加载示例数据集 Titanic
df = sns.load_dataset("titanic")

# 创建一个自定义调色板，用不同颜色表示不同性别
pal = dict(male="#6495ED", female="#F08080")

# 绘制图表，显示年龄和性别对生存概率的逻辑回归结果，分成男性和女性两列
g = sns.lmplot(x="age", y="survived", col="sex", hue="sex", data=df,
               palette=pal, y_jitter=.02, logistic=True, truncate=False)
# 设置 x 轴和 y 轴的限制范围
g.set(xlim=(0, 80), ylim=(-.05, 1.05))
```