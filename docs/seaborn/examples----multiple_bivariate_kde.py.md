# `D:\src\scipysrc\seaborn\examples\multiple_bivariate_kde.py`

```
"""
Multiple bivariate KDE plots
============================

_thumb: .6, .45
"""
# 导入 seaborn 和 matplotlib.pyplot 库
import seaborn as sns
import matplotlib.pyplot as plt

# 设置 seaborn 的主题风格为 'darkgrid'
sns.set_theme(style="darkgrid")

# 从 seaborn 自带数据集中加载鸢尾花数据集 'iris'
iris = sns.load_dataset("iris")

# 创建一个包含一个子图的 figure 对象，并设置尺寸为 8x8 英寸
f, ax = plt.subplots(figsize=(8, 8))
# 设置子图的纵横比为相等，即使得 x 和 y 轴单位长度相等
ax.set_aspect("equal")

# 绘制每个双变量密度的轮廓图
sns.kdeplot(
    data=iris.query("species != 'versicolor'"),  # 从鸢尾花数据集中选择非 'versicolor' 种类的数据
    x="sepal_width",  # 横轴使用 sepal_width（花萼宽度）
    y="sepal_length",  # 纵轴使用 sepal_length（花萼长度）
    hue="species",  # 根据不同的物种着色
    thresh=.1,  # 设置密度轮廓的阈值
)
```