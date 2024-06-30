# `D:\src\scipysrc\seaborn\examples\horizontal_boxplot.py`

```
"""
Horizontal boxplot with observations
====================================

_thumb: .7, .37
"""
# 导入 seaborn 和 matplotlib.pyplot 库
import seaborn as sns
import matplotlib.pyplot as plt

# 设置 seaborn 的主题为 ticks 风格
sns.set_theme(style="ticks")

# 初始化图表，设置图表大小为 7x6 英寸，并使用对数 x 轴
f, ax = plt.subplots(figsize=(7, 6))
ax.set_xscale("log")

# 加载示例数据集 planets
planets = sns.load_dataset("planets")

# 绘制水平箱线图，显示轨道周期
sns.boxplot(
    planets, x="distance", y="method", hue="method",
    whis=[0, 100], width=.6, palette="vlag"
)

# 添加散点图以显示每个观测值
sns.stripplot(planets, x="distance", y="method", size=4, color=".3")

# 调整视觉呈现
ax.xaxis.grid(True)  # 在 x 轴上显示网格线
ax.set(ylabel="")  # 设置 y 轴标签为空字符串，即不显示 y 轴标签
sns.despine(trim=True, left=True)  # 美化图表外观，去掉右边和上边的轴线
```