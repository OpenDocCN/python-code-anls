# `D:\src\scipysrc\seaborn\examples\palette_choices.py`

```
"""
Color palette choices
=====================

"""
# 导入所需的库
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 设置 seaborn 的主题和绘图上下文
sns.set_theme(style="white", context="talk")

# 创建一个随机数生成器，用于生成随机数据
rs = np.random.RandomState(8)

# 设置 matplotlib 图形的布局，创建一个包含三个子图的图形
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 5), sharex=True)

# 生成一些顺序数据，并在第一个子图中绘制条形图
x = np.array(list("ABCDEFGHIJ"))
y1 = np.arange(1, 11)
sns.barplot(x=x, y=y1, hue=x, palette="rocket", ax=ax1)
ax1.axhline(0, color="k", clip_on=False)  # 在第一个子图中添加一条水平线
ax1.set_ylabel("Sequential")  # 设置第一个子图的纵轴标签

# 将数据居中以生成分散数据，并在第二个子图中绘制条形图
y2 = y1 - 5.5
sns.barplot(x=x, y=y2, hue=x, palette="vlag", ax=ax2)
ax2.axhline(0, color="k", clip_on=False)  # 在第二个子图中添加一条水平线
ax2.set_ylabel("Diverging")  # 设置第二个子图的纵轴标签

# 随机重新排序数据以生成定性数据，并在第三个子图中绘制条形图
y3 = rs.choice(y1, len(y1), replace=False)
sns.barplot(x=x, y=y3, hue=x, palette="deep", ax=ax3)
ax3.axhline(0, color="k", clip_on=False)  # 在第三个子图中添加一条水平线
ax3.set_ylabel("Qualitative")  # 设置第三个子图的纵轴标签

# 完成绘图，去除图形下方的边界
sns.despine(bottom=True)

# 设置所有子图的纵轴刻度为空
plt.setp(f.axes, yticks=[])

# 调整子图之间的布局，使它们之间的垂直间距增加
plt.tight_layout(h_pad=2)
```