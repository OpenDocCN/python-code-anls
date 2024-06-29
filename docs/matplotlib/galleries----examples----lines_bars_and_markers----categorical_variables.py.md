# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\categorical_variables.py`

```py
"""
==============================
Plotting categorical variables
==============================

You can pass categorical values (i.e. strings) directly as x- or y-values to
many plotting functions:
"""

# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt

# 定义一个字典数据，包含水果名和对应的数值
data = {'apple': 10, 'orange': 15, 'lemon': 5, 'lime': 20}

# 获取字典的键（水果名）和值（数值），分别存储在列表中
names = list(data.keys())
values = list(data.values())

# 创建一个包含三个子图的图像，每个子图的大小为 9x3 英寸，并且共享y轴
fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

# 在第一个子图上绘制柱状图，x轴为水果名，y轴为对应的数值
axs[0].bar(names, values)

# 在第二个子图上绘制散点图，x轴为水果名，y轴为对应的数值
axs[1].scatter(names, values)

# 在第三个子图上绘制折线图，x轴为水果名，y轴为对应的数值
axs[2].plot(names, values)

# 设置整个图像的标题
fig.suptitle('Categorical Plotting')

# %%
# This works on both Axes:

# 定义两个分类数据列表
cat = ["bored", "happy", "bored", "bored", "happy", "bored"]
dog = ["happy", "happy", "happy", "happy", "bored", "bored"]

# 定义活动列表
activity = ["combing", "drinking", "feeding", "napping", "playing", "washing"]

# 创建一个图像和一个轴对象
fig, ax = plt.subplots()

# 在轴对象上绘制两条折线，分别表示猫和狗的活动情况
ax.plot(activity, dog, label="dog")
ax.plot(activity, cat, label="cat")

# 添加图例到轴对象上，用于标识猫和狗
ax.legend()

# 显示图像
plt.show()
```