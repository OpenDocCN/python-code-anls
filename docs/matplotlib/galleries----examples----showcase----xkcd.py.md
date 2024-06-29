# `D:\src\scipysrc\matplotlib\galleries\examples\showcase\xkcd.py`

```
"""
====
XKCD
====

Shows how to create an xkcd-like plot.
"""
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

# %%

with plt.xkcd():
    # Based on "Stove Ownership" from XKCD by Randall Munroe
    # https://xkcd.com/418/

    fig = plt.figure()  # 创建一个新的图形窗口
    ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))  # 在图形窗口中添加一个坐标轴，并指定位置和大小
    ax.spines[['top', 'right']].set_visible(False)  # 隐藏坐标轴的顶部和右侧边框
    ax.set_xticks([])  # 设置 x 轴刻度为空列表，即不显示 x 轴刻度
    ax.set_yticks([])  # 设置 y 轴刻度为空列表，即不显示 y 轴刻度
    ax.set_ylim([-30, 10])  # 设置 y 轴的数值范围为 [-30, 10]

    data = np.ones(100)  # 创建一个长度为 100 的全为 1 的数组
    data[70:] -= np.arange(30)  # 将数组后 30 个元素逐渐减小，从 1 到 -29

    ax.annotate(
        'THE DAY I REALIZED\nI COULD COOK BACON\nWHENEVER I WANTED',  # 给图中添加文本注释
        xy=(70, 1), arrowprops=dict(arrowstyle='->'), xytext=(15, -10))  # 设置注释的位置和箭头样式，以及文本偏移位置

    ax.plot(data)  # 绘制折线图，使用 data 数组的数据

    ax.set_xlabel('time')  # 设置 x 轴标签为 'time'
    ax.set_ylabel('my overall health')  # 设置 y 轴标签为 'my overall health'
    fig.text(
        0.5, 0.05,
        '"Stove Ownership" from xkcd by Randall Munroe',  # 在图形窗口中添加文本说明
        ha='center')  # 文本水平居中显示

# %%

with plt.xkcd():
    # Based on "The Data So Far" from XKCD by Randall Munroe
    # https://xkcd.com/373/

    fig = plt.figure()  # 创建一个新的图形窗口
    ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))  # 在图形窗口中添加一个坐标轴，并指定位置和大小
    ax.bar([0, 1], [0, 100], 0.25)  # 绘制柱状图，设置柱的位置和高度，以及柱的宽度
    ax.spines[['top', 'right']].set_visible(False)  # 隐藏坐标轴的顶部和右侧边框
    ax.xaxis.set_ticks_position('bottom')  # 设置 x 轴刻度位置在底部
    ax.set_xticks([0, 1])  # 设置 x 轴刻度位置
    ax.set_xticklabels(['CONFIRMED BY\nEXPERIMENT', 'REFUTED BY\nEXPERIMENT'])  # 设置 x 轴刻度标签
    ax.set_xlim([-0.5, 1.5])  # 设置 x 轴的数值范围为 [-0.5, 1.5]
    ax.set_yticks([])  # 设置 y 轴刻度为空列表，即不显示 y 轴刻度
    ax.set_ylim([0, 110])  # 设置 y 轴的数值范围为 [0, 110]

    ax.set_title("CLAIMS OF SUPERNATURAL POWERS")  # 设置图的标题为 "CLAIMS OF SUPERNATURAL POWERS"

    fig.text(
        0.5, 0.05,
        '"The Data So Far" from xkcd by Randall Munroe',  # 在图形窗口中添加文本说明
        ha='center')  # 文本水平居中显示

plt.show()  # 显示绘制的图形
```