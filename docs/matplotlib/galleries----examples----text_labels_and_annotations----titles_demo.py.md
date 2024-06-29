# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\titles_demo.py`

```py
"""
=================
Title positioning
=================

Matplotlib可以将标题显示为居中、靠左或靠右对齐到一组Axes对象的位置上。

"""
# 导入Matplotlib的pyplot模块，用于绘图
import matplotlib.pyplot as plt

# 绘制简单的折线图，横坐标为0到9
plt.plot(range(10))

# 设置居中标题
plt.title('Center Title')
# 设置靠左标题，使用loc参数指定位置为'left'
plt.title('Left Title', loc='left')
# 设置靠右标题，使用loc参数指定位置为'right'
plt.title('Right Title', loc='right')

# 显示图形
plt.show()

# %%
# 垂直位置会自动选择，以避免顶部x轴上的装饰（如标签和刻度）：

# 创建包含两个子图的图形对象，使用constrained_layout参数布局
fig, axs = plt.subplots(1, 2, layout='constrained')

# 左侧子图设置
ax = axs[0]
ax.plot(range(10))
ax.xaxis.set_label_position('top')  # 设置x轴标签位置在顶部
ax.set_xlabel('X-label')  # 设置x轴标签
ax.set_title('Center Title')  # 设置居中标题

# 右侧子图设置
ax = axs[1]
ax.plot(range(10))
ax.xaxis.set_label_position('top')  # 设置x轴标签位置在顶部
ax.xaxis.tick_top()  # 设置x轴刻度在顶部显示
ax.set_xlabel('X-label')  # 设置x轴标签
ax.set_title('Center Title')  # 设置居中标题

# 显示图形
plt.show()

# %%
# 可以通过手动指定标题的y关键字参数或设置rcParams中的axes.titley来关闭自动定位。

# 创建包含两个子图的图形对象，使用constrained_layout参数布局
fig, axs = plt.subplots(1, 2, layout='constrained')

# 左侧子图设置
ax = axs[0]
ax.plot(range(10))
ax.xaxis.set_label_position('top')  # 设置x轴标签位置在顶部
ax.set_xlabel('X-label')  # 设置x轴标签
# 手动指定标题位置y=1.0，pad=-14用于调整标题的垂直位置和间距
ax.set_title('Manual y', y=1.0, pad=-14)

# 设置rcParams中的axes.titley为1.0，使用axes-relative坐标
plt.rcParams['axes.titley'] = 1.0    # y使用axes-relative坐标
plt.rcParams['axes.titlepad'] = -14  # pad使用points单位
# 右侧子图设置
ax = axs[1]
ax.plot(range(10))
ax.set_xlabel('X-label')  # 设置x轴标签
ax.set_title('rcParam y')  # 设置rcParam y标题

# 显示图形
plt.show()
```