# `D:\src\scipysrc\matplotlib\galleries\examples\statistics\boxplot_demo.py`

```py
"""
========
Boxplots
========

Visualizing boxplots with matplotlib.

The following examples show off how to visualize boxplots with
Matplotlib. There are many options to control their appearance and
the statistics that they use to summarize the data.

.. redirect-from:: /gallery/pyplots/boxplot_demo_pyplot
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于绘图
import numpy as np  # 导入 NumPy 数学库

from matplotlib.patches import Polygon  # 导入 matplotlib 的 Polygon 模块，用于绘制多边形

# Fixing random state for reproducibility
np.random.seed(19680801)  # 设置随机数种子，以便结果可重现

# fake up some data
spread = np.random.rand(50) * 100  # 生成50个 [0, 100) 之间的随机数，并放大100倍
center = np.ones(25) * 50  # 生成一个长度为25的数组，每个元素为50
flier_high = np.random.rand(10) * 100 + 100  # 生成10个 [100, 200) 之间的随机数
flier_low = np.random.rand(10) * -100  # 生成10个 [-100, 0) 之间的随机数
data = np.concatenate((spread, center, flier_high, flier_low))  # 将以上数据合并成一个数组

fig, axs = plt.subplots(2, 3)  # 创建一个2行3列的子图布局

# basic plot
axs[0, 0].boxplot(data)  # 在第一个子图中绘制箱线图
axs[0, 0].set_title('basic plot')  # 设置子图标题为 'basic plot'

# notched plot
axs[0, 1].boxplot(data, notch=True)  # 在第二个子图中绘制有缺口的箱线图
axs[0, 1].set_title('notched plot')  # 设置子图标题为 'notched plot'

# change outlier point symbols
axs[0, 2].boxplot(data, sym='gD')  # 在第三个子图中绘制箱线图，并将异常值点的符号改为绿色的菱形
axs[0, 2].set_title('change outlier\npoint symbols')  # 设置子图标题为 'change outlier\npoint symbols'，使用换行符分行显示

# don't show outlier points
axs[1, 0].boxplot(data, sym='')  # 在第四个子图中绘制箱线图，但不显示异常值点
axs[1, 0].set_title("don't show\noutlier points")  # 设置子图标题为 "don't show\noutlier points"，使用换行符分行显示

# horizontal boxes
axs[1, 1].boxplot(data, sym='rs', orientation='horizontal')  # 在第五个子图中绘制水平方向的箱线图，并将异常值点的符号改为红色的正方形
axs[1, 1].set_title('horizontal boxes')  # 设置子图标题为 'horizontal boxes'

# change whisker length
axs[1, 2].boxplot(data, sym='rs', orientation='horizontal', whis=0.75)  # 在第六个子图中绘制水平方向的箱线图，并设置箱须长度系数为0.75
axs[1, 2].set_title('change whisker length')  # 设置子图标题为 'change whisker length'

fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                    hspace=0.4, wspace=0.3)  # 调整子图的布局参数

# fake up some more data
spread = np.random.rand(50) * 100  # 生成50个 [0, 100) 之间的随机数，并放大100倍
center = np.ones(25) * 40  # 生成一个长度为25的数组，每个元素为40
flier_high = np.random.rand(10) * 100 + 100  # 生成10个 [100, 200) 之间的随机数
flier_low = np.random.rand(10) * -100  # 生成10个 [-100, 0) 之间的随机数
d2 = np.concatenate((spread, center, flier_high, flier_low))  # 将以上数据合并成一个数组
# Making a 2-D array only works if all the columns are the
# same length.  If they are not, then use a list instead.
# This is actually more efficient because boxplot converts
# a 2-D array into a list of vectors internally anyway.
data = [data, d2, d2[::2]]  # 创建一个包含多个数组的列表，用于绘制多个箱线图

# Multiple box plots on one Axes
fig, ax = plt.subplots()  # 创建一个新的图形和一个轴对象
ax.boxplot(data)  # 在轴对象上绘制多个箱线图

plt.show()  # 显示图形


# %%
# Below we'll generate data from five different probability distributions,
# each with different characteristics. We want to play with how an IID
# bootstrap resample of the data preserves the distributional
# properties of the original sample, and a boxplot is one visual tool
# to make this assessment

random_dists = ['Normal(1, 1)', 'Lognormal(1, 1)', 'Exp(1)', 'Gumbel(6, 4)',
                'Triangular(2, 9, 11)']
N = 500  # 设定样本大小为500

norm = np.random.normal(1, 1, N)  # 从正态分布 N(1, 1) 中生成500个随机数
logn = np.random.lognormal(1, 1, N)  # 从对数正态分布 LogN(1, 1) 中生成500个随机数
expo = np.random.exponential(1, N)  # 从指数分布 Exp(1) 中生成500个随机数
gumb = np.random.gumbel(6, 4, N)  # 从古贝尔分布 Gumbel(6, 4) 中生成500个随机数
tria = np.random.triangular(2, 9, 11, N)  # 从三角分布 Triangular(2, 9, 11) 中生成500个随机数

# Generate some random indices that we'll use to resample the original data
# arrays. For code brevity, just use the same random indices for each array
bootstrap_indices = np.random.randint(0, N, N)  # 生成用于自助法重采样的随机索引
data = [
    norm, norm[bootstrap_indices],  # 将正态分布数据及其自助法重采样结果加入列表
    logn, logn[bootstrap_indices],  # 将对数正态分布数据及其自助法重采样结果加入列表
    expo, expo[bootstrap_indices],  # 将指数分布数据及其自助法重采样结果加入列表
    gumb, gumb[bootstrap_indices],
    tria, tria[bootstrap_indices],



    # 使用布尔索引从变量 gumb 中选择特定的元素子集，将其赋值给变量 gumb
    gumb, gumb[bootstrap_indices],
    # 使用布尔索引从变量 tria 中选择特定的元素子集，将其赋值给变量 tria
    tria, tria[bootstrap_indices],


这段代码是在使用布尔索引（即使用布尔值作为索引来选择数组或列表中的元素）从变量 `gumb` 和 `tria` 中选择特定的元素子集，并将结果重新赋值给这两个变量。
# 创建一个包含两个子图的图形，一个10x6大小的子图，并设置窗口标题为'A Boxplot Example'
fig, ax1 = plt.subplots(figsize=(10, 6))
fig.canvas.manager.set_window_title('A Boxplot Example')
# 调整子图的边界，使得图形内容在窗口中的位置合适
fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

# 使用数据创建一个箱线图，并设置其属性
bp = ax1.boxplot(data, notch=False, sym='+', orientation='vertical', whis=1.5)
plt.setp(bp['boxes'], color='black')   # 设置箱体的颜色为黑色
plt.setp(bp['whiskers'], color='black')  # 设置箱线的颜色为黑色
plt.setp(bp['fliers'], color='red', marker='+')  # 设置离群值的颜色为红色，标记为+

# 添加水平网格到图中，颜色很浅以免干扰数据的可读性
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)

# 设置图的各种属性：隐藏网格在绘图对象后面，设置标题、X轴标签和Y轴标签
ax1.set(
    axisbelow=True,  # 隐藏网格在绘图对象后面
    title='Comparison of IID Bootstrap Resampling Across Five Distributions',  # 设置图的标题
    xlabel='Distribution',  # 设置X轴标签
    ylabel='Value',  # 设置Y轴标签
)

# 现在用指定的颜色填充箱体
box_colors = ['darkkhaki', 'royalblue']
num_boxes = len(data)
medians = np.empty(num_boxes)
for i in range(num_boxes):
    box = bp['boxes'][i]
    box_x = []
    box_y = []
    for j in range(5):
        box_x.append(box.get_xdata()[j])
        box_y.append(box.get_ydata()[j])
    box_coords = np.column_stack([box_x, box_y])
    # 使用交替的颜色填充箱体
    ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))
    # 绘制中位线，覆盖填充的区域
    med = bp['medians'][i]
    median_x = []
    median_y = []
    for j in range(2):
        median_x.append(med.get_xdata()[j])
        median_y.append(med.get_ydata()[j])
        ax1.plot(median_x, median_y, 'k')  # 绘制中位线
    medians[i] = median_y[0]
    # 最后，覆盖绘制样本平均值，水平对齐在每个箱子的中心
    ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
             color='w', marker='*', markeredgecolor='k')

# 设置坐标轴的范围和标签
ax1.set_xlim(0.5, num_boxes + 0.5)
top = 40
bottom = -5
ax1.set_ylim(bottom, top)
ax1.set_xticklabels(np.repeat(random_dists, 2),
                    rotation=45, fontsize=8)

# 由于Y轴刻度在不同样本间不同，比较中位数差异可能比较困难。为了辅助比较，添加上部X轴刻度标签，显示样本的中位数值（保留两位小数）
pos = np.arange(num_boxes) + 1
upper_labels = [str(round(s, 2)) for s in medians]
weights = ['bold', 'semibold']
for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
    k = tick % 2
    ax1.text(pos[tick], .95, upper_labels[tick],
             transform=ax1.get_xaxis_transform(),
             horizontalalignment='center', size='x-small',
             weight=weights[k], color=box_colors[k])

# 最后，添加一个基本的图例说明
fig.text(0.80, 0.08, f'{N} Random Numbers',
         backgroundcolor=box_colors[0], color='black', weight='roman',
         size='x-small')
fig.text(0.80, 0.045, 'IID Bootstrap Resample',
         backgroundcolor=box_colors[1],
         color='white', weight='roman', size='x-small')

# 在图形上添加文本，显示"IID Bootstrap Resample"，位于特定位置 (0.80, 0.045)，背景颜色为预定义的盒子颜色的第二个元素，文本颜色为白色，字体粗细为正常，大小为特小号。


fig.text(0.80, 0.015, '*', color='white', backgroundcolor='silver',
         weight='roman', size='medium')

# 在图形上添加文本，显示一个星号，位于特定位置 (0.80, 0.015)，文本颜色为白色，背景颜色为银色，字体粗细为正常，大小为中号。


fig.text(0.815, 0.013, ' Average Value', color='black', weight='roman',
         size='x-small')

# 在图形上添加文本，显示" Average Value"，位于特定位置 (0.815, 0.013)，文本颜色为黑色，字体粗细为正常，大小为特小号。


plt.show()

# 显示图形。


def fake_bootstrapper(n):
    """
    This is just a placeholder for the user's method of
    bootstrapping the median and its confidence intervals.

    Returns an arbitrary median and confidence interval packed into a tuple.
    """
    if n == 1:
        med = 0.1
        ci = (-0.25, 0.25)
    else:
        med = 0.2
        ci = (-0.35, 0.50)
    return med, ci

# 定义一个虚拟的函数 `fake_bootstrapper`，用于模拟用户使用自定义方法进行中位数和置信区间的自助法（bootstrap）估计。根据输入的 `n` 返回一个任意的中位数和置信区间的元组。


inc = 0.1
e1 = np.random.normal(0, 1, size=500)
e2 = np.random.normal(0, 1, size=500)
e3 = np.random.normal(0, 1 + inc, size=500)
e4 = np.random.normal(0, 1 + 2*inc, size=500)

treatments = [e1, e2, e3, e4]
med1, ci1 = fake_bootstrapper(1)
med2, ci2 = fake_bootstrapper(2)
medians = [None, None, med1, med2]
conf_intervals = [None, None, ci1, ci2]

# 初始化四组数据 `e1` 到 `e4`，分别为正态分布的随机样本，增加了一些随机扰动。调用 `fake_bootstrapper` 函数获取两个虚拟的中位数和置信区间，分别存储在 `medians` 和 `conf_intervals` 列表中。


fig, ax = plt.subplots()
pos = np.arange(len(treatments)) + 1
bp = ax.boxplot(treatments, sym='k+', positions=pos,
                notch=True, bootstrap=5000,
                usermedians=medians,
                conf_intervals=conf_intervals)

ax.set_xlabel('treatment')
ax.set_ylabel('response')
plt.setp(bp['whiskers'], color='k', linestyle='-')
plt.setp(bp['fliers'], markersize=3.0)
plt.show()

# 创建一个箱线图，并在图形上显示四组数据 `treatments` 的分布情况。使用 `usermedians` 和 `conf_intervals` 参数指定虚拟的中位数和置信区间。设置图形的 x 轴标签为 'treatment'，y 轴标签为 'response'，设置箱线图的风格和细节。


x = np.linspace(-7, 7, 140)
x = np.hstack([-25, x, 25])
fig, ax = plt.subplots()

ax.boxplot([x, x], notch=True, capwidths=[0.01, 0.2])

plt.show()

# 创建一个箱线图，并在图形上显示由 `x` 组成的两组数据。自定义每组数据的上下误差线帽的宽度。


#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.boxplot` / `matplotlib.pyplot.boxplot`
#    - `matplotlib.artist.Artist.set` / `matplotlib.pyplot.setp`

# 添加一个提醒部分，用于引用此示例中使用的特定函数、方法、类和模块，包括 `matplotlib` 中的 `boxplot` 函数和 `set` 方法。
```