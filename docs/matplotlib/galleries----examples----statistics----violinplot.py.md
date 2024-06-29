# `D:\src\scipysrc\matplotlib\galleries\examples\statistics\violinplot.py`

```py
"""
==================
Violin plot basics
==================

Violin plots are similar to histograms and box plots in that they show
an abstract representation of the probability distribution of the
sample. Rather than showing counts of data points that fall into bins
or order statistics, violin plots use kernel density estimation (KDE) to
compute an empirical distribution of the sample. That computation
is controlled by several parameters. This example demonstrates how to
modify the number of points at which the KDE is evaluated (``points``)
and how to modify the bandwidth of the KDE (``bw_method``).

For more information on violin plots and KDE, the scikit-learn docs
have a great section: https://scikit-learn.org/stable/modules/density.html
"""

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块，用于绘图
import numpy as np  # 导入 numpy 模块，用于数值计算

# Fixing random state for reproducibility
np.random.seed(19680801)  # 设置随机数种子，保证结果可重现

# fake data
fs = 10  # 设置字体大小为 10
pos = [1, 2, 4, 5, 7, 8]  # 位置信息，用于指定每个小提琴图的位置
data = [np.random.normal(0, std, size=100) for std in pos]  # 生成假数据，每个位置对应一组数据

fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(10, 4))  # 创建一个 2x6 的子图布局，图像尺寸为 10x4

axs[0, 0].violinplot(data, pos, points=20, widths=0.3,
                     showmeans=True, showextrema=True, showmedians=True)
# 在第一个子图中绘制小提琴图，数据为 data，位置为 pos，点数为 20，宽度为 0.3，显示均值、极值和中位数
axs[0, 0].set_title('Custom violin 1', fontsize=fs)  # 设置子图标题为 'Custom violin 1'，字体大小为 fs

axs[0, 1].violinplot(data, pos, points=40, widths=0.5,
                     showmeans=True, showextrema=True, showmedians=True,
                     bw_method='silverman')
# 在第二个子图中绘制小提琴图，数据为 data，位置为 pos，点数为 40，宽度为 0.5，显示均值、极值和中位数，采用银子曼带宽法
axs[0, 1].set_title('Custom violin 2', fontsize=fs)  # 设置子图标题为 'Custom violin 2'，字体大小为 fs

axs[0, 2].violinplot(data, pos, points=60, widths=0.7, showmeans=True,
                     showextrema=True, showmedians=True, bw_method=0.5)
# 在第三个子图中绘制小提琴图，数据为 data，位置为 pos，点数为 60，宽度为 0.7，显示均值、极值和中位数，带宽为 0.5
axs[0, 2].set_title('Custom violin 3', fontsize=fs)  # 设置子图标题为 'Custom violin 3'，字体大小为 fs

axs[0, 3].violinplot(data, pos, points=60, widths=0.7, showmeans=True,
                     showextrema=True, showmedians=True, bw_method=0.5,
                     quantiles=[[0.1], [], [], [0.175, 0.954], [0.75], [0.25]])
# 在第四个子图中绘制小提琴图，数据为 data，位置为 pos，点数为 60，宽度为 0.7，显示均值、极值和中位数，带宽为 0.5，
# 自定义分位数为每个小提琴图的显示量化分位数
axs[0, 3].set_title('Custom violin 4', fontsize=fs)  # 设置子图标题为 'Custom violin 4'，字体大小为 fs

axs[0, 4].violinplot(data[-1:], pos[-1:], points=60, widths=0.7,
                     showmeans=True, showextrema=True, showmedians=True,
                     quantiles=[0.05, 0.1, 0.8, 0.9], bw_method=0.5)
# 在第五个子图中绘制小提琴图，仅使用最后一组数据，位置为最后一个位置，点数为 60，宽度为 0.7，显示均值、极值和中位数，
# 自定义分位数为 [0.05, 0.1, 0.8, 0.9]，带宽为 0.5
axs[0, 4].set_title('Custom violin 5', fontsize=fs)  # 设置子图标题为 'Custom violin 5'，字体大小为 fs

axs[0, 5].violinplot(data[-1:], pos[-1:], points=60, widths=0.7,
                     showmeans=True, showextrema=True, showmedians=True,
                     quantiles=[0.05, 0.1, 0.8, 0.9], bw_method=0.5, side='low')
# 在第六个子图中绘制小提琴图，仅使用最后一组数据，位置为最后一个位置，点数为 60，宽度为 0.7，显示均值、极值和中位数，
# 自定义分位数为 [0.05, 0.1, 0.8, 0.9]，带宽为 0.5，绘制在低侧

axs[0, 5].violinplot(data[-1:], pos[-1:], points=60, widths=0.7,
                     showmeans=True, showextrema=True, showmedians=True,
                     quantiles=[0.05, 0.1, 0.8, 0.9], bw_method=0.5, side='high')
# 继续在第六个子图中绘制小提琴图，使用同样的参数，但这次绘制在高侧
axs[0, 5].set_title('Custom violin 6', fontsize=fs)  # 设置子图标题为 'Custom violin 6'，字体大小为 fs

axs[1, 0].violinplot(data, pos, points=80, orientation='horizontal', widths=0.7,
                     showmeans=True, showextrema=True, showmedians=True)
# 在第七个子图中绘制横向小提琴图，数据为 data，位置为 pos，点数为 80，宽度为 0.7，显示均值、极值和中位数
axs[1, 0].set_title('Custom violin 7', fontsize=fs)  # 设置子图标题为 'Custom violin 7'，字体大小为 fs
axs[1, 1].violinplot(data, pos, points=100, orientation='horizontal', widths=0.9,
                     showmeans=True, showextrema=True, showmedians=True,
                     bw_method='silverman')
# 在图表的第二行第二列绘制水平方向的小提琴图，使用给定的数据和位置参数进行绘制，
# 设置小提琴形状的精细程度为100个点，显示均值、极值和中位数，使用银斯维尔德法则估算带宽

axs[1, 1].set_title('Custom violin 8', fontsize=fs)
# 在同一子图中设置标题为'Custom violin 8'，字体大小由变量 fs 控制

axs[1, 2].violinplot(data, pos, points=200, orientation='horizontal', widths=1.1,
                     showmeans=True, showextrema=True, showmedians=True,
                     bw_method=0.5)
# 在图表的第二行第三列绘制水平方向的小提琴图，使用给定的数据和位置参数进行绘制，
# 设置小提琴形状的精细程度为200个点，显示均值、极值和中位数，带宽方法设定为0.5

axs[1, 2].set_title('Custom violin 9', fontsize=fs)
# 在同一子图中设置标题为'Custom violin 9'，字体大小由变量 fs 控制

axs[1, 3].violinplot(data, pos, points=200, orientation='horizontal', widths=1.1,
                     showmeans=True, showextrema=True, showmedians=True,
                     quantiles=[[0.1], [], [], [0.175, 0.954], [0.75], [0.25]],
                     bw_method=0.5)
# 在图表的第二行第四列绘制水平方向的小提琴图，使用给定的数据和位置参数进行绘制，
# 设置小提琴形状的精细程度为200个点，显示均值、极值和中位数，以及指定的分位数，
# 带宽方法设定为0.5

axs[1, 3].set_title('Custom violin 10', fontsize=fs)
# 在同一子图中设置标题为'Custom violin 10'，字体大小由变量 fs 控制

axs[1, 4].violinplot(data[-1:], pos[-1:], points=200, orientation='horizontal',
                     widths=1.1, showmeans=True, showextrema=True, showmedians=True,
                     quantiles=[0.05, 0.1, 0.8, 0.9], bw_method=0.5)
# 在图表的第二行第五列绘制水平方向的小提琴图，使用数据和位置参数的最后一个元素进行绘制，
# 设置小提琴形状的精细程度为200个点，显示均值、极值和中位数，以及指定的分位数，
# 带宽方法设定为0.5

axs[1, 4].set_title('Custom violin 11', fontsize=fs)
# 在同一子图中设置标题为'Custom violin 11'，字体大小由变量 fs 控制

axs[1, 5].violinplot(data[-1:], pos[-1:], points=200, orientation='horizontal',
                     widths=1.1, showmeans=True, showextrema=True, showmedians=True,
                     quantiles=[0.05, 0.1, 0.8, 0.9], bw_method=0.5, side='low')
# 在图表的第二行第六列绘制水平方向的小提琴图，使用数据和位置参数的最后一个元素进行绘制，
# 设置小提琴形状的精细程度为200个点，显示均值、极值和中位数，以及指定的分位数，
# 带宽方法设定为0.5，并将小提琴图绘制在低侧（'low'）

axs[1, 5].violinplot(data[-1:], pos[-1:], points=200, orientation='horizontal',
                     widths=1.1, showmeans=True, showextrema=True, showmedians=True,
                     quantiles=[0.05, 0.1, 0.8, 0.9], bw_method=0.5, side='high')
# 在图表的第二行第六列绘制水平方向的小提琴图，使用数据和位置参数的最后一个元素进行绘制，
# 设置小提琴形状的精细程度为200个点，显示均值、极值和中位数，以及指定的分位数，
# 带宽方法设定为0.5，并将小提琴图绘制在高侧（'high'）

axs[1, 5].set_title('Custom violin 12', fontsize=fs)
# 在同一子图中设置标题为'Custom violin 12'，字体大小由变量 fs 控制


for ax in axs.flat:
    ax.set_yticklabels([])
# 对所有子图对象进行遍历，设置y轴刻度标签为空列表，即不显示y轴刻度标签

fig.suptitle("Violin Plotting Examples")
# 设置整个图形的标题为"Violin Plotting Examples"

fig.subplots_adjust(hspace=0.4)
# 调整子图之间的水平间距为0.4个单位

plt.show()
# 显示绘制好的图形
```