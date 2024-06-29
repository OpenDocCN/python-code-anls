# `D:\src\scipysrc\matplotlib\galleries\examples\event_handling\pick_event_demo2.py`

```py
"""
=================
Pick event demo 2
=================

Compute the mean (mu) and standard deviation (sigma) of 100 data sets and plot
mu vs. sigma.  When you click on one of the (mu, sigma) points, plot the raw
data from the dataset that generated this point.

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""

# 导入需要的库
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以便结果可重复
np.random.seed(19680801)

# 生成一个 100x1000 的随机数组
X = np.random.rand(100, 1000)

# 计算每行数据的均值和标准差
xs = np.mean(X, axis=1)
ys = np.std(X, axis=1)

# 创建一个新的图形和坐标轴
fig, ax = plt.subplots()
ax.set_title('click on point to plot time series')

# 绘制均值 vs. 标准差的散点图，并启用选取事件
line, = ax.plot(xs, ys, 'o', picker=True, pickradius=5)

# 定义选取事件的处理函数
def onpick(event):
    # 如果触发事件的对象不是散点图线对象，直接返回
    if event.artist != line:
        return
    
    # 获取选取点的索引
    N = len(event.ind)
    if not N:
        return
    
    # 创建新的子图来展示选取点对应的原始数据
    figi, axs = plt.subplots(N, squeeze=False)
    for ax, dataind in zip(axs.flat, event.ind):
        # 在子图上绘制原始数据
        ax.plot(X[dataind])
        # 在子图上添加均值和标准差的文本注释
        ax.text(.05, .9, f'mu={xs[dataind]:1.3f}\nsigma={ys[dataind]:1.3f}',
                transform=ax.transAxes, va='top')
        # 设置子图的纵坐标范围
        ax.set_ylim(-0.5, 1.5)
    
    # 显示子图
    figi.show()

# 将选取事件连接到图形的选取事件处理函数
fig.canvas.mpl_connect('pick_event', onpick)

# 显示图形界面
plt.show()
```