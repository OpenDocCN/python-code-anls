# `D:\src\scipysrc\matplotlib\galleries\examples\event_handling\resample.py`

```py
"""
===============
Resampling Data
===============

Downsampling lowers the sample rate or sample size of a signal. In
this tutorial, the signal is downsampled when the plot is adjusted
through dragging and zooming.

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""

import matplotlib.pyplot as plt
import numpy as np

# A class that will downsample the data and recompute when zoomed.
class DataDisplayDownsampler:
    def __init__(self, xdata, ydata):
        self.origYData = ydata
        self.origXData = xdata
        self.max_points = 50  # 设定最大数据点数
        self.delta = xdata[-1] - xdata[0]  # 计算数据范围的长度

    def downsample(self, xstart, xend):
        # get the points in the view range
        # 获取视图范围内的数据点
        mask = (self.origXData > xstart) & (self.origXData < xend)
        # dilate the mask by one to catch the points just outside
        # of the view range to not truncate the line
        # 将掩码进行扩张，以捕获视图范围外的数据点，避免截断线条
        mask = np.convolve([1, 1, 1], mask, mode='same').astype(bool)
        # sort out how many points to drop
        # 确定需要丢弃的数据点数量
        ratio = max(np.sum(mask) // self.max_points, 1)

        # mask data
        # 根据掩码过滤数据
        xdata = self.origXData[mask]
        ydata = self.origYData[mask]

        # downsample data
        # 对数据进行降采样
        xdata = xdata[::ratio]
        ydata = ydata[::ratio]

        print(f"using {len(ydata)} of {np.sum(mask)} visible points")  # 打印使用的可见数据点数目

        return xdata, ydata

    def update(self, ax):
        # Update the line
        # 更新折线图
        lims = ax.viewLim
        if abs(lims.width - self.delta) > 1e-8:
            self.delta = lims.width
            xstart, xend = lims.intervalx
            self.line.set_data(*self.downsample(xstart, xend))  # 更新数据
            ax.figure.canvas.draw_idle()  # 在画布上绘制更新后的图像


# Create a signal
# 创建一个信号
xdata = np.linspace(16, 365, (365-16)*4)
ydata = np.sin(2*np.pi*xdata/153) + np.cos(2*np.pi*xdata/127)

d = DataDisplayDownsampler(xdata, ydata)

fig, ax = plt.subplots()

# Hook up the line
# 连接折线图
d.line, = ax.plot(xdata, ydata, 'o-')
ax.set_autoscale_on(False)  # 关闭自动缩放，避免无限循环

# Connect for changing the view limits
# 连接视图范围变化的事件
ax.callbacks.connect('xlim_changed', d.update)
ax.set_xlim(16, 365)
plt.show()

# %%
# .. tags:: interactivity: zoom, event-handling
```