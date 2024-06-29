# `D:\src\scipysrc\matplotlib\galleries\examples\statistics\time_series_histogram.py`

```py
# 导入必要的库
import time  # 导入时间模块，用于计时

import matplotlib.pyplot as plt  # 导入绘图库 matplotlib
import numpy as np  # 导入数值计算库 numpy

# 创建一个包含三个子图的图像对象
fig, axes = plt.subplots(nrows=3, figsize=(6, 8), layout='constrained')

# 设置随机种子以保证结果的可复现性
np.random.seed(19680801)

# 定义数据生成参数
num_series = 1000  # 时间序列的数量
num_points = 100  # 每个时间序列的数据点数
SNR = 0.10  # 信噪比（Signal to Noise Ratio）
x = np.linspace(0, 4 * np.pi, num_points)  # 生成横坐标数据

# 生成1000个无偏高斯随机漫步序列作为背景噪声
Y = np.cumsum(np.random.randn(num_series, num_points), axis=-1)

# 生成一小部分正弦信号序列，共1000 * 0.1 = 100个
num_signal = round(SNR * num_series)
phi = (np.pi / 8) * np.random.randn(num_signal, 1)  # 小幅随机偏移
# 对正弦信号进行叠加，考虑随机漫步的均方根标准差作为振幅，并加入少量随机噪声
Y[-num_signal:] = (
    np.sqrt(np.arange(num_points)) * (np.sin(x - phi) + 0.05 * np.random.randn(num_signal, num_points))
)

# 在第一个子图中绘制所有时间序列，使用小的透明度值 alpha。由于重叠的序列较多，很难观察到正弦波的行为。
# 同时，由于需要创建大量的图元对象，这一过程也需要一些时间来运行。
tic = time.time()  # 记录绘图开始时间
axes[0].plot(x, Y.T, color="C0", alpha=0.1)  # 绘制线图
toc = time.time()  # 记录绘图结束时间
axes[0].set_title("Line plot with alpha")  # 设置子图标题
print(f"{toc-tic:.3f} sec. elapsed")  # 输出绘图所花费的时间

# 现在我们将多个时间序列转换为直方图。不仅隐藏的信号更加明显，而且整个过程也更加快速。
tic = time.time()  # 记录直方图转换开始时间
# 在每个时间序列中线性插值点之间的数据，使得数据更加连续
num_fine = 800  # 精细化的数据点数
x_fine = np.linspace(x.min(), x.max(), num_fine)  # 生成更加密集的横坐标数据
y_fine = np.concatenate([np.interp(x_fine, x, y_row) for y_row in Y])  # 对所有时间序列进行插值
# 将 x_fine 广播成指定形状，然后展平成一维数组
x_fine = np.broadcast_to(x_fine, (num_series, num_fine)).ravel()

# 在二维直方图中使用对数颜色映射绘制 (x_fine, y_fine) 点的分布图
# 显然在噪声下存在某种结构
# 可以通过调节 vmax 来增强信号的可见性
cmap = plt.colormaps["plasma"]
cmap = cmap.with_extremes(bad=cmap(0))
h, xedges, yedges = np.histogram2d(x_fine, y_fine, bins=[400, 100])
pcm = axes[1].pcolormesh(xedges, yedges, h.T, cmap=cmap,
                         norm="log", vmax=1.5e2, rasterized=True)
fig.colorbar(pcm, ax=axes[1], label="# points", pad=0)
axes[1].set_title("2d histogram and log color scale")

# 同样的数据，但使用线性颜色映射
pcm = axes[2].pcolormesh(xedges, yedges, h.T, cmap=cmap,
                         vmax=1.5e2, rasterized=True)
fig.colorbar(pcm, ax=axes[2], label="# points", pad=0)
axes[2].set_title("2d histogram and linear color scale")

# 计算代码段执行的时间并打印出来
toc = time.time()
print(f"{toc-tic:.3f} sec. elapsed")
plt.show()

# %%
#
# .. admonition:: References
#
#    This code utilizes the following functions, methods, classes, and modules:
#
#    - `matplotlib.axes.Axes.pcolormesh` / `matplotlib.pyplot.pcolormesh`
#    - `matplotlib.figure.Figure.colorbar`
```