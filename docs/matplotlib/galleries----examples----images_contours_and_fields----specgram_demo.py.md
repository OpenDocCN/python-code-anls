# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\specgram_demo.py`

```py
"""
===========
Spectrogram
===========

Plotting a spectrogram using `~.Axes.specgram`.
"""
# 导入 matplotlib.pyplot 库，并使用 plt 别名
import matplotlib.pyplot as plt
# 导入 numpy 库，并使用 np 别名
import numpy as np

# 设置随机种子以便结果可复现
np.random.seed(19680801)

# 定义时间步长
dt = 0.0005
# 生成时间序列 t，从 0 到 20.5，步长为 dt
t = np.arange(0.0, 20.5, dt)
# 生成频率为 100 Hz 的正弦波信号 s1
s1 = np.sin(2 * np.pi * 100 * t)
# 生成频率为 400 Hz 的正弦波信号 s2
s2 = 2 * np.sin(2 * np.pi * 400 * t)

# 在时间小于等于 10 和大于等于 12 的范围内将 s2 设置为 0，创建一个短暂的“啁啾”信号
s2[t <= 10] = s2[12 <= t] = 0

# 生成长度为 len(t) 的随机噪声信号 nse
nse = 0.01 * np.random.random(size=len(t))

# 将信号 s1、s2 和噪声信号 nse 加和得到信号 x
x = s1 + s2 + nse  # the signal

# 定义窗口长度 NFFT 为 1024
NFFT = 1024
# 计算采样频率 Fs
Fs = 1/dt

# 创建包含两个子图的图像窗口 fig，ax1 和 ax2 分别是两个子图的 Axes 对象
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

# 在 ax1 中绘制信号 x 随时间变化的图像
ax1.plot(t, x)
# 设置 ax1 的 y 轴标签为 'Signal'
ax1.set_ylabel('Signal')

# 调用 ax2 的 specgram 方法绘制 x 的频谱图
Pxx, freqs, bins, im = ax2.specgram(x, NFFT=NFFT, Fs=Fs)
# specgram 方法返回四个对象，它们分别是：
# - Pxx: 周期图
# - freqs: 频率向量
# - bins: 时间间隔的中心
# - im: 表示图中数据的 .image.AxesImage 实例
# 设置 ax2 的 x 轴标签为 'Time (s)'
ax2.set_xlabel('Time (s)')
# 设置 ax2 的 y 轴标签为 'Frequency (Hz)'
ax2.set_ylabel('Frequency (Hz)')
# 设置 ax2 的 x 轴范围在 0 到 20
ax2.set_xlim(0, 20)

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.specgram` / `matplotlib.pyplot.specgram`
```