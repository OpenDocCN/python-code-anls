# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\spectrum_demo.py`

```py
"""
========================
Spectrum representations
========================

The plots show different spectrum representations of a sine signal with
additive noise. A (frequency) spectrum of a discrete-time signal is calculated
by utilizing the fast Fourier transform (FFT).
"""

# 导入 matplotlib 库并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库并重命名为 np
import numpy as np

# 设置随机数种子，以便结果可重复
np.random.seed(0)

# 定义采样间隔
dt = 0.01
# 根据采样间隔计算采样频率
Fs = 1 / dt
# 生成时间序列
t = np.arange(0, 10, dt)

# 生成加性高斯噪声:
nse = np.random.randn(len(t))
# 定义一个衰减函数
r = np.exp(-t / 0.05)
# 将噪声信号与衰减函数进行卷积，然后乘以采样间隔
cnse = np.convolve(nse, r) * dt
# 截取卷积结果与原始信号等长
cnse = cnse[:len(t)]

# 生成包含噪声的正弦信号
s = 0.1 * np.sin(4 * np.pi * t) + cnse  # the signal

# 创建一个大小为 (7, 7) 的新图形
fig = plt.figure(figsize=(7, 7), layout='constrained')
# 使用 subplot_mosaic 方法创建一个布局，包含三行两列的子图
axs = fig.subplot_mosaic([["signal", "signal"],
                          ["magnitude", "log_magnitude"],
                          ["phase", "angle"]])

# 在子图 "signal" 中绘制时间信号
axs["signal"].set_title("Signal")
axs["signal"].plot(t, s, color='C0')
axs["signal"].set_xlabel("Time (s)")
axs["signal"].set_ylabel("Amplitude")

# 在子图 "magnitude" 中绘制幅度谱
axs["magnitude"].set_title("Magnitude Spectrum")
axs["magnitude"].magnitude_spectrum(s, Fs=Fs, color='C1')

# 在子图 "log_magnitude" 中绘制对数幅度谱
axs["log_magnitude"].set_title("Log. Magnitude Spectrum")
axs["log_magnitude"].magnitude_spectrum(s, Fs=Fs, scale='dB', color='C1')

# 在子图 "phase" 中绘制相位谱
axs["phase"].set_title("Phase Spectrum ")
axs["phase"].phase_spectrum(s, Fs=Fs, color='C2')

# 在子图 "angle" 中绘制角度谱
axs["angle"].set_title("Angle Spectrum")
axs["angle"].angle_spectrum(s, Fs=Fs, color='C2')

# 显示图形
plt.show()
```