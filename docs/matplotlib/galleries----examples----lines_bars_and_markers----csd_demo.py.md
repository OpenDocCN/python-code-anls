# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\csd_demo.py`

```py
"""
============================
Cross spectral density (CSD)
============================

Plot the cross spectral density (CSD) of two signals using `~.Axes.csd`.
"""
import matplotlib.pyplot as plt  # 导入 matplotlib 库，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

fig, (ax1, ax2) = plt.subplots(2, 1, layout='constrained')  # 创建包含两个子图的图像对象

dt = 0.01  # 时间步长
t = np.arange(0, 30, dt)  # 创建时间序列，从0到30，步长为dt

# Fixing random state for reproducibility
np.random.seed(19680801)  # 设置随机种子，以便结果可重现

nse1 = np.random.randn(len(t))  # 生成长度为t的随机白噪声序列1
nse2 = np.random.randn(len(t))  # 生成长度为t的随机白噪声序列2
r = np.exp(-t / 0.05)  # 创建一个指数衰减的序列r

cnse1 = np.convolve(nse1, r, mode='same') * dt  # 使用r对白噪声序列1进行卷积，生成有色噪声1
cnse2 = np.convolve(nse2, r, mode='same') * dt  # 使用r对白噪声序列2进行卷积，生成有色噪声2

# two signals with a coherent part and a random part
s1 = 0.01 * np.sin(2 * np.pi * 10 * t) + cnse1  # 生成信号s1，包含相干部分和随机部分
s2 = 0.01 * np.sin(2 * np.pi * 10 * t) + cnse2  # 生成信号s2，包含相干部分和随机部分

ax1.plot(t, s1, t, s2)  # 在第一个子图上绘制信号s1和s2
ax1.set_xlim(0, 5)  # 设置第一个子图的X轴范围
ax1.set_xlabel('Time (s)')  # 设置第一个子图的X轴标签
ax1.set_ylabel('s1 and s2')  # 设置第一个子图的Y轴标签
ax1.grid(True)  # 在第一个子图上显示网格线

cxy, f = ax2.csd(s1, s2, NFFT=256, Fs=1. / dt)  # 计算信号s1和s2的交叉谱密度，返回值分别为CSD和频率f
ax2.set_ylabel('CSD (dB)')  # 设置第二个子图的Y轴标签

plt.show()  # 显示绘制的图像
```