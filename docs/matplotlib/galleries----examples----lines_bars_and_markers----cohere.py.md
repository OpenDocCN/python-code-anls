# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\cohere.py`

```py
"""
=====================================
Plotting the coherence of two signals
=====================================

An example showing how to plot the coherence of two signals using `~.Axes.cohere`.
"""

# 引入matplotlib绘图库和numpy科学计算库
import matplotlib.pyplot as plt
import numpy as np

# 设定随机种子以便结果可复现
np.random.seed(19680801)

# 设置时间步长
dt = 0.01
# 创建时间数组
t = np.arange(0, 30, dt)
# 生成第一个白噪声信号
nse1 = np.random.randn(len(t))                 # white noise 1
# 生成第二个白噪声信号
nse2 = np.random.randn(len(t))                 # white noise 2

# 生成两个信号，包含10 Hz的相干部分和随机部分
s1 = np.sin(2 * np.pi * 10 * t) + nse1
s2 = np.sin(2 * np.pi * 10 * t) + nse2

# 创建一个包含两个子图的图形对象
fig, axs = plt.subplots(2, 1, layout='constrained')

# 在第一个子图中绘制信号s1和s2随时间的变化
axs[0].plot(t, s1, t, s2)
# 设定第一个子图的x轴范围
axs[0].set_xlim(0, 2)
# 设定第一个子图的x轴标签
axs[0].set_xlabel('Time (s)')
# 设定第一个子图的y轴标签
axs[0].set_ylabel('s1 and s2')
# 给第一个子图添加网格线
axs[0].grid(True)

# 计算并绘制两个信号的相干性
cxy, f = axs[1].cohere(s1, s2, NFFT=256, Fs=1. / dt)
# 设定第二个子图的y轴标签
axs[1].set_ylabel('Coherence')

# 显示图形
plt.show()
```