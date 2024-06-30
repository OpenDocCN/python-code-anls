# `D:\src\scipysrc\scipy\doc\source\tutorial\examples\signal_SpectralAnalysis_TwoSinesNoWindow.py`

```
import matplotlib.pyplot as plt  # 导入matplotlib绘图库
import numpy as np  # 导入NumPy科学计算库

from scipy.fft import rfft, rfftfreq  # 从scipy中导入快速傅里叶变换相关函数

n, T = 100, 0.01  # 设置采样点数和采样间隔
fcc = (20, 20.5)  # 定义正弦波的频率
t = np.arange(n) * T  # 生成时间序列

xx = (np.sin(2 * np.pi * fx_ * t) for fx_ in fcc)  # 生成两个正弦信号

f = rfftfreq(n, T)  # 计算频率轴，从0Hz到Nyquist频率
XX = (rfft(x_) / n for x_ in xx)  # 计算每个信号的单边幅值谱

fg1, ax1 = plt.subplots(1, 1, tight_layout=True, figsize=(6., 3.))  # 创建一个画布和一个子图
ax1.set(title=r"Magnitude Spectrum (no window) of $x(t) = \sin(2\pi f_x t)$ ",  # 设置子图标题
        xlabel=rf"Frequency $f$ in Hertz (bin width $\Delta f = {f[1]}\,$Hz)",  # 设置X轴标签
        ylabel=r"Magnitude $|X(f)|/\tau$",  # 设置Y轴标签
        xlim=(f[0], f[-1]))  # 设置X轴范围

for X_, fc_, m_ in zip(XX, fcc, ('x-', '.-')):
    ax1.plot(f, abs(X_), m_, label=rf"$f_x={fc_}\,$Hz")  # 绘制每个信号的幅值谱曲线

ax1.grid(True)  # 显示网格
ax1.legend()  # 显示图例
plt.show()  # 显示图形
```