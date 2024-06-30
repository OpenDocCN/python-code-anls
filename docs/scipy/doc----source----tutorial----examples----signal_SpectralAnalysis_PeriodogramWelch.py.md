# `D:\src\scipysrc\scipy\doc\source\tutorial\examples\signal_SpectralAnalysis_PeriodogramWelch.py`

```
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算
import scipy.signal as signal  # 导入 scipy.signal 库，用于信号处理

rng = np.random.default_rng(73625)  # 使用种子 73625 初始化随机数生成器 rng，确保结果可重现性

fs, n = 10e3, 10_000  # 设置采样频率 fs 和采样点数 n
f_x, noise_power = 1270, 1e-3 * fs / 2  # 设置信号频率 f_x 和噪声功率 noise_power
t = np.arange(n) / fs  # 生成时间数组 t，表示从 0 到 (n-1)/fs 的时间点

# 生成带噪声的信号 x，由正弦信号和高斯噪声组成
x = (np.sqrt(2) * np.sin(2 * np.pi * f_x * t) +
     rng.normal(scale=np.sqrt(noise_power), size=t.shape))

# 创建包含两个子图的图形窗口 fg 和子图对象数组 axx
fg, axx = plt.subplots(1, 2, sharex='all', tight_layout=True, figsize=(7, 3.5))

# 设置子图 axx[0] 的标题和纵轴标签
axx[0].set(title="Squared Magnitude Spectrum", ylabel="Square of Magnitude in V²")

# 设置子图 axx[1] 的标题和纵轴标签
axx[1].set(title="Power Spectral Density", ylabel="Power Spectral Density in V²/Hz")

# 遍历子图数组 axx 和字符串数组 ('spectrum', 'density')
for ax_, s_ in zip(axx, ('spectrum', 'density')):
    # 计算信号 x 的周期图，返回频率数组 f_p 和功率谱数组 P_p
    f_p, P_p = signal.periodogram(x, fs, 'hann', scaling=s_)
    # 计算信号 x 的 Welch 方法功率谱估计，返回频率数组 f_w 和功率谱数组 P_w
    f_w, P_w = signal.welch(x, fs, scaling=s_)
    
    # 绘制 ax_ 子图上的周期图和Welch方法功率谱，使用半对数坐标绘制
    ax_.semilogy(f_p/1e3, P_p, label=f"Periodogram ({len(f_p)} bins)")
    ax_.semilogy(f_w/1e3, P_w, label=f"Welch's Method ({len(f_w)} bins)")
    
    # 设置子图的 x 轴标签、x 轴范围、y 轴范围
    ax_.set(xlabel="Frequency in kHz", xlim=(0, 2), ylim=(1e-7, 1.3))
    # 开启子图网格
    ax_.grid(True)
    # 设置图例位置为下方中心
    ax_.legend(loc='lower center')

# 显示绘制的图形
plt.show()
```