# `D:\src\scipysrc\scipy\doc\source\tutorial\examples\signal_SpectralAnalysis_SpectrumPhaseDelay.py`

```
# 导入必要的库
import matplotlib.gridspec as gridspec  # 导入matplotlib中的gridspec模块，用于创建多子图布局
import matplotlib.pyplot as plt  # 导入matplotlib中的pyplot模块，用于绘制图形
import numpy as np  # 导入NumPy库，用于数值计算

from scipy import signal  # 从SciPy库中导入signal模块，用于信号处理
from scipy.fft import rfft, rfftfreq  # 从SciPy库的fft子模块中导入rfft和rfftfreq函数，用于快速傅里叶变换

# 创建输入信号：
n = 50  # 信号长度
x = np.zeros(n)  # 创建长度为n的全零数组
x[0] = n  # 设置第一个元素为n

# 应用FIR滤波器，将信号延迟3个样本：
y = signal.lfilter([0, 0, 0, 1], 1, x)  # 对信号x应用FIR滤波器，延迟为3个样本

# 对输入信号x和输出信号y进行快速傅里叶变换，并归一化：
X, Y = (rfft(z_) / n for z_ in (x, y))  # 对x和y进行快速傅里叶变换，并归一化处理
f = rfftfreq(n, 1)  # 计算频率轴，采样间隔T = 1秒

# 创建绘图对象
fig = plt.figure(tight_layout=True, figsize=(6., 4.))  # 创建画布对象，设置紧凑布局和尺寸

# 创建网格布局，分为3行1列
gs = gridspec.GridSpec(3, 1)

# 在网格布局中添加子图
ax0 = fig.add_subplot(gs[0, :])  # 第一行，占据所有列
ax1 = fig.add_subplot(gs[1:, :], sharex=ax0)  # 第二行到最后一行，占据所有列，并共享x轴

# 绘制幅度和相位响应曲线
for Z_, n_, m_ in zip((X, Y), ("Input $X(f)$", "Output $Y(f)$"), ('+-', 'x-')):
    ax0.plot(f, abs(Z_), m_, alpha=.5, label=n_)  # 绘制幅度响应曲线
    ax1.plot(f, np.unwrap(np.angle(Z_)), m_, alpha=.5, label=n_)  # 绘制相位响应曲线

# 设置子图0的标题和坐标轴标签
ax0.set(title="Frequency Response of 3 Sample Delay Filter (no window)",
        ylabel="Magnitude", xlim=(0, f[-1]), ylim=(0, 1.1))

# 设置子图1的坐标轴标签和y轴刻度
ax1.set(xlabel=rf"Frequency $f$ in Hertz ($\Delta f = {1/n}\,$Hz)",
        ylabel=r"Phase in rad")
ax1.set_yticks(-np.arange(0, 7)*np.pi/2,
               ['0', '-π/2', '-π', '-3/2π', '-2π', '-4/3π', '-3π'])

# 创建子图2，并设置其y轴标签和刻度
ax2 = ax1.twinx()
ax2.set(ylabel=r"Phase in Degrees", ylim=np.rad2deg(ax1.get_ylim()),
        yticks=np.arange(-540, 90, 90))

# 对所有子图添加图例和网格线，并显示图形
for ax_ in (ax0, ax1):
    ax_.legend()  # 添加图例
    ax_.grid()  # 添加网格线
plt.show()  # 显示图形
```