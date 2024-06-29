# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\psd_demo.py`

```py
"""
============================
Power spectral density (PSD)
============================

Plotting power spectral density (PSD) using `~.Axes.psd`.

The PSD is a common plot in the field of signal processing. NumPy has
many useful libraries for computing a PSD. Below we demo a few examples
of how this can be accomplished and visualized with Matplotlib.
"""

# 导入 Matplotlib 库并命名为 plt
import matplotlib.pyplot as plt
# 导入 NumPy 库并命名为 np
import numpy as np
# 导入 Matplotlib 的 mlab 模块
import matplotlib.mlab as mlab

# 设定随机种子以便结果可重现性
np.random.seed(19680801)

# 设定时间步长 dt 为 0.01
dt = 0.01
# 生成时间数组 t，从 0 到 10，步长为 dt
t = np.arange(0, 10, dt)
# 生成长度与 t 相同的随机噪声信号 nse
nse = np.random.randn(len(t))
# 设定指数衰减函数 r
r = np.exp(-t / 0.05)

# 对随机噪声信号 nse 进行卷积运算，得到 cnse
cnse = np.convolve(nse, r) * dt
# 截取与 t 相同长度的卷积结果 cnse
cnse = cnse[:len(t)]
# 生成信号 s，为正弦信号和卷积结果的叠加
s = 0.1 * np.sin(2 * np.pi * t) + cnse

# 创建包含两个子图的图像窗口 fig，分别命名为 ax0 和 ax1
fig, (ax0, ax1) = plt.subplots(2, 1, layout='constrained')
# 在 ax0 上绘制时间 t 对应的信号 s
ax0.plot(t, s)
# 设置 ax0 的 x 轴标签为 'Time (s)'
ax0.set_xlabel('Time (s)')
# 设置 ax0 的 y 轴标签为 'Signal'
ax0.set_ylabel('Signal')
# 在 ax1 上绘制信号 s 的功率谱密度图，NFFT 为 512，采样频率为 1/dt
ax1.psd(s, NFFT=512, Fs=1 / dt)

# 显示图形
plt.show()

# %%
# 将此示例与等效的 Matlab 代码进行比较，实现相同的功能::

#     dt = 0.01;
#     t = [0:dt:10];
#     nse = randn(size(t));
#     r = exp(-t/0.05);
#     cnse = conv(nse, r)*dt;
#     cnse = cnse(1:length(t));
#     s = 0.1*sin(2*pi*t) + cnse;
#
#     subplot(211)
#     plot(t, s)
#     subplot(212)
#     psd(s, 512, 1/dt)
#
# 下面我们将展示一个稍微复杂的例子，演示填充如何影响得到的 PSD。

# 设定时间步长 dt 为 pi / 100
dt = np.pi / 100.
# 计算采样频率 fs
fs = 1. / dt
# 生成时间数组 t，从 0 到 8，步长为 dt
t = np.arange(0, 8, dt)
# 生成信号 y，包含两个正弦波成分和随机噪声
y = 10. * np.sin(2 * np.pi * 4 * t) + 5. * np.sin(2 * np.pi * 4.25 * t)
y = y + np.random.randn(*t.shape)

# 绘制原始时间序列图
fig, axs = plt.subplot_mosaic([
    ['signal', 'signal', 'signal'],
    ['zero padding', 'block size', 'overlap'],
], layout='constrained')

# 在 axs['signal'] 上绘制时间 t 对应的信号 y
axs['signal'].plot(t, y)
# 设置 axs['signal'] 的 x 轴标签为 'Time (s)'
axs['signal'].set_xlabel('Time (s)')
# 设置 axs['signal'] 的 y 轴标签为 'Signal'
axs['signal'].set_ylabel('Signal')

# 使用不同数量的零填充绘制功率谱密度图
axs['zero padding'].psd(y, NFFT=len(t), pad_to=len(t), Fs=fs)
axs['zero padding'].psd(y, NFFT=len(t), pad_to=len(t) * 2, Fs=fs)
axs['zero padding'].psd(y, NFFT=len(t), pad_to=len(t) * 4, Fs=fs)

# 使用不同块大小绘制功率谱密度图，零填充到原始数据序列的长度
axs['block size'].psd(y, NFFT=len(t), pad_to=len(t), Fs=fs)
axs['block size'].psd(y, NFFT=len(t) // 2, pad_to=len(t), Fs=fs)
axs['block size'].psd(y, NFFT=len(t) // 4, pad_to=len(t), Fs=fs)
axs['block size'].set_ylabel('')

# 使用不同块之间重叠量绘制功率谱密度图
axs['overlap'].psd(y, NFFT=len(t) // 2, pad_to=len(t), noverlap=0, Fs=fs)
axs['overlap'].psd(y, NFFT=len(t) // 2, pad_to=len(t),
                   noverlap=int(0.025 * len(t)), Fs=fs)
axs['overlap'].psd(y, NFFT=len(t) // 2, pad_to=len(t),
                   noverlap=int(0.1 * len(t)), Fs=fs)
axs['overlap'].set_ylabel('')
axs['overlap'].set_title('overlap')

# 遍历 axs.items() 中的每个键值对，设置标题并共享 x 轴和 y 轴
for title, ax in axs.items():
    if title == 'signal':
        continue

    ax.set_title(title)
    ax.sharex(axs['zero padding'])
    ax.sharey(axs['zero padding'])

# 显示图形
plt.show()


# %%
# 这是一个从信号处理工具包中移植的 MATLAB 示例的 Python 版本，用于展示一段时间内 Matplotlib 和 MATLAB 在功率谱密度(PSD)缩放上的差异。

# 设置采样频率为1000 Hz
fs = 1000

# 在0到0.3秒之间生成包含301个点的时间序列
t = np.linspace(0, 0.3, 301)

# 定义两个振幅的数组，并重塑为列向量
A = np.array([2, 8]).reshape(-1, 1)

# 定义两个频率的数组，并重塑为列向量
f = np.array([150, 140]).reshape(-1, 1)

# 生成复杂信号，包括正弦波和高斯噪声
xn = (A * np.sin(2 * np.pi * f * t)).sum(axis=0)
xn += 5 * np.random.randn(*t.shape)

# 创建包含两个子图的图形窗口，布局为'constrained'
fig, (ax0, ax1) = plt.subplots(ncols=2, layout='constrained')

# 设置y轴刻度范围和刻度标签
yticks = np.arange(-50, 30, 10)
yrange = (yticks[0], yticks[-1])

# 设置x轴刻度标签
xticks = np.arange(0, 550, 100)

# 绘制第一个子图的功率谱密度估计（Periodogram）
ax0.psd(xn, NFFT=301, Fs=fs, window=mlab.window_none, pad_to=1024,
        scale_by_freq=True)
ax0.set_title('Periodogram')  # 设置子图标题为'Periodogram'
ax0.set_yticks(yticks)  # 设置y轴刻度为预定义的刻度
ax0.set_xticks(xticks)  # 设置x轴刻度为预定义的刻度
ax0.grid(True)  # 启用网格线
ax0.set_ylim(yrange)  # 设置y轴范围

# 绘制第二个子图的功率谱密度估计（Welch）
ax1.psd(xn, NFFT=150, Fs=fs, window=mlab.window_none, pad_to=512, noverlap=75,
        scale_by_freq=True)
ax1.set_title('Welch')  # 设置子图标题为'Welch'
ax1.set_xticks(xticks)  # 设置x轴刻度为预定义的刻度
ax1.set_yticks(yticks)  # 设置y轴刻度为预定义的刻度
ax1.set_ylabel('')  # 清除`psd`函数添加的y轴标签
ax1.grid(True)  # 启用网格线
ax1.set_ylim(yrange)  # 设置y轴范围

plt.show()  # 显示图形窗口



# 这是一个从信号处理工具包中移植的 MATLAB 示例的 Python 版本，用于展示一段时间内 Matplotlib 和 MATLAB 在功率谱密度(PSD)缩放上的差异。
#
# 该示例使用复杂信号，以验证复杂功率谱密度的正常工作。

# 设置随机数种子，以保证结果的可重复性
prng = np.random.RandomState(19680801)

# 设置采样频率为1000 Hz
fs = 1000

# 在0到0.3秒之间生成包含301个点的时间序列
t = np.linspace(0, 0.3, 301)

# 定义两个振幅的数组，并重塑为列向量
A = np.array([2, 8]).reshape(-1, 1)

# 定义两个频率的数组，并重塑为列向量
f = np.array([150, 140]).reshape(-1, 1)

# 生成复杂信号，包括正弦波和高斯噪声
xn = (A * np.exp(2j * np.pi * f * t)).sum(axis=0) + 5 * prng.randn(*t.shape)

# 创建包含两个子图的图形窗口，布局为'constrained'
fig, (ax0, ax1) = plt.subplots(ncols=2, layout='constrained')

# 设置y轴刻度范围和刻度标签
yticks = np.arange(-50, 30, 10)
yrange = (yticks[0], yticks[-1])

# 设置x轴刻度标签
xticks = np.arange(-500, 550, 200)

# 绘制第一个子图的功率谱密度估计（Periodogram）
ax0.psd(xn, NFFT=301, Fs=fs, window=mlab.window_none, pad_to=1024,
        scale_by_freq=True)
ax0.set_title('Periodogram')  # 设置子图标题为'Periodogram'
ax0.set_yticks(yticks)  # 设置y轴刻度为预定义的刻度
ax0.set_xticks(xticks)  # 设置x轴刻度为预定义的刻度
ax0.grid(True)  # 启用网格线
ax0.set_ylim(yrange)  # 设置y轴范围

# 绘制第二个子图的功率谱密度估计（Welch）
ax1.psd(xn, NFFT=150, Fs=fs, window=mlab.window_none, pad_to=512, noverlap=75,
        scale_by_freq=True)
ax1.set_title('Welch')  # 设置子图标题为'Welch'
ax1.set_xticks(xticks)  # 设置x轴刻度为预定义的刻度
ax1.set_yticks(yticks)  # 设置y轴刻度为预定义的刻度
ax1.set_ylabel('')  # 清除`psd`函数添加的y轴标签
ax1.grid(True)  # 启用网格线
ax1.set_ylim(yrange)  # 设置y轴范围

plt.show()  # 显示图形窗口
```