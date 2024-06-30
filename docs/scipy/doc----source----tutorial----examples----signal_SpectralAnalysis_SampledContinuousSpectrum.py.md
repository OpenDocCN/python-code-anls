# `D:\src\scipysrc\scipy\doc\source\tutorial\examples\signal_SpectralAnalysis_SampledContinuousSpectrum.py`

```
import matplotlib.pyplot as plt  # 导入matplotlib绘图库
import numpy as np  # 导入numpy数学计算库

from scipy.fft import rfft, rfftfreq  # 从scipy的fft模块中导入rfft和rfftfreq函数

n, T = 100, 0.01  # 样本数和采样间隔
tau = n*T  # 总时间
t = np.arange(n) * T  # 时间轴

fcc = (20, 20.5)  # 正弦信号的频率
xx = (np.sin(2 * np.pi * fc_ * t) for fc_ in fcc)  # 生成两个正弦信号

f = rfftfreq(n, T)  # 计算FFT的频率范围，从0Hz到Nyquist频率
XX = (rfft(x_) / n for x_ in xx)  # 对每个正弦信号进行单边FFT，并归一化幅度

i0, i1 = 15, 25  # 定义频率连续区间的索引范围
f_cont = np.linspace(f[i0], f[i1], 501)  # 在频率范围内生成连续频率值

fg1, axx = plt.subplots(1, 2, sharey='all', tight_layout=True,
                        figsize=(6., 3.))  # 创建一个包含两个子图的图形窗口

for c_, (ax_, X_, fx_) in enumerate(zip(axx, XX, fcc)):
    Xc_ = (np.sinc(tau * (f_cont - fx_)) +
           np.sinc(tau * (f_cont + fx_))) / 2  # 计算连续频谱
    ax_.plot(f_cont, abs(Xc_), f'-C{c_}', alpha=.5, label=rf"$f_x={fx_}\,$Hz")  # 绘制连续频谱图像
    m_line, _, _, = ax_.stem(f[i0:i1+1], abs(X_[i0:i1+1]), markerfmt=f'dC{c_}',
                             linefmt=f'-C{c_}', basefmt=' ')  # 绘制离散样本频谱
    plt.setp(m_line, markersize=5)  # 设置样本频谱的标记大小

    ax_.legend(loc='upper left', frameon=False)  # 添加图例
    ax_.set(xlabel="Frequency $f$ in Hertz", xlim=(f[i0], f[i1]),
            ylim=(0, 0.59))  # 设置轴标签和范围

axx[0].set(ylabel=r'Magnitude $|X(f)/\tau|$')  # 设置y轴标签
fg1.suptitle("Continuous and Sampled Magnitude Spectrum ", x=0.55, y=0.93)  # 设置总标题
fg1.tight_layout()  # 调整子图布局
plt.show()  # 显示图形
```