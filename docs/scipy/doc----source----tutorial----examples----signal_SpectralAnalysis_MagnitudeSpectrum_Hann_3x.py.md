# `D:\src\scipysrc\scipy\doc\source\tutorial\examples\signal_SpectralAnalysis_MagnitudeSpectrum_Hann_3x.py`

```
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库，用于绘图
import numpy as np  # 导入numpy库，用于数值计算

from scipy.fft import rfft, rfftfreq  # 从scipy.fft模块导入rfft和rfftfreq函数，用于快速傅里叶变换
from scipy.signal.windows import hann  # 从scipy.signal.windows模块导入hann函数，用于生成汉宁窗口

n, T = 100, 0.01  # 样本数量和采样间隔
tau = n*T  # 总时间
q = 3  # 过采样因子
t = np.arange(n) * T  # 时间向量

fcc = (20, 20.5)  # 正弦波的频率
xx = [np.sin(2 * np.pi * fc_ * t) for fc_ in fcc]  # 生成两个正弦信号

w = hann(n)  # 生成长度为n的汉宁窗口
c_w = abs(sum(w))  # 汉宁窗口的归一化常数

f_X = rfftfreq(n, T)  # 计算频率轴，从0到Nyquist频率
XX = (rfft(x_ * w) / c_w for x_ in xx)  # 计算单边幅度谱

# 过采样后的频谱:
f_Y = rfftfreq(n*q, T)  # 计算频率轴，从0到Nyquist频率
YY = (rfft(x_ * w, n=q*n) / c_w for x_ in xx)  # 计算单边幅度谱

i0, i1 = 15, 25  # 第一个图中频率区间的索引范围
j0, j1 = i0*q, i1*q  # 第二个图中频率区间的索引范围

fg1, axx = plt.subplots(1, 2, sharey='all', tight_layout=True,  # 创建两个子图
                        figsize=(6., 3.))
for c_, (ax_, X_, Y_, fx_) in enumerate(zip(axx, XX, YY, fcc)):
    ax_.plot(f_Y[j0:j1 + 1], abs(Y_[j0:j1 + 1]), f'.-C{c_}',  # 绘制过采样频谱
             label=rf"$f_x={fx_}\,$Hz")
    m_ln, s_ln, _, = ax_.stem(f_X[i0:i1 + 1], abs(X_[i0:i1 + 1]), basefmt=' ',  # 绘制原始频谱
                              markerfmt=f'dC{c_}', linefmt=f'-C{c_}')
    plt.setp(m_ln, markersize=5)
    plt.setp(s_ln, alpha=0.5)

    ax_.legend(loc='upper left', frameon=False)  # 添加图例
    ax_.set(xlabel="Frequency $f$ in Hertz", xlim=(f_X[15], f_X[25]),  # 设置坐标轴标签和范围
            ylim=(0, 0.59))

axx[0].set(ylabel=r'Magnitude $|X(f)/\tau|$')  # 设置第一个子图的y轴标签
fg1.suptitle(r"Magnitude Spectrum (Hann window, $%d\times$oversampled)" % q,  # 设置整体标题
             x=0.55, y=0.93)
plt.show()  # 显示图形
```