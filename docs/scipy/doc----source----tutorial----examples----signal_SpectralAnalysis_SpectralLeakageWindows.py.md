# `D:\src\scipysrc\scipy\doc\source\tutorial\examples\signal_SpectralAnalysis_SpectralLeakageWindows.py`

```
# 导入 matplotlib.pyplot 库，用于绘制图形
import matplotlib.pyplot as plt
# 导入 numpy 库，用于数值计算
import numpy as np

# 从 scipy.fft 库中导入 rfft 和 rfftfreq 函数，用于执行快速傅里叶变换相关操作
from scipy.fft import rfft, rfftfreq
# 从 scipy.signal 库中导入 get_window 函数，用于获取窗口函数
from scipy.signal import get_window

# 定义两个变量 n 和 n_zp，分别表示没有零填充和有零填充的样本数
n, n_zp = 128, 16384
# 生成长度为 n 的时间序列 t
t = np.arange(n)
# 使用 rfftfreq 函数生成长度为 n_zp 的频率序列 f，采样间隔为 1/n
f = rfftfreq(n_zp, 1 / n)

# 定义窗口函数的名称列表
ww = ['boxcar', 'hann', 'hamming', 'tukey', 'blackman', 'flattop']
# 创建一个图形对象 fg0 和包含子图的轴对象 axx
fg0, axx = plt.subplots(len(ww), 1, sharex='all', sharey='all', figsize=(6., 4.))

# 遍历窗口函数名称和对应的子图轴对象
for c_, (w_name_, ax_) in enumerate(zip(ww, axx)):
    # 使用 get_window 函数获取名称为 w_name_ 的窗口函数 w_
    w_ = get_window(w_name_, n, fftbins=False)
    # 对窗口函数进行归一化，并进行快速傅里叶变换，并计算其绝对值后取对数得到 dB 值
    W_ = rfft(w_ / abs(sum(w_)), n=n_zp)
    W_dB = 20*np.log10(np.maximum(abs(W_), 1e-250))
    # 在子图上绘制频谱图
    ax_.plot(f, W_dB, f'C{c_}-', label=w_name_)
    # 在子图上添加窗口函数名称的文本标签
    ax_.text(0.1, -50, w_name_, color=f'C{c_}', verticalalignment='bottom',
             horizontalalignment='left', bbox={'color': 'white', 'pad': 0})
    # 设置 y 轴刻度为 [-20, -60]，并显示 x 轴的网格线
    ax_.set_yticks([-20, -60])
    ax_.grid(axis='x')

# 设置第一个子图的标题
axx[0].set_title("Spectral Leakage of various Windows")
# 设置整个图形对象 fg0 的 y 轴标签
fg0.supylabel(r"Normalized Magnitude $20\,\log_{10}|W(f)/c^\operatorname{amp}|$ in dB",
              x=0.04, y=0.5, fontsize='medium')
# 设置最后一个子图的 x 轴标签，并限制 x 轴和 y 轴的范围
axx[-1].set(xlabel=r"Normalized frequency $f/\Delta f$ in bins",
            xlim=(0, 9), ylim=(-75, 3))

# 调整子图布局，设置水平间距为 0.4
fg0.tight_layout(h_pad=0.4)
# 显示图形
plt.show()
```