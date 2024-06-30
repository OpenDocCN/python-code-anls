# `D:\src\scipysrc\scipy\doc\source\tutorial\examples\signal_SpectralAnalysis_ContinuousSpectralRepresentations.py`

```
import matplotlib.pyplot as plt  # 导入matplotlib的绘图模块
import numpy as np  # 导入numpy库，并用np作为别名

aa = [1, 1, 2, 2]  # 定义幅度列表
taus = [1, 2, 1, 2]  # 定义持续时间列表

# 创建一个包含3个子图的Figure对象fg0和对应的Axes对象axx数组
fg0, axx = plt.subplots(3, 1, sharex='all', tight_layout=True, figsize=(6., 4.))

# 设置第一个子图的标题和y轴标签
axx[0].set(title=r"Spectrum $|X(f)|$", ylabel="V/Hz")

# 设置第二个子图的标题和y轴标签
axx[1].set(title=r"Magnitude Spectrum $|X(f)/\tau|$ ", ylabel=r"V")

# 设置第三个子图的标题、y轴标签和x轴标签
axx[2].set(title=r"Amplitude Spectral Density $|X(f)/\sqrt{\tau}|$",
           ylabel=r"$\operatorname{V} / \sqrt{\operatorname{Hz}}$",
           xlabel="Frequency $f$ in Hertz")

x_labels, x_ticks = [], []  # 初始化用于存储x轴标签和刻度的空列表
f = np.linspace(-2.5, 2.5, 400)  # 生成从-2.5到2.5的等间距400个数的数组作为频率f的取值

# 遍历aa和taus列表中的每对元素，同时使用索引c_进行迭代
for c_, (a_, tau_) in enumerate(zip(aa, taus), start=1):
    # 计算每对幅度a_和持续时间tau_对应的信号的幅度谱密度
    aZ_, f_ = abs(a_ * tau_ * np.sinc(tau_ * f) / 2), f + c_ * 5
    
    # 在第一个子图上绘制频率f_和幅度谱密度aZ_
    axx[0].plot(f_, aZ_)
    
    # 在第二个子图上绘制频率f_和幅度谱密度除以持续时间tau_得到的幅度谱
    axx[1].plot(f_, aZ_ / tau_)
    
    # 在第三个子图上绘制频率f_和幅度谱密度除以持续时间tau_的平方根得到的幅度谱密度
    axx[2].plot(f_, aZ_ / np.sqrt(tau_))
    
    # 为当前迭代生成的标签字符串，并添加到x_labels列表中
    x_labels.append(rf"$a={a_:g}$, $\tau={tau_:g}$")
    
    # 生成并添加当前迭代所在位置的x轴刻度值到x_ticks列表中
    x_ticks.append(c_ * 5)

# 设置第三个子图的x轴刻度和标签
axx[2].set_xticks(x_ticks)
axx[2].set_xticklabels(x_labels)

# 显示绘制的图形
plt.show()
```