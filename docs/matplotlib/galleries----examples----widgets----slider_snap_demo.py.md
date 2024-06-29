# `D:\src\scipysrc\matplotlib\galleries\examples\widgets\slider_snap_demo.py`

```py
# 导入必要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，并简称为 plt
import numpy as np  # 导入 numpy 库，并简称为 np
from matplotlib.widgets import Button, Slider  # 从 matplotlib.widgets 模块导入 Button 和 Slider 类

# 创建时间数组 t，范围从 0 到 1，步长为 0.001
t = np.arange(0.0, 1.0, 0.001)
a0 = 5  # 初始振幅设为 5
f0 = 3  # 初始频率设为 3
s = a0 * np.sin(2 * np.pi * f0 * t)  # 计算正弦波信号 s

# 创建绘图窗口和轴
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.25)  # 调整绘图窗口底部空白部分
l, = ax.plot(t, s, lw=2)  # 绘制正弦波曲线

# 创建用于调节振幅和频率的滑动条轴
ax_freq = fig.add_axes([0.25, 0.1, 0.65, 0.03])  # 频率滑动条的位置和大小
ax_amp = fig.add_axes([0.25, 0.15, 0.65, 0.03])  # 振幅滑动条的位置和大小

# 定义振幅滑动条的允许取值范围
allowed_amplitudes = np.concatenate([np.linspace(.1, 5, 100), [6, 7, 8, 9]])

# 创建振幅和频率滑动条对象
samp = Slider(
    ax_amp, "Amp", 0.1, 9.0,
    valinit=a0, valstep=allowed_amplitudes,
    color="green"  # 设置滑动条颜色为绿色
)

sfreq = Slider(
    ax_freq, "Freq", 0, 10*np.pi,
    valinit=2*np.pi, valstep=np.pi,
    initcolor='none'  # 移除标记初始位置的线
)

# 定义更新函数，用于更新正弦波曲线的振幅和频率
def update(val):
    amp = samp.val  # 获取当前振幅滑动条的值
    freq = sfreq.val  # 获取当前频率滑动条的值
    l.set_ydata(amp*np.sin(2*np.pi*freq*t))  # 更新正弦波的 y 值数据
    fig.canvas.draw_idle()  # 重新绘制图形

# 将更新函数绑定到滑动条的值改变事件上
sfreq.on_changed(update)
samp.on_changed(update)

# 创建重置按钮的轴并定义重置按钮的行为
ax_reset = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(ax_reset, 'Reset', hovercolor='0.975')  # 创建标签为 Reset 的按钮

def reset(event):
    sfreq.reset()  # 重置频率滑动条到初始值
    samp.reset()   # 重置振幅滑动条到初始值
button.on_clicked(reset)  # 将 reset 函数绑定到按钮点击事件上

plt.show()  # 显示绘图窗口
```