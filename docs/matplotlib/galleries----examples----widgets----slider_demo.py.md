# `D:\src\scipysrc\matplotlib\galleries\examples\widgets\slider_demo.py`

```py
"""
======
Slider
======

In this example, sliders are used to control the frequency and amplitude of
a sine wave.

See :doc:`/gallery/widgets/slider_snap_demo` for an example of having
the ``Slider`` snap to discrete values.

See :doc:`/gallery/widgets/range_slider` for an example of using
a ``RangeSlider`` to define a range of values.
"""

# 导入必要的库和模块
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

from matplotlib.widgets import Button, Slider  # 从 matplotlib.widgets 导入 Button 和 Slider 模块


# 定义参数化的函数，用于绘制的正弦波
def f(t, amplitude, frequency):
    return amplitude * np.sin(2 * np.pi * frequency * t)

t = np.linspace(0, 1, 1000)  # 生成 0 到 1 之间的 1000 个均匀间隔的数值作为时间 t

# 定义初始参数
init_amplitude = 5
init_frequency = 3

# 创建图形和要操作的线条
fig, ax = plt.subplots()  # 创建图形对象和坐标轴对象
line, = ax.plot(t, f(t, init_amplitude, init_frequency), lw=2)  # 绘制初始的正弦波，并保存线条对象
ax.set_xlabel('Time [s]')  # 设置 x 轴标签为时间 [s]

# 调整主图以留出空间放置滑动条
fig.subplots_adjust(left=0.25, bottom=0.25)

# 创建水平滑动条来控制频率
axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label='Frequency [Hz]',
    valmin=0.1,
    valmax=30,
    valinit=init_frequency,
)

# 创建垂直滑动条来控制振幅
axamp = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
amp_slider = Slider(
    ax=axamp,
    label="Amplitude",
    valmin=0,
    valmax=10,
    valinit=init_amplitude,
    orientation="vertical"
)


# 定义当滑动条数值改变时调用的更新函数
def update(val):
    line.set_ydata(f(t, amp_slider.val, freq_slider.val))  # 更新线条的 y 数据
    fig.canvas.draw_idle()  # 在图形上绘制更新后的数据


# 将更新函数与每个滑动条注册
freq_slider.on_changed(update)
amp_slider.on_changed(update)

# 创建一个按钮，用于将滑动条重置为初始值
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    freq_slider.reset()  # 重置频率滑动条
    amp_slider.reset()  # 重置振幅滑动条
button.on_clicked(reset)

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.widgets.Button`
#    - `matplotlib.widgets.Slider`
```