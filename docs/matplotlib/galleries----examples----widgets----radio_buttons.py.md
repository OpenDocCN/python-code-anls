# `D:\src\scipysrc\matplotlib\galleries\examples\widgets\radio_buttons.py`

```py
"""
=============
Radio Buttons
=============

Using radio buttons to choose properties of your plot.

Radio buttons let you choose between multiple options in a visualization.
In this case, the buttons let the user choose one of the three different sine
waves to be shown in the plot.

Radio buttons may be styled using the *label_props* and *radio_props* parameters, which
each take a dictionary with keys of artist property names and values of lists of
settings with length matching the number of buttons.
"""

# 导入 matplotlib 库和 numpy 库
import matplotlib.pyplot as plt
import numpy as np

# 从 matplotlib.widgets 中导入 RadioButtons 类
from matplotlib.widgets import RadioButtons

# 生成时间序列数据
t = np.arange(0.0, 2.0, 0.01)
s0 = np.sin(2*np.pi*t)   # 生成频率为 1 Hz 的正弦波数据
s1 = np.sin(4*np.pi*t)   # 生成频率为 2 Hz 的正弦波数据
s2 = np.sin(8*np.pi*t)   # 生成频率为 4 Hz 的正弦波数据

# 创建画布和子图布局
fig, ax = plt.subplot_mosaic(
    [
        ['main', 'freq'],
        ['main', 'color'],
        ['main', 'linestyle'],
    ],
    width_ratios=[5, 1],
    layout='constrained',
)

# 在主子图中绘制频率为 1 Hz 的正弦波，并设置线宽和颜色
l, = ax['main'].plot(t, s0, lw=2, color='red')

# 设置单选按钮背景色
radio_background = 'lightgoldenrodyellow'

# 设置频率选择单选按钮，标签为 '1 Hz', '2 Hz', '4 Hz'，配置标签属性和单选按钮属性
ax['freq'].set_facecolor(radio_background)
radio = RadioButtons(ax['freq'], ('1 Hz', '2 Hz', '4 Hz'),
                     label_props={'color': 'cmy', 'fontsize': [12, 14, 16]},
                     radio_props={'s': [16, 32, 64]})


def hzfunc(label):
    # 根据选择的标签切换对应的正弦波数据
    hzdict = {'1 Hz': s0, '2 Hz': s1, '4 Hz': s2}
    ydata = hzdict[label]
    l.set_ydata(ydata)
    fig.canvas.draw()

# 绑定频率单选按钮的点击事件
radio.on_clicked(hzfunc)

# 设置颜色选择单选按钮，标签为 'red', 'blue', 'green'，配置标签属性和单选按钮属性
ax['color'].set_facecolor(radio_background)
radio2 = RadioButtons(
    ax['color'], ('red', 'blue', 'green'),
    label_props={'color': ['red', 'blue', 'green']},
    radio_props={
        'facecolor': ['red', 'blue', 'green'],
        'edgecolor': ['darkred', 'darkblue', 'darkgreen'],
    })


def colorfunc(label):
    # 根据选择的颜色设置曲线的颜色
    l.set_color(label)
    fig.canvas.draw()

# 绑定颜色单选按钮的点击事件
radio2.on_clicked(colorfunc)

# 设置线型选择单选按钮，标签为 '-', '--', '-.', ':'，默认配置线型属性
ax['linestyle'].set_facecolor(radio_background)
radio3 = RadioButtons(ax['linestyle'], ('-', '--', '-.', ':'))


def stylefunc(label):
    # 根据选择的线型设置曲线的线型
    l.set_linestyle(label)
    fig.canvas.draw()

# 绑定线型单选按钮的点击事件
radio3.on_clicked(stylefunc)

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.widgets.RadioButtons`
```