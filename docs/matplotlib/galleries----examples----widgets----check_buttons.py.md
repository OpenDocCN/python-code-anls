# `D:\src\scipysrc\matplotlib\galleries\examples\widgets\check_buttons.py`

```
"""
=============
Check buttons
=============

Turning visual elements on and off with check buttons.

This program shows the use of `.CheckButtons` which is similar to
check boxes. There are 3 different sine waves shown, and we can choose which
waves are displayed with the check buttons.

Check buttons may be styled using the *check_props*, *frame_props*, and *label_props*
parameters. The parameters each take a dictionary with keys of artist property names and
values of lists of settings with length matching the number of buttons.
"""

# 导入 matplotlib 的 pyplot 和 numpy 模块
import matplotlib.pyplot as plt
import numpy as np

# 从 matplotlib.widgets 模块导入 CheckButtons 类
from matplotlib.widgets import CheckButtons

# 生成时间序列 t
t = np.arange(0.0, 2.0, 0.01)
# 生成三个不同频率的正弦波
s0 = np.sin(2*np.pi*t)
s1 = np.sin(4*np.pi*t)
s2 = np.sin(6*np.pi*t)

# 创建图形窗口和坐标轴
fig, ax = plt.subplots()
# 绘制三条线，并设置初始可见性为 False
l0, = ax.plot(t, s0, visible=False, lw=2, color='black', label='1 Hz')
l1, = ax.plot(t, s1, lw=2, color='red', label='2 Hz')
l2, = ax.plot(t, s2, lw=2, color='green', label='3 Hz')

# 根据线条的标签创建一个字典，方便后续通过标签控制线条的可见性
lines_by_label = {l.get_label(): l for l in [l0, l1, l2]}
# 获取每条线的颜色，用于定制化 CheckButtons
line_colors = [l.get_color() for l in lines_by_label.values()]

# 在坐标轴的指定位置创建 CheckButtons 控件
rax = ax.inset_axes([0.0, 0.0, 0.12, 0.2])
check = CheckButtons(
    ax=rax,
    labels=lines_by_label.keys(),
    actives=[l.get_visible() for l in lines_by_label.values()],
    label_props={'color': line_colors},    # 设置标签文本颜色
    frame_props={'edgecolor': line_colors},    # 设置边框颜色
    check_props={'facecolor': line_colors},    # 设置选中框的颜色
)

# 定义回调函数，控制线条的可见性
def callback(label):
    ln = lines_by_label[label]
    ln.set_visible(not ln.get_visible())
    ln.figure.canvas.draw_idle()

# 将回调函数绑定到 CheckButtons 控件上
check.on_clicked(callback)

# 显示图形
plt.show()
```