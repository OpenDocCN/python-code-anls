# `D:\src\scipysrc\matplotlib\galleries\examples\pyplots\pyplot_text.py`

```py
"""
==============================
Text and mathtext using pyplot
==============================

Set the special text objects `~.pyplot.title`, `~.pyplot.xlabel`, and
`~.pyplot.ylabel` through the dedicated pyplot functions.  Additional text
objects can be placed in the Axes using `~.pyplot.text`.

You can use TeX-like mathematical typesetting in all texts; see also
:ref:`mathtext`.

.. redirect-from:: /gallery/pyplots/pyplot_mathtext
"""

# 导入matplotlib.pyplot库，用于绘图
import matplotlib.pyplot as plt
# 导入numpy库，用于数值计算
import numpy as np

# 生成时间数组t，从0到2秒，步长为0.01秒
t = np.arange(0.0, 2.0, 0.01)
# 计算正弦波信号s，频率为1Hz
s = np.sin(2*np.pi*t)

# 绘制t和s的图形
plt.plot(t, s)
# 在坐标(0, -1)处添加文本'Hello, world!'，字体大小为15
plt.text(0, -1, r'Hello, world!', fontsize=15)
# 设置图形的标题，使用数学公式$\mathcal{A}\sin(\omega t)$，字体大小为20
plt.title(r'$\mathcal{A}\sin(\omega t)$', fontsize=20)
# 设置x轴的标签为'Time [s]'
plt.xlabel('Time [s]')
# 设置y轴的标签为'Voltage [mV]'
plt.ylabel('Voltage [mV]')
# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.pyplot.hist`
#    - `matplotlib.pyplot.xlabel`
#    - `matplotlib.pyplot.ylabel`
#    - `matplotlib.pyplot.text`
#    - `matplotlib.pyplot.grid`
#    - `matplotlib.pyplot.show`
```