# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\multiple_figs_demo.py`

```py
"""
===================================
Managing multiple figures in pyplot
===================================

`matplotlib.pyplot` uses the concept of a *current figure* and *current Axes*.
Figures are identified via a figure number that is passed to `~.pyplot.figure`.
The figure with the given number is set as *current figure*. Additionally, if
no figure with the number exists, a new one is created.

.. note::

    We discourage working with multiple figures through the implicit pyplot
    interface because managing the *current figure* is cumbersome and
    error-prone. Instead, we recommend using the explicit approach and call
    methods on Figure and Axes instances. See :ref:`api_interfaces` for an
    explanation of the trade-offs between the implicit and explicit interfaces.

"""
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.0, 2.0, 0.01)
s1 = np.sin(2*np.pi*t)
s2 = np.sin(4*np.pi*t)

# %%
# Create figure 1

# 创建一个新的图形对象，编号为1
plt.figure(1)
# 在当前图形对象中创建两个子图中的第一个
plt.subplot(211)
# 绘制sin函数曲线，使用t作为横坐标，s1作为纵坐标
plt.plot(t, s1)
# 在当前图形对象中创建两个子图中的第二个
plt.subplot(212)
# 绘制2*s1的曲线，使用t作为横坐标
plt.plot(t, 2*s1)

# %%
# Create figure 2

# 创建一个新的图形对象，编号为2
plt.figure(2)
# 绘制s2的曲线，使用t作为横坐标
plt.plot(t, s2)

# %%
# Now switch back to figure 1 and make some changes

# 切换回图形对象1，并进行一些更改
plt.figure(1)
# 在第一个子图中绘制s2的散点图，保留原有曲线
plt.subplot(211)
plt.plot(t, s2, 's')
# 获取当前Axes对象
ax = plt.gca()
# 将横坐标的刻度标签设为空列表，即不显示横坐标刻度
ax.set_xticklabels([])

# 显示图形对象1及其更改
plt.show()

# %%
# .. tags:: component: figure, plot-type: line
```