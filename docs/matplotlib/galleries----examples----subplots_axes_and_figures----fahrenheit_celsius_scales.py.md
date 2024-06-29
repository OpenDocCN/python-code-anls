# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\fahrenheit_celsius_scales.py`

```py
"""
=================================
Different scales on the same Axes
=================================

Demo of how to display two scales on the left and right y-axis.

This example uses the Fahrenheit and Celsius scales.
"""
# 导入 matplotlib.pyplot 作为 plt，导入 numpy 作为 np
import matplotlib.pyplot as plt
import numpy as np


def fahrenheit2celsius(temp):
    """
    Returns temperature in Celsius given Fahrenheit temperature.
    """
    # 根据给定的华氏温度计算摄氏温度
    return (5. / 9.) * (temp - 32)


def make_plot():
    # 定义一个闭包函数，注册为回调函数
    def convert_ax_c_to_celsius(ax_f):
        """
        Update second axis according to first axis.
        """
        # 获取第一个轴的当前 ylim
        y1, y2 = ax_f.get_ylim()
        # 根据第一个轴的 ylim 更新第二个轴的 ylim，转换为摄氏温度
        ax_c.set_ylim(fahrenheit2celsius(y1), fahrenheit2celsius(y2))
        # 更新绘图
        ax_c.figure.canvas.draw()

    # 创建一个新的图形和两个子图，ax_f 是主轴，ax_c 是共享x轴的次轴
    fig, ax_f = plt.subplots()
    ax_c = ax_f.twinx()

    # 当 ax1 的 ylim 改变时，自动更新 ax2 的 ylim
    ax_f.callbacks.connect("ylim_changed", convert_ax_c_to_celsius)
    # 在 ax1 上绘制从 -40 到 120 的等间距数据点，总共 100 个点
    ax_f.plot(np.linspace(-40, 120, 100))
    # 设置 ax1 的 x 轴范围
    ax_f.set_xlim(0, 100)

    # 设置图的标题
    ax_f.set_title('Two scales: Fahrenheit and Celsius')
    # 设置 ax1 的 y 轴标签
    ax_f.set_ylabel('Fahrenheit')
    # 设置 ax2 的 y 轴标签
    ax_c.set_ylabel('Celsius')

    # 显示图形
    plt.show()

make_plot()
```