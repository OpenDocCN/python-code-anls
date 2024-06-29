# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\custom_figure_class.py`

```py
"""
========================
Custom Figure subclasses
========================

You can pass a `.Figure` subclass to `.pyplot.figure` if you want to change
the default behavior of the figure.

This example defines a `.Figure` subclass ``WatermarkFigure`` that accepts an
additional parameter ``watermark`` to display a custom watermark text. The
figure is created using the ``FigureClass`` parameter of `.pyplot.figure`.
The additional ``watermark`` parameter is passed on to the subclass
constructor.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 numpy 数学计算模块

from matplotlib.figure import Figure  # 从 matplotlib 的 figure 模块导入 Figure 类


class WatermarkFigure(Figure):
    """A figure with a text watermark."""

    def __init__(self, *args, watermark=None, **kwargs):
        super().__init__(*args, **kwargs)  # 调用父类 Figure 的构造方法，传递所有参数

        if watermark is not None:
            # 定义水印文本的样式和属性
            bbox = dict(boxstyle='square', lw=3, ec='gray',
                        fc=(0.9, 0.9, .9, .5), alpha=0.5)
            # 在图形中心绘制水印文本
            self.text(0.5, 0.5, watermark,
                      ha='center', va='center', rotation=30,
                      fontsize=40, color='gray', alpha=0.5, bbox=bbox)


x = np.linspace(-3, 3, 201)  # 创建一个从 -3 到 3 的间隔为 0.03 的数组
y = np.tanh(x) + 0.1 * np.cos(5 * x)  # 计算 tanh(x) + 0.1 * cos(5x) 的值

plt.figure(FigureClass=WatermarkFigure, watermark='draft')  # 创建一个 WatermarkFigure 实例，并传入水印文本 'draft'
plt.plot(x, y)  # 绘制 x 和 y 的图形


# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.pyplot.figure`  # 显示了使用 matplotlib.pyplot.figure 函数
#    - `matplotlib.figure.Figure`  # 显示了使用 matplotlib.figure.Figure 类
#    - `matplotlib.figure.Figure.text`  # 显示了使用 matplotlib.figure.Figure 类的 text 方法
```