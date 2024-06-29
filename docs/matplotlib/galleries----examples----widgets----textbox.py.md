# `D:\src\scipysrc\matplotlib\galleries\examples\widgets\textbox.py`

```py
"""
=======
Textbox
=======

The Textbox widget lets users interactively provide text input, including
formulas. In this example, the plot is updated using the `.on_submit` method.
This method triggers the execution of the *submit* function when the
user presses enter in the textbox or leaves the textbox.

Note:  The `matplotlib.widgets.TextBox` widget is different from the following
static elements: :ref:`annotations` and
:doc:`/gallery/text_labels_and_annotations/placing_text_boxes`.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库
import numpy as np  # 导入 numpy 库

# 从 matplotlib.widgets 模块中导入 TextBox 类
from matplotlib.widgets import TextBox

# 创建一个图形窗口和轴
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)  # 调整图形的底部边距

# 创建一个时间数组
t = np.arange(-2.0, 2.0, 0.001)
l, = ax.plot(t, np.zeros_like(t), lw=2)  # 绘制初始图形


def submit(expression):
    """
    Update the plotted function to the new math *expression*.

    *expression* is a string using "t" as its independent variable, e.g.
    "t ** 3".
    """
    # 根据输入的表达式更新图形的数据
    ydata = eval(expression, {'np': np}, {'t': t})
    l.set_ydata(ydata)  # 设置图形的纵坐标数据
    ax.relim()  # 重新计算轴的数据限制
    ax.autoscale_view()  # 自动调整视图范围
    plt.draw()  # 重新绘制图形


# 在图形中添加一个文本框
axbox = fig.add_axes([0.1, 0.05, 0.8, 0.075])
text_box = TextBox(axbox, "Evaluate", textalignment="center")
text_box.on_submit(submit)  # 设置文本框的提交事件处理函数为 submit
text_box.set_val("t ** 2")  # 初始化文本框的值，并触发一次 submit 函数的执行

plt.show()  # 显示图形窗口
```