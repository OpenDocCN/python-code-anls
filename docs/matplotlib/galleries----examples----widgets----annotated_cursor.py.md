# `D:\src\scipysrc\matplotlib\galleries\examples\widgets\annotated_cursor.py`

```py
"""
================
Annotated cursor
================

Display a data cursor including a text box, which shows the plot point close
to the mouse pointer.

The new cursor inherits from `~matplotlib.widgets.Cursor` and demonstrates the
creation of new widgets and their event callbacks.

See also the :doc:`cross hair cursor
</gallery/event_handling/cursor_demo>`, which implements a cursor tracking the
plotted data, but without using inheritance and without displaying the
currently tracked coordinates.

.. note::
    The figure related to this example does not show the cursor, because that
    figure is automatically created in a build queue, where the first mouse
    movement, which triggers the cursor creation, is missing.

"""
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 numpy 模块

from matplotlib.backend_bases import MouseEvent  # 导入 MouseEvent 类
from matplotlib.widgets import Cursor  # 导入 Cursor 类


class AnnotatedCursor(Cursor):
    """
    A crosshair cursor like `~matplotlib.widgets.Cursor` with a text showing \
    the current coordinates.

    For the cursor to remain responsive you must keep a reference to it.
    The data of the axis specified as *dataaxis* must be in ascending
    order. Otherwise, the `numpy.searchsorted` call might fail and the text
    disappears. You can satisfy the requirement by sorting the data you plot.
    Usually the data is already sorted (if it was created e.g. using
    `numpy.linspace`), but e.g. scatter plots might cause this problem.
    The cursor sticks to the plotted line.

    Parameters
    ----------
    line : `matplotlib.lines.Line2D`
        The plot line from which the data coordinates are displayed.

    numberformat : `python format string <https://docs.python.org/3/\
    library/string.html#formatstrings>`_, optional, default: "{0:.4g};{1:.4g}"
        The displayed text is created by calling *format()* on this string
        with the two coordinates.

    offset : (float, float) default: (5, 5)
        The offset in display (pixel) coordinates of the text position
        relative to the cross-hair.

    dataaxis : {"x", "y"}, optional, default: "x"
        If "x" is specified, the vertical cursor line sticks to the mouse
        pointer. The horizontal cursor line sticks to *line*
        at that x value. The text shows the data coordinates of *line*
        at the pointed x value. If you specify "y", it works in the opposite
        manner. But: For the "y" value, where the mouse points to, there might
        be multiple matching x values, if the plotted function is not biunique.
        Cursor and text coordinate will always refer to only one x value.
        So if you use the parameter value "y", ensure that your function is
        biunique.

    Other Parameters
    ----------------
    textprops : `matplotlib.text` properties as dictionary
        Specifies the appearance of the rendered text object.
    """
    # 定义一个新的游标类 AnnotatedCursor，继承自 matplotlib.widgets.Cursor

    def __init__(self, line, numberformat="{0:.4g};{1:.4g}", offset=(5, 5),
                 dataaxis="x", **textprops):
        """
        Initialize the AnnotatedCursor object.

        Parameters
        ----------
        line : `matplotlib.lines.Line2D`
            The plot line from which the data coordinates are displayed.

        numberformat : `python format string`, optional
            The format string for displaying the coordinates.

        offset : (float, float), optional
            The offset in pixels of the text relative to the crosshair.

        dataaxis : {"x", "y"}, optional
            Specifies whether the cursor follows the x or y axis.

        **textprops : `matplotlib.text` properties as keyword arguments
            Additional properties for the text display.
        """
        super().__init__(line, **textprops)  # 调用父类 Cursor 的初始化方法
        self.numberformat = numberformat  # 设置坐标显示格式
        self.offset = offset  # 设置文本显示位置的偏移量
        self.dataaxis = dataaxis  # 设置游标跟随的数据轴

        # 提示信息
        if dataaxis == "x":
            msg = (
                "The cursor line sticks to the mouse pointer. The text shows "
                "the data coordinates of the plotted line at the pointed x value."
            )
        elif dataaxis == "y":
            msg = (
                "The cursor line sticks to the plotted line at the specified x value. "
                "The text shows the data coordinates of the plotted line at the "
                "pointed y value."
            )
        else:
            msg = "Invalid dataaxis parameter. Please specify 'x' or 'y'."
        
        # 打印提示信息
        print(msg)
    """
    **cursorargs : `matplotlib.widgets.Cursor` properties
        Arguments passed to the internal `~matplotlib.widgets.Cursor` instance.
        The `matplotlib.axes.Axes` argument is mandatory! The parameter
        *useblit* can be set to *True* in order to achieve faster rendering.
    """

    # 初始化函数，用于创建一个显示光标位置的工具
    def __init__(self, line, numberformat="{0:.4g};{1:.4g}", offset=(5, 5),
                 dataaxis='x', textprops=None, **cursorargs):
        # 如果没有提供文本属性，则设为空字典
        if textprops is None:
            textprops = {}
        
        # 将传入的线对象保存为属性，用于显示坐标
        self.line = line
        # 格式化字符串，用于显示坐标值
        self.numberformat = numberformat
        # 文本位置的偏移量，作为numpy数组保存
        self.offset = np.array(offset)
        # 指定光标检索位置的轴，默认为'x'
        self.dataaxis = dataaxis

        # 调用父类构造函数进行初始化
        # 绘制光标并保存背景用于快速渲染
        # 将ax保存为类属性
        super().__init__(**cursorargs)

        # 设置文本初始位置为线条的第一个数据点位置
        self.set_position(self.line.get_xdata()[0], self.line.get_ydata()[0])
        
        # 创建一个不可见的动画文本对象
        self.text = self.ax.text(
            self.ax.get_xbound()[0],  # 文本的初始x位置
            self.ax.get_ybound()[0],  # 文本的初始y位置
            "0, 0",  # 初始文本内容
            animated=bool(self.useblit),  # 是否使用快速渲染来进行动画
            visible=False,  # 初始不可见
            **textprops  # 传入的文本属性
        )
        
        # 上次绘制光标的位置，初始为None
        self.lastdrawnplotpoint = None
    def set_position(self, xpos, ypos):
        """
        Finds the coordinates, which have to be shown in text.

        The behaviour depends on the *dataaxis* attribute. Function looks
        up the matching plot coordinate for the given mouse position.

        Parameters
        ----------
        xpos : float
            The current x position of the cursor in data coordinates.
            Important if *dataaxis* is set to 'x'.
        ypos : float
            The current y position of the cursor in data coordinates.
            Important if *dataaxis* is set to 'y'.

        Returns
        -------
        ret : {2D array-like, None}
            The coordinates which should be displayed.
            *None* is the fallback value.
        """

        # Get plot line data
        xdata = self.line.get_xdata()  # 获取绘图线的 x 数据
        ydata = self.line.get_ydata()  # 获取绘图线的 y 数据

        # The dataaxis attribute decides, in which axis we look up which cursor
        # coordinate.
        if self.dataaxis == 'x':
            pos = xpos  # 如果 dataaxis 设置为 'x'，使用 xpos 作为位置
            data = xdata  # 使用 xdata 作为数据
            lim = self.ax.get_xlim()  # 获取 x 轴的显示范围
        elif self.dataaxis == 'y':
            pos = ypos  # 如果 dataaxis 设置为 'y'，使用 ypos 作为位置
            data = ydata  # 使用 ydata 作为数据
            lim = self.ax.get_ylim()  # 获取 y 轴的显示范围
        else:
            raise ValueError(f"The data axis specifier {self.dataaxis} should "
                             f"be 'x' or 'y'")  # 抛出异常，如果 dataaxis 不是 'x' 或 'y'

        # If position is valid and in valid plot data range.
        if pos is not None and lim[0] <= pos <= lim[-1]:
            # Find closest x value in sorted x vector.
            # This requires the plotted data to be sorted.
            index = np.searchsorted(data, pos)  # 在排序后的 data 中查找 pos 的索引
            # Return none, if this index is out of range.
            if index < 0 or index >= len(data):
                return None  # 如果索引超出范围，返回 None
            # Return plot point as tuple.
            return (xdata[index], ydata[index])  # 返回匹配坐标点的元组形式

        # Return none if there is no good related point for this x position.
        return None  # 如果没有与该 x 位置相关的好的点，则返回 None

    def clear(self, event):
        """
        Overridden clear callback for cursor, called before drawing the figure.
        """

        # The base class saves the clean background for blitting.
        # Text and cursor are invisible,
        # until the first mouse move event occurs.
        super().clear(event)  # 调用父类的清除方法，用于 blitting
        if self.ignore(event):
            return  # 如果忽略当前事件，则返回

        self.text.set_visible(False)  # 设置文本不可见

    def _update(self):
        """
        Overridden method for either blitting or drawing the widget canvas.

        Passes call to base class if blitting is activated, only.
        In other cases, one draw_idle call is enough, which is placed
        explicitly in this class (see *onmove()*).
        In that case, `~matplotlib.widgets.Cursor` is not supposed to draw
        something using this method.
        """

        if self.useblit:
            super()._update()  # 如果使用 blit，将调用传递给基类的更新方法
# 创建一个新的图形窗口和子图，指定图形的尺寸为8x6英寸
fig, ax = plt.subplots(figsize=(8, 6))

# 设置子图的标题为"Cursor Tracking x Position"
ax.set_title("Cursor Tracking x Position")

# 生成一个从-5到5的等间距的包含1000个点的数组作为x坐标
x = np.linspace(-5, 5, 1000)

# 计算y值，这里y=x^2
y = x**2

# 绘制线条并将其赋给变量line
line, = ax.plot(x, y)

# 设置x轴的显示范围为-5到5
ax.set_xlim(-5, 5)

# 设置y轴的显示范围为0到25
ax.set_ylim(0, 25)

# 创建AnnotatedCursor对象，用于显示带有注释的光标追踪效果
# 这里演示了更高级的调用方式，设置了格式化数字、数据轴为x、偏移量、文本属性、使用双缓存等参数
cursor = AnnotatedCursor(
    line=line,
    numberformat="{0:.2f}\n{1:.2f}",  # 格式化数字显示的格式
    dataaxis='x',  # 指定数据轴为x
    offset=[10, 10],  # 设置注释文本的偏移量
    textprops={'color': 'blue', 'fontweight': 'bold'},  # 设置注释文本的样式
    ax=ax,  # 指定子图对象
    useblit=True,  # 启用双缓存以提高性能
    color='red',  # 设置光标线的颜色
    linewidth=2  # 设置光标线的宽度
)

# 模拟鼠标移动事件到(-2, 10)，用于在线文档的演示
t = ax.transData
MouseEvent(
    "motion_notify_event", ax.figure.canvas, *t.transform((-2, 10))
)._process()

# 显示图形
plt.show()

# %%
# 非一对一函数的问题
# --------------------
# 这里演示了使用dataaxis='y'参数时可能遇到的问题。
# 文本注释现在查找当前光标y位置对应的匹配x值，而不是相反。
# 将光标悬停在y=4处，会发现有两个x值(-2和2)对应于该y值。
# 函数是唯一的，但不是一对一的。文本中只显示一个值。

# 创建一个新的图形窗口和子图，指定图形的尺寸为8x6英寸
fig, ax = plt.subplots(figsize=(8, 6))

# 设置子图的标题为"Cursor Tracking y Position"
ax.set_title("Cursor Tracking y Position")

# 绘制线条并将其赋给变量line
line, = ax.plot(x, y)

# 设置x轴的显示范围为-5到5
ax.set_xlim(-5, 5)

# 设置y轴的显示范围为0到25
ax.set_ylim(0, 25)

# 创建AnnotatedCursor对象，用于显示带有注释的光标追踪效果
# 这里设置了格式化数字、数据轴为y、偏移量、文本属性、使用双缓存等参数
cursor = AnnotatedCursor(
    line=line,
    numberformat="{0:.2f}\n{1:.2f}",  # 格式化数字显示的格式
    dataaxis='y',  # 指定数据轴为y
    offset=[10, 10],  # 设置注释文本的偏移量
    textprops={'color': 'blue', 'fontweight': 'bold'},  # 设置注释文本的样式
    ax=ax,  # 指定子图对象
    useblit=True,  # 启用双缓存以提高性能
    color='red',  # 设置光标线的颜色
    linewidth=2  # 设置光标线的宽度
)

# 模拟鼠标移动事件到(-2, 10)，用于在线文档的演示
t = ax.transData
MouseEvent(
    "motion_notify_event", ax.figure.canvas, *t.transform((-2, 10))
)._process()

# 显示图形
plt.show()
```