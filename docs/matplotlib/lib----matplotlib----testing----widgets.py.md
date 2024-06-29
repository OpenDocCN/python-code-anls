# `D:\src\scipysrc\matplotlib\lib\matplotlib\testing\widgets.py`

```py
"""
========================
Widget testing utilities
========================

See also :mod:`matplotlib.tests.test_widgets`.
"""

from unittest import mock  # 导入 mock 模块，用于创建模拟对象

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块，并简写为 plt


def get_ax():
    """Create a plot and return its Axes."""
    fig, ax = plt.subplots(1, 1)  # 创建一个包含一个子图的图形对象
    ax.plot([0, 200], [0, 200])  # 在坐标系中绘制一条线
    ax.set_aspect(1.0)  # 设置坐标轴的纵横比为1:1
    ax.figure.canvas.draw()  # 绘制图形
    return ax  # 返回创建的 Axes 对象


def noop(*args, **kwargs):
    pass  # 定义一个空函数，什么也不做


def mock_event(ax, button=1, xdata=0, ydata=0, key=None, step=1):
    r"""
    Create a mock event that can stand in for `.Event` and its subclasses.

    This event is intended to be used in tests where it can be passed into
    event handling functions.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The Axes the event will be in.
    xdata : float
        x coord of mouse in data coords.
    ydata : float
        y coord of mouse in data coords.
    button : None or `MouseButton` or {'up', 'down'}
        The mouse button pressed in this event (see also `.MouseEvent`).
    key : None or str
        The key pressed when the mouse event triggered (see also `.KeyEvent`).
    step : int
        Number of scroll steps (positive for 'up', negative for 'down').

    Returns
    -------
    event
        A `.Event`\-like Mock instance.
    """
    event = mock.Mock()  # 创建一个 Mock 对象作为事件的模拟
    event.button = button  # 设置事件的鼠标按钮属性
    event.x, event.y = ax.transData.transform([(xdata, ydata),  # 将数据坐标转换为画布坐标
                                               (xdata, ydata)])[0]
    event.xdata, event.ydata = xdata, ydata  # 设置事件的数据坐标
    event.inaxes = ax  # 设置事件发生的坐标系
    event.canvas = ax.figure.canvas  # 设置事件所属的画布
    event.key = key  # 设置事件的按键
    event.step = step  # 设置事件的滚动步数
    event.guiEvent = None  # 设置事件的 GUI 事件为 None
    event.name = 'Custom'  # 设置事件的名称为 Custom
    return event  # 返回创建的事件对象


def do_event(tool, etype, button=1, xdata=0, ydata=0, key=None, step=1):
    """
    Trigger an event on the given tool.

    Parameters
    ----------
    tool : matplotlib.widgets.AxesWidget
        The widget or tool to trigger the event on.
    etype : str
        The event to trigger.
    xdata : float
        x coord of mouse in data coords.
    ydata : float
        y coord of mouse in data coords.
    button : None or `MouseButton` or {'up', 'down'}
        The mouse button pressed in this event (see also `.MouseEvent`).
    key : None or str
        The key pressed when the mouse event triggered (see also `.KeyEvent`).
    step : int
        Number of scroll steps (positive for 'up', negative for 'down').
    """
    event = mock_event(tool.ax, button, xdata, ydata, key, step)  # 创建一个模拟事件
    func = getattr(tool, etype)  # 获取工具对象上对应事件类型的方法
    func(event)  # 调用对应事件类型的方法并传入模拟的事件对象


def click_and_drag(tool, start, end, key=None):
    """
    Helper to simulate a mouse drag operation.

    Parameters
    ----------
    tool : `~matplotlib.widgets.Widget`
        The widget or tool to perform the drag operation on.
    start : [float, float]
        Starting point in data coordinates.
    end : [float, float]
        End point in data coordinates.
    key : None or str
         An optional key that is pressed during the whole operation
         (see also `.KeyEvent`).
    """
    # 如果给定了按键信息，则执行按键事件处理
    if key is not None:
        # 调用事件处理函数，模拟按键按下事件
        do_event(tool, 'on_key_press', xdata=start[0], ydata=start[1],
                 button=1, key=key)
    
    # 调用事件处理函数，模拟鼠标按下事件
    do_event(tool, 'press', xdata=start[0], ydata=start[1], button=1)
    
    # 调用事件处理函数，模拟鼠标移动事件
    do_event(tool, 'onmove', xdata=end[0], ydata=end[1], button=1)
    
    # 调用事件处理函数，模拟鼠标释放事件
    do_event(tool, 'release', xdata=end[0], ydata=end[1], button=1)
    
    # 如果给定了按键信息，则执行按键释放事件处理
    if key is not None:
        # 调用事件处理函数，模拟按键释放事件
        do_event(tool, 'on_key_release', xdata=end[0], ydata=end[1],
                 button=1, key=key)
```