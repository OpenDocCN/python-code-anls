# `D:\src\scipysrc\matplotlib\galleries\examples\user_interfaces\wxcursor_demo_sgskip.py`

```py
"""
=====================
Adding a cursor in WX
=====================

Example to draw a cursor and report the data coords in wx.
"""

import wx  # 导入 wxPython 库

import numpy as np  # 导入 NumPy 库

from matplotlib.backends.backend_wx import NavigationToolbar2Wx  # 导入 Matplotlib WX 导航工具栏
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas  # 导入 Matplotlib WX 聚合画布
from matplotlib.figure import Figure  # 导入 Matplotlib 图形对象


class CanvasFrame(wx.Frame):
    def __init__(self, ):
        super().__init__(None, -1, 'CanvasFrame', size=(550, 350))  # 创建一个 wxPython 框架

        self.figure = Figure()  # 创建一个 Matplotlib 图形对象
        self.axes = self.figure.add_subplot()  # 在图形对象中添加一个子图
        t = np.arange(0.0, 3.0, 0.01)  # 创建一个 NumPy 数组
        s = np.sin(2*np.pi*t)  # 计算正弦函数值

        self.axes.plot(t, s)  # 绘制图形
        self.axes.set_xlabel('t')  # 设置 x 轴标签
        self.axes.set_ylabel('sin(t)')  # 设置 y 轴标签
        self.figure_canvas = FigureCanvas(self, -1, self.figure)  # 创建 Matplotlib WX 聚合画布对象

        # 注意：event 是一个 MplEvent
        self.figure_canvas.mpl_connect(
            'motion_notify_event', self.UpdateStatusBar)  # 绑定鼠标移动事件处理函数 UpdateStatusBar
        self.figure_canvas.Bind(wx.EVT_ENTER_WINDOW, self.ChangeCursor)  # 绑定鼠标进入窗口事件处理函数 ChangeCursor

        self.sizer = wx.BoxSizer(wx.VERTICAL)  # 创建一个垂直 BoxSizer 布局管理器
        self.sizer.Add(self.figure_canvas, 1, wx.LEFT | wx.TOP | wx.GROW)  # 将画布对象添加到布局管理器中
        self.SetSizer(self.sizer)  # 设置框架的布局管理器
        self.Fit()  # 调整框架尺寸以适应其内容

        self.statusBar = wx.StatusBar(self, -1)  # 创建一个状态栏对象
        self.SetStatusBar(self.statusBar)  # 设置框架的状态栏

        self.toolbar = NavigationToolbar2Wx(self.figure_canvas)  # 创建 Matplotlib 导航工具栏对象
        self.sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)  # 将工具栏添加到布局管理器中
        self.toolbar.Show()  # 显示工具栏

    def ChangeCursor(self, event):
        self.figure_canvas.SetCursor(wx.Cursor(wx.CURSOR_BULLSEYE))  # 改变鼠标光标形状为十字光标

    def UpdateStatusBar(self, event):
        if event.inaxes:  # 如果鼠标事件发生在坐标系内
            self.statusBar.SetStatusText(f"x={event.xdata}  y={event.ydata}")  # 更新状态栏显示鼠标位置信息


class App(wx.App):
    def OnInit(self):
        """Create the main window and insert the custom frame."""
        frame = CanvasFrame()  # 创建 CanvasFrame 对象
        self.SetTopWindow(frame)  # 将框架设置为顶级窗口
        frame.Show(True)  # 显示框架
        return True


if __name__ == '__main__':
    app = App()  # 创建应用程序对象
    app.MainLoop()  # 运行应用程序主循环
```