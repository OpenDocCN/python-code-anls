# `D:\src\scipysrc\matplotlib\galleries\examples\user_interfaces\embedding_in_wx4_sgskip.py`

```py
"""
==================
Embedding in wx #4
==================

An example of how to use wxagg in a wx application with a custom toolbar.
"""

import wx  # 导入 wxPython 库

import numpy as np  # 导入 NumPy 库

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas  # 导入 Matplotlib 的 wxAgg 图形画布
from matplotlib.backends.backend_wxagg import \
    NavigationToolbar2WxAgg as NavigationToolbar  # 导入 Matplotlib 的 wxAgg 导航工具栏
from matplotlib.figure import Figure  # 导入 Matplotlib 的 Figure 类


class MyNavigationToolbar(NavigationToolbar):
    """Extend the default wx toolbar with your own event handlers."""

    def __init__(self, canvas):
        super().__init__(canvas)
        # We use a stock wx bitmap, but you could also use your own image file.
        bmp = wx.ArtProvider.GetBitmap(wx.ART_CROSS_MARK, wx.ART_TOOLBAR)
        tool = self.AddTool(wx.ID_ANY, 'Click me', bmp,
                            'Activate custom control')
        self.Bind(wx.EVT_TOOL, self._on_custom, id=tool.GetId())

    def _on_custom(self, event):
        # add some text to the Axes in a random location in axes coords with a
        # random color
        ax = self.canvas.figure.axes[0]
        x, y = np.random.rand(2)  # generate a random location
        rgb = np.random.rand(3)  # generate a random color
        ax.text(x, y, 'You clicked me', transform=ax.transAxes, color=rgb)  # 在图表中随机位置添加文本
        self.canvas.draw()  # 重新绘制画布
        event.Skip()


class CanvasFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, -1, 'CanvasFrame', size=(550, 350))  # 创建窗口对象

        self.figure = Figure(figsize=(5, 4), dpi=100)  # 创建 Matplotlib 图形对象
        self.axes = self.figure.add_subplot()  # 添加子图

        t = np.arange(0.0, 3.0, 0.01)
        s = np.sin(2 * np.pi * t)
        self.axes.plot(t, s)  # 在子图中绘制正弦曲线

        self.canvas = FigureCanvas(self, -1, self.figure)  # 创建 Matplotlib 图形画布对象

        self.sizer = wx.BoxSizer(wx.VERTICAL)  # 创建垂直布局管理器
        self.sizer.Add(self.canvas, 1, wx.TOP | wx.LEFT | wx.EXPAND)  # 将画布添加到布局管理器中

        self.toolbar = MyNavigationToolbar(self.canvas)  # 创建自定义的导航工具栏
        self.toolbar.Realize()  # 显示工具栏

        # By adding toolbar in sizer, we are able to put it at the bottom
        # of the frame - so appearance is closer to GTK version.
        self.sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)  # 将工具栏添加到布局管理器中

        # update the axes menu on the toolbar
        self.toolbar.update()  # 更新工具栏上的坐标轴菜单

        self.SetSizer(self.sizer)  # 设置窗口的布局管理器
        self.Fit()  # 调整窗口尺寸以适应布局


class App(wx.App):
    def OnInit(self):
        """Create the main window and insert the custom frame."""
        frame = CanvasFrame()  # 创建 CanvasFrame 对象
        frame.Show(True)  # 显示窗口
        return True


if __name__ == "__main__":
    app = App()  # 创建应用程序对象
    app.MainLoop()  # 运行主事件循环
```