# `D:\src\scipysrc\matplotlib\galleries\examples\user_interfaces\embedding_in_wx2_sgskip.py`

```
"""
==================
Embedding in wx #2
==================

An example of how to use wxagg in an application with the new
toolbar - comment out the add_toolbar line for no toolbar.
"""

# 导入 wxPython 库
import wx
import wx.lib.mixins.inspection as WIT  # 导入用于调试的 mixin 库

# 导入必要的数学和绘图库
import numpy as np

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas  # 导入 Matplotlib 的 wxAgg 图形绘制库
from matplotlib.backends.backend_wxagg import \
    NavigationToolbar2WxAgg as NavigationToolbar  # 导入 Matplotlib 的 wxAgg 导航工具条
from matplotlib.figure import Figure  # 导入 Matplotlib 的图形对象


class CanvasFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, -1, 'CanvasFrame', size=(550, 350))  # 创建一个 wx.Frame 窗口

        self.figure = Figure()  # 创建一个 Matplotlib 的 Figure 对象
        self.axes = self.figure.add_subplot()  # 在 Figure 对象中添加一个子图
        t = np.arange(0.0, 3.0, 0.01)  # 创建一个 NumPy 数组
        s = np.sin(2 * np.pi * t)  # 计算正弦函数

        self.axes.plot(t, s)  # 在子图中绘制正弦曲线
        self.canvas = FigureCanvas(self, -1, self.figure)  # 创建一个 Matplotlib 的画布对象

        self.sizer = wx.BoxSizer(wx.VERTICAL)  # 创建一个垂直尺寸器
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.EXPAND)  # 将画布添加到尺寸器中
        self.SetSizer(self.sizer)  # 设置窗口的尺寸器
        self.Fit()  # 调整窗口大小以适应其内容

        self.add_toolbar()  # 添加工具栏到窗口，注释此行以移除工具栏

    def add_toolbar(self):
        self.toolbar = NavigationToolbar(self.canvas)  # 创建一个 Matplotlib 导航工具栏对象
        self.toolbar.Realize()  # 显示工具栏
        # 将工具栏添加到尺寸器中，使其显示在窗口底部，使外观更接近 GTK 版本
        self.sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        # 更新工具栏上的坐标轴菜单
        self.toolbar.update()


# 或者你可以使用以下方式:
# class App(wx.App):
class App(WIT.InspectableApp):
    def OnInit(self):
        """创建主窗口并插入自定义框架。"""
        self.Init()  # 初始化调试工具
        frame = CanvasFrame()  # 创建 CanvasFrame 对象
        frame.Show(True)  # 显示窗口

        return True


if __name__ == "__main__":
    app = App()  # 创建应用程序对象
    app.MainLoop()  # 运行主事件循环
```