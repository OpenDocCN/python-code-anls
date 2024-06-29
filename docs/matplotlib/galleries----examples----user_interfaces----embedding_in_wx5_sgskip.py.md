# `D:\src\scipysrc\matplotlib\galleries\examples\user_interfaces\embedding_in_wx5_sgskip.py`

```py
"""
==================
Embedding in wx #5
==================

"""

# 导入 wxPython 库
import wx
import wx.lib.agw.aui as aui  # 导入 wx.aui 库
import wx.lib.mixins.inspection as wit  # 导入 wx.inspect 库

# 导入 matplotlib 的相关组件
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
from matplotlib.figure import Figure


# 定义绘图面板类
class Plot(wx.Panel):
    def __init__(self, parent, id=-1, dpi=None, **kwargs):
        super().__init__(parent, id=id, **kwargs)
        # 创建一个 matplotlib 图形对象
        self.figure = Figure(dpi=dpi, figsize=(2, 2))
        # 创建一个 wxWidgets 的画布对象
        self.canvas = FigureCanvas(self, -1, self.figure)
        # 创建一个 matplotlib 导航工具栏对象
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()

        # 使用 wx 垂直布局管理器进行布局
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        self.SetSizer(sizer)


# 定义绘图笔记本类
class PlotNotebook(wx.Panel):
    def __init__(self, parent, id=-1):
        super().__init__(parent, id=id)
        # 创建一个 wx.aui 笔记本控件对象
        self.nb = aui.AuiNotebook(self)
        sizer = wx.BoxSizer()
        sizer.Add(self.nb, 1, wx.EXPAND)
        self.SetSizer(sizer)

    # 添加新页到笔记本
    def add(self, name="plot"):
        # 创建一个绘图面板对象
        page = Plot(self.nb)
        # 将新页添加到笔记本中
        self.nb.AddPage(page, name)
        # 返回页的 matplotlib 图形对象
        return page.figure


# 定义演示函数
def demo():
    # 创建一个可检查的 wx 应用对象
    app = wit.InspectableApp()
    # 创建一个顶层框架
    frame = wx.Frame(None, -1, 'Plotter')
    # 创建一个绘图笔记本对象
    plotter = PlotNotebook(frame)
    # 在第一个页上添加子图并绘制
    axes1 = plotter.add('figure 1').add_subplot()
    axes1.plot([1, 2, 3], [2, 1, 4])
    # 在第二个页上添加子图并绘制
    axes2 = plotter.add('figure 2').add_subplot()
    axes2.plot([1, 2, 3, 4, 5], [2, 1, 4, 2, 3])
    # 显示框架
    frame.Show()
    # 运行应用主循环
    app.MainLoop()


if __name__ == "__main__":
    # 执行演示函数
    demo()
```