# `D:\src\scipysrc\matplotlib\galleries\examples\user_interfaces\embedding_in_wx3_sgskip.py`

```py
"""
==================
Embedding in wx #3
==================

Copyright (C) 2003-2004 Andrew Straw, Jeremy O'Donoghue and others

License: This work is licensed under the PSF. A copy should be included
with this source code, and is also available at
https://docs.python.org/3/license.html

This is yet another example of using matplotlib with wx.  Hopefully
this is pretty full-featured:

- both matplotlib toolbar and WX buttons manipulate plot
- full wxApp framework, including widget interaction
- XRC (XML wxWidgets resource) file to create GUI (made with XRCed)

This was derived from embedding_in_wx and dynamic_image_wxagg.

Thanks to matplotlib and wx teams for creating such great software!
"""

# 导入 wxPython 库
import wx
import wx.xrc as xrc

# 导入 numpy 库
import numpy as np

# 导入 matplotlib 相关模块
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import \
    NavigationToolbar2WxAgg as NavigationToolbar
import matplotlib.cbook as cbook
import matplotlib.cm as cm
from matplotlib.figure import Figure

# 定义浮点数精度容差常量
ERR_TOL = 1e-5  # floating point slop for peak-detection

# 定义绘图面板类
class PlotPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent, -1)

        # 创建一个 Figure 对象，设置其大小和 DPI
        self.fig = Figure((5, 4), 75)
        # 创建 Figure 对象的画布
        self.canvas = FigureCanvas(self, -1, self.fig)
        # 创建 matplotlib 工具栏对象
        self.toolbar = NavigationToolbar(self.canvas)  # matplotlib toolbar
        self.toolbar.Realize()

        # 将画布和工具栏添加到垂直方向的 sizer 中
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        sizer.Add(self.toolbar, 0, wx.GROW)
        self.SetSizer(sizer)
        self.Fit()

    # 初始化绘图数据
    def init_plot_data(self):
        # 在 Figure 对象上添加一个子图
        ax = self.fig.add_subplot()

        # 生成 x 和 y 的网格数据
        x = np.arange(120.0) * 2 * np.pi / 60.0
        y = np.arange(100.0) * 2 * np.pi / 50.0
        self.x, self.y = np.meshgrid(x, y)
        # 计算 z 值，这里使用了正弦和余弦函数
        z = np.sin(self.x) + np.cos(self.y)
        # 在子图上绘制 z 的颜色映射图
        self.im = ax.imshow(z, cmap=cm.RdBu, origin='lower')

        # 计算 z 的最大值，并找到其对应的坐标
        zmax = np.max(z) - ERR_TOL
        ymax_i, xmax_i = np.nonzero(z >= zmax)
        # 根据图像的原点属性调整坐标
        if self.im.origin == 'upper':
            ymax_i = z.shape[0] - ymax_i
        # 在子图上绘制最大值点
        self.lines = ax.plot(xmax_i, ymax_i, 'ko')

        # 更新工具栏
        self.toolbar.update()  # Not sure why this is needed - ADS

    def GetToolBar(self):
        # 如果在框架中使用未管理的工具栏，则需要重写此方法
        return self.toolbar

    def OnWhiz(self, event):
        # 更新数据，产生动画效果
        self.x += np.pi / 15
        self.y += np.pi / 20
        z = np.sin(self.x) + np.cos(self.y)
        self.im.set_array(z)

        # 计算新数据的最大值，并找到其对应的坐标
        zmax = np.max(z) - ERR_TOL
        ymax_i, xmax_i = np.nonzero(z >= zmax)
        # 根据图像的原点属性调整坐标
        if self.im.origin == 'upper':
            ymax_i = z.shape[0] - ymax_i
        # 更新最大值点的位置
        self.lines[0].set_data(xmax_i, ymax_i)

        # 重新绘制画布
        self.canvas.draw()

# 定义 wx 应用程序类
class MyApp(wx.App):
    # 初始化方法，在应用程序启动时被调用
    def OnInit(self):
        # 获取名为 'embedding_in_wx3.xrc' 的示例数据文件路径
        xrcfile = cbook.get_sample_data('embedding_in_wx3.xrc', asfileobj=False)
        # 打印加载的文件路径
        print('loading', xrcfile)

        # 使用 XmlResource 加载 XRC 文件
        self.res = xrc.XmlResource(xrcfile)

        # 创建主窗口和面板 ---------

        # 加载名为 "MainFrame" 的主窗口
        self.frame = self.res.LoadFrame(None, "MainFrame")
        # 获取名为 "MainPanel" 的面板
        self.panel = xrc.XRCCTRL(self.frame, "MainPanel")

        # matplotlib 面板 -------------

        # 创建 matplotlib 面板的容器（我喜欢在 XRCed 中为面板创建一个容器面板，这样我知道它在哪里。）
        plot_container = xrc.XRCCTRL(self.frame, "plot_container_panel")
        sizer = wx.BoxSizer(wx.VERTICAL)

        # 创建 matplotlib 面板对象
        self.plotpanel = PlotPanel(plot_container)
        # 初始化 matplotlib 面板的绘图数据
        self.plotpanel.init_plot_data()

        # 添加 matplotlib 面板到 sizer 中
        sizer.Add(self.plotpanel, 1, wx.EXPAND)
        plot_container.SetSizer(sizer)

        # "whiz" 按钮 ------------------

        # 获取名为 "whiz_button" 的按钮对象
        whiz_button = xrc.XRCCTRL(self.frame, "whiz_button")
        # 绑定 "whiz" 按钮的点击事件到 self.plotpanel.OnWhiz 方法
        whiz_button.Bind(wx.EVT_BUTTON, self.plotpanel.OnWhiz)

        # "bang" 按钮 ------------------

        # 获取名为 "bang_button" 的按钮对象
        bang_button = xrc.XRCCTRL(self.frame, "bang_button")
        # 绑定 "bang" 按钮的点击事件到 self.OnBang 方法
        bang_button.Bind(wx.EVT_BUTTON, self.OnBang)

        # 完成最后的设置 ------------------

        # 显示主窗口
        self.frame.Show()

        # 将主窗口设置为顶级窗口
        self.SetTopWindow(self.frame)

        # 返回 True 表示初始化成功
        return True

    # "bang" 按钮事件处理方法
    def OnBang(self, event):
        # 获取名为 "bang_count" 的控件对象
        bang_count = xrc.XRCCTRL(self.frame, "bang_count")
        # 获取当前 "bang_count" 控件的值
        bangs = bang_count.GetValue()
        # 将获取到的值转换为整数，并加一
        bangs = int(bangs) + 1
        # 将增加后的值重新设置给 "bang_count" 控件
        bang_count.SetValue(str(bangs)))
if __name__ == '__main__':
    # 检查当前模块是否作为主程序运行
    app = MyApp()
    # 创建一个名为app的MyApp实例
    app.MainLoop()
    # 调用MyApp实例的MainLoop方法，开始应用程序的主事件循环
```