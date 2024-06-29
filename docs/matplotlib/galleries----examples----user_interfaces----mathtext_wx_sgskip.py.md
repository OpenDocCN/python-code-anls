# `D:\src\scipysrc\matplotlib\galleries\examples\user_interfaces\mathtext_wx_sgskip.py`

```py
"""
======================
Display mathtext in WX
======================

Demonstrates how to convert (math)text to a wx.Bitmap for display in various
controls on wxPython.
"""

from io import BytesIO  # 导入 BytesIO 类，用于操作二进制数据流

import wx  # 导入 wxPython 库

import numpy as np  # 导入 NumPy 库，用于数值计算

from matplotlib.backends.backend_wx import NavigationToolbar2Wx  # 导入 wxPython 的 matplotlib 导航工具条
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas  # 导入 wxPython 的 matplotlib 图像画布
from matplotlib.figure import Figure  # 导入 matplotlib 的图像对象

IS_WIN = 'wxMSW' in wx.PlatformInfo  # 判断是否在 Windows 平台下运行


def mathtext_to_wxbitmap(s):
    # 将数学文本转换为 wx.Bitmap 对象的函数

    # 创建一个透明背景的图像对象
    fig = Figure(facecolor="none")

    # 获取系统文本颜色并转换为 numpy 数组
    text_color = np.array(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT)) / 255

    # 在图像对象上绘制数学文本
    fig.text(0, 0, s, fontsize=10, color=text_color)

    # 将图像保存为 PNG 格式到内存缓冲区
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0)
    s = buf.getvalue()

    # 从 PNG 数据创建 wx.Bitmap 对象并返回
    return wx.Bitmap.NewFromPNGData(s, len(s))


functions = [
    (r'$\sin(2 \pi x)$', lambda x: np.sin(2*np.pi*x)),  # 正弦函数示例
    (r'$\frac{4}{3}\pi x^3$', lambda x: (4/3)*np.pi*x**3),  # 立方函数示例
    (r'$\cos(2 \pi x)$', lambda x: np.cos(2*np.pi*x)),  # 余弦函数示例
    (r'$\log(x)$', lambda x: np.log(x))  # 对数函数示例
]


class CanvasFrame(wx.Frame):
    def __init__(self, parent, title):
        # 初始化窗口对象
        super().__init__(parent, -1, title, size=(550, 350))

        # 创建一个 matplotlib 图像对象和对应的坐标轴
        self.figure = Figure()
        self.axes = self.figure.add_subplot()

        # 创建一个 wxPython 的 matplotlib 图像画布
        self.canvas = FigureCanvas(self, -1, self.figure)

        # 初始化显示第一个函数图像
        self.change_plot(0)

        # 创建一个垂直方向的布局管理器
        self.sizer = wx.BoxSizer(wx.VERTICAL)

        # 添加按钮工具栏
        self.add_buttonbar()

        # 将图像画布添加到布局管理器中，并指定布局属性
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)

        # 添加工具条到窗口中（如果需要工具条功能）
        self.add_toolbar()  # 如果不需要工具条功能，可以注释掉这一行

        # 创建菜单栏对象
        menuBar = wx.MenuBar()

        # 文件菜单
        menu = wx.Menu()
        m_exit = menu.Append(wx.ID_EXIT, "E&xit\tAlt-X", "Exit this simple sample")
        menuBar.Append(menu, "&File")
        self.Bind(wx.EVT_MENU, self.OnClose, m_exit)

        if IS_WIN:
            # 函数菜单（仅在 Windows 平台下）
            menu = wx.Menu()
            for i, (mt, func) in enumerate(functions):
                # 将数学文本转换为 wx.Bitmap 并添加到菜单项中
                bm = mathtext_to_wxbitmap(mt)
                item = wx.MenuItem(menu, 1000 + i, " ")
                item.SetBitmap(bm)
                menu.Append(item)
                self.Bind(wx.EVT_MENU, self.OnChangePlot, item)
            menuBar.Append(menu, "&Functions")

        # 将菜单栏设置到窗口中
        self.SetMenuBar(menuBar)

        # 设置窗口布局管理器并调整窗口大小以适应内容
        self.SetSizer(self.sizer)
        self.Fit()
    def add_buttonbar(self):
        # 创建一个面板用于放置按钮条
        self.button_bar = wx.Panel(self)
        # 创建一个水平方向的布局管理器
        self.button_bar_sizer = wx.BoxSizer(wx.HORIZONTAL)
        # 将按钮条面板添加到主布局管理器中，设置可扩展的左、上边距
        self.sizer.Add(self.button_bar, 0, wx.LEFT | wx.TOP | wx.GROW)

        # 遍历函数列表中的每个函数元组，并创建对应的按钮
        for i, (mt, func) in enumerate(functions):
            # 将数学文本转换为 wxPython 可用的位图
            bm = mathtext_to_wxbitmap(mt)
            # 创建一个位图按钮，将其添加到按钮条面板上
            button = wx.BitmapButton(self.button_bar, 1000 + i, bm)
            self.button_bar_sizer.Add(button, 1, wx.GROW)
            # 绑定按钮点击事件到 self.OnChangePlot 方法
            self.Bind(wx.EVT_BUTTON, self.OnChangePlot, button)

        # 将按钮条布局管理器应用到按钮条面板上
        self.button_bar.SetSizer(self.button_bar_sizer)

    def add_toolbar(self):
        """从 embedding_wx2.py 中直接复制而来"""
        # 创建一个导航工具栏并实现它
        self.toolbar = NavigationToolbar2Wx(self.canvas)
        self.toolbar.Realize()
        # 通过将工具栏添加到主布局管理器中，使其能够位于窗口底部，使外观更接近 GTK 版本
        self.sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        # 更新工具栏上的坐标轴菜单
        self.toolbar.update()

    def OnChangePlot(self, event):
        # 根据事件的 ID 更新绘图
        self.change_plot(event.GetId() - 1000)

    def change_plot(self, plot_number):
        # 创建时间序列数据 t
        t = np.arange(1.0, 3.0, 0.01)
        # 使用选定的函数计算数据序列 s
        s = functions[plot_number][1](t)
        # 清空当前绘图区域
        self.axes.clear()
        # 绘制新的数据曲线
        self.axes.plot(t, s)
        # 重新绘制画布
        self.canvas.draw()

    def OnClose(self, event):
        # 关闭窗口
        self.Destroy()
# 定义一个自定义的 wxPython 应用程序类 MyApp，继承自 wx.App
class MyApp(wx.App):
    # 重写 OnInit 方法，初始化应用程序
    def OnInit(self):
        # 创建一个 CanvasFrame 对象作为主窗口，第一个参数为父窗口（这里为None表示顶级窗口），第二个参数为窗口标题
        frame = CanvasFrame(None, "wxPython mathtext demo app")
        # 设置主窗口
        self.SetTopWindow(frame)
        # 显示主窗口
        frame.Show(True)
        # 返回 True，表示初始化成功
        return True


# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 创建 MyApp 类的实例
    app = MyApp()
    # 进入应用程序的主事件循环
    app.MainLoop()
```