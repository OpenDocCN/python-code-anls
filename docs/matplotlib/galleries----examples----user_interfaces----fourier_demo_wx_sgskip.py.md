# `D:\src\scipysrc\matplotlib\galleries\examples\user_interfaces\fourier_demo_wx_sgskip.py`

```py
"""
===============
Fourier Demo WX
===============

"""

# 导入 wxPython 库
import wx

# 导入 numpy 库并用 np 别名
import numpy as np

# 导入 Matplotlib 的 WXAgg 后端和 FigureCanvas
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
# 导入 Matplotlib 的 Figure 类
from matplotlib.figure import Figure

# 定义一个旋钮类
class Knob:
    """
    Knob - simple class with a "setKnob" method.
    A Knob instance is attached to a Param instance, e.g., param.attach(knob)
    Base class is for documentation purposes.
    """

    # 设置旋钮的方法，但在这个基类中是空的
    def setKnob(self, value):
        pass


# 定义一个参数类
class Param:
    """
    The idea of the "Param" class is that some parameter in the GUI may have
    several knobs that both control it and reflect the parameter's state, e.g.
    a slider, text, and dragging can all change the value of the frequency in
    the waveform of this example.
    The class allows a cleaner way to update/"feedback" to the other knobs when
    one is being changed.  Also, this class handles min/max constraints for all
    the knobs.
    Idea - knob list - in "set" method, knob object is passed as well
      - the other knobs in the knob list have a "set" method which gets
        called for the others.
    """

    # 初始化方法，设置参数的初始值和范围
    def __init__(self, initialValue=None, minimum=0., maximum=1.):
        self.minimum = minimum
        self.maximum = maximum
        # 如果初始值不在范围内，则引发 ValueError 异常
        if initialValue != self.constrain(initialValue):
            raise ValueError('illegal initial value')
        self.value = initialValue
        self.knobs = []

    # 将旋钮对象附加到参数实例的方法
    def attach(self, knob):
        self.knobs += [knob]

    # 设置参数值的方法，同时更新所有附加的旋钮
    def set(self, value, knob=None):
        self.value = value
        self.value = self.constrain(value)
        for feedbackKnob in self.knobs:
            if feedbackKnob != knob:
                feedbackKnob.setKnob(self.value)
        return self.value

    # 约束参数值在最小和最大值之间的方法
    def constrain(self, value):
        if value <= self.minimum:
            value = self.minimum
        if value >= self.maximum:
            value = self.maximum
        return value


# 定义一个滑块组类，继承自 Knob 类
class SliderGroup(Knob):
    def __init__(self, parent, label, param):
        # 创建静态文本和文本框，以及滑块，并设置其范围
        self.sliderLabel = wx.StaticText(parent, label=label)
        self.sliderText = wx.TextCtrl(parent, -1, style=wx.TE_PROCESS_ENTER)
        self.slider = wx.Slider(parent, -1)
        self.slider.SetRange(0, int(param.maximum * 1000))
        self.setKnob(param.value)

        # 创建水平布局 sizer，并添加控件
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.sliderLabel, 0,
                  wx.EXPAND | wx.ALL,
                  border=2)
        sizer.Add(self.sliderText, 0,
                  wx.EXPAND | wx.ALL,
                  border=2)
        sizer.Add(self.slider, 1, wx.EXPAND)
        self.sizer = sizer

        # 绑定滑块和文本框的事件处理方法
        self.slider.Bind(wx.EVT_SLIDER, self.sliderHandler)
        self.sliderText.Bind(wx.EVT_TEXT_ENTER, self.sliderTextHandler)

        # 将参数实例与滑块组关联
        self.param = param
        self.param.attach(self)

    # 滑块事件处理方法，更新参数值
    def sliderHandler(self, event):
        value = event.GetInt() / 1000.
        self.param.set(value)
    # 处理滑动条文本框的事件，更新参数值
    def sliderTextHandler(self, event):
        # 获取文本框中的值并转换为浮点数
        value = float(self.sliderText.GetValue())
        # 调用对象的方法设置参数值
        self.param.set(value)
    
    # 设置旋钮的数值和滑动条位置
    def setKnob(self, value):
        # 将数值格式化为字符串，并设置为滑动条文本框的值
        self.sliderText.SetValue(f'{value:g}')
        # 将数值乘以1000后转换为整数，设置为滑动条的位置
        self.slider.SetValue(int(value * 1000))
class FourierDemoFrame(wx.Frame):
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
        # 在主窗口上创建一个面板
        panel = wx.Panel(self)

        # 创建 GUI 元素
        self.createCanvas(panel)  # 创建绘图画布
        self.createSliders(panel)  # 创建滑动条控件

        # 将元素放置在一个垂直布局的 sizer 中
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)  # 将画布添加到 sizer 中并扩展
        sizer.Add(self.frequencySliderGroup.sizer, 0, wx.EXPAND | wx.ALL, border=5)  # 添加频率滑动条组
        sizer.Add(self.amplitudeSliderGroup.sizer, 0, wx.EXPAND | wx.ALL, border=5)  # 添加振幅滑动条组
        panel.SetSizer(sizer)  # 将 sizer 应用到面板上

    def createCanvas(self, parent):
        # 初始化绘图相关变量
        self.lines = []
        self.figure = Figure()  # 创建一个新的图形对象
        self.canvas = FigureCanvas(parent, -1, self.figure)  # 创建绘图画布
        # 连接鼠标事件处理函数
        self.canvas.callbacks.connect('button_press_event', self.mouseDown)
        self.canvas.callbacks.connect('motion_notify_event', self.mouseMotion)
        self.canvas.callbacks.connect('button_release_event', self.mouseUp)
        self.state = ''  # 初始化鼠标状态为空
        self.mouseInfo = (None, None, None, None)  # 初始化鼠标信息元组
        self.f0 = Param(2., minimum=0., maximum=6.)  # 创建频率参数对象
        self.A = Param(1., minimum=0.01, maximum=2.)  # 创建振幅参数对象
        self.createPlots()  # 创建绘图

        # 不确定是否喜欢将两个参数附加到同一个控制旋钮上，
        # 但我们这里确实有这样的情况... 它可以工作，但感觉有点笨拙 -
        # 尽管也许并不太糟糕，因为旋钮在拖动时同时改变两个参数
        self.f0.attach(self)  # 将频率参数附加到当前对象
        self.A.attach(self)  # 将振幅参数附加到当前对象

    def createSliders(self, panel):
        # 创建频率滑动条组件
        self.frequencySliderGroup = SliderGroup(
            panel,
            label='Frequency f0:',
            param=self.f0)
        # 创建振幅滑动条组件
        self.amplitudeSliderGroup = SliderGroup(panel, label=' Amplitude a:',
                                                param=self.A)

    def mouseDown(self, event):
        # 根据鼠标事件确定当前操作状态
        if self.lines[0].contains(event)[0]:
            self.state = 'frequency'  # 如果点击在第一个线条上，则设置状态为频率调整
        elif self.lines[1].contains(event)[0]:
            self.state = 'time'  # 如果点击在第二个线条上，则设置状态为时间调整
        else:
            self.state = ''  # 否则状态为空
        self.mouseInfo = (event.xdata, event.ydata,
                          max(self.f0.value, .1),
                          self.A.value)  # 记录鼠标信息

    def mouseMotion(self, event):
        # 处理鼠标移动事件
        if self.state == '':  # 如果状态为空则返回
            return
        x, y = event.xdata, event.ydata
        if x is None:  # 如果鼠标在绘图区外，则返回
            return
        x0, y0, f0Init, AInit = self.mouseInfo
        # 根据鼠标移动的距离调整振幅参数
        self.A.set(AInit + (AInit * (y - y0) / y0), self)
        if self.state == 'frequency':  # 如果状态为频率调整
            # 根据鼠标移动的距离调整频率参数
            self.f0.set(f0Init + (f0Init * (x - x0) / x0))
        elif self.state == 'time':  # 如果状态为时间调整
            if (x - x0) / x0 != -1.:
                # 根据鼠标移动的距离调整频率参数（逆）
                self.f0.set(1. / (1. / f0Init + (1. / f0Init * (x - x0) / x0)))

    def mouseUp(self, event):
        self.state = ''  # 鼠标释放，状态重置为空
    # 创建图形的子图和波形以及标签
    # 当后续拖动波形或滑块时，只会更新波形数据（不在此处更新，在下面的setKnob方法中）
    self.subplot1, self.subplot2 = self.figure.subplots(2)
    # 计算频率和振幅为 self.f0.value 和 self.A.value 时的两组波形数据
    x1, y1, x2, y2 = self.compute(self.f0.value, self.A.value)
    # 设置波形的颜色
    color = (1., 0., 0.)
    # 在 subplot1 中绘制第一组波形，并将线条对象加入到self.lines中
    self.lines += self.subplot1.plot(x1, y1, color=color, linewidth=2)
    # 在 subplot2 中绘制第二组波形，并将线条对象加入到self.lines中
    self.lines += self.subplot2.plot(x2, y2, color=color, linewidth=2)
    # 设置 subplot1 的标题和坐标轴标签
    self.subplot1.set_title(
        "Click and drag waveforms to change frequency and amplitude",
        fontsize=12)
    self.subplot1.set_ylabel("Frequency Domain Waveform X(f)", fontsize=8)
    self.subplot1.set_xlabel("frequency f", fontsize=8)
    # 设置 subplot2 的坐标轴标签
    self.subplot2.set_ylabel("Time Domain Waveform x(t)", fontsize=8)
    self.subplot2.set_xlabel("time t", fontsize=8)
    # 设置 subplot1 和 subplot2 的坐标轴范围
    self.subplot1.set_xlim([-6, 6])
    self.subplot1.set_ylim([0, 1])
    self.subplot2.set_xlim([-2, 2])
    self.subplot2.set_ylim([-2, 2])
    # 在 subplot1 中添加文本标注
    self.subplot1.text(0.05, .95,
                       r'$X(f) = \mathcal{F}\{x(t)\}$',
                       verticalalignment='top',
                       transform=self.subplot1.transAxes)
    # 在 subplot2 中添加文本标注
    self.subplot2.text(0.05, .95,
                       r'$x(t) = a \cdot \cos(2\pi f_0 t) e^{-\pi t^2}$',
                       verticalalignment='top',
                       transform=self.subplot2.transAxes)
# 定义一个名为 App 的类，继承自 wx.App，用于管理整个应用程序的生命周期
class App(wx.App):
    # 覆盖 wx.App 中的 OnInit 方法，在应用程序初始化时被调用
    def OnInit(self):
        # 创建一个名为 frame1 的 FourierDemoFrame 对象作为主窗口，无父窗口，标题为 "Fourier Demo"，大小为 640x480 像素
        self.frame1 = FourierDemoFrame(parent=None, title="Fourier Demo",
                                       size=(640, 480))
        # 显示 frame1 窗口
        self.frame1.Show()
        # 返回 True 表示初始化成功
        return True


# 如果当前脚本作为主程序运行，则执行以下代码
if __name__ == "__main__":
    # 创建一个 App 类的实例对象，即创建应用程序对象
    app = App()
    # 进入应用程序的主事件循环，处理事件并保持应用程序运行
    app.MainLoop()
```