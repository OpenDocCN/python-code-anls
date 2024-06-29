# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axes_grid1\mpl_axes.py`

```py
import matplotlib.axes as maxes  # 导入 matplotlib.axes 模块并重命名为 maxes
from matplotlib.artist import Artist  # 导入 matplotlib.artist 模块中的 Artist 类
from matplotlib.axis import XAxis, YAxis  # 导入 matplotlib.axis 模块中的 XAxis 和 YAxis 类


class SimpleChainedObjects:
    def __init__(self, objects):
        self._objects = objects  # 初始化对象列表 _objects

    def __getattr__(self, k):
        # 返回一个新的 SimpleChainedObjects 实例，其中包含从 self._objects 中获取属性 k 的结果列表
        _a = SimpleChainedObjects([getattr(a, k) for a in self._objects])
        return _a

    def __call__(self, *args, **kwargs):
        # 对 self._objects 中的每个对象调用相同名称的方法，传递给方法的参数是 *args 和 **kwargs
        for m in self._objects:
            m(*args, **kwargs)


class Axes(maxes.Axes):  # 继承自 maxes.Axes 类的 Axes 类

    class AxisDict(dict):
        def __init__(self, axes):
            self.axes = axes  # 初始化 axes 属性为传入的 axes
            super().__init__()

        def __getitem__(self, k):
            if isinstance(k, tuple):
                # 如果 k 是 tuple 类型，返回一个新的 SimpleChainedObjects 实例，包含从父类中获取 k1 键的结果列表
                r = SimpleChainedObjects(
                    [super(Axes.AxisDict, self).__getitem__(k1) for k1 in k])
                return r
            elif isinstance(k, slice):
                if k.start is None and k.stop is None and k.step is None:
                    # 如果是空切片，则返回一个包含所有值的 SimpleChainedObjects 实例
                    return SimpleChainedObjects(list(self.values()))
                else:
                    raise ValueError("Unsupported slice")
            else:
                return dict.__getitem__(self, k)

        def __call__(self, *v, **kwargs):
            # 调用 maxes.Axes.axis 方法，传递参数 *v 和 **kwargs
            return maxes.Axes.axis(self.axes, *v, **kwargs)

    @property
    def axis(self):
        # 返回 _axislines 属性
        return self._axislines

    def clear(self):
        # 调用父类的 clear 方法，清除绘图区域
        super().clear()
        # 初始化 _axislines 属性为一个 AxisDict 实例，并更新各个轴线的 SimpleAxisArtist 实例
        self._axislines = self.AxisDict(self)
        self._axislines.update(
            bottom=SimpleAxisArtist(self.xaxis, 1, self.spines["bottom"]),
            top=SimpleAxisArtist(self.xaxis, 2, self.spines["top"]),
            left=SimpleAxisArtist(self.yaxis, 1, self.spines["left"]),
            right=SimpleAxisArtist(self.yaxis, 2, self.spines["right"]))


class SimpleAxisArtist(Artist):  # 继承自 Artist 类的 SimpleAxisArtist 类
    def __init__(self, axis, axisnum, spine):
        self._axis = axis  # 初始化 _axis 属性为传入的 axis
        self._axisnum = axisnum  # 初始化 _axisnum 属性为传入的 axisnum
        self.line = spine  # 初始化 line 属性为传入的 spine

        if isinstance(axis, XAxis):
            # 如果 axis 是 XAxis 的实例，设置 _axis_direction 属性为 "bottom" 或 "top"
            self._axis_direction = ["bottom", "top"][axisnum-1]
        elif isinstance(axis, YAxis):
            # 如果 axis 是 YAxis 的实例，设置 _axis_direction 属性为 "left" 或 "right"
            self._axis_direction = ["left", "right"][axisnum-1]
        else:
            # 如果 axis 不是 XAxis 或 YAxis 的实例，抛出 ValueError 异常
            raise ValueError(
                f"axis must be instance of XAxis or YAxis, but got {axis}")
        super().__init__()

    @property
    def major_ticks(self):
        # 返回一个 SimpleChainedObjects 实例，其中包含所有主刻度线的 tick%dline 属性
        tickline = "tick%dline" % self._axisnum
        return SimpleChainedObjects([getattr(tick, tickline)
                                     for tick in self._axis.get_major_ticks()])

    @property
    def major_ticklabels(self):
        # 返回一个 SimpleChainedObjects 实例，其中包含所有主刻度线的 label%d 属性
        label = "label%d" % self._axisnum
        return SimpleChainedObjects([getattr(tick, label)
                                     for tick in self._axis.get_major_ticks()])

    @property
    def label(self):
        # 返回 _axis 的 label 属性
        return self._axis.label
    # 设置对象可见性，控制所有相关元素的可见性状态
    def set_visible(self, b):
        # 调用 toggle 方法，根据参数 b 设置所有元素的可见性
        self.toggle(all=b)
        # 设置对象的线条可见性
        self.line.set_visible(b)
        # 设置轴的可见性为 True
        self._axis.set_visible(True)
        # 调用父类方法设置对象可见性
        super().set_visible(b)

    # 设置对象的标签文本
    def set_label(self, txt):
        # 调用轴对象的方法设置标签文本
        self._axis.set_label_text(txt)

    # 控制对象的各个部分的显示与隐藏状态
    def toggle(self, all=None, ticks=None, ticklabels=None, label=None):

        # 根据 all 参数设置 ticks, ticklabels, label 的默认状态
        if all:
            _ticks, _ticklabels, _label = True, True, True
        elif all is not None:
            _ticks, _ticklabels, _label = False, False, False
        else:
            _ticks, _ticklabels, _label = None, None, None

        # 根据 ticks, ticklabels, label 参数更新各自的状态
        if ticks is not None:
            _ticks = ticks
        if ticklabels is not None:
            _ticklabels = ticklabels
        if label is not None:
            _label = label

        # 根据 _ticks 更新轴的 ticks 的可见性
        if _ticks is not None:
            tickparam = {f"tick{self._axisnum}On": _ticks}
            self._axis.set_tick_params(**tickparam)
        
        # 根据 _ticklabels 更新轴的 tick labels 的可见性
        if _ticklabels is not None:
            tickparam = {f"label{self._axisnum}On": _ticklabels}
            self._axis.set_tick_params(**tickparam)

        # 根据 _label 更新轴的标签可见性及位置
        if _label is not None:
            pos = self._axis.get_label_position()
            # 若标签位置与轴方向一致且 _label 为 False，则隐藏标签
            if (pos == self._axis_direction) and not _label:
                self._axis.label.set_visible(False)
            elif _label:
                # 否则显示标签，并设置标签位置为轴方向
                self._axis.label.set_visible(True)
                self._axis.set_label_position(self._axis_direction)
```