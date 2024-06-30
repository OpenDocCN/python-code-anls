# `D:\src\scipysrc\sympy\sympy\plotting\pygletplot\plot_axes.py`

```
import pyglet.gl as pgl  # 导入 pyglet 的 OpenGL 模块，并重命名为 pgl
from pyglet import font  # 导入 pyglet 的字体模块

from sympy.core import S  # 导入 SymPy 的 S 对象
from sympy.plotting.pygletplot.plot_object import PlotObject  # 导入 SymPy 的绘图对象 PlotObject
from sympy.plotting.pygletplot.util import (billboard_matrix, dot_product,  # 导入 SymPy 绘图的实用函数
        get_direction_vectors, strided_range, vec_mag, vec_sub)
from sympy.utilities.iterables import is_sequence  # 导入 SymPy 的迭代工具中的 is_sequence 函数


class PlotAxes(PlotObject):
    # 继承自 PlotObject 的绘图坐标轴类

    def __init__(self, *args,
            style='', none=None, frame=None, box=None, ordinate=None,
            stride=0.25,
            visible='', overlay='', colored='', label_axes='', label_ticks='',
            tick_length=0.1,
            font_face='Arial', font_size=28,
            **kwargs):
        # 初始化函数，接受多个参数和关键字参数

        # 初始化样式参数
        style = style.lower()

        # 允许别名关键字参数覆盖样式关键字参数
        if none is not None:
            style = 'none'
        if frame is not None:
            style = 'frame'
        if box is not None:
            style = 'box'
        if ordinate is not None:
            style = 'ordinate'

        if style in ['', 'ordinate']:
            self._render_object = PlotAxesOrdinate(self)  # 创建 PlotAxesOrdinate 实例
        elif style in ['frame', 'box']:
            self._render_object = PlotAxesFrame(self)  # 创建 PlotAxesFrame 实例
        elif style in ['none']:
            self._render_object = None  # 不渲染任何对象
        else:
            raise ValueError(("Unrecognized axes style %s.") % (style))  # 抛出异常，样式无法识别

        # 初始化步长参数
        try:
            stride = eval(stride)  # 尝试将 stride 转换为表达式的结果
        except TypeError:
            pass
        if is_sequence(stride):  # 如果 stride 是一个序列
            if len(stride) != 3:
                raise ValueError("length should be equal to 3")  # 如果长度不等于 3，抛出异常
            self._stride = stride  # 设置步长为给定的序列
        else:
            self._stride = [stride, stride, stride]  # 设置步长为给定的单个值的序列
        self._tick_length = float(tick_length)  # 设置刻度长度为给定的浮点数值

        # 设置边界框和刻度
        self._origin = [0, 0, 0]  # 设置原点为 [0, 0, 0]
        self.reset_bounding_box()  # 调用重置边界框的方法

        def flexible_boolean(input, default):
            # 定义一个灵活的布尔值转换函数
            if input in [True, False]:
                return input
            if input in ('f', 'F', 'false', 'False'):
                return False
            if input in ('t', 'T', 'true', 'True'):
                return True
            return default

        # 初始化剩余参数
        self.visible = flexible_boolean(kwargs, True)  # 可见性参数，默认为 True
        self._overlay = flexible_boolean(overlay, True)  # 叠加效果参数，默认为 True
        self._colored = flexible_boolean(colored, False)  # 色彩化参数，默认为 False
        self._label_axes = flexible_boolean(label_axes, False)  # 标签轴参数，默认为 False
        self._label_ticks = flexible_boolean(label_ticks, True)  # 标签刻度参数，默认为 True

        # 设置标签字体
        self.font_face = font_face  # 设置字体类型
        self.font_size = font_size  # 设置字体大小

        # 这也用于重新初始化字体
        # 在窗口关闭/重新打开时
        self.reset_resources()  # 调用重置资源的方法

    def reset_resources(self):
        # 重置资源的方法
        self.label_font = None  # 标签字体设为 None

    def reset_bounding_box(self):
        # 重置边界框的方法
        self._bounding_box = [[None, None], [None, None], [None, None]]  # 初始化边界框
        self._axis_ticks = [[], [], []]  # 初始化轴刻度
    # 绘制函数，用于渲染对象到屏幕上
    def draw(self):
        # 检查是否存在要渲染的对象
        if self._render_object:
            # 保存当前 OpenGL 的状态
            pgl.glPushAttrib(pgl.GL_ENABLE_BIT | pgl.GL_POLYGON_BIT | pgl.GL_DEPTH_BUFFER_BIT)
            # 如果是叠加层，则禁用深度测试
            if self._overlay:
                pgl.glDisable(pgl.GL_DEPTH_TEST)
            # 调用渲染对象的绘制方法
            self._render_object.draw()
            # 恢复 OpenGL 的状态
            pgl.glPopAttrib()

    # 调整边界函数，根据子对象的边界调整当前对象的边界框
    def adjust_bounds(self, child_bounds):
        # 获取当前对象和子对象的边界框
        b = self._bounding_box
        c = child_bounds
        # 对每个轴进行迭代
        for i in range(3):
            # 如果子对象在该轴上的边界为无穷大，则忽略
            if abs(c[i][0]) is S.Infinity or abs(c[i][1]) is S.Infinity:
                continue
            # 调整当前对象的边界框的最小值和最大值
            b[i][0] = c[i][0] if b[i][0] is None else min([b[i][0], c[i][0]])
            b[i][1] = c[i][1] if b[i][1] is None else max([b[i][1], c[i][1]])
            # 更新当前对象的边界框
            self._bounding_box = b
            # 重新计算该轴上的刻度
            self._recalculate_axis_ticks(i)

    # 重新计算指定轴上的刻度函数
    def _recalculate_axis_ticks(self, axis):
        # 获取当前对象的边界框
        b = self._bounding_box
        # 如果指定轴上的边界框最小值或最大值为 None，则该轴上的刻度为空列表
        if b[axis][0] is None or b[axis][1] is None:
            self._axis_ticks[axis] = []
        else:
            # 否则，根据指定的步长计算该轴上的刻度
            self._axis_ticks[axis] = strided_range(b[axis][0], b[axis][1],
                                                   self._stride[axis])

    # 切换可见性函数，反转当前对象的可见性状态
    def toggle_visible(self):
        self.visible = not self.visible

    # 切换颜色函数，反转当前对象是否应该着色的状态
    def toggle_colors(self):
        self._colored = not self._colored
# 定义一个绘图坐标轴基类，继承自绘图对象基类 PlotObject
class PlotAxesBase(PlotObject):

    # 构造函数，初始化父级坐标轴对象
    def __init__(self, parent_axes):
        self._p = parent_axes

    # 绘制函数，绘制坐标轴的背景和三条轴线
    def draw(self):
        # 根据父级坐标轴对象的颜色状态选择背景色
        color = [([0.2, 0.1, 0.3], [0.2, 0.1, 0.3], [0.2, 0.1, 0.3]),
                 ([0.9, 0.3, 0.5], [0.5, 1.0, 0.5], [0.3, 0.3, 0.9])][self._p._colored]
        # 绘制背景
        self.draw_background(color)
        # 分别绘制三个坐标轴
        self.draw_axis(2, color[2])
        self.draw_axis(1, color[1])
        self.draw_axis(0, color[0])

    # 绘制背景的占位函数，可以选择性地实现
    def draw_background(self, color):
        pass  # optional

    # 抽象方法，子类需要实现的方法，用于绘制单个坐标轴
    def draw_axis(self, axis, color):
        raise NotImplementedError()

    # 绘制文本信息的方法
    def draw_text(self, text, position, color, scale=1.0):
        # 如果颜色是 RGB 格式，则转换为带 Alpha 通道的 RGBA 格式
        if len(color) == 3:
            color = (color[0], color[1], color[2], 1.0)

        # 如果父级坐标轴对象的标签字体为 None，则加载字体文件
        if self._p.label_font is None:
            self._p.label_font = font.load(self._p.font_face,
                                           self._p.font_size,
                                           bold=True, italic=False)

        # 创建文本对象
        label = font.Text(self._p.label_font, text,
                          color=color,
                          valign=font.Text.BASELINE,
                          halign=font.Text.CENTER)

        # 对文本进行平移、矩阵变换和缩放，然后绘制
        pgl.glPushMatrix()
        pgl.glTranslatef(*position)
        billboard_matrix()
        scale_factor = 0.005 * scale
        pgl.glScalef(scale_factor, scale_factor, scale_factor)
        pgl.glColor4f(0, 0, 0, 0)
        label.draw()
        pgl.glPopMatrix()

    # 绘制线条的方法
    def draw_line(self, v, color):
        # 获取父级坐标轴对象的原点坐标
        o = self._p._origin
        # 开始绘制线段
        pgl.glBegin(pgl.GL_LINES)
        pgl.glColor3f(*color)
        # 绘制线段的起点和终点
        pgl.glVertex3f(v[0][0] + o[0], v[0][1] + o[1], v[0][2] + o[2])
        pgl.glVertex3f(v[1][0] + o[0], v[1][1] + o[1], v[1][2] + o[2])
        pgl.glEnd()


# 绘图坐标轴子类，继承自绘图坐标轴基类 PlotAxesBase
class PlotAxesOrdinate(PlotAxesBase):

    # 构造函数，调用父类的构造函数初始化父级坐标轴对象
    def __init__(self, parent_axes):
        super().__init__(parent_axes)

    # 实现抽象方法，绘制单个坐标轴
    def draw_axis(self, axis, color):
        # 获取父级坐标轴对象中对应轴的刻度值
        ticks = self._p._axis_ticks[axis]
        # 计算刻度线的半径
        radius = self._p._tick_length / 2.0
        # 如果刻度值少于两个，不绘制
        if len(ticks) < 2:
            return

        # 计算当前轴的向量
        axis_lines = [[0, 0, 0], [0, 0, 0]]
        axis_lines[0][axis], axis_lines[1][axis] = ticks[0], ticks[-1]
        axis_vector = vec_sub(axis_lines[1], axis_lines[0])

        # 计算轴向量与 Z 方向向量的夹角
        pos_z = get_direction_vectors()[2]
        d = abs(dot_product(axis_vector, pos_z))
        d = d / vec_mag(axis_vector)

        # 如果我们正对轴朝向，则不绘制标签
        labels_visible = abs(d - 1.0) > 0.02

        # 绘制刻度和标签
        for tick in ticks:
            self.draw_tick_line(axis, color, radius, tick, labels_visible)

        # 绘制轴线和标签
        self.draw_axis_line(axis, color, ticks[0], ticks[-1], labels_visible)
    # 绘制轴线，给定轴、颜色、最小值、最大值和标签可见性
    def draw_axis_line(self, axis, color, a_min, a_max, labels_visible):
        # 创建轴线的起始和结束点坐标
        axis_line = [[0, 0, 0], [0, 0, 0]]
        axis_line[0][axis], axis_line[1][axis] = a_min, a_max
        # 调用 draw_line 方法绘制轴线
        self.draw_line(axis_line, color)
        # 如果标签可见，调用 draw_axis_line_labels 方法绘制轴线上的标签
        if labels_visible:
            self.draw_axis_line_labels(axis, color, axis_line)

    # 绘制轴线上的标签，给定轴、颜色和轴线的坐标
    def draw_axis_line_labels(self, axis, color, axis_line):
        # 如果不需要标签，直接返回
        if not self._p._label_axes:
            return
        # 创建轴线上标签的坐标点
        axis_labels = [axis_line[0][::], axis_line[1][::]]
        axis_labels[0][axis] -= 0.3
        axis_labels[1][axis] += 0.3
        # 根据轴的索引确定标签的字符，如 X、Y、Z
        a_str = ['X', 'Y', 'Z'][axis]
        # 调用 draw_text 方法绘制轴线上的标签
        self.draw_text("-" + a_str, axis_labels[0], color)
        self.draw_text("+" + a_str, axis_labels[1], color)

    # 绘制刻度线，给定轴、颜色、半径、刻度位置和标签可见性
    def draw_tick_line(self, axis, color, radius, tick, labels_visible):
        # 确定刻度线在轴和垂直轴上的起始和结束点坐标
        tick_axis = {0: 1, 1: 0, 2: 1}[axis]
        tick_line = [[0, 0, 0], [0, 0, 0]]
        tick_line[0][axis] = tick_line[1][axis] = tick
        tick_line[0][tick_axis], tick_line[1][tick_axis] = -radius, radius
        # 调用 draw_line 方法绘制刻度线
        self.draw_line(tick_line, color)
        # 如果标签可见，调用 draw_tick_line_label 方法绘制刻度线上的标签
        if labels_visible:
            self.draw_tick_line_label(axis, color, radius, tick)

    # 绘制刻度线上的标签，给定轴、颜色、半径和刻度位置
    def draw_tick_line_label(self, axis, color, radius, tick):
        # 如果不需要标签，直接返回
        if not self._p._label_axes:
            return
        # 创建刻度线上标签的坐标点
        tick_label_vector = [0, 0, 0]
        tick_label_vector[axis] = tick
        tick_label_vector[{0: 1, 1: 0, 2: 1}[axis]] = [-1, 1, 1][axis] * radius * 3.5
        # 调用 draw_text 方法绘制刻度线上的标签，标签文字为刻度值的字符串形式
        self.draw_text(str(tick), tick_label_vector, color, scale=0.5)
# 定义一个名为 PlotAxesFrame 的类，继承自 PlotAxesBase 类
class PlotAxesFrame(PlotAxesBase):

    # 初始化方法，接受一个 parent_axes 参数，并调用父类的初始化方法
    def __init__(self, parent_axes):
        super().__init__(parent_axes)

    # 定义一个名为 draw_background 的方法，用于绘制背景，但当前实现是空的
    def draw_background(self, color):
        # pass 表示此处暂不实现任何功能，保留方法的声明而无实际操作
        pass

    # 定义一个名为 draw_axis 的方法，用于绘制坐标轴，但当前抛出未实现异常
    def draw_axis(self, axis, color):
        # 抛出一个 NotImplementedError 异常，提示子类需要实现这个方法
        raise NotImplementedError()
```