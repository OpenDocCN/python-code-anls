# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\angle_annotation.py`

```py
"""
===========================
Scale invariant angle label
===========================

This example shows how to create a scale invariant angle annotation. It is
often useful to mark angles between lines or inside shapes with a circular arc.
While Matplotlib provides an `~.patches.Arc`, an inherent problem when directly
using it for such purposes is that an arc being circular in data space is not
necessarily circular in display space. Also, the arc's radius is often best
defined in a coordinate system which is independent of the actual data
coordinates - at least if you want to be able to freely zoom into your plot
without the annotation growing to infinity.

This calls for a solution where the arc's center is defined in data space, but
its radius in a physical unit like points or pixels, or as a ratio of the Axes
dimension. The following ``AngleAnnotation`` class provides such solution.

The example below serves two purposes:

* It provides a ready-to-use solution for the problem of easily drawing angles
  in graphs.
* It shows how to subclass a Matplotlib artist to enhance its functionality, as
  well as giving a hands-on example on how to use Matplotlib's :ref:`transform
  system <transforms_tutorial>`.

If mainly interested in the former, you may copy the below class and jump to
the :ref:`angle-annotation-usage` section.
"""

# %%
# AngleAnnotation class
# ---------------------
# The essential idea here is to subclass `~.patches.Arc` and set its transform
# to the `~.transforms.IdentityTransform`, making the parameters of the arc
# defined in pixel space.
# We then override the ``Arc``'s attributes ``_center``, ``theta1``,
# ``theta2``, ``width`` and ``height`` and make them properties, coupling to
# internal methods that calculate the respective parameters each time the
# attribute is accessed and thereby ensuring that the arc in pixel space stays
# synchronized with the input points and size.
# For example, each time the arc's drawing method would query its ``_center``
# attribute, instead of receiving the same number all over again, it will
# instead receive the result of the ``get_center_in_pixels`` method we defined
# in the subclass. This method transforms the center in data coordinates to
# pixels via the Axes transform ``ax.transData``. The size and the angles are
# calculated in a similar fashion, such that the arc changes its shape
# automatically when e.g. zooming or panning interactively.
#
# The functionality of this class allows to annotate the arc with a text. This
# text is a `~.text.Annotation` stored in an attribute ``text``. Since the
# arc's position and radius are defined only at draw time, we need to update
# the text's position accordingly. This is done by reimplementing the ``Arc``'s
# ``draw()`` method to let it call an updating method for the text.
#
# The arc and the text will be added to the provided Axes at instantiation: it
# is hence not strictly necessary to keep a reference to it.
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Arc
from matplotlib.transforms import Bbox, IdentityTransform, TransformedBbox


class AngleAnnotation(Arc):
    """
    Draws an arc between two vectors which appears circular in display space.
    """
    
    def __init__(self, xy, p1, p2, size=75, unit="points", ax=None,
                 text="", textposition="inside", text_kw=None, **kwargs):
        """
        Parameters
        ----------
        xy, p1, p2 : tuple or array of two floats
            Center position and two points. Angle annotation is drawn between
            the two vectors connecting *p1* and *p2* with *xy*, respectively.
            Units are data coordinates.

        size : float
            Diameter of the angle annotation in units specified by *unit*.

        unit : str
            One of the following strings to specify the unit of *size*:

            * "pixels": pixels
            * "points": points, use points instead of pixels to not have a
              dependence on the DPI
            * "axes width", "axes height": relative units of Axes width, height
            * "axes min", "axes max": minimum or maximum of relative Axes
              width, height

        ax : `matplotlib.axes.Axes`
            The Axes to add the angle annotation to.

        text : str
            The text to mark the angle with.

        textposition : {"inside", "outside", "edge"}
            Whether to show the text in- or outside the arc. "edge" can be used
            for custom positions anchored at the arc's edge.

        text_kw : dict
            Dictionary of arguments passed to the Annotation.

        **kwargs
            Further parameters are passed to `matplotlib.patches.Arc`. Use this
            to specify, color, linewidth etc. of the arc.

        """
        # Initialize the object with parameters
        self.ax = ax or plt.gca()  # Get current axes if ax is None
        self._xydata = xy  # Center position of the arc in data coordinates
        self.vec1 = p1  # First vector endpoint
        self.vec2 = p2  # Second vector endpoint
        self.size = size  # Diameter of the arc in specified units
        self.unit = unit  # Unit of size (pixels, points, axes width/height, etc.)
        self.textposition = textposition  # Position of text relative to the arc

        # Call superclass constructor to create the Arc object
        super().__init__(self._xydata, size, size, angle=0.0,
                         theta1=self.theta1, theta2=self.theta2, **kwargs)

        # Set the transformation of the Arc object to IdentityTransform
        self.set_transform(IdentityTransform())

        # Add the Arc object to the Axes
        self.ax.add_patch(self)

        # Define default text annotation parameters
        self.kw = dict(ha="center", va="center",
                       xycoords=IdentityTransform(),
                       xytext=(0, 0), textcoords="offset points",
                       annotation_clip=True)
        
        # Update default parameters with user-specified text_kw
        self.kw.update(text_kw or {})
        
        # Annotate the arc with the specified text at its center
        self.text = ax.annotate(text, xy=self._center, **self.kw)
    # 计算并返回尺寸，单位为像素
    def get_size(self):
        factor = 1.
        # 如果单位为 "points"，则计算缩放因子为当前图形的 DPI 与标准点（72）的比值
        if self.unit == "points":
            factor = self.ax.figure.dpi / 72.
        # 如果单位以 "axes" 开头，则计算基于坐标轴的尺寸信息
        elif self.unit[:4] == "axes":
            # 创建一个转换后的边界框对象，基于坐标轴的转换
            b = TransformedBbox(Bbox.unit(), self.ax.transAxes)
            # 构建包含最大、最小宽度和高度的字典
            dic = {"max": max(b.width, b.height),
                   "min": min(b.width, b.height),
                   "width": b.width, "height": b.height}
            # 根据属性的具体子属性索引选择缩放因子
            factor = dic[self.unit[5:]]
        # 返回尺寸乘以计算得到的缩放因子
        return self.size * factor

    # 设置对象的尺寸属性
    def set_size(self, size):
        self.size = size

    # 返回中心点的像素坐标
    def get_center_in_pixels(self):
        """返回像素空间中的中心点"""
        return self.ax.transData.transform(self._xydata)

    # 设置中心点坐标，参数为数据坐标
    def set_center(self, xy):
        """设置数据坐标中的中心点"""
        self._xydata = xy

    # 计算给定向量与中心点之间的角度（以度为单位）
    def get_theta(self, vec):
        # 将向量转换为像素坐标，并计算其相对于中心点的角度
        vec_in_pixels = self.ax.transData.transform(vec) - self._center
        return np.rad2deg(np.arctan2(vec_in_pixels[1], vec_in_pixels[0]))

    # 获取第一个向量与中心点的角度
    def get_theta1(self):
        return self.get_theta(self.vec1)

    # 获取第二个向量与中心点的角度
    def get_theta2(self):
        return self.get_theta(self.vec2)

    # 设置角度属性的占位方法
    def set_theta(self, angle):
        pass

    # 使用属性(property)重定义 Arc 对象的属性，以始终返回像素空间中的值
    _center = property(get_center_in_pixels, set_center)
    theta1 = property(get_theta1, set_theta)
    theta2 = property(get_theta2, set_theta)
    width = property(get_size, set_size)
    height = property(get_size, set_size)

    # 以下两个方法用于更新文本位置
    # 绘制对象，更新文本位置后调用超类的绘制方法
    def draw(self, renderer):
        self.update_text()
        super().draw(renderer)
    # 定义一个方法用于更新文本位置和角度
    def update_text(self):
        # 获取图形对象的中心点坐标
        c = self._center
        # 获取图形对象的尺寸
        s = self.get_size()
        # 计算角度跨度，确保在360度内
        angle_span = (self.theta2 - self.theta1) % 360
        # 计算文本所在角度的弧度值
        angle = np.deg2rad(self.theta1 + angle_span / 2)
        # 计算半径，根据文本位置调整
        r = s / 2
        if self.textposition == "inside":
            # 根据角度跨度插值计算内部文本的半径
            r = s / np.interp(angle_span, [60, 90, 135, 180],
                                          [3.3, 3.5, 3.8, 4])
        # 设置文本的坐标，根据中心点和角度计算
        self.text.xy = c + r * np.array([np.cos(angle), np.sin(angle)])
        if self.textposition == "outside":
            # 定义函数用于计算文本位于外部时的距离
            def R90(a, r, w, h):
                # 根据角度计算文本到中心的距离
                if a < np.arctan(h/2/(r+w/2)):
                    return np.sqrt((r+w/2)**2 + (np.tan(a)*(r+w/2))**2)
                else:
                    c = np.sqrt((w/2)**2+(h/2)**2)
                    T = np.arcsin(c * np.cos(np.pi/2 - a + np.arcsin(h/2/c))/r)
                    xy = r * np.array([np.cos(a + T), np.sin(a + T)])
                    xy += np.array([w/2, h/2])
                    return np.sqrt(np.sum(xy**2))

            # 定义函数用于计算文本位于外部时的距离
            def R(a, r, w, h):
                aa = (a % (np.pi/4))*((a % (np.pi/2)) <= np.pi/4) + \
                     (np.pi/4 - (a % (np.pi/4)))*((a % (np.pi/2)) >= np.pi/4)
                return R90(aa, r, *[w, h][::int(np.sign(np.cos(2*a)))])

            # 获取文本的包围框
            bbox = self.text.get_window_extent()
            # 计算文本位于外部时的距离
            X = R(angle, r, bbox.width, bbox.height)
            # 转换为像素坐标系
            trans = self.ax.figure.dpi_scale_trans.inverted()
            # 计算文本偏移量
            offs = trans.transform(((X-s/2), 0))[0] * 72
            # 设置文本的位置
            self.text.set_position([offs*np.cos(angle), offs*np.sin(angle)])
# %%
# .. _angle-annotation-usage:
#
# Usage
# -----
#
# Required arguments to ``AngleAnnotation`` are the center of the arc, *xy*,
# and two points, such that the arc spans between the two vectors connecting
# *p1* and *p2* with *xy*, respectively. Those are given in data coordinates.
# Further arguments are the *size* of the arc and its *unit*. Additionally, a
# *text* can be specified, that will be drawn either in- or outside of the arc,
# according to the value of *textposition*. Usage of those arguments is shown
# below.

fig, ax = plt.subplots()
fig.canvas.draw()  # 绘制图形以定义渲染器
ax.set_title("AngleLabel example")

# 绘制两条交叉线，并在每个角度处使用上述``AngleAnnotation``工具进行标注。
center = (4.5, 650)
p1 = [(2.5, 710), (6.0, 605)]
p2 = [(3.0, 275), (5.5, 900)]
line1, = ax.plot(*zip(*p1))
line2, = ax.plot(*zip(*p2))
point, = ax.plot(*center, marker="o")

am1 = AngleAnnotation(center, p1[1], p2[1], ax=ax, size=75, text=r"$\alpha$")
am2 = AngleAnnotation(center, p2[1], p1[0], ax=ax, size=35, text=r"$\beta$")
am3 = AngleAnnotation(center, p1[0], p2[0], ax=ax, size=75, text=r"$\gamma$")
am4 = AngleAnnotation(center, p2[0], p1[1], ax=ax, size=35, text=r"$\theta$")


# 展示角度弧线的一些样式选项，以及文本的选项。
p = [(6.0, 400), (5.3, 410), (5.6, 300)]
ax.plot(*zip(*p))
am5 = AngleAnnotation(p[1], p[0], p[2], ax=ax, size=40, text=r"$\Phi$",
                      linestyle="--", color="gray", textposition="outside",
                      text_kw=dict(fontsize=16, color="gray"))


# %%
# ``AngleLabel`` options
# ----------------------
#
# The *textposition* and *unit* keyword arguments may be used to modify the
# location of the text label, as shown below:


# 辅助函数，用于轻松绘制角度。
def plot_angle(ax, pos, angle, length=0.95, acol="C0", **kwargs):
    vec2 = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    xy = np.c_[[length, 0], [0, 0], vec2*length].T + np.array(pos)
    ax.plot(*xy.T, color=acol)
    return AngleAnnotation(pos, xy[0], xy[2], ax=ax, **kwargs)


fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
fig.suptitle("AngleLabel keyword arguments")
fig.canvas.draw()  # 绘制图形以定义渲染器

# 展示不同的文本位置。
ax1.margins(y=0.4)
ax1.set_title("textposition")
kw = dict(size=75, unit="points", text=r"$60°$")

am6 = plot_angle(ax1, (2.0, 0), 60, textposition="inside", **kw)
am7 = plot_angle(ax1, (3.5, 0), 60, textposition="outside", **kw)
am8 = plot_angle(ax1, (5.0, 0), 60, textposition="edge",
                 text_kw=dict(bbox=dict(boxstyle="round", fc="w")), **kw)
am9 = plot_angle(ax1, (6.5, 0), 60, textposition="edge",
                 text_kw=dict(xytext=(30, 20), arrowprops=dict(arrowstyle="->",
                              connectionstyle="arc3,rad=-0.2")), **kw)
# 遍历两个列表，一个是浮点数列表，一个是字符串列表，进行循环注释
for x, text in zip([2.0, 3.5, 5.0, 6.5], ['"inside"', '"outside"', '"edge"',
                                          '"edge", custom arrow']):
    # 在图表 ax1 上添加注释，位置为 (x, 0)，使用 x 轴的坐标系
    ax1.annotate(text, xy=(x, 0), xycoords=ax1.get_xaxis_transform(),
                 # 设置注释框样式为圆角矩形，填充颜色为白色
                 bbox=dict(boxstyle="round", fc="w"),
                 # 水平对齐方式为左对齐
                 ha="left", fontsize=8,
                 # 注释裁剪在图表区域内
                 annotation_clip=True)

# 调整 ax2 图表的垂直边距为 0.4
ax2.margins(y=0.4)
# 设置 ax2 图表的标题为 "unit"
ax2.set_title("unit")
# 定义一个包含文本和位置参数的字典
kw = dict(text=r"$60°$", textposition="outside")

# 在 ax2 图表上绘制四个角度标记，分别使用不同的单位大小
am10 = plot_angle(ax2, (2.0, 0), 60, size=50, unit="pixels", **kw)
am11 = plot_angle(ax2, (3.5, 0), 60, size=50, unit="points", **kw)
am12 = plot_angle(ax2, (5.0, 0), 60, size=0.25, unit="axes min", **kw)
am13 = plot_angle(ax2, (6.5, 0), 60, size=0.25, unit="axes max", **kw)

# 遍历两个列表，一个是浮点数列表，一个是字符串列表，进行循环注释
for x, text in zip([2.0, 3.5, 5.0, 6.5], ['"pixels"', '"points"',
                                          '"axes min"', '"axes max"']):
    # 在图表 ax2 上添加注释，位置为 (x, 0)，使用 x 轴的坐标系
    ax2.annotate(text, xy=(x, 0), xycoords=ax2.get_xaxis_transform(),
                 # 设置注释框样式为圆角矩形，填充颜色为白色
                 bbox=dict(boxstyle="round", fc="w"),
                 # 水平对齐方式为左对齐
                 ha="left", fontsize=8,
                 # 注释裁剪在图表区域内
                 annotation_clip=True)

# 显示 matplotlib 图形
plt.show()
```