# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_skew.py`

```py
"""
Testing that skewed Axes properly work.
"""

# 导入所需的库和模块
from contextlib import ExitStack  # 用于创建上下文管理器，确保资源被正确释放
import itertools  # 用于生成迭代器的函数
import platform  # 提供关于操作系统平台的信息

import matplotlib.pyplot as plt  # Matplotlib 的绘图模块
from matplotlib.testing.decorators import image_comparison  # 用于测试图像对比的装饰器

from matplotlib.axes import Axes  # Matplotlib 的轴对象
import matplotlib.transforms as transforms  # 提供用于转换的类和函数
import matplotlib.axis as maxis  # 轴对象的基类
import matplotlib.spines as mspines  # 轴脊柱对象
import matplotlib.patches as mpatch  # 图形基类
from matplotlib.projections import register_projection  # 注册投影


# 该类的唯一目的是查看上限、下限或总间隔（取决于情况），并确定要绘制的刻度的哪些部分。
class SkewXTick(maxis.XTick):
    def draw(self, renderer):
        with ExitStack() as stack:
            for artist in [self.gridline, self.tick1line, self.tick2line,
                           self.label1, self.label2]:
                stack.callback(artist.set_visible, artist.get_visible())
            needs_lower = transforms.interval_contains(
                self.axes.lower_xlim, self.get_loc())
            needs_upper = transforms.interval_contains(
                self.axes.upper_xlim, self.get_loc())
            self.tick1line.set_visible(
                self.tick1line.get_visible() and needs_lower)
            self.label1.set_visible(
                self.label1.get_visible() and needs_lower)
            self.tick2line.set_visible(
                self.tick2line.get_visible() and needs_upper)
            self.label2.set_visible(
                self.label2.get_visible() and needs_upper)
            super().draw(renderer)

    def get_view_interval(self):
        return self.axes.xaxis.get_view_interval()


# 该类存在的目的是为刻度提供两组独立的间隔，并创建自定义刻度的实例。
class SkewXAxis(maxis.XAxis):
    def _get_tick(self, major):
        return SkewXTick(self.axes, None, major=major)

    def get_view_interval(self):
        return self.axes.upper_xlim[0], self.axes.lower_xlim[1]


# 该类存在的目的是计算上部 X 轴的单独数据范围并在那里绘制脊柱。
# 它还为 X 轴艺术家提供这一范围以进行刻度和网格线的绘制。
class SkewSpine(mspines.Spine):
    def _adjust_location(self):
        pts = self._path.vertices
        if self.spine_type == 'top':
            pts[:, 0] = self.axes.upper_xlim
        else:
            pts[:, 0] = self.axes.lower_xlim


# 该类处理将 skew-xaxes 注册为投影，并设置适当的转换。
# 它还根据需要重写标准的脊柱和轴实例。
class SkewXAxes(Axes):
    # 投影必须指定一个名称。用户将使用此名称选择投影，例如 ``subplot(projection='skewx')``。
    name = 'skewx'
    # 初始化坐标轴设置，替换默认的 X 轴为 SkewXAxis
    def _init_axis(self):
        # 使用修改后的 X 轴对象 SkewXAxis
        self.xaxis = SkewXAxis(self)
        # 将顶部和底部的脊柱注册为修改后的 X 轴对象
        self.spines.top.register_axis(self.xaxis)
        self.spines.bottom.register_axis(self.xaxis)
        # 创建默认的 Y 轴对象
        self.yaxis = maxis.YAxis(self)
        # 将左侧和右侧的脊柱注册为默认的 Y 轴对象
        self.spines.left.register_axis(self.yaxis)
        self.spines.right.register_axis(self.yaxis)

    # 生成和返回脊柱的字典，包括顶部、底部、左侧和右侧
    def _gen_axes_spines(self):
        spines = {'top': SkewSpine.linear_spine(self, 'top'),
                  'bottom': mspines.Spine.linear_spine(self, 'bottom'),
                  'left': mspines.Spine.linear_spine(self, 'left'),
                  'right': mspines.Spine.linear_spine(self, 'right')}
        return spines

    # 设置数据的限制和转换，这在绘图创建时调用一次，用于设置数据、文本和网格的所有转换
    def _set_lim_and_transforms(self):
        rot = 30

        # 调用父类的方法设置数据的限制和转换
        super()._set_lim_and_transforms()

        # 在转换链中插入斜率变换，以确保在数据转换到坐标轴之前斜率已经应用
        self.transDataToAxes = (self.transScale +
                                (self.transLimits +
                                 transforms.Affine2D().skew_deg(rot, 0)))

        # 创建从数据到像素的完整转换
        self.transData = self.transDataToAxes + self.transAxes

        # 创建混合转换，确保斜率应用于两个轴的坐标系
        self._xaxis_transform = (transforms.blended_transform_factory(
            self.transScale + self.transLimits,
            transforms.IdentityTransform()) +
            transforms.Affine2D().skew_deg(rot, 0)) + self.transAxes

    # 返回当前坐标系下 X 轴的下限
    @property
    def lower_xlim(self):
        return self.axes.viewLim.intervalx

    # 返回当前坐标系下 X 轴的上限
    @property
    def upper_xlim(self):
        # 创建一个虚拟点集，通过转换获取其在数据坐标系下的位置，然后返回 X 轴的上限
        pts = [[0., 1.], [1., 1.]]
        return self.transDataToAxes.inverted().transform(pts)[:, 0]
# 现在将投影注册到 matplotlib 中，以便用户可以选择它。
register_projection(SkewXAxes)

# 使用图像比较测试函数装饰器，比较生成的图像与预期的图像（skew_axes），并移除文本信息。
@image_comparison(['skew_axes'], remove_text=True)
def test_set_line_coll_dash_image():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 添加一个子图，使用投影为 'skewx'
    ax = fig.add_subplot(1, 1, 1, projection='skewx')
    # 设置 X 轴的显示范围
    ax.set_xlim(-50, 50)
    # 设置 Y 轴的显示范围，注意是反向的
    ax.set_ylim(50, -50)
    # 打开网格显示
    ax.grid(True)

    # 创建一个垂直于 X 轴的蓝色直线
    ax.axvline(0, color='b')


# 使用图像比较测试函数装饰器，比较生成的图像与预期的图像（skew_rects），并移除文本信息，
# 如果平台为 arm64，则容差为 0.009，否则为 0。
@image_comparison(['skew_rects'], remove_text=True,
                  tol=0.009 if platform.machine() == 'arm64' else 0)
def test_skew_rectangle():
    # 创建一个包含 5x5 个子图的图形对象，共享 X 轴和 Y 轴，大小为 8x8 英寸
    fix, axes = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(8, 8))
    axes = axes.flat

    # 生成所有可能的旋转组合
    rotations = list(itertools.product([-3, -1, 0, 1, 3], repeat=2))

    # 设置第一个子图的 X 和 Y 轴显示范围
    axes[0].set_xlim([-3, 3])
    axes[0].set_ylim([-3, 3])
    # 设置纵横比为相等，并共享此设置
    axes[0].set_aspect('equal', share=True)

    # 对于每个子图和对应的旋转角度组合
    for ax, (xrots, yrots) in zip(axes, rotations):
        # 计算 X 和 Y 的角度偏移
        xdeg, ydeg = 45 * xrots, 45 * yrots
        # 创建一个仿射变换对象，并斜切指定的角度
        t = transforms.Affine2D().skew_deg(xdeg, ydeg)

        # 设置子图的标题，显示 X 和 Y 方向的斜切角度
        ax.set_title(f'Skew of {xdeg} in X and {ydeg} in Y')
        # 添加一个矩形补丁到子图中，基于变换 t 加上数据坐标系
        ax.add_patch(mpatch.Rectangle([-1, -1], 2, 2,
                                      transform=t + ax.transData,
                                      alpha=0.5, facecolor='coral'))

    # 调整子图之间的间距和边界
    plt.subplots_adjust(wspace=0, left=0.01, right=0.99, bottom=0.01, top=0.99)
```