# `D:\src\scipysrc\matplotlib\lib\matplotlib\spines.py`

```
# 导入MutableMapping抽象基类
from collections.abc import MutableMapping
# 导入functools模块
import functools

# 导入NumPy库，用于数值计算和数组操作
import numpy as np

# 导入Matplotlib库及其子模块和函数
import matplotlib as mpl
from matplotlib import _api, _docstring
# 导入allow_rasterization函数，允许光栅化
from matplotlib.artist import allow_rasterization
# 导入mtransforms模块，用于数据变换
import matplotlib.transforms as mtransforms
# 导入mpatches模块，包含图形的各种形状补丁
import matplotlib.patches as mpatches
# 导入mpath模块，用于处理路径对象
import matplotlib.path as mpath

# 定义一个名为Spine的类，继承自mpatches.Patch类
class Spine(mpatches.Patch):
    """
    An axis spine -- the line noting the data area boundaries.

    Spines are the lines connecting the axis tick marks and noting the
    boundaries of the data area. They can be placed at arbitrary
    positions. See `~.Spine.set_position` for more information.

    The default position is ``('outward', 0)``.

    Spines are subclasses of `.Patch`, and inherit much of their behavior.

    Spines draw a line, a circle, or an arc depending on if
    `~.Spine.set_patch_line`, `~.Spine.set_patch_circle`, or
    `~.Spine.set_patch_arc` has been called. Line-like is the default.

    For examples see :ref:`spines_examples`.
    """

    # 返回对象的字符串表示形式
    def __str__(self):
        return "Spine"

    # 构造函数，初始化一个Spine对象
    @_docstring.dedent_interpd
    def __init__(self, axes, spine_type, path, **kwargs):
        """
        Parameters
        ----------
        axes : `~matplotlib.axes.Axes`
            The `~.axes.Axes` instance containing the spine.
        spine_type : str
            The spine type.
        path : `~matplotlib.path.Path`
            The `.Path` instance used to draw the spine.

        Other Parameters
        ----------------
        **kwargs
            Valid keyword arguments are:

            %(Patch:kwdoc)s
        """
        # 调用父类构造函数初始化补丁对象
        super().__init__(**kwargs)
        # 设置axes属性为传入的Axes对象
        self.axes = axes
        # 设置补丁对象所属的Figure对象
        self.set_figure(self.axes.figure)
        # 设置脊柱类型
        self.spine_type = spine_type
        # 设置面颜色为none
        self.set_facecolor('none')
        # 设置边缘颜色为默认的axes.edgecolor参数
        self.set_edgecolor(mpl.rcParams['axes.edgecolor'])
        # 设置线宽为默认的axes.linewidth参数
        self.set_linewidth(mpl.rcParams['axes.linewidth'])
        # 设置线帽风格为'projecting'
        self.set_capstyle('projecting')
        # 初始化axis属性为None
        self.axis = None

        # 设置绘制顺序为2.5
        self.set_zorder(2.5)
        # 设置默认变换为axes.transData
        self.set_transform(self.axes.transData)

        # 初始化边界为None
        self._bounds = None

        # 延迟初始位置的确定
        self._position = None
        # 检查path参数是否为matplotlib.path.Path类型的实例
        _api.check_isinstance(mpath.Path, path=path)
        # 设置_path属性为传入的Path对象
        self._path = path

        # 设置补丁类型为'line'，用于表示补丁的默认行为为线性
        self._patch_type = 'line'

        # 行为来自mpatches.Ellipse的复制
        # 注意：这只有在添加到Axes后才能计算
        self._patch_transform = mtransforms.IdentityTransform()
    # 设置图形为弧形
    def set_patch_arc(self, center, radius, theta1, theta2):
        """Set the spine to be arc-like."""
        self._patch_type = 'arc'  # 设置图形类型为弧形
        self._center = center  # 设置图形中心点
        self._width = radius * 2  # 设置图形宽度
        self._height = radius * 2  # 设置图形高度
        self._theta1 = theta1  # 设置弧的起始角度
        self._theta2 = theta2  # 设置弧的终止角度
        self._path = mpath.Path.arc(theta1, theta2)  # 创建弧形路径
        # 在坐标轴变换下绘制弧形
        self.set_transform(self.axes.transAxes)
        self.stale = True  # 标记为需要更新

    # 设置图形为圆形
    def set_patch_circle(self, center, radius):
        """Set the spine to be circular."""
        self._patch_type = 'circle'  # 设置图形类型为圆形
        self._center = center  # 设置图形中心点
        self._width = radius * 2  # 设置图形宽度
        self._height = radius * 2  # 设置图形高度
        # 在坐标轴变换下绘制圆形
        self.set_transform(self.axes.transAxes)
        self.stale = True  # 标记为需要更新

    # 设置图形为直线
    def set_patch_line(self):
        """Set the spine to be linear."""
        self._patch_type = 'line'  # 设置图形类型为直线
        self.stale = True  # 标记为需要更新

    # 从 mpatches.Ellipse 中复制的行为
    def _recompute_transform(self):
        """
        Notes
        -----
        This cannot be called until after this has been added to an Axes,
        otherwise unit conversion will fail. This makes it very important to
        call the accessor method and not directly access the transformation
        member variable.
        """
        assert self._patch_type in ('arc', 'circle')  # 断言图形类型为弧形或圆形
        center = (self.convert_xunits(self._center[0]),
                  self.convert_yunits(self._center[1]))  # 转换中心点坐标单位
        width = self.convert_xunits(self._width)  # 转换宽度单位
        height = self.convert_yunits(self._height)  # 转换高度单位
        # 创建仿射变换对象并进行缩放和平移
        self._patch_transform = mtransforms.Affine2D() \
            .scale(width * 0.5, height * 0.5) \
            .translate(*center)

    # 获取图形变换
    def get_patch_transform(self):
        if self._patch_type in ('arc', 'circle'):
            self._recompute_transform()  # 重新计算变换
            return self._patch_transform  # 返回图形的变换
        else:
            return super().get_patch_transform()  # 返回默认的图形变换方法
    def get_window_extent(self, renderer=None):
        """
        Return the window extent of the spines in display space, including
        padding for ticks (but not their labels)

        See Also
        --------
        matplotlib.axes.Axes.get_tightbbox
        matplotlib.axes.Axes.get_window_extent
        """
        # 确保位置已更新，以便正确使用变换等操作：
        self._adjust_location()
        # 调用父类方法获取当前边框的窗口范围
        bb = super().get_window_extent(renderer=renderer)
        # 如果轴为空或不可见，直接返回边框
        if self.axis is None or not self.axis.get_visible():
            return bb
        # 初始化边框列表，包含当前边框
        bboxes = [bb]
        # 更新并获取绘制的刻度
        drawn_ticks = self.axis._update_ticks()

        # 获取主要刻度和次要刻度
        major_tick = next(iter({*drawn_ticks} & {*self.axis.majorTicks}), None)
        minor_tick = next(iter({*drawn_ticks} & {*self.axis.minorTicks}), None)
        for tick in [major_tick, minor_tick]:
            if tick is None:
                continue
            # 冻结当前边框状态
            bb0 = bb.frozen()
            tickl = tick._size
            tickdir = tick._tickdir
            # 根据刻度的方向确定外部和内部的填充值
            if tickdir == 'out':
                padout = 1
                padin = 0
            elif tickdir == 'in':
                padout = 0
                padin = 1
            else:
                padout = 0.5
                padin = 0.5
            # 计算填充的像素值
            padout = padout * tickl / 72 * self.figure.dpi
            padin = padin * tickl / 72 * self.figure.dpi

            # 如果刻度线1可见，根据边框类型调整边框的范围
            if tick.tick1line.get_visible():
                if self.spine_type == 'left':
                    bb0.x0 = bb0.x0 - padout
                    bb0.x1 = bb0.x1 + padin
                elif self.spine_type == 'bottom':
                    bb0.y0 = bb0.y0 - padout
                    bb0.y1 = bb0.y1 + padin

            # 如果刻度线2可见，根据边框类型调整边框的范围
            if tick.tick2line.get_visible():
                if self.spine_type == 'right':
                    bb0.x1 = bb0.x1 + padout
                    bb0.x0 = bb0.x0 - padin
                elif self.spine_type == 'top':
                    bb0.y1 = bb0.y1 + padout
                    bb0.y0 = bb0.y0 - padout
            # 将调整后的边框加入边框列表
            bboxes.append(bb0)

        # 返回所有边框的联合边框
        return mtransforms.Bbox.union(bboxes)

    def get_path(self):
        return self._path

    def _ensure_position_is_set(self):
        if self._position is None:
            # 默认位置
            self._position = ('outward', 0.0)  # 单位为点
            self.set_position(self._position)

    def register_axis(self, axis):
        """
        注册一个轴。

        应该使用其对应的轴实例来注册轴。这允许脊柱在需要时清除任何轴属性。
        """
        self.axis = axis
        self.stale = True

    def clear(self):
        """清除当前脊柱。"""
        self._clear()
        if self.axis is not None:
            self.axis.clear()
    def _clear(self):
        """
        Clear things directly related to the spine.

        In this way it is possible to avoid clearing the Axis as well when calling
        from library code where it is known that the Axis is cleared separately.
        """
        self._position = None  # clear position



    def _adjust_location(self):
        """Automatically set spine bounds to the view interval."""

        # 如果脊柱类型是圆形，则直接返回，不进行调整
        if self.spine_type == 'circle':
            return

        # 根据不同的脊柱类型设置低和高的边界值
        if self._bounds is not None:
            low, high = self._bounds
        elif self.spine_type in ('left', 'right'):
            low, high = self.axes.viewLim.intervaly
        elif self.spine_type in ('top', 'bottom'):
            low, high = self.axes.viewLim.intervalx
        else:
            raise ValueError(f'unknown spine spine_type: {self.spine_type}')

        # 如果路径类型是弧线并且脊柱类型是顶部或底部
        if self._patch_type == 'arc':
            if self.spine_type in ('bottom', 'top'):
                # 获取极坐标系的方向和偏移量
                try:
                    direction = self.axes.get_theta_direction()
                except AttributeError:
                    direction = 1
                try:
                    offset = self.axes.get_theta_offset()
                except AttributeError:
                    offset = 0
                # 根据方向和偏移量调整低和高的值
                low = low * direction + offset
                high = high * direction + offset
                if low > high:
                    low, high = high, low

                # 创建弧线路径
                self._path = mpath.Path.arc(np.rad2deg(low), np.rad2deg(high))

                # 如果脊柱类型是底部
                if self.spine_type == 'bottom':
                    rmin, rmax = self.axes.viewLim.intervaly
                    try:
                        rorigin = self.axes.get_rorigin()
                    except AttributeError:
                        rorigin = rmin
                    # 计算缩放后的直径
                    scaled_diameter = (rmin - rorigin) / (rmax - rorigin)
                    self._height = scaled_diameter
                    self._width = scaled_diameter

            else:
                # 如果脊柱类型不是底部或顶部，抛出异常
                raise ValueError('unable to set bounds for spine "%s"' %
                                 self.spine_type)
        else:
            # 如果路径类型不是弧线，则直接设置路径的顶点坐标
            v1 = self._path.vertices
            assert v1.shape == (2, 2), 'unexpected vertices shape'
            if self.spine_type in ['left', 'right']:
                v1[0, 1] = low
                v1[1, 1] = high
            elif self.spine_type in ['bottom', 'top']:
                v1[0, 0] = low
                v1[1, 0] = high
            else:
                raise ValueError('unable to set bounds for spine "%s"' %
                                 self.spine_type)

    @allow_rasterization
    def draw(self, renderer):
        """
        Draw the spine using the provided renderer.

        Adjusts the spine location before drawing, marks as not stale after drawing.
        """
        # 调整脊柱的位置
        self._adjust_location()
        # 调用父类的 draw 方法进行绘制，并获取返回值
        ret = super().draw(renderer)
        # 将 stale 属性设置为 False，表示不再过时
        self.stale = False
        return ret
    # 设置脊柱的位置

    Spine（脊柱）的位置由一个二元组（位置类型，位置量）指定。位置类型包括：

    * 'outward': 将脊柱从数据区域向外移动指定的点数。（负值将脊柱向内移动。）
    * 'axes': 将脊柱放置在指定的坐标轴坐标上（0到1之间）。
    * 'data': 将脊柱放置在指定的数据坐标上。

    此外，还有特殊位置的简写定义：

    * 'center' -> `('axes', 0.5)`
    * 'zero' -> `('data', 0.0)`

    示例
    --------
    :doc:`/gallery/spines/spine_placement_demo`

    def set_position(self, position):
        """设置脊柱的位置。

        参数 position 可以是特殊位置 'center' 或 'zero'。
        如果不是特殊位置，则应为一个包含两个元素的元组。

        Raises
        ------
        ValueError
            如果 position 不是 'center' 或 2 元组。
            或者 position[0] 不是 'outward', 'axes' 或 'data' 中的一个。

        """
        if position in ('center', 'zero'):  # 特殊位置
            pass
        else:
            if len(position) != 2:
                raise ValueError("position should be 'center' or 2-tuple")
            if position[0] not in ['outward', 'axes', 'data']:
                raise ValueError("position[0] should be one of 'outward', "
                                 "'axes', or 'data' ")
        self._position = position  # 设置脊柱的位置
        self.set_transform(self.get_spine_transform())  # 更新脊柱的变换
        if self.axis is not None:
            self.axis.reset_ticks()  # 重置刻度
        self.stale = True  # 标记为需要重新绘制

    def get_position(self):
        """返回脊柱的位置。"""
        self._ensure_position_is_set()  # 确保位置已设置
        return self._position  # 返回脊柱的位置
    def get_spine_transform(self):
        """Return the spine transform."""
        self._ensure_position_is_set()  # 确保位置已设置

        position = self._position  # 获取当前位置

        if isinstance(position, str):
            if position == 'center':
                position = ('axes', 0.5)  # 如果位置为'center'，则设置为 ('axes', 0.5)
            elif position == 'zero':
                position = ('data', 0)  # 如果位置为'zero'，则设置为 ('data', 0)

        assert len(position) == 2, 'position should be 2-tuple'  # 断言位置应为长度为2的元组
        position_type, amount = position  # 将位置信息解包

        _api.check_in_list(['axes', 'outward', 'data'],  # 检查 position_type 是否在允许的列表中
                           position_type=position_type)

        if self.spine_type in ['left', 'right']:
            base_transform = self.axes.get_yaxis_transform(which='grid')  # 获取基础变换
        elif self.spine_type in ['top', 'bottom']:
            base_transform = self.axes.get_xaxis_transform(which='grid')  # 获取基础变换
        else:
            raise ValueError(f'unknown spine spine_type: {self.spine_type!r}')  # 抛出异常，未知的 spine_type

        if position_type == 'outward':
            if amount == 0:  # 如果 amount 为 0，则直接返回基础变换
                return base_transform
            else:
                offset_vec = {'left': (-1, 0), 'right': (1, 0),  # 计算偏移向量
                              'bottom': (0, -1), 'top': (0, 1),
                              }[self.spine_type]
                # 计算以点为单位的 x 和 y 偏移量
                offset_dots = amount * np.array(offset_vec) / 72
                return (base_transform
                        + mtransforms.ScaledTranslation(
                            *offset_dots, self.figure.dpi_scale_trans))  # 返回基础变换加上缩放平移变换

        elif position_type == 'axes':
            if self.spine_type in ['left', 'right']:
                # 保持 y 不变，将 x 固定在 amount 处
                return (mtransforms.Affine2D.from_values(0, 0, 0, 1, amount, 0)
                        + base_transform)  # 返回仿射变换加上基础变换
            elif self.spine_type in ['bottom', 'top']:
                # 保持 x 不变，将 y 固定在 amount 处
                return (mtransforms.Affine2D.from_values(1, 0, 0, 0, 0, amount)
                        + base_transform)  # 返回仿射变换加上基础变换

        elif position_type == 'data':
            if self.spine_type in ('right', 'top'):
                # 右边和顶部 spine 的默认位置在 axes 坐标中为 1。当使用数据坐标指定位置时，需要相对于 0 计算位置。
                amount -= 1  # 如果 spine_type 是 'right' 或 'top'，则将 amount 减去 1
            if self.spine_type in ('left', 'right'):
                # 返回混合变换工厂，结合 x 方向上的平移和数据坐标变换
                return mtransforms.blended_transform_factory(
                    mtransforms.Affine2D().translate(amount, 0)
                    + self.axes.transData,
                    self.axes.transData)
            elif self.spine_type in ('bottom', 'top'):
                # 返回混合变换工厂，结合 y 方向上的平移和数据坐标变换
                return mtransforms.blended_transform_factory(
                    self.axes.transData,
                    mtransforms.Affine2D().translate(0, amount)
                    + self.axes.transData)
    def set_bounds(self, low=None, high=None):
        """
        Set the spine bounds.

        Parameters
        ----------
        low : float or None, optional
            The lower spine bound. Passing *None* leaves the limit unchanged.

            The bounds may also be passed as the tuple (*low*, *high*) as the
            first positional argument.

            .. ACCEPTS: (low: float, high: float)

        high : float or None, optional
            The higher spine bound. Passing *None* leaves the limit unchanged.
        """
        # 如果脊柱类型为圆形，抛出错误，因为圆形脊柱不兼容 set_bounds() 方法
        if self.spine_type == 'circle':
            raise ValueError(
                'set_bounds() method incompatible with circular spines')
        
        # 如果 high 参数为 None 且 low 是可迭代的（tuple 或 list），则解包 low 为 (low, high)
        if high is None and np.iterable(low):
            low, high = low
        
        # 获取当前的脊柱边界，并将其保存在 old_low 和 old_high 中
        old_low, old_high = self.get_bounds() or (None, None)
        
        # 如果 low 为 None，则使用旧的低边界
        if low is None:
            low = old_low
        
        # 如果 high 为 None，则使用旧的高边界
        if high is None:
            high = old_high
        
        # 将新的边界设定为 (low, high)
        self._bounds = (low, high)
        
        # 将 stale 属性设置为 True，表示对象需要更新
        self.stale = True

    def get_bounds(self):
        """Get the bounds of the spine."""
        # 返回当前存储在 _bounds 属性中的边界
        return self._bounds

    @classmethod
    def linear_spine(cls, axes, spine_type, **kwargs):
        """Create and return a linear `Spine`."""
        # 根据脊柱类型创建线性路径，具体路径根据 spine_type 不同而不同
        if spine_type == 'left':
            path = mpath.Path([(0.0, 0.999), (0.0, 0.999)])
        elif spine_type == 'right':
            path = mpath.Path([(1.0, 0.999), (1.0, 0.999)])
        elif spine_type == 'bottom':
            path = mpath.Path([(0.999, 0.0), (0.999, 0.0)])
        elif spine_type == 'top':
            path = mpath.Path([(0.999, 1.0), (0.999, 1.0)])
        else:
            raise ValueError('unable to make path for spine "%s"' % spine_type)
        
        # 使用创建的路径和其他参数创建 Spine 对象 result
        result = cls(axes, spine_type, path, **kwargs)
        
        # 根据 matplotlib 的默认配置，设置脊柱的可见性
        result.set_visible(mpl.rcParams[f'axes.spines.{spine_type}'])

        return result

    @classmethod
    def arc_spine(cls, axes, spine_type, center, radius, theta1, theta2,
                  **kwargs):
        """Create and return an arc `Spine`."""
        # 根据给定的 theta1 和 theta2 创建弧线路径
        path = mpath.Path.arc(theta1, theta2)
        
        # 使用路径和其他参数创建弧线 Spine 对象 result
        result = cls(axes, spine_type, path, **kwargs)
        
        # 设置弧线的 patch 参数，即中心、半径、起始角度和终止角度
        result.set_patch_arc(center, radius, theta1, theta2)
        
        return result

    @classmethod
    def circular_spine(cls, axes, center, radius, **kwargs):
        """Create and return a circular `Spine`."""
        # 创建单位圆的路径
        path = mpath.Path.unit_circle()
        
        # 将脊柱类型设置为 'circle'
        spine_type = 'circle'
        
        # 使用路径和其他参数创建圆形 Spine 对象 result
        result = cls(axes, spine_type, path, **kwargs)
        
        # 设置圆形的 patch 参数，即中心和半径
        result.set_patch_circle(center, radius)
        
        return result
    # 设置边框颜色的方法

    # 使用传入的颜色参数 c 来设置对象的边框颜色
    self.set_edgecolor(c)

    # 将对象的 stale 属性设置为 True，表示对象需要更新
    self.stale = True
class SpinesProxy:
    """
    A proxy to broadcast ``set_*()`` and ``set()`` method calls to contained `.Spines`.

    The proxy cannot be used for any other operations on its members.

    The supported methods are determined dynamically based on the contained
    spines. If not all spines support a given method, it's executed only on
    the subset of spines that support it.
    """

    def __init__(self, spine_dict):
        # 初始化方法，接受一个包含`.Spines`对象的字典作为参数
        self._spine_dict = spine_dict

    def __getattr__(self, name):
        # 获取属性方法，用于动态调用`.Spines`对象的方法

        # 筛选出所有包含指定方法名的`.Spines`对象
        broadcast_targets = [spine for spine in self._spine_dict.values()
                             if hasattr(spine, name)]
        
        # 如果方法名不是以'set_'开头或者不是'set'，或者没有找到对应的`.Spines`对象，则抛出属性错误
        if (name != 'set' and not name.startswith('set_')) or not broadcast_targets:
            raise AttributeError(
                f"'SpinesProxy' object has no attribute '{name}'")

        # 定义一个局部函数x，用于调用指定方法名的方法，并传递相应的参数
        def x(_targets, _funcname, *args, **kwargs):
            for spine in _targets:
                getattr(spine, _funcname)(*args, **kwargs)
        
        # 使用functools.partial将参数_targets和_funcname绑定到局部函数x上
        x = functools.partial(x, broadcast_targets, name)
        # 设置局部函数x的文档字符串为第一个匹配到的`.Spines`对象的文档字符串
        x.__doc__ = broadcast_targets[0].__doc__
        return x

    def __dir__(self):
        # 返回该对象的属性列表，包括所有包含'set_'开头的方法名
        names = []
        for spine in self._spine_dict.values():
            names.extend(name
                         for name in dir(spine) if name.startswith('set_'))
        return list(sorted(set(names)))


class Spines(MutableMapping):
    r"""
    The container of all `.Spine`\s in an Axes.

    The interface is dict-like mapping names (e.g. 'left') to `.Spine` objects.
    Additionally, it implements some pandas.Series-like features like accessing
    elements by attribute::

        spines['top'].set_visible(False)
        spines.top.set_visible(False)

    Multiple spines can be addressed simultaneously by passing a list::

        spines[['top', 'right']].set_visible(False)

    Use an open slice to address all spines::

        spines[:].set_visible(False)

    The latter two indexing methods will return a `SpinesProxy` that broadcasts all
    ``set_*()`` and ``set()`` calls to its members, but cannot be used for any other
    operation.
    """

    def __init__(self, **kwargs):
        # 初始化方法，接受关键字参数，作为`.Spines`对象的字典
        self._dict = kwargs

    @classmethod
    def from_dict(cls, d):
        # 类方法，从字典创建一个`.Spines`对象
        return cls(**d)

    def __getstate__(self):
        # 返回对象的状态信息，即包含的`.Spines`对象字典
        return self._dict

    def __setstate__(self, state):
        # 设置对象的状态信息，重新初始化`.Spines`对象
        self.__init__(**state)

    def __getattr__(self, name):
        # 获取属性方法，用于访问指定名称的`.Spine`对象

        try:
            # 尝试返回指定名称对应的`.Spine`对象
            return self._dict[name]
        except KeyError:
            # 如果名称不存在于字典中，则抛出属性错误
            raise AttributeError(
                f"'Spines' object does not contain a '{name}' spine")
    # 定义特殊方法，用于支持通过索引访问 SpinesProxy 实例
    def __getitem__(self, key):
        # 如果 key 是列表，检查列表中是否存在未知的键
        if isinstance(key, list):
            unknown_keys = [k for k in key if k not in self._dict]
            # 如果存在未知的键，抛出 KeyError 异常
            if unknown_keys:
                raise KeyError(', '.join(unknown_keys))
            # 返回一个新的 SpinesProxy 对象，包含 key 中存在的键值对
            return SpinesProxy({k: v for k, v in self._dict.items()
                                if k in key})
        # 如果 key 是元组，抛出 ValueError 异常，因为多个 spines 必须作为单个列表传递
        if isinstance(key, tuple):
            raise ValueError('Multiple spines must be passed as a single list')
        # 如果 key 是切片对象，检查其是否为全开放切片 [:]，如果是则返回包含所有 spines 的新 SpinesProxy 对象
        if isinstance(key, slice):
            if key.start is None and key.stop is None and key.step is None:
                return SpinesProxy(self._dict)
            else:
                # 否则，抛出 ValueError 异常，因为 Spines 不支持除全开放切片 [:] 外的任何切片操作
                raise ValueError(
                    'Spines does not support slicing except for the fully '
                    'open slice [:] to access all spines.')
        # 对于其他情况，直接返回 self._dict[key]，即返回对应键的值
        return self._dict[key]

    # 定义特殊方法，用于设置 SpinesProxy 实例中的键值对
    def __setitem__(self, key, value):
        # TODO: 是否需要废弃添加 spines 的功能？（待讨论）
        self._dict[key] = value

    # 定义特殊方法，用于删除 SpinesProxy 实例中的键值对
    def __delitem__(self, key):
        # TODO: 是否需要废弃删除 spines 的功能？（待讨论）
        del self._dict[key]

    # 定义特殊方法，使 SpinesProxy 实例可迭代
    def __iter__(self):
        return iter(self._dict)

    # 定义特殊方法，返回 SpinesProxy 实例中键值对的数量
    def __len__(self):
        return len(self._dict)
```