# `D:\src\scipysrc\matplotlib\lib\matplotlib\projections\polar.py`

```
# 导入数学库
import math
# 导入 types 模块
import types

# 导入 NumPy 库并简称为 np
import numpy as np

# 导入 matplotlib 库并简称为 mpl
import matplotlib as mpl
# 从 matplotlib 中导入 _api 和 cbook 模块
from matplotlib import _api, cbook
# 从 matplotlib.axes 模块导入 Axes 类
from matplotlib.axes import Axes
# 导入 matplotlib.axis 模块并简称为 maxis
import matplotlib.axis as maxis
# 导入 matplotlib.markers 模块并简称为 mmarkers
import matplotlib.markers as mmarkers
# 导入 matplotlib.patches 模块并简称为 mpatches
import matplotlib.patches as mpatches
# 从 matplotlib.path 模块导入 Path 类
from matplotlib.path import Path
# 导入 matplotlib.ticker 模块并简称为 mticker
import matplotlib.ticker as mticker
# 导入 matplotlib.transforms 模块并简称为 mtransforms
import matplotlib.transforms as mtransforms
# 从 matplotlib.spines 模块导入 Spine 类
from matplotlib.spines import Spine


# 定义一个私有函数 _apply_theta_transforms_warn，用于发出关于过时功能的警告
def _apply_theta_transforms_warn():
    _api.warn_deprecated(
                "3.9",
                message=(
                    "Passing `apply_theta_transforms=True` (the default) "
                    "is deprecated since Matplotlib %(since)s. "
                    "Support for this will be removed in Matplotlib %(removal)s. "
                    "To prevent this warning, set `apply_theta_transforms=False`, "
                    "and make sure to shift theta values before being passed to "
                    "this transform."
                )
            )


# 定义一个极坐标变换类 PolarTransform，继承自 mtransforms.Transform 类
class PolarTransform(mtransforms.Transform):
    r"""
    The base polar transform.

    This transform maps polar coordinates :math:`\theta, r` into Cartesian
    coordinates :math:`x, y = r \cos(\theta), r \sin(\theta)`
    (but does not fully transform into Axes coordinates or
    handle positioning in screen space).

    This transformation is designed to be applied to data after any scaling
    along the radial axis (e.g. log-scaling) has been applied to the input
    data.

    Path segments at a fixed radius are automatically transformed to circular
    arcs as long as ``path._interpolation_steps > 1``.
    """

    # 输入维度和输出维度都是 2
    input_dims = output_dims = 2

    # 构造函数，初始化极坐标变换的参数
    def __init__(self, axis=None, use_rmin=True, *,
                 apply_theta_transforms=True, scale_transform=None):
        """
        Parameters
        ----------
        axis : `~matplotlib.axis.Axis`, optional
            Axis associated with this transform. This is used to get the
            minimum radial limit.
        use_rmin : `bool`, optional
            If ``True``, subtract the minimum radial axis limit before
            transforming to Cartesian coordinates. *axis* must also be
            specified for this to take effect.
        """
        # 调用父类的构造函数
        super().__init__()
        # 关联的坐标轴对象
        self._axis = axis
        # 是否使用最小径向坐标轴限制
        self._use_rmin = use_rmin
        # 是否应用 theta 变换，默认为 True
        self._apply_theta_transforms = apply_theta_transforms
        # 缩放变换对象
        self._scale_transform = scale_transform
        # 如果应用 theta 变换，则发出警告
        if apply_theta_transforms:
            _apply_theta_transforms_warn()

    # 返回对象的字符串表示形式，包括关联的坐标轴、是否使用最小径向坐标轴限制、是否应用 theta 变换的信息
    __str__ = mtransforms._make_str_method(
        "_axis",
        use_rmin="_use_rmin",
        apply_theta_transforms="_apply_theta_transforms")

    # 获取原点的 r 值
    def _get_rorigin(self):
        # 获取经过径向缩放变换后的较低 r 限制
        return self._scale_transform.transform(
            (0, self._axis.get_rorigin()))[1]

    # 装饰器函数，重命名参数为 values，用于向后兼容
    @_api.rename_parameter("3.8", "tr", "values")
    # 继承的文档字符串说明
    theta, r = np.transpose(values)
    # 从输入的二维数组中解包出极坐标角度和半径信息
    # 如果要应用极坐标的 theta 转换，并且当前坐标轴对象存在
    if self._apply_theta_transforms and self._axis is not None:
        # 根据坐标轴的方向调整 theta 值
        theta *= self._axis.get_theta_direction()
        # 加上坐标轴的 theta 偏移量
        theta += self._axis.get_theta_offset()
    # 如果使用 rmin 并且当前坐标轴对象存在
    if self._use_rmin and self._axis is not None:
        # 调整 r 值，使得原点偏移并乘以半径的符号
        r = (r - self._get_rorigin()) * self._axis.get_rsign()
    # 将所有小于零的 r 值设为 NaN
    r = np.where(r >= 0, r, np.nan)
    # 返回转换后的笛卡尔坐标点数组，根据极坐标转换公式计算
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    def transform_path_non_affine(self, path):
        # docstring inherited
        # 如果路径为空或者插值步数为1，则返回非仿射变换后的路径
        if not len(path) or path._interpolation_steps == 1:
            return Path(self.transform_non_affine(path.vertices), path.codes)
        
        xys = []  # 存储变换后的顶点坐标
        codes = []  # 存储路径代码
        
        last_t = last_r = None  # 上一个角度和半径的初始值为 None
        
        # 遍历路径的每个线段和对应的代码
        for trs, c in path.iter_segments():
            trs = trs.reshape((-1, 2))  # 将路径线段的顶点重新排列为二维数组
            
            if c == Path.LINETO:  # 如果是直线段
                (t, r), = trs  # 取出角度和半径
                if t == last_t:  # 如果角度相同，画一条直线
                    xys.extend(self.transform_non_affine(trs))  # 将变换后的顶点添加到 xys 中
                    codes.append(Path.LINETO)  # 添加直线代码
                elif r == last_r:  # 如果半径相同，画一个弧线
                    # 下面的操作复杂，因为 Path.arc() 会自动展开角度，但这里不需要这种行为
                    last_td, td = np.rad2deg([last_t, t])  # 将角度转换为度数
                    if self._use_rmin and self._axis is not None:
                        r = ((r - self._get_rorigin())
                             * self._axis.get_rsign())  # 计算半径的调整
                    
                    if last_td <= td:
                        while td - last_td > 360:
                            # 绘制多个完整的圆弧
                            arc = Path.arc(last_td, last_td + 360)
                            xys.extend(arc.vertices[1:] * r)  # 添加变换后的顶点
                            codes.extend(arc.codes[1:])  # 添加路径代码
                            last_td += 360
                        arc = Path.arc(last_td, td)
                        xys.extend(arc.vertices[1:] * r)
                        codes.extend(arc.codes[1:])
                    else:
                        while last_td - td > 360:
                            # 绘制反向的多个圆弧
                            arc = Path.arc(last_td - 360, last_td)
                            xys.extend(arc.vertices[::-1][1:] * r)
                            codes.extend(arc.codes[1:])
                            last_td -= 360
                        arc = Path.arc(td, last_td)
                        xys.extend(arc.vertices[::-1][1:] * r)
                        codes.extend(arc.codes[1:])
                else:
                    # 插值
                    trs = cbook.simple_linear_interpolation(
                        np.vstack([(last_t, last_r), trs]),
                        path._interpolation_steps)[1:]  # 简单线性插值
                    xys.extend(self.transform_non_affine(trs))
                    codes.extend([Path.LINETO] * len(trs))  # 添加多个直线代码
            else:
                # 非直线段，直接进行非仿射变换
                xys.extend(self.transform_non_affine(trs))
                codes.extend([c] * len(trs))  # 添加多个相同的路径代码
            
            last_t, last_r = trs[-1]  # 更新最后一个角度和半径的值
        
        return Path(xys, codes)  # 返回变换后的路径对象

    def inverted(self):
        # docstring inherited
        return PolarAxes.InvertedPolarTransform(
            self._axis,  # 极坐标轴对象
            self._use_rmin,  # 使用最小半径标志
            apply_theta_transforms=self._apply_theta_transforms  # 应用角度变换的标志
        )
class PolarAffine(mtransforms.Affine2DBase):
    r"""
    The affine part of the polar projection.

    Scales the output so that maximum radius rests on the edge of the Axes
    circle and the origin is mapped to (0.5, 0.5). The transform applied is
    the same to x and y components and given by:

    .. math::

        x_{1} = 0.5 \left [ \frac{x_{0}}{(r_{\max} - r_{\min})} + 1 \right ]

    :math:`r_{\min}, r_{\max}` are the minimum and maximum radial limits after
    any scaling (e.g. log scaling) has been removed.
    """
    def __init__(self, scale_transform, limits):
        """
        Parameters
        ----------
        scale_transform : `~matplotlib.transforms.Transform`
            Scaling transform for the data. This is used to remove any scaling
            from the radial view limits.
        limits : `~matplotlib.transforms.BboxBase`
            View limits of the data. The only part of its bounds that is used
            is the y limits (for the radius limits).
        """
        super().__init__()
        # 设置缩放变换和视图限制
        self._scale_transform = scale_transform
        self._limits = limits
        # 将缩放变换和视图限制设置为子元素
        self.set_children(scale_transform, limits)
        self._mtx = None

    __str__ = mtransforms._make_str_method("_scale_transform", "_limits")

    def get_matrix(self):
        # docstring inherited
        # 如果当前变换无效，则重新计算变换矩阵
        if self._invalid:
            # 对视图限制应用缩放变换
            limits_scaled = self._limits.transformed(self._scale_transform)
            # 计算y轴方向的缩放比例
            yscale = limits_scaled.ymax - limits_scaled.ymin
            # 创建仿射变换对象，使得最大半径映射到Axes边缘，原点映射到(0.5, 0.5)
            affine = mtransforms.Affine2D() \
                .scale(0.5 / yscale) \
                .translate(0.5, 0.5)
            # 获取变换矩阵
            self._mtx = affine.get_matrix()
            self._inverted = None
            self._invalid = 0
        # 返回计算得到的变换矩阵
        return self._mtx


class InvertedPolarTransform(mtransforms.Transform):
    """
    The inverse of the polar transform, mapping Cartesian
    coordinate space *x* and *y* back to *theta* and *r*.
    """
    input_dims = output_dims = 2

    def __init__(self, axis=None, use_rmin=True,
                 *, apply_theta_transforms=True):
        """
        Parameters
        ----------
        axis : `~matplotlib.axis.Axis`, optional
            Axis associated with this transform. This is used to get the
            minimum radial limit.
        use_rmin : `bool`, optional
            If ``True``, add the minimum radial axis limit after
            transforming from Cartesian coordinates. *axis* must also be
            specified for this to take effect.
        """
        super().__init__()
        # 关联的坐标轴和是否使用最小径向限制
        self._axis = axis
        self._use_rmin = use_rmin
        self._apply_theta_transforms = apply_theta_transforms
        # 如果应用theta转换，则发出警告
        if apply_theta_transforms:
            _apply_theta_transforms_warn()

    __str__ = mtransforms._make_str_method(
        "_axis",
        use_rmin="_use_rmin",
        apply_theta_transforms="_apply_theta_transforms")

    @_api.rename_parameter("3.8", "xy", "values")
    # 定义一个方法用于转换非仿射变换的坐标值
    def transform_non_affine(self, values):
        # 继承的文档字符串，说明该方法是从父类继承而来的
        x, y = values.T
        # 计算点到原点的距离
        r = np.hypot(x, y)
        # 计算点的极角（弧度），并确保角度在 [0, 2π) 范围内
        theta = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)
        
        # 如果需要应用极角的转换，并且存在坐标轴对象，则进行以下操作以保持向后兼容性
        if self._apply_theta_transforms and self._axis is not None:
            # 减去坐标轴的极角偏移量
            theta -= self._axis.get_theta_offset()
            # 根据坐标轴的极角方向进行调整
            theta *= self._axis.get_theta_direction()
            # 确保角度在 [0, 2π) 范围内
            theta %= 2 * np.pi
        
        # 如果需要使用最小半径，并且存在坐标轴对象，则进行以下操作
        if self._use_rmin and self._axis is not None:
            # 增加到原点距离的偏移量
            r += self._axis.get_rorigin()
            # 根据坐标轴的半径符号进行调整
            r *= self._axis.get_rsign()
        
        # 返回极坐标下的坐标值，将极角和距离组合成一个二维数组
        return np.column_stack([theta, r])

    # 定义一个方法用于返回反转的极坐标变换对象
    def inverted(self):
        # 继承的文档字符串，说明该方法是从父类继承而来
        return PolarAxes.PolarTransform(
            self._axis,  # 使用的坐标轴对象
            self._use_rmin,  # 是否使用最小半径
            apply_theta_transforms=self._apply_theta_transforms  # 是否应用极角变换
        )
# 自定义 Formatter 类，用于格式化 theta 轴的刻度标签
class ThetaFormatter(mticker.Formatter):
    """
    Used to format the *theta* tick labels.  Converts the native
    unit of radians into degrees and adds a degree symbol.
    """

    # 实现 __call__ 方法，用于格式化标签
    def __call__(self, x, pos=None):
        # 获取当前轴的视图区间
        vmin, vmax = self.axis.get_view_interval()
        # 计算视图区间的角度差，并转换为角度
        d = np.rad2deg(abs(vmax - vmin))
        # 确定标签的小数位数，使其合适显示
        digits = max(-int(np.log10(d) - 1.5), 0)
        # 返回格式化后的标签，包括角度值和度数符号
        return f"{np.rad2deg(x):0.{digits}f}\N{DEGREE SIGN}"


# 辅助类 _AxisWrapper，用于包装轴对象
class _AxisWrapper:
    def __init__(self, axis):
        self._axis = axis

    # 获取视图区间的角度值
    def get_view_interval(self):
        return np.rad2deg(self._axis.get_view_interval())

    # 设置视图区间，转换为弧度值后设置给轴对象
    def set_view_interval(self, vmin, vmax):
        self._axis.set_view_interval(*np.deg2rad((vmin, vmax)))

    # 获取最小正数值的角度
    def get_minpos(self):
        return np.rad2deg(self._axis.get_minpos())

    # 获取数据区间的角度值
    def get_data_interval(self):
        return np.rad2deg(self._axis.get_data_interval())

    # 设置数据区间，转换为弧度值后设置给轴对象
    def set_data_interval(self, vmin, vmax):
        self._axis.set_data_interval(*np.deg2rad((vmin, vmax)))

    # 获取刻度间隔
    def get_tick_space(self):
        return self._axis.get_tick_space()


# 自定义 Locator 类，用于定位 theta 轴的刻度
class ThetaLocator(mticker.Locator):
    """
    Used to locate theta ticks.

    This will work the same as the base locator except in the case that the
    view spans the entire circle. In such cases, the previously used default
    locations of every 45 degrees are returned.
    """

    def __init__(self, base):
        self.base = base
        # 使用 _AxisWrapper 包装基础 Locator 的轴对象
        self.axis = self.base.axis = _AxisWrapper(self.base.axis)

    # 设置轴对象
    def set_axis(self, axis):
        self.axis = _AxisWrapper(axis)
        self.base.set_axis(self.axis)

    # 实现 __call__ 方法，用于返回刻度位置数组
    def __call__(self):
        # 获取当前视图区间
        lim = self.axis.get_view_interval()
        # 如果视图跨越整个圆，返回每 45 度一个的默认位置数组
        if _is_full_circle_deg(lim[0], lim[1]):
            return np.deg2rad(min(lim)) + np.arange(8) * 2 * np.pi / 8
        else:
            # 否则返回基础 Locator 的结果
            return np.deg2rad(self.base())

    # 调整视图限制范围并返回结果
    def view_limits(self, vmin, vmax):
        vmin, vmax = np.rad2deg((vmin, vmax))
        return np.deg2rad(self.base.view_limits(vmin, vmax))


# 自定义 Tick 类，继承自 maxis.XTick
class ThetaTick(maxis.XTick):
    """
    A theta-axis tick.

    This subclass of `.XTick` provides angular ticks with some small
    modification to their re-positioning such that ticks are rotated based on
    tick location. This results in ticks that are correctly perpendicular to
    the arc spine.

    When 'auto' rotation is enabled, labels are also rotated to be parallel to
    the spine. The label padding is also applied here since it's not possible
    to use a generic axes transform to produce tick-specific padding.
    """
    # 初始化方法，用于设置两个文本标签的平移变换
    def __init__(self, axes, *args, **kwargs):
        # 创建第一个文本标签的平移变换对象，初始化为原点，dpi_scale_trans为图像 DPI 缩放变换
        self._text1_translate = mtransforms.ScaledTranslation(
            0, 0, axes.figure.dpi_scale_trans)
        # 创建第二个文本标签的平移变换对象，初始化为原点，dpi_scale_trans为图像 DPI 缩放变换
        self._text2_translate = mtransforms.ScaledTranslation(
            0, 0, axes.figure.dpi_scale_trans)
        # 调用父类构造方法，初始化对象
        super().__init__(axes, *args, **kwargs)
        # 设置第一个文本标签的属性：旋转模式为锚点，应用平移变换
        self.label1.set(
            rotation_mode='anchor',
            transform=self.label1.get_transform() + self._text1_translate)
        # 设置第二个文本标签的属性：旋转模式为锚点，应用平移变换
        self.label2.set(
            rotation_mode='anchor',
            transform=self.label2.get_transform() + self._text2_translate)

    # 应用参数的私有方法，确保文本标签的变换正确
    def _apply_params(self, **kwargs):
        # 调用父类方法，应用参数
        super()._apply_params(**kwargs)
        # 获取第一个文本标签的变换对象
        trans = self.label1.get_transform()
        # 如果当前变换不包含_text1_translate变换，则添加该变换
        if not trans.contains_branch(self._text1_translate):
            self.label1.set_transform(trans + self._text1_translate)
        # 获取第二个文本标签的变换对象
        trans = self.label2.get_transform()
        # 如果当前变换不包含_text2_translate变换，则添加该变换
        if not trans.contains_branch(self._text2_translate):
            self.label2.set_transform(trans + self._text2_translate)

    # 更新填充值的私有方法，根据角度调整两个文本标签的平移变换
    def _update_padding(self, pad, angle):
        # 计算填充在x和y方向上的偏移量，以72为单位
        padx = pad * np.cos(angle) / 72
        pady = pad * np.sin(angle) / 72
        # 更新第一个文本标签的平移变换偏移量
        self._text1_translate._t = (padx, pady)
        self._text1_translate.invalidate()  # 使平移变换无效，以便后续重新计算
        # 更新第二个文本标签的平移变换偏移量
        self._text2_translate._t = (-padx, -pady)
        self._text2_translate.invalidate()  # 使平移变换无效，以便后续重新计算
    # 调用父类方法更新位置信息
    def update_position(self, loc):
        super().update_position(loc)
        # 获取极坐标系对象
        axes = self.axes
        # 计算标签角度，考虑坐标系的角度偏移和方向
        angle = loc * axes.get_theta_direction() + axes.get_theta_offset()
        # 将角度转换为文本标签的角度，并转换为0到360度之间
        text_angle = np.rad2deg(angle) % 360 - 90
        # 将角度调整为合适的旋转角度
        angle -= np.pi / 2

        # 获取第一个刻度线的标记类型
        marker = self.tick1line.get_marker()
        # 根据标记类型选择适当的变换方式
        if marker in (mmarkers.TICKUP, '|'):
            trans = mtransforms.Affine2D().scale(1, 1).rotate(angle)
        elif marker == mmarkers.TICKDOWN:
            trans = mtransforms.Affine2D().scale(1, -1).rotate(angle)
        else:
            # 不修改自定义刻度线标记
            trans = self.tick1line._marker._transform
        # 应用变换到第一个刻度线的标记对象
        self.tick1line._marker._transform = trans

        # 获取第二个刻度线的标记类型
        marker = self.tick2line.get_marker()
        # 根据标记类型选择适当的变换方式
        if marker in (mmarkers.TICKUP, '|'):
            trans = mtransforms.Affine2D().scale(1, 1).rotate(angle)
        elif marker == mmarkers.TICKDOWN:
            trans = mtransforms.Affine2D().scale(1, -1).rotate(angle)
        else:
            # 不修改自定义刻度线标记
            trans = self.tick2line._marker._transform
        # 应用变换到第二个刻度线的标记对象
        self.tick2line._marker._transform = trans

        # 获取标签旋转模式和用户定义的角度
        mode, user_angle = self._labelrotation
        # 根据旋转模式设置文本标签的角度
        if mode == 'default':
            text_angle = user_angle
        else:
            if text_angle > 90:
                text_angle -= 180
            elif text_angle < -90:
                text_angle += 180
            text_angle += user_angle
        # 设置第一个和第二个标签的旋转角度
        self.label1.set_rotation(text_angle)
        self.label2.set_rotation(text_angle)

        # 添加额外的内边距以保持与之前版本的外观一致，同时因为标签锚定在其中心点，所以这也是必需的。
        pad = self._pad + 7
        # 更新内边距和位置
        self._update_padding(pad,
                             self._loc * axes.get_theta_direction() +
                             axes.get_theta_offset())
    """
    A theta Axis.

    This overrides certain properties of an `.XAxis` to provide special-casing
    for an angular axis.
    """
    # 类名称
    __name__ = 'thetaaxis'
    # 轴的名称
    axis_name = 'theta'  #: Read-only name identifying the axis.
    # 刻度使用的类
    _tick_class = ThetaTick

    def _wrap_locator_formatter(self):
        # 设置主要定位器为 ThetaLocator 类的实例
        self.set_major_locator(ThetaLocator(self.get_major_locator()))
        # 设置主要格式化器为 ThetaFormatter 类的实例
        self.set_major_formatter(ThetaFormatter())
        # 标记为默认主定位器和主格式化器
        self.isDefault_majloc = True
        self.isDefault_majfmt = True

    def clear(self):
        # 清除继承的文档字符串
        super().clear()
        # 设置刻度位置为 'none'
        self.set_ticks_position('none')
        # 调用 _wrap_locator_formatter 方法
        self._wrap_locator_formatter()

    def _set_scale(self, value, **kwargs):
        # 如果 value 不是 'linear'，则抛出未实现错误
        if value != 'linear':
            raise NotImplementedError(
                "The xscale cannot be set on a polar plot")
        # 调用父类的 _set_scale 方法
        super()._set_scale(value, **kwargs)
        # 设置主要定位器的参数，使其在合理的角度倍数上显示刻度
        self.get_major_locator().set_params(steps=[1, 1.5, 3, 4.5, 9, 10])
        # 调用 _wrap_locator_formatter 方法
        self._wrap_locator_formatter()

    def _copy_tick_props(self, src, dest):
        """Copy the props from src tick to dest tick."""
        # 如果 src 或 dest 为 None，则返回
        if src is None or dest is None:
            return
        # 调用父类的 _copy_tick_props 方法，复制属性
        super()._copy_tick_props(src, dest)

        # 确保刻度转换是独立的，以使填充有效
        trans = dest._get_text1_transform()[0]
        dest.label1.set_transform(trans + dest._text1_translate)
        trans = dest._get_text2_transform()[0]
        dest.label2.set_transform(trans + dest._text2_translate)
    # 继承的文档字符串描述
    def nonsingular(self, vmin, vmax):
        # 检查是否存在零界限，并且视图范围为 (-inf, inf) 时
        if self._zero_in_bounds() and (vmin, vmax) == (-np.inf, np.inf):
            # 返回默认的非奇异视图范围 (0, 1)
            return (0, 1)
        else:
            # 调用基类的 nonsingular 方法处理视图范围
            return self.base.nonsingular(vmin, vmax)

    def view_limits(self, vmin, vmax):
        # 调用基类的 view_limits 方法获取处理后的视图范围
        vmin, vmax = self.base.view_limits(vmin, vmax)
        # 如果存在零界限，并且 vmax 大于 vmin
        if self._zero_in_bounds() and vmax > vmin:
            # 如果允许反向的 r/y-lims，则将 vmin 调整为最小值与 0 的较小值
            vmin = min(0, vmin)
        # 返回处理后的视图范围
        return mtransforms.nonsingular(vmin, vmax)
class _ThetaShift(mtransforms.ScaledTranslation):
    """
    Apply a padding shift based on axes theta limits.

    This is used to create padding for radial ticks.

    Parameters
    ----------
    axes : `~matplotlib.axes.Axes`
        The owning Axes; used to determine limits.
    pad : float
        The padding to apply, in points.
    mode : {'min', 'max', 'rlabel'}
        Whether to shift away from the start (``'min'``) or the end (``'max'``)
        of the axes, or using the rlabel position (``'rlabel'``).
    """
    def __init__(self, axes, pad, mode):
        # 调用父类构造函数，初始化平移对象
        super().__init__(pad, pad, axes.figure.dpi_scale_trans)
        # 设置平移对象的子对象为 axes._realViewLim
        self.set_children(axes._realViewLim)
        # 保存传入的参数
        self.axes = axes
        self.mode = mode
        self.pad = pad

    # 定义 __str__ 方法，用于返回对象描述的字符串表示
    __str__ = mtransforms._make_str_method("axes", "pad", "mode")

    def get_matrix(self):
        # 如果对象无效，则重新计算平移矩阵
        if self._invalid:
            if self.mode == 'rlabel':
                # 根据 'rlabel' 模式计算角度
                angle = (
                    np.deg2rad(self.axes.get_rlabel_position()
                               * self.axes.get_theta_direction())
                    + self.axes.get_theta_offset()
                    - np.pi / 2
                )
            elif self.mode == 'min':
                # 根据 'min' 模式计算角度
                angle = self.axes._realViewLim.xmin - np.pi / 2
            elif self.mode == 'max':
                # 根据 'max' 模式计算角度
                angle = self.axes._realViewLim.xmax + np.pi / 2
            # 计算平移矩阵，以点为单位，考虑角度和 padding
            self._t = (self.pad * np.cos(angle) / 72, self.pad * np.sin(angle) / 72)
        # 返回计算得到的平移矩阵
        return super().get_matrix()


class RadialTick(maxis.YTick):
    """
    A radial-axis tick.

    This subclass of `.YTick` provides radial ticks with some small
    modification to their re-positioning such that ticks are rotated based on
    axes limits.  This results in ticks that are correctly perpendicular to
    the spine. Labels are also rotated to be perpendicular to the spine, when
    'auto' rotation is enabled.
    """

    def __init__(self, *args, **kwargs):
        # 调用父类构造函数初始化 radial-axis tick 对象
        super().__init__(*args, **kwargs)
        # 设置 label1 和 label2 的旋转模式为 'anchor'
        self.label1.set_rotation_mode('anchor')
        self.label2.set_rotation_mode('anchor')
    def _determine_anchor(self, mode, angle, start):
        # 如果 mode 参数为 'auto'，根据 start 参数和 angle 角度确定锚点位置
        # 注意：angle 是 (spine angle - 90)，因为它用于刻度和文本设置，所以下面的数字都是从 (标准化的) 脊柱角度减去 90 后的值。
        if mode == 'auto':
            # 如果 start 为 True
            if start:
                # 根据 angle 的范围确定水平和垂直方向上的锚点位置
                if -90 <= angle <= 90:
                    return 'left', 'center'  # 左对齐，垂直居中
                else:
                    return 'right', 'center'  # 右对齐，垂直居中
            else:
                # 如果 start 为 False
                if -90 <= angle <= 90:
                    return 'right', 'center'  # 右对齐，垂直居中
                else:
                    return 'left', 'center'  # 左对齐，垂直居中
        else:
            # 如果 mode 参数不为 'auto'
            if start:
                # 根据 angle 的范围确定水平和垂直方向上的锚点位置
                if angle < -68.5:
                    return 'center', 'top'  # 水平居中，顶部对齐
                elif angle < -23.5:
                    return 'left', 'top'  # 左对齐，顶部对齐
                elif angle < 22.5:
                    return 'left', 'center'  # 左对齐，垂直居中
                elif angle < 67.5:
                    return 'left', 'bottom'  # 左对齐，底部对齐
                elif angle < 112.5:
                    return 'center', 'bottom'  # 水平居中，底部对齐
                elif angle < 157.5:
                    return 'right', 'bottom'  # 右对齐，底部对齐
                elif angle < 202.5:
                    return 'right', 'center'  # 右对齐，垂直居中
                elif angle < 247.5:
                    return 'right', 'top'  # 右对齐，顶部对齐
                else:
                    return 'center', 'top'  # 水平居中，顶部对齐
            else:
                # 如果 start 为 False
                if angle < -68.5:
                    return 'center', 'bottom'  # 水平居中，底部对齐
                elif angle < -23.5:
                    return 'right', 'bottom'  # 右对齐，底部对齐
                elif angle < 22.5:
                    return 'right', 'center'  # 右对齐，垂直居中
                elif angle < 67.5:
                    return 'right', 'top'  # 右对齐，顶部对齐
                elif angle < 112.5:
                    return 'center', 'top'  # 水平居中，顶部对齐
                elif angle < 157.5:
                    return 'left', 'top'  # 左对齐，顶部对齐
                elif angle < 202.5:
                    return 'left', 'center'  # 左对齐，垂直居中
                elif angle < 247.5:
                    return 'left', 'bottom'  # 左对齐，底部对齐
                else:
                    return 'center', 'bottom'  # 水平居中，底部对齐
class RadialAxis(maxis.YAxis):
    """
    A radial Axis.

    This overrides certain properties of a `.YAxis` to provide special-casing
    for a radial axis.
    """
    # 设置类的名称为 'radialaxis'
    __name__ = 'radialaxis'
    # 设置轴的名称为 'radius'，只读属性
    axis_name = 'radius'  #: Read-only name identifying the axis.
    # 使用 RadialTick 类作为 ticks 的类别
    _tick_class = RadialTick

    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
        # 将 0 添加到 y 轴的 sticky_edges 中
        self.sticky_edges.y.append(0)

    def _wrap_locator_formatter(self):
        # 设置主刻度定位器为 RadialLocator，并将当前 axes 传递给它
        self.set_major_locator(RadialLocator(self.get_major_locator(),
                                             self.axes))
        # 将 isDefault_majloc 属性设置为 True
        self.isDefault_majloc = True

    def clear(self):
        # 清空方法，继承自父类
        super().clear()
        # 设置 ticks 的位置为 'none'
        self.set_ticks_position('none')
        # 调用 _wrap_locator_formatter 方法
        self._wrap_locator_formatter()

    def _set_scale(self, value, **kwargs):
        # 设置刻度尺度，调用父类的 _set_scale 方法
        super()._set_scale(value, **kwargs)
        # 调用 _wrap_locator_formatter 方法
        self._wrap_locator_formatter()


def _is_full_circle_deg(thetamin, thetamax):
    """
    Determine if a wedge (in degrees) spans the full circle.

    The condition is derived from :class:`~matplotlib.patches.Wedge`.
    """
    # 判断角度制的楔形是否完整覆盖了圆周
    return abs(abs(thetamax - thetamin) - 360.0) < 1e-12


def _is_full_circle_rad(thetamin, thetamax):
    """
    Determine if a wedge (in radians) spans the full circle.

    The condition is derived from :class:`~matplotlib.patches.Wedge`.
    """
    # 判断弧度制的楔形是否完整覆盖了圆周
    return abs(abs(thetamax - thetamin) - 2 * np.pi) < 1.74e-14


class _WedgeBbox(mtransforms.Bbox):
    """
    Transform (theta, r) wedge Bbox into Axes bounding box.

    Parameters
    ----------
    center : (float, float)
        Center of the wedge
    viewLim : `~matplotlib.transforms.Bbox`
        Bbox determining the boundaries of the wedge
    originLim : `~matplotlib.transforms.Bbox`
        Bbox determining the origin for the wedge, if different from *viewLim*
    """
    def __init__(self, center, viewLim, originLim, **kwargs):
        # 调用父类 mtransforms.Bbox 的初始化方法
        super().__init__([[0, 0], [1, 1]], **kwargs)
        # 设置中心点属性
        self._center = center
        # 设置视图限制属性
        self._viewLim = viewLim
        # 设置原点限制属性
        self._originLim = originLim
        # 设置子对象为 viewLim 和 originLim
        self.set_children(viewLim, originLim)

    # 定义字符串表示方法，展示中心点、视图限制和原点限制
    __str__ = mtransforms._make_str_method("_center", "_viewLim", "_originLim")
    def get_points(self):
        # 继承的文档字符串描述函数作用
        if self._invalid:
            # 如果标记为无效，复制视图极限的点集合
            points = self._viewLim.get_points().copy()
            # 将角度限制缩放以适应Wedge对象
            points[:, 0] *= 180 / np.pi
            # 如果第一个角度大于第二个角度，反转点集合中的角度限制
            if points[0, 0] > points[1, 0]:
                points[:, 0] = points[::-1, 0]

            # 根据原点半径调整径向限制
            points[:, 1] -= self._originLim.y0

            # 缩放径向限制以匹配坐标轴限制
            rscale = 0.5 / points[1, 1]
            points[:, 1] *= rscale
            # 计算楔形图的宽度，取最小值
            width = min(points[1, 1] - points[0, 1], 0.5)

            # 生成楔形图的边界框
            wedge = mpatches.Wedge(self._center, points[1, 1],
                                   points[0, 0], points[1, 0],
                                   width=width)
            # 更新当前对象以适应楔形图路径
            self.update_from_path(wedge.get_path())

            # 确保等比例缩放
            w, h = self._points[1] - self._points[0]
            deltah = max(w - h, 0) / 2
            deltaw = max(h - w, 0) / 2
            self._points += np.array([[-deltaw, -deltah], [deltaw, deltah]])

            # 标记为有效
            self._invalid = 0

        # 返回点集合
        return self._points
class PolarAxes(Axes):
    """
    A polar graph projection, where the input dimensions are *theta*, *r*.

    Theta starts pointing east and goes anti-clockwise.
    """
    # 类名，表示这是一个极坐标图形投影类
    name = 'polar'

    def __init__(self, *args,
                 theta_offset=0, theta_direction=1, rlabel_position=22.5,
                 **kwargs):
        # 构造函数，初始化极坐标图形的参数和属性
        # 设置默认的 theta 偏移量、方向和 r 轴标签位置
        self._default_theta_offset = theta_offset
        self._default_theta_direction = theta_direction
        self._default_rlabel_position = np.deg2rad(rlabel_position)
        # 调用父类的构造函数初始化
        super().__init__(*args, **kwargs)
        # 使用粘性边缘
        self.use_sticky_edges = True
        # 设置纵横比为 'equal'，可调整为盒状，锚点为 'C'
        self.set_aspect('equal', adjustable='box', anchor='C')
        # 清除图形
        self.clear()

    def clear(self):
        # 清空图形
        super().clear()

        # 设置标题位置在顶部
        self.title.set_y(1.05)

        # 获取和隐藏起始和结束的脊柱
        start = self.spines.get('start', None)
        if start:
            start.set_visible(False)
        end = self.spines.get('end', None)
        if end:
            end.set_visible(False)
        
        # 设置 X 轴的限制在 0 到 2π 之间
        self.set_xlim(0.0, 2 * np.pi)

        # 根据 matplotlib 默认参数设置极坐标网格
        self.grid(mpl.rcParams['polaraxes.grid'])
        
        # 获取和隐藏内部脊柱
        inner = self.spines.get('inner', None)
        if inner:
            inner.set_visible(False)
        
        # 设置极径坐标原点
        self.set_rorigin(None)
        # 设置 theta 偏移量为默认值
        self.set_theta_offset(self._default_theta_offset)
        # 设置 theta 方向为默认值
        self.set_theta_direction(self._default_theta_direction)

    def _init_axis(self):
        # 初始化轴
        # 这被移出 __init__ 函数，因为非可分离轴不使用它
        self.xaxis = ThetaAxis(self, clear=False)
        self.yaxis = RadialAxis(self, clear=False)
        self.spines['polar'].register_axis(self.yaxis)

    def get_xaxis_transform(self, which='grid'):
        # 获取 X 轴变换方法，可选的包括 'tick1', 'tick2', 'grid'
        _api.check_in_list(['tick1', 'tick2', 'grid'], which=which)
        return self._xaxis_transform

    def get_xaxis_text1_transform(self, pad):
        # 获取 X 轴文本1的变换方法和对齐方式
        return self._xaxis_text_transform, 'center', 'center'

    def get_xaxis_text2_transform(self, pad):
        # 获取 X 轴文本2的变换方法和对齐方式
        return self._xaxis_text_transform, 'center', 'center'

    def get_yaxis_transform(self, which='grid'):
        # 获取 Y 轴变换方法，可选的包括 'tick1', 'tick2', 'grid'
        if which in ('tick1', 'tick2'):
            return self._yaxis_text_transform
        elif which == 'grid':
            return self._yaxis_transform
        else:
            _api.check_in_list(['tick1', 'tick2', 'grid'], which=which)

    def get_yaxis_text1_transform(self, pad):
        # 获取 Y 轴文本1的变换方法和对齐方式
        thetamin, thetamax = self._realViewLim.intervalx
        # 检查是否是完整的圆形弧度
        if _is_full_circle_rad(thetamin, thetamax):
            return self._yaxis_text_transform, 'bottom', 'left'
        elif self.get_theta_direction() > 0:
            halign = 'left'
            pad_shift = _ThetaShift(self, pad, 'min')
        else:
            halign = 'right'
            pad_shift = _ThetaShift(self, pad, 'max')
        return self._yaxis_text_transform + pad_shift, 'center', halign
    # 根据当前角度方向确定垂直于极坐标轴的文本水平对齐方式
    def get_yaxis_text2_transform(self, pad):
        if self.get_theta_direction() > 0:
            halign = 'right'  # 如果角度方向大于0，则水平对齐方式为右对齐
            pad_shift = _ThetaShift(self, pad, 'max')  # 获取根据最大角度偏移后的文本位置
        else:
            halign = 'left'  # 如果角度方向不大于0，则水平对齐方式为左对齐
            pad_shift = _ThetaShift(self, pad, 'min')  # 获取根据最小角度偏移后的文本位置
        # 返回垂直于极坐标轴的文本变换以及文本垂直对齐方式为居中，水平对齐方式
        return self._yaxis_text_transform + pad_shift, 'center', halign

    # 绘制极坐标图形
    def draw(self, renderer):
        self._unstale_viewLim()  # 更新视图界限
        thetamin, thetamax = np.rad2deg(self._realViewLim.intervalx)  # 将视图限制的角度转换为度数
        if thetamin > thetamax:
            thetamin, thetamax = thetamax, thetamin  # 确保角度范围正确（thetamin <= thetamax）
        rmin, rmax = ((self._realViewLim.intervaly - self.get_rorigin()) *
                      self.get_rsign())  # 计算极径范围

        if isinstance(self.patch, mpatches.Wedge):
            # 向后兼容性：任何子类化的Axes可能会覆盖patch，不一定是PolarAxes使用的Wedge。
            center = self.transWedge.transform((0.5, 0.5))  # 将中心点转换为画布坐标系
            self.patch.set_center(center)  # 设置Wedge的中心点
            self.patch.set_theta1(thetamin)  # 设置Wedge的起始角度
            self.patch.set_theta2(thetamax)  # 设置Wedge的终止角度

            edge, _ = self.transWedge.transform((1, 0))  # 将边缘点转换为画布坐标系
            radius = edge - center[0]  # 计算半径
            width = min(radius * (rmax - rmin) / rmax, radius)  # 计算宽度
            self.patch.set_radius(radius)  # 设置Wedge的半径
            self.patch.set_width(width)  # 设置Wedge的宽度

            inner_width = radius - width  # 计算内部宽度
            inner = self.spines.get('inner', None)  # 获取内边框
            if inner:
                inner.set_visible(inner_width != 0.0)  # 根据内部宽度设置内边框可见性

        visible = not _is_full_circle_deg(thetamin, thetamax)  # 检查角度范围是否为整圆
        # 向后兼容性：任何子类化的Axes可能会覆盖spines，不一定包含PolarAxes使用的start/end。
        start = self.spines.get('start', None)  # 获取起始边框
        end = self.spines.get('end', None)  # 获取结束边框
        if start:
            start.set_visible(visible)  # 设置起始边框的可见性
        if end:
            end.set_visible(visible)  # 设置结束边框的可见性

        if visible:
            yaxis_text_transform = self._yaxis_transform  # 如果角度范围不为整圆，则文本变换为_yaxis_transform
        else:
            yaxis_text_transform = self._r_label_position + self.transData  # 如果角度范围为整圆，则文本变换为_r_label_position + transData

        if self._yaxis_text_transform != yaxis_text_transform:
            self._yaxis_text_transform.set(yaxis_text_transform)  # 设置文本变换
            self.yaxis.reset_ticks()  # 重置刻度
            self.yaxis.set_clip_path(self.patch)  # 设置刻度的裁剪路径为patch

        super().draw(renderer)  # 调用父类的绘制方法

    # 生成Axes的图形补丁
    def _gen_axes_patch(self):
        return mpatches.Wedge((0.5, 0.5), 0.5, 0.0, 360.0)  # 创建一个Wedge补丁对象，表示一个完整的圆形

    # 生成Axes的边框
    def _gen_axes_spines(self):
        # 创建Axes的边框字典
        spines = {
            'polar': Spine.arc_spine(self, 'top', (0.5, 0.5), 0.5, 0, 360),  # 极坐标顶部的弧形边框
            'start': Spine.linear_spine(self, 'left'),  # 左侧的线性边框
            'end': Spine.linear_spine(self, 'right'),  # 右侧的线性边框
            'inner': Spine.arc_spine(self, 'bottom', (0.5, 0.5), 0.0, 0, 360),  # 极坐标底部的内部弧形边框
        }
        spines['polar'].set_transform(self.transWedge + self.transAxes)  # 设置极坐标顶部边框的变换
        spines['inner'].set_transform(self.transWedge + self.transAxes)  # 设置极坐标底部内部边框的变换
        spines['start'].set_transform(self._yaxis_transform)  # 设置左侧边框的变换
        spines['end'].set_transform(self._yaxis_transform)  # 设置右侧边框的变换
        return spines  # 返回边框字典
    # 设置最大角度限制（单位为度）
    def set_thetamax(self, thetamax):
        """Set the maximum theta limit in degrees."""
        self.viewLim.x1 = np.deg2rad(thetamax)

    # 获取最大角度限制（单位为度）
    def get_thetamax(self):
        """Return the maximum theta limit in degrees."""
        return np.rad2deg(self.viewLim.xmax)

    # 设置最小角度限制（单位为度）
    def set_thetamin(self, thetamin):
        """Set the minimum theta limit in degrees."""
        self.viewLim.x0 = np.deg2rad(thetamin)

    # 获取最小角度限制（单位为度）
    def get_thetamin(self):
        """Get the minimum theta limit in degrees."""
        return np.rad2deg(self.viewLim.xmin)

    # 设置角度范围的上下限
    def set_thetalim(self, *args, **kwargs):
        r"""
        Set the minimum and maximum theta values.

        Can take the following signatures:

        - ``set_thetalim(minval, maxval)``: Set the limits in radians.
        - ``set_thetalim(thetamin=minval, thetamax=maxval)``: Set the limits
          in degrees.

        where minval and maxval are the minimum and maximum limits. Values are
        wrapped in to the range :math:`[0, 2\pi]` (in radians), so for example
        it is possible to do ``set_thetalim(-np.pi / 2, np.pi / 2)`` to have
        an axis symmetric around 0. A ValueError is raised if the absolute
        angle difference is larger than a full circle.
        """
        orig_lim = self.get_xlim()  # in radians
        if 'thetamin' in kwargs:
            kwargs['xmin'] = np.deg2rad(kwargs.pop('thetamin'))
        if 'thetamax' in kwargs:
            kwargs['xmax'] = np.deg2rad(kwargs.pop('thetamax'))
        new_min, new_max = self.set_xlim(*args, **kwargs)
        # Parsing all permutations of *args, **kwargs is tricky; it is simpler
        # to let set_xlim() do it and then validate the limits.
        if abs(new_max - new_min) > 2 * np.pi:
            self.set_xlim(orig_lim)  # un-accept the change
            raise ValueError("The angle range must be less than a full circle")
        return tuple(np.rad2deg((new_min, new_max)))

    # 设置角度偏移量（单位为弧度）
    def set_theta_offset(self, offset):
        """
        Set the offset for the location of 0 in radians.
        """
        mtx = self._theta_offset.get_matrix()
        mtx[0, 2] = offset
        self._theta_offset.invalidate()

    # 获取角度偏移量（单位为弧度）
    def get_theta_offset(self):
        """
        Get the offset for the location of 0 in radians.
        """
        return self._theta_offset.get_matrix()[0, 2]
    def set_theta_zero_location(self, loc, offset=0.0):
        """
        Set the location of theta's zero.

        This method sets the initial angle (theta=0) for a plot, based on a given location
        and an optional offset in degrees.

        Parameters
        ----------
        loc : str
            Specifies the cardinal direction for theta=0. Allowed values are "N", "NW", "W",
            "SW", "S", "SE", "E", or "NE".
        offset : float, default: 0
            Optional offset in degrees from the specified loc. The offset is applied
            counter-clockwise.

        Returns
        -------
        None
        """
        mapping = {
            'N': np.pi * 0.5,
            'NW': np.pi * 0.75,
            'W': np.pi,
            'SW': np.pi * 1.25,
            'S': np.pi * 1.5,
            'SE': np.pi * 1.75,
            'E': 0,
            'NE': np.pi * 0.25}
        # Calls another method to set theta offset based on the mapped angle and offset in radians
        return self.set_theta_offset(mapping[loc] + np.deg2rad(offset))

    def set_theta_direction(self, direction):
        """
        Set the direction in which theta increases.

        This method sets the direction in which the angular coordinate theta increases
        relative to the plot.

        Parameters
        ----------
        direction : str or int
            Specifies the direction of theta increment. Values can be:
            - 'clockwise' or -1: Indicates theta increases clockwise.
            - 'counterclockwise', 'anticlockwise', or 1: Indicates theta increases
              counterclockwise.

        Returns
        -------
        None
        """
        mtx = self._direction.get_matrix()
        if direction in ('clockwise', -1):
            mtx[0, 0] = -1
        elif direction in ('counterclockwise', 'anticlockwise', 1):
            mtx[0, 0] = 1
        else:
            # Validates the input direction against allowed values
            _api.check_in_list(
                [-1, 1, 'clockwise', 'counterclockwise', 'anticlockwise'],
                direction=direction)
        # Invalidates cached data related to direction
        self._direction.invalidate()

    def get_theta_direction(self):
        """
        Get the direction in which theta increases.

        Returns
        -------
        int
            Returns -1 if theta increases clockwise, and 1 if theta increases
            counterclockwise.
        """
        return self._direction.get_matrix()[0, 0]

    def set_rmax(self, rmax):
        """
        Set the outer radial limit.

        This method sets the maximum radial limit (outer boundary) for the plot.

        Parameters
        ----------
        rmax : float
            The value to set as the outer radial limit.

        Returns
        -------
        None
        """
        self.viewLim.y1 = rmax

    def get_rmax(self):
        """
        Returns
        -------
        float
            Returns the current outer radial limit.
        """
        return self.viewLim.ymax

    def set_rmin(self, rmin):
        """
        Set the inner radial limit.

        This method sets the minimum radial limit (inner boundary) for the plot.

        Parameters
        ----------
        rmin : float
            The value to set as the inner radial limit.

        Returns
        -------
        None
        """
        self.viewLim.y0 = rmin

    def get_rmin(self):
        """
        Returns
        -------
        float
            Returns the current inner radial limit.
        """
        return self.viewLim.ymin

    def set_rorigin(self, rorigin):
        """
        Update the radial origin.

        This method updates the radial origin, which is the starting point for radial
        coordinates in the plot.

        Parameters
        ----------
        rorigin : float
            The value to set as the radial origin.

        Returns
        -------
        None
        """
        self._originViewLim.locked_y0 = rorigin

    def get_rorigin(self):
        """
        Returns
        -------
        float
            Returns the current radial origin.
        """
        return self._originViewLim.y0
    # 获取极径轴的符号，即视图限制的y1和y0之差的符号
    def get_rsign(self):
        return np.sign(self._originViewLim.y1 - self._originViewLim.y0)

    # 设置极径轴的视图限制
    def set_rlim(self, bottom=None, top=None, *,
                 emit=True, auto=False, **kwargs):
        """
        设置极径轴的视图限制。

        此函数类似于 `.Axes.set_ylim`，但额外支持 *rmin* 和 *rmax* 作为 *bottom* 和 *top* 的别名。

        参见
        --------
        .Axes.set_ylim
        """
        # 如果在kwargs中存在'rmin'，则处理bottom参数
        if 'rmin' in kwargs:
            if bottom is None:
                bottom = kwargs.pop('rmin')
            else:
                raise ValueError('Cannot supply both positional "bottom"'
                                 'argument and kwarg "rmin"')
        # 如果在kwargs中存在'rmax'，则处理top参数
        if 'rmax' in kwargs:
            if top is None:
                top = kwargs.pop('rmax')
            else:
                raise ValueError('Cannot supply both positional "top"'
                                 'argument and kwarg "rmax"')
        # 调用基类方法设置y轴限制
        return self.set_ylim(bottom=bottom, top=top, emit=emit, auto=auto,
                             **kwargs)

    # 获取极径标签位置的角度值
    def get_rlabel_position(self):
        """
        返回
        -------
        float
            极径标签在角度上的位置。
        """
        return np.rad2deg(self._r_label_position.get_matrix()[0, 2])

    # 设置极径标签位置的角度值
    def set_rlabel_position(self, value):
        """
        更新极径标签的角度位置。

        参数
        ----------
        value : number
            极径标签在角度上的位置。
        """
        self._r_label_position.clear().translate(np.deg2rad(value), 0.0)

    # 设置极径轴的纵向缩放
    def set_yscale(self, *args, **kwargs):
        super().set_yscale(*args, **kwargs)
        # 设置主要刻度定位器为RadialLocator，将当前轴和y轴定位器作为参数传递
        self.yaxis.set_major_locator(
            self.RadialLocator(self.yaxis.get_major_locator(), self))

    # 设置极径轴的纵向缩放（别名）
    def set_rscale(self, *args, **kwargs):
        return Axes.set_yscale(self, *args, **kwargs)

    # 设置极径轴的刻度线位置
    def set_rticks(self, *args, **kwargs):
        return Axes.set_yticks(self, *args, **kwargs)
    def set_thetagrids(self, angles, labels=None, fmt=None, **kwargs):
        """
        Set the theta gridlines in a polar plot.

        Parameters
        ----------
        angles : tuple with floats, degrees
            The angles of the theta gridlines.

        labels : tuple with strings or None
            The labels to use at each theta gridline. The
            `.projections.polar.ThetaFormatter` will be used if None.

        fmt : str or None
            Format string used in `matplotlib.ticker.FormatStrFormatter`.
            For example '%f'. Note that the angle that is used is in
            radians.

        Returns
        -------
        lines : list of `.lines.Line2D`
            The theta gridlines.

        labels : list of `.text.Text`
            The tick labels.

        Other Parameters
        ----------------
        **kwargs
            *kwargs* are optional `.Text` properties for the labels.

            .. warning::

                This only sets the properties of the current ticks.
                Ticks are not guaranteed to be persistent. Various operations
                can create, delete and modify the Tick instances. There is an
                imminent risk that these settings can get lost if you work on
                the figure further (including also panning/zooming on a
                displayed figure).

                Use `.set_tick_params` instead if possible.

        See Also
        --------
        .PolarAxes.set_rgrids
        .Axis.get_gridlines
        .Axis.get_ticklabels
        """

        # 确保考虑到数据的单位化
        angles = self.convert_yunits(angles)  # 调用对象的方法，将角度转换为与数据单位匹配的值
        angles = np.deg2rad(angles)  # 将角度转换为弧度
        self.set_xticks(angles)  # 设置极坐标图中的 X 轴刻度位置为给定的角度
        if labels is not None:
            self.set_xticklabels(labels)  # 如果有标签参数，设置 X 轴刻度的标签
        elif fmt is not None:
            self.xaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))  # 如果有格式化字符串参数，使用给定的格式化器设置 X 轴主要刻度的格式
        for t in self.xaxis.get_ticklabels():
            t._internal_update(kwargs)  # 更新 X 轴刻度标签的属性
        return self.xaxis.get_ticklines(), self.xaxis.get_ticklabels()  # 返回 X 轴刻度的网格线和刻度标签
    def set_rgrids(self, radii, labels=None, angle=None, fmt=None, **kwargs):
        """
        在极坐标图上设置径向网格线。

        Parameters
        ----------
        radii : tuple with floats
            径向网格线的半径值。

        labels : tuple with strings or None
            每个径向网格线上使用的标签。如果为 None，则使用 `matplotlib.ticker.ScalarFormatter`。

        angle : float
            半径标签的角度位置，单位为度。

        fmt : str or None
            在 `matplotlib.ticker.FormatStrFormatter` 中使用的格式字符串，例如 '%f'。

        Returns
        -------
        lines : list of `.lines.Line2D`
            径向网格线的列表。

        labels : list of `.text.Text`
            刻度标签的列表。

        Other Parameters
        ----------------
        **kwargs
            标签的可选 `.Text` 属性。

            .. warning::

                仅设置当前刻度的属性。刻度不保证持久存在。各种操作可能会创建、删除和修改刻度实例。如果进一步操作图形（包括显示图形后的平移/缩放），这些设置可能会丢失。

                如果可能，请改用 `.set_tick_params`。

        See Also
        --------
        .PolarAxes.set_thetagrids
        .Axis.get_gridlines
        .Axis.get_ticklabels
        """
        # 确保考虑单位化的数据
        radii = self.convert_xunits(radii)
        radii = np.asarray(radii)

        # 设置径向刻度
        self.set_yticks(radii)
        if labels is not None:
            # 设置径向刻度标签
            self.set_yticklabels(labels)
        elif fmt is not None:
            # 使用指定格式设置主要刻度标签的格式
            self.yaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))
        if angle is None:
            # 获取默认的半径标签位置
            angle = self.get_rlabel_position()
        # 设置半径标签的位置
        self.set_rlabel_position(angle)

        # 更新刻度标签的内部属性
        for t in self.yaxis.get_ticklabels():
            t._internal_update(kwargs)
        
        # 返回径向网格线和刻度标签
        return self.yaxis.get_gridlines(), self.yaxis.get_ticklabels()
    def format_coord(self, theta, r):
        # docstring inherited
        # 将极坐标 (theta, r) 转换为屏幕坐标
        screen_xy = self.transData.transform((theta, r))
        # 构建一个3x3的网格，用于计算邻近点的屏幕坐标
        screen_xys = screen_xy + np.stack(
            np.meshgrid([-1, 0, 1], [-1, 0, 1])).reshape((2, -1)).T
        # 将屏幕坐标转换回极坐标 (theta, r) 的数组
        ts, rs = self.transData.inverted().transform(screen_xys).T
        # 计算角度方向上的最大误差
        delta_t = abs((ts - theta + np.pi) % (2 * np.pi) - np.pi).max()
        # 将角度误差转换为半圈数
        delta_t_halfturns = delta_t / np.pi
        # 将角度误差转换为度数
        delta_t_degrees = delta_t_halfturns * 180
        # 计算径向上的最大误差
        delta_r = abs(rs - r).max()
        # 若 theta 小于 0，则加上 2π，确保角度在正常范围内
        if theta < 0:
            theta += 2 * np.pi
        # 将角度转换为半圈数
        theta_halfturns = theta / np.pi
        # 将角度转换为度数
        theta_degrees = theta_halfturns * 180

        # 格式化 r 的显示，使用 #g 格式化
        def format_sig(value, delta, opt, fmt):
            # 对于 "f" 格式，只计算小数点后的位数
            prec = (max(0, -math.floor(math.log10(delta))) if fmt == "f" else
                    cbook._g_sig_digits(value, delta))
            return f"{value:-{opt}.{prec}{fmt}}"

        # 若未指定 fmt_ydata，则使用默认格式化方式
        if self.fmt_ydata is None:
            # 格式化 r 的标签显示
            r_label = format_sig(r, delta_r, "#", "g")
        else:
            r_label = self.format_ydata(r)

        # 若未指定 fmt_xdata，则格式化 theta 和 r 的显示
        if self.fmt_xdata is None:
            return ('\N{GREEK SMALL LETTER THETA}={}\N{GREEK SMALL LETTER PI} '
                    '({}\N{DEGREE SIGN}), r={}').format(
                    format_sig(theta_halfturns, delta_t_halfturns, "", "f"),
                    format_sig(theta_degrees, delta_t_degrees, "", "f"),
                    r_label
                )
        else:
            # 使用指定的格式化方式格式化 theta 和 r 的显示
            return '\N{GREEK SMALL LETTER THETA}={}, r={}'.format(
                        self.format_xdata(theta),
                        r_label
                        )

    def get_data_ratio(self):
        """
        Return the aspect ratio of the data itself.  For a polar plot,
        this should always be 1.0
        """
        # 返回数据本身的纵横比。对于极坐标图，始终返回 1.0
        return 1.0

    # # # Interactive panning

    def can_zoom(self):
        """
        Return whether this Axes supports the zoom box button functionality.

        A polar Axes does not support zoom boxes.
        """
        # 返回此 Axes 是否支持缩放框按钮功能。
        # 极坐标图不支持缩放框。
        return False

    def can_pan(self):
        """
        Return whether this Axes supports the pan/zoom button functionality.

        For a polar Axes, this is slightly misleading. Both panning and
        zooming are performed by the same button. Panning is performed
        in azimuth while zooming is done along the radial.
        """
        # 返回此 Axes 是否支持平移/缩放按钮功能。
        # 对于极坐标图，这有点误导性。平移和缩放都由同一个按钮执行。
        # 平移是在方位角上进行的，而缩放是沿径向进行的。
        return True
    # 定义一个方法 start_pan，用于处理开始拖动操作
    def start_pan(self, x, y, button):
        # 获取极坐标标签的角度并转换为弧度制
        angle = np.deg2rad(self.get_rlabel_position())
        # 初始化模式为空字符串
        mode = ''
        # 如果按下鼠标左键（button == 1）
        if button == 1:
            # 设置角度偏差值
            epsilon = np.pi / 45.0
            # 转换传入的坐标 (x, y) 到极坐标中的 (t, r)
            t, r = self.transData.inverted().transform((x, y))
            # 如果鼠标位置在当前极坐标标签的角度范围内
            if angle - epsilon <= t <= angle + epsilon:
                # 设置模式为 'drag_r_labels'，表示拖动极坐标标签
                mode = 'drag_r_labels'
        # 如果按下鼠标右键（button == 3）
        elif button == 3:
            # 设置模式为 'zoom'，表示缩放操作
            mode = 'zoom'

        # 存储当前拖动操作的起始状态和参数
        self._pan_start = types.SimpleNamespace(
            rmax=self.get_rmax(),  # 获取当前的最大半径
            trans=self.transData.frozen(),  # 获取当前的数据变换对象
            trans_inverse=self.transData.inverted().frozen(),  # 获取当前的反向数据变换对象
            r_label_angle=self.get_rlabel_position(),  # 获取当前的极坐标标签角度
            x=x,  # 记录起始鼠标位置的 x 坐标
            y=y,  # 记录起始鼠标位置的 y 坐标
            mode=mode  # 记录操作模式（拖动标签或缩放）
        )

    # 定义一个方法 end_pan，用于结束拖动操作
    def end_pan(self):
        # 删除存储的拖动操作起始状态和参数
        del self._pan_start

    # 定义一个方法 drag_pan，用于处理拖动操作中的具体行为
    def drag_pan(self, button, key, x, y):
        # 获取拖动操作的起始状态和参数
        p = self._pan_start

        # 如果当前操作模式为 'drag_r_labels'
        if p.mode == 'drag_r_labels':
            # 计算起始点 (p.x, p.y) 和当前点 (x, y) 在反向数据变换下的坐标
            (startt, startr), (t, r) = p.trans_inverse.transform(
                [(p.x, p.y), (x, y)])

            # 处理角度变化
            dt = np.rad2deg(startt - t)
            self.set_rlabel_position(p.r_label_angle - dt)  # 调整极坐标标签的角度

            # 更新纵轴文本的显示位置和水平对齐方式
            trans, vert1, horiz1 = self.get_yaxis_text1_transform(0.0)
            trans, vert2, horiz2 = self.get_yaxis_text2_transform(0.0)
            for t in self.yaxis.majorTicks + self.yaxis.minorTicks:
                t.label1.set_va(vert1)
                t.label1.set_ha(horiz1)
                t.label2.set_va(vert2)
                t.label2.set_ha(horiz2)

        # 如果当前操作模式为 'zoom'
        elif p.mode == 'zoom':
            # 计算起始点 (p.x, p.y) 和当前点 (x, y) 在反向数据变换下的坐标
            (startt, startr), (t, r) = p.trans_inverse.transform(
                [(p.x, p.y), (x, y)])

            # 处理半径变化
            scale = r / startr
            self.set_rmax(p.rmax / scale)  # 调整极径的最大值
# 将上面定义的 Polar 类别名赋值给 PolarAxes 类的相应属性。
# 这并非绝对必要，但它能提升代码可读性，并提供向后兼容的 Polar API。
# 特别地，在 :doc:`/gallery/specialty_plots/radar_chart` 示例中，会用到这些别名来覆盖 PolarAxes 子类上的 PolarTransform，
# 确保在更改这些别名之前，该示例不受影响。
# 将 PolarAxes.PolarTransform 设置为 PolarTransform 类别名
PolarAxes.PolarTransform = PolarTransform
# 将 PolarAxes.PolarAffine 设置为 PolarAffine 类别名
PolarAxes.PolarAffine = PolarAffine
# 将 PolarAxes.InvertedPolarTransform 设置为 InvertedPolarTransform 类别名
PolarAxes.InvertedPolarTransform = InvertedPolarTransform
# 将 PolarAxes.ThetaFormatter 设置为 ThetaFormatter 类别名
PolarAxes.ThetaFormatter = ThetaFormatter
# 将 PolarAxes.RadialLocator 设置为 RadialLocator 类别名
PolarAxes.RadialLocator = RadialLocator
# 将 PolarAxes.ThetaLocator 设置为 ThetaLocator 类别名
PolarAxes.ThetaLocator = ThetaLocator
```