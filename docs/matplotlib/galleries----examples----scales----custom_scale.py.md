# `D:\src\scipysrc\matplotlib\galleries\examples\scales\custom_scale.py`

```
"""
.. _custom_scale:

============
Custom scale
============

Create a custom scale, by implementing the scaling use for latitude data in a
Mercator Projection.

Unless you are making special use of the `.Transform` class, you probably
don't need to use this verbose method, and instead can use `~.scale.FuncScale`
and the ``'function'`` option of `~.Axes.set_xscale` and `~.Axes.set_yscale`.
See the last example in :doc:`/gallery/scales/scales`.
"""

# 导入必要的库
import numpy as np  # 导入 numpy 库，并简写为 np
from numpy import ma  # 导入 numpy 中的 ma 模块

from matplotlib import scale as mscale  # 从 matplotlib 中导入 scale 模块，并简写为 mscale
from matplotlib import transforms as mtransforms  # 导入 matplotlib 中的 transforms 模块，并简写为 mtransforms
from matplotlib.ticker import FixedLocator, FuncFormatter  # 从 matplotlib.ticker 中导入 FixedLocator 和 FuncFormatter 类


class MercatorLatitudeScale(mscale.ScaleBase):
    """
    Scales data in range -pi/2 to pi/2 (-90 to 90 degrees) using
    the system used to scale latitudes in a Mercator__ projection.

    The scale function:
      ln(tan(y) + sec(y))

    The inverse scale function:
      atan(sinh(y))

    Since the Mercator scale tends to infinity at +/- 90 degrees,
    there is user-defined threshold, above and below which nothing
    will be plotted.  This defaults to +/- 85 degrees.

    __ https://en.wikipedia.org/wiki/Mercator_projection
    """

    # The scale class must have a member ``name`` that defines the string used
    # to select the scale.  For example, ``ax.set_yscale("mercator")`` would be
    # used to select this scale.
    name = 'mercator'  # 定义标识该比例尺的名称为 'mercator'

    def __init__(self, axis, *, thresh=np.deg2rad(85), **kwargs):
        """
        Any keyword arguments passed to ``set_xscale`` and ``set_yscale`` will
        be passed along to the scale's constructor.

        thresh: The degree above which to crop the data.
        """
        super().__init__(axis)  # 调用父类的构造方法初始化
        if thresh >= np.pi / 2:
            raise ValueError("thresh must be less than pi/2")
        self.thresh = thresh  # 初始化阈值属性

    def get_transform(self):
        """
        Override this method to return a new instance that does the
        actual transformation of the data.

        The MercatorLatitudeTransform class is defined below as a
        nested class of this one.
        """
        return self.MercatorLatitudeTransform(self.thresh)  # 返回 MercatorLatitudeTransform 类的实例

    class MercatorLatitudeTransform(mtransforms.Transform):
        """
        The transform for Mercator latitude data.
        """

        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh

        def transform_non_affine(self, a):
            """
            This transform takes an Nx1 ``numpy.ndarray`` and returns a transformed
            version.  Since the range of Mercator latitude is limited by the
            user-specified threshold, the input array should be masked before
            applying this transform.
            """
            return np.log(np.tan(a / 2 + np.pi / 4))

        def inverted(self):
            return MercatorLatitudeScale.InvertedMercatorLatitudeTransform(self.thresh)

        class InvertedMercatorLatitudeTransform(mtransforms.Transform):
            input_dims = 1
            output_dims = 1
            is_separable = True

            def __init__(self, thresh):
                mtransforms.Transform.__init__(self)
                self.thresh = thresh

            def transform_non_affine(self, a):
                return 2 * np.arctan(np.exp(a)) - np.pi / 2

            def inverted(self):
                return MercatorLatitudeScale.MercatorLatitudeTransform(self.thresh)

# 注释结束
    def set_default_locators_and_formatters(self, axis):
        """
        Override to set up the locators and formatters to use with the
        scale.  This is only required if the scale requires custom
        locators and formatters.  Writing custom locators and
        formatters is rather outside the scope of this example, but
        there are many helpful examples in :mod:`.ticker`.

        In our case, the Mercator example uses a fixed locator from -90 to 90
        degrees and a custom formatter to convert the radians to degrees and
        put a degree symbol after the value.
        """
        # 创建一个自定义格式化器，将弧度转换为度，并在值后加上度符号
        fmt = FuncFormatter(
            lambda x, pos=None: f"{np.degrees(x):.0f}\N{DEGREE SIGN}")
        # 设置主要刻度的定位器为固定的从 -90 到 90 度的弧度范围
        axis.set(major_locator=FixedLocator(np.radians(range(-90, 90, 10))),
                 # 设置主要刻度的格式化器为上面创建的自定义格式化器
                 major_formatter=fmt, 
                 # 设置次要刻度的格式化器为上面创建的自定义格式化器（与主要刻度共享同一格式化器）
                 minor_formatter=fmt)

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Override to limit the bounds of the axis to the domain of the
        transform.  In the case of Mercator, the bounds should be
        limited to the threshold that was passed in.  Unlike the
        autoscaling provided by the tick locators, this range limiting
        will always be adhered to, whether the axis range is set
        manually, determined automatically or changed through panning
        and zooming.
        """
        # 将轴的范围限制在变换的域范围内
        return max(vmin, -self.thresh), min(vmax, self.thresh)
    class MercatorLatitudeTransform(mtransforms.Transform):
        # 定义了一个自定义的坐标变换类 MercatorLatitudeTransform，继承自 mtransforms.Transform
        # input_dims 和 output_dims 分别指定了输入和输出的维度，对于比例尺变换，这些应该总是设为 1
        input_dims = output_dims = 1
    
        def __init__(self, thresh):
            # 构造函数，初始化 MercatorLatitudeTransform 对象
            mtransforms.Transform.__init__(self)
            self.thresh = thresh  # 保存用户传入的阈值参数
    
        def transform_non_affine(self, a):
            """
            这个方法接受一个 numpy 数组作为输入，并返回一个经过变换的副本。
            由于 Mercator 比例尺的范围受用户指定的阈值限制，输入数组必须被掩码以包含只有有效值。
            Matplotlib 将处理掩码数组，并从绘图中删除超出范围的数据。
            但是返回的数组必须与输入数组具有相同的形状，因为这些值需要与另一维度中的值保持同步。
            """
            masked = ma.masked_where((a < -self.thresh) | (a > self.thresh), a)  # 创建掩码数组
            if masked.mask.any():
                return ma.log(np.abs(ma.tan(masked) + 1 / ma.cos(masked)))  # 返回处理后的掩码数组
            else:
                return np.log(np.abs(np.tan(a) + 1 / np.cos(a)))  # 返回处理后的数组
    
        def inverted(self):
            """
            重写此方法以便 Matplotlib 知道如何获取此变换的反向变换。
            """
            return MercatorLatitudeScale.InvertedMercatorLatitudeTransform(self.thresh)
    
    class InvertedMercatorLatitudeTransform(mtransforms.Transform):
        input_dims = output_dims = 1
    
        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh
    
        def transform_non_affine(self, a):
            """
            这个方法接受一个 numpy 数组作为输入，并返回一个经过变换的副本。
            """
            return np.arctan(np.sinh(a))  # 返回经过反转处理的数组
    
        def inverted(self):
            """
            返回 MercatorLatitudeScale.MercatorLatitudeTransform 的实例，使用当前的阈值参数。
            """
            return MercatorLatitudeScale.MercatorLatitudeTransform(self.thresh)
# 导入所需的 Matplotlib 库
import matplotlib.pyplot as plt

# 创建一个数组 t，包含从 -180.0 到 180.0 的数值，步长为 0.1
t = np.arange(-180.0, 180.0, 0.1)
# 将 t 中的数值转换为弧度，并除以 2，存储在数组 s 中
s = np.radians(t) / 2.

# 绘制 t 和 s 的图形，使用实线 ('-')，线宽为 2
plt.plot(t, s, '-', lw=2)
# 设置 y 轴的缩放为 Mercator 投影
plt.yscale('mercator')

# 设置 x 轴标签为 'Longitude'
plt.xlabel('Longitude')
# 设置 y 轴标签为 'Latitude'
plt.ylabel('Latitude')
# 设置图形标题为 'Mercator projection'
plt.title('Mercator projection')
# 打开网格显示
plt.grid(True)

# 显示绘制的图形
plt.show()
```