# `D:\src\scipysrc\matplotlib\galleries\examples\misc\custom_projection.py`

```
"""
=================
Custom projection
=================

Showcase Hammer projection by alleviating many features of Matplotlib.
"""

import numpy as np

import matplotlib
from matplotlib.axes import Axes
import matplotlib.axis as maxis
from matplotlib.patches import Circle
from matplotlib.path import Path
from matplotlib.projections import register_projection
import matplotlib.spines as mspines
from matplotlib.ticker import FixedLocator, Formatter, NullLocator
from matplotlib.transforms import Affine2D, BboxTransformTo, Transform

rcParams = matplotlib.rcParams

# This example projection class is rather long, but it is designed to
# illustrate many features, not all of which will be used every time.
# It is also common to factor out a lot of these methods into common
# code used by a number of projections with similar characteristics
# (see geo.py).


class GeoAxes(Axes):
    """
    An abstract base class for geographic projections
    """
    
    class ThetaFormatter(Formatter):
        """
        Used to format the theta tick labels.  Converts the native
        unit of radians into degrees and adds a degree symbol.
        """
        def __init__(self, round_to=1.0):
            self._round_to = round_to

        def __call__(self, x, pos=None):
            degrees = round(np.rad2deg(x) / self._round_to) * self._round_to
            return f"{degrees:0.0f}\N{DEGREE SIGN}"

    RESOLUTION = 75

    def _init_axis(self):
        # Initialize xaxis and yaxis
        self.xaxis = maxis.XAxis(self)
        self.yaxis = maxis.YAxis(self)
        # Do not register xaxis or yaxis with spines -- as done in
        # Axes._init_axis() -- until GeoAxes.xaxis.clear() works.
        # self.spines['geo'].register_axis(self.yaxis)

    def clear(self):
        """
        Clear the current axis.

        This method sets up the geographic projection with default settings:
        - Longitude grid every 30 degrees
        - Latitude grid every 15 degrees
        - Longitude grid extends to 75 degrees
        - Minor ticks are turned off
        - Tick positions are set to none
        - Y-axis tick labels are turned on
        - X-axis tick labels remain on by default

        The axis limits are set to:
        - X-axis: -π to π
        - Y-axis: -π/2 to π/2
        """
        super().clear()

        self.set_longitude_grid(30)
        self.set_latitude_grid(15)
        self.set_longitude_grid_ends(75)
        self.xaxis.set_minor_locator(NullLocator())
        self.yaxis.set_minor_locator(NullLocator())
        self.xaxis.set_ticks_position('none')
        self.yaxis.set_ticks_position('none')
        self.yaxis.set_tick_params(label1On=True)
        # Why do we need to turn on yaxis tick labels, but
        # xaxis tick labels are already on?

        self.grid(rcParams['axes.grid'])

        Axes.set_xlim(self, -np.pi, np.pi)
        Axes.set_ylim(self, -np.pi / 2.0, np.pi / 2.0)

    def _get_affine_transform(self):
        """
        Return the affine transform for the geographic projection.

        This method computes an affine transformation based on the core
        transform, scaling the projection appropriately and translating it.

        Returns:
        -------
        Affine2D
            Affine transformation object for the geographic projection.
        """
        transform = self._get_core_transform(1)
        xscale, _ = transform.transform((np.pi, 0))
        _, yscale = transform.transform((0, np.pi/2))
        return Affine2D() \
            .scale(0.5 / xscale, 0.5 / yscale) \
            .translate(0.5, 0.5)
    def get_xaxis_transform(self, which='grid'):
        """
        Override this method to provide a transformation for the
        x-axis tick labels.

        Returns a tuple of the form (transform, valign, halign)
        """
        # 检查参数 'which' 是否在合法取值范围内，如果不在则抛出异常
        if which not in ['tick1', 'tick2', 'grid']:
            raise ValueError(
                "'which' must be one of 'tick1', 'tick2', or 'grid'")
        # 返回 x 轴标签的变换信息
        return self._xaxis_transform

    def get_xaxis_text1_transform(self, pad):
        # 返回第一组 x 轴文本的变换信息，以及文本垂直对齐方式为底部，水平对齐方式为居中
        return self._xaxis_text1_transform, 'bottom', 'center'

    def get_xaxis_text2_transform(self, pad):
        """
        Override this method to provide a transformation for the
        secondary x-axis tick labels.

        Returns a tuple of the form (transform, valign, halign)
        """
        # 返回第二组 x 轴文本的变换信息，以及文本垂直对齐方式为顶部，水平对齐方式为居中
        return self._xaxis_text2_transform, 'top', 'center'

    def get_yaxis_transform(self, which='grid'):
        """
        Override this method to provide a transformation for the
        y-axis grid and ticks.
        """
        # 检查参数 'which' 是否在合法取值范围内，如果不在则抛出异常
        if which not in ['tick1', 'tick2', 'grid']:
            raise ValueError(
                "'which' must be one of 'tick1', 'tick2', or 'grid'")
        # 返回 y 轴的变换信息
        return self._yaxis_transform

    def get_yaxis_text1_transform(self, pad):
        """
        Override this method to provide a transformation for the
        y-axis tick labels.

        Returns a tuple of the form (transform, valign, halign)
        """
        # 返回第一组 y 轴文本的变换信息，以及文本垂直对齐方式为居中，水平对齐方式为右侧
        return self._yaxis_text1_transform, 'center', 'right'

    def get_yaxis_text2_transform(self, pad):
        """
        Override this method to provide a transformation for the
        secondary y-axis tick labels.

        Returns a tuple of the form (transform, valign, halign)
        """
        # 返回第二组 y 轴文本的变换信息，以及文本垂直对齐方式为居中，水平对齐方式为左侧
        return self._yaxis_text2_transform, 'center', 'left'

    def _gen_axes_patch(self):
        """
        Override this method to define the shape that is used for the
        background of the plot.  It should be a subclass of Patch.

        In this case, it is a Circle (that may be warped by the Axes
        transform into an ellipse).  Any data and gridlines will be
        clipped to this shape.
        """
        # 返回用于绘图背景的形状，这里返回一个圆形的 Patch 对象
        return Circle((0.5, 0.5), 0.5)

    def _gen_axes_spines(self):
        # 返回轴脊的定义，这里返回一个圆形轴脊对象
        return {'geo': mspines.Spine.circular_spine(self, (0.5, 0.5), 0.5)}

    def set_yscale(self, *args, **kwargs):
        # 如果尝试设置非线性的 y 轴比例尺，则抛出未实现的异常
        if args[0] != 'linear':
            raise NotImplementedError

    # 防止用户对 x 轴或者 y 轴进行缩放操作
    # 在这个特定的情况下，缩放轴并没有意义，因此禁止这样的操作。
    set_xscale = set_yscale

    # 防止用户修改坐标轴的限制
    # 在我们的情况下，始终显示整个球体是我们的需求，因此我们重写了 set_xlim 和 set_ylim 以忽略任何输入。
    # 这也适用于 GUI 界面中的交互式平移和缩放。
    def set_xlim(self, *args, **kwargs):
        # 抛出类型错误，因为地理投影不支持更改坐标轴限制
        raise TypeError("Changing axes limits of a geographic projection is "
                        "not supported.  Please consider using Cartopy.")

    set_ylim = set_xlim

    def format_coord(self, lon, lat):
        """
        Override this method to change how the values are displayed in
        the status bar.

        In this case, we want them to be displayed in degrees N/S/E/W.
        """
        # 将经纬度从弧度转换为度数
        lon, lat = np.rad2deg([lon, lat])
        # 根据纬度确定南北方向
        ns = 'N' if lat >= 0.0 else 'S'
        # 根据经度确定东西方向
        ew = 'E' if lon >= 0.0 else 'W'
        # 返回格式化后的经纬度字符串
        return ('%f\N{DEGREE SIGN}%s, %f\N{DEGREE SIGN}%s'
                % (abs(lat), ns, abs(lon), ew))

    def set_longitude_grid(self, degrees):
        """
        Set the number of degrees between each longitude grid.

        This is an example method that is specific to this projection
        class -- it provides a more convenient interface to set the
        ticking than set_xticks would.
        """
        # 生成经度网格线位置，跳过固定的 -180 和 180 度
        grid = np.arange(-180 + degrees, 180, degrees)
        # 设置主要刻度定位器为经度网格线位置
        self.xaxis.set_major_locator(FixedLocator(np.deg2rad(grid)))
        # 使用 ThetaFormatter 格式化经度刻度
        self.xaxis.set_major_formatter(self.ThetaFormatter(degrees))

    def set_latitude_grid(self, degrees):
        """
        Set the number of degrees between each longitude grid.

        This is an example method that is specific to this projection
        class -- it provides a more convenient interface than
        set_yticks would.
        """
        # 生成纬度网格线位置，跳过固定的 -90 和 90 度
        grid = np.arange(-90 + degrees, 90, degrees)
        # 设置主要刻度定位器为纬度网格线位置
        self.yaxis.set_major_locator(FixedLocator(np.deg2rad(grid)))
        # 使用 ThetaFormatter 格式化纬度刻度
        self.yaxis.set_major_formatter(self.ThetaFormatter(degrees))

    def set_longitude_grid_ends(self, degrees):
        """
        Set the latitude(s) at which to stop drawing the longitude grids.

        Often, in geographic projections, you wouldn't want to draw
        longitude gridlines near the poles.  This allows the user to
        specify the degree at which to stop drawing longitude grids.

        This is an example method that is specific to this projection
        class -- it provides an interface to something that has no
        analogy in the base Axes class.
        """
        # 设置停止绘制经度网格线的纬度阈值，转换为弧度
        self._longitude_cap = np.deg2rad(degrees)
        # 预变换 x 轴，以限制经度网格线的绘制范围
        self._xaxis_pretransform \
            .clear() \
            .scale(1.0, self._longitude_cap * 2.0) \
            .translate(0.0, -self._longitude_cap)

    def get_data_ratio(self):
        """
        Return the aspect ratio of the data itself.

        This method should be overridden by any Axes that have a
        fixed data ratio.
        """
        # 返回数据本身的纵横比例，默认为 1.0
        return 1.0

    # Interactive panning and zooming is not supported with this projection,
    # so we override all of the following methods to disable it.
    # 返回当前 Axes 对象是否支持缩放框按钮功能
    def can_zoom(self):
        """
        Return whether this Axes supports the zoom box button functionality.

        This Axes object does not support interactive zoom box.
        """
        return False

    # 返回当前 Axes 对象是否支持平移/缩放按钮功能
    def can_pan(self):
        """
        Return whether this Axes supports the pan/zoom button functionality.

        This Axes object does not support interactive pan/zoom.
        """
        return False

    # 开始进行平移操作，但是此方法在当前的实现中没有具体实现功能
    def start_pan(self, x, y, button):
        pass

    # 结束平移操作，但是此方法在当前的实现中没有具体实现功能
    def end_pan(self):
        pass

    # 进行平移操作时拖动事件的处理，但是此方法在当前的实现中没有具体实现功能
    def drag_pan(self, button, key, x, y):
        pass
# 自定义类 HammerAxes 继承自 GeoAxes，用于 Aitoff-Hammer 投影，一种等面积地图投影。
class HammerAxes(GeoAxes):
    """
    A custom class for the Aitoff-Hammer projection, an equal-area map
    projection.

    https://en.wikipedia.org/wiki/Hammer_projection
    """

    # 投影必须指定一个名称，用户可以通过该名称选择投影，
    # 例如 ``subplot(projection='custom_hammer')``.
    name = 'custom_hammer'

    # 内部类 HammerTransform，继承自 Transform，实现 Hammer 投影的基础转换。
    class HammerTransform(Transform):
        """The base Hammer transform."""
        input_dims = output_dims = 2

        # 初始化 HammerTransform 实例，resolution 是每个输入线段之间插值步数，以逼近其在曲线 Hammer 空间中的路径。
        def __init__(self, resolution):
            """
            Create a new Hammer transform.  Resolution is the number of steps
            to interpolate between each input line segment to approximate its
            path in curved Hammer space.
            """
            Transform.__init__(self)
            self._resolution = resolution

        # 非仿射变换函数，将经度纬度转换为 Hammer 投影下的坐标。
        def transform_non_affine(self, ll):
            longitude, latitude = ll.T

            # 预先计算一些值
            half_long = longitude / 2
            cos_latitude = np.cos(latitude)
            sqrt2 = np.sqrt(2)

            alpha = np.sqrt(1 + cos_latitude * np.cos(half_long))
            x = (2 * sqrt2) * (cos_latitude * np.sin(half_long)) / alpha
            y = (sqrt2 * np.sin(latitude)) / alpha
            return np.column_stack([x, y])

        # 对路径进行非仿射变换，返回转换后的路径。
        def transform_path_non_affine(self, path):
            ipath = path.interpolated(self._resolution)
            return Path(self.transform(ipath.vertices), ipath.codes)

        # 返回反转换对象 InvertedHammerTransform。
        def inverted(self):
            return HammerAxes.InvertedHammerTransform(self._resolution)

    # 内部类 InvertedHammerTransform，继承自 Transform，实现 Hammer 投影的反向转换。
    class InvertedHammerTransform(Transform):
        input_dims = output_dims = 2

        # 初始化 InvertedHammerTransform 实例，resolution 是每个输入线段之间插值步数。
        def __init__(self, resolution):
            Transform.__init__(self)
            self._resolution = resolution

        # 非仿射变换函数，将 Hammer 投影下的坐标转换为经度纬度。
        def transform_non_affine(self, xy):
            x, y = xy.T
            z = np.sqrt(1 - (x / 4) ** 2 - (y / 2) ** 2)
            longitude = 2 * np.arctan((z * x) / (2 * (2 * z ** 2 - 1)))
            latitude = np.arcsin(y*z)
            return np.column_stack([longitude, latitude])

        # 返回反转对象 HammerTransform。
        def inverted(self):
            return HammerAxes.HammerTransform(self._resolution)

    # HammerAxes 类的初始化函数，设置了经度的上限，调用了父类 GeoAxes 的初始化函数，设定了纵横比和清除操作。
    def __init__(self, *args, **kwargs):
        self._longitude_cap = np.pi / 2.0
        super().__init__(*args, **kwargs)
        self.set_aspect(0.5, adjustable='box', anchor='C')
        self.clear()

    # 获取核心转换函数，返回一个 HammerTransform 实例。
    def _get_core_transform(self, resolution):
        return self.HammerTransform(resolution)


# 使用 register_projection 函数注册 HammerAxes 投影，以便用户可以选择它。
register_projection(HammerAxes)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 使用自定义投影制作一个简单的示例。
    fig, ax = plt.subplots(subplot_kw={'projection': 'custom_hammer'})
    ax.plot([-1, 1, 1], [-1, -1, 1], "o-")
    ax.grid()

    plt.show()
```