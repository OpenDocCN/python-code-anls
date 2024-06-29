# `D:\src\scipysrc\matplotlib\lib\matplotlib\projections\geo.py`

```
import numpy as np

import matplotlib as mpl
from matplotlib import _api  # 导入 matplotlib 的私有 API
from matplotlib.axes import Axes  # 导入 Axes 类
import matplotlib.axis as maxis  # 导入 maxis 模块中的 axis
from matplotlib.patches import Circle  # 导入 Circle 类
from matplotlib.path import Path  # 导入 Path 类
import matplotlib.spines as mspines  # 导入 mspines 模块
from matplotlib.ticker import (
    Formatter, NullLocator, FixedLocator, NullFormatter)  # 导入各种 ticker 类
from matplotlib.transforms import Affine2D, BboxTransformTo, Transform  # 导入各种 transforms

class GeoAxes(Axes):
    """An abstract base class for geographic projections."""

    class ThetaFormatter(Formatter):
        """
        Used to format the theta tick labels.  Converts the native
        unit of radians into degrees and adds a degree symbol.
        """
        def __init__(self, round_to=1.0):
            self._round_to = round_to

        def __call__(self, x, pos=None):
            degrees = round(np.rad2deg(x) / self._round_to) * self._round_to
            return f"{degrees:0.0f}\N{DEGREE SIGN}"  # 返回角度格式的字符串

    RESOLUTION = 75  # 类属性，分辨率设为 75

    def _init_axis(self):
        self.xaxis = maxis.XAxis(self, clear=False)  # 创建 x 轴对象
        self.yaxis = maxis.YAxis(self, clear=False)  # 创建 y 轴对象
        self.spines['geo'].register_axis(self.yaxis)  # 在 'geo' 边框上注册 y 轴对象

    def clear(self):
        # docstring inherited
        super().clear()  # 调用父类的 clear 方法清除图形

        self.set_longitude_grid(30)  # 设置经度网格间隔为 30 度
        self.set_latitude_grid(15)  # 设置纬度网格间隔为 15 度
        self.set_longitude_grid_ends(75)  # 设置经度网格结束位置为 75 度
        self.xaxis.set_minor_locator(NullLocator())  # 设置 x 轴次刻度定位器为空
        self.yaxis.set_minor_locator(NullLocator())  # 设置 y 轴次刻度定位器为空
        self.xaxis.set_ticks_position('none')  # 设置 x 轴主刻度位置为空
        self.yaxis.set_ticks_position('none')  # 设置 y 轴主刻度位置为空
        self.yaxis.set_tick_params(label1On=True)  # 打开 y 轴主刻度标签
        # 为什么需要打开 y 轴刻度标签，但 x 轴刻度标签已经打开？

        self.grid(mpl.rcParams['axes.grid'])  # 根据默认参数设置网格显示

        Axes.set_xlim(self, -np.pi, np.pi)  # 设置 x 轴限制为 -π 到 π
        Axes.set_ylim(self, -np.pi / 2.0, np.pi / 2.0)  # 设置 y 轴限制为 -π/2 到 π/2
    # 设置限制和变换函数
    def _set_lim_and_transforms(self):
        # 对已经缩放的数据进行（可能非线性的）投影
        self.transProjection = self._get_core_transform(self.RESOLUTION)

        # 获取仿射变换
        self.transAffine = self._get_affine_transform()

        # 将数据转换为坐标轴上的边界框坐标系
        self.transAxes = BboxTransformTo(self.bbox)

        # 完整的数据变换堆栈 -- 从数据到显示坐标的完整转换
        self.transData = \
            self.transProjection + \
            self.transAffine + \
            self.transAxes

        # 这是经度刻度的转换
        self._xaxis_pretransform = \
            Affine2D() \
            .scale(1, self._longitude_cap * 2) \
            .translate(0, -self._longitude_cap)
        self._xaxis_transform = \
            self._xaxis_pretransform + \
            self.transData
        self._xaxis_text1_transform = \
            Affine2D().scale(1, 0) + \
            self.transData + \
            Affine2D().translate(0, 4)
        self._xaxis_text2_transform = \
            Affine2D().scale(1, 0) + \
            self.transData + \
            Affine2D().translate(0, -4)

        # 这是纬度刻度的转换
        yaxis_stretch = Affine2D().scale(np.pi * 2, 1).translate(-np.pi, 0)
        yaxis_space = Affine2D().scale(1, 1.1)
        self._yaxis_transform = \
            yaxis_stretch + \
            self.transData
        yaxis_text_base = \
            yaxis_stretch + \
            self.transProjection + \
            (yaxis_space +
             self.transAffine +
             self.transAxes)
        self._yaxis_text1_transform = \
            yaxis_text_base + \
            Affine2D().translate(-8, 0)
        self._yaxis_text2_transform = \
            yaxis_text_base + \
            Affine2D().translate(8, 0)

    # 获取仿射变换对象
    def _get_affine_transform(self):
        transform = self._get_core_transform(1)
        xscale, _ = transform.transform((np.pi, 0))
        _, yscale = transform.transform((0, np.pi/2))
        return Affine2D() \
            .scale(0.5 / xscale, 0.5 / yscale) \
            .translate(0.5, 0.5)

    # 获取 x 轴的变换
    def get_xaxis_transform(self, which='grid'):
        _api.check_in_list(['tick1', 'tick2', 'grid'], which=which)
        return self._xaxis_transform

    # 获取 x 轴第一个文本的变换和对齐方式
    def get_xaxis_text1_transform(self, pad):
        return self._xaxis_text1_transform, 'bottom', 'center'

    # 获取 x 轴第二个文本的变换和对齐方式
    def get_xaxis_text2_transform(self, pad):
        return self._xaxis_text2_transform, 'top', 'center'

    # 获取 y 轴的变换
    def get_yaxis_transform(self, which='grid'):
        _api.check_in_list(['tick1', 'tick2', 'grid'], which=which)
        return self._yaxis_transform

    # 获取 y 轴第一个文本的变换和对齐方式
    def get_yaxis_text1_transform(self, pad):
        return self._yaxis_text1_transform, 'center', 'right'

    # 获取 y 轴第二个文本的变换和对齐方式
    def get_yaxis_text2_transform(self, pad):
        return self._yaxis_text2_transform, 'center', 'left'

    # 生成坐标轴补丁对象
    def _gen_axes_patch(self):
        return Circle((0.5, 0.5), 0.5)
    # 生成地理投影的坐标轴脊柱
    def _gen_axes_spines(self):
        return {'geo': mspines.Spine.circular_spine(self, (0.5, 0.5), 0.5)}

    # 设置纵轴的比例尺
    def set_yscale(self, *args, **kwargs):
        if args[0] != 'linear':
            raise NotImplementedError

    # 设置横轴的比例尺与纵轴相同
    set_xscale = set_yscale

    # 设置横轴的限制范围，不支持该操作，建议使用 Cartopy 库
    def set_xlim(self, *args, **kwargs):
        """Not supported. Please consider using Cartopy."""
        raise TypeError("Changing axes limits of a geographic projection is "
                        "not supported.  Please consider using Cartopy.")

    # 设置纵轴的限制范围与横轴相同
    set_ylim = set_xlim

    # 格式化显示经纬度坐标的格式字符串
    def format_coord(self, lon, lat):
        """Return a format string formatting the coordinate."""
        lon, lat = np.rad2deg([lon, lat])
        ns = 'N' if lat >= 0.0 else 'S'
        ew = 'E' if lon >= 0.0 else 'W'
        return ('%f\N{DEGREE SIGN}%s, %f\N{DEGREE SIGN}%s'
                % (abs(lat), ns, abs(lon), ew))

    # 设置经度网格线的间隔
    def set_longitude_grid(self, degrees):
        """
        Set the number of degrees between each longitude grid.
        """
        # 跳过 -180 和 180 度，因为它们是固定的限制
        grid = np.arange(-180 + degrees, 180, degrees)
        self.xaxis.set_major_locator(FixedLocator(np.deg2rad(grid)))
        self.xaxis.set_major_formatter(self.ThetaFormatter(degrees))

    # 设置纬度网格线的间隔
    def set_latitude_grid(self, degrees):
        """
        Set the number of degrees between each latitude grid.
        """
        # 跳过 -90 和 90 度，因为它们是固定的限制
        grid = np.arange(-90 + degrees, 90, degrees)
        self.yaxis.set_major_locator(FixedLocator(np.deg2rad(grid)))
        self.yaxis.set_major_formatter(self.ThetaFormatter(degrees))

    # 设置停止绘制经度网格线的纬度
    def set_longitude_grid_ends(self, degrees):
        """
        Set the latitude(s) at which to stop drawing the longitude grids.
        """
        self._longitude_cap = np.deg2rad(degrees)
        self._xaxis_pretransform \
            .clear() \
            .scale(1.0, self._longitude_cap * 2.0) \
            .translate(0.0, -self._longitude_cap)

    # 返回数据本身的纵横比例
    def get_data_ratio(self):
        """Return the aspect ratio of the data itself."""
        return 1.0

    ### 交互式平移

    # 返回该坐标轴是否支持缩放框按钮功能
    def can_zoom(self):
        """
        Return whether this Axes supports the zoom box button functionality.

        This Axes object does not support interactive zoom box.
        """
        return False

    # 返回该坐标轴是否支持平移/缩放按钮功能
    def can_pan(self):
        """
        Return whether this Axes supports the pan/zoom button functionality.

        This Axes object does not support interactive pan/zoom.
        """
        return False

    # 开始进行平移操作
    def start_pan(self, x, y, button):
        pass

    # 结束平移操作
    def end_pan(self):
        pass

    # 拖动进行平移操作
    def drag_pan(self, button, key, x, y):
        pass
class _GeoTransform(Transform):
    # Factoring out some common functionality.
    input_dims = output_dims = 2

    def __init__(self, resolution):
        """
        Create a new geographical transform.

        Resolution is the number of steps to interpolate between each input
        line segment to approximate its path in curved space.
        """
        super().__init__()
        self._resolution = resolution

    def __str__(self):
        return f"{type(self).__name__}({self._resolution})"

    def transform_path_non_affine(self, path):
        # docstring inherited
        # Interpolate the path with the given resolution
        ipath = path.interpolated(self._resolution)
        # Transform the interpolated path vertices and return a new Path object
        return Path(self.transform(ipath.vertices), ipath.codes)


class AitoffAxes(GeoAxes):
    name = 'aitoff'

    class AitoffTransform(_GeoTransform):
        """The base Aitoff transform."""

        @_api.rename_parameter("3.8", "ll", "values")
        def transform_non_affine(self, values):
            # docstring inherited
            # Extract longitude and latitude values from input
            longitude, latitude = values.T

            # Pre-compute some values for Aitoff projection
            half_long = longitude / 2.0
            cos_latitude = np.cos(latitude)

            alpha = np.arccos(cos_latitude * np.cos(half_long))
            sinc_alpha = np.sinc(alpha / np.pi)  # np.sinc is sin(pi*x)/(pi*x).

            # Perform Aitoff projection calculations
            x = (cos_latitude * np.sin(half_long)) / sinc_alpha
            y = np.sin(latitude) / sinc_alpha
            return np.column_stack([x, y])

        def inverted(self):
            # docstring inherited
            # Return the inverted transform using InvertedAitoffTransform
            return AitoffAxes.InvertedAitoffTransform(self._resolution)

    class InvertedAitoffTransform(_GeoTransform):

        @_api.rename_parameter("3.8", "xy", "values")
        def transform_non_affine(self, values):
            # docstring inherited
            # Return NaN-filled array for the inverted transform
            # MGDTODO: Math is hard ;(
            return np.full_like(values, np.nan)

        def inverted(self):
            # docstring inherited
            # Return the non-inverted transform using AitoffTransform
            return AitoffAxes.AitoffTransform(self._resolution)

    def __init__(self, *args, **kwargs):
        self._longitude_cap = np.pi / 2.0
        super().__init__(*args, **kwargs)
        self.set_aspect(0.5, adjustable='box', anchor='C')
        self.clear()

    def _get_core_transform(self, resolution):
        # Return an instance of AitoffTransform for core transformation
        return self.AitoffTransform(resolution)


class HammerAxes(GeoAxes):
    name = 'hammer'
    class HammerTransform(_GeoTransform):
        """The base Hammer transform."""
    
        @_api.rename_parameter("3.8", "ll", "values")
        def transform_non_affine(self, values):
            # 继承的文档字符串
            # 从输入的二维数组中提取经度和纬度
            longitude, latitude = values.T
            # 计算经度的一半
            half_long = longitude / 2.0
            # 计算纬度的余弦值
            cos_latitude = np.cos(latitude)
            # 计算常数 sqrt(2)
            sqrt2 = np.sqrt(2.0)
            # 计算变换中的 alpha 参数
            alpha = np.sqrt(1.0 + cos_latitude * np.cos(half_long))
            # 计算变换后的 x 坐标
            x = (2.0 * sqrt2) * (cos_latitude * np.sin(half_long)) / alpha
            # 计算变换后的 y 坐标
            y = (sqrt2 * np.sin(latitude)) / alpha
            # 返回变换后的坐标数组
            return np.column_stack([x, y])
    
        def inverted(self):
            # 继承的文档字符串
            # 返回反向变换对象 InvertedHammerTransform
            return HammerAxes.InvertedHammerTransform(self._resolution)
    
    
    class InvertedHammerTransform(_GeoTransform):
        """The inverted Hammer transform."""
    
        @_api.rename_parameter("3.8", "xy", "values")
        def transform_non_affine(self, values):
            # 继承的文档字符串
            # 从输入的二维数组中提取 x 和 y 坐标
            x, y = values.T
            # 计算变换中的 z 参数
            z = np.sqrt(1 - (x / 4) ** 2 - (y / 2) ** 2)
            # 计算反向变换后的经度
            longitude = 2 * np.arctan((z * x) / (2 * (2 * z ** 2 - 1)))
            # 计算反向变换后的纬度
            latitude = np.arcsin(y*z)
            # 返回反向变换后的经纬度数组
            return np.column_stack([longitude, latitude])
    
        def inverted(self):
            # 继承的文档字符串
            # 返回反向的 HammerTransform 对象
            return HammerAxes.HammerTransform(self._resolution)
    
    
    def __init__(self, *args, **kwargs):
        # 初始化函数，设置默认的经度上限为 π/2
        self._longitude_cap = np.pi / 2.0
        # 调用父类的初始化函数
        super().__init__(*args, **kwargs)
        # 设置图形的长宽比为 0.5，可调整为盒子，锚点为中心
        self.set_aspect(0.5, adjustable='box', anchor='C')
        # 清除当前状态
        self.clear()
    
    def _get_core_transform(self, resolution):
        # 返回 HammerTransform 对象，用给定的分辨率初始化
        return self.HammerTransform(resolution)
class MollweideAxes(GeoAxes):
    name = 'mollweide'

    class MollweideTransform(_GeoTransform):
        """The base Mollweide transform."""

        @_api.rename_parameter("3.8", "ll", "values")
        def transform_non_affine(self, values):
            # docstring inherited
            # 定义牛顿-拉弗森迭代函数
            def d(theta):
                delta = (-(theta + np.sin(theta) - pi_sin_l)
                         / (1 + np.cos(theta)))
                return delta, np.abs(delta) > 0.001

            longitude, latitude = values.T

            clat = np.pi/2 - np.abs(latitude)
            ihigh = clat < 0.087  # 在极点附近5度内
            ilow = ~ihigh
            aux = np.empty(latitude.shape, dtype=float)

            if ilow.any():  # 使用牛顿-拉弗森迭代法
                pi_sin_l = np.pi * np.sin(latitude[ilow])
                theta = 2.0 * latitude[ilow]
                delta, large_delta = d(theta)
                while np.any(large_delta):
                    theta[large_delta] += delta[large_delta]
                    delta, large_delta = d(theta)
                aux[ilow] = theta / 2

            if ihigh.any():  # 使用基于泰勒级数的近似解法
                e = clat[ihigh]
                d = 0.5 * (3 * np.pi * e**2) ** (1.0/3)
                aux[ihigh] = (np.pi/2 - d) * np.sign(latitude[ihigh])

            xy = np.empty(values.shape, dtype=float)
            xy[:, 0] = (2.0 * np.sqrt(2.0) / np.pi) * longitude * np.cos(aux)
            xy[:, 1] = np.sqrt(2.0) * np.sin(aux)

            return xy

        def inverted(self):
            # docstring inherited
            # 返回反向转换对象
            return MollweideAxes.InvertedMollweideTransform(self._resolution)

    class InvertedMollweideTransform(_GeoTransform):

        @_api.rename_parameter("3.8", "xy", "values")
        def transform_non_affine(self, values):
            # docstring inherited
            x, y = values.T
            # 根据 Mollweide 投影公式(7, 8)进行反向变换
            theta = np.arcsin(y / np.sqrt(2))
            longitude = (np.pi / (2 * np.sqrt(2))) * x / np.cos(theta)
            latitude = np.arcsin((2 * theta + np.sin(2 * theta)) / np.pi)
            return np.column_stack([longitude, latitude])

        def inverted(self):
            # docstring inherited
            # 返回正向转换对象
            return MollweideAxes.MollweideTransform(self._resolution)

    def __init__(self, *args, **kwargs):
        self._longitude_cap = np.pi / 2.0
        super().__init__(*args, **kwargs)
        # 设置投影的长宽比为0.5，可调整为盒状，锚定在中心
        self.set_aspect(0.5, adjustable='box', anchor='C')
        self.clear()

    def _get_core_transform(self, resolution):
        # 返回 Mollweide 投影的核心转换对象
        return self.MollweideTransform(resolution)


class LambertAxes(GeoAxes):
    name = 'lambert'
    class LambertTransform(_GeoTransform):
        """The base Lambert transform."""

        def __init__(self, center_longitude, center_latitude, resolution):
            """
            创建一个新的 Lambert 变换对象。Resolution 是插值步数，
            用于在曲线 Lambert 空间中近似路径的每个输入线段之间插值的次数。
            """
            _GeoTransform.__init__(self, resolution)
            self._center_longitude = center_longitude
            self._center_latitude = center_latitude

        @_api.rename_parameter("3.8", "ll", "values")
        def transform_non_affine(self, values):
            # 继承的文档字符串
            longitude, latitude = values.T
            clong = self._center_longitude
            clat = self._center_latitude
            cos_lat = np.cos(latitude)
            sin_lat = np.sin(latitude)
            diff_long = longitude - clong
            cos_diff_long = np.cos(diff_long)

            inner_k = np.maximum(  # 防止除零问题
                1 + np.sin(clat)*sin_lat + np.cos(clat)*cos_lat*cos_diff_long,
                1e-15)
            k = np.sqrt(2 / inner_k)
            x = k * cos_lat*np.sin(diff_long)
            y = k * (np.cos(clat)*sin_lat - np.sin(clat)*cos_lat*cos_diff_long)

            return np.column_stack([x, y])

        def inverted(self):
            # 继承的文档字符串
            return LambertAxes.InvertedLambertTransform(
                self._center_longitude,
                self._center_latitude,
                self._resolution)

    class InvertedLambertTransform(_GeoTransform):

        def __init__(self, center_longitude, center_latitude, resolution):
            _GeoTransform.__init__(self, resolution)
            self._center_longitude = center_longitude
            self._center_latitude = center_latitude

        @_api.rename_parameter("3.8", "xy", "values")
        def transform_non_affine(self, values):
            # 继承的文档字符串
            x, y = values.T
            clong = self._center_longitude
            clat = self._center_latitude
            p = np.maximum(np.hypot(x, y), 1e-9)
            c = 2 * np.arcsin(0.5 * p)
            sin_c = np.sin(c)
            cos_c = np.cos(c)

            latitude = np.arcsin(cos_c*np.sin(clat) +
                                 ((y*sin_c*np.cos(clat)) / p))
            longitude = clong + np.arctan(
                (x*sin_c) / (p*np.cos(clat)*cos_c - y*np.sin(clat)*sin_c))

            return np.column_stack([longitude, latitude])

        def inverted(self):
            # 继承的文档字符串
            return LambertAxes.LambertTransform(
                self._center_longitude,
                self._center_latitude,
                self._resolution)
    def __init__(self, *args, center_longitude=0, center_latitude=0, **kwargs):
        # 设置纬度的最大值为 π/2
        self._longitude_cap = np.pi / 2
        # 设置中心经度和纬度
        self._center_longitude = center_longitude
        self._center_latitude = center_latitude
        # 调用父类的初始化方法，传递位置参数和关键字参数
        super().__init__(*args, **kwargs)
        # 设置绘图的纵横比为 'equal'，并且可调整为方框形式，锚点在中心
        self.set_aspect('equal', adjustable='box', anchor='C')
        # 调用 clear 方法进行清空操作
        self.clear()

    def clear(self):
        # 继承的方法文档字符串
        super().clear()
        # 设置 y 轴的主要刻度格式化器为空格式化器
        self.yaxis.set_major_formatter(NullFormatter())

    def _get_core_transform(self, resolution):
        # 返回 LambertTransform 对象，使用指定的中心经度、纬度和分辨率
        return self.LambertTransform(
            self._center_longitude,
            self._center_latitude,
            resolution)

    def _get_affine_transform(self):
        # 返回一个 Affine2D 对象，进行缩放和平移操作
        return Affine2D() \
            .scale(0.25) \
            .translate(0.5, 0.5)
```