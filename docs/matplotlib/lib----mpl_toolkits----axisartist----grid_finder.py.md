# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axisartist\grid_finder.py`

```py
# 导入NumPy库，用于数值计算
import numpy as np

# 从Matplotlib库中导入ticker模块和_api模块
from matplotlib import ticker as mticker, _api

# 从Matplotlib的transforms模块中导入Bbox类和Transform类
from matplotlib.transforms import Bbox, Transform


def _find_line_box_crossings(xys, bbox):
    """
    查找折线与边界框交叉点及交叉角度。

    Parameters
    ----------
    xys : (N, 2) 数组
        折线的坐标。
    bbox : `.Bbox`
        边界框对象。

    Returns
    -------
    list of ((float, float), float)
        四个不同的列表，分别代表边界框的左、右、底、顶四条边的交点。
        每个列表中的条目是交点的((x, y), 逆时针角度（以度为单位）)，其中角度为0表示折线在交点处向右移动。

        条目是通过在每个交点处线性插值计算得到的，插值是在边界框边缘的最近点之间进行的。
    """
    crossings = []
    dxys = xys[1:] - xys[:-1]
    for sl in [slice(None), slice(None, None, -1)]:
        us, vs = xys.T[sl]  # "this" coord, "other" coord
        dus, dvs = dxys.T[sl]
        umin, vmin = bbox.min[sl]
        umax, vmax = bbox.max[sl]
        for u0, inside in [(umin, us > umin), (umax, us < umax)]:
            cross = []
            idxs, = (inside[:-1] ^ inside[1:]).nonzero()
            for idx in idxs:
                v = vs[idx] + (u0 - us[idx]) * dvs[idx] / dus[idx]
                if not vmin <= v <= vmax:
                    continue
                crossing = (u0, v)[sl]
                theta = np.degrees(np.arctan2(*dxys[idx][::-1]))
                cross.append((crossing, theta))
            crossings.append(cross)
    return crossings


class ExtremeFinderSimple:
    """
    一个辅助类，用于确定需要绘制的网格线的范围。
    """

    def __init__(self, nx, ny):
        """
        Parameters
        ----------
        nx, ny : int
            每个方向上的样本数。
        """
        self.nx = nx
        self.ny = ny
    def __call__(self, transform_xy, x1, y1, x2, y2):
        """
        Compute an approximation of the bounding box obtained by applying
        *transform_xy* to the box delimited by ``(x1, y1, x2, y2)``.

        The intended use is to have ``(x1, y1, x2, y2)`` in axes coordinates,
        and have *transform_xy* be the transform from axes coordinates to data
        coordinates; this method then returns the range of data coordinates
        that span the actual axes.

        The computation is done by sampling ``nx * ny`` equispaced points in
        the ``(x1, y1, x2, y2)`` box and finding the resulting points with
        extremal coordinates; then adding some padding to take into account the
        finite sampling.

        As each sampling step covers a relative range of *1/nx* or *1/ny*,
        the padding is computed by expanding the span covered by the extremal
        coordinates by these fractions.
        """
        # 使用 np.meshgrid 在指定范围内生成 nx * ny 个等间距的点坐标网格
        x, y = np.meshgrid(
            np.linspace(x1, x2, self.nx), np.linspace(y1, y2, self.ny))
        # 对生成的网格点坐标应用 transform_xy 进行坐标转换
        xt, yt = transform_xy(np.ravel(x), np.ravel(y))
        # 返回经过添加填充的坐标范围，以确保涵盖极端坐标的有限采样
        return self._add_pad(xt.min(), xt.max(), yt.min(), yt.max())

    def _add_pad(self, x_min, x_max, y_min, y_max):
        """Perform the padding mentioned in `__call__`."""
        # 计算 x 和 y 方向的填充量，以保证覆盖采样极值坐标的范围
        dx = (x_max - x_min) / self.nx
        dy = (y_max - y_min) / self.ny
        # 返回扩展后的坐标范围
        return x_min - dx, x_max + dx, y_min - dy, y_max + dy
# 定义一个继承自 Transform 的二维用户自定义变换类
class _User2DTransform(Transform):
    """A transform defined by two user-set functions."""

    # 输入和输出维度均为 2
    input_dims = output_dims = 2

    def __init__(self, forward, backward):
        """
        Parameters
        ----------
        forward, backward : callable
            The forward and backward transforms, taking ``x`` and ``y`` as
            separate arguments and returning ``(tr_x, tr_y)``.
        """
        # 调用父类的构造方法
        super().__init__()
        # 设置正向和反向变换函数
        self._forward = forward
        self._backward = backward

    def transform_non_affine(self, values):
        # docstring inherited
        # 对非仿射变换进行转换，使用正向变换函数对数据进行转置处理
        return np.transpose(self._forward(*np.transpose(values)))

    def inverted(self):
        # docstring inherited
        # 返回当前对象类型的反向变换对象，使用反向变换函数作为正向变换函数，正向变换函数作为反向变换函数
        return type(self)(self._backward, self._forward)


class GridFinder:
    """
    Internal helper for `~.grid_helper_curvelinear.GridHelperCurveLinear`, with
    the same constructor parameters; should not be directly instantiated.
    """

    def __init__(self,
                 transform,
                 extreme_finder=None,
                 grid_locator1=None,
                 grid_locator2=None,
                 tick_formatter1=None,
                 tick_formatter2=None):
        # 如果未提供极值查找器，则使用默认的 ExtremeFinderSimple(20, 20)
        if extreme_finder is None:
            extreme_finder = ExtremeFinderSimple(20, 20)
        # 如果未提供网格定位器1，则使用默认的 MaxNLocator()
        if grid_locator1 is None:
            grid_locator1 = MaxNLocator()
        # 如果未提供网格定位器2，则使用默认的 MaxNLocator()
        if grid_locator2 is None:
            grid_locator2 = MaxNLocator()
        # 如果未提供刻度格式化器1，则使用默认的 FormatterPrettyPrint()
        if tick_formatter1 is None:
            tick_formatter1 = FormatterPrettyPrint()
        # 如果未提供刻度格式化器2，则使用默认的 FormatterPrettyPrint()
        if tick_formatter2 is None:
            tick_formatter2 = FormatterPrettyPrint()
        
        # 设置极值查找器、网格定位器1和2、刻度格式化器1和2
        self.extreme_finder = extreme_finder
        self.grid_locator1 = grid_locator1
        self.grid_locator2 = grid_locator2
        self.tick_formatter1 = tick_formatter1
        self.tick_formatter2 = tick_formatter2
        # 设置变换对象
        self.set_transform(transform)

    def _format_ticks(self, idx, direction, factor, levels):
        """
        Helper to support both standard formatters (inheriting from
        `.mticker.Formatter`) and axisartist-specific ones; should be called instead of
        directly calling ``self.tick_formatter1`` and ``self.tick_formatter2``.  This
        method should be considered as a temporary workaround which will be removed in
        the future at the same time as axisartist-specific formatters.
        """
        # 获取对应索引的格式化器
        fmt = _api.check_getitem(
            {1: self.tick_formatter1, 2: self.tick_formatter2}, idx=idx)
        # 如果格式化器是 mticker.Formatter 类型，则调用其 format_ticks 方法，否则直接调用格式化器函数
        return (fmt.format_ticks(levels) if isinstance(fmt, mticker.Formatter)
                else fmt(direction, factor, levels))
    def get_grid_info(self, x1, y1, x2, y2):
        """
        lon_values, lat_values : list of grid values. if integer is given,
                           rough number of grids in each direction.
        """
        # 调用 self.extreme_finder 方法，获取指定区域的极值信息
        extremes = self.extreme_finder(self.inv_transform_xy, x1, y1, x2, y2)

        # 计算经度和纬度的最小值和最大值
        lon_min, lon_max, lat_min, lat_max = extremes

        # 使用 self.grid_locator1 方法计算经度的刻度线位置及相关参数
        lon_levs, lon_n, lon_factor = self.grid_locator1(lon_min, lon_max)
        lon_levs = np.asarray(lon_levs)

        # 使用 self.grid_locator2 方法计算纬度的刻度线位置及相关参数
        lat_levs, lat_n, lat_factor = self.grid_locator2(lat_min, lat_max)
        lat_levs = np.asarray(lat_levs)

        # 根据计算得到的刻度线位置和因子，计算最终的经度和纬度值
        lon_values = lon_levs[:lon_n] / lon_factor
        lat_values = lat_levs[:lat_n] / lat_factor

        # 调用 self._get_raw_grid_lines 方法获取原始经度和纬度网格线
        lon_lines, lat_lines = self._get_raw_grid_lines(lon_values,
                                                        lat_values,
                                                        lon_min, lon_max,
                                                        lat_min, lat_max)

        # 创建包围框对象，扩展边界以确保覆盖所有网格线
        bb = Bbox.from_extents(x1, y1, x2, y2).expanded(1 + 2e-10, 1 + 2e-10)

        # 初始化 grid_info 字典，包含极值信息和空的经度和纬度字段
        grid_info = {
            "extremes": extremes,
            # "lon", "lat", filled below.
        }

        # 遍历经度和纬度信息，填充 grid_info 字典中的 "lon" 和 "lat" 字段
        for idx, lon_or_lat, levs, factor, values, lines in [
                (1, "lon", lon_levs, lon_factor, lon_values, lon_lines),
                (2, "lat", lat_levs, lat_factor, lat_values, lat_lines),
        ]:
            grid_info[lon_or_lat] = gi = {
                "lines": [[l] for l in lines],  # 将网格线格式化为所需的数据结构
                "ticks": {"left": [], "right": [], "bottom": [], "top": []},  # 初始化空的刻度字典
            }
            # 遍历每条网格线，计算其与包围框的交叉点，并添加到刻度字典中
            for (lx, ly), v, level in zip(lines, values, levs):
                all_crossings = _find_line_box_crossings(np.column_stack([lx, ly]), bb)
                for side, crossings in zip(
                        ["left", "right", "bottom", "top"], all_crossings):
                    for crossing in crossings:
                        gi["ticks"][side].append({"level": level, "loc": crossing})
            # 对每个方向的刻度进行格式化处理，添加标签信息
            for side in gi["ticks"]:
                levs = [tick["level"] for tick in gi["ticks"][side]]
                labels = self._format_ticks(idx, side, factor, levs)
                for tick, label in zip(gi["ticks"][side], labels):
                    tick["label"] = label

        # 返回完整的 grid_info 字典，包含所有的经度和纬度信息
        return grid_info

    def _get_raw_grid_lines(self,
                            lon_values, lat_values,
                            lon_min, lon_max, lat_min, lat_max):
        """
        Calculate raw grid lines based on given lon_values and lat_values.
        """
        # 使用 linspace 方法生成经度和纬度的插值点
        lons_i = np.linspace(lon_min, lon_max, 100)  # for interpolation
        lats_i = np.linspace(lat_min, lat_max, 100)

        # 根据给定的经度值生成经度网格线
        lon_lines = [self.transform_xy(np.full_like(lats_i, lon), lats_i)
                     for lon in lon_values]
        # 根据给定的纬度值生成纬度网格线
        lat_lines = [self.transform_xy(lons_i, np.full_like(lons_i, lat))
                     for lat in lat_values]

        # 返回生成的经度和纬度网格线
        return lon_lines, lat_lines
    # 设置辅助变换对象。如果 aux_trans 是 Transform 的实例，则将其赋给 _aux_transform。
    # 如果 aux_trans 是一个包含两个可调用对象的长度为2的列表，则创建一个 _User2DTransform 对象并赋值给 _aux_transform。
    # 否则，引发 TypeError 异常，指示 'aux_trans' 必须是 Transform 实例或一对可调用对象。
    def set_transform(self, aux_trans):
        if isinstance(aux_trans, Transform):
            self._aux_transform = aux_trans
        elif len(aux_trans) == 2 and all(map(callable, aux_trans)):
            self._aux_transform = _User2DTransform(*aux_trans)
        else:
            raise TypeError("'aux_trans' must be either a Transform "
                            "instance or a pair of callables")

    # 返回当前辅助变换对象 _aux_transform。
    def get_transform(self):
        return self._aux_transform

    # 将 set_transform 方法设置为 update_transform 的别名，以保持向后兼容性。
    update_transform = set_transform  # backcompat alias.

    # 使用辅助变换对象 _aux_transform 对给定的 x 和 y 进行变换，并返回结果的转置。
    def transform_xy(self, x, y):
        return self._aux_transform.transform(np.column_stack([x, y])).T

    # 使用 _aux_transform 的逆变换对给定的 x 和 y 进行反变换，并返回结果的转置。
    def inv_transform_xy(self, x, y):
        return self._aux_transform.inverted().transform(
            np.column_stack([x, y])).T

    # 更新对象的属性，接受关键字参数。允许更新的属性包括特定的字符串列表。
    # 如果关键字不在允许更新的列表中，引发 ValueError 异常。
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k in ["extreme_finder",
                     "grid_locator1",
                     "grid_locator2",
                     "tick_formatter1",
                     "tick_formatter2"]:
                setattr(self, k, v)
            else:
                raise ValueError(f"Unknown update property {k!r}")
class MaxNLocator(mticker.MaxNLocator):
    # MaxNLocator 类的子类，用于定位最大的 N 个刻度值
    def __init__(self, nbins=10, steps=None,
                 trim=True,
                 integer=False,
                 symmetric=False,
                 prune=None):
        # trim 参数无效，仅为保持 API 兼容性而保留
        super().__init__(nbins, steps=steps, integer=integer,
                         symmetric=symmetric, prune=prune)
        # 创建虚拟轴
        self.create_dummy_axis()

    def __call__(self, v1, v2):
        # 调用父类的 tick_values 方法获取刻度值
        locs = super().tick_values(v1, v2)
        return np.array(locs), len(locs), 1  # 1: factor (see angle_helper)


class FixedLocator:
    # 固定位置刻度定位器
    def __init__(self, locs):
        self._locs = locs

    def __call__(self, v1, v2):
        v1, v2 = sorted([v1, v2])
        # 从预设位置中选择 v1 和 v2 之间的刻度位置
        locs = np.array([l for l in self._locs if v1 <= l <= v2])
        return locs, len(locs), 1  # 1: factor (see angle_helper)


# Tick Formatter

class FormatterPrettyPrint:
    # 美观打印格式化器
    def __init__(self, useMathText=True):
        self._fmt = mticker.ScalarFormatter(
            useMathText=useMathText, useOffset=False)
        self._fmt.create_dummy_axis()

    def __call__(self, direction, factor, values):
        # 使用内部的 ScalarFormatter 格式化刻度值
        return self._fmt.format_ticks(values)


class DictFormatter:
    # 字典格式化器
    def __init__(self, format_dict, formatter=None):
        """
        format_dict : 用于格式化字符串的字典
        formatter : 回退格式化器
        """
        super().__init__()
        self._format_dict = format_dict
        self._fallback_formatter = formatter

    def __call__(self, direction, factor, values):
        """
        如果值在字典中找到，则忽略 factor
        """
        if self._fallback_formatter:
            fallback_strings = self._fallback_formatter(
                direction, factor, values)
        else:
            fallback_strings = [""] * len(values)
        # 根据字典中的键值对格式化刻度值
        return [self._format_dict.get(k, v)
                for k, v in zip(values, fallback_strings)]
```