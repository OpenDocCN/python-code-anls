# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axisartist\angle_helper.py`

```py
import numpy as np  # 导入NumPy库，用于数值计算
import math  # 导入math库，用于数学计算

from mpl_toolkits.axisartist.grid_finder import ExtremeFinderSimple  # 导入特定模块

def select_step_degree(dv):
    """
    根据给定的角度差值选择合适的步长和因子。

    Args:
        dv: 角度差值

    Returns:
        step: 步长
        factor: 因子
    """
    degree_limits_ = [1.5, 3, 7, 13, 20, 40, 70, 120, 270, 520]  # 角度限制
    degree_steps_  = [1,   2, 5, 10, 15, 30, 45,  90, 180, 360]   # 步长
    degree_factors = [1.] * len(degree_steps_)  # 因子初始化为1

    minsec_limits_ = [1.5, 2.5, 3.5, 8, 11, 18, 25, 45]  # 分秒限制
    minsec_steps_  = [1,   2,   3,   5, 10, 15, 20, 30]  # 分秒步长

    minute_limits_ = np.array(minsec_limits_) / 60  # 分转换成小时
    minute_factors = [60.] * len(minute_limits_)  # 分的因子为60

    second_limits_ = np.array(minsec_limits_) / 3600  # 秒转换成小时
    second_factors = [3600.] * len(second_limits_)  # 秒的因子为3600

    degree_limits = [*second_limits_, *minute_limits_, *degree_limits_]  # 合并所有限制
    degree_steps = [*minsec_steps_, *minsec_steps_, *degree_steps_]  # 合并所有步长
    degree_factors = [*second_factors, *minute_factors, *degree_factors]  # 合并所有因子

    n = np.searchsorted(degree_limits, dv)  # 查找角度差值所在的区间
    step = degree_steps[n]  # 获取对应区间的步长
    factor = degree_factors[n]  # 获取对应区间的因子

    return step, factor


def select_step_hour(dv):
    """
    根据给定的时间差值选择合适的步长和因子。

    Args:
        dv: 时间差值

    Returns:
        step: 步长
        factor: 因子
    """
    hour_limits_ = [1.5, 2.5, 3.5, 5, 7, 10, 15, 21, 36]  # 小时限制
    hour_steps_  = [1,   2,   3,   4, 6,  8, 12, 18, 24]   # 小时步长
    hour_factors = [1.] * len(hour_steps_)  # 因子初始化为1

    minsec_limits_ = [1.5, 2.5, 3.5, 4.5, 5.5, 8, 11, 14, 18, 25, 45]  # 分秒限制
    minsec_steps_  = [1,   2,   3,   4,   5,   6, 10, 12, 15, 20, 30]   # 分秒步长

    minute_limits_ = np.array(minsec_limits_) / 60  # 分转换成小时
    minute_factors = [60.] * len(minute_limits_)  # 分的因子为60

    second_limits_ = np.array(minsec_limits_) / 3600  # 秒转换成小时
    second_factors = [3600.] * len(second_limits_)  # 秒的因子为3600

    hour_limits = [*second_limits_, *minute_limits_, *hour_limits_]  # 合并所有限制
    hour_steps = [*minsec_steps_, *minsec_steps_, *hour_steps_]  # 合并所有步长
    hour_factors = [*second_factors, *minute_factors, *hour_factors]  # 合并所有因子

    n = np.searchsorted(hour_limits, dv)  # 查找时间差值所在的区间
    step = hour_steps[n]  # 获取对应区间的步长
    factor = hour_factors[n]  # 获取对应区间的因子

    return step, factor


def select_step_sub(dv):
    """
    根据给定的子角度或角度选择合适的步长和因子。

    Args:
        dv: 角度或子角度差值

    Returns:
        step: 步长
        factor: 因子
    """
    # subarcsec or degree
    tmp = 10.**(int(math.log10(dv))-1.)  # 计算临时值

    factor = 1./tmp  # 计算因子

    if 1.5*tmp >= dv:
        step = 1
    elif 3.*tmp >= dv:
        step = 2
    elif 7.*tmp >= dv:
        step = 5
    else:
        step = 1
        factor = 0.1*factor

    return step, factor


def select_step(v1, v2, nv, hour=False, include_last=True,
                threshold_factor=3600.):
    """
    根据给定的范围和步数选择合适的步长和因子。

    Args:
        v1: 起始值
        v2: 结束值
        nv: 步数
        hour: 是否处理小时范围
        include_last: 是否包含最后一个值
        threshold_factor: 阈值因子

    Returns:
        levs: 步长列表
    """
    if v1 > v2:
        v1, v2 = v2, v1  # 确保v1小于等于v2

    dv = (v2 - v1) / nv  # 计算范围内的步长

    if hour:
        _select_step = select_step_hour  # 处理小时范围
        cycle = 24.
    else:
        _select_step = select_step_degree  # 处理角度范围
        cycle = 360.

    # for degree
    if dv > 1 / threshold_factor:
        step, factor = _select_step(dv)  # 选择角度步长和因子
    else:
        step, factor = select_step_sub(dv*threshold_factor)  # 选择子角度步长和因子

        factor = factor * threshold_factor  # 根据阈值因子调整因子

    levs = np.arange(np.floor(v1 * factor / step),
                     np.ceil(v2 * factor / step) + 0.5,
                     dtype=int) * step  # 根据计算得到的步长生成等级列表

    # n : number of valid levels. If there is a cycle, e.g., [0, 90, 180,
    # 270, 360], the grid line needs to be extended from 0 to 360, so
    # 我们需要返回整个数组。然而，通常需要忽略最后一个级别（360）。
    # 在这种情况下，我们返回 n=4。

    n = len(levs)

    # 我们需要检查数值范围
    # 例如，-90 到 90，0 到 360，

    if factor == 1. and levs[-1] >= levs[0] + cycle:  # 检查是否有周期
        nv = int(cycle / step)
        if include_last:
            levs = levs[0] + np.arange(0, nv+1, 1) * step
        else:
            levs = levs[0] + np.arange(0, nv, 1) * step

        n = len(levs)

    # 返回 NumPy 数组，包括修正后的 levels，数量 n，以及 factor
    return np.array(levs), n, factor
# 定义函数 select_step24，计算基于小时的步长选择
def select_step24(v1, v2, nv, include_last=True, threshold_factor=3600):
    # 将输入值 v1 和 v2 分别除以 15
    v1, v2 = v1 / 15, v2 / 15
    # 调用 select_step 函数，计算步长及相关参数
    levs, n, factor = select_step(v1, v2, nv, hour=True,
                                  include_last=include_last,
                                  threshold_factor=threshold_factor)
    # 返回转换后的步长 levs、参数 n 和 factor
    return levs * 15, n, factor


# 定义函数 select_step360，计算基于度的步长选择
def select_step360(v1, v2, nv, include_last=True, threshold_factor=3600):
    # 调用 select_step 函数，计算步长及相关参数，不基于小时
    return select_step(v1, v2, nv, hour=False,
                       include_last=include_last,
                       threshold_factor=threshold_factor)


# 定义 LocatorBase 类，表示位置定位器的基础类
class LocatorBase:
    def __init__(self, nbins, include_last=True):
        # 初始化 nbins 属性，表示 bin 数量
        self.nbins = nbins
        # 初始化 _include_last 属性，表示是否包含最后一个值
        self._include_last = include_last

    # 设置参数的方法
    def set_params(self, nbins=None):
        # 如果传入了 nbins 参数，则更新 self.nbins
        if nbins is not None:
            self.nbins = int(nbins)


# 定义 LocatorHMS 类，表示小时-分钟-秒钟的位置定位器
class LocatorHMS(LocatorBase):
    def __call__(self, v1, v2):
        # 调用 select_step24 函数，基于小时返回步长选择结果
        return select_step24(v1, v2, self.nbins, self._include_last)


# 定义 LocatorHM 类，表示小时-分钟的位置定位器
class LocatorHM(LocatorBase):
    def __call__(self, v1, v2):
        # 调用 select_step24 函数，基于分钟返回步长选择结果，阈值因子为 60
        return select_step24(v1, v2, self.nbins, self._include_last,
                             threshold_factor=60)


# 定义 LocatorH 类，表示小时的位置定位器
class LocatorH(LocatorBase):
    def __call__(self, v1, v2):
        # 调用 select_step24 函数，基于小时返回步长选择结果，阈值因子为 1
        return select_step24(v1, v2, self.nbins, self._include_last,
                             threshold_factor=1)


# 定义 LocatorDMS 类，表示度-分-秒的位置定位器
class LocatorDMS(LocatorBase):
    def __call__(self, v1, v2):
        # 调用 select_step360 函数，基于度-分-秒返回步长选择结果
        return select_step360(v1, v2, self.nbins, self._include_last)


# 定义 LocatorDM 类，表示度-分的位置定位器
class LocatorDM(LocatorBase):
    def __call__(self, v1, v2):
        # 调用 select_step360 函数，基于度-分返回步长选择结果，阈值因子为 60
        return select_step360(v1, v2, self.nbins, self._include_last,
                              threshold_factor=60)


# 定义 LocatorD 类，表示度的位置定位器
class LocatorD(LocatorBase):
    def __call__(self, v1, v2):
        # 调用 select_step360 函数，基于度返回步长选择结果，阈值因子为 1
        return select_step360(v1, v2, self.nbins, self._include_last,
                              threshold_factor=1)


# 定义 FormatterDMS 类，表示度-分-秒的格式化器
class FormatterDMS:
    # 定义度、分、秒的标记
    deg_mark = r"^{\circ}"
    min_mark = r"^{\prime}"
    sec_mark = r"^{\prime\prime}"

    # 格式化度的整数部分
    fmt_d = "$%d" + deg_mark + "$"
    # 格式化度的小数部分
    fmt_ds = r"$%d.%s" + deg_mark + "$"

    # 格式化度-分
    fmt_d_m = r"$%s%d" + deg_mark + r"\,%02d" + min_mark + "$"
    fmt_d_ms = r"$%s%d" + deg_mark + r"\,%02d.%s" + min_mark + "$"

    # 部分格式化度-分，用于分数表示
    fmt_d_m_partial = "$%s%d" + deg_mark + r"\,%02d" + min_mark + r"\,"
    # 部分格式化秒，用于分数表示
    fmt_s_partial = "%02d" + sec_mark + "$"
    # 部分格式化秒，用于小数表示
    fmt_ss_partial = "%02d.%s" + sec_mark + "$"

    # 获取数值的分数部分，例如 1.5 -> 3
    def _get_number_fraction(self, factor):
        ## check for fractional numbers
        number_fraction = None
        # 检查因子是否为分数
        for threshold in [1, 60, 3600]:
            if factor <= threshold:
                break

            d = factor // threshold
            int_log_d = int(np.floor(np.log10(d)))
            # 如果为整数且不为 1，则为分数
            if 10**int_log_d == d and d != 1:
                number_fraction = int_log_d
                factor = factor // 10**int_log_d
                return factor, number_fraction

        return factor, number_fraction
    # 定义一个方法，根据给定的方向、因子和数值列表进行处理
    def __call__(self, direction, factor, values):
        # 如果数值列表为空，则返回空列表
        if len(values) == 0:
            return []

        # 对数值取符号，生成符号列表
        ss = np.sign(values)
        signs = ["-" if v < 0 else "" for v in values]

        # 获取因子和小数部分位数
        factor, number_fraction = self._get_number_fraction(factor)

        # 对数值取绝对值
        values = np.abs(values)

        # 如果存在小数部分位数
        if number_fraction is not None:
            # 对数值进行整除和取余操作
            values, frac_part = divmod(values, 10 ** number_fraction)
            # 格式化小数部分为字符串列表
            frac_fmt = "%%0%dd" % (number_fraction,)
            frac_str = [frac_fmt % (f1,) for f1 in frac_part]

        # 根据不同的因子进行处理
        if factor == 1:
            # 如果因子为1且没有小数部分，则返回格式化的整数列表
            if number_fraction is None:
                return [self.fmt_d % (s * int(v),) for s, v in zip(ss, values)]
            else:
                # 否则返回格式化的整数和小数部分列表
                return [self.fmt_ds % (s * int(v), f1)
                        for s, v, f1 in zip(ss, values, frac_str)]
        elif factor == 60:
            # 如果因子为60，则进行度和分的转换
            deg_part, min_part = divmod(values, 60)
            if number_fraction is None:
                # 如果没有小数部分，则返回格式化的度分列表
                return [self.fmt_d_m % (s1, d1, m1)
                        for s1, d1, m1 in zip(signs, deg_part, min_part)]
            else:
                # 否则返回格式化的度分和小数部分列表
                return [self.fmt_d_ms % (s, d1, m1, f1)
                        for s, d1, m1, f1
                        in zip(signs, deg_part, min_part, frac_str)]
        elif factor == 3600:
            # 如果因子为3600，则进行度分秒的转换
            if ss[-1] == -1:
                # 如果最后一个数值为负数，则标记为逆序
                inverse_order = True
                values = values[::-1]
                signs = signs[::-1]
            else:
                inverse_order = False

            l_hm_old = ""
            r = []

            # 对数值进行度分秒的分解
            deg_part, min_part_ = divmod(values, 3600)
            min_part, sec_part = divmod(min_part_, 60)

            if number_fraction is None:
                # 如果没有小数部分，则格式化秒部分
                sec_str = [self.fmt_s_partial % (s1,) for s1 in sec_part]
            else:
                # 否则格式化秒和小数部分
                sec_str = [self.fmt_ss_partial % (s1, f1)
                           for s1, f1 in zip(sec_part, frac_str)]

            # 根据标记进行结果列表的生成
            for s, d1, m1, s1 in zip(signs, deg_part, min_part, sec_str):
                l_hm = self.fmt_d_m_partial % (s, d1, m1)
                if l_hm != l_hm_old:
                    l_hm_old = l_hm
                    l = l_hm + s1
                else:
                    l = "$" + s + s1
                r.append(l)

            # 如果是逆序，则反转结果列表
            if inverse_order:
                return r[::-1]
            else:
                return r
        else:  # factor > 3600.
            # 对于大于3600的因子，返回格式化的特殊字符串列表
            return [r"$%s^{\circ}$" % v for v in ss*values]
class FormatterHMS(FormatterDMS):
    # 继承自 FormatterDMS 类

    deg_mark = r"^\mathrm{h}"
    # 度标记，表示小时的符号

    min_mark = r"^\mathrm{m}"
    # 分标记，表示分钟的符号

    sec_mark = r"^\mathrm{s}"
    # 秒标记，表示秒的符号

    fmt_d = "$%d" + deg_mark + "$"
    # 格式化字符串，用于只显示度数的情况

    fmt_ds = r"$%d.%s" + deg_mark + "$"
    # 格式化字符串，用于显示度数和部分秒数的情况

    # %s for sign
    fmt_d_m = r"$%s%d" + deg_mark + r"\,%02d" + min_mark+"$"
    # 格式化字符串，用于显示度数、分钟数及其符号的情况

    fmt_d_ms = r"$%s%d" + deg_mark + r"\,%02d.%s" + min_mark+"$"
    # 格式化字符串，用于显示度数、分钟数、部分秒数及其符号的情况

    fmt_d_m_partial = "$%s%d" + deg_mark + r"\,%02d" + min_mark + r"\,"
    # 格式化字符串，用于显示度数、分钟数及其符号的部分情况

    fmt_s_partial = "%02d" + sec_mark + "$"
    # 格式化字符串，用于显示秒数及其符号的部分情况

    fmt_ss_partial = "%02d.%s" + sec_mark + "$"
    # 格式化字符串，用于显示秒数、部分秒数及其符号的部分情况

    def __call__(self, direction, factor, values):  # hour
        # 重写父类方法，处理小时单位的转换
        return super().__call__(direction, factor, np.asarray(values) / 15)


class ExtremeFinderCycle(ExtremeFinderSimple):
    # 继承自 ExtremeFinderSimple 类
    # docstring inherited

    def __init__(self, nx, ny,
                 lon_cycle=360., lat_cycle=None,
                 lon_minmax=None, lat_minmax=(-90, 90)):
        """
        This subclass handles the case where one or both coordinates should be
        taken modulo 360, or be restricted to not exceed a specific range.

        Parameters
        ----------
        nx, ny : int
            The number of samples in each direction.

        lon_cycle, lat_cycle : 360 or None
            If not None, values in the corresponding direction are taken modulo
            *lon_cycle* or *lat_cycle*; in theory this can be any number but
            the implementation actually assumes that it is 360 (if not None);
            other values give nonsensical results.

            This is done by "unwrapping" the transformed grid coordinates so
            that jumps are less than a half-cycle; then normalizing the span to
            no more than a full cycle.

            For example, if values are in the union of the [0, 2] and
            [358, 360] intervals (typically, angles measured modulo 360), the
            values in the second interval are normalized to [-2, 0] instead so
            that the values now cover [-2, 2].  If values are in a range of
            [5, 1000], this gets normalized to [5, 365].

        lon_minmax, lat_minmax : (float, float) or None
            If not None, the computed bounding box is clipped to the given
            range in the corresponding direction.
        """
        # 初始化方法，处理周期性和范围限制的坐标情况
        self.nx, self.ny = nx, ny
        self.lon_cycle, self.lat_cycle = lon_cycle, lat_cycle
        self.lon_minmax = lon_minmax
        self.lat_minmax = lat_minmax
    # 继承自父类的文档字符串，描述此方法的作用
    def __call__(self, transform_xy, x1, y1, x2, y2):
        # 创建一个二维网格，其中 x 从 x1 到 x2 均匀分布，y 从 y1 到 y2 均匀分布
        x, y = np.meshgrid(
            np.linspace(x1, x2, self.nx), np.linspace(y1, y2, self.ny))
        # 使用给定的坐标转换函数 transform_xy 处理所有网格点的经纬度坐标
        lon, lat = transform_xy(np.ravel(x), np.ravel(y))

        # 解决经度跳跃问题，但此算法可能需要改进
        # 这只是一种简单的方法，对某些情况可能无效
        # 考虑使用 numpy.unwrap 替代这种方法
        # 我们忽略无效警告。当使用 np.nanmin 和 np.nanmax 比较带 NaN 的数组时会触发这些警告
        with np.errstate(invalid='ignore'):
            # 如果 lon_cycle 不为 None，则调整经度值，确保其在指定范围内
            if self.lon_cycle is not None:
                lon0 = np.nanmin(lon)
                lon -= 360. * ((lon - lon0) > 180.)
            # 如果 lat_cycle 不为 None，则调整纬度值，确保其在指定范围内
            if self.lat_cycle is not None:
                lat0 = np.nanmin(lat)
                lat -= 360. * ((lat - lat0) > 180.)

        # 计算经纬度的最小和最大值（忽略 NaN 值）
        lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
        lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)

        # 使用内部方法 _add_pad 对经纬度范围进行填充处理
        lon_min, lon_max, lat_min, lat_max = \
            self._add_pad(lon_min, lon_max, lat_min, lat_max)

        # 检查经度是否需要调整为周期性值
        if self.lon_cycle:
            lon_max = min(lon_max, lon_min + self.lon_cycle)
        # 检查纬度是否需要调整为周期性值
        if self.lat_cycle:
            lat_max = min(lat_max, lat_min + self.lat_cycle)

        # 如果指定了 lon_minmax，则确保经度范围在指定的最小和最大值之间
        if self.lon_minmax is not None:
            min0 = self.lon_minmax[0]
            lon_min = max(min0, lon_min)
            max0 = self.lon_minmax[1]
            lon_max = min(max0, lon_max)

        # 如果指定了 lat_minmax，则确保纬度范围在指定的最小和最大值之间
        if self.lat_minmax is not None:
            min0 = self.lat_minmax[0]
            lat_min = max(min0, lat_min)
            max0 = self.lat_minmax[1]
            lat_max = min(max0, lat_max)

        # 返回经度和纬度的最终范围
        return lon_min, lon_max, lat_min, lat_max
```