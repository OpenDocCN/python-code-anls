# `D:\src\scipysrc\matplotlib\lib\matplotlib\dates.py`

```py
"""
Matplotlib provides sophisticated date plotting capabilities, standing on the
shoulders of python :mod:`datetime` and the add-on module dateutil_.

By default, Matplotlib uses the units machinery described in
`~matplotlib.units` to convert `datetime.datetime`, and `numpy.datetime64`
objects when plotted on an x- or y-axis. The user does not
need to do anything for dates to be formatted, but dates often have strict
formatting needs, so this module provides many tick locators and formatters.
A basic example using `numpy.datetime64` is::

    import numpy as np

    times = np.arange(np.datetime64('2001-01-02'),
                      np.datetime64('2002-02-03'), np.timedelta64(75, 'm'))
    y = np.random.randn(len(times))

    fig, ax = plt.subplots()
    ax.plot(times, y)

.. seealso::

    - :doc:`/gallery/text_labels_and_annotations/date`
    - :doc:`/gallery/ticks/date_concise_formatter`
    - :doc:`/gallery/ticks/date_demo_convert`

.. _date-format:

Matplotlib date format
----------------------

Matplotlib represents dates using floating point numbers specifying the number
of days since a default epoch of 1970-01-01 UTC; for example,
1970-01-01, 06:00 is the floating point number 0.25. The formatters and
locators require the use of `datetime.datetime` objects, so only dates between
year 0001 and 9999 can be represented.  Microsecond precision
is achievable for (approximately) 70 years on either side of the epoch, and
20 microseconds for the rest of the allowable range of dates (year 0001 to
9999). The epoch can be changed at import time via `.dates.set_epoch` or
:rc:`dates.epoch` to other dates if necessary; see
:doc:`/gallery/ticks/date_precision_and_epochs` for a discussion.

.. note::

   Before Matplotlib 3.3, the epoch was 0000-12-31 which lost modern
   microsecond precision and also made the default axis limit of 0 an invalid
   datetime.  In 3.3 the epoch was changed as above.  To convert old
   ordinal floats to the new epoch, users can do::

     new_ordinal = old_ordinal + mdates.date2num(np.datetime64('0000-12-31'))


There are a number of helper functions to convert between :mod:`datetime`
objects and Matplotlib dates:

.. currentmodule:: matplotlib.dates

.. autosummary::
   :nosignatures:

   datestr2num
   date2num
   num2date
   num2timedelta
   drange
   set_epoch
   get_epoch
"""

# 导入需要的库
import numpy as np

# 创建一个包含时间序列的 numpy.ndarray 对象
times = np.arange(np.datetime64('2001-01-02'),
                  np.datetime64('2002-02-03'), np.timedelta64(75, 'm'))
# 创建一个与 times 长度相同的随机数据序列
y = np.random.randn(len(times))

# 创建一个新的图形和坐标轴
fig, ax = plt.subplots()
# 在坐标轴上绘制时间序列和相应的数据
ax.plot(times, y)
# 导入 matplotlib.dates 中的 MO, TU, WE, TH, FR, SA, SU 常量，用于指定星期几
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU

# WeekdayLocator 可以指定每周的某一天作为刻度
# 例如，每周的星期一
loc = WeekdayLocator(byweekday=MO, tz=tz)

# 或者同时指定多个星期几作为刻度，例如星期一和星期六
loc = WeekdayLocator(byweekday=(MO, SA))

# WeekdayLocator 的构造函数可以接受 interval 参数，用于指定刻度的间隔
# 例如，每两周的星期一
loc = WeekdayLocator(byweekday=MO, interval=2)

# rrulelocator 允许完全通用的日期刻度设置
# 例如，每隔5个复活节进行刻度
rule = rrulewrapper(YEARLY, byeaster=1, interval=5)
loc = RRuleLocator(rule)
# `RRuleLocator`: 根据 `rrulewrapper` 定位器定位日期。`rrulewrapper` 是对 dateutil_ 库中 `dateutil.rrule` 的简单包装器，允许几乎任意的日期刻度规范。
# 查看 :doc:`rrule example </gallery/ticks/date_demo_rrule>` 以获取示例。

# `AutoDateLocator`: 在自动缩放时，此类选择最佳的 `DateLocator`（例如 `RRuleLocator`）来设置视图限制和刻度位置。
# 如果使用 `interval_multiples=True` 参数调用，它将使刻度线与刻度间隔的合理倍数对齐。
# 例如，如果间隔为 4 小时，则会选择小时为 0、4、8 等作为刻度。默认情况下不保证此行为。

# `AutoDateFormatter`: 尝试找出最佳的日期格式来使用。在与 `AutoDateLocator` 结合使用时最为有用。

# `ConciseDateFormatter`: 同样尝试找出最佳的日期格式来使用，并使格式尽可能紧凑，同时仍具有完整的日期信息。
# 在与 `AutoDateLocator` 结合使用时最为有用。

# `DateFormatter`: 使用 `~datetime.datetime.strftime` 格式字符串来格式化日期。
# Time-related constants.

# EPOCH_OFFSET is a constant representing the offset in days from the epoch (1970-01-01)
# as a float value.
EPOCH_OFFSET = float(datetime.datetime(1970, 1, 1).toordinal())

# MICROSECONDLY is a constant derived from SECONDLY, representing one second plus one microsecond.
MICROSECONDLY = SECONDLY + 1

# HOURS_PER_DAY is a constant representing the number of hours in a day.
HOURS_PER_DAY = 24.

# MIN_PER_HOUR is a constant representing the number of minutes in an hour.
MIN_PER_HOUR = 60.

# SEC_PER_MIN is a constant representing the number of seconds in a minute.
SEC_PER_MIN = 60.

# MONTHS_PER_YEAR is a constant representing the number of months in a year.
MONTHS_PER_YEAR = 12.

# DAYS_PER_WEEK is a constant representing the number of days in a week.
DAYS_PER_WEEK = 7.

# DAYS_PER_MONTH is a constant representing the average number of days in a month.
DAYS_PER_MONTH = 30.

# DAYS_PER_YEAR is a constant representing the average number of days in a year.
DAYS_PER_YEAR = 365.0

# MINUTES_PER_DAY is a constant representing the number of minutes in a day.
MINUTES_PER_DAY = MIN_PER_HOUR * HOURS_PER_DAY

# SEC_PER_HOUR is a constant representing the number of seconds in an hour.
SEC_PER_HOUR = SEC_PER_MIN * MIN_PER_HOUR

# SEC_PER_DAY is a constant representing the number of seconds in a day.
SEC_PER_DAY = SEC_PER_HOUR * HOURS_PER_DAY

# SEC_PER_WEEK is a constant representing the number of seconds in a week.
SEC_PER_WEEK = SEC_PER_DAY * DAYS_PER_WEEK

# MUSECONDS_PER_DAY is a constant representing the number of microseconds in a day.
MUSECONDS_PER_DAY = 1e6 * SEC_PER_DAY

# Constants for days of the week, used to represent weekdays in some contexts.
MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY = (
    MO, TU, WE, TH, FR, SA, SU)

# WEEKDAYS is a tuple containing all days of the week, used in various computations.
WEEKDAYS = (MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY)

# default epoch: passed to np.datetime64...
_epoch = None


def _reset_epoch_test_example():
    """
    Reset the Matplotlib date epoch so it can be set again.

    Only for use in tests and examples.
    """
    global _epoch
    _epoch = None


def set_epoch(epoch):
    """
    Set the epoch (origin for dates) for datetime calculations.

    The default epoch is :rc:`dates.epoch` (by default 1970-01-01T00:00).

    If microsecond accuracy is desired, the date being plotted needs to be
    within approximately 70 years of the epoch. Matplotlib internally
    represents dates as days since the epoch, so floating point dynamic
    range needs to be within a factor of 2^52.

    `~.dates.set_epoch` must be called before any dates are converted
    (i.e. near the import section) or a RuntimeError will be raised.

    See also :doc:`/gallery/ticks/date_precision_and_epochs`.

    Parameters
    ----------
    epoch : str
        valid UTC date parsable by `numpy.datetime64` (do not include
        timezone).

    """
    global _epoch
    if _epoch is not None:
        raise RuntimeError('set_epoch must be called before dates plotted.')
    _epoch = epoch


def get_epoch():
    """
    Get the epoch used by `.dates`.

    Returns
    -------
    epoch : str
        String for the epoch (parsable by `numpy.datetime64`).
    """
    global _epoch

    _epoch = mpl._val_or_rc(_epoch, 'date.epoch')
    return _epoch


def _dt64_to_ordinalf(d):
    """
    Convert `numpy.datetime64` or an `numpy.ndarray` of those types to
    Gregorian date as UTC float relative to the epoch (see `.get_epoch`).
    Roundoff is float64 precision.  Practically: microseconds for dates
    between 290301 BC, 294241 AD, milliseconds for larger dates
    (see `numpy.datetime64`).
    """

    # the "extra" ensures that we at least allow the dynamic range out to
    # seconds.  That should get out to +/-2e11 years.
    dseconds = d.astype('datetime64[s]')
    extra = (d - dseconds).astype('timedelta64[ns]')
    t0 = np.datetime64(get_epoch(), 's')
    dt = (dseconds - t0).astype(np.float64)
    dt += extra.astype(np.float64) / 1.0e9
    dt = dt / SEC_PER_DAY

    NaT_int = np.datetime64('NaT').astype(np.int64)
    d_int = d.astype(np.int64)
    dt[d_int == NaT_int] = np.nan
    return dt
def _from_ordinalf(x, tz=None):
    """
    Convert Gregorian float of the date, preserving hours, minutes,
    seconds and microseconds.  Return value is a `.datetime`.

    The input date *x* is a float in ordinal days at UTC, and the output will
    be the specified `.datetime` object corresponding to that time in
    timezone *tz*, or if *tz* is ``None``, in the timezone specified in
    :rc:`timezone`.
    """

    # 获取时区信息，如果未指定则使用默认时区
    tz = _get_tzinfo(tz)

    # 计算从纪元时开始的日期时间（以微秒为单位）
    dt = (np.datetime64(get_epoch()) +
          np.timedelta64(int(np.round(x * MUSECONDS_PER_DAY)), 'us'))
    
    # 检查计算得到的日期时间是否在有效范围内
    if dt < np.datetime64('0001-01-01') or dt >= np.datetime64('10000-01-01'):
        raise ValueError(f'Date ordinal {x} converts to {dt} (using '
                         f'epoch {get_epoch()}), but Matplotlib dates must be '
                          'between year 0001 and 9999.')
    
    # 将 datetime64 对象转换为 datetime 对象
    dt = dt.tolist()

    # 将 datetime64 对象设定为 UTC 时区
    dt = dt.replace(tzinfo=dateutil.tz.gettz('UTC'))
    
    # 将 datetime 对象转换为指定时区的时间
    dt = dt.astimezone(tz)
    
    # 修正由于浮点数舍入误差可能引起的微秒误差
    if np.abs(x) > 70 * 365:
        # 如果 x 值较大，则将微秒舍入到最接近的二十微秒
        ms = round(dt.microsecond / 20) * 20
        if ms == 1000000:
            dt = dt.replace(microsecond=0) + datetime.timedelta(seconds=1)
        else:
            dt = dt.replace(microsecond=ms)

    return dt


# 对 _from_ordinalf 进行向量化处理，使其能够处理 numpy 数组
_from_ordinalf_np_vectorized = np.vectorize(_from_ordinalf, otypes="O")

# 对 dateutil.parser.parse 进行向量化处理，使其能够处理 numpy 数组
_dateutil_parser_parse_np_vectorized = np.vectorize(dateutil.parser.parse)


def datestr2num(d, default=None):
    """
    Convert a date string to a datenum using `dateutil.parser.parse`.

    Parameters
    ----------
    d : str or sequence of str
        The dates to convert.

    default : datetime.datetime, optional
        The default date to use when fields are missing in *d*.
    """
    if isinstance(d, str):
        # 将单个日期字符串转换为日期时间对象，使用指定的默认日期
        dt = dateutil.parser.parse(d, default=default)
        return date2num(dt)
    else:
        if default is not None:
            # 将多个日期字符串转换为日期时间对象数组，使用指定的默认日期
            d = [date2num(dateutil.parser.parse(s, default=default))
                 for s in d]
            return np.asarray(d)
        d = np.asarray(d)
        if not d.size:
            return d
        return date2num(_dateutil_parser_parse_np_vectorized(d))


def date2num(d):
    """
    Convert datetime objects to Matplotlib dates.

    Parameters
    ----------
    d : `datetime.datetime` or `numpy.datetime64` or sequences of these

    Returns
    -------
    float or sequence of floats
        Number of days since the epoch.  See `.get_epoch` for the
        epoch, which can be changed by :rc:`date.epoch` or `.set_epoch`.  If
        the epoch is "1970-01-01T00:00:00" (default) then noon Jan 1 1970
        ("1970-01-01T12:00:00") returns 0.5.
    """
    # 从 cbook 模块中解包数据 d，以便处理像 Pandas 或 xarray 对象的情况
    d = cbook._unpack_to_numpy(d)
    
    # 将 d 转换为可迭代对象，并保存状态以便稍后解包
    iterable = np.iterable(d)
    if not iterable:
        # 如果 d 不可迭代，则将其封装为列表以便后续处理
        d = [d]
    
    # 检查 d 是否含有掩码
    masked = np.ma.is_masked(d)
    # 获取 d 的掩码
    mask = np.ma.getmask(d)
    # 将 d 转换为 NumPy 数组
    d = np.asarray(d)
    
    # 如果 d 的数据类型不是 datetime64，则进行转换
    if not np.issubdtype(d.dtype, np.datetime64):
        # 处理 datetime 数组
        if not d.size:
            # 处理空数组的情况
            return d
        # 获取第一个元素的时区信息
        tzi = getattr(d[0], 'tzinfo', None)
        if tzi is not None:
            # 将所有 datetime 对象转换为 naive datetime
            d = [dt.astimezone(UTC).replace(tzinfo=None) for dt in d]
            d = np.asarray(d)
        # 将 d 转换为微秒精度的 datetime64 数组
        d = d.astype('datetime64[us]')
    
    # 如果 d 含有掩码，则将其封装为掩码数组
    d = np.ma.masked_array(d, mask=mask) if masked else d
    # 将 datetime64 数组转换为序数数组
    d = _dt64_to_ordinalf(d)
    
    # 如果最初 d 是可迭代的，则返回处理后的对象；否则返回单个元素的处理结果
    return d if iterable else d[0]
# 将 Matplotlib 的日期数字转换为 datetime.datetime 对象或对象序列
def num2date(x, tz=None):
    tz = _get_tzinfo(tz)  # 获取时区信息
    return _from_ordinalf_np_vectorized(x, tz).tolist()


# 使用 NumPy 的向量化功能将日期序数转换为 datetime.timedelta 对象
_ordinalf_to_timedelta_np_vectorized = np.vectorize(
    lambda x: datetime.timedelta(days=x), otypes="O")


# 将天数转换为 datetime.timedelta 对象或对象序列
def num2timedelta(x):
    return _ordinalf_to_timedelta_np_vectorized(x).tolist()


# 返回一个等间距的 Matplotlib 日期序列
def drange(dstart, dend, delta):
    f1 = date2num(dstart)  # 起始日期转为 Matplotlib 格式
    f2 = date2num(dend)  # 结束日期转为 Matplotlib 格式
    step = delta.total_seconds() / SEC_PER_DAY  # 计算步长

    # 计算在指定步长下的日期数
    num = int(np.ceil((f2 - f1) / step))

    # 计算生成的时间间隔的结束时间
    dinterval_end = dstart + num * delta

    # 确保生成的时间间隔为半开放区间 [dstart, dend)
    if dinterval_end >= dend:
        dinterval_end -= delta
        num -= 1

    f2 = date2num(dinterval_end)  # 更新结束日期的 Matplotlib 格式
    return np.linspace(f1, f2, num + 1)


# 将输入的文本进行 TeX 格式包装
def _wrap_in_tex(text):
    p = r'([a-zA-Z]+)'
    ret_text = re.sub(p, r'}$\1$\\mathdefault{', text)  # 替换字母为 TeX 格式

    # 确保符号不会像二元操作符一样间隔
    ret_text = ret_text.replace('-', '{-}').replace(':', '{:}')
    # 避免数字之间的空格连接
    ret_text = ret_text.replace(' ', r'\;')
    # 将字符串 ret_text 包裹在 LaTeX 数学模式标记 $...$ 中
    ret_text = '$\\mathdefault{' + ret_text + '}$'
    # 替换掉没有内容的数学模式标记 $\\mathdefault{}$
    ret_text = ret_text.replace('$\\mathdefault{}$', '')
    # 返回处理后的字符串 ret_text
    return ret_text
## date tick locators and formatters ###

class DateFormatter(ticker.Formatter):
    """
    Format a tick (in days since the epoch) with a
    `~datetime.datetime.strftime` format string.
    """

    def __init__(self, fmt, tz=None, *, usetex=None):
        """
        Parameters
        ----------
        fmt : str
            `~datetime.datetime.strftime` format string
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        usetex : bool, default: :rc:`text.usetex`
            To enable/disable the use of TeX's math mode for rendering the
            results of the formatter.
        """
        # 设置时区信息
        self.tz = _get_tzinfo(tz)
        # 设置日期格式字符串
        self.fmt = fmt
        # 设置是否使用TeX渲染
        self._usetex = mpl._val_or_rc(usetex, 'text.usetex')

    def __call__(self, x, pos=0):
        # 将数值 x 转换为日期对象，应用格式化字符串 fmt，并返回结果
        result = num2date(x, self.tz).strftime(self.fmt)
        # 如果启用了TeX渲染，则将结果包装在TeX标记中返回
        return _wrap_in_tex(result) if self._usetex else result

    def set_tzinfo(self, tz):
        # 设置时区信息
        self.tz = _get_tzinfo(tz)


class ConciseDateFormatter(ticker.Formatter):
    """
    A `.Formatter` which attempts to figure out the best format to use for the
    date, and to make it as compact as possible, but still be complete. This is
    most useful when used with the `AutoDateLocator`::

    >>> locator = AutoDateLocator()
    >>> formatter = ConciseDateFormatter(locator)

    Parameters
    ----------
    locator : `.ticker.Locator`
        Locator that this axis is using.

    tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
        Ticks timezone, passed to `.dates.num2date`.

    formats : list of 6 strings, optional
        Format strings for 6 levels of tick labelling: mostly years,
        months, days, hours, minutes, and seconds.  Strings use
        the same format codes as `~datetime.datetime.strftime`.  Default is
        ``['%Y', '%b', '%d', '%H:%M', '%H:%M', '%S.%f']``

    zero_formats : list of 6 strings, optional
        Format strings for tick labels that are "zeros" for a given tick
        level.  For instance, if most ticks are months, ticks around 1 Jan 2005
        will be labeled "Dec", "2005", "Feb".  The default is
        ``['', '%Y', '%b', '%b-%d', '%H:%M', '%H:%M']``

    offset_formats : list of 6 strings, optional
        Format strings for the 6 levels that is applied to the "offset"
        string found on the right side of an x-axis, or top of a y-axis.
        Combined with the tick labels this should completely specify the
        date.  The default is::

            ['', '%Y', '%Y-%b', '%Y-%b-%d', '%Y-%b-%d', '%Y-%b-%d %H:%M']

    show_offset : bool, default: True
        Whether to show the offset or not.

    usetex : bool, default: :rc:`text.usetex`
        To enable/disable the use of TeX's math mode for rendering the results
        of the formatter.

    Examples
    --------
    See :doc:`/gallery/ticks/date_concise_formatter`
    ```
    # 导入必要的库：datetime 用于处理日期时间，matplotlib.dates 用于日期格式化
    import datetime
    import matplotlib.dates as mdates
    
    # 创建一个基准日期时间对象，设定为 2005 年 2 月 1 日
    base = datetime.datetime(2005, 2, 1)
    
    # 创建一个包含732个日期时间对象的数组，每个对象间隔2小时
    dates = np.array([base + datetime.timedelta(hours=(2 * i))
                      for i in range(732)])
    
    # 获取日期时间数组的长度
    N = len(dates)
    
    # 设置随机数种子，确保结果可复现
    np.random.seed(19680801)
    
    # 生成随机游走数据，作为 y 轴数据
    y = np.cumsum(np.random.randn(N))
    
    # 创建一个新的图形窗口和坐标轴对象
    fig, ax = plt.subplots(constrained_layout=True)
    
    # 创建一个自动日期定位器
    locator = mdates.AutoDateLocator()
    
    # 创建一个简洁的日期格式化器
    formatter = mdates.ConciseDateFormatter(locator)
    
    # 设置 x 轴的主要定位器为自动日期定位器
    ax.xaxis.set_major_locator(locator)
    
    # 设置 x 轴的主要格式化器为简洁的日期格式化器
    ax.xaxis.set_major_formatter(formatter)
    
    # 绘制日期与随机数据的折线图
    ax.plot(dates, y)
    
    # 设置图表标题
    ax.set_title('Concise Date Formatter')
    # 初始化函数，用于初始化日期标签的自动格式化
    def __init__(self, locator, tz=None, formats=None, offset_formats=None,
                 zero_formats=None, show_offset=True, *, usetex=None):
        """
        Autoformat the date labels.  The default format is used to form an
        initial string, and then redundant elements are removed.
        """
        # 保存定位器
        self._locator = locator
        # 保存时区信息
        self._tz = tz
        # 默认日期格式为年份 '%Y'
        self.defaultfmt = '%Y'

        # 每个级别的时间格式
        # 0: 年份, 1: 月份, 2: 天数,
        # 3: 小时, 4: 分钟, 5: 秒数
        if formats:
            if len(formats) != 6:
                raise ValueError('formats argument must be a list of '
                                 '6 format strings (or None)')
            self.formats = formats
        else:
            # 如果未提供格式列表，则使用默认格式列表
            self.formats = ['%Y',    # 大部分是年份
                            '%b',    # 大部分是月份
                            '%d',    # 大部分是天数
                            '%H:%M', # 小时:分钟
                            '%H:%M', # 小时:分钟
                            '%S.%f', # 秒.微秒
                            ]

        # 零刻度的格式
        # 这些是应该使用上一级别信息标记的刻度，例如：1月可以只标记为"Jan"，02:02:00 可以只标记为 02:02
        if zero_formats:
            if len(zero_formats) != 6:
                raise ValueError('zero_formats argument must be a list of '
                                 '6 format strings (or None)')
            self.zero_formats = zero_formats
        elif formats:
            # 如果提供了格式列表，则使用用户提供的格式来设置零刻度格式
            self.zero_formats = [''] + self.formats[:-1]
        else:
            # 否则，使用默认的零刻度格式
            self.zero_formats = [''] + self.formats[:-1]
            self.zero_formats[3] = '%b-%d'  # 小时刻度使用 '%b-%d' 格式

        # 偏移格式
        if offset_formats:
            if len(offset_formats) != 6:
                raise ValueError('offset_formats argument must be a list of '
                                 '6 format strings (or None)')
            self.offset_formats = offset_formats
        else:
            # 默认偏移格式列表
            self.offset_formats = ['',
                                   '%Y',
                                   '%Y-%b',
                                   '%Y-%b-%d',
                                   '%Y-%b-%d',
                                   '%Y-%b-%d %H:%M']
        self.offset_string = ''  # 初始化偏移字符串为空
        self.show_offset = show_offset  # 是否显示偏移量
        self._usetex = mpl._val_or_rc(usetex, 'text.usetex')

    # 调用实例时的操作，返回默认格式的日期字符串
    def __call__(self, x, pos=None):
        formatter = DateFormatter(self.defaultfmt, self._tz,
                                  usetex=self._usetex)
        return formatter(x, pos=pos)

    # 获取偏移量字符串
    def get_offset(self):
        return self.offset_string

    # 将数值转换为短格式日期时间字符串
    def format_data_short(self, value):
        return num2date(value, tz=self._tz).strftime('%Y-%m-%d %H:%M:%S')
# 自动日期格式化器，继承自 ticker.Formatter 类
class AutoDateFormatter(ticker.Formatter):
    """
    A `.Formatter` which attempts to figure out the best format to use.  This
    is most useful when used with the `AutoDateLocator`.

    `.AutoDateFormatter` has a ``.scale`` dictionary that maps tick scales (the
    interval in days between one major tick) to format strings; this dictionary
    defaults to ::

        self.scaled = {
            DAYS_PER_YEAR: rcParams['date.autoformatter.year'],  # 一年的天数，使用年份格式
            DAYS_PER_MONTH: rcParams['date.autoformatter.month'],  # 一个月的天数，使用月份格式
            1: rcParams['date.autoformatter.day'],  # 每天，使用日期格式
            1 / HOURS_PER_DAY: rcParams['date.autoformatter.hour'],  # 每小时，使用小时格式
            1 / MINUTES_PER_DAY: rcParams['date.autoformatter.minute'],  # 每分钟，使用分钟格式
            1 / SEC_PER_DAY: rcParams['date.autoformatter.second'],  # 每秒，使用秒格式
            1 / MUSECONDS_PER_DAY: rcParams['date.autoformatter.microsecond'],  # 每微秒，使用微秒格式
        }

    The formatter uses the format string corresponding to the lowest key in
    the dictionary that is greater or equal to the current scale.  Dictionary
    entries can be customized::

        locator = AutoDateLocator()
        formatter = AutoDateFormatter(locator)
        formatter.scaled[1/(24*60)] = '%M:%S' # only show min and sec

    Custom callables can also be used instead of format strings.  The following
    example shows how to use a custom format function to strip trailing zeros
    from decimal seconds and adds the date to the first ticklabel::

        def my_format_function(x, pos=None):
            x = matplotlib.dates.num2date(x)
            if pos == 0:
                fmt = '%D %H:%M:%S.%f'
            else:
                fmt = '%H:%M:%S.%f'
            label = x.strftime(fmt)
            label = label.rstrip("0")
            label = label.rstrip(".")
            return label

        formatter.scaled[1/(24*60)] = my_format_function
    """

    # This can be improved by providing some user-level direction on
    # how to choose the best format (precedence, etc.).

    # Perhaps a 'struct' that has a field for each time-type where a
    # zero would indicate "don't show" and a number would indicate
    # "show" with some sort of priority.  Same priorities could mean
    # show all with the same priority.

    # Or more simply, perhaps just a format string for each
    # possibility...
    # 初始化函数，设置日期标签的自动格式化。
    def __init__(self, locator, tz=None, defaultfmt='%Y-%m-%d', *,
                 usetex=None):
        """
        Autoformat the date labels.

        Parameters
        ----------
        locator : `.ticker.Locator`
            Locator that this axis is using.

        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.

        defaultfmt : str
            The default format to use if none of the values in ``self.scaled``
            are greater than the unit returned by ``locator._get_unit()``.

        usetex : bool, default: :rc:`text.usetex`
            To enable/disable the use of TeX's math mode for rendering the
            results of the formatter. If any entries in ``self.scaled`` are set
            as functions, then it is up to the customized function to enable or
            disable TeX's math mode itself.
        """
        # 设置定位器和时区属性
        self._locator = locator
        self._tz = tz
        # 设置默认的日期格式字符串
        self.defaultfmt = defaultfmt
        # 创建日期格式化器对象，使用默认格式和时区
        self._formatter = DateFormatter(self.defaultfmt, tz)
        # 从全局配置中获取参数设置
        rcParams = mpl.rcParams
        # 设置是否使用TeX渲染结果的标志
        self._usetex = mpl._val_or_rc(usetex, 'text.usetex')
        # 初始化缩放字典，映射不同时间单位到自动格式化参数
        self.scaled = {
            DAYS_PER_YEAR: rcParams['date.autoformatter.year'],
            DAYS_PER_MONTH: rcParams['date.autoformatter.month'],
            1: rcParams['date.autoformatter.day'],
            1 / HOURS_PER_DAY: rcParams['date.autoformatter.hour'],
            1 / MINUTES_PER_DAY: rcParams['date.autoformatter.minute'],
            1 / SEC_PER_DAY: rcParams['date.autoformatter.second'],
            1 / MUSECONDS_PER_DAY: rcParams['date.autoformatter.microsecond']
        }

    # 设置定位器的方法
    def _set_locator(self, locator):
        self._locator = locator

    # 调用实例对象时执行的方法，用于格式化日期标签
    def __call__(self, x, pos=None):
        try:
            # 获取定位器的单位刻度，并转换为浮点数
            locator_unit_scale = float(self._locator._get_unit())
        except AttributeError:
            # 如果定位器没有单位刻度属性，则默认为1
            locator_unit_scale = 1
        # 选择第一个比定位器单位刻度大的缩放比例对应的格式
        fmt = next((fmt for scale, fmt in sorted(self.scaled.items())
                    if scale >= locator_unit_scale),
                   self.defaultfmt)

        if isinstance(fmt, str):
            # 如果格式是字符串，创建日期格式化器对象，并用其格式化结果
            self._formatter = DateFormatter(fmt, self._tz, usetex=self._usetex)
            result = self._formatter(x, pos)
        elif callable(fmt):
            # 如果格式是可调用对象，直接调用并获取格式化结果
            result = fmt(x, pos)
        else:
            # 如果格式不是预期的类型，抛出类型错误异常
            raise TypeError(f'Unexpected type passed to {self!r}.')

        return result
class rrulewrapper:
    """
    A simple wrapper around a `dateutil.rrule` allowing flexible
    date tick specifications.
    """

    def __init__(self, freq, tzinfo=None, **kwargs):
        """
        Parameters
        ----------
        freq : {YEARLY, MONTHLY, WEEKLY, DAILY, HOURLY, MINUTELY, SECONDLY}
            Tick frequency. These constants are defined in `dateutil.rrule`,
            but they are accessible from `matplotlib.dates` as well.
        tzinfo : `datetime.tzinfo`, optional
            Time zone information. The default is None.
        **kwargs
            Additional keyword arguments are passed to the `dateutil.rrule`.
        """
        # 设置频率参数到 kwargs 中
        kwargs['freq'] = freq
        # 初始化基础时区信息
        self._base_tzinfo = tzinfo

        # 更新 rrule 实例
        self._update_rrule(**kwargs)

    def set(self, **kwargs):
        """Set parameters for an existing wrapper."""
        # 更新已有包装器的参数
        self._construct.update(kwargs)

        # 重新更新 rrule 实例
        self._update_rrule(**self._construct)

    def _update_rrule(self, **kwargs):
        # 获取基础时区信息
        tzinfo = self._base_tzinfo

        # rrule 不支持时区的良好支持，特别是 pytz 时区，最好使用 naive 时区，
        # 并在返回 datetime 对象后再附加时区信息
        if 'dtstart' in kwargs:
            dtstart = kwargs['dtstart']
            # 如果 dtstart 包含时区信息
            if dtstart.tzinfo is not None:
                # 如果基础时区信息为 None，则使用 dtstart 的时区信息
                if tzinfo is None:
                    tzinfo = dtstart.tzinfo
                else:
                    # 将 dtstart 转换为基础时区的时区信息
                    dtstart = dtstart.astimezone(tzinfo)

                # 去除 dtstart 的时区信息，以便重新创建时区感知的 rrule
                kwargs['dtstart'] = dtstart.replace(tzinfo=None)

        if 'until' in kwargs:
            until = kwargs['until']
            # 如果 until 包含时区信息
            if until.tzinfo is not None:
                # 如果基础时区信息不为 None，则将 until 转换为基础时区的时区信息
                if tzinfo is not None:
                    until = until.astimezone(tzinfo)
                else:
                    # 如果 dtstart 是 naive 而基础时区信息为 None，则抛出错误
                    raise ValueError('until cannot be aware if dtstart '
                                     'is naive and tzinfo is None')

                # 去除 until 的时区信息，以便重新创建时区感知的 rrule
                kwargs['until'] = until.replace(tzinfo=None)

        # 复制当前的 kwargs 作为构造参数，用于后续可能的更新
        self._construct = kwargs.copy()
        # 设置时区信息
        self._tzinfo = tzinfo
        # 创建 rrule 实例
        self._rrule = rrule(**self._construct)

    def _attach_tzinfo(self, dt, tzinfo):
        # 对于 pytz 时区，通过 "localize" 方法来附加时区信息
        if hasattr(tzinfo, 'localize'):
            return tzinfo.localize(dt, is_dst=True)

        # 对于其它时区，直接替换时区信息
        return dt.replace(tzinfo=tzinfo)
    def _aware_return_wrapper(self, f, returns_list=False):
        """Decorator function that allows rrule methods to handle tzinfo."""
        # 如果没有设置时区信息，则直接返回原始函数
        if self._tzinfo is None:
            return f

        # 所有的 datetime 参数必须是 naive 的。如果它们不是 naive，则在丢弃时区之前将其转换为 _tzinfo 时区。
        def normalize_arg(arg):
            if isinstance(arg, datetime.datetime) and arg.tzinfo is not None:
                if arg.tzinfo is not self._tzinfo:
                    arg = arg.astimezone(self._tzinfo)

                return arg.replace(tzinfo=None)

            return arg

        def normalize_args(args, kwargs):
            # 对传入的所有参数进行 normalize 处理
            args = tuple(normalize_arg(arg) for arg in args)
            kwargs = {kw: normalize_arg(arg) for kw, arg in kwargs.items()}

            return args, kwargs

        # 我们关心两种类型的函数 - 返回日期的和返回日期列表的。
        if not returns_list:
            # 对于返回日期的函数，包装后附加时区信息并返回
            def inner_func(*args, **kwargs):
                args, kwargs = normalize_args(args, kwargs)
                dt = f(*args, **kwargs)
                return self._attach_tzinfo(dt, self._tzinfo)
        else:
            # 对于返回日期列表的函数，包装后逐个附加时区信息并返回列表
            def inner_func(*args, **kwargs):
                args, kwargs = normalize_args(args, kwargs)
                dts = f(*args, **kwargs)
                return [self._attach_tzinfo(dt, self._tzinfo) for dt in dts]

        # 返回经过装饰后的函数，并保留原始函数的元信息
        return functools.wraps(f)(inner_func)

    def __getattr__(self, name):
        # 如果属性名存在于实例的字典中，则直接返回对应的值
        if name in self.__dict__:
            return self.__dict__[name]

        # 否则，尝试从 _rrule 对象中获取同名方法
        f = getattr(self._rrule, name)

        # 对于 'after' 和 'before' 方法，使用 _aware_return_wrapper 包装
        if name in {'after', 'before'}:
            return self._aware_return_wrapper(f)
        # 对于 'xafter', 'xbefore', 'between' 方法，使用 _aware_return_wrapper 包装，并指定返回列表
        elif name in {'xafter', 'xbefore', 'between'}:
            return self._aware_return_wrapper(f, returns_list=True)
        else:
            # 其他情况直接返回获取到的方法
            return f

    def __setstate__(self, state):
        # 更新对象状态
        self.__dict__.update(state)
class DateLocator(ticker.Locator):
    """
    Determines the tick locations when plotting dates.

    This class is subclassed by other Locators and
    is not meant to be used on its own.
    """
    hms0d = {'byhour': 0, 'byminute': 0, 'bysecond': 0}

    def __init__(self, tz=None):
        """
        Parameters
        ----------
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        self.tz = _get_tzinfo(tz)  # 初始化时设置时区信息

    def set_tzinfo(self, tz):
        """
        Set timezone info.

        Parameters
        ----------
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        self.tz = _get_tzinfo(tz)  # 设置时区信息的方法

    def datalim_to_dt(self):
        """Convert axis data interval to datetime objects."""
        dmin, dmax = self.axis.get_data_interval()  # 获取数据轴的数据间隔
        if dmin > dmax:
            dmin, dmax = dmax, dmin  # 确保dmin小于等于dmax

        return num2date(dmin, self.tz), num2date(dmax, self.tz)  # 将数值转换为对应时区的日期对象

    def viewlim_to_dt(self):
        """Convert the view interval to datetime objects."""
        vmin, vmax = self.axis.get_view_interval()  # 获取视图间隔
        if vmin > vmax:
            vmin, vmax = vmax, vmin  # 确保vmin小于等于vmax

        return num2date(vmin, self.tz), num2date(vmax, self.tz)  # 将数值转换为对应时区的日期对象

    def _get_unit(self):
        """
        Return how many days a unit of the locator is; used for
        intelligent autoscaling.
        """
        return 1  # 返回单位间隔，默认为1天

    def _get_interval(self):
        """
        Return the number of units for each tick.
        """
        return 1  # 返回每个刻度的单位数，默认为1

    def nonsingular(self, vmin, vmax):
        """
        Given the proposed upper and lower extent, adjust the range
        if it is too close to being singular (i.e. a range of ~0).
        """
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            # Except if there is no data, then use 1970 as default.
            return (date2num(datetime.date(1970, 1, 1)),
                    date2num(datetime.date(1970, 1, 2)))  # 如果vmin或vmax不是有限数，则返回默认的日期范围

        if vmax < vmin:
            vmin, vmax = vmax, vmin  # 确保vmin小于等于vmax

        unit = self._get_unit()  # 获取单位间隔
        interval = self._get_interval()  # 获取刻度的单位数
        if abs(vmax - vmin) < 1e-6:
            vmin -= 2 * unit * interval
            vmax += 2 * unit * interval  # 如果范围接近于0，则扩展范围

        return vmin, vmax  # 返回调整后的范围


class RRuleLocator(DateLocator):
    # use the dateutil rrule instance

    def __init__(self, o, tz=None):
        super().__init__(tz)
        self.rule = o  # 初始化时设置RRule实例

    def __call__(self):
        # if no data have been set, this will tank with a ValueError
        try:
            dmin, dmax = self.viewlim_to_dt()  # 获取视图的时间范围
        except ValueError:
            return []  # 如果获取失败，则返回空列表

        return self.tick_values(dmin, dmax)  # 返回计算的刻度值
    # 返回时间范围内的 tick 值列表
    def tick_values(self, vmin, vmax):
        # 使用传入的 vmin 和 vmax 创建重复规则的起始和结束时间
        start, stop = self._create_rrule(vmin, vmax)
        # 在规则中查找位于 start 和 stop 之间的日期
        dates = self.rule.between(start, stop, True)
        # 如果找不到日期，则返回 vmin 和 vmax 对应的 tick 值
        if len(dates) == 0:
            return date2num([vmin, vmax])
        # 否则，返回超出范围的日期的 tick 值
        return self.raise_if_exceeds(date2num(dates))

    # 创建重复规则的起始和结束时间，并返回它们
    def _create_rrule(self, vmin, vmax):
        # 计算时间差
        delta = relativedelta(vmax, vmin)

        # 尝试设置起始时间，如果出错则设置为一个日期时间对象的最小值
        try:
            start = vmin - delta
        except (ValueError, OverflowError):
            # 设置为最小的日期时间
            start = datetime.datetime(1, 1, 1, 0, 0, 0,
                                      tzinfo=datetime.timezone.utc)

        # 尝试设置结束时间，如果出错则设置为一个日期时间对象的最大值
        try:
            stop = vmax + delta
        except (ValueError, OverflowError):
            # 设置为最大的日期时间
            stop = datetime.datetime(9999, 12, 31, 23, 59, 59,
                                     tzinfo=datetime.timezone.utc)

        # 设置规则的起始时间和结束时间
        self.rule.set(dtstart=start, until=stop)

        return vmin, vmax

    # 获取时间单位，根据频率转换成对应的天数
    def _get_unit(self):
        # 获取规则的频率
        freq = self.rule._rrule._freq
        # 根据频率获取对应的时间单位
        return self.get_unit_generic(freq)

    # 根据频率获取通用的时间单位
    @staticmethod
    def get_unit_generic(freq):
        # 根据频率返回对应的天数或比例
        if freq == YEARLY:
            return DAYS_PER_YEAR
        elif freq == MONTHLY:
            return DAYS_PER_MONTH
        elif freq == WEEKLY:
            return DAYS_PER_WEEK
        elif freq == DAILY:
            return 1.0
        elif freq == HOURLY:
            return 1.0 / HOURS_PER_DAY
        elif freq == MINUTELY:
            return 1.0 / MINUTES_PER_DAY
        elif freq == SECONDLY:
            return 1.0 / SEC_PER_DAY
        else:
            # 如果频率不匹配，则返回错误标志
            return -1  # 或者应该返回 '1'？

    # 获取规则的间隔
    def _get_interval(self):
        # 获取规则的间隔值
        return self.rule._rrule._interval
class AutoDateLocator(DateLocator):
    """
    On autoscale, this class picks the best `DateLocator` to set the view
    limits and the tick locations.

    Attributes
    ----------
    intervald : dict

        Mapping of tick frequencies to multiples allowed for that ticking.
        The default is ::

            self.intervald = {
                YEARLY  : [1, 2, 4, 5, 10, 20, 40, 50, 100, 200, 400, 500,
                           1000, 2000, 4000, 5000, 10000],
                MONTHLY : [1, 2, 3, 4, 6],
                DAILY   : [1, 2, 3, 7, 14, 21],
                HOURLY  : [1, 2, 3, 4, 6, 12],
                MINUTELY: [1, 5, 10, 15, 30],
                SECONDLY: [1, 5, 10, 15, 30],
                MICROSECONDLY: [1, 2, 5, 10, 20, 50, 100, 200, 500,
                                1000, 2000, 5000, 10000, 20000, 50000,
                                100000, 200000, 500000, 1000000],
            }

        where the keys are defined in `dateutil.rrule`.

        The interval is used to specify multiples that are appropriate for
        the frequency of ticking. For instance, every 7 days is sensible
        for daily ticks, but for minutes/seconds, 15 or 30 make sense.

        When customizing, you should only modify the values for the existing
        keys. You should not add or delete entries.

        Example for forcing ticks every 3 hours::

            locator = AutoDateLocator()
            locator.intervald[HOURLY] = [3]  # only show every 3 hours
    """

    def __call__(self):
        # docstring inherited
        # 将视图限制转换为日期时间格式，并获取适当的定位器对象
        dmin, dmax = self.viewlim_to_dt()
        locator = self.get_locator(dmin, dmax)
        # 返回由定位器对象生成的刻度值
        return locator()

    def tick_values(self, vmin, vmax):
        # 调用父类方法获取特定视图范围内的刻度值
        return self.get_locator(vmin, vmax).tick_values(vmin, vmax)

    def nonsingular(self, vmin, vmax):
        # 处理任意的视图范围，确保日期单位的合理缩放
        # 默认情况下，非奇异日期绘图以大约4年的周期。
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            # 如果没有数据，则使用1970年1月1日作为默认日期范围
            return (date2num(datetime.date(1970, 1, 1)),
                    date2num(datetime.date(1970, 1, 2)))
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        if vmin == vmax:
            # 调整最小和最大值，确保它们不相等，使用2年的时间跨度
            vmin = vmin - DAYS_PER_YEAR * 2
            vmax = vmax + DAYS_PER_YEAR * 2
        return vmin, vmax

    def _get_unit(self):
        # 根据频率设置日期单位
        if self._freq in [MICROSECONDLY]:
            return 1. / MUSECONDS_PER_DAY
        else:
            return RRuleLocator.get_unit_generic(self._freq)

class YearLocator(RRuleLocator):
    """
    Make ticks on a given day of each year that is a multiple of base.

    Examples::

      # Tick every year on Jan 1st
      locator = YearLocator()

      # Tick every 5 years on July 4th
      locator = YearLocator(5, month=7, day=4)
    """
    def __init__(self, base=1, month=1, day=1, tz=None):
        """
        Parameters
        ----------
        base : int, default: 1
            Mark ticks every *base* years.
        month : int, default: 1
            The month on which to place the ticks, starting from 1. Default is
            January.
        day : int, default: 1
            The day on which to place the ticks.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        # 创建一个年复合规则对象，设置年间隔为base，月份为month，日期为day，其余时间参数使用self.hms0d
        rule = rrulewrapper(YEARLY, interval=base, bymonth=month,
                            bymonthday=day, **self.hms0d)
        # 调用父类的构造函数，传入规则对象rule和时区tz
        super().__init__(rule, tz=tz)
        # 初始化self.base为一个由ticker._Edge_integer类处理的整数base，最小值为0
        self.base = ticker._Edge_integer(base, 0)

    def _create_rrule(self, vmin, vmax):
        # 计算年份的下限和上限，确保它们是base的倍数以便创建适当间隔的刻度
        ymin = max(self.base.le(vmin.year) * self.base.step, 1)
        ymax = min(self.base.ge(vmax.year) * self.base.step, 9999)

        # 从self.rule中提取构建规则的字典c
        c = self.rule._construct
        # 创建一个包含年份ymin、月份c中的bymonth或默认为1、日期c中的bymonthday或默认为1、时分秒为0的字典replace
        replace = {'year': ymin,
                   'month': c.get('bymonth', 1),
                   'day': c.get('bymonthday', 1),
                   'hour': 0, 'minute': 0, 'second': 0}

        # 使用vmin的日期，替换为replace中指定的日期部分，作为开始时间start
        start = vmin.replace(**replace)
        # 将开始时间start的年份替换为ymax，得到结束时间stop
        stop = start.replace(year=ymax)
        # 设置self.rule的起始时间为start，结束时间为stop
        self.rule.set(dtstart=start, until=stop)

        # 返回起始时间start和结束时间stop
        return start, stop
class MonthLocator(RRuleLocator):
    """
    Make ticks on occurrences of each month, e.g., 1, 3, 12.
    """

    def __init__(self, bymonth=None, bymonthday=1, interval=1, tz=None):
        """
        Parameters
        ----------
        bymonth : int or list of int, default: all months
            Ticks will be placed on every month in *bymonth*. Default is
            ``range(1, 13)``, i.e. every month.
        bymonthday : int, default: 1
            The day on which to place the ticks.
        interval : int, default: 1
            The interval between each iteration. For example, if
            ``interval=2``, mark every second occurrence.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        # 如果未指定 bymonth，则默认为所有月份（1 到 12 月）
        if bymonth is None:
            bymonth = range(1, 13)

        # 创建一个基于月份的重复规则对象
        rule = rrulewrapper(MONTHLY, bymonth=bymonth, bymonthday=bymonthday,
                            interval=interval, **self.hms0d)
        # 调用父类的初始化方法，传入规则对象和时区信息
        super().__init__(rule, tz=tz)


class WeekdayLocator(RRuleLocator):
    """
    Make ticks on occurrences of each weekday.
    """

    def __init__(self, byweekday=1, interval=1, tz=None):
        """
        Parameters
        ----------
        byweekday : int or list of int, default: all days
            Ticks will be placed on every weekday in *byweekday*. Default is
            every day.

            Elements of *byweekday* must be one of MO, TU, WE, TH, FR, SA,
            SU, the constants from :mod:`dateutil.rrule`, which have been
            imported into the :mod:`matplotlib.dates` namespace.
        interval : int, default: 1
            The interval between each iteration. For example, if
            ``interval=2``, mark every second occurrence.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        # 创建一个基于工作日的重复规则对象
        rule = rrulewrapper(DAILY, byweekday=byweekday,
                            interval=interval, **self.hms0d)
        # 调用父类的初始化方法，传入规则对象和时区信息
        super().__init__(rule, tz=tz)


class DayLocator(RRuleLocator):
    """
    Make ticks on occurrences of each day of the month.  For example,
    1, 15, 30.
    """
    def __init__(self, bymonthday=None, interval=1, tz=None):
        """
        Parameters
        ----------
        bymonthday : int or list of int, default: all days
            Ticks will be placed on every day in *bymonthday*. Default is
            ``bymonthday=range(1, 32)``, i.e., every day of the month.
        interval : int, default: 1
            The interval between each iteration. For example, if
            ``interval=2``, mark every second occurrence.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        
        # 检查 interval 是否为整数且大于 0，若不是则抛出异常
        if interval != int(interval) or interval < 1:
            raise ValueError("interval must be an integer greater than 0")
        
        # 如果 bymonthday 为 None，则设定为默认值 range(1, 32)，即一个月中的每一天
        if bymonthday is None:
            bymonthday = range(1, 32)

        # 使用指定的 bymonthday、interval 和 self.hms0d 参数创建一个日规则对象
        rule = rrulewrapper(DAILY, bymonthday=bymonthday,
                            interval=interval, **self.hms0d)
        
        # 调用父类的构造方法，传入创建的规则对象和时区信息 tz
        super().__init__(rule, tz=tz)
class HourLocator(RRuleLocator):
    """
    Make ticks on occurrences of each hour.
    """
    def __init__(self, byhour=None, interval=1, tz=None):
        """
        Parameters
        ----------
        byhour : int or list of int, default: all hours
            Ticks will be placed on every hour in *byhour*. Default is
            ``byhour=range(24)``, i.e., every hour.
        interval : int, default: 1
            The interval between each iteration. For example, if
            ``interval=2``, mark every second occurrence.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        # 如果没有指定 byhour 参数，则默认为包含所有小时的范围
        if byhour is None:
            byhour = range(24)

        # 创建一个 HOURLY 规则对象，指定了每小时触发的规则
        rule = rrulewrapper(HOURLY, byhour=byhour, interval=interval,
                            byminute=0, bysecond=0)
        # 调用父类 RRuleLocator 的初始化方法，传递规则和时区参数
        super().__init__(rule, tz=tz)


class MinuteLocator(RRuleLocator):
    """
    Make ticks on occurrences of each minute.
    """
    def __init__(self, byminute=None, interval=1, tz=None):
        """
        Parameters
        ----------
        byminute : int or list of int, default: all minutes
            Ticks will be placed on every minute in *byminute*. Default is
            ``byminute=range(60)``, i.e., every minute.
        interval : int, default: 1
            The interval between each iteration. For example, if
            ``interval=2``, mark every second occurrence.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        # 如果没有指定 byminute 参数，则默认为包含所有分钟的范围
        if byminute is None:
            byminute = range(60)

        # 创建一个 MINUTELY 规则对象，指定了每分钟触发的规则
        rule = rrulewrapper(MINUTELY, byminute=byminute, interval=interval,
                            bysecond=0)
        # 调用父类 RRuleLocator 的初始化方法，传递规则和时区参数
        super().__init__(rule, tz=tz)


class SecondLocator(RRuleLocator):
    """
    Make ticks on occurrences of each second.
    """
    def __init__(self, bysecond=None, interval=1, tz=None):
        """
        Parameters
        ----------
        bysecond : int or list of int, default: all seconds
            Ticks will be placed on every second in *bysecond*. Default is
            ``bysecond = range(60)``, i.e., every second.
        interval : int, default: 1
            The interval between each iteration. For example, if
            ``interval=2``, mark every second occurrence.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        # 如果没有指定 bysecond 参数，则默认为包含所有秒数的范围
        if bysecond is None:
            bysecond = range(60)

        # 创建一个 SECONDLY 规则对象，指定了每秒触发的规则
        rule = rrulewrapper(SECONDLY, bysecond=bysecond, interval=interval)
        # 调用父类 RRuleLocator 的初始化方法，传递规则和时区参数
        super().__init__(rule, tz=tz)


class MicrosecondLocator(DateLocator):
    """
    Make ticks on regular intervals of one or more microsecond(s).
    """

    # 此类暂未实现 __init__ 方法，继承自 DateLocator 的行为
    """
    A custom ticker for Matplotlib supporting sub-microsecond resolution time plots.

    This ticker converts time values to appropriate tick positions on the axis.

    """

    def __init__(self, interval=1, tz=None):
        """
        Parameters
        ----------
        interval : int, default: 1
            The interval between each tick mark.
            E.g., if interval=2, marks every second occurrence.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Timezone for the ticks. If a string, it's passed to `dateutil.tz`.
        """
        super().__init__(tz=tz)
        self._interval = interval
        self._wrapped_locator = ticker.MultipleLocator(interval)

    def set_axis(self, axis):
        """
        Set the axis for the wrapped locator and update the axis for the super class.

        Parameters
        ----------
        axis : `matplotlib.axis.Axis`
            The axis to set.
        """
        self._wrapped_locator.set_axis(axis)
        return super().set_axis(axis)

    def __call__(self):
        """
        Compute the tick values based on the view limits.

        Returns
        -------
        list
            List of tick values.
        """
        # if no data have been set, this will tank with a ValueError
        try:
            dmin, dmax = self.viewlim_to_dt()
        except ValueError:
            return []

        return self.tick_values(dmin, dmax)

    def tick_values(self, vmin, vmax):
        """
        Compute tick values in microseconds since the epoch.

        Parameters
        ----------
        vmin : float
            Minimum value of the axis.
        vmax : float
            Maximum value of the axis.

        Returns
        -------
        ticks : np.ndarray
            Array of tick values.
        """
        nmin, nmax = date2num((vmin, vmax))
        t0 = np.floor(nmin)
        nmax = nmax - t0
        nmin = nmin - t0
        nmin *= MUSECONDS_PER_DAY
        nmax *= MUSECONDS_PER_DAY

        ticks = self._wrapped_locator.tick_values(nmin, nmax)

        ticks = ticks / MUSECONDS_PER_DAY + t0
        return ticks

    def _get_unit(self):
        """
        Get the time unit as the reciprocal of microseconds per day.

        Returns
        -------
        float
            Time unit in days.
        """
        # docstring inherited
        return 1. / MUSECONDS_PER_DAY

    def _get_interval(self):
        """
        Get the interval between each tick mark.

        Returns
        -------
        int
            Interval between tick marks.
        """
        # docstring inherited
        return self._interval
class DateConverter(units.ConversionInterface):
    """
    Converter for `datetime.date` and `datetime.datetime` data, or for
    date/time data represented as it would be converted by `date2num`.

    The 'unit' tag for such data is None or a `~datetime.tzinfo` instance.
    """

    def __init__(self, *, interval_multiples=True):
        # 初始化日期转换器，设定是否使用间隔的倍数
        self._interval_multiples = interval_multiples
        # 调用父类的初始化方法
        super().__init__()

    def axisinfo(self, unit, axis):
        """
        Return the `~matplotlib.units.AxisInfo` for *unit*.

        *unit* is a `~datetime.tzinfo` instance or None.
        The *axis* argument is required but not used.
        """
        # 获取时区信息
        tz = unit

        # 使用AutoDateLocator根据时区和是否使用间隔的倍数创建主要定位器
        majloc = AutoDateLocator(tz=tz,
                                 interval_multiples=self._interval_multiples)
        # 使用AutoDateFormatter创建主要格式化器
        majfmt = AutoDateFormatter(majloc, tz=tz)
        # 设定最小日期和最大日期为1970年1月1日到1970年1月2日
        datemin = datetime.date(1970, 1, 1)
        datemax = datetime.date(1970, 1, 2)

        # 返回AxisInfo对象，包含主要定位器、主要格式化器、空字符串作为标签、默认限制日期范围
        return units.AxisInfo(majloc=majloc, majfmt=majfmt, label='',
                              default_limits=(datemin, datemax))

    @staticmethod
    def convert(value, unit, axis):
        """
        If *value* is not already a number or sequence of numbers, convert it
        with `date2num`.

        The *unit* and *axis* arguments are not used.
        """
        # 使用date2num将value转换为数值（如果尚未是数值或数值序列）
        return date2num(value)

    @staticmethod
    def default_units(x, axis):
        """
        Return the `~datetime.tzinfo` instance of *x* or of its first element,
        or None
        """
        # 如果x是np.ndarray，将其展平为一维数组
        if isinstance(x, np.ndarray):
            x = x.ravel()

        # 获取x的第一个有限元素
        try:
            x = cbook._safe_first_finite(x)
        except (TypeError, StopIteration):
            pass

        # 尝试返回x的时区信息
        try:
            return x.tzinfo
        except AttributeError:
            pass
        # 如果无法获取时区信息，则返回None
        return None


class ConciseDateConverter(DateConverter):
    # docstring inherited

    def __init__(self, formats=None, zero_formats=None, offset_formats=None,
                 show_offset=True, *, interval_multiples=True):
        # 初始化精简日期转换器，设定日期格式、零值格式、偏移格式、是否显示偏移以及是否使用间隔的倍数
        self._formats = formats
        self._zero_formats = zero_formats
        self._offset_formats = offset_formats
        self._show_offset = show_offset
        self._interval_multiples = interval_multiples
        # 调用父类DateConverter的初始化方法
        super().__init__()

    def axisinfo(self, unit, axis):
        # docstring inherited
        # 获取时区信息
        tz = unit
        # 使用AutoDateLocator根据时区和是否使用间隔的倍数创建主要定位器
        majloc = AutoDateLocator(tz=tz,
                                 interval_multiples=self._interval_multiples)
        # 使用ConciseDateFormatter创建主要格式化器，设定日期格式、零值格式、偏移格式、是否显示偏移
        majfmt = ConciseDateFormatter(majloc, tz=tz, formats=self._formats,
                                      zero_formats=self._zero_formats,
                                      offset_formats=self._offset_formats,
                                      show_offset=self._show_offset)
        # 设定最小日期和最大日期为1970年1月1日到1970年1月2日
        datemin = datetime.date(1970, 1, 1)
        datemax = datetime.date(1970, 1, 2)
        # 返回AxisInfo对象，包含主要定位器、主要格式化器、空字符串作为标签、默认限制日期范围
        return units.AxisInfo(majloc=majloc, majfmt=majfmt, label='',
                              default_limits=(datemin, datemax))


class _SwitchableDateConverter:
    # This class definition is empty and does not require annotations.
    pass
    """
    Helper converter-like object that generates and dispatches to
    temporary ConciseDateConverter or DateConverter instances based on
    :rc:`date.converter` and :rc:`date.interval_multiples`.
    """
    
    # 静态方法，用于获取合适的日期转换器实例，根据配置决定使用 ConciseDateConverter 或 DateConverter
    @staticmethod
    def _get_converter():
        # 从全局配置中获取日期转换器的类别
        converter_cls = {
            "concise": ConciseDateConverter, "auto": DateConverter}[
                mpl.rcParams["date.converter"]]
        # 获取日期间隔的倍数配置
        interval_multiples = mpl.rcParams["date.interval_multiples"]
        # 返回相应类型的转换器实例，并传入间隔倍数配置
        return converter_cls(interval_multiples=interval_multiples)
    
    # 返回由 _get_converter 方法返回的转换器实例的 axisinfo 方法的结果
    def axisinfo(self, *args, **kwargs):
        return self._get_converter().axisinfo(*args, **kwargs)
    
    # 返回由 _get_converter 方法返回的转换器实例的 default_units 方法的结果
    def default_units(self, *args, **kwargs):
        return self._get_converter().default_units(*args, **kwargs)
    
    # 返回由 _get_converter 方法返回的转换器实例的 convert 方法的结果
    def convert(self, *args, **kwargs):
        return self._get_converter().convert(*args, **kwargs)
# 将 np.datetime64 对象注册到 units.registry 中，并使用 _SwitchableDateConverter 进行转换
units.registry[np.datetime64] = \
    # 将 datetime.date 对象也注册到 units.registry 中，并使用相同的转换器
    units.registry[datetime.date] = \
    # 将 datetime.datetime 对象也注册到 units.registry 中，并仍然使用相同的转换器
    units.registry[datetime.datetime] = \
    # 创建一个 _SwitchableDateConverter 实例，用于处理不同日期类型的转换
    _SwitchableDateConverter()
```