# `D:\src\scipysrc\pandas\pandas\core\indexes\datetimes.py`

```
@inherit_names(
    # 继承 DatetimeArray._field_ops 中的方法和属性
    DatetimeArray._field_ops
)
    # 列表推导式，用于生成 DatetimeArray 类中的特定方法列表，排除了 "tz_localize", "tz_convert", "strftime"
    [
        method
        for method in DatetimeArray._datetimelike_methods
        if method not in ("tz_localize", "tz_convert", "strftime")
    ],
    # DatetimeArray 类，作为列表推导式的迭代对象
    DatetimeArray,
    # wrap 参数设置为 True，用于指示是否要包装生成的函数
    wrap=True,
# 定义 DatetimeIndex 类，表示不可变的 datetime64 数据的类似 ndarray 结构
@inherit_names(["is_normalized"], DatetimeArray, cache=True)
# 继承 DatetimeArray 类的 is_normalized 属性，并缓存结果
@inherit_names(
    [
        "tz",
        "tzinfo",
        "dtype",
        "to_pydatetime",
        "date",
        "time",
        "timetz",
        "std",
    ]
    + DatetimeArray._bool_ops,
    DatetimeArray,
)
# 继承 DatetimeArray 类的多个属性和方法，包括时区信息、数据类型、日期时间转换等，并缓存结果
class DatetimeIndex(DatetimeTimedeltaMixin):
    """
    Immutable ndarray-like of datetime64 data.

    Represented internally as int64, and which can be boxed to Timestamp objects
    that are subclasses of datetime and carry metadata.

    .. versionchanged:: 2.0.0
        The various numeric date/time attributes (:attr:`~DatetimeIndex.day`,
        :attr:`~DatetimeIndex.month`, :attr:`~DatetimeIndex.year` etc.) now have dtype
        ``int32``. Previously they had dtype ``int64``.

    Parameters
    ----------
    data : array-like (1-dimensional)
        Datetime-like data to construct index with.
    freq : str or pandas offset object, optional
        One of pandas date offset strings or corresponding objects. The string
        'infer' can be passed in order to set the frequency of the index as the
        inferred frequency upon creation.
    tz : zoneinfo.ZoneInfo, pytz.timezone, dateutil.tz.tzfile, datetime.tzinfo or str
        Set the Timezone of the data.
    ambiguous : 'infer', bool-ndarray, 'NaT', default 'raise'
        When clocks moved backward due to DST, ambiguous times may arise.
        For example in Central European Time (UTC+01), when going from 03:00
        DST to 02:00 non-DST, 02:30:00 local time occurs both at 00:30:00 UTC
        and at 01:30:00 UTC. In such a situation, the `ambiguous` parameter
        dictates how ambiguous times should be handled.

        - 'infer' will attempt to infer fall dst-transition hours based on
          order
        - bool-ndarray where True signifies a DST time, False signifies a
          non-DST time (note that this flag is only applicable for ambiguous
          times)
        - 'NaT' will return NaT where there are ambiguous times
        - 'raise' will raise an AmbiguousTimeError if there are ambiguous times.
    dayfirst : bool, default False
        If True, parse dates in `data` with the day first order.
    yearfirst : bool, default False
        If True parse dates in `data` with the year first order.
    dtype : numpy.dtype or DatetimeTZDtype or str, default None
        Note that the only NumPy dtype allowed is `datetime64[ns]`.
    copy : bool, default False
        Make a copy of input ndarray.
    name : label, default None
        Name to be stored in the index.

    Attributes
    ----------
    year
    month
    day
    hour
    minute
    second
    microsecond
    nanosecond
    date
    time
    timetz
    dayofyear
    day_of_year
    dayofweek
    day_of_week
    weekday
    quarter
    tz
    freq
    freqstr
    is_month_start
    is_month_end
    is_quarter_start
    is_quarter_end
    is_year_start
    is_year_end
    is_leap_year
    inferred_freq

    Methods
    -------

    """
    # 方法用于标准化（normalize）时间戳，使其在时间序列中具有一致的表示形式
    normalize
    # 将时间戳格式化为指定格式的字符串表示
    strftime
    # 快照（snap）时间戳到指定的时间粒度
    snap
    # 将时间戳转换到指定时区
    tz_convert
    # 本地化（localize）没有时区信息的时间戳到指定时区
    tz_localize
    # 将时间戳进行四舍五入
    round
    # 将时间戳向下取整
    floor
    # 将时间戳向上取整
    ceil
    # 将时间戳转换为周期（Period）对象
    to_period
    # 将时间戳转换为 Python datetime 对象
    to_pydatetime
    # 将时间戳转换为 Pandas Series 对象
    to_series
    # 将时间戳转换为 Pandas DataFrame 对象
    to_frame
    # 获取月份的名称
    month_name
    # 获取星期几的名称
    day_name
    # 计算时间戳序列的均值
    mean
    # 计算时间戳序列的标准差
    std

    See Also
    --------
    Index : Pandas 的基本索引类型。
    TimedeltaIndex : 存储 timedelta64 数据的索引。
    PeriodIndex : 存储 Period 数据的索引。
    to_datetime : 将参数转换为 datetime 对象。
    date_range : 创建固定频率的 DatetimeIndex。

    Notes
    -----
    若要了解更多频率字符串的信息，请参阅 :ref:`此链接<timeseries.offset_aliases>`。

    Examples
    --------
    >>> idx = pd.DatetimeIndex(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])
    >>> idx
    DatetimeIndex(['2020-01-01 10:00:00+00:00', '2020-02-01 11:00:00+00:00'],
    dtype='datetime64[s, UTC]', freq=None)
    """

    # 定义类型为 "datetimeindex"
    _typ = "datetimeindex"

    # 数据类为 DatetimeArray
    _data_cls = DatetimeArray
    # 支持部分字符串索引
    _supports_partial_string_indexing = True

    @property
    def _engine_type(self) -> type[libindex.DatetimeEngine]:
        return libindex.DatetimeEngine

    # 数据为 DatetimeArray 类型
    _data: DatetimeArray
    # 值为 DatetimeArray 类型
    _values: DatetimeArray
    # 时区信息，可能为 None
    tz: dt.tzinfo | None

    # --------------------------------------------------------------------
    # 调用 DatetimeArray 的方法并封装结果的方法

    @doc(DatetimeArray.strftime)
    def strftime(self, date_format) -> Index:
        # 调用 DatetimeArray 的 strftime 方法格式化时间戳数组
        arr = self._data.strftime(date_format)
        # 返回格式化后的索引对象，保留原始名称和对象类型为 object
        return Index(arr, name=self.name, dtype=object)

    @doc(DatetimeArray.tz_convert)
    def tz_convert(self, tz) -> Self:
        # 调用 DatetimeArray 的 tz_convert 方法转换时区
        arr = self._data.tz_convert(tz)
        # 返回转换时区后的新对象，保留原始名称和引用
        return type(self)._simple_new(arr, name=self.name, refs=self._references)

    @doc(DatetimeArray.tz_localize)
    def tz_localize(
        self,
        tz,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self:
        # 调用 DatetimeArray 的 tz_localize 方法本地化时区
        arr = self._data.tz_localize(tz, ambiguous, nonexistent)
        # 返回本地化时区后的新对象，保留原始名称
        return type(self)._simple_new(arr, name=self.name)

    @doc(DatetimeArray.to_period)
    def to_period(self, freq=None) -> PeriodIndex:
        from pandas.core.indexes.api import PeriodIndex

        # 调用 DatetimeArray 的 to_period 方法转换为周期对象
        arr = self._data.to_period(freq)
        # 返回新的周期索引对象，保留原始名称
        return PeriodIndex._simple_new(arr, name=self.name)

    @doc(DatetimeArray.to_julian_date)
    def to_julian_date(self) -> Index:
        # 调用 DatetimeArray 的 to_julian_date 方法转换为儒略日
        arr = self._data.to_julian_date()
        # 返回新的索引对象，保留原始名称
        return Index._simple_new(arr, name=self.name)

    @doc(DatetimeArray.isocalendar)
    def isocalendar(self) -> DataFrame:
        # 调用 DatetimeArray 的 isocalendar 方法获取 ISO 日历信息
        df = self._data.isocalendar()
        # 将结果设置为索引并返回 DataFrame 对象
        return df.set_index(self)

    @cache_readonly
    def _resolution_obj(self) -> Resolution:
        # 返回时间分辨率对象
        return self._data._resolution_obj

    # --------------------------------------------------------------------
    # 构造函数
    # 创建一个新的实例方法，用于初始化对象
    def __new__(
        cls,
        data=None,
        freq: Frequency | lib.NoDefault = lib.no_default,
        tz=lib.no_default,
        ambiguous: TimeAmbiguous = "raise",
        dayfirst: bool = False,
        yearfirst: bool = False,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable | None = None,
    ) -> Self:
        # 如果数据是标量，则抛出错误
        if is_scalar(data):
            cls._raise_scalar_data_error(data)

        # - 上述所有情况的检查在到达此处之前都会返回/抛出 - #

        # 从数据中提取名称
        name = maybe_extract_name(name, data, cls)

        # 快速路径，当数据是DatetimeArray且freq、tz和dtype均为默认时
        if (
            isinstance(data, DatetimeArray)
            and freq is lib.no_default
            and tz is lib.no_default
            and dtype is None
        ):
            # 如果需要复制数据，则复制数据
            if copy:
                data = data.copy()
            # 使用简单的新建方法创建实例并返回
            return cls._simple_new(data, name=name)

        # 从序列非严格地创建DatetimeArray对象
        dtarr = DatetimeArray._from_sequence_not_strict(
            data,
            dtype=dtype,
            copy=copy,
            tz=tz,
            freq=freq,
            dayfirst=dayfirst,
            yearfirst=yearfirst,
            ambiguous=ambiguous,
        )
        refs = None
        # 如果不需要复制且数据是Index或者ABCSeries的实例，则获取其引用
        if not copy and isinstance(data, (Index, ABCSeries)):
            refs = data._references

        # 使用简单的新建方法创建实例并返回
        subarr = cls._simple_new(dtarr, name=name, refs=refs)
        return subarr

    # --------------------------------------------------------------------

    @cache_readonly
    def _is_dates_only(self) -> bool:
        """
        返回一个布尔值，指示是否仅包含日期（没有时区信息）

        Returns
        -------
        bool
        """
        # 如果频率是Tick对象
        if isinstance(self.freq, Tick):
            delta = Timedelta(self.freq)

            # 如果delta不是整天的时间间隔，则返回False
            if delta % dt.timedelta(days=1) != dt.timedelta(days=0):
                return False

        # 否则调用内部值的_is_dates_only方法并返回其结果
        return self._values._is_dates_only

    # 返回一个字典，用于对象的序列化和反序列化
    def __reduce__(self):
        d = {"data": self._data, "name": self.name}
        return _new_DatetimeIndex, (type(self), d), None

    # 检查给定的dtype对象是否可以与当前对象的值进行比较
    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        """
        是否可以比较给定dtype的值与当前对象的值
        """
        # 如果对象有时区信息，则可以与DatetimeTZDtype类型进行比较
        if self.tz is not None:
            return isinstance(dtype, DatetimeTZDtype)
        # 如果对象没有时区信息，则只能与tznaive类型进行比较
        return lib.is_np_dtype(dtype, "M")

    # --------------------------------------------------------------------
    # 渲染方法

    @cache_readonly
    def _formatter_func(self):
        # 注意，这与DatetimeIndexOpsMixin中的方法等效，
        # 但使用了可能缓存的self._is_dates_only而不是重新计算它。
        from pandas.io.formats.format import get_format_datetime64

        # 获取日期时间格式化函数
        formatter = get_format_datetime64(is_dates_only=self._is_dates_only)
        return lambda x: f"'{formatter(x)}'"
    # --------------------------------------------------------------------
    # Set Operation Methods

    def _can_range_setop(self, other) -> bool:
        # GH 46702: If self or other have non-UTC timezones, DST transitions prevent
        # range representation due to no singular step
        if (
            self.tz is not None
            and not timezones.is_utc(self.tz)
            and not timezones.is_fixed_offset(self.tz)
        ):
            # 如果self或者other具有非UTC时区，并且不是固定偏移时区，则返回False
            return False
        if (
            other.tz is not None
            and not timezones.is_utc(other.tz)
            and not timezones.is_fixed_offset(other.tz)
        ):
            # 如果self或者other具有非UTC时区，并且不是固定偏移时区，则返回False
            return False
        # 调用父类的_can_range_setop方法进行进一步判断
        return super()._can_range_setop(other)

    # --------------------------------------------------------------------

    def _get_time_micros(self) -> npt.NDArray[np.int64]:
        """
        Return the number of microseconds since midnight.

        Returns
        -------
        ndarray[int64_t]
            返回自午夜以来的微秒数
        """
        # 获取本地时间戳数组
        values = self._data._local_timestamps()

        # 每天的周期数，根据时间分辨率确定
        ppd = periods_per_day(self._data._creso)

        # 计算时间戳的小数部分
        frac = values % ppd

        # 根据时间单位(unit)将小数部分转换为微秒数
        if self.unit == "ns":
            micros = frac // 1000
        elif self.unit == "us":
            micros = frac
        elif self.unit == "ms":
            micros = frac * 1000
        elif self.unit == "s":
            micros = frac * 1_000_000
        else:  # pragma: no cover
            # 如果单位不在支持的范围内，则抛出未实现错误
            raise NotImplementedError(self.unit)

        # 将NaN值对应的微秒数设为-1
        micros[self._isnan] = -1
        return micros
    def snap(self, freq: Frequency = "S") -> DatetimeIndex:
        """
        Snap time stamps to nearest occurring frequency.

        Parameters
        ----------
        freq : str, Timedelta, datetime.timedelta, or DateOffset, default 'S'
            Frequency strings can have multiples, e.g. '5h'. See
            :ref:`here <timeseries.offset_aliases>` for a list of
            frequency aliases.

        Returns
        -------
        DatetimeIndex
            Time stamps snapped to the nearest occurring `freq`.

        See Also
        --------
        DatetimeIndex.round : Perform round operation on the data to the
            specified `freq`.
        DatetimeIndex.floor : Perform floor operation on the data to the
            specified `freq`.

        Examples
        --------
        >>> idx = pd.DatetimeIndex(
        ...     ["2023-01-01", "2023-01-02", "2023-02-01", "2023-02-02"],
        ...     dtype="M8[ns]",
        ... )
        >>> idx
        DatetimeIndex(['2023-01-01', '2023-01-02', '2023-02-01', '2023-02-02'],
        dtype='datetime64[ns]', freq=None)
        >>> idx.snap("MS")
        DatetimeIndex(['2023-01-01', '2023-01-01', '2023-02-01', '2023-02-01'],
        dtype='datetime64[ns]', freq=None)
        """
        # 将频率参数转换为偏移量对象
        freq = to_offset(freq)

        # 复制数据以避免直接修改原始数据
        dta = self._data.copy()

        # 遍历索引中的每个时间戳
        for i, v in enumerate(self):
            s = v
            # 如果时间戳不在指定频率上，则进行调整
            if not freq.is_on_offset(s):
                # 回滚和前滚时间戳，选择与时间戳最接近的时间戳
                t0 = freq.rollback(s)
                t1 = freq.rollforward(s)
                if abs(s - t0) < abs(t1 - s):
                    s = t0
                else:
                    s = t1
            # 更新调整后的时间戳到数据副本中
            dta[i] = s

        # 使用调整后的数据副本创建新的DatetimeIndex对象
        return DatetimeIndex._simple_new(dta, name=self.name)

    # --------------------------------------------------------------------
    # Indexing Methods

    def _parsed_string_to_bounds(
        self, reso: Resolution, parsed: dt.datetime
    ):
        """
        Placeholder for method to convert parsed string to bounds.

        This method is intended to convert a parsed datetime object into
        appropriate bounds based on a specified resolution.

        Parameters
        ----------
        reso : Resolution
            Resolution of the time bounds.
        parsed : datetime.datetime
            Parsed datetime object.

        Returns
        -------
        Bounds
            Calculated bounds based on the parsed datetime and resolution.
        """
        # Placeholder method for converting parsed datetime to bounds
        pass
    ) -> tuple[Timestamp, Timestamp]:
        """
        Calculate datetime bounds for parsed time string and its resolution.

        Parameters
        ----------
        reso : Resolution
            Resolution provided by parsed string.
        parsed : datetime
            Datetime from parsed string.

        Returns
        -------
        lower, upper: pd.Timestamp
        """
        # 根据解析出的时间字符串的分辨率获取对应的频率字符串
        freq = OFFSET_TO_PERIOD_FREQSTR.get(reso.attr_abbrev, reso.attr_abbrev)
        # 使用时间段对象创建 Period 对象
        per = Period(parsed, freq=freq)
        # 获取时间段的开始时间和结束时间
        start, end = per.start_time, per.end_time
        # 将开始时间和结束时间转换为指定的时间单位
        start = start.as_unit(self.unit)
        end = end.as_unit(self.unit)

        # GH 24076
        # 如果输入的日期字符串包含了UTC偏移量，需要先将解析出的日期本地化到该偏移量
        # 然后再与索引的时区进行对齐
        start = start.tz_localize(parsed.tzinfo)
        end = end.tz_localize(parsed.tzinfo)

        # 检查解析出的日期是否具有时区信息
        if parsed.tzinfo is not None:
            # 如果索引的时区为None，则抛出异常
            if self.tz is None:
                raise ValueError(
                    "The index must be timezone aware when indexing "
                    "with a date string with a UTC offset"
                )
        # flipped case with parsed.tz is None and self.tz is not None
        # parsed 和 reso 由 _parse_with_reso 生成，会对 parsed 进行本地化，因此不会出现这种情况

        # 返回开始时间和结束时间
        return start, end

    def _parse_with_reso(self, label: str) -> tuple[Timestamp, Resolution]:
        # 调用父类方法解析标签并返回解析出的时间戳和分辨率
        parsed, reso = super()._parse_with_reso(label)

        # 将解析出的时间戳转换为 Timestamp 对象
        parsed = Timestamp(parsed)

        # 如果当前对象具有时区信息且解析出的时间戳没有时区信息
        if self.tz is not None and parsed.tzinfo is None:
            # 特殊处理时区非感知的字符串和时区感知的 DatetimeIndex
            parsed = parsed.tz_localize(self.tz)

        # 返回处理后的时间戳和分辨率
        return parsed, reso

    def _disallow_mismatched_indexing(self, key) -> None:
        """
        Check for mismatched-tzawareness indexing and re-raise as KeyError.
        """
        # 检查索引键是否与时区感知兼容
        try:
            self._data._assert_tzawareness_compat(key)
        except TypeError as err:
            # 如果不兼容，则抛出 KeyError 异常
            raise KeyError(key) from err
    def get_loc(self, key):
        """
        Get integer location for requested label

        Returns
        -------
        loc : int
        """
        # 检查索引错误
        self._check_indexing_error(key)

        # 保存原始的键值
        orig_key = key

        # 如果 key 是一个适合数据类型的 NA（缺失值），则将其设为 NaT
        if is_valid_na_for_dtype(key, self.dtype):
            key = NaT

        # 如果 key 是一个已识别的标量类型
        if isinstance(key, self._data._recognized_scalars):
            # 需要本地化的本地日期时间
            self._disallow_mismatched_indexing(key)
            key = Timestamp(key)

        # 如果 key 是一个字符串
        elif isinstance(key, str):
            try:
                # 尝试解析字符串并确定其分辨率
                parsed, reso = self._parse_with_reso(key)
            except (ValueError, pytz.NonExistentTimeError) as err:
                # 如果解析失败，抛出 KeyError 异常
                raise KeyError(key) from err
            self._disallow_mismatched_indexing(parsed)

            # 如果可以进行部分日期切片
            if self._can_partial_date_slice(reso):
                try:
                    return self._partial_date_slice(reso, parsed)
                except KeyError as err:
                    raise KeyError(key) from err

            key = parsed

        # 如果 key 是一个时间增量对象
        elif isinstance(key, dt.timedelta):
            # 抛出类型错误，不能使用时间增量对象进行索引
            raise TypeError(
                f"Cannot index {type(self).__name__} with {type(key).__name__}"
            )

        # 如果 key 是一个时间对象
        elif isinstance(key, dt.time):
            # 返回时间点的索引位置
            return self.indexer_at_time(key)

        # 如果 key 是一个未被识别的类型
        else:
            # 抛出 KeyError 异常
            raise KeyError(key)

        # 尝试获取 key 在 Index 中的位置
        try:
            return Index.get_loc(self, key)
        except KeyError as err:
            # 如果获取位置失败，抛出带有原始 key 的 KeyError 异常
            raise KeyError(orig_key) from err

    @doc(DatetimeTimedeltaMixin._maybe_cast_slice_bound)
    def _maybe_cast_slice_bound(self, label, side: str):
        """
        Maybe cast slice bound to Timestamp

        Parameters
        ----------
        label : object
            Label to potentially cast
        side : str
            Side of the slice

        Returns
        -------
        Timestamp
            Timestamp representation of the label
        """
        # 处理日期而不是在 get_slice_bound 中处理
        if isinstance(label, dt.date) and not isinstance(label, dt.datetime):
            # Pandas 支持使用日期进行切片，将其视为午夜的日期时间
            # https://github.com/pandas-dev/pandas/issues/31501
            label = Timestamp(label).to_pydatetime()

        # 调用父类方法，可能将 label 转换为 Timestamp 类型
        label = super()._maybe_cast_slice_bound(label, side)
        # 确保数据具有时区意识的兼容性
        self._data._assert_tzawareness_compat(label)
        return Timestamp(label)
    def slice_indexer(self, start=None, end=None, step=None):
        """
        Return indexer for specified label slice.
        Index.slice_indexer, customized to handle time slicing.

        In addition to functionality provided by Index.slice_indexer, does the
        following:

        - if both `start` and `end` are instances of `datetime.time`, it
          invokes `indexer_between_time`
        - if `start` and `end` are both either string or None perform
          value-based selection in non-monotonic cases.

        """
        # 如果 `start` 和 `end` 都是 `datetime.time` 的实例，则调用 `indexer_between_time`
        if isinstance(start, dt.time) and isinstance(end, dt.time):
            # 如果指定了 `step` 并且不是 1，则抛出 ValueError 异常
            if step is not None and step != 1:
                raise ValueError("Must have step size of 1 with time slices")
            return self.indexer_between_time(start, end)

        # 如果 `start` 或 `end` 是 `datetime.time` 的实例，则抛出 KeyError 异常
        if isinstance(start, dt.time) or isinstance(end, dt.time):
            raise KeyError("Cannot mix time and non-time slice keys")

        def check_str_or_none(point) -> bool:
            return point is not None and not isinstance(point, str)

        # 如果 `start` 和 `end` 是字符串或者 None，并且索引不是单调的，无法使用 `Index.slice_indexer`
        if (
            check_str_or_none(start)
            or check_str_or_none(end)
            or self.is_monotonic_increasing
        ):
            return Index.slice_indexer(self, start, end, step)

        mask = np.array(True)
        in_index = True
        if start is not None:
            start_casted = self._maybe_cast_slice_bound(start, "left")
            mask = start_casted <= self
            in_index &= (start_casted == self).any()

        if end is not None:
            end_casted = self._maybe_cast_slice_bound(end, "right")
            mask = (self <= end_casted) & mask
            in_index &= (end_casted == self).any()

        if not in_index:
            raise KeyError(
                "Value based partial slicing on non-monotonic DatetimeIndexes "
                "with non-existing keys is not allowed.",
            )
        indexer = mask.nonzero()[0][::step]
        if len(indexer) == len(self):
            return slice(None)
        else:
            return indexer

    # --------------------------------------------------------------------

    @property
    def inferred_type(self) -> str:
        # 因为 datetime 表示为自纪元以来的微秒数，确保索引不会有歧义
        return "datetime64"
    def indexer_at_time(self, time, asof: bool = False) -> npt.NDArray[np.intp]:
        """
        Return index locations of values at a particular time of day.

        Parameters
        ----------
        time : datetime.time or str
            Time passed in either as object (datetime.time) or as string in
            appropriate format ("%H:%M", "%H%M", "%I:%M%p", "%I%M%p",
            "%H:%M:%S", "%H%M%S", "%I:%M:%S%p", "%I%M%S%p").
        asof : bool, default False
            This parameter is currently not supported.

        Returns
        -------
        np.ndarray[np.intp]
            Index locations of values at the given `time` of day.

        See Also
        --------
        indexer_between_time : Get index locations of values between particular
            times of day.
        DataFrame.at_time : Select values at a particular time of day.

        Examples
        --------
        >>> idx = pd.DatetimeIndex(
        ...     ["1/1/2020 10:00", "2/1/2020 11:00", "3/1/2020 10:00"]
        ... )
        >>> idx.indexer_at_time("10:00")
        array([0, 2])
        """
        # 如果参数 asof 为 True，则抛出未实现错误
        if asof:
            raise NotImplementedError("'asof' argument is not supported")

        # 如果传入的时间参数是字符串，则使用 dateutil.parser 中的 parse 函数解析为 datetime.time 对象
        if isinstance(time, str):
            from dateutil.parser import parse
            time = parse(time).time()

        # 如果时间对象有时区信息，并且当前对象的时区属性为 None，则抛出值错误
        if time.tzinfo:
            if self.tz is None:
                raise ValueError("Index must be timezone aware.")
            # 转换时间到当前对象的时区，并获取微秒级别的时间
            time_micros = self.tz_convert(time.tzinfo)._get_time_micros()
        else:
            # 否则，获取当前对象的微秒级别时间
            time_micros = self._get_time_micros()

        # 将传入的时间转换为微秒级别的时间，并返回与当前对象时间相等的索引位置数组
        micros = _time_to_micros(time)
        return (time_micros == micros).nonzero()[0]
        """
        Return index locations of values between particular times of day.

        Parameters
        ----------
        start_time, end_time : datetime.time, str
            Time passed either as object (datetime.time) or as string in
            appropriate format ("%H:%M", "%H%M", "%I:%M%p", "%I%M%p",
            "%H:%M:%S", "%H%M%S", "%I:%M:%S%p","%I%M%S%p").
        include_start : bool, default True
            Include boundaries; whether to set start bound as closed or open.
        include_end : bool, default True
            Include boundaries; whether to set end bound as closed or open.

        Returns
        -------
        np.ndarray[np.intp]
            Index locations of values between particular times of day.

        See Also
        --------
        indexer_at_time : Get index locations of values at particular time of day.
        DataFrame.between_time : Select values between particular times of day.

        Examples
        --------
        >>> idx = pd.date_range("2023-01-01", periods=4, freq="h")
        >>> idx
        DatetimeIndex(['2023-01-01 00:00:00', '2023-01-01 01:00:00',
                           '2023-01-01 02:00:00', '2023-01-01 03:00:00'],
                          dtype='datetime64[ns]', freq='h')
        >>> idx.indexer_between_time("00:00", "2:00", include_end=False)
        array([0, 1])
        """
        # 将输入的开始时间和结束时间转换为时间对象
        start_time = to_time(start_time)
        end_time = to_time(end_time)
        # 获取时间的微秒表示
        time_micros = self._get_time_micros()
        # 将开始时间和结束时间转换为微秒
        start_micros = _time_to_micros(start_time)
        end_micros = _time_to_micros(end_time)

        # 根据参数include_start和include_end设置lop和rop函数
        if include_start and include_end:
            lop = rop = operator.le
        elif include_start:
            lop = operator.le
            rop = operator.lt
        elif include_end:
            lop = operator.lt
            rop = operator.le
        else:
            lop = rop = operator.lt

        # 根据开始时间和结束时间的顺序确定连接操作
        if start_time <= end_time:
            join_op = operator.and_
        else:
            join_op = operator.or_

        # 创建一个布尔掩码，指示时间是否在指定范围内
        mask = join_op(lop(start_micros, time_micros), rop(time_micros, end_micros))

        # 返回掩码数组中非零元素的索引
        return mask.nonzero()[0]
def date_range(
    start=None,                              # 左边界，生成日期的起始点，可以是字符串或类似日期时间的对象，可选参数
    end=None,                                # 右边界，生成日期的结束点，可以是字符串或类似日期时间的对象，可选参数
    periods=None,                            # 生成的周期数，可选参数
    freq=None,                               # 频率字符串、时间增量或日期偏移对象，默认为'D'（每日），可选参数
    tz=None,                                 # 时区名称或时区信息，用于返回本地化的DatetimeIndex，可选参数
    normalize: bool = False,                 # 是否将开始/结束日期标准化到午夜之前再生成日期范围，默认为False，可选参数
    name: Hashable | None = None,            # 结果DatetimeIndex的名称，默认为None，可选参数
    inclusive: IntervalClosedType = "both",  # 包含边界的方式，可以是"both"、"neither"、"left"、"right"，默认为"both"
    *,
    unit: str | None = None,                 # 指定结果的期望分辨率，可选参数
    **kwargs,
) -> DatetimeIndex:
    """
    Return a fixed frequency DatetimeIndex.

    Returns the range of equally spaced time points (where the difference between any
    two adjacent points is specified by the given frequency) such that they fall in the
    range `[start, end]` , where the first one and the last one are, resp., the first
    and last time points in that range that fall on the boundary of ``freq`` (if given
    as a frequency string) or that are valid for ``freq`` (if given as a
    :class:`pandas.tseries.offsets.DateOffset`). If ``freq`` is positive, the points
    satisfy `start <[=] x <[=] end`, and if ``freq`` is negative, the points satisfy
    `end <[=] x <[=] start`. (If exactly one of ``start``, ``end``, or ``freq`` is *not*
    specified, this missing parameter can be computed given ``periods``, the number of
    timesteps in the range. See the note below.)

    Parameters
    ----------
    start : str or datetime-like, optional
        Left bound for generating dates.
    end : str or datetime-like, optional
        Right bound for generating dates.
    periods : int, optional
        Number of periods to generate.
    freq : str, Timedelta, datetime.timedelta, or DateOffset, default 'D'
        Frequency strings can have multiples, e.g. '5h'. See
        :ref:`here <timeseries.offset_aliases>` for a list of
        frequency aliases.
    tz : str or tzinfo, optional
        Time zone name for returning localized DatetimeIndex, for example
        'Asia/Hong_Kong'. By default, the resulting DatetimeIndex is
        timezone-naive unless timezone-aware datetime-likes are passed.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    name : str, default None
        Name of the resulting DatetimeIndex.
    inclusive : {"both", "neither", "left", "right"}, default "both"
        Include boundaries; Whether to set each bound as closed or open.

        .. versionadded:: 1.4.0
    unit : str, default None
        Specify the desired resolution of the result.

        .. versionadded:: 2.0.0
    **kwargs
        For compatibility. Has no effect on the result.

    Returns
    -------
    DatetimeIndex

    See Also
    --------
    DatetimeIndex : An immutable container for datetimes.
    timedelta_range : Return a fixed frequency TimedeltaIndex.
    period_range : Return a fixed frequency PeriodIndex.
    interval_range : Return a fixed frequency IntervalIndex.

    Notes
    -----
    Of the four parameters ``start``, ``end``, ``periods``, and ``freq``,
    exactly three must be specified. If ``freq`` is omitted, the resulting
    ``DatetimeIndex`` will have ``periods`` linearly spaced elements between
    """
    # 通过指定 `start` 和 `end` 参数来生成一个DatetimeIndex，闭区间（两端都包含）。
    # 若要了解更多关于频率字符串的信息，请参考此链接：ref:`this link<timeseries.offset_aliases>`
    #
    # 示例
    # --------
    # **指定数值**
    #
    # 下面的四个示例生成相同的 `DatetimeIndex`，但是 `start`、`end` 和 `periods` 的组合不同。
    #
    # 指定 `start` 和 `end`，使用默认的每日频率。
    >>> pd.date_range(start="1/1/2018", end="1/08/2018")
    # 生成包含从 '2018-01-01' 到 '2018-01-08' 的日期索引，每天一个日期。
    # 返回类型为 `DatetimeIndex`，频率为 'D'，即每日。
    DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
                   '2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08'],
                  dtype='datetime64[ns]', freq='D')

    # 指定带时区信息的 `start` 和 `end`，使用默认的每日频率。
    >>> pd.date_range(
    ...     start=pd.to_datetime("1/1/2018").tz_localize("Europe/Berlin"),
    ...     end=pd.to_datetime("1/08/2018").tz_localize("Europe/Berlin"),
    ... )
    # 生成包含从 '2018-01-01' 到 '2018-01-08' 的日期索引，每天一个日期。
    # 返回类型为 `DatetimeIndex`，带有欧洲柏林时区的日期信息，频率为 'D'，即每日。
    DatetimeIndex(['2018-01-01 00:00:00+01:00', '2018-01-02 00:00:00+01:00',
                   '2018-01-03 00:00:00+01:00', '2018-01-04 00:00:00+01:00',
                   '2018-01-05 00:00:00+01:00', '2018-01-06 00:00:00+01:00',
                   '2018-01-07 00:00:00+01:00', '2018-01-08 00:00:00+01:00'],
                  dtype='datetime64[ns, Europe/Berlin]', freq='D')

    # 指定 `start` 和 `periods`，即期间的数量（天数）。
    >>> pd.date_range(start="1/1/2018", periods=8)
    # 生成从 '2018-01-01' 开始的8个连续日期。
    # 返回类型为 `DatetimeIndex`，频率为 'D'，即每日。
    DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
                   '2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08'],
                  dtype='datetime64[ns]', freq='D')

    # 指定 `end` 和 `periods`，即期间的数量（天数）。
    >>> pd.date_range(end="1/1/2018", periods=8)
    # 生成从 '2017-12-25' 开始的8个连续日期，直到 '2018-01-01'。
    # 返回类型为 `DatetimeIndex`，频率为 'D'，即每日。
    DatetimeIndex(['2017-12-25', '2017-12-26', '2017-12-27', '2017-12-28',
                   '2017-12-29', '2017-12-30', '2017-12-31', '2018-01-01'],
                  dtype='datetime64[ns]', freq='D')

    # 指定 `start`、`end` 和 `periods`；频率会自动生成（线性间隔）。
    >>> pd.date_range(start="2018-04-24", end="2018-04-27", periods=3)
    # 生成一个包含 '2018-04-24' 到 '2018-04-27' 的日期索引，共3个日期。
    # 返回类型为 `DatetimeIndex`，频率为 `None`。
    DatetimeIndex(['2018-04-24 00:00:00', '2018-04-25 12:00:00',
                   '2018-04-27 00:00:00'],
                  dtype='datetime64[ns]', freq=None)

    # **其他参数**
    #
    # 将 `freq`（频率）更改为 `'ME'`（月末频率）。
    >>> pd.date_range(start="1/1/2018", periods=5, freq="ME")
    # 生成包含从 '2018-01-31' 到 '2018-05-31' 的日期索引，每月最后一天。
    # 返回类型为 `DatetimeIndex`，频率为 `'ME'`，即每月末。
    DatetimeIndex(['2018-01-31', '2018-02-28', '2018-03-31', '2018-04-30',
                   '2018-05-31'],
                  dtype='datetime64[ns]', freq='ME')

    # 允许指定倍数
    >>> pd.date_range(start="1/1/2018", periods=5, freq="3ME")
    # 生成包含从 '2018-01-31' 到 '2019-01-31' 的日期索引，每3个月末一次。
    # 返回类型为 `DatetimeIndex`，频率为 `'3ME'`，即每3个月末。
    DatetimeIndex(['2018-01-31', '2018-04-30', '2018-07-31', '2018-10-31',
                   '2019-01-31'],
                  dtype='datetime64[ns]', freq='3ME')

    # `freq` 也可以指定为一个偏移对象。
    >>> pd.date_range(start="1/1/2018", periods=5, freq=pd.offsets.MonthEnd(3))
    # 生成包含从 '2018-01-31' 到 '2018-05-31' 的日期索引，每3个月的最后一天。
    # 返回类型为 `DatetimeIndex`，频率为 `'3M'`，即每3个月的月末。
    # 创建一个日期时间索引，包含指定日期
    DatetimeIndex(['2018-01-31', '2018-04-30', '2018-07-31', '2018-10-31',
                   '2019-01-31'],
                  dtype='datetime64[ns]', freq='3ME')

    # 指定 `tz` 参数以设置时区信息
    Specify `tz` to set the timezone.

    # 使用指定的起始日期、周期数和时区来创建日期时间索引
    >>> pd.date_range(start="1/1/2018", periods=5, tz="Asia/Tokyo")
    DatetimeIndex(['2018-01-01 00:00:00+09:00', '2018-01-02 00:00:00+09:00',
                   '2018-01-03 00:00:00+09:00', '2018-01-04 00:00:00+09:00',
                   '2018-01-05 00:00:00+09:00'],
                  dtype='datetime64[ns, Asia/Tokyo]', freq='D')

    # `inclusive` 控制是否包括边界上的 `start` 和 `end`。默认为 "both"，包括两端的边界点。
    >>> pd.date_range(start="2017-01-01", end="2017-01-04", inclusive="both")
    DatetimeIndex(['2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04'],
                  dtype='datetime64[ns]', freq='D')

    # 使用 ``inclusive='left'`` 可以排除 `end` 如果它恰好在边界上的情况。
    >>> pd.date_range(start="2017-01-01", end="2017-01-04", inclusive="left")
    DatetimeIndex(['2017-01-01', '2017-01-02', '2017-01-03'],
                  dtype='datetime64[ns]', freq='D')

    # 使用 ``inclusive='right'`` 可以排除 `start` 如果它恰好在边界上的情况，类似地，``inclusive='neither'`` 将排除 `start` 和 `end` 都在边界上的情况。
    >>> pd.date_range(start="2017-01-01", end="2017-01-04", inclusive="right")
    DatetimeIndex(['2017-01-02', '2017-01-03', '2017-01-04'],
                  dtype='datetime64[ns]', freq='D')

    # **指定单位**
    Specify a unit

    # 使用指定的起始日期、周期数、频率和单位来创建日期时间索引
    >>> pd.date_range(start="2017-01-01", periods=10, freq="100YS", unit="s")
    DatetimeIndex(['2017-01-01', '2117-01-01', '2217-01-01', '2317-01-01',
                   '2417-01-01', '2517-01-01', '2617-01-01', '2717-01-01',
                   '2817-01-01', '2917-01-01'],
                  dtype='datetime64[s]', freq='100YS-JAN')
    """
    # 如果 `freq` 为 None 并且 `periods`, `start`, `end` 中有任何一个为 None，则将 `freq` 设置为 "D"
    if freq is None and com.any_none(periods, start, end):
        freq = "D"

    # 调用 DatetimeArray._generate_range 方法生成一个日期时间数组
    dtarr = DatetimeArray._generate_range(
        start=start,
        end=end,
        periods=periods,
        freq=freq,
        tz=tz,
        normalize=normalize,
        inclusive=inclusive,
        unit=unit,
        **kwargs,
    )
    # 返回一个新的 DatetimeIndex 对象，使用生成的日期时间数组和指定的名称
    return DatetimeIndex._simple_new(dtarr, name=name)
# 返回一个固定频率的DatetimeIndex，以工作日为默认频率
def bdate_range(
    start=None,  # 开始日期，可以是字符串或类似datetime的对象，默认为None
    end=None,  # 结束日期，可以是字符串或类似datetime的对象，默认为None
    periods: int | None = None,  # 生成日期的数量，默认为None
    freq: Frequency | dt.timedelta = "B",  # 频率字符串或时间增量，用于生成日期，默认为工作日('B')
    tz=None,  # 返回本地化DatetimeIndex时使用的时区名，例如'Asia/Beijing'
    normalize: bool = True,  # 是否将开始/结束日期规范化到午夜之前再生成日期范围，默认为True
    name: Hashable | None = None,  # 结果DatetimeIndex的名称，默认为None
    weekmask=None,  # 有效工作日的星期掩码，传递给numpy.busdaycalendar，仅在使用自定义频率字符串时有效，默认为None
    holidays=None,  # 要排除的日期列表，传递给numpy.busdaycalendar，仅在使用自定义频率字符串时有效，默认为None
    inclusive: IntervalClosedType = "both",  # 包含边界的方式，可选值为"both", "neither", "left", "right"，默认为"both"
    **kwargs,  # 兼容性参数，对结果无影响
) -> DatetimeIndex:  # 返回类型为DatetimeIndex
    """
    返回一个固定频率的DatetimeIndex，以工作日为默认。

    参数
    ----------
    start : str or datetime-like, default None
        生成日期的左边界。
    end : str or datetime-like, default None
        生成日期的右边界。
    periods : int, default None
        要生成的日期数量。
    freq : str, Timedelta, datetime.timedelta, or DateOffset, default 'B'
        频率字符串可以有倍数，例如 '5h'。默认为工作日 ('B')。
    tz : str or None
        返回本地化DatetimeIndex时使用的时区名，例如'Asia/Beijing'。
    normalize : bool, default False
        是否将开始/结束日期规范化到午夜之前再生成日期范围。
    name : str, default None
        结果DatetimeIndex的名称。
    weekmask : str or None, default None
        有效工作日的星期掩码，传递给``numpy.busdaycalendar``，仅在使用自定义频率字符串时有效。默认为'Mon Tue Wed Thu Fri'。
    holidays : list-like or None, default None
        要排除的日期列表，传递给``numpy.busdaycalendar``，仅在使用自定义频率字符串时有效。
    inclusive : {"both", "neither", "left", "right"}, default "both"
        包含边界的方式；设置每个边界是封闭还是开放的。

        .. versionadded:: 1.4.0
    **kwargs
        兼容性参数，对结果无影响。

    返回
    -------
    DatetimeIndex
        固定频率的DatetimeIndex。

    参见
    --------
    date_range : 返回一个固定频率的DatetimeIndex。
    period_range : 返回一个固定频率的PeriodIndex。
    timedelta_range : 返回一个固定频率的TimedeltaIndex。

    注意
    -----
    在四个参数中：``start``, ``end``, ``periods``, 和 ``freq``，必须确切指定三个。
    对于 ``bdate_range``，指定 ``freq`` 是必须的。如果不希望指定 ``freq``，请使用 ``date_range``。

    要了解更多有关频率字符串的信息，请参阅:ref:`此链接<timeseries.offset_aliases>`。

    示例
    --------
    注意结果中跳过了两个周末天数。

    >>> pd.bdate_range(start="1/1/2018", end="1/08/2018")
    DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
               '2018-01-05', '2018-01-08'],
              dtype='datetime64[ns]', freq='B')
    """
    if freq is None:
        msg = "freq must be specified for bdate_range; use date_range instead"
        raise TypeError(msg)
    # 如果频率 freq 是字符串并且以 "C" 开头
    if isinstance(freq, str) and freq.startswith("C"):
        # 如果未指定 weekmask，则默认为 "Mon Tue Wed Thu Fri"
        weekmask = weekmask or "Mon Tue Wed Thu Fri"
        # 根据 freq 使用前缀映射表创建相应的频率对象，传入 holidays 和 weekmask 参数
        freq = prefix_mapping[freq](holidays=holidays, weekmask=weekmask)
        # 如果在创建频率对象时出现 KeyError 或 TypeError 异常，则抛出 ValueError
        except (KeyError, TypeError) as err:
            msg = f"invalid custom frequency string: {freq}"
            raise ValueError(msg) from err
    # 如果传入了 holidays 或 weekmask 参数
    elif holidays or weekmask:
        # 抛出 ValueError，提示必须提供自定义频率字符串
        msg = (
            "a custom frequency string is required when holidays or "
            f"weekmask are passed, got frequency {freq}"
        )
        raise ValueError(msg)

    # 调用 date_range 函数生成日期范围
    return date_range(
        start=start,
        end=end,
        periods=periods,
        freq=freq,
        tz=tz,
        normalize=normalize,
        name=name,
        inclusive=inclusive,
        **kwargs,
    )
# 将给定的时间对象转换为微秒数表示
def _time_to_micros(time_obj: dt.time) -> int:
    # 计算总秒数，包括小时、分钟和秒
    seconds = time_obj.hour * 60 * 60 + 60 * time_obj.minute + time_obj.second
    # 将总秒数转换为微秒数，并加上时间对象的微秒部分
    return 1_000_000 * seconds + time_obj.microsecond
```