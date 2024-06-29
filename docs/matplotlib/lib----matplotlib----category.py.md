# `D:\src\scipysrc\matplotlib\lib\matplotlib\category.py`

```
"""
Plotting of string "category" data: ``plot(['d', 'f', 'a'], [1, 2, 3])`` will
plot three points with x-axis values of 'd', 'f', 'a'.

See :doc:`/gallery/lines_bars_and_markers/categorical_variables` for an
example.

The module uses Matplotlib's `matplotlib.units` mechanism to convert from
strings to integers and provides a tick locator, a tick formatter, and the
`.UnitData` class that creates and stores the string-to-integer mapping.
"""

# 导入必要的库
from collections import OrderedDict  # 导入 OrderedDict 类
import dateutil.parser  # 导入 dateutil.parser 模块
import itertools  # 导入 itertools 模块，用于迭代操作
import logging  # 导入 logging 模块，用于记录日志信息

import numpy as np  # 导入 NumPy 库

from matplotlib import _api, ticker, units  # 导入 Matplotlib 库中的 _api、ticker 和 units 模块


_log = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class StrCategoryConverter(units.ConversionInterface):
    @staticmethod
    def convert(value, unit, axis):
        """
        Convert strings in *value* to floats using mapping information stored
        in the *unit* object.

        Parameters
        ----------
        value : str or iterable
            Value or list of values to be converted.
        unit : `.UnitData`
            An object mapping strings to integers.
        axis : `~matplotlib.axis.Axis`
            The axis on which the converted value is plotted.

            .. note:: *axis* is unused.

        Returns
        -------
        float or `~numpy.ndarray` of float
        """
        if unit is None:
            raise ValueError(
                'Missing category information for StrCategoryConverter; '
                'this might be caused by unintendedly mixing categorical and '
                'numeric data')
        StrCategoryConverter._validate_unit(unit)
        # dtype = object preserves numerical pass throughs
        values = np.atleast_1d(np.array(value, dtype=object))
        # force an update so it also does type checking
        unit.update(values)
        return np.vectorize(unit._mapping.__getitem__, otypes=[float])(values)

    @staticmethod
    def axisinfo(unit, axis):
        """
        Set the default axis ticks and labels.

        Parameters
        ----------
        unit : `.UnitData`
            object string unit information for value
        axis : `~matplotlib.axis.Axis`
            axis for which information is being set

            .. note:: *axis* is not used

        Returns
        -------
        `~matplotlib.units.AxisInfo`
            Information to support default tick labeling

        """
        StrCategoryConverter._validate_unit(unit)
        # locator and formatter take mapping dict because
        # args need to be pass by reference for updates
        majloc = StrCategoryLocator(unit._mapping)
        majfmt = StrCategoryFormatter(unit._mapping)
        return units.AxisInfo(majloc=majloc, majfmt=majfmt)

    @staticmethod
    def _validate_unit(unit):
        """
        Validate that the unit provided is a `.UnitData` instance.

        Parameters
        ----------
        unit : object
            The unit object to validate.

        Raises
        ------
        ValueError
            If the unit object is not a `.UnitData` instance.

        """
        if not isinstance(unit, units.UnitData):
            raise ValueError('Expected a UnitData instance')
    # 定义一个静态方法 `_validate_unit`，用于验证单位是否有效
    @staticmethod
    def _validate_unit(unit):
        # 如果提供的单位 `unit` 没有 `_mapping` 属性，抛出值错误异常
        if not hasattr(unit, '_mapping'):
            raise ValueError(
                f'Provided unit "{unit}" is not valid for a categorical '
                'converter, as it does not have a _mapping attribute.')
class StrCategoryLocator(ticker.Locator):
    """Tick at every integer mapping of the string data."""
    def __init__(self, units_mapping):
        """
        Parameters
        ----------
        units_mapping : dict
            Mapping of category names (str) to indices (int).
        """
        self._units = units_mapping  # 初始化实例变量 _units，用于存储类别名称到索引的映射

    def __call__(self):
        # docstring inherited
        return list(self._units.values())  # 返回所有类别名称对应的索引值列表

    def tick_values(self, vmin, vmax):
        # docstring inherited
        return self()  # 调用 __call__ 方法返回所有类别名称对应的索引值列表


class StrCategoryFormatter(ticker.Formatter):
    """String representation of the data at every tick."""
    def __init__(self, units_mapping):
        """
        Parameters
        ----------
        units_mapping : dict
            Mapping of category names (str) to indices (int).
        """
        self._units = units_mapping  # 初始化实例变量 _units，用于存储类别名称到索引的映射

    def __call__(self, x, pos=None):
        # docstring inherited
        return self.format_ticks([x])[0]  # 调用 format_ticks 方法返回 x 对应的格式化字符串

    def format_ticks(self, values):
        # docstring inherited
        r_mapping = {v: self._text(k) for k, v in self._units.items()}  # 创建索引到格式化字符串的映射字典
        return [r_mapping.get(round(val), '') for val in values]  # 返回 values 中每个值对应的格式化字符串列表

    @staticmethod
    def _text(value):
        """Convert text values into utf-8 or ascii strings."""
        if isinstance(value, bytes):
            value = value.decode(encoding='utf-8')  # 如果值是字节串，解码成 UTF-8 字符串
        elif not isinstance(value, str):
            value = str(value)  # 如果值不是字符串，转换成字符串
        return value


class UnitData:
    def __init__(self, data=None):
        """
        Create mapping between unique categorical values and integer ids.

        Parameters
        ----------
        data : iterable
            sequence of string values
        """
        self._mapping = OrderedDict()  # 初始化一个有序字典 _mapping，用于存储类别值到整数 ID 的映射
        self._counter = itertools.count()  # 初始化一个计数器，用于生成唯一整数 ID
        if data is not None:
            self.update(data)  # 如果提供了 data 参数，则更新映射表

    @staticmethod
    def _str_is_convertible(val):
        """
        Helper method to check whether a string can be parsed as float or date.
        """
        try:
            float(val)  # 尝试将值转换成浮点数
        except ValueError:
            try:
                dateutil.parser.parse(val)  # 尝试使用日期解析器解析值
            except (ValueError, TypeError):
                # TypeError 如果 dateutil >= 2.8.1，否则 ValueError
                return False
        return True  # 如果能成功转换或解析，则返回 True
    def update(self, data):
        """
        Map new values to integer identifiers.

        Parameters
        ----------
        data : iterable of str or bytes
            输入数据，可以是字符串或字节流的可迭代对象。

        Raises
        ------
        TypeError
            如果 *data* 中的元素既不是 str 也不是 bytes 类型。

        """
        # 将输入数据转换为至少为1维的对象数组，并且将其元素类型设定为 object 类型
        data = np.atleast_1d(np.array(data, dtype=object))

        # 检查是否可转换为数字：
        convertible = True
        # 使用 OrderedDict 来迭代数据中的唯一值。
        for val in OrderedDict.fromkeys(data):
            # 检查 val 是否为 str 或 bytes 类型
            _api.check_isinstance((str, bytes), value=val)
            if convertible:
                # 只有在 convertible 为 True 时才会调用此函数。
                convertible = self._str_is_convertible(val)
            if val not in self._mapping:
                # 如果 val 不在映射表中，则将其映射到下一个计数器的值。
                self._mapping[val] = next(self._counter)
        
        # 如果数据非空并且所有元素均可转换，则记录信息。
        if data.size and convertible:
            _log.info('Using categorical units to plot a list of strings '
                      'that are all parsable as floats or dates. If these '
                      'strings should be plotted as numbers, cast to the '
                      'appropriate data type before plotting.')
# 将 StrCategoryConverter 注册到 Matplotlib 单位框架中用于处理字符串类型的单位转换
# 注册 str 类型的转换器
units.registry[str] = StrCategoryConverter()
# 注册 numpy 中的 np.str_ 类型的转换器
units.registry[np.str_] = StrCategoryConverter()
# 注册 bytes 类型的转换器
units.registry[bytes] = StrCategoryConverter()
# 注册 numpy 中的 np.bytes_ 类型的转换器
units.registry[np.bytes_] = StrCategoryConverter()
```