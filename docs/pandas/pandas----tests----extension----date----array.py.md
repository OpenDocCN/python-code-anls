# `D:\src\scipysrc\pandas\pandas\tests\extension\date\array.py`

```
# 导入未来的注解功能，用于支持类型注解
from __future__ import annotations

# 导入日期时间模块，并简称为 dt
import datetime as dt
# 导入类型相关的模块和函数
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

# 导入 NumPy 库，并简称为 np
import numpy as np

# 导入 Pandas 库中与数据类型相关的函数
from pandas.core.dtypes.dtypes import register_extension_dtype

# 导入 Pandas 扩展类型相关的类和函数
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
)
# 导入 Pandas 数据类型相关的函数
from pandas.api.types import pandas_dtype

# 如果支持类型检查，则导入 Sequence 和 PositionalIndexer 类型
if TYPE_CHECKING:
    from collections.abc import Sequence
    from pandas._typing import (
        Dtype,
        PositionalIndexer,
    )

# 注册一个新的扩展数据类型，名为 DateDtype
@register_extension_dtype
class DateDtype(ExtensionDtype):
    @property
    def type(self):
        # 返回此数据类型对应的 Python 原生类型，这里是日期对象
        return dt.date

    @property
    def name(self):
        # 返回此数据类型的名称
        return "DateDtype"

    @classmethod
    def construct_from_string(cls, string: str):
        # 如果传入的参数不是字符串，则抛出类型错误
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )

        # 如果传入的字符串与类名匹配，则返回一个新的 DateDtype 实例
        if string == cls.__name__:
            return cls()
        else:
            # 否则，抛出类型错误，无法从给定字符串构造此数据类型
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")

    @classmethod
    def construct_array_type(cls):
        # 返回此数据类型对应的扩展数组类型，这里应该是 DateArray 类
        return DateArray

    @property
    def na_value(self):
        # 返回此数据类型的缺失值表示，这里是日期的最小值
        return dt.date.min

    def __repr__(self) -> str:
        # 返回此数据类型的字符串表示形式，即其名称
        return self.name


# 定义一个 DateArray 类，继承自 ExtensionArray
class DateArray(ExtensionArray):
    def __init__(
        self,
        dates: (
            dt.date
            | Sequence[dt.date]
            | tuple[np.ndarray, np.ndarray, np.ndarray]
            | np.ndarray
        ),
    ) -> None:
        # 如果 dates 是单个日期对象，则将年、月、日分别存储为单元素数组
        if isinstance(dates, dt.date):
            self._year = np.array([dates.year])
            self._month = np.array([dates.month])
            self._day = np.array([dates.year])  # 错误：应为 dates.day
            return

        ldates = len(dates)
        if isinstance(dates, list):
            # 预先分配数组大小，因为我们预先知道大小
            self._year = np.zeros(ldates, dtype=np.uint16)  # 65535 (0, 9999)
            self._month = np.zeros(ldates, dtype=np.uint8)  # 255 (1, 31)
            self._day = np.zeros(ldates, dtype=np.uint8)  # 255 (1, 12)
            # 填充数组
            for i, (y, m, d) in enumerate(
                (date.year, date.month, date.day) for date in dates
            ):
                self._year[i] = y
                self._month[i] = m
                self._day[i] = d

        elif isinstance(dates, tuple):
            # 仅支持三元组
            if ldates != 3:
                raise ValueError("only triples are valid")
            # 检查所有元素是否具有相同类型
            if any(not isinstance(x, np.ndarray) for x in dates):
                raise TypeError("invalid type")
            ly, lm, ld = (len(cast(np.ndarray, d)) for d in dates)
            if not ly == lm == ld:
                raise ValueError(
                    f"tuple members must have the same length: {(ly, lm, ld)}"
                )
            self._year = dates[0].astype(np.uint16)
            self._month = dates[1].astype(np.uint8)
            self._day = dates[2].astype(np.uint8)

        elif isinstance(dates, np.ndarray) and dates.dtype == "U10":
            # 如果 dates 是字符串数组且每个字符串表示日期（如 "YYYY-MM-DD"）
            self._year = np.zeros(ldates, dtype=np.uint16)  # 65535 (0, 9999)
            self._month = np.zeros(ldates, dtype=np.uint8)  # 255 (1, 31)
            self._day = np.zeros(ldates, dtype=np.uint8)  # 255 (1, 12)

            # 将字符串数组拆分为年、月、日，然后转换为整数数组
            obj = np.char.split(dates, sep="-")
            for (i,), (y, m, d) in np.ndenumerate(obj):  # type: ignore[misc]
                self._year[i] = int(y)
                self._month[i] = int(m)
                self._day[i] = int(d)

        else:
            # 如果 dates 的类型不支持，则引发类型错误
            raise TypeError(f"{type(dates)} is not supported")

    @property
    def dtype(self) -> ExtensionDtype:
        # 返回该对象的数据类型
        return DateDtype()

    def astype(self, dtype, copy=True):
        # 将对象转换为指定的数据类型
        dtype = pandas_dtype(dtype)

        if isinstance(dtype, DateDtype):
            # 如果目标数据类型是日期类型，则复制或不复制当前对象
            data = self.copy() if copy else self
        else:
            # 否则，将对象转换为 numpy 数组，同时处理缺失值
            data = self.to_numpy(dtype=dtype, copy=copy, na_value=dt.date.min)

        return data

    @property
    def nbytes(self) -> int:
        # 返回存储在对象中的字节数总和
        return self._year.nbytes + self._month.nbytes + self._day.nbytes

    def __len__(self) -> int:
        # 返回对象中年、月、日数组的长度，它们应该是相等的
        return len(self._year)  # 所有三个数组都应有相同的长度
    # 定义类方法，用于通过索引访问元素
    def __getitem__(self, item: PositionalIndexer):
        # 如果索引是整数，返回对应日期的 datetime.date 对象
        if isinstance(item, int):
            return dt.date(self._year[item], self._month[item], self._day[item])
        else:
            # 如果索引不是整数，抛出未实现的异常
            raise NotImplementedError("only ints are supported as indexes")

    # 定义类方法，用于通过索引设置元素
    def __setitem__(self, key: int | slice | np.ndarray, value: Any) -> None:
        # 如果索引不是整数，抛出未实现的异常
        if not isinstance(key, int):
            raise NotImplementedError("only ints are supported as indexes")

        # 如果设置的值不是 datetime.date 类型，抛出类型错误异常
        if not isinstance(value, dt.date):
            raise TypeError("you can only set datetime.date types")

        # 设置年、月、日数组的对应索引处的值
        self._year[key] = value.year
        self._month[key] = value.month
        self._day[key] = value.day

    # 定义类方法，用于返回对象的字符串表示形式
    def __repr__(self) -> str:
        # 返回对象的字符串表示形式，包含年、月、日的列表
        return f"DateArray{list(zip(self._year, self._month, self._day))}"

    # 定义实例方法，用于复制对象
    def copy(self) -> DateArray:
        # 返回当前对象的副本，包括年、月、日的副本
        return DateArray((self._year.copy(), self._month.copy(), self._day.copy()))

    # 定义实例方法，用于检查是否存在缺失值
    def isna(self) -> np.ndarray:
        # 返回一个布尔数组，表示对象中每个元素是否为缺失值
        return np.logical_and(
            np.logical_and(
                self._year == dt.date.min.year, self._month == dt.date.min.month
            ),
            self._day == dt.date.min.day,
        )

    # 定义类方法，用于从序列创建对象
    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None = None, copy=False):
        # 如果输入是 datetime.date 对象，抛出类型错误异常
        if isinstance(scalars, dt.date):
            raise TypeError
        # 如果输入是 DateArray 对象
        elif isinstance(scalars, DateArray):
            # 如果指定了数据类型 dtype，则转换为指定类型的 DateArray 对象
            if dtype is not None:
                return scalars.astype(dtype, copy=copy)
            # 如果需要复制对象，则返回对象的副本
            if copy:
                return scalars.copy()
            # 否则返回对象的切片
            return scalars[:]
        # 如果输入是 numpy 数组
        elif isinstance(scalars, np.ndarray):
            # 将 numpy 数组转换为字符串数组，每个日期占用 10 个字符，格式为 yyyy-mm-dd
            scalars = scalars.astype("U10")  # 10 chars for yyyy-mm-dd
            # 返回由字符串数组创建的 DateArray 对象
            return DateArray(scalars)
```