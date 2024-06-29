# `D:\src\scipysrc\pandas\pandas\core\dtypes\generic.py`

```
"""define generic base classes for pandas objects"""

# 导入必要的模块和类型定义
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Type,
    cast,
)

# 如果类型检查开启，则从pandas库中导入各种索引和数据结构类型
if TYPE_CHECKING:
    from pandas import (
        Categorical,
        CategoricalIndex,
        DataFrame,
        DatetimeIndex,
        Index,
        IntervalIndex,
        MultiIndex,
        PeriodIndex,
        RangeIndex,
        Series,
        TimedeltaIndex,
    )
    from pandas.core.arrays import (
        DatetimeArray,
        ExtensionArray,
        NumpyExtensionArray,
        PeriodArray,
        TimedeltaArray,
    )
    from pandas.core.generic import NDFrame


# 定义一个函数用于创建抽象基类，以便可以使用 isinstance 进行类型检查
def create_pandas_abc_type(name, attr, comp) -> type:
    # 内部函数，用于检查对象的类型是否符合要求
    def _check(inst) -> bool:
        return getattr(inst, attr, "_typ") in comp

    # 类方法，用于检查实例是否符合指定的类型
    @classmethod  # type: ignore[misc]
    def _instancecheck(cls, inst) -> bool:
        return _check(inst) and not isinstance(inst, type)

    # 类方法，用于检查子类是否符合指定的类型
    @classmethod  # type: ignore[misc]
    def _subclasscheck(cls, inst) -> bool:
        # 如果传入的 inst 不是类对象，则抛出 TypeError 异常
        if not isinstance(inst, type):
            raise TypeError("issubclass() arg 1 must be a class")
        # 否则，调用 _check 函数检查是否符合类型要求
        return _check(inst)

    # 创建一个新的类型，名为 name，基类为 type，包含上面定义的类方法
    dct = {"__instancecheck__": _instancecheck, "__subclasscheck__": _subclasscheck}
    meta = type("ABCBase", (type,), dct)
    return meta(name, (), dct)


# 以下是一系列具体的抽象基类定义，每个都是通过 create_pandas_abc_type 函数创建的
ABCRangeIndex = cast(
    "Type[RangeIndex]",
    create_pandas_abc_type("ABCRangeIndex", "_typ", ("rangeindex",)),
)
ABCMultiIndex = cast(
    "Type[MultiIndex]",
    create_pandas_abc_type("ABCMultiIndex", "_typ", ("multiindex",)),
)
ABCDatetimeIndex = cast(
    "Type[DatetimeIndex]",
    create_pandas_abc_type("ABCDatetimeIndex", "_typ", ("datetimeindex",)),
)
ABCTimedeltaIndex = cast(
    "Type[TimedeltaIndex]",
    create_pandas_abc_type("ABCTimedeltaIndex", "_typ", ("timedeltaindex",)),
)
ABCPeriodIndex = cast(
    "Type[PeriodIndex]",
    create_pandas_abc_type("ABCPeriodIndex", "_typ", ("periodindex",)),
)
ABCCategoricalIndex = cast(
    "Type[CategoricalIndex]",
    create_pandas_abc_type("ABCCategoricalIndex", "_typ", ("categoricalindex",)),
)
ABCIntervalIndex = cast(
    "Type[IntervalIndex]",
    create_pandas_abc_type("ABCIntervalIndex", "_typ", ("intervalindex",)),
)
ABCIndex = cast(
    "Type[Index]",
    create_pandas_abc_type(
        "ABCIndex",
        "_typ",
        {
            "index",
            "rangeindex",
            "multiindex",
            "datetimeindex",
            "timedeltaindex",
            "periodindex",
            "categoricalindex",
            "intervalindex",
        },
    ),
)

# 定义一个抽象基类，用于表示 pandas 的核心数据结构
ABCNDFrame = cast(
    "Type[NDFrame]",
    create_pandas_abc_type("ABCNDFrame", "_typ", ("series", "dataframe")),
)
ABCSeries = cast(
    "Type[Series]",
    create_pandas_abc_type("ABCSeries", "_typ", ("series",)),
)
    create_pandas_abc_type("ABCSeries", "_typ", ("series",)),


# 使用函数 create_pandas_abc_type 创建名为 "ABCSeries" 的抽象基类类型，传入参数 "_typ" 和元组 ("series",)
# 创建一个类型别名 ABCDataFrame，表示它是一个 DataFrame 类型的子类或相关类型
ABCDataFrame = cast(
    "Type[DataFrame]", create_pandas_abc_type("ABCDataFrame", "_typ", ("dataframe",))
)

# 创建一个类型别名 ABCCategorical，表示它是一个 Categorical 类型的子类或相关类型
ABCCategorical = cast(
    "Type[Categorical]",
    create_pandas_abc_type("ABCCategorical", "_typ", ("categorical")),
)

# 创建一个类型别名 ABCDatetimeArray，表示它是一个 DatetimeArray 类型的子类或相关类型
ABCDatetimeArray = cast(
    "Type[DatetimeArray]",
    create_pandas_abc_type("ABCDatetimeArray", "_typ", ("datetimearray")),
)

# 创建一个类型别名 ABCTimedeltaArray，表示它是一个 TimedeltaArray 类型的子类或相关类型
ABCTimedeltaArray = cast(
    "Type[TimedeltaArray]",
    create_pandas_abc_type("ABCTimedeltaArray", "_typ", ("timedeltaarray")),
)

# 创建一个类型别名 ABCPeriodArray，表示它是一个 PeriodArray 类型的子类或相关类型
ABCPeriodArray = cast(
    "Type[PeriodArray]",
    create_pandas_abc_type("ABCPeriodArray", "_typ", ("periodarray",)),
)

# 创建一个类型别名 ABCExtensionArray，表示它是一个 ExtensionArray 类型的子类或相关类型
ABCExtensionArray = cast(
    "Type[ExtensionArray]",
    create_pandas_abc_type(
        "ABCExtensionArray",
        "_typ",
        # 注意：IntervalArray 和 SparseArray 被包括在内，因为它们的 _typ="extension"
        {"extension", "categorical", "periodarray", "datetimearray", "timedeltaarray"},
    ),
)

# 创建一个类型别名 ABCNumpyExtensionArray，表示它是一个 NumpyExtensionArray 类型的子类或相关类型
ABCNumpyExtensionArray = cast(
    "Type[NumpyExtensionArray]",
    create_pandas_abc_type("ABCNumpyExtensionArray", "_typ", ("npy_extension",)),
)
```