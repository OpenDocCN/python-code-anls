# `D:\src\scipysrc\pandas\pandas\core\methods\describe.py`

```
"""
Module responsible for execution of NDFrame.describe() method.

Method NDFrame.describe() delegates actual execution to function describe_ndframe().
"""

from __future__ import annotations

from abc import (  # 导入抽象基类相关模块
    ABC,  # Python 中的抽象基类
    abstractmethod,  # 用于声明抽象方法
)
from typing import (  # 导入类型提示相关模块
    TYPE_CHECKING,  # 类型检查标志
    cast,  # 用于类型转换
)

import numpy as np  # 导入 NumPy 库

from pandas._typing import (  # 导入 Pandas 类型相关模块
    DtypeObj,  # Pandas 中的数据类型对象
    NDFrameT,  # Pandas 中的 NDFrame 类型
    npt,  # NumPy 的类型提示
)
from pandas.util._validators import validate_percentile  # 导入百分位数验证函数

from pandas.core.dtypes.common import (  # 导入 Pandas 中常见数据类型相关模块
    is_bool_dtype,  # 判断是否为布尔类型数据
    is_numeric_dtype,  # 判断是否为数值类型数据
)
from pandas.core.dtypes.dtypes import (  # 导入 Pandas 中数据类型相关模块
    ArrowDtype,  # Arrow 数据类型
    DatetimeTZDtype,  # 带时区的日期时间数据类型
    ExtensionDtype,  # 扩展数据类型
)

from pandas.core.arrays.floating import Float64Dtype  # 导入 Pandas 浮点数数据类型模块
from pandas.core.reshape.concat import concat  # 导入 Pandas 连接函数

from pandas.io.formats.format import format_percentiles  # 导入 Pandas 格式化百分位数函数

if TYPE_CHECKING:
    from collections.abc import (  # 导入集合抽象基类相关模块
        Callable,  # 可调用对象
        Hashable,  # 可散列对象
        Sequence,  # 序列类型
    )

    from pandas import (  # 导入 Pandas 库中 DataFrame 和 Series 类型
        DataFrame,  # 数据帧类型
        Series,  # 系列类型
    )


def describe_ndframe(  # 定义 describe_ndframe 函数，用于描述 DataFrame 或 Series
    *,
    obj: NDFrameT,  # NDFrameT 类型的参数 obj，可以是 DataFrame 或 Series
    include: str | Sequence[str] | None,  # 包含的数据类型白名单或 'all' 或 None
    exclude: str | Sequence[str] | None,  # 排除的数据类型黑名单或 None
    percentiles: Sequence[float] | np.ndarray | None,  # 要包含在输出中的百分位数列表或数组，可选
) -> NDFrameT:  # 返回类型为 NDFrameT（DataFrame 或 Series）
    """Describe series or dataframe.

    Called from pandas.core.generic.NDFrame.describe()

    Parameters
    ----------
    obj: DataFrame or Series
        Either dataframe or series to be described.
    include : 'all', list-like of dtypes or None (default), optional
        A white list of data types to include in the result. Ignored for ``Series``.
    exclude : list-like of dtypes or None (default), optional,
        A black list of data types to omit from the result. Ignored for ``Series``.
    percentiles : list-like of numbers, optional
        The percentiles to include in the output. All should fall between 0 and 1.
        The default is ``[.25, .5, .75]``, which returns the 25th, 50th, and
        75th percentiles.

    Returns
    -------
    Dataframe or series description.
    """
    percentiles = _refine_percentiles(percentiles)  # 调用 _refine_percentiles 函数处理百分位数列表

    describer: NDFrameDescriberAbstract  # 声明 describer 变量类型为 NDFrameDescriberAbstract

    if obj.ndim == 1:  # 如果 obj 的维度为 1，即为 Series
        describer = SeriesDescriber(  # 创建 SeriesDescriber 实例
            obj=cast("Series", obj),  # 将 obj 转换为 Series 类型
        )
    else:  # 否则，为 DataFrame
        describer = DataFrameDescriber(  # 创建 DataFrameDescriber 实例
            obj=cast("DataFrame", obj),  # 将 obj 转换为 DataFrame 类型
            include=include,  # 设置 include 参数
            exclude=exclude,  # 设置 exclude 参数
        )

    result = describer.describe(percentiles=percentiles)  # 调用 describer 的 describe 方法进行描述
    return cast(NDFrameT, result)  # 返回结果，并将其类型转换为 NDFrameT


class NDFrameDescriberAbstract(ABC):
    """Abstract class for describing dataframe or series.

    Parameters
    ----------
    obj : Series or DataFrame
        Object to be described.
    """

    def __init__(self, obj: DataFrame | Series) -> None:
        self.obj = obj  # 初始化 obj 属性为传入的 DataFrame 或 Series 对象

    @abstractmethod
    def describe(self, percentiles: Sequence[float]) -> NDFrameT:
        """Abstract method to describe the object.

        Parameters
        ----------
        percentiles : list-like of numbers
            The percentiles to include in the output.

        Returns
        -------
        Dataframe or series description.
        """
        pass  # 抽象方法，子类需实现
    # 定义一个方法 describe，用于描述数据集（可以是 Series 或 DataFrame）

    def describe(self, percentiles: Sequence[float] | np.ndarray) -> DataFrame | Series:
        """Do describe either series or dataframe.

        Parameters
        ----------
        percentiles : list-like of numbers
            要包含在输出中的百分位数。

        Returns
        -------
        DataFrame or Series
            如果调用对象是 DataFrame，则返回描述统计信息的 DataFrame。
            如果调用对象是 Series，则返回描述统计信息的 Series。
        """
class SeriesDescriber(NDFrameDescriberAbstract):
    """Class responsible for creating series description."""

    obj: Series  # 声明实例变量 obj，表示该类处理的 Series 对象

    def describe(self, percentiles: Sequence[float] | np.ndarray) -> Series:
        """Generate a description of the Series object.

        Parameters
        ----------
        percentiles : Sequence[float] | np.ndarray
            Percentiles to compute.

        Returns
        -------
        Series
            Descriptive statistics of the Series.
        """
        describe_func = select_describe_func(
            self.obj,
        )  # 调用辅助函数 select_describe_func 选择合适的描述函数
        return describe_func(self.obj, percentiles)  # 调用选定的描述函数处理 Series 对象


class DataFrameDescriber(NDFrameDescriberAbstract):
    """Class responsible for creating dataobj description.

    Parameters
    ----------
    obj : DataFrame
        DataFrame to be described.
    include : 'all', list-like of dtypes or None
        A white list of data types to include in the result.
    exclude : list-like of dtypes or None
        A black list of data types to omit from the result.
    """

    obj: DataFrame  # 声明实例变量 obj，表示该类处理的 DataFrame 对象

    def __init__(
        self,
        obj: DataFrame,
        *,
        include: str | Sequence[str] | None,
        exclude: str | Sequence[str] | None,
    ) -> None:
        """Initialize DataFrameDescriber object.

        Parameters
        ----------
        obj : DataFrame
            DataFrame to be described.
        include : str | Sequence[str] | None
            Specifies which data types to include in the description.
        exclude : str | Sequence[str] | None
            Specifies which data types to exclude from the description.

        Raises
        ------
        ValueError
            If DataFrame has no columns when it's expected to have 2D structure.
        """
        self.include = include  # 初始化 include 属性
        self.exclude = exclude  # 初始化 exclude 属性

        if obj.ndim == 2 and obj.columns.size == 0:
            raise ValueError("Cannot describe a DataFrame without columns")

        super().__init__(obj)  # 调用父类的构造函数初始化

    def describe(self, percentiles: Sequence[float] | np.ndarray) -> DataFrame:
        """Generate a description of the DataFrame object.

        Parameters
        ----------
        percentiles : Sequence[float] | np.ndarray
            Percentiles to compute.

        Returns
        -------
        DataFrame
            Descriptive statistics of the DataFrame.
        """
        data = self._select_data()  # 调用内部方法选择需要描述的数据

        ldesc: list[Series] = []
        for _, series in data.items():
            describe_func = select_describe_func(series)  # 选择适合该 Series 的描述函数
            ldesc.append(describe_func(series, percentiles))  # 将描述结果添加到列表中

        col_names = reorder_columns(ldesc)  # 重新排序列名
        d = concat(
            [x.reindex(col_names) for x in ldesc],  # 按照新顺序重新索引每个 Series
            axis=1,
            ignore_index=True,
            sort=False,
        )
        d.columns = data.columns.copy()  # 设置 DataFrame 的列名为原始数据的列名副本
        return d

    def _select_data(self) -> DataFrame:
        """Select columns to be described."""
        if (self.include is None) and (self.exclude is None):
            # 当没有指定包含或排除条件时，选择数值型和日期时间类型的列
            default_include: list[npt.DTypeLike] = [np.number, "datetime"]
            data = self.obj.select_dtypes(include=default_include)
            if len(data.columns) == 0:
                data = self.obj  # 如果未找到符合条件的列，则使用所有列
        elif self.include == "all":
            if self.exclude is not None:
                msg = "exclude must be None when include is 'all'"
                raise ValueError(msg)
            data = self.obj  # 包含所有列
        else:
            data = self.obj.select_dtypes(
                include=self.include,
                exclude=self.exclude,
            )  # 根据指定的包含和排除条件选择列
        return data


def reorder_columns(ldesc: Sequence[Series]) -> list[Hashable]:
    """Set a convenient order for rows for display.

    Parameters
    ----------
    ldesc : Sequence[Series]
        List of Series objects to reorder.

    Returns
    -------
    list[Hashable]
        Ordered list of column names or indices.
    """
    names: list[Hashable] = []
    seen_names: set[Hashable] = set()
    ldesc_indexes = sorted((x.index for x in ldesc), key=len)  # 按索引长度排序 Series
    for idxnames in ldesc_indexes:
        for name in idxnames:
            if name not in seen_names:
                seen_names.add(name)
                names.append(name)  # 将未重复的索引名添加到列表中
    return names  # 返回重新排序后的列名列表
# 描述包含数值数据的序列。
def describe_numeric_1d(series: Series, percentiles: Sequence[float]) -> Series:
    # 从 pandas 库导入 Series 类

    # 格式化百分位数
    formatted_percentiles = format_percentiles(percentiles)

    # 统计指标名称列表
    stat_index = ["count", "mean", "std", "min"] + formatted_percentiles + ["max"]

    # 计算数据集的统计指标值，包括计数、均值、标准差、最小值、指定百分位数、最大值
    d = (
        [series.count(), series.mean(), series.std(), series.min()]
        + series.quantile(percentiles).tolist()
        + [series.max()]
    )

    # GH#48340 - 对于非复数数值数据，始终返回 float 类型
    dtype: DtypeObj | None
    if isinstance(series.dtype, ExtensionDtype):
        if isinstance(series.dtype, ArrowDtype):
            if series.dtype.kind == "m":
                # GH53001: 对于时间增量数据使用对象类型
                dtype = None
            else:
                import pyarrow as pa

                dtype = ArrowDtype(pa.float64())
        else:
            dtype = Float64Dtype()
    elif series.dtype.kind in "iufb":
        # 即数值型但排除复数类型
        dtype = np.dtype("float")
    else:
        dtype = None

    # 返回 Series 对象，包括统计指标值和对应的索引、名称、数据类型
    return Series(d, index=stat_index, name=series.name, dtype=dtype)


# 描述包含分类数据的序列。
def describe_categorical_1d(
    data: Series,
    percentiles_ignored: Sequence[float],
) -> Series:
    # 统计指标名称列表
    names = ["count", "unique", "top", "freq"]

    # 统计每个分类值的出现次数
    objcounts = data.value_counts()

    # 计算唯一值的数量
    count_unique = len(objcounts[objcounts != 0])

    # 如果存在唯一值，确定最频繁出现的值及其频率
    if count_unique > 0:
        top, freq = objcounts.index[0], objcounts.iloc[0]
        dtype = None
    else:
        # 如果数据集为空，设置 'top' 和 'freq' 为 None，以保持输出形状的一致性
        top, freq = np.nan, np.nan
        dtype = "object"

    # 返回包含统计指标值的 Series 对象，包括计数、唯一值数量、最频繁出现值及其频率
    return Series([data.count(), count_unique, top, freq], index=names, name=data.name, dtype=dtype)


# 描述包含 datetime64 类型数据的序列。
def describe_timestamp_1d(data: Series, percentiles: Sequence[float]) -> Series:
    # 从 pandas 库导入 Series 类

    # GH-30164

    # 格式化百分位数
    formatted_percentiles = format_percentiles(percentiles)

    # 统计指标名称列表
    stat_index = ["count", "mean", "min"] + formatted_percentiles + ["max"]

    # 计算数据集的统计指标值，包括计数、均值、最小值、指定百分位数、最大值
    d = (
        [data.count(), data.mean(), data.min()]
        + data.quantile(percentiles).tolist()
        + [data.max()]
    )

    # 返回包含统计指标值的 Series 对象，包括索引、名称
    return Series(d, index=stat_index, name=data.name)


# 选择适用于描述数据的函数
def select_describe_func(
    data: Series,
def _refine_percentiles(
    percentiles: Sequence[float] | np.ndarray | None,  # 定义函数 `_refine_percentiles`，接受一个参数 `percentiles`，可以是浮点数序列、NumPy 数组或空值

) -> npt.NDArray[np.float64]:  # 函数返回一个 NumPy 数组，包含浮点数

    """
    Ensure that percentiles are unique and sorted.

    Parameters
    ----------
    percentiles : list-like of numbers, optional
        The percentiles to include in the output.
    """

    if percentiles is None:  # 如果 `percentiles` 参数为空
        return np.array([0.25, 0.5, 0.75])  # 返回默认的 percentiles 数组：0.25, 0.5, 0.75

    # explicit conversion of `percentiles` to list
    percentiles = list(percentiles)  # 将 percentiles 转换为列表

    # get them all to be in [0, 1]
    validate_percentile(percentiles)  # 验证 percentiles 是否在 [0, 1] 范围内

    # median should always be included
    if 0.5 not in percentiles:  # 如果中位数（0.5）不在 percentiles 中，则加入
        percentiles.append(0.5)

    percentiles = np.asarray(percentiles)  # 将 percentiles 转换为 NumPy 数组

    # sort and check for duplicates
    unique_pcts = np.unique(percentiles)  # 对 percentiles 进行排序并去重
    assert percentiles is not None  # 断言 percentiles 不为空
    if len(unique_pcts) < len(percentiles):  # 如果去重后的数组长度小于原数组长度，则抛出 ValueError
        raise ValueError("percentiles cannot contain duplicates")

    return unique_pcts  # 返回经过去重和排序后的 percentiles 数组
```