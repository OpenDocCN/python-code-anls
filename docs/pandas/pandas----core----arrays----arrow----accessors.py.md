# `D:\src\scipysrc\pandas\pandas\core\arrays\arrow\accessors.py`

```
"""Accessors for arrow-backed data."""

from __future__ import annotations

from abc import (
    ABCMeta,  # 导入抽象基类元类
    abstractmethod,  # 导入抽象方法装饰器
)
from typing import (
    TYPE_CHECKING,  # 导入类型检查标志
    cast,  # 强制类型转换函数
)

from pandas.compat import (
    pa_version_under10p1,  # 检查 pyarrow 版本是否小于 10.1
    pa_version_under11p0,  # 检查 pyarrow 版本是否小于 11.0
)

from pandas.core.dtypes.common import is_list_like  # 导入判断是否为类列表的函数

if not pa_version_under10p1:
    import pyarrow as pa  # 导入 pyarrow 库
    import pyarrow.compute as pc  # 导入 pyarrow 的 compute 模块

    from pandas.core.dtypes.dtypes import ArrowDtype  # 导入自定义的 ArrowDtype 类型

if TYPE_CHECKING:
    from collections.abc import Iterator  # 导入迭代器抽象基类

    from pandas import (
        DataFrame,  # 导入 DataFrame 类
        Series,  # 导入 Series 类
    )


class ArrowAccessor(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, data, validation_msg: str) -> None:
        self._data = data  # 初始化数据对象
        self._validation_msg = validation_msg  # 初始化验证消息
        self._validate(data)  # 调用验证方法对数据进行验证

    @abstractmethod
    def _is_valid_pyarrow_dtype(self, pyarrow_dtype) -> bool:
        pass  # 抽象方法，子类需实现对 pyarrow 数据类型的验证

    def _validate(self, data) -> None:
        dtype = data.dtype  # 获取数据的 dtype
        if not isinstance(dtype, ArrowDtype):
            # 如果 dtype 不是 ArrowDtype 类型，则抛出 AttributeError 异常
            raise AttributeError(self._validation_msg.format(dtype=dtype))

        if not self._is_valid_pyarrow_dtype(dtype.pyarrow_dtype):
            # 如果 pyarrow 数据类型不合法，则抛出 AttributeError 异常
            raise AttributeError(self._validation_msg.format(dtype=dtype))

    @property
    def _pa_array(self):
        return self._data.array._pa_array  # 返回数据对象的 pyarrow 数组属性


class ListAccessor(ArrowAccessor):
    """
    Accessor object for list data properties of the Series values.

    Parameters
    ----------
    data : Series
        Series containing Arrow list data.
    """

    def __init__(self, data=None) -> None:
        super().__init__(
            data,
            validation_msg="Can only use the '.list' accessor with "
            "'list[pyarrow]' dtype, not {dtype}.",
        )  # 调用父类构造函数，初始化数据和验证消息

    def _is_valid_pyarrow_dtype(self, pyarrow_dtype) -> bool:
        return (
            pa.types.is_list(pyarrow_dtype)  # 检查是否为列表类型
            or pa.types.is_fixed_size_list(pyarrow_dtype)  # 检查是否为固定大小列表类型
            or pa.types.is_large_list(pyarrow_dtype)  # 检查是否为大列表类型
        )

    def len(self) -> Series:
        """
        Return the length of each list in the Series.

        Returns
        -------
        pandas.Series
            The length of each list.

        Examples
        --------
        >>> import pyarrow as pa
        >>> s = pd.Series(
        ...     [
        ...         [1, 2, 3],
        ...         [3],
        ...     ],
        ...     dtype=pd.ArrowDtype(pa.list_(pa.int64())),
        ... )
        >>> s.list.len()
        0    3
        1    1
        dtype: int32[pyarrow]
        """
        from pandas import Series

        value_lengths = pc.list_value_length(self._pa_array)  # 计算每个列表的长度
        return Series(
            value_lengths, dtype=ArrowDtype(value_lengths.type), index=self._data.index
        )  # 返回包含每个列表长度的 Series 对象，保留与原数据相同的索引
    # 定义特殊方法 __getitem__，用于通过索引或切片访问 Series 中的列表数据

    def __getitem__(self, key: int | slice) -> Series:
        """
        Index or slice lists in the Series.

        Parameters
        ----------
        key : int | slice
            Index or slice of indices to access from each list.

        Returns
        -------
        pandas.Series
            The list at requested index.

        Examples
        --------
        >>> import pyarrow as pa
        >>> s = pd.Series(
        ...     [
        ...         [1, 2, 3],
        ...         [3],
        ...     ],
        ...     dtype=pd.ArrowDtype(pa.list_(pa.int64())),
        ... )
        >>> s.list[0]
        0    1
        1    3
        dtype: int64[pyarrow]
        """

        # 从 pandas 库中导入 Series 类
        from pandas import Series

        # 如果 key 是整数
        if isinstance(key, int):
            # TODO: Support negative key but pyarrow does not allow
            # element index to be an array.
            # 如果 key 是负数，尝试支持，但 pyarrow 不允许元素索引为数组。
            # if key < 0:
            #     key = pc.add(key, pc.list_value_length(self._pa_array))
            
            # 获取指定索引处的列表元素
            element = pc.list_element(self._pa_array, key)
            
            # 返回一个新的 Series 对象，其中包含指定索引处的列表元素
            return Series(
                element, dtype=ArrowDtype(element.type), index=self._data.index
            )
        
        # 如果 key 是切片对象
        elif isinstance(key, slice):
            # 如果 pyarrow 的版本低于 11.0，暂不支持列表切片
            if pa_version_under11p0:
                raise NotImplementedError(
                    f"List slice not supported by pyarrow {pa.__version__}."
                )
            
            # TODO: Support negative start/stop/step, ideally this would be added
            # upstream in pyarrow.
            # 解析切片的起始、终止和步长
            start, stop, step = key.start, key.stop, key.step
            
            # 如果起始索引为 None，则设置为 0
            if start is None:
                # TODO: When adding negative step support
                #  this should be setto last element of array
                # when step is negative.
                start = 0
            
            # 如果步长为 None，则设置为 1
            if step is None:
                step = 1
            
            # 对列表进行切片操作
            sliced = pc.list_slice(self._pa_array, start, stop, step)
            
            # 返回一个新的 Series 对象，包含切片后的列表数据
            return Series(sliced, dtype=ArrowDtype(sliced.type), index=self._data.index)
        
        # 如果 key 类型既不是整数也不是切片，则抛出异常
        else:
            raise ValueError(f"key must be an int or slice, got {type(key).__name__}")

    # 定义特殊方法 __iter__，表示该对象不支持迭代操作，抛出类型错误异常
    def __iter__(self) -> Iterator:
        raise TypeError(f"'{type(self).__name__}' object is not iterable")
    def flatten(self) -> Series:
        """
        Flatten list values.

        Returns
        -------
        pandas.Series
            The data from all lists in the series flattened.

        Examples
        --------
        >>> import pyarrow as pa
        >>> s = pd.Series(
        ...     [
        ...         [1, 2, 3],
        ...         [3],
        ...     ],
        ...     dtype=pd.ArrowDtype(pa.list_(pa.int64())),
        ... )

        >>> s.list.flatten()
        0    1
        0    2
        0    3
        1    3
        dtype: int64[pyarrow]
        """
        # 导入 pandas 中的 Series 类
        from pandas import Series
        
        # 计算每个列表值的长度
        counts = pa.compute.list_value_length(self._pa_array)
        
        # 将列表展平
        flattened = pa.compute.list_flatten(self._pa_array)
        
        # 根据原始数据的索引重复索引，以匹配展平后数据的长度
        index = self._data.index.repeat(counts.fill_null(pa.scalar(0, counts.type)))
        
        # 返回一个新的 Series 对象，包含展平后的数据，指定数据类型和索引
        return Series(flattened, dtype=ArrowDtype(flattened.type), index=index)
class StructAccessor(ArrowAccessor):
    """
    Accessor object for structured data properties of the Series values.

    Parameters
    ----------
    data : Series
        Series containing Arrow struct data.
    """

    def __init__(self, data=None) -> None:
        # 调用父类的构造函数，初始化数据验证信息
        super().__init__(
            data,
            validation_msg=(
                "Can only use the '.struct' accessor with 'struct[pyarrow]' "
                "dtype, not {dtype}."
            ),
        )

    def _is_valid_pyarrow_dtype(self, pyarrow_dtype) -> bool:
        # 检查给定的 PyArrow 数据类型是否为 struct 类型
        return pa.types.is_struct(pyarrow_dtype)

    @property
    def dtypes(self) -> Series:
        """
        Return the dtype object of each child field of the struct.

        Returns
        -------
        pandas.Series
            The data type of each child field.

        Examples
        --------
        >>> import pyarrow as pa
        >>> s = pd.Series(
        ...     [
        ...         {"version": 1, "project": "pandas"},
        ...         {"version": 2, "project": "pandas"},
        ...         {"version": 1, "project": "numpy"},
        ...     ],
        ...     dtype=pd.ArrowDtype(
        ...         pa.struct([("version", pa.int64()), ("project", pa.string())])
        ...     ),
        ... )
        >>> s.struct.dtypes
        version     int64[pyarrow]
        project    string[pyarrow]
        dtype: object
        """
        from pandas import (
            Index,
            Series,
        )

        # 获取 PyArrow 数据类型对象
        pa_type = self._data.dtype.pyarrow_dtype
        # 生成每个子字段的 ArrowDtype 对象
        types = [ArrowDtype(struct.type) for struct in pa_type]
        # 获取每个子字段的名称
        names = [struct.name for struct in pa_type]
        # 返回一个 Series 对象，包含每个子字段的数据类型，以子字段名为索引
        return Series(types, index=Index(names))

    def field(
        self,
        name_or_index: list[str]
        | list[bytes]
        | list[int]
        | pc.Expression
        | bytes
        | str
        | int,
    ):
        # 从结构体中提取单个子字段的值
        ...

    def explode(self) -> DataFrame:
        """
        Extract all child fields of a struct as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            The data corresponding to all child fields.

        See Also
        --------
        Series.struct.field : Return a single child field as a Series.

        Examples
        --------
        >>> import pyarrow as pa
        >>> s = pd.Series(
        ...     [
        ...         {"version": 1, "project": "pandas"},
        ...         {"version": 2, "project": "pandas"},
        ...         {"version": 1, "project": "numpy"},
        ...     ],
        ...     dtype=pd.ArrowDtype(
        ...         pa.struct([("version", pa.int64()), ("project", pa.string())])
        ...     ),
        ... )

        >>> s.struct.explode()
           version project
        0        1  pandas
        1        2  pandas
        2        1   numpy
        """
        from pandas import concat

        # 获取 PyArrow 数组的类型信息
        pa_type = self._pa_array.type
        # 对结构体的每个字段调用 field 方法，将结果拼接成一个 DataFrame
        return concat(
            [self.field(i) for i in range(pa_type.num_fields)], axis="columns"
        )
```