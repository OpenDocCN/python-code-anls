# `D:\src\scipysrc\pandas\pandas\core\reshape\encoding.py`

```
# 导入必要的库和模块
from __future__ import annotations  # 允许在类型提示中使用类型本身

from collections import defaultdict  # 导入 defaultdict 类
from collections.abc import (  # 导入抽象基类中的 Hashable 和 Iterable
    Hashable,
    Iterable,
)
import itertools  # 导入 itertools 模块
from typing import TYPE_CHECKING  # 导入 TYPE_CHECKING 类型提示

import numpy as np  # 导入 numpy 库

from pandas._libs.sparse import IntIndex  # 导入 pandas 稀疏数据索引

from pandas.core.dtypes.common import (  # 导入 pandas 中的数据类型检查函数
    is_integer_dtype,
    is_list_like,
    is_object_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (  # 导入 pandas 中的特殊数据类型
    ArrowDtype,
    CategoricalDtype,
)

from pandas.core.arrays import SparseArray  # 导入 pandas 中的稀疏数组
from pandas.core.arrays.categorical import factorize_from_iterable  # 导入从可迭代对象创建分类变量的函数
from pandas.core.arrays.string_ import StringDtype  # 导入 pandas 中的字符串类型
from pandas.core.frame import DataFrame  # 导入 pandas 数据帧
from pandas.core.indexes.api import (  # 导入 pandas 索引相关函数
    Index,
    default_index,
)
from pandas.core.series import Series  # 导入 pandas 系列类型

if TYPE_CHECKING:
    from pandas._typing import NpDtype  # 导入用于类型提示的 NpDtype 类型

def get_dummies(  # 定义函数 get_dummies，用于将分类变量转换为哑变量
    data,  # 输入的数据，可以是 array-like, Series 或 DataFrame
    prefix=None,  # 列名前缀，默认为 None
    prefix_sep: str | Iterable[str] | dict[str, str] = "_",  # 前缀分隔符，默认为 '_'
    dummy_na: bool = False,  # 是否添加 NaN 变量，默认为 False
    columns=None,  # 要编码的列名列表，默认为 None
    sparse: bool = False,  # 是否使用稀疏数组存储编码结果，默认为 False
    drop_first: bool = False,  # 是否删除第一个级别的哑变量，默认为 False
    dtype: NpDtype | None = None,  # 新列的数据类型，默认为 bool 类型
) -> DataFrame:  # 返回 DataFrame 类型的哑变量编码结果
    """
    Convert categorical variable into dummy/indicator variables.

    Each variable is converted in as many 0/1 variables as there are different
    values. Columns in the output are each named after a value; if the input is
    a DataFrame, the name of the original variable is prepended to the value.

    Parameters
    ----------
    data : array-like, Series, or DataFrame
        Data of which to get dummy indicators.
    prefix : str, list of str, or dict of str, default None
        String to append DataFrame column names.
        Pass a list with length equal to the number of columns
        when calling get_dummies on a DataFrame. Alternatively, `prefix`
        can be a dictionary mapping column names to prefixes.
    prefix_sep : str, default '_'
        If appending prefix, separator/delimiter to use. Or pass a
        list or dictionary as with `prefix`.
    dummy_na : bool, default False
        Add a column to indicate NaNs, if False NaNs are ignored.
    columns : list-like, default None
        Column names in the DataFrame to be encoded.
        If `columns` is None then all the columns with
        `object`, `string`, or `category` dtype will be converted.
    sparse : bool, default False
        Whether the dummy-encoded columns should be backed by
        a :class:`SparseArray` (True) or a regular NumPy array (False).
    drop_first : bool, default False
        Whether to get k-1 dummies out of k categorical levels by removing the
        first level.
    dtype : dtype, default bool
        Data type for new columns. Only a single dtype is allowed.

    Returns
    -------
    DataFrame
        Dummy-coded data. If `data` contains other columns than the
        dummy-coded one(s), these will be prepended, unaltered, to the result.

    See Also
    --------
    Series.str.get_dummies : Convert Series of strings to dummy codes.
    """
    # 导入 pandas 库中的 concat 函数，用于连接数据框
    from pandas.core.reshape.concat import concat
    
    # 定义需要编码的数据类型列表，包括对象、字符串和分类类型
    dtypes_to_encode = ["object", "string", "category"]
    # 检查数据类型是否为 DataFrame
    if isinstance(data, DataFrame):
        # 确定要进行编码的列
        if columns is None:
            # 选择需要编码的数据类型列
            data_to_encode = data.select_dtypes(include=dtypes_to_encode)
        elif not is_list_like(columns):
            # 如果 columns 不是类列表对象，则抛出类型错误
            raise TypeError("Input must be a list-like for parameter `columns`")
        else:
            # 根据指定的 columns 获取需要编码的数据列
            data_to_encode = data[columns]

        # 验证前缀和分隔符，避免静默丢弃列
        def check_len(item, name: str) -> None:
            # 检查长度是否匹配
            if is_list_like(item):
                if not len(item) == data_to_encode.shape[1]:
                    len_msg = (
                        f"Length of '{name}' ({len(item)}) did not match the "
                        "length of the columns being encoded "
                        f"({data_to_encode.shape[1]})."
                    )
                    raise ValueError(len_msg)

        # 检查前缀长度是否与列数匹配
        check_len(prefix, "prefix")
        # 检查分隔符长度是否与列数匹配
        check_len(prefix_sep, "prefix_sep")

        # 如果前缀是字符串，则使用循环生成器
        if isinstance(prefix, str):
            prefix = itertools.cycle([prefix])
        # 如果前缀是字典，则根据列名从字典中获取前缀列表
        if isinstance(prefix, dict):
            prefix = [prefix[col] for col in data_to_encode.columns]

        # 如果前缀为空，则使用数据集的列名作为前缀
        if prefix is None:
            prefix = data_to_encode.columns

        # 验证分隔符，如果是字符串，则使用循环生成器
        if isinstance(prefix_sep, str):
            prefix_sep = itertools.cycle([prefix_sep])
        # 如果分隔符是字典，则根据列名从字典中获取分隔符列表
        elif isinstance(prefix_sep, dict):
            prefix_sep = [prefix_sep[col] for col in data_to_encode.columns]

        # 初始化用于存放编码结果的列表
        with_dummies: list[DataFrame]
        # 如果要编码的数据集与原始数据集形状相同，则不添加任何丢弃的列
        if data_to_encode.shape == data.shape:
            with_dummies = []
        # 如果指定了 columns 参数，则只编码指定的列，其他列添加到结果前面
        elif columns is not None:
            with_dummies = [data.drop(columns, axis=1)]
        else:
            # 只编码对象和类别数据类型列，其他列添加到结果前面
            with_dummies = [data.select_dtypes(exclude=dtypes_to_encode)]

        # 遍历需要编码的列及其对应的前缀和分隔符
        for col, pre, sep in zip(data_to_encode.items(), prefix, prefix_sep):
            # 获取单列的哑变量编码
            dummy = _get_dummies_1d(
                col[1],
                prefix=pre,
                prefix_sep=sep,
                dummy_na=dummy_na,
                sparse=sparse,
                drop_first=drop_first,
                dtype=dtype,
            )
            # 将编码结果添加到 with_dummies 列表中
            with_dummies.append(dummy)
        # 将所有编码结果连接成最终的 DataFrame
        result = concat(with_dummies, axis=1)
    else:
        # 对于非 DataFrame 类型的数据，进行单列的哑变量编码
        result = _get_dummies_1d(
            data,
            prefix,
            prefix_sep,
            dummy_na,
            sparse=sparse,
            drop_first=drop_first,
            dtype=dtype,
        )
    # 返回编码结果
    return result
# 定义一个函数 `_get_dummies_1d`，用于将一维数据转换为虚拟变量的数据框
def _get_dummies_1d(
    data,
    prefix,
    prefix_sep: str | Iterable[str] | dict[str, str] = "_",
    dummy_na: bool = False,
    sparse: bool = False,
    drop_first: bool = False,
    dtype: NpDtype | None = None,
) -> DataFrame:
    from pandas.core.reshape.concat import concat  # 导入 pandas 中的 concat 函数用于数据合并

    # 使用 Series 来避免 NaN 处理不一致的问题，调用 factorize_from_iterable 函数处理数据
    codes, levels = factorize_from_iterable(Series(data, copy=False))

    # 如果 dtype 为 None 并且 data 具有 dtype 属性，则获取其 dtype
    if dtype is None and hasattr(data, "dtype"):
        input_dtype = data.dtype
        # 如果输入数据类型为分类类型，则获取其类别的 dtype
        if isinstance(input_dtype, CategoricalDtype):
            input_dtype = input_dtype.categories.dtype

        # 根据不同的数据类型设置 dtype
        if isinstance(input_dtype, ArrowDtype):
            import pyarrow as pa

            dtype = ArrowDtype(pa.bool_())  # 设置 ArrowDtype 为布尔类型
        elif (
            isinstance(input_dtype, StringDtype)
            and input_dtype.storage != "pyarrow_numpy"
        ):
            dtype = pandas_dtype("boolean")  # 设置 pandas dtype 为布尔类型
        else:
            dtype = np.dtype(bool)
    elif dtype is None:
        dtype = np.dtype(bool)

    _dtype = pandas_dtype(dtype)  # 将 dtype 转换为 pandas dtype

    # 如果数据类型为 object，则抛出异常，因为 object 类型不支持 get_dummies
    if is_object_dtype(_dtype):
        raise ValueError("dtype=object is not a valid dtype for get_dummies")

    # 定义一个内部函数，根据数据类型创建一个空的 DataFrame
    def get_empty_frame(data) -> DataFrame:
        index: Index | np.ndarray
        if isinstance(data, Series):
            index = data.index
        else:
            index = default_index(len(data))
        return DataFrame(index=index)

    # 如果 dummy_na 为 False 且 levels 为空，则返回一个空的 DataFrame
    if not dummy_na and len(levels) == 0:
        return get_empty_frame(data)

    codes = codes.copy()

    # 如果 dummy_na 为 True，则将 codes 中的 -1 替换为 levels 的长度，并将 np.nan 添加到 levels 中
    if dummy_na:
        codes[codes == -1] = len(levels)
        levels = levels.insert(len(levels), np.nan)

    # 如果 drop_first 为 True 且 levels 只有一个元素，则返回一个空的 DataFrame
    if drop_first and len(levels) == 1:
        return get_empty_frame(data)

    number_of_cols = len(levels)  # 获取 levels 的长度作为列数

    # 如果 prefix 为 None，则 dummy_cols 直接使用 levels
    if prefix is None:
        dummy_cols = levels
    else:
        # 否则使用 prefix、prefix_sep 和 levels 构建 dummy_cols
        dummy_cols = Index([f"{prefix}{prefix_sep}{level}" for level in levels])

    index: Index | None
    if isinstance(data, Series):
        index = data.index  # 如果 data 是 Series，则使用其索引
    else:
        index = None  # 否则索引为 None
    # 如果稀疏标志为真，则执行以下操作
    if sparse:
        # 填充值的类型可以是布尔值或浮点数
        fill_value: bool | float
        # 如果数据类型是整数类型，则填充值为0
        if is_integer_dtype(dtype):
            fill_value = 0
        # 如果数据类型是布尔类型，则填充值为False
        elif dtype == np.dtype(bool):
            fill_value = False
        # 否则，填充值为0.0
        else:
            fill_value = 0.0

        # 稀疏系列列表初始化为空
        sparse_series = []
        # 数据的长度
        N = len(data)
        # 为每个虚拟列创建一个空列表作为稀疏索引
        sp_indices: list[list] = [[] for _ in range(len(dummy_cols))]
        # 生成一个布尔掩码，排除掉值为-1的条目
        mask = codes != -1
        # 使用掩码过滤掉值为-1的条目
        codes = codes[mask]
        # 获取有效索引
        n_idx = np.arange(N)[mask]

        # 遍历有效索引和对应的编码，将索引添加到对应编码的稀疏索引列表中
        for ndx, code in zip(n_idx, codes):
            sp_indices[code].append(ndx)

        # 如果需要删除第一个分类级别以避免完美共线性
        if drop_first:
            # 删除第一个分类级别
            sp_indices = sp_indices[1:]
            dummy_cols = dummy_cols[1:]

        # 遍历虚拟列和稀疏索引列表，创建稀疏数组并添加到稀疏系列中
        for col, ixs in zip(dummy_cols, sp_indices):
            # 创建稀疏数组
            sarr = SparseArray(
                np.ones(len(ixs), dtype=dtype),  # 使用稀疏数组创建值为1的数组
                sparse_index=IntIndex(N, ixs),   # 稀疏索引
                fill_value=fill_value,           # 填充值
                dtype=dtype,                     # 数据类型
            )
            # 将稀疏数组作为数据创建系列，并添加到稀疏系列列表中
            sparse_series.append(Series(data=sarr, index=index, name=col, copy=False))

        # 将所有稀疏系列连接成一个DataFrame并沿轴1拼接返回
        return concat(sparse_series, axis=1)

    else:
        # 如果稀疏标志为假，则执行以下操作
        # 确保ndarray的布局是列主序
        shape = len(codes), number_of_cols
        # 虚拟数据类型的初始化
        dummy_dtype: NpDtype
        # 如果数据类型是np.dtype实例，则将其分配给虚拟数据类型
        if isinstance(_dtype, np.dtype):
            dummy_dtype = _dtype
        # 否则，虚拟数据类型为布尔类型
        else:
            dummy_dtype = np.bool_
        # 创建一个形状为shape的零矩阵，使用列主序布局
        dummy_mat = np.zeros(shape=shape, dtype=dummy_dtype, order="F")
        # 将虚拟矩阵中对应编码位置的元素设置为1
        dummy_mat[np.arange(len(codes)), codes] = 1

        # 如果不允许NaN值
        if not dummy_na:
            # 重置NaN值为0
            dummy_mat[codes == -1] = 0

        # 如果需要删除第一个分类级别
        if drop_first:
            # 删除第一个分类级别
            dummy_mat = dummy_mat[:, 1:]
            dummy_cols = dummy_cols[1:]

        # 返回一个DataFrame，其数据为虚拟矩阵，索引为index，列名为dummy_cols，数据类型为_dtype
        return DataFrame(dummy_mat, index=index, columns=dummy_cols, dtype=_dtype)
def from_dummies(
    data: DataFrame,
    sep: None | str = None,
    default_category: None | Hashable | dict[str, Hashable] = None,
) -> DataFrame:
    """
    Create a categorical ``DataFrame`` from a ``DataFrame`` of dummy variables.

    Inverts the operation performed by :func:`~pandas.get_dummies`.

    .. versionadded:: 1.5.0

    Parameters
    ----------
    data : DataFrame
        Data which contains dummy-coded variables in form of integer columns of
        1's and 0's.
    sep : str, default None
        Separator used in the column names of the dummy categories they are
        character indicating the separation of the categorical names from the prefixes.
        For example, if your column names are 'prefix_A' and 'prefix_B',
        you can strip the underscore by specifying sep='_'.
    default_category : None, Hashable or dict of Hashables, default None
        The default category is the implied category when a value has none of the
        listed categories specified with a one, i.e. if all dummies in a row are
        zero. Can be a single value for all variables or a dict directly mapping
        the default categories to a prefix of a variable.

    Returns
    -------
    DataFrame
        Categorical data decoded from the dummy input-data.

    Raises
    ------
    ValueError
        * When the input ``DataFrame`` ``data`` contains NA values.
        * When the input ``DataFrame`` ``data`` contains column names with separators
          that do not match the separator specified with ``sep``.
        * When a ``dict`` passed to ``default_category`` does not include an implied
          category for each prefix.
        * When a value in ``data`` has more than one category assigned to it.
        * When ``default_category=None`` and a value in ``data`` has no category
          assigned to it.
    TypeError
        * When the input ``data`` is not of type ``DataFrame``.
        * When the input ``DataFrame`` ``data`` contains non-dummy data.
        * When the passed ``sep`` is of a wrong data type.
        * When the passed ``default_category`` is of a wrong data type.

    See Also
    --------
    :func:`~pandas.get_dummies` : Convert ``Series`` or ``DataFrame`` to dummy codes.
    :class:`~pandas.Categorical` : Represent a categorical variable in classic.

    Notes
    -----
    The columns of the passed dummy data should only include 1's and 0's,
    or boolean values.

    Examples
    --------
    >>> df = pd.DataFrame({"a": [1, 0, 0, 1], "b": [0, 1, 0, 0], "c": [0, 0, 1, 0]})

    >>> df
       a  b  c
    0  1  0  0
    1  0  1  0
    2  0  0  1
    3  1  0  0

    >>> pd.from_dummies(df)
    0     a
    1     b
    2     c
    3     a

    >>> df = pd.DataFrame(
    ...     {
    ...         "col1_a": [1, 0, 1],
    ...         "col1_b": [0, 1, 0],
    ...         "col2_a": [0, 1, 0],
    ...         "col2_b": [1, 0, 0],
    ...         "col2_c": [0, 0, 1],
    ...     }
    ... )
    """

    # Create an empty DataFrame to store the decoded categorical data
    decoded_data = pd.DataFrame()

    # Iterate through each column in the input `data` DataFrame
    for col in data.columns:
        # If `sep` is specified and found in the column name, split the column name
        if sep and sep in col:
            prefix, suffix = col.rsplit(sep, 1)
        else:
            prefix = col  # Otherwise, the prefix is the entire column name

        # If `default_category` is a dictionary and `prefix` is in the keys, use the mapped default category
        if isinstance(default_category, dict) and prefix in default_category:
            category = default_category[prefix]
        else:
            category = default_category  # Otherwise, use the general default category

        # Ensure each column only contains 0's and 1's or boolean values
        if not data[col].isin([0, 1]).all():
            raise TypeError("Input DataFrame contains non-dummy data.")

        # If a column contains only 1's in a row, add the prefix to `decoded_data`
        decoded_data = decoded_data.append(data.loc[data[col] == 1].assign(**{col: prefix}))

    # Return the decoded categorical data as a DataFrame
    return decoded_data
    # 导入 pandas 中的 concat 函数，用于在后续操作中进行数据合并
    from pandas.core.reshape.concat import concat
    
    # 检查传入的数据是否为 DataFrame 类型，若不是则抛出类型错误异常
    if not isinstance(data, DataFrame):
        raise TypeError(
            "Expected 'data' to be a 'DataFrame'; "
            f"Received 'data' of type: {type(data).__name__}"
        )
    
    # 检查数据中是否存在任何 NaN 值，如果有则抛出数值错误异常，显示包含 NaN 值的列名
    col_isna_mask = data.isna().any()
    if col_isna_mask.any():
        raise ValueError(
            "Dummy DataFrame contains NA value in column: "
            f"'{col_isna_mask.idxmax()}'"
        )
    
    # 尝试将数据转换为布尔类型，以便后续操作处理虚拟变量
    try:
        data_to_decode = data.astype("boolean")
    except TypeError as err:
        raise TypeError("Passed DataFrame contains non-dummy data") from err
    
    # 创建一个 defaultdict，用于存储每个前缀对应的列名列表，以便后续按前缀处理数据
    variables_slice = defaultdict(list)
    
    # 根据 sep 参数分割列名并分组，处理每个前缀对应的列名列表
    if sep is None:
        # 如果 sep 为 None，则将所有列名放入空前缀的列表中
        variables_slice[""] = list(data.columns)
    elif isinstance(sep, str):
        # 如果 sep 是字符串，则按照 sep 分割每列名，将同一前缀的列名放入对应的列表中
        for col in data_to_decode.columns:
            prefix = col.split(sep)[0]
            if len(prefix) == len(col):
                raise ValueError(f"Separator not specified for column: {col}")
            variables_slice[prefix].append(col)
    else:
        # 如果 sep 不是字符串或 None，则抛出类型错误异常
        raise TypeError(
            "Expected 'sep' to be of type 'str' or 'None'; "
            f"Received 'sep' of type: {type(sep).__name__}"
        )
    # 如果传入了默认分类信息
    if default_category is not None:
        # 如果默认分类信息是字典类型
        if isinstance(default_category, dict):
            # 检查默认分类信息的长度是否与变量切片的长度相同
            if not len(default_category) == len(variables_slice):
                # 构造长度不匹配的错误消息
                len_msg = (
                    f"Length of 'default_category' ({len(default_category)}) "
                    f"did not match the length of the columns being encoded "
                    f"({len(variables_slice)})"
                )
                # 抛出值错误异常
                raise ValueError(len_msg)
        # 如果默认分类信息是可散列类型
        elif isinstance(default_category, Hashable):
            # 使用默认分类信息构建字典，每个变量切片映射到同一默认分类值
            default_category = dict(
                zip(variables_slice, [default_category] * len(variables_slice))
            )
        # 如果默认分类信息既不是字典也不是可散列类型，抛出类型错误异常
        else:
            raise TypeError(
                "Expected 'default_category' to be of type "
                "'None', 'Hashable', or 'dict'; "
                "Received 'default_category' of type: "
                f"{type(default_category).__name__}"
            )

    # 初始化一个空字典用于存储分类数据
    cat_data = {}
    # 遍历变量切片字典中的每个前缀和其对应的切片
    for prefix, prefix_slice in variables_slice.items():
        # 如果分隔符为None，直接复制前缀切片作为分类
        if sep is None:
            cats = prefix_slice.copy()
        # 如果有分隔符，去除前缀和分隔符后的列名作为分类
        else:
            cats = [col[len(prefix + sep) :] for col in prefix_slice]
        
        # 计算每行中被分配值的总和
        assigned = data_to_decode.loc[:, prefix_slice].sum(axis=1)
        
        # 如果有任何行被分配了多个值，抛出值错误异常
        if any(assigned > 1):
            raise ValueError(
                "Dummy DataFrame contains multi-assignment(s); "
                f"First instance in row: {assigned.idxmax()}"
            )
        
        # 如果有任何行未被分配值，根据默认分类信息补充分类
        if any(assigned == 0):
            if isinstance(default_category, dict):
                cats.append(default_category[prefix])
            else:
                raise ValueError(
                    "Dummy DataFrame contains unassigned value(s); "
                    f"First instance in row: {assigned.idxmin()}"
                )
            # 将未分配值的列信息添加到数据切片中
            data_slice = concat(
                (data_to_decode.loc[:, prefix_slice], assigned == 0), axis=1
            )
        else:
            # 否则，只保留原始数据切片
            data_slice = data_to_decode.loc[:, prefix_slice]
        
        # 根据分类列表创建Series对象
        cats_array = data._constructor_sliced(cats, dtype=data.columns.dtype)
        
        # 获取每行中第一个True值的列索引
        true_values = data_slice.idxmax(axis=1)
        indexer = data_slice.columns.get_indexer_for(true_values)
        
        # 将分类数据映射到数据索引上，并赋值给分类数据字典中的前缀键
        cat_data[prefix] = cats_array.take(indexer).set_axis(data.index)

    # 使用分类数据字典创建DataFrame对象
    result = DataFrame(cat_data)
    
    # 如果指定了分隔符，将结果DataFrame的列类型转换为原始数据列的类型
    if sep is not None:
        result.columns = result.columns.astype(data.columns.dtype)
    
    # 返回最终结果DataFrame
    return result
```