# `D:\src\scipysrc\pandas\pandas\core\arrays\sparse\accessor.py`

```
# 导入必要的库和模块
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas.compat._optional import import_optional_dependency

from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.dtypes import SparseDtype

from pandas.core.accessor import (
    PandasDelegate,
    delegate_names,
)
from pandas.core.arrays.sparse.array import SparseArray

if TYPE_CHECKING:
    from scipy.sparse import (
        coo_matrix,
        spmatrix,
    )

    from pandas import (
        DataFrame,
        Series,
    )


# 定义基础访问器类
class BaseAccessor:
    _validation_msg = "Can only use the '.sparse' accessor with Sparse data."

    def __init__(self, data=None) -> None:
        # 初始化时保存数据的引用
        self._parent = data
        # 执行数据验证
        self._validate(data)

    # 数据验证方法，子类需要实现
    def _validate(self, data) -> None:
        raise NotImplementedError


# 使用装饰器将指定的属性委托给SparseArray类
@delegate_names(
    SparseArray, ["npoints", "density", "fill_value", "sp_values"], typ="property"
)
# 稀疏访问器类，继承自基础访问器类和PandasDelegate
class SparseAccessor(BaseAccessor, PandasDelegate):
    """
    Accessor for SparseSparse from other sparse matrix data types.

    Examples
    --------
    >>> ser = pd.Series([0, 0, 2, 2, 2], dtype="Sparse[int]")
    >>> ser.sparse.density
    0.6
    >>> ser.sparse.sp_values
    array([2, 2, 2])
    """

    # 数据验证方法重载，检查数据是否为稀疏类型
    def _validate(self, data) -> None:
        if not isinstance(data.dtype, SparseDtype):
            raise AttributeError(self._validation_msg)

    # 委托属性获取方法，通过getattr获取父对象的SparseArray属性
    def _delegate_property_get(self, name: str, *args, **kwargs):
        return getattr(self._parent.array, name)

    # 委托方法调用方法，根据方法名调用相应的方法或抛出异常
    def _delegate_method(self, name: str, *args, **kwargs):
        if name == "from_coo":
            return self.from_coo(*args, **kwargs)
        elif name == "to_coo":
            return self.to_coo(*args, **kwargs)
        else:
            raise ValueError

    # 类方法，以下内容还未给出
    @classmethod
    def from_coo(cls, A, dense_index: bool = False) -> Series:
        """
        Create a Series with sparse values from a scipy.sparse.coo_matrix.

        Parameters
        ----------
        A : scipy.sparse.coo_matrix
            The input sparse matrix in COO (Coordinate format).
        dense_index : bool, default False
            If False (default), the index consists of only the
            coords of the non-null entries of the original coo_matrix.
            If True, the index consists of the full sorted
            (row, col) coordinates of the coo_matrix.

        Returns
        -------
        s : Series
            A Series with sparse values.

        Examples
        --------
        >>> from scipy import sparse

        >>> A = sparse.coo_matrix(
        ...     ([3.0, 1.0, 2.0], ([1, 0, 0], [0, 2, 3])), shape=(3, 4)
        ... )
        >>> A
        <COOrdinate sparse matrix of dtype 'float64'
            with 3 stored elements and shape (3, 4)>

        >>> A.todense()
        matrix([[0., 0., 1., 2.],
                [3., 0., 0., 0.],
                [0., 0., 0., 0.]])

        >>> ss = pd.Series.sparse.from_coo(A)
        >>> ss
        0  2    1.0
           3    2.0
        1  0    3.0
        dtype: Sparse[float64, nan]
        """
        # 导入 pandas 的 Series 类和 coo_to_sparse_series 函数
        from pandas import Series
        from pandas.core.arrays.sparse.scipy_sparse import coo_to_sparse_series

        # 调用 pandas 提供的函数 coo_to_sparse_series，将 COO 稀疏矩阵转换为稀疏 Series
        result = coo_to_sparse_series(A, dense_index=dense_index)
        # 将结果转换为 pandas 的 Series 对象，保持引用关系而不复制数据
        result = Series(result.array, index=result.index, copy=False)

        # 返回创建的稀疏 Series 对象
        return result
    ) -> tuple[coo_matrix, list, list]:
        """
        Create a scipy.sparse.coo_matrix from a Series with MultiIndex.

        Use row_levels and column_levels to determine the row and column
        coordinates respectively. row_levels and column_levels are the names
        (labels) or numbers of the levels. {row_levels, column_levels} must be
        a partition of the MultiIndex level names (or numbers).

        Parameters
        ----------
        row_levels : tuple/list
            Names or numbers of the levels from which to determine row coordinates.
        column_levels : tuple/list
            Names or numbers of the levels from which to determine column coordinates.
        sort_labels : bool, default False
            Sort the row and column labels before forming the sparse matrix.
            When `row_levels` and/or `column_levels` refer to a single level,
            set to `True` for faster execution.

        Returns
        -------
        y : scipy.sparse.coo_matrix
            The resulting sparse matrix.
        rows : list
            List of row labels.
        columns : list
            List of column labels.

        Examples
        --------
        >>> s = pd.Series([3.0, np.nan, 1.0, 3.0, np.nan, np.nan])
        >>> s.index = pd.MultiIndex.from_tuples(
        ...     [
        ...         (1, 2, "a", 0),
        ...         (1, 2, "a", 1),
        ...         (1, 1, "b", 0),
        ...         (1, 1, "b", 1),
        ...         (2, 1, "b", 0),
        ...         (2, 1, "b", 1),
        ...     ],
        ...     names=["A", "B", "C", "D"],
        ... )

        >>> ss = s.astype("Sparse")

        >>> A, rows, columns = ss.sparse.to_coo(
        ...     row_levels=["A", "B"], column_levels=["C", "D"], sort_labels=True
        ... )
        >>> A
        <COOrdinate sparse matrix of dtype 'float64'
            with 3 stored elements and shape (3, 4)>
        >>> A.todense()
        matrix([[0., 0., 1., 3.],
                [3., 0., 0., 0.],
                [0., 0., 0., 0.]])

        >>> rows
        [(1, 1), (1, 2), (2, 1)]
        >>> columns
        [('a', 0), ('a', 1), ('b', 0), ('b', 1)]
        """
        from pandas.core.arrays.sparse.scipy_sparse import sparse_series_to_coo

        # Call a function to convert a sparse Pandas Series to a COO sparse matrix
        A, rows, columns = sparse_series_to_coo(
            self._parent, row_levels, column_levels, sort_labels=sort_labels
        )
        # Return the COO matrix and its associated row and column labels
        return A, rows, columns
    def to_dense(self) -> Series:
        """
        Convert a Series from sparse values to dense.

        Returns
        -------
        Series:
            A Series with the same values, stored as a dense array.

        Examples
        --------
        >>> series = pd.Series(pd.arrays.SparseArray([0, 1, 0]))
        >>> series
        0    0
        1    1
        2    0
        dtype: Sparse[int64, 0]

        >>> series.sparse.to_dense()
        0    0
        1    1
        2    0
        dtype: int64
        """
        
        from pandas import Series  # 导入 Series 类

        # 使用父类对象的稀疏数组（SparseArray）的 to_dense 方法，生成一个密集的 Series 对象
        return Series(
            self._parent.array.to_dense(),  # 将稀疏的数组转换为密集的数组
            index=self._parent.index,  # 设置返回 Series 的索引与父类的索引相同
            name=self._parent.name,    # 设置返回 Series 的名称与父类的名称相同
            copy=False,  # 指定返回的 Series 不需要复制数据，直接引用
        )
    """
    DataFrame accessor for sparse data.

    Parameters
    ----------
    data : scipy.sparse.spmatrix
        Must be convertible to csc format.

    See Also
    --------
    DataFrame.sparse.density : Ratio of non-sparse points to total (dense) data points.

    Examples
    --------
    >>> df = pd.DataFrame({"a": [1, 2, 0, 0], "b": [3, 0, 0, 4]}, dtype="Sparse[int]")
    >>> df.sparse.density
    0.5
    """
    
    def _validate(self, data) -> None:
        """
        Validate the data for sparse DataFrame creation.

        Parameters
        ----------
        data : scipy.sparse.spmatrix
            The sparse matrix to validate.

        Raises
        ------
        AttributeError
            If any column in the sparse matrix is not of SparseDtype.

        Notes
        -----
        This method ensures that all columns of the sparse matrix are of SparseDtype.
        """
        dtypes = data.dtypes
        if not all(isinstance(t, SparseDtype) for t in dtypes):
            raise AttributeError(self._validation_msg)

    @classmethod
    def from_spmatrix(cls, data, index=None, columns=None) -> DataFrame:
        """
        Create a new DataFrame from a scipy sparse matrix.

        Parameters
        ----------
        data : scipy.sparse.spmatrix
            The input sparse matrix, must be convertible to csc format.
        index, columns : Index, optional
            Row and column labels to use for the resulting DataFrame.
            Defaults to a RangeIndex if not provided.

        Returns
        -------
        DataFrame
            The resulting DataFrame with each column stored as a SparseArray.

        See Also
        --------
        DataFrame.sparse.to_coo : Return the contents of the frame as a sparse SciPy COO matrix.

        Examples
        --------
        >>> import scipy.sparse
        >>> mat = scipy.sparse.eye(3, dtype=int)
        >>> pd.DataFrame.sparse.from_spmatrix(mat)
             0    1    2
        0    1    0    0
        1    0    1    0
        2    0    0    1
        """
        from pandas._libs.sparse import IntIndex  # Importing required class from pandas sparse module
        from pandas import DataFrame  # Importing DataFrame class from pandas

        data = data.tocsc()  # Convert the sparse matrix to CSC (compressed sparse column) format
        index, columns = cls._prep_index(data, index, columns)  # Prepare index and columns for the new DataFrame
        n_rows, n_columns = data.shape  # Get the shape (dimensions) of the sparse matrix
        # Ensure that indices are sorted; this is important for creating IntIndex without validation
        # This step has minimal overhead if indices are already sorted in scipy
        data.sort_indices()
        indices = data.indices  # Get the indices array of the sparse matrix
        indptr = data.indptr  # Get the indptr array of the sparse matrix
        array_data = data.data  # Get the data array of the sparse matrix
        dtype = SparseDtype(array_data.dtype)  # Define the dtype for the SparseArray based on data dtype
        arrays = []
        for i in range(n_columns):
            sl = slice(indptr[i], indptr[i + 1])  # Create a slice for each column's data in the sparse matrix
            idx = IntIndex(n_rows, indices[sl], check_integrity=False)  # Create IntIndex for column's indices
            arr = SparseArray._simple_new(array_data[sl], idx, dtype)  # Create SparseArray for the column
            arrays.append(arr)  # Append SparseArray to arrays list
        # Create DataFrame from arrays with provided or default columns and index,
        # with integrity check disabled for performance reasons
        return DataFrame._from_arrays(
            arrays, columns=columns, index=index, verify_integrity=False
        )
    # 将稀疏值的 DataFrame 转换为密集形式的 DataFrame。

    # 导入 DataFrame 类
    from pandas import DataFrame

    # 从父 DataFrame 的每个项中获取稀疏数组，并将其转换为密集数组
    data = {k: v.array.to_dense() for k, v in self._parent.items()}

    # 返回一个新的 DataFrame，其值以密集数组形式存储，并保留与父 DataFrame 相同的索引和列
    return DataFrame(data, index=self._parent.index, columns=self._parent.columns)

    def to_coo(self) -> spmatrix:
        """
        Return the contents of the frame as a sparse SciPy COO matrix.

        Returns
        -------
        scipy.sparse.spmatrix
            If the caller is heterogeneous and contains booleans or objects,
            the result will be of dtype=object. See Notes.

        See Also
        --------
        DataFrame.sparse.to_dense : Convert a DataFrame with sparse values to dense.

        Notes
        -----
        The dtype will be the lowest-common-denominator type (implicit
        upcasting); that is to say if the dtypes (even of numeric types)
        are mixed, the one that accommodates all will be chosen.

        e.g. If the dtypes are float16 and float32, dtype will be upcast to
        float32. By numpy.find_common_type convention, mixing int64 and
        and uint64 will result in a float64 dtype.

        Examples
        --------
        >>> df = pd.DataFrame({"A": pd.arrays.SparseArray([0, 1, 0, 1])})
        >>> df.sparse.to_coo()
        <COOrdinate sparse matrix of dtype 'int64'
            with 2 stored elements and shape (4, 1)>
        """
        # 导入必要的依赖项
        import_optional_dependency("scipy")
        from scipy.sparse import coo_matrix

        # 找到最适合的数据类型，用于创建 COO 稀疏矩阵
        dtype = find_common_type(self._parent.dtypes.to_list())
        if isinstance(dtype, SparseDtype):
            dtype = dtype.subtype

        # 初始化存储稀疏矩阵所需的列表
        cols, rows, data = [], [], []

        # 遍历父 DataFrame 的每一列及其稀疏数组
        for col, (_, ser) in enumerate(self._parent.items()):
            sp_arr = ser.array

            # 获取稀疏数组的非零元素的行索引
            row = sp_arr.sp_index.indices

            # 将列索引重复，以匹配稀疏数组的长度
            cols.append(np.repeat(col, len(row)))

            # 将稀疏数组的行索引添加到行索引列表中
            rows.append(row)

            # 获取稀疏数组的非零值，并将其转换为指定数据类型
            data.append(sp_arr.sp_values.astype(dtype, copy=False))

        # 合并列索引、行索引和数据，创建 COO 稀疏矩阵
        cols = np.concatenate(cols)
        rows = np.concatenate(rows)
        data = np.concatenate(data)
        return coo_matrix((data, (rows, cols)), shape=self._parent.shape)
    def density(self) -> float:
        """
        Ratio of non-sparse points to total (dense) data points.

        See Also
        --------
        DataFrame.sparse.from_spmatrix : Create a new DataFrame from a
            scipy sparse matrix.

        Examples
        --------
        >>> df = pd.DataFrame({"A": pd.arrays.SparseArray([0, 1, 0, 1])})
        >>> df.sparse.density
        0.5
        """
        # 计算稠密数据的比率，即非稀疏点占总数据点的比例
        tmp = np.mean([column.array.density for _, column in self._parent.items()])
        # 返回稠密度比率
        return tmp

    @staticmethod
    def _prep_index(data, index, columns):
        from pandas.core.indexes.api import (
            default_index,
            ensure_index,
        )

        # 获取数据矩阵的行数 N 和列数 K
        N, K = data.shape
        # 如果 index 为 None，则使用默认的索引
        if index is None:
            index = default_index(N)
        else:
            # 确保 index 是一个有效的索引对象
            index = ensure_index(index)
        # 如果 columns 为 None，则使用默认的索引
        if columns is None:
            columns = default_index(K)
        else:
            # 确保 columns 是一个有效的索引对象
            columns = ensure_index(columns)

        # 检查 columns 的长度是否与 K 相符
        if len(columns) != K:
            raise ValueError(f"Column length mismatch: {len(columns)} vs. {K}")
        # 检查 index 的长度是否与 N 相符
        if len(index) != N:
            raise ValueError(f"Index length mismatch: {len(index)} vs. {N}")
        # 返回处理后的 index 和 columns
        return index, columns
```