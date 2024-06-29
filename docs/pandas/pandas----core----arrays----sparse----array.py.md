# `D:\src\scipysrc\pandas\pandas\core\arrays\sparse\array.py`

```
# ----------------------------------------------------------------------------
# Array

# 定义用于 SparseArray 的文档字符串关键字参数
_sparray_doc_kwargs = {"klass": "SparseArray"}

# 定义一个函数 _get_fill，用于从 SparseArray 中获取填充值并返回一个零维 ndarray
def _get_fill(arr: SparseArray) -> np.ndarray:
    """
    Create a 0-dim ndarray containing the fill value

    Parameters
    ----------
    arr : SparseArray
        输入的 SparseArray 对象

    Returns
    -------
    fill_value : ndarray
        包含填充值的零维 ndarray.

    Notes
    -----
    """
    # 尝试将 fill_value 转换为 arr 的数据类型（dtype），如果可能的话
    try:
        # 使用 arr 的数据类型的子类型将 fill_value 转换为 NumPy 数组
        return np.asarray(arr.fill_value, dtype=arr.dtype.subtype)
    # 如果转换过程中出现值错误（ValueError），则执行以下操作
    except ValueError:
        # 将 fill_value 转换为 NumPy 数组，无需指定数据类型
        return np.asarray(arr.fill_value)
def _sparse_array_op(
    left: SparseArray, right: SparseArray, op: Callable, name: str
) -> SparseArray:
    """
    Perform a binary operation between two arrays.

    Parameters
    ----------
    left : Union[SparseArray, ndarray]
        The left operand of the binary operation, either SparseArray or ndarray.
    right : Union[SparseArray, ndarray]
        The right operand of the binary operation, either SparseArray or ndarray.
    op : Callable
        The binary operation to perform.
    name str
        Name of the operation.

    Returns
    -------
    SparseArray
        Result of the binary operation as a SparseArray.
    """
    if name.startswith("__"):
        # Strip leading and trailing underscores for library lookup
        name = name[2:-2]

    # Determine subtype of the operands for consistency
    ltype = left.dtype.subtype
    rtype = right.dtype.subtype

    if ltype != rtype:
        # Find common type for the operands
        subtype = find_common_type([ltype, rtype])
        # Create SparseDtype objects with common subtype and fill values
        ltype = SparseDtype(subtype, left.fill_value)
        rtype = SparseDtype(subtype, right.fill_value)
        # Convert operands to the common SparseDtype
        left = left.astype(ltype, copy=False)
        right = right.astype(rtype, copy=False)
        dtype = ltype.subtype
    else:
        dtype = ltype

    # Initialize result_dtype to None
    result_dtype = None

    if left.sp_index.ngaps == 0 or right.sp_index.ngaps == 0:
        # If either operand is fully dense
        with np.errstate(all="ignore"):
            # Perform operation on dense representations
            result = op(left.to_dense(), right.to_dense())
            # Compute fill value operation
            fill = op(_get_fill(left), _get_fill(right))

        # Select index based on which operand is sparse
        if left.sp_index.ngaps == 0:
            index = left.sp_index
        else:
            index = right.sp_index
    elif left.sp_index.equals(right.sp_index):
        # If both operands have identical sparse indices
        with np.errstate(all="ignore"):
            # Perform operation directly on sparse values
            result = op(left.sp_values, right.sp_values)
            # Compute fill value operation
            fill = op(_get_fill(left), _get_fill(right))
        index = left.sp_index
    else:
        # Operate on different sparse indices
        if name[0] == "r":
            # Handle reverse operation if name starts with 'r'
            left, right = right, left
            name = name[1:]

        if name in ("and", "or", "xor") and dtype == "bool":
            # Special case for boolean operations
            opname = f"sparse_{name}_uint8"
            # Cast sparse values to uint8 for simplicity
            left_sp_values = left.sp_values.view(np.uint8)
            right_sp_values = right.sp_values.view(np.uint8)
            result_dtype = bool
        else:
            # General case for other operations
            opname = f"sparse_{name}_{dtype}"
            left_sp_values = left.sp_values
            right_sp_values = right.sp_values

        if (
            name in ["floordiv", "mod"]
            and (right == 0).any()
            and left.dtype.kind in "iu"
        ):
            # Adjust for division and modulus by zero behavior
            opname = f"sparse_{name}_float64"
            left_sp_values = left_sp_values.astype("float64")
            right_sp_values = right_sp_values.astype("float64")

        # Fetch the appropriate sparse operation function
        sparse_op = getattr(splib, opname)

        with np.errstate(all="ignore"):
            # Perform the sparse operation
            result, index, fill = sparse_op(
                left_sp_values,
                left.sp_index,
                left.fill_value,
                right_sp_values,
                right.sp_index,
                right.fill_value,
            )
    if name == "divmod":
        # 如果名字是 "divmod"，则返回一个包含两个元素的元组
        return (  # type: ignore[return-value]
            # 对结果的第一个元素进行包装处理
            _wrap_result(name, result[0], index, fill[0], dtype=result_dtype),
            # 对结果的第二个元素进行包装处理
            _wrap_result(name, result[1], index, fill[1], dtype=result_dtype),
        )

    if result_dtype is None:
        # 如果结果数据类型为 None，则将其设置为 result 的数据类型
        result_dtype = result.dtype

    # 返回对结果进行包装处理后的值
    return _wrap_result(name, result, index, fill, dtype=result_dtype)
# 定义一个函数 `_wrap_result`，用于将操作结果包装成正确的数据类型 SparseArray
def _wrap_result(
    name: str, data, sparse_index, fill_value, dtype: Dtype | None = None
) -> SparseArray:
    """
    wrap op result to have correct dtype
    """
    # 如果操作名以双下划线开始和结束，如 __eq__，则去掉双下划线
    if name.startswith("__"):
        # e.g. __eq__ --> eq
        name = name[2:-2]

    # 如果操作名在以下列表中，设置数据类型为布尔值
    if name in ("eq", "ne", "lt", "gt", "le", "ge"):
        dtype = bool

    # 将 fill_value 转换成标量
    fill_value = lib.item_from_zerodim(fill_value)

    # 如果数据类型是布尔类型，确保 fill_value 是布尔值
    if is_bool_dtype(dtype):
        # fill_value 可能是 np.bool_
        fill_value = bool(fill_value)
    
    # 返回 SparseArray 对象，传入数据、稀疏索引、填充值和数据类型
    return SparseArray(
        data, sparse_index=sparse_index, fill_value=fill_value, dtype=dtype
    )


class SparseArray(OpsMixin, PandasObject, ExtensionArray):
    """
    An ExtensionArray for storing sparse data.

    Parameters
    ----------
    data : array-like or scalar
        A dense array of values to store in the SparseArray. This may contain
        `fill_value`.
    sparse_index : SparseIndex, optional
    fill_value : scalar, optional
        Elements in data that are ``fill_value`` are not stored in the
        SparseArray. For memory savings, this should be the most common value
        in `data`. By default, `fill_value` depends on the dtype of `data`:

        =========== ==========
        data.dtype  na_value
        =========== ==========
        float       ``np.nan``
        int         ``0``
        bool        False
        datetime64  ``pd.NaT``
        timedelta64 ``pd.NaT``
        =========== ==========

        The fill value is potentially specified in three ways. In order of
        precedence, these are

        1. The `fill_value` argument
        2. ``dtype.fill_value`` if `fill_value` is None and `dtype` is
           a ``SparseDtype``
        3. ``data.dtype.fill_value`` if `fill_value` is None and `dtype`
           is not a ``SparseDtype`` and `data` is a ``SparseArray``.

    kind : str
        Can be 'integer' or 'block', default is 'integer'.
        The type of storage for sparse locations.

        * 'block': Stores a `block` and `block_length` for each
          contiguous *span* of sparse values. This is best when
          sparse data tends to be clumped together, with large
          regions of ``fill-value`` values between sparse values.
        * 'integer': uses an integer to store the location of
          each sparse value.

    dtype : np.dtype or SparseDtype, optional
        The dtype to use for the SparseArray. For numpy dtypes, this
        determines the dtype of ``self.sp_values``. For SparseDtype,
        this determines ``self.sp_values`` and ``self.fill_value``.
    copy : bool, default False
        Whether to explicitly copy the incoming `data` array.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> from pandas.arrays import SparseArray
    >>> arr = SparseArray([0, 0, 1, 2])
    >>> arr
    [0, 0, 1, 2]
    Fill: 0
    IntIndex
    Indices: array([2, 3], dtype=int32)
    """

    _subtyp = "sparse_array"  # register ABCSparseArray
    # 将 PandasObject 的隐藏属性与空集合合并，生成一个新的隐藏属性集合
    _hidden_attrs = PandasObject._hidden_attrs | frozenset([])

    # 声明一个稀疏索引对象，类型为 SparseIndex
    _sparse_index: SparseIndex

    # 声明一个 NumPy 数组，用于存储稀疏数据值
    _sparse_values: np.ndarray

    # 声明一个稀疏数据类型对象，类型为 SparseDtype
    _dtype: SparseDtype

    # 构造函数，初始化 SparseArray 对象
    def __init__(
        self,
        data,  # 接收数据作为参数
        sparse_index=None,  # 稀疏索引，默认为 None
        fill_value=None,  # 填充值，默认为 None
        kind: SparseIndexKind = "integer",  # 稀疏索引类型，默认为 "integer"
        dtype: Dtype | None = None,  # 数据类型，默认为 None
        copy: bool = False,  # 是否复制数据，默认为 False
    ):

    @classmethod
    # 类方法，用于创建一个新的 SparseArray 对象
    def _simple_new(
        cls,
        sparse_array: np.ndarray,  # 稀疏数组作为参数
        sparse_index: SparseIndex,  # 稀疏索引对象作为参数
        dtype: SparseDtype,  # 稀疏数据类型对象作为参数
    ) -> Self:  # 返回类型为当前类的实例
        # 创建一个未初始化的对象
        new = object.__new__(cls)
        # 初始化对象的稀疏索引属性
        new._sparse_index = sparse_index
        # 初始化对象的稀疏数据值属性
        new._sparse_values = sparse_array
        # 初始化对象的稀疏数据类型属性
        new._dtype = dtype
        # 返回新创建的对象
        return new

    @classmethod
    # 类方法，从 scipy.sparse 矩阵创建 SparseArray 对象
    def from_spmatrix(cls, data: spmatrix) -> Self:
        """
        从 scipy.sparse 矩阵创建 SparseArray 对象。

        Parameters
        ----------
        data : scipy.sparse.sp_matrix
            应为一个 SciPy 稀疏矩阵，第二维度的大小应为 1。换句话说，是一个只有一列的稀疏矩阵。

        Returns
        -------
        SparseArray

        Examples
        --------
        >>> import scipy.sparse
        >>> mat = scipy.sparse.coo_matrix((4, 1))
        >>> pd.arrays.SparseArray.from_spmatrix(mat)
        [0.0, 0.0, 0.0, 0.0]
        Fill: 0.0
        IntIndex
        Indices: array([], dtype=int32)
        """
        # 获取数据矩阵的行数和列数
        length, ncol = data.shape

        # 如果列数不等于 1，则抛出 ValueError 异常
        if ncol != 1:
            raise ValueError(f"'data' must have a single column, not '{ncol}'")

        # 将输入的稀疏矩阵转换为 CSC 格式，并排序索引
        data = data.tocsc()
        data.sort_indices()
        # 获取排序后的数据值数组和索引数组
        arr = data.data
        idx = data.indices

        # 创建零值数组，用于创建 SparseDtype 对象
        zero = np.array(0, dtype=arr.dtype).item()
        dtype = SparseDtype(arr.dtype, zero)

        # 创建整数索引对象
        index = IntIndex(length, idx)

        # 使用 _simple_new 方法创建并返回 SparseArray 对象
        return cls._simple_new(arr, index, dtype)

    # 数组转换方法，用于将 SparseArray 转换为数组
    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        fill_value = self.fill_value  # 获取当前对象的填充值

        if self.sp_index.ngaps == 0:
            # 如果没有间隙，返回存储的稀疏值
            return self.sp_values
        if dtype is None:
            # 判断 NumPy 能否表示这种数据类型
            # 如果不能，`np.result_type` 将会引发异常。我们捕获异常并返回对象类型。
            if self.sp_values.dtype.kind == "M":
                # 然而，我们特别处理了常见的情况，即带有 pandas NaT 的 datetime64。
                if fill_value is NaT:
                    # 无法将 pd.NaT 放入 datetime64[ns]
                    fill_value = np.datetime64("NaT")
            try:
                dtype = np.result_type(self.sp_values.dtype, type(fill_value))
            except TypeError:
                dtype = object

        out = np.full(self.shape, fill_value, dtype=dtype)  # 使用指定的填充值和数据类型创建全新的数组
        out[self.sp_index.indices] = self.sp_values  # 将稀疏值插入到指定索引位置
        return out  # 返回最终生成的数组

    def __setitem__(self, key, value) -> None:
        # 我认为我们可以允许设置非填充值元素。
        # TODO(SparseArray.__setitem__): 在 ExtensionBlock.where 中移除特殊情况
        msg = "SparseArray does not support item assignment via setitem"
        raise TypeError(msg)

    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype: Dtype | None = None, copy: bool = False
    ) -> Self:
        return cls(scalars, dtype=dtype)  # 从序列创建 SparseArray 实例

    @classmethod
    def _from_factorized(cls, values, original) -> Self:
        return cls(values, dtype=original.dtype)  # 从因子化数据创建 SparseArray 实例

    # ------------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------------
    @property
    def sp_index(self) -> SparseIndex:
        """
        包含非 `fill_value` 点位置的 SparseIndex。
        """
        return self._sparse_index  # 返回存储在 `_sparse_index` 中的 SparseIndex 对象

    @property
    def sp_values(self) -> np.ndarray:
        """
        包含非 `fill_value` 值的 ndarray。

        Examples
        --------
        >>> from pandas.arrays import SparseArray
        >>> s = SparseArray([0, 0, 1, 0, 2], fill_value=0)
        >>> s.sp_values
        array([1, 2])
        """
        return self._sparse_values  # 返回存储在 `_sparse_values` 中的 ndarray

    @property
    def dtype(self) -> SparseDtype:
        return self._dtype  # 返回 SparseArray 的数据类型

    @property
    def fill_value(self):
        """
        `data` 中的 `fill_value` 元素不会被存储。

        为了节省内存，应选择数组中最常见的值作为填充值。

        Examples
        --------
        >>> ser = pd.Series([0, 0, 2, 2, 2], dtype="Sparse[int]")
        >>> ser.sparse.fill_value
        0
        >>> spa_dtype = pd.SparseDtype(dtype=np.int32, fill_value=2)
        >>> ser = pd.Series([0, 0, 2, 2, 2], dtype=spa_dtype)
        >>> ser.sparse.fill_value
        2
        """
        return self.dtype.fill_value  # 返回 SparseArray 的填充值

    @fill_value.setter
    def fill_value(self, value) -> None:
        # 设置稀疏数组的数据类型为指定的值类型
        self._dtype = SparseDtype(self.dtype.subtype, value)

    @property
    def kind(self) -> SparseIndexKind:
        """
        返回稀疏索引类型，可能是 {'integer', 'block'} 中的一个。
        """
        if isinstance(self.sp_index, IntIndex):
            return "integer"
        else:
            return "block"

    @property
    def _valid_sp_values(self) -> np.ndarray:
        # 获取非空值的稀疏数组数据
        sp_vals = self.sp_values
        mask = notna(sp_vals)
        return sp_vals[mask]

    def __len__(self) -> int:
        # 返回稀疏数组索引的长度
        return self.sp_index.length

    @property
    def _null_fill_value(self) -> bool:
        # 返回是否为 null 填充值
        return self._dtype._is_na_fill_value

    @property
    def nbytes(self) -> int:
        # 返回稀疏数组值和索引的字节数总和
        return self.sp_values.nbytes + self.sp_index.nbytes

    @property
    def density(self) -> float:
        """
        返回非填充值占总数的百分比，以小数表示。
        
        示例
        --------
        >>> from pandas.arrays import SparseArray
        >>> s = SparseArray([0, 0, 1, 1, 1], fill_value=0)
        >>> s.density
        0.6
        """
        return self.sp_index.npoints / self.sp_index.length

    @property
    def npoints(self) -> int:
        """
        返回非填充值的数量。
        
        示例
        --------
        >>> from pandas.arrays import SparseArray
        >>> s = SparseArray([0, 0, 1, 1, 1], fill_value=0)
        >>> s.npoints
        3
        """
        return self.sp_index.npoints

    # error: Return type "SparseArray" of "isna" incompatible with return type
    # "ndarray[Any, Any] | ExtensionArraySupportsAnyAll" in supertype "ExtensionArray"
    def isna(self) -> Self:  # type: ignore[override]
        # 如果使用空值填充，则返回 SparseDtype[bool, true] 以保持相同的内存使用。
        dtype = SparseDtype(bool, self._null_fill_value)
        if self._null_fill_value:
            return type(self)._simple_new(isna(self.sp_values), self.sp_index, dtype)
        mask = np.full(len(self), False, dtype=np.bool_)
        mask[self.sp_index.indices] = isna(self.sp_values)
        return type(self)(mask, fill_value=False, dtype=dtype)

    def fillna(
        self,
        value,
        limit: int | None = None,
        copy: bool = True,
        backfill: bool = False,
    ) -> Self:
        """
        Fill missing values with `value`.

        Parameters
        ----------
        value : scalar
            填充缺失值的值
        limit : int, optional
            不支持稀疏数组，必须为 None。
        copy: bool, default True
            对于稀疏数组而言无效。
        
        Returns
        -------
        SparseArray
            填充后的稀疏数组
        
        Notes
        -----
        当指定了 `value` 时，结果的 ``fill_value`` 取决于 ``self.fill_value``。目标是保持低内存使用。

        如果 ``self.fill_value`` 是 NA，则结果的 dtype 将为 ``SparseDtype(self.dtype, fill_value=value)``。
        这将在填充前后保持使用的内存量。

        当 ``self.fill_value`` 不是 NA 时，结果的 dtype 将为 ``self.dtype``。同样，这保持了使用的内存量。
        """
        if limit is not None:
            raise ValueError("limit must be None")
        new_values = np.where(isna(self.sp_values), value, self.sp_values)

        if self._null_fill_value:
            # This is essentially just updating the dtype.
            new_dtype = SparseDtype(self.dtype.subtype, fill_value=value)
        else:
            new_dtype = self.dtype

        return self._simple_new(new_values, self._sparse_index, new_dtype)

    def shift(self, periods: int = 1, fill_value=None) -> Self:
        if not len(self) or periods == 0:
            return self.copy()

        if isna(fill_value):
            fill_value = self.dtype.na_value

        subtype = np.result_type(fill_value, self.dtype.subtype)

        if subtype != self.dtype.subtype:
            # just coerce up front
            arr = self.astype(SparseDtype(subtype, self.fill_value))
        else:
            arr = self

        empty = self._from_sequence(
            [fill_value] * min(abs(periods), len(self)), dtype=arr.dtype
        )

        if periods > 0:
            a = empty
            b = arr[:-periods]
        else:
            a = arr[abs(periods) :]
            b = empty
        return arr._concat_same_type([a, b])

    def _first_fill_value_loc(self):
        """
        Get the location of the first fill value.

        Returns
        -------
        int
            第一个填充值的位置
        """
        if len(self) == 0 or self.sp_index.npoints == len(self):
            return -1

        indices = self.sp_index.indices
        if not len(indices) or indices[0] > 0:
            return 0

        # a number larger than 1 should be appended to
        # the last in case of fill value only appears
        # in the tail of array
        diff = np.r_[np.diff(indices), 2]
        return indices[(diff > 1).argmax()] + 1

    @doc(ExtensionArray.duplicated)
    def duplicated(
        self, keep: Literal["first", "last", False] = "first"
    ) -> npt.NDArray[np.bool_]:
        values = np.asarray(self)
        mask = np.asarray(self.isna())
        return algos.duplicated(values, keep=keep, mask=mask)
    # 返回一个包含唯一值的 SparseArray 对象
    def unique(self) -> Self:
        # 使用算法模块中的 unique 函数找到 self.sp_values 中的唯一值
        uniques = algos.unique(self.sp_values)
        # 如果 self.sp_values 的长度不等于 self 的长度
        if len(self.sp_values) != len(self):
            # 找到第一个填充值的位置
            fill_loc = self._first_fill_value_loc()
            # 为了保持与 pd.unique 或 pd.Series.unique 相似的行为，
            # 我们应该保持原始顺序，这里再次使用 unique 函数找到插入位置。
            # 由于 sp_values 的长度不大，可能会稍微影响性能，但为了正确性这是值得的。
            insert_loc = len(algos.unique(self.sp_values[:fill_loc]))
            # 在 uniques 中插入填充值，保持正确的顺序
            uniques = np.insert(uniques, insert_loc, self.fill_value)
        # 使用 type(self)._from_sequence 方法创建一个新的对象，包含唯一值
        return type(self)._from_sequence(uniques, dtype=self.dtype)

    # 返回一个包含用于因子化的值的 numpy 数组和填充值
    def _values_for_factorize(self):
        # 仍然覆盖此方法以支持 hash_pandas_object
        return np.asarray(self), self.fill_value

    # 因子化方法，返回因子化后的代码和稀疏数组
    def factorize(
        self,
        use_na_sentinel: bool = True,
    ) -> tuple[np.ndarray, SparseArray]:
        # ExtensionArray.factorize 当前返回的是 Tuple[ndarray, EA]
        # 这里的稀疏性与 Sparse 希望的相反。希望 ExtensionArray.factorize 返回 Tuple[EA, EA]
        # 考虑到我们必须返回一个密集的代码数组，为什么要实现一个高效的因子化呢？
        # 使用 algos.factorize 函数找到 self 中值的代码和唯一值
        codes, uniques = algos.factorize(
            np.asarray(self), use_na_sentinel=use_na_sentinel
        )
        # 创建一个 SparseArray 对象来表示唯一值，使用 self 的数据类型
        uniques_sp = SparseArray(uniques, dtype=self.dtype)
        # 返回代码数组和稀疏数组对象
        return codes, uniques_sp
    # 返回一个包含唯一值计数的 Series 对象

    @overload
    # 指定键为标量索引器时的重载方法
    def __getitem__(self, key: ScalarIndexer) -> Any: ...

    @overload
    # 指定键为序列索引器或元组时的重载方法
    def __getitem__(
        self,
        key: SequenceIndexer | tuple[int | ellipsis, ...],
    ) -> Self: ...

    # 获取索引处的值
    def __getitem__(
        self,
        key: PositionalIndexer | tuple[int | ellipsis, ...],
    ):
        loc = validate_insert_loc(loc, len(self))  # 验证插入位置是否有效

        sp_loc = self.sp_index.lookup(loc)  # 查找位置在稀疏索引中的实际位置
        if sp_loc == -1:
            return self.fill_value  # 如果位置无效，返回填充值
        else:
            val = self.sp_values[sp_loc]  # 获取对应位置的值
            val = maybe_box_datetimelike(val, self.sp_values.dtype)  # 将值转换为日期时间类型（如果可能）
            return val

    # 采用给定索引数组获取数据，可以选择是否填充缺失值
    def take(self, indices, *, allow_fill: bool = False, fill_value=None) -> Self:
        if is_scalar(indices):
            raise ValueError(f"'indices' must be an array, not a scalar '{indices}'.")
        indices = np.asarray(indices, dtype=np.int32)  # 将索引转换为 NumPy 数组，指定数据类型为 int32

        dtype = None
        if indices.size == 0:
            result = np.array([], dtype="object")  # 如果索引为空，则结果是一个空的对象数组
            dtype = self.dtype  # 设置结果的数据类型为对象数组的数据类型
        elif allow_fill:
            result = self._take_with_fill(indices, fill_value=fill_value)  # 使用填充值进行索引操作
        else:
            return self._take_without_fill(indices)  # 不使用填充值进行索引操作

        # 返回一个与当前对象类型相同的新对象，结果是根据索引操作后得到的数组
        return type(self)(
            result, fill_value=self.fill_value, kind=self.kind, dtype=dtype
        )
    # 使用 `_take_with_fill` 方法从数组中取出指定索引的元素，如果未提供填充值则使用数组的缺失值
    def _take_with_fill(self, indices, fill_value=None) -> np.ndarray:
        # 如果未提供填充值，则使用数组的缺失值
        if fill_value is None:
            fill_value = self.dtype.na_value

        # 检查索引数组中是否有小于 -1 的值，如果有则抛出 ValueError 异常
        if indices.min() < -1:
            raise ValueError(
                "Invalid value in 'indices'. Must be between -1 "
                "and the length of the array."
            )

        # 检查索引数组中是否有超出数组长度的值，如果有则抛出 IndexError 异常
        if indices.max() >= len(self):
            raise IndexError("out of bounds value in 'indices'.")

        # 如果数组为空，则根据条件允许取值为 -1 的情况，并返回填充后的数组
        if len(self) == 0:
            if (indices == -1).all():
                # 创建一个与索引数组相同形状的空数组，并填充为指定的填充值
                dtype = np.result_type(self.sp_values, type(fill_value))
                taken = np.empty_like(indices, dtype=dtype)
                taken.fill(fill_value)
                return taken
            else:
                # 如果数组不为空且试图从空轴上取值，则抛出 IndexError 异常
                raise IndexError("cannot do a non-empty take from an empty axes.")

        # 使用索引器 `self.sp_index` 查找对应索引的位置
        sp_indexer = self.sp_index.lookup_array(indices)
        new_fill_indices = indices == -1
        old_fill_indices = (sp_indexer == -1) & ~new_fill_indices

        # 如果稀疏数组 `self.sp_index` 中没有有效点，并且所有索引为旧的填充值，则填充为指定的填充值
        if self.sp_index.npoints == 0 and old_fill_indices.all():
            taken = np.full(
                sp_indexer.shape, fill_value=self.fill_value, dtype=self.dtype.subtype
            )

        elif self.sp_index.npoints == 0:
            # 使用旧的填充值，除非索引为 -1
            _dtype = np.result_type(self.dtype.subtype, type(fill_value))
            taken = np.full(sp_indexer.shape, fill_value=fill_value, dtype=_dtype)
            taken[old_fill_indices] = self.fill_value
        else:
            # 使用稀疏值数组 `self.sp_values` 根据索引数组 `sp_indexer` 取出对应元素
            taken = self.sp_values.take(sp_indexer)

            # 分两步填充：
            # 旧的填充值
            # 新的填充值
            # 每个阶段可能需要在每个阶段转换为新的数据类型。

            m0 = sp_indexer[old_fill_indices] < 0
            m1 = sp_indexer[new_fill_indices] < 0

            result_type = taken.dtype

            # 如果存在旧的填充值的情况，则将其转换为新的数据类型并填充为指定的填充值
            if m0.any():
                result_type = np.result_type(result_type, type(self.fill_value))
                taken = taken.astype(result_type)
                taken[old_fill_indices] = self.fill_value

            # 如果存在新的填充值的情况，则将其转换为新的数据类型并填充为指定的填充值
            if m1.any():
                result_type = np.result_type(result_type, type(fill_value))
                taken = taken.astype(result_type)
                taken[new_fill_indices] = fill_value

        return taken
    # 返回不包含填充值的新对象，使用给定的索引数组
    def _take_without_fill(self, indices) -> Self:
        # 判断哪些索引需要向后偏移
        to_shift = indices < 0
        
        # 获取当前对象的长度
        n = len(self)
        
        # 检查索引是否超出范围
        if (indices.max() >= n) or (indices.min() < -n):
            if n == 0:
                raise IndexError("cannot do a non-empty take from an empty axes.")
            raise IndexError("out of bounds value in 'indices'.")
        
        # 如果存在需要向后偏移的索引，进行处理
        if to_shift.any():
            indices = indices.copy()
            indices[to_shift] += n
        
        # 使用索引数组从稀疏矩阵的索引结构中查找对应位置
        sp_indexer = self.sp_index.lookup_array(indices)
        
        # 创建值掩码，标记找到的有效值位置
        value_mask = sp_indexer != -1
        
        # 从稀疏矩阵的值数组中获取新的稀疏矩阵值
        new_sp_values = self.sp_values[sp_indexer[value_mask]]
        
        # 获取有效值的索引数组
        value_indices = np.flatnonzero(value_mask).astype(np.int32, copy=False)
        
        # 根据有效值索引和长度创建新的稀疏矩阵索引结构
        new_sp_index = make_sparse_index(len(indices), value_indices, kind=self.kind)
        
        # 返回新对象，使用类方法 _simple_new 创建
        return type(self)._simple_new(new_sp_values, new_sp_index, dtype=self.dtype)

    # 在稀疏矩阵中搜索指定值的插入点索引
    def searchsorted(
        self,
        v: ArrayLike | object,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        # 如果开启了性能警告选项，发出警告
        if get_option("performance_warnings"):
            msg = "searchsorted requires high memory usage."
            warnings.warn(msg, PerformanceWarning, stacklevel=find_stack_level())
        
        # 将输入值 v 转换为 NumPy 数组
        v = np.asarray(v)
        
        # 调用 NumPy 的 searchsorted 方法进行搜索
        return np.asarray(self, dtype=self.dtype.subtype).searchsorted(v, side, sorter)

    # 创建当前稀疏矩阵对象的副本
    def copy(self) -> Self:
        # 复制稀疏矩阵的值数组
        values = self.sp_values.copy()
        # 使用 _simple_new 方法创建并返回副本对象
        return self._simple_new(values, self.sp_index, self.dtype)

    # 类方法装饰器，用于定义类方法
    @classmethod
    # 定义一个类方法用于连接相同类型的稀疏数据
    def _concat_same_type(cls, to_concat: Sequence[Self]) -> Self:
        # 获取要连接数据的填充值
        fill_value = to_concat[0].fill_value

        # 初始化空列表和长度
        values = []
        length = 0

        # 如果有要连接的数据
        if to_concat:
            # 获取第一个数组的稀疏类型
            sp_kind = to_concat[0].kind
        else:
            # 如果没有数据，默认为整数类型
            sp_kind = "integer"

        # 声明稀疏索引对象
        sp_index: SparseIndex

        # 如果稀疏类型为整数
        if sp_kind == "integer":
            # 初始化索引列表
            indices = []

            # 遍历要连接的数组
            for arr in to_concat:
                # 复制整数索引并做偏移处理
                int_idx = arr.sp_index.indices.copy()
                int_idx += length  # TODO: wraparound
                length += arr.sp_index.length

                # 添加稀疏值和索引
                values.append(arr.sp_values)
                indices.append(int_idx)

            # 连接稀疏值数组
            data = np.concatenate(values)
            # 连接整数索引数组
            indices_arr = np.concatenate(indices)
            # 创建整数索引对象
            sp_index = IntIndex(length, indices_arr)  # type: ignore[arg-type]

        else:
            # 当连接块索引时，不保证得到与连接值然后创建新索引相同的索引
            # 不希望在`to_concat`中的数组之间合并块，所以结果的块索引可能会更多
            blengths = []
            blocs = []

            # 遍历要连接的数组
            for arr in to_concat:
                # 转换为块索引对象
                block_idx = arr.sp_index.to_block_index()

                # 添加稀疏值和块索引
                values.append(arr.sp_values)
                blocs.append(block_idx.blocs.copy() + length)
                blengths.append(block_idx.blengths)
                length += arr.sp_index.length

            # 连接稀疏值数组
            data = np.concatenate(values)
            # 连接块索引数组
            blocs_arr = np.concatenate(blocs)
            blengths_arr = np.concatenate(blengths)

            # 创建块索引对象
            sp_index = BlockIndex(length, blocs_arr, blengths_arr)

        # 返回连接后的新实例，包括数据、稀疏索引和填充值
        return cls(data, sparse_index=sp_index, fill_value=fill_value)
    def astype(self, dtype: AstypeArg | None = None, copy: bool = True):
        """
        Change the dtype of a SparseArray.

        The output will always be a SparseArray. To convert to a dense
        ndarray with a certain dtype, use :meth:`numpy.asarray`.

        Parameters
        ----------
        dtype : np.dtype or ExtensionDtype
            For SparseDtype, this changes the dtype of
            ``self.sp_values`` and the ``self.fill_value``.

            For other dtypes, this only changes the dtype of
            ``self.sp_values``.

        copy : bool, default True
            Whether to ensure a copy is made, even if not necessary.

        Returns
        -------
        SparseArray

        Examples
        --------
        >>> arr = pd.arrays.SparseArray([0, 0, 1, 2])
        >>> arr
        [0, 0, 1, 2]
        Fill: 0
        IntIndex
        Indices: array([2, 3], dtype=int32)

        >>> arr.astype(SparseDtype(np.dtype("int32")))
        [0, 0, 1, 2]
        Fill: 0
        IntIndex
        Indices: array([2, 3], dtype=int32)

        Using a NumPy dtype with a different kind (e.g. float) will coerce
        just ``self.sp_values``.

        >>> arr.astype(SparseDtype(np.dtype("float64")))
        ... # doctest: +NORMALIZE_WHITESPACE
        [nan, nan, 1.0, 2.0]
        Fill: nan
        IntIndex
        Indices: array([2, 3], dtype=int32)

        Using a SparseDtype, you can also change the fill value as well.

        >>> arr.astype(SparseDtype("float64", fill_value=0.0))
        ... # doctest: +NORMALIZE_WHITESPACE
        [0.0, 0.0, 1.0, 2.0]
        Fill: 0.0
        IntIndex
        Indices: array([2, 3], dtype=int32)
        """
        # 如果指定的 dtype 与当前的 dtype 相同，并且不需要复制，则直接返回自身
        if dtype == self._dtype:
            if not copy:
                return self
            else:
                # 复制当前对象并返回副本
                return self.copy()

        # 将要转换的 dtype 转换为 Pandas 的 dtype 对象
        future_dtype = pandas_dtype(dtype)

        # 如果转换后的 dtype 不是 SparseDtype 类型
        if not isinstance(future_dtype, SparseDtype):
            # GH#34457
            # 将 SparseArray 转换为密集数组
            values = np.asarray(self)
            # 确保如果数据类似于日期时间，则进行包装
            values = ensure_wrapped_if_datetimelike(values)
            # 调用 astype_array 方法进行数组类型转换，返回新对象
            return astype_array(values, dtype=future_dtype, copy=False)

        # 更新当前 dtype
        dtype = self.dtype.update_dtype(dtype)
        # 获取 subtype
        subtype = pandas_dtype(dtype._subtype_with_str)
        # 确保 subtype 是 np.dtype 类型
        subtype = cast(np.dtype, subtype)  # ensured by update_dtype
        # 确保 sp_values 是包装过的数据
        values = ensure_wrapped_if_datetimelike(self.sp_values)
        # 使用 astype_array 方法转换数据类型，返回新对象
        sp_values = astype_array(values, subtype, copy=copy)
        # 将 sp_values 转换为 ndarray
        sp_values = np.asarray(sp_values)

        # 调用 _simple_new 方法创建新的 SparseArray 对象并返回
        return self._simple_new(sp_values, self.sp_index, dtype)
    def map(self, mapper, na_action: Literal["ignore"] | None = None) -> Self:
        """
        Map categories using an input mapping or function.

        Parameters
        ----------
        mapper : dict, Series, callable
            The correspondence from old values to new.
        na_action : {None, 'ignore'}, default None
            If 'ignore', propagate NA values, without passing them to the
            mapping correspondence.

        Returns
        -------
        SparseArray
            The output array will have the same density as the input.
            The output fill value will be the result of applying the
            mapping to ``self.fill_value``

        Examples
        --------
        >>> arr = pd.arrays.SparseArray([0, 1, 2])
        >>> arr.map(lambda x: x + 10)
        [10, 11, 12]
        Fill: 10
        IntIndex
        Indices: array([1, 2], dtype=int32)

        >>> arr.map({0: 10, 1: 11, 2: 12})
        [10, 11, 12]
        Fill: 10
        IntIndex
        Indices: array([1, 2], dtype=int32)

        >>> arr.map(pd.Series([10, 11, 12], index=[0, 1, 2]))
        [10, 11, 12]
        Fill: 10
        IntIndex
        Indices: array([1, 2], dtype=int32)
        """
        is_map = isinstance(mapper, (abc.Mapping, ABCSeries))  # 检查 mapper 是否是字典或 Series 类型的实例

        fill_val = self.fill_value  # 获取当前 SparseArray 的填充值

        if na_action is None or notna(fill_val):
            # 如果 na_action 是 None 或者 fill_val 不是 NA 值，则根据 mapper 更新 fill_val
            fill_val = mapper.get(fill_val, fill_val) if is_map else mapper(fill_val)

        def func(sp_val):
            # 定义一个函数，用于映射每个稀疏值到新值
            new_sp_val = mapper.get(sp_val, None) if is_map else mapper(sp_val)
            # 检查新值是否与 fill_val 相同，如果是则抛出 ValueError
            # 这是因为稀疏值中的 fill_val 不支持
            if new_sp_val is fill_val or new_sp_val == fill_val:
                msg = "fill value in the sparse values not supported"
                raise ValueError(msg)
            return new_sp_val

        # 对每个稀疏值应用 func 函数进行映射
        sp_values = [func(x) for x in self.sp_values]

        # 返回一个新的 SparseArray 对象，保持与原始对象相同的密度和填充值
        return type(self)(sp_values, sparse_index=self.sp_index, fill_value=fill_val)

    def to_dense(self) -> np.ndarray:
        """
        Convert SparseArray to a NumPy array.

        Returns
        -------
        arr : NumPy array
        """
        return np.asarray(self, dtype=self.sp_values.dtype)  # 将 SparseArray 转换为 NumPy 数组

    def _where(self, mask, value):
        # NB: may not preserve dtype, e.g. result may be Sparse[float64]
        #  while self is Sparse[int64]
        # 使用 np.where 对 SparseArray 中符合条件的值进行替换
        naive_implementation = np.where(mask, self, value)
        # 创建一个与当前 SparseArray 相同类型的新对象，使用 naive_implementation 的数据类型
        dtype = SparseDtype(naive_implementation.dtype, fill_value=self.fill_value)
        result = type(self)._from_sequence(naive_implementation, dtype=dtype)
        return result

    # ------------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------------
    def __setstate__(self, state) -> None:
        """Necessary for making this object picklable"""
        # 检查状态是否为元组，兼容旧版 pandas (< 0.24.0)
        if isinstance(state, tuple):
            # 获取状态中的第一个元素和第二个元素（fill_value, sp_index）
            nd_state, (fill_value, sp_index) = state
            # 创建空的 NumPy 数组 sparse_values
            sparse_values = np.array([])
            # 使用状态中的 nd_state 设置 sparse_values 的状态
            sparse_values.__setstate__(nd_state)

            # 更新对象的稀疏值、稀疏索引和数据类型
            self._sparse_values = sparse_values
            self._sparse_index = sp_index
            self._dtype = SparseDtype(sparse_values.dtype, fill_value)
        else:
            # 直接更新对象的状态字典
            self.__dict__.update(state)

    def nonzero(self) -> tuple[npt.NDArray[np.int32]]:
        # 如果 fill_value 为 0，则返回稀疏索引的 indices 属性
        if self.fill_value == 0:
            return (self.sp_index.indices,)
        else:
            # 否则返回稀疏索引中值不为 0 的元素的 indices 属性
            return (self.sp_index.indices[self.sp_values != 0],)

    # ------------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------------

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ):
        # 获取要执行的方法
        method = getattr(self, name, None)

        # 如果方法不存在，则抛出类型错误
        if method is None:
            raise TypeError(f"cannot perform {name} with type {self.dtype}")

        # 根据 skipna 参数选择处理数组的方式
        if skipna:
            arr = self
        else:
            arr = self.dropna()

        # 调用指定方法计算结果
        result = getattr(arr, name)(**kwargs)

        # 根据 keepdims 参数决定返回结果的类型
        if keepdims:
            return type(self)([result], dtype=self.dtype)
        else:
            return result

    def all(self, axis=None, *args, **kwargs):
        """
        Tests whether all elements evaluate True

        Returns
        -------
        all : bool

        See Also
        --------
        numpy.all
        """
        # 验证 all 方法的参数
        nv.validate_all(args, kwargs)

        # 获取稀疏值数组
        values = self.sp_values

        # 如果稀疏值数组的长度不等于对象的长度且 fill_value 全部为 True，则返回 False
        if len(values) != len(self) and not np.all(self.fill_value):
            return False

        # 否则返回稀疏值数组的 all 方法结果
        return values.all()

    def any(self, axis: AxisInt = 0, *args, **kwargs) -> bool:
        """
        Tests whether at least one of elements evaluate True

        Returns
        -------
        any : bool

        See Also
        --------
        numpy.any
        """
        # 验证 any 方法的参数
        nv.validate_any(args, kwargs)

        # 获取稀疏值数组
        values = self.sp_values

        # 如果稀疏值数组的长度不等于对象的长度且 fill_value 至少有一个 True，则返回 True
        if len(values) != len(self) and np.any(self.fill_value):
            return True

        # 否则返回稀疏值数组的 any 方法结果的单个元素
        return values.any().item()

    def sum(
        self,
        axis: AxisInt = 0,
        min_count: int = 0,
        skipna: bool = True,
        *args,
        **kwargs,
    def sum(self, axis: int = 0, min_count: int = 0, *args, **kwargs) -> Scalar:
        """
        Sum of non-NA/null values

        Parameters
        ----------
        axis : int, default 0
            Not Used. NumPy compatibility.
        min_count : int, default 0
            The required number of valid values to perform the summation. If fewer
            than `min_count` valid values are present, the result will be the missing
            value indicator for subarray type.
        *args, **kwargs
            Not Used. NumPy compatibility.

        Returns
        -------
        scalar
        """
        # Validate arguments related to summation
        nv.validate_sum(args, kwargs)

        # Extract valid non-NA/null values
        valid_vals = self._valid_sp_values

        # Compute the sum of valid values
        sp_sum = valid_vals.sum()

        # Check if there are missing values (NA) and handle accordingly
        has_na = self.sp_index.ngaps > 0 and not self._null_fill_value
        if has_na and not skipna:
            return na_value_for_dtype(self.dtype.subtype, compat=False)

        # Handle cases based on null fill value presence
        if self._null_fill_value:
            # Check if the valid count is below the specified min_count
            if check_below_min_count(valid_vals.shape, None, min_count):
                return na_value_for_dtype(self.dtype.subtype, compat=False)
            return sp_sum
        else:
            # Calculate the sum considering sparse gaps and fill values
            nsparse = self.sp_index.ngaps
            if check_below_min_count(valid_vals.shape, None, min_count - nsparse):
                return na_value_for_dtype(self.dtype.subtype, compat=False)
            return sp_sum + self.fill_value * nsparse

    def cumsum(self, axis: AxisInt = 0, *args, **kwargs) -> SparseArray:
        """
        Cumulative sum of non-NA/null values.

        When performing the cumulative summation, any non-NA/null values will
        be skipped. The resulting SparseArray will preserve the locations of
        NaN values, but the fill value will be `np.nan` regardless.

        Parameters
        ----------
        axis : int or None
            Axis over which to perform the cumulative summation. If None,
            perform cumulative summation over flattened array.

        Returns
        -------
        cumsum : SparseArray
        """
        # Validate arguments related to cumulative summation
        nv.validate_cumsum(args, kwargs)

        # Mimic behavior of ndarray for axis out of bounds
        if axis is not None and axis >= self.ndim:
            raise ValueError(f"axis(={axis}) out of bounds")

        # Perform cumulative summation based on null fill value existence
        if not self._null_fill_value:
            return SparseArray(self.to_dense()).cumsum()

        return SparseArray(
            self.sp_values.cumsum(),
            sparse_index=self.sp_index,
            fill_value=self.fill_value,
        )

    def mean(self, axis: Axis = 0, *args, **kwargs):
        """
        Mean of non-NA/null values

        Returns
        -------
        mean : float
        """
        # Validate arguments related to mean calculation
        nv.validate_mean(args, kwargs)

        # Extract valid non-NA/null values
        valid_vals = self._valid_sp_values

        # Calculate sum of valid values
        sp_sum = valid_vals.sum()

        # Count of valid values
        ct = len(valid_vals)

        # Handle mean calculation based on null fill value presence
        if self._null_fill_value:
            return sp_sum / ct
        else:
            # Calculate mean considering sparse gaps and fill values
            nsparse = self.sp_index.ngaps
            return (sp_sum + self.fill_value * nsparse) / (ct + nsparse)
    def max(self, *, axis: AxisInt | None = None, skipna: bool = True):
        """
        Max of array values, ignoring NA values if specified.

        Parameters
        ----------
        axis : int, default 0
            Not Used. NumPy compatibility.
        skipna : bool, default True
            Whether to ignore NA values.

        Returns
        -------
        scalar
        """
        # Validate axis input against array dimensionality
        nv.validate_minmax_axis(axis, self.ndim)
        # Call _min_max method with 'max' operation
        return self._min_max("max", skipna=skipna)

    def min(self, *, axis: AxisInt | None = None, skipna: bool = True):
        """
        Min of array values, ignoring NA values if specified.

        Parameters
        ----------
        axis : int, default 0
            Not Used. NumPy compatibility.
        skipna : bool, default True
            Whether to ignore NA values.

        Returns
        -------
        scalar
        """
        # Validate axis input against array dimensionality
        nv.validate_minmax_axis(axis, self.ndim)
        # Call _min_max method with 'min' operation
        return self._min_max("min", skipna=skipna)

    def _min_max(self, kind: Literal["min", "max"], skipna: bool) -> Scalar:
        """
        Min/max of non-NA/null values

        Parameters
        ----------
        kind : {"min", "max"}
            Specifies whether to calculate minimum or maximum.
        skipna : bool
            Whether to skip NA values.

        Returns
        -------
        scalar
            Minimum or maximum value depending on 'kind'.
        """
        # Obtain valid values excluding NA/null
        valid_vals = self._valid_sp_values
        # Check if there are non-null fill values and non-gaps in index
        has_nonnull_fill_vals = not self._null_fill_value and self.sp_index.ngaps > 0

        if len(valid_vals) > 0:
            # Compute minimum or maximum of valid values
            sp_min_max = getattr(valid_vals, kind)()

            # If there are non-null fill values, consider them in the calculation
            if has_nonnull_fill_vals:
                func = max if kind == "max" else min
                return func(sp_min_max, self.fill_value)
            elif skipna:
                return sp_min_max
            elif self.sp_index.ngaps == 0:
                # Return computed value when no NA values are present
                return sp_min_max
            else:
                # Return NA value appropriate for the data type
                return na_value_for_dtype(self.dtype.subtype, compat=False)
        elif has_nonnull_fill_vals:
            # Return fill value when no valid values are present but there are non-null fills
            return self.fill_value
        else:
            # Return NA value appropriate for the data type when no valid values or fills are present
            return na_value_for_dtype(self.dtype.subtype, compat=False)
    # 定义一个方法，用于计算稀疏数据中的最小值索引或最大值索引
    def _argmin_argmax(self, kind: Literal["argmin", "argmax"]) -> int:
        # 获取稀疏值数组
        values = self._sparse_values
        # 获取稀疏索引的索引数组
        index = self._sparse_index.indices
        # 创建一个布尔掩码，标记稀疏值中的缺失值
        mask = np.asarray(isna(values))
        # 根据参数确定使用 np.argmax 还是 np.argmin 函数
        func = np.argmax if kind == "argmax" else np.argmin

        # 创建一个包含稀疏值数组长度的索引数组
        idx = np.arange(values.shape[0])
        # 获取非缺失值的稀疏值和对应的索引
        non_nans = values[~mask]
        non_nan_idx = idx[~mask]

        # 找到最大值或最小值的索引
        _candidate = non_nan_idx[func(non_nans)]
        # 根据最大值或最小值的索引找到对应的稀疏索引
        candidate = index[_candidate]

        # 如果填充值是缺失的，则直接返回找到的候选索引
        if isna(self.fill_value):
            return candidate
        # 如果是最小值索引并且候选值小于填充值，则返回候选索引
        if kind == "argmin" and self[candidate] < self.fill_value:
            return candidate
        # 如果是最大值索引并且候选值大于填充值，则返回候选索引
        if kind == "argmax" and self[candidate] > self.fill_value:
            return candidate
        # 否则，查找填充值第一次出现的位置索引
        _loc = self._first_fill_value_loc()
        if _loc == -1:
            # 如果填充值不存在，则返回候选索引
            return candidate
        else:
            # 否则返回填充值的位置索引
            return _loc

    # 定义一个方法，计算数组中的最大值索引
    def argmax(self, skipna: bool = True) -> int:
        # 验证布尔关键字参数
        validate_bool_kwarg(skipna, "skipna")
        # 如果 skipna 为 False 且存在 NA 值，则引发 ValueError 异常
        if not skipna and self._hasna:
            raise ValueError("Encountered an NA value with skipna=False")
        # 调用 _argmin_argmax 方法计算最大值索引，并返回结果
        return self._argmin_argmax("argmax")

    # 定义一个方法，计算数组中的最小值索引
    def argmin(self, skipna: bool = True) -> int:
        # 验证布尔关键字参数
        validate_bool_kwarg(skipna, "skipna")
        # 如果 skipna 为 False 且存在 NA 值，则引发 ValueError 异常
        if not skipna and self._hasna:
            raise ValueError("Encountered an NA value with skipna=False")
        # 调用 _argmin_argmax 方法计算最小值索引，并返回结果
        return self._argmin_argmax("argmin")

    # ------------------------------------------------------------------------
    # Ufuncs
    # ------------------------------------------------------------------------

    # 定义一个元组，包含可以处理的类型，包括 numpy 数组和数字类型
    _HANDLED_TYPES = (np.ndarray, numbers.Number)
    # 定义特殊方法 __array_ufunc__，用于处理 NumPy 的通用函数操作
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        # 获取参数中的 "out"，默认为空元组
        out = kwargs.get("out", ())

        # 检查所有输入和输出是否属于指定的处理类型或 SparseArray 类型
        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES + (SparseArray,)):
                return NotImplemented

        # 对于二元操作，尝试使用自定义的双下划线方法处理
        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is not NotImplemented:
            return result

        # 如果 kwargs 中包含 "out"，例如用于原地操作的场景
        if "out" in kwargs:
            res = arraylike.dispatch_ufunc_with_out(
                self, ufunc, method, *inputs, **kwargs
            )
            return res

        # 如果方法是 "reduce"，则尝试调度为降维操作
        if method == "reduce":
            result = arraylike.dispatch_reduction_ufunc(
                self, ufunc, method, *inputs, **kwargs
            )
            if result is not NotImplemented:
                return result

        # 如果只有一个输入，不需要对齐
        if len(inputs) == 1:
            # 获取 ufunc 对应方法的操作结果，对 sp_values 和 fill_value 分别进行操作
            sp_values = getattr(ufunc, method)(self.sp_values, **kwargs)
            fill_value = getattr(ufunc, method)(self.fill_value, **kwargs)

            # 如果 ufunc 有多个输出
            if ufunc.nout > 1:
                # 例如 modf 等操作，返回多个数组
                arrays = tuple(
                    self._simple_new(
                        sp_value, self.sp_index, SparseDtype(sp_value.dtype, fv)
                    )
                    for sp_value, fv in zip(sp_values, fill_value)
                )
                return arrays
            elif method == "reduce":
                # 对于 reduce 操作，返回 sp_values
                return sp_values

            # 否则，返回新的 SparseArray 对象
            return self._simple_new(
                sp_values, self.sp_index, SparseDtype(sp_values.dtype, fill_value)
            )

        # 如果有多个输入，将它们转换为 NumPy 数组，然后调用对应的 ufunc 方法
        new_inputs = tuple(np.asarray(x) for x in inputs)
        result = getattr(ufunc, method)(*new_inputs, **kwargs)

        # 如果有指定输出，处理输出逻辑
        if out:
            if len(out) == 1:
                out = out[0]
            return out

        # 如果 ufunc 有多个输出，返回多个 SparseArray 类型的对象
        if ufunc.nout > 1:
            return tuple(type(self)(x) for x in result)
        elif method == "at":
            # 对于 "at" 方法，没有返回值
            return None
        else:
            # 否则，返回新的 SparseArray 对象
            return type(self)(result)

    # ------------------------------------------------------------------------
    # Ops
    # ------------------------------------------------------------------------
    # 定义一个私有方法，用于实现基本的算术运算和比较操作
    def _arith_method(self, other, op):
        # 获取操作的名称字符串
        op_name = op.__name__

        # 如果 `other` 是 SparseArray 类型的实例，则调用 _sparse_array_op 函数处理
        if isinstance(other, SparseArray):
            return _sparse_array_op(self, other, op, op_name)

        # 如果 `other` 是标量值
        elif is_scalar(other):
            # 忽略所有的数值错误
            with np.errstate(all="ignore"):
                # 对填充值和 `other` 执行操作
                fill = op(_get_fill(self), np.asarray(other))
                # 对稀疏数组的值和 `other` 执行操作
                result = op(self.sp_values, other)

            # 如果操作是 "divmod"，则需要特别处理返回结果
            if op_name == "divmod":
                left, right = result
                lfill, rfill = fill
                return (
                    _wrap_result(op_name, left, self.sp_index, lfill),
                    _wrap_result(op_name, right, self.sp_index, rfill),
                )

            # 将操作结果和填充值包装成稀疏数组返回
            return _wrap_result(op_name, result, self.sp_index, fill)

        # 如果 `other` 不是 SparseArray 类型，转换为 numpy 数组处理
        else:
            other = np.asarray(other)
            # 忽略所有的数值错误
            with np.errstate(all="ignore"):
                # 如果长度不匹配，则抛出断言错误
                if len(self) != len(other):
                    raise AssertionError(
                        f"length mismatch: {len(self)} vs. {len(other)}"
                    )
                # 如果 `other` 不是 SparseArray 类型，则转换为 SparseArray 对象
                if not isinstance(other, SparseArray):
                    dtype = getattr(other, "dtype", None)
                    other = SparseArray(other, fill_value=self.fill_value, dtype=dtype)
                # 调用 _sparse_array_op 处理稀疏数组的操作
                return _sparse_array_op(self, other, op, op_name)

    # 定义一个私有方法，用于实现比较操作，并返回 SparseArray 类型的结果
    def _cmp_method(self, other, op) -> SparseArray:
        # 如果 `other` 不是标量值且不是 SparseArray 类型，则转换为 numpy 数组
        if not is_scalar(other) and not isinstance(other, type(self)):
            # 将类似列表的 `other` 转换为 ndarray
            other = np.asarray(other)

        # 如果 `other` 是 ndarray 类型，则转换为 SparseArray 类型对象
        if isinstance(other, np.ndarray):
            # TODO: 使得这一步比仅仅是 ndarray 更加灵活...
            other = SparseArray(other, fill_value=self.fill_value)

        # 如果 `other` 是 SparseArray 类型对象
        if isinstance(other, SparseArray):
            # 如果长度不匹配，则抛出 ValueError
            if len(self) != len(other):
                raise ValueError(
                    f"operands have mismatched length {len(self)} and {len(other)}"
                )

            # 获取操作名称并调用 _sparse_array_op 处理稀疏数组的操作
            op_name = op.__name__.strip("_")
            return _sparse_array_op(self, other, op, op_name)
        else:
            # 如果 `other` 是标量值，则对填充值和 `other` 执行操作
            fill_value = op(self.fill_value, other)
            # 创建一个全是布尔值的数组，用 `op` 处理稀疏值和 `other`
            result = np.full(len(self), fill_value, dtype=np.bool_)
            result[self.sp_index.indices] = op(self.sp_values, other)

            # 返回一个新的 SparseArray 对象，使用 `op` 处理的结果和填充值
            return type(self)(
                result,
                fill_value=fill_value,
                dtype=np.bool_,
            )

    # 将 `_cmp_method` 方法赋值给 `_logical_method`，逻辑操作与比较操作相同
    _logical_method = _cmp_method

    # 定义一个私有方法，用于实现一元操作，并返回 SparseArray 类型的结果
    def _unary_method(self, op) -> SparseArray:
        # 使用 `op` 处理填充值，并将结果转换为标量值
        fill_value = op(np.array(self.fill_value)).item()
        # 创建一个新的 SparseDtype 对象，指定数据类型和填充值
        dtype = SparseDtype(self.dtype.subtype, fill_value)
        # 如果填充值没有变化，直接对稀疏值应用 `op`
        if isna(self.fill_value) or fill_value == self.fill_value:
            values = op(self.sp_values)
            # 创建一个简单的新 SparseArray 对象，使用新的稀疏值和索引
            return type(self)._simple_new(values, self.sp_index, self.dtype)
        # 否则需要重新计算索引，创建一个新的 SparseArray 对象
        return type(self)(op(self.to_dense()), dtype=dtype)
    # 定义正数运算符重载方法，返回对应的 SparseArray 对象
    def __pos__(self) -> SparseArray:
        return self._unary_method(operator.pos)

    # 定义负数运算符重载方法，返回对应的 SparseArray 对象
    def __neg__(self) -> SparseArray:
        return self._unary_method(operator.neg)

    # 定义按位取反运算符重载方法，返回对应的 SparseArray 对象
    def __invert__(self) -> SparseArray:
        return self._unary_method(operator.invert)

    # 定义绝对值运算符重载方法，返回对应的 SparseArray 对象
    def __abs__(self) -> SparseArray:
        return self._unary_method(operator.abs)

    # ----------
    # Formatting
    # -----------
    # 定义对象的字符串表示形式方法，返回格式化后的字符串
    def __repr__(self) -> str:
        # 将对象、填充值和索引分别格式化为字符串
        pp_str = printing.pprint_thing(self)
        pp_fill = printing.pprint_thing(self.fill_value)
        pp_index = printing.pprint_thing(self.sp_index)
        # 返回包含对象字符串、填充值和索引信息的格式化字符串
        return f"{pp_str}\nFill: {pp_fill}\n{pp_index}"

    # error: Return type "None" of "_formatter" incompatible with return
    # type "Callable[[Any], str | None]" in supertype "ExtensionArray"
    # 定义格式化方法，用于打印对象的内容，返回 None
    def _formatter(self, boxed: bool = False) -> None:  # type: ignore[override]
        # 委托给 GenericArrayFormatter 的格式化方法处理。
        # 这将根据值的数据类型推断正确的格式化方式。
        return None
# 将给定的 ndarray 转换为稀疏格式

def _make_sparse(
    arr: np.ndarray,
    kind: SparseIndexKind = "block",
    fill_value=None,
    dtype: np.dtype | None = None,
):
    """
    Convert ndarray to sparse format

    Parameters
    ----------
    arr : ndarray
        输入的多维数组
    kind : {'block', 'integer'}
        稀疏索引类型，可以是 'block' 或 'integer'
    fill_value : NaN 或其他数值
        填充值，默认为 None
    dtype : np.dtype, optional
        数据类型，可选

    Returns
    -------
    (sparse_values, index, fill_value) : (ndarray, SparseIndex, Scalar)
        稀疏数组值，稀疏索引，填充值
    """
    assert isinstance(arr, np.ndarray)

    # 检查数组维度是否大于1
    if arr.ndim > 1:
        raise TypeError("expected dimension <= 1 data")

    # 如果填充值为 None，则使用 arr 的数据类型获取填充值
    if fill_value is None:
        fill_value = na_value_for_dtype(arr.dtype)

    # 如果填充值为 NaN，则生成布尔掩码
    if isna(fill_value):
        mask = notna(arr)
    else:
        # 将 arr 的数据类型转换为对象以进行比较
        if is_string_dtype(arr.dtype):
            arr = arr.astype(object)

        # 如果 arr 的数据类型是对象型，则调用特定方法生成对象型数组的掩码
        if is_object_dtype(arr.dtype):
            mask = splib.make_mask_object_ndarray(arr, fill_value)
        else:
            # 否则，通过比较 arr 和填充值生成掩码
            mask = arr != fill_value

    # 获取 arr 的长度
    length = len(arr)

    # 如果 arr 的长度与掩码长度不同，说明 arr 是 SparseArray
    if length != len(mask):
        indices = mask.sp_index.indices
    else:
        # 否则，获取非零元素的索引
        indices = mask.nonzero()[0].astype(np.int32)

    # 生成稀疏索引
    index = make_sparse_index(length, indices, kind)

    # 获取稀疏化后的值
    sparsified_values = arr[mask]

    # 如果指定了数据类型，则将稀疏化后的值转换为指定类型
    if dtype is not None:
        sparsified_values = ensure_wrapped_if_datetimelike(sparsified_values)
        sparsified_values = astype_array(sparsified_values, dtype=dtype)
        sparsified_values = np.asarray(sparsified_values)

    # TODO: copy
    return sparsified_values, index, fill_value


@overload
def make_sparse_index(length: int, indices, kind: Literal["block"]) -> BlockIndex: ...


@overload
def make_sparse_index(length: int, indices, kind: Literal["integer"]) -> IntIndex: ...


def make_sparse_index(length: int, indices, kind: SparseIndexKind) -> SparseIndex:
    """
    Create a sparse index based on the provided parameters.

    Parameters
    ----------
    length : int
        Length of the sparse index.
    indices : array-like
        Indices for the sparse index.
    kind : SparseIndexKind
        Type of sparse index, either 'block' or 'integer'.

    Returns
    -------
    SparseIndex
        The created sparse index object.
    """
    index: SparseIndex
    if kind == "block":
        # 获取块的位置和长度
        locs, lens = splib.get_blocks(indices)
        index = BlockIndex(length, locs, lens)
    elif kind == "integer":
        # 如果是整数类型，直接创建整数索引
        index = IntIndex(length, indices)
    else:  # pragma: no cover
        raise ValueError("must be block or integer type")
    return index
```