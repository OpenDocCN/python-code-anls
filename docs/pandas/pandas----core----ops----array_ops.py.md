# `D:\src\scipysrc\pandas\pandas\core\ops\array_ops.py`

```
# -----------------------------------------------------------------------------
# Masking NA values and fallbacks for operations numpy does not support

# 填充二进制操作的函数，用于处理左右数组中的空值
def fill_binop(left, right, fill_value):
    """
    If a non-None fill_value is given, replace null entries in left and right
    with this value, but only in positions where _one_ of left/right is null,
    not both.

    Parameters
    ----------
    left : array-like
        左侧数组
    right : array-like
        右侧数组
    fill_value : object
        填充的值

    Returns
    -------
    left : array-like
        填充后的左侧数组
    right : array-like
        填充后的右侧数组

    Notes
    -----
    Makes copies if fill_value is not None and NAs are present.
    """
    if fill_value is not None:
        left_mask = isna(left)  # 检查左侧数组中的空值
        right_mask = isna(right)  # 检查右侧数组中的空值

        # one but not both
        mask = left_mask ^ right_mask  # 仅在左右数组中一个为空的情况下应用填充值

        if left_mask.any():
            # 避免在可能的情况下进行复制
            left = left.copy()
            left[left_mask & mask] = fill_value  # 将填充值应用于左侧数组中的空值位置

        if right_mask.any():
            # 避免在可能的情况下进行复制
            right = right.copy()
            right[right_mask & mask] = fill_value  # 将填充值应用于右侧数组中的空值位置

    return left, right


def comp_method_OBJECT_ARRAY(op, x, y):
    if isinstance(y, list):
        # e.g. test_tuple_categories
        y = construct_1d_object_array_from_listlike(y)  # 如果y是列表，则将其转换为1维对象数组

    if isinstance(y, (np.ndarray, ABCSeries, ABCIndex)):
        if not is_object_dtype(y.dtype):
            y = y.astype(np.object_)  # 将y转换为对象类型的numpy数组

        if isinstance(y, (ABCSeries, ABCIndex)):
            y = y._values  # 获取y的值数组

        if x.shape != y.shape:
            raise ValueError("Shapes must match", x.shape, y.shape)  # 检查x和y的形状是否匹配
        result = libops.vec_compare(x.ravel(), y.ravel(), op)  # 使用vec_compare函数比较x和y的向量化版本
    else:
        # 如果不是向量化操作，则调用 libops 模块的 scalar_compare 函数进行标量比较
        result = libops.scalar_compare(x.ravel(), y, op)
    # 返回比较结果，并将结果重新调整为输入张量 x 的形状
    return result.reshape(x.shape)
# 定义一个函数用于处理带掩码的数组运算，支持处理输入数组中的非空元素
def _masked_arith_op(x: np.ndarray, y, op) -> np.ndarray:
    """
    If the given arithmetic operation fails, attempt it again on
    only the non-null elements of the input array(s).

    Parameters
    ----------
    x : np.ndarray
        输入的数组 x
    y : np.ndarray, Series, Index
        输入的数组 y，可以是 Series 或 Index
    op : binary operator
        二元运算符，例如加法、减法等
    """
    # 对 Series 进行 ravel() 操作是为了确保逻辑对 Series 和 DataFrame 都有效
    xrav = x.ravel()

    if isinstance(y, np.ndarray):
        # 找到 x 和 y 的公共数据类型
        dtype = find_common_type([x.dtype, y.dtype])
        # 创建一个空数组来存放结果，数据类型为找到的公共类型
        result = np.empty(x.size, dtype=dtype)

        if len(x) != len(y):
            # 如果 x 和 y 的长度不同，抛出 ValueError 异常
            raise ValueError(x.shape, y.shape)
        # 使用 notna() 函数获取 y 的掩码
        ymask = notna(y)

        # 根据历史记录，使用 ravel() 是安全的，因为 y 是 ndarray 类型
        yrav = y.ravel()
        # 创建 x 和 y 非空元素的掩码
        mask = notna(xrav) & ymask.ravel()

        # 根据历史参考，如果掩码中有值，则进行运算操作
        if mask.any():
            result[mask] = op(xrav[mask], yrav[mask])

    else:
        if not is_scalar(y):
            # 如果 y 不是标量，则抛出 TypeError 异常
            raise TypeError(
                f"Cannot broadcast np.ndarray with operand of type { type(y) }"
            )

        # 对于 x 来说，掩码只有在有意义的情况下才有用
        result = np.empty(x.size, dtype=x.dtype)
        mask = notna(xrav)

        # 处理特殊情况，例如 pow 操作
        if op is pow:
            mask = np.where(x == 1, False, mask)
        elif op is roperator.rpow:
            mask = np.where(y == 1, False, mask)

        # 如果掩码中有值，则进行运算操作
        if mask.any():
            result[mask] = op(xrav[mask], y)

    # 将未被掩码覆盖的位置设置为 NaN
    np.putmask(result, ~mask, np.nan)
    # 将结果重新调整为 x 的形状，以保证与输入的二维数组兼容
    result = result.reshape(x.shape)  # 2D compat
    return result


```  
# 返回使用给定操作符计算左右值的结果

## 参数
Evaluatefunc是一个
    # 如果 is_cmp 为真，并且 result 是标量或者为 NotImplemented
    # numpy 返回了一个标量而不是按元素操作
    # 例如，数值数组与字符串的比较
    # TODO: 在删除某些未来的 numpy 版本后可以移除这部分代码？
    # 调用 invalid_comparison 函数处理无效的比较操作，传入左操作数 left、右操作数 right 和操作符 op
    return invalid_comparison(left, right, op)
def arithmetic_op(left: ArrayLike, right: Any, op):
    """
    Evaluate an arithmetic operation `+`, `-`, `*`, `/`, `//`, `%`, `**`, ...

    Note: the caller is responsible for ensuring that numpy warnings are
    suppressed (with np.errstate(all="ignore")) if needed.

    Parameters
    ----------
    left : np.ndarray or ExtensionArray
        The left operand of the arithmetic operation.
    right : object
        The right operand of the arithmetic operation. Cannot be a DataFrame or Index.
        Series is *not* excluded.
    op : {operator.add, operator.sub, ...}
        The operator function for the arithmetic operation.

    Returns
    -------
    ndarray or ExtensionArray
        Result of the arithmetic operation.
        Or a 2-tuple of these in the case of divmod or rdivmod.
    """
    # NB: We assume that extract_array and ensure_wrapped_if_datetimelike
    #  have already been called on `left` and `right`,
    #  and `maybe_prepare_scalar_for_op` has already been called on `right`
    # We need to special-case datetime64/timedelta64 dtypes (e.g. because numpy
    # casts integer dtypes to timedelta64 when operating with timedelta64 - GH#22390)

    if (
        should_extension_dispatch(left, right)
        or isinstance(right, (Timedelta, BaseOffset, Timestamp))
        or right is NaT
    ):
        # Timedelta/Timestamp and other custom scalars are included in the check
        # because numexpr will fail on it, see GH#31457
        res_values = op(left, right)
    else:
        # TODO we should handle EAs consistently and move this check before the if/else
        # (https://github.com/pandas-dev/pandas/issues/41165)
        # error: Argument 2 to "_bool_arith_check" has incompatible type
        # "Union[ExtensionArray, ndarray[Any, Any]]"; expected "ndarray[Any, Any]"
        _bool_arith_check(op, left, right)  # type: ignore[arg-type]

        # error: Argument 1 to "_na_arithmetic_op" has incompatible type
        # "Union[ExtensionArray, ndarray[Any, Any]]"; expected "ndarray[Any, Any]"
        res_values = _na_arithmetic_op(left, right, op)  # type: ignore[arg-type]

    return res_values


def comparison_op(left: ArrayLike, right: Any, op) -> ArrayLike:
    """
    Evaluate a comparison operation `=`, `!=`, `>=`, `>`, `<=`, or `<`.

    Note: the caller is responsible for ensuring that numpy warnings are
    suppressed (with np.errstate(all="ignore")) if needed.

    Parameters
    ----------
    left : np.ndarray or ExtensionArray
        The left operand of the comparison operation.
    right : object
        The right operand of the comparison operation. Cannot be a DataFrame, Series, or Index.
    op : {operator.eq, operator.ne, operator.gt, operator.ge, operator.lt, operator.le}
        The operator function for the comparison operation.

    Returns
    -------
    ndarray or ExtensionArray
        Boolean array resulting from the comparison operation.
    """
    # NB: We assume extract_array has already been called on left and right
    lvalues = ensure_wrapped_if_datetimelike(left)
    rvalues = ensure_wrapped_if_datetimelike(right)

    rvalues = lib.item_from_zerodim(rvalues)
    # 如果 rvalues 是列表类型
    if isinstance(rvalues, list):
        # 这里不捕获元组类型，因为可能需要比较 MultiIndex 和表示单个条目的元组，参见 test_compare_tuple_strs
        rvalues = np.asarray(rvalues)

    # 如果 rvalues 是 numpy 数组或扩展数组类型
    if isinstance(rvalues, (np.ndarray, ABCExtensionArray)):
        # TODO: 使得这种处理在所有操作和类中保持一致。
        # 这里没有捕获所有类似列表的情况（如 frozenset、tuple）
        # 不确定的情况是对象类型数据。参见 GH#27803
        if len(lvalues) != len(rvalues):
            raise ValueError(
                "Lengths must match to compare", lvalues.shape, rvalues.shape
            )

    # 如果应该使用扩展分派（extension dispatch）或者 rvalues 是时间增量、基础偏移或时间戳类型，
    # 或者 right 是 NaT（Not a Time），并且 lvalues 的数据类型不是对象类型
    if should_extension_dispatch(lvalues, rvalues) or (
        (isinstance(rvalues, (Timedelta, BaseOffset, Timestamp)) or right is NaT)
        and lvalues.dtype != object
    ):
        # 在 lvalues 上调用操作方法
        res_values = op(lvalues, rvalues)

    # 如果 rvalues 是标量并且是缺失值（NaN）
    elif is_scalar(rvalues) and isna(rvalues):  # TODO: 但不包括 pd.NA？
        # numpy 不喜欢与 None 比较
        if op is operator.ne:
            res_values = np.ones(lvalues.shape, dtype=bool)
        else:
            res_values = np.zeros(lvalues.shape, dtype=bool)

    # 如果 lvalues 和 rvalues 是数值和字符串类型的比较
    elif is_numeric_v_string_like(lvalues, rvalues):
        # 通过 numexpr 路径会错误地引发异常 GH#36377
        return invalid_comparison(lvalues, rvalues, op)

    # 如果 lvalues 的数据类型是对象类型或者 rvalues 是字符串类型
    elif lvalues.dtype == object or isinstance(rvalues, str):
        res_values = comp_method_OBJECT_ARRAY(op, lvalues, rvalues)

    # 否则，调用 _na_arithmetic_op 处理 lvalues 和 rvalues 的非数值计算操作
    else:
        res_values = _na_arithmetic_op(lvalues, rvalues, op, is_cmp=True)

    # 返回比较结果的值
    return res_values
def na_logical_op(x: np.ndarray, y, op):
    try:
        # 尝试进行逻辑操作
        result = op(x, y)
    except TypeError:
        if isinstance(y, np.ndarray):
            # 处理布尔类型数组的情况
            assert not (x.dtype.kind == "b" and y.dtype.kind == "b")
            x = ensure_object(x)
            y = ensure_object(y)
            # 对向量进行二元操作
            result = libops.vec_binop(x.ravel(), y.ravel(), op)
        else:
            # 处理标量类型的情况
            assert lib.is_scalar(y)
            if not isna(y):
                y = bool(y)
            try:
                # 对标量进行二元操作
                result = libops.scalar_binop(x, y, op)
            except (
                TypeError,
                ValueError,
                AttributeError,
                OverflowError,
                NotImplementedError,
            ) as err:
                typ = type(y).__name__
                # 抛出类型错误，说明无法执行给定操作
                raise TypeError(
                    f"Cannot perform '{op.__name__}' with a dtyped [{x.dtype}] array "
                    f"and scalar of type [{typ}]"
                ) from err

    return result.reshape(x.shape)


def logical_op(left: ArrayLike, right: Any, op) -> ArrayLike:
    """
    Evaluate a logical operation `|`, `&`, or `^`.

    Parameters
    ----------
    left : np.ndarray or ExtensionArray
    right : object
        Cannot be a DataFrame, Series, or Index.
    op : {operator.and_, operator.or_, operator.xor}
        Or one of the reversed variants from roperator.

    Returns
    -------
    ndarray or ExtensionArray
    """

    def fill_bool(x, left=None):
        # 如果 `left` 不是布尔类型，不进行类型转换
        if x.dtype.kind in "cfO":
            # 可以包含 NA 的数据类型
            mask = isna(x)
            if mask.any():
                x = x.astype(object)
                x[mask] = False

        if left is None or left.dtype.kind == "b":
            # 如果 `left` 是 None 或者布尔类型，将数组转换为布尔类型
            x = x.astype(bool)
        return x

    right = lib.item_from_zerodim(right)
    if is_list_like(right) and not hasattr(right, "dtype"):
        # 例如列表、元组等不带有 dtype 的序列
        raise TypeError(
            # GH#52264
            "Logical ops (and, or, xor) between Pandas objects and dtype-less "
            "sequences (e.g. list, tuple) are no longer supported. "
            "Wrap the object in a Series, Index, or np.array "
            "before operating instead.",
        )

    # 注意：我们假设 extract_array 已经在 left 和 right 上调用过
    lvalues = ensure_wrapped_if_datetimelike(left)
    rvalues = right
    # 如果应该根据扩展名分派操作，则调用操作函数 op 对 lvalues 和 rvalues 进行计算
    if should_extension_dispatch(lvalues, rvalues):
        # 在 lvalues 上调用操作函数 op，并将结果存储在 res_values 中
        res_values = op(lvalues, rvalues)

    else:
        # 如果 rvalues 是一个 NumPy 数组
        if isinstance(rvalues, np.ndarray):
            # 检查 rvalues 的数据类型是否为整数类型
            is_other_int_dtype = rvalues.dtype.kind in "iu"
            # 如果不是整数类型，则用 lvalues 填充 rvalues
            if not is_other_int_dtype:
                rvalues = fill_bool(rvalues, lvalues)

        else:
            # 即标量情况
            # 检查 rvalues 是否为整数
            is_other_int_dtype = lib.is_integer(rvalues)

        # 对 lvalues 和 rvalues 执行逻辑操作 na_logical_op，并将结果存储在 res_values 中
        res_values = na_logical_op(lvalues, rvalues, op)

        # 对于整数之间的按位操作 `^`, `|`, `&`，返回整数数据类型；否则返回布尔操作结果
        if not (left.dtype.kind in "iu" and is_other_int_dtype):
            # 将结果 res_values 转换为布尔类型
            res_values = fill_bool(res_values)

    # 返回操作结果 res_values
    return res_values
def get_array_op(op):
    """
    Return a binary array operation corresponding to the given operator op.

    Parameters
    ----------
    op : function
        Binary operator from operator or roperator module.

    Returns
    -------
    functools.partial
    """
    if isinstance(op, partial):
        # 如果操作符 op 已经是 functools.partial 类型，则直接返回，例如在 DataFrame 情况下通过 dispatch_to_series 调用，比如 test_rolling_consistency_var_debiasing_factors
        return op

    op_name = op.__name__.strip("_").lstrip("r")
    if op_name == "arith_op":
        # 如果操作符 op 是 "arith_op"，通过 DataFrame._combine_frame 方法调用，比如 test_df_add_flex_filled_mixed_dtypes
        return op

    if op_name in {"eq", "ne", "lt", "le", "gt", "ge"}:
        # 如果操作符 op 是比较操作符之一，则返回一个 functools.partial，调用 comparison_op 函数，传入操作符 op
        return partial(comparison_op, op=op)
    elif op_name in {"and", "or", "xor", "rand", "ror", "rxor"}:
        # 如果操作符 op 是逻辑操作符之一，则返回一个 functools.partial，调用 logical_op 函数，传入操作符 op
        return partial(logical_op, op=op)
    elif op_name in {
        "add",
        "sub",
        "mul",
        "truediv",
        "floordiv",
        "mod",
        "divmod",
        "pow",
    }:
        # 如果操作符 op 是算术操作符之一，则返回一个 functools.partial，调用 arithmetic_op 函数，传入操作符 op
        return partial(arithmetic_op, op=op)
    else:
        # 如果操作符 op 无法识别，则抛出 NotImplementedError 异常，传入操作符 op 的名称 op_name
        raise NotImplementedError(op_name)


def maybe_prepare_scalar_for_op(obj, shape: Shape):
    """
    Cast non-pandas objects to pandas types to unify behavior of arithmetic
    and comparison operations.

    Parameters
    ----------
    obj: object
    shape : tuple[int]

    Returns
    -------
    out : object

    Notes
    -----
    Be careful to call this *after* determining the `name` attribute to be
    attached to the result of the arithmetic operation.
    """
    if type(obj) is datetime.timedelta:
        # 如果 obj 的类型是 datetime.timedelta，则将其转换为 Timedelta 类型对象，确保依赖 Timedelta 的实现，否则对 numeric-dtype 进行操作会引发 TypeError
        return Timedelta(obj)
    elif type(obj) is datetime.datetime:
        # 如果 obj 的类型是 datetime.datetime，则将其转换为 Timestamp 类型对象，确保依赖 Timestamp 的实现，参见上文对 Timedelta 的描述
        return Timestamp(obj)
    elif isinstance(obj, np.datetime64):
        # 如果 obj 是 np.datetime64 类型
        # GH#28080 numpy 在执行 array[int] + datetime64 时将 integer-dtype 转换为 datetime64，我们不允许这种操作
        if isna(obj):
            from pandas.core.arrays import DatetimeArray

            # 避免可能的歧义情况，使用 pd.NaT
            # GH 52295
            if is_unitless(obj.dtype):
                obj = obj.astype("datetime64[ns]")
            elif not is_supported_dtype(obj.dtype):
                new_dtype = get_supported_dtype(obj.dtype)
                obj = obj.astype(new_dtype)
            right = np.broadcast_to(obj, shape)
            # 创建一个新的 DatetimeArray 对象，类型为 right 的 dtype
            return DatetimeArray._simple_new(right, dtype=right.dtype)

        return Timestamp(obj)
    # 如果对象是 np.timedelta64 类型
    elif isinstance(obj, np.timedelta64):
        # 如果对象是空值
        if isna(obj):
            from pandas.core.arrays import TimedeltaArray

            # 将 timedelta64("NaT") 包装在 Timedelta 中返回 NaT，
            # 这样会错误地被视为 datetime-NaT，因此我们进行广播并包装在 TimedeltaArray 中
            # GH 52295
            # 如果时间单位是无单位的，则转换为 "timedelta64[ns]"
            if is_unitless(obj.dtype):
                obj = obj.astype("timedelta64[ns]")
            # 如果数据类型不受支持，则转换为支持的数据类型
            elif not is_supported_dtype(obj.dtype):
                new_dtype = get_supported_dtype(obj.dtype)
                obj = obj.astype(new_dtype)
            # 将 obj 广播到指定的 shape
            right = np.broadcast_to(obj, shape)
            # 返回一个新的 TimedeltaArray 对象
            return TimedeltaArray._simple_new(right, dtype=right.dtype)

        # 特别是非纳秒级的 timedelta64 需要转换为纳秒级，否则会出现不希望的行为，如
        # np.timedelta64(3, 'D') / 2 == np.timedelta64(1, 'D')
        # 返回一个 Timedelta 对象
        return Timedelta(obj)

    # 我们希望 NumPy 的数值标量表现得像 Python 的标量
    # 在 NEP 50 之后
    elif isinstance(obj, np.integer):
        # 将 NumPy 整数类型转换为 Python 整数
        return int(obj)

    elif isinstance(obj, np.floating):
        # 将 NumPy 浮点数类型转换为 Python 浮点数
        return float(obj)

    # 返回原始对象，如果不是以上任何类型的对象
    return obj
# 定义一个集合 `_BOOL_OP_NOT_ALLOWED`，包含不允许的布尔运算操作的运算符函数
_BOOL_OP_NOT_ALLOWED = {
    operator.truediv,       # 真除运算符
    roperator.rtruediv,     # 右侧真除运算符
    operator.floordiv,      # 地板除运算符
    roperator.rfloordiv,    # 右侧地板除运算符
    operator.pow,           # 幂运算符
    roperator.rpow,         # 右侧幂运算符
}


def _bool_arith_check(op, a: np.ndarray, b) -> None:
    """
    与 numpy 不同，pandas 在某些布尔操作上会引发错误。
    """
    # 检查操作符是否在不允许的布尔运算操作集合中
    if op in _BOOL_OP_NOT_ALLOWED:
        # 如果数组 a 的数据类型为布尔类型，并且 b 是布尔类型或者由 pandas 提供的布尔类型，则抛出未实现错误
        if a.dtype.kind == "b" and (is_bool_dtype(b) or lib.is_bool(b)):
            # 获取操作符的名称，剥离下划线并移除开头的 'r'
            op_name = op.__name__.strip("_").lstrip("r")
            # 抛出未实现错误，提示该操作符不支持布尔类型数据
            raise NotImplementedError(
                f"operator '{op_name}' not implemented for bool dtypes"
            )
```