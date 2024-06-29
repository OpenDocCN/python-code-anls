# `.\numpy\numpy\_expired_attrs_2_0.py`

```py
"""
Dict of expired attributes that are discontinued since 2.0 release.
Each item is associated with a migration note.
"""

# 这是一个包含自2.0版本以来已停用属性的字典。
# 每个条目都附带有迁移说明。

__expired_attributes__ = {
    "geterrobj": "Use the np.errstate context manager instead.",
    # 使用 np.errstate 上下文管理器替代。
    "seterrobj": "Use the np.errstate context manager instead.",
    # 使用 np.errstate 上下文管理器替代。
    "cast": "Use `np.asarray(arr, dtype=dtype)` instead.",
    # 使用 `np.asarray(arr, dtype=dtype)` 替代。
    "source": "Use `inspect.getsource` instead.",
    # 使用 `inspect.getsource` 替代。
    "lookfor":  "Search NumPy's documentation directly.",
    # 直接搜索 NumPy 文档。
    "who": "Use an IDE variable explorer or `locals()` instead.",
    # 使用 IDE 的变量资源管理器或 `locals()` 替代。
    "fastCopyAndTranspose": "Use `arr.T.copy()` instead.",
    # 使用 `arr.T.copy()` 替代。
    "set_numeric_ops": 
        "For the general case, use `PyUFunc_ReplaceLoopBySignature`. "
        "For ndarray subclasses, define the ``__array_ufunc__`` method "
        "and override the relevant ufunc.",
    # 对于一般情况，请使用 `PyUFunc_ReplaceLoopBySignature`。
    # 对于 ndarray 子类，请定义 `__array_ufunc__` 方法并重写相关的 ufunc。
    "NINF": "Use `-np.inf` instead.",
    # 使用 `-np.inf` 替代。
    "PINF": "Use `np.inf` instead.",
    # 使用 `np.inf` 替代。
    "NZERO": "Use `-0.0` instead.",
    # 使用 `-0.0` 替代。
    "PZERO": "Use `0.0` instead.",
    # 使用 `0.0` 替代。
    "add_newdoc": 
        "It's still available as `np.lib.add_newdoc`.",
    # 仍然可以使用 `np.lib.add_newdoc`。
    "add_docstring": 
        "It's still available as `np.lib.add_docstring`.",
    # 仍然可以使用 `np.lib.add_docstring`。
    "add_newdoc_ufunc": 
        "It's an internal function and doesn't have a replacement.",
    # 这是一个内部函数，没有替代方法。
    "compat": "There's no replacement, as Python 2 is no longer supported.",
    # 没有替代方法，因为不再支持 Python 2。
    "safe_eval": "Use `ast.literal_eval` instead.",
    # 使用 `ast.literal_eval` 替代。
    "float_": "Use `np.float64` instead.",
    # 使用 `np.float64` 替代。
    "complex_": "Use `np.complex128` instead.",
    # 使用 `np.complex128` 替代。
    "longfloat": "Use `np.longdouble` instead.",
    # 使用 `np.longdouble` 替代。
    "singlecomplex": "Use `np.complex64` instead.",
    # 使用 `np.complex64` 替代。
    "cfloat": "Use `np.complex128` instead.",
    # 使用 `np.complex128` 替代。
    "longcomplex": "Use `np.clongdouble` instead.",
    # 使用 `np.clongdouble` 替代。
    "clongfloat": "Use `np.clongdouble` instead.",
    # 使用 `np.clongdouble` 替代。
    "string_": "Use `np.bytes_` instead.",
    # 使用 `np.bytes_` 替代。
    "unicode_": "Use `np.str_` instead.",
    # 使用 `np.str_` 替代。
    "Inf": "Use `np.inf` instead.",
    # 使用 `np.inf` 替代。
    "Infinity": "Use `np.inf` instead.",
    # 使用 `np.inf` 替代。
    "NaN": "Use `np.nan` instead.",
    # 使用 `np.nan` 替代。
    "infty": "Use `np.inf` instead.",
    # 使用 `np.inf` 替代。
    "issctype": "Use `issubclass(rep, np.generic)` instead.",
    # 使用 `issubclass(rep, np.generic)` 替代。
    "maximum_sctype":
        "Use a specific dtype instead. You should avoid relying "
        "on any implicit mechanism and select the largest dtype of "
        "a kind explicitly in the code.",
    # 使用特定的 dtype 替代。应避免依赖任何隐式机制，并在代码中明确选择一种 dtype 的最大值。
    "obj2sctype": "Use `np.dtype(obj).type` instead.",
    # 使用 `np.dtype(obj).type` 替代。
    "sctype2char": "Use `np.dtype(obj).char` instead.",
    # 使用 `np.dtype(obj).char` 替代。
    "sctypes": "Access dtypes explicitly instead.",
    # 直接访问 dtypes。
    "issubsctype": "Use `np.issubdtype` instead.",
    # 使用 `np.issubdtype` 替代。
    "set_string_function": 
        "Use `np.set_printoptions` instead with a formatter for "
        "custom printing of NumPy objects.",
    # 使用 `np.set_printoptions` 并为自定义打印 NumPy 对象设置格式化器。
    "asfarray": "Use `np.asarray` with a proper dtype instead.",
    # 使用带有正确 dtype 的 `np.asarray` 替代。
    "issubclass_": "Use `issubclass` builtin instead.",
    # 使用内置的 `issubclass` 替代。
    "tracemalloc_domain": "It's now available from `np.lib`.",
    # 现在可以从 `np.lib` 获取。
    "mat": "Use `np.asmatrix` instead.",
    # 使用 `np.asmatrix` 替代。
    "recfromcsv": "Use `np.genfromtxt` with comma delimiter instead.",
    # 使用带有逗号分隔符的 `np.genfromtxt` 替代。
    "recfromtxt": "Use `np.genfromtxt` instead.",
    # 使用 `np.genfromtxt` 替代。
    "deprecate": "Emit `DeprecationWarning` with `warnings.warn` directly, "
        "or use `typing.deprecated`.",
    # 直接发出 `DeprecationWarning`，或者使用 `warnings.warn`。
    "deprecate_with_doc": "Emit `DeprecationWarning` with `warnings.warn` "
        "directly, or use `typing.deprecated`.",
    # 建议使用 `warnings.warn` 直接发出 `DeprecationWarning`，或者使用 `typing.deprecated`。
    "disp": "Use your own printing function instead.",
    # 建议使用自己的打印函数代替。
    "find_common_type": 
        "Use `numpy.promote_types` or `numpy.result_type` instead. "
        "To achieve semantics for the `scalar_types` argument, use "
        "`numpy.result_type` and pass the Python values `0`, `0.0`, or `0j`.",
    # 建议使用 `numpy.promote_types` 或 `numpy.result_type`。为了实现 `scalar_types` 参数的语义，可以使用 `numpy.result_type` 并传递 Python 值 `0`、`0.0` 或 `0j`。
    "round_": "Use `np.round` instead.",
    # 建议使用 `np.round` 替代。
    "get_array_wrap": "",
    # 无注释内容，可能表示此处无特别需要说明的内容。
    "DataSource": "It's still available as `np.lib.npyio.DataSource`.", 
    # 仍可通过 `np.lib.npyio.DataSource` 获取此功能。
    "nbytes": "Use `np.dtype(<dtype>).itemsize` instead.",  
    # 建议使用 `np.dtype(<dtype>).itemsize` 替代。
    "byte_bounds": "Now it's available under `np.lib.array_utils.byte_bounds`",
    # 现在可以在 `np.lib.array_utils.byte_bounds` 下找到此功能。
    "compare_chararrays": 
        "It's still available as `np.char.compare_chararrays`.",
    # 仍可通过 `np.char.compare_chararrays` 获取此功能。
    "format_parser": "It's still available as `np.rec.format_parser`."
    # 仍可通过 `np.rec.format_parser` 获取此功能。
}


注释：

# 这是一个单独的右大括号，用于结束一个代码块或语句。
```