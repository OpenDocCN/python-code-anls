# `D:\src\scipysrc\pandas\pandas\core\strings\accessor.py`

```
from __future__ import annotations
# 引入将来版本兼容性的声明，确保代码在较老版本的Python中也能正常运行

import codecs
# 导入codecs模块，用于字符编解码

from functools import wraps
# 导入functools模块中的wraps装饰器，用于包装函数，保留原函数的元数据

import re
# 导入re模块，用于正则表达式操作

from typing import (
    TYPE_CHECKING,
    Literal,
    cast,
)
# 导入typing模块中的类型相关定义，包括TYPE_CHECKING、Literal和cast等

import warnings
# 导入warnings模块，用于警告控制

import numpy as np
# 导入numpy库，并使用别名np

from pandas._libs import lib
# 从pandas._libs模块导入lib

from pandas._typing import (
    AlignJoin,
    DtypeObj,
    F,
    Scalar,
    npt,
)
# 从pandas._typing模块导入AlignJoin、DtypeObj、F、Scalar和npt等类型定义

from pandas.util._decorators import Appender
# 从pandas.util._decorators模块导入Appender装饰器

from pandas.util._exceptions import find_stack_level
# 从pandas.util._exceptions模块导入find_stack_level函数

from pandas.core.dtypes.common import (
    ensure_object,
    is_bool_dtype,
    is_integer,
    is_list_like,
    is_object_dtype,
    is_re,
)
# 从pandas.core.dtypes.common模块导入各种数据类型判断函数，如ensure_object、is_bool_dtype等

from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    CategoricalDtype,
)
# 从pandas.core.dtypes.dtypes模块导入ArrowDtype和CategoricalDtype等数据类型定义

from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCMultiIndex,
    ABCSeries,
)
# 从pandas.core.dtypes.generic模块导入ABCDataFrame、ABCIndex、ABCMultiIndex和ABCSeries等类定义

from pandas.core.dtypes.missing import isna
# 从pandas.core.dtypes.missing模块导入isna函数，用于检查缺失值

from pandas.core.arrays import ExtensionArray
# 从pandas.core.arrays模块导入ExtensionArray类，用于扩展数组操作

from pandas.core.base import NoNewAttributesMixin
# 从pandas.core.base模块导入NoNewAttributesMixin类，用于阻止新属性的添加

from pandas.core.construction import extract_array
# 从pandas.core.construction模块导入extract_array函数，用于提取数组

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Iterator,
    )
    # 如果在类型检查模式下，则导入collections.abc模块中的Callable、Hashable和Iterator等类型

    from pandas import (
        DataFrame,
        Index,
        Series,
    )
    # 导入pandas模块中的DataFrame、Index和Series类

_shared_docs: dict[str, str] = {}
# 定义一个空字典_shared_docs，用于存储共享文档字符串，键和值都是字符串类型

_cpython_optimized_encoders = (
    "utf-8",
    "utf8",
    "latin-1",
    "latin1",
    "iso-8859-1",
    "mbcs",
    "ascii",
)
# 定义一个元组_cpython_optimized_encoders，包含多种Python解释器优化的编码格式名称

_cpython_optimized_decoders = _cpython_optimized_encoders + ("utf-16", "utf-32")
# 定义一个元组_cpython_optimized_decoders，包含_cpython_optimized_encoders中的编码格式和额外的"utf-16"和"utf-32"

def forbid_nonstring_types(
    forbidden: list[str] | None, name: str | None = None
) -> Callable[[F], F]:
    """
    Decorator to forbid specific types for a method of StringMethods.

    For calling `.str.{method}` on a Series or Index, it is necessary to first
    initialize the :class:`StringMethods` object, and then call the method.
    However, different methods allow different input types, and so this can not
    be checked during :meth:`StringMethods.__init__`, but must be done on a
    per-method basis. This decorator exists to facilitate this process, and
    make it explicit which (inferred) types are disallowed by the method.

    :meth:`StringMethods.__init__` allows the *union* of types its different
    methods allow (after skipping NaNs; see :meth:`StringMethods._validate`),
    namely: ['string', 'empty', 'bytes', 'mixed', 'mixed-integer'].

    The default string types ['string', 'empty'] are allowed for all methods.
    For the additional types ['bytes', 'mixed', 'mixed-integer'], each method
    then needs to forbid the types it is not intended for.

    Parameters
    ----------
    forbidden : list-of-str or None
        List of forbidden non-string types, may be one or more of
        `['bytes', 'mixed', 'mixed-integer']`.
    name : str, default None
        Name of the method to use in the error message. By default, this is
        None, in which case the name from the method being wrapped will be
        copied. However, for working with further wrappers (like _pat_wrapper
        and _noarg_wrapper), it is necessary to specify the name.

    Returns
    -------
    Callable[[F], F]
        Decorator that forbids specific non-string types for a method.
    """
    # 函数装饰器，用于限制StringMethods中某些方法的输入类型为字符串类型或禁止特定非字符串类型
    pass
    # 返回实际的装饰器函数
    func : wrapper
        The method to which the decorator is applied, with an added check that
        enforces the inferred type to not be in the list of forbidden types.

    Raises
    ------
    TypeError
        If the inferred type of the underlying data is in `forbidden`.
    """
    # 处理禁止类型为空的情况
    forbidden = [] if forbidden is None else forbidden

    # 计算允许的类型，从预定义类型集合中排除禁止的类型
    allowed_types = {"string", "empty", "bytes", "mixed", "mixed-integer"} - set(
        forbidden
    )

    def _forbid_nonstring_types(func: F) -> F:
        # 获取函数的名称，如果未指定则使用函数本身的名称
        func_name = func.__name__ if name is None else name

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # 检查推断出的数据类型是否在允许的类型集合中
            if self._inferred_dtype not in allowed_types:
                msg = (
                    f"Cannot use .str.{func_name} with values of "
                    f"inferred dtype '{self._inferred_dtype}'."
                )
                # 如果不在允许的类型中，抛出类型错误异常
                raise TypeError(msg)
            return func(self, *args, **kwargs)

        # 设置装饰器函数的名称为原始函数的名称
        wrapper.__name__ = func_name
        return cast(F, wrapper)

    return _forbid_nonstring_types
    # 定义一个函数 _map_and_wrap，接受两个参数：name 和 docstring，都可以为 None
    def _map_and_wrap(name: str | None, docstring: str | None):
        # 使用 forbid_nonstring_types 装饰器，限制 wrapper 函数的参数类型不包括 bytes 类型
        @forbid_nonstring_types(["bytes"], name=name)
        # 定义内部函数 wrapper，它没有参数
        def wrapper(self):
            # 调用 self._data.array 的属性 getattr(self._data.array, f"_str_{name}")()，将结果赋给 result
            result = getattr(self._data.array, f"_str_{name}")()
            # 调用 self._wrap_result 方法，将 result 和返回字符串的条件返回作为参数
            return self._wrap_result(
                result, returns_string=name not in ("isnumeric", "isdecimal")
            )

        # 将 wrapper 函数的文档字符串设置为传入的 docstring
        wrapper.__doc__ = docstring
        # 返回 wrapper 函数对象
        return wrapper

    # 定义 StringMethods 类，继承自 NoNewAttributesMixin 类
    class StringMethods(NoNewAttributesMixin):
        """
        Vectorized string functions for Series and Index.

        NAs stay NA unless handled otherwise by a particular method.
        Patterned after Python's string methods, with some inspiration from
        R's stringr package.

        Parameters
        ----------
        data : Series or Index
            The content of the Series or Index.

        See Also
        --------
        Series.str : Vectorized string functions for Series.
        Index.str : Vectorized string functions for Index.

        Examples
        --------
        >>> s = pd.Series(["A_Str_Series"])
        >>> s
        0    A_Str_Series
        dtype: object

        >>> s.str.split("_")
        0    [A, Str, Series]
        dtype: object

        >>> s.str.replace("_", "")
        0    AStrSeries
        dtype: object
        """

        # Note: see the docstring in pandas.core.strings.__init__
        # for an explanation of the implementation.
        # TODO: Dispatch all the methods
        # Currently the following are not dispatched to the array
        # * cat
        # * extractall

        # 构造函数，接受一个参数 data
        def __init__(self, data) -> None:
            # 导入 StringDtype 类
            from pandas.core.arrays.string_ import StringDtype

            # 验证并设置 self._inferred_dtype
            self._inferred_dtype = self._validate(data)
            # 判断 data 是否是 CategoricalDtype 类型，并设置 self._is_categorical
            self._is_categorical = isinstance(data.dtype, CategoricalDtype)
            # 判断 data 是否是 StringDtype 类型，并设置 self._is_string
            self._is_string = isinstance(data.dtype, StringDtype)
            # 设置 self._data 为传入的 data
            self._data = data

            self._index = self._name = None
            # 如果 data 是 ABCSeries 的实例，设置 self._index 和 self._name
            if isinstance(data, ABCSeries):
                self._index = data.index
                self._name = data.name

            # 如果 self._is_categorical 为 True，self._parent 指向 data._values.categories；否则指向 data
            self._parent = data._values.categories if self._is_categorical else data
            # 将 data 保存到 self._orig，用于在需要时还原为正确的类型
            self._orig = data
            # 执行 _freeze 方法，冻结实例
            self._freeze()

        @staticmethod
    def _validate(data):
        """
        辅助函数用于 StringMethods，推断并检查数据的 dtype。

        这是在创建 StringMethods 对象时的第一道防线，只是检查 dtype 是否在所有以下字符串方法的允许类型的*联合*中；
        然后在每个方法基础上使用装饰器 @forbid_nonstring_types 进行细化（有关更多信息，请参阅相应的文档字符串）。

        这实际上应该排除所有包含非字符串值的 Series/Index，但由于性能原因，这在没有 str dtype 之前是不现实的（GH 9343 / 13877）。

        Parameters
        ----------
        data : Series 的内容

        Returns
        -------
        dtype : 推断出的 data 的 dtype
        """
        if isinstance(data, ABCMultiIndex):
            raise AttributeError(
                "Can only use .str accessor with Index, not MultiIndex"
            )

        # 查看 _libs/lib.pyx 获取推断出的类型列表
        allowed_types = ["string", "empty", "bytes", "mixed", "mixed-integer"]

        data = extract_array(data)

        values = getattr(data, "categories", data)  # categorical / normal

        inferred_dtype = lib.infer_dtype(values, skipna=True)

        if inferred_dtype not in allowed_types:
            raise AttributeError("Can only use .str accessor with string values!")
        return inferred_dtype

    def __getitem__(self, key):
        """
        通过调用 self._data.array._str_getitem(key) 获取结果，并将其包装返回。

        Parameters
        ----------
        key : 索引键

        Returns
        -------
        result : 处理后的结果
        """
        result = self._data.array._str_getitem(key)
        return self._wrap_result(result)

    def __iter__(self) -> Iterator:
        """
        抛出 TypeError 异常，指示该对象不可迭代。

        Returns
        -------
        TypeError 异常：'类型名' object is not iterable
        """
        raise TypeError(f"'{type(self).__name__}' object is not iterable")

    def _wrap_result(
        self,
        result,
        name=None,
        expand: bool | None = None,
        fill_value=np.nan,
        returns_string: bool = True,
        dtype=None,
        ):
        """
        包装处理结果并返回，支持各种参数来定制包装结果的行为。

        Parameters
        ----------
        result : 处理的结果
        name : 结果的名称
        expand : 是否扩展结果
        fill_value : 填充值，默认为 np.nan
        returns_string : 是否返回字符串，默认为 True
        dtype : 结果的 dtype

        Returns
        -------
        result : 包装后的结果
        """
    def _get_series_list(self, others):
        """
        Auxiliary function for :meth:`str.cat`. Turn potentially mixed input
        into a list of Series (elements without an index must match the length
        of the calling Series/Index).

        Parameters
        ----------
        others : Series, DataFrame, np.ndarray, list-like or list-like of
            Objects that are either Series, Index or np.ndarray (1-dim).

        Returns
        -------
        list of Series
            Others transformed into list of Series.
        """
        from pandas import (
            DataFrame,
            Series,
        )

        # self._orig is either Series or Index
        idx = self._orig if isinstance(self._orig, ABCIndex) else self._orig.index

        # Generally speaking, all objects without an index inherit the index
        # `idx` of the calling Series/Index - i.e. must have matching length.
        # Objects with an index (i.e. Series/Index/DataFrame) keep their own.
        if isinstance(others, ABCSeries):
            # Return `others` wrapped in a list if it is already a Series
            return [others]
        elif isinstance(others, ABCIndex):
            # Wrap `others` in a Series, inheriting the index `idx` of the calling object
            return [Series(others, index=idx, dtype=others.dtype)]
        elif isinstance(others, ABCDataFrame):
            # Convert `others` into a list of Series, using the DataFrame's columns as keys
            return [others[x] for x in others]
        elif isinstance(others, np.ndarray) and others.ndim == 2:
            # Convert 2D numpy array `others` into a DataFrame indexed by `idx`
            others = DataFrame(others, index=idx)
            # Return `others` as a list of Series, using the DataFrame's columns as keys
            return [others[x] for x in others]
        elif is_list_like(others, allow_sets=False):
            try:
                others = list(others)  # Ensure `others` is treated as a list
            except TypeError:
                # If `others` cannot be converted to a list, raise an exception
                pass
            else:
                # For list-like `others`, ensure all elements are compatible types
                if all(
                    isinstance(x, (ABCSeries, ABCIndex, ExtensionArray))
                    or (isinstance(x, np.ndarray) and x.ndim == 1)
                    for x in others
                ):
                    los: list[Series] = []
                    # Iterate through `others`, recursively processing each element
                    while others:
                        los = los + self._get_series_list(others.pop(0))
                    return los
                # If elements of `others` are just strings, convert `others` into a Series
                elif all(not is_list_like(x) for x in others):
                    return [Series(others, index=idx)]
        # Raise an error if `others` does not match any expected type
        raise TypeError(
            "others must be Series, Index, DataFrame, np.ndarray "
            "or list-like (either containing only strings or "
            "containing only objects of type Series/Index/"
            "np.ndarray[1-dim])"
        )

    @forbid_nonstring_types(["bytes", "mixed", "mixed-integer"])
    def cat(
        self,
        others=None,
        sep: str | None = None,
        na_rep=None,
        join: AlignJoin = "left",
    ):
        """
        Concatenate strings in the Series/Index with given separator `sep`.

        Parameters
        ----------
        others : None or list-like, default None
            Additional Series, Index, DataFrame, np.ndarray or list-like objects
            to concatenate with the calling Series/Index.
        sep : str, default None
            Separator/delimiter to use between strings.
        na_rep : str, default None
            Missing values representation.
        join : {'left', 'inner', 'outer'}, default 'left'
            How to handle indexes on other axis (or axes).

        Notes
        -----
        This function concatenates strings within the Series/Index, handling
        various input types and alignment issues.

        See Also
        --------
        Series.str.cat : Concatenate strings in a Series with given separator.
        """
        _shared_docs["str_split"] = r"""
        Split strings around given separator/delimiter.
        """
    # Splits the string in the Series/Index from the %(side)s,
    # at the specified delimiter string.
    
    Parameters
    ----------
    pat : str%(pat_regex)s, optional
        %(pat_description)s.
        If not specified, split on whitespace.
    n : int, default -1 (all)
        Limit number of splits in output.
        ``None``, 0 and -1 will be interpreted as return all splits.
    expand : bool, default False
        Expand the split strings into separate columns.
    
        - If ``True``, return DataFrame/MultiIndex expanding dimensionality.
        - If ``False``, return Series/Index, containing lists of strings.
    %(regex_argument)s
    
    Returns
    -------
    Series, Index, DataFrame or MultiIndex
        Type matches caller unless ``expand=True`` (see Notes).
    %(raises_split)s
    
    See Also
    --------
    Series.str.split : Split strings around given separator/delimiter.
    Series.str.rsplit : Splits string around given separator/delimiter,
        starting from the right.
    Series.str.join : Join lists contained as elements in the Series/Index
        with passed delimiter.
    str.split : Standard library version for split.
    str.rsplit : Standard library version for rsplit.
    
    Notes
    -----
    The handling of the `n` keyword depends on the number of found splits:
    
    - If found splits > `n`,  make first `n` splits only
    - If found splits <= `n`, make all splits
    - If for a certain row the number of found splits < `n`,
      append `None` for padding up to `n` if ``expand=True``
    
    If using ``expand=True``, Series and Index callers return DataFrame and
    MultiIndex objects, respectively.
    %(regex_pat_note)s
    
    Examples
    --------
    >>> s = pd.Series(
    ...     [
    ...         "this is a regular sentence",
    ...         "https://docs.python.org/3/tutorial/index.html",
    ...         np.nan
    ...     ]
    ... )
    >>> s
    0                       this is a regular sentence
    1    https://docs.python.org/3/tutorial/index.html
    2                                              NaN
    dtype: object
    
    In the default setting, the string is split by whitespace.
    
    >>> s.str.split()
    0                   [this, is, a, regular, sentence]
    1    [https://docs.python.org/3/tutorial/index.html]
    2                                                NaN
    dtype: object
    
    Without the `n` parameter, the outputs of `rsplit` and `split`
    are identical.
    
    >>> s.str.rsplit()
    0                   [this, is, a, regular, sentence]
    1    [https://docs.python.org/3/tutorial/index.html]
    2                                                NaN
    dtype: object
    
    The `n` parameter can be used to limit the number of splits on the
    delimiter. The outputs of `split` and `rsplit` are different.
    
    >>> s.str.split(n=2)
    0                     [this, is, a regular sentence]
    1    [https://docs.python.org/3/tutorial/index.html]
    2                                                NaN
    dtype: object



# 设置 dtype 为 object 类型
>>> s.str.rsplit(n=2)
# 使用 str 对象的 rsplit 方法进行右侧分割，n=2 表示分割的次数
0                     [this is a, regular, sentence]
1    [https://docs.python.org/3/tutorial/index.html]
2                                                NaN
dtype: object



# pat 参数可以用于指定其他字符进行分割
>>> s.str.split(pat="/")
# 使用 str 对象的 split 方法，通过 "/" 字符进行分割
0                         [this is a regular sentence]
1    [https:, , docs.python.org, 3, tutorial, index...
2                                                  NaN
dtype: object



# 当 expand=True 时，分割的元素会扩展到单独的列中。如果存在 NaN，则在分割过程中会传播至各列。
>>> s.str.split(expand=True)
# 使用 str 对象的 split 方法，并展开结果到多列
                                                   0     1     2        3         4
0                                           this    is     a  regular  sentence
1  https://docs.python.org/3/tutorial/index.html  None  None     None      None
2                                            NaN   NaN   NaN      NaN       NaN



# 对于稍微复杂的用例，比如从 URL 中分离 HTML 文档名称，可以结合不同的参数设置来使用。
>>> s.str.rsplit("/", n=1, expand=True)
# 使用 str 对象的 rsplit 方法，通过 "/" 字符进行右侧分割，n=1 表示只分割一次，并展开结果到多列
                                        0           1
0          this is a regular sentence        None
1  https://docs.python.org/3/tutorial  index.html
2                                 NaN         NaN
%(regex_examples)s"""



@Appender(
    _shared_docs["str_split"]
    % {
        "side": "beginning",
        "pat_regex": " or compiled regex",
        "pat_description": "String or regular expression to split on",
        "regex_argument": """
regex : bool, default None
    Determines if the passed-in pattern is a regular expression:

    - If ``True``, assumes the passed-in pattern is a regular expression
    - If ``False``, treats the pattern as a literal string.
    - If ``None`` and `pat` length is 1, treats `pat` as a literal string.
    - If ``None`` and `pat` length is not 1, treats `pat` as a regular expression.
    - Cannot be set to False if `pat` is a compiled regex

    .. versionadded:: 1.4.0
     """,
        "raises_split": """
                  Raises
                  ------
                  ValueError
                      * if `regex` is False and `pat` is a compiled regex
                  """,
        "regex_pat_note": """
Use of `regex =False` with a `pat` as a compiled regex will raise an error.
        """,
        "method": "split",
        "regex_examples": r"""
Remember to escape special characters when explicitly using regular expressions.

>>> s = pd.Series(["foo and bar plus baz"])
>>> s.str.split(r"and|plus", expand=True)
    0   1   2
0 foo bar baz

Regular expressions can be used to handle urls or file names.



# 添加文档信息到 _shared_docs 的 str_split 键处，包括方法、正则表达式的相关参数和例子。
@Appender(
    _shared_docs["str_split"]
    % {
        "side": "beginning",
        "pat_regex": " or compiled regex",
        "pat_description": "String or regular expression to split on",
        "regex_argument": """
regex : bool, default None
    Determines if the passed-in pattern is a regular expression:

    - If ``True``, assumes the passed-in pattern is a regular expression
    - If ``False``, treats the pattern as a literal string.
    - If ``None`` and `pat` length is 1, treats `pat` as a literal string.
    - If ``None`` and `pat` length is not 1, treats `pat` as a regular expression.
    - Cannot be set to False if `pat` is a compiled regex

    .. versionadded:: 1.4.0
     """,
        "raises_split": """
                  Raises
                  ------
                  ValueError
                      * if `regex` is False and `pat` is a compiled regex
                  """,
        "regex_pat_note": """
Use of `regex =False` with a `pat` as a compiled regex will raise an error.
        """,
        "method": "split",
        "regex_examples": r"""
Remember to escape special characters when explicitly using regular expressions.

>>> s = pd.Series(["foo and bar plus baz"])
>>> s.str.split(r"and|plus", expand=True)
    0   1   2
0 foo bar baz

Regular expressions can be used to handle urls or file names.
    _shared_docs["str_partition"] = """
    Split the string at the %(side)s occurrence of `sep`.

    This method splits the string at the %(side)s occurrence of `sep`,
    and returns 3 elements containing the part before the separator,
    the separator itself, and the part after the separator.
    If the separator is not found, return %(return)s.

    Parameters
    ----------
    sep : str, default whitespace
        String to split on.
    expand : bool, default True
        If True, return DataFrame/MultiIndex expanding dimensionality.
        If False, return Series/Index.

    Returns
    -------
    DataFrame/MultiIndex or Series/Index of objects

    See Also
    --------
    pandas.Series.str.rsplit : Split strings around given separator, starting from the end.
    """


@Appender(
    _shared_docs["str_split"]
    % {
        "side": "end",
        "pat_regex": "",
        "pat_description": "String to split on",
        "regex_argument": "",
        "raises_split": "",
        "regex_pat_note": "",
        "method": "rsplit",
        "regex_examples": "",
    }
)
@forbid_nonstring_types(["bytes"])
def rsplit(self, pat=None, *, n=-1, expand: bool = False):
    # 使用 _str_rsplit 方法对字符串数组进行拆分操作
    result = self._data.array._str_rsplit(pat, n=n)
    # 根据数据类型设置返回结果的数据类型
    dtype = object if self._data.dtype == object else None
    # 封装结果并返回
    return self._wrap_result(
        result, expand=expand, returns_string=expand, dtype=dtype
    )
    # 将字符串序列按指定分隔符分割成多个部分，返回一个包含分割结果的 DataFrame
    @Appender(
        _shared_docs["str_partition"]
        % {
            "side": "first",
            "return": "3 elements containing the string itself, followed by two "
            "empty strings",
            "also": "rpartition : Split the string at the last occurrence of `sep`.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def partition(self, sep: str = " ", expand: bool = True):
        # 调用底层的字符串数组方法来执行分割操作
        result = self._data.array._str_partition(sep, expand)
        # 根据数据类型确定返回的数据类型
        if self._data.dtype == "category":
            dtype = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        # 包装结果并返回
        return self._wrap_result(
            result, expand=expand, returns_string=expand, dtype=dtype
        )
    
    # 将字符串序列按指定分隔符从右向左分割成多个部分，返回一个包含分割结果的 DataFrame
    @Appender(
        _shared_docs["str_partition"]
        % {
            "side": "last",
            "return": "3 elements containing two empty strings, followed by the "
            "string itself",
            "also": "partition : Split the string at the first occurrence of `sep`.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def rpartition(self, sep: str = " ", expand: bool = True):
        # 调用底层的字符串数组方法来执行从右向左的分割操作
        result = self._data.array._str_rpartition(sep, expand)
        # 根据数据类型确定返回的数据类型
        if self._data.dtype == "category":
            dtype = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        # 包装结果并返回
        return self._wrap_result(
            result, expand=expand, returns_string=expand, dtype=dtype
        )
    # 定义一个方法 `get`，用于从每个组件中的指定位置或具有指定键提取元素。
    def get(self, i):
        """
        Extract element from each component at specified position or with specified key.

        Extract element from lists, tuples, dict, or strings in each element in the
        Series/Index.

        Parameters
        ----------
        i : int or hashable dict label
            Position or key of element to extract.

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> s = pd.Series(
        ...     [
        ...         "String",
        ...         (1, 2, 3),
        ...         ["a", "b", "c"],
        ...         123,
        ...         -456,
        ...         {1: "Hello", "2": "World"},
        ...     ]
        ... )
        >>> s
        0                        String
        1                     (1, 2, 3)
        2                     [a, b, c]
        3                           123
        4                          -456
        5    {1: 'Hello', '2': 'World'}
        dtype: object

        >>> s.str.get(1)
        0        t
        1        2
        2        b
        3      NaN
        4      NaN
        5    Hello
        dtype: object

        >>> s.str.get(-1)
        0      g
        1      3
        2      c
        3    NaN
        4    NaN
        5    None
        dtype: object

        Return element with given key

        >>> s = pd.Series(
        ...     [
        ...         {"name": "Hello", "value": "World"},
        ...         {"name": "Goodbye", "value": "Planet"},
        ...     ]
        ... )
        >>> s.str.get("name")
        0      Hello
        1    Goodbye
        dtype: object
        """
        # 调用内部 `_str_get` 方法从数据数组中提取元素
        result = self._data.array._str_get(i)
        # 将结果封装并返回
        return self._wrap_result(result)

    # 禁止非字符串类型为参数的装饰器
    @forbid_nonstring_types(["bytes"])
    def join(self, sep: str):
        """
        Join lists contained as elements in the Series/Index with passed delimiter.

        If the elements of a Series are lists themselves, join the content of these
        lists using the delimiter passed to the function.
        This function is an equivalent to :meth:`str.join`.

        Parameters
        ----------
        sep : str
            Delimiter to use between list entries.

        Returns
        -------
        Series/Index: object
            The list entries concatenated by intervening occurrences of the
            delimiter.

        Raises
        ------
        AttributeError
            If the supplied Series contains neither strings nor lists.

        See Also
        --------
        str.join : Standard library version of this method.
        Series.str.split : Split strings around given separator/delimiter.

        Notes
        -----
        If any of the list items is not a string object, the result of the join
        will be `NaN`.

        Examples
        --------
        Example with a list that contains non-string elements.

        >>> s = pd.Series(
        ...     [
        ...         ["lion", "elephant", "zebra"],
        ...         [1.1, 2.2, 3.3],
        ...         ["cat", np.nan, "dog"],
        ...         ["cow", 4.5, "goat"],
        ...         ["duck", ["swan", "fish"], "guppy"],
        ...     ]
        ... )
        >>> s
        0        [lion, elephant, zebra]
        1                [1.1, 2.2, 3.3]
        2                [cat, nan, dog]
        3               [cow, 4.5, goat]
        4    [duck, [swan, fish], guppy]
        dtype: object

        Join all lists using a '-'. The lists containing object(s) of types other
        than str will produce a NaN.

        >>> s.str.join("-")
        0    lion-elephant-zebra
        1                    NaN
        2                    NaN
        3                    NaN
        4                    NaN
        dtype: object
        """
        # 调用内部方法 _str_join 对 Series/Index 中的列表进行连接操作
        result = self._data.array._str_join(sep)
        # 将处理后的结果封装成相应的对象并返回
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def contains(
        self, pat, case: bool = True, flags: int = 0, na=None, regex: bool = True
    ):
        """
        Check whether each string element contains the specified pattern.

        This function is used to test if the specified pattern or regex is
        contained within each string in the Series/Index.

        Parameters
        ----------
        pat : str or list-like
            Character sequence or regular expression.
        case : bool, default True
            If True, case sensitive.
        flags : int, default 0 (no flags)
            Regex flags, e.g., re.IGNORECASE.
        na : object, default None
            Fill value for NA or None for existing NA values.
        regex : bool, default True
            If True, assumes the pat is a regular expression.

        Returns
        -------
        Series/Index: bool or object
            Series/Index of boolean values indicating if each string contains
            the specified pattern.

        Raises
        ------
        TypeError
            If `pat` is not a string or compiled regex.

        See Also
        --------
        Series.str.match : Determine if each string starts with a match of a
            regular expression.
        Series.str.contains : Test if pattern or regex is contained within a
            string of a Series/Index.
        Series.str.extract : Extract capture groups in the regex `pat` as columns
            in DataFrame.

        Examples
        --------
        Example with a Series of strings.

        >>> s = pd.Series(["A", "B", "C", "Aaba", "Baca", np.nan, "CABA", "dog", "cat"])
        >>> s.str.contains("A")
        0     True
        1    False
        2    False
        3     True
        4    False
        5    False
        6     True
        7    False
        8    False
        dtype: bool

        Example with a Series of strings, using regex.

        >>> s.str.contains("A", case=False)
        0     True
        1    False
        2    False
        3     True
        4     True
        5    False
        6     True
        7    False
        8    False
        dtype: bool
        """
        # 根据传入参数调用内部方法进行字符串匹配或正则匹配操作
        return self._data.forbid_nonstring_types(["bytes"])
    @forbid_nonstring_types(["bytes"])
    def replace(
        self,
        pat: str | re.Pattern | dict,
        repl: str | Callable | None = None,
        n: int = -1,
        case: bool | None = None,
        flags: int = 0,
        regex: bool = False,
    ):
        """
        Replace occurrences of pattern/regex in the Series/Index with some other
        string or the result of callable.

        Parameters
        ----------
        pat : str | re.Pattern | dict
            Character sequence, regular expression pattern, or dictionary mapping patterns to replacements.
        repl : str | Callable | None, optional
            Replacement string or callable. If None, the value in `pat` will be used.
        n : int, default -1
            Maximum number of replacements. -1 (default) means all replacements.
        case : bool | None, optional
            If True, case sensitive. If None, default to the pandas setting.
        flags : int, default 0 (no flags)
            Regex module flags, e.g. re.IGNORECASE.
        regex : bool, default False
            If True, treat `pat` as a regex pattern.

        Returns
        -------
        Series/Index with replaced values.

        See Also
        --------
        str.replace : String method for element-wise replacement.

        Examples
        --------
        >>> ser = pd.Series(["apple", "banana", "cherry"])
        >>> ser.str.replace("a", "X")
        0    Xpple
        1    bXnXnX
        2    cherry
        dtype: object
        """
        result = self._data.array._str_replace(
            pat, repl=repl, n=n, case=case, flags=flags, regex=regex
        )
        return self._wrap_result(result, returns_string=True)
    # 应用装饰器，限制重复操作的输入类型不包括字节串(bytes)
    @forbid_nonstring_types(["bytes"])
    # 定义字符串重复操作的方法，作用于 Series 或 Index 对象
    def repeat(self, repeats):
        """
        Duplicate each string in the Series or Index.

        Parameters
        ----------
        repeats : int or sequence of int
            Same value for all (int) or different value per (sequence).

        Returns
        -------
        Series or pandas.Index
            Series or Index of repeated string objects specified by
            input parameter repeats.

        Examples
        --------
        >>> s = pd.Series(["a", "b", "c"])
        >>> s
        0    a
        1    b
        2    c
        dtype: object

        Single int repeats string in Series

        >>> s.str.repeat(repeats=2)
        0    aa
        1    bb
        2    cc
        dtype: object

        Sequence of int repeats corresponding string in Series

        >>> s.str.repeat(repeats=[1, 2, 3])
        0      a
        1     bb
        2    ccc
        dtype: object
        """
        # 调用底层数据结构的字符串重复方法，返回重复操作后的结果
        result = self._data.array._str_repeat(repeats)
        # 封装结果并返回，以适应 Series 或 Index 对象
        return self._wrap_result(result)

    # 应用装饰器，限制填充操作的输入类型不包括字节串(bytes)
    @forbid_nonstring_types(["bytes"])
    # 定义字符串填充操作的方法，作用于 Series 或 Index 对象
    def pad(
        self,
        width: int,
        side: Literal["left", "right", "both"] = "left",
        fillchar: str = " ",
    ):
        """
        Pad strings in the Series/Index up to width.

        Parameters
        ----------
        width : int
            Minimum width of resulting string; additional characters will be filled
            with character defined in `fillchar`.
        side : {'left', 'right', 'both'}, default 'left'
            Side from which to fill resulting string.
        fillchar : str, default ' '
            Additional character for filling, default is whitespace.

        Returns
        -------
        Series or Index of object
            Returns Series or Index with minimum number of char in object.

        See Also
        --------
        Series.str.rjust : Fills the left side of strings with an arbitrary
            character. Equivalent to ``Series.str.pad(side='left')``.
        Series.str.ljust : Fills the right side of strings with an arbitrary
            character. Equivalent to ``Series.str.pad(side='right')``.
        Series.str.center : Fills both sides of strings with an arbitrary
            character. Equivalent to ``Series.str.pad(side='both')``.
        Series.str.zfill : Pad strings in the Series/Index by prepending '0'
            character. Equivalent to ``Series.str.pad(side='left', fillchar='0')``.

        Examples
        --------
        >>> s = pd.Series(["caribou", "tiger"])
        >>> s
        0    caribou
        1      tiger
        dtype: object

        >>> s.str.pad(width=10)
        0       caribou
        1         tiger
        dtype: object

        >>> s.str.pad(width=10, side="right", fillchar="-")
        0    caribou---
        1    tiger-----
        dtype: object

        >>> s.str.pad(width=10, side="both", fillchar="-")
        0    -caribou--
        1    --tiger---
        dtype: object
        """
        # 检查 fillchar 是否为字符串类型，若不是则抛出类型错误异常
        if not isinstance(fillchar, str):
            msg = f"fillchar must be a character, not {type(fillchar).__name__}"
            raise TypeError(msg)

        # 检查 fillchar 是否为单个字符，若不是则抛出类型错误异常
        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")

        # 检查 width 是否为整数类型，若不是则抛出类型错误异常
        if not is_integer(width):
            msg = f"width must be of integer type, not {type(width).__name__}"
            raise TypeError(msg)

        # 调用内部方法进行字符串填充操作，返回填充后的结果
        result = self._data.array._str_pad(width, side=side, fillchar=fillchar)
        # 包装结果并返回
        return self._wrap_result(result)

    _shared_docs["str_pad"] = """
    Pad %(side)s side of strings in the Series/Index.

    Equivalent to :meth:`str.%(method)s`.

    Parameters
    ----------
    width : int
        Minimum width of resulting string; additional characters will be filled
        with ``fillchar``.
    fillchar : str
        Additional character for filling, default is whitespace.

    Returns
    -------
    Series/Index of objects.

    Examples
    --------
    For Series.str.center:

    >>> ser = pd.Series(['dog', 'bird', 'mouse'])
    >>> ser.str.center(8, fillchar='.')
    0   ..dog...
    1   ..bird..
    2   .mouse..
    dtype: object

    For Series.str.ljust:
    """
    # 创建一个包含字符串的 Pandas Series 对象，内容为 ['dog', 'bird', 'mouse']
    ser = pd.Series(['dog', 'bird', 'mouse'])
    # 对 Series 中的每个字符串进行左对齐操作，使其长度为 8，不足部分用 '.' 填充
    ser.str.ljust(8, fillchar='.')
    """
    0   dog.....
    1   bird....
    2   mouse...
    dtype: object
    """

    # 创建一个包含字符串的 Pandas Series 对象，内容为 ['dog', 'bird', 'mouse']
    ser = pd.Series(['dog', 'bird', 'mouse'])
    # 对 Series 中的每个字符串进行右对齐操作，使其长度为 8，不足部分用 '.' 填充
    ser.str.rjust(8, fillchar='.')
    """
    0   .....dog
    1   ....bird
    2   ...mouse
    dtype: object
    """

    @Appender(_shared_docs["str_pad"] % {"side": "left and right", "method": "center"})
    @forbid_nonstring_types(["bytes"])
    # 定义一个用于字符串居中操作的方法 center，调用 pad 方法进行填充，填充字符由 fillchar 参数指定
    def center(self, width: int, fillchar: str = " "):
        return self.pad(width, side="both", fillchar=fillchar)

    @Appender(_shared_docs["str_pad"] % {"side": "right", "method": "ljust"})
    @forbid_nonstring_types(["bytes"])
    # 定义一个用于字符串右对齐操作的方法 ljust，调用 pad 方法进行填充，填充字符由 fillchar 参数指定
    def ljust(self, width: int, fillchar: str = " "):
        return self.pad(width, side="right", fillchar=fillchar)

    @Appender(_shared_docs["str_pad"] % {"side": "left", "method": "rjust"})
    @forbid_nonstring_types(["bytes"])
    # 定义一个用于字符串左对齐操作的方法 rjust，调用 pad 方法进行填充，填充字符由 fillchar 参数指定
    def rjust(self, width: int, fillchar: str = " "):
        return self.pad(width, side="left", fillchar=fillchar)

    @forbid_nonstring_types(["bytes"])
    # 装饰器，用于禁止非字符串类型的参数（比如 bytes 类型）调用相关方法
    def zfill(self, width: int):
        """
        Pad strings in the Series/Index by prepending '0' characters.

        Strings in the Series/Index are padded with '0' characters on the
        left of the string to reach a total string length  `width`. Strings
        in the Series/Index with length greater or equal to `width` are
        unchanged.

        Parameters
        ----------
        width : int
            Minimum length of resulting string; strings with length less
            than `width` will be prepended with '0' characters.

        Returns
        -------
        Series/Index of objects.

        See Also
        --------
        Series.str.rjust : Fills the left side of strings with an arbitrary
            character.
        Series.str.ljust : Fills the right side of strings with an arbitrary
            character.
        Series.str.pad : Fills the specified sides of strings with an arbitrary
            character.
        Series.str.center : Fills both sides of strings with an arbitrary
            character.

        Notes
        -----
        Differs from :meth:`str.zfill` which has special handling
        for '+'/'-' in the string.

        Examples
        --------
        >>> s = pd.Series(["-1", "1", "1000", 10, np.nan])
        >>> s
        0      -1
        1       1
        2    1000
        3      10
        4     NaN
        dtype: object

        Note that ``10`` and ``NaN`` are not strings, therefore they are
        converted to ``NaN``. The minus sign in ``'-1'`` is treated as a
        special character and the zero is added to the right of it
        (:meth:`str.zfill` would have moved it to the left). ``1000``
        remains unchanged as it is longer than `width`.

        >>> s.str.zfill(3)
        0     -01
        1     001
        2    1000
        3     NaN
        4     NaN
        dtype: object
        """
        # 检查 `width` 是否为整数类型，若不是则抛出类型错误异常
        if not is_integer(width):
            msg = f"width must be of integer type, not {type(width).__name__}"
            raise TypeError(msg)
        # 定义 lambda 函数 `f`，用于对每个元素进行 zfill 操作
        f = lambda x: x.zfill(width)
        # 调用 `_str_map` 方法将 `f` 应用到 Series/Index 的每个元素上
        result = self._data.array._str_map(f)
        # 将处理后的结果封装成新的 Series/Index 对象并返回
        return self._wrap_result(result)
    def slice(self, start=None, stop=None, step=None):
        """
        Slice substrings from each element in the Series or Index.

        Parameters
        ----------
        start : int, optional
            开始切片的位置。
        stop : int, optional
            结束切片的位置。
        step : int, optional
            切片步长。

        Returns
        -------
        Series or Index of object
            返回切片后的 Series 或 Index 对象，包含原始字符串对象的切片子串。

        See Also
        --------
        Series.str.slice_replace : 替换切片位置的字符串。
        Series.str.get : 返回指定位置的元素。
            相当于 `Series.str.slice(start=i, stop=i+1)`，其中 `i`
            是位置索引。

        Examples
        --------
        >>> s = pd.Series(["koala", "dog", "chameleon"])
        >>> s
        0        koala
        1          dog
        2    chameleon
        dtype: object

        >>> s.str.slice(start=1)
        0        oala
        1          og
        2    hameleon
        dtype: object

        >>> s.str.slice(start=-1)
        0           a
        1           g
        2           n
        dtype: object

        >>> s.str.slice(stop=2)
        0    ko
        1    do
        2    ch
        dtype: object

        >>> s.str.slice(step=2)
        0      kaa
        1       dg
        2    caeen
        dtype: object

        >>> s.str.slice(start=0, stop=5, step=3)
        0    kl
        1     d
        2    cm
        dtype: object

        Equivalent behaviour to:

        >>> s.str[0:5:3]
        0    kl
        1     d
        2    cm
        dtype: object
        """
        # 调用底层方法对字符串进行切片操作
        result = self._data.array._str_slice(start, stop, step)
        # 将切片结果封装成相应的 Series 或 Index 对象并返回
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def slice_replace(self, start=None, stop=None, repl=None):
        """
        Replace a positional slice of a string with another value.

        Parameters
        ----------
        start : int, optional
            Left index position to use for the slice. If not specified (None),
            the slice is unbounded on the left, i.e. slice from the start
            of the string.
        stop : int, optional
            Right index position to use for the slice. If not specified (None),
            the slice is unbounded on the right, i.e. slice until the
            end of the string.
        repl : str, optional
            String for replacement. If not specified (None), the sliced region
            is replaced with an empty string.

        Returns
        -------
        Series or Index
            Same type as the original object.

        See Also
        --------
        Series.str.slice : Just slicing without replacement.

        Examples
        --------
        >>> s = pd.Series(["a", "ab", "abc", "abdc", "abcde"])
        >>> s
        0        a
        1       ab
        2      abc
        3     abdc
        4    abcde
        dtype: object

        Specify just `start`, meaning replace `start` until the end of the
        string with `repl`.

        >>> s.str.slice_replace(1, repl="X")
        0    aX
        1    aX
        2    aX
        3    aX
        4    aX
        dtype: object

        Specify just `stop`, meaning the start of the string to `stop` is replaced
        with `repl`, and the rest of the string is included.

        >>> s.str.slice_replace(stop=2, repl="X")
        0       X
        1       X
        2      Xc
        3     Xdc
        4    Xcde
        dtype: object

        Specify `start` and `stop`, meaning the slice from `start` to `stop` is
        replaced with `repl`. Everything before or after `start` and `stop` is
        included as is.

        >>> s.str.slice_replace(start=1, stop=3, repl="X")
        0      aX
        1      aX
        2      aX
        3     aXc
        4    aXde
        dtype: object
        """
        # 调用底层方法 _str_slice_replace 执行字符串的切片替换操作
        result = self._data.array._str_slice_replace(start, stop, repl)
        # 将替换后的结果封装成相同类型的 Series 或 Index 对象并返回
        return self._wrap_result(result)
    def decode(self, encoding, errors: str = "strict"):
        """
        Decode character string in the Series/Index using indicated encoding.

        Equivalent to :meth:`str.decode` in python2 and :meth:`bytes.decode` in
        python3.

        Parameters
        ----------
        encoding : str
            The encoding to be used for decoding the strings.
        errors : str, optional
            Specifies how errors are to be handled during decoding.

        Returns
        -------
        Series or Index
            A new Series or Index with decoded strings.

        Examples
        --------
        For Series:

        >>> ser = pd.Series([b"cow", b"123", b"()"])
        >>> ser.str.decode("ascii")
        0   cow
        1   123
        2   ()
        dtype: object
        """
        # TODO: Add a similar _bytes interface.
        
        # Check if the specified encoding is optimized for CPython
        if encoding in _cpython_optimized_decoders:
            # Use CPython's optimized decoding function
            f = lambda x: x.decode(encoding, errors)
        else:
            # Use codecs module to get a decoder function for the encoding
            decoder = codecs.getdecoder(encoding)
            f = lambda x: decoder(x, errors)[0]
        
        # Obtain the underlying array from the Series/Index
        arr = self._data.array
        
        # Map the decoding function 'f' to each string in the array
        result = arr._str_map(f)
        
        # Wrap the result into a new Series/Index and return
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def encode(self, encoding, errors: str = "strict"):
        """
        Encode character string in the Series/Index using indicated encoding.

        Equivalent to :meth:`str.encode`.

        Parameters
        ----------
        encoding : str
            The encoding to be used for encoding the strings.
        errors : str, optional
            Specifies how errors are to be handled during encoding.

        Returns
        -------
        Series/Index of objects
            A new Series or Index containing encoded byte objects.

        Examples
        --------
        >>> ser = pd.Series(["cow", "123", "()"])
        >>> ser.str.encode(encoding="ascii")
        0     b'cow'
        1     b'123'
        2      b'()'
        dtype: object
        """
        # Encode strings in the Series/Index using specified encoding and errors
        result = self._data.array._str_encode(encoding, errors)
        
        # Wrap the result into a new Series/Index and return
        return self._wrap_result(result, returns_string=False)

    _shared_docs["str_strip"] = r"""
    Remove %(position)s characters.

    Strip whitespaces (including newlines) or a set of specified characters
    from each string in the Series/Index from %(side)s.
    Replaces any non-strings in Series with NaNs.
    Equivalent to :meth:`str.%(method)s`.

    Parameters
    ----------
    to_strip : str or None, default None
        Specifying the set of characters to be removed.
        All combinations of this set of characters will be stripped.
        If None then whitespaces are removed.

    Returns
    -------
    Series or Index of object
        A new Series or Index with stripped strings.

    See Also
    --------
    Series.str.strip : Remove leading and trailing characters in Series/Index.
    Series.str.lstrip : Remove leading characters in Series/Index.
    Series.str.rstrip : Remove trailing characters in Series/Index.

    Examples
    --------
    >>> s = pd.Series(['1. Ant.  ', '2. Bee!\n', '3. Cat?\t', np.nan, 10, True])
    >>> s
    0    1. Ant.
    1    2. Bee!\n
    2    3. Cat?\t
    3          NaN
    4           10
    5         True
    dtype: object

    >>> s.str.strip()
    0    1. Ant.
    1    2. Bee!
    2    3. Cat?
    dtype: object
    """
    @Appender(
        _shared_docs["str_removefix"] % {"side": "suffix", "other_side": "prefix"}
    )
    @forbid_nonstring_types(["bytes"])
    def removesuffix(self, suffix: str):
        # 调用数据数组的内部方法来移除字符串的后缀
        result = self._data.array._str_removesuffix(suffix)
        # 将处理后的结果封装成新的对象并返回
        return self._wrap_result(result)
    # 定义一个方法用于从字符串末尾移除指定的后缀
    def removesuffix(self, suffix: str):
        # 调用底层数据结构的方法来移除后缀
        result = self._data.array._str_removesuffix(suffix)
        # 封装处理结果并返回
        return self._wrap_result(result)

    # 应用装饰器，禁止传入非字符串类型为参数，如 bytes
    @forbid_nonstring_types(["bytes"])
    # 定义一个方法用于将字符串列包装成指定宽度的文本块
    def wrap(
        self,
        width: int,
        expand_tabs: bool = True,
        tabsize: int = 8,
        replace_whitespace: bool = True,
        drop_whitespace: bool = True,
        initial_indent: str = "",
        subsequent_indent: str = "",
        fix_sentence_endings: bool = False,
        break_long_words: bool = True,
        break_on_hyphens: bool = True,
        max_lines: int | None = None,
        placeholder: str = " [...]",
    ):
        """
        Wrap each string in the Series to fit within the specified width.

        Parameters
        ----------
        width : int
            Maximum width of each line.
        expand_tabs : bool, default True
            Expand tab characters to spaces before processing.
        tabsize : int, default 8
            Number of spaces per tab character.
        replace_whitespace : bool, default True
            Replace each whitespace character with a single space.
        drop_whitespace : bool, default True
            Drop leading and trailing whitespace from each line.
        initial_indent : str, default ""
            String that will be prepended to the first line.
        subsequent_indent : str, default ""
            String that will be prepended to all lines after the first.
        fix_sentence_endings : bool, default False
            Ensure sentence-ending punctuation is followed by two spaces.
        break_long_words : bool, default True
            Break long words at the end of lines.
        break_on_hyphens : bool, default True
            Break lines at hyphens.
        max_lines : int or None, default None
            Maximum number of lines to wrap to; None means unlimited.
        placeholder : str, default " [...]"
            Placeholder for truncated text.

        Returns
        -------
        None
        """
        # 底层调用字符串数组的方法来进行文本包装
        result = self._data.array._str_wrap(
            width,
            expand_tabs=expand_tabs,
            tabsize=tabsize,
            replace_whitespace=replace_whitespace,
            drop_whitespace=drop_whitespace,
            initial_indent=initial_indent,
            subsequent_indent=subsequent_indent,
            fix_sentence_endings=fix_sentence_endings,
            break_long_words=break_long_words,
            break_on_hyphens=break_on_hyphens,
            max_lines=max_lines,
            placeholder=placeholder,
        )
        # 封装处理结果并返回
        return self._wrap_result(result)

    # 应用装饰器，禁止传入非字符串类型为参数，如 bytes
    @forbid_nonstring_types(["bytes"])
    # 定义一个方法用于生成 Series 中每个字符串值的虚拟变量 DataFrame
    def get_dummies(self, sep: str = "|"):
        """
        Return DataFrame of dummy/indicator variables for Series.

        Each string in Series is split by sep and returned as a DataFrame
        of dummy/indicator variables.

        Parameters
        ----------
        sep : str, default "|"
            String to split on.

        Returns
        -------
        DataFrame
            Dummy variables corresponding to values of the Series.

        See Also
        --------
        get_dummies : Convert categorical variable into dummy/indicator
            variables.

        Examples
        --------
        >>> pd.Series(["a|b", "a", "a|c"]).str.get_dummies()
           a  b  c
        0  1  1  0
        1  1  0  0
        2  1  0  1

        >>> pd.Series(["a|b", np.nan, "a|c"]).str.get_dummies()
           a  b  c
        0  1  1  0
        1  0  0  0
        2  1  0  1
        """
        # 调用底层字符串数组的方法来生成虚拟变量 DataFrame
        result, name = self._data.array._str_get_dummies(sep)
        # 封装处理结果并返回
        return self._wrap_result(
            result,
            name=name,
            expand=True,
            returns_string=False,
        )

    # 应用装饰器，禁止传入非字符串类型为参数，如 bytes
    @forbid_nonstring_types(["bytes"])
    def translate(self, table):
        """
        Map all characters in the string through the given mapping table.

        This method is equivalent to the standard :meth:`str.translate`
        method for strings. It maps each character in the string to a new
        character according to the translation table provided. Unmapped
        characters are left unchanged, while characters mapped to None
        are removed.

        Parameters
        ----------
        table : dict
            Table is a mapping of Unicode ordinals to Unicode ordinals, strings, or
            None. Unmapped characters are left untouched.
            Characters mapped to None are deleted. :meth:`str.maketrans` is a
            helper function for making translation tables.

        Returns
        -------
        Series or Index
            A new Series or Index with translated strings.

        See Also
        --------
        Series.str.replace : Replace occurrences of pattern/regex in the
            Series with some other string.
        Index.str.replace : Replace occurrences of pattern/regex in the
            Index with some other string.

        Examples
        --------
        >>> ser = pd.Series(["El niño", "Françoise"])
        >>> mytable = str.maketrans({"ñ": "n", "ç": "c"})
        >>> ser.str.translate(mytable)
        0   El nino
        1   Francoise
        dtype: object
        """
        # 调用底层字符串数据的 _str_translate 方法进行字符映射操作
        result = self._data.array._str_translate(table)
        # 根据原始数据的数据类型确定结果的数据类型
        dtype = object if self._data.dtype == "object" else None
        # 使用内部方法将结果封装成新的 Series 或 Index 对象并返回
        return self._wrap_result(result, dtype=dtype)

    @forbid_nonstring_types(["bytes"])
    def count(self, pat, flags: int = 0):
        r"""
        Count occurrences of pattern in each string of the Series/Index.

        This function is used to count the number of times a particular regex
        pattern is repeated in each of the string elements of the
        :class:`~pandas.Series`.

        Parameters
        ----------
        pat : str
            Valid regular expression.
        flags : int, default 0, meaning no flags
            Flags for the `re` module. For a complete list, `see here
            <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.

        Returns
        -------
        Series or Index
            Same type as the calling object containing the integer counts.

        See Also
        --------
        re : Standard library module for regular expressions.
        str.count : Standard library version, without regular expression support.

        Notes
        -----
        Some characters need to be escaped when passing in `pat`.
        eg. ``'$'`` has a special meaning in regex and must be escaped when
        finding this literal character.

        Examples
        --------
        >>> s = pd.Series(["A", "B", "Aaba", "Baca", np.nan, "CABA", "cat"])
        >>> s.str.count("a")
        0    0.0
        1    0.0
        2    2.0
        3    2.0
        4    NaN
        5    0.0
        6    1.0
        dtype: float64

        Escape ``'$'`` to find the literal dollar sign.

        >>> s = pd.Series(["$", "B", "Aab$", "$$ca", "C$B$", "cat"])
        >>> s.str.count("\\$")
        0    1
        1    0
        2    1
        3    2
        4    2
        5    0
        dtype: int64

        This is also available on Index

        >>> pd.Index(["A", "A", "Aaba", "cat"]).str.count("a")
        Index([0, 0, 2, 1], dtype='int64')
        """
        # 调用底层字符串数组对象的 `_str_count` 方法来执行实际的计数操作
        result = self._data.array._str_count(pat, flags)
        # 包装计数结果并返回，确保返回的对象类型与调用对象相同
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def startswith(
        self, pat: str | tuple[str, ...], na: Scalar | None = None
    ) -> Series | Index:
        """
        Test if the start of each string element matches a pattern.

        Equivalent to :meth:`str.startswith`.

        Parameters
        ----------
        pat : str or tuple[str, ...]
            Character sequence or tuple of strings. Regular expressions are not
            accepted.
        na : object, default NaN
            Object shown if element tested is not a string. The default depends
            on dtype of the array. For object-dtype, ``numpy.nan`` is used.
            For ``StringDtype``, ``pandas.NA`` is used.

        Returns
        -------
        Series or Index of bool
            A Series of booleans indicating whether the given pattern matches
            the start of each string element.

        See Also
        --------
        str.startswith : Python standard library string method.
        Series.str.endswith : Same as startswith, but tests the end of string.
        Series.str.contains : Tests if string element contains a pattern.

        Examples
        --------
        >>> s = pd.Series(["bat", "Bear", "cat", np.nan])
        >>> s
        0     bat
        1    Bear
        2     cat
        3     NaN
        dtype: object

        >>> s.str.startswith("b")
        0     True
        1    False
        2    False
        3      NaN
        dtype: object

        >>> s.str.startswith(("b", "B"))
        0     True
        1     True
        2    False
        3      NaN
        dtype: object

        Specifying `na` to be `False` instead of `NaN`.

        >>> s.str.startswith("b", na=False)
        0     True
        1    False
        2    False
        3    False
        dtype: bool
        """
        # 检查参数 `pat` 是否为字符串或字符串元组，否则抛出类型错误
        if not isinstance(pat, (str, tuple)):
            msg = f"expected a string or tuple, not {type(pat).__name__}"
            raise TypeError(msg)
        # 调用内部方法 `_str_startswith` 进行字符串起始匹配，返回布尔值数组
        result = self._data.array._str_startswith(pat, na=na)
        # 调用 `_wrap_result` 方法包装结果，返回 Series 或 Index
        return self._wrap_result(result, returns_string=False)
    ) -> Series | Index:
        """
        Test if the end of each string element matches a pattern.

        Equivalent to :meth:`str.endswith`.

        Parameters
        ----------
        pat : str or tuple[str, ...]
            Character sequence or tuple of strings. Regular expressions are not
            accepted.
        na : object, default NaN
            Object shown if element tested is not a string. The default depends
            on dtype of the array. For object-dtype, ``numpy.nan`` is used.
            For ``StringDtype``, ``pandas.NA`` is used.

        Returns
        -------
        Series or Index of bool
            A Series of booleans indicating whether the given pattern matches
            the end of each string element.

        See Also
        --------
        str.endswith : Python standard library string method.
        Series.str.startswith : Same as endswith, but tests the start of string.
        Series.str.contains : Tests if string element contains a pattern.

        Examples
        --------
        >>> s = pd.Series(["bat", "bear", "caT", np.nan])
        >>> s
        0     bat
        1    bear
        2     caT
        3     NaN
        dtype: object

        >>> s.str.endswith("t")
        0     True
        1    False
        2    False
        3      NaN
        dtype: object

        >>> s.str.endswith(("t", "T"))
        0     True
        1    False
        2     True
        3      NaN
        dtype: object

        Specifying `na` to be `False` instead of `NaN`.

        >>> s.str.endswith("t", na=False)
        0     True
        1    False
        2    False
        3    False
        dtype: bool
        """
        # 检查 pat 是否为 str 或 tuple 类型，否则抛出 TypeError 异常
        if not isinstance(pat, (str, tuple)):
            msg = f"expected a string or tuple, not {type(pat).__name__}"
            raise TypeError(msg)
        # 调用内部方法 _str_endswith 进行实际的匹配操作，返回匹配结果
        result = self._data.array._str_endswith(pat, na=na)
        # 包装结果并返回，确保返回值类型为 Series 或 Index
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def findall(self, pat, flags: int = 0):
        """
        Find all occurrences of pattern or regular expression in the Series/Index.

        Equivalent to applying :func:`re.findall` to all the elements in the
        Series/Index.

        Parameters
        ----------
        pat : str
            Pattern or regular expression. 模式或正则表达式。
        flags : int, default 0
            Flags from ``re`` module, e.g. `re.IGNORECASE` (default is 0, which
            means no flags). 来自 ``re`` 模块的标志，例如 `re.IGNORECASE`（默认为0，表示没有标志）。

        Returns
        -------
        Series/Index of lists of strings
            All non-overlapping matches of pattern or regular expression in each
            string of this Series/Index. 返回包含字符串列表的 Series/Index，
            包含每个字符串中模式或正则表达式的所有非重叠匹配。

        See Also
        --------
        count : Count occurrences of pattern or regular expression in each string
            of the Series/Index.
        extractall : For each string in the Series, extract groups from all matches
            of regular expression and return a DataFrame with one row for each
            match and one column for each group.
        re.findall : The equivalent ``re`` function to all non-overlapping matches
            of pattern or regular expression in string, as a list of strings. 
            re.findall 函数等效于字符串中模式或正则表达式的所有非重叠匹配，返回字符串列表。

        Examples
        --------
        >>> s = pd.Series(["Lion", "Monkey", "Rabbit"])

        The search for the pattern 'Monkey' returns one match:

        >>> s.str.findall("Monkey")
        0          []
        1    [Monkey]
        2          []
        dtype: object

        On the other hand, the search for the pattern 'MONKEY' doesn't return any
        match:

        >>> s.str.findall("MONKEY")
        0    []
        1    []
        2    []
        dtype: object

        Flags can be added to the pattern or regular expression. For instance,
        to find the pattern 'MONKEY' ignoring the case:

        >>> import re
        >>> s.str.findall("MONKEY", flags=re.IGNORECASE)
        0          []
        1    [Monkey]
        2          []
        dtype: object

        When the pattern matches more than one string in the Series, all matches
        are returned:

        >>> s.str.findall("on")
        0    [on]
        1    [on]
        2      []
        dtype: object

        Regular expressions are supported too. For instance, the search for all the
        strings ending with the word 'on' is shown next:

        >>> s.str.findall("on$")
        0    [on]
        1      []
        2      []
        dtype: object

        If the pattern is found more than once in the same string, then a list of
        multiple strings is returned:

        >>> s.str.findall("b")
        0        []
        1        []
        2    [b, b]
        dtype: object
        """
        # 使用 self._data.array._str_findall 方法在 Series/Index 中查找所有模式或正则表达式的匹配项
        result = self._data.array._str_findall(pat, flags)
        # 将结果包装成 Series/Index，并指定 returns_string=False 表示返回的是列表而不是字符串
        return self._wrap_result(result, returns_string=False)
    # 定义一个方法，用于从 Series 中提取正则表达式 `pat` 中的捕获组，并以 DataFrame 的形式返回结果
    def extractall(self, pat, flags: int = 0) -> DataFrame:
        """
        从 DataFrame 中的每个字符串中提取正则表达式 `pat` 的捕获组作为列。

        对于 Series 中的每个主题字符串，从正则表达式 `pat` 的所有匹配中提取组。当 Series 中的每个主题字符串恰好有一个匹配时，
        extractall(pat).xs(0, level='match') 与 extract(pat) 是相同的。

        Parameters
        ----------
        pat : str
            带有捕获组的正则表达式模式。
        flags : int, 默认 0 (无标志)
            一个 ``re`` 模块标志，例如 ``re.IGNORECASE``。这些标志允许修改正则表达式匹配的行为，例如大小写、空格等。
            可以使用按位 OR 运算符组合多个标志，例如 ``re.IGNORECASE | re.MULTILINE``。

        Returns
        -------
        DataFrame
            一个 ``DataFrame``，每个匹配都有一行，每个捕获组都有一列。其行具有来自主题 ``Series`` 的第一级 ``MultiIndex``。
            最后一个级别命名为 'match'，用于索引每个 ``Series`` 项中的匹配项。任何正则表达式 `pat` 中的捕获组名称将用作列名；
            否则将使用捕获组编号。

        See Also
        --------
        extract : 仅返回第一个匹配项（而非所有匹配项）。

        Examples
        --------
        具有一个组的模式将返回一个具有一列的 DataFrame。
        没有匹配的索引不会出现在结果中。

        >>> s = pd.Series(["a1a2", "b1", "c1"], index=["A", "B", "C"])
        >>> s.str.extractall(r"[ab](\d)")
                0
        match
        A 0      1
          1      2
        B 0      1

        捕获组名称用作结果的列名。

        >>> s.str.extractall(r"[ab](?P<digit>\d)")
                digit
        match
        A 0         1
          1         2
        B 0         1

        具有两个组的模式将返回一个具有两列的 DataFrame。

        >>> s.str.extractall(r"(?P<letter>[ab])(?P<digit>\d)")
                letter digit
        match
        A 0          a     1
          1          a     2
        B 0          b     1

        不匹配的可选组在结果中为 NaN。

        >>> s.str.extractall(r"(?P<letter>[ab])?(?P<digit>\d)")
                letter digit
        match
        A 0          a     1
          1          a     2
        B 0          b     1
        C 0        NaN     1
        """
        # TODO: dispatch
        # 调用实际的提取函数 str_extractall，并返回其结果
        return str_extractall(self._orig, pat, flags)
    @Appender(
        _shared_docs["find"]
        % {
            "side": "lowest",
            "method": "find",
            "also": "rfind : Return highest indexes in each strings.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def find(self, sub, start: int = 0, end=None):
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)

        # 使用字符串的内部方法 _str_find 查找子字符串的最低索引
        result = self._data.array._str_find(sub, start, end)
        # 包装结果并返回，不返回字符串
        return self._wrap_result(result, returns_string=False)

    @Appender(
        _shared_docs["find"]
        % {
            "side": "highest",
            "method": "rfind",
            "also": "find : Return lowest indexes in each strings.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def rfind(self, sub, start: int = 0, end=None):
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)

        # 使用字符串的内部方法 _str_rfind 查找子字符串的最高索引
        result = self._data.array._str_rfind(sub, start=start, end=end)
        # 包装结果并返回，不返回字符串
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def normalize(self, form):
        """
        Return the Unicode normal form for the strings in the Series/Index.

        For more information on the forms, see the
        :func:`unicodedata.normalize`.

        Parameters
        ----------
        form : {'NFC', 'NFKC', 'NFD', 'NFKD'}
            Unicode form.

        Returns
        -------
        Series/Index of objects

        Examples
        --------
        >>> ser = pd.Series(["ñ"])
        >>> ser.str.normalize("NFC") == ser.str.normalize("NFD")
        0   False
        dtype: bool
        """
        # 使用字符串的内部方法 _str_normalize 对 Series/Index 中的字符串进行 Unicode 规范化
        result = self._data.array._str_normalize(form)
        # 包装结果并返回
        return self._wrap_result(result)

    _shared_docs["index"] = """
    Return %(side)s indexes in each string in Series/Index.

    Each of the returned indexes corresponds to the position where the
    substring is fully contained between [start:end]. This is the same
    as ``str.%(similar)s`` except instead of returning -1, it raises a
    ValueError when the substring is not found. Equivalent to standard
    ``str.%(method)s``.

    Parameters
    ----------
    sub : str
        Substring being searched.

    start : int
        Left edge index.
    
    end : int
        Right edge index.

    Returns
    -------
    Series or Index of int.

    See Also
    --------
    %(also)s

    Examples
    --------
    For Series.str.find:

    >>> ser = pd.Series(["cow_", "duck_", "do_ve"])
    >>> ser.str.find("_")
    0   3
    1   4
    2   2
    dtype: int64

    For Series.str.rfind:

    >>> ser = pd.Series(["_cow_", "duck_", "do_v_e"])
    >>> ser.str.rfind("_")
    0   4
    1   4
    2   4
    dtype: int64
    """
    @Appender(
        _shared_docs["index"]
        % {
            "side": "lowest",
            "similar": "find",
            "method": "index",
            "also": "rindex : Return highest indexes in each strings.",
        }
    )
    # 将关于索引操作的文档添加到当前方法的文档中，包括"lowest"（最低索引）、"find"（查找）、"index"（索引）、"rindex"（反向索引）的相关信息
    @forbid_nonstring_types(["bytes"])
    # 禁止非字符串类型的参数，如果参数不是字符串，则抛出类型错误异常
    def index(self, sub, start: int = 0, end=None):
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)
        # 如果 sub 不是字符串类型，抛出类型错误异常，提示期望的参数类型

        result = self._data.array._str_index(sub, start=start, end=end)
        # 调用内部数组对象的字符串索引方法，返回结果
        return self._wrap_result(result, returns_string=False)
        # 将结果进行封装并返回，确保结果不是字符串类型

    @Appender(
        _shared_docs["index"]
        % {
            "side": "highest",
            "similar": "rfind",
            "method": "rindex",
            "also": "index : Return lowest indexes in each strings.",
        }
    )
    # 将关于反向索引操作的文档添加到当前方法的文档中，包括"highest"（最高索引）、"rfind"（反向查找）、"rindex"（反向索引）、"index"（索引）的相关信息
    @forbid_nonstring_types(["bytes"])
    # 禁止非字符串类型的参数，如果参数不是字符串，则抛出类型错误异常
    def rindex(self, sub, start: int = 0, end=None):
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)
        # 如果 sub 不是字符串类型，抛出类型错误异常，提示期望的参数类型

        result = self._data.array._str_rindex(sub, start=start, end=end)
        # 调用内部数组对象的字符串反向索引方法，返回结果
        return self._wrap_result(result, returns_string=False)
        # 将结果进行封装并返回，确保结果不是字符串类型
    def len(self):
        """
        Compute the length of each element in the Series/Index.

        The element may be a sequence (such as a string, tuple or list) or a collection
        (such as a dictionary).

        Returns
        -------
        Series or Index of int
            A Series or Index of integer values indicating the length of each
            element in the Series or Index.

        See Also
        --------
        str.len : Python built-in function returning the length of an object.
        Series.size : Returns the length of the Series.

        Examples
        --------
        Returns the length (number of characters) in a string. Returns the
        number of entries for dictionaries, lists or tuples.

        >>> s = pd.Series(
        ...     ["dog", "", 5, {"foo": "bar"}, [2, 3, 5, 7], ("one", "two", "three")]
        ... )
        >>> s
        0                  dog
        1
        2                    5
        3       {'foo': 'bar'}
        4         [2, 3, 5, 7]
        5    (one, two, three)
        dtype: object
        >>> s.str.len()
        0    3.0
        1    0.0
        2    NaN
        3    1.0
        4    4.0
        5    3.0
        dtype: float64
        """
        # 调用底层数据的字符串长度计算方法
        result = self._data.array._str_len()
        # 将计算结果包装并返回，禁止将结果作为字符串返回
        return self._wrap_result(result, returns_string=False)

    _shared_docs["casemethods"] = """
    Convert strings in the Series/Index to %(type)s.
    %(version)s
    Equivalent to :meth:`str.%(method)s`.

    Returns
    -------
    Series or Index of object

    See Also
    --------
    Series.str.lower : Converts all characters to lowercase.
    Series.str.upper : Converts all characters to uppercase.
    Series.str.title : Converts first character of each word to uppercase and
        remaining to lowercase.
    Series.str.capitalize : Converts first character to uppercase and
        remaining to lowercase.
    Series.str.swapcase : Converts uppercase to lowercase and lowercase to
        uppercase.
    Series.str.casefold: Removes all case distinctions in the string.

    Examples
    --------
    >>> s = pd.Series(['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe'])
    >>> s
    0                 lower
    1              CAPITALS
    2    this is a sentence
    3              SwApCaSe
    dtype: object

    >>> s.str.lower()
    0                 lower
    1              capitals
    2    this is a sentence
    3              swapcase
    dtype: object

    >>> s.str.upper()
    0                 LOWER
    1              CAPITALS
    2    THIS IS A SENTENCE
    3              SWAPCASE
    dtype: object

    >>> s.str.title()
    0                 Lower
    1              Capitals
    2    This Is A Sentence
    3              Swapcase
    dtype: object

    >>> s.str.capitalize()
    0                 Lower
    1              Capitals
    2    This is a sentence
    3              Swapcase
    dtype: object

    >>> s.str.swapcase()
    0                 LOWER
    # Types:
    #   cases:
    #       upper, lower, title, capitalize, swapcase, casefold
    #   boolean:
    #     isalpha, isnumeric isalnum isdigit isdecimal isspace islower isupper istitle
    # _doc_args holds dict of strings to use in substituting casemethod docs
    # _doc_args 是一个字典，用于存储各种大小写操作的文档参数
    _doc_args: dict[str, dict[str, str]] = {}
    
    # 设置 lower 方法的文档参数
    _doc_args["lower"] = {"type": "lowercase", "method": "lower", "version": ""}
    # 设置 upper 方法的文档参数
    _doc_args["upper"] = {"type": "uppercase", "method": "upper", "version": ""}
    # 设置 title 方法的文档参数
    _doc_args["title"] = {"type": "titlecase", "method": "title", "version": ""}
    # 设置 capitalize 方法的文档参数
    _doc_args["capitalize"] = {
        "type": "be capitalized",
        "method": "capitalize",
        "version": "",
    }
    # 设置 swapcase 方法的文档参数
    _doc_args["swapcase"] = {
        "type": "be swapcased",
        "method": "swapcase",
        "version": "",
    }
    # 设置 casefold 方法的文档参数
    _doc_args["casefold"] = {
        "type": "be casefolded",
        "method": "casefold",
        "version": "",
    }
    
    @Appender(_shared_docs["casemethods"] % _doc_args["lower"])
    @forbid_nonstring_types(["bytes"])
    def lower(self):
        # 调用数据的字符串小写化方法
        result = self._data.array._str_lower()
        return self._wrap_result(result)
    
    @Appender(_shared_docs["casemethods"] % _doc_args["upper"])
    @forbid_nonstring_types(["bytes"])
    def upper(self):
        # 调用数据的字符串大写化方法
        result = self._data.array._str_upper()
        return self._wrap_result(result)
    
    @Appender(_shared_docs["casemethods"] % _doc_args["title"])
    @forbid_nonstring_types(["bytes"])
    def title(self):
        # 调用数据的字符串标题化方法
        result = self._data.array._str_title()
        return self._wrap_result(result)
    
    @Appender(_shared_docs["casemethods"] % _doc_args["capitalize"])
    @forbid_nonstring_types(["bytes"])
    def capitalize(self):
        # 调用数据的字符串首字母大写方法
        result = self._data.array._str_capitalize()
        return self._wrap_result(result)
    
    @Appender(_shared_docs["casemethods"] % _doc_args["swapcase"])
    @forbid_nonstring_types(["bytes"])
    def swapcase(self):
        # 调用数据的字符串大小写互换方法
        result = self._data.array._str_swapcase()
        return self._wrap_result(result)
    
    @Appender(_shared_docs["casemethods"] % _doc_args["casefold"])
    @forbid_nonstring_types(["bytes"])
    def casefold(self):
        # 调用数据的字符串大小写折叠方法
        result = self._data.array._str_casefold()
        return self._wrap_result(result)
    
    _shared_docs["ismethods"] = """
    Check whether all characters in each string are %(type)s.
    
    This is equivalent to running the Python string method
    :meth:`str.%(method)s` for each element of the Series/Index. If a string
    has zero characters, ``False`` is returned for that check.
    
    Returns
    -------
    Series or Index of bool
        Series or Index of boolean values with the same length as the original
        Series/Index.
    
    See Also
    --------
    Series.str.isalpha : Check whether all characters are alphabetic.
    Series.str.isnumeric : Check whether all characters are numeric.
    """
    # Series.str.isalnum : 检查字符串中的所有字符是否都是字母数字。
    # Series.str.isdigit : 检查字符串中的所有字符是否都是数字。
    # Series.str.isdecimal : 检查字符串中的所有字符是否都是十进制数字。
    # Series.str.isspace : 检查字符串中的所有字符是否都是空白字符。
    # Series.str.islower : 检查字符串中的所有字符是否都是小写字母。
    # Series.str.isupper : 检查字符串中的所有字符是否都是大写字母。
    # Series.str.istitle : 检查字符串中的所有字符是否都是标题形式。

    Examples
    --------
    **检查字母和数字字符**

    >>> s1 = pd.Series(['one', 'one1', '1', ''])

    >>> s1.str.isalpha()
    0     True
    1    False
    2    False
    3    False
    dtype: bool

    >>> s1.str.isnumeric()
    0    False
    1    False
    2     True
    3    False
    dtype: bool

    >>> s1.str.isalnum()
    0     True
    1     True
    2     True
    3    False
    dtype: bool

    注意，包含任何额外标点符号或空格的字符将导致字母数字检查结果为假。

    >>> s2 = pd.Series(['A B', '1.5', '3,000'])
    >>> s2.str.isalnum()
    0    False
    1    False
    2    False
    dtype: bool

    **更详细的数字字符检查**

    有几组不同但有重叠的数字字符集可供检查。

    >>> s3 = pd.Series(['23', '³', '⅕', ''])

    ``s3.str.isdecimal`` 方法检查十进制数字字符。

    >>> s3.str.isdecimal()
    0     True
    1    False
    2    False
    3    False
    dtype: bool

    ``s.str.isdigit`` 方法与 ``s3.str.isdecimal`` 相同，但还包括特殊的数字，如Unicode中的上标和下标数字。

    >>> s3.str.isdigit()
    0     True
    1     True
    2    False
    3    False
    dtype: bool

    ``s.str.isnumeric`` 方法与 ``s3.str.isdigit`` 相同，但还包括其他可表示数量的字符，如Unicode分数。

    >>> s3.str.isnumeric()
    0     True
    1     True
    2     True
    3    False
    dtype: bool

    **检查空白字符**

    >>> s4 = pd.Series([' ', '\t\r\n ', ''])
    >>> s4.str.isspace()
    0     True
    1     True
    2    False
    dtype: bool

    **检查字符大小写**

    >>> s5 = pd.Series(['leopard', 'Golden Eagle', 'SNAKE', ''])

    >>> s5.str.islower()
    0     True
    1    False
    2    False
    3    False
    dtype: bool

    >>> s5.str.isupper()
    0    False
    1    False
    2     True
    3    False
    dtype: bool

    ``s5.str.istitle`` 方法检查所有单词是否都是标题形式（每个单词的首字母大写，其余小写）。
    单词被假设为由任何非数字字符序列组成，这些字符由空白字符分隔。

    >>> s5.str.istitle()
    0    False
    1     True
    2    False
    3    False
    dtype: bool
    # 将 "isalnum" 方法的描述信息添加到 _doc_args 字典中，定义其类型和方法
    _doc_args["isalnum"] = {"type": "alphanumeric", "method": "isalnum"}
    # 将 "isalpha" 方法的描述信息添加到 _doc_args 字典中，定义其类型和方法
    _doc_args["isalpha"] = {"type": "alphabetic", "method": "isalpha"}
    # 将 "isdigit" 方法的描述信息添加到 _doc_args 字典中，定义其类型和方法
    _doc_args["isdigit"] = {"type": "digits", "method": "isdigit"}
    # 将 "isspace" 方法的描述信息添加到 _doc_args 字典中，定义其类型和方法
    _doc_args["isspace"] = {"type": "whitespace", "method": "isspace"}
    # 将 "islower" 方法的描述信息添加到 _doc_args 字典中，定义其类型和方法
    _doc_args["islower"] = {"type": "lowercase", "method": "islower"}
    # 将 "isupper" 方法的描述信息添加到 _doc_args 字典中，定义其类型和方法
    _doc_args["isupper"] = {"type": "uppercase", "method": "isupper"}
    # 将 "istitle" 方法的描述信息添加到 _doc_args 字典中，定义其类型和方法
    _doc_args["istitle"] = {"type": "titlecase", "method": "istitle"}
    # 将 "isnumeric" 方法的描述信息添加到 _doc_args 字典中，定义其类型和方法
    _doc_args["isnumeric"] = {"type": "numeric", "method": "isnumeric"}
    # 将 "isdecimal" 方法的描述信息添加到 _doc_args 字典中，定义其类型和方法
    _doc_args["isdecimal"] = {"type": "decimal", "method": "isdecimal"}
    
    # 使用 _shared_docs 中 "ismethods" 键对应的文档模板，填充 isalnum 方法的文档
    isalnum = _map_and_wrap(
        "isalnum", docstring=_shared_docs["ismethods"] % _doc_args["isalnum"]
    )
    # 使用 _shared_docs 中 "ismethods" 键对应的文档模板，填充 isalpha 方法的文档
    isalpha = _map_and_wrap(
        "isalpha", docstring=_shared_docs["ismethods"] % _doc_args["isalpha"]
    )
    # 使用 _shared_docs 中 "ismethods" 键对应的文档模板，填充 isdigit 方法的文档
    isdigit = _map_and_wrap(
        "isdigit", docstring=_shared_docs["ismethods"] % _doc_args["isdigit"]
    )
    # 使用 _shared_docs 中 "ismethods" 键对应的文档模板，填充 isspace 方法的文档
    isspace = _map_and_wrap(
        "isspace", docstring=_shared_docs["ismethods"] % _doc_args["isspace"]
    )
    # 使用 _shared_docs 中 "ismethods" 键对应的文档模板，填充 islower 方法的文档
    islower = _map_and_wrap(
        "islower", docstring=_shared_docs["ismethods"] % _doc_args["islower"]
    )
    # 使用 _shared_docs 中 "ismethods" 键对应的文档模板，填充 isupper 方法的文档
    isupper = _map_and_wrap(
        "isupper", docstring=_shared_docs["ismethods"] % _doc_args["isupper"]
    )
    # 使用 _shared_docs 中 "ismethods" 键对应的文档模板，填充 istitle 方法的文档
    istitle = _map_and_wrap(
        "istitle", docstring=_shared_docs["ismethods"] % _doc_args["istitle"]
    )
    # 使用 _shared_docs 中 "ismethods" 键对应的文档模板，填充 isnumeric 方法的文档
    isnumeric = _map_and_wrap(
        "isnumeric", docstring=_shared_docs["ismethods"] % _doc_args["isnumeric"]
    )
    # 使用 _shared_docs 中 "ismethods" 键对应的文档模板，填充 isdecimal 方法的文档
    isdecimal = _map_and_wrap(
        "isdecimal", docstring=_shared_docs["ismethods"] % _doc_args["isdecimal"]
    )
def cat_safe(list_of_columns: list[npt.NDArray[np.object_]], sep: str):
    """
    Auxiliary function for :meth:`str.cat`.

    Same signature as cat_core, but handles TypeErrors in concatenation, which
    happen if the arrays in list_of columns have the wrong dtypes or content.

    Parameters
    ----------
    list_of_columns : list of numpy arrays
        List of arrays to be concatenated with sep;
        these arrays may not contain NaNs!
    sep : string
        The separator string for concatenating the columns.

    Returns
    -------
    nd.array
        The concatenation of list_of_columns with sep.
    """
    try:
        # Attempt to concatenate the list of columns using cat_core function
        result = cat_core(list_of_columns, sep)
    except TypeError:
        # Handle TypeError which may occur due to non-string values
        for column in list_of_columns:
            # Infer the dtype of the column to identify offending values
            dtype = lib.infer_dtype(column, skipna=True)
            if dtype not in ["string", "empty"]:
                # Raise a more specific TypeError with diagnostic message
                raise TypeError(
                    "Concatenation requires list-likes containing only "
                    "strings (or missing values). Offending values found in "
                    f"column {dtype}"
                ) from None
    # Return the concatenated result
    return result


def cat_core(list_of_columns: list, sep: str):
    """
    Auxiliary function for :meth:`str.cat`

    Parameters
    ----------
    list_of_columns : list of numpy arrays
        List of arrays to be concatenated with sep;
        these arrays may not contain NaNs!
    sep : string
        The separator string for concatenating the columns.

    Returns
    -------
    nd.array
        The concatenation of list_of_columns with sep.
    """
    if sep == "":
        # If sep is an empty string, concatenate directly without interleaving
        arr_of_cols = np.asarray(list_of_columns, dtype=object)
        return np.sum(arr_of_cols, axis=0)
    
    # Prepare a list where sep interleaves the columns
    list_with_sep = [sep] * (2 * len(list_of_columns) - 1)
    list_with_sep[::2] = list_of_columns
    arr_with_sep = np.asarray(list_with_sep, dtype=object)
    
    # Sum along the first axis to concatenate the arrays
    return np.sum(arr_with_sep, axis=0)


def _result_dtype(arr):
    # workaround #27953
    # ideally we just pass `dtype=arr.dtype` unconditionally, but this fails
    # when the list of values is empty.
    
    # Check the dtype of arr and return appropriate dtype
    from pandas.core.arrays.string_ import StringDtype
    if isinstance(arr.dtype, (ArrowDtype, StringDtype)):
        return arr.dtype
    return object


def _get_single_group_name(regex: re.Pattern) -> Hashable:
    if regex.groupindex:
        # Return the first named group from regex's groupindex
        return next(iter(regex.groupindex))
    else:
        # Return None if no named groups are found
        return None


def _get_group_names(regex: re.Pattern) -> list[Hashable] | range:
    """
    Get named groups from compiled regex.

    Unnamed groups are numbered.

    Parameters
    ----------
    regex : compiled regex

    Returns
    -------
    list of column labels
    """
    # Create a range of numbers up to regex's groups count
    rng = range(regex.groups)
    # Create a dictionary mapping groupindex values to their keys
    names = {v: k for k, v in regex.groupindex.items()}
    if not names:
        # If no named groups exist, return the range of numbered groups
        return rng
    # 通过列表推导式生成一个结果列表，对于每个 i 在 rng 中，从 names 中获取索引为 1+i 的值，若不存在则使用 i 本身
    result: list[Hashable] = [names.get(1 + i, i) for i in rng]
    # 将结果列表 result 转换为 NumPy 数组
    arr = np.array(result)
    # 检查数组 arr 的数据类型是否为整数类型，并且检查是否符合作为范围索引器的条件
    if arr.dtype.kind == "i" and lib.is_range_indexer(arr, len(arr)):
        # 如果符合条件，则返回原始的 rng 列表
        return rng
    # 否则返回生成的 result 列表
    return result
def str_extractall(arr, pat, flags: int = 0) -> DataFrame:
    # 编译正则表达式，使用给定的标志
    regex = re.compile(pat, flags=flags)
    # 正则表达式必须包含捕获组
    if regex.groups == 0:
        raise ValueError("pattern contains no capture groups")

    # 如果 arr 是索引的实例，则转换为 Series，并重新设置索引
    if isinstance(arr, ABCIndex):
        arr = arr.to_series().reset_index(drop=True).astype(arr.dtype)

    # 获取正则表达式中的组名列表
    columns = _get_group_names(regex)
    # 匹配结果列表
    match_list = []
    # 索引列表
    index_list = []
    # 检查 arr 是否为多级索引
    is_mi = arr.index.nlevels > 1

    # 遍历 arr 的每个条目
    for subject_key, subject in arr.items():
        # 如果条目是字符串类型
        if isinstance(subject, str):
            # 如果不是多级索引，则将 subject_key 转为元组
            if not is_mi:
                subject_key = (subject_key,)
            
            # 遍历正则表达式在字符串中找到的所有匹配项
            for match_i, match_tuple in enumerate(regex.findall(subject)):
                # 如果 match_tuple 是字符串，则转为元组
                if isinstance(match_tuple, str):
                    match_tuple = (match_tuple,)
                # 将空字符串转换为 NaN，构成匹配结果的元组
                na_tuple = [np.nan if group == "" else group for group in match_tuple]
                match_list.append(na_tuple)
                # 构建结果的索引键
                result_key = tuple(subject_key + (match_i,))
                index_list.append(result_key)

    # 导入 MultiIndex 类
    from pandas import MultiIndex

    # 使用 index_list 创建 MultiIndex 对象，索引名由 arr.index.names 和 "match" 组成
    index = MultiIndex.from_tuples(index_list, names=arr.index.names + ["match"])
    # 获取结果的数据类型
    dtype = _result_dtype(arr)

    # 使用 match_list、index 和 columns 构建结果 DataFrame
    result = arr._constructor_expanddim(
        match_list, index=index, columns=columns, dtype=dtype
    )
    # 返回结果
    return result
```