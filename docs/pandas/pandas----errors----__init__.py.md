# `D:\src\scipysrc\pandas\pandas\errors\__init__.py`

```
"""
Expose public exceptions & warnings
"""

# 导入必要的模块和类
from __future__ import annotations  # 导入未来版本的注释特性

import ctypes  # 导入 ctypes 库

from pandas._config.config import OptionError  # 导入 OptionError 异常类

from pandas._libs.tslibs import (
    OutOfBoundsDatetime,  # 导入 OutOfBoundsDatetime 异常类
    OutOfBoundsTimedelta,  # 导入 OutOfBoundsTimedelta 异常类
)

from pandas.util.version import InvalidVersion  # 导入 InvalidVersion 异常类


class IntCastingNaNError(ValueError):
    """
    Exception raised when converting (``astype``) an array with NaN to an integer type.

    Examples
    --------
    >>> pd.DataFrame(np.array([[1, np.nan], [2, 3]]), dtype="i8")
    Traceback (most recent call last):
    IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer
    """


class NullFrequencyError(ValueError):
    """
    Exception raised when a ``freq`` cannot be null.

    Particularly ``DatetimeIndex.shift``, ``TimedeltaIndex.shift``,
    ``PeriodIndex.shift``.

    Examples
    --------
    >>> df = pd.DatetimeIndex(["2011-01-01 10:00", "2011-01-01"], freq=None)
    >>> df.shift(2)
    Traceback (most recent call last):
    NullFrequencyError: Cannot shift with no freq
    """


class PerformanceWarning(Warning):
    """
    Warning raised when there is a possible performance impact.

    Examples
    --------
    >>> df = pd.DataFrame(
    ...     {"jim": [0, 0, 1, 1], "joe": ["x", "x", "z", "y"], "jolie": [1, 2, 3, 4]}
    ... )
    >>> df = df.set_index(["jim", "joe"])
    >>> df
              jolie
    jim  joe
    0    x    1
         x    2
    1    z    3
         y    4
    >>> df.loc[(1, "z")]  # doctest: +SKIP
    # PerformanceWarning: indexing past lexsort depth may impact performance.
    df.loc[(1, 'z')]
              jolie
    jim  joe
    1    z        3
    """


class UnsupportedFunctionCall(ValueError):
    """
    Exception raised when attempting to call an unsupported numpy function.

    For example, ``np.cumsum(groupby_object)``.

    Examples
    --------
    >>> df = pd.DataFrame(
    ...     {"A": [0, 0, 1, 1], "B": ["x", "x", "z", "y"], "C": [1, 2, 3, 4]}
    ... )
    >>> np.cumsum(df.groupby(["A"]))
    Traceback (most recent call last):
    UnsupportedFunctionCall: numpy operations are not valid with groupby.
    Use .groupby(...).cumsum() instead
    """


class UnsortedIndexError(KeyError):
    """
    Error raised when slicing a MultiIndex which has not been lexsorted.

    Subclass of `KeyError`.

    Examples
    --------
    >>> df = pd.DataFrame(
    ...     {
    ...         "cat": [0, 0, 1, 1],
    ...         "color": ["white", "white", "brown", "black"],
    ...         "lives": [4, 4, 3, 7],
    ...     },
    ... )
    >>> df = df.set_index(["cat", "color"])
    >>> df
                lives
    cat  color
    0    white    4
         white    4
    1    brown    3
         black    7
    >>> df.loc[(0, "black") : (1, "white")]
    Traceback (most recent call last):
    UnsortedIndexError: 'Key length (2) was greater
    than MultiIndex lexsort depth (1)'
    """


class ParserError(ValueError):
    """
    Placeholder for ParserError, intended for future use.
    Currently, it doesn't have any additional functionality or examples.
    """
    Exception that is raised by an error encountered in parsing file contents.

    This is a generic error raised for errors encountered when functions like
    `read_csv` or `read_html` are parsing contents of a file.

    See Also
    --------
    read_csv : Read CSV (comma-separated) file into a DataFrame.
    read_html : Read HTML table into a DataFrame.

    Examples
    --------
    >>> data = '''a,b,c
    ... cat,foo,bar
    ... dog,foo,"baz'''
    >>> from io import StringIO
    >>> pd.read_csv(StringIO(data), skipfooter=1, engine="python")
    Traceback (most recent call last):
    ParserError: ',' expected after '"'. Error could possibly be due
    to parsing errors in the skipped footer rows
    """
# 文件内容解析时遇到错误时引发的异常
# 这是一个通用错误，当像 `read_csv` 或 `read_html` 这样的函数解析文件内容时遇到错误时会引发它
# 参见
# --------
# read_csv：将CSV（逗号分隔）文件读取到DataFrame中。
# read_html：将HTML表格读取到DataFrame中。
#
# 示例
# --------
# >>> data = '''a,b,c
# ... cat,foo,bar
# ... dog,foo,"baz'''
# >>> from io import StringIO
# >>> pd.read_csv(StringIO(data), skipfooter=1, engine="python")
# Traceback (most recent call last):
# ParserError: ',' expected after '"'. Error could possibly be due
# to parsing errors in the skipped footer rows
class DtypeWarning(Warning):
    """
    当从文件中读取到不同的数据类型时引发的警告。

    如果 `read_csv` 或 `read_table` 遇到给定 CSV 文件中某一列的数据类型不一致时，会引发此警告。

    参见
    --------
    read_csv : 将 CSV（逗号分隔）文件读取到 DataFrame 中。
    read_table : 将通用分隔文件读取到 DataFrame 中。

    注意
    -----
    当处理较大的文件时会发出此警告，因为数据类型检查是在每个数据块读取时进行的。

    尽管有此警告，CSV 文件仍会以单列中包含混合类型（将作为对象类型处理）的方式被读取。请参阅下面的示例以更好地理解此问题。

    示例
    --------
    以下示例创建并读取一个包含 `int` 和 `str` 的大型 CSV 文件的列。

    >>> df = pd.DataFrame(
    ...     {
    ...         "a": (["1"] * 100000 + ["X"] * 100000 + ["1"] * 100000),
    ...         "b": ["b"] * 300000,
    ...     }
    ... )  # doctest: +SKIP
    >>> df.to_csv("test.csv", index=False)  # doctest: +SKIP
    >>> df2 = pd.read_csv("test.csv")  # doctest: +SKIP
    ... # DtypeWarning: Columns (0: a) have mixed types

    需要注意的是，``df2`` 将包含相同输入 '1' 的 `str` 和 `int`。

    >>> df2.iloc[262140, 0]  # doctest: +SKIP
    '1'
    >>> type(df2.iloc[262140, 0])  # doctest: +SKIP
    <class 'str'>
    >>> df2.iloc[262150, 0]  # doctest: +SKIP
    1
    >>> type(df2.iloc[262150, 0])  # doctest: +SKIP
    <class 'int'>

    解决此问题的一种方法是在 `read_csv` 和 `read_table` 函数中使用 `dtype` 参数显式地指定转换：

    >>> df2 = pd.read_csv("test.csv", sep=",", dtype={"a": str})  # doctest: +SKIP

    这样就不会发出警告了。
    """


class EmptyDataError(ValueError):
    """
    当 `pd.read_csv` 遇到空数据或空标题时引发的异常。

    示例
    --------
    >>> from io import StringIO
    >>> empty = StringIO()
    >>> pd.read_csv(empty)
    Traceback (most recent call last):
    EmptyDataError: No columns to parse from file
    """


class ParserWarning(Warning):
    """
    当读取文件时使用了非默认的 'c' 解析器时引发的警告。

    在需要从默认 'c' 解析器更改为 'python' 解析器时，由 `pd.read_csv` 和 `pd.read_table` 引发。

    这种情况发生是因为请求的引擎无法支持或处理 CSV 文件的特定属性。

    目前，不支持 'c' 解析器的选项包括以下参数：

    1. `sep` 不是单个字符（例如正则表达式分隔符）
    2. `skipfooter` 大于 0

    可以通过在 `pd.read_csv` 和 `pd.read_table` 方法中添加 `engine='python'` 参数来避免此警告。

    参见
    --------
    pd.read_csv : 将 CSV（逗号分隔）文件读取到 DataFrame 中。
    """
    # 使用 pandas 的 read_table 函数读取通用的分隔文件，并将其转换为 DataFrame。
    
    Examples
    --------
    指定在 pd.read_csv 中使用非单字符分隔符 `sep` 的情况：
    
    >>> import io
    >>> csv = '''a;b;c
    ...           1;1,8
    ...           1;2,1'''
    >>> df = pd.read_csv(io.StringIO(csv), sep="[;,]")  # doctest: +SKIP
    ... # ParserWarning: Falling back to the 'python' engine...
    
    添加 `engine='python'` 到 pd.read_csv 可以消除警告信息：
    
    >>> df = pd.read_csv(io.StringIO(csv), sep="[;,]", engine="python")
# 定义一个自定义异常类，用于在数据合并时引发异常
class MergeError(ValueError):
    """
    Exception raised when merging data.

    Subclass of ``ValueError``.

    Examples
    --------
    >>> left = pd.DataFrame(
    ...     {"a": ["a", "b", "b", "d"], "b": ["cat", "dog", "weasel", "horse"]},
    ...     index=range(4),
    ... )
    >>> right = pd.DataFrame(
    ...     {"a": ["a", "b", "c", "d"], "c": ["meow", "bark", "chirp", "nay"]},
    ...     index=range(4),
    ... ).set_index("a")
    >>> left.join(
    ...     right,
    ...     on="a",
    ...     validate="one_to_one",
    ... )
    Traceback (most recent call last):
    MergeError: Merge keys are not unique in left dataset; not a one-to-one merge
    """


# 定义一个自定义异常类，用于抽象方法未实现时引发异常
class AbstractMethodError(NotImplementedError):
    """
    Raise this error instead of NotImplementedError for abstract methods.

    Examples
    --------
    >>> class Foo:
    ...     @classmethod
    ...     def classmethod(cls):
    ...         raise pd.errors.AbstractMethodError(cls, methodtype="classmethod")
    ...
    ...     def method(self):
    ...         raise pd.errors.AbstractMethodError(self)
    >>> test = Foo.classmethod()
    Traceback (most recent call last):
    AbstractMethodError: This classmethod must be defined in the concrete class Foo

    >>> test2 = Foo().method()
    Traceback (most recent call last):
    AbstractMethodError: This classmethod must be defined in the concrete class Foo
    """

    def __init__(self, class_instance, methodtype: str = "method") -> None:
        # 检查 methodtype 是否为预定义的类型之一
        types = {"method", "classmethod", "staticmethod", "property"}
        if methodtype not in types:
            raise ValueError(
                f"methodtype must be one of {methodtype}, got {types} instead."
            )
        self.methodtype = methodtype
        self.class_instance = class_instance

    def __str__(self) -> str:
        # 根据 methodtype 返回不同的错误信息
        if self.methodtype == "classmethod":
            name = self.class_instance.__name__
        else:
            name = type(self.class_instance).__name__
        return f"This {self.methodtype} must be defined in the concrete class {name}"


# 定义一个自定义异常类，用于在使用不支持的 Numba 引擎例程时引发异常
class NumbaUtilError(Exception):
    """
    Error raised for unsupported Numba engine routines.

    Examples
    --------
    >>> df = pd.DataFrame(
    ...     {"key": ["a", "a", "b", "b"], "data": [1, 2, 3, 4]}, columns=["key", "data"]
    ... )
    >>> def incorrect_function(x):
    ...     return sum(x) * 2.7
    >>> df.groupby("key").agg(incorrect_function, engine="numba")
    Traceback (most recent call last):
    NumbaUtilError: The first 2 arguments to incorrect_function
    must be ['values', 'index']
    """


# 定义一个自定义异常类，用于在操作可能引入重复标签时引发异常
class DuplicateLabelError(ValueError):
    """
    Error raised when an operation would introduce duplicate labels.

    Examples
    --------
    >>> s = pd.Series([0, 1, 2], index=["a", "b", "c"]).set_flags(
    ...     allows_duplicate_labels=False
    ... )
    >>> s.reindex(["a", "a", "b"])
    Traceback (most recent call last):
       ...
    """
    DuplicateLabelError: Index has duplicates.
    # 定义一个异常类 DuplicateLabelError，用于表示索引存在重复的情况
          positions
    # 声明了一个名为 positions 的列，可能用于存储某种位置信息或索引位置的数据
    label
    # 声明了一个名为 label 的列，通常用于存储数据的标签或索引标签
    a        [0, 1]
    # 在 label 列中添加了一个标签为 'a' 的索引，其值是一个列表 [0, 1]
# 定义一个自定义的异常类，用于处理使用无效索引键时的错误情况
class InvalidIndexError(Exception):
    """
    Exception raised when attempting to use an invalid index key.

    Examples
    --------
    >>> idx = pd.MultiIndex.from_product([["x", "y"], [0, 1]])
    >>> df = pd.DataFrame([[1, 1, 2, 2], [3, 3, 4, 4]], columns=idx)
    >>> df
        x       y
        0   1   0   1
    0   1   1   2   2
    1   3   3   4   4
    >>> df[:, 0]
    Traceback (most recent call last):
    InvalidIndexError: (slice(None, None, None), 0)
    """


# 定义一个自定义的异常类，用于处理在非数值数据上执行操作时引发的错误
class DataError(Exception):
    """
    Exception raised when performing an operation on non-numerical data.

    For example, calling ``ohlc`` on a non-numerical column or a function
    on a rolling window.

    Examples
    --------
    >>> ser = pd.Series(["a", "b", "c"])
    >>> ser.rolling(2).sum()
    Traceback (most recent call last):
    DataError: No numeric types to aggregate
    """


# 定义一个自定义的异常类，用于处理在函数定义不当时引发的错误
class SpecificationError(Exception):
    """
    Exception raised by ``agg`` when the functions are ill-specified.

    The exception raised in two scenarios.

    The first way is calling ``agg`` on a
    Dataframe or Series using a nested renamer (dict-of-dict).

    The second way is calling ``agg`` on a Dataframe with duplicated functions
    names without assigning column name.

    Examples
    --------
    >>> df = pd.DataFrame({"A": [1, 1, 1, 2, 2], "B": range(5), "C": range(5)})
    >>> df.groupby("A").B.agg({"foo": "count"})  # doctest: +SKIP
    ... # SpecificationError: nested renamer is not supported

    >>> df.groupby("A").agg({"B": {"foo": ["sum", "max"]}})  # doctest: +SKIP
    ... # SpecificationError: nested renamer is not supported

    >>> df.groupby("A").agg(["min", "min"])  # doctest: +SKIP
    ... # SpecificationError: nested renamer is not supported
    """


# 定义一个警告类，用于处理试图使用链式赋值时引发的警告
class ChainedAssignmentError(Warning):
    """
    Warning raised when trying to set using chained assignment.

    When the ``mode.copy_on_write`` option is enabled, chained assignment can
    never work. In such a situation, we are always setting into a temporary
    object that is the result of an indexing operation (getitem), which under
    Copy-on-Write always behaves as a copy. Thus, assigning through a chain
    can never update the original Series or DataFrame.

    For more information on Copy-on-Write,
    see :ref:`the user guide<copy_on_write>`.

    Examples
    --------
    >>> pd.options.mode.copy_on_write = True
    >>> df = pd.DataFrame({"A": [1, 1, 1, 2, 2]}, columns=["A"])
    >>> df["A"][0:3] = 10  # doctest: +SKIP
    ... # ChainedAssignmentError: ...
    >>> pd.options.mode.copy_on_write = False
    """


# 定义一个异常类，用于处理试图将内置的 numexpr 名称用作变量名时引发的错误
class NumExprClobberingError(NameError):
    """
    Exception raised when trying to use a built-in numexpr name as a variable name.

    ``eval`` or ``query`` will throw the error if the engine is set
    to 'numexpr'. 'numexpr' is the default engine value for these methods if the
    numexpr package is installed.

    Examples
    --------
    ...
    """
    >>> df = pd.DataFrame({"abs": [1, 1, 1]})
    >>> df.query("abs > 2")  # doctest: +SKIP
    # 查询DataFrame df中满足条件"abs > 2"的行，由于+SKIP标记，此行代码不会执行
    ... # NumExprClobberingError: Variables in expression "(abs) > (2)" overlap...
    # NumExprClobberingError: 表达式中的变量"(abs) > (2)"重叠，可能是由于变量名称冲突导致的错误
    
    >>> sin, a = 1, 2
    >>> pd.eval("sin + a", engine="numexpr")  # doctest: +SKIP
    # 使用Numexpr引擎计算字符串表达式"sin + a"的结果，由于+SKIP标记，此行代码不会执行
    ... # NumExprClobberingError: Variables in expression "(sin) + (a)" overlap...
    # NumExprClobberingError: 表达式中的变量"(sin) + (a)"重叠，可能是由于变量名称冲突导致的错误
# 定义一个自定义异常类，用于处理未定义变量的错误
class UndefinedVariableError(NameError):
    """
    Exception raised by ``query`` or ``eval`` when using an undefined variable name.

    It will also specify whether the undefined variable is local or not.

    Examples
    --------
    >>> df = pd.DataFrame({"A": [1, 1, 1]})
    >>> df.query("A > x")  # doctest: +SKIP
    ... # UndefinedVariableError: name 'x' is not defined
    >>> df.query("A > @y")  # doctest: +SKIP
    ... # UndefinedVariableError: local variable 'y' is not defined
    >>> pd.eval("x + 1")  # doctest: +SKIP
    ... # UndefinedVariableError: name 'x' is not defined
    """

    def __init__(self, name: str, is_local: bool | None = None) -> None:
        # 根据变量是否本地定义，构建错误消息
        base_msg = f"{name!r} is not defined"
        if is_local:
            msg = f"local variable {base_msg}"
        else:
            msg = f"name {base_msg}"
        super().__init__(msg)


# 定义一个异常类，用于处理索引错误
class IndexingError(Exception):
    """
    Exception is raised when trying to index and there is a mismatch in dimensions.

    Raised by properties like :attr:`.pandas.DataFrame.iloc` when
    an indexer is out of bounds or :attr:`.pandas.DataFrame.loc` when its index is
    unalignable to the frame index.

    See Also
    --------
    DataFrame.iloc : Purely integer-location based indexing for \
    selection by position.
    DataFrame.loc : Access a group of rows and columns by label(s) \
    or a boolean array.

    Examples
    --------
    >>> df = pd.DataFrame({"A": [1, 1, 1]})
    >>> df.loc[..., ..., "A"]  # doctest: +SKIP
    ... # IndexingError: indexer may only contain one '...' entry
    >>> df = pd.DataFrame({"A": [1, 1, 1]})
    >>> df.loc[1, ..., ...]  # doctest: +SKIP
    ... # IndexingError: Too many indexers
    >>> df[pd.Series([True], dtype=bool)]  # doctest: +SKIP
    ... # IndexingError: Unalignable boolean Series provided as indexer...
    >>> s = pd.Series(range(2), index=pd.MultiIndex.from_product([["a", "b"], ["c"]]))
    >>> s.loc["a", "c", "d"]  # doctest: +SKIP
    ... # IndexingError: Too many indexers
    """


# 定义一个运行时异常类，用于处理剪贴板功能不支持的情况
class PyperclipException(RuntimeError):
    """
    Exception raised when clipboard functionality is unsupported.

    Raised by ``to_clipboard()`` and ``read_clipboard()``.
    """


# 定义一个特定于 Windows 平台的剪贴板异常类，继承自 PyperclipException
class PyperclipWindowsException(PyperclipException):
    """
    Exception raised when clipboard functionality is unsupported by Windows.

    Access to the clipboard handle would be denied due to some other
    window process is accessing it.
    """

    def __init__(self, message: str) -> None:
        # 如果在 Windows 平台上，附加 Windows 错误信息到异常消息中
        message += f" ({ctypes.WinError()})"  # type: ignore[attr-defined]
        super().__init__(message)


# 定义一个用户警告类，用于处理 CSS 样式转换失败的情况
class CSSWarning(UserWarning):
    """
    Warning is raised when converting css styling fails.

    This can be due to the styling not having an equivalent value or because the
    styling isn't properly formatted.

    Examples
    --------
    >>> df = pd.DataFrame({"A": [1, 1, 1]})
    """
    # 用于DataFrame样式设置：将样式映射到Excel，并保存为'styled.xlsx'文件
    df.style.map(lambda x: "background-color: blueGreenRed;").to_excel(
        "styled.xlsx"
    )  # doctest: +SKIP
    # CSS警告：未处理的颜色格式为'blueGreenRed'
    
    # 用于DataFrame样式设置：将样式映射到Excel，并保存为'styled.xlsx'文件
    df.style.map(lambda x: "border: 1px solid red red;").to_excel(
        "styled.xlsx"
    )  # doctest: +SKIP
    # CSS警告：未处理的颜色格式为'blueGreenRed'
class PossibleDataLossError(Exception):
    """
    HDFStore 文件已经打开时尝试打开的异常。

    Examples
    --------
    >>> store = pd.HDFStore("my-store", "a")  # doctest: +SKIP
    >>> store.open("w")  # doctest: +SKIP
    """


class ClosedFileError(Exception):
    """
    尝试对已关闭的 HDFStore 文件执行操作时引发的异常。

    Examples
    --------
    >>> store = pd.HDFStore("my-store", "a")  # doctest: +SKIP
    >>> store.close()  # doctest: +SKIP
    >>> store.keys()  # doctest: +SKIP
    ... # ClosedFileError: my-store file is not open!
    """


class IncompatibilityWarning(Warning):
    """
    尝试在不兼容的 HDF5 文件上使用 where 条件时引发的警告。
    """


class AttributeConflictWarning(Warning):
    """
    在使用 HDFStore 时，当索引属性发生冲突时引发的警告。

    当尝试追加一个与 HDFStore 中现有索引名称不同的索引或尝试追加一个具有与现有索引不同频率的索引时发生。

    Examples
    --------
    >>> idx1 = pd.Index(["a", "b"], name="name1")
    >>> df1 = pd.DataFrame([[1, 2], [3, 4]], index=idx1)
    >>> df1.to_hdf("file", "data", "w", append=True)  # doctest: +SKIP
    >>> idx2 = pd.Index(["c", "d"], name="name2")
    >>> df2 = pd.DataFrame([[5, 6], [7, 8]], index=idx2)
    >>> df2.to_hdf("file", "data", "a", append=True)  # doctest: +SKIP
    AttributeConflictWarning: the [index_name] attribute of the existing index is
    [name1] which conflicts with the new [name2]...
    """


class DatabaseError(OSError):
    """
    在执行具有错误语法或引发错误的 SQL 时引发的错误。

    当 :func:`.pandas.read_sql` 传入错误的 SQL 语句时引发。

    See Also
    --------
    read_sql : 将 SQL 查询或数据库表读入 DataFrame。

    Examples
    --------
    >>> from sqlite3 import connect
    >>> conn = connect(":memory:")
    >>> pd.read_sql("select * test", conn)  # doctest: +SKIP
    """


class PossiblePrecisionLoss(Warning):
    """
    在 to_stata 中遇到值在 int64 范围内或外的列时引发的警告。

    当列的值在 int64 范围内或外时，列将被转换为 float64 类型。

    Examples
    --------
    >>> df = pd.DataFrame({"s": pd.Series([1, 2**53], dtype=np.int64)})
    >>> df.to_stata("test")  # doctest: +SKIP
    """


class ValueLabelTypeMismatch(Warning):
    """
    在 to_stata 中对包含非字符串值的类别列引发的警告。

    Examples
    --------
    >>> df = pd.DataFrame({"categories": pd.Series(["a", 2], dtype="category")})
    >>> df.to_stata("test")  # doctest: +SKIP
    """


class InvalidColumnName(Warning):
    """
    在 to_stata 中发现列名包含非有效 Stata 变量时引发的警告。

    因为列名是无效的 Stata 变量，需要更改该名称。

    注：此处示例未提供。
    """
    converted.

    See Also
    --------
    DataFrame.to_stata : Export DataFrame object to Stata dta format.

    Examples
    --------
    >>> df = pd.DataFrame({"0categories": pd.Series([2, 2])})
    >>> df.to_stata("test")  # doctest: +SKIP
    """
class CategoricalConversionWarning(Warning):
    """
    在使用迭代器读取部分带标签的 Stata 文件时引发警告。

    示例
    --------
    >>> from pandas.io.stata import StataReader
    >>> with StataReader("dta_file", chunksize=2) as reader:  # doctest: +SKIP
    ...     for i, block in enumerate(reader):
    ...         print(i, block)
    ... # CategoricalConversionWarning: One or more series with value labels...
    """


class LossySetitemError(Exception):
    """
    尝试对不是无损的 np.ndarray 进行 __setitem__ 操作时引发的异常。

    注意
    -----
    这是一个内部错误。
    """


class NoBufferPresent(Exception):
    """
    在 _get_data_buffer 中发出信号，指示没有请求的缓冲区。
    """


class InvalidComparison(Exception):
    """
    在 _validate_comparison_value 中引发，指示无效的比较。

    注意
    -----
    这是一个内部错误。
    """


__all__ = [
    "AbstractMethodError",
    "AttributeConflictWarning",
    "CategoricalConversionWarning",
    "ChainedAssignmentError",
    "ClosedFileError",
    "CSSWarning",
    "DatabaseError",
    "DataError",
    "DtypeWarning",
    "DuplicateLabelError",
    "EmptyDataError",
    "IncompatibilityWarning",
    "IntCastingNaNError",
    "InvalidColumnName",
    "InvalidComparison",
    "InvalidIndexError",
    "InvalidVersion",
    "IndexingError",
    "LossySetitemError",
    "MergeError",
    "NoBufferPresent",
    "NullFrequencyError",
    "NumbaUtilError",
    "NumExprClobberingError",
    "OptionError",
    "OutOfBoundsDatetime",
    "OutOfBoundsTimedelta",
    "ParserError",
    "ParserWarning",
    "PerformanceWarning",
    "PossibleDataLossError",
    "PossiblePrecisionLoss",
    "PyperclipException",
    "PyperclipWindowsException",
    "SpecificationError",
    "UndefinedVariableError",
    "UnsortedIndexError",
    "UnsupportedFunctionCall",
    "ValueLabelTypeMismatch",
]
```