# `D:\src\scipysrc\pandas\pandas\core\shared_docs.py`

```
# 导入未来版本支持的注解
from __future__ import annotations

# 创建一个空字典 _shared_docs，用于存储文档字符串
_shared_docs: dict[str, str] = {}

# 添加名为 "aggregate" 的文档字符串，描述数据聚合操作的参数和返回值
_shared_docs["aggregate"] = """
Aggregate using one or more operations over the specified axis.

Parameters
----------
func : function, str, list or dict
    Function to use for aggregating the data. If a function, must either
    work when passed a {klass} or when passed to {klass}.apply.

    Accepted combinations are:

    - function
    - string function name
    - list of functions and/or function names, e.g. ``[np.sum, 'mean']``
    - dict of axis labels -> functions, function names or list of such.
{axis}
*args
    Positional arguments to pass to `func`.
**kwargs
    Keyword arguments to pass to `func`.

Returns
-------
scalar, Series or DataFrame

    The return can be:

    * scalar : when Series.agg is called with single function
    * Series : when DataFrame.agg is called with a single function
    * DataFrame : when DataFrame.agg is called with several functions
{see_also}
Notes
-----
The aggregation operations are always performed over an axis, either the
index (default) or the column axis. This behavior is different from
`numpy` aggregation functions (`mean`, `median`, `prod`, `sum`, `std`,
`var`), where the default is to compute the aggregation of the flattened
array, e.g., ``numpy.mean(arr_2d)`` as opposed to
``numpy.mean(arr_2d, axis=0)``.

`agg` is an alias for `aggregate`. Use the alias.

Functions that mutate the passed object can produce unexpected
behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
for more details.

A passed user-defined-function will be passed a Series for evaluation.
{examples}"""

# 添加名为 "compare" 的文档字符串，描述对象比较操作的参数
_shared_docs["compare"] = """
Compare to another {klass} and show the differences.

Parameters
----------
other : {klass}
    Object to compare with.

align_axis : {{0 or 'index', 1 or 'columns'}}, default 1
    Determine which axis to align the comparison on.

    * 0, or 'index' : Resulting differences are stacked vertically
        with rows drawn alternately from self and other.
    * 1, or 'columns' : Resulting differences are aligned horizontally
        with columns drawn alternately from self and other.

keep_shape : bool, default False
    If true, all rows and columns are kept.
    Otherwise, only the ones with different values are kept.

keep_equal : bool, default False
    If true, the result keeps values that are equal.
    Otherwise, equal values are shown as NaNs.

result_names : tuple, default ('self', 'other')
    Set the dataframes names in the comparison.

    .. versionadded:: 1.5.0
"""

# 添加名为 "groupby" 的文档字符串，描述分组操作的参数
_shared_docs["groupby"] = """
Group %(klass)s using a mapper or by a Series of columns.

A groupby operation involves some combination of splitting the
object, applying a function, and combining the results. This can be
used to group large amounts of data and compute operations on these
groups.

Parameters
----------
by : mapping, function, label, pd.Grouper or list of such
    Used to determine the groups for the groupby.
    # 如果 `by` 是一个函数，它将被用于对象索引的每个值。
    # 如果传递了一个字典或者 Series，将使用 Series 或字典的值来确定分组（首先会对 Series 的值进行对齐，参见 `.align()` 方法）。
    # 如果传递了一个长度等于选择轴的列表或 ndarray（参见 `groupby 用户指南 <https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#splitting-an-object-into-groups>`_），
    # 这些值将直接用于确定分组。可以传递一个标签或标签列表来按 `self` 中的列分组。
    # 注意，元组会被解释为单个键。
# level : int, level name, or sequence of such, default None
# 如果轴是多级索引（层次化），则按特定级别或多个级别分组。不要同时指定“by”和“level”。

# as_index : bool, default True
# 返回的对象使用分组标签作为索引。仅适用于DataFrame输入。as_index=False实际上是“SQL风格”分组输出。
# 此参数不影响过滤（参见用户指南中的“过滤”），如“head()”，“tail()”，“nth()”以及变换（参见用户指南中的“变换”）。

# sort : bool, default True
# 对分组键进行排序。通过关闭此选项可以获得更好的性能。
# 注意这不会影响每个组内观察值的顺序。Groupby保留每个组内行的顺序。如果为False，组将按照它们在原始DataFrame中的顺序出现。
# 此参数不影响过滤（参见用户指南中的“过滤”），如“head()”，“tail()”，“nth()”以及变换（参见用户指南中的“变换”）。

# group_keys : bool, default True
# 在调用apply时，如果“by”参数产生类似索引的（例如：参考`groupby.transform`）结果，则将组键添加到索引以标识各个片段。
# 默认情况下，当结果的索引（和列）标签与输入匹配时，不包括组键；否则包括。

# observed : bool, default True
# 仅当任何分组器为分类变量时才适用。
# 如果为True：仅显示分类分组器的观察值。
# 如果为False：显示分类分组器的所有值。

# dropna : bool, default True
# 如果为True，并且分组键包含NA值，则NA值与行/列一起将被删除。
# 如果为False，则NA值也将被视为组中的键。

# Returns
# -------
# pandas.api.typing.%(klass)sGroupBy
# 返回一个包含有关分组信息的GroupBy对象。

# See Also
# --------
# resample : 用于时间序列的频率转换和重新采样的便捷方法。

# Notes
# -----
# 引用了 pandas 文档中的一段说明，详细描述了 groupby 方法的用法和示例，包括对象分组、迭代组、选择组、聚合等。
_shared_docs["transform"] = """
# 将 func 应用于自身，生成一个与自身轴形状相同的 {klass} 对象。
Call ``func`` on self producing a {klass} with the same axis shape as self.

Parameters
----------
func : function, str, list-like or dict-like
    要用于数据转换的函数。如果是函数，必须能够处理传递给 {klass} 或 {klass}.apply 的参数。
    如果 func 同时是类似列表和类似字典，以字典的行为为优先。

    可接受的组合包括：

    - 函数
    - 字符串函数名
    - 函数和/或函数名的类似列表，例如 ``[np.exp, 'sqrt']``
    - 轴标签 -> 函数、函数名或类似列表的字典形式。
{axis}
*args
    传递给 `func` 的位置参数。
**kwargs
    传递给 `func` 的关键字参数。

Returns
-------
{klass}
    一个与 self 长度相同的 {klass} 对象。

Raises
------
ValueError : 如果返回的 {klass} 长度与 self 不同。

See Also
--------
{klass}.agg : 只执行聚合类型操作。
{klass}.apply : 在 {klass} 上调用函数。

Notes
-----
对传递对象进行变异的函数可能会产生意外行为或错误，不被支持。请参阅 :ref:`gotchas.udf-mutation` 了解更多详情。

Examples
--------
>>> df = pd.DataFrame({{'A': range(3), 'B': range(1, 4)}})
>>> df
   A  B
0  0  1
1  1  2
2  2  3
>>> df.transform(lambda x: x + 1)
   A  B
0  1  2
1  2  3
2  3  4

即使生成的 {klass} 必须与输入的 {klass} 长度相同，也可以提供多个输入函数：

>>> s = pd.Series(range(3))
>>> s
0    0
1    1
2    2
dtype: int64
>>> s.transform([np.sqrt, np.exp])
       sqrt        exp
0  0.000000   1.000000
1  1.000000   2.718282
2  1.414214   7.389056

您可以在 GroupBy 对象上调用 transform：

>>> df = pd.DataFrame({{
...     "Date": [
...         "2015-05-08", "2015-05-07", "2015-05-06", "2015-05-05",
...         "2015-05-08", "2015-05-07", "2015-05-06", "2015-05-05"],
...     "Data": [5, 8, 6, 1, 50, 100, 60, 120],
... }})
>>> df
         Date  Data
0  2015-05-08     5
1  2015-05-07     8
2  2015-05-06     6
3  2015-05-05     1
4  2015-05-08    50
5  2015-05-07   100
6  2015-05-06    60
7  2015-05-05   120
>>> df.groupby('Date')['Data'].transform('sum')
0     55
1    108
2     66
3    121
4     55
5    108
6     66
7    121
Name: Data, dtype: int64

>>> df = pd.DataFrame({{
...     "c": [1, 1, 1, 2, 2, 2, 2],
"""
_shared_docs["storage_options"] = """storage_options : dict, optional
    Extra options that make sense for a particular storage connection, e.g.
    host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
    are forwarded to ``urllib.request.Request`` as header options. For other
    URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
    forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
    details, and for more examples on storage options refer `here
    <https://pandas.pydata.org/docs/user_guide/io.html?
    highlight=storage_options#reading-writing-remote-files>`_."""

# 定义了 storage_options 文档字符串，描述了用于存储连接的额外选项的含义和用法，包括对不同类型的 URL 的处理方式

_shared_docs["compression_options"] = """compression : str or dict, default 'infer'
    For on-the-fly compression of the output data. If 'infer' and '%s' is
    path-like, then detect compression from the following extensions: '.gz',
    '.bz2', '.zip', '.xz', '.zst', '.tar', '.tar.gz', '.tar.xz' or '.tar.bz2'
    (otherwise no compression).
    Set to ``None`` for no compression.
    Can also be a dict with key ``'method'`` set
    to one of {``'zip'``, ``'gzip'``, ``'bz2'``, ``'zstd'``, ``'xz'``, ``'tar'``} and
    other key-value pairs are forwarded to
    ``zipfile.ZipFile``, ``gzip.GzipFile``,
    ``bz2.BZ2File``, ``zstandard.ZstdCompressor``, ``lzma.LZMAFile`` or
    ``tarfile.TarFile``, respectively.
    As an example, the following could be passed for faster compression and to create
    a reproducible gzip archive:
    ``compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1}``.

    .. versionadded:: 1.5.0
        Added support for `.tar` files."""

# 定义了 compression_options 文档字符串，描述了对输出数据进行即时压缩的选项说明，包括支持的压缩方法和相关参数，以及版本更新的信息

_shared_docs["decompression_options"] = """compression : str or dict, default 'infer'
    For on-the-fly decompression of on-disk data. If 'infer' and '%s' is
    path-like, then detect compression from the following extensions: '.gz',
    '.bz2', '.zip', '.xz', '.zst', '.tar', '.tar.gz', '.tar.xz' or '.tar.bz2'
    (otherwise no compression).
    If using 'zip' or 'tar', the ZIP file must contain only one data file to be read in.
    Set to ``None`` for no decompression.
    Can also be a dict with key ``'method'`` set
    to one of {``'zip'``, ``'gzip'``, ``'bz2'``, ``'zstd'``, ``'xz'``, ``'tar'``} and
    other key-value pairs are forwarded to
    ``zipfile.ZipFile``, ``gzip.GzipFile``,
    ``bz2.BZ2File``, ``zstandard.ZstdDecompressor``, ``lzma.LZMAFile`` or
    ``tarfile.TarFile``, respectively.
    As an example, the following could be passed for Zstandard decompression using a
    custom compression dictionary:
    ``compression={'method': 'zstd', 'dict_data': my_compression_dict}``.

# 定义了 decompression_options 文档字符串，描述了对磁盘数据进行即时解压缩的选项说明，包括支持的解压缩方法和相关参数，以及解压条件的说明
    # 标记：版本新增功能，引入了对 .tar 文件的支持
    .. versionadded:: 1.5.0
        Added support for `.tar` files."""
_shared_docs["replace"] = """
    Replace values given in `to_replace` with `value`.

    Values of the {klass} are replaced with other values dynamically.
    This differs from updating with ``.loc`` or ``.iloc``, which require
    you to specify a location to update with some value.

    Parameters
    ----------
    to_replace : str, regex, list, dict, Series, int, float, or None
        How to find the values that will be replaced.
        For example, if `to_replace` is a dictionary, then it is used to
        map `to_replace` values to `value` values.
    value : scalar, dict, list, str, regex, default None
        Value to replace any values matching `to_replace` with.
        For a DataFrame, a dict of values specifying which value to use for
        each column (columns not in the dict will not be filled).
        If `value` is a list, `value` should be of the same length and type
        as `to_replace`.
    inplace : bool, default False
        If True, in place. Note: this will modify any other views on this
        object (e.g., a DataFrame from a Series).
    limit : int, default None
        Maximum size gap to forward or backward fill.
    regex : bool or same types as `to_replace`, default False
        Whether to interpret `to_replace` and/or `value` as regular
        expressions. If this is True then to_replace must be a string.
        Alternatively, this could be a regular expression or a list,
        dict, or array of regular expressions in which case to_replace
        must be a list, dict, or array with the same length as `value`.
    method : {'pad', 'ffill', 'bfill', None}, default None
        The method to use when for replacement, when `to_replace` is
        a scalar or sequence that is only partially found in the
        subject. The `pad` method replaces a match that starts at
        the beginning of the string. `ffill` and `bfill` are
        forward and backward fills respectively.
    errors : {'raise', 'ignore'}, default 'raise'
        If 'raise', then invalid parsing will raise an exception.
        If 'ignore', then invalid parsing will return the input.

    Returns
    -------
    None
    """
    to_replace : str, regex, list, dict, Series, int, float, or None
        # 参数 `to_replace` 可以是字符串、正则表达式、列表、字典、Series、整数、浮点数或 None
        How to find the values that will be replaced.
        # 指定将要被替换的值的查找方式

        * numeric, str or regex:

            - numeric: numeric values equal to `to_replace` will be
              replaced with `value`
            # 如果 `to_replace` 是数值型，则与 `value` 相等的数值将被替换为指定的 `value`
            - str: string exactly matching `to_replace` will be replaced
              with `value`
            # 如果 `to_replace` 是字符串，则精确匹配的字符串将被替换为指定的 `value`
            - regex: regexs matching `to_replace` will be replaced with
              `value`
            # 如果 `to_replace` 是正则表达式，则匹配的正则表达式将被替换为指定的 `value`

        * list of str, regex, or numeric:

            - First, if `to_replace` and `value` are both lists, they
              **must** be the same length.
            # 如果 `to_replace` 和 `value` 都是列表，则它们必须具有相同的长度
            - Second, if ``regex=True`` then all of the strings in **both**
              lists will be interpreted as regexs otherwise they will match
              directly.
            # 如果设置了 `regex=True`，则两个列表中的所有字符串都将被解释为正则表达式，否则它们将直接匹配
            This doesn't matter much for `value` since there
              are only a few possible substitution regexes you can use.
            # 对于 `value` 参数来说，由于可以使用的替换正则表达式很少，这并不重要。
            - str, regex and numeric rules apply as above.
            # 字符串、正则表达式和数值的规则同上述描述

        * dict:

            - Dicts can be used to specify different replacement values
              for different existing values.
            # 字典可以用于指定不同的替换值，以替换不同的现有值
            For example,
              ``{{'a': 'b', 'y': 'z'}}`` replaces the value 'a' with 'b' and
              'y' with 'z'. To use a dict in this way, the optional `value`
              parameter should not be given.
            # 例如，`{'a': 'b', 'y': 'z'}` 将值 'a' 替换为 'b'，将 'y' 替换为 'z'。在这种用法下，`value` 参数应该不需要给出。
            - For a DataFrame a dict can specify that different values
              should be replaced in different columns.
            # 对于 DataFrame，字典可以指定在不同的列中替换不同的值
            For example,
              ``{{'a': 1, 'b': 'z'}}`` looks for the value 1 in column 'a'
              and the value 'z' in column 'b' and replaces these values
              with whatever is specified in `value`.
            # 例如，`{'a': 1, 'b': 'z'}` 在列 'a' 中查找值 1，在列 'b' 中查找值 'z'，并用 `value` 中指定的内容替换这些值。
              The `value` parameter
              should not be ``None`` in this case. You can treat this as a
              special case of passing two lists except that you are
              specifying the column to search in.
            # 在这种情况下，`value` 参数不应为 `None`。您可以将其视为传递两个列表的特殊情况，除了您在指定要搜索的列之外。
            - For a DataFrame nested dictionaries, e.g.,
              ``{{'a': {{'b': np.nan}}}}``, are read as follows: look in column
              'a' for the value 'b' and replace it with NaN.
            # 对于 DataFrame 中的嵌套字典，例如 `{'a': {'b': np.nan}}`，解析如下：在列 'a' 中查找值 'b'，并将其替换为 NaN。
              The optional `value`
              parameter should not be specified to use a nested dict in this
              way.
            # 在这种用法下，不应指定可选的 `value` 参数来使用嵌套字典。
              You can nest regular expressions as well.
            # 您也可以嵌套正则表达式。
              Note that
              column names (the top-level dictionary keys in a nested
              dictionary) **cannot** be regular expressions.
            # 请注意，列名（嵌套字典中的顶级字典键）不能是正则表达式。

        * None:

            - This means that the `regex` argument must be a string,
              compiled regular expression, or list, dict, ndarray or
              Series of such elements.
            # 这意味着 `regex` 参数必须是字符串、编译的正则表达式、列表、字典、ndarray 或 Series 类型的元素。
              If `value` is also ``None`` then
              this **must** be a nested dictionary or Series.
            # 如果 `value` 也是 `None`，则必须是一个嵌套字典或 Series。

        See the examples section for examples of each of these.
        # 请参阅示例部分，了解每种情况的示例。
    # 参数value可以是标量、字典、列表、字符串、正则表达式或默认值None，用来替换与to_replace匹配的值。
    # 对于DataFrame，可以使用字典指定每列要使用的值（不在字典中的列不会填充）。
    # 正则表达式、字符串、列表或这些对象的字典也是允许的。
    {inplace}
    # 是否将to_replace和value解释为正则表达式。也可以是一个正则表达式，或者一个正则表达式的列表、字典或数组，
    # 在这种情况下，to_replace必须为None。
    regex : bool or same types as `to_replace`, default False
    # 返回替换后的对象。
    {klass}
    # 抛出AssertionError异常的情况：
    # * 如果regex不是布尔值且to_replace不是None。
    # 抛出TypeError异常的情况：
    # * 如果to_replace不是标量、类数组、字典或None。
    # * 如果to_replace是字典且value不是列表、字典、ndarray或Series。
    # * 如果to_replace为None且regex无法编译为正则表达式，或者是列表、字典、ndarray或Series。
    # * 当替换多个布尔值或datetime64对象时，to_replace的参数类型与被替换的值类型不匹配时。
    # 抛出ValueError异常的情况：
    # * 如果将列表或ndarray传递给to_replace和value，但它们的长度不相同。
    # 查看也可以参考
    # Series.fillna：填充NA值。
    # DataFrame.fillna：填充NA值。
    # Series.where：根据布尔条件替换值。
    # DataFrame.where：根据布尔条件替换值。
    # DataFrame.map：对DataFrame逐元素应用函数。
    # Series.map：根据输入映射或函数映射Series的值。
    # Series.str.replace：简单的字符串替换。
    # 注意事项：
    # * 在幕后使用re.sub执行正则表达式替换。替换规则与re.sub相同。
    # * 正则表达式仅在字符串上执行替换，这意味着您不能提供例如匹配浮点数的正则表达式并期望您的数据帧中具有数值dtype的列被匹配。
    #   但是，如果这些浮点数是字符串，则可以做到这一点。
    # * 此方法有很多选项。鼓励您尝试和使用此方法，以便更好地理解其工作原理。
    # * 当字典用作to_replace值时，类似于字典中的键是要替换的部分，字典中的值是value参数。
    # 示例：
    # 标量to_replace和value的示例：
    >>> s = pd.Series([1, 2, 3, 4, 5])
    >>> s.replace(1, 5)
    # 输出：
    # 0    5
    # 1    2
    # 2    3
    # 3    4
    # 4    5
    # dtype: int64
    # 创建一个包含三列（'A', 'B', 'C'）的DataFrame对象，每列包含相应的数据
    >>> df = pd.DataFrame({'A': [0, 1, 2, 3, 4],
    ...                    'B': [5, 6, 7, 8, 9],
    ...                    'C': ['a', 'b', 'c', 'd', 'e']})
    
    # 使用replace方法将DataFrame中的0替换为5，生成新的DataFrame
    >>> df.replace(0, 5)
        A  B  C
    0  5  5  a
    1  1  6  b
    2  2  7  c
    3  3  8  d
    4  4  9  e

    # 使用列表形式的to_replace参数，将0, 1, 2, 3替换为4，生成新的DataFrame
    **List-like `to_replace`**
    >>> df.replace([0, 1, 2, 3], 4)
        A  B  C
    0  4  5  a
    1  4  6  b
    2  4  7  c
    3  4  8  d
    4  4  9  e

    # 使用字典形式的to_replace参数，将0替换为10，1替换为100，生成新的DataFrame
    >>> df.replace({0: 10, 1: 100})
        A  B  C
    0  10  5  a
    1  100  6  b
    2    2  7  c
    3    3  8  d
    4    4  9  e

    # 使用字典形式的to_replace参数，将'A'列中的0替换为100，'B'列中的5替换为100，生成新的DataFrame
    >>> df.replace({'A': 0, 'B': 5}, 100)
        A    B  C
    0  100  100  a
    1    1    6  b
    2    2    7  c
    3    3    8  d
    4    4    9  e

    # 使用嵌套字典形式的to_replace参数，将'A'列中的0替换为100，'A'列中的4替换为400，生成新的DataFrame
    >>> df.replace({'A': {0: 100, 4: 400}})
        A  B  C
    0  100  5  a
    1    1  6  b
    2    2  7  c
    3    3  8  d
    4  400  9  e

    # 使用正则表达式进行替换，将'A'列中符合'^ba.$'的字符串替换为'new'，生成新的DataFrame
    >>> df = pd.DataFrame({'A': ['bat', 'foo', 'bait'],
    ...                    'B': ['abc', 'bar', 'xyz']})
    >>> df.replace(to_replace=r'^ba.$', value='new', regex=True)
            A    B
    0   new  abc
    1   foo  new
    2  bait  xyz

    # 使用字典形式的to_replace参数和正则表达式，将'A'列中符合'^ba.$'的字符串替换为'new'，生成新的DataFrame
    >>> df.replace({'A': r'^ba.$'}, {'A': 'new'}, regex=True)
            A    B
    0   new  abc
    1   foo  bar
    2  bait  xyz

    # 使用正则表达式进行替换，将DataFrame中符合'^ba.$'的字符串替换为'new'，生成新的DataFrame
    >>> df.replace(regex=r'^ba.$', value='new')
            A    B
    0   new  abc
    1   foo  new
    2  bait  xyz

    # 使用嵌套字典形式的to_replace参数和正则表达式，将'A'列中符合'^ba.$'的字符串替换为'new'，'foo'替换为'xyz'，生成新的DataFrame
    >>> df.replace(regex={r'^ba.$': 'new', 'foo': 'xyz'})
            A    B
    0   new  abc
    1   xyz  new
    2  bait  xyz

    # 比较`s.replace({'a': None})`和`s.replace('a', None)`的行为，理解to_replace参数的特殊情况
    >>> s = pd.Series([10, 'a', 'a', 'b', 'a'])

    # 当to_replace参数为字典时，相当于字典中的值等于value参数，例如`s.replace({'a': None})`等价于`s.replace(to_replace={'a': None}, value=None)`：
    >>> s.replace({'a': None})
    0      10
    1    None
    2    None
    3       b
    4    None
    dtype: object

    # 如果value参数显式地传递为None，则会被尊重：
    >>> s.replace('a', None)
    0      10
    1    None
    2    None
    3       b
    4    None
    dtype: object

        # 自版本1.4.0起发生了变化
        # 之前会默默地忽略显式的None值

    # 当regex=True时，value参数不为None且to_replace是字符串时，替换将应用于DataFrame的所有列：
    >>> df = pd.DataFrame({'A': [0, 1, 2, 3, 4],
    ...                    'B': ['a', 'b', 'c', 'd', 'e'],
    # 创建一个示例的DataFrame，包含三列A、B、C，每列各有五行数据
    >>> df = pd.DataFrame({'A': [0, 1, 2, 3, 4],
                          'B': ['a', 'b', 'c', 'd', 'e'],
                          'C': ['f', 'g', 'h', 'i', 'j']})

    # 使用正则表达式替换DataFrame中列B和列C中以字母a到g开头的字符串为'e'
    >>> df.replace(to_replace='^[a-g]', value='e', regex=True)
        A  B  C
    0  0  e  e
    1  1  e  e
    2  2  e  h
    3  3  e  i
    4  4  e  j

    # 如果`value`不是`None`且`to_replace`是一个字典，字典的键将是要替换的DataFrame的列
    # 用正则表达式替换DataFrame中列B匹配字母a到c开头和列C匹配字母h到j开头的字符串为'e'
    >>> df.replace(to_replace={'B': '^[a-c]', 'C': '^[h-j]'}, value='e', regex=True)
        A  B  C
    0  0  e  f
    1  1  e  g
    2  2  e  e
    3  3  d  e
    4  4  e  e
# 定义一个文档字符串，描述了 DataFrame 的 idxmin 方法的功能和用法
_shared_docs["idxmin"] = """
    Return index of first occurrence of minimum over requested axis.

    NA/null values are excluded.

    Parameters
    ----------
    axis : {{0 or 'index', 1 or 'columns'}}, default 0
        The axis to use. 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
    skipna : bool, default True
        Exclude NA/null values. If the entire Series is NA, or if ``skipna=False``
        and there is an NA value, this method will raise a ``ValueError``.
    numeric_only : bool, default {numeric_only_default}
        Include only `float`, `int` or `boolean` data.

        .. versionadded:: 1.5.0

    Returns
    -------
    Series
        Indexes of minima along the specified axis.

    Raises
    ------
    ValueError
        * If the row/column is empty

    See Also
    --------
    Series.idxmin : Return index of the minimum element.

    Notes
    -----
    This method is the DataFrame version of ``ndarray.argmin``.

    Examples
    --------
    Consider a dataset containing food consumption in Argentina.

    >>> df = pd.DataFrame({{'consumption': [10.51, 103.11, 55.48],
    ...                   'co2_emissions': [37.2, 19.66, 1712]}},
    ...                   index=['Pork', 'Wheat Products', 'Beef'])

    >>> df
                    consumption  co2_emissions
    Pork                  10.51         37.20
    Wheat Products       103.11         19.66
    Beef                  55.48       1712.00

    By default, it returns the index for the minimum value in each column.

    >>> df.idxmin()
    consumption                Pork
    co2_emissions    Wheat Products
    dtype: object

    To return the index for the minimum value in each row, use ``axis="columns"``.

    >>> df.idxmin(axis="columns")
    Pork                consumption
    Wheat Products    co2_emissions
    Beef                consumption
    dtype: object
"""

# 定义一个文档字符串，描述了 DataFrame 的 idxmax 方法的功能和用法
_shared_docs["idxmax"] = """
    Return index of first occurrence of maximum over requested axis.

    NA/null values are excluded.

    Parameters
    ----------
    axis : {{0 or 'index', 1 or 'columns'}}, default 0
        The axis to use. 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
    skipna : bool, default True
        Exclude NA/null values. If the entire Series is NA, or if ``skipna=False``
        and there is an NA value, this method will raise a ``ValueError``.
    numeric_only : bool, default {numeric_only_default}
        Include only `float`, `int` or `boolean` data.

        .. versionadded:: 1.5.0

    Returns
    -------
    Series
        Indexes of maxima along the specified axis.

    Raises
    ------
    ValueError
        * If the row/column is empty

    See Also
    --------
    Series.idxmax : Return index of the maximum element.

    Notes
    -----
    This method is the DataFrame version of ``ndarray.argmax``.

    Examples
    --------
    Consider a dataset containing food consumption in Argentina.
"""
    # 创建一个 Pandas 数据框，包含两列 'consumption' 和 'co2_emissions'，以及三行数据
    df = pd.DataFrame({'consumption': [10.51, 103.11, 55.48],
                       'co2_emissions': [37.2, 19.66, 1712]},
                      index=['Pork', 'Wheat Products', 'Beef'])
    
    # 打印显示数据框 df 的内容，展示出每行的消费量和二氧化碳排放量
    """
                    consumption  co2_emissions
    Pork                  10.51         37.20
    Wheat Products       103.11         19.66
    Beef                  55.48       1712.00
    """
    
    # 默认情况下，返回每列中最大值的索引
    """
    By default, it returns the index for the maximum value in each column.
    """
    
    # 调用 DataFrame 的 idxmax() 方法，返回每列中最大值的索引
    df.idxmax()
    """
    consumption     Wheat Products
    co2_emissions             Beef
    dtype: object
    """
    
    # 若要返回每行中最大值的索引，使用参数 axis="columns"
    """
    To return the index for the maximum value in each row, use ``axis="columns"``.
    """
    
    # 调用 DataFrame 的 idxmax() 方法，并指定 axis="columns" 参数，返回每行中最大值的索引
    df.idxmax(axis="columns")
    """
    Pork              co2_emissions
    Wheat Products     consumption
    Beef              co2_emissions
    dtype: object
    """
# 导入需要使用的库
import os
import sys
import json

# 定义一个函数，用于将指定路径下的所有 JSON 文件内容读取并合并为一个字典
def merge_json_files(folder):
    # 初始化一个空字典，用于存储所有 JSON 文件的内容
    merged_data = {}
    
    # 遍历指定文件夹下的所有文件和子文件夹
    for dirpath, _, filenames in os.walk(folder):
        # 遍历当前文件夹中的所有文件名
        for filename in filenames:
            # 检查文件是否以 .json 结尾
            if filename.endswith('.json'):
                # 拼接文件的完整路径
                filepath = os.path.join(dirpath, filename)
                # 打开文件并加载 JSON 数据
                with open(filepath, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    # 将当前 JSON 数据合并到总字典中
                    merged_data.update(json_data)
    
    # 返回合并后的总字典
    return merged_data
```