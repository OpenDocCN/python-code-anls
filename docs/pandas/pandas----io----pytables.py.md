# `D:\src\scipysrc\pandas\pandas\io\pytables.py`

```
# PyTables的高级接口，用于读写pandas数据结构到磁盘
"""
High level interface to PyTables for reading and writing pandas data structures
to disk
"""

# 导入必要的模块和函数
from __future__ import annotations

from contextlib import suppress  # 用于上下文管理中忽略异常
import copy  # 用于复制对象
from datetime import (  # 导入日期相关的类和函数
    date,  # 日期类
    tzinfo,  # 时区信息类
)
import itertools  # 用于生成迭代器的函数
import os  # 提供与操作系统相关的功能
import re  # 提供正则表达式操作
from textwrap import dedent  # 用于去除多行字符串的缩进
from typing import (  # 提供类型提示相关的功能
    TYPE_CHECKING,  # 类型检查标志
    Any,  # 任意类型
    Final,  # 声明常量
    Literal,  # 字面值类型
    cast,  # 类型强制转换函数
    overload,  # 函数重载装饰器
)
import warnings  # 控制警告的输出

import numpy as np  # 导入NumPy库，用于数值计算

from pandas._config import (  # 导入Pandas的配置信息和函数
    config,  # Pandas的配置对象
    get_option,  # 获取配置选项的函数
    using_pyarrow_string_dtype,  # 是否使用PyArrow的字符串类型
)

from pandas._libs import (  # 导入Pandas底层库
    lib,  # Pandas的C语言库
    writers as libwriters,  # Pandas的写入库
)
from pandas._libs.lib import is_string_array  # 检查是否为字符串数组
from pandas._libs.tslibs import timezones  # Pandas时区相关的功能
from pandas.compat._optional import import_optional_dependency  # 导入可选的依赖项
from pandas.compat.pickle_compat import patch_pickle  # 兼容性处理，用于Pickle的补丁
from pandas.errors import (  # 导入Pandas的错误和警告类
    AttributeConflictWarning,  # 属性冲突警告
    ClosedFileError,  # 文件关闭错误
    IncompatibilityWarning,  # 不兼容性警告
    PerformanceWarning,  # 性能警告
    PossibleDataLossError,  # 可能的数据丢失错误
)
from pandas.util._decorators import cache_readonly  # 缓存只读的装饰器
from pandas.util._exceptions import find_stack_level  # 查找调用栈的层级

from pandas.core.dtypes.common import (  # 导入Pandas常见数据类型相关函数
    ensure_object,  # 确保是对象类型
    is_bool_dtype,  # 判断是否为布尔类型
    is_complex_dtype,  # 判断是否为复数类型
    is_list_like,  # 判断是否为列表类型
    is_string_dtype,  # 判断是否为字符串类型
    needs_i8_conversion,  # 判断是否需要int64类型转换
)
from pandas.core.dtypes.dtypes import (  # 导入Pandas的数据类型
    CategoricalDtype,  # 分类数据类型
    DatetimeTZDtype,  # 带时区的日期时间数据类型
    ExtensionDtype,  # 扩展数据类型
    PeriodDtype,  # 周期数据类型
)
from pandas.core.dtypes.missing import array_equivalent  # 判断两个数组是否等价

from pandas import (  # 导入Pandas顶层API
    DataFrame,  # 数据框
    DatetimeIndex,  # 日期时间索引
    Index,  # 索引
    MultiIndex,  # 多重索引
    PeriodIndex,  # 周期索引
    RangeIndex,  # 范围索引
    Series,  # 序列
    TimedeltaIndex,  # 时间差索引
    concat,  # 合并函数
    isna,  # 检查缺失值函数
)
from pandas.core.arrays import (  # 导入Pandas数组模块
    Categorical,  # 分类数组
    DatetimeArray,  # 日期时间数组
    PeriodArray,  # 周期数组
)
from pandas.core.arrays.datetimes import tz_to_dtype  # 时区到数据类型的转换函数
import pandas.core.common as com  # 导入Pandas核心公共函数
from pandas.core.computation.pytables import (  # 导入PyTables表达式相关功能
    PyTablesExpr,  # PyTables表达式对象
    maybe_expression,  # 可能是表达式
)
from pandas.core.construction import extract_array  # 提取数组的函数
from pandas.core.indexes.api import ensure_index  # 确保索引的函数

from pandas.io.common import stringify_path  # 路径字符串化函数
from pandas.io.formats.printing import (  # 导入打印格式相关的功能
    adjoin,  # 连接函数
    pprint_thing,  # 打印对象函数
)

if TYPE_CHECKING:
    from collections.abc import (  # 导入标准集合抽象基类
        Callable,  # 可调用对象
        Hashable,  # 可哈希对象
        Iterator,  # 迭代器
        Sequence,  # 序列
    )
    from types import TracebackType  # 异常回溯类型

    from tables import (  # 导入PyTables相关类和函数
        Col,  # 列对象
        File,  # 文件对象
        Node,  # 节点对象
    )

    from pandas._typing import (  # 导入Pandas类型提示相关
        AnyArrayLike,  # 任意类数组
        ArrayLike,  # 类数组
        AxisInt,  # 轴整数
        DtypeArg,  # 数据类型参数
        FilePath,  # 文件路径
        Self,  # 自引用类型
        Shape,  # 形状
        npt,  # NumPy类型
    )

    from pandas.core.internals.blocks import Block  # 导入数据块

# 版本信息
_version = "0.15.2"

# 默认编码
_default_encoding = "UTF-8"


def _ensure_encoding(encoding: str | None) -> str:
    # 如果编码为None，则使用默认编码
    if encoding is None:
        encoding = _default_encoding

    return encoding


def _ensure_str(name):
    """
    Ensure that an index / column name is a str (python 3); otherwise they
    may be np.string dtype. Non-string dtypes are passed through unchanged.
    """
    # 如果 name 是字符串类型，强制转换为字符串
    if isinstance(name, str):
        name = str(name)
    # 返回处理后的 name 变量
    return name
# 将 PyTablesExpr 赋值给 Term，用于表达式操作
Term = PyTablesExpr

# 确保 where 参数是 Term 或 Term 列表
def _ensure_term(where, scope_level: int):
    """
    Ensure that the where is a Term or a list of Term.

    This makes sure that we are capturing the scope of variables that are
    passed create the terms here with a frame_level=2 (we are 2 levels down)
    """
    # 计算当前作用域的级别
    level = scope_level + 1
    # 如果 where 是列表或元组，则将其中每个元素转换为 Term 对象，并递归处理可能是表达式的 term
    if isinstance(where, (list, tuple)):
        where = [
            Term(term, scope_level=level + 1) if maybe_expression(term) else term
            for term in where
            if term is not None
        ]
    # 如果 where 可能是表达式，则将其转换为 Term 对象
    elif maybe_expression(where):
        where = Term(where, scope_level=level)
    # 返回处理后的 where，如果其为 None 或长度为零则返回 None
    return where if where is None or len(where) else None

# 定义当版本太旧或未定义时忽略 where 条件的文档字符串模板
incompatibility_doc: Final = """
where criteria is being ignored as this version [%s] is too old (or
not-defined), read the file in and write it out to a new file to upgrade (with
the copy_to method)
"""

# 定义属性冲突时重置属性为 None 的文档字符串模板
attribute_conflict_doc: Final = """
the [%s] attribute of the existing index is [%s] which conflicts with the new
[%s], resetting the attribute to None
"""

# 定义 PyTables 性能警告的文档字符串模板
performance_doc: Final = """
your performance may suffer as PyTables will pickle object types that it cannot
map directly to c-types [inferred_type->%s,key->%s] [items->%s]
"""

# 定义格式映射字典
_FORMAT_MAP = {"f": "fixed", "fixed": "fixed", "t": "table", "table": "table"}

# 定义 DataFrame 对应的轴映射列表
_AXES_MAP = {DataFrame: [0]}

# 注册配置选项，指定是否在将数据附加到表时删除所有包含 NaN 的行
dropna_doc: Final = """
: boolean
    drop ALL nan rows when appending to a table
"""

# 注册配置选项，指定默认的写入格式，支持 'fixed'、'table' 或 None
format_doc: Final = """
: format
    default format writing format, if None, then
    put will default to 'fixed' and append will default to 'table'
"""

# 配置项的上下文管理器，用于注册 HDF 存储相关的配置选项
with config.config_prefix("io.hdf"):
    # 注册是否在将数据附加到表时删除所有包含 NaN 的行的配置选项
    config.register_option("dropna_table", False, dropna_doc, validator=config.is_bool)
    # 注册默认的数据写入格式配置选项，支持 'fixed'、'table' 或 None
    config.register_option(
        "default_format",
        None,
        format_doc,
        validator=config.is_one_of_factory(["fixed", "table", None]),
    )

# 初始化 PyTables 模块和相关变量
_table_mod = None
_table_file_open_policy_is_strict = False

# 返回 PyTables 模块对象，如果未加载则进行加载
def _tables():
    global _table_mod
    global _table_file_open_policy_is_strict
    if _table_mod is None:
        import tables

        _table_mod = tables

        # 设置文件打开策略，如果 PyTables 版本 >= 3.1，则设置为严格模式
        with suppress(AttributeError):
            _table_file_open_policy_is_strict = (
                tables.file._FILE_OPEN_POLICY == "strict"
            )

    return _table_mod

# HDF 存储接口函数 to_hdf 的定义与参数说明
def to_hdf(
    path_or_buf: FilePath | HDFStore,
    key: str,
    value: DataFrame | Series,
    mode: str = "a",
    complevel: int | None = None,
    complib: str | None = None,
    append: bool = False,
    format: str | None = None,
    index: bool = True,
    min_itemsize: int | dict[str, int] | None = None,
    nan_rep=None,
    dropna: bool | None = None,
    data_columns: Literal[True] | list[str] | None = None,
    # data_columns 是一个参数，可以是 Literal 类型中的 True，也可以是字符串列表，或者可以是 None。默认为 None。
    errors: str = "strict",
    # errors 是一个参数，用于指定在解码过程中遇到错误时的处理方式，默认为 "strict"，即严格模式。
    encoding: str = "UTF-8",
    # encoding 是一个参数，指定要使用的字符编码方式，默认为 UTF-8。
# 定义一个函数，用于将数据存储到 HDF5 文件中，并在需要时关闭文件
def store(
    key,
    value,
    format=None,
    index=None,
    min_itemsize=None,
    nan_rep=None,
    dropna=None,
    data_columns=None,
    errors='strict',
    encoding=None,
    append=False,
) -> None:
    """store this object, close it if we opened it"""
    # 如果需要追加数据到已有键中，则使用 append 方法
    if append:
        f = lambda store: store.append(
            key,
            value,
            format=format,
            index=index,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            dropna=dropna,
            data_columns=data_columns,
            errors=errors,
            encoding=encoding,
        )
    else:
        # 否则使用 put 方法，注意 dropna 参数不会传递给 put 方法
        # NB: dropna is not passed to `put`
        f = lambda store: store.put(
            key,
            value,
            format=format,
            index=index,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            data_columns=data_columns,
            errors=errors,
            encoding=encoding,
            dropna=dropna,
        )

    # 如果 path_or_buf 是 HDFStore 对象，则直接调用 f 函数
    if isinstance(path_or_buf, HDFStore):
        f(path_or_buf)
    else:
        # 否则将 path_or_buf 转换为字符串路径
        path_or_buf = stringify_path(path_or_buf)
        # 使用 HDFStore 打开路径，并使用指定的模式和压缩参数
        with HDFStore(
            path_or_buf, mode=mode, complevel=complevel, complib=complib
        ) as store:
            # 调用 f 函数来存储数据
            f(store)


# 定义一个函数，用于从 HDF5 文件中读取数据
def read_hdf(
    path_or_buf: FilePath | HDFStore,
    key=None,
    mode: str = "r",
    errors: str = "strict",
    where: str | list | None = None,
    start: int | None = None,
    stop: int | None = None,
    columns: list[str] | None = None,
    iterator: bool = False,
    chunksize: int | None = None,
    **kwargs,
):
    """
    Read from the store, close it if we opened it.

    Retrieve pandas object stored in file, optionally based on where
    criteria.

    .. warning::

       Pandas uses PyTables for reading and writing HDF5 files, which allows
       serializing object-dtype data with pickle when using the "fixed" format.
       Loading pickled data received from untrusted sources can be unsafe.

       See: https://docs.python.org/3/library/pickle.html for more.

    Parameters
    ----------
    path_or_buf : str, path object, pandas.HDFStore
        Any valid string path is acceptable. Only supports the local file system,
        remote URLs and file-like objects are not supported.

        If you want to pass in a path object, pandas accepts any
        ``os.PathLike``.

        Alternatively, pandas accepts an open :class:`pandas.HDFStore` object.

    key : object, optional
        The group identifier in the store. Can be omitted if the HDF file
        contains a single pandas object.
    mode : {'r', 'r+', 'a'}, default 'r'
        Mode to use when opening the file. Ignored if path_or_buf is a
        :class:`pandas.HDFStore`. Default is 'r'.
    errors : str, default 'strict'
        Specifies how encoding and decoding errors are to be handled.
        See the errors argument for :func:`open` for a full list
        of options.
    where : list, optional
        A list of Term (or convertible) objects.
    start : int, optional
        Row number to start selection.
    stop : int, optional
        Row number to stop selection.
    """
    # columns参数：要返回的列名列表，可选
    columns : list, optional
    # iterator参数：是否返回迭代器对象，可选
    iterator : bool, optional
    # chunksize参数：使用迭代器时每次迭代中包含的行数，可选
    chunksize : int, optional
    # **kwargs：传递给HDFStore的额外关键字参数
    **kwargs
    # 返回值
    # 选择的对象。返回类型取决于存储的对象类型。
    Returns
    -------
    object

    # 参见
    # --------
    # DataFrame.to_hdf：从DataFrame写入HDF文件。
    # HDFStore：HDF文件的低级访问。

    See Also
    --------
    DataFrame.to_hdf : Write a HDF file from a DataFrame.
    HDFStore : Low-level access to HDF files.

    # 示例
    # --------
    # >>> df = pd.DataFrame([[1, 1.0, "a"]], columns=["x", "y", "z"])  # doctest: +SKIP
    # >>> df.to_hdf("./store.h5", "data")  # doctest: +SKIP
    # >>> reread = pd.read_hdf("./store.h5")  # doctest: +SKIP
    """
    # 如果模式不是"r", "r+"或"a"中的一种，抛出值错误
    if mode not in ["r", "r+", "a"]:
        raise ValueError(
            f"mode {mode} is not allowed while performing a read. "
            f"Allowed modes are r, r+ and a."
        )
    # 如果where不是None，则确保它是一个条件表达式
    if where is not None:
        where = _ensure_term(where, scope_level=1)

    # 如果path_or_buf是HDFStore对象
    if isinstance(path_or_buf, HDFStore):
        # 如果HDFStore对象未打开，抛出操作系统错误
        if not path_or_buf.is_open:
            raise OSError("The HDFStore must be open for reading.")

        # 设置store为path_or_buf
        store = path_or_buf
        # 禁止自动关闭
        auto_close = False
    else:
        # 将path_or_buf转换为字符串表示形式
        path_or_buf = stringify_path(path_or_buf)
        # 如果path_or_buf不是字符串，抛出未实现错误
        if not isinstance(path_or_buf, str):
            raise NotImplementedError(
                "Support for generic buffers has not been implemented."
            )
        try:
            # 检查路径是否存在
            exists = os.path.exists(path_or_buf)

        # 如果文件路径太长
        except (TypeError, ValueError):
            exists = False

        # 如果路径不存在，抛出文件未找到错误
        if not exists:
            raise FileNotFoundError(f"File {path_or_buf} does not exist")

        # 打开HDFStore对象，设置store为HDFStore对象
        store = HDFStore(path_or_buf, mode=mode, errors=errors, **kwargs)
        # 如果使用迭代器，则无法自动打开/关闭，委托给迭代器处理
        # 因此设置自动关闭为True
        auto_close = True
    # 尝试执行以下代码块，捕获可能的异常：值错误、类型错误、查找错误
    try:
        # 如果未提供键值（key），则获取存储中的所有组
        if key is None:
            groups = store.groups()
            # 如果组的数量为0，则抛出值错误
            if len(groups) == 0:
                raise ValueError(
                    "Dataset(s) incompatible with Pandas data types, "
                    "not table, or no datasets found in HDF5 file."
                )
            # 将第一个组设为候选组
            candidate_only_group = groups[0]

            # 对于只包含一个数据集的 HDF 文件，其它所有组应该是候选组的元数据组。
            # （这假设 groups() 方法会先列出父组，然后是它们的子组。）
            for group_to_check in groups[1:]:
                # 如果某个组不是候选组的元数据，则抛出值错误
                if not _is_metadata_of(group_to_check, candidate_only_group):
                    raise ValueError(
                        "key must be provided when HDF5 "
                        "file contains multiple datasets."
                    )
            # 将候选组的路径作为键值（key）
            key = candidate_only_group._v_pathname
        
        # 使用指定的键值（key）从存储中选择数据集
        return store.select(
            key,
            where=where,
            start=start,
            stop=stop,
            columns=columns,
            iterator=iterator,
            chunksize=chunksize,
            auto_close=auto_close,
        )
    
    # 捕获可能的异常类型：值错误、类型错误、查找错误
    except (ValueError, TypeError, LookupError):
        # 如果 path_or_buf 不是 HDFStore 的实例，则关闭存储（如果它是我们打开的）
        if not isinstance(path_or_buf, HDFStore):
            with suppress(AttributeError):
                store.close()  # 如果发生错误，关闭存储

        # 重新抛出捕获到的异常
        raise
# 检查给定的节点组是否是指定父节点组的元数据组
def _is_metadata_of(group: Node, parent_group: Node) -> bool:
    if group._v_depth <= parent_group._v_depth:
        return False

    current = group
    # 从当前节点往上遍历，直到根节点（_v_depth > 1），查找是否有符合条件的元数据组
    while current._v_depth > 1:
        parent = current._v_parent
        # 如果找到了符合条件的元数据组，返回True
        if parent == parent_group and current._v_name == "meta":
            return True
        current = current._v_parent
    # 如果没有找到符合条件的元数据组，返回False
    return False


class HDFStore:
    """
    Dict-like IO interface for storing pandas objects in PyTables.

    Either Fixed or Table format.

    .. warning::

       Pandas uses PyTables for reading and writing HDF5 files, which allows
       serializing object-dtype data with pickle when using the "fixed" format.
       Loading pickled data received from untrusted sources can be unsafe.

       See: https://docs.python.org/3/library/pickle.html for more.

    Parameters
    ----------
    path : str
        File path to HDF5 file.
    mode : {'a', 'w', 'r', 'r+'}, default 'a'

        ``'r'``
            Read-only; no data can be modified.
        ``'w'``
            Write; a new file is created (an existing file with the same
            name would be deleted).
        ``'a'``
            Append; an existing file is opened for reading and writing,
            and if the file does not exist it is created.
        ``'r+'``
            It is similar to ``'a'``, but the file must already exist.
    complevel : int, 0-9, default None
        Specifies a compression level for data.
        A value of 0 or None disables compression.
    complib : {'zlib', 'lzo', 'bzip2', 'blosc'}, default 'zlib'
        Specifies the compression library to be used.
        These additional compressors for Blosc are supported
        (default if no compressor specified: 'blosc:blosclz'):
        {'blosc:blosclz', 'blosc:lz4', 'blosc:lz4hc', 'blosc:snappy',
         'blosc:zlib', 'blosc:zstd'}.
        Specifying a compression library which is not available issues
        a ValueError.
    fletcher32 : bool, default False
        If applying compression use the fletcher32 checksum.
    **kwargs
        These parameters will be passed to the PyTables open_file method.

    Examples
    --------
    >>> bar = pd.DataFrame(np.random.randn(10, 4))
    >>> store = pd.HDFStore("test.h5")
    >>> store["foo"] = bar  # write to HDF5
    >>> bar = store["foo"]  # retrieve
    >>> store.close()

    **Create or load HDF5 file in-memory**

    When passing the `driver` option to the PyTables open_file method through
    **kwargs, the HDF5 file is loaded or created in-memory and will only be
    written when closed:

    >>> bar = pd.DataFrame(np.random.randn(10, 4))
    >>> store = pd.HDFStore("test.h5", driver="H5FD_CORE")
    >>> store["foo"] = bar
    >>> store.close()  # only now, data is written to disk
    """

    _handle: File | None
    _mode: str
    # 初始化方法，设置 HDFStore 对象的属性
    def __init__(
        self,
        path,
        mode: str = "a",
        complevel: int | None = None,
        complib=None,
        fletcher32: bool = False,
        **kwargs,
    ) -> None:
        # 如果传入参数中包含 "format"，则抛出数值错误
        if "format" in kwargs:
            raise ValueError("format is not a defined argument for HDFStore")

        # 导入 tables 库
        tables = import_optional_dependency("tables")

        # 如果 complib 不为空且不在 tables.filters.all_complibs 中，则抛出数值错误
        if complib is not None and complib not in tables.filters.all_complibs:
            raise ValueError(
                f"complib only supports {tables.filters.all_complibs} compression."
            )

        # 如果 complib 为空且 complevel 不为空，则使用 tables.filters.default_complib
        if complib is None and complevel is not None:
            complib = tables.filters.default_complib

        # 将 path 转换为字符串
        self._path = stringify_path(path)
        # 如果 mode 为空，则设置为 "a"
        if mode is None:
            mode = "a"
        self._mode = mode
        self._handle = None
        self._complevel = complevel if complevel else 0
        self._complib = complib
        self._fletcher32 = fletcher32
        self._filters = None
        # 打开 HDFStore 对象
        self.open(mode=mode, **kwargs)

    # 返回 HDFStore 对象的路径
    def __fspath__(self) -> str:
        return self._path

    # 返回 HDFStore 对象的根节点
    @property
    def root(self):
        """return the root node"""
        self._check_if_open()
        assert self._handle is not None  # for mypy
        return self._handle.root

    # 返回 HDFStore 对象的文件名
    @property
    def filename(self) -> str:
        return self._path

    # 获取指定键的值
    def __getitem__(self, key: str):
        return self.get(key)

    # 设置指定键的值
    def __setitem__(self, key: str, value) -> None:
        self.put(key, value)

    # 删除指定键的值
    def __delitem__(self, key: str) -> int | None:
        return self.remove(key)

    # 允许通过属性访问获取存储
    def __getattr__(self, name: str):
        """allow attribute access to get stores"""
        try:
            return self.get(name)
        except (KeyError, ClosedFileError):
            pass
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    # 检查键是否存在
    def __contains__(self, key: str) -> bool:
        """
        check for existence of this key
        can match the exact pathname or the pathnm w/o the leading '/'
        """
        node = self.get_node(key)
        if node is not None:
            name = node._v_pathname
            if key in (name, name[1:]):
                return True
        return False

    # 返回 HDFStore 对象中的组数
    def __len__(self) -> int:
        return len(self.groups())

    # 返回 HDFStore 对象的字符串表示形式
    def __repr__(self) -> str:
        pstr = pprint_thing(self._path)
        return f"{type(self)}\nFile path: {pstr}\n"

    # 进入上下文管理器时返回自身
    def __enter__(self) -> Self:
        return self

    # 退出上下文管理器时关闭 HDFStore 对象
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
    def keys(self, include: str = "pandas") -> list[str]:
        """
        Return a list of keys corresponding to objects stored in HDFStore.

        Parameters
        ----------
        include : str, default 'pandas'
            Determines the type of objects to include in the result:
            - 'pandas': Returns paths of pandas objects.
            - 'native': Returns paths of native HDF5 Table objects.

        Returns
        -------
        list
            List of ABSOLUTE path-names (e.g. have the leading '/').

        Raises
        ------
        ValueError
            If `include` has an illegal value.

        See Also
        --------
        HDFStore.info : Prints detailed information on the store.
        HDFStore.get_node : Returns the node with the key.
        HDFStore.get_storer : Returns the storer object for a key.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data", df)  # doctest: +SKIP
        >>> store.get("data")  # doctest: +SKIP
        >>> print(store.keys())  # doctest: +SKIP
        ['/data1', '/data2']
        >>> store.close()  # doctest: +SKIP
        """
        if include == "pandas":
            # Return paths of pandas objects in the store
            return [n._v_pathname for n in self.groups()]
        elif include == "native":
            # Ensure the HDF5 handle exists
            assert self._handle is not None  # mypy
            # Return paths of native HDF5 Table objects in the store
            return [
                n._v_pathname for n in self._handle.walk_nodes("/", classname="Table")
            ]
        # Raise an error for invalid `include` value
        raise ValueError(
            f"`include` should be either 'pandas' or 'native' but is '{include}'"
        )

    def __iter__(self) -> Iterator[str]:
        """
        Returns an iterator over the keys of the HDFStore object.
        """
        return iter(self.keys())

    def items(self) -> Iterator[tuple[str, list]]:
        """
        Returns an iterator yielding tuples of (key, group) for each group in the HDFStore.
        """
        for g in self.groups():
            yield g._v_pathname, g
    # 打开文件，根据给定的模式打开文件
    def open(self, mode: str = "a", **kwargs) -> None:
        """
        Open the file in the specified mode

        Parameters
        ----------
        mode : {'a', 'w', 'r', 'r+'}, default 'a'
            See HDFStore docstring or tables.open_file for info about modes
        **kwargs
            These parameters will be passed to the PyTables open_file method.
        """
        # 导入 PyTables 的相关模块
        tables = _tables()

        # 如果当前模式与要求的模式不同
        if self._mode != mode:
            # 如果从写入模式切换到读取模式，允许
            if self._mode in ["a", "w"] and mode in ["r", "r+"]:
                pass
            elif mode in ["w"]:
                # 如果文件已经打开，则引发可能导致数据丢失的错误
                if self.is_open:
                    raise PossibleDataLossError(
                        f"Re-opening the file [{self._path}] with mode [{self._mode}] "
                        "will delete the current file!"
                    )

            # 更新当前对象的模式为指定模式
            self._mode = mode

        # 如果文件已经打开，先关闭
        if self.is_open:
            self.close()

        # 如果设置了压缩等级并且大于0，设置相应的过滤器
        if self._complevel and self._complevel > 0:
            self._filters = _tables().Filters(
                self._complevel, self._complib, fletcher32=self._fletcher32
            )

        # 如果文件打开策略是严格的，并且文件已经打开，则引发错误
        if _table_file_open_policy_is_strict and self.is_open:
            msg = (
                "Cannot open HDF5 file, which is already opened, "
                "even in read-only mode."
            )
            raise ValueError(msg)

        # 使用 PyTables 的 open_file 方法打开文件
        self._handle = tables.open_file(self._path, self._mode, **kwargs)

    # 关闭文件，关闭 PyTables 文件句柄
    def close(self) -> None:
        """
        Close the PyTables file handle
        """
        # 如果文件句柄不为空，则关闭文件句柄
        if self._handle is not None:
            self._handle.close()
        # 将文件句柄置为空
        self._handle = None

    # 属性方法，返回文件是否打开的布尔值
    @property
    def is_open(self) -> bool:
        """
        return a boolean indicating whether the file is open
        """
        # 如果文件句柄为空，则返回 False；否则返回文件句柄的 isopen 属性的布尔值
        if self._handle is None:
            return False
        return bool(self._handle.isopen)

    # 刷新方法，将所有缓冲的修改写入磁盘
    def flush(self, fsync: bool = False) -> None:
        """
        Force all buffered modifications to be written to disk.

        Parameters
        ----------
        fsync : bool (default False)
          call ``os.fsync()`` on the file handle to force writing to disk.

        Notes
        -----
        Without ``fsync=True``, flushing may not guarantee that the OS writes
        to disk. With fsync, the operation will block until the OS claims the
        file has been written; however, other caching layers may still
        interfere.
        """
        # 如果文件句柄不为空
        if self._handle is not None:
            # 刷新文件句柄
            self._handle.flush()
            # 如果 fsync 参数为 True，则调用 os.fsync() 强制写入磁盘
            if fsync:
                with suppress(OSError):
                    os.fsync(self._handle.fileno())
    def get(self, key: str):
        """
        Retrieve pandas object stored in file.

        Parameters
        ----------
        key : str
            Object to retrieve from file. Raises KeyError if not found.

        Returns
        -------
        object
            Same type as object stored in file.

        See Also
        --------
        HDFStore.get_node : Returns the node with the key.
        HDFStore.get_storer : Returns the storer object for a key.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data", df)  # doctest: +SKIP
        >>> store.get("data")  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        """
        with patch_pickle():
            # GH#31167 Without this patch, pickle doesn't know how to unpickle
            #  old DateOffset objects now that they are cdef classes.
            # 获取指定键值对应的节点
            group = self.get_node(key)
            # 如果节点不存在，则抛出 KeyError 异常
            if group is None:
                raise KeyError(f"No object named {key} in the file")
            # 读取并返回节点中存储的数据对象
            return self._read_group(group)

    def select(
        self,
        key: str,
        where=None,
        start=None,
        stop=None,
        columns=None,
        iterator: bool = False,
        chunksize: int | None = None,
        auto_close: bool = False,
        """
        Retrieve pandas object stored in file, optionally based on where criteria.

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.

        Parameters
        ----------
        key : str
            Object being retrieved from file.
        where : list or None
            List of Term (or convertible) objects, optional.
        start : int or None
            Row number to start selection.
        stop : int, default None
            Row number to stop selection.
        columns : list or None
            A list of columns that if not None, will limit the return columns.
        iterator : bool or False
            Returns an iterator.
        chunksize : int or None
            Number or rows to include in iteration, return an iterator.
        auto_close : bool or False
            Should automatically close the store when finished.

        Returns
        -------
        object
            Retrieved object from file.

        See Also
        --------
        HDFStore.select_as_coordinates : Returns the selection as an index.
        HDFStore.select_column : Returns a single column from the table.
        HDFStore.select_as_multiple : Retrieves pandas objects from multiple tables.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data", df)  # doctest: +SKIP
        >>> store.get("data")  # doctest: +SKIP
        >>> print(store.keys())  # doctest: +SKIP
        ['/data1', '/data2']
        >>> store.select("/data1")  # doctest: +SKIP
           A  B
        0  1  2
        1  3  4
        >>> store.select("/data1", where="columns == A")  # doctest: +SKIP
           A
        0  1
        1  3
        >>> store.close()  # doctest: +SKIP
        """
        # 获取指定 key 对应的节点对象
        group = self.get_node(key)
        if group is None:
            # 如果节点对象不存在，则抛出 KeyError 异常
            raise KeyError(f"No object named {key} in the file")

        # 创建存储器和轴
        where = _ensure_term(where, scope_level=1)
        # 根据节点对象创建存储器对象
        s = self._create_storer(group)
        # 推断轴信息
        s.infer_axes()

        # 定义在迭代时调用的函数
        def func(_start, _stop, _where):
            return s.read(start=_start, stop=_stop, where=_where, columns=columns)

        # 创建迭代器对象
        it = TableIterator(
            self,
            s,
            func,
            where=where,
            nrows=s.nrows,
            start=start,
            stop=stop,
            iterator=iterator,
            chunksize=chunksize,
            auto_close=auto_close,
        )

        # 返回迭代器对象的结果
        return it.get_result()
    def select_as_coordinates(
        self,
        key: str,
        where=None,
        start: int | None = None,
        stop: int | None = None,
    ):
        """
        return the selection as an Index

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.


        Parameters
        ----------
        key : str
            The key (identifier) for the HDF5 store.
        where : list of Term (or convertible) objects, optional
            Condition to select rows.
        start : int or None, default None
            Start index for selection.
        stop  : int or None, default None
            Stop index for selection.
        """
        where = _ensure_term(where, scope_level=1)  # Ensure 'where' is a valid Term object or convertible
        tbl = self.get_storer(key)  # Retrieve the HDF5 table object for the given key
        if not isinstance(tbl, Table):
            raise TypeError("can only read_coordinates with a table")  # Raise error if the object is not a Table
        return tbl.read_coordinates(where=where, start=start, stop=stop)  # Read coordinates from the table


    def select_column(
        self,
        key: str,
        column: str,
        start: int | None = None,
        stop: int | None = None,
    ):
        """
        return a single column from the table. This is generally only useful to
        select an indexable

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.

        Parameters
        ----------
        key : str
            The key (identifier) for the HDF5 store.
        column : str
            The column name to be selected.
        start : int or None, default None
            Start index for selection.
        stop : int or None, default None
            Stop index for selection.

        Raises
        ------
        raises KeyError if the column is not found (or key is not a valid
            store)
        raises ValueError if the column can not be extracted individually (it
            is part of a data block)
        """
        tbl = self.get_storer(key)  # Retrieve the HDF5 table object for the given key
        if not isinstance(tbl, Table):
            raise TypeError("can only read_column with a table")  # Raise error if the object is not a Table
        return tbl.read_column(column=column, start=start, stop=stop)  # Read specified column from the table


    def select_as_multiple(
        self,
        keys,
        where=None,
        selector=None,
        columns=None,
        start=None,
        stop=None,
        iterator: bool = False,
        chunksize: int | None = None,
        auto_close: bool = False,
    ):
        """
        return a generator to read multiple sets of rows

        Parameters
        ----------
        keys : list
            List of keys (identifiers) for the HDF5 stores.
        where : list of Term (or convertible) objects, optional
            Condition to select rows.
        selector : callable, optional
            Function to apply to each result, defaults to None.
        columns : list, optional
            List of columns to select from each store, defaults to None.
        start : int or None, optional
            Start index for selection.
        stop : int or None, optional
            Stop index for selection.
        iterator : bool, default False
            If True, returns an iterator; otherwise, returns a list.
        chunksize : int or None, optional
            Size of chunks to read, defaults to None.
        auto_close : bool, default False
            If True, automatically closes the HDF5 store after reading.

        Returns
        -------
        Generator or list
            Depending on the 'iterator' parameter, returns either a generator or a list.
        """
        # Implementation details depend on the specific behavior required for selecting multiple data from HDF5 stores.
        pass
    def put(
        self,
        key: str,
        value: DataFrame | Series,
        format=None,
        index: bool = True,
        append: bool = False,
        complib=None,
        complevel: int | None = None,
        min_itemsize: int | dict[str, int] | None = None,
        nan_rep=None,
        data_columns: Literal[True] | list[str] | None = None,
        encoding=None,
        errors: str = "strict",
        track_times: bool = True,
        dropna: bool = False,
    ):
        """
        Store pandas object in the HDFStore

        Parameters
        ----------
        key : str
            The key under which the object should be stored
        value : DataFrame or Series
            The pandas object to store
        format : str, optional
            The storage format of the object (default: None)
        index : bool, optional
            Whether to write the index (default: True)
        append : bool, optional
            Whether to append to the existing HDFStore (default: False)
        complib : str, optional
            The compression library to use (default: None)
        complevel : int or None, optional
            The level of compression (default: None)
        min_itemsize : int or dict, optional
            Minimum size of items (default: None)
        nan_rep : any, optional
            The representation for NaN values (default: None)
        data_columns : bool or list of str or None, optional
            Whether to create data columns (default: None)
        encoding : str, optional
            The encoding to use (default: None)
        errors : str, optional
            The error handling scheme (default: "strict")
        track_times : bool, optional
            Whether to track times (default: True)
        dropna : bool, optional
            Whether to drop NaN elements (default: False)

        Notes
        -----
        This method stores a pandas DataFrame or Series object into an HDF5 file.
        If `append` is True, the object is appended to the existing file under `key`.
        """



    def remove(self, key: str, where=None, start=None, stop=None) -> int | None:
        """
        Remove pandas object partially by specifying the where condition

        Parameters
        ----------
        key : str
            Node to remove or delete rows from
        where : list of Term (or convertible) objects, optional
            Condition to select rows to remove (default: None)
        start : integer, optional
            Row number to start selection (default: None)
        stop : integer, optional
            Row number to stop selection (default: None)

        Returns
        -------
        int or None
            Number of rows removed, or None if not a Table

        Raises
        ------
        KeyError
            If `key` is not a valid store
        ValueError
            If trying to remove a node with a non-None where clause
        """
        where = _ensure_term(where, scope_level=1)  # Ensure `where` is a valid term

        try:
            s = self.get_storer(key)  # Get the storage object for `key`
        except KeyError:
            # If `key` is not found, raise KeyError
            raise
        except AssertionError:
            # Catch assertion errors and raise them
            raise
        except Exception as err:
            # Catch other exceptions (e.g., ClosedFileError, TypeError)
            # and handle specific cases
            if where is not None:
                # If `where` is not None, it's invalid to remove with a where clause
                raise ValueError("trying to remove a node with a non-None where clause!") from err

            # If trying to remove a node (with children), perform recursive removal
            node = self.get_node(key)
            if node is not None:
                node._f_remove(recursive=True)
                return None

        # If `where`, `start`, and `stop` are all None, remove the entire group
        if com.all_none(where, start, stop):
            s.group._f_remove(recursive=True)
            return None

        # If `s` is not a table, raise an error (can only remove from tables)
        if not s.is_table:
            raise ValueError("can only remove with where on objects written as tables")

        # Delete rows from the table according to specified conditions
        return s.delete(where=where, start=start, stop=stop)



    def append(
        self,
        key: str,
        value: DataFrame | Series,
        format=None,
        axes=None,
        index: bool | list[str] = True,
        append: bool = True,
        complib=None,
        complevel: int | None = None,
        columns=None,
        min_itemsize: int | dict[str, int] | None = None,
        nan_rep=None,
        chunksize: int | None = None,
        expectedrows=None,
        dropna: bool | None = None,
        data_columns: Literal[True] | list[str] | None = None,
        encoding=None,
        errors: str = "strict",
    ):
        """
        Append pandas object to the HDFStore

        Parameters
        ----------
        key : str
            The key under which the object should be stored
        value : DataFrame or Series
            The pandas object to append
        format : str, optional
            The storage format of the object (default: None)
        axes : list of Axis, optional
            The axes to use (default: None)
        index : bool or list of str, optional
            Whether to write the index (default: True)
        append : bool, optional
            Whether to append to the existing HDFStore (default: True)
        complib : str, optional
            The compression library to use (default: None)
        complevel : int or None, optional
            The level of compression (default: None)
        columns : list of str, optional
            The columns to write (default: None)
        min_itemsize : int or dict, optional
            Minimum size of items (default: None)
        nan_rep : any, optional
            The representation for NaN values (default: None)
        chunksize : int or None, optional
            The chunksize to use (default: None)
        expectedrows : int, optional
            The expected number of rows (default: None)
        dropna : bool or None, optional
            Whether to drop NaN elements (default: None)
        data_columns : bool or list of str or None, optional
            Whether to create data columns (default: None)
        encoding : str, optional
            The encoding to use (default: None)
        errors : str, optional
            The error handling scheme (default: "strict")

        Notes
        -----
        This method appends a pandas DataFrame or Series object to an existing HDF5 file.
        """
    def append_to_multiple(
        self,
        d: dict,
        value,
        selector,
        data_columns=None,
        axes=None,
        dropna: bool = False,
        **kwargs,
    ):
        """
        Append a value to multiple elements in a dictionary based on a selector.

        Parameters
        ----------
        d : dict
            Dictionary where values are appended.
        value : object
            Value to append to selected elements.
        selector : callable
            Function or callable used to select elements in the dictionary.
        data_columns : None or list, optional
            Columns to apply the selector on, if applicable.
        axes : None or list, optional
            Axes to consider for selection.
        dropna : bool, default False
            Whether to drop NA/null values before appending.
        **kwargs : dict
            Additional keyword arguments for future extensions.
        """
        # Version requirement check
        _tables()
        
        # Get the storer for the specified key
        s = self.get_storer(key)
        
        # If storer is not found, return without creating an index
        if s is None:
            return
        
        # Raise an error if the storer is not of type Table
        if not isinstance(s, Table):
            raise TypeError("cannot create table index on a Fixed format store")
        
        # Create index on specified columns with given options
        s.create_index(columns=columns, optlevel=optlevel, kind=kind)

    def groups(self) -> list:
        """
        Return a list of top-level nodes in the HDFStore.

        Each node returned is not a pandas storage object.

        Returns
        -------
        list
            List of objects representing top-level nodes in the HDFStore.

        See Also
        --------
        HDFStore.get_node : Returns the node with the key.
        """
        # Version requirement check
        _tables()
        
        # Ensure that the HDFStore is open
        self._check_if_open()
        
        # Assertion checks for type safety (for mypy)
        assert self._handle is not None
        assert _table_mod is not None
        
        # Return a filtered list of top-level groups excluding certain types
        return [
            g
            for g in self._handle.walk_groups()
            if (
                not isinstance(g, _table_mod.link.Link)
                and (
                    getattr(g._v_attrs, "pandas_type", None)
                    or getattr(g, "table", None)
                    or (isinstance(g, _table_mod.table.Table) and g._v_name != "table")
                )
            )
        ]
    def walk(self, where: str = "/") -> Iterator[tuple[str, list[str], list[str]]]:
        """
        Walk the pytables group hierarchy for pandas objects.

        This generator will yield the group path, subgroups and pandas object
        names for each group.

        Any non-pandas PyTables objects that are not a group will be ignored.

        The `where` group itself is listed first (preorder), then each of its
        child groups (following an alphanumerical order) is also traversed,
        following the same procedure.

        Parameters
        ----------
        where : str, default "/"
            Group where to start walking.

        Yields
        ------
        path : str
            Full path to a group (without trailing '/').
        groups : list
            Names (strings) of the groups contained in `path`.
        leaves : list
            Names (strings) of the pandas objects contained in `path`.

        See Also
        --------
        HDFStore.info : Prints detailed information on the store.

        Examples
        --------
        >>> df1 = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        >>> store = pd.HDFStore("store.h5", "w")  # doctest: +SKIP
        >>> store.put("data", df1, format="table")  # doctest: +SKIP
        >>> df2 = pd.DataFrame([[5, 6], [7, 8]], columns=["A", "B"])
        >>> store.append("data", df2)  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        >>> for group in store.walk():  # doctest: +SKIP
        ...     print(group)  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        """
        # 调用私有方法 _tables()
        _tables()
        # 检查 HDFStore 是否已经打开
        self._check_if_open()
        # 断言 HDF 文件句柄不为 None，用于类型检查（mypy）
        assert self._handle is not None  # for mypy
        # 断言 _table_mod 不为 None，用于类型检查（mypy）
        assert _table_mod is not None  # for mypy

        # 遍历指定路径下的所有分组
        for g in self._handle.walk_groups(where):
            # 如果 g 不是 pandas 类型的 PyTables 对象，则跳过
            if getattr(g._v_attrs, "pandas_type", None) is not None:
                continue

            # 初始化存储子组和叶子对象名的列表
            groups = []
            leaves = []

            # 遍历当前分组的所有子节点
            for child in g._v_children.values():
                # 获取子节点的 pandas 类型属性
                pandas_type = getattr(child._v_attrs, "pandas_type", None)
                # 如果子节点不是 pandas 对象且是一个分组，则添加到 groups 列表中
                if pandas_type is None:
                    if isinstance(child, _table_mod.group.Group):
                        groups.append(child._v_name)
                # 否则，将其添加到 leaves 列表中
                else:
                    leaves.append(child._v_name)

            # 生成器返回当前分组的完整路径（去除尾部的 '/'）、子组列表和叶子对象列表
            yield (g._v_pathname.rstrip("/"), groups, leaves)

    def get_node(self, key: str) -> Node | None:
        """return the node with the key or None if it does not exist"""
        # 检查 HDFStore 是否已经打开
        self._check_if_open()
        # 如果 key 不以 '/' 开头，则在其前面添加 '/'
        if not key.startswith("/"):
            key = "/" + key

        # 断言 HDF 文件句柄不为 None
        assert self._handle is not None
        # 断言 _table_mod 不为 None，用于类型检查（mypy）
        assert _table_mod is not None  # for mypy
        try:
            # 获取指定 key 对应的节点
            node = self._handle.get_node(self.root, key)
        except _table_mod.exceptions.NoSuchNodeError:
            # 如果节点不存在，则返回 None
            return None

        # 断言返回的节点是 _table_mod.Node 类型
        assert isinstance(node, _table_mod.Node), type(node)
        # 返回获取到的节点
        return node
    # 返回给定键的存储器对象，如果不存在则抛出异常
    def get_storer(self, key: str) -> GenericFixed | Table:
        """return the storer object for a key, raise if not in the file"""
        # 获取存储器对象所在的节点
        group = self.get_node(key)
        # 如果节点不存在，则抛出 KeyError 异常
        if group is None:
            raise KeyError(f"No object named {key} in the file")

        # 创建存储器对象
        s = self._create_storer(group)
        # 推断存储器对象的轴信息
        s.infer_axes()
        return s

    # 复制现有的存储到一个新文件中，并进行原地更新
    def copy(
        self,
        file,
        mode: str = "w",
        propindexes: bool = True,
        keys=None,
        complib=None,
        complevel: int | None = None,
        fletcher32: bool = False,
        overwrite: bool = True,
    ) -> HDFStore:
        """
        Copy the existing store to a new file, updating in place.

        Parameters
        ----------
        propindexes : bool, default True
            Restore indexes in copied file.
        keys : list, optional
            List of keys to include in the copy (defaults to all).
        overwrite : bool, default True
            Whether to overwrite (remove and replace) existing nodes in the new store.
        mode, complib, complevel, fletcher32 same as in HDFStore.__init__

        Returns
        -------
        open file handle of the new store
        """
        # 创建一个新的 HDFStore 对象，用于存储复制后的数据
        new_store = HDFStore(
            file, mode=mode, complib=complib, complevel=complevel, fletcher32=fletcher32
        )
        # 如果未指定 keys，则使用所有键
        if keys is None:
            keys = list(self.keys())
        # 如果 keys 不是列表或元组，则转为列表形式
        if not isinstance(keys, (tuple, list)):
            keys = [keys]
        # 遍历每个指定的键
        for k in keys:
            # 获取键 k 对应的存储器对象
            s = self.get_storer(k)
            # 如果存储器对象存在
            if s is not None:
                # 如果新存储中已存在同名键 k，并且 overwrite 参数为 True，则移除该键
                if k in new_store:
                    if overwrite:
                        new_store.remove(k)

                # 从当前存储中选择键 k 的数据
                data = self.select(k)
                # 如果存储器对象是 Table 类型
                if isinstance(s, Table):
                    # 默认不使用索引
                    index: bool | list[str] = False
                    # 如果 propindexes 参数为 True，则设置索引
                    if propindexes:
                        index = [a.name for a in s.axes if a.is_indexed]
                    # 将数据追加到新存储中，保留数据列和编码方式
                    new_store.append(
                        k,
                        data,
                        index=index,
                        data_columns=getattr(s, "data_columns", None),
                        encoding=s.encoding,
                    )
                else:
                    # 将数据以指定编码方式放入新存储中
                    new_store.put(k, data, encoding=s.encoding)

        # 返回新存储的文件句柄
        return new_store
    # ------------------------------------------------------------------------
    # private methods

    def _check_if_open(self) -> None:
        # 检查存储是否打开，如果未打开则抛出 ClosedFileError 异常
        if not self.is_open:
            raise ClosedFileError(f"{self._path} file is not open!")

    def _validate_format(self, format: str) -> str:
        """validate / deprecate formats"""
        # 验证和处理格式参数
        # 尝试将给定的格式参数转换为标准格式，若失败则抛出 TypeError 异常
        try:
            format = _FORMAT_MAP[format.lower()]
        except KeyError as err:
            raise TypeError(f"invalid HDFStore format specified [{format}]") from err

        return format

    def _create_storer(
        self,
        group,
        format=None,
        value: DataFrame | Series | None = None,
        encoding: str = "UTF-8",
        errors: str = "strict",
        ):
        """
        Create a storer for a given group.

        Parameters
        ----------
        group : object
            Group to create storer for.
        format : str, optional
            Format of the storage. If not specified, default is used.
        value : DataFrame | Series | None, optional
            Data to store. Default is None.
        encoding : str, optional
            Encoding type. Default is 'UTF-8'.
        errors : str, optional
            Error handling scheme. Default is 'strict'.

        Notes
        -----
        This method is used internally to create a storer object for a given group
        in the HDFStore.

        """
        # 创建给定组的 storer 对象
        pass  # Placeholder for actual implementation, not defined here
    def _write_to_group(
        self,
        key: str,
        value: DataFrame | Series,
        format,
        axes=None,
        index: bool | list[str] = True,
        append: bool = False,
        complib=None,
        complevel: int | None = None,
        fletcher32=None,
        min_itemsize: int | dict[str, int] | None = None,
        chunksize: int | None = None,
        expectedrows=None,
        dropna: bool = False,
        nan_rep=None,
        data_columns=None,
        encoding=None,
        errors: str = "strict",
        track_times: bool = True,
    ) -> None:
        # we don't want to store a table node at all if our object is 0-len
        # as there are not dtypes
        # 检查数据是否为空，如果为空且格式要求为表格（table）或者追加模式（append），则直接返回
        if getattr(value, "empty", None) and (format == "table" or append):
            return

        # 确定或创建组对象
        group = self._identify_group(key, append)

        # 创建数据存储器对象
        s = self._create_storer(group, format, value, encoding=encoding, errors=errors)

        # 如果是追加模式，则设置对象信息；否则设置对象信息并检查表格是否存在
        if append:
            # 如果不是表格对象或者是表格对象但是格式要求为"fixed"且已存在，则抛出错误
            if not s.is_table or (s.is_table and format == "fixed" and s.is_exists):
                raise ValueError("Can only append to Tables")
            # 如果表格对象不存在，则设置对象信息
            if not s.is_exists:
                s.set_object_info()
        else:
            s.set_object_info()

        # 如果不是表格对象且指定了压缩库，则抛出错误
        if not s.is_table and complib:
            raise ValueError("Compression not supported on Fixed format stores")

        # 写入对象数据
        s.write(
            obj=value,
            axes=axes,
            append=append,
            complib=complib,
            complevel=complevel,
            fletcher32=fletcher32,
            min_itemsize=min_itemsize,
            chunksize=chunksize,
            expectedrows=expectedrows,
            dropna=dropna,
            nan_rep=nan_rep,
            data_columns=data_columns,
            track_times=track_times,
        )

        # 如果数据存储对象是表格（Table）类型且指定了索引列，则创建索引
        if isinstance(s, Table) and index:
            s.create_index(columns=index)

    def _read_group(self, group: Node):
        # 创建数据存储器对象
        s = self._create_storer(group)
        # 推断数据存储器的轴信息
        s.infer_axes()
        # 读取数据并返回
        return s.read()

    def _identify_group(self, key: str, append: bool) -> Node:
        """Identify HDF5 group based on key, delete/create group if needed."""
        # 获取指定键对应的节点对象
        group = self.get_node(key)

        # 对于类型检查工具（如mypy）的断言；如果没有处理句柄，则已经在get_node调用中抛出错误
        assert self._handle is not None

        # 如果节点存在且非追加模式，则递归删除该节点
        if group is not None and not append:
            self._handle.remove_node(group, recursive=True)
            group = None

        # 如果节点不存在，则创建节点和组
        if group is None:
            group = self._create_nodes_and_group(key)

        # 返回最终确定的组对象
        return group
    def _create_nodes_and_group(self, key: str) -> Node:
        """Create nodes from key and return group name."""
        # 确保对象属性_handle不为空，用于类型检查
        assert self._handle is not None

        # 将路径按照"/"分割成列表
        paths = key.split("/")

        # 递归创建组
        path = "/"
        for p in paths:
            # 如果路径段为空，则跳过
            if not len(p):
                continue
            
            # 构建新路径
            new_path = path
            if not path.endswith("/"):
                new_path += "/"
            new_path += p
            
            # 获取或创建节点
            group = self.get_node(new_path)
            if group is None:
                group = self._handle.create_group(path, p)
            
            # 更新当前路径
            path = new_path
        
        # 返回最终创建或获取的组
        return group
class TableIterator:
    """
    Define the iteration interface on a table

    Parameters
    ----------
    store : HDFStore
        The HDFStore containing the data table.
    s : GenericFixed | Table
        The reference to the data table.
    func : callable
        The function to execute the query on the table.
    where : query expression
        The condition to apply on the table.
    nrows : int
        Number of rows to iterate over.
    start : int or None, default None
        Starting index for iteration.
    stop : int or None, default None
        Stopping index for iteration.
    iterator : bool, default False
        Whether to use the default iterator.
    chunksize : int or None, default 100000
        Size of chunks to iterate over.
    auto_close : bool, default False
        Whether to automatically close the store at the end of iteration.
    """

    chunksize: int | None
    store: HDFStore
    s: GenericFixed | Table

    def __init__(
        self,
        store: HDFStore,
        s: GenericFixed | Table,
        func,
        where,
        nrows,
        start=None,
        stop=None,
        iterator: bool = False,
        chunksize: int | None = None,
        auto_close: bool = False,
    ) -> None:
        self.store = store
        self.s = s
        self.func = func
        self.where = where

        # Set start/stop if not specified when dealing with a table
        if self.s.is_table:
            if nrows is None:
                nrows = 0
            if start is None:
                start = 0
            if stop is None:
                stop = nrows
            stop = min(nrows, stop)

        self.nrows = nrows
        self.start = start
        self.stop = stop

        self.coordinates = None
        # Determine chunksize based on iterator flag or specified value
        if iterator or chunksize is not None:
            if chunksize is None:
                chunksize = 100000
            self.chunksize = int(chunksize)
        else:
            self.chunksize = None

        self.auto_close = auto_close

    def __iter__(self) -> Iterator:
        # Iterator method to iterate over the table data
        current = self.start
        if self.coordinates is None:
            raise ValueError("Cannot iterate until get_result is called.")
        while current < self.stop:
            stop = min(current + self.chunksize, self.stop)
            value = self.func(None, None, self.coordinates[current:stop])
            current = stop
            if value is None or not len(value):
                continue

            yield value

        self.close()

    def close(self) -> None:
        # Method to close the HDFStore if auto_close is True
        if self.auto_close:
            self.store.close()
    def get_result(self, coordinates: bool = False):
        # 如果设置了块大小，则验证 self.s 是否为 Table 对象，否则抛出类型错误
        if self.chunksize is not None:
            if not isinstance(self.s, Table):
                raise TypeError("can only use an iterator or chunksize on a table")

            # 如果是 Table 对象，则根据条件读取坐标
            self.coordinates = self.s.read_coordinates(where=self.where)

            # 返回当前对象自身，作为迭代器
            return self

        # 如果指定了 coordinates 参数为 True，则验证 self.s 是否为 Table 对象，否则抛出类型错误
        if coordinates:
            if not isinstance(self.s, Table):
                raise TypeError("can only read_coordinates on a table")

            # 根据指定的条件和范围读取坐标
            where = self.s.read_coordinates(
                where=self.where, start=self.start, stop=self.stop
            )
        else:
            # 否则直接使用当前的 where 条件
            where = self.where

        # 调用预定义的 func 函数，传入起始点、终止点和 where 条件，获取结果
        results = self.func(self.start, self.stop, where)

        # 关闭当前对象
        self.close()

        # 返回 func 函数的结果
        return results
class IndexCol:
    """
    an index column description class

    Parameters
    ----------
    axis   : axis which I reference
    values : the ndarray like converted values
    kind   : a string description of this type
    typ    : the pytables type
    pos    : the position in the pytables

    """

    # 定义类变量，表示是否可索引和是否数据可索引
    is_an_indexable: bool = True
    is_data_indexable: bool = True
    # 定义类变量，表示额外信息的字段名列表
    _info_fields = ["freq", "tz", "index_name"]

    def __init__(
        self,
        name: str,
        values=None,
        kind=None,
        typ=None,
        cname: str | None = None,
        axis=None,
        pos=None,
        freq=None,
        tz=None,
        index_name=None,
        ordered=None,
        table=None,
        meta=None,
        metadata=None,
    ) -> None:
        # 检查name是否为字符串类型，若不是则引发值错误异常
        if not isinstance(name, str):
            raise ValueError("`name` must be a str.")

        # 实例变量赋值
        self.values = values
        self.kind = kind
        self.typ = typ
        self.name = name
        self.cname = cname or name
        self.axis = axis
        self.pos = pos
        self.freq = freq
        self.tz = tz
        self.index_name = index_name
        self.ordered = ordered
        self.table = table
        self.meta = meta
        self.metadata = metadata

        # 如果pos不为None，则调用set_pos方法设置位置
        if pos is not None:
            self.set_pos(pos)

        # 断言确保传入的参数符合构造函数的注解要求
        assert isinstance(self.name, str)
        assert isinstance(self.cname, str)

    @property
    def itemsize(self) -> int:
        # 返回typ的项目大小，假设typ已经被初始化
        return self.typ.itemsize

    @property
    def kind_attr(self) -> str:
        # 返回格式化的属性名称，形式为"{name}_kind"
        return f"{self.name}_kind"

    def set_pos(self, pos: int) -> None:
        """set the position of this column in the Table"""
        # 设置列在表中的位置，并且如果位置和typ都不为None，则设置typ的_v_pos属性为pos
        self.pos = pos
        if pos is not None and self.typ is not None:
            self.typ._v_pos = pos

    def __repr__(self) -> str:
        # 返回列对象的字符串表示形式，包括name、cname、axis、pos和kind的信息
        temp = tuple(
            map(pprint_thing, (self.name, self.cname, self.axis, self.pos, self.kind))
        )
        return ",".join(
            [
                f"{key}->{value}"
                for key, value in zip(["name", "cname", "axis", "pos", "kind"], temp)
            ]
        )

    def __eq__(self, other: object) -> bool:
        """compare 2 col items"""
        # 比较两个列对象是否相等，比较name、cname、axis和pos属性是否相等
        return all(
            getattr(self, a, None) == getattr(other, a, None)
            for a in ["name", "cname", "axis", "pos"]
        )

    def __ne__(self, other) -> bool:
        # 判断两个列对象是否不相等，使用__eq__方法的相反结果
        return not self.__eq__(other)

    @property
    def is_indexed(self) -> bool:
        """return whether I am an indexed column"""
        # 返回该列是否已经被索引
        if not hasattr(self.table, "cols"):
            # 例如，如果还未调用infer方法，self.table将为None
            return False
        return getattr(self.table.cols, self.cname).is_indexed

    def convert(
        self, values: np.ndarray, nan_rep, encoding: str, errors: str
        # 将给定的值数组转换为指定编码和错误处理方式下的数据
    ) -> tuple[np.ndarray, np.ndarray] | tuple[Index, Index]:
        """
        Convert the data from this selection to the appropriate pandas type.
        """
        assert isinstance(values, np.ndarray), type(values)

        # values is a recarray
        if values.dtype.fields is not None:
            # Copy, otherwise values will be a view
            # preventing the original recarray from being freed
            values = values[self.cname].copy()

        val_kind = self.kind
        # Convert values to the appropriate type considering the kind and encoding
        values = _maybe_convert(values, val_kind, encoding, errors)
        kwargs = {}
        kwargs["name"] = self.index_name

        if self.freq is not None:
            kwargs["freq"] = self.freq

        # Determine the factory type for creating the Index
        factory: type[Index | DatetimeIndex] = Index
        if lib.is_np_dtype(values.dtype, "M") or isinstance(
            values.dtype, DatetimeTZDtype
        ):
            factory = DatetimeIndex
        elif values.dtype == "i8" and "freq" in kwargs:
            # PeriodIndex data is stored as i8
            # error: Incompatible types in assignment (expression has type
            # "Callable[[Any, KwArg(Any)], PeriodIndex]", variable has type
            # "Union[Type[Index], Type[DatetimeIndex]]")
            # Use lambda function to create PeriodIndex from ordinals with optional frequency
            factory = lambda x, **kwds: PeriodIndex.from_ordinals(  # type: ignore[assignment]
                x, freq=kwds.get("freq", None)
            )._rename(kwds["name"])

        # Attempt to create a new Index instance
        try:
            new_pd_index = factory(values, **kwargs)
        except ValueError:
            # Handle ValueError by setting freq to None if it differs from recorded value
            if "freq" in kwargs:
                kwargs["freq"] = None
            new_pd_index = factory(values, **kwargs)

        # Handle timezone localization if tz is specified
        final_pd_index: Index
        if self.tz is not None and isinstance(new_pd_index, DatetimeIndex):
            final_pd_index = new_pd_index.tz_localize("UTC").tz_convert(self.tz)
        else:
            final_pd_index = new_pd_index

        # Return the final pandas Index instance
        return final_pd_index, final_pd_index

    def take_data(self):
        """return the values"""
        # Return the stored values
        return self.values

    @property
    def attrs(self):
        # Return the attributes of the table
        return self.table._v_attrs

    @property
    def description(self):
        # Return the description of the table
        return self.table.description

    @property
    def col(self):
        """return my current col description"""
        # Return the description of the column
        return getattr(self.description, self.cname, None)

    @property
    def cvalues(self):
        """return my cython values"""
        # Return the Cython values stored
        return self.values

    def __iter__(self) -> Iterator:
        # Return an iterator over the values
        return iter(self.values)
    # 如果数据类型是字符串，根据条件设定字符串列的长度
    def maybe_set_size(self, min_itemsize=None) -> None:
        """
        maybe set a string col itemsize:
            min_itemsize can be an integer or a dict with this columns name
            with an integer size
        """
        # 检查数据类型是否为字符串
        if self.kind == "string":
            # 如果 min_itemsize 是字典，则获取当前列名对应的最小长度
            if isinstance(min_itemsize, dict):
                min_itemsize = min_itemsize.get(self.name)

            # 如果 min_itemsize 不为 None 并且当前列的长度小于 min_itemsize，则设定列的长度为 min_itemsize
            if min_itemsize is not None and self.typ.itemsize < min_itemsize:
                self.typ = _tables().StringCol(itemsize=min_itemsize, pos=self.pos)

    # 空函数，用于验证列名
    def validate_names(self) -> None:
        pass

    # 验证并设置列属性
    def validate_and_set(self, handler: AppendableTable, append: bool) -> None:
        # 将处理器的表格赋给当前对象的表格属性
        self.table = handler.table
        # 验证列
        self.validate_col()
        # 验证属性，根据需要追加
        self.validate_attr(append)
        # 验证元数据信息
        self.validate_metadata(handler)
        # 将元数据写入处理器
        self.write_metadata(handler)
        # 设置属性
        self.set_attr()

    # 验证列，处理字符串截断或重置最大大小
    def validate_col(self, itemsize=None):
        """validate this column: return the compared against itemsize"""
        # 如果数据类型是字符串
        if self.kind == "string":
            c = self.col
            # 如果列对象存在
            if c is not None:
                # 如果 itemsize 为 None，则使用对象本身的 itemsize
                if itemsize is None:
                    itemsize = self.itemsize
                # 如果列的 itemsize 小于指定的 itemsize，则抛出异常
                if c.itemsize < itemsize:
                    raise ValueError(
                        f"Trying to store a string with len [{itemsize}] in "
                        f"[{self.cname}] column but\nthis column has a limit of "
                        f"[{c.itemsize}]!\nConsider using min_itemsize to "
                        "preset the sizes on these columns"
                    )
                return c.itemsize

        return None

    # 验证属性，检查是否存在不兼容的列类型
    def validate_attr(self, append: bool) -> None:
        # 如果需要追加数据
        if append:
            # 获取当前属性的类型
            existing_kind = getattr(self.attrs, self.kind_attr, None)
            # 如果当前属性存在且与当前列的类型不一致，则抛出类型错误异常
            if existing_kind is not None and existing_kind != self.kind:
                raise TypeError(
                    f"incompatible kind in col [{existing_kind} - {self.kind}]"
                )
    def update_info(self, info) -> None:
        """
        set/update the info for this indexable with the key/value
        if there is a conflict raise/warn as needed
        """
        # 遍历所有信息字段
        for key in self._info_fields:
            # 获取当前对象上对应字段的值
            value = getattr(self, key, None)
            # 获取或设置当前对象在信息字典中的索引
            idx = info.setdefault(self.name, {})

            # 获取已存在的字段值
            existing_value = idx.get(key)
            # 如果字段已存在且当前值与已存在的值不相同
            if key in idx and value is not None and existing_value != value:
                # 对于 "freq" 和 "index_name"，发出警告
                if key in ["freq", "index_name"]:
                    # 构造属性冲突的警告信息
                    ws = attribute_conflict_doc % (key, existing_value, value)
                    # 发出警告
                    warnings.warn(
                        ws, AttributeConflictWarning, stacklevel=find_stack_level()
                    )

                    # 重置对象上对应字段的值为 None
                    idx[key] = None
                    setattr(self, key, None)

                else:
                    # 否则，抛出数值错误，指示冲突的信息
                    raise ValueError(
                        f"invalid info for [{self.name}] for [{key}], "
                        f"existing_value [{existing_value}] conflicts with "
                        f"new value [{value}]"
                    )
            # 如果当前值或已存在的值不为空，则更新索引中的值
            elif value is not None or existing_value is not None:
                idx[key] = value

    def set_info(self, info) -> None:
        """set my state from the passed info"""
        # 从信息字典中获取当前对象的信息
        idx = info.get(self.name)
        # 如果信息存在，则更新对象的状态
        if idx is not None:
            self.__dict__.update(idx)

    def set_attr(self) -> None:
        """set the kind for this column"""
        # 设置当前列的属性
        setattr(self.attrs, self.kind_attr, self.kind)

    def validate_metadata(self, handler: AppendableTable) -> None:
        """validate that kind=category does not change the categories"""
        # 验证元数据，确保 kind=category 不会改变类别
        if self.meta == "category":
            # 获取新旧元数据
            new_metadata = self.metadata
            cur_metadata = handler.read_metadata(self.cname)
            # 如果新旧元数据存在且不相等，则抛出数值错误
            if (
                new_metadata is not None
                and cur_metadata is not None
                and not array_equivalent(
                    new_metadata, cur_metadata, strict_nan=True, dtype_equal=True
                )
            ):
                raise ValueError(
                    "cannot append a categorical with "
                    "different categories to the existing"
                )

    def write_metadata(self, handler: AppendableTable) -> None:
        """set the meta data"""
        # 设置元数据
        if self.metadata is not None:
            handler.write_metadata(self.cname, self.metadata)
class GenericIndexCol(IndexCol):
    """an index which is not represented in the data of the table"""

    @property
    def is_indexed(self) -> bool:
        # 返回 False，表明这个索引在表的数据中并不存在
        return False

    def convert(
        self, values: np.ndarray, nan_rep, encoding: str, errors: str
    ) -> tuple[Index, Index]:
        """
        Convert the data from this selection to the appropriate pandas type.

        Parameters
        ----------
        values : np.ndarray
            包含要转换数据的 NumPy 数组
        nan_rep : str
            用于表示 NaN 值的字符串表示
        encoding : str
            字符编码方式
        errors : str
            解码错误时的处理方式
        """
        assert isinstance(values, np.ndarray), type(values)

        # 创建一个 RangeIndex 对象，表示从 0 到 len(values)-1 的索引范围
        index = RangeIndex(len(values))
        return index, index

    def set_attr(self) -> None:
        # 空方法，没有具体的操作，用于设置属性
        pass


class DataCol(IndexCol):
    """
    a data holding column, by definition this is not indexable

    Parameters
    ----------
    data   : the actual data
        实际的数据
    cname  : the column name in the table to hold the data (typically
                values)
        表中存储数据的列名
    meta   : a string description of the metadata
        元数据的字符串描述
    metadata : the actual metadata
        实际的元数据
    """

    is_an_indexable = False
    is_data_indexable = False
    _info_fields = ["tz", "ordered"]

    def __init__(
        self,
        name: str,
        values=None,
        kind=None,
        typ=None,
        cname: str | None = None,
        pos=None,
        tz=None,
        ordered=None,
        table=None,
        meta=None,
        metadata=None,
        dtype: DtypeArg | None = None,
        data=None,
    ) -> None:
        super().__init__(
            name=name,
            values=values,
            kind=kind,
            typ=typ,
            pos=pos,
            cname=cname,
            tz=tz,
            ordered=ordered,
            table=table,
            meta=meta,
            metadata=metadata,
        )
        self.dtype = dtype  # 设置列的数据类型
        self.data = data  # 设置列的数据

    @property
    def dtype_attr(self) -> str:
        # 返回列的数据类型属性名称
        return f"{self.name}_dtype"

    @property
    def meta_attr(self) -> str:
        # 返回列的元数据属性名称
        return f"{self.name}_meta"

    def __repr__(self) -> str:
        # 返回列对象的字符串表示形式，包括列名、列在表中的名称、数据类型、种类和形状
        temp = tuple(
            map(
                pprint_thing, (self.name, self.cname, self.dtype, self.kind, self.shape)
            )
        )
        return ",".join(
            [
                f"{key}->{value}"
                for key, value in zip(["name", "cname", "dtype", "kind", "shape"], temp)
            ]
        )

    def __eq__(self, other: object) -> bool:
        """compare 2 col items"""
        # 比较两个列对象的各个属性是否相等
        return all(
            getattr(self, a, None) == getattr(other, a, None)
            for a in ["name", "cname", "dtype", "pos"]
        )

    def set_data(self, data: ArrayLike) -> None:
        assert data is not None  # 断言数据不为空
        assert self.dtype is None  # 断言数据类型为空

        # 获取数据及其数据类型名称
        data, dtype_name = _get_data_and_dtype_name(data)

        self.data = data  # 设置列的数据
        self.dtype = dtype_name  # 设置列的数据类型
        self.kind = _dtype_to_kind(dtype_name)  # 根据数据类型确定种类

    def take_data(self):
        """return the data"""
        # 返回列的数据
        return self.data

    @classmethod
    def _get_atom(cls, values: ArrayLike) -> Col:
        """
        Get an appropriately typed and shaped pytables.Col object for values.
        """
        # 获取值的数据类型
        dtype = values.dtype
        # 获取数据类型的字节大小
        itemsize = dtype.itemsize  # type: ignore[union-attr]

        # 获取值的形状
        shape = values.shape
        # 如果值是一维的，则假装其形状为二维
        if values.ndim == 1:
            # EA，使用块形状，假装是二维的
            # TODO(EA2D): 对于二维 EA 不需要
            shape = (1, values.size)

        # 如果值是分类类型
        if isinstance(values, Categorical):
            # 获取分类的编码
            codes = values.codes
            # 根据编码数据类型的名称获取适当的原子对象
            atom = cls.get_atom_data(shape, kind=codes.dtype.name)
        # 如果数据类型是日期时间或带时区的日期时间
        elif lib.is_np_dtype(dtype, "M") or isinstance(dtype, DatetimeTZDtype):
            # 获取日期时间类型的原子对象
            atom = cls.get_atom_datetime64(shape)
        # 如果数据类型是时间间隔
        elif lib.is_np_dtype(dtype, "m"):
            # 获取时间间隔类型的原子对象
            atom = cls.get_atom_timedelta64(shape)
        # 如果数据类型是复数
        elif is_complex_dtype(dtype):
            # 根据数据类型的字节大小和形状创建复数类型的原子对象
            atom = _tables().ComplexCol(itemsize=itemsize, shape=shape[0])
        # 如果数据类型是字符串
        elif is_string_dtype(dtype):
            # 获取字符串类型的原子对象
            atom = cls.get_atom_string(shape, itemsize)
        else:
            # 根据数据类型的名称获取适当的原子对象
            atom = cls.get_atom_data(shape, kind=dtype.name)

        # 返回获取到的原子对象
        return atom

    @classmethod
    def get_atom_string(cls, shape, itemsize):
        """
        Return a StringCol object with specified shape and item size.
        """
        return _tables().StringCol(itemsize=itemsize, shape=shape[0])

    @classmethod
    def get_atom_coltype(cls, kind: str) -> type[Col]:
        """
        Return the PyTables column class corresponding to the given kind.
        """
        # 根据列的类型名称返回对应的 PyTables 列类
        if kind.startswith("uint"):
            k4 = kind[4:]
            col_name = f"UInt{k4}Col"
        elif kind.startswith("period"):
            # 将 period 类型存储为整数
            col_name = "Int64Col"
        else:
            kcap = kind.capitalize()
            col_name = f"{kcap}Col"

        return getattr(_tables(), col_name)

    @classmethod
    def get_atom_data(cls, shape, kind: str) -> Col:
        """
        Return a Col object with specified shape and data kind.
        """
        # 根据数据类型的种类返回对应的原子对象
        return cls.get_atom_coltype(kind=kind)(shape=shape[0])

    @classmethod
    def get_atom_datetime64(cls, shape):
        """
        Return an Int64Col object for datetime64 data with specified shape.
        """
        return _tables().Int64Col(shape=shape[0])

    @classmethod
    def get_atom_timedelta64(cls, shape):
        """
        Return an Int64Col object for timedelta64 data with specified shape.
        """
        return _tables().Int64Col(shape=shape[0])

    @property
    def shape(self):
        """
        Return the shape attribute of the data property.
        """
        return getattr(self.data, "shape", None)

    @property
    def cvalues(self):
        """
        Return the Cython values stored in the data property.
        """
        return self.data
    # 验证属性的有效性，确保与现有属性相同的顺序和数据类型
    def validate_attr(self, append) -> None:
        """validate that we have the same order as the existing & same dtype"""
        # 如果 append 为 True，则进行验证
        if append:
            # 获取现有的字段列表
            existing_fields = getattr(self.attrs, self.kind_attr, None)
            # 如果存在现有字段，并且与当前值列表不同，则抛出数值错误异常
            if existing_fields is not None and existing_fields != list(self.values):
                raise ValueError("appended items do not match existing items in table!")

            # 获取现有的数据类型
            existing_dtype = getattr(self.attrs, self.dtype_attr, None)
            # 如果存在现有数据类型，并且与当前数据类型不同，则抛出数值错误异常
            if existing_dtype is not None and existing_dtype != self.dtype:
                raise ValueError(
                    "appended items dtype do not match existing items dtype in table!"
                )

    # 设置这一列的数据属性
    def set_attr(self) -> None:
        """set the data for this column"""
        # 设置字段的值
        setattr(self.attrs, self.kind_attr, self.values)
        # 设置元数据属性
        setattr(self.attrs, self.meta_attr, self.meta)
        # 确保数据类型不为 None
        assert self.dtype is not None
        # 设置数据类型属性
        setattr(self.attrs, self.dtype_attr, self.dtype)
class DataIndexableCol(DataCol):
    """represent a data column that can be indexed"""

    is_data_indexable = True  # 声明这是一个可索引的数据列

    def validate_names(self) -> None:
        if not is_string_dtype(Index(self.values).dtype):
            # 如果数据列中的索引类型不是字符串，则抛出异常
            raise ValueError("cannot have non-object label DataIndexableCol")

    @classmethod
    def get_atom_string(cls, shape, itemsize):
        # 返回一个字符串类型的列
        return _tables().StringCol(itemsize=itemsize)

    @classmethod
    def get_atom_data(cls, shape, kind: str) -> Col:
        # 根据指定的数据类型获取相应的列类型
        return cls.get_atom_coltype(kind=kind)()

    @classmethod
    def get_atom_datetime64(cls, shape):
        # 返回一个日期时间类型的列
        return _tables().Int64Col()

    @classmethod
    def get_atom_timedelta64(cls, shape):
        # 返回一个时间差类型的列
        return _tables().Int64Col()


class GenericDataIndexableCol(DataIndexableCol):
    """represent a generic pytables data column"""


class Fixed:
    """
    represent an object in my store
    facilitate read/write of various types of objects
    this is an abstract base class

    Parameters
    ----------
    parent : HDFStore
        HDF 存储对象，作为父对象
    group : Node
        表所在的节点
    """

    pandas_kind: str
    format_type: str = "fixed"  # GH#30962 needed by dask
    obj_type: type[DataFrame | Series]
    ndim: int
    parent: HDFStore
    is_table: bool = False

    def __init__(
        self,
        parent: HDFStore,
        group: Node,
        encoding: str | None = "UTF-8",
        errors: str = "strict",
    ) -> None:
        assert isinstance(parent, HDFStore), type(parent)
        assert _table_mod is not None  # needed for mypy
        assert isinstance(group, _table_mod.Node), type(group)
        self.parent = parent  # 设置父对象
        self.group = group  # 设置表所在节点
        self.encoding = _ensure_encoding(encoding)  # 确保编码方式
        self.errors = errors  # 设置错误处理方式

    @property
    def is_old_version(self) -> bool:
        # 判断是否为旧版本
        return self.version[0] <= 0 and self.version[1] <= 10 and self.version[2] < 1

    @property
    def version(self) -> tuple[int, int, int]:
        """compute and set our version"""
        version = getattr(self.group._v_attrs, "pandas_version", None)
        if isinstance(version, str):
            version_tup = tuple(int(x) for x in version.split("."))
            if len(version_tup) == 2:
                version_tup = version_tup + (0,)
            assert len(version_tup) == 3  # needed for mypy
            return version_tup
        else:
            return (0, 0, 0)  # 默认返回版本为 (0, 0, 0)

    @property
    def pandas_type(self):
        # 返回存储对象的 Pandas 类型
        return getattr(self.group._v_attrs, "pandas_type", None)

    def __repr__(self) -> str:
        """return a pretty representation of myself"""
        self.infer_axes()  # 推断轴信息
        s = self.shape  # 获取形状信息
        if s is not None:
            if isinstance(s, (list, tuple)):
                jshape = ",".join([pprint_thing(x) for x in s])  # 格式化形状信息
                s = f"[{jshape}]"  # 格式化输出形状
            return f"{self.pandas_type:12.12} (shape->{s})"  # 返回对象的字符串表示形式
        return self.pandas_type
    # 设置对象的 pandas 类型和版本信息
    def set_object_info(self) -> None:
        """set my pandas type & version"""
        self.attrs.pandas_type = str(self.pandas_kind)
        self.attrs.pandas_version = str(_version)

    # 创建当前对象的浅拷贝并返回
    def copy(self) -> Fixed:
        new_self = copy.copy(self)
        return new_self

    # 返回对象的形状，即行数
    @property
    def shape(self):
        return self.nrows

    # 返回对象的路径名
    @property
    def pathname(self):
        return self.group._v_pathname

    # 返回对象的处理器句柄
    @property
    def _handle(self):
        return self.parent._handle

    # 返回对象的过滤器
    @property
    def _filters(self):
        return self.parent._filters

    # 返回对象数据的压缩级别
    @property
    def _complevel(self) -> int:
        return self.parent._complevel

    # 返回对象数据是否使用 Fletcher32 校验
    @property
    def _fletcher32(self) -> bool:
        return self.parent._fletcher32

    # 返回对象的属性集合
    @property
    def attrs(self):
        return self.group._v_attrs

    # 设置对象的属性
    def set_attrs(self) -> None:
        """set our object attributes"""

    # 获取对象的属性
    def get_attrs(self) -> None:
        """get our object attributes"""

    # 返回对象的存储器
    @property
    def storable(self):
        """return my storable"""
        return self.group

    # 返回对象是否存在的布尔值
    @property
    def is_exists(self) -> bool:
        return False

    # 返回对象的行数
    @property
    def nrows(self):
        return getattr(self.storable, "nrows", None)

    # 验证对象与另一个存储器是否匹配
    def validate(self, other) -> Literal[True] | None:
        """validate against an existing storable"""
        if other is None:
            return None
        return True

    # 验证操作的版本是否过旧
    def validate_version(self, where=None) -> None:
        """are we trying to operate on an old version?"""

    # 推断对象的轴信息是否有效
    def infer_axes(self) -> bool:
        """
        infer the axes of my storer
        return a boolean indicating if we have a valid storer or not
        """
        s = self.storable
        if s is None:
            return False
        self.get_attrs()
        return True

    # 抽象方法，需要子类实现，在抽象存储器上读取数据
    def read(
        self,
        where=None,
        columns=None,
        start: int | None = None,
        stop: int | None = None,
    ) -> Series | DataFrame:
        raise NotImplementedError(
            "cannot read on an abstract storer: subclasses should implement"
        )

    # 抽象方法，需要子类实现，在抽象存储器上写入数据
    def write(self, obj, **kwargs) -> None:
        raise NotImplementedError(
            "cannot write on an abstract storer: subclasses should implement"
        )

    # 删除对象的节点，支持完全删除节点（仅限全空指定）
    def delete(
        self, where=None, start: int | None = None, stop: int | None = None
    ) -> int | None:
        """
        support fully deleting the node in its entirety (only) - where
        specification must be None
        """
        if com.all_none(where, start, stop):
            self._handle.remove_node(self.group, recursive=True)
            return None

        raise TypeError("cannot delete on an abstract storer")
class GenericFixed(Fixed):
    """a generified fixed version"""

    # 索引类型映射字典，将索引类映射到字符串别名
    _index_type_map = {DatetimeIndex: "datetime", PeriodIndex: "period"}
    # 反向索引映射字典，将字符串别名映射回索引类
    _reverse_index_map = {v: k for k, v in _index_type_map.items()}
    # 属性列表
    attributes: list[str] = []

    # 索引器辅助函数：将类转换为别名字符串
    def _class_to_alias(self, cls) -> str:
        return self._index_type_map.get(cls, "")

    # 索引器辅助函数：将别名字符串转换为类
    def _alias_to_class(self, alias):
        if isinstance(alias, type):  # pragma: no cover
            # 兼容性：短期内主分支存储类型
            return alias
        return self._reverse_index_map.get(alias, Index)

    # 获取索引工厂函数，根据属性返回适当的索引创建函数及其参数
    def _get_index_factory(self, attrs):
        index_class = self._alias_to_class(getattr(attrs, "index_class", ""))

        factory: Callable

        if index_class == DatetimeIndex:
            # 如果是DatetimeIndex，则创建对应的工厂函数
            def f(values, freq=None, tz=None):
                # 数据已经是UTC，根据情况进行本地化和转换时区
                dta = DatetimeArray._simple_new(
                    values.values, dtype=values.dtype, freq=freq
                )
                result = DatetimeIndex._simple_new(dta, name=None)
                if tz is not None:
                    result = result.tz_localize("UTC").tz_convert(tz)
                return result

            factory = f
        elif index_class == PeriodIndex:
            # 如果是PeriodIndex，则创建对应的工厂函数
            def f(values, freq=None, tz=None):
                dtype = PeriodDtype(freq)
                parr = PeriodArray._simple_new(values, dtype=dtype)
                return PeriodIndex._simple_new(parr, name=None)

            factory = f
        else:
            # 否则，直接使用索引类作为工厂函数
            factory = index_class

        kwargs = {}
        if "freq" in attrs:
            kwargs["freq"] = attrs["freq"]
            if index_class is Index:
                # 如果是Index类，则将工厂函数设为TimedeltaIndex
                factory = TimedeltaIndex

        if "tz" in attrs:
            kwargs["tz"] = attrs["tz"]
            assert index_class is DatetimeIndex  # 只是检查

        return factory, kwargs

    # 验证读取函数，检查是否传递了不为None的列或条件参数
    def validate_read(self, columns, where) -> None:
        """
        raise if any keywords are passed which are not-None
        """
        if columns is not None:
            raise TypeError(
                "cannot pass a column specification when reading "
                "a Fixed format store. this store must be selected in its entirety"
            )
        if where is not None:
            raise TypeError(
                "cannot pass a where specification when reading "
                "from a Fixed format store. this store must be selected in its entirety"
            )

    # 是否存在属性，总是返回True
    @property
    def is_exists(self) -> bool:
        return True

    # 设置属性函数，设置对象的编码和错误处理属性
    def set_attrs(self) -> None:
        """set our object attributes"""
        self.attrs.encoding = self.encoding
        self.attrs.errors = self.errors
    def get_attrs(self) -> None:
        """获取对象的属性"""
        # 设置编码属性，确保编码存在或使用默认值
        self.encoding = _ensure_encoding(getattr(self.attrs, "encoding", None))
        # 设置错误处理属性，如果不存在则默认使用严格模式
        self.errors = getattr(self.attrs, "errors", "strict")
        # 遍历属性列表，将属性值设置到当前对象中
        for n in self.attributes:
            setattr(self, n, getattr(self.attrs, n, None))

    def write(self, obj, **kwargs) -> None:
        """写操作，暂未实现具体功能"""
        self.set_attrs()

    def read_array(self, key: str, start: int | None = None, stop: int | None = None):
        """读取指定节点（从组中）的数组"""
        import tables

        # 获取指定键对应的节点
        node = getattr(self.group, key)
        # 获取节点的属性
        attrs = node._v_attrs

        # 检查是否需要转置数组
        transposed = getattr(attrs, "transposed", False)

        if isinstance(node, tables.VLArray):
            # 如果节点是可变长度数组，则读取第一个元素的指定切片
            ret = node[0][start:stop]
        else:
            # 否则，获取节点的数据类型和形状
            dtype = getattr(attrs, "value_type", None)
            shape = getattr(attrs, "shape", None)

            if shape is not None:
                # 如果存在形状信息，则创建指定形状和数据类型的空数组
                ret = np.empty(shape, dtype=dtype)
            else:
                # 否则，直接从节点中读取指定切片的数据
                ret = node[start:stop]

            if dtype and dtype.startswith("datetime64"):
                # 如果数据类型是日期时间类型，则根据属性中的时区信息重新设置时区
                tz = getattr(attrs, "tz", None)
                ret = _set_tz(ret, tz, dtype)

            elif dtype == "timedelta64":
                # 如果数据类型是时间间隔类型，则将其转换为纳秒精度的 numpy 数组
                ret = np.asarray(ret, dtype="m8[ns]")

        if transposed:
            return ret.T  # 如果需要转置，则返回转置后的数组
        else:
            return ret  # 否则返回原始数组

    def read_index(
        self, key: str, start: int | None = None, stop: int | None = None
    ) -> Index:
        """读取指定节点的索引"""
        # 获取指定键对应的索引类型
        variety = getattr(self.attrs, f"{key}_variety")

        if variety == "multi":
            # 如果索引类型为多级索引，则调用读取多级索引的方法
            return self.read_multi_index(key, start=start, stop=stop)
        elif variety == "regular":
            # 如果索引类型为普通索引，则获取节点并读取索引节点的内容
            node = getattr(self.group, key)
            index = self.read_index_node(node, start=start, stop=stop)
            return index
        else:  # pragma: no cover
            # 如果类型不在预期范围内，则引发类型错误
            raise TypeError(f"unrecognized index variety: {variety}")

    def write_index(self, key: str, index: Index) -> None:
        """写入索引"""
        if isinstance(index, MultiIndex):
            # 如果索引是多级索引，则设置相应的索引类型属性，并写入多级索引数据
            setattr(self.attrs, f"{key}_variety", "multi")
            self.write_multi_index(key, index)
        else:
            # 否则，设置索引类型属性为普通索引，并转换索引并写入数据
            setattr(self.attrs, f"{key}_variety", "regular")
            converted = _convert_index("index", index, self.encoding, self.errors)

            self.write_array(key, converted.values)

            node = getattr(self.group, key)
            node._v_attrs.kind = converted.kind
            node._v_attrs.name = index.name

            if isinstance(index, (DatetimeIndex, PeriodIndex)):
                node._v_attrs.index_class = self._class_to_alias(type(index))

            if isinstance(index, (DatetimeIndex, PeriodIndex, TimedeltaIndex)):
                node._v_attrs.freq = index.freq

            if isinstance(index, DatetimeIndex) and index.tz is not None:
                node._v_attrs.tz = _get_tz(index.tz)
    # 将 MultiIndex 的层级数量设置为属性值
    setattr(self.attrs, f"{key}_nlevels", index.nlevels)

    # 遍历 MultiIndex 的每一个层级，获取层级、层级代码和名称
    for i, (lev, level_codes, name) in enumerate(
        zip(index.levels, index.codes, index.names)
    ):
        # 如果层级的数据类型是 ExtensionDtype，抛出未实现的错误
        if isinstance(lev.dtype, ExtensionDtype):
            raise NotImplementedError(
                "Saving a MultiIndex with an extension dtype is not supported."
            )
        
        # 构建层级的键名
        level_key = f"{key}_level{i}"
        # 转换层级数据为适合保存的格式，并写入数据存储
        conv_level = _convert_index(level_key, lev, self.encoding, self.errors)
        self.write_array(level_key, conv_level.values)
        # 获取或创建层级节点，并设置节点属性
        node = getattr(self.group, level_key)
        node._v_attrs.kind = conv_level.kind
        node._v_attrs.name = name

        # 设置层级名称属性
        setattr(node._v_attrs, f"{key}_name{name}", name)

        # 构建标签的键名
        label_key = f"{key}_label{i}"
        # 写入层级代码数据
        self.write_array(label_key, level_codes)

    # 读取存储中的 MultiIndex 数据并返回
    def read_multi_index(
        self, key: str, start: int | None = None, stop: int | None = None
    ) -> MultiIndex:
        # 获取存储中 MultiIndex 的层级数量
        nlevels = getattr(self.attrs, f"{key}_nlevels")

        # 初始化列表，用于存储读取的层级数据和名称
        levels = []
        codes = []
        names: list[Hashable] = []
        
        # 遍历每一个层级，读取层级数据和标签数据
        for i in range(nlevels):
            # 构建层级数据的键名
            level_key = f"{key}_level{i}"
            # 获取层级节点并读取数据
            node = getattr(self.group, level_key)
            lev = self.read_index_node(node, start=start, stop=stop)
            levels.append(lev)
            names.append(lev.name)

            # 构建层级标签数据的键名
            label_key = f"{key}_label{i}"
            # 读取层级标签数据
            level_codes = self.read_array(label_key, start=start, stop=stop)
            codes.append(level_codes)

        # 构建并返回 MultiIndex 对象
        return MultiIndex(
            levels=levels, codes=codes, names=names, verify_integrity=True
        )

    # 从节点中读取索引数据
    def read_index_node(
        self, node: Node, start: int | None = None, stop: int | None = None
    ) -> Index:
    ) -> Index:
        # 从节点中提取指定切片范围的数据
        data = node[start:stop]
        # 如果索引是空数组，则用原始数据替换写入的哨兵值
        if "shape" in node._v_attrs and np.prod(node._v_attrs.shape) == 0:
            data = np.empty(node._v_attrs.shape, dtype=node._v_attrs.value_type)
        # 获取节点的类型
        kind = node._v_attrs.kind
        name = None

        # 如果节点包含名称属性，则将其转换为字符串
        if "name" in node._v_attrs:
            name = _ensure_str(node._v_attrs.name)

        # 获取节点的属性
        attrs = node._v_attrs
        # 获取索引工厂和相关参数
        factory, kwargs = self._get_index_factory(attrs)

        # 根据节点的类型创建索引对象
        if kind in ("date", "object"):
            index = factory(
                _unconvert_index(
                    data, kind, encoding=self.encoding, errors=self.errors
                ),
                dtype=object,
                **kwargs,
            )
        else:
            index = factory(
                _unconvert_index(
                    data, kind, encoding=self.encoding, errors=self.errors
                ),
                **kwargs,
            )

        # 设置索引的名称
        index.name = name

        # 返回创建的索引对象
        return index

    def write_array_empty(self, key: str, value: ArrayLike) -> None:
        """write a 0-len array"""
        # 为长度为0的数组写入一个空的 numpy 数组
        arr = np.empty((1,) * value.ndim)
        # 在指定的组和键下创建一个数组节点
        self._handle.create_array(self.group, key, arr)
        # 获取刚创建的数组节点
        node = getattr(self.group, key)
        # 设置节点的值类型属性为 value 的数据类型字符串表示
        node._v_attrs.value_type = str(value.dtype)
        # 设置节点的形状属性为 value 的形状
        node._v_attrs.shape = value.shape

    def write_array(
        self, key: str, obj: AnyArrayLike, items: Index | None = None
class SeriesFixed(GenericFixed):
    # 定义类属性 pandas_kind 表示这是一个 Series 类型的对象
    pandas_kind = "series"
    # 定义类属性 attributes 包含对象的属性名
    attributes = ["name"]

    # 定义实例属性 name，必须是可哈希的对象
    name: Hashable

    @property
    def shape(self) -> tuple[int] | None:
        try:
            # 尝试返回 Series 对象的形状，这里是元组形式的长度信息
            return (len(self.group.values),)
        except (TypeError, AttributeError):
            # 如果出现异常（TypeError 或 AttributeError），返回 None
            return None

    # 定义 read 方法，用于读取数据生成 Series 对象
    def read(
        self,
        where=None,
        columns=None,
        start: int | None = None,
        stop: int | None = None,
    ) -> Series:
        # 验证读取操作的有效性，根据传入的列和位置信息
        self.validate_read(columns, where)
        # 读取索引数据
        index = self.read_index("index", start=start, stop=stop)
        # 读取数值数据
        values = self.read_array("values", start=start, stop=stop)
        # 创建 Series 对象，包括索引、数值和名称等信息
        result = Series(values, index=index, name=self.name, copy=False)
        # 如果正在使用 pyarrow 的字符串数据类型，并且数值是字符串数组，转换成 pyarrow_numpy 类型的字符串
        if using_pyarrow_string_dtype() and is_string_array(values, skipna=True):
            result = result.astype("string[pyarrow_numpy]")
        # 返回生成的 Series 对象
        return result

    # 定义 write 方法，用于将对象写入存储中
    def write(self, obj, **kwargs) -> None:
        # 调用父类的 write 方法来执行写入操作
        super().write(obj, **kwargs)
        # 写入索引数据
        self.write_index("index", obj.index)
        # 写入数值数据
        self.write_array("values", obj)
        # 设置对象的属性 name 为传入对象的名称
        self.attrs.name = obj.name


class BlockManagerFixed(GenericFixed):
    # 定义类属性 attributes 包含对象的属性名
    attributes = ["ndim", "nblocks"]

    # 定义实例属性 nblocks 表示数据块的数量
    nblocks: int

    @property
    def shape(self) -> Shape | None:
        try:
            # 尝试获取对象的形状信息
            ndim = self.ndim

            # 计算 items 的总数
            items = 0
            for i in range(self.nblocks):
                # 获取每个数据块的 items 数量
                node = getattr(self.group, f"block{i}_items")
                shape = getattr(node, "shape", None)
                if shape is not None:
                    items += shape[0]

            # 计算数据的形状信息
            node = self.group.block0_values
            shape = getattr(node, "shape", None)
            if shape is not None:
                # 取前 ndim-1 维度的形状信息
                shape = list(shape[0 : (ndim - 1)])
            else:
                shape = []

            # 将 items 数量作为最后一个维度添加到形状信息中
            shape.append(items)

            # 返回计算得到的形状信息
            return shape
        except AttributeError:
            # 如果出现 AttributeError 异常，返回 None
            return None

    # 定义 read 方法，用于读取数据
    def read(
        self,
        where=None,
        columns=None,
        start: int | None = None,
        stop: int | None = None,
        # 注意：read 方法的定义还未结束，尚有下文未注释部分
    ) -> DataFrame:
        # 定义函数的返回类型为 DataFrame
        # 根据传入的列和条件验证读取操作的有效性
        self.validate_read(columns, where)
        # 获取当前对象的类型并确定选择的轴
        select_axis = self.obj_type()._get_block_manager_axis(0)

        axes = []
        # 遍历对象的维度
        for i in range(self.ndim):
            # 如果当前维度是选择的轴，则使用传入的 start 和 stop
            # 否则设为 None
            _start, _stop = (start, stop) if i == select_axis else (None, None)
            # 读取索引并添加到 axes 列表中
            ax = self.read_index(f"axis{i}", start=_start, stop=_stop)
            axes.append(ax)

        # 获取第一个轴的内容作为 items
        items = axes[0]
        dfs = []

        # 遍历数据块的数量
        for i in range(self.nblocks):
            # 读取当前数据块的 items
            blk_items = self.read_index(f"block{i}_items")
            # 根据 start 和 stop 读取数据块的值
            values = self.read_array(f"block{i}_values", start=_start, stop=_stop)

            # 根据 items 创建对应的列名，并生成 DataFrame 对象
            columns = items[items.get_indexer(blk_items)]
            df = DataFrame(values.T, columns=columns, index=axes[1], copy=False)
            # 如果使用了 pyarrow 的 string 数据类型且值为字符串数组，则进行类型转换
            if using_pyarrow_string_dtype() and is_string_array(values, skipna=True):
                df = df.astype("string[pyarrow_numpy]")
            dfs.append(df)

        # 如果生成了 DataFrame 对象，则进行连接并复制结果
        if len(dfs) > 0:
            out = concat(dfs, axis=1).copy()
            return out.reindex(columns=items)

        # 如果没有生成 DataFrame，则创建一个只包含轴信息的空 DataFrame
        return DataFrame(columns=axes[0], index=axes[1])

    def write(self, obj, **kwargs) -> None:
        # 调用父类的写入方法，并传入参数 obj 和 kwargs
        super().write(obj, **kwargs)

        # 获取数据管理器
        data = obj._mgr
        # 如果数据没有被合并，则进行合并操作
        if not data.is_consolidated():
            data = data.consolidate()

        # 将数据的维度存储到属性中
        self.attrs.ndim = data.ndim
        # 遍历数据的轴并写入索引信息
        for i, ax in enumerate(data.axes):
            # 如果是第一个轴且索引不唯一，则抛出错误
            if i == 0 and (not ax.is_unique):
                raise ValueError("Columns index has to be unique for fixed format")
            self.write_index(f"axis{i}", ax)

        # 将数据块的数量存储到属性中
        self.attrs.nblocks = len(data.blocks)
        # 遍历数据的块并写入值和 items
        for i, blk in enumerate(data.blocks):
            # 修复问题 #2299：在写入 items 之前需要先写入值
            blk_items = data.items.take(blk.mgr_locs)
            self.write_array(f"block{i}_values", blk.values, items=blk_items)
            self.write_index(f"block{i}_items", blk_items)
class FrameFixed(BlockManagerFixed):
    pandas_kind = "frame"
    obj_type = DataFrame


class Table(Fixed):
    """
    represent a table:
        facilitate read/write of various types of tables

    Attrs in Table Node
    -------------------
    These are attributes that are store in the main table node, they are
    necessary to recreate these tables when read back in.

    index_axes    : a list of tuples of the (original indexing axis and
        index column)
    non_index_axes: a list of tuples of the (original index axis and
        columns on a non-indexing axis)
    values_axes   : a list of the columns which comprise the data of this
        table
    data_columns  : a list of the columns that we are allowing indexing
        (these become single columns in values_axes)
    nan_rep       : the string to use for nan representations for string
        objects
    levels        : the names of levels
    metadata      : the names of the metadata columns
    """

    pandas_kind = "wide_table"
    format_type: str = "table"  # GH#30962 needed by dask
    table_type: str
    levels: int | list[Hashable] = 1
    is_table = True

    metadata: list

    def __init__(
        self,
        parent: HDFStore,
        group: Node,
        encoding: str | None = None,
        errors: str = "strict",
        index_axes: list[IndexCol] | None = None,
        non_index_axes: list[tuple[AxisInt, Any]] | None = None,
        values_axes: list[DataCol] | None = None,
        data_columns: list | None = None,
        info: dict | None = None,
        nan_rep=None,
    ) -> None:
        """
        Initialize a Table object.

        Parameters:
        - parent: The parent HDFStore object managing this table.
        - group: The Node representing this table in HDFStore.
        - encoding: Optional encoding for data.
        - errors: Error handling strategy for data operations.
        - index_axes: List of IndexCol objects representing original indexing axes.
        - non_index_axes: List of tuples representing original index axes and columns on non-indexing axes.
        - values_axes: List of DataCol objects representing columns comprising table data.
        - data_columns: List of columns allowing indexing.
        - info: Additional metadata.
        - nan_rep: String representation for NaN values.
        """
        super().__init__(parent, group, encoding=encoding, errors=errors)
        self.index_axes = index_axes or []  # Initialize or set index_axes to an empty list if None.
        self.non_index_axes = non_index_axes or []  # Initialize or set non_index_axes to an empty list if None.
        self.values_axes = values_axes or []  # Initialize or set values_axes to an empty list if None.
        self.data_columns = data_columns or []  # Initialize or set data_columns to an empty list if None.
        self.info = info or {}  # Initialize or set info to an empty dictionary if None.
        self.nan_rep = nan_rep  # Store the representation for NaN values.

    @property
    def table_type_short(self) -> str:
        """
        Return a short representation of the table type based on splitting table_type by '_'.

        Returns:
        - str: Short representation of the table type.
        """
        return self.table_type.split("_")[0]

    def __repr__(self) -> str:
        """
        Return a string representation of the Table object.

        This representation includes pandas_type, version (if old version), table_type_short,
        number of rows and columns, indexers, and data columns.

        Returns:
        - str: String representation of the Table object.
        """
        self.infer_axes()  # Infer axes information.
        jdc = ",".join(self.data_columns) if len(self.data_columns) else ""  # Join data columns into a string.
        dc = f",dc->[{jdc}]"  # Format data columns string.

        ver = ""
        if self.is_old_version:
            jver = ".".join([str(x) for x in self.version])
            ver = f"[{jver}]"  # Format version information if it's an old version.

        jindex_axes = ",".join([a.name for a in self.index_axes])  # Join index axes names into a string.
        return (
            f"{self.pandas_type:12.12}{ver} "
            f"(typ->{self.table_type_short},nrows->{self.nrows},"
            f"ncols->{self.ncols},indexers->[{jindex_axes}]{dc})"
        )  # Return formatted representation string.

    def __getitem__(self, c: str):
        """
        Retrieve the axis corresponding to the given column name.

        Parameters:
        - c: Column name to retrieve.

        Returns:
        - Axis object corresponding to the column name, or None if not found.
        """
        for a in self.axes:
            if c == a.name:
                return a  # Return the axis object if column name matches.
        return None  # Return None if column name is not found.
    def validate(self, other) -> None:
        """
        validate against an existing table
        """
        # 如果参数 other 为 None，则直接返回，不进行验证
        if other is None:
            return
        
        # 检查其他表格对象的 table_type 是否与当前对象的 table_type 相同
        if other.table_type != self.table_type:
            # 如果类型不匹配，则抛出 TypeError 异常
            raise TypeError(
                "incompatible table_type with existing "
                f"[{other.table_type} - {self.table_type}]"
            )
        
        # 遍历要比较的属性列表
        for c in ["index_axes", "non_index_axes", "values_axes"]:
            # 获取当前对象和其他对象的属性值
            sv = getattr(self, c, None)
            ov = getattr(other, c, None)
            
            # 如果当前属性值与其他属性值不相等，则进行详细比较
            if sv != ov:
                # 遍历当前属性值的索引和值，与其他属性值的相应元素进行比较
                for i, sax in enumerate(sv):  # type: ignore[arg-type]
                    # 获取其他属性值对应索引的值
                    oax = ov[i]  # type: ignore[index]
                    
                    # 如果当前值与其他值不相等，则抛出 ValueError 异常
                    if sax != oax:
                        raise ValueError(
                            f"invalid combination of [{c}] on appending data "
                            f"[{sax}] vs current table [{oax}]"
                        )
                
                # 如果程序执行到这里，则抛出异常，表示出现了意料之外的情况
                raise Exception(
                    f"invalid combination of [{c}] on appending data [{sv}] vs "
                    f"current table [{ov}]"
                )

    @property
    def is_multi_index(self) -> bool:
        """
        the levels attribute is 1 or a list in the case of a multi-index
        """
        # 判断 levels 属性是否为列表，从而确定是否为多级索引
        return isinstance(self.levels, list)

    def validate_multiindex(
        self, obj: DataFrame | Series
    ) -> tuple[DataFrame, list[Hashable]]:
        """
        validate that we can store the multi-index; reset and return the
        new object
        """
        # 填充缺失的索引名称，并重置对象
        levels = com.fill_missing_names(obj.index.names)
        try:
            # 尝试将对象重置为 DataFrame
            reset_obj = obj.reset_index()
        except ValueError as err:
            # 如果在重置过程中遇到重复的名称或列，则抛出 ValueError 异常
            raise ValueError(
                "duplicate names/columns in the multi-index when storing as a table"
            ) from err
        assert isinstance(reset_obj, DataFrame)  # for mypy
        # 返回重置后的对象和填充后的索引名称列表
        return reset_obj, levels

    @property
    def nrows_expected(self) -> int:
        """
        based on our axes, compute the expected nrows
        """
        # 计算预期的行数，基于索引轴的形状信息
        return np.prod([i.cvalues.shape[0] for i in self.index_axes])

    @property
    def is_exists(self) -> bool:
        """
        has this table been created
        """
        # 检查当前表是否已经创建（存在于 group 属性中）
        return "table" in self.group

    @property
    def storable(self):
        # 返回 group 对象的 table 属性
        return getattr(self.group, "table", None)

    @property
    def table(self):
        """
        return the table group (this is my storable)
        """
        # 返回当前对象的 storable 属性，用于表示表格组
        return self.storable

    @property
    def dtype(self):
        # 返回表格的数据类型（由 table 属性决定）
        return self.table.dtype

    @property
    def description(self):
        # 返回表格的描述信息（由 table 属性决定）
        return self.table.description

    @property
    def axes(self) -> itertools.chain[IndexCol]:
        """
        return an iterator over all axes (index and values axes)
        """
        # 返回索引和值轴的迭代器链
        return itertools.chain(self.index_axes, self.values_axes)
    def ncols(self) -> int:
        """返回值轴中所有列的总数"""
        # 计算所有值轴中值的数量之和作为总列数
        return sum(len(a.values) for a in self.values_axes)

    @property
    def is_transposed(self) -> bool:
        """返回是否转置的布尔值，始终为 False"""
        # 始终返回 False，表示未转置
        return False

    @property
    def data_orientation(self) -> tuple[int, ...]:
        """返回一个元组，包含重新排列后的轴及其索引的信息"""
        # 使用 itertools.chain 组合非索引轴的第一个元素和索引轴的轴号，返回元组
        return tuple(
            itertools.chain(
                [int(a[0]) for a in self.non_index_axes],
                [int(a.axis) for a in self.index_axes],
            )
        )

    def queryables(self) -> dict[str, Any]:
        """返回一个字典，包含此对象可用的各种查询列的信息"""
        # 构建查询列的字典，包括索引轴、非索引轴和数据轴中指定的列
        axis_names = {0: "index", 1: "columns"}  # 定义轴号与名称的映射关系
        d1 = [(a.cname, a) for a in self.index_axes]  # 索引轴列及其对象的列表
        d2 = [(axis_names[axis], None) for axis, values in self.non_index_axes]  # 非索引轴列及其对象的列表
        d3 = [(v.cname, v) for v in self.values_axes if v.name in set(self.data_columns)]  # 数据轴中指定列及其对象的列表
        return dict(d1 + d2 + d3)  # 返回字典，包含所有查询列的信息

    def index_cols(self) -> list[tuple[Any, Any]]:
        """返回一个列表，包含此对象的索引列"""
        # 返回索引列的列表，每个元素是一个元组，包含轴号和列名
        return [(i.axis, i.cname) for i in self.index_axes]

    def values_cols(self) -> list[str]:
        """返回一个列表，包含此对象的值列名"""
        # 返回值列的列名列表
        return [i.cname for i in self.values_axes]

    def _get_metadata_path(self, key: str) -> str:
        """返回给定键的元数据路径名"""
        # 构建元数据的路径，格式为 "{group}/meta/{key}/meta"
        group = self.group._v_pathname
        return f"{group}/meta/{key}/meta"

    def write_metadata(self, key: str, values: np.ndarray) -> None:
        """
        将元数据数组按固定格式写入指定键的 Series 中。

        Parameters
        ----------
        key : str
            元数据的键名
        values : ndarray
            要写入的元数据数组
        """
        # 将元数据数组以固定格式写入指定路径下的 Series 中
        self.parent.put(
            self._get_metadata_path(key),
            Series(values, copy=False),
            format="table",
            encoding=self.encoding,
            errors=self.errors,
            nan_rep=self.nan_rep,
        )

    def read_metadata(self, key: str):
        """返回给定键的元数据数组"""
        # 如果元数据存在，则返回对应路径下的元数据数组；否则返回 None
        if getattr(getattr(self.group, "meta", None), key, None) is not None:
            return self.parent.select(self._get_metadata_path(key))
        return None

    def set_attrs(self) -> None:
        """设置表类型及其可索引属性"""
        # 设置对象的属性，包括表类型、索引列、值列、非索引轴、数据列、NaN 表示、编码、错误处理、级别、信息
        self.attrs.table_type = str(self.table_type)
        self.attrs.index_cols = self.index_cols()
        self.attrs.values_cols = self.values_cols()
        self.attrs.non_index_axes = self.non_index_axes
        self.attrs.data_columns = self.data_columns
        self.attrs.nan_rep = self.nan_rep
        self.attrs.encoding = self.encoding
        self.attrs.errors = self.errors
        self.attrs.levels = self.levels
        self.attrs.info = self.info
    def get_attrs(self) -> None:
        """retrieve our attributes"""
        # 获取或设置非索引轴列表，如果未定义则设置为空列表
        self.non_index_axes = getattr(self.attrs, "non_index_axes", None) or []
        # 获取或设置数据列列表，如果未定义则设置为空列表
        self.data_columns = getattr(self.attrs, "data_columns", None) or []
        # 获取或设置信息字典，如果未定义则设置为空字典
        self.info = getattr(self.attrs, "info", None) or {}
        # 获取或设置NaN表示方式
        self.nan_rep = getattr(self.attrs, "nan_rep", None)
        # 获取或设置编码方式，并确保编码合法
        self.encoding = _ensure_encoding(getattr(self.attrs, "encoding", None))
        # 获取或设置错误处理方式，默认为严格模式
        self.errors = getattr(self.attrs, "errors", "strict")
        # 获取或设置层级列表，如果未定义则设置为空列表
        self.levels: list[Hashable] = getattr(self.attrs, "levels", None) or []
        # 根据可索引属性列表生成索引轴列表
        self.index_axes = [a for a in self.indexables if a.is_an_indexable]
        # 根据可索引属性列表生成数值轴列表
        self.values_axes = [a for a in self.indexables if not a.is_an_indexable]

    def validate_version(self, where=None) -> None:
        """are we trying to operate on an old version?"""
        # 如果where参数不为None，且当前对象是旧版本
        if where is not None:
            if self.is_old_version:
                # 构建不兼容警告信息
                ws = incompatibility_doc % ".".join([str(x) for x in self.version])
                # 发出警告
                warnings.warn(
                    ws,
                    IncompatibilityWarning,
                    stacklevel=find_stack_level(),
                )

    def validate_min_itemsize(self, min_itemsize) -> None:
        """
        validate the min_itemsize doesn't contain items that are not in the
        axes this needs data_columns to be defined
        """
        # 如果min_itemsize为None，直接返回
        if min_itemsize is None:
            return
        # 如果min_itemsize不是字典，直接返回
        if not isinstance(min_itemsize, dict):
            return

        # 获取查询属性的集合
        q = self.queryables()
        # 遍历min_itemsize的键
        for k in min_itemsize:
            # 对于特定键"values"，跳过检查
            if k == "values":
                continue
            # 如果键不在查询属性集合中，引发数值错误
            if k not in q:
                raise ValueError(
                    f"min_itemsize has the key [{k}] which is not an axis or "
                    "data_column"
                )

    @cache_readonly
    # 创建或缓存索引对象列表 `_indexables`，如果它们不存在的话
    _indexables = []

    # 从 `self` 对象中获取 `description` 属性
    desc = self.description
    # 从 `self.table` 对象中获取 `attrs` 属性
    table_attrs = self.table.attrs

    # Note: 下面每个 `name` 参数都由 `index_cols` 中的定义保证为字符串类型
    # 遍历索引列
    for i, (axis, name) in enumerate(self.attrs.index_cols):
        # 从 `desc` 对象中获取 `name` 对应的属性值 `atom`
        atom = getattr(desc, name)
        # 读取名为 `name` 的元数据
        md = self.read_metadata(name)
        # 如果存在元数据 `md`，则将 `meta` 设置为 "category"，否则为 `None`
        meta = "category" if md is not None else None

        # 构造属性名为 `name` 的变量名 `kind_attr`
        kind_attr = f"{name}_kind"
        # 从 `table_attrs` 中获取属性名为 `kind_attr` 的值赋给 `kind`
        kind = getattr(table_attrs, kind_attr, None)

        # 创建 `IndexCol` 对象并添加到 `_indexables` 列表中
        index_col = IndexCol(
            name=name,
            axis=axis,
            pos=i,
            kind=kind,
            typ=atom,
            table=self.table,
            meta=meta,
            metadata=md,
        )
        _indexables.append(index_col)

    # Note: 定义 `values_cols` 确保下面每个 `c` 都是字符串类型
    # 处理值列
    dc = set(self.data_columns)
    base_pos = len(_indexables)

    # 定义函数 `f` 处理每个值列 `c`
    def f(i, c: str) -> DataCol:
        assert isinstance(c, str)
        # 默认为 `DataCol` 类，如果 `c` 在 `dc` 中，则为 `DataIndexableCol` 类
        klass = DataCol
        if c in dc:
            klass = DataIndexableCol

        # 从 `desc` 对象中获取属性名为 `c` 的值赋给 `atom`
        atom = getattr(desc, c)
        # 根据 `self.version` 可能调整 `c` 的名称为 `adj_name`
        adj_name = _maybe_adjust_name(c, self.version)

        # TODO: 这里为什么需要 `kind_attr`？
        # 从 `table_attrs` 中获取属性名为 `{adj_name}_kind` 的值赋给 `values`
        values = getattr(table_attrs, f"{adj_name}_kind", None)
        # 从 `table_attrs` 中获取属性名为 `{adj_name}_dtype` 的值赋给 `dtype`
        dtype = getattr(table_attrs, f"{adj_name}_dtype", None)
        # 将 `dtype` 转换为对应的 `kind` 类型
        kind = _dtype_to_kind(dtype)  # type: ignore[arg-type]

        # 读取名为 `c` 的元数据
        md = self.read_metadata(c)
        # 从 `table_attrs` 中获取属性名为 `{adj_name}_meta` 的值赋给 `meta`
        meta = getattr(table_attrs, f"{adj_name}_meta", None)

        # 创建 `DataCol` 或 `DataIndexableCol` 对象并返回
        obj = klass(
            name=adj_name,
            cname=c,
            values=values,
            kind=kind,
            pos=base_pos + i,
            typ=atom,
            table=self.table,
            meta=meta,
            metadata=md,
            dtype=dtype,
        )
        return obj

    # 遍历处理每个值列 `self.attrs.values_cols` 中的每个 `c`
    _indexables.extend([f(i, c) for i, c in enumerate(self.attrs.values_cols)])

    # 返回构建好的索引对象列表 `_indexables`
    return _indexables
    def _create_axes(
        self,
        axes,
        obj: DataFrame,
        validate: bool = True,
        nan_rep=None,
        data_columns=None,
        min_itemsize=None,
    ):
        """
        Create the axes based on provided parameters and DataFrame object.

        Parameters
        ----------
        axes : list
            List of axes to be created.
        obj : DataFrame
            The DataFrame object to use for axis creation.
        validate : bool, default True
            Whether to perform validation during axis creation.
        nan_rep : any, optional
            Value to represent NaN (Not a Number) values.
        data_columns : list or None, optional
            Columns of data to use for axis creation.
        min_itemsize : any, optional
            Minimum item size to consider during axis creation.

        Returns
        -------
        List
            List of created axes based on the parameters.
        """

    @staticmethod
    def _get_blocks_and_items(
        frame: DataFrame,
        table_exists: bool,
        new_non_index_axes,
        values_axes,
        data_columns,
    ) -> list:
        """
        Extract blocks and items from the DataFrame based on provided parameters.

        Parameters
        ----------
        frame : DataFrame
            The DataFrame from which blocks and items are extracted.
        table_exists : bool
            Flag indicating if the table exists.
        new_non_index_axes : any
            New non-index axes for consideration.
        values_axes : any
            Axes of values to be processed.
        data_columns : any
            Columns of data to be used.

        Returns
        -------
        List
            Extracted blocks and items based on the input DataFrame.
        """
        # Helper to clarify non-state-altering parts of _create_axes
        # 定义一个辅助函数，用于澄清 _create_axes 中不会改变状态的部分

        def get_blk_items(mgr):
            # Retrieve items from the manager's blocks
            return [mgr.items.take(blk.mgr_locs) for blk in mgr.blocks]
            # 返回由管理器中每个块的位置索引组成的项目列表

        mgr = frame._mgr
        # Obtain the internal manager from the frame
        blocks: list[Block] = list(mgr.blocks)
        # Create a list of blocks from the manager

        blk_items: list[Index] = get_blk_items(mgr)
        # Obtain block items using the helper function

        if len(data_columns):
            # Check if data_columns has any elements

            # TODO: prove that we only get here with axis == 1?
            #  It is the case in all extant tests, but NOT the case
            #  outside this `if len(data_columns)` check.
            # 
            # 证明我们只有在 axis == 1 的情况下才会进入此处？
            # 在所有现有的测试中都是这样，但在不在 `if len(data_columns)` 检查之外的情况下是不是也是如此。

            axis, axis_labels = new_non_index_axes[0]
            # Retrieve axis and axis_labels from new_non_index_axes

            new_labels = Index(axis_labels).difference(Index(data_columns))
            # Create new_labels by finding the difference between axis_labels and data_columns
            mgr = frame.reindex(new_labels, axis=axis)._mgr
            # Reindex the frame based on new_labels and axis, then get the internal manager

            blocks = list(mgr.blocks)
            # Update blocks with the blocks from the reindexed manager
            blk_items = get_blk_items(mgr)
            # Update blk_items using the helper function with the reindexed manager

            for c in data_columns:
                # Iterate through each column in data_columns

                # This reindex would raise ValueError if we had a duplicate
                #  index, so we can infer that (as long as axis==1) we
                #  get a single column back, so a single block.
                # 如果存在重复的索引，此重新索引将引发 ValueError，因此我们可以推断（只要 axis==1），我们将得到单个列，因此是单个块。

                mgr = frame.reindex([c], axis=axis)._mgr
                # Reindex the frame for the column c based on axis, then get the internal manager
                blocks.extend(mgr.blocks)
                # Extend blocks with the blocks from the reindexed manager
                blk_items.extend(get_blk_items(mgr))
                # Extend blk_items using the helper function with the reindexed manager

        # reorder the blocks in the same order as the existing table if we can
        # 如果可能的话，按照现有表格的顺序重新排序块

        if table_exists:
            # Check if table_exists is True

            by_items = {
                tuple(b_items.tolist()): (b, b_items)
                for b, b_items in zip(blocks, blk_items)
            }
            # Create a dictionary by_items where keys are tuples of block items converted to lists and values are tuples of blocks and block items

            new_blocks: list[Block] = []
            new_blk_items = []
            for ea in values_axes:
                # Iterate through each item in values_axes

                items = tuple(ea.values)
                # Create items tuple from ea.values

                try:
                    b, b_items = by_items.pop(items)
                    # Pop corresponding block and block items from by_items based on items
                    new_blocks.append(b)
                    # Append block b to new_blocks
                    new_blk_items.append(b_items)
                    # Append block items b_items to new_blk_items
                except (IndexError, KeyError) as err:
                    jitems = ",".join([pprint_thing(item) for item in items])
                    # Join items into a string jitems for error message
                    raise ValueError(
                        f"cannot match existing table structure for [{jitems}] "
                        "on appending data"
                    ) from err
                    # Raise a ValueError with a detailed message if a match for existing table structure cannot be found

            blocks = new_blocks
            # Update blocks with new_blocks
            blk_items = new_blk_items
            # Update blk_items with new_blk_items

        return blocks, blk_items
        # Return blocks and blk_items as the final result of the function
    # 处理对象的轴过滤器
    def process_axes(self, obj, selection: Selection, columns=None) -> DataFrame:
        """process axes filters"""
        # 复制一份以避免副作用
        if columns is not None:
            columns = list(columns)

        # 如果存在多级索引，确保包含级别
        if columns is not None and self.is_multi_index:
            assert isinstance(self.levels, list)  # 由 is_multi_index 保证
            # 将级别插入到列列表中，以确保顺序正确
            for n in self.levels:
                if n not in columns:
                    columns.insert(0, n)

        # 根据非索引轴重新排序并限制选择的列
        for axis, labels in self.non_index_axes:
            # 重新索引轴，并根据列重新排序
            obj = _reindex_axis(obj, axis, labels, columns)

            def process_filter(field, filt, op):
                # 遍历对象的所有轴，查找匹配的轴名或轴值
                for axis_name in obj._AXIS_ORDERS:
                    axis_number = obj._get_axis_number(axis_name)
                    axis_values = obj._get_axis(axis_name)
                    assert axis_number is not None

                    # 如果字段是一个轴的名称
                    if field == axis_name:
                        # 对于多级索引，需要包含级别
                        if self.is_multi_index:
                            filt = filt.union(Index(self.levels))

                        # 使用给定的操作符对轴值进行过滤，并返回匹配的子集
                        takers = op(axis_values, filt)
                        return obj.loc(axis=axis_number)[takers]

                    # 如果字段是某个轴的值
                    elif field in axis_values:
                        # 需要在这个维度上进行过滤
                        values = ensure_index(getattr(obj, field).values)
                        filt = ensure_index(filt)

                        # 暂时处理，直到支持反向维度标志
                        if isinstance(obj, DataFrame):
                            axis_number = 1 - axis_number

                        # 使用给定的操作符对轴值进行过滤，并返回匹配的子集
                        takers = op(values, filt)
                        return obj.loc(axis=axis_number)[takers]

                # 如果未找到匹配的轴或值，则抛出异常
                raise ValueError(f"cannot find the field [{field}] for filtering!")

        # 应用选择过滤器（但保持顺序不变）
        if selection.filter is not None:
            # 遍历选择器中的过滤器字段、操作符和过滤条件
            for field, op, filt in selection.filter.format():
                obj = process_filter(field, filt, op)

        # 返回处理后的对象
        return obj

    def create_description(
        self,
        complib,
        complevel: int | None,
        fletcher32: bool,
        expectedrows: int | None,
    ) -> dict[str, Any]:
        """从轴和数值创建表的描述"""
        # 如果未提供预期行数，则使用最大值和预期的默认行数
        if expectedrows is None:
            expectedrows = max(self.nrows_expected, 10000)

        # 创建表的描述字典
        d = {"name": "table", "expectedrows": expectedrows}

        # 从轴和数值中获取表的描述信息
        d["description"] = {a.cname: a.typ for a in self.axes}

        # 如果指定了压缩库，设置压缩级别和过滤器
        if complib:
            if complevel is None:
                complevel = self._complevel or 9
            filters = _tables().Filters(
                complevel=complevel,
                complib=complib,
                fletcher32=fletcher32 or self._fletcher32,
            )
            d["filters"] = filters
        # 否则，如果存在自定义的过滤器，则使用它
        elif self._filters is not None:
            d["filters"] = self._filters

        # 返回表的描述信息字典
        return d

    def read_coordinates(
        self, where=None, start: int | None = None, stop: int | None = None
    ):
        """
        从表中选择坐标（行号）；返回坐标对象
        """
        # 验证版本信息
        self.validate_version(where)

        # 推断数据类型
        if not self.infer_axes():
            return False

        # 创建选择对象
        selection = Selection(self, where=where, start=start, stop=stop)
        # 选择坐标
        coords = selection.select_coords()
        # 如果存在过滤器，则应用过滤器
        if selection.filter is not None:
            for field, op, filt in selection.filter.format():
                data = self.read_column(
                    field, start=coords.min(), stop=coords.max() + 1
                )
                coords = coords[op(data.iloc[coords - coords.min()], filt).values]

        # 返回坐标的索引对象
        return Index(coords)

    def read_column(
        self,
        column: str,
        where=None,
        start: int | None = None,
        stop: int | None = None,
    ):
        """
        return a single column from the table, generally only indexables
        are interesting
        """
        # 验证版本信息的有效性
        self.validate_version()

        # 推断数据的种类
        if not self.infer_axes():
            return False

        # 如果有where参数，则抛出TypeError异常
        if where is not None:
            raise TypeError("read_column does not currently accept a where clause")

        # 查找列轴
        for a in self.axes:
            # 如果找到与指定列名相符的列轴
            if column == a.name:
                # 如果该列不可索引，则抛出ValueError异常
                if not a.is_data_indexable:
                    raise ValueError(
                        f"column [{column}] can not be extracted individually; "
                        "it is not data indexable"
                    )

                # 获取指定列的数据对象
                c = getattr(self.table.cols, column)
                # 设置列轴的附加信息
                a.set_info(self.info)
                # 转换列的数据值
                col_values = a.convert(
                    c[start:stop],
                    nan_rep=self.nan_rep,
                    encoding=self.encoding,
                    errors=self.errors,
                )
                # 提取转换后的数据值
                cvs = col_values[1]
                # 返回一个Series对象，代表提取的列数据
                return Series(cvs, name=column, copy=False)

        # 若未找到指定的列名，则抛出KeyError异常
        raise KeyError(f"column [{column}] not found in the table")
class WORMTable(Table):
    """
    a write-once read-many table: this format DOES NOT ALLOW appending to a
    table. writing is a one-time operation the data are stored in a format
    that allows for searching the data on disk
    """

    table_type = "worm"

    def read(
        self,
        where=None,
        columns=None,
        start: int | None = None,
        stop: int | None = None,
    ):
        """
        read the indices and the indexing array, calculate offset rows and return
        """
        # 抛出未实现错误，需要子类实现具体的读取方法
        raise NotImplementedError("WORMTable needs to implement read")

    def write(self, obj, **kwargs) -> None:
        """
        write in a format that we can search later on (but cannot append
        to): write out the indices and the values using _write_array
        (e.g. a CArray) create an indexing table so that we can search
        """
        # 抛出未实现错误，需要子类实现具体的写入方法
        raise NotImplementedError("WORMTable needs to implement write")


class AppendableTable(Table):
    """support the new appendable table formats"""

    table_type = "appendable"

    # error: Signature of "write" incompatible with supertype "Fixed"
    def write(  # type: ignore[override]
        self,
        obj,
        axes=None,
        append: bool = False,
        complib=None,
        complevel=None,
        fletcher32=None,
        min_itemsize=None,
        chunksize: int | None = None,
        expectedrows=None,
        dropna: bool = False,
        nan_rep=None,
        data_columns=None,
        track_times: bool = True,
    ) -> None:
        """
        Write data to the table. If append is False and the table exists, remove the existing table.
        Create axes, validate names, and create or update the table with specified options.
        Finally, validate axes, set kinds, and write data to the table.
        """
        if not append and self.is_exists:
            # 如果不是追加模式并且表已存在，则移除现有表
            self._handle.remove_node(self.group, "table")

        # 创建轴（axes）
        table = self._create_axes(
            axes=axes,
            obj=obj,
            validate=append,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            data_columns=data_columns,
        )

        for a in table.axes:
            # 验证轴的名称是否有效
            a.validate_names()

        if not table.is_exists:
            # 如果表不存在，则创建表
            options = table.create_description(
                complib=complib,
                complevel=complevel,
                fletcher32=fletcher32,
                expectedrows=expectedrows,
            )

            # 设置表的属性
            table.set_attrs()

            options["track_times"] = track_times

            # 创建表
            table._handle.create_table(table.group, **options)

        # 更新信息
        table.attrs.info = table.info

        # 验证轴并设置类型
        for a in table.axes:
            a.validate_and_set(table, append)

        # 添加行数据
        table.write_data(chunksize, dropna=dropna)
    def write_data(self, chunksize: int | None, dropna: bool = False) -> None:
        """
        将数据按块写入，包括索引、数值和掩码
        """
        # 获取数据的字段名
        names = self.dtype.names
        # 预期的行数
        nrows = self.nrows_expected

        # 如果 dropna==True，则删除所有包含 NaN 的行
        masks = []
        if dropna:
            for a in self.values_axes:
                # 计算掩码：仅在成功处理列时使用，否则忽略掩码
                mask = isna(a.data).all(axis=0)
                if isinstance(mask, np.ndarray):
                    masks.append(mask.astype("u1", copy=False))

        # 合并掩码
        if len(masks):
            mask = masks[0]
            for m in masks[1:]:
                mask = mask & m
            mask = mask.ravel()
        else:
            mask = None

        # 如果需要，广播索引
        indexes = [a.cvalues for a in self.index_axes]
        nindexes = len(indexes)
        assert nindexes == 1, nindexes  # 确保我们不需要广播索引

        # 调整数值使得第一个维度是最后一个
        # 根据需要重塑数值
        values = [a.take_data() for a in self.values_axes]
        values = [v.transpose(np.roll(np.arange(v.ndim), v.ndim - 1)) for v in values]
        bvalues = []
        for i, v in enumerate(values):
            new_shape = (nrows,) + self.dtype[names[nindexes + i]].shape
            bvalues.append(v.reshape(new_shape))

        # 写入数据块
        if chunksize is None:
            chunksize = 100000

        rows = np.empty(min(chunksize, nrows), dtype=self.dtype)
        chunks = nrows // chunksize + 1
        for i in range(chunks):
            start_i = i * chunksize
            end_i = min((i + 1) * chunksize, nrows)
            if start_i >= end_i:
                break

            self.write_data_chunk(
                rows,
                indexes=[a[start_i:end_i] for a in indexes],
                mask=mask[start_i:end_i] if mask is not None else None,
                values=[v[start_i:end_i] for v in bvalues],
            )

    def write_data_chunk(
        self,
        rows: np.ndarray,
        indexes: list[np.ndarray],
        mask: npt.NDArray[np.bool_] | None,
        values: list[np.ndarray],
        ```

        rows 参数：要写入的行数据，numpy 数组类型。
        indexes 参数：索引列表，包含要写入的索引数据的 numpy 数组。
        mask 参数：掩码数组，用于指示哪些行要写入（可选），numpy 布尔数组或 None。
        values 参数：数值列表，包含要写入的数值数据的 numpy 数组。
    ) -> None:
        """
        Parameters
        ----------
        rows : an empty memory space where we are putting the chunk
        indexes : an array of the indexes
        mask : an array of the masks
        values : an array of the values
        """
        # 0 len
        # 检查每个值的形状是否为空，如果有空值则返回
        for v in values:
            if not np.prod(v.shape):
                return

        nrows = indexes[0].shape[0]
        if nrows != len(rows):
            # 如果行数不等于指定的长度，则重新分配空间
            rows = np.empty(nrows, dtype=self.dtype)
        names = self.dtype.names
        nindexes = len(indexes)

        # indexes
        # 将索引数组中的值存储到对应的列中
        for i, idx in enumerate(indexes):
            rows[names[i]] = idx

        # values
        # 将数值数组中的值存储到对应的列中
        for i, v in enumerate(values):
            rows[names[i + nindexes]] = v

        # mask
        # 如果存在掩码数组，则根据掩码数组进行行过滤
        if mask is not None:
            m = ~mask.ravel().astype(bool, copy=False)
            if not m.all():
                rows = rows[m]

        if len(rows):
            # 将处理好的行添加到表格中并刷新
            self.table.append(rows)
            self.table.flush()

    def delete(
        self, where=None, start: int | None = None, stop: int | None = None
    ) -> int | None:
        # delete all rows (and return the nrows)
        # 如果未指定删除条件或条件为空，则删除所有行并返回删除的行数
        if where is None or not len(where):
            if start is None and stop is None:
                nrows = self.nrows
                # 删除表中的所有行
                self._handle.remove_node(self.group, recursive=True)
            else:
                # 在旧版 pytables<3.0 中，如果 stop=None 则删除单行
                if stop is None:
                    stop = self.nrows
                # 删除指定范围内的行，并返回删除的行数
                nrows = self.table.remove_rows(start=start, stop=stop)
                self.table.flush()
            return nrows

        # infer the data kind
        # 推断数据类型
        if not self.infer_axes():
            return None

        # create the selection
        # 创建选择器
        table = self.table
        selection = Selection(self, where, start=start, stop=stop)
        values = selection.select_coords()

        # delete the rows in reverse order
        # 按照逆序删除行
        sorted_series = Series(values, copy=False).sort_values()
        ln = len(sorted_series)

        if ln:
            # construct groups of consecutive rows
            # 构建连续行的组
            diff = sorted_series.diff()
            groups = list(diff[diff > 1].index)

            # 1 group
            # 如果没有组，则默认一个组
            if not len(groups):
                groups = [0]

            # final element
            # 最后一个元素
            if groups[-1] != ln:
                groups.append(ln)

            # initial element
            # 初始元素
            if groups[0] != 0:
                groups.insert(0, 0)

            # we must remove in reverse order!
            # 必须按逆序删除
            pg = groups.pop()
            for g in reversed(groups):
                rows = sorted_series.take(range(g, pg))
                table.remove_rows(
                    start=rows[rows.index[0]], stop=rows[rows.index[-1]] + 1
                )
                pg = g

            self.table.flush()

        # return the number of rows removed
        # 返回删除的行数
        return ln
# 定义一个名为 AppendableFrameTable 的类，继承自 AppendableTable 类
class AppendableFrameTable(AppendableTable):
    """support the new appendable table formats"""
    
    # 设置类属性 pandas_kind，表示 Pandas 对象的类型为 'frame_table'
    pandas_kind = "frame_table"
    
    # 设置类属性 table_type，表示表格类型为 'appendable_frame'
    table_type = "appendable_frame"
    
    # 设置类属性 ndim，表示对象的维度为 2
    ndim = 2
    
    # 设置类属性 obj_type，指定对象类型为 DataFrame 或 Series
    obj_type: type[DataFrame | Series] = DataFrame
    
    # 定义 is_transposed 属性的 getter 方法，返回是否已经转置的布尔值
    @property
    def is_transposed(self) -> bool:
        return self.index_axes[0].axis == 1

    # 定义一个类方法 get_object，用于获取对象，并在需要时进行转置
    @classmethod
    def get_object(cls, obj, transposed: bool):
        """these are written transposed"""
        # 如果 transposed 为 True，则对 obj 进行转置操作
        if transposed:
            obj = obj.T
        # 返回处理后的 obj 对象
        return obj

    # 定义 read 方法，用于读取表格数据的特定部分
    def read(
        self,
        where=None,
        columns=None,
        start: int | None = None,
        stop: int | None = None,
        # validate the version
        self.validate_version(where)

        # infer the data kind
        # 推断数据类型的轴信息
        if not self.infer_axes():
            return None

        # 读取轴数据
        result = self._read_axes(where=where, start=start, stop=stop)

        # 获取第一个非索引轴的信息
        info = (
            self.info.get(self.non_index_axes[0][0], {})
            if len(self.non_index_axes)
            else {}
        )

        # 找到索引轴的索引
        inds = [i for i, ax in enumerate(self.axes) if ax is self.index_axes[0]]
        assert len(inds) == 1
        ind = inds[0]

        # 获取索引轴的值
        index = result[ind][0]

        # 初始化空列表以存储数据框
        frames = []

        # 遍历轴
        for i, a in enumerate(self.axes):
            # 如果轴不在值轴中，则跳过
            if a not in self.values_axes:
                continue

            # 获取索引值和列值
            index_vals, cvalues = result[i]

            # 如果不是 MultiIndex 类型，则创建 Index 对象
            if info.get("type") != "MultiIndex":
                cols = Index(index_vals)
            else:
                cols = MultiIndex.from_tuples(index_vals)

            # 设置列名
            names = info.get("names")
            if names is not None:
                cols.set_names(names, inplace=True)

            # 根据是否转置来选择数据和索引的处理方式
            if self.is_transposed:
                values = cvalues
                index_ = cols
                cols_ = Index(index, name=getattr(index, "name", None))
            else:
                values = cvalues.T
                index_ = Index(index, name=getattr(index, "name", None))
                cols_ = cols

            # 如果值的维度为1且是 ndarray 类型，则重塑其形状
            if values.ndim == 1 and isinstance(values, np.ndarray):
                values = values.reshape((1, values.shape[0]))

            # 根据值的类型创建 DataFrame 对象
            if isinstance(values, (np.ndarray, DatetimeArray)):
                df = DataFrame(values.T, columns=cols_, index=index_, copy=False)
            elif isinstance(values, Index):
                df = DataFrame(values, columns=cols_, index=index_)
            else:
                # 如果是分类数据
                df = DataFrame._from_arrays([values], columns=cols_, index=index_)

            # 检查数据类型是否与预期一致
            if not (using_pyarrow_string_dtype() and values.dtype.kind == "O"):
                assert (df.dtypes == values.dtype).all(), (df.dtypes, values.dtype)

            # 如果使用 PyArrow 并且值是字符串数组，则转换为 PyArrow 字符串类型
            if using_pyarrow_string_dtype() and is_string_array(
                values,  # type: ignore[arg-type]
                skipna=True,
            ):
                df = df.astype("string[pyarrow_numpy]")

            # 将数据框添加到 frames 列表中
            frames.append(df)

        # 如果 frames 中只有一个数据框，则直接返回该数据框；否则进行拼接
        if len(frames) == 1:
            df = frames[0]
        else:
            df = concat(frames, axis=1)

        # 创建 Selection 对象，应用选择过滤器和轴排序
        selection = Selection(self, where=where, start=start, stop=stop)
        df = self.process_axes(df, selection=selection, columns=columns)
        # 应用选择过滤器和轴排序，返回最终的数据框
        return df
class AppendableSeriesTable(AppendableFrameTable):
    """support the new appendable table formats"""

    pandas_kind = "series_table"  # 指定 Pandas 中表的类型为 series_table
    table_type = "appendable_series"  # 指定表的类型为 appendable_series
    ndim = 2  # 表的维度为 2
    obj_type = Series  # 表的对象类型为 Pandas 的 Series

    @property
    def is_transposed(self) -> bool:
        """返回是否为转置表格，这里始终返回 False"""
        return False

    @classmethod
    def get_object(cls, obj, transposed: bool):
        """返回传入的对象，不做任何修改"""
        return obj

    # error: Signature of "write" incompatible with supertype "Fixed"
    def write(self, obj, data_columns=None, **kwargs) -> None:  # type: ignore[override]
        """将数据写入为框架表格格式"""
        if not isinstance(obj, DataFrame):
            name = obj.name or "values"
            obj = obj.to_frame(name)
        super().write(obj=obj, data_columns=obj.columns.tolist(), **kwargs)

    def read(
        self,
        where=None,
        columns=None,
        start: int | None = None,
        stop: int | None = None,
    ) -> Series:
        """从表中读取数据作为 Series"""
        is_multi_index = self.is_multi_index
        if columns is not None and is_multi_index:
            assert isinstance(self.levels, list)  # needed for mypy
            for n in self.levels:
                if n not in columns:
                    columns.insert(0, n)
        s = super().read(where=where, columns=columns, start=start, stop=stop)
        if is_multi_index:
            s.set_index(self.levels, inplace=True)

        s = s.iloc[:, 0]

        # remove the default name
        if s.name == "values":
            s.name = None
        return s


class AppendableMultiSeriesTable(AppendableSeriesTable):
    """support the new appendable table formats"""

    pandas_kind = "series_table"  # 指定 Pandas 中表的类型为 series_table
    table_type = "appendable_multiseries"  # 指定表的类型为 appendable_multiseries

    #  error: Signature of "write" incompatible with supertype "Fixed"
    def write(self, obj, **kwargs) -> None:  # type: ignore[override]
        """将数据写入为框架表格格式，支持多系列"""
        name = obj.name or "values"
        newobj, self.levels = self.validate_multiindex(obj)
        assert isinstance(self.levels, list)  # for mypy
        cols = list(self.levels)
        cols.append(name)
        newobj.columns = Index(cols)
        super().write(obj=newobj, **kwargs)


class GenericTable(AppendableFrameTable):
    """a table that read/writes the generic pytables table format"""

    pandas_kind = "frame_table"  # 指定 Pandas 中表的类型为 frame_table
    table_type = "generic_table"  # 指定表的类型为 generic_table
    ndim = 2  # 表的维度为 2
    obj_type = DataFrame  # 表的对象类型为 Pandas 的 DataFrame
    levels: list[Hashable]

    @property
    def pandas_type(self) -> str:
        """返回表格的 Pandas 类型"""
        return self.pandas_kind

    @property
    def storable(self):
        """返回可存储对象"""
        return getattr(self.group, "table", None) or self.group

    def get_attrs(self) -> None:
        """获取表格的属性"""
        self.non_index_axes = []
        self.nan_rep = None
        self.levels = []

        self.index_axes = [a for a in self.indexables if a.is_an_indexable]
        self.values_axes = [a for a in self.indexables if not a.is_an_indexable]
        self.data_columns = [a.name for a in self.values_axes]
    # 使用装饰器 @cache_readonly 将此方法设置为只读缓存方法，避免重复计算
    @cache_readonly
    def indexables(self):
        """从表描述中创建可索引对象列表"""
        # 获取表的描述信息
        d = self.description

        # 从数据库中读取名为 "index" 的元数据
        md = self.read_metadata("index")
        # 如果存在元数据，则设置 meta 为 "category"，否则为 None
        meta = "category" if md is not None else None
        # 创建一个通用的索引列对象
        index_col = GenericIndexCol(
            name="index", axis=0, table=self.table, meta=meta, metadata=md
        )

        # 初始化可索引对象列表，初始只包含 index_col
        _indexables: list[GenericIndexCol | GenericDataIndexableCol] = [index_col]

        # 遍历表描述的名称列表
        for i, n in enumerate(d._v_names):
            # 断言名称是字符串类型
            assert isinstance(n, str)

            # 获取属性对应的原子数据类型
            atom = getattr(d, n)
            # 从数据库中读取名为 n 的元数据
            md = self.read_metadata(n)
            # 如果存在元数据，则设置 meta 为 "category"，否则为 None
            meta = "category" if md is not None else None
            # 创建一个通用的数据可索引列对象
            dc = GenericDataIndexableCol(
                name=n,
                pos=i,
                values=[n],
                typ=atom,
                table=self.table,
                meta=meta,
                metadata=md,
            )
            # 将创建的数据可索引列对象添加到 _indexables 列表中
            _indexables.append(dc)

        # 返回最终的可索引对象列表
        return _indexables

    # 错误: "write" 的签名与父类型 "AppendableTable" 不兼容
    def write(self, **kwargs) -> None:  # type: ignore[override]
        # 抛出未实现的错误，因为无法在通用表上进行写操作
        raise NotImplementedError("cannot write on an generic table")
class AppendableMultiFrameTable(AppendableFrameTable):
    """一个带有多级索引的数据框架"""

    table_type = "appendable_multiframe"
    obj_type = DataFrame
    ndim = 2
    _re_levels = re.compile(r"^level_\d+$")

    @property
    def table_type_short(self) -> str:
        """返回表格类型的简称字符串"""
        return "appendable_multi"

    # error: Signature of "write" incompatible with supertype "Fixed"
    def write(self, obj, data_columns=None, **kwargs) -> None:  # type: ignore[override]
        """将数据写入表格中。

        Parameters
        ----------
        obj : DataFrame
            要写入的数据框架。
        data_columns : list, optional
            数据列名列表，默认为None。
            如果为True，则使用数据框架的列名列表。
        **kwargs : dict
            其他关键字参数传递给父类方法。

        Raises
        ------
        AssertionError
            如果级别不在数据列中，则触发断言错误。
        """
        if data_columns is None:
            data_columns = []
        elif data_columns is True:
            data_columns = obj.columns.tolist()
        obj, self.levels = self.validate_multiindex(obj)
        assert isinstance(self.levels, list)  # for mypy
        for n in self.levels:
            if n not in data_columns:
                data_columns.insert(0, n)
        super().write(obj=obj, data_columns=data_columns, **kwargs)

    def read(
        self,
        where=None,
        columns=None,
        start: int | None = None,
        stop: int | None = None,
    ) -> DataFrame:
        """从表格中读取数据框架。

        Parameters
        ----------
        where : str, optional
            查询条件，默认为None。
        columns : list, optional
            列名列表，默认为None。
        start : int or None, optional
            起始位置，默认为None。
        stop : int or None, optional
            结束位置，默认为None。

        Returns
        -------
        DataFrame
            返回读取的数据框架。
        """
        df = super().read(where=where, columns=columns, start=start, stop=stop)
        df = df.set_index(self.levels)

        # remove names for 'level_%d'
        df.index = df.index.set_names(
            [None if self._re_levels.search(name) else name for name in df.index.names]
        )

        return df


def _reindex_axis(
    obj: DataFrame, axis: AxisInt, labels: Index, other=None
) -> DataFrame:
    """重新索引指定轴上的对象。

    Parameters
    ----------
    obj : DataFrame
        要重新索引的数据框架。
    axis : int
        要操作的轴编号。
    labels : Index
        新的标签索引。
    other : Index or None, optional
        另一个索引对象，默认为None。

    Returns
    -------
    DataFrame
        返回重新索引后的数据框架。
    """
    ax = obj._get_axis(axis)
    labels = ensure_index(labels)

    # try not to reindex even if other is provided
    # if it equals our current index
    if other is not None:
        other = ensure_index(other)
    if (other is None or labels.equals(other)) and labels.equals(ax):
        return obj

    labels = ensure_index(labels.unique())
    if other is not None:
        labels = ensure_index(other.unique()).intersection(labels, sort=False)
    if not labels.equals(ax):
        slicer: list[slice | Index] = [slice(None, None)] * obj.ndim
        slicer[axis] = labels
        obj = obj.loc[tuple(slicer)]
    return obj


# tz to/from coercion


def _get_tz(tz: tzinfo) -> str | tzinfo:
    """获取时区的编码表示。

    Parameters
    ----------
    tz : tzinfo
        时区对象。

    Returns
    -------
    str or tzinfo
        返回编码后的时区表示。
    """
    zone = timezones.get_timezone(tz)
    return zone


def _set_tz(
    values: npt.NDArray[np.int64], tz: str | tzinfo | None, datetime64_dtype: str
) -> DatetimeArray:
    """
    将值强制转换为带有适当时区的DatetimeArray。

    Parameters
    ----------
    values : ndarray[int64]
        要转换的整数数组。
    tz : str, tzinfo, or None
        时区信息，可以是字符串、时区对象或None。
    datetime64_dtype : str
        数据类型字符串，例如 "datetime64[ns]", "datetime64[25s]"。

    Returns
    -------
    DatetimeArray
        返回带有适当时区的DatetimeArray对象。
    """
    assert values.dtype == "i8", values.dtype
    # Argument "tz" to "tz_to_dtype" has incompatible type "str | tzinfo | None";
    # expected "tzinfo"
    unit, _ = np.datetime_data(datetime64_dtype)  # parsing dtype: unit, count
    dtype = tz_to_dtype(tz=tz, unit=unit)  # type: ignore[arg-type]
    dta = DatetimeArray._from_sequence(values, dtype=dtype)
    # 返回变量 dta 的值作为函数的返回结果
    return dta
def _convert_index(name: str, index: Index, encoding: str, errors: str) -> IndexCol:
    assert isinstance(name, str)

    index_name = index.name
    # 调用 _get_data_and_dtype_name 函数，获取转换后的数据和数据类型名称
    # 注意：此处存在类型错误，因为 index 参数的类型不符合预期的类型
    converted, dtype_name = _get_data_and_dtype_name(index)  # type: ignore[arg-type]
    # 将数据类型名称转换为数据类型的种类（如 integer、floating、datetime64 等）
    kind = _dtype_to_kind(dtype_name)
    # 获取数据的原子类型
    atom = DataIndexableCol._get_atom(converted)

    if (
        lib.is_np_dtype(index.dtype, "iu")
        or needs_i8_conversion(index.dtype)
        or is_bool_dtype(index.dtype)
    ):
        # 处理包括 Index、RangeIndex、DatetimeIndex、TimedeltaIndex、PeriodIndex 的情况
        # 其中，"kind" 可能是 "integer"、"integer"、"datetime64"、"timedelta64"、"integer"
        return IndexCol(
            name,
            values=converted,
            kind=kind,
            typ=atom,
            freq=getattr(index, "freq", None),
            tz=getattr(index, "tz", None),
            index_name=index_name,
        )

    if isinstance(index, MultiIndex):
        # 如果是 MultiIndex 类型，则抛出类型错误
        raise TypeError("MultiIndex not supported here!")

    inferred_type = lib.infer_dtype(index, skipna=False)
    # 对于推断出的类型为 "datetime64" 或 "timedelta64" 的情况，
    # 已经在上面的条件分支中处理了 DatetimeIndex 和 TimedeltaIndex 的情况

    values = np.asarray(index)

    if inferred_type == "date":
        # 如果推断类型为 "date"，将日期转换为其序数值（int32 类型）
        converted = np.asarray([v.toordinal() for v in values], dtype=np.int32)
        return IndexCol(
            name, converted, "date", _tables().Time32Col(), index_name=index_name
        )
    elif inferred_type == "string":
        # 如果推断类型为 "string"，转换字符串数组为适当的格式
        converted = _convert_string_array(values, encoding, errors)
        itemsize = converted.dtype.itemsize
        return IndexCol(
            name,
            converted,
            "string",
            _tables().StringCol(itemsize),
            index_name=index_name,
        )

    elif inferred_type in ["integer", "floating"]:
        # 处理整数或浮点数类型
        return IndexCol(
            name, values=converted, kind=kind, typ=atom, index_name=index_name
        )
    else:
        # 对于其他类型，确保 converted 是 ndarray 类型且 dtype 为 object
        assert isinstance(converted, np.ndarray) and converted.dtype == object
        assert kind == "object", kind
        # 获取数据的原子类型为 ObjectAtom
        atom = _tables().ObjectAtom()
        return IndexCol(name, converted, kind, atom, index_name=index_name)


def _unconvert_index(data, kind: str, encoding: str, errors: str) -> np.ndarray | Index:
    index: Index | np.ndarray

    if kind.startswith("datetime64"):
        if kind == "datetime64":
            # 在存储分辨率信息之前创建的 DatetimeIndex
            index = DatetimeIndex(data)
        else:
            index = DatetimeIndex(data.view(kind))
    elif kind == "timedelta64":
        # 创建 TimedeltaIndex
        index = TimedeltaIndex(data)
    elif kind == "date":
        try:
            # 尝试将数据转换为日期对象数组
            index = np.asarray([date.fromordinal(v) for v in data], dtype=object)
        except ValueError:
            # 如果无法从序数值创建日期对象，则尝试从时间戳创建日期对象数组
            index = np.asarray([date.fromtimestamp(v) for v in data], dtype=object)
    elif kind in ("integer", "float", "bool"):
        # 如果索引类型是整数、浮点数或布尔值，则直接转换为 NumPy 数组
        index = np.asarray(data)
    elif kind in ("string"):
        # 如果索引类型是字符串，则调用函数将其转换为字符串数组
        index = _unconvert_string_array(
            data, nan_rep=None, encoding=encoding, errors=errors
        )
    elif kind == "object":
        # 如果索引类型是对象，则直接转换第一个数据元素为 NumPy 数组
        index = np.asarray(data[0])
    else:  # pragma: no cover
        # 如果索引类型未被识别，则引发 ValueError 异常
        raise ValueError(f"unrecognized index type {kind}")
    # 返回转换后的索引数组
    return index
def _maybe_convert_for_string_atom(
    name: str,                               # 函数参数：列名
    bvalues: ArrayLike,                      # 函数参数：包含字符串的数组或类数组
    existing_col,                            # 函数参数：已存在的列对象
    min_itemsize,                            # 函数参数：最小字符串长度
    nan_rep,                                 # 函数参数：NaN 值的替代方案
    encoding,                                # 函数参数：编码方式
    errors,                                  # 函数参数：编码错误处理方式
    columns: list[str],                      # 函数参数：列名列表
):
    if bvalues.dtype != object:              # 如果数组的数据类型不是对象类型，直接返回原数组
        return bvalues

    bvalues = cast(np.ndarray, bvalues)      # 强制转换为 NumPy 数组类型

    dtype_name = bvalues.dtype.name          # 获取数组元素的数据类型名称
    inferred_type = lib.infer_dtype(bvalues, skipna=False)  # 推断数组元素的数据类型，不跳过 NaN 值

    if inferred_type == "date":              # 如果推断类型为日期类型，抛出类型错误
        raise TypeError("[date] is not implemented as a table column")
    if inferred_type == "datetime":          # 如果推断类型为日期时间类型，抛出类型错误
        # after GH#8260
        # this only would be hit for a multi-timezone dtype which is an error
        raise TypeError(
            "too many timezones in this block, create separate data columns"
        )

    if not (inferred_type == "string" or dtype_name == "object"):  # 如果推断类型不是字符串或者数据类型名称不是对象
        return bvalues                       # 返回原始数组

    mask = isna(bvalues)                     # 判断数组中的 NaN 值，并生成掩码
    data = bvalues.copy()                    # 复制原始数据数组
    data[mask] = nan_rep                     # 使用指定的 NaN 替代值填充 NaN 位置

    # 查看是否有有效的字符串类型
    inferred_type = lib.infer_dtype(data, skipna=False)  # 重新推断处理后的数据类型，不跳过 NaN 值
    if inferred_type != "string":            # 如果推断类型不是字符串
        # 无法序列化这些数据，因此逐列报告异常

        # 预期行为：
        # 逐列搜索块中的非字符串对象列
        for i in range(data.shape[0]):       # 遍历数据数组的每一行
            col = data[i]                    # 获取当前行的数据列
            inferred_type = lib.infer_dtype(col, skipna=False)  # 推断当前列的数据类型，不跳过 NaN 值
            if inferred_type != "string":    # 如果推断类型不是字符串
                error_column_label = columns[i] if len(columns) > i else f"No.{i}"  # 获取列的标签名或者使用默认标签名
                raise TypeError(
                    f"Cannot serialize the column [{error_column_label}]\n"
                    f"because its data contents are not [string] but "
                    f"[{inferred_type}] object dtype"
                )

    # itemsize 是字符串的最大长度（沿任何维度）
    data_converted = _convert_string_array(data, encoding, errors).reshape(data.shape)  # 转换字符串数组为固定长度的字符串类型
    itemsize = data_converted.itemsize       # 获取转换后数据的元素大小

    # 指定了 min_itemsize 吗？
    if isinstance(min_itemsize, dict):       # 如果 min_itemsize 是字典类型
        min_itemsize = int(min_itemsize.get(name) or min_itemsize.get("values") or 0)  # 获取指定列名的最小长度或者默认最小长度
    itemsize = max(min_itemsize or 0, itemsize)  # 取最大的最小长度和转换后数据的元素大小

    # 检查值是否存在冲突的列
    if existing_col is not None:             # 如果存在已有的列对象
        eci = existing_col.validate_col(itemsize)  # 验证列的大小是否符合要求
        if eci is not None and eci > itemsize:  # 如果验证结果存在且超过了当前的元素大小
            itemsize = eci                   # 更新元素大小为验证结果的大小

    data_converted = data_converted.astype(f"|S{itemsize}", copy=False)  # 将转换后的数据数组类型转换为固定长度的字符串类型
    return data_converted                    # 返回转换后的数据数组
    # 检查数据长度是否非零，如果是，则执行以下操作
    if len(data):
        # 将数据展平，并使用指定的编码和错误处理方式编码为字节串
        data = (
            Series(data.ravel(), copy=False)
            .str.encode(encoding, errors)
            ._values.reshape(data.shape)
        )

    # 创建适当大小的数据类型
    # 确保数据展平后的类型为对象类型
    ensured = ensure_object(data.ravel())
    # 计算对象数组中最长字符串的长度
    itemsize = max(1, libwriters.max_len_string_array(ensured))

    # 将数据转换为指定长度为itemsize的字符串类型的NumPy数组
    data = np.asarray(data, dtype=f"S{itemsize}")
    # 返回处理后的数据
    return data
def _unconvert_string_array(
    data: np.ndarray, nan_rep, encoding: str, errors: str
) -> np.ndarray:
    """
    Inverse of _convert_string_array.

    Parameters
    ----------
    data : np.ndarray[fixed-length-string]
        Input array of fixed-length strings to be decoded.
    nan_rep : the storage repr of NaN
        Representation of NaN values to handle.
    encoding : str
        The character encoding scheme used for decoding strings.
    errors : str
        Specifies how encoding errors are handled.

    Returns
    -------
    np.ndarray[object]
        Decoded data as an array of objects.
    """
    # 获取数据的形状
    shape = data.shape
    # 将输入数组展平为一维，并转换为对象类型的 NumPy 数组
    data = np.asarray(data.ravel(), dtype=object)

    # 如果数据长度大于零
    if len(data):
        # 调用 libwriters.max_len_string_array 函数获取字符串数组中最大长度
        itemsize = libwriters.max_len_string_array(ensure_object(data))
        # 根据最大长度创建 Unicode 字符串类型的 dtype
        dtype = f"U{itemsize}"

        # 如果数据的第一个元素是字节对象，则使用 Series 对象进行解码
        if isinstance(data[0], bytes):
            data = Series(data, copy=False).str.decode(encoding, errors=errors)._values
        else:
            # 否则将数据转换为指定的 dtype，然后再转换为对象类型
            data = data.astype(dtype, copy=False).astype(object, copy=False)

    # 如果 nan_rep 为 None，则将其设为字符串 "nan"
    if nan_rep is None:
        nan_rep = "nan"

    # 调用 libwriters.string_array_replace_from_nan_rep 函数，将数组中的 NaN 替换为 nan_rep
    libwriters.string_array_replace_from_nan_rep(data, nan_rep)
    # 将数据数组重新整形为原始形状并返回
    return data.reshape(shape)


def _maybe_convert(values: np.ndarray, val_kind: str, encoding: str, errors: str):
    """
    Conditional conversion of values based on their kind.

    Parameters
    ----------
    values : np.ndarray
        Input array of values to potentially convert.
    val_kind : str
        Type descriptor indicating the kind of values.
    encoding : str
        The character encoding scheme used for string conversions.
    errors : str
        Specifies how encoding errors are handled.

    Returns
    -------
    np.ndarray
        Converted or unchanged values array.
    """
    # 断言 val_kind 是字符串类型
    assert isinstance(val_kind, str), type(val_kind)
    # 如果需要转换（根据 val_kind 判断）
    if _need_convert(val_kind):
        # 获取转换器函数
        conv = _get_converter(val_kind, encoding, errors)
        # 应用转换器函数并返回转换后的值
        values = conv(values)
    # 返回可能已经转换过的或未变化的值数组
    return values


def _get_converter(kind: str, encoding: str, errors: str):
    """
    Retrieve a converter function based on the kind of data.

    Parameters
    ----------
    kind : str
        Type descriptor indicating the kind of data.
    encoding : str
        The character encoding scheme used for string conversions.
    errors : str
        Specifies how encoding errors are handled.

    Returns
    -------
    callable
        Converter function for the specified kind of data.
    """
    # 根据 kind 的不同返回对应的转换器函数
    if kind == "datetime64":
        return lambda x: np.asarray(x, dtype="M8[ns]")
    elif "datetime64" in kind:
        return lambda x: np.asarray(x, dtype=kind)
    elif kind == "string":
        return lambda x: _unconvert_string_array(
            x, nan_rep=None, encoding=encoding, errors=errors
        )
    else:  # pragma: no cover
        # 如果 kind 不是已知类型，则引发 ValueError 异常
        raise ValueError(f"invalid kind {kind}")


def _need_convert(kind: str) -> bool:
    """
    Check if conversion is needed based on the kind of data.

    Parameters
    ----------
    kind : str
        Type descriptor indicating the kind of data.

    Returns
    -------
    bool
        True if conversion is needed, False otherwise.
    """
    # 如果 kind 是 "datetime64"、"string" 或包含 "datetime64" 的字符串，则需要转换
    if kind in ("datetime64", "string") or "datetime64" in kind:
        return True
    # 否则不需要转换
    return False


def _maybe_adjust_name(name: str, version: Sequence[int]) -> str:
    """
    Adjust the given name if necessary based on the version.

    Parameters
    ----------
    name : str
        Original name to potentially adjust.
    version : Tuple[int, int, int]
        Version tuple indicating the library version.

    Returns
    -------
    str
        Adjusted name based on version.
    """
    # 如果 version 是字符串类型或长度小于 3，则引发 ValueError 异常
    if isinstance(version, str) or len(version) < 3:
        raise ValueError("Version is incorrect, expected sequence of 3 integers.")

    # 如果版本号符合特定条件，则调整名称格式
    if version[0] == 0 and version[1] <= 10 and version[2] == 0:
        m = re.search(r"values_block_(\d+)", name)
        if m:
            grp = m.groups()[0]
            name = f"values_{grp}"
    # 返回调整后的名称
    return name


def _dtype_to_kind(dtype_str: str) -> str:
    """
    Determine the data kind string from the given dtype name.

    Parameters
    ----------
    dtype_str : str
        Data type name to analyze.

    Returns
    -------
    str
        Kind descriptor string corresponding to the dtype.
    """
    # 根据 dtype 名称的开头确定数据类型的 kind
    if dtype_str.startswith(("string", "bytes")):
        kind = "string"
    elif dtype_str.startswith("float"):
        kind = "float"
    elif dtype_str.startswith("complex"):
        kind = "complex"
    elif dtype_str.startswith(("int", "uint")):
        kind = "integer"
    elif dtype_str.startswith("datetime64"):
        kind = dtype_str
    # 返回对应的 kind
    return kind
    # 如果数据类型字符串以 "timedelta" 开头，则设置 kind 为 "timedelta64"
    elif dtype_str.startswith("timedelta"):
        kind = "timedelta64"
    # 如果数据类型字符串以 "bool" 开头，则设置 kind 为 "bool"
    elif dtype_str.startswith("bool"):
        kind = "bool"
    # 如果数据类型字符串以 "category" 开头，则设置 kind 为 "category"
    elif dtype_str.startswith("category"):
        kind = "category"
    # 如果数据类型字符串以 "period" 开头
    elif dtype_str.startswith("period"):
        # 存储 `freq` 属性以便从整数恢复
        kind = "integer"
    # 如果数据类型字符串为 "object"，则设置 kind 为 "object"
    elif dtype_str == "object":
        kind = "object"
    # 如果以上条件都不符合，则抛出值错误异常，显示无法解释的数据类型
    else:
        raise ValueError(f"cannot interpret dtype of [{dtype_str}]")

    # 返回确定的数据类型 kind
    return kind
def _get_data_and_dtype_name(data: ArrayLike):
    """
    Convert the passed data into a storable form and a dtype string.
    """
    # 如果数据是 Categorical 类型，则将其转换为其编码（codes）
    if isinstance(data, Categorical):
        data = data.codes

    # 检查数据类型是否为 DatetimeTZDtype 类型
    if isinstance(data.dtype, DatetimeTZDtype):
        # 对于 datetime64tz 类型，需要在测试中去掉时区信息 TODO: 为什么要这样做？
        dtype_name = f"datetime64[{data.dtype.unit}]"
    else:
        dtype_name = data.dtype.name

    # 如果数据类型的种类是 'm' 或 'M'，将数据视为 int64 类型处理
    if data.dtype.kind in "mM":
        data = np.asarray(data.view("i8"))
        # TODO: 以前我们为 dt64tz 案例重塑数据，但不再这样做似乎也没问题。为什么？

    # 如果数据是 PeriodIndex 类型，则将其转换为 int64 类型
    elif isinstance(data, PeriodIndex):
        data = data.asi8

    # 确保数据是 numpy 数组
    data = np.asarray(data)
    return data, dtype_name


class Selection:
    """
    Carries out a selection operation on a tables.Table object.

    Parameters
    ----------
    table : a Table object
    where : list of Terms (or convertible to)
    start, stop: indices to start and/or stop selection

    """

    def __init__(
        self,
        table: Table,
        where=None,
        start: int | None = None,
        stop: int | None = None,
    ) -> None:
        # 初始化 Selection 对象，设置属性
        self.table = table
        self.where = where
        self.start = start
        self.stop = stop
        self.condition = None
        self.filter = None
        self.terms = None
        self.coordinates = None

        # 如果 where 是类列表对象
        if is_list_like(where):
            # 查看是否传入了坐标
            with suppress(ValueError):
                # 推断 where 的数据类型，跳过空值检查
                inferred = lib.infer_dtype(where, skipna=False)
                # 如果推断结果为 "integer" 或 "boolean"
                if inferred in ("integer", "boolean"):
                    where = np.asarray(where)
                    # 如果 where 是布尔类型的数组
                    if where.dtype == np.bool_:
                        start, stop = self.start, self.stop
                        if start is None:
                            start = 0
                        if stop is None:
                            stop = self.table.nrows
                        # 根据布尔条件选择坐标
                        self.coordinates = np.arange(start, stop)[where]
                    # 如果 where 是整数类型的数组
                    elif issubclass(where.dtype.type, np.integer):
                        # 检查条件是否符合要求
                        if (self.start is not None and (where < self.start).any()) or (
                            self.stop is not None and (where >= self.stop).any()
                        ):
                            raise ValueError(
                                "where must have index locations >= start and < stop"
                            )
                        # 设置坐标为 where
                        self.coordinates = where

        # 如果没有明确的坐标传入，生成查询条件
        if self.coordinates is None:
            self.terms = self.generate(where)

            # 创建 numexpr 表达式和过滤器
            if self.terms is not None:
                self.condition, self.filter = self.terms.evaluate()

    @overload
    def generate(self, where: dict | list | tuple | str) -> PyTablesExpr: ...
    @overload
    def generate(self, where: None) -> None: ...
    def generate(self, where: dict | list | tuple | str | None) -> PyTablesExpr | None:
        """where can be a : dict,list,tuple,string"""
        # 检查 where 参数是否为 None
        if where is None:
            # 如果是 None，则返回 None
            return None

        # 获取查询条件的可用项
        q = self.table.queryables()
        try:
            # 尝试创建 PyTablesExpr 对象，使用传入的 where 条件和查询项
            return PyTablesExpr(where, queryables=q, encoding=self.table.encoding)
        except NameError as err:
            # 如果出现 NameError，生成详细的错误消息
            qkeys = ",".join(q.keys())
            msg = dedent(
                f"""\
                The passed where expression: {where}
                            contains an invalid variable reference
                            all of the variable references must be a reference to
                            an axis (e.g. 'index' or 'columns'), or a data_column
                            The currently defined references are: {qkeys}
                """
            )
            # 抛出 ValueError 异常，附带原始的 NameError
            raise ValueError(msg) from err

    def select(self):
        """
        generate the selection
        """
        # 如果有条件 self.condition，则执行读取操作
        if self.condition is not None:
            # 使用表格对象的 read_where 方法根据条件读取数据
            return self.table.table.read_where(
                self.condition.format(), start=self.start, stop=self.stop
            )
        # 如果有坐标 self.coordinates，则执行坐标读取操作
        elif self.coordinates is not None:
            return self.table.table.read_coordinates(self.coordinates)
        # 否则，根据指定的起始和结束位置读取数据
        return self.table.table.read(start=self.start, stop=self.stop)

    def select_coords(self):
        """
        generate the selection
        """
        # 初始化起始和结束位置，以及行数 nrows
        start, stop = self.start, self.stop
        nrows = self.table.nrows
        # 如果 start 为 None，则设置为 0
        if start is None:
            start = 0
        # 如果 start 小于 0，则转换为从末尾向前数的索引
        elif start < 0:
            start += nrows
        # 如果 stop 为 None，则设置为 nrows
        if stop is None:
            stop = nrows
        # 如果 stop 小于 0，则转换为从末尾向前数的索引
        elif stop < 0:
            stop += nrows

        # 如果有条件 self.condition，则执行条件获取列表操作
        if self.condition is not None:
            return self.table.table.get_where_list(
                self.condition.format(), start=start, stop=stop, sort=True
            )
        # 如果有坐标 self.coordinates，则返回坐标数据
        elif self.coordinates is not None:
            return self.coordinates
        # 否则，返回从 start 到 stop 的范围数组
        return np.arange(start, stop)
```