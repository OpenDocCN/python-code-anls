# `D:\src\scipysrc\pandas\pandas\io\orc.py`

```
"""orc compat"""

# 导入必要的模块和函数
from __future__ import annotations  # 支持类型提示中的注解

import io  # 导入io模块，用于处理输入输出
from typing import (  # 引入类型提示的相关类型
    TYPE_CHECKING,  # 类型检查标志
    Any,  # 通用类型
    Literal,  # 字面值类型
)

from pandas._config import using_pyarrow_string_dtype  # 导入配置模块中的特定函数

from pandas._libs import lib  # 导入Pandas底层库中的lib模块
from pandas.compat._optional import import_optional_dependency  # 导入可选依赖项的导入函数
from pandas.util._validators import check_dtype_backend  # 导入数据类型后端验证函数

import pandas as pd  # 导入Pandas库并简称为pd
from pandas.core.indexes.api import default_index  # 从索引API中导入默认索引函数

from pandas.io._util import arrow_string_types_mapper  # 导入字符串类型映射函数
from pandas.io.common import (  # 从通用模块中导入多个函数
    get_handle,  # 获取处理函数
    is_fsspec_url,  # 检查是否为fsspec URL的函数
)

if TYPE_CHECKING:
    import fsspec  # 如果进行类型检查，导入fsspec模块
    import pyarrow.fs  # 如果进行类型检查，导入pyarrow.fs模块

    from pandas._typing import (  # 从Pandas类型提示中导入多个类型
        DtypeBackend,  # 数据类型后端类型
        FilePath,  # 文件路径类型
        ReadBuffer,  # 读取缓冲区类型
        WriteBuffer,  # 写入缓冲区类型
    )

    from pandas.core.frame import DataFrame  # 从DataFrame中导入DataFrame类型

def read_orc(
    path: FilePath | ReadBuffer[bytes],  # 定义函数read_orc，接受文件路径或二进制读取缓冲区作为参数
    columns: list[str] | None = None,  # 列表类型的列名或None，默认为None
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,  # 数据类型后端或无默认值
    filesystem: pyarrow.fs.FileSystem | fsspec.spec.AbstractFileSystem | None = None,  # pyarrow或fsspec文件系统或None，默认为None
    **kwargs: Any,  # 其余关键字参数
) -> DataFrame:  # 返回DataFrame类型
    """
    Load an ORC object from the file path, returning a DataFrame.

    Parameters
    ----------
    path : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``read()`` function. The string could be a URL.
        Valid URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.orc``.
    columns : list, default None
        If not None, only these columns will be read from the file.
        Output always follows the ordering of the file and not the columns list.
        This mirrors the original behaviour of
        :external+pyarrow:py:meth:`pyarrow.orc.ORCFile.read`.
    dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0

    filesystem : fsspec or pyarrow filesystem, default None
        Filesystem object to use when reading the orc file.

        .. versionadded:: 2.1.0

    **kwargs
        Any additional kwargs are passed to pyarrow.

    Returns
    -------
    DataFrame
        DataFrame based on the ORC file.

    See Also
    --------
    read_csv : Read a comma-separated values (csv) file into a pandas DataFrame.
    read_excel : Read an Excel file into a pandas DataFrame.
    read_spss : Read an SPSS file into a pandas DataFrame.
    read_sas : Load a SAS file into a pandas DataFrame.
    read_feather : Load a feather-format object into a pandas DataFrame.

    Notes
    -----
    """
    # 函数文档字符串，描述了如何从文件路径加载ORC对象并返回DataFrame
    pass  # 函数体暂未实现，保留pass关键字，即不执行任何操作
    """
    Before using this function you should read the :ref:`user guide about ORC <io.orc>`
    and :ref:`install optional dependencies <install.warn_orc>`.
    
    If ``path`` is a URI scheme pointing to a local or remote file (e.g. "s3://"),
    a ``pyarrow.fs`` filesystem will be attempted to read the file. You can also pass a
    pyarrow or fsspec filesystem object into the filesystem keyword to override this
    behavior.
    
    Examples
    --------
    >>> result = pd.read_orc("example_pa.orc")  # doctest: +SKIP
    """
    
    # 检查是否需要较新版本的 pyarrow 来支持 ORC 格式
    orc = import_optional_dependency("pyarrow.orc")
    
    # 检查数据类型后端是否符合要求
    check_dtype_backend(dtype_backend)
    
    # 获取文件句柄以读取数据，以二进制形式打开文件
    with get_handle(path, "rb", is_text=False) as handles:
        source = handles.handle
    
        # 如果路径是 fsspec 格式且未指定文件系统，则尝试使用 pyarrow 解析路径
        if is_fsspec_url(path) and filesystem is None:
            pa = import_optional_dependency("pyarrow")
            pa_fs = import_optional_dependency("pyarrow.fs")
            try:
                # 尝试从 URI 创建文件系统对象
                filesystem, source = pa_fs.FileSystem.from_uri(path)
            except (TypeError, pa.ArrowInvalid):
                pass
    
        # 使用 pyarrow 的 orc 模块读取数据表
        pa_table = orc.read_table(
            source=source, columns=columns, filesystem=filesystem, **kwargs
        )
    
    # 根据 dtype_backend 指定的数据类型后端处理数据表
    if dtype_backend is not lib.no_default:
        if dtype_backend == "pyarrow":
            # 如果使用 pyarrow 后端，则转换为 pandas DataFrame
            df = pa_table.to_pandas(types_mapper=pd.ArrowDtype)
        else:
            # 否则，根据 pandas 的类型映射进行转换
            from pandas.io._util import _arrow_dtype_mapping
    
            mapping = _arrow_dtype_mapping()
            df = pa_table.to_pandas(types_mapper=mapping.get)
        return df
    else:
        if using_pyarrow_string_dtype():
            # 如果使用 pyarrow 的字符串类型处理，则使用相应的类型映射器
            types_mapper = arrow_string_types_mapper()
        else:
            types_mapper = None
        # 转换数据表为 pandas DataFrame
        return pa_table.to_pandas(types_mapper=types_mapper)
def to_orc(
    df: DataFrame,
    path: FilePath | WriteBuffer[bytes] | None = None,
    *,
    engine: Literal["pyarrow"] = "pyarrow",
    index: bool | None = None,
    engine_kwargs: dict[str, Any] | None = None,
) -> bytes | None:
    """
    Write a DataFrame to the ORC format.

    .. versionadded:: 1.5.0

    Parameters
    ----------
    df : DataFrame
        要写入ORC格式的数据帧。如果一个或多个列的dtype是category、unsigned integers、intervals、periods或sparse，则会引发NotImplementedError。
    path : str, file-like object or None, default None
        如果是字符串，则在写入分区数据集时将用作根目录路径。如果是带有write()方法的文件类对象（例如通过内置的open函数获得的文件句柄），则会使用它。如果path为None，则返回一个bytes对象。
    engine : str, default 'pyarrow'
        要使用的ORC库。
    index : bool, optional
        如果为True，则将数据帧的索引包含在文件输出中。如果为False，则不会写入文件。如果为None，则类似于推断，数据帧的索引将被保存。但是，RangeIndex将作为元数据中的范围存储，因此不需要太多空间且速度更快。其他索引将作为文件输出中的列包含。
    engine_kwargs : dict[str, Any] or None, default None
        传递给pyarrow.orc.write_table函数的额外关键字参数。

    Returns
    -------
    bytes if no path argument is provided else None

    Raises
    ------
    NotImplementedError
        如果一个或多个列的dtype是category、unsigned integers、interval、period或sparse。
    ValueError
        如果engine不是pyarrow。

    Notes
    -----
    * 在使用此函数之前，您应该阅读关于ORC的用户指南和可选依赖项安装警告。
    * 此函数需要`pyarrow <https://arrow.apache.org/docs/python/>`_库。
    * 有关支持的dtype，请参阅Arrow中支持的ORC功能。
    * 当将数据帧转换为ORC文件时，当前不保留datetime列中的时区信息。
    """

    # 如果index为None，则根据数据帧的索引名称确定是否包含索引
    if index is None:
        index = df.index.names[0] is not None

    # 如果engine_kwargs为None，则设为一个空字典
    if engine_kwargs is None:
        engine_kwargs = {}

    # validate index
    # --------------

    # 验证我们只有一个默认索引
    # 如果有其他索引，则引发异常，因为我们不会序列化索引

    if not df.index.equals(default_index(len(df))):
        raise ValueError(
            "orc does not support serializing a non-default index for the index; "
            "you can .reset_index() to make the index into column(s)"
        )
    # 检查数据框的索引名是否为 None，如果不是则引发数值错误异常
    if df.index.name is not None:
        raise ValueError("orc does not serialize index meta-data on a default index")

    # 检查引擎是否不为 "pyarrow"，如果是则引发数值错误异常
    if engine != "pyarrow":
        raise ValueError("engine must be 'pyarrow'")
    
    # 导入 pyarrow 库作为可选依赖
    pyarrow = import_optional_dependency(engine, min_version="10.0.1")
    
    # 导入 pyarrow 库
    pa = import_optional_dependency("pyarrow")
    
    # 导入 pyarrow.orc 模块
    orc = import_optional_dependency("pyarrow.orc")

    # 判断原始路径是否为 None
    was_none = path is None
    
    # 如果原始路径为 None，则使用 BytesIO 创建一个新的字节流对象
    if was_none:
        path = io.BytesIO()
    
    # 断言路径不为 None，用于类型检查（仅用于类型检查，无实际作用）
    assert path is not None  # For mypy
    
    # 使用 get_handle 函数获取路径的句柄，以二进制写入模式打开，不使用文本模式
    with get_handle(path, "wb", is_text=False) as handles:
        try:
            # 将 pandas 的 DataFrame 转换为 pyarrow.Table，并将其写入到 orc 文件中
            orc.write_table(
                pyarrow.Table.from_pandas(df, preserve_index=index),
                handles.handle,
                **engine_kwargs,
            )
        except (TypeError, pa.ArrowNotImplementedError) as e:
            # 捕获类型错误或者箭头库未实现的错误，引发未实现错误异常
            raise NotImplementedError(
                "The dtype of one or more columns is not supported yet."
            ) from e

    # 如果原始路径为 None，则断言路径类型为 BytesIO，用于类型检查（仅用于类型检查，无实际作用）
    if was_none:
        assert isinstance(path, io.BytesIO)  # For mypy
        # 返回字节流对象的值
        return path.getvalue()
    
    # 原始路径不为 None，则返回 None
    return None
```