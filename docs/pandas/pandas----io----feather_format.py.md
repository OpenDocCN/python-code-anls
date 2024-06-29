# `D:\src\scipysrc\pandas\pandas\io\feather_format.py`

```
"""feather-format compat"""

# 引入未来的注解支持
from __future__ import annotations

# 引入必要的类型提示
from typing import (
    TYPE_CHECKING,
    Any,
)
# 引入警告模块
import warnings

# 引入 pandas 中的一些模块和函数
from pandas._config import using_pyarrow_string_dtype

# 引入 pandas 库
import pandas as pd

# 引入 pandas 的核心 DataFrame 类
from pandas.core.api import DataFrame

# 引入共享文档字符串
from pandas.core.shared_docs import _shared_docs

# 引入 pandas IO 相关模块
from pandas.io._util import arrow_string_types_mapper
from pandas.io.common import get_handle

# 如果是类型检查状态，引入特定的类型
if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Sequence,
    )

    from pandas._typing import (
        DtypeBackend,
        FilePath,
        ReadBuffer,
        StorageOptions,
        WriteBuffer,
    )

# 使用共享文档定义的 storage_options，作为函数的文档字符串
@doc(storage_options=_shared_docs["storage_options"])
def to_feather(
    df: DataFrame,
    path: FilePath | WriteBuffer[bytes],
    storage_options: StorageOptions | None = None,
    **kwargs: Any,
) -> None:
    """
    Write a DataFrame to the binary Feather format.

    Parameters
    ----------
    df : DataFrame
        要写入的 DataFrame 对象
    path : str, path object, or file-like object
        文件路径或类文件对象，用于写入
    {storage_options}
        存储选项，参考共享文档定义
    **kwargs :
        传递给 `pyarrow.feather.write_feather` 的额外关键字参数

    """
    # 动态导入 pyarrow 依赖
    import_optional_dependency("pyarrow")
    # 从 pyarrow 中导入 feather 模块
    from pyarrow import feather

    # 如果 df 不是 DataFrame 类型，则抛出数值错误异常
    if not isinstance(df, DataFrame):
        raise ValueError("feather only support IO with DataFrames")

    # 获取文件句柄，用于二进制写入，并调用 pyarrow 的 feather.write_feather 方法写入数据
    with get_handle(
        path, "wb", storage_options=storage_options, is_text=False
    ) as handles:
        feather.write_feather(df, handles.handle, **kwargs)


# 使用共享文档定义的 storage_options，作为函数的文档字符串
@doc(storage_options=_shared_docs["storage_options"])
def read_feather(
    path: FilePath | ReadBuffer[bytes],
    columns: Sequence[Hashable] | None = None,
    use_threads: bool = True,
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
) -> DataFrame:
    """
    Load a feather-format object from the file path.

    Parameters
    ----------
    path : str, path object, or file-like object
        文件路径或类文件对象，用于读取
    columns : sequence, default None
        要读取的列的序列，如果未提供，则读取所有列
    use_threads : bool, default True
        是否使用多线程并行读取
    {storage_options}
        存储选项，参考共享文档定义
    dtype_backend : DtypeBackend | lib.NoDefault, default lib.no_default
        数据类型后端

    Returns
    -------
    DataFrame
        从 feather 文件加载的 DataFrame 对象

    """
    # 动态导入 pyarrow 依赖
    import_optional_dependency("pyarrow")
    # 从 pyarrow 中导入 feather 模块
    from pyarrow import feather

    # 调用 pyarrow 的 feather.read_feather 方法读取 feather 格式文件内容并返回 DataFrame 对象
    return feather.read_feather(
        path,
        columns=columns,
        use_threads=use_threads,
        storage_options=storage_options,
        dtype_backend=dtype_backend,
    )
    dtype_backend : {{'numpy_nullable', 'pyarrow'}}, default 'numpy_nullable'
        结果DataFrame的后端数据类型（仍在实验阶段）。行为如下：

        * ``"numpy_nullable"``: 返回支持可空dtype的:class:`DataFrame`（默认）。
        * ``"pyarrow"``: 返回基于pyarrow的可空:class:`ArrowDtype` DataFrame。

        .. versionadded:: 2.0
            版本添加说明: 2.0

    Returns
    -------
    type of object stored in file
        存储在文件中的DataFrame对象。

    See Also
    --------
    read_csv : 从逗号分隔值（csv）文件中读取数据到pandas DataFrame。
    read_excel : 从Excel文件中读取数据到pandas DataFrame。
    read_spss : 从SPSS文件中读取数据到pandas DataFrame。
    read_orc : 将ORC对象加载到pandas DataFrame中。
    read_sas : 将SAS文件读取到pandas DataFrame中。

    Examples
    --------
    >>> df = pd.read_feather("path/to/file.feather")  # doctest: +SKIP
        示例
    """
    import_optional_dependency("pyarrow")
    from pyarrow import feather

    # 导入utils以注册pyarrow扩展类型
    import pandas.core.arrays.arrow.extension_types  # pyright: ignore[reportUnusedImport] # noqa: F401

    check_dtype_backend(dtype_backend)

    with get_handle(
        path, "rb", storage_options=storage_options, is_text=False
    ) as handles:
        if dtype_backend is lib.no_default and not using_pyarrow_string_dtype():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    "make_block is deprecated",
                    DeprecationWarning,
                )

                return feather.read_feather(
                    handles.handle, columns=columns, use_threads=bool(use_threads)
                )

        pa_table = feather.read_table(
            handles.handle, columns=columns, use_threads=bool(use_threads)
        )

        if dtype_backend == "numpy_nullable":
            from pandas.io._util import _arrow_dtype_mapping

            return pa_table.to_pandas(types_mapper=_arrow_dtype_mapping().get)

        elif dtype_backend == "pyarrow":
            return pa_table.to_pandas(types_mapper=pd.ArrowDtype)

        elif using_pyarrow_string_dtype():
            return pa_table.to_pandas(types_mapper=arrow_string_types_mapper())
        else:
            raise NotImplementedError
```