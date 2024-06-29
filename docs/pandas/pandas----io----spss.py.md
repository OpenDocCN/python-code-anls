# `D:\src\scipysrc\pandas\pandas\io\spss.py`

```
# 从未来版本导入注解类型支持
from __future__ import annotations

# 导入类型检查标志
from typing import TYPE_CHECKING

# 导入 pandas 库的内部函数库
from pandas._libs import lib

# 导入可选的依赖项导入函数
from pandas.compat._optional import import_optional_dependency

# 导入数据类型验证函数
from pandas.util._validators import check_dtype_backend

# 导入判断是否为列表样式的数据类型推断函数
from pandas.core.dtypes.inference import is_list_like

# 导入路径字符串化函数
from pandas.io.common import stringify_path

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 导入序列抽象基类
    from collections.abc import Sequence
    # 导入路径对象类
    from pathlib import Path

    # 导入数据类型后端
    from pandas._typing import DtypeBackend

    # 导入 DataFrame 类
    from pandas import DataFrame


# 定义读取 SPSS 文件的函数
def read_spss(
    path: str | Path,  # 文件路径参数
    usecols: Sequence[str] | None = None,  # 列名序列或 None
    convert_categoricals: bool = True,  # 是否转换分类变量
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,  # 数据类型后端选择
) -> DataFrame:  # 返回 DataFrame 类型
    """
    从文件路径中加载 SPSS 文件，并返回一个 DataFrame。

    Parameters
    ----------
    path : str or Path
        文件路径。
    usecols : list-like, optional
        返回列的子集。如果为 None，则返回所有列。
    convert_categoricals : bool, default is True
        是否将分类列转换为 pd.Categorical。
    dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
        应用于结果 DataFrame 的后端数据类型（仍处于实验阶段）。行为如下：

        * ``"numpy_nullable"``: 返回支持可空 dtype 的 DataFrame（默认）。
        * ``"pyarrow"``: 返回支持 pyarrow 可空 ArrowDtype 的 DataFrame。

        .. versionadded:: 2.0

    Returns
    -------
    DataFrame
        基于 SPSS 文件的 DataFrame。

    See Also
    --------
    read_csv : 从逗号分隔值（csv）文件中读取数据到 pandas DataFrame 中。
    read_excel : 从 Excel 文件中读取数据到 pandas DataFrame 中。
    read_sas : 从 SAS 文件中读取数据到 pandas DataFrame 中。
    read_orc : 将 ORC 对象加载到 pandas DataFrame 中。
    read_feather : 将 feather 格式对象加载到 pandas DataFrame 中。

    Examples
    --------
    >>> df = pd.read_spss("spss_data.sav")  # doctest: +SKIP
    """
    # 导入 pyreadstat 库并赋值给 pyreadstat 变量
    pyreadstat = import_optional_dependency("pyreadstat")
    
    # 检查 dtype_backend 参数的有效性
    check_dtype_backend(dtype_backend)

    # 如果 usecols 不为 None，则检查其是否为列表样式的数据类型
    if usecols is not None:
        if not is_list_like(usecols):
            raise TypeError("usecols must be list-like.")
        usecols = list(usecols)  # pyreadstat 要求一个列表

    # 使用 pyreadstat 库读取 .sav 文件，并获取 DataFrame 和元数据
    df, metadata = pyreadstat.read_sav(
        stringify_path(path), usecols=usecols, apply_value_formats=convert_categoricals
    )
    
    # 将元数据的属性赋值给 DataFrame 的 attrs 属性
    df.attrs = metadata.__dict__
    
    # 如果 dtype_backend 参数不是 lib.no_default，则按照指定的数据类型后端进行转换
    if dtype_backend is not lib.no_default:
        df = df.convert_dtypes(dtype_backend=dtype_backend)
    
    # 返回最终的 DataFrame 对象
    return df
```