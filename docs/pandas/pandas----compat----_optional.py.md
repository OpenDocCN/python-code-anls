# `D:\src\scipysrc\pandas\pandas\compat\_optional.py`

```
# 从未来版本中导入注解，使得在代码中可以使用类型注解
from __future__ import annotations

# 导入用于动态导入模块的标准库
import importlib
# 导入系统相关的功能
import sys
# 导入用于类型注解的标准库，包括Literal和overload
from typing import (
    TYPE_CHECKING,
    Literal,
    overload,
)
# 导入警告相关的功能，用于处理警告信息
import warnings

# 从pandas.util._exceptions模块中导入find_stack_level函数
from pandas.util._exceptions import find_stack_level

# 从pandas.util.version模块中导入Version类
from pandas.util.version import Version

# 如果正在进行类型检查，则导入types模块
if TYPE_CHECKING:
    import types

# 版本信息的字典，键为包名，值为版本号
VERSIONS = {
    "adbc-driver-postgresql": "0.10.0",
    "adbc-driver-sqlite": "0.8.0",
    "bs4": "4.11.2",
    "blosc": "1.21.3",
    "bottleneck": "1.3.6",
    "fastparquet": "2023.10.0",
    "fsspec": "2022.11.0",
    "html5lib": "1.1",
    "hypothesis": "6.46.1",
    "gcsfs": "2022.11.0",
    "jinja2": "3.1.2",
    "lxml.etree": "4.9.2",
    "matplotlib": "3.6.3",
    "numba": "0.56.4",
    "numexpr": "2.8.4",
    "odfpy": "1.4.1",
    "openpyxl": "3.1.0",
    "psycopg2": "2.9.6",  # (dt dec pq3 ext lo64)
    "pymysql": "1.0.2",
    "pyarrow": "10.0.1",
    "pyreadstat": "1.2.0",
    "pytest": "7.3.2",
    "python-calamine": "0.1.7",
    "pyxlsb": "1.0.10",
    "s3fs": "2022.11.0",
    "scipy": "1.10.0",
    "sqlalchemy": "2.0.0",
    "tables": "3.8.0",
    "tabulate": "0.9.0",
    "xarray": "2022.12.0",
    "xlrd": "2.0.1",
    "xlsxwriter": "3.0.5",
    "zstandard": "0.19.0",
    "tzdata": "2022.7",
    "qtpy": "2.3.0",
    "pyqt5": "5.15.9",
}

# 从导入名称到PyPI包名的映射字典
INSTALL_MAPPING = {
    "bs4": "beautifulsoup4",
    "bottleneck": "Bottleneck",
    "jinja2": "Jinja2",
    "lxml.etree": "lxml",
    "odf": "odfpy",
    "python_calamine": "python-calamine",
    "sqlalchemy": "SQLAlchemy",
    "tables": "pytables",
}

# 获取给定模块的版本信息
def get_version(module: types.ModuleType) -> str:
    version = getattr(module, "__version__", None)

    # 如果无法确定模块的版本，则抛出ImportError异常
    if version is None:
        raise ImportError(f"Can't determine version for {module.__name__}")
    
    # 如果模块是psycopg2，则仅获取版本号的第一个部分
    if module.__name__ == "psycopg2":
        # psycopg2的版本信息会附加 " (dt dec pq3 ext lo64)"，只取第一个部分
        version = version.split()[0]
    
    return version

# 用装饰器标记的重载函数，导入可选依赖模块
@overload
def import_optional_dependency(
    name: str,
    extra: str = ...,
    min_version: str | None = ...,
    *,
    errors: Literal["raise"] = ...,
) -> types.ModuleType: ...

@overload
def import_optional_dependency(
    name: str,
    extra: str = ...,
    min_version: str | None = ...,
    *,
    errors: Literal["warn", "ignore"],
) -> types.ModuleType | None: ...

# 实际的导入可选依赖模块函数定义
def import_optional_dependency(
    name: str,
    extra: str = "",
    min_version: str | None = None,
    *,
    errors: Literal["raise", "warn", "ignore"] = "raise",
) -> types.ModuleType | None:
    """
    导入可选依赖模块。

    默认情况下，如果依赖模块缺失，将会引发ImportError并附带友好的错误信息。如果依赖模块存在但版本过低，则同样会引发错误。

    Parameters
    ----------
    name : str
        模块名。
    extra : str
        # 附加文本，用于在 ImportError 消息中包含额外信息。
    errors : str {'raise', 'warn', 'ignore'}
        # 当依赖项未找到或其版本过旧时的处理方式。

        * raise : 抛出 ImportError
        * warn : 仅在模块版本过旧时适用。警告版本过旧并返回 None。
        * ignore: 如果模块未安装，返回 None；否则，返回模块，即使版本过旧。
          预期用户在使用 ``errors="ignore"`` 时在本地验证版本（见 ``io/html.py``）。
    min_version : str, default None
        # 指定一个与全局 pandas 最低要求版本不同的最小版本。
    Returns
    -------
    maybe_module : Optional[ModuleType]
        # 当找到且版本正确时返回导入的模块。
        # 当未找到包且 `errors` 为 False 时返回 None；
        # 或者当包的版本过旧且 `errors` 为 ``'warn'`` 或 ``'ignore'`` 时返回 None。
    """
    assert errors in {"warn", "raise", "ignore"}

    # 根据 name 获取包名，若找不到则使用 name 本身
    package_name = INSTALL_MAPPING.get(name)
    install_name = package_name if package_name is not None else name

    # 构造 ImportError 的错误消息
    msg = (
        f"Missing optional dependency '{install_name}'. {extra} "
        f"Use pip or conda to install {install_name}."
    )
    try:
        # 尝试导入指定的模块
        module = importlib.import_module(name)
    except ImportError as err:
        # 捕获导入错误，根据 errors 参数处理
        if errors == "raise":
            raise ImportError(msg) from err
        return None

    # 处理子模块：如果有子模块，从 sys.modules 获取父模块
    parent = name.split(".")[0]
    if parent != name:
        install_name = parent
        module_to_get = sys.modules[install_name]
    else:
        module_to_get = module

    # 获取指定模块的最低版本要求
    minimum_version = min_version if min_version is not None else VERSIONS.get(parent)
    if minimum_version:
        # 获取模块的当前版本并进行比较
        version = get_version(module_to_get)
        if version and Version(version) < Version(minimum_version):
            # 构造版本过低的警告或错误消息
            msg = (
                f"Pandas requires version '{minimum_version}' or newer of '{parent}' "
                f"(version '{version}' currently installed)."
            )
            if errors == "warn":
                # 发出版本过低警告并返回 None
                warnings.warn(
                    msg,
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
                return None
            elif errors == "raise":
                # 抛出 ImportError
                raise ImportError(msg)
            else:
                return None

    return module
```