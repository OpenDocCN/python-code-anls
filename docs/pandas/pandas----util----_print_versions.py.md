# `D:\src\scipysrc\pandas\pandas\util\_print_versions.py`

```
from __future__ import annotations
# 引入用于类型提示的注解支持

import codecs
# 导入处理编解码的模块
import json
# 导入处理 JSON 数据的模块
import locale
# 导入处理本地化设置的模块
import os
# 导入操作系统相关功能的模块
import platform
# 导入获取平台信息的模块
import struct
# 导入处理 Python 数据结构和 C 结构转换的模块
import sys
# 导入 Python 解释器的相关信息
from typing import TYPE_CHECKING
# 导入用于类型提示的特定类型

if TYPE_CHECKING:
    from pandas._typing import JSONSerializable
    # 如果是类型检查阶段，则导入 JSON 可序列化类型的定义

from pandas.compat._optional import (
    VERSIONS,
    get_version,
    import_optional_dependency,
)
# 导入 pandas 的兼容性函数和依赖管理函数


def _get_commit_hash() -> str | None:
    """
    Use vendored versioneer code to get git hash, which handles
    git worktree correctly.
    """
    try:
        from pandas._version_meson import (  # pyright: ignore [reportMissingImports]
            __git_version__,
        )
        # 尝试从 meson 版本的 pandas._version_meson 模块导入 __git_version__ 函数

        return __git_version__
        # 返回获取的 git 版本号
    except ImportError:
        from pandas._version import get_versions

        versions = get_versions()  # type: ignore[no-untyped-call]
        # 引入 pandas._version 模块中的 get_versions 函数，忽略类型检查
        return versions["full-revisionid"]
        # 返回完整的修订版本标识符


def _get_sys_info() -> dict[str, JSONSerializable]:
    """
    Returns system information as a JSON serializable dictionary.
    """
    uname_result = platform.uname()
    # 获取当前平台的信息
    language_code, encoding = locale.getlocale()
    # 获取当前系统的语言设置和编码方式
    return {
        "commit": _get_commit_hash(),
        # 返回获取的 git 提交版本号
        "python": ".".join([str(i) for i in sys.version_info]),
        # 返回 Python 版本号的字符串表示
        "python-bits": struct.calcsize("P") * 8,
        # 返回 Python 解释器的位数
        "OS": uname_result.system,
        # 返回操作系统名称
        "OS-release": uname_result.release,
        # 返回操作系统版本号
        "Version": uname_result.version,
        # 返回操作系统的详细版本信息
        "machine": uname_result.machine,
        # 返回操作系统的硬件架构
        "processor": uname_result.processor,
        # 返回操作系统的处理器信息
        "byteorder": sys.byteorder,
        # 返回系统的字节顺序
        "LC_ALL": os.environ.get("LC_ALL"),
        # 返回 LC_ALL 环境变量的设置值
        "LANG": os.environ.get("LANG"),
        # 返回 LANG 环境变量的设置值
        "LOCALE": {"language-code": language_code, "encoding": encoding},
        # 返回系统当前的语言设置和编码方式
    }


def _get_dependency_info() -> dict[str, JSONSerializable]:
    """
    Returns dependency information as a JSON serializable dictionary.
    """
    deps = [
        "pandas",
        # pandas 库的依赖信息
        # required
        "numpy",
        # numpy 库的依赖信息
        "pytz",
        # pytz 库的依赖信息
        "dateutil",
        # dateutil 库的依赖信息
        # install / build,
        "setuptools",
        # setuptools 库的依赖信息
        "pip",
        # pip 工具的依赖信息
        "Cython",
        # Cython 编译器的依赖信息
        # test
        "pytest",
        # pytest 测试框架的依赖信息
        "hypothesis",
        # hypothesis 测试框架的依赖信息
        # docs
        "sphinx",
        # sphinx 文档生成工具的依赖信息
        # Other, need a min version
        "blosc",
        # blosc 压缩库的依赖信息
        "feather",
        # feather 文件格式库的依赖信息
        "xlsxwriter",
        # xlsxwriter Excel 文件写入库的依赖信息
        "lxml.etree",
        # lxml.etree XML 处理库的依赖信息
        "html5lib",
        # html5lib HTML 处理库的依赖信息
        "pymysql",
        # pymysql 数据库库的依赖信息
        "psycopg2",
        # psycopg2 PostgreSQL 数据库适配器的依赖信息
        "jinja2",
        # jinja2 模板引擎的依赖信息
        # Other, not imported.
        "IPython",
        # IPython 交互式 Python 环境的依赖信息
        "pandas_datareader",
        # pandas_datareader 数据读取库的依赖信息
    ]
    deps.extend(list(VERSIONS))
    # 将 pandas 的版本信息扩展到依赖列表中

    result: dict[str, JSONSerializable] = {}
    # 初始化结果字典

    for modname in deps:
        mod = import_optional_dependency(modname, errors="ignore")
        # 尝试导入每个依赖项，如果导入失败则忽略错误
        result[modname] = get_version(mod) if mod else None
        # 获取每个模块的版本信息并存入结果字典

    return result
    # 返回依赖信息的字典


def show_versions(as_json: str | bool = False) -> None:
    """
    Provide useful information, important for bug reports.

    It comprises info about hosting operation system, pandas version,
    and versions of other installed relative packages.

    Parameters
    ----------
    as_json : str | bool, default False
        If True or 'json', return result as JSON string. If 'split', also separate pandas and dependencies info.

    """
    as_json : str or bool, default False
        * 如果为 False，则以人类可读形式将信息输出到控制台。
        * 如果为 str，则被视为文件路径，信息将以 JSON 格式写入到该文件。
        * 如果为 True，则以 JSON 格式将信息输出到控制台。

    See Also
    --------
    get_option : 获取指定选项的值。
    set_option : 设置指定选项或多个选项的值。

    Examples
    --------
    >>> pd.show_versions()  # doctest: +SKIP
    输出结果类似于以下内容：
    INSTALLED VERSIONS
    ------------------
    commit           : 37ea63d540fd27274cad6585082c91b1283f963d
    python           : 3.10.6.final.0
    python-bits      : 64
    OS               : Linux
    OS-release       : 5.10.102.1-microsoft-standard-WSL2
    Version          : #1 SMP Wed Mar 2 00:30:59 UTC 2022
    machine          : x86_64
    processor        : x86_64
    byteorder        : little
    LC_ALL           : None
    LANG             : en_GB.UTF-8
    LOCALE           : en_GB.UTF-8
    pandas           : 2.0.1
    numpy            : 1.24.3
    ...

    """
    # 获取系统信息和依赖信息
    sys_info = _get_sys_info()
    deps = _get_dependency_info()

    # 如果 as_json 不为 False，则处理输出 JSON 格式的逻辑
    if as_json:
        j = {"system": sys_info, "dependencies": deps}

        # 如果 as_json 为 True，则将 JSON 格式的信息输出到控制台
        if as_json is True:
            sys.stdout.writelines(json.dumps(j, indent=2))
        # 如果 as_json 是 str，则将 JSON 格式的信息写入到指定路径的文件中
        else:
            assert isinstance(as_json, str)  # 针对 mypy 的必要断言
            with codecs.open(as_json, "wb", encoding="utf8") as f:
                json.dump(j, f, indent=2)

    # 如果 as_json 为 False，则输出人类可读形式的信息到控制台
    else:
        assert isinstance(sys_info["LOCALE"], dict)  # 针对 mypy 的必要断言
        language_code = sys_info["LOCALE"]["language-code"]
        encoding = sys_info["LOCALE"]["encoding"]
        sys_info["LOCALE"] = f"{language_code}.{encoding}"

        maxlen = max(len(x) for x in deps)
        print("\nINSTALLED VERSIONS")
        print("------------------")
        # 输出系统信息
        for k, v in sys_info.items():
            print(f"{k:<{maxlen}}: {v}")
        print("")
        # 输出依赖信息
        for k, v in deps.items():
            print(f"{k:<{maxlen}}: {v}")
```