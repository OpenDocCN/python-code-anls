# `D:\src\scipysrc\scikit-learn\sklearn\utils\_show_versions.py`

```
"""
Utility methods to print system info for debugging

adapted from :func:`pandas.show_versions`
"""

# SPDX-License-Identifier: BSD-3-Clause

import platform  # 导入 platform 模块，用于获取系统平台信息
import sys  # 导入 sys 模块，用于获取 Python 解释器相关信息

from threadpoolctl import threadpool_info  # 导入 threadpool_info 函数来获取线程池控制信息

from .. import __version__  # 导入当前包的版本信息
from ._openmp_helpers import _openmp_parallelism_enabled  # 导入 OpenMP 并行支持的帮助函数


def _get_sys_info():
    """System information

    Returns
    -------
    sys_info : dict
        system and Python version information

    """
    python = sys.version.replace("\n", " ")  # 获取 Python 解释器的版本信息并格式化为字符串

    blob = [
        ("python", python),
        ("executable", sys.executable),  # 获取 Python 解释器的可执行文件路径
        ("machine", platform.platform()),  # 获取当前平台的详细信息
    ]

    return dict(blob)  # 返回包含系统和 Python 版本信息的字典


def _get_deps_info():
    """Overview of the installed version of main dependencies

    This function does not import the modules to collect the version numbers
    but instead relies on standard Python package metadata.

    Returns
    -------
    deps_info: dict
        version information on relevant Python libraries

    """
    deps = [
        "pip",
        "setuptools",
        "numpy",
        "scipy",
        "Cython",
        "pandas",
        "matplotlib",
        "joblib",
        "threadpoolctl",
    ]

    deps_info = {
        "sklearn": __version__,  # 将当前包的版本信息加入依赖信息字典
    }

    from importlib.metadata import PackageNotFoundError, version  # 导入版本信息获取相关模块

    for modname in deps:
        try:
            deps_info[modname] = version(modname)  # 尝试获取每个依赖库的版本信息
        except PackageNotFoundError:
            deps_info[modname] = None
    return deps_info  # 返回依赖信息字典


def show_versions():
    """Print useful debugging information"

    .. versionadded:: 0.20

    Examples
    --------
    >>> from sklearn import show_versions
    >>> show_versions()  # doctest: +SKIP
    """

    sys_info = _get_sys_info()  # 获取系统信息
    deps_info = _get_deps_info()  # 获取依赖库信息

    print("\nSystem:")  # 打印系统信息部分的标题
    for k, stat in sys_info.items():
        print("{k:>10}: {stat}".format(k=k, stat=stat))  # 打印系统信息

    print("\nPython dependencies:")  # 打印依赖库信息部分的标题
    for k, stat in deps_info.items():
        print("{k:>13}: {stat}".format(k=k, stat=stat))  # 打印依赖库信息

    print(
        "\n{k}: {stat}".format(
            k="Built with OpenMP", stat=_openmp_parallelism_enabled()
        )  # 打印是否使用了 OpenMP 并行支持
    )

    # show threadpoolctl results
    threadpool_results = threadpool_info()  # 获取线程池信息
    if threadpool_results:
        print()
        print("threadpoolctl info:")  # 打印线程池控制信息部分的标题

        for i, result in enumerate(threadpool_results):
            for key, val in result.items():
                print(f"{key:>15}: {val}")  # 打印每个线程池的详细信息
            if i != len(threadpool_results) - 1:
                print()
```