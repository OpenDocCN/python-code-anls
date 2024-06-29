# `D:\src\scipysrc\pandas\pandas\util\_tester.py`

```
"""
Entrypoint for testing from the top-level namespace.
"""

# 从未来模块中导入annotations特性，支持类型注解
from __future__ import annotations

# 导入标准库模块
import os
import sys

# 从pandas.compat._optional导入import_optional_dependency函数
from pandas.compat._optional import import_optional_dependency

# 获取上一级目录路径，作为PKG的值
PKG = os.path.dirname(os.path.dirname(__file__))


def test(extra_args: list[str] | None = None, run_doctests: bool = False) -> None:
    """
    Run the pandas test suite using pytest.

    By default, runs with the marks -m "not slow and not network and not db"

    Parameters
    ----------
    extra_args : list[str], default None
        Extra marks to run the tests.
    run_doctests : bool, default False
        Whether to only run the Python and Cython doctests. If you would like to run
        both doctests/regular tests, just append "--doctest-modules"/"--doctest-cython"
        to extra_args.

    See Also
    --------
    pytest.main : The main entry point for pytest testing framework.

    Examples
    --------
    >>> pd.test()  # doctest: +SKIP
    running: pytest...
    """
    # 导入pytest模块，作为可选依赖
    pytest = import_optional_dependency("pytest")
    # 导入hypothesis模块，作为可选依赖
    import_optional_dependency("hypothesis")
    
    # 默认测试命令，不包含慢速、网络和数据库相关标记
    cmd = ["-m not slow and not network and not db"]
    
    # 如果传入了额外的参数，将其转换为列表格式
    if extra_args:
        if not isinstance(extra_args, list):
            extra_args = [extra_args]
        cmd = extra_args
    
    # 如果设置了run_doctests参数为True，设置命令为运行Python和Cython的doctests
    if run_doctests:
        cmd = [
            "--doctest-modules",
            "--doctest-cython",
            f"--ignore={os.path.join(PKG, 'tests')}",
        ]
    
    # 将PKG路径添加到命令列表末尾
    cmd += [PKG]
    # 将命令列表转换为字符串格式
    joined = " ".join(cmd)
    # 打印运行pytest的命令
    print(f"running: pytest {joined}")
    # 退出程序，使用pytest.main执行测试
    sys.exit(pytest.main(cmd))


# 声明test函数为模块的公共接口
__all__ = ["test"]
```