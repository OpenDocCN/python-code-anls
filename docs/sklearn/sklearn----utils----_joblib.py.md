# `D:\src\scipysrc\scikit-learn\sklearn\utils\_joblib.py`

```
# TODO(1.7): remove this file
# 引入警告模块作为 _warnings 的别名
import warnings as _warnings

# 使用 catch_warnings() 方法捕获警告，对警告进行简单过滤以忽略它们
with _warnings.catch_warnings():
    _warnings.simplefilter("ignore")
    # joblib 可能在特定 Python 版本上引发 DeprecationWarning 警告
    # 导入 joblib 库及其部分模块
    import joblib
    from joblib import (
        Memory,
        Parallel,
        __version__,
        cpu_count,
        delayed,
        dump,
        effective_n_jobs,
        hash,
        load,
        logger,
        parallel_backend,
        register_parallel_backend,
    )

# 定义 __all__ 列表，指定了模块中对外公开的对象名称
__all__ = [
    "parallel_backend",
    "register_parallel_backend",
    "cpu_count",
    "Parallel",
    "Memory",
    "delayed",
    "effective_n_jobs",
    "hash",
    "logger",
    "dump",
    "load",
    "joblib",
    "__version__",
]
```