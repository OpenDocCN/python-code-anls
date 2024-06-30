# `D:\src\scipysrc\scipy\scipy\_lib\_highs_utils.py`

```
"""Helper functions to get location of source files."""

import pathlib  # 导入处理路径的标准库


def _highs_dir() -> pathlib.Path:
    """Directory where root highs/ directory lives."""
    # 获取当前文件的路径，并定位其父目录，然后加上 'highs' 子目录构成路径对象
    p = pathlib.Path(__file__).parent / 'highs'
    # 返回构建好的路径对象，表示 'highs' 目录的位置
    return p
```