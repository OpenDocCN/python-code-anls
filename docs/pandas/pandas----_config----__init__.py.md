# `D:\src\scipysrc\pandas\pandas\_config\__init__.py`

```
"""
pandas._config is considered explicitly upstream of everything else in pandas,
should have no intra-pandas dependencies.

importing `dates` and `display` ensures that keys needed by _libs
are initialized.
"""

# 定义模块的公开接口列表
__all__ = [
    "config",
    "detect_console_encoding",
    "get_option",
    "set_option",
    "reset_option",
    "describe_option",
    "option_context",
    "options",
]

# 从 pandas._config 模块导入 config 对象
from pandas._config import config

# 导入 pandas._config 中的 dates 模块，并忽略 pyright 的未使用导入报告和 noqa 标记
from pandas._config import dates  # pyright: ignore[reportUnusedImport]  # noqa: F401

# 从 pandas._config.config 模块中导入多个函数和对象
from pandas._config.config import (
    _global_config,
    describe_option,
    get_option,
    option_context,
    options,
    reset_option,
    set_option,
)

# 从 pandas._config.display 模块中导入 detect_console_encoding 函数
from pandas._config.display import detect_console_encoding


# 定义函数 using_pyarrow_string_dtype，返回一个布尔值
def using_pyarrow_string_dtype() -> bool:
    # 从 _global_config 字典中获取 "future" 键对应的值，存储在 _mode_options 变量中
    _mode_options = _global_config["future"]
    # 返回 _mode_options 字典中的 "infer_string" 键对应的布尔值
    return _mode_options["infer_string"]
```