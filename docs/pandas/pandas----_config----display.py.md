# `D:\src\scipysrc\pandas\pandas\_config\display.py`

```
"""
Unopinionated display configuration.
"""

from __future__ import annotations  # 导入未来版本的注解支持

import locale  # 导入locale模块，用于处理本地化相关操作
import sys  # 导入sys模块，用于访问系统相关信息

from pandas._config import config as cf  # 从pandas._config模块中导入config对象，并重命名为cf

# -----------------------------------------------------------------------------
# Global formatting options
_initial_defencoding: str | None = None  # 初始化全局变量_initial_defencoding，用于存储默认编码


def detect_console_encoding() -> str:
    """
    Try to find the most capable encoding supported by the console.
    slightly modified from the way IPython handles the same issue.
    """
    global _initial_defencoding  # 声明全局变量_initial_defencoding

    encoding = None  # 初始化encoding变量为None
    try:
        encoding = sys.stdout.encoding or sys.stdin.encoding  # 尝试获取标准输出和标准输入的编码
    except (AttributeError, OSError):
        pass

    # try again for something better
    if not encoding or "ascii" in encoding.lower():
        try:
            encoding = locale.getpreferredencoding()  # 尝试获取系统首选编码
        except locale.Error:
            # can be raised by locale.setlocale(), which is
            #  called by getpreferredencoding
            #  (on some systems, see stdlib locale docs)
            pass

    # when all else fails. this will usually be "ascii"
    if not encoding or "ascii" in encoding.lower():
        encoding = sys.getdefaultencoding()  # 获取系统默认编码

    # GH#3360, save the reported defencoding at import time
    # MPL backends may change it. Make available for debugging.
    if not _initial_defencoding:
        _initial_defencoding = sys.getdefaultencoding()  # 在导入时保存默认编码以供调试使用

    return encoding  # 返回最终检测到的编码


pc_encoding_doc = """
: str/unicode
    Defaults to the detected encoding of the console.
    Specifies the encoding to be used for strings returned by to_string,
    these are generally strings meant to be displayed on the console.
"""

with cf.config_prefix("display"):
    cf.register_option(
        "encoding", detect_console_encoding(), pc_encoding_doc, validator=cf.is_text
    )
```