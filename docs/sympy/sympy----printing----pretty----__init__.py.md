# `D:\src\scipysrc\sympy\sympy\printing\pretty\__init__.py`

```
"""ASCII-ART 2D pretty-printer"""

# 从当前目录导入以下函数：pretty, pretty_print, pprint, pprint_use_unicode, pprint_try_use_unicode, pager_print
from .pretty import (pretty, pretty_print, pprint, pprint_use_unicode,
    pprint_try_use_unicode, pager_print)

# 尝试使用Unicode输出，如果可用的话
pprint_try_use_unicode()

# 定义一个列表，包含了模块中公开的函数名
__all__ = [
    'pretty', 'pretty_print', 'pprint', 'pprint_use_unicode',
    'pprint_try_use_unicode', 'pager_print',
]
```