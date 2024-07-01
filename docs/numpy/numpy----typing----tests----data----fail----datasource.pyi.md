# `.\numpy\numpy\typing\tests\data\fail\datasource.pyi`

```py
# 导入 Path 类从 pathlib 模块，导入 numpy 模块中的 np 对象
from pathlib import Path
import numpy as np

# 声明变量 path，类型为 Path
path: Path
# 声明变量 d1，类型为 numpy 中的 DataSource 类
d1: np.lib.npyio.DataSource

# 调用 d1 对象的 abspath 方法，传入 path 参数，返回类型不兼容的错误
d1.abspath(path)  # E: incompatible type
# 再次调用 d1 对象的 abspath 方法，传入字节字符串参数，返回类型不兼容的错误
d1.abspath(b"...")  # E: incompatible type

# 调用 d1 对象的 exists 方法，传入 path 参数，返回类型不兼容的错误
d1.exists(path)  # E: incompatible type
# 再次调用 d1 对象的 exists 方法，传入字节字符串参数，返回类型不兼容的错误
d1.exists(b"...")  # E: incompatible type

# 调用 d1 对象的 open 方法，传入 path 和 "r" 参数，返回类型不兼容的错误
d1.open(path, "r")  # E: incompatible type
# 再次调用 d1 对象的 open 方法，传入字节字符串参数和 encoding 参数，返回类型不兼容的错误
d1.open(b"...", encoding="utf8")  # E: incompatible type
# 再次调用 d1 对象的 open 方法，传入 None 和 newline 参数，返回类型不兼容的错误
d1.open(None, newline="/n")  # E: incompatible type
```