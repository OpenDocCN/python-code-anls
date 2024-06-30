# `D:\src\scipysrc\scipy\scipy\_lib\_boost_utils.py`

```
'''Helper functions to get location of header files.'''

# 导入路径操作模块和类型提示模块
import pathlib
from typing import Union

# 定义一个带有参数和返回类型注解的函数，用于获取 Boost 根目录的路径或路径字符串
def _boost_dir(ret_path: bool = False) -> Union[pathlib.Path, str]:
    # 获取当前脚本文件的父目录路径，并拼接上 Boost 头文件所在目录
    p = pathlib.Path(__file__).parent / 'boost_math/include'
    # 如果 ret_path 参数为 True，则返回路径对象 p，否则返回路径字符串表示
    return p if ret_path else str(p)
```