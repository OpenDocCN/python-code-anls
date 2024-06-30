# `D:\src\scipysrc\scipy\scipy\_lib\_unuran_utils.py`

```
# 导入 pathlib 模块用于处理路径，导入 Union 类型用于指定返回类型
import pathlib
from typing import Union

# 定义一个名为 _unuran_dir 的函数，接受一个布尔型参数 ret_path，返回一个路径对象或字符串
def _unuran_dir(ret_path: bool = False) -> Union[pathlib.Path, str]:
    # 获取当前文件的路径对象，然后找到其父目录，并加上 "unuran" 子目录，形成 unuran 目录的路径
    p = pathlib.Path(__file__).parent / "unuran"
    # 如果 ret_path 参数为 True，则返回路径对象 p；否则返回路径对象 p 的字符串表示
    return p if ret_path else str(p)
```