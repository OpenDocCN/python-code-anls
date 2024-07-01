# `.\numpy\numpy\typing\tests\data\fail\npyio.pyi`

```py
import pathlib                      # 导入pathlib模块，用于处理路径
from typing import IO               # 导入IO类型，用于类型注解

import numpy.typing as npt          # 导入numpy的类型注解模块
import numpy as np                  # 导入numpy库，用于科学计算

str_path: str                       # 声明str_path为字符串类型
bytes_path: bytes                   # 声明bytes_path为字节类型
pathlib_path: pathlib.Path          # 声明pathlib_path为pathlib.Path类型
str_file: IO[str]                   # 声明str_file为文本IO对象，内容为字符串类型
AR_i8: npt.NDArray[np.int64]        # 声明AR_i8为numpy的int64类型的数组

np.load(str_file)  # E: incompatible type   # 载入文件到numpy数组，但str_file的类型不兼容

np.save(bytes_path, AR_i8)  # E: incompatible type   # 将数组AR_i8保存到字节路径bytes_path，但类型不兼容

np.savez(bytes_path, AR_i8)  # E: incompatible type   # 将数组AR_i8保存为未压缩的字节格式文件，但类型不兼容

np.savez_compressed(bytes_path, AR_i8)  # E: incompatible type   # 将数组AR_i8保存为压缩的字节格式文件，但类型不兼容

np.loadtxt(bytes_path)  # E: incompatible type   # 从字节路径bytes_path加载数据到numpy数组，但类型不兼容

np.fromregex(bytes_path, ".", np.int64)  # E: No overload variant   # 使用正则表达式从字节路径bytes_path中加载数据到int64类型的numpy数组，但没有匹配的重载版本
```