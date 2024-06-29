# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\type_check.pyi`

```
import numpy as np  # 导入 NumPy 库，使用 np 作为别名

DTYPE_i8: np.dtype[np.int64]  # 定义一个类型别名 DTYPE_i8，表示 np.int64 类型

np.mintypecode(DTYPE_i8)  # 调用 NumPy 函数 mintypecode，传入 DTYPE_i8，返回类型码，但类型不兼容错误
np.iscomplexobj(DTYPE_i8)  # 调用 NumPy 函数 iscomplexobj，传入 DTYPE_i8，检查其是否为复数对象，但类型不兼容错误
np.isrealobj(DTYPE_i8)  # 调用 NumPy 函数 isrealobj，传入 DTYPE_i8，检查其是否为实数对象，但类型不兼容错误

np.typename(DTYPE_i8)  # 调用 NumPy 函数 typename，传入 DTYPE_i8，返回其类型名称，但未找到匹配的重载变体错误
np.typename("invalid")  # 调用 NumPy 函数 typename，传入字符串 "invalid"，但未找到匹配的重载变体错误

np.common_type(np.timedelta64())  # 调用 NumPy 函数 common_type，传入 np.timedelta64 类型，但类型不兼容错误
```