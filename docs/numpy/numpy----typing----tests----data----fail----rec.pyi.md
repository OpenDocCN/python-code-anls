# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\rec.pyi`

```py
import numpy as np  # 导入 NumPy 库

AR_i8: npt.NDArray[np.int64]  # 声明一个类型为 int64 的 NumPy 数组类型标注

np.rec.fromarrays(1)  # E: No overload variant
np.rec.fromarrays([1, 2, 3], dtype=[("f8", "f8")], formats=["f8", "f8"])  # E: No overload variant
# 使用 np.rec.fromarrays() 函数尝试创建结构化数组，但提供的参数没有合适的重载变体

np.rec.fromrecords(AR_i8)  # E: incompatible type
np.rec.fromrecords([(1.5,)], dtype=[("f8", "f8")], formats=["f8", "f8"])  # E: No overload variant
# 使用 np.rec.fromrecords() 函数尝试创建结构化数组，但提供的参数类型不兼容或没有合适的重载变体

np.rec.fromstring("string", dtype=[("f8", "f8")])  # E: No overload variant
np.rec.fromstring(b"bytes")  # E: No overload variant
np.rec.fromstring(b"(1.5,)", dtype=[("f8", "f8")], formats=["f8", "f8"])  # E: No overload variant
# 使用 np.rec.fromstring() 函数尝试从字符串或字节串创建结构化数组，但提供的参数没有合适的重载变体

with open("test", "r") as f:
    np.rec.fromfile(f, dtype=[("f8", "f8")])  # E: No overload variant
# 使用 np.rec.fromfile() 函数尝试从文件中读取数据创建结构化数组，但提供的参数没有合适的重载变体
```