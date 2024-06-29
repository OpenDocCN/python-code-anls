# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\comparisons.pyi`

```
# 导入 NumPy 库，并引入 NumPy 类型注解模块
import numpy as np
import numpy.typing as npt

# 定义不同类型的 NumPy 数组变量，分别注解为特定的数据类型
AR_i: npt.NDArray[np.int64]  # 整型数组，元素为 np.int64 类型
AR_f: npt.NDArray[np.float64]  # 浮点型数组，元素为 np.float64 类型
AR_c: npt.NDArray[np.complex128]  # 复数型数组，元素为 np.complex128 类型
AR_m: npt.NDArray[np.timedelta64]  # 时间间隔数组，元素为 np.timedelta64 类型
AR_M: npt.NDArray[np.datetime64]  # 日期时间数组，元素为 np.datetime64 类型

# 下面的表达式均会引发错误，因为这些 NumPy 数组类型之间的比较不受支持

AR_f > AR_m  # E: Unsupported operand types
AR_c > AR_m  # E: Unsupported operand types

AR_m > AR_f  # E: Unsupported operand types
AR_m > AR_c  # E: Unsupported operand types

AR_i > AR_M  # E: Unsupported operand types
AR_f > AR_M  # E: Unsupported operand types
AR_m > AR_M  # E: Unsupported operand types

AR_M > AR_i  # E: Unsupported operand types
AR_M > AR_f  # E: Unsupported operand types
AR_M > AR_m  # E: Unsupported operand types

# 下面的表达式均会引发错误，因为无法将整数或字节串与这些 NumPy 数组类型之一进行比较

AR_i > str()  # E: No overload variant
AR_i > bytes()  # E: No overload variant
str() > AR_M  # E: Unsupported operand types
bytes() > AR_M  # E: Unsupported operand types
```