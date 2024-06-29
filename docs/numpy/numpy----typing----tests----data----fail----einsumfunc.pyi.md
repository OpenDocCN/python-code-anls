# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\einsumfunc.pyi`

```py
import numpy as np  # 导入 NumPy 库，用于科学计算
import numpy.typing as npt  # 导入 NumPy 类型注解模块，用于类型提示

AR_i: npt.NDArray[np.int64]  # 定义 AR_i 为 np.int64 类型的 NumPy 数组
AR_f: npt.NDArray[np.float64]  # 定义 AR_f 为 np.float64 类型的 NumPy 数组
AR_m: npt.NDArray[np.timedelta64]  # 定义 AR_m 为 np.timedelta64 类型的 NumPy 数组
AR_U: npt.NDArray[np.str_]  # 定义 AR_U 为 np.str_ 类型的 NumPy 数组

np.einsum("i,i->i", AR_i, AR_m)  # E: 不兼容的类型。执行 Einstein 求和，但 AR_m 的类型不兼容。
np.einsum("i,i->i", AR_f, AR_f, dtype=np.int32)  # E: 不兼容的类型。执行 Einstein 求和，但 AR_f 的类型不兼容。
np.einsum("i,i->i", AR_i, AR_i, out=AR_U)  # E: "einsum" 函数的类型变量 "_ArrayType" 的值无法确定。
np.einsum("i,i->i", AR_i, AR_i, out=AR_U, casting="unsafe")  # E: 没有匹配的重载变体。执行 Einstein 求和，并指定了不安全的类型转换。
```