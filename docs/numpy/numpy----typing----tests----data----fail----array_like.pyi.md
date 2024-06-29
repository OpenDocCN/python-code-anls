# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\array_like.pyi`

```
# 导入 NumPy 库
import numpy as np
# 导入 ArrayLike 类型提示
from numpy._typing import ArrayLike

# 定义空类 A
class A:
    pass

# 创建生成器表达式，生成包含 0 到 9 的整数序列
x1: ArrayLike = (i for i in range(10))  # E: Incompatible types in assignment
# 尝试将类 A 的实例赋值给 x2，类型不匹配
x2: ArrayLike = A()  # E: Incompatible types in assignment
# 尝试将字典赋值给 x3，类型不匹配
x3: ArrayLike = {1: "foo", 2: "bar"}  # E: Incompatible types in assignment

# 创建 NumPy 的 int64 标量
scalar = np.int64(1)
# 调用标量的 __array__ 方法，尝试转换为 float64 类型的 NumPy 数组，但方法不存在对应的重载
scalar.__array__(dtype=np.float64)  # E: No overload variant

# 创建 NumPy 的一维数组
array = np.array([1])
# 调用数组的 __array__ 方法，尝试转换为 float64 类型的 NumPy 数组，但方法不存在对应的重载
array.__array__(dtype=np.float64)  # E: No overload variant
```