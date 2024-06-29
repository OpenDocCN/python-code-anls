# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\dtype.pyi`

```
import numpy as np  # 导入 numpy 库

class Test1:  # 定义类 Test1
    not_dtype = np.dtype(float)  # 定义类属性 not_dtype，其类型为 numpy 的 float 数据类型

class Test2:  # 定义类 Test2
    dtype = float  # 定义类属性 dtype，其类型为 Python 的 float 类型

np.dtype(Test1())  # 创建 Test1 类的实例，并尝试将其传递给 np.dtype() 函数，此处报错 E: No overload variant of "dtype" matches

np.dtype(Test2())  # 创建 Test2 类的实例，并尝试将其传递给 np.dtype() 函数，此处报错 E: incompatible type

np.dtype(  # 创建一个 numpy 数据类型对象，包含两个字段的描述
    {
        "field1": (float, 1),  # 字段 "field1"，类型为 float，长度为 1
        "field2": (int, 3),    # 字段 "field2"，类型为 int，长度为 3
    }
)  # 此处报错 E: No overload variant of "dtype" matches
```