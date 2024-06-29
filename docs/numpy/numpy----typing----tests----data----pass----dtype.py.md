# `.\numpy\numpy\typing\tests\data\pass\dtype.py`

```
import numpy as np  # 导入 NumPy 库

dtype_obj = np.dtype(np.str_)  # 创建一个数据类型对象，表示字符串
void_dtype_obj = np.dtype([("f0", np.float64), ("f1", np.float32)])  # 创建一个复合数据类型对象，包含两个字段 f0 和 f1

np.dtype(dtype=np.int64)  # 创建一个整数类型的数据类型对象，指定为 int64
np.dtype(int)  # 创建一个整数类型的数据类型对象，没有指定精度，默认为 int64
np.dtype("int")  # 创建一个整数类型的数据类型对象，同样默认为 int64
np.dtype(None)  # 创建一个空的数据类型对象，未指定类型信息，默认为 void

np.dtype((int, 2))  # 创建一个包含两个整数的数据类型对象，表示一个长度为 2 的数组
np.dtype((int, (1,)))  # 创建一个包含一个整数的数据类型对象，表示一个长度为 1 的数组

np.dtype({"names": ["a", "b"], "formats": [int, float]})  # 创建一个结构化数据类型对象，包含两个字段 a 和 b，分别为 int 和 float 类型
np.dtype({"names": ["a"], "formats": [int], "titles": [object]})  # 创建一个结构化数据类型对象，包含一个字段 a，类型为 int，具有一个标题对象
np.dtype({"names": ["a"], "formats": [int], "titles": [object()]})  # 创建一个结构化数据类型对象，包含一个字段 a，类型为 int，具有一个标题对象

np.dtype([("name", np.str_, 16), ("grades", np.float64, (2,)), ("age", "int32")])  # 创建一个结构化数据类型对象，包含三个字段：name 为固定长度字符串，grades 为长度为 2 的浮点数数组，age 为 int32

np.dtype(
    {
        "names": ["a", "b"],
        "formats": [int, float],
        "itemsize": 9,
        "aligned": False,
        "titles": ["x", "y"],
        "offsets": [0, 1],
    }
)  # 创建一个自定义的结构化数据类型对象，包含两个字段 a 和 b，分别为 int 和 float，设定字节大小为 9，不进行对齐，具有自定义标题和偏移量信息

np.dtype((np.float64, float))  # 创建一个复合数据类型对象，包含两种浮点数类型

class Test:
    dtype = np.dtype(float)

np.dtype(Test())  # 使用类 Test 的数据类型属性创建一个数据类型对象

# Methods and attributes
dtype_obj.base  # 获取数据类型对象的基类
dtype_obj.subdtype  # 获取数据类型对象的子数据类型信息
dtype_obj.newbyteorder()  # 创建一个具有新字节顺序的数据类型对象
dtype_obj.type  # 获取数据类型对象的类型
dtype_obj.name  # 获取数据类型对象的名称
dtype_obj.names  # 获取结构化数据类型对象的字段名称

dtype_obj * 0  # 将数据类型对象与整数相乘
dtype_obj * 2  # 将数据类型对象与整数相乘

0 * dtype_obj  # 将整数与数据类型对象相乘
2 * dtype_obj  # 将整数与数据类型对象相乘

void_dtype_obj["f0"]  # 访问复合数据类型对象的字段 f0
void_dtype_obj[0]  # 通过索引访问复合数据类型对象的第一个字段
void_dtype_obj[["f0", "f1"]]  # 通过列表访问复合数据类型对象的多个字段
void_dtype_obj[["f0"]]  # 通过列表访问复合数据类型对象的单个字段
```