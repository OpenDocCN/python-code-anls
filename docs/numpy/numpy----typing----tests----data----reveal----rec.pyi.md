# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\rec.pyi`

```
import io  # 导入io模块，用于处理文件和流相关的操作
import sys  # 导入sys模块，用于访问系统相关的参数和功能
from typing import Any  # 从typing模块导入Any类型，表示任意类型的对象

import numpy as np  # 导入NumPy库，并用np作为别名
import numpy.typing as npt  # 导入NumPy的类型提示模块

if sys.version_info >= (3, 11):  # 如果Python版本大于等于3.11
    from typing import assert_type  # 导入3.11版本后新增的assert_type函数
else:
    from typing_extensions import assert_type  # 否则从typing_extensions中导入assert_type函数

AR_i8: npt.NDArray[np.int64]  # 定义AR_i8变量，类型为NumPy整型数组
REC_AR_V: np.recarray[Any, np.dtype[np.record]]  # 定义REC_AR_V变量，类型为NumPy记录数组
AR_LIST: list[npt.NDArray[np.int64]]  # 定义AR_LIST变量，类型为包含NumPy整型数组的列表

record: np.record  # 定义record变量，类型为NumPy记录
file_obj: io.BufferedIOBase  # 定义file_obj变量，类型为IO库中的缓冲IO对象

assert_type(np.rec.format_parser(
    formats=[np.float64, np.int64, np.bool],  # 指定格式列表为float64, int64, bool
    names=["f8", "i8", "?"],  # 指定字段名称列表为f8, i8, ?
    titles=None,  # 没有标题
    aligned=True,  # 对齐为True
), np.rec.format_parser)  # 断言返回类型为np.rec.format_parser

assert_type(np.rec.format_parser.dtype, np.dtype[np.void])  # 断言np.rec.format_parser.dtype的类型为np.dtype[np.void]

assert_type(record.field_a, Any)  # 断言record的field_a属性类型为Any
assert_type(record.field_b, Any)  # 断言record的field_b属性类型为Any
assert_type(record["field_a"], Any)  # 断言record的"field_a"字段类型为Any
assert_type(record["field_b"], Any)  # 断言record的"field_b"字段类型为Any
assert_type(record.pprint(), str)  # 断言record的pprint方法返回类型为str
record.field_c = 5  # 给record的field_c属性赋值为5

assert_type(REC_AR_V.field(0), Any)  # 断言REC_AR_V的第一个字段的类型为Any
assert_type(REC_AR_V.field("field_a"), Any)  # 断言REC_AR_V的"field_a"字段的类型为Any
assert_type(REC_AR_V.field(0, AR_i8), None)  # 断言设置REC_AR_V的第一个字段为AR_i8类型后返回None
assert_type(REC_AR_V.field("field_a", AR_i8), None)  # 断言设置REC_AR_V的"field_a"字段为AR_i8类型后返回None
assert_type(REC_AR_V["field_a"], npt.NDArray[Any])  # 断言REC_AR_V的"field_a"字段类型为NumPy任意类型的数组
assert_type(REC_AR_V.field_a, Any)  # 断言REC_AR_V的field_a属性类型为Any
assert_type(REC_AR_V.__array_finalize__(object()), None)  # 断言调用REC_AR_V的__array_finalize__方法返回None

assert_type(
    np.recarray(
        shape=(10, 5),  # 形状为(10, 5)
        formats=[np.float64, np.int64, np.bool],  # 指定格式列表为float64, int64, bool
        order="K",  # 存储顺序为"K"
        byteorder="|",  # 字节顺序为"|"
    ),
    np.recarray[Any, np.dtype[np.record]],  # 断言返回类型为np.recarray[Any, np.dtype[np.record]]
)

assert_type(
    np.recarray(
        shape=(10, 5),  # 形状为(10, 5)
        dtype=[("f8", np.float64), ("i8", np.int64)],  # 指定字段类型列表为f8: float64, i8: int64
        strides=(5, 5),  # 步幅为(5, 5)
    ),
    np.recarray[Any, np.dtype[Any]],  # 断言返回类型为np.recarray[Any, np.dtype[Any]]
)

assert_type(np.rec.fromarrays(AR_LIST), np.recarray[Any, np.dtype[Any]])  # 断言从AR_LIST创建的记录数组类型为np.recarray[Any, np.dtype[Any]]
assert_type(
    np.rec.fromarrays(AR_LIST, dtype=np.int64),  # 指定dtype为np.int64
    np.recarray[Any, np.dtype[Any]],  # 断言返回类型为np.recarray[Any, np.dtype[Any]]
)
assert_type(
    np.rec.fromarrays(
        AR_LIST,
        formats=[np.int64, np.float64],  # 指定格式列表为np.int64, np.float64
        names=["i8", "f8"]  # 指定字段名称列表为i8, f8
    ),
    np.recarray[Any, np.dtype[np.record]],  # 断言返回类型为np.recarray[Any, np.dtype[np.record]]
)

assert_type(
    np.rec.fromrecords((1, 1.5)),  # 从元组(1, 1.5)创建记录数组
    np.recarray[Any, np.dtype[np.record]]  # 断言返回类型为np.recarray[Any, np.dtype[np.record]]
)

assert_type(
    np.rec.fromrecords(
        [(1, 1.5)],  # 从元组列表[(1, 1.5)]创建记录数组
        dtype=[("i8", np.int64), ("f8", np.float64)],  # 指定字段类型列表为i8: int64, f8: float64
    ),
    np.recarray[Any, np.dtype[np.record]],  # 断言返回类型为np.recarray[Any, np.dtype[np.record]]
)

assert_type(
    np.rec.fromrecords(
        REC_AR_V,  # 从REC_AR_V创建记录数组
        formats=[np.int64, np.float64],  # 指定格式列表为np.int64, np.float64
        names=["i8", "f8"]  # 指定字段名称列表为i8, f8
    ),
    np.recarray[Any, np.dtype[np.record]],  # 断言返回类型为np.recarray[Any, np.dtype[np.record]]
)

assert_type(
    np.rec.fromstring(
        b"(1, 1.5)",  # 字符串b"(1, 1.5)"表示的数据
        dtype=[("i8", np.int64), ("f8", np.float64)],  # 指定字段类型列表为i8: int64, f8: float64
    ),
    np.recarray[Any, np.dtype[np.record]],  # 断言返回类型为np.recarray[Any, np.dtype[np.record]]
)

assert_type(
    np.rec.fromstring(
        REC_AR_V,  # 从REC_AR_V创建记录数组
        formats=[np.int64, np.float64],  # 指定格式列表为np.int64, np.float64
        names=["i8", "f8"]  # 指定字段名称列表为i8, f8
    ),
    np.recarray[Any, np.dtype[np.record]],  # 断言返回类型为np.recarray[Any, np.dtype[np.record]]
)

assert_type(np.rec.fromfile(
    "test_file.txt",  # 文件路径为"test_file.txt"
    dtype=[("i8", np.int64), ("f8", np.float64)],  # 指定字段类型列表为i8: int64, f8: float64
), np.recarray[Any, np.dtype[Any]])  # 断言返回类型为np.recarray[Any, np.dtype[Any]]

assert_type(
    np.rec.fromfile(
        file_obj,  # 文件对象为file_obj
        formats=[np.int64, np.float64],  # 指定格式列表为np.int64, np.float64
        names=["i8", "f8"]  # 指定字段名称列表为i8, f8
    ),
    np.recarray[Any, np.dtype[np.record]],  # 断言返回类型为np.recarray[Any, np.dtype[np.record]]
)

assert_type(np.rec.array(AR_i8), np.recarray[Any, np.dtype[np.int64]])  # 断言从AR_i8创建的记录数组类型为np.recarray[Any, np.dtype
    # 创建一个结构化的 NumPy 数组，包含一个元组 (1, 1.5)，指定数据类型为 [("i8", np.int64), ("f8", np.float64)]
    np.rec.array([(1, 1.5)], dtype=[("i8", np.int64), ("f8", np.float64)]),
    # 声明一个类型为 np.recarray 的类型提示，该类型的元素可以是任意类型
    np.recarray[Any, np.dtype[Any]],
# 使用 assert_type 函数验证 np.rec.array 的返回类型
assert_type(
    # 创建一个记录数组，包含一个元组 (1, 1.5)，格式分别为 np.int64 和 np.float64，字段名分别为 "i8" 和 "f8"
    np.rec.array(
        [(1, 1.5)],
        formats=[np.int64, np.float64],
        names=["i8", "f8"]
    ),
    # 预期验证返回类型为 np.recarray，其中包含任意类型的元素和任意记录类型的数据类型
    np.recarray[Any, np.dtype[np.record]],
)

# 使用 assert_type 函数验证 np.rec.array 的返回类型
assert_type(
    # 创建一个空的记录数组，数据类型为 np.float64，形状为 (10, 3)
    np.rec.array(
        None,
        dtype=np.float64,
        shape=(10, 3),
    ),
    # 预期验证返回类型为 np.recarray，其中包含任意类型的元素和任意类型的数据类型
    np.recarray[Any, np.dtype[Any]],
)

# 使用 assert_type 函数验证 np.rec.array 的返回类型
assert_type(
    # 创建一个空的记录数组，格式分别为 np.int64 和 np.float64，字段名分别为 "i8" 和 "f8"，形状为 (10, 3)
    np.rec.array(
        None,
        formats=[np.int64, np.float64],
        names=["i8", "f8"],
        shape=(10, 3),
    ),
    # 预期验证返回类型为 np.recarray，其中包含任意类型的元素和任意记录类型的数据类型
    np.recarray[Any, np.dtype[np.record]],
)

# 使用 assert_type 函数验证 np.rec.array 的返回类型
assert_type(
    # 根据 file_obj 创建记录数组，数据类型为 np.float64
    np.rec.array(file_obj, dtype=np.float64),
    # 预期验证返回类型为 np.recarray，其中包含任意类型的元素和任意类型的数据类型
    np.recarray[Any, np.dtype[Any]],
)

# 使用 assert_type 函数验证 np.rec.array 的返回类型
assert_type(
    # 根据 file_obj 创建记录数组，格式分别为 np.int64 和 np.float64，字段名分别为 "i8" 和 "f8"
    np.rec.array(file_obj, formats=[np.int64, np.float64], names=["i8", "f8"]),
    # 预期验证返回类型为 np.recarray，其中包含任意类型的元素和任意记录类型的数据类型
    np.recarray[Any, np.dtype[np.record]],
)
```