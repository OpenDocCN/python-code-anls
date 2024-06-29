# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\arrayterator.pyi`

```py
import sys
from typing import Any  # 导入 Any 类型用于泛型
from collections.abc import Generator  # 导入 Generator 类型用于定义生成器对象

import numpy as np  # 导入 NumPy 库并用 np 别名表示
import numpy.typing as npt  # 导入 NumPy 的类型提示模块

if sys.version_info >= (3, 11):
    from typing import assert_type  # 根据 Python 版本导入 assert_type 函数
else:
    from typing_extensions import assert_type  # 使用 typing_extensions 模块导入 assert_type 函数

AR_i8: npt.NDArray[np.int64]  # 声明 AR_i8 变量为 NumPy 的 int64 类型数组
ar_iter = np.lib.Arrayterator(AR_i8)  # 使用 AR_i8 创建一个数组迭代器对象

assert_type(ar_iter.var, npt.NDArray[np.int64])  # 断言 ar_iter.var 的类型为 int64 类型的 NumPy 数组
assert_type(ar_iter.buf_size, None | int)  # 断言 ar_iter.buf_size 的类型可以是 None 或者 int
assert_type(ar_iter.start, list[int])  # 断言 ar_iter.start 的类型为 int 类型的列表
assert_type(ar_iter.stop, list[int])  # 断言 ar_iter.stop 的类型为 int 类型的列表
assert_type(ar_iter.step, list[int])  # 断言 ar_iter.step 的类型为 int 类型的列表
assert_type(ar_iter.shape, tuple[int, ...])  # 断言 ar_iter.shape 的类型为不定长的 int 类型元组
assert_type(ar_iter.flat, Generator[np.int64, None, None])  # 断言 ar_iter.flat 的类型为生成器，生成 int64 类型的数据

assert_type(ar_iter.__array__(), npt.NDArray[np.int64])  # 断言 ar_iter.__array__() 方法返回的类型为 int64 类型的 NumPy 数组

for i in ar_iter:  # 迭代 ar_iter 中的元素
    assert_type(i, npt.NDArray[np.int64])  # 断言每个迭代出的 i 的类型为 int64 类型的 NumPy 数组

assert_type(ar_iter[0], np.lib.Arrayterator[Any, np.dtype[np.int64]])  # 断言 ar_iter[0] 的类型为 Arrayterator 对象
assert_type(ar_iter[...], np.lib.Arrayterator[Any, np.dtype[np.int64]])  # 断言 ar_iter[...] 的类型为 Arrayterator 对象
assert_type(ar_iter[:], np.lib.Arrayterator[Any, np.dtype[np.int64]])  # 断言 ar_iter[:] 的类型为 Arrayterator 对象
assert_type(ar_iter[0, 0, 0], np.lib.Arrayterator[Any, np.dtype[np.int64]])  # 断言 ar_iter[0, 0, 0] 的类型为 Arrayterator 对象
assert_type(ar_iter[..., 0, :], np.lib.Arrayterator[Any, np.dtype[np.int64]])  # 断言 ar_iter[..., 0, :] 的类型为 Arrayterator 对象
```