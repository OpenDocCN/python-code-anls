# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\npyio.pyi`

```
import re  # 导入正则表达式模块
import sys  # 导入系统相关的模块
import zipfile  # 导入处理 ZIP 文件的模块
import pathlib  # 提供处理路径的类和函数
from typing import IO, Any  # 导入类型提示相关的类和函数
from collections.abc import Mapping  # 导入映射类型相关的抽象基类

import numpy.typing as npt  # 导入 NumPy 类型提示
import numpy as np  # 导入 NumPy 数学计算库
from numpy.lib._npyio_impl import BagObj  # 导入 NumPy 内部使用的对象

if sys.version_info >= (3, 11):
    from typing import assert_type  # 如果 Python 版本大于等于 3.11，导入 assert_type 函数
else:
    from typing_extensions import assert_type  # 否则，导入 typing_extensions 中的 assert_type 函数

str_path: str  # 声明一个字符串类型的变量 str_path
pathlib_path: pathlib.Path  # 声明一个路径对象的变量 pathlib_path
str_file: IO[str]  # 声明一个字符串类型的文件对象变量 str_file
bytes_file: IO[bytes]  # 声明一个字节类型的文件对象变量 bytes_file

npz_file: np.lib.npyio.NpzFile  # 声明一个 NumPy .npz 文件对象变量 npz_file

AR_i8: npt.NDArray[np.int64]  # 声明一个 NumPy int64 数组类型变量 AR_i8
AR_LIKE_f8: list[float]  # 声明一个浮点数列表类型变量 AR_LIKE_f8

class BytesWriter:  # 定义一个字节流写入类 BytesWriter
    def write(self, data: bytes) -> None: ...  # 定义一个接受 bytes 类型参数的写入方法

class BytesReader:  # 定义一个字节流读取类 BytesReader
    def read(self, n: int = ...) -> bytes: ...  # 定义一个返回 bytes 类型的读取方法
    def seek(self, offset: int, whence: int = ...) -> int: ...  # 定义一个设置读取位置的方法

bytes_writer: BytesWriter  # 声明一个 BytesWriter 类型的变量 bytes_writer
bytes_reader: BytesReader  # 声明一个 BytesReader 类型的变量 bytes_reader

assert_type(npz_file.zip, zipfile.ZipFile)  # 断言 npz_file.zip 是一个 zipfile.ZipFile 类型的对象
assert_type(npz_file.fid, None | IO[str])  # 断言 npz_file.fid 是 None 或者 IO[str] 类型的对象
assert_type(npz_file.files, list[str])  # 断言 npz_file.files 是一个字符串列表
assert_type(npz_file.allow_pickle, bool)  # 断言 npz_file.allow_pickle 是布尔类型
assert_type(npz_file.pickle_kwargs, None | Mapping[str, Any])  # 断言 npz_file.pickle_kwargs 是 None 或者 Mapping[str, Any] 类型
assert_type(npz_file.f, BagObj[np.lib.npyio.NpzFile])  # 断言 npz_file.f 是 BagObj[np.lib.npyio.NpzFile] 类型的对象
assert_type(npz_file["test"], npt.NDArray[Any])  # 断言 npz_file["test"] 是任意类型的 NumPy 数组
assert_type(len(npz_file), int)  # 断言 len(npz_file) 返回整数类型的长度
with npz_file as f:
    assert_type(f, np.lib.npyio.NpzFile)  # 在上下文中使用 npz_file，断言 f 是 np.lib.npyio.NpzFile 类型的对象

assert_type(np.load(bytes_file), Any)  # 断言 np.load(bytes_file) 返回任意类型的结果
assert_type(np.load(pathlib_path, allow_pickle=True), Any)  # 断言 np.load(pathlib_path, allow_pickle=True) 返回任意类型的结果
assert_type(np.load(str_path, encoding="bytes"), Any)  # 断言 np.load(str_path, encoding="bytes") 返回任意类型的结果
assert_type(np.load(bytes_reader), Any)  # 断言 np.load(bytes_reader) 返回任意类型的结果

assert_type(np.save(bytes_file, AR_LIKE_f8), None)  # 断言 np.save(bytes_file, AR_LIKE_f8) 返回 None
assert_type(np.save(pathlib_path, AR_i8, allow_pickle=True), None)  # 断言 np.save(pathlib_path, AR_i8, allow_pickle=True) 返回 None
assert_type(np.save(str_path, AR_LIKE_f8), None)  # 断言 np.save(str_path, AR_LIKE_f8) 返回 None
assert_type(np.save(bytes_writer, AR_LIKE_f8), None)  # 断言 np.save(bytes_writer, AR_LIKE_f8) 返回 None

assert_type(np.savez(bytes_file, AR_LIKE_f8), None)  # 断言 np.savez(bytes_file, AR_LIKE_f8) 返回 None
assert_type(np.savez(pathlib_path, ar1=AR_i8, ar2=AR_i8), None)  # 断言 np.savez(pathlib_path, ar1=AR_i8, ar2=AR_i8) 返回 None
assert_type(np.savez(str_path, AR_LIKE_f8, ar1=AR_i8), None)  # 断言 np.savez(str_path, AR_LIKE_f8, ar1=AR_i8) 返回 None
assert_type(np.savez(bytes_writer, AR_LIKE_f8, ar1=AR_i8), None)  # 断言 np.savez(bytes_writer, AR_LIKE_f8, ar1=AR_i8) 返回 None

assert_type(np.savez_compressed(bytes_file, AR_LIKE_f8), None)  # 断言 np.savez_compressed(bytes_file, AR_LIKE_f8) 返回 None
assert_type(np.savez_compressed(pathlib_path, ar1=AR_i8, ar2=AR_i8), None)  # 断言 np.savez_compressed(pathlib_path, ar1=AR_i8, ar2=AR_i8) 返回 None
assert_type(np.savez_compressed(str_path, AR_LIKE_f8, ar1=AR_i8), None)  # 断言 np.savez_compressed(str_path, AR_LIKE_f8, ar1=AR_i8) 返回 None
assert_type(np.savez_compressed(bytes_writer, AR_LIKE_f8, ar1=AR_i8), None)  # 断言 np.savez_compressed(bytes_writer, AR_LIKE_f8, ar1=AR_i8) 返回 None

assert_type(np.loadtxt(bytes_file), npt.NDArray[np.float64])  # 断言 np.loadtxt(bytes_file) 返回 npt.NDArray[np.float64] 类型的 NumPy 数组
assert_type(np.loadtxt(pathlib_path, dtype=np.str_), npt.NDArray[np.str_])  # 断言 np.loadtxt(pathlib_path, dtype=np.str_) 返回 npt.NDArray[np.str_] 类型的 NumPy 数组
assert_type(np.loadtxt(str_path, dtype=str, skiprows=2), npt.NDArray[Any])  # 断言 np.loadtxt(str_path, dtype=str, skiprows=2) 返回 npt.NDArray[Any] 类型的 NumPy 数组
assert_type(np.loadtxt(str_file, comments="test"), npt.NDArray[np.float64])  # 断言 np.loadtxt(str_file, comments="test") 返回 npt.NDArray[np.float64] 类型的 NumPy 数组
assert_type(np.loadtxt(str_file, comments=None), npt.NDArray[np.float64])  # 断言 np.loadtxt(str_file, comments=None) 返回 npt.NDArray[np.float64] 类型的 NumPy 数组
assert_type(np.loadtxt(str_path, delimiter="\n"), npt.NDArray[np.float64])  # 断言 np.loadtxt(str_path, delimiter="\n") 返回 npt.NDArray[np.float64] 类型的 NumPy 数组
assert_type(np.loadtxt(str_path, ndmin=2), npt.NDArray[np.float64])  # 断言 np.loadtxt(str_path, ndmin=2) 返回 npt.NDArray[np.float64] 类型的 NumPy 数组
assert_type(np.loadtxt(["1", "2", "3"]), npt.NDArray[np.float64])  # 断言 np.loadtxt(["1", "2", "3"]) 返回 npt.NDArray[np.float64] 类型的 NumPy 数组

assert_type(np.fromregex(bytes_file, "test", np.float64), npt.NDArray[np.float64])  # 断言 np.fromregex(bytes_file, "test", np.float64) 返回 npt.NDArray[np.float64] 类型的 NumPy 数组
assert_type(np.fromregex(str_file, b"test", dtype=float), npt.NDArray[Any])
# 调用 numpy 的 fromregex 函数，从指定的 pathlib 路径中读取数据，使用正则表达式 "test" 匹配内容，并期望返回 np.float64 类型的数组
assert_type(np.fromregex(pathlib_path, "test", np.float64), npt.NDArray[np.float64])

# 调用 numpy 的 fromregex 函数，从给定的字节流中读取数据，使用正则表达式 "test" 匹配内容，并期望返回 np.float64 类型的数组
assert_type(np.fromregex(bytes_reader, "test", np.float64), npt.NDArray[np.float64])

# 调用 numpy 的 genfromtxt 函数，从给定的字节流中读取数据，并期望返回任意类型的数组
assert_type(np.genfromtxt(bytes_file), npt.NDArray[Any])

# 调用 numpy 的 genfromtxt 函数，从 pathlib 路径中读取数据，期望返回 np.str_ 类型的数组，并使用指定的数据类型 dtype=np.str_
assert_type(np.genfromtxt(pathlib_path, dtype=np.str_), npt.NDArray[np.str_])

# 调用 numpy 的 genfromtxt 函数，从指定的字符串路径中读取数据，跳过前两行标题，期望返回任意类型的数组
assert_type(np.genfromtxt(str_path, dtype=str, skip_header=2), npt.NDArray[Any])

# 调用 numpy 的 genfromtxt 函数，从指定的字符串文件中读取数据，"test" 字符串作为注释，期望返回任意类型的数组
assert_type(np.genfromtxt(str_file, comments="test"), npt.NDArray[Any])

# 调用 numpy 的 genfromtxt 函数，从指定的字符串路径中读取数据，使用换行符作为分隔符，期望返回任意类型的数组
assert_type(np.genfromtxt(str_path, delimiter="\n"), npt.NDArray[Any])

# 调用 numpy 的 genfromtxt 函数，从指定的字符串路径中读取数据，期望返回至少包含两个维度的数组
assert_type(np.genfromtxt(str_path, ndmin=2), npt.NDArray[Any])

# 调用 numpy 的 genfromtxt 函数，从提供的字符串列表中读取数据，期望返回至少包含两个维度的数组
assert_type(np.genfromtxt(["1", "2", "3"], ndmin=2), npt.NDArray[Any])
```