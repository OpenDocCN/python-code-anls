# `D:\src\scipysrc\scipy\scipy\io\matlab\tests\test_mio_utils.py`

```
"""
Testing 模块测试用例

"""

import numpy as np  # 导入 NumPy 库

from numpy.testing import assert_array_equal, assert_  # 导入 NumPy 测试模块中的断言函数

from scipy.io.matlab._mio_utils import squeeze_element, chars_to_strings  # 导入 SciPy 中的工具函数


def test_squeeze_element():
    a = np.zeros((1,3))  # 创建一个形状为 (1,3) 的全零 NumPy 数组
    assert_array_equal(np.squeeze(a), squeeze_element(a))  # 使用 NumPy 的 squeeze 函数和自定义的 squeeze_element 函数比较结果是否相等
    # 从 squeeze 函数中得到的 0 维输出给出标量
    sq_int = squeeze_element(np.zeros((1,1), dtype=float))  # 调用 squeeze_element 函数，传入一个形状为 (1,1) 的全零数组，期望返回一个浮点数
    assert_(isinstance(sq_int, float))  # 使用 NumPy 的 assert_ 函数检查返回的 sq_int 是否是 float 类型
    # 除非它是一个结构化数组
    sq_sa = squeeze_element(np.zeros((1,1),dtype=[('f1', 'f')]))  # 调用 squeeze_element 函数，传入一个结构化数组，期望返回一个 NumPy 数组
    assert_(isinstance(sq_sa, np.ndarray))  # 使用 NumPy 的 assert_ 函数检查返回的 sq_sa 是否是 NumPy 数组类型
    # 对于空数组，squeeze 保持其数据类型不变
    sq_empty = squeeze_element(np.empty(0, np.uint8))  # 调用 squeeze_element 函数，传入一个空的 uint8 类型数组，期望返回一个空数组
    assert sq_empty.dtype == np.uint8  # 检查返回的 sq_empty 的数据类型是否为 uint8


def test_chars_strings():
    # 将字符数组转换为字符串数组
    strings = ['learn ', 'python', 'fast  ', 'here  ']  # 创建一个字符串列表
    str_arr = np.array(strings, dtype='U6')  # 使用 NumPy 创建一个 Unicode 字符串数组，指定每个字符串的最大长度为 6，形状为 (4,)
    chars = [list(s) for s in strings]  # 将每个字符串转换为字符列表
    char_arr = np.array(chars, dtype='U1')  # 使用 NumPy 创建一个 Unicode 字符数组，每个字符都是长度为 1 的字符串，形状为 (4,6)
    assert_array_equal(chars_to_strings(char_arr), str_arr)  # 调用 chars_to_strings 函数，将字符数组转换为字符串数组，并与预期的 str_arr 进行比较
    ca2d = char_arr.reshape((2,2,6))  # 将字符数组重塑为形状为 (2,2,6) 的三维数组
    sa2d = str_arr.reshape((2,2))  # 将字符串数组重塑为形状为 (2,2) 的二维数组
    assert_array_equal(chars_to_strings(ca2d), sa2d)  # 再次调用 chars_to_strings 函数，并与预期的 sa2d 进行比较
    ca3d = char_arr.reshape((1,2,2,6))  # 将字符数组重塑为形状为 (1,2,2,6) 的四维数组
    sa3d = str_arr.reshape((1,2,2))  # 将字符串数组重塑为形状为 (1,2,2) 的三维数组
    assert_array_equal(chars_to_strings(ca3d), sa3d)  # 再次调用 chars_to_strings 函数，并与预期的 sa3d 进行比较
    # Fortran 排序的数组
    char_arrf = np.array(chars, dtype='U1', order='F')  # 创建一个 Fortran 排序的字符数组，形状为 (4,6)
    assert_array_equal(chars_to_strings(char_arrf), str_arr)  # 调用 chars_to_strings 函数，将字符数组转换为字符串数组，并与预期的 str_arr 进行比较
    # 空数组
    arr = np.array([['']], dtype='U1')  # 创建一个包含空字符串的字符数组
    out_arr = np.array([''], dtype='U1')  # 创建一个包含空字符串的输出字符串数组
    assert_array_equal(chars_to_strings(arr), out_arr)  # 调用 chars_to_strings 函数，将字符数组转换为字符串数组，并与预期的 out_arr 进行比较
```