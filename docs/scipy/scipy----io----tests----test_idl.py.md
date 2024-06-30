# `D:\src\scipysrc\scipy\scipy\io\tests\test_idl.py`

```
# 导入操作系统路径模块
from os import path
# 导入警告模块
import warnings

# 导入 NumPy 库并分别导入测试模块和功能函数
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
                           assert_, suppress_warnings)
# 导入 pytest 测试框架
import pytest

# 导入 SciPy 输入输出模块的特定部分
from scipy.io import readsav
from scipy.io import _idl

# 定义数据路径常量，使用当前文件所在路径拼接 'data' 子目录路径
DATA_PATH = path.join(path.dirname(__file__), 'data')


# 自定义断言函数，用于比较值和类型是否相同
def assert_identical(a, b):
    """Assert whether value AND type are the same"""
    assert_equal(a, b)
    if isinstance(b, str):
        assert_equal(type(a), type(b))
    else:
        assert_equal(np.asarray(a).dtype.type, np.asarray(b).dtype.type)


# 自定义断言函数，用于比较数组值和类型是否相同
def assert_array_identical(a, b):
    """Assert whether values AND type are the same"""
    assert_array_equal(a, b)
    assert_equal(a.dtype.type, b.dtype.type)


# 定义用于指针数组的向量化 ID 函数
vect_id = np.vectorize(id)


# 定义测试类 TestIdict，测试读取字典结构
class TestIdict:

    # 测试 ID 字典读取
    def test_idict(self):
        custom_dict = {'a': np.int16(999)}
        original_id = id(custom_dict)
        s = readsav(path.join(DATA_PATH, 'scalar_byte.sav'),
                    idict=custom_dict, verbose=False)
        assert_equal(original_id, id(s))  # 检查返回的 ID 是否与原始 ID 相同
        assert_('a' in s)  # 检查返回的结构中是否包含键 'a'
        assert_identical(s['a'], np.int16(999))  # 检查返回的值是否与预期相同
        assert_identical(s['i8u'], np.uint8(234))  # 检查返回的值是否与预期相同


# 定义测试类 TestScalars，用于测试不同标量数据类型的读取
class TestScalars:

    # 测试读取 8 位整数标量数据
    def test_byte(self):
        s = readsav(path.join(DATA_PATH, 'scalar_byte.sav'), verbose=False)
        assert_identical(s.i8u, np.uint8(234))  # 检查返回的值是否与预期相同

    # 测试读取 16 位整数标量数据
    def test_int16(self):
        s = readsav(path.join(DATA_PATH, 'scalar_int16.sav'), verbose=False)
        assert_identical(s.i16s, np.int16(-23456))  # 检查返回的值是否与预期相同

    # 测试读取 32 位整数标量数据
    def test_int32(self):
        s = readsav(path.join(DATA_PATH, 'scalar_int32.sav'), verbose=False)
        assert_identical(s.i32s, np.int32(-1234567890))  # 检查返回的值是否与预期相同

    # 测试读取 32 位浮点数标量数据
    def test_float32(self):
        s = readsav(path.join(DATA_PATH, 'scalar_float32.sav'), verbose=False)
        assert_identical(s.f32, np.float32(-3.1234567e+37))  # 检查返回的值是否与预期相同

    # 测试读取 64 位浮点数标量数据
    def test_float64(self):
        s = readsav(path.join(DATA_PATH, 'scalar_float64.sav'), verbose=False)
        assert_identical(s.f64, np.float64(-1.1976931348623157e+307))  # 检查返回的值是否与预期相同

    # 测试读取 32 位复数标量数据
    def test_complex32(self):
        s = readsav(path.join(DATA_PATH, 'scalar_complex32.sav'), verbose=False)
        assert_identical(s.c32, np.complex64(3.124442e13-2.312442e31j))  # 检查返回的值是否与预期相同

    # 测试读取字符串标量数据
    def test_bytes(self):
        s = readsav(path.join(DATA_PATH, 'scalar_string.sav'), verbose=False)
        msg = "The quick brown fox jumps over the lazy python"
        assert_identical(s.s, np.bytes_(msg))  # 检查返回的值是否与预期相同

    # 测试读取 64 位复数标量数据
    def test_complex64(self):
        s = readsav(path.join(DATA_PATH, 'scalar_complex64.sav'), verbose=False)
        assert_identical(
            s.c64,
            np.complex128(1.1987253647623157e+112-5.1987258887729157e+307j)
        )  # 检查返回的值是否与预期相同

    # 测试结构数据的读取
    def test_structure(self):
        pass  # 待实现

    # 测试堆指针的读取
    def test_heap_pointer(self):
        pass  # 待实现

    # 测试对象引用的读取
    def test_object_reference(self):
        pass  # 待实现
    # 测试读取保存文件中的 uint16 类型数据
    def test_uint16(self):
        # 使用 readsav 函数读取给定路径下的 scalar_uint16.sav 文件，关闭详细输出
        s = readsav(path.join(DATA_PATH, 'scalar_uint16.sav'), verbose=False)
        # 断言 s.i16u 的值与预期的 np.uint16(65511) 相等
        assert_identical(s.i16u, np.uint16(65511))
    
    # 测试读取保存文件中的 uint32 类型数据
    def test_uint32(self):
        # 使用 readsav 函数读取给定路径下的 scalar_uint32.sav 文件，关闭详细输出
        s = readsav(path.join(DATA_PATH, 'scalar_uint32.sav'), verbose=False)
        # 断言 s.i32u 的值与预期的 np.uint32(4294967233) 相等
        assert_identical(s.i32u, np.uint32(4294967233))
    
    # 测试读取保存文件中的 int64 类型数据
    def test_int64(self):
        # 使用 readsav 函数读取给定路径下的 scalar_int64.sav 文件，关闭详细输出
        s = readsav(path.join(DATA_PATH, 'scalar_int64.sav'), verbose=False)
        # 断言 s.i64s 的值与预期的 np.int64(-9223372036854774567) 相等
        assert_identical(s.i64s, np.int64(-9223372036854774567))
    
    # 测试读取保存文件中的 uint64 类型数据
    def test_uint64(self):
        # 使用 readsav 函数读取给定路径下的 scalar_uint64.sav 文件，关闭详细输出
        s = readsav(path.join(DATA_PATH, 'scalar_uint64.sav'), verbose=False)
        # 断言 s.i64u 的值与预期的 np.uint64(18446744073709529285) 相等
        assert_identical(s.i64u, np.uint64(18446744073709529285))
class TestCompressed(TestScalars):
    # Test that compressed .sav files can be read in

    def test_compressed(self):
        # 使用readsav函数读取压缩的.sav文件，不显示详细信息
        s = readsav(path.join(DATA_PATH, 'various_compressed.sav'), verbose=False)

        # 断言每个变量的值与给定的numpy数组相等
        assert_identical(s.i8u, np.uint8(234))
        assert_identical(s.f32, np.float32(-3.1234567e+37))
        assert_identical(
            s.c64,
            np.complex128(1.1987253647623157e+112-5.1987258887729157e+307j)
        )
        # 断言数组的形状与预期相等
        assert_equal(s.array5d.shape, (4, 3, 4, 6, 5))
        # 断言数组的特定元素与给定的numpy数组相等
        assert_identical(s.arrays.a[0], np.array([1, 2, 3], dtype=np.int16))
        assert_identical(s.arrays.b[0], np.array([4., 5., 6., 7.], dtype=np.float32))
        assert_identical(s.arrays.c[0],
                         np.array([np.complex64(1+2j), np.complex64(7+8j)]))
        assert_identical(s.arrays.d[0],
                         np.array([b"cheese", b"bacon", b"spam"], dtype=object))


class TestArrayDimensions:
    # Test that multi-dimensional arrays are read in with the correct dimensions

    def test_1d(self):
        # 使用readsav函数读取特定路径下的1维浮点数组的.sav文件，不显示详细信息
        s = readsav(path.join(DATA_PATH, 'array_float32_1d.sav'), verbose=False)
        # 断言数组的形状与预期相等
        assert_equal(s.array1d.shape, (123, ))

    def test_2d(self):
        # 使用readsav函数读取特定路径下的2维浮点数组的.sav文件，不显示详细信息
        s = readsav(path.join(DATA_PATH, 'array_float32_2d.sav'), verbose=False)
        # 断言数组的形状与预期相等
        assert_equal(s.array2d.shape, (22, 12))

    def test_3d(self):
        # 使用readsav函数读取特定路径下的3维浮点数组的.sav文件，不显示详细信息
        s = readsav(path.join(DATA_PATH, 'array_float32_3d.sav'), verbose=False)
        # 断言数组的形状与预期相等
        assert_equal(s.array3d.shape, (11, 22, 12))

    def test_4d(self):
        # 使用readsav函数读取特定路径下的4维浮点数组的.sav文件，不显示详细信息
        s = readsav(path.join(DATA_PATH, 'array_float32_4d.sav'), verbose=False)
        # 断言数组的形状与预期相等
        assert_equal(s.array4d.shape, (4, 5, 8, 7))

    def test_5d(self):
        # 使用readsav函数读取特定路径下的5维浮点数组的.sav文件，不显示详细信息
        s = readsav(path.join(DATA_PATH, 'array_float32_5d.sav'), verbose=False)
        # 断言数组的形状与预期相等
        assert_equal(s.array5d.shape, (4, 3, 4, 6, 5))

    def test_6d(self):
        # 使用readsav函数读取特定路径下的6维浮点数组的.sav文件，不显示详细信息
        s = readsav(path.join(DATA_PATH, 'array_float32_6d.sav'), verbose=False)
        # 断言数组的形状与预期相等
        assert_equal(s.array6d.shape, (3, 6, 4, 5, 3, 4))

    def test_7d(self):
        # 使用readsav函数读取特定路径下的7维浮点数组的.sav文件，不显示详细信息
        s = readsav(path.join(DATA_PATH, 'array_float32_7d.sav'), verbose=False)
        # 断言数组的形状与预期相等
        assert_equal(s.array7d.shape, (2, 1, 2, 3, 4, 3, 2))

    def test_8d(self):
        # 使用readsav函数读取特定路径下的8维浮点数组的.sav文件，不显示详细信息
        s = readsav(path.join(DATA_PATH, 'array_float32_8d.sav'), verbose=False)
        # 断言数组的形状与预期相等
        assert_equal(s.array8d.shape, (4, 3, 2, 1, 2, 3, 5, 4))


class TestStructures:

    def test_scalars(self):
        # 使用readsav函数读取特定路径下的结构体标量的.sav文件，不显示详细信息
        s = readsav(path.join(DATA_PATH, 'struct_scalars.sav'), verbose=False)
        # 断言每个标量的值与给定的numpy数组相等
        assert_identical(s.scalars.a, np.array(np.int16(1)))
        assert_identical(s.scalars.b, np.array(np.int32(2)))
        assert_identical(s.scalars.c, np.array(np.float32(3.)))
        assert_identical(s.scalars.d, np.array(np.float64(4.)))
        assert_identical(s.scalars.e, np.array([b"spam"], dtype=object))
        assert_identical(s.scalars.f, np.array(np.complex64(-1.+3j)))
    # 定义测试方法，用于验证读取的数据结构是否正确
    def test_scalars_replicated(self):
        # 使用readsav函数读取结构化数据文件的内容，关闭详细输出
        s = readsav(path.join(DATA_PATH, 'struct_scalars_replicated.sav'),
                    verbose=False)
        # 断言验证属性a的值是否为5个重复的int16类型的1
        assert_identical(s.scalars_rep.a, np.repeat(np.int16(1), 5))
        # 断言验证属性b的值是否为5个重复的int32类型的2
        assert_identical(s.scalars_rep.b, np.repeat(np.int32(2), 5))
        # 断言验证属性c的值是否为5个重复的float32类型的3.0
        assert_identical(s.scalars_rep.c, np.repeat(np.float32(3.), 5))
        # 断言验证属性d的值是否为5个重复的float64类型的4.0
        assert_identical(s.scalars_rep.d, np.repeat(np.float64(4.), 5))
        # 断言验证属性e的值是否为5个重复的b"spam"对象，类型转换为object
        assert_identical(s.scalars_rep.e, np.repeat(b"spam", 5).astype(object))
        # 断言验证属性f的值是否为5个重复的complex64类型的-1.0+3.0j
        assert_identical(s.scalars_rep.f, np.repeat(np.complex64(-1.+3j), 5))

    # 定义测试方法，用于验证读取的3D结构化数据是否正确
    def test_scalars_replicated_3d(self):
        # 使用readsav函数读取结构化数据文件的内容，关闭详细输出
        s = readsav(path.join(DATA_PATH, 'struct_scalars_replicated_3d.sav'),
                    verbose=False)
        # 断言验证属性a的值是否为24个重复的int16类型的1，并reshape为4x3x2的数组
        assert_identical(s.scalars_rep.a, np.repeat(np.int16(1), 24).reshape(4, 3, 2))
        # 断言验证属性b的值是否为24个重复的int32类型的2，并reshape为4x3x2的数组
        assert_identical(s.scalars_rep.b, np.repeat(np.int32(2), 24).reshape(4, 3, 2))
        # 断言验证属性c的值是否为24个重复的float32类型的3.0，并reshape为4x3x2的数组
        assert_identical(s.scalars_rep.c,
                         np.repeat(np.float32(3.), 24).reshape(4, 3, 2))
        # 断言验证属性d的值是否为24个重复的float64类型的4.0，并reshape为4x3x2的数组
        assert_identical(s.scalars_rep.d,
                         np.repeat(np.float64(4.), 24).reshape(4, 3, 2))
        # 断言验证属性e的值是否为24个重复的b"spam"对象，并reshape为4x3x2的数组，类型转换为object
        assert_identical(s.scalars_rep.e,
                         np.repeat(b"spam", 24).reshape(4, 3, 2).astype(object))
        # 断言验证属性f的值是否为24个重复的complex64类型的-1.0+3.0j，并reshape为4x3x2的数组
        assert_identical(s.scalars_rep.f,
                         np.repeat(np.complex64(-1.+3j), 24).reshape(4, 3, 2))

    # 定义测试方法，用于验证读取的数组数据是否正确
    def test_arrays(self):
        # 使用readsav函数读取数组数据文件的内容，关闭详细输出
        s = readsav(path.join(DATA_PATH, 'struct_arrays.sav'), verbose=False)
        # 断言验证数组a的第一个元素是否与给定的int16类型数组相等
        assert_array_identical(s.arrays.a[0], np.array([1, 2, 3], dtype=np.int16))
        # 断言验证数组b的第一个元素是否与给定的float32类型数组相等
        assert_array_identical(s.arrays.b[0],
                               np.array([4., 5., 6., 7.], dtype=np.float32))
        # 断言验证数组c的第一个元素是否与给定的complex64类型数组相等
        assert_array_identical(s.arrays.c[0],
                               np.array([np.complex64(1+2j), np.complex64(7+8j)]))
        # 断言验证数组d的第一个元素是否与给定的object类型数组相等
        assert_array_identical(s.arrays.d[0],
                               np.array([b"cheese", b"bacon", b"spam"], dtype=object))
    def test_arrays_replicated(self):
        # 从指定路径读取 'struct_arrays_replicated.sav' 文件的数据结构
        s = readsav(path.join(DATA_PATH, 'struct_arrays_replicated.sav'), verbose=False)

        # 检查列的数据类型
        assert_(s.arrays_rep.a.dtype.type is np.object_)
        assert_(s.arrays_rep.b.dtype.type is np.object_)
        assert_(s.arrays_rep.c.dtype.type is np.object_)
        assert_(s.arrays_rep.d.dtype.type is np.object_)

        # 检查列的形状
        assert_equal(s.arrays_rep.a.shape, (5, ))
        assert_equal(s.arrays_rep.b.shape, (5, ))
        assert_equal(s.arrays_rep.c.shape, (5, ))
        assert_equal(s.arrays_rep.d.shape, (5, ))

        # 检查值
        for i in range(5):
            # 检查数组 'a' 的第 i 个元素是否与给定的 int16 类型数组相等
            assert_array_identical(s.arrays_rep.a[i],
                                   np.array([1, 2, 3], dtype=np.int16))
            # 检查数组 'b' 的第 i 个元素是否与给定的 float32 类型数组相等
            assert_array_identical(s.arrays_rep.b[i],
                                   np.array([4., 5., 6., 7.], dtype=np.float32))
            # 检查数组 'c' 的第 i 个元素是否与给定的 complex64 类型数组相等
            assert_array_identical(s.arrays_rep.c[i],
                                   np.array([np.complex64(1+2j),
                                             np.complex64(7+8j)]))
            # 检查数组 'd' 的第 i 个元素是否与给定的 object 类型数组相等
            assert_array_identical(s.arrays_rep.d[i],
                                   np.array([b"cheese", b"bacon", b"spam"],
                                            dtype=object))

    def test_arrays_replicated_3d(self):
        # 从指定路径读取 'struct_arrays_replicated_3d.sav' 文件的数据结构
        s = readsav(path.join(DATA_PATH, 'struct_arrays_replicated_3d.sav'),
                    verbose=False)

        # 检查列的数据类型
        assert_(s.arrays_rep.a.dtype.type is np.object_)
        assert_(s.arrays_rep.b.dtype.type is np.object_)
        assert_(s.arrays_rep.c.dtype.type is np.object_)
        assert_(s.arrays_rep.d.dtype.type is np.object_)

        # 检查列的形状
        assert_equal(s.arrays_rep.a.shape, (4, 3, 2))
        assert_equal(s.arrays_rep.b.shape, (4, 3, 2))
        assert_equal(s.arrays_rep.c.shape, (4, 3, 2))
        assert_equal(s.arrays_rep.d.shape, (4, 3, 2))

        # 检查值
        for i in range(4):
            for j in range(3):
                for k in range(2):
                    # 检查三维数组 'a' 的第 (i, j, k) 元素是否与给定的 int16 类型数组相等
                    assert_array_identical(s.arrays_rep.a[i, j, k],
                                           np.array([1, 2, 3], dtype=np.int16))
                    # 检查三维数组 'b' 的第 (i, j, k) 元素是否与给定的 float32 类型数组相等
                    assert_array_identical(s.arrays_rep.b[i, j, k],
                                           np.array([4., 5., 6., 7.],
                                                    dtype=np.float32))
                    # 检查三维数组 'c' 的第 (i, j, k) 元素是否与给定的 complex64 类型数组相等
                    assert_array_identical(s.arrays_rep.c[i, j, k],
                                           np.array([np.complex64(1+2j),
                                                     np.complex64(7+8j)]))
                    # 检查三维数组 'd' 的第 (i, j, k) 元素是否与给定的 object 类型数组相等
                    assert_array_identical(s.arrays_rep.d[i, j, k],
                                           np.array([b"cheese", b"bacon", b"spam"],
                                                    dtype=object))
    # 定义测试方法，用于测试继承功能
    def test_inheritance(self):
        # 从指定路径读取 'struct_inherit.sav' 文件的数据，禁用详细输出模式
        s = readsav(path.join(DATA_PATH, 'struct_inherit.sav'), verbose=False)
        # 断言检查 s.fc.x 的值是否与预期的 np.int16 数组 [0] 相同
        assert_identical(s.fc.x, np.array([0], dtype=np.int16))
        # 断言检查 s.fc.y 的值是否与预期的 np.int16 数组 [0] 相同
        assert_identical(s.fc.y, np.array([0], dtype=np.int16))
        # 断言检查 s.fc.r 的值是否与预期的 np.int16 数组 [0] 相同
        assert_identical(s.fc.r, np.array([0], dtype=np.int16))
        # 断言检查 s.fc.c 的值是否与预期的 np.int16 数组 [4] 相同
        assert_identical(s.fc.c, np.array([4], dtype=np.int16))

    # 定义测试方法，用于测试可能由于 IDL 8.0 .sav 文件中缺失 nbyte 信息而导致的字节数组问题
    def test_arrays_corrupt_idl80(self):
        # 使用 suppress_warnings 上下文管理器，过滤掉特定的 UserWarning 提示
        with suppress_warnings() as sup:
            # 将特定路径下的 'struct_arrays_byte_idl80.sav' 文件读取为数据对象 s，禁用详细输出模式
            sup.filter(UserWarning, "Not able to verify number of bytes from header")
            s = readsav(path.join(DATA_PATH,'struct_arrays_byte_idl80.sav'),
                        verbose=False)

        # 断言检查 s.y.x[0] 的值是否与预期的 np.uint8 数组 [55, 66] 相同
        assert_identical(s.y.x[0], np.array([55, 66], dtype=np.uint8))
class TestPointers:
    # 检查 .sav 文件中指针是否在 Python 中产生对同一对象的引用

    def test_pointers(self):
        # 读取指定路径下的 .sav 文件，禁用详细输出
        s = readsav(path.join(DATA_PATH, 'scalar_heap_pointer.sav'), verbose=False)
        # 断言两个对象是否相同
        assert_identical(
            s.c64_pointer1,
            np.complex128(1.1987253647623157e+112-5.1987258887729157e+307j)
        )
        # 断言两个对象是否相同
        assert_identical(
            s.c64_pointer2,
            np.complex128(1.1987253647623157e+112-5.1987258887729157e+307j)
        )
        # 断言两个对象是否为同一对象
        assert_(s.c64_pointer1 is s.c64_pointer2)


class TestPointerArray:
    # 测试数组中的指针是否被正确读取

    def test_1d(self):
        # 读取指定路径下的一维数组 .sav 文件，禁用详细输出
        s = readsav(path.join(DATA_PATH, 'array_float32_pointer_1d.sav'), verbose=False)
        # 断言数组的形状是否符合预期
        assert_equal(s.array1d.shape, (123, ))
        # 断言数组的所有元素是否与指定的浮点数相等
        assert_(np.all(s.array1d == np.float32(4.)))
        # 断言向量化后的数组元素是否共享相同的内存地址
        assert_(np.all(vect_id(s.array1d) == id(s.array1d[0])))

    def test_2d(self):
        # 读取指定路径下的二维数组 .sav 文件，禁用详细输出
        s = readsav(path.join(DATA_PATH, 'array_float32_pointer_2d.sav'), verbose=False)
        # 断言数组的形状是否符合预期
        assert_equal(s.array2d.shape, (22, 12))
        # 断言数组的所有元素是否与指定的浮点数相等
        assert_(np.all(s.array2d == np.float32(4.)))
        # 断言向量化后的数组元素是否共享相同的内存地址
        assert_(np.all(vect_id(s.array2d) == id(s.array2d[0,0])))

    def test_3d(self):
        # 读取指定路径下的三维数组 .sav 文件，禁用详细输出
        s = readsav(path.join(DATA_PATH, 'array_float32_pointer_3d.sav'), verbose=False)
        # 断言数组的形状是否符合预期
        assert_equal(s.array3d.shape, (11, 22, 12))
        # 断言数组的所有元素是否与指定的浮点数相等
        assert_(np.all(s.array3d == np.float32(4.)))
        # 断言向量化后的数组元素是否共享相同的内存地址
        assert_(np.all(vect_id(s.array3d) == id(s.array3d[0,0,0])))

    def test_4d(self):
        # 读取指定路径下的四维数组 .sav 文件，禁用详细输出
        s = readsav(path.join(DATA_PATH, 'array_float32_pointer_4d.sav'), verbose=False)
        # 断言数组的形状是否符合预期
        assert_equal(s.array4d.shape, (4, 5, 8, 7))
        # 断言数组的所有元素是否与指定的浮点数相等
        assert_(np.all(s.array4d == np.float32(4.)))
        # 断言向量化后的数组元素是否共享相同的内存地址
        assert_(np.all(vect_id(s.array4d) == id(s.array4d[0,0,0,0])))

    def test_5d(self):
        # 读取指定路径下的五维数组 .sav 文件，禁用详细输出
        s = readsav(path.join(DATA_PATH, 'array_float32_pointer_5d.sav'), verbose=False)
        # 断言数组的形状是否符合预期
        assert_equal(s.array5d.shape, (4, 3, 4, 6, 5))
        # 断言数组的所有元素是否与指定的浮点数相等
        assert_(np.all(s.array5d == np.float32(4.)))
        # 断言向量化后的数组元素是否共享相同的内存地址
        assert_(np.all(vect_id(s.array5d) == id(s.array5d[0,0,0,0,0])))

    def test_6d(self):
        # 读取指定路径下的六维数组 .sav 文件，禁用详细输出
        s = readsav(path.join(DATA_PATH, 'array_float32_pointer_6d.sav'), verbose=False)
        # 断言数组的形状是否符合预期
        assert_equal(s.array6d.shape, (3, 6, 4, 5, 3, 4))
        # 断言数组的所有元素是否与指定的浮点数相等
        assert_(np.all(s.array6d == np.float32(4.)))
        # 断言向量化后的数组元素是否共享相同的内存地址
        assert_(np.all(vect_id(s.array6d) == id(s.array6d[0,0,0,0,0,0])))

    def test_7d(self):
        # 读取指定路径下的七维数组 .sav 文件，禁用详细输出
        s = readsav(path.join(DATA_PATH, 'array_float32_pointer_7d.sav'), verbose=False)
        # 断言数组的形状是否符合预期
        assert_equal(s.array7d.shape, (2, 1, 2, 3, 4, 3, 2))
        # 断言数组的所有元素是否与指定的浮点数相等
        assert_(np.all(s.array7d == np.float32(4.)))
        # 断言向量化后的数组元素是否共享相同的内存地址
        assert_(np.all(vect_id(s.array7d) == id(s.array7d[0,0,0,0,0,0,0])))

    def test_8d(self):
        # 读取指定路径下的八维数组 .sav 文件，禁用详细输出
        s = readsav(path.join(DATA_PATH, 'array_float32_pointer_8d.sav'), verbose=False)
        # 断言数组的形状是否符合预期
        assert_equal(s.array8d.shape, (4, 3, 2, 1, 2, 3, 5, 4))
        # 断言数组的所有元素是否与指定的浮点数相等
        assert_(np.all(s.array8d == np.float32(4.)))
        # 断言向量化后的数组元素是否共享相同的内存地址
        assert_(np.all(vect_id(s.array8d) == id(s.array8d[0,0,0,0,0,0,0,0])))
    # Test that structures are correctly read in

    # 测试结构是否被正确读取

    def test_scalars(self):
        # 读取结构指针保存的文件，关闭详细输出
        s = readsav(path.join(DATA_PATH, 'struct_pointers.sav'), verbose=False)
        # 断言结构中指针 g 的内容与预期相同，且数据类型为 np.object_
        assert_identical(s.pointers.g, np.array(np.float32(4.), dtype=np.object_))
        # 断言结构中指针 h 的内容与预期相同，且数据类型为 np.object_
        assert_identical(s.pointers.h, np.array(np.float32(4.), dtype=np.object_))
        # 断言结构中 g[0] 和 h[0] 的内存地址相同
        assert_(id(s.pointers.g[0]) == id(s.pointers.h[0]))

    def test_pointers_replicated(self):
        # 读取复制结构指针保存的文件，关闭详细输出
        s = readsav(path.join(DATA_PATH, 'struct_pointers_replicated.sav'),
                    verbose=False)
        # 断言结构中复制指针 g 的内容与预期相同，且数据类型为 np.object_
        assert_identical(s.pointers_rep.g,
                         np.repeat(np.float32(4.), 5).astype(np.object_))
        # 断言结构中复制指针 h 的内容与预期相同，且数据类型为 np.object_
        assert_identical(s.pointers_rep.h,
                         np.repeat(np.float32(4.), 5).astype(np.object_))
        # 断言结构中复制指针 g 和 h 的每个元素的内存地址相同
        assert_(np.all(vect_id(s.pointers_rep.g) == vect_id(s.pointers_rep.h)))

    def test_pointers_replicated_3d(self):
        # 读取三维复制结构指针保存的文件，关闭详细输出
        s = readsav(path.join(DATA_PATH, 'struct_pointers_replicated_3d.sav'),
                    verbose=False)
        # 生成预期的三维复制指针数据
        s_expect = np.repeat(np.float32(4.), 24).reshape(4, 3, 2).astype(np.object_)
        # 断言结构中复制指针 g 的内容与预期相同
        assert_identical(s.pointers_rep.g, s_expect)
        # 断言结构中复制指针 h 的内容与预期相同
        assert_identical(s.pointers_rep.h, s_expect)
        # 断言结构中复制指针 g 和 h 的每个元素的内存地址相同
        assert_(np.all(vect_id(s.pointers_rep.g) == vect_id(s.pointers_rep.h)))

    def test_arrays(self):
        # 读取结构数组指针保存的文件，关闭详细输出
        s = readsav(path.join(DATA_PATH, 'struct_pointer_arrays.sav'), verbose=False)
        # 断言结构中数组 g 的第一个元素与预期相同，且数据类型为 np.object_
        assert_array_identical(s.arrays.g[0],
                               np.repeat(np.float32(4.), 2).astype(np.object_))
        # 断言结构中数组 h 的第一个元素与预期相同，且数据类型为 np.object_
        assert_array_identical(s.arrays.h[0],
                               np.repeat(np.float32(4.), 3).astype(np.object_))
        # 断言结构中数组 g[0] 的每个元素的内存地址相同
        assert_(np.all(vect_id(s.arrays.g[0]) == id(s.arrays.g[0][0])))
        # 断言结构中数组 h[0] 的每个元素的内存地址相同
        assert_(np.all(vect_id(s.arrays.h[0]) == id(s.arrays.h[0][0])))
        # 断言结构中数组 g[0][0] 和 h[0][0] 的内存地址相同
        assert_(id(s.arrays.g[0][0]) == id(s.arrays.h[0][0]))

    def test_arrays_replicated(self):
        # 读取复制结构数组指针保存的文件，关闭详细输出
        s = readsav(path.join(DATA_PATH, 'struct_pointer_arrays_replicated.sav'),
                    verbose=False)

        # 检查列的数据类型是否为 np.object_
        assert_(s.arrays_rep.g.dtype.type is np.object_)
        assert_(s.arrays_rep.h.dtype.type is np.object_)

        # 检查列的形状是否正确
        assert_equal(s.arrays_rep.g.shape, (5, ))
        assert_equal(s.arrays_rep.h.shape, (5, ))

        # 检查每行的值是否正确
        for i in range(5):
            # 断言结构中复制数组 g[i] 的内容与预期相同
            assert_array_identical(s.arrays_rep.g[i],
                                   np.repeat(np.float32(4.), 2).astype(np.object_))
            # 断言结构中复制数组 h[i] 的内容与预期相同
            assert_array_identical(s.arrays_rep.h[i],
                                   np.repeat(np.float32(4.), 3).astype(np.object_))
            # 断言结构中复制数组 g[i] 的每个元素的内存地址相同
            assert_(np.all(vect_id(s.arrays_rep.g[i]) == id(s.arrays_rep.g[0][0])))
            # 断言结构中复制数组 h[i] 的每个元素的内存地址相同
            assert_(np.all(vect_id(s.arrays_rep.h[i]) == id(s.arrays_rep.h[0][0])))
    # 定义测试函数，测试读取的三维结构化指针数组数据
    def test_arrays_replicated_3d(self):
        # 拼接数据路径和文件名，生成文件路径
        pth = path.join(DATA_PATH, 'struct_pointer_arrays_replicated_3d.sav')
        # 使用readsav函数读取.sav文件内容，并关闭详细信息输出
        s = readsav(pth, verbose=False)

        # 检查列的数据类型是否为np.object_
        assert_(s.arrays_rep.g.dtype.type is np.object_)
        assert_(s.arrays_rep.h.dtype.type is np.object_)

        # 检查列的形状是否符合预期
        assert_equal(s.arrays_rep.g.shape, (4, 3, 2))
        assert_equal(s.arrays_rep.h.shape, (4, 3, 2))

        # 检查具体数值
        for i in range(4):
            for j in range(3):
                for k in range(2):
                    # 断言arrays_rep.g[i, j, k]的内容与重复的np.float32(4.)的对象类型数组相等
                    assert_array_identical(s.arrays_rep.g[i, j, k],
                            np.repeat(np.float32(4.), 2).astype(np.object_))
                    # 断言arrays_rep.h[i, j, k]的内容与重复的np.float32(4.)的对象类型数组相等
                    assert_array_identical(s.arrays_rep.h[i, j, k],
                            np.repeat(np.float32(4.), 3).astype(np.object_))
                    # 获取arrays_rep.g[i, j, k]的向量标识
                    g0 = vect_id(s.arrays_rep.g[i, j, k])
                    # 获取arrays_rep.g[0, 0, 0][0]的标识
                    g1 = id(s.arrays_rep.g[0, 0, 0][0])
                    # 检查g0是否与g1具有相同的所有元素
                    assert np.all(g0 == g1)
                    # 获取arrays_rep.h[i, j, k]的向量标识
                    h0 = vect_id(s.arrays_rep.h[i, j, k])
                    # 获取arrays_rep.h[0, 0, 0][0]的标识
                    h1 = id(s.arrays_rep.h[0, 0, 0][0])
                    # 检查h0是否与h1具有相同的所有元素
                    assert np.all(h0 == h1)
class TestTags:
    '''Test that sav files with description tag read at all'''

    # 定义测试方法，验证读取带有描述标签的 sav 文件是否成功
    def test_description(self):
        # 使用 readsav 函数读取特定路径下的 scalar_byte_descr.sav 文件，关闭详细输出
        s = readsav(path.join(DATA_PATH, 'scalar_byte_descr.sav'), verbose=False)
        # 断言 s.i8u 等于无符号整数 234
        assert_identical(s.i8u, np.uint8(234))


def test_null_pointer():
    # 回归测试空指针问题
    # 使用 readsav 函数读取特定路径下的 null_pointer.sav 文件，关闭详细输出
    s = readsav(path.join(DATA_PATH, 'null_pointer.sav'), verbose=False)
    # 断言 s.point 等于 None
    assert_identical(s.point, None)
    # 断言 s.check 等于整数 5 的 16 位有符号整数类型
    assert_identical(s.check, np.int16(5))


def test_invalid_pointer():
    # 回归测试无效指针问题 (gh-4613)

    # 在某些实际文件中，指针有时可能指向不存在的堆变量。
    # 在这种情况下，我们现在会优雅地处理这个变量，并用 None 替换该变量并发出警告。
    # 由于很难人工制造这样的文件，这里使用的文件已被编辑以强制使指针引用无效。
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # 使用 readsav 函数读取特定路径下的 invalid_pointer.sav 文件，关闭详细输出
        s = readsav(path.join(DATA_PATH, 'invalid_pointer.sav'), verbose=False)
    # 断言警告列表长度为 1
    assert_(len(w) == 1)
    # 断言警告信息字符串与指定的消息相匹配
    assert_(str(w[0].message) == ("Variable referenced by pointer not found in "
                                  "heap: variable will be set to None"))
    # 断言 s['a'] 的值为包含两个 None 的 NumPy 数组
    assert_identical(s['a'], np.array([None, None]))


def test_attrdict():
    # 创建 _idl.AttrDict 对象 d，初始化包含键 'one' 和值 1
    d = _idl.AttrDict({'one': 1})
    # 断言 d['one'] 等于 1
    assert d['one'] == 1
    # 断言 d.one 等于 1
    assert d.one == 1
    # 使用 pytest 引发 KeyError 异常，断言 d['two'] 不存在
    with pytest.raises(KeyError):
        d['two']
    # 使用 pytest 引发 AttributeError 异常，断言 d.two 属性不存在
    with pytest.raises(AttributeError, match='has no attribute'):
        d.two
```