# `.\numpy\numpy\tests\test_ctypeslib.py`

```
# 导入系统相关的模块
import sys
# 提供对 Python 系统配置信息的访问
import sysconfig
# 提供对弱引用对象的支持
import weakref
# 提供处理路径相关操作的功能
from pathlib import Path

# 导入 Pytest 测试框架
import pytest

# 导入 NumPy 库，并从其中导入需要的函数和类
import numpy as np
from numpy.ctypeslib import ndpointer, load_library, as_array
from numpy.testing import assert_, assert_array_equal, assert_raises, assert_equal

# 尝试导入 ctypes 库，如果导入失败则设置为 None
try:
    import ctypes
except ImportError:
    ctypes = None
else:
    # 定义一些变量，并初始化为 None
    cdll = None
    test_cdll = None
    # 检查是否存在 sys.gettotalrefcount 函数，若存在则进一步加载特定的动态链接库
    if hasattr(sys, 'gettotalrefcount'):
        # 尝试加载名为 '_multiarray_umath_d' 的动态链接库
        try:
            cdll = load_library(
                '_multiarray_umath_d', np._core._multiarray_umath.__file__
            )
        except OSError:
            pass
        # 尝试加载名为 '_multiarray_tests' 的动态链接库
        try:
            test_cdll = load_library(
                '_multiarray_tests', np._core._multiarray_tests.__file__
            )
        except OSError:
            pass
    # 如果 cdll 仍为 None，则尝试加载名为 '_multiarray_umath' 的动态链接库
    if cdll is None:
        cdll = load_library(
            '_multiarray_umath', np._core._multiarray_umath.__file__)
    # 如果 test_cdll 仍为 None，则尝试加载名为 '_multiarray_tests' 的动态链接库
    if test_cdll is None:
        test_cdll = load_library(
            '_multiarray_tests', np._core._multiarray_tests.__file__
        )

    # 从 test_cdll 中获取名为 'forward_pointer' 的符号
    c_forward_pointer = test_cdll.forward_pointer


# 使用 pytest.mark.skipif 标记，如果 ctypes 为 None，则跳过测试
@pytest.mark.skipif(ctypes is None,
                    reason="ctypes not available in this python")
# 使用 pytest.mark.skipif 标记，如果操作系统平台为 'cygwin'，则跳过测试
@pytest.mark.skipif(sys.platform == 'cygwin',
                    reason="Known to fail on cygwin")
# 定义测试类 TestLoadLibrary
class TestLoadLibrary:
    # 定义测试方法 test_basic
    def test_basic(self):
        # 获取 numpy._core._multiarray_umath 模块的文件路径
        loader_path = np._core._multiarray_umath.__file__

        # 使用 load_library 函数加载名为 '_multiarray_umath' 的动态链接库
        out1 = load_library('_multiarray_umath', loader_path)
        # 使用 load_library 函数加载名为 '_multiarray_umath' 的动态链接库（使用 pathlib.Path 对象）
        out2 = load_library(Path('_multiarray_umath'), loader_path)
        # 使用 load_library 函数加载名为 '_multiarray_umath' 的动态链接库（使用 pathlib.Path 对象）
        out3 = load_library('_multiarray_umath', Path(loader_path))
        # 使用 load_library 函数加载名为 '_multiarray_umath' 的动态链接库（传入字节串作为库名）
        out4 = load_library(b'_multiarray_umath', loader_path)

        # 断言 out1 到 out4 均为 ctypes.CDLL 类型的对象
        assert isinstance(out1, ctypes.CDLL)
        # 断言 out1 到 out4 均为同一个对象
        assert out1 is out2 is out3 is out4

    # 定义测试方法 test_basic2
    def test_basic2(self):
        # 尝试处理 #801 号问题：使用完整的库名（包括扩展名）调用 load_library 时出现问题
        try:
            # 获取系统配置变量 EXT_SUFFIX（用于获取库文件的扩展名）
            so_ext = sysconfig.get_config_var('EXT_SUFFIX')
            # 尝试加载名为 '_multiarray_umath' 加上扩展名的动态链接库
            load_library('_multiarray_umath%s' % so_ext,
                         np._core._multiarray_umath.__file__)
        # 如果导入出错，捕获 ImportError 异常
        except ImportError as e:
            # 输出错误消息，说明 ctypes 在当前 Python 环境下不可用，跳过该测试
            msg = ("ctypes is not available on this python: skipping the test"
                   " (import error was: %s)" % str(e))
            print(msg)
    # 测试数据类型
    def test_dtype(self):
        # 设置数据类型为 np.intc
        dt = np.intc
        # 创建指向该数据类型的指针对象 p
        p = ndpointer(dtype=dt)
        # 验证从参数中创建指针是否成功
        assert_(p.from_param(np.array([1], dt)))
        
        # 修改数据类型为 '<i4'
        dt = '<i4'
        # 更新指针对象 p 指向新的数据类型
        p = ndpointer(dtype=dt)
        # 验证从参数中创建指针是否成功
        assert_(p.from_param(np.array([1], dt)))
        
        # 设置数据类型为 '>i4' 的 numpy.dtype 对象
        dt = np.dtype('>i4')
        # 更新指针对象 p 指向新的数据类型
        p = ndpointer(dtype=dt)
        # 尝试从参数中创建指针，此处应成功
        p.from_param(np.array([1], dt))
        
        # 通过修改字节序创建新的数据类型对象
        assert_raises(TypeError, p.from_param,
                          np.array([1], dt.newbyteorder('swap')))
        
        # 定义数据类型的结构描述
        dtnames = ['x', 'y']
        dtformats = [np.intc, np.float64]
        dtdescr = {'names': dtnames, 'formats': dtformats}
        # 创建数据类型对象 dt
        dt = np.dtype(dtdescr)
        # 更新指针对象 p 指向新的数据类型
        p = ndpointer(dtype=dt)
        # 验证从参数中创建指针是否成功
        assert_(p.from_param(np.zeros((10,), dt)))
        
        # 创建与 dtdescr 相同的数据类型对象
        samedt = np.dtype(dtdescr)
        # 更新指针对象 p 指向新的数据类型
        p = ndpointer(dtype=samedt)
        # 验证从参数中创建指针是否成功
        assert_(p.from_param(np.zeros((10,), dt)))
        
        # 创建具有 align=True 属性的新数据类型对象 dt2
        dt2 = np.dtype(dtdescr, align=True)
        # 检查两个数据类型的 itemsize 是否相等
        if dt.itemsize != dt2.itemsize:
            # 如果不相等，预期抛出 TypeError 异常
            assert_raises(TypeError, p.from_param, np.zeros((10,), dt2))
        else:
            # 如果相等，验证从参数中创建指针是否成功
            assert_(p.from_param(np.zeros((10,), dt2)))

    # 测试数组的维度
    def test_ndim(self):
        # 创建指向零维数组的指针对象 p
        p = ndpointer(ndim=0)
        # 验证从参数中创建指针是否成功
        assert_(p.from_param(np.array(1)))
        # 预期抛出 TypeError 异常，因为参数是一维数组
        assert_raises(TypeError, p.from_param, np.array([1]))
        
        # 创建指向一维数组的指针对象 p
        p = ndpointer(ndim=1)
        # 预期抛出 TypeError 异常，因为参数是零维数组
        assert_raises(TypeError, p.from_param, np.array(1))
        # 验证从参数中创建指针是否成功
        assert_(p.from_param(np.array([1])))
        
        # 创建指向二维数组的指针对象 p
        p = ndpointer(ndim=2)
        # 验证从参数中创建指针是否成功
        assert_(p.from_param(np.array([[1]])))

    # 测试数组的形状
    def test_shape(self):
        # 创建指向形状为 (1, 2) 的数组的指针对象 p
        p = ndpointer(shape=(1, 2))
        # 验证从参数中创建指针是否成功
        assert_(p.from_param(np.array([[1, 2]])))
        # 预期抛出 TypeError 异常，因为参数的形状不匹配
        assert_raises(TypeError, p.from_param, np.array([[1], [2]]))
        
        # 创建指向零维数组的指针对象 p
        p = ndpointer(shape=())
        # 验证从参数中创建指针是否成功
        assert_(p.from_param(np.array(1)))

    # 测试数组的 flags 属性
    def test_flags(self):
        # 创建列序优先的二维数组 x
        x = np.array([[1, 2], [3, 4]], order='F')
        # 创建指向列序优先数组的指针对象 p
        p = ndpointer(flags='FORTRAN')
        # 验证从参数中创建指针是否成功
        assert_(p.from_param(x))
        
        # 创建连续存储的二维数组的指针对象 p
        p = ndpointer(flags='CONTIGUOUS')
        # 预期抛出 TypeError 异常，因为参数数组不是连续存储的
        assert_raises(TypeError, p.from_param, x)
        
        # 创建指向与 x 具有相同 flags 的数组的指针对象 p
        p = ndpointer(flags=x.flags.num)
        # 验证从参数中创建指针是否成功
        assert_(p.from_param(x))
        # 预期抛出 TypeError 异常，因为参数数组不是二维的
        assert_raises(TypeError, p.from_param, np.array([[1, 2], [3, 4]]))

    # 测试缓存功能
    def test_cache(self):
        # 验证相同 dtype 的指针对象是否相等
        assert_(ndpointer(dtype=np.float64) is ndpointer(dtype=np.float64))

        # 形状被规范化为元组形式的测试
        assert_(ndpointer(shape=2) is ndpointer(shape=(2,)))

        # 1.12 <= v < 1.16 版本存在的 bug，此处验证修复情况
        assert_(ndpointer(shape=2) is not ndpointer(ndim=2))
        assert_(ndpointer(ndim=2) is not ndpointer(shape=2))
@pytest.mark.skipif(ctypes is None,
                    reason="ctypes not available on this python installation")
class TestNdpointerCFunc:
    # 测试 NdpointerCFunc 类，当 ctypes 不可用时跳过测试
    def test_arguments(self):
        """ Test that arguments are coerced from arrays """
        # 设置 c_forward_pointer 函数的返回类型为 ctypes.c_void_p
        c_forward_pointer.restype = ctypes.c_void_p
        # 设置 c_forward_pointer 函数的参数类型为 ndpointer，维度为 2
        c_forward_pointer.argtypes = (ndpointer(ndim=2),)

        # 调用 c_forward_pointer 函数传入一个 2x3 的零数组
        c_forward_pointer(np.zeros((2, 3)))
        # 测试：维度过多的情况
        assert_raises(
            ctypes.ArgumentError, c_forward_pointer, np.zeros((2, 3, 4)))

    @pytest.mark.parametrize(
        'dt', [
            float,
            np.dtype(dict(
                formats=['<i4', '<i4'],
                names=['a', 'b'],
                offsets=[0, 2],
                itemsize=6
            ))
        ], ids=[
            'float',
            'overlapping-fields'
        ]
    )
    def test_return(self, dt):
        """ Test that return values are coerced to arrays """
        # 创建一个 2x3 的零数组，数据类型为 dt
        arr = np.zeros((2, 3), dt)
        # 创建指向 arr 的指针类型，形状和数据类型与 arr 相同
        ptr_type = ndpointer(shape=arr.shape, dtype=arr.dtype)

        # 设置 c_forward_pointer 函数的返回类型为 ptr_type
        c_forward_pointer.restype = ptr_type
        # 设置 c_forward_pointer 函数的参数类型为 ptr_type
        c_forward_pointer.argtypes = (ptr_type,)

        # 检查返回的数组 arr2 和 arr 是否指向相同的数据
        arr2 = c_forward_pointer(arr)
        assert_equal(arr2.dtype, arr.dtype)
        assert_equal(arr2.shape, arr.shape)
        assert_equal(
            arr2.__array_interface__['data'],
            arr.__array_interface__['data']
        )

    def test_vague_return_value(self):
        """ Test that vague ndpointer return values do not promote to arrays """
        # 创建一个 2x3 的零数组
        arr = np.zeros((2, 3))
        # 创建指向 arr 的指针类型，数据类型与 arr 相同
        ptr_type = ndpointer(dtype=arr.dtype)

        # 设置 c_forward_pointer 函数的返回类型为 ptr_type
        c_forward_pointer.restype = ptr_type
        # 设置 c_forward_pointer 函数的参数类型为 ptr_type
        c_forward_pointer.argtypes = (ptr_type,)

        # 调用 c_forward_pointer 函数传入 arr，检查返回值类型为 ptr_type
        ret = c_forward_pointer(arr)
        assert_(isinstance(ret, ptr_type))


@pytest.mark.skipif(ctypes is None,
                    reason="ctypes not available on this python installation")
class TestAsArray:
    # 测试 AsArray 类，当 ctypes 不可用时跳过测试
    def test_array(self):
        from ctypes import c_int

        # 定义一个 c_int 数组类型 pair_t，包含两个元素
        pair_t = c_int * 2
        # 将 pair_t(1, 2) 转换为 numpy 数组 a，并检查其形状和内容是否正确
        a = as_array(pair_t(1, 2))
        assert_equal(a.shape, (2,))
        assert_array_equal(a, np.array([1, 2]))
        
        # 创建包含多个 pair_t 元素的数组，转换为 numpy 数组并检查形状和内容
        a = as_array((pair_t * 3)(pair_t(1, 2), pair_t(3, 4), pair_t(5, 6)))
        assert_equal(a.shape, (3, 2))
        assert_array_equal(a, np.array([[1, 2], [3, 4], [5, 6]]))

    def test_pointer(self):
        from ctypes import c_int, cast, POINTER

        # 创建一个 c_int 类型的数组 p，包含 0 到 9 的整数
        p = cast((c_int * 10)(*range(10)), POINTER(c_int))

        # 将 p 转换为 numpy 数组 a，并检查其形状和内容是否正确
        a = as_array(p, shape=(10,))
        assert_equal(a.shape, (10,))
        assert_array_equal(a, np.arange(10))

        # 将 p 转换为 2x5 的 numpy 数组 a，并检查其形状和内容是否正确
        a = as_array(p, shape=(2, 5))
        assert_equal(a.shape, (2, 5))
        assert_array_equal(a, np.arange(10).reshape((2, 5)))

        # 必须提供 shape 参数，否则会抛出 TypeError
        assert_raises(TypeError, as_array, p)

    @pytest.mark.skipif(
            sys.version_info[:2] == (3, 12),
            reason="Broken in 3.12.0rc1, see gh-24399",
    )
    def test_struct_array_pointer(self):
        from ctypes import c_int16, Structure, pointer
        
        # 定义一个简单的结构体类 Struct，包含一个 c_int16 类型的字段 'a'
        class Struct(Structure):
            _fields_ = [('a', c_int16)]
        
        # 创建一个包含 3 个 Struct 实例的数组 Struct3
        Struct3 = 3 * Struct
        
        # 创建一个 C 风格的二维数组 c_array，每个元素是一个 Struct3 数组
        c_array = (2 * Struct3)(
            Struct3(Struct(a=1), Struct(a=2), Struct(a=3)),
            Struct3(Struct(a=4), Struct(a=5), Struct(a=6))
        )
        
        # 创建预期的 NumPy 数组 expected，包含与 c_array 相同的数据
        expected = np.array([
            [(1,), (2,), (3,)],
            [(4,), (5,), (6,)],
        ], dtype=[('a', np.int16)])
        
        # 定义一个用来检查数组是否符合预期的函数 check
        def check(x):
            assert_equal(x.dtype, expected.dtype)
            assert_equal(x, expected)
        
        # 检查各种方式转换 c_array 到 NumPy 数组的结果是否与预期一致
        check(as_array(c_array))  # 将 c_array 直接转换为 NumPy 数组
        check(as_array(pointer(c_array), shape=()))  # 将 c_array 的指针转换为 NumPy 数组
        check(as_array(pointer(c_array[0]), shape=(2,)))  # 将 c_array 的第一行的指针转换为 NumPy 数组
        check(as_array(pointer(c_array[0][0]), shape=(2, 3)))  # 将 c_array 的第一个元素的指针转换为 NumPy 数组
    
    def test_reference_cycles(self):
        # 导入 ctypes 模块
        import ctypes
        
        # 创建一个包含 100 个元素的 NumPy 短整型数组 a
        N = 100
        a = np.arange(N, dtype=np.short)
        
        # 获取数组 a 的 ctypes 指针 pnt
        pnt = np.ctypeslib.as_ctypes(a)
        
        # 使用 assert_no_gc_cycles 断言上下文管理器，确保没有循环引用
        with np.testing.assert_no_gc_cycles():
            # 将 pnt 转换为 ctypes.POINTER(ctypes.c_short) 类型的 newpnt
            newpnt = ctypes.cast(pnt, ctypes.POINTER(ctypes.c_short))
            # 使用 newpnt 构造一个新的 NumPy 数组 b
            b = np.ctypeslib.as_array(newpnt, (N,))
            # 删除 newpnt 和 b，应该清理掉这两个对象
            del newpnt, b
    
    def test_segmentation_fault(self):
        # 创建一个形状为 (224, 224, 3) 的全零 NumPy 数组 arr
        arr = np.zeros((224, 224, 3))
        
        # 将 arr 转换为 ctypes 数组 c_arr
        c_arr = np.ctypeslib.as_ctypes(arr)
        
        # 创建 arr 的弱引用 arr_ref
        arr_ref = weakref.ref(arr)
        
        # 删除 arr 变量本身
        del arr
        
        # 断言 arr_ref 的引用仍然存在，即 arr 尚未被垃圾回收
        assert_(arr_ref() is not None)
        
        # 访问 c_arr 的第一个元素，检查是否会导致分段错误
        c_arr[0][0][0]
@pytest.mark.skipif(ctypes is None,
                    reason="ctypes not available on this python installation")
# 创建一个测试类 TestAsCtypesType，用于测试从 NumPy 数据类型到 ctypes 类型的转换
class TestAsCtypesType:
    """ Test conversion from dtypes to ctypes types """

    # 测试标量类型的转换
    def test_scalar(self):
        # 定义一个小端序无符号 16 位整数数据类型
        dt = np.dtype('<u2')
        # 将 NumPy 数据类型转换为对应的 ctypes 类型
        ct = np.ctypeslib.as_ctypes_type(dt)
        # 断言转换后的 ctypes 类型与预期的 ctypes.c_uint16.__ctype_le__ 相等
        assert_equal(ct, ctypes.c_uint16.__ctype_le__)

        # 定义一个大端序无符号 16 位整数数据类型
        dt = np.dtype('>u2')
        # 将 NumPy 数据类型转换为对应的 ctypes 类型
        ct = np.ctypeslib.as_ctypes_type(dt)
        # 断言转换后的 ctypes 类型与预期的 ctypes.c_uint16.__ctype_be__ 相等
        assert_equal(ct, ctypes.c_uint16.__ctype_be__)

        # 定义一个无符号 16 位整数数据类型
        dt = np.dtype('u2')
        # 将 NumPy 数据类型转换为对应的 ctypes 类型
        ct = np.ctypeslib.as_ctypes_type(dt)
        # 断言转换后的 ctypes 类型与预期的 ctypes.c_uint16 相等
        assert_equal(ct, ctypes.c_uint16)

    # 测试子数组类型的转换
    def test_subarray(self):
        # 定义一个包含 2 行 3 列的 np.int32 子数组数据类型
        dt = np.dtype((np.int32, (2, 3)))
        # 将 NumPy 数据类型转换为对应的 ctypes 类型
        ct = np.ctypeslib.as_ctypes_type(dt)
        # 断言转换后的 ctypes 类型与预期的 2 * (3 * ctypes.c_int32) 相等
        assert_equal(ct, 2 * (3 * ctypes.c_int32))

    # 测试结构体类型的转换
    def test_structure(self):
        # 定义一个包含 'a' 和 'b' 两个字段的结构体数据类型，分别为 np.uint16 和 np.uint32
        dt = np.dtype([
            ('a', np.uint16),
            ('b', np.uint32),
        ])
        # 将 NumPy 数据类型转换为对应的 ctypes 类型
        ct = np.ctypeslib.as_ctypes_type(dt)
        # 断言转换后的 ctypes 类型是 ctypes.Structure 的子类
        assert_(issubclass(ct, ctypes.Structure))
        # 断言转换后的结构体大小与 NumPy 数据类型的 itemsize 相等
        assert_equal(ctypes.sizeof(ct), dt.itemsize)
        # 断言转换后的结构体字段与预期的字段列表相等
        assert_equal(ct._fields_, [
            ('a', ctypes.c_uint16),
            ('b', ctypes.c_uint32),
        ])

    # 测试对齐的结构体类型的转换
    def test_structure_aligned(self):
        # 定义一个包含 'a' 和 'b' 两个字段的结构体数据类型，并设置对齐为 True
        dt = np.dtype([
            ('a', np.uint16),
            ('b', np.uint32),
        ], align=True)
        # 将 NumPy 数据类型转换为对应的 ctypes 类型
        ct = np.ctypeslib.as_ctypes_type(dt)
        # 断言转换后的 ctypes 类型是 ctypes.Structure 的子类
        assert_(issubclass(ct, ctypes.Structure))
        # 断言转换后的结构体大小与 NumPy 数据类型的 itemsize 相等
        assert_equal(ctypes.sizeof(ct), dt.itemsize)
        # 断言转换后的结构体字段与预期的字段列表相等，包括对齐时可能添加的填充字段
        assert_equal(ct._fields_, [
            ('a', ctypes.c_uint16),
            ('', ctypes.c_char * 2),  # 填充字段
            ('b', ctypes.c_uint32),
        ])

    # 测试联合体类型的转换
    def test_union(self):
        # 定义一个包含 'a' 和 'b' 两个字段的联合体数据类型，分别为 np.uint16 和 np.uint32
        dt = np.dtype(dict(
            names=['a', 'b'],
            offsets=[0, 0],
            formats=[np.uint16, np.uint32]
        ))
        # 将 NumPy 数据类型转换为对应的 ctypes 类型
        ct = np.ctypeslib.as_ctypes_type(dt)
        # 断言转换后的 ctypes 类型是 ctypes.Union 的子类
        assert_(issubclass(ct, ctypes.Union))
        # 断言转换后的联合体大小与 NumPy 数据类型的 itemsize 相等
        assert_equal(ctypes.sizeof(ct), dt.itemsize)
        # 断言转换后的联合体字段与预期的字段列表相等
        assert_equal(ct._fields_, [
            ('a', ctypes.c_uint16),
            ('b', ctypes.c_uint32),
        ])

    # 测试带填充的联合体类型的转换
    def test_padded_union(self):
        # 定义一个带填充的联合体数据类型，包含 'a' 和 'b' 两个字段，总长度为 5
        dt = np.dtype(dict(
            names=['a', 'b'],
            offsets=[0, 0],
            formats=[np.uint16, np.uint32],
            itemsize=5,
        ))
        # 将 NumPy 数据类型转换为对应的 ctypes 类型
        ct = np.ctypeslib.as_ctypes_type(dt)
        # 断言转换后的 ctypes 类型是 ctypes.Union 的子类
        assert_(issubclass(ct, ctypes.Union))
        # 断言转换后的联合体大小与 NumPy 数据类型的 itemsize 相等
        assert_equal(ctypes.sizeof(ct), dt.itemsize)
        # 断言转换后的联合体字段与预期的字段列表相等，包括填充字段
        assert_equal(ct._fields_, [
            ('a', ctypes.c_uint16),
            ('b', ctypes.c_uint32),
            ('', ctypes.c_char * 5),  # 填充字段
        ])

    # 测试重叠的字段的转换
    def test_overlapping(self):
        # 定义一个包含 'a' 和 'b' 两个字段，偏移量分别为 0 和 2，数据类型均为 np.uint32 的数据类型
        dt = np.dtype(dict(
            names=['a', 'b'],
            offsets=[0, 2],
            formats=[np.uint32, np.uint32]
        ))
        # 断言尝试将这种重叠字段的数据类型转换为 ctypes 类型会引发 NotImplementedError
        assert_raises(NotImplementedError, np.ctypeslib.as_ctypes_type, dt)
```