# `.\numpy\numpy\_core\tests\test_api.py`

```
# 导入 sys 模块，用于获取系统相关信息
import sys

# 导入 numpy 库，并将其命名为 np
import numpy as np

# 导入 numpy 内部的 umath 模块，用于数学运算
import numpy._core.umath as ncu

# 导入 numpy 内部的 _rational_tests 模块中的 rational 对象
from numpy._core._rational_tests import rational

# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 导入 numpy.testing 模块中的多个断言函数，用于测试 numpy 数组的行为和属性
from numpy.testing import (
     assert_, assert_equal, assert_array_equal, assert_raises, assert_warns,
     HAS_REFCOUNT
    )


# 定义测试函数 test_array_array
def test_array_array():
    # 获取 Python 内置对象 object 的类型
    tobj = type(object)

    # 创建一个形状为 (1, 1) 的 numpy 数组，元素值为 1.0，数据类型为 np.float64
    ones11 = np.ones((1, 1), np.float64)

    # 获取 ones11 对象的类型
    tndarray = type(ones11)

    # 测试 np.array 函数对于 numpy 数组的行为
    assert_equal(np.array(ones11, dtype=np.float64), ones11)

    # 如果支持引用计数（HAS_REFCOUNT 为 True），则测试引用计数的变化
    if HAS_REFCOUNT:
        old_refcount = sys.getrefcount(tndarray)
        np.array(ones11)
        assert_equal(old_refcount, sys.getrefcount(tndarray))

    # 测试 np.array 函数对于 None 的行为
    assert_equal(np.array(None, dtype=np.float64),
                 np.array(np.nan, dtype=np.float64))

    # 如果支持引用计数（HAS_REFCOUNT 为 True），则测试引用计数的变化
    if HAS_REFCOUNT:
        old_refcount = sys.getrefcount(tobj)
        np.array(None, dtype=np.float64)
        assert_equal(old_refcount, sys.getrefcount(tobj))

    # 测试 np.array 函数对于标量的行为
    assert_equal(np.array(1.0, dtype=np.float64),
                 np.ones((), dtype=np.float64))

    # 如果支持引用计数（HAS_REFCOUNT 为 True），则测试引用计数的变化
    if HAS_REFCOUNT:
        old_refcount = sys.getrefcount(np.float64)
        np.array(np.array(1.0, dtype=np.float64), dtype=np.float64)
        assert_equal(old_refcount, sys.getrefcount(np.float64))

    # 测试 np.array 函数对于字符串的行为
    S2 = np.dtype((bytes, 2))
    S3 = np.dtype((bytes, 3))
    S5 = np.dtype((bytes, 5))
    assert_equal(np.array(b"1.0", dtype=np.float64),
                 np.ones((), dtype=np.float64))
    assert_equal(np.array(b"1.0").dtype, S3)
    assert_equal(np.array(b"1.0", dtype=bytes).dtype, S3)
    assert_equal(np.array(b"1.0", dtype=S2), np.array(b"1."))  # 转换为指定长度的 bytes 类型
    assert_equal(np.array(b"1", dtype=S5), np.ones((), dtype=S5))  # 转换为指定长度的 bytes 类型

    # 测试 np.array 函数对于 Unicode 字符串的行为
    U2 = np.dtype((str, 2))
    U3 = np.dtype((str, 3))
    U5 = np.dtype((str, 5))
    assert_equal(np.array("1.0", dtype=np.float64),
                 np.ones((), dtype=np.float64))
    assert_equal(np.array("1.0").dtype, U3)
    assert_equal(np.array("1.0", dtype=str).dtype, U3)
    assert_equal(np.array("1.0", dtype=U2), np.array(str("1.")))  # 转换为指定长度的 str 类型
    assert_equal(np.array("1", dtype=U5), np.ones((), dtype=U5))  # 转换为指定长度的 str 类型

    # 获取内置模块 __builtins__ 中的 get 方法，并断言其存在
    builtins = getattr(__builtins__, '__dict__', __builtins__)
    assert_(hasattr(builtins, 'get'))

    # 测试 np.array 函数对于 memoryview 对象的行为
    dat = np.array(memoryview(b'1.0'), dtype=np.float64)
    assert_equal(dat, [49.0, 46.0, 48.0])  # 转换为 float64 数组
    assert_(dat.dtype.type is np.float64)

    dat = np.array(memoryview(b'1.0'))
    assert_equal(dat, [49, 46, 48])  # 转换为 uint8 数组
    assert_(dat.dtype.type is np.uint8)

    # 测试 np.array 函数对于具有 array interface 的对象的行为
    a = np.array(100.0, dtype=np.float64)
    o = type("o", (object,),
             dict(__array_interface__=a.__array_interface__))
    assert_equal(np.array(o, dtype=np.float64), a)

    # 测试 np.array 函数对于具有 array_struct interface 的对象的行为
    a = np.array([(1, 4.0, 'Hello'), (2, 6.0, 'World')],
                 dtype=[('f0', int), ('f1', float), ('f2', str)])
    o = type("o", (object,),
             dict(__array_struct__=a.__array_struct__))
    # 对比两个对象 o 和 a 的字节表示是否相等
    assert_equal(bytes(np.array(o).data), bytes(a.data))

    # 定义一个自定义的 __array__ 方法，返回一个包含单个元素 100.0 的 numpy 数组
    def custom__array__(self, dtype=None, copy=None):
        return np.array(100.0, dtype=dtype, copy=copy)

    # 创建一个名为 o 的类实例，该实例包含自定义的 __array__ 方法
    o = type("o", (object,), dict(__array__=custom__array__))()

    # 断言使用 np.float64 类型创建 o 的 numpy 数组与给定的 np.float64 数组相等
    assert_equal(np.array(o, dtype=np.float64), np.array(100.0, np.float64))

    # 初始化一个嵌套列表 nested，包含 1.5，循环添加维度直到达到 ncu.MAXDIMS 上限
    nested = 1.5
    for i in range(ncu.MAXDIMS):
        nested = [nested]

    # 尝试将 nested 转换为 numpy 数组，不应该引发错误
    np.array(nested)

    # 当尝试使用包含嵌套列表 nested 的列表创建 numpy 数组时，应该引发 ValueError
    assert_raises(ValueError, np.array, [nested], dtype=np.float64)

    # 使用 None 初始化的列表，期望结果为包含 np.nan 的 numpy 数组
    assert_equal(np.array([None] * 10, dtype=np.float32),
                 np.full((10,), np.nan, dtype=np.float32))
    assert_equal(np.array([[None]] * 10, dtype=np.float32),
                 np.full((10, 1), np.nan, dtype=np.float32))
    assert_equal(np.array([[None] * 10], dtype=np.float32),
                 np.full((1, 10), np.nan, dtype=np.float32))
    assert_equal(np.array([[None] * 10] * 10, dtype=np.float32),
                 np.full((10, 10), np.nan, dtype=np.float32))

    # 使用 None 初始化的列表，期望结果为包含 np.nan 的 numpy 数组，但数据类型为 np.float64
    assert_equal(np.array([None] * 10, dtype=np.float64),
                 np.full((10,), np.nan, dtype=np.float64))
    assert_equal(np.array([[None]] * 10, dtype=np.float64),
                 np.full((10, 1), np.nan, dtype=np.float64))
    assert_equal(np.array([[None] * 10], dtype=np.float64),
                 np.full((1, 10), np.nan, dtype=np.float64))
    assert_equal(np.array([[None] * 10] * 10, dtype=np.float64),
                 np.full((10, 10), np.nan, dtype=np.float64))

    # 使用 1.0 初始化的列表，期望结果为包含 1.0 的 numpy 数组，数据类型为 np.float64
    assert_equal(np.array([1.0] * 10, dtype=np.float64),
                 np.ones((10,), dtype=np.float64))
    assert_equal(np.array([[1.0]] * 10, dtype=np.float64),
                 np.ones((10, 1), dtype=np.float64))
    assert_equal(np.array([[1.0] * 10], dtype=np.float64),
                 np.ones((1, 10), dtype=np.float64))
    assert_equal(np.array([[1.0] * 10] * 10, dtype=np.float64),
                 np.ones((10, 10), dtype=np.float64))

    # 使用 None 或 1.0 初始化的元组，期望结果与相应的列表初始化相同
    assert_equal(np.array((None,) * 10, dtype=np.float64),
                 np.full((10,), np.nan, dtype=np.float64))
    assert_equal(np.array([(None,)] * 10, dtype=np.float64),
                 np.full((10, 1), np.nan, dtype=np.float64))
    assert_equal(np.array([(None,) * 10], dtype=np.float64),
                 np.full((1, 10), np.nan, dtype=np.float64))
    assert_equal(np.array([(None,) * 10] * 10, dtype=np.float64),
                 np.full((10, 10), np.nan, dtype=np.float64))

    assert_equal(np.array((1.0,) * 10, dtype=np.float64),
                 np.ones((10,), dtype=np.float64))
    assert_equal(np.array([(1.0,)] * 10, dtype=np.float64),
                 np.ones((10, 1), dtype=np.float64))
    # 断言：验证生成的 NumPy 数组是否与预期的一维全 1 数组相等，数据类型为 np.float64
    assert_equal(np.array([(1.0,) * 10], dtype=np.float64),
                 np.ones((1, 10), dtype=np.float64))
    
    # 断言：验证生成的 NumPy 数组是否与预期的二维全 1 数组相等，数据类型为 np.float64
    assert_equal(np.array([(1.0,) * 10] * 10, dtype=np.float64),
                 np.ones((10, 10), dtype=np.float64))
# 使用 pytest.mark.parametrize 装饰器，定义一个参数化测试函数，测试不同情况下的数组转换行为
@pytest.mark.parametrize("array", [True, False])
def test_array_impossible_casts(array):
    # 所有内置类型理论上都可以强制转换，但用户自定义类型不一定能够。
    # 创建一个 rational 类型对象 rt，分子为 1，分母为 2
    rt = rational(1, 2)
    # 如果 array 为 True，则将 rt 转换为 NumPy 数组
    if array:
        rt = np.array(rt)
    # 使用 assert_raises 检查 TypeError 是否被抛出
    with assert_raises(TypeError):
        np.array(rt, dtype="M8")


# 定义一个测试函数，测试 NumPy 数组的 astype 方法
def test_array_astype():
    # 创建一个包含 0 到 5 的浮点数数组 a，形状为 2x3
    a = np.arange(6, dtype='f4').reshape(2, 3)
    # 默认行为：允许不安全的类型转换，保持内存布局，总是复制。
    # 将数组 a 转换为整数类型 'i4'
    b = a.astype('i4')
    assert_equal(a, b)
    assert_equal(b.dtype, np.dtype('i4'))
    assert_equal(a.strides, b.strides)
    
    # 将数组 a 转置后再转换为整数类型 'i4'
    b = a.T.astype('i4')
    assert_equal(a.T, b)
    assert_equal(b.dtype, np.dtype('i4'))
    assert_equal(a.T.strides, b.strides)
    
    # 将数组 a 转换回浮点数类型 'f4'
    b = a.astype('f4')
    assert_equal(a, b)
    assert_(not (a is b))

    # copy=False 参数跳过复制操作
    b = a.astype('f4', copy=False)
    assert_(a is b)

    # order 参数允许覆盖内存布局，如果布局错误则强制复制
    b = a.astype('f4', order='F', copy=False)
    assert_equal(a, b)
    assert_(not (a is b))
    assert_(b.flags.f_contiguous)

    b = a.astype('f4', order='C', copy=False)
    assert_equal(a, b)
    assert_(a is b)
    assert_(b.flags.c_contiguous)

    # casting 参数允许捕获不良类型转换
    b = a.astype('c8', casting='safe')
    assert_equal(a, b)
    assert_equal(b.dtype, np.dtype('c8'))

    assert_raises(TypeError, a.astype, 'i4', casting='safe')

    # subok=False 参数传递非子类化数组
    b = a.astype('f4', subok=0, copy=False)
    assert_(a is b)

    # 定义一个名为 MyNDArray 的子类
    class MyNDArray(np.ndarray):
        pass

    a = np.array([[0, 1, 2], [3, 4, 5]], dtype='f4').view(MyNDArray)

    # subok=True 参数传递子类化数组
    b = a.astype('f4', subok=True, copy=False)
    assert_(a is b)

    # subok=True 是默认行为，在类型转换时创建子类型
    b = a.astype('i4', copy=False)
    assert_equal(a, b)
    assert_equal(type(b), MyNDArray)

    # subok=False 永远不会返回子类化数组
    b = a.astype('f4', subok=False, copy=False)
    assert_equal(a, b)
    assert_(not (a is b))
    assert_(type(b) is not MyNDArray)

    # 确保从字符串对象转换为固定长度字符串时不会截断。
    a = np.array([b'a'*100], dtype='O')
    b = a.astype('S')
    assert_equal(a, b)
    assert_equal(b.dtype, np.dtype('S100'))
    a = np.array(['a'*100], dtype='O')
    b = a.astype('U')
    assert_equal(a, b)
    assert_equal(b.dtype, np.dtype('U100'))

    # 对于长度小于 64 个字符的字符串，进行相同的测试
    a = np.array([b'a'*10], dtype='O')
    b = a.astype('S')
    assert_equal(a, b)
    assert_equal(b.dtype, np.dtype('S10'))
    a = np.array(['a'*10], dtype='O')
    b = a.astype('U')
    assert_equal(a, b)
    assert_equal(b.dtype, np.dtype('U10'))

    # 将整数对象转换为字符串类型的固定长度字符串
    a = np.array(123456789012345678901234567890, dtype='O').astype('S')
    # 使用 numpy.testing.assert_array_equal 函数比较数组 a 和给定的数组是否相等
    assert_array_equal(a, np.array(b'1234567890' * 3, dtype='S30'))
    
    # 创建一个包含很长整数的 numpy 数组 a，使用对象类型 'O'，然后将其转换为 Unicode 字符串类型 'U'
    a = np.array(123456789012345678901234567890, dtype='O').astype('U')
    # 使用 numpy.testing.assert_array_equal 函数比较数组 a 和给定的 Unicode 字符串数组是否相等
    assert_array_equal(a, np.array('1234567890' * 3, dtype='U30'))
    
    # 创建一个包含很长整数的 numpy 数组，使用对象类型 'O'，然后将其转换为字节字符串类型 'S'
    a = np.array([123456789012345678901234567890], dtype='O').astype('S')
    # 使用 numpy.testing.assert_array_equal 函数比较数组 a 和给定的字节字符串数组是否相等
    assert_array_equal(a, np.array(b'1234567890' * 3, dtype='S30'))
    
    # 创建一个很长整数的 numpy 数组，使用字符串类型 'S'
    a = np.array(123456789012345678901234567890, dtype='S')
    # 使用 numpy.testing.assert_array_equal 函数比较数组 a 和给定的字节字符串数组是否相等
    assert_array_equal(a, np.array(b'1234567890' * 3, dtype='S30'))
    
    # 创建一个 Unicode 字符串数组 a，包含特定的 Unicode 字符
    a = np.array('a\u0140', dtype='U')
    # 使用 numpy.ndarray 构造函数创建一个新的数组 b，从给定的缓冲区 a 中创建，数据类型为 uint32，形状为 (2,)
    b = np.ndarray(buffer=a, dtype='uint32', shape=2)
    # 使用 assert_ 断言检查数组 b 的大小是否为 2
    assert_(b.size == 2)
    
    # 创建一个整数数组 a，包含单个元素 1000，数据类型为 'i4'
    a = np.array([1000], dtype='i4')
    # 使用 assert_raises 断言检查尝试将数组 a 的元素转换为 'S1' 类型时是否引发了 TypeError 异常
    assert_raises(TypeError, a.astype, 'S1', casting='safe')
    
    # 创建一个整数数组 a，包含单个元素 1000，数据类型为 'i4'
    a = np.array(1000, dtype='i4')
    # 使用 assert_raises 断言检查尝试将数组 a 转换为 'U1' 类型时是否引发了 TypeError 异常
    assert_raises(TypeError, a.astype, 'U1', casting='safe')
    
    # 对于问题 gh-24023，使用 assert_raises 断言检查调用 a.astype() 函数时是否引发了 TypeError 异常
    assert_raises(TypeError, a.astype)
# 使用 pytest.mark.parametrize 装饰器为 test_array_astype_to_string_discovery_empty 函数参数化，参数为字符串 "S" 和 "U"
@pytest.mark.parametrize("dt", ["S", "U"])
def test_array_astype_to_string_discovery_empty(dt):
    # 引用 GitHub issue 19085
    arr = np.array([""], dtype=object)
    # 注意，itemsize 是 `0 -> 1` 的逻辑，应该会改变。
    # 重要的是测试不报错。
    assert arr.astype(dt).dtype.itemsize == np.dtype(f"{dt}1").itemsize

    # 使用 np.can_cast 检查相同的情况（因为它接受数组）
    assert np.can_cast(arr, dt, casting="unsafe")
    assert not np.can_cast(arr, dt, casting="same_kind")
    # 同样也适用于对象作为描述符：
    assert np.can_cast("O", dt, casting="unsafe")

# 使用 pytest.mark.parametrize 装饰器为 test_array_astype_to_void 函数参数化，参数为字符串 "d", "f", "S13", "U32"
@pytest.mark.parametrize("dt", ["d", "f", "S13", "U32"])
def test_array_astype_to_void(dt):
    dt = np.dtype(dt)
    arr = np.array([], dtype=dt)
    assert arr.astype("V").dtype.itemsize == dt.itemsize

# test_object_array_astype_to_void 函数不接受参数，测试对象数组转换为 void 类型
def test_object_array_astype_to_void():
    # 与 `test_array_astype_to_void` 不同，因为对象数组被检查。
    # 默认的 void 是 "V8"（8 是 double 的长度）
    arr = np.array([], dtype="O").astype("V")
    assert arr.dtype == "V8"

# 使用 pytest.mark.parametrize 装饰器为 test_array_astype_warning 函数参数化，参数为 np._core.sctypes['uint'] + np._core.sctypes['int'] + np._core.sctypes['float'] 的各个类型
@pytest.mark.parametrize("t", np._core.sctypes['uint'] + np._core.sctypes['int'] + np._core.sctypes['float'])
def test_array_astype_warning(t):
    # 测试从复数到浮点数或整数的转换时的 ComplexWarning
    a = np.array(10, dtype=np.complex128)
    assert_warns(np.exceptions.ComplexWarning, a.astype, t)

# 使用 pytest.mark.parametrize 装饰器为 test_string_to_boolean_cast 函数参数化，参数为不同的 dtype 和 out_dtype 组合
@pytest.mark.parametrize(["dtype", "out_dtype"],
        [(np.bytes_, np.bool),
         (np.str_, np.bool),
         (np.dtype("S10,S9"), np.dtype("?,?")),
         (np.dtype("S7,U9"), np.dtype("?,?"))])
def test_string_to_boolean_cast(dtype, out_dtype):
    # 只有最后两个（空）字符串是假值（`\0` 被剥离）：
    arr = np.array(
            ["10", "10\0\0\0", "0\0\0", "0", "False", " ", "", "\0"],
            dtype=dtype)
    expected = np.array(
            [True, True, True, True, True, True, False, False],
            dtype=out_dtype)
    assert_array_equal(arr.astype(out_dtype), expected)
    # 因为相似，检查非零行为是否相同（结构体如果所有条目都不为零）
    assert_array_equal(np.nonzero(arr), np.nonzero(expected))

# 使用 pytest.mark.parametrize 装饰器为 test_string_to_complex_cast 函数参数化，参数为不同的 str_type 和 scalar_type 组合
@pytest.mark.parametrize("str_type", [str, bytes, np.str_])
@pytest.mark.parametrize("scalar_type",
        [np.complex64, np.complex128, np.clongdouble])
def test_string_to_complex_cast(str_type, scalar_type):
    value = scalar_type(b"1+3j")
    assert scalar_type(value) == 1+3j
    assert np.array([value], dtype=object).astype(scalar_type)[()] == 1+3j
    assert np.array(value).astype(scalar_type)[()] == 1+3j
    arr = np.zeros(1, dtype=scalar_type)
    arr[0] = value
    assert arr[0] == 1+3j

# 使用 pytest.mark.parametrize 装饰器为 test_none_to_nan_cast 函数参数化，参数为 np.typecodes["AllFloat"] 中的各种 dtype
@pytest.mark.parametrize("dtype", np.typecodes["AllFloat"])
def test_none_to_nan_cast(dtype):
    # 注意，在编写此测试时，标量构造函数拒绝 None
    # 创建一个包含一个元素的 NumPy 数组，元素值为 0.0，数据类型为参数 dtype 指定的类型
    arr = np.zeros(1, dtype=dtype)
    # 将数组中第一个元素赋值为 None
    arr[0] = None
    # 使用 NumPy 的函数检查数组中第一个元素是否为 NaN（Not a Number）
    assert np.isnan(arr)[0]
    # 使用 NumPy 创建一个包含单个元素为 None 的数组，并检查该元素是否为 NaN
    assert np.isnan(np.array(None, dtype=dtype))[()]
    # 使用 NumPy 创建一个包含单个元素为 None 的数组，并检查第一个元素是否为 NaN
    assert np.isnan(np.array([None], dtype=dtype))[0]
    # 将 None 转换为 dtype 指定的数据类型后，使用 NumPy 检查该元素是否为 NaN
    assert np.isnan(np.array(None).astype(dtype))[()]
# 定义一个测试函数，用于测试 np.copyto() 函数的不同用法和参数组合
def test_copyto_fromscalar():
    # 创建一个包含6个单精度浮点数的 NumPy 数组，形状为 (2, 3)
    a = np.arange(6, dtype='f4').reshape(2, 3)

    # 简单的复制操作，将标量值 1.5 复制到数组 a 中的所有元素
    np.copyto(a, 1.5)
    # 断言数组 a 中的所有元素是否都等于 1.5
    assert_equal(a, 1.5)

    # 将标量值 2.5 复制到数组 a 的转置矩阵中的所有元素
    np.copyto(a.T, 2.5)
    # 断言数组 a 中的所有元素是否都等于 2.5
    assert_equal(a, 2.5)

    # 使用掩码进行复制操作，将标量值 3.5 复制到数组 a 中的特定位置
    mask = np.array([[0, 1, 0], [0, 0, 1]], dtype='?')
    np.copyto(a, 3.5, where=mask)
    # 断言数组 a 中的元素是否按预期进行了复制
    assert_equal(a, [[2.5, 3.5, 2.5], [2.5, 2.5, 3.5]])

    # 再次使用不同的掩码进行复制操作，将标量值 4.5 复制到数组 a 的转置矩阵中的特定位置
    mask = np.array([[0, 1], [1, 1], [1, 0]], dtype='?')
    np.copyto(a.T, 4.5, where=mask)
    # 断言数组 a 中的元素是否按预期进行了复制
    assert_equal(a, [[2.5, 4.5, 4.5], [4.5, 4.5, 3.5]])

# 定义另一个测试函数，用于测试 np.copyto() 的更多用法和参数组合
def test_copyto():
    # 创建一个包含6个32位整数的 NumPy 数组，形状为 (2, 3)
    a = np.arange(6, dtype='i4').reshape(2, 3)

    # 简单的复制操作，将指定数组的值复制到数组 a 中
    np.copyto(a, [[3, 1, 5], [6, 2, 1]])
    # 断言数组 a 中的元素是否按预期进行了复制
    assert_equal(a, [[3, 1, 5], [6, 2, 1]])

    # 进行重叠复制，将数组 a 的部分区域复制到另一部分区域
    np.copyto(a[:, :2], a[::-1, 1::-1])
    # 断言数组 a 中的元素是否按预期进行了复制
    assert_equal(a, [[2, 6, 5], [1, 3, 1]])

    # 测试默认的类型转换规则，预期会引发 TypeError 异常
    assert_raises(TypeError, np.copyto, a, 1.5)

    # 使用 'unsafe' 转换模式进行复制，将标量值 1.5 复制到数组 a 中
    np.copyto(a, 1.5, casting='unsafe')
    # 断言数组 a 中的元素是否按预期进行了复制，由于 'unsafe' 模式，1.5 被截断为 1
    assert_equal(a, 1)

    # 使用掩码进行复制操作，将标量值 3 复制到数组 a 中的特定位置
    np.copyto(a, 3, where=[True, False, True])
    # 断言数组 a 中的元素是否按预期进行了复制
    assert_equal(a, [[3, 1, 3], [3, 1, 3]])

    # 测试在使用掩码时的类型转换规则，预期会引发 TypeError 异常
    assert_raises(TypeError, np.copyto, a, 3.5, where=[True, False, True])

    # 使用列表形式的掩码进行复制操作，将标量值 4.0 复制到数组 a 中的特定位置
    np.copyto(a, 4.0, casting='unsafe', where=[[0, 1, 1], [1, 0, 0]])
    # 断言数组 a 中的元素是否按预期进行了复制
    assert_equal(a, [[3, 4, 4], [4, 1, 3]])

    # 再次进行重叠复制，结合掩码进行复制操作
    np.copyto(a[:, :2], a[::-1, 1::-1], where=[[0, 1], [1, 1]])
    # 断言数组 a 中的元素是否按预期进行了复制
    assert_equal(a, [[3, 4, 4], [4, 3, 3]])

    # 测试 'dst' 参数必须是一个数组，预期会引发 TypeError 异常
    assert_raises(TypeError, np.copyto, [1, 2, 3], [2, 3, 4])

# 定义另一个测试函数，用于测试特定复制场景下的 np.copyto() 行为
def test_copyto_permut():
    # 设置一个较大的填充数值
    pad = 500
    # 创建一个布尔类型的列表，大部分为 True，最后四个为 True
    l = [True] * pad + [True, True, True, True]
    # 创建一个长度为 l 去掉前 pad 部分的 0 数组
    r = np.zeros(len(l)-pad)
    # 创建一个长度为 l 去掉前 pad 部分的 1 数组
    d = np.ones(len(l)-pad)
    # 根据反向掩码创建一个数组
    mask = np.array(l)[pad:]
    # 使用掩码对数组 r 进行复制，以便数组 d 的值

    # 复制到 r 的末尾
    np.copyto(r, d, where=mask[::-1])

    # 测试所有可能掩码的排列组合
    power = 9
    d = np.ones(power)
    for i in range(2**power):
        # 创建一个长度为 power 的全零数组 r
        r = np.zeros(power)
        # 生成掩码列表 l，用于判断每个位是否为1
        l = [(i & x) != 0 for x in range(power)]
        # 将 l 转换为布尔数组 mask
        mask = np.array(l)
        # 根据 mask 复制数组 d 的部分元素到 r 中
        np.copyto(r, d, where=mask)
        # 断言 r 中的元素是否与 l 相等（应为布尔值 True/False）
        assert_array_equal(r == 1, l)
        # 断言 r 中的元素之和是否等于 l 中值为 True 的数量
        assert_equal(r.sum(), sum(l))

        # 将 r 重新初始化为全零数组
        r = np.zeros(power)
        # 根据 mask 的反转复制数组 d 的部分元素到 r 中
        np.copyto(r, d, where=mask[::-1])
        # 断言 r 中的元素是否与 l 的反转相等
        assert_array_equal(r == 1, l[::-1])
        # 断言 r 中的元素之和是否等于 l 反转后值为 True 的数量
        assert_equal(r.sum(), sum(l))

        # 将 r 重新初始化为全零数组
        r = np.zeros(power)
        # 仅复制偶数索引位置的数组 d 的部分元素到 r 中，根据 mask 的条件
        np.copyto(r[::2], d[::2], where=mask[::2])
        # 断言 r 的偶数索引位置的元素是否与 l 的偶数索引位置相等
        assert_array_equal(r[::2] == 1, l[::2])
        # 断言 r 的偶数索引位置的元素之和是否等于 l 的偶数索引位置值为 True 的数量
        assert_equal(r[::2].sum(), sum(l[::2]))

        # 将 r 重新初始化为全零数组
        r = np.zeros(power)
        # 仅复制偶数索引位置的数组 d 的部分元素到 r 中，根据 mask 反转的条件
        np.copyto(r[::2], d[::2], where=mask[::-2])
        # 断言 r 的偶数索引位置的元素是否与 l 的偶数索引位置反转后相等
        assert_array_equal(r[::2] == 1, l[::-2])
        # 断言 r 的偶数索引位置的元素之和是否等于 l 的偶数索引位置反转后值为 True 的数量
        assert_equal(r[::2].sum(), sum(l[::-2]))

        # 遍历每个常量 c，针对 r 进行处理
        for c in [0xFF, 0x7F, 0x02, 0x10]:
            # 将 r 重新初始化为全零数组
            r = np.zeros(power)
            # 将 mask 转换为整数数组 imask，并根据 mask 的值设置对应位置为常量 c
            mask = np.array(l)
            imask = np.array(l).view(np.uint8)
            imask[mask != 0] = c
            # 根据 mask 复制数组 d 的部分元素到 r 中
            np.copyto(r, d, where=mask)
            # 断言 r 中的元素是否与 l 相等
            assert_array_equal(r == 1, l)
            # 断言 r 中的元素之和是否等于 l 中值为 True 的数量
            assert_equal(r.sum(), sum(l))

    # 最后的处理步骤
    r = np.zeros(power)
    # 将数组 d 的所有元素复制到 r 中
    np.copyto(r, d, where=True)
    # 断言 r 中的元素之和是否等于 r 的大小（即所有元素的数量）
    assert_equal(r.sum(), r.size)
    # 将数组 r 初始化为全一数组，数组 d 初始化为全零数组
    r = np.ones(power)
    d = np.zeros(power)
    # 将数组 d 的所有元素复制到数组 r 中（但不改变 r 中为1的位置）
    np.copyto(r, d, where=False)
    # 断言 r 中的元素之和是否等于 r 的大小
    assert_equal(r.sum(), r.size)
# 定义一个测试函数，用于验证数组拷贝操作的顺序性
def test_copy_order():
    # 创建一个形状为 (2, 1, 3, 4) 的数组 a，填充从 0 到 23 的连续整数
    a = np.arange(24).reshape(2, 1, 3, 4)
    # 使用 order='F' 参数按列主序拷贝数组 a，得到数组 b
    b = a.copy(order='F')
    # 创建一个形状为 (2, 1, 4, 3) 的数组 c，填充从 0 到 23 的连续整数，并交换轴 2 和 3
    c = np.arange(24).reshape(2, 1, 4, 3).swapaxes(2, 3)

    # 定义一个内部函数，用于检查拷贝结果的连续性
    def check_copy_result(x, y, ccontig, fcontig, strides=False):
        # 断言 x 和 y 不是同一个对象
        assert_(not (x is y))
        # 断言 x 和 y 的内容相等
        assert_equal(x, y)
        # 断言 x 的标志中 C 连续性为 ccontig
        assert_equal(x.flags.c_contiguous, ccontig)
        # 断言 x 的标志中 F 连续性为 fcontig
        assert_equal(x.flags.f_contiguous, fcontig)

    # 验证数组 a、b 和 c 的初始状态
    assert_(a.flags.c_contiguous)
    assert_(not a.flags.f_contiguous)
    assert_(not b.flags.c_contiguous)
    assert_(b.flags.f_contiguous)
    assert_(not c.flags.c_contiguous)
    assert_(not c.flags.f_contiguous)

    # 使用 order='C' 参数进行拷贝
    res = a.copy(order='C')
    check_copy_result(res, a, ccontig=True, fcontig=False, strides=True)
    res = b.copy(order='C')
    check_copy_result(res, b, ccontig=True, fcontig=False, strides=False)
    res = c.copy(order='C')
    check_copy_result(res, c, ccontig=True, fcontig=False, strides=False)
    res = np.copy(a, order='C')
    check_copy_result(res, a, ccontig=True, fcontig=False, strides=True)
    res = np.copy(b, order='C')
    check_copy_result(res, b, ccontig=True, fcontig=False, strides=False)
    res = np.copy(c, order='C')
    check_copy_result(res, c, ccontig=True, fcontig=False, strides=False)

    # 使用 order='F' 参数进行拷贝
    res = a.copy(order='F')
    check_copy_result(res, a, ccontig=False, fcontig=True, strides=False)
    res = b.copy(order='F')
    check_copy_result(res, b, ccontig=False, fcontig=True, strides=True)
    res = c.copy(order='F')
    check_copy_result(res, c, ccontig=False, fcontig=True, strides=False)
    res = np.copy(a, order='F')
    check_copy_result(res, a, ccontig=False, fcontig=True, strides=False)
    res = np.copy(b, order='F')
    check_copy_result(res, b, ccontig=False, fcontig=True, strides=True)
    res = np.copy(c, order='F')
    check_copy_result(res, c, ccontig=False, fcontig=True, strides=False)

    # 使用 order='K' 参数进行拷贝
    res = a.copy(order='K')
    check_copy_result(res, a, ccontig=True, fcontig=False, strides=True)
    res = b.copy(order='K')
    check_copy_result(res, b, ccontig=False, fcontig=True, strides=True)
    res = c.copy(order='K')
    check_copy_result(res, c, ccontig=False, fcontig=False, strides=True)
    res = np.copy(a, order='K')
    check_copy_result(res, a, ccontig=True, fcontig=False, strides=True)
    res = np.copy(b, order='K')
    check_copy_result(res, b, ccontig=False, fcontig=True, strides=True)
    res = np.copy(c, order='K')
    check_copy_result(res, c, ccontig=False, fcontig=False, strides=True)

# 定义一个测试函数，用于验证数组的连续性标志
def test_contiguous_flags():
    # 创建一个形状为 (4, 4, 1) 的数组 a，所有元素为 1，并按条件切片
    a = np.ones((4, 4, 1))[::2,:,:]
    # 修改数组 a 的步长
    a.strides = a.strides[:2] + (-123,)
    # 创建一个形状为 (2, 2, 1, 2, 2) 的数组 b，并交换轴 3 和 4
    b = np.ones((2, 2, 1, 2, 2)).swapaxes(3, 4)

    # 定义一个内部函数，用于检查数组的连续性标志
    def check_contig(a, ccontig, fcontig):
        # 断言数组 a 的 C 连续性标志为 ccontig
        assert_(a.flags.c_contiguous == ccontig)
        # 断言数组 a 的 F 连续性标志为 fcontig
        assert_(a.flags.f_contiguous == fcontig)

    # 检查新数组的连续性
    check_contig(a, False, False)
    check_contig(b, False, False)
    # 调用 check_contig 函数，检查空的4维数组是否连续，并要求检查数组的内容是否被复制
    check_contig(np.empty((2, 2, 0, 2, 2)), True, True)
    # 调用 check_contig 函数，检查按列主序存储的数组是否连续，并要求检查数组的内容是否被复制
    check_contig(np.array([[[1], [2]]], order='F'), True, True)
    # 调用 check_contig 函数，检查空的2维数组是否连续，并要求检查数组的内容是否被复制
    check_contig(np.empty((2, 2)), True, False)
    # 调用 check_contig 函数，检查按列主序存储的空的2维数组是否连续，并要求检查数组的内容是否被复制
    check_contig(np.empty((2, 2), order='F'), False, True)

    # 检查 np.array 创建数组时连续标志的正确性：
    check_contig(np.array(a, copy=None), False, False)
    # 检查 np.array 按行主序存储创建数组时连续标志的正确性，并要求不复制数组内容
    check_contig(np.array(a, copy=None, order='C'), True, False)
    # 检查 np.array 创建4维数组时按列主序存储的连续性标志的正确性，并要求不复制数组内容
    check_contig(np.array(a, ndmin=4, copy=None, order='F'), False, True)

    # 检查切片更新连续性标志：
    check_contig(a[0], True, True)
    # 检查切片更新连续性标志，包括步进为4和省略符号
    check_contig(a[None, ::4, ..., None], True, True)
    # 检查切片更新连续性标志，仅限第一个维度的索引为0，其余维度为省略符号
    check_contig(b[0, 0, ...], False, True)
    # 检查切片更新连续性标志，仅限第三维度的切片范围为0:0，其余维度全取
    check_contig(b[:, :, 0:0, :, :], True, True)

    # 测试 ravel 和 squeeze 方法
    # 检查数组通过 ravel 方法展平后的连续性标志
    check_contig(a.ravel(), True, True)
    # 检查通过 squeeze 方法去除维数为1的数组的连续性标志
    check_contig(np.ones((1, 3, 1)).squeeze(), True, True)
# 定义测试函数，用于测试 np.broadcast_arrays() 函数的功能
def test_broadcast_arrays():
    # 创建具有用户定义数据类型 'u4,u4,u4' 的 numpy 数组 a 和 b
    a = np.array([(1, 2, 3)], dtype='u4,u4,u4')
    b = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)], dtype='u4,u4,u4')
    # 对数组 a 和 b 进行广播，得到广播后的结果
    result = np.broadcast_arrays(a, b)
    # 断言第一个广播数组的结果与预期结果相等
    assert_equal(result[0], np.array([(1, 2, 3), (1, 2, 3), (1, 2, 3)], dtype='u4,u4,u4'))
    # 断言第二个广播数组的结果与预期结果相等
    assert_equal(result[1], np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)], dtype='u4,u4,u4'))

# 使用 pytest 的 parametrize 装饰器定义多组参数化测试
@pytest.mark.parametrize(["shape", "fill_value", "expected_output"],
        [((2, 2), [5.0,  6.0], np.array([[5.0, 6.0], [5.0, 6.0]])),
         ((3, 2), [1.0,  2.0], np.array([[1.0, 2.0], [1.0, 2.0], [1.0,  2.0]]))])
def test_full_from_list(shape, fill_value, expected_output):
    # 使用 np.full() 函数根据指定形状和填充值创建数组
    output = np.full(shape, fill_value)
    # 断言创建的数组与预期输出数组相等
    assert_equal(output, expected_output)

# 定义测试函数，测试 numpy 数组的 astype() 方法中的复制标志
def test_astype_copyflag():
    # 创建一个包含 0 到 9 的整数 numpy 数组 arr
    arr = np.arange(10, dtype=np.intp)

    # 测试使用 copy=True 参数进行类型转换
    res_true = arr.astype(np.intp, copy=True)
    # 断言结果数组不共享内存
    assert not np.shares_memory(arr, res_true)

    # 测试使用 copy=False 参数进行类型转换
    res_false = arr.astype(np.intp, copy=False)
    # 断言结果数组与原始数组共享内存
    assert np.shares_memory(arr, res_false)

    # 测试将整数数组转换为浮点数数组，并使用 copy=False 参数
    res_false_float = arr.astype(np.float64, copy=False)
    # 断言结果数组不共享内存
    assert not np.shares_memory(arr, res_false_float)

    # 测试不允许使用未公开的 _CopyMode 枚举值
    # 断言调用 astype() 方法时会抛出 ValueError 异常
    assert_raises(ValueError, arr.astype, np.float64,
                  copy=np._CopyMode.NEVER)
```