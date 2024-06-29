# `.\numpy\numpy\_core\tests\test_scalar_ctors.py`

```py
"""
Test the scalar constructors, which also do type-coercion
"""
# 导入 pytest 库，用于测试
import pytest

# 导入 numpy 库，并引入测试函数
import numpy as np
from numpy.testing import (
    assert_equal, assert_almost_equal, assert_warns,
    )

# 定义一个测试类 TestFromString，用于测试从字符串构造浮点数的情况
class TestFromString:
    def test_floating(self):
        # Ticket #640, floats from string
        # 使用字符串构造不同精度的浮点数
        fsingle = np.single('1.234')
        fdouble = np.double('1.234')
        flongdouble = np.longdouble('1.234')
        # 断言各个浮点数与预期值的近似程度
        assert_almost_equal(fsingle, 1.234)
        assert_almost_equal(fdouble, 1.234)
        assert_almost_equal(flongdouble, 1.234)

    def test_floating_overflow(self):
        """ Strings containing an unrepresentable float overflow """
        # 测试字符串表示的浮点数超出范围的情况
        fhalf = np.half('1e10000')
        assert_equal(fhalf, np.inf)
        fsingle = np.single('1e10000')
        assert_equal(fsingle, np.inf)
        fdouble = np.double('1e10000')
        assert_equal(fdouble, np.inf)
        # 使用 assert_warns 检查超出范围时 longdouble 的情况
        flongdouble = assert_warns(RuntimeWarning, np.longdouble, '1e10000')
        assert_equal(flongdouble, np.inf)

        # 测试负无穷大的情况
        fhalf = np.half('-1e10000')
        assert_equal(fhalf, -np.inf)
        fsingle = np.single('-1e10000')
        assert_equal(fsingle, -np.inf)
        fdouble = np.double('-1e10000')
        assert_equal(fdouble, -np.inf)
        # 使用 assert_warns 检查负无穷大时 longdouble 的情况
        flongdouble = assert_warns(RuntimeWarning, np.longdouble, '-1e10000')
        assert_equal(flongdouble, -np.inf)


# 定义一个测试类 TestExtraArgs，用于测试特殊参数情况
class TestExtraArgs:
    def test_superclass(self):
        # try both positional and keyword arguments
        # 测试 str_ 函数使用不同编码方式转换字节串为字符串
        s = np.str_(b'\x61', encoding='unicode-escape')
        assert s == 'a'
        s = np.str_(b'\x61', 'unicode-escape')
        assert s == 'a'

        # 之前的版本会返回 '\\xx'，现在应该抛出 UnicodeDecodeError
        with pytest.raises(UnicodeDecodeError):
            np.str_(b'\\xx', encoding='unicode-escape')
        with pytest.raises(UnicodeDecodeError):
            np.str_(b'\\xx', 'unicode-escape')

        # 测试 np.bytes_ 函数将整数转换为字节串
        # 先前会失败，但 numpy 现在能成功转换
        assert np.bytes_(-2) == b'-2'

    def test_datetime(self):
        # 测试 np.datetime64 函数的用法
        dt = np.datetime64('2000-01', ('M', 2))
        assert np.datetime_data(dt) == ('M', 2)

        # 检查 np.datetime64 函数使用无效参数时是否引发 TypeError
        with pytest.raises(TypeError):
            np.datetime64('2000', garbage=True)

    def test_bool(self):
        # 检查 np.bool 函数使用无效参数时是否引发 TypeError
        with pytest.raises(TypeError):
            np.bool(False, garbage=True)

    def test_void(self):
        # 检查 np.void 函数使用无效参数时是否引发 TypeError
        with pytest.raises(TypeError):
            np.void(b'test', garbage=True)


# 定义一个测试类 TestFromInt，用于测试从整数构造的情况
class TestFromInt:
    def test_intp(self):
        # Ticket #99
        # 测试 np.intp 函数的用法
        assert_equal(1024, np.intp(1024))

    def test_uint64_from_negative(self):
        # 检查 np.uint64 函数使用负数参数时是否引发 OverflowError
        with pytest.raises(OverflowError):
            np.uint64(-2)


# 定义几个类型列表，用于测试数组从标量构造的情况
int_types = [np.byte, np.short, np.intc, np.long, np.longlong]
uint_types = [np.ubyte, np.ushort, np.uintc, np.ulong, np.ulonglong]
float_types = [np.half, np.single, np.double, np.longdouble]
cfloat_types = [np.csingle, np.cdouble, np.clongdouble]


# 定义一个测试类 TestArrayFromScalar，用于测试从标量构造数组的情况
class TestArrayFromScalar:
    """ gh-15467 and gh-19125 """
    # 定义一个名为 _do_test 的方法，用于执行测试，接受 t1 和 t2 作为参数，arg 默认为 2
    def _do_test(self, t1, t2, arg=2):
        # 如果 arg 是 None，则调用 t1() 来获取结果
        if arg is None:
            x = t1()
        # 如果 arg 是一个元组，则根据 t1 的类型来调用相应的方法
        elif isinstance(arg, tuple):
            # 特殊情况：如果 t1 是 np.clongdouble，则标记测试为失败
            if t1 is np.clongdouble:
                pytest.xfail("creating a clongdouble from real and "
                             "imaginary parts isn't supported")
            x = t1(*arg)
        # 否则，使用 arg 作为参数调用 t1
        else:
            x = t1(arg)
        # 将结果转换为 numpy 数组，使用 t2 指定的数据类型
        arr = np.array(x, dtype=t2)
        # 断言：确保数据类型被精确地保留
        # 如果 t2 是 None，则检查数组的数据类型是否与 t1 的类型相同
        if t2 is None:
            assert arr.dtype.type is t1
        # 否则，检查数组的数据类型是否与 t2 相同
        else:
            assert arr.dtype.type is t2
    
    # 使用 pytest 的参数化装饰器，对整数类型进行测试
    @pytest.mark.parametrize('t1', int_types + uint_types)
    @pytest.mark.parametrize('t2', int_types + uint_types + [None])
    def test_integers(self, t1, t2):
        return self._do_test(t1, t2)
    
    # 使用 pytest 的参数化装饰器，对浮点数类型进行测试
    @pytest.mark.parametrize('t1', float_types)
    @pytest.mark.parametrize('t2', float_types + [None])
    def test_reals(self, t1, t2):
        return self._do_test(t1, t2)
    
    # 使用 pytest 的参数化装饰器，对复数类型进行测试，同时参数化 arg 参数
    @pytest.mark.parametrize('t1', cfloat_types)
    @pytest.mark.parametrize('t2', cfloat_types + [None])
    @pytest.mark.parametrize('arg', [2, 1 + 3j, (1, 2), None])
    def test_complex(self, t1, t2, arg):
        self._do_test(t1, t2, arg)
    
    # 使用 pytest 的参数化装饰器，对复数类型进行异常情况测试
    @pytest.mark.parametrize('t', cfloat_types)
    def test_complex_errors(self, t):
        # 断言：应该抛出 TypeError 异常
        with pytest.raises(TypeError):
            t(1j, 1j)
        with pytest.raises(TypeError):
            t(1, None)
        with pytest.raises(TypeError):
            t(None, 1)
@pytest.mark.parametrize("length",
        [5, np.int8(5), np.array(5, dtype=np.uint16)])
def test_void_via_length(length):
    # 使用给定的长度创建一个空的 np.void 对象
    res = np.void(length)
    # 断言返回结果的类型是 np.void
    assert type(res) is np.void
    # 断言返回结果的 item 是由长度个字节零组成的 bytes 对象
    assert res.item() == b"\0" * 5
    # 断言返回结果的数据类型是 "V5"
    assert res.dtype == "V5"

@pytest.mark.parametrize("bytes_",
        [b"spam", np.array(567.)])
def test_void_from_byteslike(bytes_):
    # 使用给定的字节数据创建一个 np.void 对象
    res = np.void(bytes_)
    expected = bytes(bytes_)
    # 断言返回结果的类型是 np.void
    assert type(res) is np.void
    # 断言返回结果的 item 与输入的 bytes 对象相同
    assert res.item() == expected

    # 通过传递 dtype 参数可以扩展 np.void 对象的长度
    res = np.void(bytes_, dtype="V100")
    assert type(res) is np.void
    # 断言返回结果的前部分与输入的 bytes 对象相同
    assert res.item()[:len(expected)] == expected
    # 断言返回结果的后部分是用零填充的，以使总长度为 res.nbytes
    assert res.item()[len(expected):] == b"\0" * (res.nbytes - len(expected))
    
    # 通过传递 dtype 参数可以缩短 np.void 对象的长度
    res = np.void(bytes_, dtype="V4")
    assert type(res) is np.void
    # 断言返回结果只包含输入 bytes 对象的前 4 个字节
    assert res.item() == expected[:4]

def test_void_arraylike_trumps_byteslike():
    # memoryview 对象被转换为一个形状为 (18,) 的数组，而不是一个长度为 18 的单个 bytes-like 对象。
    m = memoryview(b"just one mintleaf?")
    res = np.void(m)
    assert type(res) is np.ndarray
    # 断言返回结果的数据类型是 "V1"
    assert res.dtype == "V1"
    # 断言返回结果的形状是 (18,)
    assert res.shape == (18,)

def test_void_dtype_arg():
    # 测试 dtype 参数的基本用法（位置参数和关键字参数）
    res = np.void((1, 2), dtype="i,i")
    # 断言返回结果的 item 等于输入的元组 (1, 2)
    assert res.item() == (1, 2)
    res = np.void((2, 3), "i,i")
    # 断言返回结果的 item 等于输入的元组 (2, 3)
    assert res.item() == (2, 3)

@pytest.mark.parametrize("data",
        [5, np.int8(5), np.array(5, dtype=np.uint16)])
def test_void_from_integer_with_dtype(data):
    # "length" 的含义被忽略，使用 data 的值：
    res = np.void(data, dtype="i,i")
    assert type(res) is np.void
    # 断言返回结果的数据类型是 "i,i"
    assert res.dtype == "i,i"
    # 断言返回结果的字段 "f0" 和 "f1" 分别等于 data 的值
    assert res["f0"] == 5 and res["f1"] == 5

def test_void_from_structure():
    # 创建一个复合 dtype
    dtype = np.dtype([('s', [('f', 'f8'), ('u', 'U1')]), ('i', 'i2')])
    data = np.array(((1., 'a'), 2), dtype=dtype)
    # 使用复合数据类型创建 np.void 对象
    res = np.void(data[()], dtype=dtype)
    assert type(res) is np.void
    # 断言返回结果的数据类型与输入数据的 dtype 相同
    assert res.dtype == dtype
    # 断言返回结果与输入数据相同
    assert res == data[()]

def test_void_bad_dtype():
    with pytest.raises(TypeError,
            match="void: descr must be a `void.*int64"):
        # 传递不合法的 dtype 引发 TypeError 异常
        np.void(4, dtype="i8")

    # 不合法的子数组 dtype (形状为 `(4,)`) 被拒绝
    with pytest.raises(TypeError,
            match=r"void: descr must be a `void.*\(4,\)"):
        np.void(4, dtype="4i")
```