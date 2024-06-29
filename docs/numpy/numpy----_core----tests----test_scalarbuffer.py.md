# `.\numpy\numpy\_core\tests\test_scalarbuffer.py`

```py
"""
Test scalar buffer interface adheres to PEP 3118
"""
# 导入必要的库和模块
import numpy as np
from numpy._core._rational_tests import rational
from numpy._core._multiarray_tests import get_buffer_info
import pytest

from numpy.testing import assert_, assert_equal, assert_raises

# PEP3118 格式字符串，用于本机类型（标准对齐和字节顺序）
scalars_and_codes = [
    (np.bool, '?'),
    (np.byte, 'b'),
    (np.short, 'h'),
    (np.intc, 'i'),
    (np.long, 'l'),
    (np.longlong, 'q'),
    (np.ubyte, 'B'),
    (np.ushort, 'H'),
    (np.uintc, 'I'),
    (np.ulong, 'L'),
    (np.ulonglong, 'Q'),
    (np.half, 'e'),
    (np.single, 'f'),
    (np.double, 'd'),
    (np.longdouble, 'g'),
    (np.csingle, 'Zf'),
    (np.cdouble, 'Zd'),
    (np.clongdouble, 'Zg'),
]
scalars_only, codes_only = zip(*scalars_and_codes)

# 测试用例类，用于测试标量符合 PEP3118 的缓冲接口
class TestScalarPEP3118:

    # 参数化测试，对每种标量类型进行测试
    @pytest.mark.parametrize('scalar', scalars_only, ids=codes_only)
    def test_scalar_match_array(self, scalar):
        # 创建标量对象和相应的空数组
        x = scalar()
        a = np.array([], dtype=np.dtype(scalar))
        # 创建标量的内存视图
        mv_x = memoryview(x)
        mv_a = memoryview(a)
        # 断言两者的格式相同
        assert_equal(mv_x.format, mv_a.format)

    # 参数化测试，对每种标量类型进行测试
    @pytest.mark.parametrize('scalar', scalars_only, ids=codes_only)
    def test_scalar_dim(self, scalar):
        # 创建标量对象并获取其内存视图
        x = scalar()
        mv_x = memoryview(x)
        # 断言标量的字节大小与其 dtype 的字节大小相等
        assert_equal(mv_x.itemsize, np.dtype(scalar).itemsize)
        # 断言标量的维度为 0
        assert_equal(mv_x.ndim, 0)
        # 断言标量的形状为 ()
        assert_equal(mv_x.shape, ())
        # 断言标量的步幅为 ()
        assert_equal(mv_x.strides, ())
        # 断言标量的子偏移为 ()
        assert_equal(mv_x.suboffsets, ())

    # 参数化测试，对每种标量类型进行测试
    @pytest.mark.parametrize('scalar, code', scalars_and_codes, ids=codes_only)
    def test_scalar_code_and_properties(self, scalar, code):
        # 创建标量对象
        x = scalar()
        # 期望的内存视图属性字典
        expected = dict(strides=(), itemsize=x.dtype.itemsize, ndim=0,
                        shape=(), format=code, readonly=True)
        # 获取标量对象的内存视图
        mv_x = memoryview(x)
        # 断言内存视图的属性与期望的属性相等
        assert self._as_dict(mv_x) == expected

    # 参数化测试，对每种标量类型进行测试
    @pytest.mark.parametrize('scalar', scalars_only, ids=codes_only)
    def test_scalar_buffers_readonly(self, scalar):
        # 创建标量对象
        x = scalar()
        # 使用 pytest 断言捕获 BufferError 异常，确保标量缓冲区只读
        with pytest.raises(BufferError, match="scalar buffer is readonly"):
            get_buffer_info(x, ["WRITABLE"])
    def test_void_scalar_structured_data(self):
        # 定义一个结构化数据类型，包含一个长度为16的字符串和一个形状为(2,)的双精度浮点数数组
        dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])
        # 创建一个标量 ndarray，符合上述定义的数据类型，并取出其中的标量值
        x = np.array(('ndarray_scalar', (1.2, 3.0)), dtype=dt)[()]
        # 断言 x 是 np.void 类型的对象
        assert_(isinstance(x, np.void))
        # 从 x 中创建一个 memoryview 对象
        mv_x = memoryview(x)
        # 计算预期的 memoryview 大小，包括字符串和浮点数数组的大小
        expected_size = 16 * np.dtype((np.str_, 1)).itemsize
        expected_size += 2 * np.dtype(np.float64).itemsize
        # 断言 memoryview 的 itemsize 符合预期大小
        assert_equal(mv_x.itemsize, expected_size)
        # 断言 memoryview 的维度为 0
        assert_equal(mv_x.ndim, 0)
        # 断言 memoryview 的形状为空元组
        assert_equal(mv_x.shape, ())
        # 断言 memoryview 的 strides 为空元组
        assert_equal(mv_x.strides, ())
        # 断言 memoryview 的 suboffsets 为空元组
        assert_equal(mv_x.suboffsets, ())

        # 检查标量格式字符串与 ndarray 格式字符串是否匹配
        a = np.array([('Sarah', (8.0, 7.0)), ('John', (6.0, 7.0))], dtype=dt)
        # 断言 a 是 np.ndarray 类型的对象
        assert_(isinstance(a, np.ndarray))
        # 从 a 中创建一个 memoryview 对象
        mv_a = memoryview(a)
        # 断言 memoryview 的 itemsize 与 mv_x 的 itemsize 相等
        assert_equal(mv_x.itemsize, mv_a.itemsize)
        # 断言 memoryview 的 format 与 mv_a 的 format 相等

        # 检查不允许导出可写缓冲区（技术上有时可能会允许在此处...）
        with pytest.raises(BufferError, match="scalar buffer is readonly"):
            get_buffer_info(x, ["WRITABLE"])

    def _as_dict(self, m):
        # 返回 memoryview 对象 m 的属性字典
        return dict(strides=m.strides, shape=m.shape, itemsize=m.itemsize,
                    ndim=m.ndim, format=m.format, readonly=m.readonly)

    def test_datetime_memoryview(self):
        # gh-11656
        # 使用 np.datetime64 创建一个 datetime 对象 dt1
        dt1 = np.datetime64('2016-01-01')
        # 使用 np.datetime64 创建一个 datetime 对象 dt2
        dt2 = np.datetime64('2017-01-01')
        # 定义预期的 memoryview 属性字典，对于 dt1 和 dt2 都应相同
        expected = dict(strides=(1,), itemsize=1, ndim=1, shape=(8,),
                        format='B', readonly=True)
        # 创建 dt1 的 memoryview 对象 v，并断言其属性字典与预期相同
        v = memoryview(dt1)
        assert self._as_dict(v) == expected
        # 创建 dt2 - dt1 的 memoryview 对象 v，并断言其属性字典与预期相同
        v = memoryview(dt2 - dt1)
        assert self._as_dict(v) == expected

        # 创建一个 dtype，包含一个 'uint16' 类型的字段 'a' 和一个 'M8[s]' 类型的字段 'b'
        dt = np.dtype([('a', 'uint16'), ('b', 'M8[s]')])
        # 创建一个空的 dt 类型的 ndarray a
        a = np.empty(1, dt)
        # 尝试创建一个 PEP 3118 格式不合法的缓冲区，预期会失败
        assert_raises((ValueError, BufferError), memoryview, a[0])

        # 检查不允许导出可写缓冲区
        with pytest.raises(BufferError, match="scalar buffer is readonly"):
            get_buffer_info(dt1, ["WRITABLE"])

    @pytest.mark.parametrize('s', [
        pytest.param("\x32\x32", id="ascii"),
        pytest.param("\uFE0F\uFE0F", id="basic multilingual"),
        pytest.param("\U0001f4bb\U0001f4bb", id="non-BMP"),
    ])
    # 定义一个测试方法，用于测试 UCS4 字符串的特性
    def test_str_ucs4(self, s):
        # 将输入的字符串 s 转换为 numpy.str_ 类型，仅子类实现了缓冲区协议
        s = np.str_(s)  # only our subclass implements the buffer protocol

        # 预期结果，表示 UCS4 编码的字符串特性
        expected = dict(strides=(), itemsize=8, ndim=0, shape=(), format='2w',
                        readonly=True)

        # 创建内存视图对象 v
        v = memoryview(s)
        # 断言内存视图的属性与预期的字典 expected 相符
        assert self._as_dict(v) == expected

        # 将内存视图 v 视作 i4 类型的 numpy 数组，获取其对应的整数码点
        code_points = np.frombuffer(v, dtype='i4')

        # 断言获取的码点与字符串 s 中字符的 Unicode 编码相符
        assert_equal(code_points, [ord(c) for c in s])

        # 检查不允许导出可写缓冲区的情况
        with pytest.raises(BufferError, match="scalar buffer is readonly"):
            get_buffer_info(s, ["WRITABLE"])

    # 测试当用户自定义标量类型无法导出缓冲区时的情况
    def test_user_scalar_fails_buffer(self):
        # 创建一个有理数对象 r
        r = rational(1)
        # 使用 assert_raises 检查期望会抛出 TypeError 异常
        with assert_raises(TypeError):
            memoryview(r)

        # 再次检查不允许导出可写缓冲区的情况
        with pytest.raises(BufferError, match="scalar buffer is readonly"):
            get_buffer_info(r, ["WRITABLE"])
```