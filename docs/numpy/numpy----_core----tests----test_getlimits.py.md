# `.\numpy\numpy\_core\tests\test_getlimits.py`

```py
# 导入警告模块，用于处理警告信息
import warnings
# 导入 numpy 库
import numpy as np
# 导入 pytest 模块，用于编写和运行测试
import pytest
# 从 numpy._core 模块导入 finfo 和 iinfo 类
from numpy._core import finfo, iinfo
# 导入 half、single、double、longdouble 类型
from numpy import half, single, double, longdouble
# 导入 numpy.testing 模块中的断言函数
from numpy.testing import assert_equal, assert_, assert_raises
# 从 numpy._core.getlimits 模块中导入 _discovered_machar 和 _float_ma 函数

##################################################

# 定义 TestPythonFloat 测试类，测试 float 类型的 finfo 对象
class TestPythonFloat:
    # 测试 finfo 对象是否为单例
    def test_singleton(self):
        # 获取 float 类型的 finfo 对象
        ftype = finfo(float)
        # 再次获取 float 类型的 finfo 对象
        ftype2 = finfo(float)
        # 断言两个 finfo 对象的内存地址相同
        assert_equal(id(ftype), id(ftype2))

# 定义 TestHalf 测试类，测试 half 类型的 finfo 对象
class TestHalf:
    # 测试 finfo 对象是否为单例
    def test_singleton(self):
        # 获取 half 类型的 finfo 对象
        ftype = finfo(half)
        # 再次获取 half 类型的 finfo 对象
        ftype2 = finfo(half)
        # 断言两个 finfo 对象的内存地址相同
        assert_equal(id(ftype), id(ftype2))

# 定义 TestSingle 测试类，测试 single 类型的 finfo 对象
class TestSingle:
    # 测试 finfo 对象是否为单例
    def test_singleton(self):
        # 获取 single 类型的 finfo 对象
        ftype = finfo(single)
        # 再次获取 single 类型的 finfo 对象
        ftype2 = finfo(single)
        # 断言两个 finfo 对象的内存地址相同
        assert_equal(id(ftype), id(ftype2))

# 定义 TestDouble 测试类，测试 double 类型的 finfo 对象
class TestDouble:
    # 测试 finfo 对象是否为单例
    def test_singleton(self):
        # 获取 double 类型的 finfo 对象
        ftype = finfo(double)
        # 再次获取 double 类型的 finfo 对象
        ftype2 = finfo(double)
        # 断言两个 finfo 对象的内存地址相同
        assert_equal(id(ftype), id(ftype2))

# 定义 TestLongdouble 测试类，测试 longdouble 类型的 finfo 对象
class TestLongdouble:
    # 测试 finfo 对象是否为单例
    def test_singleton(self):
        # 获取 longdouble 类型的 finfo 对象
        ftype = finfo(longdouble)
        # 再次获取 longdouble 类型的 finfo 对象
        ftype2 = finfo(longdouble)
        # 断言两个 finfo 对象的内存地址相同
        assert_equal(id(ftype), id(ftype2))

# 定义一个辅助函数 assert_finfo_equal，用于断言两个 finfo 实例具有相同的属性值
def assert_finfo_equal(f1, f2):
    # 断言两个 finfo 实例在所有指定属性上的值相同
    for attr in ('bits', 'eps', 'epsneg', 'iexp', 'machep',
                 'max', 'maxexp', 'min', 'minexp', 'negep', 'nexp',
                 'nmant', 'precision', 'resolution', 'tiny',
                 'smallest_normal', 'smallest_subnormal'):
        assert_equal(getattr(f1, attr), getattr(f2, attr),
                     f'finfo instances {f1} and {f2} differ on {attr}')

# 定义一个辅助函数 assert_iinfo_equal，用于断言两个 iinfo 实例具有相同的属性值
def assert_iinfo_equal(i1, i2):
    # 断言两个 iinfo 实例在所有指定属性上的值相同
    for attr in ('bits', 'min', 'max'):
        assert_equal(getattr(i1, attr), getattr(i2, attr),
                     f'iinfo instances {i1} and {i2} differ on {attr}')

# 定义 TestFinfo 测试类，测试 finfo 对象的基本功能
class TestFinfo:
    # 测试基本的 finfo 功能
    def test_basic(self):
        # 定义一个包含不同数据类型的列表
        dts = list(zip(['f2', 'f4', 'f8', 'c8', 'c16'],
                       [np.float16, np.float32, np.float64, np.complex64,
                        np.complex128]))
        # 遍历列表中的每一对数据类型
        for dt1, dt2 in dts:
            # 断言两个不同数据类型的 finfo 实例具有相同的属性值
            assert_finfo_equal(finfo(dt1), finfo(dt2))

        # 断言当传入无效数据类型 'i4' 时会抛出 ValueError 异常
        assert_raises(ValueError, finfo, 'i4')

    # 测试 finfo 对象的回归测试 (Regression Test) - gh23108
    def test_regression_gh23108(self):
        # np.float32(1.0) 和 np.float64(1.0) 具有相同的哈希值，并且在 == 运算符下相等
        f1 = np.finfo(np.float32(1.0))
        f2 = np.finfo(np.float64(1.0))
        # 断言 f1 和 f2 不相等
        assert f1 != f2

    # 测试 finfo 对象的回归测试 (Regression Test) - gh23867
    def test_regression_gh23867(self):
        # 定义一个具有 dtype 属性的非可哈希对象
        class NonHashableWithDtype:
            __hash__ = None
            dtype = np.dtype('float32')
  
        x = NonHashableWithDtype()
        # 断言 np.finfo(x) 与 np.finfo(x.dtype) 相等
        assert np.finfo(x) == np.finfo(x.dtype)

# 定义 TestIinfo 测试类，用于测试整数信息对象 iinfo
class TestIinfo:
    # 定义一个测试方法，用于测试基本功能
    def test_basic(self):
        # 创建一个包含数据类型和对应的numpy数据类型的元组列表
        dts = list(zip(['i1', 'i2', 'i4', 'i8',
                   'u1', 'u2', 'u4', 'u8'],
                  [np.int8, np.int16, np.int32, np.int64,
                   np.uint8, np.uint16, np.uint32, np.uint64]))
        # 遍历元组列表，分别使用 assert_iinfo_equal 函数对数据类型信息进行断言比较
        for dt1, dt2 in dts:
            assert_iinfo_equal(iinfo(dt1), iinfo(dt2))

        # 使用 assert_raises 函数验证调用 iinfo('f4') 会引发 ValueError 异常
        assert_raises(ValueError, iinfo, 'f4')

    # 定义一个测试方法，用于测试无符号整数类型的最大值计算
    def test_unsigned_max(self):
        # 获取numpy支持的所有无符号整数类型
        types = np._core.sctypes['uint']
        # 遍历每种类型
        for T in types:
            # 在忽略溢出的错误状态下，计算当前类型 T 的最大值
            with np.errstate(over="ignore"):
                max_calculated = T(0) - T(1)
            # 断言 iinfo(T).max 等于上面计算得到的最大值
            assert_equal(iinfo(T).max, max_calculated)
class TestRepr:
    # 测试 np.iinfo 函数的输出是否符合预期
    def test_iinfo_repr(self):
        expected = "iinfo(min=-32768, max=32767, dtype=int16)"
        assert_equal(repr(np.iinfo(np.int16)), expected)

    # 测试 np.finfo 函数的输出是否符合预期
    def test_finfo_repr(self):
        expected = "finfo(resolution=1e-06, min=-3.4028235e+38," + \
                   " max=3.4028235e+38, dtype=float32)"
        assert_equal(repr(np.finfo(np.float32)), expected)


def test_instances():
    # 测试在数值实例上使用 iinfo 和 finfo 是否与对应类型的结果一致
    for c in [int, np.int16, np.int32, np.int64]:
        class_iinfo = iinfo(c)
        instance_iinfo = iinfo(c(12))

        assert_iinfo_equal(class_iinfo, instance_iinfo)

    for c in [float, np.float16, np.float32, np.float64]:
        class_finfo = finfo(c)
        instance_finfo = finfo(c(1.2))
        assert_finfo_equal(class_finfo, instance_finfo)

    # 测试当传入不合法参数时是否抛出 ValueError 异常
    with pytest.raises(ValueError):
        iinfo(10.)

    with pytest.raises(ValueError):
        iinfo('hi')

    with pytest.raises(ValueError):
        finfo(np.int64(1))


def assert_ma_equal(discovered, ma_like):
    # 检查 MachAr 类对象是否与计算出的 MachAr 实例相同
    for key, value in discovered.__dict__.items():
        assert_equal(value, getattr(ma_like, key))
        if hasattr(value, 'shape'):
            assert_equal(value.shape, getattr(ma_like, key).shape)
            assert_equal(value.dtype, getattr(ma_like, key).dtype)


def test_known_types():
    # 测试对已知类型编译参数是否正确
    for ftype, ma_like in ((np.float16, _float_ma[16]),
                           (np.float32, _float_ma[32]),
                           (np.float64, _float_ma[64])):
        assert_ma_equal(_discovered_machar(ftype), ma_like)
    # 抑制对 PPC 平台上双倍精度发现的警告
    with np.errstate(all='ignore'):
        ld_ma = _discovered_machar(np.longdouble)
    bytes = np.dtype(np.longdouble).itemsize
    if (ld_ma.it, ld_ma.maxexp) == (63, 16384) and bytes in (12, 16):
        # 80 位扩展精度
        assert_ma_equal(ld_ma, _float_ma[80])
    elif (ld_ma.it, ld_ma.maxexp) == (112, 16384) and bytes == 16:
        # IEE 754 128 位
        assert_ma_equal(ld_ma, _float_ma[128])


def test_subnormal_warning():
    """测试是否会引发次正常为零的警告"""
    with np.errstate(all='ignore'):
        ld_ma = _discovered_machar(np.longdouble)
    bytes = np.dtype(np.longdouble).itemsize
    # 使用 `warnings` 模块捕获警告信息
    with warnings.catch_warnings(record=True) as w:
        # 设置警告过滤器，始终显示警告
        warnings.simplefilter('always')
        # 如果 ld_ma.it 等于 63，ld_ma.maxexp 等于 16384，且 bytes 是 12 或 16
        if (ld_ma.it, ld_ma.maxexp) == (63, 16384) and bytes in (12, 16):
            # 对于80位扩展精度的情况
            ld_ma.smallest_subnormal
            # 断言：期望没有警告被触发
            assert len(w) == 0
        # 如果 ld_ma.it 等于 112，ld_ma.maxexp 等于 16384，且 bytes 是 16
        elif (ld_ma.it, ld_ma.maxexp) == (112, 16384) and bytes == 16:
            # 对于 IEEE 754 128位的情况
            ld_ma.smallest_subnormal
            # 断言：期望没有警告被触发
            assert len(w) == 0
        else:
            # 对于双倍精度的情况
            ld_ma.smallest_subnormal
            # 提示：这个测试可能在某些平台上失败
            assert len(w) == 0
# 定义一个测试函数，用于验证 np.finfo 函数对所有类型的浮点数和复数返回合理的结果
def test_plausible_finfo():
    # 遍历所有浮点数和复数类型，包括浮点数和复数的子类型
    for ftype in np._core.sctypes['float'] + np._core.sctypes['complex']:
        # 调用 np.finfo 函数获取特定类型的信息
        info = np.finfo(ftype)
        # 断言该类型的尾数位数大于1
        assert_(info.nmant > 1)
        # 断言该类型的最小指数小于-1
        assert_(info.minexp < -1)
        # 断言该类型的最大指数大于1
        assert_(info.maxexp > 1)
```