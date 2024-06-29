# `.\numpy\numpy\_core\tests\test_print.py`

```
# 导入系统模块 sys
import sys

# 导入 pytest 模块
import pytest

# 导入 numpy 库，并从中导入 assert_, assert_equal, IS_MUSL
import numpy as np
from numpy.testing import assert_, assert_equal, IS_MUSL

# 从 numpy._core.tests._locales 中导入 CommaDecimalPointLocale
from numpy._core.tests._locales import CommaDecimalPointLocale

# 导入 io 模块中的 StringIO 类
from io import StringIO

# 定义一个全局变量 _REF，用于存储特定浮点数值与字符串表示的对应关系
_REF = {np.inf: 'inf', -np.inf: '-inf', np.nan: 'nan'}


# 使用 pytest 的 parametrize 装饰器，测试不同浮点数类型的字符串格式化输出
@pytest.mark.parametrize('tp', [np.float32, np.double, np.longdouble])
def test_float_types(tp):
    """ Check formatting.

        This is only for the str function, and only for simple types.
        The precision of np.float32 and np.longdouble aren't the same as the
        python float precision.

    """
    # 遍历不同的浮点数值，并进行字符串格式化输出的测试
    for x in [0, 1, -1, 1e20]:
        # 断言字符串格式化输出结果与预期的 Python 浮点数格式化输出一致
        assert_equal(str(tp(x)), str(float(x)),
                     err_msg='Failed str formatting for type %s' % tp)

    # 对于 itemsize 大于 4 的浮点数类型，继续进行字符串格式化输出测试
    if tp(1e16).itemsize > 4:
        assert_equal(str(tp(1e16)), str(float('1e16')),
                     err_msg='Failed str formatting for type %s' % tp)
    else:
        # 否则，使用预定义的字符串格式化输出进行断言
        ref = '1e+16'
        assert_equal(str(tp(1e16)), ref,
                     err_msg='Failed str formatting for type %s' % tp)


# 使用 pytest 的 parametrize 装饰器，测试 nan 和 inf 的字符串格式化输出
@pytest.mark.parametrize('tp', [np.float32, np.double, np.longdouble])
def test_nan_inf_float(tp):
    """ Check formatting of nan & inf.

        This is only for the str function, and only for simple types.
        The precision of np.float32 and np.longdouble aren't the same as the
        python float precision.

    """
    # 遍历 nan 和 inf 值，并进行字符串格式化输出的测试
    for x in [np.inf, -np.inf, np.nan]:
        # 断言字符串格式化输出结果与预期的特定值对应
        assert_equal(str(tp(x)), _REF[x],
                     err_msg='Failed str formatting for type %s' % tp)


# 使用 pytest 的 parametrize 装饰器，测试复数类型的字符串格式化输出
@pytest.mark.parametrize('tp', [np.complex64, np.cdouble, np.clongdouble])
def test_complex_types(tp):
    """Check formatting of complex types.

        This is only for the str function, and only for simple types.
        The precision of np.float32 and np.longdouble aren't the same as the
        python float precision.

    """
    # 遍历不同的实数值和虚数值，并进行复数类型的字符串格式化输出测试
    for x in [0, 1, -1, 1e20]:
        # 断言字符串格式化输出结果与预期的复数格式化输出一致
        assert_equal(str(tp(x)), str(complex(x)),
                     err_msg='Failed str formatting for type %s' % tp)
        assert_equal(str(tp(x*1j)), str(complex(x*1j)),
                     err_msg='Failed str formatting for type %s' % tp)
        assert_equal(str(tp(x + x*1j)), str(complex(x + x*1j)),
                     err_msg='Failed str formatting for type %s' % tp)

    # 对于 itemsize 大于 8 的复数类型，继续进行字符串格式化输出测试
    if tp(1e16).itemsize > 8:
        assert_equal(str(tp(1e16)), str(complex(1e16)),
                     err_msg='Failed str formatting for type %s' % tp)
    else:
        # 否则，使用预定义的字符串格式化输出进行断言
        ref = '(1e+16+0j)'
        assert_equal(str(tp(1e16)), ref,
                     err_msg='Failed str formatting for type %s' % tp)


# 使用 pytest 的 parametrize 装饰器，测试复数类型中的 inf 和 nan 的字符串格式化输出
@pytest.mark.parametrize('dtype', [np.complex64, np.cdouble, np.clongdouble])
def test_complex_inf_nan(dtype):
    """Check inf/nan formatting of complex types."""
    # 定义一个字典 TESTS，包含各种复数值和对应的字符串表示
    TESTS = {
        complex(np.inf, 0): "(inf+0j)",      # 复数值为正无穷大加零，期望结果是字符串 "(inf+0j)"
        complex(0, np.inf): "infj",          # 复数值为零加正无穷大，期望结果是字符串 "infj"
        complex(-np.inf, 0): "(-inf+0j)",    # 复数值为负无穷大加零，期望结果是字符串 "(-inf+0j)"
        complex(0, -np.inf): "-infj",        # 复数值为零加负无穷大，期望结果是字符串 "-infj"
        complex(np.inf, 1): "(inf+1j)",      # 复数值为正无穷大加一，期望结果是字符串 "(inf+1j)"
        complex(1, np.inf): "(1+infj)",      # 复数值为一加正无穷大，期望结果是字符串 "(1+infj)"
        complex(-np.inf, 1): "(-inf+1j)",    # 复数值为负无穷大加一，期望结果是字符串 "(-inf+1j)"
        complex(1, -np.inf): "(1-infj)",     # 复数值为一加负无穷大，期望结果是字符串 "(1-infj)"
        complex(np.nan, 0): "(nan+0j)",      # 复数值为 NaN 加零，期望结果是字符串 "(nan+0j)"
        complex(0, np.nan): "nanj",          # 复数值为零加 NaN，期望结果是字符串 "nanj"
        complex(-np.nan, 0): "(nan+0j)",     # 复数值为负 NaN 加零，期望结果是字符串 "(nan+0j)"
        complex(0, -np.nan): "nanj",         # 复数值为零加负 NaN，期望结果是字符串 "nanj"
        complex(np.nan, 1): "(nan+1j)",      # 复数值为 NaN 加一，期望结果是字符串 "(nan+1j)"
        complex(1, np.nan): "(1+nanj)",      # 复数值为一加 NaN，期望结果是字符串 "(1+nanj)"
        complex(-np.nan, 1): "(nan+1j)",     # 复数值为负 NaN 加一，期望结果是字符串 "(nan+1j)"
        complex(1, -np.nan): "(1+nanj)",     # 复数值为一加负 NaN，期望结果是字符串 "(1+nanj)"
    }
    # 遍历 TESTS 字典中的每个复数和对应的字符串表示
    for c, s in TESTS.items():
        # 使用 assert_equal 函数检查复数 c 的数据类型的字符串表示是否等于预期字符串 s
        assert_equal(str(dtype(c)), s)
# print tests
# 定义用于测试重定向打印输出的函数，接受参数 x (输入值)、tp (类型)、ref (参考值，默认为 None)
def _test_redirected_print(x, tp, ref=None):
    # 创建一个内存中的字符串流对象用于捕获标准输出
    file = StringIO()
    # 创建另一个字符串流对象，用于比较标准输出
    file_tp = StringIO()
    # 保存当前标准输出对象
    stdout = sys.stdout
    try:
        # 重定向标准输出到 file_tp 对象
        sys.stdout = file_tp
        # 打印 tp(x) 的结果到重定向的标准输出
        print(tp(x))
        # 再次重定向标准输出到 file 对象
        sys.stdout = file
        # 如果有提供参考值 ref，则打印 ref；否则打印 x
        if ref:
            print(ref)
        else:
            print(x)
    finally:
        # 恢复原始的标准输出
        sys.stdout = stdout

    # 断言重定向后的输出与预期一致，用于测试目的
    assert_equal(file.getvalue(), file_tp.getvalue(),
                 err_msg='print failed for type%s' % tp)


# 使用 pytest 的 parametrize 装饰器，对不同的浮点类型进行格式化输出测试
@pytest.mark.parametrize('tp', [np.float32, np.double, np.longdouble])
def test_float_type_print(tp):
    """Check formatting when using print """
    # 对一些常见的浮点数进行测试
    for x in [0, 1, -1, 1e20]:
        _test_redirected_print(float(x), tp)

    # 对无穷大、负无穷大和 NaN 进行特殊测试
    for x in [np.inf, -np.inf, np.nan]:
        _test_redirected_print(float(x), tp, _REF[x])

    # 如果浮点类型 tp 的字节大小大于 4，则继续测试 1e16 的输出
    if tp(1e16).itemsize > 4:
        _test_redirected_print(float(1e16), tp)
    else:
        # 否则使用参考值 '1e+16'
        ref = '1e+16'
        _test_redirected_print(float(1e16), tp, ref)


# 使用 pytest 的 parametrize 装饰器，对不同的复数类型进行格式化输出测试
@pytest.mark.parametrize('tp', [np.complex64, np.cdouble, np.clongdouble])
def test_complex_type_print(tp):
    """Check formatting when using print """
    # 对一些常见的复数进行测试
    # 因为 Python < 2.6 不支持直接创建具有无穷大和 NaN 的复数，所以避免直接使用它们
    for x in [0, 1, -1, 1e20]:
        _test_redirected_print(complex(x), tp)

    # 如果复数类型 tp 的字节大小大于 8，则继续测试 1e16 的输出
    if tp(1e16).itemsize > 8:
        _test_redirected_print(complex(1e16), tp)
    else:
        # 否则使用参考值 '(1e+16+0j)'
        ref = '(1e+16+0j)'
        _test_redirected_print(complex(1e16), tp, ref)

    # 测试特殊的复数情况：无穷大、负无穷大和 NaN
    _test_redirected_print(complex(np.inf, 1), tp, '(inf+1j)')
    _test_redirected_print(complex(-np.inf, 1), tp, '(-inf+1j)')
    _test_redirected_print(complex(-np.nan, 1), tp, '(nan+1j)')


# 测试 NumPy 标量类型的字符串格式化输出方法
def test_scalar_format():
    """Test the str.format method with NumPy scalar types"""
    # 定义一系列测试用例，包括格式字符串、值和值的类型
    tests = [('{0}', True, np.bool),
            ('{0}', False, np.bool),
            ('{0:d}', 130, np.uint8),
            ('{0:d}', 50000, np.uint16),
            ('{0:d}', 3000000000, np.uint32),
            ('{0:d}', 15000000000000000000, np.uint64),
            ('{0:d}', -120, np.int8),
            ('{0:d}', -30000, np.int16),
            ('{0:d}', -2000000000, np.int32),
            ('{0:d}', -7000000000000000000, np.int64),
            ('{0:g}', 1.5, np.float16),
            ('{0:g}', 1.5, np.float32),
            ('{0:g}', 1.5, np.float64),
            ('{0:g}', 1.5, np.longdouble),
            ('{0:g}', 1.5+0.5j, np.complex64),
            ('{0:g}', 1.5+0.5j, np.complex128),
            ('{0:g}', 1.5+0.5j, np.clongdouble)]

    # 遍历测试用例并执行测试
    for (fmat, val, valtype) in tests:
        try:
            # 使用格式化字符串 fmat 对 val 和 valtype(val) 进行格式化
            assert_equal(fmat.format(val), fmat.format(valtype(val)),
                         "failed with val %s, type %s" % (val, valtype))
        except ValueError as e:
            assert_(False,
                    "format raised exception (fmt='%s', val=%s, type=%s, exc='%s')" %
                    (fmat, repr(val), repr(valtype), str(e)))


#
# Locale tests: scalar types formatting should be independent of the locale
#
# 定义一个测试类 TestCommaDecimalPointLocale，继承自 CommaDecimalPointLocale 类
class TestCommaDecimalPointLocale(CommaDecimalPointLocale):

    # 测试单精度浮点数在本地化环境下的字符串表示是否正确
    def test_locale_single(self):
        # 断言单精度浮点数的字符串表示应与普通浮点数相同
        assert_equal(str(np.float32(1.2)), str(float(1.2)))

    # 测试双精度浮点数在本地化环境下的字符串表示是否正确
    def test_locale_double(self):
        # 断言双精度浮点数的字符串表示应与普通浮点数相同
        assert_equal(str(np.double(1.2)), str(float(1.2)))

    # 根据条件标记，跳过测试长双精度浮点数在本地化环境下的字符串表示
    @pytest.mark.skipif(IS_MUSL,
                        reason="test flaky on musllinux")
    def test_locale_longdouble(self):
        # 断言长双精度浮点数的字符串表示应与普通浮点数相同
        assert_equal(str(np.longdouble('1.2')), str(float(1.2)))
```