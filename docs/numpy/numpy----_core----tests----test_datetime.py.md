# `.\numpy\numpy\_core\tests\test_datetime.py`

```
# 导入标准库中的 datetime 和 pickle 模块
import datetime
import pickle

# 导入 pytest 测试框架及相关库
import pytest

# 导入 numpy 库，并给 numpy 别名 np
import numpy
import numpy as np

# 从 numpy.testing 模块导入多个函数和常量
from numpy.testing import (
    IS_WASM,                    # 是否是 WASM 环境的标志
    assert_,                    # 断言函数
    assert_equal,               # 断言相等
    assert_raises,              # 断言引发异常
    assert_warns,               # 断言引发警告
    suppress_warnings,          # 忽略警告
    assert_raises_regex,        # 断言引发指定正则表达式匹配的异常
    assert_array_equal,         # 断言数组相等
)

# 尝试导入 pytz 库中的 timezone 类
try:
    from pytz import timezone as tz
    _has_pytz = True            # 标记是否成功导入 pytz 库
except ImportError:
    _has_pytz = False           # 若导入失败则置为 False

# 尝试定义 RecursionError，兼容 Python < 3.5 的情况
try:
    RecursionError
except NameError:
    RecursionError = RuntimeError  # 将 NameError 映射为 RuntimeError

# 定义测试类 TestDateTime
class TestDateTime:

    # 定义测试方法 test_string
    def test_string(self):
        # 准备警告消息
        msg = "no explicit representation of timezones available for np.datetime64"
        # 使用 pytest 捕获 UserWarning 类型的警告，并匹配特定的消息字符串
        with pytest.warns(UserWarning, match=msg):
            # 创建一个 np.datetime64 对象，带时区信息 '+01'，触发警告
            np.datetime64('2000-01-01T00+01')

    # 定义测试方法 test_datetime
    def test_datetime(self):
        # 准备警告消息
        msg = "no explicit representation of timezones available for np.datetime64"
        # 使用 pytest 捕获 UserWarning 类型的警告，并匹配特定的消息字符串
        with pytest.warns(UserWarning, match=msg):
            # 创建一个 np.datetime64 对象，带 'Z' 时区信息，触发警告
            t0 = np.datetime64('2023-06-09T12:18:40Z', 'ns')

        # 创建一个 np.datetime64 对象，无显式时区信息
        t0 = np.datetime64('2023-06-09T12:18:40', 'ns')
    # 定义测试函数，用于测试 datetime 数据类型的创建
    def test_datetime_dtype_creation(self):
        # 遍历不同的时间单位
        for unit in ['Y', 'M', 'W', 'D',
                     'h', 'm', 's', 'ms', 'us',
                     'μs',  # alias for us
                     'ns', 'ps', 'fs', 'as']:
            # 创建一个 datetime64 数据类型的对象 dt1
            dt1 = np.dtype('M8[750%s]' % unit)
            # 断言 dt1 的类型与指定的 datetime64 数据类型相同
            assert_(dt1 == np.dtype('datetime64[750%s]' % unit))
            # 创建一个 timedelta64 数据类型的对象 dt2
            dt2 = np.dtype('m8[%s]' % unit)
            # 断言 dt2 的类型与指定的 timedelta64 数据类型相同
            assert_(dt2 == np.dtype('timedelta64[%s]' % unit))

        # 对于通用单位，不应在末尾添加 []
        assert_equal(str(np.dtype("M8")), "datetime64")

        # 应当可以指定字节顺序
        assert_equal(np.dtype("=M8"), np.dtype("M8"))
        assert_equal(np.dtype("=M8[s]"), np.dtype("M8[s]"))
        # 断言大端序或小端序的 datetime64 数据类型与普通的 datetime64 相同
        assert_(np.dtype(">M8") == np.dtype("M8") or
                np.dtype("<M8") == np.dtype("M8"))
        assert_(np.dtype(">M8[D]") == np.dtype("M8[D]") or
                np.dtype("<M8[D]") == np.dtype("M8[D]"))
        # 断言大端序与小端序的 datetime64 数据类型不相同
        assert_(np.dtype(">M8") != np.dtype("<M8"))

        assert_equal(np.dtype("=m8"), np.dtype("m8"))
        assert_equal(np.dtype("=m8[s]"), np.dtype("m8[s]"))
        # 断言大端序或小端序的 timedelta64 数据类型与普通的 timedelta64 相同
        assert_(np.dtype(">m8") == np.dtype("m8") or
                np.dtype("<m8") == np.dtype("m8"))
        assert_(np.dtype(">m8[D]") == np.dtype("m8[D]") or
                np.dtype("<m8[D]") == np.dtype("m8[D]"))
        # 断言大端序与小端序的 timedelta64 数据类型不相同
        assert_(np.dtype(">m8") != np.dtype("<m8"))

        # 检查解析器是否拒绝错误的 datetime 类型
        assert_raises(TypeError, np.dtype, 'M8[badunit]')
        assert_raises(TypeError, np.dtype, 'm8[badunit]')
        assert_raises(TypeError, np.dtype, 'M8[YY]')
        assert_raises(TypeError, np.dtype, 'm8[YY]')
        assert_raises(TypeError, np.dtype, 'm4')
        assert_raises(TypeError, np.dtype, 'M7')
        assert_raises(TypeError, np.dtype, 'm7')
        assert_raises(TypeError, np.dtype, 'M16')
        assert_raises(TypeError, np.dtype, 'm16')
        assert_raises(TypeError, np.dtype, 'M8[3000000000ps]')
    def test_datetime_prefix_conversions(self):
        # regression tests related to gh-19631;
        # test metric prefixes from seconds down to
        # attoseconds for bidirectional conversions

        # 定义较小时间单位的列表，用于测试从秒到atto秒的双向转换
        smaller_units = ['M8[7000ms]',
                         'M8[2000us]',
                         'M8[1000ns]',
                         'M8[5000ns]',
                         'M8[2000ps]',
                         'M8[9000fs]',
                         'M8[1000as]',
                         'M8[2000000ps]',
                         'M8[1000000as]',
                         'M8[2000000000ps]',
                         'M8[1000000000as]']

        # 定义较大时间单位的列表，用于测试从秒到atto秒的双向转换
        larger_units = ['M8[7s]',
                        'M8[2ms]',
                        'M8[us]',
                        'M8[5us]',
                        'M8[2ns]',
                        'M8[9ps]',
                        'M8[1fs]',
                        'M8[2us]',
                        'M8[1ps]',
                        'M8[2ms]',
                        'M8[1ns]']

        # 遍历较大和较小时间单位列表，进行类型转换的安全性断言
        for larger_unit, smaller_unit in zip(larger_units, smaller_units):
            assert np.can_cast(larger_unit, smaller_unit, casting='safe')
            assert np.can_cast(smaller_unit, larger_unit, casting='safe')

    @pytest.mark.parametrize("unit", [
        "s", "ms", "us", "ns", "ps", "fs", "as"])
    def test_prohibit_negative_datetime(self, unit):
        # 使用断言检测是否能够禁止负的时间单位
        with assert_raises(TypeError):
            np.array([1], dtype=f"M8[-1{unit}]")

    def test_compare_generic_nat(self):
        # regression tests for gh-6452
        # 检测泛型 NaT 的比较行为，测试 gh-6452 的回归情况
        assert_(np.datetime64('NaT') !=
                np.datetime64('2000') + np.timedelta64('NaT'))
        assert_(np.datetime64('NaT') != np.datetime64('NaT', 'us'))
        assert_(np.datetime64('NaT', 'us') != np.datetime64('NaT'))

    @pytest.mark.parametrize("size", [
        3, 21, 217, 1000])
    def test_datetime_nat_argsort_stability(self, size):
        # NaT < NaT should be False internally for
        # sort stability
        # 测试日期时间 NaT 的 argsort 稳定性，确保 NaT < NaT 返回 False
        expected = np.arange(size)
        arr = np.tile(np.datetime64('NaT'), size)
        assert_equal(np.argsort(arr, kind='mergesort'), expected)

    @pytest.mark.parametrize("size", [
        3, 21, 217, 1000])
    def test_timedelta_nat_argsort_stability(self, size):
        # NaT < NaT should be False internally for
        # sort stability
        # 测试时间间隔 NaT 的 argsort 稳定性，确保 NaT < NaT 返回 False
        expected = np.arange(size)
        arr = np.tile(np.timedelta64('NaT'), size)
        assert_equal(np.argsort(arr, kind='mergesort'), expected)
    @pytest.mark.parametrize("arr, expected", [
        # 使用 pytest 的 parametrize 装饰器定义多组输入和期望输出的参数
        # 示例 gh-12629
        (['NaT', 1, 2, 3],
         [1, 2, 3, 'NaT']),
        # 包含多个 NaT 的情况
        (['NaT', 9, 'NaT', -707],
         [-707, 9, 'NaT', 'NaT']),
        # 探索另一种 NaT 排序的代码路径
        ([1, -2, 3, 'NaT'],
         [-2, 1, 3, 'NaT']),
        # 二维数组
        ([[51, -220, 'NaT'],
          [-17, 'NaT', -90]],
         [[-220, 51, 'NaT'],
          [-90, -17, 'NaT']]),
    ])
    @pytest.mark.parametrize("dtype", [
        'M8[ns]', 'M8[us]',
        'm8[ns]', 'm8[us]'])
    def test_datetime_timedelta_sort_nat(self, arr, expected, dtype):
        # 修复 gh-12629 和 gh-15063; 将 NaT 排序到数组末尾
        arr = np.array(arr, dtype=dtype)
        expected = np.array(expected, dtype=dtype)
        arr.sort()
        assert_equal(arr, expected)

    def test_datetime_scalar_construction_timezone(self):
        # 提供关于 np.datetime64 显式时区的警告验证
        msg = "no explicit representation of timezones available for " \
              "np.datetime64"
        with pytest.warns(UserWarning, match=msg):
            assert_equal(np.datetime64('2000-01-01T00Z'),
                         np.datetime64('2000-01-01T00'))
        with pytest.warns(UserWarning, match=msg):
            assert_equal(np.datetime64('2000-01-01T00-08'),
                         np.datetime64('2000-01-01T08'))

    def test_datetime_array_find_type(self):
        dt = np.datetime64('1970-01-01', 'M')
        arr = np.array([dt])
        assert_equal(arr.dtype, np.dtype('M8[M]'))

        # 目前，我们不自动将这些转换为 datetime64

        dt = datetime.date(1970, 1, 1)
        arr = np.array([dt])
        assert_equal(arr.dtype, np.dtype('O'))

        dt = datetime.datetime(1970, 1, 1, 12, 30, 40)
        arr = np.array([dt])
        assert_equal(arr.dtype, np.dtype('O'))

        # 查找非日期和日期的“超类型”

        b = np.bool(True)
        dm = np.datetime64('1970-01-01', 'M')
        d = datetime.date(1970, 1, 1)
        dt = datetime.datetime(1970, 1, 1, 12, 30, 40)

        arr = np.array([b, dm])
        assert_equal(arr.dtype, np.dtype('O'))

        arr = np.array([b, d])
        assert_equal(arr.dtype, np.dtype('O'))

        arr = np.array([b, dt])
        assert_equal(arr.dtype, np.dtype('O'))

        arr = np.array([d, d]).astype('datetime64')
        assert_equal(arr.dtype, np.dtype('M8[D]'))

        arr = np.array([dt, dt]).astype('datetime64')
        assert_equal(arr.dtype, np.dtype('M8[us]'))

    @pytest.mark.parametrize("unit", [
    # 测试所有日期/时间单位并使用
    # "generic" 选择通用单位
    ("Y"), ("M"), ("W"), ("D"), ("h"), ("m"),
    ("s"), ("ms"), ("us"), ("ns"), ("ps"),
    ("fs"), ("as"), ("generic") ])
    # 定义一个测试方法，用于检验 np.timedelta64 和 np.int64 的构造方式是否正常
    def test_timedelta_np_int_construction(self, unit):
        # 用于回归测试 gh-7617
        # 如果单位不是 "generic"，则断言 np.timedelta64(np.int64(123), unit) 等于 np.timedelta64(123, unit)
        if unit != "generic":
            assert_equal(np.timedelta64(np.int64(123), unit),
                         np.timedelta64(123, unit))
        else:
            # 如果单位是 "generic"，则断言 np.timedelta64(np.int64(123)) 等于 np.timedelta64(123)
            assert_equal(np.timedelta64(np.int64(123)),
                         np.timedelta64(123))

    # 定义一个测试方法，检验 datetime.timedelta 对象数组向 np.timedelta64[D] 类型的数组转换是否正常
    def test_timedelta_object_array_conversion(self):
        # 回归测试 gh-11096
        # 创建包含 datetime.timedelta 对象的列表作为输入
        inputs = [datetime.timedelta(28),
                  datetime.timedelta(30),
                  datetime.timedelta(31)]
        # 创建预期的 np.timedelta64[D] 类型的数组
        expected = np.array([28, 30, 31], dtype='timedelta64[D]')
        # 将输入的 datetime.timedelta 对象数组转换为 np.timedelta64[D] 类型的数组
        actual = np.array(inputs, dtype='timedelta64[D]')
        # 断言转换后的数组与预期数组相等
        assert_equal(expected, actual)

    # 定义一个测试方法，检验将 datetime.timedelta 对象转换为 np.timedelta64 类型的数组是否正常
    def test_timedelta_0_dim_object_array_conversion(self):
        # 回归测试 gh-11151
        # 创建包含 datetime.timedelta(seconds=20) 的 np.ndarray 对象
        test = np.array(datetime.timedelta(seconds=20))
        # 将 test 数组的元素转换为 np.timedelta64 类型的元素
        actual = test.astype(np.timedelta64)
        # 使用数组构造器的 workaround 描述的方式创建预期值
        expected = np.array(datetime.timedelta(seconds=20),
                            np.timedelta64)
        # 断言转换后的数组与预期数组相等
        assert_equal(actual, expected)

    # 定义一个测试方法，检验 np.timedelta64('nat') 的格式化输出是否为 'NaT'
    def test_timedelta_nat_format(self):
        # gh-17552
        # 断言 np.timedelta64('nat') 的格式化输出是否为 'NaT'
        assert_equal('NaT', '{0}'.format(np.timedelta64('nat')))
    def test_datetime_nat_casting(self):
        # 创建一个包含 'NaT' 的 NumPy 数组，数据类型为 'M8[D]'
        a = np.array('NaT', dtype='M8[D]')
        # 创建一个 'NaT' 的 NumPy datetime64 标量，精度为 '[D]'

        b = np.datetime64('NaT', '[D]')

        # 数组操作
        assert_equal(a.astype('M8[s]'), np.array('NaT', dtype='M8[s]'))
        # 将数组 a 转换为秒精度的 datetime64 数组，并与预期结果进行比较
        assert_equal(a.astype('M8[ms]'), np.array('NaT', dtype='M8[ms]'))
        # 将数组 a 转换为毫秒精度的 datetime64 数组，并与预期结果进行比较
        assert_equal(a.astype('M8[M]'), np.array('NaT', dtype='M8[M]'))
        # 将数组 a 转换为月精度的 datetime64 数组，并与预期结果进行比较
        assert_equal(a.astype('M8[Y]'), np.array('NaT', dtype='M8[Y]'))
        # 将数组 a 转换为年精度的 datetime64 数组，并与预期结果进行比较
        assert_equal(a.astype('M8[W]'), np.array('NaT', dtype='M8[W]'))
        # 将数组 a 转换为周精度的 datetime64 数组，并与预期结果进行比较

        # 标量到标量的操作
        assert_equal(np.datetime64(b, '[s]'), np.datetime64('NaT', '[s]'))
        # 将标量 b 转换为秒精度的 datetime64 标量，并与预期结果进行比较
        assert_equal(np.datetime64(b, '[ms]'), np.datetime64('NaT', '[ms]'))
        # 将标量 b 转换为毫秒精度的 datetime64 标量，并与预期结果进行比较
        assert_equal(np.datetime64(b, '[M]'), np.datetime64('NaT', '[M]'))
        # 将标量 b 转换为月精度的 datetime64 标量，并与预期结果进行比较
        assert_equal(np.datetime64(b, '[Y]'), np.datetime64('NaT', '[Y]'))
        # 将标量 b 转换为年精度的 datetime64 标量，并与预期结果进行比较
        assert_equal(np.datetime64(b, '[W]'), np.datetime64('NaT', '[W]'))
        # 将标量 b 转换为周精度的 datetime64 标量，并与预期结果进行比较

        # 数组到标量的操作
        assert_equal(np.datetime64(a, '[s]'), np.datetime64('NaT', '[s]'))
        # 将数组 a 转换为秒精度的 datetime64 标量，并与预期结果进行比较
        assert_equal(np.datetime64(a, '[ms]'), np.datetime64('NaT', '[ms]'))
        # 将数组 a 转换为毫秒精度的 datetime64 标量，并与预期结果进行比较
        assert_equal(np.datetime64(a, '[M]'), np.datetime64('NaT', '[M]'))
        # 将数组 a 转换为月精度的 datetime64 标量，并与预期结果进行比较
        assert_equal(np.datetime64(a, '[Y]'), np.datetime64('NaT', '[Y]'))
        # 将数组 a 转换为年精度的 datetime64 标量，并与预期结果进行比较
        assert_equal(np.datetime64(a, '[W]'), np.datetime64('NaT', '[W]'))
        # 将数组 a 转换为周精度的 datetime64 标量，并与预期结果进行比较

        # NaN 转换为 NaT
        nan = np.array([np.nan] * 8 + [0])
        # 创建一个包含 NaN 的 NumPy 数组
        fnan = nan.astype('f')
        # 将数组 nan 转换为单精度浮点数数组
        lnan = nan.astype('g')
        # 将数组 nan 转换为长整型数组
        cnan = nan.astype('D')
        # 将数组 nan 转换为复数型数组
        cfnan = nan.astype('F')
        # 将数组 nan 转换为复数型数组
        clnan = nan.astype('G')
        # 将数组 nan 转换为长复数型数组
        hnan = nan.astype(np.half)
        # 将数组 nan 转换为半精度浮点数数组

        nat = np.array([np.datetime64('NaT')] * 8 + [np.datetime64(0, 'D')])
        # 创建一个包含 'NaT' 和 '1970-01-01' 的 NumPy datetime64 数组
        assert_equal(nan.astype('M8[ns]'), nat)
        # 将数组 nan 转换为纳秒精度的 datetime64 数组，并与预期结果进行比较
        assert_equal(fnan.astype('M8[ns]'), nat)
        # 将数组 fnan 转换为纳秒精度的 datetime64 数组，并与预期结果进行比较
        assert_equal(lnan.astype('M8[ns]'), nat)
        # 将数组 lnan 转换为纳秒精度的 datetime64 数组，并与预期结果进行比较
        assert_equal(cnan.astype('M8[ns]'), nat)
        # 将数组 cnan 转换为纳秒精度的 datetime64 数组，并与预期结果进行比较
        assert_equal(cfnan.astype('M8[ns]'), nat)
        # 将数组 cfnan 转换为纳秒精度的 datetime64 数组，并与预期结果进行比较
        assert_equal(clnan.astype('M8[ns]'), nat)
        # 将数组 clnan 转换为纳秒精度的 datetime64 数组，并与预期结果进行比较
        assert_equal(hnan.astype('M8[ns]'), nat)
        # 将数组 hnan 转换为纳秒精度的 datetime64 数组，并与预期结果进行比较

        nat = np.array([np.timedelta64('NaT')] * 8 + [np.timedelta64(0)])
        # 创建一个包含 'NaT' 和 '0' 的 NumPy timedelta64 数组
        assert_equal(nan.astype('timedelta64[ns]'), nat)
        # 将数组 nan 转换为纳秒精度的 timedelta64 数组，并与预期结果进行比较
        assert_equal(fnan.astype('timedelta64[ns]'), nat)
        # 将数组 fnan 转换为纳秒精度的 timedelta64 数组，并与预期结果进行比较
        assert_equal(lnan.astype('timedelta64[ns]'), nat)
        # 将数组 lnan 转换为纳秒精度的 timedelta64 数组，并与预期结果进行比较
        assert_equal(cnan.astype('timedelta64[ns]'), nat)
        # 将数组 cnan 转换为纳秒精度的 timedelta64 数组，并与预期结果进行比较
        assert_equal(cfnan.astype('timedelta64[ns]'), nat)
        # 将数组 cfnan 转换为纳秒精度的 timedelta64 数组，并与预期结果进行比较
        assert_equal(clnan.astype('timedelta64[ns]'), nat)
        # 将数组 clnan 转换为纳秒精度的 timedelta64 数组，并与预期结果进行比较
        assert_equal(hnan.astype('timedelta64[ns]'), nat)
        # 将数组 hnan 转换为纳秒精度的 timedelta64 数组，并与预期结果进行比较
    # 定义一个测试方法，用于验证日期创建的功能

        # 断言：对于日期 '1599-01-01'，计算从1970年到该日期经过的天数
        assert_equal(np.array('1599', dtype='M8[D]').astype('i8'),
                (1600-1970)*365 - (1972-1600)/4 + 3 - 365)
        
        # 断言：对于日期 '1600-01-01'，计算从1970年到该日期经过的天数
        assert_equal(np.array('1600', dtype='M8[D]').astype('i8'),
                (1600-1970)*365 - (1972-1600)/4 + 3)
        
        # 断言：对于日期 '1601-01-01'，计算从1970年到该日期经过的天数
        assert_equal(np.array('1601', dtype='M8[D]').astype('i8'),
                (1600-1970)*365 - (1972-1600)/4 + 3 + 366)
        
        # 断言：对于日期 '1900-01-01'，计算从1970年到该日期经过的天数
        assert_equal(np.array('1900', dtype='M8[D]').astype('i8'),
                (1900-1970)*365 - (1970-1900)//4)
        
        # 断言：对于日期 '1901-01-01'，计算从1970年到该日期经过的天数
        assert_equal(np.array('1901', dtype='M8[D]').astype('i8'),
                (1900-1970)*365 - (1970-1900)//4 + 365)
        
        # 断言：对于日期 '1967-01-01'，计算从1970年到该日期经过的天数
        assert_equal(np.array('1967', dtype='M8[D]').astype('i8'), -3*365 - 1)
        
        # 断言：对于日期 '1968-01-01'，计算从1970年到该日期经过的天数
        assert_equal(np.array('1968', dtype='M8[D]').astype('i8'), -2*365 - 1)
        
        # 断言：对于日期 '1969-01-01'，计算从1970年到该日期经过的天数
        assert_equal(np.array('1969', dtype='M8[D]').astype('i8'), -1*365)
        
        # 断言：对于日期 '1970-01-01'，计算从1970年到该日期经过的天数
        assert_equal(np.array('1970', dtype='M8[D]').astype('i8'), 0*365)
        
        # 断言：对于日期 '1971-01-01'，计算从1970年到该日期经过的天数
        assert_equal(np.array('1971', dtype='M8[D]').astype('i8'), 1*365)
        
        # 断言：对于日期 '1972-01-01'，计算从1970年到该日期经过的天数
        assert_equal(np.array('1972', dtype='M8[D]').astype('i8'), 2*365)
        
        # 断言：对于日期 '1973-01-01'，计算从1970年到该日期经过的天数
        assert_equal(np.array('1973', dtype='M8[D]').astype('i8'), 3*365 + 1)
        
        # 断言：对于日期 '1974-01-01'，计算从1970年到该日期经过的天数
        assert_equal(np.array('1974', dtype='M8[D]').astype('i8'), 4*365 + 1)
        
        # 断言：对于日期 '2000-01-01'，计算从1970年到该日期经过的天数
        assert_equal(np.array('2000', dtype='M8[D]').astype('i8'),
                 (2000 - 1970)*365 + (2000 - 1972)//4)
        
        # 断言：对于日期 '2001-01-01'，计算从1970年到该日期经过的天数
        assert_equal(np.array('2001', dtype='M8[D]').astype('i8'),
                 (2000 - 1970)*365 + (2000 - 1972)//4 + 366)
        
        # 断言：对于日期 '2400-01-01'，计算从1970年到该日期经过的天数
        assert_equal(np.array('2400', dtype='M8[D]').astype('i8'),
                 (2400 - 1970)*365 + (2400 - 1972)//4 - 3)
        
        # 断言：对于日期 '2401-01-01'，计算从1970年到该日期经过的天数
        assert_equal(np.array('2401', dtype='M8[D]').astype('i8'),
                 (2400 - 1970)*365 + (2400 - 1972)//4 - 3 + 366)

        # 断言：对于日期 '1600-02-29'，计算从1970年到该日期经过的天数
        assert_equal(np.array('1600-02-29', dtype='M8[D]').astype('i8'),
                (1600-1970)*365 - (1972-1600)//4 + 3 + 31 + 28)
        
        # 断言：对于日期 '1600-03-01'，计算从1970年到该日期经过的天数
        assert_equal(np.array('1600-03-01', dtype='M8[D]').astype('i8'),
                (1600-1970)*365 - (1972-1600)//4 + 3 + 31 + 29)
        
        # 断言：对于日期 '2000-02-29'，计算从1970年到该日期经过的天数
        assert_equal(np.array('2000-02-29', dtype='M8[D]').astype('i8'),
                 (2000 - 1970)*365 + (2000 - 1972)//4 + 31 + 28)
        
        # 断言：对于日期 '2000-03-01'，计算从1970年到该日期经过的天数
        assert_equal(np.array('2000-03-01', dtype='M8[D]').astype('i8'),
                 (2000 - 1970)*365 + (2000 - 1972)//4 + 31 + 29)
        
        # 断言：对于日期 '2001-03-22'，计算从1970年到该日期经过的天数
        assert_equal(np.array('2001-03-22', dtype='M8[D]').astype('i8'),
                 (2000 - 1970)*365 + (2000 - 1972)//4 + 366 + 31 + 28 + 21)
    # 测试函数：将字符串表示的日期转换为 datetime.date 对象，并与指定的日期进行比较
    def test_days_to_pydate(self):
        # 断言：将字符串 '1599' 转换为 numpy datetime 数组，再转换为 Python datetime.date 对象，并比较是否等于指定日期
        assert_equal(np.array('1599', dtype='M8[D]').astype('O'),
                     datetime.date(1599, 1, 1))
        assert_equal(np.array('1600', dtype='M8[D]').astype('O'),
                     datetime.date(1600, 1, 1))
        assert_equal(np.array('1601', dtype='M8[D]').astype('O'),
                     datetime.date(1601, 1, 1))
        assert_equal(np.array('1900', dtype='M8[D]').astype('O'),
                     datetime.date(1900, 1, 1))
        assert_equal(np.array('1901', dtype='M8[D]').astype('O'),
                     datetime.date(1901, 1, 1))
        assert_equal(np.array('2000', dtype='M8[D]').astype('O'),
                     datetime.date(2000, 1, 1))
        assert_equal(np.array('2001', dtype='M8[D]').astype('O'),
                     datetime.date(2001, 1, 1))
        assert_equal(np.array('1600-02-29', dtype='M8[D]').astype('O'),
                     datetime.date(1600, 2, 29))
        assert_equal(np.array('1600-03-01', dtype='M8[D]').astype('O'),
                     datetime.date(1600, 3, 1))
        assert_equal(np.array('2001-03-22', dtype='M8[D]').astype('O'),
                     datetime.date(2001, 3, 22))

    # 测试函数：比较不同的 numpy datetime 类型是否相等
    def test_dtype_comparison(self):
        assert_(not (np.dtype('M8[us]') == np.dtype('M8[ms]')))
        assert_(np.dtype('M8[us]') != np.dtype('M8[ms]'))
        assert_(np.dtype('M8[2D]') != np.dtype('M8[D]'))
        assert_(np.dtype('M8[D]') != np.dtype('M8[2D]'))

    # 测试函数：创建 numpy datetime 数组并与 Python datetime.date 对象进行比较
    def test_pydatetime_creation(self):
        a = np.array(['1960-03-12', datetime.date(1960, 3, 12)], dtype='M8[D]')
        assert_equal(a[0], a[1])
        a = np.array(['1999-12-31', datetime.date(1999, 12, 31)], dtype='M8[D]')
        assert_equal(a[0], a[1])
        a = np.array(['2000-01-01', datetime.date(2000, 1, 1)], dtype='M8[D]')
        assert_equal(a[0], a[1])
        # 如果日期在刚好转换时发生变化，则断言将会失败
        a = np.array(['today', datetime.date.today()], dtype='M8[D]')
        assert_equal(a[0], a[1])
        # datetime.datetime.now() 返回本地时间，不是 UTC 时间
        #a = np.array(['now', datetime.datetime.now()], dtype='M8[s]')
        #assert_equal(a[0], a[1])

        # 可以将 datetime.date 对象转换为 M8[s] 时间单位的 numpy datetime 数组
        assert_equal(np.array(datetime.date(1960, 3, 12), dtype='M8[s]'),
                     np.array(np.datetime64('1960-03-12T00:00:00')))
    # 定义一个测试方法，用于测试日期时间字符串转换的功能
    def test_datetime_string_conversion(self):
        # 创建包含日期字符串的列表
        a = ['2011-03-16', '1920-01-01', '2013-05-19']
        # 创建一个包含这些日期字符串的 NumPy 数组，数据类型为字节串 'S'
        str_a = np.array(a, dtype='S')
        # 创建一个包含这些日期字符串的 NumPy 数组，数据类型为 Unicode 字符串 'U'
        uni_a = np.array(a, dtype='U')
        # 创建一个包含这些日期字符串的 NumPy 数组，数据类型为日期时间 'M'
        dt_a = np.array(a, dtype='M')

        # 测试：字符串转换为日期时间
        assert_equal(dt_a, str_a.astype('M'))
        assert_equal(dt_a.dtype, str_a.astype('M').dtype)
        # 创建一个与 dt_a 相同形状的空数组 dt_b，并将 str_a 中的数据复制到 dt_b
        dt_b = np.empty_like(dt_a)
        dt_b[...] = str_a
        assert_equal(dt_a, dt_b)

        # 测试：日期时间转换为字符串
        assert_equal(str_a, dt_a.astype('S0'))
        # 创建一个与 str_a 相同形状的空数组 str_b，并将 dt_a 中的数据复制到 str_b
        str_b = np.empty_like(str_a)
        str_b[...] = dt_a
        assert_equal(str_a, str_b)

        # 测试：Unicode 转换为日期时间
        assert_equal(dt_a, uni_a.astype('M'))
        assert_equal(dt_a.dtype, uni_a.astype('M').dtype)
        # 创建一个与 dt_a 相同形状的空数组 dt_b，并将 uni_a 中的数据复制到 dt_b
        dt_b = np.empty_like(dt_a)
        dt_b[...] = uni_a
        assert_equal(dt_a, dt_b)

        # 测试：日期时间转换为 Unicode
        assert_equal(uni_a, dt_a.astype('U'))
        # 创建一个与 uni_a 相同形状的空数组 uni_b，并将 dt_a 中的数据复制到 uni_b
        uni_b = np.empty_like(uni_a)
        uni_b[...] = dt_a
        assert_equal(uni_a, uni_b)

        # 测试：日期时间转换为长字符串（gh-9712）
        assert_equal(str_a, dt_a.astype((np.bytes_, 128)))
        # 创建一个数据类型为 (np.bytes_, 128) 的空数组 str_b，与 str_a 的形状相同，并将 dt_a 中的数据复制到 str_b
        str_b = np.empty(str_a.shape, dtype=(np.bytes_, 128))
        str_b[...] = dt_a
        assert_equal(str_a, str_b)
    def test_datetime_conversions_byteorders(self, str_dtype, time_dtype):
        # 创建一个包含时间字符串和NaT（Not a Time）的NumPy数组，使用指定的数据类型
        times = np.array(["2017", "NaT"], dtype=time_dtype)
        # 不幸的是，时间间隔无法往返转换：
        # 创建一个包含字符串日期和NaT的NumPy数组，使用指定的数据类型
        from_strings = np.array(["2017", "NaT"], dtype=str_dtype)
        # 假设这是正确的，将时间数组转换为指定数据类型的字符串数组
        to_strings = times.astype(str_dtype)

        # 检查如果源数组已经交换字节顺序，从时间到字符串的转换是否工作：
        times_swapped = times.astype(times.dtype.newbyteorder())
        res = times_swapped.astype(str_dtype)
        # 断言结果数组与目标字符串数组相等
        assert_array_equal(res, to_strings)

        # 检查如果源数组和目标字符串数组都交换了字节顺序，从时间到字符串的转换是否工作：
        res = times_swapped.astype(to_strings.dtype.newbyteorder())
        assert_array_equal(res, to_strings)

        # 检查只有目标字符串数组交换了字节顺序，从时间到字符串的转换是否工作：
        res = times.astype(to_strings.dtype.newbyteorder())
        assert_array_equal(res, to_strings)

        # 检查如果源字符串数组已经交换字节顺序，从字符串到时间的转换是否工作：
        from_strings_swapped = from_strings.astype(from_strings.dtype.newbyteorder())
        res = from_strings_swapped.astype(time_dtype)
        assert_array_equal(res, times)

        # 检查如果源字符串数组和时间数组都交换了字节顺序，从字符串到时间的转换是否工作：
        res = from_strings_swapped.astype(times.dtype.newbyteorder())
        assert_array_equal(res, times)

        # 检查只有时间数组交换了字节顺序，从字符串到时间的转换是否工作：
        res = from_strings.astype(times.dtype.newbyteorder())
        assert_array_equal(res, times)

    def test_datetime_array_str(self):
        # 创建一个包含日期字符串的NumPy数组，使用'M'（datetime64）数据类型
        a = np.array(['2011-03-16', '1920-01-01', '2013-05-19'], dtype='M')
        # 断言将数组转换为字符串后结果是否符合预期格式
        assert_equal(str(a), "['2011-03-16' '1920-01-01' '2013-05-19']")

        # 创建一个包含带时间的日期字符串的NumPy数组，使用'M'（datetime64）数据类型
        a = np.array(['2011-03-16T13:55', '1920-01-01T03:12'], dtype='M')
        # 断言将数组转换为字符串后结果是否符合预期格式，使用自定义格式化器
        assert_equal(np.array2string(a, separator=', ',
                    formatter={'datetime': lambda x:
                            "'%s'" % np.datetime_as_string(x, timezone='UTC')}),
                     "['2011-03-16T13:55Z', '1920-01-01T03:12Z']")

        # 检查当数组中存在NaT时，确保后续条目不受其影响
        a = np.array(['2010', 'NaT', '2030']).astype('M')
        # 断言将数组转换为字符串后结果是否符合预期格式
        assert_equal(str(a), "['2010'  'NaT' '2030']")

    def test_timedelta_array_str(self):
        # 创建一个包含负数、零和正数的NumPy数组，使用'm'（timedelta64）数据类型
        a = np.array([-1, 0, 100], dtype='m')
        # 断言将数组转换为字符串后结果是否符合预期格式
        assert_equal(str(a), "[ -1   0 100]")
        
        # 创建一个包含NaT的NumPy数组，使用'm'（timedelta64）数据类型
        a = np.array(['NaT', 'NaT'], dtype='m')
        # 断言将数组转换为字符串后结果是否符合预期格式
        assert_equal(str(a), "['NaT' 'NaT']")

        # 检查当NaT右对齐时的情况
        a = np.array([-1, 'NaT', 0], dtype='m')
        # 断言将数组转换为字符串后结果是否符合预期格式
        assert_equal(str(a), "[   -1 'NaT'     0]")
        
        # 创建一个包含负数、NaT和正数的NumPy数组，使用'm'（timedelta64）数据类型
        a = np.array([-1, 'NaT', 1234567], dtype='m')
        # 断言将数组转换为字符串后结果是否符合预期格式
        assert_equal(str(a), "[     -1   'NaT' 1234567]")

        # 测试使用其他字节顺序的情况
        a = np.array([-1, 'NaT', 1234567], dtype='>m')
        # 断言将数组转换为字符串后结果是否符合预期格式
        assert_equal(str(a), "[     -1   'NaT' 1234567]")
        
        # 测试使用其他字节顺序的情况
        a = np.array([-1, 'NaT', 1234567], dtype='<m')
        # 断言将数组转换为字符串后结果是否符合预期格式
        assert_equal(str(a), "[     -1   'NaT' 1234567]")
    # 测试 pickle 序列化和反序列化是否正常工作
    def test_pickle(self):
        # 检查不同协议版本下的 pickle 序列化和反序列化
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            # 创建日期时间类型的 NumPy 数组
            dt = np.dtype('M8[7D]')
            # 断言 pickle 序列化后再反序列化仍然保持不变
            assert_equal(pickle.loads(pickle.dumps(dt, protocol=proto)), dt)
            
            # 创建周时间类型的 NumPy 数组
            dt = np.dtype('M8[W]')
            # 断言 pickle 序列化后再反序列化仍然保持不变
            assert_equal(pickle.loads(pickle.dumps(dt, protocol=proto)), dt)
            
            # 创建标量日期时间对象
            scalar = np.datetime64('2016-01-01T00:00:00.000000000')
            # 断言 pickle 序列化后再反序列化仍然保持不变
            assert_equal(pickle.loads(pickle.dumps(scalar, protocol=proto)), scalar)
            
            # 计算日期时间对象与另一日期时间对象之间的时间差
            delta = scalar - np.datetime64('2015-01-01T00:00:00.000000000')
            # 断言 pickle 序列化后再反序列化仍然保持不变
            assert_equal(pickle.loads(pickle.dumps(delta, protocol=proto)), delta)

        # 检查从旧版本（1.6）pickle 加载是否正常工作
        pkl = b"cnumpy\ndtype\np0\n(S'M8'\np1\nI0\nI1\ntp2\nRp3\n" + \
              b"(I4\nS'<'\np4\nNNNI-1\nI-1\nI0\n((dp5\n(S'D'\np6\n" + \
              b"I7\nI1\nI1\ntp7\ntp8\ntp9\nb."
        assert_equal(pickle.loads(pkl), np.dtype('<M8[7D]'))
        
        pkl = b"cnumpy\ndtype\np0\n(S'M8'\np1\nI0\nI1\ntp2\nRp3\n" + \
              b"(I4\nS'<'\np4\nNNNI-1\nI-1\nI0\n((dp5\n(S'W'\np6\n" + \
              b"I1\nI1\nI1\ntp7\ntp8\ntp9\nb."
        assert_equal(pickle.loads(pkl), np.dtype('<M8[W]'))
        
        pkl = b"cnumpy\ndtype\np0\n(S'M8'\np1\nI0\nI1\ntp2\nRp3\n" + \
              b"(I4\nS'>'\np4\nNNNI-1\nI-1\nI0\n((dp5\n(S'us'\np6\n" + \
              b"I1\nI1\nI1\ntp7\ntp8\ntp9\nb."
        assert_equal(pickle.loads(pkl), np.dtype('>M8[us]'))

    # 验证 datetime 数据类型的 __setstate__ 方法是否能处理错误参数
    def test_setstate(self):
        dt = np.dtype('>M8[us]')
        # 断言传入错误参数时能够抛出 ValueError 异常
        assert_raises(ValueError, dt.__setstate__, (4, '>', None, None, None, -1, -1, 0, 1))
        # 断言 __reduce__ 方法返回的信息与预期一致
        assert_(dt.__reduce__()[2] == np.dtype('>M8[us]').__reduce__()[2])
        # 断言传入错误参数时能够抛出 TypeError 异常
        assert_raises(TypeError, dt.__setstate__, (4, '>', None, None, None, -1, -1, 0, ({}, 'xxx')))
        # 断言 __reduce__ 方法返回的信息与预期一致
        assert_(dt.__reduce__()[2] == np.dtype('>M8[us]').__reduce__()[2])
    # 测试数据类型的提升规则
    def test_dtype_promotion(self):
        # datetime <op> datetime 计算元数据的最大公约数
        # timedelta <op> timedelta 计算元数据的最大公约数
        for mM in ['m', 'M']:
            # 检查同一类型的 datetime 数据类型提升规则
            assert_equal(
                np.promote_types(np.dtype(mM+'8[2Y]'), np.dtype(mM+'8[2Y]')),
                np.dtype(mM+'8[2Y]'))
            # 检查不同年份单位的 datetime 数据类型提升规则
            assert_equal(
                np.promote_types(np.dtype(mM+'8[12Y]'), np.dtype(mM+'8[15Y]')),
                np.dtype(mM+'8[3Y]'))
            # 检查不同月份单位的 datetime 数据类型提升规则
            assert_equal(
                np.promote_types(np.dtype(mM+'8[62M]'), np.dtype(mM+'8[24M]')),
                np.dtype(mM+'8[2M]'))
            # 检查周和日之间的 timedelta 数据类型提升规则
            assert_equal(
                np.promote_types(np.dtype(mM+'8[1W]'), np.dtype(mM+'8[2D]')),
                np.dtype(mM+'8[1D]'))
            # 检查周和秒之间的 timedelta 数据类型提升规则
            assert_equal(
                np.promote_types(np.dtype(mM+'8[W]'), np.dtype(mM+'8[13s]')),
                np.dtype(mM+'8[s]'))
            # 检查周和秒之间的 timedelta 数据类型提升规则
            assert_equal(
                np.promote_types(np.dtype(mM+'8[13W]'), np.dtype(mM+'8[49s]')),
                np.dtype(mM+'8[7s]'))
        # 检查没有合理的最大公约数时 timedelta <op> timedelta 的异常情况
        assert_raises(TypeError, np.promote_types,
                            np.dtype('m8[Y]'), np.dtype('m8[D]'))
        assert_raises(TypeError, np.promote_types,
                            np.dtype('m8[M]'), np.dtype('m8[W]'))
        # 检查 timedelta 和 float 之间无法安全转换的情况
        assert_raises(TypeError, np.promote_types, "float32", "m8")
        assert_raises(TypeError, np.promote_types, "m8", "float32")
        assert_raises(TypeError, np.promote_types, "uint64", "m8")
        assert_raises(TypeError, np.promote_types, "m8", "uint64")

        # 检查大单位范围下 timedelta <op> timedelta 可能溢出的情况
        assert_raises(OverflowError, np.promote_types,
                            np.dtype('m8[W]'), np.dtype('m8[fs]'))
        assert_raises(OverflowError, np.promote_types,
                            np.dtype('m8[s]'), np.dtype('m8[as]'))

    # 测试类型转换时的溢出情况
    def test_cast_overflow(self):
        # gh-4486: 检查 datetime64 到较小时间单位的转换是否会溢出
        def cast():
            numpy.datetime64("1971-01-01 00:00:00.000000000000000").astype("<M8[D]")
        assert_raises(OverflowError, cast)

        # 检查将年份转换为更小单位的情况是否会溢出
        def cast2():
            numpy.datetime64("2014").astype("<M8[fs]")
        assert_raises(OverflowError, cast2)
    def test_pyobject_roundtrip(self):
        # 测试函数：test_pyobject_roundtrip，用于测试各种日期时间类型通过对象类型的往返转换

        # 创建一个包含多种日期时间值的 NumPy 数组，以 np.int64 类型存储
        a = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0,
                      -1020040340, -2942398, -1, 0, 1, 234523453, 1199164176],
                     dtype=np.int64)

        # 使用不同的日期单位进行循环测试
        for unit in ['M8[D]', 'M8[W]', 'M8[M]', 'M8[Y]']:
            # 复制数组 a 并按指定日期单位进行视图转换
            b = a.copy().view(dtype=unit)

            # 设置不同的日期值
            b[0] = '-0001-01-01'
            b[1] = '-0001-12-31'
            b[2] = '0000-01-01'
            b[3] = '0001-01-01'
            b[4] = '1969-12-31'
            b[5] = '1970-01-01'
            b[6] = '9999-12-31'
            b[7] = '10000-01-01'
            b[8] = 'NaT'

            # 断言往返转换后的结果与原始值相等
            assert_equal(b.astype(object).astype(unit), b,
                         "Error roundtripping unit %s" % unit)

        # 使用不同的时间单位进行循环测试
        for unit in ['M8[as]', 'M8[16fs]', 'M8[ps]', 'M8[us]',
                     'M8[300as]', 'M8[20us]']:
            # 复制数组 a 并按指定时间单位进行视图转换
            b = a.copy().view(dtype=unit)

            # 设置不同的时间值
            b[0] = '-0001-01-01T00'
            b[1] = '-0001-12-31T00'
            b[2] = '0000-01-01T00'
            b[3] = '0001-01-01T00'
            b[4] = '1969-12-31T23:59:59.999999'
            b[5] = '1970-01-01T00'
            b[6] = '9999-12-31T23:59:59.999999'
            b[7] = '10000-01-01T00'
            b[8] = 'NaT'

            # 断言往返转换后的结果与原始值相等
            assert_equal(b.astype(object).astype(unit), b,
                         "Error roundtripping unit %s" % unit)


    def test_month_truncation(self):
        # 测试函数：test_month_truncation，用于验证月份截断是否正确

        # 断言两个相同月份的日期数组相等
        assert_equal(np.array('1945-03-01', dtype='M8[M]'),
                     np.array('1945-03-31', dtype='M8[M]'))

        # 断言截断到月份的日期与对应的月末日期相等
        assert_equal(np.array('1969-11-01', dtype='M8[M]'),
                     np.array('1969-11-30T23:59:59.99999', dtype='M').astype('M8[M]'))

        # 断言截断到月份的日期与对应的月末日期相等
        assert_equal(np.array('1969-12-01', dtype='M8[M]'),
                     np.array('1969-12-31T23:59:59.99999', dtype='M').astype('M8[M]'))

        # 断言截断到月份的日期与对应的月末日期相等
        assert_equal(np.array('1970-01-01', dtype='M8[M]'),
                     np.array('1970-01-31T23:59:59.99999', dtype='M').astype('M8[M]'))

        # 断言截断到月份的日期与对应的月末日期相等
        assert_equal(np.array('1980-02-01', dtype='M8[M]'),
                     np.array('1980-02-29T23:59:59.99999', dtype='M').astype('M8[M]'))


    def test_datetime_like(self):
        # 测试函数：test_datetime_like，用于验证 np.ones_like、np.zeros_like 和 np.empty_like 的行为

        # 创建一个包含单个日期时间值的 NumPy 数组，以 'm8[4D]' 类型存储
        a = np.array([3], dtype='m8[4D]')

        # 创建一个包含单个日期值的 NumPy 数组，以 'M8[D]' 类型存储
        b = np.array(['2012-12-21'], dtype='M8[D]')

        # 断言 np.ones_like(a) 的数据类型与 a 相同
        assert_equal(np.ones_like(a).dtype, a.dtype)

        # 断言 np.zeros_like(a) 的数据类型与 a 相同
        assert_equal(np.zeros_like(a).dtype, a.dtype)

        # 断言 np.empty_like(a) 的数据类型与 a 相同
        assert_equal(np.empty_like(a).dtype, a.dtype)

        # 断言 np.ones_like(b) 的数据类型与 b 相同
        assert_equal(np.ones_like(b).dtype, b.dtype)

        # 断言 np.zeros_like(b) 的数据类型与 b 相同
        assert_equal(np.zeros_like(b).dtype, b.dtype)

        # 断言 np.empty_like(b) 的数据类型与 b 相同
        assert_equal(np.empty_like(b).dtype, b.dtype)
    def test_datetime_unary(self):
        for tda, tdb, tdzero, tdone, tdmone in \
                [
                 # One-dimensional arrays
                 (np.array([3], dtype='m8[D]'),
                  np.array([-3], dtype='m8[D]'),
                  np.array([0], dtype='m8[D]'),
                  np.array([1], dtype='m8[D]'),
                  np.array([-1], dtype='m8[D]')),
                 # NumPy scalars
                 (np.timedelta64(3, '[D]'),
                  np.timedelta64(-3, '[D]'),
                  np.timedelta64(0, '[D]'),
                  np.timedelta64(1, '[D]'),
                  np.timedelta64(-1, '[D]'))]:
            # 对于每组测试数据进行以下操作

            # 负数的一元操作
            assert_equal(-tdb, tda)
            assert_equal((-tdb).dtype, tda.dtype)
            assert_equal(np.negative(tdb), tda)
            assert_equal(np.negative(tdb).dtype, tda.dtype)

            # 正数的一元操作
            assert_equal(np.positive(tda), tda)
            assert_equal(np.positive(tda).dtype, tda.dtype)
            assert_equal(np.positive(tdb), tdb)
            assert_equal(np.positive(tdb).dtype, tdb.dtype)

            # 绝对值的一元操作
            assert_equal(np.absolute(tdb), tda)
            assert_equal(np.absolute(tdb).dtype, tda.dtype)

            # 符号的一元操作
            assert_equal(np.sign(tda), tdone)
            assert_equal(np.sign(tdb), tdmone)
            assert_equal(np.sign(tdzero), tdzero)
            assert_equal(np.sign(tda).dtype, tda.dtype)

            # 运算函数总是产生本机字节顺序的结果
            assert_
    def test_datetime_multiply(self):
        # 迭代器，依次处理不同的测试用例
        for dta, tda, tdb, tdc in \
                    [
                     # One-dimensional arrays
                     (np.array(['2012-12-21'], dtype='M8[D]'),
                      np.array([6], dtype='m8[h]'),
                      np.array([9], dtype='m8[h]'),
                      np.array([12], dtype='m8[h]')),
                     # NumPy scalars
                     (np.datetime64('2012-12-21', '[D]'),
                      np.timedelta64(6, '[h]'),
                      np.timedelta64(9, '[h]'),
                      np.timedelta64(12, '[h]'))]:
            # m8 * int
            assert_equal(tda * 2, tdc)
            assert_equal((tda * 2).dtype, np.dtype('m8[h]'))
            # int * m8
            assert_equal(2 * tda, tdc)
            assert_equal((2 * tda).dtype, np.dtype('m8[h]'))
            # m8 * float
            assert_equal(tda * 1.5, tdb)
            assert_equal((tda * 1.5).dtype, np.dtype('m8[h]'))
            # float * m8
            assert_equal(1.5 * tda, tdb)
            assert_equal((1.5 * tda).dtype, np.dtype('m8[h]'))

            # m8 * m8
            assert_raises(TypeError, np.multiply, tda, tdb)
            # m8 * M8
            assert_raises(TypeError, np.multiply, dta, tda)
            # M8 * m8
            assert_raises(TypeError, np.multiply, tda, dta)
            # M8 * int
            assert_raises(TypeError, np.multiply, dta, 2)
            # int * M8
            assert_raises(TypeError, np.multiply, 2, dta)
            # M8 * float
            assert_raises(TypeError, np.multiply, dta, 1.5)
            # float * M8
            assert_raises(TypeError, np.multiply, 1.5, dta)

        # NaTs
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in multiply")
            # 创建 NaT（Not a Time）对象
            nat = np.timedelta64('NaT')
            # 定义检查函数，验证 NaT 与其他类型数据的乘法运算结果
            def check(a, b, res):
                assert_equal(a * b, res)
                assert_equal(b * a, res)
            # 对整数和浮点数类型进行检查
            for tp in (int, float):
                check(nat, tp(2), nat)
                check(nat, tp(0), nat)
            # 对于特殊浮点数值（无穷大和NaN）进行检查
            for f in (float('inf'), float('nan')):
                check(np.timedelta64(1), f, nat)
                check(np.timedelta64(0), f, nat)
                check(nat, f, nat)
    @pytest.mark.parametrize("op1, op2, exp", [
        # 定义参数化测试用例，测试 timedelta64 类型的整数除法
        # 当两个正时间差相同单位时，向下取整
        (np.timedelta64(7, 's'),
         np.timedelta64(4, 's'),
         1),
        # 当一个时间差为正，另一个为负且单位相同时，向下取整
        (np.timedelta64(7, 's'),
         np.timedelta64(-4, 's'),
         -2),
        # 当一个时间差为正，另一个为负且单位相同时，向下取整
        (np.timedelta64(8, 's'),
         np.timedelta64(-4, 's'),
         -2),
        # 当两个时间差单位不同时
        (np.timedelta64(1, 'm'),
         np.timedelta64(31, 's'),
         1),
        # 当两个时间差为通用单位时
        (np.timedelta64(1890),
         np.timedelta64(31),
         60),
        # 年 // 月运算
        (np.timedelta64(2, 'Y'),
         np.timedelta64('13', 'M'),
         1),
        # 处理一维数组
        (np.array([1, 2, 3], dtype='m8'),
         np.array([2], dtype='m8'),
         np.array([0, 1, 1], dtype=np.int64)),
        ])
    def test_timedelta_floor_divide(self, op1, op2, exp):
        # 断言整数除法的结果与预期相等
        assert_equal(op1 // op2, exp)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize("op1, op2", [
        # 被零除
        (np.timedelta64(10, 'us'),
         np.timedelta64(0, 'us')),
        # 包含 NaT 的除法
        (np.timedelta64('NaT'),
         np.timedelta64(50, 'us')),
        # int64 最小值的特殊情况
        # 在整数除法中
        (np.timedelta64(np.iinfo(np.int64).min),
         np.timedelta64(-1)),
        ])
    def test_timedelta_floor_div_warnings(self, op1, op2):
        # 断言在运行时发出 RuntimeWarning
        with assert_warns(RuntimeWarning):
            actual = op1 // op2
            assert_equal(actual, 0)
            assert_equal(actual.dtype, np.int64)

    @pytest.mark.parametrize("val1, val2", [
        # 无法在整数除法操作中明确划分年份和月份
        (9007199254740993, 1),
        # 压力测试替代整数除法代码路径
        # 操作数符号不匹配且余数不为 0
        (9007199254740999, -2),
        ])
    def test_timedelta_floor_div_precision(self, val1, val2):
        op1 = np.timedelta64(val1)
        op2 = np.timedelta64(val2)
        actual = op1 // op2
        # Python 参考整数向下取整
        expected = val1 // val2
        # 断言结果与预期相等
        assert_equal(actual, expected)

    @pytest.mark.parametrize("val1, val2", [
        # 年份和月份有时无法明确地划分
        # 用于整数向下取整操作
        (np.timedelta64(7, 'Y'),
         np.timedelta64(3, 's')),
        (np.timedelta64(7, 'M'),
         np.timedelta64(1, 'D')),
        ])
    def test_timedelta_floor_div_error(self, val1, val2):
        # 断言抛出 TypeError 异常，包含特定错误信息
        with assert_raises_regex(TypeError, "common metadata divisor"):
            val1 // val2
    @pytest.mark.parametrize("op1, op2", [
        # 使用 pytest.mark.parametrize 装饰器标记测试用例的参数化，op1 和 op2 是测试函数的输入参数
        # 从 floordiv 中重用测试用例
        (np.timedelta64(7, 's'),  # op1 设置为 7 秒的 np.timedelta64 类型
         np.timedelta64(4, 's')),  # op2 设置为 4 秒的 np.timedelta64 类型
        # m8 相同单位向下取整，包含负数
        (np.timedelta64(7, 's'),  # op1 设置为 7 秒的 np.timedelta64 类型
         np.timedelta64(-4, 's')),  # op2 设置为 -4 秒的 np.timedelta64 类型
        # m8 相同单位负数且无向下取整
        (np.timedelta64(8, 's'),  # op1 设置为 8 秒的 np.timedelta64 类型
         np.timedelta64(-4, 's')),  # op2 设置为 -4 秒的 np.timedelta64 类型
        # m8 不同单位
        (np.timedelta64(1, 'm'),  # op1 设置为 1 分钟的 np.timedelta64 类型
         np.timedelta64(31, 's')),  # op2 设置为 31 秒的 np.timedelta64 类型
        # m8 通用单位
        (np.timedelta64(1890),  # op1 设置为 1890 个时间单位的 np.timedelta64 类型
         np.timedelta64(31)),  # op2 设置为 31 个时间单位的 np.timedelta64 类型
        # Y // M 工作
        (np.timedelta64(2, 'Y'),  # op1 设置为 2 年的 np.timedelta64 类型
         np.timedelta64('13', 'M')),  # op2 设置为 '13' 个月的 np.timedelta64 类型
        # 处理 1D 数组
        (np.array([1, 2, 3], dtype='m8'),  # op1 设置为包含 [1, 2, 3] 的 1D 数组，数据类型为 'm8'
         np.array([2], dtype='m8')),  # op2 设置为包含 [2] 的 1D 数组，数据类型为 'm8'
        ])
    # 定义测试函数 test_timedelta_divmod，测试 np.timedelta64 类型的除法和取模操作
    def test_timedelta_divmod(self, op1, op2):
        # 预期结果是 op1 除以 op2 的商和余数
        expected = (op1 // op2, op1 % op2)
        # 使用 assert_equal 断言函数检查实际计算结果是否与预期结果相同
        assert_equal(divmod(op1, op2), expected)

    @pytest.mark.skipif(IS_WASM, reason="does not work in wasm")
    @pytest.mark.parametrize("op1, op2", [
        # 从 floordiv 中重用用例
        # 被零除
        (np.timedelta64(10, 'us'),  # op1 设置为 10 微秒的 np.timedelta64 类型
         np.timedelta64(0, 'us')),  # op2 设置为 0 微秒的 np.timedelta64 类型
        # 用 NaT 除法
        (np.timedelta64('NaT'),  # op1 设置为 NaT (Not a Time) 的 np.timedelta64 类型
         np.timedelta64(50, 'us')),  # op2 设置为 50 微秒的 np.timedelta64 类型
        # 整数 floor division 的 int64 min 的特殊情况
        (np.timedelta64(np.iinfo(np.int64).min),  # op1 设置为 np.int64 类型的最小值的 np.timedelta64 类型
         np.timedelta64(-1)),  # op2 设置为 -1 的 np.timedelta64 类型
        ])
    # 定义测试函数 test_timedelta_divmod_warnings，测试 np.timedelta64 类型的除法和取模操作，包含警告
    def test_timedelta_divmod_warnings(self, op1, op2):
        # 使用 assert_warns 上下文管理器确保警告被触发
        with assert_warns(RuntimeWarning):
            # 预期结果是 op1 除以 op2 的商和余数
            expected = (op1 // op2, op1 % op2)
        with assert_warns(RuntimeWarning):
            # 计算实际的除法和取模操作结果
            actual = divmod(op1, op2)
        # 使用 assert_equal 断言函数检查实际计算结果是否与预期结果相同
        assert_equal(actual, expected)
    # 定义测试方法 test_datetime_divide，使用 self 参数表示当前测试对象
    def test_datetime_divide(self):
        # 使用 for 循环遍历元组列表，每个元组包含多个变量
        for dta, tda, tdb, tdc, tdd in \
                    [
                     # One-dimensional arrays
                     (np.array(['2012-12-21'], dtype='M8[D]'),
                      np.array([6], dtype='m8[h]'),
                      np.array([9], dtype='m8[h]'),
                      np.array([12], dtype='m8[h]'),
                      np.array([6], dtype='m8[m]')),
                     # NumPy scalars
                     (np.datetime64('2012-12-21', '[D]'),
                      np.timedelta64(6, '[h]'),
                      np.timedelta64(9, '[h]'),
                      np.timedelta64(12, '[h]'),
                      np.timedelta64(6, '[m]'))]:
            # 对于每个元组，执行以下操作：

            # m8 / int
            assert_equal(tdc / 2, tda)  # 断言计算结果 tdc / 2 等于 tda
            assert_equal((tdc / 2).dtype, np.dtype('m8[h]'))  # 断言结果的数据类型为 'm8[h]'

            # m8 / float
            assert_equal(tda / 0.5, tdc)  # 断言计算结果 tda / 0.5 等于 tdc
            assert_equal((tda / 0.5).dtype, np.dtype('m8[h]'))  # 断言结果的数据类型为 'm8[h]'

            # m8 / m8
            assert_equal(tda / tdb, 6 / 9)  # 断言计算结果 tda / tdb 等于 6 / 9
            assert_equal(np.divide(tda, tdb), 6 / 9)  # 使用 np.divide 函数做相同断言
            assert_equal(np.true_divide(tda, tdb), 6 / 9)  # 使用 np.true_divide 函数做相同断言
            assert_equal(tdb / tda, 9 / 6)  # 断言计算结果 tdb / tda 等于 9 / 6
            assert_equal((tda / tdb).dtype, np.dtype('f8'))  # 断言结果的数据类型为 'f8'
            assert_equal(tda / tdd, 60)  # 断言计算结果 tda / tdd 等于 60
            assert_equal(tdd / tda, 1 / 60)  # 断言计算结果 tdd / tda 等于 1 / 60

            # int / m8
            assert_raises(TypeError, np.divide, 2, tdb)  # 断言调用 np.divide(2, tdb) 会抛出 TypeError 异常

            # float / m8
            assert_raises(TypeError, np.divide, 0.5, tdb)  # 断言调用 np.divide(0.5, tdb) 会抛出 TypeError 异常

            # m8 / M8
            assert_raises(TypeError, np.divide, dta, tda)  # 断言调用 np.divide(dta, tda) 会抛出 TypeError 异常

            # M8 / m8
            assert_raises(TypeError, np.divide, tda, dta)  # 断言调用 np.divide(tda, dta) 会抛出 TypeError 异常

            # M8 / int
            assert_raises(TypeError, np.divide, dta, 2)  # 断言调用 np.divide(dta, 2) 会抛出 TypeError 异常

            # int / M8
            assert_raises(TypeError, np.divide, 2, dta)  # 断言调用 np.divide(2, dta) 会抛出 TypeError 异常

            # M8 / float
            assert_raises(TypeError, np.divide, dta, 1.5)  # 断言调用 np.divide(dta, 1.5) 会抛出 TypeError 异常

            # float / M8
            assert_raises(TypeError, np.divide, 1.5, dta)  # 断言调用 np.divide(1.5, dta) 会抛出 TypeError 异常

        # NaTs
        with suppress_warnings() as sup:  # 使用 suppress_warnings 上下文管理器
            sup.filter(RuntimeWarning,  r".*encountered in divide")  # 过滤 RuntimeWarning，正则匹配".*encountered in divide"
            nat = np.timedelta64('NaT')  # 创建 NaT（Not-a-Time）时间间隔对象
            for tp in (int, float):  # 遍历 int 和 float 类型
                assert_equal(np.timedelta64(1) / tp(0), nat)  # 断言计算结果 np.timedelta64(1) / tp(0) 等于 nat
                assert_equal(np.timedelta64(0) / tp(0), nat)  # 断言计算结果 np.timedelta64(0) / tp(0) 等于 nat
                assert_equal(nat / tp(0), nat)  # 断言计算结果 nat / tp(0) 等于 nat
                assert_equal(nat / tp(2), nat)  # 断言计算结果 nat / tp(2) 等于 nat

            # Division by inf
            assert_equal(np.timedelta64(1) / float('inf'), np.timedelta64(0))  # 断言计算结果 np.timedelta64(1) / float('inf') 等于 np.timedelta64(0)
            assert_equal(np.timedelta64(0) / float('inf'), np.timedelta64(0))  # 断言计算结果 np.timedelta64(0) / float('inf') 等于 np.timedelta64(0)
            assert_equal(nat / float('inf'), nat)  # 断言计算结果 nat / float('inf') 等于 nat

            # Division by nan
            assert_equal(np.timedelta64(1) / float('nan'), nat)  # 断言计算结果 np.timedelta64(1) / float('nan') 等于 nat
            assert_equal(np.timedelta64(0) / float('nan'), nat)  # 断言计算结果 np.timedelta64(0) / float('nan') 等于 nat
            assert_equal(nat / float('nan'), nat)  # 断言计算结果 nat / float('nan') 等于 nat
    def test_datetime_compare(self):
        # 测试所有比较操作符
        a = np.datetime64('2000-03-12T18:00:00.000000')
        b = np.array(['2000-03-12T18:00:00.000000',
                      '2000-03-12T17:59:59.999999',
                      '2000-03-12T18:00:00.000001',
                      '1970-01-11T12:00:00.909090',
                      '2016-01-11T12:00:00.909090'],
                      dtype='datetime64[us]')
        # 断言a与b中的元素逐个进行相等比较，结果为[1, 0, 0, 0, 0]
        assert_equal(np.equal(a, b), [1, 0, 0, 0, 0])
        # 断言a与b中的元素逐个进行不等比较，结果为[0, 1, 1, 1, 1]
        assert_equal(np.not_equal(a, b), [0, 1, 1, 1, 1])
        # 断言a与b中的元素逐个进行小于比较，结果为[0, 0, 1, 0, 1]
        assert_equal(np.less(a, b), [0, 0, 1, 0, 1])
        # 断言a与b中的元素逐个进行小于等于比较，结果为[1, 0, 1, 0, 1]
        assert_equal(np.less_equal(a, b), [1, 0, 1, 0, 1])
        # 断言a与b中的元素逐个进行大于比较，结果为[0, 1, 0, 1, 0]
        assert_equal(np.greater(a, b), [0, 1, 0, 1, 0])
        # 断言a与b中的元素逐个进行大于等于比较，结果为[1, 1, 0, 1, 0]
        assert_equal(np.greater_equal(a, b), [1, 1, 0, 1, 0])

    def test_datetime_compare_nat(self):
        dt_nat = np.datetime64('NaT', 'D')
        dt_other = np.datetime64('2000-01-01')
        td_nat = np.timedelta64('NaT', 'h')
        td_other = np.timedelta64(1, 'h')

        for op in [np.equal, np.less, np.less_equal,
                   np.greater, np.greater_equal]:
            # 断言NaT与自身以及其他日期时间的比较结果都为假
            assert_(not op(dt_nat, dt_nat))
            assert_(not op(dt_nat, dt_other))
            assert_(not op(dt_other, dt_nat))

            # 断言NaT与自身以及其他时间间隔的比较结果都为假
            assert_(not op(td_nat, td_nat))
            assert_(not op(td_nat, td_other))
            assert_(not op(td_other, td_nat))

        # 断言NaT与自身以及其他日期时间的不等比较结果都为真
        assert_(np.not_equal(dt_nat, dt_nat))
        assert_(np.not_equal(dt_nat, dt_other))
        assert_(np.not_equal(dt_other, dt_nat))

        # 断言NaT与自身以及其他时间间隔的不等比较结果都为真
        assert_(np.not_equal(td_nat, td_nat))
        assert_(np.not_equal(td_nat, td_other))
        assert_(np.not_equal(td_other, td_nat))
    def test_datetime_minmax(self):
        # 测试日期时间的最小值和最大值计算功能

        # 创建具有特定日期时间类型的数组a和b
        a = np.array('1999-03-12T13', dtype='M8[2m]')
        b = np.array('1999-03-12T12', dtype='M8[s]')
        
        # 测试 np.minimum 函数：返回最小值
        assert_equal(np.minimum(a, b), b)
        # 检查返回结果的数据类型是否符合预期
        assert_equal(np.minimum(a, b).dtype, np.dtype('M8[s]'))
        
        # 测试 np.fmin 函数：返回最小值
        assert_equal(np.fmin(a, b), b)
        # 检查返回结果的数据类型是否符合预期
        assert_equal(np.fmin(a, b).dtype, np.dtype('M8[s]'))
        
        # 测试 np.maximum 函数：返回最大值
        assert_equal(np.maximum(a, b), a)
        # 检查返回结果的数据类型是否符合预期
        assert_equal(np.maximum(a, b).dtype, np.dtype('M8[s]'))
        
        # 测试 np.fmax 函数：返回最大值
        assert_equal(np.fmax(a, b), a)
        # 检查返回结果的数据类型是否符合预期
        assert_equal(np.fmax(a, b).dtype, np.dtype('M8[s]'))
        
        # 将数组a和b视为整数进行比较，由于单位不同，结果与上述相反
        assert_equal(np.minimum(a.view('i8'), b.view('i8')), a.view('i8'))

        # 测试与 NaT (Not-a-Time) 的交互
        a = np.array('1999-03-12T13', dtype='M8[2m]')
        dtnat = np.array('NaT', dtype='M8[h]')
        
        # 测试 np.minimum 函数处理 NaT 的情况
        assert_equal(np.minimum(a, dtnat), dtnat)
        assert_equal(np.minimum(dtnat, a), dtnat)
        
        # 测试 np.maximum 函数处理 NaT 的情况
        assert_equal(np.maximum(a, dtnat), dtnat)
        assert_equal(np.maximum(dtnat, a), dtnat)
        
        # 测试 np.fmin 函数处理 NaT 的情况
        assert_equal(np.fmin(dtnat, a), a)
        assert_equal(np.fmin(a, dtnat), a)
        
        # 测试 np.fmax 函数处理 NaT 的情况
        assert_equal(np.fmax(dtnat, a), a)
        assert_equal(np.fmax(a, dtnat), a)

        # 测试 timedelta 的情况
        a = np.array(3, dtype='m8[h]')
        b = np.array(3*3600 - 3, dtype='m8[s]')
        
        # 测试 np.minimum 函数：返回最小值
        assert_equal(np.minimum(a, b), b)
        # 检查返回结果的数据类型是否符合预期
        assert_equal(np.minimum(a, b).dtype, np.dtype('m8[s]'))
        
        # 测试 np.fmin 函数：返回最小值
        assert_equal(np.fmin(a, b), b)
        # 检查返回结果的数据类型是否符合预期
        assert_equal(np.fmin(a, b).dtype, np.dtype('m8[s]'))
        
        # 测试 np.maximum 函数：返回最大值
        assert_equal(np.maximum(a, b), a)
        # 检查返回结果的数据类型是否符合预期
        assert_equal(np.maximum(a, b).dtype, np.dtype('m8[s]'))
        
        # 测试 np.fmax 函数：返回最大值
        assert_equal(np.fmax(a, b), a)
        # 检查返回结果的数据类型是否符合预期
        assert_equal(np.fmax(a, b).dtype, np.dtype('m8[s]'))
        
        # 将数组a和b视为整数进行比较，由于单位不同，结果与上述相反
        assert_equal(np.minimum(a.view('i8'), b.view('i8')), a.view('i8'))

        # 测试 datetime 和 timedelta 之间的比较是否引发 TypeError
        a = np.array(3, dtype='m8[h]')
        b = np.array('1999-03-12T12', dtype='M8[s]')
        assert_raises(TypeError, np.minimum, a, b, casting='same_kind')
        assert_raises(TypeError, np.maximum, a, b, casting='same_kind')
        assert_raises(TypeError, np.fmin, a, b, casting='same_kind')
        assert_raises(TypeError, np.fmax, a, b, casting='same_kind')

    def test_hours(self):
        # 测试操作时间数组的小时属性

        # 创建一个长度为3的全1数组，数据类型为'M8[s]'
        t = np.ones(3, dtype='M8[s]')
        
        # 修改数组第一个元素的时间值为一天零10小时
        t[0] = 60*60*24 + 60*60*10
        
        # 断言：检查数组第一个元素的小时属性是否为10
        assert_(t[0].item().hour == 10)
    # 定义测试函数，用于验证时间类型的单位转换是否正确

    def test_divisor_conversion_year(self):
        # 断言：年份除以4，等同于3个月
        assert_(np.dtype('M8[Y/4]') == np.dtype('M8[3M]'))
        # 断言：年份除以13，等同于4周
        assert_(np.dtype('M8[Y/13]') == np.dtype('M8[4W]'))
        # 断言：3年除以73，等同于15天
        assert_(np.dtype('M8[3Y/73]') == np.dtype('M8[15D]'))

    def test_divisor_conversion_month(self):
        # 断言：月份除以2，等同于2周
        assert_(np.dtype('M8[M/2]') == np.dtype('M8[2W]'))
        # 断言：月份除以15，等同于2天
        assert_(np.dtype('M8[M/15]') == np.dtype('M8[2D]'))
        # 断言：3个月除以40，等同于54小时
        assert_(np.dtype('M8[3M/40]') == np.dtype('M8[54h]'))

    def test_divisor_conversion_week(self):
        # 断言：周数除以7，等同于1天
        assert_(np.dtype('m8[W/7]') == np.dtype('m8[D]'))
        # 断言：3周除以14，等同于36小时
        assert_(np.dtype('m8[3W/14]') == np.dtype('m8[36h]'))
        # 断言：5周除以140，等同于360分钟
        assert_(np.dtype('m8[5W/140]') == np.dtype('m8[360m]'))

    def test_divisor_conversion_day(self):
        # 断言：天数除以12，等同于2小时
        assert_(np.dtype('M8[D/12]') == np.dtype('M8[2h]'))
        # 断言：天数除以120，等同于12分钟
        assert_(np.dtype('M8[D/120]') == np.dtype('M8[12m]'))
        # 断言：3天除以960，等同于270秒
        assert_(np.dtype('M8[3D/960]') == np.dtype('M8[270s]'))

    def test_divisor_conversion_hour(self):
        # 断言：小时除以30，等同于2分钟
        assert_(np.dtype('m8[h/30]') == np.dtype('m8[2m]'))
        # 断言：3小时除以300，等同于36秒
        assert_(np.dtype('m8[3h/300]') == np.dtype('m8[36s]'))

    def test_divisor_conversion_minute(self):
        # 断言：分钟除以30，等同于2秒
        assert_(np.dtype('m8[m/30]') == np.dtype('m8[2s]'))
        # 断言：3分钟除以300，等同于600毫秒
        assert_(np.dtype('m8[3m/300]') == np.dtype('m8[600ms]'))

    def test_divisor_conversion_second(self):
        # 断言：秒数除以100，等同于10毫秒
        assert_(np.dtype('m8[s/100]') == np.dtype('m8[10ms]'))
        # 断言：3秒除以10000，等同于300微秒
        assert_(np.dtype('m8[3s/10000]') == np.dtype('m8[300us]'))

    def test_divisor_conversion_fs(self):
        # 断言：飞秒除以100，等同于10阿托秒
        assert_(np.dtype('M8[fs/100]') == np.dtype('M8[10as]'))
        # 断言：3飞秒除以10000，引发值错误异常，不支持超过10阿托秒的转换
        assert_raises(ValueError, lambda: np.dtype('M8[3fs/10000]'))

    def test_divisor_conversion_as(self):
        # 断言：阿托秒除以10，引发值错误异常，不支持阿托秒的单位转换
        assert_raises(ValueError, lambda: np.dtype('M8[as/10]'))
    def test_string_parser_variants(self):
        msg = "no explicit representation of timezones available for " \
              "np.datetime64"
        # 空格允许在日期和时间之间代替'T'
        assert_equal(np.array(['1980-02-29T01:02:03'], np.dtype('M8[s]')),
                     np.array(['1980-02-29 01:02:03'], np.dtype('M8[s]')))
        # 允许正年份
        assert_equal(np.array(['+1980-02-29T01:02:03'], np.dtype('M8[s]')),
                     np.array(['+1980-02-29 01:02:03'], np.dtype('M8[s]')))
        # 允许负年份
        assert_equal(np.array(['-1980-02-29T01:02:03'], np.dtype('M8[s]')),
                     np.array(['-1980-02-29 01:02:03'], np.dtype('M8[s]')))
        # UTC 指定符
        with pytest.warns(UserWarning, match=msg):
            assert_equal(
                np.array(['+1980-02-29T01:02:03'], np.dtype('M8[s]')),
                np.array(['+1980-02-29 01:02:03Z'], np.dtype('M8[s]')))
        with pytest.warns(UserWarning, match=msg):
            assert_equal(
                np.array(['-1980-02-29T01:02:03'], np.dtype('M8[s]')),
                np.array(['-1980-02-29 01:02:03Z'], np.dtype('M8[s]')))
        # 时间偏移
        with pytest.warns(UserWarning, match=msg):
            assert_equal(
                np.array(['1980-02-29T02:02:03'], np.dtype('M8[s]')),
                np.array(['1980-02-29 00:32:03-0130'], np.dtype('M8[s]')))
        with pytest.warns(UserWarning, match=msg):
            assert_equal(
                np.array(['1980-02-28T22:32:03'], np.dtype('M8[s]')),
                np.array(['1980-02-29 00:02:03+01:30'], np.dtype('M8[s]')))
        with pytest.warns(UserWarning, match=msg):
            assert_equal(
                np.array(['1980-02-29T02:32:03.506'], np.dtype('M8[s]')),
                np.array(['1980-02-29 00:32:03.506-02'], np.dtype('M8[s]')))
        with pytest.warns(UserWarning, match=msg):
            assert_equal(np.datetime64('1977-03-02T12:30-0230'),
                         np.datetime64('1977-03-02T15:00'))

    def test_creation_overflow(self):
        date = '1980-03-23 20:00:00'
        # 将日期转换为秒级别的时间戳
        timesteps = np.array([date], dtype='datetime64[s]')[0].astype(np.int64)
        for unit in ['ms', 'us', 'ns']:
            timesteps *= 1000
            # 使用不同精度单位创建日期时间对象
            x = np.array([date], dtype='datetime64[%s]' % unit)

            assert_equal(timesteps, x[0].astype(np.int64),
                         err_msg='Datetime conversion error for unit %s' % unit)

        assert_equal(x[0].astype(np.int64), 322689600000000000)

        # gh-13062
        # 检查溢出错误
        with pytest.raises(OverflowError):
            np.datetime64(2**64, 'D')
        with pytest.raises(OverflowError):
            np.timedelta64(2**64, 'D')

    @pytest.mark.skipif(not _has_pytz, reason="The pytz module is not available.")
    def test_datetime_as_string_timezone(self):
        # 测试函数：test_datetime_as_string_timezone，用于测试 np.datetime_as_string 方法的时区功能

        # 创建 numpy datetime 对象 a 和 b
        a = np.datetime64('2010-03-15T06:30', 'm')
        
        # 断言：将 np.datetime_as_string 应用于 a，不带时区信息，结果应为 '2010-03-15T06:30'
        assert_equal(np.datetime_as_string(a),
                    '2010-03-15T06:30')
        
        # 断言：将 np.datetime_as_string 应用于 a，使用 'naive' 时区参数，结果应为 '2010-03-15T06:30'
        assert_equal(np.datetime_as_string(a, timezone='naive'),
                    '2010-03-15T06:30')
        
        # 断言：将 np.datetime_as_string 应用于 a，使用 'UTC' 时区参数，结果应为 '2010-03-15T06:30Z'
        assert_equal(np.datetime_as_string(a, timezone='UTC'),
                    '2010-03-15T06:30Z')
        
        # 断言：将 np.datetime_as_string 应用于 a，使用 'local' 时区参数，结果应不等于 '2010-03-15T06:30'
        assert_(np.datetime_as_string(a, timezone='local') !=
                '2010-03-15T06:30')

        # 创建 numpy datetime 对象 b
        b = np.datetime64('2010-02-15T06:30', 'm')
        
        # 断言：将 np.datetime_as_string 应用于 a，使用 'US/Central' 时区，结果应为 '2010-03-15T01:30-0500'
        assert_equal(np.datetime_as_string(a, timezone=tz('US/Central')),
                     '2010-03-15T01:30-0500')
        
        # 断言：将 np.datetime_as_string 应用于 a，使用 'US/Eastern' 时区，结果应为 '2010-03-15T02:30-0400'
        assert_equal(np.datetime_as_string(a, timezone=tz('US/Eastern')),
                     '2010-03-15T02:30-0400')
        
        # 断言：将 np.datetime_as_string 应用于 a，使用 'US/Pacific' 时区，结果应为 '2010-03-14T23:30-0700'
        assert_equal(np.datetime_as_string(a, timezone=tz('US/Pacific')),
                     '2010-03-14T23:30-0700')

        # 断言：将 np.datetime_as_string 应用于 b，使用 'US/Central' 时区，结果应为 '2010-02-15T00:30-0600'
        assert_equal(np.datetime_as_string(b, timezone=tz('US/Central')),
                     '2010-02-15T00:30-0600')
        
        # 断言：将 np.datetime_as_string 应用于 b，使用 'US/Eastern' 时区，结果应为 '2010-02-15T01:30-0500'
        assert_equal(np.datetime_as_string(b, timezone=tz('US/Eastern')),
                     '2010-02-15T01:30-0500')
        
        # 断言：将 np.datetime_as_string 应用于 b，使用 'US/Pacific' 时区，结果应为 '2010-02-14T22:30-0800'
        assert_equal(np.datetime_as_string(b, timezone=tz('US/Pacific')),
                     '2010-02-14T22:30-0800')

        # 断言：当尝试将日期转换为带时区的字符串时，默认情况下会引发 TypeError 异常
        assert_raises(TypeError, np.datetime_as_string, a, unit='D',
                      timezone=tz('US/Pacific'))
        
        # 断言：使用 'unit' 参数为 'D'，将日期以 'US/Pacific' 时区的格式打印，使用 'casting' 参数为 'unsafe'
        assert_equal(np.datetime_as_string(a, unit='D',
                      timezone=tz('US/Pacific'), casting='unsafe'),
                     '2010-03-14')
        
        # 断言：使用 'unit' 参数为 'D'，将日期以 'US/Central' 时区的格式打印，使用 'casting' 参数为 'unsafe'
        assert_equal(np.datetime_as_string(b, unit='D',
                      timezone=tz('US/Central'), casting='unsafe'),
                     '2010-02-15')
    def test_datetime_arange(self):
        # 使用字符串指定的两个日期范围创建日期数组
        a = np.arange('2010-01-05', '2010-01-10', dtype='M8[D]')
        assert_equal(a.dtype, np.dtype('M8[D]'))
        assert_equal(a,
            np.array(['2010-01-05', '2010-01-06', '2010-01-07',
                      '2010-01-08', '2010-01-09'], dtype='M8[D]'))

        # 使用倒序创建日期数组，指定步长为-1
        a = np.arange('1950-02-10', '1950-02-06', -1, dtype='M8[D]')
        assert_equal(a.dtype, np.dtype('M8[D]'))
        assert_equal(a,
            np.array(['1950-02-10', '1950-02-09', '1950-02-08',
                      '1950-02-07'], dtype='M8[D]'))

        # 在月份单位下创建日期数组，步长为2
        a = np.arange('1969-05', '1970-05', 2, dtype='M8')
        assert_equal(a.dtype, np.dtype('M8[M]'))
        assert_equal(a,
            np.datetime64('1969-05') + np.arange(12, step=2))

        # 使用年份单位创建日期数组，步长为3
        a = np.arange('1969', 18, 3, dtype='M8')
        assert_equal(a.dtype, np.dtype('M8[Y]'))
        assert_equal(a,
            np.datetime64('1969') + np.arange(18, step=3))

        # 使用时间增量创建日期数组，步长为2天
        a = np.arange('1969-12-19', 22, np.timedelta64(2), dtype='M8')
        assert_equal(a.dtype, np.dtype('M8[D]'))
        assert_equal(a,
            np.datetime64('1969-12-19') + np.arange(22, step=2))

        # 步长为0是不允许的，应引发 ValueError 异常
        assert_raises(ValueError, np.arange, np.datetime64('today'),
                                np.datetime64('today') + 3, 0)
        # 跨越非线性单位边界的提升是不允许的，应引发 TypeError 异常
        assert_raises(TypeError, np.arange, np.datetime64('2011-03-01', 'D'),
                                np.timedelta64(5, 'M'))
        assert_raises(TypeError, np.arange,
                                np.datetime64('2012-02-03T14', 's'),
                                np.timedelta64(5, 'Y'))

    def test_datetime_arange_no_dtype(self):
        # 使用单个日期创建日期数组，并检查抛出 ValueError 异常
        d = np.array('2010-01-04', dtype="M8[D]")
        assert_equal(np.arange(d, d + 1), d)
        assert_raises(ValueError, np.arange, d)

    def test_timedelta_arange(self):
        # 创建时间增量数组，从3到10
        a = np.arange(3, 10, dtype='m8')
        assert_equal(a.dtype, np.dtype('m8'))
        assert_equal(a, np.timedelta64(0) + np.arange(3, 10))

        # 使用时间增量作为起始值，步长为2秒
        a = np.arange(np.timedelta64(3, 's'), 10, 2, dtype='m8')
        assert_equal(a.dtype, np.dtype('m8[s]'))
        assert_equal(a, np.timedelta64(0, 's') + np.arange(3, 10, 2))

        # 步长为0是不允许的，应引发 ValueError 异常
        assert_raises(ValueError, np.arange, np.timedelta64(0),
                                np.timedelta64(5), 0)
        # 跨越非线性单位边界的提升是不允许的，应引发 TypeError 异常
        assert_raises(TypeError, np.arange, np.timedelta64(0, 'D'),
                                np.timedelta64(5, 'M'))
        assert_raises(TypeError, np.arange, np.timedelta64(0, 'Y'),
                                np.timedelta64(5, 'D'))
    @pytest.mark.parametrize("val1, val2, expected", [
        # case from gh-12092
        # 定义测试参数：val1, val2 和期望的结果 expected，用于测试时间差的取模运算

        (np.timedelta64(7, 's'),     # 第一个测试用例：7 秒
         np.timedelta64(3, 's'),     # 第二个测试用例：3 秒
         np.timedelta64(1, 's')),    # 期望的结果：1 秒

        # negative value cases
        # 负值情况的测试

        (np.timedelta64(3, 's'),     # 3 秒
         np.timedelta64(-2, 's'),    # -2 秒
         np.timedelta64(-1, 's')),   # 期望的结果：-1 秒

        (np.timedelta64(-3, 's'),    # -3 秒
         np.timedelta64(2, 's'),     # 2 秒
         np.timedelta64(1, 's')),    # 期望的结果：1 秒

        # larger value cases
        # 较大值的测试

        (np.timedelta64(17, 's'),    # 17 秒
         np.timedelta64(22, 's'),    # 22 秒
         np.timedelta64(17, 's')),   # 期望的结果：17 秒

        (np.timedelta64(22, 's'),    # 22 秒
         np.timedelta64(17, 's'),    # 17 秒
         np.timedelta64(5, 's')),    # 期望的结果：5 秒

        # different units
        # 不同单位的测试

        (np.timedelta64(1, 'm'),     # 1 分钟
         np.timedelta64(57, 's'),    # 57 秒
         np.timedelta64(3, 's')),    # 期望的结果：3 秒

        (np.timedelta64(1, 'us'),    # 1 微秒
         np.timedelta64(727, 'ns'),  # 727 纳秒
         np.timedelta64(273, 'ns')), # 期望的结果：273 纳秒

        # NaT is propagated
        # NaT 值传播测试

        (np.timedelta64('NaT'),      # NaT (Not a Time)
         np.timedelta64(50, 'ns'),   # 50 纳秒
         np.timedelta64('NaT')),     # 期望的结果：NaT

        # Y % M works
        # 年份 % 月份 的测试

        (np.timedelta64(2, 'Y'),     # 2 年
         np.timedelta64(22, 'M'),    # 22 个月
         np.timedelta64(2, 'M')),    # 期望的结果：2 个月
    ])
    def test_timedelta_modulus(self, val1, val2, expected):
        # 断言：val1 % val2 的结果应为 expected
        assert_equal(val1 % val2, expected)

    @pytest.mark.parametrize("val1, val2", [
        # years and months sometimes can't be unambiguously
        # divided for modulus operation
        # 年份和月份有时无法明确地进行取模操作

        (np.timedelta64(7, 'Y'),     # 7 年
         np.timedelta64(3, 's')),    # 3 秒

        (np.timedelta64(7, 'M'),     # 7 个月
         np.timedelta64(1, 'D')),    # 1 天
    ])
    def test_timedelta_modulus_error(self, val1, val2):
        # 使用断言验证 TypeError 异常，错误消息包含 "common metadata divisor"
        with assert_raises_regex(TypeError, "common metadata divisor"):
            val1 % val2

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_timedelta_modulus_div_by_zero(self):
        # 如果在 WASM 环境下，则跳过测试：浮点错误在 WASM 中不起作用
        with assert_warns(RuntimeWarning):
            # 执行取模运算，验证结果为 NaT (Not a Time)
            actual = np.timedelta64(10, 's') % np.timedelta64(0, 's')
            assert_equal(actual, np.timedelta64('NaT'))

    @pytest.mark.parametrize("val1, val2", [
        # cases where one operand is not
        # timedelta64
        # 其中一个操作数不是 timedelta64 的情况

        (np.timedelta64(7, 'Y'),     # 7 年
         15,),                       # 整数 15

        (7.5,                         # 浮点数 7.5
         np.timedelta64(1, 'D')),    # 1 天
    ])
    def test_timedelta_modulus_type_resolution(self, val1, val2):
        # 注意：将来可能会支持一些操作
        # 使用断言验证 TypeError 异常，错误消息包含 "'remainder' cannot use operands with types"
        with assert_raises_regex(TypeError,
                                 "'remainder' cannot use operands with types"):
            val1 % val2

    def test_timedelta_arange_no_dtype(self):
        # 创建一个包含单个元素 5 的 numpy 数组，数据类型为 "m8[D]"
        d = np.array(5, dtype="m8[D]")
        # 断言：np.arange(d, d + 1) 应该等于 d
        assert_equal(np.arange(d, d + 1), d)
        # 断言：np.arange(d) 应该等于 np.arange(0, d)
        assert_equal(np.arange(d), np.arange(0, d))
    # 定义一个测试函数，用于验证 np.maximum.reduce 的功能
    def test_datetime_maximum_reduce(self):
        # 创建一个包含日期的 NumPy 数组，以 'M8[D]' 类型存储
        a = np.array(['2010-01-02', '1999-03-14', '1833-03'], dtype='M8[D]')
        # 断言 np.maximum.reduce 的结果数据类型应为 'M8[D]'
        assert_equal(np.maximum.reduce(a).dtype, np.dtype('M8[D]'))
        # 断言 np.maximum.reduce 的结果应为 '2010-01-02' 对应的 np.datetime64 对象
        assert_equal(np.maximum.reduce(a),
                     np.datetime64('2010-01-02'))

        # 创建一个包含时间间隔的 NumPy 数组，以 'm8[s]' 类型存储
        a = np.array([1, 4, 0, 7, 2], dtype='m8[s]')
        # 断言 np.maximum.reduce 的结果数据类型应为 'm8[s]'
        assert_equal(np.maximum.reduce(a).dtype, np.dtype('m8[s]'))
        # 断言 np.maximum.reduce 的结果应为 7 秒对应的 np.timedelta64 对象
        assert_equal(np.maximum.reduce(a),
                     np.timedelta64(7, 's'))

    # 定义一个测试函数，用于验证 np.mean 方法在 timedelta 类型上的正确性
    def test_timedelta_correct_mean(self):
        # 创建一个包含 timedelta 类型数据的 NumPy 数组
        a = np.arange(1000, dtype="m8[s]")
        # 断言数组的平均值等于总和除以数组长度，验证 np.mean 在 timedelta 类型上的表现
        assert_array_equal(a.mean(), a.sum() / len(a))

    # 定义一个测试函数，用于验证不能对 datetime64 类型数据使用 reduce 相关方法的行为
    def test_datetime_no_subtract_reducelike(self):
        # 创建一个包含 datetime64 类型数据的 NumPy 数组
        arr = np.array(["2021-12-02", "2019-05-12"], dtype="M8[ms]")
        # 设置匹配的错误消息
        msg = r"the resolved dtypes are not compatible"

        # 使用 pytest 来断言 np.subtract.reduce 在 datetime64 类型数组上会引发 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            np.subtract.reduce(arr)

        # 使用 pytest 来断言 np.subtract.accumulate 在 datetime64 类型数组上会引发 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            np.subtract.accumulate(arr)

        # 使用 pytest 来断言 np.subtract.reduceat 在 datetime64 类型数组上会引发 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            np.subtract.reduceat(arr, [0])
    def test_datetime_busdaycalendar(self):
        # 检查去除 NaT（Not a Time）、重复日期和周末，并确保结果排序正确。
        bdd = np.busdaycalendar(
            holidays=['NaT', '2011-01-17', '2011-03-06', 'NaT',
                       '2011-12-26', '2011-05-30', '2011-01-17'])
        # 断言节假日数组与预期结果相符
        assert_equal(bdd.holidays,
            np.array(['2011-01-17', '2011-05-30', '2011-12-26'], dtype='M8'))
        # 默认的工作日掩码为星期一到星期五
        assert_equal(bdd.weekmask, np.array([1, 1, 1, 1, 1, 0, 0], dtype='?'))

        # 检查包含不同空白符的字符串工作日掩码
        bdd = np.busdaycalendar(weekmask="Sun TueWed  Thu\tFri")
        # 断言工作日掩码与预期结果相符
        assert_equal(bdd.weekmask, np.array([0, 1, 1, 1, 1, 0, 1], dtype='?'))

        # 检查长度为7的0/1字符串工作日掩码
        bdd = np.busdaycalendar(weekmask="0011001")
        # 断言工作日掩码与预期结果相符
        assert_equal(bdd.weekmask, np.array([0, 0, 1, 1, 0, 0, 1], dtype='?'))

        # 检查长度为7的字符串工作日掩码
        bdd = np.busdaycalendar(weekmask="Mon Tue")
        # 断言工作日掩码与预期结果相符
        assert_equal(bdd.weekmask, np.array([1, 1, 0, 0, 0, 0, 0], dtype='?'))

        # 全零工作日掩码应该引发 ValueError
        assert_raises(ValueError, np.busdaycalendar, weekmask=[0, 0, 0, 0, 0, 0, 0])
        # 工作日名称必须正确大小写
        assert_raises(ValueError, np.busdaycalendar, weekmask="satsun")
        # 全空字符串工作日掩码应该引发 ValueError
        assert_raises(ValueError, np.busdaycalendar, weekmask="")
        # 无效的工作日名称代码应该引发 ValueError
        assert_raises(ValueError, np.busdaycalendar, weekmask="Mon Tue We")
        assert_raises(ValueError, np.busdaycalendar, weekmask="Max")
        assert_raises(ValueError, np.busdaycalendar, weekmask="Monday Tue")
    # 定义测试函数，用于测试 np.busday_count 的不同用例
    def test_datetime_busday_holidays_count(self):
        # 定义节假日列表
        holidays = ['2011-01-01', '2011-10-10', '2011-11-11', '2011-11-24',
                    '2011-12-25', '2011-05-30', '2011-02-21', '2011-01-17',
                    '2011-12-26', '2012-01-02', '2011-02-21', '2011-05-30',
                    '2011-07-01', '2011-07-04', '2011-09-05', '2011-10-10']
        
        # 创建一个工作日日历对象，使用指定的工作日掩码和节假日列表
        bdd = np.busdaycalendar(weekmask='1111100', holidays=holidays)

        # 使用 np.busday_offset 生成日期序列，检验 np.busday_count 的正确性
        dates = np.busday_offset('2011-01-01', np.arange(366),
                        roll='forward', busdaycal=bdd)
        # 断言 np.busday_count 返回的结果与预期的一致
        assert_equal(np.busday_count('2011-01-01', dates, busdaycal=bdd),
                     np.arange(366))
        
        # 当日期反向时，np.busday_count 返回负值
        assert_equal(np.busday_count(dates, '2011-01-01', busdaycal=bdd),
                     -np.arange(366) - 1)

        # 2011-12-31 是星期六，使用 np.busday_offset 生成日期序列
        dates = np.busday_offset('2011-12-31', -np.arange(366),
                        roll='forward', busdaycal=bdd)
        
        # 检验 np.busday_count 的结果是否正确
        expected = np.arange(366)
        expected[0] = -1
        assert_equal(np.busday_count(dates, '2011-12-31', busdaycal=bdd),
                     expected)
        
        # 当日期反向时，np.busday_count 返回负值
        expected = -np.arange(366) + 1
        expected[0] = 0
        assert_equal(np.busday_count('2011-12-31', dates, busdaycal=bdd),
                     expected)

        # 当同时提供了 weekmask/holidays 和 busdaycal 时，应当引发 ValueError 异常
        assert_raises(ValueError, np.busday_offset, '2012-01-03', '2012-02-03',
                        weekmask='1111100', busdaycal=bdd)
        assert_raises(ValueError, np.busday_offset, '2012-01-03', '2012-02-03',
                        holidays=holidays, busdaycal=bdd)

        # 检验 2011 年 3 月的星期一数量
        assert_equal(np.busday_count('2011-03', '2011-04', weekmask='Mon'), 4)
        
        # 当日期反向时，np.busday_count 返回负值
        assert_equal(np.busday_count('2011-04', '2011-03', weekmask='Mon'), -4)

        # 使用 np.datetime64 定义日期对象
        sunday = np.datetime64('2023-03-05')
        monday = sunday + 1
        friday = sunday + 5
        saturday = sunday + 6
        
        # 检验 np.busday_count 对不同日期的计算结果
        assert_equal(np.busday_count(sunday, monday), 0)
        assert_equal(np.busday_count(monday, sunday), -1)

        assert_equal(np.busday_count(friday, saturday), 1)
        assert_equal(np.busday_count(saturday, friday), 0)
    # 测试 datetime 模块中的 is_busday 函数
    def test_datetime_is_busday(self):
        # 定义一组节假日列表
        holidays = ['2011-01-01', '2011-10-10', '2011-11-11', '2011-11-24',
                    '2011-12-25', '2011-05-30', '2011-02-21', '2011-01-17',
                    '2011-12-26', '2012-01-02', '2011-02-21', '2011-05-30',
                    '2011-07-01', '2011-07-04', '2011-09-05', '2011-10-10',
                    'NaT']
        
        # 使用 numpy 的 busdaycalendar 函数创建一个自定义工作日历对象
        bdd = np.busdaycalendar(weekmask='1111100', holidays=holidays)

        # 测试周末和工作日的情况
        assert_equal(np.is_busday('2011-01-01'), False)  # 断言：2011-01-01 不是工作日
        assert_equal(np.is_busday('2011-01-02'), False)  # 断言：2011-01-02 不是工作日
        assert_equal(np.is_busday('2011-01-03'), True)   # 断言：2011-01-03 是工作日

        # 断言：所有节假日都不是工作日
        assert_equal(np.is_busday(holidays, busdaycal=bdd),
                     np.zeros(len(holidays), dtype='?'))

    # 测试 datetime 模块中的 datetime64 类对 Y2038 问题的处理
    def test_datetime_y2038(self):
        msg = "no explicit representation of timezones available for " \
              "np.datetime64"
        
        # 测试 Y2038 问题边界的解析
        a = np.datetime64('2038-01-19T03:14:07')
        assert_equal(a.view(np.int64), 2**31 - 1)  # 断言：a 的 int64 视图等于 2^31 - 1
        
        a = np.datetime64('2038-01-19T03:14:08')
        assert_equal(a.view(np.int64), 2**31)       # 断言：a 的 int64 视图等于 2^31
        
        # 测试 Y2038 问题边界的解析，带有手动指定的时区偏移
        with pytest.warns(UserWarning, match=msg):
            a = np.datetime64('2038-01-19T04:14:07+0100')
            assert_equal(a.view(np.int64), 2**31 - 1)  # 断言：a 的 int64 视图等于 2^31 - 1
        
        with pytest.warns(UserWarning, match=msg):
            a = np.datetime64('2038-01-19T04:14:08+0100')
            assert_equal(a.view(np.int64), 2**31)       # 断言：a 的 int64 视图等于 2^31
        
        # 测试解析 Y2038 后的日期
        a = np.datetime64('2038-01-20T13:21:14')
        assert_equal(str(a), '2038-01-20T13:21:14')   # 断言：a 转换为字符串后与指定的字符串相等

    # 测试 datetime 模块中的 isnat 函数
    def test_isnat(self):
        assert_(np.isnat(np.datetime64('NaT', 'ms')))  # 断言：NaT 表示的日期时间是 NaT（非有效日期时间）
        assert_(np.isnat(np.datetime64('NaT', 'ns')))  # 断言：NaT 表示的日期时间是 NaT（非有效日期时间）
        assert_(not np.isnat(np.datetime64('2038-01-19T03:14:07')))  # 断言：指定的日期时间不是 NaT

        assert_(np.isnat(np.timedelta64('NaT', "ms")))  # 断言：NaT 表示的时间间隔是 NaT（非有效时间间隔）
        assert_(not np.isnat(np.timedelta64(34, "ms")))  # 断言：指定的时间间隔不是 NaT

        res = np.array([False, False, True])
        for unit in ['Y', 'M', 'W', 'D',
                     'h', 'm', 's', 'ms', 'us',
                     'ns', 'ps', 'fs', 'as']:
            arr = np.array([123, -321, "NaT"], dtype='<datetime64[%s]' % unit)
            assert_equal(np.isnat(arr), res)  # 断言：arr 中的日期时间是否为 NaT
            arr = np.array([123, -321, "NaT"], dtype='>datetime64[%s]' % unit)
            assert_equal(np.isnat(arr), res)  # 断言：arr 中的日期时间是否为 NaT
            arr = np.array([123, -321, "NaT"], dtype='<timedelta64[%s]' % unit)
            assert_equal(np.isnat(arr), res)  # 断言：arr 中的时间间隔是否为 NaT
            arr = np.array([123, -321, "NaT"], dtype='>timedelta64[%s]' % unit)
            assert_equal(np.isnat(arr), res)  # 断言：arr 中的时间间隔是否为 NaT
    def test_isnat_error(self):
        # 检查只有 datetime 数据类型的数组被接受
        for t in np.typecodes["All"]:
            if t in np.typecodes["Datetime"]:
                continue
            # 断言应该引发 TypeError 异常，因为只有 datetime 类型的数组能被接受
            assert_raises(TypeError, np.isnat, np.zeros(10, t))

    def test_isfinite_scalar(self):
        # 检查 np.isfinite 对标量的操作
        assert_(not np.isfinite(np.datetime64('NaT', 'ms')))
        assert_(not np.isfinite(np.datetime64('NaT', 'ns')))
        assert_(np.isfinite(np.datetime64('2038-01-19T03:14:07')))

        assert_(not np.isfinite(np.timedelta64('NaT', "ms")))
        assert_(np.isfinite(np.timedelta64(34, "ms")))

    @pytest.mark.parametrize('unit', ['Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms',
                                      'us', 'ns', 'ps', 'fs', 'as'])
    @pytest.mark.parametrize('dstr', ['<datetime64[%s]', '>datetime64[%s]',
                                      '<timedelta64[%s]', '>timedelta64[%s]'])
    def test_isfinite_isinf_isnan_units(self, unit, dstr):
        '''
        检查 <M, >M, <m, >m 数据类型的所有单位的 isfinite, isinf, isnan
        '''
        arr_val = [123, -321, "NaT"]
        arr = np.array(arr_val,  dtype= dstr % unit)
        pos = np.array([True, True,  False])
        neg = np.array([False, False,  True])
        false = np.array([False, False,  False])
        assert_equal(np.isfinite(arr), pos)
        assert_equal(np.isinf(arr), false)
        assert_equal(np.isnan(arr), neg)

    def test_assert_equal(self):
        # 断言应该引发 AssertionError，因为 np.datetime64('nat') 和 np.timedelta64('nat') 不相等
        assert_raises(AssertionError, assert_equal,
                np.datetime64('nat'), np.timedelta64('nat'))

    def test_corecursive_input(self):
        # 构造一个共递归列表
        a, b = [], []
        a.append(b)
        b.append(a)
        obj_arr = np.array([None])
        obj_arr[0] = a

        # 在某些情况下会导致堆栈溢出 (gh-11154)。现在会引发 ValueError，因为嵌套列表无法转换为 datetime。
        assert_raises(ValueError, obj_arr.astype, 'M8')
        assert_raises(ValueError, obj_arr.astype, 'm8')

    @pytest.mark.parametrize("shape", [(), (1,)])
    def test_discovery_from_object_array(self, shape):
        arr = np.array("2020-10-10", dtype=object).reshape(shape)
        res = np.array("2020-10-10", dtype="M8").reshape(shape)
        assert res.dtype == np.dtype("M8[D]")
        assert_equal(arr.astype("M8"), res)
        arr[...] = np.bytes_("2020-10-10")  # 尝试使用 numpy 字符串类型
        assert_equal(arr.astype("M8"), res)
        arr = arr.astype("S")
        assert_equal(arr.astype("S").astype("M8"), res)

    @pytest.mark.parametrize("time_unit", [
        "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as",
        # 复合单位
        "10D", "2M",
    ])
    def test_limit_symmetry(self, time_unit):
        """
        Dates should have symmetric limits around the unix epoch at +/-np.int64
        """
        # 创建一个表示 Unix 纪元的 numpy datetime 对象
        epoch = np.datetime64(0, time_unit)
        # 创建一个表示 np.int64 的最大值的 numpy datetime 对象
        latest = np.datetime64(np.iinfo(np.int64).max, time_unit)
        # 创建一个表示 np.int64 的最小值的 numpy datetime 对象
        earliest = np.datetime64(-np.iinfo(np.int64).max, time_unit)

        # 断言确保最早日期小于纪元日期小于最晚日期
        assert earliest < epoch < latest

    @pytest.mark.parametrize("time_unit", [
        "Y", "M",
        # 对于 "W" 时间单位，标记为预期失败，因为类似于 gh-13197
        pytest.param("W", marks=pytest.mark.xfail(reason="gh-13197")),
        "D", "h", "m",
        "s", "ms", "us", "ns", "ps", "fs", "as",
        # 对于 "10D" 时间单位，标记为预期失败，类似于 gh-13197
        pytest.param("10D", marks=pytest.mark.xfail(reason="similar to gh-13197")),
    ])
    @pytest.mark.parametrize("sign", [-1, 1])
    def test_limit_str_roundtrip(self, time_unit, sign):
        """
        Limits should roundtrip when converted to strings.

        This tests the conversion to and from npy_datetimestruct.
        """
        # TODO: add absolute (gold standard) time span limit strings
        # 根据时间单位和符号创建一个限制日期的 numpy datetime 对象
        limit = np.datetime64(np.iinfo(np.int64).max * sign, time_unit)

        # 将日期转换为字符串然后再转换回来。由于天数和周数的表示方式不可区分，需要显式指定时间单位。
        limit_via_str = np.datetime64(str(limit), time_unit)
        # 断言转换后的日期与原始日期相等
        assert limit_via_str == limit
class TestDateTimeData:

    def test_basic(self):
        # 创建一个包含单个日期字符串的 numpy 数组，指定日期时间类型为 np.datetime64
        a = np.array(['1980-03-23'], dtype=np.datetime64)
        # 断言 np.datetime_data 函数返回的日期时间数据元组
        assert_equal(np.datetime_data(a.dtype), ('D', 1))

    def test_bytes(self):
        # 以字节单位创建 numpy datetime64 对象，并转换为 Unicode
        dt = np.datetime64('2000', (b'ms', 5))
        # 断言 np.datetime_data 函数返回的日期时间数据元组
        assert np.datetime_data(dt.dtype) == ('ms', 5)

        # 再次以字节单位创建 numpy datetime64 对象，字节单位被解释为时间单位
        dt = np.datetime64('2000', b'5ms')
        # 断言 np.datetime_data 函数返回的日期时间数据元组
        assert np.datetime_data(dt.dtype) == ('ms', 5)

    def test_non_ascii(self):
        # 使用带有非 ASCII 字符的单位字符串创建 numpy datetime64 对象，规范化为 Unicode
        dt = np.datetime64('2000', ('μs', 5))
        # 断言 np.datetime_data 函数返回的日期时间数据元组
        assert np.datetime_data(dt.dtype) == ('us', 5)

        # 再次使用带有非 ASCII 字符的单位字符串创建 numpy datetime64 对象，规范化为 Unicode
        dt = np.datetime64('2000', '5μs')
        # 断言 np.datetime_data 函数返回的日期时间数据元组
        assert np.datetime_data(dt.dtype) == ('us', 5)


def test_comparisons_return_not_implemented():
    # GH#17017

    class custom:
        __array_priority__ = 10000

    obj = custom()

    # 创建一个纳秒精度的 numpy datetime64 对象
    dt = np.datetime64('2000', 'ns')
    # 计算该对象与自身的时间差
    td = dt - dt

    # 对 dt 和 td 进行循环处理
    for item in [dt, td]:
        # 断言比较操作返回 NotImplemented
        assert item.__eq__(obj) is NotImplemented
        assert item.__ne__(obj) is NotImplemented
        assert item.__le__(obj) is NotImplemented
        assert item.__lt__(obj) is NotImplemented
        assert item.__ge__(obj) is NotImplemented
        assert item.__gt__(obj) is NotImplemented
```