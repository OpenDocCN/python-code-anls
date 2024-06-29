# `D:\src\scipysrc\pandas\pandas\tests\indexes\period\methods\test_shift.py`

```
# 导入所需的库和模块
import numpy as np
import pytest
from pandas import (
    PeriodIndex,
    period_range,
)
import pandas._testing as tm

# 定义测试类 TestPeriodIndexShift，用于测试 PeriodIndex 对象的位移操作
class TestPeriodIndexShift:
    # ---------------------------------------------------------------
    # PeriodIndex.shift is used by __add__ and __sub__

    # 测试使用 ndarray 对象进行 PeriodIndex 的位移操作
    def test_pi_shift_ndarray(self):
        # 创建一个 PeriodIndex 对象 idx，包含日期和 NaT（Not a Time）值
        idx = PeriodIndex(
            ["2011-01", "2011-02", "NaT", "2011-04"], freq="M", name="idx"
        )
        # 对 idx 进行位移操作，使用 np.array([1, 2, 3, 4]) 数组
        result = idx.shift(np.array([1, 2, 3, 4]))
        # 预期的位移结果 PeriodIndex 对象 expected
        expected = PeriodIndex(
            ["2011-02", "2011-04", "NaT", "2011-08"], freq="M", name="idx"
        )
        # 使用 pandas._testing.assert_index_equal 函数比较 result 和 expected
        tm.assert_index_equal(result, expected)

        # 再次进行位移操作，使用 np.array([1, -2, 3, -4]) 数组
        result = idx.shift(np.array([1, -2, 3, -4]))
        # 预期的位移结果 PeriodIndex 对象 expected
        expected = PeriodIndex(
            ["2011-02", "2010-12", "NaT", "2010-12"], freq="M", name="idx"
        )
        # 使用 pandas._testing.assert_index_equal 函数比较 result 和 expected
        tm.assert_index_equal(result, expected)

    # 测试不同情况下的 PeriodIndex 的位移操作
    def test_shift(self):
        # 创建两个不同的 PeriodIndex 对象 pi1 和 pi2，频率为年，起始和结束日期不同
        pi1 = period_range(freq="Y", start="1/1/2001", end="12/1/2009")
        pi2 = period_range(freq="Y", start="1/1/2002", end="12/1/2010")

        # 使用 pandas._testing.assert_index_equal 函数比较位移前后的索引对象
        tm.assert_index_equal(pi1.shift(0), pi1)

        # 检查两个 PeriodIndex 对象的长度是否相等
        assert len(pi1) == len(pi2)
        # 进行位移操作，并使用 pandas._testing.assert_index_equal 函数比较结果
        tm.assert_index_equal(pi1.shift(1), pi2)

        # 创建两个不同的 PeriodIndex 对象 pi1 和 pi2，频率为年，起始和结束日期不同
        pi1 = period_range(freq="Y", start="1/1/2001", end="12/1/2009")
        pi2 = period_range(freq="Y", start="1/1/2000", end="12/1/2008")
        # 检查两个 PeriodIndex 对象的长度是否相等
        assert len(pi1) == len(pi2)
        # 进行负数位移操作，并使用 pandas._testing.assert_index_equal 函数比较结果
        tm.assert_index_equal(pi1.shift(-1), pi2)

        # 创建两个不同的 PeriodIndex 对象 pi1 和 pi2，频率为月，起始和结束日期不同
        pi1 = period_range(freq="M", start="1/1/2001", end="12/1/2009")
        pi2 = period_range(freq="M", start="2/1/2001", end="1/1/2010")
        # 检查两个 PeriodIndex 对象的长度是否相等
        assert len(pi1) == len(pi2)
        # 进行位移操作，并使用 pandas._testing.assert_index_equal 函数比较结果
        tm.assert_index_equal(pi1.shift(1), pi2)

        # 创建两个不同的 PeriodIndex 对象 pi1 和 pi2，频率为月，起始和结束日期不同
        pi1 = period_range(freq="M", start="1/1/2001", end="12/1/2009")
        pi2 = period_range(freq="M", start="12/1/2000", end="11/1/2009")
        # 检查两个 PeriodIndex 对象的长度是否相等
        assert len(pi1) == len(pi2)
        # 进行负数位移操作，并使用 pandas._testing.assert_index_equal 函数比较结果
        tm.assert_index_equal(pi1.shift(-1), pi2)

        # 创建两个不同的 PeriodIndex 对象 pi1 和 pi2，频率为日，起始和结束日期不同
        pi1 = period_range(freq="D", start="1/1/2001", end="12/1/2009")
        pi2 = period_range(freq="D", start="1/2/2001", end="12/2/2009")
        # 检查两个 PeriodIndex 对象的长度是否相等
        assert len(pi1) == len(pi2)
        # 进行位移操作，并使用 pandas._testing.assert_index_equal 函数比较结果
        tm.assert_index_equal(pi1.shift(1), pi2)

        # 创建两个不同的 PeriodIndex 对象 pi1 和 pi2，频率为日，起始和结束日期不同
        pi1 = period_range(freq="D", start="1/1/2001", end="12/1/2009")
        pi2 = period_range(freq="D", start="12/31/2000", end="11/30/2009")
        # 检查两个 PeriodIndex 对象的长度是否相等
        assert len(pi1) == len(pi2)
        # 进行负数位移操作，并使用 pandas._testing.assert_index_equal 函数比较结果
        tm.assert_index_equal(pi1.shift(-1), pi2)
    def test_shift_corner_cases(self):
        # 创建一个空的 PeriodIndex 对象，用于测试边缘情况
        idx = PeriodIndex([], name="xxx", freq="h")

        # 定义错误消息，用于检测是否会抛出 TypeError 异常
        msg = "`freq` argument is not supported for PeriodIndex.shift"
        
        # 断言期望抛出 TypeError 异常，并且错误消息匹配定义的 msg
        with pytest.raises(TypeError, match=msg):
            # 调用 idx.shift 方法，传递期望抛出异常的参数 freq="h"
            idx.shift(1, freq="h")

        # 断言当参数为 0 时，idx.shift 返回自身
        tm.assert_index_equal(idx.shift(0), idx)
        
        # 断言当参数为 3 时，idx.shift 返回自身
        tm.assert_index_equal(idx.shift(3), idx)

        # 创建一个包含时间戳的 PeriodIndex 对象，用于进一步测试
        idx = PeriodIndex(
            ["2011-01-01 10:00", "2011-01-01 11:00", "2011-01-01 12:00"],
            name="xxx",
            freq="h",
        )
        
        # 断言当参数为 0 时，idx.shift 返回自身
        tm.assert_index_equal(idx.shift(0), idx)
        
        # 创建一个期望的结果 PeriodIndex 对象，用于测试参数为 3 时的偏移
        exp = PeriodIndex(
            ["2011-01-01 13:00", "2011-01-01 14:00", "2011-01-01 15:00"],
            name="xxx",
            freq="h",
        )
        
        # 断言当参数为 3 时，idx.shift 返回期望的 exp 结果
        tm.assert_index_equal(idx.shift(3), exp)
        
        # 创建一个期望的结果 PeriodIndex 对象，用于测试参数为 -3 时的偏移
        exp = PeriodIndex(
            ["2011-01-01 07:00", "2011-01-01 08:00", "2011-01-01 09:00"],
            name="xxx",
            freq="h",
        )
        
        # 断言当参数为 -3 时，idx.shift 返回期望的 exp 结果
        tm.assert_index_equal(idx.shift(-3), exp)

    def test_shift_nat(self):
        # 创建一个包含 NaT 值的 PeriodIndex 对象，用于测试
        idx = PeriodIndex(
            ["2011-01", "2011-02", "NaT", "2011-04"], freq="M", name="idx"
        )
        
        # 执行 idx.shift(1) 操作
        result = idx.shift(1)
        
        # 创建一个期望的结果 PeriodIndex 对象
        expected = PeriodIndex(
            ["2011-02", "2011-03", "NaT", "2011-05"], freq="M", name="idx"
        )
        
        # 断言 idx.shift(1) 返回期望的 expected 结果
        tm.assert_index_equal(result, expected)
        
        # 断言结果对象的名称与期望结果相同
        assert result.name == expected.name

    def test_shift_gh8083(self):
        # 创建一个 PeriodIndex 对象，用于测试 shift 方法
        # GH#8083
        drange = period_range("20130101", periods=5, freq="D")
        
        # 执行 drange.shift(1) 操作
        result = drange.shift(1)
        
        # 创建一个期望的结果 PeriodIndex 对象
        expected = PeriodIndex(
            ["2013-01-02", "2013-01-03", "2013-01-04", "2013-01-05", "2013-01-06"],
            freq="D",
        )
        
        # 断言 drange.shift(1) 返回期望的 expected 结果
        tm.assert_index_equal(result, expected)

    def test_shift_periods(self):
        # 创建一个 PeriodIndex 对象，用于测试 shift 方法的 periods 参数
        # GH #22458 : argument 'n' was deprecated in favor of 'periods'
        idx = period_range(freq="Y", start="1/1/2001", end="12/1/2009")
        
        # 断言当 periods=0 时，idx.shift 返回自身
        tm.assert_index_equal(idx.shift(periods=0), idx)
        
        # 断言当 periods=0 (使用 n 参数，已被废弃) 时，idx.shift 返回自身
        tm.assert_index_equal(idx.shift(0), idx)
```