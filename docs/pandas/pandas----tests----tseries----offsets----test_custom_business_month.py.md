# `D:\src\scipysrc\pandas\pandas\tests\tseries\offsets\test_custom_business_month.py`

```
"""
Tests for the following offsets:
- CustomBusinessMonthBase
- CustomBusinessMonthBegin
- CustomBusinessMonthEnd
"""

# 引入必要的库和模块
from __future__ import annotations

from datetime import (
    date,             # 导入 date 类
    datetime,         # 导入 datetime 类
    timedelta,        # 导入 timedelta 类
)

import numpy as np   # 导入 NumPy 库
import pytest        # 导入 Pytest 测试框架

# 导入 pandas 库中的时间序列偏移对象
from pandas._libs.tslibs.offsets import (
    CBMonthBegin,    # 导入 CustomBusinessMonthBegin 偏移对象
    CBMonthEnd,      # 导入 CustomBusinessMonthEnd 偏移对象
    CDay,            # 导入 CustomBusinessDay 偏移对象
)

import pandas._testing as tm   # 导入 pandas 测试工具
from pandas.tests.tseries.offsets.common import (
    assert_is_on_offset,    # 导入用于断言偏移对象的工具函数
    assert_offset_equal,    # 导入用于断言两个偏移对象相等的工具函数
)

from pandas.tseries import offsets   # 导入 pandas 时间序列偏移模块


@pytest.fixture
def dt():
    return datetime(2008, 1, 1)


class TestCommonCBM:
    @pytest.mark.parametrize("offset2", [CBMonthBegin(2), CBMonthEnd(2)])
    def test_eq(self, offset2):
        assert offset2 == offset2

    @pytest.mark.parametrize("offset2", [CBMonthBegin(2), CBMonthEnd(2)])
    def test_hash(self, offset2):
        assert hash(offset2) == hash(offset2)

    @pytest.mark.parametrize("_offset", [CBMonthBegin, CBMonthEnd])
    def test_roundtrip_pickle(self, _offset):
        def _check_roundtrip(obj):
            unpickled = tm.round_trip_pickle(obj)
            assert unpickled == obj

        _check_roundtrip(_offset())
        _check_roundtrip(_offset(2))
        _check_roundtrip(_offset() * 2)

    @pytest.mark.parametrize("_offset", [CBMonthBegin, CBMonthEnd])
    def test_copy(self, _offset):
        # GH 17452
        off = _offset(weekmask="Mon Wed Fri")
        assert off == off.copy()


class TestCustomBusinessMonthBegin:
    def test_different_normalize_equals(self):
        # GH#21404 changed __eq__ to return False when `normalize` does not match
        offset = CBMonthBegin()
        offset2 = CBMonthBegin(normalize=True)
        assert offset != offset2

    def test_repr(self):
        assert repr(CBMonthBegin()) == "<CustomBusinessMonthBegin>"
        assert repr(CBMonthBegin(2)) == "<2 * CustomBusinessMonthBegins>"

    def test_add_datetime(self, dt):
        assert CBMonthBegin(2) + dt == datetime(2008, 3, 3)

    def testRollback1(self):
        assert CDay(10).rollback(datetime(2007, 12, 31)) == datetime(2007, 12, 31)

    def testRollback2(self, dt):
        assert CBMonthBegin(10).rollback(dt) == datetime(2008, 1, 1)

    def testRollforward1(self, dt):
        assert CBMonthBegin(10).rollforward(dt) == datetime(2008, 1, 1)

    def test_roll_date_object(self):
        offset = CBMonthBegin()

        dt = date(2012, 9, 15)

        result = offset.rollback(dt)
        assert result == datetime(2012, 9, 3)

        result = offset.rollforward(dt)
        assert result == datetime(2012, 10, 1)

        offset = offsets.Day()
        result = offset.rollback(dt)
        assert result == datetime(2012, 9, 15)

        result = offset.rollforward(dt)
        assert result == datetime(2012, 9, 15)

    on_offset_cases = [
        (CBMonthBegin(), datetime(2008, 1, 1), True),
        (CBMonthBegin(), datetime(2008, 1, 31), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    # 定义一个测试函数，用于验证 is_on_offset 函数的正确性
    def test_is_on_offset(self, case):
        offset, dt, expected = case
        # 断言 is_on_offset 函数的执行结果是否符合预期
        assert_is_on_offset(offset, dt, expected)

    # 定义应用案例列表，包含多个元组，每个元组代表一个测试案例
    apply_cases = [
        (
            CBMonthBegin(),  # 创建一个 CBMonthBegin 对象作为偏移量
            {  # 创建一个字典，包含 datetime 对象到 datetime 对象的映射
                datetime(2008, 1, 1): datetime(2008, 2, 1),
                datetime(2008, 2, 7): datetime(2008, 3, 3),
            },
        ),
        (
            2 * CBMonthBegin(),  # 创建 CBMonthBegin 对象的两倍作为偏移量
            {
                datetime(2008, 1, 1): datetime(2008, 3, 3),
                datetime(2008, 2, 7): datetime(2008, 4, 1),
            },
        ),
        (
            -CBMonthBegin(),  # 创建 CBMonthBegin 对象的负值作为偏移量
            {
                datetime(2008, 1, 1): datetime(2007, 12, 3),
                datetime(2008, 2, 8): datetime(2008, 2, 1),
            },
        ),
        (
            -2 * CBMonthBegin(),  # 创建 CBMonthBegin 对象的两倍负值作为偏移量
            {
                datetime(2008, 1, 1): datetime(2007, 11, 1),
                datetime(2008, 2, 9): datetime(2008, 1, 1),
            },
        ),
        (
            CBMonthBegin(0),  # 创建一个 CBMonthBegin 对象，指定偏移量为 0
            {
                datetime(2008, 1, 1): datetime(2008, 1, 1),
                datetime(2008, 1, 7): datetime(2008, 2, 1),
            },
        ),
    ]

    # 使用 pytest 的 parametrize 装饰器，对 test_apply 函数进行参数化测试
    @pytest.mark.parametrize("case", apply_cases)
    def test_apply(self, case):
        offset, cases = case
        # 遍历 cases 字典，对每个测试用例执行断言
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)

    # 定义测试函数，验证在较大的偏移量下的应用场景
    def test_apply_large_n(self):
        dt = datetime(2012, 10, 23)

        # 测试 CBMonthBegin(10) 的偏移效果
        result = dt + CBMonthBegin(10)
        assert result == datetime(2013, 8, 1)

        # 测试 CDay(100) - CDay(100) 的效果，应为原日期 dt
        result = dt + CDay(100) - CDay(100)
        assert result == dt

        # 测试 CBMonthBegin() * 6 的偏移效果
        off = CBMonthBegin() * 6
        rs = datetime(2012, 1, 1) - off
        xp = datetime(2011, 7, 1)
        assert rs == xp

        # 测试 CBMonthBegin() * 6 的逆向偏移效果
        st = datetime(2011, 12, 18)
        rs = st + off
        xp = datetime(2012, 6, 1)
        assert rs == xp

    # 定义测试函数，验证在假期设置下的偏移效果
    def test_holidays(self):
        # 定义一个 TradingDay 偏移量，指定了假期列表
        holidays = ["2012-02-01", datetime(2012, 2, 2), np.datetime64("2012-03-01")]
        bm_offset = CBMonthBegin(holidays=holidays)
        dt = datetime(2012, 1, 1)

        # 测试假期影响下 CBMonthBegin 的偏移效果
        assert dt + bm_offset == datetime(2012, 1, 2)
        assert dt + 2 * bm_offset == datetime(2012, 2, 3)
    # 使用 pytest 的 parametrize 装饰器为 test_apply_with_extra_offset 方法参数化测试用例
    @pytest.mark.parametrize(
        "case",
        [
            (
                CBMonthBegin(n=1, offset=timedelta(days=5)),  # 创建 CBMonthBegin 对象，设置 n=1, offset=5天
                {
                    datetime(2021, 3, 1): datetime(2021, 4, 1) + timedelta(days=5),  # 预期结果：2021年3月1日对应的结果是2021年4月1日加上5天
                    datetime(2021, 4, 17): datetime(2021, 5, 3) + timedelta(days=5),  # 预期结果：2021年4月17日对应的结果是2021年5月3日加上5天
                },
            ),
            (
                CBMonthBegin(n=2, offset=timedelta(days=40)),  # 创建 CBMonthBegin 对象，设置 n=2, offset=40天
                {
                    datetime(2021, 3, 10): datetime(2021, 5, 3) + timedelta(days=40),  # 预期结果：2021年3月10日对应的结果是2021年5月3日加上40天
                    datetime(2021, 4, 30): datetime(2021, 6, 1) + timedelta(days=40),   # 预期结果：2021年4月30日对应的结果是2021年6月1日加上40天
                },
            ),
            (
                CBMonthBegin(n=1, offset=timedelta(days=-5)),  # 创建 CBMonthBegin 对象，设置 n=1, offset=-5天
                {
                    datetime(2021, 3, 1): datetime(2021, 4, 1) - timedelta(days=5),  # 预期结果：2021年3月1日对应的结果是2021年4月1日减去5天
                    datetime(2021, 4, 11): datetime(2021, 5, 3) - timedelta(days=5),  # 预期结果：2021年4月11日对应的结果是2021年5月3日减去5天
                },
            ),
            (
                -2 * CBMonthBegin(n=1, offset=timedelta(days=10)),  # 创建 CBMonthBegin 对象，设置 n=-2, offset=10天
                {
                    datetime(2021, 3, 1): datetime(2021, 1, 1) + timedelta(days=10),  # 预期结果：2021年3月1日对应的结果是2021年1月1日加上10天
                    datetime(2021, 4, 3): datetime(2021, 3, 1) + timedelta(days=10),  # 预期结果：2021年4月3日对应的结果是2021年3月1日加上10天
                },
            ),
            (
                CBMonthBegin(n=0, offset=timedelta(days=1)),  # 创建 CBMonthBegin 对象，设置 n=0, offset=1天
                {
                    datetime(2021, 3, 2): datetime(2021, 4, 1) + timedelta(days=1),  # 预期结果：2021年3月2日对应的结果是2021年4月1日加上1天
                    datetime(2021, 4, 1): datetime(2021, 4, 1) + timedelta(days=1),  # 预期结果：2021年4月1日对应的结果是2021年4月1日加上1天
                },
            ),
            (
                CBMonthBegin(
                    n=1, holidays=["2021-04-01", "2021-04-02"], offset=timedelta(days=1)  # 创建 CBMonthBegin 对象，设置 n=1, offset=1天，指定假期列表
                ),
                {
                    datetime(2021, 3, 2): datetime(2021, 4, 5) + timedelta(days=1),  # 预期结果：2021年3月2日对应的结果是2021年4月5日加上1天
                },
            ),
        ],
    )
    # 定义测试方法 test_apply_with_extra_offset，传入参数 case
    def test_apply_with_extra_offset(self, case):
        offset, cases = case
        # 遍历 cases 中的每个测试用例
        for base, expected in cases.items():
            # 断言 offset 对象应用于 base 时的结果与预期的 expected 相等
            assert_offset_equal(offset, base, expected)
class TestCustomBusinessMonthEnd:
    def test_different_normalize_equals(self):
        # GH#21404 changed __eq__ to return False when `normalize` does not match
        # 创建一个 CBMonthEnd 的实例对象 offset
        offset = CBMonthEnd()
        # 创建另一个 CBMonthEnd 的实例对象 offset2，传入 normalize=True
        offset2 = CBMonthEnd(normalize=True)
        # 断言 offset 不等于 offset2
        assert offset != offset2

    def test_repr(self):
        # 断言 CBMonthEnd() 对象的字符串表示为 "<CustomBusinessMonthEnd>"
        assert repr(CBMonthEnd()) == "<CustomBusinessMonthEnd>"
        # 断言 CBMonthEnd(2) 对象的字符串表示为 "<2 * CustomBusinessMonthEnds>"
        assert repr(CBMonthEnd(2)) == "<2 * CustomBusinessMonthEnds>"

    def test_add_datetime(self, dt):
        # 断言 CBMonthEnd(2) 对象加上 dt 后等于 datetime(2008, 2, 29)
        assert CBMonthEnd(2) + dt == datetime(2008, 2, 29)

    def testRollback1(self):
        # 断言 CDay(10) 对象对 datetime(2007, 12, 31) 进行 rollback 后等于 datetime(2007, 12, 31)
        assert CDay(10).rollback(datetime(2007, 12, 31)) == datetime(2007, 12, 31)

    def testRollback2(self, dt):
        # 断言 CBMonthEnd(10) 对象对 dt 进行 rollback 后等于 datetime(2007, 12, 31)
        assert CBMonthEnd(10).rollback(dt) == datetime(2007, 12, 31)

    def testRollforward1(self, dt):
        # 断言 CBMonthEnd(10) 对象对 dt 进行 rollforward 后等于 datetime(2008, 1, 31)
        assert CBMonthEnd(10).rollforward(dt) == datetime(2008, 1, 31)

    def test_roll_date_object(self):
        # 创建一个 CBMonthEnd 的实例对象 offset
        offset = CBMonthEnd()
        # 创建一个 date 对象 dt
        dt = date(2012, 9, 15)
        
        # 断言 offset 对象对 dt 进行 rollback 后等于 datetime(2012, 8, 31)
        result = offset.rollback(dt)
        assert result == datetime(2012, 8, 31)
        
        # 断言 offset 对象对 dt 进行 rollforward 后等于 datetime(2012, 9, 28)
        result = offset.rollforward(dt)
        assert result == datetime(2012, 9, 28)
        
        # 创建一个 offsets.Day 的实例对象 offset
        offset = offsets.Day()
        # 断言 offset 对象对 dt 进行 rollback 后等于 datetime(2012, 9, 15)
        result = offset.rollback(dt)
        assert result == datetime(2012, 9, 15)
        
        # 断言 offset 对象对 dt 进行 rollforward 后等于 datetime(2012, 9, 15)
        result = offset.rollforward(dt)
        assert result == datetime(2012, 9, 15)

    on_offset_cases = [
        # 定义一组测试用例，每个元素包含 CBMonthEnd() 对象、datetime(2008, 1, 31) 和预期的布尔值
        (CBMonthEnd(), datetime(2008, 1, 31), True),
        (CBMonthEnd(), datetime(2008, 1, 1), False),
    ]

    @pytest.mark.parametrize("case", on_offset_cases)
    def test_is_on_offset(self, case):
        # 解包测试用例
        offset, dt, expected = case
        # 调用 assert_is_on_offset 函数，断言 offset 对象对 dt 的 is_on_offset 结果与 expected 相等
        assert_is_on_offset(offset, dt, expected)

    apply_cases = [
        # 定义一组 apply 函数的测试用例，每个元素包含一个 offset 对象和一个包含 datetime 对应结果的字典 cases
        (
            CBMonthEnd(),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 2, 7): datetime(2008, 2, 29),
            },
        ),
        (
            2 * CBMonthEnd(),
            {
                datetime(2008, 1, 1): datetime(2008, 2, 29),
                datetime(2008, 2, 7): datetime(2008, 3, 31),
            },
        ),
        (
            -CBMonthEnd(),
            {
                datetime(2008, 1, 1): datetime(2007, 12, 31),
                datetime(2008, 2, 8): datetime(2008, 1, 31),
            },
        ),
        (
            -2 * CBMonthEnd(),
            {
                datetime(2008, 1, 1): datetime(2007, 11, 30),
                datetime(2008, 2, 9): datetime(2007, 12, 31),
            },
        ),
        (
            CBMonthEnd(0),
            {
                datetime(2008, 1, 1): datetime(2008, 1, 31),
                datetime(2008, 2, 7): datetime(2008, 2, 29),
            },
        ),
    ]

    @pytest.mark.parametrize("case", apply_cases)
    def test_apply(self, case):
        # 解包测试用例
        offset, cases = case
        # 遍历 cases 字典，每个键值对进行断言验证 offset 对象的 apply 函数应用于 base 的结果等于 expected
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)
    def test_apply_large_n(self):
        # 创建一个 datetime 对象，表示2012年10月23日
        dt = datetime(2012, 10, 23)

        # 对 dt 应用 CBMonthEnd(10) 偏移量，预期结果为2013年7月31日
        result = dt + CBMonthEnd(10)
        assert result == datetime(2013, 7, 31)

        # 对 dt 应用 CDay(100) 偏移量，再减去 CDay(100) 偏移量，预期结果为原始 dt
        result = dt + CDay(100) - CDay(100)
        assert result == dt

        # 创建 CBMonthEnd() * 6 的偏移量对象
        off = CBMonthEnd() * 6

        # 用 datetime(2012, 1, 1) 减去 off 偏移量，预期结果为 datetime(2011, 7, 29)
        rs = datetime(2012, 1, 1) - off

        # xp 为预期结果 datetime(2011, 7, 29)
        xp = datetime(2011, 7, 29)
        assert rs == xp

        # 创建 datetime(2011, 12, 18)
        st = datetime(2011, 12, 18)

        # 对 st 应用 off 偏移量，预期结果为 datetime(2012, 5, 31)
        rs = st + off

        # xp 为预期结果 datetime(2012, 5, 31)
        xp = datetime(2012, 5, 31)
        assert rs == xp

    def test_holidays(self):
        # 定义一个 CBMonthEnd 对象，其中包含假期列表
        holidays = ["2012-01-31", datetime(2012, 2, 28), np.datetime64("2012-02-29")]
        bm_offset = CBMonthEnd(holidays=holidays)

        # 创建 datetime(2012, 1, 1)
        dt = datetime(2012, 1, 1)

        # 对 dt 应用 bm_offset 偏移量，预期结果为 datetime(2012, 1, 30)
        assert dt + bm_offset == datetime(2012, 1, 30)

        # 对 dt 应用 2 * bm_offset 偏移量，预期结果为 datetime(2012, 2, 27)
        assert dt + 2 * bm_offset == datetime(2012, 2, 27)

    @pytest.mark.parametrize(
        "case",
        [
            (
                # 创建 CBMonthEnd(n=1, offset=timedelta(days=5)) 对象
                CBMonthEnd(n=1, offset=timedelta(days=5)),
                {
                    # 测试用例1
                    datetime(2021, 3, 1): datetime(2021, 3, 31) + timedelta(days=5),
                    datetime(2021, 4, 17): datetime(2021, 4, 30) + timedelta(days=5),
                },
            ),
            (
                # 创建 CBMonthEnd(n=2, offset=timedelta(days=40)) 对象
                CBMonthEnd(n=2, offset=timedelta(days=40)),
                {
                    # 测试用例2
                    datetime(2021, 3, 10): datetime(2021, 4, 30) + timedelta(days=40),
                    datetime(2021, 4, 30): datetime(2021, 6, 30) + timedelta(days=40),
                },
            ),
            (
                # 创建 CBMonthEnd(n=1, offset=timedelta(days=-5)) 对象
                CBMonthEnd(n=1, offset=timedelta(days=-5)),
                {
                    # 测试用例3
                    datetime(2021, 3, 1): datetime(2021, 3, 31) - timedelta(days=5),
                    datetime(2021, 4, 11): datetime(2021, 4, 30) - timedelta(days=5),
                },
            ),
            (
                # 创建 -2 * CBMonthEnd(n=1, offset=timedelta(days=10)) 对象
                -2 * CBMonthEnd(n=1, offset=timedelta(days=10)),
                {
                    # 测试用例4
                    datetime(2021, 3, 1): datetime(2021, 1, 29) + timedelta(days=10),
                    datetime(2021, 4, 3): datetime(2021, 2, 26) + timedelta(days=10),
                },
            ),
            (
                # 创建 CBMonthEnd(n=0, offset=timedelta(days=1)) 对象
                CBMonthEnd(n=0, offset=timedelta(days=1)),
                {
                    # 测试用例5
                    datetime(2021, 3, 2): datetime(2021, 3, 31) + timedelta(days=1),
                    datetime(2021, 4, 1): datetime(2021, 4, 30) + timedelta(days=1),
                },
            ),
            (
                # 创建 CBMonthEnd(n=1, holidays=["2021-03-31"], offset=timedelta(days=1)) 对象
                CBMonthEnd(n=1, holidays=["2021-03-31"], offset=timedelta(days=1)),
                {
                    # 测试用例6
                    datetime(2021, 3, 2): datetime(2021, 3, 30) + timedelta(days=1),
                },
            ),
        ],
    )
    def test_apply_with_extra_offset(self, case):
        offset, cases = case

        # 遍历测试用例集合
        for base, expected in cases.items():
            assert_offset_equal(offset, base, expected)
```