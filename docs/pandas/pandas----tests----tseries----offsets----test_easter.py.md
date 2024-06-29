# `D:\src\scipysrc\pandas\pandas\tests\tseries\offsets\test_easter.py`

```
"""
Tests for the following offsets:
- Easter
"""

from __future__ import annotations  # 导入未来版本的 annotations 特性

from datetime import datetime  # 导入 datetime 模块中的 datetime 类

import pytest  # 导入 pytest 测试框架

from pandas.tests.tseries.offsets.common import assert_offset_equal  # 导入偏移量测试的公共函数

from pandas.tseries.offsets import Easter  # 导入复活节偏移量类


class TestEaster:
    @pytest.mark.parametrize(  # 使用 pytest 的 parametrize 装饰器进行参数化测试
        "offset,date,expected",  # 参数化的参数列表
        [
            (Easter(), datetime(2010, 1, 1), datetime(2010, 4, 4)),  # 测试复活节偏移量对日期的影响
            (Easter(), datetime(2010, 4, 5), datetime(2011, 4, 24)),
            (Easter(2), datetime(2010, 1, 1), datetime(2011, 4, 24)),
            (Easter(), datetime(2010, 4, 4), datetime(2011, 4, 24)),
            (Easter(2), datetime(2010, 4, 4), datetime(2012, 4, 8)),
            (-Easter(), datetime(2011, 1, 1), datetime(2010, 4, 4)),
            (-Easter(), datetime(2010, 4, 5), datetime(2010, 4, 4)),
            (-Easter(2), datetime(2011, 1, 1), datetime(2009, 4, 12)),
            (-Easter(), datetime(2010, 4, 4), datetime(2009, 4, 12)),
            (-Easter(2), datetime(2010, 4, 4), datetime(2008, 3, 23)),
        ],
    )
    def test_offset(self, offset, date, expected):
        assert_offset_equal(offset, date, expected)  # 调用断言函数验证复活节偏移量的计算是否正确
```