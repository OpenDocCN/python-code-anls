# `D:\src\scipysrc\pandas\pandas\tests\series\indexing\test_mask.py`

```
import numpy as np
import pytest

from pandas import Series
import pandas._testing as tm

# 定义测试函数，用于测试 Series 的 mask 方法
def test_mask():
    # 创建一个包含随机标准正态分布数据的 Series 对象
    s = Series(np.random.default_rng(2).standard_normal(5))
    # 创建一个条件，判断 Series 中哪些值大于 0
    cond = s > 0

    # 使用 where 方法，将满足条件的值设为 NaN，并进行断言比较
    rs = s.where(~cond, np.nan)
    tm.assert_series_equal(rs, s.mask(cond))

    # 使用 where 方法，仅指定条件，进行断言比较
    rs = s.where(~cond)
    rs2 = s.mask(cond)
    tm.assert_series_equal(rs, rs2)

    # 使用 where 方法，指定条件并设定替换值为相反数，进行断言比较
    rs = s.where(~cond, -s)
    rs2 = s.mask(cond, -s)
    tm.assert_series_equal(rs, rs2)

    # 创建一个新的条件 Series，指定一部分索引位置为 True
    cond = Series([True, False, False, True, False], index=s.index)
    # 对 s 取绝对值并取反，得到 s2
    s2 = -(s.abs())

    # 使用 where 方法，仅指定部分条件，进行断言比较
    rs = s2.where(~cond[:3])
    rs2 = s2.mask(cond[:3])
    tm.assert_series_equal(rs, rs2)

    # 使用 where 方法，指定部分条件并设定替换值为相反数，进行断言比较
    rs = s2.where(~cond[:3], -s2)
    rs2 = s2.mask(cond[:3], -s2)
    tm.assert_series_equal(rs, rs2)

    # 测试当条件数组形状不匹配时是否会引发 ValueError 异常
    msg = "Array conditional must be same shape as self"
    with pytest.raises(ValueError, match=msg):
        s.mask(1)
    with pytest.raises(ValueError, match=msg):
        s.mask(cond[:3].values, -s)


# 定义测试函数，用于测试 Series 的 mask 方法的类型转换
def test_mask_casts():
    # 创建一个包含整数的 Series 对象
    ser = Series([1, 2, 3, 4])
    # 使用 mask 方法，将大于 2 的值设为 NaN，并进行断言比较
    result = ser.mask(ser > 2, np.nan)
    expected = Series([1, 2, np.nan, np.nan])
    tm.assert_series_equal(result, expected)


# 定义测试函数，用于测试 Series 的 mask 方法对布尔数组的处理
def test_mask_casts2():
    # 创建一个包含整数的 Series 对象
    ser = Series([1, 2])
    # 使用 mask 方法，根据布尔数组进行条件筛选，并进行断言比较
    res = ser.mask([True, False])

    # 期望结果中第一个值应为 NaN，其它值不变
    exp = Series([np.nan, 2])
    tm.assert_series_equal(res, exp)


# 定义测试函数，用于测试 Series 的 mask 方法的就地修改功能
def test_mask_inplace():
    # 创建一个包含随机标准正态分布数据的 Series 对象
    s = Series(np.random.default_rng(2).standard_normal(5))
    # 创建一个条件，判断 Series 中哪些值大于 0
    cond = s > 0

    # 复制 Series 对象，进行 inplace 的 mask 操作，并进行断言比较
    rs = s.copy()
    rs.mask(cond, inplace=True)
    tm.assert_series_equal(rs.dropna(), s[~cond])
    tm.assert_series_equal(rs, s.mask(cond))

    # 复制 Series 对象，进行 inplace 的 mask 操作，设定替换值为相反数，并进行断言比较
    rs = s.copy()
    rs.mask(cond, -s, inplace=True)
    tm.assert_series_equal(rs, s.mask(cond, -s))
```