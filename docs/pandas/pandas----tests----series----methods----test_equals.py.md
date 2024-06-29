# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_equals.py`

```
# 导入所需模块和函数
from contextlib import nullcontext  # 从contextlib模块导入nullcontext上下文管理器
import copy  # 导入copy模块，用于对象复制

import numpy as np  # 导入NumPy库并重命名为np
import pytest  # 导入pytest测试框架

from pandas._libs.missing import is_matching_na  # 从pandas._libs.missing模块导入is_matching_na函数
from pandas.compat.numpy import np_version_gte1p25  # 从pandas.compat.numpy导入np_version_gte1p25函数

from pandas.core.dtypes.common import is_float  # 从pandas.core.dtypes.common模块导入is_float函数

from pandas import (  # 导入pandas库中的以下类和函数
    Index,  # Index类，用于创建索引对象
    MultiIndex,  # MultiIndex类，用于创建多级索引对象
    Series,  # Series类，用于创建序列对象
)
import pandas._testing as tm  # 导入pandas._testing模块并重命名为tm


@pytest.mark.parametrize(  # 使用pytest.mark.parametrize装饰器定义参数化测试
    "arr, idx",  # 参数化测试的参数命名为arr和idx
    [  # 参数化的测试数据列表
        ([1, 2, 3, 4], [0, 2, 1, 3]),  # 第一组参数：列表和索引列表
        ([1, np.nan, 3, np.nan], [0, 2, 1, 3]),  # 第二组参数：包含NaN的列表和索引列表
        (  # 第三组参数：包含NaN的列表和MultiIndex多级索引
            [1, np.nan, 3, np.nan],
            MultiIndex.from_tuples([(0, "a"), (1, "b"), (2, "c"), (3, "c")]),
        ),
    ],
)
def test_equals(arr, idx):
    s1 = Series(arr, index=idx)  # 创建Series对象s1，使用arr和idx作为数据和索引
    s2 = s1.copy()  # 复制s1生成s2
    assert s1.equals(s2)  # 断言s1和s2相等

    s1[1] = 9  # 修改s1的第一个元素
    assert not s1.equals(s2)  # 断言s1和s2不相等


@pytest.mark.parametrize(  # 使用pytest.mark.parametrize装饰器定义参数化测试
    "val", [1, 1.1, 1 + 1j, True, "abc", [1, 2], (1, 2), {1, 2}, {"a": 1}, None]  # 参数化测试的参数命名为val
)
def test_equals_list_array(val):
    # GH20676 验证Numpy数组列表的等号操作符
    arr = np.array([1, 2])  # 创建NumPy数组arr
    s1 = Series([arr, arr])  # 创建包含两个arr的Series对象s1
    s2 = s1.copy()  # 复制s1生成s2
    assert s1.equals(s2)  # 断言s1和s2相等

    s1[1] = val  # 修改s1的第二个元素

    cm = (  # 根据条件创建上下文管理器cm
        tm.assert_produces_warning(FutureWarning, check_stacklevel=False)  # 如果val是字符串且不满足np_version_gte1p25，则产生FutureWarning警告
        if isinstance(val, str) and not np_version_gte1p25 else nullcontext()
    )
    with cm:  # 使用上下文管理器cm
        assert not s1.equals(s2)  # 断言s1和s2不相等


def test_equals_false_negative():
    # GH8437 验证equals函数对dtype为object的false negative行为
    arr = [False, np.nan]  # 创建包含布尔值和NaN的列表arr
    s1 = Series(arr)  # 创建包含arr的Series对象s1
    s2 = s1.copy()  # 复制s1生成s2
    s3 = Series(index=range(2), dtype=object)  # 创建dtype为object的索引为range(2)的Series对象s3
    s4 = s3.copy()  # 复制s3生成s4
    s5 = s3.copy()  # 复制s3生成s5
    s6 = s3.copy()  # 复制s3生成s6

    s3[:-1] = s4[:-1] = s5[0] = s6[0] = False  # 修改s3、s4、s5和s6的部分元素
    assert s1.equals(s1)  # 断言s1等于自身
    assert s1.equals(s2)  # 断言s1等于s2
    assert s1.equals(s3)  # 断言s1等于s3
    assert s1.equals(s4)  # 断言s1等于s4
    assert s1.equals(s5)  # 断言s1等于s5
    assert s5.equals(s6)  # 断言s5等于s6


def test_equals_matching_nas():
    # 匹配但不完全相同的NAs
    left = Series([np.datetime64("NaT")], dtype=object)  # 创建dtype为object的包含NaT的Series对象left
    right = Series([np.datetime64("NaT")], dtype=object)  # 创建dtype为object的包含NaT的Series对象right
    assert left.equals(right)  # 断言left等于right
    assert Index(left).equals(Index(right))  # 断言Index(left)等于Index(right)
    assert left.array.equals(right.array)  # 断言left.array等于right.array

    left = Series([np.timedelta64("NaT")], dtype=object)  # 创建dtype为object的包含NaT的Series对象left
    right = Series([np.timedelta64("NaT")], dtype=object)  # 创建dtype为object的包含NaT的Series对象right
    assert left.equals(right)  # 断言left等于right
    assert Index(left).equals(Index(right))  # 断言Index(left)等于Index(right)
    assert left.array.equals(right.array)  # 断言left.array等于right.array

    left = Series([np.float64("NaN")], dtype=object)  # 创建dtype为object的包含NaN的Series对象left
    right = Series([np.float64("NaN")], dtype=object)  # 创建dtype为object的包含NaN的Series对象right
    assert left.equals(right)  # 断言left等于right
    assert Index(left, dtype=left.dtype).equals(Index(right, dtype=right.dtype))  # 断言Index(left, dtype=left.dtype)等于Index(right, dtype=right.dtype)
    assert left.array.equals(right.array)  # 断言left.array等于right.array


def test_equals_mismatched_nas(nulls_fixture, nulls_fixture2):
    # GH#39650
    left = nulls_fixture  # 使用nulls_fixture创建left对象
    right = nulls_fixture2  # 使用nulls_fixture2创建right对象
    if hasattr(right, "copy"):  # 如果right对象具有copy属性
        right = right.copy()  # 复制right对象
    else:
        right = copy.copy(right)  # 否则使用copy模块的copy函数复制right对象

    ser = Series([left], dtype=object)  # 创建dtype为object的包含left的Series对象ser
    ser2 = Series([right], dtype=object)  # 创建dtype为object的包含right的Series对象ser2

    if is_matching_na(left, right):  # 如果left和right匹配NaN
        assert ser.equals(ser2)  # 断言ser等于ser2
    # 如果左边的值为 None 并且右边的值是浮点数，或者右边的值为 None 并且左边的值是浮点数，则执行以下操作
    elif (left is None and is_float(right)) or (right is None and is_float(left)):
        # 断言两个序列（ser 和 ser2）相等
        assert ser.equals(ser2)
    # 如果以上条件都不满足，则执行以下操作
    else:
        # 断言两个序列（ser 和 ser2）不相等
        assert not ser.equals(ser2)
# GH#39650
# 创建包含整数和None的Series对象，dtype设置为object类型
ser = Series([1, None], dtype=object)
# 创建包含整数和NaN的Series对象，dtype设置为object类型
ser2 = Series([1, np.nan], dtype=object)

# 断言ser与ser2的内容相等
assert ser.equals(ser2)
# 断言使用ser和ser2创建的Index对象相等，dtype保持一致
assert Index(ser, dtype=ser.dtype).equals(Index(ser2, dtype=ser2.dtype))
# 断言ser和ser2的底层数据数组相等
assert ser.array.equals(ser2.array)


# GH#44190
# 创建包含特定值的Series对象，dtype设置为object类型
left = Series([-np.inf, np.nan, -1.0, 0.0, 1.0, 10 / 3, np.inf], dtype=object)
# 创建与left相同长度但值全为None的Series对象
right = Series([None] * len(left))

# 这些Series对象由于一个bug被错误地认为相等，因此在此验证它们实际上不相等
# 断言left与right不相等
assert not left.equals(right)
# 断言right与left不相等
assert not right.equals(left)
# 断言left和right转为DataFrame后不相等
assert not left.to_frame().equals(right.to_frame())
# 断言right和left转为DataFrame后不相等
assert not right.to_frame().equals(left.to_frame())
# 断言使用left和right创建的Index对象不相等，dtype设置为object类型
assert not Index(left, dtype="object").equals(Index(right, dtype="object"))
# 断言使用right和left创建的Index对象不相等，dtype设置为object类型
assert not Index(right, dtype="object").equals(Index(left, dtype="object"))
```