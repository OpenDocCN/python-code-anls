# `D:\src\scipysrc\pandas\pandas\tests\base\test_fillna.py`

```
"""
Though Index.fillna and Series.fillna has separate impl,
test here to confirm these works as the same
"""

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 Pytest 测试框架

from pandas import MultiIndex  # 从 Pandas 导入 MultiIndex 类
import pandas._testing as tm  # 导入 Pandas 测试模块
from pandas.tests.base.common import allow_na_ops  # 从 Pandas 测试模块导入 allow_na_ops 函数


def test_fillna(index_or_series_obj):
    # GH 11343
    obj = index_or_series_obj  # 将传入的 index_or_series_obj 赋值给变量 obj

    if isinstance(obj, MultiIndex):
        msg = "isna is not defined for MultiIndex"
        # 如果 obj 是 MultiIndex 类型，则抛出 NotImplementedError 异常，匹配异常信息 msg
        with pytest.raises(NotImplementedError, match=msg):
            obj.fillna(0)
        return  # 函数返回，结束执行

    # values will not be changed
    fill_value = obj.values[0] if len(obj) > 0 else 0
    # 获取 obj 的第一个值作为 fill_value，如果 obj 不为空
    result = obj.fillna(fill_value)  # 使用 fill_value 来填充缺失值

    tm.assert_equal(obj, result)  # 使用 Pandas 测试模块的 assert_equal 函数比较 obj 和 result

    # check shallow_copied
    assert obj is not result  # 断言 obj 和 result 是不同的对象


@pytest.mark.parametrize("null_obj", [np.nan, None])
def test_fillna_null(null_obj, index_or_series_obj):
    # GH 11343
    obj = index_or_series_obj  # 将传入的 index_or_series_obj 赋值给变量 obj
    klass = type(obj)  # 获取 obj 的类型，并赋值给变量 klass

    if not allow_na_ops(obj):
        pytest.skip(f"{klass} doesn't allow for NA operations")
        # 如果 obj 不支持 NA 操作，则跳过测试，给出相应的提示信息
    elif len(obj) < 1:
        pytest.skip("Test doesn't make sense on empty data")
        # 如果 obj 的长度小于 1，则跳过测试，给出相应的提示信息
    elif isinstance(obj, MultiIndex):
        pytest.skip(f"MultiIndex can't hold '{null_obj}'")
        # 如果 obj 是 MultiIndex 类型，则跳过测试，给出相应的提示信息

    values = obj._values  # 获取 obj 的值数组，并赋值给变量 values
    fill_value = values[0]  # 获取 values 的第一个值作为 fill_value
    expected = values.copy()  # 复制 values 数组并赋值给 expected
    values[0:2] = null_obj  # 将 values 数组的前两个元素设置为 null_obj
    expected[0:2] = fill_value  # 将 expected 数组的前两个元素设置为 fill_value

    expected = klass(expected)  # 使用 klass 类型将 expected 数组重新封装成对象
    obj = klass(values)  # 使用 klass 类型将 values 数组重新封装成对象

    result = obj.fillna(fill_value)  # 使用 fill_value 来填充 obj 中的缺失值
    tm.assert_equal(result, expected)  # 使用 Pandas 测试模块的 assert_equal 函数比较 result 和 expected

    # check shallow_copied
    assert obj is not result  # 断言 obj 和 result 是不同的对象
```