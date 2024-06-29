# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\methods\test_snap.py`

```
# 导入 pytest 库，用于测试框架
import pytest

# 从 pandas 库中导入 DatetimeIndex 和 date_range 类
from pandas import (
    DatetimeIndex,
    date_range,
)

# 导入 pandas._testing 库，用于测试辅助功能
import pandas._testing as tm

# 使用 pytest.mark.parametrize 装饰器，定义参数化测试，参数包括时区和名称
@pytest.mark.parametrize("tz", [None, "Asia/Shanghai", "Europe/Berlin"])
@pytest.mark.parametrize("name", [None, "my_dti"])
# 定义测试函数 test_dti_snap，接受名称、时区和单元参数
def test_dti_snap(name, tz, unit):
    # 创建 DatetimeIndex 对象 dti，包含指定日期范围、名称、时区和频率
    dti = DatetimeIndex(
        [
            "1/1/2002",
            "1/2/2002",
            "1/3/2002",
            "1/4/2002",
            "1/5/2002",
            "1/6/2002",
            "1/7/2002",
        ],
        name=name,  # 设置对象的名称
        tz=tz,  # 设置对象的时区
        freq="D",  # 设置对象的频率为每天
    )
    
    # 将 dti 对象重新设置为指定单元
    dti = dti.as_unit(unit)

    # 对 dti 对象执行 snap 操作，设置频率为每周一，并将结果保存到 result 变量中
    result = dti.snap(freq="W-MON")
    
    # 创建预期的 DatetimeIndex 对象 expected，包含指定日期范围、名称、时区和频率
    expected = date_range("12/31/2001", "1/7/2002", name=name, tz=tz, freq="w-mon")
    expected = expected.repeat([3, 4])  # 重复序列，以生成与结果相同的索引
    
    # 将预期对象重新设置为指定单元
    expected = expected.as_unit(unit)
    
    # 使用 pandas._testing.assert_index_equal 函数比较 result 和 expected 对象
    tm.assert_index_equal(result, expected)
    
    # 断言 result 对象的时区与 expected 对象相同
    assert result.tz == expected.tz
    
    # 断言 result 对象的频率为 None
    assert result.freq is None
    
    # 断言 expected 对象的频率为 None
    assert expected.freq is None

    # 对 dti 对象执行 snap 操作，设置频率为每工作日，并将结果保存到 result 变量中
    result = dti.snap(freq="B")
    
    # 创建预期的 DatetimeIndex 对象 expected，包含指定日期范围、名称、时区和频率
    expected = date_range("1/1/2002", "1/7/2002", name=name, tz=tz, freq="b")
    expected = expected.repeat([1, 1, 1, 2, 2])  # 重复序列，以生成与结果相同的索引
    
    # 将预期对象重新设置为指定单元
    expected = expected.as_unit(unit)
    
    # 使用 pandas._testing.assert_index_equal 函数比较 result 和 expected 对象
    tm.assert_index_equal(result, expected)
    
    # 断言 result 对象的时区与 expected 对象相同
    assert result.tz == expected.tz
    
    # 断言 result 对象的频率为 None
    assert result.freq is None
    
    # 断言 expected 对象的频率为 None
    assert expected.freq is None
```