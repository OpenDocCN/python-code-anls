# `D:\src\scipysrc\pandas\pandas\tests\util\test_validate_inclusive.py`

```
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入Pytest库，用于编写和运行测试

from pandas.util._validators import validate_inclusive  # 导入Pandas库中的验证函数

import pandas as pd  # 导入Pandas库，用于数据处理


@pytest.mark.parametrize(  # 使用Pytest的参数化装饰器，用于多次运行同一个测试函数
    "invalid_inclusive",  # 参数名为invalid_inclusive
    (  # 参数取值为以下几种
        "ccc",  # 字符串类型的无效参数
        2,  # 整数类型的无效参数
        object(),  # Python对象类型的无效参数
        None,  # None类型的无效参数
        np.nan,  # NumPy的NaN值类型的无效参数
        pd.NA,  # Pandas的NA值类型的无效参数
        pd.DataFrame(),  # Pandas的DataFrame类型的无效参数
    ),
)
def test_invalid_inclusive(invalid_inclusive):
    with pytest.raises(  # 使用Pytest断言上下文管理器，期望捕获异常
        ValueError,  # 期望捕获的异常类型为ValueError
        match="Inclusive has to be either 'both', 'neither', 'left' or 'right'",  # 异常消息匹配条件
    ):
        validate_inclusive(invalid_inclusive)  # 调用验证函数验证无效参数


@pytest.mark.parametrize(  # 使用Pytest的参数化装饰器，用于多次运行同一个测试函数
    "valid_inclusive, expected_tuple",  # 参数名为valid_inclusive和expected_tuple
    (  # 参数取值为以下几种
        ("left", (True, False)),  # 参数为'left'时，期望返回(True, False)
        ("right", (False, True)),  # 参数为'right'时，期望返回(False, True)
        ("both", (True, True)),  # 参数为'both'时，期望返回(True, True)
        ("neither", (False, False)),  # 参数为'neither'时，期望返回(False, False)
    ),
)
def test_valid_inclusive(valid_inclusive, expected_tuple):
    resultant_tuple = validate_inclusive(valid_inclusive)  # 调用验证函数获取结果元组
    assert expected_tuple == resultant_tuple  # 断言验证结果与期望结果相等
```