# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_bunch.py`

```
import warnings  # 导入警告模块

import numpy as np  # 导入NumPy库
import pytest  # 导入pytest库

from sklearn.utils import Bunch  # 从sklearn工具模块中导入Bunch类


def test_bunch_attribute_deprecation():
    """检查Bunch对象在使用`__getattr__`时是否会引发弃用警告消息。"""
    bunch = Bunch()  # 创建一个Bunch对象
    values = np.asarray([1, 2, 3])  # 创建一个NumPy数组作为示例值
    msg = (
        "Key: 'values', is deprecated in 1.3 and will be "
        "removed in 1.5. Please use 'grid_values' instead"
    )
    bunch._set_deprecated(
        values, new_key="grid_values", deprecated_key="values", warning_message=msg
    )  # 设置一个属性的弃用信息

    with warnings.catch_warnings():
        # 设置警告过滤器，将所有警告转换为异常，以便捕获检查
        warnings.simplefilter("error")
        v = bunch["grid_values"]  # 通过索引获取属性值

    assert v is values  # 断言获取的值与设置的值相同

    with pytest.warns(FutureWarning, match=msg):
        # 使用pytest的warns函数检查是否会发出FutureWarning警告，并匹配预期的消息
        v = bunch["values"]  # 尝试获取被弃用的属性值

    assert v is values  # 断言获取的值与设置的值相同
```