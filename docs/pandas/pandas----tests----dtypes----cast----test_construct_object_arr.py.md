# `D:\src\scipysrc\pandas\pandas\tests\dtypes\cast\test_construct_object_arr.py`

```
# 导入 pytest 模块，用于测试框架
import pytest

# 从 pandas.core.dtypes.cast 模块中导入 construct_1d_object_array_from_listlike 函数
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike

# 使用 pytest 的 parametrize 装饰器，为 test_cast_1d_array 函数参数化测试数据
@pytest.mark.parametrize("datum1", [1, 2.0, "3", (4, 5), [6, 7], None])
@pytest.mark.parametrize("datum2", [8, 9.0, "10", (11, 12), [13, 14], None])
def test_cast_1d_array(datum1, datum2):
    # 创建包含两个测试数据的列表
    data = [datum1, datum2]
    # 调用 construct_1d_object_array_from_listlike 函数处理数据
    result = construct_1d_object_array_from_listlike(data)

    # 断言结果的数据类型为 "object"
    assert result.dtype == "object"
    # 断言处理后的结果列表与原始数据列表相同
    assert list(result) == data


# 使用 pytest 的 parametrize 装饰器，为 test_cast_1d_array_invalid_scalar 函数参数化测试数据
@pytest.mark.parametrize("val", [1, 2.0, None])
def test_cast_1d_array_invalid_scalar(val):
    # 使用 pytest.raises 检查是否抛出 TypeError 异常，并匹配指定的错误消息
    with pytest.raises(TypeError, match="has no len()"):
        construct_1d_object_array_from_listlike(val)
```