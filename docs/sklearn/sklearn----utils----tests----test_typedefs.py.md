# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_typedefs.py`

```
# 导入必要的库：numpy 和 pytest
import numpy as np
import pytest

# 从 sklearn.utils._typedefs 模块中导入 testing_make_array_from_typed_val 函数
from sklearn.utils._typedefs import testing_make_array_from_typed_val

# 使用 pytest 的 parametrize 装饰器定义多个参数化测试用例
@pytest.mark.parametrize(
    "type_t, value, expected_dtype",
    [
        ("float64_t", 1.0, np.float64),   # 测试 float64_t 类型的值 1.0 是否为 np.float64 类型
        ("float32_t", 1.0, np.float32),   # 测试 float32_t 类型的值 1.0 是否为 np.float32 类型
        ("intp_t", 1, np.intp),           # 测试 intp_t 类型的值 1 是否为 np.intp 类型
        ("int8_t", 1, np.int8),           # 测试 int8_t 类型的值 1 是否为 np.int8 类型
        ("int32_t", 1, np.int32),         # 测试 int32_t 类型的值 1 是否为 np.int32 类型
        ("int64_t", 1, np.int64),         # 测试 int64_t 类型的值 1 是否为 np.int64 类型
        ("uint8_t", 1, np.uint8),         # 测试 uint8_t 类型的值 1 是否为 np.uint8 类型
        ("uint32_t", 1, np.uint32),       # 测试 uint32_t 类型的值 1 是否为 np.uint32 类型
        ("uint64_t", 1, np.uint64),       # 测试 uint64_t 类型的值 1 是否为 np.uint64 类型
    ],
)
# 定义单元测试函数 test_types，用于验证 _typedefs 中定义的类型与对应的 numpy 数据类型是否一致
def test_types(type_t, value, expected_dtype):
    """Check that the types defined in _typedefs correspond to the expected
    numpy dtypes.
    """
    # 断言 testing_make_array_from_typed_val 函数根据 type_t 转换 value 后的 dtype 是否与 expected_dtype 相等
    assert testing_make_array_from_typed_val[type_t](value).dtype == expected_dtype
```