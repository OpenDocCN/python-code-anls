# `D:\src\scipysrc\pandas\pandas\tests\arrays\floating\test_concat.py`

```
# 导入 pytest 模块，用于测试
import pytest

# 导入 pandas 库，并导入 pandas 内部的测试模块 _testing
import pandas as pd
import pandas._testing as tm

# 使用 pytest 的 parametrize 装饰器，定义多组参数化测试参数
@pytest.mark.parametrize(
    "to_concat_dtypes, result_dtype",
    [
        (["Float64", "Float64"], "Float64"),  # 测试用例：两个 Float64 的 Series 连接后结果的数据类型
        (["Float32", "Float64"], "Float64"),  # 测试用例：一个 Float32 和一个 Float64 的 Series 连接后结果的数据类型
        (["Float32", "Float32"], "Float32"),  # 测试用例：两个 Float32 的 Series 连接后结果的数据类型
    ],
)
def test_concat_series(to_concat_dtypes, result_dtype):
    # 构造待连接的 Series 列表，根据不同的数据类型进行创建
    result = pd.concat([pd.Series([1, 2, pd.NA], dtype=t) for t in to_concat_dtypes])
    
    # 构造期望的结果 Series，使用 object 类型的 Series 来重复两次并转换为指定的结果数据类型
    expected = pd.concat([pd.Series([1, 2, pd.NA], dtype=object)] * 2).astype(
        result_dtype
    )
    
    # 使用 pandas 提供的测试工具函数 assert_series_equal 检查 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)
```