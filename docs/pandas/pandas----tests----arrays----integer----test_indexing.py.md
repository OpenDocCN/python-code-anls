# `D:\src\scipysrc\pandas\pandas\tests\arrays\integer\test_indexing.py`

```
# 导入 pandas 库并命名为 pd
import pandas as pd
# 导入 pandas 测试工具模块，并命名为 tm
import pandas._testing as tm


# 定义测试函数 test_array_setitem_nullable_boolean_mask
def test_array_setitem_nullable_boolean_mask():
    # 标识：GH 31446
    # 创建一个包含整数的 Series，指定 dtype 为 "Int64"
    ser = pd.Series([1, 2], dtype="Int64")
    # 使用 where 方法根据条件筛选 Series 中的值，并返回结果
    result = ser.where(ser > 1)
    # 创建预期结果的 Series，包含 NA 值，指定 dtype 为 "Int64"
    expected = pd.Series([pd.NA, 2], dtype="Int64")
    # 使用测试工具 tm 的 assert_series_equal 方法比较结果和预期值是否相等
    tm.assert_series_equal(result, expected)


# 定义测试函数 test_array_setitem
def test_array_setitem():
    # 标识：GH 31446
    # 创建一个包含整数的 Series，指定 dtype 为 "Int64"，然后获取其 array 属性
    arr = pd.Series([1, 2], dtype="Int64").array
    # 使用布尔掩码选择 arr 中大于 1 的元素，并将它们设置为 1
    arr[arr > 1] = 1

    # 创建预期结果的 ExtensionArray，包含整数 1，指定 dtype 为 "Int64"
    expected = pd.array([1, 1], dtype="Int64")
    # 使用测试工具 tm 的 assert_extension_array_equal 方法比较 arr 和预期值是否相等
    tm.assert_extension_array_equal(arr, expected)
```