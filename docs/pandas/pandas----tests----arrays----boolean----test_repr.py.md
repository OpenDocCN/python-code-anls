# `D:\src\scipysrc\pandas\pandas\tests\arrays\boolean\test_repr.py`

```
# 导入 pandas 库，用于数据处理和分析
import pandas as pd

# 定义一个函数 test_repr，用于测试对象的字符串表示形式
def test_repr():
    # 创建一个包含布尔类型数据的 DataFrame 对象 df
    df = pd.DataFrame({"A": pd.array([True, False, None], dtype="boolean")})
    
    # 预期的字符串表示形式，包含 DataFrame 的结构和数据
    expected = "       A\n0   True\n1  False\n2   <NA>"
    # 断言 DataFrame 的字符串表示形式与预期结果相同
    assert repr(df) == expected

    # 预期的 Series 对象 A 的字符串表示形式
    expected = "0     True\n1    False\n2     <NA>\nName: A, dtype: boolean"
    # 断言 Series 对象 A 的字符串表示形式与预期结果相同
    assert repr(df.A) == expected

    # 预期的 BooleanArray 对象 A.array 的字符串表示形式
    expected = "<BooleanArray>\n[True, False, <NA>]\nLength: 3, dtype: boolean"
    # 断言 BooleanArray 对象 A.array 的字符串表示形式与预期结果相同
    assert repr(df.A.array) == expected
```