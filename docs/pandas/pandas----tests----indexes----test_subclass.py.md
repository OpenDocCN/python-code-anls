# `D:\src\scipysrc\pandas\pandas\tests\indexes\test_subclass.py`

```
"""
Tests involving custom Index subclasses
"""

# 导入必要的库
import numpy as np

# 从 pandas 库中导入 DataFrame 和 Index 类
from pandas import (
    DataFrame,
    Index,
)
# 导入 pandas 内部的测试工具模块
import pandas._testing as tm

# 定义一个自定义的 Index 子类
class CustomIndex(Index):
    def __new__(cls, data, name=None):
        # 断言这个索引类不能包含字符串类型的数据
        if any(isinstance(val, str) for val in data):
            raise TypeError("CustomIndex cannot hold strings")

        # 如果没有指定名称但数据对象有名称，则使用数据对象的名称
        if name is None and hasattr(data, "name"):
            name = data.name
        # 将数据转换为对象类型的 NumPy 数组
        data = np.array(data, dtype="O")

        # 调用 Index 类的 _simple_new 方法创建新的 Index 对象
        return cls._simple_new(data, name)


# 定义一个测试函数，测试插入操作对基础索引的回退行为
def test_insert_fallback_to_base_index():
    # 创建一个 CustomIndex 实例
    idx = CustomIndex([1, 2, 3])
    # 在索引的开头插入一个字符串，期望会回退到基础的 Index 类
    result = idx.insert(0, "string")
    expected = Index(["string", 1, 2, 3], dtype=object)
    # 使用测试工具模块中的函数检查插入操作的结果是否符合预期
    tm.assert_index_equal(result, expected)

    # 创建一个 DataFrame 对象
    df = DataFrame(
        np.random.default_rng(2).standard_normal((2, 3)),
        columns=idx,
        index=Index([1, 2], name="string"),
    )
    # 对 DataFrame 执行重置索引的操作
    result = df.reset_index()
    # 使用测试工具模块中的函数检查重置索引后的列是否符合预期
    tm.assert_index_equal(result.columns, expected)
```