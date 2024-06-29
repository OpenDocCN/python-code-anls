# `D:\src\scipysrc\pandas\pandas\tests\arrays\interval\test_formats.py`

```
# 从 pandas.core.arrays 模块中导入 IntervalArray 类
from pandas.core.arrays import IntervalArray

# 定义测试函数 test_repr
def test_repr():
    # GH#25022 - GitHub 上的问题编号，说明此处测试的背景
    # 创建一个 IntervalArray 对象，从元组列表 [(0, 1), (1, 2)] 中生成
    arr = IntervalArray.from_tuples([(0, 1), (1, 2)])
    # 调用 IntervalArray 对象的 repr 方法，将其转换为字符串表示
    result = repr(arr)
    # 预期结果字符串
    expected = (
        "<IntervalArray>\n"
        "[(0, 1], (1, 2]]\n"
        "Length: 2, dtype: interval[int64, right]"
    )
    # 断言实际结果与预期结果相等，用于测试结果的正确性
    assert result == expected
```