# `D:\src\scipysrc\pandas\pandas\tests\apply\test_series_apply_relabeling.py`

```
import pandas as pd  # 导入 pandas 库，命名为 pd
import pandas._testing as tm  # 导入 pandas 测试模块，命名为 tm


def test_relabel_no_duplicated_method():
    # 测试 agg 方法中没有重复使用相同方法名的情况
    df = pd.DataFrame({"A": [1, 2, 1, 2], "B": [1, 2, 3, 4]})  # 创建一个包含两列的 DataFrame

    result = df["A"].agg(foo="sum")  # 对列 'A' 应用聚合函数 sum，并将结果命名为 'foo'
    expected = df["A"].agg({"foo": "sum"})  # 期望的结果是将聚合结果作为字典返回，键为 'foo'
    tm.assert_series_equal(result, expected)  # 断言结果与期望相等

    result = df["B"].agg(foo="min", bar="max")  # 对列 'B' 应用多个聚合函数，并分别命名为 'foo' 和 'bar'
    expected = df["B"].agg({"foo": "min", "bar": "max"})  # 期望的结果是将多个聚合函数结果作为字典返回
    tm.assert_series_equal(result, expected)  # 断言结果与期望相等

    result = df["B"].agg(foo=sum, bar=min, cat="max")  # 对列 'B' 应用多个聚合函数，包括自定义函数 sum 和 min，以及命名为 'cat' 的 max 函数
    expected = df["B"].agg({"foo": sum, "bar": min, "cat": "max"})  # 期望的结果是将多个聚合函数及其名称作为字典返回
    tm.assert_series_equal(result, expected)  # 断言结果与期望相等


def test_relabel_duplicated_method():
    # 测试在嵌套重命名的情况下，可以使用重复的聚合方法，只要它们分别被分配不同的新名称
    df = pd.DataFrame({"A": [1, 2, 1, 2], "B": [1, 2, 3, 4]})  # 创建一个包含两列的 DataFrame

    result = df["A"].agg(foo="sum", bar="sum")  # 对列 'A' 应用相同的聚合函数 sum，但分别命名为 'foo' 和 'bar'
    expected = pd.Series([6, 6], index=["foo", "bar"], name="A")  # 期望的结果是一个具有指定索引和名称的 Series
    tm.assert_series_equal(result, expected)  # 断言结果与期望相等

    result = df["B"].agg(foo=min, bar="min")  # 对列 'B' 应用多个聚合函数，包括自定义函数 min 和命名为 'bar' 的 min 函数
    expected = pd.Series([1, 1], index=["foo", "bar"], name="B")  # 期望的结果是一个具有指定索引和名称的 Series
    tm.assert_series_equal(result, expected)  # 断言结果与期望相等
```