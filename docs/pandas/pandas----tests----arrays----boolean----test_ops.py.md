# `D:\src\scipysrc\pandas\pandas\tests\arrays\boolean\test_ops.py`

```
import pandas as pd  # 导入 pandas 库，通常用 pd 作为别名
import pandas._testing as tm  # 导入 pandas 内部测试模块，用于测试辅助函数

class TestUnaryOps:
    def test_invert(self):
        a = pd.array([True, False, None], dtype="boolean")  # 创建一个布尔类型的 pandas 数组 a
        expected = pd.array([False, True, None], dtype="boolean")  # 创建预期的布尔类型的 pandas 数组 expected
        tm.assert_extension_array_equal(~a, expected)  # 断言对 a 应用按位取反操作后结果与 expected 相等

        expected = pd.Series(expected, index=["a", "b", "c"], name="name")  # 创建预期的 pandas Series 对象 expected
        result = ~pd.Series(a, index=["a", "b", "c"], name="name")  # 对 pandas Series 对象 a 应用按位取反操作，得到 result
        tm.assert_series_equal(result, expected)  # 断言 result 与 expected 的 Series 对象相等

        df = pd.DataFrame({"A": a, "B": [True, False, False]}, index=["a", "b", "c"])  # 创建包含 pandas 数组的 DataFrame df
        result = ~df  # 对 DataFrame df 中的所有元素应用按位取反操作，得到 result
        expected = pd.DataFrame(
            {"A": expected, "B": [False, True, True]}, index=["a", "b", "c"]
        )  # 创建预期的 DataFrame 对象 expected
        tm.assert_frame_equal(result, expected)  # 断言 result 与 expected 的 DataFrame 对象相等

    def test_abs(self):
        # matching numpy behavior, abs is the identity function
        arr = pd.array([True, False, None], dtype="boolean")  # 创建一个布尔类型的 pandas 数组 arr
        result = abs(arr)  # 对 arr 应用绝对值函数，实际上是返回原数组 arr

        tm.assert_extension_array_equal(result, arr)  # 断言 result 与 arr 的 ExtensionArray 相等
```