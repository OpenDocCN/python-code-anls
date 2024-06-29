# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_assign.py`

```
import pytest  # 导入 pytest 测试框架

from pandas import DataFrame  # 从 pandas 库中导入 DataFrame 类
import pandas._testing as tm  # 导入 pandas 内部测试模块


class TestAssign:
    def test_assign(self):
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        original = df.copy()  # 复制原始 DataFrame
        result = df.assign(C=df.B / df.A)  # 创建新的 DataFrame，添加列 C = B / A
        expected = df.copy()  # 复制预期的 DataFrame
        expected["C"] = [4, 2.5, 2]  # 设置预期的列 C
        tm.assert_frame_equal(result, expected)  # 使用测试模块验证结果与预期是否相同

        # lambda 语法
        result = df.assign(C=lambda x: x.B / x.A)  # 使用 lambda 函数创建新的 DataFrame，计算 C = B / A
        tm.assert_frame_equal(result, expected)  # 使用测试模块验证结果与预期是否相同

        # 原始 DataFrame 未被修改
        tm.assert_frame_equal(df, original)  # 使用测试模块验证结果与原始是否相同

        # 非 Series 类型的数组
        result = df.assign(C=[4, 2.5, 2])  # 创建新的 DataFrame，添加列 C = [4, 2.5, 2]
        tm.assert_frame_equal(result, expected)  # 使用测试模块验证结果与预期是否相同
        # 原始 DataFrame 未被修改
        tm.assert_frame_equal(df, original)  # 使用测试模块验证结果与原始是否相同

        result = df.assign(B=df.B / df.A)  # 创建新的 DataFrame，将列 B 替换为 B / A
        expected = expected.drop("B", axis=1).rename(columns={"C": "B"})  # 删除 B 列，重命名 C 列为 B
        tm.assert_frame_equal(result, expected)  # 使用测试模块验证结果与预期是否相同

        # 覆盖
        result = df.assign(A=df.A + df.B)  # 创建新的 DataFrame，将列 A 替换为 A + B
        expected = df.copy()  # 复制预期的 DataFrame
        expected["A"] = [5, 7, 9]  # 设置预期的列 A
        tm.assert_frame_equal(result, expected)  # 使用测试模块验证结果与预期是否相同

        # lambda
        result = df.assign(A=lambda x: x.A + x.B)  # 使用 lambda 函数创建新的 DataFrame，计算 A = A + B
        tm.assert_frame_equal(result, expected)  # 使用测试模块验证结果与预期是否相同

    def test_assign_multiple(self):
        df = DataFrame([[1, 4], [2, 5], [3, 6]], columns=["A", "B"])
        result = df.assign(C=[7, 8, 9], D=df.A, E=lambda x: x.B)  # 创建新的 DataFrame，添加多列 C, D, E
        expected = DataFrame(
            [[1, 4, 7, 1, 4], [2, 5, 8, 2, 5], [3, 6, 9, 3, 6]], columns=list("ABCDE")
        )
        tm.assert_frame_equal(result, expected)  # 使用测试模块验证结果与预期是否相同

    def test_assign_order(self):
        # GH 9818
        df = DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        result = df.assign(D=df.A + df.B, C=df.A - df.B)  # 创建新的 DataFrame，添加列 D = A + B, C = A - B

        expected = DataFrame([[1, 2, 3, -1], [3, 4, 7, -1]], columns=list("ABDC"))
        tm.assert_frame_equal(result, expected)  # 使用测试模块验证结果与预期是否相同

        result = df.assign(C=df.A - df.B, D=df.A + df.B)  # 创建新的 DataFrame，添加列 C = A - B, D = A + B
        expected = DataFrame([[1, 2, -1, 3], [3, 4, -1, 7]], columns=list("ABCD"))

        tm.assert_frame_equal(result, expected)  # 使用测试模块验证结果与预期是否相同

    def test_assign_bad(self):
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        # 非关键字参数
        msg = r"assign\(\) takes 1 positional argument but 2 were given"
        with pytest.raises(TypeError, match=msg):
            df.assign(lambda x: x.A)  # 测试是否引发 TypeError 异常

        msg = "'DataFrame' object has no attribute 'C'"
        with pytest.raises(AttributeError, match=msg):
            df.assign(C=df.A, D=df.A + df.C)  # 测试是否引发 AttributeError 异常

    def test_assign_dependent(self):
        df = DataFrame({"A": [1, 2], "B": [3, 4]})

        result = df.assign(C=df.A, D=lambda x: x["A"] + x["C"])  # 创建新的 DataFrame，添加列 C = A, D = A + C
        expected = DataFrame([[1, 3, 1, 2], [2, 4, 2, 4]], columns=list("ABCD"))
        tm.assert_frame_equal(result, expected)  # 使用测试模块验证结果与预期是否相同

        result = df.assign(C=lambda df: df.A, D=lambda df: df["A"] + df["C"])  # 使用 lambda 函数创建新的 DataFrame
        tm.assert_frame_equal(result, expected)  # 使用测试模块验证结果与预期是否相同
```