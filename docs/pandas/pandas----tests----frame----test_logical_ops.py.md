# `D:\src\scipysrc\pandas\pandas\tests\frame\test_logical_ops.py`

```
# 导入必要的库
import operator  # 导入operator模块，用于操作符函数
import re  # 导入re模块，用于正则表达式操作

import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于单元测试

from pandas import (  # 从pandas库中导入以下子模块
    CategoricalIndex,  # 用于分类索引
    DataFrame,  # 用于操作数据框
    Interval,  # 用于表示间隔
    Series,  # 用于操作序列数据
    isnull,  # 用于检查缺失值
)
import pandas._testing as tm  # 导入pandas._testing模块，用于测试支持函数


class TestDataFrameLogicalOperators:
    # 定义一个测试类，用于测试数据框的逻辑运算符 (&, |, ^)

    @pytest.mark.parametrize(  # 使用pytest的参数化装饰器，定义参数化测试
        "left, right, op, expected",  # 参数有left, right, op, expected
        [  # 参数化测试的具体参数列表
            (
                [True, False, np.nan],  # 左操作数为布尔值和NaN混合的列表
                [True, False, True],  # 右操作数为布尔值列表
                operator.and_,  # 使用and运算符函数
                [True, False, False],  # 预期结果为布尔值列表
            ),
            (
                [True, False, True],  # 左操作数为布尔值列表
                [True, False, np.nan],  # 右操作数为布尔值和NaN混合的列表
                operator.and_,  # 使用and运算符函数
                [True, False, False],  # 预期结果为布尔值列表
            ),
            (
                [True, False, np.nan],  # 左操作数为布尔值和NaN混合的列表
                [True, False, True],  # 右操作数为布尔值列表
                operator.or_,  # 使用or运算符函数
                [True, False, False],  # 预期结果为布尔值列表
            ),
            (
                [True, False, True],  # 左操作数为布尔值列表
                [True, False, np.nan],  # 右操作数为布尔值和NaN混合的列表
                operator.or_,  # 使用or运算符函数
                [True, False, True],  # 预期结果为布尔值列表
            ),
        ],
    )
    def test_logical_operators_nans(self, left, right, op, expected, frame_or_series):
        # 测试逻辑运算符处理NaN值的情况
        # GH#13896

        result = op(frame_or_series(left), frame_or_series(right))  # 执行逻辑运算
        expected = frame_or_series(expected)  # 构造预期结果

        tm.assert_equal(result, expected)  # 使用测试框架验证结果是否符合预期

    def test_logical_ops_empty_frame(self):
        # 测试空数据框的逻辑运算
        # GH#5808

        # 创建一个空数据框，仅含索引为1的行
        df = DataFrame(index=[1])

        result = df & df  # 执行逻辑与运算
        tm.assert_frame_equal(result, df)  # 使用测试框架验证结果是否符合预期

        result = df | df  # 执行逻辑或运算
        tm.assert_frame_equal(result, df)  # 使用测试框架验证结果是否符合预期

        df2 = DataFrame(index=[1, 2])  # 创建含有索引1和2的数据框
        result = df & df2  # 执行逻辑与运算
        tm.assert_frame_equal(result, df2)  # 使用测试框架验证结果是否符合预期

        dfa = DataFrame(index=[1], columns=["A"])  # 创建含有列"A"的数据框

        result = dfa & dfa  # 执行逻辑与运算
        expected = DataFrame(False, index=[1], columns=["A"])  # 预期结果为所有值为False的数据框
        tm.assert_frame_equal(result, expected)  # 使用测试框架验证结果是否符合预期

    def test_logical_ops_bool_frame(self):
        # 测试布尔类型数据框的逻辑运算
        # GH#5808

        df1a_bool = DataFrame(True, index=[1], columns=["A"])  # 创建布尔类型数据框

        result = df1a_bool & df1a_bool  # 执行逻辑与运算
        tm.assert_frame_equal(result, df1a_bool)  # 使用测试框架验证结果是否符合预期

        result = df1a_bool | df1a_bool  # 执行逻辑或运算
        tm.assert_frame_equal(result, df1a_bool)  # 使用测试框架验证结果是否符合预期

    def test_logical_ops_int_frame(self):
        # 测试整数类型数据框的逻辑运算
        # GH#5808

        df1a_int = DataFrame(1, index=[1], columns=["A"])  # 创建整数类型数据框
        df1a_bool = DataFrame(True, index=[1], columns=["A"])  # 创建布尔类型数据框

        result = df1a_int | df1a_bool  # 执行逻辑或运算
        tm.assert_frame_equal(result, df1a_bool)  # 使用测试框架验证结果是否符合预期

        # 检查与Series行为的一致性
        res_ser = df1a_int["A"] | df1a_bool["A"]  # 执行逻辑或运算
        tm.assert_series_equal(res_ser, df1a_bool["A"])  # 使用测试框架验证结果是否符合预期
    # 定义一个测试方法，用于测试逻辑运算的非法情况，使用推断字符串
    def test_logical_ops_invalid(self, using_infer_string):
        # GH#5808

        # 创建一个数据帧 df1，填充值为 1.0，索引为 [1]，列为 ["A"]
        df1 = DataFrame(1.0, index=[1], columns=["A"])
        # 创建一个数据帧 df2，填充值为 True，索引为 [1]，列为 ["A"]
        df2 = DataFrame(True, index=[1], columns=["A"])
        # 生成一个用于匹配的正则表达式消息，用于捕获类型错误异常信息
        msg = re.escape("unsupported operand type(s) for |: 'float' and 'bool'")
        # 断言会触发类型错误异常，并匹配预期的消息
        with pytest.raises(TypeError, match=msg):
            # 对 df1 和 df2 进行按位或操作
            df1 | df2

        # 创建一个数据帧 df1，填充值为 "foo"，索引为 [1]，列为 ["A"]
        df1 = DataFrame("foo", index=[1], columns=["A"])
        # 创建一个数据帧 df2，填充值为 True，索引为 [1]，列为 ["A"]
        df2 = DataFrame(True, index=[1], columns=["A"])
        # 生成一个用于匹配的正则表达式消息，用于捕获类型错误异常信息
        msg = re.escape("unsupported operand type(s) for |: 'str' and 'bool'")
        # 如果使用推断字符串
        if using_infer_string:
            # 导入 pyarrow 库作为 pa
            import pyarrow as pa
            # 断言会触发 ArrowNotImplementedError 异常，并匹配预期的消息
            with pytest.raises(pa.lib.ArrowNotImplementedError, match="|has no kernel"):
                # 对 df1 和 df2 进行按位或操作
                df1 | df2
        else:
            # 断言会触发类型错误异常，并匹配预期的消息
            with pytest.raises(TypeError, match=msg):
                # 对 df1 和 df2 进行按位或操作
                df1 | df2

    # 定义一个测试方法，用于测试逻辑运算符的功能
    def test_logical_operators(self):
        # 定义一个内部函数，用于检查二元操作符
        def _check_bin_op(op):
            # 执行操作 op，并得到结果
            result = op(df1, df2)
            # 创建一个期望的数据帧，使用 op 在 df1 和 df2 的值上进行操作，保持索引和列不变
            expected = DataFrame(
                op(df1.values, df2.values), index=df1.index, columns=df1.columns
            )
            # 断言结果数据类型为布尔型
            assert result.values.dtype == np.bool_
            # 断言结果与期望数据帧相等
            tm.assert_frame_equal(result, expected)

        # 定义一个内部函数，用于检查一元操作符
        def _check_unary_op(op):
            # 执行操作 op，并得到结果
            result = op(df1)
            # 创建一个期望的数据帧，使用 op 在 df1 的值上进行操作，保持索引和列不变
            expected = DataFrame(op(df1.values), index=df1.index, columns=df1.columns)
            # 断言结果数据类型为布尔型
            assert result.values.dtype == np.bool_
            # 断言结果与期望数据帧相等
            tm.assert_frame_equal(result, expected)

        # 定义一个字典，表示数据帧 df1 的内容
        df1 = {
            "a": {"a": True, "b": False, "c": False, "d": True, "e": True},
            "b": {"a": False, "b": True, "c": False, "d": False, "e": False},
            "c": {"a": False, "b": False, "c": True, "d": False, "e": False},
            "d": {"a": True, "b": False, "c": False, "d": True, "e": True},
            "e": {"a": True, "b": False, "c": False, "d": True, "e": True},
        }

        # 定义一个字典，表示数据帧 df2 的内容
        df2 = {
            "a": {"a": True, "b": False, "c": True, "d": False, "e": False},
            "b": {"a": False, "b": True, "c": False, "d": False, "e": False},
            "c": {"a": True, "b": False, "c": True, "d": False, "e": False},
            "d": {"a": False, "b": False, "c": False, "d": True, "e": False},
            "e": {"a": False, "b": False, "c": False, "d": False, "e": True},
        }

        # 创建数据帧 df1 和 df2
        df1 = DataFrame(df1)
        df2 = DataFrame(df2)

        # 测试逻辑与操作
        _check_bin_op(operator.and_)
        # 测试逻辑或操作
        _check_bin_op(operator.or_)
        # 测试逻辑异或操作
        _check_bin_op(operator.xor)

        # 测试逻辑非操作
        _check_unary_op(operator.inv)  # TODO: belongs elsewhere
    def test_logical_with_nas(self):
        # 创建一个包含 NaN 和布尔值的 DataFrame
        d = DataFrame({"a": [np.nan, False], "b": [True, True]})

        # GH4947
        # 布尔比较应返回布尔值
        result = d["a"] | d["b"]
        expected = Series([False, True])
        tm.assert_series_equal(result, expected)

        # GH4604, 自动类型转换
        result = d["a"].fillna(False) | d["b"]
        expected = Series([True, True])
        tm.assert_series_equal(result, expected)
        result = d["a"].fillna(False) | d["b"]
        expected = Series([True, True])
        tm.assert_series_equal(result, expected)

    def test_logical_ops_categorical_columns(self):
        # GH#38367
        # 创建一个包含区间的 DataFrame
        intervals = [Interval(1, 2), Interval(3, 4)]
        data = DataFrame(
            [[1, np.nan], [2, np.nan]],
            columns=CategoricalIndex(
                intervals, categories=intervals + [Interval(5, 6)]
            ),
        )
        # 创建一个与 data 相同结构的布尔值 DataFrame
        mask = DataFrame(
            [[False, False], [False, False]], columns=data.columns, dtype=bool
        )
        # 对 mask 和 data 执行逻辑或运算
        result = mask | isnull(data)
        expected = DataFrame(
            [[False, True], [False, True]],
            columns=CategoricalIndex(
                intervals, categories=intervals + [Interval(5, 6)]
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_int_dtype_different_index_not_bool(self):
        # GH 52500
        # 创建两个带有不同索引的整数值 DataFrame
        df1 = DataFrame([1, 2, 3], index=[10, 11, 23], columns=["a"])
        df2 = DataFrame([10, 20, 30], index=[11, 10, 23], columns=["a"])
        # 对 df1 和 df2 执行按位异或运算
        result = np.bitwise_xor(df1, df2)
        expected = DataFrame([21, 8, 29], index=[10, 11, 23], columns=["a"])
        tm.assert_frame_equal(result, expected)

        # 使用 ^ 运算符执行按位异或运算
        result = df1 ^ df2
        tm.assert_frame_equal(result, expected)

    def test_different_dtypes_different_index_raises(self):
        # GH 52538
        # 创建两个带有不同索引的整数值 DataFrame
        df1 = DataFrame([1, 2], index=["a", "b"])
        df2 = DataFrame([3, 4], index=["b", "c"])
        # 确保在不同数据类型和索引的 DataFrame 上执行逻辑与运算会引发 TypeError
        with pytest.raises(TypeError, match="unsupported operand type"):
            df1 & df2
```