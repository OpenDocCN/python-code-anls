# `D:\src\scipysrc\pandas\pandas\tests\series\test_logical_ops.py`

```
    from datetime import datetime  # 导入 datetime 模块中的 datetime 类
    import operator  # 导入 operator 模块，用于支持运算符的函数形式调用

    import numpy as np  # 导入 NumPy 库并使用别名 np
    import pytest  # 导入 pytest 测试框架

    from pandas import (  # 从 pandas 库中导入多个子模块和函数
        DataFrame,  # 导入 DataFrame 类
        Index,  # 导入 Index 类
        Series,  # 导入 Series 类
        bdate_range,  # 导入工作日范围生成函数 bdate_range
    )
    import pandas._testing as tm  # 导入 pandas 测试模块并使用别名 tm
    from pandas.core import ops  # 从 pandas 核心模块中导入 ops 模块

    class TestSeriesLogicalOps:
        @pytest.mark.parametrize("bool_op", [operator.and_, operator.or_, operator.xor])
        def test_bool_operators_with_nas(self, bool_op):
            # 对象数组上的逻辑运算符 &、|、^ 应该能处理 NaN，并传播 NaN
            ser = Series(bdate_range("1/1/2000", periods=10), dtype=object)
            ser[::2] = np.nan  # 每隔两个元素设为 NaN

            mask = ser.isna()  # 创建一个标记 NaN 的掩码
            filled = ser.fillna(ser[0])  # 用第一个非 NaN 元素填充 NaN

            result = bool_op(ser < ser[9], ser > ser[3])  # 应用给定的逻辑操作符

            expected = bool_op(filled < filled[9], filled > filled[3])  # 期望的结果
            expected[mask] = False  # 将掩码标记的位置设为 False
            tm.assert_series_equal(result, expected)  # 断言结果与期望相等

        def test_logical_operators_bool_dtype_with_empty(self):
            # GH#9016: 支持整数类型的位运算操作
            index = list("bca")  # 创建索引列表

            s_tft = Series([True, False, True], index=index)  # 创建布尔类型的 Series 对象
            s_fff = Series([False, False, False], index=index)  # 创建布尔类型的 Series 对象
            s_empty = Series([], dtype=object)  # 创建空的对象类型的 Series 对象

            res = s_tft & s_empty  # 对 s_tft 和 s_empty 进行逻辑与运算
            expected = s_fff.sort_index()  # 排序后的预期结果
            tm.assert_series_equal(res, expected)  # 断言结果与预期相等

            res = s_tft | s_empty  # 对 s_tft 和 s_empty 进行逻辑或运算
            expected = s_tft.sort_index()  # 排序后的预期结果
            tm.assert_series_equal(res, expected)  # 断言结果与预期相等

        def test_logical_operators_int_dtype_with_int_dtype(self):
            # GH#9016: 支持整数类型的位运算操作

            s_0123 = Series(range(4), dtype="int64")  # 创建整数类型的 Series 对象
            s_3333 = Series([3] * 4)  # 创建值全为 3 的 Series 对象
            s_4444 = Series([4] * 4)  # 创建值全为 4 的 Series 对象

            res = s_0123 & s_3333  # 对 s_0123 和 s_3333 进行逻辑与运算
            expected = Series(range(4), dtype="int64")  # 期望的结果
            tm.assert_series_equal(res, expected)  # 断言结果与期望相等

            res = s_0123 | s_4444  # 对 s_0123 和 s_4444 进行逻辑或运算
            expected = Series(range(4, 8), dtype="int64")  # 期望的结果
            tm.assert_series_equal(res, expected)  # 断言结果与期望相等

            s_1111 = Series([1] * 4, dtype="int8")  # 创建值全为 1 的 int8 类型的 Series 对象
            res = s_0123 & s_1111  # 对 s_0123 和 s_1111 进行逻辑与运算
            expected = Series([0, 1, 0, 1], dtype="int64")  # 期望的结果
            tm.assert_series_equal(res, expected)  # 断言结果与期望相等

            res = s_0123.astype(np.int16) | s_1111.astype(np.int32)  # 对 s_0123 和 s_1111 进行逻辑或运算
            expected = Series([1, 1, 3, 3], dtype="int32")  # 期望的结果
            tm.assert_series_equal(res, expected)  # 断言结果与期望相等

        def test_logical_operators_int_dtype_with_int_scalar(self):
            # GH#9016: 支持整数类型的位运算操作
            s_0123 = Series(range(4), dtype="int64")  # 创建整数类型的 Series 对象

            res = s_0123 & 0  # 对 s_0123 和整数 0 进行逻辑与运算
            expected = Series([0] * 4)  # 期望的结果
            tm.assert_series_equal(res, expected)  # 断言结果与期望相等

            res = s_0123 & 1  # 对 s_0123 和整数 1 进行逻辑与运算
            expected = Series([0, 1, 0, 1])  # 期望的结果
            tm.assert_series_equal(res, expected)  # 断言结果与期望相等
    # 测试整数类型数据与浮点数进行逻辑运算的情况
    def test_logical_operators_int_dtype_with_float(self):
        # 创建一个包含整数的序列对象，数据类型为 int64
        s_0123 = Series(range(4), dtype="int64")

        # 错误消息模式，用于匹配预期的异常信息
        err_msg = (
            r"Logical ops \(and, or, xor\) between Pandas objects and "
            "dtype-less sequences"
        )

        # 错误消息模式，用于匹配预期的异常信息
        msg = "Cannot perform.+with a dtyped.+array and scalar of type"

        # 检查按位与操作中整数序列与 NaN 的操作是否引发 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            s_0123 & np.nan
        # 检查按位与操作中整数序列与浮点数的操作是否引发 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            s_0123 & 3.14

        # 错误消息模式，用于匹配预期的异常信息
        msg = "unsupported operand type.+for &:"

        # 检查按位与操作中整数序列与包含浮点数的列表的操作是否引发 TypeError 异常
        with pytest.raises(TypeError, match=err_msg):
            s_0123 & [0.1, 4, 3.14, 2]
        # 检查按位与操作中整数序列与包含浮点数的 NumPy 数组的操作是否引发 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            s_0123 & np.array([0.1, 4, 3.14, 2])
        # 检查按位与操作中整数序列与包含浮点数的 Pandas Series 的操作是否引发 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            s_0123 & Series([0.1, 4, -3.14, 2])

    # 测试整数类型数据与字符串进行逻辑运算的情况
    def test_logical_operators_int_dtype_with_str(self):
        # 创建一个包含整数的序列对象，数据类型为 int8
        s_1111 = Series([1] * 4, dtype="int8")

        # 错误消息模式，用于匹配预期的异常信息
        err_msg = (
            r"Logical ops \(and, or, xor\) between Pandas objects and "
            "dtype-less sequences"
        )

        # 错误消息模式，用于匹配预期的异常信息
        msg = "Cannot perform 'and_' with a dtyped.+array and scalar of type"

        # 检查按位与操作中整数序列与字符串的操作是否引发 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            s_1111 & "a"
        # 检查按位与操作中整数序列与包含字符串的列表的操作是否引发 TypeError 异常
        with pytest.raises(TypeError, match=err_msg):
            s_1111 & ["a", "b", "c", "d"]

    # 测试整数类型数据与布尔值进行逻辑运算的情况
    def test_logical_operators_int_dtype_with_bool(self):
        # 创建一个包含整数的序列对象，数据类型为 int64
        s_0123 = Series(range(4), dtype="int64")

        # 创建一个期望的结果序列，全为 False
        expected = Series([False] * 4)

        # 执行按位与操作，并检查结果是否与预期相符
        result = s_0123 & False
        tm.assert_series_equal(result, expected)

        # 错误消息模式，用于匹配预期的异常信息
        msg = (
            r"Logical ops \(and, or, xor\) between Pandas objects and "
            "dtype-less sequences"
        )

        # 检查按位与操作中整数序列与包含单个布尔值的列表的操作是否引发 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            s_0123 & [False]

        # 检查按位与操作中整数序列与包含单个布尔值的元组的操作是否引发 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            s_0123 & (False,)

        # 执行按位异或操作，并检查结果是否与预期相符
        result = s_0123 ^ False
        expected = Series([False, True, True, True])
        tm.assert_series_equal(result, expected)

    # 测试整数类型数据与对象类型数据进行逻辑运算的情况
    def test_logical_operators_int_dtype_with_object(self, using_infer_string):
        # 创建一个包含整数的序列对象，数据类型为 int64
        s_0123 = Series(range(4), dtype="int64")

        # 执行按位与操作，将整数序列与包含布尔值和 NaN 的 Series 进行逻辑运算，并检查结果是否与预期相符
        result = s_0123 & Series([False, np.nan, False, False])
        expected = Series([False] * 4)
        tm.assert_series_equal(result, expected)

        # 创建一个包含字符串的序列对象
        s_abNd = Series(["a", "b", np.nan, "d"])

        # 根据条件选择不同的异常处理方式
        if using_infer_string:
            # 如果使用推断字符串类型，则期望引发 ArrowNotImplementedError 异常
            import pyarrow as pa
            with pytest.raises(pa.lib.ArrowNotImplementedError, match="has no kernel"):
                s_0123 & s_abNd
        else:
            # 否则期望引发 TypeError 异常，因为整数与字符串的按位与操作不受支持
            with pytest.raises(TypeError, match="unsupported.* 'int' and 'str'"):
                s_0123 & s_abNd
    # 定义一个测试方法，测试布尔类型数据与整数的逻辑操作
    def test_logical_operators_bool_dtype_with_int(self):
        # 创建一个索引列表
        index = list("bca")

        # 创建一个布尔类型的 Pandas Series 对象，指定索引
        s_tft = Series([True, False, True], index=index)
        # 创建另一个布尔类型的 Pandas Series 对象，指定索引
        s_fff = Series([False, False, False], index=index)

        # 对 s_tft 和整数 0 进行按位与操作，结果存入 res
        res = s_tft & 0
        # 预期结果是 s_fff
        expected = s_fff
        # 检查 res 是否与 expected 相等
        tm.assert_series_equal(res, expected)

        # 对 s_tft 和整数 1 进行按位与操作，结果存入 res
        res = s_tft & 1
        # 预期结果是 s_tft 本身
        expected = s_tft
        # 检查 res 是否与 expected 相等
        tm.assert_series_equal(res, expected)

    # 定义一个测试方法，测试布尔类型数据与 ndarray 的逻辑操作
    def test_logical_ops_bool_dtype_with_ndarray():
        # 确保我们在 ndarray 上进行与 Series 相同的操作
        # 创建一个布尔类型的 Pandas Series 对象
        left = Series([True, True, True, False, True])
        # 创建一个包含 True、False、None 和 NaN 的列表
        right = [True, False, None, True, np.nan]

        # 错误消息字符串，用于匹配 pytest 异常
        msg = (
            r"Logical ops \(and, or, xor\) between Pandas objects and "
            "dtype-less sequences"
        )

        # 预期的结果是将 left 与 right 执行按位与操作得到的 Series 对象
        expected = Series([True, False, False, False, False])
        # 检查执行 left & right 时是否会引发 TypeError 异常，并匹配错误消息
        with pytest.raises(TypeError, match=msg):
            left & right
        # 将 left 与 ndarray right 执行按位与操作，将结果存入 result
        result = left & np.array(right)
        # 检查 result 是否与 expected 相等
        tm.assert_series_equal(result, expected)
        # 将 left 与 Index(right) 执行按位与操作，将结果存入 result
        result = left & Index(right)
        # 检查 result 是否与 expected 相等
        tm.assert_series_equal(result, expected)
        # 将 left 与 Series(right) 执行按位与操作，将结果存入 result
        result = left & Series(right)
        # 检查 result 是否与 expected 相等
        tm.assert_series_equal(result, expected)

        # 预期的结果是将 left 与 right 执行按位或操作得到的 Series 对象
        expected = Series([True, True, True, True, True])
        # 检查执行 left | right 时是否会引发 TypeError 异常，并匹配错误消息
        with pytest.raises(TypeError, match=msg):
            left | right
        # 将 left 与 ndarray right 执行按位或操作，将结果存入 result
        result = left | np.array(right)
        # 检查 result 是否与 expected 相等
        tm.assert_series_equal(result, expected)
        # 将 left 与 Index(right) 执行按位或操作，将结果存入 result
        result = left | Index(right)
        # 检查 result 是否与 expected 相等
        tm.assert_series_equal(result, expected)
        # 将 left 与 Series(right) 执行按位或操作，将结果存入 result
        result = left | Series(right)
        # 检查 result 是否与 expected 相等
        tm.assert_series_equal(result, expected)

        # 预期的结果是将 left 与 right 执行按位异或操作得到的 Series 对象
        expected = Series([False, True, True, True, True])
        # 检查执行 left ^ right 时是否会引发 TypeError 异常，并匹配错误消息
        with pytest.raises(TypeError, match=msg):
            left ^ right
        # 将 left 与 ndarray right 执行按位异或操作，将结果存入 result
        result = left ^ np.array(right)
        # 检查 result 是否与 expected 相等
        tm.assert_series_equal(result, expected)
        # 将 left 与 Index(right) 执行按位异或操作，将结果存入 result
        result = left ^ Index(right)
        # 检查 result 是否与 expected 相等
        tm.assert_series_equal(result, expected)
        # 将 left 与 Series(right) 执行按位异或操作，将结果存入 result
        result = left ^ Series(right)
        # 检查 result 是否与 expected 相等
        tm.assert_series_equal(result, expected)
    def test_logical_operators_int_dtype_with_bool_dtype_and_reindex(self):
        # GH#9016: support bitwise op for integer types
        # 定义一个索引列表
        index = list("bca")

        # 创建包含布尔值的 Series 对象 s_tft 和 s_tff，并指定索引
        s_tft = Series([True, False, True], index=index)
        s_tff = Series([True, False, False], index=index)

        # 创建一个包含整数的 Series 对象 s_0123
        s_0123 = Series(range(4), dtype="int64")

        # 进行按位与运算，并将结果与预期的 Series 对象 expected 进行比较
        # 这里 s_0123 会根据 s_tft 的索引进行重新索引，结果全为 False
        expected = Series([False] * 7, index=[0, 1, 2, 3, "a", "b", "c"])
        result = s_tft & s_0123
        tm.assert_series_equal(result, expected)

        # 进行按位与运算，并预期引发 TypeError 异常，错误信息为指定的消息
        msg = r"unsupported operand type\(s\) for &: 'float' and 'bool'"
        with pytest.raises(TypeError, match=msg):
            s_0123 & s_tft

        # 创建一个包含整数的 Series 对象 s_a0b1c0，指定其索引
        s_a0b1c0 = Series([1], list("b"))

        # 进行按位与运算，并将结果与预期的 Series 对象 expected 进行比较
        res = s_tft & s_a0b1c0
        expected = s_tff.reindex(list("abc"))
        tm.assert_series_equal(res, expected)

        # 进行按位或运算，并将结果与预期的 Series 对象 expected 进行比较
        res = s_tft | s_a0b1c0
        expected = s_tft.reindex(list("abc"))
        tm.assert_series_equal(res, expected)

    def test_scalar_na_logical_ops_corners(self):
        # 创建一个包含整数的 Series 对象 s
        s = Series([2, 3, 4, 5, 6, 7, 8, 9, 10])

        # 预期引发 TypeError 异常，错误信息为指定的消息
        msg = "Cannot perform.+with a dtyped.+array and scalar of type"
        with pytest.raises(TypeError, match=msg):
            s & datetime(2005, 1, 1)

        # 修改 s 的部分值为 NaN
        s = Series([2, 3, 4, 5, 6, 7, 8, 9, datetime(2005, 1, 1)])
        s[::2] = np.nan

        # 创建一个预期的 Series 对象 expected，全为 True，索引与 s 相同
        expected = Series(True, index=s.index)
        expected[::2] = False

        # 预期引发 TypeError 异常，错误信息为指定的消息
        msg = (
            r"Logical ops \(and, or, xor\) between Pandas objects and "
            "dtype-less sequences"
        )
        with pytest.raises(TypeError, match=msg):
            s & list(s)

    def test_scalar_na_logical_ops_corners_aligns(self):
        # 创建一个包含整数和 NaN 的 Series 对象 s
        s = Series([2, 3, 4, 5, 6, 7, 8, 9, datetime(2005, 1, 1)])
        s[::2] = np.nan

        # 根据 s 创建一个 DataFrame 对象 d
        d = DataFrame({"A": s})

        # 创建一个预期的 DataFrame 对象 expected，全为 False，索引与 d 相同，列名为 ["A", 0, 1, 2, 3, 4, 5, 6, 7, 8]
        expected = DataFrame(False, index=range(9), columns=["A"] + list(range(9)))

        # 进行按位与运算，并将结果与预期的 DataFrame 对象 expected 进行比较
        result = s & d
        tm.assert_frame_equal(result, expected)

        # 进行按位与运算，并将结果与预期的 DataFrame 对象 expected 进行比较
        result = d & s
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("op", [operator.and_, operator.or_, operator.xor])
    def test_logical_ops_with_index(self, op):
        # 创建一个包含布尔值的 Series 对象 ser 和两个不同的索引 idx1、idx2
        ser = Series([True, True, False, False])
        idx1 = Index([True, False, True, False])
        idx2 = Index([1, 0, 1, 0])

        # 创建一个预期的 Series 对象 expected，根据 op 参数的函数逐元素计算
        expected = Series([op(ser[n], idx1[n]) for n in range(len(ser))])

        # 进行逻辑运算，并将结果与预期的 Series 对象 expected 进行比较
        result = op(ser, idx1)
        tm.assert_series_equal(result, expected)

        # 创建一个预期的 Series 对象 expected，根据 op 参数的函数逐元素计算，结果为布尔类型
        expected = Series([op(ser[n], idx2[n]) for n in range(len(ser))], dtype=bool)

        # 进行逻辑运算，并将结果与预期的 Series 对象 expected 进行比较
        result = op(ser, idx2)
        tm.assert_series_equal(result, expected)
    def test_reversed_xor_with_index_returns_series(self):
        # 测试反向逻辑异或操作是否返回序列
        # GH#22092, GH#19792 在2.0版本之前，这些功能被别名为setops
        ser = Series([True, True, False, False])  # 创建包含布尔值的序列
        idx1 = Index([True, False, True, False], dtype=bool)  # 创建布尔类型的索引
        idx2 = Index([1, 0, 1, 0])  # 创建整数索引

        expected = Series([False, True, True, False])  # 预期的结果序列
        result = idx1 ^ ser  # 执行逻辑异或操作
        tm.assert_series_equal(result, expected)  # 检查结果是否符合预期

        result = idx2 ^ ser  # 执行逻辑异或操作
        tm.assert_series_equal(result, expected)  # 检查结果是否符合预期

    @pytest.mark.parametrize(
        "op",
        [
            ops.rand_,
            ops.ror_,
        ],
    )
    def test_reversed_logical_op_with_index_returns_series(self, op):
        # 测试带索引的反向逻辑操作是否返回序列
        # GH#22092, GH#19792
        ser = Series([True, True, False, False])  # 创建包含布尔值的序列
        idx1 = Index([True, False, True, False])  # 创建布尔类型的索引
        idx2 = Index([1, 0, 1, 0])  # 创建整数索引

        expected = Series(op(idx1.values, ser.values))  # 使用操作函数计算预期结果
        result = op(ser, idx1)  # 执行反向逻辑操作
        tm.assert_series_equal(result, expected)  # 检查结果是否符合预期

        expected = op(ser, Series(idx2))  # 使用操作函数计算预期结果
        result = op(ser, idx2)  # 执行反向逻辑操作
        tm.assert_series_equal(result, expected)  # 检查结果是否符合预期

    @pytest.mark.parametrize(
        "op, expected",
        [
            (ops.rand_, [False, False]),  # 使用随机操作函数，期望的结果
            (ops.ror_, [True, True]),  # 使用右位旋转操作函数，期望的结果
            (ops.rxor, [True, True]),  # 使用逆逻辑异或操作函数，期望的结果
        ],
    )
    def test_reverse_ops_with_index(self, op, expected):
        # 测试带索引的反向操作
        # https://github.com/pandas-dev/pandas/pull/23628
        # 多重集合索引操作存在缺陷，因此避免重复...
        # GH#49503
        ser = Series([True, False])  # 创建包含布尔值的序列
        idx = Index([False, True])  # 创建布尔类型的索引

        result = op(ser, idx)  # 执行反向操作
        expected = Series(expected)  # 创建预期的结果序列
        tm.assert_series_equal(result, expected)  # 检查结果是否符合预期
    # 测试逻辑运算在 DataFrame 兼容性上的表现
    def test_logical_ops_df_compat(self):
        # GH#1134

        # 创建两个 Series 对象，每个对象包含布尔值，索引不同
        s1 = Series([True, False, True], index=list("ABC"), name="x")
        s2 = Series([True, True, False], index=list("ABD"), name="x")

        # 预期结果：进行逻辑与操作后的 Series 对象
        exp = Series([True, False, False, False], index=list("ABCD"), name="x")
        tm.assert_series_equal(s1 & s2, exp)
        tm.assert_series_equal(s2 & s1, exp)

        # True | np.nan => True
        exp_or1 = Series([True, True, True, False], index=list("ABCD"), name="x")
        tm.assert_series_equal(s1 | s2, exp_or1)
        # np.nan | True => np.nan，用 False 填充
        exp_or = Series([True, True, False, False], index=list("ABCD"), name="x")
        tm.assert_series_equal(s2 | s1, exp_or)

        # DataFrame 不会用 False 填充 NaN
        tm.assert_frame_equal(s1.to_frame() & s2.to_frame(), exp.to_frame())
        tm.assert_frame_equal(s2.to_frame() & s1.to_frame(), exp.to_frame())

        # 预期结果：将 Series 转为 DataFrame 后进行逻辑与操作
        exp = DataFrame({"x": [True, True, np.nan, np.nan]}, index=list("ABCD"))
        tm.assert_frame_equal(s1.to_frame() | s2.to_frame(), exp_or1.to_frame())
        tm.assert_frame_equal(s2.to_frame() | s1.to_frame(), exp_or.to_frame())

        # 创建两个不同长度的 Series 对象
        s3 = Series([True, False, True], index=list("ABC"), name="x")
        s4 = Series([True, True, True, True], index=list("ABCD"), name="x")

        # 预期结果：进行逻辑与操作后的 Series 对象
        exp = Series([True, False, True, False], index=list("ABCD"), name="x")
        tm.assert_series_equal(s3 & s4, exp)
        tm.assert_series_equal(s4 & s3, exp)

        # np.nan | True => np.nan，用 False 填充
        exp_or1 = Series([True, True, True, False], index=list("ABCD"), name="x")
        tm.assert_series_equal(s3 | s4, exp_or1)
        # True | np.nan => True
        exp_or = Series([True, True, True, True], index=list("ABCD"), name="x")
        tm.assert_series_equal(s4 | s3, exp_or)

        # 预期结果：将 Series 转为 DataFrame 后进行逻辑与操作
        tm.assert_frame_equal(s3.to_frame() & s4.to_frame(), exp.to_frame())
        tm.assert_frame_equal(s4.to_frame() & s3.to_frame(), exp.to_frame())

        # 预期结果：将 Series 转为 DataFrame 后进行逻辑或操作
        tm.assert_frame_equal(s3.to_frame() | s4.to_frame(), exp_or1.to_frame())
        tm.assert_frame_equal(s4.to_frame() | s3.to_frame(), exp_or.to_frame())

    # 测试整数数据类型，不同索引，并非布尔值的异或操作
    def test_int_dtype_different_index_not_bool(self):
        # GH 52500

        # 创建两个 Series 对象，包含整数值和不同的索引
        ser1 = Series([1, 2, 3], index=[10, 11, 23], name="a")
        ser2 = Series([10, 20, 30], index=[11, 10, 23], name="a")

        # 使用 np.bitwise_xor 执行异或操作
        result = np.bitwise_xor(ser1, ser2)
        # 预期结果：异或操作后的 Series 对象
        expected = Series([21, 8, 29], index=[10, 11, 23], name="a")
        tm.assert_series_equal(result, expected)

        # 使用 ^ 运算符执行异或操作
        result = ser1 ^ ser2
        tm.assert_series_equal(result, expected)
    def test_pyarrow_numpy_string_invalid(self):
        # 定义名为 test_pyarrow_numpy_string_invalid 的测试函数
        # GH#56008 表示 GitHub issue 编号为 56008，这里是一种参考注释格式
        pytest.importorskip("pyarrow")  # 导入 pytest 并检查是否存在 pyarrow 模块，否则跳过测试
        ser = Series([False, True])  # 创建一个包含布尔值的 Pandas Series 对象
        ser2 = Series(["a", "b"], dtype="string[pyarrow_numpy]")  # 创建一个指定数据类型的 Pandas Series 对象
        result = ser == ser2  # 执行 Series 对象之间的比较操作
        expected = Series(False, index=ser.index)  # 创建预期结果的 Pandas Series 对象
        tm.assert_series_equal(result, expected)  # 使用 pytest 的测试辅助函数比较 Series 对象是否相等

        result = ser != ser2  # 执行另一种 Series 对象之间的比较操作
        expected = Series(True, index=ser.index)  # 创建另一种预期结果的 Pandas Series 对象
        tm.assert_series_equal(result, expected)  # 使用 pytest 的测试辅助函数比较 Series 对象是否相等

        with pytest.raises(TypeError, match="Invalid comparison"):
            # 使用 pytest 的上下文管理器捕获预期的 TypeError 异常，匹配错误信息为 "Invalid comparison"
            ser > ser2  # 执行不合法的比较操作，预期抛出 TypeError 异常
```