# `D:\src\scipysrc\pandas\pandas\tests\frame\test_arithmetic.py`

```
from collections import deque  # 导入双向队列模块
from datetime import (  # 导入日期时间和时区相关模块
    datetime,  # 日期时间类
    timezone,  # 时区类
)
from enum import Enum  #
    @pytest.mark.parametrize(
        "arg, arg2",
        [  # 参数化测试的参数，包含多个测试用例
            [  # 第一个测试用例
                {  # 参数 arg
                    "a": np.random.default_rng(2).integers(10, size=10),  # 创建随机整数数组 a
                    "b": pd.date_range("20010101", periods=10),  # 创建日期范围数组 b
                },
                {  # 参数 arg2
                    "a": np.random.default_rng(2).integers(10, size=10),  # 创建随机整数数组 a
                    "b": np.random.default_rng(2).integers(10, size=10),  # 创建随机整数数组 b
                },
            ],
            [  # 第二个测试用例
                {  # 参数 arg
                    "a": np.random.default_rng(2).integers(10, size=10),  # 创建随机整数数组 a
                    "b": np.random.default_rng(2).integers(10, size=10),  # 创建随机整数数组 b
                },
                {  # 参数 arg2
                    "a": np.random.default_rng(2).integers(10, size=10),  # 创建随机整数数组 a
                    "b": pd.date_range("20010101", periods=10),  # 创建日期范围数组 b
                },
            ],
            [  # 第三个测试用例
                {  # 参数 arg
                    "a": pd.date_range("20010101", periods=10),  # 创建日期范围数组 a
                    "b": pd.date_range("20010101", periods=10),  # 创建日期范围数组 b
                },
                {  # 参数 arg2
                    "a": np.random.default_rng(2).integers(10, size=10),  # 创建随机整数数组 a
                    "b": np.random.default_rng(2).integers(10, size=10),  # 创建随机整数数组 b
                },
            ],
            [  # 第四个测试用例
                {  # 参数 arg
                    "a": np.random.default_rng(2).integers(10, size=10),  # 创建随机整数数组 a
                    "b": pd.date_range("20010101", periods=10),  # 创建日期范围数组 b
                },
                {  # 参数 arg2
                    "a": pd.date_range("20010101", periods=10),  # 创建日期范围数组 a
                    "b": pd.date_range("20010101", periods=10),  # 创建日期范围数组 b
                },
            ],
        ],
    )
    def test_comparison_invalid(self, arg, arg2):
        # GH4968
        # invalid date/int comparisons
        # 创建 DataFrame 对象 x 和 y，用于进行比较
        x = DataFrame(arg)
        y = DataFrame(arg2)
        
        # 我们期望结果匹配 Series 的比较结果
        # 对于 == 和 !=，不等式应该引发错误
        # 比较 x 和 y，得到结果 result
        result = x == y
        # 期望的比较结果，创建 DataFrame 对象 expected
        expected = DataFrame(
            {col: x[col] == y[col] for col in x.columns},  # 对每列进行比较
            index=x.index,
            columns=x.columns,
        )
        # 使用测试工具比较结果和期望结果
        tm.assert_frame_equal(result, expected)

        # 再次比较 x 和 y，得到结果 result
        result = x != y
        # 期望的比较结果，创建 DataFrame 对象 expected
        expected = DataFrame(
            {col: x[col] != y[col] for col in x.columns},  # 对每列进行比较
            index=x.index,
            columns=x.columns,
        )
        # 使用测试工具比较结果和期望结果
        tm.assert_frame_equal(result, expected)

        # 定义一组错误消息
        msgs = [
            r"Invalid comparison between dtype=datetime64\[ns\] and ndarray",
            "invalid type promotion",
            (
                # npdev 1.20.0
                r"The DTypes <class 'numpy.dtype\[.*\]'> and "
                r"<class 'numpy.dtype\[.*\]'> do not have a common DType."
            ),
        ]
        # 将错误消息组合成一个正则表达式字符串
        msg = "|".join(msgs)
        
        # 使用 pytest 检查是否抛出预期的 TypeError 异常，并匹配错误消息
        with pytest.raises(TypeError, match=msg):
            x >= y
        with pytest.raises(TypeError, match=msg):
            x > y
        with pytest.raises(TypeError, match=msg):
            x < y
        with pytest.raises(TypeError, match=msg):
            x <= y
    @pytest.mark.parametrize(
        "left, right",
        [  # 参数化测试用例，用于多次运行同一个测试函数，测试不同的参数组合
            ("gt", "lt"),  # 第一组参数：left='gt', right='lt'
            ("lt", "gt"),  # 第二组参数：left='lt', right='gt'
            ("ge", "le"),  # 第三组参数：left='ge', right='le'
            ("le", "ge"),  # 第四组参数：left='le', right='ge'
            ("eq", "eq"),  # 第五组参数：left='eq', right='eq'
            ("ne", "ne"),  # 第六组参数：left='ne', right='ne'
        ],
    )
    def test_timestamp_compare(self, left, right):
        # 确保可以在左右两侧比较时间戳
        # GH#4982
        # 创建一个包含多列数据的 DataFrame 对象
        df = DataFrame(
            {
                "dates1": pd.date_range("20010101", periods=10),  # 创建日期范围为10天的日期列
                "dates2": pd.date_range("20010102", periods=10),  # 创建日期范围为10天的日期列
                "intcol": np.random.default_rng(2).integers(1000000000, size=10),  # 创建随机整数列
                "floatcol": np.random.default_rng(2).standard_normal(10),  # 创建随机标准正态分布的浮点数列
                "stringcol": [chr(100 + i) for i in range(10)],  # 创建由ASCII字符组成的字符串列表
            }
        )
        df.loc[np.random.default_rng(2).random(len(df)) > 0.5, "dates2"] = pd.NaT  # 将部分 'dates2' 列的值设为 NaT

        left_f = getattr(operator, left)  # 获取 operator 模块中 left 操作符对应的函数
        right_f = getattr(operator, right)  # 获取 operator 模块中 right 操作符对应的函数

        # 没有 NaT 的情况下进行比较
        if left in ["eq", "ne"]:
            # 使用 left_f 函数比较 DataFrame 和特定时间戳
            expected = left_f(df, pd.Timestamp("20010109"))
            # 使用 right_f 函数比较特定时间戳和 DataFrame
            result = right_f(pd.Timestamp("20010109"), df)
            # 断言比较结果的DataFrame相等性
            tm.assert_frame_equal(result, expected)
        else:
            # 出现错误类型的情况下，抛出 TypeError 异常并匹配特定的错误信息
            msg = (
                "'(<|>)=?' not supported between "
                "instances of 'numpy.ndarray' and 'Timestamp'"
            )
            with pytest.raises(TypeError, match=msg):
                # 使用 left_f 函数比较 DataFrame 和特定时间戳
                left_f(df, pd.Timestamp("20010109"))
            with pytest.raises(TypeError, match=msg):
                # 使用 right_f 函数比较特定时间戳和 DataFrame
                right_f(pd.Timestamp("20010109"), df)

        # 存在 NaT 的情况下进行比较
        if left in ["eq", "ne"]:
            # 使用 left_f 函数比较 DataFrame 和 NaT
            expected = left_f(df, pd.Timestamp("nat"))
            # 使用 right_f 函数比较 NaT 和 DataFrame
            result = right_f(pd.Timestamp("nat"), df)
            # 断言比较结果的DataFrame相等性
            tm.assert_frame_equal(result, expected)
        else:
            # 出现错误类型的情况下，抛出 TypeError 异常并匹配特定的错误信息
            msg = (
                "'(<|>)=?' not supported between "
                "instances of 'numpy.ndarray' and 'NaTType'"
            )
            with pytest.raises(TypeError, match=msg):
                # 使用 left_f 函数比较 DataFrame 和 NaT
                left_f(df, pd.Timestamp("nat"))
            with pytest.raises(TypeError, match=msg):
                # 使用 right_f 函数比较 NaT 和 DataFrame
                right_f(pd.Timestamp("nat"), df)

    @pytest.mark.xfail(
        using_pyarrow_string_dtype(), reason="can't compare string and int"
    )
    def test_mixed_comparison(self):
        # GH#13128, GH#22163 != datetime64 vs non-dt64 should be False,
        # not raise TypeError
        # (this appears to be fixed before GH#22163, not sure when)
        # 创建包含混合类型数据的 DataFrame
        df = DataFrame([["1989-08-01", 1], ["1989-08-01", 2]])
        # 创建另一个包含混合类型数据的 DataFrame
        other = DataFrame([["a", "b"], ["c", "d"]])

        # 比较两个 DataFrame 中的值是否相等，预期为全假（False）
        result = df == other
        assert not result.any().any()

        # 比较两个 DataFrame 中的值是否不相等，预期为全真（True）
        result = df != other
        assert result.all().all()
    # 测试 DataFrame 的布尔值比较出错情况
    def test_df_boolean_comparison_error(self):
        # GH#4576, GH#22880
        # 对比 DataFrame 和长度匹配的列表/元组，len(obj) 等于 len(df.columns) 的支持从 GH#22800 开始
        # 创建一个 3x2 的 DataFrame，数据从 0 到 5
        df = DataFrame(np.arange(6).reshape((3, 2)))

        # 预期的结果 DataFrame，与 df 进行比较
        expected = DataFrame([[False, False], [True, False], [False, False]])

        # 对 DataFrame df 进行比较，期望结果与 expected 相同
        result = df == (2, 2)
        tm.assert_frame_equal(result, expected)

        # 对 DataFrame df 进行比较，期望结果与 expected 相同
        result = df == [2, 2]
        tm.assert_frame_equal(result, expected)

    # 测试 DataFrame 与 None 的浮点数比较
    def test_df_float_none_comparison(self):
        # 创建一个 8x3 的 DataFrame，数据为标准正态分布随机数
        df = DataFrame(
            np.random.default_rng(2).standard_normal((8, 3)),
            index=range(8),
            columns=["A", "B", "C"],
        )

        # 将 DataFrame df 与 None 进行比较，期望结果所有元素都不为 True
        result = df.__eq__(None)
        assert not result.any().any()

    # 测试 DataFrame 的字符串比较
    def test_df_string_comparison(self):
        # 创建一个包含字典的 DataFrame
        df = DataFrame([{"a": 1, "b": "foo"}, {"a": 2, "b": "bar"}])

        # 创建一个布尔掩码，选择 df 中 a 列大于 1 的行
        mask_a = df.a > 1
        # 比较 df 中满足 mask_a 条件的行与预期行，期望结果与 df.loc[1:1, :] 相同
        tm.assert_frame_equal(df[mask_a], df.loc[1:1, :])
        # 比较 df 中不满足 mask_a 条件的行与预期行，期望结果与 df.loc[0:0, :] 相同
        tm.assert_frame_equal(df[-mask_a], df.loc[0:0, :])

        # 创建一个布尔掩码，选择 df 中 b 列等于 "foo" 的行
        mask_b = df.b == "foo"
        # 比较 df 中满足 mask_b 条件的行与预期行，期望结果与 df.loc[0:0, :] 相同
        tm.assert_frame_equal(df[mask_b], df.loc[0:0, :])
        # 比较 df 中不满足 mask_b 条件的行与预期行，期望结果与 df.loc[1:1, :] 相同
        tm.assert_frame_equal(df[-mask_b], df.loc[1:1, :])
class TestFrameFlexComparisons:
    # TODO: test_bool_flex_frame needs a better name
    # 定义测试类 TestFrameFlexComparisons，用于比较灵活的 DataFrame 操作

    def test_bool_flex_frame(self, comparison_op):
        # 在测试方法 test_bool_flex_frame 中，传入了比较操作函数 comparison_op

        # 生成随机数据
        data = np.random.default_rng(2).standard_normal((5, 3))
        other_data = np.random.default_rng(2).standard_normal((5, 3))

        # 创建 DataFrame 对象
        df = DataFrame(data)
        other = DataFrame(other_data)

        # 扩展维度
        ndim_5 = np.ones(df.shape + (1, 3))

        # 测试 DataFrame 相等性
        assert df.eq(df).values.all()
        # 测试 DataFrame 不等性
        assert not df.ne(df).values.any()

        # 获取比较操作的函数对象
        f = getattr(df, comparison_op.__name__)
        o = comparison_op

        # 测试在没有缺失值的情况下的 DataFrame 相等性断言
        tm.assert_frame_equal(f(other), o(df, other))

        # 测试不对齐的情况
        part_o = other.loc[3:, 1:].copy()
        rs = f(part_o)
        xp = o(df, part_o.reindex(index=df.index, columns=df.columns))
        tm.assert_frame_equal(rs, xp)

        # 测试 ndarray 的情况
        tm.assert_frame_equal(f(other.values), o(df, other.values))

        # 测试标量的情况
        tm.assert_frame_equal(f(0), o(df, 0))

        # 测试含有缺失值的情况
        msg = "Unable to coerce to Series/DataFrame"
        tm.assert_frame_equal(f(np.nan), o(df, np.nan))
        with pytest.raises(ValueError, match=msg):
            f(ndim_5)

    @pytest.mark.parametrize("box", [np.array, Series])
    def test_bool_flex_series(self, box):
        # 在测试方法 test_bool_flex_series 中，使用 pytest 的参数化装饰器，测试 Series 相关操作

        # 生成随机数据并创建 DataFrame 对象
        data = np.random.default_rng(2).standard_normal((5, 3))
        df = DataFrame(data)

        # 创建索引和列的 Series 对象
        idx_ser = box(np.random.default_rng(2).standard_normal(5))
        col_ser = box(np.random.default_rng(2).standard_normal(3))

        # 比较索引和列的相等性和不等性
        idx_eq = df.eq(idx_ser, axis=0)
        col_eq = df.eq(col_ser)
        idx_ne = df.ne(idx_ser, axis=0)
        col_ne = df.ne(col_ser)
        tm.assert_frame_equal(col_eq, df == Series(col_ser))
        tm.assert_frame_equal(col_eq, -col_ne)
        tm.assert_frame_equal(idx_eq, -idx_ne)
        tm.assert_frame_equal(idx_eq, df.T.eq(idx_ser).T)
        tm.assert_frame_equal(col_eq, df.eq(list(col_ser)))
        tm.assert_frame_equal(idx_eq, df.eq(Series(idx_ser), axis=0))
        tm.assert_frame_equal(idx_eq, df.eq(list(idx_ser), axis=0))

        # 比较索引和列的大于、小于、大于等于、小于等于操作
        idx_gt = df.gt(idx_ser, axis=0)
        col_gt = df.gt(col_ser)
        idx_le = df.le(idx_ser, axis=0)
        col_le = df.le(col_ser)

        tm.assert_frame_equal(col_gt, df > Series(col_ser))
        tm.assert_frame_equal(col_gt, -col_le)
        tm.assert_frame_equal(idx_gt, -idx_le)
        tm.assert_frame_equal(idx_gt, df.T.gt(idx_ser).T)

        idx_ge = df.ge(idx_ser, axis=0)
        col_ge = df.ge(col_ser)
        idx_lt = df.lt(idx_ser, axis=0)
        col_lt = df.lt(col_ser)

        tm.assert_frame_equal(col_ge, df >= Series(col_ser))
        tm.assert_frame_equal(col_ge, -col_lt)
        tm.assert_frame_equal(idx_ge, -idx_lt)
        tm.assert_frame_equal(idx_ge, df.T.ge(idx_ser).T)

        # 更新索引和列的 Series 对象为新的 Series
        idx_ser = Series(np.random.default_rng(2).standard_normal(5))
        col_ser = Series(np.random.default_rng(2).standard_normal(3))
    def test_bool_flex_frame_na(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        # 将第一个元素设为 NaN
        df.loc[0, 0] = np.nan
        # 检查元素是否等于自身，返回布尔值的DataFrame
        rs = df.eq(df)
        # 断言第一个元素不等于自身
        assert not rs.loc[0, 0]
        # 检查元素是否不等于自身，返回布尔值的DataFrame
        rs = df.ne(df)
        # 断言第一个元素不等于自身
        assert rs.loc[0, 0]
        # 检查元素是否大于自身，返回布尔值的DataFrame
        rs = df.gt(df)
        # 断言第一个元素不大于自身
        assert not rs.loc[0, 0]
        # 检查元素是否小于自身，返回布尔值的DataFrame
        rs = df.lt(df)
        # 断言第一个元素不小于自身
        assert not rs.loc[0, 0]
        # 检查元素是否大于等于自身，返回布尔值的DataFrame
        rs = df.ge(df)
        # 断言第一个元素不大于等于自身
        assert not rs.loc[0, 0]
        # 检查元素是否小于等于自身，返回布尔值的DataFrame
        rs = df.le(df)
        # 断言第一个元素不小于等于自身
        assert not rs.loc[0, 0]

    def test_bool_flex_frame_complex_dtype(self):
        # complex情况
        arr = np.array([np.nan, 1, 6, np.nan])
        arr2 = np.array([2j, np.nan, 7, None])
        df = DataFrame({"a": arr})
        df2 = DataFrame({"a": arr2})

        # 设置错误消息，用于检查预期的错误类型和消息
        msg = "|".join(
            [
                "'>' not supported between instances of '.*' and 'complex'",
                r"unorderable types: .*complex\(\)",  # PY35
            ]
        )
        # 使用pytest断言引发指定类型和消息的TypeError异常
        with pytest.raises(TypeError, match=msg):
            # 复数类型不支持不等式比较
            df.gt(df2)
        with pytest.raises(TypeError, match=msg):
            # 验证Series上相同的行为
            df["a"].gt(df2["a"])
        with pytest.raises(TypeError, match=msg):
            # 验证numpy上的相同行为
            df.values > df2.values

        # 检查元素是否不等于另一个DataFrame的对应元素
        rs = df.ne(df2)
        # 断言所有值为True
        assert rs.values.all()

        arr3 = np.array([2j, np.nan, None])
        df3 = DataFrame({"a": arr3})

        with pytest.raises(TypeError, match=msg):
            # 复数类型不支持不等式比较
            df3.gt(2j)
        with pytest.raises(TypeError, match=msg):
            # 验证Series上相同的行为
            df3["a"].gt(2j)
        with pytest.raises(TypeError, match=msg):
            # 验证numpy上的相同行为
            df3.values > 2j

    def test_bool_flex_frame_object_dtype(self):
        # corner情况，dtype=object
        df1 = DataFrame({"col": ["foo", np.nan, "bar"]}, dtype=object)
        df2 = DataFrame({"col": ["foo", datetime.now(), "bar"]}, dtype=object)
        # 检查两个DataFrame是否不相等
        result = df1.ne(df2)
        # 期望的结果DataFrame
        exp = DataFrame({"col": [False, True, False]})
        # 使用测试工具函数验证结果是否与期望一致
        tm.assert_frame_equal(result, exp)

    def test_flex_comparison_nat(self):
        # GH 15697, GH 22163 df.eq(pd.NaT) 应该与 df == pd.NaT 一致，
        # 绝对不应该是 NaN
        df = DataFrame([pd.NaT])

        # 检查DataFrame中的元素是否等于pd.NaT
        result = df == pd.NaT
        # 使用np.bool_对象来验证结果
        assert result.iloc[0, 0].item() is False

        # 检查DataFrame中的元素是否等于pd.NaT
        result = df.eq(pd.NaT)
        # 使用np.bool_对象来验证结果
        assert result.iloc[0, 0].item() is False

        # 检查DataFrame中的元素是否不等于pd.NaT
        result = df != pd.NaT
        # 使用np.bool_对象来验证结果
        assert result.iloc[0, 0].item() is True

        # 检查DataFrame中的元素是否不等于pd.NaT
        result = df.ne(pd.NaT)
        # 使用np.bool_对象来验证结果
        assert result.iloc[0, 0].item() is True
    # 测试函数：测试DataFrame与常数的灵活比较操作的返回类型
    
    def test_df_flex_cmp_constant_return_types(self, comparison_op):
        # GH 15077, 非空DataFrame
        df = DataFrame({"x": [1, 2, 3], "y": [1.0, 2.0, 3.0]})
        const = 2
    
        # 使用getattr动态调用DataFrame的比较操作方法，比较DataFrame与常数const的结果
        result = getattr(df, comparison_op.__name__)(const).dtypes.value_counts()
        # 断言比较结果的数据类型计数为两种类型，都应为布尔类型(np.dtype(bool))
        tm.assert_series_equal(
            result, Series([2], index=[np.dtype(bool)], name="count")
        )
    
    # 测试函数：测试空DataFrame与常数的灵活比较操作的返回类型
    
    def test_df_flex_cmp_constant_return_types_empty(self, comparison_op):
        # GH 15077 空DataFrame
        df = DataFrame({"x": [1, 2, 3], "y": [1.0, 2.0, 3.0]})
        const = 2
    
        # 创建一个空的DataFrame
        empty = df.iloc[:0]
        # 使用getattr动态调用空DataFrame的比较操作方法，比较空DataFrame与常数const的结果
        result = getattr(empty, comparison_op.__name__)(const).dtypes.value_counts()
        # 断言比较结果的数据类型计数为两种类型，都应为布尔类型(np.dtype(bool))
        tm.assert_series_equal(
            result, Series([2], index=[np.dtype(bool)], name="count")
        )
    
    # 测试函数：测试DataFrame与ndarray Series的数据类型比较
    
    def test_df_flex_cmp_ea_dtype_with_ndarray_series(self):
        # 创建一个IntervalIndex对象ii
        ii = pd.IntervalIndex.from_breaks([1, 2, 3])
        # 创建DataFrame对象df，包含两列A和B，值为IntervalIndex对象ii
        df = DataFrame({"A": ii, "B": ii})
    
        # 创建一个Series对象ser，值为[0, 0]
        ser = Series([0, 0])
        # 使用DataFrame的eq方法，沿axis=0比较DataFrame df与Series ser的结果
        res = df.eq(ser, axis=0)
    
        # 创建一个期望的DataFrame对象expected，预期值为全False的DataFrame
        expected = DataFrame({"A": [False, False], "B": [False, False]})
        # 断言DataFrame res 与 expected 相等
        tm.assert_frame_equal(res, expected)
    
        # 创建一个Series对象ser2，值为[1, 2]，索引为["A", "B"]
        ser2 = Series([1, 2], index=["A", "B"])
        # 使用DataFrame的eq方法，沿axis=1比较DataFrame df与Series ser2的结果
        res2 = df.eq(ser2, axis=1)
        # 断言DataFrame res2 与 expected 相等
        tm.assert_frame_equal(res2, expected)
# -------------------------------------------------------------------
# Arithmetic

class TestFrameFlexArithmetic:
    def test_floordiv_axis0(self):
        # 确保 df.floordiv(ser, axis=0) 结果与按列计算的结果匹配
        arr = np.arange(3)  # 创建一个包含0到2的数组
        ser = Series(arr)  # 将数组转换为 Pandas Series
        df = DataFrame({"A": ser, "B": ser})  # 创建包含两列的 DataFrame，使用同一 Series 作为数据

        result = df.floordiv(ser, axis=0)  # 对 DataFrame 按 axis=0 进行整除操作

        expected = DataFrame({col: df[col] // ser for col in df.columns})  # 创建预期的 DataFrame，按列进行整除操作

        tm.assert_frame_equal(result, expected)  # 断言两个 DataFrame 是否相等

        result2 = df.floordiv(ser.values, axis=0)  # 对 DataFrame 的值数组进行整除操作
        tm.assert_frame_equal(result2, expected)  # 断言结果是否与预期相等

    def test_df_add_td64_columnwise(self):
        # GH 22534 检查列逐列的加法是否广播正确
        dti = pd.date_range("2016-01-01", periods=10)  # 创建一个日期时间索引
        tdi = pd.timedelta_range("1", periods=10)  # 创建一个时间增量索引
        tser = Series(tdi)  # 将时间增量索引转换为 Series
        df = DataFrame({0: dti, 1: tdi})  # 创建包含两列的 DataFrame，包括日期和时间增量

        result = df.add(tser, axis=0)  # 对 DataFrame 按 axis=0 进行加法操作

        expected = DataFrame({0: dti + tdi, 1: tdi + tdi})  # 创建预期的 DataFrame，按列进行加法操作
        tm.assert_frame_equal(result, expected)  # 断言两个 DataFrame 是否相等

    def test_df_add_flex_filled_mixed_dtypes(self):
        # GH 19611 检查混合类型 DataFrame 的填充加法操作
        dti = pd.date_range("2016-01-01", periods=3)  # 创建一个日期时间索引
        ser = Series(["1 Day", "NaT", "2 Days"], dtype="timedelta64[ns]")  # 创建一个时间增量 Series
        df = DataFrame({"A": dti, "B": ser})  # 创建包含日期和时间增量的 DataFrame
        other = DataFrame({"A": ser, "B": ser})  # 创建另一个包含时间增量的 DataFrame
        fill = pd.Timedelta(days=1).to_timedelta64()  # 创建一个填充值，表示一天的时间增量

        result = df.add(other, fill_value=fill)  # 使用填充值对 DataFrame 进行加法操作

        expected = DataFrame(
            {
                "A": Series(
                    ["2016-01-02", "2016-01-03", "2016-01-05"], dtype="datetime64[ns]"
                ),  # 创建预期的日期时间 Series
                "B": ser * 2,  # 对时间增量 Series 进行加法操作
            }
        )
        tm.assert_frame_equal(result, expected)  # 断言两个 DataFrame 是否相等

    def test_arith_flex_frame(
        self, all_arithmetic_operators, float_frame, mixed_float_frame
    ):
        # 使用参数化的 fixture 进行灵活的 DataFrame 算术操作
        op = all_arithmetic_operators  # 获取当前参数化的算术操作

        def f(x, y):
            # 如果操作以 "__r" 开头，则从 operator 模块获取相反操作
            if op.startswith("__r"):
                return getattr(operator, op.replace("__r", "__"))(y, x)
            return getattr(operator, op)(x, y)  # 否则直接获取操作

        result = getattr(float_frame, op)(2 * float_frame)  # 对 float 类型的 DataFrame 进行算术操作
        expected = f(float_frame, 2 * float_frame)  # 获取预期的操作结果
        tm.assert_frame_equal(result, expected)  # 断言两个 DataFrame 是否相等

        # 对混合 float 类型的 DataFrame 进行相同的算术操作
        result = getattr(mixed_float_frame, op)(2 * mixed_float_frame)
        expected = f(mixed_float_frame, 2 * mixed_float_frame)
        tm.assert_frame_equal(result, expected)  # 断言两个 DataFrame 是否相等
        _check_mixed_float(result, dtype={"C": None})  # 检查混合 float 的特定操作

    @pytest.mark.parametrize("op", ["__add__", "__sub__", "__mul__"])
    def test_arith_flex_frame_mixed(
        self,
        op,
        int_frame,
        mixed_int_frame,
        mixed_float_frame,
        switch_numexpr_min_elements,
        **kwargs
    ):
        # 参数化测试不同算术操作对混合 DataFrame 的影响
    ):
        # 获取操作符对应的函数对象
        f = getattr(operator, op)

        # 对混合整数进行操作
        result = getattr(mixed_int_frame, op)(2 + mixed_int_frame)
        # 预期结果，使用 Python 内建的操作符函数
        expected = f(mixed_int_frame, 2 + mixed_int_frame)

        # 在无符号整数中不会溢出
        dtype = None
        if op in ["__sub__"]:
            dtype = {"B": "uint64", "C": None}
        elif op in ["__add__", "__mul__"]:
            dtype = {"C": None}
        
        # 当使用 numexpr 时，类型转换规则略有不同：
        # 在 `2 + mixed_int_frame` 操作中，int32 列会变为 int64 列
        # （与 Python 标量在操作中不保留 dtype 的不同），然后 int32/int64
        # 的组合会得到 int64 结果
        if expr.USE_NUMEXPR and switch_numexpr_min_elements == 0:
            dtype["A"] = (2 + mixed_int_frame)["A"].dtype
        # 检查混合整数的结果
        tm.assert_frame_equal(result, expected)
        _check_mixed_int(result, dtype=dtype)

        # 对混合浮点数进行操作
        result = getattr(mixed_float_frame, op)(2 * mixed_float_frame)
        expected = f(mixed_float_frame, 2 * mixed_float_frame)
        tm.assert_frame_equal(result, expected)
        _check_mixed_float(result, dtype={"C": None})

        # 对普通整数进行操作
        result = getattr(int_frame, op)(2 * int_frame)
        expected = f(int_frame, 2 * int_frame)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dim", range(3, 6))
    def test_arith_flex_frame_raise(self, all_arithmetic_operators, float_frame, dim):
        # parametrize 装饰器的一个实例
        op = all_arithmetic_operators

        # 检查维度大于等于3时是否会抛出异常
        arr = np.ones((1,) * dim)
        msg = "Unable to coerce to Series/DataFrame"
        with pytest.raises(ValueError, match=msg):
            getattr(float_frame, op)(arr)

    def test_arith_flex_frame_corner(self, float_frame):
        # 常数加法
        const_add = float_frame.add(1)
        tm.assert_frame_equal(const_add, float_frame + 1)

        # 边界情况
        result = float_frame.add(float_frame[:0])
        expected = float_frame.sort_index() * np.nan
        tm.assert_frame_equal(result, expected)

        result = float_frame[:0].add(float_frame)
        expected = float_frame.sort_index() * np.nan
        tm.assert_frame_equal(result, expected)

        # 带有特定参数的异常情况
        with pytest.raises(NotImplementedError, match="fill_value"):
            float_frame.add(float_frame.iloc[0], fill_value=3)

        with pytest.raises(NotImplementedError, match="fill_value"):
            float_frame.add(float_frame.iloc[0], axis="index", fill_value=3)

    @pytest.mark.parametrize("op", ["add", "sub", "mul", "mod"])
    def test_arith_flex_series_ops(self, simple_frame, op):
        # after arithmetic refactor, add truediv here
        # 从参数中获取简单数据框架和操作符
        df = simple_frame

        # 从数据框架中获取名为"a"的行
        row = df.xs("a")
        # 从数据框架中获取名为"two"的列
        col = df["two"]
        # 根据操作符获取数据框架中对应的操作函数
        f = getattr(df, op)
        # 根据操作符获取对应的操作函数（如加法、减法等）
        op = getattr(operator, op)
        # 断言应用操作函数于行的结果与操作函数应用于数据框架和行的结果相等
        tm.assert_frame_equal(f(row), op(df, row))
        # 断言应用操作函数于列的结果与转置后的数据框架和列的结果的转置相等
        tm.assert_frame_equal(f(col, axis=0), op(df.T, col).T)

    def test_arith_flex_series(self, simple_frame):
        # 从参数中获取简单数据框架
        df = simple_frame

        # 从数据框架中获取名为"a"的行
        row = df.xs("a")
        # 从数据框架中获取名为"two"的列
        col = df["two"]
        # 特殊情况的断言，未来会进行大的算术重构
        tm.assert_frame_equal(df.add(row, axis=None), df + row)

        # 断言数据框架除以行的结果与数据框架除以行的运算符相等
        tm.assert_frame_equal(df.div(row), df / row)
        # 断言数据框架除以列的结果与数据框架转置后除以列的结果的转置相等
        tm.assert_frame_equal(df.div(col, axis=0), (df.T / col).T)

    def test_arith_flex_series_broadcasting(self, any_real_numpy_dtype):
        # GH 7325中的广播问题
        # 创建一个具有特定数据类型的数据框架，用于广播问题
        df = DataFrame(np.arange(3 * 2).reshape((3, 2)), dtype=any_real_numpy_dtype)
        # 预期的结果数据框架
        expected = DataFrame([[np.nan, np.inf], [1.0, 1.5], [1.0, 1.25]])
        if any_real_numpy_dtype == "float32":
            expected = expected.astype(any_real_numpy_dtype)
        # 对数据框架进行除法运算，指定轴为索引
        result = df.div(df[0], axis="index")
        # 断言结果数据框架与预期数据框架相等
        tm.assert_frame_equal(result, expected)

    def test_arith_flex_zero_len_raises(self):
        # GH 19522中的特殊情况，传递fill_value给数据框架灵活算术方法应该引发异常
        # 创建空长度序列和数据框架
        ser_len0 = Series([], dtype=object)
        df_len0 = DataFrame(columns=["A", "B"])
        df = DataFrame([[1, 2], [3, 4]], columns=["A", "B"])

        # 使用pytest断言引发NotImplementedError异常，匹配异常信息包含"fill_value"
        with pytest.raises(NotImplementedError, match="fill_value"):
            df.add(ser_len0, fill_value="E")

        # 使用pytest断言引发NotImplementedError异常，匹配异常信息包含"fill_value"
        with pytest.raises(NotImplementedError, match="fill_value"):
            df_len0.sub(df["A"], axis=None, fill_value=3)

    def test_flex_add_scalar_fill_value(self):
        # GH#12723
        # 创建一个包含NaN值的浮点数数组，并将其转换为数据框架
        dat = np.array([0, 1, np.nan, 3, 4, 5], dtype="float")
        df = DataFrame({"foo": dat}, index=range(6))

        # 期望的结果数据框架，将NaN填充为0后加上2
        exp = df.fillna(0).add(2)
        # 使用fill_value=0进行标量加法运算
        res = df.add(2, fill_value=0)
        # 断言结果数据框架与期望数据框架相等
        tm.assert_frame_equal(res, exp)

    def test_sub_alignment_with_duplicate_index(self):
        # GH#5185重复索引对齐操作应该有效
        # 创建具有重复索引的数据框架
        df1 = DataFrame([1, 2, 3, 4, 5], index=[1, 2, 1, 2, 3])
        df2 = DataFrame([1, 2, 3], index=[1, 2, 3])
        # 期望的结果数据框架，df1减去df2的结果
        expected = DataFrame([0, 2, 0, 2, 2], index=[1, 1, 2, 2, 3])
        # 执行数据框架减法操作
        result = df1.sub(df2)
        # 断言结果数据框架与期望数据框架相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("op", ["__add__", "__mul__", "__sub__", "__truediv__"])
    def test_arithmetic_with_duplicate_columns(self, op):
        # 进行操作
        # 创建包含"A"和"B"列的数据框架，以及随机数生成器为2的随机数
        df = DataFrame({"A": np.arange(10), "B": np.random.default_rng(2).random(10)})
        # 获取预期的数据框架，使用getattr获取操作函数并应用于数据框架
        expected = getattr(df, op)(df)
        expected.columns = ["A", "A"]
        df.columns = ["A", "A"]
        # 获取结果数据框架，使用getattr获取操作函数并应用于数据框架
        result = getattr(df, op)(df)
        # 断言结果数据框架与预期数据框架相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("level", [0, None])
    def test_broadcast_multiindex(self, level):
        # GH34388
        # 创建一个包含两列的DataFrame
        df1 = DataFrame({"A": [0, 1, 2], "B": [1, 2, 3]})
        # 设置第一列的名称为"L1"
        df1.columns = df1.columns.set_names("L1")

        # 创建一个包含两列的DataFrame，列名为二级索引
        df2 = DataFrame({("A", "C"): [0, 0, 0], ("A", "D"): [0, 0, 0]})
        # 设置列名为多级索引
        df2.columns = df2.columns.set_names(["L1", "L2"])

        # 对df1和df2进行加法操作，指定level参数
        result = df1.add(df2, level=level)
        # 创建预期的DataFrame，列名为多级索引
        expected = DataFrame({("A", "C"): [0, 1, 2], ("A", "D"): [0, 1, 2]})
        # 设置预期DataFrame的列名为多级索引
        expected.columns = expected.columns.set_names(["L1", "L2"])

        # 使用测试工具比较结果DataFrame和预期DataFrame
        tm.assert_frame_equal(result, expected)

    def test_frame_multiindex_operations(self):
        # GH 43321
        # 创建一个包含两列的DataFrame，带有多级索引
        df = DataFrame(
            {2010: [1, 2, 3], 2020: [3, 4, 5]},
            index=MultiIndex.from_product(
                [["a"], ["b"], [0, 1, 2]], names=["scen", "mod", "id"]
            ),
        )

        # 创建一个包含单列的Series，带有多级索引
        series = Series(
            [0.4],
            index=MultiIndex.from_product([["b"], ["a"]], names=["mod", "scen"]),
        )

        # 创建预期的DataFrame，带有多级索引
        expected = DataFrame(
            {2010: [1.4, 2.4, 3.4], 2020: [3.4, 4.4, 5.4]},
            index=MultiIndex.from_product(
                [["a"], ["b"], [0, 1, 2]], names=["scen", "mod", "id"]
            ),
        )
        # 对df和series进行沿轴0方向的加法操作
        result = df.add(series, axis=0)

        # 使用测试工具比较结果DataFrame和预期DataFrame
        tm.assert_frame_equal(result, expected)

    def test_frame_multiindex_operations_series_index_to_frame_index(self):
        # GH 43321
        # 创建一个包含单列的DataFrame，带有多级索引
        df = DataFrame(
            {2010: [1], 2020: [3]},
            index=MultiIndex.from_product([["a"], ["b"]], names=["scen", "mod"]),
        )

        # 创建一个包含三个元素的Series，带有多级索引
        series = Series(
            [10.0, 20.0, 30.0],
            index=MultiIndex.from_product(
                [["a"], ["b"], [0, 1, 2]], names=["scen", "mod", "id"]
            ),
        )

        # 创建预期的DataFrame，带有多级索引
        expected = DataFrame(
            {2010: [11.0, 21, 31.0], 2020: [13.0, 23.0, 33.0]},
            index=MultiIndex.from_product(
                [["a"], ["b"], [0, 1, 2]], names=["scen", "mod", "id"]
            ),
        )
        # 对df和series进行沿轴0方向的加法操作
        result = df.add(series, axis=0)

        # 使用测试工具比较结果DataFrame和预期DataFrame
        tm.assert_frame_equal(result, expected)

    def test_frame_multiindex_operations_no_align(self):
        # 创建一个包含两列的DataFrame，带有多级索引
        df = DataFrame(
            {2010: [1, 2, 3], 2020: [3, 4, 5]},
            index=MultiIndex.from_product(
                [["a"], ["b"], [0, 1, 2]], names=["scen", "mod", "id"]
            ),
        )

        # 创建一个包含单列的Series，带有多级索引
        series = Series(
            [0.4],
            index=MultiIndex.from_product([["c"], ["a"]], names=["mod", "scen"]),
        )

        # 创建预期的DataFrame，带有多级索引
        expected = DataFrame(
            {2010: np.nan, 2020: np.nan},
            index=MultiIndex.from_tuples(
                [
                    ("a", "b", 0),
                    ("a", "b", 1),
                    ("a", "b", 2),
                    ("a", "c", np.nan),
                ],
                names=["scen", "mod", "id"],
            ),
        )
        # 对df和series进行沿轴0方向的加法操作
        result = df.add(series, axis=0)

        # 使用测试工具比较结果DataFrame和预期DataFrame
        tm.assert_frame_equal(result, expected)
    # 定义一个测试函数，用于测试多级索引 DataFrame 的加法操作
    def test_frame_multiindex_operations_part_align(self):
        # 创建一个 DataFrame 对象 df，包含两列数据 2010 和 2020
        df = DataFrame(
            {2010: [1, 2, 3], 2020: [3, 4, 5]},
            # 使用 MultiIndex.from_tuples 创建多级索引，包含三个级别的元组
            index=MultiIndex.from_tuples(
                [
                    ("a", "b", 0),
                    ("a", "b", 1),
                    ("a", "c", 2),
                ],
                # 指定多级索引的名称为 scen, mod, id
                names=["scen", "mod", "id"],
            ),
        )

        # 创建一个 Series 对象 series，包含一个值为 0.4 的元素
        # 使用 MultiIndex.from_product 创建多级索引，包含 mod 和 scen 两个级别
        series = Series(
            [0.4],
            index=MultiIndex.from_product([["b"], ["a"]], names=["mod", "scen"]),
        )

        # 创建一个预期结果的 DataFrame 对象 expected
        expected = DataFrame(
            {2010: [1.4, 2.4, np.nan], 2020: [3.4, 4.4, np.nan]},
            # 使用 MultiIndex.from_tuples 创建多级索引，与 df 的索引相同
            index=MultiIndex.from_tuples(
                [
                    ("a", "b", 0),
                    ("a", "b", 1),
                    ("a", "c", 2),
                ],
                names=["scen", "mod", "id"],
            ),
        )
        
        # 调用 DataFrame 的 add 方法进行加法操作，axis=0 表示按行进行加法
        result = df.add(series, axis=0)

        # 使用 tm.assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
class TestFrameArithmetic:
    def test_td64_op_nat_casting(self):
        # Make sure we don't accidentally treat timedelta64(NaT) as datetime64
        # when calling dispatch_to_series in DataFrame arithmetic
        
        # 创建一个包含两个"NaT"字符串的 Series，指定数据类型为 timedelta64[ns]
        ser = Series(["NaT", "NaT"], dtype="timedelta64[ns]")
        # 创建一个 2x2 的 DataFrame
        df = DataFrame([[1, 2], [3, 4]])

        # 在 DataFrame 上执行乘法操作
        result = df * ser
        # 期望的结果是一个 DataFrame，其中每列都与 ser 相同
        expected = DataFrame({0: ser, 1: ser})
        # 断言两个 DataFrame 相等
        tm.assert_frame_equal(result, expected)

    def test_df_add_2d_array_rowlike_broadcasts(self):
        # GH#23000
        
        # 创建一个 3x2 的数组 arr
        arr = np.arange(6).reshape(3, 2)
        # 创建一个与 arr 对应的 DataFrame，列名为 [True, False]，行名为 ["A", "B", "C"]
        df = DataFrame(arr, columns=[True, False], index=["A", "B", "C"])

        # 选择 arr 的第二行，变为形状为 (1, ncols) 的行向量
        rowlike = arr[[1], :]  # shape --> (1, ncols)
        # 断言 rowlike 的形状为 (1, df.shape[1])
        assert rowlike.shape == (1, df.shape[1])

        # 创建期望的结果 DataFrame，是将 df 的每个元素与 rowlike 相加
        expected = DataFrame(
            [[2, 4], [4, 6], [6, 8]],
            columns=df.columns,
            index=df.index,
            # 明确指定数据类型以避免在 32 位构建中失败
            dtype=arr.dtype,
        )
        # 对 df 和 rowlike 执行加法操作，并断言结果与 expected 相等
        result = df + rowlike
        tm.assert_frame_equal(result, expected)
        # 对 rowlike 和 df 执行加法操作，并断言结果与 expected 相等
        result = rowlike + df
        tm.assert_frame_equal(result, expected)

    def test_df_add_2d_array_collike_broadcasts(self):
        # GH#23000
        
        # 创建一个 3x2 的数组 arr
        arr = np.arange(6).reshape(3, 2)
        # 创建一个与 arr 对应的 DataFrame，列名为 [True, False]，行名为 ["A", "B", "C"]
        df = DataFrame(arr, columns=[True, False], index=["A", "B", "C"])

        # 选择 arr 的第二列，变为形状为 (nrows, 1) 的列向量
        collike = arr[:, [1]]  # shape --> (nrows, 1)
        # 断言 collike 的形状为 (df.shape[0], 1)
        assert collike.shape == (df.shape[0], 1)

        # 创建期望的结果 DataFrame，是将 df 的每个元素与 collike 相加
        expected = DataFrame(
            [[1, 2], [5, 6], [9, 10]],
            columns=df.columns,
            index=df.index,
            # 明确指定数据类型以避免在 32 位构建中失败
            dtype=arr.dtype,
        )
        # 对 df 和 collike 执行加法操作，并断言结果与 expected 相等
        result = df + collike
        tm.assert_frame_equal(result, expected)
        # 对 collike 和 df 执行加法操作，并断言结果与 expected 相等
        result = collike + df
        tm.assert_frame_equal(result, expected)

    def test_df_arith_2d_array_rowlike_broadcasts(
        self, request, all_arithmetic_operators
    ):
        # GH#23000
        
        # 从参数中获取所有的算术操作符的名称
        opname = all_arithmetic_operators
        # 创建一个 3x2 的数组 arr
        arr = np.arange(6).reshape(3, 2)
        # 创建一个与 arr 对应的 DataFrame，列名为 [True, False]，行名为 ["A", "B", "C"]
        df = DataFrame(arr, columns=[True, False], index=["A", "B", "C"])

        # 选择 arr 的第二行，变为形状为 (1, ncols) 的行向量
        rowlike = arr[[1], :]  # shape --> (1, ncols)
        # 断言 rowlike 的形状为 (1, df.shape[1])
        assert rowlike.shape == (1, df.shape[1])

        # 创建一个列表 exvals，包含 df 每行与 rowlike 执行指定操作后的结果
        exvals = [
            getattr(df.loc["A"], opname)(rowlike.squeeze()),
            getattr(df.loc["B"], opname)(rowlike.squeeze()),
            getattr(df.loc["C"], opname)(rowlike.squeeze()),
        ]

        # 创建期望的结果 DataFrame，每行的结果是 df 对应行与 rowlike 执行指定操作后的结果
        expected = DataFrame(exvals, columns=df.columns, index=df.index)

        # 对 df 和 rowlike 执行指定操作，并断言结果与 expected 相等
        result = getattr(df, opname)(rowlike)
        tm.assert_frame_equal(result, expected)

    def test_df_arith_2d_array_collike_broadcasts(
        self, request, all_arithmetic_operators
    ):
    ):
        # GH#23000
        # 设置操作符名称为所有算术运算符
        opname = all_arithmetic_operators
        # 创建一个包含6个元素的NumPy数组，并将其重塑为3行2列的DataFrame
        arr = np.arange(6).reshape(3, 2)
        # 使用数组创建DataFrame，列名分别为True和False，索引为"A", "B", "C"
        df = DataFrame(arr, columns=[True, False], index=["A", "B", "C"])

        # 提取数组的第二列，并保持其形状为(nrows, 1)
        collike = arr[:, [1]]  # shape --> (nrows, 1)
        # 断言提取的列形状与DataFrame的行数相同
        assert collike.shape == (df.shape[0], 1)

        # 根据列名（True或False），调用DataFrame的对应方法对collike进行操作
        exvals = {
            True: getattr(df[True], opname)(collike.squeeze()),
            False: getattr(df[False], opname)(collike.squeeze()),
        }

        # 如果操作名在["__rmod__", "__rfloordiv__"]中，则可能返回混合的int/float类型，
        # 这时我们需要确定期望的数据类型
        dtype = None
        if opname in ["__rmod__", "__rfloordiv__"]:
            dtype = np.common_type(*(x.values for x in exvals.values()))

        # 创建期望的DataFrame对象，列名和索引与原DataFrame相同，数据类型为之前确定的dtype
        expected = DataFrame(exvals, columns=df.columns, index=df.index, dtype=dtype)

        # 对DataFrame df调用操作名对应的方法，得到结果DataFrame
        result = getattr(df, opname)(collike)
        # 使用测试模块中的方法，断言结果DataFrame与期望的DataFrame相等
        tm.assert_frame_equal(result, expected)

    def test_df_bool_mul_int(self):
        # GH 22047, GH 22163 multiplication by 1 should result in int dtype,
        # not object dtype
        # 创建一个包含布尔值的2x2的DataFrame
        df = DataFrame([[False, True], [False, False]])
        # 将DataFrame的每个元素乘以1，预期结果的数据类型应为int而不是object
        result = df * 1

        # 在Appveyor上，结果可能是np.int32而不是np.int64，因此检查dtype.kind而不仅仅是dtype
        kinds = result.dtypes.apply(lambda x: x.kind)
        # 断言结果DataFrame的所有列的数据类型的kind为"i"
        assert (kinds == "i").all()

        # 另一种方式，1乘以DataFrame的每个元素
        result = 1 * df
        kinds = result.dtypes.apply(lambda x: x.kind)
        # 再次断言结果DataFrame的所有列的数据类型的kind为"i"
        assert (kinds == "i").all()

    def test_arith_mixed(self):
        # 创建一个包含字符串和整数的DataFrame
        left = DataFrame({"A": ["a", "b", "c"], "B": [1, 2, 3]})

        # 对DataFrame中的每列进行加法操作，结果应为字符串和整数的组合
        result = left + left
        # 创建期望的DataFrame，对每列进行相同的字符串和整数的加法操作
        expected = DataFrame({"A": ["aa", "bb", "cc"], "B": [2, 4, 6]})
        # 使用测试模块中的方法，断言结果DataFrame与期望的DataFrame相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("col", ["A", "B"])
    def test_arith_getitem_commute(self, all_arithmetic_functions, col):
        # 创建一个包含浮点数的DataFrame
        df = DataFrame({"A": [1.1, 3.3], "B": [2.5, -3.9]})
        # 调用测试函数，并对DataFrame的某列进行操作，返回结果Series
        result = all_arithmetic_functions(df, 1)[col]
        # 调用测试函数对DataFrame的某列进行操作，期望返回结果Series
        expected = all_arithmetic_functions(df[col], 1)
        # 使用测试模块中的方法，断言结果Series与期望的Series相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "values", [[1, 2], (1, 2), np.array([1, 2]), range(1, 3), deque([1, 2])]
    )
    def test_arith_alignment_non_pandas_object(self, values):
        # GH#17901
        # 创建一个包含整数的DataFrame
        df = DataFrame({"A": [1, 1], "B": [1, 1]})
        # 创建期望的DataFrame，将df的每个元素与values中的每个元素进行加法操作
        expected = DataFrame({"A": [2, 2], "B": [3, 3]})
        # 对DataFrame df和非Pandas对象values进行加法操作，得到结果DataFrame
        result = df + values
        # 使用测试模块中的方法，断言结果DataFrame与期望的DataFrame相等
        tm.assert_frame_equal(result, expected)
    # 定义测试函数，测试非 Pandas 对象的算术操作
    def test_arith_non_pandas_object(self):
        # 创建一个包含浮点数的 DataFrame，形状为 3x3，列名为 ["one", "two", "three"]，行索引为 ["a", "b", "c"]
        df = DataFrame(
            np.arange(1, 10, dtype="f8").reshape(3, 3),
            columns=["one", "two", "three"],
            index=["a", "b", "c"],
        )

        # 获取行索引为 "a" 的行的值
        val1 = df.xs("a").values
        # 创建一个新的 DataFrame，其值为原始 DataFrame 和 val1 的值之和，索引和列名与原始 DataFrame 相同
        added = DataFrame(df.values + val1, index=df.index, columns=df.columns)
        # 使用测试工具函数确认两个 DataFrame 相等
        tm.assert_frame_equal(df + val1, added)

        # 创建一个新的 DataFrame，其值为原始 DataFrame 的转置与 val1 的值之和的转置，索引和列名与原始 DataFrame 相同
        added = DataFrame((df.values.T + val1).T, index=df.index, columns=df.columns)
        # 使用测试工具函数确认两个 DataFrame 相等
        tm.assert_frame_equal(df.add(val1, axis=0), added)

        # 获取列名为 "two" 的列的值，并转换为列表
        val2 = list(df["two"])
        # 创建一个新的 DataFrame，其值为原始 DataFrame 和 val2 的值之和，索引和列名与原始 DataFrame 相同
        added = DataFrame(df.values + val2, index=df.index, columns=df.columns)
        # 使用测试工具函数确认两个 DataFrame 相等
        tm.assert_frame_equal(df + val2, added)

        # 创建一个新的 DataFrame，其值为原始 DataFrame 的转置与 val2 的值之和的转置，索引和列名与原始 DataFrame 相同
        added = DataFrame((df.values.T + val2).T, index=df.index, columns=df.columns)
        # 使用测试工具函数确认两个 DataFrame 相等
        tm.assert_frame_equal(df.add(val2, axis="index"), added)

        # 使用随机数生成器生成与原始 DataFrame 相同形状的随机数值数组
        val3 = np.random.default_rng(2).random(df.shape)
        # 创建一个新的 DataFrame，其值为原始 DataFrame 和 val3 的值之和，索引和列名与原始 DataFrame 相同
        added = DataFrame(df.values + val3, index=df.index, columns=df.columns)
        # 使用测试工具函数确认两个 DataFrame 相等
        tm.assert_frame_equal(df.add(val3), added)



    # 定义测试函数，测试带有区间类别索引的 DataFrame 的操作
    def test_operations_with_interval_categories_index(self, all_arithmetic_operators):
        # GH#27415
        # 选择所有算术运算符
        op = all_arithmetic_operators
        # 创建一个包含区间索引的分类索引对象，范围从 0.0 到 2.0
        ind = pd.CategoricalIndex(pd.interval_range(start=0.0, end=2.0))
        # 创建一个包含单行数据的 DataFrame，列名为区间索引
        data = [1, 2]
        df = DataFrame([data], columns=ind)
        # 定义一个数值
        num = 10
        # 执行指定运算符操作并生成结果 DataFrame
        result = getattr(df, op)(num)
        # 创建一个预期的 DataFrame，其值为对每个数据应用指定运算符操作的结果
        expected = DataFrame([[getattr(n, op)(num) for n in data]], columns=ind)
        # 使用测试工具函数确认两个 DataFrame 相等
        tm.assert_frame_equal(result, expected)



    # 定义测试函数，测试带有重索引的 DataFrame 之间的操作
    def test_frame_with_frame_reindex(self):
        # GH#31623
        # 创建一个包含时间戳数据的 DataFrame，列名为 ["foo", "bar"]，数据类型为 'M8[ns]'
        df = DataFrame(
            {
                "foo": [pd.Timestamp("2019"), pd.Timestamp("2020")],
                "bar": [pd.Timestamp("2018"), pd.Timestamp("2021")],
            },
            columns=["foo", "bar"],
            dtype="M8[ns]",
        )
        # 从 df 中选择 "foo" 列并创建新的 DataFrame
        df2 = df[["foo"]]

        # 计算 df 与 df2 之间的差值
        result = df - df2

        # 创建一个预期的 DataFrame，其值为对每个元素应用差值操作的结果
        expected = DataFrame(
            {"foo": [pd.Timedelta(0), pd.Timedelta(0)], "bar": [np.nan, np.nan]},
            columns=["bar", "foo"],
        )
        # 使用测试工具函数确认两个 DataFrame 相等
        tm.assert_frame_equal(result, expected)



    # 使用参数化测试的方式，定义多个测试用例
    @pytest.mark.parametrize(
        "value, dtype",
        [
            (1, "i8"),
            (1.0, "f8"),
            (2**63, "f8"),
            (1j, "complex128"),
            (2**63, "complex128"),
            (True, "bool"),
            (np.timedelta64(20, "ns"), "<m8[ns]"),
            (np.datetime64(20, "ns"), "<M8[ns]"),
        ],
    )
    @pytest.mark.parametrize(
        "op",
        [
            operator.add,
            operator.sub,
            operator.mul,
            operator.truediv,
            operator.mod,
            operator.pow,
        ],
        ids=lambda x: x.__name__,
    )
    # 定义一个测试函数，测试二元操作符在特定条件下的行为
    def test_binop_other(self, op, value, dtype, switch_numexpr_min_elements):
        # 定义不应测试的操作符和数据类型的组合
        skip = {
            (operator.truediv, "bool"),
            (operator.pow, "bool"),
            (operator.add, "bool"),
            (operator.mul, "bool"),
        }

        # 创建一个虚拟的元素对象
        elem = DummyElement(value, dtype)
        # 创建一个包含虚拟元素的 DataFrame，用于操作测试
        df = DataFrame({"A": [elem.value, elem.value]}, dtype=elem.dtype)

        # 定义无效操作符和数据类型的组合，需要引发异常
        invalid = {
            (operator.pow, "<M8[ns]"),
            (operator.mod, "<M8[ns]"),
            (operator.truediv, "<M8[ns]"),
            (operator.mul, "<M8[ns]"),
            (operator.add, "<M8[ns]"),
            (operator.pow, "<m8[ns]"),
            (operator.mul, "<m8[ns]"),
            (operator.sub, "bool"),
            (operator.mod, "complex128"),
        }

        # 如果操作符和数据类型组合在无效列表中
        if (op, dtype) in invalid:
            # 初始化警告信息为空
            warn = None
            # 检查特定条件下的消息内容
            if (dtype == "<M8[ns]" and op == operator.add) or (
                dtype == "<m8[ns]" and op == operator.mul
            ):
                msg = None
            # 对于复数类型的异常情况，设置特定的消息
            elif dtype == "complex128":
                msg = "ufunc 'remainder' not supported for the input types"
            # 对于减法操作的异常情况，设置特定的消息
            elif op is operator.sub:
                msg = "numpy boolean subtract, the `-` operator, is "
                # 在特定条件下，如果满足 Numexpr 使用且设置要求为 0，设置警告类型为 UserWarning
                if (
                    dtype == "bool"
                    and expr.USE_NUMEXPR
                    and switch_numexpr_min_elements == 0
                ):
                    warn = UserWarning
            else:
                # 对于其他无效操作符和数据类型组合，设置通用的错误消息
                msg = (
                    f"cannot perform __{op.__name__}__ with this "
                    "index type: (DatetimeArray|TimedeltaArray)"
                )

            # 断言引发 TypeError 异常，并匹配特定的异常消息
            with pytest.raises(TypeError, match=msg):
                # 在引发异常时，检查是否产生了特定类型的警告
                with tm.assert_produces_warning(warn, match="evaluating in Python"):
                    op(df, elem.value)

        # 如果操作符和数据类型组合在跳过列表中
        elif (op, dtype) in skip:
            # 对于加法和乘法操作，根据条件产生警告
            if op in [operator.add, operator.mul]:
                if expr.USE_NUMEXPR and switch_numexpr_min_elements == 0:
                    warn = UserWarning
                else:
                    warn = None
                # 断言产生特定类型的警告
                with tm.assert_produces_warning(warn, match="evaluating in Python"):
                    op(df, elem.value)

            else:
                # 对于其他操作符，断言引发 NotImplementedError 异常，并匹配特定的异常消息
                msg = "operator '.*' not implemented for .* dtypes"
                with pytest.raises(NotImplementedError, match=msg):
                    op(df, elem.value)

        else:
            # 对于其余情况，断言不会产生任何警告
            with tm.assert_produces_warning(None):
                # 执行操作，并检查返回结果的数据类型是否符合预期
                result = op(df, elem.value).dtypes
                expected = op(df, value).dtypes
            tm.assert_series_equal(result, expected)
    # 测试方法：测试当多级索引列包含不同数据类型时的算术操作
    def test_arithmetic_midx_cols_different_dtypes(self):
        # 标识：GitHub issue #49769
        # 创建第一个多级索引对象，包含两个整数系列
        midx = MultiIndex.from_arrays([Series([1, 2]), Series([3, 4])])
        # 创建第二个多级索引对象，包含一个字节整数和一个普通整数系列
        midx2 = MultiIndex.from_arrays([Series([1, 2], dtype="Int8"), Series([3, 4])])
        # 创建左侧数据框，使用第一个多级索引对象作为列名
        left = DataFrame([[1, 2], [3, 4]], columns=midx)
        # 创建右侧数据框，使用第二个多级索引对象作为列名
        right = DataFrame([[1, 2], [3, 4]], columns=midx2)
        # 执行左侧数据框与右侧数据框的减法操作
        result = left - right
        # 创建预期结果数据框，使用第一个多级索引对象作为列名
        expected = DataFrame([[0, 0], [0, 0]], columns=midx)
        # 使用测试框架中的断言方法验证结果是否符合预期
        tm.assert_frame_equal(result, expected)

    # 测试方法：测试当多级索引列包含不同数据类型且顺序不同时的算术操作
    def test_arithmetic_midx_cols_different_dtypes_different_order(self):
        # 标识：GitHub issue #49769
        # 创建第一个多级索引对象，包含两个整数系列
        midx = MultiIndex.from_arrays([Series([1, 2]), Series([3, 4])])
        # 创建第二个多级索引对象，包含一个字节整数和一个普通整数系列，但顺序相反
        midx2 = MultiIndex.from_arrays([Series([2, 1], dtype="Int8"), Series([4, 3])])
        # 创建左侧数据框，使用第一个多级索引对象作为列名
        left = DataFrame([[1, 2], [3, 4]], columns=midx)
        # 创建右侧数据框，使用第二个多级索引对象作为列名
        right = DataFrame([[1, 2], [3, 4]], columns=midx2)
        # 执行左侧数据框与右侧数据框的减法操作
        result = left - right
        # 创建预期结果数据框，使用第一个多级索引对象作为列名
        expected = DataFrame([[-1, 1], [-1, 1]], columns=midx)
        # 使用测试框架中的断言方法验证结果是否符合预期
        tm.assert_frame_equal(result, expected)
def test_frame_with_zero_len_series_corner_cases():
    # GH#28600
    # easy all-float case
    # 创建一个 DataFrame，包含两列，每列有三个随机生成的标准正态分布值，列名为"A"和"B"
    df = DataFrame(
        np.random.default_rng(2).standard_normal(6).reshape(3, 2), columns=["A", "B"]
    )
    # 创建一个空的 Series，数据类型为 np.float64
    ser = Series(dtype=np.float64)

    # 将 DataFrame 和 Series 相加，结果应该是一个与 df 大小相同，值为 NaN 的 DataFrame
    result = df + ser
    expected = DataFrame(df.values * np.nan, columns=df.columns)
    tm.assert_frame_equal(result, expected)

    # 检查是否会引发 ValueError 异常，匹配错误消息 "not aligned"
    with pytest.raises(ValueError, match="not aligned"):
        # 自动对齐比较已废弃 GH#36795，在 2.0 版本中强制执行
        df == ser

    # 非浮点数情况下，比较不应引发 TypeError 异常
    # 创建一个新的 DataFrame，其值的视图类型为 "M8[ns]"，列名与 df 相同
    df2 = DataFrame(df.values.view("M8[ns]"), columns=df.columns)
    with pytest.raises(ValueError, match="not aligned"):
        # 自动对齐比较已废弃
        df2 == ser


def test_zero_len_frame_with_series_corner_cases():
    # GH#28600
    # 创建一个空的 DataFrame，包含两列，数据类型为 np.float64，列名为 "A" 和 "B"
    df = DataFrame(columns=["A", "B"], dtype=np.float64)
    # 创建一个 Series，包含两个值 [1, 2]，索引分别为 "A" 和 "B"
    ser = Series([1, 2], index=["A", "B"])

    # 将 DataFrame 和 Series 相加，预期结果应为 df 本身
    result = df + ser
    expected = df
    tm.assert_frame_equal(result, expected)


def test_frame_single_columns_object_sum_axis_1():
    # GH 13758
    # 创建一个字典，包含一个键为 "One" 的 Series，其值为 ["A", 1.2, np.nan]
    data = {
        "One": Series(["A", 1.2, np.nan]),
    }
    # 根据 data 字典创建 DataFrame
    df = DataFrame(data)
    # 对 DataFrame 沿着 axis=1 方向求和，生成一个 Series
    result = df.sum(axis=1)
    # 预期的结果是一个包含 ["A", 1.2, 0] 的 Series
    expected = Series(["A", 1.2, 0])
    tm.assert_series_equal(result, expected)


# -------------------------------------------------------------------
# Unsorted
#  These arithmetic tests were previously in other files, eventually
#  should be parametrized and put into tests.arithmetic


class TestFrameArithmeticUnsorted:
    def test_frame_add_tz_mismatch_converts_to_utc(self):
        # 创建一个日期范围，每小时一个时间点，时区为 "US/Eastern"
        rng = pd.date_range("1/1/2011", periods=10, freq="h", tz="US/Eastern")
        # 创建一个 DataFrame，包含一个列 "a"，列值为 10 个随机生成的标准正态分布值，索引为 rng
        df = DataFrame(
            np.random.default_rng(2).standard_normal(len(rng)), index=rng, columns=["a"]
        )

        # 将 df 转换为 "Europe/Moscow" 时区
        df_moscow = df.tz_convert("Europe/Moscow")
        # 对 df 和 df_moscow 进行相加，结果的索引应该是时区为 UTC
        result = df + df_moscow
        assert result.index.tz is timezone.utc

        # 再次相加，顺序颠倒，结果的索引应该是时区为 UTC
        result = df_moscow + df
        assert result.index.tz is timezone.utc

    def test_align_frame(self):
        # 创建一个时期范围，每年一个时期
        rng = pd.period_range("1/1/2000", "1/1/2010", freq="Y")
        # 创建一个 DataFrame，包含一个形状为 (len(rng), 3) 的随机生成的标准正态分布值，索引为 rng
        ts = DataFrame(
            np.random.default_rng(2).standard_normal((len(rng), 3)), index=rng
        )

        # 将 ts 与 ts 的每隔两个取值相加，期望结果是 ts 与 ts 相加，但每隔两个值设置为 NaN
        result = ts + ts[::2]
        expected = ts + ts
        expected.iloc[1::2] = np.nan
        tm.assert_frame_equal(result, expected)

        # 取 ts 的每隔两个值，与随机排列后的 half 相加
        half = ts[::2]
        result = ts + half.take(np.random.default_rng(2).permutation(len(half)))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "op", [operator.add, operator.sub, operator.mul, operator.truediv]
    )
    # 定义测试方法，测试当填充值为 None 时的操作
    def test_operators_none_as_na(self, op):
        # 创建一个包含混合数据类型的 DataFrame 对象
        df = DataFrame(
            {"col1": [2, 5.0, 123, None], "col2": [1, 2, 3, 4]}, dtype=object
        )

        # 由于填充操作会将数据类型从 object 转换为 float64，所以期望结果的数据类型也是 object
        filled = df.fillna(np.nan)
        # 对 DataFrame 进行操作，并将结果与期望结果进行比较
        result = op(df, 3)
        expected = op(filled, 3).astype(object)
        expected[pd.isna(expected)] = np.nan
        tm.assert_frame_equal(result, expected)

        # 对两个相同结构的 DataFrame 进行操作，再次比较结果与期望结果
        result = op(df, df)
        expected = op(filled, filled).astype(object)
        expected[pd.isna(expected)] = np.nan
        tm.assert_frame_equal(result, expected)

        # 对原始 DataFrame 与填充后的 DataFrame 进行操作，比较结果与期望结果
        result = op(df, df.fillna(7))
        tm.assert_frame_equal(result, expected)

        # 对填充后的 DataFrame 与原始 DataFrame 进行操作，比较结果与期望结果
        result = op(df.fillna(7), df)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("op,res", [("__eq__", False), ("__ne__", True)])
    # TODO: not sure what's correct here.
    @pytest.mark.filterwarnings("ignore:elementwise:FutureWarning")
    # 测试当逻辑操作符应用于非有效类型时的行为
    def test_logical_typeerror_with_non_valid(self, op, res, float_frame):
        # 在这里我们比较浮点数和字符串的行为
        result = getattr(float_frame, op)("foo")
        assert bool(result.all().all()) is res

    @pytest.mark.parametrize("op", ["add", "sub", "mul", "div", "truediv"])
    # 测试二元操作的对齐行为
    def test_binary_ops_align(self, op):
        # 测试二元操作的对齐行为

        # GH 6681
        # 创建一个多重索引的 DataFrame 对象
        index = MultiIndex.from_product(
            [list("abc"), ["one", "two", "three"], [1, 2, 3]],
            names=["first", "second", "third"],
        )

        # 创建一个具有指定索引和列名的 DataFrame 对象，并排序索引
        df = DataFrame(
            np.arange(27 * 3).reshape(27, 3),
            index=index,
            columns=["value1", "value2", "value3"],
        ).sort_index()

        # 用于切片的索引对象
        idx = pd.IndexSlice
        # 获取操作符的函数引用
        opa = getattr(operator, op, None)
        if opa is None:
            return

        # 创建一个 Series 对象，作为操作的参数
        x = Series([1.0, 10.0, 100.0], [1, 2, 3])
        # 对 DataFrame 执行二元操作，指定对齐的级别和轴
        result = getattr(df, op)(x, level="third", axis=0)

        # 创建期望的结果 DataFrame，对每个项应用操作符
        expected = pd.concat(
            [opa(df.loc[idx[:, :, i], :], v) for i, v in x.items()]
        ).sort_index()
        tm.assert_frame_equal(result, expected)

        # 创建另一个 Series 对象，作为操作的参数
        x = Series([1.0, 10.0], ["two", "three"])
        # 对 DataFrame 执行二元操作，指定对齐的级别和轴
        result = getattr(df, op)(x, level="second", axis=0)

        # 创建期望的结果 DataFrame，对每个项应用操作符，并重新索引以匹配 df 的结构
        expected = (
            pd.concat([opa(df.loc[idx[:, i], :], v) for i, v in x.items()])
            .reindex_like(df)
            .sort_index()
        )
        tm.assert_frame_equal(result, expected)
    def test_binary_ops_align_series_dataframe(self):
        # 测试二进制操作对齐数据框架与系列的功能
        # GH9463 (数据框架与系列的对齐级别)

        # 创建一个多级索引
        midx = MultiIndex.from_product([["A", "B"], ["a", "b"]])
        # 创建一个数据框架，填充为整数 1，列使用上面定义的多级索引
        df = DataFrame(np.ones((2, 4), dtype="int64"), columns=midx)
        # 创建一个系列，包含键值对 {"a": 1, "b": 2}
        s = Series({"a": 1, "b": 2})

        # 复制数据框架 df，并设置其列的名称为 ["lvl0", "lvl1"]
        df2 = df.copy()
        df2.columns.names = ["lvl0", "lvl1"]
        # 复制系列 s，并设置其索引的名称为 "lvl1"
        s2 = s.copy()
        s2.index.name = "lvl1"

        # 不同情况下的整数/字符串级别名称:
        # 对数据框架 df 的列与系列 s 进行乘法操作，按 level=1 对齐
        res1 = df.mul(s, axis=1, level=1)
        # 对数据框架 df 的列与系列 s2 进行乘法操作，按 level=1 对齐
        res2 = df.mul(s2, axis=1, level=1)
        # 对数据框架 df2 的列与系列 s 进行乘法操作，按 level=1 对齐
        res3 = df2.mul(s, axis=1, level=1)
        # 对数据框架 df2 的列与系列 s2 进行乘法操作，按 level=1 对齐
        res4 = df2.mul(s2, axis=1, level=1)
        # 对数据框架 df2 的列与系列 s 进行乘法操作，按 level="lvl1" 对齐
        res5 = df2.mul(s, axis=1, level="lvl1")
        # 对数据框架 df2 的列与系列 s2 进行乘法操作，按 level="lvl1" 对齐
        res6 = df2.mul(s2, axis=1, level="lvl1")

        # 创建期望的数据框架 exp，填充为指定整数数组，列使用之前定义的多级索引
        exp = DataFrame(
            np.array([[1, 2, 1, 2], [1, 2, 1, 2]], dtype="int64"), columns=midx
        )

        # 对 res1 和 res2 进行循环，验证其与期望值 exp 的相等性
        for res in [res1, res2]:
            tm.assert_frame_equal(res, exp)

        # 设置期望数据框架 exp 的列名称为 ["lvl0", "lvl1"]
        exp.columns.names = ["lvl0", "lvl1"]
        # 对 res3、res4、res5、res6 进行循环，验证其与期望值 exp 的相等性
        for res in [res3, res4, res5, res6]:
            tm.assert_frame_equal(res, exp)

    def test_add_with_dti_mismatched_tzs(self):
        # 测试在不匹配的时区下，使用日期时间索引进行加法操作的功能

        # 创建一个基础的日期时间索引，时区为 UTC
        base = pd.DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"], tz="UTC")
        # 从基础索引中创建一个在 Asia/Tokyo 时区下的子索引（前两个元素）
        idx1 = base.tz_convert("Asia/Tokyo")[:2]
        # 从基础索引中创建一个在 US/Eastern 时区下的子索引（后两个元素）
        idx2 = base.tz_convert("US/Eastern")[1:]

        # 创建一个数据框架 df1，包含一个列 "A"，使用 idx1 作为索引
        df1 = DataFrame({"A": [1, 2]}, index=idx1)
        # 创建一个数据框架 df2，包含一个列 "A"，使用 idx2 作为索引
        df2 = DataFrame({"A": [1, 1]}, index=idx2)
        # 创建期望的数据框架 exp，包含一个列 "A"，使用 base 作为索引
        exp = DataFrame({"A": [np.nan, 3, np.nan]}, index=base)
        # 验证 df1 和 df2 的加法结果与期望值 exp 的相等性
        tm.assert_frame_equal(df1 + df2, exp)
    # 定义一个测试函数，用于测试数据框之间的加法操作
    def test_combineFrame(self, float_frame, mixed_float_frame, mixed_int_frame):
        # 对浮点数数据框进行重新索引，仅保留偶数行
        frame_copy = float_frame.reindex(float_frame.index[::2])

        # 删除复制数据框中的列"D"
        del frame_copy["D"]
        
        # 向列"C"的前5个值添加缺失值（NaN）
        frame_copy.loc[:frame_copy.index[4], "C"] = np.nan
        
        # 将原始数据框float_frame和修改后的数据框frame_copy进行加法操作
        added = float_frame + frame_copy

        # 获取加法结果中"A"列非NaN值的索引
        indexer = added["A"].dropna().index
        
        # 期望值为原始数据框中"A"列的每个值乘以2
        exp = (float_frame["A"] * 2).copy()
        
        # 使用测试工具比较加法结果中"A"列非NaN值的部分与期望值的对应部分
        tm.assert_series_equal(added["A"].dropna(), exp.loc[indexer])

        # 将期望值中未包含在索引器中的部分设为NaN
        exp.loc[~exp.index.isin(indexer)] = np.nan
        
        # 使用测试工具比较加法结果中"A"列的所有值与期望值的对应部分
        tm.assert_series_equal(added["A"], exp.loc[added["A"].index])

        # 断言加法结果中的"C"列在frame_copy索引处的前5个值是否全部为NaN
        assert np.isnan(added["C"].reindex(frame_copy.index)[:5]).all()

        # 断言加法结果中的"D"列是否全部为NaN
        assert np.isnan(added["D"]).all()

        # 对浮点数数据框自身进行加法操作
        self_added = float_frame + float_frame
        
        # 使用测试工具比较自加法结果的索引与原始数据框的索引
        tm.assert_index_equal(self_added.index, float_frame.index)

        # 将修改后的数据框frame_copy与浮点数数据框进行加法操作
        added_rev = frame_copy + float_frame
        
        # 断言加法结果中的"D"列是否全部为NaN
        assert np.isnan(added["D"]).all()
        assert np.isnan(added_rev["D"]).all()

        # 边界情况

        # 空数据框加法测试：浮点数数据框与空数据框相加
        plus_empty = float_frame + DataFrame()
        assert np.isnan(plus_empty.values).all()

        # 空数据框加法测试：空数据框与浮点数数据框相加
        empty_plus = DataFrame() + float_frame
        assert np.isnan(empty_plus.values).all()

        # 空数据框加法测试：两个空数据框相加
        empty_empty = DataFrame() + DataFrame()
        assert empty_empty.empty

        # 数据框列的顺序相反的加法测试：将浮点数数据框列的顺序颠倒后与原始数据框相加
        reverse = float_frame.reindex(columns=float_frame.columns[::-1])
        tm.assert_frame_equal(reverse + float_frame, float_frame * 2)

        # 混合数据类型的加法测试：浮点数数据框与混合浮点数数据框相加
        added = float_frame + mixed_float_frame
        _check_mixed_float(added, dtype="float64")
        
        # 混合数据类型的加法测试：混合浮点数数据框与浮点数数据框相加
        added = mixed_float_frame + float_frame
        _check_mixed_float(added, dtype="float64")

        # 混合数据类型的加法测试：两个混合浮点数数据框相加
        added = mixed_float_frame + mixed_float_frame
        _check_mixed_float(added, dtype={"C": None})

        # 浮点数数据框与混合整数数据框相加的测试
        added = float_frame + mixed_int_frame
        _check_mixed_float(added, dtype="float64")
    # 定义一个测试方法，用于测试混合数据帧和浮点数据帧之间的操作
    def test_combine_series(self, float_frame, mixed_float_frame, mixed_int_frame):
        # 从浮点数据帧中取出第一个行索引对应的序列
        series = float_frame.xs(float_frame.index[0])

        # 将浮点数据帧和序列相加
        added = float_frame + series

        # 遍历相加后的结果，检查每个项是否与预期的相加结果一致
        for key, s in added.items():
            tm.assert_series_equal(s, float_frame[key] + series[key])

        # 将序列转换为字典，并在其中添加一个新的键值对
        larger_series = series.to_dict()
        larger_series["E"] = 1
        larger_series = Series(larger_series)
        
        # 将浮点数据帧和更新后的较大序列相加
        larger_added = float_frame + larger_series

        # 再次遍历浮点数据帧的每个项，检查相加后的结果是否符合预期
        for key, s in float_frame.items():
            tm.assert_series_equal(larger_added[key], s + series[key])

        # 断言新添加的键 "E" 是否在结果中，并且其值是否全为 NaN
        assert "E" in larger_added
        assert np.isnan(larger_added["E"]).all()

        # 对于混合浮点数据帧和序列的相加操作，验证结果的数据类型是否与序列一致
        added = mixed_float_frame + series
        assert np.all(added.dtypes == series.dtype)

        # 对混合浮点数据帧和将序列转换为特定浮点类型后的相加操作，进行类型检查
        added = mixed_float_frame + series.astype("float32")
        _check_mixed_float(added, dtype={"C": None})
        added = mixed_float_frame + series.astype("float16")
        _check_mixed_float(added, dtype={"C": None})

        # 将整数混合数据帧和序列乘以100后转换为整型相加，进行类型检查
        added = mixed_int_frame + (100 * series).astype("int64")
        _check_mixed_int(
            added, dtype={"A": "int64", "B": "float64", "C": "int64", "D": "int64"}
        )
        added = mixed_int_frame + (100 * series).astype("int32")
        _check_mixed_int(
            added, dtype={"A": "int32", "B": "float64", "C": "int32", "D": "int64"}
        )
    # 定义一个测试方法，用于测试时间序列的合并功能，接受一个日期时间框架作为参数
    def test_combine_timeseries(self, datetime_frame):
        # 从日期时间框架中获取名为"A"的时间序列
        ts = datetime_frame["A"]

        # 创建一个新的日期时间框架，将ts时间序列与日期时间框架的每一列相加，并指定在索引轴上进行操作
        added = datetime_frame.add(ts, axis="index")

        # 遍历日期时间框架的每一列及其对应的时间序列相加的结果
        for key, col in datetime_frame.items():
            # 计算当前列与ts时间序列相加后的结果
            result = col + ts
            # 使用断言验证添加后的时间序列与预期结果相等，忽略列名的检查
            tm.assert_series_equal(added[key], result, check_names=False)
            # 使用断言验证添加后的时间序列的名称与列名相同
            assert added[key].name == key
            # 如果当前列的名称与ts时间序列的名称相同，使用断言验证结果的名称为"A"，否则结果的名称应为None
            if col.name == ts.name:
                assert result.name == "A"
            else:
                assert result.name is None

        # 创建一个较小的日期时间框架，包含比原框架少5行数据
        smaller_frame = datetime_frame[:-5]
        # 将ts时间序列与较小的日期时间框架相加
        smaller_added = smaller_frame.add(ts, axis="index")

        # 使用断言验证较小的日期时间框架的索引与原始日期时间框架的索引相等
        tm.assert_index_equal(smaller_added.index, datetime_frame.index)

        # 创建一个较小的ts时间序列，包含比原序列少5个数据点
        smaller_ts = ts[:-5]
        # 将较小的ts时间序列与原始日期时间框架相加
        smaller_added2 = datetime_frame.add(smaller_ts, axis="index")
        # 使用断言验证两种不同方式计算的结果相等
        tm.assert_frame_equal(smaller_added, smaller_added2)

        # 当ts时间序列长度为0时，相加的结果应该全为NaN
        result = datetime_frame.add(ts[:0], axis="index")
        expected = DataFrame(
            np.nan, index=datetime_frame.index, columns=datetime_frame.columns
        )
        # 使用断言验证相加的结果与预期的全NaN的DataFrame相等
        tm.assert_frame_equal(result, expected)

        # 当日期时间框架为空时，相加的结果应该全为NaN
        result = datetime_frame[:0].add(ts, axis="index")
        expected = DataFrame(
            np.nan, index=datetime_frame.index, columns=datetime_frame.columns
        )
        # 使用断言验证相加的结果与预期的全NaN的DataFrame相等
        tm.assert_frame_equal(result, expected)

        # 当日期时间框架为空但具有非空索引时，相乘的结果应该长度与ts时间序列相等
        frame = datetime_frame[:1].reindex(columns=[])
        result = frame.mul(ts, axis="index")
        # 使用断言验证相乘的结果的长度与ts时间序列的长度相等
        assert len(result) == len(ts)

    # 定义一个测试方法，用于测试混合类型浮点数框架的乘法运算功能，接受两种不同类型的浮点数框架作为参数
    def test_combineFunc(self, float_frame, mixed_float_frame):
        # 对浮点数框架中的每个值乘以2，返回结果
        result = float_frame * 2
        # 使用断言验证乘法运算的结果与期望结果相等
        tm.assert_numpy_array_equal(result.values, float_frame.values * 2)

        # 对混合类型浮点数框架中的每个值乘以2，返回结果
        result = mixed_float_frame * 2
        # 遍历乘法运算的结果中的每列，并使用断言验证每列的值与预期值相等
        for c, s in result.items():
            tm.assert_numpy_array_equal(s.values, mixed_float_frame[c].values * 2)
        # 对乘法运算结果进行类型检查
        _check_mixed_float(result, dtype={"C": None})

        # 对空DataFrame进行乘法运算，预期结果应保留索引并且列数为0
        result = DataFrame() * 2
        # 使用断言验证乘法运算的结果与预期的空DataFrame的索引相等
        assert result.index.equals(DataFrame().index)
        # 使用断言验证乘法运算的结果的列数为0
        assert len(result.columns) == 0

    # 使用pytest.mark.parametrize装饰器标记的测试方法，测试多个比较运算符的功能
    @pytest.mark.parametrize(
        "func",
        [operator.eq, operator.ne, operator.lt, operator.gt, operator.ge, operator.le],
    )
    # 测试比较功能的方法，接受三个参数：simple_frame、float_frame、func
    def test_comparisons(self, simple_frame, float_frame, func):
        # 创建一个包含随机标准正态分布数据的 DataFrame df1
        df1 = DataFrame(
            np.random.default_rng(2).standard_normal((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=pd.date_range("2000-01-01", periods=30, freq="B"),
        )
        # 复制 df1 生成 df2
        df2 = df1.copy()

        # 从 simple_frame 中选择标签为 "a" 的行，存储在变量 row 中
        row = simple_frame.xs("a")
        # 创建一个全为 1 的多维数组 ndim_5，与 df1 的形状相同
        ndim_5 = np.ones(df1.shape + (1, 1, 1))

        # 调用 func 函数对 df1 和 df2 进行比较，存储结果在 result 中
        result = func(df1, df2)
        # 断言 result 的值与 func 应用于 df1.values 和 df2.values 后的结果相等
        tm.assert_numpy_array_equal(result.values, func(df1.values, df2.values))

        # 准备错误信息 msg，用于检查 func 对 ndim_5 的调用是否引发 ValueError
        msg = (
            "Unable to coerce to Series/DataFrame, "
            "dimension must be <= 2: (30, 4, 1, 1, 1)"
        )
        # 使用 pytest.raises 检查 func 对 ndim_5 的调用是否引发预期的 ValueError，并匹配 msg
        with pytest.raises(ValueError, match=re.escape(msg)):
            func(df1, ndim_5)

        # 调用 func 函数对 simple_frame 和 row 进行比较，存储结果在 result2 中
        result2 = func(simple_frame, row)
        # 断言 result2 的值与 func 应用于 simple_frame.values 和 row.values 后的结果相等
        tm.assert_numpy_array_equal(
            result2.values, func(simple_frame.values, row.values)
        )

        # 调用 func 函数对 float_frame 和数值 0 进行比较，存储结果在 result3 中
        result3 = func(float_frame, 0)
        # 断言 result3 的值与 func 应用于 float_frame.values 和数值 0 后的结果相等
        tm.assert_numpy_array_equal(result3.values, func(float_frame.values, 0))

        # 准备错误信息 msg，用于检查 func 对 simple_frame 和 simple_frame[:2] 的调用是否引发 ValueError
        msg = (
            r"Can only compare identically-labeled \(both index and columns\) "
            "DataFrame objects"
        )
        # 使用 pytest.raises 检查 func 对 simple_frame 和 simple_frame[:2] 的调用是否引发预期的 ValueError，并匹配 msg
        with pytest.raises(ValueError, match=msg):
            func(simple_frame, simple_frame[:2])
    def test_boolean_comparison(self):
        # GH 4576
        # boolean comparisons with a tuple/list give unexpected results
        
        # 创建一个 DataFrame 对象，包含 0 到 5 的整数，形状为 (3, 2)
        df = DataFrame(np.arange(6).reshape((3, 2)))
        
        # 创建一个包含 [2, 2] 的 NumPy 数组
        b = np.array([2, 2])
        
        # 将 b 转换为至少二维的数组
        b_r = np.atleast_2d([2, 2])
        
        # 将 b_r 进行转置，形成 b_c
        b_c = b_r.T
        
        # 创建一个包含 [2, 2, 2] 的列表
        lst = [2, 2, 2]
        
        # 将 lst 转换为元组
        tup = tuple(lst)
        
        # gt 比较（大于）
        expected = DataFrame([[False, False], [False, True], [True, True]])
        
        # 对 DataFrame df 中的元素与 b 进行大于比较，得到布尔值 DataFrame
        result = df > b
        tm.assert_frame_equal(result, expected)
        
        # 对 DataFrame df 中的元素转换为 NumPy 数组后，与 b 进行大于比较
        result = df.values > b
        tm.assert_numpy_array_equal(result, expected.values)
        
        # 定义错误消息
        msg1d = "Unable to coerce to Series, length must be 2: given 3"
        msg2d = "Unable to coerce to DataFrame, shape must be"
        msg2db = "operands could not be broadcast together with shapes"
        
        # 检查是否引发 ValueError 异常，匹配 msg1d 消息
        with pytest.raises(ValueError, match=msg1d):
            df > lst
        
        # 检查是否引发 ValueError 异常，匹配 msg1d 消息
        with pytest.raises(ValueError, match=msg1d):
            df > tup
        
        # broadcasts like ndarray (GH#23000)
        # 使用类似 NumPy 数组的广播机制进行比较
        result = df > b_r
        tm.assert_frame_equal(result, expected)
        
        # 使用类似 NumPy 数组的广播机制进行比较
        result = df.values > b_r
        tm.assert_numpy_array_equal(result, expected.values)
        
        # 检查是否引发 ValueError 异常，匹配 msg2d 消息
        with pytest.raises(ValueError, match=msg2d):
            df > b_c
        
        # 检查是否引发 ValueError 异常，匹配 msg2db 消息
        with pytest.raises(ValueError, match=msg2db):
            df.values > b_c
        
        # == 比较（等于）
        expected = DataFrame([[False, False], [True, False], [False, False]])
        
        # 对 DataFrame df 中的元素与 b 进行等于比较，得到布尔值 DataFrame
        result = df == b
        tm.assert_frame_equal(result, expected)
        
        # 检查是否引发 ValueError 异常，匹配 msg1d 消息
        with pytest.raises(ValueError, match=msg1d):
            df == lst
        
        # 检查是否引发 ValueError 异常，匹配 msg1d 消息
        with pytest.raises(ValueError, match=msg1d):
            df == tup
        
        # broadcasts like ndarray (GH#23000)
        # 使用类似 NumPy 数组的广播机制进行比较
        result = df == b_r
        tm.assert_frame_equal(result, expected)
        
        # 使用类似 NumPy 数组的广播机制进行比较
        result = df.values == b_r
        tm.assert_numpy_array_equal(result, expected.values)
        
        # 检查是否引发 ValueError 异常，匹配 msg2d 消息
        with pytest.raises(ValueError, match=msg2d):
            df == b_c
        
        # 检查 df.values.shape 与 b_c.shape 是否不相等
        assert df.values.shape != b_c.shape
        
        # with alignment
        # 创建一个具有指定列和索引的 DataFrame 对象
        df = DataFrame(
            np.arange(6).reshape((3, 2)), columns=list("AB"), index=list("abc")
        )
        
        # 将期望 DataFrame 对象的索引和列与 df 对象相匹配
        expected.index = df.index
        expected.columns = df.columns
        
        # 检查是否引发 ValueError 异常，匹配 msg1d 消息
        with pytest.raises(ValueError, match=msg1d):
            df == lst
        
        # 检查是否引发 ValueError 异常，匹配 msg1d 消息
        with pytest.raises(ValueError, match=msg1d):
            df == tup
    def test_inplace_ops_alignment(self):
        # 定义测试方法：测试原地操作的对齐性
        # GH 8511

        # 定义列名列表
        columns = list("abcdefg")
        # 创建原始数据框 X_orig，包含一定范围内的数据，行数为10，列数为len(columns)
        X_orig = DataFrame(
            np.arange(10 * len(columns)).reshape(-1, len(columns)),
            columns=columns,
            index=range(10),
        )
        # 根据 X_orig 的部分列生成 Z 数据框，乘以100，并复制
        Z = 100 * X_orig.iloc[:, 1:-1].copy()
        # 定义 block1 和 subs 列表
        block1 = list("bedcf")
        subs = list("bcdef")

        # add
        # 复制 X_orig 为 X
        X = X_orig.copy()
        # 计算结果 result1，对 X 的 block1 列加上 Z，并按 subs 列重新索引
        result1 = (X[block1] + Z).reindex(columns=subs)

        # 在 X 上原地操作，将 block1 列加上 Z
        X[block1] += Z
        # 计算结果 result2，按 subs 列重新索引 X
        result2 = X.reindex(columns=subs)

        # 复制 X_orig 为 X
        X = X_orig.copy()
        # 计算结果 result3，对 X 的 block1 列加上 Z 的 block1 列，并按 subs 列重新索引
        result3 = (X[block1] + Z[block1]).reindex(columns=subs)

        # 在 X 上原地操作，将 block1 列加上 Z 的 block1 列
        X[block1] += Z[block1]
        # 计算结果 result4，按 subs 列重新索引 X
        result4 = X.reindex(columns=subs)

        # 使用测试框架验证 result1 与 result2 相等
        tm.assert_frame_equal(result1, result2)
        # 使用测试框架验证 result1 与 result3 相等
        tm.assert_frame_equal(result1, result3)
        # 使用测试框架验证 result1 与 result4 相等
        tm.assert_frame_equal(result1, result4)

        # sub
        # 复制 X_orig 为 X
        X = X_orig.copy()
        # 计算结果 result1，对 X 的 block1 列减去 Z，并按 subs 列重新索引
        result1 = (X[block1] - Z).reindex(columns=subs)

        # 在 X 上原地操作，将 block1 列减去 Z
        X[block1] -= Z
        # 计算结果 result2，按 subs 列重新索引 X
        result2 = X.reindex(columns=subs)

        # 复制 X_orig 为 X
        X = X_orig.copy()
        # 计算结果 result3，对 X 的 block1 列减去 Z 的 block1 列，并按 subs 列重新索引
        result3 = (X[block1] - Z[block1]).reindex(columns=subs)

        # 在 X 上原地操作，将 block1 列减去 Z 的 block1 列
        X[block1] -= Z[block1]
        # 计算结果 result4，按 subs 列重新索引 X
        result4 = X.reindex(columns=subs)

        # 使用测试框架验证 result1 与 result2 相等
        tm.assert_frame_equal(result1, result2)
        # 使用测试框架验证 result1 与 result3 相等
        tm.assert_frame_equal(result1, result3)
        # 使用测试框架验证 result1 与 result4 相等
        tm.assert_frame_equal(result1, result4)
    def test_inplace_ops_identity(self):
        # GH 5104
        # 确保我们确实在改变对象
        # 创建包含整数的 Series 对象
        s_orig = Series([1, 2, 3])
        # 创建包含随机整数的 DataFrame 对象
        df_orig = DataFrame(
            np.random.default_rng(2).integers(0, 5, size=10).reshape(-1, 5)
        )

        # 没有 dtype 改变
        # 复制 Series 对象 s_orig
        s = s_orig.copy()
        # 将 s 赋值给 s2
        s2 = s
        # 在 s 上执行原地加法操作
        s += 1
        # 断言 s 和 s2 相等
        tm.assert_series_equal(s, s2)
        # 断言 s_orig 加 1 后与 s 相等
        tm.assert_series_equal(s_orig + 1, s)
        # 断言 s 和 s2 是同一个对象
        assert s is s2
        # 断言 s 和 s2 的底层数据管理器相同
        assert s._mgr is s2._mgr

        # 复制 DataFrame 对象 df_orig
        df = df_orig.copy()
        # 将 df 赋值给 df2
        df2 = df
        # 在 df 上执行原地加法操作
        df += 1
        # 断言 df 和 df2 相等
        tm.assert_frame_equal(df, df2)
        # 断言 df_orig 加 1 后与 df 相等
        tm.assert_frame_equal(df_orig + 1, df)
        # 断言 df 和 df2 是同一个对象
        assert df is df2
        # 断言 df 和 df2 的底层数据管理器相同
        assert df._mgr is df2._mgr

        # dtype 改变
        # 复制 Series 对象 s_orig
        s = s_orig.copy()
        # 将 s 赋值给 s2
        s2 = s
        # 在 s 上执行原地加法操作（浮点数）
        s += 1.5
        # 断言 s 和 s2 相等
        tm.assert_series_equal(s, s2)
        # 断言 s_orig 加 1.5 后与 s 相等
        tm.assert_series_equal(s_orig + 1.5, s)

        # 复制 DataFrame 对象 df_orig
        df = df_orig.copy()
        # 将 df 赋值给 df2
        df2 = df
        # 在 df 上执行原地加法操作（浮点数）
        df += 1.5
        # 断言 df 和 df2 相等
        tm.assert_frame_equal(df, df2)
        # 断言 df_orig 加 1.5 后与 df 相等
        tm.assert_frame_equal(df_orig + 1.5, df)
        # 断言 df 和 df2 是同一个对象
        assert df is df2
        # 断言 df 和 df2 的底层数据管理器相同
        assert df._mgr is df2._mgr

        # 混合 dtype
        # 创建包含随机整数的数组 arr
        arr = np.random.default_rng(2).integers(0, 10, size=5)
        # 根据 arr 和 "foo" 创建 DataFrame 对象 df_orig
        df_orig = DataFrame({"A": arr.copy(), "B": "foo"})
        # 复制 DataFrame 对象 df_orig
        df = df_orig.copy()
        # 将 df 赋值给 df2
        df2 = df
        # 在 df 的 "A" 列上执行原地加法操作
        df["A"] += 1
        # 创建预期的 DataFrame 对象 expected
        expected = DataFrame({"A": arr.copy() + 1, "B": "foo"})
        # 断言 df 和 expected 相等
        tm.assert_frame_equal(df, expected)
        # 断言 df2 和 expected 相等
        tm.assert_frame_equal(df2, expected)
        # 断言 df 的底层数据管理器与 df2 的相同
        assert df._mgr is df2._mgr

        # 复制 DataFrame 对象 df_orig
        df = df_orig.copy()
        # 将 df 赋值给 df2
        df2 = df
        # 在 df 的 "A" 列上执行原地加法操作（浮点数）
        df["A"] += 1.5
        # 创建预期的 DataFrame 对象 expected
        expected = DataFrame({"A": arr.copy() + 1.5, "B": "foo"})
        # 断言 df 和 expected 相等
        tm.assert_frame_equal(df, expected)
        # 断言 df2 和 expected 相等
        tm.assert_frame_equal(df2, expected)
        # 断言 df 的底层数据管理器与 df2 的相同
        assert df._mgr is df2._mgr

    @pytest.mark.parametrize(
        "op",
        [
            "add",
            "and",
            pytest.param(
                "div",
                marks=pytest.mark.xfail(
                    raises=AttributeError, reason="__idiv__ not implemented"
                ),
            ),
            "floordiv",
            "mod",
            "mul",
            "or",
            "pow",
            "sub",
            "truediv",
            "xor",
        ],
    )
    def test_inplace_ops_identity2(self, op):
        # 创建包含两列数据的 DataFrame 对象 df
        df = DataFrame({"a": [1.0, 2.0, 3.0], "b": [1, 2, 3]})

        # 操作数设置为 2
        operand = 2
        # 如果操作是布尔运算，则不能使用浮点数
        if op in ("and", "or", "xor"):
            # 将 "a" 列设置为布尔值
            df["a"] = [True, False, True]

        # 复制 DataFrame 对象 df
        df_copy = df.copy()
        # 构建原地操作方法名
        iop = f"__i{op}__"
        # 构建操作方法名
        op = f"__{op}__"

        # 调用原地操作方法
        getattr(df, iop)(operand)
        # 获取预期的结果
        expected = getattr(df_copy, op)(operand)
        # 断言 df 和 expected 相等
        tm.assert_frame_equal(df, expected)
        # 断言 df 的 id 与预期相同
        expected = id(df)
        assert id(df) == expected

    @pytest.mark.parametrize(
        "val",
        [
            [1, 2, 3],
            (1, 2, 3),
            np.array([1, 2, 3], dtype=np.int64),
            range(1, 4),
        ],
    )
    # 测试非 Pandas 对象的对齐功能，使用指定的值进行测试
    def test_alignment_non_pandas(self, val):
        # 创建索引和列名称列表
        index = ["A", "B", "C"]
        columns = ["X", "Y", "Z"]
        # 创建一个随机数填充的 DataFrame 对象
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            index=index,
            columns=columns,
        )

        # 获取 DataFrame 类的 _align_for_op 静态方法
        align = DataFrame._align_for_op

        # 预期的结果 DataFrame，每列使用给定的值 val，并保持索引不变
        expected = DataFrame({"X": val, "Y": val, "Z": val}, index=df.index)
        # 断言 align 方法对 df 执行 axis=0 操作后的结果与预期相等
        tm.assert_frame_equal(align(df, val, axis=0)[1], expected)

        # 另一组预期结果 DataFrame，每行使用 [1, 2, 3]，[2, 2, 2]，[3, 3, 3]，并保持索引不变
        expected = DataFrame(
            {"X": [1, 1, 1], "Y": [2, 2, 2], "Z": [3, 3, 3]}, index=df.index
        )
        # 断言 align 方法对 df 执行 axis=1 操作后的结果与预期相等
        tm.assert_frame_equal(align(df, val, axis=1)[1], expected)

    @pytest.mark.parametrize("val", [[1, 2], (1, 2), np.array([1, 2]), range(1, 3)])
    # 测试非 Pandas 对象的对齐功能，使用不同长度的值进行测试
    def test_alignment_non_pandas_length_mismatch(self, val):
        # 创建索引和列名称列表
        index = ["A", "B", "C"]
        columns = ["X", "Y", "Z"]
        # 创建一个随机数填充的 DataFrame 对象
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            index=index,
            columns=columns,
        )

        # 获取 DataFrame 类的 _align_for_op 静态方法
        align = DataFrame._align_for_op
        # 长度不匹配的错误信息
        msg = "Unable to coerce to Series, length must be 3: given 2"
        # 使用 pytest 断言检查在执行 axis=0 操作时是否引发 ValueError，并匹配错误信息
        with pytest.raises(ValueError, match=msg):
            align(df, val, axis=0)

        with pytest.raises(ValueError, match=msg):
            align(df, val, axis=1)

    # 测试非 Pandas 对象的对齐功能，使用包含索引和列的值进行测试
    def test_alignment_non_pandas_index_columns(self):
        # 创建索引和列名称列表
        index = ["A", "B", "C"]
        columns = ["X", "Y", "Z"]
        # 创建一个随机数填充的 DataFrame 对象
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            index=index,
            columns=columns,
        )

        # 获取 DataFrame 类的 _align_for_op 静态方法
        align = DataFrame._align_for_op
        # 创建一个 3x3 的二维数组作为值
        val = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # 断言 align 方法对 df 执行 axis=0 操作后的结果与预期相等
        tm.assert_frame_equal(
            align(df, val, axis=0)[1],
            DataFrame(val, index=df.index, columns=df.columns),
        )
        # 断言 align 方法对 df 执行 axis=1 操作后的结果与预期相等
        tm.assert_frame_equal(
            align(df, val, axis=1)[1],
            DataFrame(val, index=df.index, columns=df.columns),
        )

        # 形状不匹配的错误信息
        msg = "Unable to coerce to DataFrame, shape must be"
        # 创建一个 2x3 的二维数组作为值
        val = np.array([[1, 2, 3], [4, 5, 6]])
        # 使用 pytest 断言检查在执行 axis=0 操作时是否引发 ValueError，并匹配错误信息
        with pytest.raises(ValueError, match=msg):
            align(df, val, axis=0)

        with pytest.raises(ValueError, match=msg):
            align(df, val, axis=1)

        # 创建一个 3x3x3 的三维数组作为值
        val = np.zeros((3, 3, 3))
        # 创建一个与维度不匹配的错误信息
        msg = re.escape(
            "Unable to coerce to Series/DataFrame, dimension must be <= 2: (3, 3, 3)"
        )
        # 使用 pytest 断言检查在执行 axis=0 和 axis=1 操作时是否引发 ValueError，并匹配错误信息
        with pytest.raises(ValueError, match=msg):
            align(df, val, axis=0)
        with pytest.raises(ValueError, match=msg):
            align(df, val, axis=1)

    # 测试不产生警告的情况，使用所有算术运算符
    def test_no_warning(self, all_arithmetic_operators):
        # 创建一个包含浮点数和 None 的 DataFrame 对象
        df = DataFrame({"A": [0.0, 0.0], "B": [0.0, None]})
        # 获取 DataFrame 列 'B'
        b = df["B"]
        # 使用 pytest 断言检查执行所有算术运算符时是否不会产生警告
        with tm.assert_produces_warning(None):
            getattr(df, all_arithmetic_operators)(b)
    # 测试 DataFrame 对象的 dunder 方法是否能正确处理二进制操作
    def test_dunder_methods_binary(self, all_arithmetic_operators):
        # GH#??? frame.__foo__ should only accept one argument
        # 创建一个 DataFrame 对象，包含两列 "A" 和 "B"，"B" 列包括 None 值
        df = DataFrame({"A": [0.0, 0.0], "B": [0.0, None]})
        # 选择 DataFrame 中的 "B" 列
        b = df["B"]
        # 使用 pytest 检查调用 dunder 方法时是否抛出预期的 TypeError 异常
        with pytest.raises(TypeError, match="takes 2 positional arguments"):
            # 调用 DataFrame 对象的指定 dunder 方法，传递 b 和 0 作为参数
            getattr(df, all_arithmetic_operators)(b, 0)

    # 测试 DataFrame 对象的对齐和填充功能是否存在 Bug
    def test_align_int_fill_bug(self):
        # GH#910
        # 创建一个 10x10 的浮点数数组 X
        X = np.arange(10 * 10, dtype="float64").reshape(10, 10)
        # 创建一个全为 1 的 10x1 整数数组 Y
        Y = np.ones((10, 1), dtype=int)

        # 创建一个 DataFrame 对象 df1，使用数组 X 初始化
        df1 = DataFrame(X)
        # 将 Y 数组的压缩形式作为新列 "0.X" 添加到 df1 中
        df1["0.X"] = Y.squeeze()

        # 将 df1 转换为浮点数类型，并赋给 df2
        df2 = df1.astype(float)

        # 计算 df1 减去其均值后的结果
        result = df1 - df1.mean()
        # 计算 df2 减去其均值后的期望结果
        expected = df2 - df2.mean()

        # 使用 pytest 中的 assert_frame_equal 函数检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
def test_pow_with_realignment():
    # GH#32685 pow has special semantics for operating with null values
    # 创建一个包含"A"列的DataFrame，包含值[0, 1, 2]
    left = DataFrame({"A": [0, 1, 2]})
    # 创建一个空的DataFrame，索引为[0, 1, 2]
    right = DataFrame(index=[0, 1, 2])

    # 对left中的每个元素进行right对齐的幂运算
    result = left**right
    # 创建一个期望的DataFrame，包含"A"列和对应的预期结果[NaN, 1.0, NaN]
    expected = DataFrame({"A": [np.nan, 1.0, np.nan]})
    # 断言结果DataFrame与期望DataFrame相等
    tm.assert_frame_equal(result, expected)


def test_dataframe_series_extension_dtypes():
    # https://github.com/pandas-dev/pandas/issues/34311
    # 创建一个包含10行3列随机整数的DataFrame，列名为"a", "b", "c"
    df = DataFrame(
        np.random.default_rng(2).integers(0, 100, (10, 3)), columns=["a", "b", "c"]
    )
    # 创建一个包含索引为["a", "b", "c"]的Series，值为[1, 2, 3]
    ser = Series([1, 2, 3], index=["a", "b", "c"])

    # 计算期望的结果，将df和ser的整数数组相加，并转换为Int64类型
    expected = df.to_numpy("int64") + ser.to_numpy("int64").reshape(-1, 3)
    expected = DataFrame(expected, columns=df.columns, dtype="Int64")

    # 将df转换为Int64类型
    df_ea = df.astype("Int64")
    # 计算结果DataFrame与期望DataFrame相加的结果
    result = df_ea + ser
    # 断言结果DataFrame与期望DataFrame相等
    tm.assert_frame_equal(result, expected)
    # 将ser转换为Int64类型，再次计算结果DataFrame与期望DataFrame相等的断言
    result = df_ea + ser.astype("Int64")
    tm.assert_frame_equal(result, expected)


def test_dataframe_blockwise_slicelike():
    # GH#34367
    # 创建一个包含100行10列随机整数的DataFrame
    arr = np.random.default_rng(2).integers(0, 1000, (100, 10))
    df1 = DataFrame(arr)
    # 将df1的第1、3、7列显式转换为float类型，并将首行对应位置设置为NaN
    df2 = df1.copy().astype({1: "float", 3: "float", 7: "float"})
    df2.iloc[0, [1, 3, 7]] = np.nan

    # 将df1的第5列显式转换为float类型，并将首行对应位置设置为NaN
    df3 = df1.copy().astype({5: "float"})
    df3.iloc[0, [5]] = np.nan

    # 将df1的第2到4列显式转换为float类型，并将首行对应位置设置为NaN
    df4 = df1.copy().astype({2: "float", 3: "float", 4: "float"})
    df4.iloc[0, np.arange(2, 5)] = np.nan
    # 将df1的第4到6列显式转换为float类型，并将首行对应位置设置为NaN
    df5 = df1.copy().astype({4: "float", 5: "float", 6: "float"})
    df5.iloc[0, np.arange(4, 7)] = np.nan

    # 对于每对(left, right)，执行块状的DataFrame加法操作
    for left, right in [(df1, df2), (df2, df3), (df4, df5)]:
        res = left + right

        # 创建期望的DataFrame，包含左右DataFrame对应列的加法结果
        expected = DataFrame({i: left[i] + right[i] for i in left.columns})
        # 断言结果DataFrame与期望DataFrame相等
        tm.assert_frame_equal(res, expected)


@pytest.mark.parametrize(
    "df, col_dtype",
    [
        # 创建一个包含2行2列浮点数的DataFrame，列名为"a", "b"
        (DataFrame([[1.0, 2.0], [4.0, 5.0]], columns=list("ab")), "float64"),
        # 创建一个包含2行2列数据的DataFrame，列名为"a", "b"，其中第二列为字符串
        (
            DataFrame([[1.0, "b"], [4.0, "b"]], columns=list("ab")).astype(
                {"b": object}
            ),
            "object",
        ),
    ],
)
def test_dataframe_operation_with_non_numeric_types(df, col_dtype):
    # GH #22663
    # 创建一个期望的DataFrame，包含对应列的预期结果
    expected = DataFrame([[0.0, np.nan], [3.0, np.nan]], columns=list("ab"))
    expected = expected.astype({"b": col_dtype})
    # 对df和以列表形式的Series执行加法操作
    result = df + Series([-1.0], index=list("a"))
    # 断言结果DataFrame与期望DataFrame相等
    tm.assert_frame_equal(result, expected)


def test_arith_reindex_with_duplicates():
    # https://github.com/pandas-dev/pandas/issues/35194
    # 创建一个包含"second"列的DataFrame，值为[[0]]
    df1 = DataFrame(data=[[0]], columns=["second"])
    # 创建一个包含"first", "second", "second"列的DataFrame，值为[[0, 0, 0]]
    df2 = DataFrame(data=[[0, 0, 0]], columns=["first", "second", "second"])
    # 执行DataFrame之间的算术运算
    result = df1 + df2
    # 创建一个期望的DataFrame，包含对应列的预期结果
    expected = DataFrame([[np.nan, 0, 0]], columns=["first", "second", "second"])
    # 断言结果DataFrame与期望DataFrame相等
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize("to_add", [[Series([1, 1])], [Series([1, 1]), Series([1, 1])]])
def test_arith_list_of_arraylike_raise(to_add):
    # GH 36702. Raise when trying to add list of array-like to DataFrame

    # 创建一个包含两列 'x' 和 'y' 的 DataFrame
    df = DataFrame({"x": [1, 2], "y": [1, 2]})

    # 准备错误信息，说明无法将 array-like 列表强制转换为 Series/DataFrame
    msg = f"Unable to coerce list of {type(to_add[0])} to Series/DataFrame"

    # 断言在执行 df + to_add 操作时会抛出 ValueError 异常，并匹配错误信息 msg
    with pytest.raises(ValueError, match=msg):
        df + to_add
    # 同样断言在执行 to_add + df 操作时会抛出 ValueError 异常，并匹配错误信息 msg
    with pytest.raises(ValueError, match=msg):
        to_add + df


def test_inplace_arithmetic_series_update():
    # https://github.com/pandas-dev/pandas/issues/36373

    # 创建一个包含列 'A' 的 DataFrame
    df = DataFrame({"A": [1, 2, 3]})
    # 备份原始的 DataFrame
    df_orig = df.copy()
    # 选择 'A' 列，存储其内部的值数组
    series = df["A"]
    vals = series._values

    # 将 'A' 列中的每个元素加一，原数组 vals 应该不等于新的 series._values
    series += 1
    assert series._values is not vals
    # 断言更新后的 DataFrame 与原始备份 df_orig 相等
    tm.assert_frame_equal(df, df_orig)


def test_arithmetic_multiindex_align():
    """
    Regression test for: https://github.com/pandas-dev/pandas/issues/33765
    """

    # 创建一个 MultiIndex 结构的 DataFrame df1
    df1 = DataFrame(
        [[1]],
        index=["a"],
        columns=MultiIndex.from_product([[0], [1]], names=["a", "b"]),
    )
    # 创建一个普通 Index 结构的 DataFrame df2
    df2 = DataFrame([[1]], index=["a"], columns=Index([0], name="a"))
    # 创建期望的结果 DataFrame，与 df1 结构相同
    expected = DataFrame(
        [[0]],
        index=["a"],
        columns=MultiIndex.from_product([[0], [1]], names=["a", "b"]),
    )
    # 执行 df1 - df2 操作，得到结果 result
    result = df1 - df2
    # 断言 result 与期望的 expected DataFrame 相等
    tm.assert_frame_equal(result, expected)


def test_bool_frame_mult_float():
    # GH 18549

    # 创建一个布尔值 DataFrame，形状为 (2, 2)，索引为 'a', 'b'，列名为 'c', 'd'
    df = DataFrame(True, list("ab"), list("cd"))
    # 将 df 中的所有值乘以 1.0 得到结果 result
    result = df * 1.0
    # 创建一个所有元素为 1 的 DataFrame，形状与 df 相同
    expected = DataFrame(np.ones((2, 2)), list("ab"), list("cd"))
    # 断言 result 与期望的 expected DataFrame 相等
    tm.assert_frame_equal(result, expected)


def test_frame_sub_nullable_int(any_int_ea_dtype):
    # GH 32822

    # 创建一个包含 None 值的 Series1，指定 dtype 为 any_int_ea_dtype
    series1 = Series([1, 2, None], dtype=any_int_ea_dtype)
    # 创建一个不包含 None 值的 Series2，指定 dtype 为 any_int_ea_dtype
    series2 = Series([1, 2, 3], dtype=any_int_ea_dtype)
    # 创建期望的结果 DataFrame，包含一个 None 值，dtype 与 series1 相同
    expected = DataFrame([0, 0, None], dtype=any_int_ea_dtype)
    # 执行 series1.to_frame() - series2.to_frame() 操作得到结果 result
    result = series1.to_frame() - series2.to_frame()
    # 断言 result 与期望的 expected DataFrame 相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning"
)
def test_frame_op_subclass_nonclass_constructor():
    # GH#43201 subclass._constructor is a function, not the subclass itself

    # 定义一个继承自 Series 的子类 SubclassedSeries
    class SubclassedSeries(Series):
        @property
        def _constructor(self):
            return SubclassedSeries

        @property
        def _constructor_expanddim(self):
            return SubclassedDataFrame

    # 定义一个继承自 DataFrame 的子类 SubclassedDataFrame
    class SubclassedDataFrame(DataFrame):
        _metadata = ["my_extra_data"]

        def __init__(self, my_extra_data, *args, **kwargs) -> None:
            self.my_extra_data = my_extra_data
            super().__init__(*args, **kwargs)

        @property
        def _constructor(self):
            return functools.partial(type(self), self.my_extra_data)

        @property
        def _constructor_sliced(self):
            return SubclassedSeries

    # 创建一个 SubclassedDataFrame 实例 sdf
    sdf = SubclassedDataFrame("some_data", {"A": [1, 2, 3], "B": [4, 5, 6]})
    # 对 sdf 执行乘以 2 的操作，得到结果 result
    result = sdf * 2
    # 创建一个期望的 SubclassedDataFrame，数据乘以 2
    expected = SubclassedDataFrame("some_data", {"A": [2, 4, 6], "B": [8, 10, 12]})
    # 使用 pandas.testing 模块中的 assert_frame_equal 函数来比较 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
    
    # 将 Spark DataFrame sdf 与自身相加，生成新的 Spark DataFrame result
    result = sdf + sdf
    
    # 使用 pandas.testing 模块中的 assert_frame_equal 函数来比较新生成的 DataFrame result 是否与预期的 DataFrame expected 相等
    tm.assert_frame_equal(result, expected)
# 定义一个用于测试的函数，验证枚举类型列的相等性
def test_enum_column_equality():
    # 创建一个枚举类型 Cols，包含 col1 和 col2 两个成员
    Cols = Enum("Cols", "col1 col2")

    # 创建两个 DataFrame 对象，分别以 Cols.col1 作为列名，数据为 [1, 2, 3]
    q1 = DataFrame({Cols.col1: [1, 2, 3]})
    q2 = DataFrame({Cols.col1: [1, 2, 3]})

    # 比较两个 DataFrame 的同名列 Cols.col1 的相等性，生成布尔型 Series
    result = q1[Cols.col1] == q2[Cols.col1]

    # 创建期望的 Series 对象，其值为 [True, True, True]，列名为 Cols.col1
    expected = Series([True, True, True], name=Cols.col1)

    # 使用测试工具函数验证 result 和 expected 的 Series 是否相等
    tm.assert_series_equal(result, expected)


# 定义一个用于测试的函数，验证混合列索引数据类型的问题
def test_mixed_col_index_dtype():
    # GH 47382
    # 创建 DataFrame 对象 df1，列名为 ['a', 'b', 'c']，数据为 1.0，索引为 [0]
    df1 = DataFrame(columns=list("abc"), data=1.0, index=[0])
    
    # 创建 DataFrame 对象 df2，列名为 ['a', 'b', 'c']，数据为 0.0，索引为 [0]
    df2 = DataFrame(columns=list("abc"), data=0.0, index=[0])
    
    # 将 df2 的列名类型转换为字符串，并赋值给 df1 的列名
    df1.columns = df2.columns.astype("string")
    
    # 计算 df1 和 df2 的元素级加法结果，生成新的 DataFrame 对象 result
    result = df1 + df2
    
    # 创建期望的 DataFrame 对象 expected，列名为 ['a', 'b', 'c']，数据为 1.0，索引为 [0]
    expected = DataFrame(columns=list("abc"), data=1.0, index=[0])
    
    # 使用测试工具函数验证 result 和 expected 的 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
```