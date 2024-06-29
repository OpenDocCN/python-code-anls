# `D:\src\scipysrc\pandas\pandas\tests\frame\test_stack_unstack.py`

```
    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    # 定义一个测试类 TestDataFrameReshape，用于测试 DataFrame 的重塑操作
    class TestDataFrameReshape:
        
        @pytest.mark.filterwarnings(
            "ignore:The previous implementation of stack is deprecated"
        )
        # 定义一个测试方法 test_stack_unstack，测试 stack 和 unstack 方法的功能
        def test_stack_unstack(self, float_frame, future_stack):
            # 复制 float_frame 作为 df
            df = float_frame.copy()
            # 使用 arange 生成数据填充 df
            df[:] = np.arange(np.prod(df.shape)).reshape(df.shape)

            # 执行 stack 操作，并根据 future_stack 参数进行堆叠
            stacked = df.stack(future_stack=future_stack)
            # 创建一个 DataFrame，包含堆叠后的数据，并命名列为 "foo" 和 "bar"
            stacked_df = DataFrame({"foo": stacked, "bar": stacked})

            # 执行 unstack 操作
            unstacked = stacked.unstack()
            # 对比 unstack 后的结果与原始 df，确认是否相等
            tm.assert_frame_equal(unstacked, df)

            # 对 stacked_df 进行 unstack 操作
            unstacked_df = stacked_df.unstack()
            # 对比 unstack 后的 "bar" 列与原始 df，确认是否相等
            tm.assert_frame_equal(unstacked_df["bar"], df)

            # 按行对 stacked 进行 unstack 操作
            unstacked_cols = stacked.unstack(0)
            # 对比按行 unstack 后的结果与原始 df 的转置，确认是否相等
            tm.assert_frame_equal(unstacked_cols.T, df)

            # 对 stacked_df 按行进行 unstack 操作
            unstacked_cols_df = stacked_df.unstack(0)
            # 对比按行 unstack 后的 "bar" 列与原始 df 的转置，确认是否相等
            tm.assert_frame_equal(unstacked_cols_df["bar"].T, df)

        @pytest.mark.filterwarnings(
            "ignore:The previous implementation of stack is deprecated"
        )
        # 定义一个测试方法 test_stack_mixed_level，测试混合级别的 stack 操作
        def test_stack_mixed_level(self, future_stack):
            # 定义多个层级的数据 levels
            levels = [range(3), [3, "a", "b"], [1, 2]]

            # 创建一个 flat columns 的 DataFrame df
            df = DataFrame(1, index=levels[0], columns=levels[1])
            # 执行 stack 操作，根据 future_stack 参数进行堆叠
            result = df.stack(future_stack=future_stack)
            # 生成预期的 Series 结果 expected
            expected = Series(1, index=MultiIndex.from_product(levels[:2]))
            # 对比结果与预期，确认是否相等
            tm.assert_series_equal(result, expected)

            # 创建一个 MultiIndex columns 的 DataFrame df
            df = DataFrame(1, index=levels[0], columns=MultiIndex.from_product(levels[1:]))
            # 执行按列堆叠操作，根据 future_stack 参数进行堆叠
            result = df.stack(1, future_stack=future_stack)
            # 生成预期的 DataFrame 结果 expected
            expected = DataFrame(
                1, index=MultiIndex.from_product([levels[0], levels[2]]), columns=levels[1]
            )
            # 对比结果与预期，确认是否相等
            tm.assert_frame_equal(result, expected)

            # 对 df 中的特定列进行按列堆叠操作，根据 future_stack 参数进行堆叠
            result = df[["a", "b"]].stack(1, future_stack=future_stack)
            # 对比结果与预期，确认是否相等
            expected = expected[["a", "b"]]
            tm.assert_frame_equal(result, expected)

        # 定义一个测试方法 test_unstack_not_consolidated，测试未整理的 unstack 操作
        def test_unstack_not_consolidated(self):
            # 创建一个包含 NaN 的 DataFrame df
            df = DataFrame({"x": [1, 2, np.nan], "y": [3.0, 4, np.nan]})
            # 从 df 中选择部分列形成 df2
            df2 = df[["x"]]
            # 将 df 的 "y" 列添加到 df2
            df2["y"] = df["y"]
            # 断言 df2 的数据块数量为 2
            assert len(df2._mgr.blocks) == 2

            # 执行 unstack 操作
            res = df2.unstack()
            # 生成预期的 unstack 结果
            expected = df.unstack()
            # 对比结果与预期，确认是否相等
            tm.assert_series_equal(res, expected)

        @pytest.mark.filterwarnings(
            "ignore:The previous implementation of stack is deprecated"
        )
    # 定义一个测试方法，用于验证 Series 和 DataFrame 的 unstack 方法在填充值（fill_value）关键字参数方面的行为
    def test_unstack_fill(self, future_stack):
        # GH #9746: fill_value keyword argument for Series
        # and DataFrame unstack
        # 从一个 Series 开始

        # 创建一个包含整数数据的 Series 对象
        data = Series([1, 2, 4, 5], dtype=np.int16)
        # 将 Series 对象的索引设置为 MultiIndex
        data.index = MultiIndex.from_tuples(
            [("x", "a"), ("x", "b"), ("y", "b"), ("z", "a")]
        )

        # 调用 unstack 方法，使用 fill_value=-1 来展开 Series
        result = data.unstack(fill_value=-1)
        # 创建预期的 DataFrame 对象
        expected = DataFrame(
            {"a": [1, -1, 5], "b": [2, 4, -1]}, index=["x", "y", "z"], dtype=np.int16
        )
        # 使用测试框架验证结果是否符合预期
        tm.assert_frame_equal(result, expected)

        # 使用不正确的数据类型（float）作为 fill_value
        result = data.unstack(fill_value=0.5)
        expected = DataFrame(
            {"a": [1, 0.5, 5], "b": [2, 4, 0.5]}, index=["x", "y", "z"], dtype=float
        )
        tm.assert_frame_equal(result, expected)

        # GH #13971: fill_value when unstacking multiple levels:
        # 创建一个 DataFrame 对象，设置其 MultiIndex
        df = DataFrame(
            {"x": ["a", "a", "b"], "y": ["j", "k", "j"], "z": [0, 1, 2], "w": [0, 1, 2]}
        ).set_index(["x", "y", "z"])
        # 使用 fill_value=0 展开多层级的 unstack 操作
        unstacked = df.unstack(["x", "y"], fill_value=0)
        # 选择预期的结果
        key = ("w", "b", "j")
        expected = unstacked[key]
        result = Series([0, 0, 2], index=unstacked.index, name=key)
        tm.assert_series_equal(result, expected)

        # 使用 future_stack 参数对 unstacked 结果进行 stack 操作
        stacked = unstacked.stack(["x", "y"], future_stack=future_stack)
        # 调整索引顺序以匹配原始 DataFrame
        stacked.index = stacked.index.reorder_levels(df.index.names)
        # 为了解决 GH #17886（不必要地转换为浮点数）的问题，将 stacked 转换为 np.int64 类型
        stacked = stacked.astype(np.int64)
        # 最终结果与原始 DataFrame 进行比较
        result = stacked.loc[df.index]
        tm.assert_frame_equal(result, df)

        # 从一个 Series 开始
        s = df["w"]
        # 使用 fill_value=0 对 Series 进行 unstack 操作
        result = s.unstack(["x", "y"], fill_value=0)
        expected = unstacked["w"]
        tm.assert_frame_equal(result, expected)
    def test_unstack_fill_frame(self):
        # 创建一个包含数据的二维列表
        rows = [[1, 2], [3, 4], [5, 6], [7, 8]]
        # 用列表创建一个 DataFrame，指定列名和数据类型
        df = DataFrame(rows, columns=list("AB"), dtype=np.int32)
        # 用 MultiIndex 创建 DataFrame 的行索引
        df.index = MultiIndex.from_tuples(
            [("x", "a"), ("x", "b"), ("y", "b"), ("z", "a")]
        )

        # 对 DataFrame 进行 unstack 操作，用指定值填充缺失值
        result = df.unstack(fill_value=-1)

        # 创建一个预期的 DataFrame，包含特定数据和索引
        rows = [[1, 3, 2, 4], [-1, 5, -1, 6], [7, -1, 8, -1]]
        expected = DataFrame(rows, index=list("xyz"), dtype=np.int32)
        # 使用 MultiIndex 设置预期 DataFrame 的列索引
        expected.columns = MultiIndex.from_tuples(
            [("A", "a"), ("A", "b"), ("B", "a"), ("B", "b")]
        )
        # 检查结果和预期 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 将 DataFrame 中列"A"的数据类型转换为 np.int16
        df["A"] = df["A"].astype(np.int16)
        # 将 DataFrame 中列"B"的数据类型转换为 np.float64
        df["B"] = df["B"].astype(np.float64)

        # 再次对 DataFrame 进行 unstack 操作，用指定值填充缺失值
        result = df.unstack(fill_value=-1)
        # 将预期 DataFrame 中列"A"的数据类型转换为 np.int16
        expected["A"] = expected["A"].astype(np.int16)
        # 将预期 DataFrame 中列"B"的数据类型转换为 np.float64
        expected["B"] = expected["B"].astype(np.float64)
        # 检查结果和预期 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 对 DataFrame 进行 unstack 操作，用不正确的数据类型填充缺失值
        result = df.unstack(fill_value=0.5)

        # 创建一个预期的 DataFrame，包含特定数据和索引，数据类型为 float
        rows = [[1, 3, 2, 4], [0.5, 5, 0.5, 6], [7, 0.5, 8, 0.5]]
        expected = DataFrame(rows, index=list("xyz"), dtype=float)
        # 使用 MultiIndex 设置预期 DataFrame 的列索引
        expected.columns = MultiIndex.from_tuples(
            [("A", "a"), ("A", "b"), ("B", "a"), ("B", "b")]
        )
        # 检查结果和预期 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    def test_unstack_fill_frame_datetime(self):
        # 测试带有日期时间的 unstack 操作
        dv = date_range("2012-01-01", periods=4).values
        data = Series(dv)
        # 用 MultiIndex 创建 Series 的索引
        data.index = MultiIndex.from_tuples(
            [("x", "a"), ("x", "b"), ("y", "b"), ("z", "a")]
        )

        # 对 Series 进行 unstack 操作
        result = data.unstack()
        # 创建一个预期的 DataFrame，包含特定数据和索引
        expected = DataFrame(
            {"a": [dv[0], pd.NaT, dv[3]], "b": [dv[1], dv[2], pd.NaT]},
            index=["x", "y", "z"],
        )
        # 检查结果和预期 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 再次对 Series 进行 unstack 操作，用指定值填充缺失值
        result = data.unstack(fill_value=dv[0])
        # 创建一个预期的 DataFrame，包含特定数据和索引
        expected = DataFrame(
            {"a": [dv[0], dv[0], dv[3]], "b": [dv[1], dv[2], dv[0]]},
            index=["x", "y", "z"],
        )
        # 检查结果和预期 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    def test_unstack_fill_frame_timedelta(self):
        # 测试带有时间差的 unstack 操作
        td = [Timedelta(days=i) for i in range(4)]
        data = Series(td)
        # 用 MultiIndex 创建 Series 的索引
        data.index = MultiIndex.from_tuples(
            [("x", "a"), ("x", "b"), ("y", "b"), ("z", "a")]
        )

        # 对 Series 进行 unstack 操作
        result = data.unstack()
        # 创建一个预期的 DataFrame，包含特定数据和索引
        expected = DataFrame(
            {"a": [td[0], pd.NaT, td[3]], "b": [td[1], td[2], pd.NaT]},
            index=["x", "y", "z"],
        )
        # 检查结果和预期 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 再次对 Series 进行 unstack 操作，用指定值填充缺失值
        result = data.unstack(fill_value=td[1])
        # 创建一个预期的 DataFrame，包含特定数据和索引
        expected = DataFrame(
            {"a": [td[0], td[1], td[3]], "b": [td[1], td[2], td[1]]},
            index=["x", "y", "z"],
        )
        # 检查结果和预期 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)
    def test_unstack_fill_frame_period(self):
        # Test unstacking with period
        
        # 创建一个包含四个 Period 对象的列表
        periods = [
            Period("2012-01"),
            Period("2012-02"),
            Period("2012-03"),
            Period("2012-04"),
        ]
        
        # 将 periods 列表转换为 Series 对象，并设置其索引为 MultiIndex
        data = Series(periods)
        data.index = MultiIndex.from_tuples(
            [("x", "a"), ("x", "b"), ("y", "b"), ("z", "a")]
        )
        
        # 对 data 进行 unstack 操作，将 MultiIndex 转换为 DataFrame
        result = data.unstack()
        
        # 创建预期的 DataFrame 对象 expected
        expected = DataFrame(
            {"a": [periods[0], None, periods[3]], "b": [periods[1], periods[2], None]},
            index=["x", "y", "z"],
        )
        
        # 使用 assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
        
        # 再次对 data 进行 unstack 操作，指定 fill_value 参数为 periods[1]
        result = data.unstack(fill_value=periods[1])
        
        # 创建另一个预期的 DataFrame 对象 expected
        expected = DataFrame(
            {
                "a": [periods[0], periods[1], periods[3]],
                "b": [periods[1], periods[2], periods[1]],
            },
            index=["x", "y", "z"],
        )
        
        # 使用 assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    def test_unstack_fill_frame_categorical(self):
        # Test unstacking with categorical
        
        # 创建一个包含字符串的 Series 对象，并指定其数据类型为 'category'
        data = Series(["a", "b", "c", "a"], dtype="category")
        
        # 设置 data 的索引为 MultiIndex
        data.index = MultiIndex.from_tuples(
            [("x", "a"), ("x", "b"), ("y", "b"), ("z", "a")]
        )
        
        # 默认情况下，缺失值将为 NaN
        result = data.unstack()
        
        # 创建预期的 DataFrame 对象 expected
        expected = DataFrame(
            {
                "a": pd.Categorical(list("axa"), categories=list("abc")),
                "b": pd.Categorical(list("bcx"), categories=list("abc")),
            },
            index=list("xyz"),
        )
        
        # 使用 assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
        
        # 使用字符串 "d" 作为 fill_value 参数，预期会引发 ValueError 异常
        msg = r"Cannot setitem on a Categorical with a new category \(d\)"
        with pytest.raises(TypeError, match=msg):
            data.unstack(fill_value="d")
        
        # 使用字符串 "c" 作为 fill_value 参数，填充缺失值
        result = data.unstack(fill_value="c")
        
        # 创建另一个预期的 DataFrame 对象 expected
        expected = DataFrame(
            {
                "a": pd.Categorical(list("aca"), categories=list("abc")),
                "b": pd.Categorical(list("bcc"), categories=list("abc")),
            },
            index=list("xyz"),
        )
        
        # 使用 assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
    def test_unstack_tuplename_in_multiindex(self):
        # GH 19966
        # 创建一个多级索引，其中包含元组作为名称的一部分
        idx = MultiIndex.from_product(
            [["a", "b", "c"], [1, 2, 3]], names=[("A", "a"), ("B", "b")]
        )
        # 创建一个数据帧，使用上述多级索引作为行索引，包含两列 'd' 和 'e'
        df = DataFrame({"d": [1] * 9, "e": [2] * 9}, index=idx)
        # 对数据帧进行 unstack 操作，根据元组 ("A", "a") 进行展开
        result = df.unstack(("A", "a"))

        # 创建预期的数据帧，包含指定的数据值、列索引和行索引
        expected = DataFrame(
            [[1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2]],
            columns=MultiIndex.from_tuples(
                [
                    ("d", "a"),
                    ("d", "b"),
                    ("d", "c"),
                    ("e", "a"),
                    ("e", "b"),
                    ("e", "c"),
                ],
                names=[None, ("A", "a")],
            ),
            index=Index([1, 2, 3], name=("B", "b")),
        )
        # 使用 assert_frame_equal 函数比较结果和预期，确保它们相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "unstack_idx, expected_values, expected_index, expected_columns",
        [
            (
                ("A", "a"),
                # 预期值1
                [[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]],
                # 预期行索引1
                MultiIndex.from_tuples(
                    [(1, 3), (1, 4), (2, 3), (2, 4)], names=["B", "C"]
                ),
                # 预期列索引1
                MultiIndex.from_tuples(
                    [("d", "a"), ("d", "b"), ("e", "a"), ("e", "b")],
                    names=[None, ("A", "a")],
                ),
            ),
            (
                (("A", "a"), "B"),
                # 预期值2
                [[1, 1, 1, 1, 2, 2, 2, 2], [1, 1, 1, 1, 2, 2, 2, 2]],
                # 预期行索引2
                Index([3, 4], name="C"),
                # 预期列索引2
                MultiIndex.from_tuples(
                    [
                        ("d", "a", 1),
                        ("d", "a", 2),
                        ("d", "b", 1),
                        ("d", "b", 2),
                        ("e", "a", 1),
                        ("e", "a", 2),
                        ("e", "b", 1),
                        ("e", "b", 2),
                    ],
                    names=[None, ("A", "a"), "B"],
                ),
            ),
        ],
    )
    def test_unstack_mixed_type_name_in_multiindex(
        self, unstack_idx, expected_values, expected_index, expected_columns
    ):
        # GH 19966
        # 创建一个多级索引，其中包含元组和非元组作为名称的一部分
        idx = MultiIndex.from_product(
            [["a", "b"], [1, 2], [3, 4]], names=[("A", "a"), "B", "C"]
        )
        # 创建一个数据帧，使用上述多级索引作为行索引，包含两列 'd' 和 'e'
        df = DataFrame({"d": [1] * 8, "e": [2] * 8}, index=idx)
        # 对数据帧进行 unstack 操作，根据 unstack_idx 参数进行展开
        result = df.unstack(unstack_idx)

        # 创建预期的数据帧，包含指定的数据值、列索引和行索引
        expected = DataFrame(
            expected_values, columns=expected_columns, index=expected_index
        )
        # 使用 assert_frame_equal 函数比较结果和预期，确保它们相等
        tm.assert_frame_equal(result, expected)
    def test_unstack_preserve_dtypes(self):
        # 检查对 #11847 的修复
        # 创建一个包含各种数据类型的 DataFrame
        df = DataFrame(
            {
                "state": ["IL", "MI", "NC"],
                "index": ["a", "b", "c"],
                "some_categories": Series(["a", "b", "c"]).astype("category"),
                "A": np.random.default_rng(2).random(3),
                "B": 1,
                "C": "foo",
                "D": pd.Timestamp("20010102"),
                "E": Series([1.0, 50.0, 100.0]).astype("float32"),
                "F": Series([3.0, 4.0, 5.0]).astype("float64"),
                "G": False,
                "H": Series([1, 200, 923442]).astype("int8"),
            }
        )

        def unstack_and_compare(df, column_name):
            # 对 DataFrame 根据指定列进行 unstack 操作
            unstacked1 = df.unstack([column_name])
            unstacked2 = df.unstack(column_name)
            # 比较两次 unstack 的结果是否相等
            tm.assert_frame_equal(unstacked1, unstacked2)

        # 根据不同的列组合对 DataFrame 进行 set_index 和 unstack 操作
        df1 = df.set_index(["state", "index"])
        unstack_and_compare(df1, "index")

        df1 = df.set_index(["state", "some_categories"])
        unstack_and_compare(df1, "some_categories")

        df1 = df.set_index(["F", "C"])
        unstack_and_compare(df1, "F")

        df1 = df.set_index(["G", "B", "state"])
        unstack_and_compare(df1, "B")

        df1 = df.set_index(["E", "A"])
        unstack_and_compare(df1, "E")

        df1 = df.set_index(["state", "index"])
        s = df1["A"]
        unstack_and_compare(s, "index")

    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    def test_stack_ints(self, future_stack):
        # 创建一个具有特定列组合的 DataFrame
        columns = MultiIndex.from_tuples(list(itertools.product(range(3), repeat=3)))
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 27)), columns=columns
        )

        # 使用 future_stack 参数调用 stack 函数，并比较结果
        tm.assert_frame_equal(
            df.stack(level=[1, 2], future_stack=future_stack),
            df.stack(level=1, future_stack=future_stack).stack(
                level=1, future_stack=future_stack
            ),
        )
        tm.assert_frame_equal(
            df.stack(level=[-2, -1], future_stack=future_stack),
            df.stack(level=1, future_stack=future_stack).stack(
                level=1, future_stack=future_stack
            ),
        )

        # 对 DataFrame 的列进行命名，并验证返回值为 None
        df_named = df.copy()
        return_value = df_named.columns.set_names(range(3), inplace=True)
        assert return_value is None

        # 使用 future_stack 参数调用 stack 函数，并比较结果
        tm.assert_frame_equal(
            df_named.stack(level=[1, 2], future_stack=future_stack),
            df_named.stack(level=1, future_stack=future_stack).stack(
                level=1, future_stack=future_stack
            ),
        )
    # 定义一个测试函数，用于测试带有不同层级混合的堆叠操作
    def test_stack_mixed_levels(self, future_stack):
        # 创建一个多级索引，列名包括实验(exp)、动物(animal)和毛发长度(hair_length)
        columns = MultiIndex.from_tuples(
            [
                ("A", "cat", "long"),
                ("B", "cat", "long"),
                ("A", "dog", "short"),
                ("B", "dog", "short"),
            ],
            names=["exp", "animal", "hair_length"],
        )
        # 使用随机生成的标准正态分布数据创建一个 DataFrame
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)), columns=columns
        )

        # 在指定的层级["animal", "hair_length"]上堆叠 DataFrame，并传递 future_stack 参数
        animal_hair_stacked = df.stack(
            level=["animal", "hair_length"], future_stack=future_stack
        )
        # 在指定的层级["exp", "hair_length"]上堆叠 DataFrame，并传递 future_stack 参数
        exp_hair_stacked = df.stack(
            level=["exp", "hair_length"], future_stack=future_stack
        )

        # GH #8584: 需要检查在传入既是层级名称又在层级编号范围内的数字时，堆叠操作是否正常工作
        df2 = df.copy()
        df2.columns.names = ["exp", "animal", 1]
        # 断言堆叠后的 DataFrame 与预期的 animal_hair_stacked 相等，忽略列名检查
        tm.assert_frame_equal(
            df2.stack(level=["animal", 1], future_stack=future_stack),
            animal_hair_stacked,
            check_names=False,
        )
        # 断言堆叠后的 DataFrame 与预期的 exp_hair_stacked 相等，忽略列名检查
        tm.assert_frame_equal(
            df2.stack(level=["exp", 1], future_stack=future_stack),
            exp_hair_stacked,
            check_names=False,
        )

        # 当传入的层级类型混合且整数不是层级名称时，应该抛出 ValueError 异常
        msg = (
            "level should contain all level names or all level numbers, not "
            "a mixture of the two"
        )
        with pytest.raises(ValueError, match=msg):
            df2.stack(level=["animal", 0], future_stack=future_stack)

        # GH #8584: 如果在列名中包含数字 0，可能会引发与 lexsort 深度相关的奇怪错误
        df3 = df.copy()
        df3.columns.names = ["exp", "animal", 0]
        # 断言堆叠后的 DataFrame 与预期的 animal_hair_stacked 相等，忽略列名检查
        tm.assert_frame_equal(
            df3.stack(level=["animal", 0], future_stack=future_stack),
            animal_hair_stacked,
            check_names=False,
        )

    # 使用 pytest 的标记，忽略有关堆叠函数旧实现的警告信息
    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    # 定义一个测试方法，测试堆栈操作在整数级别列名上的效果
    def test_stack_int_level_names(self, future_stack):
        # 创建一个多级索引，用于数据框的列，包含不同层级的类别和属性信息
        columns = MultiIndex.from_tuples(
            [
                ("A", "cat", "long"),
                ("B", "cat", "long"),
                ("A", "dog", "short"),
                ("B", "dog", "short"),
            ],
            names=["exp", "animal", "hair_length"],  # 设置列索引的名称
        )
        # 使用随机数填充的数据框，形状为4x4，列使用上述多级索引
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)), columns=columns
        )

        # 在指定的层级上对数据框进行堆栈操作，返回堆栈后的结果
        exp_animal_stacked = df.stack(
            level=["exp", "animal"], future_stack=future_stack
        )
        animal_hair_stacked = df.stack(
            level=["animal", "hair_length"], future_stack=future_stack
        )
        exp_hair_stacked = df.stack(
            level=["exp", "hair_length"], future_stack=future_stack
        )

        # 复制数据框df，设置复制后数据框的列索引名称
        df2 = df.copy()
        df2.columns.names = [0, 1, 2]

        # 使用测试工具函数检查堆栈操作后的数据框是否与预期结果相等
        tm.assert_frame_equal(
            df2.stack(level=[1, 2], future_stack=future_stack),
            animal_hair_stacked,
            check_names=False,
        )
        tm.assert_frame_equal(
            df2.stack(level=[0, 1], future_stack=future_stack),
            exp_animal_stacked,
            check_names=False,
        )
        tm.assert_frame_equal(
            df2.stack(level=[0, 2], future_stack=future_stack),
            exp_hair_stacked,
            check_names=False,
        )

        # 测试不按顺序排列的整数列名的情况
        df3 = df.copy()
        df3.columns.names = [2, 0, 1]
        tm.assert_frame_equal(
            df3.stack(level=[0, 1], future_stack=future_stack),
            animal_hair_stacked,
            check_names=False,
        )
        tm.assert_frame_equal(
            df3.stack(level=[2, 0], future_stack=future_stack),
            exp_animal_stacked,
            check_names=False,
        )
        tm.assert_frame_equal(
            df3.stack(level=[2, 1], future_stack=future_stack),
            exp_hair_stacked,
            check_names=False,
        )

    # 定义一个测试方法，测试在布尔值索引情况下的反堆栈操作
    def test_unstack_bool(self):
        # 创建一个数据框，包含布尔值，使用多级索引作为行索引和列索引
        df = DataFrame(
            [False, False],
            index=MultiIndex.from_arrays([["a", "b"], ["c", "l"]]),
            columns=["col"],
        )
        # 对数据框进行反堆栈操作，将多级行索引转换为列索引
        rs = df.unstack()
        # 创建一个期望的数据框，包含NaN值的布尔值矩阵，行索引为'a'和'b'，列索引为多级结构
        xp = DataFrame(
            np.array([[False, np.nan], [np.nan, False]], dtype=object),
            index=["a", "b"],
            columns=MultiIndex.from_arrays([["col", "col"], ["c", "l"]]),
        )
        # 使用测试工具函数检查反堆栈操作后的结果是否与期望值相等
        tm.assert_frame_equal(rs, xp)

    # 标记当前测试方法将忽略特定警告信息
    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    # 测试函数，用于测试 unstack 方法在绑定未来堆栈时的行为
    def test_unstack_level_binding(self, future_stack):
        # 创建一个多级索引对象 mi
        mi = MultiIndex(
            # 定义多级索引的级别和对应的代码
            levels=[["foo", "bar"], ["one", "two"], ["a", "b"]],
            codes=[[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0]],
            # 设置每个级别的名称
            names=["first", "second", "third"],
        )
        # 创建一个 Series 对象 s，初始值为 0，使用 mi 作为索引
        s = Series(0, index=mi)
        # 对 s 执行 unstack 操作，指定要解堆栈的级别，同时传入 future_stack 参数
        result = s.unstack([1, 2]).stack(0, future_stack=future_stack)

        # 预期的多级索引对象 expected_mi
        expected_mi = MultiIndex(
            levels=[["foo", "bar"], ["one", "two"]],
            codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
            names=["first", "second"],
        )

        # 创建一个 DataFrame 对象 expected，包含预期的数据内容
        expected = DataFrame(
            np.array(
                [[0, np.nan], [np.nan, 0], [0, np.nan], [np.nan, 0]], dtype=np.float64
            ),
            index=expected_mi,
            columns=Index(["b", "a"], name="third"),
        )

        # 使用 assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 测试函数，用于测试 unstack 方法转换为 Series 对象的行为
    def test_unstack_to_series(self, float_frame):
        # 检查 unstack 方法的逆过程
        data = float_frame.unstack()

        # 断言 data 是否为 Series 对象
        assert isinstance(data, Series)
        # 执行 data 的 unstack 操作，再进行转置操作得到 undo
        undo = data.unstack().T
        # 使用 assert_frame_equal 检查 undo 和 float_frame 是否相等
        tm.assert_frame_equal(undo, float_frame)

        # 检查 NA 值的处理
        data = DataFrame({"x": [1, 2, np.nan], "y": [3.0, 4, np.nan]})
        data.index = Index(["a", "b", "c"])
        result = data.unstack()

        # 创建预期的 MultiIndex 对象 midx
        midx = MultiIndex(
            levels=[["x", "y"], ["a", "b", "c"]],
            codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]],
        )
        # 创建预期的 Series 对象 expected
        expected = Series([1, 2, np.nan, 3, 4, np.nan], index=midx)

        # 使用 assert_series_equal 检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 检查 unstack 方法的可组合性
        old_data = data.copy()
        for _ in range(4):
            data = data.unstack()
        # 使用 assert_frame_equal 检查 old_data 和 data 是否相等
        tm.assert_frame_equal(old_data, data)
    # 测试用例，验证在使用推断字符串时是否能正确处理
    def test_unstack_dtypes(self, using_infer_string):
        # GH 2929
        # 创建包含四行数据的二维列表
        rows = [[1, 1, 3, 4], [1, 2, 3, 4], [2, 1, 3, 4], [2, 2, 3, 4]]

        # 用列表创建 DataFrame 对象，列名为 "ABCD"
        df = DataFrame(rows, columns=list("ABCD"))
        
        # 获取 DataFrame 的数据类型 Series
        result = df.dtypes
        
        # 创建预期的数据类型 Series，每列类型为 np.dtype("int64")
        expected = Series([np.dtype("int64")] * 4, index=list("ABCD"))
        
        # 断言两个 Series 是否相等
        tm.assert_series_equal(result, expected)

        # 将 DataFrame 按照 ["A", "B"] 列设置索引
        df2 = df.set_index(["A", "B"])
        
        # 对新的 DataFrame 执行 unstack 操作，按照 "B" 列展开
        df3 = df2.unstack("B")
        
        # 获取展开后 DataFrame 的数据类型 Series
        result = df3.dtypes
        
        # 创建预期的数据类型 Series，每列类型为 np.dtype("int64")，使用 MultiIndex 指定列名
        expected = Series(
            [np.dtype("int64")] * 4,
            index=MultiIndex.from_arrays(
                [["C", "C", "D", "D"], [1, 2, 1, 2]], names=(None, "B")
            ),
        )
        
        # 断言两个 Series 是否相等
        tm.assert_series_equal(result, expected)

        # 向 DataFrame 中混合数据类型
        df2 = df.set_index(["A", "B"])
        df2["C"] = 3.0
        
        # 再次按照 "B" 列展开 DataFrame
        df3 = df2.unstack("B")
        
        # 获取展开后 DataFrame 的数据类型 Series
        result = df3.dtypes
        
        # 创建预期的数据类型 Series，前两列类型为 np.dtype("float64")，后两列类型为 np.dtype("int64")
        expected = Series(
            [np.dtype("float64")] * 2 + [np.dtype("int64")] * 2,
            index=MultiIndex.from_arrays(
                [["C", "C", "D", "D"], [1, 2, 1, 2]], names=(None, "B")
            ),
        )
        
        # 断言两个 Series 是否相等
        tm.assert_series_equal(result, expected)
        
        # 向 DataFrame 中添加字符串类型数据
        df2["D"] = "foo"
        
        # 再次按照 "B" 列展开 DataFrame
        df3 = df2.unstack("B")
        
        # 获取展开后 DataFrame 的数据类型 Series
        result = df3.dtypes
        
        # 根据 using_infer_string 是否为 True，选择不同的 dtype
        dtype = "string" if using_infer_string else np.dtype("object")
        
        # 创建预期的数据类型 Series，前两列类型为 np.dtype("float64")，后两列类型为 dtype
        expected = Series(
            [np.dtype("float64")] * 2 + [dtype] * 2,
            index=MultiIndex.from_arrays(
                [["C", "C", "D", "D"], [1, 2, 1, 2]], names=(None, "B")
            ),
        )
        
        # 断言两个 Series 是否相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "c, d",
        (
            (np.zeros(5), np.zeros(5)),  # 参数化测试，传入两个长度为 5 的零数组
            (np.arange(5, dtype="f8"), np.arange(5, 10, dtype="f8")),  # 参数化测试，传入浮点数数组
        ),
    )
    # 测试特定情况下的 unstack 操作
    def test_unstack_dtypes_mixed_date(self, c, d):
        # GH7405
        # 创建包含四列数据的 DataFrame，包括字符串 "a"、浮点数数组 c 和 d，日期范围从 "2012-01-01" 开始，共五个周期
        df = DataFrame(
            {
                "A": ["a"] * 5,
                "C": c,
                "D": d,
                "B": date_range("2012-01-01", periods=5),
            }
        )
        
        # 复制 DataFrame 的前三行数据并命名为 right
        right = df.iloc[:3].copy(deep=True)
        
        # 将 DataFrame 按照 ["A", "B"] 列设置索引
        df = df.set_index(["A", "B"])
        
        # 将 "D" 列数据类型转换为 int64
        df["D"] = df["D"].astype("int64")
        
        # 对 df 的前三行数据按照 "A" 列展开
        left = df.iloc[:3].unstack(0)
        
        # 对 right 的前三行数据按照 "A" 列展开
        right = right.set_index(["A", "B"]).unstack(0)
        
        # 将 right 中的 ("D", "a") 列数据类型转换为 int64
        right[("D", "a")] = right[("D", "a")].astype("int64")
        
        # 断言 left 的形状是否为 (3, 2)，并且 left 与 right 是否相等
        assert left.shape == (3, 2)
        tm.assert_frame_equal(left, right)

    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    # 测试在非唯一索引名称下执行 unstack 操作是否会抛出 ValueError 异常
    def test_unstack_non_unique_index_names(self, future_stack):
        # 创建 MultiIndex，其中索引名称 "c1" 出现多次
        idx = MultiIndex.from_tuples([("a", "b"), ("c", "d")], names=["c1", "c1"])
        
        # 创建包含两行数据的 DataFrame，使用 MultiIndex 作为索引
        df = DataFrame([1, 2], index=idx)
        
        # 预期抛出的 ValueError 异常消息
        msg = "The name c1 occurs multiple times, use a level number"
        
        # 断言执行 df.unstack("c1") 会抛出 ValueError 异常，异常消息符合预期
        with pytest.raises(ValueError, match=msg):
            df.unstack("c1")
        
        # 断言执行 df.T.stack("c1", future_stack=future_stack) 会抛出 ValueError 异常，异常消息符合预期
        with pytest.raises(ValueError, match=msg):
            df.T.stack("c1", future_stack=future_stack)
    def test_unstack_unused_levels(self):
        # GH 17845: unused codes in index make unstack() cast int to float
        # 创建一个多级索引，其中第一级包含单个元素 "a"，第二级包含 ["A", "B", "C", "D"]，然后去除最后一个元素
        idx = MultiIndex.from_product([["a"], ["A", "B", "C", "D"]])[:-1]
        # 创建一个数据框，包含多次重复的 [[1, 0]]，并使用上述索引
        df = DataFrame([[1, 0]] * 3, index=idx)

        # 对数据框进行 unstack 操作
        result = df.unstack()
        # 期望的列索引，包含两个层级 [0, 1] 和 ["A", "B", "C"]
        exp_col = MultiIndex.from_product([[0, 1], ["A", "B", "C"]])
        # 创建期望的数据框，包含 [[1, 1, 1, 0, 0, 0]]，使用 "a" 作为索引，exp_col 作为列索引
        expected = DataFrame([[1, 1, 1, 0, 0, 0]], index=["a"], columns=exp_col)
        # 断言结果和期望是否相等
        tm.assert_frame_equal(result, expected)
        # 断言结果的第二级列索引与原始索引的第二级是否完全相同
        assert (result.columns.levels[1] == idx.levels[1]).all()

        # 未使用的项目存在于两个级别上
        # 创建新的多级索引，包含两个层级，levels = [[0, 1, 7], [0, 1, 2, 3]]，codes = [[0, 0, 1, 1], [0, 2, 0, 2]]
        idx = MultiIndex(levels, codes)
        # 创建包含两次重复的 block 的数据框，使用上述索引
        block = np.arange(4).reshape(2, 2)
        df = DataFrame(np.concatenate([block, block + 4]), index=idx)
        # 对数据框进行 unstack 操作
        result = df.unstack()
        # 创建期望的数据框，使用 idx 作为列索引，包含 np.concatenate([block * 2, block * 2 + 1], axis=1) 的数据
        expected = DataFrame(
            np.concatenate([block * 2, block * 2 + 1], axis=1), columns=idx
        )
        # 断言结果和期望是否相等
        tm.assert_frame_equal(result, expected)
        # 断言结果的第二级列索引与原始索引的第二级是否完全相同
        assert (result.columns.levels[1] == idx.levels[1]).all()

    @pytest.mark.parametrize(
        "level, idces, col_level, idx_level",
        (
            (0, [13, 16, 6, 9, 2, 5, 8, 11], [np.nan, "a", 2], [np.nan, 5, 1]),
            (1, [8, 11, 1, 4, 12, 15, 13, 16], [np.nan, 5, 1], [np.nan, "a", 2]),
        ),
    )
    def test_unstack_unused_levels_mixed_with_nan(
        self, level, idces, col_level, idx_level
    ):
        # With mixed dtype and NaN
        # 创建一个多级索引，包含两个层级，levels = [["a", 2, "c"], [1, 3, 5, 7]]，codes = [[0, -1, 1, 1], [0, 2, -1, 2]]
        idx = MultiIndex(levels, codes)
        # 创建包含连续数字的数据数组
        data = np.arange(8)
        df = DataFrame(data.reshape(4, 2), index=idx)

        # 对数据框进行 unstack 操作，使用给定的 level 参数
        result = df.unstack(level=level)
        # 创建期望的数据数组，其中包含大量 NaN 值
        exp_data = np.zeros(18) * np.nan
        exp_data[idces] = data
        # 创建期望的列索引，包含两个层级 [0, 1] 和 col_level
        cols = MultiIndex.from_product([[0, 1], col_level])
        # 创建期望的数据框，使用 idx_level 作为索引，cols 作为列索引
        expected = DataFrame(exp_data.reshape(3, 6), index=idx_level, columns=cols)
        # 断言结果和期望是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("cols", [["A", "C"], slice(None)])
    def test_unstack_unused_level(self, cols):
        # GH 18562 : unused codes on the unstacked level
        # 创建一个包含多个列表的数据框
        df = DataFrame([[2010, "a", "I"], [2011, "b", "II"]], columns=["A", "B", "C"])

        # 设置索引为 ["A", "B", "C"]，保留列
        ind = df.set_index(["A", "B", "C"], drop=False)
        # 使用切片选择数据框的子集，然后对其进行 unstack 操作
        selection = ind.loc[(slice(None), slice(None), "I"), cols]
        result = selection.unstack()

        # 创建期望的数据框，包含 ind 的第一行数据和 cols 作为列索引
        expected = ind.iloc[[0]][cols]
        expected.columns = MultiIndex.from_product(
            [expected.columns, ["I"]], names=[None, "C"]
        )
        expected.index = expected.index.droplevel("C")
        # 断言结果和期望是否相等
        tm.assert_frame_equal(result, expected)
    def test_unstack_long_index(self):
        # PH 32624: Error when using a lot of indices to unstack.
        # The error occurred only, if a lot of indices are used.
        # 创建一个包含单个值的DataFrame，具有多层索引和多级列
        df = DataFrame(
            [[1]],
            columns=MultiIndex.from_tuples([[0]], names=["c1"]),
            index=MultiIndex.from_tuples(
                [[0, 0, 1, 0, 0, 0, 1]],
                names=["i1", "i2", "i3", "i4", "i5", "i6", "i7"],
            ),
        )
        # 对DataFrame进行多级unstack操作，使用多个索引进行展开
        result = df.unstack(["i2", "i3", "i4", "i5", "i6", "i7"])
        # 创建一个期望的DataFrame，进行结果比较
        expected = DataFrame(
            [[1]],
            columns=MultiIndex.from_tuples(
                [[0, 0, 1, 0, 0, 0, 1]],
                names=["c1", "i2", "i3", "i4", "i5", "i6", "i7"],
            ),
            index=Index([0], name="i1"),
        )
        # 使用测试工具检查结果DataFrame与期望DataFrame是否相等
        tm.assert_frame_equal(result, expected)

    def test_unstack_multi_level_cols(self):
        # PH 24729: Unstack a df with multi level columns
        # 创建一个包含浮点数的DataFrame，具有多级列和多层索引
        df = DataFrame(
            [[0.0, 0.0], [0.0, 0.0]],
            columns=MultiIndex.from_tuples(
                [["B", "C"], ["B", "D"]], names=["c1", "c2"]
            ),
            index=MultiIndex.from_tuples(
                [[10, 20, 30], [10, 20, 40]], names=["i1", "i2", "i3"]
            ),
        )
        # 对DataFrame进行多级unstack操作，使用多个索引进行展开
        assert df.unstack(["i2", "i1"]).columns.names[-2:] == ["i2", "i1"]

    def test_unstack_multi_level_rows_and_cols(self):
        # PH 28306: Unstack df with multi level cols and rows
        # 创建一个包含整数的DataFrame，具有多级列和多层索引
        df = DataFrame(
            [[1, 2], [3, 4], [-1, -2], [-3, -4]],
            columns=MultiIndex.from_tuples([["a", "b", "c"], ["d", "e", "f"]]),
            index=MultiIndex.from_tuples(
                [
                    ["m1", "P3", 222],
                    ["m1", "A5", 111],
                    ["m2", "P3", 222],
                    ["m2", "A5", 111],
                ],
                names=["i1", "i2", "i3"],
            ),
        )
        # 对DataFrame进行多级unstack操作，使用多个索引进行展开
        result = df.unstack(["i3", "i2"])
        # 创建一个期望的DataFrame，进行结果比较
        expected = df.unstack(["i3"]).unstack(["i2"])
        # 使用测试工具检查结果DataFrame与期望DataFrame是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("idx", [("jim", "joe"), ("joe", "jim")])
    @pytest.mark.parametrize("lev", list(range(2)))
    # 定义一个测试方法，用于测试在特定条件下的 DataFrame 的解索引操作
    def test_unstack_nan_index1(self, idx, lev):
        # GH7466
        # 定义一个内部函数 cast，用于将值转换为字符串格式，如果是 NaN 则转换为空字符串
        def cast(val):
            val_str = "" if val != val else val
            return f"{val_str:1}"

        # 创建一个 DataFrame 对象 df，包含三列数据，其中一列包含 NaN 值
        df = DataFrame(
            {
                "jim": ["a", "b", np.nan, "d"],
                "joe": ["w", "x", "y", "z"],
                "jolie": ["a.w", "b.x", " .y", "d.z"],
            }
        )

        # 根据 ["jim", "joe"] 列设置索引，并将结果展开为新的 DataFrame，选择 "jolie" 列作为数据
        left = df.set_index(["jim", "joe"]).unstack()["jolie"]
        # 根据 ["joe", "jim"] 列设置索引，并将结果展开为新的 DataFrame，选择 "jolie" 列作为数据，然后转置
        right = df.set_index(["joe", "jim"]).unstack()["jolie"].T
        # 使用测试框架中的断言方法验证 left 和 right 是否相等
        tm.assert_frame_equal(left, right)

        # 根据给定的 idx 列列表设置 DataFrame 的多级索引
        mi = df.set_index(list(idx))
        # 对设置了多级索引的 DataFrame 进行 level=lev 的解索引操作，得到新的 DataFrame udf
        udf = mi.unstack(level=lev)
        # 使用断言验证 udf 中非 NaN 值的数量是否等于 df 的总长度
        assert udf.notna().values.sum() == len(df)

        # 定义一个 lambda 函数 mk_list，用于将元组转换为列表，否则直接返回列表
        mk_list = lambda a: list(a) if isinstance(a, tuple) else [a]
        # 找到 udf["jolie"] 中非 NaN 值的行列索引
        rows, cols = udf["jolie"].notna().values.nonzero()
        # 遍历非 NaN 值的行列索引，分别比较左右两侧的排序后结果是否相等
        for i, j in zip(rows, cols):
            left = sorted(udf["jolie"].iloc[i, j].split("."))
            right = mk_list(udf["jolie"].index[i]) + mk_list(udf["jolie"].columns[j])
            right = sorted(map(cast, right))
            # 使用断言验证 left 和 right 是否相等
            assert left == right

    # 使用 pytest 的参数化标记，为 idx 参数提供全排列的测试数据
    @pytest.mark.parametrize("idx", itertools.permutations(["1st", "2nd", "3rd"]))
    # 使用 pytest 的参数化标记，为 lev 参数提供范围在 0 到 2 之间的测试数据
    @pytest.mark.parametrize("lev", list(range(3)))
    # 使用 pytest 的参数化标记，为 col 参数提供 "4th" 和 "5th" 两个值的测试数据
    @pytest.mark.parametrize("col", ["4th", "5th"])
    def test_unstack_nan_index_repeats(self, idx, lev, col):
        # 定义一个内部函数cast，用于将值转换为字符串格式，如果值为NaN，则转换为空字符串
        def cast(val):
            val_str = "" if val != val else val
            return f"{val_str:1}"

        # 创建一个DataFrame对象df，包含四列数据，其中第一列包含字符串"d"和NaN值，其它列包含字符串和NaN值
        df = DataFrame(
            {
                "1st": ["d"] * 3 + [np.nan] * 5 + ["a"] * 2 + ["c"] * 3 + ["e"] * 2 + ["b"] * 5,
                "2nd": ["y"] * 2 + ["w"] * 3 + [np.nan] * 3 + ["z"] * 4 + [np.nan] * 3 + ["x"] * 3 + [np.nan] * 2,
                "3rd": [
                    67, 39, 53, 72, 57, 80, 31, 18, 11, 30, 59, 50, 62, 59, 76, 52, 14, 53, 60, 51,
                ],
            }
        )

        # 添加两列到df，分别为第四列和第五列，应用lambda函数将每行数据转换为由点号连接的字符串
        df["4th"], df["5th"] = (
            df.apply(lambda r: ".".join(map(cast, r)), axis=1),  # 第四列转换
            df.apply(lambda r: ".".join(map(cast, r.iloc[::-1])), axis=1),  # 第五列转换
        )

        # 将df按照idx中的列作为多级索引设置给mi
        mi = df.set_index(list(idx))

        # 对mi执行unstack操作，根据lev级别展开数据
        udf = mi.unstack(level=lev)

        # 断言udf中非NaN值的数量等于df中元素个数的两倍
        assert udf.notna().values.sum() == 2 * len(df)

        # 定义lambda函数mk_list，用于将元组转换为列表
        mk_list = lambda a: list(a) if isinstance(a, tuple) else [a]

        # 找出udf[col]中非NaN值的行和列索引
        rows, cols = udf[col].notna().values.nonzero()

        # 遍历非NaN值的行和列索引
        for i, j in zip(rows, cols):
            # 对比左右两边的数据是否相等，左边是udf[col]的拆分结果，右边是mk_list(udf[col].index[i])和mk_list(udf[col].columns[j])的组合结果
            left = sorted(udf[col].iloc[i, j].split("."))
            right = mk_list(udf[col].index[i]) + mk_list(udf[col].columns[j])
            right = sorted(map(cast, right))
            assert left == right
    def test_unstack_nan_index2(self):
        # 测试用例名称：test_unstack_nan_index2
        # GH7403：GitHub issue编号
        # 创建一个DataFrame对象df，包含三列："A"列为['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']，"B"列为[0, 1, 2, 3, 4, 5, 6, 7]，"C"列同样为[0, 1, 2, 3, 4, 5, 6, 7]
        df = DataFrame({"A": list("aaaabbbb"), "B": range(8), "C": range(8)})
        
        # 将"B"列显式转换为浮点类型，以避免设置为np.nan时的隐式转换
        df = df.astype({"B": "float"})
        
        # 在df的第四行（索引为3），第二列（索引为1）设置为np.nan
        df.iloc[3, 1] = np.nan
        
        # 使用["A", "B"]作为索引创建一个新的DataFrame对象left，然后进行unstack操作
        left = df.set_index(["A", "B"]).unstack(0)

        # 预期的DataFrame数据值
        vals = [
            [3, 0, 1, 2, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 4, 5, 6, 7],
        ]
        # 转置vals列表
        vals = list(map(list, zip(*vals)))
        
        # 创建一个Index对象idx，包含值[np.nan, 0, 1, 2, 4, 5, 6, 7]，命名为"B"
        idx = Index([np.nan, 0, 1, 2, 4, 5, 6, 7], name="B")
        
        # 创建一个MultiIndex对象cols，包含两级，第一级为["C"]，第二级为["a", "b"]，codes分别为[0, 0]和[0, 1]，没有名称
        cols = MultiIndex(
            levels=[["C"], ["a", "b"]], codes=[[0, 0], [0, 1]], names=[None, "A"]
        )

        # 创建一个DataFrame对象right，数据为vals，列索引为cols，行索引为idx
        right = DataFrame(vals, columns=cols, index=idx)
        
        # 使用assert_frame_equal函数比较left和right，确保它们相等
        tm.assert_frame_equal(left, right)

        # 以下是第二个测试用例，与上面的代码结构类似，不再重复注释具体语句作用

        # 创建第二个DataFrame对象df
        df = DataFrame({"A": list("aaaabbbb"), "B": list(range(4)) * 2, "C": range(8)})
        df = df.astype({"B": "float"})
        df.iloc[2, 1] = np.nan
        left = df.set_index(["A", "B"]).unstack(0)

        vals = [[2, np.nan], [0, 4], [1, 5], [np.nan, 6], [3, 7]]
        cols = MultiIndex(
            levels=[["C"], ["a", "b"]], codes=[[0, 0], [0, 1]], names=[None, "A"]
        )
        idx = Index([np.nan, 0, 1, 2, 3], name="B")
        right = DataFrame(vals, columns=cols, index=idx)
        tm.assert_frame_equal(left, right)

        # 创建第三个DataFrame对象df
        df = DataFrame({"A": list("aaaabbbb"), "B": list(range(4)) * 2, "C": range(8)})
        df = df.astype({"B": "float"})
        df.iloc[3, 1] = np.nan
        left = df.set_index(["A", "B"]).unstack(0)

        vals = [[3, np.nan], [0, 4], [1, 5], [2, 6], [np.nan, 7]]
        cols = MultiIndex(
            levels=[["C"], ["a", "b"]], codes=[[0, 0], [0, 1]], names=[None, "A"]
        )
        idx = Index([np.nan, 0, 1, 2, 3], name="B")
        right = DataFrame(vals, columns=cols, index=idx)
        tm.assert_frame_equal(left, right)

    def test_unstack_nan_index3(self):
        # 测试用例名称：test_unstack_nan_index3
        # GH7401：GitHub issue编号
        # 创建一个DataFrame对象df，包含三列："A"列为['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b']，"B"列为一组日期列表，"C"列为[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        df = DataFrame(
            {
                "A": list("aaaaabbbbb"),
                "B": (date_range("2012-01-01", periods=5).tolist() * 2),
                "C": np.arange(10),
            }
        )

        # 在df的第四行（索引为3），第二列（索引为1）设置为np.nan
        df.iloc[3, 1] = np.nan
        
        # 使用["A", "B"]作为索引创建一个新的DataFrame对象left，然后进行unstack操作
        left = df.set_index(["A", "B"]).unstack()

        # 预期的数据值vals，使用np.array创建
        vals = np.array([[3, 0, 1, 2, np.nan, 4], [np.nan, 5, 6, 7, 8, 9]])
        
        # 创建一个Index对象idx，包含值["a", "b"]，命名为"A"
        idx = Index(["a", "b"], name="A")
        
        # 创建一个MultiIndex对象cols，包含两级，第一级为["C"]，第二级为日期范围，codes分别为[0, 0, 0, 0, 0, 0]和[-1, 0, 1, 2, 3, 4]，没有名称
        cols = MultiIndex(
            levels=[["C"], date_range("2012-01-01", periods=5)],
            codes=[[0, 0, 0, 0, 0, 0], [-1, 0, 1, 2, 3, 4]],
            names=[None, "B"],
        )

        # 创建一个DataFrame对象right，数据为vals，列索引为cols，行索引为idx
        right = DataFrame(vals, columns=cols, index=idx)
        
        # 使用assert_frame_equal函数比较left和right，确保它们相等
        tm.assert_frame_equal(left, right)
    def test_unstack_nan_index4(self):
        # GH4862 - 测试用例名称，引用 GitHub issue #4862
        vals = [
            ["Hg", np.nan, np.nan, 680585148],  # 数据值列表，包含字符串和数字，其中包含 NaN
            ["U", 0.0, np.nan, 680585148],
            ["Pb", 7.07e-06, np.nan, 680585148],
            ["Sn", 2.3614e-05, 0.0133, 680607017],
            ["Ag", 0.0, 0.0133, 680607017],
            ["Hg", -0.00015, 0.0133, 680607017],
        ]
        df = DataFrame(
            vals,
            columns=["agent", "change", "dosage", "s_id"],  # 列名定义
            index=[17263, 17264, 17265, 17266, 17267, 17268],  # 索引定义
        )

        left = df.copy().set_index(["s_id", "dosage", "agent"]).unstack()
        # 将 DataFrame 按指定列进行多级索引的堆叠操作，并赋值给 left

        vals = [
            [np.nan, np.nan, 7.07e-06, np.nan, 0.0],  # 值列表，包含 NaN
            [0.0, -0.00015, np.nan, 2.3614e-05, np.nan],
        ]

        idx = MultiIndex(
            levels=[[680585148, 680607017], [0.0133]],  # 多级索引的层级定义
            codes=[[0, 1], [-1, 0]],  # 索引代码，指定层级值的索引
            names=["s_id", "dosage"],  # 索引名称定义
        )

        cols = MultiIndex(
            levels=[["change"], ["Ag", "Hg", "Pb", "Sn", "U"]],  # 多级列索引的层级定义
            codes=[[0, 0, 0, 0, 0], [0, 1, 2, 3, 4]],  # 列索引代码，指定层级值的索引
            names=[None, "agent"],  # 列索引名称定义
        )

        right = DataFrame(vals, columns=cols, index=idx)
        # 创建 DataFrame right，使用指定的索引和列索引，赋值给 right

        tm.assert_frame_equal(left, right)
        # 使用测试框架中的 assert_frame_equal 检查 left 和 right 是否相等

        left = df.loc[17264:].copy().set_index(["s_id", "dosage", "agent"])
        # 将 DataFrame 按条件选择后，再进行多级索引的堆叠操作，并赋值给 left
        tm.assert_frame_equal(left.unstack(), right)
        # 使用测试框架中的 assert_frame_equal 检查 left 的堆叠结果和 right 是否相等

    def test_unstack_nan_index5(self):
        # GH9497 - 测试用例名称，引用 GitHub issue #9497，处理带有空值的多重堆叠
        df = DataFrame(
            {
                "1st": [1, 2, 1, 2, 1, 2],
                "2nd": date_range("2014-02-01", periods=6, freq="D"),
                "jim": 100 + np.arange(6),
                "joe": (np.random.default_rng(2).standard_normal(6) * 10).round(2),
            }
        )

        df["3rd"] = df["2nd"] - pd.Timestamp("2014-02-02")
        df.loc[1, "2nd"] = df.loc[3, "2nd"] = np.nan
        df.loc[1, "3rd"] = df.loc[4, "3rd"] = np.nan
        # 设置 DataFrame df 的列 "2nd" 和 "3rd" 的部分值为 NaN

        left = df.set_index(["1st", "2nd", "3rd"]).unstack(["2nd", "3rd"])
        # 将 DataFrame 按多列进行多级索引的堆叠操作，并赋值给 left

        assert left.notna().values.sum() == 2 * len(df)
        # 使用断言检查 left 中非 NaN 值的数量是否等于 2 倍的 DataFrame df 的长度

        for col in ["jim", "joe"]:
            for _, r in df.iterrows():
                key = r["1st"], (col, r["2nd"], r["3rd"])
                assert r[col] == left.loc[key]
                # 使用断言逐行检查 DataFrame df 中的值与 left 中对应位置的值是否相等

    def test_stack_datetime_column_multiIndex(self, future_stack):
        # GH 8039 - 测试用例名称，引用 GitHub issue #8039，测试带有日期时间列的多级堆叠
        t = datetime(2014, 1, 1)
        df = DataFrame([1, 2, 3, 4], columns=MultiIndex.from_tuples([(t, "A", "B")]))
        # 创建具有日期时间列的 DataFrame df

        warn = None if future_stack else FutureWarning
        msg = "The previous implementation of stack is deprecated"
        # 设置警告信息，用于将来的堆叠操作

        with tm.assert_produces_warning(warn, match=msg):
            result = df.stack(future_stack=future_stack)
            # 使用将来的堆叠操作对 DataFrame df 进行堆叠，并将结果赋值给 result

        eidx = MultiIndex.from_product([(0, 1, 2, 3), ("B",)])
        ecols = MultiIndex.from_tuples([(t, "A")])
        expected = DataFrame([1, 2, 3, 4], index=eidx, columns=ecols)
        # 创建预期的 DataFrame expected，包含指定的索引和列索引

        tm.assert_frame_equal(result, expected)
        # 使用测试框架中的 assert_frame_equal 检查 result 和 expected 是否相等

    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    # 使用 pytest 的 mark 注释来忽略特定警告
    # 使用 pytest 的参数化装饰器，为测试方法 test_stack_partial_multiIndex 提供多组参数进行测试
    @pytest.mark.parametrize(
        "multiindex_columns",
        [
            [0, 1, 2, 3, 4],     # 测试用例：列索引为 [0, 1, 2, 3, 4]
            [0, 1, 2, 3],        # 测试用例：列索引为 [0, 1, 2, 3]
            [0, 1, 2, 4],        # 测试用例：列索引为 [0, 1, 2, 4]
            [0, 1, 2],           # 测试用例：列索引为 [0, 1, 2]
            [1, 2, 3],           # 测试用例：列索引为 [1, 2, 3]
            [2, 3, 4],           # 测试用例：列索引为 [2, 3, 4]
            [0, 1],              # 测试用例：列索引为 [0, 1]
            [0, 2],              # 测试用例：列索引为 [0, 2]
            [0, 3],              # 测试用例：列索引为 [0, 3]
            [0],                 # 测试用例：列索引为 [0]
            [2],                 # 测试用例：列索引为 [2]
            [4],                 # 测试用例：列索引为 [4]
            [4, 3, 2, 1, 0],     # 测试用例：列索引为 [4, 3, 2, 1, 0]
            [3, 2, 1, 0],        # 测试用例：列索引为 [3, 2, 1, 0]
            [4, 2, 1, 0],        # 测试用例：列索引为 [4, 2, 1, 0]
            [2, 1, 0],           # 测试用例：列索引为 [2, 1, 0]
            [3, 2, 1],           # 测试用例：列索引为 [3, 2, 1]
            [4, 3, 2],           # 测试用例：列索引为 [4, 3, 2]
            [1, 0],              # 测试用例：列索引为 [1, 0]
            [2, 0],              # 测试用例：列索引为 [2, 0]
            [3, 0],              # 测试用例：列索引为 [3, 0]
        ],
    )
    # 使用 pytest 的参数化装饰器，为测试方法 test_stack_partial_multiIndex 提供多组参数进行测试
    @pytest.mark.parametrize("level", (-1, 0, 1, [0, 1], [1, 0]))
    def test_stack_partial_multiIndex(self, multiindex_columns, level, future_stack):
        # GH 8844
        # 根据条件设置 dropna 变量，如果 future_stack 不存在则设置为 False
        dropna = False if not future_stack else lib.no_default
        # 创建一个完整的多级索引对象 full_multiindex
        full_multiindex = MultiIndex.from_tuples(
            [("B", "x"), ("B", "z"), ("A", "y"), ("C", "x"), ("C", "u")],
            names=["Upper", "Lower"],
        )
        # 根据给定的列索引列表 multiindex_columns，从完整的多级索引对象中选取子集，创建 multiindex
        multiindex = full_multiindex[multiindex_columns]
        # 创建一个 DataFrame 对象 df，使用 NumPy 生成数据，列索引为 multiindex
        df = DataFrame(
            np.arange(3 * len(multiindex)).reshape(3, len(multiindex)),
            columns=multiindex,
        )
        # 执行 DataFrame 的 stack 操作，生成 result 结果
        result = df.stack(level=level, dropna=dropna, future_stack=future_stack)

        if isinstance(level, int) and not future_stack:
            # 如果 level 是整数且 future_stack 不存在
            # 预期的结果 expected 为使用 dropna=True 进行的 stack 操作结果
            expected = df.stack(level=level, dropna=True, future_stack=future_stack)
            # 断言 result 与 expected 相等，根据返回类型选择使用 assert_series_equal 或 assert_frame_equal
            if isinstance(expected, Series):
                tm.assert_series_equal(result, expected)
            else:
                tm.assert_frame_equal(result, expected)

        # 将 DataFrame df 的列索引转换为多级索引格式，并执行 stack 操作，生成 expected 结果
        df.columns = MultiIndex.from_tuples(
            df.columns.to_numpy(), names=df.columns.names
        )
        # 根据 dropna 和 future_stack 执行 DataFrame 的 stack 操作，生成 expected 结果
        expected = df.stack(level=level, dropna=dropna, future_stack=future_stack)
        # 断言 result 与 expected 相等，根据返回类型选择使用 assert_series_equal 或 assert_frame_equal
        if isinstance(expected, Series):
            tm.assert_series_equal(result, expected)
        else:
            tm.assert_frame_equal(result, expected)
    
    # 使用 pytest 的 filterwarnings 装饰器，忽略特定警告信息
    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    # 定义一个测试方法，用于测试多层索引的情况下堆叠操作的正确性
    def test_stack_full_multiIndex(self, future_stack):
        # GH 8844
        # 创建一个多层索引对象，包含指定的层级和名称
        full_multiindex = MultiIndex.from_tuples(
            [("B", "x"), ("B", "z"), ("A", "y"), ("C", "x"), ("C", "u")],
            names=["Upper", "Lower"],
        )
        # 根据给定的多层索引创建一个数据帧
        df = DataFrame(np.arange(6).reshape(2, 3), columns=full_multiindex[[0, 1, 3]])
        # 根据是否有未来堆叠的需求，决定是否设置 dropna 参数
        dropna = False if not future_stack else lib.no_default
        # 进行堆叠操作，传递 dropna 和 future_stack 参数
        result = df.stack(dropna=dropna, future_stack=future_stack)
        # 创建期望的数据帧，用于与结果进行比较
        expected = DataFrame(
            [[0, 2], [1, np.nan], [3, 5], [4, np.nan]],
            index=MultiIndex(
                levels=[[0, 1], ["u", "x", "y", "z"]],
                codes=[[0, 0, 1, 1], [1, 3, 1, 3]],
                names=[None, "Lower"],
            ),
            columns=Index(["B", "C"], name="Upper"),
        )
        # 将 "B" 列的数据类型转换为与 df 第一列相同的类型
        expected["B"] = expected["B"].astype(df.dtypes.iloc[0])
        # 使用测试框架比较结果和期望值，确保它们一致
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    @pytest.mark.parametrize("ordered", [False, True])
    # 定义测试方法，验证在保留分类数据类型时进行堆叠操作的正确性
    def test_stack_preserve_categorical_dtype(self, ordered, future_stack):
        # GH13854
        # 创建一个有序或无序的分类索引对象
        cidx = pd.CategoricalIndex(list("yxz"), categories=list("xyz"), ordered=ordered)
        # 根据分类索引创建一个数据帧
        df = DataFrame([[10, 11, 12]], columns=cidx)
        # 执行堆叠操作，传递 future_stack 参数
        result = df.stack(future_stack=future_stack)

        # 使用 `MultiIndex.from_product` 创建预期的系列对象，保留分类数据类型
        midx = MultiIndex.from_product([df.index, cidx])
        expected = Series([10, 11, 12], index=midx)

        # 使用测试框架比较结果和期望值，确保它们一致
        tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    @pytest.mark.parametrize("ordered", [False, True])
    @pytest.mark.parametrize(
        "labels,data",
        [
            (list("xyz"), [10, 11, 12, 13, 14, 15]),
            (list("zyx"), [14, 15, 12, 13, 10, 11]),
        ],
    )
    # 定义测试方法，验证在多层分类数据类型保留时进行堆叠操作的正确性
    def test_stack_multi_preserve_categorical_dtype(
        self, ordered, labels, data, future_stack
    ):
        # GH-36991
        # 创建两个分类索引对象
        cidx = pd.CategoricalIndex(labels, categories=sorted(labels), ordered=ordered)
        cidx2 = pd.CategoricalIndex(["u", "v"], ordered=ordered)
        # 使用两个分类索引创建一个多层索引对象
        midx = MultiIndex.from_product([cidx, cidx2])
        # 根据多层索引创建一个数据帧
        df = DataFrame([sorted(data)], columns=midx)
        # 执行堆叠操作，传递 future_stack 参数
        result = df.stack([0, 1], future_stack=future_stack)

        # 根据 future_stack 的值确定预期的标签顺序和数据
        labels = labels if future_stack else sorted(labels)
        s_cidx = pd.CategoricalIndex(labels, ordered=ordered)
        expected_data = sorted(data) if future_stack else data
        expected = Series(
            expected_data, index=MultiIndex.from_product([[0], s_cidx, cidx2])
        )

        # 使用测试框架比较结果和期望值，确保它们一致
        tm.assert_series_equal(result, expected)
    def test_stack_preserve_categorical_dtype_values(self, future_stack):
        # GH-23077
        # 创建一个包含类别数据的Categorical对象
        cat = pd.Categorical(["a", "a", "b", "c"])
        # 创建一个DataFrame，其中'A'和'B'列都使用上面创建的Categorical对象
        df = DataFrame({"A": cat, "B": cat})
        # 对DataFrame进行stack操作，使用future_stack参数
        result = df.stack(future_stack=future_stack)
        # 创建一个期望的Series对象，其索引是MultiIndex
        index = MultiIndex.from_product([[0, 1, 2, 3], ["A", "B"]])
        expected = Series(
            pd.Categorical(["a", "a", "a", "a", "b", "b", "c", "c"]), index=index
        )
        # 断言结果Series与期望的Series相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    @pytest.mark.parametrize(
        "index",
        [
            [0, 0, 1, 1],
            [0, 0, 2, 3],
            [0, 1, 2, 3],
        ],
    )
    def test_stack_multi_columns_non_unique_index(self, index, future_stack):
        # GH-28301
        # 创建一个包含MultiIndex的DataFrame对象，使用给定的index和columns
        columns = MultiIndex.from_product([[1, 2], ["a", "b"]])
        df = DataFrame(index=index, columns=columns).fillna(1)
        # 对DataFrame进行stack操作，使用future_stack参数
        stacked = df.stack(future_stack=future_stack)
        # 将stacked对象的索引转换为新的MultiIndex对象
        new_index = MultiIndex.from_tuples(stacked.index.to_numpy())
        # 创建一个期望的DataFrame，其数据和索引与stacked相同
        expected = DataFrame(
            stacked.to_numpy(), index=new_index, columns=stacked.columns
        )
        # 断言stacked与期望的DataFrame相等
        tm.assert_frame_equal(stacked, expected)
        # 检查stacked对象的索引编码与期望的编码是否相等
        stacked_codes = np.asarray(stacked.index.codes)
        expected_codes = np.asarray(new_index.codes)
        tm.assert_numpy_array_equal(stacked_codes, expected_codes)

    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    @pytest.mark.parametrize(
        "vals1, vals2, dtype1, dtype2, expected_dtype",
        [
            ([1, 2], [3.0, 4.0], "Int64", "Float64", "Float64"),
            ([1, 2], ["foo", "bar"], "Int64", "string", "object"),
        ],
    )
    def test_stack_multi_columns_mixed_extension_types(
        self, vals1, vals2, dtype1, dtype2, expected_dtype, future_stack
    ):
        # GH45740
        # 创建一个包含多种数据类型的DataFrame对象
        df = DataFrame(
            {
                ("A", 1): Series(vals1, dtype=dtype1),
                ("A", 2): Series(vals2, dtype=dtype2),
            }
        )
        # 对DataFrame进行stack操作，使用future_stack参数
        result = df.stack(future_stack=future_stack)
        # 创建一个期望的DataFrame对象，首先将df转换为object类型再进行stack和类型转换
        expected = (
            df.astype(object).stack(future_stack=future_stack).astype(expected_dtype)
        )
        # 断言结果DataFrame与期望的DataFrame相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("level", [0, 1])
    # 定义一个测试方法，用于测试混合扩展类型的解压操作
    def test_unstack_mixed_extension_types(self, level):
        # 创建一个多级索引对象，包含元组列表和索引名称
        index = MultiIndex.from_tuples([("A", 0), ("A", 1), ("B", 1)], names=["a", "b"])
        # 创建一个数据帧对象，包含两列数据：整数和分类数据
        df = DataFrame(
            {
                "A": pd.array([0, 1, None], dtype="Int64"),
                "B": pd.Categorical(["a", "a", "b"]),
            },
            index=index,
        )

        # 对数据帧进行按指定级别（column level）的解堆叠操作
        result = df.unstack(level=level)
        # 将数据帧转换为对象类型，并按指定级别（column level）解堆叠作为预期结果
        expected = df.astype(object).unstack(level=level)

        # 如果级别为0，则对应的列填充缺失值（pd.NA）
        if level == 0:
            expected[("A", "B")] = expected[("A", "B")].fillna(pd.NA)
        else:
            expected[("A", 0)] = expected[("A", 0)].fillna(pd.NA)

        # 创建一个预期的数据类型序列，用于比较结果的列数据类型
        expected_dtypes = Series(
            [df.A.dtype] * 2 + [df.B.dtype] * 2, index=result.columns
        )
        # 使用测试框架中的方法断言结果的数据类型与预期数据类型一致
        tm.assert_series_equal(result.dtypes, expected_dtypes)
        # 使用测试框架中的方法断言结果数据帧与预期数据帧在转为对象类型后一致
        tm.assert_frame_equal(result.astype(object), expected)

    @pytest.mark.parametrize("level", [0, "baz"])
    # 定义一个参数化测试方法，测试 unstack、swaplevel 和 sortlevel 方法的组合
    def test_unstack_swaplevel_sortlevel(self, level):
        # 创建一个多级索引对象，包含两个级别的元素组合和索引名称
        mi = MultiIndex.from_product([[0], ["d", "c"]], names=["bar", "baz"])
        # 创建一个数据帧对象，包含两行两列的数据，列名为 foo
        df = DataFrame([[0, 2], [1, 3]], index=mi, columns=["B", "A"])
        df.columns.name = "foo"

        # 创建预期的数据帧对象，包含一行四列数据，列名为 baz 和 foo
        expected = DataFrame(
            [[3, 1, 2, 0]],
            columns=MultiIndex.from_tuples(
                [("c", "A"), ("c", "B"), ("d", "A"), ("d", "B")], names=["baz", "foo"]
            ),
        )
        expected.index.name = "bar"

        # 对数据帧进行 unstack 操作后，交换列级别并按指定级别（level）排序
        result = df.unstack().swaplevel(axis=1).sort_index(axis=1, level=level)
        # 使用测试框架中的方法断言结果数据帧与预期数据帧一致
        tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize("dtype", ["float64", "Float64"])
# 使用 pytest 的 parametrize 装饰器，为测试函数 test_unstack_sort_false 提供不同的参数组合
def test_unstack_sort_false(frame_or_series, dtype):
    # GH 15105
    # 创建一个多级索引对象，包含四个元组作为索引值
    index = MultiIndex.from_tuples(
        [("two", "z", "b"), ("two", "y", "a"), ("one", "z", "b"), ("one", "y", "a")]
    )
    # 根据给定的数据、索引和数据类型创建一个 DataFrame 或 Series 对象
    obj = frame_or_series(np.arange(1.0, 5.0), index=index, dtype=dtype)

    # 对对象进行 unstack 操作，根据 level 参数展开，不进行排序
    result = obj.unstack(level=0, sort=False)

    if frame_or_series is DataFrame:
        # 如果对象是 DataFrame，则设置预期的列名为多级索引
        expected_columns = MultiIndex.from_tuples([(0, "two"), (0, "one")])
    else:
        # 否则设置预期的列名为列表
        expected_columns = ["two", "one"]
    # 创建预期的 DataFrame 对象，包含特定的数据、索引和列名
    expected = DataFrame(
        [[1.0, 3.0], [2.0, 4.0]],
        index=MultiIndex.from_tuples([("z", "b"), ("y", "a")]),
        columns=expected_columns,
        dtype=dtype,
    )
    # 断言结果与预期是否一致
    tm.assert_frame_equal(result, expected)

    # 继续对对象进行 unstack 操作，根据 level 参数展开，不进行排序
    result = obj.unstack(level=-1, sort=False)

    if frame_or_series is DataFrame:
        # 如果对象是 DataFrame，则设置预期的列名为多级索引
        expected_columns = MultiIndex.from_tuples([(0, "b"), (0, "a")])
    else:
        # 否则设置预期的列名为列表
        expected_columns = ["b", "a"]
    # 创建预期的 DataFrame 对象，包含特定的数据、索引和列名
    expected = DataFrame(
        [[1.0, np.nan], [np.nan, 2.0], [3.0, np.nan], [np.nan, 4.0]],
        columns=expected_columns,
        index=MultiIndex.from_tuples(
            [("two", "z"), ("two", "y"), ("one", "z"), ("one", "y")]
        ),
        dtype=dtype,
    )
    # 断言结果与预期是否一致
    tm.assert_frame_equal(result, expected)

    # 继续对对象进行 unstack 操作，根据 level 参数展开，不进行排序
    result = obj.unstack(level=[1, 2], sort=False)

    if frame_or_series is DataFrame:
        # 如果对象是 DataFrame，则设置预期的列名为多级索引
        expected_columns = MultiIndex.from_tuples([(0, "z", "b"), (0, "y", "a")])
    else:
        # 否则设置预期的列名为多级索引
        expected_columns = MultiIndex.from_tuples([("z", "b"), ("y", "a")])
    # 创建预期的 DataFrame 对象，包含特定的数据、索引和列名
    expected = DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=["two", "one"],
        columns=expected_columns,
        dtype=dtype,
    )
    # 断言结果与预期是否一致
    tm.assert_frame_equal(result, expected)


def test_unstack_fill_frame_object():
    # GH12815 Test unstacking with object.
    # 创建一个包含对象类型数据的 Series 对象，设置其索引为多级索引
    data = Series(["a", "b", "c", "a"], dtype="object")
    data.index = MultiIndex.from_tuples(
        [("x", "a"), ("x", "b"), ("y", "b"), ("z", "a")]
    )

    # 默认情况下，缺失值将填充为 NaN
    result = data.unstack()
    # 创建预期的 DataFrame 对象，包含特定的数据、索引和数据类型
    expected = DataFrame(
        {"a": ["a", np.nan, "a"], "b": ["b", "c", np.nan]},
        index=list("xyz"),
        dtype=object,
    )
    # 断言结果与预期是否一致
    tm.assert_frame_equal(result, expected)

    # 使用指定的值填充缺失值
    result = data.unstack(fill_value="d")
    # 创建预期的 DataFrame 对象，包含特定的数据、索引和数据类型
    expected = DataFrame(
        {"a": ["a", "d", "a"], "b": ["b", "c", "d"]}, index=list("xyz"), dtype=object
    )
    # 断言结果与预期是否一致
    tm.assert_frame_equal(result, expected)


def test_unstack_timezone_aware_values():
    # GH 18338
    # 创建一个 DataFrame 对象，包含时间戳列和其他列
    df = DataFrame(
        {
            "timestamp": [pd.Timestamp("2017-08-27 01:00:00.709949+0000", tz="UTC")],
            "a": ["a"],
            "b": ["b"],
            "c": ["c"],
        },
        columns=["timestamp", "a", "b", "c"],
    )
    # 对 DataFrame 对象进行索引设置，并进行 unstack 操作
    result = df.set_index(["a", "b"]).unstack()
    # 创建预期的 DataFrame 对象，包含一个具有时区的时间戳和一个字符串列
    expected = DataFrame(
        [[pd.Timestamp("2017-08-27 01:00:00.709949+0000", tz="UTC"), "c"]],
        index=Index(["a"], name="a"),  # 设置 DataFrame 的索引为一个单索引，名称为 "a"
        columns=MultiIndex(  # 使用 MultiIndex 创建多级列索引
            levels=[["timestamp", "c"], ["b"]],  # 第一级列名为 "timestamp" 和 "c"，第二级为 "b"
            codes=[[0, 1], [0, 0]],  # 指定每个级别的列的编码，对应到 levels 中的索引
            names=[None, "b"],  # 第二级列索引的名称为 "b"
        ),
    )
    # 使用测试工具函数 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
@pytest.mark.filterwarnings("ignore:The previous implementation of stack is deprecated")
# 使用 pytest 的标记来忽略特定的警告消息

def test_stack_timezone_aware_values(future_stack):
    # GH 19420
    # 创建一个日期范围，包含时区信息的时间戳
    ts = date_range(freq="D", start="20180101", end="20180103", tz="America/New_York")
    # 创建一个 DataFrame，其中包含一个列'A'，索引为['a', 'b', 'c']，值为时间戳
    df = DataFrame({"A": ts}, index=["a", "b", "c"])
    # 使用 stack 方法堆叠 DataFrame
    result = df.stack(future_stack=future_stack)
    # 创建预期的 Series 对象，其中包含时间戳，并设置了 MultiIndex
    expected = Series(
        ts,
        index=MultiIndex(levels=[["a", "b", "c"], ["A"]], codes=[[0, 1, 2], [0, 0, 0]]),
    )
    # 使用 pytest 的断言方法验证结果与预期是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.filterwarnings("ignore:The previous implementation of stack is deprecated")
@pytest.mark.parametrize("dropna", [True, False, lib.no_default])
# 使用 pytest 的参数化标记，测试不同的 dropna 参数值，包括 lib.no_default

def test_stack_empty_frame(dropna, future_stack):
    # GH 36113
    # 创建一个空的 MultiIndex，并期望得到一个空的 Series 对象
    levels = [np.array([], dtype=np.int64), np.array([], dtype=np.int64)]
    expected = Series(dtype=np.float64, index=MultiIndex(levels=levels, codes=[[], []]))
    # 如果 future_stack 为真且 dropna 不是 lib.no_default，则测试是否引发 ValueError 异常
    if future_stack and dropna is not lib.no_default:
        with pytest.raises(ValueError, match="dropna must be unspecified"):
            # 使用 stack 方法堆叠空的 DataFrame，并指定 dropna 参数
            DataFrame(dtype=np.float64).stack(dropna=dropna, future_stack=future_stack)
    else:
        # 否则，执行堆叠操作，并验证结果是否与预期相等
        result = DataFrame(dtype=np.float64).stack(
            dropna=dropna, future_stack=future_stack
        )
        tm.assert_series_equal(result, expected)


@pytest.mark.filterwarnings("ignore:The previous implementation of stack is deprecated")
@pytest.mark.parametrize("dropna", [True, False, lib.no_default])
@pytest.mark.parametrize("fill_value", [None, 0])
# 使用 pytest 的参数化标记，测试不同的 dropna 和 fill_value 参数组合

def test_stack_unstack_empty_frame(dropna, fill_value, future_stack):
    # GH 36113
    # 如果 future_stack 为真且 dropna 不是 lib.no_default，则测试是否引发 ValueError 异常
    if future_stack and dropna is not lib.no_default:
        with pytest.raises(ValueError, match="dropna must be unspecified"):
            # 先堆叠再展开 DataFrame，并指定 fill_value 参数
            DataFrame(dtype=np.int64).stack(
                dropna=dropna, future_stack=future_stack
            ).unstack(fill_value=fill_value)
    else:
        # 否则，执行堆叠和展开操作，并验证结果是否与预期相等
        result = (
            DataFrame(dtype=np.int64)
            .stack(dropna=dropna, future_stack=future_stack)
            .unstack(fill_value=fill_value)
        )
        expected = DataFrame(dtype=np.int64)
        tm.assert_frame_equal(result, expected)


def test_unstack_single_index_series():
    # GH 36113
    # 准备一个匹配特定 ValueError 异常消息的正则表达式
    msg = r"index must be a MultiIndex to unstack.*"
    # 测试是否在调用 unstack 方法时引发特定异常
    with pytest.raises(ValueError, match=msg):
        Series(dtype=np.int64).unstack()


def test_unstacking_multi_index_df():
    # see gh-30740
    # 创建一个包含多级索引的 DataFrame
    df = DataFrame(
        {
            "name": ["Alice", "Bob"],
            "score": [9.5, 8],
            "employed": [False, True],
            "kids": [0, 0],
            "gender": ["female", "male"],
        }
    )
    # 将指定列设置为索引，创建多级索引 DataFrame
    df = df.set_index(["name", "employed", "kids", "gender"])
    # 使用 unstack 方法展开 DataFrame，并使用 fill_value 参数填充缺失值
    df = df.unstack(["gender"], fill_value=0)
    # 创建预期的展开结果，并验证实际结果与预期结果是否相等
    expected = df.unstack("employed", fill_value=0).unstack("kids", fill_value=0)
    result = df.unstack(["employed", "kids"], fill_value=0)
    # 创建预期的 DataFrame 对象，包含两行数据，四列，每列分别代表不同的性别和就业状况
    expected = DataFrame(
        [[9.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 8.0]],
        index=Index(["Alice", "Bob"], name="name"),  # 设置行索引为姓名，命名为'name'
        columns=MultiIndex.from_tuples(
            [
                ("score", "female", False, 0),  # 多级列索引的第一列
                ("score", "female", True, 0),   # 多级列索引的第二列
                ("score", "male", False, 0),    # 多级列索引的第三列
                ("score", "male", True, 0),     # 多级列索引的第四列
            ],
            names=[None, "gender", "employed", "kids"],  # 设置多级列索引的各级名称
        ),
    )
    
    # 使用测试框架的函数来比较实际结果和预期结果的 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
@pytest.mark.filterwarnings("ignore:The previous implementation of stack is deprecated")
# 定义一个测试函数，用于测试堆叠操作中的位置参数和重复列名
def test_stack_positional_level_duplicate_column_names(future_stack):
    # 创建一个多级索引，其中包含重复的列名 'a'
    # 此处链接到一个 GitHub 问题页面，详细描述了问题
    columns = MultiIndex.from_product([("x", "y"), ("y", "z")], names=["a", "a"])
    # 创建一个 DataFrame，包含单行数据并使用上述多级索引作为列名
    df = DataFrame([[1, 1, 1, 1]], columns=columns)
    # 调用 stack 方法，堆叠数据框，使用 future_stack 参数作为未来的堆叠标志
    result = df.stack(0, future_stack=future_stack)

    # 创建预期的 DataFrame
    new_columns = Index(["y", "z"], name="a")
    new_index = MultiIndex.from_tuples([(0, "x"), (0, "y")], names=[None, "a"])
    expected = DataFrame([[1, 1], [1, 1]], index=new_index, columns=new_columns)

    # 使用测试工具方法验证结果是否符合预期
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于测试非切片块的解堆叠操作
def test_unstack_non_slice_like_blocks():
    # 创建一个多级索引
    mi = MultiIndex.from_product([range(5), ["A", "B", "C"]])
    # 创建一个 DataFrame，包含随机数据，索引为 mi
    df = DataFrame(
        {
            0: np.random.default_rng(2).standard_normal(15),
            1: np.random.default_rng(2).standard_normal(15).astype(np.int64),
            2: np.random.default_rng(2).standard_normal(15),
            3: np.random.default_rng(2).standard_normal(15),
        },
        index=mi,
    )
    # 断言，检查 DataFrame 的底层块是否存在非切片样式的块
    assert any(not x.mgr_locs.is_slice_like for x in df._mgr.blocks)

    # 调用 unstack 方法，对 DataFrame 进行解堆叠操作
    res = df.unstack()

    # 创建预期的 DataFrame
    expected = pd.concat([df[n].unstack() for n in range(4)], keys=range(4), axis=1)
    # 使用测试工具方法验证结果是否符合预期
    tm.assert_frame_equal(res, expected)


@pytest.mark.filterwarnings("ignore:The previous implementation of stack is deprecated")
# 定义一个测试函数，用于测试 stack 方法中 sort 参数为 False 的情况
def test_stack_sort_false(future_stack):
    # 创建一个包含浮点数和 NaN 值的二维列表
    data = [[1, 2, 3.0, 4.0], [2, 3, 4.0, 5.0], [3, 4, np.nan, np.nan]]
    # 创建一个 DataFrame，其中包含多级列名
    df = DataFrame(
        data,
        columns=MultiIndex(
            levels=[["B", "A"], ["x", "y"]], codes=[[0, 0, 1, 1], [0, 1, 0, 1]]
        ),
    )
    # 根据 future_stack 参数的值选择合适的关键字参数
    kwargs = {} if future_stack else {"sort": False}
    # 调用 stack 方法，堆叠数据框，可能包含未来堆叠的标志和排序参数
    result = df.stack(level=0, future_stack=future_stack, **kwargs)
    # 根据 future_stack 参数的值选择预期的 DataFrame
    if future_stack:
        expected = DataFrame(
            {
                "x": [1.0, 3.0, 2.0, 4.0, 3.0, np.nan],
                "y": [2.0, 4.0, 3.0, 5.0, 4.0, np.nan],
            },
            index=MultiIndex.from_arrays(
                [[0, 0, 1, 1, 2, 2], ["B", "A", "B", "A", "B", "A"]]
            ),
        )
    else:
        expected = DataFrame(
            {"x": [1.0, 3.0, 2.0, 4.0, 3.0], "y": [2.0, 4.0, 3.0, 5.0, 4.0]},
            index=MultiIndex.from_arrays([[0, 0, 1, 1, 2], ["B", "A", "B", "A", "B"]]),
        )
    # 使用测试工具方法验证结果是否符合预期
    tm.assert_frame_equal(result, expected)

    # 根据 future_stack 参数的值，重新创建 DataFrame，使用不同的多级列名
    df = DataFrame(
        data,
        columns=MultiIndex.from_arrays([["B", "B", "A", "A"], ["x", "y", "x", "y"]]),
    )
    # 根据 future_stack 参数的值选择合适的关键字参数
    kwargs = {} if future_stack else {"sort": False}
    # 再次调用 stack 方法，堆叠数据框，可能包含未来堆叠的标志和排序参数
    result = df.stack(level=0, future_stack=future_stack, **kwargs)
    # 使用测试工具方法验证结果是否符合预期
    tm.assert_frame_equal(result, expected)


@pytest.mark.filterwarnings("ignore:The previous implementation of stack is deprecated")
# 定义一个测试函数，用于测试多级索引下 stack 方法中 sort 参数为 False 的情况
def test_stack_sort_false_multi_level(future_stack):
    # GH 15105
    # 创建一个多级索引对象，包含 [("weight", "kg"), ("height", "m")]
    idx = MultiIndex.from_tuples([("weight", "kg"), ("height", "m")])
    # 创建一个数据帧对象，包含数据 [[1.0, 2.0], [3.0, 4.0]]，使用上面创建的多级索引作为列索引，["cat", "dog"] 作为行索引
    df = DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["cat", "dog"], columns=idx)
    # 根据条件 future_stack 创建一个空字典或包含 {"sort": False} 的字典
    kwargs = {} if future_stack else {"sort": False}
    # 对数据帧 df 进行堆叠操作，使用 [0, 1] 作为堆叠的级别，同时传递 future_stack 和 kwargs 中的参数
    result = df.stack([0, 1], future_stack=future_stack, **kwargs)
    # 创建一个预期的多级索引对象，包含指定的索引元组
    expected_index = MultiIndex.from_tuples(
        [
            ("cat", "weight", "kg"),
            ("cat", "height", "m"),
            ("dog", "weight", "kg"),
            ("dog", "height", "m"),
        ]
    )
    # 创建一个预期的系列对象，包含指定的数据和多级索引
    expected = Series([1.0, 2.0, 3.0, 4.0], index=expected_index)
    # 使用测试模块中的函数 tm.assert_series_equal 检查 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)
class TestStackUnstackMultiLevel:
    def test_unstack(self, multiindex_year_month_day_dataframe_random_data):
        # 获取传入的多层索引数据框架
        ymd = multiindex_year_month_day_dataframe_random_data

        # 对数据框架进行 unstack 操作，使得索引的最后一层转变为列
        unstacked = ymd.unstack()
        unstacked.unstack()  # 连续进行 unstack 操作，将索引进一步转变为列

        # 测试转换为整数类型后的 unstack 操作
        ymd.astype(int).unstack()

        # 测试转换为 int32 类型后的 unstack 操作
        ymd.astype(np.int32).unstack()

    @pytest.mark.parametrize(
        "result_rows,result_columns,index_product,expected_row",
        [
            (
                [[1, 1, None, None, 30.0, None], [2, 2, None, None, 30.0, None]],
                ["ix1", "ix2", "col1", "col2", "col3", "col4"],
                2,
                [None, None, 30.0, None],
            ),
            (
                [[1, 1, None, None, 30.0], [2, 2, None, None, 30.0]],
                ["ix1", "ix2", "col1", "col2", "col3"],
                2,
                [None, None, 30.0],
            ),
            (
                [[1, 1, None, None, 30.0], [2, None, None, None, 30.0]],
                ["ix1", "ix2", "col1", "col2", "col3"],
                None,
                [None, None, 30.0],
            ),
        ],
    )
    def test_unstack_partial(
        self, result_rows, result_columns, index_product, expected_row
    ):
        # 检查在以下问题上的回归：
        # https://github.com/pandas-dev/pandas/issues/19351
        # 确保当在数据框架的子集上运行 DataFrame.unstack() 时，Index 层包含不在子集中的值时，其仍能正常工作
        result = DataFrame(result_rows, columns=result_columns).set_index(
            ["ix1", "ix2"]
        )
        # 选择数据框架的第 1 行到第 2 行，并对 "ix2" 列进行 unstack 操作
        result = result.iloc[1:2].unstack("ix2")
        # 构建预期结果，用于断言
        expected = DataFrame(
            [expected_row],
            columns=MultiIndex.from_product(
                [result_columns[2:], [index_product]], names=[None, "ix2"]
            ),
            index=Index([2], name="ix1"),
        )
        tm.assert_frame_equal(result, expected)

    def test_unstack_multiple_no_empty_columns(self):
        # 创建多层索引
        index = MultiIndex.from_tuples(
            [(0, "foo", 0), (0, "bar", 0), (1, "baz", 1), (1, "qux", 1)]
        )
        # 创建带索引的 Series
        s = Series(np.random.default_rng(2).standard_normal(4), index=index)

        # 对 Series 进行多层索引的 unstack 操作，删除所有空列
        unstacked = s.unstack([1, 2])
        expected = unstacked.dropna(axis=1, how="all")
        tm.assert_frame_equal(unstacked, expected)

    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    @pytest.mark.parametrize(
        "idx, exp_idx",
        [  # 使用 pytest 的 parametrize 装饰器，定义参数化测试数据
            [
                list("abab"),  # 第一个测试用例的输入参数 idx
                MultiIndex(  # 期望输出的 MultiIndex 对象
                    levels=[["a", "b"], ["1st", "2nd"]],  # MultiIndex 对象的层级标签
                    codes=[np.tile(np.arange(2).repeat(3), 2), np.tile([0, 1, 0], 4)],  # MultiIndex 对象的编码
                ),
            ],
            [
                MultiIndex.from_tuples((("a", 2), ("b", 1), ("a", 1), ("b", 2))),  # 第二个测试用例的输入参数 idx
                MultiIndex(  # 期望输出的 MultiIndex 对象
                    levels=[["a", "b"], [1, 2], ["1st", "2nd"]],  # MultiIndex 对象的层级标签
                    codes=[  # MultiIndex 对象的编码
                        np.tile(np.arange(2).repeat(3), 2),
                        np.repeat([1, 0, 1], [3, 6, 3]),
                        np.tile([0, 1, 0], 4),
                    ],
                ),
            ],
        ],
    )
    def test_stack_duplicate_index(self, idx, exp_idx, future_stack):
        # GH10417  # 标识 GitHub 问题编号
        df = DataFrame(  # 创建 DataFrame 对象
            np.arange(12).reshape(4, 3),  # 使用给定的数据创建 DataFrame 对象
            index=idx,  # 设置 DataFrame 的索引
            columns=["1st", "2nd", "1st"],  # 设置 DataFrame 的列标签，包含重复值
        )
        if future_stack:
            msg = "Columns with duplicate values are not supported in stack"  # 错误消息
            with pytest.raises(ValueError, match=msg):  # 断言引发特定异常
                df.stack(future_stack=future_stack)  # 调用 stack 方法并传递参数
        else:
            result = df.stack(future_stack=future_stack)  # 调用 stack 方法并传递参数
            expected = Series(np.arange(12), index=exp_idx)  # 创建预期的 Series 对象
            tm.assert_series_equal(result, expected)  # 断言两个 Series 对象相等
            assert result.index.is_unique is False  # 断言结果的索引不是唯一的
            li, ri = result.index, expected.index  # 获取结果和预期索引
            tm.assert_index_equal(li, ri)  # 断言两个索引相等

    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    def test_unstack_odd_failure(self, future_stack):
        mi = MultiIndex.from_arrays(  # 使用数组创建 MultiIndex 对象
            [
                ["Fri"] * 4 + ["Sat"] * 2 + ["Sun"] * 2 + ["Thu"] * 3,  # 设置 MultiIndex 对象的第一个级别
                ["Dinner"] * 2 + ["Lunch"] * 2 + ["Dinner"] * 5 + ["Lunch"] * 2,  # 设置 MultiIndex 对象的第二个级别
                ["No", "Yes"] * 4 + ["No", "No", "Yes"],  # 设置 MultiIndex 对象的第三个级别
            ],
            names=["day", "time", "smoker"],  # 设置 MultiIndex 对象的级别名称
        )
        df = DataFrame(  # 创建 DataFrame 对象
            {
                "sum": np.arange(11, dtype="float64"),  # 第一列数据
                "len": np.arange(11, dtype="float64"),  # 第二列数据
            },
            index=mi,  # 设置 DataFrame 的索引为创建的 MultiIndex 对象
        )
        # it works, #2100  # 注释说明特定的工作情况或问题编号
        result = df.unstack(2)  # 调用 DataFrame 的 unstack 方法，并传递参数

        recons = result.stack(future_stack=future_stack)  # 调用结果的 stack 方法，并传递参数
        if future_stack:
            # NA values in unstacked persist to restacked in version 3
            recons = recons.dropna(how="all")  # 如果存在 NA 值，则删除所有 NA 值的行
        tm.assert_frame_equal(recons, df)  # 断言两个 DataFrame 对象相等

    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    # 定义测试方法，用于测试具有多级索引和随机数据的 DataFrame
    def test_stack_mixed_dtype(self, multiindex_dataframe_random_data, future_stack):
        # 从参数中获取多级索引的随机数据帧
        frame = multiindex_dataframe_random_data

        # 转置数据帧
        df = frame.T
        # 在列名为 ('foo', 'four') 的列上添加新值 'foo'
        df["foo", "four"] = "foo"
        # 按照第二级索引排序列
        df = df.sort_index(level=1, axis=1)

        # 对整个 DataFrame 进行堆叠操作，使用 future_stack 参数
        stacked = df.stack(future_stack=future_stack)
        # 对列 'foo' 进行堆叠操作，使用 future_stack 参数，并按索引排序
        result = df["foo"].stack(future_stack=future_stack).sort_index()
        # 断言两个堆叠操作的结果是否相等，不检查名称
        tm.assert_series_equal(stacked["foo"], result, check_names=False)
        # 断言结果 Series 的名称为空
        assert result.name is None
        # 断言堆叠后的 'bar' 列的数据类型为 np.float64
        assert stacked["bar"].dtype == np.float64

    # 使用 pytest 标记忽略警告，指示前一个实现的 stack 方法已弃用
    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    # 测试 unstack 方法的 bug
    def test_unstack_bug(self, future_stack):
        # 创建包含不同列的 DataFrame
        df = DataFrame(
            {
                "state": ["naive", "naive", "naive", "active", "active", "active"],
                "exp": ["a", "b", "b", "b", "a", "a"],
                "barcode": [1, 2, 3, 4, 1, 3],
                "v": ["hi", "hi", "bye", "bye", "bye", "peace"],
                "extra": np.arange(6.0),
            }
        )

        # 定义警告消息内容，指示 DataFrameGroupBy.apply 操作了分组列
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        # 使用 pytest 断言产生特定警告类型和匹配消息的警告
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            # 对 DataFrame 按指定列分组，并对每个分组应用 len 函数
            result = df.groupby(["state", "exp", "barcode", "v"]).apply(len)

        # 对结果进行 unstack 操作
        unstacked = result.unstack()
        # 将 unstack 后的结果再次进行堆叠，使用 future_stack 参数
        restacked = unstacked.stack(future_stack=future_stack)
        # 断言重新堆叠后的结果与重新索引后的原始结果相等，并转换为 float 类型
        tm.assert_series_equal(restacked, result.reindex(restacked.index).astype(float))

    # 使用 pytest 标记忽略警告，指示前一个实现的 stack 方法已弃用
    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    # 测试在堆叠和取消堆叠操作中保留索引名称
    def test_stack_unstack_preserve_names(
        self, multiindex_dataframe_random_data, future_stack
    ):
        # 从参数中获取多级索引的随机数据帧
        frame = multiindex_dataframe_random_data

        # 对数据帧进行取消堆叠操作
        unstacked = frame.unstack()
        # 断言取消堆叠后的索引名称为 "first"
        assert unstacked.index.name == "first"
        # 断言取消堆叠后的列名称为 ["exp", "second"]
        assert unstacked.columns.names == ["exp", "second"]

        # 将取消堆叠后的数据帧再次进行堆叠，使用 future_stack 参数
        restacked = unstacked.stack(future_stack=future_stack)
        # 断言重新堆叠后的索引名称与原始数据帧的索引名称相同
        assert restacked.index.names == frame.index.names

    # 使用 pytest 标记忽略警告，指示前一个实现的 stack 方法已弃用
    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    # 使用参数化测试堆叠和取消堆叠方法的错误级别名称
    @pytest.mark.parametrize("method", ["stack", "unstack"])
    def test_stack_unstack_wrong_level_name(
        self, method, multiindex_dataframe_random_data, future_stack
    ):
        # GH 18303 - 测试错误的级别名称是否会引发 KeyError
        frame = multiindex_dataframe_random_data

        # 从数据帧中选择具有平坦轴的 DataFrame
        df = frame.loc["foo"]

        # 根据方法名称选择要调用的方法，并传入 future_stack 参数
        kwargs = {"future_stack": future_stack} if method == "stack" else {}
        # 使用 pytest 断言是否会引发 KeyError，并检查异常消息中是否包含 "does not match index name"
        with pytest.raises(KeyError, match="does not match index name"):
            getattr(df, method)("mistake", **kwargs)

        # 如果方法是 unstack，则在 Series 上进行相同的测试
        if method == "unstack":
            s = df.iloc[:, 0]
            with pytest.raises(KeyError, match="does not match index name"):
                getattr(s, method)("mistake", **kwargs)
    # 定义测试函数，用于测试 DataFrame 的 unstack 方法在指定级别名称上的行为
    def test_unstack_level_name(self, multiindex_dataframe_random_data):
        # 从参数中获取多级索引的随机数据帧
        frame = multiindex_dataframe_random_data

        # 对数据帧使用 unstack 方法，将 "second" 级别展开
        result = frame.unstack("second")
        # 使用 level=1 的方式对比期望结果
        expected = frame.unstack(level=1)
        # 使用测试工具比较实际结果和期望结果
        tm.assert_frame_equal(result, expected)

    # 定义测试函数，测试 DataFrame 的 stack 和 unstack 方法在指定级别名称上的行为
    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    def test_stack_level_name(self, multiindex_dataframe_random_data, future_stack):
        # 从参数中获取多级索引的随机数据帧
        frame = multiindex_dataframe_random_data

        # 对数据帧使用 unstack 方法，将 "second" 级别展开
        unstacked = frame.unstack("second")
        # 对展开后的数据帧使用 stack 方法，在 "exp" 级别叠加数据，考虑未来堆叠的情况
        result = unstacked.stack("exp", future_stack=future_stack)
        # 使用默认参数对整个数据帧使用 unstack 和 stack 方法
        expected = frame.unstack().stack(0, future_stack=future_stack)
        # 使用测试工具比较实际结果和期望结果
        tm.assert_frame_equal(result, expected)

        # 对数据帧直接使用 stack 方法，在 "exp" 级别叠加数据，考虑未来堆叠的情况
        result = frame.stack("exp", future_stack=future_stack)
        # 使用默认参数对整个数据帧使用 stack 方法，考虑未来堆叠的情况
        expected = frame.stack(future_stack=future_stack)
        # 使用测试工具比较实际结果和期望结果
        tm.assert_series_equal(result, expected)

    # 定义测试函数，测试 DataFrame 的 unstack 方法在多个级别名称上的行为
    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    def test_stack_unstack_multiple(
        self, multiindex_year_month_day_dataframe_random_data, future_stack
    ):
        # 从参数中获取多级索引的年月日随机数据帧
        ymd = multiindex_year_month_day_dataframe_random_data

        # 对数据帧使用 unstack 方法，将 ["year", "month"] 级别展开
        unstacked = ymd.unstack(["year", "month"])
        # 使用两次单独的 unstack 方法，对比期望结果
        expected = ymd.unstack("year").unstack("month")
        # 使用测试工具比较实际结果和期望结果
        tm.assert_frame_equal(unstacked, expected)
        # 断言展开后的列名与期望结果一致
        assert unstacked.columns.names == expected.columns.names

        # 对 Series 数据进行测试
        s = ymd["A"]
        # 对 Series 使用 unstack 方法，将 ["year", "month"] 级别展开
        s_unstacked = s.unstack(["year", "month"])
        # 使用测试工具比较实际结果和期望结果
        tm.assert_frame_equal(s_unstacked, expected["A"])

        # 对展开后的数据使用 stack 方法，在 ["year", "month"] 级别叠加数据，考虑未来堆叠的情况
        restacked = unstacked.stack(["year", "month"], future_stack=future_stack)
        # 如果考虑未来堆叠，则在版本 3 中，未展开的 NA 值会保留在重新堆叠后的结果中
        if future_stack:
            restacked = restacked.dropna(how="all")
        # 重新排序索引级别
        restacked = restacked.swaplevel(0, 1).swaplevel(1, 2)
        restacked = restacked.sort_index(level=0)

        # 使用测试工具比较重新堆叠后的结果和原始数据帧
        tm.assert_frame_equal(restacked, ymd)
        # 断言重新堆叠后的索引名与原始数据帧一致
        assert restacked.index.names == ymd.index.names

        # 对数据帧使用 unstack 方法，将 [1, 2] 级别展开
        unstacked = ymd.unstack([1, 2])
        # 使用两次单独的 unstack 方法，删除所有行和列的 NA 值，对比期望结果
        expected = ymd.unstack(1).unstack(1).dropna(axis=1, how="all")
        # 使用测试工具比较实际结果和期望结果
        tm.assert_frame_equal(unstacked, expected)

        # 对数据帧使用 unstack 方法，将 [2, 1] 级别展开
        unstacked = ymd.unstack([2, 1])
        # 使用两次单独的 unstack 方法，删除所有列的 NA 值，对比期望结果
        expected = ymd.unstack(2).unstack(1).dropna(axis=1, how="all")
        # 只保留与实际结果列名匹配的期望结果列
        tm.assert_frame_equal(unstacked, expected.loc[:, unstacked.columns])

    # 定义测试函数，测试 DataFrame 的 unstack 方法在混合级别名称和数字的情况下的行为
    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    def test_stack_names_and_numbers(
        self, multiindex_year_month_day_dataframe_random_data, future_stack
    ):
        # 从参数中获取多级索引的年月日随机数据帧
        ymd = multiindex_year_month_day_dataframe_random_data

        # 对数据帧使用 unstack 方法，将 ["year", "month"] 级别展开
        unstacked = ymd.unstack(["year", "month"])

        # 尝试在堆叠时使用混合级别名称和数字，预期会引发 ValueError 异常
        with pytest.raises(ValueError, match="level should contain"):
            unstacked.stack([0, "month"], future_stack=future_stack)
    # 标记此测试函数，忽略关于堆栈先前实现已弃用的警告
    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    # 测试多层索引 DataFrame 的堆栈操作，使用随机数据和给定的 future_stack 参数
    def test_stack_multiple_out_of_bounds(
        self, multiindex_year_month_day_dataframe_random_data, future_stack
    ):
        # 获取多层索引 DataFrame 对象
        ymd = multiindex_year_month_day_dataframe_random_data

        # 对 DataFrame 进行 "year" 和 "month" 两个级别的解堆栈操作
        unstacked = ymd.unstack(["year", "month"])

        # 使用 pytest 检查是否会抛出 IndexError 异常，并验证错误消息是否包含 "Too many levels"
        with pytest.raises(IndexError, match="Too many levels"):
            unstacked.stack([2, 3], future_stack=future_stack)
        # 使用 pytest 检查是否会抛出 IndexError 异常，并验证错误消息是否包含 "not a valid level number"
        with pytest.raises(IndexError, match="not a valid level number"):
            unstacked.stack([-4, -3], future_stack=future_stack)

    # 测试针对 PeriodIndex 类型的 Series 对象的解堆栈操作
    def test_unstack_period_series(self):
        # GH4342
        # 创建一个 PeriodIndex 对象，频率为月份，名称为 "period"
        idx1 = pd.PeriodIndex(
            ["2013-01", "2013-01", "2013-02", "2013-02", "2013-03", "2013-03"],
            freq="M",
            name="period",
        )
        # 创建一个 Index 对象，包含字符串 "A" 和 "B"，名称为 "str"
        idx2 = Index(["A", "B"] * 3, name="str")
        value = [1, 2, 3, 4, 5, 6]

        # 使用 MultiIndex.from_arrays() 创建多层索引对象 idx
        idx = MultiIndex.from_arrays([idx1, idx2])
        # 创建 Series 对象 s，使用上面创建的多层索引 idx 和数值 value
        s = Series(value, index=idx)

        # 对 Series 进行默认的解堆栈操作，返回结果 result1
        result1 = s.unstack()
        # 对 Series 进行 level=1 的解堆栈操作，返回结果 result2
        result2 = s.unstack(level=1)
        # 对 Series 进行 level=0 的解堆栈操作，返回结果 result3
        result3 = s.unstack(level=0)

        # 创建一个期望的 DataFrame 对象 expected，包含指定的索引和列
        e_idx = pd.PeriodIndex(
            ["2013-01", "2013-02", "2013-03"], freq="M", name="period"
        )
        expected = DataFrame(
            {"A": [1, 3, 5], "B": [2, 4, 6]}, index=e_idx, columns=["A", "B"]
        )
        expected.columns.name = "str"

        # 使用 pandas 测试工具（tm.assert_frame_equal）验证 result1 与 expected 是否相等
        tm.assert_frame_equal(result1, expected)
        # 使用 pandas 测试工具（tm.assert_frame_equal）验证 result2 与 expected 是否相等
        tm.assert_frame_equal(result2, expected)
        # 使用 pandas 测试工具（tm.assert_frame_equal）验证 result3 与 expected 转置后是否相等
        tm.assert_frame_equal(result3, expected.T)

        # 创建另一个 PeriodIndex 对象 idx1，使用不同的日期和名称 "period1"
        idx1 = pd.PeriodIndex(
            ["2013-01", "2013-01", "2013-02", "2013-02", "2013-03", "2013-03"],
            freq="M",
            name="period1",
        )

        # 创建另一个 PeriodIndex 对象 idx2，使用不同的日期和名称 "period2"
        idx2 = pd.PeriodIndex(
            ["2013-12", "2013-11", "2013-10", "2013-09", "2013-08", "2013-07"],
            freq="M",
            name="period2",
        )
        # 使用 MultiIndex.from_arrays() 创建多层索引对象 idx
        idx = MultiIndex.from_arrays([idx1, idx2])
        # 创建 Series 对象 s，使用上面创建的多层索引 idx 和数值 value
        s = Series(value, index=idx)

        # 对 Series 进行默认的解堆栈操作，返回结果 result1
        result1 = s.unstack()
        # 对 Series 进行 level=1 的解堆栈操作，返回结果 result2
        result2 = s.unstack(level=1)
        # 对 Series 进行 level=0 的解堆栈操作，返回结果 result3
        result3 = s.unstack(level=0)

        # 创建一个期望的 DataFrame 对象 expected，包含指定的索引和列
        e_idx = pd.PeriodIndex(
            ["2013-01", "2013-02", "2013-03"], freq="M", name="period1"
        )
        e_cols = pd.PeriodIndex(
            ["2013-07", "2013-08", "2013-09", "2013-10", "2013-11", "2013-12"],
            freq="M",
            name="period2",
        )
        expected = DataFrame(
            [
                [np.nan, np.nan, np.nan, np.nan, 2, 1],
                [np.nan, np.nan, 4, 3, np.nan, np.nan],
                [6, 5, np.nan, np.nan, np.nan, np.nan],
            ],
            index=e_idx,
            columns=e_cols,
        )

        # 使用 pandas 测试工具（tm.assert_frame_equal）验证 result1 与 expected 是否相等
        tm.assert_frame_equal(result1, expected)
        # 使用 pandas 测试工具（tm.assert_frame_equal）验证 result2 与 expected 是否相等
        tm.assert_frame_equal(result2, expected)
        # 使用 pandas 测试工具（tm.assert_frame_equal）验证 result3 与 expected 转置后是否相等
        tm.assert_frame_equal(result3, expected.T)
    def test_unstack_period_frame(self):
        # GH4342
        # 创建第一个周期性索引对象，指定频率为月份，名称为period1
        idx1 = pd.PeriodIndex(
            ["2014-01", "2014-02", "2014-02", "2014-02", "2014-01", "2014-01"],
            freq="M",
            name="period1",
        )
        # 创建第二个周期性索引对象，指定频率为月份，名称为period2
        idx2 = pd.PeriodIndex(
            ["2013-12", "2013-12", "2014-02", "2013-10", "2013-10", "2014-02"],
            freq="M",
            name="period2",
        )
        # 创建包含'A'和'B'列数据的字典
        value = {"A": [1, 2, 3, 4, 5, 6], "B": [6, 5, 4, 3, 2, 1]}
        # 使用idx1和idx2创建多重索引对象idx
        idx = MultiIndex.from_arrays([idx1, idx2])
        # 创建DataFrame对象df，指定索引为idx，数据为value
        df = DataFrame(value, index=idx)

        # 对DataFrame进行unstack操作，结果存入result1
        result1 = df.unstack()
        # 对DataFrame进行level=1的unstack操作，结果存入result2
        result2 = df.unstack(level=1)
        # 对DataFrame进行level=0的unstack操作，结果存入result3
        result3 = df.unstack(level=0)

        # 创建期望的周期性索引对象e_1和e_2
        e_1 = pd.PeriodIndex(["2014-01", "2014-02"], freq="M", name="period1")
        e_2 = pd.PeriodIndex(
            ["2013-10", "2013-12", "2014-02", "2013-10", "2013-12", "2014-02"],
            freq="M",
            name="period2",
        )
        # 创建期望的多重索引列对象e_cols
        e_cols = MultiIndex.from_arrays(["A A A B B B".split(), e_2])
        # 创建期望的DataFrame对象expected，指定数据和索引
        expected = DataFrame(
            [[5, 1, 6, 2, 6, 1], [4, 2, 3, 3, 5, 4]], index=e_1, columns=e_cols
        )

        # 使用assert_frame_equal断言result1和expected是否相等
        tm.assert_frame_equal(result1, expected)
        # 使用assert_frame_equal断言result2和expected是否相等
        tm.assert_frame_equal(result2, expected)

        # 创建期望的周期性索引对象e_1和e_2
        e_1 = pd.PeriodIndex(
            ["2014-01", "2014-02", "2014-01", "2014-02"], freq="M", name="period1"
        )
        e_2 = pd.PeriodIndex(
            ["2013-10", "2013-12", "2014-02"], freq="M", name="period2"
        )
        # 创建期望的多重索引列对象e_cols
        e_cols = MultiIndex.from_arrays(["A A B B".split(), e_1])
        # 创建期望的DataFrame对象expected，指定数据和索引
        expected = DataFrame(
            [[5, 4, 2, 3], [1, 2, 6, 5], [6, 3, 1, 4]], index=e_2, columns=e_cols
        )

        # 使用assert_frame_equal断言result3和expected是否相等
        tm.assert_frame_equal(result3, expected)

    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    def test_stack_multiple_bug(self, future_stack):
        # bug when some uniques are not present in the data GH#3170
        # 创建包含ID、NAME、DATE和VAR1列数据的DataFrame对象df
        id_col = ([1] * 3) + ([2] * 3)
        name = (["a"] * 3) + (["b"] * 3)
        date = pd.to_datetime(["2013-01-03", "2013-01-04", "2013-01-05"] * 2)
        var1 = np.random.default_rng(2).integers(0, 100, 6)
        df = DataFrame({"ID": id_col, "NAME": name, "DATE": date, "VAR1": var1})

        # 将DATE和ID设置为多重索引multi，并设置列名为Params
        multi = df.set_index(["DATE", "ID"])
        multi.columns.name = "Params"
        # 对multi进行unstack操作，结果存入unst
        unst = multi.unstack("ID")
        # 设置预期的错误消息
        msg = re.escape("agg function failed [how->mean,dtype->")
        # 使用pytest.raises断言在执行unst.resample("W-THU").mean()时会抛出TypeError，并匹配msg
        with pytest.raises(TypeError, match=msg):
            unst.resample("W-THU").mean()
        # 对unst进行resample("W-THU").mean()操作，结果存入down
        down = unst.resample("W-THU").mean(numeric_only=True)
        # 对down进行stack("ID", future_stack=future_stack)操作，结果存入rs
        rs = down.stack("ID", future_stack=future_stack)
        # 对unst的"VAR1"列进行resample("W-THU").mean()和stack("ID", future_stack=future_stack)操作，结果存入xp
        xp = (
            unst.loc[:, ["VAR1"]]
            .resample("W-THU")
            .mean()
            .stack("ID", future_stack=future_stack)
        )
        # 设置xp的列名为Params
        xp.columns.name = "Params"
        # 使用assert_frame_equal断言rs和xp是否相等
        tm.assert_frame_equal(rs, xp)
    # 测试 stack 方法中的 dropna 参数设置，用于处理缺失值
    def test_stack_dropna(self, future_stack):
        # 创建一个包含多列的 DataFrame 对象
        df = DataFrame({"A": ["a1", "a2"], "B": ["b1", "b2"], "C": [1, 1]})
        # 将指定列设为索引，形成层次化索引
        df = df.set_index(["A", "B"])

        # 如果 future_stack 为 False，则设置 dropna 参数为 False；否则设置为 lib.no_default
        dropna = False if not future_stack else lib.no_default
        # 对 DataFrame 进行 unstack 操作，并根据 dropna 参数进行 stack 操作
        stacked = df.unstack().stack(dropna=dropna, future_stack=future_stack)
        # 断言 stack 后的数据行数大于去除缺失值后的数据行数
        assert len(stacked) > len(stacked.dropna())

        # 如果 future_stack 为 True，则预期会引发 ValueError 异常，且异常信息包含 "dropna must be unspecified"
        if future_stack:
            with pytest.raises(ValueError, match="dropna must be unspecified"):
                df.unstack().stack(dropna=True, future_stack=future_stack)
        else:
            # 否则，设置 dropna 参数为 True，并对 DataFrame 进行 stack 操作
            stacked = df.unstack().stack(dropna=True, future_stack=future_stack)
            # 使用 assert_frame_equal 方法断言 stack 操作后的结果与去除缺失值后的结果相等
            tm.assert_frame_equal(stacked, stacked.dropna())

    # 测试对多层次索引进行 unstack 操作
    def test_unstack_multiple_hierarchical(self, future_stack):
        # 创建一个具有多层次索引和多层次列的 DataFrame 对象
        df = DataFrame(
            index=[
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 1, 1, 0, 0, 1, 1],
                [0, 1, 0, 1, 0, 1, 0, 1],
            ],
            columns=[[0, 0, 1, 1], [0, 1, 0, 1]],
        )

        # 设置索引名称
        df.index.names = ["a", "b", "c"]
        # 设置列名称
        df.columns.names = ["d", "e"]

        # 对指定索引进行 unstack 操作
        df.unstack(["b", "c"])

    # 测试对稀疏键空间进行 unstack 操作
    def test_unstack_sparse_keyspace(self):
        # 处理使用 naive 实现可能出现的内存问题 GH#2278
        # 生成大型文件并测试数据透视
        NUM_ROWS = 1000

        # 创建包含随机数据的 DataFrame 对象
        df = DataFrame(
            {
                "A": np.random.default_rng(2).integers(100, size=NUM_ROWS),
                "B": np.random.default_rng(3).integers(300, size=NUM_ROWS),
                "C": np.random.default_rng(4).integers(-7, 7, size=NUM_ROWS),
                "D": np.random.default_rng(5).integers(-19, 19, size=NUM_ROWS),
                "E": np.random.default_rng(6).integers(3000, size=NUM_ROWS),
                "F": np.random.default_rng(7).standard_normal(NUM_ROWS),
            }
        )

        # 根据指定列设置索引
        idf = df.set_index(["A", "B", "C", "D", "E"])

        # 对指定列进行 unstack 操作
        idf.unstack("E")

    # 测试对未观察到的键进行 unstack 操作
    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    def test_unstack_unobserved_keys(self, future_stack):
        # 与 GH#2278 重构相关
        levels = [[0, 1], [0, 1, 2, 3]]
        codes = [[0, 0, 1, 1], [0, 2, 0, 2]]

        # 创建具有 MultiIndex 的 DataFrame 对象
        index = MultiIndex(levels, codes)
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 2)), index=index)

        # 对 DataFrame 进行 unstack 操作
        result = df.unstack()
        # 断言结果 DataFrame 的列数为 4
        assert len(result.columns) == 4

        # 对结果 DataFrame 进行 stack 操作，并使用 assert_frame_equal 方法断言与原始 DataFrame 相等
        recons = result.stack(future_stack=future_stack)
        tm.assert_frame_equal(recons, df)

    # 测试处理级别数量大于 int32 的 unstack 操作
    @pytest.mark.slow
    def test_unstack_number_of_levels_larger_than_int32(
        self, performance_warning, monkeypatch
    ):
        # GH#20601
        # GH 26314: Change ValueError to PerformanceWarning

        # 定义一个MockUnstacker类，继承自reshape_lib._Unstacker，用于模拟unstack操作
        class MockUnstacker(reshape_lib._Unstacker):
            def __init__(self, *args, **kwargs) -> None:
                # 在初始化时会触发警告
                super().__init__(*args, **kwargs)
                # 抛出异常以防止计算最终结果
                raise Exception("Don't compute final result.")

        # 使用monkeypatch上下文，替换reshape_lib._Unstacker为MockUnstacker
        with monkeypatch.context() as m:
            m.setattr(reshape_lib, "_Unstacker", MockUnstacker)
            # 创建一个DataFrame，包含2^16行和2列的零数组
            df = DataFrame(
                np.zeros((2**16, 2)),
                index=[np.arange(2**16), np.arange(2**16)],
            )
            # 提示消息
            msg = "The following operation may generate"
            # 使用assert_produces_warning检查性能警告是否被触发
            with tm.assert_produces_warning(performance_warning, match=msg):
                # 使用pytest.raises检查是否抛出预期的异常信息
                with pytest.raises(Exception, match="Don't compute final result."):
                    # 执行DataFrame的unstack操作
                    df.unstack()

    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    @pytest.mark.parametrize(
        # 参数化测试，生成多个levels组合
        "levels",
        itertools.chain.from_iterable(
            itertools.product(itertools.permutations([0, 1, 2], width), repeat=2)
            for width in [2, 3]
        ),
    )
    @pytest.mark.parametrize("stack_lev", range(2))
    def test_stack_order_with_unsorted_levels(
        self, levels, stack_lev, sort, future_stack
    ):
        # GH#16323
        # 对于1行的情况进行深度检查

        # 创建一个MultiIndex列，使用给定的levels和codes
        columns = MultiIndex(levels=levels, codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
        # 创建一个DataFrame，包含指定列和一行数据
        df = DataFrame(columns=columns, data=[range(4)])
        # 如果future_stack为False，则kwargs为空字典；否则包含sort参数
        kwargs = {} if future_stack else {"sort": sort}
        # 执行DataFrame的stack操作
        df_stacked = df.stack(stack_lev, future_stack=future_stack, **kwargs)
        # 遍历DataFrame的索引和列，检查堆叠后的结果是否与预期一致
        for row in df.index:
            for col in df.columns:
                expected = df.loc[row, col]
                result_row = row, col[stack_lev]
                result_col = col[1 - stack_lev]
                result = df_stacked.loc[result_row, result_col]
                assert result == expected

    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    def test_stack_order_with_unsorted_levels_multi_row(self, future_stack):
        # GH#16323

        # 对于多行的情况进行检查

        # 创建一个MultiIndex列，包含指定的levels和codes
        mi = MultiIndex(
            levels=[["A", "C", "B"], ["B", "A", "C"]],
            codes=[np.repeat(range(3), 3), np.tile(range(3), 3)],
        )
        # 创建一个DataFrame，包含指定列和行，并填充数据
        df = DataFrame(
            columns=mi, index=range(5), data=np.arange(5 * len(mi)).reshape(5, -1)
        )
        # 使用assert检查所有元素是否符合预期
        assert all(
            df.loc[row, col]
            == df.stack(0, future_stack=future_stack).loc[(row, col[0]), col[1]]
            for row in df.index
            for col in df.columns
        )

    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    # 测试函数，验证未排序层级的多行情况下的堆叠顺序
    def test_stack_order_with_unsorted_levels_multi_row_2(self, future_stack):
        # GH#53636
        # 定义多层级元组，表示多级索引的层级结构
        levels = ((0, 1), (1, 0))
        # 指定堆叠的层级
        stack_lev = 1
        # 创建具有多级索引的数据帧，包括列名和数据
        columns = MultiIndex(levels=levels, codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
        df = DataFrame(columns=columns, data=[range(4)], index=[1, 0, 2, 3])
        # 根据 future_stack 参数确定是否排序，生成堆叠后的结果
        kwargs = {} if future_stack else {"sort": True}
        result = df.stack(stack_lev, future_stack=future_stack, **kwargs)
        # 期望的索引结构，包括多层级索引的各层级和编码
        expected_index = MultiIndex(
            levels=[[0, 1, 2, 3], [0, 1]],
            codes=[[1, 1, 0, 0, 2, 2, 3, 3], [1, 0, 1, 0, 1, 0, 1, 0]],
        )
        # 期望的数据帧结果，包括列名和相应的数据
        expected = DataFrame(
            {
                0: [0, 1, 0, 1, 0, 1, 0, 1],
                1: [2, 3, 2, 3, 2, 3, 2, 3],
            },
            index=expected_index,
        )
        # 使用测试工具函数检查结果是否符合预期
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    # 测试函数，验证未排序的多级索引的堆叠和取消堆叠操作
    def test_stack_unstack_unordered_multiindex(self, future_stack):
        # GH# 18265
        # 创建包含 0 到 4 的数组
        values = np.arange(5)
        # 创建一个二维数组，包含两个行，分别表示 'b0' 到 'b4' 和 'a0' 到 'a4'
        data = np.vstack(
            [
                [f"b{x}" for x in values],  # b0, b1, ..
                [f"a{x}" for x in values],  # a0, a1, ..
            ]
        )
        # 创建数据帧，设置列名为 'b' 和 'a'
        df = DataFrame(data.T, columns=["b", "a"])
        df.columns.name = "first"
        # 创建字典，将数据帧作为值，键为 'x'
        second_level_dict = {"x": df}
        # 使用 concat 函数将字典中的数据帧沿列方向连接起来
        multi_level_df = pd.concat(second_level_dict, axis=1)
        # 设置连接后的数据帧的列索引名称为 ['second', 'first']
        multi_level_df.columns.names = ["second", "first"]
        # 根据列名排序多级索引的列，生成新的数据帧
        df = multi_level_df.reindex(sorted(multi_level_df.columns), axis=1)
        # 对堆叠后的数据帧执行取消堆叠操作，生成结果数据帧
        result = df.stack(["first", "second"], future_stack=future_stack).unstack(
            ["first", "second"]
        )
        # 期望的数据帧结果，包括数据和多级索引的层级名称
        expected = DataFrame(
            [["a0", "b0"], ["a1", "b1"], ["a2", "b2"], ["a3", "b3"], ["a4", "b4"]],
            index=[0, 1, 2, 3, 4],
            columns=MultiIndex.from_tuples(
                [("a", "x"), ("b", "x")], names=["first", "second"]
            ),
        )
        # 使用测试工具函数检查结果是否符合预期
        tm.assert_frame_equal(result, expected)

    # 测试函数，验证取消堆叠操作后数据类型的保留情况
    def test_unstack_preserve_types(
        self, multiindex_year_month_day_dataframe_random_data, using_infer_string
    ):
        # GH#403
        # 获取包含随机数据的多级年月日数据帧
        ymd = multiindex_year_month_day_dataframe_random_data
        # 向数据帧添加新列 'E'，其值为 'foo'
        ymd["E"] = "foo"
        # 向数据帧添加新列 'F'，其值为 2
        ymd["F"] = 2

        # 对数据帧执行取消堆叠操作，生成新的数据帧
        unstacked = ymd.unstack("month")
        # 使用断言检查取消堆叠后 'A' 列在索引为 1 的行的数据类型是否为 np.float64
        assert unstacked["A", 1].dtype == np.float64
        # 使用断言检查取消堆叠后 'E' 列在索引为 1 的行的数据类型是否为 np.object_ 或 "string"（取决于 using_infer_string 参数）
        assert (
            unstacked["E", 1].dtype == np.object_
            if not using_infer_string
            else "string"
        )
        # 使用断言检查取消堆叠后 'F' 列在索引为 1 的行的数据类型是否为 np.float64
        assert unstacked["F", 1].dtype == np.float64

    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    # 定义一个测试函数，测试在使用未堆叠（unstacked）的情况下索引溢出
    def test_unstack_group_index_overflow(self, future_stack):
        # 创建一个包含500个元素的数组，并重复两次
        codes = np.tile(np.arange(500), 2)
        # 创建一个包含500个元素的数组作为索引的级别
        level = np.arange(500)

        # 创建一个多级索引对象
        index = MultiIndex(
            # 创建包含8个相同级别的列表，每个列表包含500个元素，再加上一个包含0和1的列表作为最后一个级别
            levels=[level] * 8 + [[0, 1]],
            # 创建包含8个相同编码的列表，每个列表包含500个元素，再加上一个包含重复0和1的数组作为最后一个编码
            codes=[codes] * 8 + [np.arange(2).repeat(500)],
        )

        # 创建一个包含1000个元素的序列，使用上述创建的多级索引
        s = Series(np.arange(1000), index=index)
        # 对序列进行解堆叠操作
        result = s.unstack()
        # 断言解堆叠后的结果形状为(500, 2)
        assert result.shape == (500, 2)

        # 测试解堆叠操作的逆操作（重新堆叠），使用未来堆叠的参数
        stacked = result.stack(future_stack=future_stack)
        # 断言重新堆叠后的序列与原始序列在重新索引后相等
        tm.assert_series_equal(s, stacked.reindex(s.index))

        # 将新创建的多级索引放置在开头
        index = MultiIndex(
            # 创建一个包含一个包含0和1的列表作为第一个级别，再加上包含8个相同级别的列表，每个列表包含500个元素
            levels=[[0, 1]] + [level] * 8,
            # 创建一个包含一个包含重复0和1的数组作为第一个编码，再加上包含8个相同编码的列表，每个列表包含500个元素
            codes=[np.arange(2).repeat(500)] + [codes] * 8,
        )

        # 创建一个包含1000个元素的序列，使用上述创建的多级索引
        s = Series(np.arange(1000), index=index)
        # 对序列进行解堆叠操作，指定解堆叠的级别为0
        result = s.unstack(0)
        # 断言解堆叠后的结果形状为(500, 2)
        assert result.shape == (500, 2)

        # 将新创建的多级索引放置在中间
        index = MultiIndex(
            # 创建包含4个相同级别的列表，每个列表包含500个元素，再加上一个包含0和1的列表作为中间级别，再加上包含4个相同级别的列表，每个列表包含500个元素
            levels=[level] * 4 + [[0, 1]] + [level] * 4,
            # 创建包含4个相同编码的列表，每个列表包含500个元素，再加上一个包含重复0和1的数组作为中间编码，再加上包含4个相同编码的列表，每个列表包含500个元素
            codes=([codes] * 4 + [np.arange(2).repeat(500)] + [codes] * 4),
        )

        # 创建一个包含1000个元素的序列，使用上述创建的多级索引
        s = Series(np.arange(1000), index=index)
        # 对序列进行解堆叠操作，指定解堆叠的级别为4
        result = s.unstack(4)
        # 断言解堆叠后的结果形状为(500, 2)
        assert result.shape == (500, 2)

    # 定义一个测试函数，测试在缺失整数转换为浮点数时的解堆叠操作
    def test_unstack_with_missing_int_cast_to_float(self):
        # 创建一个数据框，包含三列，其中两列的值为字符串"ca"和"cb"，第三列的值为10，将前两列设为索引
        df = DataFrame(
            {
                "a": ["A", "A", "B"],
                "b": ["ca", "cb", "cb"],
                "v": [10] * 3,
            }
        ).set_index(["a", "b"])

        # 添加另一列整数以获取2个数据块
        df["is_"] = 1
        # 断言数据框内部数据块的数量为2
        assert len(df._mgr.blocks) == 2

        # 对数据框进行解堆叠操作，按照"b"列进行解堆叠
        result = df.unstack("b")
        # 将解堆叠后的结果中("is_", "ca")列的缺失值填充为0
        result[("is_", "ca")] = result[("is_", "ca")].fillna(0)

        # 创建一个预期的数据框，包含浮点数值的列表，并设定正确的索引和列
        expected = DataFrame(
            [[10.0, 10.0, 1.0, 1.0], [np.nan, 10.0, 0.0, 1.0]],
            index=Index(["A", "B"], name="a"),
            columns=MultiIndex.from_tuples(
                [("v", "ca"), ("v", "cb"), ("is_", "ca"), ("is_", "cb")],
                names=[None, "b"],
            ),
        )
        # 断言解堆叠后的结果与预期的结果数据框相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试函数，测试在级别中存在NaN值时的解堆叠操作
    def test_unstack_with_level_has_nan(self):
        # 创建一个数据框，包含四列，分别是"L1", "L2", "L3"和"x"，并设定"L1", "L2", "L3"三列为索引
        df1 = DataFrame(
            {
                "L1": [1, 2, 3, 4],
                "L2": [3, 4, 1, 2],
                "L3": [1, 1, 1, 1],
                "x": [1, 2, 3, 4],
            }
        )
        df1 = df1.set_index(["L1", "L2", "L3"])

        # 创建新的级别名称列表，包含四个元素，其中一个为None
        new_levels = ["n1", "n2", "n3", None]
        # 设置"L1"和"L2"列的级别名称为新创建的级别名称列表
        df1.index = df1.index.set_levels(levels=new_levels, level="L1")
        df1.index = df1.index.set_levels(levels=new_levels, level="L2")

        # 对数据框进行解堆叠操作，按照"L3"列进行解堆叠，并选择结果中("x", 1)列并排序索引
        result = df1.unstack("L3")[("x", 1)].sort_index().index
        # 创建一个预期的多级索引对象，包含两个级别和对应的编码
        expected = MultiIndex(
            levels=[["n1", "n2", "n3", None], ["n1", "n2", "n3", None]],
            codes=[[0, 1, 2, 3], [2, 3, 0, 1]],
            names=["L1", "L2"],
        )
        # 断言解堆叠后的索引与预期的索引对象相等
        tm.assert_index_equal(result, expected)

    # 使用pytest的标记语法，忽略警告："The previous implementation of stack is deprecated"
    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    # 定义一个测试方法，用于测试在多级索引列中堆叠操作中的 NaN 处理
    def test_stack_nan_in_multiindex_columns(self, future_stack):
        # GH#39481
        # 创建一个包含一行五列的 DataFrame，初始值全为零
        df = DataFrame(
            np.zeros([1, 5]),
            columns=MultiIndex.from_tuples(
                [
                    (0, None, None),
                    (0, 2, 0),
                    (0, 2, 1),
                    (0, 3, 0),
                    (0, 3, 1),
                ],
            ),
        )
        # 调用 stack 方法进行堆叠操作，第二层索引标签从 future_stack 参数中取值
        result = df.stack(2, future_stack=future_stack)
        # 根据 future_stack 参数的值选择预期的索引和列
        if future_stack:
            # 如果 future_stack 为 True，则期望的索引
            index = MultiIndex(levels=[[0], [0.0, 1.0]], codes=[[0, 0, 0], [-1, 0, 1]])
            # 如果 future_stack 为 True，则期望的列
            columns = MultiIndex(levels=[[0], [2, 3]], codes=[[0, 0, 0], [-1, 0, 1]])
        else:
            # 如果 future_stack 为 False，则期望的索引
            index = Index([(0, None), (0, 0), (0, 1)])
            # 如果 future_stack 为 False，则期望的列
            columns = Index([(0, None), (0, 2), (0, 3)])
        # 创建预期的 DataFrame 对象，用于和实际结果比较
        expected = DataFrame(
            [[0.0, np.nan, np.nan], [np.nan, 0.0, 0.0], [np.nan, 0.0, 0.0]],
            index=index,
            columns=columns,
        )
        # 使用 assert_frame_equal 检查实际结果和预期结果是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    # 定义一个测试方法，用于测试多级别堆叠中的分类数据处理
    def test_multi_level_stack_categorical(self, future_stack):
        # GH 15239
        # 创建一个多级索引对象
        midx = MultiIndex.from_arrays(
            [
                ["A"] * 2 + ["B"] * 2,
                pd.Categorical(list("abab")),
                pd.Categorical(list("ccdd")),
            ]
        )
        # 创建一个二行四列的 DataFrame，初始值从 0 到 7
        df = DataFrame(np.arange(8).reshape(2, 4), columns=midx)
        # 调用 stack 方法进行堆叠操作，第一层和第二层索引标签从 future_stack 参数中取值
        result = df.stack([1, 2], future_stack=future_stack)
        # 根据 future_stack 参数的值选择预期的 DataFrame 结构
        if future_stack:
            # 如果 future_stack 为 True，则期望的 DataFrame 结构
            expected = DataFrame(
                [
                    [0, np.nan],
                    [1, np.nan],
                    [np.nan, 2],
                    [np.nan, 3],
                    [4, np.nan],
                    [5, np.nan],
                    [np.nan, 6],
                    [np.nan, 7],
                ],
                columns=["A", "B"],
                index=MultiIndex.from_arrays(
                    [
                        [0] * 4 + [1] * 4,
                        pd.Categorical(list("abababab")),
                        pd.Categorical(list("ccddccdd")),
                    ]
                ),
            )
        else:
            # 如果 future_stack 为 False，则期望的 DataFrame 结构
            expected = DataFrame(
                [
                    [0, np.nan],
                    [np.nan, 2],
                    [1, np.nan],
                    [np.nan, 3],
                    [4, np.nan],
                    [np.nan, 6],
                    [5, np.nan],
                    [np.nan, 7],
                ],
                columns=["A", "B"],
                index=MultiIndex.from_arrays(
                    [
                        [0] * 4 + [1] * 4,
                        pd.Categorical(list("aabbaabb")),
                        pd.Categorical(list("cdcdcdcd")),
                    ]
                ),
            )
        # 使用 assert_frame_equal 检查实际结果和预期结果是否相等
        tm.assert_frame_equal(result, expected)
    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    # 标记此测试函数，忽略关于堆叠操作旧实现已弃用的警告信息
    def test_stack_nan_level(self, future_stack):
        # GH 9406
        # 创建一个包含 NaN 值的 DataFrame
        df_nan = DataFrame(
            np.arange(4).reshape(2, 2),
            columns=MultiIndex.from_tuples(
                [("A", np.nan), ("B", "b")], names=["Upper", "Lower"]
            ),
            index=Index([0, 1], name="Num"),
            dtype=np.float64,
        )
        # 进行堆叠操作，并传递 future_stack 参数
        result = df_nan.stack(future_stack=future_stack)
        # 根据 future_stack 的值选择不同的 MultiIndex
        if future_stack:
            index = MultiIndex(
                levels=[[0, 1], [np.nan, "b"]],
                codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
                names=["Num", "Lower"],
            )
        else:
            index = MultiIndex.from_tuples(
                [(0, np.nan), (0, "b"), (1, np.nan), (1, "b")], names=["Num", "Lower"]
            )
        # 创建预期的 DataFrame
        expected = DataFrame(
            [[0.0, np.nan], [np.nan, 1], [2.0, np.nan], [np.nan, 3.0]],
            columns=Index(["A", "B"], name="Upper"),
            index=index,
        )
        # 使用 pytest 的断言方法检查结果是否符合预期
        tm.assert_frame_equal(result, expected)

    def test_unstack_categorical_columns(self):
        # GH 14018
        # 创建一个包含分类列的 DataFrame，并进行反堆叠操作
        idx = MultiIndex.from_product([["A"], [0, 1]])
        df = DataFrame({"cat": pd.Categorical(["a", "b"])}, index=idx)
        result = df.unstack()
        # 创建预期的 DataFrame，包含分类数据的反堆叠结果
        expected = DataFrame(
            {
                0: pd.Categorical(["a"], categories=["a", "b"]),
                1: pd.Categorical(["b"], categories=["a", "b"]),
            },
            index=["A"],
        )
        expected.columns = MultiIndex.from_tuples([("cat", 0), ("cat", 1)])
        # 使用 pytest 的断言方法检查结果是否符合预期
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    # 标记此测试函数，忽略关于堆叠操作旧实现已弃用的警告信息
    def test_stack_unsorted(self, future_stack):
        # GH 16925
        # 创建一个 MultiIndex
        PAE = ["ITA", "FRA"]
        VAR = ["A1", "A2"]
        TYP = ["CRT", "DBT", "NET"]
        MI = MultiIndex.from_product([PAE, VAR, TYP], names=["PAE", "VAR", "TYP"])

        V = list(range(len(MI)))
        DF = DataFrame(data=V, index=MI, columns=["VALUE"])

        # 对 DataFrame 进行反堆叠操作，并去除第一层级的列标签
        DF = DF.unstack(["VAR", "TYP"])
        DF.columns = DF.columns.droplevel(0)
        # 设置特定位置的值为 9999
        DF.loc[:, ("A0", "NET")] = 9999

        # 进行堆叠操作，并传递 future_stack 参数，然后排序结果
        result = DF.stack(["VAR", "TYP"], future_stack=future_stack).sort_index()
        # 创建预期的堆叠结果，并排序结果
        expected = (
            DF.sort_index(axis=1)
            .stack(["VAR", "TYP"], future_stack=future_stack)
            .sort_index()
        )
        # 使用 pytest 的断言方法检查结果是否符合预期
        tm.assert_series_equal(result, expected)
    # 测试函数，用于测试带有可空数据类型的堆栈操作
    def test_stack_nullable_dtype(self, future_stack):
        # 创建一个多级索引，包含两个级别的元组列表，每个级别都有两个值
        columns = MultiIndex.from_product(
            [["54511", "54515"], ["r", "t_mean"]],
            names=["station", "element"]
        )
        # 创建一个索引对象，包含整数 1、2、3，命名为"time"
        index = Index([1, 2, 3], name="time")

        # 创建一个二维数组 arr，将其转换为 DataFrame 对象 df
        arr = np.array([[50, 226, 10, 215], [10, 215, 9, 220], [305, 232, 111, 220]])
        df = DataFrame(arr, columns=columns, index=index, dtype=pd.Int64Dtype())

        # 对 DataFrame df 执行堆栈操作，指定堆栈的级别为"station"，并传递 future_stack 参数
        result = df.stack("station", future_stack=future_stack)

        # 期望的 DataFrame 对象，先将 df 转换为 int64 类型，然后执行堆栈操作，再转换为 pd.Int64Dtype 类型
        expected = (
            df.astype(np.int64)
            .stack("station", future_stack=future_stack)
            .astype(pd.Int64Dtype())
        )
        # 使用测试框架进行结果比较
        tm.assert_frame_equal(result, expected)

        # 处理非同质情况：将 df 的第一列转换为 pd.Float64Dtype 类型，然后执行堆栈操作
        df[df.columns[0]] = df[df.columns[0]].astype(pd.Float64Dtype())
        result = df.stack("station", future_stack=future_stack)

        # 期望的 DataFrame 对象，包含两列："r" 和 "t_mean"，数据类型分别为 pd.Float64Dtype 和 pd.Int64Dtype
        expected = DataFrame(
            {
                "r": pd.array(
                    [50.0, 10.0, 10.0, 9.0, 305.0, 111.0], dtype=pd.Float64Dtype()
                ),
                "t_mean": pd.array(
                    [226, 215, 215, 220, 232, 220], dtype=pd.Int64Dtype()
                ),
            },
            # 创建一个多级索引，包含原始索引 index 和 columns 的第一级值
            index=MultiIndex.from_product([index, columns.levels[0]]),
        )
        # 设置期望 DataFrame 对象的列名
        expected.columns.name = "element"
        # 使用测试框架进行结果比较
        tm.assert_frame_equal(result, expected)

    # 测试函数，用于测试混合级别名称的取消堆栈操作
    def test_unstack_mixed_level_names(self):
        # 创建一个多级索引对象 idx，包含三个级别的数组列表，命名为("x", 0, "y")
        arrays = [["a", "a"], [1, 2], ["red", "blue"]]
        idx = MultiIndex.from_arrays(arrays, names=("x", 0, "y"))
        # 创建一个 DataFrame 对象 df，包含一个列 "m"，使用 idx 作为索引
        df = DataFrame({"m": [1, 2]}, index=idx)
        # 执行取消堆栈操作，指定取消堆栈的级别为"x"
        result = df.unstack("x")
        # 期望的 DataFrame 对象，包含一列，使用 MultiIndex 指定的列名和索引名
        expected = DataFrame(
            [[1], [2]],
            columns=MultiIndex.from_tuples([("m", "a")], names=[None, "x"]),
            index=MultiIndex.from_tuples([(1, "red"), (2, "blue")], names=[0, "y"]),
        )
        # 使用测试框架进行结果比较
        tm.assert_frame_equal(result, expected)
# 使用 pytest 的标记来忽略特定的警告信息
@pytest.mark.filterwarnings("ignore:The previous implementation of stack is deprecated")
def test_stack_tuple_columns(future_stack):
    # GH#54948 - 测试在输入具有非 MultiIndex 且包含元组的情况下的堆叠操作
    # 创建一个 DataFrame，包含三行三列的数据，列使用元组作为标签
    df = DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=[("a", 1), ("a", 2), ("b", 1)]
    )
    # 对 DataFrame 进行堆叠操作，传入 future_stack 参数
    result = df.stack(future_stack=future_stack)
    # 期望的结果是一个 Series，索引是 MultiIndex
    expected = Series(
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        index=MultiIndex(
            levels=[[0, 1, 2], [("a", 1), ("a", 2), ("b", 1)]],
            codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]],
        ),
    )
    # 断言堆叠后的结果与期望的结果相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "dtype, na_value",
    [
        ("float64", np.nan),
        ("Float64", np.nan),
        ("Float64", pd.NA),
        ("Int64", pd.NA),
    ],
)
@pytest.mark.parametrize("test_multiindex", [True, False])
def test_stack_preserves_na(dtype, na_value, test_multiindex):
    # GH#56573
    # 根据 test_multiindex 参数决定创建 MultiIndex 或者单个 Index
    if test_multiindex:
        index = MultiIndex.from_arrays(2 * [Index([na_value], dtype=dtype)])
    else:
        index = Index([na_value], dtype=dtype)
    # 创建一个 DataFrame，其中包含一个列 "a"，并使用上面创建的索引
    df = DataFrame({"a": [1]}, index=index)
    # 对 DataFrame 进行堆叠操作
    result = df.stack()

    # 根据 test_multiindex 参数决定期望的索引类型
    if test_multiindex:
        expected_index = MultiIndex.from_arrays(
            [
                Index([na_value], dtype=dtype),
                Index([na_value], dtype=dtype),
                Index(["a"]),
            ]
        )
    else:
        expected_index = MultiIndex.from_arrays(
            [
                Index([na_value], dtype=dtype),
                Index(["a"]),
            ]
        )
    # 创建期望的 Series 对象，其中数据部分均为 1，索引为上面计算得到的 expected_index
    expected = Series(1, index=expected_index)
    # 断言堆叠后的结果与期望的结果相等
    tm.assert_series_equal(result, expected)
```