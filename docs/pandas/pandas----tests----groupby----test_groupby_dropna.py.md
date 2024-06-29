# `D:\src\scipysrc\pandas\pandas\tests\groupby\test_groupby_dropna.py`

```
@pytest.mark.parametrize(
    "dropna, tuples, outputs",
    [  # 参数化测试用例，测试不同的输入组合
        (
            True,  # 是否丢弃 NA 值
            [["A", "B"], ["B", "A"]],  # 多级索引的元组列表
            {"c": [12.0, 123.23], "d": [12.0, 123.0], "e": [12.0, 1.0]},  # 预期的输出字典
        ),
        (
            False,  # 是否丢弃 NA 值
            [["A", "B"], ["A", np.nan], ["B", "A"], [np.nan, "B"]],  # 多级索引的元组列表
            {
                "c": [12.0, 13.3, 123.23, 1.0],
                "d": [12.0, 234.0, 123.0, 1.0],
                "e": [12.0, 13.0, 1.0, 1.0],
            },  # 预期的输出字典
        ),
    ],
)
def test_groupby_dropna_multi_index_dataframe_nan_in_two_groups(
    dropna, tuples, outputs, nulls_fixture, nulls_fixture2
):
    # GH 3729 this is to test that NA in different groups with different representations
    # 函数用来测试多级索引下不同组中的 NA 值及其不同表示方式
    df_list = [
        ["A", "B", 12, 12, 12],
        ["A", nulls_fixture, 12.3, 233.0, 12],
        ["B", "A", 123.23, 123, 1],
        [nulls_fixture2, "B", 1, 1, 1.0],
        ["A", nulls_fixture2, 1, 1, 1.0],
    ]
    df = pd.DataFrame(df_list, columns=["a", "b", "c", "d", "e"])
    grouped = df.groupby(["a", "b"], dropna=dropna).sum()

    mi = pd.MultiIndex.from_tuples(tuples, names=list("ab"))

    # Since right now, by default MI will drop NA from levels when we create MI
    # via `from_*`, so we need to add NA for level manually afterwards.
    # 由于默认情况下，通过 `from_*` 创建 MI 时，MI 会从层级中丢弃 NA 值，因此我们需要手动添加 NA 值到层级中
    if not dropna:
        mi = mi.set_levels([["A", "B", np.nan], ["A", "B", np.nan]])  # 手动添加 NA 值到层级中
    expected = pd.DataFrame(outputs, index=mi)  # 创建预期的输出 DataFrame

    tm.assert_frame_equal(grouped, expected)  # 使用测试框架中的断言函数比较 grouped 和 expected 的内容
    # 使用测试框架中的 assert_frame_equal 函数比较两个数据框架对象 grouped 和 expected 是否相等
    tm.assert_frame_equal(grouped, expected)
@pytest.mark.parametrize(
    "dropna, idx, outputs",
    [  # 参数化测试的参数设置
        (True, ["A", "B"], {"b": [123.23, 13.0], "c": [123.0, 13.0], "d": [1.0, 13.0]}),  # 第一组参数化测试数据，期望输出结果
        (
            False,
            ["A", "B", np.nan],
            {
                "b": [123.23, 13.0, 12.3],
                "c": [123.0, 13.0, 233.0],
                "d": [1.0, 13.0, 12.0],
            },  # 第二组参数化测试数据，期望输出结果
        ),
    ],
)
def test_groupby_dropna_normal_index_dataframe(dropna, idx, outputs):
    # GH 3729
    # 创建包含数据的列表
    df_list = [
        ["B", 12, 12, 12],
        [None, 12.3, 233.0, 12],
        ["A", 123.23, 123, 1],
        ["B", 1, 1, 1.0],
    ]
    # 创建数据框，指定列名
    df = pd.DataFrame(df_list, columns=["a", "b", "c", "d"])
    # 按照列 'a' 进行分组求和
    grouped = df.groupby("a", dropna=dropna).sum()

    # 创建期望的数据框，指定索引和输出结果
    expected = pd.DataFrame(outputs, index=pd.Index(idx, dtype="object", name="a"))

    # 断言两个数据框是否相等
    tm.assert_frame_equal(grouped, expected)


@pytest.mark.parametrize(
    "dropna, idx, expected",
    [  # 参数化测试的参数设置
        (True, ["a", "a", "b", np.nan], pd.Series([3, 3], index=["a", "b"])),  # 第一组参数化测试数据，期望输出结果
        (
            False,
            ["a", "a", "b", np.nan],
            pd.Series([3, 3, 3], index=["a", "b", np.nan]),
        ),  # 第二组参数化测试数据，期望输出结果
    ],
)
def test_groupby_dropna_series_level(dropna, idx, expected):
    # 创建序列对象，指定索引和数据
    ser = pd.Series([1, 2, 3, 3], index=idx)

    # 按照索引级别进行分组求和
    result = ser.groupby(level=0, dropna=dropna).sum()

    # 断言两个序列对象是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "dropna, expected",
    [  # 参数化测试的参数设置
        (True, pd.Series([210.0, 350.0], index=["a", "b"], name="Max Speed")),  # 第一组参数化测试数据，期望输出结果
        (
            False,
            pd.Series([210.0, 350.0, 20.0], index=["a", "b", np.nan], name="Max Speed"),
        ),  # 第二组参数化测试数据，期望输出结果
    ],
)
def test_groupby_dropna_series_by(dropna, expected):
    # 创建序列对象，指定索引和数据
    ser = pd.Series(
        [390.0, 350.0, 30.0, 20.0],
        index=["Falcon", "Falcon", "Parrot", "Parrot"],
        name="Max Speed",
    )

    # 按照给定的列表进行分组并求均值
    result = ser.groupby(["a", "b", "a", np.nan], dropna=dropna).mean()

    # 断言两个序列对象是否相等
    tm.assert_series_equal(result, expected)


def test_grouper_dropna_propagation(dropna):
    # GH 36604
    # 创建数据框，包含两列，其中一列存在缺失值
    df = pd.DataFrame({"A": [0, 0, 1, None], "B": [1, 2, 3, None]})
    # 按照列 'A' 进行分组
    gb = df.groupby("A", dropna=dropna)
    # 断言分组对象的 dropna 属性与输入参数 dropna 是否一致
    assert gb._grouper.dropna == dropna


@pytest.mark.parametrize(
    "index",
    [  # 参数化测试的参数设置
        pd.RangeIndex(0, 4),
        list("abcd"),
        pd.MultiIndex.from_product([(1, 2), ("R", "B")], names=["num", "col"]),
    ],
)
def test_groupby_dataframe_slice_then_transform(dropna, index):
    # GH35014 & GH35612
    # 预期的数据字典，指定列 'B' 的预期结果
    expected_data = {"B": [2, 2, 1, np.nan if dropna else 1]}

    # 创建数据框，指定数据和索引
    df = pd.DataFrame({"A": [0, 0, 1, None], "B": [1, 2, 3, None]}, index=index)
    # 按照列 'A' 进行分组
    gb = df.groupby("A", dropna=dropna)

    # 对数据框进行转换操作，计算长度
    result = gb.transform(len)
    # 创建预期的数据框，指定数据和索引
    expected = pd.DataFrame(expected_data, index=index)
    # 断言两个数据框是否相等
    tm.assert_frame_equal(result, expected)

    # 对数据框的列 'B' 进行长度计算转换
    result = gb[["B"]].transform(len)
    # 创建预期的数据框，指定数据和索引
    expected = pd.DataFrame(expected_data, index=index)
    # 断言两个数据框是否相等
    tm.assert_frame_equal(result, expected)

    # 对数据框的列 'B' 进行长度计算转换，返回序列对象
    result = gb["B"].transform(len)
    # 创建预期的序列对象，指定数据和索引
    expected = pd.Series(expected_data["B"], index=index, name="B")
    # 断言两个序列对象是否相等
    tm.assert_series_equal(result, expected)
    # 使用测试框架中的方法比较结果序列和期望序列是否相等
    tm.assert_series_equal(result, expected)
@pytest.mark.parametrize(
    "dropna, tuples, outputs",
    [  # 参数化测试，定义了三个参数：dropna，tuples，outputs
        (
            True,  # 第一个测试用例的dropna值为True
            [["A", "B"], ["B", "A"]],  # 第一个测试用例的tuples值为[["A", "B"], ["B", "A"]]
            {"c": [13.0, 123.23], "d": [12.0, 123.0], "e": [1.0, 1.0]},  # 第一个测试用例的outputs值为指定的字典
        ),
        (
            False,  # 第二个测试用例的dropna值为False
            [["A", "B"], ["A", np.nan], ["B", "A"]],  # 第二个测试用例的tuples值包含NaN
            {  # 第二个测试用例的outputs值为指定的字典
                "c": [13.0, 12.3, 123.23],
                "d": [12.0, 233.0, 123.0],
                "e": [1.0, 12.0, 1.0],
            },
        ),
    ],
)
def test_groupby_dropna_multi_index_dataframe_agg(dropna, tuples, outputs):
    # GH 3729
    # 创建一个DataFrame对象，用于测试分组和聚合操作
    df_list = [
        ["A", "B", 12, 12, 12],
        ["A", None, 12.3, 233.0, 12],
        ["B", "A", 123.23, 123, 1],
        ["A", "B", 1, 1, 1.0],
    ]
    df = pd.DataFrame(df_list, columns=["a", "b", "c", "d", "e"])
    agg_dict = {"c": "sum", "d": "max", "e": "min"}
    # 根据指定的列进行分组，应用聚合函数，并得到分组后的结果
    grouped = df.groupby(["a", "b"], dropna=dropna).agg(agg_dict)

    mi = pd.MultiIndex.from_tuples(tuples, names=list("ab"))

    # 由于默认情况下，通过from_*方法创建MultiIndex时，会删除级别中的NA值，因此需要手动添加NA值
    if not dropna:
        mi = mi.set_levels(["A", "B", np.nan], level="b")
    expected = pd.DataFrame(outputs, index=mi)

    # 使用测试框架的方法比较实际结果和期望结果是否一致
    tm.assert_frame_equal(grouped, expected)


@pytest.mark.arm_slow
@pytest.mark.parametrize(
    "datetime1, datetime2",
    [
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")),
        (pd.Timedelta("-2 days"), pd.Timedelta("-1 days")),
        (pd.Period("2020-01-01"), pd.Period("2020-02-01")),
    ],
)
@pytest.mark.parametrize("dropna, values", [(True, [12, 3]), (False, [12, 3, 6])])
def test_groupby_dropna_datetime_like_data(
    dropna, values, datetime1, datetime2, unique_nulls_fixture, unique_nulls_fixture2
):
    # 3729
    # 创建包含日期时间数据的DataFrame对象，用于测试根据日期时间进行分组和聚合操作
    df = pd.DataFrame(
        {
            "values": [1, 2, 3, 4, 5, 6],
            "dt": [
                datetime1,
                unique_nulls_fixture,
                datetime2,
                unique_nulls_fixture2,
                datetime1,
                datetime1,
            ],
        }
    )

    if dropna:
        indexes = [datetime1, datetime2]
    else:
        indexes = [datetime1, datetime2, np.nan]

    # 根据日期时间列进行分组，应用聚合函数，并得到分组后的结果
    grouped = df.groupby("dt", dropna=dropna).agg({"values": "sum"})
    expected = pd.DataFrame({"values": values}, index=pd.Index(indexes, name="dt"))

    # 使用测试框架的方法比较实际结果和期望结果是否一致
    tm.assert_frame_equal(grouped, expected)


@pytest.mark.parametrize(
    "dropna, data, selected_data, levels",
    [
        # 第一个测试参数组，测试 dropna=False 的情况，包含 NaN 值
        pytest.param(
            False,
            {"groups": ["a", "a", "b", np.nan], "values": [10, 10, 20, 30]},
            {"values": [0, 1, 0, 0]},
            ["a", "b", np.nan],
            id="dropna_false_has_nan",
        ),
        # 第二个测试参数组，测试 dropna=True 的情况，包含 NaN 值
        pytest.param(
            True,
            {"groups": ["a", "a", "b", np.nan], "values": [10, 10, 20, 30]},
            {"values": [0, 1, 0]},
            None,
            id="dropna_true_has_nan",
        ),
        # 第三个测试参数组，测试 dropna=False 的情况，不包含 NaN 值
        pytest.param(
            # no nan in "groups"; dropna=True|False should be same.
            False,
            {"groups": ["a", "a", "b", "c"], "values": [10, 10, 20, 30]},
            {"values": [0, 1, 0, 0]},
            None,
            id="dropna_false_no_nan",
        ),
        # 第四个测试参数组，测试 dropna=True 的情况，不包含 NaN 值
        pytest.param(
            # no nan in "groups"; dropna=True|False should be same.
            True,
            {"groups": ["a", "a", "b", "c"], "values": [10, 10, 20, 30]},
            {"values": [0, 1, 0, 0]},
            None,
            id="dropna_true_no_nan",
        ),
    ],
# 测试函数：对多级索引输入使用 groupby，并且在应用时处理 NA 值
def test_groupby_apply_with_dropna_for_multi_index(dropna, data, selected_data, levels):
    # GH 35889

    # 创建 DataFrame 对象
    df = pd.DataFrame(data)
    # 根据 "groups" 列分组，并根据 dropna 参数决定是否丢弃 NA 值
    gb = df.groupby("groups", dropna=dropna)
    # 准备警告信息，用于断言检查
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 断言在使用 apply 方法时会产生 DeprecationWarning 警告
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对每个分组应用 lambda 函数，创建包含 "values" 列的 DataFrame
        result = gb.apply(lambda grp: pd.DataFrame({"values": range(len(grp))}))

    # 创建 MultiIndex 元组，由 data["groups"] 和 selected_data["values"] 组成
    mi_tuples = tuple(zip(data["groups"], selected_data["values"]))
    # 创建 MultiIndex 对象，并指定其名称为 "groups"，默认情况下会丢弃 NA 值
    mi = pd.MultiIndex.from_tuples(mi_tuples, names=["groups", None])
    # 如果 dropna 为 False 并且 levels 存在，则手动为 "groups" 级别添加 NA 值
    if not dropna and levels:
        mi = mi.set_levels(levels, level="groups")

    # 根据预期结果创建 DataFrame，使用选定的数据和创建的 MultiIndex 作为索引
    expected = pd.DataFrame(selected_data, index=mi)
    # 断言 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数：对多级索引输入使用 groupby，并且在丢弃 NA 值时进行测试
@pytest.mark.parametrize("input_index", [None, ["a"], ["a", "b"]])
@pytest.mark.parametrize("keys", [["a"], ["a", "b"]])
@pytest.mark.parametrize("series", [True, False])
def test_groupby_dropna_with_multiindex_input(input_index, keys, series):
    # GH#46783
    # 创建包含 NaN 值的 DataFrame 对象
    obj = pd.DataFrame(
        {
            "a": [1, np.nan],
            "b": [1, 1],
            "c": [2, 3],
        }
    )

    # 根据 keys 列表创建预期的 DataFrame，将 obj 设置为以 keys 列为索引
    expected = obj.set_index(keys)
    # 如果 series 为 True，则只保留 "c" 列作为 Series
    if series:
        expected = expected["c"]
    # 如果 input_index 为 ["a", "b"] 并且 keys 为 ["a"]，则只保留 "c" 列
    elif input_index == ["a", "b"] and keys == ["a"]:
        expected = expected[["c"]]

    # 如果 input_index 不为 None，则将 obj 设置为以 input_index 为索引
    if input_index is not None:
        obj = obj.set_index(input_index)
    # 根据 keys 列进行分组，并根据 dropna 参数决定是否丢弃 NA 值
    gb = obj.groupby(keys, dropna=False)
    # 如果 series 为 True，则将结果限定为 "c" 列的 Series
    if series:
        gb = gb["c"]
    # 对分组后的结果求和
    result = gb.sum()

    # 断言 result 和 expected 是否相等
    tm.assert_equal(result, expected)


# 测试函数：测试在分组中包含 NaN 值的情况
def test_groupby_nan_included():
    # GH 35646
    # 创建包含 NaN 值的 DataFrame 对象
    data = {"group": ["g1", np.nan, "g1", "g2", np.nan], "B": [0, 1, 2, 3, 4]}
    df = pd.DataFrame(data)
    # 根据 "group" 列进行分组，且设置 dropna 参数为 False
    grouped = df.groupby("group", dropna=False)
    # 获取分组后的索引
    result = grouped.indices
    # 设置期望的结果，包含不同分组及其对应的索引数组
    dtype = np.intp
    expected = {
        "g1": np.array([0, 2], dtype=dtype),
        "g2": np.array([3], dtype=dtype),
        np.nan: np.array([1, 4], dtype=dtype),
    }
    # 逐一断言 result 的值和 expected 的值是否相等
    for result_values, expected_values in zip(result.values(), expected.values()):
        tm.assert_numpy_array_equal(result_values, expected_values)
    # 断言 result 的第三个键是否为 NaN
    assert np.isnan(list(result.keys())[2])
    # 断言 result 的前两个键是否分别为 "g1" 和 "g2"
    assert list(result.keys())[0:2] == ["g1", "g2"]


# 测试函数：测试在多级索引中处理 NaN 值的情况
def test_groupby_drop_nan_with_multi_index():
    # GH 39895
    # 创建包含 NaN 值的 DataFrame 对象
    df = pd.DataFrame([[np.nan, 0, 1]], columns=["a", "b", "c"])
    # 将 "a" 和 "b" 列设置为多级索引
    df = df.set_index(["a", "b"])
    # 根据 "a" 和 "b" 列进行分组，且设置 dropna 参数为 False，然后取第一个值
    result = df.groupby(["a", "b"], dropna=False).first()
    # 设置预期的结果为原始 DataFrame
    expected = df
    # 断言 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 参数化测试：sequence_index 枚举由 x、y、z 组成的长度为 4 的所有字符串
@pytest.mark.parametrize("sequence_index", range(3**4))
# 参数化测试：dtype 可以是 None
@pytest.mark.parametrize(
    "dtype",
    # 一个包含多种数据类型和参数的列表
    [
        # 第一个元素为 None 类型
        None,
        # 8 位无符号整数类型
        "UInt8",
        # 8 位有符号整数类型
        "Int8",
        # 16 位无符号整数类型
        "UInt16",
        # 16 位有符号整数类型
        "Int16",
        # 32 位无符号整数类型
        "UInt32",
        # 32 位有符号整数类型
        "Int32",
        # 64 位无符号整数类型
        "UInt64",
        # 64 位有符号整数类型
        "Int64",
        # 单精度浮点数类型
        "Float32",
        # 双精度浮点数类型
        "Float64",
        # 类别（分类）类型
        "category",
        # 字符串类型
        "string",
        # 参数化的字符串类型，用于 pyarrow，带有相关标记
        pytest.param(
            "string[pyarrow]",
            # 根据条件跳过 pytest 标记，如果 pa_version_under10p1 为真，则跳过
            marks=pytest.mark.skipif(
                pa_version_under10p1, reason="pyarrow is not installed"
            ),
        ),
        # 带有纳秒精度的日期时间类型
        "datetime64[ns]",
        # 带有天为周期的时间间隔类型
        "period[d]",
        # 稀疏浮点数类型
        "Sparse[float]",
    ]
@pytest.mark.parametrize("test_series", [True, False])
# 使用pytest的参数化装饰器，定义一个测试函数，参数为test_series，取值为True和False
def test_no_sort_keep_na(sequence_index, dtype, test_series, as_index):
    # 测试函数，用于验证不排序并保留NA值的情况

    # GH#46584, GH#48794
    # 关联GitHub的issue编号，用于跟踪相关问题

    # Convert sequence_index into a string sequence, e.g. 5 becomes "xxyz"
    # This sequence is used for the grouper.
    # 将sequence_index转换为字符串序列，例如，5转换为"xxyz"
    # 此序列用于分组器中

    sequence = "".join(
        [{0: "x", 1: "y", 2: "z"}[sequence_index // (3**k) % 3] for k in range(4)]
    )
    # 使用数学计算将sequence_index转换为字符串序列，存储在sequence中

    # Unique values to use for grouper, depends on dtype
    # 根据dtype确定用于分组的唯一值

    if dtype in ("string", "string[pyarrow]"):
        uniques = {"x": "x", "y": "y", "z": pd.NA}
    elif dtype in ("datetime64[ns]", "period[d]"):
        uniques = {"x": "2016-01-01", "y": "2017-01-01", "z": pd.NA}
    else:
        uniques = {"x": 1, "y": 2, "z": np.nan}
    # 根据dtype选择不同的唯一值字典，用于分组键

    df = pd.DataFrame(
        {
            "key": pd.Series([uniques[label] for label in sequence], dtype=dtype),
            "a": [0, 1, 2, 3],
        }
    )
    # 创建包含"key"和"a"列的DataFrame，"key"列使用sequence中的标签对应的唯一值

    gb = df.groupby("key", dropna=False, sort=False, as_index=as_index, observed=False)
    # 对DataFrame进行按"key"列分组，保留NA值，不排序，根据as_index参数设定是否作为索引，不考虑观察到的值

    if test_series:
        gb = gb["a"]
    # 如果test_series为True，则返回分组后的"a"列Series

    result = gb.sum()
    # 计算分组后的求和结果

    # Manually compute the groupby sum, use the labels "x", "y", and "z" to avoid
    # issues with hashing np.nan
    # 手动计算分组求和，使用标签"x"、"y"和"z"避免np.nan哈希问题

    summed = {}
    for idx, label in enumerate(sequence):
        summed[label] = summed.get(label, 0) + idx
    # 手动计算分组求和，将结果存储在summed字典中

    if dtype == "category":
        index = pd.CategoricalIndex(
            [uniques[e] for e in summed],
            df["key"].cat.categories,
            name="key",
        )
    elif isinstance(dtype, str) and dtype.startswith("Sparse"):
        index = pd.Index(
            pd.array([uniques[label] for label in summed], dtype=dtype), name="key"
        )
    else:
        index = pd.Index([uniques[label] for label in summed], dtype=dtype, name="key")
    # 根据dtype类型创建索引对象index，以存储预期的求和结果

    expected = pd.Series(summed.values(), index=index, name="a", dtype=None)
    # 创建预期的Series对象，包含手动计算得到的求和结果

    if not test_series:
        expected = expected.to_frame()
    # 如果test_series为False，则将预期结果转换为DataFrame对象

    if not as_index:
        expected = expected.reset_index()
        # 如果不按索引分组，则重置预期结果的索引

        if dtype is not None and dtype.startswith("Sparse"):
            expected["key"] = expected["key"].astype(dtype)
    # 如果dtype不为空且以"Sparse"开头，则将预期结果中的"key"列转换为指定的dtype类型

    tm.assert_equal(result, expected)
    # 使用pytest的测试辅助工具tm.assert_equal比较实际结果和预期结果


@pytest.mark.parametrize("test_series", [True, False])
# 使用pytest的参数化装饰器，定义一个测试函数，参数为test_series，取值为True和False
@pytest.mark.parametrize("dtype", [object, None])
# 使用pytest的参数化装饰器，定义一个测试函数，参数为dtype，取值为object和None
def test_null_is_null_for_dtype(
    sort, dtype, nulls_fixture, nulls_fixture2, test_series
):
    # 测试函数，用于验证对于dtype类型，null值应该保持不变

    # GH#48506 - groups should always result in using the null for the dtype
    # 关联GitHub的issue编号，验证分组应始终使用dtype的null值

    df = pd.DataFrame({"a": [1, 2]})
    # 创建包含单列"a"的DataFrame对象

    groups = pd.Series([nulls_fixture, nulls_fixture2], dtype=dtype)
    # 创建Series对象groups，包含dtype类型的null值

    obj = df["a"] if test_series else df
    # 如果test_series为True，则使用DataFrame的"a"列，否则使用整个DataFrame

    gb = obj.groupby(groups, dropna=False, sort=sort)
    # 根据groups对obj进行分组，保留NA值，根据sort参数决定是否排序

    result = gb.sum()
    # 计算分组后的求和结果

    index = pd.Index([na_value_for_dtype(groups.dtype)])
    # 创建索引对象index，包含dtype类型的null值

    expected = pd.DataFrame({"a": [3]}, index=index)
    # 创建预期的DataFrame对象，包含预期的求和结果

    if test_series:
        tm.assert_series_equal(result, expected["a"])
    else:
        tm.assert_frame_equal(result, expected)
    # 如果test_series为True，则比较实际结果和预期结果的Series对象；否则比较DataFrame对象


@pytest.mark.parametrize("index_kind", ["range", "single", "multi"])
# 使用pytest的参数化装饰器，定义一个测试函数，参数为index_kind，取值为"range"、"single"和"multi"
def test_categorical_reducers(reduction_func, observed, sort, as_index, index_kind):
    # 测试函数，用于验证分类约减函数
    # 确保通过将空值附加到末尾来至少包含一个空值
    values = np.append(np.random.default_rng(2).choice([1, 2, None], size=19), None)
    # 创建一个包含两列的 Pandas DataFrame，其中 'x' 是类别变量，'y' 是从 0 到 19 的整数
    df = pd.DataFrame(
        {"x": pd.Categorical(values, categories=[1, 2, 3]), "y": range(20)}
    )

    # 策略：通过填充空值来与 dropna=True 比较
    df_filled = df.copy()
    # 将 'x' 列的空值填充为新的类别码 4
    df_filled["x"] = pd.Categorical(values, categories=[1, 2, 3, 4]).fillna(4)

    if index_kind == "range":
        keys = ["x"]
    elif index_kind == "single":
        keys = ["x"]
        # 将 DataFrame 设置以 'x' 列作为索引
        df = df.set_index("x")
        df_filled = df_filled.set_index("x")
    else:
        keys = ["x", "x2"]
        # 在 DataFrame 中添加 'x2' 列，与 'x' 列相同
        df["x2"] = df["x"]
        df = df.set_index(["x", "x2"])
        df_filled["x2"] = df_filled["x"]
        df_filled = df_filled.set_index(["x", "x2"])

    # 获取聚合方法的参数
    args = get_groupby_method_args(reduction_func, df)
    args_filled = get_groupby_method_args(reduction_func, df_filled)

    if reduction_func == "corrwith" and index_kind == "range":
        # 不包括分组列，这样我们可以调用 reset_index
        args = (args[0].drop(columns=keys),)
        args_filled = (args_filled[0].drop(columns=keys),)

    # 创建一个以 keys 分组的 DataFrameGroupBy 对象，并保留 NaN 值，根据 observed、sort 和 as_index 参数设置
    gb_keepna = df.groupby(
        keys, dropna=False, observed=observed, sort=sort, as_index=as_index
    )

    if not observed and reduction_func in ["idxmin", "idxmax"]:
        # 如果不是 observed 并且 reduction_func 是 "idxmin" 或 "idxmax"，则引发异常
        with pytest.raises(
            ValueError, match="empty group due to unobserved categories"
        ):
            getattr(gb_keepna, reduction_func)(*args)
        return

    # 创建一个以 keys 分组的填充后的 DataFrameGroupBy 对象，根据 observed、sort 和 as_index 参数设置
    gb_filled = df_filled.groupby(keys, observed=observed, sort=sort, as_index=True)

    if reduction_func == "corrwith":
        warn = FutureWarning
        msg = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn = None
        msg = ""

    # 确保使用 assert_produces_warning 检查未来警告
    with tm.assert_produces_warning(warn, match=msg):
        # 调用指定的 reduction_func 方法，并将结果重置索引
        expected = getattr(gb_filled, reduction_func)(*args_filled).reset_index()

    # 从 'x' 列中移除类别码为 4 的值
    expected["x"] = expected["x"].cat.remove_categories([4])

    if index_kind == "multi":
        # 如果索引类型为多重索引，则从 'x2' 列中移除类别码为 4 的值
        expected["x2"] = expected["x2"].cat.remove_categories([4])

    if as_index:
        if index_kind == "multi":
            # 如果 as_index 为 True 并且索引类型为多重索引，则设置索引为 ["x", "x2"]
            expected = expected.set_index(["x", "x2"])
        else:
            # 如果 as_index 为 True 并且索引类型为单一索引，则设置索引为 "x"
            expected = expected.set_index("x")

    if reduction_func in ("idxmax", "idxmin") and index_kind != "range":
        # 如果 reduction_func 是 "idxmax" 或 "idxmin"，并且索引类型不是范围索引，则需要转换为索引值
        values = expected["y"].values.tolist()

        if index_kind == "single":
            # 如果索引类型为单一索引，则将值为 4 的元素转换为 NaN
            values = [np.nan if e == 4 else e for e in values]
            expected["y"] = pd.Categorical(values, categories=[1, 2, 3])
        else:
            # 如果索引类型为多重索引，则将值为 (4, 4) 的元素转换为 (NaN, NaN)
            values = [(np.nan, np.nan) if e == (4, 4) else e for e in values]
            expected["y"] = values
    # 如果使用的是 size 函数作为 reduction_func
    if reduction_func == "size":
        # 将列名 0 改为 "size"，以符合预期行为（见 GH#49519）
        expected = expected.rename(columns={0: "size"})
        # 如果指定了 as_index，则只保留 "size" 列并移除索引
        if as_index:
            expected = expected["size"].rename(None)

    # 如果 reduction_func 是 "corrwith"，则发出未来警告
    if reduction_func == "corrwith":
        warn = FutureWarning
        msg = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn = None
        msg = ""

    # 断言在执行 reduction_func 操作时会产生警告
    with tm.assert_produces_warning(warn, match=msg):
        # 调用 gb_keepna 对象的 reduction_func 方法执行操作
        result = getattr(gb_keepna, reduction_func)(*args)

    # 如果 reduction_func 是 "size"，则返回的是一个 Series，否则返回 DataFrame
    # 断言结果与期望值相等
    tm.assert_equal(result, expected)
# 定义一个测试函数，用于测试分类数据的转换器函数
def test_categorical_transformers(transformation_func, observed, sort, as_index):
    # GH#36327: GitHub issue reference
    # 创建一个包含随机选择值和空值的DataFrame
    values = np.append(np.random.default_rng(2).choice([1, 2, None], size=19), None)
    df = pd.DataFrame(
        {"x": pd.Categorical(values, categories=[1, 2, 3]), "y": range(20)}
    )
    # 获取特定转换函数的参数
    args = get_groupby_method_args(transformation_func, df)

    # 计算空组的结果
    null_group_values = df[df["x"].isnull()]["y"]
    if transformation_func == "cumcount":
        # 如果是累计计数函数，生成空组数据
        null_group_data = list(range(len(null_group_values)))
    elif transformation_func == "ngroup":
        if sort:
            if observed:
                # 计算非空值观察的分组数
                na_group = df["x"].nunique(dropna=False) - 1
            else:
                # TODO: 这里应该是3吗？
                na_group = df["x"].nunique(dropna=False) - 1
        else:
            # 计算非排序情况下的分组数
            na_group = df.iloc[: null_group_values.index[0]]["x"].nunique()
        null_group_data = len(null_group_values) * [na_group]
    else:
        # 对于其他转换函数，应用相应的方法
        null_group_data = getattr(null_group_values, transformation_func)(*args)
    null_group_result = pd.DataFrame({"y": null_group_data})

    # 按列 "x" 对DataFrame进行分组，保留空值
    gb_keepna = df.groupby(
        "x", dropna=False, observed=observed, sort=sort, as_index=as_index
    )
    # 按列 "x" 对DataFrame进行分组，删除空值
    gb_dropna = df.groupby("x", dropna=True, observed=observed, sort=sort)

    # 应用转换函数到保留空值的分组结果
    result = getattr(gb_keepna, transformation_func)(*args)
    # 应用转换函数到删除空值的分组结果
    expected = getattr(gb_dropna, transformation_func)(*args)

    # 将空组的计算结果填充到期望结果的相应位置
    for iloc, value in zip(
        df[df["x"].isnull()].index.tolist(), null_group_result.values.ravel()
    ):
        if expected.ndim == 1:
            expected.iloc[iloc] = value
        else:
            expected.iloc[iloc, 0] = value
    if transformation_func == "ngroup":
        # 调整非空值且大于或等于na_group的期望结果
        expected[df["x"].notnull() & expected.ge(na_group)] += 1
    if transformation_func not in ("rank", "diff", "pct_change", "shift"):
        # 将非上述转换函数的结果类型转换为int64
        expected = expected.astype("int64")

    # 使用测试模块中的方法验证结果和期望值是否相等
    tm.assert_equal(result, expected)


# 使用参数化装饰器指定多个测试方法，并为每个方法添加参数
@pytest.mark.parametrize("method", ["head", "tail"])
def test_categorical_head_tail(method, observed, sort, as_index):
    # GH#36327: GitHub issue reference
    # 创建一个包含随机选择值和空值的DataFrame
    values = np.random.default_rng(2).choice([1, 2, None], 30)
    df = pd.DataFrame(
        {"x": pd.Categorical(values, categories=[1, 2, 3]), "y": range(len(values))}
    )
    # 按列 "x" 对DataFrame进行分组，保留空值或者删除空值
    gb = df.groupby("x", dropna=False, observed=observed, sort=sort, as_index=as_index)
    # 根据指定的方法调用分组对象的方法
    result = getattr(gb, method)()

    if method == "tail":
        # 如果是"tail"方法，反转values数组
        values = values[::-1]
    # 从每个组中选择前5个值
    mask = (
        ((values == 1) & ((values == 1).cumsum() <= 5))
        | ((values == 2) & ((values == 2).cumsum() <= 5))
        | ((values == None) & ((values == None).cumsum() <= 5))  # noqa: E711
    )
    if method == "tail":
        # 如果是"tail"方法，反转mask数组
        mask = mask[::-1]
    # 根据mask选择DataFrame中的期望结果
    expected = df[mask]

    # 使用测试模块中的方法验证结果和期望值是否相等
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于测试分类数据的聚合函数
def test_categorical_agg():
    # GH#36327: GitHub issue reference
    # 创建一个包含随机选择值和空值的数组
    values = np.random.default_rng(2).choice([1, 2, None], 30)
    # 使用给定的值创建一个 Pandas DataFrame，包含两列：'x'列使用指定的类别创建分类变量，'y'列是连续递增的整数
    df = pd.DataFrame(
        {"x": pd.Categorical(values, categories=[1, 2, 3]), "y": range(len(values))}
    )
    
    # 根据 'x' 列进行分组，参数设置为不删除缺失值和不观察未出现的类别
    gb = df.groupby("x", dropna=False, observed=False)
    
    # 对分组后的数据进行聚合操作，对每个组内的数据执行求和操作
    result = gb.agg(lambda x: x.sum())
    
    # 对分组后的数据进行总和操作，得到期望的结果
    expected = gb.sum()
    
    # 使用测试工具比较两个 DataFrame 是否相等，预期是它们应当相等
    tm.assert_frame_equal(result, expected)
# 定义一个用于测试类别转换的函数
def test_categorical_transform():
    # GH#36327: 引用 GitHub 上的 issue 编号，用于跟踪问题
    # 生成一个包含随机选择值的数组，包括整数 1、2 和空值 None，共30个元素
    values = np.random.default_rng(2).choice([1, 2, None], 30)
    
    # 创建一个 DataFrame 对象 df，包含两列：
    # - "x" 列：使用 pd.Categorical 将 values 转换为分类数据，指定可能的类别为 [1, 2, 3]
    # - "y" 列：包含从0到29的整数，与 values 的长度相同
    df = pd.DataFrame(
        {"x": pd.Categorical(values, categories=[1, 2, 3]), "y": range(len(values))}
    )
    
    # 对 df 根据 "x" 列进行分组，设置 dropna=False 表示保留空值，observed=False 表示未观察到的值不会被包含
    gb = df.groupby("x", dropna=False, observed=False)
    
    # 使用 transform 方法对分组 gb 进行变换，传入 lambda 函数计算每个分组的和，存储在 result 中
    result = gb.transform(lambda x: x.sum())
    
    # 期望的结果通过 transform 方法传入字符串 "sum" 来计算每个分组的和，存储在 expected 中
    expected = gb.transform("sum")
    
    # 使用 tm.assert_frame_equal 函数断言 result 和 expected 两个 DataFrame 对象相等，即验证变换结果是否符合期望
    tm.assert_frame_equal(result, expected)
```