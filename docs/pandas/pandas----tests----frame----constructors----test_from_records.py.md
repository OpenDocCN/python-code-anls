# `D:\src\scipysrc\pandas\pandas\tests\frame\constructors\test_from_records.py`

```
    # 导入必要的模块和库
    from collections.abc import Iterator
    from datetime import (
        datetime,
        timezone,
    )
    from decimal import Decimal

    import numpy as np  # 导入 NumPy 库
    import pytest  # 导入 Pytest 库

    from pandas._config import using_pyarrow_string_dtype  # 导入 pandas 内部配置模块
    from pandas.compat import is_platform_little_endian  # 导入 pandas 兼容性模块

    from pandas import (  # 导入 pandas 的多个重要组件
        CategoricalIndex,
        DataFrame,
        Index,
        Interval,
        RangeIndex,
        Series,
    )
    import pandas._testing as tm  # 导入 pandas 测试模块


class TestFromRecords:
    def test_from_records_dt64tz_frame(self):
        # GH#51697 测试用例标识：处理 DataFrame 的 from_records 方法，针对 datetime64 与时区的情况
        df = DataFrame({"a": [1, 2, 3]})
        with pytest.raises(TypeError, match="not supported"):
            DataFrame.from_records(df)

    def test_from_records_with_datetimes(self):
        # this may fail on certain platforms because of a numpy issue
        # related GH#6140 测试用例标识：处理包含日期时间的 from_records 方法
        if not is_platform_little_endian():
            pytest.skip("known failure of test on non-little endian")

        # construction with a null in a recarray
        # GH#6140 用具有空值的 recarray 构造 DataFrame
        expected = DataFrame({"EXPIRY": [datetime(2005, 3, 1, 0, 0), None]})

        arrdata = [np.array([datetime(2005, 3, 1, 0, 0), None])]
        dtypes = [("EXPIRY", "<M8[us]")]

        recarray = np.rec.fromarrays(arrdata, dtype=dtypes)

        result = DataFrame.from_records(recarray)
        tm.assert_frame_equal(result, expected)

        # coercion should work too 强制转换应该也能正常工作
        arrdata = [np.array([datetime(2005, 3, 1, 0, 0), None])]
        dtypes = [("EXPIRY", "<M8[m]")]
        recarray = np.rec.fromarrays(arrdata, dtype=dtypes)
        result = DataFrame.from_records(recarray)
        # we get the closest supported unit, "s" 得到最接近的支持单位，即 "s"
        expected["EXPIRY"] = expected["EXPIRY"].astype("M8[s]")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.skipif(
        using_pyarrow_string_dtype(), reason="dtype checking logic doesn't work"
    )
    def test_from_records_sequencelike_empty(self):
        # empty case 空情况测试用例：from_records 处理空数据集的情况
        result = DataFrame.from_records([], columns=["foo", "bar", "baz"])
        assert len(result) == 0
        tm.assert_index_equal(result.columns, Index(["foo", "bar", "baz"]))

        result = DataFrame.from_records([])
        assert len(result) == 0
        assert len(result.columns) == 0
    # 定义一个测试方法，用于测试从字典样式数据结构创建 DataFrame 的功能
    def test_from_records_dictlike(self):
        # 创建一个 DataFrame 对象，包含多列数据
        df = DataFrame(
            {
                "A": np.array(
                    np.random.default_rng(2).standard_normal(6), dtype=np.float64
                ),
                "A1": np.array(
                    np.random.default_rng(2).standard_normal(6), dtype=np.float64
                ),
                "B": np.array(np.arange(6), dtype=np.int64),
                "C": ["foo"] * 6,
                "D": np.array([True, False] * 3, dtype=bool),
                "E": np.array(
                    np.random.default_rng(2).standard_normal(6), dtype=np.float32
                ),
                "E1": np.array(
                    np.random.default_rng(2).standard_normal(6), dtype=np.float32
                ),
                "F": np.array(np.arange(6), dtype=np.int32),
            }
        )

        # 获取 DataFrame 内部的数据块字典
        blocks = df._to_dict_of_blocks()
        # 创建一个空列表用于存储所有列名
        columns = []
        # 遍历数据块字典的每个值（数据块）
        for b in blocks.values():
            # 将每个数据块的列名扩展到 columns 列表中
            columns.extend(b.columns)

        # 将 DataFrame 转换为字典，每个列名映射到其对应的 Series 对象
        asdict = dict(df.items())
        # 将 DataFrame 转换为字典，每个列名映射到其对应的 ndarray 对象的值数组
        asdict2 = {x: y.values for x, y in df.items()}

        # 创建一个空列表，用于存储不同方式从字典创建的 DataFrame 对象
        results = []
        # 将第一种字典形式的数据转换为 DataFrame，然后重新索引列名
        results.append(DataFrame.from_records(asdict).reindex(columns=df.columns))
        # 将第二种字典形式的数据转换为 DataFrame，指定列名顺序，并重新索引列名
        results.append(
            DataFrame.from_records(asdict, columns=columns).reindex(columns=df.columns)
        )
        # 将第三种字典形式的数据转换为 DataFrame，指定列名顺序，并重新索引列名
        results.append(
            DataFrame.from_records(asdict2, columns=columns).reindex(columns=df.columns)
        )

        # 遍历所有结果 DataFrame，确保它们与原始 DataFrame 相等
        for r in results:
            tm.assert_frame_equal(r, df)

    # 定义一个测试方法，用于测试从非元组数据创建 DataFrame 的功能
    def test_from_records_non_tuple(self):
        # 定义一个模拟记录的类
        class Record:
            def __init__(self, *args) -> None:
                self.args = args

            def __getitem__(self, i):
                return self.args[i]

            def __iter__(self) -> Iterator:
                return iter(self.args)

        # 创建一组记录对象
        recs = [Record(1, 2, 3), Record(4, 5, 6), Record(7, 8, 9)]
        # 将每个记录对象转换为元组
        tups = [tuple(rec) for rec in recs]

        # 使用记录对象列表创建 DataFrame
        result = DataFrame.from_records(recs)
        # 使用元组列表创建预期的 DataFrame
        expected = DataFrame.from_records(tups)
        # 断言结果 DataFrame 与预期 DataFrame 相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，用于测试从空记录列表创建 DataFrame 的功能，并指定列名
    def test_from_records_len0_with_columns(self):
        # GH#2633
        # 从空记录列表创建 DataFrame，指定索引和列名
        result = DataFrame.from_records([], index="foo", columns=["foo", "bar"])
        # 预期结果是空的索引对象
        expected = Index(["bar"])

        # 断言结果 DataFrame 的长度为 0
        assert len(result) == 0
        # 断言结果 DataFrame 的索引名称为 "foo"
        assert result.index.name == "foo"
        # 断言结果 DataFrame 的列与预期的列相等
        tm.assert_index_equal(result.columns, expected)

    # 定义一个测试方法，用于测试从 Series、列表、字典混合数据结构创建 DataFrame 的功能
    def test_from_records_series_list_dict(self):
        # GH#27358
        # 创建预期的 DataFrame，包含一个列，每行为一个包含字典的列表
        expected = DataFrame([[{"a": 1, "b": 2}, {"a": 3, "b": 4}]]).T
        # 创建一个 Series 对象，每个元素为一个包含字典的列表
        data = Series([[{"a": 1, "b": 2}], [{"a": 3, "b": 4}]])
        # 使用 Series 对象创建 DataFrame
        result = DataFrame.from_records(data)
        # 断言结果 DataFrame 与预期 DataFrame 相等
        tm.assert_frame_equal(result, expected)
    def test_from_records_series_categorical_index(self):
        # 测试函数：从系列数据创建 DataFrame，使用分类索引
        # GH#32805
        # 创建一个分类索引对象，包含三个区间对象
        index = CategoricalIndex(
            [Interval(-20, -10), Interval(-10, 0), Interval(0, 10)]
        )
        # 创建一个系列，每个元素是一个字典，使用上面创建的分类索引作为索引
        series_of_dicts = Series([{"a": 1}, {"a": 2}, {"b": 3}], index=index)
        # 使用 from_records 方法从系列数据创建 DataFrame，指定索引为分类索引
        frame = DataFrame.from_records(series_of_dicts, index=index)
        # 创建一个预期的 DataFrame，包含两列 'a' 和 'b'，NaN 表示缺失值
        expected = DataFrame(
            {"a": [1, 2, np.nan], "b": [np.nan, np.nan, 3]}, index=index
        )
        # 使用 assert_frame_equal 检查生成的 DataFrame 和预期的是否相等
        tm.assert_frame_equal(frame, expected)

    def test_frame_from_records_utc(self):
        # 测试函数：从记录中创建 DataFrame，使用 UTC 时区
        # 创建一个记录字典，包含浮点数和带有时区信息的日期时间
        rec = {"datum": 1.5, "begin_time": datetime(2006, 4, 27, tzinfo=timezone.utc)}
        # 使用 from_records 方法从记录列表中创建 DataFrame，指定索引为 "begin_time"
        # 注意，此处使用字符串 "begin_time" 作为索引名称
        DataFrame.from_records([rec], index="begin_time")

    def test_from_records_to_records(self):
        # 测试函数：从记录数组创建 DataFrame，并进行相应操作
        # from numpy documentation
        # 创建一个结构化数组 arr，包含整数、浮点数和字符串类型的数据
        arr = np.zeros((2,), dtype=("i4,f4,S10"))
        arr[:] = [(1, 2.0, "Hello"), (2, 3.0, "World")]

        # 使用 from_records 方法从结构化数组创建 DataFrame
        DataFrame.from_records(arr)

        # 创建一个索引对象，包含逆序的数组长度
        index = Index(np.arange(len(arr))[::-1])
        # 使用 from_records 方法从结构化数组创建 DataFrame，并指定索引为 index
        indexed_frame = DataFrame.from_records(arr, index=index)
        # 使用 assert_index_equal 检查生成的 DataFrame 的索引是否与预期的 index 相等
        tm.assert_index_equal(indexed_frame.index, index)

        # 当结构化数组没有字段名时，应该使用最后的备选方案
        arr2 = np.zeros((2, 3))
        # 使用 from_records 方法从数组 arr2 创建 DataFrame，应与手动创建的 DataFrame 相等
        tm.assert_frame_equal(DataFrame.from_records(arr2), DataFrame(arr2))

        # 当长度不匹配时，应该抛出 ValueError 异常
        msg = "|".join(
            [
                r"Length of values \(2\) does not match length of index \(1\)",
            ]
        )
        with pytest.raises(ValueError, match=msg):
            # 使用 from_records 方法从结构化数组创建 DataFrame，并指定索引为 index[:-1]
            DataFrame.from_records(arr, index=index[:-1])

        # 使用 from_records 方法从 DataFrame 创建记录数组，并指定索引为 "f1"
        indexed_frame = DataFrame.from_records(arr, index="f1")

        # 检查生成的记录数组的字段名数量是否为 3
        records = indexed_frame.to_records()
        assert len(records.dtype.names) == 3

        # 检查生成的记录数组的字段名数量是否为 2，且不包含 "index" 字段
        records = indexed_frame.to_records(index=False)
        assert len(records.dtype.names) == 2
        assert "index" not in records.dtype.names

    def test_from_records_nones(self):
        # 测试函数：处理包含 None 值的元组列表，创建 DataFrame
        # 创建一个包含元组的列表，其中包括 None 值
        tuples = [(1, 2, None, 3), (1, 2, None, 3), (None, 2, 5, 3)]

        # 使用 from_records 方法从元组列表创建 DataFrame，并指定列名为 ["a", "b", "c", "d"]
        df = DataFrame.from_records(tuples, columns=["a", "b", "c", "d"])
        # 断言 DataFrame 中 "c" 列的第一个元素是否为 NaN
        assert np.isnan(df["c"][0])
    # 定义一个测试方法，用于测试从记录迭代器创建DataFrame的功能
    def test_from_records_iterator(self):
        # 创建一个NumPy数组，包含四个元组，每个元组有四个元素
        arr = np.array(
            [(1.0, 1.0, 2, 2), (3.0, 3.0, 4, 4), (5.0, 5.0, 6, 6), (7.0, 7.0, 8, 8)],
            dtype=[
                ("x", np.float64),
                ("u", np.float32),
                ("y", np.int64),
                ("z", np.int32),
            ],
        )
        # 从数组的迭代器创建DataFrame，仅使用前两行
        df = DataFrame.from_records(iter(arr), nrows=2)
        # 创建一个期望的DataFrame，只包含前两行的内容
        xp = DataFrame(
            {
                "x": np.array([1.0, 3.0], dtype=np.float64),
                "u": np.array([1.0, 3.0], dtype=np.float32),
                "y": np.array([2, 4], dtype=np.int64),
                "z": np.array([2, 4], dtype=np.int32),
            }
        )
        # 断言两个DataFrame在重建索引后是否相等
        tm.assert_frame_equal(df.reindex_like(xp), xp)

        # 在这里未指定dtype，因此与默认的DataFrame进行比较
        # 创建一个包含元组的列表，用于创建DataFrame
        arr = [(1.0, 2), (3.0, 4), (5.0, 6), (7.0, 8)]
        # 从列表的迭代器创建DataFrame，指定列名为["x", "y"]，仅使用前两行
        df = DataFrame.from_records(iter(arr), columns=["x", "y"], nrows=2)
        # 断言两个DataFrame是否相等，忽略数据类型的检查
        tm.assert_frame_equal(df, xp.reindex(columns=["x", "y"]), check_dtype=False)

    # 定义一个测试方法，用于测试从元组生成器创建DataFrame的功能
    def test_from_records_tuples_generator(self):
        # 定义一个生成器函数，生成包含元组的数据
        def tuple_generator(length):
            for i in range(length):
                letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                yield (i, letters[i % len(letters)], i / length)

        # 列名列表
        columns_names = ["Integer", "String", "Float"]
        # 使用生成器函数创建包含多列数据的列表
        columns = [
            [i[j] for i in tuple_generator(10)] for j in range(len(columns_names))
        ]
        # 创建一个期望的DataFrame，包含生成器函数生成的数据
        data = {"Integer": columns[0], "String": columns[1], "Float": columns[2]}
        expected = DataFrame(data, columns=columns_names)

        # 使用生成器对象创建DataFrame，指定列名
        generator = tuple_generator(10)
        result = DataFrame.from_records(generator, columns=columns_names)
        # 断言生成的DataFrame与期望的DataFrame是否相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，用于测试从列表生成器创建DataFrame的功能
    def test_from_records_lists_generator(self):
        # 定义一个生成器函数，生成包含列表的数据
        def list_generator(length):
            for i in range(length):
                letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                yield [i, letters[i % len(letters)], i / length]

        # 列名列表
        columns_names = ["Integer", "String", "Float"]
        # 使用生成器函数创建包含多列数据的列表
        columns = [
            [i[j] for i in list_generator(10)] for j in range(len(columns_names))
        ]
        # 创建一个期望的DataFrame，包含生成器函数生成的数据
        data = {"Integer": columns[0], "String": columns[1], "Float": columns[2]}
        expected = DataFrame(data, columns=columns_names)

        # 使用生成器对象创建DataFrame，指定列名
        generator = list_generator(10)
        result = DataFrame.from_records(generator, columns=columns_names)
        # 断言生成的DataFrame与期望的DataFrame是否相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，用于测试从记录创建DataFrame时列名不被修改的功能
    def test_from_records_columns_not_modified(self):
        # 定义包含元组的列表
        tuples = [(1, 2, 3), (1, 2, 3), (2, 5, 3)]

        # 列名列表
        columns = ["a", "b", "c"]
        # 备份原始列名
        original_columns = list(columns)

        # 从元组列表创建DataFrame，指定列名和索引
        DataFrame.from_records(tuples, columns=columns, index="a")

        # 断言列名列表未被修改
        assert columns == original_columns
    # 测试从元组列表创建 DataFrame，其中每个元组包含一个 Decimal 对象
    def test_from_records_decimal(self):
        # 创建包含 Decimal 对象的元组列表
        tuples = [(Decimal("1.5"),), (Decimal("2.5"),), (None,)]

        # 使用元组列表创建 DataFrame，指定列名为 ["a"]
        df = DataFrame.from_records(tuples, columns=["a"])
        # 断言 DataFrame 的列 "a" 的数据类型为 object
        assert df["a"].dtype == object

        # 使用元组列表创建 DataFrame，指定列名为 ["a"]，并强制转换为 float 类型
        df = DataFrame.from_records(tuples, columns=["a"], coerce_float=True)
        # 断言 DataFrame 的列 "a" 的数据类型为 np.float64
        assert df["a"].dtype == np.float64
        # 断言 DataFrame 的列 "a" 的最后一个值为 NaN
        assert np.isnan(df["a"].values[-1])

    # 测试从元组列表创建 DataFrame，包含重复的列名
    def test_from_records_duplicates(self):
        # 使用包含重复列名的元组列表创建 DataFrame，指定列名为 ["a", "b", "a"]
        result = DataFrame.from_records([(1, 2, 3), (4, 5, 6)], columns=["a", "b", "a"])

        # 创建预期的 DataFrame，包含相同的数据和列名
        expected = DataFrame([(1, 2, 3), (4, 5, 6)], columns=["a", "b", "a"])

        # 使用测试工具比较两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    # 测试从字典列表创建 DataFrame，并设置单个索引名
    def test_from_records_set_index_name(self):
        # 定义一个创建字典的函数，用于生成示例数据
        def create_dict(order_id):
            return {
                "order_id": order_id,
                "quantity": np.random.default_rng(2).integers(1, 10),
                "price": np.random.default_rng(2).integers(1, 10),
            }

        # 创建包含多个字典的列表作为示例数据
        documents = [create_dict(i) for i in range(10)]
        # 向列表中添加一个字典，演示缺失数据情况
        documents.append({"order_id": 10, "quantity": 5})

        # 使用字典列表创建 DataFrame，并设置 "order_id" 作为索引
        result = DataFrame.from_records(documents, index="order_id")
        # 断言 DataFrame 的索引名为 "order_id"
        assert result.index.name == "order_id"

        # 使用字典列表创建 DataFrame，并设置多列作为多级索引
        result = DataFrame.from_records(documents, index=["order_id", "quantity"])
        # 断言 DataFrame 的索引名为 ("order_id", "quantity")
        assert result.index.names == ("order_id", "quantity")

    # 测试从字典或列表创建 DataFrame，处理不规范数据
    def test_from_records_misc_brokenness(self):
        # GH#2179

        # 创建包含字典的数据，其中键为整数，值为列表
        data = {1: ["foo"], 2: ["bar"]}

        # 使用字典创建 DataFrame，并指定列名为 ["a", "b"]
        result = DataFrame.from_records(data, columns=["a", "b"])
        # 创建预期的 DataFrame，包含相同的数据和列名
        exp = DataFrame(data, columns=["a", "b"])
        # 使用测试工具比较两个 DataFrame 是否相等
        tm.assert_frame_equal(result, exp)

        # 处理索引与索引名重叠的情况

        # 创建包含键为列名，值为列表的字典数据
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}

        # 使用字典创建 DataFrame，并将 ["a", "b", "c"] 设置为索引
        result = DataFrame.from_records(data, index=["a", "b", "c"])
        # 创建预期的 DataFrame，包含相同的数据和索引
        exp = DataFrame(data, index=["a", "b", "c"])
        # 使用测试工具比较两个 DataFrame 是否相等
        tm.assert_frame_equal(result, exp)

    # 测试从列表创建 DataFrame，处理特定情况下的数据
    def test_from_records_misc_brokenness2(self):
        # GH#2623
        # 创建包含多个列表的列表，每个列表包含日期和数值
        rows = []
        rows.append([datetime(2010, 1, 1), 1])
        rows.append([datetime(2010, 1, 2), "hi"])  # 测试将列升级为对象类型

        # 使用列表创建 DataFrame，并指定列名为 ["date", "test"]
        result = DataFrame.from_records(rows, columns=["date", "test"])
        # 创建预期的 DataFrame，包含相同的数据和列名
        expected = DataFrame(
            {"date": [row[0] for row in rows], "test": [row[1] for row in rows]}
        )
        # 使用测试工具比较两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)
        # 断言 DataFrame 的 "test" 列的数据类型为对象类型
        assert result.dtypes["test"] == np.dtype(object)

    # 测试从列表创建 DataFrame，处理另一种特定情况下的数据
    def test_from_records_misc_brokenness3(self):
        # 创建包含多个列表的列表，每个列表包含日期和数值
        rows = []
        rows.append([datetime(2010, 1, 1), 1])
        rows.append([datetime(2010, 1, 2), 1])

        # 使用列表创建 DataFrame，并指定列名为 ["date", "test"]
        result = DataFrame.from_records(rows, columns=["date", "test"])
        # 创建预期的 DataFrame，包含相同的数据和列名
        expected = DataFrame(
            {"date": [row[0] for row in rows], "test": [row[1] for row in rows]}
        )
        # 使用测试工具比较两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)
    def test_from_records_empty(self):
        # 测试空记录情况，GH#3562
        # 创建空记录的DataFrame，指定列名为["a", "b", "c"]
        result = DataFrame.from_records([], columns=["a", "b", "c"])
        # 创建预期的空DataFrame，指定列名为["a", "b", "c"]
        expected = DataFrame(columns=["a", "b", "c"])
        # 断言两个DataFrame是否相等
        tm.assert_frame_equal(result, expected)

        # 创建空记录的DataFrame，指定列名为["a", "b", "b"]
        result = DataFrame.from_records([], columns=["a", "b", "b"])
        # 创建预期的空DataFrame，指定列名为["a", "b", "b"]
        expected = DataFrame(columns=["a", "b", "b"])
        # 断言两个DataFrame是否相等
        tm.assert_frame_equal(result, expected)

    def test_from_records_empty_with_nonempty_fields_gh3682(self):
        # GH#3682，测试带非空字段的空记录情况
        # 创建NumPy数组a，包含一个记录，字段为("id", np.int64), ("value", np.int64)
        a = np.array([(1, 2)], dtype=[("id", np.int64), ("value", np.int64)])
        # 从记录创建DataFrame，指定"id"字段作为索引
        df = DataFrame.from_records(a, index="id")

        # 创建预期的DataFrame，包含索引为[1]，列为["value"]，值为[2]
        ex_index = Index([1], name="id")
        expected = DataFrame({"value": [2]}, index=ex_index, columns=["value"])
        # 断言两个DataFrame是否相等
        tm.assert_frame_equal(df, expected)

        # 获取空切片b为数组a的前0个记录
        b = a[:0]
        # 从记录创建DataFrame，指定"id"字段作为索引
        df2 = DataFrame.from_records(b, index="id")
        # 断言两个DataFrame的前0个记录是否相等
        tm.assert_frame_equal(df2, df.iloc[:0])

    def test_from_records_empty2(self):
        # GH#42456，第二个空记录测试情况
        # 定义dtype为[("prop", int)]
        dtype = [("prop", int)]
        # 创建形状为(0, len(dtype))的空数组arr，指定dtype
        shape = (0, len(dtype))
        arr = np.empty(shape, dtype=dtype)

        # 从记录创建DataFrame
        result = DataFrame.from_records(arr)
        # 创建预期的DataFrame，包含一个列"prop"，值为空的int数组
        expected = DataFrame({"prop": np.array([], dtype=int)})
        # 断言两个DataFrame是否相等
        tm.assert_frame_equal(result, expected)

        # 从数组创建DataFrame，指定dtype
        alt = DataFrame(arr)
        # 断言两个DataFrame是否相等
        tm.assert_frame_equal(alt, expected)
```