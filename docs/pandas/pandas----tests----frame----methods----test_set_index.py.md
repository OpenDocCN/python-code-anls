# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_set_index.py`

```
"""
See also: test_reindex.py:TestReindexSetIndex
"""

from datetime import (  # 导入 datetime 模块中的特定函数和类
    datetime,         # 日期时间类
    timedelta,        # 时间间隔类
)

import numpy as np   # 导入 NumPy 库，并使用 np 别名

import pytest         # 导入 pytest 测试框架

from pandas import (  # 从 pandas 库中导入多个类和函数
    Categorical,      # 类别数据类型
    CategoricalIndex, # 类别索引类
    DataFrame,        # 数据框类
    DatetimeIndex,    # 日期时间索引类
    Index,            # 索引类
    MultiIndex,       # 多重索引类
    Series,           # 系列类
    date_range,       # 日期范围函数
    period_range,     # 时期范围函数
    to_datetime,      # 转换为日期时间函数
)
import pandas._testing as tm  # 导入 pandas 测试模块，并使用 tm 别名


@pytest.fixture
def frame_of_index_cols():
    """
    Fixture for DataFrame of columns that can be used for indexing

    Columns are ['A', 'B', 'C', 'D', 'E', ('tuple', 'as', 'label')];
    'A' & 'B' contain duplicates (but are jointly unique), the rest are unique.

         A      B  C         D         E  (tuple, as, label)
    0  foo    one  a  0.608477 -0.012500           -1.664297
    1  foo    two  b -0.633460  0.249614           -0.364411
    2  foo  three  c  0.615256  2.154968           -0.834666
    3  bar    one  d  0.234246  1.085675            0.718445
    4  bar    two  e  0.533841 -0.005702           -3.533912
    """
    df = DataFrame(
        {
            "A": ["foo", "foo", "foo", "bar", "bar"],
            "B": ["one", "two", "three", "one", "two"],
            "C": ["a", "b", "c", "d", "e"],
            "D": np.random.default_rng(2).standard_normal(5),
            "E": np.random.default_rng(2).standard_normal(5),
            ("tuple", "as", "label"): np.random.default_rng(2).standard_normal(5),
        }
    )
    return df


class TestSetIndex:
    def test_set_index_multiindex(self):
        # segfault in GH#3308
        d = {"t1": [2, 2.5, 3], "t2": [4, 5, 6]}
        df = DataFrame(d)
        tuples = [(0, 1), (0, 2), (1, 2)]
        df["tuples"] = tuples

        index = MultiIndex.from_tuples(df["tuples"])
        # it works!
        df.set_index(index)

    def test_set_index_empty_column(self):
        # GH#1971
        df = DataFrame(
            [
                {"a": 1, "p": 0},
                {"a": 2, "m": 10},
                {"a": 3, "m": 11, "p": 20},
                {"a": 4, "m": 12, "p": 21},
            ],
            columns=["a", "m", "p", "x"],
        )

        result = df.set_index(["a", "x"])

        expected = df[["m", "p"]]
        expected.index = MultiIndex.from_arrays([df["a"], df["x"]], names=["a", "x"])
        tm.assert_frame_equal(result, expected)

    def test_set_index_empty_dataframe(self):
        # GH#38419
        df1 = DataFrame(
            {"a": Series(dtype="datetime64[ns]"), "b": Series(dtype="int64"), "c": []}
        )

        df2 = df1.set_index(["a", "b"])
        result = df2.index.to_frame().dtypes
        expected = df1[["a", "b"]].dtypes
        tm.assert_series_equal(result, expected)
    # 定义测试函数，用于测试DataFrame的set_index方法
    def test_set_index_multiindexcolumns(self):
        # 创建一个多级列索引
        columns = MultiIndex.from_tuples([("foo", 1), ("foo", 2), ("bar", 1)])
        # 创建一个随机数据的DataFrame，使用默认随机数生成器和指定的列索引
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)), columns=columns
        )

        # 对DataFrame使用set_index方法，以第一列作为新的索引
        result = df.set_index(df.columns[0])

        # 生成预期的DataFrame，包括剔除第一列后的数据和新的索引设置
        expected = df.iloc[:, 1:]
        expected.index = df.iloc[:, 0].values
        expected.index.names = [df.columns[0]]
        # 使用测试工具比较结果和预期是否相等
        tm.assert_frame_equal(result, expected)

    # 定义测试函数，测试在时区意识下Series保留时区信息的情况
    def test_set_index_timezone(self):
        # 创建一个时区意识的DatetimeIndex，转换时区为欧洲/罗马
        idx = DatetimeIndex(["2014-01-01 10:10:10"], tz="UTC").tz_convert("Europe/Rome")
        # 创建一个DataFrame，包含时区意识的Series "A"
        df = DataFrame({"A": idx})
        # 使用Series "A"作为索引，验证索引的小时是否为11点
        assert df.set_index(idx).index[0].hour == 11
        # 将Series "A"转换为DatetimeIndex，验证索引的小时是否为11点
        assert DatetimeIndex(Series(df.A))[0].hour == 11
        # 使用Series "A"作为索引，验证索引的小时是否为11点
        assert df.set_index(df.A).index[0].hour == 11

    # 定义测试函数，测试将列转换为DatetimeIndex的情况
    def test_set_index_cast_datetimeindex(self):
        # 创建一个DataFrame，包含"A"列为日期时间对象，"B"列为随机数值
        df = DataFrame(
            {
                "A": [datetime(2000, 1, 1) + timedelta(i) for i in range(1000)],
                "B": np.random.default_rng(2).standard_normal(1000),
            }
        )

        # 使用"A"列作为索引
        idf = df.set_index("A")
        # 验证索引是否为DatetimeIndex类型
        assert isinstance(idf.index, DatetimeIndex)

    # 定义测试函数，测试在夏令时时区下的索引设置情况
    def test_set_index_dst(self):
        # 创建一个时区为美国/太平洋的日期时间范围
        di = date_range("2006-10-29 00:00:00", periods=3, freq="h", tz="US/Pacific")

        # 创建一个包含"a"和"b"列的DataFrame，以日期时间范围作为索引，然后重置索引
        df = DataFrame(data={"a": [0, 1, 2], "b": [3, 4, 5]}, index=di).reset_index()
        # 对DataFrame使用"index"列作为单级索引
        res = df.set_index("index")
        # 创建预期的DataFrame，包含"a"和"b"列数据，以日期时间范围为索引，并移除频率信息
        exp = DataFrame(
            data={"a": [0, 1, 2], "b": [3, 4, 5]},
            index=Index(di, name="index"),
        )
        exp.index = exp.index._with_freq(None)
        # 使用测试工具比较结果和预期是否相等
        tm.assert_frame_equal(res, exp)

        # 创建预期的多级索引，包含日期时间范围和"a"列
        exp_index = MultiIndex.from_arrays([di, [0, 1, 2]], names=["index", "a"])
        exp = DataFrame({"b": [3, 4, 5]}, index=exp_index)
        # 使用测试工具比较结果和预期是否相等
        tm.assert_frame_equal(res, exp)

    # 定义测试函数，测试DataFrame的普通索引设置情况
    def test_set_index(self, float_string_frame):
        # 获取包含浮点数和字符串数据的DataFrame
        df = float_string_frame
        # 创建一个索引，该索引是逆序排列的整数
        idx = Index(np.arange(len(df) - 1, -1, -1, dtype=np.int64))

        # 使用逆序索引设置DataFrame的新索引
        df = df.set_index(idx)
        # 使用测试工具验证DataFrame的索引是否与预期的逆序索引相等
        tm.assert_index_equal(df.index, idx)
        # 使用断言验证在长度不匹配情况下是否抛出值错误异常
        with pytest.raises(ValueError, match="Length mismatch"):
            df.set_index(idx[::2])
    # 定义测试方法：测试设置索引名称
    def test_set_index_names(self):
        # 创建一个包含全为1的 DataFrame，包括四列 'ABCD'，十行 'i-0' 到 'i-9'
        df = DataFrame(
            np.ones((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(10)], dtype=object),
        )
        # 设置 DataFrame 的索引名称为 "name"
        df.index.name = "name"

        # 断言设置索引后的索引名称为 ["name"]
        assert df.set_index(df.index).index.names == ["name"]

        # 从 DataFrame 的 ['A', 'B'] 列创建一个 MultiIndex，并指定列名为 ["A", "B"]
        mi = MultiIndex.from_arrays(df[["A", "B"]].T.values, names=["A", "B"])
        # 从 DataFrame 的 ['A', 'B', 'A', 'B'] 列创建一个 MultiIndex，并指定列名为 ["A", "B", "C", "D"]
        mi2 = MultiIndex.from_arrays(
            df[["A", "B", "A", "B"]].T.values, names=["A", "B", "C", "D"]
        )

        # 将 DataFrame 按列 ["A", "B"] 设置为索引
        df = df.set_index(["A", "B"])

        # 断言设置索引后的索引名称为 ["A", "B"]
        assert df.set_index(df.index).index.names == ["A", "B"]

        # 检查 set_index 是否将 MultiIndex 转换成 Index
        assert isinstance(df.set_index(df.index).index, MultiIndex)

        # 检查索引的实际相等性
        tm.assert_index_equal(df.set_index(df.index).index, mi)

        # 将当前索引重命名为 ["C", "D"]
        idx2 = df.index.rename(["C", "D"])

        # 检查 [MultiIndex, MultiIndex] 设置索引是否返回 MultiIndex 而不是元组对
        assert isinstance(df.set_index([df.index, idx2]).index, MultiIndex)

        # 检查设置索引后的实际相等性
        tm.assert_index_equal(df.set_index([df.index, idx2]).index, mi2)

    # A 列具有重复值，C 列没有
    @pytest.mark.parametrize("keys", ["A", "C", ["A", "B"], ("tuple", "as", "label")])
    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("drop", [True, False])
    # 定义测试方法：测试设置索引并处理 inplace 操作和 drop 参数
    def test_set_index_drop_inplace(self, frame_of_index_cols, drop, inplace, keys):
        # 从给定的 DataFrame 创建 df
        df = frame_of_index_cols

        # 如果 keys 是列表，则使用 MultiIndex 从列表中的列创建索引，指定列名为 keys
        if isinstance(keys, list):
            idx = MultiIndex.from_arrays([df[x] for x in keys], names=keys)
        else:
            # 否则，使用 Index 从单个列 keys 创建索引，指定列名为 keys
            idx = Index(df[keys], name=keys)
        
        # 如果 drop 为 True，则从 df 中删除 keys 指定的列；否则保持不变
        expected = df.drop(keys, axis=1) if drop else df
        expected.index = idx

        # 如果 inplace 为 True，则进行原地修改并返回 None；否则返回新的 DataFrame
        if inplace:
            result = df.copy()
            return_value = result.set_index(keys, drop=drop, inplace=True)
            assert return_value is None
        else:
            result = df.set_index(keys, drop=drop)

        # 断言设置索引后的 DataFrame 与预期的 DataFrame 相等
        tm.assert_frame_equal(result, expected)

    # A 列具有重复值，C 列没有
    @pytest.mark.parametrize("keys", ["A", "C", ["A", "B"], ("tuple", "as", "label")])
    @pytest.mark.parametrize("drop", [True, False])
    # 定义测试方法：测试追加索引
    def test_set_index_append(self, frame_of_index_cols, drop, keys):
        # 从给定的 DataFrame 创建 df
        df = frame_of_index_cols

        # 如果 keys 是列表，则将 df.index 与 keys 指定的列合并为 MultiIndex，列名为 [None] + keys
        keys = keys if isinstance(keys, list) else [keys]
        idx = MultiIndex.from_arrays(
            [df.index] + [df[x] for x in keys], names=[None] + keys
        )
        
        # 如果 drop 为 True，则从 df 中删除 keys 指定的列；否则保持不变
        expected = df.drop(keys, axis=1) if drop else df.copy()
        expected.index = idx

        # 对 df 进行追加索引操作，并返回结果
        result = df.set_index(keys, drop=drop, append=True)

        # 断言追加索引后的 DataFrame 与预期的 DataFrame 相等
        tm.assert_frame_equal(result, expected)
    # 定义一个测试方法，用于测试在现有多级索引上追加索引列的设置行为
    def test_set_index_append_to_multiindex(self, frame_of_index_cols, drop, keys):
        # 将索引列"D"追加到现有的多级索引上，返回新的DataFrame
        df = frame_of_index_cols.set_index(["D"], drop=drop, append=True)

        # 如果keys不是列表，则将其转换为列表
        keys = keys if isinstance(keys, list) else [keys]
        # 期望的结果DataFrame，将"D"和keys列表作为索引列追加到原有的多级索引上
        expected = frame_of_index_cols.set_index(["D"] + keys, drop=drop, append=True)

        # 对df再次设置索引列keys，保持追加到现有多级索引的行为
        result = df.set_index(keys, drop=drop, append=True)

        # 使用测试框架的函数来比较结果DataFrame和期望的DataFrame是否相等
        tm.assert_frame_equal(result, expected)

    # 测试设置索引列后的数据框是否符合预期
    def test_set_index_after_mutation(self):
        # 创建一个包含"val"和"key"列的DataFrame
        df = DataFrame({"val": [0, 1, 2], "key": ["a", "b", "c"]})
        # 期望的结果DataFrame，设置"key"列为索引，并仅保留索引为"b"和"c"的行
        expected = DataFrame({"val": [1, 2]}, index=["b", "c"], columns=["val"])

        # 对df进行行选择后，设置"key"列为索引
        df2 = df.loc[df.index.map(lambda indx: indx >= 1)]
        result = df2.set_index("key")

        # 使用测试框架的函数来比较结果DataFrame和期望的DataFrame是否相等
        tm.assert_frame_equal(result, expected)

    # 多级索引构造函数不能直接用于Series -> lambda表达式
    # 添加列表列表构造函数，因为列表具有歧义性 -> lambda表达式
    # 同时测试如果append=True时的索引名称（这里的名称对于B是重复的）
    @pytest.mark.parametrize(
        "box",
        [
            Series,
            Index,
            np.array,
            list,
            lambda x: [list(x)],
            lambda x: MultiIndex.from_arrays([x]),
        ],
    )
    @pytest.mark.parametrize(
        "append, index_name", [(True, None), (True, "B"), (True, "test"), (False, None)]
    )
    @pytest.mark.parametrize("drop", [True, False])
    # 测试设置单一数组作为索引列的行为
    def test_set_index_pass_single_array(
        self, frame_of_index_cols, drop, append, index_name, box
    ):
        # 获取待测试的DataFrame
        df = frame_of_index_cols
        # 设置索引的名称为index_name
        df.index.name = index_name

        # 使用box函数处理"B"列数据，如果box为列表，会触发KeyError异常
        key = box(df["B"])
        if box == list:
            # 当box为列表时，字符串列表会被解释为键列表，预期会触发KeyError异常
            msg = "['one', 'two', 'three', 'one', 'two']"
            with pytest.raises(KeyError, match=msg):
                df.set_index(key, drop=drop, append=append)
        else:
            # 当box为np.array或列表列表时，"B"列的名称可能丢失
            name_mi = getattr(key, "names", None)
            name = [getattr(key, "name", None)] if name_mi is None else name_mi

            # 设置key列作为索引，保持追加到现有多级索引的行为
            result = df.set_index(key, drop=drop, append=append)

            # 期望的结果DataFrame，将"B"列作为索引列，保持追加到现有多级索引的行为
            expected = df.set_index(["B"], drop=False, append=append)
            expected.index.names = [index_name] + name if append else name

            # 使用测试框架的函数来比较结果DataFrame和期望的DataFrame是否相等
            tm.assert_frame_equal(result, expected)

    # 多级索引构造函数不能直接用于Series -> lambda表达式
    # 同时测试如果append=True时的索引名称（这里的名称对于A和B是重复的）
    @pytest.mark.parametrize(
        "box", [Series, Index, np.array, list, lambda x: MultiIndex.from_arrays([x])]
    )
    @pytest.mark.parametrize(
        "append, index_name",
        [(True, None), (True, "A"), (True, "B"), (True, "test"), (False, None)],
    )
    @pytest.mark.parametrize("drop", [True, False])
    # 定义一个测试方法，用于测试设置索引时处理数组的情况
    def test_set_index_pass_arrays(
        self, frame_of_index_cols, drop, append, index_name, box
    ):
        # 将输入参数赋值给本地变量 df
        df = frame_of_index_cols
        # 设置 DataFrame 的索引名称为 index_name
        df.index.name = index_name

        # 创建索引键列表，包括 "A" 和通过 box 处理后的 "B"
        keys = ["A", box(df["B"])]
        
        # 根据 box 的类型决定索引名称列表 names
        # 如果 box 是 np.array/list/tuple/iter 中的一种，则 B 的名称为 None
        names = ["A", None if box in [np.array, list, tuple, iter] else "B"]

        # 使用指定的键设置索引，根据参数 drop 和 append 进行操作
        result = df.set_index(keys, drop=drop, append=append)

        # 构建预期的 DataFrame，设置索引为 ["A", "B"]，根据 drop 决定是否删除 "A" 列
        expected = df.set_index(["A", "B"], drop=False, append=append)
        expected = expected.drop("A", axis=1) if drop else expected
        # 根据 append 决定是否在索引名称前加上 index_name，结合 names 列表
        expected.index.names = [index_name] + names if append else names

        # 断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # MultiIndex 构造函数不能直接应用于 Series -> lambda
    # 我们还模拟了一个标签的 "构造函数" -> lambda
    # 如果 append=True，还测试索引名称是否重复（A）
    @pytest.mark.parametrize(
        "box2",
        [
            Series,
            Index,
            np.array,
            list,
            iter,
            lambda x: MultiIndex.from_arrays([x]),
            lambda x: x.name,
        ],
    )
    @pytest.mark.parametrize(
        "box1",
        [
            Series,
            Index,
            np.array,
            list,
            iter,
            lambda x: MultiIndex.from_arrays([x]),
            lambda x: x.name,
        ],
    )
    @pytest.mark.parametrize(
        "append, index_name", [(True, None), (True, "A"), (True, "test"), (False, None)]
    )
    @pytest.mark.parametrize("drop", [True, False])
    # 定义另一个测试方法，用于测试设置索引时处理数组的情况（带有重复情况）
    def test_set_index_pass_arrays_duplicate(
        self, frame_of_index_cols, drop, append, index_name, box1, box2
    ):
        # 将输入参数赋值给本地变量 df
        df = frame_of_index_cols
        # 设置 DataFrame 的索引名称为 index_name
        df.index.name = index_name

        # 创建索引键列表，包括 box1 处理后的 "A" 和 box2 处理后的 "A"
        keys = [box1(df["A"]), box2(df["A"])]
        
        # 使用指定的键设置索引，根据参数 drop 和 append 进行操作
        result = df.set_index(keys, drop=drop, append=append)

        # 如果任一 box 是 iter，则它已被使用；重新读取 keys
        keys = [box1(df["A"]), box2(df["A"])]

        # 需要适应第一个 drop 的情况，即当两个 keys 都为 'A' 时无法重复删除相同列
        first_drop = (
            False
            if (
                isinstance(keys[0], str)
                and keys[0] == "A"
                and isinstance(keys[1], str)
                and keys[1] == "A"
            )
            else drop
        )
        # 为了测试已经测试过的行为，我们顺序添加，因此第二个 append 始终为 True；
        # 必须将 keys 包装在列表中，否则 box = list 会被解释为 keys
        expected = df.set_index([keys[0]], drop=first_drop, append=append)
        expected = expected.set_index([keys[1]], drop=drop, append=True)
        # 断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
    # 使用 pytest.mark.parametrize 装饰器标记测试函数，以多个参数值运行测试函数
    @pytest.mark.parametrize("drop", [True, False])
    # 定义测试函数 test_set_index_pass_multiindex，接受 frame_of_index_cols、drop 和 append 作为参数
    def test_set_index_pass_multiindex(self, frame_of_index_cols, drop, append):
        # 从参数中获取数据框 df
        df = frame_of_index_cols
        # 从数据框 df 的列 "A" 和 "B" 创建 MultiIndex 对象 keys，指定名称为 "A" 和 "B"
        keys = MultiIndex.from_arrays([df["A"], df["B"]], names=["A", "B"])

        # 使用 set_index 方法设置数据框 df 的索引为 keys，根据 drop 和 append 参数决定是否丢弃现有列和追加索引
        result = df.set_index(keys, drop=drop, append=append)

        # 创建预期结果 expected，使用 set_index 方法设置数据框 df 的索引为 ["A", "B"]，始终保留列不丢弃
        expected = df.set_index(["A", "B"], drop=False, append=append)

        # 使用 assert_frame_equal 方法比较 result 和 expected 是否相等，用于测试断言
        tm.assert_frame_equal(result, expected)

    # 定义测试函数 test_construction_with_categorical_index，不接受任何参数
    def test_construction_with_categorical_index(self):
        # 创建名为 B 的分类索引 ci，其中的值为列表 "ab" 重复 5 次
        ci = CategoricalIndex(list("ab") * 5, name="B")

        # 使用 DataFrame 构造函数创建数据框 df，包含列 "A" 和 "B"，其中 "B" 的值来自分类索引 ci
        df = DataFrame(
            {"A": np.random.default_rng(2).standard_normal(10), "B": ci.values}
        )
        # 使用 set_index 方法将数据框 df 按列 "B" 设置为索引 idf
        idf = df.set_index("B")
        # 使用 assert_index_equal 方法比较 idf 的索引是否与 ci 相等，用于测试断言
        tm.assert_index_equal(idf.index, ci)

        # 使用 DataFrame 构造函数创建数据框 df，包含列 "A" 和 "B"，其中 "B" 是分类索引 ci
        df = DataFrame({"A": np.random.default_rng(2).standard_normal(10), "B": ci})
        # 使用 set_index 方法将数据框 df 按列 "B" 设置为索引 idf
        idf = df.set_index("B")
        # 使用 assert_index_equal 方法比较 idf 的索引是否与 ci 相等，用于测试断言
        tm.assert_index_equal(idf.index, ci)

        # 对 idf 执行重置索引操作，然后再次按列 "B" 设置为索引 idf
        idf = idf.reset_index().set_index("B")
        # 使用 assert_index_equal 方法比较 idf 的索引是否与 ci 相等，用于测试断言
        tm.assert_index_equal(idf.index, ci)

    # 定义测试函数 test_set_index_preserve_categorical_dtype，不接受任何参数
    def test_set_index_preserve_categorical_dtype(self):
        # 创建数据框 df，包含列 "A"、"B"、"C1" 和 "C2"
        df = DataFrame(
            {
                "A": [1, 2, 1, 1, 2],
                "B": [10, 16, 22, 28, 34],
                # 创建非有序分类列 C1，使用列表 "abaab" 和类别列表 "bac"
                "C1": Categorical(list("abaab"), categories=list("bac"), ordered=False),
                # 创建有序分类列 C2，使用列表 "abaab" 和类别列表 "bac"
                "C2": Categorical(list("abaab"), categories=list("bac"), ordered=True),
            }
        )
        # 对于每个 cols 组合，执行以下操作
        for cols in ["C1", "C2", ["A", "C1"], ["A", "C2"], ["C1", "C2"]]:
            # 使用 set_index 方法将数据框 df 按 cols 列设置为索引，然后重置索引为 result
            result = df.set_index(cols).reset_index()
            # 重新索引 result 的列，以与 df 的列相同
            result = result.reindex(columns=df.columns)
            # 使用 assert_frame_equal 方法比较 result 和 df 是否相等，用于测试断言
            tm.assert_frame_equal(result, df)
    def test_set_index_datetime(self):
        # 测试用例名称：test_set_index_datetime
        # GH#3950：GitHub 上的 issue 编号

        # 创建一个 DataFrame 对象，包含三列：label, datetime, value
        df = DataFrame(
            {
                "label": ["a", "a", "a", "b", "b", "b"],
                "datetime": [
                    "2011-07-19 07:00:00",
                    "2011-07-19 08:00:00",
                    "2011-07-19 09:00:00",
                    "2011-07-19 07:00:00",
                    "2011-07-19 08:00:00",
                    "2011-07-19 09:00:00",
                ],
                "value": range(6),
            }
        )
        
        # 将 datetime 列转换为 DateTimeIndex，并设置时区为 UTC
        df.index = to_datetime(df.pop("datetime"), utc=True)
        
        # 将 DateTimeIndex 的时区转换为 "US/Pacific"
        df.index = df.index.tz_convert("US/Pacific")

        # 预期的 DateTimeIndex，时区为 UTC 转换为 "US/Pacific"
        expected = DatetimeIndex(
            ["2011-07-19 07:00:00", "2011-07-19 08:00:00", "2011-07-19 09:00:00"],
            name="datetime",
        )
        expected = expected.tz_localize("UTC").tz_convert("US/Pacific")

        # 将 label 列作为索引，追加到现有索引的末尾
        df = df.set_index("label", append=True)
        
        # 检查索引的第一级是否与预期的一致
        tm.assert_index_equal(df.index.levels[0], expected)
        
        # 检查索引的第二级是否包含 ["a", "b"]，名称为 "label"
        tm.assert_index_equal(df.index.levels[1], Index(["a", "b"], name="label"))
        
        # 断言索引的名称为 ["datetime", "label"]
        assert df.index.names == ["datetime", "label"]

        # 交换索引的第一级和第二级
        df = df.swaplevel(0, 1)
        
        # 检查交换后索引的第一级是否为 ["a", "b"]，名称为 "label"
        tm.assert_index_equal(df.index.levels[0], Index(["a", "b"], name="label"))
        
        # 检查交换后索引的第二级是否与预期的一致
        tm.assert_index_equal(df.index.levels[1], expected)
        
        # 断言交换后索引的名称为 ["label", "datetime"]

        assert df.index.names == ["label", "datetime"]

        # 创建一个新的 DataFrame 对象，包含随机生成的数据
        df = DataFrame(np.random.default_rng(2).random(6))
        
        # 创建三个不同的 DateTimeIndex 对象，分别为 idx1, idx2, idx3
        idx1 = DatetimeIndex(
            [
                "2011-07-19 07:00:00",
                "2011-07-19 08:00:00",
                "2011-07-19 09:00:00",
                "2011-07-19 07:00:00",
                "2011-07-19 08:00:00",
                "2011-07-19 09:00:00",
            ],
            tz="US/Eastern",
        )
        idx2 = DatetimeIndex(
            [
                "2012-04-01 09:00",
                "2012-04-01 09:00",
                "2012-04-01 09:00",
                "2012-04-02 09:00",
                "2012-04-02 09:00",
                "2012-04-02 09:00",
            ],
            tz="US/Eastern",
        )
        idx3 = date_range("2011-01-01 09:00", periods=6, tz="Asia/Tokyo")
        
        # 移除 idx3 对象的频率信息
        idx3 = idx3._with_freq(None)

        # 将 DataFrame 的索引依次设置为 idx1, idx2, idx3，并追加到已有索引的末尾
        df = df.set_index(idx1)
        df = df.set_index(idx2, append=True)
        df = df.set_index(idx3, append=True)

        # 预期的第一级索引为 expected1，时区为 "US/Eastern"
        expected1 = DatetimeIndex(
            ["2011-07-19 07:00:00", "2011-07-19 08:00:00", "2011-07-19 09:00:00"],
            tz="US/Eastern",
        )
        
        # 预期的第二级索引为 expected2，时区为 "US/Eastern"
        expected2 = DatetimeIndex(
            ["2012-04-01 09:00", "2012-04-02 09:00"], tz="US/Eastern"
        )

        # 检查 DataFrame 索引的每个级别是否与预期一致
        tm.assert_index_equal(df.index.levels[0], expected1)
        tm.assert_index_equal(df.index.levels[1], expected2)
        tm.assert_index_equal(df.index.levels[2], idx3)

        # GH#7092：GitHub 上的 issue 编号

        # 检查 DataFrame 索引的每个级别是否分别等于 idx1, idx2, idx3
        tm.assert_index_equal(df.index.get_level_values(0), idx1)
        tm.assert_index_equal(df.index.get_level_values(1), idx2)
        tm.assert_index_equal(df.index.get_level_values(2), idx3)
    # 定义测试方法 test_set_index_period
    def test_set_index_period(self):
        # 标识此测试相关的 GitHub issue 编号为 6631
        # 创建一个 DataFrame，填充随机生成的数据，使用默认随机数生成器种子为 2
        df = DataFrame(np.random.default_rng(2).random(6))
        
        # 创建一个时间区间索引 idx1，从"2011-01-01"开始，包含3个月，频率为每月
        idx1 = period_range("2011-01-01", periods=3, freq="M")
        # 将 idx1 重复一次并附加到自身
        idx1 = idx1.append(idx1)
        
        # 创建一个时间区间索引 idx2，从"2013-01-01 09:00"开始，包含2小时，频率为每小时
        idx2 = period_range("2013-01-01 09:00", periods=2, freq="h")
        # 将 idx2 重复三次并附加到自身
        idx2 = idx2.append(idx2).append(idx2)
        
        # 创建一个年频率的时间区间索引 idx3，从"2005"开始，包含6年
        idx3 = period_range("2005", periods=6, freq="Y")

        # 将 df 的索引设置为 idx1，原地修改 DataFrame
        df = df.set_index(idx1)
        # 将 df 的索引设置为 idx2，追加到已有的索引层级中
        df = df.set_index(idx2, append=True)
        # 将 df 的索引设置为 idx3，追加到已有的索引层级中
        df = df.set_index(idx3, append=True)

        # 预期的第一层级索引应该与 expected1 相等
        expected1 = period_range("2011-01-01", periods=3, freq="M")
        tm.assert_index_equal(df.index.levels[0], expected1)
        
        # 预期的第二层级索引应该与 expected2 相等
        expected2 = period_range("2013-01-01 09:00", periods=2, freq="h")
        tm.assert_index_equal(df.index.levels[1], expected2)
        
        # 第三层级索引应该与 idx3 相等
        tm.assert_index_equal(df.index.levels[2], idx3)

        # 获取 df 的第一层级索引值，应该与 idx1 相等
        tm.assert_index_equal(df.index.get_level_values(0), idx1)
        
        # 获取 df 的第二层级索引值，应该与 idx2 相等
        tm.assert_index_equal(df.index.get_level_values(1), idx2)
        
        # 获取 df 的第三层级索引值，应该与 idx3 相等
        tm.assert_index_equal(df.index.get_level_values(2), idx3)
class TestSetIndexInvalid:
    # 测试在设置索引时验证完整性
    def test_set_index_verify_integrity(self, frame_of_index_cols):
        # 获取测试用的数据框架
        df = frame_of_index_cols

        # 期望引发 ValueError 异常，且异常消息为 "Index has duplicate keys"
        with pytest.raises(ValueError, match="Index has duplicate keys"):
            # 尝试以 "A" 列为索引，验证完整性为 True
            df.set_index("A", verify_integrity=True)
        
        # 当使用 MultiIndex 时
        with pytest.raises(ValueError, match="Index has duplicate keys"):
            # 尝试以 [df["A"], df["A"]] 为索引，验证完整性为 True
            df.set_index([df["A"], df["A"]], verify_integrity=True)

    # 测试在设置索引时引发 KeyError 异常
    @pytest.mark.parametrize("append", [True, False])
    @pytest.mark.parametrize("drop", [True, False])
    def test_set_index_raise_keys(self, frame_of_index_cols, drop, append):
        # 获取测试用的数据框架
        df = frame_of_index_cols

        with pytest.raises(KeyError, match="['foo', 'bar', 'baz']"):
            # 尝试以 ["foo", "bar", "baz"] 作为索引，设置 drop 和 append 参数
            df.set_index(["foo", "bar", "baz"], drop=drop, append=append)

        with pytest.raises(KeyError, match="X"):
            # 尝试以 [df["A"], df["B"], "X"] 作为索引，设置 drop 和 append 参数
            df.set_index([df["A"], df["B"], "X"], drop=drop, append=append)

        msg = "[('foo', 'foo', 'foo', 'bar', 'bar')]"
        with pytest.raises(KeyError, match=msg):
            # 尝试以 tuple(df["A"]) 作为索引，设置 drop 和 append 参数
            df.set_index(tuple(df["A"]), drop=drop, append=append)

        with pytest.raises(KeyError, match=msg):
            # 尝试以 ["A", df["A"], tuple(df["A"])] 作为索引，设置 drop 和 append 参数
            df.set_index(["A", df["A"], tuple(df["A"])], drop=drop, append=append)

    # 测试在设置索引时引发 TypeError 异常
    @pytest.mark.parametrize("append", [True, False])
    @pytest.mark.parametrize("drop", [True, False])
    def test_set_index_raise_on_type(self, frame_of_index_cols, drop, append):
        # 设置 box 变量为 set
        box = set
        # 获取测试用的数据框架
        df = frame_of_index_cols

        msg = 'The parameter "keys" may be a column key, .*'
        with pytest.raises(TypeError, match=msg):
            # 尝试以 box(df["A"]) 作为索引，设置 drop 和 append 参数
            df.set_index(box(df["A"]), drop=drop, append=append)

        with pytest.raises(TypeError, match=msg):
            # 尝试以 ["A", df["A"], box(df["A"])] 作为索引，设置 drop 和 append 参数
            df.set_index(["A", df["A"], box(df["A"])], drop=drop, append=append)

    # 测试在设置索引时引发异常，根据长度不匹配
    @pytest.mark.parametrize(
        "box",
        [Series, Index, np.array, iter, lambda x: MultiIndex.from_arrays([x])],
        ids=["Series", "Index", "np.array", "iter", "MultiIndex"],
    )
    @pytest.mark.parametrize("length", [4, 6], ids=["too_short", "too_long"])
    @pytest.mark.parametrize("append", [True, False])
    @pytest.mark.parametrize("drop", [True, False])
    def test_set_index_raise_on_len(
        self, frame_of_index_cols, box, length, drop, append
    ):
        # 获取测试用的数据框架
        df = frame_of_index_cols

        # 定义异常消息格式
        msg = 'Length mismatch: Expected axis has {} elements, new values have {} elements'
        # 根据不同的 box 函数设置索引，预期引发 ValueError 异常
        with pytest.raises(ValueError, match=msg.format(len(df), length)):
            df.set_index(box(range(length)), drop=drop, append=append)
    ):
        # GH 24984
        # 将 frame_of_index_cols 赋值给 df，这个变量长度为 5
        df = frame_of_index_cols  # has length 5

        # 使用 numpy 生成器创建一个长度为 length 的随机整数数组
        values = np.random.default_rng(2).integers(0, 10, (length,))

        # 定义一个错误消息，用于匹配长度不匹配的异常信息
        msg = "Length mismatch: Expected 5 rows, received array of length.*"

        # 使用 pytest 检查设置索引时直接传入长度不匹配的 values 的情况，预期会引发 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            df.set_index(box(values), drop=drop, append=append)

        # 使用 pytest 检查设置索引时传入列表中存在长度不匹配的元素的情况，预期会引发 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            df.set_index(["A", df.A, box(values)], drop=drop, append=append)
class TestSetIndexCustomLabelType:
    def test_set_index_custom_label_type(self):
        # GH#24969

        # 定义一个名为Thing的类，具有name和color两个属性
        class Thing:
            def __init__(self, name, color) -> None:
                self.name = name
                self.color = color

            # 返回对象的字符串表示，格式为"<Thing {name}>"
            def __str__(self) -> str:
                return f"<Thing {self.name!r}>"

            # 将__repr__方法重写为__str__的内容，用于更好地处理KeyError异常
            __repr__ = __str__

        # 创建两个Thing对象
        thing1 = Thing("One", "red")
        thing2 = Thing("Two", "blue")

        # 创建一个DataFrame对象，包含两个Thing对象作为列名
        df = DataFrame({thing1: [0, 1], thing2: [2, 3]})

        # 创建一个期望的DataFrame对象，指定thing2作为索引名
        expected = DataFrame({thing1: [0, 1]}, index=Index([2, 3], name=thing2))

        # 使用thing2作为自定义标签直接设置索引
        result = df.set_index(thing2)
        tm.assert_frame_equal(result, expected)

        # 使用包含thing2的列表作为自定义标签设置索引
        result = df.set_index([thing2])
        tm.assert_frame_equal(result, expected)

        # 准备一个新的Thing对象作为缺失的键
        thing3 = Thing("Three", "pink")
        msg = "<Thing 'Three'>"

        # 使用pytest检查是否会引发KeyError，并匹配预期的消息
        with pytest.raises(KeyError, match=msg):
            # 直接使用缺失的标签设置索引
            df.set_index(thing3)

        with pytest.raises(KeyError, match=msg):
            # 使用包含缺失标签的列表设置索引
            df.set_index([thing3])

    def test_set_index_custom_label_hashable_iterable(self):
        # GH#24969

        # 实际示例讨论在GH 24984，例如shapely.geometry对象（例如一组Points）
        # 可以既是可哈希又是可迭代的；这里使用frozenset作为测试的替代品

        # 定义一个名为Thing的类，继承自frozenset，并稳定其__repr__方法以处理KeyError
        class Thing(frozenset):
            def __repr__(self) -> str:
                tmp = sorted(self)
                joined_reprs = ", ".join(map(repr, tmp))
                return f"frozenset({{{joined_reprs}}})"

        # 创建两个Thing对象
        thing1 = Thing(["One", "red"])
        thing2 = Thing(["Two", "blue"])

        # 创建一个DataFrame对象，包含两个Thing对象作为列名
        df = DataFrame({thing1: [0, 1], thing2: [2, 3]})

        # 创建一个期望的DataFrame对象，指定thing2作为索引名
        expected = DataFrame({thing1: [0, 1]}, index=Index([2, 3], name=thing2))

        # 使用thing2作为自定义标签直接设置索引
        result = df.set_index(thing2)
        tm.assert_frame_equal(result, expected)

        # 使用包含thing2的列表作为自定义标签设置索引
        result = df.set_index([thing2])
        tm.assert_frame_equal(result, expected)

        # 准备一个新的Thing对象作为缺失的键
        thing3 = Thing(["Three", "pink"])
        msg = r"frozenset\(\{'Three', 'pink'\}\)"

        # 使用pytest检查是否会引发KeyError，并匹配预期的消息
        with pytest.raises(KeyError, match=msg):
            # 直接使用缺失的标签设置索引
            df.set_index(thing3)

        with pytest.raises(KeyError, match=msg):
            # 使用包含缺失标签的列表设置索引
            df.set_index([thing3])
    def test_set_index_custom_label_type_raises():
        # 定义一个测试函数，用于测试设置索引时使用自定义标签引发异常的情况
        # GH#24969

        # 定义一个继承自不可哈希类型的类
        class Thing(set):
            def __init__(self, name, color) -> None:
                self.name = name
                self.color = color

            def __str__(self) -> str:
                return f"<Thing {self.name!r}>"

        # 创建两个自定义对象
        thing1 = Thing("One", "red")
        thing2 = Thing("Two", "blue")
        
        # 创建一个 DataFrame，使用自定义对象作为列名
        df = DataFrame([[0, 2], [1, 3]], columns=[thing1, thing2])

        # 设置错误消息的正则表达式模式
        msg = 'The parameter "keys" may be a column key, .*'

        # 测试设置索引时是否会抛出 TypeError 异常，并检查是否匹配指定的错误消息
        with pytest.raises(TypeError, match=msg):
            # 直接使用自定义标签进行索引设置
            df.set_index(thing2)

        # 再次测试，这次使用包含自定义标签的列表进行索引设置
        with pytest.raises(TypeError, match=msg):
            df.set_index([thing2])

    def test_set_index_periodindex():
        # 定义一个测试函数，用于测试设置 PeriodIndex 作为索引的情况
        # GH#6631

        # 创建一个包含随机数据的 DataFrame
        df = DataFrame(np.random.default_rng(2).random(6))

        # 创建两个不同的 PeriodIndex
        idx1 = period_range("2011/01/01", periods=6, freq="M")
        idx2 = period_range("2013", periods=6, freq="Y")

        # 测试将 idx1 设置为索引后，索引是否相等
        df = df.set_index(idx1)
        tm.assert_index_equal(df.index, idx1)

        # 再次测试将 idx2 设置为索引后，索引是否相等
        df = df.set_index(idx2)
        tm.assert_index_equal(df.index, idx2)
```