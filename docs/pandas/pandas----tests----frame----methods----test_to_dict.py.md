# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_to_dict.py`

```
from collections import (
    OrderedDict,
    defaultdict,
)
from datetime import (
    datetime,
    timezone,
)

import numpy as np
import pytest

from pandas import (
    NA,
    DataFrame,
    Index,
    Interval,
    MultiIndex,
    Period,
    Series,
    Timedelta,
    Timestamp,
)
import pandas._testing as tm


class TestDataFrameToDict:
    def test_to_dict_timestamp(self):
        # GH#11247
        # 对于 datetime64[ns] 类型的 DataFrame，使用 to_dict('records') 方法会将 np.datetime64 转换为 Timestamp
        tsmp = Timestamp("20130101")
        test_data = DataFrame({"A": [tsmp, tsmp], "B": [tsmp, tsmp]})
        test_data_mixed = DataFrame({"A": [tsmp, tsmp], "B": [1, 2]})

        expected_records = [{"A": tsmp, "B": tsmp}, {"A": tsmp, "B": tsmp}]
        expected_records_mixed = [{"A": tsmp, "B": 1}, {"A": tsmp, "B": 2}]

        assert test_data.to_dict(orient="records") == expected_records
        assert test_data_mixed.to_dict(orient="records") == expected_records_mixed

        expected_series = {
            "A": Series([tsmp, tsmp], name="A"),
            "B": Series([tsmp, tsmp], name="B"),
        }
        expected_series_mixed = {
            "A": Series([tsmp, tsmp], name="A"),
            "B": Series([1, 2], name="B"),
        }

        # 使用 pandas._testing 模块的 assert_dict_equal 方法验证 to_dict('series') 方法的输出
        tm.assert_dict_equal(test_data.to_dict(orient="series"), expected_series)
        tm.assert_dict_equal(
            test_data_mixed.to_dict(orient="series"), expected_series_mixed
        )

        expected_split = {
            "index": [0, 1],
            "data": [[tsmp, tsmp], [tsmp, tsmp]],
            "columns": ["A", "B"],
        }
        expected_split_mixed = {
            "index": [0, 1],
            "data": [[tsmp, 1], [tsmp, 2]],
            "columns": ["A", "B"],
        }

        # 使用 pandas._testing 模块的 assert_dict_equal 方法验证 to_dict('split') 方法的输出
        tm.assert_dict_equal(test_data.to_dict(orient="split"), expected_split)
        tm.assert_dict_equal(
            test_data_mixed.to_dict(orient="split"), expected_split_mixed
        )

    def test_to_dict_index_not_unique_with_index_orient(self):
        # GH#22801
        # 当索引不唯一时，使用 orient='index' 时会导致数据丢失，抛出 ValueError 异常
        df = DataFrame({"a": [1, 2], "b": [0.5, 0.75]}, index=["A", "A"])
        msg = "DataFrame index must be unique for orient='index'"
        with pytest.raises(ValueError, match=msg):
            df.to_dict(orient="index")

    def test_to_dict_invalid_orient(self):
        # 当指定了不支持的 orient 参数时，应该抛出 ValueError 异常
        df = DataFrame({"A": [0, 1]})
        msg = "orient 'xinvalid' not understood"
        with pytest.raises(ValueError, match=msg):
            df.to_dict(orient="xinvalid")

    @pytest.mark.parametrize("orient", ["d", "l", "r", "sp", "s", "i"])
    def test_to_dict_short_orient_raises(self, orient):
        # GH#32515
        # 当指定了不支持的 orient 参数时，应该抛出 ValueError 异常
        df = DataFrame({"A": [0, 1]})
        with pytest.raises(ValueError, match="not understood"):
            df.to_dict(orient=orient)

    @pytest.mark.parametrize("mapping", [dict, defaultdict(list), OrderedDict])
    # 定义一个测试方法，用于测试DataFrame对象的to_dict方法
    def test_to_dict(self, mapping):
        # orient=参数应仅接受列出的选项
        # 参见GH#32515

        # 准备测试数据，包含两个字典作为示例
        test_data = {"A": {"1": 1, "2": 2}, "B": {"1": "1", "2": "2", "3": "3"}}

        # 使用DataFrame将测试数据转换为字典，指定转换方式为传入的mapping参数
        recons_data = DataFrame(test_data).to_dict(into=mapping)

        # 遍历原始测试数据，验证转换后的数据与原始数据一致性
        for k, v in test_data.items():
            for k2, v2 in v.items():
                assert v2 == recons_data[k][k2]

        # 再次使用DataFrame将测试数据转换为字典，这次指定转换方式为"list"
        recons_data = DataFrame(test_data).to_dict("list", into=mapping)

        # 遍历原始测试数据，验证转换后的数据与原始数据一致性
        for k, v in test_data.items():
            for k2, v2 in v.items():
                assert v2 == recons_data[k][int(k2) - 1]

        # 使用DataFrame将测试数据转换为字典，转换方式为"series"
        recons_data = DataFrame(test_data).to_dict("series", into=mapping)

        # 遍历原始测试数据，验证转换后的数据与原始数据一致性
        for k, v in test_data.items():
            for k2, v2 in v.items():
                assert v2 == recons_data[k][k2]

        # 使用DataFrame将测试数据转换为字典，转换方式为"split"
        recons_data = DataFrame(test_data).to_dict("split", into=mapping)

        # 预期的split转换结果
        expected_split = {
            "columns": ["A", "B"],
            "index": ["1", "2", "3"],
            "data": [[1.0, "1"], [2.0, "2"], [np.nan, "3"]],
        }

        # 使用pytest的断言函数验证转换后的数据与预期结果一致
        tm.assert_dict_equal(recons_data, expected_split)

        # 使用DataFrame将测试数据转换为字典，转换方式为"records"
        recons_data = DataFrame(test_data).to_dict("records", into=mapping)

        # 预期的records转换结果
        expected_records = [
            {"A": 1.0, "B": "1"},
            {"A": 2.0, "B": "2"},
            {"A": np.nan, "B": "3"},
        ]

        # 使用普通断言验证转换后的数据类型为列表，并且长度为3，然后逐项比较转换后的数据与预期结果
        assert isinstance(recons_data, list)
        assert len(recons_data) == 3
        for left, right in zip(recons_data, expected_records):
            tm.assert_dict_equal(left, right)

        # GH#10844
        # 使用DataFrame将测试数据转换为字典，转换方式为"index"
        recons_data = DataFrame(test_data).to_dict("index")

        # 遍历原始测试数据，验证转换后的数据与原始数据一致性
        for k, v in test_data.items():
            for k2, v2 in v.items():
                assert v2 == recons_data[k2][k]

        # 创建一个DataFrame对象，并在其上进行操作
        df = DataFrame(test_data)
        # 在DataFrame中添加一个名为"duped"的列，该列复制第一列的数据
        df["duped"] = df[df.columns[0]]
        # 将操作后的DataFrame转换为字典，转换方式为"index"
        recons_data = df.to_dict("index")
        # 复制测试数据，将"duped"列添加到复制后的数据中
        comp_data = test_data.copy()
        comp_data["duped"] = comp_data[df.columns[0]]
        # 遍历复制后的数据，验证转换后的数据与预期结果一致性
        for k, v in comp_data.items():
            for k2, v2 in v.items():
                assert v2 == recons_data[k2][k]

    # 使用pytest的参数化装饰器，对test_to_dict方法传入不同的mapping参数进行测试
    @pytest.mark.parametrize("mapping", [list, defaultdict, []])
    def test_to_dict_errors(self, mapping):
        # GH#16122
        # 创建一个DataFrame对象，包含随机生成的3x3数组
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)))
        # 准备错误消息内容
        msg = "|".join(
            [
                "unsupported type: <class 'list'>",
                r"to_dict\(\) only accepts initialized defaultdicts",
            ]
        )
        # 使用pytest的上下文管理器验证调用to_dict方法时是否会抛出TypeError，并匹配错误消息
        with pytest.raises(TypeError, match=msg):
            df.to_dict(into=mapping)

    # 定义一个测试方法，用于测试当DataFrame中存在非唯一列名时，调用to_dict方法会抛出UserWarning的情况
    def test_to_dict_not_unique_warning(self):
        # GH#16927: 当转换为字典时，如果列名非唯一，将会被忽略，并抛出警告
        # 创建一个DataFrame对象，包含一个包含非唯一列名的行
        df = DataFrame([[1, 2, 3]], columns=["a", "a", "b"])
        # 使用pytest的上下文管理器验证调用to_dict方法是否会产生UserWarning，并匹配警告消息
        with tm.assert_produces_warning(UserWarning, match="columns will be omitted"):
            df.to_dict()

    # 使用pytest的标记，忽略掉这个测试方法中的UserWarning类警告
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.parametrize(
        "orient,expected",
        [
            ("list", {"A": [2, 5], "B": [3, 6]}),
            ("dict", {"A": {0: 2, 1: 5}, "B": {0: 3, 1: 6}}),
        ],
    )
    # 定义测试函数 test_to_dict_not_unique，使用参数化测试来验证不唯一列的行为
    def test_to_dict_not_unique(self, orient, expected):
        # GH#54824: This is to make sure that dataframes with non-unique column
        # would have uniform behavior throughout different orients
        # 创建包含非唯一列的 DataFrame 对象
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "A", "B"])
        # 调用 DataFrame 的 to_dict 方法，根据给定的 orient 转换成字典
        result = df.to_dict(orient)
        # 断言转换结果与预期结果相等
        assert result == expected

    # orient - to_dict 函数的 orient 参数
    # item_getter - 用于从结果字典中提取值的函数，接受列名和索引作为参数
    @pytest.mark.parametrize(
        "orient,item_getter",
        [
            ("dict", lambda d, col, idx: d[col][idx]),
            ("records", lambda d, col, idx: d[idx][col]),
            ("list", lambda d, col, idx: d[col][idx]),
            ("split", lambda d, col, idx: d["data"][idx][d["columns"].index(col)]),
            ("index", lambda d, col, idx: d[idx][col]),
        ],
    )
    # 定义测试函数 test_to_dict_box_scalars，使用参数化测试验证各种 orient 下的转换行为
    def test_to_dict_box_scalars(self, orient, item_getter):
        # GH#14216, GH#23753
        # 确保正确的数据转换
        # 创建包含不同类型数据的 DataFrame 对象
        df = DataFrame({"a": [1, 2], "b": [0.1, 0.2]})
        # 调用 DataFrame 的 to_dict 方法，指定 orient 参数进行数据转换
        result = df.to_dict(orient=orient)
        # 断言转换后的数据类型符合预期
        assert isinstance(item_getter(result, "a", 0), int)
        assert isinstance(item_getter(result, "b", 0), float)

    # 定义测试函数 test_to_dict_tz，验证 orient='records' 下的时区感知日期时间列的转换
    def test_to_dict_tz(self):
        # GH#18372 When converting to dict with orient='records' columns of
        # datetime that are tz-aware were not converted to required arrays
        # 创建包含时区感知日期时间数据的 DataFrame 对象
        data = [
            (datetime(2017, 11, 18, 21, 53, 0, 219225, tzinfo=timezone.utc),),
            (datetime(2017, 11, 18, 22, 6, 30, 61810, tzinfo=timezone.utc),),
        ]
        df = DataFrame(list(data), columns=["d"])

        # 调用 DataFrame 的 to_dict 方法，指定 orient='records' 进行转换
        result = df.to_dict(orient="records")
        # 预期的转换结果
        expected = [
            {"d": Timestamp("2017-11-18 21:53:00.219225+0000", tz=timezone.utc)},
            {"d": Timestamp("2017-11-18 22:06:30.061810+0000", tz=timezone.utc)},
        ]
        # 使用 pandas.testing 模块的 assert_dict_equal 方法进行断言
        tm.assert_dict_equal(result[0], expected[0])
        tm.assert_dict_equal(result[1], expected[1])
    @pytest.mark.parametrize(
        "into, expected",
        [  # 参数化测试用例，测试不同的参数组合
            (
                dict,
                {  # 当 into 参数为 dict 时的预期输出
                    0: {"int_col": 1, "float_col": 1.0},
                    1: {"int_col": 2, "float_col": 2.0},
                    2: {"int_col": 3, "float_col": 3.0},
                },
            ),
            (
                OrderedDict,
                OrderedDict(  # 当 into 参数为 OrderedDict 时的预期输出
                    [
                        (0, {"int_col": 1, "float_col": 1.0}),
                        (1, {"int_col": 2, "float_col": 2.0}),
                        (2, {"int_col": 3, "float_col": 3.0}),
                    ]
                ),
            ),
            (
                defaultdict(dict),
                defaultdict(  # 当 into 参数为 defaultdict(dict) 时的预期输出
                    dict,
                    {
                        0: {"int_col": 1, "float_col": 1.0},
                        1: {"int_col": 2, "float_col": 2.0},
                        2: {"int_col": 3, "float_col": 3.0},
                    },
                ),
            ),
        ],
    )
    def test_to_dict_index_dtypes(self, into, expected):
        # GH#18580
        # 当使用 DataFrame 的 to_dict(orient='index') 方法时，用于处理整数和浮点数列时，只有整数列被转换为浮点数的问题

        df = DataFrame({"int_col": [1, 2, 3], "float_col": [1.0, 2.0, 3.0]})

        result = df.to_dict(orient="index", into=into)  # 将 DataFrame 转换为字典，按照索引方向，使用指定的类型 into
        cols = ["int_col", "float_col"]
        result = DataFrame.from_dict(result, orient="index")[cols]  # 从字典创建 DataFrame，按照索引方向，并选择指定列
        expected = DataFrame.from_dict(expected, orient="index")[cols]  # 从预期的字典创建 DataFrame，按照索引方向，并选择指定列
        tm.assert_frame_equal(result, expected)  # 断言结果 DataFrame 与预期 DataFrame 相等

    def test_to_dict_numeric_names(self):
        # GH#24940
        # 当 DataFrame 的列名为数值时，to_dict("records") 方法的行为

        df = DataFrame({str(i): [i] for i in range(5)})  # 创建列名为数值的 DataFrame
        result = set(df.to_dict("records")[0].keys())  # 获取转换为字典记录的第一个条目的键集合
        expected = set(df.columns)  # 获取 DataFrame 的列名集合作为预期结果
        assert result == expected  # 断言结果与预期相同

    def test_to_dict_wide(self):
        # GH#24939
        # 当 DataFrame 有大量列时，to_dict("records") 方法的行为

        df = DataFrame({(f"A_{i:d}"): [i] for i in range(256)})  # 创建具有大量列的 DataFrame
        result = df.to_dict("records")[0]  # 获取转换为字典记录的第一个条目
        expected = {f"A_{i:d}": i for i in range(256)}  # 生成预期的字典记录
        assert result == expected  # 断言结果与预期相同

    @pytest.mark.parametrize(
        "data,dtype",
        (  # 参数化测试用例，测试不同的参数组合
            [True, True, False], bool,  # 布尔类型的情况
            [
                [datetime(2018, 1, 1), datetime(2019, 2, 2), datetime(2020, 3, 3)], Timestamp,  # 日期时间和 Timestamp 类型的情况
            ],
            [[1.0, 2.0, 3.0], float],  # 浮点数类型的情况
            [[1, 2, 3], int],  # 整数类型的情况
            [["X", "Y", "Z"], str],  # 字符串类型的情况
        ),
    )
    def test_to_dict_orient_dtype(self, data, dtype):
        # GH22620 & GH21256
        # 当使用 DataFrame 的 to_dict(orient='records') 方法，并指定数据类型时的行为

        df = DataFrame({"a": data})  # 创建包含特定数据的 DataFrame
        d = df.to_dict(orient="records")  # 将 DataFrame 转换为字典记录
        assert all(type(record["a"]) is dtype for record in d)  # 断言所有记录中的 'a' 列的类型与指定的 dtype 相同
    @pytest.mark.parametrize(
        "data,expected_dtype",
        (
            [np.uint64(2), int],                      # 定义参数化测试的数据和预期数据类型
            [np.int64(-9), int],
            [np.float64(1.1), float],
            [np.bool_(True), bool],
            [np.datetime64("2005-02-25"), Timestamp],
        ),
    )
    def test_to_dict_scalar_constructor_orient_dtype(self, data, expected_dtype):
        # GH22620 & GH21256
        # 测试方法，验证 DataFrame 中单个元素的数据类型转换是否正确

        df = DataFrame({"a": data}, index=[0])       # 创建 DataFrame，包含单个数据元素
        d = df.to_dict(orient="records")             # 将 DataFrame 转换为字典列表
        result = type(d[0]["a"])                     # 获取转换后字典中元素 "a" 的数据类型
        assert result is expected_dtype               # 断言数据类型是否符合预期

    def test_to_dict_mixed_numeric_frame(self):
        # GH 12859
        # 测试方法，验证 DataFrame 包含混合类型数值的情况下转换为字典列表的正确性

        df = DataFrame({"a": [1.0], "b": [9.0]})     # 创建包含混合数值类型的 DataFrame
        result = df.reset_index().to_dict("records") # 将重置索引后的 DataFrame 转换为字典列表
        expected = [{"index": 0, "a": 1.0, "b": 9.0}] # 预期的转换结果
        assert result == expected                     # 断言转换结果是否符合预期

    @pytest.mark.parametrize(
        "index",
        [
            None,
            Index(["aa", "bb"]),                      # 定义参数化测试的索引类型
            Index(["aa", "bb"], name="cc"),
            MultiIndex.from_tuples([("a", "b"), ("a", "c")]),
            MultiIndex.from_tuples([("a", "b"), ("a", "c")], names=["n1", "n2"]),
        ],
    )
    @pytest.mark.parametrize(
        "columns",
        [
            ["x", "y"],                              # 定义参数化测试的列类型
            Index(["x", "y"]),
            Index(["x", "y"], name="z"),
            MultiIndex.from_tuples([("x", 1), ("y", 2)]),
            MultiIndex.from_tuples([("x", 1), ("y", 2)], names=["z1", "z2"]),
        ],
    )
    def test_to_dict_orient_tight(self, index, columns):
        # 测试方法，验证 DataFrame 通过不同的索引和列类型转换为紧凑字典的正确性

        df = DataFrame.from_records(
            [[1, 3], [2, 4]],
            columns=columns,                        # 使用参数化的列类型创建 DataFrame
            index=index,                            # 使用参数化的索引类型设置索引
        )
        roundtrip = DataFrame.from_dict(df.to_dict(orient="tight"), orient="tight")  # 将 DataFrame 转换为紧凑字典，再转换回 DataFrame

        tm.assert_frame_equal(df, roundtrip)        # 断言转换回的 DataFrame 是否与原始 DataFrame 相等

    @pytest.mark.parametrize(
        "orient",
        ["dict", "list", "split", "records", "index", "tight"],  # 定义参数化测试的 orient 类型
    )
    @pytest.mark.parametrize(
        "data,expected_types",
        (
            (
                {
                    "a": [np.int64(1), 1, np.int64(3)],  # 定义包含不同类型数据的字典，包括 np.int64 对象、整数和 np.int64 对象
                    "b": [np.float64(1.0), 2.0, np.float64(3.0)],  # 同上，但是包括 np.float64 对象和浮点数
                    "c": [np.float64(1.0), 2, np.int64(3)],  # 同上，包含 np.float64 对象、整数和 np.int64 对象
                    "d": [np.float64(1.0), "a", np.int64(3)],  # 同上，包含 np.float64 对象、字符串和 np.int64 对象
                    "e": [np.float64(1.0), ["a"], np.int64(3)],  # 同上，包含 np.float64 对象、列表和 np.int64 对象
                    "f": [np.float64(1.0), ("a",), np.int64(3)],  # 同上，包含 np.float64 对象、元组和 np.int64 对象
                },
                {
                    "a": [int, int, int],  # 期望结果的类型列表，包含整数类型
                    "b": [float, float, float],  # 期望结果的类型列表，包含浮点数类型
                    "c": [float, float, float],  # 期望结果的类型列表，包含浮点数类型
                    "d": [float, str, int],  # 期望结果的类型列表，包含浮点数类型、字符串类型和整数类型
                    "e": [float, list, int],  # 期望结果的类型列表，包含浮点数类型、列表类型和整数类型
                    "f": [float, tuple, int],  # 期望结果的类型列表，包含浮点数类型、元组类型和整数类型
                },
            ),
            (
                {
                    "a": [1, 2, 3],  # 定义只包含整数的字典
                    "b": [1.1, 2.2, 3.3],  # 定义只包含浮点数的字典
                },
                {
                    "a": [int, int, int],  # 期望结果的类型列表，包含整数类型
                    "b": [float, float, float],  # 期望结果的类型列表，包含浮点数类型
                },
            ),
            (  # 确保我们有一个包含全部对象类型列的数据帧
                {
                    "a": [1, "hello", 3],  # 定义包含混合类型数据的字典，包括整数和字符串
                    "b": [1.1, "world", 3.3],  # 定义包含混合类型数据的字典，包括浮点数和字符串
                },
                {
                    "a": [int, str, int],  # 期望结果的类型列表，包含整数类型和字符串类型
                    "b": [float, str, float],  # 期望结果的类型列表，包含浮点数类型和字符串类型
                },
            ),
        ),
    )
    def test_to_dict_returns_native_types(self, orient, data, expected_types):
        # GH 46751
        # 检查确保对于所有方向类型都返回本机类型
        df = DataFrame(data)  # 根据给定数据创建数据帧
        result = df.to_dict(orient)  # 调用数据帧的 to_dict 方法，按照给定方向转换为字典形式

        if orient == "dict":
            assertion_iterator = (
                (i, key, value)
                for key, index_value_map in result.items()
                for i, value in index_value_map.items()
            )
        elif orient == "list":
            assertion_iterator = (
                (i, key, value)
                for key, values in result.items()
                for i, value in enumerate(values)
            )
        elif orient in {"split", "tight"}:
            assertion_iterator = (
                (i, key, result["data"][i][j])
                for i in result["index"]
                for j, key in enumerate(result["columns"])
            )
        elif orient == "records":
            assertion_iterator = (
                (i, key, value)
                for i, record in enumerate(result)
                for key, value in record.items()
            )
        elif orient == "index":
            assertion_iterator = (
                (i, key, value)
                for i, record in result.items()
                for key, value in record.items()
            )

        for i, key, value in assertion_iterator:
            assert value == data[key][i]  # 断言确保返回值与原始数据相匹配
            assert type(value) is expected_types[key][i]  # 断言确保返回值类型与预期类型相符
    # 使用 pytest 的参数化装饰器，对 test_to_dict_index_false_error 方法进行多次参数化测试
    @pytest.mark.parametrize("orient", ["dict", "list", "series", "records", "index"])
    def test_to_dict_index_false_error(self, orient):
        # GH#46398
        # 创建一个 DataFrame 对象，包含两列 "col1" 和 "col2"，以及自定义索引 ["row1", "row2"]
        df = DataFrame({"col1": [1, 2], "col2": [3, 4]}, index=["row1", "row2"])
        # 准备异常消息，验证 'index=False' 仅在 'orient' 是 'split' 或 'tight' 时有效
        msg = "'index=False' is only valid when 'orient' is 'split' or 'tight'"
        # 使用 pytest 的断言来验证在特定条件下是否会引发 ValueError 异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            df.to_dict(orient=orient, index=False)

    # 使用 pytest 的参数化装饰器，对 test_to_dict_index_false 方法进行多次参数化测试
    @pytest.mark.parametrize(
        "orient, expected",
        [
            # 当 orient 为 "split" 时，预期结果是列名为 "col1" 和 "col2"，数据为 [[1, 3], [2, 4]]
            ("split", {"columns": ["col1", "col2"], "data": [[1, 3], [2, 4]]}),
            # 当 orient 为 "tight" 时，预期结果包括列名 "col1" 和 "col2"，数据为 [[1, 3], [2, 4]]，列名和索引名为 None
            (
                "tight",
                {
                    "columns": ["col1", "col2"],
                    "data": [[1, 3], [2, 4]],
                    "column_names": [None],
                },
            ),
        ],
    )
    def test_to_dict_index_false(self, orient, expected):
        # GH#46398
        # 创建一个 DataFrame 对象，包含两列 "col1" 和 "col2"，以及自定义索引 ["row1", "row2"]
        df = DataFrame({"col1": [1, 2], "col2": [3, 4]}, index=["row1", "row2"])
        # 使用 DataFrame 的 to_dict 方法生成结果字典，验证生成的结果与预期的 expected 字典是否相等
        result = df.to_dict(orient=orient, index=False)
        tm.assert_dict_equal(result, expected)

    # 使用 pytest 的参数化装饰器，对 test_to_dict_na_to_none 方法进行多次参数化测试
    @pytest.mark.parametrize(
        "orient, expected",
        [
            # 当 orient 为 "dict" 时，预期结果是 {"a": {0: 1, 1: None}}
            ("dict", {"a": {0: 1, 1: None}}),
            # 当 orient 为 "list" 时，预期结果是 {"a": [1, None]}
            ("list", {"a": [1, None]}),
            # 当 orient 为 "split" 时，预期结果是 {"index": [0, 1], "columns": ["a"], "data": [[1], [None]]}
            ("split", {"index": [0, 1], "columns": ["a"], "data": [[1], [None]]}),
            # 当 orient 为 "tight" 时，预期结果包括 {"index": [0, 1], "columns": ["a"], "data": [[1], [None]], "index_names": [None], "column_names": [None]}
            (
                "tight",
                {
                    "index": [0, 1],
                    "columns": ["a"],
                    "data": [[1], [None]],
                    "index_names": [None],
                    "column_names": [None],
                },
            ),
            # 当 orient 为 "records" 时，预期结果是 [{"a": 1}, {"a": None}]
            ("records", [{"a": 1}, {"a": None}]),
            # 当 orient 为 "index" 时，预期结果是 {0: {"a": 1}, 1: {"a": None}}
            ("index", {0: {"a": 1}, 1: {"a": None}}),
        ],
    )
    def test_to_dict_na_to_none(self, orient, expected):
        # GH#50795
        # 创建一个 DataFrame 对象，包含列 "a"，其中有一个值为 NA
        df = DataFrame({"a": [1, NA]}, dtype="Int64")
        # 使用 DataFrame 的 to_dict 方法生成结果字典，验证生成的结果与预期的 expected 字典是否相等
        result = df.to_dict(orient=orient)
        assert result == expected

    # 测试 DataFrame 的 to_dict 方法，验证对于包含 Int64 类型的 Series 对象的处理
    def test_to_dict_masked_native_python(self):
        # GH#34665
        # 创建一个 DataFrame 对象，包含列 "a"，其中有一个值为 NA，并指定 Series 的数据类型为 "Int64"
        df = DataFrame({"a": Series([1, 2], dtype="Int64"), "B": 1})
        # 使用 DataFrame 的 to_dict 方法生成结果字典，验证生成的结果中 "a" 列的数据类型是否为 int
        result = df.to_dict(orient="records")
        assert isinstance(result[0]["a"], int)

        # 创建另一个 DataFrame 对象，包含列 "a"，其中有一个值为 NA，并指定 Series 的数据类型为 "Int64"
        df = DataFrame({"a": Series([1, NA], dtype="Int64"), "B": 1})
        # 使用 DataFrame 的 to_dict 方法生成结果字典，验证生成的结果中 "a" 列的数据类型是否为 int
        result = df.to_dict(orient="records")
        assert isinstance(result[0]["a"], int)

    # 测试 DataFrame 的 to_dict 方法，验证在存在重复列时的处理
    def test_to_dict_tight_no_warning_with_duplicate_column(self):
        # GH#58281
        # 创建一个 DataFrame 对象，包含重复的列名 "A"，数据为 [[1, 2], [3, 4], [5, 6]]
        df = DataFrame([[1, 2], [3, 4], [5, 6]], columns=["A", "A"])
        # 使用 tm.assert_produces_warning 上下文管理器，验证在执行 to_dict 方法时不会产生警告
        with tm.assert_produces_warning(None):
            # 使用 DataFrame 的 to_dict 方法生成结果字典，验证生成的结果与预期的 expected 字典是否相等
            result = df.to_dict(orient="tight")
        # 预期的结果包括索引、列名、数据以及它们的名称均为 None
        expected = {
            "index": [0, 1, 2],
            "columns": ["A", "A"],
            "data": [[1, 2], [3, 4], [5, 6]],
            "index_names": [None],
            "column_names": [None],
        }
        assert result == expected
# 使用 pytest 的参数化装饰器，定义多组参数化测试数据
@pytest.mark.parametrize(
    "val", [Timestamp(2020, 1, 1), Timedelta(1), Period("2020"), Interval(1, 2)]
)
# 定义名为 test_to_dict_list_pd_scalars 的测试函数，用于测试将 Pandas 标量转换为字典列表的功能
def test_to_dict_list_pd_scalars(val):
    # GH 54824: 关联 GitHub 问题编号，用于说明这个测试的背景或问题来源
    # 创建一个包含单列 'a' 的 DataFrame，其中的值为参数化测试数据 val
    df = DataFrame({"a": [val]})
    # 调用 DataFrame 的 to_dict 方法，将 DataFrame 转换为字典列表，orient 参数指定转换方式为 "list"
    result = df.to_dict(orient="list")
    # 预期的结果字典，包含 'a' 列的值列表 [val]
    expected = {"a": [val]}
    # 使用断言检查实际结果和预期结果是否相等
    assert result == expected
```