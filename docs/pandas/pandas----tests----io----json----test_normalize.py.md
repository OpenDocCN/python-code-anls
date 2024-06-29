# `D:\src\scipysrc\pandas\pandas\tests\io\json\test_normalize.py`

```
import json  # 导入 JSON 模块

import numpy as np  # 导入 NumPy 模块
import pytest  # 导入 Pytest 模块

from pandas import (  # 从 Pandas 中导入以下模块：
    DataFrame,  # 数据框
    Index,  # 索引
    Series,  # 序列
    json_normalize,  # JSON 标准化函数
)
import pandas._testing as tm  # 导入 Pandas 测试工具模块

from pandas.io.json._normalize import nested_to_record  # 导入 Pandas 中 JSON 标准化模块中的特定函数


@pytest.fixture
def deep_nested():
    # 深度嵌套的数据
    return [
        {
            "country": "USA",
            "states": [
                {
                    "name": "California",
                    "cities": [
                        {"name": "San Francisco", "pop": 12345},
                        {"name": "Los Angeles", "pop": 12346},
                    ],
                },
                {
                    "name": "Ohio",
                    "cities": [
                        {"name": "Columbus", "pop": 1234},
                        {"name": "Cleveland", "pop": 1236},
                    ],
                },
            ],
        },
        {
            "country": "Germany",
            "states": [
                {"name": "Bayern", "cities": [{"name": "Munich", "pop": 12347}]},
                {
                    "name": "Nordrhein-Westfalen",
                    "cities": [
                        {"name": "Duesseldorf", "pop": 1238},
                        {"name": "Koeln", "pop": 1239},
                    ],
                },
            ],
        },
    ]


@pytest.fixture
def state_data():
    # 状态数据
    return [
        {
            "counties": [
                {"name": "Dade", "population": 12345},
                {"name": "Broward", "population": 40000},
                {"name": "Palm Beach", "population": 60000},
            ],
            "info": {"governor": "Rick Scott"},
            "shortname": "FL",
            "state": "Florida",
        },
        {
            "counties": [
                {"name": "Summit", "population": 1234},
                {"name": "Cuyahoga", "population": 1337},
            ],
            "info": {"governor": "John Kasich"},
            "shortname": "OH",
            "state": "Ohio",
        },
    ]


@pytest.fixture
def missing_metadata():
    # 缺失的元数据
    return [
        {
            "name": "Alice",
            "addresses": [
                {
                    "number": 9562,
                    "street": "Morris St.",
                    "city": "Massillon",
                    "state": "OH",
                    "zip": 44646,
                }
            ],
            "previous_residences": {"cities": [{"city_name": "Foo York City"}]},
        },
        {
            "addresses": [
                {
                    "number": 8449,
                    "street": "Spring St.",
                    "city": "Elizabethton",
                    "state": "TN",
                    "zip": 37643,
                }
            ],
            "previous_residences": {"cities": [{"city_name": "Barmingham"}]},
        },
    ]


class TestJSONNormalize:
    # 测试简单的记录
    def test_simple_records(self):
        # 创建包含多个字典的列表
        recs = [
            {"a": 1, "b": 2, "c": 3},
            {"a": 4, "b": 5, "c": 6},
            {"a": 7, "b": 8, "c": 9},
            {"a": 10, "b": 11, "c": 12},
        ]

        # 对记录进行规范化
        result = json_normalize(recs)
        # 创建预期的 DataFrame
        expected = DataFrame(recs)

        # 断言结果与预期相等
        tm.assert_frame_equal(result, expected)

    # 测试简单的规范化
    def test_simple_normalize(self, state_data):
        # 对 state_data 中的第一个元素的 "counties" 进行规范化
        result = json_normalize(state_data[0], "counties")
        expected = DataFrame(state_data[0]["counties"])
        tm.assert_frame_equal(result, expected)

        # 对 state_data 中的所有元素的 "counties" 进行规范化
        result = json_normalize(state_data, "counties")

        # 创建预期的 DataFrame
        expected = []
        for rec in state_data:
            expected.extend(rec["counties"])
        expected = DataFrame(expected)

        # 断言结果与预期相等
        tm.assert_frame_equal(result, expected)

        # 对 state_data 中的所有元素的 "counties" 进行规范化，并添加 "state" 作为元数据
        result = json_normalize(state_data, "counties", meta="state")
        expected["state"] = np.array(["Florida", "Ohio"]).repeat([3, 2])

        # 断言结果与预期相等
        tm.assert_frame_equal(result, expected)

    # 测试字段列表类型的规范化
    def test_fields_list_type_normalize(self):
        # 创建包含特定结构的列表
        parse_metadata_fields_list_type = [
            {"values": [1, 2, 3], "metadata": {"listdata": [1, 2]}}
        ]
        # 对列表进行规范化
        result = json_normalize(
            parse_metadata_fields_list_type,
            record_path=["values"],
            meta=[["metadata", "listdata"]],
        )
        expected = DataFrame(
            {0: [1, 2, 3], "metadata.listdata": [[1, 2], [1, 2], [1, 2]]}
        )
        # 断言结果与预期相等
        tm.assert_frame_equal(result, expected)

    # 测试空数组
    def test_empty_array(self):
        # 对空数组进行规范化
        result = json_normalize([])
        expected = DataFrame()
        # 断言结果与预期相等
        tm.assert_frame_equal(result, expected)

    # 测试接受的输入
    @pytest.mark.parametrize(
        "data, record_path, exception_type",
        [
            ([{"a": 0}, {"a": 1}], None, None),
            ({"a": [{"a": 0}, {"a": 1}]}, "a", None),
            ('{"a": [{"a": 0}, {"a": 1}]}', None, NotImplementedError),
            (None, None, NotImplementedError),
        ],
    )
    def test_accepted_input(self, data, record_path, exception_type):
        if exception_type is not None:
            # 如果有异常类型，则断言引发该异常
            with pytest.raises(exception_type, match=""):
                json_normalize(data, record_path=record_path)
        else:
            # 否则对数据进行规范化
            result = json_normalize(data, record_path=record_path)
            expected = DataFrame([0, 1], columns=["a"])
            # 断言结果与预期相等
            tm.assert_frame_equal(result, expected)
    # 测试简单的 JSON 数据规范化操作，使用默认分隔符 "."
    def test_simple_normalize_with_separator(self, deep_nested):
        # GH 14883
        # 对 JSON 数据进行规范化，并验证结果是否与预期一致
        result = json_normalize({"A": {"A": 1, "B": 2}})
        expected = DataFrame([[1, 2]], columns=["A.A", "A.B"])
        tm.assert_frame_equal(result.reindex_like(expected), expected)

        # 对 JSON 数据进行规范化，使用自定义分隔符 "_"
        result = json_normalize({"A": {"A": 1, "B": 2}}, sep="_")
        expected = DataFrame([[1, 2]], columns=["A_A", "A_B"])
        tm.assert_frame_equal(result.reindex_like(expected), expected)

        # 对 JSON 数据进行规范化，使用 Unicode 分隔符 "\u03c3"
        result = json_normalize({"A": {"A": 1, "B": 2}}, sep="\u03c3")
        expected = DataFrame([[1, 2]], columns=["A\u03c3A", "A\u03c3B"])
        tm.assert_frame_equal(result.reindex_like(expected), expected)

        # 对深度嵌套的 JSON 数据进行规范化，指定路径及元数据，并使用分隔符 "_"
        result = json_normalize(
            deep_nested,
            ["states", "cities"],
            meta=["country", ["states", "name"]],
            sep="_",
        )
        expected = Index(["name", "pop", "country", "states_name"]).sort_values()
        # 验证结果的列名是否与预期相符
        assert result.columns.sort_values().equals(expected)

    # 测试使用多字符分隔符的 JSON 数据规范化
    def test_normalize_with_multichar_separator(self):
        # GH #43831
        # 准备包含多层结构的 JSON 数据，并使用 "__" 作为分隔符进行规范化
        data = {"a": [1, 2], "b": {"b_1": 2, "b_2": (3, 4)}}
        result = json_normalize(data, sep="__")
        expected = DataFrame([[[1, 2], 2, (3, 4)]], columns=["a", "b__b_1", "b__b_2"])
        tm.assert_frame_equal(result, expected)

    # 测试使用记录前缀的 JSON 数据规范化
    def test_value_array_record_prefix(self):
        # GH 21536
        # 对包含数组的 JSON 数据进行规范化，并指定记录前缀为 "Prefix."
        result = json_normalize({"A": [1, 2]}, "A", record_prefix="Prefix.")
        expected = DataFrame([[1], [2]], columns=["Prefix.0"])
        tm.assert_frame_equal(result, expected)

    # 测试使用记录路径的 JSON 数据规范化
    def test_nested_object_record_path(self):
        # GH 22706
        # 准备包含嵌套对象及记录路径的 JSON 数据
        data = {
            "state": "Florida",
            "info": {
                "governor": "Rick Scott",
                "counties": [
                    {"name": "Dade", "population": 12345},
                    {"name": "Broward", "population": 40000},
                    {"name": "Palm Beach", "population": 60000},
                ],
            },
        }
        # 对记录路径 ["info", "counties"] 下的 JSON 数据进行规范化
        result = json_normalize(data, record_path=["info", "counties"])
        expected = DataFrame(
            [["Dade", 12345], ["Broward", 40000], ["Palm Beach", 60000]],
            columns=["name", "population"],
        )
        tm.assert_frame_equal(result, expected)
    # 定义一个测试方法，测试更深层次的嵌套数据
    def test_more_deeply_nested(self, deep_nested):
        # 使用 json_normalize 函数将深度嵌套的数据规范化，展开到指定的数据结构
        result = json_normalize(
            deep_nested, ["states", "cities"], meta=["country", ["states", "name"]]
        )
        # 定义预期的展开数据结构，包括国家、州名、城市名和人口数据
        ex_data = {
            "country": ["USA"] * 4 + ["Germany"] * 3,
            "states.name": [
                "California",
                "California",
                "Ohio",
                "Ohio",
                "Bayern",
                "Nordrhein-Westfalen",
                "Nordrhein-Westfalen",
            ],
            "name": [
                "San Francisco",
                "Los Angeles",
                "Columbus",
                "Cleveland",
                "Munich",
                "Duesseldorf",
                "Koeln",
            ],
            "pop": [12345, 12346, 1234, 1236, 12347, 1238, 1239],
        }
        # 根据预期数据创建 DataFrame 对象
        expected = DataFrame(ex_data, columns=result.columns)
        # 使用测试工具函数检查结果是否与预期一致
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，测试浅层次的嵌套数据
    def test_shallow_nested(self):
        # 定义示例数据，包括州名、简称、州长信息及其包含的县级数据
        data = [
            {
                "state": "Florida",
                "shortname": "FL",
                "info": {"governor": "Rick Scott"},
                "counties": [
                    {"name": "Dade", "population": 12345},
                    {"name": "Broward", "population": 40000},
                    {"name": "Palm Beach", "population": 60000},
                ],
            },
            {
                "state": "Ohio",
                "shortname": "OH",
                "info": {"governor": "John Kasich"},
                "counties": [
                    {"name": "Summit", "population": 1234},
                    {"name": "Cuyahoga", "population": 1337},
                ],
            },
        ]
        # 使用 json_normalize 函数将嵌套的县级数据展开，同时保留州名、简称和州长信息
        result = json_normalize(
            data, "counties", ["state", "shortname", ["info", "governor"]]
        )
        # 定义预期的展开数据结构，包括县名、州名、简称、州长和人口数据
        ex_data = {
            "name": ["Dade", "Broward", "Palm Beach", "Summit", "Cuyahoga"],
            "state": ["Florida"] * 3 + ["Ohio"] * 2,
            "shortname": ["FL", "FL", "FL", "OH", "OH"],
            "info.governor": ["Rick Scott"] * 3 + ["John Kasich"] * 2,
            "population": [12345, 40000, 60000, 1234, 1337],
        }
        # 根据预期数据创建 DataFrame 对象
        expected = DataFrame(ex_data, columns=result.columns)
        # 使用测试工具函数检查结果是否与预期一致
        tm.assert_frame_equal(result, expected)
    # 测试函数，用于处理嵌套数据和元数据路径的情况
    def test_nested_meta_path_with_nested_record_path(self, state_data):
        # GH 27220
        # 使用 json_normalize 函数处理数据，将 "counties" 作为记录路径，["state", "shortname", ["info", "governor"]] 作为元数据
        result = json_normalize(
            data=state_data,
            record_path=["counties"],
            meta=["state", "shortname", ["info", "governor"]],
            errors="ignore",
        )

        # 预期的数据结构
        ex_data = {
            "name": ["Dade", "Broward", "Palm Beach", "Summit", "Cuyahoga"],
            "population": [12345, 40000, 60000, 1234, 1337],
            "state": ["Florida"] * 3 + ["Ohio"] * 2,
            "shortname": ["FL"] * 3 + ["OH"] * 2,
            "info.governor": ["Rick Scott"] * 3 + ["John Kasich"] * 2,
        }

        expected = DataFrame(ex_data)
        # 断言结果与预期相同
        tm.assert_frame_equal(result, expected)

    # 测试函数，验证元数据字段名称冲突的情况
    def test_meta_name_conflict(self):
        data = [
            {
                "foo": "hello",
                "bar": "there",
                "data": [
                    {"foo": "something", "bar": "else"},
                    {"foo": "something2", "bar": "else2"},
                ],
            }
        ]

        # 预期抛出值错误异常，指定的错误消息
        msg = r"Conflicting metadata name (foo|bar), need distinguishing prefix"
        with pytest.raises(ValueError, match=msg):
            json_normalize(data, "data", meta=["foo", "bar"])

        # 使用 meta_prefix 参数解决元数据字段名称冲突问题
        result = json_normalize(data, "data", meta=["foo", "bar"], meta_prefix="meta")

        # 断言结果中包含特定的字段
        for val in ["metafoo", "metabar", "foo", "bar"]:
            assert val in result

    # 测试函数，验证元数据参数不被修改的情况
    def test_meta_parameter_not_modified(self):
        # GH 18610
        data = [
            {
                "foo": "hello",
                "bar": "there",
                "data": [
                    {"foo": "something", "bar": "else"},
                    {"foo": "something2", "bar": "else2"},
                ],
            }
        ]

        # 定义列名常量
        COLUMNS = ["foo", "bar"]
        # 使用 json_normalize 函数处理数据，并指定 meta_prefix 参数
        result = json_normalize(data, "data", meta=COLUMNS, meta_prefix="meta")

        # 断言 COLUMNS 列名未被修改
        assert COLUMNS == ["foo", "bar"]
        # 断言结果中包含特定的字段
        for val in ["metafoo", "metabar", "foo", "bar"]:
            assert val in result

    # 测试函数，验证记录前缀的使用情况
    def test_record_prefix(self, state_data):
        # 使用 json_normalize 函数处理数据，将 "counties" 作为记录路径
        result = json_normalize(state_data[0], "counties")
        # 预期的 DataFrame 结构
        expected = DataFrame(state_data[0]["counties"])
        # 断言结果与预期相同
        tm.assert_frame_equal(result, expected)

        # 使用 json_normalize 函数处理数据，将 "counties" 作为记录路径，"state" 作为元数据，并指定 record_prefix 参数
        result = json_normalize(
            state_data, "counties", meta="state", record_prefix="county_"
        )

        # 构建预期的 DataFrame 结构
        expected = []
        for rec in state_data:
            expected.extend(rec["counties"])
        expected = DataFrame(expected)
        expected = expected.rename(columns=lambda x: "county_" + x)
        expected["state"] = np.array(["Florida", "Ohio"]).repeat([3, 2])

        # 断言结果与预期相同
        tm.assert_frame_equal(result, expected)
    def test_non_ascii_key(self):
        # 定义一个包含非ASCII键的JSON字符串，使用UTF-8解码
        testjson = (
            b'[{"\xc3\x9cnic\xc3\xb8de":0,"sub":{"A":1, "B":2}},'
            b'{"\xc3\x9cnic\xc3\xb8de":1,"sub":{"A":3, "B":4}}]'
        ).decode("utf8")

        # 预期的测试数据，包含一个非ASCII键的字典，键名使用UTF-8解码
        testdata = {
            b"\xc3\x9cnic\xc3\xb8de".decode("utf8"): [0, 1],
            "sub.A": [1, 3],
            "sub.B": [2, 4],
        }
        # 生成预期的DataFrame对象
        expected = DataFrame(testdata)

        # 使用json_normalize函数对JSON字符串进行规范化
        result = json_normalize(json.loads(testjson))
        # 断言结果与预期的DataFrame对象相等
        tm.assert_frame_equal(result, expected)

    def test_missing_field(self):
        # GH20030:
        # 定义一个包含缺失字段的JSON数据列表
        author_missing_data = [
            {"info": None},
            {
                "info": {"created_at": "11/08/1993", "last_updated": "26/05/2012"},
                "author_name": {"first": "Jane", "last_name": "Doe"},
            },
        ]
        # 对JSON数据进行规范化
        result = json_normalize(author_missing_data)
        
        # 期望的规范化后的数据，包含缺失字段的DataFrame对象
        ex_data = [
            {
                "info": np.nan,
                "info.created_at": np.nan,
                "info.last_updated": np.nan,
                "author_name.first": np.nan,
                "author_name.last_name": np.nan,
            },
            {
                "info": None,
                "info.created_at": "11/08/1993",
                "info.last_updated": "26/05/2012",
                "author_name.first": "Jane",
                "author_name.last_name": "Doe",
            },
        ]
        # 生成预期的DataFrame对象
        expected = DataFrame(ex_data)
        # 断言结果与预期的DataFrame对象相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "max_level,expected",
        [
            (
                0,
                [
                    {
                        "TextField": "Some text",
                        "UserField": {"Id": "ID001", "Name": "Name001"},
                        "CreatedBy": {"Name": "User001"},
                        "Image": {"a": "b"},
                    },
                    {
                        "TextField": "Some text",
                        "UserField": {"Id": "ID001", "Name": "Name001"},
                        "CreatedBy": {"Name": "User001"},
                        "Image": {"a": "b"},
                    },
                ],
            ),
            (
                1,
                [
                    {
                        "TextField": "Some text",
                        "UserField.Id": "ID001",
                        "UserField.Name": "Name001",
                        "CreatedBy": {"Name": "User001"},
                        "Image": {"a": "b"},
                    },
                    {
                        "TextField": "Some text",
                        "UserField.Id": "ID001",
                        "UserField.Name": "Name001",
                        "CreatedBy": {"Name": "User001"},
                        "Image": {"a": "b"},
                    },
                ],
            ),
        ],
    )
    def test_max_level_with_records_path(self, max_level, expected):
        # GH23843: Enhanced JSON normalize
        # 定义测试输入，包含复杂的嵌套结构
        test_input = [
            {
                "CreatedBy": {"Name": "User001"},
                "Lookup": [
                    {
                        "TextField": "Some text",
                        "UserField": {"Id": "ID001", "Name": "Name001"},
                    },
                    {
                        "TextField": "Some text",
                        "UserField": {"Id": "ID001", "Name": "Name001"},
                    },
                ],
                "Image": {"a": "b"},
                "tags": [
                    {"foo": "something", "bar": "else"},
                    {"foo": "something2", "bar": "else2"},
                ],
            }
        ]

        # 调用 json_normalize 函数，处理复杂的嵌套结构数据
        result = json_normalize(
            test_input,
            record_path=["Lookup"],  # 指定记录路径为 "Lookup"
            meta=[["CreatedBy"], ["Image"]],  # 设置元数据为 "CreatedBy" 和 "Image"
            max_level=max_level,  # 设置最大处理层级
        )
        
        # 根据预期结果创建 DataFrame 对象
        expected_df = DataFrame(data=expected, columns=result.columns.values)
        # 使用 pytest 的 assert_equal 方法比较预期结果和实际结果
        tm.assert_equal(expected_df, result)

    def test_nested_flattening_consistent(self):
        # see gh-21537
        # 调用 json_normalize 函数，处理简单的嵌套结构数据
        df1 = json_normalize([{"A": {"B": 1}}])
        df2 = json_normalize({"dummy": [{"A": {"B": 1}}]}, "dummy")

        # 比较两个 DataFrame 对象是否相同
        # 期望 df1 和 df2 结果一致
        tm.assert_frame_equal(df1, df2)

    def test_nonetype_record_path(self, nulls_fixture):
        # see gh-30148
        # 调用 json_normalize 函数，处理包含空值的数据
        # 不应该引发 TypeError 异常
        result = json_normalize(
            [
                {"state": "Texas", "info": nulls_fixture},  # 包含空值的情况
                {"state": "Florida", "info": [{"i": 2}]},  # 包含非空值的情况
            ],
            record_path=["info"],  # 指定记录路径为 "info"
        )
        # 创建预期结果的 DataFrame 对象
        expected = DataFrame({"i": 2}, index=[0])
        # 使用 pytest 的 assert_equal 方法比较预期结果和实际结果
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("value", ["false", "true", "{}", "1", '"text"'])
    def test_non_list_record_path_errors(self, value):
        # see gh-30148, GH 26284
        # 解析传入的值为 JSON 对象
        parsed_value = json.loads(value)
        test_input = {"state": "Texas", "info": parsed_value}
        test_path = "info"
        # 构建错误信息，指明路径必须包含列表或空值
        msg = (
            f"Path must contain list or null, "
            f"but got {type(parsed_value).__name__} at 'info'"
        )
        # 使用 pytest 的 raises 方法检查是否引发了预期的 TypeError 异常，并包含指定错误信息
        with pytest.raises(TypeError, match=msg):
            json_normalize([test_input], record_path=[test_path])

    def test_meta_non_iterable(self):
        # GH 31507
        # 定义 JSON 格式的数据字符串
        data = """[{"id": 99, "data": [{"one": 1, "two": 2}]}]"""

        # 调用 json_normalize 函数，处理数据中的 "data" 字段
        result = json_normalize(json.loads(data), record_path=["data"], meta=["id"])
        # 创建预期结果的 DataFrame 对象
        expected = DataFrame(
            {"one": [1], "two": [2], "id": np.array([99], dtype=object)}
        )
        # 使用 pytest 的 assert_frame_equal 方法比较预期结果和实际结果
        tm.assert_frame_equal(result, expected)
    # 定义一个测试生成器函数，用于测试修复的生成器输入问题
    def test_generator(self, state_data):
        # GH35923 修复 pd.json_normalize 在生成器输入中不跳过第一个元素的问题

        # 定义一个生成器函数，从 state_data 的第一个元素中的 "counties" 键中生成数据
        def generator_data():
            yield from state_data[0]["counties"]

        # 使用 json_normalize 处理生成器生成的数据，并保存结果
        result = json_normalize(generator_data())
        
        # 创建期望的 DataFrame，与 state_data 的第一个元素中的 "counties" 相对应
        expected = DataFrame(state_data[0]["counties"])

        # 使用测试框架检查结果和期望值是否相等
        tm.assert_frame_equal(result, expected)

    # 测试以带有前导下划线的顶级列名
    def test_top_column_with_leading_underscore(self):
        # 49861

        # 定义一个包含下划线开头的列名的数据字典
        data = {"_id": {"a1": 10, "l2": {"l3": 0}}, "gg": 4}

        # 使用 json_normalize 处理数据，指定下划线作为分隔符
        result = json_normalize(data, sep="_")

        # 创建期望的 DataFrame，包含按指定规则展开的列
        expected = DataFrame([[4, 10, 0]], columns=["gg", "_id_a1", "_id_l2_l3"])

        # 使用测试框架检查结果和期望值是否相等
        tm.assert_frame_equal(result, expected)

    # 测试 Series 的索引
    def test_series_index(self, state_data):
        # 创建一个 Index 对象作为索引
        idx = Index([7, 8])

        # 使用 state_data 创建一个 Series 对象，并指定索引
        series = Series(state_data, index=idx)

        # 使用 json_normalize 处理 Series 对象
        result = json_normalize(series)

        # 使用测试框架检查处理后的索引是否与预期的索引相等
        tm.assert_index_equal(result.index, idx)

        # 使用 json_normalize 处理 Series 对象中的 "counties" 列表
        result = json_normalize(series, "counties")

        # 使用测试框架检查处理后的索引是否与预期的索引相等，重复了特定的次数
        tm.assert_index_equal(result.index, idx.repeat([3, 2]))
class TestNestedToRecord:
    # 定义测试类 TestNestedToRecord，用于测试 nested_to_record 函数

    def test_flat_stays_flat(self):
        # 测试函数：验证当输入数据已经是扁平化格式时，nested_to_record 函数的输出与输入一致
        recs = [{"flat1": 1, "flat2": 2}, {"flat3": 3, "flat2": 4}]
        result = nested_to_record(recs)
        expected = recs
        assert result == expected

    def test_one_level_deep_flattens(self):
        # 测试函数：验证当输入数据包含一层深度的字典时，nested_to_record 函数能正确扁平化
        data = {"flat1": 1, "dict1": {"c": 1, "d": 2}}
        
        result = nested_to_record(data)
        expected = {"dict1.c": 1, "dict1.d": 2, "flat1": 1}
        
        assert result == expected

    def test_nested_flattens(self):
        # 测试函数：验证当输入数据包含多层嵌套字典时，nested_to_record 函数能正确扁平化
        data = {
            "flat1": 1,
            "dict1": {"c": 1, "d": 2},
            "nested": {"e": {"c": 1, "d": 2}, "d": 2},
        }
        
        result = nested_to_record(data)
        expected = {
            "dict1.c": 1,
            "dict1.d": 2,
            "flat1": 1,
            "nested.d": 2,
            "nested.e.c": 1,
            "nested.e.d": 2,
        }
        
        assert result == expected

    def test_json_normalize_errors(self, missing_metadata):
        # 测试函数：验证处理缺失元数据时的异常情况处理
        # GH14583:
        # 如果元数据中的键不总是存在，实现了一个新选项设置 errors='ignore'
        
        msg = (
            "Key 'name' not found. To replace missing values of "
            "'name' with np.nan, pass in errors='ignore'"
        )
        with pytest.raises(KeyError, match=msg):
            json_normalize(
                data=missing_metadata,
                record_path="addresses",
                meta="name",
                errors="raise",
            )

    def test_missing_meta(self, missing_metadata):
        # 测试函数：验证处理可空元数据时的情况
        # GH25468
        # 如果元数据可为空且设置 errors='ignore'，则空值应为 numpy.nan
        
        result = json_normalize(
            data=missing_metadata, record_path="addresses", meta="name", errors="ignore"
        )
        ex_data = [
            [9562, "Morris St.", "Massillon", "OH", 44646, "Alice"],
            [8449, "Spring St.", "Elizabethton", "TN", 37643, np.nan],
        ]
        columns = ["number", "street", "city", "state", "zip", "name"]
        expected = DataFrame(ex_data, columns=columns)
        tm.assert_frame_equal(result, expected)
    def test_missing_nested_meta(self):
        # GH44312
        # 如果 errors="ignore" 并且嵌套元数据为空，则应返回 NaN
        data = {"meta": "foo", "nested_meta": None, "value": [{"rec": 1}, {"rec": 2}]}
        # 调用 json_normalize 函数，展开记录路径 "value"，提取元数据 ["meta", ["nested_meta", "leaf"]]，处理错误时忽略
        result = json_normalize(
            data,
            record_path="value",
            meta=["meta", ["nested_meta", "leaf"]],
            errors="ignore",
        )
        # 预期的数据结果
        ex_data = [[1, "foo", np.nan], [2, "foo", np.nan]]
        columns = ["rec", "meta", "nested_meta.leaf"]
        expected = DataFrame(ex_data, columns=columns).astype(
            {"nested_meta.leaf": object}
        )
        # 断言返回结果与期望结果相等
        tm.assert_frame_equal(result, expected)

        # 如果 errors="raise" 并且嵌套元数据为空，则应该抛出异常，并指明缺失的键名
        with pytest.raises(KeyError, match="'leaf' not found"):
            json_normalize(
                data,
                record_path="value",
                meta=["meta", ["nested_meta", "leaf"]],
                errors="raise",
            )

    def test_missing_meta_multilevel_record_path_errors_raise(self, missing_metadata):
        # GH41876
        # 确保 errors='raise' 在传入长度大于一的 record_path 时能够按预期工作
        msg = (
            "Key 'name' not found. To replace missing values of "
            "'name' with np.nan, pass in errors='ignore'"
        )
        # 预期抛出异常，并匹配特定的错误消息
        with pytest.raises(KeyError, match=msg):
            json_normalize(
                data=missing_metadata,
                record_path=["previous_residences", "cities"],
                meta="name",
                errors="raise",
            )

    def test_missing_meta_multilevel_record_path_errors_ignore(self, missing_metadata):
        # GH41876
        # 确保 errors='ignore' 在传入长度大于一的 record_path 时能够按预期工作
        result = json_normalize(
            data=missing_metadata,
            record_path=["previous_residences", "cities"],
            meta="name",
            errors="ignore",
        )
        # 预期的数据结果
        ex_data = [
            ["Foo York City", "Alice"],
            ["Barmingham", np.nan],
        ]
        columns = ["city_name", "name"]
        expected = DataFrame(ex_data, columns=columns)
        # 断言返回结果与期望结果相等
        tm.assert_frame_equal(result, expected)
    def test_donot_drop_nonevalues(self):
        # GH21356
        # 定义测试数据，包含两个字典，每个字典可能嵌套有子字典
        data = [
            {"info": None, "author_name": {"first": "Smith", "last_name": "Appleseed"}},
            {
                "info": {"created_at": "11/08/1993", "last_updated": "26/05/2012"},
                "author_name": {"first": "Jane", "last_name": "Doe"},
            },
        ]
        # 调用 nested_to_record 函数处理测试数据
        result = nested_to_record(data)
        # 定义预期输出结果，每个字典中的键根据层级关系进行了扁平化处理
        expected = [
            {
                "info": None,
                "author_name.first": "Smith",
                "author_name.last_name": "Appleseed",
            },
            {
                "author_name.first": "Jane",
                "author_name.last_name": "Doe",
                "info.created_at": "11/08/1993",
                "info.last_updated": "26/05/2012",
            },
        ]

        # 断言处理后的结果与预期结果相等
        assert result == expected

    def test_nonetype_top_level_bottom_level(self):
        # GH21158: 如果内部 JSON 中有一个键的值为 null
        # 确保不会对 new_d.pop 执行两次，并且保持异常处理
        # 定义测试数据，包含多层嵌套的字典结构
        data = {
            "id": None,
            "location": {
                "country": {
                    "state": {
                        "id": None,
                        "town.info": {
                            "id": None,
                            "region": None,
                            "x": 49.151580810546875,
                            "y": -33.148521423339844,
                            "z": 27.572303771972656,
                        },
                    }
                }
            },
        }
        # 调用 nested_to_record 函数处理测试数据
        result = nested_to_record(data)
        # 定义预期输出结果，将每个键根据层级结构扁平化处理
        expected = {
            "id": None,
            "location.country.state.id": None,
            "location.country.state.town.info.id": None,
            "location.country.state.town.info.region": None,
            "location.country.state.town.info.x": 49.151580810546875,
            "location.country.state.town.info.y": -33.148521423339844,
            "location.country.state.town.info.z": 27.572303771972656,
        }
        # 断言处理后的结果与预期结果相等
        assert result == expected
    def test_nonetype_multiple_levels(self):
        # 测试用例：GH21158
        # 如果内部 JSON 包含键为 null 值，确保不会重复执行 new_d.pop 并引发异常
        data = {
            "id": None,
            "location": {
                "id": None,
                "country": {
                    "id": None,
                    "state": {
                        "id": None,
                        "town.info": {
                            "region": None,
                            "x": 49.151580810546875,
                            "y": -33.148521423339844,
                            "z": 27.572303771972656,
                        },
                    },
                },
            },
        }
        # 调用函数 nested_to_record 处理数据
        result = nested_to_record(data)
        # 期望的结果字典
        expected = {
            "id": None,
            "location.id": None,
            "location.country.id": None,
            "location.country.state.id": None,
            "location.country.state.town.info.region": None,
            "location.country.state.town.info.x": 49.151580810546875,
            "location.country.state.town.info.y": -33.148521423339844,
            "location.country.state.town.info.z": 27.572303771972656,
        }
        # 断言结果与期望是否一致
        assert result == expected

    @pytest.mark.parametrize(
        "max_level, expected",
        [
            (
                None,
                [
                    {
                        "CreatedBy.Name": "User001",
                        "Lookup.TextField": "Some text",
                        "Lookup.UserField.Id": "ID001",
                        "Lookup.UserField.Name": "Name001",
                        "Image.a": "b",
                    }
                ],
            ),
            (
                0,
                [
                    {
                        "CreatedBy": {"Name": "User001"},
                        "Lookup": {
                            "TextField": "Some text",
                            "UserField": {"Id": "ID001", "Name": "Name001"},
                        },
                        "Image": {"a": "b"},
                    }
                ],
            ),
            (
                1,
                [
                    {
                        "CreatedBy.Name": "User001",
                        "Lookup.TextField": "Some text",
                        "Lookup.UserField": {"Id": "ID001", "Name": "Name001"},
                        "Image.a": "b",
                    }
                ],
            ),
        ],
    )
    # 定义一个测试方法，用于测试带有最大层级限制的情况，验证 GH23843 的增强型 JSON 标准化功能
    def test_with_max_level(self, max_level, expected):
        # 准备带有特定结构的测试输入数据，用于验证处理嵌套结构转换的功能
        max_level_test_input_data = [
            {
                "CreatedBy": {"Name": "User001"},
                "Lookup": {
                    "TextField": "Some text",
                    "UserField": {"Id": "ID001", "Name": "Name001"},
                },
                "Image": {"a": "b"},
            }
        ]
        # 调用被测试的函数 nested_to_record，处理测试数据并返回转换后的结果
        output = nested_to_record(max_level_test_input_data, max_level=max_level)
        # 断言处理后的结果与预期结果相符
        assert output == expected

    # 定义一个测试方法，用于测试带有大量最大层级限制的情况，验证 GH23843 的增强型 JSON 标准化功能
    def test_with_large_max_level(self):
        # 设置较大的层级限制数
        max_level = 100
        # 准备带有深层嵌套结构的测试输入数据，用于验证处理嵌套结构转换的功能
        input_data = [
            {
                "CreatedBy": {
                    "user": {
                        "name": {"firstname": "Leo", "LastName": "Thomson"},
                        "family_tree": {
                            "father": {
                                "name": "Father001",
                                "father": {
                                    "Name": "Father002",
                                    "father": {
                                        "name": "Father003",
                                        "father": {"Name": "Father004"},
                                    },
                                },
                            }
                        },
                    }
                }
            }
        ]
        # 准备预期的转换结果，包含了将深层嵌套结构转换为扁平结构的数据格式
        expected = [
            {
                "CreatedBy.user.name.firstname": "Leo",
                "CreatedBy.user.name.LastName": "Thomson",
                "CreatedBy.user.family_tree.father.name": "Father001",
                "CreatedBy.user.family_tree.father.father.Name": "Father002",
                "CreatedBy.user.family_tree.father.father.father.name": "Father003",
                "CreatedBy.user.family_tree.father.father.father.father.Name": "Father004",
            }
        ]
        # 调用被测试的函数 nested_to_record，处理测试数据并返回转换后的结果
        output = nested_to_record(input_data, max_level=max_level)
        # 断言处理后的结果与预期结果相符
        assert output == expected

    # 定义一个测试方法，用于测试 Series 对象的 JSON 标准化功能，验证 GH 19020 的功能
    def test_series_non_zero_index(self):
        # 准备一个包含非零索引的数据字典，用于构建 Series 对象
        data = {
            0: {"id": 1, "name": "Foo", "elements": {"a": 1}},
            1: {"id": 2, "name": "Bar", "elements": {"b": 2}},
            2: {"id": 3, "name": "Baz", "elements": {"c": 3}},
        }
        # 使用数据字典构建 Series 对象
        s = Series(data)
        # 设置 Series 对象的索引值
        s.index = [1, 2, 3]
        # 调用 pandas 的 json_normalize 方法，将 Series 对象转换为规范化的 JSON 格式
        result = json_normalize(s)
        # 准备预期的 DataFrame 结果，包含了元素字段的扁平化表示
        expected = DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Foo", "Bar", "Baz"],
                "elements.a": [1.0, np.nan, np.nan],
                "elements.b": [np.nan, 2.0, np.nan],
                "elements.c": [np.nan, np.nan, 3.0],
            },
            index=[1, 2, 3],
        )
        # 使用 pandas 的 assert_frame_equal 方法断言处理后的结果与预期结果相符
        tm.assert_frame_equal(result, expected)
```