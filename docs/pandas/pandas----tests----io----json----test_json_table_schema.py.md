# `D:\src\scipysrc\pandas\pandas\tests\io\json\test_json_table_schema.py`

```
# 导入所需的库和模块
from collections import OrderedDict  # 导入有序字典模块
from io import StringIO  # 导入字符串IO模块
import json  # 导入JSON处理模块

import numpy as np  # 导入NumPy库
import pytest  # 导入pytest测试框架

import pandas as pd  # 导入Pandas库
from pandas import DataFrame  # 导入DataFrame类
import pandas._testing as tm  # 导入Pandas测试模块

from pandas.io.json._table_schema import (  # 导入JSON表格模式相关函数
    as_json_table_type,
    build_table_schema,
    convert_json_field_to_pandas_type,
    convert_pandas_type_to_json_field,
    set_default_names,
)


@pytest.fixture
def df_schema():
    # 创建一个包含特定数据类型和索引的DataFrame，用于测试
    return DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": ["a", "b", "c", "c"],
            "C": pd.date_range("2016-01-01", freq="d", periods=4),
            "D": pd.timedelta_range("1h", periods=4, freq="min"),
        },
        index=pd.Index(range(4), name="idx"),
    )


@pytest.fixture
def df_table():
    # 创建一个包含更多数据类型的DataFrame，用于更复杂的测试
    return DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": ["a", "b", "c", "c"],
            "C": pd.date_range("2016-01-01", freq="d", periods=4),
            "D": pd.timedelta_range("1h", periods=4, freq="min"),
            "E": pd.Series(pd.Categorical(["a", "b", "c", "c"])),
            "F": pd.Series(pd.Categorical(["a", "b", "c", "c"], ordered=True)),
            "G": [1.0, 2.0, 3, 4.0],
            "H": pd.date_range("2016-01-01", freq="d", periods=4, tz="US/Central"),
        },
        index=pd.Index(range(4), name="idx"),
    )


class TestBuildSchema:
    def test_build_table_schema(self, df_schema, using_infer_string):
        # 测试构建DataFrame的表格模式
        result = build_table_schema(df_schema, version=False)
        expected = {
            "fields": [
                {"name": "idx", "type": "integer"},
                {"name": "A", "type": "integer"},
                {"name": "B", "type": "string"},
                {"name": "C", "type": "datetime"},
                {"name": "D", "type": "duration"},
            ],
            "primaryKey": ["idx"],
        }
        if using_infer_string:
            expected["fields"][2] = {"name": "B", "type": "any", "extDtype": "string"}
        assert result == expected
        result = build_table_schema(df_schema)
        assert "pandas_version" in result

    def test_series(self):
        # 测试构建Series的表格模式
        s = pd.Series([1, 2, 3], name="foo")
        result = build_table_schema(s, version=False)
        expected = {
            "fields": [
                {"name": "index", "type": "integer"},
                {"name": "foo", "type": "integer"},
            ],
            "primaryKey": ["index"],
        }
        assert result == expected
        result = build_table_schema(s)
        assert "pandas_version" in result
    # 定义一个测试函数，测试处理未命名的 Pandas Series 的表格模式生成
    def test_series_unnamed(self):
        # 调用 build_table_schema 函数生成未命名 Series 的表格模式
        result = build_table_schema(pd.Series([1, 2, 3]), version=False)
        # 期望的结果是一个包含字段信息和主键信息的字典
        expected = {
            "fields": [
                {"name": "index", "type": "integer"},
                {"name": "values", "type": "integer"},
            ],
            "primaryKey": ["index"],
        }
        # 断言生成的结果与期望的结果相等
        assert result == expected

    # 定义一个测试函数，测试处理具有多级索引的 DataFrame 的表格模式生成
    def test_multiindex(self, df_schema, using_infer_string):
        # 获取一个具有多级索引的 DataFrame
        df = df_schema
        idx = pd.MultiIndex.from_product([("a", "b"), (1, 2)])
        df.index = idx

        # 调用 build_table_schema 函数生成多级索引 DataFrame 的表格模式
        result = build_table_schema(df, version=False)
        # 期望的结果是一个包含字段信息和主键信息的字典
        expected = {
            "fields": [
                {"name": "level_0", "type": "string"},
                {"name": "level_1", "type": "integer"},
                {"name": "A", "type": "integer"},
                {"name": "B", "type": "string"},
                {"name": "C", "type": "datetime"},
                {"name": "D", "type": "duration"},
            ],
            "primaryKey": ["level_0", "level_1"],
        }

        # 根据是否使用 infer_string 参数来调整期望结果的字段类型和扩展数据类型
        if using_infer_string:
            expected["fields"][0] = {
                "name": "level_0",
                "type": "any",
                "extDtype": "string",
            }
            expected["fields"][3] = {"name": "B", "type": "any", "extDtype": "string"}

        # 断言生成的结果与期望的结果相等
        assert result == expected

        # 修改 DataFrame 的索引名称和主键设置，再次调用 build_table_schema 函数生成表格模式
        df.index.names = ["idx0", None]
        expected["fields"][0]["name"] = "idx0"
        expected["primaryKey"] = ["idx0", "level_1"]
        result = build_table_schema(df, version=False)
        # 再次断言生成的结果与修改后的期望结果相等
        assert result == expected
# 定义一个测试类 TestTableSchemaType，用于测试 as_json_table_type 函数的各种数据类型
class TestTableSchemaType:
    
    # 使用 pytest.mark.parametrize 装饰器，参数化 int_type 变量为 int、np.int16、np.int32、np.int64
    @pytest.mark.parametrize("int_type", [int, np.int16, np.int32, np.int64])
    def test_as_json_table_type_int_data(self, int_type):
        # 准备整型数据 int_data
        int_data = [1, 2, 3]
        # 断言调用 as_json_table_type 函数对整型数据类型返回 "integer"
        assert as_json_table_type(np.array(int_data, dtype=int_type).dtype) == "integer"

    # 使用 pytest.mark.parametrize 装饰器，参数化 float_type 变量为 float、np.float16、np.float32、np.float64
    @pytest.mark.parametrize("float_type", [float, np.float16, np.float32, np.float64])
    def test_as_json_table_type_float_data(self, float_type):
        # 准备浮点型数据 float_data
        float_data = [1.0, 2.0, 3.0]
        # 断言调用 as_json_table_type 函数对浮点型数据类型返回 "number"
        assert as_json_table_type(np.array(float_data, dtype=float_type).dtype) == "number"

    # 使用 pytest.mark.parametrize 装饰器，参数化 bool_type 变量为 bool、np.bool_
    @pytest.mark.parametrize("bool_type", [bool, np.bool_])
    def test_as_json_table_type_bool_data(self, bool_type):
        # 准备布尔型数据 bool_data
        bool_data = [True, False]
        # 断言调用 as_json_table_type 函数对布尔型数据类型返回 "boolean"
        assert as_json_table_type(np.array(bool_data, dtype=bool_type).dtype) == "boolean"

    # 使用 pytest.mark.parametrize 装饰器，参数化 date_data 变量为不同日期时间数据类型
    @pytest.mark.parametrize(
        "date_data",
        [
            pd.to_datetime(["2016"]),
            pd.to_datetime(["2016"], utc=True),
            pd.Series(pd.to_datetime(["2016"])),
            pd.Series(pd.to_datetime(["2016"], utc=True)),
            pd.period_range("2016", freq="Y", periods=3),
        ],
    )
    def test_as_json_table_type_date_data(self, date_data):
        # 断言调用 as_json_table_type 函数对日期时间数据类型返回 "datetime"
        assert as_json_table_type(date_data.dtype) == "datetime"

    # 使用 pytest.mark.parametrize 装饰器，参数化 klass 变量为 pd.Series、pd.Index
    def test_as_json_table_type_string_data(self, klass):
        # 准备字符串数据 str_data
        str_data = klass(["a", "b"], dtype=object)
        # 断言调用 as_json_table_type 函数对字符串数据类型返回 "string"
        assert as_json_table_type(str_data.dtype) == "string"

    # 使用 pytest.mark.parametrize 装饰器，参数化 cat_data 变量为不同分类数据类型
    @pytest.mark.parametrize(
        "cat_data",
        [
            pd.Categorical(["a"]),
            pd.Categorical([1]),
            pd.Series(pd.Categorical([1])),
            pd.CategoricalIndex([1]),
        ],
    )
    def test_as_json_table_type_categorical_data(self, cat_data):
        # 断言调用 as_json_table_type 函数对分类数据类型返回 "any"
        assert as_json_table_type(cat_data.dtype) == "any"

    # ------
    # dtypes
    # ------

    # 使用 pytest.mark.parametrize 装饰器，参数化 int_dtype 变量为 int、np.int16、np.int32、np.int64
    @pytest.mark.parametrize("int_dtype", [int, np.int16, np.int32, np.int64])
    def test_as_json_table_type_int_dtypes(self, int_dtype):
        # 断言调用 as_json_table_type 函数对整型数据类型返回 "integer"
        assert as_json_table_type(int_dtype) == "integer"

    # 使用 pytest.mark.parametrize 装饰器，参数化 float_dtype 变量为 float、np.float16、np.float32、np.float64
    @pytest.mark.parametrize("float_dtype", [float, np.float16, np.float32, np.float64])
    def test_as_json_table_type_float_dtypes(self, float_dtype):
        # 断言调用 as_json_table_type 函数对浮点型数据类型返回 "number"
        assert as_json_table_type(float_dtype) == "number"

    # 使用 pytest.mark.parametrize 装饰器，参数化 bool_dtype 变量为 bool、np.bool_
    @pytest.mark.parametrize("bool_dtype", [bool, np.bool_])
    def test_as_json_table_type_bool_dtypes(self, bool_dtype):
        # 断言调用 as_json_table_type 函数对布尔型数据类型返回 "boolean"
        assert as_json_table_type(bool_dtype) == "boolean"

    # 使用 pytest.mark.parametrize 装饰器，参数化 date_dtype 变量为不同日期时间数据类型
    @pytest.mark.parametrize(
        "date_dtype",
        [
            np.dtype("<M8[ns]"),
            PeriodDtype("D"),
            DatetimeTZDtype("ns", "US/Central"),
        ],
    )
    def test_as_json_table_type_date_dtypes(self, date_dtype):
        # TODO: datedate.date? datetime.time?
        # 断言调用 as_json_table_type 函数对日期时间数据类型返回 "datetime"
        assert as_json_table_type(date_dtype) == "datetime"

    # 使用 pytest.mark.parametrize 装饰器，参数化 td_dtype 变量为 np.dtype("<m8[ns]")
    @pytest.mark.parametrize("td_dtype", [np.dtype("<m8[ns]")])
    # 测试函数：验证给定 timedelta 数据类型的 JSON 表格类型
    def test_as_json_table_type_timedelta_dtypes(self, td_dtype):
        # 断言：将给定 timedelta 数据类型转换为 JSON 表格类型应为 "duration"
        assert as_json_table_type(td_dtype) == "duration"

    # 使用参数化测试，测试不同字符串数据类型的 JSON 表格类型
    @pytest.mark.parametrize("str_dtype", [object])  # TODO(GH#14904) flesh out dtypes?
    def test_as_json_table_type_string_dtypes(self, str_dtype):
        # 断言：将给定字符串数据类型转换为 JSON 表格类型应为 "string"
        assert as_json_table_type(str_dtype) == "string"

    # 测试函数：验证分类数据类型的 JSON 表格类型
    def test_as_json_table_type_categorical_dtypes(self):
        # 断言：将给定的 Pandas 分类数据类型转换为 JSON 表格类型应为 "any"
        assert as_json_table_type(pd.Categorical(["a"]).dtype) == "any"
        # 断言：将给定的自定义分类数据类型转换为 JSON 表格类型应为 "any"
        assert as_json_table_type(CategoricalDtype()) == "any"
    # 定义一个测试类 TestTableOrient
    class TestTableOrient:
        
        # 测试方法：测试构建 Series 对象并转换为 JSON 表格式
        def test_build_series(self):
            # 创建一个 Series 对象 s，包含值 [1, 2]，列名为 "a"
            s = pd.Series([1, 2], name="a")
            # 设置 Series 对象 s 的索引名为 "id"
            s.index.name = "id"
            # 将 Series 对象 s 转换为 JSON 表格式，日期格式为 ISO 8601
            result = s.to_json(orient="table", date_format="iso")
            # 解析 JSON 结果为 OrderedDict 对象 result
            result = json.loads(result, object_pairs_hook=OrderedDict)
    
            # 断言：检查结果字典中是否包含键 "pandas_version"
            assert "pandas_version" in result["schema"]
            # 移除结果字典中的 "pandas_version" 键
            result["schema"].pop("pandas_version")
    
            # 定义期望的 OrderedDict 对象 expected，包含表结构 schema 和数据 data
            fields = [{"name": "id", "type": "integer"}, {"name": "a", "type": "integer"}]
            schema = {"fields": fields, "primaryKey": ["id"]}
            expected = OrderedDict(
                [
                    ("schema", schema),
                    (
                        "data",
                        [
                            OrderedDict([("id", 0), ("a", 1)]),
                            OrderedDict([("id", 1), ("a", 2)]),
                        ],
                    ),
                ]
            )
    
            # 断言：检查结果与期望是否相等
            assert result == expected
    
        # 测试方法：验证从 JSON 转回 DataFrame 后的数据一致性
        def test_read_json_from_to_json_results(self):
            # 创建一个 DataFrame 对象 df，包含多个列和行索引
            df = DataFrame(
                {
                    "_id": {"row_0": 0},
                    "category": {"row_0": "Goods"},
                    "recommender_id": {"row_0": 3},
                    "recommender_name_jp": {"row_0": "浦田"},
                    "recommender_name_en": {"row_0": "Urata"},
                    "name_jp": {"row_0": "博多人形(松尾吉将まつお よしまさ)"},
                    "name_en": {"row_0": "Hakata Dolls Matsuo"},
                }
            )
    
            # 从 DataFrame 对象 df 转换为 JSON 字符串，再由 JSON 字符串转回 DataFrame 对象 result1
            result1 = pd.read_json(StringIO(df.to_json()))
            # 从 DataFrame 对象 df 转换为 JSON 字典，再由 JSON 字典转回 DataFrame 对象 result2
            result2 = DataFrame.from_dict(json.loads(df.to_json()))
    
            # 断言：验证转换后的 DataFrame 对象与原始 df 对象是否相等
            tm.assert_frame_equal(result1, df)
            tm.assert_frame_equal(result2, df)
    
        # 测试方法：测试将带有浮点数索引的 Series 对象转换为 JSON 表格式
        def test_to_json_float_index(self):
            # 创建一个带有浮点数索引的 Series 对象 data，值为 1，索引为 [1.0, 2.0]
            data = pd.Series(1, index=[1.0, 2.0])
            # 将 Series 对象 data 转换为 JSON 表格式，日期格式为 ISO 8601
            result = data.to_json(orient="table", date_format="iso")
            # 解析 JSON 结果为 OrderedDict 对象 result
            result = json.loads(result, object_pairs_hook=OrderedDict)
            # 移除结果字典中的 "pandas_version" 键
            result["schema"].pop("pandas_version")
    
            # 定义期望的 OrderedDict 对象 expected，包含表结构 schema 和数据 data
            expected = OrderedDict(
                [
                    (
                        "schema",
                        {
                            "fields": [
                                {"name": "index", "type": "number"},
                                {"name": "values", "type": "integer"},
                            ],
                            "primaryKey": ["index"],
                        },
                    ),
                    (
                        "data",
                        [
                            OrderedDict([("index", 1.0), ("values", 1)]),
                            OrderedDict([("index", 2.0), ("values", 1)]),
                        ],
                    ),
                ]
            )
    
            # 断言：检查结果与期望是否相等
            assert result == expected
    # 定义一个测试方法，用于测试将周期索引转换为 JSON 格式
    def test_to_json_period_index(self):
        # 创建一个时间周期范围，从"2016"开始，频率为每季度，共2个周期
        idx = pd.period_range("2016", freq="Q-JAN", periods=2)
        # 创建一个包含数据的时间周期序列
        data = pd.Series(1, idx)
        # 将时间周期序列转换为 JSON 格式，使用表格格式，并指定日期格式为 ISO 格式
        result = data.to_json(orient="table", date_format="iso")
        # 将 JSON 字符串解析为 OrderedDict 对象
        result = json.loads(result, object_pairs_hook=OrderedDict)
        # 从结果中删除键为"pandas_version"的元素
        result["schema"].pop("pandas_version")

        # 定义字段列表，描述数据的结构
        fields = [
            {"freq": "QE-JAN", "name": "index", "type": "datetime"},
            {"name": "values", "type": "integer"},
        ]

        # 定义数据的结构，使用 OrderedDict 表示每条数据记录
        schema = {"fields": fields, "primaryKey": ["index"]}
        data = [
            OrderedDict([("index", "2015-11-01T00:00:00.000"), ("values", 1)]),
            OrderedDict([("index", "2016-02-01T00:00:00.000"), ("values", 1)]),
        ]
        # 定义预期的结果结构，包含 schema 和 data 两个部分
        expected = OrderedDict([("schema", schema), ("data", data)])

        # 断言结果与预期结果相等
        assert result == expected

    # 定义一个测试方法，用于测试将分类索引转换为 JSON 格式
    def test_to_json_categorical_index(self):
        # 创建一个包含数据的分类索引序列
        data = pd.Series(1, pd.CategoricalIndex(["a", "b"]))
        # 将分类索引序列转换为 JSON 格式，使用表格格式，并指定日期格式为 ISO 格式
        result = data.to_json(orient="table", date_format="iso")
        # 将 JSON 字符串解析为 OrderedDict 对象
        result = json.loads(result, object_pairs_hook=OrderedDict)
        # 从结果中删除键为"pandas_version"的元素
        result["schema"].pop("pandas_version")

        # 定义预期的结果结构，包含 schema 和 data 两个部分
        expected = OrderedDict(
            [
                (
                    "schema",
                    {
                        "fields": [
                            {
                                "name": "index",
                                "type": "any",
                                "constraints": {"enum": ["a", "b"]},
                                "ordered": False,
                            },
                            {"name": "values", "type": "integer"},
                        ],
                        "primaryKey": ["index"],
                    },
                ),
                (
                    "data",
                    [
                        OrderedDict([("index", "a"), ("values", 1)]),
                        OrderedDict([("index", "b"), ("values", 1)]),
                    ],
                ),
            ]
        )

        # 断言结果与预期结果相等
        assert result == expected

    # 定义一个测试方法，用于测试以'epoch'日期格式写入时的异常情况
    def test_date_format_raises(self, df_table):
        # 定义错误消息和警告消息
        error_msg = (
            "Trying to write with `orient='table'` and `date_format='epoch'`. Table "
            "Schema requires dates to be formatted with `date_format='iso'`"
        )
        warning_msg = (
            "'epoch' date format is deprecated and will be removed in a future "
            "version, please use 'iso' date format instead."
        )
        # 断言在使用'epoch'日期格式时抛出 ValueError 异常，并匹配指定错误消息
        with pytest.raises(ValueError, match=error_msg):
            # 断言在产生 FutureWarning 警告时匹配指定警告消息
            with tm.assert_produces_warning(FutureWarning, match=warning_msg):
                # 尝试将 DataFrame 以表格格式和'epoch'日期格式写入 JSON
                df_table.to_json(orient="table", date_format="epoch")

        # 断言其它情况下能正常工作，将 DataFrame 以表格格式和ISO日期格式写入 JSON
        df_table.to_json(orient="table", date_format="iso")
        # 将 DataFrame 以表格格式写入 JSON，不指定日期格式
        df_table.to_json(orient="table")
    # 测试将 Pandas 类型转换为 JSON 字段的函数，处理整数类型
    def test_convert_pandas_type_to_json_field_int(self, index_or_series):
        # 从参数中获取数据类型（索引或系列）
        kind = index_or_series
        # 创建一个整数数据列表
        data = [1, 2, 3]
        # 调用函数转换 Pandas 类型到 JSON 字段
        result = convert_pandas_type_to_json_field(kind(data, name="name"))
        # 预期的 JSON 字段
        expected = {"name": "name", "type": "integer"}
        # 断言结果是否符合预期
        assert result == expected

    # 测试将 Pandas 类型转换为 JSON 字段的函数，处理浮点数类型
    def test_convert_pandas_type_to_json_field_float(self, index_or_series):
        # 从参数中获取数据类型（索引或系列）
        kind = index_or_series
        # 创建一个浮点数数据列表
        data = [1.0, 2.0, 3.0]
        # 调用函数转换 Pandas 类型到 JSON 字段
        result = convert_pandas_type_to_json_field(kind(data, name="name"))
        # 预期的 JSON 字段
        expected = {"name": "name", "type": "number"}
        # 断言结果是否符合预期
        assert result == expected

    # 使用参数化测试标记，测试将 Pandas 类型转换为 JSON 字段的函数，处理日期时间类型
    @pytest.mark.parametrize(
        "dt_args,extra_exp", [({}, {}), ({"utc": True}, {"tz": "UTC"})]
    )
    @pytest.mark.parametrize("wrapper", [None, pd.Series])
    def test_convert_pandas_type_to_json_field_datetime(
        self, dt_args, extra_exp, wrapper
    ):
        # 创建一个浮点数数据列表
        data = [1.0, 2.0, 3.0]
        # 使用给定参数将数据转换为日期时间类型
        data = pd.to_datetime(data, **dt_args)
        # 如果 wrapper 是 pd.Series，则创建一个系列对象
        if wrapper is pd.Series:
            data = pd.Series(data, name="values")
        # 调用函数转换 Pandas 类型到 JSON 字段
        result = convert_pandas_type_to_json_field(data)
        # 预期的 JSON 字段
        expected = {"name": "values", "type": "datetime"}
        # 更新预期 JSON 字段的额外期望值
        expected.update(extra_exp)
        # 断言结果是否符合预期
        assert result == expected

    # 测试将 Pandas 类型转换为 JSON 字段的函数，处理期间范围类型
    def test_convert_pandas_type_to_json_period_range(self):
        # 创建一个期间范围对象，从2016年开始，以年为周期，周期数为4
        arr = pd.period_range("2016", freq="Y-DEC", periods=4)
        # 调用函数转换 Pandas 类型到 JSON 字段
        result = convert_pandas_type_to_json_field(arr)
        # 预期的 JSON 字段
        expected = {"name": "values", "type": "datetime", "freq": "Y-DEC"}
        # 断言结果是否符合预期
        assert result == expected

    # 使用参数化测试标记，测试将 Pandas 类型转换为 JSON 字段的函数，处理分类数据类型
    @pytest.mark.parametrize("kind", [pd.Categorical, pd.CategoricalIndex])
    @pytest.mark.parametrize("ordered", [True, False])
    def test_convert_pandas_type_to_json_field_categorical(self, kind, ordered):
        # 创建一个分类数据列表
        data = ["a", "b", "c"]
        # 根据 kind 的不同，创建不同的 Pandas 对象
        if kind is pd.Categorical:
            arr = pd.Series(kind(data, ordered=ordered), name="cats")
        elif kind is pd.CategoricalIndex:
            arr = kind(data, ordered=ordered, name="cats")

        # 调用函数转换 Pandas 类型到 JSON 字段
        result = convert_pandas_type_to_json_field(arr)
        # 预期的 JSON 字段
        expected = {
            "name": "cats",
            "type": "any",
            "constraints": {"enum": data},
            "ordered": ordered,
        }
        # 断言结果是否符合预期
        assert result == expected
    @pytest.mark.parametrize(
        "inp,exp",
        [  # 参数化测试输入和期望输出
            ({"type": "integer"}, "int64"),  # 输入为整数类型，期望输出为'int64'
            ({"type": "number"}, "float64"),  # 输入为数字类型，期望输出为'float64'
            ({"type": "boolean"}, "bool"),  # 输入为布尔类型，期望输出为'bool'
            ({"type": "duration"}, "timedelta64"),  # 输入为持续时间类型，期望输出为'timedelta64'
            ({"type": "datetime"}, "datetime64[ns]"),  # 输入为日期时间类型，期望输出为'datetime64[ns]'
            ({"type": "datetime", "tz": "US/Hawaii"}, "datetime64[ns, US/Hawaii]"),  # 输入为带时区的日期时间类型，期望输出为'datetime64[ns, US/Hawaii]'
            ({"type": "any"}, "object"),  # 输入为任意类型，期望输出为'object'
            (  # 复杂输入和期望输出，使用CategoricalDtype表示
                {
                    "type": "any",
                    "constraints": {"enum": ["a", "b", "c"]},
                    "ordered": False,
                },
                CategoricalDtype(categories=["a", "b", "c"], ordered=False),
            ),
            (  # 复杂输入和期望输出，使用CategoricalDtype表示
                {
                    "type": "any",
                    "constraints": {"enum": ["a", "b", "c"]},
                    "ordered": True,
                },
                CategoricalDtype(categories=["a", "b", "c"], ordered=True),
            ),
            ({"type": "string"}, "object"),  # 输入为字符串类型，期望输出为'object'
        ],
    )
    def test_convert_json_field_to_pandas_type(self, inp, exp):
        # 准备字段字典，更新输入字段
        field = {"name": "foo"}
        field.update(inp)
        # 断言调用函数将字段转换为 Pandas 类型后与期望输出一致
        assert convert_json_field_to_pandas_type(field) == exp

    @pytest.mark.parametrize("inp", ["geopoint", "geojson", "fake_type"])
    def test_convert_json_field_to_pandas_type_raises(self, inp):
        # 准备包含类型字段的字典
        field = {"type": inp}
        # 使用 pytest.raises 检查调用函数时是否抛出 ValueError 异常，异常信息与输入类型匹配
        with pytest.raises(
            ValueError, match=f"Unsupported or invalid field type: {inp}"
        ):
            convert_json_field_to_pandas_type(field)

    def test_categorical(self):
        # 创建包含分类数据的 Pandas Series
        s = pd.Series(pd.Categorical(["a", "b", "a"]))
        s.index.name = "idx"  # 设置索引名称
        # 将 Series 转换为 JSON 格式数据，保留顺序
        result = s.to_json(orient="table", date_format="iso")
        result = json.loads(result, object_pairs_hook=OrderedDict)
        result["schema"].pop("pandas_version")  # 移除 schema 中的 pandas 版本信息

        # 准备预期的 OrderedDict 结构
        fields = [
            {"name": "idx", "type": "integer"},
            {
                "constraints": {"enum": ["a", "b"]},
                "name": "values",
                "ordered": False,
                "type": "any",
            },
        ]
        expected = OrderedDict(
            [
                ("schema", {"fields": fields, "primaryKey": ["idx"]}),
                (
                    "data",
                    [
                        OrderedDict([("idx", 0), ("values", "a")]),
                        OrderedDict([("idx", 1), ("values", "b")]),
                        OrderedDict([("idx", 2), ("values", "a")]),
                    ],
                ),
            ]
        )

        # 断言结果与预期输出一致
        assert result == expected
    @pytest.mark.parametrize(
        "idx,nm,prop",
        [   # 使用 pytest 的 parametrize 装饰器来定义多组测试参数
            (pd.Index([1]), "index", "name"),  # 创建单索引对象，nm 为 "index"，prop 为 "name"
            (pd.Index([1], name="myname"), "myname", "name"),  # 创建命名为 "myname" 的单索引对象，prop 为 "name"
            (
                pd.MultiIndex.from_product([("a", "b"), ("c", "d")]),  # 创建两层的 MultiIndex 对象
                ["level_0", "level_1"],  # 预期的索引级别名称列表
                "names",  # 属性名称应为 "names"
            ),
            (
                pd.MultiIndex.from_product(
                    [("a", "b"), ("c", "d")], names=["n1", "n2"]  # 创建命名为 "n1", "n2" 的 MultiIndex 对象
                ),
                ["n1", "n2"],  # 预期的索引级别名称列表
                "names",  # 属性名称应为 "names"
            ),
            (
                pd.MultiIndex.from_product(
                    [("a", "b"), ("c", "d")], names=["n1", None]  # 创建命名为 "n1" 和未命名的 MultiIndex 对象
                ),
                ["n1", "level_1"],  # 预期的索引级别名称列表
                "names",  # 属性名称应为 "names"
            ),
        ],
    )
    def test_set_names_unset(self, idx, nm, prop):
        # 创建测试数据
        data = pd.Series(1, idx)
        # 调用被测函数 set_default_names 处理数据
        result = set_default_names(data)
        # 断言结果中的属性 prop 是否等于预期的名称 nm
        assert getattr(result.index, prop) == nm

    @pytest.mark.parametrize(
        "idx",
        [
            pd.Index([], name="index"),  # 创建一个空的命名为 "index" 的索引对象
            pd.MultiIndex.from_arrays([["foo"], ["bar"]], names=("level_0", "level_1")),  # 创建两层的 MultiIndex 对象，命名为 "level_0" 和 "level_1"
            pd.MultiIndex.from_arrays([["foo"], ["bar"]], names=("foo", "level_1")),  # 创建两层的 MultiIndex 对象，命名为 "foo" 和 "level_1"
        ],
    )
    def test_warns_non_roundtrippable_names(self, idx):
        # GH 19130
        # 创建 DataFrame 对象，使用给定的索引
        df = DataFrame(index=idx)
        # 设置 DataFrame 的索引名称为 "index"
        df.index.name = "index"
        # 使用 pytest 的 assert_produces_warning 断言，确保在操作中会产生 UserWarning 并匹配 "not round-trippable"
        with tm.assert_produces_warning(UserWarning, match="not round-trippable"):
            # 调用被测函数 set_default_names 处理 DataFrame
            set_default_names(df)

    def test_timestamp_in_columns(self):
        # 创建具有时间戳和时间增量列的 DataFrame
        df = DataFrame(
            [[1, 2]], columns=[pd.Timestamp("2016"), pd.Timedelta(10, unit="s")]
        )
        # 将 DataFrame 转换为 JSON 字符串，指定 orient="table"
        result = df.to_json(orient="table")
        # 解析 JSON 结果
        js = json.loads(result)
        # 断言 JSON 结果中的字段名是否与预期的时间戳和时间增量匹配
        assert js["schema"]["fields"][1]["name"] == "2016-01-01T00:00:00.000"
        assert js["schema"]["fields"][2]["name"] == "P0DT0H0M10S"

    @pytest.mark.parametrize(
        "case",
        [
            pd.Series([1], index=pd.Index([1], name="a"), name="a"),  # 创建命名为 "a" 的单索引 Series 对象
            DataFrame({"A": [1]}, index=pd.Index([1], name="A")),  # 创建命名为 "A" 的 DataFrame
            DataFrame(
                {"A": [1]},
                index=pd.MultiIndex.from_arrays([["a"], [1]], names=["A", "a"]),  # 创建两层的 MultiIndex 对象，命名为 "A" 和 "a"
            ),
        ],
    )
    def test_overlapping_names(self, case):
        # 使用 pytest 的 raises 断言，检查是否会引发 ValueError 异常并匹配 "Overlapping"
        with pytest.raises(ValueError, match="Overlapping"):
            # 将测试数据 case 转换为 JSON 字符串，orient="table"
            case.to_json(orient="table")

    def test_mi_falsey_name(self):
        # GH 16203
        # 创建具有随机数据的 DataFrame，使用 MultiIndex
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=pd.MultiIndex.from_product([("A", "B"), ("a", "b")]),  # 创建两层的 MultiIndex 对象
        )
        # 调用 build_table_schema 函数，获取其字段列表中的 "name" 属性
        result = [x["name"] for x in build_table_schema(df)["fields"]]
        # 断言结果是否符合预期的字段名列表
        assert result == ["level_0", "level_1", 0, 1, 2, 3]
class TestTableOrientReader:
    # 使用 pytest 的参数化标记，测试不同的 index_nm 值
    @pytest.mark.parametrize(
        "index_nm",
        [None, "idx", pytest.param("index", marks=pytest.mark.xfail), "level_0"],
    )
    # 使用 pytest 的参数化标记，测试不同的 vals 数据
    @pytest.mark.parametrize(
        "vals",
        [
            {"ints": [1, 2, 3, 4]},  # 整数列表
            {"objects": ["a", "b", "c", "d"]},  # 对象列表
            {"objects": ["1", "2", "3", "4"]},  # 对象列表（字符串）
            {"date_ranges": pd.date_range("2016-01-01", freq="d", periods=4)},  # 日期范围
            {"categoricals": pd.Series(pd.Categorical(["a", "b", "c", "c"]))},  # 分类数据
            {"ordered_cats": pd.Series(pd.Categorical(["a", "b", "c", "c"], ordered=True))},  # 有序分类数据
            {"floats": [1.0, 2.0, 3.0, 4.0]},  # 浮点数列表
            {"floats": [1.1, 2.2, 3.3, 4.4]},  # 浮点数列表
            {"bools": [True, False, False, True]},  # 布尔值列表
            {"timezones": pd.date_range("2016-01-01", freq="d", periods=4, tz="US/Central")},  # 时区日期范围
        ],
    )
    # 测试读取 JSON 表格格式数据的方法
    def test_read_json_table_orient(self, index_nm, vals):
        # 根据 vals 创建 DataFrame，设置索引名称为 index_nm
        df = DataFrame(vals, index=pd.Index(range(4), name=index_nm))
        # 将 DataFrame 转换为 JSON 字符串，并封装成 StringIO 对象
        out = StringIO(df.to_json(orient="table"))
        # 使用 pd.read_json 方法读取 JSON 字符串，指定 orient="table"
        result = pd.read_json(out, orient="table")
        # 断言 DataFrame 是否相等
        tm.assert_frame_equal(df, result)

    # 使用 pytest 的参数化标记，测试不同的 index_nm 值
    @pytest.mark.parametrize(
        "index_nm",
        [
            None,
            "idx",
            pytest.param(
                "index",
                marks=pytest.mark.filterwarnings("ignore:Index name:UserWarning"),
            ),
        ],
    )
    # 测试读取 JSON 表格格式数据时抛出异常的情况
    def test_read_json_table_orient_raises(self, index_nm):
        # 创建一个包含 timedelta 数据的 vals 字典
        vals = {"timedeltas": pd.timedelta_range("1h", periods=4, freq="min")}
        # 根据 vals 创建 DataFrame，设置索引名称为 index_nm
        df = DataFrame(vals, index=pd.Index(range(4), name=index_nm))
        # 将 DataFrame 转换为 JSON 字符串，并封装成 StringIO 对象
        out = StringIO(df.to_json(orient="table"))
        # 使用 pytest 的断言来检查是否抛出预期的 NotImplementedError 异常
        with pytest.raises(NotImplementedError, match="can not yet read "):
            pd.read_json(out, orient="table")

    # 使用 pytest 的参数化标记，测试不同的 index_nm 值
    @pytest.mark.parametrize(
        "index_nm",
        [None, "idx", pytest.param("index", marks=pytest.mark.xfail), "level_0"],
    )
    # 使用 pytest 的参数化标记，测试不同的 vals 数据，与第一个测试方法内容相同
    @pytest.mark.parametrize(
        "vals",
        [
            {"ints": [1, 2, 3, 4]},
            {"objects": ["a", "b", "c", "d"]},
            {"objects": ["1", "2", "3", "4"]},
            {"date_ranges": pd.date_range("2016-01-01", freq="d", periods=4)},
            {"categoricals": pd.Series(pd.Categorical(["a", "b", "c", "c"]))},
            {"ordered_cats": pd.Series(pd.Categorical(["a", "b", "c", "c"], ordered=True))},
            {"floats": [1.0, 2.0, 3.0, 4.0]},
            {"floats": [1.1, 2.2, 3.3, 4.4]},
            {"bools": [True, False, False, True]},
            {"timezones": pd.date_range("2016-01-01", freq="d", periods=4, tz="US/Central")},
        ],
    )
    # 测试读取 JSON 表格格式数据并按周期导向索引的情况
    def test_read_json_table_period_orient(self, index_nm, vals):
        # 创建 DataFrame 对象，使用周期作为索引名称，vals 提供数据
        df = DataFrame(
            vals,
            index=pd.Index(
                (pd.Period(f"2022Q{q}") for q in range(1, 5)), name=index_nm
            ),
        )
        # 将 DataFrame 转换为 JSON 格式并写入 StringIO 对象
        out = StringIO(df.to_json(orient="table"))
        # 从 JSON 字符串中读取数据并转换为 DataFrame 对象
        result = pd.read_json(out, orient="table")
        # 使用 pytest 框架断言两个 DataFrame 对象是否相等
        tm.assert_frame_equal(df, result)

    @pytest.mark.parametrize(
        "idx",
        [
            pd.Index(range(4)),  # 创建标准整数索引
            pd.date_range(
                "2020-08-30",
                freq="d",
                periods=4,
            )._with_freq(None),  # 创建无频率的日期时间索引
            pd.date_range(
                "2020-08-30", freq="d", periods=4, tz="US/Central"
            )._with_freq(None),  # 创建带时区的日期时间索引
            pd.MultiIndex.from_product(
                [
                    pd.date_range("2020-08-30", freq="d", periods=2, tz="US/Central"),
                    ["x", "y"],
                ],  # 创建多级索引
            ),
        ],
    )
    @pytest.mark.parametrize(
        "vals",
        [
            {"floats": [1.1, 2.2, 3.3, 4.4]},  # 浮点数值字典
            {"dates": pd.date_range("2020-08-30", freq="d", periods=4)},  # 日期时间序列字典
            {
                "timezones": pd.date_range(
                    "2020-08-30", freq="d", periods=4, tz="Europe/London"
                )
            },  # 带时区的日期时间序列字典
        ],
    )
    # 测试读取 JSON 表格格式数据并处理时区索引的情况
    def test_read_json_table_timezones_orient(self, idx, vals):
        # 创建 DataFrame 对象，使用给定的索引和数据
        df = DataFrame(vals, index=idx)
        # 将 DataFrame 转换为 JSON 格式并写入 StringIO 对象
        out = StringIO(df.to_json(orient="table"))
        # 从 JSON 字符串中读取数据并转换为 DataFrame 对象
        result = pd.read_json(out, orient="table")
        # 使用 pytest 框架断言两个 DataFrame 对象是否相等
        tm.assert_frame_equal(df, result)

    # 综合测试多种数据类型的 DataFrame 转换和读取
    def test_comprehensive(self):
        # 创建包含多种数据类型的 DataFrame 对象
        df = DataFrame(
            {
                "A": [1, 2, 3, 4],
                "B": ["a", "b", "c", "c"],
                "C": pd.date_range("2016-01-01", freq="d", periods=4),
                "E": pd.Series(pd.Categorical(["a", "b", "c", "c"])),
                "F": pd.Series(pd.Categorical(["a", "b", "c", "c"], ordered=True)),
                "G": [1.1, 2.2, 3.3, 4.4],
                "H": pd.date_range("2016-01-01", freq="d", periods=4, tz="US/Central"),
                "I": [True, False, False, True],
            },
            index=pd.Index(range(4), name="idx"),  # 设置整数索引并命名
        )

        # 将 DataFrame 转换为 JSON 格式并写入 StringIO 对象
        out = StringIO(df.to_json(orient="table"))
        # 从 JSON 字符串中读取数据并转换为 DataFrame 对象
        result = pd.read_json(out, orient="table")
        # 使用 pytest 框架断言两个 DataFrame 对象是否相等
        tm.assert_frame_equal(df, result)

    @pytest.mark.parametrize(
        "index_names",
        [[None, None], ["foo", "bar"], ["foo", None], [None, "foo"], ["index", "foo"]],
    )
    # 定义一个测试函数，用于测试多重索引的情况
    def test_multiindex(self, index_names):
        # GH 18912
        # 创建一个 DataFrame 对象，包含多重索引和数据
        df = DataFrame(
            [["Arr", "alpha", [1, 2, 3, 4]], ["Bee", "Beta", [10, 20, 30, 40]]],
            index=[["A", "B"], ["Null", "Eins"]],  # 设置多重索引的名称
            columns=["Aussprache", "Griechisch", "Args"],  # 设置列名
        )
        df.index.names = index_names  # 将索引名称设置为给定的参数 index_names
        # 将 DataFrame 转换为 JSON 格式，并写入 StringIO 对象
        out = StringIO(df.to_json(orient="table"))
        # 从 JSON 数据读取 DataFrame 对象
        result = pd.read_json(out, orient="table")
        # 使用测试工具比较两个 DataFrame 对象是否相等
        tm.assert_frame_equal(df, result)

    # 定义一个测试函数，测试空 DataFrame 的序列化和反序列化过程
    def test_empty_frame_roundtrip(self):
        # GH 21287
        # 创建一个空的 DataFrame 对象，列名为 ["a", "b", "c"]
        df = DataFrame(columns=["a", "b", "c"])
        expected = df.copy()  # 复制预期结果
        # 将 DataFrame 对象转换为 JSON 格式，并写入 StringIO 对象
        out = StringIO(df.to_json(orient="table"))
        # 从 JSON 数据读取 DataFrame 对象
        result = pd.read_json(out, orient="table")
        # 使用测试工具比较预期结果和实际结果的 DataFrame 对象是否相等
        tm.assert_frame_equal(expected, result)

    # 定义一个测试函数，测试读取 JSON 数据并转换为 DataFrame 的过程
    def test_read_json_orient_table_old_schema_version(self):
        # 准备一个 JSON 字符串，描述 DataFrame 结构和数据
        df_json = """
        {
            "schema":{
                "fields":[
                    {"name":"index","type":"integer"},
                    {"name":"a","type":"string"}
                ],
                "primaryKey":["index"],
                "pandas_version":"0.20.0"
            },
            "data":[
                {"index":0,"a":1},
                {"index":1,"a":2.0},
                {"index":2,"a":"s"}
            ]
        }
        """
        expected = DataFrame({"a": [1, 2.0, "s"]})  # 期望的 DataFrame 结果
        # 从 JSON 字符串读取 DataFrame 对象
        result = pd.read_json(StringIO(df_json), orient="table")
        # 使用测试工具比较预期结果和实际结果的 DataFrame 对象是否相等
        tm.assert_frame_equal(expected, result)

    # 使用 pytest 的参数化装饰器，定义多个频率的测试用例
    @pytest.mark.parametrize("freq", ["M", "2M", "Q", "2Q", "Y", "2Y"])
    def test_read_json_table_orient_period_depr_freq(self, freq):
        # GH#9586
        # 创建一个带有 PeriodIndex 的 DataFrame 对象
        df = DataFrame(
            {"ints": [1, 2]},
            index=pd.PeriodIndex(["2020-01", "2021-06"], freq=freq),  # 设置 PeriodIndex 索引
        )
        # 将 DataFrame 对象转换为 JSON 格式，并写入 StringIO 对象
        out = StringIO(df.to_json(orient="table"))
        # 从 JSON 数据读取 DataFrame 对象
        result = pd.read_json(out, orient="table")
        # 使用测试工具比较两个 DataFrame 对象是否相等
        tm.assert_frame_equal(df, result)
```