# `D:\src\scipysrc\pandas\pandas\tests\io\json\test_json_table_schema_ext_dtype.py`

```
"""Tests for ExtensionDtype Table Schema integration."""

# 引入所需的模块和库
from collections import OrderedDict  # 从 collections 库中引入 OrderedDict
import datetime as dt  # 导入 datetime 库并用别名 dt
import decimal  # 导入 decimal 模块
from io import StringIO  # 从 io 库中导入 StringIO 类
import json  # 导入 json 模块

import pytest  # 导入 pytest 测试框架

from pandas import (  # 从 pandas 库中导入多个类和函数
    NA,  # 导入 NA 常量
    DataFrame,  # 导入 DataFrame 类
    Index,  # 导入 Index 类
    array,  # 导入 array 函数
    read_json,  # 导入 read_json 函数
)
import pandas._testing as tm  # 导入 pandas 内部测试模块
from pandas.core.arrays.integer import Int64Dtype  # 从 pandas 库中导入 Int64Dtype 类
from pandas.core.arrays.string_ import StringDtype  # 从 pandas 库中导入 StringDtype 类
from pandas.core.series import Series  # 从 pandas 库中导入 Series 类
from pandas.tests.extension.date import (  # 导入扩展日期相关模块
    DateArray,  # 导入 DateArray 类
    DateDtype,  # 导入 DateDtype 类
)
from pandas.tests.extension.decimal.array import (  # 导入扩展十进制相关模块
    DecimalArray,  # 导入 DecimalArray 类
    DecimalDtype,  # 导入 DecimalDtype 类
)

from pandas.io.json._table_schema import (  # 从 pandas 的 JSON IO 模块中导入表格模式相关函数
    as_json_table_type,  # 导入 as_json_table_type 函数
    build_table_schema,  # 导入 build_table_schema 函数
)


class TestBuildSchema:
    def test_build_table_schema(self):
        # 创建一个包含不同数据类型的 DataFrame 对象 df
        df = DataFrame(
            {
                "A": DateArray([dt.date(2021, 10, 10)]),  # 使用 DateArray 类创建日期数组
                "B": DecimalArray([decimal.Decimal(10)]),  # 使用 DecimalArray 类创建十进制数组
                "C": array(["pandas"], dtype="string"),  # 创建一个字符串类型的数组
                "D": array([10], dtype="Int64"),  # 创建一个整数类型的数组，使用 Int64 类型
            }
        )
        # 调用 build_table_schema 函数生成数据框架的表格模式
        result = build_table_schema(df, version=False)
        # 期望的表格模式结构
        expected = {
            "fields": [
                {"name": "index", "type": "integer"},  # 索引字段的描述
                {"name": "A", "type": "any", "extDtype": "DateDtype"},  # A 列的描述，使用 DateDtype 扩展类型
                {"name": "B", "type": "number", "extDtype": "decimal"},  # B 列的描述，使用 decimal 扩展类型
                {"name": "C", "type": "any", "extDtype": "string"},  # C 列的描述，使用 string 扩展类型
                {"name": "D", "type": "integer", "extDtype": "Int64"},  # D 列的描述，使用 Int64 扩展类型
            ],
            "primaryKey": ["index"],  # 主键定义为索引
        }
        # 断言结果与期望相符
        assert result == expected
        # 再次调用 build_table_schema 函数，检查结果中是否包含 pandas 版本信息
        result = build_table_schema(df)
        assert "pandas_version" in result


class TestTableSchemaType:
    @pytest.mark.parametrize("box", [lambda x: x, Series])
    def test_as_json_table_type_ext_date_array_dtype(self, box):
        # 创建包含日期的数据对象 date_data
        date_data = box(DateArray([dt.date(2021, 10, 10)]))
        # 调用 as_json_table_type 函数，验证日期数组的类型描述
        assert as_json_table_type(date_data.dtype) == "any"

    def test_as_json_table_type_ext_date_dtype(self):
        # 调用 as_json_table_type 函数，验证日期类型的类型描述
        assert as_json_table_type(DateDtype()) == "any"

    @pytest.mark.parametrize("box", [lambda x: x, Series])
    def test_as_json_table_type_ext_decimal_array_dtype(self, box):
        # 创建包含十进制数据的数据对象 decimal_data
        decimal_data = box(DecimalArray([decimal.Decimal(10)]))
        # 调用 as_json_table_type 函数，验证十进制数组的类型描述
        assert as_json_table_type(decimal_data.dtype) == "number"

    def test_as_json_table_type_ext_decimal_dtype(self):
        # 调用 as_json_table_type 函数，验证十进制类型的类型描述
        assert as_json_table_type(DecimalDtype()) == "number"

    @pytest.mark.parametrize("box", [lambda x: x, Series])
    def test_as_json_table_type_ext_string_array_dtype(self, box):
        # 创建包含字符串数据的数据对象 string_data
        string_data = box(array(["pandas"], dtype="string"))
        # 调用 as_json_table_type 函数，验证字符串数组的类型描述
        assert as_json_table_type(string_data.dtype) == "any"

    def test_as_json_table_type_ext_string_dtype(self):
        # 调用 as_json_table_type 函数，验证字符串类型的类型描述
        assert as_json_table_type(StringDtype()) == "any"

    @pytest.mark.parametrize("box", [lambda x: x, Series])
    # 定义测试方法，验证对于带有整数数组数据类型扩展的情况，as_json_table_type 函数的行为是否正确
    def test_as_json_table_type_ext_integer_array_dtype(self, box):
        # 创建包含一个整数的数组，并使用 box 函数包装
        integer_data = box(array([10], dtype="Int64"))
        # 断言调用 as_json_table_type 函数返回的类型为 "integer"
        assert as_json_table_type(integer_data.dtype) == "integer"

    # 定义测试方法，验证对于整数数据类型扩展的情况，as_json_table_type 函数的行为是否正确
    def test_as_json_table_type_ext_integer_dtype(self):
        # 断言调用 as_json_table_type 函数返回的类型为 "integer"，传入 Int64Dtype 类型作为参数
        assert as_json_table_type(Int64Dtype()) == "integer"
class TestTableOrient:
    # 定义一个 pytest 的 fixture，返回一个包含日期对象的 DateArray
    @pytest.fixture
    def da(self):
        return DateArray([dt.date(2021, 10, 10)])

    # 定义一个 pytest 的 fixture，返回一个包含 Decimal 对象的 DecimalArray
    @pytest.fixture
    def dc(self):
        return DecimalArray([decimal.Decimal(10)])

    # 定义一个 pytest 的 fixture，返回一个包含字符串的 array 对象
    @pytest.fixture
    def sa(self):
        return array(["pandas"], dtype="string")

    # 定义一个 pytest 的 fixture，返回一个包含整数的 array 对象
    @pytest.fixture
    def ia(self):
        return array([10], dtype="Int64")

    # 测试构建日期系列的方法，接受一个名为 da 的日期数组 fixture
    def test_build_date_series(self, da):
        # 使用 da 构建一个 Series 对象，并指定名称为 "a"
        s = Series(da, name="a")
        # 设置 Series 对象的索引名称为 "id"
        s.index.name = "id"
        # 将 Series 对象转换为 JSON 格式，使用 table 方式，并指定日期格式为 ISO
        result = s.to_json(orient="table", date_format="iso")
        # 将 JSON 格式的结果解析为 OrderedDict 对象
        result = json.loads(result, object_pairs_hook=OrderedDict)

        # 断言结果中包含 "pandas_version" 字段
        assert "pandas_version" in result["schema"]
        # 移除结果中的 "pandas_version" 字段
        result["schema"].pop("pandas_version")

        # 定义期望的字段结构列表
        fields = [
            {"name": "id", "type": "integer"},
            {"name": "a", "type": "any", "extDtype": "DateDtype"},
        ]

        # 定义期望的 schema 结构，包含字段结构和主键信息
        schema = {"fields": fields, "primaryKey": ["id"]}

        # 定义期望的完整 JSON 结果，使用 OrderedDict 保证顺序
        expected = OrderedDict(
            [
                ("schema", schema),
                ("data", [OrderedDict([("id", 0), ("a", "2021-10-10T00:00:00.000")])]),
            ]
        )

        # 断言实际结果与期望结果相等
        assert result == expected

    # 测试构建 Decimal 系列的方法，接受一个名为 dc 的 Decimal 数组 fixture
    def test_build_decimal_series(self, dc):
        # 使用 dc 构建一个 Series 对象，并指定名称为 "a"
        s = Series(dc, name="a")
        # 设置 Series 对象的索引名称为 "id"
        s.index.name = "id"
        # 将 Series 对象转换为 JSON 格式，使用 table 方式，并指定日期格式为 ISO
        result = s.to_json(orient="table", date_format="iso")
        # 将 JSON 格式的结果解析为 OrderedDict 对象
        result = json.loads(result, object_pairs_hook=OrderedDict)

        # 断言结果中包含 "pandas_version" 字段
        assert "pandas_version" in result["schema"]
        # 移除结果中的 "pandas_version" 字段
        result["schema"].pop("pandas_version")

        # 定义期望的字段结构列表
        fields = [
            {"name": "id", "type": "integer"},
            {"name": "a", "type": "number", "extDtype": "decimal"},
        ]

        # 定义期望的 schema 结构，包含字段结构和主键信息
        schema = {"fields": fields, "primaryKey": ["id"]}

        # 定义期望的完整 JSON 结果，使用 OrderedDict 保证顺序
        expected = OrderedDict(
            [
                ("schema", schema),
                ("data", [OrderedDict([("id", 0), ("a", 10.0)])]),
            ]
        )

        # 断言实际结果与期望结果相等
        assert result == expected

    # 测试构建字符串系列的方法，接受一个名为 sa 的字符串数组 fixture
    def test_build_string_series(self, sa):
        # 使用 sa 构建一个 Series 对象，并指定名称为 "a"
        s = Series(sa, name="a")
        # 设置 Series 对象的索引名称为 "id"
        s.index.name = "id"
        # 将 Series 对象转换为 JSON 格式，使用 table 方式，并指定日期格式为 ISO
        result = s.to_json(orient="table", date_format="iso")
        # 将 JSON 格式的结果解析为 OrderedDict 对象
        result = json.loads(result, object_pairs_hook=OrderedDict)

        # 断言结果中包含 "pandas_version" 字段
        assert "pandas_version" in result["schema"]
        # 移除结果中的 "pandas_version" 字段
        result["schema"].pop("pandas_version")

        # 定义期望的字段结构列表
        fields = [
            {"name": "id", "type": "integer"},
            {"name": "a", "type": "any", "extDtype": "string"},
        ]

        # 定义期望的 schema 结构，包含字段结构和主键信息
        schema = {"fields": fields, "primaryKey": ["id"]}

        # 定义期望的完整 JSON 结果，使用 OrderedDict 保证顺序
        expected = OrderedDict(
            [
                ("schema", schema),
                ("data", [OrderedDict([("id", 0), ("a", "pandas")])]),
            ]
        )

        # 断言实际结果与期望结果相等
        assert result == expected
    # 定义测试函数，测试构建 Int64 类型的 Series
    def test_build_int64_series(self, ia):
        # 使用传入的 ia 创建一个 Series，命名为 "a"
        s = Series(ia, name="a")
        # 设置 Series 的索引名称为 "id"
        s.index.name = "id"
        # 将 Series 转换为 JSON 格式的字符串，按表格方式，日期格式为 ISO
        result = s.to_json(orient="table", date_format="iso")
        # 将 JSON 字符串解析为有序字典对象
        result = json.loads(result, object_pairs_hook=OrderedDict)

        # 断言结果中包含 "pandas_version" 字段
        assert "pandas_version" in result["schema"]
        # 移除结果中的 "pandas_version" 字段
        result["schema"].pop("pandas_version")

        # 定义字段列表，包括 id 和 a，a 的类型为 integer，扩展类型为 Int64
        fields = [
            {"name": "id", "type": "integer"},
            {"name": "a", "type": "integer", "extDtype": "Int64"},
        ]

        # 定义 schema，包含字段列表和主键 "id"
        schema = {"fields": fields, "primaryKey": ["id"]}

        # 定义预期结果，为有序字典对象，包含 schema 和 data
        expected = OrderedDict(
            [
                ("schema", schema),
                ("data", [OrderedDict([("id", 0), ("a", 10)])]),
            ]
        )

        # 断言结果与预期结果相等
        assert result == expected

    # 定义测试函数，测试 DataFrame 的 to_json 方法
    def test_to_json(self, da, dc, sa, ia):
        # 使用传入的 da, dc, sa, ia 创建一个 DataFrame
        df = DataFrame(
            {
                "A": da,
                "B": dc,
                "C": sa,
                "D": ia,
            }
        )
        # 设置 DataFrame 的索引名称为 "idx"
        df.index.name = "idx"
        # 将 DataFrame 转换为 JSON 格式的字符串，按表格方式，日期格式为 ISO
        result = df.to_json(orient="table", date_format="iso")
        # 将 JSON 字符串解析为有序字典对象
        result = json.loads(result, object_pairs_hook=OrderedDict)

        # 断言结果中包含 "pandas_version" 字段
        assert "pandas_version" in result["schema"]
        # 移除结果中的 "pandas_version" 字段
        result["schema"].pop("pandas_version")

        # 定义字段列表，每个字段使用有序字典包装，指定 name 和 type，部分字段还指定 extDtype
        fields = [
            OrderedDict({"name": "idx", "type": "integer"}),
            OrderedDict({"name": "A", "type": "any", "extDtype": "DateDtype"}),
            OrderedDict({"name": "B", "type": "number", "extDtype": "decimal"}),
            OrderedDict({"name": "C", "type": "any", "extDtype": "string"}),
            OrderedDict({"name": "D", "type": "integer", "extDtype": "Int64"}),
        ]

        # 定义 schema，包含字段列表和主键 "idx"
        schema = OrderedDict({"fields": fields, "primaryKey": ["idx"]})
        # 定义 data，为包含一个有序字典对象的列表
        data = [
            OrderedDict(
                [
                    ("idx", 0),
                    ("A", "2021-10-10T00:00:00.000"),
                    ("B", 10.0),
                    ("C", "pandas"),
                    ("D", 10),
                ]
            )
        ]
        # 定义预期结果，为有序字典对象，包含 schema 和 data
        expected = OrderedDict([("schema", schema), ("data", data)])

        # 断言结果与预期结果相等
        assert result == expected

    # 定义测试函数，测试 JSON 扩展数据类型的读写往返
    def test_json_ext_dtype_reading_roundtrip(self):
        # GH#40255
        # 创建包含不同扩展数据类型的 DataFrame
        df = DataFrame(
            {
                "a": Series([2, NA], dtype="Int64"),
                "b": Series([1.5, NA], dtype="Float64"),
                "c": Series([True, NA], dtype="boolean"),
            },
            # 设置 DataFrame 的索引类型为 Int64
            index=Index([1, NA], dtype="Int64"),
        )
        # 复制 DataFrame 作为预期结果
        expected = df.copy()
        # 将 DataFrame 转换为美观格式的 JSON 字符串，缩进为 4
        data_json = df.to_json(orient="table", indent=4)
        # 从 JSON 字符串读取数据，转换为 DataFrame
        result = read_json(StringIO(data_json), orient="table")
        # 断言结果与预期结果相等
        tm.assert_frame_equal(result, expected)
    def test_json_ext_dtype_reading(self):
        # 定义一个测试方法，测试 JSON 文件扩展数据类型读取功能
        # GH#40255 是 GitHub 上的一个问题或特性请求的编号
        data_json = """{
            "schema":{
                "fields":[
                    {
                        "name":"a",
                        "type":"integer",
                        "extDtype":"Int64"
                    }
                ],
            },
            "data":[
                {
                    "a":2
                },
                {
                    "a":null
                }
            ]
        }"""
        # 调用 read_json 函数读取内存中的 JSON 数据，按表格方式解析
        result = read_json(StringIO(data_json), orient="table")
        # 创建期望的 DataFrame 对象，包含列 'a'，数据类型为 'Int64'
        expected = DataFrame({"a": Series([2, NA], dtype="Int64")})
        # 使用测试框架中的 assert_frame_equal 检查结果和期望值是否相等
        tm.assert_frame_equal(result, expected)
```