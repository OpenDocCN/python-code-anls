# `D:\src\scipysrc\pandas\pandas\tests\series\accessors\test_struct_accessor.py`

```
# 导入正则表达式模块
import re

# 导入 pytest 测试框架
import pytest

# 从 pandas.compat.pyarrow 中导入特定函数
from pandas.compat.pyarrow import (
    pa_version_under11p0,  # 导入判断 pyarrow 版本是否小于 11.0.0 的函数
    pa_version_under13p0,  # 导入判断 pyarrow 版本是否小于 13.0.0 的函数
)

# 从 pandas 中导入特定类和函数
from pandas import (
    ArrowDtype,  # 导入 ArrowDtype 类
    DataFrame,   # 导入 DataFrame 类
    Index,       # 导入 Index 类
    Series,      # 导入 Series 类
)

# 导入 pandas._testing 模块作为 tm 别名
import pandas._testing as tm

# 导入 pytest 测试框架并用 pa 别名引用，如果导入失败则跳过测试
pa = pytest.importorskip("pyarrow")

# 导入 pytest 测试框架并用 pc 别名引用，如果导入失败则跳过测试
pc = pytest.importorskip("pyarrow.compute")


# 定义测试函数 test_struct_accessor_dtypes
def test_struct_accessor_dtypes():
    # 创建一个空的 Series 对象，指定其 dtype 为 ArrowDtype
    ser = Series(
        [],
        dtype=ArrowDtype(
            pa.struct(  # 定义一个 pyarrow 的 struct 类型
                [
                    ("int_col", pa.int64()),        # 定义 int_col 字段为 int64 类型
                    ("string_col", pa.string()),    # 定义 string_col 字段为 string 类型
                    (
                        "struct_col",                # 定义 struct_col 字段为 struct 类型
                        pa.struct(                  # struct_col 内部定义一个新的 struct 类型
                            [
                                ("int_col", pa.int64()),     # 定义 int_col 字段为 int64 类型
                                ("float_col", pa.float64()),  # 定义 float_col 字段为 float64 类型
                            ]
                        ),
                    ),
                ]
            )
        ),
    )
    # 获取 ser 对象的 struct 属性的 dtypes 属性，并将结果赋给 actual 变量
    actual = ser.struct.dtypes
    # 创建一个期望的 Series 对象 expected
    expected = Series(
        [
            ArrowDtype(pa.int64()),                     # int_col 字段的预期类型
            ArrowDtype(pa.string()),                   # string_col 字段的预期类型
            ArrowDtype(
                pa.struct(                             # struct_col 字段的预期类型
                    [
                        ("int_col", pa.int64()),        # 内部 int_col 字段的预期类型
                        ("float_col", pa.float64()),    # 内部 float_col 字段的预期类型
                    ]
                )
            ),
        ],
        index=Index(["int_col", "string_col", "struct_col"]),  # 指定索引的预期值
    )
    # 使用 pandas._testing 模块的 assert_series_equal 方法比较 actual 和 expected 是否相等
    tm.assert_series_equal(actual, expected)


# 用 pytest.mark.skipif 装饰器标记，如果 pyarrow 版本小于 13.0.0，则跳过测试
@pytest.mark.skipif(pa_version_under13p0, reason="pyarrow>=13.0.0 required")
# 定义测试函数 test_struct_accessor_field
def test_struct_accessor_field():
    # 创建一个 Index 对象 index
    index = Index([-100, 42, 123])
    # 创建一个包含字典的 Series 对象 ser，指定其 dtype 为 ArrowDtype
    ser = Series(
        [
            {"rice": 1.0, "maize": -1, "wheat": "a"},
            {"rice": 2.0, "maize": 0, "wheat": "b"},
            {"rice": 3.0, "maize": 1, "wheat": "c"},
        ],
        dtype=ArrowDtype(
            pa.struct(                             # 定义一个 pyarrow 的 struct 类型
                [
                    ("rice", pa.float64()),        # 定义 rice 字段为 float64 类型
                    ("maize", pa.int64()),         # 定义 maize 字段为 int64 类型
                    ("wheat", pa.string()),        # 定义 wheat 字段为 string 类型
                ]
            )
        ),
        index=index,                             # 指定 Series 对象的索引为 index
    )
    # 使用 struct 属性的 field 方法获取 "maize" 字段的 Series，并将结果赋给 by_name 变量
    by_name = ser.struct.field("maize")
    # 创建一个期望的 Series 对象 by_name_expected
    by_name_expected = Series(
        [-1, 0, 1],                              # 预期的 "maize" 字段的值
        dtype=ArrowDtype(pa.int64()),             # 预期的数据类型为 int64
        index=index,                             # 预期的索引与 ser 相同
        name="maize",                            # 预期的 Series 对象名称为 "maize"
    )
    # 使用 pandas._testing 模块的 assert_series_equal 方法比较 by_name 和 by_name_expected 是否相等
    tm.assert_series_equal(by_name, by_name_expected)

    # 使用 struct 属性的 field 方法获取索引为 2 的字段的 Series，并将结果赋给 by_index 变量
    by_index = ser.struct.field(2)
    # 创建一个期望的 Series 对象 by_index_expected
    by_index_expected = Series(
        ["a", "b", "c"],                         # 预期的索引为 2 的字段的值
        dtype=ArrowDtype(pa.string()),           # 预期的数据类型为 string
        index=index,                             # 预期的索引与 ser 相同
        name="wheat",                            # 预期的 Series 对象名称为 "wheat"
    )
    # 使用 pandas._testing 模块的 assert_series_equal 方法比较 by_index 和 by_index_expected 是否相等
    tm.assert_series_equal(by_index, by_index_expected)


# 定义测试函数 test_struct_accessor_field_with_invalid_name_or_index
def test_struct_accessor_field_with_invalid_name_or_index():
    # 创建一个空的 Series 对象 ser，指定其 dtype 为 ArrowDtype
    ser = Series([], dtype=ArrowDtype(pa.struct([("field", pa.int64())])))

    # 使用 pytest.raises 检查是否抛出 ValueError 异常，并匹配指定的错误消息
    with pytest.raises(ValueError, match="name_or_index must be an int, str,"):
        ser.struct.field(1.1)


# 用 pytest.mark.skipif 装饰器标记，如果 pyarrow 版本小于 11.0.0，则跳过测试
@pytest.mark.skipif(pa_version_under11p0, reason="pyarrow>=11.0.0 required")
# 定义测试函数 test_struct_accessor_explode
def test_struct_accessor_explode():
    # 创建一个 Index 对象 index
    index = Index([-100, 42, 123])
    # 创建一个包含多个字典的序列，每个字典包括一个整数和一个嵌套的字典
    ser = Series(
        [
            {"painted": 1, "snapping": {"sea": "green"}},
            {"painted": 2, "snapping": {"sea": "leatherback"}},
            {"painted": 3, "snapping": {"sea": "hawksbill"}},
        ],
        # 定义序列的数据类型为 ArrowDtype，其中包含一个结构，包括整数和嵌套结构字段
        dtype=ArrowDtype(
            pa.struct(
                [
                    ("painted", pa.int64()),
                    ("snapping", pa.struct([("sea", pa.string())])),
                ]
            )
        ),
        # 使用给定的索引初始化序列的索引
        index=index,
    )
    # 展开序列中的结构字段，返回一个新的序列
    actual = ser.struct.explode()
    # 创建一个数据框，包含两列："painted" 和 "snapping"，每列分别是一个序列
    expected = DataFrame(
        {
            "painted": Series([1, 2, 3], index=index, dtype=ArrowDtype(pa.int64())),
            "snapping": Series(
                [{"sea": "green"}, {"sea": "leatherback"}, {"sea": "hawksbill"}],
                index=index,
                dtype=ArrowDtype(pa.struct([("sea", pa.string())])),
            ),
        },
    )
    # 使用测试工具比较两个数据框的内容，确保它们相等
    tm.assert_frame_equal(actual, expected)
# 使用 pytest.mark.parametrize 装饰器标记测试函数，参数化测试输入
@pytest.mark.parametrize(
    "invalid",  # 参数名称为 invalid
    [
        pytest.param(Series([1, 2, 3], dtype="int64"), id="int64"),  # 参数为整数类型 Series，标记为 'int64'
        pytest.param(
            Series(["a", "b", "c"], dtype="string[pyarrow]"), id="string-pyarrow"
        ),  # 参数为字符串类型 Series，标记为 'string-pyarrow'
    ],
)
# 定义测试函数 test_struct_accessor_api_for_invalid，接受参数 invalid
def test_struct_accessor_api_for_invalid(invalid):
    # 使用 pytest.raises 检查是否抛出 AttributeError 异常
    with pytest.raises(
        AttributeError,
        match=re.escape(
            "Can only use the '.struct' accessor with 'struct[pyarrow]' dtype, "
            f"not {invalid.dtype}."
        ),  # 匹配异常信息，指出不能使用 '.struct' 访问器的 dtype
    ):
        invalid.struct  # 调用被测对象的 .struct 属性


# 使用 pytest.mark.parametrize 装饰器标记测试函数，参数化测试输入
@pytest.mark.parametrize(
    ["indices", "name"],  # 参数名称为 indices 和 name
    [
        (0, "int_col"),  # 参数为整数 0，名称为 'int_col'
        ([1, 2], "str_col"),  # 参数为整数列表 [1, 2]，名称为 'str_col'
        (pc.field("int_col"), "int_col"),  # 参数为字段 "int_col"，名称为 'int_col'
        ("int_col", "int_col"),  # 参数为字符串 "int_col"，名称为 'int_col'
        (b"string_col", b"string_col"),  # 参数为字节字符串 b"string_col"，名称为 b"string_col"
        ([b"string_col"], "string_col"),  # 参数为字节字符串列表 [b"string_col"]，名称为 'string_col'
    ],
)
# 根据 pyarrow 版本条件，如果版本小于 13.0.0，则跳过测试
@pytest.mark.skipif(pa_version_under13p0, reason="pyarrow>=13.0.0 required")
# 定义测试函数 test_struct_accessor_field_expanded，接受参数 indices 和 name
def test_struct_accessor_field_expanded(indices, name):
    # 创建 pyarrow 结构类型 arrow_type
    arrow_type = pa.struct(
        [
            ("int_col", pa.int64()),  # 包含整数字段 "int_col"
            (
                "struct_col",  # 包含结构字段 "struct_col"
                pa.struct(
                    [
                        ("int_col", pa.int64()),  # 结构字段包含整数字段 "int_col"
                        ("float_col", pa.float64()),  # 结构字段包含浮点数字段 "float_col"
                        ("str_col", pa.string()),  # 结构字段包含字符串字段 "str_col"
                    ]
                ),
            ),
            (b"string_col", pa.string()),  # 包含字节字符串字段 b"string_col"
        ]
    )

    data = pa.array([], type=arrow_type)  # 创建空数组 data，类型为 arrow_type
    ser = Series(data, dtype=ArrowDtype(arrow_type))  # 创建 pandas Series 对象 ser，数据类型为 ArrowDtype(arrow_type)
    expected = pc.struct_field(data, indices)  # 获取预期的结构字段值，根据 indices
    result = ser.struct.field(indices)  # 获取 ser 对象的结构字段值，根据 indices
    tm.assert_equal(result.array._pa_array.combine_chunks(), expected)  # 断言 result 与 expected 相等
    assert result.name == name  # 断言 result 的名称与 name 相等
```