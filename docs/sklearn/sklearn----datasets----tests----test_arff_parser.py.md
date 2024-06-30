# `D:\src\scipysrc\scikit-learn\sklearn\datasets\tests\test_arff_parser.py`

```
# 导入所需的模块和函数
import textwrap  # 导入用于处理文本格式的模块
from io import BytesIO  # 导入用于处理字节流的模块

import pytest  # 导入 pytest 测试框架

# 导入与 ARFF 解析相关的函数
from sklearn.datasets._arff_parser import (
    _liac_arff_parser,
    _pandas_arff_parser,
    _post_process_frame,
    load_arff_from_gzip_file,
)

# 使用 pytest 的参数化装饰器，为测试用例传递不同的参数组合
@pytest.mark.parametrize(
    "feature_names, target_names",
    [
        (
            [
                "col_int_as_integer",
                "col_int_as_numeric",
                "col_float_as_real",
                "col_float_as_numeric",
            ],
            ["col_categorical", "col_string"],
        ),
        (
            [
                "col_int_as_integer",
                "col_int_as_numeric",
                "col_float_as_real",
                "col_float_as_numeric",
            ],
            ["col_categorical"],
        ),
        (
            [
                "col_int_as_integer",
                "col_int_as_numeric",
                "col_float_as_real",
                "col_float_as_numeric",
            ],
            [],
        ),
    ],
)
def test_post_process_frame(feature_names, target_names):
    """检查对数据帧进行后处理以分割的函数行为。"""
    pd = pytest.importorskip("pandas")  # 导入 pandas 并跳过失败的情况

    # 创建原始的测试数据帧 X_original
    X_original = pd.DataFrame(
        {
            "col_int_as_integer": [1, 2, 3],
            "col_int_as_numeric": [1, 2, 3],
            "col_float_as_real": [1.0, 2.0, 3.0],
            "col_float_as_numeric": [1.0, 2.0, 3.0],
            "col_categorical": ["a", "b", "c"],
            "col_string": ["a", "b", "c"],
        }
    )

    # 调用 _post_process_frame 函数进行数据帧的后处理
    X, y = _post_process_frame(X_original, feature_names, target_names)

    # 断言返回的 X 是 pandas 的 DataFrame 类型
    assert isinstance(X, pd.DataFrame)

    # 根据目标名称列表的长度进行不同的断言
    if len(target_names) >= 2:
        assert isinstance(y, pd.DataFrame)
    elif len(target_names) == 1:
        assert isinstance(y, pd.Series)
    else:
        assert y is None


def test_load_arff_from_gzip_file_error_parser():
    """如果解析器未知，则会引发错误。"""
    # 由于首先进行解析器的检查，因此输入参数的准确性不是必须的。

    # 预期的错误信息
    err_msg = "Unknown parser: 'xxx'. Should be 'liac-arff' or 'pandas'"

    # 使用 pytest 的断言捕获检查特定错误消息的 ValueError 异常
    with pytest.raises(ValueError, match=err_msg):
        load_arff_from_gzip_file("xxx", "xxx", "xxx", "xxx", "xxx", "xxx")


@pytest.mark.parametrize("parser_func", [_liac_arff_parser, _pandas_arff_parser])
def test_pandas_arff_parser_strip_single_quotes(parser_func):
    """检查我们是否正确地去除数据中的单引号。"""
    pd = pytest.importorskip("pandas")  # 导入 pandas 并跳过失败的情况

    # 创建一个模拟的 ARFF 文件字节流对象 arff_file
    arff_file = BytesIO(
        textwrap.dedent(
            """
            @relation 'toy'
            @attribute 'cat_single_quote' {'A', 'B', 'C'}
            @attribute 'str_single_quote' string
            @attribute 'str_nested_quote' string
            @attribute 'class' numeric
            @data
            'A','some text','\"expect double quotes\"',0
            """
        ).encode("utf-8")
    )
    # 定义一个字典，包含列名及其对应的数据类型和名称
    columns_info = {
        "cat_single_quote": {
            "data_type": "nominal",
            "name": "cat_single_quote",
        },
        "str_single_quote": {
            "data_type": "string",
            "name": "str_single_quote",
        },
        "str_nested_quote": {
            "data_type": "string",
            "name": "str_nested_quote",
        },
        "class": {
            "data_type": "numeric",
            "name": "class",
        },
    }

    # 定义一个列表，包含需要作为特征使用的列名
    feature_names = [
        "cat_single_quote",
        "str_single_quote",
        "str_nested_quote",
    ]

    # 定义一个列表，包含目标列名
    target_names = ["class"]

    # 定义一个字典，包含各列的预期值
    expected_values = {
        "cat_single_quote": "A",
        "str_single_quote": (
            "some text" if parser_func is _liac_arff_parser else "'some text'"
        ),
        "str_nested_quote": (
            '"expect double quotes"'
            if parser_func is _liac_arff_parser
            else "'\"expect double quotes\"'"
        ),
        "class": 0,
    }

    # 调用解析函数 parser_func 处理 arff_file 文件，返回的第三个元素作为数据帧 frame
    _, _, frame, _ = parser_func(
        arff_file,
        output_arrays_type="pandas",
        openml_columns_info=columns_info,
        feature_names_to_select=feature_names,
        target_names_to_select=target_names,
    )

    # 断言数据帧的列名列表与特征列名和目标列名的组合相等
    assert frame.columns.tolist() == feature_names + target_names

    # 断言数据帧的第一行数据等于预期的值，使用 pandas 的测试工具进行比较
    pd.testing.assert_series_equal(frame.iloc[0], pd.Series(expected_values, name=0))
@pytest.mark.parametrize("parser_func", [_liac_arff_parser, _pandas_arff_parser])
# 使用 pytest 的 parametrize 装饰器，为测试函数提供多个参数化的调用方式，每次调用将传入不同的 parser_func 参数
def test_pandas_arff_parser_strip_double_quotes(parser_func):
    """Check that we properly strip double quotes from the data."""
    # 导入 pytest 库，并跳过如果 pandas 库不存在的情况
    pd = pytest.importorskip("pandas")

    # 创建一个 BytesIO 对象，内容是 ARFF 格式的数据，使用 textwrap.dedent 去除缩进后编码成 utf-8
    arff_file = BytesIO(
        textwrap.dedent(
            """
            @relation 'toy'
            @attribute 'cat_double_quote' {"A", "B", "C"}
            @attribute 'str_double_quote' string
            @attribute 'str_nested_quote' string
            @attribute 'class' numeric
            @data
            "A","some text","\'expect double quotes\'",0
            """
        ).encode("utf-8")
    )

    # 定义列的信息字典，包含每列的数据类型和名称
    columns_info = {
        "cat_double_quote": {
            "data_type": "nominal",
            "name": "cat_double_quote",
        },
        "str_double_quote": {
            "data_type": "string",
            "name": "str_double_quote",
        },
        "str_nested_quote": {
            "data_type": "string",
            "name": "str_nested_quote",
        },
        "class": {
            "data_type": "numeric",
            "name": "class",
        },
    }

    # 定义需要选择的特征列名和目标列名
    feature_names = [
        "cat_double_quote",
        "str_double_quote",
        "str_nested_quote",
    ]
    target_names = ["class"]

    # 定义期望的值字典，包含每列的预期值
    expected_values = {
        "cat_double_quote": "A",
        "str_double_quote": "some text",
        "str_nested_quote": "'expect double quotes'",
        "class": 0,
    }

    # 调用 parser_func 函数，解析 ARFF 文件，并返回四个结果，我们只需要最后一个结果 frame
    _, _, frame, _ = parser_func(
        arff_file,
        output_arrays_type="pandas",
        openml_columns_info=columns_info,
        feature_names_to_select=feature_names,
        target_names_to_select=target_names,
    )

    # 断言 frame 的列名列表等于特征列名加上目标列名
    assert frame.columns.tolist() == feature_names + target_names
    # 使用 pandas.testing.assert_series_equal 检查 frame 的第一行数据是否等于预期的值，设置 Series 名称为 0
    pd.testing.assert_series_equal(frame.iloc[0], pd.Series(expected_values, name=0))


@pytest.mark.parametrize(
    "parser_func",
    [
        # 对于 LIAC ARFF，内部引号不符合 ARFF 规范，标记为失败
        pytest.param(_liac_arff_parser, marks=pytest.mark.xfail),
        _pandas_arff_parser,
    ],
)
# 使用 pytest 的 parametrize 装饰器，为测试函数提供多个参数化的调用方式，其中 _liac_arff_parser 标记为失败
def test_pandas_arff_parser_strip_no_quotes(parser_func):
    """Check that we properly parse with no quotes characters."""
    # 导入 pytest 库，并跳过如果 pandas 库不存在的情况
    pd = pytest.importorskip("pandas")

    # 创建一个 BytesIO 对象，内容是 ARFF 格式的数据，使用 textwrap.dedent 去除缩进后编码成 utf-8
    arff_file = BytesIO(
        textwrap.dedent(
            """
            @relation 'toy'
            @attribute 'cat_without_quote' {A, B, C}
            @attribute 'str_without_quote' string
            @attribute 'str_internal_quote' string
            @attribute 'class' numeric
            @data
            A,some text,'internal' quote,0
            """
        ).encode("utf-8")
    )
    # 定义列的信息字典，每个键表示一个特征列，值是一个包含数据类型和列名的字典
    columns_info = {
        "cat_without_quote": {
            "data_type": "nominal",
            "name": "cat_without_quote",
        },
        "str_without_quote": {
            "data_type": "string",
            "name": "str_without_quote",
        },
        "str_internal_quote": {
            "data_type": "string",
            "name": "str_internal_quote",
        },
        "class": {
            "data_type": "numeric",
            "name": "class",
        },
    }

    # 定义要选取的特征列名列表
    feature_names = [
        "cat_without_quote",
        "str_without_quote",
        "str_internal_quote",
    ]
    
    # 定义目标列名列表
    target_names = ["class"]

    # 定义期望的特征值字典，每个键对应特征列名，值是预期的特征值
    expected_values = {
        "cat_without_quote": "A",
        "str_without_quote": "some text",
        "str_internal_quote": "'internal' quote",
        "class": 0,
    }

    # 调用解析函数，解析给定的 ARFF 文件
    # 将解析结果存储在 frame 变量中，使用 Pandas 数据结构表示，其他返回值未使用（使用下划线 _ 表示）
    _, _, frame, _ = parser_func(
        arff_file,
        output_arrays_type="pandas",
        openml_columns_info=columns_info,
        feature_names_to_select=feature_names,
        target_names_to_select=target_names,
    )

    # 断言 DataFrame 的列列表与预期的特征列名和目标列名一致
    assert frame.columns.tolist() == feature_names + target_names
    
    # 使用 Pandas 的测试工具，断言 DataFrame 的第一行数据与预期的特征值字典相等
    pd.testing.assert_series_equal(frame.iloc[0], pd.Series(expected_values, name=0))
```