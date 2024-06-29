# `D:\src\scipysrc\pandas\pandas\tests\arrays\string_\test_string_arrow.py`

```
# 导入pickle模块，用于序列化和反序列化Python对象
import pickle
# 导入re模块，用于处理正则表达式
import re

# 导入numpy库，并使用np作为别名
import numpy as np
# 导入pytest库，用于编写和运行测试
import pytest

# 导入pandas.util._test_decorators模块，提供用于测试的装饰器
import pandas.util._test_decorators as td

# 导入pandas库，并使用pd作为别名
import pandas as pd
# 导入pandas._testing模块，包含用于测试的实用函数和类
import pandas._testing as tm

# 导入pandas.core.arrays.string_模块中的StringArray和StringDtype类
from pandas.core.arrays.string_ import (
    StringArray,
    StringDtype,
)
# 导入pandas.core.arrays.string_arrow模块中的ArrowStringArray和ArrowStringArrayNumpySemantics类
from pandas.core.arrays.string_arrow import (
    ArrowStringArray,
    ArrowStringArrayNumpySemantics,
)


# 定义测试函数test_eq_all_na()
def test_eq_all_na():
    # 确保pyarrow库已导入，否则跳过测试
    pytest.importorskip("pyarrow")
    # 创建包含两个pd.NA值的StringDtype类型的Pandas数组a
    a = pd.array([pd.NA, pd.NA], dtype=StringDtype("pyarrow"))
    # 对数组a执行相等比较操作
    result = a == a
    # 创建期望结果，包含两个pd.NA值的Pandas数组，数据类型为"boolean[pyarrow]"
    expected = pd.array([pd.NA, pd.NA], dtype="boolean[pyarrow]")
    # 使用测试工具函数验证扩展数组的相等性
    tm.assert_extension_array_equal(result, expected)


# 定义测试函数test_config，接受三个参数string_storage、request和using_infer_string
def test_config(string_storage, request, using_infer_string):
    # 如果使用infer_string并且string_storage不为"pyarrow_numpy"，则标记为xfail
    if using_infer_string and string_storage != "pyarrow_numpy":
        request.applymarker(pytest.mark.xfail(reason="infer string takes precedence"))
    # 使用指定的string_storage上下文设置Pandas选项
    with pd.option_context("string_storage", string_storage):
        # 断言StringDtype的storage属性等于string_storage
        assert StringDtype().storage == string_storage
        # 创建包含字符串"a"和"b"的Pandas数组result
        result = pd.array(["a", "b"])
        # 断言result数组的数据类型的storage属性等于string_storage
        assert result.dtype.storage == string_storage

    # 使用指定的string_storage创建StringDtype对象dtype
    dtype = StringDtype(string_storage)
    # 创建期望结果，使用dtype构造的Pandas数组，包含字符串"a"和"b"
    expected = dtype.construct_array_type()._from_sequence(["a", "b"], dtype=dtype)
    # 使用测试工具函数验证两个Pandas数组的相等性
    tm.assert_equal(result, expected)


# 定义测试函数test_config_bad_storage_raises
def test_config_bad_storage_raises():
    # 设置错误消息的正则表达式，用于匹配值必须为python或pyarrow的错误信息
    msg = re.escape("Value must be one of python|pyarrow")
    # 使用pytest.raises检查是否抛出值错误，并匹配错误消息
    with pytest.raises(ValueError, match=msg):
        # 设置Pandas选项的string_storage属性为"foo"，引发错误
        pd.options.mode.string_storage = "foo"


# 使用@parametrize装饰器，参数化chunked为True和False
# 使用@parametrize装饰器，参数化array为"numpy"和"pyarrow"
def test_constructor_not_string_type_raises(array, chunked, arrow_string_storage):
    # 导入pyarrow库，如果未导入则跳过测试
    pa = pytest.importorskip("pyarrow")

    # 根据array是否在arrow_string_storage中选择使用pa还是np数组
    array = pa if array in arrow_string_storage else np

    # 创建数组arr，根据array类型选择使用pa或np数组
    arr = array.array([1, 2, 3])
    # 如果chunked为True，处理数组为分块数组
    if chunked:
        # 如果数组为np类型，则跳过chunked不适用于numpy数组
        if array is np:
            pytest.skip("chunked not applicable to numpy array")
        # 否则使用pa.chunked_array处理数组
        arr = pa.chunked_array(arr)
    # 如果数组为np类型，设置错误消息
    if array is np:
        msg = "Unsupported type '<class 'numpy.ndarray'>' for ArrowExtensionArray"
    else:
        # 否则设置错误消息，指示ArrowStringArray需要large_string类型的PyArrow（分块）数组
        msg = re.escape(
            "ArrowStringArray requires a PyArrow (chunked) array of large_string type"
        )
    # 使用pytest.raises检查是否抛出值错误，并匹配错误消息
    with pytest.raises(ValueError, match=msg):
        # 创建ArrowStringArray对象，传入处理后的数组arr
        ArrowStringArray(arr)


# 使用@parametrize装饰器，参数化chunked为True和False
def test_constructor_not_string_type_value_dictionary_raises(chunked):
    # 导入pyarrow库，如果未导入则跳过测试
    pa = pytest.importorskip("pyarrow")

    # 使用pa.array创建数组arr，数据为[1, 2, 3]，类型为字典类型的pa.int32()
    arr = pa.array([1, 2, 3], pa.dictionary(pa.int32(), pa.int32()))
    # 如果chunked为True，处理数组为分块数组
    if chunked:
        # 使用pa.chunked_array处理数组arr
        arr = pa.chunked_array(arr)

    # 设置错误消息，指示ArrowStringArray需要large_string类型的PyArrow（分块）数组
    msg = re.escape(
        "ArrowStringArray requires a PyArrow (chunked) array of large_string type"
    )
    # 使用pytest.raises检查是否抛出值错误，并匹配错误消息
    with pytest.raises(ValueError, match=msg):
        # 创建ArrowStringArray对象，传入处理后的数组arr
        ArrowStringArray(arr)


# 使用@parametrize装饰器，标记为xfail，原因是在arrow中似乎未实现大字符串的字典转换
# 使用@parametrize装饰器，参数化chunked为True和False
def test_constructor_valid_string_type_value_dictionary(chunked):
    # 导入pyarrow库，如果未导入则跳过测试
    pa = pytest.importorskip("pyarrow")

    # 使用pa.array创建数组arr，数据为["1", "2", "3"]，类型为large_string()
    arr = pa.array(["1", "2", "3"], pa.large_string()).dictionary_encode()
    # 如果 chunked 参数为 True，则对数组进行分块处理
    if chunked:
        arr = pa.chunked_array(arr)

    # 将数组 arr 转换为 ArrowStringArray 类型的对象
    arr = ArrowStringArray(arr)

    # 使用断言确保 arr 对象的底层类型是字符串类型
    assert pa.types.is_string(arr._pa_array.type.value_type)
# 定义一个测试函数，用于测试从列表构造对象的行为
def test_constructor_from_list():
    # 导入pytest模块，如果未安装则跳过当前测试（GH#27673）
    pytest.importorskip("pyarrow")
    # 使用pyarrow存储类型创建一个包含单个字符串"E"的Series对象
    result = pd.Series(["E"], dtype=StringDtype(storage="pyarrow"))
    # 断言result的dtype是StringDtype类型
    assert isinstance(result.dtype, StringDtype)
    # 断言result的dtype的storage属性值为"pyarrow"
    assert result.dtype.storage == "pyarrow"


# 定义一个测试函数，用于测试当输入序列类型不正确时抛出异常的情况
def test_from_sequence_wrong_dtype_raises(using_infer_string):
    # 导入pytest模块，如果未安装则跳过当前测试
    pytest.importorskip("pyarrow")
    
    # 在"python"存储模式下，使用ArrowStringArray的_from_sequence方法处理输入序列
    with pd.option_context("string_storage", "python"):
        ArrowStringArray._from_sequence(["a", None, "c"], dtype="string")

    # 在"pyarrow"存储模式下，使用ArrowStringArray的_from_sequence方法处理输入序列
    with pd.option_context("string_storage", "pyarrow"):
        ArrowStringArray._from_sequence(["a", None, "c"], dtype="string")

    # 断言使用"string[python]"作为dtype时会抛出AssertionError异常
    with pytest.raises(AssertionError, match=None):
        ArrowStringArray._from_sequence(["a", None, "c"], dtype="string[python]")

    # 使用"string[pyarrow]"作为dtype，调用ArrowStringArray的_from_sequence方法处理输入序列
    ArrowStringArray._from_sequence(["a", None, "c"], dtype="string[pyarrow]")

    # 如果非using_infer_string状态，则在"python"存储模式下使用StringDtype对象处理输入序列时抛出AssertionError异常
    if not using_infer_string:
        with pytest.raises(AssertionError, match=None):
            with pd.option_context("string_storage", "python"):
                ArrowStringArray._from_sequence(["a", None, "c"], dtype=StringDtype())

    # 在"pyarrow"存储模式下，使用StringDtype对象处理输入序列
    with pd.option_context("string_storage", "pyarrow"):
        ArrowStringArray._from_sequence(["a", None, "c"], dtype=StringDtype())

    # 如果非using_infer_string状态，则使用StringDtype对象处理输入序列时抛出AssertionError异常
    if not using_infer_string:
        with pytest.raises(AssertionError, match=None):
            ArrowStringArray._from_sequence(
                ["a", None, "c"], dtype=StringDtype("python")
            )

    # 使用StringDtype对象处理输入序列，dtype设定为"pyarrow"
    ArrowStringArray._from_sequence(["a", None, "c"], dtype=StringDtype("pyarrow"))

    # 在"python"存储模式下，使用StringArray的_from_sequence方法处理输入序列
    with pd.option_context("string_storage", "python"):
        StringArray._from_sequence(["a", None, "c"], dtype="string")

    # 在"pyarrow"存储模式下，使用StringArray的_from_sequence方法处理输入序列
    with pd.option_context("string_storage", "pyarrow"):
        StringArray._from_sequence(["a", None, "c"], dtype="string")

    # 使用"string[python]"作为dtype时，调用StringArray的_from_sequence方法处理输入序列
    StringArray._from_sequence(["a", None, "c"], dtype="string[python]")

    # 断言使用"string[pyarrow]"作为dtype时会抛出AssertionError异常
    with pytest.raises(AssertionError, match=None):
        StringArray._from_sequence(["a", None, "c"], dtype="string[pyarrow]")

    # 如果非using_infer_string状态，则在"python"存储模式下使用StringDtype对象处理输入序列时抛出AssertionError异常
    if not using_infer_string:
        with pytest.raises(AssertionError, match=None):
            with pd.option_context("string_storage", "python"):
                StringArray._from_sequence(["a", None, "c"], dtype=StringDtype())

    # 如果非using_infer_string状态，则使用StringDtype对象处理输入序列时抛出AssertionError异常
    if not using_infer_string:
        with pytest.raises(AssertionError, match=None):
            with pd.option_context("string_storage", "pyarrow"):
                StringArray._from_sequence(["a", None, "c"], dtype=StringDtype())

    # 使用StringDtype对象处理输入序列，dtype设定为"python"
    StringArray._from_sequence(["a", None, "c"], dtype=StringDtype("python"))

    # 断言使用StringDtype对象处理输入序列时，dtype设定为"pyarrow"会抛出AssertionError异常
    with pytest.raises(AssertionError, match=None):
        StringArray._from_sequence(["a", None, "c"], dtype=StringDtype("pyarrow"))


# 使用td.skip_if_installed装饰器定义一个测试函数，如果pyarrow已安装则跳过该测试
@td.skip_if_installed("pyarrow")
def test_pyarrow_not_installed_raises():
    # 创建匹配字符串msg，用于匹配pytest.raises抛出的ImportError异常信息
    msg = re.escape("pyarrow>=10.0.1 is required for PyArrow backed")

    # 断言创建StringDtype对象时，如果pyarrow未安装会抛出ImportError异常
    with pytest.raises(ImportError, match=msg):
        StringDtype(storage="pyarrow")

    # 断言创建ArrowStringArray对象时，如果pyarrow未安装会抛出ImportError异常
    with pytest.raises(ImportError, match=msg):
        ArrowStringArray([])

    # 断言创建ArrowStringArrayNumpySemantics对象时，如果pyarrow未安装会抛出ImportError异常
    with pytest.raises(ImportError, match=msg):
        ArrowStringArrayNumpySemantics([])
    # 使用 pytest 模块来测试代码，期望引发 ImportError 异常，并且异常消息与变量 msg 匹配
    with pytest.raises(ImportError, match=msg):
        # 调用 ArrowStringArray 类的 _from_sequence 方法，传入包含字符串和 None 值的列表作为参数
        ArrowStringArray._from_sequence(["a", None, "b"])
@pytest.mark.parametrize("multiple_chunks", [False, True])
@pytest.mark.parametrize(
    "key, value, expected",
    [
        (-1, "XX", ["a", "b", "c", "d", "XX"]),  # 使用负数索引，在最后插入新值
        (1, "XX", ["a", "XX", "c", "d", "e"]),  # 使用正数索引替换指定位置的值
        (1, None, ["a", None, "c", "d", "e"]),  # 替换为 None
        (1, pd.NA, ["a", None, "c", "d", "e"]),  # 替换为 pd.NA（缺失值）
        ([1, 3], "XX", ["a", "XX", "c", "XX", "e"]),  # 同时替换多个位置的值
        ([1, 3], ["XX", "YY"], ["a", "XX", "c", "YY", "e"]),  # 使用列表替换多个位置的值
        ([1, 3], ["XX", None], ["a", "XX", "c", None, "e"]),  # 使用 None 替换多个位置的值
        ([1, 3], ["XX", pd.NA], ["a", "XX", "c", None, "e"]),  # 使用 pd.NA 替换多个位置的值
        ([0, -1], ["XX", "YY"], ["XX", "b", "c", "d", "YY"]),  # 替换首尾两个位置的值
        ([-1, 0], ["XX", "YY"], ["YY", "b", "c", "d", "XX"]),  # 替换首尾两个位置的值（倒序）
        (slice(3, None), "XX", ["a", "b", "c", "XX", "XX"]),  # 使用切片替换多个连续位置的值
        (slice(2, 4), ["XX", "YY"], ["a", "b", "XX", "YY", "e"]),  # 使用切片替换多个连续位置的值
        (slice(3, 1, -1), ["XX", "YY"], ["a", "b", "YY", "XX", "e"]),  # 使用反向切片替换多个连续位置的值
        (slice(None), "XX", ["XX", "XX", "XX", "XX", "XX"]),  # 使用切片替换所有位置的值
        ([False, True, False, True, False], ["XX", "YY"], ["a", "XX", "c", "YY", "e"]),  # 使用布尔索引替换多个位置的值
    ],
)
def test_setitem(multiple_chunks, key, value, expected):
    pa = pytest.importorskip("pyarrow")

    result = pa.array(list("abcde"))  # 创建 pyarrow 数组对象
    expected = pa.array(expected)  # 创建期望的 pyarrow 数组对象

    if multiple_chunks:
        result = pa.chunked_array([result[:3], result[3:]])  # 如果使用多块数组，切分成两部分
        expected = pa.chunked_array([expected[:3], expected[3:]])  # 期望结果也进行切分

    result = ArrowStringArray(result)  # 使用自定义的 ArrowStringArray 封装结果
    expected = ArrowStringArray(expected)  # 使用自定义的 ArrowStringArray 封装期望结果

    result[key] = value  # 执行索引赋值操作
    tm.assert_equal(result, expected)  # 断言结果与期望一致


def test_setitem_invalid_indexer_raises():
    pa = pytest.importorskip("pyarrow")

    arr = ArrowStringArray(pa.array(list("abcde")))  # 创建包含字符串的 Arrow 数组对象

    with tm.external_error_raised(IndexError):  # 检查是否引发 IndexError 异常
        arr[5] = "foo"

    with tm.external_error_raised(IndexError):
        arr[-6] = "foo"

    with tm.external_error_raised(IndexError):
        arr[[0, 5]] = "foo"

    with tm.external_error_raised(IndexError):
        arr[[0, -6]] = "foo"

    with tm.external_error_raised(IndexError):
        arr[[True, True, False]] = "foo"

    with tm.external_error_raised(ValueError):  # 检查是否引发 ValueError 异常
        arr[[0, 1]] = ["foo", "bar", "baz"]


@pytest.mark.parametrize("dtype", ["string[pyarrow]", "string[pyarrow_numpy]"])
def test_pickle_roundtrip(dtype):
    # GH 42600
    pytest.importorskip("pyarrow")
    expected = pd.Series(range(10), dtype=dtype)  # 创建具有指定 dtype 的 Pandas Series
    expected_sliced = expected.head(2)  # 获取 Series 的前两个元素
    full_pickled = pickle.dumps(expected)  # 对整个 Series 进行序列化
    sliced_pickled = pickle.dumps(expected_sliced)  # 对部分 Series 进行序列化

    assert len(full_pickled) > len(sliced_pickled)  # 检查完整序列化的长度是否大于部分序列化的长度

    result = pickle.loads(full_pickled)  # 反序列化整个 Series
    tm.assert_series_equal(result, expected)  # 断言反序列化结果与原始 Series 一致

    result_sliced = pickle.loads(sliced_pickled)  # 反序列化部分 Series
    tm.assert_series_equal(result_sliced, expected_sliced)  # 断言反序列化结果与部分 Series 一致


def test_string_dtype_error_message():
    # GH#55051
    pytest.importorskip("pyarrow")
    msg = "Storage must be 'python', 'pyarrow' or 'pyarrow_numpy'."
    with pytest.raises(ValueError, match=msg):  # 检查是否引发特定错误消息的 ValueError 异常
        StringDtype("bla")
```