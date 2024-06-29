# `D:\src\scipysrc\pandas\pandas\tests\strings\test_api.py`

```
# 引入弱引用模块，用于避免循环引用问题
import weakref

# 引入 NumPy 库，并使用别名 np
import numpy as np

# 引入 Pytest 测试框架
import pytest

# 从 Pandas 库中引入特定的子模块和类
from pandas import (
    CategoricalDtype,  # 引入分类数据类型
    DataFrame,          # 引入数据框类
    Index,              # 引入索引类
    MultiIndex,         # 引入多重索引类
    Series,             # 引入序列类
    _testing as tm,     # 引入测试模块并使用别名 tm
    option_context,     # 引入选项上下文管理器
)

# 从 Pandas 核心字符串访问器模块中引入 StringMethods 类
from pandas.core.strings.accessor import StringMethods

# 从 pandas/conftest.py 中选择的一部分测试数据
_any_allowed_skipna_inferred_dtype = [
    ("string", ["a", np.nan, "c"]),    # 字符串类型及其值
    ("bytes", [b"a", np.nan, b"c"]),   # 字节类型及其值
    ("empty", [np.nan, np.nan, np.nan]),  # 空类型及其值
    ("empty", []),                     # 空类型，空列表
    ("mixed-integer", ["a", np.nan, 2]),  # 混合整数类型及其值
]

# 从 _any_allowed_skipna_inferred_dtype 中获取 ids
ids, _ = zip(*_any_allowed_skipna_inferred_dtype)  # 使用推断类型作为 id


@pytest.fixture(params=_any_allowed_skipna_inferred_dtype, ids=ids)
def any_allowed_skipna_inferred_dtype(request):
    """
    StringMethods.__init__ 可接受的所有（推断的）数据类型的 Fixture

    涵盖的推断类型有：
    * 'string'
    * 'empty'
    * 'bytes'
    * 'mixed'
    * 'mixed-integer'

    返回
    -------
    inferred_dtype : str
        来自 _libs.lib.infer_dtype 推断出的数据类型的字符串
    values : np.ndarray
        一个对象 dtype 的数组，其中包含会被推断为 `inferred_dtype` 的值

    示例
    --------
    >>> from pandas._libs import lib
    >>>
    >>> def test_something(any_allowed_skipna_inferred_dtype):
    ...     inferred_dtype, values = any_allowed_skipna_inferred_dtype
    ...     # 将通过测试
    ...     assert lib.infer_dtype(values, skipna=True) == inferred_dtype
    ...
    ...     # 使用 .str 访问器的构造函数也将通过测试
    ...     Series(values).str
    """
    inferred_dtype, values = request.param
    values = np.array(values, dtype=object)  # 使用对象 dtype 避免类型转换

    # 推断的正确性在 tests/dtypes/test_inference.py 中进行测试
    return inferred_dtype, values


def test_api(any_string_dtype):
    # GH 6106, GH 9322
    assert Series.str is StringMethods
    assert isinstance(Series([""], dtype=any_string_dtype).str, StringMethods)


def test_no_circular_reference(any_string_dtype):
    # GH 47667
    ser = Series([""], dtype=any_string_dtype)
    ref = weakref.ref(ser)
    ser.str  # 用于缓存并导致循环引用
    del ser
    assert ref() is None


def test_api_mi_raises():
    # GH 23679
    mi = MultiIndex.from_arrays([["a", "b", "c"]])
    msg = "Can only use .str accessor with Index, not MultiIndex"
    with pytest.raises(AttributeError, match=msg):
        mi.str
    assert not hasattr(mi, "str")


@pytest.mark.parametrize("dtype", [object, "category"])
def test_api_per_dtype(index_or_series, dtype, any_skipna_inferred_dtype):
    # parametrize 的实例
    box = index_or_series
    inferred_dtype, values = any_skipna_inferred_dtype

    t = box(values, dtype=dtype)  # 显式指定 dtype 避免类型转换

    types_passing_constructor = [
        "string",
        "unicode",
        "empty",
        "bytes",
        "mixed",
        "mixed-integer",
    ]
    # 如果推断出的数据类型在允许使用构造函数的类型列表中
    if inferred_dtype in types_passing_constructor:
        # GH 6106
        # 断言确保 t.str 是 StringMethods 类型的实例
        assert isinstance(t.str, StringMethods)
    else:
        # 否则，对于 GH 9184, GH 23011, GH 23163
        # 准备错误消息，指出只能在字符串值上使用 .str 访问器
        msg = "Can only use .str accessor with string values.*"
        # 使用 pytest 来检查是否会抛出 AttributeError，且错误消息匹配预期的消息
        with pytest.raises(AttributeError, match=msg):
            t.str
        # 断言 t 对象没有属性 "str"
        assert not hasattr(t, "str")
@pytest.mark.parametrize("dtype", [object, "category"])
# 使用 pytest 的 parametrize 装饰器，对 dtype 参数进行参数化测试，分别测试 object 和 "category" 两种类型
def test_api_per_method(
    index_or_series,
    dtype,
    any_allowed_skipna_inferred_dtype,
    any_string_method,
    request,
):
    # 这个测试不检查不同方法的正确性，
    # 只验证这些方法在指定的（推断出的）数据类型上工作，并在其他类型上引发异常

    box = index_or_series
    # 将 index_or_series 参数赋值给 box

    # 获取任意一个参数化的 fixture 的实例
    inferred_dtype, values = any_allowed_skipna_inferred_dtype
    method_name, args, kwargs = any_string_method

    reason = None
    # 初始化 reason 变量为 None
    if box is Index and values.size == 0:
        # 如果 box 是 Index 类型且 values 的大小为 0
        if method_name in ["partition", "rpartition"] and kwargs.get("expand", True):
            # 如果 method_name 是 "partition" 或 "rpartition"，且 kwargs 中的 "expand" 参数为 True
            raises = TypeError
            reason = "Method cannot deal with empty Index"
            # raises 设置为 TypeError，reason 设置为相关提示信息
        elif method_name == "split" and kwargs.get("expand", None):
            # 如果 method_name 是 "split"，且 kwargs 中的 "expand" 参数不为 None
            raises = TypeError
            reason = "Split fails on empty Series when expand=True"
            # raises 设置为 TypeError，reason 设置为相关提示信息
        elif method_name == "get_dummies":
            # 如果 method_name 是 "get_dummies"
            raises = ValueError
            reason = "Need to fortify get_dummies corner cases"
            # raises 设置为 ValueError，reason 设置为相关提示信息

    elif (
        box is Index
        and inferred_dtype == "empty"
        and dtype == object
        and method_name == "get_dummies"
    ):
        # 否则如果 box 是 Index 类型，inferred_dtype 是 "empty"，dtype 是 object，method_name 是 "get_dummies"
        raises = ValueError
        reason = "Need to fortify get_dummies corner cases"
        # raises 设置为 ValueError，reason 设置为相关提示信息

    if reason is not None:
        mark = pytest.mark.xfail(raises=raises, reason=reason)
        request.applymarker(mark)
        # 如果 reason 不为 None，则使用 pytest.mark.xfail 设置测试为预期失败，并添加相关原因

    t = box(values, dtype=dtype)  # explicit dtype to avoid casting
    # 使用 box 构造一个对象 t，指定 dtype 避免类型转换

    method = getattr(t.str, method_name)
    # 获取 t 对象的 str 属性中的 method_name 方法

    bytes_allowed = method_name in ["decode", "get", "len", "slice"]
    # 检查 method_name 是否在 ["decode", "get", "len", "slice"] 中，返回布尔值

    # 截至 v0.23.4，除 'cat' 外的所有方法在允许的数据类型上非常宽松，
    # 对于导致错误的条目仅返回 NaN。可以通过 str 访问器的 'errors' 关键字参数进行更改，
    # 参见 GH 13877
    mixed_allowed = method_name not in ["cat"]
    # 检查 method_name 是否不在 ["cat"] 中，返回布尔值

    allowed_types = (
        ["string", "unicode", "empty"]
        + ["bytes"] * bytes_allowed
        + ["mixed", "mixed-integer"] * mixed_allowed
    )
    # 根据 bytes_allowed 和 mixed_allowed 构造允许的数据类型列表

    if inferred_dtype in allowed_types:
        # 如果 inferred_dtype 在允许的类型列表中
        # xref GH 23555, GH 23556
        with option_context("future.no_silent_downcasting", True):
            method(*args, **kwargs)  # works!
            # 调用 method 方法，传入 args 和 kwargs 参数
    else:
        # 否则
        # GH 23011, GH 23163
        msg = (
            f"Cannot use .str.{method_name} with values of "
            f"inferred dtype {inferred_dtype!r}."
        )
        # 构造错误消息字符串
        with pytest.raises(TypeError, match=msg):
            method(*args, **kwargs)
            # 使用 pytest.raises 断言捕获 TypeError 异常，并匹配错误消息 msg，调用 method 方法，传入 args 和 kwargs 参数


def test_api_for_categorical(any_string_method, any_string_dtype):
    # https://github.com/pandas-dev/pandas/issues/10661
    # 创建一个 Series 对象 s，使用 any_string_dtype 作为数据类型
    s = Series(list("aabb"), dtype=any_string_dtype)
    s = s + " " + s
    # 将 s 的每个元素加上空格
    c = s.astype("category")
    # 将 s 转换为 category 类型的对象 c
    c = c.astype(CategoricalDtype(c.dtype.categories.astype("object")))
    # 再次转换 c 的数据类型为 CategoricalDtype，类别为 object
    assert isinstance(c.str, StringMethods)
    # 断言 c.str 是 StringMethods 类型的实例

    method_name, args, kwargs = any_string_method
    # 获取 any_string_method 的三个元素

    result = getattr(c.str, method_name)(*args, **kwargs)
    # 获取 c.str 中的 method_name 方法，并传入 args 和 kwargs 参数调用
    # 使用 getattr 函数根据 method_name 调用 s 对象的 str 属性，并将 s 转换为对象类型为 "object"
    # 然后调用其返回的对象的 method_name 方法，并传入 *args 和 **kwargs 参数
    expected = getattr(s.astype("object").str, method_name)(*args, **kwargs)

    # 检查 result 的类型，如果是 DataFrame，则使用 tm.assert_frame_equal 进行比较
    # 如果是 Series，则使用 tm.assert_series_equal 进行比较
    # 否则，假定 result 和 expected 应该相等，否则触发断言错误
    if isinstance(result, DataFrame):
        tm.assert_frame_equal(result, expected)
    elif isinstance(result, Series):
        tm.assert_series_equal(result, expected)
    else:
        # str.cat(others=None) 返回一个字符串，例如
        assert result == expected
```