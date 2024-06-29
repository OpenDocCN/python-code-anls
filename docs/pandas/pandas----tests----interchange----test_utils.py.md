# `D:\src\scipysrc\pandas\pandas\tests\interchange\test_utils.py`

```
# 导入必要的库
import numpy as np
import pytest

# 导入 pandas 库及相关的工具函数
import pandas as pd
from pandas.core.interchange.utils import dtype_to_arrow_c_fmt

# 通过注释提醒待处理的问题，此处需要使用 ArrowSchema 获取参考的 C 字符串。
# 目前无法直接从 Python 访问持有类型格式字符串的 ArrowSchema。
# 唯一的访问方法是将结构导出到 C 指针中，参见 DataType._export_to_c() 方法，
# 该方法定义在 https://github.com/apache/arrow/blob/master/python/pyarrow/types.pxi 中。

# 使用 pytest 的 parametrize 装饰器来定义多组测试参数
@pytest.mark.parametrize(
    "pandas_dtype, c_string",
    [
        (np.dtype("bool"), "b"),
        (np.dtype("int8"), "c"),
        (np.dtype("uint8"), "C"),
        (np.dtype("int16"), "s"),
        (np.dtype("uint16"), "S"),
        (np.dtype("int32"), "i"),
        (np.dtype("uint32"), "I"),
        (np.dtype("int64"), "l"),
        (np.dtype("uint64"), "L"),
        (np.dtype("float16"), "e"),
        (np.dtype("float32"), "f"),
        (np.dtype("float64"), "g"),
        (pd.Series(["a"]).dtype, "u"),
        (
            pd.Series([0]).astype("datetime64[ns]").dtype,
            "tsn:",
        ),
        (pd.CategoricalDtype(["a"]), "l"),
        (np.dtype("O"), "u"),
    ],
)
# 定义测试函数 test_dtype_to_arrow_c_fmt，测试 dtype_to_arrow_c_fmt 函数的功能
def test_dtype_to_arrow_c_fmt(pandas_dtype, c_string):  # PR01
    """Test ``dtype_to_arrow_c_fmt`` utility function."""
    assert dtype_to_arrow_c_fmt(pandas_dtype) == c_string

# 再次使用 pytest 的 parametrize 装饰器定义更多的测试参数
@pytest.mark.parametrize(
    "pa_dtype, args_kwargs, c_string",
    [
        ["null", {}, "n"],
        ["bool_", {}, "b"],
        ["uint8", {}, "C"],
        ["uint16", {}, "S"],
        ["uint32", {}, "I"],
        ["uint64", {}, "L"],
        ["int8", {}, "c"],
        ["int16", {}, "S"],
        ["int32", {}, "i"],
        ["int64", {}, "l"],
        ["float16", {}, "e"],
        ["float32", {}, "f"],
        ["float64", {}, "g"],
        ["string", {}, "u"],
        ["binary", {}, "z"],
        ["time32", ("s",), "tts"],
        ["time32", ("ms",), "ttm"],
        ["time64", ("us",), "ttu"],
        ["time64", ("ns",), "ttn"],
        ["date32", {}, "tdD"],
        ["date64", {}, "tdm"],
        ["timestamp", {"unit": "s"}, "tss:"],
        ["timestamp", {"unit": "ms"}, "tsm:"],
        ["timestamp", {"unit": "us"}, "tsu:"],
        ["timestamp", {"unit": "ns"}, "tsn:"],
        ["timestamp", {"unit": "ns", "tz": "UTC"}, "tsn:UTC"],
        ["duration", ("s",), "tDs"],
        ["duration", ("ms",), "tDm"],
        ["duration", ("us",), "tDu"],
        ["duration", ("ns",), "tDn"],
        ["decimal128", {"precision": 4, "scale": 2}, "d:4,2"],
    ],
)
# 定义测试函数 test_dtype_to_arrow_c_fmt_arrowdtype，测试 dtype_to_arrow_c_fmt 函数对 ArrowDtype 的支持
def test_dtype_to_arrow_c_fmt_arrowdtype(pa_dtype, args_kwargs, c_string):
    # GH 52323: 通过 pytest 的 importorskip 方法导入 pyarrow 库
    pa = pytest.importorskip("pyarrow")
    # 根据参数 args_kwargs 实例化 pyarrow 中的数据类型 pa_type
    if not args_kwargs:
        pa_type = getattr(pa, pa_dtype)()
    elif isinstance(args_kwargs, tuple):
        pa_type = getattr(pa, pa_dtype)(*args_kwargs)
    else:
        pa_type = getattr(pa, pa_dtype)(**args_kwargs)
    # 将 pa_type 转换为 pandas 的 ArrowDtype 类型
    arrow_type = pd.ArrowDtype(pa_type)
    # 断言 dtype_to_arrow_c_fmt 函数的返回值是否与预期的 c_string 相等
    assert dtype_to_arrow_c_fmt(arrow_type) == c_string
```