# `D:\src\scipysrc\pandas\pandas\tests\copy_view\test_copy_deprecation.py`

```
import pytest  # 导入 pytest 模块

import pandas as pd  # 导入 pandas 库，并将其命名为 pd
from pandas import (  # 从 pandas 中导入 concat 和 merge 函数
    concat,
    merge,
)
import pandas._testing as tm  # 导入 pandas 内部测试模块

@pytest.mark.parametrize(  # 使用 pytest 的 parametrize 装饰器定义参数化测试
    "meth, kwargs",  # 参数列表包括 meth 和 kwargs
    [  # 参数化的测试数据列表开始
        ("truncate", {}),  # 调用 truncate 方法，不传递额外参数
        ("tz_convert", {"tz": "UTC"}),  # 调用 tz_convert 方法，传递参数 {"tz": "UTC"}
        ("tz_localize", {"tz": "UTC"}),  # 调用 tz_localize 方法，传递参数 {"tz": "UTC"}
        ("infer_objects", {}),  # 调用 infer_objects 方法，不传递额外参数
        ("astype", {"dtype": "float64"}),  # 调用 astype 方法，传递参数 {"dtype": "float64"}
        ("reindex", {"index": [2, 0, 1]}),  # 调用 reindex 方法，传递参数 {"index": [2, 0, 1]}
        ("transpose", {}),  # 调用 transpose 方法，不传递额外参数
        ("set_axis", {"labels": [1, 2, 3]}),  # 调用 set_axis 方法，传递参数 {"labels": [1, 2, 3]}
        ("rename", {"index": {1: 2}}),  # 调用 rename 方法，传递参数 {"index": {1: 2}}
        ("set_flags", {}),  # 调用 set_flags 方法，不传递额外参数
        ("to_period", {}),  # 调用 to_period 方法，不传递额外参数
        ("to_timestamp", {}),  # 调用 to_timestamp 方法，不传递额外参数
        ("swaplevel", {"i": 0, "j": 1}),  # 调用 swaplevel 方法，传递参数 {"i": 0, "j": 1}
    ],  # 参数化的测试数据列表结束
)
def test_copy_deprecation(meth, kwargs):  # 定义测试函数 test_copy_deprecation，接受 meth 和 kwargs 作为参数
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": 1})  # 创建一个 DataFrame 对象 df

    if meth in ("tz_convert", "tz_localize", "to_period"):  # 如果 meth 是 "tz_convert", "tz_localize" 或 "to_period"
        tz = None if meth in ("tz_localize", "to_period") else "US/Eastern"  # 如果是 "tz_localize" 或 "to_period" 则 tz 为 None，否则为 "US/Eastern"
        df.index = pd.date_range("2020-01-01", freq="D", periods=len(df), tz=tz)  # 重新设置 DataFrame 的索引
    elif meth == "to_timestamp":  # 如果 meth 是 "to_timestamp"
        df.index = pd.period_range("2020-01-01", freq="D", periods=len(df))  # 使用 period_range 设置 DataFrame 的索引
    elif meth == "swaplevel":  # 如果 meth 是 "swaplevel"
        df = df.set_index(["b", "c"])  # 使用 set_index 方法重新设置 DataFrame 的索引

    if meth != "swaplevel":  # 如果 meth 不是 "swaplevel"
        with tm.assert_produces_warning(DeprecationWarning, match="copy"):  # 使用 assert_produces_warning 检查是否产生 DeprecationWarning，匹配 "copy"
            getattr(df, meth)(copy=False, **kwargs)  # 动态调用 DataFrame 对象 df 的 meth 方法，传递参数 copy=False 和 kwargs

    if meth != "transpose":  # 如果 meth 不是 "transpose"
        with tm.assert_produces_warning(DeprecationWarning, match="copy"):  # 使用 assert_produces_warning 检查是否产生 DeprecationWarning，匹配 "copy"
            getattr(df.a, meth)(copy=False, **kwargs)  # 动态调用 Series 对象 df.a 的 meth 方法，传递参数 copy=False 和 kwargs


def test_copy_deprecation_reindex_like_align():  # 定义测试函数 test_copy_deprecation_reindex_like_align
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})  # 创建一个 DataFrame 对象 df

    # 某种情况下堆栈级别检查不正确
    with tm.assert_produces_warning(  # 使用 assert_produces_warning 检查是否产生 DeprecationWarning，禁用堆栈级别检查
        DeprecationWarning, match="copy", check_stacklevel=False
    ):
        df.reindex_like(df, copy=False)  # 调用 DataFrame 的 reindex_like 方法，传递参数 copy=False

    with tm.assert_produces_warning(  # 使用 assert_produces_warning 检查是否产生 DeprecationWarning，禁用堆栈级别检查
        DeprecationWarning, match="copy", check_stacklevel=False
    ):
        df.a.reindex_like(df.a, copy=False)  # 调用 Series 对象 df.a 的 reindex_like 方法，传递参数 copy=False

    with tm.assert_produces_warning(  # 使用 assert_produces_warning 检查是否产生 DeprecationWarning，禁用堆栈级别检查
        DeprecationWarning, match="copy", check_stacklevel=False
    ):
        df.align(df, copy=False)  # 调用 DataFrame 的 align 方法，传递参数 copy=False

    with tm.assert_produces_warning(  # 使用 assert_produces_warning 检查是否产生 DeprecationWarning，禁用堆栈级别检查
        DeprecationWarning, match="copy", check_stacklevel=False
    ):
        df.a.align(df.a, copy=False)  # 调用 Series 对象 df.a 的 align 方法，传递参数 copy=False


def test_copy_deprecation_merge_concat():  # 定义测试函数 test_copy_deprecation_merge_concat
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})  # 创建一个 DataFrame 对象 df

    with tm.assert_produces_warning(  # 使用 assert_produces_warning 检查是否产生 DeprecationWarning，禁用堆栈级别检查
        DeprecationWarning, match="copy", check_stacklevel=False
    ):
        df.merge(df, copy=False)  # 调用 DataFrame 的 merge 方法，传递参数 copy=False

    with tm.assert_produces_warning(  # 使用 assert_produces_warning 检查是否产生 DeprecationWarning，禁用堆栈级别检查
        DeprecationWarning, match="copy", check_stacklevel=False
    ):
        merge(df, df, copy=False)  # 调用 pandas 的 merge 函数，传递参数 copy=False

    with tm.assert_produces_warning(  # 使用 assert_produces_warning 检查是否产生 DeprecationWarning，禁用堆栈级别检查
        DeprecationWarning, match="copy", check_stacklevel=False
    ):
        concat([df, df], copy=False)  # 调用 pandas 的 concat 函数，传递参数 copy=False
```