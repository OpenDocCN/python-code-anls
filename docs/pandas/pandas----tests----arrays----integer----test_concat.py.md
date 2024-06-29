# `D:\src\scipysrc\pandas\pandas\tests\arrays\integer\test_concat.py`

```
# 导入必要的库
import numpy as np
import pytest

# 导入 pandas 库及其测试模块
import pandas as pd
import pandas._testing as tm

# 定义测试函数，使用 pytest 的参数化装饰器指定多组测试参数
@pytest.mark.parametrize(
    "to_concat_dtypes, result_dtype",
    [
        (["Int64", "Int64"], "Int64"),
        (["UInt64", "UInt64"], "UInt64"),
        (["Int8", "Int8"], "Int8"),
        (["Int8", "Int16"], "Int16"),
        (["UInt8", "Int8"], "Int16"),
        (["Int32", "UInt32"], "Int64"),
        (["Int64", "UInt64"], "Float64"),
        (["Int64", "boolean"], "object"),
        (["UInt8", "boolean"], "object"),
    ],
)
def test_concat_series(to_concat_dtypes, result_dtype):
    # 创建由不同数据类型的 Series 组成的列表，并进行合并
    result = pd.concat([pd.Series([0, 1, pd.NA], dtype=t) for t in to_concat_dtypes])
    # 创建预期结果：两个相同的 Series 合并后转换为指定的结果数据类型
    expected = pd.concat([pd.Series([0, 1, pd.NA], dtype=object)] * 2).astype(
        result_dtype
    )
    # 使用测试模块中的函数检查两个 Series 是否相等
    tm.assert_series_equal(result, expected)

    # 顺序对调输入顺序，结果应该一致
    result = pd.concat(
        [pd.Series([0, 1, pd.NA], dtype=t) for t in to_concat_dtypes[::-1]]
    )
    expected = pd.concat([pd.Series([0, 1, pd.NA], dtype=object)] * 2).astype(
        result_dtype
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "to_concat_dtypes, result_dtype",
    [
        (["Int64", "int64"], "Int64"),
        (["UInt64", "uint64"], "UInt64"),
        (["Int8", "int8"], "Int8"),
        (["Int8", "int16"], "Int16"),
        (["UInt8", "int8"], "Int16"),
        (["Int32", "uint32"], "Int64"),
        (["Int64", "uint64"], "Float64"),
        (["Int64", "bool"], "object"),
        (["UInt8", "bool"], "object"),
    ],
)
def test_concat_series_with_numpy(to_concat_dtypes, result_dtype):
    # 创建两个不同的 Series，分别从不同的数据类型数组转换而来，然后进行合并
    s1 = pd.Series([0, 1, pd.NA], dtype=to_concat_dtypes[0])
    s2 = pd.Series(np.array([0, 1], dtype=to_concat_dtypes[1]))
    result = pd.concat([s1, s2], ignore_index=True)
    # 创建预期结果：两个 Series 合并后转换为指定的结果数据类型
    expected = pd.Series([0, 1, pd.NA, 0, 1], dtype=object).astype(result_dtype)
    # 使用测试模块中的函数检查两个 Series 是否相等
    tm.assert_series_equal(result, expected)

    # 顺序对调输入顺序，结果应该一致
    result = pd.concat([s2, s1], ignore_index=True)
    expected = pd.Series([0, 1, 0, 1, pd.NA], dtype=object).astype(result_dtype)
    tm.assert_series_equal(result, expected)
```