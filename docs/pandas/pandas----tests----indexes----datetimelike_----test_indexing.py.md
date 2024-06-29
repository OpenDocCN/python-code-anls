# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimelike_\test_indexing.py`

```
import numpy as np
import pytest

# 导入 pandas 库，并从中导入特定的模块和类
import pandas as pd
from pandas import (
    DatetimeIndex,
    Index,
)
# 导入 pandas 的测试工具模块
import pandas._testing as tm

# 定义一组日期时间相关的数据类型
dtlike_dtypes = [
    np.dtype("timedelta64[ns]"),
    np.dtype("datetime64[ns]"),
    pd.DatetimeTZDtype("ns", "Asia/Tokyo"),
    pd.PeriodDtype("ns"),
]

# 使用 pytest 的 parametrize 装饰器定义测试参数化
@pytest.mark.parametrize("ldtype", dtlike_dtypes)
@pytest.mark.parametrize("rdtype", dtlike_dtypes)
def test_get_indexer_non_unique_wrong_dtype(ldtype, rdtype):
    # 创建一个包含重复值的数组
    vals = np.tile(3600 * 10**9 * np.arange(3, dtype=np.int64), 2)

    # 定义一个构造函数，根据不同的数据类型创建不同类型的索引对象
    def construct(dtype):
        if dtype is dtlike_dtypes[-1]:
            # 如果数据类型是 PeriodDtype，则将整数转换为字符串
            return DatetimeIndex(vals).astype(dtype)
        # 否则创建一个普通的索引对象
        return Index(vals, dtype=dtype)

    # 分别使用 ldtype 和 rdtype 构造左右两个索引对象
    left = construct(ldtype)
    right = construct(rdtype)

    # 获取左索引相对于右索引的非唯一匹配索引
    result = left.get_indexer_non_unique(right)

    if ldtype is rdtype:
        # 如果左右数据类型相同，则期望得到的结果
        ex1 = np.array([0, 3, 1, 4, 2, 5] * 2, dtype=np.intp)
        ex2 = np.array([], dtype=np.intp)
        # 断言结果与期望值相等
        tm.assert_numpy_array_equal(result[0], ex1)
        tm.assert_numpy_array_equal(result[1], ex2)
    else:
        # 如果左右数据类型不同，则期望得到的结果
        no_matches = np.array([-1] * 6, dtype=np.intp)
        missing = np.arange(6, dtype=np.intp)
        # 断言结果与期望值相等
        tm.assert_numpy_array_equal(result[0], no_matches)
        tm.assert_numpy_array_equal(result[1], missing)
```