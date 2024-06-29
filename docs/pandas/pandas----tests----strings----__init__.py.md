# `D:\src\scipysrc\pandas\pandas\tests\strings\__init__.py`

```
# 导入 numpy 库，使用 'np' 作为别名
import numpy as np

# 导入 pandas 库，使用 'pd' 作为别名
import pandas as pd

# 定义一个元组 'object_pyarrow_numpy'，包含两个元素："object" 和 "string[pyarrow_numpy]"
object_pyarrow_numpy = ("object", "string[pyarrow_numpy]")

# 定义一个函数 '_convert_na_value'，接受两个参数：'ser' 表示 pandas Series 对象，'expected' 表示期望的填充值
def _convert_na_value(ser, expected):
    # 检查 'ser' 的数据类型是否不是 'object'
    if ser.dtype != object:
        # 如果 'ser' 的数据类型的存储类型为 "pyarrow_numpy"
        if ser.dtype.storage == "pyarrow_numpy":
            # 将 'expected' 中的缺失值填充为 np.nan
            expected = expected.fillna(np.nan)
        else:
            # 否则，使用 'pd.NA' 对 'expected' 进行填充
            # GH#18463 表示 GitHub 上的 issue 编号
            expected = expected.fillna(pd.NA)
    # 返回填充后的 'expected' 值
    return expected
```