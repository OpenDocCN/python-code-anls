# `D:\src\scipysrc\pandas\pandas\io\_util.py`

```
# 从未来模块导入注释，以便支持类型提示
from __future__ import annotations

# 导入类型检查模块，用于检查类型是否可用
from typing import TYPE_CHECKING

# 导入可选依赖项导入功能
from pandas.compat._optional import import_optional_dependency

# 导入 pandas 库并使用别名 pd
import pandas as pd

# 如果正在进行类型检查
if TYPE_CHECKING:
    # 导入 Callable 类型
    from collections.abc import Callable


# 定义函数 _arrow_dtype_mapping，返回一个字典，映射 pyarrow 类型到 pandas 对应的类型
def _arrow_dtype_mapping() -> dict:
    # 导入 pyarrow 作为 pa
    pa = import_optional_dependency("pyarrow")
    
    # 返回字典，将 pyarrow 的整数类型映射到 pandas 的整数类型
    return {
        pa.int8(): pd.Int8Dtype(),
        pa.int16(): pd.Int16Dtype(),
        pa.int32(): pd.Int32Dtype(),
        pa.int64(): pd.Int64Dtype(),
        pa.uint8(): pd.UInt8Dtype(),
        pa.uint16(): pd.UInt16Dtype(),
        pa.uint32(): pd.UInt32Dtype(),
        pa.uint64(): pd.UInt64Dtype(),
        pa.bool_(): pd.BooleanDtype(),
        pa.string(): pd.StringDtype(),
        pa.float32(): pd.Float32Dtype(),
        pa.float64(): pd.Float64Dtype(),
    }


# 定义函数 arrow_string_types_mapper，返回一个函数，将 pyarrow 字符串类型映射到 pandas 字符串类型
def arrow_string_types_mapper() -> Callable:
    # 导入 pyarrow 作为 pa
    pa = import_optional_dependency("pyarrow")

    # 返回一个字典，将 pyarrow 的字符串类型映射到 pandas 的字符串类型，指定存储方式为 pyarrow_numpy
    return {
        pa.string(): pd.StringDtype(storage="pyarrow_numpy"),
        pa.large_string(): pd.StringDtype(storage="pyarrow_numpy"),
    }.get
```