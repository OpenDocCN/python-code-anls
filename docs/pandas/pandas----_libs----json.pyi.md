# `D:\src\scipysrc\pandas\pandas\_libs\json.pyi`

```
# 导入需要的模块和类型定义
from typing import (
    Any,
    Callable,
)

# 定义函数 ujson_dumps，用于将对象转换为 JSON 格式的字符串
def ujson_dumps(
    obj: Any,  # 参数 obj 表示要转换的对象，可以是任意类型
    ensure_ascii: bool = ...,  # 控制是否使用 ASCII 编码
    double_precision: int = ...,  # 控制浮点数精度
    indent: int = ...,  # 控制输出的缩进空格数
    orient: str = ...,  # 控制 JSON 对象的方向
    date_unit: str = ...,  # 控制日期的单位
    iso_dates: bool = ...,  # 控制是否使用 ISO 日期格式
    default_handler: None
    | Callable[[Any], str | float | bool | list | dict | None] = ...,  # 默认处理程序，可以是函数或 None
) -> str:  # 返回一个 JSON 格式的字符串
    ...

# 定义函数 ujson_loads，用于将 JSON 格式的字符串转换为 Python 对象
def ujson_loads(
    s: str,  # 参数 s 表示要解析的 JSON 字符串
    precise_float: bool = ...,  # 控制是否使用精确的浮点数解析
    numpy: bool = ...,  # 控制是否将特定类型解析为 NumPy 对象
    dtype: None = ...,  # 保留参数，未使用
    labelled: bool = ...,  # 控制是否解析为带标签的 JSON 字符串
) -> Any:  # 返回解析后的 Python 对象，可以是任意类型
    ...
```