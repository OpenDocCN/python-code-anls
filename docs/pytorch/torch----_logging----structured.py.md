# `.\pytorch\torch\_logging\structured.py`

```py
"""
Utilities for converting data types into structured JSON for dumping.
"""

# 导入 traceback 模块用于处理堆栈跟踪信息
import traceback
# 导入 Dict 和 Sequence 用于类型提示
from typing import Dict, Sequence
# 导入 torch._logging._internal 用于日志记录
import torch._logging._internal

# 创建全局变量 INTERN_TABLE，用于字符串的国际化映射
INTERN_TABLE: Dict[str, int] = {}

# 定义函数 intern_string，用于将字符串 s 映射为整数
def intern_string(s: str) -> int:
    # 检查字符串 s 是否在 INTERN_TABLE 中
    r = INTERN_TABLE.get(s, None)
    # 如果 s 不在 INTERN_TABLE 中，则将其添加，并记录日志
    if r is None:
        r = len(INTERN_TABLE)
        INTERN_TABLE[s] = r
        torch._logging._internal.trace_structured(
            "str", lambda: (s, r), suppress_context=True
        )
    return r

# 定义函数 from_traceback，用于将 traceback.FrameSummary 转换为对象列表
def from_traceback(tb: Sequence[traceback.FrameSummary]) -> object:
    r = []
    # 遍历 traceback 中的每个帧信息
    for frame in tb:
        # 将每个帧信息转换为字典格式，用于构建 JSON 结构
        r.append(
            {
                "line": frame.lineno,          # 行号
                "name": frame.name,            # 函数名
                "filename": intern_string(frame.filename),  # 文件名，通过 intern_string 进行国际化映射
            }
        )
    return r
```