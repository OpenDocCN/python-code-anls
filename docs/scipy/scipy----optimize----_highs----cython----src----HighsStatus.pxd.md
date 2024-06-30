# `D:\src\scipysrc\scipy\scipy\optimize\_highs\cython\src\HighsStatus.pxd`

```
# 设定 Cython 的语言级别为3
# 导入 C++ 的标准字符串类型string，从 libcpp.string 模块中引入
from libcpp.string cimport string

# 从外部头文件 "HighsStatus.h" 中声明 HighsStatus 枚举类型，nogil 表示没有 GIL（全局解释器锁）保护
cdef extern from "HighsStatus.h" nogil:
    # 定义 HighsStatus 枚举类型
    ctypedef enum HighsStatus:
        # HighsStatusError 对应值为 -1，表示 HighsStatus::kError
        HighsStatusError "HighsStatus::kError" = -1
        # HighsStatusOK 对应值为 0，表示 HighsStatus::kOk
        HighsStatusOK "HighsStatus::kOk" = 0
        # HighsStatusWarning 对应值为 1，表示 HighsStatus::kWarning
        HighsStatusWarning "HighsStatus::kWarning" = 1

    # 声明 highsStatusToString 函数，接受 HighsStatus 枚举类型作为参数，返回一个字符串类型
    string highsStatusToString(HighsStatus status)
```