# `D:\src\scipysrc\scipy\scipy\optimize\_highs\cython\src\HighsIO.pxd`

```
# cython: language_level=3

# 从头文件 "HighsIO.h" 中导入外部定义，使用 nogil 模式（不需要 GIL，全局解释器锁）
cdef extern from "HighsIO.h" nogil:
    # 由于 Cython < 3.x 不支持 enum class，因此这里使用的是一种变通方法

    # 定义 C++ 枚举类 HighsLogType，用于表示不同的日志类型
    cdef cppclass HighsLogType:
        pass

    # 下面是具体的枚举值，用于表示不同的日志类型
    cdef HighsLogType kInfo "HighsLogType::kInfo"          # 表示信息日志
    cdef HighsLogType kDetailed "HighsLogType::kDetailed"  # 表示详细信息日志
    cdef HighsLogType kVerbose "HighsLogType::kVerbose"    # 表示详细输出日志
    cdef HighsLogType kWarning "HighsLogType::kWarning"    # 表示警告日志
    cdef HighsLogType kError "HighsLogType::kError"        # 表示错误日志
```