# `D:\src\scipysrc\scipy\scipy\optimize\_highs\cython\src\HighsRuntimeOptions.pxd`

```
# 在 Cython 中设置语言级别为 3
# 从 libcpp 中导入 C++ 中的 bool 类型
from libcpp cimport bool

# 从当前目录中的 HighsOptions 模块导入 HighsOptions 类
from .HighsOptions cimport HighsOptions

# 使用 nogil 声明，表示以下代码不会释放全局解释器锁 (GIL)，用于优化性能

cdef extern from "HighsRuntimeOptions.h" nogil:
    # 从 HiGHS/src/lp_data/HighsRuntimeOptions.h 头文件中导入以下函数声明

    # 声明 loadOptions 函数，接受 argc（参数数量）、argv（参数数组）、options（HighsOptions 对象引用）作为参数
    bool loadOptions(int argc, char** argv, HighsOptions& options)
```