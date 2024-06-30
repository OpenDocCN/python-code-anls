# `D:\src\scipysrc\scipy\scipy\optimize\_highs\cython\src\HighsLpUtils.pxd`

```
# 指定Cython的语言级别为3，这影响Cython代码的编译和语法特性
# 根据当前文件结构导入所需的Cython类型和函数
from .HighsStatus cimport HighsStatus
from .HighsLp cimport HighsLp
from .HighsOptions cimport HighsOptions

# 从外部头文件"HighsLpUtils.h"中导入函数声明，声明为无GIL（全局解释器锁）的函数
cdef extern from "HighsLpUtils.h" nogil:
    # 从"HiGHS/src/lp_data/HighsLpUtils.h"文件中导入的函数"assessLp"
    # 该函数对给定的线性规划对象（HighsLp）进行评估，并根据给定的选项（HighsOptions）返回评估结果（HighsStatus）
    HighsStatus assessLp(HighsLp& lp, const HighsOptions& options)
```