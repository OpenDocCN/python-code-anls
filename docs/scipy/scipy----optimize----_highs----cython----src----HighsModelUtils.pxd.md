# `D:\src\scipysrc\scipy\scipy\optimize\_highs\cython\src\HighsModelUtils.pxd`

```
# 指定Cython编译器选项，设置语言级别为3（Cython的语法版本）
# 这个声明告诉Cython编译器要使用C++的字符串处理库
# 并且从当前目录下的HConst模块中导入HighsModelStatus类
from libcpp.string cimport string

# 从HConst模块中导入HighsModelStatus类，供后续使用
from .HConst cimport HighsModelStatus

# 使用nogil声明，告诉Cython不使用GIL（全局解释器锁）
# 从HiGHS/src/lp_data/HighsModelUtils.h文件中导入函数声明
cdef extern from "HighsModelUtils.h" nogil:
    # 声明一个C++函数，将HighsModelStatus转换为字符串
    string utilHighsModelStatusToString(const HighsModelStatus model_status)
    # 声明一个C++函数，将整数类型的primal_dual_status转换为字符串
    string utilBasisStatusToString(const int primal_dual_status)
```