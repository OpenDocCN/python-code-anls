# `D:\src\scipysrc\scipy\scipy\optimize\_highs\cython\src\HighsLp.pxd`

```
# cython: language_level=3

# 导入必要的 C++ 类型和模块
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

# 从本地模块中导入特定类和枚举类型
from .HConst cimport HighsBasisStatus, ObjSense, HighsVarType
from .HighsSparseMatrix cimport HighsSparseMatrix

# 定义外部 C 函数接口，声明了 HiGHS 线性规划的结构体 HighsLp
cdef extern from "HighsLp.h" nogil:
    # HighsLp 类定义
    cdef cppclass HighsLp:
        int num_col_                    # 列数
        int num_row_                    # 行数

        vector[double] col_cost_        # 列成本
        vector[double] col_lower_       # 列下界
        vector[double] col_upper_       # 列上界
        vector[double] row_lower_       # 行下界
        vector[double] row_upper_       # 行上界

        HighsSparseMatrix a_matrix_     # 稀疏矩阵 A

        ObjSense sense_                 # 目标函数方向
        double offset_                 # 偏移量

        string model_name_              # 模型名称

        vector[string] row_names_       # 行名称
        vector[string] col_names_       # 列名称

        vector[HighsVarType] integrality_  # 变量整数性

        bool isMip() const             # 是否为整数规划的方法声明

    # HighsSolution 类定义
    cdef cppclass HighsSolution:
        vector[double] col_value        # 列值
        vector[double] col_dual         # 列对偶
        vector[double] row_value        # 行值
        vector[double] row_dual         # 行对偶

    # HighsBasis 类定义
    cdef cppclass HighsBasis:
        bool valid_                     # 是否有效
        vector[HighsBasisStatus] col_status  # 列基状态
        vector[HighsBasisStatus] row_status  # 行基状态
```