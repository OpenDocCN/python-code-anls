# `D:\src\scipysrc\scipy\scipy\optimize\_highs\cython\src\HighsSparseMatrix.pxd`

```
# cython: language_level=3
# 导入必要的 C++ 类型和函数声明
from libcpp.vector cimport vector  # 导入 C++ 标准库中的向量容器类型

# 导入 MatrixFormat 类型定义
from .HConst cimport MatrixFormat  # 从当前包的 HConst 模块中导入 MatrixFormat 类型定义

# 外部声明 C++ 类 HighsSparseMatrix，对应于 HighsSparseMatrix.h 头文件
cdef extern from "HighsSparseMatrix.h" nogil:
    # 定义 C++ 类 HighsSparseMatrix
    cdef cppclass HighsSparseMatrix:
        MatrixFormat format_        # 矩阵格式属性
        int num_col_                # 列数属性
        int num_row_                # 行数属性
        vector[int] start_          # 起始索引向量属性
        vector[int] index_          # 索引向量属性
        vector[double] value_       # 值向量属性
```