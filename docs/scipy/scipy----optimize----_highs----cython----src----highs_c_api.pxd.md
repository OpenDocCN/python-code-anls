# `D:\src\scipysrc\scipy\scipy\optimize\_highs\cython\src\highs_c_api.pxd`

```
# 设置 Cython 的语言级别为 3，确保与 Python 3 兼容

cdef extern from "highs_c_api.h" nogil:
    # 声明外部函数 Highs_passLp，接受指向 Highs 对象的指针及线性规划问题的描述
    int Highs_passLp(void* highs, int numcol, int numrow, int numnz,
                     double* colcost, double* collower, double* colupper,
                     double* rowlower, double* rowupper,
                     int* astart, int* aindex, double* avalue)


这段代码是用于 Cython 的声明语法，用于导入 C 语言头文件中的函数声明，并将其用于 Cython 程序中。
```