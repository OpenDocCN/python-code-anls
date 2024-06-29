# `D:\src\scipysrc\matplotlib\extern\agg24-svn\include\agg_simul_eq.h`

```py
#ifndef AGG_SIMUL_EQ_INCLUDED
#define AGG_SIMUL_EQ_INCLUDED

// 如果未定义 AGG_SIMUL_EQ_INCLUDED 宏，则定义它，避免重复包含


#include <math.h>
#include "agg_basics.h"

// 包含数学函数头文件 math.h 和自定义基础头文件 agg_basics.h


namespace agg
{

// 命名空间声明，将后续的代码放在 agg 命名空间中


    //=============================================================swap_arrays
    template<class T> void swap_arrays(T* a1, T* a2, unsigned n)
    {
        unsigned i;
        for(i = 0; i < n; i++)
        {
            T tmp = *a1;
            *a1++ = *a2;
            *a2++ = tmp;
        }
    }

// swap_arrays 函数模板定义，用于交换两个数组的内容，参数为两个指向类型 T 的指针 a1 和 a2，以及数组长度 n


    //============================================================matrix_pivot
    template<unsigned Rows, unsigned Cols>
    struct matrix_pivot
    {
        static int pivot(double m[Rows][Cols], unsigned row)
        {
            int k = int(row);
            double max_val, tmp;

            max_val = -1.0;
            unsigned i;
            for(i = row; i < Rows; i++)
            {
                if((tmp = fabs(m[i][row])) > max_val && tmp != 0.0)
                {
                    max_val = tmp;
                    k = i;
                }
            }

            if(m[k][row] == 0.0)
            {
                return -1;
            }

            if(k != int(row))
            {
                swap_arrays(m[k], m[row], Cols);
                return k;
            }
            return 0;
        }
    };

// matrix_pivot 结构模板定义，用于在矩阵中进行主元素选取和交换，参数为矩阵维度 Rows 和 Cols，实现了 pivot 方法来执行主元素选取和交换的操作


    //===============================================================simul_eq
    template<unsigned Size, unsigned RightCols>
    struct simul_eq

// simul_eq 结构模板定义，用于解决 Size 个方程同时的问题，其中 RightCols 表示方程右侧列数的参数



#ifndef AGG_SIMUL_EQ_INCLUDED
#define AGG_SIMUL_EQ_INCLUDED
    {
        // 定义一个静态函数 solve，用于解决大小为 Size x Size 的线性方程组
        static bool solve(const double left[Size][Size], 
                          const double right[Size][RightCols],
                          double result[Size][RightCols])
        {
            unsigned i, j, k;
            double a1;
    
            // 创建临时数组 tmp，大小为 Size x (Size + RightCols)，用于存储增广矩阵
            double tmp[Size][Size + RightCols];
    
            // 将左矩阵 left 复制到 tmp 的左侧部分
            for(i = 0; i < Size; i++)
            {
                for(j = 0; j < Size; j++)
                {
                    tmp[i][j] = left[i][j];
                } 
                // 将右矩阵 right 复制到 tmp 的右侧部分
                for(j = 0; j < RightCols; j++)
                {
                    tmp[i][Size + j] = right[i][j];
                }
            }
    
            // 进行高斯消元法处理增广矩阵
            for(k = 0; k < Size; k++)
            {
                // 如果主元为零，则返回 false，表示奇异矩阵
                if(matrix_pivot<Size, Size + RightCols>::pivot(tmp, k) < 0)
                {
                    return false; // Singularity....
                }
    
                // 主元归一化
                a1 = tmp[k][k];
                for(j = k; j < Size + RightCols; j++)
                {
                    tmp[k][j] /= a1;
                }
    
                // 消元操作
                for(i = k + 1; i < Size; i++)
                {
                    a1 = tmp[i][k];
                    for (j = k; j < Size + RightCols; j++)
                    {
                        tmp[i][j] -= a1 * tmp[k][j];
                    }
                }
            }
    
            // 回代过程，求解右侧矩阵 result
            for(k = 0; k < RightCols; k++)
            {
                int m;
                // 从最后一行开始向上回代
                for(m = int(Size - 1); m >= 0; m--)
                {
                    result[m][k] = tmp[m][Size + k];
                    // 利用已知的 result 求解剩余部分
                    for(j = m + 1; j < Size; j++)
                    {
                        result[m][k] -= tmp[m][j] * result[j][k];
                    }
                }
            }
            // 返回 true 表示成功求解线性方程组
            return true;
        }
    };
}


注释：

// 关闭一个 #ifdef 或 #ifndef 块的结束标记



#endif


注释：

// 结束一个 #ifdef 或 #ifndef 块的条件编译
```