# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zlaqgs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file zlaqgs.c
 * \brief Equlibrates a general sparse matrix
 *
 * <pre>
 * -- SuperLU routine (version 2.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November 15, 1997
 * 
 * Modified from LAPACK routine ZLAQGE
 * </pre>
 */
/*
 * File name:    zlaqgs.c
 * History:     Modified from LAPACK routine ZLAQGE
 */
#include <math.h>
#include "slu_zdefs.h"

/*! \brief
 *
 * <pre>
 *   Purpose   
 *   =======   
 *
 *   ZLAQGS equilibrates a general sparse M by N matrix A using the row and   
 *   scaling factors in the vectors R and C.   
 *
 *   See supermatrix.h for the definition of 'SuperMatrix' structure.
 *
 *   Arguments   
 *   =========   
 *
 *   A       (input/output) SuperMatrix*
 *           On exit, the equilibrated matrix.  See EQUED for the form of 
 *           the equilibrated matrix. The type of A can be:
 *        Stype = NC; Dtype = SLU_Z; Mtype = GE.
 *        
 *   R       (input) double*, dimension (A->nrow)
 *           The row scale factors for A.
 *        
 *   C       (input) double*, dimension (A->ncol)
 *           The column scale factors for A.
 *        
 *   ROWCND  (input) double
 *           Ratio of the smallest R(i) to the largest R(i).
 *        
 *   COLCND  (input) double
 *           Ratio of the smallest C(i) to the largest C(i).
 *        
 *   AMAX    (input) double
 *           Absolute value of largest matrix entry.
 *        
 *   EQUED   (output) char*
 *           Specifies the form of equilibration that was done.   
 *           = 'N':  No equilibration   
 *           = 'R':  Row equilibration, i.e., A has been premultiplied by  
 *                   diag(R).   
 *           = 'C':  Column equilibration, i.e., A has been postmultiplied  
 *                   by diag(C).   
 *           = 'B':  Both row and column equilibration, i.e., A has been
 *                   replaced by diag(R) * A * diag(C).   
 *
 *   Internal Parameters   
 *   ===================   
 *
 *   THRESH is a threshold value used to decide if row or column scaling   
 *   should be done based on the ratio of the row or column scaling   
 *   factors.  If ROWCND < THRESH, row scaling is done, and if   
 *   COLCND < THRESH, column scaling is done.   
 *
 *   LARGE and SMALL are threshold values used to decide if row scaling   
 *   should be done based on the absolute size of the largest matrix   
 *   element.  If AMAX > LARGE or AMAX < SMALL, row scaling is done.   
 *
 *   ===================================================================== 
 * </pre>
 */

void
zlaqgs(SuperMatrix *A, double *r, double *c, 
      double rowcnd, double colcnd, double amax, char *equed)
{
    // 如果超过阈值，进行行或列的缩放
    // 根据 R 和 C 向量中的缩放因子，对稀疏矩阵 A 进行均衡化
    // equed 参数指示均衡化的形式
    // 'N': 无均衡化
    // 'R': 行均衡化，即 A 已经被 diag(R) 预乘
    // 'C': 列均衡化，即 A 已经被 diag(C) 后乘
    // 'B': 行列均衡化，即 A 已经被 diag(R) * A * diag(C) 替换
}
    double rowcnd, double colcnd, double amax, char *equed)



// 定义函数参数列表，参数说明如下：
// - rowcnd: 行条件数
// - colcnd: 列条件数
// - amax:   绝对值最大元素
// - equed:  用于存储等式的字符指针，可能表示等式的类型或者需要被修改的状态
{
    
#define THRESH    (0.1)
    
    /* Local variables */
    NCformat *Astore;  // Astore 是 NCformat 类型指针，用于存储矩阵 A 的存储格式
    doublecomplex   *Aval;  // Aval 是双精度复数类型指针，指向矩阵 A 的非零元素数组
    int_t i, j;  // i, j 是整数类型变量，用于循环迭代
    int   irow;  // irow 是整数类型变量，用于存储行索引
    double large, small, cj;  // large, small, cj 是双精度浮点数变量，用于存储临时数据
    double temp;  // temp 是双精度浮点数变量，用于临时存储计算结果

    
    /* Quick return if possible */
    if (A->nrow <= 0 || A->ncol <= 0) {  // 如果 A 的行数或列数小于等于 0，则设置 equed 为 'N' 并返回
    *(unsigned char *)equed = 'N';
    return;
    }

    Astore = A->Store;  // 将 A 的存储结构赋值给 Astore
    Aval = Astore->nzval;  // 将 A 的非零元素数组赋值给 Aval
    
    /* Initialize LARGE and SMALL. */
    small = dmach("Safe minimum") / dmach("Precision");  // 计算并初始化 small，为安全最小值除以精度
    large = 1. / small;  // 计算并初始化 large，为 1 除以 small

    if (rowcnd >= THRESH && amax >= small && amax <= large) {  // 如果行条件数大于等于 THRESH 并且最大绝对值元素大于等于 small 且小于等于 large
    if (colcnd >= THRESH)  // 如果列条件数大于等于 THRESH
        *(unsigned char *)equed = 'N';  // 设置 equed 为 'N'
    else {
        /* Column scaling */
        for (j = 0; j < A->ncol; ++j) {  // 遍历每一列
        cj = c[j];  // 获取列缩放因子
        for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; ++i) {  // 遍历列 j 中的每一个非零元素
            zd_mult(&Aval[i], &Aval[i], cj);  // 使用 zd_mult 函数对非零元素进行缩放
                }
        }
        *(unsigned char *)equed = 'C';  // 设置 equed 为 'C'
    }
    } else if (colcnd >= THRESH) {
    /* Row scaling, no column scaling */
    for (j = 0; j < A->ncol; ++j)  // 遍历每一列
        for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; ++i) {  // 遍历列 j 中的每一个非零元素
        irow = Astore->rowind[i];  // 获取元素所在的行索引
        zd_mult(&Aval[i], &Aval[i], r[irow]);  // 使用 zd_mult 函数对非零元素进行行缩放
        }
    *(unsigned char *)equed = 'R';  // 设置 equed 为 'R'
    } else {
    /* Row and column scaling */
    for (j = 0; j < A->ncol; ++j) {  // 遍历每一列
        cj = c[j];  // 获取列缩放因子
        for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; ++i) {  // 遍历列 j 中的每一个非零元素
        irow = Astore->rowind[i];  // 获取元素所在的行索引
        temp = cj * r[irow];  // 计算行列缩放因子的乘积
        zd_mult(&Aval[i], &Aval[i], temp);  // 使用 zd_mult 函数对非零元素进行行列缩放
        }
    }
    *(unsigned char *)equed = 'B';  // 设置 equed 为 'B'
    }

    return;

} /* zlaqgs */
```