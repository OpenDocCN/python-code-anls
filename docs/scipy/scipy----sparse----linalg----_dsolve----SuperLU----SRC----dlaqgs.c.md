# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dlaqgs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file dlaqgs.c
 * \brief Equlibrates a general sprase matrix
 *
 * <pre>
 * -- SuperLU routine (version 2.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November 15, 1997
 * 
 * Modified from LAPACK routine DLAQGE
 * </pre>
 */
/*
 * File name:    dlaqgs.c
 * History:     Modified from LAPACK routine DLAQGE
 */
#include <math.h>
#include "slu_ddefs.h"

/*! \brief
 *
 * <pre>
 *   Purpose   
 *   =======   
 *
 *   DLAQGS equilibrates a general sparse M by N matrix A using the row and   
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
 *        Stype = NC; Dtype = SLU_D; Mtype = GE.
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
dlaqgs(SuperMatrix *A, double *r, double *c, 
       double rowcnd, double colcnd, double amax, char *equed) {
    // 检查行和列的比例条件是否低于阈值，以决定是否进行行或列的缩放
    double thresh = 0.1;
    // 定义行缩放的阈值，用于基于最大矩阵元素的绝对大小决定是否进行行缩放
    double large = 0.5;
    double small = 0.3;

    // 根据行和列的条件比较决定 equed 的值，指示进行了何种均衡操作
    if (rowcnd < thresh) {
        if (colcnd < thresh) {
            *equed = 'B'; // 行列均衡
        } else {
            *equed = 'R'; // 仅行均衡
        }
    } else if (colcnd < thresh) {
        *equed = 'C'; // 仅列均衡
    } else {
        *equed = 'N'; // 没有进行均衡操作
    }

    // 返回 equed 指示的均衡操作结果
    return;
}
    double rowcnd,      // double 类型变量，用于存储行条件数
           colcnd,      // double 类型变量，用于存储列条件数
           amax;        // double 类型变量，用于存储矩阵的绝对值最大元素
    char *equed         // 指向 char 类型的指针，用于表示矩阵的均衡状态
/* 定义阈值 THRESH 为 0.1 */
#define THRESH    (0.1)
    
/* 声明局部变量 */
NCformat *Astore;  // 指向稀疏矩阵 A 的存储结构
double   *Aval;    // 指向矩阵 A 的非零元素数组
int_t i, j;        // 循环变量
int   irow;        // 行索引变量
double large, small, cj;  // 用于存储计算中的数值

/* 如果矩阵 A 的行数或列数小于等于 0，则直接返回 */
if (A->nrow <= 0 || A->ncol <= 0) {
    *(unsigned char *)equed = 'N';  // 将 equed 设置为 'N'
    return;  // 函数退出
}

Astore = A->Store;  // 获取矩阵 A 的存储结构
Aval = Astore->nzval;  // 获取矩阵 A 的非零元素数组

/* 初始化 LARGE 和 SMALL */
small = dmach("Safe minimum") / dmach("Precision");  // 计算安全最小值
large = 1. / small;  // 计算相对较大的值

/* 根据条件选择不同的操作路径 */
if (rowcnd >= THRESH && amax >= small && amax <= large) {
    if (colcnd >= THRESH)
        *(unsigned char *)equed = 'N';  // 如果列条件满足阈值，则将 equed 设置为 'N'
    else {
        /* 列缩放 */
        for (j = 0; j < A->ncol; ++j) {
            cj = c[j];  // 获取列缩放因子
            for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; ++i) {
                Aval[i] *= cj;  // 对第 j 列进行缩放
            }
        }
        *(unsigned char *)equed = 'C';  // 将 equed 设置为 'C'
    }
} else if (colcnd >= THRESH) {
    /* 行缩放，无列缩放 */
    for (j = 0; j < A->ncol; ++j) {
        for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; ++i) {
            irow = Astore->rowind[i];  // 获取行索引
            Aval[i] *= r[irow];  // 对第 irow 行进行缩放
        }
    }
    *(unsigned char *)equed = 'R';  // 将 equed 设置为 'R'
} else {
    /* 行列均缩放 */
    for (j = 0; j < A->ncol; ++j) {
        cj = c[j];  // 获取列缩放因子
        for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; ++i) {
            irow = Astore->rowind[i];  // 获取行索引
            Aval[i] *= cj * r[irow];  // 对第 j 列第 irow 行进行缩放
        }
    }
    *(unsigned char *)equed = 'B';  // 将 equed 设置为 'B'
}

return;  // 函数返回

} /* dlaqgs */


这段代码是一个函数 `dlaqgs`，用于根据一些条件对稀疏矩阵进行行和列的缩放操作，并根据缩放的情况设定一个标志 `equed` 的值。
```