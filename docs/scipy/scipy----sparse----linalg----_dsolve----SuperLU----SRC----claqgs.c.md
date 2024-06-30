# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\claqgs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file claqgs.c
 * \brief Equlibrates a general sprase matrix
 *
 * <pre>
 * -- SuperLU routine (version 2.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November 15, 1997
 * 
 * Modified from LAPACK routine CLAQGE
 * </pre>
 */
/*
 * File name:    claqgs.c
 * History:     Modified from LAPACK routine CLAQGE
 */
#include <math.h>
#include "slu_cdefs.h"

/*! \brief
 *
 * <pre>
 *   Purpose   
 *   =======   
 *
 *   CLAQGS equilibrates a general sparse M by N matrix A using the row and   
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
 *        Stype = NC; Dtype = SLU_C; Mtype = GE.
 *        
 *   R       (input) float*, dimension (A->nrow)
 *           The row scale factors for A.
 *        
 *   C       (input) float*, dimension (A->ncol)
 *           The column scale factors for A.
 *        
 *   ROWCND  (input) float
 *           Ratio of the smallest R(i) to the largest R(i).
 *        
 *   COLCND  (input) float
 *           Ratio of the smallest C(i) to the largest C(i).
 *        
 *   AMAX    (input) float
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
claqgs(SuperMatrix *A, float *r, float *c, 
       float rowcnd, float colcnd, float amax, char *equed)
{
    // THRESH 是用于判断是否进行行或列缩放的阈值
    float thresh = 0.1;
    
    // LARGE 和 SMALL 是用于判断是否进行行缩放的阈值
    float large = 0.75;
    float small = 0.25;
    
    // 计算行数和列数
    int m = A->nrow;
    int n = A->ncol;
    
    // 根据行和列的缩放条件进行矩阵 A 的均衡化操作
    if (rowcnd < thresh && colcnd < thresh) {
        // 如果行和列的缩放条件都小于阈值，则进行行和列均衡化
        *equed = 'B';
        for (int i = 0; i < m; ++i) {
            r[i] = 1.0 / sqrt(r[i]);
        }
        for (int j = 0; j < n; ++j) {
            c[j] = sqrt(c[j]);
        }
        for (int jcol = 0; jcol < n; ++jcol) {
            for (int p = A->colptr[jcol]; p < A->colptr[jcol+1]; ++p) {
                A->nzval[p].r *= c[jcol];
                A->nzval[p].i *= c[jcol];
            }
        }
    } else if (rowcnd < thresh && amax > large) {
        // 如果行的缩放条件小于阈值且最大元素的绝对值大于大阈值，则进行行均衡化
        *equed = 'R';
        for (int i = 0; i < m; ++i) {
            r[i] = 1.0 / sqrt(r[i]);
        }
        for (int jcol = 0; jcol < n; ++jcol) {
            for (int p = A->colptr[jcol]; p < A->colptr[jcol+1]; ++p) {
                A->nzval[p].r *= c[jcol];
                A->nzval[p].i *= c[jcol];
            }
        }
    } else if (colcnd < thresh) {
        // 如果列的缩放条件小于阈值，则进行列均衡化
        *equed = 'C';
        for (int j = 0; j < n; ++j) {
            c[j] = sqrt(c[j]);
        }
        for (int jcol = 0; jcol < n; ++jcol) {
            for (int p = A->colptr[jcol]; p < A->colptr[jcol+1]; ++p) {
                A->nzval[p].r *= c[jcol];
                A->nzval[p].i *= c[jcol];
            }
        }
    } else {
        // 否则，不进行均衡化
        *equed = 'N';
    }
}
    float rowcnd, float colcnd, float amax, char *equed)



    // 定义函数参数列表，包括四个参数：
    // - rowcnd: 行条件数
    // - colcnd: 列条件数
    // - amax: 矩阵的绝对值最大元素
    // - equed: 指向字符的指针，表示均衡条件
/* 定义阈值THRESH为0.1 */

#define THRESH    (0.1)
    
/* 声明本地变量 */
NCformat *Astore;
singlecomplex   *Aval;
int_t i, j;
int   irow;
float large, small, cj;
float temp;


/* 如果矩阵A的行数或列数小于等于0，则快速返回 */
if (A->nrow <= 0 || A->ncol <= 0) {
    *(unsigned char *)equed = 'N';
    return;
}

/* 从矩阵A中获取存储格式为NC的结构体Astore */
Astore = A->Store;
/* 获取矩阵A存储的非零元素数组Aval */
Aval = Astore->nzval;
    
/* 初始化变量LARGE和SMALL */
small = smach("Safe minimum") / smach("Precision");
large = 1. / small;

/* 根据条件判断是否需要进行行和列的缩放操作 */
if (rowcnd >= THRESH && amax >= small && amax <= large) {
    if (colcnd >= THRESH)
        *(unsigned char *)equed = 'N';
    else {
        /* 对列进行缩放 */
        for (j = 0; j < A->ncol; ++j) {
            cj = c[j];
            for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; ++i) {
                cs_mult(&Aval[i], &Aval[i], cj);
            }
        }
        *(unsigned char *)equed = 'C';
    }
} else if (colcnd >= THRESH) {
    /* 对行进行缩放，不对列进行缩放 */
    for (j = 0; j < A->ncol; ++j) {
        for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; ++i) {
            irow = Astore->rowind[i];
            cs_mult(&Aval[i], &Aval[i], r[irow]);
        }
    }
    *(unsigned char *)equed = 'R';
} else {
    /* 对行和列同时进行缩放 */
    for (j = 0; j < A->ncol; ++j) {
        cj = c[j];
        for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; ++i) {
            irow = Astore->rowind[i];
            temp = cj * r[irow];
            cs_mult(&Aval[i], &Aval[i], temp);
        }
    }
    *(unsigned char *)equed = 'B';
}

return;
```