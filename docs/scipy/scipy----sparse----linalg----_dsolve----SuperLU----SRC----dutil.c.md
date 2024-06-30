# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dutil.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file dutil.c
 * \brief Matrix utility functions
 *
 * <pre>
 * -- SuperLU routine (version 3.1) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * August 1, 2008
 *
 * Copyright (c) 1994 by Xerox Corporation.  All rights reserved.
 *
 * THIS MATERIAL IS PROVIDED AS IS, WITH ABSOLUTELY NO WARRANTY
 * EXPRESSED OR IMPLIED.  ANY USE IS AT YOUR OWN RISK.
 * 
 * Permission is hereby granted to use or copy this program for any
 * purpose, provided the above notices are retained on all copies.
 * Permission to modify the code and to distribute modified code is
 * granted, provided the above notices are retained, and a notice that
 * the code was modified is included with the above copyright notice.
 * </pre>
 */


#include <math.h>
#include "slu_ddefs.h"

/*! \brief Create a compressed column matrix A. */
void
dCreate_CompCol_Matrix(SuperMatrix *A, int m, int n, int_t nnz, 
               double *nzval, int_t *rowind, int_t *colptr,
               Stype_t stype, Dtype_t dtype, Mtype_t mtype)
{
    NCformat *Astore;

    A->Stype = stype;  // 设置稀疏矩阵类型为压缩列存储
    A->Dtype = dtype;  // 设置数据类型
    A->Mtype = mtype;  // 设置矩阵类型
    A->nrow = m;       // 设置矩阵行数
    A->ncol = n;       // 设置矩阵列数
    A->Store = (void *) SUPERLU_MALLOC( sizeof(NCformat) );  // 分配存储空间给稀疏矩阵的存储结构
    if ( !(A->Store) ) ABORT("SUPERLU_MALLOC fails for A->Store");  // 失败处理，若内存分配失败则终止程序
    Astore = A->Store;  // 获取稀疏矩阵的存储结构指针
    Astore->nnz = nnz;  // 设置非零元素个数
    Astore->nzval = nzval;  // 设置非零元素值数组
    Astore->rowind = rowind;  // 设置行索引数组
    Astore->colptr = colptr;  // 设置列指针数组
}

/*! \brief Create a compressed row matrix A. */
void
dCreate_CompRow_Matrix(SuperMatrix *A, int m, int n, int_t nnz, 
               double *nzval, int_t *colind, int_t *rowptr,
               Stype_t stype, Dtype_t dtype, Mtype_t mtype)
{
    NRformat *Astore;

    A->Stype = stype;  // 设置稀疏矩阵类型为压缩行存储
    A->Dtype = dtype;  // 设置数据类型
    A->Mtype = mtype;  // 设置矩阵类型
    A->nrow = m;       // 设置矩阵行数
    A->ncol = n;       // 设置矩阵列数
    A->Store = (void *) SUPERLU_MALLOC( sizeof(NRformat) );  // 分配存储空间给稀疏矩阵的存储结构
    if ( !(A->Store) ) ABORT("SUPERLU_MALLOC fails for A->Store");  // 失败处理，若内存分配失败则终止程序
    Astore = A->Store;  // 获取稀疏矩阵的存储结构指针
    Astore->nnz = nnz;  // 设置非零元素个数
    Astore->nzval = nzval;  // 设置非零元素值数组
    Astore->colind = colind;  // 设置列索引数组
    Astore->rowptr = rowptr;  // 设置行指针数组
}

/*! \brief Copy matrix A into matrix B. */
void
dCopy_CompCol_Matrix(SuperMatrix *A, SuperMatrix *B)
{
    NCformat *Astore, *Bstore;
    int      ncol, nnz, i;

    B->Stype = A->Stype;  // 设置矩阵B的稀疏矩阵类型与A相同
    B->Dtype = A->Dtype;  // 设置矩阵B的数据类型与A相同
    B->Mtype = A->Mtype;  // 设置矩阵B的矩阵类型与A相同
    B->nrow  = A->nrow;   // 设置矩阵B的行数与A相同
    B->ncol  = ncol = A->ncol;  // 设置矩阵B的列数与A相同
    Astore   = (NCformat *) A->Store;  // 获取矩阵A的存储结构指针
    Bstore   = (NCformat *) B->Store;  // 获取矩阵B的存储结构指针
    Bstore->nnz = nnz = Astore->nnz;   // 设置矩阵B的非零元素个数与A相同
    for (i = 0; i < nnz; ++i)
        ((double *)Bstore->nzval)[i] = ((double *)Astore->nzval)[i];  // 复制矩阵A的非零元素值到矩阵B
    for (i = 0; i < nnz; ++i)
        Bstore->rowind[i] = Astore->rowind[i];  // 复制矩阵A的行索引到矩阵B
    for (i = 0; i <= ncol; ++i)
        Bstore->colptr[i] = Astore->colptr[i];  // 复制矩阵A的列指针到矩阵B
}
void
dCreate_Dense_Matrix(SuperMatrix *X, int m, int n, double *x, int ldx,
            Stype_t stype, Dtype_t dtype, Mtype_t mtype)
{
    DNformat    *Xstore;
    
    X->Stype = stype;                           // 设置稀疏矩阵 X 的存储类型
    X->Dtype = dtype;                           // 设置稀疏矩阵 X 的数据类型
    X->Mtype = mtype;                           // 设置稀疏矩阵 X 的矩阵类型
    X->nrow = m;                                // 设置稀疏矩阵 X 的行数
    X->ncol = n;                                // 设置稀疏矩阵 X 的列数
    X->Store = (void *) SUPERLU_MALLOC( sizeof(DNformat) );   // 分配稀疏矩阵 X 的存储空间
    if ( !(X->Store) ) ABORT("SUPERLU_MALLOC fails for X->Store");   // 若分配失败则终止程序
    Xstore = (DNformat *) X->Store;             // 将分配的存储空间转换为 DNformat 类型并赋给 Xstore
    Xstore->lda = ldx;                          // 设置 Xstore 的列偏移量
    Xstore->nzval = (double *) x;               // 将数组 x 的指针赋给 Xstore 的数据指针
}

void
dCopy_Dense_Matrix(int M, int N, double *X, int ldx,
            double *Y, int ldy)
{
/*! \brief Copies a two-dimensional matrix X to another matrix Y.
 */
    int    i, j;
    
    for (j = 0; j < N; ++j)                     // 外层循环遍历列
        for (i = 0; i < M; ++i)                 // 内层循环遍历行
            Y[i + j*ldy] = X[i + j*ldx];       // 将矩阵 X 中的元素复制到矩阵 Y 中对应位置
}

void
dCreate_SuperNode_Matrix(SuperMatrix *L, int m, int n, int_t nnz, 
            double *nzval, int_t *nzval_colptr, int_t *rowind,
            int_t *rowind_colptr, int *col_to_sup, int *sup_to_col,
            Stype_t stype, Dtype_t dtype, Mtype_t mtype)
{
    SCformat *Lstore;

    L->Stype = stype;                           // 设置超节点矩阵 L 的存储类型
    L->Dtype = dtype;                           // 设置超节点矩阵 L 的数据类型
    L->Mtype = mtype;                           // 设置超节点矩阵 L 的矩阵类型
    L->nrow = m;                                // 设置超节点矩阵 L 的行数
    L->ncol = n;                                // 设置超节点矩阵 L 的列数
    L->Store = (void *) SUPERLU_MALLOC( sizeof(SCformat) );   // 分配超节点矩阵 L 的存储空间
    if ( !(L->Store) ) ABORT("SUPERLU_MALLOC fails for L->Store");   // 若分配失败则终止程序
    Lstore = L->Store;                          // 将分配的存储空间赋给 Lstore
    Lstore->nnz = nnz;                          // 设置 Lstore 中非零元素的数量
    Lstore->nsuper = col_to_sup[n];             // 设置 Lstore 中超节点的数量
    Lstore->nzval = nzval;                      // 设置 Lstore 中非零元素值的指针
    Lstore->nzval_colptr = nzval_colptr;        // 设置 Lstore 中非零元素列指针的指针
    Lstore->rowind = rowind;                    // 设置 Lstore 中行指标的指针
    Lstore->rowind_colptr = rowind_colptr;      // 设置 Lstore 中行指标列指针的指针
    Lstore->col_to_sup = col_to_sup;            // 设置 Lstore 中列到超节点映射的指针
    Lstore->sup_to_col = sup_to_col;            // 设置 Lstore 中超节点到列映射的指针
}

/*! \brief Convert a row compressed storage into a column compressed storage.
 */
void
dCompRow_to_CompCol(int m, int n, int_t nnz, 
            double *a, int_t *colind, int_t *rowptr,
            double **at, int_t **rowind, int_t **colptr)
{
    register int i, j, col, relpos;
    int_t *marker;

    /* Allocate storage for another copy of the matrix. */
    *at = (double *) doubleMalloc(nnz);          // 分配存储转置后矩阵的非零元素值数组
    *rowind = (int_t *) intMalloc(nnz);          // 分配存储转置后矩阵的行指标数组
    *colptr = (int_t *) intMalloc(n+1);          // 分配存储转置后矩阵的列指针数组
    marker = (int_t *) intCalloc(n);             // 分配存储列中非零元素计数的数组
    
    /* Get counts of each column of A, and set up column pointers */
    for (i = 0; i < m; ++i)
        for (j = rowptr[i]; j < rowptr[i+1]; ++j) ++marker[colind[j]];   // 统计每列中的非零元素个数
    (*colptr)[0] = 0;
    for (j = 0; j < n; ++j) {
        (*colptr)[j+1] = (*colptr)[j] + marker[j];  // 设置列指针数组
        marker[j] = (*colptr)[j];                   // 重置 marker 数组为起始位置
    }

    /* Transfer the matrix into the compressed column storage. */
    for (i = 0; i < m; ++i) {
        for (j = rowptr[i]; j < rowptr[i+1]; ++j) {
            col = colind[j];
            relpos = marker[col];
            (*rowind)[relpos] = i;                   // 设置转置后矩阵的行指标
            (*at)[relpos] = a[j];                    // 设置转置后矩阵的非零元素值
            ++marker[col];                           // 移动 marker 到下一个位置
        }
    }

    SUPERLU_FREE(marker);                           // 释放 marker 数组的内存空间
}

void
dPrint_CompCol_Matrix(char *what, SuperMatrix *A)
{
    NCformat     *Astore;
    register int_t i;
    register int n;
    double       *dp;
    
    printf("\nCompCol matrix %s:\n", what);          // 打印输出矩阵的类型信息
    // 输出 A 结构体中的 Stype、Dtype 和 Mtype 三个成员变量的值
    printf("Stype %d, Dtype %d, Mtype %d\n", A->Stype, A->Dtype, A->Mtype);
    // 将 A 结构体中的 ncol 成员赋值给变量 n
    n = A->ncol;
    // 将 A 结构体中的 Store 成员转换为 NCformat 指针类型，并赋值给 Astore
    Astore = (NCformat *) A->Store;
    // 将 Astore 结构体中的 nzval 成员转换为 double 类型指针，并赋值给 dp
    dp = (double *) Astore->nzval;
    // 输出 A 结构体中的 nrow、ncol 和 Astore 结构体中的 nnz 三个成员变量的值
    printf("nrow %d, ncol %d, nnz %ld\n", (int)A->nrow, (int)A->ncol, (long)Astore->nnz);
    // 输出 dp 数组中的元素，即 Astore 结构体中的 nzval 数组
    printf("nzval: ");
    for (i = 0; i < Astore->colptr[n]; ++i) printf("%f  ", dp[i]);
    printf("\n");
    // 输出 Astore 结构体中的 rowind 数组
    printf("rowind: ");
    for (i = 0; i < Astore->colptr[n]; ++i) printf("%ld  ", (long)Astore->rowind[i]);
    printf("\n");
    // 输出 Astore 结构体中的 colptr 数组
    printf("colptr: ");
    for (i = 0; i <= n; ++i) printf("%ld  ", (long)Astore->colptr[i]);
    printf("\n");
    // 清空输出缓冲区，确保所有内容都输出到终端
    fflush(stdout);
void
dPrint_SuperNode_Matrix(char *what, SuperMatrix *A)
{
    SCformat     *Astore;
    register int_t i, j, k, c, d, n, nsup;
    double       *dp;
    int *col_to_sup, *sup_to_col;
    int_t *rowind, *rowind_colptr;
    
    // 打印超节点矩阵的信息，包括类型和维度信息
    printf("\nSuperNode matrix %s:\n", what);
    printf("Stype %d, Dtype %d, Mtype %d\n", A->Stype,A->Dtype,A->Mtype);
    
    // 获取矩阵的列数
    n = A->ncol;
    
    // 获取矩阵的存储格式为SCformat，并获取相关数据指针
    Astore = (SCformat *) A->Store;
    dp = (double *) Astore->nzval;
    col_to_sup = Astore->col_to_sup;
    sup_to_col = Astore->sup_to_col;
    rowind_colptr = Astore->rowind_colptr;
    rowind = Astore->rowind;
    
    // 打印矩阵的行数、列数、非零元素数以及超节点数
    printf("nrow %d, ncol %d, nnz %lld, nsuper %d\n", 
       (int)A->nrow, (int)A->ncol, (long long) Astore->nnz, (int)Astore->nsuper);
    
    // 打印矩阵的非零元素值
    printf("nzval:\n");
    for (k = 0; k <= Astore->nsuper; ++k) {
      c = sup_to_col[k];
      nsup = sup_to_col[k+1] - c;
      for (j = c; j < c + nsup; ++j) {
        d = Astore->nzval_colptr[j];
        for (i = rowind_colptr[c]; i < rowind_colptr[c+1]; ++i) {
          printf("%d\t%d\t%e\n", (int)rowind[i], (int)j, dp[d++]);
        }
      }
    }
    
    // 打印矩阵的列指针数组
#if 0
    for (i = 0; i < Astore->nzval_colptr[n]; ++i) printf("%f  ", dp[i]);
#endif
    printf("\nnzval_colptr: ");
    for (i = 0; i <= n; ++i) printf("%lld  ", (long long)Astore->nzval_colptr[i]);
    
    // 打印矩阵的行索引数组
    printf("\nrowind: ");
    for (i = 0; i < Astore->rowind_colptr[n]; ++i) 
        printf("%lld  ", (long long)Astore->rowind[i]);
    
    // 打印矩阵的行指针数组
    printf("\nrowind_colptr: ");
    for (i = 0; i <= n; ++i) printf("%lld  ", (long long)Astore->rowind_colptr[i]);
    
    // 打印矩阵的列到超节点映射数组
    printf("\ncol_to_sup: ");
    for (i = 0; i < n; ++i) printf("%d  ", col_to_sup[i]);
    
    // 打印矩阵的超节点到列映射数组
    printf("\nsup_to_col: ");
    for (i = 0; i <= Astore->nsuper+1; ++i) 
        printf("%d  ", sup_to_col[i]);
    
    // 输出空行，并刷新标准输出流
    printf("\n");
    fflush(stdout);
}

void
dPrint_Dense_Matrix(char *what, SuperMatrix *A)
{
    DNformat     *Astore = (DNformat *) A->Store;
    register int i, j, lda = Astore->lda;
    double       *dp;
    
    // 打印稠密矩阵的信息，包括类型和维度信息
    printf("\nDense matrix %s:\n", what);
    printf("Stype %d, Dtype %d, Mtype %d\n", A->Stype,A->Dtype,A->Mtype);
    
    // 获取矩阵的非零元素数组指针
    dp = (double *) Astore->nzval;
    printf("nrow %d, ncol %d, lda %d\n", (int)A->nrow, (int)A->ncol, lda);
    
    // 打印矩阵的非零元素值
    printf("\nnzval: ");
    for (j = 0; j < A->ncol; ++j) {
        for (i = 0; i < A->nrow; ++i) printf("%f  ", dp[i + j*lda]);
        printf("\n");
    }
    
    // 输出空行
    printf("\n");
    fflush(stdout);
}

/*! \brief Diagnostic print of column "jcol" in the U/L factor.
 */
void
dprint_lu_col(char *msg, int jcol, int pivrow, int_t *xprune, GlobalLU_t *Glu)
{
    int_t    i, k;
    int     *xsup, *supno, fsupc;
    int_t   *xlsub, *lsub;
    double  *lusup;
    int_t   *xlusup;
    double  *ucol;
    int_t   *usub, *xusub;

    // 从全局LU数据结构中获取必要的数组指针
    xsup    = Glu->xsup;
    supno   = Glu->supno;
    lsub    = Glu->lsub;
    xlsub   = Glu->xlsub;
    lusup   = (double *) Glu->lusup;
    xlusup  = Glu->xlusup;
    ucol    = (double *) Glu->ucol;
    usub    = Glu->usub;
    xusub   = Glu->xusub;
    
    // 打印消息字符串
    printf("%s", msg);
    //
    ```cpp`
        # 打印格式化输出，显示列号 jcol、主元行号 pivrow、supno[jcol]的超节点号、xprune[jcol]的长长整型值
        printf("col %d: pivrow %d, supno %d, xprune %lld\n", 
               jcol, pivrow, supno[jcol], (long long) xprune[jcol]);
    
        # 打印缩进后的信息，表示这是 U 列
        printf("\tU-col:\n");
    
        # 遍历 xusub[jcol] 到 xusub[jcol+1] 之间的索引，打印每个索引对应的值 (int)usub[i] 和 ucol[i]
        for (i = xusub[jcol]; i < xusub[jcol+1]; i++)
            printf("\t%d%10.4f\n", (int)usub[i], ucol[i]);
    
        # 打印缩进后的信息，表示这是在矩形超节点内的 L 列
        printf("\tL-col in rectangular snode:\n");
    
        # 获取超节点 snode 的第一列 fsupc
        fsupc = xsup[supno[jcol]];
    
        # 初始化 i 和 k，分别为超节点 snode 的首列 xlsub[fsupc] 和当前列 jcol 的首列 xlusup[jcol]
        i = xlsub[fsupc];
        k = xlusup[jcol];
    
        # 当 i 在超节点 snode 的列索引范围内且 k 在当前列 jcol 的超节点列索引范围内时，循环执行
        while ( i < xlsub[fsupc+1] && k < xlusup[jcol+1] ) {
            # 打印 lsub[i] 和 lusup[k] 的值
            printf("\t%d\t%10.4f\n", (int)lsub[i], lusup[k]);
            # 同时递增 i 和 k
            i++; k++;
        }
    
        # 刷新标准输出缓冲区
        fflush(stdout);
/*! \brief Check whether tempv[] == 0. This should be true before and after calling any numeric routines, i.e., "panel_bmod" and "column_bmod". 
 */
void dcheck_tempv(int n, double *tempv)
{
    int i;
    
    // 遍历数组 tempv，检查是否所有元素都为 0
    for (i = 0; i < n; i++) {
        if (tempv[i] != 0.0) 
        {
            // 如果发现非零元素，输出错误信息并中止程序
            fprintf(stderr,"tempv[%d] = %f\n", i,tempv[i]);
            ABORT("dcheck_tempv");
        }
    }
}


void
dGenXtrue(int n, int nrhs, double *x, int ldx)
{
    int  i, j;
    // 循环填充二维数组 x，每列的元素都设为 1.0
    for (j = 0; j < nrhs; ++j)
        for (i = 0; i < n; ++i) {
            x[i + j*ldx] = 1.0;
        }
}

/*! \brief Let rhs[i] = sum of i-th row of A, so the solution vector is all 1's
 */
void
dFillRHS(trans_t trans, int nrhs, double *x, int ldx,
         SuperMatrix *A, SuperMatrix *B)
{
    DNformat *Bstore;
    double   *rhs;
    double one = 1.0;
    double zero = 0.0;
    int      ldc;
    char transc[1];

    Bstore = B->Store;
    rhs    = Bstore->nzval;
    ldc    = Bstore->lda;
    
    // 根据传入的 trans 参数确定是否转置操作
    if ( trans == NOTRANS ) *(unsigned char *)transc = 'N';
    else *(unsigned char *)transc = 'T';

    // 调用稀疏矩阵乘法函数 sp_dgemm 计算 rhs = A * x
    sp_dgemm(transc, "N", A->nrow, nrhs, A->ncol, one, A,
         x, ldx, zero, rhs, ldc);

}

/*! \brief Fills a double precision array with a given value.
 */
void 
dfill(double *a, int alen, double dval)
{
    register int i;
    // 循环将数组 a 的每个元素设为给定的值 dval
    for (i = 0; i < alen; i++) a[i] = dval;
}



/*! \brief Check the inf-norm of the error vector 
 */
void dinf_norm_error(int nrhs, SuperMatrix *X, double *xtrue)
{
    DNformat *Xstore;
    double err, xnorm;
    double *Xmat, *soln_work;
    int i, j;

    Xstore = X->Store;
    Xmat = Xstore->nzval;

    // 计算解向量 X 与真实解 xtrue 之间的无穷范数误差
    for (j = 0; j < nrhs; j++) {
      soln_work = &Xmat[j*Xstore->lda];
      err = xnorm = 0.0;
      for (i = 0; i < X->nrow; i++) {
        err = SUPERLU_MAX(err, fabs(soln_work[i] - xtrue[i]));
        xnorm = SUPERLU_MAX(xnorm, fabs(soln_work[i]));
      }
      err = err / xnorm;
      printf("||X - Xtrue||/||X|| = %e\n", err);
    }
}



/*! \brief Print performance of the code. */
void
dPrintPerf(SuperMatrix *L, SuperMatrix *U, mem_usage_t *mem_usage,
           double rpg, double rcond, double *ferr,
           double *berr, char *equed, SuperLUStat_t *stat)
{
    SCformat *Lstore;
    NCformat *Ustore;
    double   *utime;
    flops_t  *ops;
    
    utime = stat->utime;
    ops   = stat->ops;
    
    // 打印因子分解的性能信息
    if ( utime[FACT] != 0. )
        printf("Factor flops = %e\tMflops = %8.2f\n", ops[FACT],
               ops[FACT]*1e-6/utime[FACT]);
    printf("Identify relaxed snodes    = %8.2f\n", utime[RELAX]);
    // 打印解算过程的性能信息
    if ( utime[SOLVE] != 0. )
        printf("Solve flops = %.0f, Mflops = %8.2f\n", ops[SOLVE],
               ops[SOLVE]*1e-6/utime[SOLVE]);
    
    // 获取因子 L 和 U 的非零元素数目并打印
    Lstore = (SCformat *) L->Store;
    Ustore = (NCformat *) U->Store;
    printf("\tNo of nonzeros in factor L = %lld\n", (long long) Lstore->nnz);
    printf("\tNo of nonzeros in factor U = %lld\n", (long long) Ustore->nnz);

}
    # 输出 L 和 U 的非零元素数量
    printf("\tNo of nonzeros in L+U = %lld\n", (long long) Lstore->nnz + Ustore->nnz);
    
    # 输出 L 和 U 的内存使用情况和总内存需求
    printf("L\\U MB %.3f\ttotal MB needed %.3f\n",
       mem_usage->for_lu/1e6, mem_usage->total_needed/1e6);
    
    # 输出内存扩展的次数
    printf("Number of memory expansions: %d\n", stat->expansions);
    
    # 输出性能指标的表头
    printf("\tFactor\tMflops\tSolve\tMflops\tEtree\tEquil\tRcond\tRefine\n");
    
    # 输出性能指标的具体数值
    printf("PERF:%8.2f%8.2f%8.2f%8.2f%8.2f%8.2f%8.2f%8.2f\n",
       utime[FACT], ops[FACT]*1e-6/utime[FACT],
       utime[SOLVE], ops[SOLVE]*1e-6/utime[SOLVE],
       utime[ETREE], utime[EQUIL], utime[RCOND], utime[REFINE]);
    
    # 输出数值计算相关的指标
    printf("\tRpg\t\tRcond\t\tFerr\t\tBerr\t\tEquil?\n");
    printf("NUM:\t%e\t%e\t%e\t%e\t%s\n",
       rpg, rcond, ferr[0], berr[0], equed);
}

int
print_double_vec(char *what, int n, double *vec)
{
    // 输出传入的标识字符串和向量的长度
    printf("%s: n %d\n", what, n);
    // 遍历并输出向量中每个元素的索引和数值
    for (int i = 0; i < n; ++i) printf("%d\t%f\n", i, vec[i]);
    // 返回 0 表示函数执行成功
    return 0;
}
```