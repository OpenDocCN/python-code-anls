# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\cutil.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file cutil.c
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
#include "slu_cdefs.h"

/*! \brief Create a sparse matrix in compressed column format. */
void
cCreate_CompCol_Matrix(SuperMatrix *A, int m, int n, int_t nnz, 
               singlecomplex *nzval, int_t *rowind, int_t *colptr,
               Stype_t stype, Dtype_t dtype, Mtype_t mtype)
{
    NCformat *Astore;

    // 设置稀疏矩阵的存储类型、数据类型和存储模式
    A->Stype = stype;
    A->Dtype = dtype;
    A->Mtype = mtype;
    A->nrow = m;
    A->ncol = n;
    
    // 分配存储空间并初始化存储格式为 NCformat
    A->Store = (void *) SUPERLU_MALLOC( sizeof(NCformat) );
    if ( !(A->Store) ) ABORT("SUPERLU_MALLOC fails for A->Store");
    Astore = A->Store;
    Astore->nnz = nnz;
    Astore->nzval = nzval;
    Astore->rowind = rowind;
    Astore->colptr = colptr;
}

/*! \brief Create a sparse matrix in compressed row format. */
void
cCreate_CompRow_Matrix(SuperMatrix *A, int m, int n, int_t nnz, 
               singlecomplex *nzval, int_t *colind, int_t *rowptr,
               Stype_t stype, Dtype_t dtype, Mtype_t mtype)
{
    NRformat *Astore;

    // 设置稀疏矩阵的存储类型、数据类型和存储模式
    A->Stype = stype;
    A->Dtype = dtype;
    A->Mtype = mtype;
    A->nrow = m;
    A->ncol = n;
    
    // 分配存储空间并初始化存储格式为 NRformat
    A->Store = (void *) SUPERLU_MALLOC( sizeof(NRformat) );
    if ( !(A->Store) ) ABORT("SUPERLU_MALLOC fails for A->Store");
    Astore = A->Store;
    Astore->nnz = nnz;
    Astore->nzval = nzval;
    Astore->colind = colind;
    Astore->rowptr = rowptr;
}

/*! \brief Copy matrix A into matrix B. */
void
cCopy_CompCol_Matrix(SuperMatrix *A, SuperMatrix *B)
{
    NCformat *Astore, *Bstore;
    int      ncol, nnz, i;

    // 复制矩阵 A 的属性到矩阵 B
    B->Stype = A->Stype;
    B->Dtype = A->Dtype;
    B->Mtype = A->Mtype;
    B->nrow  = A->nrow;;
    B->ncol  = ncol = A->ncol;
    
    // 获取矩阵 A 的存储格式为 NCformat
    Astore   = (NCformat *) A->Store;
    Bstore   = (NCformat *) B->Store;
    Bstore->nnz = nnz = Astore->nnz;
    
    // 复制非零元素的值、行索引和列指针
    for (i = 0; i < nnz; ++i)
        ((singlecomplex *)Bstore->nzval)[i] = ((singlecomplex *)Astore->nzval)[i];
    for (i = 0; i < nnz; ++i)
        Bstore->rowind[i] = Astore->rowind[i];
    for (i = 0; i <= ncol; ++i)
        Bstore->colptr[i] = Astore->colptr[i];
}
/* 创建稠密矩阵，存储在 SuperMatrix *X 中 */
void
cCreate_Dense_Matrix(SuperMatrix *X, int m, int n, singlecomplex *x, int ldx,
            Stype_t stype, Dtype_t dtype, Mtype_t mtype)
{
    DNformat    *Xstore;
    
    X->Stype = stype;  // 设置稠密矩阵类型
    X->Dtype = dtype;  // 设置数据类型
    X->Mtype = mtype;  // 设置存储类型
    X->nrow = m;       // 设置行数
    X->ncol = n;       // 设置列数
    X->Store = (void *) SUPERLU_MALLOC( sizeof(DNformat) );  // 分配存储空间
    if ( !(X->Store) ) ABORT("SUPERLU_MALLOC fails for X->Store");  // 失败处理
    Xstore = (DNformat *) X->Store;  // 强制类型转换为 DNformat
    Xstore->lda = ldx;  // 设置主存储数组的列数
    Xstore->nzval = (singlecomplex *) x;  // 将数据指针赋值给 nzval
}

/* 复制稠密矩阵 X 到 Y */
void
cCopy_Dense_Matrix(int M, int N, singlecomplex *X, int ldx,
            singlecomplex *Y, int ldy)
{
    int    i, j;
    
    for (j = 0; j < N; ++j)
        for (i = 0; i < M; ++i)
            Y[i + j*ldy] = X[i + j*ldx];  // 复制 X 到 Y
}

/* 创建超节点矩阵，存储在 SuperMatrix *L 中 */
void
cCreate_SuperNode_Matrix(SuperMatrix *L, int m, int n, int_t nnz, 
            singlecomplex *nzval, int_t *nzval_colptr, int_t *rowind,
            int_t *rowind_colptr, int *col_to_sup, int *sup_to_col,
            Stype_t stype, Dtype_t dtype, Mtype_t mtype)
{
    SCformat *Lstore;

    L->Stype = stype;  // 设置超节点矩阵类型
    L->Dtype = dtype;  // 设置数据类型
    L->Mtype = mtype;  // 设置存储类型
    L->nrow = m;       // 设置行数
    L->ncol = n;       // 设置列数
    L->Store = (void *) SUPERLU_MALLOC( sizeof(SCformat) );  // 分配存储空间
    if ( !(L->Store) ) ABORT("SUPERLU_MALLOC fails for L->Store");  // 失败处理
    Lstore = L->Store;  // 强制类型转换为 SCformat
    Lstore->nnz = nnz;  // 设置非零元素个数
    Lstore->nsuper = col_to_sup[n];  // 设置超节点数目
    Lstore->nzval = nzval;  // 非零元素值数组
    Lstore->nzval_colptr = nzval_colptr;  // 非零元素列指针
    Lstore->rowind = rowind;  // 行索引数组
    Lstore->rowind_colptr = rowind_colptr;  // 行索引列指针
    Lstore->col_to_sup = col_to_sup;  // 列到超节点的映射
    Lstore->sup_to_col = sup_to_col;  // 超节点到列的映射
}

/*! \brief 将行压缩存储的稀疏矩阵转换为列压缩存储形式 */
void
cCompRow_to_CompCol(int m, int n, int_t nnz, 
            singlecomplex *a, int_t *colind, int_t *rowptr,
            singlecomplex **at, int_t **rowind, int_t **colptr)
{
    register int i, j, col, relpos;
    int_t *marker;

    /* 为另一份矩阵分配存储空间 */
    *at = (singlecomplex *) complexMalloc(nnz);  // 分配非零元素值存储空间
    *rowind = (int_t *) intMalloc(nnz);  // 分配行索引存储空间
    *colptr = (int_t *) intMalloc(n+1);  // 分配列指针存储空间
    marker = (int_t *) intCalloc(n);  // 分配列标记数组空间
    
    /* 统计每列的元素个数，并设置列指针 */
    for (i = 0; i < m; ++i)
        for (j = rowptr[i]; j < rowptr[i+1]; ++j) ++marker[colind[j]];
    (*colptr)[0] = 0;
    for (j = 0; j < n; ++j) {
        (*colptr)[j+1] = (*colptr)[j] + marker[j];
        marker[j] = (*colptr)[j];
    }

    /* 将矩阵转换为压缩列存储形式 */
    for (i = 0; i < m; ++i) {
        for (j = rowptr[i]; j < rowptr[i+1]; ++j) {
            col = colind[j];
            relpos = marker[col];
            (*rowind)[relpos] = i;
            (*at)[relpos] = a[j];
            ++marker[col];
        }
    }

    SUPERLU_FREE(marker);  // 释放标记数组的内存
}

/* 打印压缩列存储格式的矩阵 */
void
cPrint_CompCol_Matrix(char *what, SuperMatrix *A)
{
    NCformat     *Astore;
    register int_t i;
    register int n;
    float       *dp;
    
    printf("\nCompCol matrix %s:\n", what);
}
    // 打印稀疏矩阵 A 的存储类型 (Stype)、数据类型 (Dtype) 和存储方式 (Mtype)
    printf("Stype %d, Dtype %d, Mtype %d\n", A->Stype,A->Dtype,A->Mtype);
    // 将矩阵列数赋给变量 n
    n = A->ncol;
    // 将 A 的存储格式转换为 NCformat 类型，并获取非零元素的值
    Astore = (NCformat *) A->Store;
    // 将非零元素数组的指针赋给 dp，假设其类型为 float
    dp = (float *) Astore->nzval;
    // 打印矩阵的行数 (nrow)、列数 (ncol) 和非零元素个数 (nnz)
    printf("nrow %d, ncol %d, nnz %ld\n", (int)A->nrow, (int)A->ncol, (long)Astore->nnz);
    // 打印非零元素数组 nzval 的内容
    printf("nzval: ");
    // 遍历并打印 dp 中前 2*n 个元素的值
    for (i = 0; i < 2*Astore->colptr[n]; ++i) printf("%f  ", dp[i]);
    printf("\nrowind: ");
    // 打印行索引数组 rowind 的内容
    // 遍历并打印前 n 列的行索引值
    for (i = 0; i < Astore->colptr[n]; ++i) printf("%ld  ", (long)Astore->rowind[i]);
    printf("\ncolptr: ");
    // 打印列指针数组 colptr 的内容
    // 遍历并打印前 n+1 个元素的值，表示每列非零元素在 nzval 中的起始位置
    for (i = 0; i <= n; ++i) printf("%ld  ", (long)Astore->colptr[i]);
    printf("\n");
    // 刷新标准输出缓冲区，确保输出立即显示
    fflush(stdout);
/*! \brief 打印超节点矩阵的信息和数据。
 *
 * 该函数用于打印给定超节点矩阵的各种信息和数据。
 *
 * \param what 矩阵的描述字符串，用于标识打印的是哪个矩阵。
 * \param A 指向 SuperMatrix 结构体的指针，表示要打印的超节点矩阵。
 */
void
cPrint_SuperNode_Matrix(char *what, SuperMatrix *A)
{
    SCformat     *Astore;       /* 存储超节点矩阵的 SCformat 结构体指针 */
    register int_t i, j, k, c, d, n, nsup;  /* 寄存器变量声明 */
    float       *dp;            /* 浮点数类型指针，指向非零元素数组 */
    int *col_to_sup, *sup_to_col;  /* 列到超节点映射和超节点到列映射的整型数组指针 */
    int_t *rowind, *rowind_colptr; /* 行索引数组和行索引列指针数组的整型数组指针 */

    printf("\nSuperNode matrix %s:\n", what);  /* 打印矩阵名称 */
    printf("Stype %d, Dtype %d, Mtype %d\n", A->Stype,A->Dtype,A->Mtype);  /* 打印矩阵类型信息 */
    n = A->ncol;  /* 获取矩阵的列数 */
    Astore = (SCformat *) A->Store;  /* 强制类型转换为 SCformat 结构体指针 */
    dp = (float *) Astore->nzval;   /* 指向非零元素数组的浮点数指针 */
    col_to_sup = Astore->col_to_sup;  /* 指向列到超节点映射数组的指针 */
    sup_to_col = Astore->sup_to_col;  /* 指向超节点到列映射数组的指针 */
    rowind_colptr = Astore->rowind_colptr;  /* 指向行索引列指针数组的指针 */
    rowind = Astore->rowind;    /* 指向行索引数组的指针 */
    printf("nrow %d, ncol %d, nnz %lld, nsuper %d\n", 
           (int)A->nrow, (int)A->ncol, (long long) Astore->nnz, (int)Astore->nsuper);  /* 打印矩阵的行数、列数、非零元素数和超节点数 */
    printf("nzval:\n");
    for (k = 0; k <= Astore->nsuper; ++k) {   /* 遍历超节点 */
        c = sup_to_col[k];   /* 获取超节点对应的起始列 */
        nsup = sup_to_col[k+1] - c;   /* 计算超节点的列数 */
        for (j = c; j < c + nsup; ++j) {   /* 遍历超节点的每一列 */
            d = Astore->nzval_colptr[j];   /* 获取非零元素列指针 */
            for (i = rowind_colptr[c]; i < rowind_colptr[c+1]; ++i) {   /* 遍历当前列的行索引 */
                printf("%d\t%d\t%e\t%e\n", (int)rowind[i], (int) j, dp[d], dp[d+1]);   /* 打印行索引、列索引、非零元素值 */
                d += 2;    /* 移动到下一个非零元素的位置 */
            }
        }
    }
#if 0
    for (i = 0; i < 2*Astore->nzval_colptr[n]; ++i) printf("%f  ", dp[i]);
#endif
    printf("\nnzval_colptr: ");
    for (i = 0; i <= n; ++i) printf("%lld  ", (long long)Astore->nzval_colptr[i]);   /* 打印非零元素列指针 */
    printf("\nrowind: ");
    for (i = 0; i < Astore->rowind_colptr[n]; ++i) 
        printf("%lld  ", (long long)Astore->rowind[i]);   /* 打印行索引 */
    printf("\nrowind_colptr: ");
    for (i = 0; i <= n; ++i) printf("%lld  ", (long long)Astore->rowind_colptr[i]);   /* 打印行索引列指针 */
    printf("\ncol_to_sup: ");
    for (i = 0; i < n; ++i) printf("%d  ", col_to_sup[i]);   /* 打印列到超节点映射 */
    printf("\nsup_to_col: ");
    for (i = 0; i <= Astore->nsuper+1; ++i) 
        printf("%d  ", sup_to_col[i]);   /* 打印超节点到列映射 */
    printf("\n");
    fflush(stdout);   /* 清空输出缓冲区，确保输出立即可见 */
}
    xusub   = Glu->xusub;
    // 从全局结构体 Glu 中获取 xusub 指针，用于访问 U 非零元素行索引

    printf("%s", msg);
    // 打印字符串 msg

    printf("col %d: pivrow %d, supno %d, xprune %lld\n", 
       jcol, pivrow, supno[jcol], (long long) xprune[jcol]);
    // 打印格式化字符串，输出列 jcol 的信息：主元行号 pivrow，超节点编号 supno[jcol]，以及 xprune[jcol] 的值

    printf("\tU-col:\n");
    // 打印 U 列的标题

    for (i = xusub[jcol]; i < xusub[jcol+1]; i++)
    // 遍历列 jcol 在 U 的非零元素索引范围内
    printf("\t%d%10.4f, %10.4f\n", (int)usub[i], ucol[i].r, ucol[i].i);
    // 格式化打印非零元素的行索引 usub[i]，实部 ucol[i].r 和虚部 ucol[i].i

    printf("\tL-col in rectangular snode:\n");
    // 打印矩形超节点中 L 列的标题
    fsupc = xsup[supno[jcol]];    /* first col of the snode */
    // 获取超节点 supno[jcol] 的第一列在全局列索引 xsup 中的位置
    i = xlsub[fsupc];
    k = xlusup[jcol];
    // 初始化 i 和 k 为列 jcol 的 L 和 U 非零元素的索引
    while ( i < xlsub[fsupc+1] && k < xlusup[jcol+1] ) {
    // 当 L 和 U 列的索引仍在范围内时执行循环
    printf("\t%d\t%10.4f, %10.4f\n", (int)lsub[i], lusup[k].r, lusup[k].i);
    // 格式化打印 L 列的行索引 lsub[i]，实部 lusup[k].r 和虚部 lusup[k].i
    i++; k++;
    // 更新 L 和 U 列的索引
    }
    fflush(stdout);
    // 清空输出缓冲区，确保输出即时显示
/*! \brief Check whether tempv[] == 0. This should be true before and after calling any numeric routines, i.e., "panel_bmod" and "column_bmod". 
 */
void ccheck_tempv(int n, singlecomplex *tempv)
{
    int i;
    
    // 遍历 tempv 数组
    for (i = 0; i < n; i++) {
        // 如果 tempv[i] 不为零，则输出错误信息并终止程序
        if ((tempv[i].r != 0.0) || (tempv[i].i != 0.0))
        {
            fprintf(stderr,"tempv[%d] = {%f, %f}\n", i, tempv[i].r, tempv[i].i);
            ABORT("ccheck_tempv");
        }
    }
}


void
cGenXtrue(int n, int nrhs, singlecomplex *x, int ldx)
{
    int  i, j;
    // 遍历每个解向量的列
    for (j = 0; j < nrhs; ++j)
        // 遍历当前列中的每个元素
        for (i = 0; i < n; ++i) {
            // 设置解向量的实部为 1.0，虚部为 0.0
            x[i + j*ldx].r = 1.0;
            x[i + j*ldx].i = 0.0;
        }
}

/*! \brief Let rhs[i] = sum of i-th row of A, so the solution vector is all 1's
 */
void
cFillRHS(trans_t trans, int nrhs, singlecomplex *x, int ldx,
         SuperMatrix *A, SuperMatrix *B)
{
    DNformat *Bstore;
    singlecomplex   *rhs;
    singlecomplex one = {1.0, 0.0};
    singlecomplex zero = {0.0, 0.0};
    int      ldc;
    char transc[1];

    // 获取 B 矩阵的存储格式
    Bstore = B->Store;
    // 获取右端向量 rhs 的起始地址
    rhs    = Bstore->nzval;
    // 获取 B 矩阵的列偏移量
    ldc    = Bstore->lda;
    
    // 根据 trans 设置字符 transc
    if ( trans == NOTRANS ) *(unsigned char *)transc = 'N';
    else *(unsigned char *)transc = 'T';

    // 执行复杂矩阵乘法运算
    sp_cgemm(transc, "N", A->nrow, nrhs, A->ncol, one, A,
         x, ldx, zero, rhs, ldc);

}

/*! \brief Fills a complex precision array with a given value.
 */
void 
cfill(singlecomplex *a, int alen, singlecomplex dval)
{
    register int i;
    // 遍历数组 a，并赋值为 dval
    for (i = 0; i < alen; i++) a[i] = dval;
}

/*! \brief Check the inf-norm of the error vector 
 */
void cinf_norm_error(int nrhs, SuperMatrix *X, singlecomplex *xtrue)
{
    DNformat *Xstore;
    float err, xnorm;
    singlecomplex *Xmat, *soln_work;
    singlecomplex temp;
    int i, j;

    // 获取 X 矩阵的存储格式
    Xstore = X->Store;
    // 获取 X 矩阵的数值部分
    Xmat = Xstore->nzval;

    // 遍历每个解向量的列
    for (j = 0; j < nrhs; j++) {
        // 获取当前列的起始地址
        soln_work = &Xmat[j*Xstore->lda];
        // 初始化误差和范数
        err = xnorm = 0.0;
        // 遍历当前列的每个元素
        for (i = 0; i < X->nrow; i++) {
            // 计算误差向量的元素与真实解向量的元素之差
            c_sub(&temp, &soln_work[i], &xtrue[i]);
            // 更新误差的无穷范数
            err = SUPERLU_MAX(err, c_abs(&temp));
            // 更新解向量的无穷范数
            xnorm = SUPERLU_MAX(xnorm, c_abs(&soln_work[i]));
        }
        // 计算误差相对于解向量的无穷范数的比值，并输出
        err = err / xnorm;
        printf("||X - Xtrue||/||X|| = %e\n", err);
    }
}

/*! \brief Print performance of the code. */
void
cPrintPerf(SuperMatrix *L, SuperMatrix *U, mem_usage_t *mem_usage,
           float rpg, float rcond, float *ferr,
           float *berr, char *equed, SuperLUStat_t *stat)
{
    SCformat *Lstore;
    NCformat *Ustore;
    double   *utime;
    flops_t  *ops;
    
    // 获取统计信息中的计时和操作数
    utime = stat->utime;
    ops   = stat->ops;
    
    // 如果因子分解阶段的运行时间不为零，输出因子运算的浮点运算量和性能
    if ( utime[FACT] != 0. )
        printf("Factor flops = %e\tMflops = %8.2f\n", ops[FACT],
               ops[FACT]*1e-6/utime[FACT]);
    
    // 输出松弛结点识别的运行时间
    printf("Identify relaxed snodes    = %8.2f\n", utime[RELAX]);
    
    // 如果求解阶段的运行时间不为零，输出求解操作的浮点运算量和性能
    if ( utime[SOLVE] != 0. )
        printf("Solve flops = %.0f, Mflops = %8.2f\n", ops[SOLVE],
               ops[SOLVE]*1e-6/utime[SOLVE]);
    
    // 获取 L 和 U 矩阵的存储格式
    Lstore = (SCformat *) L->Store;
    Ustore = (NCformat *) U->Store;
}
    // 打印因子 L 中的非零元素数量
    printf("\tNo of nonzeros in factor L = %lld\n", (long long) Lstore->nnz);
    // 打印因子 U 中的非零元素数量
    printf("\tNo of nonzeros in factor U = %lld\n", (long long) Ustore->nnz);
    // 打印因子 L 和因子 U 中非零元素的总数量
    printf("\tNo of nonzeros in L+U = %lld\n", (long long) Lstore->nnz + Ustore->nnz);
    
    // 打印 L/U 使用的内存量（以兆字节为单位）和总共需要的内存量（以兆字节为单位）
    printf("L\\U MB %.3f\ttotal MB needed %.3f\n",
       mem_usage->for_lu/1e6, mem_usage->total_needed/1e6);
    // 打印内存扩展的次数
    printf("Number of memory expansions: %d\n", stat->expansions);
    
    // 打印性能数据的表头
    printf("\tFactor\tMflops\tSolve\tMflops\tEtree\tEquil\tRcond\tRefine\n");
    // 打印性能数据：因子化的时间、每秒执行的浮点运算数（MFLOPS）、解决方案的时间、每秒执行的浮点运算数（MFLOPS）、消耗的时间等等
    printf("PERF:%8.2f%8.2f%8.2f%8.2f%8.2f%8.2f%8.2f%8.2f\n",
       utime[FACT], ops[FACT]*1e-6/utime[FACT],
       utime[SOLVE], ops[SOLVE]*1e-6/utime[SOLVE],
       utime[ETREE], utime[EQUIL], utime[RCOND], utime[REFINE]);
    
    // 打印 Rpg、Rcond、Ferr、Berr 和是否使用了均衡的相关信息
    printf("\tRpg\t\tRcond\t\tFerr\t\tBerr\t\tEquil?\n");
    printf("NUM:\t%e\t%e\t%e\t%e\t%s\n",
       rpg, rcond, ferr[0], berr[0], equed);
}
```