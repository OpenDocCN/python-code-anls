# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\sutil.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file sutil.c
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
#include "slu_sdefs.h"

/*! \brief Create a compressed column matrix A.
 *
 *  Initializes the attributes of the SuperMatrix A to represent a compressed
 *  column matrix format.
 *
 *  \param A SuperMatrix pointer to the matrix to be initialized.
 *  \param m Number of rows in the matrix.
 *  \param n Number of columns in the matrix.
 *  \param nnz Number of nonzeros in the matrix.
 *  \param nzval Array of nonzero values.
 *  \param rowind Array of row indices for each nonzero.
 *  \param colptr Array of column pointers indicating where each column starts in nzval.
 *  \param stype Storage type of the matrix (e.g., SLU_NC).
 *  \param dtype Data type of the matrix (e.g., SLU_S).
 *  \param mtype Memory type of the matrix (e.g., SLU_GE).
 */
void
sCreate_CompCol_Matrix(SuperMatrix *A, int m, int n, int_t nnz, 
               float *nzval, int_t *rowind, int_t *colptr,
               Stype_t stype, Dtype_t dtype, Mtype_t mtype)
{
    NCformat *Astore;

    A->Stype = stype;            // 设置矩阵的存储类型
    A->Dtype = dtype;            // 设置矩阵的数据类型
    A->Mtype = mtype;            // 设置矩阵的内存类型
    A->nrow = m;                 // 设置矩阵的行数
    A->ncol = n;                 // 设置矩阵的列数
    A->Store = (void *) SUPERLU_MALLOC( sizeof(NCformat) );  // 分配内存用于存储矩阵数据结构
    if ( !(A->Store) ) ABORT("SUPERLU_MALLOC fails for A->Store");  // 如果内存分配失败，则终止程序
    Astore = A->Store;           // 获取矩阵存储结构体的指针
    Astore->nnz = nnz;           // 设置矩阵非零元素的个数
    Astore->nzval = nzval;       // 设置非零元素值的数组
    Astore->rowind = rowind;     // 设置行索引的数组
    Astore->colptr = colptr;     // 设置列指针的数组
}

/*! \brief Create a compressed row matrix A.
 *
 *  Initializes the attributes of the SuperMatrix A to represent a compressed
 *  row matrix format.
 *
 *  \param A SuperMatrix pointer to the matrix to be initialized.
 *  \param m Number of rows in the matrix.
 *  \param n Number of columns in the matrix.
 *  \param nnz Number of nonzeros in the matrix.
 *  \param nzval Array of nonzero values.
 *  \param colind Array of column indices for each nonzero.
 *  \param rowptr Array of row pointers indicating where each row starts in nzval.
 *  \param stype Storage type of the matrix (e.g., SLU_NR).
 *  \param dtype Data type of the matrix (e.g., SLU_S).
 *  \param mtype Memory type of the matrix (e.g., SLU_GE).
 */
void
sCreate_CompRow_Matrix(SuperMatrix *A, int m, int n, int_t nnz, 
               float *nzval, int_t *colind, int_t *rowptr,
               Stype_t stype, Dtype_t dtype, Mtype_t mtype)
{
    NRformat *Astore;

    A->Stype = stype;            // 设置矩阵的存储类型
    A->Dtype = dtype;            // 设置矩阵的数据类型
    A->Mtype = mtype;            // 设置矩阵的内存类型
    A->nrow = m;                 // 设置矩阵的行数
    A->ncol = n;                 // 设置矩阵的列数
    A->Store = (void *) SUPERLU_MALLOC( sizeof(NRformat) );  // 分配内存用于存储矩阵数据结构
    if ( !(A->Store) ) ABORT("SUPERLU_MALLOC fails for A->Store");  // 如果内存分配失败，则终止程序
    Astore = A->Store;           // 获取矩阵存储结构体的指针
    Astore->nnz = nnz;           // 设置矩阵非零元素的个数
    Astore->nzval = nzval;       // 设置非零元素值的数组
    Astore->colind = colind;     // 设置列索引的数组
    Astore->rowptr = rowptr;     // 设置行指针的数组
}

/*! \brief Copy matrix A into matrix B.
 *
 *  Copies the contents of matrix A into matrix B. Assumes A and B have the same
 *  structure and allocated memory.
 *
 *  \param A SuperMatrix pointer to the source matrix.
 *  \param B SuperMatrix pointer to the destination matrix.
 */
void
sCopy_CompCol_Matrix(SuperMatrix *A, SuperMatrix *B)
{
    NCformat *Astore, *Bstore;
    int      ncol, nnz, i;

    B->Stype = A->Stype;         // 复制源矩阵的存储类型到目标矩阵
    B->Dtype = A->Dtype;         // 复制源矩阵的数据类型到目标矩阵
    B->Mtype = A->Mtype;         // 复制源矩阵的内存类型到目标矩阵
    B->nrow  = A->nrow;;         // 复制源矩阵的行数到目标矩阵
    B->ncol  = ncol = A->ncol;   // 复制源矩阵的列数到目标矩阵
    Astore   = (NCformat *) A->Store;  // 获取源矩阵存储结构体的指针
    Bstore   = (NCformat *) B->Store;  // 获取目标矩阵存储结构体的指针
    Bstore->nnz = nnz = Astore->nnz;   // 复制源矩阵非零元素个数到目标矩阵
    for (i = 0; i < nnz; ++i)
        ((float *)Bstore->nzval)[i] = ((float *)Astore->nzval)[i];  // 复制非零元素值
    for (i = 0; i < nnz; ++i)
        Bstore->rowind[i] = Astore->rowind[i];  // 复制行索引
    for (i = 0; i <= ncol; ++i)
        Bstore->colptr[i] = Astore->colptr[i];  // 复制列指针
}
/*! \brief Creates a dense matrix in SuperLU format.
 *
 *  Initializes the fields of the SuperMatrix structure to represent a dense matrix.
 *  Sets the matrix type, dimensions, and allocates memory for storing the matrix data.
 *
 *  \param X       Pointer to the SuperMatrix structure to be initialized.
 *  \param m       Number of rows in the matrix.
 *  \param n       Number of columns in the matrix.
 *  \param x       Pointer to the array storing the matrix elements.
 *  \param ldx     Leading dimension of the matrix x.
 *  \param stype   Indicates the storage type of the matrix (e.g., SLU_DN for dense).
 *  \param dtype   Indicates the data type of the matrix elements (e.g., SLU_S for single precision float).
 *  \param mtype   Indicates the mathematical type of the matrix (e.g., SLU_GE for a general matrix).
 */
void
sCreate_Dense_Matrix(SuperMatrix *X, int m, int n, float *x, int ldx,
            Stype_t stype, Dtype_t dtype, Mtype_t mtype)
{
    DNformat    *Xstore;
    
    // Initialize the fields of the SuperMatrix structure
    X->Stype = stype;
    X->Dtype = dtype;
    X->Mtype = mtype;
    X->nrow = m;
    X->ncol = n;

    // Allocate memory for the internal storage format of the dense matrix
    X->Store = (void *) SUPERLU_MALLOC( sizeof(DNformat) );
    if ( !(X->Store) ) ABORT("SUPERLU_MALLOC fails for X->Store");
    Xstore = (DNformat *) X->Store;
    Xstore->lda = ldx;  // Set the leading dimension of the matrix
    Xstore->nzval = (float *) x;  // Point to the array holding the matrix elements
}

/*! \brief Copies a dense matrix X to another matrix Y.
 *
 *  Copies a dense matrix stored in a one-dimensional array X to another
 *  matrix Y, also stored in a one-dimensional array.
 *
 *  \param M       Number of rows in the matrix.
 *  \param N       Number of columns in the matrix.
 *  \param X       Pointer to the source matrix stored in a one-dimensional array.
 *  \param ldx     Leading dimension of the source matrix X.
 *  \param Y       Pointer to the destination matrix stored in a one-dimensional array.
 *  \param ldy     Leading dimension of the destination matrix Y.
 */
void
sCopy_Dense_Matrix(int M, int N, float *X, int ldx,
            float *Y, int ldy)
{
    int    i, j;
    
    // Copy each element from matrix X to matrix Y
    for (j = 0; j < N; ++j)
        for (i = 0; i < M; ++i)
            Y[i + j*ldy] = X[i + j*ldx];
}

/*! \brief Creates a supernodal matrix in SuperLU format.
 *
 *  Initializes the fields of the SuperMatrix structure to represent a supernodal matrix.
 *  Sets the matrix type, dimensions, and allocates memory for storing the matrix data.
 *
 *  \param L           Pointer to the SuperMatrix structure to be initialized.
 *  \param m           Number of rows in the matrix.
 *  \param n           Number of columns in the matrix.
 *  \param nnz         Number of nonzeros in the matrix.
 *  \param nzval       Pointer to the array storing the nonzero values of the matrix.
 *  \param nzval_colptr Pointer to the array storing the column pointers for nzval.
 *  \param rowind      Pointer to the array storing the row indices for nzval.
 *  \param rowind_colptr Pointer to the array storing the column pointers for rowind.
 *  \param col_to_sup  Pointer to the array mapping column indices to supernode indices.
 *  \param sup_to_col  Pointer to the array mapping supernode indices to column indices.
 *  \param stype       Indicates the storage type of the matrix (e.g., SLU_SC for supernodal column-wise).
 *  \param dtype       Indicates the data type of the matrix elements (e.g., SLU_S for single precision float).
 *  \param mtype       Indicates the mathematical type of the matrix (e.g., SLU_GE for a general matrix).
 */
void
sCreate_SuperNode_Matrix(SuperMatrix *L, int m, int n, int_t nnz, 
            float *nzval, int_t *nzval_colptr, int_t *rowind,
            int_t *rowind_colptr, int *col_to_sup, int *sup_to_col,
            Stype_t stype, Dtype_t dtype, Mtype_t mtype)
{
    SCformat *Lstore;

    // Initialize the fields of the SuperMatrix structure
    L->Stype = stype;
    L->Dtype = dtype;
    L->Mtype = mtype;
    L->nrow = m;
    L->ncol = n;

    // Allocate memory for the internal storage format of the supernodal matrix
    L->Store = (void *) SUPERLU_MALLOC( sizeof(SCformat) );
    if ( !(L->Store) ) ABORT("SUPERLU_MALLOC fails for L->Store");
    Lstore = L->Store;
    Lstore->nnz = nnz;
    Lstore->nsuper = col_to_sup[n];
    Lstore->nzval = nzval;
    Lstore->nzval_colptr = nzval_colptr;
    Lstore->rowind = rowind;
    Lstore->rowind_colptr = rowind_colptr;
    Lstore->col_to_sup = col_to_sup;
    Lstore->sup_to_col = sup_to_col;
}

/*! \brief Converts a matrix from row compressed storage to column compressed storage.
 *
 *  Converts a matrix stored in row compressed format (CRS) to column compressed format (CCS).
 *  Allocates memory for the CCS format and transfers the matrix data accordingly.
 *
 *  \param m       Number of rows in the matrix.
 *  \param n       Number of columns in the matrix.
 *  \param nnz     Number of nonzeros in the matrix.
 *  \param a       Pointer to the array storing the nonzero values of the matrix in CRS format.
 *  \param colind  Pointer to the array storing the column indices of nonzeros in CRS format.
 *  \param rowptr  Pointer to the array storing the row pointers of nonzeros in CRS format.
 *  \param at      Pointer to the array storing the nonzero values of the matrix in CCS format (output).
 *  \param rowind  Pointer to the array storing the row indices of nonzeros in CCS format (output).
 *  \param colptr  Pointer to the array storing the column pointers of nonzeros in CCS format (output).
 */
void
sCompRow_to_CompCol(int m, int n, int_t nnz, 
            float *a, int_t *colind, int_t *rowptr,
            float **at, int_t **rowind, int_t **colptr)
{
    register int i, j, col, relpos;
    int_t *marker;

    // Allocate memory for the CCS format arrays
    *at = (float *) floatMalloc(nnz);
    *rowind = (int_t *) intMalloc(nnz);
    *colptr = (int_t *) intMalloc(n+1);
    marker = (int_t *) intCalloc(n);

    // Count the number of entries in each column of matrix A and set up column pointers
    for (i = 0; i < m; ++i)
        for (j = rowptr[i]; j < rowptr[i+1]; ++j) ++marker[colind[j]];
    (*colptr)[0] = 0;
    for (j = 0; j < n; ++j) {
        (*colptr)[j+1] = (*colptr)[j] + marker[j];
        marker[j] = (*colptr)[j];
    }

    // Transfer the matrix from CRS format to CCS format
    for (i = 0; i < m; ++i) {
        for (j = rowptr[i]; j < rowptr[i+1]; ++j) {
            col = colind[j];
            relpos = marker[col];
            (*rowind)[relpos] = i;
            (*at)[relpos] = a[j];
            ++marker[col];
        }
    }

    SUPERLU_FREE(marker);
}

/*! \brief Prints information about a column compressed matrix.
 *
 *  Prints information such as the matrix type and dimensions.
 *
 *  \param what  String describing the matrix (e.g., "L" for matrix L).
 *  \param A     Pointer to the SuperMatrix structure representing the matrix.
 */
void
sPrint_CompCol_Matrix(char *what, SuperMatrix *A)
{
    NCformat     *Astore;
    register int_t i;
    register int n;
    float       *dp;
    
    printf("\nCompCol matrix %s:\n", what);
    printf("Stype %d, Dtype %d, Mtype %d\n", A->Stype,A->Dtype,A->Mtype);
}
    // 获取稀疏矩阵 A 的列数
    n = A->ncol;
    // 将 A 的存储结构转换为 NCformat 类型，并赋值给 Astore
    Astore = (NCformat *) A->Store;
    // 获取非零元素数组的指针
    dp = (float *) Astore->nzval;
    // 打印稀疏矩阵 A 的行数、列数和非零元素个数
    printf("nrow %d, ncol %d, nnz %ld\n", (int)A->nrow, (int)A->ncol, (long)Astore->nnz);
    // 打印非零元素数组 nzval 中的元素
    printf("nzval: ");
    for (i = 0; i < Astore->colptr[n]; ++i) printf("%f  ", dp[i]);
    // 打印行指针数组 rowind 中的元素
    printf("\nrowind: ");
    for (i = 0; i < Astore->colptr[n]; ++i) printf("%ld  ", (long)Astore->rowind[i]);
    // 打印列指针数组 colptr 中的元素
    printf("\ncolptr: ");
    for (i = 0; i <= n; ++i) printf("%ld  ", (long)Astore->colptr[i]);
    // 打印换行符以结束当前行的输出
    printf("\n");
    // 刷新标准输出缓冲区，确保所有输出即时可见
    fflush(stdout);
/*! \brief Diagnostic print of column "jcol" in the U/L factor.
 */
void
sprint_lu_col(char *msg, int jcol, int pivrow, int_t *xprune, GlobalLU_t *Glu)
{
    // 定义变量
    int_t    i, k;
    // 从 GlobalLU_t 结构体中获取所需的指针
    int     *xsup, *supno, fsupc;
    int_t   *xlsub, *lsub;
    float  *lusup;
    int_t   *xlusup;
    float  *ucol;
    int_t   *usub, *xusub;

    // 从 GlobalLU_t 结构体中获取各个指针
    xsup    = Glu->xsup;
    supno   = Glu->supno;
    lsub    = Glu->lsub;
    xlsub   = Glu->xlsub;
    lusup   = (float *) Glu->lusup;
    xlusup  = Glu->xlusup;
    ucol    = (float *) Glu->ucol;
    usub    = Glu->usub;
    xusub   = Glu->xusub;
    
    // 打印消息
    printf("%s", msg);
}
    // 打印格式化字符串，输出列号 jcol，主元行号 pivrow，supno[jcol] 的超节点号，xprune[jcol] 的长长整型值
    printf("col %d: pivrow %d, supno %d, xprune %lld\n", 
       jcol, pivrow, supno[jcol], (long long) xprune[jcol]);
    
    // 打印 U 列的标题
    printf("\tU-col:\n");
    // 遍历 jcol 对应的 U 列
    for (i = xusub[jcol]; i < xusub[jcol+1]; i++)
    // 打印 U 列中每个非零元素的行号 usub[i] 和值 ucol[i]
    printf("\t%d%10.4f\n", (int)usub[i], ucol[i]);
    
    // 打印 L 列在矩形超节点内的标题
    printf("\tL-col in rectangular snode:\n");
    // 获取超节点 snode 的第一列 fsupc
    fsupc = xsup[supno[jcol]];    /* first col of the snode */
    // 初始化在 L 列中遍历的起始位置 i 和在 L supernode 的 U 列中的起始位置 k
    i = xlsub[fsupc];
    k = xlusup[jcol];
    // 遍历超节点 snode 内的 L 列和对应的 U 列
    while ( i < xlsub[fsupc+1] && k < xlusup[jcol+1] ) {
    // 打印 L 列中每个非零元素的行号 lsub[i] 和对应的值 lusup[k]
    printf("\t%d\t%10.4f\n", (int)lsub[i], lusup[k]);
    // 移动到下一个元素
    i++; k++;
    }
    // 刷新标准输出缓冲区
    fflush(stdout);
/*! \brief Check whether tempv[] == 0. This should be true before and after calling any numeric routines, i.e., "panel_bmod" and "column_bmod". 
 */
void scheck_tempv(int n, float *tempv)
{
    int i;
    
    // 遍历数组 tempv，检查每个元素是否为零
    for (i = 0; i < n; i++) {
        // 如果发现不为零的元素，输出错误信息并终止程序
        if (tempv[i] != 0.0) {
            fprintf(stderr,"tempv[%d] = %f\n", i,tempv[i]);
            ABORT("scheck_tempv");
        }
    }
}


void
sGenXtrue(int n, int nrhs, float *x, int ldx)
{
    int  i, j;
    // 嵌套循环设置数组 x 的值为 1.0
    for (j = 0; j < nrhs; ++j)
        for (i = 0; i < n; ++i) {
            x[i + j*ldx] = 1.0; /* + (float)(i+1.)/n; */
        }
}

/*! \brief Let rhs[i] = sum of i-th row of A, so the solution vector is all 1's
 */
void
sFillRHS(trans_t trans, int nrhs, float *x, int ldx,
         SuperMatrix *A, SuperMatrix *B)
{
    DNformat *Bstore;
    float   *rhs;
    float one = 1.0;
    float zero = 0.0;
    int      ldc;
    char transc[1];

    // 获取 B 矩阵的存储格式
    Bstore = B->Store;
    rhs    = Bstore->nzval;
    ldc    = Bstore->lda;
    
    // 根据 trans 设置 transc 字符串的值
    if ( trans == NOTRANS ) *(unsigned char *)transc = 'N';
    else *(unsigned char *)transc = 'T';

    // 调用 BLAS 库中的矩阵乘法函数 sp_sgemm 计算 rhs = A * x
    sp_sgemm(transc, "N", A->nrow, nrhs, A->ncol, one, A,
         x, ldx, zero, rhs, ldc);

}

/*! \brief Fills a float precision array with a given value.
 */
void 
sfill(float *a, int alen, float dval)
{
    register int i;
    // 将数组 a 中的所有元素设置为 dval
    for (i = 0; i < alen; i++) a[i] = dval;
}



/*! \brief Check the inf-norm of the error vector 
 */
void sinf_norm_error(int nrhs, SuperMatrix *X, float *xtrue)
{
    DNformat *Xstore;
    float err, xnorm;
    float *Xmat, *soln_work;
    int i, j;

    // 获取 X 矩阵的存储格式
    Xstore = X->Store;
    Xmat = Xstore->nzval;

    // 计算误差的无穷范数并打印结果
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
sPrintPerf(SuperMatrix *L, SuperMatrix *U, mem_usage_t *mem_usage,
           float rpg, float rcond, float *ferr,
           float *berr, char *equed, SuperLUStat_t *stat)
{
    SCformat *Lstore;
    NCformat *Ustore;
    double   *utime;
    flops_t  *ops;
    
    utime = stat->utime;
    ops   = stat->ops;
    
    // 如果计算因子化的时间不为零，打印计算量和计算效率
    if ( utime[FACT] != 0. )
        printf("Factor flops = %e\tMflops = %8.2f\n", ops[FACT],
               ops[FACT]*1e-6/utime[FACT]);
    
    // 打印松弛节点识别的时间
    printf("Identify relaxed snodes    = %8.2f\n", utime[RELAX]);
    
    // 如果解的计算时间不为零，打印解的计算量和计算效率
    if ( utime[SOLVE] != 0. )
        printf("Solve flops = %.0f, Mflops = %8.2f\n", ops[SOLVE],
               ops[SOLVE]*1e-6/utime[SOLVE]);
    
    // 打印 L 和 U 因子中的非零元素个数
    Lstore = (SCformat *) L->Store;
    Ustore = (NCformat *) U->Store;
    printf("\tNo of nonzeros in factor L = %lld\n", (long long) Lstore->nnz);
    printf("\tNo of nonzeros in factor U = %lld\n", (long long) Ustore->nnz);
}
    # 打印 L+U 中非零元素的数量
    printf("\tNo of nonzeros in L+U = %lld\n", (long long) Lstore->nnz + Ustore->nnz);
    
    # 打印 L 和 U 的内存占用情况以及总共需要的内存
    printf("L\\U MB %.3f\ttotal MB needed %.3f\n",
       mem_usage->for_lu/1e6, mem_usage->total_needed/1e6);
    
    # 打印内存扩展的次数
    printf("Number of memory expansions: %d\n", stat->expansions);
    
    # 打印性能指标表头
    printf("\tFactor\tMflops\tSolve\tMflops\tEtree\tEquil\tRcond\tRefine\n");
    
    # 打印性能数据，包括因子分解的耗时和每秒百万次浮点运算数 (Mflops)，以及解的耗时和Mflops，树结构等的耗时
    printf("PERF:%8.2f%8.2f%8.2f%8.2f%8.2f%8.2f%8.2f%8.2f\n",
       utime[FACT], ops[FACT]*1e-6/utime[FACT],
       utime[SOLVE], ops[SOLVE]*1e-6/utime[SOLVE],
       utime[ETREE], utime[EQUIL], utime[RCOND], utime[REFINE]);
    
    # 打印数值和条件数、前向误差、后向误差、以及是否均衡的标志
    printf("\tRpg\t\tRcond\t\tFerr\t\tBerr\t\tEquil?\n");
    printf("NUM:\t%e\t%e\t%e\t%e\t%s\n",
       rpg, rcond, ferr[0], berr[0], equed);
}

int
print_float_vec(char *what, int n, float *vec)
{
    int i;
    // 打印传入的标识符和浮点向量长度
    printf("%s: n %d\n", what, n);
    // 循环打印浮点向量中每个元素的索引和值
    for (i = 0; i < n; ++i) printf("%d\t%f\n", i, vec[i]);
    // 返回成功标志
    return 0;
}
```