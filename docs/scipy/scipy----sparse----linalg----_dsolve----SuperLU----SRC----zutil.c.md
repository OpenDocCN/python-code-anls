# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zutil.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file zutil.c
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
#include "slu_zdefs.h"

/*! \brief Creates a compressed column matrix.
 *
 * This function initializes a SuperMatrix structure as a compressed column matrix,
 * allocating memory for the matrix data and storing metadata about its structure.
 *
 * @param A Pointer to the SuperMatrix structure to initialize.
 * @param m Number of rows in the matrix.
 * @param n Number of columns in the matrix.
 * @param nnz Number of non-zero elements.
 * @param nzval Array of non-zero values.
 * @param rowind Array of row indices for each non-zero element.
 * @param colptr Array of column pointers indicating where each column starts in nzval and rowind.
 * @param stype Storage format of the matrix (e.g., SLU_NC for compressed column).
 * @param dtype Data type of matrix elements.
 * @param mtype Memory type of the matrix.
 */
void
zCreate_CompCol_Matrix(SuperMatrix *A, int m, int n, int_t nnz, 
               doublecomplex *nzval, int_t *rowind, int_t *colptr,
               Stype_t stype, Dtype_t dtype, Mtype_t mtype)
{
    NCformat *Astore;

    A->Stype = stype;  // 设置存储类型为列压缩存储
    A->Dtype = dtype;  // 设置数据类型
    A->Mtype = mtype;  // 设置内存类型
    A->nrow = m;       // 设置矩阵的行数
    A->ncol = n;       // 设置矩阵的列数
    A->Store = (void *) SUPERLU_MALLOC( sizeof(NCformat) );  // 分配存储空间给矩阵
    if ( !(A->Store) ) ABORT("SUPERLU_MALLOC fails for A->Store");  // 检查分配是否成功
    Astore = A->Store;
    Astore->nnz = nnz;         // 设置非零元素的个数
    Astore->nzval = nzval;     // 指定非零元素值数组
    Astore->rowind = rowind;   // 指定行索引数组
    Astore->colptr = colptr;   // 指定列指针数组
}

/*! \brief Creates a compressed row matrix.
 *
 * This function initializes a SuperMatrix structure as a compressed row matrix,
 * allocating memory for the matrix data and storing metadata about its structure.
 *
 * @param A Pointer to the SuperMatrix structure to initialize.
 * @param m Number of rows in the matrix.
 * @param n Number of columns in the matrix.
 * @param nnz Number of non-zero elements.
 * @param nzval Array of non-zero values.
 * @param colind Array of column indices for each non-zero element.
 * @param rowptr Array of row pointers indicating where each row starts in nzval and colind.
 * @param stype Storage format of the matrix (e.g., SLU_NR for compressed row).
 * @param dtype Data type of matrix elements.
 * @param mtype Memory type of the matrix.
 */
void
zCreate_CompRow_Matrix(SuperMatrix *A, int m, int n, int_t nnz, 
               doublecomplex *nzval, int_t *colind, int_t *rowptr,
               Stype_t stype, Dtype_t dtype, Mtype_t mtype)
{
    NRformat *Astore;

    A->Stype = stype;  // 设置存储类型为行压缩存储
    A->Dtype = dtype;  // 设置数据类型
    A->Mtype = mtype;  // 设置内存类型
    A->nrow = m;       // 设置矩阵的行数
    A->ncol = n;       // 设置矩阵的列数
    A->Store = (void *) SUPERLU_MALLOC( sizeof(NRformat) );  // 分配存储空间给矩阵
    if ( !(A->Store) ) ABORT("SUPERLU_MALLOC fails for A->Store");  // 检查分配是否成功
    Astore = A->Store;
    Astore->nnz = nnz;         // 设置非零元素的个数
    Astore->nzval = nzval;     // 指定非零元素值数组
    Astore->colind = colind;   // 指定列索引数组
    Astore->rowptr = rowptr;   // 指定行指针数组
}

/*! \brief Copies a compressed column matrix A into matrix B.
 *
 * This function copies all data from matrix A (compressed column format) into matrix B,
 * which is also expected to be initialized as a compressed column matrix.
 *
 * @param A Pointer to the source SuperMatrix (compressed column format).
 * @param B Pointer to the destination SuperMatrix (compressed column format).
 */
void
zCopy_CompCol_Matrix(SuperMatrix *A, SuperMatrix *B)
{
    NCformat *Astore, *Bstore;
    int      ncol, nnz, i;

    B->Stype = A->Stype;  // 设置目标矩阵存储类型与源矩阵相同
    B->Dtype = A->Dtype;  // 设置目标矩阵数据类型与源矩阵相同
    B->Mtype = A->Mtype;  // 设置目标矩阵内存类型与源矩阵相同
    B->nrow  = A->nrow;   // 设置目标矩阵行数与源矩阵相同
    B->ncol  = ncol = A->ncol;  // 设置目标矩阵列数与源矩阵相同
    Astore   = (NCformat *) A->Store;  // 获取源矩阵的存储格式
    Bstore   = (NCformat *) B->Store;  // 获取目标矩阵的存储格式
    Bstore->nnz = nnz = Astore->nnz;   // 设置目标矩阵非零元素个数与源矩阵相同
    for (i = 0; i < nnz; ++i)
        ((doublecomplex *)Bstore->nzval)[i] = ((doublecomplex *)Astore->nzval)[i];  // 复制非零元素值
    for (i = 0; i < nnz; ++i)
        Bstore->rowind[i] = Astore->rowind[i];  // 复制行索引
    for (i = 0; i <= ncol; ++i)
        Bstore->colptr[i] = Astore->colptr[i];  // 复制列指针
}
/*! \brief 创建稠密矩阵存储结构。
 *
 *  使用给定的参数初始化一个 SuperMatrix 对象 X，存储为稠密矩阵格式。
 *
 *  \param X       SuperMatrix 对象指针，将被初始化为稠密矩阵格式。
 *  \param m       矩阵行数。
 *  \param n       矩阵列数。
 *  \param x       输入的复数双精度数组，作为矩阵的实部和虚部。
 *  \param ldx     x 数组的 leading dimension。
 *  \param stype   矩阵的存储类型。
 *  \param dtype   矩阵元素的数据类型。
 *  \param mtype   矩阵的类型。
 */
zCreate_Dense_Matrix(SuperMatrix *X, int m, int n, doublecomplex *x, int ldx,
            Stype_t stype, Dtype_t dtype, Mtype_t mtype)
{
    DNformat    *Xstore;
    
    X->Stype = stype;            // 设置矩阵的存储类型
    X->Dtype = dtype;            // 设置矩阵元素的数据类型
    X->Mtype = mtype;            // 设置矩阵的类型
    X->nrow = m;                 // 设置矩阵的行数
    X->ncol = n;                 // 设置矩阵的列数
    X->Store = (void *) SUPERLU_MALLOC( sizeof(DNformat) );  // 分配存储空间
    if ( !(X->Store) ) ABORT("SUPERLU_MALLOC fails for X->Store");  // 分配失败时中止程序
    Xstore = (DNformat *) X->Store;
    Xstore->lda = ldx;           // 设置 leading dimension
    Xstore->nzval = (doublecomplex *) x;  // 设置矩阵值数组
}

/*! \brief 复制一个稠密矩阵 X 到另一个矩阵 Y。
 *
 *  将输入矩阵 X 的内容复制到输出矩阵 Y 中。
 *
 *  \param M      输入矩阵 X 的行数。
 *  \param N      输入矩阵 X 的列数。
 *  \param X      输入矩阵 X 的复数双精度数组。
 *  \param ldx    X 数组的 leading dimension。
 *  \param Y      输出矩阵 Y 的复数双精度数组。
 *  \param ldy    Y 数组的 leading dimension。
 */
void
zCopy_Dense_Matrix(int M, int N, doublecomplex *X, int ldx,
            doublecomplex *Y, int ldy)
{
    int    i, j;
    
    for (j = 0; j < N; ++j)       // 遍历列
        for (i = 0; i < M; ++i)   // 遍历行
            Y[i + j*ldy] = X[i + j*ldx];  // 复制矩阵元素
}

/*! \brief 创建超节点矩阵存储结构。
 *
 *  使用给定的参数初始化一个 SuperMatrix 对象 L，存储为超节点矩阵格式。
 *
 *  \param L            SuperMatrix 对象指针，将被初始化为超节点矩阵格式。
 *  \param m            矩阵行数。
 *  \param n            矩阵列数。
 *  \param nnz          非零元素的个数。
 *  \param nzval        输入的复数双精度数组，存储非零元素的值。
 *  \param nzval_colptr 列偏移指针数组。
 *  \param rowind       行索引数组。
 *  \param rowind_colptr 行偏移指针数组。
 *  \param col_to_sup   列到超节点映射数组。
 *  \param sup_to_col   超节点到列映射数组。
 *  \param stype        矩阵的存储类型。
 *  \param dtype        矩阵元素的数据类型。
 *  \param mtype        矩阵的类型。
 */
void
zCreate_SuperNode_Matrix(SuperMatrix *L, int m, int n, int_t nnz, 
            doublecomplex *nzval, int_t *nzval_colptr, int_t *rowind,
            int_t *rowind_colptr, int *col_to_sup, int *sup_to_col,
            Stype_t stype, Dtype_t dtype, Mtype_t mtype)
{
    SCformat *Lstore;

    L->Stype = stype;            // 设置矩阵的存储类型
    L->Dtype = dtype;            // 设置矩阵元素的数据类型
    L->Mtype = mtype;            // 设置矩阵的类型
    L->nrow = m;                 // 设置矩阵的行数
    L->ncol = n;                 // 设置矩阵的列数
    L->Store = (void *) SUPERLU_MALLOC( sizeof(SCformat) );  // 分配存储空间
    if ( !(L->Store) ) ABORT("SUPERLU_MALLOC fails for L->Store");  // 分配失败时中止程序
    Lstore = L->Store;
    Lstore->nnz = nnz;           // 设置非零元素的个数
    Lstore->nsuper = col_to_sup[n];  // 设置超节点的个数
    Lstore->nzval = nzval;       // 设置非零元素值数组
    Lstore->nzval_colptr = nzval_colptr;  // 设置列偏移指针数组
    Lstore->rowind = rowind;     // 设置行索引数组
    Lstore->rowind_colptr = rowind_colptr;  // 设置行偏移指针数组
    Lstore->col_to_sup = col_to_sup;  // 设置列到超节点映射数组
    Lstore->sup_to_col = sup_to_col;  // 设置超节点到列映射数组
}

/*! \brief 将行压缩存储的矩阵转换为列压缩存储。
 *
 *  将输入的行压缩存储格式矩阵转换为输出的列压缩存储格式矩阵。
 *
 *  \param m        矩阵行数。
 *  \param n        矩阵列数。
 *  \param nnz      非零元素的个数。
 *  \param a        输入的复数双精度数组，存储矩阵的非零元素。
 *  \param colind   输入的列索引数组。
 *  \param rowptr   输入的行偏移指针数组。
 *  \param at       输出的复数双精度数组指针，存储转换后的矩阵非零元素。
 *  \param rowind   输出的行索引数组指针，存储转换后的矩阵行索引。
 *  \param colptr   输出的列偏移指针数组指针，存储转换后的矩阵列偏移。
 */
void
zCompRow_to_CompCol(int m, int n, int_t nnz, 
            doublecomplex *a, int_t *colind, int_t *rowptr,
            doublecomplex **at, int_t **rowind, int_t **colptr)
{
    register int i, j, col, relpos;
    int_t *marker;

    /* Allocate storage for another copy of the matrix. */
    *at = (doublecomplex *) doublecomplexMalloc(nnz);  // 分配转置后矩阵的值存储空间
    *rowind = (int_t *) intMalloc(nnz);                // 分配转置后矩阵的行索引存储空间
    *colptr = (int_t *) intMalloc(n+1);                // 分配转置后矩阵的列偏移指针存储空间
    marker = (int_t *) intCalloc(n);                    // 分配列计数器数组

    /* Get counts of each column of A, and set up column pointers */
    for (i = 0; i < m; ++i)
        for (j = rowptr
    // 输出 A 结构体中的 Stype、Dtype 和 Mtype 字段的值
    printf("Stype %d, Dtype %d, Mtype %d\n", A->Stype,A->Dtype,A->Mtype);
    // 将 A 结构体中的 ncol 赋给变量 n
    n = A->ncol;
    // 将 A 结构体中的存储字段 A->Store 强制转换为 NCformat 类型的指针，并将其 nzval 字段强制转换为 double 类型指针赋给 dp
    Astore = (NCformat *) A->Store;
    dp = (double *) Astore->nzval;
    // 输出 A 结构体中的 nrow、ncol 和 Astore 结构体中的 nnz 字段的值
    printf("nrow %d, ncol %d, nnz %ld\n", (int)A->nrow, (int)A->ncol, (long)Astore->nnz);
    // 输出字符串 "nzval: "
    printf("nzval: ");
    // 遍历 dp 数组，输出前 2*Astore->colptr[n] 个元素的值，每个元素作为浮点数输出
    for (i = 0; i < 2*Astore->colptr[n]; ++i) printf("%f  ", dp[i]);
    // 输出换行符
    printf("\nrowind: ");
    // 遍历 Astore 结构体中的 rowind 数组，输出前 Astore->colptr[n] 个元素的值，每个元素作为长整型输出
    for (i = 0; i < Astore->colptr[n]; ++i) printf("%ld  ", (long)Astore->rowind[i]);
    // 输出换行符
    printf("\ncolptr: ");
    // 遍历 Astore 结构体中的 colptr 数组，输出前 n+1 个元素的值，每个元素作为长整型输出
    for (i = 0; i <= n; ++i) printf("%ld  ", (long)Astore->colptr[i]);
    // 输出换行符
    printf("\n");
    // 刷新标准输出缓冲区
    fflush(stdout);
void
zPrint_SuperNode_Matrix(char *what, SuperMatrix *A)
{
    SCformat     *Astore;         // 定义 SCformat 结构体指针 Astore
    register int_t i, j, k, c, d, n, nsup;   // 定义整型变量 i, j, k, c, d, n, nsup
    double       *dp;             // 定义双精度浮点数指针 dp
    int *col_to_sup, *sup_to_col; // 定义整型指针 col_to_sup 和 sup_to_col
    int_t *rowind, *rowind_colptr; // 定义 int_t 类型指针 rowind 和 rowind_colptr

    printf("\nSuperNode matrix %s:\n", what);   // 打印输出超节点矩阵的名称
    printf("Stype %d, Dtype %d, Mtype %d\n", A->Stype,A->Dtype,A->Mtype);   // 打印输出矩阵的类型信息
    n = A->ncol;    // 获取矩阵的列数
    Astore = (SCformat *) A->Store;   // 将矩阵的存储格式转换为 SCformat，并赋给 Astore
    dp = (double *) Astore->nzval;    // 获取矩阵非零值的指针 dp
    col_to_sup = Astore->col_to_sup;  // 获取列到超节点的映射数组的指针
    sup_to_col = Astore->sup_to_col;  // 获取超节点到列的映射数组的指针
    rowind_colptr = Astore->rowind_colptr;  // 获取行指标列指针的指针
    rowind = Astore->rowind;    // 获取行指标数组的指针
    printf("nrow %d, ncol %d, nnz %lld, nsuper %d\n", 
           (int)A->nrow, (int)A->ncol, (long long) Astore->nnz, (int)Astore->nsuper);   // 打印输出矩阵的行数、列数、非零元素数和超节点数
    printf("nzval:\n");    // 打印输出非零值数组的标题
    for (k = 0; k <= Astore->nsuper; ++k) {   // 遍历每个超节点
        c = sup_to_col[k];  // 获取当前超节点对应的列索引
        nsup = sup_to_col[k+1] - c;   // 获取当前超节点包含的列数
        for (j = c; j < c + nsup; ++j) {   // 遍历当前超节点的每一列
            d = Astore->nzval_colptr[j];   // 获取当前列的非零值起始位置
            for (i = rowind_colptr[c]; i < rowind_colptr[c+1]; ++i) {   // 遍历当前列的每个非零行索引
                printf("%d\t%d\t%e\t%e\n", (int)rowind[i], (int) j, dp[d], dp[d+1]);   // 打印输出行索引、列索引及对应的非零值
                d += 2;    // 更新非零值索引
            }
        }
    }
#if 0
    for (i = 0; i < 2*Astore->nzval_colptr[n]; ++i) printf("%f  ", dp[i]);
#endif
    printf("\nnzval_colptr: ");   // 打印输出非零值列指针数组的标题
    for (i = 0; i <= n; ++i) printf("%lld  ", (long long)Astore->nzval_colptr[i]);   // 打印输出非零值列指针数组的内容
    printf("\nrowind: ");   // 打印输出行指标数组的标题
    for (i = 0; i < Astore->rowind_colptr[n]; ++i) 
        printf("%lld  ", (long long)Astore->rowind[i]);   // 打印输出行指标数组的内容
    printf("\nrowind_colptr: ");   // 打印输出行指标列指针数组的标题
    for (i = 0; i <= n; ++i) printf("%lld  ", (long long)Astore->rowind_colptr[i]);   // 打印输出行指标列指针数组的内容
    printf("\ncol_to_sup: ");   // 打印输出列到超节点映射数组的标题
    for (i = 0; i < n; ++i) printf("%d  ", col_to_sup[i]);   // 打印输出列到超节点映射数组的内容
    printf("\nsup_to_col: ");   // 打印输出超节点到列映射数组的标题
    for (i = 0; i <= Astore->nsuper+1; ++i) 
        printf("%d  ", sup_to_col[i]);   // 打印输出超节点到列映射数组的内容
    printf("\n");   // 输出换行符
    fflush(stdout);   // 清空输出缓冲区，将内容输出到标准输出
}

void
zPrint_Dense_Matrix(char *what, SuperMatrix *A)
{
    DNformat     *Astore = (DNformat *) A->Store;   // 将矩阵的存储格式转换为 DNformat，并赋给 Astore
    register int i, j, lda = Astore->lda;   // 定义整型变量 i, j，并获取 Astore 的 lda 值
    double       *dp;   // 定义双精度浮点数指针 dp

    printf("\nDense matrix %s:\n", what);   // 打印输出稠密矩阵的名称
    printf("Stype %d, Dtype %d, Mtype %d\n", A->Stype,A->Dtype,A->Mtype);   // 打印输出矩阵的类型信息
    dp = (double *) Astore->nzval;    // 获取矩阵非零值的指针 dp
    printf("nrow %d, ncol %d, lda %d\n", (int)A->nrow, (int)A->ncol, lda);   // 打印输出矩阵的行数、列数和 lda 值
    printf("\nnzval: ");   // 打印输出非零值数组的标题
    for (j = 0; j < A->ncol; ++j) {   // 遍历矩阵的每一列
        for (i = 0; i < 2*A->nrow; ++i) printf("%f  ", dp[i + j*2*lda]);   // 打印输出当前列的非零值
        printf("\n");   // 输出换行符
    }
    printf("\n");   // 输出空行
    fflush(stdout);   // 清空输出缓冲区，将内容输出到标准输出
}

/*! \brief Diagnostic print of column "jcol" in the U/L factor.
 */
void
zprint_lu_col(char *msg, int jcol, int pivrow, int_t *xprune, GlobalLU_t *Glu)
{
    int_t    i, k;   // 定义 int_t 类型变量 i, k
    int     *xsup, *supno, fsupc;   // 定义整型指针 xsup, supno 和整型变量 fsupc
    int_t   *xlsub, *lsub;   // 定义 int_t 类型指针 xlsub 和 lsub
    doublecomplex  *lusup;   // 定义 doublecomplex 类型指针 lusup
    int_t   *xlusup;   // 定义 int_t 类型指针 xlusup
    doublecomplex  *ucol;   // 定义 doublecomplex 类型指针 ucol
    int_t   *usub, *xusub;   // 定义 int_t 类型指针 usub 和 xusub

    xsup    = Glu->xsup;   // 获取全局 LU 分解结构体中的 xsup 指针
    supno   = Glu->supno;   // 获取全局 LU 分解结构体中的 supno 指针
    lsub
    xusub   = Glu->xusub;
    // 将指针 G1u 的 xusub 成员赋给 xusub

    printf("%s", msg);
    // 打印字符串 msg

    printf("col %d: pivrow %d, supno %d, xprune %lld\n", 
       jcol, pivrow, supno[jcol], (long long) xprune[jcol]);
    // 打印列号 jcol、主元行号 pivrow、supno[jcol] 和 xprune[jcol] 的信息

    printf("\tU-col:\n");
    // 打印 U 列信息的标题
    for (i = xusub[jcol]; i < xusub[jcol+1]; i++)
    // 遍历 xusub[jcol] 到 xusub[jcol+1] 范围内的索引 i
    printf("\t%d%10.4f, %10.4f\n", (int)usub[i], ucol[i].r, ucol[i].i);
    // 打印 usub[i]、ucol[i].r 和 ucol[i].i 的信息

    printf("\tL-col in rectangular snode:\n");
    // 打印矩形超节点中 L 列的标题
    fsupc = xsup[supno[jcol]];    /* first col of the snode */
    // 将 xsup[supno[jcol]] 的值赋给 fsupc，这是超节点的第一列
    i = xlsub[fsupc];
    // 将 xlsub[fsupc] 的值赋给 i，即超节点第一列在 xlsub 中的起始位置
    k = xlusup[jcol];
    // 将 xlusup[jcol] 的值赋给 k，即 U 列的起始位置
    while ( i < xlsub[fsupc+1] && k < xlusup[jcol+1] ) {
    // 当 i 小于超节点第一列的下一列在 xlsub 中的位置且 k 小于 jcol+1 列在 xlusup 中的位置时循环
    printf("\t%d\t%10.4f, %10.4f\n", (int)lsub[i], lusup[k].r, lusup[k].i);
    // 打印 lsub[i]、lusup[k].r 和 lusup[k].i 的信息
    i++; k++;
    // i 和 k 向后移动一位
    }
    fflush(stdout);
    // 刷新标准输出流，确保所有输出立即显示
/*! \brief Check whether tempv[] == 0. This should be true before and after calling any numeric routines, i.e., "panel_bmod" and "column_bmod". 
 */
void zcheck_tempv(int n, doublecomplex *tempv)
{
    int i;
    
    // 遍历数组 tempv，检查每个元素是否为零
    for (i = 0; i < n; i++) {
        // 如果 tempv[i] 的实部或虚部不为零，则输出错误信息并中止程序
        if ((tempv[i].r != 0.0) || (tempv[i].i != 0.0))
        {
            fprintf(stderr,"tempv[%d] = {%f, %f}\n", i, tempv[i].r, tempv[i].i);
            ABORT("zcheck_tempv");
        }
    }
}

/*! \brief Initialize the matrix x to have all elements as 1.0 + 0.0i.
 */
void
zGenXtrue(int n, int nrhs, doublecomplex *x, int ldx)
{
    int  i, j;
    // 遍历列数 nrhs 和行数 n，将 x 的每个元素设为 1.0 + 0.0i
    for (j = 0; j < nrhs; ++j)
        for (i = 0; i < n; ++i) {
            x[i + j*ldx].r = 1.0;
            x[i + j*ldx].i = 0.0;
        }
}

/*! \brief Let rhs[i] = sum of i-th row of A, so the solution vector is all 1's
 */
void
zFillRHS(trans_t trans, int nrhs, doublecomplex *x, int ldx,
         SuperMatrix *A, SuperMatrix *B)
{
    DNformat *Bstore;
    doublecomplex   *rhs;
    doublecomplex one = {1.0, 0.0};
    doublecomplex zero = {0.0, 0.0};
    int      ldc;
    char transc[1];

    // 获取 B 矩阵的存储格式和数据
    Bstore = B->Store;
    rhs    = Bstore->nzval;
    ldc    = Bstore->lda;
    
    // 根据 trans 参数确定转置标记
    if ( trans == NOTRANS ) *(unsigned char *)transc = 'N';
    else *(unsigned char *)transc = 'T';

    // 执行矩阵乘法 A * x，并将结果存储在 rhs 中
    sp_zgemm(transc, "N", A->nrow, nrhs, A->ncol, one, A,
         x, ldx, zero, rhs, ldc);

}

/*! \brief Fill an array 'a' of length 'alen' with the complex value 'dval'.
 */
void 
zfill(doublecomplex *a, int alen, doublecomplex dval)
{
    register int i;
    // 将数组 a 的每个元素设为复数值 dval
    for (i = 0; i < alen; i++) a[i] = dval;
}

/*! \brief Compute the infinity norm of the error vector.
 */
void zinf_norm_error(int nrhs, SuperMatrix *X, doublecomplex *xtrue)
{
    DNformat *Xstore;
    double err, xnorm;
    doublecomplex *Xmat, *soln_work;
    doublecomplex temp;
    int i, j;

    // 获取解向量 X 的存储格式和数据
    Xstore = X->Store;
    Xmat = Xstore->nzval;

    // 计算每个解向量的误差和无穷范数
    for (j = 0; j < nrhs; j++) {
        soln_work = &Xmat[j*Xstore->lda];
        err = xnorm = 0.0;
        for (i = 0; i < X->nrow; i++) {
            // 计算解向量与真实解之间的差值，并更新误差和无穷范数
            z_sub(&temp, &soln_work[i], &xtrue[i]);
            err = SUPERLU_MAX(err, z_abs(&temp));
            xnorm = SUPERLU_MAX(xnorm, z_abs(&soln_work[i]));
        }
        // 输出误差向量的无穷范数
        err = err / xnorm;
        printf("||X - Xtrue||/||X|| = %e\n", err);
    }
}

/*! \brief Print performance statistics of the code.
 */
void
zPrintPerf(SuperMatrix *L, SuperMatrix *U, mem_usage_t *mem_usage,
           double rpg, double rcond, double *ferr,
           double *berr, char *equed, SuperLUStat_t *stat)
{
    SCformat *Lstore;
    NCformat *Ustore;
    double   *utime;
    flops_t  *ops;
    
    utime = stat->utime;
    ops   = stat->ops;
    
    // 打印因子化阶段的浮点运算量和性能指标
    if ( utime[FACT] != 0. )
        printf("Factor flops = %e\tMflops = %8.2f\n", ops[FACT],
               ops[FACT]*1e-6/utime[FACT]);
    
    // 打印松弛节点的识别时间
    printf("Identify relaxed snodes    = %8.2f\n", utime[RELAX]);
    
    // 打印求解阶段的浮点运算量和性能指标
    if ( utime[SOLVE] != 0. )
        printf("Solve flops = %.0f, Mflops = %8.2f\n", ops[SOLVE],
               ops[SOLVE]*1e-6/utime[SOLVE]);
    
    // 获取 L 和 U 矩阵的存储格式
    Lstore = (SCformat *) L->Store;
    Ustore = (NCformat *) U->Store;
}
    # 打印输出 L 因子中的非零元素数量
    printf("\tNo of nonzeros in factor L = %lld\n", (long long) Lstore->nnz);
    # 打印输出 U 因子中的非零元素数量
    printf("\tNo of nonzeros in factor U = %lld\n", (long long) Ustore->nnz);
    # 打印输出 L 和 U 因子总共的非零元素数量
    printf("\tNo of nonzeros in L+U = %lld\n", (long long) Lstore->nnz + Ustore->nnz);
    
    # 打印输出 L/U 使用的内存量（以MB为单位）和总共需要的内存量（以MB为单位）
    printf("L\\U MB %.3f\ttotal MB needed %.3f\n",
       mem_usage->for_lu/1e6, mem_usage->total_needed/1e6);
    # 打印输出内存扩展的次数
    printf("Number of memory expansions: %d\n", stat->expansions);
    
    # 打印输出各种性能指标的表头
    printf("\tFactor\tMflops\tSolve\tMflops\tEtree\tEquil\tRcond\tRefine\n");
    # 打印输出各种性能指标的实际数值
    printf("PERF:%8.2f%8.2f%8.2f%8.2f%8.2f%8.2f%8.2f%8.2f\n",
       utime[FACT], ops[FACT]*1e-6/utime[FACT],
       utime[SOLVE], ops[SOLVE]*1e-6/utime[SOLVE],
       utime[ETREE], utime[EQUIL], utime[RCOND], utime[REFINE]);
    
    # 打印输出残差和误差相关的表头
    printf("\tRpg\t\tRcond\t\tFerr\t\tBerr\t\tEquil?\n");
    # 打印输出具体的残差和误差数值以及Equil字段的状态
    printf("NUM:\t%e\t%e\t%e\t%e\t%s\n",
       rpg, rcond, ferr[0], berr[0], equed);
}
```