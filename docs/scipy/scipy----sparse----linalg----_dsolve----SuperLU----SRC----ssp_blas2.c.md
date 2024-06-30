# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ssp_blas2.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ssp_blas2.c
 * \brief Sparse BLAS 2, using some dense BLAS 2 operations
 *
 * <pre>
 * -- SuperLU routine (version 5.1) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * October 15, 2003
 *
 * Last update: December 3, 2015
 * </pre>
 */
/*
 * File name:        ssp_blas2.c
 * Purpose:        Sparse BLAS 2, using some dense BLAS 2 operations.
 */

#include "slu_sdefs.h"

/*! \brief Solves one of the systems of equations A*x = b,   or   A'*x = b
 * 
 * <pre>
 *   Purpose
 *   =======
 *
 *   sp_strsv() solves one of the systems of equations   
 *       A*x = b,   or   A'*x = b,
 *   where b and x are n element vectors and A is a sparse unit , or   
 *   non-unit, upper or lower triangular matrix.   
 *   No test for singularity or near-singularity is included in this   
 *   routine. Such tests must be performed before calling this routine.   
 *
 *   Parameters   
 *   ==========   
 *
 *   uplo   - (input) char*
 *            On entry, uplo specifies whether the matrix is an upper or   
 *             lower triangular matrix as follows:   
 *                uplo = 'U' or 'u'   A is an upper triangular matrix.   
 *                uplo = 'L' or 'l'   A is a lower triangular matrix.   
 *
 *   trans  - (input) char*
 *             On entry, trans specifies the equations to be solved as   
 *             follows:   
 *                trans = 'N' or 'n'   A*x = b.   
 *                trans = 'T' or 't'   A'*x = b.
 *                trans = 'C' or 'c'   A'*x = b.   
 *
 *   diag   - (input) char*
 *             On entry, diag specifies whether or not A is unit   
 *             triangular as follows:   
 *                diag = 'U' or 'u'   A is assumed to be unit triangular.   
 *                diag = 'N' or 'n'   A is not assumed to be unit   
 *                                    triangular.   
 *         
 *   L       - (input) SuperMatrix*
 *           The factor L from the factorization Pr*A*Pc=L*U. Use
 *             compressed row subscripts storage for supernodes,
 *             i.e., L has types: Stype = SC, Dtype = SLU_S, Mtype = TRLU.
 *
 *   U       - (input) SuperMatrix*
 *            The factor U from the factorization Pr*A*Pc=L*U.
 *            U has types: Stype = NC, Dtype = SLU_S, Mtype = TRU.
 *    
 *   x       - (input/output) float*
 *             Before entry, the incremented array X must contain the n   
 *             element right-hand side vector b. On exit, X is overwritten 
 *             with the solution vector x.
 *
 *   info    - (output) int*
 *             If *info = -i, the i-th argument had an illegal value.
 * </pre>
 */

#include "slu_sdefs.h"

/*! \brief Solves one of the systems of equations A*x = b,   or   A'*x = b
 * 
 * <pre>
 *   Purpose
 *   =======
 *
 *   sp_strsv() solves one of the systems of equations   
 *       A*x = b,   or   A'*x = b,
 *   where b and x are n element vectors and A is a sparse unit , or   
 *   non-unit, upper or lower triangular matrix.   
 *   No test for singularity or near-singularity is included in this   
 *   routine. Such tests must be performed before calling this routine.   
 *
 *   Parameters   
 *   ==========   
 *
 *   uplo   - (input) char*
 *            On entry, uplo specifies whether the matrix is an upper or   
 *             lower triangular matrix as follows:   
 *                uplo = 'U' or 'u'   A is an upper triangular matrix.   
 *                uplo = 'L' or 'l'   A is a lower triangular matrix.   
 *
 *   trans  - (input) char*
 *             On entry, trans specifies the equations to be solved as   
 *             follows:   
 *                trans = 'N' or 'n'   A*x = b.   
 *                trans = 'T' or 't'   A'*x = b.
 *                trans = 'C' or 'c'   A'*x = b.   
 *
 *   diag   - (input) char*
 *             On entry, diag specifies whether or not A is unit   
 *             triangular as follows:   
 *                diag = 'U' or 'u'   A is assumed to be unit triangular.   
 *                diag = 'N' or 'n'   A is not assumed to be unit   
 *                                    triangular.   
 *         
 *   L       - (input) SuperMatrix*
 *           The factor L from the factorization Pr*A*Pc=L*U. Use
 *             compressed row subscripts storage for supernodes,
 *             i.e., L has types: Stype = SC, Dtype = SLU_S, Mtype = TRLU.
 *
 *   U       - (input) SuperMatrix*
 *            The factor U from the factorization Pr*A*Pc=L*U.
 *            U has types: Stype = NC, Dtype = SLU_S, Mtype = TRU.
 *    
 *   x       - (input/output) float*
 *             Before entry, the incremented array X must contain the n   
 *             element right-hand side vector b. On exit, X is overwritten 
 *             with the solution vector x.
 *
 *   info    - (output) int*
 *             If *info = -i, the i-th argument had an illegal value.
 * </pre>
 */
#ifdef _CRAY
    _fcd ftcs1 = _cptofcd("L", strlen("L")),    /* 定义 _fcd 类型变量 ftcs1，用于存储字符 "L" 的长度 */
     ftcs2 = _cptofcd("N", strlen("N")),        /* 定义 _fcd 类型变量 ftcs2，用于存储字符 "N" 的长度 */
     ftcs3 = _cptofcd("U", strlen("U"));        /* 定义 _fcd 类型变量 ftcs3，用于存储字符 "U" 的长度 */
#endif

SCformat *Lstore;   /* 定义指向 SCformat 结构体的指针 Lstore */
NCformat *Ustore;   /* 定义指向 NCformat 结构体的指针 Ustore */
float   *Lval, *Uval;    /* 定义 float 类型指针变量 Lval 和 Uval */
int incx = 1, incy = 1;  /* 设置增量值 incx 和 incy 为 1 */
float alpha = 1.0, beta = 1.0;  /* 设置 alpha 和 beta 的初始值为 1.0 */
int nrow, irow, jcol;   /* 定义整型变量 nrow, irow, jcol */
int fsupc, nsupr, nsupc;    /* 定义整型变量 fsupc, nsupr, nsupc */
int_t luptr, istart, i, k, iptr;    /* 定义 int_t 类型变量 luptr, istart, i, k, iptr */
float *work;    /* 定义 float 类型指针变量 work */
flops_t solve_ops;  /* 定义 flops_t 类型变量 solve_ops */

/* 测试输入参数的有效性 */
*info = 0;
if ( strncmp(uplo,"L", 1)!=0 && strncmp(uplo, "U", 1)!=0 ) *info = -1;
else if ( strncmp(trans, "N", 1)!=0 && strncmp(trans, "T", 1)!=0 && 
          strncmp(trans, "C", 1)!=0) *info = -2;
else if ( strncmp(diag, "U", 1)!=0 && strncmp(diag, "N", 1)!=0 )
     *info = -3;
else if ( L->nrow != L->ncol || L->nrow < 0 ) *info = -4;
else if ( U->nrow != U->ncol || U->nrow < 0 ) *info = -5;
if ( *info ) {
int ii = -(*info);
input_error("sp_strsv", &ii);   /* 调用 input_error 函数处理错误信息 */
return 0;
}

Lstore = L->Store;  /* 将 L 的存储结构赋给 Lstore */
Lval = Lstore->nzval;   /* 获取 Lstore 中的 nzval 到 Lval */
Ustore = U->Store;  /* 将 U 的存储结构赋给 Ustore */
Uval = Ustore->nzval;   /* 获取 Ustore 中的 nzval 到 Uval */
solve_ops = 0;  /* 初始化 solve_ops 为 0 */

if ( !(work = floatCalloc(L->nrow)) )   /* 分配 L->nrow 个 float 类型的内存空间给 work */
ABORT("Malloc fails for work in sp_strsv().");   /* 内存分配失败时调用 ABORT 函数 */

if ( strncmp(trans, "N", 1)==0 ) {    /* 如果 trans 是 "N"，执行以下操作 */

if ( strncmp(uplo, "L", 1)==0 ) {
    /* 执行 x := inv(L)*x */
        if ( L->nrow == 0 ) return 0; /* 如果 L 的行数为 0，快速返回 */

    for (k = 0; k <= Lstore->nsuper; k++) {
    fsupc = L_FST_SUPC(k);  /* 获取第 k 列的第一个非零元素的超节点 */
    istart = L_SUB_START(fsupc);    /* 获取 fsupc 超节点行索引的起始位置 */
    nsupr = L_SUB_START(fsupc+1) - istart;   /* 计算超节点行索引的长度 */
    nsupc = L_FST_SUPC(k+1) - fsupc; /* 计算超节点列索引的长度 */
    luptr = L_NZ_START(fsupc);  /* 获取第 k 列的第一个非零元素的索引 */
    nrow = nsupr - nsupc;   /* 计算非超节点行的数量 */

        solve_ops += nsupc * (nsupc - 1);  /* 计算解操作次数 */
        solve_ops += 2 * nrow * nsupc;     /* 计算解操作次数 */

    if ( nsupc == 1 ) {
        for (iptr=istart+1; iptr < L_SUB_START(fsupc+1); ++iptr) {
        irow = L_SUB(iptr); /* 获取超节点的行索引 */
        ++luptr;
        x[irow] -= x[fsupc] * Lval[luptr];   /* 更新 x[irow] 的值 */
        }
    } else {
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
        STRSV(ftcs1, ftcs2, ftcs3, &nsupc, &Lval[luptr], &nsupr,
               &x[fsupc], &incx);   /* 使用 BLAS 库计算解 */
    
        SGEMV(ftcs2, &nrow, &nsupc, &alpha, &Lval[luptr+nsupc], 
               &nsupr, &x[fsupc], &incx, &beta, &work[0], &incy);   /* 使用 BLAS 库计算向量乘法 */
#else
        strsv_("L", "N", "U", &nsupc, &Lval[luptr], &nsupr,
               &x[fsupc], &incx);   /* 使用 BLAS 库计算解 */
    
        sgemv_("N", &nrow, &nsupc, &alpha, &Lval[luptr+nsupc], 
               &nsupr, &x[fsupc], &incx, &beta, &work[0], &incy);   /* 使用 BLAS 库计算向量乘法 */
#endif
#else
        slsolve ( nsupr, nsupc, &Lval[luptr], &x[fsupc]);   /* 使用自定义函数计算解 */
    
        smatvec ( nsupr, nsupr-nsupc, nsupc, &Lval[luptr+nsupc],
                         &x[fsupc], &work[0] );   /* 使用自定义函数计算矩阵向量乘法 */
#endif
    }
#else        

        iptr = istart + nsupc;  // 计算起始索引
        for (i = 0; i < nrow; ++i, ++iptr) {  // 遍历当前超节点中的所有行
            irow = L_SUB(iptr);  // 获取当前行的索引
            x[irow] -= work[i];    /* Scatter */  // 执行散播操作，更新 x 向量
            work[i] = 0.0;  // 将工作数组中的当前元素清零
        }

#endif        
    } /* for k ... */

} else {
    /* Form x := inv(U)*x */

    if ( U->nrow == 0 ) return 0; /* 快速返回，如果 U 的行数为 0 */

    for (k = Lstore->nsuper; k >= 0; k--) {  // 逆序遍历超节点
        fsupc = L_FST_SUPC(k);  // 当前超节点首列
        nsupr = L_SUB_START(fsupc+1) - L_SUB_START(fsupc);  // 当前超节点包含的行数
        nsupc = L_FST_SUPC(k+1) - fsupc;  // 当前超节点包含的列数
        luptr = L_NZ_START(fsupc);  // 当前超节点在 Lval 中的起始位置

        solve_ops += nsupc * (nsupc + 1);  // 更新求解操作数统计

        if ( nsupc == 1 ) {  // 如果超节点只包含一个列
            x[fsupc] /= Lval[luptr];  // 更新 x 向量的值
            for (i = U_NZ_START(fsupc); i < U_NZ_START(fsupc+1); ++i) {  // 遍历当前列的非零元素
                irow = U_SUB(i);  // 获取当前行的索引
                x[irow] -= x[fsupc] * Uval[i];  // 更新 x 向量
            }
        } else {  // 超节点包含多个列
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
            STRSV(ftcs3, ftcs2, ftcs2, &nsupc, &Lval[luptr], &nsupr,
               &x[fsupc], &incx);  // 使用 BLAS 解线性方程组
#else
            strsv_("U", "N", "N", &nsupc, &Lval[luptr], &nsupr,
                           &x[fsupc], &incx);  // 使用 BLAS 解线性方程组
#endif
#else        
            susolve ( nsupr, nsupc, &Lval[luptr], &x[fsupc] );  // 调用自定义的超节点解算函数
#endif        

            for (jcol = fsupc; jcol < L_FST_SUPC(k+1); jcol++) {  // 遍历当前超节点的所有列
                solve_ops += 2 * (U_NZ_START(jcol+1) - U_NZ_START(jcol));  // 更新求解操作数统计
                for (i = U_NZ_START(jcol); i < U_NZ_START(jcol+1); 
                i++) {
                    irow = U_SUB(i);  // 获取当前行的索引
                    x[irow] -= x[jcol] * Uval[i];  // 更新 x 向量
                }
            }
        }
    } /* for k ... */

}

} else { /* Form x := inv(A')*x */

if ( strncmp(uplo, "L", 1)==0 ) {
    /* Form x := inv(L')*x */
    if ( L->nrow == 0 ) return 0; /* 快速返回，如果 L 的行数为 0 */

    for (k = Lstore->nsuper; k >= 0; --k) {  // 逆序遍历超节点
        fsupc = L_FST_SUPC(k);  // 当前超节点首列
        istart = L_SUB_START(fsupc);  // 当前超节点行索引起始位置
        nsupr = L_SUB_START(fsupc+1) - istart;  // 当前超节点包含的行数
        nsupc = L_FST_SUPC(k+1) - fsupc;  // 当前超节点包含的列数
        luptr = L_NZ_START(fsupc);  // 当前超节点在 Lval 中的起始位置

        solve_ops += 2 * (nsupr - nsupc) * nsupc;  // 更新求解操作数统计

        for (jcol = fsupc; jcol < L_FST_SUPC(k+1); jcol++) {  // 遍历当前超节点的所有列
            iptr = istart + nsupc;
            for (i = L_NZ_START(jcol) + nsupc; 
                i < L_NZ_START(jcol+1); i++) {
                irow = L_SUB(iptr);  // 获取当前行的索引
                x[jcol] -= x[irow] * Lval[i];  // 更新 x 向量
                iptr++;
            }
        }

        if ( nsupc > 1 ) {  // 如果超节点包含多个列
            solve_ops += nsupc * (nsupc - 1);  // 更新求解操作数统计
#ifdef _CRAY
                    ftcs1 = _cptofcd("L", strlen("L"));
                    ftcs2 = _cptofcd("T", strlen("T"));
                    ftcs3 = _cptofcd("U", strlen("U"));
            STRSV(ftcs1, ftcs2, ftcs3, &nsupc, &Lval[luptr], &nsupr,
            &x[fsupc], &incx);  // 使用 BLAS 解线性方程组
#else
            strsv_("L", "T", "U", &nsupc, &Lval[luptr], &nsupr,
            &x[fsupc], &incx);  // 使用 BLAS 解线性方程组
#endif
        }
    }
}
    } else {
        /* 如果进入这个分支，则执行以下操作：
           计算 x := inv(U')*x */

        /* 如果 U 的行数为 0，则快速返回 */
        if ( U->nrow == 0 ) return 0;

        /* 对于每个超级节点 k，进行如下操作 */
        for (k = 0; k <= Lstore->nsuper; k++) {
            fsupc = L_FST_SUPC(k);  // 当前超级节点的第一个列索引
            nsupr = L_SUB_START(fsupc+1) - L_SUB_START(fsupc);  // 当前超级节点的行数
            nsupc = L_FST_SUPC(k+1) - fsupc;  // 当前超级节点的列数
            luptr = L_NZ_START(fsupc);  // 当前超级节点对应的 L 值起始索引

            /* 对于当前超级节点的每一列 jcol 进行操作 */
            for (jcol = fsupc; jcol < L_FST_SUPC(k+1); jcol++) {
                solve_ops += 2*(U_NZ_START(jcol+1) - U_NZ_START(jcol));  // 更新 solve_ops 计数
                /* 对于列 jcol 的每个非零元素 i 进行如下操作 */
                for (i = U_NZ_START(jcol); i < U_NZ_START(jcol+1); i++) {
                    irow = U_SUB(i);  // 获取 U 的行索引
                    x[jcol] -= x[irow] * Uval[i];  // 更新 x[jcol] 的值
                }
            }

            solve_ops += nsupc * (nsupc + 1);  // 更新 solve_ops 计数

            /* 如果当前超级节点的列数为 1，则执行如下操作 */
            if ( nsupc == 1 ) {
                x[fsupc] /= Lval[luptr];  // 更新 x[fsupc] 的值
            } else {
#ifdef _CRAY
                    ftcs1 = _cptofcd("U", strlen("U"));
                    ftcs2 = _cptofcd("T", strlen("T"));
                    ftcs3 = _cptofcd("N", strlen("N"));
            // 使用 _cptofcd 将字符串 "U", "T", "N" 转换为 Fortran 字符串描述符
            STRSV( ftcs1, ftcs2, ftcs3, &nsupc, &Lval[luptr], &nsupr,
                &x[fsupc], &incx);
#else
            // 调用 BLAS 函数 strsv_ 执行矩阵向量运算，参数 "U", "T", "N" 指定操作类型
            strsv_("U", "T", "N", &nsupc, &Lval[luptr], &nsupr,
                &x[fsupc], &incx);
#endif
        }
        } /* for k ... */
    }
    }

    // 更新统计信息，记录操作次数
    stat->ops[SOLVE] += solve_ops;
    // 释放动态分配的工作空间
    SUPERLU_FREE(work);
    // 返回操作成功
    return 0;
}


注释：
- `#ifdef _CRAY`: 如果定义了 `_CRAY` 宏，则执行下面的代码块，否则执行 `#else` 之后的代码块。
- `_cptofcd("U", strlen("U"))`, `_cptofcd("T", strlen("T"))`, `_cptofcd("N", strlen("N"))`: 使用 `_cptofcd` 函数将字符串转换为 Fortran 字符串描述符。
- `STRSV`: 调用 `STRSV` 函数执行特定类型的矩阵向量运算，参数包括转换后的字符串描述符和其他数组。
- `strsv_("U", "T", "N", &nsupc, &Lval[luptr], &nsupr, &x[fsupc], &incx)`: 调用 `strsv_` BLAS 函数执行矩阵向量运算，参数是字符常量和数组。
- `stat->ops[SOLVE] += solve_ops;`: 更新 `stat` 结构体中 `ops` 数组的 `SOLVE` 索引处的操作次数统计。
- `SUPERLU_FREE(work);`: 释放通过 `SUPERLU_FREE` 分配的工作空间。
- `return 0;`: 函数返回值，表示操作成功。
/* 
   Sparse matrix-vector multiplication routine.

   Parameters:
   - trans: Specifies the operation type ('N' for no transpose, 'T' for transpose).
   - alpha: Scalar multiplier for the matrix-vector product.
   - A: Pointer to a SuperMatrix structure representing the sparse matrix A.
   - x: Pointer to the vector x.
   - incx: Increment for elements of x.
   - beta: Scalar multiplier for the vector y.
   - y: Pointer to the vector y.
   - incy: Increment for elements of y.

   Local variables:
   - Astore: Pointer to the NCformat structure within A, storing non-zero values.
   - Aval: Pointer to the array of non-zero values in A.
   - info: Integer flag indicating errors in input parameters.
   - temp: Temporary variable for intermediate calculations.
   - lenx, leny: Lengths of vectors x and y respectively.
   - iy, jx, jy, kx, ky, irow: Integer variables used as indices and counters.

   Returns:
   - 0 on successful completion.

   Notes:
   - Handles both transpose and non-transpose operations of sparse matrix A.
   - Validates input parameters and checks for errors.
   - Implements sparse matrix-vector multiplication based on provided operation type.
*/
sp_sgemv(char *trans, float alpha, SuperMatrix *A, float *x, 
         int incx, float beta, float *y, int incy)
{
    /* Local variables */
    NCformat *Astore;
    float   *Aval;
    int info;
    float temp;
    int lenx, leny;
    int iy, jx, jy, kx, ky, irow;
    int_t i, j;
    int notran;

    /* Determine if the operation is notranspose ('N') */
    notran = ( strncmp(trans, "N", 1)==0 || strncmp(trans, "n", 1)==0 );

    /* Retrieve the non-zero values from the SuperMatrix structure */
    Astore = A->Store;
    Aval = Astore->nzval;
    
    /* Test the input parameters */
    info = 0;
    if ( !notran && strncmp(trans, "T", 1)!=0 && strncmp(trans, "C", 1)!=0 )
        info = 1;
    else if ( A->nrow < 0 || A->ncol < 0 ) info = 3;
    else if (incx == 0) info = 5;
    else if (incy == 0)    info = 8;
    if (info != 0) {
        input_error("sp_sgemv ", &info);
        return 0;
    }

    /* Quick return if possible. */
    if (A->nrow == 0 || A->ncol == 0 || (alpha == 0. && beta == 1.))
        return 0;

    /* Set  LENX  and  LENY, the lengths of the vectors x and y, and set 
       up the start points in  X  and  Y. */
    if (strncmp(trans, "N", 1)==0) {
        lenx = A->ncol;
        leny = A->nrow;
    } else {
        lenx = A->nrow;
        leny = A->ncol;
    }
    if (incx > 0) kx = 0;
    else kx =  - (lenx - 1) * incx;
    if (incy > 0) ky = 0;
    else ky =  - (leny - 1) * incy;

    /* Start the operations. In this version the elements of A are   
       accessed sequentially with one pass through A. */

    /* First form  y := beta*y. */
    if (beta != 1.) {
        if (incy == 1) {
            if (beta == 0.)
                for (i = 0; i < leny; ++i) y[i] = 0.;
            else
                for (i = 0; i < leny; ++i) y[i] = beta * y[i];
        } else {
            iy = ky;
            if (beta == 0.)
                for (i = 0; i < leny; ++i) {
                    y[iy] = 0.;
                    iy += incy;
                }
            else
                for (i = 0; i < leny; ++i) {
                    y[iy] = beta * y[iy];
                    iy += incy;
                }
        }
    }
    
    /* Return early if alpha is zero */
    if (alpha == 0.) return 0;

    if ( notran ) {
        /* Form  y := alpha*A*x + y. */
        jx = kx;
        if (incy == 1) {
            for (j = 0; j < A->ncol; ++j) {
                if (x[jx] != 0.) {
                    temp = alpha * x[jx];
                    for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; ++i) {
                        irow = Astore->rowind[i];
                        y[irow] += temp * Aval[i];
                    }
                }
                jx += incx;
            }
        } else {
            ABORT("Not implemented.");
        }
    } else {
        /* Form  y := alpha*A'*x + y. */
        jy = ky;
        if (incx == 1) {
            for (j = 0; j < A->ncol; ++j) {
                temp = 0.;
                for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; ++i) {
                    irow = Astore->rowind[i];
                    temp += Aval[i] * x[irow];
                }
                y[jy] += alpha * temp;
                jy += incy;
            }
        } else {
            ABORT("Not implemented.");
        }
    }

    return 0;
} /* sp_sgemv */
```