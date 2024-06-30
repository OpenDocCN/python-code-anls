# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dsp_blas2.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file dsp_blas2.c
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
 * File name:        dsp_blas2.c
 * Purpose:        Sparse BLAS 2, using some dense BLAS 2 operations.
 */

#include "slu_ddefs.h"
    /*! \brief Solves one of the systems of equations A*x = b,   or   A'*x = b
     * 
     * <pre>
     *   Purpose
     *   =======
     *
     *   sp_dtrsv() solves one of the systems of equations   
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
     *             i.e., L has types: Stype = SC, Dtype = SLU_D, Mtype = TRLU.
     *
     *   U       - (input) SuperMatrix*
     *            The factor U from the factorization Pr*A*Pc=L*U.
     *            U has types: Stype = NC, Dtype = SLU_D, Mtype = TRU.
     *    
     *   x       - (input/output) double*
     *             Before entry, the incremented array X must contain the n   
     *             element right-hand side vector b. On exit, X is overwritten 
     *             with the solution vector x.
     *
     *   info    - (output) int*
     *             If *info = -i, the i-th argument had an illegal value.
     * </pre>
     */
    int
    sp_dtrsv(char *uplo, char *trans, char *diag, SuperMatrix *L, 
             SuperMatrix *U, double *x, SuperLUStat_t *stat, int *info)
    {
#ifdef _CRAY
        _fcd ftcs1 = _cptofcd("L", strlen("L")),
             ftcs2 = _cptofcd("N", strlen("N")),
             ftcs3 = _cptofcd("U", strlen("U"));
#endif
        SCformat *Lstore;
        NCformat *Ustore;
        double   *Lval, *Uval;
        int incx = 1, incy = 1;
        double alpha = 1.0, beta = 1.0;
        int nrow, irow, jcol;
        int fsupc, nsupr, nsupc;
        int_t luptr, istart, i, k, iptr;
        double *work;
        flops_t solve_ops;

        /* Test the input parameters */
        *info = 0;  // Initialize info to zero
        if ( strncmp(uplo,"L", 1)!=0 && strncmp(uplo, "U", 1)!=0 ) *info = -1;  // Check if uplo is either 'L' or 'U'
    else if ( strncmp(trans, "N", 1)!=0 && strncmp(trans, "T", 1)!=0 && 
              strncmp(trans, "C", 1)!=0) *info = -2;
    # 如果 `trans` 不是 "N", "T", 或者 "C" 中的任意一个，将 `info` 设为 -2

    else if ( strncmp(diag, "U", 1)!=0 && strncmp(diag, "N", 1)!=0 )
         *info = -3;
    # 如果 `diag` 不是 "U" 或者 "N"，将 `info` 设为 -3

    else if ( L->nrow != L->ncol || L->nrow < 0 ) *info = -4;
    # 如果矩阵 `L` 的行数不等于列数，或者行数小于 0，将 `info` 设为 -4

    else if ( U->nrow != U->ncol || U->nrow < 0 ) *info = -5;
    # 如果矩阵 `U` 的行数不等于列数，或者行数小于 0，将 `info` 设为 -5

    if ( *info ) {
    int ii = -(*info);
    input_error("sp_dtrsv", &ii);
    return 0;
    }
    # 如果 `info` 不为 0，则输出错误信息并返回 0

    Lstore = L->Store;
    Lval = Lstore->nzval;
    Ustore = U->Store;
    Uval = Ustore->nzval;
    solve_ops = 0;
    # 初始化存储 L 和 U 矩阵的结构体以及相应的非零元素数组，并初始化解操作数为 0

    if ( !(work = doubleCalloc(L->nrow)) )
    ABORT("Malloc fails for work in sp_dtrsv().");
    # 分配存储空间 `work`，如果分配失败则终止程序并输出错误信息

    if ( strncmp(trans, "N", 1)==0 ) {    /* Form x := inv(A)*x. */
    # 如果 `trans` 是 "N"，表示要求解 x := inv(A)*x

    if ( strncmp(uplo, "L", 1)==0 ) {
        /* Form x := inv(L)*x */
            if ( L->nrow == 0 ) return 0; /* Quick return */
        # 如果 `uplo` 是 "L"，表示要求解 x := inv(L)*x，如果 `L` 的行数为 0，则快速返回

        for (k = 0; k <= Lstore->nsuper; k++) {
        # 对 `L` 的超节点进行循环处理
        fsupc = L_FST_SUPC(k);
        istart = L_SUB_START(fsupc);
        nsupr = L_SUB_START(fsupc+1) - istart;
        nsupc = L_FST_SUPC(k+1) - fsupc;
        luptr = L_NZ_START(fsupc);
        nrow = nsupr - nsupc;

            solve_ops += nsupc * (nsupc - 1);
            solve_ops += 2 * nrow * nsupc;
        # 更新解操作数 `solve_ops` 的统计值

        if ( nsupc == 1 ) {
            for (iptr=istart+1; iptr < L_SUB_START(fsupc+1); ++iptr) {
            irow = L_SUB(iptr);
            ++luptr;
            x[irow] -= x[fsupc] * Lval[luptr];
            }
        } else {
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
            STRSV(ftcs1, ftcs2, ftcs3, &nsupc, &Lval[luptr], &nsupr,
                   &x[fsupc], &incx);
        
            SGEMV(ftcs2, &nrow, &nsupc, &alpha, &Lval[luptr+nsupc], 
                   &nsupr, &x[fsupc], &incx, &beta, &work[0], &incy);
#else
            // 使用 BLAS 函数 dtrsv 解决线性方程 L * x = b，其中 L 是下三角矩阵
            dtrsv_("L", "N", "U", &nsupc, &Lval[luptr], &nsupr,
                   &x[fsupc], &incx);
        
            // 使用 BLAS 函数 dgemv 计算矩阵向量乘积 y = alpha * A * x + beta * y，其中 A 是矩阵 L 的一部分
            dgemv_("N", &nrow, &nsupc, &alpha, &Lval[luptr+nsupc], 
                   &nsupr, &x[fsupc], &incx, &beta, &work[0], &incy);
#endif
#else
            // 调用自定义函数 dlsolve 解决线性方程 L * x = b，其中 L 是下三角矩阵
            dlsolve ( nsupr, nsupc, &Lval[luptr], &x[fsupc]);
        
            // 调用自定义函数 dmatvec 计算矩阵向量乘积 y = A * x，其中 A 是矩阵 L 的一部分
            dmatvec ( nsupr, nsupr-nsupc, nsupc, &Lval[luptr+nsupc],
                             &x[fsupc], &work[0] );
#endif        
        
            // 处理 x 的更新：x[irow] -= work[i]，用于散射操作
            iptr = istart + nsupc;
            for (i = 0; i < nrow; ++i, ++iptr) {
            // 获取行索引 irow
            irow = L_SUB(iptr);
            // 更新 x[irow]，用 work[i] 进行减法操作
            x[irow] -= work[i];    /* Scatter */
            // 将 work[i] 重置为零
            work[i] = 0.0;

            }
         }
        } /* for k ... */
        
    } else {
        /* Form x := inv(U)*x */
        
        if ( U->nrow == 0 ) return 0; /* Quick return */
        
        for (k = Lstore->nsuper; k >= 0; k--) {
            // 获取第 k 列的起始列索引 fsupc，以及该列的行数 nsupr 和列数 nsupc
            fsupc = L_FST_SUPC(k);
            nsupr = L_SUB_START(fsupc+1) - L_SUB_START(fsupc);
            nsupc = L_FST_SUPC(k+1) - fsupc;
            // 获取第 k 列在 L 储存结构中的起始索引 luptr
            luptr = L_NZ_START(fsupc);
        
                // 更新 solve_ops 计数，每个 nsupc * (nsupc + 1) 次数
                solve_ops += nsupc * (nsupc + 1);

        if ( nsupc == 1 ) {
            // 如果 nsupc == 1，即当前处理的列只有一个元素
            // 直接计算 x[fsupc] /= Lval[luptr]
            x[fsupc] /= Lval[luptr];
            // 遍历与 fsupc 相关的行 irow，更新 x[irow]
            for (i = U_NZ_START(fsupc); i < U_NZ_START(fsupc+1); ++i) {
            // 获取行索引 irow
            irow = U_SUB(i);
            // 更新 x[irow] -= x[fsupc] * Uval[i]
            x[irow] -= x[fsupc] * Uval[i];
            }
        } else {
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
            // 使用 BLAS 函数 STRSV 解决线性方程 U * x = b，其中 U 是上三角矩阵
            STRSV(ftcs3, ftcs2, ftcs2, &nsupc, &Lval[luptr], &nsupr,
               &x[fsupc], &incx);
#else
            // 使用 BLAS 函数 dtrsv 解决线性方程 U * x = b，其中 U 是上三角矩阵
            dtrsv_("U", "N", "N", &nsupc, &Lval[luptr], &nsupr,
                           &x[fsupc], &incx);
#endif
#else        
            // 调用自定义函数 dusolve 解决线性方程 U * x = b，其中 U 是上三角矩阵
            dusolve ( nsupr, nsupc, &Lval[luptr], &x[fsupc] );
#endif        

            // 更新 x 的其他元素，处理列 jcol 的元素
            for (jcol = fsupc; jcol < L_FST_SUPC(k+1); jcol++) {
                // 更新 solve_ops 计数，每个 jcol 的更新需要处理 2*(U_NZ_START(jcol+1) - U_NZ_START(jcol)) 次数
                solve_ops += 2*(U_NZ_START(jcol+1) - U_NZ_START(jcol));
                // 遍历与 jcol 相关的行 irow，更新 x[irow]
                for (i = U_NZ_START(jcol); i < U_NZ_START(jcol+1); 
                i++) {
                // 获取行索引 irow
                irow = U_SUB(i);
                // 更新 x[irow] -= x[jcol] * Uval[i]
                x[irow] -= x[jcol] * Uval[i];
                }
                    }
        }
        } /* for k ... */
        
    }
    } else { /* Form x := inv(A')*x */
    
    // 如果是求解 x := inv(A')*x，A' 为当前矩阵的转置
    
    # 如果 uplo 的首字母是 'L'
    if ( strncmp(uplo, "L", 1)==0 ) {
        # 执行 x := inv(L')*x 的操作，即求解下三角矩阵 L 的逆的转置与向量 x 的乘积
        # 如果 L 的行数为 0，则快速返回
            if ( L->nrow == 0 ) return 0; /* Quick return */
        
        # 从最后一个超级节点开始向前遍历
        for (k = Lstore->nsuper; k >= 0; --k) {
            # 当前超级节点的第一个列索引
            fsupc = L_FST_SUPC(k);
            # 当前超级节点包含的行索引起始位置
            istart = L_SUB_START(fsupc);
            # 当前超级节点包含的行索引个数
            nsupr = L_SUB_START(fsupc+1) - istart;
            # 当前超级节点包含的列索引个数
            nsupc = L_FST_SUPC(k+1) - fsupc;
            # 当前超级节点第一个非零元素在 Lval 数组中的位置
            luptr = L_NZ_START(fsupc);

        # 更新 solve_ops 计数，用于性能分析，表示乘法操作次数的估计
        solve_ops += 2 * (nsupr - nsupc) * nsupc;

        # 遍历当前超级节点包含的所有列
        for (jcol = fsupc; jcol < L_FST_SUPC(k+1); jcol++) {
            # 当前列的行索引起始位置，从超级节点的行索引个数开始
            iptr = istart + nsupc;
            # 遍历当前列中非零元素的行索引
            for (i = L_NZ_START(jcol) + nsupc; 
                i < L_NZ_START(jcol+1); i++) {
            # 当前非零元素的行索引
            irow = L_SUB(iptr);
            # 更新向量 x 中的元素，执行 x[jcol] -= x[irow] * Lval[i]
            x[jcol] -= x[irow] * Lval[i];
            # 指向下一个行索引
            iptr++;
            }
        }
        
        # 如果当前超级节点包含的列索引个数大于 1
        if ( nsupc > 1 ) {
            # 更新 solve_ops 计数，用于性能分析，表示额外的乘法操作次数的估计
            solve_ops += nsupc * (nsupc - 1);
#ifdef _CRAY
                    ftcs1 = _cptofcd("L", strlen("L"));
                    ftcs2 = _cptofcd("T", strlen("T"));
                    ftcs3 = _cptofcd("U", strlen("U"));
            // 调用特定平台下的线性方程求解函数 STRSV，解决 L * x = U
            STRSV(ftcs1, ftcs2, ftcs3, &nsupc, &Lval[luptr], &nsupr,
            &x[fsupc], &incx);
#else
            // 调用通用的双精度矩阵向量乘法函数 dtrsv，解决 L * x = U
            dtrsv_("L", "T", "U", &nsupc, &Lval[luptr], &nsupr,
            &x[fsupc], &incx);
#endif
        }
        }
    } else {
        /* Form x := inv(U')*x */
        if ( U->nrow == 0 ) return 0; /* 快速返回 */

        // 遍历每个超节点 k
        for (k = 0; k <= Lstore->nsuper; k++) {
            fsupc = L_FST_SUPC(k);  // 当前超节点的第一列索引
            nsupr = L_SUB_START(fsupc+1) - L_SUB_START(fsupc);  // 当前超节点包含的行数
            nsupc = L_FST_SUPC(k+1) - fsupc;  // 当前超节点的列数
            luptr = L_NZ_START(fsupc);  // 当前超节点在 Lval 中的起始索引

            // 遍历超节点 k 中的每一列 jcol
            for (jcol = fsupc; jcol < L_FST_SUPC(k+1); jcol++) {
                // 计算解操作次数的增加量
                solve_ops += 2 * (U_NZ_START(jcol+1) - U_NZ_START(jcol));
                // 遍历列 jcol 中的每个非零元素 i
                for (i = U_NZ_START(jcol); i < U_NZ_START(jcol+1); i++) {
                    irow = U_SUB(i);  // 获取非零元素的行索引
                    x[jcol] -= x[irow] * Uval[i];  // 计算 x[jcol] 的值
                }
            }

            // 计算解操作次数的增加量
            solve_ops += nsupc * (nsupc + 1);

            // 如果当前超节点只有一个列，直接求解 x[fsupc]
            if ( nsupc == 1 ) {
                x[fsupc] /= Lval[luptr];
            } else {
#ifdef _CRAY
                    ftcs1 = _cptofcd("U", strlen("U"));
                    ftcs2 = _cptofcd("T", strlen("T"));
                    ftcs3 = _cptofcd("N", strlen("N"));
            // 调用特定平台下的线性方程求解函数 STRSV，解决 U * x = L
            STRSV( ftcs1, ftcs2, ftcs3, &nsupc, &Lval[luptr], &nsupr,
                &x[fsupc], &incx);
#else
            // 调用通用的双精度矩阵向量乘法函数 dtrsv，解决 U * x = L
            dtrsv_("U", "T", "N", &nsupc, &Lval[luptr], &nsupr,
                &x[fsupc], &incx);
#endif
        }
        } /* for k ... */
    }
    }

    stat->ops[SOLVE] += solve_ops;  // 更新解操作统计信息
    SUPERLU_FREE(work);  // 释放工作内存
    return 0;  // 返回成功
}
/*! \brief Performs one of the matrix-vector operations y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,   
 *
 * <pre>
 *   Purpose   
 *   =======   
 *
 *   sp_dgemv()  performs one of the matrix-vector operations   
 *      y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,   
 *   where alpha and beta are scalars, x and y are vectors and A is a
 *   sparse A->nrow by A->ncol matrix.   
 *
 *   Parameters   
 *   ==========   
 *
 *   TRANS  - (input) char*
 *            On entry, TRANS specifies the operation to be performed as   
 *            follows:   
 *               TRANS = 'N' or 'n'   y := alpha*A*x + beta*y.   
 *               TRANS = 'T' or 't'   y := alpha*A'*x + beta*y.   
 *               TRANS = 'C' or 'c'   y := alpha*A'*x + beta*y.   
 *
 *   ALPHA  - (input) double
 *            On entry, ALPHA specifies the scalar alpha.   
 *
 *   A      - (input) SuperMatrix*
 *            Matrix A with a sparse format, of dimension (A->nrow, A->ncol).
 *            Currently, the type of A can be:
 *                Stype = NC or NCP; Dtype = SLU_D; Mtype = GE. 
 *            In the future, more general A can be handled.
 *
 *   X      - (input) double*, array of DIMENSION at least   
 *            ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'   
 *            and at least   
 *            ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.   
 *            Before entry, the incremented array X must contain the   
 *            vector x.   
 *
 *   INCX   - (input) int
 *            On entry, INCX specifies the increment for the elements of   
 *            X. INCX must not be zero.   
 *
 *   BETA   - (input) double
 *            On entry, BETA specifies the scalar beta. When BETA is   
 *            supplied as zero then Y need not be set on input.   
 *
 *   Y      - (output) double*,  array of DIMENSION at least   
 *            ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'   
 *            and at least   
 *            ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.   
 *            Before entry with BETA non-zero, the incremented array Y   
 *            must contain the vector y. On exit, Y is overwritten by the 
 *            updated vector y.
 *         
 *   INCY   - (input) int
 *            On entry, INCY specifies the increment for the elements of   
 *            Y. INCY must not be zero.   
 *
 *   ==== Sparse Level 2 Blas routine.   
 * </pre>
 */

int
sp_dgemv(char *trans, double alpha, SuperMatrix *A, double *x, 
     int incx, double beta, double *y, int incy)
{
    /* Local variables */
    NCformat *Astore; // 存储 A 矩阵的非零元素的稀疏格式
    double   *Aval;   // A 矩阵的非零元素值
    int info;         // 返回的信息代码
    double temp;      // 临时变量
    int lenx, leny;   // x 和 y 向量的长度
    int iy, jx, jy, kx, ky, irow; // 循环计数器和索引变量
    int_t i, j;       // 循环索引变量
    int notran;       // 是否为非转置操作标志

    // 根据 TRANS 的值判断是否为非转置操作
    notran = ( strncmp(trans, "N", 1)==0 || strncmp(trans, "n", 1)==0 );
    // 获取 A 矩阵的存储格式
    Astore = A->Store;
    // 获取 A 矩阵的非零元素数组
    Aval = Astore->nzval;
    
    /* Test the input parameters */
    info = 0; // 初始化 info 为 0，表示输入参数正确
    /* 检查是否需要转置操作，如果需要且不支持，则设置 info 为 1 */
    if (!notran && strncmp(trans, "T", 1) != 0 && strncmp(trans, "C", 1) != 0)
        info = 1;
    /* 检查矩阵 A 的行数或列数是否为负数，若是则设置 info 为 3 */
    else if (A->nrow < 0 || A->ncol < 0)
        info = 3;
    /* 检查向量 x 的增量是否为零，若是则设置 info 为 5 */
    else if (incx == 0)
        info = 5;
    /* 检查向量 y 的增量是否为零，若是则设置 info 为 8 */
    else if (incy == 0)
        info = 8;

    /* 如果 info 不为零，则输出输入错误信息并返回 */
    if (info != 0) {
        input_error("sp_dgemv ", &info);
        return 0;
    }

    /* 如果矩阵 A 的行数或列数为零，或者 alpha 为零且 beta 为 1，则快速返回 */
    if (A->nrow == 0 || A->ncol == 0 || (alpha == 0. && beta == 1.))
        return 0;

    /* 设置向量 x 和 y 的长度 lenx 和 leny，以及它们的起始点 kx 和 ky */
    if (strncmp(trans, "N", 1) == 0) {
        lenx = A->ncol;
        leny = A->nrow;
    } else {
        lenx = A->nrow;
        leny = A->ncol;
    }
    if (incx > 0)
        kx = 0;
    else
        kx = -(lenx - 1) * incx;
    if (incy > 0)
        ky = 0;
    else
        ky = -(leny - 1) * incy;

    /* 开始运算。在此版本中，按顺序访问矩阵 A 的元素。 */

    /* 首先计算 y := beta*y */
    if (beta != 1.) {
        if (incy == 1) {
            if (beta == 0.)
                for (i = 0; i < leny; ++i)
                    y[i] = 0.;
            else
                for (i = 0; i < leny; ++i)
                    y[i] = beta * y[i];
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

    /* 如果 alpha 为零，则直接返回 */
    if (alpha == 0.)
        return 0;

    /* 根据 notran 的值选择不同的操作方式 */

    if (notran) {
        /* 计算 y := alpha*A*x + y */
        jx = kx;
        if (incy == 1) {
            for (j = 0; j < A->ncol; ++j) {
                if (x[jx] != 0.) {
                    temp = alpha * x[jx];
                    for (i = Astore->colptr[j]; i < Astore->colptr[j + 1]; ++i) {
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
        /* 计算 y := alpha*A'*x + y */
        jy = ky;
        if (incx == 1) {
            for (j = 0; j < A->ncol; ++j) {
                temp = 0.;
                for (i = Astore->colptr[j]; i < Astore->colptr[j + 1]; ++i) {
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
} /* sp_dgemv */
```