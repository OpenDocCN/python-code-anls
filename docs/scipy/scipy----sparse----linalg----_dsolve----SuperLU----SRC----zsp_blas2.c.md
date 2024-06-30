# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zsp_blas2.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file zsp_blas2.c
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
 * File name:        zsp_blas2.c
 * Purpose:        Sparse BLAS 2, using some dense BLAS 2 operations.
 */

#include "slu_zdefs.h"


注释：


# 包含版权声明和许可信息的文件头部
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

# 包含文件说明和简要描述的文件注释部分
/*! @file zsp_blas2.c
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

# 包含文件名和文件用途描述的注释
/*
 * File name:        zsp_blas2.c
 * Purpose:        Sparse BLAS 2, using some dense BLAS 2 operations.
 */

# 包含所需的头文件 "slu_zdefs.h"
#include "slu_zdefs.h"
/*! \brief Solves one of the systems of equations A*x = b,   or   A'*x = b
 * 
 * <pre>
 *   Purpose
 *   =======
 *
 *   sp_ztrsv() solves one of the systems of equations   
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
 *                trans = 'C' or 'c'   A^H*x = b.   
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
 *             i.e., L has types: Stype = SC, Dtype = SLU_Z, Mtype = TRLU.
 *
 *   U       - (input) SuperMatrix*
 *            The factor U from the factorization Pr*A*Pc=L*U.
 *            U has types: Stype = NC, Dtype = SLU_Z, Mtype = TRU.
 *    
 *   x       - (input/output) doublecomplex*
 *             Before entry, the incremented array X must contain the n   
 *             element right-hand side vector b. On exit, X is overwritten 
 *             with the solution vector x.
 *
 *   stat    - (input) SuperLUStat_t*
 *             SuperLU statistics and timing information.
 *
 *   info    - (output) int*
 *             If *info = -i, the i-th argument had an illegal value.
 * </pre>
 */
int
sp_ztrsv(char *uplo, char *trans, char *diag, SuperMatrix *L, 
         SuperMatrix *U, doublecomplex *x, SuperLUStat_t *stat, int *info)
{
#ifdef _CRAY
    _fcd ftcs1 = _cptofcd("L", strlen("L")),
         ftcs2 = _cptofcd("N", strlen("N")),
         ftcs3 = _cptofcd("U", strlen("U"));
#endif
    SCformat *Lstore;
    NCformat *Ustore;
    doublecomplex   *Lval, *Uval;
    int incx = 1, incy = 1; // 定义增量
    doublecomplex temp; // 临时变量
    doublecomplex alpha = {1.0, 0.0}, beta = {1.0, 0.0}; // 定义复数常量
    doublecomplex comp_zero = {0.0, 0.0}; // 定义复数零
    int nrow, irow, jcol; // 行数和行索引，列索引
    int fsupc, nsupr, nsupc; // 超节点首行索引、行数、列数
    int_t luptr, istart, i, k, iptr; // 整数类型变量
    doublecomplex *work; // 工作数组
    flops_t solve_ops; // 浮点运算计数

    /* Test the input parameters */
    *info = 0; // 初始化 info 为零，用于指示参数合法性
    # 检查uplo参数是否不是"L"或"U"，如果是则设置错误码为-1
    if ( strncmp(uplo,"L", 1)!=0 && strncmp(uplo, "U", 1)!=0 ) *info = -1;
    # 检查trans参数是否不是"N"、"T"或"C"，如果是则设置错误码为-2
    else if ( strncmp(trans, "N", 1)!=0 && strncmp(trans, "T", 1)!=0 && 
              strncmp(trans, "C", 1)!=0) *info = -2;
    # 检查diag参数是否不是"U"或"N"，如果是则设置错误码为-3
    else if ( strncmp(diag, "U", 1)!=0 && strncmp(diag, "N", 1)!=0 )
         *info = -3;
    # 检查L的行数和列数是否相等且大于等于0，如果不是则设置错误码为-4
    else if ( L->nrow != L->ncol || L->nrow < 0 ) *info = -4;
    # 检查U的行数和列数是否相等且大于等于0，如果不是则设置错误码为-5
    else if ( U->nrow != U->ncol || U->nrow < 0 ) *info = -5;
    
    # 如果有任何一个错误码被设置，则执行以下操作
    if ( *info ) {
        # 将错误码的绝对值取负后作为输入错误的参数，调用input_error函数
        int ii = -(*info);
        input_error("sp_ztrsv", &ii);
        # 返回0表示执行失败
        return 0;
    }

    # 从L中获取存储对象和非零值
    Lstore = L->Store;
    Lval = Lstore->nzval;
    # 从U中获取存储对象和非零值
    Ustore = U->Store;
    Uval = Ustore->nzval;
    # 解算操作数初始化为0
    solve_ops = 0;

    # 分配工作空间work，如果分配失败则调用ABORT函数
    if ( !(work = doublecomplexCalloc(L->nrow)) )
        ABORT("Malloc fails for work in sp_ztrsv().");
    
    # 如果trans为"N"，即要解的是x := inv(A)*x
    if ( strncmp(trans, "N", 1)==0 ) {
        /* Form x := inv(A)*x. */

        # 如果uplo为"L"，即要解的是x := inv(L)*x
        if ( strncmp(uplo, "L", 1)==0 ) {
            /* Form x := inv(L)*x */
            # 如果L的行数为0，直接返回，快速返回
            if ( L->nrow == 0 ) return 0; /* Quick return */
        
            # 对L的每个超节点进行循环
            for (k = 0; k <= Lstore->nsuper; k++) {
                fsupc = L_FST_SUPC(k);
                istart = L_SUB_START(fsupc);
                nsupr = L_SUB_START(fsupc+1) - istart;
                nsupc = L_FST_SUPC(k+1) - fsupc;
                luptr = L_NZ_START(fsupc);
                nrow = nsupr - nsupc;

                # 计算解算操作数，根据不同的情况计算浮点操作次数
                solve_ops += 4 * nsupc * (nsupc - 1) + 10 * nsupc;
                solve_ops += 8 * nrow * nsupc;

                # 如果超节点只有一个列，使用特定算法处理
                if ( nsupc == 1 ) {
                    for (iptr=istart+1; iptr < L_SUB_START(fsupc+1); ++iptr) {
                        irow = L_SUB(iptr);
                        ++luptr;
                        zz_mult(&comp_zero, &x[fsupc], &Lval[luptr]);
                        z_sub(&x[irow], &x[irow], &comp_zero);
                    }
                } else {
        /* Form x := inv(A)*x */

        if ( L->nrow == 0 ) return 0; /* Quick return */

        /* Loop over all supernodes */
        for (k = Lstore->nsuper; k >= 0; k--) {
            fsupc = L_FST_SUPC(k);
            nsupr = L_SUB_START(fsupc+1) - L_SUB_START(fsupc);
            nsupc = L_FST_SUPC(k+1) - fsupc;
            luptr = L_NZ_START(fsupc);
        
                /* 1 z_div costs 10 flops */
                solve_ops += 4 * nsupc * (nsupc + 1) + 10 * nsupc;

        if ( nsupc == 1 ) {
            /* Handle the case of a single supernodal column */
            z_div(&x[fsupc], &x[fsupc], &Lval[luptr]);
            /* Perform the forward substitution for the remaining part of U */
            for (i = U_NZ_START(fsupc); i < U_NZ_START(fsupc+1); ++i) {
                irow = U_SUB(i);
                /* Multiply and subtract to update x */
                zz_mult(&comp_zero, &x[fsupc], &Uval[i]);
                z_sub(&x[irow], &x[irow], &comp_zero);
            }
        } else {
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
            /* Solve using CTRSV (Cray-specific) */
            CTRSV(ftcs3, ftcs2, ftcs2, &nsupc, &Lval[luptr], &nsupr,
               &x[fsupc], &incx);
#else
            /* Solve using ztrsv (generic case) */
            ztrsv_("U", "N", "N", &nsupc, &Lval[luptr], &nsupr,
                           &x[fsupc], &incx);
#endif
#else        
            /* Solve using custom solver zusolve */
            zusolve ( nsupr, nsupc, &Lval[luptr], &x[fsupc] );
#endif        

            /* Perform the updates using Uval and x */
            for (jcol = fsupc; jcol < L_FST_SUPC(k+1); jcol++) {
                /* Calculate operations count */
                solve_ops += 8*(U_NZ_START(jcol+1) - U_NZ_START(jcol));
                /* Loop over each nonzero entry in the current column of U */
                for (i = U_NZ_START(jcol); i < U_NZ_START(jcol+1); i++) {
                    irow = U_SUB(i);
                    /* Multiply and subtract to update x */
                    zz_mult(&comp_zero, &x[jcol], &Uval[i]);
                    z_sub(&x[irow], &x[irow], &comp_zero);
                }
            }
        }
        } /* End of loop over supernodes */
        
    }
    } else if ( strncmp(trans, "T", 1)==0 ) { /* Form x := inv(A')*x */
    # 如果 uplo 的前一个字符是 'L'，则执行以下代码块
    if ( strncmp(uplo, "L", 1)==0 ) {
        # 计算 x := inv(L')*x，其中 L 是下三角矩阵的存储结构

        # 如果 L 的行数为 0，则快速返回
        if ( L->nrow == 0 ) return 0;

        # 从最后一个超节点开始向前遍历
        for (k = Lstore->nsuper; k >= 0; --k) {
            fsupc = L_FST_SUPC(k);
            istart = L_SUB_START(fsupc);
            nsupr = L_SUB_START(fsupc+1) - istart;
            nsupc = L_FST_SUPC(k+1) - fsupc;
            luptr = L_NZ_START(fsupc);

            # 计算解算操作次数的增加量
            solve_ops += 8 * (nsupr - nsupc) * nsupc;

            # 对当前超节点内的每一列进行处理
            for (jcol = fsupc; jcol < L_FST_SUPC(k+1); jcol++) {
                iptr = istart + nsupc;
                # 对当前列 jcol 的非零元素进行处理
                for (i = L_NZ_START(jcol) + nsupc; 
                    i < L_NZ_START(jcol+1); i++) {
                    irow = L_SUB(iptr);
                    # 执行复数数乘操作
                    zz_mult(&comp_zero, &x[irow], &Lval[i]);
                    # 执行复数减法操作
                    z_sub(&x[jcol], &x[jcol], &comp_zero);
                    iptr++;
                }
            }
        
            # 如果当前超节点的列数大于 1，则增加解算操作次数的另一部分
            if ( nsupc > 1 ) {
                solve_ops += 4 * nsupc * (nsupc - 1);
        # 如果定义了 _CRAY 宏，则使用 Cray 平台特定的函数和参数
#ifdef _CRAY
                    ftcs1 = _cptofcd("L", strlen("L"));
                    ftcs2 = _cptofcd("T", strlen("T"));
                    ftcs3 = _cptofcd("U", strlen("U"));
            CTRSV(ftcs1, ftcs2, ftcs3, &nsupc, &Lval[luptr], &nsupr,
            &x[fsupc], &incx);
#else
            # 否则，使用通用的 ztrsv 函数来解三角矩阵方程
            ztrsv_("L", "T", "U", &nsupc, &Lval[luptr], &nsupr,
            &x[fsupc], &incx);
#endif
        }
        }
    } else {
        /* Form x := inv(U')*x */
        if ( U->nrow == 0 ) return 0; /* 快速返回 */

        # 对每个超级节点 k 执行下列操作
        for (k = 0; k <= Lstore->nsuper; k++) {
            fsupc = L_FST_SUPC(k);
            nsupr = L_SUB_START(fsupc+1) - L_SUB_START(fsupc);
            nsupc = L_FST_SUPC(k+1) - fsupc;
            luptr = L_NZ_START(fsupc);

        # 对当前超级节点中的每一列 jcol 执行下列操作
        for (jcol = fsupc; jcol < L_FST_SUPC(k+1); jcol++) {
            solve_ops += 8*(U_NZ_START(jcol+1) - U_NZ_START(jcol));
            # 对列 jcol 的每一个非零元素 i 执行下列操作
            for (i = U_NZ_START(jcol); i < U_NZ_START(jcol+1); i++) {
            irow = U_SUB(i);
            zz_mult(&comp_zero, &x[irow], &Uval[i]);
                z_sub(&x[jcol], &x[jcol], &comp_zero);
            }
        }

                /* 1 z_div costs 10 flops */
        solve_ops += 4 * nsupc * (nsupc + 1) + 10 * nsupc;

        # 如果当前超级节点只有一个列 nsupc == 1，则执行一次 z_div 运算
        if ( nsupc == 1 ) {
            z_div(&x[fsupc], &x[fsupc], &Lval[luptr]);
        } else {
#ifdef _CRAY
                    ftcs1 = _cptofcd("U", strlen("U"));
                    ftcs2 = _cptofcd("T", strlen("T"));
                    ftcs3 = _cptofcd("N", strlen("N"));
            CTRSV( ftcs1, ftcs2, ftcs3, &nsupc, &Lval[luptr], &nsupr,
                &x[fsupc], &incx);
#else
            # 否则，使用通用的 ztrsv 函数来解三角矩阵方程
            ztrsv_("U", "T", "N", &nsupc, &Lval[luptr], &nsupr,
                &x[fsupc], &incx);
#endif
        }
        } /* for k ... */
    }
    } else { /* Form x := conj(inv(A'))*x */
    
    # 如果 uplo 的首字符是 'L'，则执行下列操作
    if ( strncmp(uplo, "L", 1)==0 ) {
        /* Form x := conj(inv(L'))*x */
            if ( L->nrow == 0 ) return 0; /* 快速返回 */
        
        # 对每个超级节点 k 执行下列操作
        for (k = Lstore->nsuper; k >= 0; --k) {
            fsupc = L_FST_SUPC(k);
            istart = L_SUB_START(fsupc);
            nsupr = L_SUB_START(fsupc+1) - istart;
            nsupc = L_FST_SUPC(k+1) - fsupc;
            luptr = L_NZ_START(fsupc);

        solve_ops += 8 * (nsupr - nsupc) * nsupc;

        # 对当前超级节点中的每一列 jcol 执行下列操作
        for (jcol = fsupc; jcol < L_FST_SUPC(k+1); jcol++) {
            iptr = istart + nsupc;
            # 对列 jcol 的每一个非零元素 i 执行下列操作
            for (i = L_NZ_START(jcol) + nsupc; 
                i < L_NZ_START(jcol+1); i++) {
            irow = L_SUB(iptr);
                        zz_conj(&temp, &Lval[i]);
            zz_mult(&comp_zero, &x[irow], &temp);
                z_sub(&x[jcol], &x[jcol], &comp_zero);
            iptr++;
            }
         }
         
         # 如果当前超级节点的列数 nsupc 大于 1，则执行下列操作
         if ( nsupc > 1 ) {
            solve_ops += 4 * nsupc * (nsupc - 1);
#ifdef _CRAY
                    ftcs1 = _cptofcd("L", strlen("L"));
                    // 将字符串"L"转换为对应的 _C_TYPE_CHAR 类型，并计算其长度，赋值给 ftcs1
                    ftcs2 = _cptofcd(trans, strlen("T"));
                    // 将字符串 trans 转换为 _C_TYPE_CHAR 类型，并计算其长度，赋值给 ftcs2
                    ftcs3 = _cptofcd("U", strlen("U"));
                    // 将字符串"U"转换为 _C_TYPE_CHAR 类型，并计算其长度，赋值给 ftcs3
            ZTRSV(ftcs1, ftcs2, ftcs3, &nsupc, &Lval[luptr], &nsupr,
            &x[fsupc], &incx);
            // 调用 ZTRSV 函数进行三角矩阵的求解，参数由上述转换后的字符类型和其他变量构成
#else
                    ztrsv_("L", trans, "U", &nsupc, &Lval[luptr], &nsupr,
                           &x[fsupc], &incx);
                    // 调用 ztrsv_ 函数进行三角矩阵的求解，使用普通的字符串参数表示
#endif
        }
        }
    } else {
        /* Form x := conj(inv(U'))*x */
        if ( U->nrow == 0 ) return 0; /* Quick return */
        
        for (k = 0; k <= Lstore->nsuper; k++) {
            fsupc = L_FST_SUPC(k);
            nsupr = L_SUB_START(fsupc+1) - L_SUB_START(fsupc);
            nsupc = L_FST_SUPC(k+1) - fsupc;
            luptr = L_NZ_START(fsupc);

        for (jcol = fsupc; jcol < L_FST_SUPC(k+1); jcol++) {
            solve_ops += 8*(U_NZ_START(jcol+1) - U_NZ_START(jcol));
            // 计算 solve_ops 值增加的操作次数
            for (i = U_NZ_START(jcol); i < U_NZ_START(jcol+1); i++) {
            irow = U_SUB(i);
                        zz_conj(&temp, &Uval[i]);
            // 调用 zz_conj 函数求取 Uval[i] 的共轭，并存储在 temp 中
            zz_mult(&comp_zero, &x[irow], &temp);
            // 调用 zz_mult 函数计算 x[irow] 与 temp 的乘积，并存储在 comp_zero 中
                z_sub(&x[jcol], &x[jcol], &comp_zero);
            // 调用 z_sub 函数计算 x[jcol] 减去 comp_zero，并将结果存储在 x[jcol] 中
            }
        }

                /* 1 z_div costs 10 flops */
        // 计算一个 z_div 操作的浮点运算次数
        solve_ops += 4 * nsupc * (nsupc + 1) + 10 * nsupc;
 
        if ( nsupc == 1 ) {
                    zz_conj(&temp, &Lval[luptr]);
            // 调用 zz_conj 函数求取 Lval[luptr] 的共轭，并存储在 temp 中
            z_div(&x[fsupc], &x[fsupc], &temp);
            // 调用 z_div 函数计算 x[fsupc] 除以 temp，并将结果存储在 x[fsupc] 中
        } else {
#ifdef _CRAY
                    ftcs1 = _cptofcd("U", strlen("U"));
                    // 将字符串"U"转换为 _C_TYPE_CHAR 类型，并计算其长度，赋值给 ftcs1
                    ftcs2 = _cptofcd(trans, strlen("T"));
                    // 将字符串 trans 转换为 _C_TYPE_CHAR 类型，并计算其长度，赋值给 ftcs2
                    ftcs3 = _cptofcd("N", strlen("N"));
                    // 将字符串"N"转换为 _C_TYPE_CHAR 类型，并计算其长度，赋值给 ftcs3
            ZTRSV( ftcs1, ftcs2, ftcs3, &nsupc, &Lval[luptr], &nsupr,
                &x[fsupc], &incx);
            // 调用 ZTRSV 函数进行三角矩阵的求解，参数由上述转换后的字符类型和其他变量构成
#else
                    ztrsv_("U", trans, "N", &nsupc, &Lval[luptr], &nsupr,
                               &x[fsupc], &incx);
                    // 调用 ztrsv_ 函数进行三角矩阵的求解，使用普通的字符串参数表示
#endif
          }
          } /* for k ... */
      }
    }

    stat->ops[SOLVE] += solve_ops;
    // 将 solve_ops 的值加到 stat 结构体的 ops 数组中对应的 SOLVE 索引位置
    SUPERLU_FREE(work);
    // 释放 work 内存
    return 0;
    // 返回 0，表示函数执行成功
}
/*! \brief Performs one of the matrix-vector operations y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y
 *
 * <pre>  
 *   Purpose   
 *   =======   
 *
 *   sp_zgemv()  performs one of the matrix-vector operations   
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
 *               TRANS = 'C' or 'c'   y := alpha*A^H*x + beta*y.   
 *
 *   ALPHA  - (input) doublecomplex
 *            On entry, ALPHA specifies the scalar alpha.   
 *
 *   A      - (input) SuperMatrix*
 *            Before entry, the leading m by n part of the array A must   
 *            contain the matrix of coefficients.   
 *
 *   X      - (input) doublecomplex*, array of DIMENSION at least   
 *            ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'   
 *           and at least   
 *            ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.   
 *            Before entry, the incremented array X must contain the   
 *            vector x.   
 * 
 *   INCX   - (input) int
 *            On entry, INCX specifies the increment for the elements of   
 *            X. INCX must not be zero.   
 *
 *   BETA   - (input) doublecomplex
 *            On entry, BETA specifies the scalar beta. When BETA is   
 *            supplied as zero then Y need not be set on input.   
 *
 *   Y      - (output) doublecomplex*,  array of DIMENSION at least   
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
 *    ==== Sparse Level 2 Blas routine.   
 * </pre>
*/
int
sp_zgemv(char *trans, doublecomplex alpha, SuperMatrix *A, doublecomplex *x, 
     int incx, doublecomplex beta, doublecomplex *y, int incy)
{

    /* Local variables */
    NCformat *Astore;       // 存储 A 稀疏矩阵的非零元素的格式
    doublecomplex   *Aval;  // A 稀疏矩阵的非零元素数组
    int info;               // 存储输入参数测试的结果
    doublecomplex temp, temp1;  // 临时变量
    int lenx, leny, i, j, irow; // 向量和矩阵的长度、循环计数器、行索引
    int iy, jx, jy, kx, ky;     // 向量 Y 和 X 的起始索引
    int notran;                 // 是否执行非转置操作的标志
    doublecomplex comp_zero = {0.0, 0.0};  // 复数零
    doublecomplex comp_one = {1.0, 0.0};   // 复数一

    notran = ( strncmp(trans, "N", 1)==0 || strncmp(trans, "n", 1)==0 );  // 判断是否进行非转置操作
    Astore = A->Store;      // 获取稀疏矩阵 A 的存储结构
    Aval = Astore->nzval;   // 获取稀疏矩阵 A 的非零元素数组
    
    /* Test the input parameters */
    info = 0;   // 初始化参数测试结果为零，表示参数正常
    if ( !notran && strncmp(trans, "T", 1)!=0 && strncmp(trans, "C", 1)!=0)
        info = 1;
    else if ( A->nrow < 0 || A->ncol < 0 ) info = 3;
    else if (incx == 0) info = 5;
    else if (incy == 0)    info = 8;
    if (info != 0) {
    input_error("sp_zgemv ", &info);
    return 0;
    }


    // 检查输入参数，如果不符合预期则设置错误代码并返回
    if ( !notran && strncmp(trans, "T", 1)!=0 && strncmp(trans, "C", 1)!=0)
        info = 1;
    else if ( A->nrow < 0 || A->ncol < 0 ) info = 3;
    else if (incx == 0) info = 5;
    else if (incy == 0)    info = 8;
    // 如果 info 不为 0，则说明有错误，调用输入错误处理函数并返回
    if (info != 0) {
        input_error("sp_zgemv ", &info);
        return 0;
    }



    /* Quick return if possible. */
    if ( A->nrow == 0 || A->ncol == 0 || 
     (z_eq(&alpha, &comp_zero) && z_eq(&beta, &comp_one)) )
    return 0;


    // 如果 A 的行数或列数为 0，或者 alpha 等于零且 beta 等于一，则快速返回
    if ( A->nrow == 0 || A->ncol == 0 || 
     (z_eq(&alpha, &comp_zero) && z_eq(&beta, &comp_one)) )
        return 0;



    /* Set  LENX  and  LENY, the lengths of the vectors x and y, and set 
       up the start points in  X  and  Y. */
    if ( notran ) {
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


    // 设置向量 x 和 y 的长度 LENX 和 LENY，并设置它们的起始点
    if ( notran ) {
        lenx = A->ncol;
        leny = A->nrow;
    } else {
        lenx = A->nrow;
        leny = A->ncol;
    }
    // 根据增量 incx 和 incy 设置 x 和 y 的起始点
    if (incx > 0) kx = 0;
    else kx =  - (lenx - 1) * incx;
    if (incy > 0) ky = 0;
    else ky =  - (leny - 1) * incy;



    /* Start the operations. In this version the elements of A are   
       accessed sequentially with one pass through A. */
    /* First form  y := beta*y. */
    if ( !z_eq(&beta, &comp_one) ) {
    if (incy == 1) {
        if ( z_eq(&beta, &comp_zero) )
        for (i = 0; i < leny; ++i) y[i] = comp_zero;
        else
        for (i = 0; i < leny; ++i) 
          zz_mult(&y[i], &beta, &y[i]);
    } else {
        iy = ky;
        if ( z_eq(&beta, &comp_zero) )
        for (i = 0; i < leny; ++i) {
            y[iy] = comp_zero;
            iy += incy;
        }
        else
        for (i = 0; i < leny; ++i) {
            zz_mult(&y[iy], &beta, &y[iy]);
            iy += incy;
        }
    }
    }


    // 开始矩阵-向量乘法操作，该版本中通过一次遍历顺序访问矩阵 A 的元素
    // 首先计算 y := beta * y
    if ( !z_eq(&beta, &comp_one) ) {
        if (incy == 1) {
            // 当 incy 为 1 时的情况
            if ( z_eq(&beta, &comp_zero) )
                // 如果 beta 等于零，则将 y 的所有元素置为零
                for (i = 0; i < leny; ++i) y[i] = comp_zero;
            else
                // 否则逐元素计算 y[i] := beta * y[i]
                for (i = 0; i < leny; ++i) 
                    zz_mult(&y[i], &beta, &y[i]);
        } else {
            // 当 incy 不为 1 时的情况
            iy = ky;
            if ( z_eq(&beta, &comp_zero) )
                // 如果 beta 等于零，则将 y 的所有元素置为零
                for (i = 0; i < leny; ++i) {
                    y[iy] = comp_zero;
                    iy += incy;
                }
            else
                // 否则逐元素计算 y[iy] := beta * y[iy]
                for (i = 0; i < leny; ++i) {
                    zz_mult(&y[iy], &beta, &y[iy]);
                    iy += incy;
                }
        }
    }



    if ( z_eq(&alpha, &comp_zero) ) return 0;


    // 如果 alpha 等于零，则直接返回
    if ( z_eq(&alpha, &comp_zero) ) return 0;



    if ( notran ) {
    /* Form  y := alpha*A*x + y. */
    jx = kx;
    if (incy == 1) {
        for (j = 0; j < A->ncol; ++j) {
        if ( !z_eq(&x[jx], &comp_zero) ) {
            zz_mult(&temp, &alpha, &x[jx]);
            for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; ++i) {
            irow = Astore->rowind[i];
            zz_mult(&temp1, &temp,  &Aval[i]);
            z_add(&y[irow], &y[irow], &temp1);
            }
        }
        jx += incx;
        }
    } else {
        ABORT("Not implemented.");
    }
    } else if (strncmp(trans, "T", 1) == 0 || strncmp(trans, "t", 1) == 0) {
    /* Form  y := alpha*A'*x + y. */
    jy = ky;
    if (incx == 1) {
        for (j = 0; j < A->ncol; ++j) {
        temp = comp_zero;
        for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; ++i) {
            irow = Astore->rowind[i];
            zz_mult(&temp1, &Aval[i], &x[irow]);
            z_add(&temp, &temp, &temp1);
        }
        zz_mult(&temp1, &alpha, &temp);
        z_add(&y[jy], &y[jy], &temp1);
        jy += incy;
        }
    } else {
        ABORT("Not implemented.");
    }
    } else { /* trans == 'C' or 'c' */
    /* Form  y := alpha * conj(A) * x + y. */
    doublecomplex temp2;
    jy = ky;


    // 根据不同的转置选项进行矩阵-向量乘法操作
    if ( notran ) {
        /* Form  y := alpha*A*x + y. */
        jx = kx;
        if (incy == 1) {
            // 当 incy 为 1 时的情况
            for (j = 0; j < A->ncol; ++j) {
                if ( !z_eq(&x[jx], &comp_zero) ) {
                    // 如果 x[jx] 不为零，则计算 alpha * x[jx]
                    zz_mult(&temp, &alpha, &x[jx]);
                    // 遍历 A 的第 j 列中的非零元素
                    for (i
    # 检查增量是否为1，表示列主循环时的处理方式
    if (incx == 1) {
        # 遍历矩阵A的每一列
        for (j = 0; j < A->ncol; ++j) {
            # 初始化临时变量temp为复数零
            temp = comp_zero;
            # 遍历矩阵A中第j列非零元素
            for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; ++i) {
                # 获取矩阵A中非零元素的行索引
                irow = Astore->rowind[i];
                # 将当前非零元素的实部赋值给temp2的实部
                temp2.r = Aval[i].r;
                # 将当前非零元素的虚部的相反数赋值给temp2的虚部，表示复数的共轭
                temp2.i = -Aval[i].i;  /* conjugation */
                # 计算temp1 = temp2 * x[irow]，其中temp2为复数，x[irow]为向量元素
                zz_mult(&temp1, &temp2, &x[irow]);
                # 计算temp = temp + temp1，将每列的贡献累加到temp中
                z_add(&temp, &temp, &temp1);
            }
            # 计算temp1 = alpha * temp，其中alpha为复数常数
            zz_mult(&temp1, &alpha, &temp);
            # 将结果temp1加到向量y的当前位置y[jy]上
            z_add(&y[jy], &y[jy], &temp1);
            # 更新向量y的索引位置jy，按照增量incy移动
            jy += incy;
        }
    } else {
        # 如果增量incx不为1，抛出异常，提示该情况未实现
        ABORT("Not implemented.");
    }
    # 返回状态码0，表示成功执行
    return 0;
} /* sp_zgemv */
```