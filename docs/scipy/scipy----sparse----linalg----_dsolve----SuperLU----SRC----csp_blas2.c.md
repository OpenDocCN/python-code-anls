# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\csp_blas2.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file csp_blas2.c
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
 * File name:        csp_blas2.c
 * Purpose:        Sparse BLAS 2, using some dense BLAS 2 operations.
 */

#include "slu_cdefs.h"


注释：
/*! \brief Solves one of the systems of equations A*x = b,   or   A'*x = b
 * 
 * <pre>
 *   Purpose
 *   =======
 *
 *   sp_ctrsv() solves one of the systems of equations   
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
 *             i.e., L has types: Stype = SC, Dtype = SLU_C, Mtype = TRLU.
 *
 *   U       - (input) SuperMatrix*
 *            The factor U from the factorization Pr*A*Pc=L*U.
 *            U has types: Stype = NC, Dtype = SLU_C, Mtype = TRU.
 *    
 *   x       - (input/output) singlecomplex*
 *             Before entry, the incremented array X must contain the n   
 *             element right-hand side vector b. On exit, X is overwritten 
 *             with the solution vector x.
 *
 *   info    - (output) int*
 *             If *info = -i, the i-th argument had an illegal value.
 * </pre>
 */
int
sp_ctrsv(char *uplo, char *trans, char *diag, SuperMatrix *L, 
         SuperMatrix *U, singlecomplex *x, SuperLUStat_t *stat, int *info)
{
#ifdef _CRAY
    _fcd ftcs1 = _cptofcd("L", strlen("L")),
     ftcs2 = _cptofcd("N", strlen("N")),
     ftcs3 = _cptofcd("U", strlen("U"));
#endif
    // 定义存储稀疏矩阵的格式
    SCformat *Lstore;
    NCformat *Ustore;
    // 定义存储稀疏矩阵的值
    singlecomplex   *Lval, *Uval;
    // 定义向量操作的增量
    int incx = 1, incy = 1;
    // 定义临时变量和复数类型的常量
    singlecomplex temp;
    singlecomplex alpha = {1.0, 0.0}, beta = {1.0, 0.0};
    singlecomplex comp_zero = {0.0, 0.0};
    // 定义整型变量
    int nrow, irow, jcol;
    int fsupc, nsupr, nsupc;
    int_t luptr, istart, i, k, iptr;
    // 定义工作区数组
    singlecomplex *work;
    // 计算浮点操作的数量
    flops_t solve_ops;

    /* Test the input parameters */
    // 测试输入参数
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
        input_error("sp_ctrsv", &ii);
        return 0;
    }


    // 检查上三角（uplo）参数是否有效，如果无效设置错误代码为-1
    // 检查转置（trans）参数是否有效，如果无效设置错误代码为-2
    // 检查对角线（diag）参数是否有效，如果无效设置错误代码为-3
    // 检查矩阵 L 的维度是否合法，如果不合法设置错误代码为-4
    // 检查矩阵 U 的维度是否合法，如果不合法设置错误代码为-5
    if ( *info ) {
        // 如果存在错误代码，则将其转换为正数 ii，并输出输入错误信息
        int ii = -(*info);
        input_error("sp_ctrsv", &ii);
        return 0;
    }



    Lstore = L->Store;
    Lval = Lstore->nzval;
    Ustore = U->Store;
    Uval = Ustore->nzval;
    solve_ops = 0;


    // 将矩阵 L 和 U 的存储结构和非零值数组分配给相应的变量
    Lstore = L->Store;
    Lval = Lstore->nzval;
    Ustore = U->Store;
    Uval = Ustore->nzval;
    // 初始化解算操作数为零
    solve_ops = 0;



    if ( !(work = complexCalloc(L->nrow)) )
    ABORT("Malloc fails for work in sp_ctrsv().");


    // 分配复数类型的工作数组 work，并检查分配是否成功
    if ( !(work = complexCalloc(L->nrow)) )
        ABORT("Malloc fails for work in sp_ctrsv().");



    if ( strncmp(trans, "N", 1)==0 ) {    /* Form x := inv(A)*x. */


    // 如果转置参数为 "N"，表示求解 x := inv(A)*x



    if ( strncmp(uplo, "L", 1)==0 ) {
        /* Form x := inv(L)*x */
            if ( L->nrow == 0 ) return 0; /* Quick return */
        
        for (k = 0; k <= Lstore->nsuper; k++) {
        fsupc = L_FST_SUPC(k);
        istart = L_SUB_START(fsupc);
        nsupr = L_SUB_START(fsupc+1) - istart;
        nsupc = L_FST_SUPC(k+1) - fsupc;
        luptr = L_NZ_START(fsupc);
        nrow = nsupr - nsupc;


        // 如果上三角参数为 "L"，表示求解 x := inv(L)*x
        // 如果矩阵 L 的行数为 0，则快速返回
        for (k = 0; k <= Lstore->nsuper; k++) {
            // 获取当前超节点的相关信息
            fsupc = L_FST_SUPC(k);
            istart = L_SUB_START(fsupc);
            nsupr = L_SUB_START(fsupc+1) - istart;
            nsupc = L_FST_SUPC(k+1) - fsupc;
            luptr = L_NZ_START(fsupc);
            nrow = nsupr - nsupc;



                /* 1 c_div costs 10 flops */
            solve_ops += 4 * nsupc * (nsupc - 1) + 10 * nsupc;
            solve_ops += 8 * nrow * nsupc;


                // 计算解算操作数，每个 c_div 操作花费 10 次浮点运算
                solve_ops += 4 * nsupc * (nsupc - 1) + 10 * nsupc;
                solve_ops += 8 * nrow * nsupc;



        if ( nsupc == 1 ) {
            for (iptr=istart+1; iptr < L_SUB_START(fsupc+1); ++iptr) {
            irow = L_SUB(iptr);
            ++luptr;
            cc_mult(&comp_zero, &x[fsupc], &Lval[luptr]);
            c_sub(&x[irow], &x[irow], &comp_zero);
            }
        } else {


        // 处理单个超节点的情况
        if ( nsupc == 1 ) {
            for (iptr=istart+1; iptr < L_SUB_START(fsupc+1); ++iptr) {
                // 获取当前行号
                irow = L_SUB(iptr);
                ++luptr;
                // 计算复数乘法和减法操作
                cc_mult(&comp_zero, &x[fsupc], &Lval[luptr]);
                c_sub(&x[irow], &x[irow], &comp_zero);
            }
        } else {
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
            // 使用 CRAY 特定函数 CTRSV 解决方程 L * x = b
            CTRSV(ftcs1, ftcs2, ftcs3, &nsupc, &Lval[luptr], &nsupr,
                   &x[fsupc], &incx);
        
            // 使用 CRAY 特定函数 CGEMV 计算乘法和加法：y := alpha * A * x + beta * y
            CGEMV(ftcs2, &nrow, &nsupc, &alpha, &Lval[luptr+nsupc], 
                   &nsupr, &x[fsupc], &incx, &beta, &work[0], &incy);
#else
            // 使用标准 BLAS 函数 ctrsv 解决方程 L * x = b
            ctrsv_("L", "N", "U", &nsupc, &Lval[luptr], &nsupr,
                   &x[fsupc], &incx);
        
            // 使用标准 BLAS 函数 cgemv 计算乘法和加法：y := alpha * A * x + beta * y
            cgemv_("N", &nrow, &nsupc, &alpha, &Lval[luptr+nsupc], 
                   &nsupr, &x[fsupc], &incx, &beta, &work[0], &incy);
#endif
#else
            // 使用自定义函数 clsolve 解决方程 L * x = b
            clsolve ( nsupr, nsupc, &Lval[luptr], &x[fsupc]);
        
            // 使用自定义函数 cmatvec 计算矩阵向量乘积：y := A * x
            cmatvec ( nsupr, nsupr-nsupc, nsupc, &Lval[luptr+nsupc],
                             &x[fsupc], &work[0] );
#endif        
        
            // 更新 x 向量中的部分元素
            iptr = istart + nsupc;
            for (i = 0; i < nrow; ++i, ++iptr) {
                irow = L_SUB(iptr);
                c_sub(&x[irow], &x[irow], &work[i]); /* Scatter */
                work[i] = comp_zero;
            }
         }
        } /* for k ... */
        
    } else {
        /* Form x := inv(U)*x */
        
        // 如果 U 矩阵为空，则快速返回
        if ( U->nrow == 0 ) return 0; /* Quick return */
        
        // 反向遍历每个超节点 k
        for (k = Lstore->nsuper; k >= 0; k--) {
            fsupc = L_FST_SUPC(k);
            nsupr = L_SUB_START(fsupc+1) - L_SUB_START(fsupc);
            nsupc = L_FST_SUPC(k+1) - fsupc;
            luptr = L_NZ_START(fsupc);
        
            // 计算操作数：4 * nsupc * (nsupc + 1) + 10 * nsupc
            solve_ops += 4 * nsupc * (nsupc + 1) + 10 * nsupc;

            // 如果超节点只有一个元素
            if ( nsupc == 1 ) {
                // 一次除法操作
                c_div(&x[fsupc], &x[fsupc], &Lval[luptr]);
                // 更新相关行的元素
                for (i = U_NZ_START(fsupc); i < U_NZ_START(fsupc+1); ++i) {
                    irow = U_SUB(i);
                    cc_mult(&comp_zero, &x[fsupc], &Uval[i]);
                    c_sub(&x[irow], &x[irow], &comp_zero);
                }
            } else {
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
                // 使用 CRAY 特定函数 CTRSV 解决方程 U * x = b
                CTRSV(ftcs3, ftcs2, ftcs2, &nsupc, &Lval[luptr], &nsupr,
                   &x[fsupc], &incx);
#else
                // 使用标准 BLAS 函数 ctrsv 解决方程 U * x = b
                ctrsv_("U", "N", "N", &nsupc, &Lval[luptr], &nsupr,
                           &x[fsupc], &incx);
#endif
#else        
                // 使用自定义函数 cusolve 解决方程 U * x = b
                cusolve ( nsupr, nsupc, &Lval[luptr], &x[fsupc] );
#endif        

                // 更新相关列的元素
                for (jcol = fsupc; jcol < L_FST_SUPC(k+1); jcol++) {
                    solve_ops += 8*(U_NZ_START(jcol+1) - U_NZ_START(jcol));
                    for (i = U_NZ_START(jcol); i < U_NZ_START(jcol+1); 
                    i++) {
                        irow = U_SUB(i);
                        cc_mult(&comp_zero, &x[jcol], &Uval[i]);
                        c_sub(&x[irow], &x[irow], &comp_zero);
                    }
                }
            }
        } /* for k ... */
        
    }
    } else if ( strncmp(trans, "T", 1)==0 ) { /* Form x := inv(A')*x */
        // 如果需要计算 A 转置的逆乘以 x，则直接返回
    # 如果 uplo 的前一个字符是 "L"
    if ( strncmp(uplo, "L", 1)==0 ) {
        # 执行 x := inv(L')*x 的计算
        /* Form x := inv(L')*x */

            # 如果 L 的行数为 0，立即返回 0
            if ( L->nrow == 0 ) return 0; /* Quick return */
        
        # 从最后一个超节点开始向前遍历
        for (k = Lstore->nsuper; k >= 0; --k) {
            # 获取当前超节点的第一个列索引
            fsupc = L_FST_SUPC(k);
            # 获取当前超节点的行索引起始位置
            istart = L_SUB_START(fsupc);
            # 当前超节点包含的行数
            nsupr = L_SUB_START(fsupc+1) - istart;
            # 当前超节点包含的列数
            nsupc = L_FST_SUPC(k+1) - fsupc;
            # 当前超节点的非零元素起始位置
            luptr = L_NZ_START(fsupc);

        # 计算解算操作的次数
        solve_ops += 8 * (nsupr - nsupc) * nsupc;

        # 遍历当前超节点的每一列
        for (jcol = fsupc; jcol < L_FST_SUPC(k+1); jcol++) {
            # 初始行指针
            iptr = istart + nsupc;
            # 遍历当前列中的每一个非零元素
            for (i = L_NZ_START(jcol) + nsupc; 
                i < L_NZ_START(jcol+1); i++) {
            # 获取当前非零元素的行索引
            irow = L_SUB(iptr);
            # 复数乘法运算，计算 x[irow] 乘以 Lval[i] 的结果并加到 comp_zero
            cc_mult(&comp_zero, &x[irow], &Lval[i]);
                # 复数减法运算，计算 x[jcol] 减去 comp_zero 的结果
                c_sub(&x[jcol], &x[jcol], &comp_zero);
            # 行指针向后移动
            iptr++;
            }
        }
        
        # 如果当前超节点的列数大于 1
        if ( nsupc > 1 ) {
            # 更新解算操作的次数
            solve_ops += 4 * nsupc * (nsupc - 1);
        # 如果定义了_CRAY宏，则使用_CPTOFCD将字符串转换为对应的长度码
        # 并调用CTRVS函数进行解方程
#ifdef _CRAY
                    ftcs1 = _cptofcd("L", strlen("L"));
                    ftcs2 = _cptofcd("T", strlen("T"));
                    ftcs3 = _cptofcd("U", strlen("U"));
            CTRSV(ftcs1, ftcs2, ftcs3, &nsupc, &Lval[luptr], &nsupr,
            &x[fsupc], &incx);
#else
            # 否则直接调用CTRVS函数进行解方程
            ctrsv_("L", "T", "U", &nsupc, &Lval[luptr], &nsupr,
            &x[fsupc], &incx);
#endif
        }
        }
    } else {
        /* Form x := inv(U')*x */
        # 如果U的行数为0，则直接返回0，快速退出
        if ( U->nrow == 0 ) return 0; /* Quick return */
        
        # 对每个超节点k执行以下操作
        for (k = 0; k <= Lstore->nsuper; k++) {
            fsupc = L_FST_SUPC(k);  # 第k个超节点的第一个列
            nsupr = L_SUB_START(fsupc+1) - L_SUB_START(fsupc);  # 超节点k的行数
            nsupc = L_FST_SUPC(k+1) - fsupc;  # 超节点k的列数
            luptr = L_NZ_START(fsupc);  # 超节点k在Lval中的起始位置索引

        # 对超节点k中的每列jcol执行以下操作
        for (jcol = fsupc; jcol < L_FST_SUPC(k+1); jcol++) {
            solve_ops += 8*(U_NZ_START(jcol+1) - U_NZ_START(jcol));  # 更新操作数
            # 对于每个非零元素i在列jcol中执行以下操作
            for (i = U_NZ_START(jcol); i < U_NZ_START(jcol+1); i++) {
            irow = U_SUB(i);  # i所在的行号
            cc_mult(&comp_zero, &x[irow], &Uval[i]);  # 计算复数乘法
                c_sub(&x[jcol], &x[jcol], &comp_zero);  # 计算复数减法
            }
        }

                /* 1 c_div costs 10 flops */
        solve_ops += 4 * nsupc * (nsupc + 1) + 10 * nsupc;  # 更新操作数

        # 如果超节点k的列数为1，则对x[fsupc]进行复数除法
        if ( nsupc == 1 ) {
            c_div(&x[fsupc], &x[fsupc], &Lval[luptr]);
        } else {
#ifdef _CRAY
                    ftcs1 = _cptofcd("U", strlen("U"));
                    ftcs2 = _cptofcd("T", strlen("T"));
                    ftcs3 = _cptofcd("N", strlen("N"));
            CTRSV( ftcs1, ftcs2, ftcs3, &nsupc, &Lval[luptr], &nsupr,
                &x[fsupc], &incx);
#else
            # 否则直接调用CTRVS函数进行解方程
            ctrsv_("U", "T", "N", &nsupc, &Lval[luptr], &nsupr,
                &x[fsupc], &incx);
#endif
        }
        } /* for k ... */
    }
    } else { /* Form x := conj(inv(A'))*x */
    
    # 如果uplo的第一个字符是"L"
    if ( strncmp(uplo, "L", 1)==0 ) {
        /* Form x := conj(inv(L'))*x */
            # 如果L的行数为0，则直接返回0，快速退出
            if ( L->nrow == 0 ) return 0; /* Quick return */
        
        # 对每个超节点k执行以下操作（倒序）
        for (k = Lstore->nsuper; k >= 0; --k) {
            fsupc = L_FST_SUPC(k);  # 第k个超节点的第一个列
            istart = L_SUB_START(fsupc);  # 超节点k的起始位置索引
            nsupr = L_SUB_START(fsupc+1) - istart;  # 超节点k的行数
            nsupc = L_FST_SUPC(k+1) - fsupc;  # 超节点k的列数
            luptr = L_NZ_START(fsupc);  # 超节点k在Lval中的起始位置索引

        solve_ops += 8 * (nsupr - nsupc) * nsupc;  # 更新操作数

        # 对超节点k中的每列jcol执行以下操作
        for (jcol = fsupc; jcol < L_FST_SUPC(k+1); jcol++) {
            iptr = istart + nsupc;
            for (i = L_NZ_START(jcol) + nsupc; 
                i < L_NZ_START(jcol+1); i++) {
            irow = L_SUB(iptr);  # i所在的行号
                        cc_conj(&temp, &Lval[i]);  # 计算复共轭
            cc_mult(&comp_zero, &x[irow], &temp);  # 计算复数乘法
                c_sub(&x[jcol], &x[jcol], &comp_zero);  # 计算复数减法
            iptr++;
            }
         }
         
         # 如果超节点k的列数大于1，则更新操作数
         if ( nsupc > 1 ) {
            solve_ops += 4 * nsupc * (nsupc - 1);
#ifdef _CRAY
                    // 将字符串 "L" 转换为 FORTRAN 字符串
                    ftcs1 = _cptofcd("L", strlen("L"));
                    // 将字符串 trans 转换为 FORTRAN 字符串
                    ftcs2 = _cptofcd(trans, strlen("T"));
                    // 将字符串 "U" 转换为 FORTRAN 字符串
                    ftcs3 = _cptofcd("U", strlen("U"));
            // 调用 FORTRAN 函数 CTRSV 处理三角矩阵求解
            CTRSV(ftcs1, ftcs2, ftcs3, &nsupc, &Lval[luptr], &nsupr,
            &x[fsupc], &incx);
#else
                    // 调用 BLAS 函数 ctrsv 处理三角矩阵求解
                    ctrsv_("L", trans, "U", &nsupc, &Lval[luptr], &nsupr,
                           &x[fsupc], &incx);
#endif
        }
        }
    } else {
        /* Form x := conj(inv(U'))*x */
        // 如果 U 的行数为 0，快速返回
        if ( U->nrow == 0 ) return 0; /* Quick return */
        
        // 对每个超节点 k 进行循环
        for (k = 0; k <= Lstore->nsuper; k++) {
            // 获取当前超节点的第一个列号
            fsupc = L_FST_SUPC(k);
            // 获取当前超节点包含的行数
            nsupr = L_SUB_START(fsupc+1) - L_SUB_START(fsupc);
            // 获取当前超节点包含的列数
            nsupc = L_FST_SUPC(k+1) - fsupc;
            // 获取当前超节点在 Lval 中的起始位置
            luptr = L_NZ_START(fsupc);

            // 对当前超节点的每一列进行循环
            for (jcol = fsupc; jcol < L_FST_SUPC(k+1); jcol++) {
                // 计算解操作的估计数
                solve_ops += 8 * (U_NZ_START(jcol+1) - U_NZ_START(jcol));
                // 对当前列的每一个非零元素进行循环
                for (i = U_NZ_START(jcol); i < U_NZ_START(jcol+1); i++) {
                    // 获取当前非零元素所在的行号
                    irow = U_SUB(i);
                    // 计算复数的共轭
                    cc_conj(&temp, &Uval[i]);
                    // 计算复数的乘积
                    cc_mult(&comp_zero, &x[irow], &temp);
                    // 复数相减
                    c_sub(&x[jcol], &x[jcol], &comp_zero);
                }
            }

            // 估算每次 c_div 操作的代价为 10 次浮点运算
            solve_ops += 4 * nsupc * (nsupc + 1) + 10 * nsupc;

            // 如果当前超节点只有一个列，则进行特殊处理
            if ( nsupc == 1 ) {
                // 计算复数的共轭
                cc_conj(&temp, &Lval[luptr]);
                // 复数除法操作
                c_div(&x[fsupc], &x[fsupc], &temp);
            } else {
#ifdef _CRAY
                    // 将字符串 "U" 转换为 FORTRAN 字符串
                    ftcs1 = _cptofcd("U", strlen("U"));
                    // 将字符串 trans 转换为 FORTRAN 字符串
                    ftcs2 = _cptofcd(trans, strlen("T"));
                    // 将字符串 "N" 转换为 FORTRAN 字符串
                    ftcs3 = _cptofcd("N", strlen("N"));
            // 调用 FORTRAN 函数 CTRSV 处理三角矩阵求解
            CTRSV( ftcs1, ftcs2, ftcs3, &nsupc, &Lval[luptr], &nsupr,
                &x[fsupc], &incx);
#else
                    // 调用 BLAS 函数 ctrsv 处理三角矩阵求解
                    ctrsv_("U", trans, "N", &nsupc, &Lval[luptr], &nsupr,
                               &x[fsupc], &incx);
#endif
          }
          } /* for k ... */
      }
    }

    // 统计解操作的总数，存储在 stat->ops[SOLVE] 中
    stat->ops[SOLVE] += solve_ops;
    // 释放工作内存
    SUPERLU_FREE(work);
    // 返回 0 表示成功
    return 0;
}


这段代码主要是对线性方程组求解过程的描述，使用了条件编译来适应不同的编译环境，通过调用不同的函数（如 CTRSV 和 ctrsv_）实现三角矩阵求解操作。
/*! \brief Performs one of the matrix-vector operations y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y
 *
 * <pre>  
 *   Purpose   
 *   =======   
 *
 *   sp_cgemv()  performs one of the matrix-vector operations   
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
 *   ALPHA  - (input) singlecomplex
 *            On entry, ALPHA specifies the scalar alpha.   
 *
 *   A      - (input) SuperMatrix*
 *            Before entry, the leading m by n part of the array A must   
 *            contain the matrix of coefficients.   
 *
 *   X      - (input) singlecomplex*, array of DIMENSION at least   
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
 *   BETA   - (input) singlecomplex
 *            On entry, BETA specifies the scalar beta. When BETA is   
 *            supplied as zero then Y need not be set on input.   
 *
 *   Y      - (output) singlecomplex*,  array of DIMENSION at least   
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
sp_cgemv(char *trans, singlecomplex alpha, SuperMatrix *A, singlecomplex *x, 
     int incx, singlecomplex beta, singlecomplex *y, int incy)
{
    /* Local variables */
    NCformat *Astore;  // Pointer to the non-column-compressed format of matrix A
    singlecomplex   *Aval;  // Pointer to the array of non-zero elements of A
    int info;  // Variable to store error information
    singlecomplex temp, temp1;  // Temporary variables for computations
    int lenx, leny, i, j, irow;  // Various integer variables for loop indices and dimensions
    int iy, jx, jy, kx, ky;  // Integer variables used for indexing in loops
    int notran;  // Flag indicating if the operation is non-transpose (N)
    singlecomplex comp_zero = {0.0, 0.0};  // Complex number with real and imaginary parts both zero
    singlecomplex comp_one = {1.0, 0.0};  // Complex number with real part one and imaginary part zero

    notran = ( strncmp(trans, "N", 1)==0 || strncmp(trans, "n", 1)==0 );  // Determine if TRANS indicates non-transpose operation
    Astore = A->Store;  // Extract the structure holding the matrix A in non-column-compressed format
    Aval = Astore->nzval;  // Get the array of non-zero elements of A
    
    /* Test the input parameters */
    info = 0;  // Initialize info to zero, indicating no error initially
    /* Check for invalid transaction type */
    if (!notran && strncmp(trans, "T", 1) != 0 && strncmp(trans, "C", 1) != 0)
        info = 1;
    else if (A->nrow < 0 || A->ncol < 0)
        info = 3;
    else if (incx == 0)
        info = 5;
    else if (incy == 0)
        info = 8;
    if (info != 0) {
        input_error("sp_cgemv ", &info);
        return 0;
    }

    /* Quick return if dimensions are zero or alpha is zero and beta is one */
    if (A->nrow == 0 || A->ncol == 0 ||
        (c_eq(&alpha, &comp_zero) && c_eq(&beta, &comp_one)))
        return 0;

    /* Set LENX and LENY, the lengths of vectors x and y, and determine
       their starting points in X and Y */
    if (notran) {
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

    /* Start the operations accessing elements of A sequentially */
    /* First, compute y := beta * y */
    if (!c_eq(&beta, &comp_one)) {
        if (incy == 1) {
            if (c_eq(&beta, &comp_zero))
                for (i = 0; i < leny; ++i)
                    y[i] = comp_zero;
            else
                for (i = 0; i < leny; ++i)
                    cc_mult(&y[i], &beta, &y[i]);
        } else {
            iy = ky;
            if (c_eq(&beta, &comp_zero))
                for (i = 0; i < leny; ++i) {
                    y[iy] = comp_zero;
                    iy += incy;
                }
            else
                for (i = 0; i < leny; ++i) {
                    cc_mult(&y[iy], &beta, &y[iy]);
                    iy += incy;
                }
        }
    }

    if (c_eq(&alpha, &comp_zero))
        return 0;

    if (notran) {
        /* Compute y := alpha * A * x + y */
        jx = kx;
        if (incy == 1) {
            for (j = 0; j < A->ncol; ++j) {
                if (!c_eq(&x[jx], &comp_zero)) {
                    cc_mult(&temp, &alpha, &x[jx]);
                    for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; ++i) {
                        irow = Astore->rowind[i];
                        cc_mult(&temp1, &temp, &Aval[i]);
                        c_add(&y[irow], &y[irow], &temp1);
                    }
                }
                jx += incx;
            }
        } else {
            ABORT("Not implemented.");
        }
    } else if (strncmp(trans, "T", 1) == 0 || strncmp(trans, "t", 1) == 0) {
        /* Compute y := alpha * A' * x + y */
        jy = ky;
        if (incx == 1) {
            for (j = 0; j < A->ncol; ++j) {
                temp = comp_zero;
                for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; ++i) {
                    irow = Astore->rowind[i];
                    cc_mult(&temp1, &Aval[i], &x[irow]);
                    c_add(&temp, &temp, &temp1);
                }
                cc_mult(&temp1, &alpha, &temp);
                c_add(&y[jy], &y[jy], &temp1);
                jy += incy;
            }
        } else {
            ABORT("Not implemented.");
        }
    } else { /* trans == 'C' or 'c' */
        /* Compute y := alpha * conj(A) * x + y */
        singlecomplex temp2;
        jy = ky;
        // Further code would continue here, but it's not provided.
    }
    # 如果增量步长为1，执行以下代码块
    if (incx == 1) {
        # 遍历矩阵 A 的每一列
        for (j = 0; j < A->ncol; ++j) {
            # 初始化临时变量 temp 为零复数
            temp = comp_zero;
            # 遍历 A 的第 j 列中的每一个非零元素
            for (i = Astore->colptr[j]; i < Astore->colptr[j+1]; ++i) {
                # 获取该非零元素的行索引
                irow = Astore->rowind[i];
                # 将 Aval[i] 的实部赋值给 temp2.r，虚部取反（共轭）
                temp2.r = Aval[i].r;
                temp2.i = -Aval[i].i;  /* conjugation */
                # 计算 temp1 = temp2 * x[irow]
                cc_mult(&temp1, &temp2, &x[irow]);
                # 计算 temp = temp + temp1
                c_add(&temp, &temp, &temp1);
            }
            # 计算 temp1 = alpha * temp
            cc_mult(&temp1, &alpha, &temp);
            # 计算 y[jy] = y[jy] + temp1
            c_add(&y[jy], &y[jy], &temp1);
            # 更新 jy 的索引位置
            jy += incy;
        }
    } else {
        # 如果增量步长不为1，则输出错误信息并终止程序
        ABORT("Not implemented.");
    }
    # 函数执行完毕，返回 0 表示成功
    return 0;    
} /* sp_cgemv */
```