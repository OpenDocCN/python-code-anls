# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dgstrs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file dgstrs.c
 * \brief Solves a system using LU factorization
 *
 * <pre>
 * -- SuperLU routine (version 3.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * October 15, 2003
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

#include "slu_ddefs.h"



注释：


/*! \file
版权声明和许可证信息，此文件描述了代码的版权和许可情况
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file dgstrs.c
 * \brief Solves a system using LU factorization
 *
 * <pre>
 * -- SuperLU routine (version 3.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * October 15, 2003
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

#include "slu_ddefs.h"
/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * DGSTRS solves a system of linear equations A*X=B or A'*X=B
 * with A sparse and B dense, using the LU factorization computed by
 * DGSTRF.
 *
 * See supermatrix.h for the definition of 'SuperMatrix' structure.
 *
 * Arguments
 * =========
 *
 * trans   (input) trans_t
 *          Specifies the form of the system of equations:
 *          = NOTRANS: A * X = B  (No transpose)
 *          = TRANS:   A'* X = B  (Transpose)
 *          = CONJ:    A**H * X = B  (Conjugate transpose)
 *
 * L       (input) SuperMatrix*
 *         The factor L from the factorization Pr*A*Pc=L*U as computed by
 *         dgstrf(). Use compressed row subscripts storage for supernodes,
 *         i.e., L has types: Stype = SLU_SC, Dtype = SLU_D, Mtype = SLU_TRLU.
 *
 * U       (input) SuperMatrix*
 *         The factor U from the factorization Pr*A*Pc=L*U as computed by
 *         dgstrf(). Use column-wise storage scheme, i.e., U has types:
 *         Stype = SLU_NC, Dtype = SLU_D, Mtype = SLU_TRU.
 *
 * perm_c  (input) int*, dimension (L->ncol)
 *       Column permutation vector, which defines the 
 *         permutation matrix Pc; perm_c[i] = j means column i of A is 
 *         in position j in A*Pc.
 *
 * perm_r  (input) int*, dimension (L->nrow)
 *         Row permutation vector, which defines the permutation matrix Pr; 
 *         perm_r[i] = j means row i of A is in position j in Pr*A.
 *
 * B       (input/output) SuperMatrix*
 *         B has types: Stype = SLU_DN, Dtype = SLU_D, Mtype = SLU_GE.
 *         On entry, the right hand side matrix.
 *         On exit, the solution matrix if info = 0;
 *
 * stat     (output) SuperLUStat_t*
 *          Record the statistics on runtime and floating-point operation count.
 *          See util.h for the definition of 'SuperLUStat_t'.
 *
 * info    (output) int*
 *        = 0: successful exit
 *       < 0: if info = -i, the i-th argument had an illegal value
 * </pre>
 */

void
dgstrs (trans_t trans, SuperMatrix *L, SuperMatrix *U,
        int *perm_c, int *perm_r, SuperMatrix *B,
        SuperLUStat_t *stat, int *info)
{

#ifdef _CRAY
    _fcd ftcs1, ftcs2, ftcs3, ftcs4;
#endif
#ifdef USE_VENDOR_BLAS
    double   alpha = 1.0, beta = 1.0;
    double   *work_col;
#endif
    DNformat *Bstore;
    double   *Bmat;
    SCformat *Lstore;
    NCformat *Ustore;
    double   *Lval, *Uval;
    int      fsupc, nrow, nsupr, nsupc, irow;
    int_t    i, j, k, luptr, istart, iptr;
    int      jcol, n, ldb, nrhs;
    double   *work, *rhs_work, *soln;
    flops_t  solve_ops;
    void dprint_soln(int n, int nrhs, double *soln);

    /* Test input parameters ... */

    // 初始化 info 为 0，表示初始状态为成功
    *info = 0;

    // 获取 B 矩阵的存储格式
    Bstore = B->Store;
    // 获取 B 矩阵的 leading dimension
    ldb = Bstore->lda;
    // 获取 B 矩阵的列数，即右手边矩阵的列数
    nrhs = B->ncol;

    // 检查 trans 参数是否合法
    if ( trans != NOTRANS && trans != TRANS && trans != CONJ )
        *info = -1;
    // 检查 L 矩阵的行数与列数是否相等且大于等于 0，以及存储类型是否正确
    else if ( L->nrow != L->ncol || L->nrow < 0 ||
              L->Stype != SLU_SC || L->Dtype != SLU_D || L->Mtype != SLU_TRLU )
        *info = -2;
    else if ( U->nrow != U->ncol || U->nrow < 0 ||
          U->Stype != SLU_NC || U->Dtype != SLU_D || U->Mtype != SLU_TRU )
    *info = -3;
    else if ( ldb < SUPERLU_MAX(0, L->nrow) ||
          B->Stype != SLU_DN || B->Dtype != SLU_D || B->Mtype != SLU_GE )
    *info = -6;
    if ( *info ) {
    // 如果 *info 不为零，表示前面出现了错误，调用 input_error 报告错误并返回
    int ii = -(*info);
    input_error("dgstrs", &ii);
    return;
    }

    n = L->nrow;
    // 分配内存空间以存储工作数组，用于解决方程
    work = doubleCalloc((size_t) n * (size_t) nrhs);
    if ( !work ) ABORT("Malloc fails for local work[].");
    // 分配内存空间以存储解向量
    soln = doubleMalloc((size_t) n);
    if ( !soln ) ABORT("Malloc fails for local soln[].");

    Bmat = Bstore->nzval;
    Lstore = L->Store;
    Lval = Lstore->nzval;
    Ustore = U->Store;
    Uval = Ustore->nzval;
    solve_ops = 0;
    
    if ( trans == NOTRANS ) {
    /* Permute right hand sides to form Pr*B */
    // 对右侧向量进行置换，形成 Pr*B
    for (i = 0; i < nrhs; i++) {
        rhs_work = &Bmat[(size_t)i * (size_t)ldb];
        for (k = 0; k < n; k++) soln[perm_r[k]] = rhs_work[k];
        for (k = 0; k < n; k++) rhs_work[k] = soln[k];
    }
    
    /* Forward solve PLy=Pb. */
    // 前向解 PLy = Pb
    for (k = 0; k <= Lstore->nsuper; k++) {
        fsupc = L_FST_SUPC(k);
        istart = L_SUB_START(fsupc);
        nsupr = L_SUB_START(fsupc+1) - istart;
        nsupc = L_FST_SUPC(k+1) - fsupc;
        nrow = nsupr - nsupc;

        // 计算解操作数
        solve_ops += nsupc * (nsupc - 1) * nrhs;
        solve_ops += 2 * nrow * nsupc * nrhs;
        
        if ( nsupc == 1 ) {
        // 如果当前超节点只有一个列，则直接进行计算
        for (j = 0; j < nrhs; j++) {
            rhs_work = &Bmat[(size_t)j * (size_t)ldb];
                luptr = L_NZ_START(fsupc);
            for (iptr=istart+1; iptr < L_SUB_START(fsupc+1); iptr++){
            irow = L_SUB(iptr);
            ++luptr;
            rhs_work[irow] -= rhs_work[fsupc] * Lval[luptr];
            }
        }
        } else {
            luptr = L_NZ_START(fsupc);
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
        // 将字符串转换为 Fortran 字符串描述符
        ftcs1 = _cptofcd("L", strlen("L"));
        ftcs2 = _cptofcd("N", strlen("N"));
        ftcs3 = _cptofcd("U", strlen("U"));
        // 调用 BLAS 库中的 STRSM 函数进行矩阵运算
        STRSM( ftcs1, ftcs1, ftcs2, ftcs3, &nsupc, &nrhs, &alpha,
               &Lval[luptr], &nsupr, &Bmat[fsupc], &ldb);
        
        // 调用 BLAS 库中的 SGEMM 函数进行矩阵运算
        SGEMM( ftcs2, ftcs2, &nrow, &nrhs, &nsupc, &alpha, 
            &Lval[luptr+nsupc], &nsupr, &Bmat[fsupc], &ldb, 
            &beta, &work[0], &n );
#else
        // 调用 BLAS 库中的 dtrsm 函数进行矩阵运算
        dtrsm_("L", "L", "N", "U", &nsupc, &nrhs, &alpha,
               &Lval[luptr], &nsupr, &Bmat[fsupc], &ldb);
        
        // 调用 BLAS 库中的 dgemm 函数进行矩阵运算
        dgemm_( "N", "N", &nrow, &nrhs, &nsupc, &alpha, 
            &Lval[luptr+nsupc], &nsupr, &Bmat[fsupc], &ldb, 
            &beta, &work[0], &n );
#endif
        // 对每个右手边向量进行计算
        for (j = 0; j < nrhs; j++) {
            // 指向当前右手边向量的工作区和结果存储区
            rhs_work = &Bmat[(size_t)j * (size_t)ldb];
            work_col = &work[(size_t)j * (size_t)n];
            // 设置起始索引
            iptr = istart + nsupc;
            // 遍历当前行
            for (i = 0; i < nrow; i++) {
                // 计算当前行的全局行索引
                irow = L_SUB(iptr);
                // 执行向后散射操作
                rhs_work[irow] -= work_col[i]; /* Scatter */
                // 重置工作区域的当前列
                work_col[i] = 0.0;
                // 更新全局行索引
                iptr++;
            }
        }
#else        
        // 没有使用厂商提供的 BLAS 库，使用自定义的求解和乘法函数
        for (j = 0; j < nrhs; j++) {
            // 指向当前右手边向量的工作区
            rhs_work = &Bmat[(size_t)j * (size_t)ldb];
            // 调用自定义的直接向后代数求解函数
            dlsolve (nsupr, nsupc, &Lval[luptr], &rhs_work[fsupc]);
            // 调用自定义的矩阵向量乘法函数
            dmatvec (nsupr, nrow, nsupc, &Lval[luptr+nsupc],
                &rhs_work[fsupc], &work[0] );

            // 设置起始索引
            iptr = istart + nsupc;
            // 遍历当前行
            for (i = 0; i < nrow; i++) {
                // 计算当前行的全局行索引
                irow = L_SUB(iptr);
                // 执行向后散射操作
                rhs_work[irow] -= work[i];
                // 重置工作区域的当前列
                work[i] = 0.0;
                // 更新全局行索引
                iptr++;
            }
        }
#endif            
        } /* else ... */
    } /* for L-solve */

#if ( DEBUGlevel>=2 )
      // 如果调试级别大于等于2，则打印解决方案
      printf("After L-solve: y=\n");
    // 调用函数打印解
    dprint_soln(n, nrhs, Bmat);
#endif

    /*
     * Back solve Ux=y.
     */
    // 执行 Ux=y 的反向求解
    for (k = Lstore->nsuper; k >= 0; k--) {
        // 获取第 k 列的首元素列号和行号范围
        fsupc = L_FST_SUPC(k);
        istart = L_SUB_START(fsupc);
        nsupr = L_SUB_START(fsupc+1) - istart;
        nsupc = L_FST_SUPC(k+1) - fsupc;
        luptr = L_NZ_START(fsupc);

        // 计算解操作的数量
        solve_ops += nsupc * (nsupc + 1) * nrhs;

        // 如果当前列只有一个超节点
        if ( nsupc == 1 ) {
        // 指向当前右手边向量的工作区
        rhs_work = &Bmat[0];
        // 遍历当前右手边向量
        for (j = 0; j < nrhs; j++) {
            // 对于单个超节点，执行简化的前向/后向代数操作
            rhs_work[fsupc] /= Lval[luptr];
            rhs_work += ldb;
        }
        } else {
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
        // 将字符串转换为 Fortran 字符串描述符
        ftcs1 = _cptofcd("L", strlen("L"));
        ftcs2 = _cptofcd("U", strlen("U"));
        ftcs3 = _cptofcd("N", strlen("N"));
        // 调用 BLAS 库中的 STRSM 函数进行矩阵运算
        STRSM( ftcs1, ftcs2, ftcs3, ftcs3, &nsupc, &nrhs, &alpha,
               &Lval[luptr], &nsupr, &Bmat[fsupc], &ldb);
#else
        // 调用 BLAS 库中的 dtrsm 函数进行矩阵运算
        dtrsm_("L", "U", "N", "N", &nsupc, &nrhs, &alpha,
               &Lval[luptr], &nsupr, &Bmat[fsupc], &ldb);
#endif
#else        
        // 没有使用厂商提供的 BLAS 库，使用自定义的求解函数
        for (j = 0; j < nrhs; j++)
            // 调用自定义的直接向前代数求解函数
            dusolve ( nsupr, nsupc, &Lval[luptr], &Bmat[(size_t)fsupc + (size_t)j * (size_t)ldb] );
#endif        
        }

        // 针对每个右手边的向量进行处理
        for (j = 0; j < nrhs; ++j) {
            rhs_work = &Bmat[(size_t)j * (size_t)ldb];
            // 对当前列 fsupc 到 fsupc+nsupc-1 的每个元素进行处理
            for (jcol = fsupc; jcol < fsupc + nsupc; jcol++) {
                // 统计 U 矩阵中非零元素的操作次数
                solve_ops += 2 * (U_NZ_START(jcol+1) - U_NZ_START(jcol));
                // 遍历 U 矩阵中当前列 jcol 的非零元素
                for (i = U_NZ_START(jcol); i < U_NZ_START(jcol+1); i++ ){
                    irow = U_SUB(i);
                    // 更新右手边向量中的元素
                    rhs_work[irow] -= rhs_work[jcol] * Uval[i];
                }
            }
        }
        
    } /* for U-solve */

#if ( DEBUGlevel>=2 )
      // 在 U 解之后打印解向量 x
      printf("After U-solve: x=\n");
    dprint_soln(n, nrhs, Bmat);
#endif

    /* 计算最终解 X := Pc*X */
    for (i = 0; i < nrhs; i++) {
        rhs_work = &Bmat[(size_t)i * (size_t)ldb];
        // 根据列置换 perm_c 对右手边向量进行重排
        for (k = 0; k < n; k++) soln[k] = rhs_work[perm_c[k]];
        // 将重排后的结果写回原向量
        for (k = 0; k < n; k++) rhs_work[k] = soln[k];
    }
    
        // 统计求解操作的次数
        stat->ops[SOLVE] = solve_ops;

    } else { /* Solve A'*X=B or CONJ(A)*X=B */
    /* 对右手边进行置换以形成 Pc'*B */
    for (i = 0; i < nrhs; i++) {
        rhs_work = &Bmat[(size_t)i * (size_t)ldb];
        // 根据列置换 perm_c 对右手边向量进行重排
        for (k = 0; k < n; k++) soln[perm_c[k]] = rhs_work[k];
        // 将重排后的结果写回原向量
        for (k = 0; k < n; k++) rhs_work[k] = soln[k];
    }

    // 初始化求解操作次数为 0
    stat->ops[SOLVE] = 0;
    // 对每个右手边进行求解
    for (k = 0; k < nrhs; ++k) {
        
        /* 乘以 U' 的逆 */
        sp_dtrsv("U", "T", "N", L, U, &Bmat[(size_t)k * (size_t)ldb], stat, info);
        
        /* 乘以 L' 的逆 */
        sp_dtrsv("L", "T", "U", L, U, &Bmat[(size_t)k * (size_t)ldb], stat, info);
        
    }
    /* 计算最终解 X := Pr'*X (=inv(Pr)*X) */
    for (i = 0; i < nrhs; i++) {
        rhs_work = &Bmat[(size_t)i * (size_t)ldb];
        // 根据行置换 perm_r 对右手边向量进行重排
        for (k = 0; k < n; k++) soln[k] = rhs_work[perm_r[k]];
        // 将重排后的结果写回原向量
        for (k = 0; k < n; k++) rhs_work[k] = soln[k];
    }

    }

    // 释放工作空间
    SUPERLU_FREE(work);
    // 释放解向量空间
    SUPERLU_FREE(soln);
}

/*
 * 打印解向量的诊断输出 
 */
void
dprint_soln(int n, int nrhs, double *soln)
{
    int i;

    // 遍历解向量并打印每个元素
    for (i = 0; i < n; i++)
      printf("\t%d: %.4f\n", i, soln[i]);
}
```