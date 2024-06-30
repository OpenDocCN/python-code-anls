# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\cgstrs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file cgstrs.c
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

#include "slu_cdefs.h"
/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * CGSTRS solves a system of linear equations A*X=B or A'*X=B
 * with A sparse and B dense, using the LU factorization computed by
 * CGSTRF.
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
 *         cgstrf(). Use compressed row subscripts storage for supernodes,
 *         i.e., L has types: Stype = SLU_SC, Dtype = SLU_C, Mtype = SLU_TRLU.
 *
 * U       (input) SuperMatrix*
 *         The factor U from the factorization Pr*A*Pc=L*U as computed by
 *         cgstrf(). Use column-wise storage scheme, i.e., U has types:
 *         Stype = SLU_NC, Dtype = SLU_C, Mtype = SLU_TRU.
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
 *         B has types: Stype = SLU_DN, Dtype = SLU_C, Mtype = SLU_GE.
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
cgstrs (trans_t trans, SuperMatrix *L, SuperMatrix *U,
        int *perm_c, int *perm_r, SuperMatrix *B,
        SuperLUStat_t *stat, int *info)
{

#ifdef _CRAY
    _fcd ftcs1, ftcs2, ftcs3, ftcs4;
#endif
#ifdef USE_VENDOR_BLAS
    singlecomplex   alpha = {1.0, 0.0}, beta = {1.0, 0.0};
    singlecomplex   *work_col;
#endif
    singlecomplex   temp_comp;
    DNformat *Bstore;
    singlecomplex   *Bmat;
    SCformat *Lstore;
    NCformat *Ustore;
    singlecomplex   *Lval, *Uval;
    int      fsupc, nrow, nsupr, nsupc, irow;
    int_t    i, j, k, luptr, istart, iptr;
    int      jcol, n, ldb, nrhs;
    singlecomplex   *work, *rhs_work, *soln;
    flops_t  solve_ops;
    void cprint_soln(int n, int nrhs, singlecomplex *soln);

    /* Test input parameters ... */

    // 初始化 info 为 0，表示输入参数合法
    *info = 0;

    // 获取 B 的存储格式
    Bstore = B->Store;
    // 获取 B 的 leading dimension
    ldb = Bstore->lda;
    // 获取 B 的列数（右手边矩阵的列数）
    nrhs = B->ncol;

    // 检查 trans 参数是否合法
    if ( trans != NOTRANS && trans != TRANS && trans != CONJ )
        *info = -1;


The code snippet provided is a function `cgstrs` that solves a system of linear equations using LU factorization. The comments above explain the purpose of the function and describe each input argument's role and expected types.
    else if ( L->nrow != L->ncol || L->nrow < 0 ||
          L->Stype != SLU_SC || L->Dtype != SLU_C || L->Mtype != SLU_TRLU )
    *info = -2;
    else if ( U->nrow != U->ncol || U->nrow < 0 ||
          U->Stype != SLU_NC || U->Dtype != SLU_C || U->Mtype != SLU_TRU )
    *info = -3;
    else if ( ldb < SUPERLU_MAX(0, L->nrow) ||
          B->Stype != SLU_DN || B->Dtype != SLU_C || B->Mtype != SLU_GE )
    *info = -6;
    if ( *info ) {
    int ii = -(*info);
    input_error("cgstrs", &ii);
    return;
    }

    n = L->nrow;  // 获取矩阵 L 的行数
    work = complexCalloc((size_t) n * (size_t) nrhs);  // 分配复数类型的工作空间
    if ( !work ) ABORT("Malloc fails for local work[].");  // 检查内存分配是否成功，失败则终止程序
    soln = complexMalloc((size_t) n);  // 分配复数类型的解空间
    if ( !soln ) ABORT("Malloc fails for local soln[].");  // 检查内存分配是否成功，失败则终止程序

    Bmat = Bstore->nzval;  // 获取矩阵 B 存储中的非零值数组
    Lstore = L->Store;  // 获取矩阵 L 的存储结构
    Lval = Lstore->nzval;  // 获取矩阵 L 的非零值数组
    Ustore = U->Store;  // 获取矩阵 U 的存储结构
    Uval = Ustore->nzval;  // 获取矩阵 U 的非零值数组
    solve_ops = 0;  // 初始化求解操作数统计变量

    if ( trans == NOTRANS ) {
    /* Permute right hand sides to form Pr*B */
    // 对右侧向量进行置换，形成 Pr*B
    for (i = 0; i < nrhs; i++) {
        rhs_work = &Bmat[(size_t)i * (size_t)ldb];
        // 根据置换 perm_r 对右侧向量进行重新排列
        for (k = 0; k < n; k++) soln[perm_r[k]] = rhs_work[k];
        // 将重新排列后的结果复制回原右侧向量
        for (k = 0; k < n; k++) rhs_work[k] = soln[k];
    }
    
    /* Forward solve PLy=Pb. */
    // 前向求解 PLy = Pb
    for (k = 0; k <= Lstore->nsuper; k++) {
        fsupc = L_FST_SUPC(k);  // 获取超节点 k 的起始列
        istart = L_SUB_START(fsupc);  // 获取超节点 k 的行索引起始位置
        nsupr = L_SUB_START(fsupc+1) - istart;  // 获取超节点 k 的行索引长度
        nsupc = L_FST_SUPC(k+1) - fsupc;  // 获取超节点 k 的列数
        nrow = nsupr - nsupc;  // 计算超节点 k 的非对角元素的行数

        // 更新求解操作数统计
        solve_ops += 4 * nsupc * (nsupc - 1) * nrhs;
        solve_ops += 8 * nrow * nsupc * nrhs;
        
        if ( nsupc == 1 ) {
        for (j = 0; j < nrhs; j++) {
            rhs_work = &Bmat[(size_t)j * (size_t)ldb];
                luptr = L_NZ_START(fsupc);
            for (iptr=istart+1; iptr < L_SUB_START(fsupc+1); iptr++){
            irow = L_SUB(iptr);
            ++luptr;
            cc_mult(&temp_comp, &rhs_work[fsupc], &Lval[luptr]);
            c_sub(&rhs_work[irow], &rhs_work[irow], &temp_comp);
            }
        }
        } else {
            luptr = L_NZ_START(fsupc);
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
        ftcs1 = _cptofcd("L", strlen("L"));
        ftcs2 = _cptofcd("N", strlen("N"));
        ftcs3 = _cptofcd("U", strlen("U"));
        CTRSM( ftcs1, ftcs1, ftcs2, ftcs3, &nsupc, &nrhs, &alpha,
               &Lval[luptr], &nsupr, &Bmat[fsupc], &ldb);
        
        CGEMM( ftcs2, ftcs2, &nrow, &nrhs, &nsupc, &alpha, 
            &Lval[luptr+nsupc], &nsupr, &Bmat[fsupc], &ldb, 
            &beta, &work[0], &n );
#else
        ctrsm_("L", "L", "N", "U", &nsupc, &nrhs, &alpha,
               &Lval[luptr], &nsupr, &Bmat[fsupc], &ldb);
        
        cgemm_( "N", "N", &nrow, &nrhs, &nsupc, &alpha, 
            &Lval[luptr+nsupc], &nsupr, &Bmat[fsupc], &ldb, 
            &beta, &work[0], &n );
#endif
        for (j = 0; j < nrhs; j++) {
            // 指向当前右手边的工作向量
            rhs_work = &Bmat[(size_t)j * (size_t)ldb];
            // 指向当前工作向量的工作列
            work_col = &work[(size_t)j * (size_t)n];
            // 指向当前超节点的起始行
            iptr = istart + nsupc;
            for (i = 0; i < nrow; i++) {
                // 当前行的行索引
                irow = L_SUB(iptr);
                // 复数减法操作：rhs_work[irow] = rhs_work[irow] - work_col[i]
                c_sub(&rhs_work[irow], &rhs_work[irow], &work_col[i]);
                // 将工作列的实部和虚部设为零
                work_col[i].r = 0.0;
                work_col[i].i = 0.0;
                // 下一行索引
                iptr++;
            }
        }
#else        
        for (j = 0; j < nrhs; j++) {
            // 指向当前右手边的工作向量
            rhs_work = &Bmat[(size_t)j * (size_t)ldb];
            // 调用CLSOLVE解决线性方程组
            clsolve (nsupr, nsupc, &Lval[luptr], &rhs_work[fsupc]);
            // 调用CMATVEC进行矩阵向量乘法
            cmatvec (nsupr, nrow, nsupc, &Lval[luptr+nsupc],
                &rhs_work[fsupc], &work[0] );

            // 指向当前超节点的起始行
            iptr = istart + nsupc;
            for (i = 0; i < nrow; i++) {
                // 当前行的行索引
                irow = L_SUB(iptr);
                // 复数减法操作：rhs_work[irow] = rhs_work[irow] - work[i]
                c_sub(&rhs_work[irow], &rhs_work[irow], &work[i]);
                // 将work[i]的实部设为零
                work[i].r = 0.;
                // 将work[i]的虚部设为零
                work[i].i = 0.;
                // 下一行索引
                iptr++;
            }
        }
#endif            
        } /* else ... */
    } /* for L-solve */

#if ( DEBUGlevel>=2 )
      // 调试级别为2以上时打印信息
      printf("After L-solve: y=\n");
    // 打印解的信息
    cprint_soln(n, nrhs, Bmat);
#endif

    /*
     * Back solve Ux=y.
     */
    for (k = Lstore->nsuper; k >= 0; k--) {
        // 当前超节点的第一个列索引
        fsupc = L_FST_SUPC(k);
        // 当前超节点第一个列索引对应的起始行索引
        istart = L_SUB_START(fsupc);
        // 当前超节点第一个列索引对应的行数
        nsupr = L_SUB_START(fsupc+1) - istart;
        // 当前超节点的列数
        nsupc = L_FST_SUPC(k+1) - fsupc;
        // 当前超节点的非零元素起始位置
        luptr = L_NZ_START(fsupc);

        // 解操作次数累计
        solve_ops += 4 * nsupc * (nsupc + 1) * nrhs;

        if ( nsupc == 1 ) {
            // 如果超节点列数为1，则直接进行复数除法操作
            rhs_work = &Bmat[0];
            for (j = 0; j < nrhs; j++) {
                // rhs_work[fsupc] = rhs_work[fsupc] / Lval[luptr]
                c_div(&rhs_work[fsupc], &rhs_work[fsupc], &Lval[luptr]);
                // 更新rhs_work指向下一个右手边的元素
                rhs_work += ldb;
            }
        } else {
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
        ftcs1 = _cptofcd("L", strlen("L"));
        ftcs2 = _cptofcd("U", strlen("U"));
        ftcs3 = _cptofcd("N", strlen("N"));
        CTRSM( ftcs1, ftcs2, ftcs3, ftcs3, &nsupc, &nrhs, &alpha,
               &Lval[luptr], &nsupr, &Bmat[fsupc], &ldb);
#else
        ctrsm_("L", "U", "N", "N", &nsupc, &nrhs, &alpha,
               &Lval[luptr], &nsupr, &Bmat[fsupc], &ldb);
#endif
#else        
        // 如果不是第一次循环（即已经计算了 L 的乘法），则继续计算 cusolve
        for (j = 0; j < nrhs; j++)
            cusolve ( nsupr, nsupc, &Lval[luptr], &Bmat[(size_t)fsupc + (size_t)j * (size_t)ldb] );
#endif        
        }

        // 对每个右侧向量进行 U 的解算
        for (j = 0; j < nrhs; ++j) {
            // 指向当前右侧向量的工作区
            rhs_work = &Bmat[(size_t)j * (size_t)ldb];
            // 对当前列范围内的每一列进行处理
            for (jcol = fsupc; jcol < fsupc + nsupc; jcol++) {
                // 更新解算操作数计数
                solve_ops += 8*(U_NZ_START(jcol+1) - U_NZ_START(jcol));
                // 对当前列的非零元素进行遍历
                for (i = U_NZ_START(jcol); i < U_NZ_START(jcol+1); i++ ){
                    irow = U_SUB(i);
                    // 计算临时复数
                    cc_mult(&temp_comp, &rhs_work[jcol], &Uval[i]);
                    // 线性组合更新右侧向量
                    c_sub(&rhs_work[irow], &rhs_work[irow], &temp_comp);
                }
            }
        }
        
    } /* for U-solve */

#if ( DEBUGlevel>=2 )
      // 调试模式下打印 U 解算后的结果
    printf("After U-solve: x=\n");
    cprint_soln(n, nrhs, Bmat);
#endif

    /* 计算最终解 X := Pc*X */
    for (i = 0; i < nrhs; i++) {
        // 指向当前右侧向量的工作区
        rhs_work = &Bmat[(size_t)i * (size_t)ldb];
        // 根据列置换重新排列解向量
        for (k = 0; k < n; k++) soln[k] = rhs_work[perm_c[k]];
        // 将重新排列后的解向量复制回原始位置
        for (k = 0; k < n; k++) rhs_work[k] = soln[k];
    }
    
        // 记录解算操作数
    stat->ops[SOLVE] = solve_ops;

    } else { /* Solve A'*X=B or CONJ(A)*X=B */
    /* 对右侧向量进行列置换以形成 Pc'*B */
    for (i = 0; i < nrhs; i++) {
        // 指向当前右侧向量的工作区
        rhs_work = &Bmat[(size_t)i * (size_t)ldb];
        // 根据列置换重新排列右侧向量
        for (k = 0; k < n; k++) soln[perm_c[k]] = rhs_work[k];
        // 将重新排列后的右侧向量复制回原始位置
        for (k = 0; k < n; k++) rhs_work[k] = soln[k];
    }

    // 对解算操作数进行初始化
    stat->ops[SOLVE] = 0;
        // 如果是转置操作
        if (trans == TRANS) {
            // 对每个右侧向量进行操作：乘以 U 的逆转置
            for (k = 0; k < nrhs; ++k) {
                sp_ctrsv("U", "T", "N", L, U, &Bmat[(size_t)k * (size_t)ldb], stat, info);
        
                // 对每个右侧向量进行操作：乘以 L 的逆转置
                sp_ctrsv("L", "T", "U", L, U, &Bmat[(size_t)k * (size_t)ldb], stat, info);
            }
         } else { /* trans == CONJ */
            // 对每个右侧向量进行操作：乘以 U 的共轭逆转置
            for (k = 0; k < nrhs; ++k) {                
                sp_ctrsv("U", "C", "N", L, U, &Bmat[(size_t)k * (size_t)ldb], stat, info);
                
                // 对每个右侧向量进行操作：乘以 L 的共轭逆转置
                sp_ctrsv("L", "C", "U", L, U, &Bmat[(size_t)k * (size_t)ldb], stat, info);
        }
         }
    /* 计算最终解 X := Pr'*X (=inv(Pr)*X) */
    for (i = 0; i < nrhs; i++) {
        // 指向当前右侧向量的工作区
        rhs_work = &Bmat[(size_t)i * (size_t)ldb];
        // 根据行置换重新排列解向量
        for (k = 0; k < n; k++) soln[k] = rhs_work[perm_r[k]];
        // 将重新排列后的解向量复制回原始位置
        for (k = 0; k < n; k++) rhs_work[k] = soln[k];
    }

    }

    // 释放工作区和解向量内存
    SUPERLU_FREE(work);
    SUPERLU_FREE(soln);
}

/*
 * 打印解向量的诊断输出
 */
void
cprint_soln(int n, int nrhs, singlecomplex *soln)
{
    int i;

    // 遍历解向量的每个元素，并打印实部和虚部
    for (i = 0; i < n; i++)
      printf("\t%d: %.4f\t%.4f\n", i, soln[i].r, soln[i].i);
}
```