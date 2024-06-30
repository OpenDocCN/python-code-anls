# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zgstrs.c`

```
/*!
 * \file
 * Copyright (c) 2003, The Regents of the University of California, through
 * Lawrence Berkeley National Laboratory (subject to receipt of any required 
 * approvals from U.S. Dept. of Energy) 
 * 
 * All rights reserved. 
 * 
 * The source code is distributed under BSD license, see the file License.txt
 * at the top-level directory.
 */

/*!
 * @file zgstrs.c
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

#include "slu_zdefs.h"
/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * ZGSTRS solves a system of linear equations A*X=B or A'*X=B
 * with A sparse and B dense, using the LU factorization computed by
 * ZGSTRF.
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
 *         zgstrf(). Use compressed row subscripts storage for supernodes,
 *         i.e., L has types: Stype = SLU_SC, Dtype = SLU_Z, Mtype = SLU_TRLU.
 *
 * U       (input) SuperMatrix*
 *         The factor U from the factorization Pr*A*Pc=L*U as computed by
 *         zgstrf(). Use column-wise storage scheme, i.e., U has types:
 *         Stype = SLU_NC, Dtype = SLU_Z, Mtype = SLU_TRU.
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
 *         B has types: Stype = SLU_DN, Dtype = SLU_Z, Mtype = SLU_GE.
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
zgstrs (trans_t trans, SuperMatrix *L, SuperMatrix *U,
        int *perm_c, int *perm_r, SuperMatrix *B,
        SuperLUStat_t *stat, int *info)
{

#ifdef _CRAY
    _fcd ftcs1, ftcs2, ftcs3, ftcs4;
#endif
#ifdef USE_VENDOR_BLAS
    doublecomplex   alpha = {1.0, 0.0}, beta = {1.0, 0.0};
    doublecomplex   *work_col;
#endif
    doublecomplex   temp_comp;
    DNformat *Bstore;
    doublecomplex   *Bmat;
    SCformat *Lstore;
    NCformat *Ustore;
    doublecomplex   *Lval, *Uval;
    int      fsupc, nrow, nsupr, nsupc, irow;
    int_t    i, j, k, luptr, istart, iptr;
    int      jcol, n, ldb, nrhs;
    doublecomplex   *work, *rhs_work, *soln;
    flops_t  solve_ops;
    void zprint_soln(int n, int nrhs, doublecomplex *soln);

    /* Test input parameters ... */

    // 初始化 info 为 0，表示没有错误
    *info = 0;

    // 获取 B 的存储格式信息
    Bstore = B->Store;
    ldb = Bstore->lda; // 获取 B 的 leading dimension
    nrhs = B->ncol;    // 获取 B 的列数作为右侧矩阵的数量

    // 检查 trans 参数是否合法
    if ( trans != NOTRANS && trans != TRANS && trans != CONJ )
        *info = -1; // 如果 trans 不是有效的枚举值，则设置 info 为 -1


继续下面的代码注释会超过1000字数限制，请问我需要继续添加代码注释吗
    // 检查矩阵 L 的属性：不是方阵、行数小于0、类型不匹配时，设置返回错误码 -2
    else if ( L->nrow != L->ncol || L->nrow < 0 ||
              L->Stype != SLU_SC || L->Dtype != SLU_Z || L->Mtype != SLU_TRLU )
        *info = -2;
    // 检查矩阵 U 的属性：不是方阵、行数小于0、类型不匹配时，设置返回错误码 -3
    else if ( U->nrow != U->ncol || U->nrow < 0 ||
              U->Stype != SLU_NC || U->Dtype != SLU_Z || U->Mtype != SLU_TRU )
        *info = -3;
    // 检查输入矩阵 B 的属性：行列数不匹配、类型不匹配时，设置返回错误码 -6
    else if ( ldb < SUPERLU_MAX(0, L->nrow) ||
              B->Stype != SLU_DN || B->Dtype != SLU_Z || B->Mtype != SLU_GE )
        *info = -6;
    
    // 如果有任何错误码被设置，打印错误信息并直接返回
    if ( *info ) {
        int ii = -(*info);
        input_error("zgstrs", &ii);
        return;
    }

    // 获取矩阵 L 的行数
    n = L->nrow;
    // 分配存储空间以存储工作数组
    work = doublecomplexCalloc((size_t) n * (size_t) nrhs);
    // 分配存储空间以存储解向量
    soln = doublecomplexMalloc((size_t) n);
    // 检查分配是否成功，如果不成功则终止程序并打印错误信息
    if ( !work ) ABORT("Malloc fails for local work[].");
    if ( !soln ) ABORT("Malloc fails for local soln[].");

    // 获取矩阵 B 中的数据
    Bmat = Bstore->nzval;
    // 获取矩阵 L 中的存储结构
    Lstore = L->Store;
    Lval = Lstore->nzval;
    // 获取矩阵 U 中的存储结构
    Ustore = U->Store;
    Uval = Ustore->nzval;
    // 初始化求解操作计数器
    solve_ops = 0;

    // 如果转置标记为 NOTRANS
    if ( trans == NOTRANS ) {
        /* Permute right hand sides to form Pr*B */
        // 对每个右侧向量进行置换以形成 Pr*B
        for (i = 0; i < nrhs; i++) {
            rhs_work = &Bmat[(size_t)i * (size_t)ldb];
            // 使用行置换向量 perm_r 对右侧向量进行置换
            for (k = 0; k < n; k++) soln[perm_r[k]] = rhs_work[k];
            // 将置换后的结果写回原始位置
            for (k = 0; k < n; k++) rhs_work[k] = soln[k];
        }
        
        /* Forward solve PLy=Pb. */
        // 执行前向求解操作 PLy = Pb
        for (k = 0; k <= Lstore->nsuper; k++) {
            // 获取超节点 k 的起始列
            fsupc = L_FST_SUPC(k);
            // 获取超节点 k 在 L 存储结构中的起始位置和结束位置
            istart = L_SUB_START(fsupc);
            // 计算超节点 k 的行数和列数
            nsupr = L_SUB_START(fsupc+1) - istart;
            nsupc = L_FST_SUPC(k+1) - fsupc;
            nrow = nsupr - nsupc;

            // 更新求解操作计数器
            solve_ops += 4 * nsupc * (nsupc - 1) * nrhs;
            solve_ops += 8 * nrow * nsupc * nrhs;
            
            // 如果超节点列数为 1，使用特定算法求解
            if ( nsupc == 1 ) {
                for (j = 0; j < nrhs; j++) {
                    rhs_work = &Bmat[(size_t)j * (size_t)ldb];
                    // 获取超节点列对应的非零元素起始位置
                    luptr = L_NZ_START(fsupc);
                    // 执行特定的矩阵向量乘法和更新操作
                    for (iptr=istart+1; iptr < L_SUB_START(fsupc+1); iptr++){
                        irow = L_SUB(iptr);
                        ++luptr;
                        zz_mult(&temp_comp, &rhs_work[fsupc], &Lval[luptr]);
                        z_sub(&rhs_work[irow], &rhs_work[irow], &temp_comp);
                    }
                }
            } else {
                // 如果超节点列数大于 1，执行通用的前向求解操作
                luptr = L_NZ_START(fsupc);
                // 此处省略部分代码，继续进行后续的求解步骤
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
        ftcs1 = _cptofcd("L", strlen("L"));
        ftcs2 = _cptofcd("N", strlen("N"));
        ftcs3 = _cptofcd("U", strlen("U"));
        // 转换字符常量为 FORTRAN 字符描述符，表示矩阵操作中的参数
        CTRSM( ftcs1, ftcs1, ftcs2, ftcs3, &nsupc, &nrhs, &alpha,
               &Lval[luptr], &nsupr, &Bmat[fsupc], &ldb);
        // 使用 BLAS 函数 CTRSM 进行矩阵相乘和矩阵分块求解
        CGEMM( ftcs2, ftcs2, &nrow, &nrhs, &nsupc, &alpha, 
            &Lval[luptr+nsupc], &nsupr, &Bmat[fsupc], &ldb, 
            &beta, &work[0], &n );
#else
        // 使用 BLAS 函数 ztrsm 进行复杂矩阵的相乘和矩阵分块求解
        ztrsm_("L", "L", "N", "U", &nsupc, &nrhs, &alpha,
               &Lval[luptr], &nsupr, &Bmat[fsupc], &ldb);
        // 使用 BLAS 函数 zgemm 进行复杂矩阵的相乘
        zgemm_( "N", "N", &nrow, &nrhs, &nsupc, &alpha, 
            &Lval[luptr+nsupc], &nsupr, &Bmat[fsupc], &ldb, 
            &beta, &work[0], &n );
#endif
        // 对每个右侧向量进行操作
        for (j = 0; j < nrhs; j++) {
            // 指向当前右侧向量的工作区
            rhs_work = &Bmat[(size_t)j * (size_t)ldb];
            // 指向当前工作向量的列
            work_col = &work[(size_t)j * (size_t)n];
            // 计算起始行指针
            iptr = istart + nsupc;
            // 对当前右侧向量的每一行进行操作
            for (i = 0; i < nrow; i++) {
                // 获取当前行的索引
                irow = L_SUB(iptr);
                // 从右侧向量中减去工作向量
                z_sub(&rhs_work[irow], &rhs_work[irow], &work_col[i]);
                // 将工作向量的实部和虚部设置为零
                work_col[i].r = 0.0;
                work_col[i].i = 0.0;
                // 递增行指针
                iptr++;
            }
        }
#else        
        // 对每个右侧向量进行操作
        for (j = 0; j < nrhs; j++) {
            // 指向当前右侧向量的工作区
            rhs_work = &Bmat[(size_t)j * (size_t)ldb];
            // 使用 LU 分解求解线性方程组的下三角部分
            zlsolve (nsupr, nsupc, &Lval[luptr], &rhs_work[fsupc]);
            // 使用矩阵向量乘法计算结果
            zmatvec (nsupr, nrow, nsupc, &Lval[luptr+nsupc],
                &rhs_work[fsupc], &work[0] );
            // 计算起始行指针
            iptr = istart + nsupc;
            // 对当前右侧向量的每一行进行操作
            for (i = 0; i < nrow; i++) {
                // 获取当前行的索引
                irow = L_SUB(iptr);
                // 从右侧向量中减去工作向量
                z_sub(&rhs_work[irow], &rhs_work[irow], &work[i]);
                // 将工作向量的实部和虚部设置为零
                work[i].r = 0.;
                work[i].i = 0.;
                // 递增行指针
                iptr++;
            }
        }
#endif            
        } /* else ... */
    } /* for L-solve */

#if ( DEBUGlevel>=2 )
      printf("After L-solve: y=\n");
    // 打印调试信息：L 分解后的解 y
    zprint_soln(n, nrhs, Bmat);
#endif

    /*
     * Back solve Ux=y.
     */
    // 对 Ux=y 进行回代求解
    for (k = Lstore->nsuper; k >= 0; k--) {
        fsupc = L_FST_SUPC(k);
        istart = L_SUB_START(fsupc);
        nsupr = L_SUB_START(fsupc+1) - istart;
        nsupc = L_FST_SUPC(k+1) - fsupc;
        luptr = L_NZ_START(fsupc);

        // 统计求解操作数
        solve_ops += 4 * nsupc * (nsupc + 1) * nrhs;

        if ( nsupc == 1 ) {
        // 如果 U 是单元素，直接计算
        rhs_work = &Bmat[0];
        for (j = 0; j < nrhs; j++) {
            z_div(&rhs_work[fsupc], &rhs_work[fsupc], &Lval[luptr]);
            rhs_work += ldb;
        }
        } else {
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
        ftcs1 = _cptofcd("L", strlen("L"));
        ftcs2 = _cptofcd("U", strlen("U"));
        ftcs3 = _cptofcd("N", strlen("N"));
        // 使用 CTRSM 进行矩阵相乘和矩阵分块求解
        CTRSM( ftcs1, ftcs2, ftcs3, ftcs3, &nsupc, &nrhs, &alpha,
               &Lval[luptr], &nsupr, &Bmat[fsupc], &ldb);
#else
        // 使用 ztrsm 进行复杂矩阵的相乘和矩阵分块求解
        ztrsm_("L", "U", "N", "N", &nsupc, &nrhs, &alpha,
               &Lval[luptr], &nsupr, &Bmat[fsupc], &ldb);
#endif
#else        
        for (j = 0; j < nrhs; j++)
            zusolve ( nsupr, nsupc, &Lval[luptr], &Bmat[(size_t)fsupc + (size_t)j * (size_t)ldb] );
#endif        


#else        
        // 如果未定义某些情况下的处理代码
        for (j = 0; j < nrhs; j++)
            // 调用 zusolve 函数解决线性方程组，传入参数为行数、列数、Lval数组中的指定位置、以及Bmat数组中特定列的部分
            zusolve ( nsupr, nsupc, &Lval[luptr], &Bmat[(size_t)fsupc + (size_t)j * (size_t)ldb] );
#endif        
        }

        for (j = 0; j < nrhs; ++j) {
        rhs_work = &Bmat[(size_t)j * (size_t)ldb];
        for (jcol = fsupc; jcol < fsupc + nsupc; jcol++) {
            solve_ops += 8*(U_NZ_START(jcol+1) - U_NZ_START(jcol));
            for (i = U_NZ_START(jcol); i < U_NZ_START(jcol+1); i++ ){
            irow = U_SUB(i);
            zz_mult(&temp_comp, &rhs_work[jcol], &Uval[i]);
            z_sub(&rhs_work[irow], &rhs_work[irow], &temp_comp);
            }
        }
        }
        
    } /* for U-solve */

#if ( DEBUGlevel>=2 )
      printf("After U-solve: x=\n");
    zprint_soln(n, nrhs, Bmat);
#endif


    /* Compute the final solution X := Pc*X. */
    for (i = 0; i < nrhs; i++) {
        rhs_work = &Bmat[(size_t)i * (size_t)ldb];
        for (k = 0; k < n; k++) soln[k] = rhs_work[perm_c[k]];
        for (k = 0; k < n; k++) rhs_work[k] = soln[k];
    }


        stat->ops[SOLVE] = solve_ops;


    } else { /* Solve A'*X=B or CONJ(A)*X=B */
    /* Permute right hand sides to form Pc'*B. */
    for (i = 0; i < nrhs; i++) {
        rhs_work = &Bmat[(size_t)i * (size_t)ldb];
        for (k = 0; k < n; k++) soln[perm_c[k]] = rhs_work[k];
        for (k = 0; k < n; k++) rhs_work[k] = soln[k];
    }

    stat->ops[SOLVE] = 0;
        if (trans == TRANS) {
        for (k = 0; k < nrhs; ++k) {
            /* Multiply by inv(U'). */
            sp_ztrsv("U", "T", "N", L, U, &Bmat[(size_t)k * (size_t)ldb], stat, info);
        
            /* Multiply by inv(L'). */
            sp_ztrsv("L", "T", "U", L, U, &Bmat[(size_t)k * (size_t)ldb], stat, info);
        }
         } else { /* trans == CONJ */
            for (k = 0; k < nrhs; ++k) {                
                /* Multiply by conj(inv(U')). */
                sp_ztrsv("U", "C", "N", L, U, &Bmat[(size_t)k * (size_t)ldb], stat, info);
                
                /* Multiply by conj(inv(L')). */
                sp_ztrsv("L", "C", "U", L, U, &Bmat[(size_t)k * (size_t)ldb], stat, info);
        }
         }
    /* Compute the final solution X := Pr'*X (=inv(Pr)*X) */
    for (i = 0; i < nrhs; i++) {
        rhs_work = &Bmat[(size_t)i * (size_t)ldb];
        for (k = 0; k < n; k++) soln[k] = rhs_work[perm_r[k]];
        for (k = 0; k < n; k++) rhs_work[k] = soln[k];
    }

    }

    SUPERLU_FREE(work);
    SUPERLU_FREE(soln);
}

/*
 * Diagnostic print of the solution vector 
 */
void
zprint_soln(int n, int nrhs, doublecomplex *soln)
{
    int i;

    for (i = 0; i < n; i++)
      printf("\t%d: %.4f\t%.4f\n", i, soln[i].r, soln[i].i);
}
```