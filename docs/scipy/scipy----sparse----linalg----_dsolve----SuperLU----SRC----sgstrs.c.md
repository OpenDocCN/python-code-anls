# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\sgstrs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file sgstrs.c
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

#include "slu_sdefs.h"
/*!
 *
 * <pre>
 * Purpose
 * =======
 *
 * SGSTRS solves a system of linear equations A*X=B or A'*X=B
 * with A sparse and B dense, using the LU factorization computed by
 * SGSTRF.
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
 *         sgstrf(). Use compressed row subscripts storage for supernodes,
 *         i.e., L has types: Stype = SLU_SC, Dtype = SLU_S, Mtype = SLU_TRLU.
 *
 * U       (input) SuperMatrix*
 *         The factor U from the factorization Pr*A*Pc=L*U as computed by
 *         sgstrf(). Use column-wise storage scheme, i.e., U has types:
 *         Stype = SLU_NC, Dtype = SLU_S, Mtype = SLU_TRU.
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
 *         B has types: Stype = SLU_DN, Dtype = SLU_S, Mtype = SLU_GE.
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
sgstrs (trans_t trans, SuperMatrix *L, SuperMatrix *U,
        int *perm_c, int *perm_r, SuperMatrix *B,
        SuperLUStat_t *stat, int *info)
{

#ifdef _CRAY
    _fcd ftcs1, ftcs2, ftcs3, ftcs4;
#endif
#ifdef USE_VENDOR_BLAS
    float   alpha = 1.0, beta = 1.0;
    float   *work_col;
#endif
    DNformat *Bstore;
    float   *Bmat;
    SCformat *Lstore;
    NCformat *Ustore;
    float   *Lval, *Uval;
    int      fsupc, nrow, nsupr, nsupc, irow;
    int_t    i, j, k, luptr, istart, iptr;
    int      jcol, n, ldb, nrhs;
    float   *work, *rhs_work, *soln;
    flops_t  solve_ops;
    void sprint_soln(int n, int nrhs, float *soln);

    /* Test input parameters ... */

    // 初始化 info 为 0，表示成功执行
    *info = 0;

    // 提取 B 的存储结构并设置 ldb 和 nrhs
    Bstore = B->Store;
    ldb = Bstore->lda;
    nrhs = B->ncol;

    // 检查 trans 的合法性
    // 如果 trans 不是 NOTRANS, TRANS, 或 CONJ，则将 info 设置为 -1
    if ( trans != NOTRANS && trans != TRANS && trans != CONJ )
        *info = -1;
    
    // 检查 L 的维度和类型
    // 如果 L 的行数不等于列数，或者行数小于 0，
    // 或者 L 的存储类型不是 SLU_SC，数据类型不是 SLU_S，或者矩阵类型不是 SLU_TRLU，
    // 则将 info 设置为 -2
    else if ( L->nrow != L->ncol || L->nrow < 0 ||
              L->Stype != SLU_SC || L->Dtype != SLU_S || L->Mtype != SLU_TRLU )
        *info = -2;

    // 继续检查其它参数...

    // 如果没有出现非法值的情况，则继续执行函数的其余部分
    else if ( U->nrow != U->ncol || U->nrow < 0 ||
          U->Stype != SLU_NC || U->Dtype != SLU_S || U->Mtype != SLU_TRU )
    *info = -3;
    else if ( ldb < SUPERLU_MAX(0, L->nrow) ||
          B->Stype != SLU_DN || B->Dtype != SLU_S || B->Mtype != SLU_GE )
    *info = -6;
    if ( *info ) {
    int ii = -(*info);
    input_error("sgstrs", &ii);
    return;
    }


    // 检查矩阵 U 的属性，如果不符合要求则设置错误码为 -3
    else if ( U->nrow != U->ncol || U->nrow < 0 ||
          U->Stype != SLU_NC || U->Dtype != SLU_S || U->Mtype != SLU_TRU )
    *info = -3;
    // 检查右侧矩阵 B 的属性，如果不符合要求则设置错误码为 -6
    else if ( ldb < SUPERLU_MAX(0, L->nrow) ||
          B->Stype != SLU_DN || B->Dtype != SLU_S || B->Mtype != SLU_GE )
    *info = -6;
    // 如果出现错误码，则通过错误处理函数打印错误信息并返回
    if ( *info ) {
    int ii = -(*info);
    input_error("sgstrs", &ii);
    return;
    }



    n = L->nrow;
    work = floatCalloc((size_t) n * (size_t) nrhs);
    if ( !work ) ABORT("Malloc fails for local work[].");
    soln = floatMalloc((size_t) n);
    if ( !soln ) ABORT("Malloc fails for local soln[].");

    Bmat = Bstore->nzval;
    Lstore = L->Store;
    Lval = Lstore->nzval;
    Ustore = U->Store;
    Uval = Ustore->nzval;
    solve_ops = 0;


    // 获取矩阵 L 的行数作为 n
    n = L->nrow;
    // 分配大小为 n * nrhs 的工作内存空间，如果分配失败则中止程序
    work = floatCalloc((size_t) n * (size_t) nrhs);
    if ( !work ) ABORT("Malloc fails for local work[].");
    // 分配大小为 n 的解向量内存空间，如果分配失败则中止程序
    soln = floatMalloc((size_t) n);
    if ( !soln ) ABORT("Malloc fails for local soln[].");

    // 获取矩阵 B 的非零值数组
    Bmat = Bstore->nzval;
    // 获取矩阵 L 的存储结构
    Lstore = L->Store;
    // 获取矩阵 L 的非零值数组
    Lval = Lstore->nzval;
    // 获取矩阵 U 的存储结构
    Ustore = U->Store;
    // 获取矩阵 U 的非零值数组
    Uval = Ustore->nzval;
    // 初始化解操作数为 0
    solve_ops = 0;



    if ( trans == NOTRANS ) {
    /* Permute right hand sides to form Pr*B */
    for (i = 0; i < nrhs; i++) {
        rhs_work = &Bmat[(size_t)i * (size_t)ldb];
        for (k = 0; k < n; k++) soln[perm_r[k]] = rhs_work[k];
        for (k = 0; k < n; k++) rhs_work[k] = soln[k];
    }
    
    /* Forward solve PLy=Pb. */
    for (k = 0; k <= Lstore->nsuper; k++) {
        fsupc = L_FST_SUPC(k);
        istart = L_SUB_START(fsupc);
        nsupr = L_SUB_START(fsupc+1) - istart;
        nsupc = L_FST_SUPC(k+1) - fsupc;
        nrow = nsupr - nsupc;

        solve_ops += nsupc * (nsupc - 1) * nrhs;
        solve_ops += 2 * nrow * nsupc * nrhs;
        
        if ( nsupc == 1 ) {
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


    // 如果不需要转置操作
    if ( trans == NOTRANS ) {
    /* Permute right hand sides to form Pr*B */
    // 对右侧矩阵 B 进行置换，得到 Pr*B
    for (i = 0; i < nrhs; i++) {
        rhs_work = &Bmat[(size_t)i * (size_t)ldb];
        // 根据行置换 perm_r 对右侧向量进行重新排列
        for (k = 0; k < n; k++) soln[perm_r[k]] = rhs_work[k];
        // 将重新排列后的向量复制回原始位置
        for (k = 0; k < n; k++) rhs_work[k] = soln[k];
    }
    
    /* Forward solve PLy=Pb. */
    // 前向求解 PLy = Pb
    for (k = 0; k <= Lstore->nsuper; k++) {
        fsupc = L_FST_SUPC(k);    // 当前超节点列索引
        istart = L_SUB_START(fsupc);    // 超节点首行在 L_SUB 中的起始位置
        nsupr = L_SUB_START(fsupc+1) - istart;    // 超节点行数
        nsupc = L_FST_SUPC(k+1) - fsupc;    // 超节点列数
        nrow = nsupr - nsupc;    // 超节点非对角元素的行数

        // 更新解操作数统计
        solve_ops += nsupc * (nsupc - 1) * nrhs;
        solve_ops += 2 * nrow * nsupc * nrhs;
        
        // 如果超节点列数为 1，则使用直接法求解
        if ( nsupc == 1 ) {
        for (j = 0; j < nrhs; j++) {
            rhs_work = &Bmat[(size_t)j * (size_t)ldb];
                luptr = L_NZ_START(fsupc);
            for (iptr=istart+1; iptr < L_SUB_START(fsupc+1); iptr++){
            irow = L_SUB(iptr);
            ++luptr;
            // 利用 L 的非零值进行前向替换
            rhs_work[irow] -= rhs_work[fsupc] * Lval[luptr];
            }
        }
        } else {
            luptr = L_NZ_START(fsupc);
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
        // 将字符串 "L" 转换成 _fcd 类型，并计算其长度，用于调用 BLAS 函数
        ftcs1 = _cptofcd("L", strlen("L"));
        // 将字符串 "N" 转换成 _fcd 类型，并计算其长度，用于调用 BLAS 函数
        ftcs2 = _cptofcd("N", strlen("N"));
        // 将字符串 "U" 转换成 _fcd 类型，并计算其长度，用于调用 BLAS 函数
        ftcs3 = _cptofcd("U", strlen("U"));
        // 调用 BLAS 的 STRSM 函数解线性方程组
        STRSM( ftcs1, ftcs1, ftcs2, ftcs3, &nsupc, &nrhs, &alpha,
               &Lval[luptr], &nsupr, &Bmat[fsupc], &ldb);
        
        // 调用 BLAS 的 SGEMM 函数进行矩阵乘法
        SGEMM( ftcs2, ftcs2, &nrow, &nrhs, &nsupc, &alpha, 
            &Lval[luptr+nsupc], &nsupr, &Bmat[fsupc], &ldb, 
            &beta, &work[0], &n );
#else
        // 调用 BLAS 的 strsm_ 函数解线性方程组
        strsm_("L", "L", "N", "U", &nsupc, &nrhs, &alpha,
               &Lval[luptr], &nsupr, &Bmat[fsupc], &ldb);
        
        // 调用 BLAS 的 sgemm_ 函数进行矩阵乘法
        sgemm_( "N", "N", &nrow, &nrhs, &nsupc, &alpha, 
            &Lval[luptr+nsupc], &nsupr, &Bmat[fsupc], &ldb, 
            &beta, &work[0], &n );
#endif
        // 对解向量进行更新操作
        for (j = 0; j < nrhs; j++) {
            // 获取当前右手边向量的起始地址
            rhs_work = &Bmat[(size_t)j * (size_t)ldb];
            // 获取当前工作向量的起始地址
            work_col = &work[(size_t)j * (size_t)n];
            // 初始化行指针
            iptr = istart + nsupc;
            // 遍历当前处理的行
            for (i = 0; i < nrow; i++) {
                // 获取当前行对应的索引
                irow = L_SUB(iptr);
                // 在右手边向量中进行更新操作
                rhs_work[irow] -= work_col[i]; /* Scatter */
                // 将工作向量中的值置为零
                work_col[i] = 0.0;
                // 更新行指针
                iptr++;
            }
        }
#else        
        // 在不使用 BLAS 的情况下，执行下列操作
        for (j = 0; j < nrhs; j++) {
            // 获取当前右手边向量的起始地址
            rhs_work = &Bmat[(size_t)j * (size_t)ldb];
            // 调用自定义的函数解线性方程组
            slsolve (nsupr, nsupc, &Lval[luptr], &rhs_work[fsupc]);
            // 调用自定义的函数进行矩阵向量乘法
            smatvec (nsupr, nrow, nsupc, &Lval[luptr+nsupc],
                &rhs_work[fsupc], &work[0] );

            // 初始化行指针
            iptr = istart + nsupc;
            // 遍历当前处理的行
            for (i = 0; i < nrow; i++) {
                // 获取当前行对应的索引
                irow = L_SUB(iptr);
                // 在右手边向量中进行更新操作
                rhs_work[irow] -= work[i];
                // 将工作向量中的值置为零
                work[i] = 0.0;
                // 更新行指针
                iptr++;
            }
        }
#endif            
        } /* else ... */
    } /* for L-solve */

#if ( DEBUGlevel>=2 )
      // 打印调试信息，显示 L 分解后的解向量
      printf("After L-solve: y=\n");
    // 调用打印函数，显示解向量的详细信息
    sprint_soln(n, nrhs, Bmat);
#endif

    /*
     * Back solve Ux=y.
     */
    // 对 Ux=y 进行回代求解
    for (k = Lstore->nsuper; k >= 0; k--) {
        // 获取当前超节点的相关信息
        fsupc = L_FST_SUPC(k);
        istart = L_SUB_START(fsupc);
        nsupr = L_SUB_START(fsupc+1) - istart;
        nsupc = L_FST_SUPC(k+1) - fsupc;
        luptr = L_NZ_START(fsupc);

        // 统计解操作的次数
        solve_ops += nsupc * (nsupc + 1) * nrhs;

        // 处理超节点为单列情况
        if ( nsupc == 1 ) {
        // 获取当前右手边向量的起始地址
        rhs_work = &Bmat[0];
        // 遍历当前处理的右手边向量
        for (j = 0; j < nrhs; j++) {
            // 执行单列超节点的回代求解
            rhs_work[fsupc] /= Lval[luptr];
            // 更新右手边向量的地址
            rhs_work += ldb;
        }
        } else {
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
        // 将字符串 "L" 转换成 _fcd 类型，并计算其长度，用于调用 BLAS 函数
        ftcs1 = _cptofcd("L", strlen("L"));
        // 将字符串 "U" 转换成 _fcd 类型，并计算其长度，用于调用 BLAS 函数
        ftcs2 = _cptofcd("U", strlen("U"));
        // 将字符串 "N" 转换成 _fcd 类型，并计算其长度，用于调用 BLAS 函数
        ftcs3 = _cptofcd("N", strlen("N"));
        // 调用 BLAS 的 STRSM 函数解线性方程组
        STRSM( ftcs1, ftcs2, ftcs3, ftcs3, &nsupc, &nrhs, &alpha,
               &Lval[luptr], &nsupr, &Bmat[fsupc], &ldb);
#else
        // 调用 BLAS 的 strsm_ 函数解线性方程组
        strsm_("L", "U", "N", "N", &nsupc, &nrhs, &alpha,
               &Lval[luptr], &nsupr, &Bmat[fsupc], &ldb);
#endif
#else        
        // 在不使用 BLAS 的情况下，执行下列操作
        for (j = 0; j < nrhs; j++)
            // 调用自定义的函数执行超节点回代求解
            susolve ( nsupr, nsupc, &Lval[luptr], &Bmat[(size_t)fsupc + (size_t)j * (size_t)ldb] );
#endif
#endif        
        }

        // 遍历每个右侧向量
        for (j = 0; j < nrhs; ++j) {
            rhs_work = &Bmat[(size_t)j * (size_t)ldb];
            // 遍历当前列的所有行
            for (jcol = fsupc; jcol < fsupc + nsupc; jcol++) {
                // 计算解的操作数
                solve_ops += 2*(U_NZ_START(jcol+1) - U_NZ_START(jcol));
                // 遍历列jcol中的非零元素
                for (i = U_NZ_START(jcol); i < U_NZ_START(jcol+1); i++ ){
                    irow = U_SUB(i);
                    // 更新右侧向量
                    rhs_work[irow] -= rhs_work[jcol] * Uval[i];
                }
            }
        }
        
    } /* for U-solve */

#if ( DEBUGlevel>=2 )
      // 调试级别为2时打印解向量
    printf("After U-solve: x=\n");
    sprint_soln(n, nrhs, Bmat);
#endif

    /* 计算最终解 X := Pc*X */
    for (i = 0; i < nrhs; i++) {
        rhs_work = &Bmat[(size_t)i * (size_t)ldb];
        // 根据列置换更新解向量
        for (k = 0; k < n; k++) soln[k] = rhs_work[perm_c[k]];
        for (k = 0; k < n; k++) rhs_work[k] = soln[k];
    }
    
    // 设置求解操作数统计
    stat->ops[SOLVE] = solve_ops;

    } else { /* 解 A'*X=B 或 CONJ(A)*X=B */
    /* 置换右侧向量以形成 Pc'*B */
    for (i = 0; i < nrhs; i++) {
        rhs_work = &Bmat[(size_t)i * (size_t)ldb];
        // 根据列置换更新解向量
        for (k = 0; k < n; k++) soln[perm_c[k]] = rhs_work[k];
        for (k = 0; k < n; k++) rhs_work[k] = soln[k];
    }

    // 初始化求解操作数统计
    stat->ops[SOLVE] = 0;
    for (k = 0; k < nrhs; ++k) {
        
        /* 乘以 inv(U'). */
        sp_strsv("U", "T", "N", L, U, &Bmat[(size_t)k * (size_t)ldb], stat, info);
        
        /* 乘以 inv(L'). */
        sp_strsv("L", "T", "U", L, U, &Bmat[(size_t)k * (size_t)ldb], stat, info);
        
    }
    /* 计算最终解 X := Pr'*X (=inv(Pr)*X) */
    for (i = 0; i < nrhs; i++) {
        rhs_work = &Bmat[(size_t)i * (size_t)ldb];
        // 根据行置换更新解向量
        for (k = 0; k < n; k++) soln[k] = rhs_work[perm_r[k]];
        for (k = 0; k < n; k++) rhs_work[k] = soln[k];
    }

    }

    // 释放内存
    SUPERLU_FREE(work);
    SUPERLU_FREE(soln);
}

/*
 * 打印解向量的诊断输出
 */
void
sprint_soln(int n, int nrhs, float *soln)
{
    int i;

    // 遍历并打印解向量
    for (i = 0; i < n; i++)
      printf("\t%d: %.4f\n", i, soln[i]);
}
```