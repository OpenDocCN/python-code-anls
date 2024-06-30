# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\cgssv.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file cgssv.c
 * \brief Solves the system of linear equations A*X=B 
 *
 * <pre>
 * -- SuperLU routine (version 3.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * October 15, 2003
 * </pre>  
 */
#include "slu_cdefs.h"

void
cgssv(superlu_options_t *options, SuperMatrix *A, int *perm_c, int *perm_r,
      SuperMatrix *L, SuperMatrix *U, SuperMatrix *B,
      SuperLUStat_t *stat, int_t *info )
{
    DNformat *Bstore;
    SuperMatrix *AA;/* A in SLU_NC format used by the factorization routine.*/
    SuperMatrix AC; /* Matrix postmultiplied by Pc */
    int      lwork = 0, *etree, i;
    GlobalLU_t Glu; /* Not needed on return. */
    
    /* Set default values for some parameters */
    int      panel_size;     /* panel size */
    int      relax;          /* no of columns in a relaxed snodes */
    int      permc_spec;
    trans_t  trans = NOTRANS;
    double   *utime;
    double   t;    /* Temporary time */

    /* Test the input parameters ... */
    *info = 0;
    
    // Store the structure of matrix B
    Bstore = B->Store;

    // Check if factorization needs to be performed
    if ( options->Fact != DOFACT )
        *info = -1;
    else if ( A->nrow != A->ncol || A->nrow < 0 ||
              (A->Stype != SLU_NC && A->Stype != SLU_NR) ||
              A->Dtype != SLU_C || A->Mtype != SLU_GE )
        *info = -2;
    else if ( B->ncol < 0 || Bstore->lda < SUPERLU_MAX(0, A->nrow) ||
              B->Stype != SLU_DN || B->Dtype != SLU_C || B->Mtype != SLU_GE )
        *info = -7;

    // Handle input errors
    if ( *info != 0 ) {
        i = -(*info);
        input_error("cgssv", &i);
        return;
    }

    // Initialize utime with the pointer from stat
    utime = stat->utime;

    // Convert A to SLU_NC format when it's initially in SLU_NR format
    if ( A->Stype == SLU_NR ) {
        NRformat *Astore = A->Store;
        AA = (SuperMatrix *) SUPERLU_MALLOC( sizeof(SuperMatrix) );
        cCreate_CompCol_Matrix(AA, A->ncol, A->nrow, Astore->nnz, 
                       Astore->nzval, Astore->colind, Astore->rowptr,
                       SLU_NC, A->Dtype, A->Mtype);
        trans = TRANS;
    }
    // Use A directly if it's already in SLU_NC format
    else if ( A->Stype == SLU_NC ) {
        AA = A;
    }
    // Handle unsupported matrix format for A
    else {
        AA = NULL;
        *info = 1;
        input_error("cgssv", &i);
    }

    // Start timer for measuring the factorization time
    t = SuperLU_timer_();
    
    /*
     * Get column permutation vector perm_c[], according to permc_spec:
     *   permc_spec = NATURAL:  natural ordering 
     *   permc_spec = MMD_AT_PLUS_A: minimum degree on structure of A'+A
     *   permc_spec = MMD_ATA:  minimum degree on structure of A'*A
     *   permc_spec = COLAMD:   approximate minimum degree column ordering
     *   permc_spec = MY_PERMC: the ordering already supplied in perm_c[]
     */
    permc_spec = options->ColPerm;
    # 如果 permc_spec 不等于 MY_PERMC 并且 options 的 Fact 成员为 DOFACT，则进行列置换
    if ( permc_spec != MY_PERMC && options->Fact == DOFACT )
      # 调用函数 get_perm_c，获取列置换 perm_c
      get_perm_c(permc_spec, AA, perm_c);
    # 记录列置换时间
    utime[COLPERM] = SuperLU_timer_() - t;

    # 分配并初始化 etree 数组，用于存储列树结构
    etree = int32Malloc(A->ncol);

    # 记录 sp_preorder 函数的执行时间
    t = SuperLU_timer_();
    # 调用 sp_preorder 函数，计算并填充 etree 数组，同时生成新的稀疏矩阵 AC
    sp_preorder(options, AA, perm_c, etree, &AC);
    # 记录 etree 计算时间
    utime[ETREE] = SuperLU_timer_() - t;

    # 获取 panel_size 和 relax 参数的值
    panel_size = sp_ienv(1);
    relax = sp_ienv(2);

    /*
    printf("Factor PA = LU ... relax %d\tw %d\tmaxsuper %d\trowblk %d\n", 
      relax, panel_size, sp_ienv(3), sp_ienv(4));
    */

    # 记录开始计算 LU 分解的时间
    t = SuperLU_timer(); 
    /* Compute the LU factorization of A. */
    # 调用 cgstrf 函数进行 LU 分解
    cgstrf(options, &AC, relax, panel_size, etree,
            NULL, lwork, perm_c, perm_r, L, U, &Glu, stat, info);
    # 记录 LU 分解的时间
    utime[FACT] = SuperLU_timer() - t;

    # 记录开始解线性方程组的时间
    t = SuperLU_timer();
    # 如果 LU 分解成功（*info == 0），则调用 cgstrs 解线性方程组 A*X = B
    if ( *info == 0 ) {
        int info1;
        cgstrs (trans, L, U, perm_c, perm_r, B, stat, &info1);
    } else {
        # 若 LU 分解失败，则输出错误信息
        printf("cgstrf info %lld\n", (long long) *info); fflush(stdout);
    }
    
    # 记录解线性方程组的时间
    utime[SOLVE] = SuperLU_timer() - t;

    # 释放 etree 所占用的内存
    SUPERLU_FREE (etree);
    # 销毁 CompCol_Permuted 结构体 AC
    Destroy_CompCol_Permuted(&AC);
    # 如果 A 的存储类型为 SLU_NR，释放 AA 所占用的内存
    if ( A->Stype == SLU_NR ) {
        Destroy_SuperMatrix_Store(AA);
        SUPERLU_FREE(AA);
    }
}



# 这行代码是一个单独的闭合大括号（}），用于结束一个代码块或函数。
```