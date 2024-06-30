# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zgssv.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file zgssv.c
 * \brief Solves the system of linear equations A*X=B 
 *
 * <pre>
 * -- SuperLU routine (version 3.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * October 15, 2003
 * </pre>  
 */
#include "slu_zdefs.h"

void
zgssv(superlu_options_t *options, SuperMatrix *A, int *perm_c, int *perm_r,
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
    int      permc_spec;     /* column permutation specifier */
    trans_t  trans = NOTRANS; /* transposition type */
    double   *utime;         /* array to store timing */
    double   t;              /* Temporary time */

    /* Test the input parameters ... */
    *info = 0;
    Bstore = B->Store;
    /* Check if factorization is needed */
    if ( options->Fact != DOFACT ) *info = -1;
    /* Check matrix A properties */
    else if ( A->nrow != A->ncol || A->nrow < 0 ||
             (A->Stype != SLU_NC && A->Stype != SLU_NR) ||
             A->Dtype != SLU_Z || A->Mtype != SLU_GE )
        *info = -2;
    /* Check matrix B properties */
    else if ( B->ncol < 0 || Bstore->lda < SUPERLU_MAX(0, A->nrow) ||
              B->Stype != SLU_DN || B->Dtype != SLU_Z || B->Mtype != SLU_GE )
        *info = -7;

    /* Handle input errors */
    if ( *info != 0 ) {
        i = -(*info);
        input_error("zgssv", &i);
        return;
    }

    utime = stat->utime;

    /* Convert A to SLU_NC format when necessary. */
    if ( A->Stype == SLU_NR ) {
        NRformat *Astore = A->Store;
        AA = (SuperMatrix *) SUPERLU_MALLOC( sizeof(SuperMatrix) );
        zCreate_CompCol_Matrix(AA, A->ncol, A->nrow, Astore->nnz, 
                       Astore->nzval, Astore->colind, Astore->rowptr,
                       SLU_NC, A->Dtype, A->Mtype);
        trans = TRANS;
    } else if ( A->Stype == SLU_NC ) {
        AA = A;
    }
    /* A is of unsupported matrix format. */
    else {
        AA = NULL;
        *info = 1;
        input_error("zgssv", &i);
    }

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
    // 如果 permc_spec 不等于 MY_PERMC 并且 options->Fact 等于 DOFACT，则获取列置换 perm_c
    if ( permc_spec != MY_PERMC && options->Fact == DOFACT )
      get_perm_c(permc_spec, AA, perm_c);
    // 计算 COLPERM 阶段的时间消耗
    utime[COLPERM] = SuperLU_timer_() - t;

    // 分配内存以存储列的父结点树
    etree = int32Malloc(A->ncol);

    t = SuperLU_timer_();
    // 对矩阵 AA 进行预排序，生成父结点树 etree，并更新 AC 矩阵
    sp_preorder(options, AA, perm_c, etree, &AC);
    // 计算 ETREE 阶段的时间消耗
    utime[ETREE] = SuperLU_timer_() - t;

    // 获取面板的大小和松弛因子
    panel_size = sp_ienv(1);
    relax = sp_ienv(2);

    /*printf("Factor PA = LU ... relax %d\tw %d\tmaxsuper %d\trowblk %d\n", 
      relax, panel_size, sp_ienv(3), sp_ienv(4));*/
    t = SuperLU_timer(); 
    // 计算矩阵 A 的 LU 分解
    /* Compute the LU factorization of A. */
    zgstrf(options, &AC, relax, panel_size, etree,
            NULL, lwork, perm_c, perm_r, L, U, &Glu, stat, info);
    // 计算 FACT 阶段的时间消耗
    utime[FACT] = SuperLU_timer_() - t;

    t = SuperLU_timer();
    if ( *info == 0 ) {
        // 如果 LU 分解成功，解线性方程组 A*X=B，结果存储在 B 中
        /* Solve the system A*X=B, overwriting B with X. */
        int info1;
        zgstrs (trans, L, U, perm_c, perm_r, B, stat, &info1);
    } else {
        // 如果 LU 分解失败，打印错误信息
        printf("zgstrf info %lld\n", (long long) *info); fflush(stdout);
    }
    
    // 计算 SOLVE 阶段的时间消耗
    utime[SOLVE] = SuperLU_timer() - t;

    // 释放 etree 所占用的内存
    SUPERLU_FREE (etree);
    // 销毁并释放 AC 的存储空间
    Destroy_CompCol_Permuted(&AC);
    // 如果 A 的存储类型为 SLU_NR，销毁并释放 AA 的存储空间
    if ( A->Stype == SLU_NR ) {
        Destroy_SuperMatrix_Store(AA);
        SUPERLU_FREE(AA);
    }
}


注释：


# 这行代码关闭了一个代码块。在大多数编程语言中，使用 '}' 来结束一个代码块，通常与 '{' 配对使用。
```