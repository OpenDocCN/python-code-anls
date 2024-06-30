# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\sgssv.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file sgssv.c
 * \brief Solves the system of linear equations A*X=B 
 *
 * <pre>
 * -- SuperLU routine (version 3.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * October 15, 2003
 * </pre>  
 */
#include "slu_sdefs.h"

void
sgssv(superlu_options_t *options, SuperMatrix *A, int *perm_c, int *perm_r,
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
    
    // 检查 B 对象的存储格式和 options 中的 Factorization 选项
    Bstore = B->Store;
    if ( options->Fact != DOFACT ) *info = -1;
    else if ( A->nrow != A->ncol || A->nrow < 0 ||
             (A->Stype != SLU_NC && A->Stype != SLU_NR) ||
             A->Dtype != SLU_S || A->Mtype != SLU_GE )
        *info = -2;
    else if ( B->ncol < 0 || Bstore->lda < SUPERLU_MAX(0, A->nrow) ||
              B->Stype != SLU_DN || B->Dtype != SLU_S || B->Mtype != SLU_GE )
        *info = -7;
    
    // 如果参数有误，报告错误并返回
    if ( *info != 0 ) {
        i = -(*info);
        input_error("sgssv", &i);
        return;
    }

    // 获取统计信息中的运行时间数组
    utime = stat->utime;

    /* Convert A to SLU_NC format when necessary. */
    if ( A->Stype == SLU_NR ) {
        // 将 A 转换为 SLU_NC 格式
        NRformat *Astore = A->Store;
        AA = (SuperMatrix *) SUPERLU_MALLOC( sizeof(SuperMatrix) );
        sCreate_CompCol_Matrix(AA, A->ncol, A->nrow, Astore->nnz, 
                               Astore->nzval, Astore->colind, Astore->rowptr,
                               SLU_NC, A->Dtype, A->Mtype);
        trans = TRANS;
    } else if ( A->Stype == SLU_NC ) {
        // A 已经是 SLU_NC 格式
        AA = A;
    }
    // 不支持的矩阵格式
    else {
        AA = NULL;
        *info = 1;
        input_error("sgssv", &i);
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
    // 根据 options 中的列置换选项选择列置换算法
    permc_spec = options->ColPerm;
    // 如果列排列参数不等于 MY_PERMC，并且选项中的 Fact 等于 DOFACT
    if ( permc_spec != MY_PERMC && options->Fact == DOFACT )
      // 获取列排列
      get_perm_c(permc_spec, AA, perm_c);
    // 计算列排列时间
    utime[COLPERM] = SuperLU_timer_() - t;

    // 分配并初始化 etree 数组，用于存储列数
    etree = int32Malloc(A->ncol);

    // 记录当前时间
    t = SuperLU_timer_();
    // 进行预排序操作，生成 etree 并返回重排序的矩阵 AC
    sp_preorder(options, AA, perm_c, etree, &AC);
    // 计算预排序时间
    utime[ETREE] = SuperLU_timer_() - t;

    // 获取面板大小和松弛因子
    panel_size = sp_ienv(1);
    relax = sp_ienv(2);

    /*printf("Factor PA = LU ... relax %d\tw %d\tmaxsuper %d\trowblk %d\n", 
      relax, panel_size, sp_ienv(3), sp_ienv(4));*/
    // 记录当前时间
    t = SuperLU_timer_();
    /* 计算矩阵 A 的 LU 分解。*/
    sgstrf(options, &AC, relax, panel_size, etree,
            NULL, lwork, perm_c, perm_r, L, U, &Glu, stat, info);
    // 计算 LU 分解时间
    utime[FACT] = SuperLU_timer_() - t;

    // 记录当前时间
    t = SuperLU_timer_();
    // 如果 LU 分解成功
    if ( *info == 0 ) {
        /* 解方程组 A*X=B，结果保存在 B 中。 */
        int info1;
        sgstrs (trans, L, U, perm_c, perm_r, B, stat, &info1);
    } else {
        // 输出错误信息，显示 sgstrf 返回的错误码
        printf("sgstrf info %lld\n", (long long) *info); fflush(stdout);
    }
    
    // 计算解方程组时间
    utime[SOLVE] = SuperLU_timer_() - t;

    // 释放 etree 的内存空间
    SUPERLU_FREE (etree);
    // 销毁被置换的压缩列矩阵 AC
    Destroy_CompCol_Permuted(&AC);
    // 如果 A 的存储类型为 SLU_NR，释放 AA 的存储空间
    if ( A->Stype == SLU_NR ) {
        Destroy_SuperMatrix_Store(AA);
        SUPERLU_FREE(AA);
    }
}



# 这行代码是一个单独的右大括号 '}'，用于结束一个代码块或语句。在此处的上下文中，它应该与其他代码一起出现，但在提供的片段中它是一个孤立的符号，可能是由于格式化错误或代码截断而导致的。
```