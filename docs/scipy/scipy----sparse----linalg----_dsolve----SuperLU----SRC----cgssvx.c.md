# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\cgssvx.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file cgssvx.c
 * \brief Solves the system of linear equations A*X=B or A'*X=B
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
cgssvx(superlu_options_t *options, SuperMatrix *A, int *perm_c, int *perm_r,
       int *etree, char *equed, float *R, float *C,
       SuperMatrix *L, SuperMatrix *U, void *work, int_t lwork,
       SuperMatrix *B, SuperMatrix *X, float *recip_pivot_growth, 
       float *rcond, float *ferr, float *berr, 
       GlobalLU_t *Glu, mem_usage_t *mem_usage, SuperLUStat_t *stat, int_t *info )
{
    DNformat  *Bstore, *Xstore;
    singlecomplex    *Bmat, *Xmat;
    int       ldb, ldx, nrhs;
    SuperMatrix *AA;/* A in SLU_NC format used by the factorization routine.*/
    SuperMatrix AC; /* Matrix postmultiplied by Pc */
    int       colequ, equil, nofact, notran, rowequ, permc_spec;
    trans_t   trant;
    char      norm[1];
    int       i, j, info1;
    float    amax, anorm, bignum, smlnum, colcnd, rowcnd, rcmax, rcmin;
    int       relax, panel_size;
    double    t0;      /* temporary time */
    double    *utime;

    /* External functions */
    extern float clangs(char *, SuperMatrix *);

    // 获取输入矩阵 B 和 X 的存储格式信息
    Bstore = B->Store;
    Xstore = X->Store;
    Bmat   = Bstore->nzval;
    Xmat   = Xstore->nzval;
    ldb    = Bstore->lda;
    ldx    = Xstore->lda;
    nrhs   = B->ncol;

    // 初始化 info 为 0，设置选项标志位
    *info = 0;
    nofact = (options->Fact != FACTORED);
    equil = (options->Equil == YES);
    notran = (options->Trans == NOTRANS);

    // 如果没有进行因式分解，则设置 equed 为 'N'，并且不进行行和列均衡
    if ( nofact ) {
        *(unsigned char *)equed = 'N';
        rowequ = FALSE;
        colequ = FALSE;
    } else {
        // 根据 equed 参数判断是否进行行和列均衡
        rowequ = strncmp(equed, "R", 1)==0 || strncmp(equed, "B", 1)==0;
        colequ = strncmp(equed, "C", 1)==0 || strncmp(equed, "B", 1)==0;
        // 获取安全最小值和其倒数，用于计算 bignum
        smlnum = smach("Safe minimum");
        bignum = 1. / smlnum;
    }

#if 0
printf("dgssvx: Fact=%4d, Trans=%4d, equed=%c\n",
       options->Fact, options->Trans, *equed);
#endif

    // 检查输入参数的合法性
    if (options->Fact != DOFACT && options->Fact != SamePattern &&
        options->Fact != SamePattern_SameRowPerm &&
        options->Fact != FACTORED &&
        options->Trans != NOTRANS && options->Trans != TRANS && 
        options->Trans != CONJ &&
        options->Equil != NO && options->Equil != YES)
    {
        *info = -1;
    }
    else if ( A->nrow != A->ncol || A->nrow < 0 ||
              (A->Stype != SLU_NC && A->Stype != SLU_NR) ||
              A->Dtype != SLU_C || A->Mtype != SLU_GE )
    {
        *info = -2;
    }
    else if ( options->Fact == FACTORED &&
         !(rowequ || colequ || strncmp(equed, "N", 1)==0) )
    /* Check if matrix is factored and not equilibrated */
    *info = -6;
    else {
    /* Begin checking row scaling */
    if (rowequ) {
        rcmin = bignum;
        rcmax = 0.;
        for (j = 0; j < A->nrow; ++j) {
        rcmin = SUPERLU_MIN(rcmin, R[j]);
        rcmax = SUPERLU_MAX(rcmax, R[j]);
        }
        /* Check minimum row scaling */
        if (rcmin <= 0.) *info = -7;
        /* Compute row condition number */
        else if ( A->nrow > 0)
        rowcnd = SUPERLU_MAX(rcmin,smlnum) / SUPERLU_MIN(rcmax,bignum);
        else rowcnd = 1.;
    }
    /* Begin checking column scaling */
    if (colequ && *info == 0) {
        rcmin = bignum;
        rcmax = 0.;
        for (j = 0; j < A->nrow; ++j) {
        rcmin = SUPERLU_MIN(rcmin, C[j]);
        rcmax = SUPERLU_MAX(rcmax, C[j]);
        }
        /* Check minimum column scaling */
        if (rcmin <= 0.) *info = -8;
        /* Compute column condition number */
        else if (A->nrow > 0)
        colcnd = SUPERLU_MAX(rcmin,smlnum) / SUPERLU_MIN(rcmax,bignum);
        else colcnd = 1.;
    }
    /* Perform additional checks */
    if (*info == 0) {
        /* Check if lwork is less than -1 */
        if ( lwork < -1 ) *info = -12;
        /* Check if B->ncol is negative */
        else if ( B->ncol < 0 ) *info = -13;
        else if ( B->ncol > 0 ) { /* Perform detailed checks if B->ncol > 0 */
             if ( Bstore->lda < SUPERLU_MAX(0, A->nrow) ||
              B->Stype != SLU_DN || B->Dtype != SLU_C || 
              B->Mtype != SLU_GE )
        *info = -13;
            }
        /* Check if X->ncol is negative */
        if ( X->ncol < 0 ) *info = -14;
            else if ( X->ncol > 0 ) { /* Perform detailed checks if X->ncol > 0 */
                 if ( Xstore->lda < SUPERLU_MAX(0, A->nrow) ||
              (B->ncol != 0 && B->ncol != X->ncol) ||
                      X->Stype != SLU_DN ||
              X->Dtype != SLU_C || X->Mtype != SLU_GE )
        *info = -14;
            }
    }
    }
    if (*info != 0) {
    i = -(*info);
    input_error("cgssvx", &i);
    return;
    }
    
    /* Initialization for factor parameters */
    panel_size = sp_ienv(1);
    relax      = sp_ienv(2);

    utime = stat->utime;
    
    /* Convert A to SLU_NC format when necessary. */
    if ( A->Stype == SLU_NR ) {
    NRformat *Astore = A->Store;
    AA = (SuperMatrix *) SUPERLU_MALLOC( sizeof(SuperMatrix) );
    /* Create a compressed column matrix AA */
    cCreate_CompCol_Matrix(AA, A->ncol, A->nrow, Astore->nnz, 
                   Astore->nzval, Astore->colind, Astore->rowptr,
                   SLU_NC, A->Dtype, A->Mtype);
    if ( notran ) { /* Reverse the transpose argument. */
        trant = TRANS;
        notran = 0;
    } else {
        trant = NOTRANS;
        notran = 1;
    }
    } else { /* A->Stype == SLU_NC */
    /* Use the original format for AA */
    trant = options->Trans;
    AA = A;
    }

    if ( nofact && equil ) {
    t0 = SuperLU_timer_();
    /* Compute row and column scalings to equilibrate the matrix A. */
    cgsequ(AA, R, C, &rowcnd, &colcnd, &amax, &info1);
    
    if ( info1 == 0 ) {
        /* Equilibrate matrix A. */
        claqgs(AA, R, C, rowcnd, colcnd, amax, equed);
        /* Check if equilibration should be applied to rows or columns */
        rowequ = strncmp(equed, "R", 1)==0 || strncmp(equed, "B", 1)==0;
        colequ = strncmp(equed, "C", 1)==0 || strncmp(equed, "B", 1)==0;
    }
    /* Measure the time spent on equilibration */
    utime[EQUIL] = SuperLU_timer_() - t0;
    }
    // 如果 nofact 为真，则执行以下代码块
    if ( nofact ) {
    
        // 记录当前时间到 t0，用于计时
        t0 = SuperLU_timer_();
    /*
     * 根据 permc_spec 对 Gnet 列置换向量 perm_c[] 进行排序:
     *   permc_spec = NATURAL:  自然顺序
     *   permc_spec = MMD_AT_PLUS_A: A'+A 结构的最小度排序
     *   permc_spec = MMD_ATA:  A'*A 结构的最小度排序
     *   permc_spec = COLAMD:   近似最小度列排序
     *   permc_spec = MY_PERMC: 已提供的 perm_c[] 中的排序
     */
    permc_spec = options->ColPerm;
    // 如果 permc_spec 不是 MY_PERMC，并且 options->Fact 为 DOFACT，则获取列置换 perm_c
    if ( permc_spec != MY_PERMC && options->Fact == DOFACT )
            get_perm_c(permc_spec, AA, perm_c);
    // 计算列置换时间并存入 utime 数组的 COLPERM 索引处
    utime[COLPERM] = SuperLU_timer_() - t0;

    // 记录当前时间到 t0，用于计时
    t0 = SuperLU_timer_();
    // 进行预排序操作，生成 AC 稀疏矩阵
    sp_preorder(options, AA, perm_c, etree, &AC);
    // 计算生成 etree 所用时间并存入 utime 数组的 ETREE 索引处
    utime[ETREE] = SuperLU_timer_() - t0;
/* 打印调试信息，输出 Factor PA = LU ... relax %d\tw %d\tmaxsuper %d\trowblk %d\n 的格式化字符串
   并刷新标准输出流 */
printf("Factor PA = LU ... relax %d\tw %d\tmaxsuper %d\trowblk %d\n", 
       relax, panel_size, sp_ienv(3), sp_ienv(4));
fflush(stdout);

/* 计算矩阵 A*Pc 的 LU 分解 */
t0 = SuperLU_timer_();
cgstrf(options, &AC, relax, panel_size, etree,
            work, lwork, perm_c, perm_r, L, U, Glu, stat, info);
utime[FACT] = SuperLU_timer_() - t0;

/* 如果 lwork 为 -1，计算内存使用情况并返回 */
if (lwork == -1) {
    mem_usage->total_needed = *info - A->ncol;
    return;
}

/* 如果 info 大于 0 */
if (*info > 0) {
    if (*info <= A->ncol) {
        /* 计算矩阵 A 前 *info 列的逆枢轴增长因子 */
        *recip_pivot_growth = cPivotGrowth(*info, AA, perm_c, L, U);
    }
    return;
}

/* 在这一点上 *info == 0 */

if (options->PivotGrowth) {
    /* 计算逆枢轴增长因子 *recip_pivot_growth */
    *recip_pivot_growth = cPivotGrowth(A->ncol, AA, perm_c, L, U);
}

if (options->ConditionNumber) {
    /* 估算矩阵 A 的条件数的倒数 */
    t0 = SuperLU_timer_();
    /* 根据 notran 的值设置 norm 参数 */
    if (notran) {
        *(unsigned char *)norm = '1';
    } else {
        *(unsigned char *)norm = 'I';
    }
    /* 计算 AA 矩阵的 norm */
    anorm = clangs(norm, AA);
    /* 计算条件数的倒数 */
    cgscon(norm, L, U, anorm, rcond, stat, &info1);
    utime[RCOND] = SuperLU_timer_() - t0;
}
    if ( nrhs > 0 ) {
        /* 如果右手边的向量个数大于0，则进行下面的操作 */

        /* 如果没有转置 */
        if ( notran ) {
            /* 如果行均衡 */
            if ( rowequ ) {
                /* 对每个右手边向量进行操作 */
                for (j = 0; j < nrhs; ++j)
                    /* 对矩阵 A 的每一行进行操作 */
                    for (i = 0; i < A->nrow; ++i)
                        cs_mult(&Bmat[i+j*ldb], &Bmat[i+j*ldb], R[i]);
            }
        } else if ( colequ ) { /* 如果列均衡 */
            /* 对每个右手边向量进行操作 */
            for (j = 0; j < nrhs; ++j)
                /* 对矩阵 A 的每一行进行操作 */
                for (i = 0; i < A->nrow; ++i)
                    cs_mult(&Bmat[i+j*ldb], &Bmat[i+j*ldb], C[i]);
        }

        /* 计算解矩阵 X */
        for (j = 0; j < nrhs; j++)  /* 保存右手边的副本 */
            for (i = 0; i < B->nrow; i++)
                Xmat[i + j*ldx] = Bmat[i + j*ldb];
    
        /* 计时开始 */
        t0 = SuperLU_timer_();
        /* 调用线性方程组求解函数 */
        cgstrs (trant, L, U, perm_c, perm_r, X, stat, &info1);
        /* 计算用时 */
        utime[SOLVE] = SuperLU_timer_() - t0;
    
        /* 使用迭代精细化提高计算的解，并计算误差界限和后向误差估计 */
        t0 = SuperLU_timer_();
        if ( options->IterRefine != NOREFINE ) {
            cgsrfs(trant, AA, L, U, perm_c, perm_r, equed, R, C, B,
                   X, ferr, berr, stat, &info1);
        } else {
            for (j = 0; j < nrhs; ++j) ferr[j] = berr[j] = 1.0;
        }
        /* 计算用时 */
        utime[REFINE] = SuperLU_timer_() - t0;

        /* 如果没有转置 */
        if ( notran ) {
            /* 如果列均衡 */
            if ( colequ ) {
                /* 对每个右手边向量进行操作 */
                for (j = 0; j < nrhs; ++j)
                    /* 对矩阵 A 的每一行进行操作 */
                    for (i = 0; i < A->nrow; ++i)
                        cs_mult(&Xmat[i+j*ldx], &Xmat[i+j*ldx], C[i]);
            }
        } else if ( rowequ ) { /* 如果行均衡 */
            /* 对每个右手边向量进行操作 */
            for (j = 0; j < nrhs; ++j)
                /* 对矩阵 A 的每一行进行操作 */
                for (i = 0; i < A->nrow; ++i)
                    cs_mult(&Xmat[i+j*ldx], &Xmat[i+j*ldx], R[i]);
        }
    } /* end if nrhs > 0 */

    /* 如果需要计算条件数 */
    if ( options->ConditionNumber ) {
        /* 如果条件数小于某个阈值，将 info 设置为 A 列数加 1 表示矩阵在工作精度下奇异 */
        if ( *rcond < smach("E") ) *info = A->ncol + 1;
    }

    /* 如果没有进行因式分解 */
    if ( nofact ) {
        /* 查询空间需求 */
        cQuerySpace(L, U, mem_usage);
        /* 销毁压缩列存储的超级矩阵 AC */
        Destroy_CompCol_Permuted(&AC);
    }
    /* 如果 A 的存储类型是非重叠右上角存储 */
    if ( A->Stype == SLU_NR ) {
        /* 销毁超级矩阵 AA */
        Destroy_SuperMatrix_Store(AA);
        /* 释放 AA 占用的内存 */
        SUPERLU_FREE(AA);
    }
}



# 这行代码表示一个代码块的结束，与之前的 "{" 配对，用于结束一个函数、循环或条件语句的定义或执行。
```