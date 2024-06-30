# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dgssvx.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file dgssvx.c
 * \brief Solves the system of linear equations A*X=B or A'*X=B
 *
 * <pre>
 * -- SuperLU routine (version 3.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * October 15, 2003
 * </pre>
 */
#include "slu_ddefs.h"

/**
 * \brief Solves a system of linear equations using SuperLU.
 *
 * \param options SuperLU options structure controlling factorization and solving.
 * \param A Pointer to the coefficient matrix A.
 * \param perm_c Column permutation vector.
 * \param perm_r Row permutation vector.
 * \param etree Elimination tree of A'*A or A*A'.
 * \param equed Specifies if and how the system should be equilibrated.
 * \param R Row scale factors for equilibration.
 * \param C Column scale factors for equilibration.
 * \param L Pointer to the factor L from the factorization of A.
 * \param U Pointer to the factor U from the factorization of A.
 * \param work Workspace array.
 * \param lwork Size of the workspace array.
 * \param B Right-hand side matrix B.
 * \param X Solution matrix X.
 * \param recip_pivot_growth Reciprocal pivot growth.
 * \param rcond Reciprocal condition number of the matrix A.
 * \param ferr Forward error bounds.
 * \param berr Backward error bounds.
 * \param Glu Global LU workspace.
 * \param mem_usage Memory usage statistics.
 * \param stat SuperLU statistics.
 * \param info Output status (0 if successful).
 */
void
dgssvx(superlu_options_t *options, SuperMatrix *A, int *perm_c, int *perm_r,
       int *etree, char *equed, double *R, double *C,
       SuperMatrix *L, SuperMatrix *U, void *work, int_t lwork,
       SuperMatrix *B, SuperMatrix *X, double *recip_pivot_growth, 
       double *rcond, double *ferr, double *berr, 
       GlobalLU_t *Glu, mem_usage_t *mem_usage, SuperLUStat_t *stat, int_t *info )
{

    DNformat  *Bstore, *Xstore;
    double    *Bmat, *Xmat;
    int       ldb, ldx, nrhs;
    SuperMatrix *AA;/* A in SLU_NC format used by the factorization routine.*/
    SuperMatrix AC; /* Matrix postmultiplied by Pc */
    int       colequ, equil, nofact, notran, rowequ, permc_spec;
    trans_t   trant;
    char      norm[1];
    int       i, j, info1;
    double    amax, anorm, bignum, smlnum, colcnd, rowcnd, rcmax, rcmin;
    int       relax, panel_size;
    double    t0;      /* temporary time */
    double    *utime;

    /* External functions */
    extern double dlangs(char *, SuperMatrix *);

    // Extract necessary information from the input matrices B and X
    Bstore = B->Store;
    Xstore = X->Store;
    Bmat   = Bstore->nzval;
    Xmat   = Xstore->nzval;
    ldb    = Bstore->lda;
    ldx    = Xstore->lda;
    nrhs   = B->ncol;

    // Initialize info to 0 (no errors yet)
    *info = 0;

    // Determine if factorization has been performed or not
    nofact = (options->Fact != FACTORED);
    equil = (options->Equil == YES);
    notran = (options->Trans == NOTRANS);

    // If no factorization performed, set equed to 'N' and disable row and column equilibration
    if ( nofact ) {
        *(unsigned char *)equed = 'N';
        rowequ = FALSE;
        colequ = FALSE;
    } else {
        // Determine if row or column equilibration was applied based on equed value
        rowequ = strncmp(equed, "R", 1)==0 || strncmp(equed, "B", 1)==0;
        colequ = strncmp(equed, "C", 1)==0 || strncmp(equed, "B", 1)==0;

        // Set up small and big numbers for numerical scaling
        smlnum = dmach("Safe minimum");   /* lamch_("Safe minimum"); */
        bignum = 1. / smlnum;
    }

    // Test the input parameters for validity
    if (options->Fact != DOFACT && options->Fact != SamePattern &&
        options->Fact != SamePattern_SameRowPerm &&
        options->Fact != FACTORED &&
        options->Trans != NOTRANS && options->Trans != TRANS && 
        options->Trans != CONJ &&
        options->Equil != NO && options->Equil != YES)
    {
        *info = -1; // Invalid input parameters
    }
    else if ( A->nrow != A->ncol || A->nrow < 0 ||
              (A->Stype != SLU_NC && A->Stype != SLU_NR) ||
              A->Dtype != SLU_D || A->Mtype != SLU_GE )
    {
        *info = -2; // Invalid matrix A
    }

    // Additional code would follow for solving the linear system using SuperLU
}
    else if ( options->Fact == FACTORED &&
         !(rowequ || colequ || strncmp(equed, "N", 1)==0) )
    *info = -6;
    else {
    // 如果需要因子分解，并且未对行或列进行等价化，并且未对矩阵A进行等价化
    *info = -6;
    }
    if (rowequ) {
        rcmin = bignum;
        rcmax = 0.;
        for (j = 0; j < A->nrow; ++j) {
        // 计算行最小值和最大值
        rcmin = SUPERLU_MIN(rcmin, R[j]);
        rcmax = SUPERLU_MAX(rcmax, R[j]);
        }
        if (rcmin <= 0.) *info = -7;
        else if ( A->nrow > 0)
        // 计算行条件数
        rowcnd = SUPERLU_MAX(rcmin,smlnum) / SUPERLU_MIN(rcmax,bignum);
        else rowcnd = 1.;
    }
    if (colequ && *info == 0) {
        rcmin = bignum;
        rcmax = 0.;
        for (j = 0; j < A->nrow; ++j) {
        // 计算列最小值和最大值
        rcmin = SUPERLU_MIN(rcmin, C[j]);
        rcmax = SUPERLU_MAX(rcmax, C[j]);
        }
        if (rcmin <= 0.) *info = -8;
        else if (A->nrow > 0)
        // 计算列条件数
        colcnd = SUPERLU_MAX(rcmin,smlnum) / SUPERLU_MIN(rcmax,bignum);
        else colcnd = 1.;
    }
    if (*info == 0) {
        if ( lwork < -1 ) *info = -12;
        else if ( B->ncol < 0 ) *info = -13;
        else if ( B->ncol > 0 ) { /* B->ncol不为0时进行检查 */
             if ( Bstore->lda < SUPERLU_MAX(0, A->nrow) ||
              B->Stype != SLU_DN || B->Dtype != SLU_D || 
              B->Mtype != SLU_GE )
        *info = -13;
            }
        if ( X->ncol < 0 ) *info = -14;
            else if ( X->ncol > 0 ) { /* X->ncol不为0时进行检查 */
                 if ( Xstore->lda < SUPERLU_MAX(0, A->nrow) ||
              (B->ncol != 0 && B->ncol != X->ncol) ||
                      X->Stype != SLU_DN ||
              X->Dtype != SLU_D || X->Mtype != SLU_GE )
        *info = -14;
            }
    }
    }
    if (*info != 0) {
    i = -(*info);
    input_error("dgssvx", &i);
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
    dCreate_CompCol_Matrix(AA, A->ncol, A->nrow, Astore->nnz, 
                   Astore->nzval, Astore->colind, Astore->rowptr,
                   SLU_NC, A->Dtype, A->Mtype);
    if ( notran ) { /* 反转转置参数 */
        trant = TRANS;
        notran = 0;
    } else {
        trant = NOTRANS;
        notran = 1;
    }
    } else { /* A->Stype == SLU_NC */
    trant = options->Trans;
    AA = A;
    }

    if ( nofact && equil ) {
    t0 = SuperLU_timer_();
    /* 计算行和列的缩放以使矩阵A等价化 */
    dgsequ(AA, R, C, &rowcnd, &colcnd, &amax, &info1);
    
    if ( info1 == 0 ) {
        /* 对矩阵A进行等价化 */
        dlaqgs(AA, R, C, rowcnd, colcnd, amax, equed);
        // 判断是否对行进行了等价化
        rowequ = strncmp(equed, "R", 1)==0 || strncmp(equed, "B", 1)==0;
        // 判断是否对列进行了等价化
        colequ = strncmp(equed, "C", 1)==0 || strncmp(equed, "B", 1)==0;
    }
    utime[EQUIL] = SuperLU_timer_() - t0;
    }
    // 如果 nofact 为真，则进行下面的操作
    if ( nofact ) {
    
        // 记录当前时间到 t0，用于计时
        t0 = SuperLU_timer_();
    /*
     * 根据 permc_spec 对 Gnet 的列置换向量 perm_c[] 进行排列：
     *   permc_spec = NATURAL: 自然顺序
     *   permc_spec = MMD_AT_PLUS_A: A'+A 结构的最小度排列
     *   permc_spec = MMD_ATA: A'*A 结构的最小度排列
     *   permc_spec = COLAMD: 近似最小度列排序
     *   permc_spec = MY_PERMC: 已经在 perm_c[] 中指定的排序
     */
    permc_spec = options->ColPerm;
    // 如果 permc_spec 不是 MY_PERMC 并且 options->Fact 是 DOFACT，调用 get_perm_c 函数计算 perm_c
    if ( permc_spec != MY_PERMC && options->Fact == DOFACT )
            get_perm_c(permc_spec, AA, perm_c);
    // 计算列置换时间并保存到 utime 数组中
    utime[COLPERM] = SuperLU_timer_() - t0;

    // 记录当前时间到 t0，用于计时
    t0 = SuperLU_timer_();
    // 对稀疏矩阵进行预排序，生成 AC，同时计算生成 etree 所需的时间
    sp_preorder(options, AA, perm_c, etree, &AC);
    // 计算 etree 生成时间并保存到 utime 数组中
    utime[ETREE] = SuperLU_timer_() - t0;
    }
/*    printf("Factor PA = LU ... relax %d\tw %d\tmaxsuper %d\trowblk %d\n", 
           relax, panel_size, sp_ienv(3), sp_ienv(4));
    fflush(stdout); */
    
    /* 上述代码段被注释掉，原本用于打印并刷新标准输出，显示因子分解的参数设置 */

    
    /* 计算 A*Pc 的LU因子分解。*/
    t0 = SuperLU_timer_();
    dgstrf(options, &AC, relax, panel_size, etree,
                work, lwork, perm_c, perm_r, L, U, Glu, stat, info);
    utime[FACT] = SuperLU_timer_() - t0;
    
    if ( lwork == -1 ) {
        mem_usage->total_needed = *info - A->ncol;
        return;
    }
    }

    if ( *info > 0 ) {
        if ( *info <= A->ncol ) {
        /* 计算 A 的前 *info 列的逆枢轴增长因子。 */
        *recip_pivot_growth = dPivotGrowth(*info, AA, perm_c, L, U);
        }
    return;
    }

    /* *info == 0 时执行以下代码。 */

    if ( options->PivotGrowth ) {
        /* 计算逆枢轴增长因子 *recip_pivot_growth。 */
        *recip_pivot_growth = dPivotGrowth(A->ncol, AA, perm_c, L, U);
    }

    if ( options->ConditionNumber ) {
        /* 估算矩阵 A 的条件数的逆。 */
        t0 = SuperLU_timer_();
        if ( notran ) {
        *(unsigned char *)norm = '1';
        } else {
        *(unsigned char *)norm = 'I';
        }
        anorm = dlangs(norm, AA);
        dgscon(norm, L, U, anorm, rcond, stat, &info1);
        utime[RCOND] = SuperLU_timer_() - t0;
    }
    if ( nrhs > 0 ) {
        /* 如果右手边的数量大于零，则执行以下操作 */

        /* 如果没有进行转置 */
        if ( notran ) {
            /* 如果进行了行平衡 */
            if ( rowequ ) {
                /* 对每个右手边向量进行行平衡 */
                for (j = 0; j < nrhs; ++j)
                    for (i = 0; i < A->nrow; ++i)
                        Bmat[i + j*ldb] *= R[i];
            }
        } else if ( colequ ) {
            /* 如果进行了列平衡 */
            for (j = 0; j < nrhs; ++j)
                for (i = 0; i < A->nrow; ++i)
                    Bmat[i + j*ldb] *= C[i];
        }

        /* 计算解矩阵 X */
        for (j = 0; j < nrhs; j++)  /* 保存右手边的副本 */
            for (i = 0; i < B->nrow; i++)
                Xmat[i + j*ldx] = Bmat[i + j*ldb];

        /* 计时开始 */
        t0 = SuperLU_timer_();
        /* 解线性方程组 */
        dgstrs (trant, L, U, perm_c, perm_r, X, stat, &info1);
        /* 计算求解时间 */
        utime[SOLVE] = SuperLU_timer_() - t0;

        /* 开始迭代细化以提高计算出的解，并计算误差界限和反向误差估计 */
        t0 = SuperLU_timer_();
        if ( options->IterRefine != NOREFINE ) {
            dgsrfs(trant, AA, L, U, perm_c, perm_r, equed, R, C, B,
                   X, ferr, berr, stat, &info1);
        } else {
            /* 如果不进行迭代细化，则设置初始误差界限为1.0 */
            for (j = 0; j < nrhs; ++j) ferr[j] = berr[j] = 1.0;
        }
        /* 计算迭代细化的时间 */
        utime[REFINE] = SuperLU_timer_() - t0;

        /* 将解矩阵 X 转换为原始系统的解 */
        if ( notran ) {
            /* 如果没有进行转置且进行了列平衡 */
            if ( colequ ) {
                /* 对每个右手边向量进行列平衡 */
                for (j = 0; j < nrhs; ++j)
                    for (i = 0; i < A->nrow; ++i)
                        Xmat[i + j*ldx] *= C[i];
            }
        } else if ( rowequ ) {
            /* 如果进行了转置且进行了行平衡 */
            for (j = 0; j < nrhs; ++j)
                for (i = 0; i < A->nrow; ++i)
                    Xmat[i + j*ldx] *= R[i];
        }
    } /* end if nrhs > 0 */

    /* 如果需要计算条件数 */
    if ( options->ConditionNumber ) {
        /* 如果条件数小于机器精度，设置 INFO = A->ncol + 1 表示矩阵在工作精度下是奇异的 */
        if ( *rcond < dmach("E") ) *info = A->ncol + 1;
    }

    /* 如果没有进行分解 */
    if ( nofact ) {
        /* 查询所需内存空间并销毁复合列压缩矩阵 AC */
        dQuerySpace(L, U, mem_usage);
        Destroy_CompCol_Permuted(&AC);
    }

    /* 如果 A 的类型为非正则格式 */
    if ( A->Stype == SLU_NR ) {
        /* 销毁超级矩阵存储 AA 并释放其内存 */
        Destroy_SuperMatrix_Store(AA);
        SUPERLU_FREE(AA);
    }
}



# 这行代码表示一个独立的右大括号 '}'，用于闭合一个代码块或数据结构
```