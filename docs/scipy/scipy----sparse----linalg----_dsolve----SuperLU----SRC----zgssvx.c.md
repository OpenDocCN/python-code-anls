# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zgssvx.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file zgssvx.c
 * \brief Solves the system of linear equations A*X=B or A'*X=B
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
zgssvx(superlu_options_t *options, SuperMatrix *A, int *perm_c, int *perm_r,
       int *etree, char *equed, double *R, double *C,
       SuperMatrix *L, SuperMatrix *U, void *work, int_t lwork,
       SuperMatrix *B, SuperMatrix *X, double *recip_pivot_growth, 
       double *rcond, double *ferr, double *berr, 
       GlobalLU_t *Glu, mem_usage_t *mem_usage, SuperLUStat_t *stat, int_t *info )
{
    DNformat  *Bstore, *Xstore;
    doublecomplex    *Bmat, *Xmat;
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
    extern double zlangs(char *, SuperMatrix *);

    // 获取B和X的存储格式及相关信息
    Bstore = B->Store;
    Xstore = X->Store;
    Bmat   = Bstore->nzval;
    Xmat   = Xstore->nzval;
    ldb    = Bstore->lda;
    ldx    = Xstore->lda;
    nrhs   = B->ncol;

    // 初始化info为0
    *info = 0;
    // 判断是否需要进行因子分解
    nofact = (options->Fact != FACTORED);
    // 判断是否需要均衡处理
    equil = (options->Equil == YES);
    // 判断是否为非转置解
    notran = (options->Trans == NOTRANS);
    if ( nofact ) {
        // 如果未进行因子分解，设置equed为'N'表示无均衡
        *(unsigned char *)equed = 'N';
        rowequ = FALSE; // 行均衡标志设为假
        colequ = FALSE; // 列均衡标志设为假
    } else {
        // 如果已进行因子分解，根据equed的值判断是否进行行列均衡
        rowequ = strncmp(equed, "R", 1)==0 || strncmp(equed, "B", 1)==0;
        colequ = strncmp(equed, "C", 1)==0 || strncmp(equed, "B", 1)==0;
        // 获取计算机浮点数的安全最小值和大数的倒数
        smlnum = dmach("Safe minimum");   /* lamch_("Safe minimum"); */
        bignum = 1. / smlnum;
    }

#if 0
printf("dgssvx: Fact=%4d, Trans=%4d, equed=%c\n",
       options->Fact, options->Trans, *equed);
#endif

    // 测试输入参数的有效性
    if (options->Fact != DOFACT && options->Fact != SamePattern &&
        options->Fact != SamePattern_SameRowPerm &&
        options->Fact != FACTORED &&
        options->Trans != NOTRANS && options->Trans != TRANS && 
        options->Trans != CONJ &&
        options->Equil != NO && options->Equil != YES)
    *info = -1; // 如果参数不合法，设置info为-1
    else if ( A->nrow != A->ncol || A->nrow < 0 ||
              (A->Stype != SLU_NC && A->Stype != SLU_NR) ||
              A->Dtype != SLU_Z || A->Mtype != SLU_GE )
    *info = -2; // 如果矩阵A的尺寸或类型不符合要求，设置info为-2
}


注释部分已按照要求添加到了代码块中，解释了每行代码的作用和意图。
    else if ( options->Fact == FACTORED &&
         !(rowequ || colequ || strncmp(equed, "N", 1)==0) )
    /* Check if the matrix is already factored and not equilibrated */
    *info = -6;
    else {
    /* If matrix is not yet factored or needs equilibration */
    if (rowequ) {
        /* Initialize rcmin and rcmax */
        rcmin = bignum;
        rcmax = 0.;
        /* Compute minimum and maximum row scaling factors */
        for (j = 0; j < A->nrow; ++j) {
        rcmin = SUPERLU_MIN(rcmin, R[j]);
        rcmax = SUPERLU_MAX(rcmax, R[j]);
        }
        /* Check row scaling */
        if (rcmin <= 0.) *info = -7;
        else if ( A->nrow > 0)
        /* Compute row condition number */
        rowcnd = SUPERLU_MAX(rcmin,smlnum) / SUPERLU_MIN(rcmax,bignum);
        else rowcnd = 1.;
    }
    if (colequ && *info == 0) {
        /* Initialize rcmin and rcmax */
        rcmin = bignum;
        rcmax = 0.;
        /* Compute minimum and maximum column scaling factors */
        for (j = 0; j < A->nrow; ++j) {
        rcmin = SUPERLU_MIN(rcmin, C[j]);
        rcmax = SUPERLU_MAX(rcmax, C[j]);
        }
        /* Check column scaling */
        if (rcmin <= 0.) *info = -8;
        else if (A->nrow > 0)
        /* Compute column condition number */
        colcnd = SUPERLU_MAX(rcmin,smlnum) / SUPERLU_MIN(rcmax,bignum);
        else colcnd = 1.;
    }
    if (*info == 0) {
        /* Check workspace size */
        if ( lwork < -1 ) *info = -12;
        else if ( B->ncol < 0 ) *info = -13;
        else if ( B->ncol > 0 ) { /* no checking if B->ncol=0 */
             /* Check properties of B matrix */
             if ( Bstore->lda < SUPERLU_MAX(0, A->nrow) ||
              B->Stype != SLU_DN || B->Dtype != SLU_Z || 
              B->Mtype != SLU_GE )
        *info = -13;
            }
        /* Check properties of X matrix */
        if ( X->ncol < 0 ) *info = -14;
            else if ( X->ncol > 0 ) { /* no checking if X->ncol=0 */
                 if ( Xstore->lda < SUPERLU_MAX(0, A->nrow) ||
              (B->ncol != 0 && B->ncol != X->ncol) ||
                      X->Stype != SLU_DN ||
              X->Dtype != SLU_Z || X->Mtype != SLU_GE )
        *info = -14;
            }
    }
    }
    if (*info != 0) {
    /* Handle input errors */
    i = -(*info);
    input_error("zgssvx", &i);
    return;
    }
    
    /* Initialization for factor parameters */
    /* Set panel size and relaxation parameter */
    panel_size = sp_ienv(1);
    relax      = sp_ienv(2);

    utime = stat->utime;
    
    /* Convert A to SLU_NC format when necessary. */
    if ( A->Stype == SLU_NR ) {
    /* Convert A matrix from row format to column format */
    NRformat *Astore = A->Store;
    AA = (SuperMatrix *) SUPERLU_MALLOC( sizeof(SuperMatrix) );
    zCreate_CompCol_Matrix(AA, A->ncol, A->nrow, Astore->nnz, 
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
    /* Use original A matrix as it is already in column format */
    trant = options->Trans;
    AA = A;
    }

    if ( nofact && equil ) {
    t0 = SuperLU_timer_();
    /* Compute row and column scalings to equilibrate the matrix A. */
    zgsequ(AA, R, C, &rowcnd, &colcnd, &amax, &info1);
    
    if ( info1 == 0 ) {
        /* Equilibrate matrix A based on computed scalings */
        zlaqgs(AA, R, C, rowcnd, colcnd, amax, equed);
        /* Determine if row and/or column equilibration is applied */
        rowequ = strncmp(equed, "R", 1)==0 || strncmp(equed, "B", 1)==0;
        colequ = strncmp(equed, "C", 1)==0 || strncmp(equed, "B", 1)==0;
    }
    /* Record the time taken for equilibration */
    utime[EQUIL] = SuperLU_timer_() - t0;
    }
    # 如果选项中没有指定列排列，则执行以下操作
    if ( nofact ) {
    
        # 记录当前时间，用于计时
        t0 = SuperLU_timer_();
    /*
     * 根据 permc_spec 进行列置换，根据 options->ColPerm 的不同值选择不同的策略：
     *   permc_spec = NATURAL: 自然顺序
     *   permc_spec = MMD_AT_PLUS_A: A'+A 结构的最小度排序
     *   permc_spec = MMD_ATA: A'*A 结构的最小度排序
     *   permc_spec = COLAMD: 近似最小度列排序
     *   permc_spec = MY_PERMC: 已经在 perm_c[] 中提供的自定义排序
     */
    permc_spec = options->ColPerm;
    
    # 如果 permc_spec 不等于 MY_PERMC 并且选项中的 Fact 为 DOFACT，则获取列置换 perm_c[]
    if ( permc_spec != MY_PERMC && options->Fact == DOFACT )
            get_perm_c(permc_spec, AA, perm_c);
    
    # 计算列置换的耗时
    utime[COLPERM] = SuperLU_timer_() - t0;

    # 记录当前时间，用于计时
    t0 = SuperLU_timer_();
    
    # 执行顺序图预处理，生成因子分解树 AC
    sp_preorder(options, AA, perm_c, etree, &AC);
    
    # 计算生成因子分解树的耗时
    utime[ETREE] = SuperLU_timer_() - t0;
    ```
    /* 打印信息到标准输出，用于调试和状态跟踪，这段代码被注释掉了
       printf("Factor PA = LU ... relax %d\tw %d\tmaxsuper %d\trowblk %d\n", 
              relax, panel_size, sp_ienv(3), sp_ienv(4));
       fflush(stdout); */

    /* 计算矩阵 A*Pc 的 LU 分解 */
    t0 = SuperLU_timer_();
    zgstrf(options, &AC, relax, panel_size, etree,
           work, lwork, perm_c, perm_r, L, U, Glu, stat, info);
    utime[FACT] = SuperLU_timer_() - t0;

    /* 如果 lwork 为 -1，则计算所需的内存空间大小并返回 */
    if ( lwork == -1 ) {
        mem_usage->total_needed = *info - A->ncol;
        return;
    }

    /* 处理可能的报错信息 */
    if ( *info > 0 ) {
        if ( *info <= A->ncol ) {
            /* 计算 A 的前 *info 列的倒数主轴增长因子 */
            *recip_pivot_growth = zPivotGrowth(*info, AA, perm_c, L, U);
        }
        return;
    }

    /* *info == 0 时执行以下操作 */

    if ( options->PivotGrowth ) {
        /* 计算倒数主轴增长因子 *recip_pivot_growth */
        *recip_pivot_growth = zPivotGrowth(A->ncol, AA, perm_c, L, U);
    }

    if ( options->ConditionNumber ) {
        /* 估计矩阵 A 的条件数的倒数 */
        t0 = SuperLU_timer_();
        if ( notran ) {
            *(unsigned char *)norm = '1';
        } else {
            *(unsigned char *)norm = 'I';
        }
        anorm = zlangs(norm, AA);
        zgscon(norm, L, U, anorm, rcond, stat, &info1);
        utime[RCOND] = SuperLU_timer_() - t0;
    }
    /* 如果输入参数个数大于0，则执行以下代码块 */
    if ( nrhs > 0 ) {
        /* 如果未进行转置操作 */
        if ( notran ) {
            /* 如果进行了行均衡 */
            if ( rowequ ) {
                /* 遍历所有的右侧向量 */
                for (j = 0; j < nrhs; ++j)
                    /* 遍历矩阵A的每一行 */
                    for (i = 0; i < A->nrow; ++i)
                        /* 对右侧向量Bmat的每个元素进行缩放 */
                        zd_mult(&Bmat[i+j*ldb], &Bmat[i+j*ldb], R[i]);
            }
        } else if ( colequ ) {
            /* 如果进行了列均衡 */
            for (j = 0; j < nrhs; ++j)
                /* 遍历矩阵A的每一行 */
                for (i = 0; i < A->nrow; ++i)
                    /* 对右侧向量Bmat的每个元素进行缩放 */
                    zd_mult(&Bmat[i+j*ldb], &Bmat[i+j*ldb], C[i]);
        }

        /* 计算解矩阵X */
        for (j = 0; j < nrhs; j++)  /* 保存右侧向量的副本 */
            for (i = 0; i < B->nrow; i++)
                /* 将Bmat中的数据复制到Xmat中 */
                Xmat[i + j*ldx] = Bmat[i + j*ldb];
    
        t0 = SuperLU_timer_();
        /* 调用 zgstrs 函数求解线性方程组 */
        zgstrs (trant, L, U, perm_c, perm_r, X, stat, &info1);
        utime[SOLVE] = SuperLU_timer_() - t0;
    
        /* 使用迭代细化来改进计算出的解，并计算误差界限和后向误差估计 */
        t0 = SuperLU_timer_();
        if ( options->IterRefine != NOREFINE ) {
            /* 调用 zgsrfs 函数进行迭代细化 */
            zgsrfs(trant, AA, L, U, perm_c, perm_r, equed, R, C, B,
                   X, ferr, berr, stat, &info1);
        } else {
            /* 如果不进行迭代细化，则初始化 ferr 和 berr 为1.0 */
            for (j = 0; j < nrhs; ++j) ferr[j] = berr[j] = 1.0;
        }
        utime[REFINE] = SuperLU_timer_() - t0;

        /* 将解矩阵X转换为原始系统的解 */
        if ( notran ) {
            /* 如果未进行转置操作，并且进行了列均衡 */
            if ( colequ ) {
                /* 遍历所有的右侧向量 */
                for (j = 0; j < nrhs; ++j)
                    /* 遍历矩阵A的每一行 */
                    for (i = 0; i < A->nrow; ++i)
                        /* 对解矩阵Xmat的每个元素进行缩放 */
                        zd_mult(&Xmat[i+j*ldx], &Xmat[i+j*ldx], C[i]);
            }
        } else if ( rowequ ) {
            /* 如果进行了转置操作，并且进行了行均衡 */
            for (j = 0; j < nrhs; ++j)
                /* 遍历矩阵A的每一行 */
                for (i = 0; i < A->nrow; ++i)
                    /* 对解矩阵Xmat的每个元素进行缩放 */
                    zd_mult(&Xmat[i+j*ldx], &Xmat[i+j*ldx], R[i]);
        }
    } /* end if nrhs > 0 */

    /* 如果 options 中指定了计算条件数的选项 */
    if ( options->ConditionNumber ) {
        /* 如果 rcond 小于机器精度阈值，则将 info 设置为 A->ncol + 1 */
        if ( *rcond < dmach("E") ) *info = A->ncol + 1;
    }

    /* 如果没有进行因式分解 */
    if ( nofact ) {
        /* 查询 L 和 U 的空间需求 */
        zQuerySpace(L, U, mem_usage);
        /* 销毁已重新排列的稀疏矩阵 AC */
        Destroy_CompCol_Permuted(&AC);
    }
    /* 如果 A 的存储类型为 SLU_NR */
    if ( A->Stype == SLU_NR ) {
        /* 销毁 AA 的存储结构 */
        Destroy_SuperMatrix_Store(AA);
        /* 释放 AA 占用的内存空间 */
        SUPERLU_FREE(AA);
    }
}



# 这行代码表示一个代码块的结束，对应于前面的一个代码块的开始或中间部分的结束。
```