# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\sgssvx.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file sgssvx.c
 * \brief Solves the system of linear equations A*X=B or A'*X=B
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
sgssvx(superlu_options_t *options, SuperMatrix *A, int *perm_c, int *perm_r,
       int *etree, char *equed, float *R, float *C,
       SuperMatrix *L, SuperMatrix *U, void *work, int_t lwork,
       SuperMatrix *B, SuperMatrix *X, float *recip_pivot_growth, 
       float *rcond, float *ferr, float *berr, 
       GlobalLU_t *Glu, mem_usage_t *mem_usage, SuperLUStat_t *stat, int_t *info )
{
    // 定义变量
    DNformat  *Bstore, *Xstore;
    float    *Bmat, *Xmat;
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
    extern float slangs(char *, SuperMatrix *);

    // 获取矩阵 B 和 X 的存储格式及其数据
    Bstore = B->Store;
    Xstore = X->Store;
    Bmat   = Bstore->nzval;
    Xmat   = Xstore->nzval;
    ldb    = Bstore->lda;
    ldx    = Xstore->lda;
    nrhs   = B->ncol;

    // 初始化 info 为 0
    *info = 0;
    // 判断是否需要进行因式分解
    nofact = (options->Fact != FACTORED);
    // 判断是否需要均衡
    equil = (options->Equil == YES);
    // 判断是否不是转置操作
    notran = (options->Trans == NOTRANS);
    // 如果未进行因式分解，则设置 equed 为 'N'，且不需要行列均衡
    if ( nofact ) {
        *(unsigned char *)equed = 'N';
        rowequ = FALSE;
        colequ = FALSE;
    } else {
        // 判断是否需要行均衡或列均衡
        rowequ = strncmp(equed, "R", 1)==0 || strncmp(equed, "B", 1)==0;
        colequ = strncmp(equed, "C", 1)==0 || strncmp(equed, "B", 1)==0;
        // 计算安全最小值和安全最大值
        smlnum = smach("Safe minimum");   /* lamch_("Safe minimum"); */
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
    *info = -1;
    else if ( A->nrow != A->ncol || A->nrow < 0 ||
              (A->Stype != SLU_NC && A->Stype != SLU_NR) ||
              A->Dtype != SLU_S || A->Mtype != SLU_GE )
    *info = -2;

    // 其他代码将在这里继续...
}
    else if ( options->Fact == FACTORED &&
         !(rowequ || colequ || strncmp(equed, "N", 1)==0) )
    *info = -6;
    else {
    if (rowequ) {
        rcmin = bignum;  // 设置 rcmin 初值为一个很大的数
        rcmax = 0.;  // 设置 rcmax 初值为 0
        for (j = 0; j < A->nrow; ++j) {  // 遍历矩阵 A 的行数
        rcmin = SUPERLU_MIN(rcmin, R[j]);  // 计算 R[j] 的最小值
        rcmax = SUPERLU_MAX(rcmax, R[j]);  // 计算 R[j] 的最大值
        }
        if (rcmin <= 0.) *info = -7;  // 若 rcmin 小于等于 0，则设置 info 为 -7
        else if ( A->nrow > 0)
        rowcnd = SUPERLU_MAX(rcmin,smlnum) / SUPERLU_MIN(rcmax,bignum);  // 计算行条件数
        else rowcnd = 1.;  // 若 A 的行数为 0，则行条件数为 1
    }
    if (colequ && *info == 0) {
        rcmin = bignum;  // 设置 rcmin 初值为一个很大的数
        rcmax = 0.;  // 设置 rcmax 初值为 0
        for (j = 0; j < A->nrow; ++j) {  // 遍历矩阵 A 的行数
        rcmin = SUPERLU_MIN(rcmin, C[j]);  // 计算 C[j] 的最小值
        rcmax = SUPERLU_MAX(rcmax, C[j]);  // 计算 C[j] 的最大值
        }
        if (rcmin <= 0.) *info = -8;  // 若 rcmin 小于等于 0，则设置 info 为 -8
        else if (A->nrow > 0)
        colcnd = SUPERLU_MAX(rcmin,smlnum) / SUPERLU_MIN(rcmax,bignum);  // 计算列条件数
        else colcnd = 1.;  // 若 A 的行数为 0，则列条件数为 1
    }
    if (*info == 0) {
        if ( lwork < -1 ) *info = -12;  // 若 lwork 小于 -1，则设置 info 为 -12
        else if ( B->ncol < 0 ) *info = -13;  // 若 B 的列数小于 0，则设置 info 为 -13
        else if ( B->ncol > 0 ) { /* no checking if B->ncol=0 */
             if ( Bstore->lda < SUPERLU_MAX(0, A->nrow) ||
              B->Stype != SLU_DN || B->Dtype != SLU_S || 
              B->Mtype != SLU_GE )
        *info = -13;  // 若 B 的列数大于 0 且不满足条件，则设置 info 为 -13
            }
        if ( X->ncol < 0 ) *info = -14;  // 若 X 的列数小于 0，则设置 info 为 -14
            else if ( X->ncol > 0 ) { /* no checking if X->ncol=0 */
                 if ( Xstore->lda < SUPERLU_MAX(0, A->nrow) ||
              (B->ncol != 0 && B->ncol != X->ncol) ||
                      X->Stype != SLU_DN ||
              X->Dtype != SLU_S || X->Mtype != SLU_GE )
        *info = -14;  // 若 X 的列数大于 0 且不满足条件，则设置 info 为 -14
            }
    }
    }
    if (*info != 0) {
    i = -(*info);
    input_error("sgssvx", &i);  // 输出错误信息
    return;  // 返回
    }
    
    /* Initialization for factor parameters */
    panel_size = sp_ienv(1);  // 获取 panel_size 参数
    relax      = sp_ienv(2);  // 获取 relax 参数

    utime = stat->utime;  // 设置 utime 参数

    /* Convert A to SLU_NC format when necessary. */
    if ( A->Stype == SLU_NR ) {
    NRformat *Astore = A->Store;  // 获取 A 的存储格式信息
    AA = (SuperMatrix *) SUPERLU_MALLOC( sizeof(SuperMatrix) );  // 分配 AA 的内存空间
    sCreate_CompCol_Matrix(AA, A->ncol, A->nrow, Astore->nnz, 
                   Astore->nzval, Astore->colind, Astore->rowptr,
                   SLU_NC, A->Dtype, A->Mtype);  // 创建 AA 的稀疏矩阵
    if ( notran ) { /* Reverse the transpose argument. */
        trant = TRANS;  // 设置 trant 为转置
        notran = 0;  // 设置 notran 为 0
    } else {
        trant = NOTRANS;  // 设置 trant 为非转置
        notran = 1;  // 设置 notran 为 1
    }
    } else { /* A->Stype == SLU_NC */
    trant = options->Trans;  // 获取转置选项
    AA = A;  // AA 指向 A
    }

    if ( nofact && equil ) {
    t0 = SuperLU_timer_();  // 记录当前时间
    /* Compute row and column scalings to equilibrate the matrix A. */
    sgsequ(AA, R, C, &rowcnd, &colcnd, &amax, &info1);  // 计算行和列的缩放因子
    
    if ( info1 == 0 ) {
        /* Equilibrate matrix A. */
        slaqgs(AA, R, C, rowcnd, colcnd, amax, equed);  // 根据缩放因子 equilibrate 矩阵 A
        rowequ = strncmp(equed, "R", 1)==0 || strncmp(equed, "B", 1)==0;  // 判断是否行 equilibrate
        colequ = strncmp(equed, "C", 1)==0 || strncmp(equed, "B", 1)==0;  // 判断是否列 equilibrate
    }
    utime[EQUIL] = SuperLU_timer_() - t0;  // 计算 equilibration 时间
    }
    # 如果 nofact 为真，则执行以下操作
    if ( nofact ) {
    
        # 记录当前时间到 t0，用于计时
        t0 = SuperLU_timer_();
        
        /*
         * 根据 permc_spec 获取列置换向量 perm_c[]
         * 根据 options->ColPerm 来决定 permc_spec 的具体含义：
         *   permc_spec = NATURAL: 使用自然顺序
         *   permc_spec = MMD_AT_PLUS_A: A'+A 结构的最小度排序
         *   permc_spec = MMD_ATA: A'*A 结构的最小度排序
         *   permc_spec = COLAMD: 近似最小度列排序
         *   permc_spec = MY_PERMC: 已经在 perm_c[] 中提供的排序
         */
        permc_spec = options->ColPerm;
        
        # 如果 permc_spec 不是 MY_PERMC，并且 options->Fact 为 DOFACT，则调用 get_perm_c() 函数
        if ( permc_spec != MY_PERMC && options->Fact == DOFACT )
            get_perm_c(permc_spec, AA, perm_c);
        
        # 计算列置换的时间消耗并记录在 utime[COLPERM] 中
        utime[COLPERM] = SuperLU_timer_() - t0;

        # 记录当前时间到 t0，用于计时
        t0 = SuperLU_timer_();
        
        # 执行填充因子分析和预排序操作，生成 AC 结构
        sp_preorder(options, AA, perm_c, etree, &AC);
        
        # 计算预排序的时间消耗并记录在 utime[ETREE] 中
        utime[ETREE] = SuperLU_timer_() - t0;
    }
    /*
    打印信息：Factor PA = LU ... relax %d\tw %d\tmaxsuper %d\trowblk %d\n，并刷新标准输出流
    printf("Factor PA = LU ... relax %d\tw %d\tmaxsuper %d\trowblk %d\n", 
           relax, panel_size, sp_ienv(3), sp_ienv(4));
    fflush(stdout);
    */
    
    /*
    计算 A*Pc 的 LU 分解。
    */
    t0 = SuperLU_timer_();
    sgstrf(options, &AC, relax, panel_size, etree,
                work, lwork, perm_c, perm_r, L, U, Glu, stat, info);
    utime[FACT] = SuperLU_timer_() - t0;
    
    /*
    如果 lwork == -1，计算内存使用量并返回。
    */
    if ( lwork == -1 ) {
        mem_usage->total_needed = *info - A->ncol;
        return;
    }
    
    /*
    如果 *info > 0，则表示 A 的某些列出现秩不足，计算这些列的倒数主轴增长因子。
    */
    if ( *info > 0 ) {
        if ( *info <= A->ncol ) {
            /*
            计算 A 的前 *info 个秩不足列的倒数主轴增长因子。
            */
            *recip_pivot_growth = sPivotGrowth(*info, AA, perm_c, L, U);
        }
        return;
    }
    
    /*
    在此时 *info == 0。
    */

    /*
    如果 options->PivotGrowth 为真，计算整体的倒数主轴增长因子 *recip_pivot_growth。
    */
    if ( options->PivotGrowth ) {
        /*
        计算整体的倒数主轴增长因子 *recip_pivot_growth。
        */
        *recip_pivot_growth = sPivotGrowth(A->ncol, AA, perm_c, L, U);
    }

    /*
    如果 options->ConditionNumber 为真，估算 A 的条件数的倒数。
    */
    if ( options->ConditionNumber ) {
        t0 = SuperLU_timer_();
        /*
        根据 notran 的值设置 norm 的字符值。
        */
        if ( notran ) {
            *(unsigned char *)norm = '1';
        } else {
            *(unsigned char *)norm = 'I';
        }
        /*
        计算矩阵 AA 的范数 anorm。
        */
        anorm = slangs(norm, AA);
        /*
        计算矩阵 A 的条件数的倒数 rcond。
        */
        sgscon(norm, L, U, anorm, rcond, stat, &info1);
        utime[RCOND] = SuperLU_timer_() - t0;
    }
    if ( nrhs > 0 ) {
        /* 如果右手边数量大于0，则进行下列操作 */

        /* 如果进行了等价处理，则对右手边进行缩放 */
        if ( notran ) {
            if ( rowequ ) {
                for (j = 0; j < nrhs; ++j)
                    for (i = 0; i < A->nrow; ++i)
                        Bmat[i + j*ldb] *= R[i];
            }
        } else if ( colequ ) {
            for (j = 0; j < nrhs; ++j)
                for (i = 0; i < A->nrow; ++i)
                    Bmat[i + j*ldb] *= C[i];
        }

        /* 计算解矩阵 X */
        for (j = 0; j < nrhs; j++)  /* 保存右手边的副本 */
            for (i = 0; i < B->nrow; i++)
                Xmat[i + j*ldx] = Bmat[i + j*ldb];
    
        t0 = SuperLU_timer_();
        sgstrs (trant, L, U, perm_c, perm_r, X, stat, &info1);
        utime[SOLVE] = SuperLU_timer_() - t0;
    
        /* 使用迭代细化来改进计算的解，并计算其误差界限和反向误差估计 */
        t0 = SuperLU_timer_();
        if ( options->IterRefine != NOREFINE ) {
            sgsrfs(trant, AA, L, U, perm_c, perm_r, equed, R, C, B,
                   X, ferr, berr, stat, &info1);
        } else {
            for (j = 0; j < nrhs; ++j) ferr[j] = berr[j] = 1.0;
        }
        utime[REFINE] = SuperLU_timer_() - t0;

        /* 将解矩阵 X 转换为原始系统的解 */
        if ( notran ) {
            if ( colequ ) {
                for (j = 0; j < nrhs; ++j)
                    for (i = 0; i < A->nrow; ++i)
                        Xmat[i + j*ldx] *= C[i];
            }
        } else if ( rowequ ) {
            for (j = 0; j < nrhs; ++j)
                for (i = 0; i < A->nrow; ++i)
                    Xmat[i + j*ldx] *= R[i];
        }
    } /* end if nrhs > 0 */

    if ( options->ConditionNumber ) {
        /* 如果需要计算条件数 */
        /* 如果矩阵的条件数小于机器精度，则将 info 设置为 A->ncol + 1 */
        if ( *rcond < smach("E") ) *info = A->ncol + 1;
    }

    if ( nofact ) {
        /* 如果未进行因式分解，则查询空间使用情况并销毁处理后的矩阵 */
        sQuerySpace(L, U, mem_usage);
        Destroy_CompCol_Permuted(&AC);
    }
    if ( A->Stype == SLU_NR ) {
        /* 如果矩阵类型为非规则存储，则销毁超级矩阵 */
        Destroy_SuperMatrix_Store(AA);
        SUPERLU_FREE(AA);
    }
}


注释：


# 这行代码是一个单独的右花括号 '}'，用于结束一个代码块或函数的定义。
```