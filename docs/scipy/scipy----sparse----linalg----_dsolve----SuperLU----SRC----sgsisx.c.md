# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\sgsisx.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file sgsisx.c
 * \brief Computes an approximate solutions of linear equations A*X=B or A'*X=B
 *
 * <pre>
 * -- SuperLU routine (version 4.2) --
 * Lawrence Berkeley National Laboratory.
 * November, 2010
 * August, 2011
 * </pre>
 */
#include "slu_sdefs.h"

void
sgsisx(superlu_options_t *options, SuperMatrix *A, int *perm_c, int *perm_r,
       int *etree, char *equed, float *R, float *C,
       SuperMatrix *L, SuperMatrix *U, void *work, int_t lwork,
       SuperMatrix *B, SuperMatrix *X,
       float *recip_pivot_growth, float *rcond,
       GlobalLU_t *Glu, mem_usage_t *mem_usage, SuperLUStat_t *stat, int_t *info)
{
    DNformat  *Bstore, *Xstore;
    float    *Bmat, *Xmat;
    int       ldb, ldx, nrhs, n;
    SuperMatrix *AA;/* A in SLU_NC format used by the factorization routine.*/
    SuperMatrix AC; /* Matrix postmultiplied by Pc */
    int       colequ, equil, nofact, notran, rowequ, permc_spec, mc64;
    trans_t   trant;
    char      norm[1];
    int_t     i, j;
    float    amax, anorm, bignum, smlnum, colcnd, rowcnd, rcmax, rcmin;
    int       relax, panel_size, info1;
    double    t0;      /* temporary time */
    double    *utime;

    int *perm = NULL; /* permutation returned from MC64 */

    /* External functions */
    extern float slangs(char *, SuperMatrix *);

    Bstore = B->Store;  // 获取矩阵 B 的存储格式
    Xstore = X->Store;  // 获取矩阵 X 的存储格式
    Bmat   = Bstore->nzval;  // 获取矩阵 B 的非零元素数组
    Xmat   = Xstore->nzval;  // 获取矩阵 X 的非零元素数组
    ldb    = Bstore->lda;  // 获取矩阵 B 的 leading dimension
    ldx    = Xstore->lda;  // 获取矩阵 X 的 leading dimension
    nrhs   = B->ncol;  // 矩阵 B 的列数，即右侧向量的个数
    n      = B->nrow;  // 矩阵 B 的行数，即线性方程组的未知数个数

    *info = 0;  // 初始化 info 为 0
    nofact = (options->Fact != FACTORED);  // 是否需要进行因式分解
    equil = (options->Equil == YES);  // 是否进行均衡处理
    notran = (options->Trans == NOTRANS);  // 是否进行转置操作
    mc64 = (options->RowPerm == LargeDiag_MC64);  // 是否使用 MC64 进行行重排
    if ( nofact ) {
    *(unsigned char *)equed = 'N';  // 如果不进行因式分解，则不做均衡处理
    rowequ = FALSE;  // 行均衡标志位为假
    colequ = FALSE;  // 列均衡标志位为假
    } else {
    rowequ = strncmp(equed, "R", 1)==0 || strncmp(equed, "B", 1)==0;  // 判断是否进行行均衡
    colequ = strncmp(equed, "C", 1)==0 || strncmp(equed, "B", 1)==0;  // 判断是否进行列均衡
    smlnum = smach("Safe minimum");  /* lamch_("Safe minimum"); */  // 获取安全最小值
    bignum = 1. / smlnum;  // 获取安全最大值
    }

    /* Test the input parameters */
    if (options->Fact != DOFACT && options->Fact != SamePattern &&
    options->Fact != SamePattern_SameRowPerm &&
    options->Fact != FACTORED &&
    options->Trans != NOTRANS && options->Trans != TRANS && 
    options->Trans != CONJ &&
    options->Equil != NO && options->Equil != YES)
    *info = -1;  // 输入参数检测，如果不符合要求则置 info 为 -1
    else if ( A->nrow != A->ncol || A->nrow < 0 ||
          (A->Stype != SLU_NC && A->Stype != SLU_NR) ||
          A->Dtype != SLU_S || A->Mtype != SLU_GE )
    *info = -2;  // 矩阵 A 的维度或类型不符合要求，置 info 为 -2
    else if ( options->Fact == FACTORED &&
         !(rowequ || colequ || strncmp(equed, "N", 1)==0) )
    *info = -6;
    else {


    // 如果传入的 *info 值为 -6，则执行以下操作；否则进入下一个条件分支。
    *info = -6;
    else {



    if (rowequ) {


    // 如果 rowequ 为真，则执行以下代码块。
    if (rowequ) {



        rcmin = bignum;
        rcmax = 0.;
        for (j = 0; j < A->nrow; ++j) {
        rcmin = SUPERLU_MIN(rcmin, R[j]);
        rcmax = SUPERLU_MAX(rcmax, R[j]);
        }
        if (rcmin <= 0.) *info = -7;
        else if ( A->nrow > 0)
        rowcnd = SUPERLU_MAX(rcmin,smlnum) / SUPERLU_MIN(rcmax,bignum);
        else rowcnd = 1.;
    }


        // 初始化 rcmin 和 rcmax 为 bignum 和 0
        rcmin = bignum;
        rcmax = 0.;
        // 遍历 R 数组，更新 rcmin 和 rcmax
        for (j = 0; j < A->nrow; ++j) {
        rcmin = SUPERLU_MIN(rcmin, R[j]);
        rcmax = SUPERLU_MAX(rcmax, R[j]);
        }
        // 检查 rcmin 是否小于等于 0，若是则设置 *info 为 -7
        if (rcmin <= 0.) *info = -7;
        // 否则计算 rowcnd 的值
        else if ( A->nrow > 0)
        rowcnd = SUPERLU_MAX(rcmin,smlnum) / SUPERLU_MIN(rcmax,bignum);
        else rowcnd = 1.;
    }



    if (colequ && *info == 0) {


    // 如果 colequ 为真且 *info 为 0，则执行以下代码块。
    if (colequ && *info == 0) {



        rcmin = bignum;
        rcmax = 0.;
        for (j = 0; j < A->nrow; ++j) {
        rcmin = SUPERLU_MIN(rcmin, C[j]);
        rcmax = SUPERLU_MAX(rcmax, C[j]);
        }
        if (rcmin <= 0.) *info = -8;
        else if (A->nrow > 0)
        colcnd = SUPERLU_MAX(rcmin,smlnum) / SUPERLU_MIN(rcmax,bignum);
        else colcnd = 1.;
    }


        // 初始化 rcmin 和 rcmax 为 bignum 和 0
        rcmin = bignum;
        rcmax = 0.;
        // 遍历 C 数组，更新 rcmin 和 rcmax
        for (j = 0; j < A->nrow; ++j) {
        rcmin = SUPERLU_MIN(rcmin, C[j]);
        rcmax = SUPERLU_MAX(rcmax, C[j]);
        }
        // 检查 rcmin 是否小于等于 0，若是则设置 *info 为 -8
        if (rcmin <= 0.) *info = -8;
        // 否则计算 colcnd 的值
        else if (A->nrow > 0)
        colcnd = SUPERLU_MAX(rcmin,smlnum) / SUPERLU_MIN(rcmax,bignum);
        else colcnd = 1.;
    }



    if (*info == 0) {


    // 如果 *info 等于 0，则执行以下代码块。
    if (*info == 0) {



        if ( lwork < -1 ) *info = -12;
        else if ( B->ncol < 0 || Bstore->lda < SUPERLU_MAX(0, A->nrow) ||
              B->Stype != SLU_DN || B->Dtype != SLU_S || 
              B->Mtype != SLU_GE )
        *info = -13;
        else if ( X->ncol < 0 || Xstore->lda < SUPERLU_MAX(0, A->nrow) ||
              (B->ncol != 0 && B->ncol != X->ncol) ||
              X->Stype != SLU_DN ||
              X->Dtype != SLU_S || X->Mtype != SLU_GE )
        *info = -14;
    }


        // 检查 lwork 是否小于 -1，若是则设置 *info 为 -12
        if ( lwork < -1 ) *info = -12;
        // 否则检查 B 的属性，如果不满足要求则设置 *info 为 -13
        else if ( B->ncol < 0 || Bstore->lda < SUPERLU_MAX(0, A->nrow) ||
              B->Stype != SLU_DN || B->Dtype != SLU_S || 
              B->Mtype != SLU_GE )
        *info = -13;
        // 否则检查 X 的属性，如果不满足要求则设置 *info 为 -14
        else if ( X->ncol < 0 || Xstore->lda < SUPERLU_MAX(0, A->nrow) ||
              (B->ncol != 0 && B->ncol != X->ncol) ||
              X->Stype != SLU_DN ||
              X->Dtype != SLU_S || X->Mtype != SLU_GE )
        *info = -14;
    }



    }
    if (*info != 0) {
    int ii = -(*info);
    input_error("sgsisx", &ii);
    return;
    }


    // 如果 *info 不等于 0，则执行以下代码块。
    if (*info != 0) {
    // 定义 ii 为 *info 的负值
    int ii = -(*info);
    // 调用 input_error 函数报告错误，传入错误码 ii 的地址
    input_error("sgsisx", &ii);
    // 函数结束返回
    return;
    }



    /* Initialization for factor parameters */
    panel_size = sp_ienv(1);
    relax      = sp_ienv(2);


    // 初始化因子参数
    // 获取 panel_size 和 relax 的值
    panel_size = sp_ienv(1);
    relax      = sp_ienv(2);



    utime = stat->utime;


    // 将 stat 结构体中的 utime 赋值给 utime 变量
    utime = stat->utime;



    /* Convert A to SLU_NC format when necessary. */
    if ( A->Stype == SLU_NR ) {
    NRformat *Astore = A->Store;
    AA = (SuperMatrix *) SUPERLU_MALLOC( sizeof(SuperMatrix) );
    sCreate_CompCol_Matrix(AA, A->ncol, A->nrow, Astore->nnz,
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
    trant = options->Trans;
    AA = A;
    }


    // 根据 A 的存储格式将其转换为 SLU_NC 格式（超节点列压缩存储格式），如果必要的话。
    if ( A->Stype == SLU_NR ) {
    // 获取 A 的 NRformat 存储格式信息
    NRformat *Astore = A->Store;
    // 分配 AA 的内存空间，用于存储转换后的超矩阵
    AA = (SuperMatrix *) SUPERLU_MALLOC( sizeof(SuperMatrix) );
    // 调用 sCreate_CompCol_Matrix 函数将 A 转换为 SLU_NC 格式的超矩阵 AA
    sCreate_CompCol_Matrix(AA, A->ncol, A->nrow, Astore->nnz,
                   Astore->nzval, Astore->colind, Astore->rowptr,
                   SLU_NC, A->Dtype, A->Mtype);
    // 如果 notran 为真，则反转置参数
    if ( notran ) { /* Reverse the transpose argument. */
        trant = TRANS;
        notran = 0;
    } else {
        trant = NOTRANS;
        notran = 1;
    }
    } else { /* A->Stype == SLU_NC */
    // 否则直接
    if ( mc64 ) {
        // 如果 mc64 为真，则执行以下操作
        t0 = SuperLU_timer_();
        // 记录当前时间，用于计时

        if ((perm = int32Malloc(n)) == NULL)
        // 分配大小为 n 的 int32 数组 perm，用于存储排列结果
        ABORT("SUPERLU_MALLOC fails for perm[]");
        // 如果分配失败，则输出错误信息并终止程序

        info1 = sldperm(5, n, nnz, colptr, rowind, nzval, perm, R, C);
        // 调用 sldperm 函数进行列排列操作，返回操作结果信息

        if (info1 != 0) { /* MC64 fails, call sgsequ() later */
            // 如果列排列操作失败，则设置 mc64 为假，稍后调用 sgsequ()
            mc64 = 0;
            // 将 mc64 设置为假
            SUPERLU_FREE(perm);
            // 释放 perm 数组的内存空间
            perm = NULL;
            // 将 perm 指针置为空
        } else {
            // 如果列排列操作成功
            if ( equil ) {
                // 如果需要进行均衡操作
                rowequ = colequ = 1;
                // 将 rowequ 和 colequ 置为真，表示进行行列均衡操作
                for (i = 0; i < n; i++) {
                    R[i] = exp(R[i]);
                    C[i] = exp(C[i]);
                    // 对 R 和 C 数组中的元素进行指数变换
                }
                /* scale the matrix */
                // 缩放矩阵
                for (j = 0; j < n; j++) {
                    for (i = colptr[j]; i < colptr[j + 1]; i++) {
                        nzval[i] *= R[rowind[i]] * C[j];
                        // 缩放矩阵的非零元素
                    }
                }
                *equed = 'B';
                // 将 equed 设置为 'B'，表示均衡操作已完成
            }

            /* permute the matrix */
            // 对矩阵进行排列
            for (j = 0; j < n; j++) {
                for (i = colptr[j]; i < colptr[j + 1]; i++) {
                    // 遍历每一列的非零元素
                    rowind[i] = perm[rowind[i]];
                    // 使用 perm 数组重新排列非零元素的行索引
                }
            }
        }
        utime[EQUIL] = SuperLU_timer_() - t0;
        // 记录均衡操作的时间消耗
    }

    if ( mc64==0 && equil ) { /* Only perform equilibration, no row perm */
        // 如果 mc64 为假且需要均衡操作（不需要行排列）
        t0 = SuperLU_timer_();
        // 记录当前时间，用于计时

        /* Compute row and column scalings to equilibrate the matrix A. */
        // 计算行列缩放因子以均衡矩阵 A
        sgsequ(AA, R, C, &rowcnd, &colcnd, &amax, &info1);

        if ( info1 == 0 ) {
            // 如果计算成功
            /* Equilibrate matrix A. */
            // 对矩阵 A 进行均衡化
            slaqgs(AA, R, C, rowcnd, colcnd, amax, equed);
            // 使用 slaqgs 函数进行均衡化操作
            rowequ = strncmp(equed, "R", 1)==0 || strncmp(equed, "B", 1)==0;
            // 检查是否需要行均衡
            colequ = strncmp(equed, "C", 1)==0 || strncmp(equed, "B", 1)==0;
            // 检查是否需要列均衡
        }
        utime[EQUIL] = SuperLU_timer_() - t0;
        // 记录均衡操作的时间消耗
    }

    if ( nofact ) {
        // 如果不需要进行因子分解

        t0 = SuperLU_timer_();
        // 记录当前时间，用于计时

        /*
         * Gnet column permutation vector perm_c[], according to permc_spec:
         *   permc_spec = NATURAL:  natural ordering 
         *   permc_spec = MMD_AT_PLUS_A: minimum degree on structure of A'+A
         *   permc_spec = MMD_ATA:  minimum degree on structure of A'*A
         *   permc_spec = COLAMD:   approximate minimum degree column ordering
         *   permc_spec = MY_PERMC: the ordering already supplied in perm_c[]
         */
        // 根据 permc_spec 对 perm_c[] 进行列排列
        permc_spec = options->ColPerm;
        // 从 options 中获取列排列的指定方式
        if ( permc_spec != MY_PERMC && options->Fact == DOFACT )
            // 如果不是已提供的排列方式且需要进行因子分解
            get_perm_c(permc_spec, AA, perm_c);
            // 调用 get_perm_c 函数获取列排列结果
        utime[COLPERM] = SuperLU_timer_() - t0;
        // 记录列排列操作的时间消耗

        t0 = SuperLU_timer_();
        // 记录当前时间，用于计时

        sp_preorder(options, AA, perm_c, etree, &AC);
        // 对矩阵 AA 进行预处理，得到排列后的矩阵 AC
        utime[ETREE] = SuperLU_timer_() - t0;
        // 记录预处理操作的时间消耗

        /* Compute the LU factorization of A*Pc. */
        // 计算 A*Pc 的 LU 分解
        t0 = SuperLU_timer_();
        // 记录当前时间，用于计时
        sgsitrf(options, &AC, relax, panel_size, etree, work, lwork,
                    perm_c, perm_r, L, U, Glu, stat, info);
        // 调用 sgsitrf 函数进行 LU 分解
        utime[FACT] = SuperLU_timer_() - t0;
        // 记录 LU 分解操作的时间消耗

        if ( lwork == -1 ) {
            // 如果 lwork 为 -1，表示只查询内存需求
            mem_usage->total_needed = *info - A->ncol;
            // 计算总需求内存大小
            return;
            // 直接返回，不执行后续操作
        }
    }
    if ( mc64 ) { /* Fold MC64's perm[] into perm_r[]. */
        NCformat *Astore = AA->Store;
        int_t nnz = Astore->nnz, *rowind = Astore->rowind;
        int *perm_tmp, *iperm;
        if ((perm_tmp = int32Malloc(2*n)) == NULL)
            ABORT("SUPERLU_MALLOC fails for perm_tmp[]");
        iperm = perm_tmp + n;
        for (i = 0; i < n; ++i) perm_tmp[i] = perm_r[perm[i]];
        for (i = 0; i < n; ++i) {
            perm_r[i] = perm_tmp[i];
            iperm[perm[i]] = i;
        }

        /* Restore A's original row indices. */
        for (i = 0; i < nnz; ++i) rowind[i] = iperm[rowind[i]];

        SUPERLU_FREE(perm); /* MC64 permutation */
        SUPERLU_FREE(perm_tmp);
    }


    }

    if ( options->PivotGrowth ) {
        if ( *info > 0 ) return;

        /* Compute the reciprocal pivot growth factor *recip_pivot_growth. */
        *recip_pivot_growth = sPivotGrowth(A->ncol, AA, perm_c, L, U);
    }


    if ( options->ConditionNumber ) {
        /* Estimate the reciprocal of the condition number of A. */
        t0 = SuperLU_timer_();
        if ( notran ) {
            *(unsigned char *)norm = '1';
        } else {
            *(unsigned char *)norm = 'I';
        }
        anorm = slangs(norm, AA);
        sgscon(norm, L, U, anorm, rcond, stat, &info1);
        utime[RCOND] = SuperLU_timer_() - t0;
    }


    if ( nrhs > 0 ) { /* Solve the system */

        /* Scale and permute the right-hand side if equilibration
           and permutation from MC64 were performed. */
        if ( notran ) {
            if ( rowequ ) {
                for (j = 0; j < nrhs; ++j)
                    for (i = 0; i < n; ++i)
                        Bmat[i + j*ldb] *= R[i];
            }
        } else if ( colequ ) {
            for (j = 0; j < nrhs; ++j)
                for (i = 0; i < n; ++i) {
                    Bmat[i + j*ldb] *= C[i];
                }
        }

        /* Compute the solution matrix X. */
        for (j = 0; j < nrhs; j++)  /* Save a copy of the right hand sides */
            for (i = 0; i < B->nrow; i++)
                Xmat[i + j*ldx] = Bmat[i + j*ldb];

        t0 = SuperLU_timer_();
        sgstrs (trant, L, U, perm_c, perm_r, X, stat, &info1);
        utime[SOLVE] = SuperLU_timer_() - t0;

        /* Transform the solution matrix X to a solution of the original
           system. */
        if ( notran ) {
            if ( colequ ) {
                for (j = 0; j < nrhs; ++j)
                    for (i = 0; i < n; ++i) {
                        Xmat[i + j*ldx] *= C[i];
                    }
            }
        } else { /* transposed system */
            if ( rowequ ) {
                for (j = 0; j < nrhs; ++j)
                    for (i = 0; i < A->nrow; ++i) {
                        Xmat[i + j*ldx] *= R[i];
                    }
            }
        }

    } /* end if nrhs > 0 */


    if ( options->ConditionNumber ) {
        /* The matrix is singular to working precision. */
        /* if ( *rcond < slamch_("E") && *info == 0) *info = A->ncol + 1; */
        if ( *rcond < smach("E") && *info == 0) *info = A->ncol + 1;
    }


    if ( nofact ) {
        ilu_sQuerySpace(L, U, mem_usage);
        Destroy_CompCol_Permuted(&AC);
    }
    // 如果 A 指针指向的结构体中的 Stype 字段等于 SLU_NR，则执行以下操作
    if ( A->Stype == SLU_NR ) {
        // 销毁 SuperMatrix 结构体 AA 所指向的存储内容
        Destroy_SuperMatrix_Store(AA);
        // 释放 SuperLU 库中分配的 AA 的内存空间
        SUPERLU_FREE(AA);
    }
}



# 这是一个单独的右大括号 '}'，用于结束一个代码块或数据结构的定义。
```