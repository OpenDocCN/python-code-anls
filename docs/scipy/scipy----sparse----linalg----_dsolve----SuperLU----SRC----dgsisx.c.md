# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dgsisx.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file dgsisx.c
 * \brief Computes an approximate solutions of linear equations A*X=B or A'*X=B
 *
 * <pre>
 * -- SuperLU routine (version 4.2) --
 * Lawrence Berkeley National Laboratory.
 * November, 2010
 * August, 2011
 * </pre>
 */
#include "slu_ddefs.h"

/**
 * @brief Computes an approximate solution of linear equations A*X=B or A'*X=B
 *
 * This function computes an approximate solution to a system of linear equations
 * using the SuperLU library routines.
 *
 * @param options SuperLU options controlling the solving process.
 * @param A Pointer to the coefficient matrix A.
 * @param perm_c Column permutation vector.
 * @param perm_r Row permutation vector.
 * @param etree Elimination tree of A.
 * @param equed Specifies whether and how to equilibrate the system.
 * @param R Scale factors for the rows of A (if equilibration is done).
 * @param C Scale factors for the columns of A (if equilibration is done).
 * @param L Pointer to the lower triangular factor L from the LU factorization.
 * @param U Pointer to the upper triangular factor U from the LU factorization.
 * @param work Workspace array.
 * @param lwork Size of the workspace array.
 * @param B Right-hand side matrix B.
 * @param X Solution matrix X.
 * @param recip_pivot_growth Reciprocal of pivot growth factor.
 * @param rcond Estimate of the reciprocal condition number of A.
 * @param Glu Pointer to the global LU data structure.
 * @param mem_usage Memory usage statistics.
 * @param stat SuperLU statistics.
 * @param info Output status indicator.
 */
void
dgsisx(superlu_options_t *options, SuperMatrix *A, int *perm_c, int *perm_r,
       int *etree, char *equed, double *R, double *C,
       SuperMatrix *L, SuperMatrix *U, void *work, int_t lwork,
       SuperMatrix *B, SuperMatrix *X,
       double *recip_pivot_growth, double *rcond,
       GlobalLU_t *Glu, mem_usage_t *mem_usage, SuperLUStat_t *stat, int_t *info)
{
    DNformat  *Bstore, *Xstore;
    double    *Bmat, *Xmat;
    int       ldb, ldx, nrhs, n;
    SuperMatrix *AA;/* A in SLU_NC format used by the factorization routine.*/
    SuperMatrix AC; /* Matrix postmultiplied by Pc */
    int       colequ, equil, nofact, notran, rowequ, permc_spec, mc64;
    trans_t   trant;
    char      norm[1];
    int_t     i, j;
    double    amax, anorm, bignum, smlnum, colcnd, rowcnd, rcmax, rcmin;
    int       relax, panel_size, info1;
    double    t0;      /* temporary time */
    double    *utime;

    int *perm = NULL; /* permutation returned from MC64 */

    /* External functions */
    extern double dlangs(char *, SuperMatrix *);

    Bstore = B->Store;
    Xstore = X->Store;
    Bmat   = Bstore->nzval;
    Xmat   = Xstore->nzval;
    ldb    = Bstore->lda;
    ldx    = Xstore->lda;
    nrhs   = B->ncol;
    n      = B->nrow;

    *info = 0;
    nofact = (options->Fact != FACTORED);
    equil = (options->Equil == YES);
    notran = (options->Trans == NOTRANS);
    mc64 = (options->RowPerm == LargeDiag_MC64);

    if ( nofact ) {
        *(unsigned char *)equed = 'N';
        rowequ = FALSE;
        colequ = FALSE;
    } else {
        rowequ = strncmp(equed, "R", 1)==0 || strncmp(equed, "B", 1)==0;
        colequ = strncmp(equed, "C", 1)==0 || strncmp(equed, "B", 1)==0;
        smlnum = dmach("Safe minimum");  /* lamch_("Safe minimum"); */
        bignum = 1. / smlnum;
    }

    /* Test the input parameters */
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
              A->Dtype != SLU_D || A->Mtype != SLU_GE )
    {
        *info = -2;
    }
    else if ( options->Fact == FACTORED &&
              !(rowequ || colequ || strncmp(equed, "N", 1)==0) )
    {
        *info = -3;
    }
}
    *info = -6;
    else {


    // 设置 info 的初始值为 -6
    *info = -6;
    // 进入 else 分支，开始处理条件语句
    else {



    if (rowequ) {


    // 如果 rowequ 为真，则执行以下代码块
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


        // 初始化 rcmin 和 rcmax
        rcmin = bignum;
        rcmax = 0.;
        // 遍历 R 数组，更新 rcmin 和 rcmax
        for (j = 0; j < A->nrow; ++j) {
            rcmin = SUPERLU_MIN(rcmin, R[j]);
            rcmax = SUPERLU_MAX(rcmax, R[j]);
        }
        // 检查 rcmin 的值是否小于等于 0，如果是则设置 info 为 -7
        if (rcmin <= 0.) *info = -7;
        // 否则计算 rowcnd 的值
        else if ( A->nrow > 0)
            rowcnd = SUPERLU_MAX(rcmin,smlnum) / SUPERLU_MIN(rcmax,bignum);
        else
            rowcnd = 1.;
    }



    if (colequ && *info == 0) {


    // 如果 colequ 为真且 info 等于 0，则执行以下代码块
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


        // 初始化 rcmin 和 rcmax
        rcmin = bignum;
        rcmax = 0.;
        // 遍历 C 数组，更新 rcmin 和 rcmax
        for (j = 0; j < A->nrow; ++j) {
            rcmin = SUPERLU_MIN(rcmin, C[j]);
            rcmax = SUPERLU_MAX(rcmax, C[j]);
        }
        // 检查 rcmin 的值是否小于等于 0，如果是则设置 info 为 -8
        if (rcmin <= 0.) *info = -8;
        // 否则计算 colcnd 的值
        else if (A->nrow > 0)
            colcnd = SUPERLU_MAX(rcmin,smlnum) / SUPERLU_MIN(rcmax,bignum);
        else
            colcnd = 1.;
    }



    if (*info == 0) {


    // 如果 info 等于 0，则执行以下代码块
    if (*info == 0) {



        if ( lwork < -1 ) *info = -12;
        else if ( B->ncol < 0 || Bstore->lda < SUPERLU_MAX(0, A->nrow) ||
              B->Stype != SLU_DN || B->Dtype != SLU_D || 
              B->Mtype != SLU_GE )
        *info = -13;
        else if ( X->ncol < 0 || Xstore->lda < SUPERLU_MAX(0, A->nrow) ||
              (B->ncol != 0 && B->ncol != X->ncol) ||
              X->Stype != SLU_DN ||
              X->Dtype != SLU_D || X->Mtype != SLU_GE )
        *info = -14;
    }


        // 检查 lwork 的值是否小于 -1，如果是则设置 info 为 -12
        if ( lwork < -1 ) *info = -12;
        // 否则检查 B 矩阵的各种属性，设置不符合条件时的 info 值
        else if ( B->ncol < 0 || Bstore->lda < SUPERLU_MAX(0, A->nrow) ||
                  B->Stype != SLU_DN || B->Dtype != SLU_D || 
                  B->Mtype != SLU_GE )
            *info = -13;
        // 检查 X 矩阵的各种属性，设置不符合条件时的 info 值
        else if ( X->ncol < 0 || Xstore->lda < SUPERLU_MAX(0, A->nrow) ||
                  (B->ncol != 0 && B->ncol != X->ncol) ||
                  X->Stype != SLU_DN ||
                  X->Dtype != SLU_D || X->Mtype != SLU_GE )
            *info = -14;
    }



    }
    if (*info != 0) {
    int ii = -(*info);
    input_error("dgsisx", &ii);
    return;
    }


    // 如果 info 不等于 0，则执行以下代码块
    if (*info != 0) {
        // 设置 ii 为 info 的负值
        int ii = -(*info);
        // 调用 input_error 函数处理错误
        input_error("dgsisx", &ii);
        // 返回函数，结束执行
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


    // 将 stat->utime 的值赋给 utime
    utime = stat->utime;



    /* Convert A to SLU_NC format when necessary. */
    if ( A->Stype == SLU_NR ) {
    NRformat *Astore = A->Store;
    AA = (SuperMatrix *) SUPERLU_MALLOC( sizeof(SuperMatrix) );
    dCreate_CompCol_Matrix(AA, A->ncol, A->nrow, Astore->nnz,
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


    // 根据 A 的存储类型转换为 SLU_NC 格式（如果必要）
    if ( A->Stype == SLU_NR ) {
        // 获取 A 的 NRformat 存储结构
        NRformat *Astore = A->Store;
        // 分配 AA 的内存空间
        AA = (SuperMatrix *) SUPERLU_MALLOC( sizeof(SuperMatrix) );
        // 创建压缩列矩阵 AA
        dCreate_CompCol_Matrix(AA, A->ncol, A->nrow, Astore->nnz,
                       Astore->nzval, Astore->colind, Astore->rowptr,
                       SLU_NC, A->Dtype, A->Mtype);
        // 根据 notran 的值确定 trant 的值
        if ( notran ) { /* 反转置参数 */
            trant = TRANS;
            notran = 0;
        } else {
            trant = NOTRANS;
            notran = 1;
        }
    } else { /* A->Stype == SLU_NC */
        // 获取 options 中的 Trans 属性作为 trant
        trant = options->Trans;
        // AA 直接指向 A
        AA = A;
    }



    if ( nofact ) {
    register int i, j;
    NCformat *Astore = AA->Store;
    int_t nnz = Astore->nnz;
    int
    if ( mc64 ) {
        // 开始计时
        t0 = SuperLU_timer_();
        // 分配存储空间给 perm[]
        if ((perm = int32Malloc(n)) == NULL)
            ABORT("SUPERLU_MALLOC fails for perm[]");

        // 调用 MC64 进行列重排序和行交换
        info1 = dldperm(5, n, nnz, colptr, rowind, nzval, perm, R, C);

        if (info1 != 0) { /* MC64 失败，稍后调用 dgsequ() */
            mc64 = 0;
            // 释放 perm[]
            SUPERLU_FREE(perm);
            perm = NULL;
        } else {
            if ( equil ) {
                // 执行行列均衡化
                rowequ = colequ = 1;
                for (i = 0; i < n; i++) {
                    // 对 R 和 C 进行指数运算
                    R[i] = exp(R[i]);
                    C[i] = exp(C[i]);
                }
                // 缩放矩阵
                for (j = 0; j < n; j++) {
                    for (i = colptr[j]; i < colptr[j + 1]; i++) {
                        nzval[i] *= R[rowind[i]] * C[j];
                    }
                }
                // 设置均衡类型为 'B'
                *equed = 'B';
            }

            // 对矩阵进行置换
            for (j = 0; j < n; j++) {
                for (i = colptr[j]; i < colptr[j + 1]; i++) {
                    // 置换行索引
                    rowind[i] = perm[rowind[i]];
                }
            }
        }
        // 记录均衡化和置换的时间
        utime[EQUIL] = SuperLU_timer_() - t0;
    }

    if ( mc64==0 && equil ) { /* 只执行均衡化，不进行行置换 */
        // 开始计时
        t0 = SuperLU_timer_();
        /* 计算行和列的缩放，使矩阵 A 均衡化 */
        dgsequ(AA, R, C, &rowcnd, &colcnd, &amax, &info1);

        if ( info1 == 0 ) {
            /* 对矩阵 A 进行均衡化 */
            dlaqgs(AA, R, C, rowcnd, colcnd, amax, equed);
            // 检查均衡类型
            rowequ = strncmp(equed, "R", 1)==0 || strncmp(equed, "B", 1)==0;
            colequ = strncmp(equed, "C", 1)==0 || strncmp(equed, "B", 1)==0;
        }
        // 记录均衡化时间
        utime[EQUIL] = SuperLU_timer_() - t0;
    }
    }


    if ( nofact ) {
        // 开始计时
        t0 = SuperLU_timer_();
        /*
         * 根据 permc_spec 获取列置换向量 perm_c[]：
         *   permc_spec = NATURAL: 自然顺序
         *   permc_spec = MMD_AT_PLUS_A: A'+A 的最小度排序
         *   permc_spec = MMD_ATA: A'*A 的最小度排序
         *   permc_spec = COLAMD: 近似最小度列排序
         *   permc_spec = MY_PERMC: 已提供的 perm_c[] 排序
         */
        permc_spec = options->ColPerm;
        // 如果 permc_spec 不是 MY_PERMC 并且 Fact 是 DOFACT，则计算 perm_c[]
        if ( permc_spec != MY_PERMC && options->Fact == DOFACT )
            get_perm_c(permc_spec, AA, perm_c);
        // 记录列置换时间
        utime[COLPERM] = SuperLU_timer_() - t0;

        // 开始计时
        t0 = SuperLU_timer_();
        // 预处理，计算 etree 和 AC
        sp_preorder(options, AA, perm_c, etree, &AC);
        // 记录 etree 计算时间
        utime[ETREE] = SuperLU_timer_() - t0;

        /* 计算矩阵 A*Pc 的 LU 分解 */
        t0 = SuperLU_timer_();
        dgsitrf(options, &AC, relax, panel_size, etree, work, lwork,
                    perm_c, perm_r, L, U, Glu, stat, info);
        // 记录 LU 分解时间
        utime[FACT] = SuperLU_timer_() - t0;

        // 如果 lwork 为 -1，则只返回所需内存大小
        if ( lwork == -1 ) {
            mem_usage->total_needed = *info - A->ncol;
            return;
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
        *recip_pivot_growth = dPivotGrowth(A->ncol, AA, perm_c, L, U);
    }


    if ( options->ConditionNumber ) {
        /* Estimate the reciprocal of the condition number of A. */
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
        dgstrs (trant, L, U, perm_c, perm_r, X, stat, &info1);
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
        /* if ( *rcond < dlamch_("E") && *info == 0) *info = A->ncol + 1; */
        if ( *rcond < dmach("E") && *info == 0) *info = A->ncol + 1;
    }


    if ( nofact ) {
        ilu_dQuerySpace(L, U, mem_usage);
        Destroy_CompCol_Permuted(&AC);
    }
    # 如果 A 的 Stype 属性等于 SLU_NR，执行以下操作：
    if ( A->Stype == SLU_NR ) {
        # 销毁 AA 指向的 SuperMatrix 对象及其存储
        Destroy_SuperMatrix_Store(AA);
        # 释放 AA 指针占用的内存空间
        SUPERLU_FREE(AA);
    }
}
```