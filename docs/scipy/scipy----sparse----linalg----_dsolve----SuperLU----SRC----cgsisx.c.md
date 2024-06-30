# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\cgsisx.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file cgsisx.c
 * \brief Computes an approximate solutions of linear equations A*X=B or A'*X=B
 *
 * <pre>
 * -- SuperLU routine (version 4.2) --
 * Lawrence Berkeley National Laboratory.
 * November, 2010
 * August, 2011
 * </pre>
 */
#include "slu_cdefs.h"

void
cgsisx(superlu_options_t *options, SuperMatrix *A, int *perm_c, int *perm_r,
       int *etree, char *equed, float *R, float *C,
       SuperMatrix *L, SuperMatrix *U, void *work, int_t lwork,
       SuperMatrix *B, SuperMatrix *X,
       float *recip_pivot_growth, float *rcond,
       GlobalLU_t *Glu, mem_usage_t *mem_usage, SuperLUStat_t *stat, int_t *info)
{
    DNformat  *Bstore, *Xstore;  // B 和 X 的存储格式
    singlecomplex    *Bmat, *Xmat;  // B 和 X 的实际数据
    int       ldb, ldx, nrhs, n;  // ldb 和 ldx 是 B 和 X 的 leading dimension，nrhs 是右侧向量的数量，n 是矩阵的大小
    SuperMatrix *AA;  // 用于因式分解例程的 SLU_NC 格式中的 A
    SuperMatrix AC;  // 矩阵后乘以 Pc
    int       colequ, equil, nofact, notran, rowequ, permc_spec, mc64;  // 一些标志和选项
    trans_t   trant;  // 转置类型
    char      norm[1];  // 范数类型
    int_t     i, j;  // 循环变量
    float    amax, anorm, bignum, smlnum, colcnd, rowcnd, rcmax, rcmin;  // 浮点数参数
    int       relax, panel_size, info1;  // 一些参数和信息

    double    t0;      /* temporary time */
    double    *utime;  // 时间相关

    int *perm = NULL; /* permutation returned from MC64 */  // 来自 MC64 返回的置换向量

    /* External functions */
    extern float clangs(char *, SuperMatrix *);  // 外部函数声明

    Bstore = B->Store;  // 获取 B 的存储结构
    Xstore = X->Store;  // 获取 X 的存储结构
    Bmat   = Bstore->nzval;  // 获取 B 的非零值数组
    Xmat   = Xstore->nzval;  // 获取 X 的非零值数组
    ldb    = Bstore->lda;  // 获取 B 的 leading dimension
    ldx    = Xstore->lda;  // 获取 X 的 leading dimension
    nrhs   = B->ncol;  // 获取右侧向量的数量
    n      = B->nrow;  // 获取矩阵的大小

    *info = 0;  // 初始化 info
    nofact = (options->Fact != FACTORED);  // 检查是否需要因式分解
    equil = (options->Equil == YES);  // 检查是否进行均衡处理
    notran = (options->Trans == NOTRANS);  // 检查是否是不转置运算
    mc64 = (options->RowPerm == LargeDiag_MC64);  // 检查是否使用 MC64 进行行置换
    if ( nofact ) {
        *(unsigned char *)equed = 'N';  // 如果不需要因式分解，设置 equed 为 'N'
        rowequ = FALSE;  // 行均衡标志为假
        colequ = FALSE;  // 列均衡标志为假
    } else {
        rowequ = strncmp(equed, "R", 1)==0 || strncmp(equed, "B", 1)==0;  // 检查行均衡标志
        colequ = strncmp(equed, "C", 1)==0 || strncmp(equed, "B", 1)==0;  // 检查列均衡标志
        smlnum = smach("Safe minimum");  /* lamch_("Safe minimum"); */  // 获取安全最小值
        bignum = 1. / smlnum;  // 计算大数值
    }

    /* Test the input parameters */
    if (options->Fact != DOFACT && options->Fact != SamePattern &&
        options->Fact != SamePattern_SameRowPerm &&
        options->Fact != FACTORED &&
        options->Trans != NOTRANS && options->Trans != TRANS && 
        options->Trans != CONJ &&
        options->Equil != NO && options->Equil != YES)
        *info = -1;  // 检查输入参数的合法性
    else if ( A->nrow != A->ncol || A->nrow < 0 ||
              (A->Stype != SLU_NC && A->Stype != SLU_NR) ||
              A->Dtype != SLU_C || A->Mtype != SLU_GE )
        *info = -2;  // 检查矩阵 A 的属性
    else if ( options->Fact == FACTORED &&
              !(rowequ || colequ || strncmp(equed, "N", 1)==0) )
    *info = -6;
    else {

// 设置 info 的初始值为 -6，表示有错误发生。如果没有错误发生，则继续执行下面的逻辑。


    if (rowequ) {

// 如果 rowequ 为真，则执行以下逻辑，用于处理行等式约束。


        rcmin = bignum;
        rcmax = 0.;
        for (j = 0; j < A->nrow; ++j) {

// 初始化 rcmin 为一个大数值 bignum，rcmax 为 0。遍历矩阵 A 的每一行 j。


        rcmin = SUPERLU_MIN(rcmin, R[j]);
        rcmax = SUPERLU_MAX(rcmax, R[j]);
        }

// 更新 rcmin 为 R[j] 和当前 rcmin 中的较小值，rcmax 为 R[j] 和当前 rcmax 中的较大值。


        if (rcmin <= 0.) *info = -7;
        else if ( A->nrow > 0)
        rowcnd = SUPERLU_MAX(rcmin,smlnum) / SUPERLU_MIN(rcmax,bignum);
        else rowcnd = 1.;
    }

// 如果 rcmin 小于等于 0，则将 info 设为 -7；否则计算行条件数 rowcnd。


    if (colequ && *info == 0) {

// 如果 colequ 为真，并且 info 等于 0，则执行以下逻辑，用于处理列等式约束。


        rcmin = bignum;
        rcmax = 0.;
        for (j = 0; j < A->nrow; ++j) {

// 初始化 rcmin 为一个大数值 bignum，rcmax 为 0。遍历矩阵 A 的每一列 j。


        rcmin = SUPERLU_MIN(rcmin, C[j]);
        rcmax = SUPERLU_MAX(rcmax, C[j]);
        }

// 更新 rcmin 为 C[j] 和当前 rcmin 中的较小值，rcmax 为 C[j] 和当前 rcmax 中的较大值。


        if (rcmin <= 0.) *info = -8;
        else if (A->nrow > 0)
        colcnd = SUPERLU_MAX(rcmin,smlnum) / SUPERLU_MIN(rcmax,bignum);
        else colcnd = 1.;
    }

// 如果 rcmin 小于等于 0，则将 info 设为 -8；否则计算列条件数 colcnd。


    if (*info == 0) {

// 如果 info 仍然等于 0，则执行以下逻辑，检查各种参数是否满足要求。


        if ( lwork < -1 ) *info = -12;
        else if ( B->ncol < 0 || Bstore->lda < SUPERLU_MAX(0, A->nrow) ||
              B->Stype != SLU_DN || B->Dtype != SLU_C || 
              B->Mtype != SLU_GE )
        *info = -13;
        else if ( X->ncol < 0 || Xstore->lda < SUPERLU_MAX(0, A->nrow) ||
              (B->ncol != 0 && B->ncol != X->ncol) ||
              X->Stype != SLU_DN ||
              X->Dtype != SLU_C || X->Mtype != SLU_GE )
        *info = -14;
    }

// 检查 lwork 是否小于 -1，以及矩阵 B、X 的各种属性是否满足要求，如果不满足则设定相应的 info 值为负数。


    }
    if (*info != 0) {
    int ii = -(*info);
    input_error("cgsisx", &ii);
    return;
    }

// 如果 info 不等于 0，则表示出现错误，调用 input_error 函数报告错误并返回。


    /* Initialization for factor parameters */
    panel_size = sp_ienv(1);
    relax      = sp_ienv(2);

// 初始化因子化参数 panel_size 和 relax，使用 sp_ienv 函数从环境中获取相应的值。


    utime = stat->utime;

// 将统计结构体 stat 中的 utime 赋值给 utime 变量。


    /* Convert A to SLU_NC format when necessary. */
    if ( A->Stype == SLU_NR ) {

// 当矩阵 A 的存储类型为 SLU_NR 时，将其转换为 SLU_NC 格式。


    NRformat *Astore = A->Store;
    AA = (SuperMatrix *) SUPERLU_MALLOC( sizeof(SuperMatrix) );
    cCreate_CompCol_Matrix(AA, A->ncol, A->nrow, Astore->nnz,
                   Astore->nzval, Astore->colind, Astore->rowptr,
                   SLU_NC, A->Dtype, A->Mtype);

// 获取 A 的 NR 格式存储结构 Astore，动态分配空间创建 AA 矩阵，并将 A 转换为 SLU_NC 格式。


    if ( notran ) { /* Reverse the transpose argument. */
        trant = TRANS;
        notran = 0;
    } else {
        trant = NOTRANS;
        notran = 1;
    }

// 根据 notran 的值设置 trant，用于控制矩阵转置的方式。


    } else { /* A->Stype == SLU_NC */
    trant = options->Trans;
    AA = A;
    }

// 当 A 的存储类型为 SLU_NC 时，直接使用 A，并根据选项设置 trant。


    if ( nofact ) {
    register int i, j;
    NCformat *Astore = AA->Store;
    int_t nnz = Astore->nnz;
    int_t *colptr = Astore->colptr;
    int_t *rowind = Astore->rowind;
    singlecomplex *nzval = (singlecomplex *)Astore->nzval;

// 如果 nofact 为真，则执行因子化前的初始化操作，并获取 AA 的 NC 格式存储结构。
    if ( mc64 ) {
        // 记录当前时间，用于计时
        t0 = SuperLU_timer_();
        // 分配空间给排列数组perm，如果失败则终止程序
        if ((perm = int32Malloc(n)) == NULL)
            ABORT("SUPERLU_MALLOC fails for perm[]");

        // 调用cldperm函数进行列置换和行置换，返回错误码info1
        info1 = cldperm(5, n, nnz, colptr, rowind, nzval, perm, R, C);

        // 如果cldperm失败，设置mc64为0，并释放perm的内存空间
        if (info1 != 0) {
            mc64 = 0;
            SUPERLU_FREE(perm);
            perm = NULL;
        } else {
            // 如果成功，并且需要均衡化
            if ( equil ) {
                rowequ = colequ = 1;
                // 对R和C数组中的每个元素应用指数函数
                for (i = 0; i < n; i++) {
                    R[i] = exp(R[i]);
                    C[i] = exp(C[i]);
                }
                /* scale the matrix */
                // 对矩阵进行缩放
                for (j = 0; j < n; j++) {
                    for (i = colptr[j]; i < colptr[j + 1]; i++) {
                        // 使用R和C数组的乘积对矩阵元素进行缩放
                        cs_mult(&nzval[i], &nzval[i], R[rowind[i]] * C[j]);
                    }
                }
                // 设置equed为'B'
                *equed = 'B';
            }

            // 执行列置换
            for (j = 0; j < n; j++) {
                for (i = colptr[j]; i < colptr[j + 1]; i++) {
                    // 使用排列数组perm对rowind进行重新排列
                    rowind[i] = perm[rowind[i]];
                }
            }
        }
        // 记录均衡化时间
        utime[EQUIL] = SuperLU_timer_() - t0;
    }

    if ( mc64==0 && equil ) {
        // 仅执行均衡化，不进行行置换
        t0 = SuperLU_timer_();
        // 计算行和列的缩放因子，以均衡矩阵AA
        cgsequ(AA, R, C, &rowcnd, &colcnd, &amax, &info1);

        // 如果计算成功
        if ( info1 == 0 ) {
            // 均衡矩阵AA
            claqgs(AA, R, C, rowcnd, colcnd, amax, equed);
            // 检查equed的值以确定是否进行行和列的均衡
            rowequ = strncmp(equed, "R", 1)==0 || strncmp(equed, "B", 1)==0;
            colequ = strncmp(equed, "C", 1)==0 || strncmp(equed, "B", 1)==0;
        }
        // 记录均衡化时间
        utime[EQUIL] = SuperLU_timer_() - t0;
    }

    if ( nofact ) {
        // 记录当前时间，用于计时
        t0 = SuperLU_timer_();
        /*
         * 根据permc_spec指定的方式对列进行排列，具体方式包括：
         *   permc_spec = NATURAL: 自然顺序
         *   permc_spec = MMD_AT_PLUS_A: A'+A结构的最小度排序
         *   permc_spec = MMD_ATA: A'*A结构的最小度排序
         *   permc_spec = COLAMD: 近似最小度列排序
         *   permc_spec = MY_PERMC: 已提供的自定义排序在perm_c[]中
         */
        permc_spec = options->ColPerm;
        // 如果不是自定义排序且需要进行因子分解
        if ( permc_spec != MY_PERMC && options->Fact == DOFACT )
            // 获取列置换向量perm_c[]
            get_perm_c(permc_spec, AA, perm_c);
        // 记录列置换时间
        utime[COLPERM] = SuperLU_timer_() - t0;

        // 记录当前时间，用于计时
        t0 = SuperLU_timer_();
        // 预处理，得到因子分解的顺序etree，并且得到预处理矩阵AC
        sp_preorder(options, AA, perm_c, etree, &AC);
        // 记录etree计算时间
        utime[ETREE] = SuperLU_timer_() - t0;

        // 计算A*Pc的LU因子分解
        t0 = SuperLU_timer_();
        cgsitrf(options, &AC, relax, panel_size, etree, work, lwork,
                perm_c, perm_r, L, U, Glu, stat, info);
        // 记录因子分解时间
        utime[FACT] = SuperLU_timer_() - t0;

        // 如果lwork为-1，表示只需计算内存使用情况，不进行因子分解
        if ( lwork == -1 ) {
            // 计算需要的总内存量
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


        /* 折叠MC64的perm[]到perm_r[]中 */

        /* 获取矩阵AA的存储结构 */
        NCformat *Astore = AA->Store;
        /* 获取非零元个数和行索引数组 */
        int_t nnz = Astore->nnz, *rowind = Astore->rowind;
        /* 分配临时数组perm_tmp和iperm */
        int *perm_tmp, *iperm;
        if ((perm_tmp = int32Malloc(2*n)) == NULL)
        ABORT("SUPERLU_MALLOC fails for perm_tmp[]");
        iperm = perm_tmp + n;
        /* 使用perm_r和perm数组更新perm_tmp */
        for (i = 0; i < n; ++i) perm_tmp[i] = perm_r[perm[i]];
        /* 更新perm_r和iperm数组 */
        for (i = 0; i < n; ++i) {
        perm_r[i] = perm_tmp[i];
        iperm[perm[i]] = i;
        }
    # 调用函数 ilu_cQuerySpace，计算 ILU 分解的空间需求，传入参数 L, U, mem_usage
    ilu_cQuerySpace(L, U, mem_usage);
    
    # 销毁类型为 CompCol_Permuted 的稀疏矩阵 AC
    Destroy_CompCol_Permuted(&AC);
    }
    
    # 如果矩阵 A 的存储类型为 SLU_NR（非规则存储），执行以下操作
    if ( A->Stype == SLU_NR ) {
        # 销毁 SuperMatrix_Store 结构体 AA
        Destroy_SuperMatrix_Store(AA);
        # 释放 SuperLU 的内存 AA 所占用的空间
        SUPERLU_FREE(AA);
    }
}



# 这行代码是一个单独的右大括号 '}'，用于结束某个代码块或函数定义。
# 在大多数编程语言中，右大括号用于标记代码块的结束。
```