# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\sgsitrf.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file sgsitrf.c
 * \brief Computes an ILU factorization of a general sparse matrix
 *
 * <pre>
 * -- SuperLU routine (version 4.1) --
 * Lawrence Berkeley National Laboratory.
 * June 30, 2009
 *
 * </pre>
 */

#include "slu_sdefs.h"

#ifdef DEBUG
int num_drop_L;
#endif

void
sgsitrf(superlu_options_t *options, SuperMatrix *A, int relax, int panel_size,
    int *etree, void *work, int_t lwork, int *perm_c, int *perm_r,
    SuperMatrix *L, SuperMatrix *U, 
        GlobalLU_t *Glu, /* persistent to facilitate multiple factorizations */
    SuperLUStat_t *stat, int_t *info)
{
    /* Local working arrays */

    // Astore is a pointer to the compressed column format of matrix A
    NCPformat *Astore;

    // iperm_r is the inverse permutation of perm_r used under specific conditions
    int       *iperm_r = NULL;

    // iperm_c is the inverse permutation of perm_c
    int       *iperm_c;

    // swap and iswap are arrays used for row permutations during factorization
    int       *swap, *iswap;

    // iwork is an array for integer workspace
    int       *iwork;

    // swork is an array for float workspace
    float     *swork;

    // segrep, repfnz, parent, xplore are arrays used in symbolic factorization
    int       *segrep, *repfnz, *parent;
    int_t     *xplore;

    // panel_lsub is used in panel factorization
    int       *panel_lsub;

    // marker and marker_relax are arrays used for marking during factorization
    int       *marker;
    int       *marker_relax;

    // dense and tempv are arrays used in numerical factorization
    float     *dense, *tempv;

    // relax_end and relax_fsupc are arrays related to relaxed supernodes
    int       *relax_end, *relax_fsupc;

    // a, asub, xa_begin, xa_end are arrays related to matrix A
    float     *a;
    int_t     *asub;
    int_t     *xa_begin, *xa_end;

    // xsup and supno are arrays related to supernodes
    int       *xsup, *supno;

    // xlsub, xlusup, xusub are arrays related to supernode columns and rows
    int_t     *xlsub, *xlusup, *xusub;

    // nzlumax is the maximum number of nonzeros in the LU factors
    int_t     nzlumax;

    // amax is an array used to store maximum absolute values in factors
    float     *amax;

    // drop_sum is a scalar used in dropping strategy
    float     drop_sum;

    // alpha and omega are parameters used in specific factorization techniques
    float     alpha, omega;

    // swork2 is an array used in secondary dropping rule
    float     *swork2;

    /* Local scalars */

    // fact is the factorization strategy
    fact_t    fact = options->Fact;

    // diag_pivot_thresh is the threshold for diagonal pivoting
    double    diag_pivot_thresh = options->DiagPivotThresh;

    // drop_tol is the tolerance for dropping small elements
    double    drop_tol = options->ILU_DropTol;

    // fill_ini is the initial fill-in tolerance
    double    fill_ini = options->ILU_FillTol;

    // gamma is the fill factor for ILU
    double    gamma = options->ILU_FillFactor;

    // drop_rule is the rule used for dropping strategy
    int       drop_rule = options->ILU_DropRule;

    // milu is the variant of incomplete LU factorization
    milu_t    milu = options->ILU_MILU;

    // fill_tol is the fill tolerance adjusted based on drop_tol and gamma
    double    fill_tol;

    // pivrow is the pivotal row number in the original matrix A
    int       pivrow;

    // nseg1 and nseg are counts of segments in U-columns during factorization
    int       nseg1, nseg;

    // jcol, jj, kcol, icol are loop indices for columns and supernodes
    register int jcol, jj;
    register int kcol;
    register int icol;

    // i, k, iinfo are loop and informational indices
    int_t     i, k, iinfo;

    // m, n, min_mn, jsupno, fsupc are matrix dimensions and supernode indices
    int       m, n, min_mn, jsupno, fsupc;

    // new_next, nextlu, nextu are pointers for structure updates
    int_t     new_next, nextlu, nextu;

    // w_def is an upper bound on panel width
    int       w_def;


（注释未完，继续）
    int       usepr, iperm_r_allocated = 0;
    int_t     nnzL, nnzU;
    int       *panel_histo = stat->panel_histo;
    flops_t   *ops = stat->ops;

    int       last_drop;/* the last column which the dropping rules applied */
    int       quota;
    int       nnzAj;    /* number of nonzeros in A(:,1:j) */
    int       nnzLj, nnzUj;
    double    tol_L = drop_tol, tol_U = drop_tol;
    float zero = 0.0;
    float one = 1.0;

    /* Executable */
    // 初始化返回信息为0
    iinfo    = 0;
    // 获取矩阵的行数和列数
    m         = A->nrow;
    n         = A->ncol;
    // 计算行列数的最小值
    min_mn   = SUPERLU_MIN(m, n);
    // 获取矩阵A的存储结构
    Astore   = A->Store;
    // 获取矩阵A的非零元素数组
    a         = Astore->nzval;
    // 获取矩阵A的行索引数组
    asub     = Astore->rowind;
    // 获取矩阵A的列起始位置数组
    xa_begin = Astore->colbeg;
    // 获取矩阵A的列结束位置数组
    xa_end   = Astore->colend;

    /* Allocate storage common to the factor routines */
    // 初始化分配存储空间并进行LU分解的内存设置
    *info = sLUMemInit(fact, work, lwork, m, n, Astore->nnz, panel_size,
               gamma, L, U, Glu, &iwork, &swork);
    // 如果初始化失败，直接返回
    if ( *info ) return;

    // 获取全局LU分解的超节点信息
    xsup    = Glu->xsup;
    supno   = Glu->supno;
    xlsub   = Glu->xlsub;
    xlusup  = Glu->xlusup;
    xusub   = Glu->xusub;

    int_t *xprune;
    // 设置整数工作空间的数据结构
    SetIWork(m, n, panel_size, iwork, &segrep, &parent, &xplore,
         &repfnz, &panel_lsub, &xprune, &marker);
    // 分配与ILU相关的松弛标记
    marker_relax = int32Malloc(m);
    // 释放xprune数组的内存空间（在ILU中未使用）
    SUPERLU_FREE(xprune);

    // 设置实数工作空间的数据结构
    sSetRWork(m, panel_size, swork, &dense, &tempv);

    // 根据是否保持相同的行置换模式，设置usepr标志
    usepr = (fact == SamePattern_SameRowPerm);
    if ( usepr ) {
    /* Compute the inverse of perm_r */
    // 计算perm_r的逆置换数组iperm_r
    iperm_r = (int *) int32Malloc(m);
    for (k = 0; k < m; ++k) iperm_r[perm_r[k]] = k;
    iperm_r_allocated = 1;
    }

    // 计算列置换数组perm_c的逆置换数组iperm_c
    iperm_c = (int *) int32Malloc(n);
    for (k = 0; k < n; ++k) iperm_c[perm_c[k]] = k;
    // 分配用于交换列索引的数组
    swap = (int *)intMalloc(n);
    for (k = 0; k < n; k++) swap[k] = iperm_c[k];
    // 分配用于交换列索引的逆数组
    iswap = (int *)intMalloc(n);
    for (k = 0; k < n; k++) iswap[k] = perm_c[k];
    // 分配用于存储panel_size个最大元素的数组
    amax = (float *) SUPERLU_MALLOC(panel_size * sizeof(float));
    // 根据是否启用次要元素丢弃规则，分配实数工作空间swork2
    if (drop_rule & DROP_SECONDARY)
    swork2 = SUPERLU_MALLOC(n * sizeof(float));
    else
    swork2 = NULL;

    // 初始化非零元素计数
    nnzAj = 0;
    nnzLj = 0;
    nnzUj = 0;
    // 计算应用丢弃规则的最后一列
    last_drop = SUPERLU_MAX(min_mn - 2 * sp_ienv(7), (int)(min_mn * 0.95));
    // 计算alpha参数值
    alpha = pow((double)n, -1.0 / options->ILU_MILU_Dim);

    /* Identify relaxed snodes */
    // 标识松弛超节点
    relax_end = (int *) intMalloc(n);
    relax_fsupc = (int *) intMalloc(n);
    // 根据对称模式设置松弛超节点
    if ( options->SymmetricMode == YES )
    ilu_heap_relax_snode(n, etree, relax, marker, relax_end, relax_fsupc);
    else
    ilu_relax_snode(n, etree, relax, marker, relax_end, relax_fsupc);

    // 将perm_r数组标记为未使用状态
    ifill (perm_r, m, EMPTY);
    // 将marker数组初始化为空标记
    ifill (marker, m * NO_MARKER, EMPTY);
    // 设置超节点信息数组初始值
    supno[0] = -1;
    // 初始化全局超节点起始索引数组
    xsup[0]  = xlsub[0] = xusub[0] = xlusup[0] = 0;
    // 设置默认面板大小
    w_def    = panel_size;

    // 标记松弛超节点使用的行
    ifill (marker_relax, m, EMPTY);
    i = mark_relax(m, relax_end, relax_fsupc, xa_begin, xa_end,
             asub, marker_relax);
#if ( PRNTlevel >= 1)
    printf("%d relaxed supernodes.\n", (int)i);
#endif

/*
 * 一次处理一个“面板”。面板可以是以下之一：
 *       (a) 树的底部的一个放松的超节点，或者
 *       (b) 用户定义的 panel_size 个连续列
 */
for (jcol = 0; jcol < min_mn; ) {

if ( relax_end[jcol] != EMPTY ) { /* 开始一个放松的超节点 */
    kcol = relax_end[jcol];      /* 放松的超节点的结束列 */
    panel_histo[kcol-jcol+1]++;

    /* 在上一个超节点中删除小行 */
    if (jcol > 0 && jcol < last_drop) {
    int first = xsup[supno[jcol - 1]];
    int last = jcol - 1;
    int quota;

    /* 计算配额 */
    if (drop_rule & DROP_PROWS)
        quota = gamma * Astore->nnz / m * (m - first) / m
            * (last - first + 1);
    else if (drop_rule & DROP_COLUMN) {
        int i;
        quota = 0;
        for (i = first; i <= last; i++)
        quota += xa_end[i] - xa_begin[i];
        quota = gamma * quota * (m - first) / m;
    } else if (drop_rule & DROP_AREA)
        quota = gamma * nnzAj * (1.0 - 0.5 * (last + 1.0) / m)
            - nnzLj;
    else
        quota = m * n;
    fill_tol = pow(fill_ini, 1.0 - 0.5 * (first + last) / min_mn);

    /* 删除小行 */
    i = ilu_sdrop_row(options, first, last, tol_L, quota, &nnzLj,
              &fill_tol, Glu, tempv, swork2, 0);
    /* 重置参数 */
    if (drop_rule & DROP_DYNAMIC) {
        if (gamma * nnzAj * (1.0 - 0.5 * (last + 1.0) / m)
             < nnzLj)
        tol_L = SUPERLU_MIN(1.0, tol_L * 2.0);
        else
        tol_L = SUPERLU_MAX(drop_tol, tol_L * 0.5);
    }
    if (fill_tol < 0) iinfo -= (int)fill_tol;
#ifdef DEBUG
    num_drop_L += i * (last - first + 1);
#endif
        }

        /* --------------------------------------
         * Factorize the relaxed supernode(jcol:kcol)
         * -------------------------------------- */
        /* 确定松散节点(snode)的行结构的并集 */
        if ( (*info = ilu_ssnode_dfs(jcol, kcol, asub, xa_begin, xa_end,
                     marker, Glu)) != 0 )
        return;

        nextu    = xusub[jcol];
        nextlu   = xlusup[jcol];
        jsupno   = supno[jcol];
        fsupc    = xsup[jsupno];
        new_next = nextlu + (xlsub[fsupc+1]-xlsub[fsupc])*(kcol-jcol+1);
        nzlumax = Glu->nzlumax;
        while ( new_next > nzlumax ) {
        if ((*info = sLUMemXpand(jcol, nextlu, LUSUP, &nzlumax, Glu)))
            return;
        }

        for (icol = jcol; icol <= kcol; icol++) {
        xusub[icol+1] = nextu;

        amax[0] = 0.0;
        /* 将稀疏模式(SPA)分散到密集数组dense[*]中 */
        for (k = xa_begin[icol]; k < xa_end[icol]; k++) {
            register float tmp = fabs(a[k]);
            if (tmp > amax[0]) amax[0] = tmp;
            dense[asub[k]] = a[k];
        }
        nnzAj += xa_end[icol] - xa_begin[icol];
        if (amax[0] == 0.0) {
            amax[0] = fill_ini;
#if ( PRNTlevel >= 1)
            printf("Column %d is entirely zero!\n", icol);
            fflush(stdout);
#endif
        }

        /* 在snode内进行数值更新 */
        ssnode_bmod(icol, jsupno, fsupc, dense, tempv, Glu, stat);

        if (usepr) pivrow = iperm_r[icol];
        fill_tol = pow(fill_ini, 1.0 - (double)icol / (double)min_mn);
        if ( (*info = ilu_spivotL(icol, diag_pivot_thresh, &usepr,
                      perm_r, iperm_c[icol], swap, iswap,
                      marker_relax, &pivrow,
                                          amax[0] * fill_tol, milu, zero,
                                          Glu, stat)) ) {
            iinfo++;
            marker[pivrow] = kcol;
        }

        }

        jcol = kcol + 1;
    } else { /* 处理一个 panel_size 列的面板 */

        /* 调整 panel_size，确保一个面板不会与下一个 relaxed snode 重叠 */
        panel_size = w_def;
        for (k = jcol + 1; k < SUPERLU_MIN(jcol+panel_size, min_mn); k++)
            if ( relax_end[k] != EMPTY ) {
                panel_size = k - jcol;
                break;
            }
        if ( k == min_mn ) panel_size = min_mn - jcol;
        panel_histo[panel_size]++;

        /* 在一组列的面板上进行符号因子分析 */
        ilu_spanel_dfs(m, panel_size, jcol, A, perm_r, &nseg1,
                          dense, amax, panel_lsub, segrep, repfnz,
                          marker, parent, xplore, Glu);

        /* 在拓扑顺序下对数值超面板进行更新 */
        spanel_bmod(m, panel_size, jcol, nseg1, dense,
            tempv, segrep, repfnz, Glu, stat);

        /* 在面板内以及面板对角线以下进行稀疏LU分解 */
        for (jj = jcol; jj < jcol + panel_size; jj++) {

            k = (jj - jcol) * m; /* w-wide 数组的列索引 */

            nseg = nseg1;    /* 从所有面板段开始 */

            nnzAj += xa_end[jj] - xa_begin[jj];

            if ((*info = ilu_scolumn_dfs(m, jj, perm_r, &nseg,
                             &panel_lsub[k], segrep, &repfnz[k],
                             marker, parent, xplore, Glu)))
                return;

            /* 数值更新 */
            if ((*info = scolumn_bmod(jj, (nseg - nseg1), &dense[k],
                          tempv, &segrep[nseg1], &repfnz[k],
                          jcol, Glu, stat)) != 0) return;

            /* 如果列完全为零，则生成一个填充位置 */
            if (xlsub[jj + 1] == xlsub[jj]) {
                register int i, row;
                int_t nextl;
                int_t nzlmax = Glu->nzlmax;
                int_t *lsub = Glu->lsub;
                int *marker2 = marker + 2 * m;

                /* 分配内存 */
                nextl = xlsub[jj] + 1;
                if (nextl >= nzlmax) {
                    int error = sLUMemXpand(jj, nextl, LSUB, &nzlmax, Glu);
                    if (error) { *info = error; return; }
                    lsub = Glu->lsub;
                }
                xlsub[jj + 1]++;
                assert(xlusup[jj]==xlusup[jj+1]);
                xlusup[jj + 1]++;
                ((float *) Glu->lusup)[xlusup[jj]] = zero;

                /* 为填充选择一行索引（pivrow） */
                for (i = jj; i < n; i++)
                    if (marker_relax[swap[i]] <= jj) break;
                row = swap[i];
                marker2[row] = jj;
                lsub[xlsub[jj]] = row;
#ifdef DEBUG
            printf("Fill col %d.\n", (int)jj);
            fflush(stdout);
#ifdef DEBUG
            num_drop_L += i * (last - first + 1);
#endif
        } /* if start a new supernode */

        } /* for */

        jcol += panel_size; /* Move to the next panel */

    } /* else */

    } /* for */

    *info = iinfo;

    if ( m > n ) {
    k = 0;
    for (i = 0; i < m; ++i)
        if ( perm_r[i] == EMPTY ) {
        perm_r[i] = n + k;
        ++k;
        }
    }

    ilu_countnz(min_mn, &nnzL, &nnzU, Glu);
    fixupL(min_mn, perm_r, Glu);

    sLUWorkFree(iwork, swork, Glu); /* Free work space and compress storage */
    SUPERLU_FREE (xplore);
    SUPERLU_FREE (marker_relax);

    if ( fact == SamePattern_SameRowPerm ) {
    /* L and U structures may have changed due to possibly different
       pivoting, even though the storage is available.
       There could also be memory expansions, so the array locations
       may have changed, */
    ((SCformat *)L->Store)->nnz = nnzL;
    ((SCformat *)L->Store)->nsuper = Glu->supno[n];
    ((SCformat *)L->Store)->nzval = (float *) Glu->lusup;
    ((SCformat *)L->Store)->nzval_colptr = Glu->xlusup;
    ((SCformat *)L->Store)->rowind = Glu->lsub;
    ((SCformat *)L->Store)->rowind_colptr = Glu->xlsub;
    ((NCformat *)U->Store)->nnz = nnzU;
    ((NCformat *)U->Store)->nzval = (float *) Glu->ucol;
    ((NCformat *)U->Store)->rowind = Glu->usub;
    ((NCformat *)U->Store)->colptr = Glu->xusub;
    } else {
    sCreate_SuperNode_Matrix(L, A->nrow, min_mn, nnzL,
              (float *) Glu->lusup, Glu->xlusup,
              Glu->lsub, Glu->xlsub, Glu->supno, Glu->xsup,
          SLU_SC, SLU_S, SLU_TRLU);
    sCreate_CompCol_Matrix(U, min_mn, min_mn, nnzU,
          (float *) Glu->ucol, Glu->usub, Glu->xusub,
          SLU_NC, SLU_S, SLU_TRU);
    }

    ops[FACT] += ops[TRSV] + ops[GEMV];
    stat->expansions = --(Glu->num_expansions);

    if ( iperm_r_allocated ) SUPERLU_FREE (iperm_r);
    SUPERLU_FREE (iperm_c);
    SUPERLU_FREE (relax_end);
    SUPERLU_FREE (swap);
    SUPERLU_FREE (iswap);
    SUPERLU_FREE (relax_fsupc);
    SUPERLU_FREE (amax);
    if ( swork2 ) SUPERLU_FREE (swork2);

}


注释：

#ifdef DEBUG
            printf("Fill col %d.\n", (int)jj);
            fflush(stdout);
#ifdef DEBUG
            num_drop_L += i * (last - first + 1);
#endif
        } /* if start a new supernode */

        } /* for */

        jcol += panel_size; /* Move to the next panel */

    } /* else */

    } /* for */

    *info = iinfo;

    if ( m > n ) {
    k = 0;
    for (i = 0; i < m; ++i)
        if ( perm_r[i] == EMPTY ) {
        perm_r[i] = n + k;
        ++k;
        }
    }

    ilu_countnz(min_mn, &nnzL, &nnzU, Glu);
    fixupL(min_mn, perm_r, Glu);

    sLUWorkFree(iwork, swork, Glu); /* 释放工作空间并压缩存储空间 */
    SUPERLU_FREE (xplore);
    SUPERLU_FREE (marker_relax);

    if ( fact == SamePattern_SameRowPerm ) {
    /* 由于可能的不同的枢轴选取，L 和 U 的结构可能已经改变，尽管存储是可用的。
       还可能存在内存扩展，因此数组位置可能已更改。 */
    ((SCformat *)L->Store)->nnz = nnzL;
    ((SCformat *)L->Store)->nsuper = Glu->supno[n];
    ((SCformat *)L->Store)->nzval = (float *) Glu->lusup;
    ((SCformat *)L->Store)->nzval_colptr = Glu->xlusup;
    ((SCformat *)L->Store)->rowind = Glu->lsub;
    ((SCformat *)L->Store)->rowind_colptr = Glu->xlsub;
    ((NCformat *)U->Store)->nnz = nnzU;
    ((NCformat *)U->Store)->nzval = (float *) Glu->ucol;
    ((NCformat *)U->Store)->rowind = Glu->usub;
    ((NCformat *)U->Store)->colptr = Glu->xusub;
    } else {
    sCreate_SuperNode_Matrix(L, A->nrow, min_mn, nnzL,
              (float *) Glu->lusup, Glu->xlusup,
              Glu->lsub, Glu->xlsub, Glu->supno, Glu->xsup,
          SLU_SC, SLU_S, SLU_TRLU);
    sCreate_CompCol_Matrix(U, min_mn, min_mn, nnzU,
          (float *) Glu->ucol, Glu->usub, Glu->xusub,
          SLU_NC, SLU_S, SLU_TRU);
    }

    ops[FACT] += ops[TRSV] + ops[GEMV];
    stat->expansions = --(Glu->num_expansions);

    if ( iperm_r_allocated ) SUPERLU_FREE (iperm_r);
    SUPERLU_FREE (iperm_c);
    SUPERLU_FREE (relax_end);
    SUPERLU_FREE (swap);
    SUPERLU_FREE (iswap);
    SUPERLU_FREE (relax_fsupc);
    SUPERLU_FREE (amax);
    if ( swork2 ) SUPERLU_FREE (swork2);

}
```