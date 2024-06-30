# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zgsitrf.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file zgsitrf.c
 * \brief Computes an ILU factorization of a general sparse matrix
 *
 * <pre>
 * -- SuperLU routine (version 4.1) --
 * Lawrence Berkeley National Laboratory.
 * June 30, 2009
 *
 * </pre>
 */

#include "slu_zdefs.h"

#ifdef DEBUG
int num_drop_L;
#endif

void
zgsitrf(superlu_options_t *options, SuperMatrix *A, int relax, int panel_size,
    int *etree, void *work, int_t lwork, int *perm_c, int *perm_r,
    SuperMatrix *L, SuperMatrix *U, 
        GlobalLU_t *Glu, /* persistent to facilitate multiple factorizations */
    SuperLUStat_t *stat, int_t *info)
{
    /* Local working arrays */
    NCPformat *Astore;
    int       *iperm_r = NULL; /* inverse of perm_r; used when
                  options->Fact == SamePattern_SameRowPerm */
    int       *iperm_c; /* inverse of perm_c */
    int       *swap, *iswap; /* swap is used to store the row permutation
                during the factorization. Initially, it is set
                to iperm_c (row indeces of Pc*A*Pc').
                iswap is the inverse of swap. After the
                factorization, it is equal to perm_r. */
    int       *iwork;
    doublecomplex   *zwork;
    int       *segrep, *repfnz, *parent;
    int_t     *xplore;
    int       *panel_lsub; /* dense[]/panel_lsub[] pair forms a w-wide SPA */
    int       *marker;
    int       *marker_relax;
    doublecomplex    *dense, *tempv;
    double *dtempv;
    int       *relax_end, *relax_fsupc;
    doublecomplex    *a;
    int_t     *asub;
    int_t     *xa_begin, *xa_end;
    int       *xsup, *supno;
    int_t     *xlsub, *xlusup, *xusub;
    int_t     nzlumax;
    double    *amax; 
    doublecomplex    drop_sum;
    double alpha, omega;  /* used in MILU, mimicing DRIC */
    double    *dwork2;       /* used by the second dropping rule */

    /* Local scalars */
    fact_t    fact = options->Fact; // 确定因子化的类型（例如，是否使用相同的模式和行排列）
    double    diag_pivot_thresh = options->DiagPivotThresh; // 对角元素主元阈值
    double    drop_tol = options->ILU_DropTol; /* tau */ // ILU下降阈值
    double    fill_ini = options->ILU_FillTol; /* tau^hat */ // ILU填充阈值
    double    gamma = options->ILU_FillFactor; // ILU填充因子
    int       drop_rule = options->ILU_DropRule; // ILU下降规则
    milu_t    milu = options->ILU_MILU; // MILU类型
    double    fill_tol; // 填充容忍度
    int       pivrow;    /* pivotal row number in the original matrix A */ // 原始矩阵A中的主元行号
    int       nseg1;    /* no of segments in U-column above panel row jcol */ // 在面板行jcol上面的U列中的段数
    int       nseg;    /* no of segments in each U-column */ // 每个U列中的段数
    register int jcol, jj;
    register int kcol;    /* end column of a relaxed snode */ // 松弛snode的结束列
    register int icol;
    int_t     i, k, iinfo;
    int       m, n, min_mn, jsupno, fsupc;
    int_t     new_next, nextlu, nextu;
    int       w_def;    /* 上限面板宽度 */
    int       usepr, iperm_r_allocated = 0;
    int_t     nnzL, nnzU;    /* L 和 U 的非零元素数量 */
    int       *panel_histo = stat->panel_histo;    /* 面板直方图指针 */
    flops_t   *ops = stat->ops;    /* 操作计数器指针 */

    int       last_drop;/* 最后应用丢弃规则的列 */
    int       quota;    /* 配额 */
    int       nnzAj;    /* A(:,1:j) 的非零元素数量 */
    int       nnzLj, nnzUj;    /* L(:,1:j) 和 U(:,1:j) 的非零元素数量 */
    double    tol_L = drop_tol, tol_U = drop_tol;    /* L 和 U 的丢弃容忍度 */
    doublecomplex zero = {0.0, 0.0};    /* 复数零 */
    double one = 1.0;    /* 浮点数 1.0 */

    /* 可执行代码 */       
    iinfo    = 0;    /* 初始化信息 */
    m         = A->nrow;    /* 矩阵 A 的行数 */
    n         = A->ncol;    /* 矩阵 A 的列数 */
    min_mn   = SUPERLU_MIN(m, n);    /* m 和 n 的最小值 */
    Astore   = A->Store;    /* A 的存储结构 */
    a         = Astore->nzval;    /* A 的非零元素值 */
    asub     = Astore->rowind;    /* A 的行索引 */
    xa_begin = Astore->colbeg;    /* A 的起始列索引 */
    xa_end   = Astore->colend;    /* A 的结束列索引 */

    /* 分配与因子化例程共享的存储空间 */
    *info = zLUMemInit(fact, work, lwork, m, n, Astore->nnz, panel_size,
               gamma, L, U, Glu, &iwork, &zwork);
    if ( *info ) return;

    xsup    = Glu->xsup;    /* 超节点起始列索引 */
    supno   = Glu->supno;    /* 超节点编号 */
    xlsub   = Glu->xlsub;    /* L 的列索引 */
    xlusup  = Glu->xlusup;    /* LU 的超节点起始行索引 */
    xusub   = Glu->xusub;    /* U 的列索引 */

    int_t *xprune;
    SetIWork(m, n, panel_size, iwork, &segrep, &parent, &xplore,
         &repfnz, &panel_lsub, &xprune, &marker);    /* 设置整型工作空间 */
    marker_relax = int32Malloc(m);    /* 分配松弛标记空间 */
    SUPERLU_FREE(xprune); /* ILU 中不使用 */
    
    zSetRWork(m, panel_size, zwork, &dense, &tempv);    /* 设置双精度实数工作空间 */

    usepr = (fact == SamePattern_SameRowPerm);    /* 使用相同行排列模式 */
    if ( usepr ) {
    /* 计算 perm_r 的逆 */
    iperm_r = (int *) int32Malloc(m);
    for (k = 0; k < m; ++k) iperm_r[perm_r[k]] = k;
    iperm_r_allocated = 1;
    }

    iperm_c = (int *) int32Malloc(n);
    for (k = 0; k < n; ++k) iperm_c[perm_c[k]] = k;
    swap = (int *)intMalloc(n);
    for (k = 0; k < n; k++) swap[k] = iperm_c[k];
    iswap = (int *)intMalloc(n);
    for (k = 0; k < n; k++) iswap[k] = perm_c[k];
    amax = (double *) SUPERLU_MALLOC(panel_size * sizeof(double));
    if (drop_rule & DROP_SECONDARY)
    dwork2 = SUPERLU_MALLOC(n * sizeof(double));
    else
    dwork2 = NULL;

    nnzAj = 0;    /* A(:,1:j) 的非零元素数量初始化 */
    nnzLj = 0;    /* L(:,1:j) 的非零元素数量初始化 */
    nnzUj = 0;    /* U(:,1:j) 的非零元素数量初始化 */
    last_drop = SUPERLU_MAX(min_mn - 2 * sp_ienv(7), (int)(min_mn * 0.95));    /* 最后丢弃列的索引 */
    alpha = pow((double)n, -1.0 / options->ILU_MILU_Dim);    /* 计算 alpha 值 */

    /* 标识松弛超节点 */
    relax_end = (int *) intMalloc(n);    /* 分配松弛结束数组 */
    relax_fsupc = (int *) intMalloc(n);    /* 分配松弛第一个超节点数组 */
    if ( options->SymmetricMode == YES )
    ilu_heap_relax_snode(n, etree, relax, marker, relax_end, relax_fsupc);    /* 堆松弛超节点 */
    else
    ilu_relax_snode(n, etree, relax, marker, relax_end, relax_fsupc);    /* 松弛超节点 */

    ifill (perm_r, m, EMPTY);    /* 用空值填充 perm_r */
    ifill (marker, m * NO_MARKER, EMPTY);    /* 用空值填充 marker */
    supno[0] = -1;    /* 第一个超节点编号设为 -1 */
    xsup[0]  = xlsub[0] = xusub[0] = xlusup[0] = 0;    /* 超节点索引初始化 */
    w_def    = panel_size;    /* 面板默认宽度设定 */

    /* 标记被松弛超节点使用的行 */
    ifill (marker_relax, m, EMPTY);    /* 用空值填充 marker_relax */
    i = mark_relax(m, relax_end, relax_fsupc, xa_begin, xa_end,
             asub, marker_relax);    /* 标记被松弛超节点使用的行 */
#if ( PRNTlevel >= 1)
    printf("%d relaxed supernodes.\n", (int)i);
#endif

/*
 * Work on one "panel" at a time. A panel is one of the following:
 *       (a) a relaxed supernode at the bottom of the etree, or
 *       (b) panel_size contiguous columns, defined by the user
 */
for (jcol = 0; jcol < min_mn; ) {

if ( relax_end[jcol] != EMPTY ) { /* start of a relaxed snode */
    kcol = relax_end[jcol];      /* end of the relaxed snode */
    panel_histo[kcol-jcol+1]++;

    /* Drop small rows in the previous supernode. */
    if (jcol > 0 && jcol < last_drop) {
        int first = xsup[supno[jcol - 1]];
        int last = jcol - 1;
        int quota;

        /* Compute the quota based on different drop rules */
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

        /* Drop small rows using ILU drop method */
        dtempv = (double *) tempv;
        i = ilu_zdrop_row(options, first, last, tol_L, quota, &nnzLj,
                  &fill_tol, Glu, dtempv, dwork2, 0);
        
        /* Reset the tolerance parameters if DROP_DYNAMIC is enabled */
        if (drop_rule & DROP_DYNAMIC) {
            if (gamma * nnzAj * (1.0 - 0.5 * (last + 1.0) / m)
                 < nnzLj)
                tol_L = SUPERLU_MIN(1.0, tol_L * 2.0);
            else
                tol_L = SUPERLU_MAX(drop_tol, tol_L * 0.5);
        }

        /* Adjust iinfo based on fill tolerance if in debug mode */
        if (fill_tol < 0)
            iinfo -= (int)fill_tol;
#ifdef DEBUG
        num_drop_L += i * (last - first + 1);
#endif
        }

        /* --------------------------------------
         * Factorize the relaxed supernode(jcol:kcol)
         * -------------------------------------- */
        /* 确定松散超节点(snode)的行结构的并集 */
        if ( (*info = ilu_zsnode_dfs(jcol, kcol, asub, xa_begin, xa_end,
                     marker, Glu)) != 0 )
        return;

        nextu    = xusub[jcol];
        nextlu   = xlusup[jcol];
        jsupno   = supno[jcol];
        fsupc    = xsup[jsupno];
        new_next = nextlu + (xlsub[fsupc+1]-xlsub[fsupc])*(kcol-jcol+1);
        nzlumax = Glu->nzlumax;
        while ( new_next > nzlumax ) {
        if ((*info = zLUMemXpand(jcol, nextlu, LUSUP, &nzlumax, Glu)))
            return;
        }

        for (icol = jcol; icol <= kcol; icol++) {
        xusub[icol+1] = nextu;

        amax[0] = 0.0;
        /* 将数据分散到SPA dense[*] */
        for (k = xa_begin[icol]; k < xa_end[icol]; k++) {
                    register double tmp = z_abs1 (&a[k]);
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

        /* 在超节点内进行数值更新 */
        zsnode_bmod(icol, jsupno, fsupc, dense, tempv, Glu, stat);

        if (usepr) pivrow = iperm_r[icol];
        fill_tol = pow(fill_ini, 1.0 - (double)icol / (double)min_mn);
        if ( (*info = ilu_zpivotL(icol, diag_pivot_thresh, &usepr,
                      perm_r, iperm_c[icol], swap, iswap,
                      marker_relax, &pivrow,
                                          amax[0] * fill_tol, milu, zero,
                                          Glu, stat)) ) {
            iinfo++;
            marker[pivrow] = kcol;
        }

        }

        jcol = kcol + 1;
    } else { /* 处理一列大小为 panel_size 的面板 */

        /* 调整 panel_size，以防止一个面板与下一个放松的 snode 重叠 */
        panel_size = w_def;
        for (k = jcol + 1; k < SUPERLU_MIN(jcol+panel_size, min_mn); k++)
        if ( relax_end[k] != EMPTY ) {
            panel_size = k - jcol;
            break;
        }
        if ( k == min_mn ) panel_size = min_mn - jcol;
        panel_histo[panel_size]++;

        /* 在一列大小为 panel_size 的面板上进行符号因子分解 */
        ilu_zpanel_dfs(m, panel_size, jcol, A, perm_r, &nseg1,
                          dense, amax, panel_lsub, segrep, repfnz,
                          marker, parent, xplore, Glu);

        /* 在拓扑顺序中对数值超面板进行更新 */
        zpanel_bmod(m, panel_size, jcol, nseg1, dense,
            tempv, segrep, repfnz, Glu, stat);

        /* 面板内和面板对角线以下的稀疏 LU 分解 */
        for (jj = jcol; jj < jcol + panel_size; jj++) {

        k = (jj - jcol) * m; /* w-wide 数组的列索引 */

        nseg = nseg1;    /* 在所有面板段之后开始 */

        nnzAj += xa_end[jj] - xa_begin[jj];

        if ((*info = ilu_zcolumn_dfs(m, jj, perm_r, &nseg,
                         &panel_lsub[k], segrep, &repfnz[k],
                         marker, parent, xplore, Glu)))
            return;

        /* 数值更新 */
        if ((*info = zcolumn_bmod(jj, (nseg - nseg1), &dense[k],
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
            int error = zLUMemXpand(jj, nextl, LSUB, &nzlmax, Glu);
            if (error) { *info = error; return; }
            lsub = Glu->lsub;
            }
            xlsub[jj + 1]++;
            assert(xlusup[jj]==xlusup[jj+1]);
            xlusup[jj + 1]++;
            ((doublecomplex *) Glu->lusup)[xlusup[jj]] = zero;

            /* 为填充选择行索引（pivrow） */
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


    // 计算 ILU 因子中的非零元素数量和修正 L 的结构
    ilu_countnz(min_mn, &nnzL, &nnzU, Glu);
    fixupL(min_mn, perm_r, Glu);

    // 释放工作空间并压缩存储
    zLUWorkFree(iwork, zwork, Glu);

    // 释放探索向量和松弛标记
    SUPERLU_FREE (xplore);
    SUPERLU_FREE (marker_relax);

    // 根据选择的因子化方式更新 L 和 U 的数据结构
    if ( fact == SamePattern_SameRowPerm ) {
    /* L and U structures may have changed due to possibly different
       pivoting, even though the storage is available.
       There could also be memory expansions, so the array locations
       may have changed, */
    ((SCformat *)L->Store)->nnz = nnzL;
    ((SCformat *)L->Store)->nsuper = Glu->supno[n];
    ((SCformat *)L->Store)->nzval = (doublecomplex *) Glu->lusup;
    ((SCformat *)L->Store)->nzval_colptr = Glu->xlusup;
    ((SCformat *)L->Store)->rowind = Glu->lsub;
    ((SCformat *)L->Store)->rowind_colptr = Glu->xlsub;
    ((NCformat *)U->Store)->nnz = nnzU;
    ((NCformat *)U->Store)->nzval = (doublecomplex *) Glu->ucol;
    ((NCformat *)U->Store)->rowind = Glu->usub;
    ((NCformat *)U->Store)->colptr = Glu->xusub;
    } else {
    // 创建超节点格式的 L 矩阵
    zCreate_SuperNode_Matrix(L, A->nrow, min_mn, nnzL,
              (doublecomplex *) Glu->lusup, Glu->xlusup,
              Glu->lsub, Glu->xlsub, Glu->supno, Glu->xsup,
          SLU_SC, SLU_Z, SLU_TRLU);
    // 创建压缩列格式的 U 矩阵
    zCreate_CompCol_Matrix(U, min_mn, min_mn, nnzU,
          (doublecomplex *) Glu->ucol, Glu->usub, Glu->xusub,
          SLU_NC, SLU_Z, SLU_TRU);
    }


    // 更新操作数统计信息
    ops[FACT] += ops[TRSV] + ops[GEMV];
    // 更新扩展数统计信息
    stat->expansions = --(Glu->num_expansions);

    // 释放额外分配的 iperm_r 数组
    if ( iperm_r_allocated ) SUPERLU_FREE (iperm_r);
    // 释放 iperm_c 数组
    SUPERLU_FREE (iperm_c);
    // 释放 relax_end 数组
    SUPERLU_FREE (relax_end);
    // 释放 swap 数组
    SUPERLU_FREE (swap);
    // 释放 iswap 数组
    SUPERLU_FREE (iswap);
    // 释放 relax_fsupc 数组
    SUPERLU_FREE (relax_fsupc);
    // 释放 amax 数组
    SUPERLU_FREE (amax);
    // 如果存在 dwork2 则释放 dwork2 数组
    if ( dwork2 ) SUPERLU_FREE (dwork2);
}
```