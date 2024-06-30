# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\cgsitrf.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file cgsitrf.c
 * \brief Computes an ILU factorization of a general sparse matrix
 *
 * <pre>
 * -- SuperLU routine (version 4.1) --
 * Lawrence Berkeley National Laboratory.
 * June 30, 2009
 *
 * </pre>
 */

#include "slu_cdefs.h"

#ifdef DEBUG
int num_drop_L;
#endif

void
cgsitrf(superlu_options_t *options, SuperMatrix *A, int relax, int panel_size,
    int *etree, void *work, int_t lwork, int *perm_c, int *perm_r,
    SuperMatrix *L, SuperMatrix *U, 
        GlobalLU_t *Glu, /* persistent to facilitate multiple factorizations */
    SuperLUStat_t *stat, int_t *info)
{
    /* Local working arrays */

    // Astore: 指向输入稀疏矩阵 A 的非压缩列格式存储结构
    NCPformat *Astore;

    // iperm_r: perm_r 的逆序列，当 options->Fact == SamePattern_SameRowPerm 时使用
    int       *iperm_r = NULL;

    // iperm_c: perm_c 的逆序列
    int       *iperm_c;

    // swap, iswap: 在因子分解期间用于存储行置换和其逆置换
    int       *swap, *iswap;

    // iwork: 整数工作数组
    int       *iwork;

    // cwork: 复数工作数组
    singlecomplex   *cwork;

    // segrep, repfnz, parent: 用于 SPA (Sparse Pattern Awareness) 的数组
    int       *segrep, *repfnz, *parent;

    // xplore: 探索数组
    int_t     *xplore;

    // panel_lsub: 用于 w-wide SPA 的数组
    int       *panel_lsub;

    // marker: 标记数组
    int       *marker;

    // marker_relax: 放松标记数组
    int       *marker_relax;

    // dense, tempv: 密集向量和临时向量
    singlecomplex    *dense, *tempv;

    // stempv: 浮点临时向量
    float *stempv;

    // relax_end, relax_fsupc: 放松结束和超级列的起始位置数组
    int       *relax_end, *relax_fsupc;

    // a, asub: 用于 L 的数据和行号
    singlecomplex    *a;
    int_t     *asub;

    // xa_begin, xa_end: 用于 U 的列指针
    int_t     *xa_begin, *xa_end;

    // xsup, supno: 超级节点的起始位置和超级节点编号
    int       *xsup, *supno;

    // xlsub, xlusup, xusub: L 和 U 的非零元素的行索引
    int_t     *xlsub, *xlusup, *xusub;

    // nzlumax: L 和 U 的最大非零元素数
    int_t     nzlumax;

    // amax: 绝对值最大元素的数组
    float    *amax; 

    // drop_sum: 用于存储下降因子的复数和
    singlecomplex    drop_sum;

    // alpha, omega: MILU 中使用的参数
    float alpha, omega;

    // swork2: 第二个丢弃规则使用的浮点数组
    float    *swork2;

    /* Local scalars */

    // fact: 因子类型选项
    fact_t    fact = options->Fact;

    // diag_pivot_thresh: 对角元主元阈值选项
    double    diag_pivot_thresh = options->DiagPivotThresh;

    // drop_tol: 下降阈值选项 (tau)
    double    drop_tol = options->ILU_DropTol;

    // fill_ini: 填充阈值选项 (tau^hat)
    double    fill_ini = options->ILU_FillTol;

    // gamma: 填充因子选项
    double    gamma = options->ILU_FillFactor;

    // drop_rule: 下降规则选项
    int       drop_rule = options->ILU_DropRule;

    // milu: MILU 类型选项
    milu_t    milu = options->ILU_MILU;

    // fill_tol: 填充容差
    double    fill_tol;

    // pivrow: 原始矩阵 A 中的主元行号
    int       pivrow;

    // nseg1, nseg: U 列中分段的数量
    int       nseg1, nseg;

    // jcol: 当前列号
    register int jcol, jj;

    // kcol: 放松 snode 的结束列号
    register int kcol;

    // icol: 当前列号
    register int icol;

    // iinfo: 输出信息
    int_t     iinfo;

    // m, n, min_mn: 矩阵的行数、列数和最小维度
    int       m, n, min_mn;

    // jsupno, fsupc: 超级节点编号和超级列号
    int_t     jsupno, fsupc;

    // new_next, nextlu, nextu: 新的下一个 LU 元素的位置
    int_t     new_next, nextlu, nextu;
    
    // 省略部分局部变量声明...
    int       w_def;    /* upper bound on panel width */ 
    int       usepr, iperm_r_allocated = 0;   /* flags for row permutation usage and allocation status of iperm_r */
    int_t     nnzL, nnzU;    /* number of nonzeros in L and U factors */
    int       *panel_histo = stat->panel_histo;   /* pointer to panel histogram in statistics structure */
    flops_t   *ops = stat->ops;    /* pointer to flop count statistics */

    int       last_drop;    /* last column where dropping rules were applied */
    int       quota;    /* parameter for controlling numerical stability */
    int       nnzAj;    /* number of nonzeros in A(:,1:j) */
    int       nnzLj, nnzUj;    /* number of nonzeros in L(:,1:j) and U(:,1:j) */
    double    tol_L = drop_tol, tol_U = drop_tol;    /* tolerances for dropping elements */
    singlecomplex zero = {0.0, 0.0};    /* complex zero */
    float one = 1.0;    /* floating-point one */

    /* Executable */
    iinfo    = 0;    /* initialize error flag */
    m         = A->nrow;    /* number of rows in matrix A */
    n         = A->ncol;    /* number of columns in matrix A */
    min_mn   = SUPERLU_MIN(m, n);    /* minimum of m and n */
    Astore   = A->Store;    /* matrix A in compressed sparse column format */
    a         = Astore->nzval;    /* array of nonzero values in A */
    asub     = Astore->rowind;    /* array of row indices of A */
    xa_begin = Astore->colbeg;    /* beginning index of columns in A */
    xa_end   = Astore->colend;    /* ending index of columns in A */

    /* Allocate storage common to the factor routines */
    *info = cLUMemInit(fact, work, lwork, m, n, Astore->nnz, panel_size,
               gamma, L, U, Glu, &iwork, &cwork);   /* initialize memory for LU factorization */
    if ( *info ) return;    /* return if initialization fails */

    xsup    = Glu->xsup;    /* supernode starting indices */
    supno   = Glu->supno;    /* supernode numbers */
    xlsub   = Glu->xlsub;    /* column pointers for L factor */
    xlusup  = Glu->xlusup;    /* column pointers for U factor */
    xusub   = Glu->xusub;    /* column indices for U factor */

    int_t *xprune;    /* array for pruning */
    SetIWork(m, n, panel_size, iwork, &segrep, &parent, &xplore,
         &repfnz, &panel_lsub, &xprune, &marker);    /* set up integer workspace arrays */
    marker_relax = int32Malloc(m);    /* allocate marker array for relaxation */
    SUPERLU_FREE(xprune); /* free unused memory */

    cSetRWork(m, panel_size, cwork, &dense, &tempv);    /* set up real workspace arrays */

    usepr = (fact == SamePattern_SameRowPerm);    /* determine if using the same row permutation */
    if ( usepr ) {
    /* Compute the inverse of perm_r */
    iperm_r = (int *) int32Malloc(m);
    for (k = 0; k < m; ++k) iperm_r[perm_r[k]] = k;    /* compute inverse permutation */
    iperm_r_allocated = 1;    /* flag that iperm_r has been allocated */
    }

    iperm_c = (int *) int32Malloc(n);    /* allocate column permutation */
    for (k = 0; k < n; ++k) iperm_c[perm_c[k]] = k;    /* compute inverse of column permutation */
    swap = (int *)intMalloc(n);    /* allocate swap array */
    for (k = 0; k < n; k++) swap[k] = iperm_c[k];    /* copy column permutation to swap array */
    iswap = (int *)intMalloc(n);    /* allocate inverse swap array */
    for (k = 0; k < n; k++) iswap[k] = perm_c[k];    /* copy column permutation to inverse swap array */
    amax = (float *) SUPERLU_MALLOC(panel_size * sizeof(float));    /* allocate array for maximum elements */
    if (drop_rule & DROP_SECONDARY)
    swork2 = SUPERLU_MALLOC(n * sizeof(float));    /* allocate secondary workspace */
    else
    swork2 = NULL;    /* no secondary workspace needed */

    nnzAj = 0;    /* initialize count of nonzeros in A(:,1:j) */
    nnzLj = 0;    /* initialize count of nonzeros in L(:,1:j) */
    nnzUj = 0;    /* initialize count of nonzeros in U(:,1:j) */
    last_drop = SUPERLU_MAX(min_mn - 2 * sp_ienv(7), (int)(min_mn * 0.95));    /* determine last column for dropping */
    alpha = pow((double)n, -1.0 / options->ILU_MILU_Dim);    /* compute alpha for ILU */

    /* Identify relaxed snodes */
    relax_end = (int *) intMalloc(n);    /* allocate relaxation end array */
    relax_fsupc = (int *) intMalloc(n);    /* allocate relaxed supernode array */
    if ( options->SymmetricMode == YES )
    ilu_heap_relax_snode(n, etree, relax, marker, relax_end, relax_fsupc);    /* heap-based relaxation for symmetric mode */
    else
    ilu_relax_snode(n, etree, relax, marker, relax_end, relax_fsupc);    /* general relaxation for non-symmetric mode */

    ifill (perm_r, m, EMPTY);    /* initialize perm_r with EMPTY */
    ifill (marker, m * NO_MARKER, EMPTY);    /* initialize marker with EMPTY */
    supno[0] = -1;    /* initialize first supernode number */
    xsup[0]  = xlsub[0] = xusub[0] = xlusup[0] = 0;    /* initialize supernode and column indices */
    w_def    = panel_size;    /* set default panel width */

    /* Mark the rows used by relaxed supernodes */
    ifill (marker_relax, m, EMPTY);    /* initialize marker_relax with EMPTY */
    i = mark_relax(m, relax_end, relax_fsupc, xa_begin, xa_end,
             asub, marker_relax);    /* mark rows used by relaxed supernodes */
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

            /* Compute the quota */
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

            /* Drop small rows */
            stempv = (float *) tempv;
            i = ilu_cdrop_row(options, first, last, tol_L, quota, &nnzLj,
                        &fill_tol, Glu, stempv, swork2, 0);
            
            /* Reset the parameters */
            if (drop_rule & DROP_DYNAMIC) {
                if (gamma * nnzAj * (1.0 - 0.5 * (last + 1.0) / m)
                     < nnzLj)
                    tol_L = SUPERLU_MIN(1.0, tol_L * 2.0);
                else
                    tol_L = SUPERLU_MAX(drop_tol, tol_L * 0.5);
            }
            
            if (fill_tol < 0)
                iinfo -= (int)fill_tol;
#ifdef DEBUG
            num_drop_L += i * (last - first + 1);
#endif
        }
    }

    jcol = kcol + 1;  /* Move to the next panel */
}


注释：


#if ( PRNTlevel >= 1)
    // 如果打印级别大于等于1，打印放松的超节点数量
    printf("%d relaxed supernodes.\n", (int)i);
#endif

/*
 * 逐个处理“panel”。一个“panel”可以是以下之一：
 *       (a) 树的底部的一个放松超节点，或者
 *       (b) 用户定义的 panel_size 个连续列
 */
for (jcol = 0; jcol < min_mn; ) {

    if ( relax_end[jcol] != EMPTY ) { /* 开始一个放松超节点 */
        kcol = relax_end[jcol];      /* 放松超节点的结束列 */
        panel_histo[kcol-jcol+1]++;

        /* 在上一个超节点中丢弃小行 */
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
            
            // 计算填充容差
            fill_tol = pow(fill_ini, 1.0 - 0.5 * (first + last) / min_mn);

            /* 丢弃小行 */
            stempv = (float *) tempv;
            i = ilu_cdrop_row(options, first, last, tol_L, quota, &nnzLj,
                        &fill_tol, Glu, stempv, swork2, 0);
            
            /* 重置参数 */
            if (drop_rule & DROP_DYNAMIC) {
                if (gamma * nnzAj * (1.0 - 0.5 * (last + 1.0) / m)
                     < nnzLj)
                    tol_L = SUPERLU_MIN(1.0, tol_L * 2.0);
                else
                    tol_L = SUPERLU_MAX(drop_tol, tol_L * 0.5);
            }
            
            // 如果填充容差小于0，更新信息
            if (fill_tol < 0)
                iinfo -= (int)fill_tol;
#ifdef DEBUG
            // 如果是调试模式，更新丢弃的 L 的数量
            num_drop_L += i * (last - first + 1);
#endif
        }
    }

    jcol = kcol + 1;  /* 移动到下一个 panel */
}
#endif
        }

        /* --------------------------------------
         * Factorize the relaxed supernode(jcol:kcol)
         * -------------------------------------- */

        /* 确定超节点snode的行结构的并集 */
        if ( (*info = ilu_csnode_dfs(jcol, kcol, asub, xa_begin, xa_end,
                     marker, Glu)) != 0 )
        return;

        nextu    = xusub[jcol];
        nextlu   = xlusup[jcol];
        jsupno   = supno[jcol];
        fsupc    = xsup[jsupno];
        new_next = nextlu + (xlsub[fsupc+1]-xlsub[fsupc])*(kcol-jcol+1);
        nzlumax = Glu->nzlumax;
        
        /* 扩展LUMemXpand直到能够容纳新的LUSUP块 */
        while ( new_next > nzlumax ) {
            if ((*info = cLUMemXpand(jcol, nextlu, LUSUP, &nzlumax, Glu)))
                return;
        }

        /* 遍历列范围内的每一列 */
        for (icol = jcol; icol <= kcol; icol++) {
            xusub[icol+1] = nextu;

            amax[0] = 0.0;
            /* 将非零元素散落到SPA密集矩阵dense[*]中 */
            for (k = xa_begin[icol]; k < xa_end[icol]; k++) {
                register float tmp = c_abs1 (&a[k]);
                if (tmp > amax[0]) amax[0] = tmp;
                dense[asub[k]] = a[k];
            }
            nnzAj += xa_end[icol] - xa_begin[icol];
            
            /* 如果amax[0]为零，将其初始化为fill_ini */
            if (amax[0] == 0.0) {
                amax[0] = fill_ini;
#if ( PRNTlevel >= 1)
                printf("Column %d is entirely zero!\n", icol);
                fflush(stdout);
#endif
            }

            /* 在超节点内进行数值更新 */
            csnode_bmod(icol, jsupno, fsupc, dense, tempv, Glu, stat);

            /* 如果使用部分置换，则获取行置换 */
            if (usepr) pivrow = iperm_r[icol];

            /* 计算填充因子容限 */
            fill_tol = pow(fill_ini, 1.0 - (double)icol / (double)min_mn);

            /* 在L部分进行主元选择 */
            if ( (*info = ilu_cpivotL(icol, diag_pivot_thresh, &usepr,
                      perm_r, iperm_c[icol], swap, iswap,
                      marker_relax, &pivrow,
                                          amax[0] * fill_tol, milu, zero,
                                          Glu, stat)) ) {
                iinfo++;
                marker[pivrow] = kcol;
            }
        }

        jcol = kcol + 1;
    } else { /* Work on one panel of panel_size columns */

        /* 处理一个包含 panel_size 列的面板 */

        /* 调整 panel_size，以确保面板不会与下一个放松的超节点重叠 */
        panel_size = w_def;
        for (k = jcol + 1; k < SUPERLU_MIN(jcol+panel_size, min_mn); k++)
        if ( relax_end[k] != EMPTY ) {
            panel_size = k - jcol;
            break;
        }
        if ( k == min_mn ) panel_size = min_mn - jcol;
        panel_histo[panel_size]++;

        /* 在一组列的面板上进行符号因子分解 */
        ilu_cpanel_dfs(m, panel_size, jcol, A, perm_r, &nseg1,
                          dense, amax, panel_lsub, segrep, repfnz,
                          marker, parent, xplore, Glu);

        /* 按拓扑顺序对数值超面板进行更新 */
        cpanel_bmod(m, panel_size, jcol, nseg1, dense,
            tempv, segrep, repfnz, Glu, stat);

        /* 在面板内和面板对角线以下进行稀疏LU分解 */
        for (jj = jcol; jj < jcol + panel_size; jj++) {

        k = (jj - jcol) * m; /* w-wide 数组的列索引 */

        nseg = nseg1;    /* 在所有面板段之后开始 */

        nnzAj += xa_end[jj] - xa_begin[jj];

        if ((*info = ilu_ccolumn_dfs(m, jj, perm_r, &nseg,
                         &panel_lsub[k], segrep, &repfnz[k],
                         marker, parent, xplore, Glu)))
            return;

        /* 数值更新 */
        if ((*info = ccolumn_bmod(jj, (nseg - nseg1), &dense[k],
                      tempv, &segrep[nseg1], &repfnz[k],
                      jcol, Glu, stat)) != 0) return;

        /* 如果列完全为零，则创建一个填充位置 */
        if (xlsub[jj + 1] == xlsub[jj]) {
            register int i, row;
            int_t nextl;
            int_t nzlmax = Glu->nzlmax;
            int_t *lsub = Glu->lsub;
            int *marker2 = marker + 2 * m;

            /* 分配内存 */
            nextl = xlsub[jj] + 1;
            if (nextl >= nzlmax) {
            int error = cLUMemXpand(jj, nextl, LSUB, &nzlmax, Glu);
            if (error) { *info = error; return; }
            lsub = Glu->lsub;
            }
            xlsub[jj + 1]++;
            assert(xlusup[jj]==xlusup[jj+1]);
            xlusup[jj + 1]++;
            ((singlecomplex *) Glu->lusup)[xlusup[jj]] = zero;

            /* 为填充选择一个行索引 (pivrow) */
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

    cLUWorkFree(iwork, cwork, Glu); /* Free work space and compress storage */
    SUPERLU_FREE (xplore);
    SUPERLU_FREE (marker_relax);

    if ( fact == SamePattern_SameRowPerm ) {
    /* L and U structures may have changed due to possibly different
       pivoting, even though the storage is available.
       There could also be memory expansions, so the array locations
       may have changed, */
    ((SCformat *)L->Store)->nnz = nnzL;
    ((SCformat *)L->Store)->nsuper = Glu->supno[n];
    ((SCformat *)L->Store)->nzval = (singlecomplex *) Glu->lusup;
    ((SCformat *)L->Store)->nzval_colptr = Glu->xlusup;
    ((SCformat *)L->Store)->rowind = Glu->lsub;
    ((SCformat *)L->Store)->rowind_colptr = Glu->xlsub;
    ((NCformat *)U->Store)->nnz = nnzU;
    ((NCformat *)U->Store)->nzval = (singlecomplex *) Glu->ucol;
    ((NCformat *)U->Store)->rowind = Glu->usub;
    ((NCformat *)U->Store)->colptr = Glu->xusub;
    } else {
    cCreate_SuperNode_Matrix(L, A->nrow, min_mn, nnzL,
              (singlecomplex *) Glu->lusup, Glu->xlusup,
              Glu->lsub, Glu->xlsub, Glu->supno, Glu->xsup,
          SLU_SC, SLU_C, SLU_TRLU);
    cCreate_CompCol_Matrix(U, min_mn, min_mn, nnzU,
          (singlecomplex *) Glu->ucol, Glu->usub, Glu->xusub,
          SLU_NC, SLU_C, SLU_TRU);
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