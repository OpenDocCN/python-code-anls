# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dgsitrf.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file dgsitrf.c
 * \brief Computes an ILU factorization of a general sparse matrix
 *
 * <pre>
 * -- SuperLU routine (version 4.1) --
 * Lawrence Berkeley National Laboratory.
 * June 30, 2009
 *
 * </pre>
 */

#include "slu_ddefs.h"

#ifdef DEBUG
int num_drop_L;
#endif

void
dgsitrf(superlu_options_t *options, SuperMatrix *A, int relax, int panel_size,
    int *etree, void *work, int_t lwork, int *perm_c, int *perm_r,
    SuperMatrix *L, SuperMatrix *U, 
        GlobalLU_t *Glu, /* persistent to facilitate multiple factorizations */
    SuperLUStat_t *stat, int_t *info)
{
    /* Local working arrays */
    NCPformat *Astore;  /*!< Pointer to the local storage of A in column-wise format */
    int       *iperm_r = NULL; /*!< Inverse of perm_r; used under specific conditions */
    int       *iperm_c; /*!< Inverse of perm_c */
    int       *swap, *iswap; /*!< Arrays for row permutations and their inverses */
    int       *iwork; /*!< Integer work array */
    double   *dwork; /*!< Double precision work array */
    int       *segrep, *repfnz, *parent; /*!< Arrays for segment representations */
    int_t     *xplore; /*!< Integer array for exploring matrix structure */
    int       *panel_lsub; /*!< Array for panel's supernode structure */
    int       *marker; /*!< Marker array */
    int       *marker_relax; /*!< Marker array for relaxation */
    double    *dense, *tempv; /*!< Dense vector and temporary array */
    int       *relax_end, *relax_fsupc; /*!< Arrays for relaxation endpoints and supernode columns */
    double    *a; /*!< Double precision array */
    int_t     *asub; /*!< Integer array for supernode subscripts */
    int_t     *xa_begin, *xa_end; /*!< Arrays for indexing supernodes */
    int       *xsup, *supno; /*!< Arrays for supernode structure */
    int_t     *xlsub, *xlusup, *xusub; /*!< Arrays for sparse LU factorization */
    int_t     nzlumax; /*!< Maximum nonzeros in LU factors */
    double    *amax;  /*!< Maximum absolute value in LU factors */
    double    drop_sum; /*!< Sum used in dropping rules */
    double alpha, omega;  /*!< Constants used in MILU */
    double    *dwork2; /*!< Secondary work array */

    /* Local scalars */
    fact_t    fact = options->Fact; /*!< Type of factorization */
    double    diag_pivot_thresh = options->DiagPivotThresh; /*!< Threshold for diagonal pivoting */
    double    drop_tol = options->ILU_DropTol; /*!< Drop tolerance */
    double    fill_ini = options->ILU_FillTol; /*!< Fill tolerance */
    double    gamma = options->ILU_FillFactor; /*!< Fill factor */
    int       drop_rule = options->ILU_DropRule; /*!< Drop rule */
    milu_t    milu = options->ILU_MILU; /*!< Type of MILU */
    double    fill_tol; /*!< Fill tolerance (computed) */
    int       pivrow;    /*!< Pivotal row number in original matrix A */
    int       nseg1;    /*!< Number of segments in U-column above panel row jcol */
    int       nseg;    /*!< Number of segments in each U-column */
    register int jcol, jj; /*!< Loop variables */
    register int kcol;    /*!< End column of a relaxed supernode */
    register int icol; /*!< Column index */
    int_t     i, k, iinfo; /*!< Loop indices and info variable */
    int       m, n, min_mn, jsupno, fsupc; /*!< Matrix dimensions and supernode information */
    int_t     new_next, nextlu, nextu; /*!< Pointers for LU factorization */
    int       w_def;    /*!< Upper bound on panel width */
    int       usepr, iperm_r_allocated = 0;
    int_t     nnzL, nnzU;
    int       *panel_histo = stat->panel_histo;
    flops_t   *ops = stat->ops;

    int       last_drop;/* the last column which the dropping rules applied */
    int       quota;
    int       nnzAj;    /* number of nonzeros in A(:,1:j) */
    int       nnzLj, nnzUj;
    double    tol_L = drop_tol, tol_U = drop_tol;
    double zero = 0.0;
    double one = 1.0;

    /* Executable */       
    iinfo    = 0; /* Initialize error flag to zero */
    m         = A->nrow; /* Number of rows in matrix A */
    n         = A->ncol; /* Number of columns in matrix A */
    min_mn   = SUPERLU_MIN(m, n); /* Minimum of m and n */
    Astore   = A->Store; /* Access the data storage of matrix A */
    a         = Astore->nzval; /* Nonzero values in A */
    asub     = Astore->rowind; /* Row indices of A */
    xa_begin = Astore->colbeg; /* Beginning positions of columns in A */
    xa_end   = Astore->colend; /* Ending positions of columns in A */

    /* Allocate storage common to the factor routines */
    *info = dLUMemInit(fact, work, lwork, m, n, Astore->nnz, panel_size,
               gamma, L, U, Glu, &iwork, &dwork);
    if ( *info ) return; /* Return if memory initialization fails */

    xsup    = Glu->xsup; /* Supernode beginning in factorized matrix */
    supno   = Glu->supno; /* Supernode numbers */
    xlsub   = Glu->xlsub; /* Starting position of each column in L-subscript */
    xlusup  = Glu->xlusup; /* Starting position of each supernode in LUSUP */
    xusub   = Glu->xusub; /* Starting position of each column in U-subscript */

    int_t *xprune; /* Pointer for pruning supernodes */
    SetIWork(m, n, panel_size, iwork, &segrep, &parent, &xplore,
         &repfnz, &panel_lsub, &xprune, &marker); /* Initialize integer workspace */
    marker_relax = int32Malloc(m); /* Allocate marker for relaxed supernodes */
    SUPERLU_FREE(xprune); /* Free memory for pruning array (not used in ILU) */
    
    dSetRWork(m, panel_size, dwork, &dense, &tempv); /* Initialize double precision workspace */

    usepr = (fact == SamePattern_SameRowPerm); /* Check if the same row permutation is used */
    if ( usepr ) {
    /* Compute the inverse of perm_r */
    iperm_r = (int *) int32Malloc(m); /* Allocate memory for row permutation */
    for (k = 0; k < m; ++k) iperm_r[perm_r[k]] = k; /* Compute inverse row permutation */
    iperm_r_allocated = 1; /* Mark row permutation as allocated */
    }

    iperm_c = (int *) int32Malloc(n); /* Allocate memory for column permutation */
    for (k = 0; k < n; ++k) iperm_c[perm_c[k]] = k; /* Compute column permutation */
    swap = (int *)intMalloc(n); /* Allocate memory for swapping */
    for (k = 0; k < n; k++) swap[k] = iperm_c[k]; /* Initialize swap array */
    iswap = (int *)intMalloc(n); /* Allocate memory for inverse swapping */
    for (k = 0; k < n; k++) iswap[k] = perm_c[k]; /* Initialize inverse swap array */
    amax = (double *) SUPERLU_MALLOC(panel_size * sizeof(double)); /* Allocate memory for maximum element in each column */
    if (drop_rule & DROP_SECONDARY)
    dwork2 = SUPERLU_MALLOC(n * sizeof(double)); /* Allocate workspace for secondary drop rule */
    else
    dwork2 = NULL; /* Set secondary workspace to NULL if secondary drop rule is not used */

    nnzAj = 0; /* Initialize number of nonzeros in A(:,1:j) */
    nnzLj = 0; /* Initialize number of nonzeros in L(:,1:j) */
    nnzUj = 0; /* Initialize number of nonzeros in U(:,1:j) */
    last_drop = SUPERLU_MAX(min_mn - 2 * sp_ienv(7), (int)(min_mn * 0.95)); /* Determine last column where dropping rules apply */
    alpha = pow((double)n, -1.0 / options->ILU_MILU_Dim); /* Compute alpha for ILU */

    /* Identify relaxed snodes */
    relax_end = (int *) intMalloc(n); /* Allocate memory for end markers of relaxed supernodes */
    relax_fsupc = (int *) intMalloc(n); /* Allocate memory for first supernode columns */
    if ( options->SymmetricMode == YES )
    ilu_heap_relax_snode(n, etree, relax, marker, relax_end, relax_fsupc); /* Relax supernodes for symmetric matrix */
    else
    ilu_relax_snode(n, etree, relax, marker, relax_end, relax_fsupc); /* Relax supernodes for general matrix */

    ifill (perm_r, m, EMPTY); /* Initialize perm_r with EMPTY */
    ifill (marker, m * NO_MARKER, EMPTY); /* Initialize marker with EMPTY */
    supno[0] = -1; /* Set first supernode number to -1 */
    xsup[0]  = xlsub[0] = xusub[0] = xlusup[0] = 0; /* Initialize supernode and column indices */
    w_def    = panel_size; /* Set default panel size */

    /* Mark the rows used by relaxed supernodes */
    ifill (marker_relax, m, EMPTY); /* Initialize marker for relaxed supernodes */
    i = mark_relax(m, relax_end, relax_fsupc, xa_begin, xa_end,
             asub, marker_relax); /* Mark rows used by relaxed supernodes */
#if ( PRNTlevel >= 1)
    // 如果 PRNTlevel 大于等于 1，则打印放松超节点的数量
    printf("%d relaxed supernodes.\n", (int)i);
#endif

    /*
     * 逐个处理一个“面板”。一个面板可以是以下之一：
     *       (a) 树的底部的一个放松超节点，或者
     *       (b) 用户定义的 panel_size 个连续列
     */
    for (jcol = 0; jcol < min_mn; ) {

    if ( relax_end[jcol] != EMPTY ) { /* start of a relaxed snode */
        kcol = relax_end[jcol];      /* end of the relaxed snode */
        panel_histo[kcol-jcol+1]++;

        /* Drop small rows in the previous supernode. */
        // 如果 jcol 大于 0 并且小于 last_drop，则处理上一个超节点中的小行
        if (jcol > 0 && jcol < last_drop) {
        int first = xsup[supno[jcol - 1]];
        int last = jcol - 1;
        int quota;

        /* Compute the quota */
        // 计算配额
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
        // 丢弃小行
        i = ilu_ddrop_row(options, first, last, tol_L, quota, &nnzLj,
                  &fill_tol, Glu, tempv, dwork2, 0);
        /* Reset the parameters */
        // 重置参数
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
        }

        /* --------------------------------------
         * Factorize the relaxed supernode(jcol:kcol)
         * -------------------------------------- */
        /* 确定松散节点(snode)的行结构的并集 */
        if ( (*info = ilu_dsnode_dfs(jcol, kcol, asub, xa_begin, xa_end,
                     marker, Glu)) != 0 )
        return;

        nextu    = xusub[jcol];
        nextlu   = xlusup[jcol];
        jsupno   = supno[jcol];
        fsupc    = xsup[jsupno];
        new_next = nextlu + (xlsub[fsupc+1]-xlsub[fsupc])*(kcol-jcol+1);
        nzlumax = Glu->nzlumax;
        while ( new_next > nzlumax ) {
        if ((*info = dLUMemXpand(jcol, nextlu, LUSUP, &nzlumax, Glu)))
            return;
        }

        for (icol = jcol; icol <= kcol; icol++) {
        xusub[icol+1] = nextu;

        amax[0] = 0.0;
        /* 将稀疏模式(asub)散布到密集向量(dense)中 */
        for (k = xa_begin[icol]; k < xa_end[icol]; k++) {
            register double tmp = fabs(a[k]);
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

        /* snode内的数值更新 */
        dsnode_bmod(icol, jsupno, fsupc, dense, tempv, Glu, stat);

        if (usepr) pivrow = iperm_r[icol];
        fill_tol = pow(fill_ini, 1.0 - (double)icol / (double)min_mn);
        if ( (*info = ilu_dpivotL(icol, diag_pivot_thresh, &usepr,
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
        /* 处理一个 panel_size 列的面板 */

        /* Adjust panel_size so that a panel won't overlap with the next
         * relaxed snode.
         */
        /* 调整 panel_size，使得一个面板不会与下一个松弛的 snode 重叠 */
        panel_size = w_def;
        for (k = jcol + 1; k < SUPERLU_MIN(jcol+panel_size, min_mn); k++)
        if ( relax_end[k] != EMPTY ) {
            panel_size = k - jcol;
            break;
        }
        if ( k == min_mn ) panel_size = min_mn - jcol;
        panel_histo[panel_size]++;

        /* symbolic factor on a panel of columns */
        /* 对列面板进行符号因子分解 */
        ilu_dpanel_dfs(m, panel_size, jcol, A, perm_r, &nseg1,
                          dense, amax, panel_lsub, segrep, repfnz,
                          marker, parent, xplore, Glu);

        /* numeric sup-panel updates in topological order */
        /* 按拓扑顺序更新数值的超面板 */
        dpanel_bmod(m, panel_size, jcol, nseg1, dense,
            tempv, segrep, repfnz, Glu, stat);

        /* Sparse LU within the panel, and below panel diagonal */
        /* 在面板内和面板对角线以下进行稀疏LU分解 */
        for (jj = jcol; jj < jcol + panel_size; jj++) {

        k = (jj - jcol) * m; /* column index for w-wide arrays */
        /* 对于 w-wide 数组，计算列索引 */

        nseg = nseg1;    /* Begin after all the panel segments */
        /* 在所有面板段之后开始 */

        nnzAj += xa_end[jj] - xa_begin[jj];
        /* 更新 nnzAj，计算列 jj 的非零元素个数 */

        if ((*info = ilu_dcolumn_dfs(m, jj, perm_r, &nseg,
                         &panel_lsub[k], segrep, &repfnz[k],
                         marker, parent, xplore, Glu)))
            return;
        /* 使用深度优先搜索进行列符号因子分解 */

        /* Numeric updates */
        /* 数值更新 */
        if ((*info = dcolumn_bmod(jj, (nseg - nseg1), &dense[k],
                      tempv, &segrep[nseg1], &repfnz[k],
                      jcol, Glu, stat)) != 0) return;

        /* Make a fill-in position if the column is entirely zero */
        /* 如果列完全为零，则进行填充位置 */
        if (xlsub[jj + 1] == xlsub[jj]) {
            register int i, row;
            int_t nextl;
            int_t nzlmax = Glu->nzlmax;
            int_t *lsub = Glu->lsub;
            int *marker2 = marker + 2 * m;

            /* Allocate memory */
            /* 分配内存 */
            nextl = xlsub[jj] + 1;
            if (nextl >= nzlmax) {
            int error = dLUMemXpand(jj, nextl, LSUB, &nzlmax, Glu);
            if (error) { *info = error; return; }
            lsub = Glu->lsub;
            }
            xlsub[jj + 1]++;
            assert(xlusup[jj]==xlusup[jj+1]);
            xlusup[jj + 1]++;
            ((double *) Glu->lusup)[xlusup[jj]] = zero;

            /* Choose a row index (pivrow) for fill-in */
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

    ilu_countnz(min_mn, &nnzL, &nnzU, Glu);
    fixupL(min_mn, perm_r, Glu);

    dLUWorkFree(iwork, dwork, Glu); /* Free work space and compress storage */
    SUPERLU_FREE(xplore);
    SUPERLU_FREE(marker_relax);

    if ( fact == SamePattern_SameRowPerm ) {
    /* L and U structures may have changed due to possibly different
       pivoting, even though the storage is available.
       There could also be memory expansions, so the array locations
       may have changed, */
    ((SCformat *)L->Store)->nnz = nnzL;
    ((SCformat *)L->Store)->nsuper = Glu->supno[n];
    ((SCformat *)L->Store)->nzval = (double *) Glu->lusup;
    ((SCformat *)L->Store)->nzval_colptr = Glu->xlusup;
    ((SCformat *)L->Store)->rowind = Glu->lsub;
    ((SCformat *)L->Store)->rowind_colptr = Glu->xlsub;
    ((NCformat *)U->Store)->nnz = nnzU;
    ((NCformat *)U->Store)->nzval = (double *) Glu->ucol;
    ((NCformat *)U->Store)->rowind = Glu->usub;
    ((NCformat *)U->Store)->colptr = Glu->xusub;
    } else {
    dCreate_SuperNode_Matrix(L, A->nrow, min_mn, nnzL,
              (double *) Glu->lusup, Glu->xlusup,
              Glu->lsub, Glu->xlsub, Glu->supno, Glu->xsup,
          SLU_SC, SLU_D, SLU_TRLU);
    dCreate_CompCol_Matrix(U, min_mn, min_mn, nnzU,
          (double *) Glu->ucol, Glu->usub, Glu->xusub,
          SLU_NC, SLU_D, SLU_TRU);
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
    if ( dwork2 ) SUPERLU_FREE (dwork2);

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

    dLUWorkFree(iwork, dwork, Glu); /* Free work space and compress storage */
    SUPERLU_FREE(xplore);
    SUPERLU_FREE(marker_relax);

    if ( fact == SamePattern_SameRowPerm ) {
    /* L and U structures may have changed due to possibly different
       pivoting, even though the storage is available.
       There could also be memory expansions, so the array locations
       may have changed, */
    ((SCformat *)L->Store)->nnz = nnzL;
    ((SCformat *)L->Store)->nsuper = Glu->supno[n];
    ((SCformat *)L->Store)->nzval = (double *) Glu->lusup;
    ((SCformat *)L->Store)->nzval_colptr = Glu->xlusup;
    ((SCformat *)L->Store)->rowind = Glu->lsub;
    ((SCformat *)L->Store)->rowind_colptr = Glu->xlsub;
    ((NCformat *)U->Store)->nnz = nnzU;
    ((NCformat *)U->Store)->nzval = (double *) Glu->ucol;
    ((NCformat *)U->Store)->rowind = Glu->usub;
    ((NCformat *)U->Store)->colptr = Glu->xusub;
    } else {
    dCreate_SuperNode_Matrix(L, A->nrow, min_mn, nnzL,
              (double *) Glu->lusup, Glu->xlusup,
              Glu->lsub, Glu->xlsub, Glu->supno, Glu->xsup,
          SLU_SC, SLU_D, SLU_TRLU);
    dCreate_CompCol_Matrix(U, min_mn, min_mn, nnzU,
          (double *) Glu->ucol, Glu->usub, Glu->xusub,
          SLU_NC, SLU_D, SLU_TRU);
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
    if ( dwork2 ) SUPERLU_FREE (dwork2);

}
```