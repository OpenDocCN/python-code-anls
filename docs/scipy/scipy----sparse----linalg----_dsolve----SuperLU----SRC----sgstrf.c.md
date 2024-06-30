# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\sgstrf.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file sgstrf.c
 * \brief Computes an LU factorization of a general sparse matrix
 *
 * <pre>
 * -- SuperLU routine (version 3.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * October 15, 2003
 * 
 * Copyright (c) 1994 by Xerox Corporation.  All rights reserved.
 *
 * THIS MATERIAL IS PROVIDED AS IS, WITH ABSOLUTELY NO WARRANTY
 * EXPRESSED OR IMPLIED.  ANY USE IS AT YOUR OWN RISK.
 * 
 * Permission is hereby granted to use or copy this program for any
 * purpose, provided the above notices are retained on all copies.
 * Permission to modify the code and to distribute modified code is
 * granted, provided the above notices are retained, and a notice that
 * the code was modified is included with the above copyright notice.
 * </pre>
 */


#include "slu_sdefs.h"

void
sgstrf (superlu_options_t *options, SuperMatrix *A,
        int relax, int panel_size, int *etree, void *work, int_t lwork,
        int *perm_c, int *perm_r, SuperMatrix *L, SuperMatrix *U,
        GlobalLU_t *Glu, /* persistent to facilitate multiple factorizations */
        SuperLUStat_t *stat, int_t *info)
{
    /* Local working arrays */
    NCPformat *Astore;
    int       *iperm_r = NULL; /* inverse of perm_r; used when 
                                  options->Fact == SamePattern_SameRowPerm */
    int       *iperm_c; /* inverse of perm_c */
    int       *iwork;
    float    *swork;
    int          *segrep, *repfnz, *parent;
    int          *panel_lsub; /* dense[]/panel_lsub[] pair forms a w-wide SPA */
    int_t     *xprune, *xplore;
    int          *marker;
    float    *dense, *tempv;
    int       *relax_end;
    float    *a;
    int_t     *asub, *xa_begin, *xa_end;
    int_t     *xlsub, *xlusup, *xusub;
    int       *xsup, *supno;
    int_t     nzlumax;
    float fill_ratio = sp_ienv(6);  /* estimated fill ratio */

    /* Local scalars */
    fact_t    fact = options->Fact;  // 获取LU分解的类型（例如，是否保留相同的模式和行排列）
    double    diag_pivot_thresh = options->DiagPivotThresh;  // 对角元素主元的阈值
    int       pivrow;   /* pivotal row number in the original matrix A */  // 原始矩阵A中的主元行号
    int       nseg1;    /* no of segments in U-column above panel row jcol */  // jcol上面的U列中的段数
    int       nseg;    /* no of segments in each U-column */  // 每个U列中的段数
    register int jcol, jj;
    register int kcol;    /* end column of a relaxed snode */  // 放松的超结点的结束列
    register int icol;
    int_t     i, k, iinfo, new_next, nextlu, nextu;
    int       m, n, min_mn, jsupno, fsupc;
    int       w_def;    /* upper bound on panel width */  // 面板宽度的上限
    int       usepr, iperm_r_allocated = 0;
    int_t     nnzL, nnzU;
    int       *panel_histo = stat->panel_histo;  // 统计信息中的面板直方图
    flops_t   *ops = stat->ops;  // 统计信息中的浮点操作数

    iinfo    = 0;  // 初始化信息
    // 获取矩阵 A 的行数和列数
    m        = A->nrow;
    n        = A->ncol;
    // 计算 m 和 n 中较小的一个
    min_mn   = SUPERLU_MIN(m, n);
    // 获取矩阵 A 的存储结构
    Astore   = A->Store;
    // 获取矩阵 A 的非零元素数组
    a        = Astore->nzval;
    // 获取矩阵 A 的行索引数组
    asub     = Astore->rowind;
    // 获取矩阵 A 的每列起始位置数组
    xa_begin = Astore->colbeg;
    // 获取矩阵 A 的每列结束位置数组
    xa_end   = Astore->colend;

    /* 分配用于因子化过程的存储空间 */
    // 初始化因子化过程中需要的存储空间
    *info = sLUMemInit(fact, work, lwork, m, n, Astore->nnz,
                       panel_size, fill_ratio, L, U, Glu, &iwork, &swork);
    // 如果初始化出错，则返回
    if ( *info ) return;
    
    // 获取全局超节点结构中的相关数组
    xsup    = Glu->xsup;
    supno   = Glu->supno;
    xlsub   = Glu->xlsub;
    xlusup  = Glu->xlusup;
    xusub   = Glu->xusub;
    
    // 设置用于因子化过程中的工作空间
    SetIWork(m, n, panel_size, iwork, &segrep, &parent, &xplore,
         &repfnz, &panel_lsub, &xprune, &marker);
    // 设置用于因子化过程中的实数工作空间
    sSetRWork(m, panel_size, swork, &dense, &tempv);
    
    // 检查是否使用相同的行置换来计算因子
    usepr = (fact == SamePattern_SameRowPerm);
    // 如果使用相同的行置换，则计算行置换的逆 iperm_r
    if ( usepr ) {
        iperm_r = (int *) int32Malloc(m);
        for (k = 0; k < m; ++k) iperm_r[perm_r[k]] = k;
        iperm_r_allocated = 1;
    }
    // 计算列置换的逆 iperm_c
    iperm_c = (int *) int32Malloc(n);
    for (k = 0; k < n; ++k) iperm_c[perm_c[k]] = k;

    /* 识别放松超节点 */
    // 分配用于放松的结束标记数组
    relax_end = (int *) intMalloc(n);
    // 如果使用对称模式，则使用堆放松超节点
    if ( options->SymmetricMode == YES ) {
        heap_relax_snode(n, etree, relax, marker, relax_end); 
    } else {
        // 否则使用常规放松超节点
        relax_snode(n, etree, relax, marker, relax_end); 
    }
    
    // 初始化 perm_r 和 marker 数组
    ifill (perm_r, m, EMPTY);
    ifill (marker, m * NO_MARKER, EMPTY);
    // 设置第一个超节点的编号和起始位置
    supno[0] = -1;
    xsup[0]  = xlsub[0] = xusub[0] = xlusup[0] = 0;
    // 设置默认的工作空间大小
    w_def    = panel_size;

    /* 
     * 逐个处理一个“面板”。面板可以是以下之一：
     *       (a) 树结构的底部的一个放松超节点，或者
     *       (b) 用户定义的 panel_size 个连续列
     */
    // 逐个处理面板
    for (jcol = 0; jcol < min_mn; ) {
    if ( relax_end[jcol] != EMPTY ) { /* 如果 relax_end[jcol] 不等于 EMPTY，则说明这是一个松弛的超节点的开始 */
           kcol = relax_end[jcol];      /* kcol 是松弛超节点的结束列号 */
        panel_histo[kcol-jcol+1]++;    /* 更新 panel_histo 数组，记录超节点的大小 */

        /* --------------------------------------
         * Factorize the relaxed supernode(jcol:kcol) 
         * 对松弛超节点(jcol:kcol)进行因式分解
         * -------------------------------------- */
        /* Determine the union of the row structure of the snode */
        /* 确定超节点的行结构的并集 */
        if ( (*info = ssnode_dfs(jcol, kcol, asub, xa_begin, xa_end,
                    xprune, marker, Glu)) != 0 )
        return;

            nextu    = xusub[jcol];     /* 下一个非零元在 xusub 中的索引 */
        nextlu   = xlusup[jcol];       /* 下一个非零元在 xlusup 中的索引 */
        jsupno   = supno[jcol];        /* 超节点的编号 */
        fsupc    = xsup[jsupno];       /* 超节点的第一列在 xsup 中的位置 */
        new_next = nextlu + (xlsub[fsupc+1]-xlsub[fsupc])*(kcol-jcol+1); /* 计算新的下一个位置 */
        nzlumax = Glu->nzlumax;        /* LU 分解内存的最大大小 */
        while ( new_next > nzlumax ) { /* 如果新的位置超过了当前内存大小，需要扩展内存 */
        if ( (*info = sLUMemXpand(jcol, nextlu, LUSUP, &nzlumax, Glu)) )
            return;
        }
    
        for (icol = jcol; icol<= kcol; icol++) { /* 遍历超节点的所有列 */
        xusub[icol+1] = nextu;           /* 更新 xusub 的下一个非零元索引 */
        
            /* Scatter into SPA dense[*] */
            /* 将数据散布到密集矩阵 dense[*] 中 */
            for (k = xa_begin[icol]; k < xa_end[icol]; k++)
                dense[asub[k]] = a[k];

               /* Numeric update within the snode */
            /* 在超节点内进行数值更新 */
            ssnode_bmod(icol, jsupno, fsupc, dense, tempv, Glu, stat);

        if ( (*info = spivotL(icol, diag_pivot_thresh, &usepr, perm_r,
                      iperm_r, iperm_c, &pivrow, Glu, stat)) )
            if ( iinfo == 0 ) iinfo = *info;
        
#if ( DEBUGlevel>=2 )
        sprint_lu_col("[1]: ", icol, pivrow, xprune, Glu);
#endif

        }

        jcol = icol;

    } else { /* Work on one panel of panel_size columns */
        
        /* Adjust panel_size so that a panel won't overlap with the next 
         * relaxed snode.
         */
        panel_size = w_def;
        for (k = jcol + 1; k < SUPERLU_MIN(jcol+panel_size, min_mn); k++) 
        if ( relax_end[k] != EMPTY ) {
            panel_size = k - jcol;
            break;
        }
        if ( k == min_mn ) panel_size = min_mn - jcol;
        panel_histo[panel_size]++;

        /* symbolic factor on a panel of columns */
        // 执行列面板的符号因子分解
        spanel_dfs(m, panel_size, jcol, A, perm_r, &nseg1,
              dense, panel_lsub, segrep, repfnz, xprune,
              marker, parent, xplore, Glu);
        
        /* numeric sup-panel updates in topological order */
        // 在拓扑顺序下更新数值超面板
        spanel_bmod(m, panel_size, jcol, nseg1, dense,
                tempv, segrep, repfnz, Glu, stat);
        
        /* Sparse LU within the panel, and below panel diagonal */
        // 在列面板内以及面板对角线以下进行稀疏LU分解
            for ( jj = jcol; jj < jcol + panel_size; jj++) {
         k = (jj - jcol) * m; /* column index for w-wide arrays */

        nseg = nseg1;    /* Begin after all the panel segments */

            if ((*info = scolumn_dfs(m, jj, perm_r, &nseg, &panel_lsub[k],
                    segrep, &repfnz[k], xprune, marker,
                    parent, xplore, Glu)) != 0) return;

              /* Numeric updates */
            // 数值更新
            if ((*info = scolumn_bmod(jj, (nseg - nseg1), &dense[k],
                     tempv, &segrep[nseg1], &repfnz[k],
                     jcol, Glu, stat)) != 0) return;
        
            /* Copy the U-segments to ucol[*] */
        // 将U段复制到ucol[*]
        if ((*info = scopy_to_ucol(jj, nseg, segrep, &repfnz[k],
                      perm_r, &dense[k], Glu)) != 0)
            return;

            if ( (*info = spivotL(jj, diag_pivot_thresh, &usepr, perm_r,
                      iperm_r, iperm_c, &pivrow, Glu, stat)) )
            if ( iinfo == 0 ) iinfo = *info;

        /* Prune columns (0:jj-1) using column jj */
            // 使用列jj修剪列(0:jj-1)
            spruneL(jj, perm_r, pivrow, nseg, segrep,
                        &repfnz[k], xprune, Glu);

        /* Reset repfnz[] for this column */
            // 重置此列的repfnz[]
            resetrep_col (nseg, segrep, &repfnz[k]);
        
#if ( DEBUGlevel>=2 )
        sprint_lu_col("[2]: ", jj, pivrow, xprune, Glu);
#endif

        }

           jcol += panel_size;    /* Move to the next panel */

    } /* else */

    } /* for */

    *info = iinfo;

    /* Complete perm_r[] for rank-deficient or tall-skinny matrices */
    /* k is the rank of U
       pivots have been completed for rows < k
       Now fill in the pivots for rows k to m */
    k = iinfo == 0 ? n : (int)iinfo - 1;
    if (m > k) {
        /* 如果 m 大于 k，则执行以下操作 */
        /* 如果 k 等于 m，则所有行排列已完成，可以提前结束对向量的遍历 */
        for (i = 0; i < m && k < m; ++i) {
            /* 遍历行索引数组 perm_r */
            if (perm_r[i] == EMPTY) {
                /* 如果当前位置为空（未填充），则将 k 填入 */
                perm_r[i] = k;
                ++k;
            }
        }
    }
    
    countnz(min_mn, xprune, &nnzL, &nnzU, Glu);
    /* 计算非零元素的数量，更新 nnzL 和 nnzU */
    fixupL(min_mn, perm_r, Glu);
    /* 修正 L 矩阵，根据行排列 perm_r 进行调整 */
    
    sLUWorkFree(iwork, swork, Glu); /* 释放工作空间和压缩存储空间 */
    SUPERLU_FREE(xplore); /* 释放 xplore 的内存空间 */
    SUPERLU_FREE(xprune); /* 释放 xprune 的内存空间 */

    if ( fact == SamePattern_SameRowPerm ) {
        /* 如果使用相同的模式和行排列 */
        /* 由于可能的不同的选主元，L 和 U 结构可能已经改变，尽管存储仍然可用 */
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
        /* 否则重新创建超节点矩阵 L 和压缩列矩阵 U */
        sCreate_SuperNode_Matrix(L, A->nrow, min_mn, nnzL, 
            (float *) Glu->lusup, Glu->xlusup, 
            Glu->lsub, Glu->xlsub, Glu->supno, Glu->xsup,
            SLU_SC, SLU_S, SLU_TRLU);
        sCreate_CompCol_Matrix(U, min_mn, min_mn, nnzU, 
            (float *) Glu->ucol, Glu->usub, Glu->xusub,
            SLU_NC, SLU_S, SLU_TRU);
    }
    
    ops[FACT] += ops[TRSV] + ops[GEMV];
    /* 更新操作统计信息，FACT += TRSV + GEMV */
    stat->expansions = --(Glu->num_expansions);
    /* 更新统计信息中的扩展次数，并减少 num_expansions */

    if ( iperm_r_allocated ) SUPERLU_FREE (iperm_r);
    /* 如果 iperm_r 已经分配，则释放其内存空间 */
    SUPERLU_FREE (iperm_c); /* 释放 iperm_c 的内存空间 */
    SUPERLU_FREE (relax_end); /* 释放 relax_end 的内存空间 */
}



# 这行代码是一个单独的右花括号 '}'，用于结束一个代码块或者字典的定义。
```