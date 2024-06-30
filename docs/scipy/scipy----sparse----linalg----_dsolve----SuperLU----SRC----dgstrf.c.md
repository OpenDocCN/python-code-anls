# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dgstrf.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file dgstrf.c
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


#include "slu_ddefs.h"

void
dgstrf (superlu_options_t *options, SuperMatrix *A,
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
    double    *dwork;
    int          *segrep, *repfnz, *parent;
    int          *panel_lsub; /* dense[]/panel_lsub[] pair forms a w-wide SPA */
    int_t     *xprune, *xplore;
    int          *marker;
    double    *dense, *tempv;
    int       *relax_end;
    double    *a;
    int_t     *asub, *xa_begin, *xa_end;
    int_t     *xlsub, *xlusup, *xusub;
    int       *xsup, *supno;
    int_t     nzlumax;
    double fill_ratio = sp_ienv(6);  /* estimated fill ratio */

    /* Local scalars */
    fact_t    fact = options->Fact;
    double    diag_pivot_thresh = options->DiagPivotThresh;
    int       pivrow;   /* pivotal row number in the original matrix A */
    int       nseg1;    /* no of segments in U-column above panel row jcol */
    int       nseg;    /* no of segments in each U-column */
    register int jcol, jj;
    register int kcol;    /* end column of a relaxed snode */
    register int icol;
    int_t     i, k, iinfo, new_next, nextlu, nextu;
    int       m, n, min_mn, jsupno, fsupc;
    int       w_def;    /* upper bound on panel width */
    int       usepr, iperm_r_allocated = 0;
    int_t     nnzL, nnzU;
    int       *panel_histo = stat->panel_histo;
    flops_t   *ops = stat->ops;

    iinfo    = 0;

    /* 
     * Beginning of the function dgstrf, which performs an LU factorization
     * of a sparse matrix A.
     */

    /* Initialize local variables and arrays */

    /* Astore points to the matrix A in column oriented format */
    Astore = A->Store;

    /* Initialize iperm_c as the inverse of column permutation perm_c */
    iperm_c = perm_c;

    /* Allocate memory for local working arrays */
    iwork = intMalloc(Astore->nrow);
    dwork = doubleMalloc(Astore->nrow * 5);

    /* Allocate memory for various arrays and structures */
    segrep = intMalloc(panel_size);
    repfnz = intMalloc(panel_size);
    parent = intMalloc(panel_size);
    panel_lsub = intMalloc(panel_size);
    xprune = intMalloc(panel_size);
    xplore = intMalloc(panel_size);
    marker = intCalloc(panel_size);
    dense = doubleCalloc(panel_size);
    tempv = doubleCalloc(panel_size);
    relax_end = intMalloc(panel_size);

    /* Initialize other variables and arrays */
    a = Astore->nzval;
    asub = Astore->rowind;
    xa_begin = Astore->colbeg;
    xa_end = Astore->colend;
    xlsub = L->Store->rowind;
    xlusup = L->Store->nzval;
    xusub = U->Store->rowind;
    xsup = Glu->xsup;
    supno = Glu->supno;

    nzlumax = sp_ienv(3); /* maximum no of nonzeros in L+U */

    /* Main computational loop begins */

    /*
     * Perform LU factorization with dynamic pivoting and numerical
     * thresholding to handle sparse matrices efficiently.
     */

    /* Compute estimated fill ratio for LU factors */
    fill_ratio = sp_ienv(6);
    // 获取矩阵 A 的行数 m
    m        = A->nrow;
    // 获取矩阵 A 的列数 n
    n        = A->ncol;
    // 计算矩阵 A 的行数和列数的最小值
    min_mn   = SUPERLU_MIN(m, n);
    // 获取矩阵 A 的存储结构
    Astore   = A->Store;
    // 获取矩阵 A 中非零元素的值数组
    a        = Astore->nzval;
    // 获取矩阵 A 中非零元素所在行的数组
    asub     = Astore->rowind;
    // 获取矩阵 A 中每列非零元素起始位置的数组
    xa_begin = Astore->colbeg;
    // 获取矩阵 A 中每列非零元素结束位置的数组
    xa_end   = Astore->colend;

    /* 分配因子化过程中通用的存储空间 */
    // 调用 dLUMemInit 函数初始化存储空间，并返回信息给 info
    *info = dLUMemInit(fact, work, lwork, m, n, Astore->nnz,
                       panel_size, fill_ratio, L, U, Glu, &iwork, &dwork);
    // 如果返回的 info 非零，则直接返回
    if ( *info ) return;
    
    // 获取全局 LU 分解数据结构中的相关数组
    xsup    = Glu->xsup;
    supno   = Glu->supno;
    xlsub   = Glu->xlsub;
    xlusup  = Glu->xlusup;
    xusub   = Glu->xusub;
    
    // 设置用于因子化过程中的工作区数组
    SetIWork(m, n, panel_size, iwork, &segrep, &parent, &xplore,
             &repfnz, &panel_lsub, &xprune, &marker);
    // 设置用于因子化过程中的双精度工作区数组
    dSetRWork(m, panel_size, dwork, &dense, &tempv);
    
    // 根据 fact 的值判断是否使用相同的行排列
    usepr = (fact == SamePattern_SameRowPerm);
    // 如果使用相同的行排列，则计算 perm_r 的逆置
    if ( usepr ) {
        /* 计算 perm_r 的逆 */
        iperm_r = (int *) int32Malloc(m);
        for (k = 0; k < m; ++k) iperm_r[perm_r[k]] = k;
        iperm_r_allocated = 1;
    }
    // 计算 perm_c 的逆置
    iperm_c = (int *) int32Malloc(n);
    for (k = 0; k < n; ++k) iperm_c[perm_c[k]] = k;

    /* 标识放松的超结点 */
    // 分配放松结束数组的存储空间
    relax_end = (int *) intMalloc(n);
    // 根据对称模式选项调用相应的函数计算放松的超结点
    if ( options->SymmetricMode == YES ) {
        heap_relax_snode(n, etree, relax, marker, relax_end); 
    } else {
        relax_snode(n, etree, relax, marker, relax_end); 
    }
    
    // 初始化 perm_r 和 marker 数组
    ifill (perm_r, m, EMPTY);
    ifill (marker, m * NO_MARKER, EMPTY);
    // 初始化超结点数组的第一个元素
    supno[0] = -1;
    // 初始化超结点相关数组的第一个元素
    xsup[0]  = xlsub[0] = xusub[0] = xlusup[0] = 0;
    // 设置默认的 panel 大小
    w_def    = panel_size;

    /* 
     * 逐个处理一个“panel”。一个“panel”可以是以下之一：
     *       (a) etree 的底部放松的超结点，或者
     *       (b) 用户定义的 panel_size 连续列
     */
    // 循环处理每个“panel”，jcol 用于追踪当前处理的列索引
    for (jcol = 0; jcol < min_mn; ) {
    if ( relax_end[jcol] != EMPTY ) { /* 如果 relax_end[jcol] 不是空值，则进入松弛列节点的处理 */
           kcol = relax_end[jcol];      /* kcol 是松弛列节点的结束列 */
        panel_histo[kcol-jcol+1]++;      /* 更新面板直方图，记录松弛节点的列数 */

        /* --------------------------------------
         * Factorize the relaxed supernode(jcol:kcol) 
         * -------------------------------------- */
        /* 因子分解松弛超节点 supernode(jcol:kcol) */

        /* Determine the union of the row structure of the snode */
        /* 确定松弛超节点的行结构的并集 */
        if ( (*info = dsnode_dfs(jcol, kcol, asub, xa_begin, xa_end,
                    xprune, marker, Glu)) != 0 )
            return;   /* 如果 dsnode_dfs 返回非零信息，则返回 */

        nextu    = xusub[jcol];     /* 下一个未使用的上三角元素的起始位置 */
        nextlu   = xlusup[jcol];    /* 下一个未使用的因子 L 的起始位置 */
        jsupno   = supno[jcol];     /* jcol 的超节点编号 */
        fsupc    = xsup[jsupno];    /* 超节点的首列 */
        new_next = nextlu + (xlsub[fsupc+1]-xlsub[fsupc])*(kcol-jcol+1); /* 计算新的因子 L 的结束位置 */
        nzlumax = Glu->nzlumax;    /* LU 因子分解所需的最大内存大小 */
        while ( new_next > nzlumax ) { /* 如果新的 L 因子位置超出了当前内存大小 */
            if ( (*info = dLUMemXpand(jcol, nextlu, LUSUP, &nzlumax, Glu)) ) /* 扩展内存 */
                return;   /* 如果扩展失败，则返回 */
        }
    
        for (icol = jcol; icol<= kcol; icol++) { /* 遍历松弛节点的每一列 */
            xusub[icol+1] = nextu;   /* 更新下一个未使用的上三角元素的起始位置 */
        
            /* Scatter into SPA dense[*] */
            /* 将数据散列到密集矩阵 dense[*] */
            for (k = xa_begin[icol]; k < xa_end[icol]; k++)
                dense[asub[k]] = a[k];

            /* Numeric update within the snode */
            /* 松弛节点内的数值更新 */
            dsnode_bmod(icol, jsupno, fsupc, dense, tempv, Glu, stat);

            if ( (*info = dpivotL(icol, diag_pivot_thresh, &usepr, perm_r,
                      iperm_r, iperm_c, &pivrow, Glu, stat)) ) /* 进行主元选取 */
                if ( iinfo == 0 ) iinfo = *info; /* 更新错误信息 */
        }
    }
#if ( DEBUGlevel>=2 )
dprint_lu_col("[1]: ", icol, pivrow, xprune, Glu);
#endif
/* 如果调试级别大于等于2，打印列的调试信息 */
}

jcol = icol;
/* 设置 jcol 等于 icol */

} else { /* Work on one panel of panel_size columns */

/* 调整 panel_size，以确保一个 panel 不会与下一个松弛的 snode 重叠 */
panel_size = w_def;
for (k = jcol + 1; k < SUPERLU_MIN(jcol+panel_size, min_mn); k++)
if ( relax_end[k] != EMPTY ) {
    panel_size = k - jcol;
    break;
}
if ( k == min_mn ) panel_size = min_mn - jcol;
panel_histo[panel_size]++;
/* 根据 relax_end 数组的状态调整 panel_size 的大小，并更新 panel_histo[] */

/* 在一组列的面板上进行符号因子分解 */
dpanel_dfs(m, panel_size, jcol, A, perm_r, &nseg1,
      dense, panel_lsub, segrep, repfnz, xprune,
      marker, parent, xplore, Glu);

/* 按照拓扑顺序对顶部面板进行数值更新 */
dpanel_bmod(m, panel_size, jcol, nseg1, dense,
        tempv, segrep, repfnz, Glu, stat);

/* 在面板内和面板对角线以下进行稀疏 LU 分解 */
    for ( jj = jcol; jj < jcol + panel_size; jj++) {
 k = (jj - jcol) * m; /* w-wide 数组的列索引 */

nseg = nseg1;    /* 在所有面板段之后开始 */

    if ((*info = dcolumn_dfs(m, jj, perm_r, &nseg, &panel_lsub[k],
            segrep, &repfnz[k], xprune, marker,
            parent, xplore, Glu)) != 0) return;

      /* 数值更新 */
    if ((*info = dcolumn_bmod(jj, (nseg - nseg1), &dense[k],
                 tempv, &segrep[nseg1], &repfnz[k],
                 jcol, Glu, stat)) != 0) return;

    /* 将 U 段复制到 ucol[*] */
if ((*info = dcopy_to_ucol(jj, nseg, segrep, &repfnz[k],
                  perm_r, &dense[k], Glu)) != 0)
    return;

    if ( (*info = dpivotL(jj, diag_pivot_thresh, &usepr, perm_r,
              iperm_r, iperm_c, &pivrow, Glu, stat)) )
    if ( iinfo == 0 ) iinfo = *info;

/* 使用列 jj 剪枝 (0:jj-1) */
    dpruneL(jj, perm_r, pivrow, nseg, segrep,
                &repfnz[k], xprune, Glu);

/* 重置此列的 repfnz[] */
    resetrep_col (nseg, segrep, &repfnz[k]);

#if ( DEBUGlevel>=2 )
dprint_lu_col("[2]: ", jj, pivrow, xprune, Glu);
#endif

}

   jcol += panel_size;    /* 移动到下一个面板 */

} /* else */

} /* for */

*info = iinfo;

/* 对于秩不足或高瘦矩阵，完成 perm_r[] */
/* k 是 U 的秩
   对于行 < k 已经完成了主元
   现在填充行 k 到 m 的主元 */
k = iinfo == 0 ? n : (int)iinfo - 1;
/* 设置 k 为 U 的秩 */
    if (m > k) {
        /* 如果 m > k，说明还有未完成的行置换 */

        /* 如果 k == m，则所有行置换都已完成，可以跳过后续向量的查找 */
        for (i = 0; i < m && k < m; ++i) {
            /* 遍历每一行，检查是否需要进行置换 */
            if (perm_r[i] == EMPTY) {
                /* 如果当前行的置换还未完成 */
                perm_r[i] = k;
                ++k; /* 将当前行置换为 k，并递增 k */
            }
        }
    }
    
    countnz(min_mn, xprune, &nnzL, &nnzU, Glu);
    /* 计算非零元素的个数，更新 nnzL 和 nnzU */
    fixupL(min_mn, perm_r, Glu);
    /* 修正 L 矩阵，应用行置换 perm_r */

    dLUWorkFree(iwork, dwork, Glu); /* 释放工作空间并压缩存储 */

    SUPERLU_FREE(xplore);
    SUPERLU_FREE(xprune);
    /* 释放超节点探索数组 xplore 和剪枝数组 xprune */

    if ( fact == SamePattern_SameRowPerm ) {
        /* 如果使用相同的模式和行置换 */

        /* 更新 L 和 U 的结构可能因不同的主元选取而改变，尽管存储空间是可用的。
           可能会有内存扩展，因此数组位置可能会改变。 */
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
        /* 否则，根据给定参数重新创建超节点矩阵 L 和压缩列矩阵 U */
        dCreate_SuperNode_Matrix(L, A->nrow, min_mn, nnzL, 
          (double *) Glu->lusup, Glu->xlusup, 
          Glu->lsub, Glu->xlsub, Glu->supno, Glu->xsup,
          SLU_SC, SLU_D, SLU_TRLU);
        dCreate_CompCol_Matrix(U, min_mn, min_mn, nnzU, 
          (double *) Glu->ucol, Glu->usub, Glu->xusub,
          SLU_NC, SLU_D, SLU_TRU);
    }
    
    ops[FACT] += ops[TRSV] + ops[GEMV];    
    /* 更新运算计数器 */

    stat->expansions = --(Glu->num_expansions);
    /* 更新扩展次数统计 */

    if ( iperm_r_allocated ) SUPERLU_FREE (iperm_r);
    /* 如果 iperm_r 已分配，则释放其内存空间 */
    SUPERLU_FREE (iperm_c);
    /* 释放 iperm_c 的内存空间 */
    SUPERLU_FREE (relax_end);
    /* 释放 relax_end 的内存空间 */
}



# 这行代码关闭了一个代码块。在大多数编程语言中，"}" 表示结束一个代码块的标志。
```