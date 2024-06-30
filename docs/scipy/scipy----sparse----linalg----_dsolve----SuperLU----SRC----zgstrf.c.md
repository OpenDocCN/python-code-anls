# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zgstrf.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file zgstrf.c
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


#include "slu_zdefs.h"

void
zgstrf (superlu_options_t *options, SuperMatrix *A,
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
    doublecomplex    *zwork;
    int          *segrep, *repfnz, *parent;
    int          *panel_lsub; /* dense[]/panel_lsub[] pair forms a w-wide SPA */
    int_t     *xprune, *xplore;
    int          *marker;
    doublecomplex    *dense, *tempv;
    int       *relax_end;
    doublecomplex    *a;
    int_t     *asub, *xa_begin, *xa_end;
    int_t     *xlsub, *xlusup, *xusub;
    int       *xsup, *supno;
    int_t     nzlumax;
    double fill_ratio = sp_ienv(6);  /* estimated fill ratio */

    /* Local scalars */
    fact_t    fact = options->Fact;  // 从参数 options 中获取 LU 分解的类型
    double    diag_pivot_thresh = options->DiagPivotThresh;  // 从 options 中获取对角线主元的阈值
    int       pivrow;   // 在原始矩阵 A 中的主元行号
    int       nseg1;    // 在 jcol 行之上 U 列中的段数
    int       nseg;     // 每个 U 列中的段数
    register int jcol, jj;  // 循环索引变量
    register int kcol;    // 放松超节点的结束列
    register int icol;
    int_t     i, k, iinfo, new_next, nextlu, nextu;
    int       m, n, min_mn, jsupno, fsupc;
    int       w_def;    // 面板宽度的上界
    int       usepr, iperm_r_allocated = 0;  // 是否使用行置换
    int_t     nnzL, nnzU;  // L 和 U 的非零元素数
    int       *panel_histo = stat->panel_histo;  // 面板直方图统计信息
    flops_t   *ops = stat->ops;  // 操作数统计信息
    iinfo    = 0;
    m        = A->nrow;
    n        = A->ncol;
    min_mn   = SUPERLU_MIN(m, n);  # 计算 m 和 n 的最小值，用于迭代的上限
    Astore   = A->Store;           # 获取矩阵 A 的存储结构
    a        = Astore->nzval;      # 获取矩阵 A 的非零元素数组
    asub     = Astore->rowind;     # 获取矩阵 A 的行索引数组
    xa_begin = Astore->colbeg;     # 获取矩阵 A 的列起始索引数组
    xa_end   = Astore->colend;     # 获取矩阵 A 的列结束索引数组

    /* Allocate storage common to the factor routines */
    *info = zLUMemInit(fact, work, lwork, m, n, Astore->nnz,
                       panel_size, fill_ratio, L, U, Glu, &iwork, &zwork);
    如果 *info 非零，说明内存初始化失败，返回结果
    if ( *info ) return;
    
    xsup    = Glu->xsup;           # 获取全局 LU 因子化结构的超节点起始索引数组
    supno   = Glu->supno;          # 获取全局 LU 因子化结构的超节点编号数组
    xlsub   = Glu->xlsub;          # 获取全局 LU 因子化结构的 L 部分列索引数组
    xlusup  = Glu->xlusup;         # 获取全局 LU 因子化结构的 U 部分超节点列索引数组
    xusub   = Glu->xusub;          # 获取全局 LU 因子化结构的 U 部分行索引数组
    
    SetIWork(m, n, panel_size, iwork, &segrep, &parent, &xplore,
         &repfnz, &panel_lsub, &xprune, &marker);  # 初始化工作空间中的整型数组
    zSetRWork(m, panel_size, zwork, &dense, &tempv);  # 初始化工作空间中的双精度浮点数组
    
    usepr = (fact == SamePattern_SameRowPerm);  # 根据是否使用相同模式和行置换判断
    if ( usepr ) {
    /* Compute the inverse of perm_r */
    iperm_r = (int *) int32Malloc(m);  # 分配存储空间来存放行置换的逆置换
    for (k = 0; k < m; ++k) iperm_r[perm_r[k]] = k;  # 计算行置换的逆置换
    iperm_r_allocated = 1;  # 标记行置换逆置换已分配
    }
    iperm_c = (int *) int32Malloc(n);  # 分配存储空间来存放列置换的逆置换
    for (k = 0; k < n; ++k) iperm_c[perm_c[k]] = k;  # 计算列置换的逆置换

    /* Identify relaxed snodes */
    relax_end = (int *) intMalloc(n);  # 分配存储空间来标识放松的超节点
    如果使用对称模式，调用对称模式的超节点放松函数，否则调用一般超节点放松函数
    if ( options->SymmetricMode == YES ) {
        heap_relax_snode(n, etree, relax, marker, relax_end); 
    } else {
        relax_snode(n, etree, relax, marker, relax_end); 
    }
    
    ifill (perm_r, m, EMPTY);  # 将行置换数组初始化为空
    ifill (marker, m * NO_MARKER, EMPTY);  # 将标记数组初始化为空
    supno[0] = -1;  # 初始化超节点编号数组的第一个元素为 -1
    xsup[0]  = xlsub[0] = xusub[0] = xlusup[0] = 0;  # 初始化超节点起始索引数组为 0
    w_def    = panel_size;  # 设置默认的面板大小

    /* 
     * Work on one "panel" at a time. A panel is one of the following: 
     *       (a) a relaxed supernode at the bottom of the etree, or
     *       (b) panel_size contiguous columns, defined by the user
     */
    for (jcol = 0; jcol < min_mn; ) {
    if ( relax_end[jcol] != EMPTY ) { /* 如果 relax_end[jcol] 不为空，表示开始一个放松的超节点 */
           kcol = relax_end[jcol];      /* kcol 是放松的超节点的结束列号 */
        panel_histo[kcol-jcol+1]++;   /* 更新 panel_histo 数组中对应超节点长度的计数 */

        /* --------------------------------------
         * Factorize the relaxed supernode(jcol:kcol) 
         * 对放松的超节点(jcol:kcol)进行因式分解
         * -------------------------------------- */
        /* Determine the union of the row structure of the snode */
        /* 确定超节点行结构的并集 */
        if ( (*info = zsnode_dfs(jcol, kcol, asub, xa_begin, xa_end,
                    xprune, marker, Glu)) != 0 )
        return;

            nextu    = xusub[jcol];  /* 下一个 U 列指针的位置 */
        nextlu   = xlusup[jcol];    /* 下一个 L 列指针的位置 */
        jsupno   = supno[jcol];     /* 当前超节点的超节点编号 */
        fsupc    = xsup[jsupno];    /* 当前超节点的首列的全局索引 */
        new_next = nextlu + (xlsub[fsupc+1]-xlsub[fsupc])*(kcol-jcol+1); /* 更新下一个 L 列指针的位置 */
        nzlumax = Glu->nzlumax;     /* LU 因子中非零元的最大数量 */
        while ( new_next > nzlumax ) { /* 如果新的 L 列指针位置超过了当前分配的空间 */
        if ( (*info = zLUMemXpand(jcol, nextlu, LUSUP, &nzlumax, Glu)) ) /* 扩展 LU 因子的内存空间 */
            return;
        }
    
        for (icol = jcol; icol<= kcol; icol++) { /* 遍历超节点中的每一列 */
        xusub[icol+1] = nextu;  /* 更新下一个 U 列指针的位置 */
        
            /* Scatter into SPA dense[*] */
            /* 将稀疏矩阵 A 的非零元散落到密集数组 dense[*] */
            for (k = xa_begin[icol]; k < xa_end[icol]; k++)
                dense[asub[k]] = a[k];

               /* Numeric update within the snode */
            /* 在超节点内进行数值更新 */
            zsnode_bmod(icol, jsupno, fsupc, dense, tempv, Glu, stat);

        if ( (*info = zpivotL(icol, diag_pivot_thresh, &usepr, perm_r,
                      iperm_r, iperm_c, &pivrow, Glu, stat)) ) /* 进行主元选取 */
            if ( iinfo == 0 ) iinfo = *info;
        }
#if ( DEBUGlevel>=2 )
zprint_lu_col("[1]: ", icol, pivrow, xprune, Glu);
#endif



// 如果调试级别大于等于2，则打印 "[1]: " 后跟 icol、pivrow、xprune、Glu 的值
if ( DEBUGlevel>=2 )
    zprint_lu_col("[1]: ", icol, pivrow, xprune, Glu);



}



// 结束 if 语句的代码块



jcol = icol;



// 将 jcol 设为 icol 的值
jcol = icol;



} else { /* Work on one panel of panel_size columns */



// 否则，处理 panel_size 列的一个面板
} else { /* Work on one panel of panel_size columns */



/* Adjust panel_size so that a panel won't overlap with the next 
 * relaxed snode.
 */



// 调整 panel_size，使得面板不会与下一个放松的超节点重叠



panel_size = w_def;
for (k = jcol + 1; k < SUPERLU_MIN(jcol+panel_size, min_mn); k++) 
if ( relax_end[k] != EMPTY ) {
    panel_size = k - jcol;
    break;
}
if ( k == min_mn ) panel_size = min_mn - jcol;
panel_histo[panel_size]++;



// 设置 panel_size 初始值为 w_def
panel_size = w_def;
// 根据 relax_end 数组，找到第一个不为空的位置，调整 panel_size
for (k = jcol + 1; k < SUPERLU_MIN(jcol+panel_size, min_mn); k++) 
    if ( relax_end[k] != EMPTY ) {
        panel_size = k - jcol;
        break;
    }
// 如果 k 等于 min_mn，则将 panel_size 设置为 min_mn - jcol
if ( k == min_mn ) panel_size = min_mn - jcol;
// 更新 panel_histo 中 panel_size 的计数
panel_histo[panel_size]++;



/* symbolic factor on a panel of columns */
zpanel_dfs(m, panel_size, jcol, A, perm_r, &nseg1,
      dense, panel_lsub, segrep, repfnz, xprune,
      marker, parent, xplore, Glu);



// 对列面板进行符号因子分解
zpanel_dfs(m, panel_size, jcol, A, perm_r, &nseg1,
      dense, panel_lsub, segrep, repfnz, xprune,
      marker, parent, xplore, Glu);



/* numeric sup-panel updates in topological order */
zpanel_bmod(m, panel_size, jcol, nseg1, dense,
        tempv, segrep, repfnz, Glu, stat);



// 按拓扑顺序更新数值子面板
zpanel_bmod(m, panel_size, jcol, nseg1, dense,
        tempv, segrep, repfnz, Glu, stat);



/* Sparse LU within the panel, and below panel diagonal */
    for ( jj = jcol; jj < jcol + panel_size; jj++) {
 k = (jj - jcol) * m; /* column index for w-wide arrays */

nseg = nseg1;    /* Begin after all the panel segments */

    if ((*info = zcolumn_dfs(m, jj, perm_r, &nseg, &panel_lsub[k],
            segrep, &repfnz[k], xprune, marker,
            parent, xplore, Glu)) != 0) return;

      /* Numeric updates */
    if ((*info = zcolumn_bmod(jj, (nseg - nseg1), &dense[k],
             tempv, &segrep[nseg1], &repfnz[k],
             jcol, Glu, stat)) != 0) return;

    /* Copy the U-segments to ucol[*] */
if ((*info = zcopy_to_ucol(jj, nseg, segrep, &repfnz[k],
              perm_r, &dense[k], Glu)) != 0)
    return;

    if ( (*info = zpivotL(jj, diag_pivot_thresh, &usepr, perm_r,
              iperm_r, iperm_c, &pivrow, Glu, stat)) )
    if ( iinfo == 0 ) iinfo = *info;

/* Prune columns (0:jj-1) using column jj */
    zpruneL(jj, perm_r, pivrow, nseg, segrep,
                &repfnz[k], xprune, Glu);

/* Reset repfnz[] for this column */
    resetrep_col (nseg, segrep, &repfnz[k]);

#if ( DEBUGlevel>=2 )
zprint_lu_col("[2]: ", jj, pivrow, xprune, Glu);
#endif

    }

       jcol += panel_size;    /* Move to the next panel */



// 针对面板内和面板对角线以下进行稀疏 LU 分解
    for ( jj = jcol; jj < jcol + panel_size; jj++) {
 k = (jj - jcol) * m; /* 计算 w 宽数组的列索引 */

nseg = nseg1;    /* 从所有面板段之后开始 */

    // 执行列 jj 的深度优先搜索符号因子分解
    if ((*info = zcolumn_dfs(m, jj, perm_r, &nseg, &panel_lsub[k],
            segrep, &repfnz[k], xprune, marker,
            parent, xplore, Glu)) != 0) return;

      /* 数值更新 */
    // 执行列 jj 的数值更新
    if ((*info = zcolumn_bmod(jj, (nseg - nseg1), &dense[k],
             tempv, &segrep[nseg1], &repfnz[k],
             jcol, Glu, stat)) != 0) return;

    /* 将 U 段复制到 ucol[*] */
    // 执行将 U 段复制到 ucol 数组
if ((*info = zcopy_to_ucol(jj, nseg, segrep, &repfnz[k],
              perm_r, &dense[k], Glu)) != 0)
    return;

    // 执行列主元选取
    if ( (*info = zpivotL(jj, diag_pivot_thresh, &usepr, perm_r,
              iperm_r, iperm_c, &pivrow, Glu, stat)) )
    if ( iinfo == 0 ) iinfo = *info;

/* 使用列 jj 来修剪列 (0:jj-1) */
    // 使用列 jj 来修剪列 (0:jj-1)
    zpruneL(jj, perm_r, pivrow, nseg, segrep,
                &repfnz[k], xprune, Glu);

/* 重置 repfnz[] 数组的值 */
    // 重置此列的 repfnz[] 数组的值
    resetrep_col (nseg, segrep, &repfnz[k]);

#if ( DEBUGlevel>=2 )
zprint_lu_col("[2]: ", jj, pivrow, xprune, Glu);
#endif

    }

       jcol += panel_size;    /* 移动到下一个面板 */



} /* else */



// else 语句的结束
} /* else */



} /* for */



// for 循环的结束



*info = iinfo;



// 将 *info 设置为 iinfo 的值
*info = iinfo;



/* Complete perm_r[] for rank-deficient or tall-skinny matrices */
/* k is the rank of U
   pivots have been completed for rows < k
   Now fill in the pivots for rows k to m */
k = iinfo == 0 ? n : (int)iinfo - 1;



// 对于秩缺陷或高瘦矩阵，完成 perm_r[] 数组
// k 是 U 的秩
// 对于行小于 k 的已完成主元
// 现在填充行 k 到 m 的主元
k = iinfo == 0 ? n : (int)iinfo - 1;






// 最后一个空行，用于标记代码块的结束
    if (m > k) {
        /* 如果 m 大于 k，则进入循环 */
        /* 如果 k == m，则所有行排列已完成，可以快速跳出向量的其余部分 */
        for (i = 0; i < m && k < m; ++i) {
            /* 遍历直到 m 或者 k 达到上限 */
            if (perm_r[i] == EMPTY) {
                /* 如果 perm_r[i] 为 EMPTY，则将其设为 k 并增加 k 的值 */
                perm_r[i] = k;
                ++k;
            }
        }
    }
    
    countnz(min_mn, xprune, &nnzL, &nnzU, Glu);
    /* 计算非零元素的个数，更新 nnzL 和 nnzU */
    fixupL(min_mn, perm_r, Glu);
    /* 修正 L 矩阵 */
    
    zLUWorkFree(iwork, zwork, Glu); /* 释放工作空间并压缩存储 */
    SUPERLU_FREE(xplore);
    SUPERLU_FREE(xprune);
    
    if ( fact == SamePattern_SameRowPerm ) {
        /* 如果 fact 是 SamePattern_SameRowPerm，则进行如下操作 */
        /* L 和 U 的结构可能由于不同的枢轴选取而改变，即使存储是可用的。
           也可能存在内存扩展，所以数组位置可能已经改变 */
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
        /* 否则，重新创建超节点矩阵 L 和压缩列矩阵 U */
        zCreate_SuperNode_Matrix(L, A->nrow, min_mn, nnzL, 
          (doublecomplex *) Glu->lusup, Glu->xlusup, 
          Glu->lsub, Glu->xlsub, Glu->supno, Glu->xsup,
          SLU_SC, SLU_Z, SLU_TRLU);
        zCreate_CompCol_Matrix(U, min_mn, min_mn, nnzU, 
          (doublecomplex *) Glu->ucol, Glu->usub, Glu->xusub,
          SLU_NC, SLU_Z, SLU_TRU);
    }
    
    ops[FACT] += ops[TRSV] + ops[GEMV];
    /* 更新运算数 */
    stat->expansions = --(Glu->num_expansions);
    /* 减少扩展次数 */
    
    if ( iperm_r_allocated ) SUPERLU_FREE (iperm_r);
    SUPERLU_FREE (iperm_c);
    SUPERLU_FREE (relax_end);
}


注释：


# 这是一个代码块的结束标记，闭合了前面的代码块或函数定义
```