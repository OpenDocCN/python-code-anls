# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\cgstrf.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file cgstrf.c
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


#include "slu_cdefs.h"

void
cgstrf (superlu_options_t *options, SuperMatrix *A,
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
    singlecomplex    *cwork;
    int          *segrep, *repfnz, *parent;
    int          *panel_lsub; /* dense[]/panel_lsub[] pair forms a w-wide SPA */
    int_t     *xprune, *xplore;
    int          *marker;
    singlecomplex    *dense, *tempv;
    int       *relax_end;
    singlecomplex    *a;
    int_t     *asub, *xa_begin, *xa_end;
    int_t     *xlsub, *xlusup, *xusub;
    int       *xsup, *supno;
    int_t     nzlumax;
    float fill_ratio = sp_ienv(6);  /* estimated fill ratio */

    /* Local scalars */
    fact_t    fact = options->Fact;  // 获取 LU 分解选项中的因子类型
    double    diag_pivot_thresh = options->DiagPivotThresh;  // 获取对角元素主元阈值
    int       pivrow;   /* pivotal row number in the original matrix A */  // 原始矩阵 A 中的主元行号
    int       nseg1;    /* no of segments in U-column above panel row jcol */  // 在列 jcol 上面的 U 列中的段数
    int       nseg;    /* no of segments in each U-column */  // 每个 U 列中的段数
    register int jcol, jj;
    register int kcol;    /* end column of a relaxed snode */  // 放松的超节点的结束列
    register int icol;
    int_t     i, k, iinfo, new_next, nextlu, nextu;
    int       m, n, min_mn, jsupno, fsupc;
    int       w_def;    /* upper bound on panel width */  // 面板宽度的上限
    int       usepr, iperm_r_allocated = 0;
    int_t     nnzL, nnzU;
    int       *panel_histo = stat->panel_histo;  // 面板直方图统计数组
    flops_t   *ops = stat->ops;  // 操作计数统计
    # 初始化变量 iinfo 为 0
    iinfo    = 0;
    # 从输入矩阵 A 中获取行数 m 和列数 n
    m        = A->nrow;
    n        = A->ncol;
    # 计算 m 和 n 中的较小值，存储在 min_mn 中
    min_mn   = SUPERLU_MIN(m, n);
    # 从矩阵 A 中获取存储结构 Astore
    Astore   = A->Store;
    # 获取矩阵 A 中非零元素的值存储在 a 中
    a        = Astore->nzval;
    # 获取矩阵 A 中非零元素的行索引存储在 asub 中
    asub     = Astore->rowind;
    # 获取矩阵 A 中每列的起始位置存储在 xa_begin 中
    xa_begin = Astore->colbeg;
    # 获取矩阵 A 中每列的结束位置存储在 xa_end 中
    xa_end   = Astore->colend;

    /* 分配用于因子化例程的存储空间 */
    # 调用 cLUMemInit 函数初始化存储空间，并将结果存储在 info 指向的位置
    *info = cLUMemInit(fact, work, lwork, m, n, Astore->nnz,
                       panel_size, fill_ratio, L, U, Glu, &iwork, &cwork);
    # 如果返回的 *info 非零，则直接返回
    if ( *info ) return;
    
    # 从 Glu 结构中获取超节点的起始索引
    xsup    = Glu->xsup;
    # 从 Glu 结构中获取每个超节点的编号
    supno   = Glu->supno;
    # 从 Glu 结构中获取每个子图的起始索引
    xlsub   = Glu->xlsub;
    # 从 Glu 结构中获取每个非主对角线上的超节点的起始索引
    xlusup  = Glu->xlusup;
    # 从 Glu 结构中获取每个超节点的结束索引
    xusub   = Glu->xusub;
    
    # 设置整数工作区数组的初始值
    SetIWork(m, n, panel_size, iwork, &segrep, &parent, &xplore,
             &repfnz, &panel_lsub, &xprune, &marker);
    # 设置复数工作区数组的初始值
    cSetRWork(m, panel_size, cwork, &dense, &tempv);
    
    # 根据 factorization 类型确定是否使用相同模式和行排列
    usepr = (fact == SamePattern_SameRowPerm);
    # 如果使用相同模式和行排列，则计算行排列的逆 iperm_r
    if ( usepr ) {
        iperm_r = (int *) int32Malloc(m);
        for (k = 0; k < m; ++k) iperm_r[perm_r[k]] = k;
        iperm_r_allocated = 1;
    }
    # 计算列排列的逆 iperm_c
    iperm_c = (int *) int32Malloc(n);
    for (k = 0; k < n; ++k) iperm_c[perm_c[k]] = k;

    /* 标识放松的超节点 */
    # 分配放松结束标记数组的存储空间
    relax_end = (int *) intMalloc(n);
    # 根据对称模式选项确定调用哪个放松超节点函数
    if ( options->SymmetricMode == YES ) {
        heap_relax_snode(n, etree, relax, marker, relax_end); 
    } else {
        relax_snode(n, etree, relax, marker, relax_end); 
    }
    
    # 将 perm_r 数组的所有元素置为 EMPTY
    ifill (perm_r, m, EMPTY);
    # 将 marker 数组的所有元素置为 EMPTY
    ifill (marker, m * NO_MARKER, EMPTY);
    # 第一个超节点的编号设为 -1
    supno[0] = -1;
    # 第一个超节点的起始索引设为 0
    xsup[0]  = xlsub[0] = xusub[0] = xlusup[0] = 0;
    # 设置默认的面板大小为 panel_size
    w_def    = panel_size;

    /* 
     * 逐个处理一个“面板”。面板可以是以下之一：
     *       (a) 树的底部的放松超节点，或者
     *       (b) 用户定义的 panel_size 连续列
     */
    for (jcol = 0; jcol < min_mn; ) {
    if ( relax_end[jcol] != EMPTY ) { /* 如果 relax_end[jcol] 不是空值，则进入松弛超节点的处理 */
           kcol = relax_end[jcol];      /* kcol 是松弛超节点的结束列 */
        panel_histo[kcol-jcol+1]++;    /* 更新面板直方图中对应的列范围 */

        /* --------------------------------------
         * Factorize the relaxed supernode(jcol:kcol) 
         * 对松弛超节点(jcol:kcol)进行因式分解
         * -------------------------------------- */
        /* Determine the union of the row structure of the snode */
        /* 确定松弛超节点的行结构的并集 */
        if ( (*info = csnode_dfs(jcol, kcol, asub, xa_begin, xa_end,
                    xprune, marker, Glu)) != 0 )
        return;

            nextu    = xusub[jcol];    /* 下一个 U 列的起始位置 */
        nextlu   = xlusup[jcol];      /* 下一个 L 列的起始位置 */
        jsupno   = supno[jcol];       /* 当前列的超节点编号 */
        fsupc    = xsup[jsupno];      /* 当前超节点的第一列 */
        new_next = nextlu + (xlsub[fsupc+1]-xlsub[fsupc])*(kcol-jcol+1); /* 新的 L 列的结束位置 */
        nzlumax = Glu->nzlumax;       /* LU 结构中可用的最大内存 */
        while ( new_next > nzlumax ) { /* 如果新的 L 列超过了可用的最大内存 */
        if ( (*info = cLUMemXpand(jcol, nextlu, LUSUP, &nzlumax, Glu)) ) /* 扩展 LU 内存 */
            return;
        }
    
        for (icol = jcol; icol<= kcol; icol++) { /* 遍历松弛超节点的列范围 */
        xusub[icol+1] = nextu;        /* 更新下一个 U 列的起始位置 */
        
            /* Scatter into SPA dense[*] */
            /* 将数据散布到稠密数组 dense[*] */
            for (k = xa_begin[icol]; k < xa_end[icol]; k++)
                dense[asub[k]] = a[k];

               /* Numeric update within the snode */
            /* 在超节点内进行数值更新 */
            csnode_bmod(icol, jsupno, fsupc, dense, tempv, Glu, stat);

        if ( (*info = cpivotL(icol, diag_pivot_thresh, &usepr, perm_r,
                      iperm_r, iperm_c, &pivrow, Glu, stat)) )
            if ( iinfo == 0 ) iinfo = *info;
        }
#if ( DEBUGlevel>=2 )
        cprint_lu_col("[1]: ", icol, pivrow, xprune, Glu);
#endif


// 如果调试级别高于等于2，则打印"[1]: "后面的信息，包括icol、pivrow、xprune和Glu的值



        }


// 结束if条件块，暂无其他代码执行



        jcol = icol;


// 将变量icol的值赋给变量jcol，用于后续计算



    } else { /* Work on one panel of panel_size columns */


// 否则，处理一个大小为panel_size的列面板



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


// 调整panel_size，以确保一个面板不会与下一个松弛的snode重叠
// 初始将panel_size设置为w_def
// 遍历列jcol + 1到jcol+panel_size和min_mn的最小值，找到第一个relax_end不为空的位置k
// 如果找到，则计算panel_size为k-jcol，然后跳出循环
// 如果未找到，且k等于min_mn，则panel_size为min_mn-jcol
// 将当前panel_size的计数增加到panel_histo数组中



        /* symbolic factor on a panel of columns */
        cpanel_dfs(m, panel_size, jcol, A, perm_r, &nseg1,
              dense, panel_lsub, segrep, repfnz, xprune,
              marker, parent, xplore, Glu);


// 对一个列面板进行符号因子分解
// 使用参数m、panel_size、jcol、A、perm_r等调用函数cpanel_dfs进行操作



        /* numeric sup-panel updates in topological order */
        cpanel_bmod(m, panel_size, jcol, nseg1, dense,
                tempv, segrep, repfnz, Glu, stat);


// 按拓扑顺序更新数值超级面板
// 使用参数m、panel_size、jcol、nseg1等调用函数cpanel_bmod进行更新



        /* Sparse LU within the panel, and below panel diagonal */
            for ( jj = jcol; jj < jcol + panel_size; jj++) {
         k = (jj - jcol) * m; /* column index for w-wide arrays */


// 在列面板内及其对角线以下执行稀疏LU分解
// 循环处理从jcol到jcol+panel_size的列jj
// 计算在w宽度数组中的列索引k



        nseg = nseg1;    /* Begin after all the panel segments */


// 设置nseg为nseg1，表示从所有面板段之后开始



            if ((*info = ccolumn_dfs(m, jj, perm_r, &nseg, &panel_lsub[k],
                    segrep, &repfnz[k], xprune, marker,
                    parent, xplore, Glu)) != 0) return;


// 调用ccolumn_dfs函数进行列的深度优先搜索
// 更新info的值，如果返回非零则直接返回



            if ((*info = ccolumn_bmod(jj, (nseg - nseg1), &dense[k],
                     tempv, &segrep[nseg1], &repfnz[k],
                     jcol, Glu, stat)) != 0) return;


// 调用ccolumn_bmod函数进行列的数值修改
// 更新info的值，如果返回非零则直接返回



        /* Copy the U-segments to ucol[*] */
        if ((*info = ccopy_to_ucol(jj, nseg, segrep, &repfnz[k],
                      perm_r, &dense[k], Glu)) != 0)
            return;


// 将U段复制到ucol数组
// 更新info的值，如果返回非零则直接返回



            if ( (*info = cpivotL(jj, diag_pivot_thresh, &usepr, perm_r,
                      iperm_r, iperm_c, &pivrow, Glu, stat)) )
            if ( iinfo == 0 ) iinfo = *info;


// 调用cpivotL函数进行列主元选取
// 更新info的值，如果返回非零则直接返回
// 如果iinfo为0，则将info的值赋给iinfo



        /* Prune columns (0:jj-1) using column jj */
            cpruneL(jj, perm_r, pivrow, nseg, segrep,
                        &repfnz[k], xprune, Glu);


// 使用列jj对列0到jj-1进行剪枝
// 调用cpruneL函数进行操作



        /* Reset repfnz[] for this column */
            resetrep_col (nseg, segrep, &repfnz[k]);


// 重置该列的repfnz数组
// 调用resetrep_col函数进行操作



#if ( DEBUGlevel>=2 )
        cprint_lu_col("[2]: ", jj, pivrow, xprune, Glu);
#endif


// 如果调试级别高于等于2，则打印"[2]: "后面的信息，包括jj、pivrow、xprune和Glu的值



        }

           jcol += panel_size;    /* Move to the next panel */


// 循环结束后，将jcol增加panel_size，移动到下一个面板



    } /* else */


// 结束else块



    } /* for */


// 结束for循环块



    *info = iinfo;


// 将iinfo的值赋给info



    /* Complete perm_r[] for rank-deficient or tall-skinny matrices */
    /* k is the rank of U
       pivots have been completed for rows < k
       Now fill in the pivots for rows k to m */
    k = iinfo == 0 ? n : (int)iinfo - 1;


// 对于秩不足或高瘦矩阵，完成perm_r[]数组
// k表示U的秩
// 对于行号小于k的行，主元已经完成
// 现在填充行号从k到m的主元
    if (m > k) {
        /* 如果 m 大于 k，则进入循环 */
        /* 如果 k 等于 m，则所有行排列已完成，可以快速结束后续向量的查找 */
        for (i = 0; i < m && k < m; ++i) {
            /* 遍历直到达到 m 或 k 等于 m */
            if (perm_r[i] == EMPTY) {
                /* 如果 perm_r[i] 等于 EMPTY */
                perm_r[i] = k;
                /* 将 perm_r[i] 设置为 k */
                ++k;
                /* k 自增 */
            }

        }
    }
    
    countnz(min_mn, xprune, &nnzL, &nnzU, Glu);
    /* 计算非零元素的个数，并存储在 nnzL 和 nnzU 中 */
    fixupL(min_mn, perm_r, Glu);
    /* 修正 L 的数据结构 */

    cLUWorkFree(iwork, cwork, Glu); /* 释放工作空间并压缩存储空间 */
    SUPERLU_FREE(xplore);
    SUPERLU_FREE(xprune);

    if ( fact == SamePattern_SameRowPerm ) {
        /* 如果 fact 等于 SamePattern_SameRowPerm */
        /* 由于可能有不同的枢轴选取，L 和 U 的结构可能已经改变，尽管存储是可用的 */
        /* 可能还会有内存扩展，因此数组位置可能已更改 */
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
        /* 否则，创建超节点矩阵 L */
        cCreate_SuperNode_Matrix(L, A->nrow, min_mn, nnzL, 
          (singlecomplex *) Glu->lusup, Glu->xlusup, 
          Glu->lsub, Glu->xlsub, Glu->supno, Glu->xsup,
          SLU_SC, SLU_C, SLU_TRLU);
        /* 创建压缩列矩阵 U */
        cCreate_CompCol_Matrix(U, min_mn, min_mn, nnzU, 
          (singlecomplex *) Glu->ucol, Glu->usub, Glu->xusub,
          SLU_NC, SLU_C, SLU_TRU);
    }
    
    ops[FACT] += ops[TRSV] + ops[GEMV];    
    /* 更新操作数统计信息 */

    stat->expansions = --(Glu->num_expansions);
    /* 更新扩展次数统计信息 */

    if ( iperm_r_allocated ) SUPERLU_FREE (iperm_r);
    /* 如果 iperm_r_allocated 为真，则释放 iperm_r */
    SUPERLU_FREE (iperm_c);
    /* 释放 iperm_c */
    SUPERLU_FREE (relax_end);
    /* 释放 relax_end */
}
```