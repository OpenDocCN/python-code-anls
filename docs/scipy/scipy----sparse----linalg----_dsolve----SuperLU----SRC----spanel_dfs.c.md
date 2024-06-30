# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\spanel_dfs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file spanel_dfs.c
 * \brief Peforms a symbolic factorization on a panel of symbols
 *
 * <pre>
 * -- SuperLU routine (version 2.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November 15, 1997
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

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 *   Performs a symbolic factorization on a panel of columns [jcol, jcol+w).
 *
 *   A supernode representative is the last column of a supernode.
 *   The nonzeros in U[*,j] are segments that end at supernodal
 *   representatives.
 *
 *   The routine returns one list of the supernodal representatives
 *   in topological order of the dfs that generates them. This list is
 *   a superset of the topological order of each individual column within
 *   the panel. 
 *   The location of the first nonzero in each supernodal segment
 *   (supernodal entry location) is also returned. Each column has a 
 *   separate list for this purpose.
 *
 *   Two marker arrays are used for dfs:
 *     marker[i] == jj, if i was visited during dfs of current column jj;
 *     marker1[i] >= jcol, if i was visited by earlier columns in this panel;
 *
 *   marker: A-row --> A-row/col (0/1)
 *   repfnz: SuperA-col --> PA-row
 *   parent: SuperA-col --> SuperA-col
 *   xplore: SuperA-col --> index to L-structure
 * </pre>
 */

void
spanel_dfs (
       const int  m,           /* in - number of rows in the matrix */
       const int  w,           /* in - width of the panel */
       const int  jcol,        /* in - starting column index of the panel */
       SuperMatrix *A,         /* in - original matrix */
       int        *perm_r,     /* in - row permutation */
       int        *nseg,       /* out - number of segments */
       float      *dense,      /* out - dense array */
       int        *panel_lsub, /* out - panel's nonzero rows */
       int        *segrep,     /* out - supernodal representatives */
       int        *repfnz,     /* out - supernodal first nonzeros */
       int_t      *xprune,     /* out - prune array */
       int        *marker,     /* out - marker array */
       int        *parent,     /* working array */
       int_t      *xplore,     /* working array */
       GlobalLU_t *Glu         /* modified - global LU data structure */
       )
{

    NCPformat *Astore;
    float    *a;

    // 获取原始矩阵 A 的存储格式
    Astore = A->Store;
    // 获取矩阵 A 中的数据数组
    a = Astore->nzval;
    # 声明整型指针变量 `asub`，用于存储稀疏矩阵的行索引
    int_t     *asub;
    # 声明整型指针变量 `xa_begin`、`xa_end`，以及整型变量 `k`
    int_t     *xa_begin, *xa_end, k;
    # 声明整型变量 `krow`, `kmark`, `kperm`
    int       krow, kmark, kperm;
    # 声明整型变量 `krep`, `chperm`, `chmark`, `chrep`, `oldrep`, `kchild`, `myfnz`, `kpar`
    int       krep, chperm, chmark, chrep, oldrep, kchild, myfnz, kpar;
    # 声明整型变量 `jj`，用于迭代遍历面板中的每一列
    int       jj;       /* index through each column in the panel */
    # 声明整型指针变量 `marker1`，用于标记面板中每列是否已被访问
    int       *marker1;       /* marker1[jj] >= jcol if vertex jj was visited 
                  by a previous column within this panel.   */
    # 声明整型指针变量 `repfnz_col`，指向面板中每列的起始位置
    int       *repfnz_col; /* start of each column in the panel */
    # 声明浮点型指针变量 `dense_col`，指向面板中每列的起始位置
    float    *dense_col;  /* start of each column in the panel */
    # 声明整型变量 `nextl_col`，表示面板中当前列的下一个可用位置
    int_t     nextl_col;   /* next available position in panel_lsub[*,jj] */
    # 声明整型指针变量 `xsup`、`supno`
    int       *xsup, *supno;
    # 声明整型指针变量 `lsub`、`xlsub`
    int_t     *lsub, *xlsub;
    int_t      xdfs, maxdfs;

    /* Initialize pointers */
    # 将稀疏矩阵 `A` 的存储结构中的相关指针赋值给对应变量
    Astore     = A->Store;
    a          = Astore->nzval;
    asub       = Astore->rowind;
    xa_begin   = Astore->colbeg;
    xa_end     = Astore->colend;
    # 将 marker 数组的一部分赋值给 marker1，起始位置从 `marker + m` 开始
    marker1    = marker + m;
    # 将 repfnz 数组赋值给 repfnz_col
    repfnz_col = repfnz;
    # 将 dense 数组赋值给 dense_col
    dense_col  = dense;
    # 将 nseg 变量初始化为 0
    *nseg      = 0;
    # 将 Glu 结构体中的相关指针赋值给对应变量
    xsup       = Glu->xsup;
    supno      = Glu->supno;
    lsub       = Glu->lsub;
    xlsub      = Glu->xlsub;

    /* For each column in the panel */
    # 对于面板中的每一列，从 `jcol` 开始到 `jcol + w` 结束
    for (jj = jcol; jj < jcol + w; jj++) {
    # 计算当前列在 panel_lsub 中的下一个可用位置
    nextl_col = (jj - jcol) * m;
#ifdef CHK_DFS
    printf("\npanel col %d: ", jj);
#endif

#ifdef CHK_DFS 指令用于条件编译，当定义了 CHK_DFS 宏时，编译器将包含此段代码；否则，将被忽略。


    /* For each nonz in A[*,jj] do dfs */
    for (k = xa_begin[jj]; k < xa_end[jj]; k++) {

对 A 矩阵中列 jj 的每个非零元素执行深度优先搜索（DFS）。


        krow = asub[k];
        dense_col[krow] = a[k];
        kmark = marker[krow];        
        if ( kmark == jj ) 
            continue;     /* krow visited before, go to the next nonzero */

获取当前非零元素的行索引 krow，并将其值存储在 dense_col 中。检查 krow 是否已经被访问过（通过 marker 数组标记），若已访问则跳过当前元素。


        /* For each unmarked nbr krow of jj
         * krow is in L: place it in structure of L[*,jj]
         */
        marker[krow] = jj;
        kperm = perm_r[krow];

标记 krow 为已访问（marker[krow] = jj），并获取 krow 的行排列索引 kperm。


        if ( kperm == EMPTY ) {
            panel_lsub[nextl_col++] = krow; /* krow is indexed into A */
        }

若 krow 在 A 矩阵的 L 部分，则将其索引存入 panel_lsub 数组。


        else {
            krep = xsup[supno[kperm]+1] - 1;
            myfnz = repfnz_col[krep];
#ifdef CHK_DFS
            printf("krep %d, myfnz %d, perm_r[%d] %d\n", krep, myfnz, krow, kperm);
#endif

否则，获取 kperm 的超节点代表 krep，并查看 repfnz_col 中的 myfnz 值。如果定义了 CHK_DFS 宏，则打印调试信息。


            if ( myfnz != EMPTY ) {    /* Representative visited before */
                if ( myfnz > kperm ) repfnz_col[krep] = kperm;
                /* continue; */
            }
            else {
                /* Otherwise, perform dfs starting at krep */
                oldrep = EMPTY;
                parent[krep] = oldrep;
                repfnz_col[krep] = kperm;
                xdfs = xlsub[krep];
                maxdfs = xprune[krep];
#ifdef CHK_DFS 
                printf("  xdfs %d, maxdfs %d: ", xdfs, maxdfs);
                for (i = xdfs; i < maxdfs; i++) printf(" %d", lsub[i]);
                printf("\n");
#endif

若 krep 的代表未被访问过，则从 krep 开始执行深度优先搜索。设置 parent 数组中 krep 的父节点为 EMPTY，并更新 repfnz_col 的值。如果定义了 CHK_DFS 宏，则打印调试信息，显示 xdfs 和 maxdfs 的值以及 lsub 数组的内容。


                do {
                    /* 
                     * For each unmarked kchild of krep 
                     */
                    while ( xdfs < maxdfs ) {
                        kchild = lsub[xdfs];
                        xdfs++;
                        chmark = marker[kchild];
                        if ( chmark != jj ) { /* Not reached yet */
                            marker[kchild] = jj;
                            chperm = perm_r[kchild];

对于 krep 的每个未被访问的 kchild，标记其为已访问（marker[kchild] = jj），并获取其行排列索引 chperm。


                            /* Case kchild is in L: place it in L[*,j] */
                            if ( chperm == EMPTY ) {
                                panel_lsub[nextl_col++] = kchild;
                            } 

若 kchild 在 A 矩阵的 L 部分，则将其索引存入 panel_lsub 数组。


                            /* Case kchild is in U: 
                             *   chrep = its supernode-rep. If its rep has 
                             *   been explored, update its repfnz[*]
                             */
                            else {
                                chrep = xsup[supno[chperm]+1] - 1;
                                myfnz = repfnz_col[chrep];
#ifdef CHK_DFS
                                printf("chrep %d,myfnz %d,perm_r[%d] %d\n",chrep,myfnz,kchild,chperm);
#endif

否则，获取 chperm 的超节点代表 chrep，并查看 repfnz_col 中的 myfnz 值。如果定义了 CHK_DFS 宏，则打印调试信息。
#else
                    if ( myfnz != EMPTY ) { /* 如果已经访问过 */
                        if ( myfnz > chperm )
                            repfnz_col[chrep] = chperm;
                    }
                    else {
                        /* 在 kchild 的 snode-rep 处继续深度优先搜索 */
                        xplore[krep] = xdfs;    
                        oldrep = krep;
                        krep = chrep; /* 深入到 G(L) 中更深的位置 */
                        parent[krep] = oldrep;
                        repfnz_col[krep] = chperm;
                        xdfs = xlsub[krep];     
                        maxdfs = xprune[krep];
#ifdef CHK_DFS 
                        printf("  xdfs %d, maxdfs %d: ", xdfs, maxdfs);
                        for (i = xdfs; i < maxdfs; i++) printf(" %d", lsub[i]);    
                        printf("\n");
#endif
                    } /* else */
                  
                } /* else */
                  
            } /* if... */
                
        } /* while xdfs < maxdfs */
            
        /* krow 没有更多未探索的邻居:
         *    如果这段首次出现，则将 snode-rep krep 放在后序 DFS 中。注意
         *    "repfnz[krep]" 可能稍后会改变。
         *    回溯 dfs 到其父节点。
         */
        if ( marker1[krep] < jcol ) {
            segrep[*nseg] = krep;
            ++(*nseg);
            marker1[krep] = jj;
        }
            
        kpar = parent[krep]; /* 弹出堆栈，模拟递归 */
        if ( kpar == EMPTY ) break; /* dfs 完成 */
        krep = kpar;
        xdfs = xplore[krep];
        maxdfs = xprune[krep];
            
#ifdef CHK_DFS 
        printf("  pop stack: krep %d, xdfs %d, maxdfs %d: ", krep, xdfs, maxdfs);
        for (i = xdfs; i < maxdfs; i++) printf(" %d", lsub[i]);
        printf("\n");
#endif
    } while ( kpar != EMPTY ); /* do-while - 直到堆栈为空 */
            
} /* else */
        
} /* else */
        
} /* for each nonz in A[*,jj] */
    
repfnz_col += m;    /* 移动到下一列 */
dense_col += m;
    
} /* for jj ... */
```