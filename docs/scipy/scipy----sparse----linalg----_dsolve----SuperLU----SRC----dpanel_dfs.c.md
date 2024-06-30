# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dpanel_dfs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file dpanel_dfs.c
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


#include "slu_ddefs.h"

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
dpanel_dfs (
       const int  m,           /* in - number of rows in the matrix */
       const int  w,           /* in - width of the panel */
       const int  jcol,        /* in - starting column index of the panel */
       SuperMatrix *A,         /* in - original matrix */
       int        *perm_r,     /* in - row permutation */
       int        *nseg,       /* out - number of supernodes */
       double     *dense,      /* out - dense array */
       int        *panel_lsub, /* out - panel's lsub array */
       int        *segrep,     /* out - supernodal representatives */
       int        *repfnz,     /* out - supernodal entry locations */
       int_t      *xprune,     /* out - index to L-structure */
       int        *marker,     /* out - marker array */
       int        *parent,     /* working array */
       int_t      *xplore,     /* working array */
       GlobalLU_t *Glu         /* modified - global LU structure */
       )
{

    NCPformat *Astore;  // 存储矩阵 A 的非零元素的格式
    double    *a;        // 指向矩阵 A 的数据数组
    int_t     *asub;            /* 指向矩阵A的非零元素行索引数组 */
    int_t     *xa_begin, *xa_end, k;  /* xa_begin和xa_end分别指向A矩阵每列非零元素在a数组中的起始和结束位置，k为临时变量 */
    int       krow, kmark, kperm;  /* krow, kmark, kperm为临时变量 */
    int          krep, chperm, chmark, chrep, oldrep, kchild, myfnz, kpar;  /* 多个临时变量 */
    int       jj;       /* 在面板中遍历每列的索引 */
    int       *marker1;       /* 如果marker1[jj] >= jcol，表示顶点jj已被前面的列访问过 */
    int       *repfnz_col; /* 指向面板中每列的起始位置 */
    double    *dense_col;  /* 指向面板中每列的起始位置 */
    int_t     nextl_col;   /* 面板中panel_lsub[*,jj]数组的下一个可用位置 */
    int       *xsup, *supno;   /* xsup指向Glu中的xsup数组，supno指向Glu中的supno数组 */
    int_t     *lsub, *xlsub;   /* lsub指向Glu中的lsub数组，xlsub指向Glu中的xlsub数组 */
    int_t      xdfs, maxdfs;   /* xdfs和maxdfs为深度优先搜索相关的临时变量 */

    /* 初始化指针 */
    Astore     = A->Store;    /* Astore指向矩阵A的存储结构 */
    a          = Astore->nzval;   /* a指向矩阵A的非零元素数组 */
    asub       = Astore->rowind;  /* asub指向矩阵A的非零元素行索引数组 */
    xa_begin   = Astore->colbeg;  /* xa_begin指向矩阵A的列开始位置数组 */
    xa_end     = Astore->colend;  /* xa_end指向矩阵A的列结束位置数组 */
    marker1    = marker + m;      /* marker1指向marker数组的第m个元素 */
    repfnz_col = repfnz;          /* repfnz_col指向repfnz数组 */
    dense_col  = dense;           /* dense_col指向dense数组 */
    *nseg      = 0;               /* 初始化nseg为0 */
    xsup       = Glu->xsup;       /* xsup指向Glu中的xsup数组 */
    supno      = Glu->supno;      /* supno指向Glu中的supno数组 */
    lsub       = Glu->lsub;       /* lsub指向Glu中的lsub数组 */
    xlsub      = Glu->xlsub;      /* xlsub指向Glu中的xlsub数组 */

    /* 对于面板中的每一列 */
    for (jj = jcol; jj < jcol + w; jj++) {
        nextl_col = (jj - jcol) * m;   /* 计算panel_lsub[*,jj]中下一个可用的位置 */
#ifdef CHK_DFS
    // 如果定义了 CHK_DFS 宏，则输出当前列的信息
    printf("\npanel col %d: ", jj);
#endif

    /* 对于 A[*,jj] 中的每个非零元素，执行深度优先搜索 */
    for (k = xa_begin[jj]; k < xa_end[jj]; k++) {
        krow = asub[k];
        // 将 A[*,jj] 中的非零元素存入 dense_col 中对应的行
        dense_col[krow] = a[k];
        kmark = marker[krow];        
        if ( kmark == jj ) 
            continue;     /* krow 已被访问过，继续下一个非零元素 */

        /* 对于 jj 的未标记邻居 krow
         * 如果 krow 在 L 中：将其放入 L[*,jj] 的结构中
         */
        marker[krow] = jj;
        kperm = perm_r[krow];
        
        if ( kperm == EMPTY ) {
            // krow 在 A 中被索引，将其加入 panel_lsub
            panel_lsub[nextl_col++] = krow;
        }
        /* 
         * 如果 krow 在 U 中：
         * 如果其超节点代表 krep 已被探索过，则更新 repfnz[*]
         */
        else {
            krep = xsup[supno[kperm]+1] - 1;
            myfnz = repfnz_col[krep];
            
#ifdef CHK_DFS
            // 如果定义了 CHK_DFS 宏，则输出调试信息
            printf("krep %d, myfnz %d, perm_r[%d] %d\n", krep, myfnz, krow, kperm);
#endif
            if ( myfnz != EMPTY ) {    /* 代表已被访问过 */
                if ( myfnz > kperm ) repfnz_col[krep] = kperm;
                /* continue; */
            }
            else {
                /* 否则，从 krep 开始执行深度优先搜索 */
                oldrep = EMPTY;
                parent[krep] = oldrep;
                repfnz_col[krep] = kperm;
                xdfs = xlsub[krep];
                maxdfs = xprune[krep];
                
#ifdef CHK_DFS 
                // 如果定义了 CHK_DFS 宏，则输出调试信息
                printf("  xdfs %d, maxdfs %d: ", xdfs, maxdfs);
                for (i = xdfs; i < maxdfs; i++) printf(" %d", lsub[i]);
                printf("\n");
#endif
                do {
                    /* 
                     * 对于 krep 的每个未标记 kchild 
                     */
                    while ( xdfs < maxdfs ) {
                        kchild = lsub[xdfs];
                        xdfs++;
                        chmark = marker[kchild];
                        
                        if ( chmark != jj ) { /* 尚未到达 */
                            marker[kchild] = jj;
                            chperm = perm_r[kchild];
                            
                            /* 如果 kchild 在 L 中：将其放入 L[*,j] */
                            if ( chperm == EMPTY ) {
                                panel_lsub[nextl_col++] = kchild;
                            } 
                            /* 如果 kchild 在 U 中： 
                             *   chrep 是其超节点代表。如果其代表已被探索，则更新其 repfnz[*]
                             */
                            else {
                                chrep = xsup[supno[chperm]+1] - 1;
                                myfnz = repfnz_col[chrep];
#ifdef CHK_DFS
                                // 如果定义了 CHK_DFS 宏，则输出调试信息
                                printf("chrep %d, myfnz %d, perm_r[%d] %d\n", chrep, myfnz, kchild, chperm);
#endif
                            }
                        }
                    }
                } while ( xdfs < maxdfs );
            }
        }
    }
#endif
                    if ( myfnz != EMPTY ) { /* 如果之前访问过 */
                    if ( myfnz > chperm )
                        repfnz_col[chrep] = chperm;
                    }
                    else {
                    /* 在 kchild 的 snode-rep 处继续深度优先搜索 */
                    xplore[krep] = xdfs;    
                    oldrep = krep;
                    krep = chrep; /* 深入到 G(L) 中更深的层次 */
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
            
            /* krow 没有更多未探索的邻居：
             *    如果这段段落第一次见到，将 snode-rep krep 放入后序 DFS 中。（注意 "repfnz[krep]" 可能会后续更改。）
             *    回溯 dfs 到其父节点。
             */
            if ( marker1[krep] < jcol ) {
                segrep[*nseg] = krep;
                ++(*nseg);
                marker1[krep] = jj;
            }
            
            kpar = parent[krep]; /* 出栈，模拟递归 */
            if ( kpar == EMPTY ) break; /* dfs 完成 */
            krep = kpar;
            xdfs = xplore[krep];
            maxdfs = xprune[krep];
            
#ifdef CHK_DFS 
            printf("  pop stack: krep %d,xdfs %d,maxdfs %d: ", krep,xdfs,maxdfs);
            for (i = xdfs; i < maxdfs; i++) printf(" %d", lsub[i]);
            printf("\n");
#endif
            } while ( kpar != EMPTY ); /* do-while - 直到栈空为止 */
            
        } /* else */
        
        } /* else */
        
    } /* for each nonz in A[*,jj] */
    
    repfnz_col += m;    /* 移动到下一列 */
        dense_col += m;
    
    } /* for jj ... */
    
}
```