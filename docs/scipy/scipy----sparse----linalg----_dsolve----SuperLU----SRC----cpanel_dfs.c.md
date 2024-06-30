# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\cpanel_dfs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file cpanel_dfs.c
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


#include "slu_cdefs.h"

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
cpanel_dfs (
       const int  m,           /* in - number of rows in the matrix */
       const int  w,           /* in - width of the panel */
       const int  jcol,        /* in - starting column index of the panel */
       SuperMatrix *A,         /* in - original matrix */
       int        *perm_r,     /* in - row permutation array */
       int        *nseg,       /* out - number of segments in the panel */
       singlecomplex     *dense,      /* out - dense array */
       int        *panel_lsub, /* out - panel data structure */
       int        *segrep,     /* out - supernodal representative */
       int        *repfnz,     /* out - supernodal first non-zero */
       int_t      *xprune,     /* out - pruned index */
       int        *marker,     /* out - marker array */
       int        *parent,     /* working array */
       int_t      *xplore,     /* working array */
       GlobalLU_t *Glu         /* modified - global LU structure */
       )
{

    NCPformat *Astore;

    /* Function begins here */
    singlecomplex    *a;   /* 指向非零元素的数组 */
    int_t     *asub;       /* 指向行索引的数组 */
    int_t     *xa_begin, *xa_end, k;   /* 指向列起始和结束位置的数组及临时变量k */
    int       krow, kmark, kperm;   /* 行和列的索引及标记变量 */
    int          krep, chperm, chmark, chrep, oldrep, kchild, myfnz, kpar;   /* 其他临时变量 */
    int       jj;       /* 遍历面板中每一列的索引 */
    int       *marker1;       /* 如果顶点jj已被前面的列访问，则marker1[jj] >= jcol */
    int       *repfnz_col; /* 面板中每一列的起始位置 */
    singlecomplex    *dense_col;  /* 面板中每一列的起始位置 */
    int_t     nextl_col;   /* 面板中存储每一列的下一个可用位置 */
    int       *xsup, *supno;   /* 全局超节点数组 */
    int_t     *lsub, *xlsub;   /* 全局非零元素索引数组 */
    int_t      xdfs, maxdfs;   /* 深度优先搜索的起始和最大值 */

    /* 初始化指针 */
    Astore     = A->Store;   /* 获取稀疏矩阵A的存储结构 */
    a          = Astore->nzval;   /* 获取非零元素数组 */
    asub       = Astore->rowind;   /* 获取行索引数组 */
    xa_begin   = Astore->colbeg;   /* 获取列起始位置数组 */
    xa_end     = Astore->colend;   /* 获取列结束位置数组 */
    marker1    = marker + m;   /* 设置marker1数组的起始位置 */
    repfnz_col = repfnz;   /* 设置repfnz_col数组的起始位置 */
    dense_col  = dense;   /* 设置dense_col数组的起始位置 */
    *nseg      = 0;   /* 初始化nseg为0 */
    xsup       = Glu->xsup;   /* 获取全局超节点数组 */
    supno      = Glu->supno;   /* 获取全局超节点编号数组 */
    lsub       = Glu->lsub;   /* 获取全局非零元素索引数组 */
    xlsub      = Glu->xlsub;   /* 获取全局非零元素索引的起始位置数组 */

    /* 对于面板中的每一列 */
    for (jj = jcol; jj < jcol + w; jj++) {
        nextl_col = (jj - jcol) * m;
#ifdef CHK_DFS
    printf("\npanel col %d: ", jj);
#endif

    /* 对于 A[*,jj] 中的每个非零元素进行深度优先搜索 */
    for (k = xa_begin[jj]; k < xa_end[jj]; k++) {
        krow = asub[k];
        dense_col[krow] = a[k];
        kmark = marker[krow];        

        if ( kmark == jj ) 
            continue;     /* krow 已经访问过，跳到下一个非零元素 */

        /* 对于 jj 的每个未标记的邻居 krow
         * 若 krow 在 L 中：将其放入 L[*,jj] 的结构中
         */
        marker[krow] = jj;
        kperm = perm_r[krow];
        
        if ( kperm == EMPTY ) {
            panel_lsub[nextl_col++] = krow; /* krow 在 A 中被索引 */
        }
        /* 
         * 若 krow 在 U 中：如果其超级节点代表 krep
         * 已经被探索过，更新 repfnz[*]
         */
        else {
            krep = xsup[supno[kperm]+1] - 1;
            myfnz = repfnz_col[krep];
        
#ifdef CHK_DFS
            printf("krep %d, myfnz %d, perm_r[%d] %d\n", krep, myfnz, krow, kperm);
#endif
            if ( myfnz != EMPTY ) {    /* 代表已经访问过 */
                if ( myfnz > kperm ) repfnz_col[krep] = kperm;
                /* continue; */
            }
            else {
                /* 否则，从 krep 开始进行深度优先搜索 */
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
                do {
                    /* 对于 krep 的每个未标记的子节点 kchild */
                    while ( xdfs < maxdfs ) {
                        kchild = lsub[xdfs];
                        xdfs++;
                        chmark = marker[kchild];
                    
                        if ( chmark != jj ) { /* 尚未到达 */
                            marker[kchild] = jj;
                            chperm = perm_r[kchild];
                      
                            /* 若 kchild 在 L 中：将其放入 L[*,j] */
                            if ( chperm == EMPTY ) {
                                panel_lsub[nextl_col++] = kchild;
                            } 
                            /* 若 kchild 在 U 中：
                             *   chrep = 其超级节点代表。若其代表已被探索，更新其 repfnz[*]
                             */
                            else {
                                chrep = xsup[supno[chperm]+1] - 1;
                                myfnz = repfnz_col[chrep];
#ifdef CHK_DFS
                                printf("chrep %d,myfnz %d,perm_r[%d] %d\n",chrep,myfnz,kchild,chperm);
#endif
                            }
                        }
                    }
                } while ( xdfs < maxdfs );
            }
        }
    }
#endif
                    if ( myfnz != EMPTY ) { /* 如果myfnz不为空，表示已经访问过 */
                    if ( myfnz > chperm )
                        repfnz_col[chrep] = chperm; /* 更新repfnz_col数组 */
                    }
                    else {
                    /* 继续在kchild的snode-rep处进行深度优先搜索 */
                    xplore[krep] = xdfs;    
                    oldrep = krep;
                    krep = chrep; /* 在G(L)中深入更深 */
                    parent[krep] = oldrep; /* 设置父节点 */
                    repfnz_col[krep] = chperm; /* 更新repfnz_col数组 */
                    xdfs = xlsub[krep];     /* 更新xdfs */
                    maxdfs = xprune[krep]; /* 更新maxdfs */
#ifdef CHK_DFS 
                    printf("  xdfs %d, maxdfs %d: ", xdfs, maxdfs);
                    for (i = xdfs; i < maxdfs; i++) printf(" %d", lsub[i]);    
                    printf("\n");
#endif
                    } /* else */
                  
                } /* else */
                  
                } /* if... */
                
            } /* while xdfs < maxdfs */ /* 循环直到xdfs不小于maxdfs */
            
            /* krow没有更多未探索的邻居:
             *    如果第一次看到这一段，将snode-rep krep放入后序DFS中。（注意"repfnz[krep]"可能会稍后改变。）
             *    回溯dfs到其父节点。
             */
            if ( marker1[krep] < jcol ) {
                segrep[*nseg] = krep; /* 将krep放入segrep数组 */
                ++(*nseg); /* 增加nseg计数 */
                marker1[krep] = jj; /* 更新marker1数组 */
            }
            
            kpar = parent[krep]; /* 弹出堆栈，模拟递归 */
            if ( kpar == EMPTY ) break; /* dfs完成 */
            krep = kpar;
            xdfs = xplore[krep]; /* 更新xdfs */
            maxdfs = xprune[krep]; /* 更新maxdfs */
            
#ifdef CHK_DFS 
            printf("  pop stack: krep %d,xdfs %d,maxdfs %d: ", krep,xdfs,maxdfs);
            for (i = xdfs; i < maxdfs; i++) printf(" %d", lsub[i]);
            printf("\n");
#endif
            } while ( kpar != EMPTY ); /* do-while - 直到堆栈为空 */
            
        } /* else */
        
        } /* else */
        
    } /* for each nonz in A[*,jj] */ /* 对A[*,jj]中的每个非零元素循环结束 */
    
    repfnz_col += m;    /* 移动到下一列 */
        dense_col += m; /* 更新dense_col指针 */
    
    } /* for jj ... */ /* 对每列结束循环 */
    
}
```