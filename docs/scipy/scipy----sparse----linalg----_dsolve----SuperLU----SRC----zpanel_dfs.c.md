# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zpanel_dfs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file zpanel_dfs.c
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


#include "slu_zdefs.h"

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
zpanel_dfs (
       const int  m,           /* in - number of rows in the matrix */
       const int  w,           /* in - width of the panel */
       const int  jcol,        /* in - starting column index of the panel */
       SuperMatrix *A,         /* in - original matrix */
       int        *perm_r,     /* in - row permutation vector */
       int        *nseg,       /* out - number of segments */
       doublecomplex *dense,   /* out - dense array */
       int        *panel_lsub, /* out - panel of L subscripts */
       int        *segrep,     /* out - segment representatives */
       int        *repfnz,     /* out - supernodal first non-zero locations */
       int_t      *xprune,     /* out - pruning array */
       int        *marker,     /* out - marker array */
       int        *parent,     /* working array */
       int_t      *xplore,     /* working array */
       GlobalLU_t *Glu         /* modified - global LU data structure */
       )
{

    NCPformat *Astore;


注释：

/*! \file
版权声明和许可信息，指定本文件在 BSD 许可下分发
*/

/*! @file zpanel_dfs.c
 * \brief 对符号分解中的一个列面板执行符号因子分解
 *
 * <pre>
 * -- SuperLU 例程 (版本 2.0) --
 * 加州大学伯克利分校，施乐帕洛阿尔托研究中心，
 * 以及劳伦斯伯克利国家实验室。
 * 1997年11月15日
 *
 * 版权所有 (c) 1994 施乐公司。保留所有权利。
 *
 * 本软件按原样提供，没有任何明示或暗示的担保。
 * 任何使用均为用户自担风险。
 *
 * 可以免费使用或复制本程序以任何目的，只要上述通知在所有副本中都保留。
 * 允许修改代码和分发修改后的代码，只要保留上述通知，并在上述版权声明中包含代码已被修改的说明。
 * </pre>
 */


#include "slu_zdefs.h"

/*! \brief 执行符号因子分解的函数
 *
 * <pre>
 * 目的
 * =======
 *
 *   对一个列面板的列 [jcol, jcol+w) 执行符号因子分解。
 *
 *   超节点代表是超节点的最后一列。
 *   U[*,j] 中的非零元素是以超节点代表结尾的段。
 *
 *   此例程按生成它们的深度优先搜索的拓扑顺序返回一个超节点代表的列表。
 *   这个列表是每个面板内每个单独列的拓扑顺序的超集。
 *   还返回每个超节点段中第一个非零元素的位置 (超节点入口位置)。每列为此有单独的列表。
 *
 *   使用两个标记数组进行深度优先搜索：
 *     如果 i 在当前列 jj 的dfs中被访问，那么 marker[i] == jj；
 *     如果 i 被此面板中之前的列访问，那么 marker1[i] >= jcol；
 *
 *   marker: A 的行 --> A 的行/列 (0/1)
 *   repfnz: 超级矩阵 A 的列 --> PA 的行
 *   parent: 超级矩阵 A 的列 --> 超级矩阵 A 的列
 *   xplore: 超级矩阵 A 的列 --> L 结构的索引
 * </pre>
 */

void
zpanel_dfs (
       const int  m,           /* 输入 - 矩阵的行数 */
       const int  w,           /* 输入 - 面板的宽度 */
       const int  jcol,        /* 输入 - 面板的起始列索引 */
       SuperMatrix *A,         /* 输入 - 原始矩阵 */
       int        *perm_r,     /* 输入 - 行置换向量 */
       int        *nseg,       /* 输出 - 段的数量 */
       doublecomplex *dense,   /* 输出 - 密集数组 */
       int        *panel_lsub, /* 输出 - L 的子脚本面板 */
       int        *segrep,     /* 输出 - 段代表 */
       int        *repfnz,     /* 输出 - 超节点第一个非零位置 */
       int_t      *xprune,     /* 输出 - 剪枝数组 */
       int        *marker,     /* 输出 - 标记数组 */
       int        *parent,     /* 工作数组 */
       int_t      *xplore,     /* 工作数组 */
       GlobalLU_t *Glu         /* 修改的全局 LU 数据结构 */
       )
{

    NCPformat *Astore;


这样的注释方式保证了对每行代码的逐一解释，包括函数目的、参数说明以及主要数据结构的用途和含义，以便他人能够理解和修改代码。
    doublecomplex    *a;    /* 指向稀疏矩阵 A 的非零元素数组的指针 */
    int_t     *asub;        /* 指向稀疏矩阵 A 的行索引数组的指针 */
    int_t     *xa_begin, *xa_end, k;    /* xa_begin 和 xa_end 分别指向每列在 asub 中的起始和结束位置，k 是计数器 */
    int       krow, kmark, kperm;    /* 各种整型变量，用于算法中的索引和标记 */
    int          krep, chperm, chmark, chrep, oldrep, kchild, myfnz, kpar;    /* 更多整型变量，用于算法中的状态和控制 */
    int       jj;       /* 遍历每个面板中的列的索引 */
    int       *marker1;       /* 如果 marker1[jj] >= jcol，表示列 jj 已经被此面板中的先前列访问过 */
    int       *repfnz_col;    /* 每个面板中列的起始位置 */
    doublecomplex    *dense_col;  /* 每个面板中列的起始位置 */
    int_t     nextl_col;   /* 面板中下一个可用位置的位置指示 */
    int       *xsup, *supno;    /* xsup 是全局超节点起始索引数组，supno 是超节点编号数组 */
    int_t     *lsub, *xlsub;    /* lsub 是全局 L 列索引数组，xlsub 是其起始位置 */

    /* Initialize pointers */
    Astore     = A->Store;    /* 获取稀疏矩阵 A 的存储结构 */
    a          = Astore->nzval;    /* 获取稀疏矩阵 A 的非零元素数组 */
    asub       = Astore->rowind;   /* 获取稀疏矩阵 A 的行索引数组 */
    xa_begin   = Astore->colbeg;   /* 获取稀疏矩阵 A 的每列在 asub 中的起始位置 */
    xa_end     = Astore->colend;   /* 获取稀疏矩阵 A 的每列在 asub 中的结束位置 */
    marker1    = marker + m;       /* 标记数组，用于记录顶点是否被访问过 */
    repfnz_col = repfnz;           /* 每个列的起始位置 */
    dense_col  = dense;            /* 密集列的起始位置 */
    *nseg      = 0;                /* 初始化 nseg 为 0 */
    xsup       = Glu->xsup;        /* 获取全局超节点起始索引数组 */
    supno      = Glu->supno;       /* 获取超节点编号数组 */
    lsub       = Glu->lsub;        /* 获取全局 L 列索引数组 */
    xlsub      = Glu->xlsub;       /* 获取全局 L 列索引数组的起始位置 */

    /* For each column in the panel */
    for (jj = jcol; jj < jcol + w; jj++) {
    nextl_col = (jj - jcol) * m;    /* 计算面板中下一个列的位置 */
#ifdef CHK_DFS
    // 如果定义了 CHK_DFS 宏，则输出面板列 jj 的信息
    printf("\npanel col %d: ", jj);
#endif

    /* 对于 A[*,jj] 中的每个非零元素执行深度优先搜索 */
    for (k = xa_begin[jj]; k < xa_end[jj]; k++) {
        krow = asub[k];
        
        // 将 A[k] 存入 dense_col 中的 krow 位置
        dense_col[krow] = a[k];
        
        kmark = marker[krow];        
        
        // 如果 krow 已经被访问过，继续下一个非零元素的处理
        if ( kmark == jj ) 
            continue;     /* krow visited before, go to the next nonzero */

        /* 对于 jj 的每个未标记的邻居 krow
         * 如果 krow 在 L 中：将其放入 L[*,jj] 的结构中
         */
        marker[krow] = jj;
        kperm = perm_r[krow];
        
        if ( kperm == EMPTY ) {
            // krow 在 A 中索引到，将其加入 panel_lsub 中
            panel_lsub[nextl_col++] = krow;
        }
        /* 
         * 如果 krow 在 U 中：如果其超节点代表 krep
         * 已经被探索过，则更新 repfnz[*]
         */
        else {
            krep = xsup[supno[kperm]+1] - 1;
            myfnz = repfnz_col[krep];
            
#ifdef CHK_DFS
            // 如果定义了 CHK_DFS 宏，则输出 krep、myfnz、perm_r[krow] 等信息
            printf("krep %d, myfnz %d, perm_r[%d] %d\n", krep, myfnz, krow, kperm);
#endif

            if ( myfnz != EMPTY ) {    /* 代表已经被访问过 */
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
                // 如果定义了 CHK_DFS 宏，则输出 xdfs 和 maxdfs 的信息，以及相关数组的内容
                printf("  xdfs %d, maxdfs %d: ", xdfs, maxdfs);
                for (i = xdfs; i < maxdfs; i++) printf(" %d", lsub[i]);
                printf("\n");
#endif

                do {
                    /* 对于 krep 的每个未标记的 kchild */
                    while ( xdfs < maxdfs ) {
                        
                        kchild = lsub[xdfs];
                        xdfs++;
                        chmark = marker[kchild];
                        
                        // 如果 kchild 尚未达到，则标记为已达到
                        if ( chmark != jj ) {
                            marker[kchild] = jj;
                            chperm = perm_r[kchild];
                            
                            /* 如果 kchild 在 L 中：将其放入 L[*,j] */
                            if ( chperm == EMPTY ) {
                                panel_lsub[nextl_col++] = kchild;
                            } 
                            /* 如果 kchild 在 U 中：chrep 是其超节点代表 */
                            else {
                                chrep = xsup[supno[chperm]+1] - 1;
                                myfnz = repfnz_col[chrep];
#ifdef CHK_DFS
                                // 如果定义了 CHK_DFS 宏，则输出 chrep、myfnz、perm_r[kchild] 等信息
                                printf("chrep %d, myfnz %d, perm_r[%d] %d\n", chrep, myfnz, kchild, chperm);
#endif
                                // 继续深度优先搜索
                            }
                        }
                    }
                } while ( /* 继续执行深度优先搜索 */ );
            }
        }
    }
#else
                    // 如果myfnz不为空，表示此前已访问过
                    if ( myfnz != EMPTY ) { 
                        // 如果myfnz大于chperm，更新repfnz_col[chrep]
                        if ( myfnz > chperm )
                            repfnz_col[chrep] = chperm;
                    }
                    else {
                        // 在snode-rep的kchild处继续深度优先搜索
                        xplore[krep] = xdfs;
                        oldrep = krep;
                        krep = chrep; // 在G(L)中向下深入
                        parent[krep] = oldrep;
                        repfnz_col[krep] = chperm;
                        xdfs = xlsub[krep];
                        maxdfs = xprune[krep];
#ifdef CHK_DFS
                        // 调试输出：显示当前的xdfs和maxdfs，以及相关的lsub数组内容
                        printf("  xdfs %d, maxdfs %d: ", xdfs, maxdfs);
                        for (i = xdfs; i < maxdfs; i++) printf(" %d", lsub[i]);
                        printf("\n");
#endif
                    } /* else */
                  
                } /* else */
                  
            } /* if... */
                
        } /* while xdfs < maxdfs */
            
        /* krow没有更多未探索的邻居：
         * 将snode-rep krep放入后序DFS中，如果该段第一次见到。（注意"repfnz[krep]"可能稍后会改变。）
         * 回溯DFS到其父节点。
         */
        if ( marker1[krep] < jcol ) {
            segrep[*nseg] = krep;
            ++(*nseg);
            marker1[krep] = jj;
        }
            
        kpar = parent[krep]; // 出栈，模拟递归
        if ( kpar == EMPTY ) break; // DFS完成
        krep = kpar;
        xdfs = xplore[krep];
        maxdfs = xprune[krep];
            
#ifdef CHK_DFS
        // 调试输出：显示出栈操作后的krep、xdfs和maxdfs，以及相关的lsub数组内容
        printf("  pop stack: krep %d, xdfs %d, maxdfs %d: ", krep, xdfs, maxdfs);
        for (i = xdfs; i < maxdfs; i++) printf(" %d", lsub[i]);
        printf("\n");
#endif
    } while ( kpar != EMPTY ); // do-while，直到栈为空

} /* else */

} /* else */

} /* for each nonz in A[*,jj] */

repfnz_col += m;    // 移动到下一列
dense_col += m;

} /* for jj ... */

}
```