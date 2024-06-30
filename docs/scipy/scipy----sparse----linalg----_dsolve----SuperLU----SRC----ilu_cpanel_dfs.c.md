# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ilu_cpanel_dfs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ilu_cpanel_dfs.c
 * \brief Peforms a symbolic factorization on a panel of symbols and
 * record the entries with maximum absolute value in each column
 *
 * <pre>
 * -- SuperLU routine (version 4.0) --
 * Lawrence Berkeley National Laboratory
 * June 30, 2009
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
ilu_cpanel_dfs(
   const int  m,       /* in - number of rows in the matrix */
   const int  w,       /* in */
   const int  jcol,       /* in */
   SuperMatrix *A,       /* in - original matrix */
   int          *perm_r,       /* in */
   int          *nseg,       /* out */
   singlecomplex     *dense,       /* out */
   float     *amax,       /* out - max. abs. value of each column in panel */
   int          *panel_lsub, /* out */
   int          *segrep,       /* out */
   int          *repfnz,       /* out */
   int          *marker,       /* out */
   int          *parent,       /* working array */
   int_t      *xplore,       /* working array */
   GlobalLU_t *Glu       /* modified */
)
{
    NCPformat *Astore;
    singlecomplex    *a;
    int_t     *asub;
    int_t     *xa_begin, *xa_end;
    int       krep, chperm, chmark, chrep, oldrep, kchild, myfnz;
    int       krow, kmark, kperm, kpar;
    int_t     xdfs, maxdfs, k;
    int       jj;       /* index through each column in the panel */
    int       *marker1;    /* marker1[jj] >= jcol if vertex jj was visited
                  by a previous column within this panel. */
    int       *repfnz_col; /* start of each column in the panel */
    singlecomplex    *dense_col;  /* start of each column in the panel */
    # 下一个可用位置在 panel_lsub[*,jj] 中，用于存储下一个非零元素的索引
    int_t     nextl_col;   /* next available position in panel_lsub[*,jj] */
    
    # 指向稀疏矩阵 A 的非零元素数组
    int       *xsup, *supno;
    # 指向 LU 分解中 L 矩阵的数据结构
    int_t     *lsub, *xlsub;
    # 指向当前列的最大绝对值
    float    *amax_col;
    # 寄存器变量，用于临时存储计算结果
    register double tmp;

    /* Initialize pointers */
    # 从稀疏矩阵 A 中获取存储结构 Astore
    Astore     = A->Store;
    # 指向稀疏矩阵 A 的非零值数组
    a           = Astore->nzval;
    # 指向稀疏矩阵 A 的行索引数组
    asub       = Astore->rowind;
    # 稀疏矩阵 A 中每列非零元素的起始索引
    xa_begin   = Astore->colbeg;
    # 稀疏矩阵 A 中每列非零元素的结束索引
    xa_end     = Astore->colend;
    # 一个标记数组的末尾
    marker1    = marker + m;
    # 重复非零元素的列标记的指针
    repfnz_col = repfnz;
    # 密集模式下的列指针
    dense_col  = dense;
    # 列最大绝对值的指针
    amax_col   = amax;
    # 段的数量，初始化为零
    *nseg      = 0;
    # 稀疏 LU 分解的超节点信息
    xsup       = Glu->xsup;
    # 稀疏 LU 分解的超节点编号
    supno      = Glu->supno;
    # 稀疏 LU 分解中 L 矩阵的列链表
    lsub       = Glu->lsub;
    # 稀疏 LU 分解中 L 矩阵的列链表起始位置
    xlsub      = Glu->xlsub;

    /* For each column in the panel */
    # 对于面板中的每一列进行循环处理
    for (jj = jcol; jj < jcol + w; jj++) {
    # 下一个可用位置在 panel_lsub[*,jj] 中的计算，用于存储下一个非零元素的索引
    nextl_col = (jj - jcol) * m;
#ifdef CHK_DFS
    // 如果定义了 CHK_DFS 宏，则输出面板列号 jj 的信息
    printf("\npanel col %d: ", jj);
#endif

    // 将 amax_col 初始化为 0.0
    *amax_col = 0.0;
    /* 对于 A[*,jj] 中的每个非零元素进行深度优先搜索 */
    for (k = xa_begin[jj]; k < xa_end[jj]; k++) {
        krow = asub[k];  // 获取行索引 krow
        tmp = c_abs1(&a[k]);  // 计算当前非零元素的绝对值
        if (tmp > *amax_col) *amax_col = tmp;  // 更新 amax_col 的最大值
        dense_col[krow] = a[k];  // 将非零元素 a[k] 放入稠密列 dense_col[krow]
        kmark = marker[krow];  // 获取 krow 的标记值

        if ( kmark == jj )
            continue;     /* krow 已经访问过，继续处理下一个非零元素 */

        /* 对于 jj 的未标记邻居 krow
         * 如果 krow 在 L 中：将其放入 L[*,jj] 的结构中
         */
        marker[krow] = jj;  // 标记 krow
        kperm = perm_r[krow];  // 获取 krow 的列排列索引

        if ( kperm == EMPTY ) {
            panel_lsub[nextl_col++] = krow; /* krow 在 A 中有索引 */  // 将 krow 的索引放入 panel_lsub 中
        }
        /*
         * 如果 krow 在 U 中：
         * 如果其超级节点代表 krep 已经被探索过，则更新 repfnz[*]
         */
        else {
            krep = xsup[supno[kperm]+1] - 1;  // 获取超级节点 krep 的结尾索引
            myfnz = repfnz_col[krep];  // 获取 repfnz_col[krep] 的值
            
#ifdef CHK_DFS
            // 如果定义了 CHK_DFS 宏，则输出相关调试信息
            printf("krep %d, myfnz %d, perm_r[%d] %d\n", krep, myfnz, krow, kperm);
#endif
            if ( myfnz != EMPTY ) { /* 代表已经访问过的超级节点 */
                if ( myfnz > kperm ) repfnz_col[krep] = kperm;  // 更新 repfnz_col[krep]
                /* continue; */
            }
            else {
                /* 否则，从 krep 开始进行深度优先搜索 */
                oldrep = EMPTY;
                parent[krep] = oldrep;
                repfnz_col[krep] = kperm;
                xdfs = xlsub[xsup[supno[krep]]];
                maxdfs = xlsub[krep + 1];

#ifdef CHK_DFS
                // 如果定义了 CHK_DFS 宏，则输出相关调试信息
                printf("  xdfs %d, maxdfs %d: ", xdfs, maxdfs);
                for (i = xdfs; i < maxdfs; i++) printf(" %d", lsub[i]);
                printf("\n");
#endif
                do {
                /*
                 * 对于 krep 的每个未标记子节点 kchild
                 */
                while ( xdfs < maxdfs ) {

                    kchild = lsub[xdfs];
                    xdfs++;
                    chmark = marker[kchild];

                    if ( chmark != jj ) { /* 尚未到达 */
                        marker[kchild] = jj;  // 标记 kchild
                        chperm = perm_r[kchild];  // 获取 kchild 的列排列索引

                        /* 如果 kchild 在 L 中：将其放入 L[*,j] 中 */
                        if ( chperm == EMPTY ) {
                            panel_lsub[nextl_col++] = kchild;  // 将 kchild 的索引放入 panel_lsub 中
                        }
                        /* 如果 kchild 在 U 中：
                         *   chrep = 其超级节点代表。如果其代表已经被探索过，则更新其 repfnz[*]
                         */
                        else {
                            chrep = xsup[supno[chperm]+1] - 1;  // 获取超级节点 chrep 的结尾索引
                            myfnz = repfnz_col[chrep];  // 获取 repfnz_col[chrep] 的值
#ifdef CHK_DFS
                            // 如果定义了 CHK_DFS 宏，则输出相关调试信息
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
                    if ( myfnz != EMPTY ) { /* 如果myfnz不为空，表示此节点已访问过 */
                    if ( myfnz > chperm )
                        repfnz_col[chrep] = chperm; /* 更新repfnz_col数组中的值 */
                    }
                    else {
                    /* 继续在kchild的snode-rep处进行深度优先搜索 */
                    xplore[krep] = xdfs; /* 更新xplore数组中的值 */
                    oldrep = krep; /* 保存旧的krep值 */
                    krep = chrep; /* 进入下一层的G(L) */
                    parent[krep] = oldrep; /* 更新parent数组中的值 */
                    repfnz_col[krep] = chperm; /* 更新repfnz_col数组中的值 */
                    xdfs = xlsub[xsup[supno[krep]]]; /* 更新xdfs的值 */
                    maxdfs = xlsub[krep + 1]; /* 更新maxdfs的值 */
#ifdef CHK_DFS
                    printf("  xdfs %d, maxdfs %d: ", xdfs, maxdfs); /* 打印调试信息 */
                    for (i = xdfs; i < maxdfs; i++) printf(" %d", lsub[i]); /* 打印调试信息 */
                    printf("\n"); /* 打印调试信息 */
#endif
                    } /* else */

                } /* else */

                } /* if... */

            } /* while xdfs < maxdfs */

            /* krow没有更多未探索的邻居：
             *    如果这段首次见到，将snode-rep krep放入后序DFS中。
             *    回溯dfs到其父节点。
             */
            if ( marker1[krep] < jcol ) {
                segrep[*nseg] = krep; /* 更新segrep数组中的值 */
                ++(*nseg); /* 更新nseg的值 */
                marker1[krep] = jj; /* 更新marker1数组中的值 */
            }

            kpar = parent[krep]; /* 出栈，模拟递归 */
            if ( kpar == EMPTY ) break; /* dfs完成 */
            krep = kpar; /* 更新krep的值 */
            xdfs = xplore[krep]; /* 更新xdfs的值 */
            maxdfs = xlsub[krep + 1]; /* 更新maxdfs的值 */

#ifdef CHK_DFS
            printf("  pop stack: krep %d,xdfs %d,maxdfs %d: ", krep,xdfs,maxdfs); /* 打印调试信息 */
            for (i = xdfs; i < maxdfs; i++) printf(" %d", lsub[i]); /* 打印调试信息 */
            printf("\n"); /* 打印调试信息 */
#endif
            } while ( kpar != EMPTY ); /* do-while - 直到栈为空 */

        } /* else */
        
        } /* else */

    } /* for each nonz in A[*,jj] */

    repfnz_col += m;    /* 移动到下一列 */
    dense_col += m; /* 更新dense_col的值 */
    amax_col++; /* 更新amax_col的值 */

    } /* for jj ... */

}
```