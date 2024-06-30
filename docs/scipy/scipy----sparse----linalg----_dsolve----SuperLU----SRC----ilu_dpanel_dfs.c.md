# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ilu_dpanel_dfs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ilu_dpanel_dfs.c
 * \brief Peforms a symbolic factorization on a panel of symbols and
 * record the entries with maximum absolute value in each column
 *
 * <pre>
 * -- SuperLU routine (version 4.0) --
 * Lawrence Berkeley National Laboratory
 * June 30, 2009
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
ilu_dpanel_dfs(
   const int  m,       /* in - number of rows in the matrix */
   const int  w,       /* in */
   const int  jcol,    /* in */
   SuperMatrix *A,     /* in - original matrix */
   int          *perm_r,  /* in */
   int          *nseg,    /* out */
   double     *dense,    /* out */
   double     *amax,     /* out - max. abs. value of each column in panel */
   int          *panel_lsub, /* out */
   int          *segrep,  /* out */
   int          *repfnz,  /* out */
   int          *marker,  /* out */
   int          *parent,  /* working array */
   int_t      *xplore,   /* working array */
   GlobalLU_t *Glu       /* modified */
)
{
    NCPformat *Astore;      /* pointer to the stored format of matrix A */
    double    *a;           /* numerical values of the matrix A */
    int_t     *asub;        /* subscript of elements in the matrix A */
    int_t     *xa_begin, *xa_end; /* pointers to beginning and end of columns */
    int       krep, chperm, chmark, chrep, oldrep, kchild, myfnz; /* integer variables used in computations */
    int       krow, kmark, kperm, kpar; /* more integer variables used in computations */
    int_t     xdfs, maxdfs, k; /* integer variables used in DFS and related calculations */
    int       jj;            /* index through each column in the panel */
    int       *marker1;      /* marker1[jj] >= jcol if vertex jj was visited by a previous column within this panel. */
    int       *repfnz_col;   /* start of each column in the panel */
    double    *dense_col;    /* start of each column in the panel */
    /* 下一个可用位置在 panel_lsub[*,jj] 中 */
    int_t     nextl_col;   /* next available position in panel_lsub[*,jj] */
    /* xsup 和 supno 是指向全局 LU 结构的指针 */
    int       *xsup, *supno;
    /* lsub 和 xlsub 是指向全局 LU 结构的指针 */
    int_t     *lsub, *xlsub;
    /* 每列的最大绝对值 */
    double    *amax_col;
    /* 临时变量 */
    register double tmp;

    /* 初始化指针 */
    Astore     = A->Store;   /* Astore 指向矩阵 A 的存储 */
    a          = Astore->nzval;   /* a 指向非零元素数组 */
    asub       = Astore->rowind;  /* asub 指向行索引数组 */
    xa_begin   = Astore->colbeg;  /* xa_begin 指向列开始位置数组 */
    xa_end     = Astore->colend;  /* xa_end 指向列结束位置数组 */
    marker1    = marker + m;      /* marker1 指向 marker 数组的第 m 个元素 */
    repfnz_col = repfnz;          /* repfnz_col 指向 repfnz 数组 */
    dense_col  = dense;           /* dense_col 指向 dense 数组 */
    amax_col   = amax;            /* amax_col 指向 amax 数组 */
    *nseg      = 0;               /* nseg 设为零 */
    xsup       = Glu->xsup;       /* xsup 指向 Glu 结构体中的 xsup 数组 */
    supno      = Glu->supno;       /* supno 指向 Glu 结构体中的 supno 数组 */
    lsub       = Glu->lsub;        /* lsub 指向 Glu 结构体中的 lsub 数组 */
    xlsub      = Glu->xlsub;       /* xlsub 指向 Glu 结构体中的 xlsub 数组 */

    /* 对于面板中的每一列 */
    for (jj = jcol; jj < jcol + w; jj++) {
        /* 计算 panel_lsub[*,jj] 中的下一个可用位置 */
        nextl_col = (jj - jcol) * m;


这段代码主要涉及了一些指针和变量的初始化，用于 LU 分解过程中的矩阵操作。
#ifdef CHK_DFS
    /* 如果定义了 CHK_DFS 宏，则打印面板列号 */
    printf("\npanel col %d: ", jj);
#endif

    /* 初始化列最大绝对值为0 */
    *amax_col = 0.0;
    /* 对于 A[*,jj] 中的每个非零元素进行深度优先搜索 */
    for (k = xa_begin[jj]; k < xa_end[jj]; k++) {
        krow = asub[k];
        tmp = fabs(a[k]);
        /* 更新当前列的最大绝对值 */
        if (tmp > *amax_col) *amax_col = tmp;
        /* 将 A[*,jj] 的非零元素复制到 dense_col 数组 */
        dense_col[krow] = a[k];
        /* 获取 krow 的标记值 */
        kmark = marker[krow];
        /* 如果 krow 已被访问过，则继续下一个非零元素 */
        if (kmark == jj)
            continue;

        /* 对于 jj 的未标记邻居 krow
         * 如果 krow 在 L 中：将其放入 L[*,jj] 的结构中
         */
        marker[krow] = jj;
        kperm = perm_r[krow];

        if (kperm == EMPTY) {
            panel_lsub[nextl_col++] = krow; /* krow 在 A 中索引 */
        }
        /*
         * 如果 krow 在 U 中：如果其超级节点代表 krep
         * 已被探索，则更新 repfnz[*]
         */
        else {
            krep = xsup[supno[kperm]+1] - 1;
            myfnz = repfnz_col[krep];
            
#ifdef CHK_DFS
            /* 如果定义了 CHK_DFS 宏，则打印 krep、myfnz 和 perm_r[krow] 的值 */
            printf("krep %d, myfnz %d, perm_r[%d] %d\n", krep, myfnz, krow, kperm);
#endif
            if (myfnz != EMPTY) { /* 代表已被访问过 */
                if (myfnz > kperm) repfnz_col[krep] = kperm;
                /* 继续; */
            }
            else {
                /* 否则，从 krep 开始执行深度优先搜索 */
                oldrep = EMPTY;
                parent[krep] = oldrep;
                repfnz_col[krep] = kperm;
                xdfs = xlsub[xsup[supno[krep]]];
                maxdfs = xlsub[krep + 1];

#ifdef CHK_DFS
                /* 如果定义了 CHK_DFS 宏，则打印 xdfs 和 maxdfs 的值以及相关数组内容 */
                printf("  xdfs %d, maxdfs %d: ", xdfs, maxdfs);
                for (i = xdfs; i < maxdfs; i++) printf(" %d", lsub[i]);
                printf("\n");
#endif
                do {
                    /*
                     * 对于 krep 的每个未标记 kchild
                     */
                    while (xdfs < maxdfs) {
                        kchild = lsub[xdfs];
                        xdfs++;
                        chmark = marker[kchild];

                        if (chmark != jj) { /* 尚未到达 */
                            marker[kchild] = jj;
                            chperm = perm_r[kchild];

                            /* 如果 kchild 在 L 中：将其放入 L[*,j] */
                            if (chperm == EMPTY) {
                                panel_lsub[nextl_col++] = kchild;
                            }
                            /* 如果 kchild 在 U 中：
                             *   chrep = 其超级节点代表。如果其代表已被
                             *   探索，则更新其 repfnz[*]
                             */
                            else {
                                chrep = xsup[supno[chperm]+1] - 1;
                                myfnz = repfnz_col[chrep];
#ifdef CHK_DFS
                                /* 如果定义了 CHK_DFS 宏，则打印 chrep、myfnz 和 perm_r[kchild] 的值 */
                                printf("chrep %d, myfnz %d, perm_r[%d] %d\n", chrep, myfnz, kchild, chperm);
#endif
#endif
                    if ( myfnz != EMPTY ) { /* 如果myfnz不是EMPTY，则说明已经访问过 */
                    if ( myfnz > chperm )
                        repfnz_col[chrep] = chperm; /* 更新repfnz_col[chrep] */
                    }
                    else {
                    /* 继续在kchild的snode-rep处进行深度优先搜索 */
                    xplore[krep] = xdfs; /* 记录当前状态的深度优先搜索值 */
                    oldrep = krep; /* 保存当前的krep状态 */
                    krep = chrep; /* 深入到G(L)中的下一层 */
                    parent[krep] = oldrep; /* 记录父节点 */
                    repfnz_col[krep] = chperm; /* 更新repfnz_col[krep] */
                    xdfs = xlsub[xsup[supno[krep]]]; /* 更新xdfs为下一层的起始值 */
                    maxdfs = xlsub[krep + 1]; /* 更新maxdfs为当前层的结束值 */
#ifdef CHK_DFS
                    printf("  xdfs %d, maxdfs %d: ", xdfs, maxdfs);
                    for (i = xdfs; i < maxdfs; i++) printf(" %d", lsub[i]);
                    printf("\n");
#endif
                    } /* else */

                } /* else */

                } /* if... */

            } /* while xdfs < maxdfs */

            /* krow没有更多未探索的邻居：
             *    如果第一次看到该段，将snode-rep krep放入后序DFS中。（注意"repfnz[krep]"可能会后续更改）
             *    回溯DFS到其父节点。
             */
            if ( marker1[krep] < jcol ) {
                segrep[*nseg] = krep; /* 将krep放入后序DFS中 */
                ++(*nseg); /* 更新段的数量 */
                marker1[krep] = jj; /* 标记该段已访问 */
            }

            kpar = parent[krep]; /* 出栈，模拟递归 */
            if ( kpar == EMPTY ) break; /* DFS完成 */
            krep = kpar; /* 更新当前的krep为父节点 */
            xdfs = xplore[krep]; /* 恢复xdfs为父节点的深度优先搜索值 */
            maxdfs = xlsub[krep + 1]; /* 更新maxdfs为父节点的结束值 */

#ifdef CHK_DFS
            printf("  pop stack: krep %d,xdfs %d,maxdfs %d: ", krep,xdfs,maxdfs);
            for (i = xdfs; i < maxdfs; i++) printf(" %d", lsub[i]);
            printf("\n");
#endif
            } while ( kpar != EMPTY ); /* 循环直到栈空 */

        } /* else */
        
        } /* else */

    } /* for each nonz in A[*,jj] */

    repfnz_col += m;    /* 移动到下一列 */
    dense_col += m; /* 更新dense_col */
    amax_col++; /* 更新amax_col */

    } /* for jj ... */

}
```