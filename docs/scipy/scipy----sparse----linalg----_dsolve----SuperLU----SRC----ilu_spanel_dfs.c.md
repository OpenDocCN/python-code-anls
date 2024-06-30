# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ilu_spanel_dfs.c`

```
/*
 * Peforms a symbolic factorization on a panel of symbols and
 * records the entries with maximum absolute value in each column.
 */

#include "slu_sdefs.h"

/*
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
 */
void
ilu_spanel_dfs(
   const int  m,         /* in - number of rows in the matrix */
   const int  w,         /* in - width of the panel */
   const int  jcol,      /* in - starting column index of the panel */
   SuperMatrix *A,       /* in - original matrix */
   int          *perm_r, /* in - row permutation */
   int          *nseg,   /* out - number of supernodal segments */
   float     *dense,     /* out - dense array */
   float     *amax,      /* out - max. abs. value of each column in panel */
   int          *panel_lsub, /* out - panel column indices */
   int          *segrep,  /* out - supernodal representatives */
   int          *repfnz,  /* out - supernodal entry locations */
   int          *marker,  /* out - marker array */
   int          *parent,  /* working array */
   int_t      *xplore,    /* working array */
   GlobalLU_t *Glu        /* modified */
)
{
    NCPformat *Astore;
    float    *a;
    int_t     *asub;
    int_t     *xa_begin, *xa_end;
    int       krep, chperm, chmark, chrep, oldrep, kchild, myfnz;
    int       krow, kmark, kperm, kpar;
    int_t     xdfs, maxdfs, k;
    int       jj;          /* index through each column in the panel */
    int       *marker1;    /* marker1[jj] >= jcol if vertex jj was visited
                              by a previous column within this panel. */
    int       *repfnz_col; /* start of each column in the panel in repfnz */
    float    *dense_col;   /* start of each column in the panel in dense */
    # 下一个可用位置在 panel_lsub[*,jj] 中的列指标
    int_t     nextl_col;   /* next available position in panel_lsub[*,jj] */
    
    # 列指针数组和超节点编号数组
    int       *xsup, *supno;
    
    # 列主元下标数组和列主元数组的起始位置数组
    int_t     *lsub, *xlsub;
    
    # 列最大绝对值数组
    float    *amax_col;
    
    # 临时寄存器变量
    register double tmp;
    
    /* Initialize pointers */
    # 初始化指针和数组
    Astore     = A->Store;
    a           = Astore->nzval;
    asub       = Astore->rowind;
    xa_begin   = Astore->colbeg;
    xa_end     = Astore->colend;
    
    # 设置标记数组的边界
    marker1    = marker + m;
    
    # 指向副本首地址的指针
    repfnz_col = repfnz;
    
    # 密集列指针
    dense_col  = dense;
    
    # 列最大值数组
    amax_col   = amax;
    
    # 初始化段号为零
    *nseg      = 0;
    
    # 全局超节点指针和超节点编号指针
    xsup       = Glu->xsup;
    supno      = Glu->supno;
    
    # 全局非零列指针和全局非零列的长度指针
    lsub       = Glu->lsub;
    xlsub      = Glu->xlsub;

    /* For each column in the panel */
    # 对于面板中的每一列
    for (jj = jcol; jj < jcol + w; jj++) {
    
        # 下一个可用位置在 panel_lsub[*,jj] 中的列偏移量
        nextl_col = (jj - jcol) * m;
#ifdef CHK_DFS
    printf("\npanel col %d: ", jj);
#endif

    *amax_col = 0.0;
    /* 对于 A[*,jj] 中的每个非零元素执行深度优先搜索 */
    for (k = xa_begin[jj]; k < xa_end[jj]; k++) {
        krow = asub[k];
        tmp = fabs(a[k]);
        if (tmp > *amax_col) *amax_col = tmp;
        dense_col[krow] = a[k];
        kmark = marker[krow];
        if ( kmark == jj )
            continue;     /* krow 已经访问过，继续处理下一个非零元素 */

        /* 对于 jj 的每个未标记邻居 krow
         * 如果 krow 在 L 中：将其放入 L[*,jj] 的结构中 */
        marker[krow] = jj;
        kperm = perm_r[krow];

        if ( kperm == EMPTY ) {
            panel_lsub[nextl_col++] = krow; /* krow 在 A 中的索引位置 */
        }
        /*
         * 如果 krow 在 U 中：如果其超级节点代表 krep
         * 已被探索过，则更新 repfnz[*]
         */
        else {
            krep = xsup[supno[kperm]+1] - 1;
            myfnz = repfnz_col[krep];
        
#ifdef CHK_DFS
            printf("krep %d, myfnz %d, perm_r[%d] %d\n", krep, myfnz, krow, kperm);
#endif
            if ( myfnz != EMPTY ) { /* 代表已经访问过 */
                if ( myfnz > kperm ) repfnz_col[krep] = kperm;
                /* continue; */
            }
            else {
                /* 否则，从 krep 开始执行深度优先搜索 */
                oldrep = EMPTY;
                parent[krep] = oldrep;
                repfnz_col[krep] = kperm;
                xdfs = xlsub[xsup[supno[krep]]];
                maxdfs = xlsub[krep + 1];

#ifdef CHK_DFS
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

                        if ( chmark != jj ) { /* 还未到达 */
                            marker[kchild] = jj;
                            chperm = perm_r[kchild];

                            /* 如果 kchild 在 L 中：将其放入 L[*,j] */
                            if ( chperm == EMPTY ) {
                                panel_lsub[nextl_col++] = kchild;
                            }
                            /* 如果 kchild 在 U 中：
                             *   chrep = 其超级节点代表。如果其代表已经
                             *   被探索过，则更新其 repfnz[*]
                             */
                            else {
                                chrep = xsup[supno[chperm]+1] - 1;
                                myfnz = repfnz_col[chrep];
#ifdef CHK_DFS
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
                    if ( myfnz != EMPTY ) { /* 如果已经访问过 */
                    if ( myfnz > chperm )
                        repfnz_col[chrep] = chperm;
                    }
                    else {
                    /* 在 kchild 的 snode-rep 处继续深度优先搜索 */
                    xplore[krep] = xdfs;
                    oldrep = krep;
                    krep = chrep; /* 进入 G(L) 的更深层 */
                    parent[krep] = oldrep;
                    repfnz_col[krep] = chperm;
                    xdfs = xlsub[xsup[supno[krep]]];
                    maxdfs = xlsub[krep + 1];
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
             *    如果第一次见到这个段，将 snode-rep krep 放入后序 DFS 中。（注意 "repfnz[krep]" 以后可能会改变。）
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
            maxdfs = xlsub[krep + 1];

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
    amax_col++;

    } /* for jj ... */

}
```