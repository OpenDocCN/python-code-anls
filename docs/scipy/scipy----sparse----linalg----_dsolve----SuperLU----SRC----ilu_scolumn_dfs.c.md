# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ilu_scolumn_dfs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ilu_scolumn_dfs.c
 * \brief Performs a symbolic factorization
 *
 * <pre>
 * -- SuperLU routine (version 4.0) --
 * Lawrence Berkeley National Laboratory
 * June 30, 2009
 * </pre>
*/

#include "slu_sdefs.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   ILU_SCOLUMN_DFS performs a symbolic factorization on column jcol, and
 *   decide the supernode boundary.
 *
 *   This routine does not use numeric values, but only use the RHS
 *   row indices to start the dfs.
 *
 *   A supernode representative is the last column of a supernode.
 *   The nonzeros in U[*,j] are segments that end at supernodal
 *   representatives. The routine returns a list of such supernodal
 *   representatives in topological order of the dfs that generates them.
 *   The location of the first nonzero in each such supernodal segment
 *   (supernodal entry location) is also returned.
 *
 * Local parameters
 * ================
 *   nseg: no of segments in current U[*,j]
 *   jsuper: jsuper=EMPTY if column j does not belong to the same
 *    supernode as j-1. Otherwise, jsuper=nsuper.
 *
 *   marker2: A-row --> A-row/col (0/1)
 *   repfnz: SuperA-col --> PA-row
 *   parent: SuperA-col --> SuperA-col
 *   xplore: SuperA-col --> index to L-structure
 *
 * Return value
 * ============
 *     0  success;
 *   > 0  number of bytes allocated when run out of space.
 * </pre>
 */
int
ilu_scolumn_dfs(
       const int  m,     /* in - number of rows in the matrix */
       const int  jcol,     /* in - current column being processed */
       int          *perm_r,     /* in - row permutation array */
       int          *nseg,     /* modified - number of segments appended */
       int          *lsub_col, /* in - defines the RHS vector to start the dfs */
       int          *segrep,     /* modified - list of supernodal representatives */
       int          *repfnz,     /* modified - supernodal entry locations */
       int          *marker,     /* modified - marker array for dfs */
       int          *parent,     /* working array - parent of supernodes */
       int_t      *xplore,     /* working array - index to L-structure */
       GlobalLU_t *Glu     /* modified - global LU data structure */
       )
{

    int     jcolp1, jcolm1, jsuper, nsuper;
    int     k, krep, krow, kmark, kperm;
    int     *marker2;        /* Used for small panel LU */
    int     fsupc;        /* First column of a snode */
    int     myfnz;        /* First nonz column of a U-segment */
    int     chperm, chmark, chrep, kchild, kpar, oldrep;
    int_t   xdfs, maxdfs; 
    int_t   jptr, jm1ptr;
    int_t   ito, ifrom;     /* Used to compress row subscripts */
    int_t   mem_error, nextl;
    int     *xsup, *supno;
    int_t   *lsub, *xlsub;
    int_t   nzlmax;
    int     maxsuper;

    xsup    = Glu->xsup;
    supno   = Glu->supno;

    // Initialization
    jcolp1 = jcol + 1;
    jcolm1 = jcol - 1;
    jsuper = EMPTY; // Initialize jsuper to indicate no supernode association with previous column
    nsuper = 0; // Initialize the number of supernodes

    // Marker2 array initialization for small panel LU
    marker2 = marker; 

    // Loop through the rows of the matrix
    for (k = 0; k < m; ++k) {
        // Current row permutation index
        kperm = perm_r[k];
        // Reset marker for the current row
        marker[kperm] = EMPTY;

        // If using compressed L subscripts, compute the offsets
        ito = EMPTY; // Initialize ito for row subscript compression

        // Check for memory error
        mem_error = 0;
    }

    // Calculate maximum number of supernodes
    maxsuper = SUPERLU_MIN(m, jcolp1);

    // Return success
    return 0;
}
    lsub    = Glu->lsub;    # 获取指向全局LU数据结构中L部分非零元素索引数组的指针
    xlsub   = Glu->xlsub;   # 获取指向全局LU数据结构中L部分列指针数组的指针
    nzlmax  = Glu->nzlmax;  # 获取全局LU数据结构中L部分非零元素索引数组的最大长度限制

    maxsuper = sp_ienv(7);  # 调用sp_ienv函数获取特定参数值，用作超节点的最大大小
    jcolp1  = jcol + 1;     # 计算jcol加一的值
    jcolm1  = jcol - 1;     # 计算jcol减一的值
    nsuper  = supno[jcol];  # 获取列jcol所属的超节点号
    jsuper  = nsuper;       # 将当前列jcol所属的超节点号赋给jsuper
    nextl   = xlsub[jcol];  # 获取列jcol在xlsub中的起始位置
    marker2 = &marker[2*m]; # 获取marker数组中的一个偏移位置

    /* 对A[*,jcol]中的每个非零元素执行dfs */
    for (k = 0; lsub_col[k] != EMPTY; k++) {

    krow = lsub_col[k];     # 获取列jcol中的非零元素对应的行索引krow
    lsub_col[k] = EMPTY;    # 清空lsub_col中的当前元素

    kmark = marker2[krow];  # 获取marker2数组中krow位置的标记值

    /* 如果krow之前已经被访问过，则跳过 */
    if ( kmark == jcol ) continue;

    /* 对于jcol的每个未标记的相邻行krow
     *    krow位于L中：将其放入L[*,jcol]的结构中
     */
    marker2[krow] = jcol;   # 将krow在marker2中标记为jcol
    kperm = perm_r[krow];   # 获取krow在perm_r中的置换位置

    if ( kperm == EMPTY ) {
        lsub[nextl++] = krow;    /* 将krow索引插入到A中 */
        if ( nextl >= nzlmax ) {
        if ((mem_error = sLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu)))
            return (mem_error);
        lsub = Glu->lsub;    # 更新lsub的指针
        }
        if ( kmark != jcolm1 ) jsuper = EMPTY;/* 行索引子集测试 */
    } else {
        /* krow is in U: if its supernode-rep krep
         * has been explored, update repfnz[*]
         */
        krep = xsup[supno[kperm]+1] - 1;
        myfnz = repfnz[krep];

        if ( myfnz != EMPTY ) {    /* Visited before */
            if ( myfnz > kperm ) repfnz[krep] = kperm;
            /* continue; */
        }
        else {
            /* Otherwise, perform dfs starting at krep */
            oldrep = EMPTY;
            parent[krep] = oldrep;
            repfnz[krep] = kperm;
            xdfs = xlsub[xsup[supno[krep]]];
            maxdfs = xlsub[krep + 1];

            do {
                /*
                 * For each unmarked kchild of krep
                 */
                while ( xdfs < maxdfs ) {

                    kchild = lsub[xdfs];
                    xdfs++;
                    chmark = marker2[kchild];

                    if ( chmark != jcol ) { /* Not reached yet */
                        marker2[kchild] = jcol;
                        chperm = perm_r[kchild];

                        /* Case kchild is in L: place it in L[*,k] */
                        if ( chperm == EMPTY ) {
                            lsub[nextl++] = kchild;
                            if ( nextl >= nzlmax ) {
                                if ( (mem_error = sLUMemXpand(jcol,nextl,
                                    LSUB,&nzlmax,Glu)) )
                                    return (mem_error);
                                lsub = Glu->lsub;
                            }
                            if ( chmark != jcolm1 ) jsuper = EMPTY;
                        } else {
                            /* Case kchild is in U:
                             * chrep = its supernode-rep. If its rep has
                             * been explored, update its repfnz[*]
                             */
                            chrep = xsup[supno[chperm]+1] - 1;
                            myfnz = repfnz[chrep];
                            if ( myfnz != EMPTY ) { /* Visited before */
                                if ( myfnz > chperm )
                                    repfnz[chrep] = chperm;
                            } else {
                                /* Continue dfs at super-rep of kchild */
                                xplore[krep] = xdfs;
                                oldrep = krep;
                                krep = chrep; /* Go deeper down G(L^t) */
                                parent[krep] = oldrep;
                                repfnz[krep] = chperm;
                                xdfs = xlsub[xsup[supno[krep]]];
                                maxdfs = xlsub[krep + 1];
                            } /* else */
                        } /* else */
                    } /* if */
                } /* while */

                /* krow has no more unexplored nbrs;
                 * place supernode-rep krep in postorder DFS.
                 * backtrack dfs to its parent
                 */
                segrep[*nseg] = krep;
                ++(*nseg);
                kpar = parent[krep]; /* Pop from stack, mimic recursion */
                if ( kpar == EMPTY ) break; /* dfs done */
                krep = kpar;
                xdfs = xplore[krep];
                maxdfs = xlsub[krep + 1];

            } while ( kpar != EMPTY );    /* Until empty stack */

        } /* else */

    } /* else */

} /* for each nonzero ... */

/* Check to see if j belongs in the same supernode as j-1 */
    if ( jcol == 0 ) { /* Do nothing for column 0 */
    nsuper = supno[0] = 0;
    } else {
    fsupc = xsup[nsuper];
    jptr = xlsub[jcol];    /* Not compressed yet */
    jm1ptr = xlsub[jcolm1];

    if ( (nextl-jptr != jptr-jm1ptr-1) ) jsuper = EMPTY;

    /* Always start a new supernode for a singular column */
    if ( nextl == jptr ) jsuper = EMPTY;

    /* Make sure the number of columns in a supernode doesn't
       exceed threshold. */
    if ( jcol - fsupc >= maxsuper ) jsuper = EMPTY;

    /* If jcol starts a new supernode, reclaim storage space in
     * lsub from the previous supernode. Note we only store
     * the subscript set of the first columns of the supernode.
     */
    if ( jsuper == EMPTY ) {    /* starts a new supernode */
        if ( (fsupc < jcolm1) ) { /* >= 2 columns in nsuper */


注释：


    if ( jcol == 0 ) { /* 如果 jcol 等于 0，则不执行任何操作 */
    nsuper = supno[0] = 0;  /* 设置 supernode 的数量为 0，并在 supno 数组中记录 */
    } else {
    fsupc = xsup[nsuper];  /* 获取当前 supernode 的第一个列号 */
    jptr = xlsub[jcol];    /* 获取列 jcol 的 lsub 起始位置（尚未压缩） */
    jm1ptr = xlsub[jcolm1];  /* 获取列 jcolm1 的 lsub 起始位置 */

    if ( (nextl-jptr != jptr-jm1ptr-1) ) jsuper = EMPTY;  /* 检查是否需要新的超级节点 */

    /* 对于单列始终开始一个新的超级节点 */
    if ( nextl == jptr ) jsuper = EMPTY;

    /* 确保超级节点中的列数不超过阈值 */
    if ( jcol - fsupc >= maxsuper ) jsuper = EMPTY;

    /* 如果 jcol 开始一个新的超级节点，则从前一个超级节点的 lsub 中回收存储空间。
     * 注意，我们只存储超级节点的第一列的下标集合。
     */
    if ( jsuper == EMPTY ) {    /* 开始一个新的超级节点 */
        if ( (fsupc < jcolm1) ) { /* nsuper 中有至少 2 列 */
#ifdef CHK_COMPRESS
        printf("  Compress lsub[] at super %d-%d\n", fsupc, jcolm1);
#endif
        // 如果定义了 CHK_COMPRESS 宏，则打印压缩 lsub[] 的信息，显示超级节点的范围
        ito = xlsub[fsupc+1];
        // 将 xlsub[fsupc+1] 的值赋给 ito
        xlsub[jcolm1] = ito;
        // 将 ito 的值存入 xlsub[jcolm1]
        xlsub[jcol] = ito;
        // 将 ito 的值存入 xlsub[jcol]
        for (ifrom = jptr; ifrom < nextl; ++ifrom, ++ito)
            // 遍历 ifrom 从 jptr 到 nextl 的范围，将 lsub[ifrom] 复制给 lsub[ito]
            lsub[ito] = lsub[ifrom];
        // 将 lsub[ifrom] 的内容复制到 lsub[ito]，并递增 ito
        nextl = ito;
        // 更新 nextl 的值为 ito
        }
        // 结束 for 循环

        nsuper++;
        // 增加 nsuper 的值
        supno[jcol] = nsuper;
        // 设置 supno[jcol] 为 nsuper 的值
    } /* if a new supernode */

    }    /* else: jcol > 0 */

    /* Tidy up the pointers before exit */
    // 在退出前清理指针
    xsup[nsuper+1] = jcolp1;
    // 设置 xsup[nsuper+1] 为 jcolp1 的值
    supno[jcolp1]  = nsuper;
    // 设置 supno[jcolp1] 为 nsuper 的值
    xlsub[jcolp1]  = nextl;
    // 设置 xlsub[jcolp1] 为 nextl 的值

    return 0;
    // 返回值为 0
}
// 结束函数
```