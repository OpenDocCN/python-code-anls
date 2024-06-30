# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ilu_ccolumn_dfs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ilu_ccolumn_dfs.c
 * \brief Performs a symbolic factorization
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
 *   ILU_CCOLUMN_DFS performs a symbolic factorization on column jcol, and
 *   decide the supernode boundary.
 *
 *   This routine does not use numeric values, but only uses the RHS
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
ilu_ccolumn_dfs(
       const int  m,     /* in - number of rows in the matrix */
       const int  jcol,     /* in - current column being processed */
       int          *perm_r,     /* in - row permutation array */
       int          *nseg,     /* modified - number of segments appended */
       int          *lsub_col, /* in - defines the RHS vector to start the
                    dfs */
       int          *segrep,     /* modified - supernodal representatives */
       int          *repfnz,     /* modified - supernodal first non-zero */
       int          *marker,     /* modified - marker array */
       int          *parent,     /* working array */
       int_t      *xplore,     /* working array */
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

    // Compute column indices for j+1 and j-1
    jcolp1 = jcol + 1;
    jcolm1 = jcol - 1;

    // Determine if jcol belongs to the same supernode as jcolm1
    jsuper = (jcol == 0 || supno[jcolp1] != supno[jcol]) ? EMPTY : nsuper;

    // Initialize marker2 for small panel LU
    marker2 = NULL;

    // Initialize various counters and flags
    fsupc = EMPTY;
    myfnz = EMPTY;
    chperm = EMPTY;
    chmark = EMPTY;
    chrep = EMPTY;
    kchild = EMPTY;
    kpar = EMPTY;
    oldrep = EMPTY;
    xdfs = EMPTY;
    maxdfs = EMPTY;
    jptr = EMPTY;
    jm1ptr = EMPTY;
    ito = EMPTY;
    ifrom = EMPTY;
    mem_error = 0;
    nextl = 0;
    nzlmax = 0;
    maxsuper = 0;

    // Return 0 on success; >0 indicates memory allocation failure
    return 0;
}
    lsub    = Glu->lsub;      # 获取指向全局LU分解数据结构中L部分非零元素行索引数组的指针
    xlsub   = Glu->xlsub;     # 获取指向全局LU分解数据结构中L部分每列起始位置的数组的指针
    nzlmax  = Glu->nzlmax;    # 获取全局LU分解数据结构中L部分非零元素行索引数组的最大长度

    maxsuper = sp_ienv(7);    # 获取超节点最大大小的系统环境设置
    jcolp1  = jcol + 1;       # 计算jcol加一的值
    jcolm1  = jcol - 1;       # 计算jcol减一的值
    nsuper  = supno[jcol];    # 获取jcol列所属的超节点编号
    jsuper  = nsuper;         # 将超节点编号赋值给jsuper
    nextl   = xlsub[jcol];    # 获取L部分第jcol列的起始位置
    marker2 = &marker[2*m];   # 获取marker数组中第2*m位置的指针

    /* For each nonzero in A[*,jcol] do dfs */
    for (k = 0; lsub_col[k] != EMPTY; k++) {  # 遍历列jcol中的每一个非零元素的行索引

    krow = lsub_col[k];         # 获取当前非零元素在列jcol中的行索引
    lsub_col[k] = EMPTY;        # 清空lsub_col数组中的当前位置，表示已处理过

    kmark = marker2[krow];      # 获取marker2中krow行的标记值

    /* krow was visited before, go to the next nonzero */
    if ( kmark == jcol ) continue;  # 如果krow已经被访问过（标记为jcol），则跳过处理

    /* For each unmarked nbr krow of jcol
     *    krow is in L: place it in structure of L[*,jcol]
     */
    marker2[krow] = jcol;       # 将krow标记为已访问过（jcol）

    kperm = perm_r[krow];       # 获取krow行在行置换数组perm_r中的置换值

    if ( kperm == EMPTY ) {
        lsub[nextl++] = krow;    /* krow is indexed into A */
        if ( nextl >= nzlmax ) {
        if ((mem_error = cLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu)))
            return (mem_error);
        lsub = Glu->lsub;       # 如果L部分的行索引数组长度超过了nzlmax，调用扩展内存函数
        }
        if ( kmark != jcolm1 ) jsuper = EMPTY;/* Row index subset testing */
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
                                if ( (mem_error = cLUMemXpand(jcol,nextl,
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

            } while ( kpar != EMPTY ); /* Until empty stack */
        } /* else */
    } /* else */
    /* for each nonzero ... */
    /* Check to see if j belongs in the same supernode as j-1 */


注释：
    // 如果 jcol 等于 0，则不执行任何操作
    if ( jcol == 0 ) { /* Do nothing for column 0 */
        // 设置超节点的数量为 0，并且将 supno[0] 设为 0
        nsuper = supno[0] = 0;
    } else {
        // 将 fsupc 设为 xsup[nsuper]
        fsupc = xsup[nsuper];
        // 将 jptr 设为 xlsub[jcol]，这里 xlsub[jcol] 还未进行压缩
        jptr = xlsub[jcol];    /* Not compressed yet */
        // 将 jm1ptr 设为 xlsub[jcolm1]

        // 如果 (nextl - jptr != jptr - jm1ptr - 1)，则 jsuper 设为 EMPTY
        if ( (nextl-jptr != jptr-jm1ptr-1) ) jsuper = EMPTY;

        // 始终为奇异列启动一个新的超节点
        if ( nextl == jptr ) jsuper = EMPTY;

        // 确保超节点中的列数不超过阈值
        if ( jcol - fsupc >= maxsuper ) jsuper = EMPTY;

        // 如果 jcol 开始一个新的超节点，则从前一个超节点中回收存储空间
        if ( jsuper == EMPTY ) {    /* starts a new supernode */
            // 如果 (fsupc < jcolm1)，则 nsuper 中有 >= 2 列
#ifdef CHK_COMPRESS
        printf("  Compress lsub[] at super %d-%d\n", fsupc, jcolm1);
#endif
        // 获取下一个超节点的起始索引
        ito = xlsub[fsupc+1];
        // 更新 xlsub 数组，将 jcolm1 和 jcol 处的元素设为 ito
        xlsub[jcolm1] = ito;
        xlsub[jcol] = ito;
        // 将 lsub 数组中 jptr 到 nextl 之间的元素复制到 ito 开始的位置
        for (ifrom = jptr; ifrom < nextl; ++ifrom, ++ito)
            lsub[ito] = lsub[ifrom];
        // 更新 nextl 到 ito 的值，表示 lsub 数组的新长度
        nextl = ito;
        // 增加超节点计数
        nsuper++;
        // 设置 supno[jcol] 为当前超节点编号
        supno[jcol] = nsuper;
    } /* if a new supernode */

    }    /* else: jcol > 0 */

    /* Tidy up the pointers before exit */
    // 设置下一个超节点的起始列指针
    xsup[nsuper+1] = jcolp1;
    // 设置 jcolp1 列的超节点编号
    supno[jcolp1]  = nsuper;
    // 设置 jcolp1 列的 xlsub 值为 nextl，标志着该超节点的结束
    xlsub[jcolp1]  = nextl;

    // 返回成功标志
    return 0;
}
```