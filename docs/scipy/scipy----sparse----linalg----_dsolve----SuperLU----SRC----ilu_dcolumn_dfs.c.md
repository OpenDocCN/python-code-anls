# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ilu_dcolumn_dfs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ilu_dcolumn_dfs.c
 * \brief Performs a symbolic factorization
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
 *   ILU_DCOLUMN_DFS performs a symbolic factorization on column jcol, and
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
ilu_dcolumn_dfs(
       const int  m,     /* in - number of rows in the matrix */
       const int  jcol,     /* in */
       int          *perm_r,     /* in */
       int          *nseg,     /* modified - with new segments appended */
       int          *lsub_col, /* in - defines the RHS vector to start the
                    dfs */
       int          *segrep,     /* modified - with new segments appended */
       int          *repfnz,     /* modified */
       int          *marker,     /* modified */
       int          *parent,     /* working array */
       int_t      *xplore,     /* working array */
       GlobalLU_t *Glu     /* modified */
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

    // Initialize variables based on input parameters and Glu structure
    jcolp1 = jcol + 1;
    jcolm1 = jcol - 1;
    jsuper = EMPTY;
    nsuper = 0;
    nzlmax = Glu->nzlmax;
    lsub   = Glu->lsub;
    xlsub  = Glu->xlsub;
    marker2 = Glu->marker2;
    maxsuper = Glu->maxsuper;

    // Loop through the rows and perform DFS to determine supernodal boundaries
    for (k = 0; k < m; k++) {
        // Perform operations to determine supernodal structure
    }

    // Return success code
    return 0;
}
    lsub    = Glu->lsub;  # 获取指向全局 LU 因子 lsub 数组的指针
    xlsub   = Glu->xlsub;  # 获取指向全局 LU 因子 xlsub 数组的指针
    nzlmax  = Glu->nzlmax;  # 获取全局 LU 因子中非零元素列表的最大长度

    maxsuper = sp_ienv(7);  # 获取超节点的最大大小
    jcolp1  = jcol + 1;  # jcol 的下一个列索引
    jcolm1  = jcol - 1;  # jcol 的上一个列索引
    nsuper  = supno[jcol];  # 获取 jcol 所属的超节点编号
    jsuper  = nsuper;  # 将超节点编号赋给 jsuper
    nextl   = xlsub[jcol];  # 获取 xlsub 数组中 jcol 列的起始位置索引
    marker2 = &marker[2*m];  # 获取 marker 数组中第 2*m 位置的指针

    /* 对于 A[*,jcol] 中的每个非零元素执行深度优先搜索 */
    for (k = 0; lsub_col[k] != EMPTY; k++) {

    krow = lsub_col[k];  # 获取 lsub_col 数组中的行索引
    lsub_col[k] = EMPTY;  # 将 lsub_col 中当前位置置为空

    kmark = marker2[krow];  # 获取 marker2 数组中 krow 对应位置的标记值

    /* 如果 krow 已被访问过，则继续处理下一个非零元素 */
    if ( kmark == jcol ) continue;

    /* 对于 jcol 的每个未标记的邻居 krow
     *    如果 krow 在 L 中：将其放入 L[*,jcol] 结构中
     */
    marker2[krow] = jcol;  # 将 krow 在 marker2 中的标记设置为 jcol
    kperm = perm_r[krow];  # 获取 krow 在行排列 perm_r 中的位置

    if ( kperm == EMPTY ) {
        lsub[nextl++] = krow;    /* 将 krow 索引插入到 A 中 */
        if ( nextl >= nzlmax ) {  /* 如果超出了 nzlmax 的限制 */
        if ((mem_error = dLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu)))
            return (mem_error);
        lsub = Glu->lsub;  # 更新 lsub 指针
        }
        if ( kmark != jcolm1 ) jsuper = EMPTY;/* 行索引子集测试 */
    } else {
        /* krow is in U: if its supernode-rep krep
         * has been explored, update repfnz[*]
         */
        // 计算当前超节点的超节点代表的索引
        krep = xsup[supno[kperm]+1] - 1;
        // 获取当前超节点的非零行的起始索引
        myfnz = repfnz[krep];

        // 如果已经访问过
        if ( myfnz != EMPTY ) {    /* Visited before */
            // 如果当前超节点的最后一个访问索引大于当前行索引，则更新为当前行索引
            if ( myfnz > kperm ) repfnz[krep] = kperm;
            /* continue; */
        }
        else {
            /* Otherwise, perform dfs starting at krep */
            // 初始化超节点的父节点为EMPTY
            oldrep = EMPTY;
            // 设置当前超节点的父节点
            parent[krep] = oldrep;
            // 将当前行索引记录为当前超节点的最后访问索引
            repfnz[krep] = kperm;
            // 获取超节点krep的子节点在L的列索引起始位置
            xdfs = xlsub[xsup[supno[krep]]];
            // 获取超节点krep的子节点在L的列索引终止位置
            maxdfs = xlsub[krep + 1];

            // 执行深度优先搜索
            do {
                /*
                 * For each unmarked kchild of krep
                 */
                while ( xdfs < maxdfs ) {

                    // 获取当前子节点在L中的行索引
                    kchild = lsub[xdfs];
                    // 增加子节点在L中的列索引位置
                    xdfs++;
                    // 获取子节点在标记数组中的标记
                    chmark = marker2[kchild];

                    // 如果子节点还未访问过
                    if ( chmark != jcol ) { /* Not reached yet */
                        // 标记子节点已访问
                        marker2[kchild] = jcol;
                        // 获取子节点在列置换数组中的索引
                        chperm = perm_r[kchild];

                        /* Case kchild is in L: place it in L[*,k] */
                        // 如果子节点在L中
                        if ( chperm == EMPTY ) {
                            // 将子节点的行索引放入L的列k中
                            lsub[nextl++] = kchild;
                            // 如果超出了最大长度，进行内存扩展
                            if ( nextl >= nzlmax ) {
                                if ( (mem_error = dLUMemXpand(jcol,nextl,
                                    LSUB,&nzlmax,Glu)) )
                                    return (mem_error);
                                lsub = Glu->lsub;
                            }
                            // 如果标记不等于jcolm1，设置jsuper为EMPTY
                            if ( chmark != jcolm1 ) jsuper = EMPTY;
                        } else {
                            /* Case kchild is in U:
                             *   chrep = its supernode-rep. If its rep has
                             *   been explored, update its repfnz[*]
                             */
                            // 获取子节点的超节点代表的索引
                            chrep = xsup[supno[chperm]+1] - 1;
                            // 获取子节点的非零行的起始索引
                            myfnz = repfnz[chrep];
                            // 如果已经访问过
                            if ( myfnz != EMPTY ) { /* Visited before */
                                // 如果当前超节点的最后一个访问索引大于当前行索引，则更新为当前行索引
                                if ( myfnz > chperm )
                                    repfnz[chrep] = chperm;
                            } else {
                                /* Continue dfs at super-rep of kchild */
                                // 记录当前超节点的索引
                                xplore[krep] = xdfs;
                                // 设置旧的超节点索引
                                oldrep = krep;
                                // 将超节点索引更新为子节点的超节点索引，继续向下搜索
                                krep = chrep; /* Go deeper down G(L^t) */
                                // 设置父节点为旧的超节点
                                parent[krep] = oldrep;
                                // 将当前行索引记录为当前超节点的最后访问索引
                                repfnz[krep] = chperm;
                                // 获取超节点krep的子节点在L的列索引起始位置
                                xdfs = xlsub[xsup[supno[krep]]];
                                // 获取超节点krep的子节点在L的列索引终止位置
                                maxdfs = xlsub[krep + 1];
                            } /* else */

                        } /* else */

                    } /* if */

                } /* while */

                /* krow has no more unexplored nbrs;
                 *      place supernode-rep krep in postorder DFS.
                 *      backtrack dfs to its parent
                 */
                // 将当前超节点索引放入postorder DFS中
                segrep[*nseg] = krep;
                // 增加已处理超节点数量
                ++(*nseg);
                // 获取超节点的父节点索引，模拟递归的出栈操作
                kpar = parent[krep]; /* Pop from stack, mimic recursion */
                // 如果父节点为EMPTY，则说明dfs完成
                if ( kpar == EMPTY ) break; /* dfs done */
                // 更新超节点索引为父节点索引
                krep = kpar;
                // 获取超节点krep的子节点在L的列索引起始位置
                xdfs = xplore[krep];
                // 获取超节点krep的子节点在L的列索引终止位置
                maxdfs = xlsub[krep + 1];

            } while ( kpar != EMPTY );    /* Until empty stack */

        } /* else */

    } /* else */

    } /* for each nonzero ... */

    /* Check to see if j belongs in the same supernode as j-1 */
    // 如果 jcol 等于 0，则不执行任何操作
    if ( jcol == 0 ) { /* Do nothing for column 0 */
    // 初始化超节点数为 0，并设置第一个超节点号为 0
    nsuper = supno[0] = 0;
    } else {
    // 获取当前超节点的第一个非零列的列索引
    fsupc = xsup[nsuper];
    // 获取列 jcol 的第一个非零元素在行索引数组中的位置，尚未压缩
    jptr = xlsub[jcol];    /* Not compressed yet */
    // 获取列 jcol-1 的第一个非零元素在行索引数组中的位置
    jm1ptr = xlsub[jcolm1];

    // 如果 nextl 和 jptr 之间的元素数不等于 jptr 和 jm1ptr 之间的元素数减一，则 jsuper 设为空
    if ( (nextl-jptr != jptr-jm1ptr-1) ) jsuper = EMPTY;

    // 总是为奇异列开始一个新的超节点
    if ( nextl == jptr ) jsuper = EMPTY;

    // 确保超节点中列的数量不超过阈值
    if ( jcol - fsupc >= maxsuper ) jsuper = EMPTY;

    // 如果 jcol 开始一个新的超节点，则从前一个超节点中的 lsub 中回收存储空间
    if ( jsuper == EMPTY ) {    /* starts a new supernode */
        // 如果当前超节点中有超过两列，则回收存储空间
        if ( (fsupc < jcolm1) ) { /* >= 2 columns in nsuper */
#ifdef CHK_COMPRESS
        printf("  Compress lsub[] at super %d-%d\n", fsupc, jcolm1);
#endif
        // 检查是否定义了 CHK_COMPRESS 宏，如果定义则打印压缩 lsub[] 的信息
        ito = xlsub[fsupc+1];
        // 获取 xlsub 数组中 fsupc+1 处的值，存入 ito 变量
        xlsub[jcolm1] = ito;
        // 将 ito 的值存入 xlsub 数组的 jcolm1 位置
        xlsub[jcol] = ito;
        // 将 ito 的值存入 xlsub 数组的 jcol 位置
        for (ifrom = jptr; ifrom < nextl; ++ifrom, ++ito)
            // 循环从 jptr 到 nextl-1，将 lsub[ifrom] 的值复制给 lsub[ito]
            lsub[ito] = lsub[ifrom];
        // 更新 nextl 的值为 ito，完成 lsub 数组的压缩操作
        nextl = ito;
        // 更新 nextl 的值为 ito
        }
        // 增加 nsuper 的值，表示超节点的数量增加了一个
        nsuper++;
        // 将 nsuper 的值存入 supno[jcol]，表示 jcol 列的超节点编号
        supno[jcol] = nsuper;
    } /* if a new supernode */

    }    /* else: jcol > 0 */

    /* Tidy up the pointers before exit */
    // 在退出之前整理指针
    xsup[nsuper+1] = jcolp1;
    // 设置 xsup[nsuper+1] 的值为 jcolp1
    supno[jcolp1]  = nsuper;
    // 将 nsuper 的值存入 supno[jcolp1]
    xlsub[jcolp1]  = nextl;
    // 将 nextl 的值存入 xlsub[jcolp1]

    return 0;
    // 返回整数值 0，表示函数执行成功
}
// 函数结束
```