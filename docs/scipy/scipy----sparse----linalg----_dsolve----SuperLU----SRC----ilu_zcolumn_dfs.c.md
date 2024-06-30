# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ilu_zcolumn_dfs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ilu_zcolumn_dfs.c
 * \brief Performs a symbolic factorization
 *
 * <pre>
 * -- SuperLU routine (version 4.0) --
 * Lawrence Berkeley National Laboratory
 * June 30, 2009
 * </pre>
*/

#include "slu_zdefs.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   ILU_ZCOLUMN_DFS performs a symbolic factorization on column jcol, and
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
ilu_zcolumn_dfs(
       const int  m,     /* in - number of rows in the matrix */
       const int  jcol,     /* in - column index for symbolic factorization */
       int          *perm_r,     /* in - row permutation vector */
       int          *nseg,     /* modified - number of segments appended */
       int          *lsub_col, /* in - defines the RHS vector to start the dfs */
       int          *segrep,     /* modified - list of supernodal representatives */
       int          *repfnz,     /* modified - supernodal entry location */
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

    // Initialization of variables
    jcolp1 = jcol + 1;
    jcolm1 = jcol - 1;

    // Determine the supernode structure and boundaries for column jcol
    jsuper = EMPTY; // Initially, assume column jcol does not belong to any supernode
    nsuper = 0;     // Number of supernodes

    // Allocate and initialize marker2 array for small panel LU
    marker2 = intCalloc(m);

    // Main symbolic factorization logic using Depth-First Search (DFS)
    for (k = 0; k < *nseg; ++k) {
        // Perform operations for each segment in U[*,jcol]
        // ...
    }

    // Free allocated memory for marker2
    SUPERLU_FREE(marker2);

    // Return 0 on success
    return 0;
}
    lsub    = Glu->lsub;  # 将Glu结构体中的lsub成员赋给lsub变量，lsub是一个指向L的非零元素的数组
    xlsub   = Glu->xlsub;  # 将Glu结构体中的xlsub成员赋给xlsub变量，xlsub是一个指向L各列起始位置的数组
    nzlmax  = Glu->nzlmax;  # 将Glu结构体中的nzlmax成员赋给nzlmax变量，nzlmax表示L的非零元素数组的最大长度

    maxsuper = sp_ienv(7);  # 调用sp_ienv函数获取第7个环境参数的值，赋给maxsuper变量
    jcolp1  = jcol + 1;  # jcolp1变量等于jcol加1
    jcolm1  = jcol - 1;  # jcolm1变量等于jcol减1
    nsuper  = supno[jcol];  # 将supno数组中索引为jcol的值赋给nsuper变量
    jsuper  = nsuper;  # 将nsuper变量的值赋给jsuper变量
    nextl   = xlsub[jcol];  # 将xlsub数组中索引为jcol的值赋给nextl变量，表示L中列jcol的起始位置
    marker2 = &marker[2*m];  # 将marker数组中第2*m个元素的地址赋给marker2变量，用于标记操作

    /* For each nonzero in A[*,jcol] do dfs */
    for (k = 0; lsub_col[k] != EMPTY; k++) {  # 遍历lsub_col数组，直到遇到空值（EMPTY）为止
        krow = lsub_col[k];  # 将lsub_col数组中索引为k的值赋给krow变量
        lsub_col[k] = EMPTY;  # 将lsub_col数组中索引为k的位置置为空值（EMPTY）
        kmark = marker2[krow];  # 将marker2中索引为krow的值赋给kmark变量，标记krow的状态

        /* krow was visited before, go to the next nonzero */
        if ( kmark == jcol ) continue;  # 如果krow之前已经访问过（标记为jcol），则跳过本次循环

        /* For each unmarked nbr krow of jcol
         *    krow is in L: place it in structure of L[*,jcol]
         */
        marker2[krow] = jcol;  # 将krow在marker2中标记为jcol，表示krow已经访问过
        kperm = perm_r[krow];  # 将perm_r数组中索引为krow的值赋给kperm变量

        if ( kperm == EMPTY ) {
            lsub[nextl++] = krow;    /* krow is indexed into A */
            if ( nextl >= nzlmax ) {  # 如果nextl超过了nzlmax的值
                if ((mem_error = zLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu)))
                    return (mem_error);  # 扩展L的存储空间失败，返回错误
                lsub = Glu->lsub;  # 更新lsub指向扩展后的L的非零元素数组
            }
            if ( kmark != jcolm1 ) jsuper = EMPTY;  /* Row index subset testing */
        }
    } else {
        /* krow is in U: if its supernode-rep krep
         * has been explored, update repfnz[*]
         */
        krep = xsup[supno[kperm]+1] - 1;  // 获取当前行对应的超节点代表的索引
        myfnz = repfnz[krep];  // 获取超节点代表的行首非零元素的索引

        if ( myfnz != EMPTY ) {    // 如果已经访问过
            if ( myfnz > kperm ) repfnz[krep] = kperm;  // 更新行首非零元素的索引
            /* continue; */
        }
        else {
            /* Otherwise, perform dfs starting at krep */
            oldrep = EMPTY;  // 初始化旧超节点代表索引为空
            parent[krep] = oldrep;  // 设置当前超节点的父节点为旧超节点代表索引
            repfnz[krep] = kperm;  // 设置当前超节点代表的行首非零元素的索引为当前行索引
            xdfs = xlsub[xsup[supno[krep]]];  // 获取超节点在L的列索引起始位置
            maxdfs = xlsub[krep + 1];  // 获取超节点在L的列索引结束位置

            do {
                /*
                 * For each unmarked kchild of krep
                 */
                while ( xdfs < maxdfs ) {

                    kchild = lsub[xdfs];  // 获取未标记的子节点的列索引
                    xdfs++;
                    chmark = marker2[kchild];  // 获取子节点的标记

                    if ( chmark != jcol ) { /* Not reached yet */
                        marker2[kchild] = jcol;  // 标记子节点已被访问
                        chperm = perm_r[kchild];  // 获取子节点在排列中的索引

                        /* Case kchild is in L: place it in L[*,k] */
                        if ( chperm == EMPTY ) {
                            lsub[nextl++] = kchild;  // 将子节点放入L的列索引中
                            if ( nextl >= nzlmax ) {
                                if ( (mem_error = zLUMemXpand(jcol,nextl,
                                    LSUB,&nzlmax,Glu)) )
                                    return (mem_error);  // 如果内存不足则扩展并返回错误
                                lsub = Glu->lsub;  // 更新L的列索引
                            }
                            if ( chmark != jcolm1 ) jsuper = EMPTY;  // 如果标记不等于jcolm1则设置jsuper为空
                        } else {
                            /* Case kchild is in U:
                             *   chrep = its supernode-rep. If its rep has
                             *   been explored, update its repfnz[*]
                             */
                            chrep = xsup[supno[chperm]+1] - 1;  // 获取子节点的超节点代表索引
                            myfnz = repfnz[chrep];  // 获取子节点超节点代表的行首非零元素索引
                            if ( myfnz != EMPTY ) { /* Visited before */
                                if ( myfnz > chperm )
                                    repfnz[chrep] = chperm;  // 更新子节点超节点代表的行首非零元素索引
                            } else {
                                /* Continue dfs at super-rep of kchild */
                                xplore[krep] = xdfs;  // 设置当前超节点的探索索引
                                oldrep = krep;  // 保存当前超节点索引
                                krep = chrep; /* Go deeper down G(L^t) */  // 移动到子节点的超节点代表索引
                                parent[krep] = oldrep;  // 设置子节点的父节点为当前超节点索引
                                repfnz[krep] = chperm;  // 设置子节点超节点代表的行首非零元素索引
                                xdfs = xlsub[xsup[supno[krep]]];  // 获取新的超节点在L的列索引起始位置
                                maxdfs = xlsub[krep + 1];  // 获取新的超节点在L的列索引结束位置
                            } /* else */

                        } /* else */

                    } /* if */

                } /* while */

                /* krow has no more unexplored nbrs;
                 *      place supernode-rep krep in postorder DFS.
                 *      backtrack dfs to its parent
                 */
                segrep[*nseg] = krep;  // 将超节点代表索引添加到后序DFS中
                ++(*nseg);  // 递增分段的数量
                kpar = parent[krep]; /* Pop from stack, mimic recursion */  // 从栈中弹出超节点索引，模拟递归
                if ( kpar == EMPTY ) break; /* dfs done */  // 如果父节点索引为空，则dfs结束
                krep = kpar;  // 设置当前超节点索引为父节点索引
                xdfs = xplore[krep];  // 获取当前超节点的探索索引
                maxdfs = xlsub[krep + 1];  // 获取当前超节点在L的列索引结束位置

            } while ( kpar != EMPTY );    /* Until empty stack */  // 直到栈为空

        } /* else */

    } /* else */

    } /* for each nonzero ... */

    /* Check to see if j belongs in the same supernode as j-1 */
    // 如果 jcol 等于 0，则什么都不做（针对第一列）
    if ( jcol == 0 ) { /* Do nothing for column 0 */
        // 设置超节点数量为 0，超节点编号列表的第一个元素为 0
        nsuper = supno[0] = 0;
    } else {
        // 当前列的第一个非零元素的列指标
        fsupc = xsup[nsuper];
        // 当前列的非零元素在行索引数组中的起始位置（未压缩）
        jptr = xlsub[jcol];    /* Not compressed yet */
        // 当前列前一列的非零元素在行索引数组中的起始位置（未压缩）
        jm1ptr = xlsub[jcolm1];

        // 如果当前列与上一列之间的非零元素数量不相等，则不属于同一个超节点
        if ( (nextl-jptr != jptr-jm1ptr-1) ) jsuper = EMPTY;

        // 总是为奇异列开始一个新的超节点
        if ( nextl == jptr ) jsuper = EMPTY;

        // 确保超节点中列的数量不超过最大阈值
        if ( jcol - fsupc >= maxsuper ) jsuper = EMPTY;

        // 如果 jcol 开始一个新的超节点，回收前一个超节点在 lsub 中的存储空间，
        // 注意我们只存储超节点中第一列的下标集合。
        if ( jsuper == EMPTY ) {    /* starts a new supernode */
            // 如果上一个超节点中至少包含两列，则回收存储空间
            if ( (fsupc < jcolm1) ) { /* >= 2 columns in nsuper */
                // ...
            }
        }
    }
#ifdef CHK_COMPRESS
        printf("  Compress lsub[] at super %d-%d\n", fsupc, jcolm1);
#endif
        // 如果定义了 CHK_COMPRESS 宏，则打印压缩 lsub[] 的信息，显示超级节点的范围
        ito = xlsub[fsupc+1];
        // 记录 xlsub 数组中 fsupc+1 处的值到 ito
        xlsub[jcolm1] = ito;
        // 将 ito 的值存入 xlsub[jcolm1] 中
        xlsub[jcol] = ito;
        // 将 ito 的值存入 xlsub[jcol] 中
        for (ifrom = jptr; ifrom < nextl; ++ifrom, ++ito)
            // 循环将 lsub[ifrom] 中的值复制到 lsub[ito] 中，同时更新 ito
            lsub[ito] = lsub[ifrom];
        // 更新 nextl 的值为 ito
        nextl = ito;
        // 结束循环后，将 ito 的值赋给 nextl
        }
        // 增加超级节点计数器
        nsuper++;
        // 将 nsuper 赋给 supno[jcol]
        supno[jcol] = nsuper;
    } /* if a new supernode */

    }    /* else: jcol > 0 */

    /* Tidy up the pointers before exit */
    // 在退出之前整理指针
    xsup[nsuper+1] = jcolp1;
    // 设置 xsup[nsuper+1] 为 jcolp1
    supno[jcolp1]  = nsuper;
    // 设置 supno[jcolp1] 为 nsuper
    xlsub[jcolp1]  = nextl;
    // 设置 xlsub[jcolp1] 为 nextl

    return 0;
    // 返回 0 表示函数执行成功结束
}
```