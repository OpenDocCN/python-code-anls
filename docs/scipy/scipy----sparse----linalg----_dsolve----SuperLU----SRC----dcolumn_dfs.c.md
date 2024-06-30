# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dcolumn_dfs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file dcolumn_dfs.c
 * \brief Performs a symbolic factorization
 *
 * <pre>
 * -- SuperLU routine (version 3.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * October 15, 2003
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

/*! \brief What type of supernodes we want */
#define T2_SUPER


/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   DCOLUMN_DFS performs a symbolic factorization on column jcol, and
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
int DCOLUMN_DFS(
    /* 定义变量 */
    int     jcolp1, jcolm1, jsuper, nsuper;
    int     krep, krow, kmark, kperm;
    int     *marker2;           /* 用于小面板LU的标记 */
    int     fsupc;              /* snode的第一列 */
    int     myfnz;              /* U段的第一个非零列 */
    int     chperm, chmark, chrep, kchild;
    int_t   xdfs, maxdfs, nextl, k;
    int     kpar, oldrep;
    int_t   jptr, jm1ptr;
    int_t   ito, ifrom, istop;  /* 用于压缩行下标 */
    int     *xsup, *supno;
    int_t   *lsub, *xlsub;
    int_t   nzlmax, mem_error;
    int     maxsuper;

    /* 从Glu结构体获取变量 */
    xsup    = Glu->xsup;
    supno   = Glu->supno;
    lsub    = Glu->lsub;
    xlsub   = Glu->xlsub;
    nzlmax  = Glu->nzlmax;

    /* 设置超节点的最大值 */
    maxsuper = sp_ienv(3);
    jcolp1  = jcol + 1;
    jcolm1  = jcol - 1;
    nsuper  = supno[jcol];
    jsuper  = nsuper;
    nextl   = xlsub[jcol];
    marker2 = &marker[2*m];

    /* 对A[*,jcol]中的每个非零元素进行dfs */
    for (k = 0; lsub_col[k] != EMPTY; k++) {
        krow = lsub_col[k];
        lsub_col[k] = EMPTY;
        kmark = marker2[krow];        

        /* 如果krow之前已被访问，则继续下一个非零元素 */
        if (kmark == jcol) continue;

        /* 对于jcol的每个未标记邻居krow，
         * 将其放入L[*,jcol]的结构中 */
        marker2[krow] = jcol;
    } /* else */

} /* for each nonzero ... */

/* 检查j是否属于与j-1相同的超节点 */
if (jcol == 0) { /* 对于第0列什么也不做 */
    nsuper = supno[0] = 0;
} else {
    fsupc = xsup[nsuper];
    jptr = xlsub[jcol];    /* 尚未压缩 */
    jm1ptr = xlsub[jcolm1];

#ifdef T2_SUPER
    /* 如果超节点中列数不等于阈值，则标记为空 */
    if ((nextl - jptr != jptr - jm1ptr - 1)) jsuper = EMPTY;
#endif

    /* 确保超节点中的列数不超过阈值 */
    if (jcol - fsupc >= maxsuper) jsuper = EMPTY;

    /* 如果jcol开始一个新的超节点，从上一个超节点中回收存储空间
     * 注意我们只存储超节点的第一列和最后一列的下标集合
     */
    if (jsuper == EMPTY) {    /* 开始一个新的超节点 */
        if (fsupc < jcolm1 - 1) {    /* nsuper中有至少3列 */
#ifdef CHK_COMPRESS
            printf("  在超节点 %d-%d 处压缩lsub[]\n", fsupc, jcolm1);
#endif
            ito = xlsub[fsupc+1];  // 将 xlsub[fsupc+1] 的值赋给 ito
        xlsub[jcolm1] = ito;  // 将 ito 的值赋给 xlsub[jcolm1]
        istop = ito + jptr - jm1ptr;  // 计算 istop 的值
        xprune[jcolm1] = istop; /* 初始化 xprune[jcol-1] */
        xlsub[jcol] = istop;  // 将 istop 的值赋给 xlsub[jcol]
        for (ifrom = jm1ptr; ifrom < nextl; ++ifrom, ++ito)
            lsub[ito] = lsub[ifrom];  // 将 lsub[ifrom] 的值复制到 lsub[ito]
        nextl = ito;            /* = istop + length(jcol) */  // 更新 nextl 的值为 ito
        }
        nsuper++;  // 增加 nsuper 的值
        supno[jcol] = nsuper;  // 将 nsuper 的值赋给 supno[jcol]
    } /* if a new supernode */

    }    /* else: jcol > 0 */ 
    
    /* Tidy up the pointers before exit */
    xsup[nsuper+1] = jcolp1;  // 将 jcolp1 的值赋给 xsup[nsuper+1]
    supno[jcolp1]  = nsuper;  // 将 nsuper 的值赋给 supno[jcolp1]
    xprune[jcol]   = nextl;    /* 初始化修剪的上界 */
    xlsub[jcolp1]  = nextl;    // 将 nextl 的值赋给 xlsub[jcolp1]

    return 0;  // 返回值为 0，表示函数正常退出
}
```