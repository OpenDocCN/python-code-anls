# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ssnode_dfs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ssnode_dfs.c
 * \brief Determines the union of row structures of columns within the relaxed node
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


#include "slu_sdefs.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *    ssnode_dfs() - Determine the union of the row structures of those 
 *    columns within the relaxed snode.
 *    Note: The relaxed snodes are leaves of the supernodal etree, therefore, 
 *    the portion outside the rectangular supernode must be zero.
 *
 * Return value
 * ============
 *     0   success;
 *    >0   number of bytes allocated when run out of memory.
 * </pre>
 */

int_t
ssnode_dfs (
       const int   jcol,        /* in - start of the supernode */
       const int   kcol,         /* in - end of the supernode */
       const int_t *asub,        /* in */
       const int_t *xa_begin,    /* in */
       const int_t *xa_end,      /* in */
       int_t      *xprune,      /* out */
       int        *marker,      /* modified */
       GlobalLU_t *Glu          /* modified */
       )
{
    int_t i, k, ifrom, ito, nextl, new_next, nzlmax;
    int   nsuper, krow, kmark;
    int_t mem_error;
    int   *xsup, *supno;
    int_t *lsub, *xlsub;
    
    xsup    = Glu->xsup;    /* Pointer to supernode boundary array */
    supno   = Glu->supno;   /* Pointer to supernode number array */
    lsub    = Glu->lsub;    /* Pointer to L-subscript array */
    xlsub   = Glu->xlsub;   /* Pointer to L-subscript starting position array */
    nzlmax  = Glu->nzlmax;  /* Maximum number of nonzeros in L-subscript */
    
    nsuper = ++supno[jcol];    /* Next available supernode number */
    nextl = xlsub[jcol];    /* Starting position in L-subscript for current column */

    for (i = jcol; i <= kcol; i++) {    /* Loop over all columns in the supernode */
        /* For each nonzero in A[*,i] */
        for (k = xa_begin[i]; k < xa_end[i]; k++) {    
            krow = asub[k];    /* Row index of the current nonzero */
            kmark = marker[krow];    /* Marker value for current row */
            if ( kmark != kcol ) { /* First time visit krow */
                marker[krow] = kcol;    /* Update marker to current column */
                lsub[nextl++] = krow;    /* Store krow in L-subscript */
                if ( nextl >= nzlmax ) {    /* Check if L-subscript needs expansion */
                    mem_error = sLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu);    /* Attempt to expand memory */
                    if ( mem_error ) return (mem_error);    /* Return if memory expansion fails */
                    lsub = Glu->lsub;    /* Update lsub pointer after potential reallocation */
                }
            }
        }
        supno[i] = nsuper;    /* Set supernode number for column i */
    }
    /* 如果 jcol < kcol，表示当前超节点包含超过一个列，需要复制剪枝用的子脚标 */
    if ( jcol < kcol ) {
        // 计算新的 nextl 值，用于扩展 lsub 数组
        new_next = nextl + (nextl - xlsub[jcol]);
        // 如果新的 nextl 超出了 nzlmax，扩展内存
        while ( new_next > nzlmax ) {
            mem_error = sLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu);
            if ( mem_error ) return (mem_error);
            lsub = Glu->lsub;
        }
        // 复制 xlsub[jcol] 到 nextl 之间的元素到 lsub 数组中
        ito = nextl;
        for (ifrom = xlsub[jcol]; ifrom < nextl; )
            lsub[ito++] = lsub[ifrom++];
        // 更新 xlsub 中从 jcol+1 到 kcol 的值为 nextl
        for (i = jcol+1; i <= kcol; i++) xlsub[i] = nextl;
        // 更新 nextl 的值为 ito
        nextl = ito;
    }

    // 更新超节点结束位置的索引
    xsup[nsuper+1] = kcol + 1;
    // 设置超节点编号
    supno[kcol+1]  = nsuper;
    // 记录 kcol 列的剪枝起始位置
    xprune[kcol]   = nextl;
    // 更新 xlsub 中 kcol+1 的值为 nextl，表示下一个超节点的起始位置
    xlsub[kcol+1]  = nextl;

    // 返回正常退出状态
    return 0;
}



# 这行代码关闭了一个代码块。在大多数编程语言中，这表示结束了一个函数、循环或条件语句的定义。
```