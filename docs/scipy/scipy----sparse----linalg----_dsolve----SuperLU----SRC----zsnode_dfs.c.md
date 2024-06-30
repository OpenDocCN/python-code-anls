# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zsnode_dfs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file zsnode_dfs.c
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


#include "slu_zdefs.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *    zsnode_dfs() - Determine the union of the row structures of those 
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
zsnode_dfs (
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
    // 定义变量
    int_t i, k, ifrom, ito, nextl, new_next, nzlmax;
    int   nsuper, krow, kmark;
    int_t mem_error;
    int   *xsup, *supno;
    int_t *lsub, *xlsub;
    
    // 从 GlobalLU_t 结构体中获取必要的数组
    xsup    = Glu->xsup;
    supno   = Glu->supno;
    lsub    = Glu->lsub;
    xlsub   = Glu->xlsub;
    nzlmax  = Glu->nzlmax;

    // 获取下一个可用的超节点号码
    nsuper = ++supno[jcol];    /* Next available supernode number */
    nextl = xlsub[jcol];

    // 遍历从 jcol 到 kcol 的每一列
    for (i = jcol; i <= kcol; i++) {
    /* For each nonzero in A[*,i] */
    // 遍历 A 矩阵中第 i 列中的每一个非零元素
    for (k = xa_begin[i]; k < xa_end[i]; k++) {    
        krow = asub[k];
        kmark = marker[krow];
        // 如果 krow 尚未被访问过
        if ( kmark != kcol ) { /* First time visit krow */
        marker[krow] = kcol;
        lsub[nextl++] = krow;
        // 如果 lsub 数组超出了预分配的内存空间 nzlmax，则进行内存扩展
        if ( nextl >= nzlmax ) {
            mem_error = zLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu);
            if ( mem_error ) return (mem_error);
            lsub = Glu->lsub;
        }
        }
        }
    // 设置第 i 列的超节点号码为 nsuper
    supno[i] = nsuper;
    }
    /* 如果 jcol 小于 kcol，则进行下面的操作 */
    if ( jcol < kcol ) {
        /* 根据现有的 nextl 计算新的 new_next，用于复制子脚标 */
        new_next = nextl + (nextl - xlsub[jcol]);
        
        /* 如果 new_next 超过了 nzlmax，进行内存扩展 */
        while ( new_next > nzlmax ) {
            mem_error = zLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu);
            if ( mem_error ) return (mem_error);
            lsub = Glu->lsub;  /* 更新 lsub 指针 */
        }
        
        ito = nextl;  /* 设置 ito 初始值为 nextl */
        
        /* 复制 xlsub[jcol] 到 nextl 之间的 lsub 元素 */
        for (ifrom = xlsub[jcol]; ifrom < nextl; )
            lsub[ito++] = lsub[ifrom++];  /* 复制 lsub 中的元素 */
        
        /* 设置 xlsub[jcol+1] 到 xlsub[kcol] 之间的值为 nextl */
        for (i = jcol+1; i <= kcol; i++)
            xlsub[i] = nextl;
        
        /* 更新 nextl 的值 */
        nextl = ito;
    }

    /* 更新 xsup[nsuper+1] 的值 */
    xsup[nsuper+1] = kcol + 1;
    
    /* 设置 supno[kcol+1] 的值为 nsuper */
    supno[kcol+1]  = nsuper;
    
    /* 设置 xprune[kcol] 的值为 nextl */
    xprune[kcol]   = nextl;
    
    /* 设置 xlsub[kcol+1] 的值为 nextl */
    xlsub[kcol+1]  = nextl;

    /* 返回操作成功 */
    return 0;
}



# 这行代码表示一个单独的右大括号 '}'，用于闭合之前的某个代码块或函数。
```