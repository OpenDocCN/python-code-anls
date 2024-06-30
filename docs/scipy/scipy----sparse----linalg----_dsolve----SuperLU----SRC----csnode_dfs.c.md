# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\csnode_dfs.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file csnode_dfs.c
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


#include "slu_cdefs.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *    csnode_dfs() - Determine the union of the row structures of those 
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
csnode_dfs (
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
    // 声明变量
    int_t i, k, ifrom, ito, nextl, new_next, nzlmax;
    int   nsuper, krow, kmark;
    int_t mem_error;
    int   *xsup, *supno;
    int_t *lsub, *xlsub;
    
    // 从全局结构体 Glu 中获取相关指针
    xsup    = Glu->xsup;    // 超节点起始列的数组
    supno   = Glu->supno;   // 超节点编号数组
    lsub    = Glu->lsub;    // 非零元素的行号数组
    xlsub   = Glu->xlsub;   // lsub 数组的起始指针
    nzlmax  = Glu->nzlmax;  // lsub 数组的最大长度

    nsuper = ++supno[jcol];    /* Next available supernode number */
    nextl = xlsub[jcol];

    // 遍历从 jcol 到 kcol 的列
    for (i = jcol; i <= kcol; i++) {
        // 对于每列 i 中的非零元素
        for (k = xa_begin[i]; k < xa_end[i]; k++) {    
            krow = asub[k];     // 获取非零元素的行号
            kmark = marker[krow];
            if ( kmark != kcol ) { /* First time visit krow */
                marker[krow] = kcol;
                lsub[nextl++] = krow;   // 将行号 krow 添加到 lsub 数组中
                if ( nextl >= nzlmax ) {
                    // 扩展 lsub 数组的长度
                    mem_error = cLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu);
                    if ( mem_error ) return (mem_error);
                    lsub = Glu->lsub;   // 更新 lsub 的指针
                }
            }
        }
        supno[i] = nsuper;   // 设置超节点编号
    }
}
    /* 如果 jcol 小于 kcol，则复制子脚本以进行修剪 */
    if ( jcol < kcol ) {
        /* 计算新的下一个元素的位置 */
        new_next = nextl + (nextl - xlsub[jcol]);
        /* 如果新位置超出了预分配的最大空间，则扩展内存 */
        while ( new_next > nzlmax ) {
            mem_error = cLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu);
            if ( mem_error ) return (mem_error);
            lsub = Glu->lsub;
        }
        ito = nextl;
        /* 复制 lsub 数组中的数据 */
        for (ifrom = xlsub[jcol]; ifrom < nextl; )
            lsub[ito++] = lsub[ifrom++];
        /* 更新 xlsub 数组 */
        for (i = jcol+1; i <= kcol; i++)
            xlsub[i] = nextl;
        /* 更新下一个元素的位置 */
        nextl = ito;
    }

    /* 更新超节点的起始列 */
    xsup[nsuper+1] = kcol + 1;
    /* 记录超节点号 */
    supno[kcol+1]  = nsuper;
    /* 记录列 kcol 的修剪起始位置 */
    xprune[kcol]   = nextl;
    /* 更新 xlsub 数组中列 kcol+1 的起始位置 */
    xlsub[kcol+1]  = nextl;

    /* 返回成功状态 */
    return 0;
}



# 这行代码表示一个代码块的结束，与一个以 '{' 开始的代码块相对应
```