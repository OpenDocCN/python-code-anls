# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dsnode_dfs.c`

```
`
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file dsnode_dfs.c
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


#include "slu_ddefs.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *    dsnode_dfs() - Determine the union of the row structures of those 
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
dsnode_dfs (
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
    // 初始化变量和指针
    int_t i, k, ifrom, ito, nextl, new_next, nzlmax;
    int   nsuper, krow, kmark;
    int_t mem_error;
    int   *xsup, *supno;
    int_t *lsub, *xlsub;
    
    // 从全局结构体 Glu 中获取指针和值
    xsup    = Glu->xsup;
    supno   = Glu->supno;
    lsub    = Glu->lsub;
    xlsub   = Glu->xlsub;
    nzlmax  = Glu->nzlmax;

    // 设置下一个可用的超节点编号
    nsuper = ++supno[jcol];    /* Next available supernode number */
    nextl = xlsub[jcol];

    // 遍历从 jcol 到 kcol 的列
    for (i = jcol; i <= kcol; i++) {
        // 遍历第 i 列中的非零元素
        for (k = xa_begin[i]; k < xa_end[i]; k++) {    
            krow = asub[k];
            kmark = marker[krow];
            // 如果 krow 第一次被访问
            if ( kmark != kcol ) {
                marker[krow] = kcol;
                lsub[nextl++] = krow;
                // 如果 lsub 数组超过预分配的最大长度，扩展内存
                if ( nextl >= nzlmax ) {
                    mem_error = dLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu);
                    if ( mem_error ) return (mem_error);
                    lsub = Glu->lsub;
                }
            }
        }
        // 设置 i 列的超节点编号为 nsuper
        supno[i] = nsuper;
    }
    `
        /* 如果 jcol 小于 kcol，则复制子脚本以进行修剪操作 */
        if ( jcol < kcol ) {
            // 计算新的 nextl 值，用于新的子脚本数组
            new_next = nextl + (nextl - xlsub[jcol]);
            // 如果新的 nextl 超出了当前分配的最大空间，进行内存扩展
            while ( new_next > nzlmax ) {
                mem_error = dLUMemXpand(jcol, nextl, LSUB, &nzlmax, Glu);
                if ( mem_error ) return (mem_error);
                // 更新 lsub 指针，指向扩展后的数组
                lsub = Glu->lsub;
            }
            // 将 xlsub[jcol] 到 nextl 之间的元素复制到 ito 开始的位置
            ito = nextl;
            for (ifrom = xlsub[jcol]; ifrom < nextl; )
                lsub[ito++] = lsub[ifrom++];
            
            // 更新 xlsub 数组，确保从 jcol+1 到 kcol 的每个元素都指向 nextl
            for (i = jcol+1; i <= kcol; i++)
                xlsub[i] = nextl;
            
            // 更新 nextl 指向新的子脚本数组的末尾
            nextl = ito;
        }
    
        // 更新超节点结束位置数组
        xsup[nsuper+1] = kcol + 1;
        // 更新超节点编号数组
        supno[kcol+1]  = nsuper;
        // 更新列压缩子脚本中的修剪位置
        xprune[kcol]   = nextl;
        // 更新列起始指针，指向下一个列的起始位置
        xlsub[kcol+1]  = nextl;
    
        // 返回成功标志
        return 0;
}



# 这是一个单独的右花括号 '}'，用于结束某个代码块或函数定义。
```