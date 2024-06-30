# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\heap_relax_snode.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file heap_relax_snode.c
 * \brief Identify the initial relaxed supernodes
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

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *    relax_snode() - Identify the initial relaxed supernodes, assuming that 
 *    the matrix has been reordered according to the postorder of the etree.
 * </pre>
 */ 

void
heap_relax_snode (
         const     int n,
         int       *et,           /* column elimination tree */
         const int relax_columns, /* max no of columns allowed in a
                     relaxed snode */
         int       *descendants,  /* no of descendants of each node
                     in the etree */
         int       *relax_end     /* last column in a supernode */
         )
{
    register int i, j, k, l, parent;
    register int snode_start;    /* beginning of a snode */
    int *et_save, *post, *inv_post, *iwork;
#if ( PRNTlevel>=1 )
    int nsuper_et = 0, nsuper_et_post = 0;
#endif

    /* The etree may not be postordered, but is heap ordered. */

    // 分配内存空间给 iwork 数组
    iwork = (int*) intMalloc(3*n+2); 
    if ( !iwork ) ABORT("SUPERLU_MALLOC fails for iwork[]");
    // 指定 iwork 中的不同部分的指针
    inv_post = iwork + n+1;
    et_save = inv_post + n+1;

    /* Post order etree */
    // 计算后序遍历的 etree
    post = (int *) TreePostorder(n, et);
    // 根据后序遍历结果，建立逆后序的映射关系
    for (i = 0; i < n+1; ++i) inv_post[post[i]] = i;

    /* Renumber etree in postorder */
    // 根据后序遍历结果重新编号 etree
    for (i = 0; i < n; ++i) {
        iwork[post[i]] = post[et[i]];
        et_save[i] = et[i]; /* Save the original etree */
    }
    // 更新原始 etree 的编号
    for (i = 0; i < n; ++i) et[i] = iwork[i];

    /* Compute the number of descendants of each node in the etree */
    // 计算每个节点在 etree 中的后代数量
    ifill (relax_end, n, EMPTY);
    for (j = 0; j < n; j++) descendants[j] = 0;
    for (j = 0; j < n; j++) {
        parent = et[j];
        if ( parent != n )  /* not the dummy root */
            descendants[parent] += descendants[j] + 1;
    }

    /* Identify the relaxed supernodes by postorder traversal of the etree. */
    for (j = 0; j < n; ) { 
        # 使用 j 作为循环变量，从 0 开始递增，直到 n
        parent = et[j];
        # 获取当前节点 j 在 etree 中的父节点索引
        snode_start = j;
        # 记录当前超节点的起始列号为 j
        while ( parent != n && descendants[parent] < relax_columns ) {
            # 在后序遍历的 etree 中找到一个超节点；j 是最后一列的索引
            j = parent;
            parent = et[j];
        }
        /* Found a supernode in postordered etree; j is the last column. */
        # 在后序遍历的 etree 中找到一个超节点；j 是最后一列的索引
`
#if ( PRNTlevel>=1 )
    // 如果 PRNTlevel 大于等于 1，则增加 nsuper_et_post 的计数
    ++nsuper_et_post;
#endif

    // 将 k 设置为 n
    k = n;

    // 遍历 snode_start 到 j 之间的索引 i
    for (i = snode_start; i <= j; ++i)
        // 更新 k 为 k 和 inv_post[i] 中的较小值
        k = SUPERLU_MIN(k, inv_post[i]);

    // 将 l 设置为 inv_post[j]
    l = inv_post[j];

    // 如果 (l - k) 等于 (j - snode_start)
    if ( (l - k) == (j - snode_start) ) {
        /* It's also a supernode in the original etree */
        // 在 relax_end 中记录最后一列
        relax_end[k] = l;
#if ( PRNTlevel>=1 )
        // 如果 PRNTlevel 大于等于 1，则增加 nsuper_et 的计数
        ++nsuper_et;
#endif
    } else {
        // 否则，对于 snode_start 到 j 之间的每个索引 i
        for (i = snode_start; i <= j; ++i) {
            // 将 l 设置为 inv_post[i]
            l = inv_post[i];
            // 如果 descendants[i] 等于 0
            if ( descendants[i] == 0 ) {
                // 在 relax_end 中记录 l
                relax_end[l] = l;
#if ( PRNTlevel>=1 )
                // 如果 PRNTlevel 大于等于 1，则增加 nsuper_et 的计数
                ++nsuper_et;
#endif
            }
        }
    }

    // 将 j 自增
    j++;

    // 查找新的叶子节点
    while ( descendants[j] != 0 && j < n ) j++;

}

#if ( PRNTlevel>=1 )
    // 如果 PRNTlevel 大于等于 1，则输出以下信息
    printf(".. heap_snode_relax:\n"
       "\tNo of relaxed snodes in postordered etree:\t%d\n"
       "\tNo of relaxed snodes in original etree:\t%d\n",
       nsuper_et_post, nsuper_et);
#endif

// 恢复原始的 etree
for (i = 0; i < n; ++i) et[i] = et_save[i];

// 释放内存
SUPERLU_FREE(post);
SUPERLU_FREE(iwork);
```