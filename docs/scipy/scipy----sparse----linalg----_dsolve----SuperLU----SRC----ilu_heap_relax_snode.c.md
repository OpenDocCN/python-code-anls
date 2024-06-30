# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ilu_heap_relax_snode.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file ilu_heap_relax_snode.c
 * \brief Identify the initial relaxed supernodes
 *
 * <pre>
 * -- SuperLU routine (version 4.0) --
 * Lawrence Berkeley National Laboratory
 * June 1, 2009
 * </pre>
 */

#include "slu_ddefs.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *    ilu_heap_relax_snode() - Identify the initial relaxed supernodes,
 *    assuming that the matrix has been reordered according to the postorder
 *    of the etree.
 * </pre>
 */

void
ilu_heap_relax_snode (
         const     int n,
         int       *et,          /* column elimination tree */
         const int relax_columns, /* max no of columns allowed in a
                     relaxed snode */
         int       *descendants,  /* no of descendants of each node
                     in the etree */
         int       *relax_end,    /* last column in a supernode
                       * if j-th column starts a relaxed
                       * supernode, relax_end[j] represents
                       * the last column of this supernode */
         int       *relax_fsupc   /* first column in a supernode
                       * relax_fsupc[j] represents the first
                       * column of j-th supernode */
         )
{
    register int i, j, k, l, f, parent;
    register int snode_start;    /* beginning of a snode */
    int *et_save, *post, *inv_post, *iwork;
#if ( PRNTlevel>=1 )
    int nsuper_et = 0, nsuper_et_post = 0;
#endif

    /* The etree may not be postordered, but is heap ordered. */

    // 分配空间以保存临时数据
    iwork = (int*) intMalloc(3*n+2);
    if ( !iwork ) ABORT("SUPERLU_MALLOC fails for iwork[]");

    // 设置 iwork 中的偏移量
    inv_post = iwork + n+1;
    et_save = inv_post + n+1;

    /* Post order etree */
    // 计算树的后序遍历顺序
    post = (int *) TreePostorder(n, et);

    // 根据后序遍历的结果更新逆序数组
    for (i = 0; i < n+1; ++i) inv_post[post[i]] = i;

    // 保存原始的列消除树
    for (i = 0; i < n; ++i) {
        iwork[post[i]] = post[et[i]];
        et_save[i] = et[i];
    }

    // 更新列消除树为后序遍历的结果
    for (i = 0; i < n; ++i) et[i] = iwork[i];

    /* Compute the number of descendants of each node in the etree */
    // 计算每个节点在树中的后代数量
    ifill (relax_end, n, EMPTY);
    ifill (relax_fsupc, n, EMPTY);
    for (j = 0; j < n; j++) descendants[j] = 0;
    for (j = 0; j < n; j++) {
        parent = et[j];
        if ( parent != n )  /* not the dummy root */
            descendants[parent] += descendants[j] + 1;
    }

    /* Identify the relaxed supernodes by postorder traversal of the etree. */
    // 根据树的后序遍历识别放松的超级节点
    for ( f = j = 0; j < n; ) {
        parent = et[j];
        snode_start = j;
        while ( parent != n && descendants[parent] < relax_columns ) {
            j = parent;
            parent = et[j];
        }
        // 更新超级节点的起始和结束列
        // 如果 j 列开始了一个放松的超级节点，则 relax_end[j] 表示此超级节点的最后一列
        if ( snode_start < j ) {
            relax_fsupc[f] = snode_start;
            relax_end[f] = j - 1;
            f++;
        }
        j++;
    }

    // 释放临时分配的空间
    SUPERLU_FREE(post);
    SUPERLU_FREE(iwork);
}
    # 在后序遍历的树中找到一个超级节点；j 是最后一列。
#if ( PRNTlevel>=1 )
    // 如果 PRNTlevel 大于等于 1，则增加 nsuper_et_post 计数
    ++nsuper_et_post;
#endif
    k = n;  // 初始化 k 为 n
    for (i = snode_start; i <= j; ++i)
        k = SUPERLU_MIN(k, inv_post[i]);  // 遍历 snode_start 到 j 的范围，更新 k 为 inv_post[i] 和当前 k 的最小值
    l = inv_post[j];  // 将 inv_post[j] 赋给 l
    if ( (l - k) == (j - snode_start) ) {
        /* It's also a supernode in the original etree */
        // 如果 l - k 等于 j - snode_start，则当前节点也在原始 etree 中是一个超节点
        relax_end[k] = l;        // 记录最后一列
        relax_fsupc[f++] = k;    // 将 k 添加到 relax_fsupc 数组中，并增加 f 的值
#if ( PRNTlevel>=1 )
        // 如果 PRNTlevel 大于等于 1，则增加 nsuper_et 计数
        ++nsuper_et;
#endif
    } else {
        // 否则处理非超节点情况
        for (i = snode_start; i <= j; ++i) {
            l = inv_post[i];  // 将 inv_post[i] 赋给 l
            if ( descendants[i] == 0 ) {
                // 如果 descendants[i] 等于 0，表示当前节点是一个叶子节点
                relax_end[l] = l;    // 记录叶子节点
                relax_fsupc[f++] = l;    // 将 l 添加到 relax_fsupc 数组中，并增加 f 的值
#if ( PRNTlevel>=1 )
                // 如果 PRNTlevel 大于等于 1，则增加 nsuper_et 计数
                ++nsuper_et;
#endif
            }
        }
    }
    j++;  // j 值增加 1，指向下一个节点
    /* Search for a new leaf */
    // 寻找一个新的叶子节点
    while ( descendants[j] != 0 && j < n ) j++;  // 向后遍历 descendants 数组，直到找到一个 descendants[j] 为 0 或者 j 达到 n

#if ( PRNTlevel>=1 )
    // 如果 PRNTlevel 大于等于 1，则输出调试信息
    printf(".. heap_snode_relax:\n"
       "\tNo of relaxed snodes in postordered etree:\t%d\n"
       "\tNo of relaxed snodes in original etree:\t%d\n",
       nsuper_et_post, nsuper_et);
#endif

    /* Recover the original etree */
    // 恢复原始的 etree
    for (i = 0; i < n; ++i) et[i] = et_save[i];  // 将 et_save 数组的值复制回 et 数组

    SUPERLU_FREE(post);  // 释放 post 数组的内存
    SUPERLU_FREE(iwork);  // 释放 iwork 数组的内存
}
```