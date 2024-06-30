# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\sp_coletree.c`

```
/*!
 * \file
 * Copyright (c) 2003, The Regents of the University of California, through
 * Lawrence Berkeley National Laboratory (subject to receipt of any required 
 * approvals from U.S. Dept. of Energy) 
 * 
 * All rights reserved. 
 * 
 * The source code is distributed under BSD license, see the file License.txt
 * at the top-level directory.
 */

/*! @file sp_coletree.c
 * \brief Tree layout and computation routines
 *
 *<pre>
 * -- SuperLU routine (version 3.1) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * August 1, 2008
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

/*  Elimination tree computation and layout routines */

#include <stdio.h>
#include <stdlib.h>
#include "slu_ddefs.h"

/* 
 *  Implementation of disjoint set union routines.
 *  Elements are integers in 0..n-1, and the 
 *  names of the sets themselves are of type int.
 *  
 *  Calls are:
 *  initialize_disjoint_sets (n) initial call.
 *  s = make_set (i)             returns a set containing only i.
 *  s = link (t, u)         returns s = t union u, destroying t and u.
 *  s = find (i)         return name of set containing i.
 *  finalize_disjoint_sets      final call.
 *
 *  This implementation uses path compression but not weighted union.
 *  See Tarjan's book for details.
 *  John Gilbert, CMI, 1987.
 *
 *  Implemented path-halving by XSL 07/05/95.
 */

/* Allocate memory for an array of integers initialized to zero */
static 
int *mxCallocInt(int n)
{
    register int i;
    int *buf;

    buf = (int *) SUPERLU_MALLOC( n * sizeof(int) );  // Allocate memory using SUPERLU_MALLOC
    if ( !buf ) {                                    // Check if allocation fails
         ABORT("SUPERLU_MALLOC fails for buf in mxCallocInt()");  // Print error message and abort if failed
       }
    for (i = 0; i < n; i++) buf[i] = 0;               // Initialize all elements of buf to zero
    return (buf);                                    // Return the allocated and initialized buffer
}
      
/* Initialize disjoint sets */
static
void initialize_disjoint_sets (
                   int n,       // Number of elements in the set
                   int **pp     // Pointer to the array of sets
                   )
{
    (*pp) = mxCallocInt(n);     // Allocate memory for the set array and assign to *pp
}

/* Create a singleton set containing element i */
static
int make_set (
          int i,       // Element to create a set for
          int *pp      // Pointer to the array of sets
          )
{
    pp[i] = i;          // Set element i as its own representative in the set
    return i;           // Return the index of the set created
}

/* Union two sets s and t */
static
int link (
      int s,        // Element in first set
      int t,        // Element in second set
      int *pp       // Pointer to the array of sets
      )
{
    pp[s] = t;      // Union sets s and t by making s point to t
    return t;       // Return the representative of the unioned set
}

/* Find the representative of the set containing element i */
/* PATH HALVING */
static
int find (
      int i,        // Element to find the representative set for
      int *pp       // Pointer to the array of sets
      )
{
    register int p, gp;
    
    p = pp[i];      // Start at element i
    gp = pp[p];     // Grandparent of p
    while (gp != p) {   // While p is not its own parent
        pp[i] = gp;     // Path halving: make every other node point to its grandparent
        i = gp;         // Move to grandparent
        p = pp[i];      // Update parent
        gp = pp[p];     // Update grandparent
    }
    return (p);     // Return the representative of the set containing element i
}

/* Finalize and free memory used by the disjoint sets */
static
void finalize_disjoint_sets (
                 int *pp    // Pointer to the array of sets
                 )
{
    // This function would typically free memory allocated for the sets,
    // but the implementation is not shown here.
}
    # 释放 SUPERLU 中分配的内存，参数 pp 是需要释放的指针或数据结构
    SUPERLU_FREE(pp);
/*
 *      Find the elimination tree for A'*A.
 *      This uses something similar to Liu's algorithm. 
 *      It runs in time O(nz(A)*log n) and does not form A'*A.
 *
 *      Input:
 *        Sparse matrix A.  Numeric values are ignored, so any
 *        explicit zeros are treated as nonzero.
 *      Output:
 *        Integer array of parents representing the elimination
 *        tree of the symbolic product A'*A.  Each vertex is a
 *        column of A, and nc means a root of the elimination forest.
 *
 *      John R. Gilbert, Xerox, 10 Dec 1990
 *      Based on code by JRG dated 1987, 1988, and 1990.
 */

/*
 * Nonsymmetric elimination tree
 */
int
sp_coletree(
        int_t *acolst, int_t *acolend, /* column start and end past 1 */
        int_t *arow,                 /* row indices of A */
        int nr, int nc,            /* dimension of A */
        int *parent                   /* parent in elim tree */
        )
{
    int    *root;            /* root of subtee of etree     */
    int     *firstcol;        /* first nonzero col in each row*/
    int    rset, cset;             
    int    row, col;
    int    rroot;
    int    p;
    int     *pp;

    root = mxCallocInt (nc); // 分配大小为 nc 的整数数组，用于存储子树的根
    initialize_disjoint_sets (nc, &pp); // 初始化并查集，pp 用于表示各个集合的父节点指针数组

    /* Compute firstcol[row] = first nonzero column in row */

    firstcol = mxCallocInt (nr); // 分配大小为 nr 的整数数组，用于存储每行的第一个非零列的索引
    for (row = 0; row < nr; firstcol[row++] = nc); // 初始化 firstcol 数组，将所有元素置为 nc（表示未找到非零列）
    for (col = 0; col < nc; col++) 
        for (p = acolst[col]; p < acolend[col]; p++) {
            row = arow[p];
            firstcol[row] = SUPERLU_MIN(firstcol[row], col); // 更新每行的第一个非零列索引
        }

    /* Compute etree by Liu's algorithm for symmetric matrices,
           except use (firstcol[r],c) in place of an edge (r,c) of A.
       Thus each row clique in A'*A is replaced by a star
       centered at its first vertex, which has the same fill. */

    for (col = 0; col < nc; col++) {
        cset = make_set (col, pp); // 创建一个集合，包含节点 col，并更新 pp 中的父节点信息
        root[cset] = col; // 将 cset 对应的根节点设置为 col
        parent[col] = nc; /* Matlab */ // 将 col 的父节点设置为 nc（Matlab 中的约定）
        for (p = acolst[col]; p < acolend[col]; p++) {
            row = firstcol[arow[p]];
            if (row >= col) continue; // 如果行索引大于等于列索引，则跳过（对称性矩阵性质）
            rset = find (row, pp); // 查找包含 row 的集合，并返回其根的索引
            rroot = root[rset]; // 获取集合 rset 的根节点
            if (rroot != col) {
                parent[rroot] = col; // 设置 rroot 的父节点为 col
                cset = link (cset, rset, pp); // 将 cset 和 rset 合并，并更新 pp 中的父节点信息
                root[cset] = col; // 更新合并后集合的根节点为 col
            }
        }
    }

    SUPERLU_FREE (root); // 释放 root 数组的内存
    SUPERLU_FREE (firstcol); // 释放 firstcol 数组的内存
    finalize_disjoint_sets (pp); // 结束并查集操作，释放 pp 相关内存
    return 0; // 返回操作成功
}
/*
 *  q = TreePostorder (n, p);
 *
 *    Postorder a tree.
 *    Input:
 *      p is a vector of parent pointers for a forest whose
 *        vertices are the integers 0 to n-1; p[root]==n.
 *    Output:
 *      q is a vector indexed by 0..n-1 such that q[i] is the
 *      i-th vertex in a postorder numbering of the tree.
 *
 *        ( 2/7/95 modified by X.Li:
 *          q is a vector indexed by 0:n-1 such that vertex i is the
 *          q[i]-th vertex in a postorder numbering of the tree.
 *          That is, this is the inverse of the previous q. )
 *
 *    In the child structure, lower-numbered children are represented
 *    first, so that a tree which is already numbered in postorder
 *    will not have its order changed.
 *    
 *  Written by John Gilbert, Xerox, 10 Dec 1990.
 *  Based on code written by John Gilbert at CMI in 1987.
 */

#if 0  // replaced by a non-recursive version 
static
/*
 * Depth-first search from vertex v.
 */
void etdfs (
        int      v,
        int   first_kid[],
        int   next_kid[],
        int   post[], 
        int   *postnum
        )
{
    int    w;

    for (w = first_kid[v]; w != -1; w = next_kid[w]) {
        etdfs (w, first_kid, next_kid, post, postnum);
    }
    /* post[postnum++] = v; in Matlab */
    post[v] = (*postnum)++;    /* Modified by X. Li on 08/10/07 */
}
#endif

static
/*
 * Depth-first search from vertex n.  No recursion.
 * This routine was contributed by Cédric Doucet, CEDRAT Group, Meylan, France.
 */
void nr_etdfs (int n, int *parent,
           int *first_kid, int *next_kid,
           int *post, int postnum)
{
    int current = n, first, next;

    while (postnum != n){
     
        /* no kid for the current node */
        first = first_kid[current];

        /* no first kid for the current node */
        if (first == -1){

            /* numbering this node because it has no kid */
            post[current] = postnum++;

            /* looking for the next kid */
            next = next_kid[current];

            while (next == -1){

                /* no more kids : back to the parent node */
                current = parent[current];

                /* numbering the parent node */
                post[current] = postnum++;

                /* get the next kid */
                next = next_kid[current];
        }
            
            /* stopping criterion */
            if (postnum==n+1) return;

            /* updating current node */
            current = next;
        }
        /* updating current node */
        else {
            current = first;
        }
    }
}

/*
 * Post order a tree
 */
int *TreePostorder(
           int n,
           int *parent
           )
{
        int    *first_kid, *next_kid;    /* Linked list of children.    */
        int    *post, postnum;
    int    v, dad;

    /* Allocate storage for working arrays and results    */
    first_kid =     mxCallocInt (n+1);    // 分配数组以保存每个节点的第一个孩子节点索引
    next_kid  =     mxCallocInt (n+1);    // 分配数组以保存每个节点的下一个兄弟节点索引

    // ...
    # 使用 mxCallocInt 分配空间给 post 数组，大小为 n+1
    post = mxCallocInt(n+1);

    # 设置描述子节点结构的数据结构
    for (v = 0; v <= n; first_kid[v++] = -1);

    # 逆序遍历节点，设置每个节点的父节点和子节点关系
    for (v = n-1; v >= 0; v--) {
        dad = parent[v];            # 获取节点 v 的父节点
        next_kid[v] = first_kid[dad];   # 将节点 v 插入到其父节点的子节点链表头部
        first_kid[dad] = v;         # 更新父节点的子节点链表头部为节点 v
    }

    # 初始化深度优先搜索的后序遍历起始序号为 0
    postnum = 0;
#if 0
    /* 递归调用 */
    etdfs (n, first_kid, next_kid, post, &postnum);
#else
    /* 非递归调用 */
    nr_etdfs(n, parent, first_kid, next_kid, post, postnum);
#endif

// 释放内存并返回 post 数组
SUPERLU_FREE (first_kid);
SUPERLU_FREE (next_kid);
return post;
}

/*
 *      p = spsymetree (A);
 *
 *      查找对称矩阵 A 的消除树（elimination tree）。
 *      使用 Liu 算法，时间复杂度为 O(nz*log n)。
 *
 *      输入：
 *        方阵稀疏矩阵 A。不检查对称性；忽略对角线以下的元素。
 *        数值被忽略，所以显式零被视为非零。
 *      输出：
 *        代表消除树的父节点的整数数组，其中 n 表示消除森林的根。
 *      注意：
 *        此例程仅使用上三角，而稀疏 Cholesky（如 spchol.c 中所示）仅使用下三角。
 *        Matlab 的稠密 Cholesky 仅使用上三角。可以通过转置矩阵或使用行遍历
 *        辅助指针和链接数组来修改此例程以使用下三角。
 *
 *      John R. Gilbert, Xerox, 1990年12月10日
 *      基于 JRG 在1987年、1988年和1990年的代码。
 *      由 X.S. Li 修改，1999年11月。
 */

/*
 * 对称消除树
 */
int
sp_symetree(
        int *acolst, int *acolend, /* 列开始和结束位置（超过1） */
        int *arow,            /* A 的行索引 */
        int n,                /* A 的维度 */
        int *parent        /* 消除树中的父节点 */
        )
{
    int    *root;            /* etree 子树的根节点 */
    int    rset, cset;             
    int    row, col;
    int    rroot;
    int    p;
    int     *pp;

    root = mxCallocInt (n);  // 分配存储空间以保存根节点
    initialize_disjoint_sets (n, &pp);  // 初始化并查集

    for (col = 0; col < n; col++) {
        cset = make_set (col, pp);  // 创建包含列 col 的集合
        root[cset] = col;  // 设置集合的根节点为 col
        parent[col] = n; /* Matlab */  // 初始化父节点为 n（Matlab）
        for (p = acolst[col]; p < acolend[col]; p++) {
            row = arow[p];  // 获取 A 中的行索引
            if (row >= col) continue;  // 忽略对角线及以上的元素
            rset = find (row, pp);  // 查找行索引所在的集合
            rroot = root[rset];  // 获取集合的根节点
            if (rroot != col) {
                parent[rroot] = col;  // 设置 rroot 的父节点为 col
                cset = link (cset, rset, pp);  // 合并两个集合
                root[cset] = col;  // 更新合并后的集合的根节点为 col
            }
        }
    }
    SUPERLU_FREE (root);  // 释放根节点数组的内存
    finalize_disjoint_sets (pp);  // 最终化并查集
    return 0;
} /* SP_SYMETREE */
```