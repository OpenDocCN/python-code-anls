# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\mmd.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

#include "superlu_config.h"

typedef int_t shortint;

/* *************************************************************** */
/* *************************************************************** */
/* ****     GENMMD ..... MULTIPLE MINIMUM EXTERNAL DEGREE     **** */
/* *************************************************************** */
/* *************************************************************** */

/*     AUTHOR - JOSEPH W.H. LIU */
/*              DEPT OF COMPUTER SCIENCE, YORK UNIVERSITY. */

/*     PURPOSE - THIS ROUTINE IMPLEMENTS THE MINIMUM DEGREE */
/*        ALGORITHM.  IT MAKES USE OF THE IMPLICIT REPRESENTATION */
/*        OF ELIMINATION GRAPHS BY QUOTIENT GRAPHS, AND THE */
/*        NOTION OF INDISTINGUISHABLE NODES.  IT ALSO IMPLEMENTS */
/*        THE MODIFICATIONS BY MULTIPLE ELIMINATION AND MINIMUM */
/*        EXTERNAL DEGREE. */
/*        --------------------------------------------- */
/*        CAUTION - THE ADJACENCY VECTOR ADJNCY WILL BE */
/*        DESTROYED. */
/*        --------------------------------------------- */

/*     INPUT PARAMETERS - */
/*        NEQNS  - NUMBER OF EQUATIONS. */
/*        (XADJ,ADJNCY) - THE ADJACENCY STRUCTURE. */
/*        DELTA  - TOLERANCE VALUE FOR MULTIPLE ELIMINATION. */
/*        MAXINT - MAXIMUM MACHINE REPRESENTABLE (SHORT) INTEGER */
/*                 (ANY SMALLER ESTIMATE WILL DO) FOR MARKING */
/*                 NODES. */

/*     OUTPUT PARAMETERS - */
/*        PERM   - THE MINIMUM DEGREE ORDERING. */
/*        INVP   - THE INVERSE OF PERM. */
/*        NOFSUB - AN UPPER BOUND ON THE NUMBER OF NONZERO */
/*                 SUBSCRIPTS FOR THE COMPRESSED STORAGE SCHEME. */

/*     WORKING PARAMETERS - */
/*        DHEAD  - VECTOR FOR HEAD OF DEGREE LISTS. */
/*        INVP   - USED TEMPORARILY FOR DEGREE FORWARD LINK. */
/*        PERM   - USED TEMPORARILY FOR DEGREE BACKWARD LINK. */
/*        QSIZE  - VECTOR FOR SIZE OF SUPERNODES. */
/*        LLIST  - VECTOR FOR TEMPORARY LINKED LISTS. */
/*        MARKER - A TEMPORARY MARKER VECTOR. */

/*     PROGRAM SUBROUTINES - */
/*        MMDELM, MMDINT, MMDNUM, MMDUPD. */

/* *************************************************************** */

/* Subroutine */ int genmmd_(int *neqns, int_t *xadj, shortint *adjncy, 
    int *invp, int *perm, int_t *delta, shortint *dhead, 
    shortint *qsize, shortint *llist, shortint *marker, int_t *maxint, 
    int_t *nofsub)
{
    /* System generated locals */
    int i__1;

    /* Local variables */
    int_t mdeg, ehead, i, mdlmt, mdnode;
    extern /* Subroutine */

    /* This subroutine implements the GENMMD algorithm for computing the minimum degree ordering of a graph. */

    /* Initialize local variables */
    /* mdeg: current minimum degree found
       ehead: head of the element list
       i, mdlmt, mdnode: loop variables and limits */

    /* Perform initializations */

    /* Main loop for computing minimum degree ordering */

    /* Return statement indicating successful completion */
}
    # 声明四个函数：slu_mmdelm_, slu_mmdupd_, slu_mmdint_, slu_mmdnum_
    # 这些函数用于执行某些与稀疏矩阵相关的最小度顺序法(Minimum Degree Ordering)的操作
    # 这些函数的参数和返回值类型分别为：
    # - slu_mmdelm_: 接受多个指针参数，并返回一个整型值
    # - slu_mmdupd_: 接受多个指针参数，并返回一个整型值
    # - slu_mmdint_: 接受多个指针参数，并返回一个整型值
    # - slu_mmdnum_: 接受三个指针参数，并返回一个整型值

    int_t nextmd, tag, num;
    # 声明三个变量：
    # - nextmd: int_t 类型，用于存储下一个最小度节点的标识
    # - tag: int_t 类型，用于标记或辅助计算中使用的标签
    # - num: int_t 类型，可能用于计数或者其他数值存储
/* *************************************************************** */
/* *************************************************************** */

    /* 参数调整 */
    --marker;  // marker 是一个数组，降低其指针以匹配实际数据的索引
    --llist;   // llist 是一个数组，降低其指针以匹配实际数据的索引
    --qsize;   // qsize 是一个数组，降低其指针以匹配实际数据的索引
    --dhead;   // dhead 是一个数组，降低其指针以匹配实际数据的索引
    --perm;    // perm 是一个数组，降低其指针以匹配实际数据的索引
    --invp;    // invp 是一个数组，降低其指针以匹配实际数据的索引
    --adjncy;  // adjncy 是一个数组，降低其指针以匹配实际数据的索引
    --xadj;    // xadj 是一个数组，降低其指针以匹配实际数据的索引

    /* 函数体 */
    if (*neqns <= 0) {  // 如果输入的节点数小于等于0，则直接返回0
    return 0;
    }

/*        ------------------------------------------------ */
/*        MINIMUM DEGREE 算法的初始化。 */
/*        ------------------------------------------------ */
    *nofsub = 0;  // 将 nofsub 初始化为0
    slu_mmdint_(neqns, &xadj[1], &adjncy[1], &dhead[1], &invp[1], &perm[1], &
        qsize[1], &llist[1], &marker[1]);  // 调用 slu_mmdint_ 函数进行初始化

/*        ---------------------------------------------- */
/*        NUM 计数有序节点的数量加1。 */
/*        ---------------------------------------------- */
    num = 1;  // 初始化 num 为1

/*        ----------------------------- */
/*        消除所有孤立节点。 */
/*        ----------------------------- */
    nextmd = dhead[1];  // 设置初始的 nextmd 为 dhead[1]
L100:
    if (nextmd <= 0) {  // 如果 nextmd 小于等于0，则跳转到 L200
    goto L200;
    }
    mdnode = nextmd;  // 将 nextmd 赋值给 mdnode
    nextmd = invp[mdnode];  // 更新 nextmd 为 mdnode 的逆
    marker[mdnode] = *maxint;  // 将 marker[mdnode] 设置为 maxint
    invp[mdnode] = -num;  // 将 invp[mdnode] 设置为 -num
    ++num;  // num 自增
    goto L100;  // 跳转到 L100 继续执行

L200:
/*        ---------------------------------------- */
/*        寻找最小度节点。 */
/*        ---------------------------------------- */
    if (num > *neqns) {  // 如果 num 大于节点数，则跳转到 L1000
    goto L1000;
    }
    tag = 1;  // 初始化 tag 为1
    dhead[1] = 0;  // 将 dhead[1] 设置为0
    mdeg = 2;  // 初始化 mdeg 为2
L300:
    if (dhead[mdeg] > 0) {  // 如果 dhead[mdeg] 大于0，则跳转到 L400
    goto L400;
    }
    ++mdeg;  // mdeg 自增
    goto L300;  // 跳转到 L300 继续执行
L400:
/*            ------------------------------------------------- */
/*            使用 DELTA 的值设置 MDLMT，用于更新度的时机。 */
/*            ------------------------------------------------- */
    mdlmt = mdeg + *delta;  // 计算 mdlmt 的值
    ehead = 0;  // 将 ehead 设置为0

L500:
    mdnode = dhead[mdeg];  // 将 dhead[mdeg] 赋值给 mdnode
    if (mdnode > 0) {  // 如果 mdnode 大于0，则跳转到 L600
    goto L600;
    }
    ++mdeg;  // mdeg 自增
    if (mdeg > mdlmt) {  // 如果 mdeg 大于 mdlmt，则跳转到 L900
    goto L900;
    }
    goto L500;  // 跳转到 L500 继续执行
L600:
/*                ---------------------------------------- */
/*                从度结构中移除 MDNODE。 */
/*                ---------------------------------------- */
    nextmd = invp[mdnode];  // 将 mdnode 的逆赋值给 nextmd
    dhead[mdeg] = nextmd;  // 更新 dhead[mdeg] 为 nextmd
    if (nextmd > 0) {  // 如果 nextmd 大于0，则将 perm[nextmd] 设置为 -mdeg
    perm[nextmd] = -mdeg;
    }
    invp[mdnode] = -num;  // 将 invp[mdnode] 设置为 -num
    *nofsub = *nofsub + mdeg + qsize[mdnode] - 2;  // 更新 nofsub 的值
    if (num + qsize[mdnode] > *neqns) {  // 如果 num 加上 qsize[mdnode] 大于节点数，则跳转到 L1000
    goto L1000;
    }
/*                ---------------------------------------------- */
/*                消除 MDNODE 并执行商图转换。必要时重置 TAG 值。 */
/*                ---------------------------------------------- */
    ++tag;  // tag 自增
    if (tag < *maxint) {  // 如果 tag 小于 maxint，则跳转到 L800
    goto L800;
    }
    tag = 1;  // 否则重置 tag 为1
    i__1 = *neqns;  // 设置循环上限为 neqns
    for (i = 1; i <= i__1; ++i) {  // 循环遍历节点数
    if (marker[i] < *maxint) {  // 如果 marker[i] 小于 maxint，则将 marker[i] 设置为0
        marker[i] = 0;
    }
/* L700: */
    }
L800:
    slu_mmdelm_(&mdnode, &xadj[1], &adjncy[1], &dhead[1], &invp[1], &perm[1], &
        qsize[1], &llist[1], &marker[1], maxint, &tag);
    num += qsize[mdnode];  // 将mdnode节点的qsize加到num中
    llist[mdnode] = ehead;  // 将mdnode节点的llist更新为ehead
    ehead = mdnode;  // 更新ehead为mdnode
    if (*delta >= 0) {  // 如果delta大于等于0
    goto L500;  // 跳转到标号L500处继续执行
    }
L900:
/*            ------------------------------------------- */
/*            UPDATE DEGREES OF THE NODES INVOLVED IN THE */
/*            MINIMUM DEGREE NODES ELIMINATION. */
/*            ------------------------------------------- */
    if (num > *neqns) {  // 如果num大于neqns
    goto L1000;  // 跳转到标号L1000处执行
    }
    slu_mmdupd_(&ehead, neqns, &xadj[1], &adjncy[1], delta, &mdeg, &dhead[1], &
        invp[1], &perm[1], &qsize[1], &llist[1], &marker[1], maxint, &tag)
        ;  // 调用更新节点度数的子程序
    goto L300;  // 执行完毕后跳转到标号L300处

L1000:
    slu_mmdnum_(neqns, &perm[1], &invp[1], &qsize[1]);  // 调用统计结果的子程序
    return 0;  // 返回0表示成功结束

} /* genmmd_ */

/* *************************************************************** */
/* *************************************************************** */
/* ***     MMDINT ..... MULT MINIMUM DEGREE INITIALIZATION     *** */
/* *************************************************************** */
/* *************************************************************** */

/*     AUTHOR - JOSEPH W.H. LIU */
/*              DEPT OF COMPUTER SCIENCE, YORK UNIVERSITY. */

/*     PURPOSE - THIS ROUTINE PERFORMS INITIALIZATION FOR THE */
/*        MULTIPLE ELIMINATION VERSION OF THE MINIMUM DEGREE */
/*        ALGORITHM. */

/*     INPUT PARAMETERS - */
/*        NEQNS  - NUMBER OF EQUATIONS. */
/*        (XADJ,ADJNCY) - ADJACENCY STRUCTURE. */

/*     OUTPUT PARAMETERS - */
/*        (DHEAD,DFORW,DBAKW) - DEGREE DOUBLY LINKED STRUCTURE. */
/*        QSIZE  - SIZE OF SUPERNODE (INITIALIZED TO ONE). */
/*        LLIST  - LINKED LIST. */
/*        MARKER - MARKER VECTOR. */

/* *************************************************************** */

/* Subroutine */ int slu_mmdint_(int *neqns, int_t *xadj, shortint *adjncy, 
    shortint *dhead, int *dforw, int *dbakw, shortint *qsize, 
    shortint *llist, shortint *marker)
{
    /* System generated locals */
    int i__1;

    /* Local variables */
    int ndeg, node, fnode;

/* *************************************************************** */

    /* Parameter adjustments */
    --marker;
    --llist;
    --qsize;
    --dbakw;
    --dforw;
    --dhead;
    --adjncy;
    --xadj;

    /* Function Body */
    i__1 = *neqns;
    for (node = 1; node <= i__1; ++node) {
    dhead[node] = 0;  // 初始化节点的度头指针为0
    qsize[node] = 1;  // 初始化节点的超节点大小为1
    marker[node] = 0;  // 初始化节点的标记为0
    llist[node] = 0;  // 初始化节点的链接列表为0
/* L100: */
    }
/*        ------------------------------------------ */
/*        INITIALIZE THE DEGREE DOUBLY LINKED LISTS. */
/*        ------------------------------------------ */
    i__1 = *neqns;
    for (node = 1; node <= i__1; ++node) {
    ndeg = xadj[node + 1] - xadj[node] + 1;  // 计算节点node的度数
    fnode = dhead[ndeg];  // 取出度为ndeg的链表头节点
    dforw[node] = fnode;  // 设置节点node的前向指针为fnode
    dhead[ndeg] = node;  // 更新度为ndeg的链表头节点为node
    if (fnode > 0) {
        dbakw[fnode] = node;  // 如果fnode大于0，则更新其后向指针为node
    }
    dbakw[node] = -ndeg;  // 设置节点node的后向指针为-ndeg
/* L200: */
    }
    return 0;  // 返回0表示成功结束
}
} /* slu_mmdint_ */

/* *************************************************************** */
/* *************************************************************** */
/* **     MMDELM ..... MULTIPLE MINIMUM DEGREE ELIMINATION     *** */
/* *************************************************************** */
/* *************************************************************** */

/*     AUTHOR - JOSEPH W.H. LIU */
/*              DEPT OF COMPUTER SCIENCE, YORK UNIVERSITY. */

/*     PURPOSE - THIS ROUTINE ELIMINATES THE NODE MDNODE OF */
/*        MINIMUM DEGREE FROM THE ADJACENCY STRUCTURE, WHICH */
/*        IS STORED IN THE QUOTIENT GRAPH FORMAT.  IT ALSO */
/*        TRANSFORMS THE QUOTIENT GRAPH REPRESENTATION OF THE */
/*        ELIMINATION GRAPH. */

/*     INPUT PARAMETERS - */
/*        MDNODE - NODE OF MINIMUM DEGREE. */
/*        MAXINT - ESTIMATE OF MAXIMUM REPRESENTABLE (SHORT) */
/*                 INT. */
/*        TAG    - TAG VALUE. */

/*     UPDATED PARAMETERS - */
/*        (XADJ,ADJNCY) - UPDATED ADJACENCY STRUCTURE. */
/*        (DHEAD,DFORW,DBAKW) - DEGREE DOUBLY LINKED STRUCTURE. */
/*        QSIZE  - SIZE OF SUPERNODE. */
/*        MARKER - MARKER VECTOR. */
/*        LLIST  - TEMPORARY LINKED LIST OF ELIMINATED NABORS. */

/* *************************************************************** */

/* Subroutine */ int slu_mmdelm_(int_t *mdnode, int_t *xadj, shortint *adjncy,
     shortint *dhead, int *dforw, int *dbakw, shortint *qsize, 
    shortint *llist, shortint *marker, int_t *maxint, int_t *tag)
{
    /* System generated locals */
    int_t i__1, i__2;

    /* Local variables */
    int_t node, link, rloc, rlmt, i, j, nabor, rnode, elmnt, xqnbr, 
    istop, jstop, istrt, jstrt, nxnode, pvnode, nqnbrs, npv;


/* *************************************************************** */


/* *************************************************************** */

/*        ----------------------------------------------- */
/*        FIND REACHABLE SET AND PLACE IN DATA STRUCTURE. */
/*        ----------------------------------------------- */
    /* Parameter adjustments */
    --marker;
    --llist;
    --qsize;
    --dbakw;
    --dforw;
    --dhead;
    --adjncy;
    --xadj;

    /* Function Body */
    marker[*mdnode] = *tag;  /* 设置 MDNODE 节点的标记为给定的 TAG 值 */

    istrt = xadj[*mdnode];  /* 获取 MDNODE 节点在 xadj 数组中的起始位置 */
    istop = xadj[*mdnode + 1] - 1;  /* 获取 MDNODE 节点在 xadj 数组中的结束位置 */

/*        ------------------------------------------------------- */
/*        ELMNT POINTS TO THE BEGINNING OF THE LIST OF ELIMINATED */
/*        NABORS OF MDNODE, AND RLOC GIVES THE STORAGE LOCATION */
/*        FOR THE NEXT REACHABLE NODE. */
/*        ------------------------------------------------------- */
    elmnt = 0;  /* 初始化 elmnt 变量，用于记录已消除邻居的起始位置 */
    rloc = istrt;  /* 初始化 rloc 变量为 MDNODE 节点在 adjncy 数组中的起始位置 */
    rlmt = istop;  /* 初始化 rlmt 变量为 MDNODE 节点在 adjncy 数组中的结束位置 */
    i__1 = istop;
    for (i = istrt; i <= i__1; ++i) {
        nabor = adjncy[i];  /* 获取当前邻居节点的编号 */
        if (nabor == 0) {  /* 如果邻居节点为0，跳转到标签 L300 */
            goto L300;
        }
        if (marker[nabor] >= *tag) {  /* 如果邻居节点的标记大于等于给定的 TAG 值，跳转到标签 L200 */
            goto L200;
        }
        marker[nabor] = *tag;  /* 将邻居节点的标记设置为给定的 TAG 值 */
        if (dforw[nabor] < 0) {  /* 如果邻居节点在 dforw 数组中的值小于0，跳转到标签 L100 */
            goto L100;
        }
        adjncy[rloc] = nabor;
    # 增加变量 rloc 的值，使其加一
    ++rloc;
    # 转移到标签 L200 处继续执行
    goto L200;
L100:
    llist[nabor] = elmnt;
    // 将 elmnt 插入到 llist[nabor] 处
    elmnt = nabor;
    // 更新 elmnt 为 nabor

L200:
    ;
    // 空语句，什么也不做
    }

L300:
/*            ----------------------------------------------------- */
/*            MERGE WITH REACHABLE NODES FROM GENERALIZED ELEMENTS. */
/*            ----------------------------------------------------- */
    // 如果 elmnt 小于等于 0，则跳转到 L1000 标签处
    if (elmnt <= 0) {
        goto L1000;
    }
    // 将 -elmnt 存入 adjncy[rlmt]，并将 link 设为 elmnt
    adjncy[rlmt] = -elmnt;
    link = elmnt;

L400:
    // 设置 jstrt 为 xadj[link]，jstop 为 xadj[link + 1] - 1
    jstrt = xadj[link];
    jstop = xadj[link + 1] - 1;
    // 对于 j 从 jstrt 到 jstop 的循环
    i__1 = jstop;
    for (j = jstrt; j <= i__1; ++j) {
        // 将 node 设为 adjncy[j]
        node = adjncy[j];
        // 将 link 设为 -node
        link = -node;
        // 如果 node 小于 0，则跳转到 L400 标签处
        if (node < 0) {
            goto L400;
        } else if (node == 0) {
            // 如果 node 等于 0，则跳转到 L900 标签处
            goto L900;
        } else {
            // 否则，跳转到 L500 标签处
            goto L500;
        }

L500:
        // 如果 marker[node] 大于等于 *tag 或者 dforw[node] 小于 0，则跳转到 L800 标签处
        if (marker[node] >= *tag || dforw[node] < 0) {
            goto L800;
        }
        // 将 marker[node] 设为 *tag
        marker[node] = *tag;
/*                            --------------------------------- */
/*                            USE STORAGE FROM ELIMINATED NODES */
/*                            IF NECESSARY. */
/*                            --------------------------------- */
L600:
        // 如果 rloc 小于 rlmt，则跳转到 L700 标签处
        if (rloc < rlmt) {
            goto L700;
        }
        // 将 link 设为 -adjncy[rlmt]，rloc 设为 xadj[link]，rlmt 设为 xadj[link + 1] - 1
        link = -adjncy[rlmt];
        rloc = xadj[link];
        rlmt = xadj[link + 1] - 1;
        // 再次跳转到 L600 标签处
        goto L600;

L700:
        // 将 adjncy[rloc] 设为 node，rloc 加一
        adjncy[rloc] = node;
        ++rloc;
L800:
        ;
        // 空语句，什么也不做
        }
L900:
    // 将 elmnt 设为 llist[elmnt]
    elmnt = llist[elmnt];
    // 跳转到 L300 标签处
    goto L300;

L1000:
    // 如果 rloc 小于等于 rlmt，则将 adjncy[rloc] 设为 0
    if (rloc <= rlmt) {
        adjncy[rloc] = 0;
    }
/*        -------------------------------------------------------- */
/*        FOR EACH NODE IN THE REACHABLE SET, DO THE FOLLOWING ... */
/*        -------------------------------------------------------- */
    // 将 link 设为 *mdnode
    link = *mdnode;
L1100:
    // 设置 istrt 为 xadj[link]，istop 为 xadj[link + 1] - 1
    istrt = xadj[link];
    istop = xadj[link + 1] - 1;
    // 对于 i 从 istrt 到 istop 的循环
    i__1 = istop;
    for (i = istrt; i <= i__1; ++i) {
        // 将 rnode 设为 adjncy[i]
        rnode = adjncy[i];
        // 将 link 设为 -rnode
        link = -rnode;
        // 如果 rnode 小于 0，则跳转到 L1100 标签处
        if (rnode < 0) {
            goto L1100;
        } else if (rnode == 0) {
            // 如果 rnode 等于 0，则跳转到 L1800 标签处
            goto L1800;
        } else {
            // 否则，跳转到 L1200 标签处
            goto L1200;
        }

L1200:
/*                -------------------------------------------- */
/*                IF RNODE IS IN THE DEGREE LIST STRUCTURE ... */
/*                -------------------------------------------- */
        // 将 pvnode 设为 dbakw[rnode]
        pvnode = dbakw[rnode];
        // 如果 pvnode 等于 0 或者 pvnode 等于 -(*maxint)，则跳转到 L1300 标签处
        if (pvnode == 0 || pvnode == -(*maxint)) {
            goto L1300;
        }
/*                    ------------------------------------- */
/*                    THEN REMOVE RNODE FROM THE STRUCTURE. */
/*                    ------------------------------------- */
        // 将 nxnode 设为 dforw[rnode]
        nxnode = dforw[rnode];
        // 如果 nxnode 大于 0，则将 dbakw[nxnode] 设为 pvnode
        if (nxnode > 0) {
            dbakw[nxnode] = pvnode;
        }
        // 如果 pvnode 大于 0，则将 dforw[pvnode] 设为 nxnode
        if (pvnode > 0) {
            dforw[pvnode] = nxnode;
        }
        // 将 npv 设为 -pvnode
        npv = -pvnode;
        // 如果 pvnode 小于 0，则将 dhead[npv] 设为 nxnode
        if (pvnode < 0) {
            dhead[npv] = nxnode;
        }
L1300:
/*                ---------------------------------------- */
/*                PURGE INACTIVE QUOTIENT NABORS OF RNODE. */
/*                ---------------------------------------- */
        // 设置 jstrt 为 xadj[rnode]，jstop 为 xadj[rnode + 1] - 1，xqnbr 设为 jstrt
        jstrt = xadj[rnode];
        jstop = xadj[rnode + 1] - 1;
        xqnbr = jstrt;
        // 对于循环 i__2 从 jstrt 到 jstop
        i__2 = jstop;
    for (j = jstrt; j <= i__2; ++j) {
        # 遍历循环，从 jstrt 开始直到 i__2 结束（包含）。
        nabor = adjncy[j];
        # 获取 adjncy 数组中索引为 j 的元素赋值给 nabor。

        if (nabor == 0) {
        # 如果 nabor 等于 0，则跳转到标签 L1500。
        goto L1500;
        }

        if (marker[nabor] >= *tag) {
        # 如果 marker 数组中索引为 nabor 的元素大于或等于指针 tag 指向的值，则跳转到标签 L1400。
        goto L1400;
        }

        adjncy[xqnbr] = nabor;
        # 将 nabor 的值存入 adjncy 数组中索引为 xqnbr 的位置。
        ++xqnbr;
        # xqnbr 自增1，准备存储下一个邻居节点。
L1400:
        ;
    }
L1500:
/*                ---------------------------------------- */
/*                IF NO ACTIVE NABOR AFTER THE PURGING ... */
/*                ---------------------------------------- */
    nqnbrs = xqnbr - jstrt;
    // 计算当前节点 rnode 的邻居数量
    if (nqnbrs > 0) {
        goto L1600;
    }
/*                    ----------------------------- */
/*                    THEN MERGE RNODE WITH MDNODE. */
/*                    ----------------------------- */
    // 若没有活跃邻居，则将 rnode 的 qsize 合并到 mdnode 上
    qsize[*mdnode] += qsize[rnode];
    // 清零 rnode 的 qsize
    qsize[rnode] = 0;
    // 将 rnode 标记为已处理过的节点
    marker[rnode] = *maxint;
    // 设置 rnode 的前向链接为负的 mdnode
    dforw[rnode] = -(*mdnode);
    // 设置 rnode 的后向链接为负的最大整数值
    dbakw[rnode] = -(*maxint);
    // 跳转到 L1700 继续执行
    goto L1700;
L1600:
/*                -------------------------------------- */
/*                ELSE FLAG RNODE FOR DEGREE UPDATE, AND */
/*                ADD MDNODE AS A NABOR OF RNODE. */
/*                -------------------------------------- */
    // 标记 rnode 需要更新其度数
    dforw[rnode] = nqnbrs + 1;
    // rnode 的后向链接置零
    dbakw[rnode] = 0;
    // 将 mdnode 添加为 rnode 的邻居
    adjncy[xqnbr] = *mdnode;
    // 增加 xqnbr 指针
    ++xqnbr;
    // 如果 xqnbr 小于等于 jstop，则将 adjncy[xqnbr] 置零
    if (xqnbr <= jstop) {
        adjncy[xqnbr] = 0;
    }

L1700:
    ;
    }
L1800:
    return 0;

} /* slu_mmdelm_ */

/* *************************************************************** */
/* *************************************************************** */
/* *****     MMDUPD ..... MULTIPLE MINIMUM DEGREE UPDATE     ***** */
/* *************************************************************** */
/* *************************************************************** */

/*     AUTHOR - JOSEPH W.H. LIU */
/*              DEPT OF COMPUTER SCIENCE, YORK UNIVERSITY. */

/*     PURPOSE - THIS ROUTINE UPDATES THE DEGREES OF NODES */
/*        AFTER A MULTIPLE ELIMINATION STEP. */

/*     INPUT PARAMETERS - */
/*        EHEAD  - THE BEGINNING OF THE LIST OF ELIMINATED */
/*                 NODES (I.E., NEWLY FORMED ELEMENTS). */
/*        NEQNS  - NUMBER OF EQUATIONS. */
/*        (XADJ,ADJNCY) - ADJACENCY STRUCTURE. */
/*        DELTA  - TOLERANCE VALUE FOR MULTIPLE ELIMINATION. */
/*        MAXINT - MAXIMUM MACHINE REPRESENTABLE (SHORT) */
/*                 INTEGER. */

/*     UPDATED PARAMETERS - */
/*        MDEG   - NEW MINIMUM DEGREE AFTER DEGREE UPDATE. */
/*        (DHEAD,DFORW,DBAKW) - DEGREE DOUBLY LINKED STRUCTURE. */
/*        QSIZE  - SIZE OF SUPERNODE. */
/*        LLIST  - WORKING LINKED LIST. */
/*        MARKER - MARKER VECTOR FOR DEGREE UPDATE. */
/*        TAG    - TAG VALUE. */

/* *************************************************************** */


/* Subroutine */ int slu_mmdupd_(int_t *ehead, int *neqns, int_t *xadj, 
    shortint *adjncy, int_t *delta, int_t *mdeg, shortint *dhead, 
        int *dforw, int *dbakw, shortint *qsize, shortint *llist, 
    shortint *marker, int_t *maxint, int_t *tag)
{
    /* System generated locals */
    int_t i__1, i__2;

    /* Local variables */
    int_t node, mtag, link, mdeg0, i, j, enode, fnode, nabor, elmnt, 
        istop, jstop, q2head, istrt, jstrt, qxhead, iq2, deg, deg0;


/* *************************************************************** */
/* *************************************************************** */

    /* 调整参数数组的指针 */
    --marker;
    --llist;
    --qsize;
    --dbakw;
    --dforw;
    --dhead;
    --adjncy;
    --xadj;

    /* 函数体开始 */
    mdeg0 = *mdeg + *delta;
    elmnt = *ehead;
L100:
/*            ------------------------------------------------------- */
/*            对于每个新形成的元素，执行以下操作。 */
/*            （必要时重置标记值。） */
/*            ------------------------------------------------------- */
    if (elmnt <= 0) {
    return 0;
    }
    mtag = *tag + mdeg0;
    if (mtag < *maxint) {
    goto L300;
    }
    *tag = 1;
    i__1 = *neqns;
    for (i = 1; i <= i__1; ++i) {
    if (marker[i] < *maxint) {
        marker[i] = 0;
    }
/* L200: */
    }
    mtag = *tag + mdeg0;
L300:
/*            --------------------------------------------- */
/*            从与元素相关联的节点创建两个链接列表：一个带有两个邻居（Q2HEAD）在邻接结构中，另一个带有多于两个邻居（QXHEAD）。
/*            同时计算DEG0，这个元素中的节点数。 */
/*            --------------------------------------------- */
    q2head = 0;
    qxhead = 0;
    deg0 = 0;
    link = elmnt;
L400:
    istrt = xadj[link];
    istop = xadj[link + 1] - 1;
    i__1 = istop;
    for (i = istrt; i <= i__1; ++i) {
    enode = adjncy[i];
    link = -enode;
    if (enode < 0) {
        goto L400;
    } else if (enode == 0) {
        goto L800;
    } else {
        goto L500;
    }

L500:
    if (qsize[enode] == 0) {
        goto L700;
    }
    deg0 += qsize[enode];
    marker[enode] = mtag;
/*                        ---------------------------------- */
/*                        如果ENODE需要更新度数，则执行以下操作。 */
/*                        ---------------------------------- */
    if (dbakw[enode] != 0) {
        goto L700;
    }
/*                            --------------------------------------- */
/*                            将ENODE放入QXHEAD或Q2HEAD列表中。 */
/*                            --------------------------------------- */
    if (dforw[enode] == 2) {
        goto L600;
    }
    llist[enode] = qxhead;
    qxhead = enode;
    goto L700;
L600:
    llist[enode] = q2head;
    q2head = enode;
L700:
    ;
    }
L800:
/*            -------------------------------------------- */
/*            对于Q2列表中的每个ENODE，执行以下操作。 */
/*            -------------------------------------------- */
    enode = q2head;
    iq2 = 1;
L900:
    if (enode <= 0) {
    goto L1500;
    }
    if (dbakw[enode] != 0) {
    goto L2200;
    }
    ++(*tag);
    deg = deg0;
/*                    ------------------------------------------ */
/*                    标识其他相邻元素的邻接元素。 */
/*                    ------------------------------------------ */


这是对给定代码块的注释，按照要求，每行代码都有详细的注释解释其作用和功能。
    # 获取节点 `enode` 的邻接节点列表的起始索引
    istrt = xadj[enode];
    # 获取起始索引 `istrt` 处的第一个邻接节点
    nabor = adjncy[istrt];
    # 如果第一个邻接节点恰好是 `elmnt` 节点本身，则取下一个邻接节点
    if (nabor == elmnt) {
        nabor = adjncy[istrt + 1];
    }
/*                    ------------------------------------------------ */
/*                    IF NABOR IS UNELIMINATED, INCREASE DEGREE COUNT. */
/*                    ------------------------------------------------ */
    link = nabor;  // 将 nabor 节点赋值给 link
    if (dforw[nabor] < 0) {  // 如果 dforw[nabor] 小于 0，执行下一步
    goto L1000;  // 跳转到标签 L1000
    }
    deg += qsize[nabor];  // 增加 deg 计数器的值，增加 qsize[nabor] 的大小
    goto L2100;  // 跳转到标签 L2100
L1000:
/*                        -------------------------------------------- */
/*                        OTHERWISE, FOR EACH NODE IN THE 2ND ELEMENT, */
/*                        DO THE FOLLOWING. */
/*                        -------------------------------------------- */
    istrt = xadj[link];  // 获取 link 节点在 xadj 数组中的起始位置
    istop = xadj[link + 1] - 1;  // 获取 link 节点在 xadj 数组中的结束位置
    i__1 = istop;
    for (i = istrt; i <= i__1; ++i) {  // 循环遍历 link 节点的邻居节点
    node = adjncy[i];  // 获取当前邻居节点的编号
    link = -node;  // 将当前邻居节点的负值赋给 link
    if (node == enode) {  // 如果当前邻居节点等于 enode，跳转到标签 L1400
        goto L1400;
    }
    if (node < 0) {  // 如果当前邻居节点小于 0，跳转到标签 L1000
        goto L1000;
    } else if (node == 0) {  // 如果当前邻居节点等于 0，跳转到标签 L2100
        goto L2100;
    } else {  // 否则跳转到标签 L1100
        goto L1100;
    }

L1100:
    if (qsize[node] == 0) {  // 如果当前邻居节点的 qsize 为 0，跳转到标签 L1400
        goto L1400;
    }
    if (marker[node] >= *tag) {  // 如果当前邻居节点的 marker 大于等于 *tag，跳转到标签 L1200
        goto L1200;
    }
/*                                -----------------------------------
-- */
/*                                CASE WHEN NODE IS NOT YET CONSIDERED
. */
/*                                -----------------------------------
-- */
    marker[node] = *tag;  // 将 *tag 的值赋给当前邻居节点的 marker
    deg += qsize[node];  // 增加 deg 计数器的值，增加当前邻居节点的 qsize 大小
    goto L1400;  // 跳转到标签 L1400
L1200:
/*                            ----------------------------------------
 */
/*                            CASE WHEN NODE IS INDISTINGUISHABLE FROM
 */
/*                            ENODE.  MERGE THEM INTO A NEW SUPERNODE.
 */
/*                            ----------------------------------------
 */
    if (dbakw[node] != 0) {  // 如果当前邻居节点的 dbakw 不等于 0，跳转到标签 L1400
        goto L1400;
    }
    if (dforw[node] != 2) {  // 如果当前邻居节点的 dforw 不等于 2，跳转到标签 L1300
        goto L1300;
    }
    qsize[enode] += qsize[node];  // 将当前邻居节点的 qsize 加到 enode 节点的 qsize 上
    qsize[node] = 0;  // 将当前邻居节点的 qsize 置为 0
    marker[node] = *maxint;  // 将 *maxint 的值赋给当前邻居节点的 marker
    dforw[node] = -enode;  // 将 -enode 的值赋给当前邻居节点的 dforw
    dbakw[node] = -(*maxint);  // 将 -*maxint 的值赋给当前邻居节点的 dbakw
    goto L1400;  // 跳转到标签 L1400
L1300:
/*                            -------------------------------------- 
*/
/*                            CASE WHEN NODE IS OUTMATCHED BY ENODE. 
*/
/*                            -------------------------------------- 
*/
    if (dbakw[node] == 0) {  // 如果当前邻居节点的 dbakw 等于 0
        dbakw[node] = -(*maxint);  // 将 -*maxint 的值赋给当前邻居节点的 dbakw
    }
L1400:
    ;
    }
    goto L2100;  // 跳转到标签 L2100
L1500:
/*                ------------------------------------------------ */
/*                FOR EACH ENODE IN THE QX LIST, DO THE FOLLOWING. */
/*                ------------------------------------------------ */
    enode = qxhead;  // 将 qxhead 节点赋值给 enode
    iq2 = 0;  // 将 iq2 置为 0
L1600:
    if (enode <= 0) {  // 如果 enode 小于等于 0，跳转到标签 L2300
    goto L2300;
    }
    if (dbakw[enode] != 0) {  // 如果 dbakw[enode] 不等于 0，跳转到标签 L2200
    goto L2200;
    }
    ++(*tag);  // 将 *tag 的值增加 1
    deg = deg0;  // 将 deg0 的值赋给 deg
/*                        --------------------------------- */
/*                        FOR EACH UNMARKED NABOR OF ENODE, */
/*                        DO THE FOLLOWING. */
/*                        --------------------------------- */
    istrt = xadj[enode];  // 获取 enode 节点在 xadj 数组中的起始位置
    istop = xadj[enode + 1] - 1;  // 获取 enode 节点在 xadj 数组中的结束位置
    i__1 = istop;
    for (i = istrt; i <= i__1; ++i) {

    node = adjncy[i];  // 获取当前邻居节点的编号
    link = -node;  // 将当前邻居节点的负值赋给 link
    if (node == enode) {  // 如果当前邻居节点等于 enode，跳转到标签 L1400
        goto L1400;
    }
    if (node < 0) {  // 如果当前邻居节点小于 0，跳转到标签 L1000
        goto L1000;
    } else if (node == 0) {  // 如果当前邻居节点等于 0，跳转到标签 L2100
        goto L2100;
    } else {  // 否则跳转到标签 L1100
        goto L1100;
    }

L1100:
    if (qsize[node] == 0) {  // 如果当前邻居节点的 qsize 为 0，跳转到标签 L1400
        goto L1400;
    }
    if (marker[node] >= *tag) {  // 如果当前邻居节点的 marker 大于等于 *tag，跳转到标签 L1200
        goto L1200;
    }
/*                                -----------------------------------
-- */
/*                                CASE WHEN NODE IS NOT YET CONSIDERED
. */
/*                                -----------------------------------
-- */
    marker[node] = *tag;  // 将 *tag 的值赋给当前邻居节点的 marker
    deg += qsize[node];  // 增加 deg 计数器的值，增加当前邻居节点的 qsize 大小
    goto L1400;  // 跳转到标签 L1400
L1200:
/*                            ----------------------------------------
 */
/*                            CASE WHEN NODE IS INDISTINGUISHABLE FROM
 */
/*                            ENODE.  MERGE THEM INTO A NEW SUPERNODE.
 */
/*                            ----------------------------------------
 */
    if (dbakw[node] != 0) {  // 如果当前邻居节点的 dbakw 不等于 0，跳转到标签 L1400
        goto L1400;
    }
    if (dforw[node] != 2) {  // 如果当前邻居节点的 dforw 不等于 2，跳转到标签 L1300
        goto L1300;
    }
    qsize[enode] += qsize[node];  // 将当前邻居节点的 qsize 加到 enode 节点的 qsize 上
    qsize[node] = 0;  // 将当前邻居节点的 qsize 置为 0
    marker[node] = *maxint;  // 将 *maxint 的值赋给当前邻居节点的 marker
    dforw[node] = -enode;  // 将 -enode 的值赋给当前邻居节点的 dforw
    dbakw[node] = -(*maxint);  // 将 -*maxint 的值赋给当前邻居节点的 dbakw
    goto L1400;  // 跳转到标签 L1400
L1300:
/*                            -------------------------------------- 
*/
/*                            CASE WHEN NODE IS OUTMATCHED BY ENODE. 
*/
/*                            -------------------------------------- 
*/
    if (dbakw[node] == 0) {  // 如果当前邻居节点的 dbakw 等于 0
        dbakw[node] = -(*maxint);  // 将 -*maxint 的值赋给当前邻居节点的 dbakw
    }
L1400:
    ;
    }
    goto L2100;  // 跳转到标签 L2100
L1500:
/*                ------------------------------------------------ */
/*                FOR EACH ENODE IN THE QX LIST, DO THE FOLLOWING. */
/*                ------------------------------------------------ */
    enode = qxhead;  // 将 qxhead 节点赋值给 enode
    iq2 = 0;  // 将 iq2 置为 0
L1600:
    if (enode <= 0) {  // 如果 enode 小于等于 0，跳转到标签 L2300
    goto L2300;
    }
    if (dbakw[enode] != 0) {  // 如果 dbakw[enode] 不等于 0，跳转到标签 L2200
    goto L2200;
    }
    ++(*tag);  // 将 *tag 的值增加 1
    deg = deg0;  // 将 deg0 的值赋给 deg
/*                        --------------------------------- */
/*                        FOR EACH UNMARKED NABOR OF ENODE, */
/*                        DO THE FOLLOWING. */
/*                        --------------------------------- */
    istrt = xadj[enode];  // 获取 enode 节点在 xadj 数组中的起始位置
    istop = xadj[enode + 1] - 1;  // 获取 enode 节点在 xadj 数组中的结束位置
    i__1 = istop;
    for (i = istrt; i <= i__1; ++i) {
    nabor = adjncy[i];
    # 从数组 adjncy 中取出索引 i 处的值赋给 nabor

    if (nabor == 0) {
        # 如果 nabor 的值为 0，则跳转到标签为 L2100 的位置
        goto L2100;
    }

    if (marker[nabor] >= *tag) {
        # 如果 marker 数组中索引为 nabor 的元素的值大于或等于指针 tag 指向的值，则跳转到标签为 L2000 的位置
        goto L2000;
    }

    marker[nabor] = *tag;
    # 将 marker 数组中索引为 nabor 的元素设置为指针 tag 指向的值

    link = nabor;
    # 将变量 link 的值设置为 nabor 的值
/*                                ------------------------------ */
/*                                IF UNELIMINATED, INCLUDE IT IN */
/*                                DEG COUNT. */
/*                                ------------------------------ */
    if (dforw[nabor] < 0) {
        // 如果节点未被消除，则将其包含在度数计数中
        goto L1700;
    }
    deg += qsize[nabor];
    goto L2000;
L1700:
/*                                    ------------------------------- 
*/
/*                                    IF ELIMINATED, INCLUDE UNMARKED 
*/
/*                                    NODES IN THIS ELEMENT INTO THE 
*/
/*                                    DEGREE COUNT. */
/*                                    ------------------------------- 
*/
    jstrt = xadj[link];
    jstop = xadj[link + 1] - 1;
    i__2 = jstop;
    // 遍历当前元素中未标记节点，将其包含到度数计数中
    for (j = jstrt; j <= i__2; ++j) {
        node = adjncy[j];
        link = -node;
        if (node < 0) {
        goto L1700;
        } else if (node == 0) {
        goto L2000;
        } else {
        goto L1800;
        }

L1800:
        if (marker[node] >= *tag) {
        goto L1900;
        }
        marker[node] = *tag;
        deg += qsize[node];
L1900:
        ;
    }
L2000:
    ;
    }
L2100:
/*                    ------------------------------------------- */
/*                    UPDATE EXTERNAL DEGREE OF ENODE IN DEGREE */
/*                    STRUCTURE, AND MDEG (MIN DEG) IF NECESSARY. */
/*                    ------------------------------------------- */
    deg = deg - qsize[enode] + 1;
    fnode = dhead[deg];
    dforw[enode] = fnode;
    dbakw[enode] = -deg;
    // 更新节点在度结构中的外部度，并在必要时更新最小度MDEG
    if (fnode > 0) {
    dbakw[fnode] = enode;
    }
    dhead[deg] = enode;
    if (deg < *mdeg) {
    *mdeg = deg;
    }
L2200:
/*                    ---------------------------------- */
/*                    GET NEXT ENODE IN CURRENT ELEMENT. */
/*                    ---------------------------------- */
    enode = llist[enode];
    if (iq2 == 1) {
    goto L900;
    }
    goto L1600;
L2300:
/*            ----------------------------- */
/*            GET NEXT ELEMENT IN THE LIST. */
/*            ----------------------------- */
    *tag = mtag;
    elmnt = llist[elmnt];
    // 获取列表中的下一个元素
    goto L100;

} /* slu_mmdupd_ */

/* *************************************************************** */
/* *************************************************************** */
/* *****     MMDNUM ..... MULTI MINIMUM DEGREE NUMBERING     ***** */
/* *************************************************************** */
/* *************************************************************** */

/*     AUTHOR - JOSEPH W.H. LIU */
/*              DEPT OF COMPUTER SCIENCE, YORK UNIVERSITY. */

/*     PURPOSE - THIS ROUTINE PERFORMS THE FINAL STEP IN */
/*        PRODUCING THE PERMUTATION AND INVERSE PERMUTATION */
/*        VECTORS IN THE MULTIPLE ELIMINATION VERSION OF THE */
/*        MINIMUM DEGREE ORDERING ALGORITHM. */

/*     INPUT PARAMETERS - */
/*        NEQNS  - NUMBER OF EQUATIONS. */


注释：
/*        QSIZE  - SIZE OF SUPERNODES AT ELIMINATION. */
/*        QSIZE - 消元时超节点的大小。 */

/*     UPDATED PARAMETERS - */
/*        INVP   - INVERSE PERMUTATION VECTOR.  ON INPUT, */
/*                 IF QSIZE(NODE)=0, THEN NODE HAS BEEN MERGED */
/*                 INTO THE NODE -INVP(NODE); OTHERWISE, */
/*                 -INVP(NODE) IS ITS INVERSE LABELLING. */
/*        INVP - 逆置换向量。在输入时，如果 QSIZE(NODE)=0，则节点已经被合并 */
/*               到节点 -INVP(NODE) 中；否则，-INVP(NODE) 是它的逆标签。 */

/*     OUTPUT PARAMETERS - */
/*        PERM   - THE PERMUTATION VECTOR. */
/*        PERM - 排列向量。 */

/* *************************************************************** */

/* Subroutine */ int slu_mmdnum_(int *neqns, int *perm, int *invp, 
    shortint *qsize)
{
    /* System generated locals */

    int i__1;

    /* Local variables */
    int_t node, root, nextf, father, nqsize, num;


/* *************************************************************** */


/* *************************************************************** */

    /* Parameter adjustments */
    --qsize;
    --invp;
    --perm;

    /* Function Body */
    i__1 = *neqns;
    for (node = 1; node <= i__1; ++node) {
    nqsize = qsize[node];
    if (nqsize <= 0) {
        perm[node] = invp[node];
    }
    if (nqsize > 0) {
        perm[node] = -invp[node];
    }
/* L100: */
    }
/*        ------------------------------------------------------ */
/*        FOR EACH NODE WHICH HAS BEEN MERGED, DO THE FOLLOWING. */
/*        ------------------------------------------------------ */
    i__1 = *neqns;
    for (node = 1; node <= i__1; ++node) {
    if (perm[node] > 0) {
        goto L500;
    }
/*                ----------------------------------------- */
/*                TRACE THE MERGED TREE UNTIL ONE WHICH HAS */
/*                NOT BEEN MERGED, CALL IT ROOT. */
/*                ----------------------------------------- */
    father = node;
L200:
    if (perm[father] > 0) {
        goto L300;
    }
    father = -perm[father];
    goto L200;
L300:
/*                ----------------------- */
/*                NUMBER NODE AFTER ROOT. */
/*                ----------------------- */
    root = father;
    num = perm[root] + 1;
    invp[node] = -num;
    perm[root] = num;
/*                ------------------------ */
/*                SHORTEN THE MERGED TREE. */
/*                ------------------------ */
    father = node;
L400:
    nextf = -perm[father];
    if (nextf <= 0) {
        goto L500;
    }
    perm[father] = -root;
    father = nextf;
    goto L400;
L500:
    ;
    }
/*        ---------------------- */
/*        READY TO COMPUTE PERM. */
/*        ---------------------- */
    i__1 = *neqns;
    for (node = 1; node <= i__1; ++node) {
    num = -invp[node];
    invp[node] = num;
    perm[num] = node;
/* L600: */
    }
    return 0;

} /* slu_mmdnum_ */
```