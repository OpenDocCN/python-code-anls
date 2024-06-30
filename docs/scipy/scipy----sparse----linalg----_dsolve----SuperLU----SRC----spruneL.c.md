# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\spruneL.c`

```
/*
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

/*
 * @file spruneL.c
 * \brief Prunes the L-structure
 *
 *<pre>
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
 *</pre>
 */

#include "slu_sdefs.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *   Prunes the L-structure of supernodes whose L-structure
 *   contains the current pivot row "pivrow"
 * </pre>
 */

void
spruneL(
       const int  jcol,         /* in */                              // jcol: 当前列号
       const int  *perm_r,   /* in */                              // perm_r: 行排列的排列数组
       const int  pivrow,    /* in */                              // pivrow: 当前主元行号
       const int  nseg,         /* in */                              // nseg: 分段的数量
       const int  *segrep,   /* in */                              // segrep: 分段的起始索引数组
       const int  *repfnz,   /* in */                              // repfnz: 每个分段的第一个非零元素的列索引
       int_t      *xprune,   /* out */                            // xprune: 标记哪些超节点需要被剪枝
       GlobalLU_t *Glu       /* modified - global LU data structures */  // Glu: 全局LU数据结构
       )
{

    float     utemp;                                                  // utemp: 临时变量用于存储浮点数
    int        jsupno, irep, irep1, kmin, kmax, krow, movnum;           // jsupno, irep, irep1, kmin, kmax, krow, movnum: 整型变量
    int_t      i, ktemp, minloc, maxloc;                               // i, ktemp, minloc, maxloc: 整型变量
    int        do_prune; /* logical variable */                        // do_prune: 逻辑变量，指示是否需要剪枝
    int        *xsup, *supno;                                          // xsup, supno: 整型数组
    int_t      *lsub, *xlsub;                                          // lsub, xlsub: 整型数组
    float     *lusup;                                                  // lusup: 浮点数数组
    int_t      *xlusup;                                                 // xlusup: 整型数组

    xsup       = Glu->xsup;                                            // 获取全局LU数据结构中的xsup数组
    supno      = Glu->supno;                                           // 获取全局LU数据结构中的supno数组
    lsub       = Glu->lsub;                                            // 获取全局LU数据结构中的lsub数组
    xlsub      = Glu->xlsub;                                           // 获取全局LU数据结构中的xlsub数组
    lusup      = (float *) Glu->lusup;                                 // 获取全局LU数据结构中的lusup数组，并将其类型转换为float *
    xlusup     = Glu->xlusup;                                          // 获取全局LU数据结构中的xlusup数组
    
    /*
     * For each supernode-rep irep in U[*,j]
     */
    jsupno = supno[jcol];                                              // 获取第jcol列对应的超节点编号

    for (i = 0; i < nseg; i++) {                                        // 遍历每个分段

        irep = segrep[i];                                               // 获取当前分段的起始索引
        irep1 = irep + 1;                                               // 下一个分段的起始索引
        do_prune = FALSE;                                               // 初始化剪枝标记为假

        /* Don't prune with a zero U-segment */
        if ( repfnz[irep] == EMPTY )                                    // 如果当前分段的第一个非零元素索引为EMPTY，跳过
            continue;

        /* If a snode overlaps with the next panel, then the U-segment 
         * is fragmented into two parts -- irep and irep1. We should let
         * pruning occur at the rep-column in irep1's snode. 
         */
        if ( supno[irep] == supno[irep1] )                               // 如果当前分段和下一个分段属于同一个超节点，跳过
            continue;

        /*
         * If it has not been pruned & it has a nonz in row L[pivrow,i]
         */
    # 如果当前 irep 对应的 supno 不等于 jsupno
    if ( supno[irep] != jsupno ) {
        # 如果 xprune[irep] 大于等于 xlsub[irep1]
        if ( xprune[irep] >= xlsub[irep1] ) {
            # 设置 kmin 为 xlsub[irep]，设置 kmax 为 xlsub[irep1] 减一
            kmin = xlsub[irep];
            kmax = xlsub[irep1] - 1;
            # 在 kmin 到 kmax 范围内循环
            for (krow = kmin; krow <= kmax; krow++) 
                # 如果 lsub[krow] 等于 pivrow
                if ( lsub[krow] == pivrow ) {
                    # 设置 do_prune 为真
                    do_prune = TRUE;
                    # 退出循环
                    break;
                }
        }
        
        # 如果 do_prune 为真
        if ( do_prune ) {

             /* Do a quicksort-type partition
              * movnum=TRUE means that the num values have to be exchanged.
              */
            # 设置 movnum 为假
            movnum = FALSE;
            # 如果 irep 等于 xsup[supno[irep]]，即 Snode 大小为 1
            if ( irep == xsup[supno[irep]] ) /* Snode of size 1 */
                # 设置 movnum 为真
                movnum = TRUE;

            # 当 kmin 小于等于 kmax 时执行循环
            while ( kmin <= kmax ) {

                # 如果 perm_r[lsub[kmax]] 等于 EMPTY
                if ( perm_r[lsub[kmax]] == EMPTY ) 
                    # 减小 kmax
                    kmax--;
                # 否则，如果 perm_r[lsub[kmin]] 不等于 EMPTY
                else if ( perm_r[lsub[kmin]] != EMPTY )
                    # 增加 kmin
                    kmin++;
                else { /* kmin below pivrow (not yet pivoted), and kmax
                            * above pivrow: interchange the two subscripts
                */
                    # 交换 lsub[kmin] 和 lsub[kmax]
                    ktemp = lsub[kmin];
                    lsub[kmin] = lsub[kmax];
                    lsub[kmax] = ktemp;

                    /* If the supernode has only one column, then we
                      * only keep one set of subscripts. For any subscript 
                     * interchange performed, similar interchange must be 
                     * done on the numerical values.
                      */
                    # 如果 movnum 为真，进行数值的交换
                    if ( movnum ) {
                        minloc = xlusup[irep] + (kmin - xlsub[irep]);
                        maxloc = xlusup[irep] + (kmax - xlsub[irep]);
                        utemp = lusup[minloc];
                        lusup[minloc] = lusup[maxloc];
                        lusup[maxloc] = utemp;
                    }

                    # 增加 kmin，减小 kmax
                    kmin++;
                    kmax--;
                }

            } /* while */

            # 更新 xprune[irep] 的值为 kmin，即剪枝操作
            xprune[irep] = kmin;    /* Pruning */

        } /* if ( do_prune ) */
#ifdef CHK_PRUNE
    #ifdef 指令用于条件编译，检查是否定义了 CHK_PRUNE 宏
    printf("    After spruneL(),using col %d:  xprune[%d] = %d\n", 
            jcol, irep, kmin);
    # 打印调试信息，显示在 spruneL() 后，使用列号 jcol，将 xprune[irep] 设置为 kmin 的值
#endif
        } /* if do_prune */
    # 如果 do_prune 为真，则执行以下代码块

    } /* if */
    # 结束 if 条件判断块

    } /* for each U-segment... */
    # 结束对每个 U-段的循环遍历
}
# 结束函数定义
```