# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zpruneL.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file zpruneL.c
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


#include "slu_zdefs.h"

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
zpruneL(
       const int  jcol,         /* in */                          // 列索引，用于确定当前操作的列
       const int  *perm_r,   /* in */                          // 行排列的排列向量
       const int  pivrow,    /* in */                          // 当前主元行的行索引
       const int  nseg,         /* in */                          // 分段数目，即超节点的数量
       const int  *segrep,   /* in */                          // 分段的起始超节点索引数组
       const int  *repfnz,   /* in */                          // 超节点中第一个非零元素的列索引数组
       int_t      *xprune,   /* out */                        // 标记数组，用于指示哪些超节点被修剪
       GlobalLU_t *Glu       /* modified - global LU data structures */  // 全局 LU 数据结构指针
       )
{

    doublecomplex     utemp;                                        // 临时复数变量
    int        jsupno, irep, irep1, kmin, kmax, krow, movnum;  // 整型变量声明
    int_t      i, ktemp, minloc, maxloc;                     // 整型长变量声明
    int        do_prune; /* logical variable */                   // 逻辑变量，表示是否进行修剪操作
    int        *xsup, *supno;                                      // 整型指针声明
    int_t      *lsub, *xlsub;                                     // 长整型指针声明
    doublecomplex     *lusup;                                       // 复数指针声明
    int_t      *xlusup;                                            // 长整型指针声明

    xsup       = Glu->xsup;                                        // 获取全局 LU 结构中的 xsup 数组
    supno      = Glu->supno;                                       // 获取全局 LU 结构中的 supno 数组
    lsub       = Glu->lsub;                                        // 获取全局 LU 结构中的 lsub 数组
    xlsub      = Glu->xlsub;                                       // 获取全局 LU 结构中的 xlsub 数组
    lusup      = (doublecomplex *) Glu->lusup;                      // 获取全局 LU 结构中的 lusup 数组
    xlusup     = Glu->xlusup;                                       // 获取全局 LU 结构中的 xlusup 数组
    
    /*
     * For each supernode-rep irep in U[*,j]
     */
    jsupno = supno[jcol];                                           // 获取列 jcol 对应的超节点编号
    for (i = 0; i < nseg; i++) {                                     // 遍历所有的超节点分段

    irep = segrep[i];                                               // 获取当前超节点分段的起始超节点索引
    irep1 = irep + 1;                                                // 下一个超节点的索引
    do_prune = FALSE;                                                // 初始化是否进行修剪的标志为假

    /* Don't prune with a zero U-segment */
     if ( repfnz[irep] == EMPTY )                                    // 如果当前超节点的第一个非零元素索引为 EMPTY
        continue;                                                    // 跳过当前超节点的修剪操作

         /* If a snode overlaps with the next panel, then the U-segment 
        * is fragmented into two parts -- irep and irep1. We should let
     * pruning occur at the rep-column in irep1's snode. 
     */
    if ( supno[irep] == supno[irep1] )                               // 如果当前超节点与下一个超节点重叠
        continue;                                                    // 则不进行修剪操作

    /*
     * If it has not been pruned & it has a nonz in row L[pivrow,i]
     */
    # 如果当前 irep 对应的 supno 不等于 jsupno，则执行以下代码块
    if ( supno[irep] != jsupno ) {
        # 如果 xprune[irep] 大于或等于 xlsub[irep1]，则执行以下代码块
        if ( xprune[irep] >= xlsub[irep1] ) {
            # 设置 kmin 和 kmax 的值用于下面的循环
            kmin = xlsub[irep];
            kmax = xlsub[irep1] - 1;
            # 在范围 [kmin, kmax] 中遍历
            for (krow = kmin; krow <= kmax; krow++) 
                # 如果 lsub[krow] 等于 pivrow，则执行以下代码块
                if ( lsub[krow] == pivrow ) {
                    # 设置 do_prune 为 TRUE，表示需要剪枝
                    do_prune = TRUE;
                    # 跳出循环
                    break;
                }
        }
        
        # 如果 do_prune 为 TRUE，则执行以下代码块
        if ( do_prune ) {

            /* Do a quicksort-type partition
             * movnum=TRUE means that the num values have to be exchanged.
             */
            # 初始化 movnum 为 FALSE
            movnum = FALSE;
            # 如果 irep 对应的索引等于其所属超节点的起始索引，则将 movnum 设置为 TRUE
            if ( irep == xsup[supno[irep]] ) /* Snode of size 1 */
                movnum = TRUE;

            # 在 kmin 小于等于 kmax 的范围内循环
            while ( kmin <= kmax ) {

                # 如果 perm_r[lsub[kmax]] 的值为 EMPTY，则减小 kmax
                if ( perm_r[lsub[kmax]] == EMPTY ) 
                    kmax--;
                # 否则，如果 perm_r[lsub[kmin]] 不等于 EMPTY，则增加 kmin
                else if ( perm_r[lsub[kmin]] != EMPTY )
                    kmin++;
                else { /* kmin below pivrow (not yet pivoted), and kmax
                         * above pivrow: interchange the two subscripts
                         */
                    # 交换 lsub[kmin] 和 lsub[kmax] 的值
                    ktemp = lsub[kmin];
                    lsub[kmin] = lsub[kmax];
                    lsub[kmax] = ktemp;

                    /* If the supernode has only one column, then we
                     * only keep one set of subscripts. For any subscript 
                     * interchange performed, similar interchange must be 
                     * done on the numerical values.
                     */
                    # 如果 movnum 为 TRUE，则需要对数值进行相似的交换
                    if ( movnum ) {
                        # 计算对应的数值索引 minloc 和 maxloc
                        minloc = xlusup[irep] + (kmin - xlsub[irep]);
                        maxloc = xlusup[irep] + (kmax - xlsub[irep]);
                        # 交换对应的数值 lusup[minloc] 和 lusup[maxloc]
                        utemp = lusup[minloc];
                        lusup[minloc] = lusup[maxloc];
                        lusup[maxloc] = utemp;
                    }

                    # 增加 kmin，减小 kmax
                    kmin++;
                    kmax--;
                }

            } /* while */

            # 将 kmin 的值赋给 xprune[irep]，表示剪枝位置的更新
            xprune[irep] = kmin;    /* Pruning */

        } /* if ( do_prune ) */
    } /* if ( supno[irep] != jsupno ) */
#ifdef CHK_PRUNE
    #ifdef 检查宏定义 CHK_PRUNE 是否已经定义
    printf("    After zpruneL(), using col %d:  xprune[%d] = %d\n", 
            jcol, irep, kmin);
    # 使用 printf 打印调试信息，显示 zpruneL() 执行后，使用的列号 jcol，以及 xprune 数组中索引 irep 处的值设为 kmin
#endif
        } /* if do_prune */
    # 如果条件 do_prune 成立，则执行以下代码块
    } /* if */
    # 结束 if 条件判断块
    } /* for each U-segment... */
    # 结束 for 循环，遍历每个 U 段落
}
# 结束函数定义
```