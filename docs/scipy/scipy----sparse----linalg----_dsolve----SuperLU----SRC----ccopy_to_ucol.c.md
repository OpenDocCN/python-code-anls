# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\ccopy_to_ucol.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file ccopy_to_ucol.c
 * \brief Copy a computed column of U to the compressed data structure
 *
 * <pre>
 * -- SuperLU routine (version 2.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November 15, 1997
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

int
ccopy_to_ucol(
          int        jcol,      /* in */                              // 输入参数：列索引
          int        nseg,      /* in */                              // 输入参数：段数
          int        *segrep,  /* in */                              // 输入参数：段表示数组
          int        *repfnz,  /* in */                              // 输入参数：首非零元素表示数组
          int        *perm_r,  /* in */                              // 输入参数：行置换数组
          singlecomplex     *dense,   /* modified - reset to zero on return */  // 修改参数：密集矩阵，返回时置零
          GlobalLU_t *Glu      /* modified */                         // 修改参数：全局 LU 因子结构体
          )
{
/* 
 * Gather from SPA dense[*] to global ucol[*].
 */
    int ksub, krep, ksupno;                                          // 局部变量定义：子列索引，重复索引，超节点号
    int i, k, kfnz, segsze;                                          // 局部变量定义：循环变量，索引，首非零元素索引，段大小
    int fsupc, irow, jsupno;                                         // 局部变量定义：前导超节点列号，行索引，当前列的超节点号
    int_t isub, nextu, new_next, mem_error;                          // 局部变量定义：行索引，下一个 U 列索引，新的下一个 U 列索引，内存错误标志
    int       *xsup, *supno;                                         // 局部变量定义：行开始索引，超节点号数组
    int_t     *lsub, *xlsub;                                         // 局部变量定义：非零元素行索引数组，行偏移数组
    singlecomplex    *ucol;                                          // 局部变量定义：U 列的复数数据
    int_t     *usub, *xusub;                                         // 局部变量定义：U 的行索引数组，行偏移数组
    int_t       nzumax;                                               // 局部变量定义：U 行索引数组的最大长度
    singlecomplex    zero = {0.0, 0.0};                               // 局部变量定义：零元素的复数表示

    xsup    = Glu->xsup;                                              // 获取全局 LU 因子结构体中的行开始索引
    supno   = Glu->supno;                                             // 获取全局 LU 因子结构体中的超节点号数组
    lsub    = Glu->lsub;                                              // 获取全局 LU 因子结构体中的非零元素行索引数组
    xlsub   = Glu->xlsub;                                             // 获取全局 LU 因子结构体中的行偏移数组
    ucol    = (singlecomplex *) Glu->ucol;                            // 获取全局 LU 因子结构体中的 U 列的复数数据
    usub    = Glu->usub;                                              // 获取全局 LU 因子结构体中的 U 的行索引数组
    xusub   = Glu->xusub;                                             // 获取全局 LU 因子结构体中的 U 行偏移数组
    nzumax  = Glu->nzumax;                                            // 获取全局 LU 因子结构体中的 U 行索引数组的最大长度
    
    jsupno = supno[jcol];                                             // 获取当前列的超节点号
    nextu  = xusub[jcol];                                             // 获取当前列的 U 行偏移

    k = nseg - 1;
    for (ksub = 0; ksub < nseg; ksub++) {                             // 循环遍历每个段
        krep = segrep[k--];                                           // 获取当前段的重复索引
        ksupno = supno[krep];                                         // 获取当前段的超节点号
    if ( ksupno != jsupno ) { /* 如果 ksupno 不等于 jsupno，则进入 ucol[] */

        kfnz = repfnz[krep];
        if ( kfnz != EMPTY ) {    /* 如果 kfnz 不为空 */

            fsupc = xsup[ksupno];
            isub = xlsub[fsupc] + kfnz - fsupc;
            segsze = krep - kfnz + 1;

            new_next = nextu + segsze;
            while ( new_next > nzumax ) { /* 当新的 nextu 超过 nzumax 时进行内存扩展 */
                mem_error = cLUMemXpand(jcol, nextu, UCOL, &nzumax, Glu);
                if (mem_error) return (mem_error);
                ucol = (singlecomplex *) Glu->ucol;
                mem_error = cLUMemXpand(jcol, nextu, USUB, &nzumax, Glu);
                if (mem_error) return (mem_error);
                usub = Glu->usub;
                lsub = Glu->lsub;
            }

            for (i = 0; i < segsze; i++) { /* 将稠密矩阵中的数据填入因子分解中 */
                irow = lsub[isub];
                usub[nextu] = perm_r[irow];
                ucol[nextu] = dense[irow];
                dense[irow] = zero;
                nextu++;
                isub++;
            } 

        }

    }

    } /* 对于每个段落... */

    xusub[jcol + 1] = nextu;      /* 完成 U[*,jcol] 的填充 */
    return 0;
}


注释：


# 这行代码表示一个函数的结束或一个代码块的结束（比如 if、for、while 的结束）。
# 在这里单独出现，可能是一个函数定义、循环、条件语句等的结尾标志。
```