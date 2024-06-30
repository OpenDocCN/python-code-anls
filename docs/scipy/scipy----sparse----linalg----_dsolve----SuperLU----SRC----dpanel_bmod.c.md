# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dpanel_bmod.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file dpanel_bmod.c
 * \brief Performs numeric block updates
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

/*

*/

#include <stdio.h>
#include <stdlib.h>
#include "slu_ddefs.h"

/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 *    Performs numeric block updates (sup-panel) in topological order.
 *    It features: col-col, 2cols-col, 3cols-col, and sup-col updates.
 *    Special processing on the supernodal portion of L\\U[*,j]
 *
 *    Before entering this routine, the original nonzeros in the panel 
 *    were already copied into the spa[m,w].
 *
 *    Updated/Output parameters-
 *    dense[0:m-1,w]: L[*,j:j+w-1] and U[*,j:j+w-1] are returned 
 *    collectively in the m-by-w vector dense[*]. 
 * </pre>
 */

void
dpanel_bmod (
        const int  m,          /* in - number of rows in the matrix */
        const int  w,          /* in */
        const int  jcol,       /* in */
        const int  nseg,       /* in */
        double     *dense,     /* out, of size n by w */
        double     *tempv,     /* working array */
        int        *segrep,    /* in */
        int        *repfnz,    /* in, of size n by w */
        GlobalLU_t *Glu,       /* modified */
        SuperLUStat_t *stat    /* output */
        )
{

#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
    _fcd ftcs1 = _cptofcd("L", strlen("L")),
         ftcs2 = _cptofcd("N", strlen("N")),
         ftcs3 = _cptofcd("U", strlen("U"));
#endif
    int          incx = 1, incy = 1;
    double       alpha, beta;
#endif

    register int k, ksub;
    int          fsupc, nsupc, nsupr, nrow;
    int          krep, krep_ind;
    double       ukj, ukj1, ukj2;
    int_t        luptr, luptr1, luptr2;
    int          segsze;
    int          block_nrow;  /* no of rows in a block row */
    int_t        lptr;          /* Points to the row subscripts of a supernode */
    int          kfnz, irow, no_zeros; 
    register int isub, isub1, i;
    register int jj;          /* Index through each column in the panel */
    int          *xsup, *supno;  /* 指向全局 LU 因子中的超节点相关数据 */
    int_t        *lsub, *xlsub;  /* LU 因子的行索引和行偏移 */
    double       *lusup;         /* LU 因子的数值部分 */
    int_t        *xlusup;        /* LU 因子的超节点列偏移 */
    int          *repfnz_col;    /* 面板中每列的非零元素的代表 */
    double       *dense_col;     /* 面板中每列的稠密部分 */
    double       *tempv1;        /* 用于一维更新的临时向量 */
    double       *TriTmp, *MatvecTmp; /* 用于二维更新的临时向量 */
    double      zero = 0.0;      /* 浮点数零常量 */
    double      one = 1.0;       /* 浮点数一常量 */
    register int ldaTmp;         /* 临时局部变量 */
    register int r_ind, r_hi;    /* 临时局部变量 */
    int  maxsuper, rowblk, colblk;  /* 最大超节点大小，行块大小，列块大小 */
    flops_t  *ops = stat->ops;   /* 统计浮点操作数 */

    xsup    = Glu->xsup;         /* 初始化指向全局 LU 因子中超节点列偏移的指针 */
    supno   = Glu->supno;        /* 初始化指向全局 LU 因子中超节点号码的指针 */
    lsub    = Glu->lsub;         /* 初始化指向全局 LU 因子中行索引的指针 */
    xlsub   = Glu->xlsub;        /* 初始化指向全局 LU 因子中行偏移的指针 */
    lusup   = (double *) Glu->lusup;  /* 初始化指向全局 LU 因子中数值部分的指针 */
    xlusup  = Glu->xlusup;       /* 初始化指向全局 LU 因子中超节点列偏移的指针 */

    maxsuper = SUPERLU_MAX( sp_ienv(3), sp_ienv(7) );  /* 计算最大超节点大小 */
    rowblk   = sp_ienv(4);       /* 获取行块大小 */
    colblk   = sp_ienv(5);       /* 获取列块大小 */
    ldaTmp   = maxsuper + rowblk; /* 计算临时局部变量 */

    /* 
     * 对于每个非零超节点段在 U[*,j] 中的拓扑顺序
     */
    k = nseg - 1;
    for (ksub = 0; ksub < nseg; ksub++) { /* 对每个更新的超节点 */

    /* krep = 当前第 k 个超节点的代表
     * fsupc = 第一个超节点列
     * nsupc = 超节点中的列数
     * nsupr = 超节点中的行数
     */
        krep = segrep[k--];
        fsupc = xsup[supno[krep]];
        nsupc = krep - fsupc + 1;
        nsupr = xlsub[fsupc+1] - xlsub[fsupc];
        nrow = nsupr - nsupc;
        lptr = xlsub[fsupc];
        krep_ind = lptr + nsupc - 1;

        repfnz_col = repfnz;   /* 初始化指向面板中每列的非零元素的代表的指针 */
        dense_col = dense;     /* 初始化指向面板中每列的稠密部分的指针 */
    if ( nsupc >= colblk && nrow > rowblk ) { /* 检查是否符合2-D块更新条件 */

        TriTmp = tempv; /* 将tempv赋给TriTmp，作为临时变量 */

        /* Sequence through each column in panel -- triangular solves */
        for (jj = jcol; jj < jcol + w; jj++, /* 迭代处理当前面板中的每一列 -- 进行三角求解 */
         repfnz_col += m, dense_col += m, TriTmp += ldaTmp ) {

        kfnz = repfnz_col[krep]; /* 获取krep位置处的重复非零元的列索引 */
        if ( kfnz == EMPTY ) continue;    /* 如果kfnz为EMPTY，跳过该段零元 */

        segsze = krep - kfnz + 1; /* 计算非零段的大小 */
        luptr = xlusup[fsupc]; /* 获取列fsupc对应的LU因子的起始指针 */

        ops[TRSV] += segsze * (segsze - 1); /* 更新TRSV操作数 */
        ops[GEMV] += 2 * nrow * segsze; /* 更新GEMV操作数 */
    
        /* Case 1: Update U-segment of size 1 -- col-col update */
        if ( segsze == 1 ) {
            ukj = dense_col[lsub[krep_ind]]; /* 获取列krep_ind位置处的密集列元素 */
            luptr += nsupr*(nsupc-1) + nsupc; /* 更新LU因子的指针位置 */

            for (i = lptr + nsupc; i < xlsub[fsupc+1]; i++) {
            irow = lsub[i]; /* 获取行索引 */
            dense_col[irow] -= ukj * lusup[luptr]; /* 执行列-列更新 */
            ++luptr;
            }

        } else if ( segsze <= 3 ) {
            ukj = dense_col[lsub[krep_ind]]; /* 获取列krep_ind位置处的密集列元素 */
            ukj1 = dense_col[lsub[krep_ind - 1]]; /* 获取列krep_ind-1位置处的密集列元素 */
            luptr += nsupr*(nsupc-1) + nsupc-1; /* 更新LU因子的指针位置 */
            luptr1 = luptr - nsupr;

            if ( segsze == 2 ) {
            ukj -= ukj1 * lusup[luptr1]; /* 执行两列更新 */
            dense_col[lsub[krep_ind]] = ukj;
            for (i = lptr + nsupc; i < xlsub[fsupc+1]; ++i) {
                irow = lsub[i]; /* 获取行索引 */
                luptr++; luptr1++;
                dense_col[irow] -= (ukj*lusup[luptr]
                        + ukj1*lusup[luptr1]);
            }
            } else {
            ukj2 = dense_col[lsub[krep_ind - 2]]; /* 获取列krep_ind-2位置处的密集列元素 */
            luptr2 = luptr1 - nsupr;
            ukj1 -= ukj2 * lusup[luptr2-1]; /* 执行三列更新 */
            ukj = ukj - ukj1*lusup[luptr1] - ukj2*lusup[luptr2];
            dense_col[lsub[krep_ind]] = ukj;
            dense_col[lsub[krep_ind-1]] = ukj1;
            for (i = lptr + nsupc; i < xlsub[fsupc+1]; ++i) {
                irow = lsub[i]; /* 获取行索引 */
                luptr++; luptr1++; luptr2++;
                dense_col[irow] -= ( ukj*lusup[luptr]
                             + ukj1*lusup[luptr1] + ukj2*lusup[luptr2] );
            }
            }

        } else  {    /* segsze >= 4 */
            
            /* Copy U[*,j] segment from dense[*] to TriTmp[*], which
               holds the result of triangular solves.    */
            no_zeros = kfnz - fsupc; /* 计算非零元的偏移量 */
            isub = lptr + no_zeros;
            for (i = 0; i < segsze; ++i) {
            irow = lsub[isub]; /* 获取行索引 */
            TriTmp[i] = dense_col[irow]; /* 将dense_col中的元素复制到TriTmp中 */
            ++isub;
            }
            
            /* start effective triangle */
            luptr += nsupr * no_zeros + no_zeros; /* 更新LU因子的指针位置 */
        } /* else ... */
        
        }  /* for jj ... end tri-solves */

        /* Block row updates; push all the way into dense[*] block */
        for ( r_ind = 0; r_ind < nrow; r_ind += rowblk ) {
        // 逐行进行块更新，将结果推送到 dense[*] 块中
        r_hi = SUPERLU_MIN(nrow, r_ind + rowblk);
        // 计算当前块的结束行号，不超过总行数
        block_nrow = SUPERLU_MIN(rowblk, r_hi - r_ind);
        // 当前行块的行数，不超过指定的块大小，考虑剩余行数不足一块的情况
        luptr = xlusup[fsupc] + nsupc + r_ind;
        // 指向 L 矩阵中的列指针，加上偏移量 nsupc 和当前行块的起始行号
        isub1 = lptr + nsupc + r_ind;
        // isub1 指向的是 L 矩阵中的行索引，加上偏移量 nsupc 和当前行块的起始行号
        
        repfnz_col = repfnz;
        // 当前列的 repfnz 值，用于确定当前列的非零元素起始位置
        TriTmp = tempv;
        // TriTmp 是临时向量，用于存储 L 矩阵中的临时结果
        dense_col = dense;
        // dense_col 是稠密矩阵 dense 的列向量
        
        /* Sequence through each column in panel -- matrix-vector */
        // 遍历每个面板中的每一列，进行矩阵-向量乘法
        for (jj = jcol; jj < jcol + w; jj++,
             repfnz_col += m, dense_col += m, TriTmp += ldaTmp) {
        // jj 是当前处理的列索引，逐列遍历当前面板的所有列
        // repfnz_col 和 dense_col 分别指向当前列的 repfnz 和 dense 中的起始位置
            
            kfnz = repfnz_col[krep];
            // 获取当前列的 repfnz[krep] 值，即当前非零元素的起始位置
            if ( kfnz == EMPTY ) continue; /* Skip any zero segment */
            // 如果 kfnz 是 EMPTY，则跳过该列，不处理
            
            segsze = krep - kfnz + 1;
            // 计算当前非零段的长度
            if ( segsze <= 3 ) continue;   /* skip unrolled cases */
            // 如果当前非零段长度小于等于 3，则跳过，不进行处理
            
            /* Perform a block update, and scatter the result of
               matrix-vector to dense[].         */
            // 执行块更新，并将矩阵-向量乘法的结果散布到 dense[] 中
            no_zeros = kfnz - fsupc;
            // 计算当前列中的零元素个数
            luptr1 = luptr + nsupr * no_zeros;
            // 计算块更新的起始位置在 L 矩阵中的偏移
            
            MatvecTmp = &TriTmp[maxsuper];
            // MatvecTmp 是临时向量，存储矩阵-向量乘法的结果
            
#ifdef USE_VENDOR_BLAS
            alpha = one; 
                    beta = zero;
#ifdef _CRAY
            SGEMV(ftcs2, &block_nrow, &segsze, &alpha, &lusup[luptr1], 
               &nsupr, TriTmp, &incx, &beta, MatvecTmp, &incy);
#else
            dgemv_("N", &block_nrow, &segsze, &alpha, &lusup[luptr1], 
               &nsupr, TriTmp, &incx, &beta, MatvecTmp, &incy);
#endif
#else
            dmatvec(nsupr, block_nrow, segsze, &lusup[luptr1],
               TriTmp, MatvecTmp);
// 使用自定义的 dmatvec 函数计算矩阵-向量乘法
#endif

/* 
 * 将 MatvecTmp[*] 散列到 SPA dense[*] 中临时存储，
 * 以便 MatvecTmp[*] 可以在下一个块行更新中重复使用。
 * 在整个面板完成后，dense[] 将被复制到全局存储中。
 */
isub = isub1;  // 设置起始位置 isub 为 isub1
for (i = 0; i < block_nrow; i++) {  // 循环遍历每一行块
    irow = lsub[isub];  // 获取当前行的行索引
    dense_col[irow] -= MatvecTmp[i];  // 将 MatvecTmp[i] 减去到 dense_col[irow] 中
    MatvecTmp[i] = zero;  // 将 MatvecTmp[i] 置零
    ++isub;  // 移动到下一个 lsub 中的索引
}

} /* for jj ... */

} /* for each block row ... */

/* 将三角求解散列到 SPA dense[*] 中 */
repfnz_col = repfnz;  // 设置 repfnz_col 为 repfnz
TriTmp = tempv;  // 设置 TriTmp 为 tempv
dense_col = dense;  // 设置 dense_col 为 dense

for (jj = jcol; jj < jcol + w; jj++,  // 循环遍历每一列
 repfnz_col += m, dense_col += m, TriTmp += ldaTmp) {
kfnz = repfnz_col[krep];  // 获取 repfnz_col[krep] 的值
if ( kfnz == EMPTY ) continue; /* 跳过任何零段 */

segsze = krep - kfnz + 1;  // 计算段的大小
if ( segsze <= 3 ) continue; /* 跳过展开的情况 */

no_zeros = kfnz - fsupc;  // 计算没有零的数量
isub = lptr + no_zeros;  // 设置 isub 为 lptr 加上没有零的数量
for (i = 0; i < segsze; i++) {  // 循环遍历每个段
    irow = lsub[isub];  // 获取当前行的行索引
    dense_col[irow] = TriTmp[i];  // 将 TriTmp[i] 赋值给 dense_col[irow]
    TriTmp[i] = zero;  // 将 TriTmp[i] 置零
    ++isub;  // 移动到下一个 lsub 中的索引
}

} /* for jj ... */
    } else { /* 1-D block modification */
        /* 对于一维块的修改 */

        /* Sequence through each column in the panel */
        /* 遍历面板中的每一列 */

        for (jj = jcol; jj < jcol + w; jj++,
             repfnz_col += m, dense_col += m) {
            /* 循环遍历面板中的每一列，同时更新指向稀疏列和密集列的指针 */

            kfnz = repfnz_col[krep];
            if ( kfnz == EMPTY ) continue;    /* Skip any zero segment */
            /* 获取非零元素的起始索引，如果是空的（零段），则跳过 */

            segsze = krep - kfnz + 1;
            luptr = xlusup[fsupc];

            ops[TRSV] += segsze * (segsze - 1);
            ops[GEMV] += 2 * nrow * segsze;
            /* 更新操作计数器：TRSV 表示三角求解，GEMV 表示一般矩阵乘向量操作 */

            /* Case 1: Update U-segment of size 1 -- col-col update */
            /* 情况1：更新大小为1的U段 -- 列-列更新 */
            if ( segsze == 1 ) {
                ukj = dense_col[lsub[krep_ind]];
                luptr += nsupr*(nsupc-1) + nsupc;

                for (i = lptr + nsupc; i < xlsub[fsupc+1]; i++) {
                    irow = lsub[i];
                    dense_col[irow] -= ukj * lusup[luptr];
                    ++luptr;
                }

            } else if ( segsze <= 3 ) {
                ukj = dense_col[lsub[krep_ind]];
                luptr += nsupr*(nsupc-1) + nsupc-1;
                ukj1 = dense_col[lsub[krep_ind - 1]];
                luptr1 = luptr - nsupr;

                if ( segsze == 2 ) {
                    ukj -= ukj1 * lusup[luptr1];
                    dense_col[lsub[krep_ind]] = ukj;
                    for (i = lptr + nsupc; i < xlsub[fsupc+1]; ++i) {
                        irow = lsub[i];
                        ++luptr;  ++luptr1;
                        dense_col[irow] -= (ukj*lusup[luptr]
                                + ukj1*lusup[luptr1]);
                    }
                } else {
                    ukj2 = dense_col[lsub[krep_ind - 2]];
                    luptr2 = luptr1 - nsupr;
                    ukj1 -= ukj2 * lusup[luptr2-1];
                    ukj = ukj - ukj1*lusup[luptr1] - ukj2*lusup[luptr2];
                    dense_col[lsub[krep_ind]] = ukj;
                    dense_col[lsub[krep_ind-1]] = ukj1;
                    for (i = lptr + nsupc; i < xlsub[fsupc+1]; ++i) {
                        irow = lsub[i];
                        ++luptr; ++luptr1; ++luptr2;
                        dense_col[irow] -= ( ukj*lusup[luptr]
                                     + ukj1*lusup[luptr1] + ukj2*lusup[luptr2] );
                    }
                }

            } else  { /* segsze >= 4 */
                /* 
                 * Perform a triangular solve and block update,
                 * then scatter the result of sup-col update to dense[].
                 */
                /* 执行三角求解和块更新，然后将超节点列更新的结果分散到密集矩阵中 */

                no_zeros = kfnz - fsupc;
                
                /* Copy U[*,j] segment from dense[*] to tempv[*]: 
                 *    The result of triangular solve is in tempv[*];
                 *    The result of matrix vector update is in dense_col[*]
                 */
                /* 从dense[]复制U[*,j]段到tempv[]:
                 *    三角求解的结果存储在tempv[]中；
                 *    矩阵向量更新的结果存储在dense_col[]中 */

                isub = lptr + no_zeros;
                for (i = 0; i < segsze; ++i) {
                    irow = lsub[isub];
                    tempv[i] = dense_col[irow]; /* Gather */
                    ++isub;
                }
                
                /* start effective triangle */
                luptr += nsupr * no_zeros + no_zeros;
                /* 开始有效三角区域的更新 */
                
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
            STRSV( ftcs1, ftcs2, ftcs3, &segsze, &lusup[luptr], 
               &nsupr, tempv, &incx );
#else
#if SCIPY_FIX
           // 如果当前的超节点行数小于当前分段大小，则提前失败并中止程序
           if (nsupr < segsze) {
            ABORT("failed to factorize matrix");
           }
#endif
            // 使用 BLAS 函数解三角线性方程组
            dtrsv_( "L", "N", "U", &segsze, &lusup[luptr], 
               &nsupr, tempv, &incx );
#endif
            
            // 更新超节点行指针
            luptr += segsze;    /* Dense matrix-vector */
            // 设置临时向量的下一个位置
            tempv1 = &tempv[segsze];
                    alpha = one;
                    beta = zero;
#ifdef _CRAY
            // 使用 BLAS 函数进行矩阵-向量乘法
            SGEMV( ftcs2, &nrow, &segsze, &alpha, &lusup[luptr], 
               &nsupr, tempv, &incx, &beta, tempv1, &incy );
#else
            // 使用 BLAS 函数进行矩阵-向量乘法
            dgemv_( "N", &nrow, &segsze, &alpha, &lusup[luptr], 
               &nsupr, tempv, &incx, &beta, tempv1, &incy );
#endif
#else
            // 解超节点的稠密子块
            dlsolve ( nsupr, segsze, &lusup[luptr], tempv );
            
            // 更新超节点行指针
            luptr += segsze;        /* Dense matrix-vector */
            // 设置临时向量的下一个位置
            tempv1 = &tempv[segsze];
            // 计算矩阵向量乘法
            dmatvec (nsupr, nrow, segsze, &lusup[luptr], tempv, tempv1);
#endif
            
            /* 将 tempv[*] 散布到 SPA 稠密[*] 中，暂时存放，
             * 以便用于解决下一列面板的三角形问题。它们将在整个面板完成后复制到 ucol[*] 中。
             */
            // 计算 tempv[*] 到 dense_col[*] 的散布
            isub = lptr + no_zeros;
            for (i = 0; i < segsze; i++) {
            irow = lsub[isub];
            dense_col[irow] = tempv[i];
            tempv[i] = zero;
            isub++;
            }
            
            /* 将 tempv1[*] 的更新散布到 SPA 稠密[*] 中 */
            /* 开始稠密的矩形 L */
            // 计算 tempv1[*] 到 dense_col[*] 的散布
            for (i = 0; i < nrow; i++) {
            irow = lsub[isub];
            dense_col[irow] -= tempv1[i];
            tempv1[i] = zero;
            ++isub;    
            }
            
        } /* else segsze>=4 ... */
        
        } /* for each column in the panel... */
        
    } /* else 1-D update ... */

    } /* for each updating supernode ... */

}
```