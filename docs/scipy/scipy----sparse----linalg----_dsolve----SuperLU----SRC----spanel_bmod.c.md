# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\spanel_bmod.c`

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
 * @file spanel_bmod.c
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

#include <stdio.h>
#include <stdlib.h>
#include "slu_sdefs.h"

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
spanel_bmod (
        const int  m,          /* in - number of rows in the matrix */
        const int  w,          /* in */
        const int  jcol,       /* in */
        const int  nseg,       /* in */
        float     *dense,     /* out, of size n by w */
        float     *tempv,     /* working array */
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
    float       alpha, beta;
#endif

    register int k, ksub;
    int          fsupc, nsupc, nsupr, nrow;
    int          krep, krep_ind;
    float       ukj, ukj1, ukj2;
    int_t        luptr, luptr1, luptr2;
    int          segsze;
    int          block_nrow;  /* no of rows in a block row */
    int_t        lptr;          /* Points to the row subscripts of a supernode */
    int          kfnz, irow, no_zeros; 
    register int isub, isub1, i;
    register int jj;          /* Index through each column in the panel */
    int          *xsup, *supno;
    int_t        *lsub, *xlsub;
    float       *lusup;
    int_t        *xlusup;
    int          *repfnz_col; /* repfnz[] for a column in the panel */
    float       *dense_col;  /* dense[] for a column in the panel */
    float       *tempv1;             /* Used in 1-D update */
    float       *TriTmp, *MatvecTmp; /* used in 2-D update */
    float      zero = 0.0;
    float      one = 1.0;
    register int ldaTmp;
    register int r_ind, r_hi;
    int  maxsuper, rowblk, colblk;
    flops_t  *ops = stat->ops;
    
    xsup    = Glu->xsup;
    supno   = Glu->supno;
    lsub    = Glu->lsub;
    xlsub   = Glu->xlsub;
    lusup   = (float *) Glu->lusup;
    xlusup  = Glu->xlusup;
    
    maxsuper = SUPERLU_MAX( sp_ienv(3), sp_ienv(7) );
    rowblk   = sp_ienv(4);
    colblk   = sp_ienv(5);
    ldaTmp   = maxsuper + rowblk;

    /* 
     * For each nonz supernode segment of U[*,j] in topological order 
     */
    k = nseg - 1;
    for (ksub = 0; ksub < nseg; ksub++) { /* for each updating supernode */

        /* krep = representative of current k-th supernode
         * fsupc = first supernodal column
         * nsupc = no of columns in a supernode
         * nsupr = no of rows in a supernode
         */
        krep = segrep[k--];
        fsupc = xsup[supno[krep]];
        nsupc = krep - fsupc + 1;
        nsupr = xlsub[fsupc+1] - xlsub[fsupc];
        nrow = nsupr - nsupc;
        lptr = xlsub[fsupc];
        krep_ind = lptr + nsupc - 1;

        repfnz_col = repfnz;
        dense_col = dense;
    if ( nsupc >= colblk && nrow > rowblk ) { /* 2-D block update */
        // 检查是否满足2D块更新的条件：当前列索引大于等于块列索引且行数大于块行数

        TriTmp = tempv;
        // 将tempv赋值给TriTmp，用于存储临时向量

        /* Sequence through each column in panel -- triangular solves */
        // 遍历面板中的每一列，进行三角求解操作
        for (jj = jcol; jj < jcol + w; jj++,
             repfnz_col += m, dense_col += m, TriTmp += ldaTmp ) {

            kfnz = repfnz_col[krep];
            // 找到列的第一个非零元素的位置

            if ( kfnz == EMPTY ) continue;    /* Skip any zero segment */
            // 如果第一个非零元素位置为EMPTY，跳过该段（即跳过全零段）

            segsze = krep - kfnz + 1;
            // 计算当前列的非零元素段长度

            luptr = xlusup[fsupc];
            // 获取LU分解中U部分对应列的起始位置

            ops[TRSV] += segsze * (segsze - 1);
            // 更新向量操作数（TRSV）计数：对角线求解操作的次数
            ops[GEMV] += 2 * nrow * segsze;
            // 更新向量操作数（GEMV）计数：矩阵向量乘法操作的次数

            /* Case 1: Update U-segment of size 1 -- col-col update */
            // 情况1：更新大小为1的U段 —— 列-列更新
            if ( segsze == 1 ) {
                ukj = dense_col[lsub[krep_ind]];
                // 获取当前列中对应的ukj元素
                luptr += nsupr*(nsupc-1) + nsupc;

                for (i = lptr + nsupc; i < xlsub[fsupc+1]; i++) {
                    irow = lsub[i];
                    dense_col[irow] -= ukj * lusup[luptr];
                    ++luptr;
                }

            } else if ( segsze <= 3 ) {
                ukj = dense_col[lsub[krep_ind]];
                ukj1 = dense_col[lsub[krep_ind - 1]];
                luptr += nsupr*(nsupc-1) + nsupc-1;
                luptr1 = luptr - nsupr;

                if ( segsze == 2 ) {
                    ukj -= ukj1 * lusup[luptr1];
                    dense_col[lsub[krep_ind]] = ukj;
                    for (i = lptr + nsupc; i < xlsub[fsupc+1]; ++i) {
                        irow = lsub[i];
                        luptr++; luptr1++;
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
                        luptr++; luptr1++; luptr2++;
                        dense_col[irow] -= ( ukj*lusup[luptr]
                                             + ukj1*lusup[luptr1] + ukj2*lusup[luptr2] );
                    }
                }

            } else  {    /* segsze >= 4 */

                /* Copy U[*,j] segment from dense[*] to TriTmp[*], which
                   holds the result of triangular solves.    */
                // 将dense[*]中的U[*,j]段复制到TriTmp[*]中，TriTmp[*]保存三角求解的结果
                no_zeros = kfnz - fsupc;
                isub = lptr + no_zeros;
                for (i = 0; i < segsze; ++i) {
                    irow = lsub[isub];
                    TriTmp[i] = dense_col[irow]; /* Gather */
                    ++isub;
                }

                /* start effective triangle */
                // 开始有效的三角形区域
                luptr += nsupr * no_zeros + no_zeros;
            #ifdef USE_VENDOR_BLAS
            #ifdef _CRAY
            调用特定于 CRAY 平台的 STRSV 函数解决线性方程组
            STRSV( ftcs1, ftcs2, ftcs3, &segsze, &lusup[luptr], 
               &nsupr, TriTmp, &incx );
            #else
            调用通用的 strsv_ 函数解决线性方程组
            strsv_( "L", "N", "U", &segsze, &lusup[luptr], 
               &nsupr, TriTmp, &incx );
            #endif
            #else        
            若未定义 USE_VENDOR_BLAS，则调用自定义的 slsolve 函数解决线性方程组
            slsolve ( nsupr, segsze, &lusup[luptr], TriTmp );
            #endif

        } /* else ... */
        
        }  /* for jj ... end tri-solves */

        /* Block row updates; push all the way into dense[*] block */
        for ( r_ind = 0; r_ind < nrow; r_ind += rowblk ) {
        
        r_hi = SUPERLU_MIN(nrow, r_ind + rowblk);
        block_nrow = SUPERLU_MIN(rowblk, r_hi - r_ind);
        luptr = xlusup[fsupc] + nsupc + r_ind;
        isub1 = lptr + nsupc + r_ind;
        
        repfnz_col = repfnz;
        TriTmp = tempv;
        dense_col = dense;
        
        /* Sequence through each column in panel -- matrix-vector */
        for (jj = jcol; jj < jcol + w; jj++,
             repfnz_col += m, dense_col += m, TriTmp += ldaTmp) {
            
            kfnz = repfnz_col[krep];
            if ( kfnz == EMPTY ) continue; /* Skip any zero segment */
            
            segsze = krep - kfnz + 1;
            if ( segsze <= 3 ) continue;   /* skip unrolled cases */
            
            /* Perform a block update, and scatter the result of
               matrix-vector to dense[].         */
            no_zeros = kfnz - fsupc;
            luptr1 = luptr + nsupr * no_zeros;
            MatvecTmp = &TriTmp[maxsuper];
            
            #ifdef USE_VENDOR_BLAS
            alpha = one; 
            beta = zero;
            #ifdef _CRAY
            调用特定于 CRAY 平台的 SGEMV 函数执行矩阵向量乘法
            SGEMV(ftcs2, &block_nrow, &segsze, &alpha, &lusup[luptr1], 
               &nsupr, TriTmp, &incx, &beta, MatvecTmp, &incy);
            #else
            调用通用的 sgemv_ 函数执行矩阵向量乘法
            sgemv_("N", &block_nrow, &segsze, &alpha, &lusup[luptr1], 
               &nsupr, TriTmp, &incx, &beta, MatvecTmp, &incy);
            #endif
            #else
            若未定义 USE_VENDOR_BLAS，则调用自定义的 smatvec 函数执行矩阵向量乘法
            smatvec(nsupr, block_nrow, segsze, &lusup[luptr1],
               TriTmp, MatvecTmp);
            #endif
#endif

/* 将 MatvecTmp[*] 散布到 SPA dense[*] 中，暂时存储，
 * 以便 MatvecTmp[*] 可以用于下一个块行更新。
 * 在整个面板完成后，dense[] 将被复制到全局存储中。
 */
isub = isub1;
for (i = 0; i < block_nrow; i++) {
    irow = lsub[isub];
    dense_col[irow] -= MatvecTmp[i];
    MatvecTmp[i] = zero;
    ++isub;
}

} /* for jj ... */

} /* for each block row ... */

/* 将三角求解结果散布到 SPA dense[*] 中 */
repfnz_col = repfnz;
TriTmp = tempv;
dense_col = dense;

for (jj = jcol; jj < jcol + w; jj++,
     repfnz_col += m, dense_col += m, TriTmp += ldaTmp) {
    kfnz = repfnz_col[krep];
    if ( kfnz == EMPTY ) continue; /* 跳过任何零段 */

    segsze = krep - kfnz + 1;
    if ( segsze <= 3 ) continue; /* 跳过展开的情况 */

    no_zeros = kfnz - fsupc;        
    isub = lptr + no_zeros;
    for (i = 0; i < segsze; i++) {
        irow = lsub[isub];
        dense_col[irow] = TriTmp[i];
        TriTmp[i] = zero;
        ++isub;
    }

} /* for jj ... */
    } else { /* 1-D block modification */
        /* 进入1维块修改的分支 */

        /* Sequence through each column in the panel */
        /* 遍历面板中的每一列 */
        for (jj = jcol; jj < jcol + w; jj++,
             repfnz_col += m, dense_col += m) {
            /* 对面板中的每一列进行循环 */

            kfnz = repfnz_col[krep];
            if ( kfnz == EMPTY ) continue;    /* Skip any zero segment */
            /* 获取 repfnz_col 中的 krep 处索引值，并检查是否为 EMPTY，如果是则跳过 */

            segsze = krep - kfnz + 1;
            luptr = xlusup[fsupc];

            ops[TRSV] += segsze * (segsze - 1);
            ops[GEMV] += 2 * nrow * segsze;
            /* 更新 TRSV 和 GEMV 操作计数 */

            /* Case 1: Update U-segment of size 1 -- col-col update */
            /* 情况1：更新大小为1的U段 - 列-列更新 */
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
                /* 执行三角求解和块更新，然后将超节点列更新的结果分散到 dense[] 中 */

                no_zeros = kfnz - fsupc;
                
                /* Copy U[*,j] segment from dense[*] to tempv[*]: 
                 *    The result of triangular solve is in tempv[*];
                 *    The result of matrix vector update is in dense_col[*]
                 */
                /* 将 dense[*] 中的 U[*,j] 段复制到 tempv[*]：
                 *    三角求解的结果存储在 tempv[*] 中；
                 *    矩阵向量更新的结果存储在 dense_col[*] 中
                 */
                isub = lptr + no_zeros;
                for (i = 0; i < segsze; ++i) {
                    irow = lsub[isub];
                    tempv[i] = dense_col[irow]; /* Gather */
                    ++isub;
                }
                
                /* start effective triangle */
                luptr += nsupr * no_zeros + no_zeros;
                /* 开始有效的三角操作 */
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
            STRSV( ftcs1, ftcs2, ftcs3, &segsze, &lusup[luptr], 
               &nsupr, tempv, &incx );
#else
#if SCIPY_FIX
           if (nsupr < segsze) {
            /* 提前失败，而不是将无效参数传递给 TRSV。 */
            ABORT("failed to factorize matrix");
           }
#endif
            strsv_( "L", "N", "U", &segsze, &lusup[luptr], 
               &nsupr, tempv, &incx );
#endif
            
            luptr += segsze;    /* 密集矩阵-向量乘法 */
            tempv1 = &tempv[segsze];
                    alpha = one;
                    beta = zero;
#ifdef _CRAY
            SGEMV( ftcs2, &nrow, &segsze, &alpha, &lusup[luptr], 
               &nsupr, tempv, &incx, &beta, tempv1, &incy );
#else
            sgemv_( "N", &nrow, &segsze, &alpha, &lusup[luptr], 
               &nsupr, tempv, &incx, &beta, tempv1, &incy );
#endif
#else
            slsolve ( nsupr, segsze, &lusup[luptr], tempv );
            
            luptr += segsze;        /* 密集矩阵-向量乘法 */
            tempv1 = &tempv[segsze];
            smatvec (nsupr, nrow, segsze, &lusup[luptr], tempv, tempv1);
#endif
            
            /* 将 tempv[*] 散布到 SPA dense[*] 中，临时存储，
             * 这样 tempv[*] 可以用于下一列面板的三角求解。
             * 在整个面板完成后，它们将被复制到 ucol[*] 中。
             */
            isub = lptr + no_zeros;
            for (i = 0; i < segsze; i++) {
            irow = lsub[isub];
            dense_col[irow] = tempv[i];
            tempv[i] = zero;
            isub++;
            }
            
            /* 将 tempv1[*] 中的更新散布到 SPA dense[*] 中 */
            /* 开始密集矩形 L */
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