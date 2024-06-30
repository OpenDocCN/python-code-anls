# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zpanel_bmod.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file zpanel_bmod.c
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
#include "slu_zdefs.h"

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
zpanel_bmod (
        const int  m,          /* in - number of rows in the matrix */
        const int  w,          /* in */
        const int  jcol,       /* in */
        const int  nseg,       /* in */
        doublecomplex     *dense,     /* out, of size n by w */
        doublecomplex     *tempv,     /* working array */
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
    doublecomplex       alpha, beta;
#endif

    register int k, ksub;
    int          fsupc, nsupc, nsupr, nrow;
    int          krep, krep_ind;
    doublecomplex       ukj, ukj1, ukj2;
    int_t        luptr, luptr1, luptr2;
    int          segsze;
    int          block_nrow;  /* no of rows in a block row */
    int_t        lptr;          /* Points to the row subscripts of a supernode */
    int          kfnz, irow, no_zeros; 
    register int isub, isub1, i;
    register int jj;          /* Index through each column in the panel */
    int          *xsup, *supno;
    int_t        *lsub, *xlsub;
    doublecomplex       *lusup;
    int_t        *xlusup;
    int          *repfnz_col; /* repfnz[] for a column in the panel */
    doublecomplex       *dense_col;  /* dense[] for a column in the panel */
    doublecomplex       *tempv1;             /* Used in 1-D update */
    doublecomplex       *TriTmp, *MatvecTmp; /* used in 2-D update */
    doublecomplex      zero = {0.0, 0.0};
    doublecomplex      one = {1.0, 0.0};
    doublecomplex      comp_temp, comp_temp1;
    register int ldaTmp;
    register int r_ind, r_hi;
    int  maxsuper, rowblk, colblk;
    flops_t  *ops = stat->ops;
    
    xsup    = Glu->xsup;    /* Array of size nsupers, supernode starting column */
    supno   = Glu->supno;   /* Array of size nsuper, supernode numbers */
    lsub    = Glu->lsub;    /* Array containing the row indices of nonzeros */
    xlsub   = Glu->xlsub;   /* Starting position of each supernode in lsub */
    lusup   = (doublecomplex *) Glu->lusup;  /* Complex values of L and U factors */
    xlusup  = Glu->xlusup;  /* Starting position of each supernode in lusup */
    
    maxsuper = SUPERLU_MAX( sp_ienv(3), sp_ienv(7) );  /* Maximum supernode size from environment */
    rowblk   = sp_ienv(4);  /* Row blocking size from environment */
    colblk   = sp_ienv(5);  /* Column blocking size from environment */
    ldaTmp   = maxsuper + rowblk;  /* Leading dimension for temporary matrices */
    
    /* 
     * For each nonz supernode segment of U[*,j] in topological order 
     */
    k = nseg - 1;  /* Initialize k for looping through supernode segments */
    for (ksub = 0; ksub < nseg; ksub++) { /* for each updating supernode */
    
        /* krep = representative of current k-th supernode
         * fsupc = first supernodal column
         * nsupc = no of columns in a supernode
         * nsupr = no of rows in a supernode
         */
        krep = segrep[k--];  /* Representative of current supernode segment */
        fsupc = xsup[supno[krep]];  /* First supernodal column of the current supernode */
        nsupc = krep - fsupc + 1;   /* Number of columns in the current supernode */
        nsupr = xlsub[fsupc+1] - xlsub[fsupc];  /* Number of rows in the current supernode */
        nrow = nsupr - nsupc;  /* Number of extra rows beyond the diagonal block */
        lptr = xlsub[fsupc];   /* Starting position of the current supernode in lsub */
        krep_ind = lptr + nsupc - 1;  /* Last row in the current supernode's diagonal block */
        
        repfnz_col = repfnz;   /* Pointer to repfnz[] for the current column in the panel */
        dense_col = dense;     /* Pointer to dense[] for the current column in the panel */
        # 如果定义了 USE_VENDOR_BLAS 宏，则执行以下代码块
#ifdef USE_VENDOR_BLAS
        # 如果在 Cray 平台上，则使用 CTRSV 函数进行三角求解
#ifdef _CRAY
            CTRSV( ftcs1, ftcs2, ftcs3, &segsze, &lusup[luptr], 
               &nsupr, TriTmp, &incx );
#else
            # 否则，使用 ztrsv_ 函数进行三角求解
            ztrsv_( "L", "N", "U", &segsze, &lusup[luptr], 
               &nsupr, TriTmp, &incx );
#endif
#else        
        # 如果未定义 USE_VENDOR_BLAS 宏，则调用 zlsolve 函数进行线性系统的求解
            zlsolve ( nsupr, segsze, &lusup[luptr], TriTmp );
#endif

        } /* else ... */
        
        }  /* for jj ... end tri-solves */

        /* Block row updates; push all the way into dense[*] block */
        # 对块行进行更新，将结果推入 dense[*] 块中
        for ( r_ind = 0; r_ind < nrow; r_ind += rowblk ) {
        
        r_hi = SUPERLU_MIN(nrow, r_ind + rowblk);
        block_nrow = SUPERLU_MIN(rowblk, r_hi - r_ind);
        luptr = xlusup[fsupc] + nsupc + r_ind;
        isub1 = lptr + nsupc + r_ind;
        
        repfnz_col = repfnz;
        TriTmp = tempv;
        dense_col = dense;
        
        /* Sequence through each column in panel -- matrix-vector */
        # 对面板中的每一列进行迭代 -- 矩阵-向量乘法
        for (jj = jcol; jj < jcol + w; jj++,
             repfnz_col += m, dense_col += m, TriTmp += ldaTmp) {
            
            kfnz = repfnz_col[krep];
            if ( kfnz == EMPTY ) continue; /* 跳过任何零段 */
            
            segsze = krep - kfnz + 1;
            if ( segsze <= 3 ) continue;   /* 跳过展开的情况 */
            
            /* Perform a block update, and scatter the result of
               matrix-vector to dense[].         */
            # 执行块更新，并将矩阵-向量乘法的结果散布到 dense[] 中
            no_zeros = kfnz - fsupc;
            luptr1 = luptr + nsupr * no_zeros;
            MatvecTmp = &TriTmp[maxsuper];
            
#ifdef USE_VENDOR_BLAS
            alpha = one; 
                    beta = zero;
#ifdef _CRAY
            # 如果在 Cray 平台上，则使用 CGEMV 函数进行矩阵-向量乘法
            CGEMV(ftcs2, &block_nrow, &segsze, &alpha, &lusup[luptr1], 
               &nsupr, TriTmp, &incx, &beta, MatvecTmp, &incy);
#else
            # 否则，使用 zgemv_ 函数进行矩阵-向量乘法
            zgemv_("N", &block_nrow, &segsze, &alpha, &lusup[luptr1], 
               &nsupr, TriTmp, &incx, &beta, MatvecTmp, &incy);
#endif
#else
            # 如果未定义 USE_VENDOR_BLAS 宏，则调用 zmatvec 函数执行矩阵-向量乘法
            zmatvec(nsupr, block_nrow, segsze, &lusup[luptr1],
               TriTmp, MatvecTmp);
#endif
#endif

            /* Scatter MatvecTmp[*] into SPA dense[*] temporarily
             * such that MatvecTmp[*] can be re-used for the
             * the next blok row update. dense[] will be copied into 
             * global store after the whole panel has been finished.
             */
            isub = isub1;
            // 循环遍历当前块的行
            for (i = 0; i < block_nrow; i++) {
                irow = lsub[isub];
                // 将 MatvecTmp[i] 散布到 dense_col[irow] 中
                z_sub(&dense_col[irow], &dense_col[irow], 
                              &MatvecTmp[i]);
                // 清空 MatvecTmp[i]，以便下一个块行更新时重用
                MatvecTmp[i] = zero;
                ++isub;
            }
            
        } /* for jj ... */
        
        } /* for each block row ... */
        
        /* Scatter the triangular solves into SPA dense[*] */
        repfnz_col = repfnz;
        TriTmp = tempv;
        dense_col = dense;
        
        // 遍历当前列的所有块
        for (jj = jcol; jj < jcol + w; jj++,
         repfnz_col += m, dense_col += m, TriTmp += ldaTmp) {
            kfnz = repfnz_col[krep];
            // 若 kfnz 为 EMPTY，则跳过零段
            if ( kfnz == EMPTY ) continue;
            
            segsze = krep - kfnz + 1;
            // 若块大小小于等于 3，则跳过展开的情况
            if ( segsze <= 3 ) continue;
            
            no_zeros = kfnz - fsupc;        
            isub = lptr + no_zeros;
            // 将 TriTmp 中的数据散布到 dense_col 中
            for (i = 0; i < segsze; i++) {
                irow = lsub[isub];
                dense_col[irow] = TriTmp[i];
                TriTmp[i] = zero;
                ++isub;
            }
            
        } /* for jj ... */
        
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
            CTRSV( ftcs1, ftcs2, ftcs3, &segsze, &lusup[luptr], 
               &nsupr, tempv, &incx );
#else
#if SCIPY_FIX
           if (nsupr < segsze) {
            /* Fail early rather than passing in invalid parameters to TRSV. */
            ABORT("failed to factorize matrix");
           }
#endif
            // 解三角方程 L * x = b
            ztrsv_( "L", "N", "U", &segsze, &lusup[luptr], 
               &nsupr, tempv, &incx );
#endif
            
            // luptr 向前推进 segsze 步，用于稠密矩阵-向量乘法
            luptr += segsze;    /* Dense matrix-vector */
            tempv1 = &tempv[segsze];
            alpha = one;
            beta = zero;
#ifdef _CRAY
            CGEMV( ftcs2, &nrow, &segsze, &alpha, &lusup[luptr], 
               &nsupr, tempv, &incx, &beta, tempv1, &incy );
#else
            // 计算稠密矩阵-向量乘积
            zgemv_( "N", &nrow, &segsze, &alpha, &lusup[luptr], 
               &nsupr, tempv, &incx, &beta, tempv1, &incy );
#endif
#else
            // 解稠密方程组 L * x = b
            zlsolve ( nsupr, segsze, &lusup[luptr], tempv );
            
            // luptr 向前推进 segsze 步，用于稠密矩阵-向量乘法
            luptr += segsze;        /* Dense matrix-vector */
            tempv1 = &tempv[segsze];
            // 计算稠密矩阵-向量乘积
            zmatvec (nsupr, nrow, segsze, &lusup[luptr], tempv, tempv1);
#endif
#endif
            
            /* Scatter tempv[*] into SPA dense[*] temporarily, such
             * that tempv[*] can be used for the triangular solve of
             * the next column of the panel. They will be copied into 
             * ucol[*] after the whole panel has been finished.
             */
            // 计算子列的起始索引
            isub = lptr + no_zeros;
            // 将tempv[*]散布到SPA dense[*]中，为面板下一列的三角求解做准备
            for (i = 0; i < segsze; i++) {
                irow = lsub[isub];
                dense_col[irow] = tempv[i];
                tempv[i] = zero;
                isub++;
            }
            
            /* Scatter the update from tempv1[*] into SPA dense[*] */
            /* Start dense rectangular L */
            // 将tempv1[*]的更新散布到SPA dense[*]中
            for (i = 0; i < nrow; i++) {
                irow = lsub[isub];
                // 执行复杂数减法操作
                z_sub(&dense_col[irow], &dense_col[irow], &tempv1[i]);
                tempv1[i] = zero;
                ++isub;    
            }
            
        } /* else segsze>=4 ... */
        
    } /* for each column in the panel... */
        
} /* else 1-D update ... */

} /* for each updating supernode ... */

// 函数结束
```