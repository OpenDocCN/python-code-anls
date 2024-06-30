# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\cpanel_bmod.c`

```
/*
 * 文件头部版权声明和许可信息
 * 该文件受BSD许可协议保护，详情请参阅顶层目录下的License.txt文件
 */

#include <stdio.h>
#include <stdlib.h>
#include "slu_cdefs.h"

/*! \brief
 *
 * <pre>
 * 目的
 * =======
 *
 *    执行数值块更新（超级面板）按拓扑顺序。
 *    它包括：列-列、2列-列、3列-列和超级列-列的更新。
 *    特别处理L\\U[*,j]的超节点部分。
 *
 *    进入此例程之前，面板中的原始非零元素已经复制到spa[m,w]中。
 *
 *    更新/输出参数-
 *    dense[0:m-1,w]: L[*,j:j+w-1] 和 U[*,j:j+w-1] 以m-by-w向量的形式返回。
 * </pre>
 */
void
cpanel_bmod (
        const int  m,          /* 输入 - 矩阵中的行数 */
        const int  w,          /* 输入 */
        const int  jcol,       /* 输入 */
        const int  nseg,       /* 输入 */
        singlecomplex     *dense,     /* 输出，大小为 n by w */
        singlecomplex     *tempv,     /* 工作数组 */
        int        *segrep,    /* 输入 */
        int        *repfnz,    /* 输入，大小为 n by w */
        GlobalLU_t *Glu,       /* 修改 */
        SuperLUStat_t *stat    /* 输出 */
        )
{

#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
    _fcd ftcs1 = _cptofcd("L", strlen("L")),
         ftcs2 = _cptofcd("N", strlen("N")),
         ftcs3 = _cptofcd("U", strlen("U"));
#endif
    int          incx = 1, incy = 1;
    singlecomplex       alpha, beta;
#endif

    register int k, ksub;
    int          fsupc, nsupc, nsupr, nrow;
    int          krep, krep_ind;
    singlecomplex       ukj, ukj1, ukj2;
    int_t        luptr, luptr1, luptr2;
    int          segsze;
    int          block_nrow;  /* 一个块行中的行数 */
    int_t        lptr;          /* 指向超节点的行下标 */
    int          kfnz, irow, no_zeros; 
    register int isub, isub1, i;
    register int jj;          /* 面板中每列的索引 */
    int          *xsup, *supno;
    int_t        *lsub, *xlsub;
    singlecomplex       *lusup;
    int_t        *xlusup;
    int          *repfnz_col; /* 面板中某列的 repfnz[] */
    singlecomplex       *dense_col;  /* 面板中某列的 dense[] */
    singlecomplex       *tempv1;             /* 用于一维更新 */
    singlecomplex       *TriTmp, *MatvecTmp; /* 用于二维更新 */
    singlecomplex      zero = {0.0, 0.0};
    singlecomplex      one = {1.0, 0.0};
    singlecomplex      comp_temp, comp_temp1;
    register int ldaTmp;
    register int r_ind, r_hi;
    int  maxsuper, rowblk, colblk;
    flops_t  *ops = stat->ops;
    
    xsup    = Glu->xsup;   /* 全局 LU 因子的超节点开始索引 */
    supno   = Glu->supno;  /* 全局 LU 因子的超节点编号 */
    lsub    = Glu->lsub;   /* LU 因子的行索引 */
    xlsub   = Glu->xlsub;  /* 行索引数组的指针 */
    lusup   = (singlecomplex *) Glu->lusup;  /* LU 因子的数值 */
    xlusup  = Glu->xlusup; /* 数值数组的指针 */
    
    maxsuper = SUPERLU_MAX( sp_ienv(3), sp_ienv(7) );  /* 最大超节点大小 */
    rowblk   = sp_ienv(4);   /* 行块大小 */
    colblk   = sp_ienv(5);   /* 列块大小 */
    ldaTmp   = maxsuper + rowblk;  /* 临时的 lda 大小 */

    /* 
     * 对于每个非零超节点段，在拓扑顺序中处理 U[*,j] 
     */
    k = nseg - 1;
    for (ksub = 0; ksub < nseg; ksub++) { /* 对每个更新超节点 */

    /* krep = 当前第 k 个超节点的代表
     * fsupc = 第一个超节点列
     * nsupc = 超节点中的列数
     * nsupr = 超节点中的行数
     */
        krep = segrep[k--];
        fsupc = xsup[supno[krep]];  /* 第一个超节点列的索引 */
        nsupc = krep - fsupc + 1;   /* 超节点中的列数 */
        nsupr = xlsub[fsupc+1] - xlsub[fsupc];  /* 超节点中的行数 */
        nrow = nsupr - nsupc;   /* 行数减去列数 */
        lptr = xlsub[fsupc];    /* 超节点行索引的起始位置 */
        krep_ind = lptr + nsupc - 1;  /* krep 的索引位置 */

        repfnz_col = repfnz;    /* 面板中某列的 repfnz[] */
        dense_col = dense;      /* 面板中某列的 dense[] */
        } /* else ... */
        
        }  /* for jj ... end tri-solves */

        /* Block row updates; push all the way into dense[*] block */
        for ( r_ind = 0; r_ind < nrow; r_ind += rowblk ) {
            // 计算当前行块的结束索引，确保不超过总行数
            r_hi = SUPERLU_MIN(nrow, r_ind + rowblk);
            // 计算当前行块的大小，确保不超过指定的行块大小
            block_nrow = SUPERLU_MIN(rowblk, r_hi - r_ind);
            // 计算当前非零LU因子的起始索引
            luptr = xlusup[fsupc] + nsupc + r_ind;
            // 计算行的起始索引
            isub1 = lptr + nsupc + r_ind;
            
            // 复制行的非零元素列指标
            repfnz_col = repfnz;
            // 暂存TriTmp的起始地址
            TriTmp = tempv;
            // 密集矩阵的起始地址
            dense_col = dense;
            
            /* Sequence through each column in panel -- matrix-vector */
            for (jj = jcol; jj < jcol + w; jj++,
                 repfnz_col += m, dense_col += m, TriTmp += ldaTmp) {
                
                // 获取krep位置的列指标
                kfnz = repfnz_col[krep];
                // 如果kfnz为空，则跳过该零段
                if ( kfnz == EMPTY ) continue; /* Skip any zero segment */
                
                // 计算零段大小
                segsze = krep - kfnz + 1;
                // 如果零段大小小于等于3，则跳过未展开的情况
                if ( segsze <= 3 ) continue;   /* skip unrolled cases */
                
                /* Perform a block update, and scatter the result of
                   matrix-vector to dense[].         */
                // 计算零段起始非零LU因子的索引
                no_zeros = kfnz - fsupc;
                // 计算luptr1的值
                luptr1 = luptr + nsupr * no_zeros;
                // 计算MatvecTmp的起始地址
                MatvecTmp = &TriTmp[maxsuper];
                
#ifdef USE_VENDOR_BLAS
                // 设置alpha和beta值
                alpha = one; 
                beta = zero;
#ifdef _CRAY
                // 使用CRAY BLAS实现CGEMV运算
                CGEMV(ftcs2, &block_nrow, &segsze, &alpha, &lusup[luptr1], 
                   &nsupr, TriTmp, &incx, &beta, MatvecTmp, &incy);
#else
                // 使用标准BLAS实现CGEMV运算
                cgemv_("N", &block_nrow, &segsze, &alpha, &lusup[luptr1], 
                   &nsupr, TriTmp, &incx, &beta, MatvecTmp, &incy);
#endif
#else
                // 使用自定义的矩阵向量乘法函数cmatvec
                cmatvec(nsupr, block_nrow, segsze, &lusup[luptr1],
                   TriTmp, MatvecTmp);
#endif
#endif
            
            /* Scatter MatvecTmp[*] into SPA dense[*] temporarily
             * such that MatvecTmp[*] can be re-used for the
             * the next blok row update. dense[] will be copied into 
             * global store after the whole panel has been finished.
             */
            isub = isub1;
            // 将 MatvecTmp[*] 分散到 SPA dense[*] 中，临时存储
            // 这样 MatvecTmp[*] 可以在下一个块行更新时被重用
            // dense[] 将在整个面板完成后复制到全局存储中
            for (i = 0; i < block_nrow; i++) {
                irow = lsub[isub];
                // 调用 c_sub 函数对 dense_col[irow] 和 MatvecTmp[i] 进行计算
                c_sub(&dense_col[irow], &dense_col[irow], &MatvecTmp[i]);
                MatvecTmp[i] = zero;  // 将 MatvecTmp[i] 置零
                ++isub;
            }
            
        } /* for jj ... */
        
        } /* for each block row ... */
        
        /* Scatter the triangular solves into SPA dense[*] */
        repfnz_col = repfnz;
        TriTmp = tempv;
        dense_col = dense;
        
        for (jj = jcol; jj < jcol + w; jj++,
         repfnz_col += m, dense_col += m, TriTmp += ldaTmp) {
        kfnz = repfnz_col[krep];
        if ( kfnz == EMPTY ) continue; /* Skip any zero segment */
        
        segsze = krep - kfnz + 1;
        if ( segsze <= 3 ) continue; /* skip unrolled cases */
        
        no_zeros = kfnz - fsupc;        
        isub = lptr + no_zeros;
        // 将三角求解结果分散到 SPA dense[*] 中
        for (i = 0; i < segsze; i++) {
            irow = lsub[isub];
            dense_col[irow] = TriTmp[i];
            TriTmp[i] = zero;  // 将 TriTmp[i] 置零
            ++isub;
        }
        
        } /* for jj ... */
        
#ifdef USE_VENDOR_BLAS
#ifdef _CRAY
            // 调用 CTRSV 函数进行向量的三角求解
            CTRSV( ftcs1, ftcs2, ftcs3, &segsze, &lusup[luptr], 
               &nsupr, tempv, &incx );
#else
#if SCIPY_FIX
           if (nsupr < segsze) {
            /* Fail early rather than passing in invalid parameters to TRSV. */
            ABORT("failed to factorize matrix");
           }
#endif
            // 调用 ctrsv_ 函数进行向量的三角求解
            ctrsv_( "L", "N", "U", &segsze, &lusup[luptr], 
               &nsupr, tempv, &incx );
#endif
            
            luptr += segsze;    /* Dense matrix-vector */
            // 更新 tempv1 指针，指向 tempv[segsze] 的位置
            tempv1 = &tempv[segsze];
                    alpha = one;
                    beta = zero;
#ifdef _CRAY
            // 调用 CGEMV 函数进行矩阵-向量乘法
            CGEMV( ftcs2, &nrow, &segsze, &alpha, &lusup[luptr], 
               &nsupr, tempv, &incx, &beta, tempv1, &incy );
#else
            // 调用 cgemv_ 函数进行矩阵-向量乘法
            cgemv_( "N", &nrow, &segsze, &alpha, &lusup[luptr], 
               &nsupr, tempv, &incx, &beta, tempv1, &incy );
#endif
#else
            // 调用 clsolve 函数进行向量的三角求解
            clsolve ( nsupr, segsze, &lusup[luptr], tempv );
            
            luptr += segsze;        /* Dense matrix-vector */
            // 更新 tempv1 指针，指向 tempv[segsze] 的位置
            tempv1 = &tempv[segsze];
            // 调用 cmatvec 函数进行矩阵-向量乘法
            cmatvec (nsupr, nrow, segsze, &lusup[luptr], tempv, tempv1);

                    alpha = one;
                    beta = zero;
#ifdef _CRAY
            // 调用 CGEMV 函数进行矩阵-向量乘法
            CGEMV( ftcs2, &nrow, &segsze, &alpha, &lusup[luptr], 
               &nsupr, tempv, &incx, &beta, tempv1, &incy );
#else
            // 调用 cgemv_ 函数进行矩阵-向量乘法
            cgemv_( "N", &nrow, &segsze, &alpha, &lusup[luptr], 
               &nsupr, tempv, &incx, &beta, tempv1, &incy );
#endif
#else
            // 调用 clsolve 函数进行向量的三角求解
            clsolve ( nsupr, segsze, &lusup[luptr], tempv );
            
            luptr += segsze;        /* Dense matrix-vector */
            // 更新 tempv1 指针，指向 tempv[segsze] 的位置
            tempv1 = &tempv[segsze];
            // 调用 cmatvec 函数进行矩阵-向量乘法
            cmatvec (nsupr, nrow, segsze, &lusup[luptr], tempv, tempv1);
#endif


注释：
#endif

/* 
 * Scatter tempv[*] into SPA dense[*] temporarily, such
 * that tempv[*] can be used for the triangular solve of
 * the next column of the panel. They will be copied into 
 * ucol[*] after the whole panel has been finished.
 */
isub = lptr + no_zeros;
for (i = 0; i < segsze; i++) {
    irow = lsub[isub];
    dense_col[irow] = tempv[i];
    tempv[i] = zero;
    isub++;
}

/* 
 * Scatter the update from tempv1[*] into SPA dense[*]
 * Start dense rectangular L
 */
for (i = 0; i < nrow; i++) {
    irow = lsub[isub];
    c_sub(&dense_col[irow], &dense_col[irow], &tempv1[i]);
    tempv1[i] = zero;
    ++isub;
}
```