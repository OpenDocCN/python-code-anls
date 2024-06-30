# `D:\src\scipysrc\scipy\scipy\sparse\sparsetools\csc.h`

```
#ifndef __CSC_H__
#define __CSC_H__

#include "csr.h"

/*
 * Compute Y += A*X for CSC matrix A and dense vectors X,Y
 *
 *
 * Input Arguments:
 *   I  n_row         - number of rows in A
 *   I  n_col         - number of columns in A
 *   I  Ap[n_row+1]   - column pointer
 *   I  Ai[nnz(A)]    - row indices
 *   T  Ax[n_col]     - nonzeros
 *   T  Xx[n_col]     - input vector
 *
 * Output Arguments:
 *   T  Yx[n_row]     - output vector
 *
 * Note:
 *   Output array Yx must be preallocated
 *
 *   Complexity: Linear.  Specifically O(nnz(A) + n_col)
 *
 */
template <class I, class T>
void csc_matvec(const I n_row,
                const I n_col,
                const I Ap[],
                const I Ai[],
                const T Ax[],
                const T Xx[],
                      T Yx[])
{
    // 遍历每列向量
    for(I j = 0; j < n_col; j++){
        // 获取当前列的起始和结束索引
        I col_start = Ap[j];
        I col_end   = Ap[j+1];

        // 遍历当前列中的每个非零元素
        for(I ii = col_start; ii < col_end; ii++){
            // 获取当前元素所在的行索引
            I i    = Ai[ii];
            // 计算 Yx[i] += Ax[ii] * Xx[j]
            Yx[i] += Ax[ii] * Xx[j];
        }
    }
}


/*
 * Compute Y += A*X for CSC matrix A and dense block vectors X,Y
 *
 *
 * Input Arguments:
 *   I  n_row            - number of rows in A
 *   I  n_col            - number of columns in A
 *   I  n_vecs           - number of column vectors in X and Y
 *   I  Ap[n_row+1]      - row pointer
 *   I  Ai[nnz(A)]       - column indices
 *   T  Ax[nnz(A)]       - nonzeros
 *   T  Xx[n_col,n_vecs] - input vector
 *
 * Output Arguments:
 *   T  Yx[n_row,n_vecs] - output vector
 *
 * Note:
 *   Output array Yx must be preallocated
 *
 */
template <class I, class T>
void csc_matvecs(const I n_row,
                 const I n_col,
                 const I n_vecs,
                 const I Ap[],
                 const I Ai[],
                 const T Ax[],
                 const T Xx[],
                       T Yx[])
{
    // 遍历每列向量
    for(I j = 0; j < n_col; j++){
        // 遍历当前列中的每个非零元素
        for(I ii = Ap[j]; ii < Ap[j+1]; ii++){
            // 获取当前元素所在的行索引
            const I i = Ai[ii];
            // 调用axpy函数，实现 Yx[i] += Ax[ii] * Xx[:,j]
            axpy(n_vecs, Ax[ii], Xx + (npy_intp)n_vecs * j, Yx + (npy_intp)n_vecs * i);
        }
    }
}




/*
 * Derived methods
 */
template <class I, class T>
void csc_diagonal(const I k,
                  const I n_row,
                  const I n_col,
                  const I Ap[],
                  const I Aj[],
                  const T Ax[],
                        T Yx[])
{ 
    // 调用csr_diagonal函数，对角线提取转换
    csr_diagonal(-k, n_col, n_row, Ap, Aj, Ax, Yx); 
}


template <class I, class T>
void csc_tocsr(const I n_row,
               const I n_col,
               const I Ap[],
               const I Ai[],
               const T Ax[],
                     I Bp[],
                     I Bj[],
                     T Bx[])
{ 
    // 调用csr_tocsc函数，CSC转CSR格式
    csr_tocsc<I,T>(n_col, n_row, Ap, Ai, Ax, Bp, Bj, Bx); 
}

template <class I>
npy_intp csc_matmat_maxnnz(const I n_row,
                           const I n_col,
                           const I Ap[],
                           const I Ai[],
                           const I Bp[],
                           const I Bi[])
# 返回一个函数调用的结果，调用 csr_matmat_maxnnz 函数，传入参数 n_col, n_row, Bp, Bi, Ap, Ai
{ return csr_matmat_maxnnz(n_col, n_row, Bp, Bi, Ap, Ai); }

# 定义一个模板函数 csc_matmat，用于执行稀疏矩阵乘法操作
template <class I, class T>
void csc_matmat(const I n_row,
                const I n_col,
                const I Ap[],
                const I Ai[],
                const T Ax[],
                const I Bp[],
                const I Bi[],
                const T Bx[],
                      I Cp[],
                      I Ci[],
                      T Cx[])
{
    # 调用 csr_matmat 函数，传入参数 n_col, n_row, Bp, Bi, Bx, Ap, Ai, Ax, Cp, Ci, Cx
    csr_matmat(n_col, n_row, Bp, Bi, Bx, Ap, Ai, Ax, Cp, Ci, Cx);
}

# 定义一个模板函数 csc_ne_csc，用于执行稀疏矩阵不等于比较操作
template <class I, class T, class T2>
void csc_ne_csc(const I n_row, const I n_col,
                const I Ap[], const I Ai[], const T Ax[],
                const I Bp[], const I Bi[], const T Bx[],
                      I Cp[],       I Ci[],      T2 Cx[])
{
    # 调用 csr_ne_csr 函数，传入参数 n_col, n_row, Ap, Ai, Ax, Bp, Bi, Bx, Cp, Ci, Cx
    csr_ne_csr(n_col, n_row, Ap, Ai, Ax, Bp, Bi, Bx, Cp, Ci, Cx);
}

# 定义一个模板函数 csc_lt_csc，用于执行稀疏矩阵小于比较操作
template <class I, class T, class T2>
void csc_lt_csc(const I n_row, const I n_col,
                const I Ap[], const I Ai[], const T Ax[],
                const I Bp[], const I Bi[], const T Bx[],
                      I Cp[],       I Ci[],      T2 Cx[])
{
    # 调用 csr_lt_csr 函数，传入参数 n_col, n_row, Ap, Ai, Ax, Bp, Bi, Bx, Cp, Ci, Cx
    csr_lt_csr(n_col, n_row, Ap, Ai, Ax, Bp, Bi, Bx, Cp, Ci, Cx);
}

# 定义一个模板函数 csc_gt_csc，用于执行稀疏矩阵大于比较操作
template <class I, class T, class T2>
void csc_gt_csc(const I n_row, const I n_col,
                const I Ap[], const I Ai[], const T Ax[],
                const I Bp[], const I Bi[], const T Bx[],
                      I Cp[],       I Ci[],      T2 Cx[])
{
    # 调用 csr_gt_csr 函数，传入参数 n_col, n_row, Ap, Ai, Ax, Bp, Bi, Bx, Cp, Ci, Cx
    csr_gt_csr(n_col, n_row, Ap, Ai, Ax, Bp, Bi, Bx, Cp, Ci, Cx);
}

# 定义一个模板函数 csc_le_csc，用于执行稀疏矩阵小于等于比较操作
template <class I, class T, class T2>
void csc_le_csc(const I n_row, const I n_col,
                const I Ap[], const I Ai[], const T Ax[],
                const I Bp[], const I Bi[], const T Bx[],
                      I Cp[],       I Ci[],      T2 Cx[])
{
    # 调用 csr_le_csr 函数，传入参数 n_col, n_row, Ap, Ai, Ax, Bp, Bi, Bx, Cp, Ci, Cx
    csr_le_csr(n_col, n_row, Ap, Ai, Ax, Bp, Bi, Bx, Cp, Ci, Cx);
}

# 定义一个模板函数 csc_ge_csc，用于执行稀疏矩阵大于等于比较操作
template <class I, class T, class T2>
void csc_ge_csc(const I n_row, const I n_col,
                const I Ap[], const I Ai[], const T Ax[],
                const I Bp[], const I Bi[], const T Bx[],
                      I Cp[],       I Ci[],      T2 Cx[])
{
    # 调用 csr_ge_csr 函数，传入参数 n_col, n_row, Ap, Ai, Ax, Bp, Bi, Bx, Cp, Ci, Cx
    csr_ge_csr(n_col, n_row, Ap, Ai, Ax, Bp, Bi, Bx, Cp, Ci, Cx);
}

# 定义一个模板函数 csc_elmul_csc，用于执行稀疏矩阵逐元素乘法操作
template <class I, class T>
void csc_elmul_csc(const I n_row, const I n_col,
                   const I Ap[], const I Ai[], const T Ax[],
                   const I Bp[], const I Bi[], const T Bx[],
                         I Cp[],       I Ci[],       T Cx[])
{
    # 调用 csr_elmul_csr 函数，传入参数 n_col, n_row, Ap, Ai, Ax, Bp, Bi, Bx, Cp, Ci, Cx
    csr_elmul_csr(n_col, n_row, Ap, Ai, Ax, Bp, Bi, Bx, Cp, Ci, Cx);
}

# 定义一个模板函数 csc_eldiv_csc，用于执行稀疏矩阵逐元素除法操作
template <class I, class T>
void csc_eldiv_csc(const I n_row, const I n_col,
                   const I Ap[], const I Ai[], const T Ax[],
                   const I Bp[], const I Bi[], const T Bx[],
                         I Cp[],       I Ci[],       T Cx[])
{
    # 调用 csr_eldiv_csr 函数，传入参数 n_col, n_row, Ap, Ai, Ax, Bp, Bi, Bx, Cp, Ci, Cx
    csr_eldiv_csr(n_col, n_row, Ap, Ai, Ax, Bp, Bi, Bx, Cp, Ci, Cx);
}
// 实现将两个压缩稀疏列矩阵（CSC 格式）相加的函数
void csc_plus_csc(const I n_row, const I n_col,
                  const I Ap[], const I Ai[], const T Ax[],
                  const I Bp[], const I Bi[], const T Bx[],
                        I Cp[],       I Ci[],       T Cx[])
{
    // 调用 csr_plus_csr 函数实现稀疏行压缩矩阵（CSR 格式）相加，参数顺序颠倒以适应 CSC 格式
    csr_plus_csr(n_col, n_row, Ap, Ai, Ax, Bp, Bi, Bx, Cp, Ci, Cx);
}

// 实现将两个压缩稀疏列矩阵（CSC 格式）相减的函数
template <class I, class T>
void csc_minus_csc(const I n_row, const I n_col,
                   const I Ap[], const I Ai[], const T Ax[],
                   const I Bp[], const I Bi[], const T Bx[],
                         I Cp[],       I Ci[],       T Cx[])
{
    // 调用 csr_minus_csr 函数实现稀疏行压缩矩阵（CSR 格式）相减，参数顺序颠倒以适应 CSC 格式
    csr_minus_csr(n_col, n_row, Ap, Ai, Ax, Bp, Bi, Bx, Cp, Ci, Cx);
}

// 实现将两个压缩稀疏列矩阵（CSC 格式）逐元素取最大值的函数
template <class I, class T>
void csc_maximum_csc(const I n_row, const I n_col,
                     const I Ap[], const I Ai[], const T Ax[],
                     const I Bp[], const I Bi[], const T Bx[],
                           I Cp[],       I Ci[],       T Cx[])
{
    // 调用 csr_maximum_csr 函数实现稀疏行压缩矩阵（CSR 格式）逐元素取最大值，参数顺序颠倒以适应 CSC 格式
    csr_maximum_csr(n_col, n_row, Ap, Ai, Ax, Bp, Bi, Bx, Cp, Ci, Cx);
}

// 实现将两个压缩稀疏列矩阵（CSC 格式）逐元素取最小值的函数
template <class I, class T>
void csc_minimum_csc(const I n_row, const I n_col,
                     const I Ap[], const I Ai[], const T Ax[],
                     const I Bp[], const I Bi[], const T Bx[],
                           I Cp[],       I Ci[],       T Cx[])
{
    // 调用 csr_minimum_csr 函数实现稀疏行压缩矩阵（CSR 格式）逐元素取最小值，参数顺序颠倒以适应 CSC 格式
    csr_minimum_csr(n_col, n_row, Ap, Ai, Ax, Bp, Bi, Bx, Cp, Ci, Cx);
}
```