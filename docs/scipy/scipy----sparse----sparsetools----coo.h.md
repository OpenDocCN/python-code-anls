# `D:\src\scipysrc\scipy\scipy\sparse\sparsetools\coo.h`

```
#ifndef __`
#ifndef __COO_H__
#define __COO_H__

#include <algorithm>

/*
 * Compute B = A for COO matrix A, CSR matrix B
 *
 *
 * Input Arguments:
 *   I  n_row      - number of rows in A
 *   I  n_col      - number of columns in A
 *   I  nnz        - number of nonzeros in A
 *   I  Ai[nnz(A)] - row indices
 *   I  Aj[nnz(A)] - column indices
 *   T  Ax[nnz(A)] - nonzeros
 * Output Arguments:
 *   I Bp  - row pointer
 *   I Bj  - column indices
 *   T Bx  - nonzeros
 *
 * Note:
 *   Output arrays Bp, Bj, and Bx must be preallocated
 *
 * Note: 
 *   Input:  row and column indices *are not* assumed to be ordered
 *           
 *   Note: duplicate entries are carried over to the CSR representation
 *
 *   Complexity: Linear.  Specifically O(nnz(A) + max(n_row,n_col))
 * 
 */
template <class I, class T>
void coo_tocsr(const I n_row,
               const I n_col,
               const I nnz,
               const I Ai[],
               const I Aj[],
               const T Ax[],
                     I Bp[],
                     I Bj[],
                     T Bx[])
{
    // 计算每行非零条目的数量，初始化 Bp 数组为 0
    std::fill(Bp, Bp + n_row, 0);

    // 遍历非零条目，累加每行的非零条目数到 Bp
    for (I n = 0; n < nnz; n++){            
        Bp[Ai[n]]++;
    }

    // 累加每行的非零条目数，得到 Bp 数组
    for(I i = 0, cumsum = 0; i < n_row; i++){     
        I temp = Bp[i];
        Bp[i] = cumsum;
        cumsum += temp;
    }
    Bp[n_row] = nnz; 

    // 将 Aj, Ax 写入到 Bj, Bx
    for(I n = 0; n < nnz; n++){
        I row  = Ai[n];
        I dest = Bp[row];

        Bj[dest] = Aj[n];
        Bx[dest] = Ax[n];

        Bp[row]++;
    }

    // 修正 Bp 数组，使其指向正确位置
    for(I i = 0, last = 0; i <= n_row; i++){
        I temp = Bp[i];
        Bp[i]  = last;
        last   = temp;
    }

    // 现在 Bp, Bj, Bx 构成 CSR 表示（可能包含重复条目）
}

/*
 * Compute B += A for COO matrix A, dense matrix B
 *
 * Input Arguments:
 *   I  n_row           - number of rows in A
 *   I  n_col           - number of columns in A
 *   npy_int64  nnz     - number of nonzeros in A
 *   I  Ai[nnz(A)]      - row indices
 *   I  Aj[nnz(A)]      - column indices
 *   T  Ax[nnz(A)]      - nonzeros 
 *   T  Bx[n_row*n_col] - dense matrix
 *
 */
template <class I, class T>
void coo_todense(const I n_row,
                 const I n_col,
                 const npy_int64 nnz,
                 const I Ai[],
                 const I Aj[],
                 const T Ax[],
                       T Bx[],
                 const int fortran)
{
    if (!fortran) {
        // 遍历 COO 矩阵 A 的非零条目，将其加入到稠密矩阵 Bx 中
        for(npy_int64 n = 0; n < nnz; n++){
            Bx[ (npy_intp)n_col * Ai[n] + Aj[n] ] += Ax[n];
        }
    }
    else {
        // 若以 Fortran 格式存储，按列主序遍历 COO 矩阵 A 的非零条目，加入到稠密矩阵 Bx 中
        for(npy_int64 n = 0; n < nnz; n++){
            Bx[ (npy_intp)n_row * Aj[n] + Ai[n] ] += Ax[n];
        }
    }
}

#endif // __COO_H__
/*
 * Compute Y += A*X for COO matrix A and dense vectors X,Y
 *
 * Input Arguments:
 *   npy_int64  nnz   - number of nonzeros in A
 *                      A 中非零元素的数量
 *   I  Ai[nnz]       - row indices
 *                      A 中每个非零元素对应的行索引数组
 *   I  Aj[nnz]       - column indices
 *                      A 中每个非零元素对应的列索引数组
 *   T  Ax[nnz]       - nonzero values
 *                      A 中每个非零元素的值数组
 *   T  Xx[n_col]     - input vector
 *                      输入向量 X
 *
 * Output Arguments:
 *   T  Yx[n_row]     - output vector
 *                      输出向量 Y
 *
 * Notes:
 *   Output array Yx must be preallocated
 *                      输出数组 Yx 必须预先分配空间
 *   Complexity: Linear.  Specifically O(nnz(A))
 *                      算法复杂度为线性，具体为 O(nnz(A))
 */
template <class I, class T>
void coo_matvec(const npy_int64 nnz,
                const I Ai[],
                const I Aj[],
                const T Ax[],
                const T Xx[],
                T Yx[])
{
    for(npy_int64 n = 0; n < nnz; n++){
        // 计算 Yx[Ai[n]] += Ax[n] * Xx[Aj[n]]
        Yx[Ai[n]] += Ax[n] * Xx[Aj[n]];
    }
}
```