# `D:\src\scipysrc\scipy\scipy\sparse\sparsetools\csr.h`

```
#ifndef __CSR_H__
#define __CSR_H__

#include <set>  // 包含集合的头文件
#include <vector>  // 包含向量的头文件
#include <algorithm>  // 包含算法的头文件
#include <functional>  // 包含函数对象的头文件
#include <numeric>  // 包含数值操作的头文件

#include "util.h"  // 包含自定义实用程序的头文件
#include "dense.h"  // 包含密集矩阵的头文件

/*
 * Extract k-th diagonal of CSR matrix A
 *
 * Input Arguments:
 *   I  k             - diagonal to extract  // 要提取的对角线
 *   I  n_row         - number of rows in A  // A 矩阵的行数
 *   I  n_col         - number of columns in A  // A 矩阵的列数
 *   I  Ap[n_row+1]   - row pointer  // 行指针数组
 *   I  Aj[nnz(A)]    - column indices  // 列索引数组
 *   T  Ax[n_col]     - nonzeros  // 非零元素数组
 *
 * Output Arguments:
 *   T  Yx[min(n_row,n_col)] - diagonal entries  // 对角线元素的输出数组
 *
 * Note:
 *   Output array Yx must be preallocated  // 输出数组 Yx 必须预先分配内存空间
 *
 *   Duplicate entries will be summed.  // 重复的元素将会求和
 *
 *   Complexity: Linear.  Specifically O(nnz(A) + min(n_row,n_col))  // 算法复杂度为线性时间，具体为 O(nnz(A) + min(n_row,n_col))
 *
 */
template <class I, class T>
void csr_diagonal(const I k,
                  const I n_row,
                  const I n_col,
                  const I Ap[],
                  const I Aj[],
                  const T Ax[],
                        T Yx[])
{
    const I first_row = (k >= 0) ? 0 : -k;  // 计算第一个要处理的行索引
    const I first_col = (k >= 0) ? k : 0;  // 计算第一个要处理的列索引
    const I N = std::min(n_row - first_row, n_col - first_col);  // 计算要处理的元素个数

    for (I i = 0; i < N; ++i) {  // 循环处理每个对角线元素
        const I row = first_row + i;  // 当前行索引
        const I col = first_col + i;  // 当前列索引
        const I row_begin = Ap[row];  // 当前行的起始索引
        const I row_end = Ap[row + 1];  // 当前行的结束索引

        T diag = 0;  // 对角线元素的累加器
        for (I j = row_begin; j < row_end; ++j) {  // 遍历当前行的非零元素
            if (Aj[j] == col) {  // 如果找到对应的列索引
                diag += Ax[j];  // 将非零元素累加到对角线元素上
            }
        }

        Yx[i] = diag;  // 将计算得到的对角线元素存入输出数组
    }

}


/*
 * Expand a compressed row pointer into a row array
 *
 * Input Arguments:
 *   I  n_row         - number of rows in A  // A 矩阵的行数
 *   I  Ap[n_row+1]   - row pointer  // 行指针数组
 *
 * Output Arguments:
 *   Bi  - row indices  // 扩展后的行索引数组
 *
 * Note:
 *   Output array Bi must be preallocated  // 输出数组 Bi 必须预先分配内存空间
 *
 * Note:
 *   Complexity: Linear  // 算法复杂度为线性时间
 *
 */
template <class I>
void expandptr(const I n_row,
               const I Ap[],
                     I Bi[])
{
    for(I i = 0; i < n_row; i++){  // 遍历每一行
        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){  // 根据行指针扩展行索引数组
            Bi[jj] = i;  // 将当前行的索引填充到对应的位置
        }
    }
}


/*
 * Scale the rows of a CSR matrix *in place*
 *
 *   A[i,:] *= X[i]
 *
 */
template <class I, class T>
void csr_scale_rows(const I n_row,
                    const I n_col,
                    const I Ap[],
                    const I Aj[],
                          T Ax[],
                    const T Xx[])
{
    for(I i = 0; i < n_row; i++){  // 遍历每一行
        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){  // 遍历当前行的非零元素
            Ax[jj] *= Xx[i];  // 对当前行的每个非零元素进行缩放
        }
    }
}


/*
 * Scale the columns of a CSR matrix *in place*
 *
 *   A[:,i] *= X[i]
 *
 */
template <class I, class T>
void csr_scale_columns(const I n_row,
                       const I n_col,
                       const I Ap[],
                       const I Aj[],
                             T Ax[],
                       const T Xx[])
{
    const I nnz = Ap[n_row];  // 获取非零元素的总数
    for(I i = 0; i < nnz; i++){  // 遍历所有非零元素
        Ax[i] *= Xx[Aj[i]];  // 对每列进行缩放操作
    }
}

#endif // __CSR_H__
/*
 * Compute the number of occupied RxC blocks in a matrix
 *
 * Input Arguments:
 *   I  n_row         - matrix的行数
 *   I  n_col         - matrix的列数
 *   I  R             - 行块大小
 *   I  C             - 列块大小
 *   I  Ap[n_row+1]   - 行指针数组
 *   I  Aj[nnz(A)]    - 列索引数组
 *
 * Output Arguments:
 *   I  num_blocks    - 块的数量
 *
 * Note:
 *   Complexity: Linear
 *
 */
template <class I>
I csr_count_blocks(const I n_row,
                   const I n_col,
                   const I R,
                   const I C,
                   const I Ap[],
                   const I Aj[])
{
    // 初始化掩码数组，表示每个块的使用情况，初始值为-1
    std::vector<I> mask(n_col/C + 1, -1);
    I n_blks = 0; // 记录块的数量

    // 遍历每一行
    for(I i = 0; i < n_row; i++){
        I bi = i / R; // 计算当前行属于哪个行块
        // 遍历当前行中的每个非零元素
        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            I bj = Aj[jj] / C; // 计算当前列属于哪个列块
            // 如果当前块未被使用过，则标记为已使用，并增加块的数量
            if(mask[bj] != bi){
                mask[bj] = bi;
                n_blks++;
            }
        }
    }
    return n_blks; // 返回块的数量
}


/*
 * Convert a CSR matrix to BSR format
 *
 * Input Arguments:
 *   I  n_row           - matrix的行数
 *   I  n_col           - matrix的列数
 *   I  R               - 行块大小
 *   I  C               - 列块大小
 *   I  Ap[n_row+1]     - 行指针数组
 *   I  Aj[nnz(A)]      - 列索引数组
 *   T  Ax[nnz(A)]      - 非零值数组
 *
 * Output Arguments:
 *   I  Bp[n_row/R + 1] - 块行指针数组
 *   I  Bj[nnz(B)]      - 块列索引数组
 *   T  Bx[nnz(B)]      - 非零块值数组
 *
 * Note:
 *   Complexity: Linear
 *   输出数组必须预先分配空间（Bx初始化为零）
 *
 */
template <class I, class T>
void csr_tobsr(const I n_row,
               const I n_col,
               const I R,
               const I C,
               const I Ap[],
               const I Aj[],
               const T Ax[],
                     I Bp[],
                     I Bj[],
                     T Bx[])
{
    // 初始化块数组，初始值为0，用于存储每个列块的非零块
    std::vector<T*> blocks(n_col/C + 1, (T*)0 );

    // 确保矩阵的行数和列数能够整除块大小
    assert(n_row % R == 0);
    assert(n_col % C == 0);

    I n_brow = n_row / R; // 计算行块的数量
    I RC = R * C; // 计算块大小
    I n_blks = 0; // 记录块的数量

    Bp[0] = 0; // 第一个块行指针为0

    // 遍历每个行块
    for(I bi = 0; bi < n_brow; bi++){
        // 遍历当前行块中的每一行
        for(I r = 0; r < R; r++){
            I i = R * bi + r; // 计算行索引
            // 遍历当前行中的每个非零元素
            for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
                I j = Aj[jj]; // 获取列索引

                I bj = j / C; // 计算列块索引
                I c  = j % C; // 计算列块内列索引

                // 如果当前列块尚未分配空间，则进行初始化，并记录块索引
                if(blocks[bj] == 0){
                    blocks[bj] = Bx + RC * n_blks;
                    Bj[n_blks] = bj;
                    n_blks++;
                }

                // 将对应位置的块值加上原始矩阵中的非零值
                *(blocks[bj] + C * r + c) += Ax[jj];
            }
        }

        // 清空当前行块处理过的列块标记
        for(I jj = Ap[R * bi]; jj < Ap[R * (bi + 1)]; jj++){
            blocks[Aj[jj] / C] = 0;
        }

        // 更新块行指针数组
        Bp[bi + 1] = n_blks;
    }
}
/*
 * 将 CSR 矩阵 A 的非零值加到 C 连续存储的密集矩阵 B 中
 *
 * 输入参数：
 *   I  n_row           - A 矩阵的行数
 *   I  n_col           - A 矩阵的列数
 *   I  Ap[n_row+1]     - 行指针数组
 *   I  Aj[nnz(A)]      - 列索引数组
 *   T  Ax[nnz(A)]      - 非零值数组
 *   T  Bx[n_row*n_col] - 行优先存储的密集矩阵 B
 *
 */
template <class I, class T>
void csr_todense(const I n_row,
                 const I n_col,
                 const I Ap[],
                 const I Aj[],
                 const T Ax[],
                       T Bx[])
{
    // 初始化指向 B 的当前行
    T * Bx_row = Bx;
    // 遍历每一行 A 的非零值
    for(I i = 0; i < n_row; i++){
        // 遍历第 i 行中的每个非零元素
        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            // 将 A 中的非零值加到 B 的相应位置
            Bx_row[Aj[jj]] += Ax[jj];
        }
        // 移动到 B 中的下一行
        Bx_row += (npy_intp)n_col;
    }
}


/*
 * 检查 CSR 格式的列索引是否按排序顺序排列
 *
 * 输入参数：
 *   I  n_row           - A 矩阵的行数
 *   I  Ap[n_row+1]     - 行指针数组
 *   I  Aj[nnz(A)]      - 列索引数组
 *
 * 返回值：
 *   如果列索引有序则返回 true，否则返回 false
 */
template <class I>
bool csr_has_sorted_indices(const I n_row,
                            const I Ap[],
                            const I Aj[])
{
    // 遍历每一行的列索引
    for(I i = 0; i < n_row; i++){
        // 检查第 i 行的列索引是否有序
        for(I jj = Ap[i]; jj < Ap[i+1] - 1; jj++){
            if(Aj[jj] > Aj[jj+1]){
                // 如果找到不按顺序的列索引则返回 false
                return false;
            }
        }
    }
    // 如果所有行的列索引都有序则返回 true
    return true;
}



/*
 * 检查 CSR 格式的矩阵结构是否为规范格式
 * 规范 CSR 要求每行的列索引 (1) 按排序顺序排列并且 (2) 唯一。
 * 满足这些条件的矩阵可以提高矩阵运算的速度。
 *
 * 输入参数：
 *   I  n_row           - A 矩阵的行数
 *   I  Ap[n_row+1]     - 行指针数组
 *   I  Aj[nnz(A)]      - 列索引数组
 *
 * 返回值：
 *   如果矩阵结构为规范 CSR 格式则返回 true，否则返回 false
 */
template <class I>
bool csr_has_canonical_format(const I n_row,
                              const I Ap[],
                              const I Aj[])
{
    // 遍历每一行的列索引
    for(I i = 0; i < n_row; i++){
        // 检查每行的起始指针是否小于等于结束指针
        if (Ap[i] > Ap[i+1])
            return false;
        // 检查每行的列索引是否按顺序排列且唯一
        for(I jj = Ap[i] + 1; jj < Ap[i+1]; jj++){
            if( !(Aj[jj-1] < Aj[jj]) ){
                return false;
            }
        }
    }
    // 如果所有行都符合规范格式则返回 true
    return true;
}


/*
 * 对 CSR 格式的列索引进行原地排序
 *
 * 输入参数：
 *   I  n_row           - A 矩阵的行数
 *   I  Ap[n_row+1]     - 行指针数组
 *   I  Aj[nnz(A)]      - 列索引数组
 *   T  Ax[nnz(A)]      - 非零值数组
 *
 */
template<class I, class T>
void csr_sort_indices(const I n_row,
                      const I Ap[],
                            I Aj[],
                            T Ax[])
{
    // 创建临时向量存储列索引及其对应的非零值
    std::vector< std::pair<I,T> > temp;

    // 遍历每一行的非零元素
    for(I i = 0; i < n_row; i++){
        // 将第 i 行的非零元素插入临时向量中
        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            temp.push_back(std::make_pair(Aj[jj], Ax[jj]));
        }
        // 对临时向量中的元素按列索引排序
        std::sort(temp.begin(), temp.end(), kv_pair_less<I,T>);
        // 将排序后的列索引及其对应的非零值放回原来的数组中
        typename std::vector< std::pair<I,T> >::iterator it = temp.begin();
        for(I jj = Ap[i]; jj < Ap[i+1]; jj++, ++it){
            Aj[jj] = it->first;
            Ax[jj] = it->second;
        }
        // 清空临时向量，为下一行准备
        temp.clear();
    }
}
    // 遍历每一行的非零元素
    for(I i = 0; i < n_row; i++){
        // 获取当前行的起始和结束索引
        I row_start = Ap[i];
        I row_end   = Ap[i+1];
    
        // 调整临时数组的大小以容纳当前行的元素
        temp.resize(row_end - row_start);
    
        // 将当前行的每个非零元素复制到临时数组中的键值对
        for (I jj = row_start, n = 0; jj < row_end; jj++, n++){
            temp[n].first  = Aj[jj];  // 存储列索引
            temp[n].second = Ax[jj];  // 存储对应的值
        }
    
        // 对临时数组中的键值对按键进行排序
        std::sort(temp.begin(), temp.end(), kv_pair_less<I,T>);
    
        // 将排序后的键值对重新写回原始数组
        for(I jj = row_start, n = 0; jj < row_end; jj++, n++){
            Aj[jj] = temp[n].first;   // 恢复列索引
            Ax[jj] = temp[n].second;  // 恢复对应的值
        }
    }
/*
 * Compute B = A for CSR matrix A, CSC matrix B
 *
 * Also, with the appropriate arguments can also be used to:
 *   - compute B = A^t for CSR matrix A, CSR matrix B
 *   - compute B = A^t for CSC matrix A, CSC matrix B
 *   - convert CSC->CSR
 *
 * Input Arguments:
 *   I  n_row         - number of rows in A
 *   I  n_col         - number of columns in A
 *   I  Ap[n_row+1]   - row pointer
 *   I  Aj[nnz(A)]    - column indices
 *   T  Ax[nnz(A)]    - nonzeros
 *
 * Output Arguments:
 *   I  Bp[n_col+1] - column pointer
 *   I  Bi[nnz(A)]  - row indices
 *   T  Bx[nnz(A)]  - nonzeros
 *
 * Note:
 *   Output arrays Bp, Bi, Bx must be preallocated
 *
 * Note:
 *   Input:  column indices *are not* assumed to be in sorted order
 *   Output: row indices *will be* in sorted order
 *
 *   Complexity: Linear.  Specifically O(nnz(A) + max(n_row,n_col))
 *
 */
template <class I, class T>
void csr_tocsc(const I n_row,
               const I n_col,
               const I Ap[],
               const I Aj[],
               const T Ax[],
                     I Bp[],
                     I Bi[],
                     T Bx[])
{
    const I nnz = Ap[n_row];

    // 计算矩阵A每列的非零元素个数
    std::fill(Bp, Bp + n_col, 0);

    for (I n = 0; n < nnz; n++){
        // 统计每列非零元素个数
        Bp[Aj[n]]++;
    }

    // 计算每列非零元素的累积和，得到列指针数组Bp
    for(I col = 0, cumsum = 0; col < n_col; col++){
        I temp  = Bp[col];
        Bp[col] = cumsum;
        cumsum += temp;
    }
    Bp[n_col] = nnz;

    // 填充输出的行索引数组Bi和非零元素数组Bx
    for(I row = 0; row < n_row; row++){
        for(I jj = Ap[row]; jj < Ap[row+1]; jj++){
            I col  = Aj[jj];
            I dest = Bp[col];

            Bi[dest] = row;
            Bx[dest] = Ax[jj];

            Bp[col]++;
        }
    }

    // 重置列指针数组Bp，使其存储起始位置而非累积和
    for(I col = 0, last = 0; col <= n_col; col++){
        I temp  = Bp[col];
        Bp[col] = last;
        last    = temp;
    }
}



/*
 * Compute B = A for CSR matrix A, ELL matrix B
 *
 * Input Arguments:
 *   I  n_row         - number of rows in A
 *   I  n_col         - number of columns in A
 *   I  Ap[n_row+1]   - row pointer
 *   I  Aj[nnz(A)]    - column indices
 *   T  Ax[nnz(A)]    - nonzeros
 *   I  row_length    - maximum nnz in a row of A
 *
 * Output Arguments:
 *   I  Bj[n_row * row_length]  - column indices
 *   T  Bx[n_row * row_length]  - nonzeros
 *
 * Note:
 *   Output arrays Bj, Bx must be preallocated
 *   Duplicate entries in A are not merged.
 *   Explicit zeros in A are carried over to B.
 *   Rows with fewer than row_length columns are padded with zeros.
 *
 */
template <class I, class T>
void csr_toell(const I n_row,
               const I n_col,
               const I Ap[],
               const I Aj[],
               const T Ax[],
               const I row_length,
                     I Bj[],
                     T Bx[])
{
    const npy_intp ell_nnz = (npy_intp)row_length * n_row;

    // 初始化输出的列索引数组Bj和非零元素数组Bx为0
    std::fill(Bj, Bj + ell_nnz, 0);
    std::fill(Bx, Bx + ell_nnz, 0);
    # 遍历每一行的稀疏矩阵数据
    for(I i = 0; i < n_row; i++){
        # 计算当前行在Bj数组中的起始位置
        I * Bj_row = Bj + (npy_intp)row_length * i;
        # 计算当前行在Bx数组中的起始位置
        T * Bx_row = Bx + (npy_intp)row_length * i;
        # 遍历当前行的非零元素
        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            # 将Aj中的非零元素赋值给Bj_row指向的位置
            *Bj_row = Aj[jj];
            # 将Ax中的非零元素赋值给Bx_row指向的位置
            *Bx_row = Ax[jj];
            # 移动到下一个Bj数组的位置
            Bj_row++;
            # 移动到下一个Bx数组的位置
            Bx_row++;
        }
    }
/*
 * Compute C = A*B for CSR matrices A,B
 *
 *
 * Input Arguments:
 *   I  n_row       - A的行数
 *   I  n_col       - B的列数（因此C的大小为n_row乘以n_col）
 *   I  Ap[n_row+1] - A的行指针
 *   I  Aj[nnz(A)]  - A的列索引
 *   T  Ax[nnz(A)]  - A的非零元素
 *   I  Bp[?]       - B的行指针
 *   I  Bj[nnz(B)]  - B的列索引
 *   T  Bx[nnz(B)]  - B的非零元素
 * Output Arguments:
 *   I  Cp[n_row+1] - C的行指针
 *   I  Cj[nnz(C)]  - C的列索引
 *   T  Cx[nnz(C)]  - C的非零元素
 *
 * Note:
 *   输出数组Cp, Cj和Cx必须预先分配空间。
 *   可以使用csr_matmat_maxnnz函数找到nnz(C)的适当类型。
 *
 * Note:
 *   输入：A和B的列索引不假定为排序。
 *   输出：C的列索引不假定为排序；Cx不包含任何零条目。
 *
 *   复杂度：O(n_row*K^2 + max(n_row,n_col))
 *            其中K是A的行和B的列中最大的非零元素数目。
 *
 *
 *  这是SMMP算法的实现：
 *
 *    "Sparse Matrix Multiplication Package (SMMP)"
 *      Randolph E. Bank 和 Craig C. Douglas
 *
 *    http://citeseer.ist.psu.edu/445062.html
 *    http://www.mgnet.org/~douglas/ccd-codes.html
 *
 */

/*
 * 计算C = A * B的结果中的非零元素数目（nnz）。
 *
 */
template <class I>
npy_intp csr_matmat_maxnnz(const I n_row,
                           const I n_col,
                           const I Ap[],
                           const I Aj[],
                           const I Bp[],
                           const I Bj[])
{
    // 使用O(n_col)的临时存储方法
    std::vector<I> mask(n_col, -1);

    npy_intp nnz = 0;
    for(I i = 0; i < n_row; i++){
        npy_intp row_nnz = 0;

        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            I j = Aj[jj];
            for(I kk = Bp[j]; kk < Bp[j+1]; kk++){
                I k = Bj[kk];
                if(mask[k] != i){
                    mask[k] = i;
                    row_nnz++;
                }
            }
        }

        npy_intp next_nnz = nnz + row_nnz;

        if (row_nnz > NPY_MAX_INTP - nnz) {
            /*
             * 索引溢出。注意row_nnz <= n_col，不会溢出。
             */
            throw std::overflow_error("结果的nnz太大");
        }

        nnz = next_nnz;
    }

    return nnz;
}

/*
 * 计算CSR格式的矩阵C = A * B的条目。
 *
 */
template <class I, class T>
void csr_matmat(const I n_row,
                const I n_col,
                const I Ap[],
                const I Aj[],
                const T Ax[],
                const I Bp[],
                const I Bj[],
                const T Bx[],
                      I Cp[],
                      I Cj[],
                      T Cx[])
{
    std::vector<I> next(n_col,-1);
    std::vector<T> sums(n_col, 0);

    I nnz = 0;

    Cp[0] = 0;
    // 外层循环，遍历稀疏矩阵的每一行
    for(I i = 0; i < n_row; i++){
        // 初始化头指针为-2，表示链表尾部；长度初始化为0
        I head   = -2;
        I length =  0;

        // 获取第i行在压缩列索引Ap中的起始和结束位置
        I jj_start = Ap[i];
        I jj_end   = Ap[i+1];
        // 内层循环，遍历第i行的每个非零元素
        for(I jj = jj_start; jj < jj_end; jj++){
            // 获取列索引Aj[jj]和对应的值Ax[jj]
            I j = Aj[jj];
            T v = Ax[jj];

            // 获取矩阵B中第j列在压缩列索引Bp中的起始和结束位置
            I kk_start = Bp[j];
            I kk_end   = Bp[j+1];
            // 遍历矩阵B中第j列的每个非零元素
            for(I kk = kk_start; kk < kk_end; kk++){
                // 获取列索引Bj[kk]和对应的值Bx[kk]
                I k = Bj[kk];

                // 更新sums[k]，累加乘积v * Bx[kk]
                sums[k] += v * Bx[kk];

                // 如果next[k]为-1，表示链表中不存在k，将k插入链表头部
                if(next[k] == -1){
                    next[k] = head;
                    head  = k;
                    length++;
                }
            }
        }

        // 根据链表长度遍历链表头部，处理非零元素
        for(I jj = 0; jj < length; jj++){

            // 如果sums[head]不为0，将head列索引和对应值添加到输出矩阵的列索引和值数组中
            if(sums[head] != 0){
                Cj[nnz] = head;
                Cx[nnz] = sums[head];
                nnz++;
            }

            // 将head保存到temp中，并更新head为链表中的下一个元素
            I temp = head;
            head = next[head];

            // 清空temp对应的数组元素，next数组中temp位置置为-1，sums数组中temp位置置为0
            next[temp] = -1; //clear arrays
            sums[temp] =  0;
        }

        // 将当前行的非零元素个数nnz保存到输出矩阵行指针数组Cp中的相应位置
        Cp[i+1] = nnz;
    }
/*
 * 计算两个 CSR 格式的矩阵 A 和 B 的二元操作 C = A (binary_op) B，
 * 这些矩阵不一定是规范的 CSR 格式。具体来说，此方法即使在输入矩阵在给定行内具有重复和/或未排序的列索引时也能正常工作。
 *
 * 有关更多信息，请参考 csr_binop_csr()。
 *
 * 注意：
 *   输出数组 Cp、Cj 和 Cx 必须预先分配空间。
 *   如果 nnz(C) 不事先已知，一个保守的上界是：
 *          nnz(C) <= nnz(A) + nnz(B)
 *
 * 注意：
 *   输入：A 和 B 的列索引不假定为排序状态。
 *   输出：C 的列索引通常不是排序的。
 *        C 不包含任何重复条目或显式的零值。
 */
template <class I, class T, class T2, class binary_op>
void csr_binop_csr_general(const I n_row, const I n_col,
                           const I Ap[], const I Aj[], const T Ax[],
                           const I Bp[], const I Bj[], const T Bx[],
                                 I Cp[],       I Cj[],       T2 Cx[],
                           const binary_op& op)
{
    // 适用于具有重复和/或未排序索引的方法

    std::vector<I>  next(n_col,-1); // 下一个非零元素的索引数组
    std::vector<T> A_row(n_col, 0); // 存储当前处理行在 A 矩阵中的累计值
    std::vector<T> B_row(n_col, 0); // 存储当前处理行在 B 矩阵中的累计值

    I nnz = 0; // C 矩阵中的非零元素计数
    Cp[0] = 0; // C 矩阵行指针数组的起始索引为 0

    for(I i = 0; i < n_row; i++){
        I head   = -2; // 当前行处理的列索引的头指针
        I length =  0; // 当前行中非零元素的数量

        // 将 A 矩阵的一行添加到 A_row 中
        I i_start = Ap[i];
        I i_end   = Ap[i+1];
        for(I jj = i_start; jj < i_end; jj++){
            I j = Aj[jj];

            A_row[j] += Ax[jj]; // 累加 A 矩阵中当前元素的值到 A_row[j]

            if(next[j] == -1){
                next[j] = head; // 更新下一个非零元素的索引
                head = j;       // 更新头指针
                length++;       // 增加当前行中非零元素的数量
            }
        }

        // 将 B 矩阵的一行添加到 B_row 中
        i_start = Bp[i];
        i_end   = Bp[i+1];
        for(I jj = i_start; jj < i_end; jj++){
            I j = Bj[jj];

            B_row[j] += Bx[jj]; // 累加 B 矩阵中当前元素的值到 B_row[j]

            if(next[j] == -1){
                next[j] = head; // 更新下一个非零元素的索引
                head = j;       // 更新头指针
                length++;       // 增加当前行中非零元素的数量
            }
        }

        // 扫描具有非零输入的列索引（来自 A 或 B 矩阵）
        for(I jj = 0; jj < length; jj++){
            T result = op(A_row[head], B_row[head]); // 应用二元操作符到 A_row[head] 和 B_row[head]

            if(result != 0){
                Cj[nnz] = head; // 存储 C 矩阵中非零元素的列索引
                Cx[nnz] = result; // 存储 C 矩阵中非零元素的值
                nnz++; // 增加非零元素计数
            }

            I temp = head;
            head = next[head]; // 移动到下一个非零元素的索引

            next[temp]  = -1; // 清除当前元素的下一个索引
            A_row[temp] =  0; // 清除 A_row 中当前元素的值
            B_row[temp] =  0; // 清除 B_row 中当前元素的值
        }

        Cp[i + 1] = nnz; // 更新 C 矩阵行指针数组
    }
}
/*
 * 计算两个稀疏矩阵 A 和 B 的二元操作结果 C = A (binary_op) B，要求这些矩阵采用标准的CSR格式。
 * 具体来说，该方法要求输入矩阵的行不含重复的列索引，并且列索引必须按顺序排序。
 *
 * 参考 csr_binop_csr() 获取更多信息
 *
 * 注意：
 *   输入：假定 A 和 B 的列索引已按排序顺序排列
 *   输出：C 的列索引将按排序顺序排列
 *         Cx 不包含任何零条目
 */
template <class I, class T, class T2, class binary_op>
void csr_binop_csr_canonical(const I n_row, const I n_col,
                             const I Ap[], const I Aj[], const T Ax[],
                             const I Bp[], const I Bj[], const T Bx[],
                                   I Cp[],       I Cj[],       T2 Cx[],
                             const binary_op& op)
{
    // 适用于标准CSR矩阵的方法

    // 初始化 C 的行指针
    Cp[0] = 0;
    // 记录 C 的非零元素数量
    I nnz = 0;

    // 遍历每一行
    for(I i = 0; i < n_row; i++){
        // 获取当前行的起始位置和结束位置
        I A_pos = Ap[i];
        I B_pos = Bp[i];
        I A_end = Ap[i+1];
        I B_end = Bp[i+1];

        // 同时遍历 A 和 B 的当前行，直到某一行结束
        while(A_pos < A_end && B_pos < B_end){
            // 获取当前列索引
            I A_j = Aj[A_pos];
            I B_j = Bj[B_pos];

            // 如果 A_j 和 B_j 相等，进行二元操作
            if(A_j == B_j){
                T result = op(Ax[A_pos], Bx[B_pos]);
                // 如果操作结果非零，将结果存入 C
                if(result != 0){
                    Cj[nnz] = A_j;
                    Cx[nnz] = result;
                    nnz++;
                }
                A_pos++;
                B_pos++;
            } else if (A_j < B_j) {
                // 如果 A_j 小于 B_j，在 A 上执行二元操作
                T result = op(Ax[A_pos], 0);
                // 如果操作结果非零，将结果存入 C
                if (result != 0){
                    Cj[nnz] = A_j;
                    Cx[nnz] = result;
                    nnz++;
                }
                A_pos++;
            } else {
                // 如果 B_j 小于 A_j，在 B 上执行二元操作
                T result = op(0, Bx[B_pos]);
                // 如果操作结果非零，将结果存入 C
                if (result != 0){
                    Cj[nnz] = B_j;
                    Cx[nnz] = result;
                    nnz++;
                }
                B_pos++;
            }
        }

        // 处理 A 或 B 中剩余的元素（tail）
        while(A_pos < A_end){
            T result = op(Ax[A_pos], 0);
            // 如果操作结果非零，将结果存入 C
            if (result != 0){
                Cj[nnz] = Aj[A_pos];
                Cx[nnz] = result;
                nnz++;
            }
            A_pos++;
        }
        while(B_pos < B_end){
            T result = op(0, Bx[B_pos]);
            // 如果操作结果非零，将结果存入 C
            if (result != 0){
                Cj[nnz] = Bj[B_pos];
                Cx[nnz] = result;
                nnz++;
            }
            B_pos++;
        }

        // 更新下一行的行指针
        Cp[i+1] = nnz;
    }
}
/*
 * Compute C = A (binary_op) B for CSR matrices A,B where the column
 * indices with the rows of A and B are known to be sorted.
 *
 *   binary_op(x,y) - binary operator to apply elementwise
 *
 * Input Arguments:
 *   I    n_row       - number of rows in A (and B)
 *   I    n_col       - number of columns in A (and B)
 *   I    Ap[n_row+1] - row pointer
 *   I    Aj[nnz(A)]  - column indices
 *   T    Ax[nnz(A)]  - nonzeros
 *   I    Bp[n_row+1] - row pointer
 *   I    Bj[nnz(B)]  - column indices
 *   T    Bx[nnz(B)]  - nonzeros
 * Output Arguments:
 *   I    Cp[n_row+1] - row pointer
 *   I    Cj[nnz(C)]  - column indices
 *   T    Cx[nnz(C)]  - nonzeros
 *
 * Note:
 *   Output arrays Cp, Cj, and Cx must be preallocated
 *   If nnz(C) is not known a priori, a conservative bound is:
 *          nnz(C) <= nnz(A) + nnz(B)
 *
 * Note:
 *   Input:  A and B column indices are not assumed to be in sorted order.
 *   Output: C column indices will be in sorted if both A and B have sorted indices.
 *           Cx will not contain any zero entries
 *
 */

template <class I, class T, class T2, class binary_op>
void csr_binop_csr(const I n_row,
                   const I n_col,
                   const I Ap[],
                   const I Aj[],
                   const T Ax[],
                   const I Bp[],
                   const I Bj[],
                   const T Bx[],
                         I Cp[],
                         I Cj[],
                         T2 Cx[],
                   const binary_op& op)
{
    // 检查输入的 CSR 矩阵 A 和 B 是否处于规范格式（行索引必须排序）
    if (csr_has_canonical_format(n_row, Ap, Aj) && csr_has_canonical_format(n_row, Bp, Bj))
        // 若都处于规范格式，则调用专门的规范格式矩阵运算函数
        csr_binop_csr_canonical(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, op);
    else
        // 否则调用通用的 CSR 矩阵运算函数
        csr_binop_csr_general(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, op);
}

/* element-wise binary operations*/

template <class I, class T, class T2>
void csr_ne_csr(const I n_row, const I n_col,
                const I Ap[], const I Aj[], const T Ax[],
                const I Bp[], const I Bj[], const T Bx[],
                      I Cp[],       I Cj[],      T2 Cx[])
{
    // 对 CSR 矩阵 A 和 B 执行元素级不等于操作
    csr_binop_csr(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, std::not_equal_to<T>());
}

template <class I, class T, class T2>
void csr_lt_csr(const I n_row, const I n_col,
                const I Ap[], const I Aj[], const T Ax[],
                const I Bp[], const I Bj[], const T Bx[],
                      I Cp[],       I Cj[],      T2 Cx[])
{
    // 对 CSR 矩阵 A 和 B 执行元素级小于操作
    csr_binop_csr(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, std::less<T>());
}

template <class I, class T, class T2>
void csr_gt_csr(const I n_row, const I n_col,
                const I Ap[], const I Aj[], const T Ax[],
                const I Bp[], const I Bj[], const T Bx[],
                      I Cp[],       I Cj[],      T2 Cx[])
{
    // 对 CSR 矩阵 A 和 B 执行元素级大于操作
    csr_binop_csr(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, std::greater<T>());
}
// 将 CSR 格式的矩阵 A 和 B 进行按元素比较操作（<=），结果存储在 CSR 格式的矩阵 C 中
void csr_le_csr(const I n_row, const I n_col,
                const I Ap[], const I Aj[], const T Ax[],
                const I Bp[], const I Bj[], const T Bx[],
                      I Cp[],       I Cj[],      T2 Cx[])
{
    // 调用 csr_binop_csr 函数，使用 std::less_equal<T>() 比较器
    csr_binop_csr(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, std::less_equal<T>());
}

// 将 CSR 格式的矩阵 A 和 B 进行按元素比较操作（>=），结果存储在 CSR 格式的矩阵 C 中
template <class I, class T, class T2>
void csr_ge_csr(const I n_row, const I n_col,
                const I Ap[], const I Aj[], const T Ax[],
                const I Bp[], const I Bj[], const T Bx[],
                      I Cp[],       I Cj[],      T2 Cx[])
{
    // 调用 csr_binop_csr 函数，使用 std::greater_equal<T>() 比较器
    csr_binop_csr(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, std::greater_equal<T>());
}

// 将 CSR 格式的矩阵 A 和 B 进行按元素乘法操作，结果存储在 CSR 格式的矩阵 C 中
template <class I, class T>
void csr_elmul_csr(const I n_row, const I n_col,
                   const I Ap[], const I Aj[], const T Ax[],
                   const I Bp[], const I Bj[], const T Bx[],
                         I Cp[],       I Cj[],       T Cx[])
{
    // 调用 csr_binop_csr 函数，使用 std::multiplies<T>() 乘法器
    csr_binop_csr(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, std::multiplies<T>());
}

// 将 CSR 格式的矩阵 A 和 B 进行按元素除法操作，结果存储在 CSR 格式的矩阵 C 中
template <class I, class T>
void csr_eldiv_csr(const I n_row, const I n_col,
                   const I Ap[], const I Aj[], const T Ax[],
                   const I Bp[], const I Bj[], const T Bx[],
                         I Cp[],       I Cj[],       T Cx[])
{
    // 调用 csr_binop_csr 函数，使用 safe_divides<T>() 除法器
    csr_binop_csr(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, safe_divides<T>());
}

// 将 CSR 格式的矩阵 A 和 B 进行按元素加法操作，结果存储在 CSR 格式的矩阵 C 中
template <class I, class T>
void csr_plus_csr(const I n_row, const I n_col,
                  const I Ap[], const I Aj[], const T Ax[],
                  const I Bp[], const I Bj[], const T Bx[],
                        I Cp[],       I Cj[],       T Cx[])
{
    // 调用 csr_binop_csr 函数，使用 std::plus<T>() 加法器
    csr_binop_csr(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, std::plus<T>());
}

// 将 CSR 格式的矩阵 A 和 B 进行按元素减法操作，结果存储在 CSR 格式的矩阵 C 中
template <class I, class T>
void csr_minus_csr(const I n_row, const I n_col,
                   const I Ap[], const I Aj[], const T Ax[],
                   const I Bp[], const I Bj[], const T Bx[],
                         I Cp[],       I Cj[],       T Cx[])
{
    // 调用 csr_binop_csr 函数，使用 std::minus<T>() 减法器
    csr_binop_csr(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, std::minus<T>());
}

// 将 CSR 格式的矩阵 A 和 B 进行按元素取最大值操作，结果存储在 CSR 格式的矩阵 C 中
template <class I, class T>
void csr_maximum_csr(const I n_row, const I n_col,
                     const I Ap[], const I Aj[], const T Ax[],
                     const I Bp[], const I Bj[], const T Bx[],
                           I Cp[],       I Cj[],       T Cx[])
{
    // 调用 csr_binop_csr 函数，使用 maximum<T>() 求最大值器
    csr_binop_csr(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, maximum<T>());
}

// 将 CSR 格式的矩阵 A 和 B 进行按元素取最小值操作，结果存储在 CSR 格式的矩阵 C 中
template <class I, class T>
void csr_minimum_csr(const I n_row, const I n_col,
                     const I Ap[], const I Aj[], const T Ax[],
                     const I Bp[], const I Bj[], const T Bx[],
                           I Cp[],       I Cj[],       T Cx[])
{
    // 调用 csr_binop_csr 函数，使用 minimum<T>() 求最小值器
    csr_binop_csr(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, minimum<T>());
}
/*
 * Sum together duplicate column entries in each row of CSR matrix A
 *
 *
 * Input Arguments:
 *   I    n_row       - number of rows in A (and B)
 *   I    n_col       - number of columns in A (and B)
 *   I    Ap[n_row+1] - row pointer
 *   I    Aj[nnz(A)]  - column indices
 *   T    Ax[nnz(A)]  - nonzeros
 *
 * Note:
 *   The column indices within each row must be in sorted order.
 *   Explicit zeros are retained.
 *   Ap, Aj, and Ax will be modified *inplace*
 *
 */
template <class I, class T>
void csr_sum_duplicates(const I n_row,
                        const I n_col,
                              I Ap[],
                              I Aj[],
                              T Ax[])
{
    I nnz = 0;                    // Initialize the count of nonzeros
    I row_end = 0;                // Initialize the end index of current row in Ap
    for(I i = 0; i < n_row; i++){ // Iterate over each row of the CSR matrix
        I jj = row_end;           // Start index of current row in Aj and Ax
        row_end = Ap[i+1];        // End index of current row in Ap
        while( jj < row_end ){    // Iterate over columns of current row
            I j = Aj[jj];         // Column index of current entry
            T x = Ax[jj];         // Value of current entry
            jj++;                 // Move to the next entry in the current row
            while( jj < row_end && Aj[jj] == j ){  // Sum duplicates of the same column index
                x += Ax[jj];      // Accumulate values for duplicate column indices
                jj++;             // Move to the next entry in the current row
            }
            Aj[nnz] = j;          // Store the column index with summed value
            Ax[nnz] = x;          // Store the summed value
            nnz++;                // Increment the count of nonzeros
        }
        Ap[i+1] = nnz;            // Update the end index of current row in Ap
    }
}

/*
 * Eliminate zero entries from CSR matrix A
 *
 *
 * Input Arguments:
 *   I    n_row       - number of rows in A (and B)
 *   I    n_col       - number of columns in A (and B)
 *   I    Ap[n_row+1] - row pointer
 *   I    Aj[nnz(A)]  - column indices
 *   T    Ax[nnz(A)]  - nonzeros
 *
 * Note:
 *   Ap, Aj, and Ax will be modified *inplace*
 *
 */
template <class I, class T>
void csr_eliminate_zeros(const I n_row,
                         const I n_col,
                               I Ap[],
                               I Aj[],
                               T Ax[])
{
    I nnz = 0;                    // Initialize the count of nonzeros
    I row_end = 0;                // Initialize the end index of current row in Ap
    for(I i = 0; i < n_row; i++){ // Iterate over each row of the CSR matrix
        I jj = row_end;           // Start index of current row in Aj and Ax
        row_end = Ap[i+1];        // End index of current row in Ap
        while( jj < row_end ){    // Iterate over columns of current row
            I j = Aj[jj];         // Column index of current entry
            T x = Ax[jj];         // Value of current entry
            if(x != 0){           // Check if the value is nonzero
                Aj[nnz] = j;      // Store the column index
                Ax[nnz] = x;      // Store the nonzero value
                nnz++;            // Increment the count of nonzeros
            }
            jj++;                 // Move to the next entry in the current row
        }
        Ap[i+1] = nnz;            // Update the end index of current row in Ap
    }
}



/*
 * Compute Y += A*X for CSR matrix A and dense vectors X,Y
 *
 *
 * Input Arguments:
 *   I  n_row         - number of rows in A
 *   I  n_col         - number of columns in A
 *   I  Ap[n_row+1]   - row pointer
 *   I  Aj[nnz(A)]    - column indices
 *   T  Ax[nnz(A)]    - nonzeros
 *   T  Xx[n_col]     - input vector
 *
 * Output Arguments:
 *   T  Yx[n_row]     - output vector
 *
 * Note:
 *   Output array Yx must be preallocated
 *
 *   Complexity: Linear.  Specifically O(nnz(A) + n_row)
 *
 */
template <class I, class T>
void csr_matvec(const I n_row,
                const I n_col,
                const I Ap[],
                const I Aj[],
                const T Ax[],
                const T Xx[],
                      T Yx[])
{
    # 对每一行进行迭代，计算稀疏矩阵向量乘法的结果
    for(I i = 0; i < n_row; i++){
        # 初始化每行的加权和为该行的偏置项
        T sum = Yx[i];
        # 遍历第 i 行的非零元素
        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            # 将稀疏矩阵的值乘以对应的向量值，加到该行的加权和上
            sum += Ax[jj] * Xx[Aj[jj]];
        }
        # 更新 Yx[i] 为计算得到的加权和
        Yx[i] = sum;
    }
/*
 * Slice rows given as an array of indices from a compressed sparse row (CSR) matrix.
 *
 * Input Arguments:
 *   I  n_row_idx       - number of row indices to slice
 *   I  rows[n_row_idx] - array of row indices to extract
 *   I  Ap[n_row+1]     - CSR row pointer array
 *   I  Aj[nnz(A)]      - CSR column indices array
 *   T  Ax[nnz(A)]      - CSR data array
 *
 * Output Arguments:
 *   I  Bj              - new column indices after slicing rows
 *   T  Bx              - new data after slicing rows
 *
 */
template<class I, class T>
/*
 * 对 CSR 格式的稀疏矩阵按行索引，复制对应行的列索引和数据。
 *
 * 输入参数：
 *   I  n_row_idx    - 要索引的行数
 *   I  rows[n_row_idx] - 要索引的行的索引数组
 *   I  Ap[]         - 行指针数组，指示每行起始位置
 *   I  Aj[nnz(A)]   - 列索引数组
 *   T  Ax[nnz(A)]   - 数据数组
 *
 * 输出参数：
 *   I  Bj[]         - 新的列索引数组
 *   T  Bx[]         - 新的数据数组
 *
 */
void csr_row_index(const I n_row_idx,
                   const I rows[],
                   const I Ap[],
                   const I Aj[],
                   const T Ax[],
                   I Bj[],
                   T Bx[])
{
    for(I i = 0; i < n_row_idx; i++){
        const I row = rows[i];              // 当前要索引的行号
        const I row_start = Ap[row];        // 当前行在 Aj 中的起始位置
        const I row_end   = Ap[row+1];      // 当前行在 Aj 中的结束位置（不含）
        Bj = std::copy(Aj + row_start, Aj + row_end, Bj);  // 复制当前行的列索引到 Bj
        Bx = std::copy(Ax + row_start, Ax + row_end, Bx);  // 复制当前行的数据到 Bx
    }
}


/*
 * 对 CSR 格式的稀疏矩阵按行切片，复制给定范围内行的列索引和数据。
 *
 * 输入参数：
 *   I  start        - 切片的起始行
 *   I  stop         - 切片的结束行
 *   I  step         - 行步长
 *   I  Ap[]         - 行指针数组，指示每行起始位置
 *   I  Aj[nnz(A)]   - 列索引数组
 *   T  Ax[nnz(A)]   - 数据数组
 *
 * 输出参数：
 *   I  Bj[]         - 新的列索引数组
 *   T  Bx[]         - 新的数据数组
 *
 */
template<class I, class T>
void csr_row_slice(const I start,
                   const I stop,
                   const I step,
                   const I Ap[],
                   const I Aj[],
                   const T Ax[],
                   I Bj[],
                   T Bx[])
{
    if (step > 0) {
        for(I row = start; row < stop; row += step){
            const I row_start = Ap[row];        // 当前行在 Aj 中的起始位置
            const I row_end   = Ap[row+1];      // 当前行在 Aj 中的结束位置（不含）
            Bj = std::copy(Aj + row_start, Aj + row_end, Bj);  // 复制当前行的列索引到 Bj
            Bx = std::copy(Ax + row_start, Ax + row_end, Bx);  // 复制当前行的数据到 Bx
        }
    } else {
        for(I row = start; row > stop; row += step){
            const I row_start = Ap[row];        // 当前行在 Aj 中的起始位置
            const I row_end   = Ap[row+1];      // 当前行在 Aj 中的结束位置（不含）
            Bj = std::copy(Aj + row_start, Aj + row_end, Bj);  // 复制当前行的列索引到 Bj
            Bx = std::copy(Ax + row_start, Ax + row_end, Bx);  // 复制当前行的数据到 Bx
        }
    }
}


/*
 * 对 CSR 格式的稀疏矩阵按列索引切片的第一步，统计索引的出现次数并计算新的 indptr。
 *
 * 输入参数：
 *   I  n_idx           - 要切片的列索引数
 *   I  col_idxs[n_idx] - 要切片的列索引数组
 *   I  n_row           - 主轴维度
 *   I  n_col           - 次轴维度
 *   I  Ap[n_row+1]     - 行指针数组，指示每行起始位置
 *   I  Aj[nnz(A)]      - 列索引数组
 *
 * 输出参数：
 *   I  col_offsets[n_col] - 索引重复计数的累加和
 *   I  Bp[n_row+1]        - 新的行指针数组
 *
 */
template<class I>
void csr_column_index1(const I n_idx,
                       const I col_idxs[],
                       const I n_row,
                       const I n_col,
                       const I Ap[],
                       const I Aj[],
                       I col_offsets[],
                       I Bp[])
{
    // 统计 col_idxs 中每个列索引的出现次数
    for(I jj = 0; jj < n_idx; jj++){
        const I j = col_idxs[jj];
        col_offsets[j]++;
    }

    // 计算新的 indptr
    I new_nnz = 0;
    Bp[0] = 0;
    for(I i = 0; i < n_row; i++){
        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            new_nnz += col_offsets[Aj[jj]];
        }
        Bp[i+1] = new_nnz;
    }

    // 在原地计算累加和
    for(I j = 1; j < n_col; j++){
        col_offsets[j] += col_offsets[j - 1];
    }
}
/*
 * Slice columns given as an array of indices (pass 2).
 * This pass populates indices/data entries for selected columns.
 *
 * Input Arguments:
 *   I  col_order[n_idx]   - order of col indices
 *   I  col_offsets[n_col] - cumsum of col index counts
 *   I  nnz                - nnz(A)
 *   I  Aj[nnz(A)]         - column indices
 *   T  Ax[nnz(A)]         - data
 *
 * Output Arguments:
 *   I  Bj[nnz(B)] - new column indices
 *   T  Bx[nnz(B)] - new data
 *
 */
template<class I, class T>
void csr_column_index2(const I col_order[],
                       const I col_offsets[],
                       const I nnz,
                       const I Aj[],
                       const T Ax[],
                       I Bj[],
                       T Bx[])
{
    I n = 0;  // Initialize output index counter
    for(I jj = 0; jj < nnz; jj++){
        const I j = Aj[jj];  // Current column index from input
        const I offset = col_offsets[j];  // Offset into col_order for current column
        const I prev_offset = j == 0 ? 0 : col_offsets[j-1];  // Previous offset into col_order
        if (offset != prev_offset) {  // Check if column has non-zero entries
            const T v = Ax[jj];  // Value corresponding to current column index
            for(I k = prev_offset; k < offset; k++){
                Bj[n] = col_order[k];  // New column indices based on col_order
                Bx[n] = v;  // Corresponding data values for new column indices
                n++;  // Increment output index counter
            }
        }
    }
}


/*
 * Count the number of occupied diagonals in CSR matrix A
 *
 * Input Arguments:
 *   I  n_row       - number of rows in A
 *   I  Ap[n_row+1] - row pointer
 *   I  Aj[nnz(A)]  - column indices
 *
 * Return:
 *   Number of distinct diagonals with non-zero elements
 */
template <class I>
I csr_count_diagonals(const I n_row,
                      const I Ap[],
                      const I Aj[])
{
    std::set<I> diagonals;  // Set to store distinct diagonal indices

    for(I i = 0; i < n_row; i++){
        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            diagonals.insert(Aj[jj] - i);  // Calculate and insert diagonal index
        }
    }
    return diagonals.size();  // Return number of distinct diagonals
}


/*
 * Sample the matrix at specific locations
 *
 * Determine the matrix value for each row,col pair
 *    Bx[n] = A(Bi[n],Bj[n])
 *
 * Input Arguments:
 *   I  n_row         - number of rows in A
 *   I  n_col         - number of columns in A
 *   I  Ap[n_row+1]   - row pointer
 *   I  Aj[nnz(A)]    - column indices
 *   T  Ax[nnz(A)]    - nonzeros
 *   I  n_samples     - number of samples
 *   I  Bi[N]         - sample rows
 *   I  Bj[N]         - sample columns
 *
 * Output Arguments:
 *   T  Bx[N]         - sample values
 *
 * Note:
 *   Output array Bx must be preallocated
 *
 *   Complexity: varies
 *
 *   TODO handle other cases with asymptotically optimal method
 *
 */
template <class I, class T>
void csr_sample_values(const I n_row,
                       const I n_col,
                       const I Ap[],
                       const I Aj[],
                       const T Ax[],
                       const I n_samples,
                       const I Bi[],
                       const I Bj[],
                             T Bx[])
{
    // ideally we'd do the following
    // Case 1: A is canonical and B is sorted by row and column
    //   -> special purpose csr_binop_csr() (optimized form)
    // Case 2: A is canonical and B is unsorted and max(log(Ap[i+1] - Ap[i])) > log(num_samples)
    //   -> do binary searches for each sample
    // Case 3: A is canonical and B is unsorted and max(log(Ap[i+1] - Ap[i])) < log(num_samples)
    //   -> sort B by row and column and use Case 1
    // Case 4: A is not canonical and num_samples ~ nnz
    //   -> special purpose csr_binop_csr() (general form)
    // Case 5: A is not canonical and num_samples << nnz
    //   -> do linear searches for each sample

    const I nnz = Ap[n_row];  // 获取稀疏矩阵 A 的非零元素数量

    const I threshold = nnz / 10; // 使用 nnz 的十分之一作为阈值，这个常数是任意选择的

    if (n_samples > threshold && csr_has_canonical_format(n_row, Ap, Aj))
    {
        for(I n = 0; n < n_samples; n++)
        {
            const I i = Bi[n] < 0 ? Bi[n] + n_row : Bi[n]; // 样本所在的行
            const I j = Bj[n] < 0 ? Bj[n] + n_col : Bj[n]; // 样本所在的列

            const I row_start = Ap[i];   // 行 i 的起始索引
            const I row_end   = Ap[i+1]; // 行 i 的结束索引

            if (row_start < row_end)
            {
                // 在行 i 中使用二分搜索找到列 j 的索引
                const I offset = std::lower_bound(Aj + row_start, Aj + row_end, j) - Aj;

                if (offset < row_end && Aj[offset] == j)
                    Bx[n] = Ax[offset]; // 如果找到了，将对应元素值赋给 Bx[n]
                else
                    Bx[n] = 0; // 否则置零
            }
            else
            {
                Bx[n] = 0; // 如果行 i 为空，则 Bx[n] 置零
            }

        }
    }
    else
    {
        for(I n = 0; n < n_samples; n++)
        {
            const I i = Bi[n] < 0 ? Bi[n] + n_row : Bi[n]; // 样本所在的行
            const I j = Bj[n] < 0 ? Bj[n] + n_col : Bj[n]; // 样本所在的列

            const I row_start = Ap[i];   // 行 i 的起始索引
            const I row_end   = Ap[i+1]; // 行 i 的结束索引

            T x = 0; // 初始化结果值为零

            for(I jj = row_start; jj < row_end; jj++)
            {
                if (Aj[jj] == j)
                    x += Ax[jj]; // 累加行 i 中与列 j 对应的非零元素值到 x
            }

            Bx[n] = x; // 将累加结果赋给 Bx[n]
        }

    }
/*
 * Determine the data offset at specific locations
 *
 * Input Arguments:
 *   I  n_row         - number of rows in A
 *   I  n_col         - number of columns in A
 *   I  Ap[n_row+1]   - row pointer
 *   I  Aj[nnz(A)]    - column indices
 *   I  n_samples     - number of samples
 *   I  Bi[N]         - sample rows
 *   I  Bj[N]         - sample columns
 *
 * Output Arguments:
 *   I  Bp[N]         - offsets into Aj; -1 if non-existent
 *
 * Return value:
 *   1 if any sought entries are duplicated, in which case the
 *   function has exited early; 0 otherwise.
 *
 * Note:
 *   Output array Bp must be preallocated
 *
 *   Complexity: varies. See csr_sample_values
 *
 */
template <class I>
int csr_sample_offsets(const I n_row,
                       const I n_col,
                       const I Ap[],
                       const I Aj[],
                       const I n_samples,
                       const I Bi[],
                       const I Bj[],
                             I Bp[])
{
    const I nnz = Ap[n_row];
    const I threshold = nnz / 10; // constant is arbitrary

    // Check if the number of samples exceeds threshold and the CSR matrix has canonical format
    if (n_samples > threshold && csr_has_canonical_format(n_row, Ap, Aj))
    {
        // Loop through each sample
        for(I n = 0; n < n_samples; n++)
        {
            const I i = Bi[n] < 0 ? Bi[n] + n_row : Bi[n]; // sample row
            const I j = Bj[n] < 0 ? Bj[n] + n_col : Bj[n]; // sample column

            const I row_start = Ap[i];
            const I row_end   = Ap[i+1];

            // Binary search for column j within the row slice
            if (row_start < row_end)
            {
                const I offset = std::lower_bound(Aj + row_start, Aj + row_end, j) - Aj;

                // Check if column j is found in the row slice
                if (offset < row_end && Aj[offset] == j)
                    Bp[n] = offset; // Store the offset into Bp
                else
                    Bp[n] = -1; // Column j not found
            }
            else
            {
                Bp[n] = -1; // Empty row
            }
        }
    }
    else // Fallback for when conditions are not met
    {
        // Loop through each sample
        for(I n = 0; n < n_samples; n++)
        {
            const I i = Bi[n] < 0 ? Bi[n] + n_row : Bi[n]; // sample row
            const I j = Bj[n] < 0 ? Bj[n] + n_col : Bj[n]; // sample column

            const I row_start = Ap[i];
            const I row_end   = Ap[i+1];

            I offset = -1;

            // Sequential search for column j within the row slice
            for(I jj = row_start; jj < row_end; jj++)
            {
                if (Aj[jj] == j) {
                    offset = jj;
                    // Check for duplicate entries of column j within the row
                    for (jj++; jj < row_end; jj++) {
                        if (Aj[jj] == j) {
                            offset = -2; // Indicates duplicated entry
                            return 1; // Early exit due to duplication
                        }
                    }
                }
            }
            Bp[n] = offset; // Store the offset into Bp
        }
    }
    return 0; // Function completed successfully
}
/*
 * Stack CSR matrices in A horizontally (column wise)
 *
 * Input Arguments:
 *   I  n_blocks                      - number of matrices in A
 *   I  n_row                         - number of rows in any matrix in A
 *   I  n_col_cat[n_blocks]           - number of columns in each matrix in A concatenated
 *   I  Ap_cat[n_blocks*(n_row + 1)]  - row indices of each matrix in A concatenated
 *   I  Aj_cat[nnz(A)]                - column indices of each matrix in A concatenated
 *   T  Ax_cat[nnz(A)]                - nonzeros of each matrix in A concatenated
 *
 * Output Arguments:
 *   I Bp  - row pointer
 *   I Bj  - column indices
 *   T Bx  - nonzeros
 *
 * Note:
 *   All output arrays Bp, Bj, Bx must be preallocated
 *
 *   Complexity: Linear.  Specifically O(nnz(A) + n_blocks)
 *
 */
template <class I, class T>
void csr_hstack(const I n_blocks,
                const I n_row,
                const I n_col_cat[],
                const I Ap_cat[],
                const I Aj_cat[],
                const T Ax_cat[],
                      I Bp[],
                      I Bj[],
                      T Bx[])
{
    // First, mark the blocks in the input data while
    // computing their column offsets:
    std::vector<I> col_offset(n_blocks);  // 创建一个大小为 n_blocks 的整数向量 col_offset，用于存储每个块的列偏移量
    std::vector<const I*> bAp(n_blocks);  // 创建一个大小为 n_blocks 的指针向量 bAp，存储指向每个块的行索引数组的指针
    std::vector<const I*> bAj(n_blocks);  // 创建一个大小为 n_blocks 的指针向量 bAj，存储指向每个块的列索引数组的指针
    std::vector<const T*> bAx(n_blocks);  // 创建一个大小为 n_blocks 的指针向量 bAx，存储指向每个块的非零值数组的指针
    col_offset[0] = 0;                    // 初始化第一个块的列偏移量为 0
    bAp[0] = Ap_cat;                      // 将第一个块的行索引数组的指针指向 Ap_cat 数组
    bAj[0] = Aj_cat;                      // 将第一个块的列索引数组的指针指向 Aj_cat 数组
    bAx[0] = Ax_cat;                      // 将第一个块的非零值数组的指针指向 Ax_cat 数组
    for (I b = 1; b < n_blocks; b++){     // 循环遍历每个块，计算列偏移量并设置指针
        col_offset[b] = col_offset[b - 1] + n_col_cat[b - 1];  // 计算当前块的列偏移量
        bAp[b] = bAp[b - 1] + (n_row + 1);  // 设置当前块的行索引数组指针
        bAj[b] = bAj[b - 1] + bAp[b - 1][n_row];  // 设置当前块的列索引数组指针
        bAx[b] = bAx[b - 1] + bAp[b - 1][n_row];  // 设置当前块的非零值数组指针
    }

    // Next, build the full output matrix:
    Bp[0] = 0;  // 初始化输出行指针数组的第一个元素为 0
    I s = 0;    // 初始化计数器 s 为 0
    for(I i = 0; i < n_row; i++){  // 外层循环遍历每一行
        for (I b = 0; b < n_blocks; b++){  // 内层循环遍历每个块
            I jj_start = bAp[b][i];  // 获取当前块在第 i 行的起始列索引
            I jj_end = bAp[b][i + 1];  // 获取当前块在第 i 行的结束列索引
            I offset = col_offset[b];  // 获取当前块的列偏移量
            // 将当前块的列索引数组中的值加上列偏移量，存入输出列索引数组 Bj 中
            std::transform(&bAj[b][jj_start], &bAj[b][jj_end],
                           &Bj[s], [&](I x){return (x + offset);});
            // 将当前块的非零值数组的值复制到输出非零值数组 Bx 中
            std::copy(&bAx[b][jj_start], &bAx[b][jj_end], &Bx[s]);
            s += jj_end - jj_start;  // 更新计数器 s
        }
        Bp[i + 1] = s;  // 设置输出行指针数组的下一行的值为 s
    }

}
```