# `D:\src\scipysrc\scipy\scipy\sparse\sparsetools\bsr.h`

```
// 防止头文件重复包含的宏定义
#ifndef __BSR_H__
#define __BSR_H__

// 包含必要的头文件：向量、算法、函数对象
#include <vector>
#include <algorithm>
#include <functional>

// 包含其他自定义头文件：CSR 和 Dense
#include "csr.h"
#include "dense.h"

// 定义静态内联函数：计算对角线大小
static inline npy_intp diagonal_size(const npy_intp k,
                                     const npy_intp rows,
                                     const npy_intp cols)
{
    // 返回最小值：行数 + min(k, 0) 和 列数 - max(k, 0)
    return std::min(rows + std::min(k, (npy_intp)0),
                    cols - std::max(k, (npy_intp)0));
}

// BSR 格式矩阵的对角线提取函数模板
template <class I, class T>
void bsr_diagonal(const I k,
                  const I n_brow,
                  const I n_bcol,
                  const I R,
                  const I C,
                  const I Ap[],
                  const I Aj[],
                  const T Ax[],
                        T Yx[])
{
    // 计算块大小 RC = R * C
    const npy_intp RC = R * C;
    // 计算对角线大小 D
    const npy_intp D = diagonal_size(k, (npy_intp)n_brow * R,
                                        (npy_intp)n_bcol * C);
    // 计算第一个行号
    const npy_intp first_row = (k >= 0) ? 0 : -k;
    /* First and next-to-last brows of the diagonal. */
    // 计算对角线的第一个和倒数第二个块的行号
    const npy_intp first_brow = first_row / R;
    const npy_intp last_brow = (first_row + D - 1) / R + 1;

    // 遍历每个块行
    for (npy_intp brow = first_brow; brow < last_brow; ++brow) {
        /* First and next-to-last bcols of the diagonal in this brow. */
        // 计算此行中对角线的第一个和倒数第二个块列号
        const npy_intp first_bcol = (brow * R + k) / C;
        const npy_intp last_bcol = ((brow + 1) * R + k - 1) / C + 1;

        // 遍历每个块行中的每个块列
        for (npy_intp jj = Ap[brow]; jj < Ap[brow + 1]; ++jj) {
            const npy_intp bcol = Aj[jj];

            // 如果块列在当前对角线范围内
            if (first_bcol <= bcol && bcol < last_bcol) {
                /*
                 * 计算并提取对应于第 k 个整体对角线的块的对角线，
                 * 并将其添加到正确的位置输出。
                 */
                const npy_intp block_k = brow * R + k - bcol * C;
                const npy_intp block_D = diagonal_size(block_k, R, C);
                const npy_intp block_first_row = (block_k >= 0) ? 0 : -block_k;
                const npy_intp Y_idx = brow * R + block_first_row - first_row;
                const npy_intp Ax_idx = RC * jj +
                                        ((block_k >= 0) ? block_k :
                                                          -C * block_k);

                // 遍历块的每个对角线元素，将其加到输出中
                for (npy_intp kk = 0; kk < block_D; ++kk) {
                    Yx[Y_idx + kk] += Ax[Ax_idx + kk * (C + 1)];
                }
            }
        }
    }
}

/*
 * 对 BSR 矩阵的行进行缩放 *in place*
 *
 *   A[i,:] *= X[i]
 *
 */
template <class I, class T>
void bsr_scale_rows(const I n_brow,
                    const I n_bcol,
                    const I R,
                    const I C,
                    const I Ap[],
                    const I Aj[],
                          T Ax[],
                    const T Xx[])
{
    // 计算 RC = R * C
    const npy_intp RC = (npy_intp)R * C;
    # 外层循环遍历稀疏矩阵的每一行
    for(I i = 0; i < n_brow; i++){
        # 计算当前行在稀疏矩阵数据数组 Xx 中的起始位置
        const T * row_scales = Xx + (npy_intp)R*i;

        # 内层循环遍历当前行对应的非零元素
        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
           # 计算当前非零元素在稀疏矩阵值数组 Ax 中的起始位置
           T * block = Ax + RC*jj;

            # 内层循环遍历块中的每一行
            for(I bi = 0; bi < R; bi++){
                # 对块中的每个元素进行按行缩放
                scal(C, row_scales[bi], block + (npy_intp)C*bi);
            }
        }
    }
/*
 * 缩放 BSR 矩阵的每一列 *就地* 操作
 *
 *   A[:,i] *= X[i]
 *
 */
template <class I, class T>
void bsr_scale_columns(const I n_brow,
                       const I n_bcol,
                       const I R,
                       const I C,
                       const I Ap[],
                       const I Aj[],
                             T Ax[],
                       const T Xx[])
{
    const I bnnz = Ap[n_brow];  // 获取非零块的数量
    const npy_intp RC = (npy_intp)R*C;  // 计算每个块的元素数
    for(I i = 0; i < bnnz; i++){  // 循环处理每个非零块
        const T * scales = Xx + (npy_intp)C * Aj[i];  // 获取缩放系数
        T * block = Ax + RC * i;  // 获取当前块的指针

        for(I bi = 0; bi < R; bi++){  // 循环处理块中的每一行
            for(I bj = 0; bj < C; bj++){  // 循环处理块中的每一列
                block[C * bi + bj] *= scales[bj];  // 对块中的每个元素应用缩放因子
            }
        }

    }
}



/*
 * 对 BSR 矩阵的列块索引进行就地排序
 *
 * 输入参数:
 *   I  n_brow        - A 矩阵中的行块数
 *   I  n_bcol        - A 矩阵中的列块数
 *   I  R             - 每个块的行数
 *   I  C             - 每个块的列数
 *   I  Ap[n_brow+1]  - 行指针
 *   I  Aj[nblk(A)]   - 列索引
 *   T  Ax[nnz(A)]    - 非零元素
 *
 */
template <class I, class T>
void bsr_sort_indices(const I n_brow,
                      const I n_bcol,
                      const I R,
                      const I C,
                            I Ap[],
                            I Aj[],
                            T Ax[])
{
    if( R == 1 && C == 1 ){
        csr_sort_indices(n_brow, Ap, Aj, Ax);  // 如果块大小为1x1，则调用 CSR 排序函数
        return;
    }


    const I nblks = Ap[n_brow];  // 获取块的数量
    const npy_intp RC = (npy_intp)R*C;  // 计算每个块的元素数
    const npy_intp nnz = (npy_intp)RC*nblks;  // 计算总的非零元素数

    // 使用 CSR 算法计算块的排列顺序
    std::vector<I> perm(nblks);

    for(I i = 0; i < nblks; i++)
        perm[i] = i;

    csr_sort_indices(n_brow, Ap, Aj, &perm[0]);

    // 创建副本以便进行安全的排序操作
    std::vector<T> Ax_copy(nnz);
    std::copy(Ax, Ax + nnz, Ax_copy.begin());

    // 将排列后的块复制回原数组
    for(I i = 0; i < nblks; i++){
        const T * input = &Ax_copy[RC * perm[i]];
              T * output = Ax + RC * i;
        std::copy(input, input + RC, output);
    }
}


/*
 * 计算 BSR 矩阵 A 的转置矩阵
 *
 * 输入参数:
 *   I  n_brow        - A 矩阵中的行块数
 *   I  n_bcol        - A 矩阵中的列块数
 *   I  R             - 每个块的行数
 *   I  C             - 每个块的列数
 *   I  Ap[n_brow+1]  - 行指针
 *   I  Aj[nblk(A)]   - 列索引
 *   T  Ax[nnz(A)]    - 非零元素
 *
 * 输出参数:
 *   I  Bp[n_col+1]   - 行指针
 *   I  Bj[nblk(A)]   - 列索引
 *   T  Bx[nnz(A)]    - 非零元素
 *
 * 注意:
 *   输出数组 Bp, Bj, Bx 必须预先分配空间
 *
 * 注意:
 *   输入: 列索引 *不假设* 是按排序顺序的
 *   输出: 行索引 *将会* 按排序顺序排列
 *
 *   复杂度: 线性，具体为 O(nnz(A) + max(n_row, n_col))
 *
 */
template <class I, class T>
void bsr_transpose(const I n_brow,
                   const I n_bcol,
                   const I R,
                   const I C,
                   const I Ap[],
                   const I Aj[],
                   const T Ax[],
                         I Bp[],
                         I Bj[],
                         T Bx[])
{
    const I nblks = Ap[n_brow];  // 获取块的数量
    const npy_intp RC    = (npy_intp)R*C;  // 计算每个块的元素个数

    // 计算块的排列顺序，使用CSR格式进行转置
    std::vector<I> perm_in (nblks);  // 创建输入块的排列数组
    std::vector<I> perm_out(nblks);  // 创建输出块的排列数组

    for(I i = 0; i < nblks; i++)
        perm_in[i] = i;  // 初始化输入排列数组

    csr_tocsc(n_brow, n_bcol, Ap, Aj, &perm_in[0], Bp, Bj, &perm_out[0]);  // 调用CSR到CSC格式转置函数

    for(I i = 0; i < nblks; i++){
        const T * Ax_blk = Ax + RC * perm_out[i];  // 计算输入块的起始位置
              T * Bx_blk = Bx + RC * i;  // 计算输出块的起始位置
        for(I r = 0; r < R; r++){
            for(I c = 0; c < C; c++){
                Bx_blk[(npy_intp)c * R + r] = Ax_blk[(npy_intp)r * C + c];  // 对每个块进行转置操作
            }
        }
    }
}


template <class I, class T>
void bsr_matmat(const I maxnnz,
                const I n_brow,  const I n_bcol,
                const I R,       const I C,       const I N,
                const I Ap[],    const I Aj[],    const T Ax[],
                const I Bp[],    const I Bj[],    const T Bx[],
                      I Cp[],          I Cj[],          T Cx[])
{
    assert(R > 0 && C > 0 && N > 0);  // 断言R、C、N均大于0

    if( R == 1 && N == 1 && C == 1 ){
        // 对于1x1块大小，使用CSR矩阵乘法
        csr_matmat(n_brow, n_bcol, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx);
        return;
    }

    const npy_intp RC = (npy_intp)R*C;  // 计算每个块的元素个数
    const npy_intp RN = (npy_intp)R*N;  // 计算A块的元素个数
    const npy_intp NC = (npy_intp)N*C;  // 计算B块的元素个数

    std::fill( Cx, Cx + RC * maxnnz, 0 ); // 清空输出数组

    std::vector<I>  next(n_bcol,-1);  // 创建下一个非零块的数组
    std::vector<T*> mats(n_bcol);  // 创建块指针数组

    npy_intp nnz = 0;  // 非零块的计数器
    Cp[0] = 0;  // 初始化输出行指针

    for(I i = 0; i < n_brow; i++){
        I head   = -2;  // 初始化头部指针
        I length =  0;  // 初始化长度计数器

        I jj_start = Ap[i];  // 获取行i的起始位置
        I jj_end   = Ap[i+1];  // 获取行i的结束位置
        for(I jj = jj_start; jj < jj_end; jj++){
            I j = Aj[jj];  // 获取列j

            I kk_start = Bp[j];  // 获取列j的起始位置
            I kk_end   = Bp[j+1];  // 获取列j的结束位置
            for(I kk = kk_start; kk < kk_end; kk++){
                I k = Bj[kk];  // 获取列k

                if(next[k] == -1){
                    next[k] = head;  // 更新下一个非零块的头部指针
                    head = k;  // 更新头部指针
                    Cj[nnz] = k;  // 记录非零块列索引
                    mats[k] = Cx + RC*nnz;  // 设置块指针数组中的指针位置
                    nnz++;  // 非零块计数器加一
                    length++;  // 长度计数器加一
                }

                const T * A = Ax + jj*RN;  // 获取A块的起始位置
                const T * B = Bx + kk*NC;  // 获取B块的起始位置

                gemm(R, C, N, A, B, mats[k]);  // 对每个块执行矩阵乘法
            }
        }

        for(I jj = 0; jj < length; jj++){
            I temp = head;
            head = next[head];
            next[temp] = -1; // 清空数组
        }

        Cp[i+1] = nnz;  // 更新输出行指针
    }
}


template <class I, class T>
bool is_nonzero_block(const T block[], const I blocksize){
    for(I i = 0; i < blocksize; i++){
        if(block[i] != 0){
            return true;  // 判断块是否非零
        }
    }
    return false;  // 块全为零
}
    # 返回布尔值 false
    return false;
/*
 * 计算 C = A (binary_op) B，其中 A 和 B 是 BSR 格式的稀疏矩阵，可能包含非规范的格式。
 * 具体来说，该方法即使在输入矩阵的行中包含重复和/或未排序的列索引时也能正常工作。
 *
 * 参考 bsr_binop_bsr() 获取更多信息
 *
 * 注意：
 *   输出数组 Cp, Cj, 和 Cx 必须预先分配空间
 *   如果 nnz(C) 事先不知道，一个保守的上界是：
 *          nnz(C) <= nnz(A) + nnz(B)
 *
 * 注意：
 *   输入：A 和 B 的列索引不假定为排序顺序
 *   输出：C 的列索引通常不会排序
 *        C 不包含任何重复条目或显式零值。
 *
 */
template <class I, class T, class T2, class bin_op>
void bsr_binop_bsr_general(const I n_brow, const I n_bcol,
                           const I R,      const I C,
                           const I Ap[],   const I Aj[],   const T Ax[],
                           const I Bp[],   const I Bj[],   const T Bx[],
                                 I Cp[],         I Cj[],        T2 Cx[],
                           const bin_op& op)
{
    // 适用于包含重复和/或未排序索引的方法
    const npy_intp RC = (npy_intp)R*C;

    Cp[0] = 0;  // 初始化结果矩阵的行指针数组
    I nnz = 0;  // 初始化结果矩阵的非零元素计数

    std::vector<I>  next(n_bcol,     -1); // 每列的下一个元素索引，初始化为-1
    std::vector<T> A_row(n_bcol * RC, 0);   // 存储 A 矩阵行的临时数组，对于大的 R 可能会存在问题
    std::vector<T> B_row(n_bcol * RC, 0);   // 存储 B 矩阵行的临时数组，对于大的 R 可能会存在问题

    for(I i = 0; i < n_brow; i++){
        I head   = -2;  // 每行的头指针初始化为-2，表示空行
        I length =  0;  // 行长度初始化为0，即空行

        // 将 A 的一行加入 A_row
        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            I j = Aj[jj];

            for(I n = 0; n < RC; n++)
                A_row[RC*j + n] += Ax[RC*jj + n];  // 累加 A 的块数据到 A_row 中的对应位置

            if(next[j] == -1){
                next[j] = head;  // 更新链表指针
                head = j;         // 更新链表头
                length++;         // 增加行长度
            }
        }

        // 将 B 的一行加入 B_row
        for(I jj = Bp[i]; jj < Bp[i+1]; jj++){
            I j = Bj[jj];

            for(I n = 0; n < RC; n++)
                B_row[RC*j + n] += Bx[RC*jj + n];  // 累加 B 的块数据到 B_row 中的对应位置

            if(next[j] == -1){
                next[j] = head;  // 更新链表指针
                head = j;         // 更新链表头
                length++;         // 增加行长度
            }
        }

        // 处理当前行的非零块
        for(I jj = 0; jj < length; jj++){
            // 计算 op(block_A, block_B)
            for(I n = 0; n < RC; n++)
                Cx[RC * nnz + n] = op(A_row[RC*head + n], B_row[RC*head + n]);

            // 如果块是非零的，则更新列索引数组
            if( is_nonzero_block(Cx + (RC * nnz), RC) )
                Cj[nnz++] = head;

            // 清空 A_row 和 B_row 的值
            for(I n = 0; n < RC; n++){
                A_row[RC*head + n] = 0;
                B_row[RC*head + n] = 0;
            }

            I temp = head;
            head = next[head];
            next[temp] = -1;
        }

        Cp[i + 1] = nnz;  // 更新行指针数组的下一个位置
    }
}
/*
 * 计算 A (binary_op) B 的结果存储在 C 中，其中 A 和 B 是按照规范的 BSR 格式表示的稀疏矩阵。
 * 具体而言，此方法要求输入矩阵的行不含有重复的列索引，并且列索引按照排序顺序排列。
 *
 * 有关更多信息，请参考 bsr_binop_bsr() 方法
 *
 * 注意：
 *   输入：A 和 B 的列索引假定为排序顺序
 *   输出：C 的列索引将按排序顺序排列
 *         Cx 不包含任何零条目
 */
template <class I, class T, class T2, class bin_op>
void bsr_binop_bsr_canonical(const I n_brow, const I n_bcol,
                             const I R,      const I C,
                             const I Ap[],   const I Aj[],   const T Ax[],
                             const I Bp[],   const I Bj[],   const T Bx[],
                                   I Cp[],         I Cj[],        T2 Cx[],
                             const bin_op& op)
{
    const npy_intp RC = (npy_intp)R*C;  // 计算结果矩阵 C 的总元素数
    T2 * result = Cx;  // result 指向结果矩阵 C 的数据数组

    Cp[0] = 0;  // 初始化 C 的行指针数组，第一行的起始索引为 0
    I nnz = 0;  // 初始化 C 的非零元素计数器为 0
    // 遍历每一行的非零元素
    for(I i = 0; i < n_brow; i++){
        // 获取当前行在A矩阵中的起始位置和结束位置
        I A_pos = Ap[i];
        I A_end = Ap[i+1];
        // 获取当前行在B矩阵中的起始位置和结束位置
        I B_pos = Bp[i];
        I B_end = Bp[i+1];

        // 在A和B矩阵当前行都未结束时执行循环
        while(A_pos < A_end && B_pos < B_end){
            // 获取A和B矩阵当前位置的列索引
            I A_j = Aj[A_pos];
            I B_j = Bj[B_pos];

            // 如果A和B矩阵当前位置的列索引相同
            if(A_j == B_j){
                // 对于当前行中相同列的每个元素执行操作
                for(I n = 0; n < RC; n++){
                    result[n] = op(Ax[RC*A_pos + n], Bx[RC*B_pos + n]);
                }

                // 如果操作后的结果块非零
                if( is_nonzero_block(result, RC) ){
                    // 将结果块所在的列索引存入结果矩阵中
                    Cj[nnz] = A_j;
                    result += RC;
                    nnz++;
                }

                // 移动到A和B矩阵中的下一个位置
                A_pos++;
                B_pos++;
            } else if (A_j < B_j) {
                // 对于A矩阵中当前列小于B矩阵当前列的情况
                for(I n = 0; n < RC; n++){
                    result[n] = op(Ax[RC*A_pos + n], 0);
                }

                // 如果操作后的结果块非零
                if(is_nonzero_block(result, RC)){
                    // 将结果块所在的列索引存入结果矩阵中
                    Cj[nnz] = A_j;
                    result += RC;
                    nnz++;
                }

                // 移动到A矩阵中的下一个位置
                A_pos++;
            } else {
                // 对于B矩阵中当前列小于A矩阵当前列的情况
                for(I n = 0; n < RC; n++){
                    result[n] = op(0, Bx[RC*B_pos + n]);
                }
                // 如果操作后的结果块非零
                if(is_nonzero_block(result, RC)){
                    // 将结果块所在的列索引存入结果矩阵中
                    Cj[nnz] = B_j;
                    result += RC;
                    nnz++;
                }

                // 移动到B矩阵中的下一个位置
                B_pos++;
            }
        }

        // 处理A矩阵当前行剩余的元素
        while(A_pos < A_end){
            for(I n = 0; n < RC; n++){
                result[n] = op(Ax[RC*A_pos + n], 0);
            }

            // 如果操作后的结果块非零
            if(is_nonzero_block(result, RC)){
                // 将结果块所在的列索引存入结果矩阵中
                Cj[nnz] = Aj[A_pos];
                result += RC;
                nnz++;
            }

            // 移动到A矩阵中的下一个位置
            A_pos++;
        }
        
        // 处理B矩阵当前行剩余的元素
        while(B_pos < B_end){
            for(I n = 0; n < RC; n++){
                result[n] = op(0,Bx[RC*B_pos + n]);
            }

            // 如果操作后的结果块非零
            if(is_nonzero_block(result, RC)){
                // 将结果块所在的列索引存入结果矩阵中
                Cj[nnz] = Bj[B_pos];
                result += RC;
                nnz++;
            }

            // 移动到B矩阵中的下一个位置
            B_pos++;
        }

        // 更新列偏移数组中的下一个位置
        Cp[i+1] = nnz;
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
 */

template <class I, class T, class T2, class bin_op>
void bsr_binop_bsr(const I n_brow, const I n_bcol,
                   const I R,      const I C,
                   const I Ap[],   const I Aj[],   const T Ax[],
                   const I Bp[],   const I Bj[],   const T Bx[],
                         I Cp[],         I Cj[],        T2 Cx[],
                   const bin_op& op)
{
    assert( R > 0 && C > 0);  // Assert that block dimensions are positive

    // Check if the block size is 1x1, use CSR format for optimization
    if( R == 1 && C == 1 ){
        csr_binop_csr(n_brow, n_bcol, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, op);
    }
    else if ( csr_has_canonical_format(n_brow, Ap, Aj) && csr_has_canonical_format(n_brow, Bp, Bj) ){
        // Prefer faster implementation if both A and B have canonical format
        bsr_binop_bsr_canonical(n_brow, n_bcol, R, C, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, op);
    }
    else {
        // Fallback to general method if canonical format check fails
        bsr_binop_bsr_general(n_brow, n_bcol, R, C, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, op);
    }
}

/* element-wise binary operations */

template <class I, class T, class T2>
void bsr_ne_bsr(const I n_row, const I n_col, const I R, const I C,
                const I Ap[], const I Aj[], const T Ax[],
                const I Bp[], const I Bj[], const T Bx[],
                      I Cp[],       I Cj[],      T2 Cx[])
{
    // Apply not_equal_to<T> binary operator element-wise
    bsr_binop_bsr(n_row,n_col,R,C,Ap,Aj,Ax,Bp,Bj,Bx,Cp,Cj,Cx,std::not_equal_to<T>());
}

template <class I, class T, class T2>
void bsr_lt_bsr(const I n_row, const I n_col, const I R, const I C,
                const I Ap[], const I Aj[], const T Ax[],
                const I Bp[], const I Bj[], const T Bx[],
                      I Cp[],       I Cj[],      T2 Cx[])
{
    // Apply less<T> binary operator element-wise
    bsr_binop_bsr(n_row,n_col,R,C,Ap,Aj,Ax,Bp,Bj,Bx,Cp,Cj,Cx,std::less<T>());
}

template <class I, class T, class T2>
// 对稀疏矩阵的按块压缩行存储格式 (BSR) 进行逐元素大于运算
void bsr_gt_bsr(const I n_row, const I n_col, const I R, const I C,
                const I Ap[], const I Aj[], const T Ax[],
                const I Bp[], const I Bj[], const T Bx[],
                      I Cp[],       I Cj[],      T2 Cx[])
{
    // 调用 bsr_binop_bsr 函数进行逐元素大于运算
    bsr_binop_bsr(n_row, n_col, R, C, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, std::greater<T>());
}

// 对稀疏矩阵的按块压缩行存储格式 (BSR) 进行逐元素小于等于运算
template <class I, class T, class T2>
void bsr_le_bsr(const I n_row, const I n_col, const I R, const I C,
                const I Ap[], const I Aj[], const T Ax[],
                const I Bp[], const I Bj[], const T Bx[],
                      I Cp[],       I Cj[],      T2 Cx[])
{
    // 调用 bsr_binop_bsr 函数进行逐元素小于等于运算
    bsr_binop_bsr(n_row, n_col, R, C, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, std::less_equal<T>());
}

// 对稀疏矩阵的按块压缩行存储格式 (BSR) 进行逐元素大于等于运算
template <class I, class T, class T2>
void bsr_ge_bsr(const I n_row, const I n_col, const I R, const I C,
                const I Ap[], const I Aj[], const T Ax[],
                const I Bp[], const I Bj[], const T Bx[],
                      I Cp[],       I Cj[],      T2 Cx[])
{
    // 调用 bsr_binop_bsr 函数进行逐元素大于等于运算
    bsr_binop_bsr(n_row, n_col, R, C, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, std::greater_equal<T>());
}

// 对稀疏矩阵的按块压缩行存储格式 (BSR) 进行逐元素乘法运算
template <class I, class T>
void bsr_elmul_bsr(const I n_row, const I n_col, const I R, const I C,
                   const I Ap[], const I Aj[], const T Ax[],
                   const I Bp[], const I Bj[], const T Bx[],
                         I Cp[],       I Cj[],       T Cx[])
{
    // 调用 bsr_binop_bsr 函数进行逐元素乘法运算
    bsr_binop_bsr(n_row, n_col, R, C, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, std::multiplies<T>());
}

// 对稀疏矩阵的按块压缩行存储格式 (BSR) 进行逐元素除法运算
template <class I, class T>
void bsr_eldiv_bsr(const I n_row, const I n_col, const I R, const I C,
                   const I Ap[], const I Aj[], const T Ax[],
                   const I Bp[], const I Bj[], const T Bx[],
                         I Cp[],       I Cj[],       T Cx[])
{
    // 调用 bsr_binop_bsr 函数进行逐元素除法运算
    bsr_binop_bsr(n_row, n_col, R, C, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, std::divides<T>());
}

// 对稀疏矩阵的按块压缩行存储格式 (BSR) 进行逐元素加法运算
template <class I, class T>
void bsr_plus_bsr(const I n_row, const I n_col, const I R, const I C,
                  const I Ap[], const I Aj[], const T Ax[],
                  const I Bp[], const I Bj[], const T Bx[],
                        I Cp[],       I Cj[],       T Cx[])
{
    // 调用 bsr_binop_bsr 函数进行逐元素加法运算
    bsr_binop_bsr(n_row, n_col, R, C, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, std::plus<T>());
}

// 对稀疏矩阵的按块压缩行存储格式 (BSR) 进行逐元素减法运算
template <class I, class T>
void bsr_minus_bsr(const I n_row, const I n_col, const I R, const I C,
                   const I Ap[], const I Aj[], const T Ax[],
                   const I Bp[], const I Bj[], const T Bx[],
                         I Cp[],       I Cj[],       T Cx[])
{
    // 调用 bsr_binop_bsr 函数进行逐元素减法运算
    bsr_binop_bsr(n_row, n_col, R, C, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, std::minus<T>());
}

// 对稀疏矩阵的按块压缩行存储格式 (BSR) 进行逐元素取最大值运算
template <class I, class T>
void bsr_maximum_bsr(const I n_row, const I n_col, const I R, const I C,
                     const I Ap[], const I Aj[], const T Ax[],
                     const I Bp[], const I Bj[], const T Bx[],
                           I Cp[],       I Cj[],       T Cx[])
{
    // 调用 bsr_binop_bsr 函数进行逐元素取最大值运算
    bsr_binop_bsr(n_row, n_col, R, C, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, maximum<T>());
}
/*
 * Perform element-wise binary operation on two BSR matrices A and B, and store the result in C
 *
 * Input Arguments:
 *   I  n_row        - number of block rows in A and B
 *   I  n_col        - number of block columns in A and B
 *   I  R            - row blocksize
 *   I  C            - column blocksize
 *   I  Ap[n_row+1]  - block row pointer for A
 *   I  Aj[nnz(A)]   - block column indices for A
 *   T  Ax[nnz(A)]   - nonzero blocks for A
 *   I  Bp[n_row+1]  - block row pointer for B
 *   I  Bj[nnz(B)]   - block column indices for B
 *   T  Bx[nnz(B)]   - nonzero blocks for B
 *
 * Output Arguments:
 *   I  Cp[n_row*R + 1] - block row pointer for C
 *   I  Cj[nnz(C)]      - block column indices for C
 *   T  Cx[nnz(C)]      - nonzero values for C
 *
 * Note:
 *   Uses a binary operation (minimum in this case) to compute C = A op B where op is the binary operation.
 */
void bsr_minimum_bsr(const I n_row, const I n_col, const I R, const I C,
                     const I Ap[], const I Aj[], const T Ax[],
                     const I Bp[], const I Bj[], const T Bx[],
                           I Cp[],       I Cj[],       T Cx[])
{
    // Call bsr_binop_bsr with minimum operator to perform binary operation on BSR matrices
    bsr_binop_bsr(n_row, n_col, R, C, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, minimum<T>());
}


/*
 * Convert BSR (Block Compressed Sparse Row) matrix A to CSR (Compressed Sparse Row) matrix B
 *
 * Input Arguments:
 *   I  n_brow          - number of block rows in A
 *   I  n_bcol          - number of block columns in A
 *   I  R               - row blocksize
 *   I  C               - column blocksize
 *   I  Ap[n_brow+1]    - block row pointer for A
 *   I  Aj[nnz(A)]      - block column indices for A
 *   T  Ax[nnz(A)]      - nonzero blocks for A
 *
 * Output Arguments:
 *   I  Bp[n_brow*R + 1]- row pointer for B
 *   I  Bj[nnz(B)]      - column indices for B
 *   T  Bx[nnz(B)]      - nonzero values for B
 *
 * Note:
 *   - Complexity: Linear, O(nnz(A) + max(n_row, n_col))
 *   - Output arrays Bp, Bj, Bx must be preallocated
 *   - Input column indices are not assumed to be sorted or unique
 *   - Output preserves unsorted block column orders, duplicates, and explicit zeros
 */
template <class I, class T>
void bsr_tocsr(const I n_brow, const I n_bcol, const I R, const I C,
               const I Ap[], const I Aj[], const T Ax[],
                     I Bp[],       I Bj[],       T Bx[])
{
    // Calculate number of elements per block
    const I RC = R * C;
    // Compute total number of nonzeros in B
    const I nnz = Ap[n_brow] * RC;
    // Set the last element in Bp to nnz
    Bp[n_brow * R] = nnz;
    // Loop through each block row of A to convert to CSR format
    // Remaining code to complete the conversion from BSR to CSR format...
}
    // 遍历每个块的行索引 brow，范围是从 0 到 n_brow
    for(I brow = 0; brow < n_brow; brow++){
        // 计算当前块行的大小，即该行的列数
        const I brow_size = Ap[brow + 1] - Ap[brow];
        // 计算当前行在 CSR 格式中的大小，即该行所有块列的总列数
        const I row_size = C * brow_size;
        // 遍历当前块行内部的行索引 r，范围是从 0 到 R
        for(I r = 0; r < R; r++){
            // 计算当前行在 CSR 格式中的行号
            const I row = R * brow + r;
            // 设置 Bp 中当前行的起始位置，用于存储该行的数据索引
            Bp[row] = RC * Ap[brow] + r * row_size;
            // 遍历当前块行内的块列索引 bjj，范围是从 0 到 brow_size
            for (I bjj = 0; bjj < brow_size; bjj++)
            {
                // 计算当前块列的索引在 Ap 中的位置
                const I b_ind = Ap[brow] + bjj;
                // 获取当前块列的列号 bcol
                const I bcol = Aj[b_ind];
                // 遍历当前块列内的列索引 c，范围是从 0 到 C
                for (I c = 0; c < C; c++)
                {
                    // 计算当前 BSR 格式中数据在 Ax 中的索引，Ax 是按照 C 顺序存储的
                    const I b_data_ind = RC * b_ind + C * r + c;
                    // 计算当前 CSR 格式中列号 col
                    const I col = C * bcol + c;
                    // 计算当前 CSR 格式中数据在 Bj 和 Bx 中的索引
                    const I data_ind = Bp[row] + bjj * C + c;
                    // 将列号 col 和数据 Ax[b_data_ind] 分配给 Bj 和 Bx
                    Bj[data_ind] = col;
                    Bx[data_ind] = Ax[b_data_ind];
                }
            }
        }
    }
/*
 * Compute Y += A*X for BSR matrix A and dense block vectors X,Y
 *
 *
 * Input Arguments:
 *   I  n_brow              - number of row blocks in A
 *   I  n_bcol              - number of column blocks in A
 *   I  n_vecs              - number of column vectors in X and Y
 *   I  R                   - rows per block
 *   I  C                   - columns per block
 *   I  Ap[n_brow+1]        - row pointer
 *   I  Aj[nblks(A)]        - column indices
 *   T  Ax[nnz(A)]          - nonzeros
 *   T  Xx[C*n_bcol,n_vecs] - input vector
 *
 * Output Arguments:
 *   T  Yx[R*n_brow,n_vecs] - output vector
 *
 */
template <class I, class T>
void bsr_matvecs(const I n_brow,
                 const I n_bcol,
                 const I n_vecs,
                 const I R,
                 const I C,
                 const I Ap[],
                 const I Aj[],
                 const T Ax[],
                 const T Xx[],
                       T Yx[])
{
    assert(R > 0 && C > 0);

    if( R == 1 && C == 1 ){
        // 如果块大小为 1x1，使用CSR格式计算
        csr_matvecs(n_brow, n_bcol, n_vecs, Ap, Aj, Ax, Xx, Yx);
        return;
    }

    const npy_intp A_bs = (npy_intp)R*C;      // 计算 Ax 块的大小
    const npy_intp Y_bs = (npy_intp)n_vecs*R; // 计算 Yx 块的大小
    const npy_intp X_bs = (npy_intp)C*n_vecs; // 计算 Xx 块的大小

    // 对每个行块进行循环
    for(I i = 0; i < n_brow; i++){
        // 指向当前 Y 块的指针
        T * y = Yx + Y_bs * i;
        // 对当前行块的每个列索引进行循环
        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            const I j = Aj[jj]; // 当前列索引
            const T * A = Ax + A_bs * jj; // 当前块 A 的指针
            const T * x = Xx + X_bs * j;  // 当前块 X 的指针
            gemm(R, n_vecs, C, A, x, y); // 矩阵乘法运算：y += A*x
        }
    }
}
```