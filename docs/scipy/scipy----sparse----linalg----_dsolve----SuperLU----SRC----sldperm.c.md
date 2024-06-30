# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\sldperm.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file 
 * \brief Finds a row permutation so that the matrix has large entries on the diagonal
 *
 * <pre>
 * -- SuperLU routine (version 4.0) --
 * Lawrence Berkeley National Laboratory.
 * June 30, 2009
 * </pre>
 */

#include "slu_sdefs.h"

// 声明外部函数 mc64id_
extern int_t mc64id_(int_t*);

// 声明外部函数 mc64ad_
extern int_t mc64ad_(int_t *job, int_t *n, int_t *ne, int_t *ip,
                     int_t *irn, double *a, int_t *num, int *cperm,
                     int_t *liw, int_t *iw, int_t *ldw, double *dw,
                     int_t *icntl, int_t *info);
/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 *   SLDPERM finds a row permutation so that the matrix has large
 *   entries on the diagonal.
 *
 * Arguments
 * =========
 *
 * job    (input) int
 *        Control the action. Possible values for JOB are:
 *        = 1 : Compute a row permutation of the matrix so that the
 *              permuted matrix has as many entries on its diagonal as
 *              possible. The values on the diagonal are of arbitrary size.
 *              HSL subroutine MC21A/AD is used for this.
 *        = 2 : Compute a row permutation of the matrix so that the smallest 
 *              value on the diagonal of the permuted matrix is maximized.
 *        = 3 : Compute a row permutation of the matrix so that the smallest
 *              value on the diagonal of the permuted matrix is maximized.
 *              The algorithm differs from the one used for JOB = 2 and may
 *              have quite a different performance.
 *        = 4 : Compute a row permutation of the matrix so that the sum
 *              of the diagonal entries of the permuted matrix is maximized.
 *        = 5 : Compute a row permutation of the matrix so that the product
 *              of the diagonal entries of the permuted matrix is maximized
 *              and vectors to scale the matrix so that the nonzero diagonal 
 *              entries of the permuted matrix are one in absolute value and 
 *              all the off-diagonal entries are less than or equal to one in 
 *              absolute value.
 *        Restriction: 1 <= JOB <= 5.
 *
 * n      (input) int
 *        The order of the matrix.
 *
 * nnz    (input) int
 *        The number of nonzeros in the matrix.
 *
 * adjncy (input) int*, of size nnz
 *        The adjacency structure of the matrix, which contains the row
 *        indices of the nonzeros.
 *
 * colptr (input) int*, of size n+1
 *        The pointers to the beginning of each column in ADJNCY.
 *
 * nzval  (input) float*, of size nnz
 *        The nonzero values of the matrix. nzval[k] is the value of
 *        the entry corresponding to adjncy[k].
 *        It is not used if job = 1.
 *
 * perm   (output) int*, of size n
 *        The permutation vector. perm[i] = j means row i in the
 *        original matrix is in row j of the permuted matrix.
 *
 * u      (output) double*, of size n
 *        If job = 5, the natural logarithms of the row scaling factors. 
 *
 * v      (output) double*, of size n
 *        If job = 5, the natural logarithms of the column scaling factors. 
 *        The scaled matrix B has entries b_ij = a_ij * exp(u_i + v_j).
 * </pre>
 */

int
sldperm(int job, int n, int_t nnz, int_t colptr[], int_t adjncy[],
    float nzval[], int *perm, float u[], float v[])
{
    // 声明变量
    int_t i, num;
    int_t icntl[10], info[10];
    int_t liw, ldw, *iw;
    double *dw;
    
    // 将 nzval 的元素转换为 double 类型并分配内存
    double *nzval_d = (double *) SUPERLU_MALLOC(nnz * sizeof(double));

#if ( DEBUGlevel>=1 )
    // 检查是否成功分配内存
    CHECK_MALLOC("Enter sldperm()");
#endif
    # 计算数组 `iw` 的长度，以及根据特定条件进行调整
    liw = 5*n;
    if ( job == 3 ) liw = 10*n + nnz;
    # 分配整型数组 `iw` 的内存空间，若分配失败则终止程序
    if ( !(iw = intMalloc(liw)) ) ABORT("Malloc fails for iw[]");
    
    # 计算数组 `dw` 的长度
    ldw = 3*n + nnz;
    # 分配双精度数组 `dw` 的内存空间，若分配失败则终止程序
    if ( !(dw = (double*) SUPERLU_MALLOC(ldw * sizeof(double))) )
          ABORT("Malloc fails for dw[]");
        
    # 对列指针数组 `colptr` 中的每个元素进行递增，实现从0-based到1-based的索引转换
    /* Increment one to get 1-based indexing. */
    for (i = 0; i <= n; ++i) ++colptr[i];
    
    # 对邻接数组 `adjncy` 中的每个元素进行递增，实现从0-based到1-based的索引转换
    for (i = 0; i < nnz; ++i) ++adjncy[i];
#if ( DEBUGlevel>=2 )
    printf("LDPERM(): n %d, nnz %d\n", n, nnz);
    slu_PrintInt10("colptr", n+1, colptr);
    slu_PrintInt10("adjncy", nnz, adjncy);
#endif

/* 
 * NOTE:
 * =====
 *
 * MC64AD assumes that column permutation vector is defined as:
 * perm(i) = j means column i of permuted A is in column j of original A.
 *
 * Since a symmetric permutation preserves the diagonal entries. Then
 * by the following relation:
 *     P'(A*P')P = P'A
 * we can apply inverse(perm) to rows of A to get large diagonal entries.
 * But, since 'perm' defined in MC64AD happens to be the reverse of
 * SuperLU's definition of permutation vector, therefore, it is already
 * an inverse for our purpose. We will thus use it directly.
 *
 */
mc64id_(icntl);

#if 0
/* Suppress error and warning messages. */
icntl[0] = -1;
icntl[1] = -1;
#endif

int_t ljob = job, ln = n;

for (i = 0; i < nnz; ++i) nzval_d[i] = nzval[i];
mc64ad_(&ljob, &ln, &nnz, colptr, adjncy, nzval_d, &num, perm,
    &liw, iw, &ldw, dw, icntl, info);

#if ( DEBUGlevel>=2 )
slu_PrintInt10("perm", n, perm);
printf(".. After MC64AD info %lld\tsize of matching %d\n", (long long)info[0], num);
#endif

if ( info[0] == 1 ) { /* Structurally singular */
    printf(".. The last %d permutations:\n", (int)(n-num));
    slu_PrintInt10("perm", n-num, &perm[num]);
}

/* Restore to 0-based indexing. */
for (i = 0; i <= n; ++i) --colptr[i];
for (i = 0; i < nnz; ++i) --adjncy[i];
for (i = 0; i < n; ++i) --perm[i];

if ( job == 5 )
    for (i = 0; i < n; ++i) {
    u[i] = dw[i];
    v[i] = dw[n+i];
}

SUPERLU_FREE(iw);
SUPERLU_FREE(dw);
SUPERLU_FREE(nzval_d);

#if ( DEBUGlevel>=1 )
CHECK_MALLOC("Exit sldperm()");
#endif

return info[0];
}


注释：


#if ( DEBUGlevel>=2 )
    // 如果调试级别大于等于2，打印LDPERM函数的参数n和nnz
    printf("LDPERM(): n %d, nnz %d\n", n, nnz);
    // 打印数组colptr的内容，数组大小为n+1
    slu_PrintInt10("colptr", n+1, colptr);
    // 打印数组adjncy的内容，数组大小为nnz
    slu_PrintInt10("adjncy", nnz, adjncy);
#endif

/* 
 * NOTE:
 * =====
 *
 * MC64AD假定列置换向量定义如下：
 * perm(i) = j 意味着置换后A的第i列在原始A的第j列。
 *
 * 由于对称置换保留对角元素。因此，根据以下关系：
 *     P'(A*P')P = P'A
 * 我们可以将perm的逆应用于A的行以获得大的对角元素。
 * 但是，由于MC64AD中定义的'perm'恰好是SuperLU置换向量定义的反向，因此，对于我们的目的它已经是逆的。
 * 因此，我们将直接使用它。
 *
 */
mc64id_(icntl);

#if 0
/* 抑制错误和警告消息。 */
icntl[0] = -1;
icntl[1] = -1;
#endif

int_t ljob = job, ln = n;

// 复制数组nzval的内容到nzval_d
for (i = 0; i < nnz; ++i) nzval_d[i] = nzval[i];
// 调用MC64AD算法进行稀疏矩阵操作
mc64ad_(&ljob, &ln, &nnz, colptr, adjncy, nzval_d, &num, perm,
    &liw, iw, &ldw, dw, icntl, info);

#if ( DEBUGlevel>=2 )
// 如果调试级别大于等于2，打印置换向量perm的内容
slu_PrintInt10("perm", n, perm);
// 打印MC64AD运行后的信息和匹配的大小
printf(".. After MC64AD info %lld\tsize of matching %d\n", (long long)info[0], num);
#endif

// 如果info[0]等于1，表示结构上奇异
if ( info[0] == 1 ) { /* Structurally singular */
    // 打印最后n-num个置换的信息
    printf(".. The last %d permutations:\n", (int)(n-num));
    slu_PrintInt10("perm", n-num, &perm[num]);
}

/* 恢复为0-based索引。 */
// 将colptr数组的每个元素减1
for (i = 0; i <= n; ++i) --colptr[i];
// 将adjncy数组的每个元素减1
for (i = 0; i < nnz; ++i) --adjncy[i];
// 将perm数组的每个元素减1
for (i = 0; i < n; ++i) --perm[i];

// 如果job等于5
if ( job == 5 )
    // 将dw数组的前n个元素复制到u数组，后n个元素复制到v数组
    for (i = 0; i < n; ++i) {
    u[i] = dw[i];
    v[i] = dw[n+i];
}

// 释放动态分配的数组空间
SUPERLU_FREE(iw);
SUPERLU_FREE(dw);
SUPERLU_FREE(nzval_d);

#if ( DEBUGlevel>=1 )
// 检查内存分配是否正常
CHECK_MALLOC("Exit sldperm()");
#endif

// 返回info数组的第一个元素
return info[0];
}
```