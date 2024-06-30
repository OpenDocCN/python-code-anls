# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dldperm.c`

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

#include "slu_ddefs.h"

// 声明外部函数 mc64id_ 和 mc64ad_
extern int_t mc64id_(int_t*);
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
 *   DLDPERM finds a row permutation so that the matrix has large
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
 * nzval  (input) double*, of size nnz
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
dldperm(int job, int n, int_t nnz, int_t colptr[], int_t adjncy[],
    double nzval[], int *perm, double u[], double v[])
{
    int_t i, num;
    int_t icntl[10], info[10];
    int_t liw, ldw, *iw;
    double *dw;

#if ( DEBUGlevel>=1 )
    // 检查内存分配是否正常
    CHECK_MALLOC("Enter dldperm()");
#endif
    // 设置整型工作区的大小
    liw = 5*n;
    // 若 JOB 等于 3，则需要更大的整型工作区
    if ( job == 3 ) liw = 10*n + nnz;
    # 如果内存分配失败，则终止程序并输出错误信息
    if ( !(iw = intMalloc(liw)) ) ABORT("Malloc fails for iw[]");
    
    # 计算数组 dw 的长度 ldw，用于存储稀疏矩阵的非零元素和一些额外的空间
    ldw = 3*n + nnz;
    
    # 分配 double 类型数组 dw 的内存空间，用于存储稀疏矩阵的非零元素
    if ( !(dw = (double*) SUPERLU_MALLOC(ldw * sizeof(double))) )
          ABORT("Malloc fails for dw[]");
        
    /* Increment one to get 1-based indexing. */
    # 将 colptr 数组中的每个元素加一，以实现从0-based到1-based的索引转换
    for (i = 0; i <= n; ++i) ++colptr[i];
    
    # 将 adjncy 数组中的每个元素加一，以实现从0-based到1-based的索引转换
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

mc64ad_(&ljob, &ln, &nnz, colptr, adjncy, nzval, &num, perm,
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

#if ( DEBUGlevel>=1 )
CHECK_MALLOC("Exit dldperm()");
#endif

return info[0];
}


注释：


#if ( DEBUGlevel>=2 )
    // 如果调试级别大于等于2，打印调试信息
    printf("LDPERM(): n %d, nnz %d\n", n, nnz);
    slu_PrintInt10("colptr", n+1, colptr);
    slu_PrintInt10("adjncy", nnz, adjncy);
#endif
    
/* 
 * NOTE:
 * =====
 *
 * MC64AD 假设列置换向量定义如下：
 * perm(i) = j 意味着置换后矩阵 A 的第 i 列在原始矩阵 A 的第 j 列上。
 *
 * 由于对称置换保留对角元素。因此，通过以下关系：
 *     P'(A*P')P = P'A
 * 我们可以对 A 的行应用 inverse(perm) 以获得较大的对角元素。
 * 但是，由于 MC64AD 中定义的 'perm' 恰好是 SuperLU 中置换向量的逆序，
 * 因此它已经对我们的目的是一个逆置换。因此，我们将直接使用它。
 *
 */
mc64id_(icntl);

#if 0
/* 抑制错误和警告消息。 */
icntl[0] = -1;
icntl[1] = -1;
#endif

int_t ljob = job, ln = n;

mc64ad_(&ljob, &ln, &nnz, colptr, adjncy, nzval, &num, perm,
    &liw, iw, &ldw, dw, icntl, info);

#if ( DEBUGlevel>=2 )
// 如果调试级别大于等于2，打印置换后的 perm 向量和 MC64AD 的信息
slu_PrintInt10("perm", n, perm);
printf(".. After MC64AD info %lld\tsize of matching %d\n", (long long)info[0], num);
#endif

if ( info[0] == 1 ) { /* 结构上奇异 */
    printf(".. 最后 %d 个置换:\n", (int)(n-num));
    slu_PrintInt10("perm", n-num, &perm[num]);
}

/* 恢复到基于0的索引。 */
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

#if ( DEBUGlevel>=1 )
// 如果调试级别大于等于1，检查内存分配情况
CHECK_MALLOC("Exit dldperm()");
#endif

return info[0];
}
```