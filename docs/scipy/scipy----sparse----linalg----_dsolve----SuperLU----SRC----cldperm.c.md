# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\cldperm.c`

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

#include "slu_cdefs.h"

// 声明外部函数 mc64id_，该函数返回整型值，接受一个整型指针参数
extern int_t mc64id_(int_t*);

// 声明外部函数 mc64ad_，该函数返回整型值，接受多个参数包括整型指针和双精度浮点数指针
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
 *   CLDPERM finds a row permutation so that the matrix has large
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
 * nzval  (input) singlecomplex*, of size nnz
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
cldperm(int job, int n, int_t nnz, int_t colptr[], int_t adjncy[],
        singlecomplex nzval[], int *perm, float u[], float v[])
{
    // 声明变量
    int_t i, num;
    int_t icntl[10], info[10];
    int_t liw, ldw, *iw;
    double *dw;
    
    // 将单精度复数数组转换为双精度数组
    double *nzval_d = (double *) SUPERLU_MALLOC(nnz * sizeof(double));

#if ( DEBUGlevel>=1 )
    // 检查内存分配
    CHECK_MALLOC("Enter cldperm()");
#endif
#endif
    liw = 5*n;  // 计算数组 iw 的长度，基于输入的 n
    if ( job == 3 ) liw = 10*n + nnz;  // 根据 job 的值重新计算 iw 的长度
    if ( !(iw = intMalloc(liw)) ) ABORT("Malloc fails for iw[]");  // 分配长度为 liw 的整型数组 iw

    ldw = 3*n + nnz;  // 计算数组 dw 的长度，基于输入的 n 和 nnz
    if ( !(dw = (double*) SUPERLU_MALLOC(ldw * sizeof(double))) )  // 分配长度为 ldw 的双精度数组 dw
          ABORT("Malloc fails for dw[]");

    /* Increment one to get 1-based indexing. */
    for (i = 0; i <= n; ++i) ++colptr[i];  // 将 colptr 数组中的每个元素加一，实现从0到n的索引变为从1到n+1

    for (i = 0; i < nnz; ++i) ++adjncy[i];  // 将 adjncy 数组中的每个元素加一，实现从0到nnz-1的索引变为从1到nnz

#if ( DEBUGlevel>=2 )
    printf("LDPERM(): n %d, nnz %d\n", n, nnz);  // 打印调试信息，输出 n 和 nnz 的值
    slu_PrintInt10("colptr", n+1, colptr);  // 打印调试信息，输出 colptr 数组的前 n+1 个元素
    slu_PrintInt10("adjncy", nnz, adjncy);  // 打印调试信息，输出 adjncy 数组的前 nnz 个元素
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
    mc64id_(icntl);  // 调用 MC64ID 函数进行初始化

#if 0
    /* Suppress error and warning messages. */
    icntl[0] = -1;
    icntl[1] = -1;
#endif

    int_t ljob = job, ln = n;  // 声明并初始化本地变量 ljob 和 ln

    for (i = 0; i < nnz; ++i) nzval_d[i] = c_abs1(&nzval[i]);  // 计算复数数组 nzval 的绝对值并存入 nzval_d 数组

    mc64ad_(&ljob, &ln, &nnz, colptr, adjncy, nzval_d, &num, perm,
        &liw, iw, &ldw, dw, icntl, info);  // 调用 MC64AD 函数进行稀疏矩阵重排序和最大权匹配

#if ( DEBUGlevel>=2 )
    slu_PrintInt10("perm", n, perm);  // 打印调试信息，输出 perm 数组的前 n 个元素
    printf(".. After MC64AD info %lld\tsize of matching %d\n", (long long)info[0], num);  // 打印调试信息，输出 MC64AD 运行后的信息和匹配大小
#endif

    if ( info[0] == 1 ) { /* Structurally singular */
        printf(".. The last %d permutations:\n", (int)(n-num));  // 如果结构上奇异，则打印最后 n-num 个排列
        slu_PrintInt10("perm", n-num, &perm[num]);  // 打印调试信息，输出 perm 数组的后 n-num 个元素
    }

    /* Restore to 0-based indexing. */
    for (i = 0; i <= n; ++i) --colptr[i];  // 将 colptr 数组中的每个元素减一，恢复到从0开始的索引
    for (i = 0; i < nnz; ++i) --adjncy[i];  // 将 adjncy 数组中的每个元素减一，恢复到从0开始的索引
    for (i = 0; i < n; ++i) --perm[i];  // 将 perm 数组中的每个元素减一，恢复到从0开始的索引

    if ( job == 5 )
        for (i = 0; i < n; ++i) {
        u[i] = dw[i];  // 如果 job 等于 5，则将 dw 数组的前 n 个元素赋值给 u 数组
        v[i] = dw[n+i];  // 如果 job 等于 5，则将 dw 数组的第 n 到 2n-1 个元素赋值给 v 数组
    }

    SUPERLU_FREE(iw);  // 释放 iw 数组的内存
    SUPERLU_FREE(dw);  // 释放 dw 数组的内存
    SUPERLU_FREE(nzval_d);  // 释放 nzval_d 数组的内存

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC("Exit cldperm()");  // 检查内存分配情况，输出退出信息
#endif

    return info[0];  // 返回 info 数组的第一个元素作为函数返回值
}
```