# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zldperm.c`

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

#include "slu_zdefs.h"

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
 *   ZLDPERM finds a row permutation so that the matrix has large
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
 * nzval  (input) doublecomplex*, of size nnz
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
zldperm(int job, int n, int_t nnz, int_t colptr[], int_t adjncy[],
    doublecomplex nzval[], int *perm, double u[], double v[])
{
    int_t i, num;
    int_t icntl[10], info[10];
    int_t liw, ldw, *iw;
    double *dw;
    double *nzval_d = (double *) SUPERLU_MALLOC(nnz * sizeof(double));

    // 检查进入函数时的内存分配情况，DEBUGlevel 为 1 或更高时启用
    CHECK_MALLOC("Enter zldperm()");

    // 根据不同的 job 类型，设定不同的算法控制参数
    icntl[0] = 0; // 设定算法控制参数的默认值
    icntl[1] = 8; // HSL MC21A/AD 算法的默认值
    if (job == 2 || job == 3) {
        icntl[1] = 10; // 用于 JOB 2 和 JOB 3 的算法参数
    } else if (job == 4) {
        icntl[1] = 11; // 用于 JOB 4 的算法参数
    } else if (job == 5) {
        icntl[1] = 12; // 用于 JOB 5 的算法参数
    }

    // 初始化工作空间的长度
    liw = 10 * n;
    ldw = 10 * n;
    iw = intMalloc(liw);
    dw = doubleMalloc(ldw);

    // 根据不同的 job 类型，执行不同的矩阵置换操作
    switch (job) {
        case 1:
            // 调用 HSL MC21A/AD 算法进行对角线元素尽可能多的矩阵置换
            mc21ad(n, colptr, adjncy, perm, icntl, info, iw, dw);
            break;
        case 2:
        case 3:
        case 4:
            // 调用其他算法进行矩阵置换以优化对角线元素
            other_algorithm(job, n, colptr, adjncy, perm, icntl, info, iw, dw);
            break;
        case 5:
            // 调用算法进行矩阵置换，同时计算对角线元素的对数尺度因子
            algorithm_with_scaling(job, n, colptr, adjncy, perm, icntl, info, iw, dw, u, v);
            break;
        default:
            // 不支持的 job 类型
            fprintf(stderr, "Unsupported JOB type: %d\n", job);
            exit(-1);
    }

    // 释放工作空间内存
    SUPERLU_FREE(nzval_d);
    SUPERLU_FREE(iw);
    SUPERLU_FREE(dw);

    // 返回函数执行状态
    return 0;
}
#endif
    // 计算数组 iw 的长度
    liw = 5*n;
    // 根据 job 的值调整 iw 的长度
    if ( job == 3 ) liw = 10*n + nnz;
    // 分配内存给数组 iw，并检查分配是否成功
    if ( !(iw = intMalloc(liw)) ) ABORT("Malloc fails for iw[]");
    // 计算数组 dw 的长度
    ldw = 3*n + nnz;
    // 分配内存给数组 dw，并检查分配是否成功
    if ( !(dw = (double*) SUPERLU_MALLOC(ldw * sizeof(double))) )
          ABORT("Malloc fails for dw[]");
        
    /* Increment one to get 1-based indexing. */
    // 将 colptr 数组中的每个元素加一，以实现从0-based到1-based的索引
    for (i = 0; i <= n; ++i) ++colptr[i];
    // 将 adjncy 数组中的每个元素加一，以实现从0-based到1-based的索引
    for (i = 0; i < nnz; ++i) ++adjncy[i];
#if ( DEBUGlevel>=2 )
    // 调试级别2以上打印调试信息
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
    // 调用 MC64AD 库函数，传递参数 icntl 控制选项
    mc64id_(icntl);
#if 0
    /* Suppress error and warning messages. */
    // 抑制错误和警告信息
    icntl[0] = -1;
    icntl[1] = -1;
#endif

    int_t ljob = job, ln = n;
    
    // 对 nzval_d 数组中的每个元素取其绝对值
    for (i = 0; i < nnz; ++i) nzval_d[i] = z_abs1(&nzval[i]);
    // 调用 MC64AD 库函数进行列排序和匹配
    mc64ad_(&ljob, &ln, &nnz, colptr, adjncy, nzval_d, &num, perm,
        &liw, iw, &ldw, dw, icntl, info);

#if ( DEBUGlevel>=2 )
    // 调试级别2以上打印调试信息
    slu_PrintInt10("perm", n, perm);
    printf(".. After MC64AD info %lld\tsize of matching %d\n", (long long)info[0], num);
#endif
    // 如果 info[0] 为1，表明结构上奇异
    if ( info[0] == 1 ) { /* Structurally singular */
        printf(".. The last %d permutations:\n", (int)(n-num));
    slu_PrintInt10("perm", n-num, &perm[num]);
    }

    // 将 colptr 数组中的每个元素减一，恢复到0-based索引
    for (i = 0; i <= n; ++i) --colptr[i];
    // 将 adjncy 数组中的每个元素减一，恢复到0-based索引
    for (i = 0; i < nnz; ++i) --adjncy[i];
    // 将 perm 数组中的每个元素减一，恢复到0-based索引
    for (i = 0; i < n; ++i) --perm[i];

    // 如果 job 等于5，将 dw 数组的前 n 个元素赋值给 u 数组，后 n 个元素赋值给 v 数组
    if ( job == 5 )
        for (i = 0; i < n; ++i) {
        u[i] = dw[i];
        v[i] = dw[n+i];
    }

    // 释放内存
    SUPERLU_FREE(iw);
    SUPERLU_FREE(dw);
    SUPERLU_FREE(nzval_d);

#if ( DEBUGlevel>=1 )
    // 调试级别1以上检查内存分配情况
    CHECK_MALLOC("Exit zldperm()");
#endif

    // 返回信息数组的第一个元素
    return info[0];
}
```