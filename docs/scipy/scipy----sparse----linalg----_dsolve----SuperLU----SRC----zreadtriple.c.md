# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zreadtriple.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file zreadtriple.c
 * \brief Read a matrix stored in triplet (coordinate) format
 *
 * <pre>
 * -- SuperLU routine (version 4.0) --
 * Lawrence Berkeley National Laboratory.
 * June 30, 2009
 * </pre>
 */

#include "slu_zdefs.h"

void
zreadtriple(int *m, int *n, int_t *nonz,
        doublecomplex **nzval, int_t **rowind, int_t **colptr)
{
    /*
     * Output parameters
     * =================
     *   (a,asub,xa): asub[*] contains the row subscripts of nonzeros
     *    in columns of matrix A; a[*] the numerical values;
     *    row i of A is given by a[k],k=xa[i],...,xa[i+1]-1.
     *
     */
    int    j, k, jsize, nnz, nz;
    doublecomplex *a, *val;
    int_t  *asub, *xa;
    int    *row, *col;
    int    zero_base = 0;

    /*  Matrix format:
     *    First line:  #rows, #cols, #non-zero
     *    Triplet in the rest of lines:
     *                 row, col, value
     */

#ifdef _LONGINT
    scanf("%d%lld", n, nonz);
#else
    scanf("%d%d", n, nonz);
#endif    
    *m = *n;
    printf("m %d, n %d, nonz %ld\n", *m, *n, (long) *nonz);
    zallocateA(*n, *nonz, nzval, rowind, colptr); /* Allocate storage */
    a    = *nzval;
    asub = *rowind;
    xa   = *colptr;

    val = (doublecomplex *) SUPERLU_MALLOC(*nonz * sizeof(doublecomplex));
    row = int32Malloc(*nonz);
    col = int32Malloc(*nonz);

    for (j = 0; j < *n; ++j) xa[j] = 0;

    /* Read into the triplet array from a file */
    for (nnz = 0, nz = 0; nnz < *nonz; ++nnz) {
        // 从文件中读取三元组数据：行、列、值
        scanf("%d%d%lf%lf\n", &row[nz], &col[nz], &val[nz].r, &val[nz].i);

        if ( nnz == 0 ) { /* 第一个非零元素 */
            // 检查行和列索引是否从0开始
            if ( row[0] == 0 || col[0] == 0 ) {
                zero_base = 1;
                printf("triplet file: row/col indices are zero-based.\n");
            } else
                printf("triplet file: row/col indices are one-based.\n");
        }

        if ( !zero_base ) { 
            // 转换为以0为基础的索引
            --row[nz];
            --col[nz];
        }

        // 检查行列索引是否在有效范围内，如果不在则报错并退出
        if (row[nz] < 0 || row[nz] >= *m || col[nz] < 0 || col[nz] >= *n
            /*|| val[nz] == 0.*/) {
            fprintf(stderr, "nz %d, (%d, %d) = (%e,%e) out of bound, removed\n",
                nz, row[nz], col[nz], val[nz].r, val[nz].i);
            exit(-1);
        } else {
            // 更新列指针数组
            ++xa[col[nz]];
            ++nz;
        }
    }

    *nonz = nz;

    /* 初始化列指针数组 */
    k = 0;
    jsize = xa[0];
    xa[0] = 0;
    for (j = 1; j < *n; ++j) {
        k += jsize;
        jsize = xa[j];
        xa[j] = k;
    }
    
    /* 将三元组复制到列优先存储结构中 */
    for (nz = 0; nz < *nonz; ++nz) {
        j = col[nz];
        k = xa[j];
        asub[k] = row[nz];
        a[k] = val[nz];
        ++xa[j];
    }
}
    /* 将列指针重新设置到每列的开头 */
    for (j = *n; j > 0; --j)
        xa[j] = xa[j-1];
    xa[0] = 0;

    /* 释放动态分配的内存：val 数组 */
    SUPERLU_FREE(val);
    /* 释放动态分配的内存：row 数组 */
    SUPERLU_FREE(row);
    /* 释放动态分配的内存：col 数组 */
    SUPERLU_FREE(col);
#ifdef CHK_INPUT
    {
    int i;
    // 遍历列索引数组 xa，打印每列的索引值和相应的内容
    for (i = 0; i < *n; i++) {
        printf("Col %d, xa %d\n", i, xa[i]);
        // 遍历从 xa[i] 到 xa[i+1] 之间的行索引，打印每行的行索引和对应的数值
        for (k = xa[i]; k < xa[i+1]; k++)
            printf("%d\t%16.10f\n", asub[k], a[k]);
    }
    }
#endif

}

void zreadrhs(int m, doublecomplex *b)
{
    // 打开文件 "b.dat" 以只读方式
    FILE *fp = fopen("b.dat", "r");
    int i;

    // 如果文件打开失败，则输出错误信息并退出程序
    if ( !fp ) {
        fprintf(stderr, "dreadrhs: file does not exist\n");
        exit(-1);
    }
    // 读取文件中的 m 行数据，每行包含两个 double 型数值，分别存入 b[i].r 和 b[i].i
    for (i = 0; i < m; ++i)
        fscanf(fp, "%lf%lf\n", &b[i].r, &b[i].i);

    // 关闭文件流
    fclose(fp);
}
```