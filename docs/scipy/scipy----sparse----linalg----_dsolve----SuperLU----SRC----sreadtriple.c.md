# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\sreadtriple.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file sreadtriple.c
 * \brief Read a matrix stored in triplet (coordinate) format
 *
 * <pre>
 * -- SuperLU routine (version 4.0) --
 * Lawrence Berkeley National Laboratory.
 * June 30, 2009
 * </pre>
 */

#include "slu_sdefs.h"

/*! 
 * \brief Read a matrix stored in triplet (coordinate) format from stdin
 *
 * This function reads a sparse matrix stored in triplet format from standard input.
 * It allocates memory for the matrix and stores it in compressed column format.
 *
 * \param m     Pointer to the number of rows (output)
 * \param n     Pointer to the number of columns (output)
 * \param nonz  Pointer to the number of non-zero elements (input/output)
 * \param nzval Pointer to store the non-zero values of the matrix (output)
 * \param rowind Pointer to store the row indices of non-zero values (output)
 * \param colptr Pointer to store the column pointers (output)
 */
void
sreadtriple(int *m, int *n, int_t *nonz,
        float **nzval, int_t **rowind, int_t **colptr)
{
    int    j, k, jsize, nnz, nz;
    float *a, *val;
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
    sallocateA(*n, *nonz, nzval, rowind, colptr); /* Allocate storage */
    a    = *nzval;
    asub = *rowind;
    xa   = *colptr;

    val = (float *) SUPERLU_MALLOC(*nonz * sizeof(float));
    row = int32Malloc(*nonz);
    col = int32Malloc(*nonz);

    for (j = 0; j < *n; ++j) xa[j] = 0;

    /* Read into the triplet array from a file */
    for (nnz = 0, nz = 0; nnz < *nonz; ++nnz) {
        scanf("%d%d%f\n", &row[nz], &col[nz], &val[nz]);

        if ( nnz == 0 ) { /* first nonzero */
            if ( row[0] == 0 || col[0] == 0 ) {
                zero_base = 1;
                printf("triplet file: row/col indices are zero-based.\n");
            } else
                printf("triplet file: row/col indices are one-based.\n");
        }

        if ( !zero_base ) { 
            /* Change to 0-based indexing. */
            --row[nz];
            --col[nz];
        }

        if (row[nz] < 0 || row[nz] >= *m || col[nz] < 0 || col[nz] >= *n) {
            fprintf(stderr, "nz %d, (%d, %d) = %e out of bound, removed\n",
                nz, row[nz], col[nz], val[nz]);
            exit(-1);
        } else {
            ++xa[col[nz]];
            ++nz;
        }
    }

    *nonz = nz;

    /* Initialize the array of column pointers */
    k = 0;
    jsize = xa[0];
    xa[0] = 0;
    for (j = 1; j < *n; ++j) {
        k += jsize;
        jsize = xa[j];
        xa[j] = k;
    }
    
    /* Copy the triplets into the column oriented storage */
    for (nz = 0; nz < *nonz; ++nz) {
        j = col[nz];
        k = xa[j];
        asub[k] = row[nz];
        a[k] = val[nz];
        ++xa[j];
    }

    /* Reset the column pointers to the beginning of each column */
    for (j = *n; j > 0; --j)
        xa[j] = xa[j-1];
}
    # 将数组 xa 的第一个元素设为 0
    xa[0] = 0;
    
    # 释放由 SUPERLU 分配的 val 数组内存
    SUPERLU_FREE(val);
    # 释放由 SUPERLU 分配的 row 数组内存
    SUPERLU_FREE(row);
    # 释放由 SUPERLU 分配的 col 数组内存
    SUPERLU_FREE(col);
#ifdef CHK_INPUT
    {
    int i;
    // 遍历列索引数组，输出每列的起始位置和相关的行索引与值
    for (i = 0; i < *n; i++) {
        printf("Col %d, xa %d\n", i, xa[i]);
        // 遍历从 xa[i] 到 xa[i+1] 之间的行索引和对应的值
        for (k = xa[i]; k < xa[i+1]; k++)
        printf("%d\t%16.10f\n", asub[k], a[k]);
    }
    }
#endif

}

void sreadrhs(int m, float *b)
{
    // 打开文件 "b.dat" 以只读模式
    FILE *fp = fopen("b.dat", "r");
    int i;

    // 如果文件打开失败，输出错误信息并退出程序
    if ( !fp ) {
        fprintf(stderr, "dreadrhs: file does not exist\n");
        exit(-1);
    }
    // 从文件中依次读取 m 个浮点数到数组 b 中
    for (i = 0; i < m; ++i)
      fscanf(fp, "%f\n", &b[i]);

    // 关闭文件流
    fclose(fp);
}
```