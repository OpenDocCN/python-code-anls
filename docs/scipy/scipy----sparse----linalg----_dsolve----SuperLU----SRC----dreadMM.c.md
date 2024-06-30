# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dreadMM.c`

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
 * \brief Read a matrix stored in Harwell-Boeing format.
 * Contributed by Francois-Henry Rouet.
 *
 */
#include <ctype.h>
#include "slu_ddefs.h"

#undef EXPAND_SYM

/*! brief
 *
 * <pre>
 * Output parameters
 * =================
 *   (nzval, rowind, colptr): (*rowind)[*] contains the row subscripts of
 *      nonzeros in columns of matrix A; (*nzval)[*] the numerical values;
 *    column i of A is given by (*nzval)[k], k = (*rowind)[i],...,
 *      (*rowind)[i+1]-1.
 * </pre>
 */

/*! 
 * Read a matrix stored in MatrixMarket format from a file pointer.
 *
 * \param fp File pointer to the MatrixMarket format file.
 * \param m Pointer to store the number of rows in the matrix.
 * \param n Pointer to store the number of columns in the matrix.
 * \param nonz Pointer to store the number of non-zero entries in the matrix.
 * \param nzval Pointer to store the values of non-zero entries (output).
 * \param rowind Pointer to store the row indices of non-zero entries (output).
 * \param colptr Pointer to store the starting indices of columns in nzval and rowind (output).
 */
void
dreadMM(FILE *fp, int *m, int *n, int_t *nonz,
        double **nzval, int_t **rowind, int_t **colptr)
{
    int_t    j, k, jsize, nnz, nz, new_nonz;
    double *a, *val;
    int_t    *asub, *xa;
    int      *row, *col;
    int    zero_base = 0;
    char *p, line[512], banner[64], mtx[64], crd[64], arith[64], sym[64];
    int expand;

    /*     File format:
     *    %%MatrixMarket matrix coordinate real general/symmetric/...
     *    % ...
     *    % (optional comments)
     *    % ...
     *    #rows    #non-zero
     *    Triplet in the rest of lines: row    col    value
     */

     /* 1/ read header */ 
     fgets(line,512,fp); // 从文件指针中读取一行，最多512个字符，存入line数组
     for (p=line; *p!='\0'; *p=tolower(*p),p++); // 将读取的行内容全部转换为小写字母

     if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, arith, sym) != 5) {
       printf("Invalid header (first line does not contain 5 tokens)\n");
       exit(-1);
     }
 
     if(strcmp(banner,"%%matrixmarket")) {
       printf("Invalid header (first token is not \"%%%%MatrixMarket\")\n");
       exit(-1);
     }

     if(strcmp(mtx,"matrix")) {
       printf("Not a matrix; this driver cannot handle that.\n");
       exit(-1);
     }

     if(strcmp(crd,"coordinate")) {
       printf("Not in coordinate format; this driver cannot handle that.\n");
       exit(-1);
     }

     if(strcmp(arith,"real")) {
       if(!strcmp(arith,"complex")) {
         printf("Complex matrix; use zreadMM instead!\n");
         exit(-1);
       }
       else if(!strcmp(arith, "pattern")) {
         printf("Pattern matrix; values are needed!\n");
         exit(-1);
       }
       else {
         printf("Unknown arithmetic\n");
         exit(-1);
       }
     }

     if(strcmp(sym,"general")) {
       printf("Symmetric matrix: will be expanded\n");
       expand=1;
     } else expand=0;

     /* 2/ Skip comments */
     while(banner[0]=='%') {
       fgets(line,512,fp); // 继续读取文件中的下一行
       sscanf(line,"%s",banner); // 从读取的行中提取第一个字符串
     }

     /* 3/ Read n and nnz */
#ifdef _LONGINT
    sscanf(line, "%d%d%lld",m, n, nonz); // 从读取的行中解析出矩阵的行数m、列数n和非零元素数nonz
#else
    sscanf(line, "%d%d%d",m, n, nonz);
#endif
    printf("m %lld, n %lld, nonz %lld\n", (long long) *m, (long long) *n, (long long) *nonz);
}
    /* 检查矩阵是否为矩形矩阵 */
    if(*m!=*n) {
      printf("Rectangular matrix!. Abort\n");
      exit(-1);
   }

    /* 根据是否扩展标志，计算新的非零元素个数 */
    if(expand)
      new_nonz = 2 * *nonz - *n;
    else
      new_nonz = *nonz;


    dallocateA(*n, new_nonz, nzval, rowind, colptr); /* 分配存储空间 */
    a    = *nzval;
    asub = *rowind;
    xa   = *colptr;

    /* 分配存储空间并检查是否成功 */
    if ( !(val = (double *) SUPERLU_MALLOC(new_nonz * sizeof(double))) )
        ABORT("Malloc fails for val[]");
    if ( !(row = int32Malloc(new_nonz)) )
        ABORT("Malloc fails for row[]");
    if ( !(col = int32Malloc(new_nonz)) )
        ABORT("Malloc fails for col[]");

    /* 初始化列指针数组 */
    for (j = 0; j < *n; ++j) xa[j] = 0;

    /* 读取三元组的值 */
    /* 4/ 读取三元组的值 */
    for (nnz = 0, nz = 0; nnz < *nonz; ++nnz) {
    fscanf(fp, "%d%d%lf\n", &row[nz], &col[nz], &val[nz]);

    /* 如果是第一个非零元素 */
    if ( nnz == 0 ) { /* first nonzero */
        /* 检查行列索引是否基于零 */
        if ( row[0] == 0 || col[0] == 0 ) {
        zero_base = 1;
        printf("triplet file: row/col indices are zero-based.\n");
        } else {
        printf("triplet file: row/col indices are one-based.\n");
            }
    }

    /* 如果不是基于零索引 */
    if ( !zero_base ) {
        /* 将索引转换为基于零的 */
        --row[nz];
        --col[nz];
    }

    /* 检查索引是否越界 */
    if (row[nz] < 0 || row[nz] >= *m || col[nz] < 0 || col[nz] >= *n
        /*|| val[nz] == 0.*/) {
        /* 输出错误信息并退出 */
        fprintf(stderr, "nz %d, (%d, %d) = %e out of bound, removed\n", 
                    (int) nz, row[nz], col[nz], val[nz]);
        exit(-1);
    } else {
        /* 更新列偏移数组 */
        ++xa[col[nz]];
            if(expand) {
            /* 如果需要扩展 */
            if ( row[nz] != col[nz] ) { /* 排除对角线元素 */
              ++nz;
              row[nz] = col[nz-1];
              col[nz] = row[nz-1];
              val[nz] = val[nz-1];
              ++xa[col[nz]];
            }
            }    
        /* 增加非零元素计数 */
        ++nz;
    }
    }

    /* 更新非零元素的数量 */
    *nonz = nz;
    /* 如果需要扩展，打印新的非零元素数目 */
    if(expand) {
      printf("new_nonz after symmetric expansion:\t%lld\n", (long long)*nonz);
    }
    

    /* 初始化列指针数组 */
    k = 0;
    jsize = xa[0];
    xa[0] = 0;
    for (j = 1; j < *n; ++j) {
    k += jsize;
    jsize = xa[j];
    xa[j] = k;
    }
    
    /* 将三元组复制到列导向的存储结构中 */
    for (nz = 0; nz < *nonz; ++nz) {
    j = col[nz];
    k = xa[j];
    asub[k] = row[nz];
    a[k] = val[nz];
    ++xa[j];
    }

    /* 重置列指针，使其指向每列的起始位置 */
    for (j = *n; j > 0; --j)
    xa[j] = xa[j-1];
    xa[0] = 0;

    /* 释放分配的内存 */
    SUPERLU_FREE(val);
    SUPERLU_FREE(row);
    SUPERLU_FREE(col);
#ifdef CHK_INPUT
    // 如果定义了 CHK_INPUT 宏，则执行以下代码块
    int i;
    // 遍历从指针 n 指向的值减一的次数，输出每列的索引和对应 xa 数组中的值
    for (i = 0; i < *n; i++) {
        printf("Col %d, xa %d\n", i, xa[i]);
        // 遍历从 xa[i] 到 xa[i+1]-1 的索引，输出对应的 asub 和 a 数组中的值
        for (k = xa[i]; k < xa[i+1]; k++)
            printf("%d\t%16.10f\n", asub[k], a[k]);
    }
#endif

}

static void dreadrhs(int m, double *b)
{
    // 打开名为 "b.dat" 的文件，以只读方式
    FILE *fp = fopen("b.dat", "r");
    int i;

    if ( !fp ) {
        // 如果文件指针为 NULL，输出错误信息到标准错误流，并退出程序
        fprintf(stderr, "dreadrhs: file does not exist\n");
        exit(-1);
    }
    // 从文件中读取 m 个双精度浮点数到数组 b 中
    for (i = 0; i < m; ++i)
        fscanf(fp, "%lf\n", &b[i]);

    // 关闭文件流
    fclose(fp);
}
```