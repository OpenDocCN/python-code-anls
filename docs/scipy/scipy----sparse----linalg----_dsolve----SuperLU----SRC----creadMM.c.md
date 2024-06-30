# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\creadMM.c`

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
#include "slu_cdefs.h"

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
 * \brief Reads a matrix stored in Matrix Market format from a file pointer.
 *
 * This function parses the Matrix Market header and data to extract matrix dimensions,
 * nonzero elements, and their locations in compressed sparse row (CSR) format.
 *
 * \param fp File pointer to the Matrix Market formatted file.
 * \param m Pointer to store the number of rows in the matrix.
 * \param n Pointer to store the number of columns in the matrix.
 * \param nonz Pointer to store the number of nonzero elements.
 * \param nzval Pointer to the array of nonzero values.
 * \param rowind Pointer to the array of row indices for nonzero elements.
 * \param colptr Pointer to the array of column pointers for the start of each column in CSR.
 */
void
creadMM(FILE *fp, int *m, int *n, int_t *nonz,
        singlecomplex **nzval, int_t **rowind, int_t **colptr)
{
    int_t    j, k, jsize, nnz, nz, new_nonz;
    singlecomplex *a, *val;
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
     fgets(line,512,fp); // 从文件流中读取一行，存储在line数组中，最多512字符
     for (p=line; *p!='\0'; *p=tolower(*p),p++); // 将line中的字符转换为小写

     if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, arith, sym) != 5) {
       printf("Invalid header (first line does not contain 5 tokens)\n");
       exit(-1);
     }
 
     if(strcmp(banner,"%%matrixmarket")) { // 检查文件头的第一个标记是否为"%%%%MatrixMarket"
       printf("Invalid header (first token is not \"%%%%MatrixMarket\")\n");
       exit(-1);
     }

     if(strcmp(mtx,"matrix")) { // 检查文件头的第二个标记是否为"matrix"
       printf("Not a matrix; this driver cannot handle that.\n");
       exit(-1);
     }

     if(strcmp(crd,"coordinate")) { // 检查文件头的第三个标记是否为"coordinate"
       printf("Not in coordinate format; this driver cannot handle that.\n");
       exit(-1);
     }

     if(strcmp(arith,"real")) { // 检查文件头的第四个标记是否为"real"
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

     if(strcmp(sym,"general")) { // 检查文件头的第五个标记是否为"general"
       printf("Symmetric matrix: will be expanded\n");
       expand=1;
     } else expand=0;

     /* 2/ Skip comments */
     while(banner[0]=='%') { // 跳过以'%'开头的注释行
       fgets(line,512,fp); // 继续读取下一行
       sscanf(line,"%s",banner); // 从新读取的行中获取第一个字符串标记
     }

     /* 3/ Read n and nnz */
#ifdef _LONGINT
    sscanf(line, "%d%d%lld",m, n, nonz); // 根据长整型情况读取m, n, nonz
#else
    sscanf(line, "%d%d%d",m, n, nonz); // 普通情况读取m, n, nonz
#endif

    if(*m!=*n) { // 检查是否为矩阵（行数与列数相同）
      printf("Rectangular matrix!. Abort\n");
      exit(-1);
   }
}
    // 根据 expand 变量确定新的非零元素个数
    if(expand)
      new_nonz = 2 * *nonz - *n;
    else
      new_nonz = *nonz;

    // 将 n 的值赋给 m
    *m = *n;
    // 打印 m、n 和 nonz 的值
    printf("m %lld, n %lld, nonz %lld\n", (long long) *m, (long long) *n, (long long) *nonz);
    // 调用 callocateA 函数分配存储空间
    callocateA(*n, new_nonz, nzval, rowind, colptr); /* Allocate storage */
    // 获取 nzval、rowind 和 colptr 指向的数组
    a    = *nzval;
    asub = *rowind;
    xa   = *colptr;

    // 分配存储空间给 val、row 和 col 数组
    if ( !(val = (singlecomplex *) SUPERLU_MALLOC(new_nonz * sizeof(double))) )
        ABORT("Malloc fails for val[]");
    if ( !(row = int32Malloc(new_nonz)) )
        ABORT("Malloc fails for row[]");
    if ( !(col = int32Malloc(new_nonz)) )
        ABORT("Malloc fails for col[]");

    // 初始化 xa 数组为 0
    for (j = 0; j < *n; ++j) xa[j] = 0;

    /* 4/ Read triplets of values */
    // 读取值的三元组
    for (nnz = 0, nz = 0; nnz < *nonz; ++nnz) {
        // 从文件中读取行、列和值
        fscanf(fp, "%d%d%f%f\n", &row[nz], &col[nz], &val[nz].r, &val[nz].i);

        // 如果是第一个非零元素
        if ( nnz == 0 ) {
            // 判断行和列索引是否从 0 开始
            if ( row[0] == 0 || col[0] == 0 ) {
                zero_base = 1;
                printf("triplet file: row/col indices are zero-based.\n");
            } else {
                printf("triplet file: row/col indices are one-based.\n");
            }
        }

        // 如果不是从 0 开始索引
        if ( !zero_base ) {
            // 将索引改为从 0 开始
            --row[nz];
            --col[nz];
        }

        // 检查索引是否超出界限
        if (row[nz] < 0 || row[nz] >= *m || col[nz] < 0 || col[nz] >= *n
            /*|| val[nz] == 0.*/) {
            fprintf(stderr, "nz %d, (%d, %d) = {%e,%e} out of bound, removed\n",
                      (int) nz, row[nz], col[nz], val[nz].r, val[nz].i);
            exit(-1);
        } else {
            // 更新 xa[col[nz]]
            ++xa[col[nz]];
            // 如果 expand 为真，且行和列索引不同，则将对称元素也加入
            if(expand) {
                if ( row[nz] != col[nz] ) { /* Excluding diagonal */
                  ++nz;
                  row[nz] = col[nz-1];
                  col[nz] = row[nz-1];
                  val[nz] = val[nz-1];
                  ++xa[col[nz]];
                }
            }    
            ++nz;
        }
    }

    // 更新 nonz 为 nz
    *nonz = nz;
    // 如果 expand 为真，打印修改后的非零元素个数
    if(expand) {
      printf("new_nonz after symmetric expansion:\t%lld\n", (long long)*nonz);
    }
    

    /* Initialize the array of column pointers */
    // 初始化列指针数组
    k = 0;
    jsize = xa[0];
    xa[0] = 0;
    for (j = 1; j < *n; ++j) {
        k += jsize;
        jsize = xa[j];
        xa[j] = k;
    }
    
    /* Copy the triplets into the column oriented storage */
    // 将三元组复制到列导向的存储结构
    for (nz = 0; nz < *nonz; ++nz) {
        j = col[nz];
        k = xa[j];
        asub[k] = row[nz];
        a[k] = val[nz];
        ++xa[j];
    }

    /* Reset the column pointers to the beginning of each column */
    // 将列指针重置为每列的开头
    for (j = *n; j > 0; --j)
        xa[j] = xa[j-1];
    xa[0] = 0;

    // 释放动态分配的内存
    SUPERLU_FREE(val);
    SUPERLU_FREE(row);
    SUPERLU_FREE(col);
#ifdef CHK_INPUT
    // 如果定义了 CHK_INPUT 宏，则执行以下代码块
    int i;
    // 遍历数组 xa，输出每列的索引和值
    for (i = 0; i < *n; i++) {
        printf("Col %d, xa %d\n", i, xa[i]);
        // 遍历当前列中的每个非零元素
        for (k = xa[i]; k < xa[i+1]; k++)
            printf("%d\t%16.10f\n", asub[k], a[k]);
    }
#endif
}


static void creadrhs(int m, singlecomplex *b)
{
    // 打开文件 "b.dat" 以只读方式
    FILE *fp = fopen("b.dat", "r");
    int i;

    // 如果文件打开失败，则输出错误信息并退出程序
    if ( !fp ) {
        fprintf(stderr, "creadrhs: file does not exist\n");
        exit(-1);
    }
    // 从文件中读取 m 行，每行包含两个浮点数，分别存储到 b[i].r 和 b[i].i 中
    for (i = 0; i < m; ++i)
        fscanf(fp, "%f%f\n", &b[i].r, &b[i].i);

    // 关闭文件流
    fclose(fp);
}
```