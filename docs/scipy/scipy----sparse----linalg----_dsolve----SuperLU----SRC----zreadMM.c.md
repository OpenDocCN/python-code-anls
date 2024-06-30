# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zreadMM.c`

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
#include "slu_zdefs.h"

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

void
zreadMM(FILE *fp, int *m, int *n, int_t *nonz,
        doublecomplex **nzval, int_t **rowind, int_t **colptr)
{
    int_t    j, k, jsize, nnz, nz, new_nonz;
    doublecomplex *a, *val;
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
     // 读取文件的第一行作为文件头
     fgets(line,512,fp);
     // 将读取到的行内容转换为小写
     for (p=line; *p!='\0'; *p=tolower(*p),p++);

     // 检查文件头的格式是否正确
     if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, arith, sym) != 5) {
       printf("Invalid header (first line does not contain 5 tokens)\n");
       exit(-1);
     }
 
     // 检查文件头是否以 "%%MatrixMarket" 开头
     if(strcmp(banner,"%%matrixmarket")) {
       printf("Invalid header (first token is not \"%%%%MatrixMarket\")\n");
       exit(-1);
     }

     // 检查文件类型是否为矩阵
     if(strcmp(mtx,"matrix")) {
       printf("Not a matrix; this driver cannot handle that.\n");
       exit(-1);
     }

     // 检查数据格式是否为坐标格式
     if(strcmp(crd,"coordinate")) {
       printf("Not in coordinate format; this driver cannot handle that.\n");
       exit(-1);
     }

     // 检查数据类型是否为实数
     if(strcmp(arith,"real")) {
       // 如果是复数，则输出错误信息并退出
       if(!strcmp(arith,"complex")) {
         printf("Complex matrix; use zreadMM instead!\n");
         exit(-1);
       }
       // 如果是模式（pattern），则输出错误信息并退出
       else if(!strcmp(arith, "pattern")) {
         printf("Pattern matrix; values are needed!\n");
         exit(-1);
       }
       // 其他未知类型，输出错误信息并退出
       else {
         printf("Unknown arithmetic\n");
         exit(-1);
       }
     }

     // 检查矩阵类型是否为一般（general）
     if(strcmp(sym,"general")) {
       printf("Symmetric matrix: will be expanded\n");
       expand=1;
     } else expand=0;

     /* 2/ Skip comments */
     // 跳过注释行（以 '%' 开头的行）
     while(banner[0]=='%') {
       fgets(line,512,fp);
       sscanf(line,"%s",banner);
     }

     /* 3/ Read n and nnz */
#ifdef _LONGINT
    sscanf(line, "%d%d%lld",m, n, nonz);
#else
    sscanf(line, "%d%d%d",m, n, nonz);
#endif

    // 检查是否为方阵
    if(*m!=*n) {
      printf("Rectangular matrix!. Abort\n");
      exit(-1);
   }

   // 这里是函数的其余部分，需要继续实现读取矩阵数据的功能
}


这段代码是用于读取存储在Harwell-Boeing格式中的矩阵数据。注释详细解释了代码中各个步骤的功能和作用，包括文件头的解析和错误检查。
    if(expand)
      new_nonz = 2 * *nonz - *n;
    else
      new_nonz = *nonz;


    // 根据条件判断是否扩展矩阵，计算新的非零元素个数
    if(expand)
      new_nonz = 2 * *nonz - *n;
    else
      new_nonz = *nonz;



    *m = *n;
    printf("m %lld, n %lld, nonz %lld\n", (long long) *m, (long long) *n, (long long) *nonz);
    zallocateA(*n, new_nonz, nzval, rowind, colptr); /* Allocate storage */
    a    = *nzval;
    asub = *rowind;
    xa   = *colptr;


    // 将 n 赋值给 m，打印当前 m、n 和 nonz 的值，为数据分配存储空间
    *m = *n;
    printf("m %lld, n %lld, nonz %lld\n", (long long) *m, (long long) *n, (long long) *nonz);
    zallocateA(*n, new_nonz, nzval, rowind, colptr); /* Allocate storage */
    a    = *nzval;   // 设置指向 nzval 的指针 a
    asub = *rowind;  // 设置指向 rowind 的指针 asub
    xa   = *colptr;  // 设置指向 colptr 的指针 xa



    if ( !(val = (doublecomplex *) SUPERLU_MALLOC(new_nonz * sizeof(double))) )
        ABORT("Malloc fails for val[]");
    if ( !(row = int32Malloc(new_nonz)) )
        ABORT("Malloc fails for row[]");
    if ( !(col = int32Malloc(new_nonz)) )
        ABORT("Malloc fails for col[]");


    // 分配内存给 val、row 和 col 数组，若分配失败则中止程序
    if ( !(val = (doublecomplex *) SUPERLU_MALLOC(new_nonz * sizeof(double))) )
        ABORT("Malloc fails for val[]");
    if ( !(row = int32Malloc(new_nonz)) )
        ABORT("Malloc fails for row[]");
    if ( !(col = int32Malloc(new_nonz)) )
        ABORT("Malloc fails for col[]");



    for (j = 0; j < *n; ++j) xa[j] = 0;


    // 初始化 xa 数组，将每个元素设为 0
    for (j = 0; j < *n; ++j) xa[j] = 0;



    /* 4/ Read triplets of values */
    for (nnz = 0, nz = 0; nnz < *nonz; ++nnz) {
    fscanf(fp, "%d%d%lf%lf\n", &row[nz], &col[nz], &val[nz].r, &val[nz].i);

    if ( nnz == 0 ) { /* first nonzero */
        if ( row[0] == 0 || col[0] == 0 ) {
        zero_base = 1;
        printf("triplet file: row/col indices are zero-based.\n");
        } else {
        printf("triplet file: row/col indices are one-based.\n");
            }
    }

    if ( !zero_base ) {
        /* Change to 0-based indexing. */
        --row[nz];
        --col[nz];
    }

    if (row[nz] < 0 || row[nz] >= *m || col[nz] < 0 || col[nz] >= *n
        /*|| val[nz] == 0.*/) {
        fprintf(stderr, "nz %d, (%d, %d) = {%e,%e} out of bound, removed\n",
                  (int) nz, row[nz], col[nz], val[nz].r, val[nz].i);
        exit(-1);
    } else {
        ++xa[col[nz]];
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


    /* 4/ 读取值的三元组 */
    for (nnz = 0, nz = 0; nnz < *nonz; ++nnz) {
        fscanf(fp, "%d%d%lf%lf\n", &row[nz], &col[nz], &val[nz].r, &val[nz].i);

        if ( nnz == 0 ) { /* 第一个非零元素 */
            if ( row[0] == 0 || col[0] == 0 ) {
                zero_base = 1;
                printf("triplet file: row/col indices are zero-based.\n");
            } else {
                printf("triplet file: row/col indices are one-based.\n");
            }
        }

        if ( !zero_base ) {
            /* 转换为以0为基础的索引 */
            --row[nz];
            --col[nz];
        }

        if (row[nz] < 0 || row[nz] >= *m || col[nz] < 0 || col[nz] >= *n
            /*|| val[nz] == 0.*/) {
            fprintf(stderr, "nz %d, (%d, %d) = {%e,%e} out of bound, removed\n",
                      (int) nz, row[nz], col[nz], val[nz].r, val[nz].i);
            exit(-1);
        } else {
            ++xa[col[nz]];
            if(expand) {
                if ( row[nz] != col[nz] ) { /* 排除对角线元素 */
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



    *nonz = nz;
    if(expand) {
      printf("new_nonz after symmetric expansion:\t%lld\n", (long long)*nonz);
    }


    // 更新非零元素个数，如果进行了扩展则打印新的非零元素个数
    *nonz = nz;
    if(expand) {
      printf("new_nonz after symmetric expansion:\t%lld\n", (long long)*nonz);
    }



    /* Initialize the array of column pointers */
    k = 0;
    jsize = xa[0];
    xa[0] = 0;
    for (j = 1; j < *n; ++j) {
    k += jsize;
    jsize = xa[j];
    xa[j] = k;
    }


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
    for (nz = 0; nz < *nonz; ++nz) {
    j = col[nz];
    k = xa[j];
    asub[k] = row[nz];
    a[k] = val[nz];
    ++xa[j];
    }


    // 将三元组复制到基于列的存储结构中
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
    xa[0] = 0;


    // 将列指针重置为每列的开头
    for (j = *n; j > 0; --j)
        xa[j] = xa[j-1];
    xa[0] = 0;



    SUPERLU_FREE(val);
    SUPERLU_FREE(row);
    SUPERLU_FREE(col);


    // 释放动态分配的内存空间
    SUPERLU_FREE(val);
    SUPERLU_FREE(row);
    SUPERLU_FREE(col);
#ifdef CHK_INPUT
    // 如果定义了 CHK_INPUT 宏，则执行以下代码块
    int i;
    // 遍历从 0 到 *n-1 的整数 i
    for (i = 0; i < *n; i++) {
        // 打印当前列号 i 和 xa[i] 的值
        printf("Col %d, xa %d\n", i, xa[i]);
        // 遍历从 xa[i] 到 xa[i+1]-1 的整数 k
        for (k = xa[i]; k < xa[i+1]; k++)
            // 打印 asub[k] 和 a[k] 的值，格式为整数和浮点数
            printf("%d\t%16.10f\n", asub[k], a[k]);
    }
#endif

}

static void zreadrhs(int m, doublecomplex *b)
{
    // 打开文件 "b.dat" 以只读方式
    FILE *fp = fopen("b.dat", "r");

    int i;
    // 如果文件打开失败
    if ( !fp ) {
        // 输出错误信息到标准错误流
        fprintf(stderr, "zreadrhs: file does not exist\n");
        // 退出程序并返回 -1
        exit(-1);
    }
    // 读取 m 行数据，每行包含两个浮点数，分别赋值给 b[i].r 和 b[i].i
    for (i = 0; i < m; ++i)
        fscanf(fp, "%lf%lf\n", &b[i].r, &b[i].i);

    // 关闭文件流
    fclose(fp);
}
```