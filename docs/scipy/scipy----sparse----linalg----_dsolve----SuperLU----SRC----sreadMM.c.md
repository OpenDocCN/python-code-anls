# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\sreadMM.c`

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
#include "slu_sdefs.h"

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
sreadMM(FILE *fp, int *m, int *n, int_t *nonz,
        float **nzval, int_t **rowind, int_t **colptr)
{
    int_t    j, k, jsize, nnz, nz, new_nonz;
    float *a, *val;
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
    // 读取文件的第一行作为头部信息
    fgets(line,512,fp);
    // 将头部信息转换为小写以便比较
    for (p=line; *p!='\0'; *p=tolower(*p),p++);

    // 检查头部信息的格式是否正确，应为 "%%MatrixMarket matrix coordinate real general/symmetric/..."
    if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, arith, sym) != 5) {
        printf("Invalid header (first line does not contain 5 tokens)\n");
        exit(-1);
    }
 
    // 检查头部信息第一个字段是否为 "%%MatrixMarket"
    if(strcmp(banner,"%%matrixmarket")) {
        printf("Invalid header (first token is not \"%%%%MatrixMarket\")\n");
        exit(-1);
    }

    // 检查头部信息第二个字段是否为 "matrix"
    if(strcmp(mtx,"matrix")) {
        printf("Not a matrix; this driver cannot handle that.\n");
        exit(-1);
    }

    // 检查头部信息第三个字段是否为 "coordinate"
    if(strcmp(crd,"coordinate")) {
        printf("Not in coordinate format; this driver cannot handle that.\n");
        exit(-1);
    }

    // 检查头部信息第四个字段是否为 "real"，如果是 "complex" 则报错并推荐使用 zreadMM
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

    // 检查头部信息第五个字段是否为 "general"，如果不是则表示是对称矩阵，需要扩展处理
    if(strcmp(sym,"general")) {
        printf("Symmetric matrix: will be expanded\n");
        expand=1;
    } else expand=0;

    /* 2/ Skip comments */
    // 跳过以 '%' 开头的注释行
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

    // 检查矩阵是否为方阵，即行数和列数是否相等
    if(*m!=*n) {
        printf("Rectangular matrix!. Abort\n");
        exit(-1);
    }
}


这段代码是用于读取存储在Harwell-Boeing格式中的矩阵数据。注释详细解释了每个步骤的作用和代码的预期输入输出。
    /* 根据 expand 的值计算 new_nonz */
    if(expand)
      new_nonz = 2 * *nonz - *n;
    else
      new_nonz = *nonz;

    /* 将 n 的值赋给 m */
    *m = *n;
    /* 打印 m, n, nonz 的当前值 */
    printf("m %lld, n %lld, nonz %lld\n", (long long) *m, (long long) *n, (long long) *nonz);
    /* 分配 nzval、rowind、colptr 对应的内存空间 */
    sallocateA(*n, new_nonz, nzval, rowind, colptr); /* Allocate storage */
    /* 获得指向 nzval、rowind、colptr 的指针 */
    a    = *nzval;
    asub = *rowind;
    xa   = *colptr;

    /* 分配 val、row、col 数组的内存空间 */
    if ( !(val = (float *) SUPERLU_MALLOC(new_nonz * sizeof(double))) )
        ABORT("Malloc fails for val[]");
    if ( !(row = int32Malloc(new_nonz)) )
        ABORT("Malloc fails for row[]");
    if ( !(col = int32Malloc(new_nonz)) )
        ABORT("Malloc fails for col[]");

    /* 初始化 xa 数组为 0 */
    for (j = 0; j < *n; ++j) xa[j] = 0;

    /* 读取三元组的值 */
    /* 4/ Read triplets of values */
    for (nnz = 0, nz = 0; nnz < *nonz; ++nnz) {
    /* 从文件中读取行、列、值 */
    fscanf(fp, "%d%d%f\n", &row[nz], &col[nz], &val[nz]);

    /* 若是第一个非零元素 */
    if ( nnz == 0 ) { /* first nonzero */
        /* 检查行列索引是否基于零 */
        if ( row[0] == 0 || col[0] == 0 ) {
        zero_base = 1;
        printf("triplet file: row/col indices are zero-based.\n");
        } else {
        printf("triplet file: row/col indices are one-based.\n");
            }
    }

    /* 若不是基于零索引，则转换为基于零索引 */
    if ( !zero_base ) {
        /* Change to 0-based indexing. */
        --row[nz];
        --col[nz];
    }

    /* 检查索引是否超出范围，若超出则退出 */
    if (row[nz] < 0 || row[nz] >= *m || col[nz] < 0 || col[nz] >= *n
        /*|| val[nz] == 0.*/) {
        fprintf(stderr, "nz %d, (%d, %d) = %e out of bound, removed\n", 
                    (int) nz, row[nz], col[nz], val[nz]);
        exit(-1);
    } else {
        /* 增加列 j 的非零元素计数 */
        ++xa[col[nz]];
            /* 若 expand 为真 */
            if(expand) {
            /* 排除对角线上的元素 */
            if ( row[nz] != col[nz] ) { /* Excluding diagonal */
              ++nz;
              /* 将对称位置的元素加入 */
              row[nz] = col[nz-1];
              col[nz] = row[nz-1];
              val[nz] = val[nz-1];
              ++xa[col[nz]];
            }
            }    
        ++nz;
    }
    }

    /* 更新 nonz 的值为 nz */
    *nonz = nz;
    /* 若 expand 为真，则打印扩展后的 new_nonz */
    if(expand) {
      printf("new_nonz after symmetric expansion:\t%lld\n", (long long)*nonz);
    }
    

    /* 初始化列指针数组 xa */
    k = 0;
    jsize = xa[0];
    xa[0] = 0;
    for (j = 1; j < *n; ++j) {
    k += jsize;
    jsize = xa[j];
    xa[j] = k;
    }
    
    /* 将三元组复制到列优先存储中 */
    for (nz = 0; nz < *nonz; ++nz) {
    j = col[nz];
    k = xa[j];
    asub[k] = row[nz];
    a[k] = val[nz];
    ++xa[j];
    }

    /* 重置列指针数组为每列的起始位置 */
    for (j = *n; j > 0; --j)
    xa[j] = xa[j-1];
    xa[0] = 0;

    /* 释放动态分配的内存空间 */
    SUPERLU_FREE(val);
    SUPERLU_FREE(row);
    SUPERLU_FREE(col);
#ifdef CHK_INPUT
    #ifdef CHK_INPUT 宏定义的条件编译开始
    int i;
    # 定义整型变量 i
    for (i = 0; i < *n; i++) {
    # 循环，从 i=0 开始，直到 i < *n 为止，每次递增 i
    printf("Col %d, xa %d\n", i, xa[i]);
    # 打印输出当前列数 i 和数组 xa 中第 i 个元素的值
    for (k = xa[i]; k < xa[i+1]; k++)
        # 嵌套循环，从数组 xa 中第 i 个元素开始，直到下一个元素，每次递增 k
        printf("%d\t%16.10f\n", asub[k], a[k]);
        # 打印输出数组 asub 中第 k 个元素和数组 a 中第 k 个元素的值
    }
    # 结束内层循环和条件编译
#endif
# 结束条件编译

}


static void sreadrhs(int m, float *b)
{
    # 定义静态函数 sreadrhs，接受参数 m（整数）和 b（浮点数指针）
    FILE *fp = fopen("b.dat", "r");
    # 打开文件 "b.dat" 用于读取，获取文件指针 fp

    int i;
    # 定义整型变量 i

    if ( !fp ) {
        # 如果文件指针 fp 为空（即文件打开失败）
        fprintf(stderr, "sreadrhs: file does not exist\n");
        # 输出错误信息到标准错误流
        exit(-1);
        # 退出程序，返回错误码 -1
    }

    for (i = 0; i < m; ++i)
      # 循环，从 i=0 开始，直到 i < m 为止，每次递增 i
      fscanf(fp, "%f\n", &b[i]);
      # 从文件 fp 中读取一个浮点数，存储到 b 数组的第 i 个位置

    fclose(fp);
    # 关闭文件 fp
}
# 结束函数定义
```