# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\sreadrb.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file sreadrb.c
 * \brief Read a matrix stored in Rutherford-Boeing format
 *
 * <pre>
 * -- SuperLU routine (version 4.0) --
 * Lawrence Berkeley National Laboratory.
 * June 30, 2009
 * </pre>
 * <pre>
 *
 * Purpose
 * =======
 *
 * Read a FLOAT PRECISION matrix stored in Rutherford-Boeing format 
 * as described below.
 *
 * Line 1 (A72, A8)
 *      Col. 1 - 72   Title (TITLE)
 *      Col. 73 - 80  Matrix name / identifier (MTRXID)
 *
 * Line 2 (I14, 3(1X, I13))
 *      Col. 1 - 14   Total number of lines excluding header (TOTCRD)
 *      Col. 16 - 28  Number of lines for pointers (PTRCRD)
 *      Col. 30 - 42  Number of lines for row (or variable) indices (INDCRD)
 *      Col. 44 - 56  Number of lines for numerical values (VALCRD)
 *
 * Line 3 (A3, 11X, 4(1X, I13))
 *      Col. 1 - 3    Matrix type (see below) (MXTYPE)
 *      Col. 15 - 28  Compressed Column: Number of rows (NROW)
 *                    Elemental: Largest integer used to index variable (MVAR)
 *      Col. 30 - 42  Compressed Column: Number of columns (NCOL)
 *                    Elemental: Number of element matrices (NELT)
 *      Col. 44 - 56  Compressed Column: Number of entries (NNZERO)
 *                    Elemental: Number of variable indeces (NVARIX)
 *      Col. 58 - 70  Compressed Column: Unused, explicitly zero
 *                    Elemental: Number of elemental matrix entries (NELTVL)
 *
 * Line 4 (2A16, A20)
 *      Col. 1 - 16   Fortran format for pointers (PTRFMT)
 *      Col. 17 - 32  Fortran format for row (or variable) indices (INDFMT)
 *      Col. 33 - 52  Fortran format for numerical values of coefficient matrix
 *                    (VALFMT)
 *                    (blank in the case of matrix patterns)
 *
 * The three character type field on line 3 describes the matrix type.
 * The following table lists the permitted values for each of the three
 * characters. As an example of the type field, RSA denotes that the matrix
 * is real, symmetric, and assembled.
 *
 * First Character:
 *      R Real matrix
 *      C Complex matrix
 *      I integer matrix
 *      P Pattern only (no numerical values supplied)
 *      Q Pattern only (numerical values supplied in associated auxiliary value
 *        file)
 *
 * Second Character:
 *      S Symmetric
 *      U Unsymmetric
 *      H Hermitian
 *      Z Skew symmetric
 *      R Rectangular
 *
 * Third Character:
 *      A Compressed column form
 *      E Elemental form
 *
 * </pre>
 */

#include <stdio.h>
#include <stdlib.h>
#include "slu_sdefs.h"

/*! \brief Eat up the rest of the current line */
static int sDumpLine(FILE *fp)
{
    register int c;
    // 读取文件流中的字符，直到遇到换行符为止，用于跳过当前行剩余内容
    while ((c = fgetc(fp)) != '\n') ;
    # 返回整数值 0
    return 0;
}

static int sParseIntFormat(char *buf, int *num, int *size)
{
    char *tmp;

    tmp = buf;  // 将指针 tmp 指向 buf 的起始位置
    while (*tmp++ != '(') ;  // 找到 '(' 字符的位置，tmp 指向 '(' 后面的字符
    sscanf(tmp, "%d", num);  // 从 tmp 指向的位置读取一个整数到 num 中
    while (*tmp != 'I' && *tmp != 'i') ++tmp;  // 寻找字符串中第一个 'I' 或 'i' 字符的位置
    ++tmp;  // 跳过 'I' 或 'i' 字符
    sscanf(tmp, "%d", size);  // 从 tmp 指向的位置读取一个整数到 size 中
    return 0;  // 返回 0 表示函数执行成功
}

static int sParseFloatFormat(char *buf, int *num, int *size)
{
    char *tmp, *period;

    tmp = buf;  // 将指针 tmp 指向 buf 的起始位置
    while (*tmp++ != '(') ;  // 找到 '(' 字符的位置，tmp 指向 '(' 后面的字符
    *num = atoi(tmp);  // 将 tmp 指向的字符串转换为整数，并赋值给 num
    while (*tmp != 'E' && *tmp != 'e' && *tmp != 'D' && *tmp != 'd'
           && *tmp != 'F' && *tmp != 'f') {
        /* 在 nE/nD/nF 之前可能会找到 kP，例如 (1P6F13.6) 这种情况下，
           选中的 num 是指 P，需要跳过它。 */
        if (*tmp=='p' || *tmp=='P') {
           ++tmp;  // 跳过 'p' 或 'P' 字符
           *num = atoi(tmp);  // 将 tmp 指向的字符串转换为整数，并赋值给 num
        } else {
           ++tmp;  // 继续移动 tmp 指针
        }
    }
    ++tmp;  // 跳过 'E'、'e'、'D'、'd'、'F'、'f' 字符
    period = tmp;
    while (*period != '.' && *period != ')') ++period ;  // 找到 '.' 或 ')' 字符的位置
    *period = '\0';  // 在 period 处添加字符串结束符 '\0'
    *size = atoi(tmp);  // 将 tmp 指向的字符串转换为整数，并赋值给 size

    return 0;  // 返回 0 表示函数执行成功
}

static int ReadVector(FILE *fp, int n, int_t *where, int perline, int persize)
{
    int_t i, j, item;
    char tmp, buf[100];

    i = 0;  // 初始化计数器 i
    while (i < n) {  // 循环直到读取 n 个元素
        fgets(buf, 100, fp);    // 从文件 fp 中读取一行，存储到 buf 中
        for (j=0; j<perline && i<n; j++) {
            tmp = buf[(j+1)*persize];     // 保存指定位置的字符
            buf[(j+1)*persize] = 0;       // 在该位置添加字符串结束符
            item = atoi(&buf[j*persize]);  // 将指定位置的字符串转换为整数，并赋值给 item
            buf[(j+1)*persize] = tmp;     // 恢复指定位置的字符
            where[i++] = item - 1;  // 将 item 减 1 后存储到 where 数组中，同时增加 i
        }
    }

    return 0;  // 返回 0 表示函数执行成功
}

static int sReadValues(FILE *fp, int n, float *destination, int perline,
        int persize)
{
    register int i, j, k, s;
    char tmp, buf[100];

    i = 0;  // 初始化计数器 i
    while (i < n) {  // 循环直到读取 n 个元素
        fgets(buf, 100, fp);    // 从文件 fp 中读取一行，存储到 buf 中
        for (j=0; j<perline && i<n; j++) {
            tmp = buf[(j+1)*persize];     // 保存指定位置的字符
            buf[(j+1)*persize] = 0;       // 在该位置添加字符串结束符
            s = j*persize;
            for (k = 0; k < persize; ++k)  // 替换 'D' 格式为 'E'，C 语言中无 'D_' 格式
                if ( buf[s+k] == 'D' || buf[s+k] == 'd' ) buf[s+k] = 'E';
            destination[i++] = atof(&buf[s]);  // 将指定位置的字符串转换为浮点数，并赋值给 destination
            buf[(j+1)*persize] = tmp;     // 恢复指定位置的字符
        }
    }

    return 0;  // 返回 0 表示函数执行成功
}



/*! \brief
 *
 * <pre>
 * On input, nonz/nzval/rowind/colptr represents lower part of a symmetric
 * matrix. On exit, it represents the full matrix with lower and upper parts.
 * </pre>
 */
static void
FormFullA(int n, int_t *nonz, float **nzval, int_t **rowind, int_t **colptr)
{
    int_t i, j, k, col, new_nnz;
    int_t *t_rowind, *t_colptr, *al_rowind, *al_colptr, *a_rowind, *a_colptr;
    int_t *marker;
    float *t_val, *al_val, *a_val;

    al_rowind = *rowind;  // 将指针 al_rowind 指向 rowind 指向的内存
    al_colptr = *colptr;  // 将指针 al_colptr 指向 colptr 指向的内存
    al_val = *nzval;  // 将指针 al_val 指向 nzval 指向的内存

    if ( !(marker = intMalloc( n+1 ) ) )  // 分配 n+1 个整数大小的内存给 marker
    ABORT("SUPERLU_MALLOC fails for marker[]");  // 如果分配失败，终止程序执行
    # 分配内存并检查是否成功，如果失败则终止程序
    if ( !(t_colptr = intMalloc( n+1 ) ) )
        ABORT("SUPERLU_MALLOC t_colptr[]");
    
    # 分配内存并检查是否成功，如果失败则终止程序
    if ( !(t_rowind = intMalloc( *nonz ) ) )
        ABORT("SUPERLU_MALLOC fails for t_rowind[]");
    
    # 分配内存并检查是否成功，如果失败则终止程序
    if ( !(t_val = (float*) SUPERLU_MALLOC( *nonz * sizeof(float)) ) )
        ABORT("SUPERLU_MALLOC fails for t_val[]");

    # 获取矩阵 T 中每一列的元素个数，并设置列指针
    for (i = 0; i < n; ++i) marker[i] = 0;
    
    for (j = 0; j < n; ++j) {
        for (i = al_colptr[j]; i < al_colptr[j+1]; ++i)
            ++marker[al_rowind[i]];
    }
    
    t_colptr[0] = 0;
    
    for (i = 0; i < n; ++i) {
        t_colptr[i+1] = t_colptr[i] + marker[i];
        marker[i] = t_colptr[i];
    }

    # 将矩阵 A 转置为矩阵 T
    for (j = 0; j < n; ++j) {
        for (i = al_colptr[j]; i < al_colptr[j+1]; ++i) {
            col = al_rowind[i];
            t_rowind[marker[col]] = j;
            t_val[marker[col]] = al_val[i];
            ++marker[col];
        }
    }

    # 计算新的非零元素个数
    new_nnz = *nonz * 2 - n;

    # 分配内存并检查是否成功，如果失败则终止程序
    if ( !(a_colptr = intMalloc( n+1 ) ) )
        ABORT("SUPERLU_MALLOC a_colptr[]");
    
    # 分配内存并检查是否成功，如果失败则终止程序
    if ( !(a_rowind = intMalloc( new_nnz) ) )
        ABORT("SUPERLU_MALLOC fails for a_rowind[]");
    
    # 分配内存并检查是否成功，如果失败则终止程序
    if ( !(a_val = (float*) SUPERLU_MALLOC( new_nnz * sizeof(float)) ) )
        ABORT("SUPERLU_MALLOC fails for a_val[]");

    a_colptr[0] = 0;
    k = 0;

    for (j = 0; j < n; ++j) {
        for (i = t_colptr[j]; i < t_colptr[j+1]; ++i) {
            if ( t_rowind[i] != j ) { /* 不是对角线上的元素 */
                a_rowind[k] = t_rowind[i];
                a_val[k] = t_val[i];
                ++k;
            }
        }
        a_colptr[j+1] = k; // 更新列指针
    }
    /* Debug 模式下，检查 a_val[k] 的绝对值是否小于极小值，打印出来 */
#ifdef DEBUG
      if ( fabs(a_val[k]) < 4.047e-300 )
          printf("%5d: %e\n", (int)k, a_val[k]);
#endif
      // 增加 k 的计数，准备处理下一个元素
      ++k;
    }
      }

      // 处理 al_colptr[j] 到 al_colptr[j+1] 之间的每个元素
      for (i = al_colptr[j]; i < al_colptr[j+1]; ++i) {
    // 将 al_rowind[i] 和 al_val[i] 分别赋值给 a_rowind[k] 和 a_val[k]
    a_rowind[k] = al_rowind[i];
    a_val[k] = al_val[i];
      // 如果 a_val[k] 的绝对值小于极小值，打印出来
      if ( fabs(a_val[k]) < 4.047e-300 )
          printf("%5d: %e\n", (int)k, a_val[k]);
    // 增加 k 的计数，准备处理下一个元素
    ++k;
      }
      
      // 更新 a_colptr[j+1] 的值为 k，表示该列的元素个数
      a_colptr[j+1] = k;
    }

    // 打印新的非零元素个数 new_nnz
    printf("FormFullA: new_nnz = %lld\n", (long long) new_nnz);

    // 释放动态分配的数组 al_val, al_rowind, al_colptr, marker, t_val, t_rowind, t_colptr 的内存
    SUPERLU_FREE(al_val);
    SUPERLU_FREE(al_rowind);
    SUPERLU_FREE(al_colptr);
    SUPERLU_FREE(marker);
    SUPERLU_FREE(t_val);
    SUPERLU_FREE(t_rowind);
    SUPERLU_FREE(t_colptr);

    // 将计算得到的结果数组赋值给指针参数
    *nzval = a_val;
    *rowind = a_rowind;
    *colptr = a_colptr;
    *nonz = new_nnz;
}

// 从标准输入流中读取矩阵的信息
void
sreadrb(int *nrow, int *ncol, int_t *nonz,
        float **nzval, int_t **rowind, int_t **colptr)
{

    register int i, numer_lines = 0;
    int tmp, colnum, colsize, rownum, rowsize, valnum, valsize;
    char buf[100], type[4];
    int sym;
    FILE *fp;

    // 将输入流指定为标准输入
    fp = stdin;

    /* Line 1 */
    // 读取并输出第一行内容
    fgets(buf, 100, fp);
    fputs(buf, stdout);

    /* Line 2 */
    // 读取并解析第二行内容，获取 numer_lines 的值
    for (i=0; i<4; i++) {
        fscanf(fp, "%14c", buf); buf[14] = 0;
        sscanf(buf, "%d", &tmp);
        if (i == 3) numer_lines = tmp;
    }
    // 跳过多余的换行符
    sDumpLine(fp);

    /* Line 3 */
    // 读取矩阵类型到 type 中
    fscanf(fp, "%3c", type);
    fscanf(fp, "%11c", buf); /* pad */
    type[3] = 0;
#ifdef DEBUG
    // 在 Debug 模式下打印矩阵类型
    printf("Matrix type %s\n", type);
#endif

    // 依次读取矩阵的行数、列数、非零元素个数和 tmp 值
    fscanf(fp, "%14c", buf); *nrow = atoi(buf);
    fscanf(fp, "%14c", buf); *ncol = atoi(buf);
    fscanf(fp, "%14c", buf); *nonz = atoi(buf);
    fscanf(fp, "%14c", buf); tmp = atoi(buf);

    // 如果 tmp 不为 0，则输出提示信息
    if (tmp != 0)
        printf("This is not an assembled matrix!\n");
    // 如果矩阵行数不等于列数，则输出提示信息
    if (*nrow != *ncol)
        printf("Matrix is not square.\n");
    // 跳过多余的换行符
    sDumpLine(fp);

    /* Allocate storage for the three arrays ( nzval, rowind, colptr ) */
    // 根据读取的列数、非零元素个数分配内存空间
    sallocateA(*ncol, *nonz, nzval, rowind, colptr);

    /* Line 4: format statement */
    // 读取并解析第四行的格式信息，获取列数、行数和值的大小
    fscanf(fp, "%16c", buf);
    sParseIntFormat(buf, &colnum, &colsize);
    fscanf(fp, "%16c", buf);
    sParseIntFormat(buf, &rownum, &rowsize);
    fscanf(fp, "%20c", buf);
    sParseFloatFormat(buf, &valnum, &valsize);
    // 跳过多余的换行符
    sDumpLine(fp);

#ifdef DEBUG
    // 在 Debug 模式下输出读取到的信息
    printf("%d rows, %lld nonzeros\n", *nrow, (long long) *nonz);
    printf("colnum %d, colsize %d\n", colnum, colsize);
    printf("rownum %d, rowsize %d\n", rownum, rowsize);
    printf("valnum %d, valsize %d\n", valnum, valsize);
#endif

    // 读取列指针数组 colptr
    ReadVector(fp, *ncol+1, *colptr, colnum, colsize);
    // 读取行索引数组 rowind
    ReadVector(fp, *nonz, *rowind, rownum, rowsize);
    // 如果存在数值行，则读取非零元素值数组 nzval
    if ( numer_lines ) {
        sReadValues(fp, *nonz, *nzval, valnum, valsize);
    }

    // 检查矩阵是否对称，如果是则调用 FormFullA 函数进行处理
    sym = (type[1] == 'S' || type[1] == 's');
    if ( sym ) {
    FormFullA(*ncol, nonz, nzval, rowind, colptr);
    }

    // 关闭文件流
    fclose(fp);
}
```