# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zreadrb.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file zreadrb.c
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
 * Read a DOUBLE COMPLEX PRECISION matrix stored in Rutherford-Boeing format 
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
#include "slu_zdefs.h"

/*! \brief Eat up the rest of the current line */
static int zDumpLine(FILE *fp)
{
    register int c;


注释：

    // 注册一个整型变量c，用于存储读取的字符
    register int c;
    # 循环读取文件流中的字符，直到遇到换行符为止
    while ((c = fgetc(fp)) != '\n') ;
    # 返回0，表示函数执行成功
    return 0;
}

static int zParseIntFormat(char *buf, int *num, int *size)
{
    char *tmp;

    tmp = buf;  // 将 buf 赋给 tmp
    while (*tmp++ != '(') ;  // 查找 '(' 字符
    sscanf(tmp, "%d", num);  // 从 tmp 中读取一个整数到 num
    while (*tmp != 'I' && *tmp != 'i') ++tmp;  // 查找下一个 'I' 或 'i'
    ++tmp;  // 跳过 'I' 或 'i'
    sscanf(tmp, "%d", size);  // 从 tmp 中读取一个整数到 size
    return 0;  // 返回 0 表示成功
}

static int zParseFloatFormat(char *buf, int *num, int *size)
{
    char *tmp, *period;

    tmp = buf;  // 将 buf 赋给 tmp
    while (*tmp++ != '(') ;  // 查找 '(' 字符
    *num = atoi(tmp); /*sscanf(tmp, "%d", num);*/  // 将 tmp 中的数字转换为整数并赋给 num
    while (*tmp != 'E' && *tmp != 'e' && *tmp != 'D' && *tmp != 'd'
           && *tmp != 'F' && *tmp != 'f') {
        /* May find kP before nE/nD/nF, like (1P6F13.6). In this case the
           num picked up refers to P, which should be skipped. */
        if (*tmp=='p' || *tmp=='P') {
           ++tmp;  // 跳过 'p' 或 'P'
           *num = atoi(tmp); /*sscanf(tmp, "%d", num);*/  // 将 tmp 中的数字转换为整数并赋给 num
        } else {
           ++tmp;  // 继续下一个字符
        }
    }
    ++tmp;  // 跳过 'E', 'e', 'D', 'd', 'F', 'f'
    period = tmp;  // 将 tmp 赋给 period
    while (*period != '.' && *period != ')') ++period ;  // 查找 '.' 或 ')' 字符
    *period = '\0';  // 将 '.' 或 ')' 替换为字符串结束符 '\0'
    *size = atoi(tmp); /*sscanf(tmp, "%2d", size);*/  // 将 tmp 中的数字转换为整数并赋给 size

    return 0;  // 返回 0 表示成功
}

static int ReadVector(FILE *fp, int n, int_t *where, int perline, int persize)
{
    int_t i, j, item;
    char tmp, buf[100];

    i = 0;  // 初始化 i 为 0
    while (i < n) {  // 当 i 小于 n 时循环
        fgets(buf, 100, fp);    /* read a line at a time */  // 从 fp 中读取一行数据到 buf
        for (j=0; j<perline && i<n; j++) {
            tmp = buf[(j+1)*persize];     /* save the char at that place */  // 保存当前位置的字符到 tmp
            buf[(j+1)*persize] = 0;       /* null terminate */  // 在当前位置添加字符串结束符
            item = atoi(&buf[j*persize]);  // 将 buf 中的数字转换为整数并赋给 item
            buf[(j+1)*persize] = tmp;     /* recover the char at that place */  // 恢复当前位置的字符
            where[i++] = item - 1;  // 将 item 减去 1 赋给 where[i]，并递增 i
        }
    }

    return 0;  // 返回 0 表示成功
}

/*! \brief Read complex numbers as pairs of (real, imaginary) */
static int zReadValues(FILE *fp, int n, doublecomplex *destination, int perline, int persize)
{
    register int i, j, k, s, pair;
    register double realpart;
    char tmp, buf[100];
    
    i = pair = 0;  // 初始化 i 和 pair 为 0
    while (i < n) {  // 当 i 小于 n 时循环
        fgets(buf, 100, fp);    /* read a line at a time */  // 从 fp 中读取一行数据到 buf
        for (j=0; j<perline && i<n; j++) {
            tmp = buf[(j+1)*persize];     /* save the char at that place */  // 保存当前位置的字符到 tmp
            buf[(j+1)*persize] = 0;       /* null terminate */  // 在当前位置添加字符串结束符
            s = j*persize;  // 计算 s 的值
            for (k = 0; k < persize; ++k) /* No D_ format in C */
            if ( buf[s+k] == 'D' || buf[s+k] == 'd' ) buf[s+k] = 'E';  // 如果 buf[s+k] 为 'D' 或 'd'，则替换为 'E'
            if ( pair == 0 ) {
              /* The value is real part */
            realpart = atof(&buf[s]);  // 将 buf[s] 开始的字符串转换为双精度浮点数并赋给 realpart
            pair = 1;  // 设置 pair 为 1
            } else {
            /* The value is imaginary part */
                destination[i].r = realpart;  // 将 realpart 赋给 destination[i].r
            destination[i++].i = atof(&buf[s]);  // 将 buf[s] 开始的字符串转换为双精度浮点数并赋给 destination[i].i，然后递增 i
            pair = 0;  // 设置 pair 为 0
            }
            buf[(j+1)*persize] = tmp;     /* recover the char at that place */  // 恢复当前位置的字符
        }
    }

    return 0;  // 返回 0 表示成功
}


/*! \brief
 *
 * <pre>
 * On input, nonz/nzval/rowind/colptr represents lower part of a symmetric
 * matrix. On exit, it represents the full matrix with lower and upper parts.
 * </pre>
 */
static void
FormFullA(int n, int_t *nonz, doublecomplex **nzval, int_t **rowind, int_t **colptr)
{
    // 声明整数变量 i, j, k, col, new_nnz
    int_t i, j, k, col, new_nnz;
    // 声明指针变量 t_rowind, t_colptr, al_rowind, al_colptr, a_rowind, a_colptr
    int_t *t_rowind, *t_colptr, *al_rowind, *al_colptr, *a_rowind, *a_colptr;
    // 声明指针变量 marker
    int_t *marker;
    // 声明双精度复数指针变量 t_val, al_val, a_val
    doublecomplex *t_val, *al_val, *a_val;

    // 将输入指针赋值给相应变量
    al_rowind = *rowind;
    al_colptr = *colptr;
    al_val = *nzval;

    // 分配 marker 数组的内存空间
    if ( !(marker = intMalloc( n+1 ) ) )
    ABORT("SUPERLU_MALLOC fails for marker[]");
    // 分配 t_colptr 数组的内存空间
    if ( !(t_colptr = intMalloc( n+1 ) ) )
    ABORT("SUPERLU_MALLOC t_colptr[]");
    // 分配 t_rowind 数组的内存空间
    if ( !(t_rowind = intMalloc( *nonz ) ) )
    ABORT("SUPERLU_MALLOC fails for t_rowind[]");
    // 分配 t_val 数组的内存空间
    if ( !(t_val = (doublecomplex*) SUPERLU_MALLOC( *nonz * sizeof(doublecomplex)) ) )
    ABORT("SUPERLU_MALLOC fails for t_val[]");

    /* Get counts of each column of T, and set up column pointers */
    // 初始化 marker 数组
    for (i = 0; i < n; ++i) marker[i] = 0;
    // 计算每列 T 的非零元素个数，并设置列指针
    for (j = 0; j < n; ++j) {
        for (i = al_colptr[j]; i < al_colptr[j+1]; ++i)
            ++marker[al_rowind[i]];
    }
    // 设置 t_colptr 的起始值
    t_colptr[0] = 0;
    // 设置每列的列指针
    for (i = 0; i < n; ++i) {
        t_colptr[i+1] = t_colptr[i] + marker[i];
        marker[i] = t_colptr[i];
    }

    /* Transpose matrix A to T */
    // 将矩阵 A 转置为 T
    for (j = 0; j < n; ++j) {
        for (i = al_colptr[j]; i < al_colptr[j+1]; ++i) {
            col = al_rowind[i];
            t_rowind[marker[col]] = j;
            t_val[marker[col]] = al_val[i];
            ++marker[col];
        }
    }

    // 计算新的非零元素个数
    new_nnz = *nonz * 2 - n;
    // 分配 a_colptr 数组的内存空间
    if ( !(a_colptr = intMalloc( n+1 ) ) )
    ABORT("SUPERLU_MALLOC a_colptr[]");
    // 分配 a_rowind 数组的内存空间
    if ( !(a_rowind = intMalloc( new_nnz) ) )
    ABORT("SUPERLU_MALLOC fails for a_rowind[]");
    // 分配 a_val 数组的内存空间
    if ( !(a_val = (doublecomplex*) SUPERLU_MALLOC( new_nnz * sizeof(doublecomplex)) ) )
    ABORT("SUPERLU_MALLOC fails for a_val[]");
#ifdef DEBUG
      // 如果定义了 DEBUG 宏，则执行下面的代码块
      if ( z_abs1(&a_val[k]) < 4.047e-300 )
          // 如果 a_val[k] 的绝对值小于 4.047e-300，则输出 k 和 a_val[k] 的实部和虚部
          printf("%5d: %e\t%e\n", (int)k, a_val[k].r, a_val[k].i);
#endif
      // 不论是否执行上面的条件语句，都将 k 的值增加 1
      ++k;
    }
      }

      // 遍历 al_colptr[j] 到 al_colptr[j+1] 之间的元素
      for (i = al_colptr[j]; i < al_colptr[j+1]; ++i) {
    // 将 al_rowind[i] 赋值给 a_rowind[k]
    a_rowind[k] = al_rowind[i];
    // 将 al_val[i] 赋值给 a_val[k]
    a_val[k] = al_val[i];
      // 如果 a_val[k] 的绝对值小于 4.047e-300，则输出 k 和 a_val[k] 的实部和虚部
      if ( z_abs1(&a_val[k]) < 4.047e-300 )
          printf("%5d: %e\t%e\n", (int)k, a_val[k].r, a_val[k].i);
    // 将 k 的值增加 1
    ++k;
      }
      
      // 将 k 的值赋给 a_colptr[j+1]
      a_colptr[j+1] = k;
    }

    // 打印输出新的非零元素个数 new_nnz
    printf("FormFullA: new_nnz = %lld\n", (long long) new_nnz);

    // 释放动态分配的内存空间
    SUPERLU_FREE(al_val);
    SUPERLU_FREE(al_rowind);
    SUPERLU_FREE(al_colptr);
    SUPERLU_FREE(marker);
    SUPERLU_FREE(t_val);
    SUPERLU_FREE(t_rowind);
    SUPERLU_FREE(t_colptr);

    // 将计算得到的结果分配给指针变量
    *nzval = a_val;
    *rowind = a_rowind;
    *colptr = a_colptr;
    *nonz = new_nnz;
}

void
zreadrb(int *nrow, int *ncol, int_t *nonz,
        doublecomplex **nzval, int_t **rowind, int_t **colptr)
{

    register int i, numer_lines = 0;
    int tmp, colnum, colsize, rownum, rowsize, valnum, valsize;
    char buf[100], type[4];
    int sym;
    FILE *fp;

    // 从标准输入读取文件指针
    fp = stdin;

    /* Line 1 */
    // 读取并输出第一行内容
    fgets(buf, 100, fp);
    fputs(buf, stdout);

    /* Line 2 */
    // 读取第二行的四个数值，并根据需要进行输出
    for (i=0; i<4; i++) {
        fscanf(fp, "%14c", buf); buf[14] = 0;
        sscanf(buf, "%d", &tmp);
        if (i == 3) numer_lines = tmp;
    }
    // 跳过剩余的输入行
    zDumpLine(fp);

    /* Line 3 */
    // 读取矩阵类型并输出（仅在 DEBUG 模式下）
    fscanf(fp, "%3c", type);
    fscanf(fp, "%11c", buf); /* pad */
    type[3] = 0;
#ifdef DEBUG
    printf("Matrix type %s\n", type);
#endif

    // 读取行数、列数、非零元素个数以及一个临时变量的值，并进行相应的检查和输出
    fscanf(fp, "%14c", buf); *nrow = atoi(buf);
    fscanf(fp, "%14c", buf); *ncol = atoi(buf);
    fscanf(fp, "%14c", buf); *nonz = atoi(buf);
    fscanf(fp, "%14c", buf); tmp = atoi(buf);

    if (tmp != 0)
        printf("This is not an assembled matrix!\n");
    if (*nrow != *ncol)
        printf("Matrix is not square.\n");
    // 跳过剩余的输入行
    zDumpLine(fp);

    // 为三个数组（nzval、rowind、colptr）分配存储空间
    zallocateA(*ncol, *nonz, nzval, rowind, colptr);

    /* Line 4: format statement */
    // 读取格式说明并解析，输出相关信息（仅在 DEBUG 模式下）
    fscanf(fp, "%16c", buf);
    zParseIntFormat(buf, &colnum, &colsize);
    fscanf(fp, "%16c", buf);
    zParseIntFormat(buf, &rownum, &rowsize);
    fscanf(fp, "%20c", buf);
    zParseFloatFormat(buf, &valnum, &valsize);
    // 跳过剩余的输入行
    zDumpLine(fp);

#ifdef DEBUG
    printf("%d rows, %lld nonzeros\n", *nrow, (long long) *nonz);
    printf("colnum %d, colsize %d\n", colnum, colsize);
    printf("rownum %d, rowsize %d\n", rownum, rowsize);
    printf("valnum %d, valsize %d\n", valnum, valsize);
#endif

    // 读取列指针数组 colptr 的值
    ReadVector(fp, *ncol+1, *colptr, colnum, colsize);
    // 读取行索引数组 rowind 的值
    ReadVector(fp, *nonz, *rowind, rownum, rowsize);
    // 如果有数值数组 nzval，则读取其值
    if ( numer_lines ) {
        zReadValues(fp, *nonz, *nzval, valnum, valsize);
    }

    // 检查矩阵是否对称，如果是，则调用 FormFullA 函数生成完整矩阵
    sym = (type[1] == 'S' || type[1] == 's');
    if ( sym ) {
    FormFullA(*ncol, nonz, nzval, rowind, colptr);
    }

    // 关闭文件指针
    fclose(fp);
}
```