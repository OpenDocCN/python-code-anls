# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dreadrb.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file dreadrb.c
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
 * Read a DOUBLE PRECISION matrix stored in Rutherford-Boeing format 
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
#include "slu_ddefs.h"

/*! \brief Eat up the rest of the current line */
static int dDumpLine(FILE *fp)
{
    register int c;
    // 逐字符读取文件流，直到遇到换行符为止，用于跳过当前行的剩余部分
    while ((c = fgetc(fp)) != '\n') ;
    # 返回整数值 0，结束函数执行并返回结果
    return 0;
static int dParseIntFormat(char *buf, int *num, int *size)
{
    char *tmp;  // 声明临时指针变量tmp

    tmp = buf;  // 将buf的地址赋给tmp，指向buf的起始位置
    while (*tmp++ != '(') ;  // 循环直到找到左括号'('，tmp指向其后一位
    sscanf(tmp, "%d", num);  // 从tmp指向的位置读取整数，存入num中
    while (*tmp != 'I' && *tmp != 'i') ++tmp;  // 继续移动tmp，直到找到'I'或'i'
    ++tmp;  // 跳过'I'或'i'
    sscanf(tmp, "%d", size);  // 从tmp指向的位置读取整数，存入size中
    return 0;  // 返回0表示函数执行成功
}



static int dParseFloatFormat(char *buf, int *num, int *size)
{
    char *tmp, *period;  // 声明临时指针变量tmp和period

    tmp = buf;  // 将buf的地址赋给tmp，指向buf的起始位置
    while (*tmp++ != '(') ;  // 循环直到找到左括号'('，tmp指向其后一位
    *num = atoi(tmp);  // 将tmp指向的字符串转换为整数，存入num中
    /*sscanf(tmp, "%d", num);*/  // 注释掉的旧代码，使用atoi替代了sscanf
    while (*tmp != 'E' && *tmp != 'e' && *tmp != 'D' && *tmp != 'd'
           && *tmp != 'F' && *tmp != 'f') {
        /* 可能会在nE/nD/nF之前找到kP，例如(1P6F13.6)。在这种情况下，
           选取的num指的是P，需要跳过。 */
        if (*tmp=='p' || *tmp=='P') {
           ++tmp;  // 跳过'p'或'P'
           *num = atoi(tmp);  // 将tmp指向的字符串转换为整数，存入num中
           /*sscanf(tmp, "%d", num);*/  // 注释掉的旧代码，使用atoi替代了sscanf
        } else {
           ++tmp;  // 继续移动tmp
        }
    }
    ++tmp;  // 跳过'E'或'e'或'D'或'd'或'F'或'f'
    period = tmp;  // 将tmp的地址赋给period
    while (*period != '.' && *period != ')') ++period ;  // 找到小数点或右括号为止
    *period = '\0';  // 在找到的位置处添加字符串结束符'\0'
    *size = atoi(tmp);  // 将tmp指向的字符串转换为整数，存入size中
    /*sscanf(tmp, "%2d", size);*/  // 注释掉的旧代码，使用atoi替代了sscanf

    return 0;  // 返回0表示函数执行成功
}



static int ReadVector(FILE *fp, int n, int_t *where, int perline, int persize)
{
    int_t i, j, item;
    char tmp, buf[100];

    i = 0;  // 初始化循环变量i为0
    while (i < n) {  // 循环直到i达到n
        fgets(buf, 100, fp);    // 从文件中读取一行内容到buf中
        for (j=0; j<perline && i<n; j++) {
            tmp = buf[(j+1)*persize];  // 保存当前位置的字符到tmp中
            buf[(j+1)*persize] = 0;    // 将当前位置的字符替换为字符串结束符'\0'
            item = atoi(&buf[j*persize]);  // 将当前位置字符串转换为整数，存入item中
            buf[(j+1)*persize] = tmp;  // 恢复当前位置的字符
            where[i++] = item - 1;  // 将item减1后存入where数组中，并递增i
        }
    }

    return 0;  // 返回0表示函数执行成功
}



static int dReadValues(FILE *fp, int n, double *destination, int perline,
        int persize)
{
    register int i, j, k, s;
    char tmp, buf[100];

    i = 0;  // 初始化循环变量i为0
    while (i < n) {  // 循环直到i达到n
        fgets(buf, 100, fp);    // 从文件中读取一行内容到buf中
        for (j=0; j<perline && i<n; j++) {
            tmp = buf[(j+1)*persize];  // 保存当前位置的字符到tmp中
            buf[(j+1)*persize] = 0;    // 将当前位置的字符替换为字符串结束符'\0'
            s = j*persize;
            for (k = 0; k < persize; ++k)  // 遍历persize次
                if ( buf[s+k] == 'D' || buf[s+k] == 'd' ) buf[s+k] = 'E';  // 如果是'D'或'd'，替换为'E'
            destination[i++] = atof(&buf[s]);  // 将当前位置字符串转换为浮点数，存入destination中，并递增i
            buf[(j+1)*persize] = tmp;  // 恢复当前位置的字符
        }
    }

    return 0;  // 返回0表示函数执行成功
}



/*! \brief
 *
 * <pre>
 * On input, nonz/nzval/rowind/colptr represents lower part of a symmetric
 * matrix. On exit, it represents the full matrix with lower and upper parts.
 * </pre>
 */
static void
FormFullA(int n, int_t *nonz, double **nzval, int_t **rowind, int_t **colptr)
{
    int_t i, j, k, col, new_nnz;
    int_t *t_rowind, *t_colptr, *al_rowind, *al_colptr, *a_rowind, *a_colptr;
    int_t *marker;
    double *t_val, *al_val, *a_val;

    al_rowind = *rowind;  // 将rowind指向的地址赋给al_rowind
    al_colptr = *colptr;  // 将colptr指向的地址赋给al_colptr
    al_val = *nzval;  // 将nzval指向的地址赋给al_val

    if ( !(marker = intMalloc( n+1 ) ) )  // 分配大小为n+1的int型数组给marker，若分配失败则输出错误信息
    ABORT("SUPERLU_MALLOC fails for marker[]");

    /* 更多代码... */

}
    # 分配并初始化 t_colptr 数组，用于存储 T 矩阵每列的起始索引位置
    if ( !(t_colptr = intMalloc( n+1 ) ) )
        ABORT("SUPERLU_MALLOC t_colptr[]");

    # 分配并初始化 t_rowind 数组，用于存储 T 矩阵中非零元素的行索引
    if ( !(t_rowind = intMalloc( *nonz ) ) )
        ABORT("SUPERLU_MALLOC fails for t_rowind[]");

    # 分配并初始化 t_val 数组，用于存储 T 矩阵中非零元素的值
    if ( !(t_val = (double*) SUPERLU_MALLOC( *nonz * sizeof(double)) ) )
        ABORT("SUPERLU_MALLOC fails for t_val[]");

    /* Get counts of each column of T, and set up column pointers */
    # 计算 T 矩阵每列的非零元素个数，并设置 t_colptr 的指针位置
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

    /* Transpose matrix A to T */
    # 将矩阵 A 转置为矩阵 T
    for (j = 0; j < n; ++j) {
        for (i = al_colptr[j]; i < al_colptr[j+1]; ++i) {
            col = al_rowind[i];
            t_rowind[marker[col]] = j;
            t_val[marker[col]] = al_val[i];
            ++marker[col];
        }
    }

    # 计算新的非零元素个数，用于分配 a_colptr、a_rowind 和 a_val 数组
    new_nnz = *nonz * 2 - n;
    if ( !(a_colptr = intMalloc( n+1 ) ) )
        ABORT("SUPERLU_MALLOC a_colptr[]");
    if ( !(a_rowind = intMalloc( new_nnz) ) )
        ABORT("SUPERLU_MALLOC fails for a_rowind[]");
    if ( !(a_val = (double*) SUPERLU_MALLOC( new_nnz * sizeof(double)) ) )
        ABORT("SUPERLU_MALLOC fails for a_val[]");

    a_colptr[0] = 0;
    k = 0;
    for (j = 0; j < n; ++j) {
        for (i = t_colptr[j]; i < t_colptr[j+1]; ++i) {
            if ( t_rowind[i] != j ) { /* not diagonal */
                a_rowind[k] = t_rowind[i];
                a_val[k] = t_val[i];
                ++k;
            }
        }
        a_colptr[j+1] = k;
    }
#ifdef DEBUG
      // 如果定义了 DEBUG 宏，则执行以下代码块
      if ( fabs(a_val[k]) < 4.047e-300 )
          // 如果数组 a_val 中第 k 个元素的绝对值小于 4.047e-300，则输出其值和索引
          printf("%5d: %e\n", (int)k, a_val[k]);
#endif
      // k 自增，无论 DEBUG 宏是否定义
      ++k;
    }
      }

      // 遍历 al_colptr[j] 到 al_colptr[j+1] 之间的元素
      for (i = al_colptr[j]; i < al_colptr[j+1]; ++i) {
    // 将 al_rowind[i] 的值赋给 a_rowind[k]
    a_rowind[k] = al_rowind[i];
    // 将 al_val[i] 的值赋给 a_val[k]
    a_val[k] = al_val[i];
      // 如果数组 a_val 中第 k 个元素的绝对值小于 4.047e-300，则输出其值和索引
      if ( fabs(a_val[k]) < 4.047e-300 )
          printf("%5d: %e\n", (int)k, a_val[k]);
    // k 自增
    ++k;
      }
      
      // 将 k 的值赋给 a_colptr[j+1]
      a_colptr[j+1] = k;
    }

    // 输出 new_nnz 的值
    printf("FormFullA: new_nnz = %lld\n", (long long) new_nnz);

    // 释放内存：释放 al_val 数组
    SUPERLU_FREE(al_val);
    // 释放内存：释放 al_rowind 数组
    SUPERLU_FREE(al_rowind);
    // 释放内存：释放 al_colptr 数组
    SUPERLU_FREE(al_colptr);
    // 释放内存：释放 marker 数组
    SUPERLU_FREE(marker);
    // 释放内存：释放 t_val 数组
    SUPERLU_FREE(t_val);
    // 释放内存：释放 t_rowind 数组
    SUPERLU_FREE(t_rowind);
    // 释放内存：释放 t_colptr 数组
    SUPERLU_FREE(t_colptr);

    // 将 a_val 的地址赋给 nzval 指针
    *nzval = a_val;
    // 将 a_rowind 的地址赋给 rowind 指针
    *rowind = a_rowind;
    // 将 a_colptr 的地址赋给 colptr 指针
    *colptr = a_colptr;
    // 将 new_nnz 的值赋给 nonz 指针
    *nonz = new_nnz;
}

// 函数定义：读取稠密矩阵文件格式，返回稀疏矩阵的三个数组指针
void
dreadrb(int *nrow, int *ncol, int_t *nonz,
        double **nzval, int_t **rowind, int_t **colptr)
{

    register int i, numer_lines = 0;
    int tmp, colnum, colsize, rownum, rowsize, valnum, valsize;
    char buf[100], type[4];
    int sym;
    FILE *fp;

    // 从标准输入读取一行，并输出到标准输出
    fp = stdin;
    fgets(buf, 100, fp);
    fputs(buf, stdout);

    // 读取四个14字符宽度的整数，存储在 buf 中，并根据情况赋值给 numer_lines
    for (i=0; i<4; i++) {
        fscanf(fp, "%14c", buf); buf[14] = 0;
        sscanf(buf, "%d", &tmp);
        if (i == 3) numer_lines = tmp;
    }
    // 跳过当前行的剩余内容
    dDumpLine(fp);

    // 读取矩阵类型到 type 数组，并在 DEBUG 模式下输出该类型
    fscanf(fp, "%3c", type);
    fscanf(fp, "%11c", buf); /* pad */
    type[3] = 0;
#ifdef DEBUG
    printf("Matrix type %s\n", type);
#endif

    // 依次读取矩阵的行数、列数、非零元素数目和一个临时值
    fscanf(fp, "%14c", buf); *nrow = atoi(buf);
    fscanf(fp, "%14c", buf); *ncol = atoi(buf);
    fscanf(fp, "%14c", buf); *nonz = atoi(buf);
    fscanf(fp, "%14c", buf); tmp = atoi(buf);

    // 如果临时值不为零，输出信息表明这不是一个组装好的矩阵
    if (tmp != 0)
        printf("This is not an assembled matrix!\n");
    // 如果行数不等于列数，输出信息表明矩阵不是方阵
    if (*nrow != *ncol)
        printf("Matrix is not square.\n");
    // 跳过当前行的剩余内容
    dDumpLine(fp);

    // 分配三个数组 nzval、rowind、colptr 的存储空间
    dallocateA(*ncol, *nonz, nzval, rowind, colptr);

    // 读取格式语句到 buf 中，并解析得到列号和列宽
    fscanf(fp, "%16c", buf);
    dParseIntFormat(buf, &colnum, &colsize);
    fscanf(fp, "%16c", buf);
    dParseIntFormat(buf, &rownum, &rowsize);
    fscanf(fp, "%20c", buf);
    dParseFloatFormat(buf, &valnum, &valsize);
    // 跳过当前行的剩余内容
    dDumpLine(fp);

#ifdef DEBUG
    // 在 DEBUG 模式下输出矩阵的行数、非零元素数目以及列、行、值的格式信息
    printf("%d rows, %lld nonzeros\n", *nrow, (long long) *nonz);
    printf("colnum %d, colsize %d\n", colnum, colsize);
    printf("rownum %d, rowsize %d\n", rownum, rowsize);
    printf("valnum %d, valsize %d\n", valnum, valsize);
#endif

    // 读取列指针数组 colptr
    ReadVector(fp, *ncol+1, *colptr, colnum, colsize);
    // 读取行索引数组 rowind
    ReadVector(fp, *nonz, *rowind, rownum, rowsize);
    // 如果存在数值行，则读取数值数组 nzval
    if ( numer_lines ) {
        dReadValues(fp, *nonz, *nzval, valnum, valsize);
    }

    // 判断矩阵是否对称，如果是则调用 FormFullA 函数生成完整的矩阵
    sym = (type[1] == 'S' || type[1] == 's');
    if ( sym ) {
    FormFullA(*ncol, nonz, nzval, rowind, colptr);
    }

    // 关闭文件流 fp
    fclose(fp);
}
```