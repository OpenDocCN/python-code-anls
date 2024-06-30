# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\creadrb.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file creadrb.c
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
 * Read a COMPLEX PRECISION matrix stored in Rutherford-Boeing format 
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
 *      I Integer matrix
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
#include "slu_cdefs.h"


/*! \brief Eat up the rest of the current line */
static int cDumpLine(FILE *fp)
{
    register int c;
    // 循环读取文件流，直到遇到换行符为止，用于跳过当前行的内容
    while ((c = fgetc(fp)) != '\n') ;
    # 返回整数值 0，表示函数执行完成并返回结果
    return 0;
}

static int cParseIntFormat(char *buf, int *num, int *size)
{
    char *tmp;  /* 声明临时指针变量tmp */

    tmp = buf;  /* 将buf的地址赋给tmp */
    while (*tmp++ != '(') ;  /* 循环直到找到'('字符 */
    sscanf(tmp, "%d", num);  /* 使用sscanf从tmp中读取一个整数，存入num */
    while (*tmp != 'I' && *tmp != 'i') ++tmp;  /* 循环直到找到'I'或'i'字符 */
    ++tmp;  /* tmp向前移动一个位置 */
    sscanf(tmp, "%d", size);  /* 使用sscanf从tmp中读取一个整数，存入size */
    return 0;  /* 返回0表示函数执行成功 */
}

static int cParseFloatFormat(char *buf, int *num, int *size)
{
    char *tmp, *period;  /* 声明临时指针变量tmp和period */

    tmp = buf;  /* 将buf的地址赋给tmp */
    while (*tmp++ != '(') ;  /* 循环直到找到'('字符 */
    *num = atoi(tmp); /*sscanf(tmp, "%d", num);*/  /* 使用atoi将tmp转换为整数，存入num */
    while (*tmp != 'E' && *tmp != 'e' && *tmp != 'D' && *tmp != 'd'
           && *tmp != 'F' && *tmp != 'f') {
        /* 循环直到找到'E', 'e', 'D', 'd', 'F'或'f'字符 */
        /* 可能会在nE/nD/nF之前找到kP，如(1P6F13.6)。在这种情况下，num指的是P，应该跳过。 */
        if (*tmp=='p' || *tmp=='P') {
           ++tmp;  /* tmp向前移动一个位置 */
           *num = atoi(tmp); /*sscanf(tmp, "%d", num);*/  /* 使用atoi将tmp转换为整数，存入num */
        } else {
           ++tmp;  /* tmp向前移动一个位置 */
        }
    }
    ++tmp;  /* tmp向前移动一个位置 */
    period = tmp;  /* 将tmp的地址赋给period */
    while (*period != '.' && *period != ')') ++period ;  /* 循环直到找到'.'或')'字符 */
    *period = '\0';  /* 将period位置处设置为字符串结束符'\0' */
    *size = atoi(tmp); /*sscanf(tmp, "%2d", size);*/  /* 使用atoi将tmp转换为整数，存入size */

    return 0;  /* 返回0表示函数执行成功 */
}

static int ReadVector(FILE *fp, int n, int_t *where, int perline, int persize)
{
    int_t i, j, item;  /* 声明整数变量i, j, item */
    char tmp, buf[100];  /* 声明字符变量tmp，以及长度为100的字符数组buf */

    i = 0;  /* 初始化i为0 */
    while (i < n) {  /* 循环直到i小于n */
        fgets(buf, 100, fp);    /* 从文件fp中读取一行，存入buf */
        for (j=0; j<perline && i<n; j++) {
            tmp = buf[(j+1)*persize];     /* 保存当前位置的字符到tmp */
            buf[(j+1)*persize] = 0;       /* 将当前位置字符设置为字符串结束符'\0' */
            item = atoi(&buf[j*persize]);  /* 使用atoi将buf中指定位置开始的字符串转换为整数，存入item */
            buf[(j+1)*persize] = tmp;     /* 恢复当前位置的字符 */
            where[i++] = item - 1;  /* 将item减1后存入where数组，并递增i */
        }
    }

    return 0;  /* 返回0表示函数执行成功 */
}

/*! \brief Read complex numbers as pairs of (real, imaginary) */
static int cReadValues(FILE *fp, int n, singlecomplex *destination, int perline, int persize)
{
    register int i, j, k, s, pair;  /* 声明寄存器变量i, j, k, s, pair */
    register float realpart;  /* 声明寄存器变量realpart */
    char tmp, buf[100];  /* 声明字符变量tmp，以及长度为100的字符数组buf */
    
    i = pair = 0;  /* 初始化i和pair为0 */
    while (i < n) {  /* 循环直到i小于n */
        fgets(buf, 100, fp);    /* 从文件fp中读取一行，存入buf */
        for (j=0; j<perline && i<n; j++) {
            tmp = buf[(j+1)*persize];     /* 保存当前位置的字符到tmp */
            buf[(j+1)*persize] = 0;       /* 将当前位置字符设置为字符串结束符'\0' */
            s = j*persize;  /* 计算s的值 */
            for (k = 0; k < persize; ++k) /* No D_ format in C */  /* 循环直到k小于persize */
                if ( buf[s+k] == 'D' || buf[s+k] == 'd' ) buf[s+k] = 'E';  /* 如果buf中特定位置字符是'D'或'd'，则替换为'E' */
            if ( pair == 0 ) {
              /* The value is real part */  /* 这个值是实部 */
                realpart = atof(&buf[s]);  /* 使用atof将buf中指定位置开始的字符串转换为浮点数，存入realpart */
                pair = 1;  /* pair设置为1 */
            } else {
            /* The value is imaginary part */  /* 这个值是虚部 */
                destination[i].r = realpart;  /* 将realpart赋给destination数组中当前位置的实部 */
                destination[i++].i = atof(&buf[s]);  /* 将buf中指定位置开始的字符串转换为浮点数，存入destination数组中当前位置的虚部，并递增i */
                pair = 0;  /* pair设置为0 */
            }
            buf[(j+1)*persize] = tmp;     /* 恢复当前位置的字符 */
        }
    }

    return 0;  /* 返回0表示函数执行成功 */
}


/*! \brief
 *
 * <pre>
 * On input, nonz/nzval/rowind/colptr represents lower part of a symmetric
 * matrix. On exit, it represents the full matrix with lower and upper parts.
 * </pre>
 */
static void
FormFullA(int n, int_t *nonz, singlecomplex **nzval, int_t **rowind, int_t **colptr)
{
    /* 定义整数变量 */
    int_t i, j, k, col, new_nnz;
    /* 定义指向整数数组的指针变量 */
    int_t *t_rowind, *t_colptr, *al_rowind, *al_colptr, *a_rowind, *a_colptr;
    /* 定义指向单精度复数数组的指针变量 */
    singlecomplex *t_val, *al_val, *a_val;

    /* 将输入指针解引用，分配存储空间 */
    al_rowind = *rowind;
    al_colptr = *colptr;
    al_val = *nzval;

    /* 分配 marker 数组的存储空间 */
    if ( !(marker = intMalloc( n+1 ) ) )
        ABORT("SUPERLU_MALLOC fails for marker[]");
    /* 分配 t_colptr 数组的存储空间 */
    if ( !(t_colptr = intMalloc( n+1 ) ) )
        ABORT("SUPERLU_MALLOC t_colptr[]");
    /* 分配 t_rowind 数组的存储空间 */
    if ( !(t_rowind = intMalloc( *nonz ) ) )
        ABORT("SUPERLU_MALLOC fails for t_rowind[]");
    /* 分配 t_val 数组的存储空间 */
    if ( !(t_val = (singlecomplex*) SUPERLU_MALLOC( *nonz * sizeof(singlecomplex)) ) )
        ABORT("SUPERLU_MALLOC fails for t_val[]");

    /* 计算 T 的每列的非零元素个数，并设置列指针 */
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

    /* 将矩阵 A 转置为 T */
    for (j = 0; j < n; ++j) {
        for (i = al_colptr[j]; i < al_colptr[j+1]; ++i) {
            col = al_rowind[i];
            t_rowind[marker[col]] = j;
            t_val[marker[col]] = al_val[i];
            ++marker[col];
        }
    }

    /* 计算新矩阵 A 的非零元素个数 */
    new_nnz = *nonz * 2 - n;
    /* 分配 a_colptr 数组的存储空间 */
    if ( !(a_colptr = intMalloc( n+1 ) ) )
        ABORT("SUPERLU_MALLOC a_colptr[]");
    /* 分配 a_rowind 数组的存储空间 */
    if ( !(a_rowind = intMalloc( new_nnz) ) )
        ABORT("SUPERLU_MALLOC fails for a_rowind[]");
    /* 分配 a_val 数组的存储空间 */
    if ( !(a_val = (singlecomplex*) SUPERLU_MALLOC( new_nnz * sizeof(singlecomplex)) ) )
        ABORT("SUPERLU_MALLOC fails for a_val[]");
#ifdef DEBUG
      // 如果处于调试模式，并且a_val[k]的绝对值小于4.047e-300，则打印出k和a_val[k]的实部和虚部
      if ( c_abs1(&a_val[k]) < 4.047e-300 )
          printf("%5d: %e\t%e\n", (int)k, a_val[k].r, a_val[k].i);
#endif
      // k自增
      ++k;
    }
      }

      // 遍历al_colptr[j]到al_colptr[j+1]之间的元素
      for (i = al_colptr[j]; i < al_colptr[j+1]; ++i) {
    // 将al_rowind[i]赋给a_rowind[k]，将al_val[i]赋给a_val[k]
    a_rowind[k] = al_rowind[i];
    a_val[k] = al_val[i];
      // 如果处于调试模式，并且a_val[k]的绝对值小于4.047e-300，则打印出k和a_val[k]的实部和虚部
      if ( c_abs1(&a_val[k]) < 4.047e-300 )
          printf("%5d: %e\t%e\n", (int)k, a_val[k].r, a_val[k].i);
    // k自增
    ++k;
      }
      
      // 设置a_colptr[j+1]的值为k
      a_colptr[j+1] = k;
    }

    // 打印输出新的非零元素数量new_nnz
    printf("FormFullA: new_nnz = %lld\n", (long long) new_nnz);

    // 释放al_val、al_rowind、al_colptr、marker、t_val、t_rowind、t_colptr的内存空间
    SUPERLU_FREE(al_val);
    SUPERLU_FREE(al_rowind);
    SUPERLU_FREE(al_colptr);
    SUPERLU_FREE(marker);
    SUPERLU_FREE(t_val);
    SUPERLU_FREE(t_rowind);
    SUPERLU_FREE(t_colptr);

    // 将a_val、a_rowind、a_colptr、new_nnz分别赋给输出指针
    *nzval = a_val;
    *rowind = a_rowind;
    *colptr = a_colptr;
    *nonz = new_nnz;
}

void
creadrb(int *nrow, int *ncol, int_t *nonz,
        singlecomplex **nzval, int_t **rowind, int_t **colptr)
{

    register int i, numer_lines = 0;
    int tmp, colnum, colsize, rownum, rowsize, valnum, valsize;
    char buf[100], type[4];
    int sym;
    FILE *fp;

    fp = stdin;

    /* Line 1 */
    // 从标准输入中读取一行，存储到buf中，并输出到标准输出
    fgets(buf, 100, fp);
    fputs(buf, stdout);

    /* Line 2 */
    // 循环读取四个14字符长度的字符串，将其中的数字转换为整数存储在tmp中，并根据索引i设置numer_lines
    for (i=0; i<4; i++) {
        fscanf(fp, "%14c", buf); buf[14] = 0;
        sscanf(buf, "%d", &tmp);
        if (i == 3) numer_lines = tmp;
    }
    // 跳过本行剩余的输入
    cDumpLine(fp);

    /* Line 3 */
    // 读取三个字符到type中，然后读取11个字符到buf中，并输出type
    fscanf(fp, "%3c", type);
    fscanf(fp, "%11c", buf); /* pad */
    type[3] = 0;
#ifdef DEBUG
    // 如果处于调试模式，则打印矩阵类型type
    printf("Matrix type %s\n", type);
#endif

    // 读取后续的14字符到buf中，并分别转换为*nrow、*ncol、*nonz、tmp的值
    fscanf(fp, "%14c", buf); *nrow = atoi(buf);
    fscanf(fp, "%14c", buf); *ncol = atoi(buf);
    fscanf(fp, "%14c", buf); *nonz = atoi(buf);
    fscanf(fp, "%14c", buf); tmp = atoi(buf);

    // 如果tmp不为0，则打印"This is not an assembled matrix!"
    if (tmp != 0)
        printf("This is not an assembled matrix!\n");
    // 如果*nrow不等于*ncol，则打印"Matrix is not square."
    if (*nrow != *ncol)
        printf("Matrix is not square.\n");
    // 跳过本行剩余的输入
    cDumpLine(fp);

    /* Allocate storage for the three arrays ( nzval, rowind, colptr ) */
    // 为nzval、rowind、colptr分配存储空间
    callocateA(*ncol, *nonz, nzval, rowind, colptr);

    /* Line 4: format statement */
    // 读取16字符到buf中，并解析出colnum和colsize
    fscanf(fp, "%16c", buf);
    cParseIntFormat(buf, &colnum, &colsize);
    // 读取16字符到buf中，并解析出rownum和rowsize
    fscanf(fp, "%16c", buf);
    cParseIntFormat(buf, &rownum, &rowsize);
    // 读取20字符到buf中，并解析出valnum和valsize
    fscanf(fp, "%20c", buf);
    cParseFloatFormat(buf, &valnum, &valsize);
    // 跳过本行剩余的输入
    cDumpLine(fp);

#ifdef DEBUG
    // 如果处于调试模式，则打印*nrow、*nonz的值，以及colnum、colsize、rownum、rowsize、valnum、valsize的解析结果
    printf("%d rows, %lld nonzeros\n", *nrow, (long long) *nonz);
    printf("colnum %d, colsize %d\n", colnum, colsize);
    printf("rownum %d, rowsize %d\n", rownum, rowsize);
    printf("valnum %d, valsize %d\n", valnum, valsize);
#endif

    // 读取*ncol+1个值到*colptr中，使用colnum和colsize进行格式解析
    ReadVector(fp, *ncol+1, *colptr, colnum, colsize);
    // 读取*nonz个值到*rowind中，使用rownum和rowsize进行格式解析
    ReadVector(fp, *nonz, *rowind, rownum, rowsize);
    // 如果numer_lines不为0，则读取*nonz个值到*nzval中，使用valnum和valsize进行格式解析
    if ( numer_lines ) {
        cReadValues(fp, *nonz, *nzval, valnum, valsize);
    }

    // 判断矩阵类型是否对称，如果是则调用FormFullA函数重新构造矩阵
    sym = (type[1] == 'S' || type[1] == 's');
    if ( sym ) {
    FormFullA(*ncol, nonz, nzval, rowind, colptr);
    }

    // 关闭文件流fp
    fclose(fp);
}
```