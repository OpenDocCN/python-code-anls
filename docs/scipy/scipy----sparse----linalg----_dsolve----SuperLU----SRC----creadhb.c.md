# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\creadhb.c`

```
# 文件注释，指明文件内容版权和许可信息，此处使用 BSD 许可证
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file creadhb.c
 * \brief Read a matrix stored in Harwell-Boeing format
 *
 * <pre>
 * -- SuperLU routine (version 2.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November 15, 1997
 *
 * Purpose
 * =======
 * 
 * Read a COMPLEX PRECISION matrix stored in Harwell-Boeing format 
 * as described below.
 * 
 * Line 1 (A72,A8) 
 *      Col. 1 - 72   Title (TITLE) 
 *    Col. 73 - 80  Key (KEY) 
 * 
 * Line 2 (5I14) 
 *     Col. 1 - 14   Total number of lines excluding header (TOTCRD) 
 *     Col. 15 - 28  Number of lines for pointers (PTRCRD) 
 *     Col. 29 - 42  Number of lines for row (or variable) indices (INDCRD) 
 *     Col. 43 - 56  Number of lines for numerical values (VALCRD) 
 *    Col. 57 - 70  Number of lines for right-hand sides (RHSCRD) 
 *                    (including starting guesses and solution vectors 
 *               if present) 
 *                     (zero indicates no right-hand side data is present) 
 *
 * Line 3 (A3, 11X, 4I14) 
 *       Col. 1 - 3    Matrix type (see below) (MXTYPE) 
 *     Col. 15 - 28  Number of rows (or variables) (NROW) 
 *     Col. 29 - 42  Number of columns (or elements) (NCOL) 
 *    Col. 43 - 56  Number of row (or variable) indices (NNZERO) 
 *                  (equal to number of entries for assembled matrices) 
 *     Col. 57 - 70  Number of elemental matrix entries (NELTVL) 
 *                  (zero in the case of assembled matrices) 
 * Line 4 (2A16, 2A20) 
 *     Col. 1 - 16   Format for pointers (PTRFMT) 
 *    Col. 17 - 32  Format for row (or variable) indices (INDFMT) 
 *    Col. 33 - 52  Format for numerical values of coefficient matrix (VALFMT) 
 *     Col. 53 - 72 Format for numerical values of right-hand sides (RHSFMT) 
 *
 * Line 5 (A3, 11X, 2I14) Only present if there are right-hand sides present 
 *        Col. 1           Right-hand side type: 
 *                   F for full storage or M for same format as matrix 
 *        Col. 2        G if a starting vector(s) (Guess) is supplied. (RHSTYP) 
 *        Col. 3        X if an exact solution vector(s) is supplied. 
 *    Col. 15 - 28  Number of right-hand sides (NRHS) 
 *    Col. 29 - 42  Number of row indices (NRHSIX) 
 *                    (ignored in case of unassembled matrices) 
 *
 * The three character type field on line 3 describes the matrix type. 
 * The following table lists the permitted values for each of the three 
 * characters. As an example of the type field, RSA denotes that the matrix 
 * is real, symmetric, and assembled. 
 *
 * First Character: 
 *    R Real matrix 
 *    C Complex matrix 
 *    P Pattern only (no numerical values supplied) 
 *
 * Second Character: 
 *    S Symmetric 
 *    U Unsymmetric 
 *    H Hermitian 
 *    Z Skew symmetric 
 *    R Rectangular 
 *
 * Third Character: 
 *    A Assembled 
 *    E Elemental matrices (unassembled) 
 *
 * </pre>
 */
#include <stdio.h>
#include <stdlib.h>
#include "slu_cdefs.h"

// No additional code to comment.
/*! \brief Eat up the rest of the current line */
int cDumpLine(FILE *fp)
{
    register int c;
    // 循环读取文件流中的字符，直到遇到换行符为止，吸收当前行的剩余内容
    while ((c = fgetc(fp)) != '\n') ;
    return 0;
}

/*! \brief Parse an integer format from a string buffer */
int cParseIntFormat(char *buf, int *num, int *size)
{
    char *tmp;

    tmp = buf;
    // 在字符串缓冲区中找到 '(' 符号
    while (*tmp++ != '(') ;
    // 从找到的位置解析出一个整数，存入 num 指向的位置
    sscanf(tmp, "%d", num);
    // 继续寻找 'I' 或 'i' 字符
    while (*tmp != 'I' && *tmp != 'i') ++tmp;
    ++tmp;
    // 解析出第二个整数，表示大小，存入 size 指向的位置
    sscanf(tmp, "%d", size);
    return 0;
}

/*! \brief Parse a floating-point format from a string buffer */
int cParseFloatFormat(char *buf, int *num, int *size)
{
    char *tmp, *period;
    
    tmp = buf;
    // 在字符串缓冲区中找到 '(' 符号
    while (*tmp++ != '(') ;
    // 使用 atoi 解析出一个整数，存入 num 指向的位置
    *num = atoi(tmp);
    // 继续寻找 'E', 'e', 'D', 'd', 'F', 'f' 中的一个字符
    while (*tmp != 'E' && *tmp != 'e' && *tmp != 'D' && *tmp != 'd'
       && *tmp != 'F' && *tmp != 'f') {
        // 如果遇到 'p' 或 'P'，则跳过当前整数解析下一个整数
        if (*tmp=='p' || *tmp=='P') {
           ++tmp;
           *num = atoi(tmp);
        } else {
           ++tmp;
        }
    }
    ++tmp;
    // 找到小数点或者 ')' 符号
    period = tmp;
    while (*period != '.' && *period != ')') ++period ;
    *period = '\0';
    // 使用 atoi 解析出浮点数大小，存入 size 指向的位置
    *size = atoi(tmp);

    return 0;
}

/*! \brief Read a vector from file */
static int ReadVector(FILE *fp, int_t n, int_t *where, int perline, int persize)
{
    int_t i, j, item;
    char tmp, buf[100];
    
    i = 0;
    // 逐行读取文件内容
    while (i <  n) {
        fgets(buf, 100, fp);    /* read a line at a time */
        // 逐个读取每行中的元素
        for (j=0; j<perline && i<n; j++) {
            tmp = buf[(j+1)*persize];     /* save the char at that place */
            buf[(j+1)*persize] = 0;       /* null terminate */
            // 将当前位置的字符串转换为整数，存入 where 数组中
            item = atoi(&buf[j*persize]); 
            buf[(j+1)*persize] = tmp;     /* recover the char at that place */
            where[i++] = item - 1;
        }
    }

    return 0;
}

/*! \brief Read complex numbers from file as (real, imaginary) pairs */
int cReadValues(FILE *fp, int n, singlecomplex *destination, int perline, int persize)
{
    register int i, j, k, s, pair;
    register float realpart;
    char tmp, buf[100];
    
    i = pair = 0;
    // 逐行读取文件内容
    while (i < n) {
        fgets(buf, 100, fp);    /* read a line at a time */
        // 逐个读取每行中的元素
        for (j=0; j<perline && i<n; j++) {
            tmp = buf[(j+1)*persize];     /* save the char at that place */
            buf[(j+1)*persize] = 0;       /* null terminate */
            s = j*persize;
            // 替换 'D' 或 'd' 字符为 'E'，在 C 语言中没有 'D_' 格式
            for (k = 0; k < persize; ++k)
                if ( buf[s+k] == 'D' || buf[s+k] == 'd' ) buf[s+k] = 'E';
            if ( pair == 0 ) {
                // 当前值为实部
                realpart = atof(&buf[s]);
                pair = 1;
            } else {
                // 当前值为虚部，存入 destination 数组中
                destination[i].r = realpart;
                destination[i++].i = atof(&buf[s]);
                pair = 0;
            }
            buf[(j+1)*persize] = tmp;     /* recover the char at that place */
        }
    }

    return 0;
}
FormFullA(int n, int_t *nonz, singlecomplex **nzval, int_t **rowind, int_t **colptr)
{
    int_t i, j, k, col, new_nnz;
    int_t *t_rowind, *t_colptr, *al_rowind, *al_colptr, *a_rowind, *a_colptr;
    int_t *marker;
    singlecomplex *t_val, *al_val, *a_val;

    al_rowind = *rowind;        // 将输入参数 *rowind 赋值给 al_rowind
    al_colptr = *colptr;        // 将输入参数 *colptr 赋值给 al_colptr
    al_val = *nzval;            // 将输入参数 *nzval 赋值给 al_val

    // 分配内存并检查
    if ( !(marker = intMalloc( (n+1) ) ) )
        ABORT("SUPERLU_MALLOC fails for marker[]");
    if ( !(t_colptr = intMalloc( (n+1) ) ) )
        ABORT("SUPERLU_MALLOC t_colptr[]");
    if ( !(t_rowind = intMalloc( *nonz ) ) )
        ABORT("SUPERLU_MALLOC fails for t_rowind[]");
    if ( !(t_val = (singlecomplex*) SUPERLU_MALLOC( *nonz * sizeof(singlecomplex)) ) )
        ABORT("SUPERLU_MALLOC fails for t_val[]");

    /* Get counts of each column of T, and set up column pointers */
    // 初始化 marker 数组为零
    for (i = 0; i < n; ++i) marker[i] = 0;
    // 遍历 al_colptr 数组，统计每列的元素个数
    for (j = 0; j < n; ++j) {
        for (i = al_colptr[j]; i < al_colptr[j+1]; ++i)
            ++marker[al_rowind[i]];
    }
    // 设置 t_colptr 数组，指示每列在 t_rowind 中的起始位置
    t_colptr[0] = 0;
    for (i = 0; i < n; ++i) {
        t_colptr[i+1] = t_colptr[i] + marker[i];
        marker[i] = t_colptr[i];
    }

    /* Transpose matrix A to T */
    // 转置矩阵 A 到 T
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
    // 分配内存并检查
    if ( !(a_colptr = intMalloc(n+1) ) )
        ABORT("SUPERLU_MALLOC a_colptr[]");
    if ( !(a_rowind = intMalloc( new_nnz ) ) )
        ABORT("SUPERLU_MALLOC fails for a_rowind[]");
    if ( !(a_val = (singlecomplex*) SUPERLU_MALLOC( new_nnz * sizeof(singlecomplex)) ) )
        ABORT("SUPERLU_MALLOC fails for a_val[]");

    // 初始化 a_colptr 数组
    a_colptr[0] = 0;
    k = 0;
    // 填充 a_rowind 和 a_val 数组，同时更新 a_colptr
    for (j = 0; j < n; ++j) {
        for (i = t_colptr[j]; i < t_colptr[j+1]; ++i) {
            if ( t_rowind[i] != j ) { /* not diagonal */
                a_rowind[k] = t_rowind[i];
                a_val[k] = t_val[i];
                // DEBUG 模式下输出绝对值小于阈值的元素
#ifdef DEBUG
                if ( c_abs1(&a_val[k]) < 4.047e-300 )
                    printf("%5d: %e\t%e\n", (int)k, a_val[k].r, a_val[k].i);
#endif
                ++k;
            }
        }

        for (i = al_colptr[j]; i < al_colptr[j+1]; ++i) {
            a_rowind[k] = al_rowind[i];
            a_val[k] = al_val[i];
#ifdef DEBUG
            if ( c_abs1(&a_val[k]) < 4.047e-300 )
                printf("%5d: %e\t%e\n", (int)k, a_val[k].r, a_val[k].i);
#endif
            ++k;
        }
        
        // 设置 a_colptr[j+1]，指示当前列结束的位置
        a_colptr[j+1] = k;
    }

    // 输出新的非零元素个数
    printf("FormFullA: new_nnz = %lld\n", (long long) new_nnz);

    // 释放内存
    SUPERLU_FREE(al_val);
    SUPERLU_FREE(al_rowind);
    SUPERLU_FREE(al_colptr);
    SUPERLU_FREE(marker);
    SUPERLU_FREE(t_val);
    SUPERLU_FREE(t_rowind);
    SUPERLU_FREE(t_colptr);

    // 更新输出参数
    *nzval = a_val;
    *rowind = a_rowind;
    *colptr = a_colptr;
    *nonz = new_nnz;
}
    // 从文件指针 fp 中读取一行内容，最多读取 99 个字符存入 buf 中，并输出到标准输出
    fgets(buf, 100, fp);
    // 将 buf 中的内容写入到标准输出
    fputs(buf, stdout);

    /* Line 2 */
    // 循环 5 次，每次从 fp 中按照格式"%14c"读取数据到 buf 中，最多读取 14 个字符
    fscanf(fp, "%14c", buf); buf[14] = 0;
    // 将 buf 中的内容解析成整数，存入 tmp 变量中
    sscanf(buf, "%d", &tmp);
    // 如果 i 等于 3，则将 tmp 赋值给 numer_lines
    if (i == 3) numer_lines = tmp;
    // 如果 i 等于 4 并且 tmp 不为零，则将 tmp 赋值给 rhscrd
    if (i == 4 && tmp) rhscrd = tmp;
    // 调用 cDumpLine 函数，传入 fp 指针，将当前行的数据转储
    cDumpLine(fp);

    /* Line 3 */
    // 从 fp 中按照格式"%3c"读取数据到 type 数组中，最多读取 3 个字符
    fscanf(fp, "%3c", type);
    // 从 fp 中按照格式"%11c"读取数据到 buf 中，最多读取 11 个字符，"/* pad */"表示此处读取用途
    fscanf(fp, "%11c", buf); /* pad */
    // 将 type 数组的内容截断，确保字符串结尾为 0
    type[3] = 0;
#if ( DEBUGlevel>=1 )
    printf("Matrix type %s\n", type);
#endif

fscanf(fp, "%14c", buf); *nrow = atoi(buf);
// 从文件流中读取一行，将其转换为整数并存储在 nrow 中

fscanf(fp, "%14c", buf); *ncol = atoi(buf);
// 从文件流中读取一行，将其转换为整数并存储在 ncol 中

fscanf(fp, "%14c", buf); *nonz = atoi(buf);
// 从文件流中读取一行，将其转换为整数并存储在 nonz 中

fscanf(fp, "%14c", buf); tmp = atoi(buf);
// 从文件流中读取一行，将其转换为整数并存储在 tmp 中

if (tmp != 0)
    printf("This is not an assembled matrix!\n");
// 如果 tmp 不等于 0，打印消息表明这不是一个已组装的矩阵

if (*nrow != *ncol)
    printf("Matrix is not square.\n");
// 如果 nrow 不等于 ncol，打印消息表明矩阵不是方阵

cDumpLine(fp);
// 跳过文件流中的当前行，用于将文件指针移动到下一行的起始位置

/* Allocate storage for the three arrays ( nzval, rowind, colptr ) */
callocateA(*ncol, *nonz, nzval, rowind, colptr);
// 分配存储空间用于三个数组（nzval, rowind, colptr），具体分配大小由 *ncol 和 *nonz 决定

/* Line 4: format statement */
fscanf(fp, "%16c", buf);
// 从文件流中读取一行，最多读取 16 个字符到 buf 中

cParseIntFormat(buf, &colnum, &colsize);
// 解析 buf 中的内容，获取列数和列大小信息，并存储在 colnum 和 colsize 中

fscanf(fp, "%16c", buf);
// 从文件流中读取一行，最多读取 16 个字符到 buf 中

cParseIntFormat(buf, &rownum, &rowsize);
// 解析 buf 中的内容，获取行数和行大小信息，并存储在 rownum 和 rowsize 中

fscanf(fp, "%20c", buf);
// 从文件流中读取一行，最多读取 20 个字符到 buf 中

cParseFloatFormat(buf, &valnum, &valsize);
// 解析 buf 中的内容，获取值数和值大小信息，并存储在 valnum 和 valsize 中

fscanf(fp, "%20c", buf);
// 从文件流中读取一行，最多读取 20 个字符到 buf 中

cDumpLine(fp);
// 跳过文件流中的当前行，用于将文件指针移动到下一行的起始位置

/* Line 5: right-hand side */
if (rhscrd) cDumpLine(fp);
// 如果 rhscrd 为真，则跳过文件流中的当前行

#ifdef DEBUG
printf("%d rows, %lld nonzeros\n", *nrow, (long long) *nonz);
printf("colnum %d, colsize %d\n", colnum, colsize);
printf("rownum %d, rowsize %d\n", rownum, rowsize);
printf("valnum %d, valsize %d\n", valnum, valsize);
#endif
// 如果定义了 DEBUG 宏，则打印相关调试信息

ReadVector(fp, *ncol+1, *colptr, colnum, colsize);
// 从文件流中读取数据，填充 *colptr 所指向的数组，长度为 *ncol+1，使用 colnum 和 colsize 进行格式化读取

ReadVector(fp, *nonz, *rowind, rownum, rowsize);
// 从文件流中读取数据，填充 *rowind 所指向的数组，长度为 *nonz，使用 rownum 和 rowsize 进行格式化读取

if (numer_lines) {
    cReadValues(fp, *nonz, *nzval, valnum, valsize);
}
// 如果 numer_lines 为真，则从文件流中读取数据，填充 *nzval 所指向的数组，长度为 *nonz，使用 valnum 和 valsize 进行格式化读取

sym = (type[1] == 'S' || type[1] == 's');
// 根据 type 的第二个字符判断是否是对称矩阵，结果存储在 sym 中

if (sym) {
    FormFullA(*ncol, nonz, nzval, rowind, colptr);
}
// 如果矩阵是对称的，则进行处理，填充完整的矩阵数据结构

fclose(fp);
// 关闭文件流
```