# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\zreadhb.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/



# 本段代码是一个文件级别的注释块，用于声明版权信息和许可证信息
/*! @file zreadhb.c
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
 * Read a DOUBLE COMPLEX PRECISION matrix stored in Harwell-Boeing format 
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
#include "slu_zdefs.h"
/*! \brief Eat up the rest of the current line */
int zDumpLine(FILE *fp)
{
    register int c;
    // 循环读取文件流中的字符，直到遇到换行符为止
    while ((c = fgetc(fp)) != '\n') ;
    return 0;
}

int zParseIntFormat(char *buf, int *num, int *size)
{
    char *tmp;
    
    tmp = buf;
    // 在缓冲区中查找 '(' 字符
    while (*tmp++ != '(') ;
    // 从 '(' 之后解析一个整数，存入 num
    sscanf(tmp, "%d", num);
    // 继续查找下一个 'I' 或 'i' 字符
    while (*tmp != 'I' && *tmp != 'i') ++tmp;
    ++tmp;
    // 解析紧随 'I' 或 'i' 字符之后的整数，存入 size
    sscanf(tmp, "%d", size);
    return 0;
}

int zParseFloatFormat(char *buf, int *num, int *size)
{
    char *tmp, *period;
    
    tmp = buf;
    // 在缓冲区中查找 '(' 字符
    while (*tmp++ != '(') ;
    // 解析括号内的整数，并存入 num
    *num = atoi(tmp);
    // 继续查找下一个 'E', 'e', 'D', 'd', 'F', 'f' 字符
    while (*tmp != 'E' && *tmp != 'e' && *tmp != 'D' && *tmp != 'd'
       && *tmp != 'F' && *tmp != 'f') {
        // 处理可能出现的 'p' 或 'P' 字符
        if (*tmp=='p' || *tmp=='P') {
           ++tmp;
           // 如果遇到 'p' 或 'P'，则解析其后的整数并存入 num
           *num = atoi(tmp);
        } else {
           ++tmp;
        }
    }
    ++tmp;
    period = tmp;
    // 继续查找 '.' 或 ')' 字符
    while (*period != '.' && *period != ')') ++period ;
    // 将找到的字符替换为字符串结尾符 '\0'
    *period = '\0';
    // 解析 period 指针指向的字符串为整数，并存入 size
    *size = atoi(tmp);

    return 0;
}

static int ReadVector(FILE *fp, int_t n, int_t *where, int perline, int persize)
{
    int_t i, j, item;
    char tmp, buf[100];
    
    i = 0;
    // 逐行读取文件流中的数据
    while (i <  n) {
        fgets(buf, 100, fp);    /* read a line at a time */
        for (j=0; j<perline && i<n; j++) {
            tmp = buf[(j+1)*persize];     /* save the char at that place */
            buf[(j+1)*persize] = 0;       /* null terminate */
            // 将缓冲区中的子字符串转换为整数
            item = atoi(&buf[j*persize]); 
            buf[(j+1)*persize] = tmp;     /* recover the char at that place */
            // 将转换后的整数存入目标数组中
            where[i++] = item - 1;
        }
    }

    return 0;
}

/*! \brief Read complex numbers as pairs of (real, imaginary) */
int zReadValues(FILE *fp, int n, doublecomplex *destination, int perline, int persize)
{
    register int i, j, k, s, pair;
    register double realpart;
    char tmp, buf[100];
    
    i = pair = 0;
    // 逐行读取文件流中的数据
    while (i < n) {
        fgets(buf, 100, fp);    /* read a line at a time */
        for (j=0; j<perline && i<n; j++) {
            tmp = buf[(j+1)*persize];     /* save the char at that place */
            buf[(j+1)*persize] = 0;       /* null terminate */
            s = j*persize;
            for (k = 0; k < persize; ++k) /* No D_ format in C */
                // 替换 'D' 或 'd' 为 'E'
                if ( buf[s+k] == 'D' || buf[s+k] == 'd' ) buf[s+k] = 'E';
            if ( pair == 0 ) {
                /* The value is real part */
                // 解析实部的浮点数值
                realpart = atof(&buf[s]);
                pair = 1;
            } else {
                /* The value is imaginary part */
                // 解析虚部的浮点数值，存入复数结构体数组中
                destination[i].r = realpart;
                destination[i++].i = atof(&buf[s]);
                pair = 0;
            }
            buf[(j+1)*persize] = tmp;     /* recover the char at that place */
        }
    }

    return 0;
}
FormFullA(int n, int_t *nonz, doublecomplex **nzval, int_t **rowind, int_t **colptr)
{
    int_t i, j, k, col, new_nnz;
    int_t *t_rowind, *t_colptr, *al_rowind, *al_colptr, *a_rowind, *a_colptr;
    int_t *marker;
    doublecomplex *t_val, *al_val, *a_val;

    al_rowind = *rowind;    // 将输入参数 *rowind 赋值给 al_rowind
    al_colptr = *colptr;    // 将输入参数 *colptr 赋值给 al_colptr
    al_val = *nzval;        // 将输入参数 *nzval 赋值给 al_val

    if ( !(marker = intMalloc( (n+1) ) ) )   // 分配 marker 数组内存，长度为 n+1
        ABORT("SUPERLU_MALLOC fails for marker[]");   // 若分配失败，输出错误信息并退出

    if ( !(t_colptr = intMalloc( (n+1) ) ) )  // 分配 t_colptr 数组内存，长度为 n+1
        ABORT("SUPERLU_MALLOC t_colptr[]");   // 若分配失败，输出错误信息并退出

    if ( !(t_rowind = intMalloc( *nonz ) ) )  // 分配 t_rowind 数组内存，长度为 *nonz
        ABORT("SUPERLU_MALLOC fails for t_rowind[]");  // 若分配失败，输出错误信息并退出

    if ( !(t_val = (doublecomplex*) SUPERLU_MALLOC( *nonz * sizeof(doublecomplex)) ) )  // 分配 t_val 数组内存，长度为 *nonz
        ABORT("SUPERLU_MALLOC fails for t_val[]");  // 若分配失败，输出错误信息并退出

    /* Get counts of each column of T, and set up column pointers */
    for (i = 0; i < n; ++i) marker[i] = 0;  // 初始化 marker 数组为零
    for (j = 0; j < n; ++j) {
        for (i = al_colptr[j]; i < al_colptr[j+1]; ++i)
            ++marker[al_rowind[i]];  // 统计每列 T 的非零元素个数，使用 marker 数组
    }

    t_colptr[0] = 0;  // 初始化 t_colptr 的第一个元素为零
    for (i = 0; i < n; ++i) {
        t_colptr[i+1] = t_colptr[i] + marker[i];  // 设置 t_colptr 的列指针
        marker[i] = t_colptr[i];  // 重置 marker 数组为 t_colptr 的起始索引
    }

    /* Transpose matrix A to T */
    for (j = 0; j < n; ++j) {
        for (i = al_colptr[j]; i < al_colptr[j+1]; ++i) {
            col = al_rowind[i];  // 获取 A 矩阵的列索引
            t_rowind[marker[col]] = j;  // 将 A 的行索引转置到 T 中
            t_val[marker[col]] = al_val[i];  // 将 A 的值转置到 T 中
            ++marker[col];  // 更新 marker 数组
        }
    }

    new_nnz = *nonz * 2 - n;  // 计算新的非零元素个数

    if ( !(a_colptr = intMalloc(n+1) ) )  // 分配 a_colptr 数组内存，长度为 n+1
        ABORT("SUPERLU_MALLOC a_colptr[]");  // 若分配失败，输出错误信息并退出

    if ( !(a_rowind = intMalloc( new_nnz ) ) )  // 分配 a_rowind 数组内存，长度为 new_nnz
        ABORT("SUPERLU_MALLOC fails for a_rowind[]");  // 若分配失败，输出错误信息并退出

    if ( !(a_val = (doublecomplex*) SUPERLU_MALLOC( new_nnz * sizeof(doublecomplex)) ) )  // 分配 a_val 数组内存，长度为 new_nnz
        ABORT("SUPERLU_MALLOC fails for a_val[]");  // 若分配失败，输出错误信息并退出
    
    a_colptr[0] = 0;  // 初始化 a_colptr 的第一个元素为零
    k = 0;
    for (j = 0; j < n; ++j) {
        for (i = t_colptr[j]; i < t_colptr[j+1]; ++i) {
            if ( t_rowind[i] != j ) { /* not diagonal */
                a_rowind[k] = t_rowind[i];  // 将非对角线元素的行索引存入 a_rowind
                a_val[k] = t_val[i];  // 将非对角线元素的值存入 a_val
                if ( z_abs1(&a_val[k]) < 4.047e-300 )  // 如果绝对值小于阈值
                    printf("%5d: %e\t%e\n", (int)k, a_val[k].r, a_val[k].i);  // 打印该元素的实部和虚部
                ++k;  // 更新 k
            }
        }

        for (i = al_colptr[j]; i < al_colptr[j+1]; ++i) {
            a_rowind[k] = al_rowind[i];  // 将 A 矩阵的行索引存入 a_rowind
            a_val[k] = al_val[i];  // 将 A 矩阵的值存入 a_val
#ifdef DEBUG
            if ( z_abs1(&a_val[k]) < 4.047e-300 )  // 如果绝对值小于阈值
                printf("%5d: %e\t%e\n", (int)k, a_val[k].r, a_val[k].i);  // 打印该元素的实部和虚部
#endif
            ++k;  // 更新 k
        }
        
        a_colptr[j+1] = k;  // 设置 a_colptr 的下一个指针
    }

    printf("FormFullA: new_nnz = %lld\n", (long long) new_nnz);  // 打印新的非零元素个数

    SUPERLU_FREE(al_val);  // 释放 al_val 内存
    SUPERLU_FREE(al_rowind);  // 释放 al_rowind 内存
    SUPERLU_FREE(al_colptr);  // 释放 al_colptr 内存
    SUPERLU_FREE(marker);  // 释放 marker 内存
    SUPERLU_FREE(t_val);  // 释放 t_val 内存
    SUPERLU_FREE(t_rowind);  // 释放 t_rowind 内存
    SUPERLU_FREE(t_colptr);  // 释放 t_colptr 内存

    *nzval = a_val;  // 更新输入参数 *nzval 的值为 a_val
    *rowind = a_rowind;  // 更新输入参数 *rowind 的值为 a_rowind
    *colptr = a_colptr;  // 更新输入参数 *colptr 的值为 a_colptr
    *nonz = new_nnz;  // 更新输入参数 *nonz 的值为 new_nnz
}
    # 从文件流 fp 中读取一行，最多读取 99 个字符存入 buf，然后将其输出到标准输出
    fgets(buf, 100, fp);
    # 将 buf 中的内容写入标准输出
    fputs(buf, stdout);

    /* Line 2 */
    # 循环 5 次，每次执行以下操作
    for (i=0; i<5; i++) {
        # 从文件流 fp 中读取 14 个字符到 buf 中，然后在 buf 末尾添加字符串结束符号
        fscanf(fp, "%14c", buf); buf[14] = 0;
        # 从 buf 中解析出一个整数，存入 tmp 中
        sscanf(buf, "%d", &tmp);
        # 如果 i 等于 3，则将 tmp 赋值给 numer_lines
        if (i == 3) numer_lines = tmp;
        # 如果 i 等于 4 且 tmp 非零，则将 tmp 赋值给 rhscrd
        if (i == 4 && tmp) rhscrd = tmp;
    }
    # 调用 zDumpLine 函数处理文件流 fp

    /* Line 3 */
    # 从文件流 fp 中读取 3 个字符到 type 中
    fscanf(fp, "%3c", type);
    # 从文件流 fp 中读取 11 个字符到 buf 中，"pad" 表示这些字符只起填充作用
    fscanf(fp, "%11c", buf); /* pad */
    # 在 type 的第四个位置添加字符串结束符号
    type[3] = 0;
#if ( DEBUGlevel>=1 )
    printf("Matrix type %s\n", type);
#endif

// 从文件中读取并解析矩阵的行数、列数、非零元素数及类型标志
fscanf(fp, "%14c", buf); *nrow = atoi(buf);  // 读取行数并转换为整数
fscanf(fp, "%14c", buf); *ncol = atoi(buf);  // 读取列数并转换为整数
fscanf(fp, "%14c", buf); *nonz = atoi(buf);  // 读取非零元素数并转换为整数
fscanf(fp, "%14c", buf); tmp = atoi(buf);    // 读取临时值并转换为整数

// 如果临时值不为0，则打印消息表明矩阵不是已组装的
if (tmp != 0)
    printf("This is not an assembled matrix!\n");

// 如果行数不等于列数，则打印消息表明矩阵不是方阵
if (*nrow != *ncol)
    printf("Matrix is not square.\n");

// 跳过文件中的一行内容
zDumpLine(fp);

/* 分配三个数组（nzval, rowind, colptr）的存储空间 */
zallocateA(*ncol, *nonz, nzval, rowind, colptr);

/* Line 4: format statement */
fscanf(fp, "%16c", buf);  // 读取格式行并存储到缓冲区
zParseIntFormat(buf, &colnum, &colsize);  // 解析整数格式并存储到colnum和colsize
fscanf(fp, "%16c", buf);  // 读取格式行并存储到缓冲区
zParseIntFormat(buf, &rownum, &rowsize);  // 解析整数格式并存储到rownum和rowsize
fscanf(fp, "%20c", buf);  // 读取格式行并存储到缓冲区
zParseFloatFormat(buf, &valnum, &valsize);  // 解析浮点数格式并存储到valnum和valsize
fscanf(fp, "%20c", buf);  // 读取格式行并存储到缓冲区
zDumpLine(fp);  // 跳过文件中的一行内容

/* Line 5: right-hand side */
if ( rhscrd ) zDumpLine(fp);  // 如果rhscrd为真，则跳过右手边的内容（RHSFMT）

#ifdef DEBUG
// 打印调试信息
printf("%d rows, %lld nonzeros\n", *nrow, (long long) *nonz);
printf("colnum %d, colsize %d\n", colnum, colsize);
printf("rownum %d, rowsize %d\n", rownum, rowsize);
printf("valnum %d, valsize %d\n", valnum, valsize);
#endif

// 读取列指针数组
ReadVector(fp, *ncol+1, *colptr, colnum, colsize);
// 读取行索引数组
ReadVector(fp, *nonz, *rowind, rownum, rowsize);
// 如果存在数值行，则读取数值数组
if ( numer_lines ) {
    zReadValues(fp, *nonz, *nzval, valnum, valsize);
}

// 根据矩阵类型判断是否对称，并在是对称矩阵时进行处理
sym = (type[1] == 'S' || type[1] == 's');
if ( sym ) {
    FormFullA(*ncol, nonz, nzval, rowind, colptr);  // 将稀疏对称矩阵转换为完整矩阵
}

fclose(fp);  // 关闭文件流
```