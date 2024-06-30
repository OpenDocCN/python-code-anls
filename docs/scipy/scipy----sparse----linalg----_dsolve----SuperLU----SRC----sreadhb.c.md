# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\sreadhb.c`

```
/*!
   \file
   Copyright (c) 2003, The Regents of the University of California, through
   Lawrence Berkeley National Laboratory (subject to receipt of any required 
   approvals from U.S. Dept. of Energy) 

   All rights reserved. 

   The source code is distributed under BSD license, see the file License.txt
   at the top-level directory.
*/
/*! @file sreadhb.c
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
 * Read a FLOAT PRECISION matrix stored in Harwell-Boeing format 
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
#include "slu_sdefs.h"

// 此文件的目的是读取存储在 Harwell-Boeing 格式中的矩阵数据
/*! \brief Eat up the rest of the current line */
int sDumpLine(FILE *fp)
{
    register int c;
    // 逐字符读取文件流，直到遇到换行符为止
    while ((c = fgetc(fp)) != '\n') ;
    return 0;
}

/*! \brief Parse integer format from a buffer */
int sParseIntFormat(char *buf, int *num, int *size)
{
    char *tmp;

    tmp = buf;
    // 在缓冲区中查找左括号 '('，直到找到为止
    while (*tmp++ != '(') ;
    // 使用 sscanf 从 tmp 中读取整数，存入 num 中
    sscanf(tmp, "%d", num);
    // 继续查找字符 'I' 或 'i'，直到找到为止
    while (*tmp != 'I' && *tmp != 'i') ++tmp;
    ++tmp;
    // 使用 sscanf 从 tmp 中读取整数，存入 size 中
    sscanf(tmp, "%d", size);
    return 0;
}

/*! \brief Parse float format from a buffer */
int sParseFloatFormat(char *buf, int *num, int *size)
{
    char *tmp, *period;
    
    tmp = buf;
    // 在缓冲区中查找左括号 '('，直到找到为止
    while (*tmp++ != '(') ;
    // 将 tmp 中的字符串转换为整数，存入 num 中
    *num = atoi(tmp); /*sscanf(tmp, "%d", num);*/
    // 继续查找字符 'E', 'e', 'D', 'd', 'F', 'f'，直到找到为止
    while (*tmp != 'E' && *tmp != 'e' && *tmp != 'D' && *tmp != 'd'
       && *tmp != 'F' && *tmp != 'f') {
        // 在某些情况下可能会在 'I'/'i' 之前找到 'P'/'p'，需要跳过这种情况
        if (*tmp=='p' || *tmp=='P') {
           ++tmp;
           // 再次尝试将 tmp 中的字符串转换为整数，存入 num 中
           *num = atoi(tmp); /*sscanf(tmp, "%d", num);*/
        } else {
           ++tmp;
        }
    }
    ++tmp;
    period = tmp;
    // 查找小数点 '.' 或者右括号 ')'，直到找到为止
    while (*period != '.' && *period != ')') ++period ;
    *period = '\0';
    // 将 tmp 中的字符串转换为整数，存入 size 中
    *size = atoi(tmp); /*sscanf(tmp, "%2d", size);*/

    return 0;
}

/*! \brief Read vector data from file */
static int ReadVector(FILE *fp, int_t n, int_t *where, int perline, int persize)
{
    int_t i, j, item;
    char tmp, buf[100];
    
    i = 0;
    // 逐行读取文件流
    while (i <  n) {
    fgets(buf, 100, fp);    /* read a line at a time */
    // 逐个处理每行数据
    for (j=0; j<perline && i<n; j++) {
        tmp = buf[(j+1)*persize];     /* save the char at that place */
        buf[(j+1)*persize] = 0;       /* null terminate */
        // 将字符串转换为整数，存入 where 数组中
        item = atoi(&buf[j*persize]); 
        buf[(j+1)*persize] = tmp;     /* recover the char at that place */
        where[i++] = item - 1;
    }
    }

    return 0;
}

/*! \brief Read floating-point values from file */
int sReadValues(FILE *fp, int n, float *destination, int perline, int persize)
{
    register int i, j, k, s;
    char tmp, buf[100];
    
    i = 0;
    // 逐行读取文件流
    while (i < n) {
    fgets(buf, 100, fp);    /* read a line at a time */
    // 逐个处理每行数据
    for (j=0; j<perline && i<n; j++) {
        tmp = buf[(j+1)*persize];     /* save the char at that place */
        buf[(j+1)*persize] = 0;       /* null terminate */
        s = j*persize;
        // 在 C 语言中没有 'D_' 格式，所以将 'D' 转换为 'E'
        for (k = 0; k < persize; ++k)
            if ( buf[s+k] == 'D' || buf[s+k] == 'd' ) buf[s+k] = 'E';
        // 将字符串转换为浮点数，存入 destination 数组中
        destination[i++] = atof(&buf[s]);
        buf[(j+1)*persize] = tmp;     /* recover the char at that place */
    }
    }

    return 0;
}

/*! \brief Form the full matrix from its lower part */
static void
FormFullA(int n, int_t *nonz, float **nzval, int_t **rowind, int_t **colptr)
{
    int_t i, j, k, col, new_nnz;
    int_t *t_rowind, *t_colptr, *al_rowind, *al_colptr, *a_rowind, *a_colptr;
    int_t *marker;
    float *t_val, *al_val, *a_val;

    al_rowind = *rowind;
    al_colptr = *colptr;
    al_val = *nzval;

    // 分配空间给 marker 数组
    if ( !(marker = intMalloc( (n+1) ) ) )
    // 报错并中止程序，显示内存分配失败的错误信息
    ABORT("SUPERLU_MALLOC fails for marker[]");
    
    // 分配长度为 n+1 的整型数组 t_colptr，用于存储列指针
    if ( !(t_colptr = intMalloc( (n+1) ) ) )
        ABORT("SUPERLU_MALLOC t_colptr[]");
    
    // 分配长度为 *nonz 的整型数组 t_rowind，用于存储行索引
    if ( !(t_rowind = intMalloc( *nonz ) ) )
        ABORT("SUPERLU_MALLOC fails for t_rowind[]");
    
    // 分配长度为 *nonz * sizeof(float) 的浮点数组 t_val，用于存储数值
    if ( !(t_val = (float*) SUPERLU_MALLOC( *nonz * sizeof(float)) ) )
        ABORT("SUPERLU_MALLOC fails for t_val[]");

    /* 获取 T 的每一列的元素数量，并设置列指针 */
    for (i = 0; i < n; ++i) marker[i] = 0;  // 初始化 marker 数组为 0
    for (j = 0; j < n; ++j) {  // 遍历原始矩阵 al 的每一列
        for (i = al_colptr[j]; i < al_colptr[j+1]; ++i)  // 遍历 al 的第 j 列的所有元素
            ++marker[al_rowind[i]];  // 统计每个列的元素个数
    }
    t_colptr[0] = 0;  // 初始化 t_colptr 的第一个元素为 0
    for (i = 0; i < n; ++i) {
        t_colptr[i+1] = t_colptr[i] + marker[i];  // 设置 t_colptr 的每个元素，表示每列开始位置
        marker[i] = t_colptr[i];  // 重置 marker 数组，用于后续的数据填充
    }

    /* 将矩阵 A 转置到 T */
    for (j = 0; j < n; ++j) {  // 遍历矩阵 A 的每一列
        for (i = al_colptr[j]; i < al_colptr[j+1]; ++i) {  // 遍历 A 的第 j 列的所有元素
            col = al_rowind[i];  // 获取当前元素的行索引
            t_rowind[marker[col]] = j;  // 将 A 的第 j 列元素转置到 T 的行 col，根据 marker[col] 确定位置
            t_val[marker[col]] = al_val[i];  // 复制对应的数值
            ++marker[col];  // 移动 marker 指针到下一个位置
        }
    }

    new_nnz = *nonz * 2 - n;  // 计算新的非零元素数量
    if ( !(a_colptr = intMalloc(n+1) ) )  // 分配长度为 n+1 的整型数组 a_colptr，用于新矩阵 A
        ABORT("SUPERLU_MALLOC a_colptr[]");

    if ( !(a_rowind = intMalloc( new_nnz ) ) )  // 分配长度为 new_nnz 的整型数组 a_rowind，用于新矩阵 A
        ABORT("SUPERLU_MALLOC fails for a_rowind[]");

    if ( !(a_val = (float*) SUPERLU_MALLOC( new_nnz * sizeof(float)) ) )  // 分配长度为 new_nnz 的浮点数组 a_val，用于新矩阵 A
        ABORT("SUPERLU_MALLOC fails for a_val[]");
    
    a_colptr[0] = 0;  // 初始化 a_colptr 的第一个元素为 0
    k = 0;  // 初始化索引 k

    for (j = 0; j < n; ++j) {  // 遍历矩阵 T 的每一列
        for (i = t_colptr[j]; i < t_colptr[j+1]; ++i) {  // 遍历 T 的第 j 列的所有元素
            if ( t_rowind[i] != j ) { /* 非对角线元素 */
                a_rowind[k] = t_rowind[i];  // 将 T 的非对角线元素行索引复制到新矩阵 A 的行索引
                a_val[k] = t_val[i];  // 复制对应的数值
                if ( fabs(a_val[k]) < 4.047e-300 )  // 检查数值是否小于指定阈值
                    printf("%5d: %e\n", (int)k, a_val[k]);  // 打印该数值
                ++k;  // 移动索引 k 到下一个位置
            }
        }

        for (i = al_colptr[j]; i < al_colptr[j+1]; ++i) {  // 遍历矩阵 A 的每一列
            a_rowind[k] = al_rowind[i];  // 将 A 的行索引复制到新矩阵 A 的行索引
            a_val[k] = al_val[i];  // 复制对应的数值
            ++k;  // 移动索引 k 到下一个位置
        }
    }
#ifdef DEBUG
    // 如果定义了 DEBUG 宏，则进行下面的调试输出
    if ( fabs(a_val[k]) < 4.047e-300 )
        // 如果数组 a_val[k] 的绝对值小于 4.047e-300，则打印该值的调试信息
        printf("%5d: %e\n", (int)k, a_val[k]);
#endif
    // 增加 k 的计数，进入下一个循环
    ++k;
      }
      
      // 将当前列的非零元素数量记录到 a_colptr 中
      a_colptr[j+1] = k;
    }

    // 打印新的非零元素数量到标准输出
    printf("FormFullA: new_nnz = %lld\n", (long long) new_nnz);

    // 释放临时数组的内存空间
    SUPERLU_FREE(al_val);
    SUPERLU_FREE(al_rowind);
    SUPERLU_FREE(al_colptr);
    SUPERLU_FREE(marker);
    SUPERLU_FREE(t_val);
    SUPERLU_FREE(t_rowind);
    SUPERLU_FREE(t_colptr);

    // 将结果指针更新到原始数组中
    *nzval = a_val;
    *rowind = a_rowind;
    *colptr = a_colptr;
    *nonz = new_nnz;
}

void
sreadhb(FILE *fp, int *nrow, int *ncol, int_t *nonz,
    float **nzval, int_t **rowind, int_t **colptr)
{

    register int i, numer_lines = 0, rhscrd = 0;
    int tmp, colnum, colsize, rownum, rowsize, valnum, valsize;
    char buf[100], type[4];
    int sym;

    /* Line 1 */
    // 从文件中读取一行，输出到标准输出
    fgets(buf, 100, fp);
    fputs(buf, stdout);

    /* Line 2 */
    // 循环读取并解析前5个字符，获取数值
    for (i=0; i<5; i++) {
    fscanf(fp, "%14c", buf); buf[14] = 0;
    sscanf(buf, "%d", &tmp);
    if (i == 3) numer_lines = tmp;
    if (i == 4 && tmp) rhscrd = tmp;
    }
    // 调用 sDumpLine 函数，跳过文件中当前行的剩余内容
    sDumpLine(fp);

    /* Line 3 */
    // 读取并解析文件中的数据类型
    fscanf(fp, "%3c", type);
    fscanf(fp, "%11c", buf); /* pad */
    type[3] = 0;
#if ( DEBUGlevel>=1 )
    // 如果定义了 DEBUGlevel 宏且值大于等于1，则打印矩阵类型信息
    printf("Matrix type %s\n", type);
#endif
    
    // 依次读取矩阵的行数、列数、非零元素数量以及其他值，并将其存储在相应的变量中
    fscanf(fp, "%14c", buf); *nrow = atoi(buf);
    fscanf(fp, "%14c", buf); *ncol = atoi(buf);
    fscanf(fp, "%14c", buf); *nonz = atoi(buf);
    fscanf(fp, "%14c", buf); tmp = atoi(buf);
    
    // 如果 tmp 不为0，则打印警告信息
    if (tmp != 0)
      printf("This is not an assembled matrix!\n");
    // 如果行数不等于列数，则打印警告信息
    if (*nrow != *ncol)
    printf("Matrix is not square.\n");
    // 调用 sDumpLine 函数，跳过文件中当前行的剩余内容
    sDumpLine(fp);

    /* Allocate storage for the three arrays ( nzval, rowind, colptr ) */
    // 分配三个数组（nzval、rowind、colptr）的存储空间
    sallocateA(*ncol, *nonz, nzval, rowind, colptr);

    /* Line 4: format statement */
    // 读取并解析格式语句，确定列号、行号和数值的格式
    fscanf(fp, "%16c", buf);
    sParseIntFormat(buf, &colnum, &colsize);
    fscanf(fp, "%16c", buf);
    sParseIntFormat(buf, &rownum, &rowsize);
    fscanf(fp, "%20c", buf);
    sParseFloatFormat(buf, &valnum, &valsize);
    fscanf(fp, "%20c", buf);
    // 调用 sDumpLine 函数，跳过文件中当前行的剩余内容
    sDumpLine(fp);

    /* Line 5: right-hand side */    
    // 如果存在右手边信息，则跳过该部分内容
    if ( rhscrd ) sDumpLine(fp); /* skip RHSFMT */
    
#ifdef DEBUG
    // 如果定义了 DEBUG 宏，则打印调试信息
    printf("%d rows, %lld nonzeros\n", *nrow, (long long) *nonz);
    printf("colnum %d, colsize %d\n", colnum, colsize);
    printf("rownum %d, rowsize %d\n", rownum, rowsize);
    printf("valnum %d, valsize %d\n", valnum, valsize);
#endif
    
    // 读取列指针数组
    ReadVector(fp, *ncol+1, *colptr, colnum, colsize);
    // 读取行索引数组
    ReadVector(fp, *nonz, *rowind, rownum, rowsize);
    // 如果存在数值信息行，则读取数值数组
    if ( numer_lines ) {
        sReadValues(fp, *nonz, *nzval, valnum, valsize);
    }
    
    // 判断矩阵类型是否对称，并在对称时调用 FormFullA 函数重新组织矩阵存储结构
    sym = (type[1] == 'S' || type[1] == 's');
    if ( sym ) {
    FormFullA(*ncol, nonz, nzval, rowind, colptr);
    }

    // 关闭文件
    fclose(fp);
}
```