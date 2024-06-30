# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dreadhb.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/



# 此处是文件级注释，声明了版权信息和许可证信息
/*! @file dreadhb.c
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
 * Read a DOUBLE PRECISION matrix stored in Harwell-Boeing format 
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
#include "slu_ddefs.h"
/*! \brief Eat up the rest of the current line */
int dDumpLine(FILE *fp)
{
    register int c;
    // 读取文件流直到换行符，用于跳过当前行剩余内容
    while ((c = fgetc(fp)) != '\n') ;
    return 0;
}

/*! \brief Parse integer format from a string buffer */
int dParseIntFormat(char *buf, int *num, int *size)
{
    char *tmp;

    tmp = buf;
    // 跳过直到遇到 '('
    while (*tmp++ != '(') ;
    // 从字符串中解析出一个整数并存储在 num 中
    sscanf(tmp, "%d", num);
    // 继续跳过直到遇到 'I' 或 'i'
    while (*tmp != 'I' && *tmp != 'i') ++tmp;
    ++tmp;
    // 从字符串中解析出一个整数并存储在 size 中
    sscanf(tmp, "%d", size);
    return 0;
}

/*! \brief Parse floating-point format from a string buffer */
int dParseFloatFormat(char *buf, int *num, int *size)
{
    char *tmp, *period;
    
    tmp = buf;
    // 跳过直到遇到 '('
    while (*tmp++ != '(') ;
    // 解析出一个整数并存储在 num 中
    *num = atoi(tmp);
    // 继续跳过直到遇到 'E', 'e', 'D', 'd', 'F', 'f' 中的一个
    // 如果遇到 'p' 或 'P'，则跳过，并再次解析整数存储在 num 中
    while (*tmp != 'E' && *tmp != 'e' && *tmp != 'D' && *tmp != 'd'
       && *tmp != 'F' && *tmp != 'f') {
        if (*tmp=='p' || *tmp=='P') {
           ++tmp;
           *num = atoi(tmp);
        } else {
           ++tmp;
        }
    }
    ++tmp;
    // 找到小数点或者 ')'，并将其替换为字符串结束符 '\0'
    period = tmp;
    while (*period != '.' && *period != ')') ++period ;
    *period = '\0';
    // 解析出一个整数并存储在 size 中
    *size = atoi(tmp);

    return 0;
}

/*! \brief Read a vector from a file */
static int ReadVector(FILE *fp, int_t n, int_t *where, int perline, int persize)
{
    int_t i, j, item;
    char tmp, buf[100];
    
    i = 0;
    // 逐行读取文件，每次读取一行到 buf 中
    while (i <  n) {
        fgets(buf, 100, fp);    /* read a line at a time */
        // 每行中逐个读取 perline 个数值，并存储在 where 数组中
        for (j=0; j<perline && i<n; j++) {
            tmp = buf[(j+1)*persize];     /* save the char at that place */
            buf[(j+1)*persize] = 0;       /* null terminate */
            // 将字符串转换为整数并存储在 where 数组中
            item = atoi(&buf[j*persize]); 
            buf[(j+1)*persize] = tmp;     /* recover the char at that place */
            where[i++] = item - 1; // 存储在 where 数组中，索引从 0 开始
        }
    }

    return 0;
}

/*! \brief Read double values from a file */
int dReadValues(FILE *fp, int n, double *destination, int perline, int persize)
{
    register int i, j, k, s;
    char tmp, buf[100];
    
    i = 0;
    // 逐行读取文件，每次读取一行到 buf 中
    while (i < n) {
        fgets(buf, 100, fp);    /* read a line at a time */
        // 每行中逐个读取 perline 个数值，并存储在 destination 数组中
        for (j=0; j<perline && i<n; j++) {
            tmp = buf[(j+1)*persize];     /* save the char at that place */
            buf[(j+1)*persize] = 0;       /* null terminate */
            s = j*persize;
            // 替换 'D' 为 'E'，然后将字符串转换为 double 并存储在 destination 数组中
            for (k = 0; k < persize; ++k)
                if ( buf[s+k] == 'D' || buf[s+k] == 'd' ) buf[s+k] = 'E';
            destination[i++] = atof(&buf[s]);
            buf[(j+1)*persize] = tmp;     /* recover the char at that place */
        }
    }

    return 0;
}

/*! \brief Form a full matrix from its lower triangular part */
static void
FormFullA(int n, int_t *nonz, double **nzval, int_t **rowind, int_t **colptr)
{
    int_t i, j, k, col, new_nnz;
    int_t *t_rowind, *t_colptr, *al_rowind, *al_colptr, *a_rowind, *a_colptr;
    int_t *marker;
    double *t_val, *al_val, *a_val;

    al_rowind = *rowind;
    al_colptr = *colptr;
    al_val = *nzval;

    // 在原地修改 lower triangular matrix 的数据结构，使其变为完整的对称矩阵
    if ( !(marker = intMalloc( (n+1) ) ) )
    # 如果内存分配失败，输出错误信息并中止程序
    ABORT("SUPERLU_MALLOC fails for marker[]");
    
    # 分配内存给 t_colptr 数组，如果失败则输出错误信息并中止程序
    if ( !(t_colptr = intMalloc( (n+1) ) ) )
        ABORT("SUPERLU_MALLOC t_colptr[]");
    
    # 分配内存给 t_rowind 数组，如果失败则输出错误信息并中止程序
    if ( !(t_rowind = intMalloc( *nonz ) ) )
        ABORT("SUPERLU_MALLOC fails for t_rowind[]");
    
    # 分配内存给 t_val 数组，如果失败则输出错误信息并中止程序
    if ( !(t_val = (double*) SUPERLU_MALLOC( *nonz * sizeof(double)) ) )
        ABORT("SUPERLU_MALLOC fails for t_val[]");

    /* Get counts of each column of T, and set up column pointers */
    # 初始化 marker 数组为 0
    for (i = 0; i < n; ++i) marker[i] = 0;
    
    # 计算每列 T 的元素个数，并设置列指针
    for (j = 0; j < n; ++j) {
        for (i = al_colptr[j]; i < al_colptr[j+1]; ++i)
            ++marker[al_rowind[i]];
    }
    
    # 设置 t_colptr 数组
    t_colptr[0] = 0;
    for (i = 0; i < n; ++i) {
        t_colptr[i+1] = t_colptr[i] + marker[i];
        marker[i] = t_colptr[i];
    }

    /* Transpose matrix A to T */
    # 转置矩阵 A 到 T
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
    
    # 分配内存给 a_colptr 数组，如果失败则输出错误信息并中止程序
    if ( !(a_colptr = intMalloc(n+1) ) )
        ABORT("SUPERLU_MALLOC a_colptr[]");
    
    # 分配内存给 a_rowind 数组，如果失败则输出错误信息并中止程序
    if ( !(a_rowind = intMalloc( new_nnz ) ) )
        ABORT("SUPERLU_MALLOC fails for a_rowind[]");
    
    # 分配内存给 a_val 数组，如果失败则输出错误信息并中止程序
    if ( !(a_val = (double*) SUPERLU_MALLOC( new_nnz * sizeof(double)) ) )
        ABORT("SUPERLU_MALLOC fails for a_val[]");
    
    # 设置 a_colptr 数组
    a_colptr[0] = 0;
    k = 0;
    for (j = 0; j < n; ++j) {
        for (i = t_colptr[j]; i < t_colptr[j+1]; ++i) {
            if ( t_rowind[i] != j ) { /* 非对角线元素 */
                a_rowind[k] = t_rowind[i];
                a_val[k] = t_val[i];
                if ( fabs(a_val[k]) < 4.047e-300 )
                    printf("%5d: %e\n", (int)k, a_val[k]);
                ++k;
            }
        }
        
        for (i = al_colptr[j]; i < al_colptr[j+1]; ++i) {
            a_rowind[k] = al_rowind[i];
            a_val[k] = al_val[i];
            ++k;
        }
    }
#ifdef DEBUG
    // 如果处于调试模式，检查 a_val[k] 的绝对值是否小于极小值，然后打印出来
    if ( fabs(a_val[k]) < 4.047e-300 )
        printf("%5d: %e\n", (int)k, a_val[k]);
#endif
    // k 自增
    ++k;
      }
      
      // 将当前列的结束索引 k 存入 a_colptr 数组
      a_colptr[j+1] = k;
    }

    // 打印新的非零元素个数 new_nnz
    printf("FormFullA: new_nnz = %lld\n", (long long) new_nnz);

    // 释放临时数组的内存空间
    SUPERLU_FREE(al_val);
    SUPERLU_FREE(al_rowind);
    SUPERLU_FREE(al_colptr);
    SUPERLU_FREE(marker);
    SUPERLU_FREE(t_val);
    SUPERLU_FREE(t_rowind);
    SUPERLU_FREE(t_colptr);

    // 将结果赋值给指针变量，完成函数的数据返回
    *nzval = a_val;
    *rowind = a_rowind;
    *colptr = a_colptr;
    *nonz = new_nnz;
}

void
dreadhb(FILE *fp, int *nrow, int *ncol, int_t *nonz,
    double **nzval, int_t **rowind, int_t **colptr)
{

    register int i, numer_lines = 0, rhscrd = 0;
    int tmp, colnum, colsize, rownum, rowsize, valnum, valsize;
    char buf[100], type[4];
    int sym;

    /* Line 1 */
    // 读取文件 fp 中的一行内容到缓冲区 buf，并输出到标准输出
    fgets(buf, 100, fp);
    fputs(buf, stdout);

    /* Line 2 */
    // 依次读取五个长度为 14 的字符串到 buf，转换为整数 tmp
    for (i=0; i<5; i++) {
    fscanf(fp, "%14c", buf); buf[14] = 0;
    sscanf(buf, "%d", &tmp);
    if (i == 3) numer_lines = tmp;
    if (i == 4 && tmp) rhscrd = tmp;
    }
    dDumpLine(fp);

    /* Line 3 */
    // 读取矩阵类型到 type 数组
    fscanf(fp, "%3c", type);
    fscanf(fp, "%11c", buf); /* pad */
    type[3] = 0;
#if ( DEBUGlevel>=1 )
    // 如果调试级别大于等于 1，打印矩阵类型
    printf("Matrix type %s\n", type);
#endif
    
    // 依次读取文件中的四个数值到对应的变量，并进行验证
    fscanf(fp, "%14c", buf); *nrow = atoi(buf);
    fscanf(fp, "%14c", buf); *ncol = atoi(buf);
    fscanf(fp, "%14c", buf); *nonz = atoi(buf);
    fscanf(fp, "%14c", buf); tmp = atoi(buf);
    
    // 如果 tmp 不为零，输出提示信息
    if (tmp != 0)
      printf("This is not an assembled matrix!\n");
    // 如果行数不等于列数，输出警告信息
    if (*nrow != *ncol)
    printf("Matrix is not square.\n");
    dDumpLine(fp);

    /* Allocate storage for the three arrays ( nzval, rowind, colptr ) */
    // 为三个数组（nzval、rowind、colptr）分配存储空间
    dallocateA(*ncol, *nonz, nzval, rowind, colptr);

    /* Line 4: format statement */
    // 读取格式语句到 buf，并解析出列数、列大小等信息
    fscanf(fp, "%16c", buf);
    dParseIntFormat(buf, &colnum, &colsize);
    fscanf(fp, "%16c", buf);
    dParseIntFormat(buf, &rownum, &rowsize);
    fscanf(fp, "%20c", buf);
    dParseFloatFormat(buf, &valnum, &valsize);
    fscanf(fp, "%20c", buf);
    dDumpLine(fp);

    /* Line 5: right-hand side */    
    // 如果有右手边向量，跳过 RHSFMT
    if ( rhscrd ) dDumpLine(fp); /* skip RHSFMT */
    
#ifdef DEBUG
    // 如果处于调试模式，打印矩阵和向量相关信息
    printf("%d rows, %lld nonzeros\n", *nrow, (long long) *nonz);
    printf("colnum %d, colsize %d\n", colnum, colsize);
    printf("rownum %d, rowsize %d\n", rownum, rowsize);
    printf("valnum %d, valsize %d\n", valnum, valsize);
#endif
    
    // 读取列指针数组
    ReadVector(fp, *ncol+1, *colptr, colnum, colsize);
    // 读取行索引数组
    ReadVector(fp, *nonz, *rowind, rownum, rowsize);
    // 如果有数值，读取数值数组
    if ( numer_lines ) {
        dReadValues(fp, *nonz, *nzval, valnum, valsize);
    }
    
    // 判断矩阵是否对称，如果是，则进行完整矩阵形成
    sym = (type[1] == 'S' || type[1] == 's');
    if ( sym ) {
    FormFullA(*ncol, nonz, nzval, rowind, colptr);
    }

    // 关闭文件流 fp
    fclose(fp);
}
```