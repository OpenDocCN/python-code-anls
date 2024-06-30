# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dreadtriple.c`

```
/*
 * 读取以三元组（坐标）格式存储的矩阵
 */

#include "slu_ddefs.h"  // 包含SLU库的定义文件

void
dreadtriple(int *m, int *n, int_t *nonz,
        double **nzval, int_t **rowind, int_t **colptr)
{
/*
 * 输出参数
 * =================
 *   (a, asub, xa): asub[*] 包含矩阵 A 中每列非零元素的行下标；
 *                  a[*] 包含数值；
 *                  矩阵 A 的第 i 行通过 a[k], k = xa[i], ..., xa[i+1]-1 给出。
 *
 */
    int    j, k, jsize, nnz, nz;
    double *a, *val;  // 数值数组
    int_t  *asub, *xa;  // 行下标数组、列指针数组
    int    *row, *col;  // 行号数组、列号数组
    int    zero_base = 0;  // 是否使用0为基础的索引

    /*  矩阵格式:
     *    第一行:  #行数, #列数, #非零元素数
     *    后续行为三元组格式:
     *             行号, 列号, 值
     */

#ifdef _LONGINT
    scanf("%d%lld", n, nonz);  // 读取行数、非零元素数（可能使用长整型）
#else
    scanf("%d%d", n, nonz);  // 读取行数、非零元素数
#endif    
    *m = *n;  // 设置行数为列数
    printf("m %d, n %d, nonz %ld\n", *m, *n, (long) *nonz);  // 输出行数、列数、非零元素数
    dallocateA(*n, *nonz, nzval, rowind, colptr); /* 分配存储空间 */
    a    = *nzval;  // 数值存储空间
    asub = *rowind;  // 行下标存储空间
    xa   = *colptr;  // 列指针存储空间

    val = (double *) SUPERLU_MALLOC(*nonz * sizeof(double));  // 分配数值数组空间
    row = int32Malloc(*nonz);  // 分配行号数组空间
    col = int32Malloc(*nonz);  // 分配列号数组空间

    for (j = 0; j < *n; ++j) xa[j] = 0;  // 初始化列指针为0

    /* 从文件中读取三元组数据 */
    for (nnz = 0, nz = 0; nnz < *nonz; ++nnz) {
    
    scanf("%d%d%lf\n", &row[nz], &col[nz], &val[nz]);  // 读取行号、列号、数值

        if ( nnz == 0 ) { /* 第一个非零元素 */
        if ( row[0] == 0 || col[0] == 0 ) {
        zero_base = 1;  // 使用0为基础的索引
        printf("triplet file: row/col indices are zero-based.\n");
        } else
        printf("triplet file: row/col indices are one-based.\n");
        }

        if ( !zero_base ) { 
       /* 转换为基于0的索引 */
      --row[nz];
      --col[nz];
        }

    if (row[nz] < 0 || row[nz] >= *m || col[nz] < 0 || col[nz] >= *n
        /*|| val[nz] == 0.*/) {
        fprintf(stderr, "nz %d, (%d, %d) = %e out of bound, removed\n",
            nz, row[nz], col[nz], val[nz]);
        exit(-1);
    } else {
        ++xa[col[nz]];  // 增加列指针
        ++nz;
    }
    }

    *nonz = nz;  // 更新非零元素数

    /* 初始化列指针数组 */
    k = 0;
    jsize = xa[0];
    xa[0] = 0;
    for (j = 1; j < *n; ++j) {
    k += jsize;
    jsize = xa[j];
    xa[j] = k;
    }
    
    /* 将三元组复制到列向量存储中 */
    for (nz = 0; nz < *nonz; ++nz) {
    j = col[nz];
    k = xa[j];
    asub[k] = row[nz];
    a[k] = val[nz];
    ++xa[j];
    }

    /* 将列指针重置为各列的起始位置 */
    for (j = *n; j > 0; --j)
    # 将数组 xa 中索引 j 处的元素赋值为数组 xa 中索引 j-1 处的元素的值
    xa[j] = xa[j-1];

    # 将数组 xa 中索引 0 处的元素赋值为 0
    xa[0] = 0;

    # 释放动态分配的数组 val 所占用的内存
    SUPERLU_FREE(val);

    # 释放动态分配的数组 row 所占用的内存
    SUPERLU_FREE(row);

    # 释放动态分配的数组 col 所占用的内存
    SUPERLU_FREE(col);
#ifdef CHK_INPUT
    {
    // 如果定义了 CHK_INPUT 宏，则执行以下代码块
    int i;
    for (i = 0; i < *n; i++) {
        // 打印当前列数 i 和 xa[i] 的值
        printf("Col %d, xa %d\n", i, xa[i]);
        // 遍历从 xa[i] 开始到 xa[i+1] 结束的索引范围
        for (k = xa[i]; k < xa[i+1]; k++)
            // 打印 asub[k] 和 a[k] 的值，使用制表符分隔，a[k] 格式为浮点数
            printf("%d\t%16.10f\n", asub[k], a[k]);
    }
    }
#endif

}


void dreadrhs(int m, double *b)
{
    // 打开文件 "b.dat" 以只读方式
    FILE *fp = fopen("b.dat", "r");
    int i;

    if ( !fp ) {
        // 如果文件打开失败，输出错误信息到标准错误流，并退出程序
        fprintf(stderr, "dreadrhs: file does not exist\n");
        exit(-1);
    }
    // 逐行读取文件内容，将读取到的双精度浮点数保存到数组 b 中
    for (i = 0; i < m; ++i)
        fscanf(fp, "%lf\n", &b[i]);

    // 关闭文件流
    fclose(fp);
}
```