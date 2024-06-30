# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\supermatrix.h`

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
 * \brief Matrix type definitions
 */

#ifndef __SUPERLU_SUPERMATRIX /* allow multiple inclusions */
#define __SUPERLU_SUPERMATRIX

/********************************************
 * The matrix types are defined as follows. *
 ********************************************/

typedef enum {
    SLU_NC,    /* column-wise, no supernode */
    SLU_NCP,   /* column-wise, column-permuted, no supernode 
                  (The consecutive columns of nonzeros, after permutation,
                  may not be stored  contiguously.) */
    SLU_NR,    /* row-wize, no supernode */
    SLU_SC,    /* column-wise, supernode */
    SLU_SCP,   /* supernode, column-wise, permuted */    
    SLU_SR,    /* row-wise, supernode */
    SLU_DN,     /* Fortran style column-wise storage for dense matrix */
    SLU_NR_loc  /* distributed compressed row format  */ 
} Stype_t;

typedef enum {
    SLU_S,     /* single */
    SLU_D,     /* double */
    SLU_C,     /* single complex */
    SLU_Z      /* double complex */
} Dtype_t;

typedef enum {
    SLU_GE,    /* general */
    SLU_TRLU,  /* lower triangular, unit diagonal */
    SLU_TRUU,  /* upper triangular, unit diagonal */
    SLU_TRL,   /* lower triangular */
    SLU_TRU,   /* upper triangular */
    SLU_SYL,   /* symmetric, store lower half */
    SLU_SYU,   /* symmetric, store upper half */
    SLU_HEL,   /* Hermitian, store lower half */
    SLU_HEU    /* Hermitian, store upper half */
} Mtype_t;

typedef struct {
    Stype_t Stype; /* Storage type: interprets the storage structure 
                      pointed to by *Store. */
    Dtype_t Dtype; /* Data type. */
    Mtype_t Mtype; /* Matrix type: describes the mathematical property of 
                      the matrix. */
    int_t  nrow;   /* number of rows */
    int_t  ncol;   /* number of columns */
    void *Store;   /* pointer to the actual storage of the matrix */
} SuperMatrix;

/***********************************************
 * The storage schemes are defined as follows. *
 ***********************************************/

/* Stype == SLU_NC (Also known as Harwell-Boeing sparse matrix format) */
typedef struct {
    int_t  nnz;        /* number of nonzeros in the matrix */
    void *nzval;    /* pointer to array of nonzero values, packed by column */
    int_t  *rowind; /* pointer to array of row indices of the nonzeros */
    int_t  *colptr; /* pointer to array of beginning of columns in nzval[] 
                       and rowind[]  */
                    /* Note:
                       Zero-based indexing is used;
                       colptr[] has ncol+1 entries, the last one pointing
                       beyond the last column, so that colptr[ncol] = nnz. */
} NCformat;
/* Stype == SLU_NR */
/* 结构体定义：稀疏矩阵的非零元素存储格式，无行压缩 */
typedef struct {
    int_t  nnz;        /* 非零元素的数量 */
    void *nzval;       /* 指向非零值数组的指针，按行存储 */
    int_t  *colind;    /* 指向非零元素列索引数组的指针 */
    int_t  *rowptr;    /* 指向行起始位置的指针，用于定位nzval[]和colind[] */
                        /* 注意：
                           使用零起始索引；
                           rowptr[]有nrow+1个条目，最后一个指向超出最后一行的位置，即rowptr[nrow] = nnz。 */
} NRformat;

/* Stype == SLU_SC */
/* 结构体定义：稀疏矩阵的非零元素存储格式，超级节点列压缩 */
typedef struct {
    int_t  nnz;            /* 非零元素的数量 */
    int_t  nsuper;         /* 超级节点的数量减1 */
    void *nzval;           /* 指向非零值数组的指针，按列存储 */
    int_t *nzval_colptr;   /* 指向列起始位置的指针 */
    int_t *rowind;         /* 指向压缩的行索引数组的指针，用于矩形超级节点 */
    int_t *rowind_colptr;  /* 指向行起始位置的指针 */
    int   *col_to_sup;     /* col_to_sup[j]表示列j所属的超级节点编号；从列到超级节点的映射 */
    int   *sup_to_col;     /* sup_to_col[s]指向第s个超级节点的起始列；从超级节点到列的映射
                               例如：col_to_sup: 0 1 2 2 3 3 3 4 4 4 4 4 4 (ncol=12)
                                     sup_to_col: 0 1 2 4 7 12           (nsuper=4) */
                            /* 注意：
                               使用零起始索引；
                               nzval_colptr[]、rowind_colptr[]、col_to_sup和sup_to_col[]有ncol+1个条目，
                               最后一个指向超出最后一列。
                               对于col_to_sup[]，只定义了前ncol个条目。
                               对于sup_to_col[]，只定义了前nsuper+2个条目。 */
} SCformat;

/* Stype == SLU_SCP */
/* 定义结构体 SCPformat，用于存储稀疏矩阵的压缩存储格式 */

typedef struct {
  int_t  nnz;         /* 矩阵中非零元素的数量 */
  int_t  nsuper;     /* 超节点的数量 */
  void *nzval;       /* 指向按列打包的非零值数组的指针 */
  int_t  *nzval_colbeg;/* nzval_colbeg[j] 指向 nzval[] 中第 j 列的起始位置 */
  int_t  *nzval_colend;/* nzval_colend[j] 指向 nzval[] 中第 j 列的末尾位置的下一个元素 */
  int_t  *rowind;      /* 指向压缩行索引数组的指针，用于存储矩形超节点的行索引 */
  int_t *rowind_colbeg;/* rowind_colbeg[j] 指向 rowind[] 中第 j 列的起始位置 */
  int_t *rowind_colend;/* rowind_colend[j] 指向 rowind[] 中第 j 列的末尾位置的下一个元素 */
  int   *col_to_sup;   /* col_to_sup[j] 表示第 j 列属于的超节点编号；从列到超节点的映射 */
  int   *sup_to_colbeg; /* sup_to_colbeg[s] 指向第 s 个超节点的起始列；从超节点到列的映射 */
  int   *sup_to_colend; /* sup_to_colend[s] 指向第 s 个超节点的末尾列的下一个位置；从超节点到列的映射 */
                     /* 例如：col_to_sup: 0 1 2 2 3 3 3 4 4 4 4 4 4 (ncol=12)
                              sup_to_colbeg: 0 1 2 4 7              (nsuper=4)
                              sup_to_colend: 1 2 4 7 12                    */
                     /* 注意：
                        使用零起始索引；
                        nzval_colptr[]、rowind_colptr[]、col_to_sup 和 sup_to_col[] 共有 ncol+1 个条目，
                        最后一个条目指向超出最后一列的位置。 */
} SCPformat;

/* Stype == SLU_NCP */
/* 定义结构体 NCPformat，用于存储非压缩列存储格式的矩阵信息 */

typedef struct {
    int_t nnz;      /* 矩阵中非零元素的数量 */
    void *nzval;  /* 指向按列打包的非零值数组的指针 */
    int_t *rowind;/* 指向非零元素行索引数组的指针 */
          /* 注意：nzval[] 和 rowind[] 总是具有相同的长度 */
    int_t *colbeg;/* colbeg[j] 指向 nzval[] 和 rowind[] 中第 j 列的起始位置 */
    int_t *colend;/* colend[j] 指向 nzval[] 和 rowind[] 中第 j 列的末尾位置的下一个元素 */
          /* 注意：
             使用零起始索引；
             非零元素的连续列在存储中可能不是连续的，因为矩阵已经通过列置换矩阵进行了后乘。 */
} NCPformat;

/* Stype == SLU_DN */
/* 定义结构体 DNformat，用于存储密集矩阵的信息 */

typedef struct {
    int_t lda;    /* 主导维度 */
    void *nzval;  /* 大小为 lda*ncol 的数组，用于表示密集矩阵 */
} DNformat;

/* Stype == SLU_NR_loc (分布式压缩行格式) */
/* 定义结构体，用于存储分布式压缩行格式矩阵的局部子矩阵信息 */

typedef struct {
    int_t nnz_loc;   /* 局部子矩阵中的非零元素数量 */
    int_t m_loc;     /* 属于本地处理器的行数 */
    int_t fst_row;   /* 全局第一行的索引 */
    void  *nzval;    /* 指向按行打包的非零值数组的指针 */
    int_t *rowptr;   /* 指向 nzval[] 和 colind[] 中每行起始位置的数组的指针 */
    int_t *colind;   /* 指向非零值对应的列索引数组的指针 */
                     /* 注意：
                        使用零起始索引；
                        rowptr[] 共有 n_loc + 1 个条目，最后一个条目指向超出最后一行的位置，
                        因此 rowptr[n_loc] = nnz_loc。*/
/* 结构体定义：用于在2D进程网格的第0层存储3D矩阵数据
   只有网格0包含这些数据结构的有效值。 */
typedef struct NRformat_loc3d
{
    NRformat_loc *A_nfmt; // 在2D网格0上收集的A矩阵
    void *B3d;  // 在整个3D进程网格上的数据
    int  ldb;   // 相对于3D进程网格的位置
    int nrhs;   // 右侧向量数量
    int m_loc;  // 相对于3D进程网格的位置
    void *B2d;  // 在2D进程层网格-0上的数据

    int *row_counts_int; // 这些计数存储在2D层网格-0上，
                         // 但计算沿Z维度的{A, B}行数
    int *row_disp;       // 行位移，用于索引计数数组
    int *nnz_counts_int; // 非零元素计数，存储在2D层网格-0上
    int *nnz_disp;       // 非零元素位移，用于索引计数数组
    int *b_counts_int;   // B矩阵行数计数，存储在2D层网格-0上
    int *b_disp;         // B矩阵行数位移，用于索引计数数组

    /* 以下4个结构用于将解X从2D网格-0散射回3D进程 */
    int num_procs_to_send;     // 发送解的进程数量
    int *procs_to_send_list;   // 要发送的进程列表
    int *send_count_list;      // 发送计数列表
    int num_procs_to_recv;     // 接收解的进程数量
    int *procs_recv_from_list; // 来自哪些进程接收解的列表
    int *recv_count_list;      // 接收计数列表
} NRformat_loc3d;

#endif  /* __SUPERLU_SUPERMATRIX */
```