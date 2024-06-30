# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\superlu_enum_consts.h`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/** @file superlu_enum_consts.h
 * \brief enum constants header file 
 *
 * -- SuperLU routine (version 4.1) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley, 
 * October 1, 2010
 * January 28, 2018
 *
 */

#ifndef __SUPERLU_ENUM_CONSTS /* allow multiple inclusions */
#define __SUPERLU_ENUM_CONSTS

// 定义用于向后兼容的宏
#define LargeDiag_AWPM LargeDiag_HWPM  /* backward compatibility */

/***********************************************************************
 * Enumerate types
 ***********************************************************************/

// 枚举类型：是否（YES/NO）
typedef enum {NO, YES}                                          yes_no_t;
// 枚举类型：因子化过程的不同阶段
typedef enum {DOFACT, SamePattern, SamePattern_SameRowPerm, FACTORED} fact_t;
// 枚举类型：行排列的类型
typedef enum {NOROWPERM, LargeDiag_MC64, LargeDiag_HWPM, MY_PERMR} rowperm_t;
// 枚举类型：列排列的类型
typedef enum {NATURAL, MMD_ATA, MMD_AT_PLUS_A, COLAMD,
          METIS_AT_PLUS_A, PARMETIS, METIS_ATA, ZOLTAN, MY_PERMC} colperm_t;
// 枚举类型：转置类型
typedef enum {NOTRANS, TRANS, CONJ}                             trans_t;
// 枚举类型：对角线的缩放类型
typedef enum {NOEQUIL, ROW, COL, BOTH}                          DiagScale_t;
// 枚举类型：迭代修正的类型
typedef enum {NOREFINE, SLU_SINGLE=1, SLU_DOUBLE, SLU_EXTRA}    IterRefine_t;
// 枚举类型：内存类型
typedef enum {LUSUP, UCOL, LSUB, USUB, LLVL, ULVL, NO_MEMTYPE}  MemType;
//typedef enum {USUB, LSUB, UCOL, LUSUP, LLVL, ULVL, NO_MEMTYPE}  MemType;
// 枚举类型：堆栈的头部或尾部
typedef enum {HEAD, TAIL}                                       stack_end_t;
// 枚举类型：LU分解中的用户或系统空间
typedef enum {SYSTEM, USER}                                     LU_space_t;
// 枚举类型：范数类型
typedef enum {ONE_NORM, TWO_NORM, INF_NORM}            norm_t;

/*
 * The following are for ILUTP in serial SuperLU
 */
// 枚举类型：ILUTP预处理中的参数类型
typedef enum {SILU, SMILU_1, SMILU_2, SMILU_3}            milu_t;
// 枚举类型：ILUTP中的丢弃规则
typedef enum {NODROP        = 0x0000,
          DROP_BASIC    = 0x0001, /* ILU(tau) */
          DROP_PROWS    = 0x0002, /* ILUTP: keep p maximum rows */
          DROP_COLUMN    = 0x0004, /* ILUTP: for j-th column, 
                         p = gamma * nnz(A(:,j)) */
          DROP_AREA     = 0x0008, /* ILUTP: for j-th column, use
                         nnz(F(:,1:j)) / nnz(A(:,1:j))
                         to limit memory growth  */
          DROP_SECONDARY    = 0x000E, /* PROWS | COLUMN | AREA */
          DROP_DYNAMIC    = 0x0010,
          DROP_INTERP    = 0x0100}            rule_t;


/* 
 * The following enumerate type is used by the statistics variable 
 * to keep track of flop count and time spent at various stages.
 *
 * Note that not all of the fields are disjoint.
 */
// 枚举类型：用于记录各阶段浮点操作次数和时间消耗的统计类型
typedef enum {
    COLPERM, /* find a column ordering that minimizes fills */
    ROWPERM, /* find a row ordering maximizes diagonal. */
    RELAX,   /* find artificial supernodes */
    ETREE,   /* compute column etree */
    EQUIL,   /* equilibrate the original matrix */
    SYMBFAC, /* symbolic factorization. */
    DIST,    /* distribute matrix. */
    FACT,    /* perform LU factorization */
    COMM,    /* communication for factorization */
    COMM_DIAG, /* Bcast diagonal block to process column */
    COMM_RIGHT, /* communicate L panel */
    COMM_DOWN, /* communicate U panel */
    SOL_COMM,/* communication for solve */
    SOL_GEMM,/* gemm for solve */
    SOL_TRSM,/* trsm for solve */
    SOL_TOT,    /* LU-solve time*/
    RCOND,   /* estimate reciprocal condition number */
    SOLVE,   /* forward and back solves */
    REFINE,  /* perform iterative refinement */
    TRSV,    /* fraction of FACT spent in xTRSV */
    GEMV,    /* fraction of FACT spent in xGEMV */
    FERR,    /* estimate error bounds after iterative refinement */
    NPHASES  /* total number of phases */
} PhaseType;


// 结束了一个枚举定义的定义，PhaseType 是这个枚举类型的名称
#endif /* __SUPERLU_ENUM_CONSTS */
```