# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\colamd.h`

```
/*
 * /*! \file
 * Copyright (c) 2003, The Regents of the University of California, through
 * Lawrence Berkeley National Laboratory (subject to receipt of any required 
 * approvals from U.S. Dept. of Energy) 
 * 
 * All rights reserved. 
 * 
 * The source code is distributed under BSD license, see the file License.txt
 * at the top-level directory.
 */

/* ========================================================================== */
/* === colamd/symamd prototypes and definitions ============================= */
/* ========================================================================== */

/* COLAMD / SYMAMD include file
 * 
 * You must include this file (colamd.h) in any routine that uses colamd,
 * symamd, or the related macros and definitions.
 * 
 * Authors:
 * 
 * The authors of the code itself are Stefan I. Larimore and Timothy A.
 * Davis (DrTimothyAldenDavis@gmail.com).  The algorithm was
 * developed in collaboration with John Gilbert, Xerox PARC, and Esmond
 * Ng, Oak Ridge National Laboratory.
 * 
 * Acknowledgements:
 * 
 * This work was supported by the National Science Foundation, under
 * grants DMS-9504974 and DMS-9803599.
 * 
 * Notice:
 * 
 * Copyright (c) 1998-2007, Timothy A. Davis, All Rights Reserved.
 * 
 * THIS MATERIAL IS PROVIDED AS IS, WITH ABSOLUTELY NO WARRANTY
 * EXPRESSED OR IMPLIED.  ANY USE IS AT YOUR OWN RISK.
 * 
 * Permission is hereby granted to use, copy, modify, and/or distribute
 * this program, provided that the Copyright, this License, and the
 * Availability of the original version is retained on all copies and made
 * accessible to the end-user of any code or package that includes COLAMD
 * or any modified version of COLAMD. 
 * 
 * Availability:
 * 
 * The colamd/symamd library is available at http://www.suitesparse.com
 * This file is required by the colamd.c, colamdmex.c, and symamdmex.c
 * files, and by any C code that calls the routines whose prototypes are
 * listed below, or that uses the colamd/symamd definitions listed below.
 */

#ifndef COLAMD_H
#define COLAMD_H

#include "superlu_config.h"

/* make it easy for C++ programs to include COLAMD */
#ifdef __cplusplus
extern "C" {
#endif

#if defined ( _LONGINT )
#define DLONG
#endif

/* ========================================================================== */
/* === Include files ======================================================== */
/* ========================================================================== */

#include <stdlib.h>
/*#include <stdint.h>*/

/* ========================================================================== */
/* === COLAMD version ======================================================= */
/* ========================================================================== */
/* COLAMD Version 2.4 and later will include the following definitions.
 * As an example, to test if the version you are using is 2.4 or later:
 *
 * #ifdef COLAMD_VERSION
 *    if (COLAMD_VERSION >= COLAMD_VERSION_CODE (2,4)) ...
 * #endif
 *
 * This also works during compile-time:
 *
 *  #if defined(COLAMD_VERSION) && (COLAMD_VERSION >= COLAMD_VERSION_CODE (2,4))
 *    printf ("This is version 2.4 or later\n") ;
 *  #else
 *    printf ("This is an early version\n") ;
 *  #endif
 *
 * Versions 2.3 and earlier of COLAMD do not include a #define'd version number.
 */

/* 定义COLAMD的版本日期 */
#define COLAMD_DATE "Oct 10, 2014"
/* 定义COLAMD的版本号生成宏 */
#define COLAMD_VERSION_CODE(main,sub) ((main) * 1000 + (sub))
/* 定义COLAMD的主版本号 */
#define COLAMD_MAIN_VERSION 2
/* 定义COLAMD的子版本号 */
#define COLAMD_SUB_VERSION 9
/* 定义COLAMD的子子版本号 */
#define COLAMD_SUBSUB_VERSION 1
/* 计算COLAMD的版本号 */
#define COLAMD_VERSION \
    COLAMD_VERSION_CODE(COLAMD_MAIN_VERSION,COLAMD_SUB_VERSION)

/* ========================================================================== */
/* === Knob and statistics definitions ====================================== */
/* ========================================================================== */

/* knobs [0] and stats [0]: dense row knob and output statistic. */
/* 稠密行的开关和输出统计信息 */
#define COLAMD_DENSE_ROW 0

/* knobs [1] and stats [1]: dense column knob and output statistic. */
/* 稠密列的开关和输出统计信息 */
#define COLAMD_DENSE_COL 1

/* knobs [2]: aggressive absorption */
/* 激进吸收开关 */
#define COLAMD_AGGRESSIVE 2

/* stats [2]: memory defragmentation count output statistic */
/* 内存碎片整理计数的输出统计信息 */
#define COLAMD_DEFRAG_COUNT 2

/* stats [3]: colamd status:  zero OK, > 0 warning or notice, < 0 error */
/* COLAMD的状态：0表示正常，>0表示警告或提示，<0表示错误 */
#define COLAMD_STATUS 3

/* stats [4..6]: error info, or info on jumbled columns */ 
/* 错误信息或混乱列信息 */
#define COLAMD_INFO1 4
#define COLAMD_INFO2 5
#define COLAMD_INFO3 6

/* error codes returned in stats [3]: */
/* 在stats [3]中返回的错误代码 */
#define COLAMD_OK                (0)
#define COLAMD_OK_BUT_JUMBLED            (1)
#define COLAMD_ERROR_A_not_present        (-1)
#define COLAMD_ERROR_p_not_present        (-2)
#define COLAMD_ERROR_nrow_negative        (-3)
#define COLAMD_ERROR_ncol_negative        (-4)
#define COLAMD_ERROR_nnz_negative        (-
#define SuiteSparse_long_max _I64_MAX
#define SuiteSparse_long_idd "I64d"

定义了 `SuiteSparse_long_max` 为 `_I64_MAX`，`SuiteSparse_long_idd` 为 `"I64d"`，这些定义通常用于指定 SuiteSparse 库中长整型 (`SuiteSparse_long`) 的最大值和格式字符串。


#else

#if 0    /* commented out by Sherry */
#define SuiteSparse_long long
#define SuiteSparse_long_max LONG_MAX
#define SuiteSparse_long_idd "ld"
#endif

#if 1
#define SuiteSparse_long long long int
#else
#define SuiteSparse_long int64_t
#endif
#define SuiteSparse_long_max LONG_MAX
#define SuiteSparse_long_idd "lld"

#endif
#define SuiteSparse_long_id "%" SuiteSparse_long_idd
#endif

根据条件设置了 `SuiteSparse_long` 类型和相关的宏定义。根据注释，这段代码似乎用于在不同情况下选择不同的长整型定义和格式化字符串。


/* ========================================================================== */
/* === int or SuiteSparse_long ============================================== */
/* ========================================================================== */

#ifdef DLONG

#define Int SuiteSparse_long
#define ID  SuiteSparse_long_id
#define Int_MAX SuiteSparse_long_max

#define COLAMD_recommended colamd_l_recommended
#define COLAMD_set_defaults colamd_l_set_defaults
#define COLAMD_MAIN colamd_l
#define SYMAMD_MAIN symamd_l
#define COLAMD_report colamd_l_report
#define SYMAMD_report symamd_l_report

#else

#define Int int
#define ID "%d"
#define Int_MAX INT_MAX

#define COLAMD_recommended colamd_recommended
#define COLAMD_set_defaults colamd_set_defaults
#define COLAMD_MAIN colamd
#define SYMAMD_MAIN symamd
#define COLAMD_report colamd_report
#define SYMAMD_report symamd_report

#endif

根据预处理器定义 `DLONG`，选择将 `Int` 定义为 `SuiteSparse_long` 或 `int`，以及相关的宏定义，用于在代码中统一整数类型和相关函数调用。


size_t colamd_recommended    /* returns recommended value of Alen, */
                /* or 0 if input arguments are erroneous */
(
    int nnz,            /* nonzeros in A */
    int n_row,            /* number of rows in A */
    int n_col            /* number of columns in A */
) ;

声明了函数 `colamd_recommended`，该函数返回推荐的 `Alen` 值，或者在输入参数错误时返回 0。函数参数包括非零元素数 `nnz`，矩阵行数 `n_row`，以及矩阵列数 `n_col`。


size_t colamd_l_recommended    /* returns recommended value of Alen, */
                /* or 0 if input arguments are erroneous */
(
    SuiteSparse_long nnz,       /* nonzeros in A */
    SuiteSparse_long n_row,     /* number of rows in A */
    SuiteSparse_long n_col      /* number of columns in A */
) ;

声明了 `colamd_l_recommended` 函数，类似于 `colamd_recommended`，但是参数和返回值类型是 `SuiteSparse_long` 类型，用于处理 SuiteSparse 长整型。


void colamd_set_defaults    /* sets default parameters */
(                /* knobs argument is modified on output */
    double knobs [COLAMD_KNOBS]    /* parameter settings for colamd */
) ;

声明了 `colamd_set_defaults` 函数，该函数设置 `colamd` 的默认参数，并修改输出的 `knobs` 参数数组。


void colamd_l_set_defaults    /* sets default parameters */
(                /* knobs argument is modified on output */
    double knobs [COLAMD_KNOBS]    /* parameter settings for colamd */
) ;

声明了 `colamd_l_set_defaults` 函数，类似于 `colamd_set_defaults`，但是处理 SuiteSparse 长整型参数。


int colamd            /* returns (1) if successful, (0) otherwise*/
(                /* A and p arguments are modified on output */
    int n_row,            /* number of rows in A */
    int n_col,            /* number of columns in A */
    int Alen,            /* size of the array A */
    int A [],            /* row indices of A, of size Alen */
    int p [],            /* column pointers of A, of size n_col+1 */
    double knobs [COLAMD_KNOBS],/* parameter settings for colamd */
    int stats [COLAMD_STATS]    /* colamd output statistics and error codes */
) ;

声明了 `colamd` 函数，该函数返回整数表示成功与否，修改输入的 `A` 和 `p` 参数数组，参数包括矩阵行数 `n_row`，列数 `n_col`，数组大小 `Alen`，以及相关的参数设置和统计信息。

以上是对给定代码的详细注释。
SuiteSparse_long colamd_l       /* 返回 (1) 如果成功，(0) 否则 */
(                /* A 和 p 参数在输出时被修改 */
    SuiteSparse_long n_row,     /* A 的行数 */
    SuiteSparse_long n_col,     /* A 的列数 */
    SuiteSparse_long Alen,      /* A 数组的大小 */
    SuiteSparse_long A [],      /* A 的行索引，大小为 Alen */
    SuiteSparse_long p [],      /* A 的列指针，大小为 n_col+1 */
    double knobs [COLAMD_KNOBS],/* colamd 的参数设置 */
    SuiteSparse_long stats [COLAMD_STATS]   /* colamd 输出的统计信息
                                             * 和错误代码 */
) ;

int symamd                /* 如果成功返回 (1)，否则返回 (0) */
(
    int n,                /* A 的行和列数 */
    int A [],                /* A 的行索引 */
    int p [],                /* A 的列指针 */
    int perm [],            /* 输出的排列，大小为 n_col+1 */
    double knobs [COLAMD_KNOBS],    /* 参数设置（如果为 NULL 则使用默认值） */
    int stats [COLAMD_STATS],        /* 输出的统计信息和错误代码 */
    void * (*allocate) (size_t, size_t),
                        /* 指向 calloc（ANSI C）或
                           mxCalloc（用于 MATLAB mexFunction）的指针 */
    void (*release) (void *)
                        /* 指向 free（ANSI C）或
                           mxFree（用于 MATLAB mexFunction）的指针 */
) ;

SuiteSparse_long symamd_l               /* 如果成功返回 (1)，否则返回 (0) */
(
    SuiteSparse_long n,                 /* A 的行和列数 */
    SuiteSparse_long A [],              /* A 的行索引 */
    SuiteSparse_long p [],              /* A 的列指针 */
    SuiteSparse_long perm [],           /* 输出的排列，大小为 n_col+1 */
    double knobs [COLAMD_KNOBS],    /* 参数设置（如果为 NULL 则使用默认值） */
    SuiteSparse_long stats [COLAMD_STATS],  /* 输出的统计信息和错误代码 */
    void * (*allocate) (size_t, size_t),
                        /* 指向 calloc（ANSI C）或
                           mxCalloc（用于 MATLAB mexFunction）的指针 */
    void (*release) (void *)
                        /* 指向 free（ANSI C）或
                           mxFree（用于 MATLAB mexFunction）的指针 */
) ;

void colamd_report
(
    int stats [COLAMD_STATS]
) ;

void colamd_l_report
(
    SuiteSparse_long stats [COLAMD_STATS]
) ;

void symamd_report
(
    int stats [COLAMD_STATS]
) ;

void symamd_l_report
(
    SuiteSparse_long stats [COLAMD_STATS]
) ;

#ifdef __cplusplus
}
#endif

#endif /* COLAMD_H */
```