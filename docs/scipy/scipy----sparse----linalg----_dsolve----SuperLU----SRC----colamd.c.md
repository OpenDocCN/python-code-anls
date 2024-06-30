# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\colamd.c`

```
"""
/*! \\file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/* ========================================================================== */
/* === colamd/symamd - a sparse matrix column ordering algorithm ============ */
/* ========================================================================== */

/* COLAMD / SYMAMD

    colamd:  an approximate minimum degree column ordering algorithm,
        for LU factorization of symmetric or unsymmetric matrices,
    QR factorization, least squares, interior point methods for
    linear programming problems, and other related problems.

    symamd:  an approximate minimum degree ordering algorithm for Cholesky
        factorization of symmetric matrices.

    Purpose:

    Colamd computes a permutation Q such that the Cholesky factorization of
    (AQ)'(AQ) has less fill-in and requires fewer floating point operations
    than A'A.  This also provides a good ordering for sparse partial
    pivoting methods, P(AQ) = LU, where Q is computed prior to numerical
    factorization, and P is computed during numerical factorization via
    conventional partial pivoting with row interchanges.  Colamd is the
    column ordering method used in SuperLU, part of the ScaLAPACK library.
    It is also available as built-in function in MATLAB Version 6,
    available from MathWorks, Inc. (http://www.mathworks.com).  This
    routine can be used in place of colmmd in MATLAB.

        Symamd computes a permutation P of a symmetric matrix A such that the
    Cholesky factorization of PAP' has less fill-in and requires fewer
    floating point operations than A.  Symamd constructs a matrix M such
    that M'M has the same nonzero pattern of A, and then orders the columns
    of M using colmmd.  The column ordering of M is then returned as the
    row and column ordering P of A. 

    Authors:

    The authors of the code itself are Stefan I. Larimore and Timothy A.
    Davis (DrTimothyAldenDavis@gmail.com).  The algorithm was
    developed in collaboration with John Gilbert, Xerox PARC, and Esmond
    Ng, Oak Ridge National Laboratory.

    Acknowledgements:

    This work was supported by the National Science Foundation, under
    grants DMS-9504974 and DMS-9803599.

    Copyright and License:

    Copyright (c) 1998-2007, Timothy A. Davis, All Rights Reserved.
    COLAMD is also available under alternate licenses, contact T. Davis
    for details.

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.
*/
    # 这部分文本是关于 GNU Lesser General Public License (LGPL) 的版权声明和许可条款，用于软件分发和使用的法律说明
    # 以下是 LGPL 许可的授权条款，允许在遵守版权、许可和原始版本可获取性的情况下使用或复制此程序
    # 用户文档必须引用版权、许可、可获取性说明以及 "Used by permission." 的信息，如果使用或修改了该代码的任何版本
    # 允许修改和分发修改后的代码，但必须保留版权、许可和可获取性说明，并包含代码已被修改的声明
    
    # 可获取性：
    # colamd/symamd 库可以在 http://www.suitesparse.com 获取，作为 ACM Algorithm 836 出现
    
    # 参考文献：
    # T. A. Davis, J. R. Gilbert, S. Larimore, E. Ng, 一种近似列最小度排序算法，出现在 ACM Transactions on Mathematical Software，2004 年，第 30 卷，第 3 期，页码 353-376
    # T. A. Davis, J. R. Gilbert, S. Larimore, E. Ng, 算法 836: COLAMD，一种近似列最小度排序算法，出现在 ACM Transactions on Mathematical Software，2004 年，第 30 卷，第 3 期，页码 377-380
/*
/* ========================================================================== */
/* === Description of user-callable routines ================================ */
/* ========================================================================== */

/* COLAMD includes both int and SuiteSparse_long versions of all its routines.
    The description below is for the int version.  For SuiteSparse_long, all
    int arguments become SuiteSparse_long.  SuiteSparse_long is normally
    defined as long, except for WIN64.
*/

/* ----------------------------------------------------------------------------
    colamd_recommended:
---------------------------------------------------------------------------- */

/* C syntax:
    
    #include "colamd.h"
    size_t colamd_recommended (int nnz, int n_row, int n_col) ;
    size_t colamd_l_recommended (SuiteSparse_long nnz,
            SuiteSparse_long n_row, SuiteSparse_long n_col) ;

Purpose:

    Returns recommended value of Alen for use by colamd.  Returns 0
    if any input argument is negative.  The use of this routine
    is optional.  Not needed for symamd, which dynamically allocates
    its own memory.

    Note that in v2.4 and earlier, these routines returned int or long.
    They now return a value of type size_t.

Arguments (all input arguments):

    int nnz ;        Number of nonzeros in the matrix A.  This must
            be the same value as p [n_col] in the call to
            colamd - otherwise you will get a wrong value
            of the recommended memory to use.

    int n_row ;        Number of rows in the matrix A.

    int n_col ;        Number of columns in the matrix A.
*/

/* ----------------------------------------------------------------------------
    colamd_set_defaults:
---------------------------------------------------------------------------- */

/* C syntax:

    #include "colamd.h"
    colamd_set_defaults (double knobs [COLAMD_KNOBS]) ;
    colamd_l_set_defaults (double knobs [COLAMD_KNOBS]) ;

Purpose:

    Sets the default parameters.  The use of this routine is optional.
*/
    Arguments:

        double knobs [COLAMD_KNOBS] ;    Output only.
        // 定义一个 double 类型的数组 knobs，用于存储算法参数，仅为输出。

        NOTE: the meaning of the dense row/col knobs has changed in v2.4
        // 注意：从 v2.4 开始，密集行/列参数的含义已发生变化。

        knobs [0] and knobs [1] control dense row and col detection:
        // knobs[0] 和 knobs[1] 控制密集行和列的检测：

        Colamd: rows with more than
        max (16, knobs [COLAMD_DENSE_ROW] * sqrt (n_col))
        entries are removed prior to ordering.  Columns with more than
        max (16, knobs [COLAMD_DENSE_COL] * sqrt (MIN (n_row,n_col)))
        entries are removed prior to
        ordering, and placed last in the output column ordering. 
        // Colamd 算法：在进行排序之前，将具有超过一定条目数的行和列移除，并将其放置在输出列排序的最后位置。

        Symamd: uses only knobs [COLAMD_DENSE_ROW], which is knobs [0].
        Rows and columns with more than
        max (16, knobs [COLAMD_DENSE_ROW] * sqrt (n))
        entries are removed prior to ordering, and placed last in the
        output ordering.
        // Symamd 算法：仅使用 knobs[COLAMD_DENSE_ROW]，即 knobs[0]。在排序之前，将具有超过一定条目数的行和列移除，并将其放置在输出排序的最后位置。

        COLAMD_DENSE_ROW and COLAMD_DENSE_COL are defined as 0 and 1,
        respectively, in colamd.h.  Default values of these two knobs
        are both 10.  Currently, only knobs [0] and knobs [1] are
        used, but future versions may use more knobs.  If so, they will
        be properly set to their defaults by the future version of
        colamd_set_defaults, so that the code that calls colamd will
        not need to change, assuming that you either use
        colamd_set_defaults, or pass a (double *) NULL pointer as the
        knobs array to colamd or symamd.
        // COLAMD_DENSE_ROW 和 COLAMD_DENSE_COL 在 colamd.h 中分别定义为 0 和 1。这两个 knobs 的默认值均为 10。目前仅使用 knobs[0] 和 knobs[1]，但未来版本可能会使用更多 knobs。如果使用 colamd_set_defaults 或将 knobs 数组传递给 colamd 或 symamd 时，它们将被适当设置为默认值，因此调用 colamd 的代码不需要更改。

        knobs [2]: aggressive absorption
        // knobs[2]：积极吸收参数

            knobs [COLAMD_AGGRESSIVE] controls whether or not to do
            aggressive absorption during the ordering.  Default is TRUE.
            // knobs[COLAMD_AGGRESSIVE] 控制排序过程中是否进行积极吸收。默认值为 TRUE。

    ----------------------------------------------------------------------------
    colamd:
    ----------------------------------------------------------------------------

    C syntax:

        #include "colamd.h"
        int colamd (int n_row, int n_col, int Alen, int *A, int *p,
            double knobs [COLAMD_KNOBS], int stats [COLAMD_STATS]) ;
        SuiteSparse_long colamd_l (SuiteSparse_long n_row,
                SuiteSparse_long n_col, SuiteSparse_long Alen,
                SuiteSparse_long *A, SuiteSparse_long *p, double knobs
                [COLAMD_KNOBS], SuiteSparse_long stats [COLAMD_STATS]) ;
        // colamd 函数的 C 语法声明，接受一些参数和 knobs 数组，返回一个整数值表示成功与否。

    Purpose:

        Computes a column ordering (Q) of A such that P(AQ)=LU or
        (AQ)'AQ=LL' have less fill-in and require fewer floating point
        operations than factorizing the unpermuted matrix A or A'A,
        respectively.
        // 计算矩阵 A 的列排序 Q，使得 P(AQ)=LU 或 (AQ)'AQ=LL' 的填充较少，需要的浮点运算次数较少，相比于对未排列的矩阵 A 或 A'A 进行因式分解。

    Returns:

        TRUE (1) if successful, FALSE (0) otherwise.
        // 如果成功，返回 TRUE（1），否则返回 FALSE（0）。
    Example:
    
        See colamd_example.c for a complete example.
    
        To order the columns of a 5-by-4 matrix with 11 nonzero entries in
        the following nonzero pattern
    
            x 0 x 0
        x 0 x x
        0 x x 0
        0 0 x x
        x x 0 0
    
        with default knobs and no output statistics, do the following:
    
        #include "colamd.h"
        #define ALEN 100
        int A [ALEN] = {0, 1, 4, 2, 4, 0, 1, 2, 3, 1, 3} ;
        int p [ ] = {0, 3, 5, 9, 11} ;
        int stats [COLAMD_STATS] ;
        colamd (5, 4, ALEN, A, p, (double *) NULL, stats) ;
    
        The permutation is returned in the array p, and A is destroyed.
    
    ----------------------------------------------------------------------------
    symamd:
    ----------------------------------------------------------------------------
    
    C syntax:
    
        #include "colamd.h"
        int symamd (int n, int *A, int *p, int *perm,
            double knobs [COLAMD_KNOBS], int stats [COLAMD_STATS],
        void (*allocate) (size_t, size_t), void (*release) (void *)) ;
        SuiteSparse_long symamd_l (SuiteSparse_long n, SuiteSparse_long *A,
                SuiteSparse_long *p, SuiteSparse_long *perm, double knobs
                [COLAMD_KNOBS], SuiteSparse_long stats [COLAMD_STATS], void
                (*allocate) (size_t, size_t), void (*release) (void *)) ;
    
    Purpose:
    
            The symamd routine computes an ordering P of a symmetric sparse
        matrix A such that the Cholesky factorization PAP' = LL' remains
        sparse.  It is based on a column ordering of a matrix M constructed
        so that the nonzero pattern of M'M is the same as A.  The matrix A
        is assumed to be symmetric; only the strictly lower triangular part
        is accessed.  You must pass your selected memory allocator (usually
        calloc/free or mxCalloc/mxFree) to symamd, for it to allocate
        memory for the temporary matrix M.
    
    Returns:
    
        TRUE (1) if successful, FALSE (0) otherwise.
    
    ----------------------------------------------------------------------------
    colamd_report:
    ----------------------------------------------------------------------------
    
    C syntax:
    
        #include "colamd.h"
        colamd_report (int stats [COLAMD_STATS]) ;
        colamd_l_report (SuiteSparse_long stats [COLAMD_STATS]) ;
    
    Purpose:
    
        Prints the error status and statistics recorded in the stats
        array on the standard error output (for a standard C routine)
        or on the MATLAB output (for a mexFunction).
    
    Arguments:
    
        int stats [COLAMD_STATS] ;    Input only.  Statistics from colamd.
    
    ----------------------------------------------------------------------------
    symamd_report:
    
    
    注释：
    # 包含头文件 "colamd.h"，这里假设是为了引入 COLAMD 相关的声明和定义
    #include "colamd.h"
    
    # 输出 stats 数组中记录的错误状态和统计信息到标准错误输出（对于标准的 C 程序）
    symamd_report (int stats [COLAMD_STATS]) ;
    
    # 输出 stats 数组中记录的错误状态和统计信息到 MATLAB 的输出（对于 mexFunction）
    symamd_l_report (SuiteSparse_long stats [COLAMD_STATS]) ;
/* ========================================================================== */
/* === Scaffolding code definitions  ======================================== */
/* ========================================================================== */

/* Ensure that debugging is turned off: */
#ifndef NDEBUG
#define NDEBUG
#endif

/* turn on debugging by uncommenting the following line
 #undef NDEBUG
*/

/*
   Our "scaffolding code" philosophy:  In our opinion, well-written library
   code should keep its "debugging" code, and just normally have it turned off
   by the compiler so as not to interfere with performance.  This serves
   several purposes:

   (1) assertions act as comments to the reader, telling you what the code
       expects at that point.  All assertions will always be true (unless
       there really is a bug, of course).

   (2) leaving in the scaffolding code assists anyone who would like to modify
       the code, or understand the algorithm (by reading the debugging output,
       one can get a glimpse into what the code is doing).

   (3) (gasp!) for actually finding bugs.  This code has been heavily tested
       and "should" be fully functional and bug-free ... but you never know...

    The code will become outrageously slow when debugging is
    enabled.  To control the level of debugging output, set an environment
    variable D to 0 (little), 1 (some), 2, 3, or 4 (lots).  When debugging,
    you should see the following message on the standard output:

        colamd: debug version, D = 1 (THIS WILL BE SLOW!)

    or a similar message for symamd.  If you don't, then debugging has not
    been enabled.

*/

/* ========================================================================== */
/* === Include files ======================================================== */
/* ========================================================================== */
#include <limits.h>   /* Provides constants for various limits */
#include <math.h>     /* Provides mathematical functions */
#include "colamd.h"   /* Header for the COLAMD sparse matrix ordering routines */

#ifdef MATLAB_MEX_FILE
#include "mex.h"      /* Interface for MATLAB MEX-file functions */
#include "matrix.h"   /* MATLAB Matrix API */
#endif /* MATLAB_MEX_FILE */

#if !defined (NPRINT) || !defined (NDEBUG)
#include <stdio.h>    /* Standard I/O functions */
#endif

#ifndef NULL
#define NULL ((void *) 0)  /* Defines NULL pointer value if not already defined */
#endif


/* ========================================================================== */
/* === Row and Column structures ============================================ */
/* ========================================================================== */

/* User code that makes use of the colamd/symamd routines need not directly */
/* reference these structures.  They are used only for colamd_recommended. */

typedef struct Colamd_Col_struct
{
    Int start ;        /* index for A of first row in this column, or DEAD */
                       /* if column is dead */
    Int length ;       /* number of rows in this column */
    union
    {
        Int thickness ;    /* number of original columns represented by this */
                          /* col, if the column is alive */
        /* Additional union members can be added as needed */
    } shared1 ;           /* Shared union for various purposes */
    /* Additional struct members can be added as needed */
} Colamd_Col ;
    Int parent ;    /* parent in parent tree super-column structure, if */
                    /* the column is dead */
    } shared1 ;
    union
    {
    Int score ;    /* the score used to maintain heap, if col is alive */
    Int order ;    /* pivot ordering of this column, if col is dead */
    } shared2 ;
    union
    {
    Int headhash ;    /* head of a hash bucket, if col is at the head of */
                    /* a degree list */
    Int hash ;    /* hash value, if col is not in a degree list */
    Int prev ;    /* previous column in degree list, if col is in a */
                    /* degree list (but not at the head of a degree list) */
    } shared3 ;
    union
    {
    Int degree_next ;    /* next column, if col is in a degree list */
    Int hash_next ;        /* next column, if col is in a hash list */
    } shared4 ;



Int parent ;    /* 定义整数变量 parent，在超级列结构中表示父节点，如果列已经无效 */
} shared1 ;
union
{
Int score ;    /* 如果列有效，表示用于维护堆的分数 */
Int order ;    /* 如果列已经无效，表示该列的轴排序 */
} shared2 ;
union
{
Int headhash ;    /* 如果列在度列表的头部，表示哈希桶的头部 */
Int hash ;    /* 如果列不在度列表中，表示哈希值 */
Int prev ;    /* 如果列在度列表中但不是头部，表示度列表中的前一列 */
} shared3 ;
union
{
Int degree_next ;    /* 如果列在度列表中，表示度列表中的下一列 */
Int hash_next ;        /* 如果列在哈希列表中，表示哈希列表中的下一列 */
} shared4 ;
} Colamd_Col ;

typedef struct Colamd_Row_struct
{
    Int start ;        /* index for A of first col in this row */
    Int length ;    /* number of principal columns in this row */
    union
    {
    Int degree ;    /* number of principal & non-principal columns in row */
    Int p ;        /* used as a row pointer in init_rows_cols () */
    } shared1 ;
    union
    {
    Int mark ;    /* for computing set differences and marking dead rows*/
    Int first_column ;/* first column in row (used in garbage collection) */
    } shared2 ;

} Colamd_Row ;

/* ========================================================================== */
/* === Definitions ========================================================== */
/* ========================================================================== */

/* Routines are either PUBLIC (user-callable) or PRIVATE (not user-callable) */
#define PUBLIC
#define PRIVATE static

#define DENSE_DEGREE(alpha,n) \
    ((Int) MAX (16.0, (alpha) * sqrt ((double) (n))))

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

#define ONES_COMPLEMENT(r) (-(r)-1)

/* -------------------------------------------------------------------------- */
/* Change for version 2.1:  define TRUE and FALSE only if not yet defined */  
/* -------------------------------------------------------------------------- */

#ifndef TRUE
#define TRUE (1)
#endif

#ifndef FALSE
#define FALSE (0)
#endif

/* -------------------------------------------------------------------------- */

#define EMPTY    (-1)

/* Row and column status */
#define ALIVE    (0)             /* 定义行和列的状态常量：活跃状态 */
#define DEAD    (-1)             /* 定义行的状态常量：死亡状态 */

/* Column status */
#define DEAD_PRINCIPAL        (-1)    /* 定义主列的状态常量：死亡主列 */
#define DEAD_NON_PRINCIPAL    (-2)    /* 定义非主列的状态常量：死亡非主列 */

/* Macros for row and column status update and checking. */
#define ROW_IS_DEAD(r)            ROW_IS_MARKED_DEAD (Row[r].shared2.mark)    /* 检查行是否死亡 */
#define ROW_IS_MARKED_DEAD(row_mark)    (row_mark < ALIVE)    /* 检查行标记是否小于活跃状态 */
#define ROW_IS_ALIVE(r)            (Row [r].shared2.mark >= ALIVE)    /* 检查行是否活跃 */
#define COL_IS_DEAD(c)            (Col [c].start < ALIVE)    /* 检查列是否死亡 */
#define COL_IS_ALIVE(c)            (Col [c].start >= ALIVE)    /* 检查列是否活跃 */
#define COL_IS_DEAD_PRINCIPAL(c)    (Col [c].start == DEAD_PRINCIPAL)    /* 检查主列是否死亡 */
#define KILL_ROW(r)            { Row [r].shared2.mark = DEAD ; }    /* 将行标记为死亡 */
#define KILL_PRINCIPAL_COL(c)        { Col [c].start = DEAD_PRINCIPAL ; }    /* 将主列标记为死亡 */
#define KILL_NON_PRINCIPAL_COL(c)    { Col [c].start = DEAD_NON_PRINCIPAL ; }    /* 将非主列标记为死亡 */

/* ========================================================================== */
/* === Colamd reporting mechanism =========================================== */
/* ========================================================================== */

#if defined (MATLAB_MEX_FILE) || defined (MATHWORKS)
/* In MATLAB, matrices are 1-based to the user, but 0-based internally */
#define INDEX(i) ((i)+1)    /* 定义 MATLAB 环境下的索引转换宏 */
#else
/* In C, matrices are 0-based and indices are reported as such in *_report */
#define INDEX(i) (i)    /* 定义 C 环境下的索引宏 */
#endif

/* ========================================================================== */
/* === Prototypes of PRIVATE routines ======================================= */
/* ========================================================================== */

/* 初始化行和列 */
PRIVATE Int init_rows_cols
(
    Int n_row,
    Int n_col,
    Colamd_Row Row [],
    Colamd_Col Col [],
    Int A [],
    Int p [],
    Int stats [COLAMD_STATS]
) ;

/* 初始化评分 */
PRIVATE void init_scoring
(
    Int n_row,
    Int n_col,
    Colamd_Row Row [],
    Colamd_Col Col [],
    Int A [],
    Int head [],
    double knobs [COLAMD_KNOBS],
    Int *p_n_row2,
    Int *p_n_col2,
    Int *p_max_deg
) ;

/* 查找顺序 */
PRIVATE Int find_ordering
(
    Int n_row,
    Int n_col,
    Int Alen,
    Colamd_Row Row [],
    Colamd_Col Col [],
    Int A [],
    Int head [],
    Int n_col2,
    Int max_deg,
    Int pfree,
    Int aggressive
) ;

/* 排序子节点 */
PRIVATE void order_children
(
    Int n_col,
    Colamd_Col Col [],
    Int p []
) ;

/* 检测超级列 */
PRIVATE void detect_super_cols
(

#ifndef NDEBUG
    Int n_col,
    Colamd_Row Row [],
#endif /* NDEBUG */

    Colamd_Col Col [],
    Int A [],
    Int head [],
    Int row_start,
    Int row_length
) ;

/* 垃圾回收 */
PRIVATE Int garbage_collection
(
    Int n_row,
    Int n_col,
    Colamd_Row Row [],
    Colamd_Col Col [],
    Int A [],
    Int *pfree
) ;

/* 清除标记 */
PRIVATE Int clear_mark
(
    Int tag_mark,
    Int max_mark,
    Int n_row,
    Colamd_Row Row []
) ;

/* 打印报告 */
PRIVATE void print_report
(
    char *method,
    Int stats [COLAMD_STATS]
) ;

/* ========================================================================== */
/* === Debugging prototypes and definitions ================================= */
/* ========================================================================== */

#ifndef NDEBUG

#include <assert.h>

/* colamd_debug 是唯一的全局变量，仅在调试时存在 */

PRIVATE Int colamd_debug = 0 ;    /* 调试打印级别 */

#define DEBUG0(params) { SUITESPARSE_PRINTF (params) ; }
#define DEBUG1(params) { if (colamd_debug >= 1) SUITESPARSE_PRINTF (params) ; }
#define DEBUG2(params) { if (colamd_debug >= 2) SUITESPARSE_PRINTF (params) ; }
#define DEBUG3(params) { if (colamd_debug >= 3) SUITESPARSE_PRINTF (params) ; }
#define DEBUG4(params) { if (colamd_debug >= 4) SUITESPARSE_PRINTF (params) ; }

#ifdef MATLAB_MEX_FILE
#define ASSERT(expression) (mxAssert ((expression), ""))
#else
#define ASSERT(expression) (assert (expression))
#endif /* MATLAB_MEX_FILE */

/* 从环境中获取调试打印级别 */
PRIVATE void colamd_get_debug
(
    char *method
) ;

/* 调试：打印度列表 */
PRIVATE void debug_deg_lists
(
    Int n_row,
    Int n_col,
    Colamd_Row Row [],
    Colamd_Col Col [],
    Int head [],
    Int min_score,
    Int should,
    Int max_deg
) ;

/* 调试：打印标记 */
PRIVATE void debug_mark
(
    Int n_row,
    Colamd_Row Row [],
    Int tag_mark,
    Int max_mark
) ;

/* 调试：打印矩阵 */
PRIVATE void debug_matrix
(
    Int n_row,
    Int n_col,
    Colamd_Row Row [],
    Colamd_Col Col [],
    Int A []
) ;

/* 调试：打印结构 */
PRIVATE void debug_structures
(
    Int n_row,
    Int n_col,
    Colamd_Row Row [],
    Colamd_Col Col [],
    # 声明一个整数类型的数组变量 A
    Int A [],
    # 声明一个整数类型的变量 n_col2
    Int n_col2
/* 以下是对一些调试宏和一个函数的定义 */

/* 调试宏定义，在 NDEBUG 没有定义时启用 */
#define DEBUG0(params) ;
#define DEBUG1(params) ;
#define DEBUG2(params) ;
#define DEBUG3(params) ;
#define DEBUG4(params) ;

/* ASSERT 宏定义，用于断言检查 */
#define ASSERT(expression)

/* ========================================================================== */
/* === USER-CALLABLE ROUTINES: ============================================== */
/* ========================================================================== */

/* ========================================================================== */
/* === colamd_recommended =================================================== */
/* ========================================================================== */

/*
    colamd_recommended 函数返回推荐的 Alen 大小。这个值经过优化，可以在垃圾回收次数和内存需求之间取得良好平衡。
    如果任何参数为负数，或者发生整数溢出，将返回 0 作为错误条件。
    矩阵的行和列索引需要 2*nnz 的空间。
    对于 colamd，Col 和 Row 数组需要 COLAMD_C(n_col) 和 COLAMD_R(n_row) 的空间，这些数组是 colamd 的内部数据结构（大约需要 6*n_col + 4*n_row 的空间）。
    额外的 n_col 空间是最小的 "elbow room"，建议为了运行时效率再额外分配 nnz/5 的空间。

    Alen 的推荐大小大约为 2.2*nnz + 7*n_col + 4*n_row + 10。

    当使用 symamd 时，不需要此函数。
*/

/* add two values of type size_t, and check for integer overflow */
static size_t t_add (size_t a, size_t b, int *ok)
{
    (*ok) = (*ok) && ((a + b) >= MAX (a,b)) ;
    return ((*ok) ? (a + b) : 0) ;
}

/* compute a*k where k is a small integer, and check for integer overflow */
static size_t t_mult (size_t a, size_t k, int *ok)
{
    size_t i, s = 0 ;
    for (i = 0 ; i < k ; i++)
    {
        s = t_add (s, a, ok) ;
    }
    return (s) ;
}

/* size of the Col and Row structures */
#define COLAMD_C(n_col,ok) \
    ((t_mult (t_add (n_col, 1, ok), sizeof (Colamd_Col), ok) / sizeof (Int)))

#define COLAMD_R(n_row,ok) \
    ((t_mult (t_add (n_row, 1, ok), sizeof (Colamd_Row), ok) / sizeof (Int)))

/* PUBLIC 是一个宏，用于指示以下函数 COLAMD_recommended 是公共可调用的 */
PUBLIC size_t COLAMD_recommended    /* returns recommended value of Alen. */
(
    /* === Parameters ======================================================= */

    Int nnz,            /* number of nonzeros in A */
    Int n_row,          /* number of rows in A */
    Int n_col           /* number of columns in A */
)
{
    size_t s, c, r ;
    int ok = TRUE ;
    if (nnz < 0 || n_row < 0 || n_col < 0)
    {
        return (0) ;
    }
    s = t_mult (nnz, 2, &ok) ;        /* 2*nnz */
    c = COLAMD_C (n_col, &ok) ;        /* size of column structures */
    r = COLAMD_R (n_row, &ok) ;        /* size of row structures */
    s = t_add (s, c, &ok) ;
    s = t_add (s, r, &ok) ;

    /* 返回推荐的 Alen 大小 */
    return s;
}
    # 调用 t_add 函数，将 s 增加 n_col 的值，同时检查操作是否成功
    s = t_add(s, n_col, &ok) ;        /* elbow room */
    # 调用 t_add 函数，将 s 增加 nnz/5 的值，同时检查操作是否成功
    s = t_add(s, nnz/5, &ok) ;        /* elbow room */
    # 检查 ok 的值，确保前面的操作都成功，并且 s 小于 Int_MAX
    ok = ok && (s < Int_MAX) ;
    # 根据 ok 的值返回 s 或者 0
    return (ok ? s : 0) ;
/* ========================================================================== */
/* === colamd_set_defaults ================================================== */
/* ========================================================================== */

/*
    The colamd_set_defaults routine sets the default values of the user-
    controllable parameters for colamd and symamd:

    Colamd: rows with more than max (16, knobs [0] * sqrt (n_col))
    entries are removed prior to ordering.  Columns with more than
    max (16, knobs [1] * sqrt (MIN (n_row,n_col))) entries are removed
    prior to ordering, and placed last in the output column ordering. 

    Symamd: Rows and columns with more than max (16, knobs [0] * sqrt (n))
    entries are removed prior to ordering, and placed last in the
    output ordering.

    knobs [0]    dense row control

    knobs [1]    dense column control

    knobs [2]    if nonzero, do aggressive absorption

    knobs [3..19]    unused, but future versions might use this

*/

PUBLIC void COLAMD_set_defaults
(
    /* === Parameters ======================================================= */

    double knobs [COLAMD_KNOBS]        /* knob array */
)
{
    /* === Local variables ================================================== */

    Int i ;

    if (!knobs)
    {
        return ;            /* no knobs to initialize */
    }
    for (i = 0 ; i < COLAMD_KNOBS ; i++)
    {
        knobs [i] = 0 ;
    }
    knobs [COLAMD_DENSE_ROW] = 10 ;
    knobs [COLAMD_DENSE_COL] = 10 ;
    knobs [COLAMD_AGGRESSIVE] = TRUE ;    /* default: do aggressive absorption*/
}


/* ========================================================================== */
/* === symamd =============================================================== */
/* ========================================================================== */

PUBLIC Int SYMAMD_MAIN            /* return TRUE if OK, FALSE otherwise */
(
    /* === Parameters ======================================================= */

    Int n,                /* number of rows and columns of A */
    Int A [],                /* row indices of A */
    Int p [],                /* column pointers of A */
    Int perm [],            /* output permutation, size n+1 */
    double knobs [COLAMD_KNOBS],    /* parameters (uses defaults if NULL) */
    Int stats [COLAMD_STATS],        /* output statistics and error codes */
    void * (*allocate) (size_t, size_t),
                        /* pointer to calloc (ANSI C) or */
                    /* mxCalloc (for MATLAB mexFunction) */
    void (*release) (void *)
                        /* pointer to free (ANSI C) or */
                        /* mxFree (for MATLAB mexFunction) */
)
{
    /* === Local variables ================================================== */

    Int *count ;        /* length of each column of M, and col pointer*/
    Int *mark ;            /* mark array for finding duplicate entries */
    Int *M ;            /* row indices of matrix M */
    size_t Mlen ;        /* M 的长度 */
    Int n_row ;            /* M 的行数 */
    Int nnz ;            /* A 中的条目数 */
    Int i ;            /* A 的行索引 */
    Int j ;            /* A 的列索引 */
    Int k ;            /* M 的行索引 */
    Int mnz ;            /* M 中的非零条目数 */
    Int pp ;            /* A 列的索引 */
    Int last_row ;        /* 当前列中最后一个看到的行 */
    Int length ;        /* 列中的非零条目数 */

    double cknobs [COLAMD_KNOBS] ;        /* colamd 的参数调节 */
    double default_knobs [COLAMD_KNOBS] ;    /* colamd 的默认参数 */
#ifndef NDEBUG
    colamd_get_debug ("symamd") ;
#endif /* NDEBUG */

/* === 检查输入参数 ======================================== */

if (!stats)
{
DEBUG0 (("symamd: stats not present\n")) ;
return (FALSE) ;
}
for (i = 0 ; i < COLAMD_STATS ; i++)
{
stats [i] = 0 ;
}
stats [COLAMD_STATUS] = COLAMD_OK ;
stats [COLAMD_INFO1] = -1 ;
stats [COLAMD_INFO2] = -1 ;

if (!A)
{
stats [COLAMD_STATUS] = COLAMD_ERROR_A_not_present ;
DEBUG0 (("symamd: A not present\n")) ;
return (FALSE) ;
}

if (!p)        /* p is not present */
{
stats [COLAMD_STATUS] = COLAMD_ERROR_p_not_present ;
DEBUG0 (("symamd: p not present\n")) ;
return (FALSE) ;
}

if (n < 0)        /* n must be >= 0 */
{
stats [COLAMD_STATUS] = COLAMD_ERROR_ncol_negative ;
stats [COLAMD_INFO1] = n ;
DEBUG0 (("symamd: n negative %d\n", n)) ;
return (FALSE) ;
}

nnz = p [n] ;
if (nnz < 0)    /* nnz must be >= 0 */
{
stats [COLAMD_STATUS] = COLAMD_ERROR_nnz_negative ;
stats [COLAMD_INFO1] = nnz ;
DEBUG0 (("symamd: number of entries negative %d\n", nnz)) ;
return (FALSE) ;
}

if (p [0] != 0)
{
stats [COLAMD_STATUS] = COLAMD_ERROR_p0_nonzero ;
stats [COLAMD_INFO1] = p [0] ;
DEBUG0 (("symamd: p[0] not zero %d\n", p [0])) ;
return (FALSE) ;
}

/* === 如果没有设置旋钮，使用默认旋钮 =================================== */

if (!knobs)
{
COLAMD_set_defaults (default_knobs) ;
knobs = default_knobs ;
}

/* === 分配计数和标记数组 ========================================== */

count = (Int *) ((*allocate) (n+1, sizeof (Int))) ;
if (!count)
{
stats [COLAMD_STATUS] = COLAMD_ERROR_out_of_memory ;
DEBUG0 (("symamd: allocate count (size %d) failed\n", n+1)) ;
return (FALSE) ;
}

mark = (Int *) ((*allocate) (n+1, sizeof (Int))) ;
if (!mark)
{
stats [COLAMD_STATUS] = COLAMD_ERROR_out_of_memory ;
(*release) ((void *) count) ;
DEBUG0 (("symamd: allocate mark (size %d) failed\n", n+1)) ;
return (FALSE) ;
}

/* === 计算 M 的列计数，检查 A 是否有效 ================== */

stats [COLAMD_INFO3] = 0 ;  /* 重复或未排序的行索引数目 */

for (i = 0 ; i < n ; i++)
{
mark [i] = -1 ;
}

for (j = 0 ; j < n ; j++)
{
last_row = -1 ;

length = p [j+1] - p [j] ;
if (length < 0)
{
/* 列指针必须非递减 */
stats [COLAMD_STATUS] = COLAMD_ERROR_col_length_negative ;
stats [COLAMD_INFO1] = j ;
stats [COLAMD_INFO2] = length ;
(*release) ((void *) count) ;
(*release) ((void *) mark) ;
DEBUG0 (("symamd: col %d negative length %d\n", j, length)) ;
return (FALSE) ;
}

for (pp = p [j] ; pp < p [j+1] ; pp++)
    {
        i = A [pp] ;  // 从数组 A 中获取索引 pp 处的元素，赋值给变量 i
        if (i < 0 || i >= n)
        {
        /* 行索引 i 在列索引 j 中越界 */
        stats [COLAMD_STATUS] = COLAMD_ERROR_row_index_out_of_bounds ;  // 设置错误状态：行索引越界
        stats [COLAMD_INFO1] = j ;  // 存储列索引 j 到状态信息数组中
        stats [COLAMD_INFO2] = i ;  // 存储行索引 i 到状态信息数组中
        stats [COLAMD_INFO3] = n ;  // 存储总行数 n 到状态信息数组中
        (*release) ((void *) count) ;  // 释放 count 数组的内存
        (*release) ((void *) mark) ;  // 释放 mark 数组的内存
        DEBUG0 (("symamd: row %d col %d out of bounds\n", i, j)) ;  // 调试输出：行 i 列 j 越界
        return (FALSE) ;  // 返回假值，表示出现错误
        }

        if (i <= last_row || mark [i] == j)
        {
        /* 行索引未排序或重复（或两者兼有），列索引混乱，这是通知而非错误条件 */
        stats [COLAMD_STATUS] = COLAMD_OK_BUT_JUMBLED ;  // 设置状态为：列索引混乱但没有错误
        stats [COLAMD_INFO1] = j ;  // 存储列索引 j 到状态信息数组中
        stats [COLAMD_INFO2] = i ;  // 存储行索引 i 到状态信息数组中
        (stats [COLAMD_INFO3]) ++ ;  // 将状态信息数组中的第三项加一
        DEBUG1 (("symamd: row %d col %d unsorted/duplicate\n", i, j)) ;  // 调试输出：行 i 列 j 未排序或重复
        }

        if (i > j && mark [i] != j)
        {
        /* 矩阵 M 的第 k 行将包含列索引 i 和 j */
        count [i]++ ;  // 计数数组中索引为 i 的位置加一
        count [j]++ ;  // 计数数组中索引为 j 的位置加一
        }

        /* 标记该行已在此列中出现 */
        mark [i] = j ;  // 将 mark 数组中索引为 i 的位置标记为 j

        last_row = i ;  // 更新最后处理的行索引为 i
    }
    }

    /* v2.4: 移除了释放 mark 数组的操作 */

    /* === 计算 M 的列指针 ===================================== */

    /* 使用输出置换 perm 来计算 M 的列指针 */
    perm [0] = 0 ;  // 初始化 perm 数组的第一个元素为 0
    for (j = 1 ; j <= n ; j++)
    {
    perm [j] = perm [j-1] + count [j-1] ;  // 根据 count 数组计算 perm 数组的每个元素
    }
    for (j = 0 ; j < n ; j++)
    {
    count [j] = perm [j] ;  // 将 perm 数组的值复制回 count 数组
    }

    /* === 构建 M 矩阵 ====================================== */

    mnz = perm [n] ;  // 获取 M 矩阵的非零元素数目
    n_row = mnz / 2 ;  // 计算 M 矩阵的行数
    Mlen = COLAMD_recommended (mnz, n_row, n) ;  // 计算推荐的 M 矩阵长度
    M = (Int *) ((*allocate) (Mlen, sizeof (Int))) ;  // 分配 M 矩阵的内存空间
    DEBUG0 (("symamd: M is %d-by-%d with %d entries, Mlen = %g\n",
        n_row, n, mnz, (double) Mlen)) ;  // 调试输出：M 矩阵大小及其非零元素数目

    if (!M)
    {
    stats [COLAMD_STATUS] = COLAMD_ERROR_out_of_memory ;  // 设置错误状态：内存分配失败
    (*release) ((void *) count) ;  // 释放 count 数组的内存
    (*release) ((void *) mark) ;  // 释放 mark 数组的内存
    DEBUG0 (("symamd: allocate M (size %g) failed\n", (double) Mlen)) ;  // 调试输出：分配 M 矩阵内存失败
    return (FALSE) ;  // 返回假值，表示出现错误
    }

    k = 0 ;  // 初始化 k 为 0

    if (stats [COLAMD_STATUS] == COLAMD_OK)
    {
    /* 矩阵正常 */
    for (j = 0 ; j < n ; j++)
    {
        ASSERT (p [j+1] - p [j] >= 0) ;  // 断言：每列中的元素数量不小于 0
        for (pp = p [j] ; pp < p [j+1] ; pp++)
        {
        i = A [pp] ;  // 从数组 A 中获取索引 pp 处的元素，赋值给变量 i
        ASSERT (i >= 0 && i < n) ;  // 断言：索引 i 大于等于 0 并且小于 n
        if (i > j)
        {
            /* 矩阵 M 的第 k 行包含列索引 i 和 j */
            M [count [i]++] = k ;  // 将 k 存储在 M 矩阵第 count[i] 个位置上，并将 count[i] 加一
            M [count [j]++] = k ;  // 将 k 存储在 M 矩阵第 count[j] 个位置上，并将 count[j] 加一
            k++ ;  // 更新 k 的值
        }
        }
    }
    }
    else
    {
    /* 矩阵混乱。不向 M 中添加重复项。未排序的列是可以接受的。 */
    DEBUG0 (("symamd: A 中存在重复项。\n")) ;  // 调试输出：A 中存在重复项
    for (i = 0 ; i < n ; i++)
    {
        mark [i] = -1 ;  // 将 mark 数组中的元素初始化为 -1
    }
    for (j = 0 ; j < n ; j++)
    {

        mark [i] = -1 ;  // 将 mark 数组中的元素初始化为 -1
    }
    for (j = 0 ; j < n ; j++)
    {
    {
        ASSERT (p [j+1] - p [j] >= 0) ;
        // 断言：确保索引 j+1 大于等于索引 j，即 p 数组中的元素是非负的增量
        for (pp = p [j] ; pp < p [j+1] ; pp++)
        {
            i = A [pp] ;
            // 断言：确保索引 i 在有效范围内
            ASSERT (i >= 0 && i < n) ;
            if (i > j && mark [i] != j)
            {
                /* row k of M contains column indices i and j */
                // 将行 k 中的列索引 i 和 j 加入到 M 数组中
                M [count [i]++] = k ;
                M [count [j]++] = k ;
                k++ ;
                mark [i] = j ;
            }
        }
    }
    /* v2.4: free(mark) moved below */
    // v2.4 版本：将释放 mark 数组的操作移动到下面

    /* count and mark no longer needed */
    // 不再需要 count 和 mark 数组
    (*release) ((void *) count) ;
    (*release) ((void *) mark) ;    /* v2.4: free (mark) moved here */
    // 释放 count 和 mark 数组；v2.4 版本：释放 mark 数组的操作移到这里
    ASSERT (k == n_row) ;

    /* === Adjust the knobs for M =========================================== */

    for (i = 0 ; i < COLAMD_KNOBS ; i++)
    {
        cknobs [i] = knobs [i] ;
    }
    // 调整 M 的参数 cknobs，复制 knobs 数组的内容到 cknobs 数组

    /* there are no dense rows in M */
    // M 中没有稠密行
    cknobs [COLAMD_DENSE_ROW] = -1 ;
    cknobs [COLAMD_DENSE_COL] = knobs [COLAMD_DENSE_ROW] ;
    // 设置 cknobs 数组中的密集行和列参数

    /* === Order the columns of M =========================================== */

    /* v2.4: colamd cannot fail here, so the error check is removed */
    // v2.4 版本：这里 colamd 不会失败，因此移除了错误检查
    (void) COLAMD_MAIN (n_row, n, (Int) Mlen, M, perm, cknobs, stats) ;
    // 调用 COLAMD_MAIN 函数对 M 的列进行排序，输出置换结果保存在 perm 数组中

    /* Note that the output permutation is now in perm */
    // 注意：输出的置换现在存储在 perm 数组中

    /* === get the statistics for symamd from colamd ======================== */

    /* a dense column in colamd means a dense row and col in symamd */
    // colamd 中的稠密列意味着 symamd 中有稠密行和列
    stats [COLAMD_DENSE_ROW] = stats [COLAMD_DENSE_COL] ;
    // 将 colamd 统计中的密集行数复制给 symamd 统计中的密集行数

    /* === Free M =========================================================== */

    (*release) ((void *) M) ;
    // 释放 M 数组的内存空间
    DEBUG0 (("symamd: done.\n")) ;
    // 调试输出信息：symamd 完成。
    return (TRUE) ;
    // 返回 TRUE，表示函数执行成功
}

/* ========================================================================== */
/* === colamd =============================================================== */
/* ========================================================================== */

'''
    colamd算法计算稀疏矩阵A的列顺序Q，使得LU分解P(AQ) = LU保持稀疏，
    其中P是通过部分主元选取而选择的。该算法也可以视为提供一个置换Q，
    使得Cholesky分解(AQ)'(AQ) = LL'保持稀疏。
'''

PUBLIC Int COLAMD_MAIN        /* 返回TRUE表示成功，FALSE表示失败*/
(
    /* === Parameters ======================================================= */

    Int n_row,            /* A矩阵的行数 */
    Int n_col,            /* A矩阵的列数 */
    Int Alen,            /* A矩阵的长度 */
    Int A [],            /* A矩阵的行索引 */
    Int p [],            /* A矩阵列的指针 */
    double knobs [COLAMD_KNOBS],/* 参数（如果为NULL，则使用默认值） */
    Int stats [COLAMD_STATS]    /* 输出统计信息和错误代码 */
)
{
    /* === Local variables ================================================== */

    Int i ;            /* 循环索引 */
    Int nnz ;            /* A矩阵中的非零元素个数 */
    size_t Row_size ;        /* Row []数组的大小（整数个数） */
    size_t Col_size ;        /* Col []数组的大小（整数个数） */
    size_t need ;        /* A数组所需的最小长度 */
    Colamd_Row *Row ;        /* 指向A数组中Row [0..n_row]范围的指针 */
    Colamd_Col *Col ;        /* 指向A数组中Col [0..n_col]范围的指针 */
    Int n_col2 ;        /* 非稠密且非空的列数 */
    Int n_row2 ;        /* 非稠密且非空的行数 */
    Int ngarbage ;        /* 执行的垃圾回收次数 */
    Int max_deg ;        /* 最大行度 */
    double default_knobs [COLAMD_KNOBS] ;    /* 默认的参数数组 */
    Int aggressive ;        /* 是否进行激进的吸收 */
    int ok ;

#ifndef NDEBUG
    colamd_get_debug ("colamd") ;
#endif /* NDEBUG */

    /* === Check the input arguments ======================================== */

    if (!stats)
    {
    DEBUG0 (("colamd: stats not present\n")) ;
    return (FALSE) ;
    }
    for (i = 0 ; i < COLAMD_STATS ; i++)
    {
    stats [i] = 0 ;
    }
    stats [COLAMD_STATUS] = COLAMD_OK ;
    stats [COLAMD_INFO1] = -1 ;
    stats [COLAMD_INFO2] = -1 ;

    if (!A)        /* A数组不存在 */
    {
    stats [COLAMD_STATUS] = COLAMD_ERROR_A_not_present ;
    DEBUG0 (("colamd: A not present\n")) ;
    return (FALSE) ;
    }

    if (!p)        /* p数组不存在 */
    {
    stats [COLAMD_STATUS] = COLAMD_ERROR_p_not_present ;
    DEBUG0 (("colamd: p not present\n")) ;
        return (FALSE) ;
    }

    if (n_row < 0)    /* n_row必须大于等于0 */
    {
    stats [COLAMD_STATUS] = COLAMD_ERROR_nrow_negative ;
    stats [COLAMD_INFO1] = n_row ;
    DEBUG0 (("colamd: nrow negative %d\n", n_row)) ;
        return (FALSE) ;


// 如果行数 n_row 是负数，则记录错误并返回 FALSE
if (n_row < 0) {
    stats [COLAMD_STATUS] = COLAMD_ERROR_nrow_negative ;
    stats [COLAMD_INFO1] = n_row ;
    DEBUG0 (("colamd: nrow negative %d\n", n_row)) ;
    return (FALSE) ;
}



    if (n_col < 0)    /* n_col must be >= 0 */
    {
    stats [COLAMD_STATUS] = COLAMD_ERROR_ncol_negative ;
    stats [COLAMD_INFO1] = n_col ;
    DEBUG0 (("colamd: ncol negative %d\n", n_col)) ;
        return (FALSE) ;
    }


// 如果列数 n_col 是负数，则记录错误并返回 FALSE
if (n_col < 0) {
    stats [COLAMD_STATUS] = COLAMD_ERROR_ncol_negative ;
    stats [COLAMD_INFO1] = n_col ;
    DEBUG0 (("colamd: ncol negative %d\n", n_col)) ;
    return (FALSE) ;
}



    nnz = p [n_col] ;
    if (nnz < 0)    /* nnz must be >= 0 */
    {
    stats [COLAMD_STATUS] = COLAMD_ERROR_nnz_negative ;
    stats [COLAMD_INFO1] = nnz ;
    DEBUG0 (("colamd: number of entries negative %d\n", nnz)) ;
    return (FALSE) ;
    }


// 如果非零元素数 nnz 是负数，则记录错误并返回 FALSE
nnz = p[n_col];
if (nnz < 0) {
    stats [COLAMD_STATUS] = COLAMD_ERROR_nnz_negative ;
    stats [COLAMD_INFO1] = nnz ;
    DEBUG0 (("colamd: number of entries negative %d\n", nnz)) ;
    return (FALSE) ;
}



    if (p [0] != 0)
    {
    stats [COLAMD_STATUS] = COLAMD_ERROR_p0_nonzero    ;
    stats [COLAMD_INFO1] = p [0] ;
    DEBUG0 (("colamd: p[0] not zero %d\n", p [0])) ;
    return (FALSE) ;
    }


// 如果 p[0] 不为零，则记录错误并返回 FALSE
if (p[0] != 0) {
    stats [COLAMD_STATUS] = COLAMD_ERROR_p0_nonzero ;
    stats [COLAMD_INFO1] = p [0] ;
    DEBUG0 (("colamd: p[0] not zero %d\n", p [0])) ;
    return (FALSE) ;
}



    /* === If no knobs, set default knobs =================================== */


// 如果没有指定参数 knobs，则设置默认的参数 knobs
if (!knobs) {
    COLAMD_set_defaults (default_knobs) ;
    knobs = default_knobs ;
}



    aggressive = (knobs [COLAMD_AGGRESSIVE] != FALSE) ;


// 根据 knobs 中 COLAMD_AGGRESSIVE 的值确定是否启用了侵略性优化
aggressive = (knobs [COLAMD_AGGRESSIVE] != FALSE) ;



    /* === Allocate the Row and Col arrays from array A ===================== */


// 从数组 A 中分配 Row 和 Col 数组
ok = TRUE ;
Col_size = COLAMD_C (n_col, &ok) ;        /* size of Col array of structs */
Row_size = COLAMD_R (n_row, &ok) ;        /* size of Row array of structs */



    /* need = 2*nnz + n_col + Col_size + Row_size ; */
    need = t_mult (nnz, 2, &ok) ;
    need = t_add (need, n_col, &ok) ;
    need = t_add (need, Col_size, &ok) ;
    need = t_add (need, Row_size, &ok) ;


// 计算执行排序所需的总空间 need
need = t_mult (nnz, 2, &ok) ;
need = t_add (need, n_col, &ok) ;
need = t_add (need, Col_size, &ok) ;
need = t_add (need, Row_size, &ok) ;



    if (!ok || need > (size_t) Alen || need > Int_MAX)
    {
    /* not enough space in array A to perform the ordering */
    stats [COLAMD_STATUS] = COLAMD_ERROR_A_too_small ;
    stats [COLAMD_INFO1] = need ;
    stats [COLAMD_INFO2] = Alen ;
    DEBUG0 (("colamd: Need Alen >= %d, given only Alen = %d\n", need,Alen));
    return (FALSE) ;
    }


// 如果数组 A 的空间不足以执行排序，则记录错误并返回 FALSE
if (!ok || need > (size_t) Alen || need > Int_MAX) {
    stats [COLAMD_STATUS] = COLAMD_ERROR_A_too_small ;
    stats [COLAMD_INFO1] = need ;
    stats [COLAMD_INFO2] = Alen ;
    DEBUG0 (("colamd: Need Alen >= %d, given only Alen = %d\n", need,Alen));
    return (FALSE) ;
}



    Alen -= Col_size + Row_size ;
    Col = (Colamd_Col *) &A [Alen] ;
    Row = (Colamd_Row *) &A [Alen + Col_size] ;


// 更新数组 A 的指针，分配给 Col 和 Row 数组
Alen -= Col_size + Row_size ;
Col = (Colamd_Col *) &A [Alen] ;
Row = (Colamd_Row *) &A [Alen + Col_size] ;



    /* === Construct the row and column data structures ===================== */


// 构造行和列的数据结构
if (!init_rows_cols (n_row, n_col, Row, Col, A, p, stats)) {
    // 如果输入矩阵无效，则返回 FALSE
    DEBUG0 (("colamd: Matrix invalid\n")) ;
    return (FALSE) ;
}



    /* === Initialize scores, kill dense rows/columns ======================= */


// 初始化分数，处理稠密行和列
init_scoring (n_row, n_col, Row, Col, A, p, knobs,
              &n_row2, &n_col2, &max_deg) ;



    /* === Order the supercolumns =========================================== */


// 对超列进行排序
ngarbage = find_ordering (n_row, n_col, Alen, Row, Col, A, p,
                          n_col2, max_deg, 2*nnz, aggressive) ;



    /* === Order the non-principal columns ================================== */


// 对非主要列进行排序
order_children (n_col, Col, p) ;



    /* === Return statistics in stats ======================================= */


// 将统计信息存入 stats 中并返回 TRUE
stats [COLAMD_DENSE_ROW] = n_row - n_row2 ;
stats [COLAMD_DENSE_COL] = n_col - n_col2 ;
stats [COLAMD_DEFRAG_COUNT] = ngarbage ;
DEBUG0 (("colamd: done.\n")) ;
return (TRUE) ;
/* ========================================================================== */
/* === init_rows_cols ======================================================= */
/* ========================================================================== */

/*
    Takes the column form of the matrix in A and creates the row form of the
    matrix.  Also, row and column attributes are stored in the Col and Row
    structs.  If the columns are un-sorted or contain duplicate row indices,
    this routine will also sort and remove duplicate row indices from the
    column form of the matrix.  Returns FALSE if the matrix is invalid,
    TRUE otherwise.  Not user-callable.
*/

PRIVATE Int init_rows_cols    /* returns TRUE if OK, or FALSE otherwise */
(
    /* === Parameters ======================================================= */

    Int n_row,            /* number of rows of A */
    Int n_col,            /* number of columns of A */
    Colamd_Row Row [],        /* of size n_row+1 */
    Colamd_Col Col [],        /* of size n_col+1 */
    Int A [],            /* row indices of A, of size Alen */
    Int p [],            /* pointers to columns in A, of size n_col+1 */
    Int stats [COLAMD_STATS]    /* colamd statistics */ 
)
{
    /* === Local variables ================================================== */

    Int col ;            /* a column index */
    Int row ;            /* a row index */
    Int *cp ;            /* a column pointer */
    Int *cp_end ;        /* a pointer to the end of a column */
    Int *rp ;            /* a row pointer */
    Int *rp_end ;        /* a pointer to the end of a row */
    Int last_row ;        /* previous row */

    /* === Initialize columns, and check column pointers ==================== */

    for (col = 0 ; col < n_col ; col++)
    {
        // 设置列的起始位置和长度
        Col [col].start = p [col] ;
        Col [col].length = p [col+1] - p [col] ;

        // 检查列的长度是否小于0
        if (Col [col].length < 0)
    {
        /* column pointers must be non-decreasing */
        stats [COLAMD_STATUS] = COLAMD_ERROR_col_length_negative ;
        stats [COLAMD_INFO1] = col ;
        stats [COLAMD_INFO2] = Col [col].length ;
        DEBUG0 (("colamd: col %d length %d < 0\n", col, Col [col].length)) ;
        return (FALSE) ;
    }
    
    Col [col].shared1.thickness = 1 ;
    Col [col].shared2.score = 0 ;
    Col [col].shared3.prev = EMPTY ;
    Col [col].shared4.degree_next = EMPTY ;
    }
    
    /* p [0..n_col] no longer needed, used as "head" in subsequent routines */
    
    /* === Scan columns, compute row degrees, and check row indices ========= */
    
    stats [COLAMD_INFO3] = 0 ;    /* number of duplicate or unsorted row indices*/
    
    for (row = 0 ; row < n_row ; row++)
    {
        Row [row].length = 0 ;
        Row [row].shared2.mark = -1 ;
    }
    
    for (col = 0 ; col < n_col ; col++)
    {
        last_row = -1 ;
    
        cp = &A [p [col]] ;
        cp_end = &A [p [col+1]] ;
    
        while (cp < cp_end)
        {
            row = *cp++ ;
    
            /* make sure row indices within range */
            if (row < 0 || row >= n_row)
            {
                stats [COLAMD_STATUS] = COLAMD_ERROR_row_index_out_of_bounds ;
                stats [COLAMD_INFO1] = col ;
                stats [COLAMD_INFO2] = row ;
                stats [COLAMD_INFO3] = n_row ;
                DEBUG0 (("colamd: row %d col %d out of bounds\n", row, col)) ;
                return (FALSE) ;
            }
    
            if (row <= last_row || Row [row].shared2.mark == col)
            {
                /* row index are unsorted or repeated (or both), thus col */
                /* is jumbled.  This is a notice, not an error condition. */
                stats [COLAMD_STATUS] = COLAMD_OK_BUT_JUMBLED ;
                stats [COLAMD_INFO1] = col ;
                stats [COLAMD_INFO2] = row ;
                (stats [COLAMD_INFO3]) ++ ;
                DEBUG1 (("colamd: row %d col %d unsorted/duplicate\n",row,col));
            }
    
            if (Row [row].shared2.mark != col)
            {
                Row [row].length++ ;
            }
            else
            {
                /* this is a repeated entry in the column, */
                /* it will be removed */
                Col [col].length-- ;
            }
    
            /* mark the row as having been seen in this column */
            Row [row].shared2.mark = col ;
    
            last_row = row ;
        }
    }
    
    /* === Compute row pointers ============================================= */
    
    /* row form of the matrix starts directly after the column */
    /* form of matrix in A */
    Row [0].start = p [n_col] ;
    Row [0].shared1.p = Row [0].start ;
    Row [0].shared2.mark = -1 ;
    for (row = 1 ; row < n_row ; row++)
    {
        Row [row].start = Row [row-1].start + Row [row-1].length ;
        Row [row].shared1.p = Row [row].start ;
        Row [row].shared2.mark = -1 ;
    }
    
    /* === Create row form ================================================== */
    
    if (stats [COLAMD_STATUS] == COLAMD_OK_BUT_JUMBLED)
    {
        /* if cols jumbled, watch for repeated row indices */
        for (col = 0 ; col < n_col ; col++)
    {
        cp = &A [p [col]] ;
        cp_end = &A [p [col+1]] ;
        while (cp < cp_end)
        {
        row = *cp++ ;
        if (Row [row].shared2.mark != col)
        {
            A [(Row [row].shared1.p)++] = col ;
            Row [row].shared2.mark = col ;
        }
        }
    }
    else
    {
    /* if cols not jumbled, we don't need the mark (this is faster) */
    for (col = 0 ; col < n_col ; col++)
    {
        cp = &A [p [col]] ;
        cp_end = &A [p [col+1]] ;
        while (cp < cp_end)
        {
        A [(Row [*cp++].shared1.p)++] = col ;
        }
    }
    }
    
    /* === Clear the row marks and set row degrees ========================== */
    
    for (row = 0 ; row < n_row ; row++)
    {
    Row [row].shared2.mark = 0 ;
    Row [row].shared1.degree = Row [row].length ;
    }
    
    /* === See if we need to re-create columns ============================== */
    
    if (stats [COLAMD_STATUS] == COLAMD_OK_BUT_JUMBLED)
    {
    DEBUG0 (("colamd: reconstructing column form, matrix jumbled\n")) ;
    }
    
    
    注释：
    
    
    {
        // 指向列的起始和结束指针
        cp = &A [p [col]] ;
        cp_end = &A [p [col+1]] ;
        // 遍历当前列中的每一行
        while (cp < cp_end)
        {
        // 从当前列中获取行号，并移动指针到下一个行号
        row = *cp++ ;
        // 如果该行未被标记过
        if (Row [row].shared2.mark != col)
        {
            // 将当前列号存入该行的数据中，并增加该行的指针
            A [(Row [row].shared1.p)++] = col ;
            // 标记该行已被处理
            Row [row].shared2.mark = col ;
        }
        }
    }
    else
    {
    /* if cols not jumbled, we don't need the mark (this is faster) */
    // 如果列没有混乱，不需要标记行号（这样更快）
    for (col = 0 ; col < n_col ; col++)
    {
        // 指向列的起始和结束指针
        cp = &A [p [col]] ;
        cp_end = &A [p [col+1]] ;
        // 遍历当前列中的每一行
        while (cp < cp_end)
        {
        // 将当前列号存入行数据中，并增加该行的指针
        A [(Row [*cp++].shared1.p)++] = col ;
        }
    }
    }
    
    /* === Clear the row marks and set row degrees ========================== */
    
    // 清除行的标记并设置行的度数
    for (row = 0 ; row < n_row ; row++)
    {
    // 清除行的标记
    Row [row].shared2.mark = 0 ;
    // 设置行的度数为行的长度
    Row [row].shared1.degree = Row [row].length ;
    }
    
    /* === See if we need to re-create columns ============================== */
    
    // 检查是否需要重新创建列
    if (stats [COLAMD_STATUS] == COLAMD_OK_BUT_JUMBLED)
    {
    // 输出调试信息：重建列形式，矩阵混乱
    DEBUG0 (("colamd: reconstructing column form, matrix jumbled\n")) ;
    }
#ifndef NDEBUG
    /* 如果处于调试模式，则确保列的长度正确 */

    /* 遍历每一列，更新列长度信息到数组 p */
    for (col = 0 ; col < n_col ; col++)
    {
        p [col] = Col [col].length ;
    }

    /* 遍历每一行，更新行相关数据 */
    for (row = 0 ; row < n_row ; row++)
    {
        /* 指向当前行的起始位置 */
        rp = &A [Row [row].start] ;
        /* 指向当前行的结束位置 */
        rp_end = rp + Row [row].length ;
        /* 对当前行的每个元素，更新列长度信息 */
        while (rp < rp_end)
        {
            p [*rp++]-- ;
        }
    }

    /* 确保所有列的长度都为零 */
    for (col = 0 ; col < n_col ; col++)
    {
        ASSERT (p [col] == 0) ;
    }
    /* 此时 p 数组所有元素均为零（与非调试模式下不同） */
#endif /* NDEBUG */

/* === 计算列指针 ========================================= */

/* 矩阵的列形式从 A [0] 开始 */
/* 注意，如果存在重复条目，则列形式和行形式之间可能会有间隙 */
/* 如果有的话，在第一次垃圾收集时会被移除 */
Col [0].start = 0 ;
p [0] = Col [0].start ;

/* 计算每一列的起始位置 */
for (col = 1 ; col < n_col ; col++)
{
    /* 注意，这里的长度是修剪后的列长度，即这些列不会有重复的行索引 */
    Col [col].start = Col [col-1].start + Col [col-1].length ;
    p [col] = Col [col].start ;
}

/* === 重新创建列形式 =========================================== */

/* 根据行形式重建列形式 */
for (row = 0 ; row < n_row ; row++)
{
    rp = &A [Row [row].start] ;
    rp_end = rp + Row [row].length ;
    /* 更新列形式 */
    while (rp < rp_end)
    {
        A [(p [*rp++])++] = row ;
    }
}

/* === 完成。矩阵未被混乱或已不再混乱 ====================== */

return (TRUE) ;
}


/* ========================================================================== */
/* === init_scoring ========================================================= */
/* ========================================================================== */

/*
    删除稠密或空列和行，计算每列的初始分数，并将所有列放入度列表中。不可由用户调用。
*/

PRIVATE void init_scoring
(
    /* === Parameters ======================================================= */

    Int n_row,            /* A 的行数 */
    Int n_col,            /* A 的列数 */
    Colamd_Row Row [],        /* 大小为 n_row+1 的行数组 */
    Colamd_Col Col [],        /* 大小为 n_col+1 的列数组 */
    Int A [],            /* A 的列形式和行形式 */
    Int head [],        /* 大小为 n_col+1 的头部数组 */
    double knobs [COLAMD_KNOBS],/* 参数 */
    Int *p_n_row2,        /* 非稠密、非空行数 */
    Int *p_n_col2,        /* 非稠密、非空列数 */
    Int *p_max_deg        /* 最大行度 */
)
{
    /* === Local variables ================================================== */

    Int c ;            /* 列索引 */
    Int r, row ;        /* 行索引 */
    Int *cp ;            /* 列指针 */
    Int deg ;            /* 行或列的度 */


注释：以上是对给定代码中每一行的详细解释，包括了条件编译部分、变量的更新与计算过程、函数的参数解释以及函数的主要功能说明。
    Int *cp_end ;        /* 指向列的末尾的指针 */
    Int *new_cp ;        /* 新列指针 */
    Int col_length ;        /* 剪枝后列的长度 */
    Int score ;            /* 当前列的分数 */
    Int n_col2 ;        /* 非稠密且非空列的数量 */
    Int n_row2 ;        /* 非稠密且非空行的数量 */
    Int dense_row_count ;    /* 要移除超过此数目条目的行 */
    Int dense_col_count ;    /* 要移除超过此数目条目的列 */
    Int min_score ;        /* 最小的列分数 */
    Int max_deg ;        /* 最大的行度 */
    Int next_col ;        /* 用于添加到度列表中的变量 */
#ifndef NDEBUG
    Int debug_count ;        /* 调试模式下的计数器变量，仅在调试时使用 */
#endif /* NDEBUG */

/* === Extract knobs ==================================================== */

/* 提取控制参数 */

/* Note: if knobs contains a NaN, this is undefined: */
/* 注意：如果控制参数 knobs 包含 NaN，则行为未定义 */
if (knobs [COLAMD_DENSE_ROW] < 0)
{
    /* only remove completely dense rows */
    /* 只移除完全稠密的行 */
    dense_row_count = n_col-1 ;
}
else
{
    dense_row_count = DENSE_DEGREE (knobs [COLAMD_DENSE_ROW], n_col) ;
}

if (knobs [COLAMD_DENSE_COL] < 0)
{
    /* only remove completely dense columns */
    /* 只移除完全稠密的列 */
    dense_col_count = n_row-1 ;
}
else
{
    dense_col_count =
        DENSE_DEGREE (knobs [COLAMD_DENSE_COL], MIN (n_row, n_col)) ;
}

DEBUG1 (("colamd: densecount: %d %d\n", dense_row_count, dense_col_count)) ;

max_deg = 0 ;
n_col2 = n_col ;
n_row2 = n_row ;

/* === Kill empty columns =============================================== */

/* 移除空列 */

/* Put the empty columns at the end in their natural order, so that LU */
/* factorization can proceed as far as possible. */
/* 将空列按照自然顺序放在末尾，以便 LU 分解尽可能进行 */
for (c = n_col-1 ; c >= 0 ; c--)
{
    deg = Col [c].length ;
    if (deg == 0)
    {
        /* this is a empty column, kill and order it last */
        /* 这是一个空列，将其移除并将其排在最后 */
        Col [c].shared2.order = --n_col2 ;
        KILL_PRINCIPAL_COL (c) ;
    }
}
DEBUG1 (("colamd: null columns killed: %d\n", n_col - n_col2)) ;

/* === Kill dense columns =============================================== */

/* 移除稠密列 */

/* Put the dense columns at the end, in their natural order */
/* 将稠密列按照自然顺序放在末尾 */
for (c = n_col-1 ; c >= 0 ; c--)
{
    /* skip any dead columns */
    /* 跳过任何已经移除的列 */
    if (COL_IS_DEAD (c))
    {
        continue ;
    }
    deg = Col [c].length ;
    if (deg > dense_col_count)
    {
        /* this is a dense column, kill and order it last */
        /* 这是一个稠密列，将其移除并将其排在最后 */
        Col [c].shared2.order = --n_col2 ;
        /* decrement the row degrees */
        /* 减少相关行的度数 */
        cp = &A [Col [c].start] ;
        cp_end = cp + Col [c].length ;
        while (cp < cp_end)
        {
            Row [*cp++].shared1.degree-- ;
        }
        KILL_PRINCIPAL_COL (c) ;
    }
}
DEBUG1 (("colamd: Dense and null columns killed: %d\n", n_col - n_col2)) ;

/* === Kill dense and empty rows ======================================== */

/* 移除稠密和空行 */

for (r = 0 ; r < n_row ; r++)
{
    deg = Row [r].shared1.degree ;
    ASSERT (deg >= 0 && deg <= n_col) ;
    if (deg > dense_row_count || deg == 0)
    {
        /* kill a dense or empty row */
        /* 移除一个稠密或空行 */
        KILL_ROW (r) ;
        --n_row2 ;
    }
    else
    {
        /* keep track of max degree of remaining rows */
        /* 更新剩余行的最大度数 */
        max_deg = MAX (max_deg, deg) ;
    }
}
DEBUG1 (("colamd: Dense and null rows killed: %d\n", n_row - n_row2)) ;

/* === Compute initial column scores ==================================== */

/* 计算初始列分数 */

/* At this point the row degrees are accurate.  They reflect the number */
/* of "live" (non-dense) columns in each row.  No empty rows exist. */
/* Some "live" columns may contain only dead rows, however.  These are */
/* 此时行的度数已经是准确的。它们反映了每行中“活跃”（非稠密）列的数量。没有空行存在。 */
/* 但是，一些“活跃”列可能仅包含已移除的行。这些列是 */
    /* now find the initial matlab score for each column */
    /* 现在计算每列的初始 MATLAB 分数 */

    for (c = n_col-1 ; c >= 0 ; c--)
    {
        /* skip dead column */
        /* 跳过已标记为死亡的列 */
        if (COL_IS_DEAD (c))
        {
            continue ;
        }
        score = 0 ;
        cp = &A [Col [c].start] ;
        new_cp = cp ;
        cp_end = cp + Col [c].length ;
        while (cp < cp_end)
        {
            /* get a row */
            /* 获取一行 */
            row = *cp++ ;
            /* skip if dead */
            /* 如果行已标记为死亡则跳过 */
            if (ROW_IS_DEAD (row))
            {
                continue ;
            }
            /* compact the column */
            /* 压缩列 */
            *new_cp++ = row ;
            /* add row's external degree */
            /* 添加行的外部度量 */
            score += Row [row].shared1.degree - 1 ;
            /* guard against integer overflow */
            /* 防止整数溢出 */
            score = MIN (score, n_col) ;
        }
        /* determine pruned column length */
        /* 确定修剪后的列长度 */
        col_length = (Int) (new_cp - &A [Col [c].start]) ;
        if (col_length == 0)
        {
            /* a newly-made null column (all rows in this col are "dense" */
            /* and have already been killed) */
            /* 新生成的空列（此列中的所有行均为“密集”并且已经被标记为死亡） */
            DEBUG2 (("Newly null killed: %d\n", c)) ;
            Col [c].shared2.order = --n_col2 ;
            KILL_PRINCIPAL_COL (c) ;
        }
        else
        {
            /* set column length and set score */
            /* 设置列长度和分数 */
            ASSERT (score >= 0) ;
            ASSERT (score <= n_col) ;
            Col [c].length = col_length ;
            Col [c].shared2.score = score ;
        }
    }

    DEBUG1 (("colamd: Dense, null, and newly-null columns killed: %d\n",
        n_col-n_col2)) ;

    /* At this point, all empty rows and columns are dead.  All live columns */
    /* are "clean" (containing no dead rows) and simplicial (no supercolumns */
    /* yet).  Rows may contain dead columns, but all live rows contain at */
    /* least one live column. */
    /* 此时，所有空行和列都已被标记为死亡。所有活跃列 */
    /* 都是“干净的”（不包含死亡行）且单列（没有超列 */
    /* ）。行可能包含死亡列，但所有活跃行都至少包含 */
    /* 一个活跃列。 */
#ifndef NDEBUG
    debug_structures (n_row, n_col, Row, Col, A, n_col2) ;
#endif /* NDEBUG */

/* === Initialize degree lists ========================================== */

#ifndef NDEBUG
    debug_count = 0 ;
#endif /* NDEBUG */

/* clear the hash buckets */
for (c = 0 ; c <= n_col ; c++)
{
    head [c] = EMPTY ;
}
min_score = n_col ;

/* place in reverse order, so low column indices are at the front */
/* of the lists.  This is to encourage natural tie-breaking */
for (c = n_col-1 ; c >= 0 ; c--)
{
    /* only add principal columns to degree lists */
    if (COL_IS_ALIVE (c))
    {
        DEBUG4 (("place %d score %d minscore %d ncol %d\n",
        c, Col [c].shared2.score, min_score, n_col)) ;

        /* === Add columns score to DList =============================== */

        score = Col [c].shared2.score ;

        ASSERT (min_score >= 0) ;
        ASSERT (min_score <= n_col) ;
        ASSERT (score >= 0) ;
        ASSERT (score <= n_col) ;
        ASSERT (head [score] >= EMPTY) ;

        /* now add this column to dList at proper score location */
        next_col = head [score] ;
        Col [c].shared3.prev = EMPTY ;
        Col [c].shared4.degree_next = next_col ;

        /* if there already was a column with the same score, set its */
        /* previous pointer to this new column */
        if (next_col != EMPTY)
        {
            Col [next_col].shared3.prev = c ;
        }
        head [score] = c ;

        /* see if this score is less than current min */
        min_score = MIN (min_score, score) ;

#ifndef NDEBUG
        debug_count++ ;
#endif /* NDEBUG */

    }
}

#ifndef NDEBUG
DEBUG1 (("colamd: Live cols %d out of %d, non-princ: %d\n",
debug_count, n_col, n_col-debug_count)) ;
ASSERT (debug_count == n_col2) ;
debug_deg_lists (n_row, n_col, Row, Col, head, min_score, n_col2, max_deg) ;
#endif /* NDEBUG */

/* === Return number of remaining columns, and max row degree =========== */

*p_n_col2 = n_col2 ;
*p_n_row2 = n_row2 ;
*p_max_deg = max_deg ;
}


注释：

#ifndef NDEBUG
    // 在调试模式下，调用 debug_structures 函数，传入相关参数进行调试
    debug_structures (n_row, n_col, Row, Col, A, n_col2) ;
#endif /* NDEBUG */

/* === Initialize degree lists ========================================== */

#ifndef NDEBUG
    // 在调试模式下，初始化 debug_count 为 0
    debug_count = 0 ;
#endif /* NDEBUG */

/* clear the hash buckets */
// 清空哈希桶
for (c = 0 ; c <= n_col ; c++)
{
    head [c] = EMPTY ;  // 将 head 数组中的每个元素设为 EMPTY
}
min_score = n_col ;  // 初始化 min_score 为 n_col

/* place in reverse order, so low column indices are at the front */
/* of the lists.  This is to encourage natural tie-breaking */
// 将列索引按逆序排列，使得低列索引在列表前部，以便进行自然的决策
for (c = n_col-1 ; c >= 0 ; c--)
{
    /* only add principal columns to degree lists */
    // 只将主要列添加到度数列表中
    if (COL_IS_ALIVE (c))
    {
        DEBUG4 (("place %d score %d minscore %d ncol %d\n",
        c, Col [c].shared2.score, min_score, n_col)) ;

        /* === Add columns score to DList =============================== */

        score = Col [c].shared2.score ;  // 获取列 c 的分数

        ASSERT (min_score >= 0) ;  // 断言 min_score 大于等于 0
        ASSERT (min_score <= n_col) ;  // 断言 min_score 小于等于 n_col
        ASSERT (score >= 0) ;  // 断言 score 大于等于 0
        ASSERT (score <= n_col) ;  // 断言 score 小于等于 n_col
        ASSERT (head [score] >= EMPTY) ;  // 断言 head[score] 大于等于 EMPTY

        /* now add this column to dList at proper score location */
        // 将该列按照分数添加到 dList 的适当位置
        next_col = head [score] ;
        Col [c].shared3.prev = EMPTY ;  // 将该列的前驱设置为 EMPTY
        Col [c].shared4.degree_next = next_col ;  // 将该列的后继设置为 next_col

        /* if there already was a column with the same score, set its */
        /* previous pointer to this new column */
        // 如果已经有分数相同的列存在，则将其前驱指针设置为这一新列
        if (next_col != EMPTY)
        {
            Col [next_col].shared3.prev = c ;
        }
        head [score] = c ;  // 更新 head[score] 为当前列 c

        /* see if this score is less than current min */
        // 检查该分数是否小于当前 min_score
        min_score = MIN (min_score, score) ;

#ifndef NDEBUG
        debug_count++ ;  // 调试计数加一
#endif /* NDEBUG */

    }
}

#ifndef NDEBUG
// 在调试模式下，输出剩余的活跃列数和非主列的数量
DEBUG1 (("colamd: Live cols %d out of %d, non-princ: %d\n",
debug_count, n_col, n_col-debug_count)) ;
ASSERT (debug_count == n_col2) ;  // 断言调试计数与 n_col2 相等
debug_deg_lists (n_row, n_col, Row, Col, head, min_score, n_col2, max_deg) ;  // 调用 debug_deg_lists 函数进行调试
#endif /* NDEBUG */

/* === Return number of remaining columns, and max row degree =========== */

*p_n_col2 = n_col2 ;  // 更新剩余列数的指针
*p_n_row2 = n_row2 ;  // 更新剩余行数的指针
*p_max_deg = max_deg ;  // 更新最大行度的指针
}
    Int A [],            /* A 是一个整数数组，用于存储矩阵 A 的列形式和行形式 */
    Int head [],        /* head 是一个整数数组，大小为 n_col+1，用于存储列形式 A 的头指针信息 */
    Int n_col2,            /* n_col2 是一个整数，表示需要排序的剩余列数 */
    Int max_deg,        /* max_deg 是一个整数，表示最大的行度 */
    Int pfree,            /* pfree 是一个整数，表示第一个空闲插槽的索引（在进入函数时为 2*nnz） */
    Int aggressive        /* aggressive 是一个整数，可能表示某种行为或策略的标志位 */
)
{
    /* === Local variables ================================================== */

    Int k ;            /* 当前的主元排序步骤 */
    Int pivot_col ;        /* 当前的主元列 */
    Int *cp ;            /* 列指针 */
    Int *rp ;            /* 行指针 */
    Int pivot_row ;        /* 当前的主元行 */
    Int *new_cp ;        /* 修改后的列指针 */
    Int *new_rp ;        /* 修改后的行指针 */
    Int pivot_row_start ;    /* 主元行的起始指针 */
    Int pivot_row_degree ;    /* 主元行中的列数 */
    Int pivot_row_length ;    /* 主元行中的超列数 */
    Int pivot_col_score ;    /* 主元列的分数 */
    Int needed_memory ;        /* 主元行所需的空闲空间 */
    Int *cp_end ;        /* 列的末尾指针 */
    Int *rp_end ;        /* 行的末尾指针 */
    Int row ;            /* 行索引 */
    Int col ;            /* 列索引 */
    Int max_score ;        /* 最大可能的分数 */
    Int cur_score ;        /* 当前列的分数 */
    unsigned Int hash ;        /* 超节点检测的哈希值 */
    Int head_column ;        /* 哈希桶的头部 */
    Int first_col ;        /* 哈希桶中的第一列 */
    Int tag_mark ;        /* 标记数组的标记值 */
    Int row_mark ;        /* 行[row].shared2.mark */
    Int set_difference ;    /* 行与主元行的集合差大小 */
    Int min_score ;        /* 最小列分数 */
    Int col_thickness ;        /* 超列中的列数（"厚度"） */
    Int max_mark ;        /* 标记值的最大值 */
    Int pivot_col_thickness ;    /* 主元列所表示的列数 */
    Int prev_col ;        /* Dlist 操作中使用的前一列 */
    Int next_col ;        /* Dlist 操作中使用的下一列 */
    Int ngarbage ;        /* 执行的垃圾收集次数 */

#ifndef NDEBUG
    Int debug_d ;        /* 调试循环计数器 */
    Int debug_step = 0 ;    /* 调试步骤计数器 */
#endif /* NDEBUG */

    /* === Initialization and clear mark ==================================== */

    max_mark = INT_MAX - n_col ;    /* INT_MAX 在 <limits.h> 中定义 */
    tag_mark = clear_mark (0, max_mark, n_row, Row) ;
    min_score = 0 ;
    ngarbage = 0 ;
    DEBUG1 (("colamd: Ordering, n_col2=%d\n", n_col2)) ;

    /* === Order the columns ================================================ */

    for (k = 0 ; k < n_col2 ; /* 'k' 在下面递增 */)
    {

#ifndef NDEBUG
    if (debug_step % 100 == 0)
    {
        DEBUG2 (("\n...       Step k: %d out of n_col2: %d\n", k, n_col2)) ;
    }
    else
    {
        DEBUG3 (("\n----------Step k: %d out of n_col2: %d\n", k, n_col2)) ;
    }
    debug_step++ ;
    debug_deg_lists (n_row, n_col, Row, Col, head,
        min_score, n_col2-k, max_deg) ;
    debug_matrix (n_row, n_col, Row, Col, A) ;
#endif /* NDEBUG */
    /* === Select pivot column, and order it ============================ */

    /* 确保度列表不为空 */
    ASSERT(min_score >= 0);
    /* 确保最小分数不超过列数 */
    ASSERT(min_score <= n_col);
    /* 确保最小分数所在的列头索引有效 */
    ASSERT(head[min_score] >= EMPTY);
#ifndef NDEBUG
    for (debug_d = 0 ; debug_d < min_score ; debug_d++)
    {
        ASSERT (head [debug_d] == EMPTY) ;
    }
#endif /* NDEBUG */

/* 从最小度数列表的头部获取主元列 */
while (head [min_score] == EMPTY && min_score < n_col)
{
    min_score++ ;
}
// 获取主元列的列索引
pivot_col = head [min_score] ;
// 断言主元列索引在有效范围内
ASSERT (pivot_col >= 0 && pivot_col <= n_col) ;
// 获取主元列的下一个列索引
next_col = Col [pivot_col].shared4.degree_next ;
// 更新最小度数列表的头部
head [min_score] = next_col ;
// 如果存在下一个列，更新其前一个列索引为空
if (next_col != EMPTY)
{
    Col [next_col].shared3.prev = EMPTY ;
}

ASSERT (COL_IS_ALIVE (pivot_col)) ;

/* 记录主元列的分数，用于碎片整理检查 */
pivot_col_score = Col [pivot_col].shared2.score ;

/* 将主元列标记为第 k 个列 */
Col [pivot_col].shared2.order = k ;

/* 根据主元列的厚度增加排序计数 */
pivot_col_thickness = Col [pivot_col].shared1.thickness ;
k += pivot_col_thickness ;
ASSERT (pivot_col_thickness > 0) ;
DEBUG3 (("Pivot col: %d thick %d\n", pivot_col, pivot_col_thickness)) ;

/* === 如果需要，进行垃圾收集 ============================= */

needed_memory = MIN (pivot_col_score, n_col - k) ;
if (pfree + needed_memory >= Alen)
{
    // 执行垃圾收集
    pfree = garbage_collection (n_row, n_col, Row, Col, A, &A [pfree]) ;
    ngarbage++ ;
    // 垃圾收集后确保有足够的空间
    ASSERT (pfree + needed_memory < Alen) ;
    // 垃圾收集清除了 Row[].shared2.mark 数组的标记
    tag_mark = clear_mark (0, max_mark, n_row, Row) ;

#ifndef NDEBUG
    // 在调试模式下打印矩阵内容
    debug_matrix (n_row, n_col, Row, Col, A) ;
#endif /* NDEBUG */
}

/* === 计算主元行模式 ==================================== */

// 设置主元行的起始位置
pivot_row_start = pfree ;

// 初始化新行的度数为零
pivot_row_degree = 0 ;

// 将主元列标记为已访问，以确保它不会包含在合并的主元行中
Col [pivot_col].shared1.thickness = -pivot_col_thickness ;

// 主元行是主元列模式中所有行的并集
cp = &A [Col [pivot_col].start] ;
cp_end = cp + Col [pivot_col].length ;
while (cp < cp_end)
    {
        /* 获取一行 */
        row = *cp++ ;
        // 输出调试信息，显示当前行的模式和状态
        DEBUG4 (("Pivot col pattern %d %d\n", ROW_IS_ALIVE (row), row)) ;
        /* 如果行是活跃的，则继续 */
        if (ROW_IS_ALIVE (row))
        {
            rp = &A [Row [row].start] ;
            rp_end = rp + Row [row].length ;
            while (rp < rp_end)
            {
                /* 获取一列 */
                col = *rp++ ;
                /* 如果列的厚度大于零且列是活跃且未被标记 */
                col_thickness = Col [col].shared1.thickness ;
                if (col_thickness > 0 && COL_IS_ALIVE (col))
                {
                    /* 在主元行中标记列 */
                    Col [col].shared1.thickness = -col_thickness ;
                    ASSERT (pfree < Alen) ;
                    /* 将列放置在主元行中 */
                    A [pfree++] = col ;
                    pivot_row_degree += col_thickness ;
                }
            }
        }
    }
    
    /* 清除主元列上的标记 */
    Col [pivot_col].shared1.thickness = pivot_col_thickness ;
    max_deg = MAX (max_deg, pivot_row_degree) ;
#ifndef NDEBUG
    DEBUG3 (("check2\n")) ;
    debug_mark (n_row, Row, tag_mark, max_mark) ;
#endif /* NDEBUG */

/* === Kill all rows used to construct pivot row ==================== */

/* 临时关闭调试模式下输出信息。遍历列中所有行索引，将它们标记为已删除状态，
   同时输出调试信息以显示哪些行被删除。 */
cp = &A [Col [pivot_col].start] ;
cp_end = cp + Col [pivot_col].length ;
while (cp < cp_end)
{
    /* 如果要删除的行已经是无效行，可能是重复操作。输出调试信息表明正在删除该行。 */
    row = *cp++ ;
    DEBUG3 (("Kill row in pivot col: %d\n", row)) ;
    KILL_ROW (row) ;
}

/* === Select a row index to use as the new pivot row =============== */

pivot_row_length = pfree - pivot_row_start ;
if (pivot_row_length > 0)
{
    /* 从列的第一个行索引中任意选择一个作为主元行。输出调试信息显示选定的主元行。 */
    pivot_row = A [Col [pivot_col].start] ;
    DEBUG3 (("Pivotal row is %d\n", pivot_row)) ;
}
else
{
    /* 主元行长度为零，因此不存在主元行。输出断言验证主元行长度确实为零。 */
    pivot_row = EMPTY ;
    ASSERT (pivot_row_length == 0) ;
}
ASSERT (Col [pivot_col].length > 0 || pivot_row_length == 0) ;

/* === Approximate degree computation =============================== */

/* 开始计算近似度量的阶数。列分数是主元行长度的总和，加上每行与主元行的差集大小，
   减去主元行本身的模式。外部列自身不包含在列分数中（因此使用近似外部阶数）。 */

/* 下面的代码计算集合差异并加总它们所需的时间与正在扫描的数据结构的大小成正比，
   即主元行中每列的大小总和。因此，计算列分数的摊销时间与该列的大小成正比。
   这里的大小指的是列的长度，或者说该列中行索引的数量。在输入到colamd时，列中行索引数量是单调递增的。 */

/* === Compute set differences ====================================== */

DEBUG3 (("** Computing set differences phase. **\n")) ;

/* 主元行当前为无效状态，稍后将重新激活。 */

DEBUG3 (("Pivot row: ")) ;
/* 遍历主元行中的每一列 */
rp = &A [pivot_row_start] ;
rp_end = rp + pivot_row_length ;
while (rp < rp_end)
    {
        // 从rp指针指向的列中读取列索引，然后自增rp指针
        col = *rp++ ;
        // 断言：确保列col仍然存在并且不是主元列pivot_col
        ASSERT (COL_IS_ALIVE (col) && col != pivot_col) ;
        // 调试输出：打印当前处理的列索引
        DEBUG3 (("Col: %d\n", col)) ;
    
        /* 清除用于构建主元行模式的标记 */
    
        // 获取列厚度并将其置为负数，表示清除操作
        col_thickness = -Col [col].shared1.thickness ;
        // 断言：确保列厚度大于0
        ASSERT (col_thickness > 0) ;
        // 更新列厚度
        Col [col].shared1.thickness = col_thickness ;
    
        /* === 从度列表中移除列 =========================== */
    
        // 获取当前列的分数
        cur_score = Col [col].shared2.score ;
        // 获取前一列和后一列的索引
        prev_col = Col [col].shared3.prev ;
        next_col = Col [col].shared4.degree_next ;
        // 断言：确保当前分数在合理范围内
        ASSERT (cur_score >= 0) ;
        ASSERT (cur_score <= n_col) ;
        ASSERT (cur_score >= EMPTY) ;
        // 如果前一列为空
        if (prev_col == EMPTY)
        {
            // 更新当前分数对应的头部索引
            head [cur_score] = next_col ;
        }
        else
        {
            // 更新前一列的后续列索引
            Col [prev_col].shared4.degree_next = next_col ;
        }
        // 如果下一列不为空
        if (next_col != EMPTY)
        {
            // 更新下一列的前一列索引
            Col [next_col].shared3.prev = prev_col ;
        }
    
        /* === 扫描该列中的行 ========================================== */
    
        // 指向该列在矩阵中的起始位置
        cp = &A [Col [col].start] ;
        // 指向该列在矩阵中的结束位置
        cp_end = cp + Col [col].length ;
        // 遍历该列中的每一行
        while (cp < cp_end)
        {
            /* 获取一行 */
            row = *cp++ ;
            // 获取行标记
            row_mark = Row [row].shared2.mark ;
            // 如果行已经标记为死亡，则跳过
            if (ROW_IS_MARKED_DEAD (row_mark))
            {
                continue ;
            }
            // 断言：确保行不是主元行pivot_row
            ASSERT (row != pivot_row) ;
            // 计算行标记与标记标记之间的差值
            set_difference = row_mark - tag_mark ;
            // 如果差值小于0，说明该行还未被看到
            if (set_difference < 0)
            {
                // 断言：确保行的度数不超过最大度数max_deg
                ASSERT (Row [row].shared1.degree <= max_deg) ;
                // 设置差值为行的度数
                set_difference = Row [row].shared1.degree ;
            }
            // 从行的差值中减去列的厚度
            set_difference -= col_thickness ;
            // 断言：确保差值大于等于0
            ASSERT (set_difference >= 0) ;
            // 如果差值为0且采取积极吸收策略
            if (set_difference == 0 && aggressive)
            {
                // 调试输出：打印采用积极吸收的行索引
                DEBUG3 (("aggressive absorption. Row: %d\n", row)) ;
                // 杀死该行（标记为死亡）
                KILL_ROW (row) ;
            }
            else
            {
                /* 保存新的标记 */
                // 更新行的标记为差值加上标记标记
                Row [row].shared2.mark = set_difference + tag_mark ;
            }
        }
    }
#ifndef NDEBUG
    // 如果未定义 NDEBUG 宏，则执行调试函数 debug_deg_lists，输出调试信息
    debug_deg_lists(n_row, n_col, Row, Col, head,
                    min_score, n_col2 - k - pivot_row_degree, max_deg);
#endif /* NDEBUG */

/* === Add up set differences for each column ======================= */

// 输出调试信息，表示正在执行“添加集合差异阶段”
DEBUG3(("** Adding set differences phase. **\n"));

/* for each column in pivot row */
// 初始化指向当前行起始位置的指针 rp
rp = &A[pivot_row_start];
// 指向当前行结束位置的指针 rp_end
rp_end = rp + pivot_row_length;
// 遍历当前行的每一列
while (rp < rp_end)
    {
        /* 获取一个列 */
        col = *rp++ ;
        ASSERT (COL_IS_ALIVE (col) && col != pivot_col) ;
        hash = 0 ;
        cur_score = 0 ;
        cp = &A [Col [col].start] ;
        /* 紧缩列 */
        new_cp = cp ;
        cp_end = cp + Col [col].length ;
    
        DEBUG4 (("Adding set diffs for Col: %d.\n", col)) ;
    
        while (cp < cp_end)
        {
            /* 获取一个行 */
            row = *cp++ ;
            ASSERT(row >= 0 && row < n_row) ;
            row_mark = Row [row].shared2.mark ;
            /* 如果行已死亡，则跳过 */
            if (ROW_IS_MARKED_DEAD (row_mark))
            {
                DEBUG4 ((" Row %d, dead\n", row)) ;
                continue ;
            }
            DEBUG4 ((" Row %d, set diff %d\n", row, row_mark-tag_mark));
            ASSERT (row_mark >= tag_mark) ;
            /* 紧缩列 */
            *new_cp++ = row ;
            /* 计算哈希函数 */
            hash += row ;
            /* 添加集合差分 */
            cur_score += row_mark - tag_mark ;
            /* 整数溢出处理 */
            cur_score = MIN (cur_score, n_col) ;
        }
    
        /* 重新计算列的长度 */
        Col [col].length = (Int) (new_cp - &A [Col [col].start]) ;
    
        /* === 进一步的大规模消除 ================================= */
    
        if (Col [col].length == 0)
        {
            DEBUG4 (("further mass elimination. Col: %d\n", col)) ;
            /* 此列中只剩下主元行 */
            KILL_PRINCIPAL_COL (col) ;
            pivot_row_degree -= Col [col].shared1.thickness ;
            ASSERT (pivot_row_degree >= 0) ;
            /* 对其排序 */
            Col [col].shared2.order = k ;
            /* 按列厚度增加顺序计数 */
            k += Col [col].shared1.thickness ;
        }
        else
        {
            /* === 准备超列检测 ==================== */
    
            DEBUG4 (("Preparing supercol detection for Col: %d.\n", col)) ;
    
            /* 保存当前得分 */
            Col [col].shared2.score = cur_score ;
    
            /* 将列添加到哈希表中，用于超列检测 */
            hash %= n_col + 1 ;
    
            DEBUG4 ((" Hash = %d, n_col = %d.\n", hash, n_col)) ;
            ASSERT (((Int) hash) <= n_col) ;
    
            head_column = head [hash] ;
            if (head_column > EMPTY)
            {
                /* degree list "hash" 非空，使用度列表的第一个列的前驱 (shared3) 作为哈希桶的头部 */
                first_col = Col [head_column].shared3.headhash ;
                Col [head_column].shared3.headhash = col ;
            }
            else
            {
                /* degree list "hash" 为空，使用 head 作为哈希桶的头部 */
                first_col = - (head_column + 2) ;
                head [hash] = - (col + 2) ;
            }
            Col [col].shared4.hash_next = first_col ;
    
            /* 在 Col [col].shared3.hash 中保存哈希函数 */
            Col [col].shared3.hash = (Int) hash ;
            ASSERT (COL_IS_ALIVE (col)) ;
        }
    }
    
    /* 现在已计算出大致的外部列度 */
    /* === Supercolumn detection ======================================== */

    // 打印调试信息，标志进入超列检测阶段
    DEBUG3 (("** Supercolumn detection phase. **\n")) ;

    // 调用超列检测函数 detect_super_cols，开始执行超列检测
    detect_super_cols (
#ifndef NDEBUG
        n_col, Row,
#endif /* NDEBUG */

        Col, A, head, pivot_row_start, pivot_row_length) ;

    /* === Kill the pivotal column ====================================== */

    KILL_PRINCIPAL_COL (pivot_col) ;
    // 杀死主列，将主列标记为不可用状态

    /* === Clear mark =================================================== */

    tag_mark = clear_mark (tag_mark+max_deg+1, max_mark, n_row, Row) ;
    // 清除标记，重置标记数组，以便重新使用

#ifndef NDEBUG
    DEBUG3 (("check3\n")) ;
    debug_mark (n_row, Row, tag_mark, max_mark) ;
#endif /* NDEBUG */
    // 调试模式下，输出调试信息并检查标记状态

    /* === Finalize the new pivot row, and column scores ================ */

    DEBUG3 (("** Finalize scores phase. **\n")) ;
    // 输出调试信息，指示进入得分最终化阶段

    /* for each column in pivot row */
    rp = &A [pivot_row_start] ;
    // rp指向当前行的起始列
    /* compact the pivot row */
    new_rp = rp ;
    rp_end = rp + pivot_row_length ;
    // rp_end指向当前行的结束位置
    while (rp < rp_end)
    {
        col = *rp++ ;
        // col是当前处理的列
        /* skip dead columns */
        if (COL_IS_DEAD (col))
        {
        continue ;
        }
        *new_rp++ = col ;
        // 将有效的列移到新位置，压缩当前行

        /* add new pivot row to column */
        A [Col [col].start + (Col [col].length++)] = pivot_row ;
        // 将新的主行元素添加到当前列中

        /* retrieve score so far and add on pivot row's degree. */
        /* (we wait until here for this in case the pivot */
        /* row's degree was reduced due to mass elimination). */
        cur_score = Col [col].shared2.score + pivot_row_degree ;
        // 计算当前列的得分，考虑主行的度数

        /* calculate the max possible score as the number of */
        /* external columns minus the 'k' value minus the */
        /* columns thickness */
        max_score = n_col - k - Col [col].shared1.thickness ;
        // 计算最大可能得分，考虑外部列数、'k'值和列的厚度

        /* make the score the external degree of the union-of-rows */
        cur_score -= Col [col].shared1.thickness ;
        // 更新得分，考虑行的并集的外部度数

        /* make sure score is less or equal than the max score */
        cur_score = MIN (cur_score, max_score) ;
        ASSERT (cur_score >= 0) ;
        // 确保得分不超过最大得分，并进行断言检查

        /* store updated score */
        Col [col].shared2.score = cur_score ;
        // 更新列的得分

        /* === Place column back in degree list ========================= */

        ASSERT (min_score >= 0) ;
        ASSERT (min_score <= n_col) ;
        ASSERT (cur_score >= 0) ;
        ASSERT (cur_score <= n_col) ;
        ASSERT (head [cur_score] >= EMPTY) ;
        next_col = head [cur_score] ;
        Col [col].shared4.degree_next = next_col ;
        Col [col].shared3.prev = EMPTY ;
        if (next_col != EMPTY)
        {
        Col [next_col].shared3.prev = col ;
        }
        head [cur_score] = col ;
        // 将列重新放回度列表中，并进行断言检查

        /* see if this score is less than current min */
        min_score = MIN (min_score, cur_score) ;
        // 更新最小得分

    }

#ifndef NDEBUG
    debug_deg_lists (n_row, n_col, Row, Col, head,
        min_score, n_col2-k, max_deg) ;
#endif /* NDEBUG */
    // 调试模式下，输出度列表的调试信息

    /* === Resurrect the new pivot row ================================== */

    if (pivot_row_degree > 0)
    // 如果新主行的度数大于0，则表示主行仍然有效，可以继续使用
    {
        /* 更新主元行长度以反映在超列检测和大规模消除过程中被删除的任何列 */
        /* 在超列检测和大规模消除过程中，更新主元行的起始位置 */
        Row[pivot_row].start = pivot_row_start ;
        /* 计算新的主元行长度，基于新的A数组的地址 */
        Row[pivot_row].length = (Int) (new_rp - &A[pivot_row_start]) ;
        /* 断言主元行的长度大于0 */
        ASSERT(Row[pivot_row].length > 0) ;
        /* 设置主元行的度数 */
        Row[pivot_row].shared1.degree = pivot_row_degree ;
        /* 清除主元行的标记 */
        Row[pivot_row].shared2.mark = 0 ;
        /* 主元行不再是死行 */

        DEBUG1(("Resurrect Pivot_row %d deg: %d\n",
            pivot_row, pivot_row_degree)) ;
    }
    }

    /* === 所有的主列现在已经被排序 ====================== */

    /* 返回ngarbage作为结果 */
    return (ngarbage) ;
/* ========================================================================== */
/* === order_children ======================================================= */
/* ========================================================================== */

/*
    find_ordering函数已经按照原则列（supercolumns的代表）排序了所有主列。
    非主列尚未排序。这个函数通过沿着父节点树（一个列是其吸收它的列的子节点）来排序这些列。
    最终的排列向量被放置在p [0 ... n_col-1]中，其中p [0]是第一列，p [n_col-1]是最后一列。
    乍一看似乎不是这样，但请确信，这个函数的时间复杂度是与列数成线性关系的。
    尽管一开始不明显，但这个函数的时间复杂度是O(n_col)，也就是与列数成线性关系的。
    不可由用户调用。
*/

PRIVATE void order_children
(
    /* === Parameters ======================================================= */

    Int n_col,            /* A的列数 */
    Colamd_Col Col [],        /* 大小为n_col+1的数组 */
    Int p []            /* p [0 ... n_col-1]是列的排列 */
)
{
    /* === Local variables ================================================== */

    Int i ;            /* 所有列的循环计数器 */
    Int c ;            /* 列索引 */
    Int parent ;        /* 列的父节点索引 */
    Int order ;            /* 列的排序顺序 */

    /* === Order each non-principal column ================================== */

    for (i = 0 ; i < n_col ; i++)
    {
    /* 找到一个未排序的非主列 */
    ASSERT (COL_IS_DEAD (i)) ;
    if (!COL_IS_DEAD_PRINCIPAL (i) && Col [i].shared2.order == EMPTY)
    {
        parent = i ;
        /* 一旦找到，找到其主要父节点 */
        do
        {
        parent = Col [parent].shared1.parent ;
        } while (!COL_IS_DEAD_PRINCIPAL (parent)) ;

        /* 现在，按照路径顺序所有未排序的非主列。同时进行树的折叠 */
        c = i ;
        /* 获取父节点的排序顺序 */
        order = Col [parent].shared2.order ;

        do
        {
        ASSERT (Col [c].shared2.order == EMPTY) ;

        /* 对这一列进行排序 */
        Col [c].shared2.order = order++ ;
        /* 折叠树 */
        Col [c].shared1.parent = parent ;

        /* 获取该列的直接父节点 */
        c = Col [c].shared1.parent ;

        /* 继续，直到我们遇到一个已排序的列。保证上面一个已排序的列上方没有未排序的列 */
        } while (Col [c].shared2.order == EMPTY) ;

        /* 重新对这个组的super_col父节点进行排序，使其顺序最大 */
        Col [parent].shared2.order = order ;
    }
    }

    /* === Generate the permutation ========================================= */
}
    # 循环遍历列索引，从0到n_col-1
    for (c = 0 ; c < n_col ; c++)
    {
        # 将列的顺序索引作为键，列索引c作为对应的值，存入数组p中
        p [Col [c].shared2.order] = c ;
    }
/* ========================================================================== */
/* === detect_super_cols ==================================================== */
/* ========================================================================== */

/*
    Detects supercolumns by finding matches between columns in the hash buckets.
    Check amongst columns in the set A [row_start ... row_start + row_length-1].
    The columns under consideration are currently *not* in the degree lists,
    and have already been placed in the hash buckets.

    The hash bucket for columns whose hash function is equal to h is stored
    as follows:

    if head [h] is >= 0, then head [h] contains a degree list, so:

        head [h] is the first column in degree bucket h.
        Col [head [h]].headhash gives the first column in hash bucket h.

    otherwise, the degree list is empty, and:

        -(head [h] + 2) is the first column in hash bucket h.

    For a column c in a hash bucket, Col [c].shared3.prev is NOT a "previous
    column" pointer.  Col [c].shared3.hash is used instead as the hash number
    for that column.  The value of Col [c].shared4.hash_next is the next column
    in the same hash bucket.

    Assuming no, or "few" hash collisions, the time taken by this routine is
    linear in the sum of the sizes (lengths) of each column whose score has
    just been computed in the approximate degree computation.
    Not user-callable.
*/

PRIVATE void detect_super_cols
(
    /* === Parameters ======================================================= */

#ifndef NDEBUG
    /* these two parameters are only needed when debugging is enabled: */
    Int n_col,            /* number of columns of A */
    Colamd_Row Row [],        /* of size n_row+1 */
#endif /* NDEBUG */

    Colamd_Col Col [],        /* of size n_col+1 */
    Int A [],            /* row indices of A */
    Int head [],        /* head of degree lists and hash buckets */
    Int row_start,        /* pointer to set of columns to check */
    Int row_length        /* number of columns to check */
)
{
    /* === Local variables ================================================== */

    Int hash ;            /* hash value for a column */
    Int *rp ;            /* pointer to a row */
    Int c ;            /* a column index */
    Int super_c ;        /* column index of the column to absorb into */
    Int *cp1 ;            /* column pointer for column super_c */
    Int *cp2 ;            /* column pointer for column c */
    Int length ;        /* length of column super_c */
    Int prev_c ;        /* column preceding c in hash bucket */
    Int i ;            /* loop counter */
    Int *rp_end ;        /* pointer to the end of the row */
    Int col ;            /* a column index in the row to check */
    Int head_column ;        /* first column in hash bucket or degree list */
    Int first_col ;        /* first column in hash bucket */

    /* Initialization and setup for detecting supercolumns */
    hash = 0;  // Initialize the hash value for columns
    rp = NULL;  // Pointer to a row, initially set to NULL
    c = 0;  // Initialize a column index
    super_c = 0;  // Initialize the index of the column to absorb into
    cp1 = NULL;  // Pointer for the first column super_c
    cp2 = NULL;  // Pointer for the column c
    length = 0;  // Initialize the length of column super_c
    prev_c = 0;  // Initialize the index of the column preceding c
    i = 0;  // Loop counter initialization
    rp_end = NULL;  // Pointer to the end of the row, initially set to NULL
    col = 0;  // Initialize a column index in the row to check
    head_column = 0;  // Initialize the index of the first column in hash bucket or degree list
    first_col = 0;  // Initialize the index of the first column in hash bucket
}
    /* === Consider each column in the row ================================== */

    rp = &A [row_start] ;
    rp_end = rp + row_length ;
    while (rp < rp_end)
    {
        col = *rp++ ;
        if (COL_IS_DEAD (col))
        {
            continue ;
        }

        /* get hash number for this column */
        hash = Col [col].shared3.hash ;
        ASSERT (hash <= n_col) ;

        /* === Get the first column in this hash bucket ===================== */

        head_column = head [hash] ;
        if (head_column > EMPTY)
        {
            /* Retrieve the first column index from the hash bucket */
            first_col = Col [head_column].shared3.headhash ;
        }
        else
        {
            /* No columns in the bucket; use negative of head_column as indicator */
            first_col = - (head_column + 2) ;
        }

        /* === Consider each column in the hash bucket ====================== */

        for (super_c = first_col ; super_c != EMPTY ;
            super_c = Col [super_c].shared4.hash_next)
        {
            ASSERT (COL_IS_ALIVE (super_c)) ;
            ASSERT (Col [super_c].shared3.hash == hash) ;
            length = Col [super_c].length ;

            /* prev_c is the column preceding column c in the hash bucket */
            prev_c = super_c ;

            /* === Compare super_c with all columns after it ================ */

            for (c = Col [super_c].shared4.hash_next ;
                c != EMPTY ; c = Col [c].shared4.hash_next)
            {
                ASSERT (c != super_c) ;
                ASSERT (COL_IS_ALIVE (c)) ;
                ASSERT (Col [c].shared3.hash == hash) ;

                /* not identical if lengths or scores are different */
                if (Col [c].length != length ||
                    Col [c].shared2.score != Col [super_c].shared2.score)
                {
                    prev_c = c ;
                    continue ;
                }

                /* compare the two columns */
                cp1 = &A [Col [super_c].start] ;
                cp2 = &A [Col [c].start] ;

                for (i = 0 ; i < length ; i++)
                {
                    /* the columns are "clean" (no dead rows) */
                    ASSERT (ROW_IS_ALIVE (*cp1))  ;
                    ASSERT (ROW_IS_ALIVE (*cp2))  ;
                    /* row indices will same order for both supercols, */
                    /* no gather scatter nessasary */
                    if (*cp1++ != *cp2++)
                    {
                        break ;
                    }
                }

                /* the two columns are different if the for-loop "broke" */
                if (i != length)
                {
                    prev_c = c ;
                    continue ;
                }

                /* === Got it!  two columns are identical =================== */

                ASSERT (Col [c].shared2.score == Col [super_c].shared2.score) ;

                /* Accumulate thickness and set parent-child relationship */
                Col [super_c].shared1.thickness += Col [c].shared1.thickness ;
                Col [c].shared1.parent = super_c ;
                /* Mark column c as non-principal */
                KILL_NON_PRINCIPAL_COL (c) ;
                /* Order c later in order_children() */
                Col [c].shared2.order = EMPTY ;
                /* Remove c from hash bucket by updating previous column's next pointer */
                Col [prev_c].shared4.hash_next = Col [c].shared4.hash_next ;
            }
        }

        /* === Empty this hash bucket ======================================= */

        if (head_column > EMPTY)
        {
            // Empty the hash bucket by setting head to EMPTY
            head [hash] = EMPTY ;
        }
    }
    {
        # 如果对应的度列表 "hash" 不为空
        Col[head_column].shared3.headhash = EMPTY;
    }
    else
    {
        # 如果对应的度列表 "hash" 是空的
        head[hash] = EMPTY;
    }
    }
/* ========================================================================== */
/* === garbage_collection =================================================== */
/* ========================================================================== */

/*
    Defragments and compacts columns and rows in the workspace A.  Used when
    all available memory has been used while performing row merging.  Returns
    the index of the first free position in A, after garbage collection.  The
    time taken by this routine is linear is the size of the array A, which is
    itself linear in the number of nonzeros in the input matrix.
    Not user-callable.
*/

PRIVATE Int garbage_collection  /* returns the new value of pfree */
(
    /* === Parameters ======================================================= */

    Int n_row,            /* number of rows */
    Int n_col,            /* number of columns */
    Colamd_Row Row [],        /* row info */
    Colamd_Col Col [],        /* column info */
    Int A [],            /* A [0 ... Alen-1] holds the matrix */
    Int *pfree            /* &A [0] ... pfree is in use */
)
{
    /* === Local variables ================================================== */

    Int *psrc ;            /* source pointer */
    Int *pdest ;        /* destination pointer */
    Int j ;            /* counter */
    Int r ;            /* a row index */
    Int c ;            /* a column index */
    Int length ;        /* length of a row or column */

#ifndef NDEBUG
    Int debug_rows ;
    DEBUG2 (("Defrag..\n")) ;
    for (psrc = &A[0] ; psrc < pfree ; psrc++) ASSERT (*psrc >= 0) ;
    debug_rows = 0 ;
#endif /* NDEBUG */

    /* === Defragment the columns =========================================== */

    pdest = &A[0] ;
    for (c = 0 ; c < n_col ; c++)
    {
        if (COL_IS_ALIVE (c))
        {
            psrc = &A [Col [c].start] ;

            /* move and compact the column */
            ASSERT (pdest <= psrc) ;
            Col [c].start = (Int) (pdest - &A [0]) ;
            length = Col [c].length ;
            for (j = 0 ; j < length ; j++)
            {
                r = *psrc++ ;
                if (ROW_IS_ALIVE (r))
                {
                    *pdest++ = r ;
                }
            }
            Col [c].length = (Int) (pdest - &A [Col [c].start]) ;
        }
    }

    /* === Prepare to defragment the rows =================================== */

    for (r = 0 ; r < n_row ; r++)
    {
        if (ROW_IS_DEAD (r) || (Row [r].length == 0))
        {
            /* This row is already dead, or is of zero length.  Cannot compact
             * a row of zero length, so kill it.  NOTE: in the current version,
             * there are no zero-length live rows.  Kill the row (for the first
             * time, or again) just to be safe. */
            KILL_ROW (r) ;
        }
        else
        {
            /* Row is alive and non-empty, no need to kill it */
        }
    }

    /* Return the new value of pfree after garbage collection */
    return (Int) (pdest - &A[0]) ;
}
    {
        /* 将行[r]中第一个列的索引保存到Row[r].shared2.first_column中 */
        psrc = &A[Row[r].start] ;
        // 将psrc指向行[r]在数组A中的起始位置

        Row[r].shared2.first_column = *psrc ;
        // 将psrc指向的值赋给Row[r].shared2.first_column，即第一个列的索引值

        ASSERT (ROW_IS_ALIVE (r)) ;
        // 确保行[r]是有效的（即活跃状态）

        /* 使用行的补码标记行的开始 */
        *psrc = ONES_COMPLEMENT (r) ;
        // 将psrc指向的值设为行[r]的补码
#ifndef NDEBUG
        debug_rows++ ;
#endif /* NDEBUG */
    }
    }

    /* === Defragment the rows ============================================== */

    psrc = pdest ;
    while (psrc < pfree)
    {
    /* find a negative number ... the start of a row */
    if (*psrc++ < 0)
    {
        psrc-- ;
        /* get the row index */
        r = ONES_COMPLEMENT (*psrc) ;
        ASSERT (r >= 0 && r < n_row) ;
        /* restore first column index */
        *psrc = Row [r].shared2.first_column ;
        ASSERT (ROW_IS_ALIVE (r)) ;
        ASSERT (Row [r].length > 0) ;
        /* move and compact the row */
        ASSERT (pdest <= psrc) ;
        Row [r].start = (Int) (pdest - &A [0]) ;
        length = Row [r].length ;
        for (j = 0 ; j < length ; j++)
        {
        c = *psrc++ ;
        if (COL_IS_ALIVE (c))
        {
            *pdest++ = c ;
        }
        }
        Row [r].length = (Int) (pdest - &A [Row [r].start]) ;
        ASSERT (Row [r].length > 0) ;
#ifndef NDEBUG
        debug_rows-- ;
#endif /* NDEBUG */
    }
    }
    /* ensure we found all the rows */
    ASSERT (debug_rows == 0) ;

    /* === Return the new value of pfree ==================================== */

    return ((Int) (pdest - &A [0])) ;
}


注释：


#ifndef NDEBUG
        debug_rows++ ;  // 如果处于调试模式，则增加调试行数计数器
#endif /* NDEBUG */
    }
    }

    /* === Defragment the rows ============================================== */

    psrc = pdest ;  // 将目标指针设置为源指针的当前位置
    while (psrc < pfree)  // 当源指针小于空闲指针时执行循环
    {
    /* find a negative number ... the start of a row */
    if (*psrc++ < 0)  // 如果当前源指针指向的值小于零
    {
        psrc-- ;  // 回退源指针一个位置
        /* get the row index */
        r = ONES_COMPLEMENT (*psrc) ;  // 获取行索引
        ASSERT (r >= 0 && r < n_row) ;  // 断言行索引在有效范围内
        /* restore first column index */
        *psrc = Row [r].shared2.first_column ;  // 恢复第一列索引
        ASSERT (ROW_IS_ALIVE (r)) ;  // 断言行是否存活
        ASSERT (Row [r].length > 0) ;  // 断言行长度大于零
        /* move and compact the row */
        ASSERT (pdest <= psrc) ;  // 断言目标指针小于等于源指针
        Row [r].start = (Int) (pdest - &A [0]) ;  // 设置行起始位置
        length = Row [r].length ;  // 获取行长度
        for (j = 0 ; j < length ; j++)  // 遍历行中的列
        {
        c = *psrc++ ;  // 获取列值
        if (COL_IS_ALIVE (c))  // 如果列是存活的
        {
            *pdest++ = c ;  // 将列值写入目标位置
        }
        }
        Row [r].length = (Int) (pdest - &A [Row [r].start]) ;  // 更新行长度
        ASSERT (Row [r].length > 0) ;  // 断言行长度大于零
#ifndef NDEBUG
        debug_rows-- ;  // 如果处于调试模式，则减少调试行数计数器
#endif /* NDEBUG */
    }
    }
    /* ensure we found all the rows */
    ASSERT (debug_rows == 0) ;  // 断言调试行数计数器为零，确保所有行都被处理完毕

    /* === Return the new value of pfree ==================================== */

    return ((Int) (pdest - &A [0])) ;  // 返回更新后的空闲指针位置
}
    i3 = stats [COLAMD_INFO3] ;
    # 从 stats 数组中获取 COLAMD_INFO3 对应的值，赋给变量 i3

    if (stats [COLAMD_STATUS] >= 0)
    {
        # 如果 stats 数组中 COLAMD_STATUS 对应的值大于等于 0，则执行以下代码块
        SUITESPARSE_PRINTF ("OK.  ") ;
        # 打印 "OK.  "
    }
    else
    {
        # 如果 stats 数组中 COLAMD_STATUS 对应的值小于 0，则执行以下代码块
        SUITESPARSE_PRINTF ("ERROR.  ") ;
        # 打印 "ERROR.  "
    }

    switch (stats [COLAMD_STATUS])
    {
        # 根据 stats 数组中 COLAMD_STATUS 的值进行不同的处理

    case COLAMD_OK_BUT_JUMBLED:
        # 如果 COLAMD_STATUS 的值为 COLAMD_OK_BUT_JUMBLED，则执行以下代码块

            SUITESPARSE_PRINTF(
                    "Matrix has unsorted or duplicate row indices.\n") ;
            # 打印 "Matrix has unsorted or duplicate row indices."

            SUITESPARSE_PRINTF(
                    "%s: number of duplicate or out-of-order row indices: %d\n",
                    method, (int) i3) ;
            # 使用 method 和 i3 的值格式化打印信息

            SUITESPARSE_PRINTF(
                    "%s: last seen duplicate or out-of-order row index:   %d\n",
                    method, (int) INDEX (i2)) ;
            # 使用 method 和 INDEX(i2) 的值格式化打印信息

            SUITESPARSE_PRINTF(
                    "%s: last seen in column:                             %d",
                    method, (int) INDEX (i1)) ;
            # 使用 method 和 INDEX(i1) 的值格式化打印信息

        /* no break - fall through to next case instead */
        # 没有 break，继续执行下一个 case 的代码

    case COLAMD_OK:
        # 如果 COLAMD_STATUS 的值为 COLAMD_OK，则执行以下代码块

            SUITESPARSE_PRINTF("\n") ;
            # 打印一个换行符

            SUITESPARSE_PRINTF(
                    "%s: number of dense or empty rows ignored:           %d\n",
                    method, (int) stats [COLAMD_DENSE_ROW]) ;
            # 使用 method 和 stats[COLAMD_DENSE_ROW] 的值格式化打印信息

            SUITESPARSE_PRINTF(
                    "%s: number of dense or empty columns ignored:        %d\n",
                    method, (int) stats [COLAMD_DENSE_COL]) ;
            # 使用 method 和 stats[COLAMD_DENSE_COL] 的值格式化打印信息

            SUITESPARSE_PRINTF(
                    "%s: number of garbage collections performed:         %d\n",
                    method, (int) stats [COLAMD_DEFRAG_COUNT]) ;
            # 使用 method 和 stats[COLAMD_DEFRAG_COUNT] 的值格式化打印信息

        break ;
        # 跳出 switch 语句块

    case COLAMD_ERROR_A_not_present:
        # 如果 COLAMD_STATUS 的值为 COLAMD_ERROR_A_not_present，则执行以下代码块

        SUITESPARSE_PRINTF(
                    "Array A (row indices of matrix) not present.\n") ;
        # 打印 "Array A (row indices of matrix) not present."

        break ;
        # 跳出 switch 语句块

    case COLAMD_ERROR_p_not_present:
        # 如果 COLAMD_STATUS 的值为 COLAMD_ERROR_p_not_present，则执行以下代码块

            SUITESPARSE_PRINTF(
                    "Array p (column pointers for matrix) not present.\n") ;
            # 打印 "Array p (column pointers for matrix) not present."

        break ;
        # 跳出 switch 语句块

    case COLAMD_ERROR_nrow_negative:
        # 如果 COLAMD_STATUS 的值为 COLAMD_ERROR_nrow_negative，则执行以下代码块

            SUITESPARSE_PRINTF("Invalid number of rows (%d).\n", (int) i1) ;
            # 使用 i1 的值格式化打印信息

        break ;
        # 跳出 switch 语句块

    case COLAMD_ERROR_ncol_negative:
        # 如果 COLAMD_STATUS 的值为 COLAMD_ERROR_ncol_negative，则执行以下代码块

            SUITESPARSE_PRINTF("Invalid number of columns (%d).\n", (int) i1) ;
            # 使用 i1 的值格式化打印信息

        break ;
        # 跳出 switch 语句块

    case COLAMD_ERROR_nnz_negative:
        # 如果 COLAMD_STATUS 的值为 COLAMD_ERROR_nnz_negative，则执行以下代码块

            SUITESPARSE_PRINTF(
                   "Invalid number of nonzero entries (%d).\n", (int) i1) ;
            # 使用 i1 的值格式化打印信息

        break ;
        # 跳出 switch 语句块

    case COLAMD_ERROR_p0_nonzero:
        # 如果 COLAMD_STATUS 的值为 COLAMD_ERROR_p0_nonzero，则执行以下代码块

            SUITESPARSE_PRINTF(
                   "Invalid column pointer, p [0] = %d, must be zero.\n", (int)i1);
            # 使用 i1 的值格式化打印信息

        break ;
        # 跳出 switch 语句块

    case COLAMD_ERROR_A_too_small:
        # 如果 COLAMD_STATUS 的值为 COLAMD_ERROR_A_too_small，则执行以下代码块

            SUITESPARSE_PRINTF("Array A too small.\n") ;
            # 打印 "Array A too small."

            SUITESPARSE_PRINTF(
                    "        Need Alen >= %d, but given only Alen = %d.\n",
                    (int) i1, (int) i2) ;
            # 使用 i1 和 i2 的值格式化打印信息

        break ;
        # 跳出 switch 语句块

    case COLAMD_ERROR_col_length_negative:
        # 如果 COLAMD_STATUS 的值为 COLAMD_ERROR_col_length_negative，则执行以下代码块

            SUITESPARSE_PRINTF
            ("Column %d has a negative number of nonzero entries (%d).\n",
         (int) INDEX (i1), (int) i2) ;
            # 使用 INDEX(i1) 和 i2 的值格式化打印信息

        break ;
        # 跳出 switch 语句块

    default:
        # 如果 COLAMD_STATUS 的值与以上所有 case 都不匹配，则执行以下代码块

        break ;
        # 跳出 switch 语句块
    # 当 COLAMD 库报告行索引超出范围错误时执行以下操作
    case COLAMD_ERROR_row_index_out_of_bounds:
        # 使用 SUITESPARSE_PRINTF 输出行索引超出范围的错误信息，包括具体的行索引值和列索引范围
        SUITESPARSE_PRINTF
        ("Row index (row %d) out of bounds (%d to %d) in column %d.\n",
         (int) INDEX (i2), (int) INDEX (0), (int) INDEX (i3-1), (int) INDEX (i1)) ;
        break ;

    # 当 COLAMD 库报告内存不足错误时执行以下操作
    case COLAMD_ERROR_out_of_memory:
        # 使用 SUITESPARSE_PRINTF 输出内存不足的错误信息
        SUITESPARSE_PRINTF("Out of memory.\n") ;
        break ;

    /* v2.4: 内部错误情况已删除 */
/* ========================================================================== */
/* === colamd debugging routines ============================================ */
/* ========================================================================== */

/* When debugging is disabled, the remainder of this file is ignored. */

#ifndef NDEBUG

/* ========================================================================== */
/* === debug_structures ===================================================== */
/* ========================================================================== */

/*
    At this point, all empty rows and columns are dead.  All live columns
    are "clean" (containing no dead rows) and simplicial (no supercolumns
    yet).  Rows may contain dead columns, but all live rows contain at
    least one live column.
*/

// 调试函数：检查数据结构的一致性和正确性
PRIVATE void debug_structures
(
    /* === Parameters ======================================================= */
    
    Int n_row,          // 行数
    Int n_col,          // 列数
    Colamd_Row Row [],  // 行数据结构数组
    Colamd_Col Col [],  // 列数据结构数组
    Int A [],           // 稀疏矩阵的列压缩格式数据
    Int n_col2          // 第二阶段处理的列数限制
)
{
    /* === Local variables ================================================== */

    Int i ;             // 通用计数器
    Int c ;             // 列索引
    Int *cp ;           // 指向列压缩格式中当前列开始位置的指针
    Int *cp_end ;       // 指向列压缩格式中当前列结束位置的指针
    Int len ;           // 列长度
    Int score ;         // 列得分
    Int r ;             // 行索引
    Int *rp ;           // 指向行压缩格式中当前行开始位置的指针
    Int *rp_end ;       // 指向行压缩格式中当前行结束位置的指针
    Int deg ;           // 行度数

    /* === Check A, Row, and Col ============================================ */

    // 检查所有活跃列的数据结构
    for (c = 0 ; c < n_col ; c++)
    {
        if (COL_IS_ALIVE (c))  // 如果列 c 是活跃的
        {
            len = Col [c].length ;                  // 获取列 c 的长度
            score = Col [c].shared2.score ;         // 获取列 c 的得分
            DEBUG4 (("initial live col %5d %5d %5d\n", c, len, score)) ;  // 调试输出列 c 的初始信息
            ASSERT (len > 0) ;                      // 确保列长度大于 0
            ASSERT (score >= 0) ;                   // 确保列得分非负
            ASSERT (Col [c].shared1.thickness == 1) ;// 确保列的厚度为 1
            cp = &A [Col [c].start] ;               // 指向列 c 在 A 中的起始位置
            cp_end = cp + len ;                     // 指向列 c 在 A 中的结束位置
            while (cp < cp_end)
            {
                r = *cp++ ;                         // 获取 A 中的行索引 r
                ASSERT (ROW_IS_ALIVE (r)) ;         // 确保行 r 是活跃的
            }
        }
        else  // 如果列 c 是非活跃的
        {
            i = Col [c].shared2.order ;            // 获取列 c 的排序顺序
            ASSERT (i >= n_col2 && i < n_col) ;    // 确保排序顺序在指定范围内
        }
    }

    // 检查所有活跃行的数据结构
    for (r = 0 ; r < n_row ; r++)
    {
        if (ROW_IS_ALIVE (r))  // 如果行 r 是活跃的
        {
            i = 0 ;
            len = Row [r].length ;                  // 获取行 r 的长度
            deg = Row [r].shared1.degree ;          // 获取行 r 的度数
            ASSERT (len > 0) ;                      // 确保行长度大于 0
            ASSERT (deg > 0) ;                      // 确保度数大于 0
            rp = &A [Row [r].start] ;               // 指向行 r 在 A 中的起始位置
            rp_end = rp + len ;                     // 指向行 r 在 A 中的结束位置
            while (rp < rp_end)
            {
                c = *rp++ ;                         // 获取 A 中的列索引 c
                if (COL_IS_ALIVE (c))               // 如果列 c 是活跃的
                {
                    i++ ;                           // 计数符合条件的列数
                }
            }
            ASSERT (i > 0) ;                         // 确保至少存在一个活跃列
        }
    }
}


/* ========================================================================== */
/* === debug_deg_lists ====================================================== */
/* ========================================================================== */

/*
    Prints the contents of the degree lists.  Counts the number of columns
    in the degree list and compares it to the total it should have.  Also
    checks the row degrees.
*/

// 调试函数：打印度数列表的内容，并检查列和行的度数
PRIVATE void debug_deg_lists
(
    /* === Parameters ======================================================= */

    Int n_row,  // 行数
    Int n_col,  // 列数
    # 声明一个 Colamd_Row 数组，表示 COLAMD 算法中的行
    Colamd_Row Row [],
    # 声明一个 Colamd_Col 数组，表示 COLAMD 算法中的列
    Colamd_Col Col [],
    # 声明一个 Int 数组，表示 COLAMD 算法中的头指针
    Int head [],
    # 声明一个 Int 变量，表示 COLAMD 算法中的最小分数
    Int min_score,
    # 声明一个 Int 变量，表示 COLAMD 算法中的应该值
    Int should,
    # 声明一个 Int 变量，表示 COLAMD 算法中的最大度
    Int max_deg
/* === Local variables ================================================== */

Int deg ;   // 声明整型变量 deg，用于迭代列的度数
Int col ;   // 声明整型变量 col，表示列索引
Int have ;  // 声明整型变量 have，用于计算列的共享厚度
Int row ;   // 声明整型变量 row，表示行索引

/* === Check the degree lists =========================================== */

if (n_col > 10000 && colamd_debug <= 0)
{
return ;  // 如果列数超过10000且不处于调试模式，则退出函数
}
have = 0 ;  // 初始化 have 变量为 0
DEBUG4 (("Degree lists: %d\n", min_score)) ;  // 输出调试信息，显示度数列表中的最小分数
for (deg = 0 ; deg <= n_col ; deg++)
{
col = head [deg] ;  // 获取度数为 deg 的第一个列索引
if (col == EMPTY)
{
continue ;  // 如果列索引为 EMPTY，则跳过当前迭代
}
DEBUG4 (("%d:", deg)) ;  // 输出调试信息，显示当前度数 deg
while (col != EMPTY)
{
DEBUG4 ((" %d", col)) ;  // 输出调试信息，显示当前列索引 col
have += Col [col].shared1.thickness ;  // 累加当前列的共享厚度到 have 变量
ASSERT (COL_IS_ALIVE (col)) ;  // 断言当前列索引 col 是活跃的
col = Col [col].shared4.degree_next ;  // 获取下一个相同度数的列索引
}
DEBUG4 (("\n")) ;  // 输出换行符，表示当前度数的列索引输出完毕
}
DEBUG4 (("should %d have %d\n", should, have)) ;  // 输出调试信息，显示预期共享厚度和实际共享厚度
ASSERT (should == have) ;  // 断言预期共享厚度应等于实际共享厚度

/* === Check the row degrees ============================================ */

if (n_row > 10000 && colamd_debug <= 0)
{
return ;  // 如果行数超过10000且不处于调试模式，则退出函数
}
for (row = 0 ; row < n_row ; row++)
{
if (ROW_IS_ALIVE (row))
{
ASSERT (Row [row].shared1.degree <= max_deg) ;  // 断言当前行的度数不超过最大度数
}
}
}



/* ========================================================================== */
/* === debug_mark =========================================================== */
/* ========================================================================== */

/*
Ensures that the tag_mark is less that the maximum and also ensures that
each entry in the mark array is less than the tag mark.
*/

PRIVATE void debug_mark
(
/* === Parameters ======================================================= */

Int n_row,
Colamd_Row Row [],
Int tag_mark,
Int max_mark
)
{
/* === Local variables ================================================== */

Int r ;  // 声明整型变量 r，用于迭代行索引

/* === Check the Row marks ============================================== */

ASSERT (tag_mark > 0 && tag_mark <= max_mark) ;  // 断言标记标记小于等于最大标记值
if (n_row > 10000 && colamd_debug <= 0)
{
return ;  // 如果行数超过10000且不处于调试模式，则退出函数
}
for (r = 0 ; r < n_row ; r++)
{
ASSERT (Row [r].shared2.mark < tag_mark) ;  // 断言当前行的标记值小于 tag_mark
}
}



/* ========================================================================== */
/* === debug_matrix ========================================================= */
/* ========================================================================== */

/*
Prints out the contents of the columns and the rows.
*/

PRIVATE void debug_matrix
(
/* === Parameters ======================================================= */

Int n_row,
Int n_col,
Colamd_Row Row [],
Colamd_Col Col [],
Int A []
)
{
/* === Local variables ================================================== */

Int r ;  // 声明整型变量 r，用于迭代行索引
Int c ;  // 声明整型变量 c，用于迭代列索引
Int *rp ;  // 声明指向整型的指针 rp，用于访问行指针数组
Int *rp_end ;  // 声明指向整型的指针 rp_end，表示行指针数组的末尾
Int *cp ;  // 声明指向整型的指针 cp，用于访问列指针数组
Int *cp_end ;  // 声明指向整型的指针 cp_end，表示列指针数组的末尾

/* === Dump the rows and columns of the matrix ========================== */

if (colamd_debug < 3)
{
return ;  // 如果调试级别小于3，则退出函数
}
DEBUG3 (("DUMP MATRIX:\n")) ;  // 输出调试信息，显示“DUMP MATRIX:”
for (r = 0 ; r < n_row ; r++)
{
    # 输出调试信息，显示行号 r 是否活跃，以及 ROW_IS_ALIVE 宏的结果
    DEBUG3 (("Row %d alive? %d\n", r, ROW_IS_ALIVE (r))) ;
    # 如果行号 r 已经死亡，跳过当前循环，继续下一次迭代
    if (ROW_IS_DEAD (r))
    {
        continue ;
    }
    # 输出调试信息，显示行号 r 的起始位置、长度和共享度
    DEBUG3 (("start %d length %d degree %d\n",
        Row [r].start, Row [r].length, Row [r].shared1.degree)) ;
    # rp 指向矩阵 A 中行号 r 的起始位置
    rp = &A [Row [r].start] ;
    # rp_end 指向矩阵 A 中行号 r 的结束位置
    rp_end = rp + Row [r].length ;
    # 遍历行号 r 对应的列索引，从 rp 到 rp_end
    while (rp < rp_end)
    {
        # c 为当前列索引，从 rp 指向的位置获取
        c = *rp++ ;
        # 输出调试信息，显示列号 c 是否活跃，以及 COL_IS_ALIVE 宏的结果
        DEBUG4 (("    %d col %d\n", COL_IS_ALIVE (c), c)) ;
    }
    }

    # 遍历所有列 c
    for (c = 0 ; c < n_col ; c++)
    {
    # 输出调试信息，显示列号 c 是否活跃，以及 COL_IS_ALIVE 宏的结果
    DEBUG3 (("Col %d alive? %d\n", c, COL_IS_ALIVE (c))) ;
    # 如果列号 c 已经死亡，跳过当前循环，继续下一次迭代
    if (COL_IS_DEAD (c))
    {
        continue ;
    }
    # 输出调试信息，显示列号 c 的起始位置、长度以及两个共享变量的值
    DEBUG3 (("start %d length %d shared1 %d shared2 %d\n",
        Col [c].start, Col [c].length,
        Col [c].shared1.thickness, Col [c].shared2.score)) ;
    # cp 指向矩阵 A 中列号 c 的起始位置
    cp = &A [Col [c].start] ;
    # cp_end 指向矩阵 A 中列号 c 的结束位置
    cp_end = cp + Col [c].length ;
    # 遍历列号 c 对应的行索引，从 cp 到 cp_end
    while (cp < cp_end)
    {
        # r 为当前行索引，从 cp 指向的位置获取
        r = *cp++ ;
        # 输出调试信息，显示行号 r 是否活跃，以及 ROW_IS_ALIVE 宏的结果
        DEBUG4 (("    %d row %d\n", ROW_IS_ALIVE (r), r)) ;
    }
    }
}

PRIVATE void colamd_get_debug
(
    char *method
)
{
    FILE *f ;
    colamd_debug = 0 ;        /* no debug printing */
    f = fopen ("debug", "r") ;  // 尝试打开名为 "debug" 的文件以供读取
    if (f == (FILE *) NULL)     // 如果文件打开失败
    {
        colamd_debug = 0 ;      // 设置调试标志为 0（不打印调试信息）
    }
    else                        // 如果文件成功打开
    {
        fscanf (f, "%d", &colamd_debug) ;  // 从文件中读取一个整数，存入 colamd_debug 中
        fclose (f) ;            // 关闭文件
    }
    DEBUG0 (("%s: debug version, D = %d (THIS WILL BE SLOW!)\n",
        method, colamd_debug)) ;  // 打印调试信息，包括方法名和调试标志值
}

#endif /* NDEBUG */
```