# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\slu_util.h`

```
/*
 * \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/** @file slu_util.h
 * \brief Utility header file 
 *
 * -- SuperLU routine (version 4.1) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November, 2010
 *
 */

#ifndef __SUPERLU_UTIL /* allow multiple inclusions */
#define __SUPERLU_UTIL

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/*
#ifndef __STDC__
#include <malloc.h>
#endif
*/
#include <assert.h>
#include "superlu_enum_consts.h"


#include "scipy_slu_config.h"

/***********************************************************************
 * Macros
 ***********************************************************************/
/*                                                                                           
 * You can support older version of SuperLU.                                              
 * At compile-time, you can catch the new release as:                                          
 *   #ifdef SUPERLU_MAJOR_VERSION == 5
 *       use the new interface                                                                 
 *   #else                                                                                     
 *       use the old interface                                                                 
 *   #endif                                                                                    
 * Versions 4.x and earlier do not include a #define'd version numbers.                        
 */
#define SUPERLU_MAJOR_VERSION     6
#define SUPERLU_MINOR_VERSION     0
#define SUPERLU_PATCH_VERSION     1


#define FIRSTCOL_OF_SNODE(i)    (xsup[i])
/* No of marker arrays used in the symbolic factorization,
   each of size n */
#define NO_MARKER     3
#define NUM_TEMPV(m,w,t,b)  ( SUPERLU_MAX(m, (t + b)*w) )

#ifndef USER_ABORT
#define USER_ABORT(msg) superlu_abort_and_exit(msg)
#endif

#define ABORT(err_msg) \
 { char msg[256];\
   sprintf(msg,"%s at line %d in file %s\n",err_msg,__LINE__, __FILE__);\
   USER_ABORT(msg); }


#ifndef USER_MALLOC
#if 1
#define USER_MALLOC(size) superlu_malloc(size)
#else
/* The following may check out some uninitialized data */
#define USER_MALLOC(size) memset (superlu_malloc(size), '\x0F', size)
#endif
#endif

#define SUPERLU_MALLOC(size) USER_MALLOC(size)

#ifndef USER_FREE
#define USER_FREE(addr) superlu_free(addr)
#endif

#define SUPERLU_FREE(addr) USER_FREE(addr)

#define CHECK_MALLOC(where) {                 \
    extern int superlu_malloc_total;        \
    printf("%s: malloc_total %d Bytes\n",     \
       where, superlu_malloc_total); \
}

#define SUPERLU_MAX(x, y)     ( (x) > (y) ? (x) : (y) )
/*
 * 定义一个宏函数，返回两个数中的最小值
 */
#define SUPERLU_MIN(x, y)     ( (x) < (y) ? (x) : (y) )

/*********************************************************
 * 用于简化稀疏矩阵条目访问的宏定义。
 *********************************************************/
/*
 * 返回 L 存储中指定列的起始位置
 */
#define L_SUB_START(col)     ( Lstore->rowind_colptr[col] )
/*
 * 返回 L 存储中指定指针位置的值
 */
#define L_SUB(ptr)           ( Lstore->rowind[ptr] )
/*
 * 返回 L 存储中指定列的非零值起始位置
 */
#define L_NZ_START(col)      ( Lstore->nzval_colptr[col] )
/*
 * 返回 L 存储中指定超节点的第一个列号
 */
#define L_FST_SUPC(superno)  ( Lstore->sup_to_col[superno] )
/*
 * 返回 U 存储中指定列的非零值起始位置
 */
#define U_NZ_START(col)      ( Ustore->colptr[col] )
/*
 * 返回 U 存储中指定指针位置的值
 */
#define U_SUB(ptr)           ( Ustore->rowind[ptr] )


/***********************************************************************
 * 常量定义
 ***********************************************************************/
/*
 * 表示空值
 */
#define EMPTY    (-1)
/*
 * FALSE 和 TRUE 常量定义
 */
#define FALSE    0
#define TRUE    1

#if 0 // 这是旧值；新值为 6，在 superlu_enum_consts.h 中定义
#define NO_MEMTYPE  4      /* 0: lusup;
                  1: ucol;
                  2: lsub;
                  3: usub */
#endif

/*
 * 返回给定 n 的 GluIntArray 大小
 */
#define GluIntArray(n)   (5 * (n) + 5)

/* 丢弃规则 */
#define  NODROP            ( 0x0000 )
#define     DROP_BASIC    ( 0x0001 )  /* ILU(tau) */
#define  DROP_PROWS    ( 0x0002 )  /* ILUTP: 保留最大的 p 行 */
#define  DROP_COLUMN    ( 0x0004 )  /* ILUTP: 对于第 j 列，p = gamma * nnz(A(:,j)) */
#define  DROP_AREA     ( 0x0008 )  /* ILUTP: 对于第 j 列，使用 nnz(F(:,1:j)) / nnz(A(:,1:j)) 
                            以限制内存增长 */
#define  DROP_SECONDARY    ( 0x000E )  /* PROWS | COLUMN | AREA */
#define  DROP_DYNAMIC    ( 0x0010 )  /* 自适应 tau */
#define  DROP_INTERP    ( 0x0100 )  /* 使用插值 */


#if 1
#define MILU_ALPHA (1.0e-2) /* 添加到对角线上的 drop_sum 的倍数 */
#else
#define MILU_ALPHA  1.0 /* 添加到对角线上的 drop_sum 的倍数 */
#endif


/***********************************************************************
 * 类型定义
 ***********************************************************************/
/*
 * 定义 flops_t 类型为 float
 */
typedef float    flops_t;
/*
 * 定义 Logical 类型为 unsigned char
 */
typedef unsigned char Logical;

/*
 * 结构体定义，包含各种数值和枚举类型的成员
 */
typedef struct {
    fact_t        Fact;
    yes_no_t      Equil;
    colperm_t     ColPerm;
    trans_t       Trans;
    IterRefine_t  IterRefine;
    double        DiagPivotThresh;
    yes_no_t      SymmetricMode;
    yes_no_t      PivotGrowth;
    yes_no_t      ConditionNumber;
    rowperm_t     RowPerm;
    int       ILU_DropRule;
    double      ILU_DropTol;    /* 丢弃阈值 */
    double      ILU_FillFactor; /* 次要丢弃中的 gamma */
    norm_t      ILU_Norm;       /* 无穷范数、1-范数或 2-范数 */
    double      ILU_FillTol;    /* 零主元扰动的阈值 */
    milu_t      ILU_MILU;
    double      ILU_MILU_Dim;   /* PDE 的维度（如果有） */
    yes_no_t      ParSymbFact;
    yes_no_t      ReplaceTinyPivot; /* 在 SuperLU_DIST 中使用 */
    yes_no_t      SolveInitialized;
    # 声明一个变量 RefineInitialized，类型为 yes_no_t，用于表示某种初始化状态或标志
    yes_no_t      RefineInitialized;

    # 声明一个变量 PrintStat，类型为 yes_no_t，用于表示是否打印统计信息的标志
    yes_no_t      PrintStat;

    # 声明两个整型变量 nnzL 和 nnzU，用于临时存储非零元素数量
    int           nnzL, nnzU;      /* used to store nnzs for now       */

    # 声明一个整型变量 num_lookaheads，表示预测向前查看的层数
    int           num_lookaheads;  /* num of levels in look-ahead      */

    # 声明一个变量 lookahead_etree，类型为 yes_no_t，表示是否使用从串行符号因子化计算得到的 etree
    yes_no_t      lookahead_etree; /* use etree computed from the
                      serial symbolic factorization */

    # 声明一个变量 SymPattern，类型为 yes_no_t，表示是否进行对称因子分解
    yes_no_t      SymPattern;      /* symmetric factorization          */
/*! \brief 结构体定义：superlu_options_t，包含 SuperLU 的选项设置 */
typedef struct {
    int_t Fact;          /* Type of factorization to perform */
    int_t Equil;         /* Specifies whether to equilibrate the matrix */
    int_t ColPerm;       /* Column permutation to use */
    int_t IterRefine;    /* Specifies whether to perform iterative refinement */
    float DiagPivotThresh; /* Threshold for diagonal pivoting */
    int_t SymmetricMode; /* Flag to indicate symmetric matrix structure */
    int_t PrintStat;     /* Specifies whether to print statistical information */
} superlu_options_t;

/*! \brief 结构体定义：ExpHeader，用于动态管理的内存头部信息 */
typedef struct e_node {
    int_t size;    /* 已使用的内存 mem[] 的长度 */
    void *mem;     /* 指向新分配的存储空间的指针 */
} ExpHeader;

/*! \brief 结构体定义：LU_stack_t，用于堆栈操作的结构体 */
typedef struct {
    int_t  size;
    int_t  used;
    int_t  top1;    /* 向上增长，相对于 &array[0] */
    int_t  top2;    /* 向下增长 */
    void *array;
} LU_stack_t;

/*! \brief 结构体定义：SuperLUStat_t，包含 SuperLU 的统计信息 */
typedef struct {
    int     *panel_histo; /* 面板大小分布的直方图 */
    double  *utime;       /* 各阶段的运行时间 */
    flops_t *ops;         /* 各阶段的操作计数 */
    int     TinyPivots;   /* 微小主元的数量 */
    int     RefineSteps;  /* 迭代细化步数 */
    int     expansions;   /* 内存扩展次数 */
} SuperLUStat_t;

/*! \brief 结构体定义：mem_usage_t，用于内存使用统计 */
typedef struct {
    float for_lu;
    float total_needed;
} mem_usage_t;

/*! \brief 结构体定义：GlobalLU_t，包含全局 LU 分解的数据结构 */
typedef struct {
    int     *xsup;    /* 超节点和列映射 */
    int     *supno;   
    int_t   *lsub;    /* 压缩后的 L 的下标 */
    int_t   *xlsub;
    void    *lusup;   /* L 的超节点 */
    int_t   *xlusup;
    void    *ucol;    /* U 的列 */
    int_t   *usub;
    int_t   *xusub;
    int_t   nzlmax;   /* lsub 的当前最大大小 */
    int_t   nzumax;   /* ucol 的当前最大大小 */
    int_t   nzlumax;  /* lusup 的当前最大大小 */
    int     n;        /* 矩阵的列数 */
    LU_space_t MemModel; /* 内存模型，0 - 系统分配；1 - 用户提供 */
    int     num_expansions;
    ExpHeader *expanders; /* 指向 4 种类型内存的指针数组 */
    LU_stack_t stack;     /* 使用用户提供的内存的堆栈 */
} GlobalLU_t;

/***********************************************************************
 * 函数原型声明
 ***********************************************************************/
#ifdef __cplusplus
extern "C" {
#endif

extern int     input_error(char *, int *);

extern void    Destroy_SuperMatrix_Store(SuperMatrix *);
extern void    Destroy_CompCol_Matrix(SuperMatrix *);
extern void    Destroy_CompRow_Matrix(SuperMatrix *);
extern void    Destroy_SuperNode_Matrix(SuperMatrix *);
extern void    Destroy_CompCol_Permuted(SuperMatrix *);
extern void    Destroy_Dense_Matrix(SuperMatrix *);
extern void    get_perm_c(int, SuperMatrix *, int *);
extern void    set_default_options(superlu_options_t *options);
extern void    ilu_set_default_options(superlu_options_t *options);
extern void    sp_preorder (superlu_options_t *, SuperMatrix*, int*, int*,
                SuperMatrix*);
extern void    superlu_abort_and_exit(char*);
extern void    *superlu_malloc (size_t);
extern int     *int32Malloc (int);
extern int     *int32Calloc (int);
extern int_t   *intMalloc (int_t);
extern int_t   *intCalloc (int_t);
extern void    superlu_free (void*);
extern void    SetIWork (int, int, int, int *, int **, int **, int_t **xplore,
                         int **, int **, int_t **xprune, int **);

#ifdef __cplusplus
}
#endif
extern int     sp_coletree (int_t *, int_t *, int_t *, int, int, int *);
// 声明一个函数 sp_coletree，接受一些参数并返回一个整数

extern void    relax_snode (const int, int *, const int, int *, int *);
// 声明一个函数 relax_snode，接受一些参数并无返回值

extern void    heap_relax_snode (const int, int *, const int, int *, int *);
// 声明一个函数 heap_relax_snode，接受一些参数并无返回值

extern int     mark_relax(int, int *, int *, int_t *, int_t *, int_t *, int *);
// 声明一个函数 mark_relax，接受一些参数并返回一个整数

extern void    countnz(const int n, int_t *xprune, int_t *nnzL, int_t *nnzU, GlobalLU_t *);
// 声明一个函数 countnz，接受一些参数并无返回值

extern void    ilu_countnz (const int, int_t *, int_t *, GlobalLU_t *);
// 声明一个函数 ilu_countnz，接受一些参数并无返回值

extern void    fixupL (const int, const int *, GlobalLU_t *);
// 声明一个函数 fixupL，接受一些参数并无返回值

extern void    ilu_relax_snode (const int, int *, const int, int *,
                int *, int *);
// 声明一个函数 ilu_relax_snode，接受一些参数并无返回值，跨多行写法

extern void    ilu_heap_relax_snode (const int, int *, const int, int *,
                     int *, int*);
// 声明一个函数 ilu_heap_relax_snode，接受一些参数并无返回值，跨多行写法

extern void    resetrep_col (const int, const int *, int *);
// 声明一个函数 resetrep_col，接受一些参数并无返回值

extern int     spcoletree (int *, int *, int *, int, int, int *);
// 声明一个函数 spcoletree，接受一些参数并返回一个整数

extern int     *TreePostorder (int, int *);
// 声明一个函数 TreePostorder，接受一些参数并返回一个整数指针

extern double  SuperLU_timer_ (void);
// 声明一个函数 SuperLU_timer_，接受无参数并返回一个双精度浮点数

extern int     sp_ienv (int);
// 声明一个函数 sp_ienv，接受一个整数参数并返回一个整数

extern int     xerbla_ (char *, int *);
// 声明一个函数 xerbla_，接受一个字符指针和一个整数指针参数并返回一个整数

extern void    ifill (int *, int, int);
// 声明一个函数 ifill，接受一些参数并无返回值

extern void    snode_profile (int, int *);
// 声明一个函数 snode_profile，接受一些参数并无返回值

extern void    super_stats (int, int *);
// 声明一个函数 super_stats，接受一些参数并无返回值

extern void    check_repfnz(int, int, int, int *);
// 声明一个函数 check_repfnz，接受一些参数并无返回值

extern void    PrintSumm (char *, int, int, int);
// 声明一个函数 PrintSumm，接受一些参数并无返回值

extern void    StatInit(SuperLUStat_t *);
// 声明一个函数 StatInit，接受一个 SuperLUStat_t 结构体指针参数并无返回值

extern void    StatPrint (SuperLUStat_t *);
// 声明一个函数 StatPrint，接受一个 SuperLUStat_t 结构体指针参数并无返回值

extern void    StatFree(SuperLUStat_t *);
// 声明一个函数 StatFree，接受一个 SuperLUStat_t 结构体指针参数并无返回值

extern void    print_panel_seg(int, int, int, int, int *, int *);
// 声明一个函数 print_panel_seg，接受一些参数并无返回值

extern int     print_int_vec(char *,int, int *);
// 声明一个函数 print_int_vec，接受一些参数并返回一个整数

extern int     slu_PrintInt10(char *, int, int *);
// 声明一个函数 slu_PrintInt10，接受一些参数并返回一个整数

extern int     check_perm(char *what, int n, int *perm);
// 声明一个函数 check_perm，接受一些参数并返回一个整数

#ifdef __cplusplus
  }
#endif

#endif /* __SUPERLU_UTIL */
```