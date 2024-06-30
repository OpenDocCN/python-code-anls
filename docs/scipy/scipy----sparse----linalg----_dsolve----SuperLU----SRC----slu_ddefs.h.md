# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\slu_ddefs.h`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file slu_ddefs.h
 * \brief Header file for real operations
 * 
 * <pre> 
 * -- SuperLU routine (version 4.1) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November, 2010
 * 
 * Global data structures used in LU factorization -
 * 
 *   nsuper: #supernodes = nsuper + 1, numbered [0, nsuper].
 *   (xsup,supno): supno[i] is the supernode no to which i belongs;
 *    xsup(s) points to the beginning of the s-th supernode.
 *    e.g.   supno 0 1 2 2 3 3 3 4 4 4 4 4   (n=12)
 *            xsup 0 1 2 4 7 12
 *    Note: dfs will be performed on supernode rep. relative to the new 
 *          row pivoting ordering
 *
 *   (xlsub,lsub): lsub[*] contains the compressed subscript of
 *    rectangular supernodes; xlsub[j] points to the starting
 *    location of the j-th column in lsub[*]. Note that xlsub 
 *    is indexed by column.
 *    Storage: original row subscripts
 *
 *      During the course of sparse LU factorization, we also use
 *    (xlsub,lsub) for the purpose of symmetric pruning. For each
 *    supernode {s,s+1,...,t=s+r} with first column s and last
 *    column t, the subscript set
 *        lsub[j], j=xlsub[s], .., xlsub[s+1]-1
 *    is the structure of column s (i.e. structure of this supernode).
 *    It is used for the storage of numerical values.
 *    Furthermore,
 *        lsub[j], j=xlsub[t], .., xlsub[t+1]-1
 *    is the structure of the last column t of this supernode.
 *    It is for the purpose of symmetric pruning. Therefore, the
 *    structural subscripts can be rearranged without making physical
 *    interchanges among the numerical values.
 *
 *    However, if the supernode has only one column, then we
 *    only keep one set of subscripts. For any subscript interchange
 *    performed, similar interchange must be done on the numerical
 *    values.
 *
 *    The last column structures (for pruning) will be removed
 *    after the numercial LU factorization phase.
 *
 *   (xlusup,lusup): lusup[*] contains the numerical values of the
 *    rectangular supernodes; xlusup[j] points to the starting
 *    location of the j-th column in storage vector lusup[*]
 *    Note: xlusup is indexed by column.
 *    Each rectangular supernode is stored by column-major
 *    scheme, consistent with Fortran 2-dim array storage.
 *
 *   (xusub,ucol,usub): ucol[*] stores the numerical values of
 *    U-columns outside the rectangular supernodes. The row
 *    subscript of nonzero ucol[k] is stored in usub[k].
 *    xusub[i] points to the starting location of column i in ucol.
 *    Storage: new row subscripts; that is subscripts of PA.
 * </pre>
 */
#ifndef __SUPERLU_dSP_DEFS /* allow multiple inclusions */
#define __SUPERLU_dSP_DEFS

/*
 * File name:        dsp_defs.h
 * Purpose:          Sparse matrix types and function prototypes
 * History:
 */

#ifdef _CRAY
#include <fortran.h>
#endif

#include <math.h>          /* 数学函数库 */
#include <limits.h>        /* 整数类型的极限值 */
#include <stdio.h>         /* 标准输入输出 */
#include <stdlib.h>        /* 常用函数及内存分配 */
#include <stdint.h>        /* 标准整数类型 */
#include <string.h>        /* 字符串处理函数 */
#include "slu_Cnames.h"    /* 定义了一些符号常量 */
#include "superlu_config.h"/* 超级LU的配置文件 */
#include "supermatrix.h"   /* 超级矩阵的定义 */
#include "slu_util.h"      /* 超级LU的实用函数 */


/* -------- Prototypes -------- */

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Driver routines */
extern void
dgssv(superlu_options_t *, SuperMatrix *, int *, int *, SuperMatrix *,
      SuperMatrix *, SuperMatrix *, SuperLUStat_t *, int_t *info);
extern void
dgssvx(superlu_options_t *, SuperMatrix *, int *, int *, int *,
       char *, double *, double *, SuperMatrix *, SuperMatrix *,
       void *, int_t lwork, SuperMatrix *, SuperMatrix *,
       double *, double *, double *, double *,
       GlobalLU_t *, mem_usage_t *, SuperLUStat_t *, int_t *info);
    /* ILU */
extern void
dgsisv(superlu_options_t *, SuperMatrix *, int *, int *, SuperMatrix *,
      SuperMatrix *, SuperMatrix *, SuperLUStat_t *, int *);
extern void
dgsisx(superlu_options_t *options, SuperMatrix *A, int *perm_c, int *perm_r,
       int *etree, char *equed, double *R, double *C,
       SuperMatrix *L, SuperMatrix *U, void *work, int_t lwork,
       SuperMatrix *B, SuperMatrix *X, double *recip_pivot_growth, double *rcond,
       GlobalLU_t *Glu, mem_usage_t *mem_usage, SuperLUStat_t *stat, int_t *info);


/*! \brief Supernodal LU factor related */
extern void
dCreate_CompCol_Matrix(SuperMatrix *, int, int, int_t, double *,
               int_t *, int_t *, Stype_t, Dtype_t, Mtype_t);
extern void
dCreate_CompRow_Matrix(SuperMatrix *, int, int, int_t, double *,
               int_t *, int_t *, Stype_t, Dtype_t, Mtype_t);
extern void dCompRow_to_CompCol(int, int, int_t, double*, int_t*, int_t*,
                           double **, int_t **, int_t **);
extern void
dCopy_CompCol_Matrix(SuperMatrix *, SuperMatrix *);
extern void
dCreate_Dense_Matrix(SuperMatrix *, int, int, double *, int,
             Stype_t, Dtype_t, Mtype_t);
extern void
dCreate_SuperNode_Matrix(SuperMatrix *, int, int, int_t, double *, 
                 int_t *, int_t *, int_t *, int *, int *,
             Stype_t, Dtype_t, Mtype_t);
extern void
dCopy_Dense_Matrix(int, int, double *, int, double *, int);

extern void    dallocateA (int, int_t, double **, int_t **, int_t **);
extern void    dgstrf (superlu_options_t*, SuperMatrix*,
                       int, int, int*, void *, int_t, int *, int *, 
                       SuperMatrix *, SuperMatrix *, GlobalLU_t *,
               SuperLUStat_t*, int_t *info);
extern int_t   dsnode_dfs (const int, const int, const int_t *, const int_t *,
                 const int_t *, int_t *, int *, GlobalLU_t *);

#ifdef __cplusplus
}
#endif

#endif /* __SUPERLU_dSP_DEFS */
extern int     dsnode_bmod (const int, const int, const int, double *,
                              double *, GlobalLU_t *, SuperLUStat_t*);
# 声明一个函数 dsnode_bmod，该函数接受多个参数，执行某种操作并返回一个整数值

extern void    dpanel_dfs (const int, const int, const int, SuperMatrix *,
               int *, int *, double *, int *, int *, int *,
               int_t *, int *, int *, int_t *, GlobalLU_t *);
# 声明一个函数 dpanel_dfs，接受多个参数，执行一种深度优先搜索的操作，不返回任何值

extern void    dpanel_bmod (const int, const int, const int, const int,
                           double *, double *, int *, int *,
               GlobalLU_t *, SuperLUStat_t*);
# 声明一个函数 dpanel_bmod，接受多个参数，执行某种矩阵块操作，不返回任何值

extern int     dcolumn_dfs (const int, const int, int *, int *, int *, int *,
               int *, int_t *, int *, int *, int_t *, GlobalLU_t *);
# 声明一个函数 dcolumn_dfs，接受多个参数，执行某种列优先搜索操作并返回一个整数值

extern int     dcolumn_bmod (const int, const int, double *,
               double *, int *, int *, int,
                           GlobalLU_t *, SuperLUStat_t*);
# 声明一个函数 dcolumn_bmod，接受多个参数，执行某种矩阵列块操作并返回一个整数值

extern int     dcopy_to_ucol (int, int, int *, int *, int *,
                              double *, GlobalLU_t *);
# 声明一个函数 dcopy_to_ucol，接受多个参数，执行某种数据复制到列操作并返回一个整数值

extern int     dpivotL (const int, const double, int *, int *, 
                         int *, int *, int *, GlobalLU_t *, SuperLUStat_t*);
# 声明一个函数 dpivotL，接受多个参数，执行某种主元选取操作并返回一个整数值

extern void    dpruneL (const int, const int *, const int, const int,
              const int *, const int *, int_t *, GlobalLU_t *);
# 声明一个函数 dpruneL，接受多个参数，执行某种剪枝操作，不返回任何值

extern void    dreadmt (int *, int *, int_t *, double **, int_t **, int_t **);
# 声明一个函数 dreadmt，接受多个参数，执行某种矩阵读取操作，不返回任何值

extern void    dGenXtrue (int, int, double *, int);
# 声明一个函数 dGenXtrue，接受多个参数，执行某种生成真实数据操作，不返回任何值

extern void    dFillRHS (trans_t, int, double *, int, SuperMatrix *,
              SuperMatrix *);
# 声明一个函数 dFillRHS，接受多个参数，执行某种填充右侧向量操作，不返回任何值

extern void    dgstrs (trans_t, SuperMatrix *, SuperMatrix *, int *, int *,
                        SuperMatrix *, SuperLUStat_t*, int *);
# 声明一个函数 dgstrs，接受多个参数，执行某种稠密矩阵求解操作，不返回任何值

/* ILU */

extern void    dgsitrf (superlu_options_t*, SuperMatrix*, int, int, int*,
                void *, int_t, int *, int *, SuperMatrix *, SuperMatrix *,
                        GlobalLU_t *, SuperLUStat_t*, int_t *info);
# 声明一个函数 dgsitrf，接受多个参数，执行某种 ILU 分解操作，不返回任何值

extern int     dldperm(int, int, int_t, int_t [], int_t [], double [],
                        int [],    double [], double []);
# 声明一个函数 dldperm，接受多个参数，执行某种对称置换操作并返回一个整数值

extern int     ilu_dsnode_dfs (const int, const int, const int_t *, const int_t *,
                   const int_t *, int *, GlobalLU_t *);
# 声明一个函数 ilu_dsnode_dfs，接受多个参数，执行某种 ILU 的节点深度优先搜索操作并返回一个整数值

extern void    ilu_dpanel_dfs (const int, const int, const int, SuperMatrix *,
                   int *, int *, double *, double *, int *, int *,
                   int *, int *, int *, int_t *, GlobalLU_t *);
# 声明一个函数 ilu_dpanel_dfs，接受多个参数，执行某种 ILU 的面板深度优先搜索操作，不返回任何值

extern int     ilu_dcolumn_dfs (const int, const int, int *, int *, int *,
                int *, int *, int *, int *, int_t *, GlobalLU_t *);
# 声明一个函数 ilu_dcolumn_dfs，接受多个参数，执行某种 ILU 的列深度优先搜索操作并返回一个整数值

extern int     ilu_dcopy_to_ucol (int, int, int *, int *, int *,
                                  double *, int, milu_t, double, int,
                                  double *, int *, GlobalLU_t *, double *);
# 声明一个函数 ilu_dcopy_to_ucol，接受多个参数，执行某种 ILU 的数据复制到列操作并返回一个整数值

extern int     ilu_dpivotL (const int, const double, int *, int *, int, int *,
                int *, int *, int *, double, milu_t,
                            double, GlobalLU_t *, SuperLUStat_t*);
# 声明一个函数 ilu_dpivotL，接受多个参数，执行某种 ILU 的主元选取操作并返回一个整数值
extern int     ilu_ddrop_row (superlu_options_t *, int, int, double,
                              int, int *, double *, GlobalLU_t *, 
                              double *, double *, int);
# 外部函数声明：ilu_ddrop_row，接受多个参数，执行特定的功能

/*! \brief Driver related */
# 驱动程序相关的说明

extern void    dgsequ (SuperMatrix *, double *, double *, double *,
            double *, double *, int *);
# 外部函数声明：dgsequ，计算矩阵的行列比例因子

extern void    dlaqgs (SuperMatrix *, double *, double *, double,
                        double, double, char *);
# 外部函数声明：dlaqgs，根据矩阵的行列比例因子调整矩阵的列

extern void    dgscon (char *, SuperMatrix *, SuperMatrix *, 
                 double, double *, SuperLUStat_t*, int *);
# 外部函数声明：dgscon，对稀疏矩阵进行条件数估计

extern double   dPivotGrowth(int, SuperMatrix *, int *, 
                            SuperMatrix *, SuperMatrix *);
# 外部函数声明：dPivotGrowth，计算 LU 分解中的主元增长因子

extern void    dgsrfs (trans_t, SuperMatrix *, SuperMatrix *,
                       SuperMatrix *, int *, int *, char *, double *, 
                       double *, SuperMatrix *, SuperMatrix *,
                       double *, double *, SuperLUStat_t*, int *);
# 外部函数声明：dgsrfs，解稀疏线性方程组并进行后续分析

extern int     sp_dtrsv (char *, char *, char *, SuperMatrix *,
            SuperMatrix *, double *, SuperLUStat_t*, int *);
# 外部函数声明：sp_dtrsv，稀疏矩阵的三角求解

extern int     sp_dgemv (char *, double, SuperMatrix *, double *,
            int, double, double *, int);
# 外部函数声明：sp_dgemv，稀疏矩阵的一般矩阵向量乘法

extern int     sp_dgemm (char *, char *, int, int, int, double,
            SuperMatrix *, double *, int, double, 
            double *, int);
# 外部函数声明：sp_dgemm，稀疏矩阵的一般矩阵乘法

extern         double dmach(char *);
# 外部函数声明：dmach，获取机器相关参数的精度信息

/*! \brief Memory-related */
# 与内存相关的说明

extern int_t   dLUMemInit (fact_t, void *, int_t, int, int, int_t, int,
                            double, SuperMatrix *, SuperMatrix *,
                            GlobalLU_t *, int **, double **);
# 外部函数声明：dLUMemInit，初始化 LU 分解所需的内存

extern void    dSetRWork (int, int, double *, double **, double **);
# 外部函数声明：dSetRWork，设置实数工作空间的大小和指针

extern void    dLUWorkFree (int *, double *, GlobalLU_t *);
# 外部函数声明：dLUWorkFree，释放 LU 分解过程中使用的工作内存

extern int_t   dLUMemXpand (int, int_t, MemType, int_t *, GlobalLU_t *);
# 外部函数声明：dLUMemXpand，扩展 LU 分解的内存空间

extern double  *doubleMalloc(size_t);
# 外部函数声明：doubleMalloc，分配指定大小的双精度浮点数数组内存

extern double  *doubleCalloc(size_t);
# 外部函数声明：doubleCalloc，分配指定大小的双精度浮点数数组内存，并初始化为零

extern int_t   dmemory_usage(const int_t, const int_t, const int_t, const int);
# 外部函数声明：dmemory_usage，计算内存使用情况

extern int     dQuerySpace (SuperMatrix *, SuperMatrix *, mem_usage_t *);
# 外部函数声明：dQuerySpace，查询稀疏矩阵所需的内存空间

extern int     ilu_dQuerySpace (SuperMatrix *, SuperMatrix *, mem_usage_t *);
# 外部函数声明：ilu_dQuerySpace，查询 ILU 分解所需的内存空间

/*! \brief Auxiliary routines */
# 辅助函数

extern void    dreadhb(FILE *, int *, int *, int_t *, double **, int_t **, int_t **);
# 外部函数声明：dreadhb，从文件中读取 Harwell-Boeing 格式的矩阵

extern void    dreadrb(int *, int *, int_t *, double **, int_t **, int_t **);
# 外部函数声明：dreadrb，从文件中读取 Rutherford-Boeing 格式的矩阵

extern void    dreadtriple(int *, int *, int_t *, double **, int_t **, int_t **);
# 外部函数声明：dreadtriple，从文件中读取三元格式的矩阵

extern void    dreadtriple_noheader(int *, int *, int_t *, double **, int_t **, int_t **);
# 外部函数声明：dreadtriple_noheader，从文件中读取无头部的三元格式矩阵

extern void    dreadMM(FILE *, int *, int *, int_t *, double **, int_t **, int_t **);
# 外部函数声明：dreadMM，从文件中读取 Matrix Market 格式的矩阵

extern void    dfill (double *, int, double);
# 外部函数声明：dfill，将数组用指定的值填充

extern void    dinf_norm_error (int, SuperMatrix *, double *);
# 外部函数声明：dinf_norm_error，计算稠密矩阵与稀疏矩阵之间的无穷范数误差

extern double  dqselect(int, double *, int);
# 外部函数声明：dqselect，快速选择算法，用于从数组中选取第 k 小的元素

/*! \brief Routines for debugging */
# 调试用例程

extern void    dPrint_CompCol_Matrix(char *, SuperMatrix *);
# 外部函数声明：dPrint_CompCol_Matrix，打印压缩列格式的稀疏矩阵

extern void    dPrint_SuperNode_Matrix(char *, SuperMatrix *);
# 外部函数声明：dPrint_SuperNode_Matrix，打印超节点格式的稀疏矩阵

extern void    dPrint_Dense_Matrix(char *, SuperMatrix *);
# 外部函数声明：dPrint_Dense_Matrix，打印稠密矩阵
/*! \brief 外部函数声明：在其他文件中定义的函数声明 */

extern void    dprint_lu_col(char *, int, int, int_t *, GlobalLU_t *);
// 函数声明：打印 LU 分解的某一列

extern int     print_double_vec(char *, int, double *);
// 函数声明：打印双精度向量

extern void    dcheck_tempv(int, double *);
// 函数声明：检查临时向量

/*! \brief BLAS */

extern int dgemm_(const char*, const char*, const int*, const int*, const int*,
                  const double*, const double*, const int*, const double*,
                  const int*, const double*, double*, const int*);
// BLAS 函数声明：矩阵乘法

extern int dtrsv_(char*, char*, char*, int*, double*, int*,
                  double*, int*);
// BLAS 函数声明：解线性方程组的特定类型

extern int dtrsm_(char*, char*, char*, char*, int*, int*,
                  double*, double*, int*, double*, int*);
// BLAS 函数声明：解线性方程组的特定类型，矩阵左右乘法

extern int dgemv_(char *, int *, int *, double *, double *a, int *,
                  double *, int *, double *, double *, int *);
// BLAS 函数声明：向量乘法

extern void dusolve(int, int, double*, double*);
// 函数声明：上三角矩阵的求解

extern void dlsolve(int, int, double*, double*);
// 函数声明：下三角矩阵的求解

extern void dmatvec(int, int, int, double*, double*, double*);
// 函数声明：矩阵乘以向量

#ifdef __cplusplus
  }
#endif

#endif /* __SUPERLU_dSP_DEFS */
// 结束条件：结束头文件的定义
```