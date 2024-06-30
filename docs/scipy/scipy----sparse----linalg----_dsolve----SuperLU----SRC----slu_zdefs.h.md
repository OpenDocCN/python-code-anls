# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\slu_zdefs.h`

```
/*!
 * \file
 * Copyright (c) 2003, The Regents of the University of California, through
 * Lawrence Berkeley National Laboratory (subject to receipt of any required 
 * approvals from U.S. Dept. of Energy) 
 * 
 * All rights reserved. 
 * 
 * The source code is distributed under BSD license, see the file License.txt
 * at the top-level directory.
 */

/*!
 * @file slu_zdefs.h
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
 *   nsuper: \#supernodes = nsuper + 1, numbered [0, nsuper].
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
#ifndef __SUPERLU_zSP_DEFS /* 允许多次包含 */
#define __SUPERLU_zSP_DEFS

/*
 * File name:        zsp_defs.h
 * Purpose:          稀疏矩阵类型和函数原型
 * History:
 */

#ifdef _CRAY
#include <fortran.h>
#endif

#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "slu_Cnames.h"
#include "superlu_config.h"
#include "supermatrix.h"
#include "slu_util.h"
#include "slu_dcomplex.h"


/* -------- 函数原型 -------- */

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief 驱动程序 */
extern void
zgssv(superlu_options_t *, SuperMatrix *, int *, int *, SuperMatrix *,
      SuperMatrix *, SuperMatrix *, SuperLUStat_t *, int_t *info);
extern void
zgssvx(superlu_options_t *, SuperMatrix *, int *, int *, int *,
       char *, double *, double *, SuperMatrix *, SuperMatrix *,
       void *, int_t lwork, SuperMatrix *, SuperMatrix *,
       double *, double *, double *, double *,
       GlobalLU_t *, mem_usage_t *, SuperLUStat_t *, int_t *info);
    /* ILU */
extern void
zgsisv(superlu_options_t *, SuperMatrix *, int *, int *, SuperMatrix *,
      SuperMatrix *, SuperMatrix *, SuperLUStat_t *, int *);
extern void
zgsisx(superlu_options_t *options, SuperMatrix *A, int *perm_c, int *perm_r,
       int *etree, char *equed, double *R, double *C,
       SuperMatrix *L, SuperMatrix *U, void *work, int_t lwork,
       SuperMatrix *B, SuperMatrix *X, double *recip_pivot_growth, double *rcond,
       GlobalLU_t *Glu, mem_usage_t *mem_usage, SuperLUStat_t *stat, int_t *info);


/*! \brief 超节点 LU 因子相关 */
extern void
zCreate_CompCol_Matrix(SuperMatrix *, int, int, int_t, doublecomplex *,
               int_t *, int_t *, Stype_t, Dtype_t, Mtype_t);
extern void
zCreate_CompRow_Matrix(SuperMatrix *, int, int, int_t, doublecomplex *,
               int_t *, int_t *, Stype_t, Dtype_t, Mtype_t);
extern void zCompRow_to_CompCol(int, int, int_t, doublecomplex*, int_t*, int_t*,
                           doublecomplex **, int_t **, int_t **);
extern void
zCopy_CompCol_Matrix(SuperMatrix *, SuperMatrix *);
extern void
zCreate_Dense_Matrix(SuperMatrix *, int, int, doublecomplex *, int,
             Stype_t, Dtype_t, Mtype_t);
extern void
zCreate_SuperNode_Matrix(SuperMatrix *, int, int, int_t, doublecomplex *, 
                 int_t *, int_t *, int_t *, int *, int *,
             Stype_t, Dtype_t, Mtype_t);
extern void
zCopy_Dense_Matrix(int, int, doublecomplex *, int, doublecomplex *, int);

extern void    zallocateA (int, int_t, doublecomplex **, int_t **, int_t **);
extern void    zgstrf (superlu_options_t*, SuperMatrix*,
                       int, int, int*, void *, int_t, int *, int *, 
                       SuperMatrix *, SuperMatrix *, GlobalLU_t *,
               SuperLUStat_t*, int_t *info);
extern int_t   zsnode_dfs (const int, const int, const int_t *, const int_t *,
                 const int_t *, int_t *, int *, GlobalLU_t *);

#endif /* __SUPERLU_zSP_DEFS */
# 定义外部函数 zsnode_bmod，计算并修改超节点
extern int zsnode_bmod (const int, const int, const int, doublecomplex *,
                        doublecomplex *, GlobalLU_t *, SuperLUStat_t*);

# 定义外部函数 zpanel_dfs，深度优先搜索并修改面板
extern void zpanel_dfs (const int, const int, const int, SuperMatrix *,
                        int *, int *, doublecomplex *, int *, int *, int *,
                        int_t *, int *, int *, int_t *, GlobalLU_t *);

# 定义外部函数 zpanel_bmod，修改面板并进行乘法
extern void zpanel_bmod (const int, const int, const int, const int,
                         doublecomplex *, doublecomplex *, int *, int *,
                         GlobalLU_t *, SuperLUStat_t*);

# 定义外部函数 zcolumn_dfs，深度优先搜索并修改列
extern int zcolumn_dfs (const int, const int, int *, int *, int *, int *,
                        int *, int_t *, int *, int *, int_t *, GlobalLU_t *);

# 定义外部函数 zcolumn_bmod，修改列并进行乘法
extern int zcolumn_bmod (const int, const int, doublecomplex *,
                         doublecomplex *, int *, int *, int,
                         GlobalLU_t *, SuperLUStat_t*);

# 定义外部函数 zcopy_to_ucol，复制到未压缩列
extern int zcopy_to_ucol (int, int, int *, int *, int *,
                          doublecomplex *, GlobalLU_t *);

# 定义外部函数 zpivotL，对 L 矩阵进行主元选取和对角线元素处理
extern int zpivotL (const int, const double, int *, int *,
                    int *, int *, int *, GlobalLU_t *, SuperLUStat_t*);

# 定义外部函数 zpruneL，对 L 矩阵进行修剪操作
extern void zpruneL (const int, const int *, const int, const int,
                     const int *, const int *, int_t *, GlobalLU_t *);

# 定义外部函数 zreadmt，读取矩阵
extern void zreadmt (int *, int *, int_t *, doublecomplex **, int_t **, int_t **);

# 定义外部函数 zGenXtrue，生成真实解向量 X
extern void zGenXtrue (int, int, doublecomplex *, int);

# 定义外部函数 zFillRHS，填充右手边的向量
extern void zFillRHS (trans_t, int, doublecomplex *, int, SuperMatrix *,
                      SuperMatrix *);

# 定义外部函数 zgstrs，求解稀疏矩阵方程
extern void zgstrs (trans_t, SuperMatrix *, SuperMatrix *, int *, int *,
                    SuperMatrix *, SuperLUStat_t*, int *);

# ILU 模块的函数声明开始

# 定义外部函数 zgsitrf，ILU 分解
extern void zgsitrf (superlu_options_t*, SuperMatrix*, int, int, int*,
                     void *, int_t, int *, int *, SuperMatrix *, SuperMatrix *,
                     GlobalLU_t *, SuperLUStat_t*, int_t *info);

# 定义外部函数 zldperm，对列进行置换
extern int zldperm(int, int, int_t, int_t [], int_t [], doublecomplex [],
                   int [], double [], double []);

# 定义外部函数 ilu_zsnode_dfs，ILU 的超节点深度优先搜索
extern int ilu_zsnode_dfs (const int, const int, const int_t *, const int_t *,
                           const int_t *, int *, GlobalLU_t *);

# 定义外部函数 ilu_zpanel_dfs，ILU 的面板深度优先搜索
extern void ilu_zpanel_dfs (const int, const int, const int, SuperMatrix *,
                            int *, int *, doublecomplex *, double *, int *, int *,
                            int *, int *, int *, int_t *, GlobalLU_t *);

# 定义外部函数 ilu_zcolumn_dfs，ILU 的列深度优先搜索
extern int ilu_zcolumn_dfs (const int, const int, int *, int *, int *,
                            int *, int *, int *, int *, int_t *, GlobalLU_t *);

# 定义外部函数 ilu_zcopy_to_ucol，ILU 的复制到未压缩列
extern int ilu_zcopy_to_ucol (int, int, int *, int *, int *,
                              doublecomplex *, int, milu_t, double, int,
                              doublecomplex *, int *, GlobalLU_t *, double *);

# 定义外部函数 ilu_zpivotL，ILU 的主元选取和对角线元素处理
extern int ilu_zpivotL (const int, const double, int *, int *, int, int *,
                        int *, int *, int *, double, milu_t,
                        doublecomplex, GlobalLU_t *, SuperLUStat_t*);
extern int     ilu_zdrop_row (superlu_options_t *, int, int, double,
                              int, int *, double *, GlobalLU_t *, 
                              double *, double *, int);
# 声明一个名为ilu_zdrop_row的外部函数，接受多个参数并返回一个整数

/*! \brief Driver related */
# 驱动程序相关

extern void    zgsequ (SuperMatrix *, double *, double *, double *,
            double *, double *, int *);
# 声明一个名为zgsequ的外部函数，接受多个参数并返回空值
# 参数包括多个SuperMatrix指针和double类型指针

extern void    zlaqgs (SuperMatrix *, double *, double *, double,
                        double, double, char *);
# 声明一个名为zlaqgs的外部函数，接受多个参数并返回空值
# 参数包括SuperMatrix指针、double类型指针和char类型指针

extern void    zgscon (char *, SuperMatrix *, SuperMatrix *, 
                 double, double *, SuperLUStat_t*, int *);
# 声明一个名为zgscon的外部函数，接受多个参数并返回空值
# 参数包括char类型指针、多个SuperMatrix指针和double类型指针

extern double   zPivotGrowth(int, SuperMatrix *, int *, 
                            SuperMatrix *, SuperMatrix *);
# 声明一个名为zPivotGrowth的外部函数，接受多个参数并返回double类型
# 参数包括整数、多个SuperMatrix指针和整数指针

extern void    zgsrfs (trans_t, SuperMatrix *, SuperMatrix *,
                       SuperMatrix *, int *, int *, char *, double *, 
                       double *, SuperMatrix *, SuperMatrix *,
                       double *, double *, SuperLUStat_t*, int *);
# 声明一个名为zgsrfs的外部函数，接受多个参数并返回空值
# 参数包括trans_t类型、多个SuperMatrix指针、整数指针、char类型指针和double类型指针

extern int     sp_ztrsv (char *, char *, char *, SuperMatrix *,
            SuperMatrix *, doublecomplex *, SuperLUStat_t*, int *);
# 声明一个名为sp_ztrsv的外部函数，接受多个参数并返回整数
# 参数包括char类型指针、SuperMatrix指针、doublecomplex类型指针和整数指针

extern int     sp_zgemv (char *, doublecomplex, SuperMatrix *, doublecomplex *,
            int, doublecomplex, doublecomplex *, int);
# 声明一个名为sp_zgemv的外部函数，接受多个参数并返回整数
# 参数包括char类型指针、doublecomplex类型、SuperMatrix指针和整数

extern int     sp_zgemm (char *, char *, int, int, int, doublecomplex,
            SuperMatrix *, doublecomplex *, int, doublecomplex, 
            doublecomplex *, int);
# 声明一个名为sp_zgemm的外部函数，接受多个参数并返回整数
# 参数包括char类型指针、int整数和多个doublecomplex类型指针

extern         double dmach(char *);   /* from C99 standard, in float.h */
# 声明一个名为dmach的外部函数，接受char类型指针参数并返回double类型
# 来自C99标准中的float.h头文件

/*! \brief Memory-related */
# 与内存相关

extern int_t   zLUMemInit (fact_t, void *, int_t, int, int, int_t, int,
                            double, SuperMatrix *, SuperMatrix *,
                            GlobalLU_t *, int **, doublecomplex **);
# 声明一个名为zLUMemInit的外部函数，接受多个参数并返回int_t类型
# 参数包括fact_t类型、void指针、int_t类型和多个SuperMatrix指针

extern void    zSetRWork (int, int, doublecomplex *, doublecomplex **, doublecomplex **);
# 声明一个名为zSetRWork的外部函数，接受多个参数并返回空值
# 参数包括整数和多个doublecomplex类型指针

extern void    zLUWorkFree (int *, doublecomplex *, GlobalLU_t *);
# 声明一个名为zLUWorkFree的外部函数，接受多个参数并返回空值
# 参数包括整数、doublecomplex类型指针和GlobalLU_t类型指针

extern int_t   zLUMemXpand (int, int_t, MemType, int_t *, GlobalLU_t *);
# 声明一个名为zLUMemXpand的外部函数，接受多个参数并返回int_t类型
# 参数包括整数、int_t类型、MemType类型和GlobalLU_t类型指针

extern doublecomplex  *doublecomplexMalloc(size_t);
# 声明一个名为doublecomplexMalloc的外部函数，接受size_t类型参数并返回doublecomplex指针

extern doublecomplex  *doublecomplexCalloc(size_t);
# 声明一个名为doublecomplexCalloc的外部函数，接受size_t类型参数并返回doublecomplex指针

extern double  *doubleMalloc(size_t);
# 声明一个名为doubleMalloc的外部函数，接受size_t类型参数并返回double指针

extern double  *doubleCalloc(size_t);
# 声明一个名为doubleCalloc的外部函数，接受size_t类型参数并返回double指针

extern int_t   zmemory_usage(const int_t, const int_t, const int_t, const int);
# 声明一个名为zmemory_usage的外部函数，接受多个int_t类型参数并返回int_t类型
# 参数包括多个int_t类型和整数

extern int     zQuerySpace (SuperMatrix *, SuperMatrix *, mem_usage_t *);
# 声明一个名为zQuerySpace的外部函数，接受多个参数并返回整数
# 参数包括多个SuperMatrix指针和mem_usage_t类型指针

extern int     ilu_zQuerySpace (SuperMatrix *, SuperMatrix *, mem_usage_t *);
# 声明一个名为ilu_zQuerySpace的外部函数，接受多个参数并返回整数
# 参数包括多个SuperMatrix指针和mem_usage_t类型指针

/*! \brief Auxiliary routines */
# 辅助程序例程

extern void    zreadhb(FILE *, int *, int *, int_t *, doublecomplex **, int_t **, int_t **);
# 声明一个名为zreadhb的外部函数，接受多个参数并返回空值
# 参数包括FILE指针、多个int类型指针和doublecomplex类型指针

extern void    zreadrb(int *, int *, int_t *, doublecomplex **, int_t **, int_t **);
# 声明一个名为zreadrb的外部函数，接受多个参数并返回空值
# 参数包括多个int类型指针和doublecomplex类型指针

extern void    zreadtriple(int *, int *, int_t *, doublecomplex **, int_t **, int_t **);
# 声明一个名为zreadtriple的外部函数，接受多个参数并返回空值
# 参数包括多个int类型指针和doublecomplex类型指针

extern void    zreadMM(FILE *, int *, int *, int_t *, doublecomplex **, int_t **, int_t **);
# 声明一个名为zreadMM的外部函数，接受多个参数并返回空值
# 参数包括FILE指针、多个int类型指针和doublecomplex类型指针

extern void    zfill (doublecomplex *, int, doublecomplex);
# 声明一个名为zfill的外部函数，接受多个参数并返回空值
# 参数包括doublecomplex类型指针和整数

extern void    zinf_norm_error (int, SuperMatrix *, doublecomplex *);
# 声明一个名为zinf_norm_error的外部函数，接受多个参数并返回空值
# 参数包括整数、SuperMatrix指针和doublecomplex类型指针

extern double  dqselect(int, double *, int);
# 声明一个名为dqselect的外部函数，接受多个参数并返回double类型
# 参数包括整数和double类型指针

/*! \brief Routines for debugging */
# 调试例程
/*! \brief 打印稀疏压缩列格式的矩阵 */
extern void zPrint_CompCol_Matrix(char *, SuperMatrix *);

/*! \brief 打印超节点格式的矩阵 */
extern void zPrint_SuperNode_Matrix(char *, SuperMatrix *);

/*! \brief 打印稠密矩阵 */
extern void zPrint_Dense_Matrix(char *, SuperMatrix *);

/*! \brief 打印 LU 分解中的列 */
extern void zprint_lu_col(char *, int, int, int_t *, GlobalLU_t *);

/*! \brief 打印双精度向量 */
extern int print_double_vec(char *, int, double *);

/*! \brief 检查临时向量 */
extern void zcheck_tempv(int, doublecomplex *);

/*! \brief BLAS */

/*! \brief 矩阵乘法 */
extern int zgemm_(const char*, const char*, const int*, const int*, const int*,
                  const doublecomplex*, const doublecomplex*, const int*, const doublecomplex*,
                  const int*, const doublecomplex*, doublecomplex*, const int*);

/*! \brief 矩阵向量乘法 */
extern int zgemv_(char *, int *, int *, doublecomplex *, doublecomplex *a, int *,
                  doublecomplex *, int *, doublecomplex *, doublecomplex *, int *);

/*! \brief 解三角矩阵方程 */
extern int ztrsv_(char*, char*, char*, int*, doublecomplex*, int*,
                  doublecomplex*, int*);

/*! \brief 解三角矩阵方程（多个方程）*/
extern int ztrsm_(char*, char*, char*, char*, int*, int*,
                  doublecomplex*, doublecomplex*, int*, doublecomplex*, int*);

/*! \brief 向后解 */
extern void zusolve(int, int, doublecomplex*, doublecomplex*);

/*! \brief 向前解 */
extern void zlsolve(int, int, doublecomplex*, doublecomplex*);

/*! \brief 矩阵向量乘法 */
extern void zmatvec(int, int, int, doublecomplex*, doublecomplex*, doublecomplex*);

#ifdef __cplusplus
  }
#endif

#endif /* __SUPERLU_zSP_DEFS */


注释：
```