# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\slu_sdefs.h`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file slu_sdefs.h
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
#ifndef __SUPERLU_sSP_DEFS /* 允许多次包含 */
#define __SUPERLU_sSP_DEFS

/*
 * File name:        ssp_defs.h
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


/* -------- 函数原型 -------- */

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief 驱动程序例程 */
extern void
sgssv(superlu_options_t *, SuperMatrix *, int *, int *, SuperMatrix *,
      SuperMatrix *, SuperMatrix *, SuperLUStat_t *, int_t *info);
extern void
sgssvx(superlu_options_t *, SuperMatrix *, int *, int *, int *,
       char *, float *, float *, SuperMatrix *, SuperMatrix *,
       void *, int_t lwork, SuperMatrix *, SuperMatrix *,
       float *, float *, float *, float *,
       GlobalLU_t *, mem_usage_t *, SuperLUStat_t *, int_t *info);
    /* ILU */
extern void
sgsisv(superlu_options_t *, SuperMatrix *, int *, int *, SuperMatrix *,
      SuperMatrix *, SuperMatrix *, SuperLUStat_t *, int *);
extern void
sgsisx(superlu_options_t *options, SuperMatrix *A, int *perm_c, int *perm_r,
       int *etree, char *equed, float *R, float *C,
       SuperMatrix *L, SuperMatrix *U, void *work, int_t lwork,
       SuperMatrix *B, SuperMatrix *X, float *recip_pivot_growth, float *rcond,
       GlobalLU_t *Glu, mem_usage_t *mem_usage, SuperLUStat_t *stat, int_t *info);


/*! \brief 超节点LU因子相关 */
extern void
sCreate_CompCol_Matrix(SuperMatrix *, int, int, int_t, float *,
               int_t *, int_t *, Stype_t, Dtype_t, Mtype_t);
extern void
sCreate_CompRow_Matrix(SuperMatrix *, int, int, int_t, float *,
               int_t *, int_t *, Stype_t, Dtype_t, Mtype_t);
extern void sCompRow_to_CompCol(int, int, int_t, float*, int_t*, int_t*,
                           float **, int_t **, int_t **);
extern void
sCopy_CompCol_Matrix(SuperMatrix *, SuperMatrix *);
extern void
sCreate_Dense_Matrix(SuperMatrix *, int, int, float *, int,
             Stype_t, Dtype_t, Mtype_t);
extern void
sCreate_SuperNode_Matrix(SuperMatrix *, int, int, int_t, float *, 
                 int_t *, int_t *, int_t *, int *, int *,
             Stype_t, Dtype_t, Mtype_t);
extern void
sCopy_Dense_Matrix(int, int, float *, int, float *, int);

extern void    sallocateA (int, int_t, float **, int_t **, int_t **);
extern void    sgstrf (superlu_options_t*, SuperMatrix*,
                       int, int, int*, void *, int_t, int *, int *, 
                       SuperMatrix *, SuperMatrix *, GlobalLU_t *,
               SuperLUStat_t*, int_t *info);
extern int_t   ssnode_dfs (const int, const int, const int_t *, const int_t *,
                 const int_t *, int_t *, int *, GlobalLU_t *);

#ifdef __cplusplus
}
#endif

#endif /* __SUPERLU_sSP_DEFS */
extern int     ssnode_bmod (const int, const int, const int, float *,
                              float *, GlobalLU_t *, SuperLUStat_t*);
# 声明一个外部可见的函数 ssnode_bmod，接受多个参数，返回一个整数

extern void    spanel_dfs (const int, const int, const int, SuperMatrix *,
               int *, int *, float *, int *, int *, int *,
               int_t *, int *, int *, int_t *, GlobalLU_t *);
# 声明一个外部可见的函数 spanel_dfs，接受多个参数，没有返回值

extern void    spanel_bmod (const int, const int, const int, const int,
                           float *, float *, int *, int *,
               GlobalLU_t *, SuperLUStat_t*);
# 声明一个外部可见的函数 spanel_bmod，接受多个参数，没有返回值

extern int     scolumn_dfs (const int, const int, int *, int *, int *, int *,
               int *, int_t *, int *, int *, int_t *, GlobalLU_t *);
# 声明一个外部可见的函数 scolumn_dfs，接受多个参数，返回一个整数

extern int     scolumn_bmod (const int, const int, float *,
               float *, int *, int *, int,
                           GlobalLU_t *, SuperLUStat_t*);
# 声明一个外部可见的函数 scolumn_bmod，接受多个参数，返回一个整数

extern int     scopy_to_ucol (int, int, int *, int *, int *,
                              float *, GlobalLU_t *);
# 声明一个外部可见的函数 scopy_to_ucol，接受多个参数，返回一个整数

extern int     spivotL (const int, const double, int *, int *, 
                         int *, int *, int *, GlobalLU_t *, SuperLUStat_t*);
# 声明一个外部可见的函数 spivotL，接受多个参数，返回一个整数

extern void    spruneL (const int, const int *, const int, const int,
              const int *, const int *, int_t *, GlobalLU_t *);
# 声明一个外部可见的函数 spruneL，接受多个参数，没有返回值

extern void    sreadmt (int *, int *, int_t *, float **, int_t **, int_t **);
# 声明一个外部可见的函数 sreadmt，接受多个参数，没有返回值

extern void    sGenXtrue (int, int, float *, int);
# 声明一个外部可见的函数 sGenXtrue，接受多个参数，没有返回值

extern void    sFillRHS (trans_t, int, float *, int, SuperMatrix *,
              SuperMatrix *);
# 声明一个外部可见的函数 sFillRHS，接受多个参数，没有返回值

extern void    sgstrs (trans_t, SuperMatrix *, SuperMatrix *, int *, int *,
                        SuperMatrix *, SuperLUStat_t*, int *);
# 声明一个外部可见的函数 sgstrs，接受多个参数，没有返回值

/* ILU */

extern void    sgsitrf (superlu_options_t*, SuperMatrix*, int, int, int*,
                void *, int_t, int *, int *, SuperMatrix *, SuperMatrix *,
                        GlobalLU_t *, SuperLUStat_t*, int_t *info);
# 声明一个外部可见的函数 sgsitrf，接受多个参数，没有返回值

extern int     sldperm(int, int, int_t, int_t [], int_t [], float [],
                        int [],    float [], float []);
# 声明一个外部可见的函数 sldperm，接受多个参数，返回一个整数

extern int     ilu_ssnode_dfs (const int, const int, const int_t *, const int_t *,
                   const int_t *, int *, GlobalLU_t *);
# 声明一个外部可见的函数 ilu_ssnode_dfs，接受多个参数，返回一个整数

extern void    ilu_spanel_dfs (const int, const int, const int, SuperMatrix *,
                   int *, int *, float *, float *, int *, int *,
                   int *, int *, int *, int_t *, GlobalLU_t *);
# 声明一个外部可见的函数 ilu_spanel_dfs，接受多个参数，没有返回值

extern int     ilu_scolumn_dfs (const int, const int, int *, int *, int *,
                int *, int *, int *, int *, int_t *, GlobalLU_t *);
# 声明一个外部可见的函数 ilu_scolumn_dfs，接受多个参数，返回一个整数

extern int     ilu_scopy_to_ucol (int, int, int *, int *, int *,
                                  float *, int, milu_t, double, int,
                                  float *, int *, GlobalLU_t *, float *);
# 声明一个外部可见的函数 ilu_scopy_to_ucol，接受多个参数，返回一个整数

extern int     ilu_spivotL (const int, const double, int *, int *, int, int *,
                int *, int *, int *, double, milu_t,
                            float, GlobalLU_t *, SuperLUStat_t*);
# 声明一个外部可见的函数 ilu_spivotL，接受多个参数，返回一个整数
# 外部函数声明：ilu_sdrop_row
extern int     ilu_sdrop_row (superlu_options_t *, int, int, double,
                              int, int *, double *, GlobalLU_t *, 
                              float *, float *, int);

# 驱动程序相关
# 设置对称矩阵的列缩放因子
extern void    sgsequ (SuperMatrix *, float *, float *, float *,
            float *, float *, int *);
# 对矩阵进行对称矩阵的对角线缩放
extern void    slaqgs (SuperMatrix *, float *, float *, float,
                        float, float, char *);
# 计算对称矩阵的条件数估计值
extern void    sgscon (char *, SuperMatrix *, SuperMatrix *, 
                 float, float *, SuperLUStat_t*, int *);
# 计算选主元过程中的增长因子
extern float   sPivotGrowth(int, SuperMatrix *, int *, 
                            SuperMatrix *, SuperMatrix *);
# 解对称矩阵的稀疏线性方程组
extern int     sp_strsv (char *, char *, char *, SuperMatrix *,
            SuperMatrix *, float *, SuperLUStat_t*, int *);
# 对称矩阵乘以向量
extern int     sp_sgemv (char *, float, SuperMatrix *, float *,
            int, float, float *, int);
# 对称矩阵乘法
extern int     sp_sgemm (char *, char *, int, int, int, float,
            SuperMatrix *, float *, int, float, 
            float *, int);
# 获取浮点运算机器精度
extern float   smach(char *);

# 内存相关
# 初始化 LU 因式分解的内存空间
extern int_t   sLUMemInit (fact_t, void *, int_t, int, int, int_t, int,
                            float, SuperMatrix *, SuperMatrix *,
                            GlobalLU_t *, int **, float **);
# 设置实数工作区数组
extern void    sSetRWork (int, int, float *, float **, float **);
# 释放 LU 因式分解过程中的工作空间
extern void    sLUWorkFree (int *, float *, GlobalLU_t *);
# 扩展 LU 因式分解的内存空间
extern int_t   sLUMemXpand (int, int_t, MemType, int_t *, GlobalLU_t *);
# 分配浮点数数组的内存空间
extern float  *floatMalloc(size_t);
# 分配并初始化浮点数数组的内存空间
extern float  *floatCalloc(size_t);
# 计算内存使用情况
extern int_t   smemory_usage(const int_t, const int_t, const int_t, const int);
# 查询 LU 因式分解过程中需要的内存空间
extern int     sQuerySpace (SuperMatrix *, SuperMatrix *, mem_usage_t *);
# 查询选主元过程中需要的内存空间
extern int     ilu_sQuerySpace (SuperMatrix *, SuperMatrix *, mem_usage_t *);

# 辅助函数
# 读取 Harwell-Boeing 格式文件
extern void    sreadhb(FILE *, int *, int *, int_t *, float **, int_t **, int_t **);
# 读取 Rutherford-Boeing 格式文件
extern void    sreadrb(int *, int *, int_t *, float **, int_t **, int_t **);
# 读取三元组格式文件
extern void    sreadtriple(int *, int *, int_t *, float **, int_t **, int_t **);
# 读取 MatrixMarket 格式文件
extern void    sreadMM(FILE *, int *, int *, int_t *, float **, int_t **, int_t **);
# 填充数组
extern void    sfill (float *, int, float);
# 计算单精度矩阵的无穷范数误差
extern void    sinf_norm_error (int, SuperMatrix *, float *);
# 快速选择算法
extern float  sqselect(int, float *, int);

# 调试相关函数
# 打印压缩列存储格式的稀疏矩阵
extern void    sPrint_CompCol_Matrix(char *, SuperMatrix *);
# 打印超节点存储格式的稀疏矩阵
extern void    sPrint_SuperNode_Matrix(char *, SuperMatrix *);
# 打印密集矩阵
extern void    sPrint_Dense_Matrix(char *, SuperMatrix *);
# 打印 LU 分解的列
extern void    sprint_lu_col(char *, int, int, int_t *, GlobalLU_t *);
# 打印双精度向量
extern int     print_double_vec(char *, int, double *);
/*! \brief 声明一个外部函数用于检查临时向量。
           该函数接受一个整数和一个浮点数数组作为参数。 */
extern void scheck_tempv(int, float *);

/*! \brief BLAS */

/*! \brief 声明 BLAS 库中的 sgemm_ 函数的接口。
           这是一个矩阵乘法函数，具有多个参数，包括矩阵大小、矩阵数据和结果数据。 */
extern int sgemm_(const char*, const char*, const int*, const int*, const int*,
                  const float*, const float*, const int*, const float*,
                  const int*, const float*, float*, const int*);

/*! \brief 声明 BLAS 库中的 strsv_ 函数的接口。
           这是一个求解三角线性系统的函数，接受三个字符参数和多个整数和浮点数参数。 */
extern int strsv_(char*, char*, char*, int*, float*, int*,
                  float*, int*);

/*! \brief 声明 BLAS 库中的 strsm_ 函数的接口。
           这是一个求解三角矩阵线性系统的函数，接受四个字符参数和多个整数和浮点数参数。 */
extern int strsm_(char*, char*, char*, char*, int*, int*,
                  float*, float*, int*, float*, int*);

/*! \brief 声明 BLAS 库中的 sgemv_ 函数的接口。
           这是一个矩阵向量乘法函数，接受多个参数，包括矩阵大小、矩阵数据和结果数据。 */
extern int sgemv_(char *, int *, int *, float *, float *a, int *,
                  float *, int *, float *, float *, int *);

/*! \brief 声明一个外部函数 susolve。
           这个函数用于解决超松弛法中的线性系统，接受整数和两个浮点数数组作为参数。 */
extern void susolve(int, int, float*, float*);

/*! \brief 声明一个外部函数 slsolve。
           这个函数用于解决选项较少的超松弛法中的线性系统，接受整数和两个浮点数数组作为参数。 */
extern void slsolve(int, int, float*, float*);

/*! \brief 声明一个外部函数 smatvec。
           这个函数用于执行矩阵和向量的乘法运算，接受多个整数和三个浮点数数组作为参数。 */
extern void smatvec(int, int, int, float*, float*, float*);

#ifdef __cplusplus
  }
#endif

#endif /* __SUPERLU_sSP_DEFS */
```