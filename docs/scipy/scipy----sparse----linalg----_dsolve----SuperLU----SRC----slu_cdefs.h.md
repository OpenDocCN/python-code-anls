# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\slu_cdefs.h`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file slu_cdefs.h
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
#ifndef __SUPERLU_cSP_DEFS /* allow multiple inclusions */
#define __SUPERLU_cSP_DEFS

/*
 * File name:        csp_defs.h
 * Purpose:          Sparse matrix types and function prototypes
 * History:
 */

#ifdef _CRAY
#include <fortran.h>
#endif

#include <math.h>        // 数学函数库
#include <limits.h>      // 定义整数数据类型的大小
#include <stdio.h>       // 标准输入输出库
#include <stdlib.h>      // 标准库函数，包含内存分配函数
#include <stdint.h>      // 定义了整数数据类型
#include <string.h>      // 字符串处理函数
#include "slu_Cnames.h"  // 超级 LU 的命名定义
#include "superlu_config.h"  // 超级 LU 的配置文件
#include "supermatrix.h"  // 超级矩阵的定义
#include "slu_util.h"    // 超级 LU 的实用工具函数
#include "slu_scomplex.h"  // 复数数据类型的定义


/* -------- Prototypes -------- */

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Driver routines */

// CGSSV: 解超级 LU 系统，求解稠密方程组
extern void
cgssv(superlu_options_t *, SuperMatrix *, int *, int *, SuperMatrix *,
      SuperMatrix *, SuperMatrix *, SuperLUStat_t *, int_t *info);

// CGSSVX: 解超级 LU 系统，带选项和求解结果统计
extern void
cgssvx(superlu_options_t *, SuperMatrix *, int *, int *, int *,
       char *, float *, float *, SuperMatrix *, SuperMatrix *,
       void *, int_t lwork, SuperMatrix *, SuperMatrix *,
       float *, float *, float *, float *,
       GlobalLU_t *, mem_usage_t *, SuperLUStat_t *, int_t *info);

// CGSISV: 解超级 ILU 系统
extern void
cgsisv(superlu_options_t *, SuperMatrix *, int *, int *, SuperMatrix *,
      SuperMatrix *, SuperMatrix *, SuperLUStat_t *, int *);

// CGSISX: 解超级 ILU 系统，带选项和求解结果统计
extern void
cgsisx(superlu_options_t *options, SuperMatrix *A, int *perm_c, int *perm_r,
       int *etree, char *equed, float *R, float *C,
       SuperMatrix *L, SuperMatrix *U, void *work, int_t lwork,
       SuperMatrix *B, SuperMatrix *X, float *recip_pivot_growth, float *rcond,
       GlobalLU_t *Glu, mem_usage_t *mem_usage, SuperLUStat_t *stat, int_t *info);


/*! \brief Supernodal LU factor related */

// 创建压缩列存储的稀疏矩阵
extern void
cCreate_CompCol_Matrix(SuperMatrix *, int, int, int_t, singlecomplex *,
               int_t *, int_t *, Stype_t, Dtype_t, Mtype_t);

// 创建压缩行存储的稀疏矩阵
extern void
cCreate_CompRow_Matrix(SuperMatrix *, int, int, int_t, singlecomplex *,
               int_t *, int_t *, Stype_t, Dtype_t, Mtype_t);

// 将压缩行存储的矩阵转换为压缩列存储
extern void cCompRow_to_CompCol(int, int, int_t, singlecomplex*, int_t*, int_t*,
                           singlecomplex **, int_t **, int_t **);

// 复制压缩列存储的矩阵
extern void
cCopy_CompCol_Matrix(SuperMatrix *, SuperMatrix *);

// 创建密集矩阵
extern void
cCreate_Dense_Matrix(SuperMatrix *, int, int, singlecomplex *, int,
             Stype_t, Dtype_t, Mtype_t);

// 创建超级节点矩阵
extern void
cCreate_SuperNode_Matrix(SuperMatrix *, int, int, int_t, singlecomplex *, 
                 int_t *, int_t *, int_t *, int *, int *,
             Stype_t, Dtype_t, Mtype_t);

// 复制密集矩阵
extern void
cCopy_Dense_Matrix(int, int, singlecomplex *, int, singlecomplex *, int);

// 分配工作空间 A
extern void    callocateA (int, int_t, singlecomplex **, int_t **, int_t **);

// 超级 LU 因子分解
extern void    cgstrf (superlu_options_t*, SuperMatrix*,
                       int, int, int*, void *, int_t, int *, int *, 
                       SuperMatrix *, SuperMatrix *, GlobalLU_t *,
               SuperLUStat_t*, int_t *info);

// 超节点 DFS 遍历
extern int_t   csnode_dfs (const int, const int, const int_t *, const int_t *,
                 const int_t *, int_t *, int *, GlobalLU_t *);
extern int     csnode_bmod (const int, const int, const int, singlecomplex *,
                              singlecomplex *, GlobalLU_t *, SuperLUStat_t*);
// 声明一个外部可见的函数 csnode_bmod，参数包括整数和复数单精度类型的指针，以及全局LU数据结构和统计数据结构

extern void    cpanel_dfs (const int, const int, const int, SuperMatrix *,
               int *, int *, singlecomplex *, int *, int *, int *,
               int_t *, int *, int *, int_t *, GlobalLU_t *);
// 声明一个外部可见的函数 cpanel_dfs，参数包括整数、超级矩阵指针、整数指针、单精度复数类型的指针，以及全局LU数据结构

extern void    cpanel_bmod (const int, const int, const int, const int,
                           singlecomplex *, singlecomplex *, int *, int *,
               GlobalLU_t *, SuperLUStat_t*);
// 声明一个外部可见的函数 cpanel_bmod，参数包括多个整数、复数单精度类型的指针，以及全局LU数据结构和统计数据结构

extern int     ccolumn_dfs (const int, const int, int *, int *, int *, int *,
               int *, int_t *, int *, int *, int_t *, GlobalLU_t *);
// 声明一个外部可见的函数 ccolumn_dfs，参数包括整数、整数指针、整数类型指针、整数类型指针、全局LU数据结构

extern int     ccolumn_bmod (const int, const int, singlecomplex *,
               singlecomplex *, int *, int *, int,
                           GlobalLU_t *, SuperLUStat_t*);
// 声明一个外部可见的函数 ccolumn_bmod，参数包括整数、复数单精度类型的指针、整数、全局LU数据结构和统计数据结构

extern int     ccopy_to_ucol (int, int, int *, int *, int *,
                              singlecomplex *, GlobalLU_t *);
// 声明一个外部可见的函数 ccopy_to_ucol，参数包括整数、整数、整数指针、复数单精度类型的指针、全局LU数据结构

extern int     cpivotL (const int, const double, int *, int *, 
                         int *, int *, int *, GlobalLU_t *, SuperLUStat_t*);
// 声明一个外部可见的函数 cpivotL，参数包括整数、双精度浮点数、整数指针、整数指针、整数指针、整数指针、整数指针、全局LU数据结构和统计数据结构

extern void    cpruneL (const int, const int *, const int, const int,
              const int *, const int *, int_t *, GlobalLU_t *);
// 声明一个外部可见的函数 cpruneL，参数包括整数、整数指针、整数、整数、整数指针、整数指针、整数类型指针、全局LU数据结构

extern void    creadmt (int *, int *, int_t *, singlecomplex **, int_t **, int_t **);
// 声明一个外部可见的函数 creadmt，参数包括多个整数类型指针、单精度复数类型的指针和两个整数类型指针的指针

extern void    cGenXtrue (int, int, singlecomplex *, int);
// 声明一个外部可见的函数 cGenXtrue，参数包括整数、整数、单精度复数类型的指针和整数

extern void    cFillRHS (trans_t, int, singlecomplex *, int, SuperMatrix *,
              SuperMatrix *);
// 声明一个外部可见的函数 cFillRHS，参数包括转置类型、整数、单精度复数类型的指针、整数、两个超级矩阵指针

extern void    cgstrs (trans_t, SuperMatrix *, SuperMatrix *, int *, int *,
                        SuperMatrix *, SuperLUStat_t*, int *);
// 声明一个外部可见的函数 cgstrs，参数包括转置类型、两个超级矩阵指针、整数指针、整数指针、超级矩阵指针、统计数据结构指针和整数指针

/* ILU */

extern void    cgsitrf (superlu_options_t*, SuperMatrix*, int, int, int*,
                void *, int_t, int *, int *, SuperMatrix *, SuperMatrix *,
                        GlobalLU_t *, SuperLUStat_t*, int_t *info);
// 声明一个外部可见的函数 cgsitrf，参数包括超LU选项结构指针、超级矩阵指针、整数、整数、整数指针、空指针、整数类型、整数指针、整数指针、两个超级矩阵指针、全局LU数据结构、统计数据结构指针和整数类型指针

extern int     cldperm(int, int, int_t, int_t [], int_t [], singlecomplex [],
                        int [],    float [], float []);
// 声明一个外部可见的函数 cldperm，参数包括整数、整数、整数类型、整数类型数组、整数类型数组、单精度复数类型数组、整数数组、浮点数数组和浮点数数组

extern int     ilu_csnode_dfs (const int, const int, const int_t *, const int_t *,
                   const int_t *, int *, GlobalLU_t *);
// 声明一个外部可见的函数 ilu_csnode_dfs，参数包括整数、整数、整数类型指针、整数类型指针、整数类型指针、整数指针和全局LU数据结构

extern void    ilu_cpanel_dfs (const int, const int, const int, SuperMatrix *,
                   int *, int *, singlecomplex *, float *, int *, int *,
                   int *, int *, int *, int_t *, GlobalLU_t *);
// 声明一个外部可见的函数 ilu_cpanel_dfs，参数包括多个整数、超级矩阵指针、整数指针、单精度复数类型的指针、浮点数类型的指针、整数指针、整数指针、整数指针、整数指针、整数指针、整数指针、整数指针和全局LU数据结构

extern int     ilu_ccolumn_dfs (const int, const int, int *, int *, int *,
                int *, int *, int *, int *, int_t *, GlobalLU_t *);
// 声明一个外部可见的函数 ilu_ccolumn_dfs，参数包括整数、整数、整数指针、整数指针、整数指针、整数指针、整数指针、整数指针、整数指针、整数类型指针和全局LU数据结构

extern int     ilu_ccopy_to_ucol (int, int, int *, int *, int *,
                                  singlecomplex *, int, milu_t, double, int,
                                  singlecomplex *, int *, GlobalLU_t *, float *);
// 声明一个外部可见的函数 ilu_ccopy_to_ucol，参数包括整数、整数、整数指针、整数指针、整数指针、复数单精度类型的指针、整数、milu_t类型、双精度浮点数、整数、复数单精度类型的指针、整数指针、全局LU数据结构和浮点数类型的指针

extern int     ilu_cpivotL (const int, const double, int *, int *, int, int *,
                int *, int *, int *, double, milu_t,
                            singlecomplex, GlobalLU_t *, SuperLUStat_t*);
// 声明一个外部可见的函数 ilu_cpivotL，参数包括整数、双精度浮点数、整数指针、整数指针、整数、整数指针、整数指针、整数指针、整数指针、双精度浮点数、milu_t类型、单精度复数类型、全局LU数据结构
extern int     ilu_cdrop_row (superlu_options_t *, int, int, double,
                              int, int *, double *, GlobalLU_t *, 
                              float *, float *, int);
# 定义一个外部函数 ilu_cdrop_row，接受多个参数，用于执行某种行丢弃操作

/*! \brief Driver related */
# 这是一个与驱动程序相关的注释，指示下面的函数都与驱动程序有关

extern void    cgsequ (SuperMatrix *, float *, float *, float *,
            float *, float *, int *);
# 定义了一个外部函数 cgsequ，接受多个参数，可能用于某种矩阵计算

extern void    claqgs (SuperMatrix *, float *, float *, float,
                        float, float, char *);
# 定义了一个外部函数 claqgs，接受多个参数，可能用于矩阵计算中的某种特定调整

extern void    cgscon (char *, SuperMatrix *, SuperMatrix *, 
                 float, float *, SuperLUStat_t*, int *);
# 定义了一个外部函数 cgscon，接受多个参数，可能用于某种条件数估计计算

extern float   cPivotGrowth(int, SuperMatrix *, int *, 
                            SuperMatrix *, SuperMatrix *);
# 定义了一个外部函数 cPivotGrowth，接受多个参数，用于计算主元增长因子

extern void    cgsrfs (trans_t, SuperMatrix *, SuperMatrix *,
                       SuperMatrix *, int *, int *, char *, float *, 
                       float *, SuperMatrix *, SuperMatrix *,
                       float *, float *, SuperLUStat_t*, int *);
# 定义了一个外部函数 cgsrfs，接受多个参数，用于求解稀疏矩阵方程组

extern int     sp_ctrsv (char *, char *, char *, SuperMatrix *,
            SuperMatrix *, singlecomplex *, SuperLUStat_t*, int *);
# 定义了一个外部函数 sp_ctrsv，接受多个参数，可能用于解决稀疏矩阵的三角线性系统

extern int     sp_cgemv (char *, singlecomplex, SuperMatrix *, singlecomplex *,
            int, singlecomplex, singlecomplex *, int);
# 定义了一个外部函数 sp_cgemv，接受多个参数，用于执行稀疏矩阵的复数向量乘法

extern int     sp_cgemm (char *, char *, int, int, int, singlecomplex,
            SuperMatrix *, singlecomplex *, int, singlecomplex, 
            singlecomplex *, int);
# 定义了一个外部函数 sp_cgemm，接受多个参数，用于执行稀疏矩阵的复数矩阵乘法

extern         float smach(char *);
# 声明了一个外部函数 smach，接受一个参数，可能与数值计算的机器精度有关

/*! \brief Memory-related */
# 这是一个与内存相关的注释，指示下面的函数都与内存操作有关

extern int_t   cLUMemInit (fact_t, void *, int_t, int, int, int_t, int,
                            float, SuperMatrix *, SuperMatrix *,
                            GlobalLU_t *, int **, singlecomplex **);
# 定义了一个外部函数 cLUMemInit，接受多个参数，用于初始化某种 LU 分解的内存

extern void    cSetRWork (int, int, singlecomplex *, singlecomplex **, singlecomplex **);
# 定义了一个外部函数 cSetRWork，接受多个参数，用于设置某种实数工作区

extern void    cLUWorkFree (int *, singlecomplex *, GlobalLU_t *);
# 定义了一个外部函数 cLUWorkFree，接受多个参数，用于释放 LU 分解的工作内存

extern int_t   cLUMemXpand (int, int_t, MemType, int_t *, GlobalLU_t *);
# 定义了一个外部函数 cLUMemXpand，接受多个参数，用于扩展 LU 分解的内存空间

extern singlecomplex  *complexMalloc(size_t);
# 声明了一个外部函数 complexMalloc，返回一个复数类型的指针，用于动态分配内存

extern singlecomplex  *complexCalloc(size_t);
# 声明了一个外部函数 complexCalloc，返回一个复数类型的指针，用于动态分配并清零内存

extern float  *floatMalloc(size_t);
# 声明了一个外部函数 floatMalloc，返回一个单精度浮点数类型的指针，用于动态分配内存

extern float  *floatCalloc(size_t);
# 声明了一个外部函数 floatCalloc，返回一个单精度浮点数类型的指针，用于动态分配并清零内存

extern int_t   cmemory_usage(const int_t, const int_t, const int_t, const int);
# 声明了一个外部函数 cmemory_usage，接受多个参数，用于计算内存使用情况

extern int     cQuerySpace (SuperMatrix *, SuperMatrix *, mem_usage_t *);
# 声明了一个外部函数 cQuerySpace，接受多个参数，用于查询内存空间使用情况

extern int     ilu_cQuerySpace (SuperMatrix *, SuperMatrix *, mem_usage_t *);
# 声明了一个外部函数 ilu_cQuerySpace，接受多个参数，用于查询 ILU 分解的内存空间使用情况

/*! \brief Auxiliary routines */
# 这是一个与辅助程序相关的注释，指示下面的函数都是辅助性质的

extern void    creadhb(FILE *, int *, int *, int_t *, singlecomplex **, int_t **, int_t **);
# 声明了一个外部函数 creadhb，接受多个参数，用于从文件中读取某种带矩阵结构的复数矩阵

extern void    creadrb(int *, int *, int_t *, singlecomplex **, int_t **, int_t **);
# 声明了一个外部函数 creadrb，接受多个参数，用于从文件中读取某种稀疏矩阵的复数矩阵

extern void    creadtriple(int *, int *, int_t *, singlecomplex **, int_t **, int_t **);
# 声明了一个外部函数 creadtriple，接受多个参数，用于从文件中读取某种三元组形式的复数矩阵

extern void    creadMM(FILE *, int *, int *, int_t *, singlecomplex **, int_t **, int_t **);
# 声明了一个外部函数 creadMM，接受多个参数，用于从文件中读取某种矩阵市场格式的复数矩阵

extern void    cfill (singlecomplex *, int, singlecomplex);
# 声明了一个外部函数 cfill，接受多个参数，用于填充一个复数数组的元素

extern void    cinf_norm_error (int, SuperMatrix *, singlecomplex *);
# 声明了一个外部函数 cinf_norm_error，接受多个参数，用于计算某种复杂度的误差

extern float  sqselect(int, float *, int);
# 声明了一个外部函数 sqselect，接受多个参数，可能与某种排序选择算法有关

/*! \brief Routines for debugging */
# 这是一个与调试相关的注释，指示下面的函数都是用于调试目的的

extern void    cPrint_CompCol_Matrix(char *, SuperMatrix *);
# 声明了一个外部函数 cPrint_CompCol_Matrix，接受多个参数，用于打印压缩列存储的复数矩阵
/*! \brief 打印超级节点矩阵到字符数组 */
extern void    cPrint_SuperNode_Matrix(char *, SuperMatrix *);

/*! \brief 打印稠密矩阵到字符数组 */
extern void    cPrint_Dense_Matrix(char *, SuperMatrix *);

/*! \brief 打印LU分解的列 */
extern void    cprint_lu_col(char *, int, int, int_t *, GlobalLU_t *);

/*! \brief 打印双精度向量到字符数组 */
extern int     print_double_vec(char *, int, double *);

/*! \brief 检查临时向量的内容 */
extern void    ccheck_tempv(int, singlecomplex *);

/*! \brief BLAS */

/*! \brief 调用BLAS的矩阵乘法 */
extern int cgemm_(const char*, const char*, const int*, const int*, const int*,
                  const singlecomplex*, const singlecomplex*, const int*, const singlecomplex*,
                  const int*, const singlecomplex*, singlecomplex*, const int*);

/*! \brief 调用BLAS的矩阵向量乘法 */
extern int cgemv_(char *, int *, int *, singlecomplex *, singlecomplex *a, int *,
                  singlecomplex *, int *, singlecomplex *, singlecomplex *, int *);

/*! \brief 调用BLAS的矩阵向量求解 */
extern int ctrsv_(char*, char*, char*, int*, singlecomplex*, int*,
                  singlecomplex*, int*);

/*! \brief 调用BLAS的矩阵解系统 */
extern int ctrsm_(char*, char*, char*, char*, int*, int*,
                  singlecomplex*, singlecomplex*, int*, singlecomplex*, int*);

/*! \brief 调用自定义的解系统方法 */
extern void cusolve(int, int, singlecomplex*, singlecomplex*);

/*! \brief 调用自定义的解系统方法 */
extern void clsolve(int, int, singlecomplex*, singlecomplex*);

/*! \brief 调用自定义的矩阵向量乘法方法 */
extern void cmatvec(int, int, int, singlecomplex*, singlecomplex*, singlecomplex*);

#ifdef __cplusplus
  }
#endif

#endif /* __SUPERLU_cSP_DEFS */
```