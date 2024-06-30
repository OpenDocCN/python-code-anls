# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\slu_Cnames.h`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file slu_Cnames.h
 * \brief Macros defining how C routines will be called
 *
 * <pre>
 * -- SuperLU routine (version 2.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November 1, 1997
 *
 * These macros define how C routines will be called.  ADD_ assumes that
 * they will be called by fortran, which expects C routines to have an
 * underscore postfixed to the name (Suns, and the Intel expect this).
 * NOCHANGE indicates that fortran will be calling, and that it expects
 * the name called by fortran to be identical to that compiled by the C
 * (RS6K's do this).  UPCASE says it expects C routines called by fortran
 * to be in all upcase (CRAY wants this). 
 * </pre>
 */
#ifndef __SUPERLU_CNAMES /* allow multiple inclusions */
#define __SUPERLU_CNAMES

#include "scipy_slu_config.h"

#define ADD_       0
#define ADD__      1
#define NOCHANGE   2
#define UPCASE     3
#define OLD_CRAY   4
#define C_CALL     5

#ifdef UpCase
#define F77_CALL_C  UPCASE
#endif

#ifdef NoChange
#define F77_CALL_C  NOCHANGE
#endif

#ifdef Add_
#define F77_CALL_C  ADD_
#endif

#ifdef Add__
#define F77_CALL_C  ADD__
#endif

#ifdef _CRAY
#define F77_CALL_C  OLD_CRAY
#endif

/* Default */
#ifndef F77_CALL_C
#define F77_CALL_C  ADD_
#endif


#if (F77_CALL_C == ADD_)
/*
 * These defines set up the naming scheme required to have a fortran 77
 * routine call a C routine
 * No redefinition necessary to have following Fortran to C interface:
 *           FORTRAN CALL               C DECLARATION
 *           call dgemm(...)           void dgemm_(...)
 *
 * This is the default.
 */

#endif

#if (F77_CALL_C == ADD__)
/*
 * These defines set up the naming scheme required to have a fortran 77
 * routine call a C routine 
 * for following Fortran to C interface:
 *           FORTRAN CALL               C DECLARATION
 *           call dgemm(...)           void dgemm__(...)
 */
/* BLAS */
#define sswap_    sswap__
#define saxpy_    saxpy__
#define sasum_    sasum__
#define isamax_   isamax__
#define scopy_    scopy__
#define sscal_    sscal__
#define sger_     sger__
#define snrm2_    snrm2__
#define ssymv_    ssymv__
#define sdot_     sdot__
#define saxpy_    saxpy__
#define ssyr2_    ssyr2__
#define srot_     srot__
#define sgemv_    sgemv__
#define strsv_    strsv__
#define sgemm_    sgemm__
#define strsm_    strsm__

#define dswap_    dswap__
#define daxpy_    daxpy__
#define dasum_    dasum__
#define idamax_   idamax__
#define dcopy_    dcopy__
#define dscal_    dscal__
#define dger_     dger__
#define dnrm2_    dnrm2__
#define dsymv_    dsymv__
#define ddot_     ddot__
#define dsyr2_    dsyr2__
/*
 * 定义替换宏，将drot_替换为drot__，用于在C代码中使用Fortran风格的符号
 */
#define drot_     drot__

/*
 * 同上，将dgemv_替换为dgemv__
 */
#define dgemv_    dgemv__

/*
 * 同上，将dtrsv_替换为dtrsv__
 */
#define dtrsv_    dtrsv__

/*
 * 同上，将dgemm_替换为dgemm__
 */
#define dgemm_    dgemm__

/*
 * 同上，将dtrsm_替换为dtrsm__
 */
#define dtrsm_    dtrsm__

/*
 * 同上，将cdotc_替换为cdotc__
 */
#define cdotc_    cdotc__

/*
 * 同上，将dcabs1_替换为dcabs1__
 */
#define dcabs1_   dcabs1__

/*
 * 同上，将cswap_替换为cswap__
 */
#define cswap_    cswap__

/*
 * 同上，将caxpy_替换为caxpy__
 */
#define caxpy_    caxpy__

/*
 * 同上，将scasum_替换为scasum__
 */
#define scasum_   scasum__

/*
 * 同上，将icamax_替换为icamax__
 */
#define icamax_   icamax__

/*
 * 同上，将ccopy_替换为ccopy__
 */
#define ccopy_    ccopy__

/*
 * 同上，将cscal_替换为cscal__
 */
#define cscal_    cscal__

/*
 * 同上，将scnrm2_替换为scnrm2__
 */
#define scnrm2_   scnrm2__

/*
 * 同上，将caxpy_替换为caxpy__
 */
#define caxpy_    caxpy__

/*
 * 同上，将cgemv_替换为cgemv__
 */
#define cgemv_    cgemv__

/*
 * 同上，将ctrsv_替换为ctrsv__
 */
#define ctrsv_    ctrsv__

/*
 * 同上，将cgemm_替换为cgemm__
 */
#define cgemm_    cgemm__

/*
 * 同上，将ctrsm_替换为ctrsm__
 */
#define ctrsm_    ctrsm__

/*
 * 同上，将cgerc_替换为cgerc__
 */
#define cgerc_    cgerc__

/*
 * 同上，将chemv_替换为chemv__
 */
#define chemv_    chemv__

/*
 * 同上，将cher2_替换为cher2__
 */
#define cher2_    cher2__

/*
 * 同上，将zdotc_替换为zdotc__
 */
#define zdotc_    zdotc__

/*
 * 同上，将zswap_替换为zswap__
 */
#define zswap_    zswap__

/*
 * 同上，将zaxpy_替换为zaxpy__
 */
#define zaxpy_    zaxpy__

/*
 * 同上，将dzasum_替换为dzasum__
 */
#define dzasum_   dzasum__

/*
 * 同上，将izamax_替换为izamax__
 */
#define izamax_   izamax__

/*
 * 同上，将zcopy_替换为zcopy__
 */
#define zcopy_    zcopy__

/*
 * 同上，将zscal_替换为zscal__
 */
#define zscal_    zscal__

/*
 * 同上，将dznrm2_替换为dznrm2__
 */
#define dznrm2_   dznrm2__

/*
 * 同上，将zaxpy_替换为zaxpy__
 */
#define zaxpy_    zaxpy__

/*
 * 同上，将zgemv_替换为zgemv__
 */
#define zgemv_    zgemv__

/*
 * 同上，将ztrsv_替换为ztrsv__
 */
#define ztrsv_    ztrsv__

/*
 * 同上，将zgemm_替换为zgemm__
 */
#define zgemm_    zgemm__

/*
 * 同上，将ztrsm_替换为ztrsm__
 */
#define ztrsm_    ztrsm__

/*
 * 同上，将zgerc_替换为zgerc__
 */
#define zgerc_    zgerc__

/*
 * 同上，将zhemv_替换为zhemv__
 */
#define zhemv_    zhemv__

/*
 * 同上，将zher2_替换为zher2__
 */
#define zher2_    zher2__

/*
 * 定义替换宏，将dlacon_替换为dlacon__，用于LAPACK库函数调用
 */
#define dlacon_   dlacon__

/*
 * 同上，将slacon_替换为slacon__
 */
#define slacon_   slacon__

/*
 * 同上，将icmax1_替换为icmax1__
 */
#define icmax1_   icmax1__

/*
 * 同上，将scsum1_替换为scsum1__
 */
#define scsum1_   scsum1__

/*
 * 同上，将clacon_替换为clacon__
 */
#define clacon_   clacon__

/*
 * 同上，将dzsum1_替换为dzsum1__
 */
#define dzsum1_   dzsum1__

/*
 * 同上，将izmax1_替换为izmax1__
 */
#define izmax1_   izmax1__

/*
 * 同上，将zlacon_替换为zlacon__
 */
#define zlacon_   zlacon__

/*
 * 定义Fortran接口替换宏，将c_bridge_dgssv_替换为c_bridge_dgssv__，用于Fortran调用C函数
 */
#define c_bridge_dgssv_ c_bridge_dgssv__

/*
 * 同上，将c_fortran_sgssv_替换为c_fortran_sgssv__
 */
#define c_fortran_sgssv_ c_fortran_sgssv__

/*
 * 同上，将c_fortran_dgssv_替换为c_fortran_dgssv__
 */
#define c_fortran_dgssv_ c_fortran_dgssv__

/*
 * 同上，将c_fortran_cgssv_替换为c_fortran_cgssv__
 */
#define c_fortran_cgssv_ c_fortran_cgssv__

/*
 * 同上，将c_fortran_zgssv_替换为c_fortran_zgssv__
 */
#define c_fortran_zgssv_ c_fortran_zgssv__

#endif

#if (F77_CALL_C == UPCASE)
/*
 * 下面的宏定义设置了Fortran 77调用C函数时所需的命名方案
 * 根据Fortran到C接口的对应关系，设置了这些宏定义
 *           FORTRAN CALL               C DECLARATION
 *           call dgemm(...)           void DGEMM(...)
 */

/*
 * 定义替换宏，将sswap_替换为SSWAP，用于在Fortran 77中调用C函数SSWAP
 */
#define sswap_    SSWAP

/*
 * 同上，将saxpy_替换为SAXPY
 */
#define saxpy_    SAXPY

/*
 * 同上，将sasum_替换为SASUM
 */
#define sasum_    SASUM

/*
 * 同上，将isamax_替换为ISAMAX
 */
#define isamax_   ISAMAX

/*
 * 同上，将scopy_替换为SCOPY
 */
#define scopy_    SCOPY

/*
 * 同上，将sscal_替换为SSCAL
 */
#define sscal_    SSCAL

/*
 * 同上，将sger_替换为SGER
 */
#define sger_     SGER

/*
 * 同上，将snrm2_替换为SNRM2
 */
#define snrm2_    SNRM2

/*
 * 同上，将ssymv_替换为SSYMV
 */
#define ssymv_    SSYMV

/*
 * 同上，将sdot_替换为SDOT
 */
#define sdot_     SDOT

/*
 * 同上，将saxpy_替换为SAXPY
 */
#define saxpy_    SAXPY

/*
 * 同上，将ssyr2_替换为SSYR2
 */
#define ssyr2_
/*
 * 定义一系列宏，用于将 Fortran 调用的函数名映射到 C 函数名
 */

/* BLAS */
#define sswap_    SSWAP       // 将 Fortran 函数名 sswap_ 映射为 C 函数名 SSWAP
#define saxpy_    SAXPY       // 将 Fortran 函数名 saxpy_ 映射为 C 函数名 SAXPY
#define sasum_    SASUM       // 将 Fortran 函数名 sasum_ 映射为 C 函数名 SASUM
#define isamax_   ISAMAX      // 将 Fortran 函数名 isamax_ 映射为 C 函数名 ISAMAX
#define scopy_    SCOPY       // 将 Fortran 函数名 scopy_ 映射为 C 函数名 SCOPY
#define sscal_    SSCAL       // 将 Fortran 函数名 sscal_ 映射为 C 函数名 SSCAL
#define sger_     SGER        // 将 Fortran 函数名 sger_ 映射为 C 函数名 SGER
#define snrm2_    SNRM2       // 将 Fortran 函数名 snrm2_ 映射为 C 函数名 SNRM2
#define ssymv_    SSYMV       // 将 Fortran 函数名 ssymv_ 映射为 C 函数名 SSYMV
#define sdot_     SDOT        // 将 Fortran 函数名 sdot_ 映射为 C 函数名 SDOT
#define ssyr2_    SSYR2       // 将 Fortran 函数名 ssyr2_ 映射为 C 函数名 SSYR2
#define srot_     SROT        // 将 Fortran 函数名 srot_ 映射为 C 函数名 SROT
#define sgemv_    SGEMV       // 将 Fortran 函数名 sgemv_ 映射为 C 函数名 SGEMV
#define strsv_    STRSV       // 将 Fortran 函数名 strsv_ 映射为 C 函数名 STRSV
#define sgemm_    SGEMM       // 将 Fortran 函数名 sgemm_ 映射为 C 函数名 SGEMM
#define strsm_    STRSM       // 将 Fortran 函数名 strsm_ 映射为 C 函数名 STRSM

#define dswap_    SSWAP       // 将 Fortran 函数名 dswap_ 映射为 C 函数名 SSWAP
#define daxpy_    SAXPY       // 将 Fortran 函数名 daxpy_ 映射为 C 函数名 SAXPY
#define dasum_    SASUM       // 将 Fortran 函数名 dasum_ 映射为 C 函数名 SASUM
#define idamax_   ISAMAX      // 将 Fortran 函数名 idamax_ 映射为 C 函数名 ISAMAX
#define dcopy_    SCOPY       // 将 Fortran 函数名 dcopy_ 映射为 C 函数名 SCOPY
#define dscal_    SSCAL       // 将 Fortran 函数名 dscal_ 映射为 C 函数名 SSCAL
#define dger_     SGER        // 将 Fortran 函数名 dger_ 映射为 C 函数名 SGER
#define dnrm2_    SNRM2       // 将 Fortran 函数名 dnrm2_ 映射为 C 函数名 SNRM2
#define dsymv_    SSYMV       // 将 Fortran 函数名 dsymv_ 映射为 C 函数名 SSYMV
#define ddot_     SDOT        // 将 Fortran 函数名 ddot_ 映射为 C 函数名 SDOT
#define dsyr2_    SSYR2       // 将 Fortran 函数名 dsyr2_ 映射为 C 函数名 SSYR2
#define drot_     SROT        // 将 Fortran 函数名 drot_ 映射为 C 函数名 SROT
#define dgemv_    SGEMV       // 将 Fortran 函数名 dgemv_ 映射为 C 函数名 SGEMV
#define dtrsv_    STRSV       // 将 Fortran 函数名 dtrsv_ 映射为 C 函数名 STRSV
#define dgemm_    SGEMM       // 将 Fortran 函数名 dgemm_ 映射为 C 函数名 SGEMM
#define dtrsm_    STRSM       // 将 Fortran 函数名 dtrsm_ 映射为 C 函数名 STRSM

#define cswap_    CSWAP       // 将 Fortran 函数名 cswap_ 映射为 C 函数名 CSWAP
#define caxpy_    CAXPY       // 将 Fortran 函数名 caxpy_ 映射为 C 函数名 CAXPY
#define scasum_   SCASUM      // 将 Fortran 函数名 scasum_ 映射为 C 函数名 SCASUM
#define icamax_   ICAMAX      // 将 Fortran 函数名 icamax_ 映射为 C 函数名 ICAMAX
#define ccopy_    CCOPY       // 将 Fortran 函数名 ccopy_ 映射为 C 函数名 CCOPY
#define cscal_    CSCAL       // 将 Fortran 函数名 cscal_ 映射为 C 函数名 CSCAL
#define scnrm2_   SCNRM2      // 将 Fortran 函数名 scnrm2_ 映射为 C 函数名 SCNRM2
#define cgemv_    CGEMV       // 将 Fortran 函数名 cgemv_ 映射为 C 函数名 CGEMV
#define ctrsv_    CTRSV       // 将 Fortran 函数名 ctrsv_ 映射为 C 函数名 CTRSV
#define cgemm_    CGEMM       // 将 Fortran 函数名 cgemm_ 映射为 C 函数名 CGEMM
#define ctrsm_    CTRSM       // 将 Fortran 函数名 ctrsm_ 映射为 C 函数名 CTRSM
#define cgerc_    CGERC       // 将 Fortran 函数名 cgerc_ 映射为 C 函数名 CGERC
#define chemv_    CHEMV       // 将 Fortran 函数名 chemv_ 映射为 C 函数名 CHEMV
#define cher2_    CHER2       // 将 Fortran 函数名 cher2_ 映射为 C 函数名 CHER2

#define zswap_    ZSWAP       // 将 Fortran 函数名 zswap_ 映射为 C 函数名 ZSWAP
#define zaxpy_    ZAXPY       // 将 Fortran 函数名 zaxpy_ 映射为 C 函数名 ZAXPY
#define dzasum_   DZASUM      // 将 Fortran 函数名 dzasum_ 映射为 C 函数名 DZASUM
#define izamax_   IZAMAX      // 将 Fortran 函数名 izamax_ 映射为 C 函数名 IZAMAX
#define zcopy_    ZCOPY       // 将 Fortran 函数名 zcopy_ 映射为 C 函数名 ZCOPY
#define zscal_    ZSCAL       // 将 Fortran 函数名 zscal_ 映射为 C 函数名 ZSCAL
#define dznrm2_   DZNRM2      // 将 Fortran 函数名 dznrm2_ 映射为 C 函数名 DZNRM2
#define zgemv_    ZGEMV       // 将 Fortran 函数名 zgemv_ 映射为 C 函数名 ZGEMV
#define ztrsv_    ZTRSV       // 将 Fortran 函数名 ztrsv_ 映射为 C 函数名 ZTRSV
#define zgemm_    ZGEMM       // 将 Fortran 函数名 zgemm_ 映射为 C 函数名 ZGEMM
#define ztrsm_    ZTRSM       // 将 Fortran 函数名 ztrsm_ 映射为 C 函数名 ZTRSM
#define zgerc_    ZGERC       // 将 Fortran 函数名 zgerc_ 映射为 C 函数名 ZGERC
#define zhemv_    ZHEMV       // 将 Fortran 函数名 zhemv_ 映射为 C 函数名 ZHEMV
#define zher2_    ZHER2       // 将 Fortran 函数名 zher2_ 映射为 C 函数名 ZHER
/*
 * 这些宏定义设置了 Fortran 77 调用 C 函数的命名方案
 * 用于以下 Fortran 到 C 接口：
 *           FORTRAN 调用               C 声明
 *           call dgemm(...)           void dgemm(...)
 */
/* BLAS */
#define sswap_    sswap           // 定义 Fortran 调用的函数名 sswap_
#define saxpy_    saxpy           // 定义 Fortran 调用的函数名 saxpy_
#define sasum_    sasum           // 定义 Fortran 调用的函数名 sasum_
#define isamax_   isamax          // 定义 Fortran 调用的函数名 isamax_
#define scopy_    scopy           // 定义 Fortran 调用的函数名 scopy_
#define sscal_    sscal           // 定义 Fortran 调用的函数名 sscal_
#define sger_     sger            // 定义 Fortran 调用的函数名 sger_
#define snrm2_    snrm2           // 定义 Fortran 调用的函数名 snrm2_
#define ssymv_    ssymv           // 定义 Fortran 调用的函数名 ssymv_
#define sdot_     sdot            // 定义 Fortran 调用的函数名 sdot_
#define saxpy_    saxpy           // 定义 Fortran 调用的函数名 saxpy_
#define ssyr2_    ssyr2           // 定义 Fortran 调用的函数名 ssyr2_
#define srot_     srot            // 定义 Fortran 调用的函数名 srot_
#define sgemv_    sgemv           // 定义 Fortran 调用的函数名 sgemv_
#define strsv_    strsv           // 定义 Fortran 调用的函数名 strsv_
#define sgemm_    sgemm           // 定义 Fortran 调用的函数名 sgemm_
#define strsm_    strsm           // 定义 Fortran 调用的函数名 strsm_

#define dswap_    dswap           // 定义 Fortran 调用的函数名 dswap_
#define daxpy_    daxpy           // 定义 Fortran 调用的函数名 daxpy_
#define dasum_    dasum           // 定义 Fortran 调用的函数名 dasum_
#define idamax_   idamax          // 定义 Fortran 调用的函数名 idamax_
#define dcopy_    dcopy           // 定义 Fortran 调用的函数名 dcopy_
#define dscal_    dscal           // 定义 Fortran 调用的函数名 dscal_
#define dger_     dger            // 定义 Fortran 调用的函数名 dger_
#define dnrm2_    dnrm2           // 定义 Fortran 调用的函数名 dnrm2_
#define dsymv_    dsymv           // 定义 Fortran 调用的函数名 dsymv_
#define ddot_     ddot            // 定义 Fortran 调用的函数名 ddot_
#define dsyr2_    dsyr2           // 定义 Fortran 调用的函数名 dsyr2_
#define drot_     drot            // 定义 Fortran 调用的函数名 drot_
#define dgemv_    dgemv           // 定义 Fortran 调用的函数名 dgemv_
#define dtrsv_    dtrsv           // 定义 Fortran 调用的函数名 dtrsv_
#define dgemm_    dgemm           // 定义 Fortran 调用的函数名 dgemm_
#define dtrsm_    dtrsm           // 定义 Fortran 调用的函数名 dtrsm_

#define cswap_    cswap           // 定义 Fortran 调用的函数名 cswap_
#define caxpy_    caxpy           // 定义 Fortran 调用的函数名 caxpy_
#define scasum_   scasum          // 定义 Fortran 调用的函数名 scasum_
#define icamax_   icamax          // 定义 Fortran 调用的函数名 icamax_
#define ccopy_    ccopy           // 定义 Fortran 调用的函数名 ccopy_
#define cscal_    cscal           // 定义 Fortran 调用的函数名 cscal_
#define scnrm2_   scnrm2          // 定义 Fortran 调用的函数名 scnrm2_
#define cgemv_    cgemv           // 定义 Fortran 调用的函数名 cgemv_
#define ctrsv_    ctrsv           // 定义 Fortran 调用的函数名 ctrsv_
#define cgemm_    cgemm           // 定义 Fortran 调用的函数名 cgemm_
#define ctrsm_    ctrsm           // 定义 Fortran 调用的函数名 ctrsm_
#define cgerc_    cgerc           // 定义 Fortran 调用的函数名 cgerc_
#define chemv_    chemv           // 定义 Fortran 调用的函数名 chemv_
#define cher2_    cher2           // 定义 Fortran 调用的函数名 cher2_

#define zswap_    zswap           // 定义 Fortran 调用的函数名 zswap_
#define zaxpy_    zaxpy           // 定义 Fortran 调用的函数名 zaxpy_
#define dzasum_   dzasum          // 定义 Fortran 调用的函数名 dzasum_
#define izamax_   izamax          // 定义 Fortran 调用的函数名 izamax_
#define zcopy_    zcopy           // 定义 Fortran 调用的函数名 zcopy_
#define zscal_    zscal           // 定义 Fortran 调用的函数名 zscal_
#define dznrm2_   dznrm2          // 定义 Fortran 调用的函数名 dznrm2_
#define zgemv_    zgemv           // 定义 Fortran 调用的函数名 zgemv_
#define ztrsv_    ztrsv           // 定义 Fortran 调用的函数名 ztrsv_
#define zgemm_    zgemm           // 定义 Fortran 调用的函数名 zgemm_
#define ztrsm_    ztrsm           // 定义 Fortran 调用的函数名 ztrsm_
#define zgerc_    zgerc           // 定义 Fortran 调用的函数名 zgerc_
#define zhemv_    zhemv           // 定义 Fortran 调用的函数名 zhemv_
#define zher2_    zher2           // 定义 Fortran 调用的函数名 zher2_

/* LAPACK */
#define dlacon_   dlacon          // 定义 Fortran 调用的函数名 dlacon_
#define slacon_   slacon          // 定义 Fortran 调用的函数名 slacon_
#define icmax1_   icmax1          // 定义 Fortran 调用的函数名 icmax1_
#define scsum1_   scsum1          // 定义 Fortran 调用的函数名 scsum1_
#define clacon_   clacon          // 定义 Fortran 调用的函数名 clacon_
#define dzsum1_   dzsum1          // 定义 Fortran 调用的函数名 dzsum1_
#define izmax1_   izmax1          // 定义 Fortran 调用的函数名 izmax1_
#define zlacon_   zlacon          // 定义 Fortran 调用的函数名 zlacon_

/* Fortran 接口 */
#define c_bridge_dgssv_ c_bridge_dgssv   // 定义 Fortran 调用的函数名 c_bridge_dgssv_
#define c_fortran_sgssv_ c_fortran_sgssv // 定义 Fortran 调用的函数名 c_fortran_sgssv_
#define c_fortran_dgssv_ c_fortran_dgssv // 定义 Fortran 调用的函数名 c_fortran_dgssv_
#define c_fortran_cgssv_ c_fortran_cgssv // 定义 Fortran 调用的函数名 c_fortran_cgssv_
#define c_fortran_zgssv_ c_fortran_zgssv // 定义 Fortran 调用的函数名 c_fortran_zgssv_

#endif /* __SUPERLU_CNAMES */
*/

#endif
```