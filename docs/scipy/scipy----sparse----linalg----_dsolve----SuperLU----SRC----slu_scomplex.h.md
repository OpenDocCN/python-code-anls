# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\slu_scomplex.h`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file slu_scomplex.h
 * \brief Header file for complex operations
 * <pre> 
 *  -- SuperLU routine (version 2.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November 15, 1997
 *
 * Contains definitions for various complex operations.
 * This header file is to be included in source files c*.c
 * </pre>
 */
#ifndef __SUPERLU_SCOMPLEX /* allow multiple inclusions */
#define __SUPERLU_SCOMPLEX


#ifndef SCOMPLEX_INCLUDE
#define SCOMPLEX_INCLUDE

typedef struct { float r, i; } singlecomplex;


/* Macro definitions */

/*! \brief Complex Addition c = a + b */
#define c_add(c, a, b) { (c)->r = (a)->r + (b)->r; \
             (c)->i = (a)->i + (b)->i; }

/*! \brief Complex Subtraction c = a - b */
#define c_sub(c, a, b) { (c)->r = (a)->r - (b)->r; \
             (c)->i = (a)->i - (b)->i; }

/*! \brief Complex-Double Multiplication */
#define cs_mult(c, a, b) { (c)->r = (a)->r * (b); \
                           (c)->i = (a)->i * (b); }

/*! \brief Complex-Complex Multiplication */
#define cc_mult(c, a, b) { \
    float cr, ci; \
        cr = (a)->r * (b)->r - (a)->i * (b)->i; \
        ci = (a)->i * (b)->r + (a)->r * (b)->i; \
        (c)->r = cr; \
        (c)->i = ci; \
    }

#define cc_conj(a, b) { \
        (a)->r = (b)->r; \
        (a)->i = -((b)->i); \
    }

/*! \brief Complex equality testing */
#define c_eq(a, b)  ( (a)->r == (b)->r && (a)->i == (b)->i )


#ifdef __cplusplus
extern "C" {
#endif

/* Prototypes for functions in scomplex.c */
void c_div(singlecomplex *, singlecomplex *, singlecomplex *);
double c_abs(singlecomplex *);     /* exact */
double c_abs1(singlecomplex *);    /* approximate */
void c_exp(singlecomplex *, singlecomplex *);
void r_cnjg(singlecomplex *, singlecomplex *);
double r_imag(singlecomplex *);
singlecomplex c_sgn(singlecomplex *);
singlecomplex c_sqrt(singlecomplex *);



#ifdef __cplusplus
  }
#endif

#endif

#endif  /* __SUPERLU_SCOMPLEX */


注释：


/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/
/*! @file slu_scomplex.h
 * \brief Header file for complex operations
 * <pre> 
 *  -- SuperLU routine (version 2.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November 15, 1997
 *
 * Contains definitions for various complex operations.
 * This header file is to be included in source files c*.c
 * </pre>
 */
#ifndef __SUPERLU_SCOMPLEX /* allow multiple inclusions */
#define __SUPERLU_SCOMPLEX
#ifndef SCOMPLEX_INCLUDE
#define SCOMPLEX_INCLUDE

// 定义单精度复数结构体
typedef struct { float r, i; } singlecomplex;


/* 宏定义 */

/*! \brief 复数加法 c = a + b */
#define c_add(c, a, b) { (c)->r = (a)->r + (b)->r; \
             (c)->i = (a)->i + (b)->i; }

/*! \brief 复数减法 c = a - b */
#define c_sub(c, a, b) { (c)->r = (a)->r - (b)->r; \
             (c)->i = (a)->i - (b)->i; }

/*! \brief 复数-实数乘法 */
#define cs_mult(c, a, b) { (c)->r = (a)->r * (b); \
                           (c)->i = (a)->i * (b); }

/*! \brief 复数-复数乘法 */
#define cc_mult(c, a, b) { \
    float cr, ci; \
        cr = (a)->r * (b)->r - (a)->i * (b)->i; \
        ci = (a)->i * (b)->r + (a)->r * (b)->i; \
        (c)->r = cr; \
        (c)->i = ci; \
    }

// 复数共轭
#define cc_conj(a, b) { \
        (a)->r = (b)->r; \
        (a)->i = -((b)->i); \
    }

/*! \brief 复数相等性测试 */
#define c_eq(a, b)  ( (a)->r == (b)->r && (a)->i == (b)->i )


#ifdef __cplusplus
extern "C" {
#endif

// scomplex.c 文件中函数的原型声明
void c_div(singlecomplex *, singlecomplex *, singlecomplex *);
double c_abs(singlecomplex *);     /* 精确值 */
double c_abs1(singlecomplex *);    /* 近似值 */
void c_exp(singlecomplex *, singlecomplex *);
void r_cnjg(singlecomplex *, singlecomplex *);
double r_imag(singlecomplex *);
singlecomplex c_sgn(singlecomplex *);
singlecomplex c_sqrt(singlecomplex *);



#ifdef __cplusplus
  }
#endif

#endif

#endif  /* __SUPERLU_SCOMPLEX */
```