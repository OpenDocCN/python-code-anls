# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dcomplex.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file dcomplex.c
 * \brief Common arithmetic for complex type
 *
 * <pre>
 * -- SuperLU routine (version 2.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November 15, 1997
 *
 * This file defines common arithmetic operations for complex type.
 * </pre>
 */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "slu_dcomplex.h"

/*! \brief Complex Division c = a/b */
void z_div(doublecomplex *c, doublecomplex *a, doublecomplex *b)
{
    double ratio, den;
    double abr, abi, cr, ci;
  
    if( (abr = b->r) < 0.)
        abr = - abr;
    if( (abi = b->i) < 0.)
        abi = - abi;
    if( abr <= abi ) {
        if (abi == 0) {
            fprintf(stderr, "z_div.c: division by zero\n");
            exit(-1);
        }      
        ratio = b->r / b->i ;
        den = b->i * (1 + ratio*ratio);
        cr = (a->r*ratio + a->i) / den;
        ci = (a->i*ratio - a->r) / den;
    } else {
        ratio = b->i / b->r ;
        den = b->r * (1 + ratio*ratio);
        cr = (a->r + a->i*ratio) / den;
        ci = (a->i - a->r*ratio) / den;
    }
    c->r = cr;
    c->i = ci;
}

/*! \brief Returns sqrt(z.r^2 + z.i^2) */
double z_abs(doublecomplex *z)
{
    double temp;
    double real = z->r;
    double imag = z->i;

    if (real < 0) real = -real;
    if (imag < 0) imag = -imag;
    if (imag > real) {
        temp = real;
        real = imag;
        imag = temp;
    }
    if ((real+imag) == real) return(real);
  
    temp = imag/real;
    temp = real*sqrt(1.0 + temp*temp);  /*overflow!!*/
    return (temp);
}

/*! \brief Approximates the abs. Returns abs(z.r) + abs(z.i) */
double z_abs1(doublecomplex *z)
{
    double real = z->r;
    double imag = z->i;
  
    if (real < 0) real = -real;
    if (imag < 0) imag = -imag;

    return (real + imag);
}

/*! \brief Return the exponentiation */
void z_exp(doublecomplex *r, doublecomplex *z)
{
    double expx;

    expx = exp(z->r);
    r->r = expx * cos(z->i);
    r->i = expx * sin(z->i);
}

/*! \brief Return the complex conjugate */
void d_cnjg(doublecomplex *r, doublecomplex *z)
{
    r->r = z->r;
    r->i = -z->i;
}

/*! \brief Return the imaginary part */
double d_imag(doublecomplex *z)
{
    return (z->i);
}

/*! \brief SIGN functions for complex number. Returns z/abs(z) */
doublecomplex z_sgn(doublecomplex *z)
{
    register double t = z_abs(z);
    register doublecomplex retval;

    if (t == 0.0) {
        retval.r = 1.0, retval.i = 0.0;
    } else {
        retval.r = z->r / t, retval.i = z->i / t;
    }

    return retval;
}

/*! \brief Square-root of a complex number. */
doublecomplex z_sqrt(doublecomplex *z)
{
    doublecomplex retval;
    # 声明并初始化变量 cr, ci, real, imag 为双精度浮点数
    register double cr, ci, real, imag;

    # 将结构体 z 中的实部赋值给变量 real
    real = z->r;
    # 将结构体 z 中的虚部赋值给变量 imag
    imag = z->i;

    # 如果虚部 imag 等于 0.0
    if ( imag == 0.0 ) {
        # 计算实数部分的平方根并赋值给 retval 的实部
        retval.r = sqrt(real);
        # 将虚数部分置为 0.0
        retval.i = 0.0;
    } else {
        # 计算复数模的平方根并赋值给变量 ci
        ci = (sqrt(real*real + imag*imag) - real) / 2.0;
        # 计算 ci 的平方根并赋值给 ci
        ci = sqrt(ci);
        # 计算实部的 cr 并赋值给 cr
        cr = imag / (2.0 * ci);
        # 将 cr 和 ci 分别赋值给 retval 的实部和虚部
        retval.r = cr;
        retval.i = ci;
    }

    # 返回结构体 retval，其中包含计算得到的复数值
    return retval;
}



# 这行代码仅包含一个右大括号 '}'，用于结束代码块或控制结构。在这个上下文中，它没有任何实际功能，因为没有前置的控制结构或函数定义，仅用作示例以进行注释。
```