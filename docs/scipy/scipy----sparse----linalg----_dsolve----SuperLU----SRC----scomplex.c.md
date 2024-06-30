# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\scomplex.c`

```
/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file scomplex.c
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
#include "slu_scomplex.h"


/*! \brief Complex Division c = a/b */
void c_div(singlecomplex *c, singlecomplex *a, singlecomplex *b)
{
    float ratio, den;
    float abr, abi, cr, ci;
  
    // 如果 b 的实部 abr 小于 0，则取其相反数
    if( (abr = b->r) < 0.)
        abr = - abr;
    // 如果 b 的虚部 abi 小于 0，则取其相反数
    if( (abi = b->i) < 0.)
        abi = - abi;
    
    // 如果 abr <= abi，执行以下操作
    if (abr <= abi) {
        // 如果 abi 等于 0，打印错误信息并退出
        if (abi == 0) {
            fprintf(stderr, "z_div.c: division by zero\n");
            exit(-1);
        }
        
        // 计算 ratio 和 den
        ratio = b->r / b->i;
        den = b->i * (1 + ratio * ratio);
        // 计算复数除法的结果的实部和虚部
        cr = (a->r * ratio + a->i) / den;
        ci = (a->i * ratio - a->r) / den;
    } else {
        // 计算 ratio 和 den
        ratio = b->i / b->r;
        den = b->r * (1 + ratio * ratio);
        // 计算复数除法的结果的实部和虚部
        cr = (a->r + a->i * ratio) / den;
        ci = (a->i - a->r * ratio) / den;
    }
    // 将结果存入 c
    c->r = cr;
    c->i = ci;
}


/*! \brief Returns sqrt(z.r^2 + z.i^2) */
double c_abs(singlecomplex *z)
{
    float temp;
    float real = z->r;
    float imag = z->i;

    // 如果实部 real 小于 0，则取其相反数
    if (real < 0) real = -real;
    // 如果虚部 imag 小于 0，则取其相反数
    if (imag < 0) imag = -imag;
    
    // 如果 imag 大于 real，交换 real 和 imag
    if (imag > real) {
        temp = real;
        real = imag;
        imag = temp;
    }
    
    // 如果 real + imag 等于 real，返回 real
    if ((real + imag) == real) return(real);
  
    // 计算 z 的模长并返回
    temp = imag / real;
    temp = real * sqrt(1.0 + temp * temp);  /*overflow!!*/
    return (temp);
}


/*! \brief Approximates the abs. Returns abs(z.r) + abs(z.i) */
double c_abs1(singlecomplex *z)
{
    float real = z->r;
    float imag = z->i;
  
    // 如果实部 real 小于 0，则取其相反数
    if (real < 0) real = -real;
    // 如果虚部 imag 小于 0，则取其相反数
    if (imag < 0) imag = -imag;

    // 返回实部和虚部的绝对值之和
    return (real + imag);
}

/*! \brief Return the exponentiation */
void c_exp(singlecomplex *r, singlecomplex *z)
{
    float expx;

    // 计算指数函数值
    expx = exp(z->r);
    // 计算复数指数函数的实部和虚部
    r->r = expx * cos(z->i);
    r->i = expx * sin(z->i);
}

/*! \brief Return the complex conjugate */
void r_cnjg(singlecomplex *r, singlecomplex *z)
{
    // 计算复共轭
    r->r = z->r;
    r->i = -z->i;
}

/*! \brief Return the imaginary part */
double r_imag(singlecomplex *z)
{
    // 返回复数的虚部
    return (z->i);
}


/*! \brief SIGN functions for complex number. Returns z/abs(z) */
singlecomplex c_sgn(singlecomplex *z)
{
    register float t = c_abs(z);
    register singlecomplex retval;

    // 如果 t 等于 0，返回 (1, 0)
    if (t == 0.0) {
        retval.r = 1.0, retval.i = 0.0;
    } else {
        // 计算复数的符号函数
        retval.r = z->r / t, retval.i = z->i / t;
    }

    return retval;
}

/*! \brief Square-root of a complex number. */
singlecomplex c_sqrt(singlecomplex *z)
{
    singlecomplex retval;
    # 声明并注册四个浮点数变量 cr, ci, real, imag
    register float cr, ci, real, imag;

    # 将复数结构体 z 中的实部赋给 real
    real = z->r;
    # 将复数结构体 z 中的虚部赋给 imag
    imag = z->i;

    # 如果虚部 imag 等于 0.0
    if ( imag == 0.0 ) {
        # 计算实部的平方根，并赋给 retval 的实部
        retval.r = sqrt(real);
        # 将 retval 的虚部设为 0.0
        retval.i = 0.0;
    } else {
        # 计算模长 sqrt(real*real + imag*imag) 的一半，并赋给 ci
        ci = (sqrt(real*real + imag*imag) - real) / 2.0;
        # 计算 ci 的平方根，并赋给 ci
        ci = sqrt(ci);
        # 计算虚部 imag 除以 (2.0 * ci)，并赋给 cr
        cr = imag / (2.0 * ci);
        # 将 cr 赋给 retval 的实部
        retval.r = cr;
        # 将 ci 赋给 retval 的虚部
        retval.i = ci;
    }

    # 返回复数 retval
    return retval;
}


注释：


# 这行代码结束了一个代码块。在这个上下文中，它结束了一个函数或者其他语法结构的定义。
```