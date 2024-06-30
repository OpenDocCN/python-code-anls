# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\dmach.c`

```
#include "slu_ddefs.h"
/* 包含自定义的头文件 "slu_ddefs.h" */

#include <float.h>
/* 包含 C 标准库中定义浮点数类型的头文件 */

#include <math.h>
/* 包含 C 标准库中定义数学函数的头文件 */

#include <stdio.h>
/* 包含 C 标准输入输出函数的头文件 */

#include <string.h>
/* 包含 C 标准库中字符串处理函数的头文件 */

double dmach(char *cmach)
{
/*  -- SuperLU auxiliary routine (version 5.0) --   
    This uses C99 standard constants, and is thread safe.

    Must be compiled with -std=c99 flag.


    Purpose   
    =======   

    DMACH returns double precision machine parameters.   

    Arguments   
    =========   

    CMACH   (input) CHARACTER*1   
            Specifies the value to be returned by DMACH:   
            = 'E' or 'e',   DMACH := eps   
            = 'S' or 's ,   DMACH := sfmin   
            = 'B' or 'b',   DMACH := base   
            = 'P' or 'p',   DMACH := eps*base   
            = 'N' or 'n',   DMACH := t   
            = 'R' or 'r',   DMACH := rnd   
            = 'M' or 'm',   DMACH := emin   
            = 'U' or 'u',   DMACH := rmin   
            = 'L' or 'l',   DMACH := emax   
            = 'O' or 'o',   DMACH := rmax   

            where   

            eps   = relative machine precision   
            sfmin = safe minimum, such that 1/sfmin does not overflow   
            base  = base of the machine   
            prec  = eps*base   
            t     = number of (base) digits in the mantissa   
            rnd   = 1.0 when rounding occurs in addition, 0.0 otherwise   
            emin  = minimum exponent before (gradual) underflow   
            rmin  = underflow threshold - base**(emin-1)   
            emax  = largest exponent before overflow   
            rmax  = overflow threshold  - (base**emax)*(1-eps)   

   ===================================================================== 
*/

    double sfmin, small, rmach;

    if (strncmp(cmach, "E", 1)==0) {
    /* 如果输入的 cmach 是 'E' */
    rmach = DBL_EPSILON * 0.5;
    /* 设置 rmach 为 double 类型的机器精度的一半 */
    } else if (strncmp(cmach, "S", 1)==0) {
    /* 如果输入的 cmach 是 'S' */
    sfmin = DBL_MIN;
    /* 设置 sfmin 为 double 类型的最小正数 */
    small = 1. / DBL_MAX;
    /* 计算 double 类型的最大值的倒数 */
    if (small >= sfmin) {
        /* 如果 small 大于等于 sfmin */
        /* Use SMALL plus a bit, to avoid the possibility of rounding   
           causing overflow when computing  1/sfmin. */
        /* 使用 SMALL 加上一点，以避免在计算 1/sfmin 时由于舍入引起溢出 */
        sfmin = small * (DBL_EPSILON*0.5 + 1.);
    }
    rmach = sfmin;
    /* 设置 rmach 为 sfmin */
    } else if (strncmp(cmach, "B", 1)==0) {
    /* 如果输入的 cmach 是 'B' */
    rmach = FLT_RADIX;
    /* 设置 rmach 为浮点数的基数 */
    } else if (strncmp(cmach, "P", 1)==0) {
    /* 如果输入的 cmach 是 'P' */
    rmach = DBL_EPSILON * 0.5 * FLT_RADIX;
    /* 设置 rmach 为 double 类型的机器精度的一半乘以浮点数的基数 */
    } else if (strncmp(cmach, "N", 1)==0) {
    /* 如果输入的 cmach 是 'N' */
    rmach = DBL_MANT_DIG;
    /* 设置 rmach 为 double 类型的尾数位数 */
    } else if (strncmp(cmach, "R", 1)==0) {
    /* 如果输入的 cmach 是 'R' */
    rmach = FLT_ROUNDS;
    /* 设置 rmach 为浮点数的舍入模式 */
    } else if (strncmp(cmach, "M", 1)==0) {
    /* 如果输入的 cmach 是 'M' */
    rmach = DBL_MIN_EXP;
    /* 设置 rmach 为 double 类型的最小指数 */
    } else if (strncmp(cmach, "U", 1)==0) {
    /* 如果输入的 cmach 是 'U' */
    rmach = DBL_MIN;
    /* 设置 rmach 为 double 类型的最小正数 */
    } else if (strncmp(cmach, "L", 1)==0) {
    /* 如果输入的 cmach 是 'L' */
    rmach = DBL_MAX_EXP;
    /* 设置 rmach 为 double 类型的最大指数 */
    } else if (strncmp(cmach, "O", 1)==0) {
    /* 如果输入的 cmach 是 'O' */
    rmach = DBL_MAX;
    /* 设置 rmach 为 double 类型的最大正数 */
    } else {
        // 如果条件不满足，执行以下代码块
        int argument = 0;  // 声明整型变量 argument，并初始化为 0
        input_error("dmach", &argument);  // 调用 input_error 函数，传递字符串 "dmach" 和 argument 的地址
        rmach = 0;  // 将 rmach 变量赋值为 0
    }

    return rmach;  // 返回 rmach 变量的值
} /* end dmach */
```