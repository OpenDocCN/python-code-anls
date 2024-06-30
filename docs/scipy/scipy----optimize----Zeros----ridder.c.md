# `D:\src\scipysrc\scipy\scipy\optimize\Zeros\ridder.c`

```
/*
 * Originally written by Charles Harris charles.harris@sdl.usu.edu.
 * Modified by Travis Oliphant to not depend on Python.
 */

#include <math.h>
#include "zeros.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))   // 定义取最小值的宏 MIN
#define SIGN(a) ((a) > 0. ? 1. : -1.)      // 定义取符号的宏 SIGN

/* Sets solver_stats->error_num
    SIGNERR for sign_error;
    CONVERR for convergence_error;
*/

// Ridder 方法求根
double
ridder(callback_type f, double xa, double xb, double xtol, double rtol,
       int iter, void *func_data_param, scipy_zeros_info *solver_stats)
{
    int i;
    double dm,dn,xm,xn=0.0,fn,fm,fa,fb,tol;
    solver_stats->error_num = INPROGRESS;  // 初始设定为正在进行中

    tol = xtol + rtol*MIN(fabs(xa), fabs(xb));  // 计算容差
    fa = (*f)(xa, func_data_param);  // 计算函数在 xa 处的值
    fb = (*f)(xb, func_data_param);  // 计算函数在 xb 处的值
    solver_stats->funcalls = 2;  // 记录函数调用次数，初始化为 2 次
    if (fa == 0) {
        solver_stats->error_num = CONVERGED;  // 如果 xa 处已经是零点，设定为已收敛
        return xa;  // 返回零点的近似值
    }
    if (fb == 0) {
        solver_stats->error_num = CONVERGED;  // 如果 xb 处已经是零点，设定为已收敛
        return xb;  // 返回零点的近似值
    }
    if (signbit(fa)==signbit(fb)) {
        solver_stats->error_num = SIGNERR;  // 如果在同一侧，则设定为符号错误
        return 0.;  // 返回错误值
    }

    solver_stats->iterations=0;  // 迭代次数初始化为 0
    for (i=0; i<iter; i++) {  // 迭代循环开始
        solver_stats->iterations++;  // 每次迭代次数加一
        dm = 0.5*(xb - xa);  // 计算区间的一半
        xm = xa + dm;  // 计算中点
        fm = (*f)(xm, func_data_param);  // 计算函数在中点处的值
        dn = SIGN(fb - fa)*dm*fm/sqrt(fm*fm - fa*fb);  // 计算新的迭代步长
        xn = xm - SIGN(dn) * MIN(fabs(dn), fabs(dm) - .5*tol);  // 计算新的近似根
        fn = (*f)(xn, func_data_param);  // 计算函数在新近似根处的值
        solver_stats->funcalls += 2;  // 函数调用次数加二
        if (signbit(fn) != signbit(fm)) {  // 如果新近似根和中点的函数值符号不同
            xa = xn; fa = fn; xb = xm; fb = fm;  // 更新区间和对应的函数值
        }
        else if (signbit(fn) != signbit(fa)) {  // 如果新近似根和 xa 的函数值符号不同
            xb = xn; fb = fn;  // 更新区间右端点和对应的函数值
        }
        else {  // 否则
            xa = xn; fa = fn;  // 更新区间左端点和对应的函数值
        }
        tol = xtol + rtol*xn;  // 更新容差
        if (fn == 0.0 || fabs(xb - xa) < tol) {  // 如果函数值为零或者区间足够小
            solver_stats->error_num = CONVERGED;  // 设定为已收敛
            return xn;  // 返回零点的近似值
        }
    }
    solver_stats->error_num = CONVERR;  // 如果迭代次数达到上限还未收敛，则设定为收敛错误
    return xn;  // 返回最后的近似根
}
```