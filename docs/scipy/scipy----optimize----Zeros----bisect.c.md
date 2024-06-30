# `D:\src\scipysrc\scipy\scipy\optimize\Zeros\bisect.c`

```
/* Written by Charles Harris charles.harris@sdl.usu.edu */

#include <math.h>
#include "zeros.h"

/* 定义二分法求根函数，接受函数回调 f，初始区间 [xa, xb]，容差 xtol 和 rtol，最大迭代次数 iter，
   函数参数 func_data_param，以及用于记录求解状态的 solver_stats 结构体指针 */
double
bisect(callback_type f, double xa, double xb, double xtol, double rtol,
       int iter, void *func_data_param, scipy_zeros_info *solver_stats)
{
    int i;
    double dm,xm,fm,fa,fb;
    solver_stats->error_num = INPROGRESS;  // 初始化求解状态为进行中

    fa = (*f)(xa, func_data_param);  // 计算左端点 xa 处的函数值 fa
    fb = (*f)(xb, func_data_param);  // 计算右端点 xb 处的函数值 fb
    solver_stats->funcalls = 2;  // 记录函数调用次数，初始为2次（计算 fa 和 fb）

    // 如果左端点 xa 处的函数值为零，表示已找到根
    if (fa == 0) {
        solver_stats->error_num = CONVERGED;  // 设置求解状态为收敛
        return xa;  // 返回找到的根 xa
    }
    
    // 如果右端点 xb 处的函数值为零，表示已找到根
    if (fb == 0) {
        solver_stats->error_num = CONVERGED;  // 设置求解状态为收敛
        return xb;  // 返回找到的根 xb
    }
    
    // 如果 fa 和 fb 同号，二分法无法保证找到根
    if (signbit(fa) == signbit(fb)) {
        solver_stats->error_num = SIGNERR;  // 设置求解状态为符号错误
        return 0.;  // 返回 0，表示未找到有效根
    }
    
    dm = xb - xa;  // 计算当前区间长度
    solver_stats->iterations = 0;  // 初始化迭代次数计数器为 0
    
    // 开始二分法迭代求解
    for (i = 0; i < iter; i++) {
        solver_stats->iterations++;  // 迭代次数加一
        dm *= .5;  // 当前区间长度减半
        xm = xa + dm;  // 计算中点 xm
        fm = (*f)(xm, func_data_param);  // 计算中点处的函数值 fm
        solver_stats->funcalls++;  // 记录函数调用次数
        
        // 如果中点处的函数值与左端点处同号，更新左端点
        if (signbit(fm) == signbit(fa)) {
            xa = xm;
        }
        
        // 如果中点处的函数值为零，或者区间长度小于容差阈值，认为找到根
        if (fm == 0 || fabs(dm) < xtol + rtol * fabs(xm)) {
            solver_stats->error_num = CONVERGED;  // 设置求解状态为收敛
            return xm;  // 返回找到的根 xm
        }
    }
    
    // 如果达到最大迭代次数仍未收敛，设置求解状态为迭代失败
    solver_stats->error_num = CONVERR;
    return xa;  // 返回当前左端点作为近似根
}
```