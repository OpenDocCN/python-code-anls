# `D:\src\scipysrc\scipy\scipy\optimize\Zeros\brentq.c`

```
/*
   Written by Charles Harris charles.harris@sdl.usu.edu
*/

#include <math.h>
#include "zeros.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

/*
  At the top of the loop the situation is the following:

    1. the root is bracketed between xa and xb
    2. xa is the most recent estimate
    3. xp is the previous estimate
    4. |fp| < |fb|

  The order of xa and xp doesn't matter, but assume xp < xb. Then xa lies to
  the right of xp and the assumption is that xa is increasing towards the root.
  In this situation we will attempt quadratic extrapolation as long as the
  condition

  *  |fa| < |fp| < |fb|

  is satisfied. That is, the function value is decreasing as we go along.
  Note the 4 above implies that the right inequality already holds.

  The first check is that xa is still to the left of the root. If not, xb is
  replaced by xp and the interval reverses, with xb < xa. In this situation
  we will try linear interpolation. That this has happened is signaled by the
  equality xb == xp;

  The second check is that |fa| < |fb|. If this is not the case, we swap
  xa and xb and resort to bisection.
*/

// 定义函数 brentq，用于求解非线性方程 f(x) = 0 的根
double
brentq(callback_type f, double xa, double xb, double xtol, double rtol,
       int iter, void *func_data_param, scipy_zeros_info *solver_stats)
{
    // 初始化变量
    double xpre = xa, xcur = xb;
    double xblk = 0., fpre, fcur, fblk = 0., spre = 0., scur = 0., sbis;
    // 计算容差的两倍作为 delta
    double delta;
    double stry, dpre, dblk;
    int i;
    // 设置初始状态为进行中
    solver_stats->error_num = INPROGRESS;

    // 计算初始点的函数值
    fpre = (*f)(xpre, func_data_param);
    fcur = (*f)(xcur, func_data_param);
    // 记录函数调用次数
    solver_stats->funcalls = 2;
    // 如果初始点已经是根，则直接返回
    if (fpre == 0) {
        solver_stats->error_num = CONVERGED;
        return xpre;
    }
    // 如果另一个初始点已经是根，则直接返回
    if (fcur == 0) {
        solver_stats->error_num = CONVERGED;
        return xcur;
    }
    // 检查初始点的函数值符号是否相同，如果相同则报错
    if (signbit(fpre)==signbit(fcur)) {
        solver_stats->error_num = SIGNERR;
        return 0.;
    }
    // 初始化迭代次数
    solver_stats->iterations = 0;
   `
    # 循环迭代，执行最大 iter 次
    for (i = 0; i < iter; i++) {
        solver_stats->iterations++;  # 统计迭代次数
        # 检查前后两个函数值符号是否不同，更新区间值
        if (fpre != 0 && fcur != 0 &&
        (signbit(fpre) != signbit(fcur))) {
            xblk = xpre;  # 更新区间上限
            fblk = fpre;  # 更新区间上限函数值
            spre = scur = xcur - xpre;  # 设置步长
        }
        # 如果当前块的函数值小于块的函数值，则更新变量
        if (fabs(fblk) < fabs(fcur)) {
            xpre = xcur;  # 更新前一个解
            xcur = xblk;  # 更新当前解
            xblk = xpre;  # 更新块解

            fpre = fcur;  # 更新前一个函数值
            fcur = fblk;  # 更新当前函数值
            fblk = fpre;  # 更新块函数值
        }

        delta = (xtol + rtol*fabs(xcur))/2;  # 计算容差
        sbis = (xblk - xcur)/2;  # 计算二分点
        # 如果当前函数值为零或 bisis 的绝对值小于 delta，收敛
        if (fcur == 0 || fabs(sbis) < delta) {
            solver_stats->error_num = CONVERGED;  # 设置错误码为已收敛
            return xcur;  # 返回当前解
        }

        # 检查步长条件，进行插值或外推
        if (fabs(spre) > delta && fabs(fcur) < fabs(fpre)) {
            if (xpre == xblk) {
                /* 插值 */
                stry = -fcur*(xcur - xpre)/(fcur - fpre);
            }
            else {
                /* 外推 */
                dpre = (fpre - fcur)/(xpre - xcur);
                dblk = (fblk - fcur)/(xblk - xcur);
                stry = -fcur*(fblk*dblk - fpre*dpre)
                    /(dblk*dpre*(fblk - fpre));
            }
            # 检查短步长的有效性
            if (2*fabs(stry) < MIN(fabs(spre), 3*fabs(sbis) - delta)) {
                /* 好的短步长 */
                spre = scur;  # 更新前步长
                scur = stry;  # 更新当前步长
            } else {
                /* 二分 */
                spre = sbis;  # 设置前步长为二分点
                scur = sbis;  # 设置当前步长为二分点
            }
        }
        else {
            /* 二分 */
            spre = sbis;  # 设置前步长为二分点
            scur = sbis;  # 设置当前步长为二分点
        }

        xpre = xcur; fpre = fcur;  # 更新前一个解和函数值
        if (fabs(scur) > delta) {
            xcur += scur;  # 更新当前解
        }
        else {
            xcur += (sbis > 0 ? delta : -delta);  # 使用二分步长更新解
        }

        fcur = (*f)(xcur, func_data_param);  # 计算当前函数值
        solver_stats->funcalls++;  # 统计函数调用次数
    }
    solver_stats->error_num = CONVERR;  # 设置错误码为计算错误
    return xcur;  # 返回当前解
}


注释：


# 这行代码结束了一个代码块，通常用于结束函数、循环或条件语句的定义。
```