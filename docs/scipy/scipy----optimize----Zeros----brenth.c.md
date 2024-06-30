# `D:\src\scipysrc\scipy\scipy\optimize\Zeros\brenth.c`

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

/*
 Function: brenth

 Description:
    This function attempts to find a root of a given function using the
    Brent's method, which combines bisection, secant, and inverse quadratic
    interpolation.

 Parameters:
    - f: Callback function for evaluating the function
    - xa: Left bracket of the initial interval containing the root
    - xb: Right bracket of the initial interval containing the root
    - xtol: Tolerance on the relative error of the root
    - rtol: Tolerance on the relative error of the function value
    - iter: Maximum number of iterations
    - func_data_param: Additional parameters for the callback function
    - solver_stats: Pointer to a struct for storing solver statistics

 Returns:
    - double: Estimated root of the function

 Notes:
    - The function modifies the solver_stats struct to record function calls,
      iterations, and error status during the root-finding process.
*/

double
brenth(callback_type f, double xa, double xb, double xtol, double rtol,
       int iter, void *func_data_param, scipy_zeros_info *solver_stats)
{
    double xpre = xa, xcur = xb;
    double xblk = 0., fpre, fcur, fblk = 0., spre = 0., scur = 0., sbis;
    /* the tolerance is 2*delta */
    double delta;
    double stry, dpre, dblk;
    int i;
    solver_stats->error_num = INPROGRESS;

    // Evaluate the function at the initial estimates
    fpre = (*f)(xpre, func_data_param);
    fcur = (*f)(xcur, func_data_param);
    solver_stats->funcalls = 2;

    // Check if one of the initial estimates is already a root
    if (fpre == 0) {
        solver_stats->error_num = CONVERGED;
        return xpre;
    }
    if (fcur == 0) {
        solver_stats->error_num = CONVERGED;
        return xcur;
    }

    // Check if the signs of the function values at the initial estimates are the same
    if (signbit(fpre) == signbit(fcur)) {
        solver_stats->error_num = SIGNERR;
        return 0.;
    }

    // Initialize the number of iterations
    solver_stats->iterations = 0;
    for (i = 0; i < iter; i++) {
        // 增加迭代次数统计
        solver_stats->iterations++;

        // 检查是否跨越了零点，更新区间和跨越点
        if (fpre != 0 && fcur != 0 && (signbit(fpre) != signbit(fcur))) {
            xblk = xpre;  // 更新区间起点
            fblk = fpre;  // 更新区间起点对应的函数值
            spre = scur = xcur - xpre;  // 更新跨越点
        }

        // 根据函数值大小比较，重新排列历史点、当前点和跨越点
        if (fabs(fblk) < fabs(fcur)) {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;

            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }

        // 计算允许误差范围
        delta = (xtol + rtol * fabs(xcur)) / 2;
        sbis = (xblk - xcur) / 2;

        // 检查是否已找到根或者步长太小，如果是则收敛并返回结果
        if (fcur == 0 || fabs(sbis) < delta) {
            solver_stats->error_num = CONVERGED;
            return xcur;
        }

        // 根据步长和函数值大小判断是插值还是外推
        if (fabs(spre) > delta && fabs(fcur) < fabs(fpre)) {
            if (xpre == xblk) {
                /* interpolate */
                stry = -fcur * (xcur - xpre) / (fcur - fpre);  // 插值计算
            } else {
                /* extrapolate */
                dpre = (fpre - fcur) / (xpre - xcur);
                dblk = (fblk - fcur) / (xblk - xcur);
                stry = -fcur * (fblk - fpre) / (fblk * dpre - fpre * dblk);  // 外推计算
            }

            // 根据计算结果确定是否接受步长
            if (2 * fabs(stry) < MIN(fabs(spre), 3 * fabs(sbis) - delta)) {
                /* accept step */
                spre = scur;
                scur = stry;
            } else {
                /* bisect */
                spre = sbis;
                scur = sbis;
            }
        } else {
            /* bisect */
            spre = sbis;
            scur = sbis;
        }

        // 更新历史点和当前点，并根据步长是否足够大更新当前点
        xpre = xcur;
        fpre = fcur;
        if (fabs(scur) > delta) {
            xcur += scur;
        } else {
            xcur += (sbis > 0 ? delta : -delta);
        }

        // 计算新的函数值，并增加函数调用次数统计
        fcur = (*f)(xcur, func_data_param);
        solver_stats->funcalls++;
    }

    // 如果迭代次数超过了设定值但未找到根，报错并返回当前点
    solver_stats->error_num = CONVERR;
    return xcur;
}



# 这是一个单独的右大括号 '}'，用于结束一个代码块或语句。在这里，它没有对应的左大括号 '{'，可能是代码片段的一部分或语法结构的一部分。
```