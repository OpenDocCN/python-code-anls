# `D:\src\scipysrc\scipy\scipy\special\special\cephes\incbi.h`

```
/*
 * Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 */

/*
 * incbi()
 *
 * Inverse of incomplete beta integral
 *
 * SYNOPSIS:
 *
 * double a, b, x, y, incbi();
 *
 * x = incbi( a, b, y );
 *
 * DESCRIPTION:
 *
 * Given y, the function finds x such that
 *
 * incbet( a, b, x ) = y .
 *
 * The routine performs interval halving or Newton iterations to find the
 * root of incbet(a,b,x) - y = 0.
 *
 * ACCURACY:
 *
 * Relative error:
 * x     a,b
 * arithmetic   domain  domain  # trials    peak       rms
 * IEEE      0,1    .5,10000   50000    5.8e-12   1.3e-13
 * IEEE      0,1   .25,100    100000    1.8e-13   3.9e-15
 * IEEE      0,1     0,5       50000    1.1e-12   5.5e-15
 * VAX       0,1    .5,100     25000    3.5e-14   1.1e-15
 *
 * With a and b constrained to half-integer or integer values:
 * IEEE      0,1    .5,10000   50000    5.8e-12   1.1e-13
 * IEEE      0,1    .5,100    100000    1.7e-14   7.9e-16
 *
 * With a = .5, b constrained to half-integer or integer values:
 * IEEE      0,1    .5,10000   10000    8.3e-11   1.0e-11
 */

/*
 * Cephes Math Library Release 2.4:  March,1996
 * Copyright 1984, 1996 by Stephen L. Moshier
 */

#pragma once

#include "../config.h"
#include "../error.h"

#include "const.h"
#include "gamma.h"
#include "incbet.h"
#include "ndtri.h"

namespace special {
namespace cephes {
        SPECFUN_HOST_DEVICE inline double incbi(double aa, double bb, double yy0) {
            // 定义局部变量
            double a, b, y0, d, y, x, x0, x1, lgm, yp, di, dithresh, yl, yh, xt;
            int i, rflg, dir, nflg;

            // 初始化迭代计数器
            i = 0;
            // 如果 yy0 小于等于 0，返回 0
            if (yy0 <= 0) {
                return (0.0);
            }
            // 如果 yy0 大于等于 1，返回 1
            if (yy0 >= 1.0) {
                return (1.0);
            }
            // 初始化区间端点和边界值
            x0 = 0.0;
            yl = 0.0;
            x1 = 1.0;
            yh = 1.0;
            nflg = 0;

            // 根据参数判断阈值
            if (aa <= 1.0 || bb <= 1.0) {
                dithresh = 1.0e-6;
                rflg = 0;
                a = aa;
                b = bb;
                y0 = yy0;
                // 计算初始点 x
                x = a / (a + b);
                // 计算初始点的 beta 分布值 y
                y = incbet(a, b, x);
                // 转到二分查找步骤
                goto ihalve;
            } else {
                dithresh = 1.0e-4;
            }
            // 近似求解反函数的初始步骤

            // 计算标准正态分布的逆
            yp = -ndtri(yy0);

            // 根据 yy0 的大小选择分布参数顺序
            if (yy0 > 0.5) {
                rflg = 1;
                a = bb;
                b = aa;
                y0 = 1.0 - yy0;
                yp = -yp;
            } else {
                rflg = 0;
                a = aa;
                b = bb;
                y0 = yy0;
            }

            // 计算参数 lgm 和 x
            lgm = (yp * yp - 3.0) / 6.0;
            x = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0));
            d = yp * std::sqrt(x + lgm) / x -
                (1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0)) * (lgm + 5.0 / 6.0 - 2.0 / (3.0 * x));
            d = 2.0 * d;
            // 如果 d 太小，转到下溢情况
            if (d < detail::MINLOG) {
                x = 1.0;
                goto under;
            }
            // 更新 x 并计算对应的 beta 分布值 y
            x = a / (a + b * std::exp(d));
            y = incbet(a, b, x);
            yp = (y - y0) / y0;
            // 如果误差小于 0.2，转到牛顿迭代步骤
            if (std::abs(yp) < 0.2) {
                goto newt;
            }

            // 如果误差较大，采用区间二分法
    ihalve:

        dir = 0;
        // 初始化方向变量为0
        di = 0.5;
        // 设置增量为0.5
        for (i = 0; i < 100; i++) {
            // 循环执行100次
            if (i != 0) {
                // 如果不是第一次迭代
                x = x0 + di * (x1 - x0);
                // 计算新的 x 值
                if (x == 1.0) {
                    // 如果 x 等于1.0
                    x = 1.0 - detail::MACHEP;
                    // 将 x 调整为1.0减去机器精度
                }
                if (x == 0.0) {
                    // 如果 x 等于0.0
                    di = 0.5;
                    // 重置增量为0.5
                    x = x0 + di * (x1 - x0);
                    // 重新计算 x 值
                    if (x == 0.0) {
                        // 如果 x 仍然为0.0
                        goto under;
                        // 跳转到 under 标签处
                    }
                }
                y = incbet(a, b, x);
                // 调用 incbet 函数计算 y 值
                yp = (x1 - x0) / (x1 + x0);
                // 计算 yp 值
                if (std::abs(yp) < dithresh) {
                    // 如果 yp 的绝对值小于 dithresh
                    goto newt;
                    // 跳转到 newt 标签处
                }
                yp = (y - y0) / y0;
                // 计算 yp 值
                if (std::abs(yp) < dithresh) {
                    // 如果 yp 的绝对值小于 dithresh
                    goto newt;
                    // 跳转到 newt 标签处
                }
            }
            if (y < y0) {
                // 如果 y 小于 y0
                x0 = x;
                // 更新 x0
                yl = y;
                // 更新 yl
                if (dir < 0) {
                    // 如果方向小于0
                    dir = 0;
                    // 重置方向为0
                    di = 0.5;
                    // 重置增量为0.5
                } else if (dir > 3) {
                    // 如果方向大于3
                    di = 1.0 - (1.0 - di) * (1.0 - di);
                    // 根据当前增量调整 di 值
                } else if (dir > 1) {
                    // 如果方向大于1
                    di = 0.5 * di + 0.5;
                    // 调整增量 di 的值
                } else {
                    // 否则
                    di = (y0 - y) / (yh - yl);
                    // 计算新的增量 di
                }
                dir += 1;
                // 方向值加1
                if (x0 > 0.75) {
                    // 如果 x0 大于0.75
                    if (rflg == 1) {
                        // 如果 rflg 等于1
                        rflg = 0;
                        // 重置 rflg 为0
                        a = aa;
                        // 更新 a 值
                        b = bb;
                        // 更新 b 值
                        y0 = yy0;
                        // 更新 y0 值
                    } else {
                        // 否则
                        rflg = 1;
                        // 设置 rflg 为1
                        a = bb;
                        // 更新 a 值
                        b = aa;
                        // 更新 b 值
                        y0 = 1.0 - yy0;
                        // 更新 y0 值
                    }
                    x = 1.0 - x;
                    // 调整 x 的值
                    y = incbet(a, b, x);
                    // 调用 incbet 函数计算 y 值
                    x0 = 0.0;
                    // 重置 x0 为0.0
                    yl = 0.0;
                    // 重置 yl 为0.0
                    x1 = 1.0;
                    // 设置 x1 为1.0
                    yh = 1.0;
                    // 设置 yh 为1.0
                    goto ihalve;
                    // 跳转到 ihalve 标签处重新执行
                }
            } else {
                // 否则
                x1 = x;
                // 更新 x1
                if (rflg == 1 && x1 < detail::MACHEP) {
                    // 如果 rflg 等于1且 x1 小于机器精度
                    x = 0.0;
                    // 设置 x 为0.0
                    goto done;
                    // 跳转到 done 标签处结束
                }
                yh = y;
                // 更新 yh
                if (dir > 0) {
                    // 如果方向大于0
                    dir = 0;
                    // 重置方向为0
                    di = 0.5;
                    // 重置增量为0.5
                } else if (dir < -3) {
                    // 如果方向小于-3
                    di = di * di;
                    // 计算新的增量 di
                } else if (dir < -1) {
                    // 如果方向小于-1
                    di = 0.5 * di;
                    // 计算新的增量 di
                } else {
                    // 否则
                    di = (y - y0) / (yh - yl);
                    // 计算新的增量 di
                }
                dir -= 1;
                // 方向值减1
            }
        }
        set_error("incbi", SF_ERROR_LOSS, NULL);
        // 设置错误信息为 "incbi"，类型为 SF_ERROR_LOSS
        if (x0 >= 1.0) {
            // 如果 x0 大于等于1.0
            x = 1.0 - detail::MACHEP;
            // 设置 x 为1.0减去机器精度
            goto done;
            // 跳转到 done 标签处结束
        }
        if (x <= 0.0) {
        under:
            // under 标签处
            set_error("incbi", SF_ERROR_UNDERFLOW, NULL);
            // 设置错误信息为 "incbi"，类型为 SF_ERROR_UNDERFLOW
            x = 0.0;
            // 设置 x 为0.0
            goto done;
            // 跳转到 done 标签处结束
        }
    newt:

        if (nflg) {
            // 如果 nflg 标志已设置，跳转到 done 标签处
            goto done;
        }
        // 设置 nflg 标志为 1，表示已进入循环
        nflg = 1;
        // 计算 lgm 值，这里 lgm 是 a+b 的对数 gamma 函数值减去 a 和 b 的对数 gamma 函数值之和
        lgm = lgam(a + b) - lgam(a) - lgam(b);

        for (i = 0; i < 8; i++) {
            /* Compute the function at this point. */
            // 计算当前点的函数值
            if (i != 0)
                y = incbet(a, b, x);
            // 根据计算结果调整 x 的值，确保 y 的范围在 yl 和 yh 之间
            if (y < yl) {
                x = x0;
                y = yl;
            } else if (y > yh) {
                x = x1;
                y = yh;
            } else if (y < y0) {
                x0 = x;
                yl = y;
            } else {
                x1 = x;
                yh = y;
            }
            // 如果 x 等于 1.0 或 0.0，退出循环
            if (x == 1.0 || x == 0.0) {
                break;
            }
            /* Compute the derivative of the function at this point. */
            // 计算当前点处函数的导数
            d = (a - 1.0) * std::log(x) + (b - 1.0) * std::log(1.0 - x) + lgm;
            // 如果导数 d 小于 MINLOG，跳转到 done 标签处
            if (d < detail::MINLOG) {
                goto done;
            }
            // 如果导数 d 大于 MAXLOG，退出循环
            if (d > detail::MAXLOG) {
                break;
            }
            // 计算 d 的指数值
            d = std::exp(d);
            /* Compute the step to the next approximation of x. */
            // 计算下一个 x 近似值的步长
            d = (y - y0) / d;
            xt = x - d;
            // 根据计算结果调整 xt 的值，确保在 x0 和 x1 之间
            if (xt <= x0) {
                y = (x - x0) / (x1 - x0);
                xt = x0 + 0.5 * y * (x - x0);
                // 如果 xt 小于等于 0.0，退出循环
                if (xt <= 0.0) {
                    break;
                }
            }
            if (xt >= x1) {
                y = (x1 - x) / (x1 - x0);
                xt = x1 - 0.5 * y * (x1 - x);
                // 如果 xt 大于等于 1.0，退出循环
                if (xt >= 1.0)
                    break;
            }
            x = xt;
            // 如果 d/x 的绝对值小于 128.0 * MACHEP，跳转到 done 标签处
            if (std::abs(d / x) < 128.0 * detail::MACHEP) {
                goto done;
            }
        }
        /* Did not converge.  */
        // 未收敛的情况下设置 dithresh 的值为 256.0 * MACHEP，跳转到 ihalve 标签处
        dithresh = 256.0 * detail::MACHEP;
        goto ihalve;

    done:

        if (rflg) {
            // 如果 rflg 标志已设置，根据 x 的值调整其值，确保大于 MACHEP
            if (x <= detail::MACHEP) {
                x = 1.0 - detail::MACHEP;
            } else {
                x = 1.0 - x;
            }
        }
        // 返回最终计算得到的 x 值
        return (x);
    }
} // namespace cephes
} // namespace special
```