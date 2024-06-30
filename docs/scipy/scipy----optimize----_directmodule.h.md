# `D:\src\scipysrc\scipy\scipy\optimize\_directmodule.h`

```
#ifndef DIRECT_H
#define DIRECT_H

#include "Python.h"  // 引入 Python.h 头文件，用于与 Python 解释器进行交互
#include <math.h>    // 引入 math.h 头文件，提供数学函数原型
#include <stdio.h>   // 引入 stdio.h 头文件，提供输入输出函数原型

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

typedef enum {
     DIRECT_ORIGINAL, DIRECT_GABLONSKY
} direct_algorithm;  // 定义 direct_algorithm 枚举类型，包含 DIRECT_ORIGINAL 和 DIRECT_GABLONSKY 两个取值

typedef enum {
     DIRECT_INVALID_BOUNDS = -1,           // 边界无效
     DIRECT_MAXFEVAL_TOOBIG = -2,          // 最大函数评估次数过大
     DIRECT_INIT_FAILED = -3,              // 初始化失败
     DIRECT_SAMPLEPOINTS_FAILED = -4,      // 采样点生成失败
     DIRECT_SAMPLE_FAILED = -5,            // 采样失败
     DIRECT_MAXLEVELS_REACHED = -6,        // 达到最大层级
     DIRECT_MAXFEVAL_EXCEEDED = 1,         // 超过最大函数评估次数
     DIRECT_MAXITER_EXCEEDED = 2,          // 超过最大迭代次数
     DIRECT_GLOBAL_FOUND = 3,              // 找到全局最优解
     DIRECT_VOLTOL = 4,                    // 电压容差
     DIRECT_SIGMATOL = 5,                  // 电导容差

     DIRECT_OUT_OF_MEMORY = -100,          // 内存不足
     DIRECT_INVALID_ARGS = -101,           // 参数无效
     DIRECT_FORCED_STOP = -102             // 强制停止
} direct_return_code;  // 定义 direct_return_code 枚举类型，包含不同的返回代码

typedef struct {
     int numfunc;      // 函数数量
     int numiter;      // 迭代次数
} direct_return_info;  // 定义 direct_return_info 结构体，包含函数数量和迭代次数

#define DIRECT_UNKNOWN_FGLOBAL (-HUGE_VAL)                   // 定义未知全局最优值
#define DIRECT_UNKNOWN_FGLOBAL_RELTOL (0.0)                 // 定义未知全局最优值的相对容差为 0.0

extern PyObject* direct_optimize(
    PyObject* f, double *x, PyObject *x_seq, PyObject *args,
    int dimension,
    const double *lower_bounds, const double *upper_bounds,
    double *minf,
    int max_feval, int max_iter,
    double magic_eps, double magic_eps_abs,
    double volume_reltol, double sigma_reltol,
    int *force_stop,
    double fglobal,
    double fglobal_reltol,
    FILE *logfile,
    direct_algorithm algorithm,
    direct_return_info *info,
    direct_return_code *ret_code,
    PyObject* callback);  // 声明 direct_optimize 函数，返回 PyObject 指针，接受多个参数用于优化

#ifdef __cplusplus
}  /* extern "C" */
#endif /* __cplusplus */

#endif /* DIRECT_H */
```