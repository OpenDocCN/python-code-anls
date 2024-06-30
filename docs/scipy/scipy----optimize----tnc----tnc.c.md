# `D:\src\scipysrc\scipy\scipy\optimize\tnc\tnc.c`

```
/*
 * tnc : 截断牛顿约束最小化算法
 *      使用梯度信息，用C语言实现
 */

/*
 * 版权所有 (c) 2002-2005, Jean-Sebastien Roy (js@jeannot.org)
 *
 * 根据以下条件，无需收取费用即可授予任何获得本软件及相关文档的人员
 * （以下简称“软件”）的许可：
 * 
 * 可自由使用、复制、修改、合并、发布、分发、再授权及/或销售此软件的副本，
 * 并允许被授权人员执行上述操作，前提是遵守以下条件：
 * 
 * 必须在所有副本或重要部分保留以上版权声明和此许可声明。
 * 
 * 本软件按“现状”提供，无任何明示或暗示的担保，
 * 包括但不限于对适销性、特定用途的适用性及非侵权的担保。
 * 在任何情况下，作者或版权持有人均无法对任何索赔、损害或其他责任负责，
 * 无论是在合同诉讼、侵权诉讼或其他诉讼中，与软件或使用或其他交易有关。
 */

/*
 * 此软件是 TNBC 的 C 语言实现，TNBC 是一个截断牛顿最小化包，
 * 最初由 Stephen G. Nash 用 Fortran 开发。
 *
 * 原始源代码可以在以下网址找到：
 * http://iris.gmu.edu/~snash/nash/software/software.html
 *
 * 原 TNBC Fortran 例程版权所有：
 * 
 *   截断牛顿方法：子程序
 *     作者：Stephen G. Nash
 *           信息技术与工程学院
 *           乔治梅森大学
 *           弗吉尼亚州费尔法克斯 22030
 * 
 * C 语言版本由 Elisabeth Nguyen 和 Jean-Sebastien Roy 转换
 * Jean-Sebastien Roy 进行了修改，2001-2002 年间
 * 
 * SciPy 版本源自 TNC 1.3：
 * $Jeannot: tnc.c,v 1.205 2005/01/28 18:27:31 js Exp $
 */

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "tnc.h"

typedef enum {
    TNC_FALSE = 0,
    TNC_TRUE
} logical;

/*
 * 返回码字符串
 */
const char *const tnc_rc_string[11] = {
    "内存分配失败",
    "参数无效（n<0）",
    "不可行（下界大于上界）",
    "达到局部极小值（|pg| ~= 0）",
    "收敛（|f_n-f_(n-1)| ~= 0）",
    "收敛（|x_n-x_(n-1)| ~= 0）",
    "达到最大函数评估次数",
    "线性搜索失败",
    "所有下界都等于上界",
    "无法进展",
    "用户请求结束最小化"
};

/*
 * getptc 返回码
 */
typedef enum {
    GETPTC_OK     = 0,          /* 找到合适的点 */
    GETPTC_EVAL   = 1,          /* 需要进行函数评估 */
    GETPTC_EINVAL = 2,          /* 输入值错误 */
    GETPTC_FAIL   = 3           /* 未找到合适的点 */
} getptc_rc;

/*
 * linearSearch 返回码
 */
typedef enum {
    LS_OK        = 0,           /* Suitable point found */
    LS_MAXFUN    = 1,           /* Max. number of function evaluations reach */
    LS_FAIL      = 2,           /* No suitable point found */
    LS_USERABORT = 3,           /* User requested end of minimization */
    LS_ENOMEM    = 4            /* Memory allocation failed */



// 定义枚举常量 LS_OK 表示找到合适的点
// 定义枚举常量 LS_MAXFUN 表示达到最大函数评估次数
// 定义枚举常量 LS_FAIL 表示未找到合适的点
// 定义枚举常量 LS_USERABORT 表示用户请求终止最小化操作
// 定义枚举常量 LS_ENOMEM 表示内存分配失败
} ls_rc;

/*
 * Prototypes
 */
// 定义 TNC 最小化函数的原型，接受一系列参数，返回 tnc_rc 类型结果
static tnc_rc tnc_minimize(int n, double x[], double *f, double g[],
                           tnc_function * function, void *state,
                           double xscale[], double xoffset[],
                           double *fscale, double low[], double up[],
                           tnc_message messages, int maxCGit,
                           int maxnfeval, int *nfeval, int *niter,
                           double eta, double stepmx, double accuracy,
                           double fmin, double ftol, double xtol,
                           double pgtol, double rescale,
                           tnc_callback * callback);

// 定义 GETPTC 初始化函数的原型，初始化各种参数和变量，并返回 getptc_rc 类型结果
static getptc_rc getptcInit(double *reltol, double *abstol, double tnytol,
                            double eta, double rmu, double xbnd,
                            double *u, double *fu, double *gu,
                            double *xmin, double *fmin, double *gmin,
                            double *xw, double *fw, double *gw, double *a,
                            double *b, double *oldf, double *b1,
                            double *scxbnd, double *e, double *step,
                            double *factor, logical * braktd,
                            double *gtest1, double *gtest2, double *tol);

// 定义 GETPTC 迭代函数的原型，接受大量参数，进行迭代并更新状态，返回 getptc_rc 类型结果
static getptc_rc getptcIter(double big, double
                            rtsmll, double *reltol, double *abstol,
                            double tnytol, double fpresn, double xbnd,
                            double *u, double *fu, double *gu,
                            double *xmin, double *fmin, double *gmin,
                            double *xw, double *fw, double *gw, double *a,
                            double *b, double *oldf, double *b1,
                            double *scxbnd, double *e, double *step,
                            double *factor, logical * braktd,
                            double *gtest1, double *gtest2, double *tol);

// 打印当前迭代状态的函数原型，输出迭代次数、函数值、梯度等信息
static void printCurrentIteration(int n, double f, double g[], int niter,
                                  int nfeval, int pivot[]);

// 计算初始步长的函数原型，基于当前函数值和梯度值等参数计算一个初始步长
static double initialStep(double fnew, double fmin, double gtp,
                          double smax);

// 线性搜索函数的原型，实现 TNC 算法中的线性搜索过程，返回 ls_rc 类型结果
static ls_rc linearSearch(int n, tnc_function * function, void *state,
                          double low[], double up[],
                          double xscale[], double xoffset[], double fscale,
                          int pivot[], double eta, double ftol,
                          double xbnd, double p[], double x[], double *f,
                          double *alpha, double gfull[], int maxnfeval,
                          int *nfeval);
/* 定义函数 `tnc_direction`，实现 TNC 算法的一步迭代 */
static int tnc_direction(double *zsol, double *diagb,
                         double *x, double *g, int n,
                         int maxCGit, int maxnfeval, int *nfeval,
                         logical upd1, double yksk, double yrsr,
                         double *sk, double *yk, double *sr, double *yr,
                         logical lreset, tnc_function * function,
                         void *state, double xscale[], double xoffset[],
                         double fscale, int *pivot, double accuracy,
                         double gnorm, double xnorm, double *low,
                         double *up);

/* 定义函数 `stepMax`，计算最大步长 */
static double stepMax(double step, int n, double x[], double p[],
                      int pivot[], double low[], double up[],
                      double xscale[], double xoffset[]);

/* 定义函数 `setConstraints`，设置活动约束集 */
static void setConstraints(int n, double x[], int pivot[], double xscale[],
                           double xoffset[], double low[], double up[]);

/* 定义函数 `addConstraint`，添加约束 */
static logical addConstraint(int n, double x[], double p[], int pivot[],
                             double low[], double up[], double xscale[],
                             double xoffset[]);

/* 定义函数 `removeConstraint`，移除约束 */
static logical removeConstraint(double gtpnew, double gnorm,
                                double pgtolfs, double f,
                                double fLastConstraint, double g[],
                                int pivot[], int n);

/* 定义函数 `project`，投影操作 */
static void project(int n, double x[], const int pivot[]);

/* 定义函数 `hessianTimesVector`，计算 Hessian 矩阵与向量的乘积 */
static int hessianTimesVector(double v[], double gv[], int n,
                              double x[], double g[],
                              tnc_function * function, void *state,
                              double xscale[], double xoffset[],
                              double fscale, double accuracy, double xnorm,
                              double low[], double up[]);

/* 定义函数 `msolve`，解线性方程组 */
static int msolve(double g[], double *y, int n,
                  double sk[], double yk[], double diagb[], double sr[],
                  double yr[], logical upd1, double yksk, double yrsr,
                  logical lreset);

/* 定义函数 `diagonalScaling`，进行对角线缩放 */
static void diagonalScaling(int n, double e[], double v[], double gv[],
                            double r[]);

/* 定义函数 `ssbfgs`，进行 SS-BFGS 更新 */
static void ssbfgs(int n, double gamma, double sj[], double *hjv,
                   double hjyj[], double yjsj,
                   double yjhyj, double vsj, double vhyj, double hjp1v[]);

/* 定义函数 `initPreconditioner`，初始化预条件器 */
static int initPreconditioner(double diagb[], double emat[], int n,
                              logical lreset, double yksk, double yrsr,
                              double sk[], double yk[], double sr[],
                              double yr[], logical upd1);

/* 定义函数 `coercex`，强制将 `x` 向约束边界 `low` 和 `up` 收敛 */
static void coercex(int n, double x[], const double low[], const double up[]);

/* 定义函数 `unscalex`，取消 `x` 的缩放 */
static void unscalex(int n, double x[], const double xscale[],
                     const double xoffset[]);

/* 定义函数 `scaleg`，对 `g` 进行缩放 */
static void scaleg(int n, double g[], const double xscale[], double fscale);
/*
 * 缩放向量 x，使其乘以 xscale 对应元素，然后加上 xoffset 对应元素
 */
static void scalex(int n, double x[], const double xscale[],
                   const double xoffset[]);

/*
 * 将向量 x 中的元素投影到其对应的约束范围内
 */
static void projectConstants(int n, double x[], const double xscale[]);

/*
 * 计算两个向量 dx 和 dy 的内积，假设增量（步长）为 1
 */
static double ddot1(int n, const double dx[], const double dy[]);

/*
 * 将向量 dx 加到向量 dy 上，假设增量（步长）为 1
 */
static void dxpy1(int n, const double dx[], double dy[]);

/*
 * 计算向量 dx 乘以常数 da 后加到向量 dy 上，假设增量（步长）为 1
 */
static void daxpy1(int n, double da, const double dx[], double dy[]);

/*
 * 将向量 dx 复制到向量 dy
 */
static void dcopy1(int n, const double dx[], double dy[]);

/*
 * 计算向量 dx 的二范数的平方
 */
static double dnrm21(int n, const double dx[]);

/*
 * 将向量 v 中的每个元素取负
 */
static void dneg1(int n, double v[]);

/*
 * 这个函数解决优化问题，寻找函数 f(x) 在约束 low <= x <= up 下的局部最小值。
 * 使用截断牛顿法算法，不假设 f 是凸函数，但假设 f 有下界。
 * 可以处理任意数量的变量，特别适用于变量数较大的情况。
 *
 * 输入参数：
 *   n: 变量的数量
 *   x: 初始的变量向量，输出时为最优解
 *   f: 指向最优值的指针
 *   g: 梯度向量
 *   function: 求解问题的目标函数
 *   state: 算法的状态信息
 *   low: 每个变量的下界
 *   up: 每个变量的上界
 *   scale: 变量的缩放因子
 *   offset: 变量的偏移量
 *   messages: 控制信息输出的开关
 *   maxCGit: 最大共轭梯度法迭代次数
 *   maxnfeval: 最大函数评估次数
 *   eta: 搜索步长的收缩比率
 *   stepmx: 最大步长
 *   accuracy: 收敛精度
 *   fmin: 最小目标函数值
 *   ftol: 相对函数收敛精度
 *   xtol: 变量变化量收敛精度
 *   pgtol: 两次迭代之间梯度范数的绝对精度
 *   rescale: 控制是否重新缩放变量
 *   nfeval: 指向函数评估次数的指针
 *   niter: 指向迭代次数的指针
 *   callback: 用户提供的回调函数
 *
 * 返回值：
 *   成功时返回 0，出错时返回错误码
 */
int tnc(int n, double x[], double *f, double g[], tnc_function * function,
        void *state, double low[], double up[], double scale[],
        double offset[], int messages, int maxCGit, int maxnfeval,
        double eta, double stepmx, double accuracy, double fmin,
        double ftol, double xtol, double pgtol, double rescale,
        int *nfeval, int *niter, tnc_callback * callback)
{
    int rc, frc, i, nc, nfeval_local, free_low = TNC_FALSE,
        free_up = TNC_FALSE, free_g = TNC_FALSE;
    double *xscale = NULL, fscale, rteps, *xoffset = NULL;

    if (nfeval == NULL) {
        /* 忽略 nfeval */
        nfeval = &nfeval_local;
    }
    *nfeval = 0;

    /* 检查输入参数中的错误 */
    if (n == 0) {
        rc = TNC_CONSTANT;
        goto cleanup;
    }

    if (n < 0) {
        rc = TNC_EINVAL;
        goto cleanup;
    }

    /* 检查边界数组 */
    if (low == NULL) {
        low = malloc(n * sizeof(*low));
        if (low == NULL) {
            rc = TNC_ENOMEM;
            goto cleanup;
        }
        free_low = TNC_TRUE;
        for (i = 0; i < n; i++) {
            low[i] = -HUGE_VAL;
        }
    }

    if (up == NULL) {
        up = malloc(n * sizeof(*up));
        if (up == NULL) {
            rc = TNC_ENOMEM;
            goto cleanup;
        }
        free_up = TNC_TRUE;
        for (i = 0; i < n; i++) {
            up[i] = HUGE_VAL;
        }
    }

    /* 一致性检查 */
    for (i = 0; i < n; i++) {
        if (low[i] > up[i]) {
            rc = TNC_INFEASIBLE;
            goto cleanup;
        }
    }

    /* 强制 x 在边界内 */
    coercex(n, x, low, up);

    if (maxnfeval < 1) {
        rc = TNC_MAXFUN;
        goto cleanup;
    }
    /* 分配 g 如果需要的话 */
    if (g == NULL) {
        // 如果 g 为空，分配 n 个元素大小的内存空间
        g = malloc(n * sizeof(*g));
        // 如果内存分配失败，设置错误码并跳转到清理步骤
        if (g == NULL) {
            rc = TNC_ENOMEM;
            goto cleanup;
        }
        // 标记 g 已经被分配，需要在清理时释放
        free_g = TNC_TRUE;
    }

    /* 初始函数评估 */
    // 调用函数 function 对 x 进行初始评估
    frc = function(x, f, g, state);
    // 增加函数评估计数器
    (*nfeval)++;
    // 如果函数评估返回非零，表示用户中断，设置错误码并跳转到清理步骤
    if (frc) {
        rc = TNC_USERABORT;
        goto cleanup;
    }

    /* 是否为常数问题？ */
    // 计算约束数量 nc，初始化 i 为 0，遍历所有维度 n
    for (nc = 0, i = 0; i < n; i++) {
        // 如果上下界相等，或者存在缩放且缩放值为零，则增加约束数量
        if ((low[i] == up[i]) || (scale != NULL && scale[i] == 0.0)) {
            nc++;
        }
    }

    // 如果所有维度都是常数问题，设置错误码并跳转到清理步骤
    if (nc == n) {
        rc = TNC_CONSTANT;
        goto cleanup;
    }

    /* 缩放参数 */
    // 分配 xscale 数组的内存空间
    xscale = malloc(sizeof(*xscale) * n);
    // 如果内存分配失败，设置错误码并跳转到清理步骤
    if (xscale == NULL) {
        rc = TNC_ENOMEM;
        goto cleanup;
    }
    // 分配 xoffset 数组的内存空间
    xoffset = malloc(sizeof(*xoffset) * n);
    // 如果内存分配失败，设置错误码并跳转到清理步骤
    if (xoffset == NULL) {
        rc = TNC_ENOMEM;
        goto cleanup;
    }
    // 设置 fscale 初始值为 1.0
    fscale = 1.0;

    // 遍历所有维度 n
    for (i = 0; i < n; i++) {
        // 如果存在缩放参数
        if (scale != NULL) {
            // 计算缩放参数 xscale[i]，如果为零，设置 xoffset[i] 为 low[i] 和 up[i] 的中间值
            xscale[i] = fabs(scale[i]);
            if (xscale[i] == 0.0) {
                xoffset[i] = low[i] = up[i] = x[i];
            }
        } else if (low[i] != -HUGE_VAL && up[i] != HUGE_VAL) {
            // 如果没有缩放参数但存在有限的上下界
            xscale[i] = up[i] - low[i];
            xoffset[i] = (up[i] + low[i]) * 0.5;
        } else {
            // 否则，设置默认的缩放和偏移参数
            xscale[i] = 1.0 + fabs(x[i]);
            xoffset[i] = x[i];
        }
        // 如果存在额外的偏移参数，使用指定的 offset[i]
        if (offset != NULL) {
            xoffset[i] = offset[i];
        }
    }

    /* 参数的默认值 */
    // 设置 rteps 为 DBL_EPSILON 的平方根
    rteps = sqrt(DBL_EPSILON);

    // 如果 stepmx 小于 rteps 的 10 倍，设置为 1.0e1
    if (stepmx < rteps * 10.0) {
        stepmx = 1.0e1;
    }
    // 如果 eta 小于 0 或者大于等于 1.0，设置为 0.25
    if (eta < 0.0 || eta >= 1.0) {
        eta = 0.25;
    }
    // 如果 rescale 小于 0，设置为 1.3
    if (rescale < 0) {
        rescale = 1.3;
    }
    // 如果 maxCGit 小于 0，设置为 n 的一半，但不小于 1，且不大于 50
    if (maxCGit < 0) {          /* maxCGit == 0 is valid */
        maxCGit = n / 2;
        if (maxCGit < 1) {
            maxCGit = 1;
        }
        else if (maxCGit > 50) {
            maxCGit = 50;
        }
    }
    // 如果 maxCGit 大于 n，设置为 n
    if (maxCGit > n) {
        maxCGit = n;
    }
    // 如果 accuracy 小于等于 DBL_EPSILON，设置为 rteps
    if (accuracy <= DBL_EPSILON) {
        accuracy = rteps;
    }
    // 如果 ftol 小于 0，设置为 accuracy
    if (ftol < 0.0) {
        ftol = accuracy;
    }
    // 如果 pgtol 小于 0，设置为 accuracy 的平方根乘以 1e-2
    if (pgtol < 0.0) {
        pgtol = 1e-2 * sqrt(accuracy);
    }
    // 如果 xtol 小于 0，设置为 rteps
    if (xtol < 0.0) {
        xtol = rteps;
    }

    /* 优化 */
    // 调用优化函数 tnc_minimize 进行优化，传入所有参数，并将结果保存在 rc 中
    rc = tnc_minimize(n, x, f, g, function, state,
                      xscale, xoffset, &fscale, low, up, messages,
                      maxCGit, maxnfeval, nfeval, niter, eta, stepmx,
                      accuracy, fmin, ftol, xtol, pgtol, rescale,
                      callback);

  cleanup:
    // 如果设置了 TNC_MSG_EXIT 标志，输出错误消息到 stderr
    if (messages & TNC_MSG_EXIT) {
        fprintf(stderr, "tnc: %s\n", tnc_rc_string[rc - TNC_MINRC]);
    }

    // 释放动态分配的内存
    free(xscale);
    // 如果 free_low 为真，释放 low 数组
    if (free_low) {
        free(low);
    }
    // 如果 free_up 为真，释放 up 数组
    if (free_up) {
        free(up);
    }
    // 如果 free_g 为真，释放 g 数组
    if (free_g) {
        free(g);
    }
    // 释放 xoffset 数组
    free(xoffset);

    // 返回结果码 rc
    return rc;
/* Coerce x into bounds */
static void coercex(int n, double x[], const double low[], const double up[])
{
    int i;

    for (i = 0; i < n; i++) {
        // 如果 x[i] 小于下界 low[i]，则将其强制设为 low[i]
        if (x[i] < low[i]) {
            x[i] = low[i];
        }
        // 如果 x[i] 大于上界 up[i]，则将其强制设为 up[i]
        else if (x[i] > up[i]) {
            x[i] = up[i];
        }
    }
}

/* Unscale x */
static void unscalex(int n, double x[], const double xscale[],
                     const double xoffset[])
{
    int i;
    for (i = 0; i < n; i++) {
        // 反向缩放 x[i]，即先减去偏移量 xoffset[i]，再乘以缩放因子 xscale[i]
        x[i] = x[i] * xscale[i] + xoffset[i];
    }
}

/* Scale x */
static void scalex(int n, double x[], const double xscale[],
                   const double xoffset[])
{
    int i;
    for (i = 0; i < n; i++) {
        // 缩放 x[i]，如果缩放因子 xscale[i] 大于 0，则先减去偏移量 xoffset[i]，再除以 xscale[i]
        if (xscale[i] > 0.0) {
            x[i] = (x[i] - xoffset[i]) / xscale[i];
        }
    }
}

/* Scale g */
static void scaleg(int n, double g[], const double xscale[], double fscale)
{
    int i;
    for (i = 0; i < n; i++) {
        // 缩放梯度向量 g[i]，乘以对应的缩放因子 xscale[i] 和 fscale
        g[i] *= xscale[i] * fscale;
    }
}

/* Calculate the pivot vector */
static void setConstraints(int n, double x[], int pivot[], double xscale[],
                           double xoffset[], double low[], double up[])
{
    int i;

    for (i = 0; i < n; i++) {
        // 计算约束条件的枢轴向量 pivot
        /* tolerances should be better ajusted */
        // 如果缩放因子 xscale[i] 等于 0，则将枢轴设为 2
        if (xscale[i] == 0.0) {
            pivot[i] = 2;
        }
        // 如果存在下界 low[i] 且满足条件，则将枢轴设为 -1
        else if (low[i] != -HUGE_VAL &&
                 (x[i] * xscale[i] + xoffset[i] - low[i] <=
                  DBL_EPSILON * 10.0 * (fabs(low[i]) + 1.0))) {
             pivot[i] = -1;
        }
        // 如果存在上界 up[i] 且满足条件，则将枢轴设为 1
        else if (up[i] != HUGE_VAL &&
                 (x[i] * xscale[i] + xoffset[i] - up[i] >=
                  DBL_EPSILON * 10.0 * (fabs(up[i]) + 1.0))) {
            pivot[i] = 1;
        }
        // 否则，将枢轴设为 0
        else {
            pivot[i] = 0;
        }
    }
}

/*
 * This routine is a bounds-constrained truncated-newton method.
 * the truncated-newton method is preconditioned by a limited-memory
 * quasi-newton method (this preconditioning strategy is developed
 * in this routine) with a further diagonal scaling
 * (see routine diagonalscaling).
 */
static tnc_rc tnc_minimize(int n, double x[],
                           double *f, double gfull[],
                           tnc_function * function, void *state,
                           double xscale[], double xoffset[],
                           double *fscale, double low[], double up[],
                           tnc_message messages, int maxCGit,
                           int maxnfeval, int *nfeval, int *niter,
                           double eta, double stepmx, double accuracy,
                           double fmin, double ftol, double xtol,
                           double pgtol, double rescale,
                           tnc_callback * callback)
{
    double fLastReset, difnew, epsred, oldgtp, difold, oldf, xnorm, newscale,
        gnorm, ustpmax, fLastConstraint, spe, yrsr, yksk,
        *temp = NULL, *sk = NULL, *yk = NULL, *diagb = NULL, *sr = NULL,
        *yr = NULL, *oldg = NULL, *pk = NULL, *g = NULL;
    double alpha = 0.0;         /* 默认未使用的值 */
    int i, icycle, oldnfeval, *pivot = NULL, frc;
    logical lreset, newcon, upd1, remcon;
    tnc_rc rc = TNC_ENOMEM;     /* 默认错误码为内存不足 */

    *niter = 0;

    /* 分配临时向量 */
    oldg = malloc(sizeof(*oldg) * n);  /* 分配旧梯度向量的内存 */
    if (oldg == NULL) {
        goto cleanup;
    }
    g = malloc(sizeof(*g) * n);  /* 分配梯度向量的内存 */
    if (g == NULL) {
        goto cleanup;
    }
    temp = malloc(sizeof(*temp) * n);  /* 分配临时向量的内存 */
    if (temp == NULL) {
        goto cleanup;
    }
    diagb = malloc(sizeof(*diagb) * n);  /* 分配对角线向量的内存 */
    if (diagb == NULL) {
        goto cleanup;
    }
    pk = malloc(sizeof(*pk) * n);  /* 分配 pk 向量的内存 */
    if (pk == NULL) {
        goto cleanup;
    }

    sk = malloc(sizeof(*sk) * n);  /* 分配 sk 向量的内存 */
    if (sk == NULL) {
        goto cleanup;
    }
    yk = malloc(sizeof(*yk) * n);  /* 分配 yk 向量的内存 */
    if (yk == NULL) {
        goto cleanup;
    }
    sr = malloc(sizeof(*sr) * n);  /* 分配 sr 向量的内存 */
    if (sr == NULL) {
        goto cleanup;
    }
    yr = malloc(sizeof(*yr) * n);  /* 分配 yr 向量的内存 */
    if (yr == NULL) {
        goto cleanup;
    }

    pivot = malloc(sizeof(*pivot) * n);  /* 分配 pivot 向量的内存 */
    if (pivot == NULL) {
        goto cleanup;
    }

    /* 初始化变量 */
    difnew = 0.0;   /* 设置 difnew 初始值为 0.0 */
    epsred = 0.05;  /* 设置 epsred 初始值为 0.05 */
    upd1 = TNC_TRUE;    /* 更新标志位设置为真 */
    icycle = n - 1;     /* 设置循环次数为 n - 1 */
    newcon = TNC_TRUE;  /* 新约束标志位设置为真 */

    /* 不需要的初始化 */
    lreset = TNC_FALSE; /* 重置标志位设置为假 */
    yrsr = 0.0;          /* yrsr 初始值为 0.0 */
    yksk = 0.0;          /* yksk 初始值为 0.0 */

    /* 初始缩放 */
    scalex(n, x, xscale, xoffset);  /* 对变量 x 进行初始缩放 */
    (*f) *= *fscale;                /* 更新目标函数值 */

    /* 初始主轴计算 */
    setConstraints(n, x, pivot, xscale, xoffset, low, up);  /* 设置约束条件 */

    dcopy1(n, gfull, g);    /* 复制 gfull 到 g */
    scaleg(n, g, xscale, *fscale);  /* 缩放梯度 g */

    /* 测试拉格朗日乘子以查看它们是否非负 */
    for (i = 0; i < n; i++) {
        if (-pivot[i] * g[i] < 0.0) {
            pivot[i] = 0;   /* 如果乘积小于 0，则将 pivot[i] 设为 0 */
        }
    }

    project(n, g, pivot);   /* 根据约束投影梯度 g */

    /* 设置其他参数的初始值 */
    gnorm = dnrm21(n, g);   /* 计算 g 的 2-范数 */

    fLastConstraint = *f;   /* 最后一个约束处的目标函数值 */
    fLastReset = *f;        /* 上次重置时的目标函数值 */

    if (messages & TNC_MSG_ITER) {
        fprintf(stderr, "  NIT   NF   F                       GTG\n");  /* 打印迭代消息表头 */
    }
    if (messages & TNC_MSG_ITER) {
        printCurrentIteration(n, *f / *fscale, gfull,
                              *niter, *nfeval, pivot);   /* 打印当前迭代信息 */
    }

    /* 将近似 Hessian 矩阵的对角线元素设为单位值 */
    for (i = 0; i < n; i++) {
        diagb[i] = 1.0;   /* 设置 diagb[i] 初始值为 1.0 */
    }

    /* 主迭代循环的开始 */

    if (messages & TNC_MSG_ITER) {
        printCurrentIteration(n, *f / *fscale, gfull,
                              *niter, *nfeval, pivot);   /* 打印当前迭代信息 */
    }

    /* 反缩放 */
    unscalex(n, x, xscale, xoffset);  /* 反缩放变量 x */
    coercex(n, x, low, up);           /* 强制在界限内 */
    (*f) /= *fscale;                  /* 更新目标函数值 */

  cleanup:
    free(oldg);   /* 释放内存：oldg */
    free(g);      /* 释放内存：g */
    free(temp);   /* 释放内存：temp */
    free(diagb);  /* 释放内存：diagb */
    free(pk);     /* 释放内存：pk */

    free(sk);     /* 释放内存：sk */
    free(yk);     /* 释放内存：yk */
    free(sr);     /* 释放内存：sr */
    free(yr);     /* 释放内存：yr */

    free(pivot);  /* 释放内存：pivot */

    return rc;    /* 返回函数执行结果 */
/*
 * Print the results of the current iteration to stderr
 */
static void printCurrentIteration(int n, double f, double g[], int niter,
                                  int nfeval, int pivot[])
{
    int i;
    double gtg;

    gtg = 0.0;
    // 计算未被约束方向上的梯度平方和
    for (i = 0; i < n; i++) {
        if (pivot[i] == 0) {
            gtg += g[i] * g[i];
        }
    }

    // 输出当前迭代的结果到 stderr
    fprintf(stderr, " %4d %4d %22.15E  %15.8E\n", niter, nfeval, f, gtg);
}

/*
 * Project x[i] to 0.0 if direction i is currently constrained
 */
static void project(int n, double x[], const int pivot[])
{
    int i;
    // 如果方向 i 当前受到约束，则将 x[i] 投影为 0.0
    for (i = 0; i < n; i++) {
        if (pivot[i] != 0) {
            x[i] = 0.0;
        }
    }
}

/*
 * Project x[i] to 0.0 if direction i is constant (xscale[i] == 0.0)
 */
static void projectConstants(int n, double x[], const double xscale[])
{
    int i;
    // 如果方向 i 是常量方向（xscale[i] == 0.0），则将 x[i] 投影为 0.0
    for (i = 0; i < n; i++) {
        if (xscale[i] == 0.0) {
            x[i] = 0.0;
        }
    }
}

/*
 * Compute the maximum allowable step length considering constraints
 */
static double stepMax(double step, int n, double x[], double dir[],
                      int pivot[], double low[], double up[],
                      double xscale[], double xoffset[])
{
    int i;
    double t;

    /* Constrained maximum step */
    // 计算受约束条件下的最大可行步长
    for (i = 0; i < n; i++) {
        if ((pivot[i] == 0) && (dir[i] != 0.0)) {
            if (dir[i] < 0.0) {
                t = (low[i] - xoffset[i]) / xscale[i] - x[i];
                if (t > step * dir[i]) {
                    step = t / dir[i];
                }
            }
            else {
                t = (up[i] - xoffset[i]) / xscale[i] - x[i];
                if (t < step * dir[i]) {
                    step = t / dir[i];
                }
            }
        }
    }

    return step;
}

/*
 * Update the constraint vector pivot if a new constraint is encountered
 */
static logical addConstraint(int n, double x[], double p[], int pivot[],
                             double low[], double up[], double xscale[],
                             double xoffset[])
{
    int i, newcon = TNC_FALSE;
    double tol;

    // 更新约束向量 pivot 如果遇到新的约束条件
    for (i = 0; i < n; i++) {
        if ((pivot[i] == 0) && (p[i] != 0.0)) {
            if (p[i] < 0.0 && low[i] != -HUGE_VAL) {
                tol = DBL_EPSILON * 10.0 * (fabs(low[i]) + 1.0);
                if (x[i] * xscale[i] + xoffset[i] - low[i] <= tol) {
                    pivot[i] = -1;
                    x[i] = (low[i] - xoffset[i]) / xscale[i];
                    newcon = TNC_TRUE;
                }
            }
            else if (up[i] != HUGE_VAL) {
                tol = DBL_EPSILON * 10.0 * (fabs(up[i]) + 1.0);
                if (up[i] - (x[i] * xscale[i] + xoffset[i]) <= tol) {
                    pivot[i] = 1;
                    x[i] = (up[i] - xoffset[i]) / xscale[i];
                    newcon = TNC_TRUE;
                }
            }
        }
    }
    return newcon;
}
/*
 * This function checks if a constraint should be removed based on the given criteria.
 * If the difference between fLastConstraint and f is less than half of gtpnew, and gnorm is greater than pgtolfs,
 * the function returns TNC_FALSE indicating the constraint should not be removed.
 */
static logical removeConstraint(double gtpnew, double gnorm,
                                double pgtolfs, double f,
                                double fLastConstraint, double g[],
                                int pivot[], int n)
{
    double cmax, t;
    int imax, i;

    if (((fLastConstraint - f) <= (gtpnew * -0.5)) && (gnorm > pgtolfs)) {
        return TNC_FALSE;
    }

    imax = -1;
    cmax = 0.0;

    for (i = 0; i < n; i++) {
        if (pivot[i] == 2) {
            continue;  // Skip this iteration if pivot[i] is 2
        }
        t = -pivot[i] * g[i];
        if (t < cmax) {
            cmax = t;
            imax = i;  // Record the index i where t is less than cmax
        }
    }

    if (imax != -1) {
        pivot[imax] = 0;  // Set pivot[imax] to 0
        return TNC_TRUE;  // Return true indicating a constraint is removed
    }
    else {
        return TNC_FALSE;  // Return false indicating no constraint is removed
    }

/*
 * For details, see gill, murray, and wright (1981, p. 308) and
 * fletcher (1981, p. 116). The multiplier tests (here, testing
 * the sign of the components of the gradient) may still need to
 * modified to incorporate tolerances for zero.
 */
}

/*
 * This routine performs a preconditioned conjugate-gradient
 * iteration in order to solve the newton equations for a search
 * direction for a truncated-newton algorithm.
 * When the value of the quadratic model is sufficiently reduced,
 * the iteration is terminated.
 */
static int tnc_direction(double *zsol, double *diagb,
                         double *x, double g[], int n,
                         int maxCGit, int maxnfeval, int *nfeval,
                         logical upd1, double yksk, double yrsr,
                         double *sk, double *yk, double *sr, double *yr,
                         logical lreset, tnc_function * function,
                         void *state, double xscale[], double xoffset[],
                         double fscale, int *pivot, double accuracy,
                         double gnorm, double xnorm, double low[],
                         double up[])
{
    double alpha, beta, qold, qnew, rhsnrm, tol, vgv, rz, rzold, qtest, pr,
        gtp;
    int i, k, frc;
    /* Temporary vectors */
    double *r = NULL, *zk = NULL, *v = NULL, *emat = NULL, *gv = NULL;

    /* No CG it. => dir = -grad */
    if (maxCGit == 0) {
        dcopy1(n, g, zsol);  // Copy vector g to zsol
        dneg1(n, zsol);      // Negate zsol
        project(n, zsol, pivot);  // Project zsol based on pivot
        return 0;  // Return success
    }

    /* General initialization */
    rhsnrm = gnorm;  // Initialize rhsnrm with gnorm
    tol = 1e-12;     // Set tolerance
    qold = 0.0;      // Initialize qold to 0
    rzold = 0.0;     /* Unneeded */

    frc = -1;         /* ENOMEM here */
    r = malloc(sizeof(*r) * n);  // Allocate memory for vector r (residual)
    if (r == NULL) {
        goto cleanup;  // Cleanup and return if allocation fails
    }
    v = malloc(sizeof(*v) * n);  // Allocate memory for vector v
    if (v == NULL) {
        goto cleanup;
    }
    zk = malloc(sizeof(*zk) * n);  // Allocate memory for vector zk
    if (zk == NULL) {
        goto cleanup;
    }
    emat = malloc(sizeof(*emat) * n);   // Allocate memory for diagonal preconditioning matrix
    if (emat == NULL) {
        goto cleanup;
    }
    gv = malloc(sizeof(*gv) * n);       // Allocate memory for hessian times v
    if (gv == NULL) {
        goto cleanup;
    }
    /* Initialization for preconditioned conjugate-gradient algorithm */
    /* 使用预条件的共轭梯度算法进行初始化 */

    frc = initPreconditioner(diagb, emat, n, lreset, yksk, yrsr, sk, yk, sr,
                             yr, upd1);
    /* 调用初始化预条件器函数，对算法进行初始化 */

    if (frc) {
        goto cleanup;
    }
    /* 如果初始化失败，跳转到清理部分 */

    for (i = 0; i < n; i++) {
        r[i] = -g[i];
        v[i] = 0.0;
        zsol[i] = 0.0;          /* Computed search direction */
    }
    /* 初始化向量 r, v, zsol */

    /* Main iteration */
    for (k = 0; k < maxCGit; k++) {
        /* CG iteration to solve system of equations */
        /* CG 迭代解方程系统 */

        project(n, r, pivot);
        /* 投影操作，将向量 r 投影到特定空间 */

        frc = msolve(r, zk, n, sk, yk, diagb, sr, yr, upd1, yksk, yrsr,
                     lreset);
        /* 调用 msolve 函数求解线性方程组 */

        if (frc) {
            goto cleanup;
        }
        /* 如果求解失败，跳转到清理部分 */

        project(n, zk, pivot);
        /* 投影操作，将向量 zk 投影到特定空间 */

        rz = ddot1(n, r, zk);
        /* 计算向量 r 和 zk 的点积 */

        if ((rz / rhsnrm < tol) || ((*nfeval) >= (maxnfeval - 1))) {
            /* Truncate algorithm in case of an emergency
               or too many function evaluations */
            /* 在紧急情况下或者函数评估过多时中止算法 */

            if (k == 0) {
                dcopy1(n, g, zsol);
                dneg1(n, zsol);
                project(n, zsol, pivot);
            }
            /* 如果是第一次迭代，初始化 zsol */

            break;
        }

        if (k == 0) {
            beta = 0.0;
        }
        else {
            beta = rz / rzold;
        }
        /* 计算 beta 参数 */

        for (i = 0; i < n; i++) {
            v[i] = zk[i] + beta * v[i];
        }
        /* 更新向量 v */

        project(n, v, pivot);
        /* 投影操作，将向量 v 投影到特定空间 */

        frc = hessianTimesVector(v, gv, n, x, g, function, state,
                                 xscale, xoffset, fscale, accuracy, xnorm,
                                 low, up);
        /* 计算 Hessian 矩阵乘以向量 v */

        ++(*nfeval);
        /* 增加函数评估次数 */

        if (frc) {
            goto cleanup;
        }
        /* 如果计算失败，跳转到清理部分 */

        project(n, gv, pivot);
        /* 投影操作，将向量 gv 投影到特定空间 */

        vgv = ddot1(n, v, gv);
        /* 计算向量 v 和 gv 的点积 */

        if (vgv / rhsnrm < tol) {
            /* Truncate algorithm in case of an emergency */
            /* 在紧急情况下中止算法 */

            if (k == 0) {
                frc = msolve(g, zsol, n, sk, yk, diagb, sr, yr, upd1, yksk,
                             yrsr, lreset);
                /* 调用 msolve 函数重新求解线性方程组 */

                if (frc) {
                    goto cleanup;
                }
                /* 如果求解失败，跳转到清理部分 */

                dneg1(n, zsol);
                project(n, zsol, pivot);
            }
            break;
        }

        diagonalScaling(n, emat, v, gv, r);
        /* 对向量 v, gv, r 进行对角线缩放 */

        /* Compute linear step length */
        /* 计算线性步长 */
        alpha = rz / vgv;

        /* Compute current solution and related vectors */
        /* 计算当前解和相关向量 */
        daxpy1(n, alpha, v, zsol);
        daxpy1(n, -alpha, gv, r);

        /* Test for convergence */
        /* 检测收敛性 */
        gtp = ddot1(n, zsol, g);
        pr = ddot1(n, r, zsol);
        qnew = (gtp + pr) * 0.5;
        qtest = (k + 1) * (1.0 - qold / qnew);
        if (qtest <= 0.5) {
            break;
        }

        /* Perform cautionary test */
        /* 执行谨慎性测试 */
        if (gtp > 0.0) {
            /* Truncate algorithm in case of an emergency */
            /* 在紧急情况下中止算法 */

            daxpy1(n, -alpha, v, zsol);
            break;
        }

        qold = qnew;
        rzold = rz;
    }

    /* Terminate algorithm */
    /* 终止算法 */

    /* Store (or restore) diagonal preconditioning */
    /* 存储（或恢复）对角预条件 */

    dcopy1(n, emat, diagb);
    /* 复制向量 emat 到 diagb */

  cleanup:
    free(r);
    free(v);
    free(zk);
    free(emat);
    /* 清理内存，释放动态分配的数组 */
    # 释放 gv 指向的内存空间
    free(gv);
    # 返回 frc 变量的值作为函数的结果
    return frc;
/*
 * Update the preconditioning matrix based on a diagonal version
 * of the bfgs quasi-newton update.
 */
static void diagonalScaling(int n, double e[], double v[], double gv[],
                            double r[])
{
    int i;
    double vr, vgv;

    vr = 1.0 / ddot1(n, v, r);  // 计算 v 和 r 的点积的倒数
    vgv = 1.0 / ddot1(n, v, gv);  // 计算 v 和 gv 的点积的倒数
    for (i = 0; i < n; i++) {
        // 更新预条件矩阵中的对角元素 e[i]
        e[i] += -r[i] * r[i] * vr + gv[i] * gv[i] * vgv;
        if (e[i] <= 1e-6) {
            e[i] = 1.0;  // 如果 e[i] 小于等于 1e-6，则将其设为 1.0
        }
    }
}

/*
 * Returns the length of the initial step to be taken along the
 * vector p in the next linear search.
 */
static double initialStep(double fnew, double fmin, double gtp,
                          double smax)
{
    double d, alpha;

    d = fabs(fnew - fmin);  // 计算 fnew 和 fmin 之差的绝对值
    alpha = 1.0;
    if (d * 2.0 <= -(gtp) && d >= DBL_EPSILON) {
        alpha = d * -2.0 / gtp;  // 根据条件计算初始步长 alpha
    }
    if (alpha >= smax) {
        alpha = smax;  // 如果 alpha 大于等于 smax，则将其设为 smax
    }

    return alpha;  // 返回计算得到的初始步长 alpha
}

/*
 * Hessian vector product through finite differences
 */
static int hessianTimesVector(double v[], double gv[], int n,
                              double x[], double g[],
                              tnc_function * function, void *state,
                              double xscale[], double xoffset[],
                              double fscale, double accuracy, double xnorm,
                              double low[], double up[])
{
    double dinv, f, delta, *xv;
    int i, frc;

    xv = malloc(sizeof(*xv) * n);  // 分配存储空间以存放新的向量 xv
    if (xv == NULL) {
        return -1;  // 内存分配失败，返回错误代码
    }

    delta = accuracy * (xnorm + 1.0);  // 计算 delta 的值
    for (i = 0; i < n; i++) {
        xv[i] = x[i] + delta * v[i];  // 根据公式计算新向量 xv 的每个分量
    }

    unscalex(n, xv, xscale, xoffset);  // 反缩放 xv 向量
    coercex(n, xv, low, up);  // 将 xv 向量强制限制在给定的范围内
    frc = function(xv, &f, gv, state);  // 调用指定函数计算 xv 处的函数值和梯度
    free(xv);  // 释放 xv 向量的内存空间
    if (frc) {
        return 1;  // 如果计算函数值和梯度出错，返回错误代码
    }
    scaleg(n, gv, xscale, fscale);  // 缩放计算得到的梯度 gv

    dinv = 1.0 / delta;  // 计算 delta 的倒数
    for (i = 0; i < n; i++) {
        gv[i] = (gv[i] - g[i]) * dinv;  // 计算 Hessian 矩阵对向量 v 的乘积
    }

    projectConstants(n, gv, xscale);  // 将结果投影到常数约束条件上

    return 0;  // 返回成功完成计算的标志
}

/*
 * This routine acts as a preconditioning step for the
 * linear conjugate-gradient routine. It is also the
 * method of computing the search direction from the
 * gradient for the non-linear conjugate-gradient code.
 * It represents a two-step self-scaled bfgs formula.
 */
static int msolve(double g[], double y[], int n,
                  double sk[], double yk[], double diagb[], double sr[],
                  double yr[], logical upd1, double yksk, double yrsr,
                  logical lreset)
{
    double ghyk, ghyr, yksr, ykhyk, ykhyr, yrhyr, rdiagb, gsr, gsk;
    int i, frc;
    double *hg = NULL, *hyk = NULL, *hyr = NULL;

    if (upd1) {
        for (i = 0; i < n; i++) {
            y[i] = g[i] / diagb[i];  // 更新 y 向量，作为预条件步骤的一部分
        }
        return 0;  // 返回成功完成计算的标志
    }

    frc = -1;  // 初始化错误代码
    gsk = ddot1(n, g, sk);  // 计算 g 和 sk 的点积
    hg = malloc(sizeof(*hg) * n);  // 分配存储空间以存放新向量 hg
    if (hg == NULL) {
        goto cleanup;  // 如果内存分配失败，则跳转到清理代码的标签
    }
    hyr = malloc(sizeof(*hyr) * n);  // 分配存储空间以存放新向量 hyr
    if (hyr == NULL) {
        goto cleanup;  // 如果内存分配失败，则跳转到清理代码的标签
    }
    hyk = malloc(sizeof(*hyk) * n);  // 分配存储空间以存放新向量 hyk

cleanup:
    if (hg != NULL) {
        free(hg);  // 释放 hg 向量的内存空间
    }
    if (hyr != NULL) {
        free(hyr);  // 释放 hyr 向量的内存空间
    }
    if (hyk != NULL) {
        free(hyk);  // 释放 hyk 向量的内存空间
    }

    return frc;  // 返回计算结果或错误代码
}
    # 如果指针 `hyk` 为空，跳转到清理步骤，确保内存和资源的正确释放
    if (hyk == NULL) {
        goto cleanup;
    }
    # 初始化返回代码为0
    frc = 0;

    /* 计算 gh 和 hy，其中 h 是对角线的倒数 */
    # 如果需要重置（lreset为真），进行以下操作
    if (lreset) {
        # 遍历索引 i 从 0 到 n-1
        for (i = 0; i < n; i++) {
            # 计算对角线的倒数
            rdiagb = 1.0 / diagb[i];
            # 计算 hg[i] 和 hyk[i]
            hg[i] = g[i] * rdiagb;
            hyk[i] = yk[i] * rdiagb;
        }
        # 计算 yk 和 hyk 的内积
        ykhyk = ddot1(n, yk, hyk);
        # 计算 g 和 hyk 的内积
        ghyk = ddot1(n, g, hyk);
        # 调用 ssbfgs 函数进行更新操作
        ssbfgs(n, 1.0, sk, hg, hyk, yksk, ykhyk, gsk, ghyk, y);
    }
    # 如果不需要重置
    else {
        # 遍历索引 i 从 0 到 n-1
        for (i = 0; i < n; i++) {
            # 计算对角线的倒数
            rdiagb = 1.0 / diagb[i];
            # 计算 hg[i]、hyk[i] 和 hyr[i]
            hg[i] = g[i] * rdiagb;
            hyk[i] = yk[i] * rdiagb;
            hyr[i] = yr[i] * rdiagb;
        }
        # 计算 g 和 sr 的内积
        gsr = ddot1(n, g, sr);
        # 计算 g 和 hyr 的内积
        ghyr = ddot1(n, g, hyr);
        # 计算 yr 和 hyr 的内积
        yrhyr = ddot1(n, yr, hyr);
        # 调用 ssbfgs 函数进行更新操作
        ssbfgs(n, 1.0, sr, hg, hyr, yrsr, yrhyr, gsr, ghyr, hg);
        # 计算 yk 和 sr 的内积
        yksr = ddot1(n, yk, sr);
        # 计算 yk 和 hyr 的内积
        ykhyr = ddot1(n, yk, hyr);
        # 调用 ssbfgs 函数进行更新操作
        ssbfgs(n, 1.0, sr, hyk, hyr, yrsr, yrhyr, yksr, ykhyr, hyk);
        # 计算 hyk 和 yk 的内积
        ykhyk = ddot1(n, hyk, yk);
        # 计算 hyk 和 g 的内积
        ghyk = ddot1(n, hyk, g);
        # 调用 ssbfgs 函数进行更新操作
        ssbfgs(n, 1.0, sk, hg, hyk, yksk, ykhyk, gsk, ghyk, y);
    }

  # 清理步骤的标签
  cleanup:
    # 释放分配的内存：hg、hyk 和 hyr
    free(hg);
    free(hyk);
    free(hyr);

    # 返回 frc 变量作为函数的结果
    return frc;
/*
 * Self-scaled BFGS
 */
static void ssbfgs(int n, double gamma, double sj[], double hjv[],
                   double hjyj[], double yjsj,
                   double yjhyj, double vsj, double vhyj, double hjp1v[])
{
    double beta, delta;
    int i;

    // 检查 yjsj 是否为零，如果是则设置 delta 和 beta 为零
    if (yjsj == 0.0) {
        delta = 0.0;
        beta = 0.0;
    }
    // 否则根据公式计算 delta 和 beta
    else {
        delta = (gamma * yjhyj / yjsj + 1.0) * vsj / yjsj
                - gamma * vhyj / yjsj;
        beta = -gamma * vsj / yjsj;
    }

    // 计算 hjp1v 数组的值
    for (i = 0; i < n; i++) {
        hjp1v[i] = gamma * hjv[i] + delta * sj[i] + beta * hjyj[i];
    }
}

/*
 * Initialize the preconditioner
 */
static int initPreconditioner(double diagb[], double emat[], int n,
                              logical lreset, double yksk, double yrsr,
                              double sk[], double yk[], double sr[],
                              double yr[], logical upd1)
{
    double srds, yrsk, td, sds;
    int i;
    double *bsk;

    // 如果 upd1 为真，则直接拷贝 diagb 数组到 emat 数组
    if (upd1) {
        dcopy1(n, diagb, emat);
        return 0;
    }

    // 否则进行下面的初始化步骤

    // 分配 bsk 数组的内存空间
    bsk = malloc(sizeof(*bsk) * n);
    if (bsk == NULL) {
        return -1;  // 内存分配失败返回 -1
    }

    // 如果 lreset 为真，则按照 sk 和 yk 的值初始化 emat 数组
    if (lreset) {
        for (i = 0; i < n; i++) {
            bsk[i] = diagb[i] * sk[i];
        }
        sds = ddot1(n, sk, bsk);
        if (yksk == 0.0) {
            yksk = 1.0;
        }
        if (sds == 0.0) {
            sds = 1.0;
        }
        for (i = 0; i < n; i++) {
            td = diagb[i];
            emat[i] = td - td * td * sk[i] * sk[i] / sds
                      + yk[i] * yk[i] / yksk;
        }
    }
    // 否则按照 sr 和 yr 的值初始化 emat 数组
    else {
        for (i = 0; i < n; i++) {
            bsk[i] = diagb[i] * sr[i];
        }
        sds = ddot1(n, sr, bsk);
        srds = ddot1(n, sk, bsk);
        yrsk = ddot1(n, yr, sk);
        if (yrsr == 0.0) {
            yrsr = 1.0;
        }
        if (sds == 0.0) {
            sds = 1.0;
        }
        for (i = 0; i < n; i++) {
            td = diagb[i];
            bsk[i] = td * sk[i] - bsk[i] * srds / sds + yr[i] * yrsk / yrsr;
            emat[i] = td - td * td * sr[i] * sr[i] / sds + yr[i] * yr[i] / yrsr;
        }
        sds = ddot1(n, sk, bsk);
        if (yksk == 0.0) {
            yksk = 1.0;
        }
        if (sds == 0.0) {
            sds = 1.0;
        }
        for (i = 0; i < n; i++) {
            emat[i] -= bsk[i] * bsk[i] / sds + yk[i] * yk[i] / yksk;
        }
    }

    free(bsk);  // 释放 bsk 数组的内存空间
    return 0;
}

/*
 * Line search algorithm of gill and murray
 */
static ls_rc linearSearch(int n, tnc_function * function, void *state,
                          double low[], double up[],
                          double xscale[], double xoffset[], double fscale,
                          int pivot[], double eta, double ftol,
                          double xbnd, double p[], double x[], double *f,
                          double *alpha, double gfull[], int maxnfeval,
                          int *nfeval)
{
    // 该函数的详细注释可能超出当前范围，不予展示
}
    double b1, big, tol, rmu, fpresn, fu, gu, fw, gw, gtest1, gtest2,
        oldf, fmin, gmin, rtsmll, step, a, b, e, u, ualpha, factor, scxbnd,
        xw, reltol, abstol, tnytol, pe, xnorm, rteps;
    double *temp = NULL, *tempgfull = NULL, *newgfull = NULL;
    int maxlsit = 64, i, itcnt, frc;
    ls_rc rc;
    getptc_rc itest;
    logical braktd;

    // 设置内存分配失败返回码为LS_ENOMEM
    rc = LS_ENOMEM;

    // 分配临时数组temp，用于存储大小为n的double类型数据
    temp = malloc(sizeof(*temp) * n);
    if (temp == NULL) {
        goto cleanup;  // 如果分配失败则跳转到清理代码块
    }

    // 分配临时数组tempgfull，用于存储大小为n的double类型数据
    tempgfull = malloc(sizeof(*tempgfull) * n);
    if (tempgfull == NULL) {
        goto cleanup;  // 如果分配失败则跳转到清理代码块
    }

    // 分配临时数组newgfull，用于存储大小为n的double类型数据
    newgfull = malloc(sizeof(*newgfull) * n);
    if (newgfull == NULL) {
        goto cleanup;  // 如果分配失败则跳转到清理代码块
    }

    // 复制数组gfull的内容到temp数组中
    dcopy1(n, gfull, temp);

    // 对temp数组进行缩放，缩放因子由xscale和fscale决定
    scaleg(n, temp, xscale, fscale);

    // 计算temp和p数组的点积，结果存储在gu中
    gu = ddot1(n, temp, p);

    // 复制数组x的内容到temp数组中
    dcopy1(n, x, temp);

    // 使用pivot进行投影操作，修改temp数组的内容
    project(n, temp, pivot);

    // 计算temp数组的2-范数，结果存储在xnorm中
    xnorm = dnrm21(n, temp);

    /* 计算线性搜索的绝对和相对容差 */
    rteps = sqrt(DBL_EPSILON);  // 计算平方根的机器精度
    pe = dnrm21(n, p) + DBL_EPSILON;  // 计算p数组的2-范数并加上机器精度
    reltol = rteps * (xnorm + 1.0) / pe;  // 计算相对容差
    abstol = -DBL_EPSILON * (1.0 + fabs(*f)) / (gu - DBL_EPSILON);  // 计算绝对容差

    /* 计算线性搜索中允许的最小间距 */
    tnytol = DBL_EPSILON * (xnorm + 1.0) / pe;

    rtsmll = DBL_EPSILON;  // 设置极小值
    big = 1.0 / (DBL_EPSILON * DBL_EPSILON);  // 设置一个大数值
    itcnt = 0;  // 迭代次数初始化为0

    /* 设置f(x)的估计相对精度 */
    fpresn = ftol;

    u = *alpha;  // 将alpha的值赋给u
    fu = *f;  // 将f的值赋给fu
    fmin = *f;  // 将f的值赋给fmin
    rmu = 1e-4;  // 设置rmu的值为0.0001

    /* 初始化getptc函数的参数 */
    itest = getptcInit(&reltol, &abstol, tnytol, eta, rmu,
                       xbnd, &u, &fu, &gu, alpha, &fmin, &gmin, &xw, &fw,
                       &gw, &a, &b, &oldf, &b1, &scxbnd, &e, &step,
                       &factor, &braktd, &gtest1, &gtest2, &tol);

    /* 如果itest == GETPTC_EVAL，则算法需要计算函数值 */
    while (itest == GETPTC_EVAL) {
        /* 检查是否达到最大迭代次数或最大函数评估次数 */
        if ((++itcnt > maxlsit) || ((*nfeval) >= maxnfeval)) {
            break;
        }

        ualpha = *alpha + u;
        for (i = 0; i < n; i++) {
            temp[i] = x[i] + ualpha * p[i];
        }

        /* 执行函数评估 */
        unscalex(n, temp, xscale, xoffset);
        coercex(n, temp, low, up);

        frc = function(temp, &fu, tempgfull, state);
        ++(*nfeval);
        if (frc) {
            rc = LS_USERABORT;
            goto cleanup;
        }

        fu *= fscale;

        dcopy1(n, tempgfull, temp);
        scaleg(n, temp, xscale, fscale);
        gu = ddot1(n, temp, p);

        itest = getptcIter(big, rtsmll, &reltol, &abstol, tnytol, fpresn,
                           xbnd, &u, &fu, &gu, alpha, &fmin, &gmin, &xw,
                           &fw, &gw, &a, &b, &oldf, &b1, &scxbnd, &e,
                           &step, &factor, &braktd, &gtest1, &gtest2,
                           &tol);

        /* 是否发现新的最优点？ */
        if (*alpha == ualpha) {
            dcopy1(n, tempgfull, newgfull);
        }
    }

    if (itest == GETPTC_OK) {
        /* 已成功找到搜索点 */
        *f = fmin;
        daxpy1(n, *alpha, p, x);
        dcopy1(n, newgfull, gfull);
        rc = LS_OK;
    }
    /* 是否达到最大迭代次数？ */
    else if (itcnt > maxlsit) {
        rc = LS_FAIL;
    }
    /* 如果itest=GETPTC_FAIL或GETPTC_EINVAL，表示未找到更低的点 */
    else if (itest != GETPTC_EVAL) {
        rc = LS_FAIL;
    }
    /* 函数评估次数过多 */
    else {
        rc = LS_MAXFUN;
    }

  cleanup:
    free(temp);
    free(tempgfull);
    free(newgfull);

    return rc;
/*
 * getptc, an algorithm for finding a steplength, called repeatedly by
 * routines which require a step length to be computed using cubic
 * interpolation. The parameters contain information about the interval
 * in which a lower point is to be found and from this getptc computes a
 * point at which the function can be evaluated by the calling program.
 */
static getptc_rc getptcInit(double *reltol, double *abstol, double tnytol,
                            double eta, double rmu, double xbnd,
                            double *u, double *fu, double *gu,
                            double *xmin, double *fmin, double *gmin,
                            double *xw, double *fw, double *gw, double *a,
                            double *b, double *oldf, double *b1,
                            double *scxbnd, double *e, double *step,
                            double *factor, logical * braktd,
                            double *gtest1, double *gtest2, double *tol)
{
    /* Check input parameters */
    if (*u <= 0.0 || xbnd <= tnytol || *gu > 0.0) {
        return GETPTC_EINVAL;
    }
    if (xbnd < *abstol) {
        *abstol = xbnd;
    }
    *tol = *abstol;

    /* a and b define the interval of uncertainty, x and xw are points */
    /* with lowest and second lowest function values so far obtained. */
    /* initialize a,smin,xw at origin and corresponding values of */
    /* function and projection of the gradient along direction of search */
    /* at values for latest estimate at minimum. */

    *a = 0.0;
    *xw = 0.0;
    *xmin = 0.0;
    *oldf = *fu;
    *fmin = *fu;
    *fw = *fu;
    *gw = *gu;
    *gmin = *gu;
    *step = *u;
    *factor = 5.0;

    /* The minimum has not yet been bracketed. */
    *braktd = TNC_FALSE;

    /* Set up xbnd as a bound on the step to be taken. (xbnd is not computed */
    /* explicitly but scxbnd is its scaled value.) Set the upper bound */
    /* on the interval of uncertainty initially to xbnd + tol(xbnd). */
    *scxbnd = xbnd;
    *b = *scxbnd + *reltol * fabs(*scxbnd) + *abstol;
    *e = *b + *b;
    *b1 = *b;

    /* Compute the constants required for the two convergence criteria. */
    *gtest1 = -rmu * *gu;
    *gtest2 = -eta * *gu;

    /* If the step is too large, replace by the scaled bound (so as to */
    /* compute the new point on the boundary). */
    if (*step >= *scxbnd) {
        *step = *scxbnd;
        /* Move sxbd to the left so that sbnd + tol(xbnd) = xbnd. */
        *scxbnd -= (*reltol * fabs(xbnd) + *abstol) / (1.0 + *reltol);
    }
    *u = *step;
    if (fabs(*step) < *tol && *step < 0.0) {
        *u = -(*tol);
    }
    if (fabs(*step) < *tol && *step >= 0.0) {
        *u = *tol;
    }
    return GETPTC_EVAL;
}
    // 更新 a, b, xw, 和 xmin
    if (*fu <= *fmin) {
        // 如果函数值没有增加，新点成为下一个起点，并且其他点按比例缩放
        chordu = *oldf - (*xmin + *u) * *gtest1;
        if (*fu > chordu) {
            // 新的函数值不满足足够的减少条件，准备将上界移动到这一点，
            // 并强制插值方案进行二分搜索或线性插值步骤以估算根
            chordm = *oldf - *xmin * *gtest1;
            *gu = -(*gmin);
            denom = chordm - *fmin;
            if (fabs(denom) < 1e-15) {
                denom = 1e-15;
                if (chordm - *fmin < 0.0) {
                    denom = -denom;
                }
            }
            if (*xmin != 0.0) {
                *gu = *gmin * (chordu - *fu) / denom;
            }
            *fu = 0.5 * *u * (*gmin + *gu) + *fmin;
            if (*fu < *fmin) {
                *fu = *fmin;
            }
        } else {
            *fw = *fmin;
            *fmin = *fu;
            *gw = *gmin;
            *gmin = *gu;
            *xmin += *u;
            *a -= *u;
            *b -= *u;
            *xw = -(*u);
            *scxbnd -= *u;
            if (*gu <= 0.0) {
                *a = 0.0;
            } else {
                *b = 0.0;
                *braktd = TNC_TRUE;
            }
            *tol = fabs(*xmin) * *reltol + *abstol;
            // 转到收敛检查阶段
            goto ConvergenceCheck;
        }
    }

    // 如果函数值增加，起点保持不变，但新点可能成为新的w
    if (*u < 0.0) {
        *a = *u;
    } else {
        *b = *u;
        *braktd = TNC_TRUE;
    }
    *xw = *u;
    *fw = *fu;
    *gw = *gu;

  ConvergenceCheck:
    twotol = *tol + *tol;
    xmidpt = 0.5 * (*a + *b);

    // 检查终止条件
    /*
     * Determine if convergence criteria are met. Convergence occurs if the
     * midpoint xmidpt is sufficiently close to the interval edge, or if gmin
     * (minimum gradient) is small relative to gtest2, and fmin (minimum function
     * value) has decreased from oldf, and either xmin is far from xbnd or braktd
     * flag is false.
     */
    convrg = (fabs(xmidpt) <= twotol - 0.5 * (*b - *a)) ||
             (fabs(*gmin) <= *gtest2 && *fmin < *oldf
              && ((fabs(*xmin - xbnd) > *tol) || (!(*braktd))));
    
    if (convrg) {
        /*
         * If the minimum has not moved significantly, check if the change in
         * f(x) is within the specified function precision fpresn. If not, return
         * failure status.
         */
        if (fabs(*oldf - *fw) <= fpresn) {
            return GETPTC_FAIL;
        }
        
        // Reduce the tolerance parameters by a factor of 10
        *tol = 0.1 * *tol;
        if (*tol < tnytol) {
            return GETPTC_FAIL;
        }
        *reltol = 0.1 * *reltol;
        *abstol = 0.1 * *abstol;
        twotol = 0.1 * twotol;
    }

    /* 
     * Initialize variables r, q, and s for computing the trial step length.
     */
    r = 0.0;
    q = 0.0;
    s = 0.0;
    if (fabs(*e) > *tol) {
        /* 检查 e 的绝对值是否大于 tol */

        /* Fit cubic through xmin and xw */
        /* 通过 xmin 和 xw 拟合三次曲线 */
        r = 3.0 * (*fmin - *fw) / *xw + *gmin + *gw;
        /* 计算拟合三次曲线的系数 r */
        absr = fabs(r);
        /* 计算 r 的绝对值 */
        q = absr;
        /* 设置 q 初始值为 r 的绝对值 */
        if (*gw != 0.0 && *gmin != 0.0) {
            /* 判断 gw 和 gmin 是否都不为零 */

            /* Compute the square root of (r*r - gmin*gw) in a way
               which avoids underflow and overflow. */
            /* 以避免下溢和上溢的方式计算 sqrt(r*r - gmin*gw) */
            abgw = fabs(*gw);
            /* 计算 gw 的绝对值 */
            abgmin = fabs(*gmin);
            /* 计算 gmin 的绝对值 */
            s = sqrt(abgmin) * sqrt(abgw);
            /* 计算 s = sqrt(gmin) * sqrt(gw) */
            if (*gw / abgw * *gmin > 0.0) {
                /* 判断 gw / |gw| * gmin 是否大于零 */

                if (r >= s || r <= -s) {
                    /* 判断 r 是否大于等于 s 或小于等于 -s */

                    /* Compute the square root of r*r - s*s */
                    /* 计算 sqrt(r*r - s*s) */
                    q = sqrt(fabs(r + s)) * sqrt(fabs(r - s));
                }
                else {
                    r = 0.0;
                    q = 0.0;
                    goto MinimumFound;
                    /* 跳转至 MinimumFound 标签处 */
                }
            }
            else {
                /* 否则 */

                /* Compute the square root of r*r + s*s. */
                /* 计算 sqrt(r*r + s*s) */
                sumsq = 1.0;
                p = 0.0;
                if (absr >= s) {
                    /* 判断 absr 是否大于等于 s */

                    /* There is a possibility of underflow. */
                    /* 存在下溢的可能性 */
                    if (absr > rtsmll) {
                        p = absr * rtsmll;
                    }
                    if (s >= p) {
                        double value = s / absr;
                        sumsq = 1.0 + value * value;
                    }
                    scale = absr;
                }
                else {
                    /* There is a possibility of overflow. */
                    /* 存在上溢的可能性 */
                    if (s > rtsmll) {
                        p = s * rtsmll;
                    }
                    if (absr >= p) {
                        double value = absr / s;
                        sumsq = 1.0 + value * value;
                    }
                    scale = s;
                }
                sumsq = sqrt(sumsq);
                /* 计算 sqrt(sumsq) */
                q = big;
                /* 设置 q 的值为 big */
                if (scale < big / sumsq) {
                    q = scale * sumsq;
                }
            }
        }

        /* Compute the minimum of fitted cubic */
        /* 计算拟合三次曲线的最小值 */
        if (*xw < 0.0) {
            q = -q;
        }
        /* 若 xw 小于零，则将 q 取负值 */
        s = *xw * (*gmin - r - q);
        /* 计算 s = xw * (gmin - r - q) */
        q = *gw - *gmin + q + q;
        /* 更新 q = gw - gmin + 2*q */
        if (q > 0.0) {
            s = -s;
        }
        /* 若 q 大于零，则将 s 取负值 */
        if (q <= 0.0) {
            q = -q;
        }
        /* 若 q 小于等于零，则将 q 取负值 */
        r = *e;
        /* 更新 r 的值为 e */
        if (*b1 != *step || *braktd) {
            *e = *step;
        }
        /* 若 b1 不等于 step 或者 braktd 为真，则更新 e 的值为 step */
    }

  MinimumFound:
    /* Construct an artificial bound on the estimated steplength */
    /* 构造估计步长的人工界限 */
    a1 = *a;
    /* 将 a1 的值设置为 a 的值 */
    *b1 = *b;
    /* 将 b1 的值设置为 b 的值 */
    *step = xmidpt;
    /* 将 step 的值设置为 xmidpt */
    // 检查条件：如果 braktd 为假，或者 a 为零且 xw 小于零，或者 b 为零且 xw 大于零
    if ((!*braktd) || ((*a == 0.0 && *xw < 0.0) || (*b == 0.0 && *xw > 0.0))) {
        if (*braktd) {
            /* 如果最小值没有被 0 和 xw 夹住，步长必须位于 (a1,b1) 内。 */
            // 设置 d1 和 d2 的值
            d1 = *xw;
            d2 = *a;
            if (*a == 0.0) {
                d2 = *b;
            }
            // 计算 u 和 step 的值
            *u = -d1 / d2;
            *step = 5.0 * d2 * (0.1 + 1.0 / *u) / 11.0;
            if (*u < 1.0) {
                *step = 0.5 * d2 * sqrt(*u);
            }
        } else {
            // 计算 step 的值
            *step = -(*factor) * *xw;
            if (*step > *scxbnd) {
                *step = *scxbnd;
            }
            if (*step != *scxbnd) {
                *factor = 5.0 * *factor;
            }
        }
        /* 如果最小值被 0 和 xw 夹住，步长必须位于 (a,b) 内 */
        // 根据 step 的正负情况分别设置 a1 或 b1 的值
        if (*step <= 0.0) {
            a1 = *step;
        }
        if (*step > 0.0) {
            *b1 = *step;
        }
    }

    /*
     * 如果插值得到的步长在所需区间之外，或者大于上一次迭代的步长的一半，则拒绝该步长。
     */
    // 检查步长是否在指定条件内，否则更新 e 的值
    if (fabs(s) <= fabs(0.5 * q * r) || s <= q * a1 || s >= q * *b1) {
        *e = *b - *a;
    }
    else {
        /* 进行三次插值步骤 */
        // 计算新的步长 step
        *step = s / q;

        /* 函数评估时不能太靠近 a 或 b。 */
        // 如果 step 距离 a 或 b 过近，调整 step 的值
        if (*step - *a < twotol || *b - *step < twotol) {
            if (xmidpt <= 0.0) {
                *step = -(*tol);
            }
            else {
                *step = *tol;
            }
        }
    }

    /* 如果步长过大，用缩放边界替换（以便计算边界上的新点）。 */
    // 如果步长大于或等于 scxbnd，则用 scxbnd 替换 step，并更新 scxbnd 的值
    if (*step >= *scxbnd) {
        *step = *scxbnd;
        /* 将 scxbnd 向左移，使得 sbnd + tol(xbnd) = xbnd。 */
        *scxbnd -= (*reltol * fabs(xbnd) + *abstol) / (1.0 + *reltol);
    }
    *u = *step;
    // 根据 step 的绝对值与 tol 的比较，更新 u 的值
    if (fabs(*step) < *tol && *step < 0.0) {
        *u = -(*tol);
    }
    if (fabs(*step) < *tol && *step >= 0.0) {
        *u = *tol;
    }
    // 返回 GETPTC_EVAL 的值
    return GETPTC_EVAL;
/* Blas like routines */

/* dy+=dx */
static void dxpy1(int n, const double dx[], double dy[])
{
    int i;
    // 遍历数组，将 dx 的每个元素加到 dy 的对应位置上
    for (i = 0; i < n; i++) {
        dy[i] += dx[i];
    }
}

/* dy+=da*dx */
static void daxpy1(int n, double da, const double dx[], double dy[])
{
    int i;
    // 遍历数组，将 da 乘以 dx 的每个元素后加到 dy 的对应位置上
    for (i = 0; i < n; i++) {
        dy[i] += da * dx[i];
    }
}

/* Copy dx -> dy */
/* Could use memcpy */
static void dcopy1(int n, const double dx[], double dy[])
{
    int i;
    // 遍历数组，将 dx 的每个元素复制到 dy 的对应位置上
    for (i = 0; i < n; i++) {
        dy[i] = dx[i];
    }
}

/* Negate */
static void dneg1(int n, double v[])
{
    int i;
    // 遍历数组，将每个元素取负值
    for (i = 0; i < n; i++) {
        v[i] = -v[i];
    }
}

/* Dot product */
static double ddot1(int n, const double dx[], const double dy[])
{
    int i;
    double dtemp = 0.0;
    // 计算向量的点积
    for (i = 0; i < n; i++) {
        dtemp += dy[i] * dx[i];
    }
    return dtemp;
}

/* Euclidian norm */
static double dnrm21(int n, const double dx[])
{
    int i;
    double dssq = 1.0, dscale = 0.0;

    // 计算向量的二范数
    for (i = 0; i < n; i++) {
        if (dx[i] != 0.0) {
            double dabsxi = fabs(dx[i]);
            if (dscale < dabsxi) {
                /* Normalization to prevent overflow */
                // 将向量元素归一化，以防止溢出
                double ratio = dscale / dabsxi;
                dssq = 1.0 + dssq * ratio * ratio;
                dscale = dabsxi;
            }
            else {
                double ratio = dabsxi / dscale;
                dssq += ratio * ratio;
            }
        }
    }

    return dscale * sqrt(dssq);
}
```