# `D:\src\scipysrc\scipy\scipy\optimize\tnc\tnc.h`

```
/*
 * tnc : 截断牛顿方法的约束最小化问题
 *      使用C语言实现，并且利用梯度信息
 */

/*
 * 版权所有 (c) 2002-2005, Jean-Sebastien Roy (js@jeannot.org)
 *
 * 在遵守以下条件的情况下，免费授予任何获得此软件及相关文档的人
 * 复制、使用、修改、合并、发布、分发、再许可、出售此软件的权利：
 *
 * 上述版权声明和此许可声明应包含在所有副本或重要部分的软件中。
 *
 * 本软件按原样提供，不附带任何形式的明示或暗示担保，
 * 包括但不限于适销性、特定用途的适用性和非侵权性担保。
 * 无论在何种情况下，作者或版权持有者对任何索赔、损害或其他
 * 责任概不负责，无论是在合同行为、侵权行为或其他方面，
 * 由于使用本软件或与之相关的使用或其他交易引起的或
 * 与之相关的，甚至在告知可能发生此类损害的情况下。
 */

/*
 * 本软件是Stephen G. Nash原始在Fortran中开发的TNBC的C语言实现。
 *
 * 原始源代码可在以下地址找到：
 * http://iris.gmu.edu/~snash/nash/software/software.html
 *
 * 原始TNBC Fortran例程的版权：
 *
 *   截断牛顿方法: 子例程
 *     作者: Stephen G. Nash
 *           信息技术与工程学院
 *           乔治梅森大学
 *           弗吉尼亚州费尔法克斯市22030
 *
 * SciPy版本是从TNC 1.3派生而来的:
 * $Jeannot: tnc.h,v 1.55 2005/01/28 18:27:31 js Exp $
 */

#ifndef _TNC_
#define _TNC_

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 冗长级别
 */
typedef enum {
  TNC_MSG_NONE = 0, /* 无任何消息 */
  TNC_MSG_ITER = 1, /* 每次迭代输出一行 */
  TNC_MSG_INFO = 2, /* 提供信息性消息 */
  TNC_MSG_EXIT = 8, /* 退出原因 */

  TNC_MSG_ALL = TNC_MSG_ITER | TNC_MSG_INFO | TNC_MSG_EXIT /* 所有消息 */
} tnc_message;

/*
 * tnc的可能返回值
 */
typedef enum
{
  TNC_MINRC        = -3, /* 定义返回码 -3，用于获取相关的返回码字符串 */
  TNC_ENOMEM       = -3, /* 内存分配失败 */
  TNC_EINVAL       = -2, /* 参数无效 (n<0) */
  TNC_INFEASIBLE   = -1, /* 问题不可行 (下界 > 上界) */
  TNC_LOCALMINIMUM =  0, /* 达到局部最小值 (|pg| ~= 0) */
  TNC_FCONVERGED   =  1, /* 收敛 (|f_n-f_(n-1)| ~= 0) */
  TNC_XCONVERGED   =  2, /* 收敛 (|x_n-x_(n-1)| ~= 0) */
  TNC_MAXFUN       =  3, /* 达到最大函数评估次数 */
  TNC_LSFAIL       =  4, /* 线性搜索失败 */
  TNC_CONSTANT     =  5, /* 所有下界等于上界 */
  TNC_NOPROGRESS   =  6, /* 无法进展 */
  TNC_USERABORT    =  7  /* 用户请求结束最小化 */
} tnc_rc;

/*
 * 返回码字符串数组
 * 使用 tnc_rc_string[rc - TNC_MINRC] 获取与返回码 rc 相关联的消息
 */
extern const char *const tnc_rc_string[11];

/*
 * 满足 tnc 要求的函数类型
 * state 是一个每次调用时提供给函数的 void 指针
 *
 * x     : 输入时是变量向量 (不应修改)
 * f     : 输出时是函数值
 * g     : 输出时是梯度值
 * state : 输入时是作为给定给 tnc 的状态变量的值
 *
 * 返回 0 表示无错误发生，返回 1 表示立即结束最小化
 */
typedef int tnc_function(double x[], double *f, double g[], void *state);

/*
 * 回调函数，接受 x 和 state 指针作为输入参数
 */
typedef void tnc_callback(double x[], void *state);
/*
 * tnc : minimize a function with variables subject to bounds, using
 *       gradient information.
 *
 * n         : number of variables (must be >= 0)
 * x         : on input, initial estimate ; on output, the solution
 * f         : on output, the function value at the solution
 * g         : on output, the gradient value at the solution
 *             g should be an allocated vector of size n or NULL,
 *             in which case the gradient value is not returned.
 * function  : the function to minimize (see tnc_function)
 * state     : used by function (see tnc_function)
 * low, up   : the bounds
 *             set low[i] to -HUGE_VAL to remove the lower bound
 *             set up[i] to HUGE_VAL to remove the upper bound
 *             if low == NULL, the lower bounds are removed.
 *             if up == NULL, the upper bounds are removed.
 * scale     : scaling factors to apply to each variable
 *             if NULL, the factors are up-low for interval bounded variables
 *             and 1+|x| for the others.
 * offset    : constant to subtract from each variable
 *             if NULL, the constants are (up+low)/2 for interval bounded
 *             variables and x for the others.
 * messages  : see the tnc_message enum
 * maxCGit   : max. number of hessian*vector evaluation per main iteration
 *             if maxCGit == 0, the direction chosen is -gradient
 *             if maxCGit < 0, maxCGit is set to max(1,min(50,n/2))
 * maxnfeval : max. number of function evaluations
 * eta       : severity of the line search. if < 0 or > 1, set to 0.25
 * stepmx    : maximum step for the line search. may be increased during call
 *             if too small, will be set to 10.0
 * accuracy  : relative precision for finite difference calculations
 *             if <= machine_precision, set to sqrt(machine_precision)
 * fmin      : minimum function value estimate
 * ftol      : precision goal for the value of f in the stopping criterion
 *             if ftol < 0.0, ftol is set to accuracy
 * xtol      : precision goal for the value of x in the stopping criterion
 *             (after applying x scaling factors)
 *             if xtol < 0.0, xtol is set to sqrt(machine_precision)
 * pgtol     : precision goal for the value of the projected gradient in the
 *             stopping criterion (after applying x scaling factors)
 *             if pgtol < 0.0, pgtol is set to 1e-2 * sqrt(accuracy)
 *             setting it to 0.0 is not recommended
 * rescale   : f scaling factor (in log10) used to trigger f value rescaling
 *             if 0, rescale at each iteration
 *             if a large value, never rescale
 *             if < 0, rescale is set to 1.3
 * nfeval    : on output, the number of function evaluations.
 *             ignored if nfeval==NULL.
 *
 * The tnc function returns a code defined in the tnc_rc enum.
 * On output, x, f and g may be very slightly out of sync because of scaling.
 *
 */
/*
   声明一个名为 tnc 的外部函数，该函数接受多个参数，包括整数 n，双精度数组 x[]，
   指向双精度变量 f 的指针，双精度数组 g[]，指向 tnc_function 结构的指针 function，
   通用指针 state，双精度数组 low[]，up[]，scale[]，offset[]，整数 messages，maxCGit，
   maxnfeval，双精度变量 eta，stepmx，accuracy，fmin，ftol，xtol，pgtol，rescale，
   整数指针 nfeval 和 niter，以及指向 tnc_callback 结构的指针 callback。
*/
extern int tnc(int n, double x[], double *f, double g[],
  tnc_function *function, void *state,
  double low[], double up[], double scale[], double offset[],
  int messages, int maxCGit, int maxnfeval, double eta, double stepmx,
  double accuracy, double fmin, double ftol, double xtol, double pgtol,
  double rescale, int *nfeval, int *niter, tnc_callback *callback);

#ifdef __cplusplus
}
#endif

#endif /* _TNC_ */
```