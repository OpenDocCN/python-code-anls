# `D:\src\scipysrc\scipy\scipy\optimize\_direct\direct_wrap.c`

```
/* C-style API for DIRECT functions.  SGJ (August 2007). */

#include "direct-internal.h"

/* Perform global minimization using (Gablonsky implementation of) DIRECT
   algorithm.   Arguments:

   f, f_data: the objective function and any user data
       -- the objective function f(n, x, undefined_flag, data) takes 4 args:
              int n: the dimension, same as dimension arg. to direct_optimize
              const double *x: array x[n] of point to evaluate
              int *undefined_flag: set to 1 on return if x violates constraints
                                   or don't touch otherwise
              void *data: same as f_data passed to direct_optimize
          return value = value of f(x)

   dimension: the number of minimization variable dimensions
   lower_bounds, upper_bounds: arrays of length dimension of variable bounds

   x: an array of length dimension, set to optimum variables upon return
   minf: on return, set to minimum f value

   magic_eps, magic_eps_abs: Jones' "magic" epsilon parameter, and
                             also an absolute version of the same
                 (not multipled by minf).  Jones suggests
                 setting this to 1e-4, but 0 also works...

   max_feval, max_iter: maximum number of function evaluations & DIRECT iters
   volume_reltol: relative tolerance on hypercube volume (0 if none)
   sigma_reltol: relative tolerance on hypercube "measure" (??) (0 if none)

   fglobal: the global minimum of f, if known ahead of time
       -- this is mainly for benchmarking, in most cases it
          is not known and you should pass DIRECT_UNKNOWN_FGLOBAL
   fglobal_reltol: relative tolerance on how close we should find fglobal
       -- ignored if fglobal is DIRECT_UNKNOWN_FGLOBAL

   logfile: an output file to write diagnostic info to (NULL for no I/O)

   algorithm: whether to use the original DIRECT algorithm (DIRECT_ORIGINAL)
              or Gablonsky's "improved" version (DIRECT_GABLONSKY)
*/
PyObject* direct_optimize(
    PyObject* f,              // Python object representing the objective function
    double *x,                // Array of length dimension to hold optimal variables
    PyObject *x_seq,          // Sequence of variables for the optimization
    PyObject* args,           // Additional arguments for the objective function
    int dimension,            // Number of variables to be optimized
    const double *lower_bounds,  // Array of length dimension defining lower bounds for variables
    const double *upper_bounds,  // Array of length dimension defining upper bounds for variables
    double *minf,             // Pointer to store the minimum value of the objective function
    int max_feval,            // Maximum number of function evaluations allowed
    int max_iter,             // Maximum number of iterations allowed for DIRECT algorithm
    double magic_eps,         // Jones' "magic" epsilon parameter for relative tolerance
    double magic_eps_abs,     // Absolute version of the magic epsilon parameter
    double volume_reltol,     // Relative tolerance on hypercube volume
    double sigma_reltol,      // Relative tolerance on hypercube "measure"
    int *force_stop,          // Flag to force stop the optimization
    double fglobal,           // Global minimum value of the objective function (if known)
    double fglobal_reltol,    // Relative tolerance on closeness to fglobal
    FILE *logfile,            // Log file for diagnostic information
    direct_algorithm algorithm,  // Type of DIRECT algorithm to use
    direct_return_info *info, // Pointer to struct for returning optimization info
    direct_return_code* ret_code,  // Pointer to store return code of optimization
    PyObject* callback        // Optional callback function for optimization
)
{
     // 检查算法是否为直接法 Gablonsky，将结果存储在 algmethod 中
     integer algmethod = algorithm == DIRECT_GABLONSKY;
     // 定义错误码变量
     integer ierror;
     // 定义指向下界和上界数组的指针
     doublereal *l, *u;
     // 定义循环计数器
     int i;

     /* 确保如果小于等于 0，则忽略体积相对容差和标准差相对容差 */
     if (volume_reltol <= 0) volume_reltol = -1;
     if (sigma_reltol <= 0) sigma_reltol = -1;

     // 如果全局函数未知，则设置全局函数相对容差为默认值
     if (fglobal == DIRECT_UNKNOWN_FGLOBAL)
      fglobal_reltol = DIRECT_UNKNOWN_FGLOBAL_RELTOL;

     // 如果维度小于 1，则设置返回码为无效参数
     if (dimension < 1) *ret_code = DIRECT_INVALID_ARGS;

     // 分配存储下界和上界数组的内存空间
     l = (doublereal *) malloc(sizeof(doublereal) * dimension * 2);
     // 如果内存分配失败，则设置返回码为内存不足
     if (!l) *ret_code = DIRECT_OUT_OF_MEMORY;
     // 设置上界数组的指针
     u = l + dimension;
     // 将传入的下界和上界数组的值复制到 l 和 u 数组中
     for (i = 0; i < dimension; ++i) {
      l[i] = lower_bounds[i];
      u[i] = upper_bounds[i];
     }

     // 定义存储函数评估和迭代次数的变量
     int numfunc;
     int numiter;
     
     // 调用外部函数 direct_direct_ 进行优化计算
     PyObject* ret = direct_direct_(f, x, x_seq, &dimension, &magic_eps, magic_eps_abs,
            &max_feval, &max_iter, 
            force_stop,
            minf,
            l, u,
            &algmethod,
            &ierror,
            logfile,
            &fglobal, &fglobal_reltol,
            &volume_reltol, &sigma_reltol,
            args, &numfunc, &numiter, callback);

    // 将函数评估和迭代次数存储到 info 结构体中
    info->numfunc = numfunc;
    info->numiter = numiter;

    // 释放 l 数组所占用的内存空间
    free(l);

    // 将 ierror 强制转换为返回码类型，并返回 ret 对象
    *ret_code = (direct_return_code) ierror;
    return ret;
}
```