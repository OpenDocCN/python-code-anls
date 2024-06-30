# `D:\src\scipysrc\scipy\scipy\optimize\_direct\DIRect.c`

```
/* DIRect-transp.f -- translated by f2c (version 20050501).

   f2c output hand-cleaned by SGJ (August 2007).
*/

#include "direct-internal.h"
#include <math.h>

/* Common Block Declarations */

/* Table of constant values */

/* +-----------------------------------------------------------------------+ */
/* | Program       : Direct.f                                              | */
/* | Last modified : 07-16-2001                                            | */
/* | Written by    : Joerg Gablonsky (jmgablon@unity.ncsu.edu)             | */
/* |                 North Carolina State University                       | */
/* |                 Dept. of Mathematics                                  | */
/* | DIRECT is a method to solve problems of the form:                     | */
/* |              min f: Q --> R,                                          | */
/* | where f is the function to be minimized and Q is an n-dimensional     | */
/* | hyperrectangle given by the following equation:                       | */
/* |                                                                       | */
/* |       Q={ x : l(i) <= x(i) <= u(i), i = 1,...,n }.                    | */
/* | Note: This version of DIRECT can also handle hidden constraints. By   | */
/* |       this we mean that the function may not be defined over the whole| */
/* |       hyperrectangle Q, but only over a subset, which is not given    | */
/* |       analytically.                                                   | */
/* |                                                                       | */
/* | We now give a brief outline of the algorithm:                         | */
/* |                                                                       | */
/* |   The algorithm starts with mapping the hyperrectangle Q to the       | */
/* |   n-dimensional unit hypercube. DIRECT then samples the function at   | */
/* |   the center of this hypercube and at 2n more points, 2 in each       | */
/* |   coordinate direction. Using these function values, DIRECT then      | */
/* |   divides the domain into hyperrectangles, each having exactly one of | */
/* |   the sampling points as its center. In each iteration, DIRECT chooses| */
/* |   some of the existing hyperrectangles to be further divided.         | */
/* |   We provide two different strategies of how to decide which          | */
/* |   hyperrectangles DIRECT divides and several different convergence    | */
/* |   criteria.                                                           | */
/* |                                                                       | */
/* |   DIRECT was designed to solve problems where the function f is       | */
/* |   Lipschitz continuous. However, DIRECT has proven to be effective on | */
/* |   more complex problems than these.                                   | */
/* +-----------------------------------------------------------------------+ */


注释：
/* Subroutine definition in C, accepting various parameters and returning a PyObject pointer */
PyObject* direct_direct_(PyObject* fcn, doublereal *x, PyObject *x_seq,
    integer *n, doublereal *eps, doublereal epsabs, integer *maxf, integer *maxt, int *force_stop,
    doublereal *minf, doublereal *l, doublereal *u, integer *algmethod, integer *ierror,
    FILE *logfile, doublereal *fglobal, doublereal *fglper, doublereal *volper,
    doublereal *sigmaper, PyObject* args, integer *numfunc, integer *numiter, PyObject* callback)
{
    PyObject *ret = NULL;   /* Initialize return object to NULL */

    /* System generated locals */
    integer i__1, i__2;   /* System-generated integer variables */

    const integer MAXDIV = 5000;   /* Constant maximum division value */

    /* Local variables declaration */
    integer increase;   /* Integer variable for increase calculation */
    doublereal *c__ = 0;   /* Pointer to doublereal array c__ of size 90000x64 */
    doublereal *f = 0;    /* Pointer to doublereal array f of size 90000x2 */
    integer i__, j;    /* Loop counters */
    integer *s = 0;    /* Pointer to integer array s of size 3000x2 */
    integer t = 0;    /* Integer variable t initialized to 0 */
    doublereal *w = 0;    /* Pointer to doublereal array w */
    doublereal divfactor;    /* Doublereal variable divfactor */
    integer ifeasiblef, iepschange, actmaxdeep;   /* Integer variables ifeasiblef, iepschange, actmaxdeep */
    integer actdeep_div__;    /* Integer variable actdeep_div__ */
    integer iinfesiblef;    /* Integer variable iinfesiblef */
    integer pos1, newtosample;    /* Integer variables pos1, newtosample */
    integer ifree, help;    /* Integer variables ifree, help */
    doublereal *oldl = 0, fmax;    /* Pointer to doublereal array oldl, doublereal variable fmax */
    integer maxi;    /* Integer variable maxi */
    doublereal kmax, *oldu = 0;    /* Doublereal variables kmax, pointer to doublereal array oldu */
    integer oops, *list2 = 0;    /* Integer variables oops, pointer to integer array list2 of size 64x2 */
    doublereal delta;    /* Doublereal variable delta */
    integer mdeep, *point = 0, start;    /* Integer variables mdeep, pointer to integer array point, start */
    integer *anchor = 0, *length = 0;    /* Pointers to integer arrays anchor, length of size 90000x64 */
    integer *arrayi = 0;    /* Pointer to integer array arrayi */
    doublereal *levels = 0, *thirds = 0;    /* Pointers to doublereal arrays levels, thirds */
    doublereal epsfix;    /* Doublereal variable epsfix */
    integer oldpos, minpos, maxpos, tstart, actdeep, ifreeold, oldmaxf;    /* Integer variables */
    integer version;    /* Integer variable version */
    integer jones;    /* Integer variable jones */

    /* Note that I've transposed c__, length, and f relative to the
       original Fortran code.  e.g. length was length(maxfunc,n)
       in Fortran [ or actually length(maxfunc, maxdims), but by
       using malloc I can just allocate n ], corresponding to
       length[n][maxfunc] in C, but I've changed the code to access
       it as length[maxfunc][n].  That is, the maxfunc direction
       is the discontiguous one.  This makes it easier to resize
       dynamically (by adding contiguous rows) using realloc, without
       having to move data around manually. */

    #define MAXMEMORY 1073741824   /* Maximum allowable memory size */
    integer MAXFUNC = *maxf <= 0 ? 101000 : (*maxf + 1000 + *maxf / 2);   /* Calculate MAXFUNC based on maxf */
    integer fixed_memory_dim = ((*n) * (sizeof(doublereal) + sizeof(integer)) +
                                      (sizeof(doublereal) * 2 + sizeof(integer)));   /* Calculate fixed_memory_dim */
    MAXFUNC = MAXFUNC * fixed_memory_dim > MAXMEMORY ? MAXMEMORY/fixed_memory_dim : MAXFUNC;   /* Adjust MAXFUNC if memory exceeds limit */
    c__ = (doublereal *) malloc(sizeof(doublereal) * (MAXFUNC * (*n)));   /* Allocate memory for array c__ */
    if (!(c__)) {
        *ierror = -100;   /* Set error code to -100 if malloc fails */
        goto cleanup;   /* Jump to cleanup label */
    }
    length = (integer*) malloc(sizeof(integer) * (MAXFUNC * (*n)));   /* Allocate memory for array length */
    if (!(length)) {
        *ierror = -100; goto cleanup;   /* Set error code to -100 if malloc fails and jump to cleanup */
    }
    f = (doublereal*) malloc(sizeof(doublereal) * (MAXFUNC * 2));   /* Allocate memory for array f */
    if (!(f)) {
        *ierror = -100; goto cleanup;   /* Set error code to -100 if malloc fails and jump to cleanup */
    }
    point = (integer*) malloc(sizeof(integer) * (MAXFUNC));   /* Allocate memory for array point */
    if (!(point)) {
        *ierror = -100; goto cleanup;   /* Set error code to -100 if malloc fails and jump to cleanup */
    }
    if (*maxf <= 0)
        *maxf = MAXFUNC - 1000;   /* Adjust maxf if it was originally less than or equal to 0 */
    else
        *maxf = 2*(MAXFUNC - 1000)/3;   /* Adjust maxf based on MAXFUNC */

cleanup:   /* Cleanup label */
    /* Cleanup code would typically go here, dealing with freeing allocated memory */
    return ret;   /* Return the PyObject pointer */
}
    // 分配内存以存储整数数组，大小为 MAXDIV * 2 个整数
    s = (integer*) malloc(sizeof(integer) * (MAXDIV * 2));
    // 如果内存分配失败，设置错误码并跳转到清理代码块
    if (!(s)) {
        *ierror = -100; goto cleanup;
    }

    // 计算 MAXDEEP 的值，根据条件判断 *maxt 的值
    integer MAXDEEP = *maxt <= 0 ? MAXFUNC/5 : *maxt + 1000;
    // 计算固定内存维度的大小
    fixed_memory_dim = (sizeof(doublereal) * 2 + sizeof(integer));
    // 计算常量内存的大小
    integer const_memory = 2 * (sizeof(doublereal) + sizeof(integer));
    // 根据内存限制调整 MAXDEEP 的值
    MAXDEEP = MAXDEEP * fixed_memory_dim + const_memory > MAXMEMORY ? (MAXMEMORY - const_memory) / fixed_memory_dim : MAXDEEP;

    // 分配内存以存储整数数组，大小为 MAXDEEP + 2 个整数
    anchor = (integer*) malloc(sizeof(integer) * (MAXDEEP + 2));
    // 如果内存分配失败，设置错误码并跳转到清理代码块
    if (!(anchor)) {
        *ierror = -100; goto cleanup;
    }

    // 分配内存以存储双精度浮点数数组，大小为 MAXDEEP + 1 个双精度浮点数
    levels = (doublereal*) malloc(sizeof(doublereal) * (MAXDEEP + 1));
    // 如果内存分配失败，设置错误码并跳转到清理代码块
    if (!(levels)) {
        *ierror = -100; goto cleanup;
    }

    // 分配内存以存储双精度浮点数数组，大小为 MAXDEEP + 1 个双精度浮点数
    thirds = (doublereal*) malloc(sizeof(doublereal) * (MAXDEEP + 1));
    // 如果内存分配失败，设置错误码并跳转到清理代码块
    if (!(thirds)) {
        *ierror = -100; goto cleanup;
    }

    // 根据条件设置 *maxt 的值
    if (*maxt <= 0)
        *maxt = MAXDEEP;
    else
        *maxt = MAXDEEP - 1000;

    // 分配内存以存储双精度浮点数数组，大小为 *n 个双精度浮点数
    w = (doublereal*) malloc(sizeof(doublereal) * (*n));
    // 如果内存分配失败，设置错误码并跳转到清理代码块
    if (!(w)) {
        *ierror = -100; goto cleanup;
    }

    // 分配内存以存储双精度浮点数数组，大小为 *n 个双精度浮点数
    oldl = (doublereal*) malloc(sizeof(doublereal) * (*n));
    // 如果内存分配失败，设置错误码并跳转到清理代码块
    if (!(oldl)) {
        *ierror = -100; goto cleanup;
    }

    // 分配内存以存储双精度浮点数数组，大小为 *n 个双精度浮点数
    oldu = (doublereal*) malloc(sizeof(doublereal) * (*n));
    // 如果内存分配失败，设置错误码并跳转到清理代码块
    if (!(oldu)) {
        *ierror = -100; goto cleanup;
    }

    // 分配内存以存储整数数组，大小为 *n * 2 个整数
    list2 = (integer*) malloc(sizeof(integer) * (*n * 2));
    // 如果内存分配失败，设置错误码并跳转到清理代码块
    if (!(list2)) {
        *ierror = -100; goto cleanup;
    }

    // 分配内存以存储整数数组，大小为 *n 个整数
    arrayi = (integer*) malloc(sizeof(integer) * (*n));
    // 如果内存分配失败，设置错误码并跳转到清理代码块
    if (!(arrayi)) {
        *ierror = -100; goto cleanup;
    }
/* +-----------------------------------------------------------------------+ */
/* |    SUBROUTINE Direct                                                  | */
/* | On entry                                                              | */
/* |     fcn -- The argument containing the name of the user-supplied      | */
/* |            SUBROUTINE that returns values for the function to be      | */
/* |            minimized.                                                 | */
/* |       n -- The dimension of the problem.                              | */
/* |     eps -- Exceeding value. If eps > 0, we use the same epsilon for   | */
/* |            all iterations. If eps < 0, we use the update formula from | */
/* |            Jones:                                                     | */
/* |               eps = max(1.D-4*abs(minf),epsfix),                      | */
/* |            where epsfix = abs(eps), the absolute value of eps which is| */
/* |            passed to the function.                                    | */
/* |    maxf -- The maximum number of function evaluations.                | */
/* |    maxT -- The maximum number of iterations.                          | */
/* |            Direct stops when either the maximum number of iterations  | */
/* |            is reached or more than maxf function-evaluations were made.| */
/* |       l -- The lower bounds of the hyperbox.                          | */
/* |       u -- The upper bounds of the hyperbox.                          | */
/* |algmethod-- Choose the method, that is either use the original method  | */
/* |            as described by Jones et.al. (0) or use our modification(1)| */
/* | logfile -- File-Handle for the logfile. DIRECT expects this file to be| */
/* |            opened and closed by the user outside of DIRECT. We moved  | */
/* |            this to the outside so the user can add extra informations | */
/* |            to this file before and after the call to DIRECT.          | */
/* | fglobal -- Function value of the global optimum. If this value is not | */
/* |            known (that is, we solve a real problem, not a testproblem)| */
/* |            set this value to -1.D100 and fglper (see below) to 0.D0.  | */
/* |  fglper -- Terminate the optimization when the relative error          | */
/* |                (f_min - fglobal)/max(1,abs(fglobal)) < fglper.     | */
/* |  volper -- Terminate the optimization when the volume of the          | */
/* |            hyperrectangle S with f(c(S)) = minf is less then volper   | */
/* |            of the volume of the original hyperrectangle.      | */
/* |sigmaper -- Terminate the optimization when the measure of the         | */
/* |            hyperrectangle S with f(c(S)) = minf is less then sigmaper.| */
/* |                                                                       | */
/* | User data that is passed through without being changed:               | */
/* +-----------------------------------------------------------------------+ */
"""
# fcn_data - 任意用户数据的不透明指针
# 
# 返回：
# 
#       x -- 优化过程中获得的最终点。X 应该是函数在超立方体内全局最小值的良好近似。
# 
#    minf -- 函数在 x 处的值。
#  Ierror -- 错误标志。如果 Ierror 小于 0，则发生错误。
#            Ierror 的值意义如下：
#            致命错误：
#             -1   对于某些 i，u(i) <= l(i)。
#             -2   maxf 过大。
#             -3   DIRpreprc 初始化失败。
#             -4   DIRSamplepoints 中的错误，即创建样本点时发生错误。
#             -5   DIRSamplef 中的错误，即在对函数进行采样时发生错误。
#             -6   DIRDoubleInsert 中的错误，即 DIRECT 尝试添加所有大小相同且函数值相同的超矩形的错误。要么增加 maxdiv，要么使用我们的修改（Jones = 1）。
#            终止值：
#              1   完成的函数评估次数大于 maxf。
#              2   迭代次数等于 maxT。
#              3   找到的最佳函数值在 fglobal 的 fglper 范围内。注意，只有当全局最优值已知时才会出现此终止信号，即优化测试函数。
#              4   在以 minf 为中心的超矩形的体积小于原始超矩形体积的 volper。
#              5   以 minf 为中心的超矩形的度量。
"""
/* |                  center is less than sigmaper.                        | */
/* |                                                                       | */
/* | SUBROUTINEs used :                                                    | */
/* |                                                                       | */
/* | DIRheader, DIRInitSpecific, DIRInitList, DIRpreprc, DIRInit, DIRChoose| */
/* | DIRDoubleInsert, DIRGet_I, DIRSamplepoints, DIRSamplef, DIRDivide     | */
/* | DIRInsertList, DIRreplaceInf, DIRWritehistbox, DIRsummary, Findareas  | */
/* |                                                                       | */
/* | Functions used :                                                      | */
/* |                                                                       | */
/* | DIRgetMaxdeep, DIRgetlevel                                            | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | Parameters                                                            | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | The maximum of function evaluations allowed.                          | */
/* | The maximum dept of the algorithm.                                    | */
/* | The maximum number of divisions allowed.                              | */
/* | The maximal dimension of the problem.                                 | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | Global Variables.                                                     | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | EXTERNAL Variables.                                                   | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | User Variables.                                                       | */
/* | These can be used to pass user defined data to the function to be     | */
/* | optimized.                                                            | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | Place to define, if needed, some application-specific variables.      | */
/* | Note: You should try to use the arrays defined above for this.        | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
# End of application-specific variables
# Internal variables:
# f -- values of functions.
# divfactor -- Factor used for termination with known global minimum.
# anchor -- anchors of lists with deepness i, -1 is anchor for list of NaN-values.
# S -- List of potentially optimal points.
# point -- lists
# ifree -- first free position
# c -- midpoints of arrays
# thirds -- Precalculated values of 1/3^i.
# levels -- Length of intervals.
# length -- Length of intervall (index)
# t -- actual iteration
# j -- loop-variable
# actdeep -- the actual minimal interval-length index
# Minpos -- position of the actual minimum
# file -- The filehandle for a datafile.
# maxpos -- The number of intervalls, which are truncated.
# help -- A help variable.
# numfunc -- The actual number of function evaluations.
# file2 -- The filehandle for another datafile.
# ArrayI -- Array with the indexes of the sides with maximum length.
# maxi -- Number of directions with maximal side length.
# oops -- Flag which shows if anything went wrong in the initialisation.
# cheat -- Obsolete. If equal 1, we don't allow Ktilde > kmax.
# writed -- If writed=1, store final division to plot with Matlab.
# List2 -- List of indicies of intervalls, which are to be truncated.
# i -- Another loop-variable.
# actmaxdeep -- The actual maximum (minimum) of possible Interval length.
# oldpos -- The old index of the minimum. Used to print only, if there is a new minimum found.
# tstart -- The start of the outer loop.
# start -- The position of the starting point in the inner loop.
# Newtosample -- The total number of points to sample in the inner loop.
w -- Array used to divide the intervalls
kmax -- Obsolete. If cheat = 1, Ktilde was not allowed to be larger than kmax. If Ktilde > kmax, we set ktilde = kmax.
delta -- The distance to new points from center of old hyperrec.
pos1 -- Help variable used as an index.
version -- Store the version number of DIRECT.
oldmaxf -- Store the original function budget.
increase -- Flag used to keep track if function budget was increased because no feasible point was found.
ifreeold -- Keep track which index was free before. Used with SUBROUTINE DIRReplaceInf.
actdeep_div -- Keep track of the current depths for divisions.
oldl -- Array used to store the original bounds of the domain.
oldu -- Array used to store the original bounds of the domain.
epsfix -- If eps < 0, we use Jones update formula. epsfix stores the absolute value of epsilon.
iepschange -- flag iepschange to store if epsilon stays fixed or is changed.
DIRgetMaxdeep -- Function to calculate the level of a hyperrectangle.
DIRgetlevel -- Function to calculate the level and stage of a hyperrec.
fmax -- Keep track of the maximum value of the function found.
Ifeasiblef -- Keep track if a feasible point has been found so far. Ifeasiblef = 0 means a feasible point has been found, Ifeasiblef = 1 no feasible point has been found.
/* |            fixed or is changed.                                       | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | JG 01/22/01 fmax is used to keep track of the maximum value found.    | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | JG 01/22/01 Ifeasiblef is used to keep track if a feasible point has  | */
/* |             been found so far. Ifeasiblef = 0 means a feasible point  | */
/* |             has been found, Ifeasiblef = 1 if not.                    | */
/* | JG 03/09/01 IInfeasible is used to keep track if an infeasible point  | */
/* |             has been found. IInfeasible > 0 means a infeasible point  | */
/* |             has been found, IInfeasible = 0 if not.                   | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* |                            Start of code.                             | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
    /* Parameter adjustments */
    --u;
    --l;
    --x;

    /* Function Body */
    jones = *algmethod;
/* +-----------------------------------------------------------------------+ */
/* | Save the upper and lower bounds.                                      | */
/* +-----------------------------------------------------------------------+ */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        oldu[i__ - 1] = u[i__];
        oldl[i__ - 1] = l[i__];
/* L150: */
    }
/* +-----------------------------------------------------------------------+ */
/* | Set version.                                                          | */
/* +-----------------------------------------------------------------------+ */
    version = 204;
/* +-----------------------------------------------------------------------+ */
/* | Set parameters.                                                       | */
/* |    If cheat > 0, we do not allow \tilde{K} to be larger than kmax, and| */
/* |    set \tilde{K} to set value if necessary. Not used anymore.         | */
/* +-----------------------------------------------------------------------+ */
    cheat = 0;
    kmax = 1e10;
    mdeep = MAXDEEP;
/* +-----------------------------------------------------------------------+ */
/* | Write the header of the logfile.                                      | */
/* +-----------------------------------------------------------------------+ */
    # 调用一个函数 direct_dirheader_，传递以下参数：
    # - logfile: 指向 logfile 的指针或引用
    # - version: 指向 version 的指针或引用
    # - x[1]: 指向 x[1] 的指针或引用
    # - x_seq: x_seq 的值
    # - n: n 的值
    # - eps: eps 的值
    # - maxf: maxf 的值
    # - maxt: maxt 的值
    # - l[1]: 指向 l[1] 的指针或引用
    # - u[1]: 指向 u[1] 的指针或引用
    # - algmethod: algmethod 的值
    # - MAXFUNC: 指向 MAXFUNC 的指针或引用
    # - MAXDEEP: 指向 MAXDEEP 的指针或引用
    # - fglobal: fglobal 的值
    # - fglper: fglper 的值
    # - ierror: 指向 ierror 的指针或引用
    # - epsfix: 指向 epsfix 的指针或引用
    # - iepschange: 指向 iepschange 的指针或引用
    # - volper: volper 的值
    # - sigmaper: sigmaper 的值
/* +-----------------------------------------------------------------------+ */
/* | 如果在写入标题时发生错误（我们在那里对一些变量进行了检查），返回主程序。  | */
/* +-----------------------------------------------------------------------+ */
if (*ierror < 0) {
    goto cleanup;
}

/* +-----------------------------------------------------------------------+ */
/* | 如果已知的全局最小值等于0，我们不能除以它。因此我们将其设置为1。如果不是，  | */
/* | 我们将divisionfactor设置为全局最小值的绝对值。                           | */
/* +-----------------------------------------------------------------------+ */
if (*fglobal == 0.) {
    divfactor = 1.;
} else {
    divfactor = fabs(*fglobal);
}

/* +-----------------------------------------------------------------------+ */
/* | 保存用户给定的预算。如果在开始时找不到可行点，则maxf变量将被更改。           | */
/* +-----------------------------------------------------------------------+ */
oldmaxf = *maxf;
increase = 0;

/* +-----------------------------------------------------------------------+ */
/* | 初始化列表。                                                          | */
/* +-----------------------------------------------------------------------+ */
direct_dirinitlist_(anchor, &ifree, point, f, &MAXFUNC, &MAXDEEP);

/* +-----------------------------------------------------------------------+ */
/* | 调用程序初始化映射x，从n维单位立方体到由u和l定义的超立方体。如果发生错误，   | */
/* | 输出错误消息并设置错误标志，然后返回主程序。                           | */
/* | JG 07/16/01 更改调用以删除未使用的数据。                                | */
/* +-----------------------------------------------------------------------+ */
direct_dirpreprc_(&u[1], &l[1], n, &l[1], &u[1], &oops);
if (oops > 0) {
    if (logfile)
         fprintf(logfile, "WARNING: 初始化DIRpreprc失败。\n");
    *ierror = -3;
    goto cleanup;
}

tstart = 2;

/* +-----------------------------------------------------------------------+ */
/* | 初始化DIRECT算法。                                                    | */
/* +-----------------------------------------------------------------------+ */

/* +-----------------------------------------------------------------------+ */
/* | 添加变量以跟踪找到的最大值。                                            | */
/* +-----------------------------------------------------------------------+ */
    // 调用名为 direct_dirinit_ 的函数，并传递以下参数：
    // f, fcn, c__, length, &actdeep, point, anchor, &ifree,
    // logfile, arrayi, &maxi, list2, w, &x[1], x_seq, &l[1], &u[1],
    // minf, &minpos, thirds, levels, &MAXFUNC, &MAXDEEP, n, n,
    // &fmax, &ifeasiblef, &iinfesiblef, ierror, args, jones,
    // force_stop。将返回值存储在 ret 变量中。
    ret = direct_dirinit_(f, fcn, c__, length, &actdeep, point, anchor, &ifree,
        logfile, arrayi, &maxi, list2, w, &x[1], x_seq, &l[1], &u[1],
        minf, &minpos, thirds, levels, &MAXFUNC, &MAXDEEP, n, n, &
        fmax, &ifeasiblef, &iinfesiblef, ierror, args, jones,
        force_stop);
    // 如果 ret 的值为假（0或NULL），则返回空指针(NULL)
    if (!ret) {
        return NULL;
    }
    /* +-----------------------------------------------------------------------+ */
    /* | Added error checking.                                                 | */
    /* +-----------------------------------------------------------------------+ */
    // 检查错误标志 *ierror，处理不同的错误情况
    if (*ierror < 0) {
        // 如果 *ierror 为 -4，输出警告信息并跳转到清理步骤
        if (*ierror == -4) {
            if (logfile)
                fprintf(logfile, "WARNING: Error occurred in routine DIRsamplepoints.\n");
            goto cleanup;
        }
        // 如果 *ierror 为 -5，输出警告信息并跳转到清理步骤
        if (*ierror == -5) {
            if (logfile)
                fprintf(logfile, "WARNING: Error occurred in routine DIRsamplef..\n");
            goto cleanup;
        }
        // 如果 *ierror 为 -102，跳转到标签 L100
        if (*ierror == -102) goto L100;
    }
    // 设置 *numfunc 为 maxi + 1 + maxi
    *numfunc = maxi + 1 + maxi;
    // 设置 actmaxdeep 为 1
    actmaxdeep = 1;
    // 设置 oldpos 为 0
    oldpos = 0;
    // 设置 tstart 为 2
    tstart = 2;
/* +-----------------------------------------------------------------------+ */
/* | If no feasible point has been found, give out the iteration, the      | */
/* | number of function evaluations and a warning. Otherwise, give out     | */
/* | the iteration, the number of function evaluations done and minf.      | */
/* +-----------------------------------------------------------------------+ */
    // 如果 ifeasiblef 大于 0，则输出警告信息表示未找到可行点的迭代次数和函数评估次数
    if (ifeasiblef > 0) {
        if (logfile)
            fprintf(logfile, "No feasible point found in %d iterations "
                "and %d function evaluations.\n", tstart-1, *numfunc);
    } else {
        // 否则输出迭代次数、函数评估次数、*minf 和 fmax 的信息
        if (logfile)
            fprintf(logfile, "%d, %d, %g, %g\n",
                tstart-1, *numfunc, *minf, fmax);
    }
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | Main loop!                                                            | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
    // 主循环开始，循环从 tstart 到 *maxt - 1
    i__1 = *maxt;
    for (t = tstart; t <= i__1 -1; ++t) {
/* +-----------------------------------------------------------------------+ */
/* | Choose the sample points. The indices of the sample points are stored | */
/* | in the list S.                                                        | */
/* +-----------------------------------------------------------------------+ */
    // 设置 actdeep 为 actmaxdeep
    actdeep = actmaxdeep;
    // 调用 direct_dirchoose_ 函数选择样本点，更新 S 列表中的索引
    direct_dirchoose_(anchor, s, &MAXDEEP, f, minf, *eps, epsabs, levels, &maxpos, length,
        &MAXFUNC, &MAXDEEP, &MAXDIV, n, logfile, &cheat, &
        kmax, &ifeasiblef, jones);
/* +-----------------------------------------------------------------------+ */
/* | Add other hyperrectangles to S, which have the same level and the same| */
/* | function value at the center as the ones found above (that are stored | */
/* | in S). This is only done if we use the original DIRECT algorithm.     | */
/* | JG 07/16/01 Added Errorflag.                                          | */
/* +-----------------------------------------------------------------------+ */
    // 如果 algmethod 指针指向的值为 0，则执行以下代码块
    if (*algmethod == 0) {
         // 调用 direct_dirdoubleinsert_ 函数，传入参数 anchor, s, maxpos, point, f,
         // MAXDEEP, MAXFUNC, MAXDIV, ierror，并修改 ierror 的值
         direct_dirdoubleinsert_(anchor, s, &maxpos, point, f, &MAXDEEP, &MAXFUNC,
             &MAXDIV, ierror);
        // 如果 ierror 的值为 -6，则执行以下代码块
        if (*ierror == -6) {
        // 如果 logfile 非空指针，则将下一行的内容输出到 logfile
        if (logfile)
             fprintf(logfile,
"WARNING: Capacity of array S in DIRDoubleInsert reached. Increase maxdiv.\n"
"This means that there are a lot of hyperrectangles with the same function\n"
"value at the center. We suggest to use our modification instead (Jones = 1)\n"
              );
        *numiter = t;
        goto cleanup;
        }
    }
    oldpos = minpos;
/* +-----------------------------------------------------------------------+ */
/* | Initialise the number of sample points in this outer loop.            | */
/* +-----------------------------------------------------------------------+ */
    newtosample = 0;
    i__2 = maxpos;
    for (j = 1; j <= i__2; ++j) {
        actdeep = s[j + MAXDIV-1];
/* +-----------------------------------------------------------------------+ */
/* | If the actual index is a point to sample, do it.                      | */
/* +-----------------------------------------------------------------------+ */
        if (s[j - 1] > 0) {
/* +-----------------------------------------------------------------------+ */
/* | JG 09/24/00 Calculate the value delta used for sampling points.       | */
/* +-----------------------------------------------------------------------+ */
        actdeep_div__ = direct_dirgetmaxdeep_(&s[j - 1], length, &MAXFUNC,
            n);
        delta = thirds[actdeep_div__ + 1];
        actdeep = s[j + MAXDIV-1];
/* +-----------------------------------------------------------------------+ */
/* | If the current dept of division is only one under the maximal allowed | */
/* | dept, stop the computation.                                           | */
/* +-----------------------------------------------------------------------+ */
        if (actdeep + 1 >= mdeep) {
            // Emit a warning message if logging is enabled
            if (logfile)
             fprintf(logfile, "WARNING: Maximum number of levels reached. Increase maxdeep.\n");
            // Set error code and number of iterations
            *ierror = -6;
            *numiter = t;
            // Jump to cleanup section
            goto L100;
        }
        actmaxdeep = MAX(actdeep,actmaxdeep);
        help = s[j - 1];
        // Update linked list structure
        if (! (anchor[actdeep + 1] == help)) {
            pos1 = anchor[actdeep + 1];
            while(! (point[pos1 - 1] == help)) {
            pos1 = point[pos1 - 1];
            }
            point[pos1 - 1] = point[help - 1];
        } else {
            anchor[actdeep + 1] = point[help - 1];
        }
        if (actdeep < 0) {
            // Adjust actdeep if necessary
            actdeep = (integer) f[(help << 1) - 2];
        }
/* +-----------------------------------------------------------------------+ */
/* | Get the Directions in which to decrease the intervall-length.         | */
/* +-----------------------------------------------------------------------+ */
        // Determine directions for interval length reduction
        direct_dirget_i__(length, &help, arrayi, &maxi, n, &MAXFUNC);
/* +-----------------------------------------------------------------------+ */
/* | Sample the function. To do this, we first calculate the points where  | */
/* | we need to sample the function. After checking for errors, we then do | */
/* | the actual evaluation of the function, again followed by checking for | */
/* | errors.                                                               | */
/* +-----------------------------------------------------------------------+ */
        direct_dirsamplepoints_(c__, arrayi, &delta, &help, &start, length,
            logfile, f, &ifree, &maxi, point, &x[
            1], &l[1], minf, &minpos, &u[1], n, &MAXFUNC, &
            MAXDEEP, &oops);
        if (oops > 0) {
            if (logfile)
             fprintf(logfile, "WARNING: Error occurred in routine DIRsamplepoints.\n");
            *ierror = -4;
            *numiter = t;
            goto cleanup;
        }
        newtosample += maxi;
/* +-----------------------------------------------------------------------+ */
/* | JG 01/22/01 Added variable to keep track of the maximum value found.  | */
/* +-----------------------------------------------------------------------+ */
        direct_dirsamplef_(c__, arrayi, &delta, &help, &start, length,
            logfile, f, &ifree, &maxi, point, fcn, &x[
            1], x_seq, &l[1], minf, &minpos, &u[1], n, &MAXFUNC, &
            MAXDEEP, &oops, &fmax, &ifeasiblef, &iinfesiblef,
            args, force_stop);
        if (force_stop && *force_stop) {
             *ierror = -102;
             *numiter = t;
             goto L100;
        }
        if (oops > 0) {
            if (logfile)
             fprintf(logfile, "WARNING: Error occurred in routine DIRsamplef.\n");
            *ierror = -5;
            *numiter = t;
            goto cleanup;
        }
/* +-----------------------------------------------------------------------+ */
/* | Divide the intervalls.                                                | */
/* +-----------------------------------------------------------------------+ */
        direct_dirdivide_(&start, &actdeep_div__, length, point, arrayi, &
            help, list2, w, &maxi, f, &MAXFUNC, &MAXDEEP, n);
/* +-----------------------------------------------------------------------+ */
/* | Insert the new intervalls into the list (sorted).                     | */
/* +-----------------------------------------------------------------------+ */
        direct_dirinsertlist_(&start, anchor, point, f, &maxi, length, &
            MAXFUNC, &MAXDEEP, n, &help, jones);
/* +-----------------------------------------------------------------------+ */
/* | Increase the number of function evaluations.                          | */
/* +-----------------------------------------------------------------------+ */
        *numfunc = *numfunc + maxi + maxi;
        }
/* +-----------------------------------------------------------------------+ */
/* | End of main loop.                                                     | */
/* +-----------------------------------------------------------------------+ */
/* L20: */
    }
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | 如果找到了新的最小值，显示当前迭代次数、函数评估次数、当前最小的 f 值和位置 | */
/* +-----------------------------------------------------------------------+ */
    if (oldpos < minpos) {
        if (logfile)
         fprintf(logfile, "%d, %d, %g, %g\n",
             t, *numfunc, *minf, fmax);
    }
/* +-----------------------------------------------------------------------+ */
/* | 如果没有找到可行点，输出迭代次数、函数评估次数及警告信息                 | */
/* +-----------------------------------------------------------------------+ */
    if (ifeasiblef > 0) {
        if (logfile)
         fprintf(logfile, "No feasible point found in %d iterations "
             "and %d function evaluations\n", t, *numfunc);
    }
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* |                           终止条件检查                                 | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | JG 01/22/01 计算 minf 所在的超矩形的索引。计算该超矩形的体积并存储在 delta 中。| */
/* |             delta 可用于在体积低于原始体积的某个比率（由 volper 决定）时停止 DIRECT。| */
/* |             由于原始体积为 1（已缩放），一旦 delta 低于某个阈值（由 volper 给出），我们可以停止。| */
/* +-----------------------------------------------------------------------+ */
    *ierror = jones;
    jones = 0;
    actdeep_div__ = direct_dirgetlevel_(&minpos, length, &MAXFUNC, n, jones);
    jones = *ierror;
/* +-----------------------------------------------------------------------+ */
/* | JG 07/16/01 使用预计算的值计算体积。                                   | */
/* +-----------------------------------------------------------------------+ */
    delta = thirds[actdeep_div__];
    if (delta <= *volper) {
        *ierror = 4;
        if (logfile)
         fprintf(logfile, "DIRECT stopped: Volume of S_min is "
             "%g < %g of the original volume.\n",
             delta, *volper);
        *numiter = t;
        goto L100;
    }
/* +-----------------------------------------------------------------------+ */
/* | JG 01/23/01 计算 minf 所在的超矩形的测度。如果测度小于 sigmaper，则停止 DIRECT。| */
/* +-----------------------------------------------------------------------+ */
/* | Calculate the level of the hyperrectangle that contains the           | */
/* | function minimum found by DIRECT algorithm. If the level is less than | */
/* | or equal to the termination criterion sigmaper, set error code 5 and  | */
/* | print a message to the log file if available.                         | */
/* +-----------------------------------------------------------------------+ */
actdeep_div__ = direct_dirgetlevel_(&minpos, length, &MAXFUNC, n, jones);
delta = levels[actdeep_div__];
if (delta <= *sigmaper) {
    *ierror = 5;
    if (logfile)
        fprintf(logfile, "DIRECT stopped: Side length measure of S_min "
                        "= %g < %g.\n", delta, *sigmaper);
    *numiter = t;
    goto L100;
}
/* +-----------------------------------------------------------------------+ */
/* | If the best found function value is within fglper of the (known)      | */
/* | global minimum value, terminate the algorithm and set error code 3.   | */
/* | Print a message to the log file if available.                         | */
/* +-----------------------------------------------------------------------+ */
if ((*minf - *fglobal) / divfactor <= *fglper) {
    *ierror = 3;
    if (logfile)
        fprintf(logfile, "DIRECT stopped: found minimum within f_min_rtol of "
                        "global minimum.\n");
    *numiter = t;
    goto L100;
}
/* +-----------------------------------------------------------------------+ */
/* | Check for infeasible points near feasible ones. Replace function      | */
/* | values at the center of hyperrectangles if necessary.                 | */
/* | If iinfesiblef > 0, call direct_dirreplaceinf_ function.              | */
/* +-----------------------------------------------------------------------+ */
if (iinfesiblef > 0) {
    direct_dirreplaceinf_(&ifree, &ifreeold, f, c__, thirds, length, anchor,
                          point, &u[1], &l[1], &MAXFUNC, &MAXDEEP, n, n,
                          logfile, &fmax, jones);
}
ifreeold = ifree;
/* +-----------------------------------------------------------------------+ */
/* | Adjust the termination threshold eps based on the minimum function    | */
/* | value found, using a formula by Jones if iepschange is set to 1.      | */
/* +-----------------------------------------------------------------------+ */
if (iepschange == 1) {
    /* Computing MAX */
    d__1 = fabs(*minf) * 1e-4;
    *eps = MAX(d__1, epsfix);
}
/* +-----------------------------------------------------------------------+ */
/* | Adjust the maximum number of function evaluations allowed if no       | */
/* | feasible point has been found yet, potentially increasing the budget. | */
/* | Reset flags appropriately if a feasible point has been found.         | */
/* +-----------------------------------------------------------------------+ */
    // 如果increase等于1，则执行以下代码块
    if (increase == 1) {
        // 将maxf指向的值增加到numfunc指向的值再加上oldmaxf的结果
        *maxf = *numfunc + oldmaxf;
        // 如果ifeasiblef等于0，则执行以下代码块
        if (ifeasiblef == 0) {
            // 如果logfile不为空，则向其写入一条日志，指示DIRECT找到一个可行点，并调整预算为maxf的值
            if (logfile)
                 fprintf(logfile, "DIRECT found a feasible point.  The "
                     "adjusted budget is now set to %d.\n", *maxf);
            // 将increase的值设为0
            increase = 0;
        }
    }
/* +-----------------------------------------------------------------------+ */
/* | 检查已完成的函数评估次数是否超过分配的预算。如果是，则检查是否找到可行点。如  | */
/* | 果找到可行点，则终止。如果未找到可行点，则增加预算并设置增加标志。           | */
/* +-----------------------------------------------------------------------+ */
    if (*numfunc > *maxf) {
        // 如果未找到可行点，设置错误码为1
        if (ifeasiblef == 0) {
            *ierror = 1;
            // 如果日志文件存在，输出日志信息并记录迭代次数
            if (logfile)
                fprintf(logfile, "DIRECT stopped: numfunc >= maxf.\n");
            *numiter = t;
            // 跳转到标签 L100
            goto L100;
        } else {
            // 设置增加标志为1，表示继续寻找可行点
            increase = 1;
            // 如果日志文件存在，输出详细信息，并更新 maxf 的值
            if (logfile)
                fprintf(logfile,
                        "DIRECT could not find a feasible point after %d function evaluations.\n"
                        "DIRECT continues until a feasible point is found.\n", *numfunc);
            *maxf = *numfunc + oldmaxf;
        }
    }

    // 如果回调函数不为 Py_None
    if( callback != Py_None ) {
        // 创建参数元组并调用回调函数
        PyObject* arg_tuple = Py_BuildValue("(O)", x_seq);
        PyObject* callback_py = PyObject_CallObject(callback, arg_tuple);
        Py_DECREF(arg_tuple);
        // 如果回调函数调用失败，返回空指针
        if( !callback_py ) {
            return NULL;
        }
    }

L10:
    // 结束主循环

/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* | 主循环结束。                                                          | */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */
/* +-----------------------------------------------------------------------+ */

/* +-----------------------------------------------------------------------+ */
/* | 算法在 maxT 次迭代后停止。                                            | */
/* +-----------------------------------------------------------------------+ */
    *ierror = 2;
    // 如果日志文件存在，输出日志信息
    if (logfile)
        fprintf(logfile, "DIRECT stopped: maxT iterations.\n");

L100:
/* +-----------------------------------------------------------------------+ */
/* | 存储最小值的位置到 x 中。                                              | */
/* +-----------------------------------------------------------------------+ */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        // 计算并存储最小值的位置到 x 中
        x[i__] = c__[i__ + minpos * i__1 - i__1-1] * l[i__] + l[i__] * u[i__];
        // 更新 Python 列表 x_seq 中的值
        PyList_SetItem(x_seq, i__ - 1, PyFloat_FromDouble(x[i__]));
        // 恢复变量 u[i__] 和 l[i__] 的原始值
        u[i__] = oldu[i__ - 1];
        l[i__] = oldl[i__ - 1];
    }

/* +-----------------------------------------------------------------------+ */
/* | 存储函数评估次数到 maxf 中。                                           | */
/* +-----------------------------------------------------------------------+ */
    *maxf = *numfunc;
    *numiter = t;
/* +-----------------------------------------------------------------------+ */
/* | 输出运行的摘要。                                                      | */
/* +-----------------------------------------------------------------------+ */
/* | 调用 direct_dirsummary_ 函数，执行目录总结操作并记录到日志文件中。          | */
/* +-----------------------------------------------------------------------+ */
/* | 格式化语句。                                                             | */
/* +-----------------------------------------------------------------------+ */

cleanup:
    // 如果 c__ 不为 NULL，则释放 c__ 所指向的内存空间
    if (c__)
        free(c__);
    // 如果 f 不为 NULL，则释放 f 所指向的内存空间
    if (f)
        free(f);
    // 如果 s 不为 NULL，则释放 s 所指向的内存空间
    if (s)
        free(s);
    // 如果 w 不为 NULL，则释放 w 所指向的内存空间
    if (w)
        free(w);
    // 如果 oldl 不为 NULL，则释放 oldl 所指向的内存空间
    if (oldl)
        free(oldl);
    // 如果 oldu 不为 NULL，则释放 oldu 所指向的内存空间
    if (oldu)
        free(oldu);
    // 如果 list2 不为 NULL，则释放 list2 所指向的内存空间
    if (list2)
        free(list2);
    // 如果 point 不为 NULL，则释放 point 所指向的内存空间
    if (point)
        free(point);
    // 如果 anchor 不为 NULL，则释放 anchor 所指向的内存空间
    if (anchor)
        free(anchor);
    // 如果 length 不为 NULL，则释放 length 所指向的内存空间
    if (length)
        free(length);
    // 如果 arrayi 不为 NULL，则释放 arrayi 所指向的内存空间
    if (arrayi)
        free(arrayi);
    // 如果 levels 不为 NULL，则释放 levels 所指向的内存空间
    if (levels)
        free(levels);
    // 如果 thirds 不为 NULL，则释放 thirds 所指向的内存空间
    if (thirds)
        free(thirds);

    // 返回 ret 变量的值，并结束函数 direct_
    return ret;
} /* direct_ */
```